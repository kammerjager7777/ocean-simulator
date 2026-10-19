// The main sea as a spectral (FFT) sea on the graphics card (GPU physics, v28).
//
// One periodic tile of side L (scenario.size) sampled N x N (256, or 512 with ?fft=512):
// the physics grid is exactly that tile. Every lattice wave k = (nx, nz) 2 pi / L keeps its
// own phase (a texture, advanced by omega dt each step on the card), and gets its amplitude
// each step from the three bands, so changing the wind, its lag or gusts, the depth or the
// size only changes amplitudes and wavenumbers smoothly: nothing jumps.
//
// The band model is kept per wave. Each band (swell, wind sea, chop) has its own lagged
// wind, direction and spread, its own spectrum (Pierson-Moskowitz, or Elfouhaily for the
// chop, matched to PM at its lower edge) and its own break-test rule (the chop takes part in
// the break test at its PM level). A lattice wave takes energy from every band whose
// frequency range holds it: E_b = S_b(omega) cg D_b(theta) / k dk^2 (the variance one
// lattice wave carries, as the ripple patches do), times a soft window that rises and
// falls over +-3% of omega at the band's edges. Where two bands meet the windows sum to
// one, so a wave crossing from one band to the next as the lagged winds move never pops.
// The chop band stops hard at the grid's cut (2.4 cells), where the ripple patches take
// over exactly (band.heldOmegaEdge), as before.
//
// Waves longer than about a quarter of the tile cannot live on it (a 250 m tile holds no
// 500 m swell, and its first few lattice rings give only a handful of directions). Those
// stay explicit components (band.js, evaluated per point like v27) with energy share
// 1 - wFft(k), wFft = smoothstep(3 dk, 5 dk, k); the lattice takes wFft. They are not
// periodic, so they also hide the tile's repeat.
//
// Outputs (after the inverse FFT; "one-sided" fields hold the field in one part and the
// same waves a quarter period on in the other, which the renderer turns to hide tiling):
//   T0 = Zh (h = im), Zdx (shift x = re)         T1 = Zdz (re), ZsxN (re)
//   T2 = ZszN (re), (dh/dx, dh/dz)                T3 = (sqXX, sqZZ), (sqXZ, break height)
//   T4 = (break dh/dx, break dh/dz), unused
// sxN, szN are the shading slopes (the chop band's slope counted twice, as surface.js).
// sq.. are the squeeze terms of the Jacobian (sum a k cicj sin). Real pairs are packed two
// to a transform (Hermitian spectra), one-sided fields one to a transform.
import { shoal } from "./band.js?v=28";
import { timed } from "./passTimer.js?v=28";
import { elfouhailyParams, spectrumFor, wavenumber } from "./spectrum.js?v=28";

const TAU = Math.PI * 2;
export const BAND_EDGE = 0.03; // soft band edges, +-3% of omega
// The grid's cut, in cells per shortest wave (?cut= for review; v27's 192-point grid over
// 420 m stopped at 5.25 m = 3.2 of v28's 1.64 m cells).
// v28 (GAD foam): ?brkgrid=<n> limits the break-test waves to what an n^2 grid over the same
// tile holds (2.4 cells, as v27's explicit 192^2 grid did); 0 = the FFT's own cut.
export const BREAK_GRID = (() => { const v = typeof location !== "undefined" ? Number(new URLSearchParams(location.search).get("brkgrid")) : NaN; return Number.isFinite(v) && v >= 0 ? v : 0; })();
export const CUT_CELLS = (() => { const v = typeof location !== "undefined" ? Number(new URLSearchParams(location.search).get("cut")) : NaN; return Number.isFinite(v) && v >= 2 ? v : 2.4; })();
export const FFT_SPLIT = [3, 5]; // lattice rings (in dk) over which long waves hand to the FFT
const MAXB = 3;

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
export const SINCOS_GLSL = `
void sinCosTurns(float t, out float s, out float c) {
  float q = floor(t * 4.0 + 0.5);
  float x = (t - q * 0.25) * 6.283185307179586;
  float x2 = x * x;
  float sp = x * (1.0 + x2 * (-0.16666666666666666 + x2 * (0.008333333333333333 + x2 * (-0.0001984126984126984 + x2 * (2.7557319223985893e-6 - x2 * 2.505210838544172e-8)))));
  float cp = 1.0 + x2 * (-0.5 + x2 * (0.041666666666666664 + x2 * (-0.001388888888888889 + x2 * (2.48015873015873e-5 - x2 * 2.755731922398589e-7))));
  int k = int(q) & 3;
  if (k == 0) { s = sp; c = cp; }
  else if (k == 1) { s = cp; c = -sp; }
  else if (k == 2) { s = -sp; c = -cp; }
  else { s = -cp; c = sp; }
}`;
const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uN;
${SINCOS_GLSL}
`;
// Lattice: dispersion (finite depth), the bands' energies and the amplitude of one wave.
const MODE_GLSL = `
uniform float uDk;
uniform float uG;
uniform float uDepth;
uniform float uKCut;
uniform float uKBreak; // v28: the break test's own band limit (<= uKCut)
uniform vec2 uSplit;      // wFft rises from uSplit.x to uSplit.y (rad/m)
uniform float uEdge;
uniform int uBands;
uniform int uChop;        // index of the chop band (its slope counts twice in the normal), -1 none
uniform vec4 uB0[${MAXB}];   // omega peak, low edge, high edge, high edge hard (1)
uniform vec4 uB1[${MAXB}];   // direction, half spread, soft width, 1 / spread
uniform vec4 uB2[${MAXB}];   // model (0 PM, 1 Elfouhaily), break rule (0 none, 1 same, 2 PM), long scale, on
uniform vec4 uE0[${MAXB}];   // Elfouhaily kp, cp, alphaP, gamma
uniform vec4 uE1[${MAXB}];   // sigma, alphaM, Omega, km
uniform vec4 uE2[${MAXB}];   // cm
ivec2 latticeOf(ivec2 p) { return ivec2(p.x < uN / 2 ? p.x : p.x - uN, p.y < uN / 2 ? p.y : p.y - uN); }
float omegaOf(float k) { return sqrt(uG * k * tanh(min(k * uDepth, 20.0))); }
float pm(float w, float wp) {
  float ww = max(w, 0.05);
  float r = wp / ww;
  return 0.0081 * uG * uG / (ww * ww * ww * ww * ww) * exp(-1.25 * r * r * r * r);
}
float elf(float w, int b) {
  float ww = max(w, 0.05);
  float k = ww * ww / uG;
  vec4 e0 = uE0[b];
  vec4 e1 = uE1[b];
  float kp = e0.x, cp = e0.y, alphaP = e0.z, gam = e0.w;
  float sig = e1.x, alphaM = e1.y, Om = e1.z, km = e1.w;
  float cm = uE2[b].x;
  float c = sqrt((uG / k) * (1.0 + (k / km) * (k / km)));
  float r = sqrt(k / kp);
  float lpm = exp(-1.25 * (kp / k) * (kp / k));
  float jp = pow(gam, exp(-((r - 1.0) * (r - 1.0)) / (2.0 * sig * sig)));
  float bl = 0.5 * alphaP * (cp / c) * lpm * jp * exp((-Om / sqrt(10.0)) * (r - 1.0));
  float bh = 0.5 * alphaM * (cm / c) * lpm * jp * exp(-0.25 * (k / km - 1.0) * (k / km - 1.0));
  return ((uB2[b].z * bl + bh) / (k * k * k)) * ((2.0 * ww) / uG);
}
float gain(float w, float k) {
  float kh = k * uDepth;
  if (kh > 2.5) return 1.0;
  float sinh2 = sinh(min(2.0 * kh, 20.0));
  float group = 0.5 * (1.0 + (2.0 * kh) / max(sinh2, 1e-4));
  float cg = (w / max(k, 1e-4)) * group;
  float kDeep = (w * w) / uG;
  float cgDeep = w / (2.0 * max(kDeep, 1e-4));
  return min(sqrt(max(cgDeep, 1e-4) / max(cg, 1e-4)), 2.5);
}
// Amplitudes of the wave at lattice index p: all bands, the chop band alone, the break test.
void waveAt(ivec2 p, out float a, out float ac, out float ab, out vec2 khat, out float k) {
  ivec2 l = latticeOf(p);
  vec2 kv = vec2(l) * uDk;
  k = length(kv);
  a = 0.0; ac = 0.0; ab = 0.0; khat = vec2(0.0);
  if (k == 0.0 || k > uKCut) return;
  khat = kv / k;
  float t = tanh(min(k * uDepth, 20.0));
  float w = sqrt(uG * k * t);
  float cg = (uG * t + uG * k * uDepth * (1.0 - t * t)) / (2.0 * w);
  float wFft = smoothstep(uSplit.x, uSplit.y, k);
  if (wFft <= 0.0) return;
  float ang = atan(kv.y, kv.x);
  float e = 0.0, ec = 0.0, eb = 0.0;
  for (int b = 0; b < ${MAXB}; b += 1) {
    if (b >= uBands) break;
    vec4 b0 = uB0[b];
    vec4 b1 = uB1[b];
    vec4 b2 = uB2[b];
    if (b2.w < 0.5) continue;
    float lo = smoothstep(b0.y * (1.0 - uEdge), b0.y * (1.0 + uEdge), w);
    float hi = b0.w > 0.5 ? (w <= b0.z ? 1.0 : 0.0) : 1.0 - smoothstep(b0.z * (1.0 - uEdge), b0.z * (1.0 + uEdge), w);
    float win = lo * hi;
    if (win <= 0.0) continue;
    float off = ang - b1.x;
    off = atan(sin(off), cos(off));
    float dir = b1.w * (1.0 - smoothstep(b1.y - b1.z, b1.y + b1.z, abs(off)));
    if (dir <= 0.0) continue;
    float s = b2.x > 0.5 ? elf(w, b) : pm(w, b0.x);
    float shared = win * wFft * cg * dir / k * uDk * uDk;
    e += s * shared;
    if (b == uChop) ec += s * shared;
    if (b2.y > 1.5) eb += pm(w, b0.x) * shared;
    else if (b2.y > 0.5) eb += s * shared;
  }
  float g = gain(w, k);
  a = sqrt(2.0 * e) * g;
  ac = sqrt(2.0 * ec) * g;
  ab = k > uKBreak ? 0.0 : sqrt(2.0 * eb) * g;
}
`;
// Phases advance by omega dt (in turns, kept in [0, 1)).
const PHASE_FS = HEAD + MODE_GLSL + `
uniform sampler2D tPhase;
uniform float uDtTurns; // dt / 2 pi
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 l = latticeOf(p);
  float k = length(vec2(l) * uDk);
  float ph = texelFetch(tPhase, p, 0).x;
  float step = k > 0.0 ? omegaOf(k) * uDtTurns : 0.0;
  ph -= fract(step);
  o = vec4(ph - floor(ph), 0.0, 0.0, 1.0);
}`;
function spectrumFS(set) {
  return HEAD + MODE_GLSL + `
uniform sampler2D tPhase;
${set === 0 ? `layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;` : `layout(location = 0) out vec4 o4;`}
vec2 mulI(vec2 z) { return vec2(-z.y, z.x); }      // i z
vec2 conj2(vec2 z) { return vec2(z.x, -z.y); }
// Hermitian spectrum of a real field Re(sum q e^(ikx)), from q at k and at -k.
vec2 herm(vec2 qk, vec2 qm) { return 0.5 * (qk + conj2(qm)); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 m = ivec2((uN - p.x) % uN, (uN - p.y) % uN);
  float a, ac, ab, k, am, acm, abm, km;
  vec2 kh, khm;
  waveAt(p, a, ac, ab, kh, k);
  waveAt(m, am, acm, abm, khm, km);
  float s, c, sm, cm;
  sinCosTurns(texelFetch(tPhase, p, 0).x, s, c);
  sinCosTurns(texelFetch(tPhase, m, 0).x, sm, cm);
  vec2 e = vec2(c, s);    // e^(i phase) at k
  vec2 em = vec2(cm, sm); // at -k
  vec2 ck = a * e;
  vec2 cmk = am * em;
  vec2 bk = ab * e;
  vec2 bmk = abm * em;
  // Shading slope: the chop band's part again (sxN = dh/dx + chop dh/dx).
  vec2 nk = (a + ac) * e;
  vec2 nmk = (am + acm) * em;
${set === 0 ? `
  o0 = vec4(ck, kh.x * ck);
  o1 = vec4(kh.y * ck, k * kh.x * nk);
  // (dh/dx, dh/dz) packed: q = k khat c
  vec2 h1 = herm(k * kh.x * ck, km * khm.x * cmk);
  vec2 h2 = herm(k * kh.y * ck, km * khm.y * cmk);
  o2 = vec4(k * kh.y * nk, h1 + mulI(h2));
  // squeeze: field = Im(sum k ci cj c e^(ikx)) = Re(sum -i k ci cj c e^(ikx))
  vec2 xx = herm(-mulI(k * kh.x * kh.x * ck), -mulI(km * khm.x * khm.x * cmk));
  vec2 zz = herm(-mulI(k * kh.y * kh.y * ck), -mulI(km * khm.y * khm.y * cmk));
  vec2 xz = herm(-mulI(k * kh.x * kh.y * ck), -mulI(km * khm.x * khm.y * cmk));
  vec2 bh = herm(-mulI(bk), -mulI(bmk));
  o3 = vec4(xx + mulI(zz), xz + mulI(bh));` : `
  vec2 bx = herm(k * kh.x * bk, km * khm.x * bmk);
  vec2 bz = herm(k * kh.y * bk, km * khm.y * bmk);
  o4 = vec4(bx + mulI(bz), 0.0, 0.0);`}
}`;
}
// One Stockham radix-2 pass of the inverse transform (sign +1, unscaled) along x or z,
// on every complex pair of 'count' textures. Natural order in and out (verified in v27).
function fftFS(count) {
  const ins = Array.from({ length: count }, (_, i) => `uniform sampler2D tIn${i};\nlayout(location = ${i}) out vec4 o${i};`).join("\n");
  const body = Array.from({ length: count }, (_, i) => `  o${i} = bfly(texelFetch(tIn${i}, q0, 0), texelFetch(tIn${i}, q1, 0), wc, ws, r < uNs);`).join("\n");
  return HEAD + `
uniform int uNs;
uniform int uAxis;
${ins}
vec4 bfly(vec4 a, vec4 b, float wc, float ws, bool plus) {
  vec4 v = vec4(b.x * wc - b.y * ws, b.x * ws + b.y * wc, b.z * wc - b.w * ws, b.z * ws + b.w * wc);
  return plus ? a + v : a - v;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int at = uAxis == 0 ? p.x : p.y;
  int r = at % (2 * uNs);
  int jm = r % uNs;
  int j = (at / (2 * uNs)) * uNs + jm;
  ivec2 q0 = uAxis == 0 ? ivec2(j, p.y) : ivec2(p.x, j);
  ivec2 q1 = uAxis == 0 ? ivec2(j + uN / 2, p.y) : ivec2(p.x, j + uN / 2);
  float ws, wc;
  sinCosTurns(float(jm) / float(2 * uNs), ws, wc);
${body}
}`;
}
// Half-float copies for the renderer (filterable, mipmapped): R0 = Zh, Zdx; R1 = Zdz, ZsxN;
// R2 = ZszN, sqXX, sqZZ; R3 = sqXZ.
const COPY_FS = HEAD + `
uniform sampler2D t0; uniform sampler2D t1; uniform sampler2D t2; uniform sampler2D t3;
layout(location = 0) out vec4 r0;
layout(location = 1) out vec4 r1;
layout(location = 2) out vec4 r2;
layout(location = 3) out vec4 r3;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(t2, p, 0);
  vec4 b = texelFetch(t3, p, 0);
  r0 = texelFetch(t0, p, 0);
  r1 = texelFetch(t1, p, 0);
  r2 = vec4(a.xy, b.xy);
  r3 = vec4(b.z, 0.0, 0.0, 1.0);
}`;
// Shading-slope variance (sxN^2 + szN^2, real parts) of each mip level of the copies:
// rows summed in 16 chunks per level, then the chunks.
const STATS_A_FS = HEAD + `
uniform sampler2D tR1; uniform sampler2D tR2;
out vec4 o;
void main() {
  int chunk = int(gl_FragCoord.x);
  int L = int(gl_FragCoord.y);
  int size = max(uN >> L, 1);
  int rows = (size + 15) / 16;
  float total = 0.0;
  for (int i = 0; i < 64; i += 1) {
    if (i >= rows) break;
    int z = chunk * rows + i;
    if (z >= size) break;
    float row = 0.0;
    for (int x = 0; x < 1024; x += 1) {
      if (x >= size) break;
      float sx = texelFetch(tR1, ivec2(x, z), L).z;
      float sz = texelFetch(tR2, ivec2(x, z), L).x;
      row += sx * sx + sz * sz;
    }
    total += row;
  }
  o = vec4(total, 0.0, 0.0, 1.0);
}`;
const STATS_B_FS = HEAD + `
uniform sampler2D tA;
out vec4 o;
void main() {
  int L = int(gl_FragCoord.x);
  int size = max(uN >> L, 1);
  float total = 0.0;
  for (int c = 0; c < 16; c += 1) total += texelFetch(tA, ivec2(c, L), 0).x;
  o = vec4(total / float(size * size), 0.0, 0.0, 1.0);
}`;

function compile(gl, type, source, name) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`FFT sea: ${name} shader failed: ${gl.getShaderInfoLog(s)}`);
  return s;
}
function program(gl, fs, name) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS, name + " vertex"));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs, name));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`FFT sea: ${name} link failed: ${gl.getProgramInfoLog(p)}`);
  const u = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i += 1) {
    const info = gl.getActiveUniform(p, i);
    u[info.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, info.name);
  }
  return { program: p, u, name };
}

function hashTurns(nx, nz) {
  const x = Math.sin(nx * 127.1 + nz * 311.7 + 0.5) * 43758.5453;
  return x - Math.floor(x);
}

// CPU side, per band (never per wave): the soft window and the FFT share, as the shader.
function smooth(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
export function fftShare(k, dk) {
  return smooth(FFT_SPLIT[0] * dk, FFT_SPLIT[1] * dk, k);
}
export function bandWindow(w, lo, hi, hiHard) {
  const a = smooth(lo * (1 - BAND_EDGE), lo * (1 + BAND_EDGE), w);
  const b = hiHard ? (w <= hi ? 1 : 0) : 1 - smooth(hi * (1 - BAND_EDGE), hi * (1 + BAND_EDGE), w);
  return a * b;
}

// What each band hands the lattice this step (also used for the CPU integrals).
export function bandSetup(bands, winds, depth, gravity, L, N) {
  const dk = TAU / L;
  const spacing = L / N;
  const kCut = TAU / (CUT_CELLS * spacing);
  const h = Math.max(depth, 0.5);
  const omegaCut = Math.sqrt(gravity * kCut * Math.tanh(Math.min(kCut * h, 20)));
  return bands.map((band, i) => {
    const spec = band.spec;
    const w = winds[i];
    const spectrum = band.spectrum || spectrumFor(spec, w.speed, gravity);
    const wp = spectrum.omegaPeak;
    const lo = spec.omegaMin * wp;
    let hi = spec.omegaMax * wp;
    // The chop band ends hard where the ripple patches take over (the grid's cut).
    const hiHard = !!spec.patch;
    if (hiHard) hi = Math.min(hi, omegaCut);
    const p = spectrum.model === "elfouhaily" ? spectrum.params : elfouhailyParams(w.speed, gravity);
    return {
      id: spec.id, wp, lo, hi, hiHard, direction: w.direction, halfSpread: spec.spread / 2,
      soft: Math.min(0.05, spec.spread / 4), invSpread: 1 / spec.spread,
      model: spectrum.model === "elfouhaily" ? 1 : 0,
      breakRule: spec.breakTest === false ? 0 : spec.breakTest === "pm" && spectrum.model !== "pm" ? 2 : 1,
      longScale: spectrum.longScale || 1, on: w.speed >= 0.05 ? 1 : 0, p, spectrum, dk, kCut, omegaCut,
    };
  });
}
// Variance and slope variance the lattice carries per band (integrals over omega; the
// direction window integrates to one). Same windows as the shader.
export function latticeIntegrals(setup, depth, gravity, L, pmFor) {
  const out = setup.map(() => ({ variance: 0, mss: 0, chopMss: 0 }));
  const dk = TAU / L;
  setup.forEach((b, i) => {
    if (!b.on) return;
    const w0 = Math.max(b.lo * (1 - BAND_EDGE), 0.05);
    const w1 = b.hiHard ? b.hi : b.hi * (1 + BAND_EDGE);
    if (!(w1 > w0)) return;
    const n = 160;
    let v = 0, m = 0;
    for (let j = 0; j <= n; j += 1) {
      const w = w0 + ((w1 - w0) * j) / n;
      const k = wavenumber(w, depth, gravity);
      if (k > b.kCut) continue;
      const wt = (j === 0 || j === n ? 1 : j % 2 ? 4 : 2) * ((w1 - w0) / n) / 3;
      const s = b.spectrum.density(w) * bandWindow(w, b.lo, b.hi, b.hiHard) * fftShare(k, dk) * shoal(1, w, k, depth, gravity) ** 2;
      v += wt * s;
      m += wt * s * k * k;
    }
    out[i] = { variance: v, mss: m };
  });
  return out;
}

export function fftSeaProblem(gl) {
  if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 4) return "fewer than 4 draw buffers";
  if (gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) < 10) return "too few vertex texture units";
  return null;
}

export function createFftSea(gl, N) {
  if (!(N >= 16 && (N & (N - 1)) === 0 && N <= 1024)) throw new Error("FFT sea: N must be a power of two up to 1024");
  const levels = Math.round(Math.log2(N)) + 1;
  const progs = {
    phase: program(gl, PHASE_FS, "phase"),
    spec0: program(gl, spectrumFS(0), "spectrum"),
    spec1: program(gl, spectrumFS(1), "spectrum (break slope)"),
    fft4: program(gl, fftFS(4), "fft"),
    fft1: program(gl, fftFS(1), "fft"),
    copy: program(gl, COPY_FS, "copy"),
    statsA: program(gl, STATS_A_FS, "stats"),
    statsB: program(gl, STATS_B_FS, "stats"),
  };
  function tex(w, h, format, filter = gl.NEAREST, mips = 1, wrap = gl.CLAMP_TO_EDGE) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, mips, format, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips > 1 ? gl.LINEAR_MIPMAP_LINEAR : filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
    return t;
  }
  function target(textures) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    textures.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    gl.drawBuffers(textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("FFT sea: render target incomplete");
    return fb;
  }
  const phase = [tex(N, N, gl.R32F), tex(N, N, gl.R32F)];
  const init = new Float32Array(N * N);
  for (let row = 0; row < N; row += 1) for (let col = 0; col < N; col += 1) {
    init[row * N + col] = hashTurns(col < N / 2 ? col : col - N, row < N / 2 ? row : row - N);
  }
  gl.bindTexture(gl.TEXTURE_2D, phase[0]);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RED, gl.FLOAT, init);
  const phaseFb = [target([phase[0]]), target([phase[1]])];
  let phaseCur = 0;
  const T = [0, 1].map(() => Array.from({ length: 5 }, () => tex(N, N, gl.RGBA32F)));
  const fb4 = T.map((set) => target(set.slice(0, 4)));
  const fb1 = T.map((set) => target([set[4]]));
  const R = Array.from({ length: 4 }, () => tex(N, N, gl.RGBA16F, gl.LINEAR, levels, gl.REPEAT));
  const copyFb = target(R);
  const statsATex = tex(16, levels, gl.RGBA32F);
  const statsAFb = target([statsATex]);
  const statsTex = tex(16, 1, gl.RGBA32F);
  const statsFb = target([statsTex]);

  function run(p, fb, w, h, inputs, set) {
    gl.useProgram(p.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    if (p.u.uN) gl.uniform1i(p.u.uN, N);
    let unit = 9;
    for (const [name, t] of inputs) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(p.u[name], unit);
      unit += 1;
    }
    gl.activeTexture(gl.TEXTURE0);
    if (set) set(p.u);
    timed("sea:" + (p.name || "pass"), () => gl.drawArrays(gl.TRIANGLES, 0, 3));
  }
  const b0 = new Float32Array(MAXB * 4), b1 = new Float32Array(MAXB * 4), b2 = new Float32Array(MAXB * 4);
  const e0 = new Float32Array(MAXB * 4), e1 = new Float32Array(MAXB * 4), e2 = new Float32Array(MAXB * 4);
  function modeUniforms(u, setup, depth, gravity, L) {
    const dk = TAU / L;
    gl.uniform1f(u.uDk, dk);
    gl.uniform1f(u.uG, gravity);
    gl.uniform1f(u.uDepth, Math.max(depth, 0.5));
    const kc = setup.length ? setup[0].kCut : TAU / (CUT_CELLS * L / N);
    gl.uniform1f(u.uKCut, kc);
    if (u.uKBreak) gl.uniform1f(u.uKBreak, BREAK_GRID > 0 ? Math.min(kc, TAU / (CUT_CELLS * L / BREAK_GRID)) : kc);
    gl.uniform2f(u.uSplit, FFT_SPLIT[0] * dk, FFT_SPLIT[1] * dk);
    gl.uniform1f(u.uEdge, BAND_EDGE);
    gl.uniform1i(u.uBands, setup.length);
    gl.uniform1i(u.uChop, setup.findIndex((b) => b.id === "chop"));
    setup.forEach((b, i) => {
      b0.set([b.wp, b.lo, b.hi, b.hiHard ? 1 : 0], i * 4);
      b1.set([b.direction, b.halfSpread, b.soft, b.invSpread], i * 4);
      b2.set([b.model, b.breakRule, b.longScale, b.on], i * 4);
      e0.set([b.p.kp, b.p.cp, b.p.alphaP, b.p.gamma], i * 4);
      e1.set([b.p.sigma, b.p.alphaM, b.p.Om, b.p.km], i * 4);
      e2.set([b.p.cm, 0, 0, 0], i * 4);
    });
    gl.uniform4fv(u.uB0, b0); gl.uniform4fv(u.uB1, b1); gl.uniform4fv(u.uB2, b2);
    gl.uniform4fv(u.uE0, e0); gl.uniform4fv(u.uE1, e1); gl.uniform4fv(u.uE2, e2);
  }
  let final = 0;
  // GL state (vertex array bound, depth test off) is the caller's.
  function step(setup, depth, gravity, L, dt) {
    lastArgs = [setup, depth, gravity, L];
    run(progs.phase, phaseFb[1 - phaseCur], N, N, [["tPhase", phase[phaseCur]]], (u) => {
      modeUniforms(u, setup, depth, gravity, L);
      gl.uniform1f(u.uDtTurns, Math.max(dt, 0) / TAU);
    });
    phaseCur = 1 - phaseCur;
    const specSet = (u) => modeUniforms(u, setup, depth, gravity, L);
    run(progs.spec0, fb4[0], N, N, [["tPhase", phase[phaseCur]]], specSet);
    run(progs.spec1, fb1[0], N, N, [["tPhase", phase[phaseCur]]], specSet);
    let src = 0;
    for (let axis = 0; axis < 2; axis += 1) {
      for (let ns = 1; ns < N; ns *= 2) {
        const set = (u) => { gl.uniform1i(u.uNs, ns); gl.uniform1i(u.uAxis, axis); };
        run(progs.fft4, fb4[1 - src], N, N, [0, 1, 2, 3].map((i) => ["tIn" + i, T[src][i]]), set);
        run(progs.fft1, fb1[1 - src], N, N, [["tIn0", T[src][4]]], set);
        src = 1 - src;
      }
    }
    final = src;
    run(progs.copy, copyFb, N, N, [["t0", T[src][0]], ["t1", T[src][1]], ["t2", T[src][2]], ["t3", T[src][3]]]);
    timed("sea:mipmap", () => { for (const t of R) { gl.bindTexture(gl.TEXTURE_2D, t); gl.generateMipmap(gl.TEXTURE_2D); } });
    run(progs.statsA, statsAFb, 16, levels, [["tR1", R[1]], ["tR2", R[2]]]);
    run(progs.statsB, statsFb, levels, 1, [["tA", statsATex]]);
    return { T: T[final], R, statsTex, levels, N, L };
  }
  // Tests only.
  function read(which) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    const t = which === "phase" ? phase[phaseCur] : which === "stats" ? statsTex : T[final][which];
    const [w, h] = which === "stats" ? [16, 1] : [N, N];
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const data = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, data);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    return data;
  }
  // Tests only: the one-sided height spectrum (set 0, texture 0) for the current phases,
  // rebuilt into the scratch textures (the final fields are untouched).
  let lastArgs = null;
  function readSpectrum() {
    const [setup, depth, gravity, L] = lastArgs;
    const s = 1 - final;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    run(progs.spec0, fb4[s], N, N, [["tPhase", phase[phaseCur]]], (u) => modeUniforms(u, setup, depth, gravity, L));
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, T[s][0], 0);
    const data = new Float32Array(N * N * 4);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, data);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    gl.bindVertexArray(null);
    gl.deleteVertexArray(vao);
    return data;
  }
  return { step, read, readSpectrum, N, levels, phaseInit: init };
}

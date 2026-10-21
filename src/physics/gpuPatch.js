// The chop band's ripple patches on the graphics card (GPU physics only; ?physics=cpu and
// the selfcheck use chopPatch.js). Same patch as chopPatch.js: the same lattice, spectrum
// (the band's own, at its lagged wind), direction spread, dispersion, shoaling, phases and
// cut-offs; the same periodic tile and quarter-period field the renderer rotates over the
// sea against tiling.
//
// The CPU keeps what is cheap and must be exact, in double precision as chopPatch.js does:
// each wave's phase (one multiply-add per wave per step), its amplitude (the spectrum is
// evaluated once per distinct wavenumber, about 1,840 of the 16,384 lattice waves) and the
// panel numbers. The direction mask is recomputed only when the wind direction changes.
// rmsSlope comes from the amplitudes: mean (dh/dx)^2 + (dh/dz)^2 of the real field is
// sum (k a)^2 / 2, exactly, because no lattice wave has its opposite in the patch (the
// spread is under pi). The card does the rest each step: sin and cos, the two inverse FFTs
// for the slopes (Stockham radix 2, 2 x log2 N passes), the 8-bit slope texture the
// renderer samples (quantised as view.js did, then mipmapped by GL), and the slope
// variance each mip level keeps (box-averaged levels, as view.js's keptPerLevel), which the
// renderer reads from a small texture. Nothing comes back to the CPU.
import { shoal } from "./band.js?v=29.1.6";
import { timed } from "./passTimer.js?v=29.1.6";

const TAU = Math.PI * 2;
// Display range of the 8-bit slope texture (view.js: CHOP_SCALE / FINE_SCALE).
export const PATCH_RANGE = 0.6;

function unitPhase(n) {
  const x = Math.sin(n * 91.7 + 3.1) * 43758.5453;
  return (x - Math.floor(x)) * Math.PI * 2;
}

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uN;
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
}
`;
// Lattice spectrum of the two slope fields: i k a e^(i phase), as (x re, x im, z re, z im).
const SPECTRUM_FS = HEAD + `
uniform sampler2D tModes; // amplitude, phase in turns
uniform float uDk;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 m = texelFetch(tModes, p, 0).xy;
  float t = m.y - floor(m.y + 0.5);
  float s, c;
  sinCosTurns(t, s, c);
  float kx = float(p.x < uN / 2 ? p.x : p.x - uN) * uDk;
  float kz = float(p.y < uN / 2 ? p.y : p.y - uN) * uDk;
  float cc = m.x * c;
  float ss = m.x * s;
  o = vec4(-kx * ss, kx * cc, -kz * ss, kz * cc);
}`;
// One Stockham radix-2 pass of the inverse transform (sign +1, unscaled) along x or z,
// on both complex fields at once. Natural order in and out.
const FFT_FS = HEAD + `
uniform sampler2D tIn;
uniform int uNs;
uniform int uAxis;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int at = uAxis == 0 ? p.x : p.y;
  int r = at % (2 * uNs);
  int jm = r % uNs;
  int j = (at / (2 * uNs)) * uNs + jm;
  ivec2 q0 = uAxis == 0 ? ivec2(j, p.y) : ivec2(p.x, j);
  ivec2 q1 = uAxis == 0 ? ivec2(j + uN / 2, p.y) : ivec2(p.x, j + uN / 2);
  vec4 a = texelFetch(tIn, q0, 0);
  vec4 b = texelFetch(tIn, q1, 0);
  float ws, wc;
  sinCosTurns(float(jm) / float(2 * uNs), ws, wc);
  vec4 v = vec4(b.x * wc - b.y * ws, b.x * ws + b.y * wc, b.z * wc - b.w * ws, b.z * ws + b.w * wc);
  o = r < uNs ? a + v : a - v;
}`;
// The renderer's 8-bit slope texture: round(slope * 127 / range) + 128, kept in 1..255.
const QUANT_FS = HEAD + `
uniform sampler2D tIn;
uniform float uScale;
out vec4 o;
void main() {
  vec4 s = texelFetch(tIn, ivec2(gl_FragCoord.xy), 0);
  o = clamp(floor(s * uScale + 0.5) + 128.0, 1.0, 255.0) / 255.0;
}`;
// Next mip level: the 2 x 2 box average (what generateMipmap does to the texture).
const MIP_FS = HEAD + `
uniform sampler2D tIn;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  o = 0.25 * (texelFetch(tIn, p, 0) + texelFetch(tIn, p + ivec2(1, 0), 0) + texelFetch(tIn, p + ivec2(0, 1), 0) + texelFetch(tIn, p + ivec2(1, 1), 0));
}`;
// Slope variance of each level (texel L of an 8 x 1 target), row sums first.
function statsFS(levels) {
  return HEAD + `
uniform sampler2D tLevel[${levels}];
out vec4 o;
float level(int L, int size) {
  float total = 0.0;
  for (int z = 0; z < 1024; z += 1) {
    if (z >= size) break;
    float row = 0.0;
    for (int x = 0; x < 1024; x += 1) {
      if (x >= size) break;
      vec4 v = vec4(0.0);
      ${Array.from({ length: levels }, (_, L) => `if (L == ${L}) v = texelFetch(tLevel[${L}], ivec2(x, z), 0);`).join("\n      ")}
      row += dot(v, v);
    }
    total += row;
  }
  return total / float(size * size);
}
void main() {
  int L = int(gl_FragCoord.x);
  o = vec4(L < ${levels} ? level(L, uN >> L) : 0.0, 0.0, 0.0, 1.0);
}`;
}

// v27.2: the slope variance of each level in two passes (was one texel summing a whole
// level serially: 16,384 dependent fetches in one GPU thread, every step, per patch).
// Pass 1: an 8 x 8 grid of threads; thread t sums its own 1/64 tile of every level
// (levels smaller than 8 texels: thread (x, z) takes texel (x, z) if it exists), levels
// 0-3 to output 0 and 4-7 to output 1. Pass 2 (gather): texel L sums the 64 partials.
function partialFS(levels) {
  return HEAD + `
uniform sampler2D tLevel[${levels}];
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
float part(int L, ivec2 t) {
  int size = uN >> L;
  int tile = max(size / 8, 1);
  if (size < 8 && (t.x >= size || t.y >= size)) return 0.0;
  ivec2 base = size < 8 ? t : t * tile;
  float total = 0.0;
  for (int z = 0; z < 16; z += 1) {
    if (z >= tile) break;
    float row = 0.0;
    for (int x = 0; x < 16; x += 1) {
      if (x >= tile) break;
      vec4 v = vec4(0.0);
      ${Array.from({ length: levels }, (_, L) => `if (L == ${L}) v = texelFetch(tLevel[${L}], base + ivec2(x, z), 0);`).join("\n      ")}
      row += dot(v, v);
    }
    total += row;
  }
  return total;
}
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  float p[8];
  for (int L = 0; L < 8; L += 1) p[L] = L < ${levels} ? part(L, t) : 0.0;
  o0 = vec4(p[0], p[1], p[2], p[3]);
  o1 = vec4(p[4], p[5], p[6], p[7]);
}`;
}
function gatherFS() {
  return HEAD + `
uniform sampler2D tP0;
uniform sampler2D tP1;
out vec4 o;
void main() {
  int L = int(gl_FragCoord.x);
  float total = 0.0;
  for (int z = 0; z < 8; z += 1) {
    float row = 0.0;
    for (int x = 0; x < 8; x += 1) {
      vec4 a = L < 4 ? texelFetch(tP0, ivec2(x, z), 0) : texelFetch(tP1, ivec2(x, z), 0);
      int c = L - (L < 4 ? 0 : 4);
      row += c == 0 ? a.x : c == 1 ? a.y : c == 2 ? a.z : a.w;
    }
    total += row;
  }
  int size = uN >> L;
  o = vec4(L < 8 && size > 0 ? total / float(size * size) : 0.0, 0.0, 0.0, 1.0);
}`;
}

function compile(gl, type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("GPU patch shader failed: " + gl.getShaderInfoLog(s));
  return s;
}
function program(gl, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("GPU patch link failed: " + gl.getProgramInfoLog(p));
  const u = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i += 1) {
    const info = gl.getActiveUniform(p, i);
    u[info.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, info.name);
  }
  return { program: p, u };
}
function floatTex(gl, w, h, format = gl.RGBA32F) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, format, w, h);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
function fbo(gl, tex) {
  const f = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("GPU patch render target incomplete");
  return f;
}

const shared = new WeakMap();
function programsFor(gl, levels) {
  let s = shared.get(gl);
  if (!s) { s = {}; shared.set(gl, s); }
  if (!s.base) s.base = { spectrum: program(gl, SPECTRUM_FS), fft: program(gl, FFT_FS), quant: program(gl, QUANT_FS), mip: program(gl, MIP_FS) };
  if (!s["stats" + levels]) s["stats" + levels] = program(gl, gatherFS());
  if (!s["part" + levels]) { s["part" + levels] = program(gl, partialFS(levels)); s["part" + levels].name = "partial"; }
  for (const [k, v] of Object.entries(s.base)) v.name = k;
  s["stats" + levels].name = "stats";
  return { ...s.base, stats: s["stats" + levels], partial: s["part" + levels] };
}

export function createGpuChopPatch(gl, bandSpec, patchSpec = bandSpec.patch) {
  const patch = patchSpec;
  const tag = patchSpec === bandSpec.patch ? "chop" : "fine";
  const N = patch.resolution;
  const L = patch.size;
  const spacing = L / N;
  const dk = (2 * Math.PI) / L;
  const kCut = (2 * Math.PI) / (2.4 * spacing);
  const count = N * N;
  const levels = Math.round(Math.log2(N)) + 1;
  if (1 << (levels - 1) !== N || levels > 8) throw new Error("GPU patch: resolution must be a power of two up to 128");
  const kx = new Float64Array(count);
  const kz = new Float64Array(count);
  const kk = new Float64Array(count);
  const angle = new Float64Array(count);
  const phase = new Float64Array(count);
  // Distinct wavenumbers: each wave points at its |k| slot.
  const slotOf = new Int32Array(count);
  const slots = new Map();
  const slotK = [];
  for (let row = 0; row < N; row += 1) {
    const nz = row < N / 2 ? row : row - N;
    for (let col = 0; col < N; col += 1) {
      const nx = col < N / 2 ? col : col - N;
      const i = row * N + col;
      kx[i] = nx * dk;
      kz[i] = nz * dk;
      kk[i] = Math.hypot(kx[i], kz[i]);
      angle[i] = Math.atan2(kz[i], kx[i]);
      phase[i] = unitPhase(i);
      const key = kk[i];
      if (!slots.has(key)) { slots.set(key, slotK.length); slotK.push(key); }
      slotOf[i] = slots.get(key);
    }
  }
  const S = slotK.length;
  const slotW = new Float64Array(S);
  const slotAmp = new Float64Array(S);
  const slotOk = new Uint8Array(S);
  const inSpread = new Uint8Array(count);
  let maskDirection = NaN;
  let maskSpread = NaN;
  const modesData = new Float32Array(count * 2);

  const progs = programsFor(gl, levels);
  const modeTex = floatTex(gl, N, N, gl.RG32F);
  const work = [floatTex(gl, N, N), floatTex(gl, N, N)];
  const workFb = work.map((t) => fbo(gl, t));
  const mipTex = [];
  const mipFb = [];
  for (let l = 1; l < levels; l += 1) { mipTex.push(floatTex(gl, N >> l, N >> l)); mipFb.push(fbo(gl, mipTex[l - 1])); }
  const statsTex = floatTex(gl, 8, 1);
  // the 8 x 8 partial sums (two RGBA32F targets, one framebuffer)
  const partTex = [floatTex(gl, 8, 8), floatTex(gl, 8, 8)];
  const partFb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, partFb);
  partTex.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const statsFb = fbo(gl, statsTex);
  // The renderer's slope texture: repeating, mipmapped, anisotropic, as view.js sets it up.
  const slopeTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, slopeTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, N, N, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(16, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) || 1));
  gl.generateMipmap(gl.TEXTURE_2D);
  const slopeFb = fbo(gl, slopeTex);
  let version = 0;
  let finalWork = 0;

  function run(p, fb, w, h, inputs, set) {
    gl.useProgram(p.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    if (p.u.uN) gl.uniform1i(p.u.uN, N);
    let unit = 8;
    for (const [name, t] of inputs) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      if (Array.isArray(name)) gl.uniform1i(name[0], unit);
      else gl.uniform1i(p.u[name], unit);
      unit += 1;
    }
    gl.activeTexture(gl.TEXTURE0);
    if (set) set(p.u);
    timed(tag + ":" + (p.name || "pass"), () => gl.drawArrays(gl.TRIANGLES, 0, 3));
  }

  // GL state (vertex array, depth test, framebuffer) is the caller's (gpuSurface.js).
  function step(band, windSample, depth, gravity, dt, kLow = 0) {
    const h = Math.max(depth, 0.5);
    const spectrum = band.spectrum;
    const omegaLow = band.heldOmegaEdge;
    const calm = windSample.speed < 0.05;
    const halfSpread = bandSpec.spread / 2;
    const dirDensity = 1 / bandSpec.spread;
    if (windSample.direction !== maskDirection || halfSpread !== maskSpread) {
      for (let i = 0; i < count; i += 1) {
        let off = angle[i] - windSample.direction;
        off = Math.atan2(Math.sin(off), Math.cos(off));
        inSpread[i] = Math.abs(off) > halfSpread ? 0 : 1;
      }
      maskDirection = windSample.direction;
      maskSpread = halfSpread;
    }
    for (let j = 0; j < S; j += 1) {
      const k = slotK[j];
      slotOk[j] = 0;
      slotAmp[j] = 0;
      if (k === 0) continue;
      const t = Math.tanh(k * h);
      const w = Math.sqrt(gravity * k * t);
      slotW[j] = w;
      if (calm || k > kCut || w <= omegaLow || k <= kLow) continue;
      const cg = (gravity * t + gravity * k * h * (1 - t * t)) / (2 * w);
      const e = spectrum.density(w) * cg * dirDensity / k * dk * dk;
      slotAmp[j] = shoal(Math.sqrt(2 * e), w, k, depth, gravity) * (band.ampScale === undefined ? 1 : band.ampScale);
      slotOk[j] = 1;
    }
    let modes = 0;
    let variance = 0;
    let slopeSum = 0;
    let kMin = Infinity;
    let kMax = 0;
    let omegaMin = Infinity;
    const stepDt = Math.max(dt, 0);
    for (let i = 0; i < count; i += 1) {
      const k = kk[i];
      let a = 0;
      if (k !== 0) {
        const j = slotOf[i];
        const w = slotW[j];
        phase[i] -= w * stepDt;
        if (slotOk[j] && inSpread[i]) {
          a = slotAmp[j];
          modes += 1;
          variance += (a * a) / 2;
          slopeSum += (kx[i] * kx[i] + kz[i] * kz[i]) * a * a;
          kMin = Math.min(kMin, k);
          kMax = Math.max(kMax, k);
          omegaMin = Math.min(omegaMin, w);
        }
      }
      let turns = phase[i] / TAU;
      turns -= Math.floor(turns);
      modesData[i * 2] = a;
      modesData[i * 2 + 1] = turns;
    }
    gl.bindTexture(gl.TEXTURE_2D, modeTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RG, gl.FLOAT, modesData);

    run(progs.spectrum, workFb[0], N, N, [["tModes", modeTex]], (u) => gl.uniform1f(u.uDk, dk));
    let src = 0;
    for (let axis = 0; axis < 2; axis += 1) {
      for (let ns = 1; ns < N; ns *= 2) {
        run(progs.fft, workFb[1 - src], N, N, [["tIn", work[src]]], (u) => { gl.uniform1i(u.uNs, ns); gl.uniform1i(u.uAxis, axis); });
        src = 1 - src;
      }
    }
    finalWork = src;
    run(progs.quant, slopeFb, N, N, [["tIn", work[src]]], (u) => gl.uniform1f(u.uScale, 127 / PATCH_RANGE));
    gl.bindTexture(gl.TEXTURE_2D, slopeTex);
    gl.generateMipmap(gl.TEXTURE_2D);
    let prev = work[src];
    for (let l = 1; l < levels; l += 1) {
      run(progs.mip, mipFb[l - 1], N >> l, N >> l, [["tIn", prev]]);
      prev = mipTex[l - 1];
    }
    const levelInputs = [work[src], ...mipTex].map((t, l) => [[gl.getUniformLocation(progs.partial.program, `tLevel[${l}]`)], t]);
    run(progs.partial, partFb, 8, 8, levelInputs);
    run(progs.stats, statsFb, 8, 1, [["tP0", partTex[0]], ["tP1", partTex[1]]]);
    version += 1;
    return {
      size: L,
      n: N,
      spacing,
      version,
      modes,
      variance,
      significantHeight: 4 * Math.sqrt(variance),
      rmsSlope: Math.sqrt(slopeSum / 2),
      kMin: modes ? kMin : 0,
      kMax,
      kCut,
      kLow,
      omegaLow,
      omegaMin: modes ? omegaMin : 0,
      longest: modes ? (2 * Math.PI) / kMin : 0,
      shortest: modes ? (2 * Math.PI) / kMax : 0,
      gpu: { slopeTex, statsTex, levels },
    };
  }
  // Tests only: the float slopes, the 8-bit texture and the per-level variances.
  function read() {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    const get = (tex, w, h, type) => {
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const data = type === "byte" ? new Uint8Array(w * h * 4) : new Float32Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, type === "byte" ? gl.UNSIGNED_BYTE : gl.FLOAT, data);
      return data;
    };
    const slopes = get(work[finalWork], N, N);
    const bytes = get(slopeTex, N, N, "byte");
    const levelVar = get(statsTex, 8, 1).filter((_, i) => i % 4 === 0);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    return { slopes, bytes, levelVar };
  }
  return { step, read };
}

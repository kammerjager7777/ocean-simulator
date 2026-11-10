// GPU physics (WebGL2), v28: the main sea is a spectral (FFT) sea on the card (fftSea.js);
// the break test, foam and ages below run on its grid exactly as in v27.
// (v27's header follows; pass 1 now reads the FFT fields and adds only the explicit long
// waves the tile cannot hold.)
//
// What stays on the CPU (cheap, per wave component, never per point): the wind and its
// per-band lag (wind.js), each band's retune (band.js: spectrum, amplitudes, the PM-level
// chop break rule, phases), the ripple patches' per-wave phases and amplitudes (gpuPatch.js;
// their FFTs run on the card) and the slope budget (roughness.js). The 46 components go to the card as uniforms; everything per grid point
// runs here as fragment-shader passes on float render targets:
//   1. wave   - the 46-wave sum: height, displacement, normal, Jacobian, slopes, break field
//   2. local  - break test, crest glow (lips), thin-crest transmission, travel-axis crests
//   3. join   - for each crest break: its partner along the crest and the gap's crest cells
//   4. paint  - foam painted on breaking fronts, as an exact gather of surface.js's scatter
//               (highest grade wins; ties go to whichever painter surface.js reaches first)
//   5. age    - lingering foam: ping-pong state (age, kept grade, kept ahead, kept run)
//   6. reduce - the panel counts, 16x16 blocks, read back asynchronously (a few KB)
// The renderer reads the textures directly; nothing per point comes back to the CPU.
// readFields() reads the whole field back, for parity tests only.
import { GRAVITY } from "./gravity.js";
import { createBand, retuneBand, bandVariance } from "./band.js?v=30.22";
import { createGpuChopPatch } from "./gpuPatch.js?v=30.22";
import { initPassTimer, timed, passTimes, timeEmpty } from "./passTimer.js?v=30.22";
import { slopeBudget } from "./roughness.js?v=30.22";
import { SEA_RATE, createFftSea, fftSeaProblem, bandSetup, latticeIntegrals, fftShare } from "./fftSea.js?v=30.22";
import { travelDirection, NO_FOAM_AHEAD, TRANSLUCENCY, crestReachAt, breakRuleAt, foamWindowAt, faceStepsAt, FACE_MAX } from "./surface.js?v=32.1";
import { foamGain } from "./steepness.js";

let simTimeCover = 0; // v30.6: clock for the cover pass eddy field
const TAU = Math.PI * 2;
// Ages of samples that have never carried foam (surface.js uses Infinity).
const NEVER = 1e30;
const BLOCK = 16;
// v30c foam coverage constants (URL overrides for tuning: ?ftd= ?ftr= ?ffeed= ?fsrc=)
const FQ = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const EN_CREF = Number(FQ.get("ecr") || 6.0).toFixed(3); // v30.19 reference crest speed (m/s) for the break-energy scale
const EN_K = Number(FQ.get("ek") || 4.0).toFixed(3);    // v30.19 coverage calibration (Monahan & O'Muircheartaigh 1980)
const FOAM_TAU_DENSE = Number(FQ.get("ftd") || 2.2); // whitewater fades over 1-2 s
const FOAM_TAU_RESID_K = Number(FQ.get("ftr") || 1.0); // residual film: 2 s at 5 m/s rising to 6 s at 25 m/s (x this)
const FOAM_FEED = Number(FQ.get("ffeed") || 0.5);
const FOAM_SRC = Number(FQ.get("fsrc") || 12.0);
const FOAM_VIS = Number(FQ.get("foamvis") || 0.3); // v30.20: FOAM SEA counts cells at or above this foam strength
const FOAM_SL = Number(FQ.get("fsl") ?? (FQ.get("fcov") === "0" ? 1 : 0.11)); // v30.22h: fit to top-down pale cover at 10/15/25 m/s // v30.22f FOAM SEA: weight of lace-only cells
const FOAM_JTH = Number(FQ.get("fjth") || 0.75); // Jacobian below this = surface folding at the tip
// v30.22f whitecap coverage vs wind (Monahan-style: ~1% at 10 m/s, 3-5% at 15, 8-15% at 20-25). ?fcov=0 restores
// v30.22e. The dense white phase of released foam (the flat pale patches) is shortened, most at low wind
// (?fcd0= at 10 m/s .. ?fcd1= at 25), and the strand spawn rate follows the wind (?fcr0= .. ?fcr1=, log ramp).
const FCOV = FQ.get("fcov") !== "0";
const FC = { d0: Number(FQ.get("fcd0") ?? 2.0), d1: Number(FQ.get("fcd1") ?? 2.5), r0: Number(FQ.get("fcr0") ?? 1.0), r1: Number(FQ.get("fcr1") ?? 1.5) };
const fcS = (U) => { const t = Math.min(1, Math.max(0, (U - 10) / 15)); return t * t * (3 - 2 * t); };
// splat gain of the dense phase (< 1: the whitecap patch is holed by the lace instead of a flat pale fill)
const FC_WG = Number(FQ.get("fwg") ?? (FCOV ? 0.6 : 1));
// big oceans (> 250 m) carry bigger, more energetic breakers (longer-lived, wider splats): their dense phase is
// shortened by (250 / size)^?fcsz= so the coverage stays roughly size-invariant (25 m/s at 1000 m read 45%)
const FC_SZ = Number(FQ.get("fcsz") ?? 1.2);
export function foamDenseK(U, L = 250) { return FCOV ? (FC.d0 + (FC.d1 - FC.d0) * fcS(U)) * Math.pow(Math.min(1, 250 / Math.max(L, 1)), FC_SZ) : 1; }
export function foamRateK(U) { return FCOV ? FC.r0 * Math.pow(FC.r1 / FC.r0, fcS(U)) : 1; }

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uN;
bool inside(ivec2 q) { return q.x >= 0 && q.y >= 0 && q.x < uN && q.y < uN; }
`;

const QUAD_VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Grid offsets surface.js takes with Math.round, computed on the CPU in double precision
// so the card steps exactly the same cells: along travel (s = -1..3), along the crest
// (j = -1..5) and the join's fractional travel steps round(t * w j / s).
const OFFSET_GLSL = `
uniform ivec2 uOffT[${FACE_MAX + 2}];   // along travel, s = -1..FACE_MAX at [s + 1]
uniform ivec2 uOffC[7];   // along crest, j = -1..5 at [j + 1]
uniform ivec2 uOffF[30];  // join gap, [pair(s, j) * 3 + w + 1]
uniform vec2 uTravel;
ivec2 offT(int s) { return uOffT[s + 1]; }
ivec2 offC(int j) { return uOffC[j + 1]; }
int pairIndex(int s, int j) { return (s - 2) * (s - 1) / 2 + (j - 1); }
`;

function waveFS(maxC) {
  return HEADER + `
#define MAXC ${maxC}
// sin and cos of a phase in turns (|t| <= 0.5), to float32 accuracy on every card: a
// quadrant reduction (exact in turns), then Taylor series on [-pi/4, pi/4]. Built-in sin
// and cos vary between GPUs (software GL's lose about 1e-4), which a 9 m sea turns into
// millimetres.
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
uniform int uCount;
uniform vec4 uW0[MAXC];  // amplitude, break amplitude, k, 1 for the chop band
uniform vec4 uW1[MAXC];  // cos dir, sin dir, turns per cell in x and z (coarse part)
uniform vec4 uW2[MAXC];  // turns per cell in x and z (fine part), phase at cell (0,0) in turns
uniform float uChop;
uniform sampler2D tT0;
uniform sampler2D tT1;
uniform sampler2D tT2;
uniform sampler2D tT3;
uniform sampler2D tT4;
uniform float uBreakSlope;
uniform float uDepthLimit;
layout(location = 0) out vec4 oA; // height, shift x, shift z, Jacobian
layout(location = 1) out vec4 oB; // normal xyz, steep
layout(location = 2) out vec4 oC; // dh/dx, dh/dz, break height, break slope
layout(location = 3) out vec4 oD; // v28 long-slope rule: explicit long-wave height, its dh/dx, dh/dz, unused
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float fx = float(p.x);
  float fz = float(p.y);
  // The FFT sea at this grid point, then the explicit long waves on top.
  vec4 f0 = texelFetch(tT0, p, 0);
  vec4 f1 = texelFetch(tT1, p, 0);
  vec4 f2 = texelFetch(tT2, p, 0);
  vec4 f3 = texelFetch(tT3, p, 0);
  vec4 f4 = texelFetch(tT4, p, 0);
  float height = f0.y, dhdx = f2.z, dhdz = f2.w, shiftX = f0.z, shiftZ = f1.x;
  float squeezeX = f3.x, squeezeZ = f3.y, squeezeXZ = f3.z, chopDx = f1.z - f2.z, chopDz = f2.x - f2.w;
  float bh = f3.w, bdx = f4.x, bdz = f4.y;
  float xh = 0.0, xdx = 0.0, xdz = 0.0;
  for (int i = 0; i < MAXC; i += 1) {
    if (i >= uCount) break;
    vec4 a = uW0[i];
    vec4 b = uW1[i];
    vec4 c = uW2[i];
    // Phase in turns, range-reduced so float32 keeps it to about 1e-7 turns: the coarse
    // part of the per-cell step has few enough bits that cell x step is exact.
    float t = fract(fx * b.z) + fract(fz * b.w) + c.z + (fx * c.x + fz * c.y);
    t -= floor(t + 0.5);
    float s;
    float co;
    sinCosTurns(t, s, co);
    float amplitude = a.x;
    float k = a.z;
    float cx = b.x;
    float cz = b.y;
    float slopeScale = amplitude * k;
    height += amplitude * s;
    float gx = slopeScale * cx * co;
    float gz = slopeScale * cz * co;
    xh += amplitude * s; xdx += gx; xdz += gz;
    dhdx += gx;
    dhdz += gz;
    shiftX += amplitude * cx * co;
    shiftZ += amplitude * cz * co;
    squeezeX += slopeScale * cx * cx * s;
    squeezeZ += slopeScale * cz * cz * s;
    squeezeXZ += slopeScale * cx * cz * s;
    if (a.w > 0.5) { chopDx += gx; chopDz += gz; }
    float ba = a.y;
    bh += ba * s;
    bdx += ba * k * cx * co;
    bdz += ba * k * cz * co;
  }
  float jxx = 1.0 - uChop * squeezeX;
  float jzz = 1.0 - uChop * squeezeZ;
  float jxz = -uChop * squeezeXZ;
  float sx = dhdx + chopDx;
  float sz = dhdz + chopDz;
  float nx = sz * jxz - sx * jzz;
  float ny = jxx * jzz - jxz * jxz;
  float nz = jxz * sx - sz * jxx;
  float len = length(vec3(nx, ny, nz));
  if (len == 0.0) len = 1.0;
  float slope = length(vec2(bdx, bdz));
  bool depthLimited = bh > 0.0 && bh >= uDepthLimit;
  float steep = (slope >= uBreakSlope || depthLimited) ? 1.0 : 0.0;
  oA = vec4(height, uChop * shiftX, uChop * shiftZ, ny);
  oB = vec4(nx / len, ny / len, nz / len, steep);
  oC = vec4(dhdx, dhdz, bh, slope);
  oD = vec4(xh, xdx, xdz, 0.0);
}`;
}

// v27.2: sum of H over x - uR .. x + uR in each row (for LOCAL_FS's translucency mean).
const ROWSUM_FS = HEADER + `
uniform sampler2D tA;
uniform sampler2D tC;
uniform int uR;
uniform sampler2D tX;
uniform int uShort; // v28 long-slope rule: the trough mean is of the short-wave height (H - explicit)
uniform int uRB; // v28 foam fix: the break test's trough window (cells); .w is that row MEAN, clamped at the edges
out vec4 o; // row sums over 2uR+1 cells: height, break height; row max of the steepest slope; .w row sum of height over 2uRB+1
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float t = 0.0, tb = 0.0, st = 0.0, tw = 0.0;
  // clamped at the tile edge: the mean over the cells that exist
  { int x0 = max(p.x - uRB, 0), x1 = min(p.x + uRB, uN - 1);
    for (int x = x0; x <= x1; x += 1) tw += texelFetch(tA, ivec2(x, p.y), 0).x - (uShort == 1 ? texelFetch(tX, ivec2(x, p.y), 0).x : 0.0);
    tw /= float(x1 - x0 + 1); }
  if (p.x >= uR && p.x < uN - uR) for (int dx = -uR; dx <= uR; dx += 1) {
    ivec2 q = p + ivec2(dx, 0);
    t += texelFetch(tA, q, 0).x;
    vec4 c = texelFetch(tC, q, 0);
    tb += c.z;
    st = max(st, length(c.xy));
  }
  o = vec4(t, tb, st, tw);
}`;

const LOCAL_FS = HEADER + OFFSET_GLSL + `
uniform sampler2D tA;
uniform sampler2D tB;
uniform sampler2D tHX;
uniform sampler2D tC;
uniform sampler2D tX;
uniform int uReach;
uniform int uR;
uniform float uSpacing;
uniform vec4 uTransl; // skin, length, max thickness, unused
uniform vec2 uOpt;    // smoothing on, lip gate on
uniform vec2 uBrkMod; // M, k_L
uniform float uBrkLg; // long-slope gain (diagnostic ?brkLg, default 1)
uniform float uDepthEta; // v29.1: depth-limited break, crest elevation of the short-wave height (the break test excludes the explicit swell) >= 0.06 d (tuned) while the Hs cap bites; 1e9 otherwise
uniform vec4 uBrk;   // v28 foam fix: mode (1 = total-height steepness rule), steepness threshold, trough window uRB (cells), curvature stencil (cells)
uniform vec4 uWidth;  // v28 width test: sample distance (cells), thick-water Jacobian, Jacobian floor, unused
out vec4 o; // breaks, lips, thins, travel-axis crest
float H(ivec2 q) { return texelFetch(tA, q, 0).x; }
// v28 long-slope rule (mode 2): the break test reads the short-wave height (total minus the
// explicit long waves) and adds the long wave's local slope to the short wave's steepness.
float HB(ivec2 q) { return texelFetch(tA, q, 0).x - (uBrk.x > 1.5 ? texelFetch(tX, q, 0).x : 0.0); }
float HBS(ivec2 q) {
  float t = 0.0;
  for (int dz = -1; dz <= 1; dz += 1) for (int dx = -1; dx <= 1; dx += 1) t += HB(q + ivec2(dx, dz));
  return t / 9.0;
}
float HS(ivec2 q) {
  float t = 0.0;
  for (int dz = -1; dz <= 1; dz += 1) for (int dx = -1; dx <= 1; dx += 1) t += H(q + ivec2(dx, dz));
  return t / 9.0;
}
float HQ(ivec2 q) { return uWidth.w > 2.5 ? texelFetch(tC, q, 0).z : (uOpt.x > 0.5 ? HS(q) : H(q)); }
// crest thickness at q by the same estimate (1e3 where not a crest or out of range)
float thickAt(ivec2 q) {
  if (q.x < 2 || q.y < 2 || q.x >= uN - 2 || q.y >= uN - 2) return 1e3;
  float h = HQ(q);
  float lap = (HQ(q - ivec2(1, 0)) + HQ(q + ivec2(1, 0)) + HQ(q - ivec2(0, 1)) + HQ(q + ivec2(0, 1)) - 4.0 * h) / (uSpacing * uSpacing);
  float jac = max(texelFetch(tA, q, 0).w, uWidth.z);
  float kappa = -lap / (jac * jac);
  if (kappa <= 1e-6) return 1e3;
  float slopeD = length(texelFetch(tC, q, 0).xy) / jac;
  float depthBelow = (slopeD * slopeD) / (2.0 * kappa) + uTransl.x;
  return 2.0 * sqrt((2.0 * depthBelow) / kappa);
}
float BH(ivec2 q) { return texelFetch(tC, q, 0).z; }
// v28 lip-width rule: thin candidate = crest thinner than max thickness by the raw estimate
// dominant travel direction (the wind's, which the FFT's dominant k follows)
ivec2 tv(int s) { vec2 t = length(uTravel) > 1e-6 ? normalize(uTravel) : vec2(1.0, 0.0); return ivec2(round(t * float(s))); }
bool candAt(ivec2 q) { return inside(q) && thickAt(q) < uTransl.z; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 steps[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
  float brk = 0.0, lip = 0.0, thin = 0.0, crest = 0.0;
  int mB = int(uBrk.w) + 1;
  if (uBrk.x > 0.5) {
    // v28 foam fix (GAD): the break test on the total height (FFT + explicit waves). A crest
    // (local max of the height) breaks when its own wave is steep: elevation above the local
    // trough level (mean over the trough window) times the cell's dominant wavenumber, from
    // the crest curvature of the 3x3-averaged height (for a wave a sin(kx): -h''/h = k^2 at
    // the crest, so a k = sqrt(-h'' h)), reaches the steepness threshold.
    if (p.x >= mB && p.y >= mB && p.x < uN - mB && p.y < uN - mB) {
      float h = HB(p);
      float drop = 0.0;
      ivec2 d = ivec2(0);
      for (int i = 0; i < 4; i += 1) {
        float fall = h - HB(p + steps[i]);
        if (fall > drop) { drop = fall; d = steps[i]; }
      }
      if (drop > 0.0 && h >= HB(p - d)) {
        int RB = int(uBrk.z);
        int z0 = max(p.y - RB, 0), z1 = min(p.y + RB, uN - 1);
        float mean = 0.0;
        for (int z = z0; z <= z1; z += 1) mean += texelFetch(tHX, ivec2(p.x, z), 0).w;
        mean /= float(z1 - z0 + 1);
        int TS = int(uBrk.w);
        float hs = HBS(p);
        float dd = float(TS) * uSpacing;
        float lap = (HBS(p - ivec2(TS, 0)) + HBS(p + ivec2(TS, 0)) + HBS(p - ivec2(0, TS)) + HBS(p + ivec2(0, TS)) - 4.0 * hs) / (dd * dd);
        float eta = hs - mean;
        // mode 2: plus the long explicit wave's local slope under this crest
        float longSlope = uBrk.x > 1.5 ? uBrkLg * length(texelFetch(tX, p, 0).yz) : 0.0;
        // GAD 20:05 (a): short waves riding a long crest are steepened: x (1 + M a_L k_L), a_L the
        // long-wave elevation here (crests only), k_L the peak wavenumber (uBrkMod = M, k_L).
        float strain = uBrk.x > 1.5 ? 1.0 + uBrkMod.x * max(0.0, texelFetch(tX, p, 0).x * uBrkMod.y) : 1.0;
        if (eta > 0.0 && lap < 0.0 && sqrt(-lap * eta) * strain + longSlope >= uBrk.y) brk = 0.25;
        // v29.1 depth-limited breaking: a crest standing half the breaker height above its trough level
        if (eta >= uDepthEta && lap < 0.0) brk = 0.25;
      }
    }
  } else if (p.x >= 1 && p.y >= 1 && p.x < uN - 1 && p.y < uN - 1) {
    float h = BH(p);
    float drop = 0.0;
    ivec2 d = ivec2(0);
    for (int i = 0; i < 4; i += 1) {
      float fall = h - BH(p + steps[i]);
      if (fall > drop) { drop = fall; d = steps[i]; }
    }
    float opposite = BH(p - d);
    bool face = false;
    for (int s = 0; s <= uReach; s += 1) {
      ivec2 q = p + d * s;
      if (!inside(q)) break;
      if (texelFetch(tB, q, 0).w > 0.5) { face = true; break; }
    }
    if (drop > 0.0 && h >= opposite && face) brk = 0.25;
  }
  if (p.x >= 1 && p.y >= 1 && p.x < uN - 1 && p.y < uN - 1) {
    float hl = H(p);
    float dropL = 0.0;
    ivec2 l = ivec2(0);
    for (int i = 0; i < 4; i += 1) {
      float fall = hl - H(p + steps[i]);
      if (fall > dropL) { dropL = fall; l = steps[i]; }
    }
    float oppositeL = H(p - l);
    if (dropL > 0.0 && hl >= oppositeL && hl > 0.0) {
      float steepest = 0.0;
      for (int s = 0; s <= uReach; s += 1) {
        ivec2 q = p + l * s;
        if (!inside(q)) break;
        float faceSlope = length(texelFetch(tC, q, 0).xy);
        if (faceSlope > steepest) steepest = faceSlope;
      }
      if (steepest >= 0.12) lip = min(1.0, (steepest - 0.12) / 0.28) * min(1.0, hl / 0.8);
    }
  }
  // Thin-crest transmission (see TRANSLUCENCY in surface.js).
  // v28: the FFT heights carry chop down to one cell. Read raw, that chop's curvature and slope
  // on a broad crest top make the whole face look thin (GAD slab, v28-5.20). Height, slope and
  // curvature are all taken from the 3x3-averaged height over a TS-cell stencil, about the
  // scale v27's grid resolved, so only a real lip is thin.
  if (p.x >= uR + 3 && p.y >= uR + 3 && p.x < uN - uR - 3 && p.y < uN - uR - 3) {
    const int TS = 2;
    float h = HQ(p);
    float hx0 = HQ(p - ivec2(TS, 0)), hx1 = HQ(p + ivec2(TS, 0));
    float hz0 = HQ(p - ivec2(0, TS)), hz1 = HQ(p + ivec2(0, TS));
    float d = float(TS) * uSpacing;
    float lap = (hx0 + hx1 + hz0 + hz1 - 4.0 * h) / (d * d);
    // FFT chop folds the crest band (J near 0); dividing by J^2 there turned a broad band into
    // a 'lip' 25x too curved. The thickness estimate keeps J >= 0.6.
    float jac = max(texelFetch(tA, p, 0).w, uWidth.z);
    float kappa = -lap / (jac * jac);
    if (kappa > 1e-6) {
      float mean = 0.0, steepN = 0.0;
      // v27.2 rowsum: 2uR + 1 fetches (was (2uR + 1)^2); the crest source's sum (break height
      // in modes 3-4, else height) and the steepest slope come from the same rows.
      for (int dz = -uR; dz <= uR; dz += 1) {
        vec4 rs = texelFetch(tHX, p + ivec2(0, dz), 0);
        mean += uWidth.w > 2.5 ? rs.y : rs.x;
        steepN = max(steepN, rs.z);
      }
      mean /= float((2 * uR + 1) * (2 * uR + 1));
      // A lip stands on a steep face (as the crest glow does, slope 0.12-0.4); a broad,
      // gently sloped crest face is not a lip and gets no transmission.
      float lipGate = mix(1.0, smoothstep(0.12, 0.4, steepN), uOpt.y);
      if (h - mean > 0.0 && lipGate > 0.0) {
        float slopeD = length(vec2(hx1 - hx0, hz1 - hz0) / (2.0 * d)) / jac;
        float depthBelow = (slopeD * slopeD) / (2.0 * kappa) + uTransl.x;
        float thickness = 2.0 * sqrt((2.0 * depthBelow) / kappa);
        if (thickness < uTransl.z) {
          float cut = 1.0 - thickness / uTransl.z;
          thin = lipGate * exp(-thickness / uTransl.y) * cut * cut * (3.0 - 2.0 * cut);
          // Width test: a lip is narrow, so the water just above and below it along the slope
          // is thick (Jacobian near 1); a broad face is low-Jacobian across and fails.
          vec2 gr = texelFetch(tC, p, 0).xy;
          if (length(gr) > 1e-6) {
            ivec2 o = ivec2(round(normalize(gr) * uWidth.x));
            ivec2 qa = p + o, qb = p - o;
            if (uWidth.w > 2.5 && uWidth.w < 3.5) {
              // mode 3: v27 crest source (band-limited height), no width test
            } else if (uWidth.w > 1.5) {
              // mode 2 (FFT height) / mode 4 (band-limited height): lip-width rule
              // mode 2: run of candidates along travel <= uWidth.x cells, and the cell on the
              // crest's forward edge (Jacobian falling along travel)
              int run = 1;
              for (int s = 1; s <= 4; s += 1) { if (candAt(p + tv(s))) run += 1; else break; }
              for (int s = 1; s <= 4; s += 1) { if (candAt(p + tv(-s))) run += 1; else break; }
              ivec2 fa = p + tv(1), fb = p + tv(-1);
              float jf = inside(fa) ? texelFetch(tA, fa, 0).w : 1.0;
              float jb = inside(fb) ? texelFetch(tA, fb, 0).w : 1.0;
              bool fwd = jf < jb - uWidth.y;
              // crest continuity (uTransl.w > 0.5): a lip runs along the crest, so a candidate
              // with no candidate beside it across travel is an isolated spot, not a lip
              ivec2 pc = tv(1).yx * ivec2(-1, 1);
              bool cont = uTransl.w < 0.5 || candAt(p + pc) || candAt(p - pc);
              if (float(run) > uWidth.x + 0.5 || !fwd || !cont) thin = 0.0;
            } else if (uWidth.w > 0.5) {
              // thickness mode: the neighbours' own crest thickness must exceed ~2 m
              thin *= smoothstep(uWidth.y - 0.5, uWidth.y + 0.5, min(thickAt(qa), thickAt(qb)));
            } else {
              float ja = inside(qa) ? texelFetch(tA, qa, 0).w : 1.0;
              float jb = inside(qb) ? texelFetch(tA, qb, 0).w : 1.0;
              thin *= smoothstep(uWidth.y - 0.1, uWidth.y + 0.1, min(ja, jb));
            }
          }
        }
      }
    }
  }
  // A crest along the travel axis: higher than the samples just behind and just ahead.
  {
    ivec2 b = p + offT(-1);
    ivec2 f = p + offT(1);
    if (inside(b) && inside(f)) {
      float h = BH(p);
      if (h > 0.0 && h >= BH(b) && h >= BH(f)) crest = 1.0;
    }
  }
  o = vec4(brk, lip, thin, crest);
}`;

const JOIN_FS = HEADER + OFFSET_GLSL + `
uniform sampler2D tL;
uniform sampler2D tC;
out vec4 o; // partner step s (0 none), w, gap snaps (base 3, m + 1 per gap cell), gap ok
float BH(ivec2 q) { return texelFetch(tC, q, 0).z; }
bool crestBreak(ivec2 q) {
  if (!inside(q)) return false;
  vec4 l = texelFetch(tL, q, 0);
  return l.x > 0.0 && l.w > 0.5;
}
bool travelCrest(ivec2 q) { return inside(q) && texelFetch(tL, q, 0).w > 0.5; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = vec4(0.0);
  if (!crestBreak(p)) return;
  int ws[3] = int[3](0, -1, 1);
  for (int s = 1; s <= 5; s += 1) {
    for (int wi = 0; wi < 3; wi += 1) {
      int w = ws[wi];
      ivec2 partner = p + offC(s) + offT(w);
      if (!inside(partner)) continue;
      if (!crestBreak(partner)) continue;
      if (s == 1) { o = vec4(1.0, float(w), 0.0, 0.0); return; }
      float packed = 0.0;
      float place = 1.0;
      for (int j = 1; j < 5; j += 1) {
        if (j >= s) break;
        ivec2 q = p + offC(j) + uOffF[pairIndex(s, j) * 3 + w + 1];
        int best = 2;
        float bestH = 0.0;
        for (int mi = 0; mi < 3; mi += 1) {
          int m = ws[mi];
          ivec2 c = q + offT(m);
          if (!travelCrest(c)) continue;
          // Never snap behind both ends of the gap: that would be the back of the crest.
          float offB = dot(vec2(c - p), uTravel);
          float offP = dot(vec2(c - partner), uTravel);
          // ...and only where the cell is ahead of an end it is within 4 cells of (CPU mirror).
          ivec2 dB = c - p; ivec2 dP = c - partner;
          bool okB = offB >= -0.5 && dB.x * dB.x + dB.y * dB.y <= 16;
          bool okP = offP >= -0.5 && dP.x * dP.x + dP.y * dP.y <= 16;
          if (!okB && !okP) continue;
          float hc = BH(c);
          if (best == 2 || hc > bestH) { best = m; bestH = hc; }
        }
        if (best == 2) { o = vec4(float(s), float(w), 0.0, 0.0); return; }
        packed += float(best + 1) * place;
        place *= 3.0;
      }
      o = vec4(float(s), float(w), packed, 1.0);
      return;
    }
  }
}`;

const PAINT_FS = HEADER + OFFSET_GLSL + `
uniform sampler2D tA;
uniform sampler2D tC;
uniform sampler2D tL;
uniform sampler2D tJ;
uniform sampler2D tJP;
uniform float uHalf;
uniform float uSpacing;
uniform float uNoAhead;
uniform int uFace;      // face steps (faceStepsAt): 0.1 x peak wavelength in cells
uniform int uJoinSteps; // joined gap cells: min(3, uFace)
out vec4 o; // flag, grade, metres ahead of the crest it was painted from, 0
// surface.js faceGrade: the 4-step shape (1, 0.5, 0.2, 0.06) stretched over F steps
float faceGrade(int s, int F) {
  float G[4] = float[4](1.0, 0.5, 0.2, 0.06);
  float u = 3.0 * float(s) / float(max(F, 1));
  int i = int(floor(u));
  if (i >= 3) return G[3];
  return G[i] + (G[i + 1] - G[i]) * (u - float(i));
}
float BH(ivec2 q) { return texelFetch(tC, q, 0).z; }
float travelPos(ivec2 q) {
  vec2 sh = texelFetch(tA, q, 0).yz;
  return (-uHalf + float(q.x) * uSpacing + sh.x) * uTravel.x + (-uHalf + float(q.y) * uSpacing + sh.y) * uTravel.y;
}
bool nearTo(ivec2 q, ivec2 a, ivec2 b) {
  ivec2 da = q - a;
  ivec2 db = q - b;
  return da.x * da.x + da.y * da.y <= 16 || db.x * db.x + db.y * db.y <= 16;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float grade = 0.0;
  float key = 1e30;
  float crestPos = 0.0;
  float idxScale = 32.0;
  // Direct paint: a break s steps behind, with the face dropping all the way to here.
  for (int s = 0; s <= ${FACE_MAX}; s += 1) {
    if (s > uFace) break;
    ivec2 b = p - offT(s);
    if (!inside(b) || !(texelFetch(tL, b, 0).x > 0.0)) continue;
    bool ok = true;
    float prev = BH(b);
    for (int k = 1; k <= ${FACE_MAX}; k += 1) {
      if (k > s) break;
      ivec2 f = b + offT(k);
      if (!inside(f)) { ok = false; break; }
      float h = BH(f);
      if (h >= prev) { ok = false; break; }
      prev = h;
    }
    if (!ok) continue;
    float g = faceGrade(s, uFace);
    float kk = float(b.y * uN + b.x) * idxScale + float(s);
    if (g > grade || (g == grade && kk < key)) { grade = g; key = kk; crestPos = travelPos(b); }
  }
  // Joined paint: a gap cell g of some crest break's join, k face steps behind here.
  // v27.2: JOINPAINT_FS has already found, per gap cell g and per k, the smallest key of
  // the joins that paint k steps ahead of g (grade depends on k alone), so this is the
  // same choice as the full loop, in 4 fetches.
  for (int k = 0; k <= 3; k += 1) {
    if (k > uJoinSteps) break;
    ivec2 g = p - offT(k);
    if (!inside(g)) continue;
    float kk = texelFetch(tJP, g, 0)[k];
    if (kk >= 1e29) continue;
    float gr = faceGrade(k, uFace);
    if (gr > grade || (gr == grade && kk < key)) { grade = gr; key = kk; crestPos = travelPos(g); }
  }
  if (grade > 0.0) o = vec4(1.0, grade, travelPos(p) - crestPos, 0.0);
  else o = vec4(0.0, 0.0, uNoAhead, 0.0);
}`;

const AGE_FS = HEADER + OFFSET_GLSL + `
uniform sampler2D tP;
uniform sampler2D tS;
uniform sampler2D tAges;  // AGE_TABLE: float32 age of a sample flagged k steps ago, at (k % 64, k / 64)
uniform int uAgeLen;      // ages from k = uAgeLen on are past the window
uniform float uStep;
uniform float uWindow;
uniform float uNoAhead;
layout(location = 0) out vec4 oS;  // step last flagged, kept grade, kept ahead, kept run
layout(location = 1) out vec4 oF;  // foam grade, fade, age / window, strength
layout(location = 2) out vec4 oG;  // ahead, run, recent, flag
bool foamAt(ivec2 q) { return inside(q) && texelFetch(tP, q, 0).y > 0.0; }
int countRun(ivec2 p, int dir) {
  int c = 0;
  ivec2 cur = p;
  int ws[3] = int[3](0, -1, 1);
  for (int s = 1; s <= 2; s += 1) {
    ivec2 x = cur + offC(dir);
    bool found = false;
    for (int wi = 0; wi < 3; wi += 1) {
      ivec2 q = x + offT(ws[wi]);
      if (foamAt(q)) { cur = q; found = true; break; }
    }
    if (!found) break;
    c += 1;
  }
  return c;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 paint = texelFetch(tP, p, 0);
  vec4 st = texelFetch(tS, p, 0);
  bool flagged = paint.x > 0.0;
  float run = 0.0;
  if (paint.y > 0.0) {
    int a = countRun(p, 1);
    int b = countRun(p, -1);
    run = a > 0 && b > 0 ? 1.0 : max(a, b) >= 2 ? 0.0 : 0.5;
  }
  // v29.1 foam fade-in: st.w packs the run flag (0, 0.5, 1) plus 2 x the step the current break
  // started (a break that was not flagged in the last 3 steps starts a new onset).
  float onset = floor(st.w / 2.0);
  if (flagged) { if (uStep - st.x > 3.0 || st.x < 0.0) onset = uStep; st = vec4(uStep, paint.y, paint.z, run + 2.0 * onset); }
  float stRun = mod(st.w, 2.0);
  float sinceOn = uStep - onset;
  float ageOn = sinceOn < float(uAgeLen) ? texelFetch(tAges, ivec2(int(sinceOn) % 64, int(sinceOn) / 64), 0).x : 1e30;
  float fadeIn = smoothstep(0.0, 0.45, ageOn); // ramps over ~0.45 s from the break's first step
  float since = uStep - st.x;
  bool live = since < float(uAgeLen);
  float age = live ? texelFetch(tAges, ivec2(int(since) % 64, int(since) / 64), 0).x : 1e30;
  if (live) {
    float u = age / uWindow;
    // 1 - u^2 (3 - 2u), written as (1 - u)^2 (1 + 2u): the same fade, still above 0 right up
    // to the window's end in float32 (surface.js works in double precision).
    float v = (uWindow - age) / uWindow;
    float fade = (flagged ? 1.0 : v * v * (1.0 + 2.0 * u)) * fadeIn;
    oF = vec4(st.y, fade, flagged ? 0.0 : age / uWindow, st.y * fade);
    oG = vec4(st.z, stRun, flagged ? 0.0 : 1.0, flagged ? 1.0 : 0.0);
  } else {
    oF = vec4(0.0, 1.0, 0.0, 0.0);
    oG = vec4(uNoAhead, 0.0, 0.0, flagged ? 1.0 : 0.0);
  }
  oS = st;
}`;

// v30c physics foam: continuous coverage on the physics grid (ping-pong tK).
//   .x dense whitewater (0..1), .y residual bubble film (0..1), .z age-weighted dense (for the bubble-cloud glow)
// Grid cells are the FFT sea's undisplaced (Lagrangian) positions, so the orbital motion is already
// carried by the displacement: a foam parcel stays on its cell while the cell is swept and squeezed by
// the waves, and its surface density is coverage / Jacobian (the renderer divides), which gathers foam
// where the surface converges (negative divergence of the orbital velocity) and thins it where it spreads.
// What is left to advect here is the mean drift (wind drift ~3% of U10 + Stokes drift), semi-Lagrangian.
// Sources: the break test's painted front (grade 0..1, same Jacobian/steepness test and wind-eased
// threshold as v30b). Decay in two stages: dense e-fold uTauD (~3 s) feeding a residual film that
// e-folds over uTauR (~30 s); a share of the dense loss becomes residual.
const COVER_FS = HEADER + `
#define ECR ${EN_CREF}
#define EK ${EN_K}

uniform sampler2D tP; uniform sampler2D tL; // v30.18.3
uniform sampler2D tCU; // v30.22c curl field (.y landing foam)
uniform sampler2D tK;
uniform sampler2D tA;
uniform float uJth;
uniform vec2 uShift;   // drift this step, in cells
uniform float uDt;
uniform float uTauD;
uniform float uTauR;
uniform float uTauA;
uniform float uFeed;   // share of dense loss that becomes residual film
uniform float uSrc;    // source rate (1/s) toward the painted grade
uniform vec2 uCrest;  // unit vector along the crest (across travel), in cells
uniform float uSrcK;   // breaking-strength gain, rising with wind (whitecap coverage ~ U^3)
uniform float uTime;
out vec4 o;
uniform float uDepth; uniform float uSp; uniform vec2 uWind; uniform sampler2D tHL; uniform float uLp; // v30.8b: depth (m), physics cell (m), wind unit vector
uniform int uM;        // v30.8: foam grid size (2x the physics grid: 512 at N=256)
vec4 kAt(ivec2 c) { return texelFetch(tK, (c % uM + uM) % uM, 0); }
// physics fields sampled bilinearly at a coarse-grid position (index space, wraps)
float bilHL(vec2 p) { vec2 i0 = floor(p), f = p - i0; ivec2 c = ivec2(i0);
  #define TH(o) texelFetch(tHL, ((c + o) % 64 + 64) % 64, 0).x
  return mix(mix(TH(ivec2(0, 0)), TH(ivec2(1, 0)), f.x), mix(TH(ivec2(0, 1)), TH(ivec2(1, 1)), f.x), f.y); }
vec4 bilA(sampler2D t, vec2 p) { vec2 i0 = floor(p), f = p - i0; ivec2 c = ivec2(i0);
  #define TW(o) texelFetch(t, ((c + o) % uN + uN) % uN, 0)
  return mix(mix(TW(ivec2(0, 0)), TW(ivec2(1, 0)), f.x), mix(TW(ivec2(0, 1)), TW(ivec2(1, 1)), f.x), f.y); }
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 p) { vec2 i = floor(p), f = p - i; f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
// v30.18.1 passive drift: eddy swirl in METRES (scale 18 m / 7 m), speed capped at SWIRL_MAX m/s, so foam
// left behind drifts with the surface current (uShift, ~3% of wind) and never outruns it or runs backward.
vec2 swirlM(vec2 m, float turb) { float e = 0.5;
  #define PSM(q) (vn((q) / 18.0 + vec2(0.0, uTime * 0.05)) + 0.5 * vn((q) / 7.0 + vec2(uTime * 0.08, 17.0)))
  vec2 c = vec2(PSM(m + vec2(0.0, e)) - PSM(m - vec2(0.0, e)), -(PSM(m + vec2(e, 0.0)) - PSM(m - vec2(e, 0.0)))) / (2.0 * e) * 18.0;
  vec2 v = c * 0.08 * (0.4 + turb); float l = length(v); return l > 0.15 ? v * (0.15 / l) : v; }
float psi(vec2 p) { return vn(p / 9.0 + vec2(0.0, uTime * 0.05)) + 0.5 * vn(p / 4.0 + vec2(uTime * 0.08, 17.0)); }
void main() {
  // v30.6 step 4: the turbulence a break leaves behind stirs the foam. A divergence-free (curl-noise,
  // Bridson 2007) eddy velocity, strongest where the bubble plume (air) says the water was just
  // churned, slowly evolving; it shears and swirls the patches and tears their ends.
  float rM = float(uM) / float(uN);           // fine cells per physics cell
  vec2 pc0 = (gl_FragCoord.xy - 0.5) / rM;    // physics-grid position
  // v31.3: the cell does not advect. A multiply by 0 still executes swirlM, psi and the
  // Langmuir noise on every coverage texel (GLSL does not drop those calls), which is
  // the same class of work that already set the Mac splat budget. The step is the
  // texel itself. Orbital motion stays in the mesh.
  vec2 x = gl_FragCoord.xy - 0.5;
  vec2 i0 = floor(x); vec2 f = x - i0; ivec2 c = ivec2(i0);
  vec4 k = mix(mix(kAt(c), kAt(c + ivec2(1, 0)), f.x), mix(kAt(c + ivec2(0, 1)), kAt(c + ivec2(1, 1)), f.x), f.y);
  // v30c.3 whitewater seed: only the crest tip. A cell seeds when the break test flags it (paint.x,
  // same Jacobian/steepness test and wind-eased threshold), the surface is folding there (Jacobian
  // below uJth) and it is a local height maximum along the travel direction (within 2 cells).
  // Smoothed along the crest only (5 taps): a thin line 1-2 cells across lying along the crest. The
  // cells then move with the water, so the crest runs ahead and leaves a band trailing behind it.
  // v30.7 whole-crest seed (foam-realism-2 rank 1): breaking happens along the whole segment that
  // passes the limit, so the seed is a soft test on the cell itself (folding Jacobian, painted grade
  // as a boost, raised above the local mean height), no local-maximum test. It is smoothed along the
  // crest's OWN direction (perpendicular to the local height gradient), not one global vector.
  vec2 gh = vec2(bilA(tA, pc0 + vec2(1, 0)).x - bilA(tA, pc0 - vec2(1, 0)).x, bilA(tA, pc0 + vec2(0, 1)).x - bilA(tA, pc0 - vec2(0, 1)).x);
  vec2 cr = length(gh) > 1e-5 ? vec2(-gh.y, gh.x) / length(gh) : uCrest;
  // v30.8b whitecap size follows its wave: local wavelength from curvature/elevation (L = 2pi/k,
  // k^2 ~ |lap h| / |h|); the seed is joined along ~0.4 L of that crest (9 taps).
  // v30.8.2: crest line and size from the LONG wave (peak band): one breaker -> one band ~0.3 lambda_p
  // long, joined along that crest; seeds only on its upper part (hl / local envelope).
  float sH = float(uN) / 64.0;  // physics cells per long-grid texel
  #define HLAT(p) bilHL((p) / sH)
  float hl = HLAT(pc0);
  vec2 ghl = vec2(HLAT(pc0 + vec2(sH, 0.0)) - HLAT(pc0 - vec2(sH, 0.0)), HLAT(pc0 + vec2(0.0, sH)) - HLAT(pc0 - vec2(0.0, sH)));
  float env = 1e-3; float rr0 = 0.5 * uLp / uSp;
  for (int a = 0; a < 8; a++) { float an = 0.785 * float(a); env = max(env, abs(HLAT(pc0 + rr0 * vec2(cos(an), sin(an))))); }
  env = max(env, abs(hl));
  float tstep = clamp(0.3 * uLp / (8.0 * uSp), 0.5, 8.0);
  if (length(ghl) > 1e-5) cr = vec2(-ghl.y, ghl.x) / length(ghl);
  float src = 0.0, ws = 0.0, eng = 0.0;
  for (int i = -4; i <= 4; i++) {
    float w = exp(-0.5 * float(i * i) / 10.0);
    vec2 pc = pc0 + tstep * float(i) * cr;
    vec4 aq = bilA(tA, pc);
    float g = bilA(tP, pc).y;
    float fold = 1.0 - smoothstep(uJth - 0.3, uJth + 0.05, aq.w);
    float tip = max(fold * (0.45 + 0.55 * smoothstep(0.2, 0.8, g)), 0.7 * smoothstep(0.5, 0.95, g)); // soft OR: folding, or the break test's steep face
    tip *= smoothstep(-0.15, 0.35, aq.x);                                         // on the raised crest/forward face, not the trough
    tip *= smoothstep(0.3, 0.7, HLAT(pc) / env);                                   // only on the long wave's crest
    { vec4 lq = bilA(tL, pc); float bS = smoothstep(0.45, 1.0, clamp(max(4.0 * lq.x, lq.y), 0.0, 1.0)); tip *= min(1.0, EK * bS * pow(sqrt(17.4 * max(aq.x, 0.0)) / ECR, 5.0)); } // v30.19 energy // v30.18.3: only under a breaking lip
    tip = max(tip, smoothstep(0.7, 0.9, 2.0 * max(aq.x, 0.0) / max(uDepth, 0.5))); // depth-limited breaking, H/d > 0.78 (surf)
    src = max(src, mix(0.6, 1.0, w) * tip);  // v30.8.3: dilate along the long crest (~0.3 lambda_p) so neighbours merge
    eng = max(eng, tip); // continuous break intensity (lip energy already in tip), before the spawn threshold
  }
  src = min(1.0, uSrcK * smoothstep(0.3, 0.6, src)); // v30.8.3: weak isolated seeds (fragments) die
  { ivec2 cq = (ivec2(floor(pc0 + 0.5)) % uN + uN) % uN; vec4 cu = texelFetch(tCU, cq, 0); if (cu.x < -0.5) src = max(src, clamp(cu.y, 0.0, 1.0)); } // v30.22c: whitewater where a curl's lip lands
  // slow turbulent spreading of the patch (~0.05 m^2/s eddy diffusion)
  vec4 nb = 0.25 * (kAt(c + ivec2(-1, 0)) + kAt(c + ivec2(2, 0)) + kAt(c + ivec2(0, -1)) + kAt(c + ivec2(0, 2)));
  k = mix(k, nb, 0.04 * rM * rM); // same eddy diffusivity on the finer grid
  float d0 = k.x;
  float dense = d0 * exp(-uDt / uTauD);
  float resid = k.y * exp(-uDt / uTauR) + uFeed * (d0 - dense);
  float fresh = k.z * exp(-uDt / 0.8);
  if (src > 0.0) { float a = 1.0 - exp(-uSrc * uDt); dense = max(dense, dense + (src - dense) * a); fresh = max(fresh, fresh + (src - fresh) * a); } // v30.7: persistence, max-accumulate
  // Bubble water, not surface foam. eng is the 9-tap lip-energy intensity above,
  // gated by the breaking lip and read before the spawn threshold. Proxy, not a
  // CFD dissipation. Air integrates that dose (0.45 per second at eng = 1), spreads
  // with the existing neighbour mix, then decays with the bubble lifetime. A weaker
  // lip deposits less. The store is not set equal to eng, and dying surface foam
  // is not a source. Foam births (src) are unchanged.
  float held = mix(k.w, nb.w, min(0.6, 0.25 * rM * rM));
  float airN = held * exp(-uDt / uTauA) + eng * uDt * 0.45;
  o = vec4(clamp(dense, 0.0, 1.0), clamp(resid, 0.0, 1.0), clamp(fresh, 0.0, 1.0), clamp(airN, 0.0, 1.0));
}`;

// v30.7 coverage stats for the controller: 16x16 blocks of the coverage grid -> (active share,
// visible share). Read back asynchronously like the boat probe.
const COVSTAT_FS = HEADER + `
uniform sampler2D tK;
uniform highp sampler2DArray tS2; uniform vec2 uWc; uniform float uWs; // v30.22f: drawn strand foam in a window (uv) around the camera
uniform int uM;
out vec4 o;
void main() {
  int B = uM / 16; ivec2 base = ivec2(gl_FragCoord.xy) * B;
  float a = 0.0, v = 0.0;
  for (int j = 0; j < 64; j += 2) { if (j >= B) break; for (int i = 0; i < 64; i += 2) { if (i >= B) break;
    vec4 k = texelFetch(tK, base + ivec2(i, j), 0);
    a += step(0.5, k.x); v += step(0.12, max(k.x, k.y)); } }
  float nn = float(B * B) / 4.0;
  // v30.22f FOAM SEA: share of the sea showing foam as the water shader draws it (whitecap: 4 x young strands
  // at lod 2-3 >= foamvis; lace: 1 - exp(-2.5 (old + 0.5 young)) >= foamvis), 8 x 8 samples per block
  float f = 0.0, fl = 0.0;
  for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) {
    vec2 uv = uWc + uWs * ((gl_FragCoord.xy - 0.5 + (vec2(i, j) + 0.5) / 8.0) / 16.0 - 0.5);
    vec4 sb = 0.5 * (textureLod(tS2, vec3(uv, 0.0), 2.0) + textureLod(tS2, vec3(uv, 0.0), 3.0));
    float sd = textureLod(tS2, vec3(uv, 0.0), 1.8).r;
    float dW = clamp(4.0 * sb.g, 0.0, 1.0), aR = 1.0 - exp(-2.5 * (sd + 0.5 * sb.g));
    // the water shader breaks lace up with its lace mask (v30.22f: also at medium range): only ~1/3 of a
    // lace-covered area shows foam (?fsl=), saturated whitecap (riding) counts in full
    bool wcFull = dW >= ${FOAM_SL < 1 ? "0.9" : FOAM_VIS.toFixed(3)}; f += wcFull ? 1.0 : 0.0; fl += !wcFull && max(dW, aR) >= ${FOAM_VIS.toFixed(3)} ? 1.0 : 0.0; // v30.22g: whitecap and lace shares apart (FOAM SEA = wc + fsl x lace on the CPU) // v30.22f: only saturated whitecap counts in full (released dense foam is lace-masked too)
  }
  o = vec4(a / nn, v / nn, f / 64.0, fl / 64.0);
}`;

// v30.8.2 long-wave surface (Concept): the height band-limited to the energetic bands near the
// spectral peak (Gaussian low-pass, sigma = 0.12 lambda_p), on a 64^2 grid. Breaking is started only
// on the crests of this surface; short bands add texture but never start a whitecap.
const STRANDS = new URLSearchParams(globalThis.location ? location.search : "").get("strands") !== "0";
const STRAND_CHAINS = Math.max(4096, Math.min(16384, Number(FQ.get("chains") || 4096))); // v30.19.3: back to 4096 (8192 doubled splat/line draws -> 17 ms p99 at 25 m/s on the Mac)
const STRAND_ROWS = STRAND_CHAINS / 16;
const STRAND_BOOST = Number(FQ.get("sboost") ?? 1.5); // v30.19.3: high-wind extra spawn (15->25 m/s)
const STRAND_RATE = Number(new URLSearchParams(globalThis.location ? location.search : "").get("srate") ?? 0.12);
const SPLATS = new URLSearchParams(globalThis.location ? location.search : "").get("splat") !== "0"; // v30.18.4 soft bubble splats
const STRAND_TRAIL = Number(new URLSearchParams(globalThis.location ? location.search : "").get("strail") ?? 0.25);
const FINE = new URLSearchParams(globalThis.location ? location.search : "").get("ffine") !== "0";
const FILAMENTS = Math.max(1, Number(new URLSearchParams(globalThis.location ? location.search : "").get("ffil") ?? 7)); // v30.14 fine-layer filaments per strand
const LONGH_FS = HEADER + `
uniform sampler2D tA;
uniform float uSig;   // sigma in physics cells
out vec4 o;
void main() {
  vec2 c = (gl_FragCoord.xy - 0.5) * float(uN) / 64.0;
  float acc = 0.0, wsum = 0.0;
  for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) {
    vec2 d = vec2(float(i), float(j)) * 0.6 * uSig;
    float w = exp(-0.5 * dot(d, d) / (uSig * uSig));
    ivec2 q = (ivec2(floor(c + d)) % uN + uN) % uN;
    acc += w * texelFetch(tA, q, 0).x; wsum += w;
  }
  o = vec4(acc / wsum, 0.0, 0.0, 1.0);
}`;

// v30.9 FOAM STRANDS (Lagrangian). Foam is drawn from material lines: each breaking event spawns a
// chain of 16 nodes laid along the crest (length ~0.03 U^2 m: 3 m at 10 m/s, 19 m at 25), placed by
// rejection sampling on the fold test (Jacobian below threshold on the raised crest), so foam only
// appears where waves break and the spawn rate follows the breaking area. Nodes live on the same
// Lagrangian grid as the coverage field (the renderer divides by the Jacobian, so strands gather where
// the surface converges) and are carried by the same drift, Langmuir windrows and curl-noise eddies,
// plus a finer eddy that folds and stretches each line into loops. A segment that is stretched past
// ~4x its rest length tears (holes open, the web breaks up with age). Young chains are drawn as the
// whitecap band, old ones as the lace web.
const STRAND_FLOW = `
#define ECR ${EN_CREF}
#define EK ${EN_K}

float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 p) { vec2 i = floor(p), f = p - i; f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
`;
const STRAND_UPD_FS = HEADER + STRAND_FLOW + `
uniform sampler2D tS; uniform sampler2D tA; uniform sampler2D tK; uniform sampler2D tHL; uniform sampler2D tL; // v30.18.3 tL: break/lip detector (same as the v30.17b lip)
uniform int uM; uniform float uTime, uDt, uSp, uLen, uRate, uJth, uLife, uC, uGrow, uRideAll; uniform vec2 uShift, uWind;
uniform vec2 uCamN; uniform float uWinN, uBias; // v30.16.1: spawn near the camera in big oceans
out vec4 o;
vec4 bilA(sampler2D t, vec2 p) { vec2 i0 = floor(p), f = p - i0; ivec2 c = ivec2(i0);
  #define TW(o) texelFetch(t, ((c + o) % uN + uN) % uN, 0)
  return mix(mix(TW(ivec2(0, 0)), TW(ivec2(1, 0)), f.x), mix(TW(ivec2(0, 1)), TW(ivec2(1, 1)), f.x), f.y); }
float psi(vec2 p) { return vn(p / 9.0 + vec2(0.0, uTime * 0.05)) + 0.5 * vn(p / 4.0 + vec2(uTime * 0.08, 17.0)); }
// v30.18.1 passive drift: eddy swirl in METRES (scale 18 m / 7 m), speed capped at SWIRL_MAX m/s, so foam
// left behind drifts with the surface current (uShift, ~3% of wind) and never outruns it or runs backward.
vec2 swirlM(vec2 m, float turb) { float e = 0.5;
  #define PSM(q) (vn((q) / 18.0 + vec2(0.0, uTime * 0.05)) + 0.5 * vn((q) / 7.0 + vec2(uTime * 0.08, 17.0)))
  vec2 c = vec2(PSM(m + vec2(0.0, e)) - PSM(m - vec2(0.0, e)), -(PSM(m + vec2(e, 0.0)) - PSM(m - vec2(e, 0.0)))) / (2.0 * e) * 18.0;
  vec2 v = c * 0.08 * (0.4 + turb); float l = length(v); return l > 0.15 ? v * (0.15 / l) : v; }
float psiF(vec2 m) { return vn(m / 2.5 + vec2(uTime * 0.11, 3.0)) + 0.6 * vn(m / 1.1 + vec2(9.0, uTime * 0.17)); } // metres
void main() {
  ivec2 id = ivec2(gl_FragCoord.xy); vec4 p = texelFetch(tS, id, 0);
  int j = id.x % 16; float cid = float(id.y * 16 + id.x / 16);
  float rM = float(uM) / float(uN);
  bool act = p.w > 1000.0; float life = act ? p.w - 1000.0 : p.w;
  bool dead = life <= 0.0 || (uTime - p.z) > life;
  if (dead) {
    vec2 seed = vec2(cid * 0.137, fract(uTime * 7.31) * 911.0);
    vec2 r = vec2(h21(seed), h21(seed + 17.3)) * float(uN);
    if (h21(seed + 23.7) < uBias) r = mod(uCamN + (vec2(h21(seed + 31.1), h21(seed + 41.9)) - 0.5) * uWinN, float(uN));
    vec4 aq = bilA(tA, r);
    // v30.18.3 foam only from a visible overturn: the cell must carry a strong lip (physL, the v30.17b lip
    // test, >= 0.5 so the crest has already steepened through the weak stage) on a raised crest; the
    // chain is laid where the lip lands, ~0.4 wave heights ahead of the crest along the wind.
    vec4 lq = bilA(tL, r); float lipS = clamp(max(4.0 * lq.x, lq.y), 0.0, 1.0);
    // v30.19 foam from dissipated energy: b = how far past the breaking threshold (lip detector), local
    // crest speed from the crest height (H/lambda ~ 1/7 at breaking: c = sqrt(g 7H / 2pi) = sqrt(10.9 H)),
    // energy per metre of crest ~ b c^5 / g, relative to a reference crest (ECR m/s). Spawn rate, chain
    // length/width and life follow it: small waves make thin short streaks or nothing, big ones wide patches.
    float bS = smoothstep(0.45, 1.0, lipS);
    float cl = sqrt(10.9 * 1.6 * max(aq.x, 0.0));
    float qE = min(3.0, bS * pow(cl / ECR, 5.0));
    float fold = step(0.05, aq.x) * step(1e-4, qE);
    if (fold > 0.0 && h21(seed + 5.1) < uRate * EK * min(qE, 1.0)) {
      float sq = sqrt(qE);
      r += uWind * (0.8 * max(aq.x, 0.0)) / uSp;
      vec2 uv = r / float(uN), eh = vec2(1.0 / 64.0, 0.0);   // crest direction from the LONG wave (one breaker -> one straight band)
      vec2 g = vec2(texture(tHL, uv + eh.xy).x - texture(tHL, uv - eh.xy).x, texture(tHL, uv + eh.yx).x - texture(tHL, uv - eh.yx).x);
      vec2 cr = length(g) > 1e-5 ? vec2(-g.y, g.x) / length(g) : vec2(-uWind.y, uWind.x);
      vec2 ac = vec2(-cr.y, cr.x);
      float t = (float(j) - 7.5) / 15.0;
      float L = uLen * (0.6 + 0.8 * h21(seed + 2.2)) * clamp(0.2 + 0.8 * sq, 0.2, 1.4);
      float bow = (1.0 - 4.0 * t * t) * 0.08 * L * (h21(seed + 3.3) - 0.5);   // crests are slightly curved
      vec2 pos = r + (cr * t * L + ac * (bow + 3.0 * clamp(sq, 0.15, 1.2) * (h21(seed + 4.4) - 0.5) + 0.25 * (h21(seed + float(j) * 1.7) - 0.5))) / uSp;
      p = vec4(mod(pos * rM, float(uM)), uTime, 1000.0 + uLife * (0.6 + 0.8 * h21(seed + 9.1)) * clamp(0.25 + 0.75 * sq, 0.25, 1.3)); // born ACTIVE (riding its crest)
    } else p = vec4(p.xy, 0.0, 0.0);
  } else if (act) {
    // v31.1: an active node stays with its breaking lip only. The previous ride stepped every
    // windward face at 0.8 * phase speed (uC is sqrt(g*L/2pi), about 10-15 m/s at these winds)
    // for 1-3 s, and the step used the raw frame dt, so foam shot across the sea and a slow
    // frame jumped tens of metres. There is no simulated current, so released foam is not
    // given the old prescribed wind drift, Langmuir rows or curl-noise either. The mesh still
    // carries the orbital displacement, which is the water motion that does exist.
    vec2 pc0 = p.xy / rM;
    vec2 uv = pc0 / float(uN), eh = vec2(1.0 / 64.0, 0.0);
    vec2 g = vec2(texture(tHL, uv + eh.xy).x - texture(tHL, uv - eh.xy).x, texture(tHL, uv + eh.yx).x - texture(tHL, uv - eh.yx).x);
    float rideDt = min(uDt, 0.05);
    float lipHere = clamp(max(4.0 * bilA(tL, pc0).x, bilA(tL, pc0).y), 0.0, 1.0);
    if (lipHere > 0.45 && length(g) > 1e-7 && dot(g, uWind) > 0.0) {
      vec2 cand = pc0 + uWind * min(uC * 0.25, 4.0) * rideDt / uSp;
      vec4 lqC = bilA(tL, cand);
      float lipC = clamp(max(4.0 * lqC.x, lqC.y), 0.0, 1.0);
      if (lipC > 0.35 && texture(tHL, cand / float(uN)).x + 0.02 >= texture(tHL, uv).x) pc0 = cand;
    }
    float M = float(uM);
    vec2 pr = texelFetch(tS, ivec2(id.x - (j > 0 ? 1 : 0), id.y), 0).xy / rM, nx = texelFetch(tS, ivec2(id.x + (j < 15 ? 1 : 0), id.y), 0).xy / rM;
    vec2 dp = pr - pc0, dn = nx - pc0; float h = float(uN) * 0.5;
    if (length(dp) < h && length(dn) < h) {
      if (j > 0 && j < 15) pc0 += 0.15 * (0.5 * (dp + dn));
      else {
        vec2 out_ = j == 0 ? -normalize(dn + 1e-6) : -normalize(dp + 1e-6);
        vec2 ext = pc0 + out_ * uGrow * rideDt / uSp;
        vec4 ae = bilA(tA, ext);
        if ((1.0 - smoothstep(uJth - 0.3, uJth + 0.05, ae.w)) > 0.2 && lipHere > 0.45) pc0 = ext;
      }
    }
    float actMax = 0.35 + 0.45 * h21(vec2(cid, 7.7));
    bool keep = (uTime - p.z) < actMax && lipHere > 0.35;
    p.xy = mod(pc0 * rM, M);
    p.w = keep ? p.w : life;
    if (!keep) p.z = uTime;   // lace clock starts now (fades in, v30.14)
  } else {
    // Stays on the parcel it was released on. Orbital motion is the surface mesh.
    p.xy = mod(p.xy, float(uM));
  }
  o = p;
}`;
const STRAND_VS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D tS; uniform int uM; uniform float uTime, uSeg, uYoung;
uniform float uFine; uniform vec2 uCam; uniform float uWin; uniform float uGain, uFil, uTex, uDk, uFc;  // v30.14 camera-following fine layer (toroidal window uWin, M-grid units)
out vec2 vI;
float h21v(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  int seg = gl_VertexID / 2, e = gl_VertexID % 2;
  int chain = seg / 15, k = seg % 15;
  int i0 = chain * 16 + k;
  ivec2 a = ivec2(i0 % 256, i0 / 256), b = ivec2((i0 + 1) % 256, (i0 + 1) / 256);
  vec4 pa = texelFetch(tS, a, 0), pb = texelFetch(tS, b, 0);
  vec2 d = pb.xy - pa.xy; float M = float(uM);
  float age = uTime - pa.z;
  bool actA = pa.w > 1000.0; float lifeA = actA ? pa.w - 1000.0 : pa.w; bool actB = pb.w > 1000.0;
  bool ok = lifeA > 0.0 && age < lifeA && age >= 0.0 && (length(d) < uSeg || (actA && actB && length(d) < 4.0 * uSeg)) && length(d) < 0.5 * M;
  vec2 q = e == 0 ? pa.xy : pa.xy + d;
  vec2 clip = q / M * 2.0 - 1.0;
  if (uFine > 0.5) {
    vec2 da = pa.xy - uCam; da -= M * floor(da / M + 0.5);            // wrapped offset from the camera
    vec2 ua = fract((uCam + da) / uWin), ub = ua + d / uWin;
    ok = ok && max(abs(da.x), abs(da.y)) < 0.47 * uWin && all(greaterThanEqual(ub, vec2(0.0))) && all(lessThan(ub, vec2(1.0)));
    clip = (e == 0 ? ua : ub) * 2.0 - 1.0;
    // v30.14 fine detail at the window's own 0.25 m resolution: each strand frays into uFil filaments.
    // Filament j is seeded per chain (hash), offset sideways 0.2-1.2 m with a meander that varies along the
    // chain and drifts with the strand's age (it advects with the parent nodes, which ride the flow).
    if (gl_InstanceID > 0) {
      float j = float(gl_InstanceID), ag = uTime - pa.z;
      vec2 nrm = normalize(vec2(-d.y, d.x) + 1e-6);
      float hs = h21v(vec2(float(chain), j)), hp = h21v(vec2(j, float(chain) * 0.37));
      float off = (hs - 0.5) * 2.4 + 0.6 * sin(float(k + e) * (0.7 + hp) + 6.28 * hp + 0.4 * ag);
      vec2 o = nrm * off / uTex;                          // metres -> window texels
      float cut = h21v(vec2(float(seg), j));              // filaments break up: ~35% of segments missing
      ok = ok && cut > 0.35 && !actA && ag > 10.0; // v30.18.4: filaments only on lace older than ~10 s
      clip += o * 2.0 / 1024.0;
    }
  }
  // v31.1 ageing is geometric: stretched links fail sooner, and a stable per-segment
  // draw drops links at different times so a chain breaks into remnants, not a fade of the same line.
  { float ageU = clamp(age / max(lifeA, 0.5), 0.0, 1.0);
    float lim = mix(uSeg, 0.42 * uSeg, ageU);
    float drop = h21v(vec2(float(seg), 2.3));
    ok = ok && length(d) < (actA ? 4.0 * uSeg : lim) && (actA || ageU < 0.28 || drop > ageU * 0.62); }
  gl_Position = ok ? vec4(clip, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  // v30.14 no pops: lace fades in over 1.2 s after its chain stops riding, out over the last 40% of its
  // life; torn segments fade before they break; a riding chain's whitecap ramps in over 0.4 s and out
  // over the last 0.8 s of its ride.
  float fade = smoothstep(lifeA, 0.6 * lifeA, age) * smoothstep(0.0, 1.2, age) * (1.0 - smoothstep(0.6 * uSeg, uSeg, length(d)));
  float young = actA && actB ? 1.0 : 0.0;
  float actMax = 0.35 + 0.45 * h21v(vec2(float(chain), 7.7)); // v31.1 matches the short lip ride in STRAND_UPD
  float gy = smoothstep(0.0, 0.4, age) * (1.0 - smoothstep(actMax - 0.8, actMax, age));
  float thin = clamp(0.25 * uSeg / max(length(d), 1e-3), 0.3, 1.0); // stretched strands thin
  // v30.18.2 three-phase life after release (Michael's drone video): DENSE white 4-6 s (drawn into G, the
  // whitecap channel), then MOTTLED bubbles over 5 s (opacity 0.6 -> 0.3), then LACE 0.3 -> 0 over the rest of life.
  float dD = max(0.8, uDk * min(4.0 + 2.0 * h21v(vec2(float(chain), 3.1)), max(1.2, 0.2 * lifeA))); // v30.19: small breaks: short dense phase; v30.22f x uDk (wind)
  float dense2 = smoothstep(0.0, 0.3, age) * (1.0 - smoothstep(dD - 1.0, dD, age));
  float op = mix(0.6, 0.3, smoothstep(dD, dD + 5.0, age)) * (1.0 - smoothstep(dD + 5.0, lifeA, age)) * smoothstep(dD - 1.0, dD, age);
  op *= 1.0 - smoothstep(0.25 * uSeg, 0.4 * uSeg, length(d)); // v30.18.4: stretched past ~1.5x rest -> dissolve (no spokes)
  // v31.1: the short lip ride must not delete the foam. For the dense seconds after release the
  // line still feeds the white channel, but the node is no longer chasing the crest.
  vI = young > 0.5 ? vec2(0.0, gy) : uFc > 0.5 ? vec2((op + 0.5 * dense2) * thin, 0.5 * dense2 * thin) : vec2(op * thin, dense2 * 0.12);
  if (gl_InstanceID > 0) vI = vec2(0.55 * vI.x + 0.25 * vI.y, 0.0);   // filaments are lace (aerated streaks), not whitecap
  vI *= uGain;
}`;
// v30.18.4 soft splats: every node is also drawn as a soft round bubble cloud (gaussian disc, ~0.6-2 m),
// so whitecaps and young foam have pillowy edges; old lace keeps the thin lines.
const STRAND_PT_VS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D tS; uniform int uM; uniform float uTime, uTex, uGk, uFine, uWin, uWk, uDk, uFc; uniform vec2 uCam;
out vec2 vI; out vec2 vW; out float vAge; out float vPs;
float h21v(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  int i = gl_VertexID, chain = i / 16; vec4 p = texelFetch(tS, ivec2(i % 256, i / 256), 0); float M = float(uM);
  bool act = p.w > 1000.0; float life = act ? p.w - 1000.0 : p.w; float age = uTime - p.z;
  float hv = h21v(vec2(float(i), 5.3));
  float dD = max(0.8, uDk * min(4.0 + 2.0 * h21v(vec2(float(chain), 3.1)), max(1.2, 0.2 * life))); // v30.19: small breaks: short dense phase; v30.22f x uDk
  float g = 0.0, r = 0.0, rad = 0.0;
  if (life > 0.0 && age >= 0.0 && age < life) {
    if (act) { g = smoothstep(0.0, 0.3, age); rad = 1.2; }
    else { float dn = 1.0 - smoothstep(dD - 1.0, dD, age), mo = smoothstep(dD - 1.0, dD, age) * (1.0 - smoothstep(dD + 3.0, dD + 6.0, age));
      g = dn; r = 0.45 * mo * (0.5 + hv); rad = mix(1.3, 2.0, smoothstep(0.0, dD + 5.0, age)); // spreads as it ages
      // v30.22g: released foam is no longer drawn as whitecap (G: a solid white disc for the whole dense phase,
      // the medium-range 'pale patches'): G only for ~0.5 s after release, the dense phase goes to the lace (R)
      if (uFc > 0.5) { r += 0.35 * dn * (0.5 + hv); g = dn * (1.0 - smoothstep(0.2, 0.6, age)); } }
  }
  rad *= (0.6 + 0.8 * hv) * clamp(life / 25.0, 0.4, 1.2) * (1.0 + 1.6 * uWk * uWk); // v30.19.1: whitecaps widen with wind (Monahan coverage rises ~U^3.4) // v30.19: patch width follows the break energy (life)
  vec2 q = p.xy / M; bool ok = rad > 0.0;
  if (uFine > 0.5) { vec2 da = p.xy - uCam; da -= M * floor(da / M + 0.5); q = fract((uCam + da) / uWin); ok = ok && max(abs(da.x), abs(da.y)) < 0.47 * uWin; }
  gl_Position = ok ? vec4(q * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = clamp(2.0 * rad / uTex, 1.0, 24.0);
  vW = p.xy; vAge = ok ? age : 0.0; vPs = gl_PointSize;
  vI = vec2(r, g * uGk * 2.5) * min(1.0, 2.0 * rad / uTex);
}`;
// v30.23 buoy foam trail (Michael t214s1054; GAD Nov 1): splats laid along a fast buoy drag, drawn into the
// same strand foam layers (G young/dense, R lace) as the breaking foam. Dense for ~2.2 s, then it falls back
// to residual lace. Each trail point gets 3 jittered round splats across its width (no straight edges).
// v30.23 trail deposit gain into young foam G (crest splats use 2.5); ?trailg= for tuning against the luma cap
const TRAIL_GAIN = Number(new URLSearchParams(globalThis.location ? location.search : "").get("trailg")) || 2.5;
const TRAIL_VS = `#version 300 es
precision highp float; precision highp int;
uniform vec4 uTr[64]; uniform vec2 uTrN[64]; uniform int uM; uniform float uL, uTex, uGk, uFine, uWin, uTrG; uniform vec2 uCam;
out vec2 vI; out vec2 vW; out float vAge; out float vPs;
float h21t(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  int i = gl_VertexID / 3, j = gl_VertexID % 3; vec4 t = uTr[i]; float M = float(uM);
  float hv = h21t(vec2(float(i), float(j) + 0.37)), hw = h21t(vec2(float(j), float(i) * 1.7));
  float age = t.z, w = t.w;
  // uTrN length is the water-relative disturbance (about 0.55-1), not a wider brush.
  vec2 rawN = uTrN[i]; float disturb = clamp(length(rawN), 0.0, 1.0);
  vec2 nrm = rawN / max(disturb, 1e-3);
  float lat = (float(j) - 1.0) * 0.33 + (hv - 0.5) * 0.25;
  vec2 xz = t.xy + nrm * lat * w + vec2(hw - 0.5, hv - 0.5) * 0.2 * w;
  // Peak brightness stays in a narrow band. Extra speed does not paint a whiter core;
  // the outer sample becomes a detached fleck instead of more white.
  float dense = (1.0 - smoothstep(1.2, 2.2, age)) * smoothstep(0.0, 0.12, age);
  // Same dense 1.2-2.2 s and lace 5-11 s envelope as v31. lifeK 0.55 ended the
  // early half of the samples by ~6 s and dropped the visible patch. A one-second
  // stagger at the end keeps remnants from dying together without clearing the body.
  float lace = smoothstep(1.4, 2.2, age) * (1.0 - smoothstep(5.0, 11.0, age)) * (0.35 + 0.3 * hw);
  if (j == 2) { xz += nrm * mix(0.05, 0.4, disturb) * w * (hv - 0.35); dense *= 1.0 - 0.8 * disturb; lace *= mix(0.35, 1.0, disturb); }
  float lifeK = mix(0.9, 1.1, hv);
  lace *= 1.0 - smoothstep(8.0, 11.0, age) * (lifeK - 0.9) * 5.0;
  float rad = w * (0.26 + 0.12 * hv) * (1.0 + 0.25 * smoothstep(0.0, 2.2, age));
  vec2 p = fract(xz / uL + 0.5) * M; vec2 q = p / M; bool ok = w > 0.0 && (dense + lace) > 0.002;
  if (uFine > 0.5) { vec2 da = p - uCam; da -= M * floor(da / M + 0.5); q = fract((uCam + da) / uWin); ok = ok && max(abs(da.x), abs(da.y)) < 0.47 * uWin; }
  gl_Position = ok ? vec4(q * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = clamp(2.0 * rad / uTex, 1.0, 64.0);
  vW = p; vAge = age; vPs = gl_PointSize;
  vI = vec2(lace, dense * uTrG * uGk) * min(1.0, 2.0 * rad / uTex);
}`;
const STRAND_PT_FS = `#version 300 es
precision highp float;
uniform float uCell; // M-grid units per pixel of this foam map
in vec2 vI; in vec2 vW; in float vAge; in float vPs; out vec4 o;
float hCell(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vCell(vec2 p) { vec2 i = floor(p), f = p - i; f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hCell(i), hCell(i + vec2(1.0, 0.0)), f.x), mix(hCell(i + vec2(0.0, 1.0)), hCell(i + vec2(1.0, 1.0)), f.x), f.y); }
void main() {
  float r = 2.0 * length(gl_PointCoord - 0.5);
  float disk = exp(-3.0 * r * r) * (1.0 - smoothstep(0.85, 1.0, r));
  // Holes are locked to the foam map, so they stay put and widen as the sample ages.
  vec2 w = vW + (gl_PointCoord - 0.5) * vPs * uCell;
  // Open after the dense phase, slowly. The old ramp (from 1.2 s over 13 s, threshold
  // up to 0.58) had removed most of a splat by 6 s, and the water shader then dropped
  // the blurred remainder. Holes are visible by ~8 s without clearing the patch at 6 s.
  float a = clamp((vAge - 3.0) / 18.0, 0.0, 1.0);
  // One value-noise. The second pit field doubled the sin count on every splat fragment;
  // that pass is the Mac frame-tail cost (same 4096-chain draw A already budgeted).
  float holes = vCell(w / mix(10.0, 6.0, a));
  float keep = a <= 0.0 ? 1.0 : smoothstep(mix(0.02, 0.32, a), mix(0.18, 0.62, a), holes);
  float edge = mix(0.08, 0.22, a); // older, thinner foam is softer; fresh stays crisper
  disk *= smoothstep(0.55 - edge, 0.55 + edge, keep);
  o = vec4(vI * disk, 0.0, 1.0);
}`;
const STRAND_FS = `#version 300 es
precision highp float;
in vec2 vI; out vec4 o;
void main() { o = vec4(vI, 0.0, 1.0); }`;

// v30.4 boat buoyancy: the sea height at up to 16 world points (bilinear on tA, the FFT tile wraps).
const PROBE_FS = HEADER + `
uniform sampler2D tA;
uniform vec2 uPts[16];
uniform vec2 uPts2[16]; // v30.21: on-sea UI points (buoy)
uniform vec4 uSec;      // cross-section line x0, z0, x1, z1 (v30.22b: texels 20..83, 4 samples each)
uniform sampler2D tG;   // v30.21: break flag (.w) for the section
uniform float uTile;
out vec4 o;
vec4 aAt(ivec2 c) { return texelFetch(tA, (c % uN + uN) % uN, 0); }
void main() {
  int k = int(gl_FragCoord.x);
  float sp = uTile / float(uN);
  if (k < 20) {
    vec2 pt = k < 16 ? uPts[k] : uPts2[k - 16];
    vec2 u = (pt + 0.5 * uTile) / sp; vec2 i0 = floor(u); vec2 f = u - i0; ivec2 c = ivec2(i0);
    o = mix(mix(aAt(c), aAt(c + ivec2(1, 0)), f.x), mix(aAt(c + ivec2(0, 1)), aAt(c + ivec2(1, 1)), f.x), f.y);
    return;
  }
  // v31.8: four domain edges, 32 heights each, packed 4 per texel in texels 84..115.
  // The cell is clamped, matching the block edge (a bilinear read at ±L/2 wraps).
  if (k >= 84) {
    int local = k - 84;
    int edge = local / 8;
    int s0 = (local - edge * 8) * 4;
    float halfT = 0.5 * uTile;
    vec4 rE = vec4(0.0);
    for (int j = 0; j < 4; j += 1) {
      float t = float(s0 + j) / 31.0;
      vec2 pt = vec2(0.0);
      if (edge == 0) pt = vec2(mix(-halfT, halfT, t), halfT);
      else if (edge == 1) pt = vec2(mix(halfT, -halfT, t), -halfT);
      else if (edge == 2) pt = vec2(-halfT, mix(halfT, -halfT, t));
      else pt = vec2(halfT, mix(-halfT, halfT, t));
      ivec2 cell = ivec2(floor((pt + 0.5 * uTile) / sp + 0.5));
      cell = clamp(cell, ivec2(0), ivec2(uN - 1));
      rE[j] = texelFetch(tA, cell, 0).x;
    }
    o = rE;
    return;
  }
  // v30.22b: the 256-sample cross-section packed 4 per texel (64 texels = 1 KB): each channel is the
  // height + 1000 where the break flag is set.
  vec4 r = vec4(0.0);
  for (int j = 0; j < 4; j += 1) {
    int i = (k - 20) * 4 + j;
    vec2 pt = mix(uSec.xy, uSec.zw, float(i) / 255.0);
    vec2 u = (pt + 0.5 * uTile) / sp; vec2 i0 = floor(u); vec2 f = u - i0; ivec2 c = ivec2(i0);
    float h = mix(mix(aAt(c).x, aAt(c + ivec2(1, 0)).x, f.x), mix(aAt(c + ivec2(0, 1)).x, aAt(c + ivec2(1, 1)).x, f.x), f.y);
    ivec2 q = ivec2(floor(u + 0.5));
    float fl = texelFetch(tG, (q % uN + uN) % uN, 0).w > 0.0 ? 1000.0 : 0.0;
    r[j] = h + fl;
  }
  o = r;
}`;

// v27.2: is any join record (tJ.w) within JOIN_REACH cells (Chebyshev) of this texel?
// Separable max: along x (uAxis 0, from tJ.w) then along z (uAxis 1, from the x pass).
// Paint runs its joined loop only where it is: every join offset that loop tries is at
// most 3 + 1 + 1 + 4 = 9 cells per axis, so the paint is the same, without 360 fetches
// on the texels no join can reach.
const JOIN_REACH = 6; // joinpaint reads the mask at the gap cell g: its joins are within 1 + 1 + 4 cells
const JMASK_FS = HEADER + `
uniform sampler2D tJ;
uniform int uAxis;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 d = uAxis == 0 ? ivec2(1, 0) : ivec2(0, 1);
  float hit = 0.0;
  for (int t = -${JOIN_REACH}; t <= ${JOIN_REACH}; t += 1) {
    ivec2 q = p + d * t;
    if (!inside(q)) continue;
    vec4 v = texelFetch(tJ, q, 0);
    if ((uAxis == 0 ? v.w : v.x) >= 0.5) hit = 1.0;
  }
  o = vec4(hit, 0.0, 0.0, 0.0);
}`;

// v27.2: per gap cell g, for k = 0..3, the smallest paint key among the joins whose gap
// cell is g with k face steps from g still near the join and dropping (PAINT_FS's former
// joined loop with g fixed). 1e30 = none. Only where a join is within JOIN_REACH (tJM).
const JOINPAINT_FS = HEADER + OFFSET_GLSL + `
uniform sampler2D tC;
uniform sampler2D tJ;
uniform sampler2D tJM;
uniform int uJoinSteps;
out vec4 o;
float BH(ivec2 q) { return texelFetch(tC, q, 0).z; }
bool nearTo(ivec2 q, ivec2 a, ivec2 b) {
  ivec2 da = q - a;
  ivec2 db = q - b;
  return da.x * da.x + da.y * da.y <= 16 || db.x * db.x + db.y * db.y <= 16;
}
void main() {
  ivec2 g = ivec2(gl_FragCoord.xy);
  vec4 best = vec4(1e30);
  if (texelFetch(tJM, g, 0).x > 0.5) {
    float idxScale = 32.0;
    float joinBase = float(uN * uN) * idxScale;
    for (int s = 2; s <= 5; s += 1) {
      for (int j = 1; j < 5; j += 1) {
        if (j >= s) break;
        for (int w = -1; w <= 1; w += 1) {
          for (int m = -1; m <= 1; m += 1) {
            ivec2 b = g - offT(m) - uOffF[pairIndex(s, j) * 3 + w + 1] - offC(j);
            if (!inside(b)) continue;
            vec4 rec = texelFetch(tJ, b, 0);
            if (rec.w < 0.5 || int(rec.x) != s || int(rec.y) != w) continue;
            int snaps = int(rec.z + 0.5);
            int div = 1;
            for (int e = 1; e < 5; e += 1) { if (e >= j) break; div *= 3; }
            if ((snaps / div) % 3 - 1 != m) continue;
            ivec2 partner = b + offC(s) + offT(w);
            if (!nearTo(g, b, partner)) continue;
            float base = joinBase + float(b.y * uN + b.x) * idxScale + float(j * 4);
            bool ok = true;
            float prev = BH(g);
            for (int k = 0; k <= 3; k += 1) {
              if (k > uJoinSteps) break;
              if (k > 0) {
                ivec2 f = g + offT(k);
                if (!inside(f) || !nearTo(f, b, partner)) ok = false;
                else { float h = BH(f); if (h >= prev) ok = false; prev = h; }
              }
              if (!ok) break;
              best[k] = min(best[k], base + float(k));
            }
          }
        }
      }
    }
  }
  o = best;
}`;

// v30.22c curl field (Michael: "realistic curls"). Four small passes on the physics grid, after the
// break detector (physL, whose .w carries the break's onset age at crests):
//   curlSrc   crest cells (height max along the travel grid axis A) -> (1, curling, curling*phase, curling*h)
//   curlJoin  each crest cell walks along its OWN crest line (next column across travel, crest found
//             within +-2 cells) for CURL_JR steps each way and Gaussian-averages the curl share, phase
//             and crest height: one breaker = one long lip that tapers at both ends, no per-cell segments.
//             Out: crest cells (amp, phase, lip height Hc, sub-cell crest offset), others (-1, 0, 0, 0).
//   curlField crest cells as above; front-face cells find their crest upwind and store the cavity under
//             the lip (-1 - cav) and the landing foam source (.y) where the lip comes down.
//   curlList  per column across travel, the first CURL_SLOTS curling crest cells (the lip draw's instances).
export const CURL_SLOTS = 12;
// v30.22e curl size boost: true size (1) up close, rising smoothly to 2 (<= 20 m/s) .. 3 (25 m/s) at the
// medium camera (~250 m), so lips still read from there. ?curlx= fixes it (no camera dependence).
export function curlX(cq, U) {
  if (cq.has("curlx")) return Number(cq.get("curlx"));
  const far = 2.0 + Math.min(1, Math.max(0, (U - 20) / 5));
  const d = Number(globalThis.__oceanCamDistFn ? globalThis.__oceanCamDistFn() : globalThis.__oceanCamDist) || 250, t = Math.min(1, Math.max(0, (d - 70) / 160));
  return 1 + (far - 1) * t * t * (3 - 2 * t);
}
export const CURL_SHAPE_GLSL = `
// lip profile (forward, up) from the crest top, s 0 root -> 1 tip; ph = time since onset / T
float curlEnv(float ph) { return smoothstep(0.0, 0.12, ph) * (1.0 - smoothstep(1.1, 1.5, ph)); }
// tg (0..1): how much bigger the lip is than the drop its wave allows (v30.22e): a tighter, rounder curl
// (smaller radius, more fold, much shorter forward throw) instead of a flat plank
vec2 curlAtT(float s, float ph, float Hc, float amp, float tg) {
  float am = smoothstep(0.0, 0.45, amp); // shape is full over most of the breaker; only its ends shrink
  float R = 0.5 * Hc * mix(0.55, 1.0, am) * mix(1.0, 0.8, tg);
  float th = mix(0.5, 2.7, smoothstep(0.0, 0.85, ph)) * mix(0.6, 1.0, am) * (1.0 + 0.3 * tg);
  float a0 = 0.3, phi = s * th;
  float thr = 0.6 * Hc * smoothstep(0.1, 0.8, ph) * am * (1.0 - 0.55 * tg);
  vec2 q = vec2(mix(1.3, 1.0, tg) * R * (sin(a0) - sin(a0 - phi)) + thr * s, R * (cos(a0 - phi) - cos(a0)));
  q.y -= smoothstep(0.85, 1.4, ph) * s * s * 0.45 * Hc; // collapse: the lip slumps onto the face
  return q;
}
vec2 curlAt(float s, float ph, float Hc, float amp) { return curlAtT(s, ph, Hc, amp, 0.0); }
float curlTight(float cx, float ph) { return smoothstep(1.2, 2.6, cx) * smoothstep(0.15, 0.5, ph); } // boosted lips curl tighter
vec2 curlDsT(float s, float ph, float Hc, float amp, float tg) { // d/ds of curlAtT (numerical, smooth)
  return (curlAtT(min(s + 0.01, 1.0), ph, Hc, amp, tg) - curlAtT(max(s - 0.01, 0.0), ph, Hc, amp, tg)) / (min(s + 0.01, 1.0) - max(s - 0.01, 0.0));
}
vec2 curlDs(float s, float ph, float Hc, float amp) { return curlDsT(s, ph, Hc, amp, 0.0); }
`;
const CURL_COMMON = `
uniform ivec2 uAx;   // one grid step downwind along the travel axis
uniform ivec2 uBx;   // one grid step across (the other axis)
vec4 tf(sampler2D t, ivec2 q) { return texelFetch(t, (q % uN + uN) % uN, 0); }
`;
// per-crest curl clock (seconds since this crest's break onset), carried along with the moving crest:
// a crest cell takes over the record of the nearest crest cell up to 1 cell ahead / 3 cells behind it in
// the last step's state (the crest travels downwind). It starts at 0 when the crest starts breaking,
// keeps counting while it breaks (so the curl plays ONCE per onset) and is cleared after 0.6 s without
// breaking, so a later, separate break plays again. Out: (crest 1/0, clock or -1, quiet time, 0).
const CURL_CLOCK_FS = HEADER + CURL_COMMON + `
uniform sampler2D tA; uniform sampler2D tL; uniform sampler2D tT; uniform float uDt;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float h = tf(tA, p).x;
  o = vec4(0.0, -1.0, 0.0, 0.0);
  if (!(h > 0.0 && h >= tf(tA, p + uAx).x && h > tf(tA, p - uAx).x && h >= tf(tA, p + 2 * uAx).x && h >= tf(tA, p - 2 * uAx).x)) return;
  bool brk = false;
  for (int k = -3; k <= 3; k += 1) if (tf(tL, p + k * uAx).x > 0.0) brk = true;
  vec4 prev = vec4(0.0); bool found = false;
  for (int d = 0; d < 5; d += 1) {
    int k = d == 0 ? 0 : d == 1 ? 1 : d == 2 ? -1 : d == 3 ? 2 : 3; // behind (upwind) first
    vec4 t = tf(tT, p - k * uAx);
    if (t.x > 0.5) { prev = t; found = true; break; }
  }
  float clock = -1.0, quiet = 0.0;
  if (found && prev.y >= 0.0) {
    quiet = brk ? 0.0 : prev.z + uDt;
    clock = quiet > 0.6 ? -1.0 : prev.y + uDt;
  } else if (brk) clock = 0.0;
  o = vec4(1.0, clock, quiet, 0.0);
}`;
const CURL_PASSES = FQ.get("curl") !== "0"; // v30.22g perf: ?curl=0 skips the curl passes
let curlIdle = false;
const CURL_SRC_FS = HEADER + CURL_COMMON + `
uniform sampler2D tA; uniform sampler2D tT;
uniform float uU; uniform float uSp; uniform float uK0; uniform float uS; uniform float uX;
out vec4 o;
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 x) { vec2 i = floor(x), f = x - i; f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float h = tf(tA, p).x, hm = tf(tA, p - uAx).x, hp = tf(tA, p + uAx).x;
  o = vec4(0.0);
  if (!(h > 0.0 && h >= hp && h > hm && h >= tf(tA, p + 2 * uAx).x && h >= tf(tA, p - 2 * uAx).x)) return;
  float den = hm - 2.0 * h + hp;
  float sub = den < -1e-6 ? clamp(0.5 * (hm - hp) / den, -0.5, 0.5) : 0.0;
  float hpk = h - 0.25 * (hm - hp) * sub;
  float age = tf(tT, p).y; // this crest's curl clock (CURL_CLOCK_FS)
  float gateU = smoothstep(11.5, 13.0, uU);
  // crest-height bar (m), box-tuned at size 250 (v30.22b): only the top ~10-25% of breaking crests
  float H0 = uK0 / 0.22 * mix(mix(1.8, 2.4, smoothstep(12.0, 15.0, uU)), 0.35, smoothstep(15.0, 25.0, uU)); // v30.22c: 25 m/s bar 0.45 -> 0.35 m; 15 m/s bar 1.6 -> 2.4 m (a few curls, not ~15)
  float big = smoothstep(H0, 1.3 * H0, hpk) * gateU;
  float cur = age >= 0.0 ? big : 0.0;
  float T = 1.1 + 0.6 * vn(vec2(p) * uSp / 45.0); // each breaker its own pace (1.1-1.7 s)
  // the lip fits its own wave: it drops at most to the trough ahead and lands before that trough
  // (profile: drop ~0.81 Hc, reach ~1.28 Hc), else at 25 m/s it plunged into the next wave
  float hT = hpk, dT = 0.0;
  for (int k = 1; k <= 24; k += 1) { float hk = tf(tA, p + k * uAx).x; if (hk > hT + 1e-4) break; hT = hk; dT = float(k) * uSp; }
  float Hc = min(2.0 * uS * hpk, min(1.3 * (hpk - hT) / 0.81, 1.4 * dT / 1.28)); // it13/14: the tip is pulled onto the water ahead (lip sheet), so the drop cap is looser
  o = vec4(1.0, cur, cur * age / T, cur * Hc * uX); // uX (?curlx=, default 1): optional exaggeration past the wave cap
}`;
const CURL_JOIN_FS = HEADER + CURL_COMMON + CURL_SHAPE_GLSL + `
uniform sampler2D tC; uniform sampler2D tA;
uniform float uSig; uniform int uR; uniform float uS; uniform int uMinLen;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 s0 = tf(tC, p);
  if (s0.x < 0.5) { o = vec4(-1.0, 0.0, 0.0, 0.0); return; }
  float W = 1.0, Wc = s0.y, Wp = s0.z, Wh = s0.w;
  float dEnd = 99.0; // steps to the nearer end of this curling stretch (the lip tapers to 0 there)
  if (s0.y < 0.25) dEnd = 0.0;
  int len = 1; // this breaker's length in columns (to uR each side): short pieces are not drawn (no chips)
  for (int sd = -1; sd <= 1; sd += 2) {
    ivec2 q = p; bool alive = true; bool run = s0.y >= 0.25; int miss = 0, low = 0;
    for (int j = 1; j <= 16; j += 1) {
      if (j > uR) break;
      float w = exp(-0.5 * float(j * j) / (uSig * uSig));
      W += w;
      if (!alive) continue;
      ivec2 b = q + sd * uBx; bool found = false; vec4 s = vec4(0.0);
      for (int d = 0; d < 7; d += 1) {
        int dd = (d + 1) / 2 * (d % 2 == 1 ? 1 : -1);
        ivec2 t = b + dd * uAx; vec4 st = tf(tC, t);
        if (st.x > 0.5) { q = t; s = st; found = true; break; }
      }
      // one missing column or one weak crest cell inside a breaker does not split it (it2 lobes)
      if (!found) { miss += 1; q = b; if (miss >= 2) { alive = false; if (run) { dEnd = min(dEnd, float(j) - 2.0); run = false; } } continue; }
      miss = 0;
      if (run) { low = s.y < 0.25 ? low + 1 : 0; if (low >= 2) { dEnd = min(dEnd, float(j) - 2.0); run = false; } }
      if (run) len += 1;
      Wc += w * s.y; Wp += w * s.z; Wh += w * s.w;
    }
  }
  float frac = Wc / W;
  float ph = Wc > 1e-4 ? Wp / Wc : 0.0, hk = Wc > 1e-4 ? Wh / Wc : 0.0;
  float amp = smoothstep(0.3, 0.7, frac) * curlEnv(ph) * smoothstep(0.0, uSig, max(dEnd, 0.0) + 0.5);
  if (len < uMinLen) amp = 0.0;
  float h = tf(tA, p).x, hm = tf(tA, p - uAx).x, hp = tf(tA, p + uAx).x;
  float den = hm - 2.0 * h + hp;
  float sub = den < -1e-6 ? clamp(0.5 * (hm - hp) / den, -0.5, 0.5) : 0.0;
  o = vec4(amp, ph, hk, sub);
}`;
// second, short smoothing along the same crest line (sigma ~1.5 cells) of the joined amp/phase/height,
// so neighbouring columns never step (no saw-tooth on the lip edge)
const CURL_SMOOTH_FS = HEADER + CURL_COMMON + `
uniform sampler2D tC; uniform sampler2D tJ; uniform float uSig;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 j0 = tf(tJ, p);
  if (j0.x < 0.0) { o = j0; return; }
  float W = 1.0, Wa = j0.x, Wq = j0.x + 1e-4, Wp = j0.y * (j0.x + 1e-4), Wh = j0.z * (j0.x + 1e-4);
  float Wo = 1.0, Oo = j0.w; // it18: the crest line's position along travel, smoothed (no saw-tooth where the crest steps a cell)
  for (int sd = -1; sd <= 1; sd += 2) {
    ivec2 q = p; bool alive = true;
    for (int j = 1; j <= 7; j += 1) {
      float w = exp(-0.5 * float(j * j) / (uSig * uSig));
      W += w;
      if (!alive) continue;
      ivec2 b = q + sd * uBx; bool found = false;
      for (int d = 0; d < 7; d += 1) {
        int dd = (d + 1) / 2 * (d % 2 == 1 ? 1 : -1);
        ivec2 t = b + dd * uAx;
        if (tf(tC, t).x > 0.5) { q = t; found = true; break; }
      }
      if (!found) { q = b; alive = false; continue; }
      vec4 jj = tf(tJ, q);
      float a = max(jj.x, 0.0), wa = w * (a + 1e-4);
      Wa += w * a; Wq += wa; Wp += wa * jj.y; Wh += wa * jj.z;
      ivec2 dq = q - p; Wo += w; Oo += w * (float(dq.x * uAx.x + dq.y * uAx.y) + jj.w);
    }
  }
  o = vec4(Wa / W, Wp / Wq, Wh / Wq, clamp(Oo / Wo, -2.0, 2.0));
}`;
const CURL_FIELD_FS = HEADER + CURL_COMMON + CURL_SHAPE_GLSL + `
uniform sampler2D tJ; uniform float uSp; uniform int uK; uniform float uX;
uniform sampler2D tC0; uniform int uAxisX; // v30.22g: column scan (no curling crest in this column -> nothing to search)
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = tf(tJ, p);
  if (c.x >= 0.0) { o = c; return; }
  o = vec4(-1.0, 0.0, 0.0, 0.0);
  if (texelFetch(tC0, ivec2(uAxisX == 1 ? p.y : p.x, 0), 0).x < 0.0) return;
  for (int k = 1; k <= 24; k += 1) {
    if (k > uK) break;
    vec4 j = tf(tJ, p - k * uAx);
    if (j.x < 0.0) continue;
    if (j.x > 0.01) {
      float dist = (float(k) + j.w) * uSp, ph = j.y, Hc = j.z;
      float xt = curlAtT(1.0, ph, Hc, j.x, curlTight(uX, ph)).x; // same (tightened) reach as the lip sheet
      float cav = j.x * smoothstep(0.1, 0.5, ph) * smoothstep(0.0, 0.2 * xt, dist) * (1.0 - smoothstep(0.7 * xt, 0.95 * xt, dist)); // it15: no shadow beyond the landing line (read as a sheet hanging over darker water)
      float land = smoothstep(0.6, 0.85, ph) * (1.0 - smoothstep(1.3, 1.6, ph)) * exp(-pow((dist - xt) / (0.35 * Hc + uSp), 2.0)) * min(1.0, 3.0 * j.x + 0.5);
      o = vec4(-1.0 - cav, land, 0.0, 0.0);
    }
    break;
  }
}`;
// v30.22g perf: one scan per crest column (N fragments) collects the first CURL_SLOTS curling crests into
// three RGBA rows (-1 = none); curlList and curlField read it instead of each scanning the whole column
// (curlList was CURL_SLOTS x N fragments x N texels; curlField walked up to 24 cells upwind in every column).
const CURL_COL_FS = HEADER + CURL_COMMON + `
uniform sampler2D tJ; uniform int uAxisX;
layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;
void main() {
  int u = int(gl_FragCoord.x); int cnt = 0; float s[12];
  for (int i = 0; i < 12; i++) s[i] = -1.0;
  for (int v = 0; v < 2048; v += 1) {
    if (v >= uN || cnt >= ${CURL_SLOTS}) break;
    ivec2 c = uAxisX == 1 ? ivec2(v, u) : ivec2(u, v);
    if (texelFetch(tJ, c, 0).x > 0.01) { s[cnt] = float(v); cnt += 1; }
  }
  o0 = vec4(s[0], s[1], s[2], s[3]); o1 = vec4(s[4], s[5], s[6], s[7]); o2 = vec4(s[8], s[9], s[10], s[11]);
}`;
const CURL_LIST_FS = HEADER + CURL_COMMON + `
uniform sampler2D tC0; uniform sampler2D tC1; uniform sampler2D tC2; uniform int uAxisX; // v30.22g: from the column scan (CURL_COL_FS)
out vec4 o;
void main() {
  int k = int(gl_FragCoord.x), u = int(gl_FragCoord.y);
  vec4 r = k < 4 ? texelFetch(tC0, ivec2(u, 0), 0) : k < 8 ? texelFetch(tC1, ivec2(u, 0), 0) : texelFetch(tC2, ivec2(u, 0), 0);
  o = vec4(r[k % 4]);
}`;

const REDUCE_FS = HEADER + `
uniform sampler2D tL;
uniform sampler2D tC;
uniform sampler2D tF;
uniform sampler2D tG;
layout(location = 0) out vec4 oCount; // breaking, flagged, carrying foam, lingering
layout(location = 1) out vec4 oMore;  // recent, max break slope
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * ${BLOCK};
  vec4 c = vec4(0.0);
  float recent = 0.0;
  float maxSlope = 0.0;
  float visible = 0.0; // v30.20: cells whose foam is visibly white (grade x fade >= FOAM_VIS)
  for (int dz = 0; dz < ${BLOCK}; dz += 1) {
    for (int dx = 0; dx < ${BLOCK}; dx += 1) {
      ivec2 q = base + ivec2(dx, dz);
      if (!inside(q)) continue;
      bool flagged = texelFetch(tG, q, 0).w > 0.0;
      float fw = texelFetch(tF, q, 0).w;
      bool foam = fw > 0.0;
      if (fw >= ${FOAM_VIS.toFixed(3)}) visible += 1.0;
      c += vec4(texelFetch(tL, q, 0).x > 0.0 ? 1.0 : 0.0, flagged ? 1.0 : 0.0, foam ? 1.0 : 0.0, foam && !flagged ? 1.0 : 0.0);
      if (texelFetch(tG, q, 0).z > 0.0) recent += 1.0;
      maxSlope = max(maxSlope, texelFetch(tC, q, 0).w);
    }
  }
  oCount = c;
  oMore = vec4(recent, maxSlope, visible, 0.0);
}`;

function compile(gl, type, source, name) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`GPU physics: ${name} shader failed: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

function makeProgram(gl, fs, name, vs) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs || QUAD_VS, name + " vertex"));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs, name));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`GPU physics: ${name} link failed: ${gl.getProgramInfoLog(program)}`);
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i += 1) {
    const info = gl.getActiveUniform(program, i);
    const base = info.name.replace(/\[0\]$/, "");
    uniforms[base] = gl.getUniformLocation(program, info.name);
  }
  return { program, uniforms };
}

// Why the card cannot run the physics, or null when it can.
export function gpuPhysicsProblem(gl) {
  if (!gl || typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) return "WebGL2 is not available";
  if (!gl.getExtension("EXT_color_buffer_float")) return "EXT_color_buffer_float is missing (float render targets)";
  if (gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) < 6) return "too few vertex texture units";
  if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 3) return "too few draw buffers";
  return fftSeaProblem(gl);
}

export function createGpuSurface(gl, scenario, wind) {
  const problem = gpuPhysicsProblem(gl);
  if (problem) throw new Error("GPU physics unavailable: " + problem);
  // The physics grid is the FFT tile: N x N over scenario.size (spacing size / N).
  const n = scenario.fft || 256;
  const M = Math.min(1024, n * Number(FQ.get("fgrid") || 2)); // v30.8 foam grid (finer than the physics grid)
  const sea = createFftSea(gl, n);
  let seaAcc = 0;
  const bands = scenario.bands.map(createBand);
  const patches = bands.map((band) => (band.spec.patch ? createGpuChopPatch(gl, band.spec) : null));
  const finePatches = bands.map((band) => (band.spec.patch && band.spec.finePatch ? createGpuChopPatch(gl, band.spec, band.spec.finePatch) : null));
  const maxC = Math.max(1, bands.reduce((sum, band) => sum + band.components.length, 0));
  const w0 = new Float32Array(maxC * 4);
  const w1 = new Float32Array(maxC * 4);
  const w2 = new Float32Array(maxC * 4);
  // The same explicit waves for the renderer, by world position (phase at the origin).
  const xw0 = new Float32Array(maxC * 4);
  const xw1 = new Float32Array(maxC * 4);

  const programs = {
    wave: makeProgram(gl, waveFS(maxC), "wave"),
    rowsum: makeProgram(gl, ROWSUM_FS, "rowsum"),
    local: makeProgram(gl, LOCAL_FS, "local"),
    join: makeProgram(gl, JOIN_FS, "join"),
    paint: makeProgram(gl, PAINT_FS, "paint"),
    age: makeProgram(gl, AGE_FS, "age"),
    cover: makeProgram(gl, COVER_FS, "cover"),
    probe: makeProgram(gl, PROBE_FS, "probe"),
    covstat: makeProgram(gl, COVSTAT_FS, "covstat"),
    longh: makeProgram(gl, LONGH_FS, "longh"),
    strandUpd: makeProgram(gl, STRAND_UPD_FS, "strandUpd"),
    strandDraw: makeProgram(gl, STRAND_FS, "strandDraw", STRAND_VS),
    strandPts: makeProgram(gl, STRAND_PT_FS, "strandPts", STRAND_PT_VS),
    foamTrail: makeProgram(gl, STRAND_PT_FS, "foamTrail", TRAIL_VS),
    strandDecay: makeProgram(gl, "#version 300 es\nprecision highp float; uniform float uK; out vec4 o; void main() { o = vec4(0.0, uK, 0.0, 1.0); }", "strandDecay"),
    jmask: makeProgram(gl, JMASK_FS, "jmask"),
    joinpaint: makeProgram(gl, JOINPAINT_FS, "joinpaint"),
    reduce: makeProgram(gl, REDUCE_FS, "reduce"),
    curlClock: makeProgram(gl, CURL_CLOCK_FS, "curlClock"),
    curlSrc: makeProgram(gl, CURL_SRC_FS, "curlSrc"),
    curlJoin: makeProgram(gl, CURL_JOIN_FS, "curlJoin"),
    curlSmooth: makeProgram(gl, CURL_SMOOTH_FS, "curlSmooth"),
    curlField: makeProgram(gl, CURL_FIELD_FS, "curlField"),
    curlList: makeProgram(gl, CURL_LIST_FS, "curlList"),
    curlCol: makeProgram(gl, CURL_COL_FS, "curlCol"),
  };
  for (const [k, v] of Object.entries(programs)) v.name = k;
  initPassTimer(gl);
  const vao = gl.createVertexArray();

  function texture(size, data, h = size) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, size, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (data) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, size, size, gl.RGBA, gl.FLOAT, data);
    return tex;
  }
  function layerTarget(t, layer) {
    const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, 0, layer); gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("GPU physics: strand layer target incomplete");
    return { fb };
  }
  function target(textures) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    textures.forEach((tex, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0));
    gl.drawBuffers(textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error("GPU physics: float render target incomplete (" + status + ")");
    return { fb, textures };
  }
  const start = new Float32Array(n * n * 4);
  for (let i = 0; i < n * n; i += 1) {
    start[i * 4] = -NEVER;
    start[i * 4 + 2] = NO_FOAM_AHEAD;
  }
  const tex = {
    A: texture(n), B: texture(n), C: texture(n), L: texture(n), J: texture(n), P: texture(n),
    S: [texture(n, start), texture(n, start)],
    F: texture(n), G: texture(n), X: texture(n),
    K: [texture(M), texture(M)],
    HL: (() => { const t = texture(64); gl.bindTexture(gl.TEXTURE_2D, t); gl.getExtension("OES_texture_float_linear");
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT); return t; })(),
    SP: [texture(256, undefined, STRAND_ROWS), texture(256, undefined, STRAND_ROWS)], // v30.19.2: 256 x (chains/16)
    // v30.14: one 1024^2 x 2 array (layer 0 = whole-tile strands, layer 1 = camera fine window) so the
    // water shader spends one sampler on strands (the Mac's 16-unit fragment limit).
    STR: (() => { const t = gl.createTexture(); const T = gl.TEXTURE_2D_ARRAY; gl.bindTexture(T, t);
      gl.texStorage3D(T, 11, gl.RGBA16F, 1024, 1024, 2);
      gl.texParameteri(T, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(T, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(T, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(T, gl.TEXTURE_WRAP_T, gl.REPEAT); gl.bindTexture(T, null); return t; })(),
  };
  // Until the first step: no foam (fade 1, ahead NO_FOAM_AHEAD), flat water.
  const g0 = new Float32Array(n * n * 4);
  for (let i = 0; i < n * n; i += 1) g0[i * 4] = NO_FOAM_AHEAD;
  gl.bindTexture(gl.TEXTURE_2D, tex.G);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n, n, gl.RGBA, gl.FLOAT, g0);
  // Foam ages (see AGE_TABLE): samples keep the step they were last flagged; the age of
  // "flagged k steps ago" comes from this table, which adds each step's dt in float32
  // exactly as surface.js does (ages[i] += dt into a Float32Array), so ages, fades and the
  // window's end match it bit for bit.
  const AGE_SIDE = 64;
  const ageTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, ageTex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, AGE_SIDE, AGE_SIDE);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  const ageData = new Float32Array(AGE_SIDE * AGE_SIDE);
  let ageLen = 0;
  function advanceAges(dt, window) {
    // New table: k = 0 is this step (age 0); entry k+1 is last step's entry k plus dt.
    let len = 1;
    const next = [0];
    for (let k = 0; k < ageLen && len < AGE_SIDE * AGE_SIDE; k += 1) {
      const v = Math.fround(ageData[k] + Math.max(dt, 0));
      if (!(v < window)) break; // ages only grow: from here on the window has passed
      next.push(v);
      len += 1;
    }
    ageData.fill(0);
    ageData.set(next);
    ageLen = len;
    gl.bindTexture(gl.TEXTURE_2D, ageTex);
    const rows = Math.ceil(len / AGE_SIDE);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, AGE_SIDE, rows, gl.RED, gl.FLOAT, ageData, 0);
  }
  const blocks = Math.ceil(n / BLOCK);
  const red = { count: texture(blocks), more: texture(blocks) };
  tex.JM = texture(n);
  tex.JX = texture(n);
  tex.HX = texture(n);
  tex.JP = texture(n);
  tex.CT = [texture(n), texture(n)]; tex.CS = texture(n); tex.CJ = texture(n); tex.CJ2 = texture(n); tex.CU = texture(n); tex.CL = texture(CURL_SLOTS, undefined, n); tex.CC = [texture(n, undefined, 1), texture(n, undefined, 1), texture(n, undefined, 1)]; // v30.22c curl field; v30.22g column scan
  const fbs = {
    wave: target([tex.A, tex.B, tex.C, tex.X]),
    rowsum: target([tex.HX]),
    local: target([tex.L]),
    join: target([tex.J]),
    jmaskX: target([tex.JX]),
    jmask: target([tex.JM]),
    joinpaint: target([tex.JP]),
    paint: target([tex.P]),
    age: [target([tex.S[1], tex.F, tex.G]), target([tex.S[0], tex.F, tex.G])],
    reduce: target([red.count, red.more]),
    cover: [target([tex.K[1]]), target([tex.K[0]])],
    longh: target([tex.HL]),
    strandUpd: [target([tex.SP[1]]), target([tex.SP[0]])],
    strandDraw: layerTarget(tex.STR, 0),
    strandFine: layerTarget(tex.STR, 1),
    curlClock: [target([tex.CT[1]]), target([tex.CT[0]])], curlSrc: target([tex.CS]), curlJoin: target([tex.CJ]), curlSmooth: target([tex.CJ2]), curlField: target([tex.CU]), curlList: target([tex.CL]), curlCol: target(tex.CC),
  };
  let kCur = 0;
  const PROBE_W = 116; // v31.8: 84 as before, plus 32 texels (4 edges x 32 heights)
  // v30.4 boat probe: 16x1 target, read back asynchronously (one read in flight, never waits)
  const probeTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, probeTex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, PROBE_W, 1);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  const probeFb = target([probeTex]);
  const probePack = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePack); gl.bufferData(gl.PIXEL_PACK_BUFFER, PROBE_W * 4 * 4, gl.STREAM_READ); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  // v30.7 coverage controller state
  const covTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, covTex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 16, 16);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  const covFb = target([covTex]);
  const covPack = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, covPack); gl.bufferData(gl.PIXEL_PACK_BUFFER, 256 * 16, gl.STREAM_READ); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  let covFence = null, covT = 0, covLastT = 0; const cov = { gain: Number(FQ.get("fgain") || 2.0), active: 0, visible: 0, target: 0 };
  const probeBufs = [new Float32Array(PROBE_W * 4), new Float32Array(PROBE_W * 4)]; let probeBi = 0; const covBuf = new Float32Array(1024); // v30.22g perf: reused readback arrays (no per-frame garbage)
  let probeFence = null; let probePts = null; let uiPts = new Float32Array(32); let uiSec = [0, 0, 0, 0]; let uiOn = false; const probeOut = { data: null, time: -1 }; let probeTime = 0;
  let current = 0; // tex.S[current] holds the latest foam state
  let curlAxis = { x: 1, ax: [1, 0], bx: [0, 1] }; let ctCur = 0;

  // Panel counts (v27.1): one pixel-pack buffer, one readback in flight at a time.
  // requestCounts() queues readPixels into the buffer (no CPU wait) and drops a fence;
  // poll() asks clientWaitSync(fence, 0, 0) once per frame (never waits) and only when the
  // fence has signalled calls getBufferSubData. While a read is in flight, newer steps are
  // not read; the newest step is asked for as soon as the buffer is free (if the reduce
  // target still holds it). So the panel trails the sea by >= 1 step; every state carries
  // countsStep (the step its counts came from) next to step.
  const readSize = blocks * blocks * 4 * 4;
  const pack = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pack);
  gl.bufferData(gl.PIXEL_PACK_BUFFER, readSize * 2, gl.STREAM_READ);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  const inFlight = { sync: null, step: -1 };
  const readback = new Float32Array(blocks * blocks * 8);
  const counts = { step: -1, breakingCount: 0, freshCount: 0, foamCount: 0, lingerCount: 0, recentCount: 0, maxSlope: 0, visibleCount: 0 };
  function absorb(data, step) {
    let breaking = 0, fresh = 0, foam = 0, linger = 0, recent = 0, maxSlope = 0, visible = 0;
    const half = blocks * blocks * 4;
    for (let i = 0; i < blocks * blocks; i += 1) {
      breaking += data[i * 4];
      fresh += data[i * 4 + 1];
      foam += data[i * 4 + 2];
      linger += data[i * 4 + 3];
      recent += data[half + i * 4];
      maxSlope = Math.max(maxSlope, data[half + i * 4 + 1]);
      visible += data[half + i * 4 + 2];
    }
    Object.assign(counts, { step, breakingCount: breaking, freshCount: fresh, foamCount: foam, lingerCount: linger, recentCount: recent, maxSlope, visibleCount: visible });
  }
  function poll() {
    let landed = false;
    if (inFlight.sync) {
      const status = gl.clientWaitSync(inFlight.sync, 0, 0);
      if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
        gl.deleteSync(inFlight.sync);
        inFlight.sync = null;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pack);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, readback);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        if (inFlight.step > counts.step) {
          absorb(readback, inFlight.step);
          const t0 = issuedAt.get(inFlight.step);
          if (t0 !== undefined) lagLog.push(performance.now() - t0);
        }
        landed = true;
      }
    }
    // The newest step was skipped while the buffer was busy and is still on the card: read it.
    if (!inFlight.sync && wanted !== null && wanted === stepIndex) requestCounts(wanted);
    if (landed && last) {
      Object.assign(last, countsView(), { countsStep: counts.step });
      if (api.onCounts) api.onCounts(last);
    }
  }
  let wanted = null;
  const issuedAt = new Map();
  const lagLog = [];
  function takeLags() { return lagLog.splice(0); }
  function requestCounts(step) {
    if (inFlight.sync) { if (inFlight.step !== step) wanted = step; return; }
    wanted = null;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbs.reduce.fb);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pack);
    // Fresh storage for every readback (orphan): Chrome otherwise drops the shadow copy it
    // keeps for fenced READ buffers ("written, then fenced, but written again before being
    // read back") and getBufferSubData falls back to a synchronous GPU read, i.e. a stall.
    gl.bufferData(gl.PIXEL_PACK_BUFFER, readSize * 2, gl.STREAM_READ);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, blocks, blocks, gl.RGBA, gl.FLOAT, 0);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, blocks, blocks, gl.RGBA, gl.FLOAT, readSize);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    inFlight.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    inFlight.step = step;
    gl.flush();
  }

  function bindInputs(prog, inputs) {
    let unit = 4;
    for (const [name, t] of Object.entries(inputs)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(prog.uniforms[name], unit);
      unit += 1;
    }
    gl.activeTexture(gl.TEXTURE0);
  }
  function run(prog, target, size, inputs, set, h = size) {
    gl.useProgram(prog.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, size, h);
    if (prog.uniforms.uN) gl.uniform1i(prog.uniforms.uN, n);
    bindInputs(prog, inputs);
    if (set) set(prog.uniforms);
    timed(prog.name || "pass", () => gl.drawArrays(gl.TRIANGLES, 0, 3));
  }

  const round = Math.round;
  let stepIndex = 0;
  let ampScale = 1;
  // v29.1: shallow water cannot carry deep-water horizontal chop without folding: the choppiness
  // eases to 40% between 10 m and 4 m depth (unchanged at 10 m and deeper).
  const chopFor = (d) => scenario.choppiness * Math.min(1, Math.max(0.4, d / 10));
  let lastDepth = 40;
  let last = null;

  let spCur = 0, strandSp = 1, strandDt = 0; const strandFine = { win: 128, on: false };
  function drawStrands() {
    if (!STRANDS) return;
    const P = programs.strandDraw;
    const L = scenario.size, win = L / Math.max(1, Math.round(L / 256));  // fine window (m, ~256 m = 0.25 m texels), divides the tile
    const cam = globalThis.__oceanCam || [0, 0];
    strandFine.win = win; strandFine.on = FINE;
    for (const fine of FINE ? [false, true] : [false]) {
    gl.useProgram(P.program); gl.bindFramebuffer(gl.FRAMEBUFFER, fine ? fbs.strandFine.fb : fbs.strandDraw.fb); gl.viewport(0, 0, 1024, 1024);
    gl.uniform1f(P.uniforms.uFine, fine ? 1 : 0); gl.uniform1f(P.uniforms.uGain, 1); gl.uniform1f(P.uniforms.uFil, fine ? FILAMENTS : 1); // v30.18.2: no 2x gain (it saturated opacity)
    gl.uniform1f(P.uniforms.uTex, fine ? (win / 1024) : (L / 1024));  // metres per texel of this layer
    gl.uniform1f(P.uniforms.uWin, win / L * M);
    gl.uniform2f(P.uniforms.uCam, (((cam[0] / L + 0.5) % 1 + 1) % 1) * M, (((cam[1] / L + 0.5) % 1 + 1) % 1) * M);
    // v30.11: G (whitecap) persists as a trail behind the riding chains, e-folding over uTrail; R (lace) redrawn
    { const D = programs.strandDecay; gl.useProgram(D.program); gl.uniform1f(D.uniforms.uK, Math.exp(-Math.max(strandDt, 0) / STRAND_TRAIL));
      gl.enable(gl.BLEND); gl.blendFunc(gl.ZERO, gl.SRC_COLOR); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.disable(gl.BLEND); gl.useProgram(P.program); }
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, tex.SP[spCur]); gl.uniform1i(P.uniforms.tS, 4); gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(P.uniforms.uM, M); gl.uniform1f(P.uniforms.uTime, simTimeCover || 0);
    gl.uniform1f(P.uniforms.uSeg, 4.0 * (0.03 * wind.speed * wind.speed / 15) / strandSp * (M / n)); // tear at 4x rest length (fine cells)
    gl.uniform1f(P.uniforms.uYoung, 1.3); gl.uniform1f(P.uniforms.uDk, foamDenseK(wind.speed, scenario.size)); gl.uniform1f(P.uniforms.uFc, FCOV ? 1 : 0);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArraysInstanced(gl.LINES, 0, STRAND_CHAINS * 15 * 2, fine ? FILAMENTS : 1);
    if (SPLATS) { const Q = programs.strandPts; gl.useProgram(Q.program); const QU = Q.uniforms;
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, tex.SP[spCur]); gl.uniform1i(QU.tS, 4); gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(QU.uM, M); gl.uniform1f(QU.uTime, simTimeCover || 0); gl.uniform1f(QU.uTex, fine ? (win / 1024) : (L / 1024));
      gl.uniform1f(QU.uGk, FC_WG * (1 - Math.exp(-Math.max(strandDt, 0) / STRAND_TRAIL))); gl.uniform1f(QU.uFine, fine ? 1 : 0); gl.uniform1f(QU.uWk, Math.min(1, Math.max(0, (wind.speed - 10) / 15))); gl.uniform1f(QU.uWin, win / L * M); gl.uniform1f(QU.uCell, (fine ? win / L * M : M) / 1024); gl.uniform1f(QU.uDk, foamDenseK(wind.speed, scenario.size)); gl.uniform1f(QU.uFc, FCOV ? 1 : 0);
      gl.uniform2f(QU.uCam, (((cam[0] / L + 0.5) % 1 + 1) % 1) * M, (((cam[1] / L + 0.5) % 1 + 1) % 1) * M);
      gl.drawArrays(gl.POINTS, 0, STRAND_CHAINS * 16); gl.useProgram(P.program); }
    { const tr = globalThis.__oceanFoamTrail; // v30.23 buoy foam trail
      if (tr && tr.n > 0) { const Q = programs.foamTrail, QU = Q.uniforms; gl.useProgram(Q.program);
        gl.uniform4fv(QU.uTr, tr.pts); gl.uniform1f(QU.uTrG, TRAIL_GAIN); gl.uniform2fv(QU.uTrN, tr.nrm); gl.uniform1i(QU.uM, M); gl.uniform1f(QU.uL, L); gl.uniform1f(QU.uTex, fine ? (win / 1024) : (L / 1024));
        gl.uniform1f(QU.uGk, 1 - Math.exp(-Math.max(strandDt, 0) / STRAND_TRAIL)); gl.uniform1f(QU.uFine, fine ? 1 : 0); gl.uniform1f(QU.uWin, win / L * M); gl.uniform1f(QU.uCell, (fine ? win / L * M : M) / 1024);
        gl.uniform2f(QU.uCam, (((cam[0] / L + 0.5) % 1 + 1) % 1) * M, (((cam[1] / L + 0.5) % 1 + 1) % 1) * M);
        gl.drawArrays(gl.POINTS, 0, tr.n * 3); gl.useProgram(P.program); } }
    { const sp = globalThis.__oceanRippleSplash; // v30.27 short touch-ripple foam ring
      if (sp && sp.n > 0) { const Q = programs.foamTrail, QU = Q.uniforms; gl.useProgram(Q.program);
        gl.uniform4fv(QU.uTr, sp.pts); gl.uniform1f(QU.uTrG, 1.15); gl.uniform2fv(QU.uTrN, sp.nrm); gl.uniform1i(QU.uM, M); gl.uniform1f(QU.uL, L); gl.uniform1f(QU.uTex, fine ? (win / 1024) : (L / 1024));
        gl.uniform1f(QU.uGk, 1 - Math.exp(-Math.max(strandDt, 0) / STRAND_TRAIL)); gl.uniform1f(QU.uFine, fine ? 1 : 0); gl.uniform1f(QU.uWin, win / L * M); gl.uniform1f(QU.uCell, (fine ? win / L * M : M) / 1024);
        gl.uniform2f(QU.uCam, (((cam[0] / L + 0.5) % 1 + 1) % 1) * M, (((cam[1] / L + 0.5) % 1 + 1) % 1) * M);
        gl.drawArrays(gl.POINTS, 0, sp.n * 3); gl.useProgram(P.program); } }
    if (!fine && globalThis.__foamRead) {
      const q = globalThis.__foamRead;
      const u = ((q.x / L + 0.5) % 1 + 1) % 1, v = ((q.z / L + 0.5) % 1 + 1) % 1;
      const px = Math.min(1022, Math.max(1, Math.floor(u * 1024))), py = Math.min(1022, Math.max(1, Math.floor(v * 1024)));
      const buf = new Float32Array(25 * 4);
      gl.readPixels(px - 2, py - 2, 5, 5, gl.RGBA, gl.FLOAT, buf);
      let maxR = 0, maxG = 0;
      for (let i = 0; i < buf.length; i += 4) { maxR = Math.max(maxR, buf[i]); maxG = Math.max(maxG, buf[i + 1]); }
      globalThis.__foamReadResult = { px, py, maxR, maxG, trailN: (globalThis.__oceanFoamTrail && globalThis.__oceanFoamTrail.n) || 0 };
    }
    gl.disable(gl.BLEND);
    }
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex.STR); gl.generateMipmap(gl.TEXTURE_2D_ARRAY); gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  }
  function step(time, dt, depth) {
    timeEmpty();
    lastDepth = depth;
    const winds = bands.map((band) => wind.sample(band.spec, time, dt));
    testLast.bands = bands; testLast.winds = winds; testLast.comps = [];
    const L = scenario.size;
    bands.forEach((band, i) => retuneBand(band, winds[i], depth, GRAVITY, dt, L / n));
    const setup = bandSetup(bands, winds, depth, GRAVITY, L, n);
    // v29.1 depth-limited waves: the drawn sea's Hs is capped at the breaker index x depth
    // (scenario.depthBreakRatio, 0.78) by one amplitude scale on every wave (grid, explicit waves,
    // ripple patches); the finite-depth dispersion and shoaling are unchanged. The break test keeps
    // the unscaled heights, so where the cap bites the depth-limited breaking (bh >= 0.78 d) rises.
    // Deep water (Hs below the limit, e.g. ~12.6 m at 25 m/s vs 15.6 m at 20 m depth) is scale 1.
    {
      const lat0 = latticeIntegrals(setup, depth, GRAVITY, L);
      const dk0 = TAU / L;
      let v = lat0.reduce((s0, b) => s0 + b.variance, 0);
      bands.forEach((band) => band.components.forEach((c) => { v += (c.amplitude * c.amplitude / 2) * Math.max(0, 1 - fftShare(c.k, dk0)); }));
      const hs = 4 * Math.sqrt(v), lim = scenario.depthBreakRatio * depth;
      ampScale = hs > lim ? lim / hs : 1;
      setup.ampScale = ampScale;
      bands.forEach((band) => { band.ampScale = ampScale; });
    }
    // The ripple patches start exactly where the grid's chop stops.
    bands.forEach((band, i) => { if (band.spec.patch) band.heldOmegaEdge = setup[i].hi; });
    const depthTest = gl.isEnabled(gl.DEPTH_TEST);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(vao);
    const patchStates = patches.map((patch, i) => (patch ? patch.step(bands[i], winds[i], depth, GRAVITY, dt) : null));
    const chopPatch = patchStates.find((p) => p) || null;
    const fineStates = finePatches.map((patch, i) => (patch ? patch.step(bands[i], winds[i], depth, GRAVITY, dt, patchStates[i].kCut) : null));
    const finePatch = fineStates.find((p) => p) || null;
    const chopIndex = bands.findIndex((band) => band.spec.patch);
    const half = L / 2;
    const spacing = L / n;
    const dk = TAU / L;
    const lattice = latticeIntegrals(setup, depth, GRAVITY, L);
    // Explicit long waves: the components the tile cannot hold, with energy share 1 - wFft.
    let explicitMss = 0;
    const explicitVar = bands.map(() => 0);
    const share = (c) => Math.sqrt(Math.max(0, 1 - fftShare(c.k, dk)));
    bands.forEach((band, i) => band.components.forEach((c) => {
      const a = c.amplitude * share(c);
      explicitMss += (a * c.k) ** 2 / 2;
      explicitVar[i] += (a * a) / 2;
    }));
    const gridMss = explicitMss + lattice.reduce((sum, b) => sum + b.mss, 0);
    const roughness = slopeBudget(bands, chopPatch, finePatch, chopIndex >= 0 ? winds[chopIndex].speed : wind.speed, gridMss);
    // GAD perf: the big (far) cascade's FFT runs at most SEA_RATE times a second; between runs its
    // fields are extrapolated in time from the last two results (sea.step skip). The patches, the
    // break test and the foam passes below still run every step. With steps of 1/SEA_RATE s or
    // longer (fixdt 0.05 captures, 60 Hz displays) every step runs the FFT, exactly as before.
    seaAcc += dt;
    const skipSea = sea.rateCap && seaAcc < 0.75 / SEA_RATE && stepIndex > 0;
    const fft = skipSea ? sea.step(setup, depth, GRAVITY, L, dt, true) : sea.step(setup, depth, GRAVITY, L, seaAcc);
    if (!skipSea || fft.extrapolated !== true) seaAcc = 0;
    // The components, in double precision here, then as float32 uniforms. Waves with no
    // amplitude are left out (surface.js skips them too).
    let count = 0;
    for (const band of bands) {
      const isChop = band.spec.id === "chop" ? 1 : 0;
      for (const c of band.components) {
        const sh = share(c);
        if (c.amplitude === 0 || sh === 0) continue;
        const cx = Math.cos(c.direction);
        const cz = Math.sin(c.direction);
        const ax = (c.k * cx * spacing) / TAU;
        const az = (c.k * cz * spacing) / TAU;
        // Coarse part: 12 fractional bits, so cell (< 2^8) x coarse is exact in float32.
        const axH = Math.round(ax * 4096) / 4096;
        const azH = Math.round(az * 4096) / 4096;
        let b0 = (c.k * (-half * cx - half * cz) + c.phase) / TAU;
        b0 -= Math.floor(b0);
        w0.set([c.amplitude * sh * ampScale, c.breakAmplitude * sh, c.k, isChop], count * 4);
        testLast.comps.push({ a: c.amplitude * sh, ba: c.breakAmplitude * sh, k: c.k, cx, cz, phase: c.phase, half, spacing });
        xw0.set([c.amplitude * sh, c.k, cx, cz], count * 4);
        let ph = c.phase / TAU;
        ph -= Math.floor(ph);
        xw1.set([ph, isChop, 0, 0], count * 4);
        w1.set([cx, cz, axH, azH], count * 4);
        w2.set([ax - axH, az - azH, b0, 0], count * 4);
        count += 1;
      }
    }
    const [tx, tz] = travelDirection(bands, wind.direction);
    const crestX = -tz;
    const crestZ = tx;
    const offT = [];
    for (let s = -1; s <= FACE_MAX; s += 1) offT.push(round(tx * s), round(tz * s));
    const face = faceStepsAt(scenario, wind.speed, spacing);
    const joinSteps = Math.min(3, face);
    const offC = [];
    for (let j = -1; j <= 5; j += 1) offC.push(round(crestX * j), round(crestZ * j));
    const offF = [];
    for (let s = 2; s <= 5; s += 1) for (let j = 1; j < s; j += 1) for (let w = -1; w <= 1; w += 1) offF.push(round(tx * ((w * j) / s)), round(tz * ((w * j) / s)));
    const setOffsets = (u) => {
      gl.uniform2iv(u.uOffT, offT);
      gl.uniform2iv(u.uOffC, offC);
      gl.uniform2iv(u.uOffF, offF);
      gl.uniform2f(u.uTravel, tx, tz);
    };

    run(programs.wave, fbs.wave, n, { tT0: fft.T[0], tT1: fft.T[1], tT2: fft.T[2], tT3: fft.T[3], tT4: fft.T[4] }, (u) => {
      gl.uniform1i(u.uCount, count);
      gl.uniform4fv(u.uW0, w0);
      gl.uniform4fv(u.uW1, w1);
      gl.uniform4fv(u.uW2, w2);
      gl.uniform1f(u.uChop, chopFor(lastDepth));
      gl.uniform1f(u.uBreakSlope, scenario.breakSlope);
      gl.uniform1f(u.uDepthLimit, scenario.depthBreakRatio * depth);
    });
    const reach = Math.max(1, round(crestReachAt(scenario, wind.speed) / spacing));
    const r = Math.max(1, round(TRANSLUCENCY.meanRadius / spacing));
    const brk = breakRuleAt(scenario, wind.speed, spacing);
    run(programs.rowsum, fbs.rowsum, n, { tA: tex.A, tC: tex.C, tX: tex.X }, (u) => { gl.uniform1i(u.uR, r); gl.uniform1i(u.uRB, brk.rb); gl.uniform1i(u.uShort, brk.mode > 1.5 ? 1 : 0); });
    run(programs.local, fbs.local, n, { tA: tex.A, tB: tex.B, tC: tex.C, tHX: tex.HX, tX: tex.X }, (u) => {
      setOffsets(u);
      gl.uniform1i(u.uReach, reach);
      gl.uniform1i(u.uR, r);
      if (u.uDepthEta) gl.uniform1f(u.uDepthEta, ampScale < 1 ? 0.06 * lastDepth : 1e9 /* TUNED: short-wave crest elevation >= 0.06 d (0.3 m at 6 m gave 169 breaking, 0.6 m 12, 1.5 m 0) */);
      gl.uniform4f(u.uBrk, brk.mode, brk.ak, brk.rb, brk.ts); gl.uniform1f(u.uBrkLg, brk.lg ?? 1); gl.uniform2f(u.uBrkMod, brk.M || 0, brk.kL || 0);
      gl.uniform1f(u.uSpacing, spacing);
      gl.uniform4f(u.uTransl, TRANSLUCENCY.skin, TRANSLUCENCY.length, TRANSLUCENCY.maxThickness, Number(FQ.get("wgC") ?? 0));
      { const q = FQ; /* lip-width rule loosened with wind: at 25 m/s the lips are longer along travel and the
        forward edge's Jacobian fall is shallower (run 3 -> 5 cells, edge tolerance 0 -> -0.1; GAD option B) */
        const hw = Math.min(1, Math.max(0, (wind.speed - 17.5) / 7.5));
        gl.uniform4f(u.uWidth, Number(q.get("wgD") ?? 3 + 2 * hw), Number(q.get("wgJ") ?? -0.1 * hw), Number(q.get("wgF") ?? 0.2), Number(q.get("wgM") ?? 4)); gl.uniform2f(u.uOpt, Number(q.get("wgS") ?? 0), Number(q.get("wgG") ?? 0)); }
    });
    if (CURL_PASSES && wind.speed >= 11.4) { // v30.22c curl field (see CURL_SRC_FS)
      const axX = Math.abs(tx) >= Math.abs(tz) ? 1 : 0;
      const ax = axX ? [Math.sign(tx) || 1, 0] : [0, Math.sign(tz) || 1], bx = axX ? [0, 1] : [1, 0];
      const cq = FQ;
      const curlS = Number(cq.get("curls") ?? 1.0) * (1 + 1.0 * Math.min(1, Math.max(0, (wind.speed - 15) / 10))); // lip size / crest-to-trough (capped by the wave itself)
      const setAx = (u) => { gl.uniform2i(u.uAx, ax[0], ax[1]); if (u.uBx) gl.uniform2i(u.uBx, bx[0], bx[1]); };
      curlAxis = { x: axX, ax, bx }; globalThis.__oceanCurlAxis = curlAxis;
      run(programs.curlClock, fbs.curlClock[ctCur], n, { tA: tex.A, tL: tex.L, tT: tex.CT[ctCur] }, (u) => { setAx(u); gl.uniform1f(u.uDt, Math.max(dt, 0)); });
      ctCur = 1 - ctCur;
      run(programs.curlSrc, fbs.curlSrc, n, { tA: tex.A, tT: tex.CT[ctCur] }, (u) => { setAx(u); gl.uniform1f(u.uU, wind.speed); gl.uniform1f(u.uSp, spacing); gl.uniform1f(u.uK0, Number(cq.get("curlk") ?? 0.22)); gl.uniform1f(u.uS, curlS); const cx = curlX(cq, wind.speed); globalThis.__oceanCurlXPhys = cx; gl.uniform1f(u.uX, cx); });
      run(programs.curlJoin, fbs.curlJoin, n, { tC: tex.CS, tA: tex.A }, (u) => { setAx(u); gl.uniform1f(u.uSig, Math.max(1.5, Number(cq.get("curlsig") ?? 4.0) / spacing)); gl.uniform1i(u.uR, Math.min(16, Math.ceil(2.5 * Math.max(1.5, Number(cq.get("curlsig") ?? 4.0) / spacing)))); gl.uniform1i(u.uMinLen, Math.ceil(Number(cq.get("curlmin") ?? 10) / spacing)); });
      run(programs.curlSmooth, fbs.curlSmooth, n, { tC: tex.CS, tJ: tex.CJ }, (u) => { setAx(u); gl.uniform1f(u.uSig, Math.max(1.0, 2.5 / spacing)); });
      run(programs.curlCol, fbs.curlCol, n, { tJ: tex.CJ2 }, (u) => { setAx(u); gl.uniform1i(u.uAxisX, axX); }, 1);
      run(programs.curlField, fbs.curlField, n, { tJ: tex.CJ2, tC0: tex.CC[0] }, (u) => { setAx(u); gl.uniform1i(u.uAxisX, axX); gl.uniform1f(u.uSp, spacing); gl.uniform1i(u.uK, Math.min(24, Math.ceil(12 / spacing))); gl.uniform1f(u.uX, curlX(cq, wind.speed)); });
      run(programs.curlList, fbs.curlList, CURL_SLOTS, { tC0: tex.CC[0], tC1: tex.CC[1], tC2: tex.CC[2] }, (u) => { setAx(u); gl.uniform1i(u.uAxisX, axX); }, n);
      curlIdle = false;
    } else if (!curlIdle) { // v30.22g perf: no curls (?curl=0, or below the 11.5 m/s gate): skip the curl passes, clear their outputs once
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbs.curlField.fb); gl.clearBufferfv(gl.COLOR, 0, [-1, 0, 0, 0]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbs.curlList.fb); gl.clearBufferfv(gl.COLOR, 0, [-1, -1, -1, -1]);
      curlIdle = true;
    }
    run(programs.join, fbs.join, n, { tL: tex.L, tC: tex.C }, setOffsets);
    run(programs.jmask, fbs.jmaskX, n, { tJ: tex.J }, (u) => gl.uniform1i(u.uAxis, 0));
    run(programs.jmask, fbs.jmask, n, { tJ: tex.JX }, (u) => gl.uniform1i(u.uAxis, 1));
    run(programs.joinpaint, fbs.joinpaint, n, { tC: tex.C, tJ: tex.J, tJM: tex.JM }, (u) => { setOffsets(u); gl.uniform1i(u.uJoinSteps, joinSteps); });
    run(programs.paint, fbs.paint, n, { tA: tex.A, tC: tex.C, tL: tex.L, tJP: tex.JP }, (u) => {
      setOffsets(u);
      gl.uniform1f(u.uHalf, half);
      gl.uniform1f(u.uSpacing, spacing);
      gl.uniform1f(u.uNoAhead, NO_FOAM_AHEAD);
      gl.uniform1i(u.uFace, face);
      gl.uniform1i(u.uJoinSteps, joinSteps);
    });
    const window = foamWindowAt(scenario, wind.speed);
    advanceAges(dt, window);
    run(programs.age, fbs.age[current], n, { tP: tex.P, tS: tex.S[current], tAges: ageTex }, (u) => {
      setOffsets(u);
      gl.uniform1f(u.uStep, stepIndex + 1);
      gl.uniform1i(u.uAgeLen, ageLen);
      gl.uniform1f(u.uWindow, window);
      gl.uniform1f(u.uNoAhead, NO_FOAM_AHEAD);
    });
    current = 1 - current;
    { // v30c coverage (see COVER_FS)
      const drift = 0.03 * wind.speed + 0.25; // m/s: wind drift ~3% of U10 plus a little Stokes drift
      const wd = wind.direction;
      const lpEff = Math.min(0.877 * wind.speed * wind.speed, 0.5 * L); // peak wavelength, capped by the box
      run(programs.longh, fbs.longh, 64, { tA: tex.A }, (u) => gl.uniform1f(u.uSig, 0.12 * lpEff / spacing));
      run(programs.cover, fbs.cover[kCur], M, { tP: tex.P, tK: tex.K[kCur], tA: tex.A, tHL: tex.HL, tL: tex.L, tCU: tex.CU }, (u) => {
        gl.uniform1f(u.uLp, lpEff);
        gl.uniform2f(u.uShift, drift * Math.cos(wd) * dt / spacing, drift * Math.sin(wd) * dt / spacing);
        gl.uniform1f(u.uDt, Math.max(dt, 0)); gl.uniform1i(u.uM, M);
        gl.uniform1f(u.uDepth, depth); gl.uniform1f(u.uSp, spacing); gl.uniform2f(u.uWind, Math.cos(wd), Math.sin(wd));
        gl.uniform1f(u.uTauD, FOAM_TAU_DENSE);
        gl.uniform1f(u.uTauR, FOAM_TAU_RESID_K * (wind.speed < 15 ? 3.0 + 3.0 * Math.max(0, (wind.speed - 5) / 10) : 6.0 + 9.0 * Math.min(1, (wind.speed - 15) / 10)) * Math.min(1, Math.max(0.45, depth / 6))); // v30.7: residual 6 s at 15 m/s -> 15 s at 25
        gl.uniform1f(u.uTauD, FOAM_TAU_DENSE * Math.min(1, Math.max(0.5, depth / 4))); // v30.7: only shortened in really shallow water // bubbles hit the bed and degas sooner
        gl.uniform1f(u.uFeed, FOAM_FEED);
        gl.uniform1f(u.uTime, simTimeCover = (simTimeCover || 0) + dt);
        gl.uniform1f(u.uTauA, 15.0 + 15.0 * Math.min(1, Math.max(0, (wind.speed - 10) / 15))); // v30.7: bubble cloud 15-30 s
        gl.uniform1f(u.uSrc, FOAM_SRC);
        gl.uniform2f(u.uCrest, -tz, tx);
        gl.uniform1f(u.uJth, FOAM_JTH);
        gl.uniform1f(u.uSrcK, (wind.speed < 4 ? 0 : cov.gain) * foamGain(scenario.steepnessS));
      });
      kCur = 1 - kCur;
      if (globalThis.__airPts && globalThis.__airPts.length) {
        const fb = fbs.cover[1 - kCur].fb;
        const prev = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        const buf = new Float32Array(4), lipb = new Float32Array(4), out = [];
        const list = globalThis.__airPts;
        for (let i = 0; i < list.length && i < 64; i++) {
          const q = list[i];
          const u = ((q.x / L + 0.5) % 1 + 1) % 1, v = ((q.z / L + 0.5) % 1 + 1) % 1;
          const px = Math.min(M - 1, Math.max(0, Math.floor(u * M))), py = Math.min(M - 1, Math.max(0, Math.floor(v * M)));
          gl.readPixels(px, py, 1, 1, gl.RGBA, gl.FLOAT, buf);
          out.push({ x: q.x, z: q.z, dense: buf[0], resid: buf[1], fresh: buf[2], air: buf[3] });
        }
        // Same cells on the lip field (broken, lip). Developer probe only.
        try {
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbs.local.fb);
          gl.readBuffer(gl.COLOR_ATTACHMENT0);
          for (let i = 0; i < out.length; i++) {
            const q = out[i];
            const u = ((q.x / L + 0.5) % 1 + 1) % 1, v = ((q.z / L + 0.5) % 1 + 1) % 1;
            const px = Math.min(n - 1, Math.max(0, Math.floor(u * n))), py = Math.min(n - 1, Math.max(0, Math.floor(v * n)));
            gl.readPixels(px, py, 1, 1, gl.RGBA, gl.FLOAT, lipb);
            q.broken = lipb[0]; q.lip = lipb[1];
          }
        } catch (err) { globalThis.__airErr = String(err); }
        globalThis.__airNow = { t: simTimeCover, out };
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prev);
      }
      // v30.9 strands (see STRAND_UPD_FS)
      if (STRANDS) {
        strandSp = spacing; strandDt = dt;
        run(programs.strandUpd, fbs.strandUpd[spCur], 256, { tS: tex.SP[spCur], tA: tex.A, tK: tex.K[kCur], tHL: tex.HL, tL: tex.L }, (u) => {
          gl.uniform1i(u.uM, M); gl.uniform1f(u.uTime, simTimeCover); gl.uniform1f(u.uDt, Math.max(dt, 0)); gl.uniform1f(u.uSp, spacing);
          gl.uniform1f(u.uLen, 0.03 * wind.speed * wind.speed);
          { const lp = Math.min(0.877 * wind.speed * wind.speed, 0.5 * L); gl.uniform1f(u.uC, Math.sqrt(9.81 * lp / (2 * Math.PI))); gl.uniform1f(u.uGrow, 0.15 * wind.speed); gl.uniform1f(u.uRideAll, L >= 500 ? 1 : 0); } gl.uniform1f(u.uJth, FOAM_JTH);
          gl.uniform1f(u.uRate, (wind.speed < 4 ? 0 : STRAND_RATE * (4096 / STRAND_CHAINS + (1 - 4096 / STRAND_CHAINS) * Math.min(1, Math.max(0, (wind.speed - 11) / 6))) * (1 + STRAND_BOOST * Math.min(1, Math.max(0, (wind.speed - 15) / 10))) * foamRateK(wind.speed)) * foamGain(scenario.steepnessS)); // v30.22f x foamRateK // steepness scales the spawn; the fold test itself is unchanged
          { // v30.16.1: the chain budget is fixed, so in oceans bigger than ~300 m most spawns go to a ~300 m
            // window around the camera (same foam density per m² there as in a 250 m ocean)
            const Ls = scenario.size, cam = globalThis.__oceanCam || [0, 0], win = Math.min(Ls, 300);
            gl.uniform1f(u.uBias, Ls > 350 ? 1 - (win * win) / (Ls * Ls) * 1.0 - 0.1 : 0);
            gl.uniform1f(u.uWinN, win / Ls * n);
            gl.uniform2f(u.uCamN, (((cam[0] / Ls + 0.5) % 1 + 1) % 1) * n, (((cam[1] / Ls + 0.5) % 1 + 1) % 1) * n); }
          gl.uniform1f(u.uLife, 20.0 + 12.0 * Math.min(1, Math.max(0, (wind.speed - 8) / 17))); // v30.18.2: lace lasts tens of seconds (12-45 s)
          gl.uniform2f(u.uShift, drift * Math.cos(wd) * dt / spacing, drift * Math.sin(wd) * dt / spacing);
          gl.uniform2f(u.uWind, Math.cos(wd), Math.sin(wd));
        }, STRAND_ROWS);
        spCur = 1 - spCur;
      }
    }
    // v30.7 coverage controller (foam-realism-2 rank 1.4): hold the ACTIVE whitecap share to
    // W_A(U) = 0.4% (U/10)^2.65 (between Monahan 1980 and Callaghan 2008, active ~1/4-1/3 of total).
    // Measured on the grid, every ~0.25 s, slow (5 s) log-proportional gain change, clamped.
    if (covFence && gl.clientWaitSync(covFence, 0, 0) !== gl.TIMEOUT_EXPIRED) {
      const d = covBuf; gl.bindBuffer(gl.PIXEL_PACK_BUFFER, covPack); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, d); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.deleteSync(covFence); covFence = null;
      let a = 0, v = 0, fs = 0, fl = 0; for (let i = 0; i < 256; i++) { a += d[i * 4]; v += d[i * 4 + 1]; fs += d[i * 4 + 2]; fl += d[i * 4 + 3]; }
      cov.active = a / 256; cov.visible = v / 256; cov.foamWc = fs / 256; cov.foamLace = fl / 256; cov.foamSea = cov.foamWc + FOAM_SL * cov.foamLace; // v30.22f: drawn foam share (FOAM SEA)
      const U = wind.speed; cov.target = U < 4 ? 0 : 0.002 * Math.pow(U / 10, 3.5); // v30.8.1: steeper (rendered/grid share ~2 at low wind) // grid share; ~2x the rendered active share (partial cells)
      const el = Math.min(1, Math.max(0, covT - covLastT)); covLastT = covT;
      // v30.8b (Concept): calibration only. It sets the breaking gain for this wind in the first ~25 s
      // after a wind change, then drifts only very slowly (tau ~ 100 s); foam still only comes from breaking cells.
      if (cov.wind !== U) { cov.wind = U; cov.since = covT; }
      const calib = covT - cov.since < 25 ? 1 : 0.05;
      if (cov.target > 0 && FQ.get("fctrl") !== "0") cov.gain *= Math.exp(Math.max(-1, Math.min(1, Math.log((cov.target + 1e-5) / (cov.active + 1e-5)))) * el * calib / 5);
      cov.gain = Math.max(0.05, Math.min(3, cov.gain));
      globalThis.__foamCov = cov;
    }
    if (!covFence && time - covT >= 0.25) {
      run(programs.covstat, covFb, 16, { tK: tex.K[kCur] }, (u) => { gl.uniform1i(u.uM, M);
        const Ls = scenario.size, cam = globalThis.__oceanCam || [0, 0];
        gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex.STR); gl.uniform1i(u.tS2, 5); gl.activeTexture(gl.TEXTURE0);
        gl.uniform2f(u.uWc, Ls > 350 ? cam[0] / Ls + 0.5 : 0.5, Ls > 350 ? cam[1] / Ls + 0.5 : 0.5); gl.uniform1f(u.uWs, Math.min(Ls, 300) / Ls); });
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, null); gl.activeTexture(gl.TEXTURE0);
      gl.viewport(0, 0, 16, 16);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, covPack); gl.readPixels(0, 0, 16, 16, gl.RGBA, gl.FLOAT, 0); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      covFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); covT = time;
    }
    if (probeFence && gl.clientWaitSync(probeFence, 0, 0) !== gl.TIMEOUT_EXPIRED) {
      const d = probeBufs[probeBi]; probeBi ^= 1; gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePack); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, d); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.deleteSync(probeFence); probeFence = null; probeOut.data = d; probeOut.time = probeTime;
    }
    if ((probePts || uiOn) && !probeFence) {
      const pw = uiOn ? PROBE_W : 16;
      run(programs.probe, probeFb, pw, { tA: tex.A, tG: tex.G }, (u) => { gl.uniform2fv(u["uPts[0]"] || u.uPts, probePts || uiPts); if (u["uPts2[0]"] || u.uPts2) gl.uniform2fv(u["uPts2[0]"] || u.uPts2, uiPts); if (u.uSec) gl.uniform4f(u.uSec, uiSec[0], uiSec[1], uiSec[2], uiSec[3]); gl.uniform1f(u.uTile, L); });
      gl.viewport(0, 0, pw, 1);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePack); gl.readPixels(0, 0, pw, 1, gl.RGBA, gl.FLOAT, 0); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      probeFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); probeTime = time;
    }
    run(programs.reduce, fbs.reduce, blocks, { tL: tex.L, tC: tex.C, tF: tex.F, tG: tex.G });
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (depthTest) gl.enable(gl.DEPTH_TEST);
    stepIndex += 1;
    issuedAt.set(stepIndex, performance.now());
    if (issuedAt.size > 256) issuedAt.delete(issuedAt.keys().next().value);
    poll();
    requestCounts(stepIndex);
    stepFences.push({ step: stepIndex, sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) });
    if (stepFences.length > 64) gl.deleteSync(stepFences.shift().sync);

    drawStrands();
    last = {
      gpu: { n, A: tex.A, B: tex.B, L: tex.L, F: tex.F, G: tex.G, CU: tex.CU, CL: tex.CL, curlAxis, curlSlots: CURL_SLOTS, K: tex.K[kCur], STR: STRANDS ? tex.STR : null, SPt: STRANDS ? tex.SP[spCur] : null, stTime: simTimeCover || 0, stM: M, fineOn: STRANDS && FINE, fineWin: strandFine.win,
        fft: { R: fft.R, stats: fft.statsTex, levels: fft.levels, N: n, L, spacing, choppiness: chopFor(depth), ampScale,
          explicitCount: count, explicit0: xw0.slice(0, Math.max(count, 1) * 4), explicit1: xw1.slice(0, Math.max(count, 1) * 4) } },
      explicitWaves: count,
      probe: probeOut,
      ...countsView(),
      countsStep: counts.step,
      stepIndex,
      foamWindow: window,
      travelX: tx,
      travelZ: tz,
      n,
      size: L,
      sampleCount: n * n,
      depth,
      chopPatch,
      finePatch,
      roughness,
      bands: bands.map((band, i) => ({
        id: band.spec.id,
        label: band.spec.label,
        significantHeight: 4 * ampScale * Math.sqrt(explicitVar[i] + lattice[i].variance),
        latticeVariance: lattice[i].variance,
        explicitVariance: explicitVar[i],
        spectrum: band.spectrum ? band.spectrum.model : "pm",
      })),
    };
    return last;
  }
  function countsView() {
    const { breakingCount, freshCount, foamCount, lingerCount, recentCount, maxSlope, visibleCount } = counts;
    // v30.22f: FOAM SEA (visibleCount / samples) is the foam the water shader draws (strands, cov.foamSea);
    // the break-test field's visible count (v30.20, never drawn since v30c) stays as brkVisibleCount (?foamsea=brk)
    const drawn = STRANDS && FQ.get("foamsea") !== "brk" && cov.foamSea != null;
    return { breakingCount, freshCount, foamCount, lingerCount, recentCount, maxSlope, visibleCount: drawn ? Math.round(cov.foamSea * n * n) : visibleCount, brkVisibleCount: visibleCount };
  }
  // Between steps (paused, or the counts' fence signalled late): pick up finished counts.
  function refresh(state) {
    poll();
    if (state && state.gpu) Object.assign(state, countsView(), { countsStep: counts.step });
    return state;
  }
  // Wait for the latest step's counts (tests only: blocks until the card is done).
  function countsNow() {
    const data = new Float32Array(blocks * blocks * 8);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbs.reduce.fb);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, blocks, blocks, gl.RGBA, gl.FLOAT, data.subarray(0, blocks * blocks * 4));
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, blocks, blocks, gl.RGBA, gl.FLOAT, data.subarray(blocks * blocks * 4));
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    absorb(data, stepIndex);
    return { ...countsView(), step: counts.step };
  }

  // Whole fields, for parity tests only (never per frame).
  const testLast = { bands: null, winds: null, comps: [] };
  function readFields() {
    const out = {};
    const read = (texture) => {
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const data = new Float32Array(n * n * 4);
      gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, data);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.deleteFramebuffer(fb);
      return data;
    };
    const channel = (data, c, count = n * n) => { const a = new Float32Array(count); for (let i = 0; i < count; i += 1) a[i] = data[i * 4 + c]; return a; };
    const readSize = (texture, size) => {
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const data = new Float32Array(size * size * 4);
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.FLOAT, data);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.deleteFramebuffer(fb);
      return data;
    };
    const X = read(tex.X);
    const A = read(tex.A), B = read(tex.B), C = read(tex.C), L = read(tex.L), P = read(tex.P), F = read(tex.F), G = read(tex.G), S = read(tex.S[current]);
    Object.assign(out, {
      xHeights: channel(X, 0), xSlopeX: channel(X, 1), xSlopeZ: channel(X, 2),
      heights: channel(A, 0), shiftX: channel(A, 1), shiftZ: channel(A, 2), jacobians: channel(A, 3),
      normalX: channel(B, 0), normalY: channel(B, 1), normalZ: channel(B, 2), steep: channel(B, 3),
      slopeX: channel(C, 0), slopeZ: channel(C, 1), breakHeights: channel(C, 2), breakSlope: channel(C, 3),
      breaks: channel(L, 0), lips: channel(L, 1), thins: channel(L, 2),
      flags: channel(P, 0), freshFoam: channel(P, 1), freshAhead: channel(P, 2),
      foamGrade: channel(F, 0), foamFade: channel(F, 1), foamAge: channel(F, 2), foam: channel(F, 3),
      ahead: channel(G, 0), runs: channel(G, 1), recent: channel(G, 2),
      longH: channel(readSize(tex.HL, 64), 0, 64 * 64),
      ages: channel(S, 0).map((flagged) => {
        const since = stepIndex - flagged;
        return since < ageLen ? ageData[since] : NEVER;
      }),
    });
    return out;
  }

  const readSea = (which) => (which === "spectrum" ? sea.readSpectrum() : sea.read(which));
  const readPatches = () => ({ chop: patches.find((p) => p)?.read(), fine: finePatches.find((p) => p)?.read() });
  // One fence per step (no readback): the oldest unsignalled one tells how far behind the
  // card is. Fences signal in order, so only the front of the queue is asked.
  const stepFences = [];
  function backlog() {
    poll();
    while (stepFences.length) {
      const st = gl.clientWaitSync(stepFences[0].sync, 0, 0);
      if (st !== gl.ALREADY_SIGNALED && st !== gl.CONDITION_SATISFIED) break;
      gl.deleteSync(stepFences.shift().sync);
    }
    return stepFences.length ? stepIndex - stepFences[0].step + 1 : 0;
  }
    // debug (v27.2): share of texels within JOIN_REACH of a join record
  const joinBlocks = () => { gl.bindFramebuffer(gl.FRAMEBUFFER, fbs.jmask.fb); const d = new Float32Array(n * n * 4); gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, d); gl.bindFramebuffer(gl.FRAMEBUFFER, null); let c = 0; for (let i = 0; i < d.length; i += 4) c += d[i] > 0.5 ? 1 : 0; return c / (n * n); };
  // tests only (parity.js): this step's bands, winds and explicit components, in double precision
  const testState = () => testLast;
  function readCurl() { // v30.22c debug/test: the curl field (RGBA32F n x n), synchronous
    const out = new Float32Array(n * n * 4); gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbs.curlField.fb); gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, out); gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null); return out; }
  globalThis.__oceanCurl = readCurl;
  const api = { readCurl, testState, N: n, joinBlocks, passTimes, takeLags, backlog, readSea, readPatches, step, refresh, countsNow, readFields, kind: "gpu", onCounts: null, setProbe(pts) { probePts = pts; }, setUiProbe(pts, sec, edges) { uiOn = !!(pts || sec || edges); if (pts) uiPts = pts; if (sec) uiSec = sec; } };
  globalThis.__oceanGpu = api;
  return api;
}

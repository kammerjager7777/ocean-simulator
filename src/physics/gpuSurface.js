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
import { createBand, retuneBand, bandVariance } from "./band.js?v=29.1.6";
import { createGpuChopPatch } from "./gpuPatch.js?v=29.1.6";
import { initPassTimer, timed, passTimes, timeEmpty } from "./passTimer.js?v=29.1.6";
import { slopeBudget } from "./roughness.js?v=29.1.6";
import { SEA_RATE, createFftSea, fftSeaProblem, bandSetup, latticeIntegrals, fftShare } from "./fftSea.js?v=29.1.6";
import { travelDirection, NO_FOAM_AHEAD, TRANSLUCENCY, crestReachAt, breakRuleAt, foamWindowAt, faceStepsAt, FACE_MAX } from "./surface.js?v=29.1.6";

const TAU = Math.PI * 2;
// Ages of samples that have never carried foam (surface.js uses Infinity).
const NEVER = 1e30;
const BLOCK = 16;

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
  for (int dz = 0; dz < ${BLOCK}; dz += 1) {
    for (int dx = 0; dx < ${BLOCK}; dx += 1) {
      ivec2 q = base + ivec2(dx, dz);
      if (!inside(q)) continue;
      bool flagged = texelFetch(tG, q, 0).w > 0.0;
      bool foam = texelFetch(tF, q, 0).w > 0.0;
      c += vec4(texelFetch(tL, q, 0).x > 0.0 ? 1.0 : 0.0, flagged ? 1.0 : 0.0, foam ? 1.0 : 0.0, foam && !flagged ? 1.0 : 0.0);
      if (texelFetch(tG, q, 0).z > 0.0) recent += 1.0;
      maxSlope = max(maxSlope, texelFetch(tC, q, 0).w);
    }
  }
  oCount = c;
  oMore = vec4(recent, maxSlope, 0.0, 0.0);
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

function makeProgram(gl, fs, name) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, QUAD_VS, name + " vertex"));
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
    jmask: makeProgram(gl, JMASK_FS, "jmask"),
    joinpaint: makeProgram(gl, JOINPAINT_FS, "joinpaint"),
    reduce: makeProgram(gl, REDUCE_FS, "reduce"),
  };
  for (const [k, v] of Object.entries(programs)) v.name = k;
  initPassTimer(gl);
  const vao = gl.createVertexArray();

  function texture(size, data) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, size, size);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (data) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, size, size, gl.RGBA, gl.FLOAT, data);
    return tex;
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
  };
  let current = 0; // tex.S[current] holds the latest foam state

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
  const counts = { step: -1, breakingCount: 0, freshCount: 0, foamCount: 0, lingerCount: 0, recentCount: 0, maxSlope: 0 };
  function absorb(data, step) {
    let breaking = 0, fresh = 0, foam = 0, linger = 0, recent = 0, maxSlope = 0;
    const half = blocks * blocks * 4;
    for (let i = 0; i < blocks * blocks; i += 1) {
      breaking += data[i * 4];
      fresh += data[i * 4 + 1];
      foam += data[i * 4 + 2];
      linger += data[i * 4 + 3];
      recent += data[half + i * 4];
      maxSlope = Math.max(maxSlope, data[half + i * 4 + 1]);
    }
    Object.assign(counts, { step, breakingCount: breaking, freshCount: fresh, foamCount: foam, lingerCount: linger, recentCount: recent, maxSlope });
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
  function run(prog, target, size, inputs, set) {
    gl.useProgram(prog.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, size, size);
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
      gl.uniform4f(u.uTransl, TRANSLUCENCY.skin, TRANSLUCENCY.length, TRANSLUCENCY.maxThickness, Number(new URLSearchParams(typeof location !== "undefined" ? location.search : "").get("wgC") ?? 0));
      { const q = new URLSearchParams(typeof location !== "undefined" ? location.search : ""); /* lip-width rule loosened with wind: at 25 m/s the lips are longer along travel and the
        forward edge's Jacobian fall is shallower (run 3 -> 5 cells, edge tolerance 0 -> -0.1; GAD option B) */
        const hw = Math.min(1, Math.max(0, (wind.speed - 17.5) / 7.5));
        gl.uniform4f(u.uWidth, Number(q.get("wgD") ?? 3 + 2 * hw), Number(q.get("wgJ") ?? -0.1 * hw), Number(q.get("wgF") ?? 0.2), Number(q.get("wgM") ?? 4)); gl.uniform2f(u.uOpt, Number(q.get("wgS") ?? 0), Number(q.get("wgG") ?? 0)); }
    });
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

    last = {
      gpu: { n, A: tex.A, B: tex.B, L: tex.L, F: tex.F, G: tex.G,
        fft: { R: fft.R, stats: fft.statsTex, levels: fft.levels, N: n, L, spacing, choppiness: chopFor(depth), ampScale,
          explicitCount: count, explicit0: xw0.slice(0, Math.max(count, 1) * 4), explicit1: xw1.slice(0, Math.max(count, 1) * 4) } },
      explicitWaves: count,
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
    const { breakingCount, freshCount, foamCount, lingerCount, recentCount, maxSlope } = counts;
    return { breakingCount, freshCount, foamCount, lingerCount, recentCount, maxSlope };
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
    const channel = (data, c) => { const a = new Float32Array(n * n); for (let i = 0; i < n * n; i += 1) a[i] = data[i * 4 + c]; return a; };
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
  const api = { testState, N: n, joinBlocks, passTimes, takeLags, backlog, readSea, readPatches, step, refresh, countsNow, readFields, kind: "gpu", onCounts: null };
  return api;
}

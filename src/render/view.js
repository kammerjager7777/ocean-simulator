import { VERT_FFT, FRAG_FFT_NORMAL, buildFftMesh, visibleRanges } from "./fftView.js";
import { currentIsland, useIslandSize, sandbarBed } from "../physics/island.js";
import { LOBE_GLSL } from "../physics/roughness.js";
import { timed } from "../physics/passTimer.js";
import { createBoatView, createBuoyView } from "./boatView.js";
import { BED_GLSL, bedRelief, setBedRippleDir } from "./seabed.js";
import { RIPPLE_GLSL } from "../physics/ripples.js";
import { WAVENUMBER_GLSL } from "../physics/spectrum.js";
import { drawSkyDome, setV23Uniforms, setBedOnly } from "./v23hooks.js";
import { createLightMaps, lightSampleGlsl, shadowGlsl } from "./light.js";
import { assertFragmentSamplers } from "./samplerBudget.js";
// Beach bisect toggles: ?beach=nodamp drops the damp darkening, ?beach=nosurf the shore foam.
const BEACH_MODE = (() => {
  const q = typeof location !== "undefined" ? new URLSearchParams(location.search).get("beach") : null;
  return q === "nodamp" ? 1 : q === "nosurf" ? 2 : 0;
})();
import { SEABED_LUT } from "./seabedLut.js";
import { CURL_SHAPE_GLSL, CURL_SLOTS, curlX } from "../physics/gpuSurface.js";
const FAR_HALVE_DEFAULT = 0; // 05:16 horizon bar: quarter-density far rings flattened the sun25 silhouette (?farhalve=1 to test)
const CHUNK_DEFAULT = 1; // v28 chunk (Michael 09:23): only the one square tile of sea (size slider), no horizon rings (?chunk=0 for the full sea)
const ONE_COPY_DEFAULT = 0; // GAD 01:08: onecopy=0 (all three FFT copies at every distance)
const TILE_THIN_DEFAULT = 1; // GAD 01:08: tile mesh at half density past ~100 m (?tilethin=0 off)
const TILE_THIN_D1 = 75; // = fftView.js TILE_THIN_D1

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ]);
}

function lookAt(eye, center, up) {
  let z0 = eye[0] - center[0];
  let z1 = eye[1] - center[1];
  let z2 = eye[2] - center[2];
  let len = Math.hypot(z0, z1, z2) || 1;
  z0 /= len; z1 /= len; z2 /= len;
  let x0 = up[1] * z2 - up[2] * z1;
  let x1 = up[2] * z0 - up[0] * z2;
  let x2 = up[0] * z1 - up[1] * z0;
  len = Math.hypot(x0, x1, x2) || 1;
  x0 /= len; x1 /= len; x2 /= len;
  const y0 = z1 * x2 - z2 * x1;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = z0 * x1 - z1 * x0;
  return new Float32Array([
    x0, y0, z0, 0,
    x1, y1, z1, 0,
    x2, y2, z2, 0,
    -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
    -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
    -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
    1,
  ]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

const VERT = `
attribute vec2 xz;
attribute float height;
attribute vec3 normal;
attribute vec2 shift;
attribute float broken;
attribute float lip;
attribute float foam;
attribute float flag;
attribute float recent;
attribute float fade;
attribute float age;
attribute float ahead;
attribute float run;
attribute float thin;
uniform mat4 viewProj;
varying float vBroken;
varying float vLip;
varying float vFoam;
varying float vFlag;
varying float vRecent;
varying float vFade;
varying float vAge;
varying float vAhead;
varying float vRun;
varying float vThin;
varying vec3 vNormal;
varying vec3 vWorld;
void main() {
  vThin = thin;
  vAge = age;
  vNormal = normal;
  vFlag = flag;
  vRecent = recent;
  vFade = fade;
  vAhead = ahead;
  vRun = run;
  vBroken = broken;
  vLip = lip;
  vFoam = foam;
  vWorld = vec3(xz.x + shift.x, height, xz.y + shift.y);
  gl_Position = viewProj * vec4(vWorld, 1.0);
}
`;

// GPU physics (gpuSurface.js): the same per-vertex values, read straight from the physics
// textures by vertex number instead of streamed from the CPU. Positions still come from xz.
const VERT_GPU = `#version 300 es
in vec2 xz;
uniform mat4 viewProj;
uniform highp sampler2D physA; // height, shift x, shift z, Jacobian
uniform highp sampler2D physB; // normal, steep
uniform highp sampler2D physL; // breaks, lips, thins, crest
uniform highp sampler2D physF; // foam grade, fade, age, strength
uniform highp sampler2D physG; // ahead, run, recent, flag
uniform int physN;
out float vBroken;
out float vLip;
out float vFoam;
out float vFlag;
out float vRecent;
out float vFade;
out float vAge;
out float vAhead;
out float vRun;
out float vThin;
out vec3 vNormal;
out vec3 vWorld;
void main() {
  ivec2 cell = ivec2(gl_VertexID % physN, gl_VertexID / physN);
  vec4 a = texelFetch(physA, cell, 0);
  vec4 l = texelFetch(physL, cell, 0);
  vec4 f = texelFetch(physF, cell, 0);
  vec4 g = texelFetch(physG, cell, 0);
  vThin = l.z;
  vAge = f.z;
  vNormal = texelFetch(physB, cell, 0).xyz;
  vFlag = g.w;
  vRecent = g.z;
  vFade = f.y;
  vAhead = g.x;
  vRun = g.y;
  vBroken = l.x;
  vLip = l.y;
  vFoam = f.x;
  vWorld = vec3(xz.x + a.y, a.x, xz.y + a.z);
  gl_Position = viewProj * vec4(vWorld, 1.0);
}
`;

// WebGL2 runs the same shader text as GLSL ES 3.00 (derivatives are core there; GLSL 1.00
// under WebGL2 has none). Only the spelling changes, so the picture is the same.
// v29 seafloor, walls and depth fade (GAD look spec ocean-sim-seafloor-look-spec.md, 2026-10-19).
// Every look constant is here. Kd and GAIN can be overridden from the address for tuning:
// ?kd=R,G,B (m^-1), ?gain=G. Model: floor R(z) = Rinf + (rho - Rinf) exp(-2 Kd z) (round trip),
// wall W(z) = fog + (face - fog) exp(-Kd z) (one way), display_linear = GAIN R; all linear, encoded
// to sRGB at the end (v28's own colour constants are display values).
const SEA_PARAMS = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const SEAFLOOR = {
  kd: (SEA_PARAMS.get("kd") || "0.233,0.0653,0.0257").split(",").map(Number), // Jerlov IA (R 611, G 549, B 464 nm)
  gain: Number(SEA_PARAMS.get("gain") || 1.2),         // exposure: sand at 5 m at luma ~127
  rInf: [0.0017, 0.0094, 0.0437],                      // deep-water reflectance; GAIN*rInf = fog #071B41
  sand: [0.48145, 0.38216, 0.29125],                 // carbonate sand: linear sRGB of the fitted sand spectrum (palette 10:01)
  coral: [0.12, 0.08, 0.05],                           // brown coral
  rock: [0.18, 0.15, 0.12],                            // rock / coralline
  face: [0.02641, 0.06404, 0.1222],                  // the water body face (palette 10:01, linear): the wall top
  meniscus: [0.08231, 0.27064, 0.56329],                 // TRANSL_COLOR (linear), 50% in the 0.10 m band
  sky: SEA_PARAMS.get("sky") === "storm" ? [0.1620, 0.1752, 0.2122] : [0.2961, 0.3935, 0.4480], // sky horizon (linear), Fresnel on the walls <= 8%
  rippleLen: 0.10, rippleAmp: 0.08,                    // 10 cm sand ripples, +-8% shading at the top
  sliceBand: 0.4,                                      // floor cross-section band at the wall bottom (m)
};
// ?sky=storm: overcast / storm sky (neutral grey ~#70747F, no direct sun); default = the approved sky.
// GAD 10:14 (b): under storm only, the reflection's horizon cut-off (SKY_SEEN) is relaxed: an
// overcast sky is bright right down to the horizon, so rays reflected near it still see grey.
const STORM = SEA_PARAMS.get("sky") === "storm";
// GAD 10:26: under sky=storm only, everything lit from above in the water (body upwelling,
// translucency lift, floor, walls, reef) is scaled by the overcast-to-clear global irradiance
// ratio. Kasten & Czeplak (1980), Solar Energy 24:177-189: G(N)/G_clear = 1 - 0.75 (N/8)^3.4,
// so at full overcast (N = 8 oktas) 0.25. Roughness, reflectance and the sky reflection are untouched.
const UPWELL = STORM ? 1 - 0.75 * Math.pow(8 / 8, 3.4) : 1;
const SKY_HORIZON = STORM ? [0.439, 0.455, 0.498] : [0.58, 0.66, 0.70];
const SKY_ZENITH_C = STORM ? [0.36, 0.37, 0.41] : [0.36, 0.47, 0.57];
const v3 = (a) => `vec3(${a.map((x) => Number(x).toFixed(6)).join(", ")})`;
const SEAFLOOR_FS = `
varying float vKind;
varying vec4 vSea; // (declared here so the wall shading can read it; dropped from FRAG_FFT_NORMAL below)
varying float vSurfY;
uniform float block;
uniform sampler2D clusterTex;
uniform vec3 clusterGrid;
uniform sampler2D seabedLut;
const float SB_N = ${SEABED_LUT.n}.0;
const float SB_DZ = ${SEABED_LUT.dz};
const vec3 SB_FOG = ${v3(SEABED_LUT.fog)};
// v30.24 (Michael: parts of the seabed ignored the clarity): the baked LUT is the default water; z runs at the clarity's
// equivalent depth (uClarK = Kd(green) / Kd(default) = 22.30 m / Zsd), so the floor, the coral and the wall's floor band fade with it
vec3 sbRow(float z, float row) { return texture2D(seabedLut, vec2((clamp(z * uClarK / SB_DZ, 0.0, SB_N - 1.0) + 0.5) / SB_N, (row + 0.5) / 3.0)).rgb; }
const vec3 SF_KD = ${v3(SEAFLOOR.kd)};
const float SF_GAIN = ${Number(SEAFLOOR.gain).toFixed(6)};
const vec3 SF_RINF = ${v3(SEAFLOOR.rInf)};
const vec3 SF_SAND = ${v3(SEAFLOOR.sand)};
const vec3 SF_CORAL = ${v3(SEAFLOOR.coral)};
const vec3 SF_ROCK = ${v3(SEAFLOOR.rock)};
const vec3 SF_FACE = ${v3(SEAFLOOR.face)};
const vec3 SF_MENISCUS = ${v3(SEAFLOOR.meniscus)};
const vec3 SF_SKY = ${v3(SEAFLOOR.sky)};
const float SF_RIPPLE_LEN = ${SEAFLOOR.rippleLen.toFixed(4)};
const float SF_RIPPLE_AMP = ${SEAFLOOR.rippleAmp.toFixed(4)};
const float SF_SLICE = ${SEAFLOOR.sliceBand.toFixed(4)};
const float SF_UPWELL = ${UPWELL.toFixed(4)};
${WAVENUMBER_GLSL}vec3 sfEncode(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float sfHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float sfNoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(sfHash(i), sfHash(i + vec2(1, 0)), f.x), mix(sfHash(i + vec2(0, 1)), sfHash(i + vec2(1, 1)), f.x), f.y); }
// Carbonate grain and the reef contact shadow. The ripple field is sfSandTex;
// this is only the sand colour, so a 10 cm stripe is not painted on twice.
vec3 sfAlbedo(vec2 q) {
  float g = sfNoise(q * 18.0) - 0.5;
  vec3 a = vec3(0.76, 0.72, 0.64) * (1.0 + 0.08 * g);
  // clusters are instanced geometry (buildClusters); the sand keeps their soft contact shadow
  // (AO up to -20%), read from the cluster cell table (x, z inside the cell, radius, present).
  float ao = 1.0;
  if (clusterGrid.z > 0.5) {
    vec2 cell = floor((q - clusterGrid.xx) / clusterGrid.y);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
      vec2 c = cell + vec2(float(i), float(j));
      if (c.x < 0.0 || c.y < 0.0 || c.x >= clusterGrid.z || c.y >= clusterGrid.z) continue;
      vec4 e = texture2D(clusterTex, (c + 0.5) / clusterGrid.z);
      if (e.w < 0.5) continue;
      vec2 ctr = clusterGrid.xx + (c + e.xy) * clusterGrid.y;
      float rad = e.z * 8.0;
      // Contact under the rock only. A wide disc read as a ring around every mound.
      ao = min(ao, 1.0 - 0.32 * (1.0 - smoothstep(rad * 0.05, rad * 0.42, length(q - ctr))));
    }
  }
  return a * ao;
}
// Orbital ripples. sp is the plane the pattern is drawn in (the floor, or the
// wall face). worldXZ is the sea position the depth and the refraction use.
uniform vec2 uWindDir;
uniform vec2 uCurDir;
uniform float uCurSpd;
uniform float uTp;
uniform float uHsRip;
uniform float uRipDbg;
uniform float uHasIsland;
uniform vec2 uRipC;
uniform float uRipBank;
uniform float uRipFace;
vec3 sfSandTex(vec2 sp, float bedY) {
  vec2 worldXZ = vWorld.xz;
  float h = max(0.08, -bedY);
  vec2 travel = uRipDir;
  float tlen = length(travel);
  travel = tlen > 0.1 ? travel / tlen : vec2(0.0, 1.0);
  vec2 wind = uWindDir;
  float wlen = length(wind);
  wind = wlen > 0.1 ? wind / wlen : travel;
  bool dry = bedY > 0.02;
  vec2 dir = travel;
  float lee = 0.0;
  if (uHasIsland > 0.5) {
    vec2 rel = worldXZ - uRipC;
    float r = length(rel);
    vec2 radial = r > 1.0 ? rel / r : vec2(0.0, 1.0);
    float turn = (1.0 - smoothstep(1.5, 16.0, h)) * (1.0 - smoothstep(uRipBank, uRipBank + 6.0, r));
    lee = smoothstep(0.15, 0.75, dot(radial, travel)) * (1.0 - smoothstep(uRipFace, uRipBank, r));
    // A full bend closes every crest into a circle around the island, lee side
    // included. Keep a partial wrap; the lee stays with the open-sea direction.
    // A stronger bend closed the crests into bullseyes around the bank and the mounds.
    dir = normalize(mix(travel, -radial, clamp(turn, 0.0, 0.12) * (1.0 - lee)));
  }
  if (uRipDbg > 0.5) {
    float ang = atan(dir.y, dir.x);
    return vec3(0.5 + 0.5 * cos(ang), 0.5 + 0.5 * sin(ang), 0.25 + 0.5 * lee);
  }
  float omega = 6.2831853 / max(uTp, 1.0);
  float k = waveNumber(omega, h, 9.81);
  float L = 6.2831853 / max(k, 1e-3);
  float snh = max(sinh(k * h), 0.2);
  float d0 = uHsRip / snh;
  float lam = clamp(0.65 * d0, 0.07, 0.55);
  float ub = 3.14159265 * uHsRip / (max(uTp, 1.0) * snh);
  float shields = (0.5 * 1025.0 * 0.04 * ub * ub) / (1650.0 * 9.81 * 0.0002);
  float sheet = smoothstep(0.55, 0.95, shields);
  float below = smoothstep(0.85, 1.15, h / max(L * 0.5, 1.0));
  // No wind and no waves: orbital ripples are absent, not a bullseye.
  float rip = (1.0 - sheet) * (1.0 - below) * smoothstep(0.06, 0.45, uHsRip);
  vec3 vd = normalize(eye - vWorld);
  float fp = max(length(dFdx(sp)), length(dFdy(sp)));
  float gScale = clamp(2.8 / max(fp, 0.03), 0.45, 32.0);
  vec3 col;
  float eta = 0.0;
  float amp = 0.0;
  float grain = 0.5;
  if (dry) {
    float spacing = clamp(0.02 + 0.004 * uHsRip, 0.02, 0.10);
    float ph = dot(sp, wind) / spacing;
    eta = sin(6.2831853 * ph);
    amp = 0.12 * spacing * (1.0 - smoothstep(spacing * 0.35, spacing * 1.4, fp));
    vec2 puv = sp + vd.xz * eta * amp * 3.0;
    grain = sfNoise(puv * gScale);
    col = vec3(0.86, 0.82, 0.74) * (0.90 + 0.22 * (grain - 0.5));
    float slope = amp * 6.2831853 / spacing * cos(6.2831853 * ph);
    col *= 0.72 + 0.28 * clamp(dot(normalize(vec3(-wind * slope, 1.0)), normalize(lightDir)), 0.0, 1.0);
  } else {
    float ph = dot(sp, dir) / lam;
    eta = sin(6.2831853 * ph);
    float cross = lee * sin(6.2831853 * dot(sp, vec2(-dir.y, dir.x)) / (lam * 1.7));
    eta = mix(eta, eta * 0.45 + cross, lee);
    amp = 0.15 * lam * rip * mix(1.0, 0.4, lee);
    amp *= 1.0 - smoothstep(lam * 0.3, lam * 1.25, fp);
    float asym = uCurSpd > 0.05 ? 0.35 * sign(dot(dir, uCurDir)) : 0.0;
    eta += asym * 0.35 * sin(12.5663706 * ph);
    vec2 puv = sp + vd.xz * eta * amp * 4.0;
    grain = sfNoise(puv * gScale);
    float wet = smoothstep(0.7, -0.9, bedY);
    vec3 dryC = vec3(0.84, 0.80, 0.72);
    // Dry sand, then a wide damp band, then the sand under clear water. No hard moat edge.
    vec3 carbonate = mix(dryC, dryC * vec3(0.90, 0.92, 0.91), wet * 0.28);
    col = carbonate * (0.88 + 0.24 * (grain - 0.5));
    col += vec3(wet * 0.05 * pow(clamp(vd.y, 0.0, 1.0), 6.0));
    float slope = amp * 6.2831853 / max(lam, 0.05) * cos(6.2831853 * ph);
    col *= 0.78 + 0.22 * clamp(dot(normalize(vec3(-dir * slope, 1.0)), normalize(lightDir)), 0.0, 1.0);
    col = mix(col, carbonate * 0.96, sheet);
  }
  float faceY = abs(normalize(cross(dFdx(vWorld), dFdy(vWorld))).y);
  float onWall = 1.0 - smoothstep(0.22, 0.62, faceY);
  float layer = sin(6.2831853 * (sp.y * 0.42 + sfNoise(sp * 0.16) * 1.6));
  col *= 1.0 + onWall * (0.11 * layer + 0.18 * (sfNoise(sp * gScale * 0.4 + 3.0) - 0.5));
  return col * sfAlbedo(sp) / vec3(0.76, 0.72, 0.64);
}
// GAD 2.2 hard cap on sand (floor and the wall's floor slice): no channel above sRGB 194, luma <= 150
vec3 sfCap(vec3 o) { o = min(o, vec3(194.0 / 255.0)); float l = dot(o, vec3(0.2126, 0.7152, 0.0722)); return l > 150.0 / 255.0 ? o * (150.0 / 255.0) / l : o; }
// floor: the baked spectral lookup, A(z) + GAIN rho T(z) (exact for the sand spectrum)
vec3 sfFloor(float z, vec3 rho) { return sbRow(z, 0.0) + SF_GAIN * rho * sbRow(z, 1.0); }
uniform float sfRaw;
#define SAND_DBG ${typeof location !== "undefined" && new URLSearchParams(location.search).get("sanddbg") === "1" ? "true" : "false"}
#define SAND_DBG2 SAND_DBG
#ifdef ISLAND_ON
uniform float uIsland;
uniform vec2 uIslandC;
uniform float uIRb, uIRf, uIRs, uIRbank, uIEase;
uniform float uIPeak, uIYb, uIYf, uIYs, uIFace, uIShelf;
uniform float uR2;
float iHermite(float s, float y0, float y1, float m0, float m1) {
  float s2 = s * s, s3 = s2 * s;
  return (2.0 * s3 - 3.0 * s2 + 1.0) * y0 + (s3 - 2.0 * s2 + s) * m0
       + (-2.0 * s3 + 3.0 * s2) * y1 + (s3 - s2) * m1;
}
// Same curve as sandbarBed. Uniforms follow the sea size; a wall ray has to
// hit this, not the flat floor.
float islandBed(vec2 xz) {
  vec2 rel = xz - uIslandC;
  float th = atan(rel.y, rel.x);
  float lump = 0.055 * sin(3.0 * th) + 0.032 * sin(5.0 * th + 1.7) + 0.018 * sin(8.0 * th + 0.4);
  float r = length(rel) / max(1.0 + lump, 0.75);
  float yFloor = -max(depth, 0.5);
  if (r >= uIRbank) return yFloor;
  float Rb = uIRb;
  float Rf = uIRf;
  float Rs = uIRs;
  float ease = uIEase > 0.0 ? uIEase : 8.0;
  float join0 = Rf - ease;
  float join1 = Rf + ease;
  float y;
  if (r <= Rb) y = iHermite(r / max(Rb, 1.0), uIPeak, uIYb, 0.0, -uIFace * Rb);
  else if (r <= join0) y = uIYb - uIFace * (r - Rb);
  else if (r < join1) {
    float y0 = uIYb - uIFace * (join0 - Rb);
    float y1 = uIYf - uIShelf * (join1 - Rf);
    float span = max(join1 - join0, 1.0);
    y = iHermite((r - join0) / span, y0, y1, -uIFace * span, -uIShelf * span);
  } else if (r <= Rs) y = uIYf - uIShelf * (r - Rf);
  else {
    float span = max(uIRbank - Rs, 1.0);
    y = iHermite((r - Rs) / span, uIYs, yFloor, -uIShelf * span, 0.0);
  }
  y += 0.16 * uIPeak * exp(-((r - Rb) / 4.0) * ((r - Rb) / 4.0));
  if (r < Rb) y += 0.10 * uIPeak * sin(0.42 * r + 2.0 * th) * sin(0.28 * r - 3.0 * th);
  float spanB = max(uIRbank - Rs, 1.0);
  float uB = clamp((r - Rs) / spanB, 0.0, 1.0);
  y += 0.55 * uIPeak * sin(7.0 * th + 0.8) * sin(3.14159265 * uB);
  return y < yFloor ? yFloor : y;
}
// The same ripples and grain the scene pass puts on the open floor, so the
// beach, the bank and the seabed are one sand.
// sfSandTex switches between dry and wet sand on a hard height test. Fed
// a vertex height (piecewise linear on the grid mesh), that switch drew an
// axis-aligned stair, and the sea refracts the scene pass, so the stair
// showed through the shallows. Blend the two looks across the analytic
// bed instead, with a little mottle so the blend is not a clean ring.
vec3 sfSandShore(vec2 sp, float b, float y) {
  float w = smoothstep(0.08, 0.55, b);
  if (w < 0.001) return sfSandTex(sp, min(y, 0.0));
  if (w > 0.999) return sfSandTex(sp, max(y, 0.05));
  return mix(sfSandTex(sp, min(b, 0.0)), sfSandTex(sp, max(b, 0.05)), w);
}
vec3 sfSandDetail(vec2 sp) {
  if (uIsland < 0.5) return sfSandTex(sp, vWorld.y);
  return sfSandShore(sp, islandBed(sp) + (sfNoise(sp * 3.4 + 4.0) - 0.5) * 0.1, vWorld.y);
}
#endif
${lightSampleGlsl()}
uniform mat4 uShVP;
uniform float uCausDay;
uniform float uCausDeep;
uniform float uSeaL;
uniform vec2 uCaustO;
uniform float uCaustTile;
uniform float uBisect;
${shadowGlsl()}
float shadowVis(vec3 w) {
  if (uBisect > 2.5) return 1.0;
  return shadowFactor(w);
}
vec3 floorMul(vec3 bedP, vec2 slope, float focus) {
  if (uCausDay < 0.01 || (uBisect > 0.5 && uBisect < 1.5)) return vec3(1.0);
  vec3 Rf = refract(-normalize(lightDir), vec3(0.0, 1.0, 0.0), 1.0 / 1.333);
  if (Rf.y > -0.02) Rf = vec3(0.0, -1.0, 0.0);
  vec2 xz = bedP.xz - bedP.y * Rf.xz / Rf.y;
  float col = max(0.0, -bedP.y);
  // v32: per colour, split along the refracted sun ray (causDisp), wider with depth
  vec3 ratio = (uBisect > 1.5 && uBisect < 2.5) ? vec3(1.0) : causNetRGB(xz, col, slope, causDisp(lightDir, col), focus);
  // Off the dry beach and the swash. Strongest from about 0.3 m to 3 m, then it fades.
  float ashore = smoothstep(0.08, 0.45, col);
  float fine = exp(-max(col - 0.5, 0.0) * 0.28);
  float far = exp(-col * max(uClarK, 0.35) / 12.0);
  float strength = ashore * clamp(fine * 0.9 + far * 0.14, 0.0, 0.78) * max(uCausDay, 0.0);
  return mix(vec3(1.0), ratio, strength);
}
// Sun on the floor under the sea. The caustic sparkle needs the sun, so it
// fades out inside a shadow instead of rippling over it; then the shadow
// leaves the shallow sand only its sky light.
vec3 floorLight(vec3 bedP, vec2 slope, float H, float shF, float focus) {
  vec3 cm = mix(vec3(1.0), floorMul(bedP, slope, focus), shF);
  return cm * mix(mix(0.06, 0.55, smoothstep(0.3, 8.0, H)), 1.0, shF);
}
vec3 underLight(vec3 rgb, vec3 w) {
  // The side wall is a vertical cut. A caustic map in x/z is constant down
  // the face, so it draws bright columns, and those columns flicker as the
  // live map is redrawn. The cut stays the water colour. Light hits the floor.
  if (vKind < 1.5) return rgb;
  float sh = shadowVis(w);
  // The first metre stays sand. Darkening ramps with depth, and the sun shadow
  // is a soft shape on top of that, not a black ring at the waterline.
  float below = smoothstep(0.4, -1.4, w.y);
  float col0 = max(0.05, -w.y);
  float dunk = mix(0.96, 0.55, smoothstep(1.5, 12.0, col0));
  rgb *= mix(1.0, mix(dunk * mix(0.48, 1.0, sh), 1.0, sh), below);
  return rgb;
}
vec4 seafloorShade() {
  vec3 fog = SB_FOG;
  if (vKind > 1.5 && sfRaw > 0.5) { // v30.5 scene pass: bare albedo, height 0
    // v30.6 sand texture: wave-formed ripples (~0.8 m crests, wavy, bifurcating via a warped phase)
    // lit on their sun side, plus fine grain; both fade with pixel footprint (no shimmer).
#ifdef ISLAND_ON
    vec3 al = sfSandDetail(vWorld.xz);
#else
    vec3 al = sfSandTex(vWorld.xz, vWorld.y);
#endif
    if (SAND_DBG) {
      vec2 sp = vWorld.xz;
      float fpS = max(length(dFdx(vWorld)), length(dFdy(vWorld)));
      float ph = dot(sp, vec2(0.81, 0.59)) * 3.9 + 5.0 * valueNoise(sp * 0.22) + 1.6 * valueNoise(sp * 1.1 + 9.0);
      float rip = (sin(ph) + 0.35 * sin(2.0 * ph + 1.0)) * (1.0 - smoothstep(0.2, 0.6, fpS)) * mix(0.5, 1.0, valueNoise(sp * 0.12 + 4.0));
      al = vec3(0.5 + 0.5 * rip, fpS, 0.0);
    }
    return vec4(sfEncode(clamp(al, 0.0, 1.0)), 0.0);
  }
  if (vKind > 1.5) return vec4(sfCap(sfEncode(SF_UPWELL * sfFloor(depth, sfSandTex(vWorld.xz, vWorld.y)))), 1.0);
  // wall: one-way column tint from v28's lit face to the fog, the same Kd
  float z = max(vSurfY - vWorld.y, 0.0);
  vec3 c = fog + (SF_FACE - fog) * sbRow(z, 2.0);   // one-way, the same spectral model
  if (z < 0.10) c = mix(c, SF_MENISCUS, 0.5);
  if (vWorld.y < -depth + bedRelief(vWorld.xz, depth) + 3.0) c = fog;
  vec3 v = normalize(eye - vWorld);
  // QM corner fix: the wall's face and corner line come from the undisplaced perimeter position
  // (vSea.xy, linear along each wall, exactly |x| = |z| on the shared corner edge), not from the
  // chop-shifted vWorld, whose diagonal wandered onto a face under a steep crest (the bright line).
  vec2 pW = vSea.xy;
  vec3 nW = abs(pW.x) > abs(pW.y) ? vec3(sign(pW.x), 0.0, 0.0) : vec3(0.0, 0.0, sign(pW.y));
  float fr = 0.08 * pow(1.0 - clamp(abs(dot(v, nW)), 0.0, 1.0), 5.0);
  // v29.1 see-through walls: an outer face seen from outside shows the inside of the block the way
  // the top surface shows the floor. Fresnel (Schlick, water 0.02) reflects the sky; the rest is
  // the view ray refracted into the water (n 1.333) and traced to the floor, the far wall or the
  // underside of the surface. Floor = the same seabedLut: its bottom radiance (floor model minus the
  // vertical upward path, G rho T(depth) / Td(depth)) carried over the in-water path s with the
  // one-way Td(s) toward the Jerlov IA fog, exactly as the wall column did. Reef patches come in
  // through sfAlbedo (the AO darkening); the inner faces (seen through the top) keep the v29 shading.
  float cosV = dot(v, nW);
  // v29.1 fix2 (Michael: walls a bit more see-through at depth): the in-wall view path counts
  // 0.65x, walls only (TUNED look value; the top surface and the floor seen from above are unchanged).
  const float WALL_PATH = 0.65;
  float wallA = 1.0;
  // v30.14 (Michael: wall colour must match the top): the wall uses the SAME water model as the top
  // surface (Beer-Lambert absorption aW, backscatter bb with the clarity slider, Rinf = 0.33 bb/(1.15 aW + bb),
  // the same 0.54 body scale, exposure and shoulder). The column seen sideways at depth z is the top's
  // deep-water body colour times the light left at z, exp(-Kd z).
  vec3 aW3 = (vec3(0.35, 0.06, 0.02) + WATER_TURB * vec3(0.03, 0.05, 0.12)) * WATER_SA;
  vec3 bb3 = (vec3(0.0005, 0.0011, 0.0024) + (0.0035 + 0.012 * WATER_TURB) * vec3(0.75, 1.75, 0.9)) * WATER_SB;
  vec3 Kd3 = 1.15 * aW3 + bb3;
  vec3 body3 = 0.54 * WATER_E * 0.33 * bb3 / (1.15 * aW3 + bb3);
  float sPathW = 0.0;
  if (cosV > 0.0 && !(vWorld.y < -depth + bedRelief(vWorld.xz, depth) + SF_SLICE)) {
    vec3 d = refract(-v, nW, 1.0 / 1.333);
    float H = max(abs(pW.x), abs(pW.y));
    vec3 q = vec3(clamp(vWorld.x, -H, H), vWorld.y, clamp(vWorld.z, -H, H));
    float tF = 1e9;
#ifdef ISLAND_ON
    if (uIsland > 0.5 && d.y < -1e-4) {
      float t = 0.05;
      for (int i = 0; i < 12; i++) {
        vec3 p = q + d * t;
        float bed = islandBed(p.xz);
        if (p.y <= bed + 0.08) { tF = t; break; }
        t += clamp((p.y - bed) / max(-d.y, 0.05), 0.35, 14.0);
        if (t > 500.0) break;
      }
      float tFlat = (-depth - q.y) / d.y;
      tF = min(tF, tFlat);
    } else
#endif
    if (d.y < -1e-4) tF = (-depth - q.y) / d.y;
    float tT = d.y > 1e-4 ? (vSurfY - q.y) / d.y : 1e9;
    float tX = abs(d.x) > 1e-4 ? (sign(d.x) * H - q.x) / d.x : 1e9;
    float tZ = abs(d.z) > 1e-4 ? (sign(d.z) * H - q.z) / d.z : 1e9;
    tX = tX < 1e-3 ? 1e9 : tX; tZ = tZ < 1e-3 ? 1e9 : tZ;
    float sW = min(tX, tZ), sPath = min(min(tF, tT), sW); sPathW = sPath;
    vec3 inside;
    if (tF <= min(tT, sW)) {
      vec3 hit = q + d * tF;
      // v29.1 fix (pink walls): the LUT rows are linear sRGB of a spectral model and go slightly
      // negative out of gamut (red Td(14 m) = -0.065), so the old T / Td ratio blew up (-64 in red).
      // No division now: the floor model itself at the equivalent vertical depth, half the down path
      // plus half the in-water view path (exactly the top view's floor when s = depth), clamped >= 0.
      inside = max(sfFloor(0.5 * (depth + WALL_PATH * sPath), sfAlbedo(hit.xz)), vec3(0.0));
      // v29.1 fix4: the real floor and coral are already drawn behind (opaque pass), so the wall is
      // blended over them: it lets through the in-wall path's luma transmittance (x (1 - Fresnel)).
      // Over sand the result stays the wall colour above; coral shows as itself, attenuated.
      vec3 trW = exp(-Kd3 * WALL_PATH * sPath);
      wallA = 1.0 - (1.0 - (0.02 + 0.98 * pow(1.0 - clamp(cosV, 0.0, 1.0), 5.0))) * clamp(dot(trW, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    } else if (tT <= sW) {
      inside = max(fog + (SF_FACE - fog) * sbRow(WALL_PATH * sPath, 2.0), vec3(0.0));   // the lit surface seen from below
    } else inside = max(fog, vec3(0.0));                     // the far wall: open water behind
    float F = 0.02 + 0.98 * pow(1.0 - clamp(cosV, 0.0, 1.0), 5.0);
    c = z < 0.10 ? mix(inside, SF_MENISCUS, 0.5) : inside;
    fr = F;
  }
  vec3 outc = sfEncode(mix(SF_UPWELL * c, SF_SKY, fr));
  // The bottom 0.4 m used to swap in the seabed lookup at the full depth. On a
  // 40 m seabed that sample is already deep-blue fog for every Secchi depth the
  // slider can set, so the strip stayed blue while the water above it changed.
  // It now uses the same absorption as the wall, and shows the sand only as far
  // as that light still gets through.
  vec3 trKZ = exp(-Kd3 * z);
  // The lower wall used to open onto the floor. The path sideways is only the
  // last fraction of a metre, so the wall went transparent and the seabed
  // lookup showed through. That lookup is already blue fog for every Secchi
  // depth once the bed is deep, which is the bright strip. Where the column
  // above has absorbed the light, keep this face opaque. Clearer shallow water
  // still sees through. The floor drawn from above is a different pass.
  float bedLine = -depth + bedRelief(vWorld.xz, depth);
#ifdef ISLAND_ON
  if (uHasIsland > 0.5) bedLine = max(bedLine, islandBed(vWorld.xz));
#endif
  // The drawn cut is 38% of the real water column (the vertex shader), so the
  // side is not a 40 m slab. The opaque sand on that cut is only a thin strip,
  // SF_SLICE (0.4 m) at its bottom, as before the cut: the bottom 22% of the
  // face was ~3.3 m of tan on a 40 m sea and read as a sand plinth under the
  // water. The strip sits at the visual cut, not at the full depth, so 0.4 m
  // of sand still reads here (the old full-depth lookup was already fog).
  // No ripple, no caustic.
  float yVis = mix(0.0, bedLine, 0.38);
  // Column colour from the surface down, and the floor shows through it.
  // The alpha follows the depth under the surface, not the long sideways
  // path, which had turned the whole cut opaque.
  float zCol = clamp(vSurfY - vWorld.y, 0.0, 16.0);
  vec3 lin3 = mix(vec3(0.28, 0.66, 0.70), vec3(0.012, 0.045, 0.13), smoothstep(0.4, 12.0, zCol));
  lin3 = mix(lin3, vec3(0.55, 0.72, 0.74), fr * 0.35);
  outc = pow(max(lin3, 0.0), vec3(1.0 / 2.2));
  wallA = clamp(1.0 - exp(-zCol * 0.085) * 0.84, 0.16, 0.92);
  // Strip top, softened over a few centimetres (and at least a pixel) so it
  // is not a hard line and does not shimmer when the strip gets thin on screen.
  float stripTop = yVis + min(SF_SLICE, max(vSurfY - yVis - 0.05, 0.0));
  float soft = max(0.04, 1.5 * fwidth(vWorld.y));
  float sandW = 1.0 - smoothstep(stripTop - soft, stripTop + 0.5 * soft, vWorld.y);
  if (sandW > 0.001) {
    // Sand in the wall plane. The horizontal coordinate is the undisplaced
    // perimeter, so the pattern stays put while the waves shift the vertices.
    // No orbital ripple: that wave stands up as columns on this cut.
    vec2 face = abs(pW.x) > abs(pW.y) ? vec2(pW.y, vWorld.y) : vec2(pW.x, vWorld.y);
    float n1 = sfNoise(face * 0.22);
    float n2 = sfNoise(face * vec2(0.9, 1.6) + 3.0);
    float n3 = sfNoise(face * 4.5 + 11.0);
    float g = n1 * 0.5 + n2 * 0.32 + n3 * 0.18;
    vec3 sand = vec3(0.76, 0.70, 0.60) * (0.72 + 0.55 * (g - 0.5));
    sand = mix(sand, sand * vec3(0.92, 0.90, 0.86), 0.35 * sfNoise(vec2(face.y * 0.55, 2.0)));
    // Darker toward the bottom of the strip (buried, out of the light), and
    // the water in front of the cut takes a little of the colour, so the
    // strip is damp sand under the column, not a bright dry edge.
    float bury = clamp((stripTop - vWorld.y) / SF_SLICE, 0.0, 1.0);
    vec3 linS = sand * mix(0.86, 0.6, bury);
    linS = mix(linS, linS * vec3(0.78, 0.9, 1.0), smoothstep(2.0, 14.0, zCol) * 0.6);
    vec3 sandC = sfCap(pow(max(linS, 0.0), vec3(1.0 / 2.2)));
    outc = mix(outc, sandC, sandW);
    wallA = mix(wallA, 1.0, sandW);
  }
  // See-through above the sand. The floor behind this face is drawn without
  // caustics, so a transparent wall does not turn them into vertical bars.
  float ce = max(abs(pW.x), abs(pW.y)) - min(abs(pW.x), abs(pW.y));
  float edgeW = max(length(vec2(dFdx(ce), dFdy(ce))), 1e-4);
  float corner = 1.0 - smoothstep(0.15 * edgeW, 0.7 * edgeW, ce);
  outc = mix(outc, vec3(0.86, 0.89, 0.92), clamp(corner * 0.62, 0.0, 0.62));
  if (corner > 0.65) wallA = 1.0;
  return vec4(max(outc, vec3(0.0)), wallA);
}
// Share of the surface pixel that lets the floor through: (1 - Fresnel) x the floor's own luma
// visibility exp(-2 Kd z) (the floor colour already carries the fog). 0 off the block.
float seafloorSeeThrough() {
  if (block < 0.5) return 0.0;
  float cosV = clamp(abs(normalize(eye - vWorld).y), 0.0, 1.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
  vec3 vis = exp(-2.0 * SF_KD * max(depth + vWorld.y, 0.0));
  return (1.0 - fresnel) * dot(vis, vec3(0.2126, 0.7152, 0.0722));
}
`;

const PHYS_Q = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const PQ = (k, d) => Number(PHYS_Q.get(k) ?? d).toFixed(4);
// v30c: water clarity is a hidden URL parameter (?clarity=1 clear tropical, default; 0 = turbid coastal)
// v30.23 clarity (modern-ui-spec Addendum A): Secchi depth Zsd (m) -> the water model. Green Kd = 1.7 / Zsd.
// 18.5..22.3 m runs the v30.8 turbidity t = (Kd - 0.076225) / 0.0785 up to t = 0.2 (22.30 m, t = 0, is the default K).
// Clearer than 22.3 m: t = 0 and absorption x sa, backscatter x sb = sa^2 (bluer, darker). Murkier than 18.5 m: t stays
// 0.2 and absorption x sa, backscatter x sa^1.3 (greener than the default, lighter), because the v30.8 t = 1 green
// measured hue 165 deg at 3-10 m (Addendum A bar 170-210). sa solves the green Kd exactly (bisection).
export const WATER_TCAP = 0.2;
export function waterFromZsd(z) {
  const K = 1.7 / Math.min(40, Math.max(3, z)), K0 = 0.076225, K1 = 0.076225 + 0.0785 * WATER_TCAP;
  if (K >= K0 && K <= K1) return { t: (K - K0) / 0.0785, sa: 1, sb: 1, K };
  const t = K < K0 ? 0 : WATER_TCAP, aG = 1.15 * (0.06 + 0.05 * t), bG = 0.0011 + 1.75 * (0.0035 + 0.012 * t), pw = K < K0 ? 2 : 1.3;
  let lo = 0.05, hi = 10; for (let i = 0; i < 60; i++) { const m = 0.5 * (lo + hi); if (aG * m + bG * Math.pow(m, pw) > K) hi = m; else lo = m; }
  const sa = 0.5 * (lo + hi); return { t, sa, sb: Math.pow(sa, pw), K };
}
const PHYS_DECL = `uniform highp sampler2D sceneTex;\nuniform mat4 viewProj;\nuniform float sceneOn;\nuniform highp sampler2D physK;\nuniform highp sampler2D physA;\nuniform vec2 foamOff;\nuniform float waterTurb;\nuniform vec2 waterSc;\nuniform highp sampler2D foamLace;\nuniform float foamT;\nuniform highp sampler2DArray physS;\nuniform float strandsOn;\nuniform float uAirOn;\nuniform float fineWin;\nuniform vec2 fineCam;\n#define WATER_TURB waterTurb\n#define WATER_SA waterSc.x\n#define WATER_SB waterSc.y\n#define SAND_ALBEDO vec3(${PQ("sandr", 0.26)}, ${PQ("sandg", 0.42)}, ${PQ("sandb", 0.34)})\n#define SAND_K ${PQ("sandk", 1.2)}\n#define WATER_E ${PQ("wate", 1.0)}\n#define PHYS_EXPO ${PQ("expo", 0.86)}\n#define PHYS_GLINT ${PQ("glk", 0.35)}\n#define FOAM_CELL ${PQ("fcell", 2.2)}\n#define WC_D ${PQ("wcd", 0.8)}\n#define WC_G ${PQ("wcg", 0.15)}\n#define WC_GE ${PQ("wcge", 0.8)}\n#define WC_HW ${PQ("wchw", 0.9)}\n#define WC_T0 ${PQ("wct0", 0.6)}\n#define WC_T1 ${PQ("wct1", 0.1)}\n#define LACE_FAR ${PQ("flf", PHYS_Q.get("fcov") === "0" ? 0.04 : 2.0)}\n#define FOAM_NEW ${PHYS_Q.get("fcov") === "0" ? "false" : "true"}\n#define LACE_YLO ${PQ("fly", 0.12)}\n#define LACE_SOFT ${PQ("lsoft", 0.8)}\n#define FAR_STREAK ${PQ("fstreak", 1.0)}\n#define FAR_MILK ${PQ("fmilk", PHYS_Q.get("fcov") === "0" ? 0.85 : 0.2)}\n#define WC_ON ${PHYS_Q.get("wc") === "0" ? "false" : "true"}\n#define FOAM_DEBUG ${PHYS_Q.get("foamdebug") === "1" ? "true" : "false"}\n`;
const PHYS_SEA = String.raw`
  // ===== v30c physical sea colour + physics foam (replaces the v30a grade and v30b foam look) =====
  float foamDetailG = 0.5; vec3 foamWhiteG = vec3(0.9); // v32: the foam's detail and lit white, for the shore surf
  {
    float spK = fftL / float(physN);
    vec2 pk = vSea.xy - foamOff;
    vec2 pw = vSea.xy + 2.6 * spK * vec2(valueNoise(pk / (3.7 * spK) + 11.0) - 0.5, valueNoise(pk / (3.7 * spK) + 47.0) - 0.5)
                     + 0.6 * spK * vec2(valueNoise(pk / (0.9 * spK) + 5.0) - 0.5, valueNoise(pk / (0.9 * spK) + 63.0) - 0.5);
    vec2 uk = (pw + 0.5 * fftL) / spK; vec2 k0 = floor(uk); vec2 kf = uk - k0; ivec2 kc = ivec2(k0);
    #define JAT(o) texelFetch(physA, ((kc + o) % physN + physN) % physN, 0).w
    // v30.8: the coverage grid is finer than the physics grid (512 at N=256)
    int kN = textureSize(physK, 0).x; float rK = float(kN) / float(physN);
    vec2 ukK = uk * rK; vec2 kK0 = floor(ukK); vec2 kfK = ukK - kK0; ivec2 kcK = ivec2(kK0);
    #define KAT(o) texelFetch(physK, ((kcK + o) % kN + kN) % kN, 0)
    // v30.6: cubic B-spline read of the coverage grid (C2, no crease at cell borders: no staircase edges)
    vec4 K = vec4(0.0);
    {
      vec2 t = kfK, t2 = t * t, t3 = t2 * t;
      vec2 w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0, w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0, w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0, w3 = t3 / 6.0;
      vec4 wx = vec4(w0.x, w1.x, w2.x, w3.x), wy = vec4(w0.y, w1.y, w2.y, w3.y);
      for (int j = 0; j < 4; j++) { vec4 row = vec4(0.0); for (int i = 0; i < 4; i++) row += wx[i] * KAT(ivec2(i - 1, j - 1)); K += wy[j] * row; }
    }
    float J = mix(mix(JAT(ivec2(0,0)), JAT(ivec2(1,0)), kf.x), mix(JAT(ivec2(0,1)), JAT(ivec2(1,1)), kf.x), kf.y);
    // surface density = carried coverage / area change (Jacobian): gathers where the surface converges
    float jv = clamp(J, 0.25, 2.5);
    float dense = clamp(K.x / jv, 0.0, 1.0), resid = clamp(K.y / jv, 0.0, 1.0), freshK = K.z;
    // v32 (A2): the same coverage 2.5 mips down (~3 m): a wide, soft under-foam mask for the
    // aeration skirt, the foam rim and the glint cut. Same sampler, another LOD.
    vec2 uvK = (ukK + 0.5) / float(kN);
    vec4 Kh = textureLod(physK, uvK, 2.5);
    float halo = clamp(1.6 * (0.4 * Kh.x + Kh.y) / jv, 0.0, 1.0);
    // --- water body: Beer-Lambert per channel, single-scattering reflectance, sand return ---
    vec3 aW = (vec3(0.35, 0.06, 0.02) + WATER_TURB * vec3(0.03, 0.05, 0.12)) * WATER_SA;       // absorption 1/m (R,G,B)
    vec3 bb = (vec3(0.0005, 0.0011, 0.0024) + (0.0035 + 0.012 * WATER_TURB) * vec3(0.75, 1.75, 0.9)) * WATER_SB; // backscatter 1/m
    vec3 Kd = 1.15 * aW + bb;                         // diffuse attenuation (mean cosine ~0.87)
    float H = block > 0.5 || depth < 200.0 ? max(depth + vWorld.y, 0.0) : 1e4;
#ifdef ISLAND_ON
    {
      // Shallow water over the bank only. Outside the bank, H is the open
      // sea, so the colour away from the island does not change.
      float openH = max(depth + vWorld.y, 0.0);
      float bedA = islandBed(vSea.xy);
      float localH = max(vWorld.y - bedA, 0.15);
      float toe = 1.0 - smoothstep(uIRbank - 6.0, uIRbank + 3.0, length(vSea.xy - uIslandC));
      H = mix(openH, min(openH, localH), toe);
    }
#endif
    vec3 Rinf = 0.33 * bb / (1.15 * aW + bb);         // deep-water irradiance reflectance
    // Ankle-deep water stays clear over the sand. Past the drop-off the
    // body shifts from turquoise toward deep blue. Clarity still drives Kd.
    float veil = smoothstep(0.2, 1.7, H);
    float plunge = smoothstep(6.0, 22.0, H);
    vec3 Rcol = mix(Rinf, Rinf * vec3(0.38, 0.55, 0.95), plunge);
    vec3 Rw;
    // v30.5 see-through: the view ray refracted at the wave normal (Snell, n = 1.33) is traced to the
    // seabed; the bed, rocks and corals under it come from the scene pass (bare albedo + rock height).
    // Light goes down to the hit (depth - rock height) and back along the refracted path, each
    // Beer-Lambert per channel; the water column in between adds its in-scatter Rinf (1 - T).
    vec3 dr = refract(-V, n, 1.0 / 1.33);
    float tb = H / max(-dr.y, 0.08);
    // v32 (C2): how strongly the live waves above the floor hit focus the sun there
    float focusC = causFocus(vWorld + dr * tb);
    vec4 cp = viewProj * vec4(vWorld + dr * tb, 1.0);
    vec2 suv = cp.xy / cp.w * 0.5 + 0.5;
    if (sceneOn > 0.5 && H < 200.0 && cp.w > 0.0 && all(greaterThan(suv, vec2(0.0))) && all(lessThan(suv, vec2(1.0)))) {
      vec4 sc = texture(sceneTex, suv);
      // The scene is cleared black. A ray that misses the floor used to read
      // the old pale clear colour and wear it as sand. On the open sea the
      // floor sits 40 m down and a wide view misses it often, so the pale
      // colour sat on the water. The island lifts the floor into that view,
      // which is why the film was gone with the island on.
      bool hit = sc.a > 0.001 || dot(sc.rgb, vec3(1.0)) > 0.04;
      if (hit) {
        vec3 alb = pow(sc.rgb, vec3(2.2)) * SAND_K;
        float hr = sc.a * 4.0;
        float down = max(H - hr, 0.0), path = tb * (down / max(H, 1e-3));
        vec3 T = exp(-Kd * (down + path));
        T = mix(vec3(1.0), T, veil);
        // Rock and coral (scene alpha > 0) are already shaded with their own
        // shadow in the cluster pass. The bed point under them would put the
        // rock in its own shadow a second time.
        float shF = sc.a > 0.001 ? 1.0 : shadowVis(vWorld + dr * tb);
        Rw = Rcol * (1.0 - T) + alb * T * floorLight(vWorld + dr * tb, n.xz, H, shF, focusC);
      } else {
        vec3 T = exp(-2.0 * Kd * H);
        T = mix(vec3(1.0), T, veil);
        float shF = shadowVis(vWorld + dr * tb);
        Rw = Rcol * (1.0 - T) + SAND_ALBEDO * T * floorLight(vWorld + dr * tb, n.xz, H, shF, focusC);
      }
    } else {
      vec3 T = exp(-2.0 * Kd * H);
      T = mix(vec3(1.0), T, veil);
      float shF = shadowVis(vWorld + dr * tb);
      Rw = Rcol * (1.0 - T) + SAND_ALBEDO * T * floorLight(vWorld + dr * tb, n.xz, H, shF, focusC);
    }
    // v30.6 aeration: bubbles are bright, non-absorbing scatterers; in a shallow layer (za ~ 2 m) they
    // raise the backscatter, so light comes back up after a short path: red still absorbed, green-blue
    // kept -> lighter turquoise, through the same Beer-Lambert. Mottled at metre scale.
    {
      // Underwater bubbles under the surface foam. v32: there is no separate 15-30 s air
      // store any more (K.w now holds the fold gate). The bubbles are a blurred copy of the
      // foam itself (the halo, ~3 m wide), so the milky turquoise skirt sits under and around
      // the white and dies with it, never lingering over empty sea. The lift is of this
      // water's own backscatter, so a weak crash stays a small lightening.
      float aer = halo * uAirOn;
      vec3 bbA = bb + aer * vec3(0.02, 0.16, 0.13) * (0.35 + 0.65 * aer);
      vec3 RinfA = 0.33 * bbA / (1.15 * aW + bbA);
      Rw = mix(Rw, RinfA, aer * 0.75 * (1.0 - exp(-2.0 * Kd * 2.0)));
    }
    float cosV = clamp(dot(n, V), 0.0, 1.0); float cosV0 = cosV;
    float Fr = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
    vec3 Rr = reflect(-V, n);
    vec3 skyLin = mix(vec3(0.36, 0.70, 0.68) * uSkyHorM, vec3(0.05, 0.35, 0.37) * uSkyZenM, sqrt(clamp(Rr.y, 0.0, 1.0))); // v30.24: sky-keyed (was fixed teal: no sunset colour)
    // v31.9: uWarm.rgb is the live horizon divided by the noon horizon, so this is the
    // actual low sky in linear light. uWarm.w is 0 at 11:00 and at night, and the noon
    // reflection above is left as it was. Fresnel (Fr) still weights it, strongest
    // when the view grazes the surface.
    vec3 horSky = uWarm.rgb * vec3(0.578, 0.723, 0.831);
    float lowSun = clamp(uWarm.w, 0.0, 1.0);
    vec3 skyWarm = mix(horSky, uSkyZenM * vec3(0.06, 0.16, 0.32), sqrt(clamp(Rr.y, 0.0, 1.0)));
    skyLin = mix(skyLin, skyWarm, lowSun);
    float seen = smoothstep(-0.02, 0.06, Rr.y);
    vec3 body = 0.54 * Rw * WATER_E * mix(vec3(1.0), uSunC, 0.2); // v30.24 item 11: the upwelling is lit by the sun's colour
    // The noon body is a dark intrinsic reflectance. Under a low sun the diffuse
    // light is the horizon, so the body goes dim and warm instead of staying a
    // teal that the exposure multiply then turns black.
    body = mix(body, body * vec3(0.72, 0.5, 0.32) + horSky * (0.05 + 0.035 * Fr), lowSun);
    // v30.6 top-down relief: (a) upwelling depends on how the facet faces the sun (more light enters
    // and returns on sun-facing faces), (b) wave-face subsurface glow on raised crests, strongest
    // looking toward the sun, (c) troughs darkened (less light reaches them, the next wave occludes).
    // A soft dim where the boat, buoy or reef blocks the sun. Foam is added later.
    body *= mix(0.42, 1.0, uBisect > 2.5 ? 1.0 : shadowFactorFast(vec3(vWorld.x, min(vWorld.y, -0.05), vWorld.z)));
    float sunFace = clamp(dot(n, L) / max(L.y, 0.25), 0.4, 1.8);
    body *= mix(1.0, sunFace, 0.65 * (1.0 - 0.8 * lowSun));
    float crestH = max(vWorld.y, 0.0), troughH = max(-vWorld.y, 0.0);
    body += WATER_E * vec3(0.004, 0.028, 0.026) * crestH * (0.4 + 0.6 * pow(max(dot(V, -L), 0.0), 4.0) + 0.4 * (1.0 - cosV0));
    body *= 1.0 - 0.38 * smoothstep(0.0, 1.6, troughH);
    // v32 (A3, Sea of Thieves / Atlas): turquoise crest subsurface light on the same mask as the
    // foam. Peak = height above still water, raised where the crest is squeezed (J < 1, the fold
    // the foam is seeded on); strongest looking toward the sun (pow(V . -L, 4)), and boosted under
    // the foam halo, so the turquoise sits under and around the white instead of beside it.
    {
      vec3 Lsn = normalize(L);
      float peak = clamp(crestH / (crestH + 0.8), 0.0, 1.0) * mix(0.5, 1.0, smoothstep(0.95, 0.55, J));
      float backL = pow(max(dot(V, -Lsn), 0.0), 4.0);
      float sss = peak * (backL * (1.0 + 2.5 * halo) + 0.3 * halo);
      body += WATER_E * vec3(0.003, 0.040, 0.034) * sss * (0.25 + 0.75 * clamp(Lsn.y, 0.0, 1.0)) * (1.0 - 0.6 * lowSun);
    }
    // v31.8 crest transmission. The old term was a hard strip: steepness had to clear
    // smoothstep(0.18, 0.5) and height had to clear smoothstep(0.2, 1.2), so the colour
    // switched on along a nearly horizontal elevation contour and sat apart from the face.
    // A heightfield still cannot see the true sheet thickness (3D curls come later). This
    // stand-in is continuous in face slope, height above still water, the physics lip
    // transmission (vThin), and the light direction. Flat water and troughs stay dark.
    // Foam is applied later and is not part of this term.
    {
      float slope = length(n.xz) / max(n.y, 0.12);
      float above = crestH / (crestH + 0.6);
      float geo = exp(-1.2 / max(slope, 0.05)) * above;
      // v32 (A3): the physics' thin-lip transmission and the foam halo are one crest mask here
      float sheet = geo + max(clamp(vThin, 0.0, 1.0), 0.6 * halo * above) * (1.0 - geo);
      vec3 Ls = normalize(L);
      float sunEl = clamp(Ls.y, 0.0, 1.0);
      float intoSun = pow(max(dot(V, -Ls), 0.0), 2.0);
      float through = (0.08 + sunEl * (0.15 + 0.95 * intoSun)) * sheet;
      body += WATER_E * vec3(0.004, 0.046, 0.038) * through;
    }
    // reflections aimed below the horizon see the next wave's face (the water body), not black
    vec3 lin = (1.0 - Fr) * body + Fr * mix(body * 1.4 + vec3(0.004, 0.012, 0.02), skyLin, seen);
    // v30.24 item 11 (Michael: the water must take on the sunrise / sunset colour, like the real photo): the warm horizon
    // sky reaches the water as a broad sheen (a rough sea reflects the low sky over a wide band of facets), strongest on
    // facets turned toward the sun and on glancing views; weight uWarm.w is 0 above ~24 deg of sun, so noon is unchanged.
    {
      // the low sky: orange at the horizon, pink a little above it (the colour the reflected rays see)
      vec3 warmC = mix(uWarm.rgb * vec3(0.58, 0.72, 0.83), vec3(0.80, 0.34, 0.42), 0.45 * smoothstep(0.02, 0.4, Rr.y));
      float toSun = max(dot(normalize(vec3(Rr.x, 0.0, Rr.z) + 1e-4), normalize(vec3(L.x, 0.0, L.z) + 1e-4)), 0.0);
      float wk = uWarm.w * ((0.025 + 1.6 * Fr) * (0.5 + 0.9 * toSun * toSun) + 0.3 * pow(toSun, 8.0) * (0.25 + 3.0 * Fr)) * (0.5 + 0.5 * seen);
      lin += warmC * wk;
    }
    // v30.24: the glitter takes the sun's colour (warm path at sunset). v32 (A4, Atlas): foam is
    // rough, so there is no mirror glint on it (the foam itself is drawn below).
    lin += vec3(glintRadiance) * PHYS_GLINT * uGlitM * (1.0 - smoothstep(0.05, 0.5, max(dense, resid)));
    lin *= PHYS_EXPO;
    lin = lin / (1.0 + 0.35 * lin);                   // gentle shoulder
    vec3 water = pow(max(lin, 0.0), vec3(1.0 / 2.2));
    // --- foam (v32: one soft mask, revealed through the lace; Sea of Thieves / Tessendorf 2020) ---
    // The 30 Hz sim (gpuSurface COVER_FS) is the mask: fresh whitewater, residual foam and the
    // crest tip. The detail texture (assets/foam-lace.png: a bubble-raft power diagram, R/G/B/A at
    // several scales) is shown where it beats the mask's threshold m = 1 - F, so as F decays the
    // foam erodes into lace and holes instead of fading as a uniform decal. The detail moves from
    // high-frequency bubble cells (fresh) to a low-frequency blotch-and-streak texture (old).
    vec2 qd = vSea.xy - foamOff;
    float fpM = max(length(dFdx(vWorld)), length(dFdy(vWorld)));
    float eyeD = length(eye - vWorld);
    vec2 dd = length(foamDrift) > 1e-4 ? normalize(foamDrift) : vec2(1.0, 0.0);
    float fFr = clamp(dense / 0.9, 0.0, 1.0), fRe = clamp(resid / 0.9, 0.0, 1.0);
    float Fm = max(fFr, fRe);
    float freshW = clamp(fFr / max(Fm, 1e-3), 0.0, 1.0);   // 1 = still being fed, 0 = residual only
    float oldK = 1.0 - freshW;
    float tip = clamp(freshK / 0.9, 0.0, 1.0);             // the lip while it is still folding
    float age = oldK;
    // v32 wind streaks: one world-locked field, long along the wind and narrow across it. Three
    // octaves (16 x 2.2 m, 7 x 0.8 m, 3 x 0.35 m); each fades to its mean once a pixel can no
    // longer resolve it, so it never shimmers and never scrolls. Old foam lines up along it.
    vec2 qw = vec2(dot(qd, dd), dot(qd, vec2(-dd.y, dd.x)));
    float oSm = 1.0 - smoothstep(0.5, 1.6, fpM), oSf = 1.0 - smoothstep(0.08, 0.3, fpM);
    float strk = 0.5 * valueNoise(vec2(qw.x / 16.0, qw.y / 2.2) + 71.0)
               + 0.3 * mix(0.5, valueNoise(vec2(qw.x / 7.0, qw.y / 0.8) + 77.0), oSm)
               + 0.2 * mix(0.5, valueNoise(vec2(qw.x / 3.0, qw.y / 0.35) + 83.0), oSf);
    // stretched along the wind as the foam ages (accumulated strain, capped)
    float stx = mix(1.0, clamp(0.84 * sqrt(max(jv, 0.36)), 0.78, 0.92), smoothstep(0.4, 0.85, age));
    vec2 qs = vec2(qw.x * stx, qw.y);
    const mat2 R2 = mat2(0.809, 0.588, -0.588, 0.809), R3 = mat2(0.276, -0.961, 0.961, 0.276);
    float o2 = 1.0 - smoothstep(0.03, 0.08, fpM), o3 = 1.0 - smoothstep(0.009, 0.025, fpM);
    vec4 t1 = texture(foamLace, qs / 40.0);
    vec4 t2 = texture(foamLace, R2 * qs / 12.0 + 0.31);
    vec4 t3 = texture(foamLace, R3 * qs / 3.5 + 0.67);
    // fresh: 0.6 m bubble pits (A) and the 0.17 m pits where resolved. The cell walls (G, a
    // sparse connected ridge network) are never a positive term: thresholded they drew a bright
    // Voronoi crackle on the water. They only open faint holes between bubbles (x 0.6 at most),
    // so the raft reads as holey bubbles, not filaments of light. Mean kept at ~0.27.
    float dHi = (0.5 * t2.a + 0.22 * mix(0.4, t3.a, o3)) * (1.0 - 0.4 * t2.g);
    // old: blurred pits (A at mip 2.5-3, blotches, no wall lines), the wind streaks and the 40 m
    // pit field (organic patches), mean as before
    float dLo = 0.3 * (texture(foamLace, qs / 10.0 + 0.43, 3.0).a + texture(foamLace, R2 * qs / 3.0 + 0.11, 2.5).a) + 0.4 * strk + 0.1 * t1.a;
    float detail = mix(clamp((dHi - 0.04) / 0.6, 0.0, 1.0), clamp((dLo - 0.22) / 0.55, 0.0, 1.0), smoothstep(0.2, 0.8, oldK));
    foamDetailG = detail;
    // A1: opacity through the mask. The edge is wider on old foam and where the halo is thin
    // (the rim of a patch, a lone fleck), so no patch has a hard cut-out outline.
    float Fn = pow(Fm, 0.8);
    float mT = 1.0 - Fn;
    float wE = mix(0.05, 0.15, oldK) + 0.1 * (1.0 - smoothstep(0.15, 0.6, halo)) + fwidth(detail);
    float aN = smoothstep(mT - wE, mT + wE, detail) * smoothstep(0.0, 0.1, Fm);
    aN = max(aN, smoothstep(0.3, 0.8, tip) * smoothstep(0.1, 0.35, detail));  // the folding lip is nearly closed
    // A5: once the lace is sub-pixel or past ~150 m, the foam is the mip-averaged coverage
    // (Dupuy-Bruneton): thresholding a ~uniform detail by 1 - F covers about F of the area,
    // so the far opacity is the coverage at the pixel's footprint, a smooth whitening.
    float lodK = log2(max(fpM * float(kN) / fftL, 1.0));
    vec4 Kf = textureLod(physK, uvK, lodK + 0.5);
    float FmF = clamp(max(Kf.x, Kf.y) / (0.9 * jv), 0.0, 1.0);
    float farW = max(smoothstep(0.35, 1.0, fpM), smoothstep(120.0, 180.0, eyeD));
    float fa = clamp(mix(aN, 0.9 * pow(FmF, 0.8), farW), 0.0, 1.0);
    float freshV = mix(freshW * smoothstep(0.1, 0.5, fFr), clamp(Kf.x / max(max(Kf.x, Kf.y), 1e-3), 0.0, 1.0), farW);
    // Lighting: wrap-Lambert on the wave normal plus the foam height bump (B channel), sun and sky.
    float e1 = 1.0 / 1024.0;
    float bh = t2.b, bhx = texture(foamLace, R2 * qs / 12.0 + 0.31 + vec2(e1, 0.0)).b, bhz = texture(foamLace, R2 * qs / 12.0 + 0.31 + vec2(0.0, e1)).b;
    // foam volume: the roller stands ~0.03 U m proud of the surface, older foam is a thin raft
    float rollH = clamp(0.1 * length(foamDrift), 0.2, 0.8);
    float hf = (0.3 * smoothstep(0.1, 0.8, fFr) + 0.15 * smoothstep(0.05, 0.5, fRe)) * rollH + 0.08 * bh * o2;
    vec3 pF = vWorld + vec3(0.0, hf, 0.0);
    vec3 nR = normalize(cross(dFdx(pF), dFdy(pF))); if (nR.y < 0.0) nR = -nR;
    vec3 nF = normalize(mix(n, nR, 0.85) + vec3(bh - bhx, 0.0, bh - bhz) * 3.0 * o2);
    float sunT = clamp(dot(nF, L), 0.0, 1.0);
    vec3 foamC = mix(vec3(0.84, 0.87, 0.89), vec3(0.98, 0.99, 1.0) * mix(vec3(1.0), uSunC, 0.45), 0.55 + 0.45 * sunT);
    foamC = mix(foamC, foamC * (0.55 + 0.9 * uWarm.rgb * vec3(0.58, 0.72, 0.83)), 0.3 * uWarm.w); // a pink/orange cast at sunset
    foamC *= 1.0 - 0.22 * t3.a * o3 * freshV;                       // 1-3 cm bubble pits
    // A4: foam albedo ~0.45 for aged foam (Koepke 1984 via Dupuy-Bruneton), the active crest
    // brighter (~0.85). foamC above is the lit colour at fresh albedo, in display space.
    foamWhiteG = foamC; // the surf is fresh foam
    float alb = mix(0.45, 0.85, max(freshV, tip));
    foamC *= pow(alb / 0.85, 1.0 / 2.2);
    // A2: white -> turquoise skirt -> deep blue. Thin residual foam shows the bubbly water under
    // it, and the halo tints the water around and between the foam.
    vec3 milkyW = mix(water, vec3(0.60, 0.86, 0.85), 0.6);
    vec3 foamCol = mix(mix(milkyW, foamC, 0.55), foamC, max(freshV, smoothstep(0.5, 0.9, mix(Fm, FmF, farW))));
    vec3 waterB = mix(water, milkyW, 0.4 * halo * (1.0 - fa));
    color = mix(waterB, foamCol, fa);
    if (FOAM_DEBUG) color = vec3(clamp(max(dense, 0.6 * resid), 0.0, 1.0));
    // 3 = (dW young strands, Sd.r old strands, fpM / 2 m)
    if (SAND_DBG2) color = texture(sceneTex, gl_FragCoord.xy / vec2(textureSize(sceneTex, 0))).rgb;
  }
`;
function toEs3(source, stage) {
  let out = source.replace(/^#extension[^\n]*\n/gm, "");
  out = out.replace(/\btexture2D\b/g, "texture");
  if (stage === "vertex") {
    out = out.replace(/\battribute\b/g, "in").replace(/\bvarying\b/g, "out");
  } else {
    out = out.replace(/\bvarying\b/g, "in").replace(/\bgl_FragColor\b/g, "fragColor");
    out = "out highp vec4 fragColor;\n" + out;
  }
  return "#version 300 es\n" + out;
}

// Reflection tuning, kept together.
// v30a colour pass (GAD): reflection sky, body colours, facing ramp and display grade.
const V30_DEFS = [
  "#define REFL_HORIZON 0.6627, 0.7961, 0.8784", "#define REFL_ZENITH 0.3098, 0.5608, 0.7686", // GAD: #A9CBE0 -> #4F8FC4 (reflection only; backdrop stays white)
  "#define BODY_DEEP 0.004, 0.018, 0.035", "#define BODY_FACE 0.05, 0.16, 0.26", "#define BODY_LIP 0.10, 0.32, 0.44",
  "#define FACE_W 0.15", "#define AWAY_REFL 0.2",
  "#define GRADE_P 0.1023", "#define GRADE_M 0.39", "#define GRADE_G 1.0", "#define GRADE_CAP 0.77", "#define GRADE_KNEE 0.56",
  "#define GRADE_HUE 204.0", "#define SUNW_LO 0.0", "#define SUNW_HI 0.7", "#define SUNW_VY0 0.15", "#define SUNW_VY1 0.45", "#define SUN_REFL_K 0.5", "#define SUN_TROUGH_K 0.25", "#define SUN_GLINT_K 0.2", "#define SUN_EXPO_K 0.72", "#define PATCH_ROT 1.0", "#define GRADE_WK 3.5", "#define GRADE_SAT 0.82", "#define GRADE_VMAX 0.898",
].join("\n") + "\n";
const REFLECT_DEFS = `#define SKY_ZENITH ${SKY_ZENITH_C.join(", ")}\n` + V30_DEFS + (STORM ? "#define SKY_SEEN_LO -0.40\n#define SKY_SEEN_HI -0.20\n" : "#define SKY_SEEN_LO -0.04\n#define SKY_SEEN_HI 0.14\n") + "#define REFLECT_GAIN 2.4\n";
// Finer patch distance fade (metres from the eye), kept together. Set by measurement:
// at full strength out to 30-40 m its ripples drove the sun glint into frame-to-frame
// sparkle there (sub-pixel normals, and glint is too nonlinear for mips to average);
// fading over 20-40 m leaves the water from 30 m out moving exactly as without it.
const FINE_DEFS = "#define FINE_FADE_NEAR 20.0\n#define FINE_FADE_FAR 40.0\n";
// Sun lighting, kept together. These are light levels, not roughness (roughness comes
// only from roughness.js). SUN_IRRADIANCE is the display exposure of the sun's glint.
// Physically the sun is about 18 times the sky's radiance (about 11.5 in these units);
// no screen shows that, so the glint is exposed down to 0.25 (the art direction's call
// from the v25 sweep). The lobe keeps its physical shape and normalisation, so its peak
// falls as the slope spread widens. Water luminance rolls off softly from WATER_KNEE (110)
// toward WATER_MAX (135), so aged foam (about 140-165) and fresh foam (195) stay brighter.
// Beyond about 60 m the shoulder starts at WATER_KNEE_FAR (100) instead (see the glint display).
const LIGHT_DEFS = `#define UPWELL ${UPWELL.toFixed(4)}\n#define SKY_HORIZON_C ${SKY_HORIZON.join(", ")}\n#define SUN_IRRADIANCE ${STORM ? "0.0" : "0.25"}\n` + "#define WATER_KNEE 0.4314\n#define WATER_MAX 0.5294\n#define WATER_KNEE_FAR 0.3922\n#define SHOULDER_BLEND_NEAR 50.0\n#define SHOULDER_BLEND_FAR 70.0\n#define SHOULDER_FAR_N 3.0\n#define LUMA 0.2126, 0.7152, 0.0722\n#define GLINT_FADE_REF 0.03\n#define GLINT_FADE_NEAR 15.0\n#define GLINT_FADE_FAR 30.0\n";
// Translucency (subsurface light through thin crests), kept together. vThin is the
// physics' transmission through the crest (surface.js, TRANSLUCENCY): 0 on flat or thick
// water. Light reaching the eye through it: the sun (TRANSL_SCATTER x irradiance x a
// Henyey-Greenstein phase of (V . -L), g = TRANSL_G, strongest with the sun behind the
// wave) and the bright sky behind the crest (TRANSL_SKY). Same constants at every wind.
// The water is pulled toward a lighter green-teal (TRANSL_COLOR, luminance about 130),
// saturating softly at TRANSL_MAX, so it
// never turns white and the body stays teal; glint and foam are drawn on top as before.
// GAD 18:03: TRANSL_COLOR is back to the v28 teal and is NOT part of the 265 hue rotation (the rotated
// 0.3177, 0.5570, 0.7756 drew blue slivers ~(66,115,160) on the lips).
const TRANSL_DEFS = "#define TRANSL_COLOR 0.22, 0.60, 0.50\n#define TRANSL_G 0.7\n#define TRANSL_SCATTER 80.0\n#define TRANSL_SKY 6.5\n#define TRANSL_TIP 0.8\n#define TRANSL_MAX 0.85\n";
const LACE_DEFS = `#define BUMP_ON ${SEA_PARAMS.get("bump") === "0" ? "0.0" : "1.0"}\n#define FOAM_BUMP 2.2\n` + "#define FOAM_WHITE 0.94\n#define FOAM_HOLE_M 1.2\n#define FOAM_HOLE_AT 0.82\n#define FOAM_OLD_DIM 0.75\n#define FRESH_LO 0.1\n#define FRESH_HI 0.38\n#define CREST_ALPHA 0.9\n#define FRESH_GRAIN 0.25\n#define DIM_LACE 0.3\n#define DIM_DENSE 0.64\n#define LACE_HOLE_BASE 0.06\n#define LACE_HOLE_DENSE 0.08\n#define LACE_HOLE_SOFT 0.08\n#define LACE_CAP 0.62\n#define LACE_WARP 0.5\n#define LACE_EDGE 0.3\n#define LACE_RIDGE_LO 0.65\n#define LACE_RIDGE_HI 0.9\n#define LACE_MASK_LO 0.6\n#define LACE_SHIFT_X 2.5\n#define LACE_SHIFT_Y 1.5\n#define LACE_DASH 0.45\n#define LACE_SHOULDER 0.8\n#define LACE_FINE_LINE 0.35\n#define LACE_SH_LO 0.45\n#define LACE_SH_HI 0.65\n#define LACE_MASK_HI 0.95\n" + `#define LACE_GRAIN ${Number(new URLSearchParams(globalThis.location?.search ?? "").get("lacegrain") ?? 0.8).toFixed(3)}\n`;

// Sun glitter and sky reflection spread by the physical slope distribution (roughness.js).
// The glint adds to the water and rolls off softly below foam brightness.
// Crest glow stays on the thin lip. Neither goes white.
const FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying float vBroken;
varying float vLip;
varying float vFoam;
varying float vFlag;
varying float vRecent;
varying float vFade;
varying float vAge;
varying float vAhead;
varying float vRun;
varying float vThin;
varying vec3 vNormal;
varying vec3 vWorld;
uniform vec3 lightDir;
uniform vec3 eye;
uniform float depth;
uniform vec3 uSkyHorM; uniform vec3 uSkyZenM; uniform vec3 uGlitM; // v30.23 sky keys (1 at 11:00)
uniform vec3 uSunC; uniform vec4 uWarm; uniform float uClarK; // v30.24: sun light colour / noon, warm horizon / noon + weight; clarity Kd scale for the seabed LUT
${BED_GLSL}${RIPPLE_GLSL}
uniform float showFlags;
uniform vec2 travel;
// The chop band's fine patch: slopes (dh/dx, its quarter-period partner, dh/dz, its
// partner), tiled every chopSize metres and sampled at the water's own position.
uniform sampler2D chopTex;
uniform float chopSize;
uniform float chopScale;
uniform float chopOn;
// The band's second, finer patch (about 1.05 to 0.25 m waves), its own texture and tile.
uniform sampler2D fineTex;
uniform float fineSize;
uniform float fineScale;
uniform float fineOn;
// Roughness (roughness.js): the sub-grid mss from the physics, each patch's resolved mss,
// and the share of each patch's slope variance its texture keeps at mip level 0..7.
uniform float subMss;
uniform float chopMss;
uniform float fineMss;
uniform float chopKept[8];
uniform float fineKept[8];
uniform float chopN;
uniform float fineN;
uniform float anisoMax;

// Cheap value noise. It is looked up at the water's own position (grid point plus
// the choppy shift), so it rides with the water and never depends on the camera.
float hash(vec2 p) {
  vec3 q = fract(vec3(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
float detailL0(vec2 qw) { return 1.0 - smoothstep(0.5, 1.0, max(qw.x / 1.6, qw.y / 0.12)); }
float valueNoise(vec2 p) {
  p = mat2(0.8, 0.6, -0.6, 0.8) * p; // v30b.4: lattice rotated 37 deg off the world/grid axes (no grid-aligned noise edges)
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
vec4 patchFetch(sampler2D tex, float size, float scale, vec2 p) {
  return (texture2D(tex, p / size) * 255.0 - 128.0) * (scale / 127.0);
}
// One sampling of the patch with every wave's phase turned by psi: the real part of
// z * e^(-i psi). A phase turn keeps each wave's length, direction and speed.
vec2 chopTurn(vec4 c, float psi) {
  float cs = cos(psi);
  float sn = sin(psi);
  return vec2(c.x * cs + c.y * sn, c.z * cs + c.w * sn);
}
// Fine chop slope at world point p. The patch repeats every chopSize metres. Three
// samplings of it, each offset by an odd fraction of a tile, are each phase-turned by
// an angle that wanders slowly over the sea, then blended with weights from slow
// world-space masks. The weights' squares sum to 1 and the three are uncorrelated, so
// the ripple strength is the same everywhere, while the pattern at any point differs
// from the same point one tile away (slopes there correlate about 0.05; where ripples
// are a little stronger or weaker correlates about 0.3). Nothing scales or rotates the
// patch, so every ripple keeps the length, heading and speed the physics gave it.
//
// The same function serves the finer patch: its noise is scaled with its tile (s =
// tile / 56 m, so every mask and phase wander keeps the same size relative to the tile)
// and shifted by o, so its masks are independent of the first patch's. Measured offline
// on the 13.5 m patch: slopes one tile apart correlate 0.00 to 0.08, ripple strength
// about 0.3, the same as the first patch. For the first patch s = 1, o = 0, exactly as before.
vec4 patchRot(sampler2D tex, float size, float scale, vec2 p, float ang) {
  float cr = cos(ang), sr = sin(ang);
  vec2 q = vec2(cr * p.x - sr * p.y, sr * p.x + cr * p.y);
  vec4 g = patchFetch(tex, size, scale, q);   // (slope x re, im, slope z re, im)
  return vec4(cr * g.xy + sr * g.zw, -sr * g.xy + cr * g.zw);
}
vec2 patchSlope(sampler2D tex, float size, float scale, vec2 p, float s, float o) {
  float psiA = 6.2832 * (valueNoise(p / (41.0 * s) + (11.0 + o)) + 0.6 * valueNoise(p / (17.0 * s) + (23.0 + o)));
  float psiB = 6.2832 * (valueNoise(p / (37.0 * s) + (41.0 + o)) + 0.6 * valueNoise(p / (15.0 * s) + (53.0 + o)));
  float psiC = 6.2832 * (valueNoise(p / (33.0 * s) + (71.0 + o)) + 0.6 * valueNoise(p / (13.0 * s) + (83.0 + o)));
  float wA = valueNoise(p / (23.0 * s) + (101.0 + o)) + 0.02;
  float wB = valueNoise(p / (23.0 * s) + (117.0 + o)) + 0.02;
  float wC = valueNoise(p / (23.0 * s) + (131.0 + o)) + 0.02;
  float total = wA + wB + wC;
  // v30a banding fix: the three fetches also read the patch through rotated coordinates
  // (0, +PATCH_ROT, -PATCH_ROT rad), so the patch grid's axis-aligned features no longer line up
  // into a cross-hatch; each slope is turned back by the same angle (grad_p = R^T grad_q).
  vec2 a = chopTurn(patchFetch(tex, size, scale, p), psiA);
  vec2 b = chopTurn(patchRot(tex, size, scale, p + vec2(0.383, 0.617) * size, PATCH_ROT), psiB);
  vec2 c = chopTurn(patchRot(tex, size, scale, p + vec2(0.711, 0.229) * size, -PATCH_ROT), psiC);
  return (sqrt(wA) * a + sqrt(wB) * b + sqrt(wC) * c) / sqrt(total);
}
vec2 fineSlope(vec2 p) {
  return patchSlope(chopTex, chopSize, chopScale, p, 1.0, 0.0);
}
vec2 finerSlope(vec2 p) {
  return patchSlope(fineTex, fineSize, fineScale, p, fineSize / 56.0, 200.0);
}
// The mip level the GPU samples a patch at (texel footprint, with anisotropic filtering
// taking up to anisoMax samples along the long axis), and the slope variance kept there.
float mipLevel(vec2 uv, float n) {
#ifdef HAS_DERIV
  vec2 dx = dFdx(uv * n);
  vec2 dy = dFdy(uv * n);
  float a = dot(dx, dx);
  float b = dot(dy, dy);
  float major = max(a, b);
  float minor = min(a, b);
  return clamp(0.5 * log2(max(max(major / (anisoMax * anisoMax), minor), 1e-8)), 0.0, 7.0);
#else
  return 0.0;
#endif
}
float keptAt(float kept[8], float level) {
  float r = 0.0;
  for (int i = 0; i < 8; i += 1) r += kept[i] * max(0.0, 1.0 - abs(level - float(i)));
  return r;
}
// Sky seen in one facet: Schlick Fresnel for water (0.02 head-on, rising toward 1 at a
// glancing angle); the view ray reflected off the facet looks up a sky gradient (the
// horizon is exactly the sky we draw, deepening toward the zenith); rays reflected just
// above the horizon mostly meet the next wave instead of the sky (skySeen). Returns the
// reflected colour already weighted; w gets the weight.
vec3 skyFacet(vec3 nn, vec3 V, out float w) {
  // v30a: the reflection sees a brighter, bluer sky than the drawn backdrop (GAD v30a).
  vec3 skyHorizon = vec3(REFL_HORIZON) * uSkyHorM;
  vec3 skyZenith = vec3(REFL_ZENITH) * uSkyZenM;
  float cosV = clamp(dot(nn, V), 0.0, 1.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
  vec3 R = reflect(-V, nn);
  vec3 skyR = mix(skyHorizon, skyZenith, sqrt(clamp(R.y, 0.0, 1.0)));
  w = min(fresnel * smoothstep(SKY_SEEN_LO, SKY_SEEN_HI, R.y) * REFLECT_GAIN, 1.0);
  return skyR * w;
}
${LOBE_GLSL}
uniform vec4 uBoat;
uniform vec3 uShadeSun;
float boatShade(vec2 xz, vec3 L) {
  if (uBoat.w < 0.5 || L.y < 0.03) return 1.0;
  float horiz = max(length(L.xz), 1e-4);
  vec2 s = L.xz / horiz;
  float elev = max(L.y, 0.04);
  vec2 ctr = uBoat.xy - s * (max(uBoat.z, 0.05) * horiz / elev);
  vec2 q = xz - ctr;
  float along = dot(q, s);
  float cross = q.x * s.y - q.y * s.x;
  float len = 5.2 * horiz / elev;
  float e = along * along / (len * len) + cross * cross / (2.3 * 2.3);
  return mix(0.62, 1.0, smoothstep(0.35, 1.15, sqrt(e)));
}
void main() {
  vec3 nBig = normalize(vNormal);
  { vec3 rp = ripple(vWorld.xz); nBig = normalize(nBig + vec3(-rp.y, 0.0, -rp.z) * nBig.y * 1.35); } // v30.27 ripples: a little more slope so the ring reads in wind sea
  vec3 L = normalize(lightDir);
  // Fine chop shades the water per pixel on top of the big grid's normal (heights add,
  // so slopes add). It fades out with distance, after the texture's own mip levels
  // have already averaged away what a pixel cannot hold.
  float distEye = length(eye - vWorld);
  float chopFade = chopOn * (1.0 - smoothstep(160.0, 320.0, distEye));
  vec2 fs = fineSlope(vWorld.xz) * chopFade;
  // The finer patch adds its slopes to the normal only (per pixel). Its mips average
  // away waves a pixel cannot hold, and it is faded out completely by FINE_FADE_FAR
  // (40 m), before its longest waves (1.05 m) shrink to a few pixels from the 6.8 m camera.
  // Foam shaping below keeps using the first patch alone (fs), so foam cover is unchanged.
  float fineFade = fineOn * (1.0 - smoothstep(FINE_FADE_NEAR, FINE_FADE_FAR, distEye));
  // Always sampled (no branch), so its mip level comes from well-defined derivatives.
  vec2 ff = finerSlope(vWorld.xz) * fineFade;
  float ny = max(nBig.y, 0.2);
#ifdef HAS_DERIV
  vec2 sBig = nBig.xz / ny;
  vec2 gridDx = dFdx(sBig);
  vec2 gridDy = dFdy(sBig);
#else
  vec2 gridDx = vec2(0.0);
  vec2 gridDy = vec2(0.0);
#endif
  // Inside the sun's glint, from GLINT_FADE_NEAR to GLINT_FADE_FAR out, the ripple
  // patches leave the normal in proportion to how strong the glint is there. At that
  // range their short, fast waves are only a few pixels across and the glint turns
  // them into frame-to-frame blinking; their slope does not vanish, it moves into the
  // glint lobe (the kept fractions below), so the roughness budget and the glint's
  // energy are unchanged. The weight comes from the glint the big grid's normal alone
  // would show with no patch slope kept; outside the glint nothing changes, and foam
  // keeps using the unfaded patch slope (fs).
  vec3 V = normalize(eye - vWorld);
  vec3 H = normalize(L + V);
  vec3 nGrid = normalize(nBig);
  float mssGrid = max(lobeVariance(subMss, chopMss, 0.0, fineMss, 0.0, gridDx, gridDy), 1e-4);
  float cbG = max(dot(nGrid, H), 1e-3);
  float cbG2 = cbG * cbG;
  float glintGrid = step(0.0, L.y) * SUN_IRRADIANCE * (0.02 + 0.98 * pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0))
                  * exp(-((1.0 - cbG2) / cbG2) / mssGrid) / (3.14159265 * mssGrid)
                  / (4.0 * max(dot(nGrid, V), 0.05) * cbG2 * cbG2);
  float glintWeight = glintGrid / (glintGrid + GLINT_FADE_REF);
  float glintFade = 1.0 - glintWeight * smoothstep(GLINT_FADE_NEAR, GLINT_FADE_FAR, distEye);
  vec2 fsN = fs * glintFade;
  vec2 ffN = ff * glintFade;
  vec3 n = normalize(vec3(nBig.x / ny - fsN.x - ffN.x, 1.0, nBig.z / ny - fsN.y - ffN.y));
  float light = clamp(dot(n, L), 0.0, 1.0);
  float lightFoam = clamp(dot(normalize(vec3(nBig.x / ny - fs.x - ff.x, 1.0, nBig.z / ny - fs.y - ff.y)), L), 0.0, 1.0);
  float lightBig = clamp(dot(nBig, L), 0.0, 1.0);
  float shade = 0.12 + 0.88 * light;
  // palette 10:01 / GAD 10:14: v28 water-body constants rotated in CIELAB to hue 265 (Forel-Ule 2-3),
  // same L*, C*. TUNED value: the model's own upwelling hue (286.6) plus the v28 sky reflection gives
  // 284.8 in the standard view (> 10 deg off 265), so the rotation is kept (hue-test-upwelling.png).
  // v30a (GAD colour pass): hue ~200 body, no 265 rotation. Slopes turned away from the camera
  // fall to a blue-black; slopes facing it show the lit body. The sky reflection (below) lights
  // the sky-tilted slopes; a final grade (V30_GRADE) sets the display range and hue.
  vec3 trough = vec3(BODY_DEEP);
  vec3 face = vec3(BODY_FACE);
  // relative to flat water at this view angle, so distance (grazing) does not read as "away"
  float facing = smoothstep(-FACE_W, FACE_W, dot(n, V) - V.y);
  vec3 color = mix(trough, face, shade * facing);

  float lip = smoothstep(0.45, 0.85, vLip);
  float depthFade = clamp(10.0 / max(depth, 2.0), 0.2, 1.0);
  color = mix(color, vec3(BODY_LIP), lip * depthFade);
  color *= UPWELL; // storm: overcast upwelling (Kasten & Czeplak 1980); 1 under the clear sky

  // Translucency: subsurface light through thin crests (see TRANSL_DEFS).
  // Sun: irradiance (the glint's SUN_IRRADIANCE exposure) x cos(incidence) on the crest
  // x the in-water scattering phase toward the eye, Henyey-Greenstein with g = TRANSL_G
  // (forward-peaked like sea water: about 10 times stronger with the sun behind the wave
  // than behind the camera, never zero). Sky: the bright sky behind the crest, diffusely.
  float mu = dot(V, -L);
  float hg = (1.0 - TRANSL_G * TRANSL_G) / pow(1.0 + TRANSL_G * TRANSL_G - 2.0 * TRANSL_G * mu, 1.5);
  // The sunlight enters through the lip's sun-side back face (for a thin symmetric lip, the
  // visible face mirrored about the vertical), so the incidence is taken there, not on
  // flat water: strongest when the sun stands behind the lip.
  float cosBack = max(dot(normalize(vec3(-n.x, n.y, -n.z)), L), 0.0);
  float sunIn = step(0.0, L.y) * SUN_IRRADIANCE * cosBack * hg;
  // Transmission x light, saturating softly (1 - e^-x) toward TRANSL_MAX: never white.
  // Applied just before the foam (below), as part of the water body.
  // Sky part saturates per lip; the sun part is kept linear (its pixel average is taken below
  // before it saturates), so a bright, sub-pixel sunlit lip can still fill its pixel.
  float transl = TRANSL_MAX * (1.0 - exp(-vThin * TRANSL_SKY));
  float translSun = vThin * TRANSL_SCATTER * sunIn;

  // Slope spread under this pixel (roughness.js): the sub-grid mss from the physics plus
  // every resolved slope this pixel does not show through n (faded or mip-averaged patch
  // slopes, and the big grid's slope changing inside the pixel). Together they add up to
  // the Cox and Munk total at the live wind, so the light spreads as the wind rises.
  float chopKeptHere = chopFade * chopFade * glintFade * glintFade * keptAt(chopKept, mipLevel(vWorld.xz / chopSize, chopN));
  float fineKeptHere = fineFade * fineFade * glintFade * glintFade * keptAt(fineKept, mipLevel(vWorld.xz / fineSize, fineN));
  float mss = max(lobeVariance(subMss, chopMss, chopKeptHere, fineMss, fineKeptHere, gridDx, gridDy), 1e-4);

  // Reflected sky, averaged over that Gaussian spread of facet slopes around n: four
  // sigma points at plus and minus sqrt(mss) along each axis (each axis carries mss / 2),
  // which keeps the mean and the variance of the spread exactly. With no spread this is
  // the single mirror reflection of n. Sky colours stay off white, the body stays teal.
  vec2 g = n.xz / n.y;
  float spread = sqrt(mss);
  float w0; float w1; float w2; float w3;
  vec3 sky = skyFacet(normalize(vec3(g.x + spread, 1.0, g.y)), V, w0)
           + skyFacet(normalize(vec3(g.x - spread, 1.0, g.y)), V, w1)
           + skyFacet(normalize(vec3(g.x, 1.0, g.y + spread)), V, w2)
           + skyFacet(normalize(vec3(g.x, 1.0, g.y - spread)), V, w3);
  // v30a: slopes turned away from the camera keep only AWAY_REFL of the sky (blue-black backs).
  float awayK = mix(AWAY_REFL, 1.0, facing);
  // v30a.3 (GAD sun-view ruling): looking toward the sun at a glancing angle, the sky reflection
  // is softened (SUN_REFL_K, troughs SUN_TROUGH_K) so the sun band keeps dark troughs and wave
  // detail instead of washing out flat. Zero away from the sun's bearing and at steeper views.
  vec2 lookXZ = -V.xz / max(length(V.xz), 1e-4);
  float sunward = smoothstep(SUNW_LO, SUNW_HI, dot(lookXZ, normalize(L.xz + vec2(1e-5))))
                * (1.0 - smoothstep(SUNW_VY0, SUNW_VY1, V.y));
  awayK *= mix(1.0, mix(SUN_TROUGH_K, SUN_REFL_K, facing), sunward);
  float reflShare = 0.25 * (w0 + w1 + w2 + w3) * awayK;
  vec3 shadeSun = uShadeSun.y > 0.02 ? normalize(uShadeSun) : lightDir;
  float bodySh = boatShade(vWorld.xz, shadeSun);
#ifdef ISLAND_ON
  {
    // Shadow only where the bed has actually risen. Open water, including
    // the border of the heightmap, stays at 1 (no dark square).
    float risen = smoothstep(0.4, 5.0, depth - max(0.0, -vCoast.r));
    bodySh *= mix(1.0, mix(0.9, 1.0, clamp(vCoast.b, 0.0, 1.0)), risen);
  }
#endif
  color *= bodySh;
  color = color * (1.0 - reflShare) + 0.25 * sky * awayK;

  // Sun glitter (Cox and Munk 1954): radiance = E * rho * p(facet) / (4 cos(view) cos^4(beta)),
  // with p the Gaussian slope distribution of mss above, centred on this pixel's normal,
  // beta the tilt from n of the facet that mirrors the sun into the eye, and rho the
  // Fresnel reflectance at that facet. p is normalised, so as the wind rises the glints
  // widen and dim together; nothing switches on or off.
  float cb = max(dot(n, H), 1e-3);
  float cb2 = cb * cb;
  float p = exp(-((1.0 - cb2) / cb2) / mss) / (3.14159265 * mss);
  float rhoSun = 0.02 + 0.98 * pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0);
  float glintRadiance = step(0.0, L.y) * SUN_IRRADIANCE * rhoSun * p / (4.0 * max(dot(n, V), 0.05) * cb2 * cb2);
  // Display: the glint is added on top of the water colour (white sunlight; the Fresnel
  // term already holds the reflectance, so there is no tint), then the water's luminance
  // rolls off softly above WATER_KNEE toward WATER_MAX. Below the knee nothing changes.
  // Foam is drawn after this and keeps its own colour.
  // Far out (blended in over SHOULDER_BLEND_NEAR to SHOULDER_BLEND_FAR metres, so there
  // is no seam) the shoulder starts lower, at WATER_KNEE_FAR, and follows
  // k + s (1 - (1 + o / (n s))^-n) with n = SHOULDER_FAR_N: slope 1 at the knee like the near
  // curve, but it approaches the top polynomially (n = 1 is Reinhard, large n the near
  // exponential), so the distant glint path, where many facets mirror the sun, keeps
  // compressing and shows its wave structure instead of pinning as a flat pale sheet, while
  // its brightest water still reaches above 120. The top (WATER_MAX) is the same everywhere.
  vec3 preGlint = color;
  glintRadiance *= mix(1.0, SUN_GLINT_K, sunward); // v30a.3: narrower, dimmer sun path at a glancing look
  color += vec3(glintRadiance) * uGlitM;
  float waterY = dot(color, vec3(LUMA));
  float overKnee = max(waterY - WATER_KNEE, 0.0);
  float yNear = waterY > WATER_KNEE ? WATER_KNEE + (WATER_MAX - WATER_KNEE) * (1.0 - exp(-overKnee / (WATER_MAX - WATER_KNEE))) : waterY;
  float spanFar = WATER_MAX - WATER_KNEE_FAR;
  float overFar = max(waterY - WATER_KNEE_FAR, 0.0);
  float yFar = waterY > WATER_KNEE_FAR ? WATER_KNEE_FAR + spanFar * (1.0 - pow(1.0 + overFar / (SHOULDER_FAR_N * spanFar), -SHOULDER_FAR_N)) : waterY;
  float rolledY = mix(yNear, yFar, smoothstep(SHOULDER_BLEND_NEAR, SHOULDER_BLEND_FAR, distEye));
  float rollK = waterY > 1e-5 ? rolledY / waterY : 1.0;
  color *= rollK;
  vec3 glintPart = vec3(glintRadiance) * uGlitM * rollK; // the glint as displayed (GAD 10:26 clamp below)

  // Foam is a band along the top of the front face. The physics grades it 1 at the
  // break or crest and lower down the face; here it is shaped softly and broken up.
  // flagMask keeps every foam pixel inside the outline of current plus recent flags
  // (the two sets never overlap, so their sum interpolates as their union; the red and
  // amber outlines are drawn at 0.5). vFoam is the grade the foam was laid with and
  // vFade its fade (1 for fresh foam): fading raises the threshold, so old foam thins
  // toward the lip and breaks up before it loses much strength, instead of greying out.
  float marked = vFlag + vRecent;
  float flagMask = smoothstep(0.55, 0.9, marked);
  float fadeCut = 1.0 - clamp(vFade, 0.0, 1.0);
  float fadeStrength = mix(0.6, 1.0, clamp(vFade, 0.0, 1.0));
  float band = smoothstep(0.15, 0.95, vFoam);
  float amount = band * flagMask;
  // Noise frame: across the crest (travel) and along it, stretched along the crest
  // so breaks look like streaks. The fine layer fades out where it would be sub-pixel.
  vec2 t = normalize(travel);
  vec2 q = vec2(dot(vWorld.xz, t), dot(vWorld.xz, vec2(-t.y, t.x)));
  float dist = length(eye - vWorld);
#ifdef HAS_DERIV
  vec2 qw = fwidth(q);
#else
  vec2 qw = vec2(dist * 0.0012, dist * 0.0012);
#endif
  vec2 gM = vec2(0.0);
  vec2 gF = vec2(0.0);
#ifdef HAS_DERIV
  vec2 qdx = dFdx(q);
  vec2 qdy = dFdy(q);
  float qdet = qdx.x * qdy.y - qdx.y * qdy.x;
  if (abs(qdet) > 1e-10) {
    float mdx = dFdx(marked), mdy = dFdy(marked), fdx = dFdx(vFoam), fdy = dFdy(vFoam);
    gM = vec2(qdy.y * mdx - qdx.y * mdy, qdx.x * mdy - qdy.x * mdx) / qdet;
    gF = vec2(qdy.y * fdx - qdx.y * fdy, qdx.x * fdy - qdy.x * fdx) / qdet;
  }
#endif
  vec2 warpQ = vec2(LACE_SHIFT_X * (valueNoise(q * vec2(1.0 / 0.9, 1.0 / 0.6) + 261.0) - 0.5 + 0.5 * (valueNoise(q * vec2(1.0 / 0.35, 1.0 / 0.3) + 263.0) - 0.5)),
                    LACE_SHIFT_Y * (valueNoise(q * vec2(1.0 / 0.8, 1.0 / 0.5) + 267.0) - 0.5 + 0.5 * (valueNoise(q * vec2(1.0 / 0.3, 1.0 / 0.3) + 269.0) - 0.5)));
  // 01:08 GAD (cell-shaped streaks): the foam edge is perturbed with world noise before every
  // threshold, as v25 did up close, but now wherever the ~1 m edge noise is resolved
  // (detailE), not only where the 0.12 m lace is (detailL), so mid-range foam no longer
  // traces the cells in straight parallel edges. ?lacegrain=0 restores the old edges.
  float detailE = LACE_GRAIN > 0.0 ? 1.0 - smoothstep(0.5, 1.0, max(qw.x / 1.0, qw.y / 0.7)) : 0.0;
  float detailW = max(detailL0(qw), detailE);
  float markedW = marked + dot(gM, warpQ) * detailW;
  float foamW = vFoam + dot(gF, warpQ) * detailW;
  // Warped flag mask and grade for the fresh band too; never past the flag outline (0.5).
#ifdef HAS_DERIV
  float fpG = max(max(length(dFdx(vWorld)), length(dFdy(vWorld))), 1e-4);
#else
  float fpG = 0.02;
#endif
  float lg2 = log2(fpG * 2.5);
  float oG = floor(lg2);
  float nS = smoothstep(0.25, 0.75, mix(valueNoise(q / exp2(oG) + 307.0), valueNoise(q / exp2(oG + 1.0) + 311.0), lg2 - oG));
  // Pixel-scale edge jitter (~2.5 px world noise): added to every foam threshold below, so
  // even where perspective squeezes the ~1 m edge noise into a few rows, no thresholded edge
  // runs straight for more than a few pixels.
  float edgeJ = LACE_GRAIN > 0.0 ? 0.16 * (nS - 0.5) : 0.0;
  if (LACE_GRAIN > 0.0) {
    flagMask = smoothstep(0.55, 0.9, markedW) * smoothstep(0.5, 0.55, marked);
    band = smoothstep(0.15, 0.95, foamW);
    amount = band * flagMask;
  }
  float fine = 1.0 - smoothstep(25.0, 70.0, dist);
  float grain = valueNoise(q * vec2(1.0 / 0.9, 1.0 / 2.6));
  grain = mix(grain, 0.6 * grain + 0.4 * valueNoise(q * vec2(1.0 / 0.35, 1.0 / 0.9) + 17.0), fine);
  // Noise only removes foam: alpha never exceeds amount, and the core near the crest
  // (amount close to 1) stays solid while the edges and ends break up.
  float alphaFar = amount * clamp((amount - 0.6 * grain - fadeCut + edgeJ) / 0.3, 0.0, 1.0) * fadeStrength;

  // Up close a grid cell is metres wide, so the per-sample grade would paint a slab.
  // Near the camera the band is measured in metres ahead of the crest it came from
  // instead (vAhead, linear across flagged cells, rising toward cells without foam): solid at the lip, fading out over about 1.5 m down the face, and the fine noise tears
  // the lower edge and the ends. Beyond ~28 m this is not used at all, so the look at
  // range is exactly the far band above.
  // Brightest right at the lip, half by ~0.35 m down the face, nearly gone by 1.5 m.
  float lipFade = exp(-abs(vAhead) / (0.5 * mix(0.4, 1.0, clamp(vFade, 0.0, 1.0))));
  float tear = 0.55 * valueNoise(q * vec2(1.0 / 0.8, 1.0 / 2.2) + 5.0)
             + 0.45 * valueNoise(q * vec2(1.0 / 0.22, 1.0 / 0.6) + 31.0);
  // Lengthwise taper: vRun falls to 0 at the last foam sample of a run, so the band
  // thins out along the crest inside the last foam cell instead of stopping in a step.
  float lengthTaper = smoothstep(0.0, 0.9, vRun);
  float amountNear = flagMask * smoothstep(0.1, 0.6, vFoam) * lipFade * lengthTaper;
  // Tearing only removes: holes open where the band is already thin, so the lower
  // edge and the ends break up while the lip itself stays whole.
  float alphaNearFresh = amountNear * smoothstep(0.0, 0.3, amountNear - 0.5 * tear - fadeCut + edgeJ) * fadeStrength;

  // Aged foam up close. Once the crest has moved on, every sample it swept over was
  // laid as a crest (grade 1, 0 m ahead, full run), so drawing it as a lip band filled
  // the whole swept area with solid foam on grid edges. Old foam has no lip: draw it
  // as streaks along the travel direction that thin into gaps as it ages, at 0.6 to
  // 0.8 strength, with a softer mask that starts further inside the amber outline.
  // vAge is 0 for foam flagged now, but it is interpolated across a cell, so it is
  // also gated by the current flag: inside the red outline (vFlag >= 0.5) fresh foam
  // keeps exactly the lip band above; the switch to streaks happens outside it.
  float oldness = smoothstep(0.0, 0.12, vAge) * (1.0 - smoothstep(0.25, 0.5, vFlag));
  float fadeV = clamp(vFade, 0.0, 1.0);
  // 1 where a streak octave is several pixels wide, 0 where it is under a pixel.
  float detailC = 1.0 - smoothstep(0.5, 1.0, max(qw.x / 2.8, qw.y / 0.45));
  float detailF = 1.0 - smoothstep(0.5, 1.0, max(qw.x / 0.9, qw.y / 0.18));
  // 01:08 GAD: the streaks (2.8 m x 0.45 m noise, thresholded) came out as straight, hard,
  // parallel strips at mid range; their across-crest coordinate now wanders with ~1 m and
  // ~0.4 m world noise (about +-0.3 m), so no strip edge runs straight.
  vec2 qS = q + (LACE_GRAIN > 0.0 ? vec2(0.3 * (valueNoise(q * vec2(1.0 / 1.5, 1.0 / 1.0) + 413.0) - 0.5),
      0.7 * (valueNoise(q * vec2(1.0 / 0.7, 1.0 / 0.6) + 401.0) - 0.5) + 0.3 * (valueNoise(q * vec2(1.0 / 0.35, 1.0 / 0.3) + 409.0) - 0.5)) : vec2(0.0));
  // with the grain on the long streak octave is less stretched (1.8 m instead of 2.8 m)
  float streakLen = LACE_GRAIN > 0.0 ? 1.8 : 2.8;
  float streak = 0.65 * mix(0.5, valueNoise(qS * vec2(1.0 / streakLen, 1.0 / 0.45) + 53.0), detailC)
               + 0.35 * mix(0.5, valueNoise(qS * vec2(1.0 / 0.9, 1.0 / 0.18) + 71.0), detailF);
  float oldAmount = smoothstep(0.6, 1.0, marked) * smoothstep(0.1, 0.6, vFoam);
  float cover = 0.3 + 0.55 * (1.0 - fadeV);
  float strengthOld = mix(0.6, 0.8, fadeV);
  float sharpW = LACE_GRAIN > 0.0 ? 0.3 : 0.18; // softer streak edges with the grain on (GAD: no hard straight edges)
  float alphaSharp = smoothstep(cover, cover + sharpW, streak * oldAmount + edgeJ);
  // Expected cover of a sub-pixel streak pattern (noise spread about 0.2 to 0.8).
  float alphaShare = clamp((0.8 * oldAmount - cover - 0.09) / max(0.6 * oldAmount, 1e-3), 0.0, 1.0);
  // Close to the camera the streak noise's widths (0.45 m and 0.18 m) are 10 to 30 px,
  // so young old foam (low cover) filled them solid and read as pale scraps whose outline
  // followed the cell-interpolated mask (cells are 2.2 m, about 170 x 85 px at 15 m).
  // Where the lace scale is resolved, dense old foam is opened from the inside instead:
  // streaks along the travel direction whose width wanders from about 0.08 to 0.3 m and
  // whose spacing is jittered (so no regular comb shows), torn by irregular holes that
  // open more where the foam is denser, with soft edges throughout. It fades out
  // (detailL) as the lace shrinks under a pixel, so the look from about 30 m out is as
  // before. Old foam stays inside the flags: oldAmount is 0 outside them.
  float detailL = 1.0 - smoothstep(0.5, 1.0, max(qw.x / 1.6, qw.y / 0.12));
  float yJ = q.y + LACE_WARP * (valueNoise(q * vec2(1.0 / 1.4, 1.0 / 0.7) + 201.0) + 0.5 * valueNoise(q * vec2(1.0 / 0.6, 1.0 / 0.4) + 205.0));
  float widthSel = valueNoise(q * vec2(1.0 / 1.6, 1.0 / 0.7) + 211.0);
  // 02:xx GAD straight-edge bar: with the grain on the lace lines are about half as long
  // (thin 0.9 m instead of 1.7 m) so no ridge runs straight past ~20 px near the camera.
  float laceLenK = LACE_GRAIN > 0.0 ? 0.53 : 1.0;
  float laceThin = valueNoise(vec2(q.x / (1.7 * laceLenK), yJ / 0.08) + 217.0);
  float laceMid = valueNoise(vec2(q.x / (2.3 * laceLenK), yJ / 0.16) + 219.0);
  float laceWide = valueNoise(vec2(q.x / (3.2 * laceLenK), yJ / 0.3) + 221.0);
  float laceLine = mix(mix(laceThin, laceMid, smoothstep(0.2, 0.5, widthSel)), laceWide, smoothstep(0.5, 0.8, widthSel));
  // Each streak is the mid contour of its noise: a ridge, brightest in its core and soft to
  // both sides, so lines stay thin and separate instead of filling between dark gaps.
  // A finer term (about 0.05 m across) keeps the mid contour from spreading into a broad
  // core wherever the coarser noise happens to sit near its mid value over an area.
  float laceRidge = 1.0 - abs(2.0 * (laceLine + LACE_FINE_LINE * (valueNoise(vec2(q.x / 0.5, yJ / 0.05) + 289.0) - 0.5)) - 1.0);
  float holeN = 0.6 * valueNoise(q * vec2(1.0 / 0.9, 1.0 / 0.55) + 223.0)
              + 0.4 * valueNoise(q * vec2(1.0 / 0.4, 1.0 / 0.26) + 229.0);
  // Near the camera the old-foam mask is a smooth noise-thresholded alpha: the cell-
  // interpolated flag (marked) and grade (vFoam) are compared with world-space noise at
  // about 1 m and 0.35 m instead of fixed levels, so the edge wanders through the 2.2 m
  // cell rather than tracing it. The threshold never goes below LACE_MASK_LO, inside the
  // flag outline (drawn at 0.5), so foam stays on the flags.
  float edgeN = 0.6 * valueNoise(q * vec2(1.0 / 1.0, 1.0 / 0.7) + 241.0)
              + 0.4 * valueNoise(q * vec2(1.0 / 0.35, 1.0 / 0.3) + 247.0);
  float edgeAt = mix(LACE_MASK_LO, LACE_MASK_HI, edgeN);
  float gradeAt = mix(0.1, 0.55, valueNoise(q * vec2(1.0 / 0.8, 1.0 / 0.5) + 251.0));
  // The lookup itself is also displaced by world-space noise (typically +-0.4 m, up to
  // about +-0.9 m, along the travel; about 0.6 of that along the crest): inside a triangle marked and vFoam are linear, so
  // their value a displacement w away is the value here plus gradient . w, with the
  // gradient in metres solved from the screen derivatives. Where the cell ramp is steep
  // this moves the edge by up to the full displacement, so it no longer traces the cell.
  // The switch from the fresh lip band to old streaks (oldness, from the cell-interpolated
  // vAge and vFlag) is displaced the same way, so it does not cut the foam along a cell.
  vec2 gA = vec2(0.0);
  vec2 gG = vec2(0.0);
#ifdef HAS_DERIV
  if (abs(qdet) > 1e-10) {
    float adx = dFdx(vAge), ady = dFdy(vAge), gdx = dFdx(vFlag), gdy = dFdy(vFlag);
    gA = vec2(qdy.y * adx - qdx.y * ady, qdx.x * ady - qdy.x * adx) / qdet;
    gG = vec2(qdy.y * gdx - qdx.y * gdy, qdx.x * gdy - qdy.x * gdx) / qdet;
  }
#endif
  float oldnessW = smoothstep(0.0, 0.12, vAge + dot(gA, warpQ)) * (1.0 - smoothstep(0.25, 0.5, vFlag + dot(gG, warpQ)));
  oldness = mix(oldness, oldnessW, detailW);
  // Never past the flag outline (drawn at marked 0.5), whatever the displacement.
  float oldAmountL = smoothstep(0.5, 0.58, marked)
                   * smoothstep(edgeAt - 0.05, edgeAt + 0.05, markedW) * smoothstep(gradeAt - 0.08, gradeAt + 0.08, foamW);
  oldAmount = mix(oldAmount, oldAmountL, detailW);
  float denseOld = smoothstep(cover, cover + 0.18, streak * oldAmount);
  float holeAt = LACE_HOLE_BASE + LACE_HOLE_DENSE * denseOld;
  // Core at full strength, then a thinner translucent shoulder (LACE_SHOULDER of the core)
  // either side, as thin foam is: the streak is no dimmer, it just has a soft margin.
  // The bright cores are cut into short dashes (about 0.1 to 0.3 m) so no bright patch
  // grows large; the shoulders run on between them, so the streak still reads as one line.
  float laceDash = smoothstep(LACE_DASH, LACE_DASH + 0.15, valueNoise(vec2(q.x / 0.3, yJ / 0.2) + 271.0));
  // A second, independent break across the streak (about 0.08 m) splits cores that come
  // out broad where the noise sits near its mid value; they drop to the shoulder there.
  laceDash *= smoothstep(0.3, 0.42, valueNoise(vec2(q.x / 0.25, q.y / 0.08) + 281.0));
  float lace = max(smoothstep(LACE_RIDGE_LO, LACE_RIDGE_HI, laceRidge) * laceDash, LACE_SHOULDER * smoothstep(LACE_SH_LO, LACE_SH_HI, laceRidge))
             * smoothstep(holeAt, holeAt + LACE_HOLE_SOFT, holeN);
  // The fill threshold wanders near the camera so the outline does not follow the
  // 2.2 m cells in straight edges (slightly biased low, so the lace keeps about the old
  // foam cover; it can only add foam where oldAmount is already non-zero, inside flags).
  float coverJ = cover + LACE_EDGE * (valueNoise(q * vec2(1.0 / 0.7, 1.0 / 0.4) + 233.0) - 0.58) * detailW;
  alphaSharp = smoothstep(coverJ, coverJ + sharpW, streak * oldAmount + edgeJ);
  float alphaOld = mix(alphaShare, alphaSharp, detailC) * strengthOld;
  // Near the camera the lace carries the foam: each streak peaks in its core and softens
  // to its edges, and the densest foam is held to LACE_CAP so it never fills solid.
  alphaOld = mix(alphaOld, min(alphaOld, LACE_CAP) * lace, detailL);
  // The fine chop breaks up the flat fill of old streaks: on the side of each ripple
  // that rises toward the travel direction some foam is taken away. Only removes.
  float rise = dot(fs, t);
  alphaOld *= 1.0 - 0.5 * smoothstep(-0.02, 0.1, rise);
  float nearWeight = 1.0 - smoothstep(20.0, 28.0, dist);
  // Old foam takes the streak treatment at every distance, not only within 28 m: seen
  // from higher up all the water is far, and the far band drew crest-swept samples as
  // solid cells. Where a streak is narrower than a pixel its noise is replaced by the
  // share of the pixel it would cover, so distant old foam thins smoothly instead of
  // shimmering. Fresh foam (oldness 0) is exactly the near or far band as before.
  float alphaFresh = mix(alphaFar, alphaNearFresh, nearWeight);
  float alpha = mix(alphaFresh, alphaOld, oldness);
#ifdef HAS_FOAMNEAR
  // v29.1 crest foam: a breaking wave whitecaps along its whole crest line, not only at the cells
  // the break test flagged. Where breaking was flagged within ~3 cells (foamNear), the crest itself
  // (the lip field, the same crest cells that glow) carries foam, feathered on both.
  {
    float crestF = smoothstep(0.35, 0.8, vLip) * smoothstep(0.03, 0.3, foamNear(vSea.xy)) * clamp(vFoamKeep, 0.0, 1.0);
    alpha = max(alpha, CREST_ALPHA * crestF); // v30b: the whole breaking crest whitecaps (was 0.6)
  }
#endif
  // GAD (c): fading foam keeps a dim lace (alpha up to DIM_LACE, peak under 130), only
  // inside the flag outline (edgeAt >= LACE_MASK_LO 0.6).
  // GAD 06:27: no floor. The lace fades to 0 with age (fadeV) and its edge is broken up by
  // world-space noise (edgeAt / gradeAt thresholds at ~1 m and 0.35 m, as in v25), so it never
  // shows as a flat sheet with a cell outline; inside, sub-pixel lace is replaced by ~1 m noise.
  float dimLace = DIM_LACE * fadeV * smoothstep(edgeAt - 0.08, edgeAt + 0.08, marked) * smoothstep(gradeAt - 0.1, gradeAt + 0.1, vFoam)
                * mix(0.25 + 0.75 * holeN, lace, detailL);
  alpha = max(alpha, dimLace);
  // Foam grain over the whole foam range (21:55 lace bar; 01:08 GAD dense grain): world-
  // anchored value noise with cells ~2.5 px at every distance (two octaves blended by the
  // pixel footprint, so it does not swim). Lace (alpha < ~0.3): alpha x (1 - 0.8 n);
  // dense foam: weaker, alpha x (1 - 0.4 n), times a second, larger noise (~15 px).
  // Each factor is divided by its mean (QM 01:09: lingering foam already reads as lighter
  // water, so the grain must not dim it on average); alpha is capped at 1.
  // ?lacegrain=0 turns it off (LACE_GRAIN 0).
  float lg15 = log2(fpG * 15.0);
  float oL = floor(lg15);
  float nL = smoothstep(0.25, 0.75, mix(valueNoise(q / exp2(oL) + 331.0), valueNoise(q / exp2(oL + 1.0) + 337.0), lg15 - oL));
  float denseW = smoothstep(0.25, 0.45, alpha);
  // v30b: fresh foam is one crisp white sheet; the grain (the streaky texture) is kept for ageing foam only.
  float grainK = mix(FRESH_GRAIN, 1.0, oldness);
  float ampS = LACE_GRAIN * grainK * mix(1.0, 0.5, denseW);           // 0.8 lace, 0.4 dense
  float ampL = LACE_GRAIN * grainK * 0.5 * denseW;                     // 0.4 on dense only
  float grainF = (1.0 - ampS * nS) / (1.0 - 0.5 * ampS) * (1.0 - ampL * nL) / (1.0 - 0.5 * ampL);
  alpha = min(alpha * grainF, 1.0);
  // v30b crisp edges: fresh foam goes from clear to solid over a short alpha step instead of a
  // grey ramp; ageing foam keeps its soft lace (oldness 1 is exactly the old alpha).
  // ageing foam breaks into holes (its threshold rises with age) instead of turning translucent grey;
  // only where a pixel covers a lot of water (far off) does fractional cover stay as it is.
  float crispW = max(1.0 - smoothstep(0.15, 0.6, fpG), 0.6);
  float lo = mix(FRESH_LO, 0.28, oldness) + 0.3 * (nS - 0.5); // noise-broken edge (~2.5 px noise, so no edge runs straight past a few px)
  alpha = mix(alpha, smoothstep(lo, lo + FRESH_HI - FRESH_LO, alpha), crispW);
  // v30b.3 (GAD 4): big fresh patches were flat solid slabs. Inside dense fresh foam (deep inside
  // the flag outline, where only large patches reach), world noise of ~1.2 m and ~0.45 m cells
  // opens holes and darker gaps (~20% of the interior); small patches and their edges are untouched.
  {
    float interior = smoothstep(0.9, 1.0, marked) * smoothstep(0.85, 1.0, alpha) * (1.0 - oldness);
    float hn = 0.65 * valueNoise(q / FOAM_HOLE_M + 401.0) + 0.35 * valueNoise(q / (0.38 * FOAM_HOLE_M) + 409.0);
    float hole = smoothstep(FOAM_HOLE_AT, FOAM_HOLE_AT + 0.06, hn);
    float gap = smoothstep(FOAM_HOLE_AT - 0.1, FOAM_HOLE_AT, hn) * (1.0 - hole);
    alpha *= 1.0 - interior * (0.9 * hole + 0.25 * gap);
  }
  // Foam takes the fine chop's shading only where it darkens: never brighter than the
  // big grid's lighting alone would make it.
  // v30b: fresh foam is near white (luma ~236 lit, FOAM_WHITE); it dulls as it ages (FOAM_OLD_DIM).
  // v32: aged foam at albedo ~0.45 against ~0.85 fresh (Koepke), (0.45 / 0.85)^(1 / 2.2) = 0.75.
  vec3 foamColor = FOAM_WHITE * vec3(0.985, 1.0, 1.0) * (0.95 + 0.05 * min(lightFoam, lightBig)) * mix(1.0, FOAM_OLD_DIM, oldness);
  // Old streaks also take a little shade on ripple sides turned from the sun
  // (up to 15% darker). Fresh foam is not touched.
  // v30b foam volume: bubbly relief (~0.2 m and ~0.6 m cells) tilts the foam's own normal, so foam
  // catches light on its sun side and shadows in its pits instead of reading as flat paint.
  // Mean factor ~1 (fresh foam stays 230-245); faded out where the cells shrink below a pixel.
  {
    vec2 bq = q / 0.2; vec2 bq2 = q / 0.6;
    float b0 = valueNoise(bq + 71.0), bx = valueNoise(bq + vec2(0.4, 0.0) + 71.0), bz = valueNoise(bq + vec2(0.0, 0.4) + 71.0);
    float c0 = valueNoise(bq2 + 19.0), cx = valueNoise(bq2 + vec2(0.4, 0.0) + 19.0), cz = valueNoise(bq2 + vec2(0.0, 0.4) + 19.0);
    vec2 gb = vec2(b0 - bx, b0 - bz) + 0.8 * vec2(c0 - cx, c0 - cz);
    vec3 nB = normalize(nBig + FOAM_BUMP * vec3(gb.x * t.x - gb.y * t.y, 0.0, gb.x * t.y + gb.y * t.x));
    float lit = clamp(dot(nB, L), 0.0, 1.0) / max(dot(nBig, L), 0.25);
    float vis = 1.0 - smoothstep(0.08, 0.35, fpG);
    float bub = clamp(0.84 + 0.16 * lit + 0.08 * (b0 + c0 - 1.0), 0.72, 1.0); // GAD v30b.2: highlights capped (brightest foam <= 245)
    foamColor *= mix(1.0, bub, vis * BUMP_ON);
  }
  float awayFromSun = smoothstep(0.0, 0.12, dot(fs, normalize(L.xz)));
  foamColor *= 1.0 - 0.06 * awayFromSun * oldness;
  // Translucency lifts the water toward TRANSL_COLOR by the body's (non-reflected) share:
  // per channel never past it (so never white, never above about 130, and glint already
  // brighter is left alone). Where foam is drawn the aerated crest scatters that light as
  // the foam's own brightness, so none shows through: foam pixels stay exactly as before.
  // Far out a pixel covers more water than the thin tip itself (TRANSL_TIP wide): it sees
  // the tip averaged with the thicker water beside it, so the lift is scaled by
  // tip / (tip + pixel footprint), about 1 near the camera.
#ifdef HAS_DERIV
  float footprint = max(length(dFdx(vWorld)), length(dFdy(vWorld)));
#else
  float footprint = 0.0;
#endif
  float translCover = TRANSL_TIP / (TRANSL_TIP + footprint);
  // Screen-space smoothing: where the transmission changes by about its own value within
  // one pixel, the lit lip is narrower than the pixel and would alias; it is scaled down
  // in proportion. Up close a lip spans many pixels and is left as it is.
#ifdef HAS_DERIV
  translCover *= vThin / (vThin + 2.0 * (abs(dFdx(vThin)) + abs(dFdy(vThin))) + 1e-6);
#endif
  // The pixel sees cover x the lip's light. Sky part: as before. Sun part: averaged first
  // (linear radiance), then the same soft cap, so the display limit acts on the pixel.
  float sunPart = TRANSL_MAX * (1.0 - exp(-translCover * translSun));
  transl = min(1.0 - (1.0 - transl * translCover) * (1.0 - sunPart), TRANSL_MAX);
  color += UPWELL * transl * (1.0 - reflShare) * max(vec3(TRANSL_COLOR) - color, 0.0) * (1.0 - smoothstep(0.0, 0.05, alpha));
  // GAD 04:15 over-130: dense foam is dimmed, not cut: where alpha nears solid the foam
  // colour drops to DIM_DENSE of itself (about 125 instead of 190), so the foam area and
  // the lace (alpha <= DIM_LACE) are unchanged.
  // v30b: dense fresh foam is no longer dimmed (it read grey); only ageing dense foam keeps DIM_DENSE.
  foamColor *= mix(1.0, DIM_DENSE, smoothstep(0.35, 0.8, alpha) * 0.0);
  // v30a grade (water only, before the foam is laid on): luma through a contrast curve about
  // GRADE_P -> GRADE_M (power GRADE_G), soft shoulder from GRADE_KNEE (143) to the cap GRADE_CAP (196), then rebuilt at hue GRADE_HUE with
  // saturation GRADE_SAT, eased off only where full saturation cannot hold that luma (bright glints).
  {
    // Exposure rises a little with the wind's slope variance: a rough sea mirrors more sky
    // (more facets tilted toward it), so the two winds land at the same mid-tone.
    float yIn = max(dot(color, vec3(LUMA)), 0.0) * (1.0 + GRADE_WK * (subMss + chopMss + fineMss))
               * mix(1.0, SUN_EXPO_K, sunward); // v30a.3: less exposure looking into the sun at a glancing angle
    float yC = GRADE_M * pow(yIn / GRADE_P, GRADE_G);
    float yOut = yC < GRADE_KNEE ? yC : GRADE_KNEE + (GRADE_CAP - GRADE_KNEE) * (1.0 - exp(-(yC - GRADE_KNEE) / (GRADE_CAP - GRADE_KNEE)));
    float hf = (240.0 - GRADE_HUE) / 60.0;                   // hue in 180..240: R min, B max
    float satMax = (1.0 - yOut / GRADE_VMAX) / (0.9278 - 0.7152 * hf); // keeps every channel <= GRADE_VMAX (229)
    float sat = clamp(min(GRADE_SAT, satMax), 0.0, 1.0);
    vec3 hc = vec3(1.0 - sat, 1.0 - sat + sat * hf, 1.0);
    color = hc * (yOut / dot(hc, vec3(LUMA)));
  }
  // GAD 10:26 glint clamp (v29): the specular under foam is scaled by (1 - foam alpha) before the
  // foam is laid over it, so sun glitter cannot shine through lace past the foam cap.
  color = mix(color - alpha * glintPart, foamColor, alpha);

  // Break flags. Red: samples flagged now (outline at vFlag = 0.5, light red tint
  // inside, stronger on the break sample). Amber: the outline of everything flagged
  // within the fade window (current plus recent), with a light amber tint where only
  // recent. All foam sits inside the amber or red outline.
  if (showFlags > 0.5) {
#ifdef HAS_DERIV
    float w = max(fwidth(vFlag), 1e-4);
    float wm = max(fwidth(marked), 1e-4);
#else
    float w = 0.06;
    float wm = 0.06;
#endif
    vec3 red = vec3(0.78, 0.20, 0.16);
    vec3 amber = vec3(0.95, 0.66, 0.12);
    float inside = smoothstep(0.5 - w * 0.5, 0.5 + w * 0.5, vFlag);
    float insideMarked = smoothstep(0.5 - wm * 0.5, 0.5 + wm * 0.5, marked);
    float edge = 1.0 - smoothstep(w * 0.75, w * 1.75, abs(vFlag - 0.5));
    float edgeMarked = 1.0 - smoothstep(wm * 0.75, wm * 1.75, abs(marked - 0.5));
    float breakSample = smoothstep(0.08, 0.16, vBroken);
    // v30.20: red only where a cell is breaking right now; the foam set (current + recent flags)
    // is just a faint amber outline, no fill, so the overlay no longer paints the whole sea.
    color = mix(color, amber, edgeMarked * 0.25);
    color = mix(color, red, breakSample * 0.85);
  }
  gl_FragColor = vec4(color, 1.0);
}
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
  }
  return shader;
}

export function seaFragmentSources(derivatives, island) {
  const location = { search: island ? "?island=1" : "" };
  const fragBody = (derivatives ? "#extension GL_OES_standard_derivatives : enable\n#define HAS_DERIV 1\n" : "") + REFLECT_DEFS + FINE_DEFS + LIGHT_DEFS + TRANSL_DEFS + LACE_DEFS + FRAG;
  const cpu = toEs3(fragBody, "fragment");
  // GPU physics: the patches' slope variance per mip level comes from their stats textures
  // (level L's variance at texel L; kept = level L / level 0, as keptPerLevel computes),
  // so nothing is read back. The rest of the shader is the same text.
  const gpu = toEs3(fragBody
    .replace("uniform float chopKept[8];\nuniform float fineKept[8];", "float chopKept[8];\nfloat fineKept[8];\nuniform highp sampler2D bandStats;")
    .replace("void main() {", `void main() {
  {
    float c0 = texelFetch(bandStats, ivec2(0, 0), 0).x;
    float f0 = texelFetch(bandStats, ivec2(0, 1), 0).x;
    for (int i = 0; i < 8; i += 1) {
      chopKept[i] = texelFetch(bandStats, ivec2(i, 0), 0).x / (c0 > 0.0 ? c0 : 1.0);
      fineKept[i] = texelFetch(bandStats, ivec2(i, 1), 0).x / (f0 > 0.0 ? f0 : 1.0);
    }
  }`), "fragment");
  const fft = toEs3((typeof location !== "undefined" && new URLSearchParams(location.search).get("island") === "1" ? "#define ISLAND_ON 1\n" : "") + fragBody
    .replace("uniform float chopKept[8];\nuniform float fineKept[8];", "float chopKept[8];\nfloat fineKept[8];\nuniform highp sampler2D bandStats;")
    .replace("uniform float subMss;", "uniform float subMssU;\nvarying float vLodLost;\nfloat subMss;")
    // v31.8: do not gate vThin through smoothstep(0.14, 0.18) * 4.2. That isosurface was a
    // second hard lip. The live colour is PHYS_SEA; this path stays the continuous transmission.
    .replace("vec3 nBig = normalize(vNormal);", "vec3 nBig = normalize(mix(normalize(vNormal), seaNormalPix(vSea.xy, vSea.z), clamp(vSea.w, 0.0, 1.0)));")
    // GAD per-pixel foam (see foamPix in fftView.js): the foam varyings become globals set per pixel.
    .replace("varying float vFoam;", "varying float vFoam;\nfloat fFoam; float rawFoam() { return vFoam; }\n#define vFoam fFoam\n")
    .replace("varying float vFlag;", "varying float vFlag;\nfloat fFlag; float rawFlag() { return vFlag; }\n#define vFlag fFlag\n")
    .replace("varying float vRecent;", "varying float vRecent;\nfloat fRecent; float rawRecent() { return vRecent; }\n#define vRecent fRecent\n")
    .replace("varying float vFade;", "varying float vFade;\nfloat fFade; float rawFade() { return vFade; }\n#define vFade fFade\n")
    .replace("varying float vAge;", "varying float vAge;\nfloat fAge; float rawAge() { return vAge; }\n#define vAge fAge\n")
    .replace("varying float vAhead;", "varying float vAhead;\nfloat fAhead; float rawAhead() { return vAhead; }\n#define vAhead fAhead\n")
    .replace("varying float vRun;", "varying float vRun;\nfloat fRun; float rawRun() { return vRun; }\n#define vRun fRun\n")
    .replace("  // Break flags. Red: samples flagged now", PHYS_SEA + "  // Break flags. Red: samples flagged now")
    .replace("gl_FragColor = vec4(color, 1.0);", `${typeof location !== "undefined" && new URLSearchParams(location.search).get("island") === "1" ? `if (uIsland > 0.5) {
    // The beach is the bed, already in the mesh. This only paints that sand
    // and the sheet that is actually on it. It does not tint the sea, so
    // there is no teal disc and no bright ring at a radius.
    vec2 rest = vSea.xy;
    // Four taps of the analytic bed on a 1.6 m box. The mean is the height,
    // the differences are the slope. One tap per pixel let the contour follow
    // a single chord of the mesh; the box mean is smooth across triangles.
    const float eB = 0.8;
    float bPP = islandBed(rest + vec2(eB, eB));
    float bPM = islandBed(rest + vec2(eB, -eB));
    float bMP = islandBed(rest + vec2(-eB, eB));
    float bMM = islandBed(rest + vec2(-eB, -eB));
    float bedS = 0.25 * (bPP + bPM + bMP + bMM);
    float cap = max(uR2, 0.15);
    float inland = smoothstep(cap - 0.35, cap + 0.55, bedS);
    // Sand grain and a broader mottle move the contour, so no fade draws a
    // clean ring. Heights: the beach face drops ~0.08 m per metre.
    float grit = sfNoise(rest * 11.0);
    float grit2 = sfNoise(rest * 3.4 + 4.0);
    float bedN = bedS + (grit - 0.5) * 0.06 + (grit2 - 0.5) * 0.1;
    // Dry sand covers from just under the waterline to well up the face, a
    // wide smoothstep with no edge. The sea under it is the clear shallows.
    float cover = max(smoothstep(-0.12, 0.7, bedN), inland);
    // Damp only darkens sand that is already covering: a band a couple of
    // metres wide just above the waterline, between the dry sand and the
    // clear shallows. It never replaces the sea colour.
    float damp = smoothstep(0.0, 0.2, bedN) * (1.0 - smoothstep(0.34, 0.6, bedN));
    damp *= (1.0 - inland) * (uBeach > 0.5 && uBeach < 1.5 ? 0.0 : 1.0);
    // Surf: a wide analytic band around the waterline that runs up and back
    // with a slow swash, broken into rounded blobs that drift. No film texel.
    float swPh = foamT * 0.75 + dot(rest, vec2(0.09, 0.06)) + grit2 * 2.5;
    float swash = sin(swPh);
    float c0 = 0.02 + 0.09 * swash;
    float band = smoothstep(c0 - 0.42, c0 - 0.12, bedN) * (1.0 - smoothstep(c0 - 0.02, c0 + 0.12, bedN));
    float foamN = sfNoise(rest * 1.35 + vec2(foamT * 0.16, foamT * 0.05));
    float foamN2 = sfNoise(rest * 3.2 - vec2(foamT * 0.09, foamT * 0.12) + 7.0);
    float blobs = smoothstep(0.4, 0.75, foamN * 0.6 + foamN2 * 0.4);
    // v32: the surf is the same foam language as the sea: a soft mask (the swash band, broken
    // by the drifting blobs) revealed through the sea foam's holey bubble detail (foamDetailG).
    // The edge is wide and feathered (plus fwidth), never a binary cut of the lace, so the surf
    // erodes into pits and holes. Residual surf is part-transparent; only the fresh lip (the
    // landward front of the run-up while the swash advances) is close to opaque white.
    float surfOn = uBeach > 1.5 ? 0.0 : 1.0;
    float surfM = band * mix(0.35, 1.0, blobs) * (1.0 - inland) * surfOn;
    float sT = 1.0 - surfM;
    float sE = 0.2 + 0.15 * (1.0 - surfM) + 1.5 * fwidth(foamDetailG);
    float surf = smoothstep(sT - sE, sT + sE, foamDetailG) * smoothstep(0.0, 0.3, surfM);
    float lip = smoothstep(c0 - 0.14, c0 - 0.04, bedN) * (1.0 - smoothstep(c0 - 0.02, c0 + 0.1, bedN))
              * smoothstep(0.0, 0.7, cos(swPh)) * blobs * (1.0 - inland) * surfOn;
    float surfA = surf * mix(0.4, 0.78, lip);
    // White -> turquoise aeration skirt on the water side of the surf (the open-ocean halo's
    // milky turquoise), broken by the drifting noise, dying out into the clear shallows.
    float aerS = smoothstep(c0 - 0.7, c0 - 0.3, bedN) * (1.0 - smoothstep(c0 - 0.1, c0 + 0.05, bedN))
               * mix(0.3, 1.0, smoothstep(0.3, 0.7, foamN)) * (1.0 - inland) * surfOn;
    if (cover > 0.002 || surf > 0.002 || aerS > 0.002) {
      // Dry and wet sand blended on the smooth contour, not the vertex height.
      vec3 sand = sfSandShore(rest, bedN, bedN);
      vec3 nBed = normalize(vec3(-(bPP + bPM - bMP - bMM) * 0.5, 2.0 * eB, -(bPP + bMP - bPM - bMM) * 0.5));
      float ndl = clamp(dot(nBed, normalize(lightDir)), 0.0, 1.0);
      sand *= 0.94 + 0.1 * ndl;
      color = mix(color, sand, clamp(cover, 0.0, 1.0));
      color = mix(color, color * vec3(0.8, 0.82, 0.82), damp * smoothstep(0.1, 0.6, cover) * 0.85);
      vec3 milkyS = mix(color, vec3(0.60, 0.86, 0.85), 0.55);
      color = mix(color, milkyS, 0.4 * aerS * (1.0 - surfA));
      vec3 surfC = mix(mix(color, vec3(0.60, 0.86, 0.85), 0.5), foamWhiteG * (0.9 + 0.1 * ndl), smoothstep(0.15, 0.7, surf) * mix(0.75, 1.0, lip));
      color = mix(color, surfC, clamp(surfA, 0.0, 0.78));
    }
  }` : ""}
  color *= dayTint;
  gl_FragColor = vec4(color, 1.0);`)
    .replace("void main() {", PHYS_DECL + SEAFLOOR_FS + "#define HAS_FOAMNEAR 1\n#define FOAM_PIX " + ((typeof location !== "undefined" && new URLSearchParams(location.search).get("foampix") === "0") ? "false" : "true") + "\n" + FRAG_FFT_NORMAL.replace("in vec4 vSea;\n", "").replace("uniform highp sampler2D physG;\n", "").replace(/\bphysG\b/g, "physK") + `in vec3 vCurl;
uniform vec3 dayTint;
// v32 (C2): caustic focusing from the live surface. Sunlight refracted through a surface with height
// Hessian Hh has its cross-section scaled by det(I + d (1 - 1/n) Hh) after a depth d (Evan Wallace's
// area ratio for a near-vertical sun, before the rays cross), so its brightness goes as the inverse.
// Hessian by central differences of the FFT shading slope (fftR1.z = dh/dx, fftR2.x = dh/dz, the
// plain copy) one mip down (~2 m), the cross term left out: 4 taps on samplers the sea already
// binds. Clamped 0.3..3. Read at the surface point the refracted sun ray through the bed point
// leaves from (the same projection as floorMul).
float causFocus(vec3 bedP) {
  float d = max(0.0, -bedP.y);
  if (d < 0.05) return 1.0;
  vec3 Rf = refract(-normalize(lightDir), vec3(0.0, 1.0, 0.0), 1.0 / 1.333);
  if (Rf.y > -0.02) Rf = vec3(0.0, -1.0, 0.0);
  vec2 xz = bedP.xz - bedP.y * Rf.xz / Rf.y;
  const float lodC = 1.0;
  float eM = 2.0 * fftL / float(physN);
  vec2 uv = (xz + 0.5 * fftL) / fftL + 0.5 / float(physN);
  vec2 du = vec2(eM / fftL, 0.0);
  float hxx = (textureLod(fftR1, uv + du, lodC).z - textureLod(fftR1, uv - du, lodC).z) / (2.0 * eM);
  float hzz = (textureLod(fftR2, uv + du.yx, lodC).x - textureLod(fftR2, uv - du.yx, lodC).x) / (2.0 * eM);
  float k = d * (1.0 - 1.0 / 1.333);
  float det = (1.0 + k * hxx) * (1.0 + k * hzz);
  return clamp(1.0 / max(abs(det), 1e-3), 0.3, 3.0);
}
${typeof location !== "undefined" && new URLSearchParams(location.search).get("island") === "1" ? `in vec4 vCoast;
in vec2 vBedG;
in float vFloorGrid;
in float vEta;
in float vWet;
uniform float uSteep;
// 0 normal, 1 beach=nodamp, 2 beach=nosurf.
uniform float uBeach;
` : ""}
void main() {
  if (vKind > 0.5) {
    gl_FragColor = seafloorShade();
    // Walls return the colour unchanged (underLight). The scene copy stays
    // plain sand: the water shader samples the caustic map once, at the
    // refracted hit, so the filaments are not applied twice. The floor drawn
    // on screen still carries them.
    if (sfRaw < 0.5) gl_FragColor.rgb = underLight(gl_FragColor.rgb, vWorld);
    ${typeof location !== "undefined" && new URLSearchParams(location.search).get("island") === "1" ? `if (uIsland > 0.5 && vFloorGrid > 0.5 && sfRaw < 0.5) {
      // Scene pass (sfRaw) keeps seafloorShade, ripples and all, on the
      // whole floor. Only the direct mound uses the local water depth, and
      // only where the bed has actually risen. The open floor is untouched.
      float bedS = islandBed(vWorld.xz);
      float D = max(0.0, -bedS);
      float risen = smoothstep(0.45, 6.0, depth - D);
      if (risen > 0.002) {
        vec3 al = sfSandDetail(vWorld.xz);
        vec3 lit = sfCap(sfEncode(SF_UPWELL * sfFloor(max(D, 0.25), al)));
        gl_FragColor.rgb = mix(gl_FragColor.rgb, lit, risen);
      }
    }` : ""}
    gl_FragColor.rgb *= dayTint;
    return;
  }
  if (FOAM_PIX) {
    vec4 pf, pg; foamPix(vSea.xy, pf, pg);
    float kp = clamp(vFoamKeep, 0.0, 1.0);
    vFoam = pf.x * kp; vFade = pf.y * kp; vAge = pf.z; vAhead = pg.x; vRun = pg.y; vRecent = pg.z * kp; vFlag = pg.w * kp;
  } else { vFoam = rawFoam(); vFlag = rawFlag(); vRecent = rawRecent(); vFade = rawFade(); vAge = rawAge(); vAhead = rawAhead(); vRun = rawRun(); }
  subMss = subMssU + vLodLost;
  {
    float c0 = texelFetch(bandStats, ivec2(0, 0), 0).x;
    float f0 = texelFetch(bandStats, ivec2(0, 1), 0).x;
    for (int i = 0; i < 8; i += 1) {
      chopKept[i] = texelFetch(bandStats, ivec2(i, 0), 0).x / (c0 > 0.0 ? c0 : 1.0);
      fineKept[i] = texelFetch(bandStats, ivec2(i, 1), 0).x / (f0 > 0.0 ? f0 : 1.0);
    }
  }`), "fragment");
  return { fragBody, cpu, gpu, fft };
}

export function createView(canvas) {
  // WebGL2 when available (GPU physics needs it); WebGL1 still draws the CPU physics.
  const gl = canvas.getContext("webgl2", { antialias: true }) || canvas.getContext("webgl", { antialias: true });
  if (!gl) throw new Error("WebGL is not available");
  const webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  const bandStatsTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, bandStatsTex);
  if (webgl2) {
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 8, 2);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 8, 2, gl.RGBA, gl.FLOAT, new Float32Array(64));
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 8, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(64));
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const statRead = gl.createFramebuffer();
  function copyBandStats(chop, fine) {
    const prev = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    const unit = gl.getParameter(gl.ACTIVE_TEXTURE);
    // Unit 7 is the bandStats unit; the caller's unit keeps its texture.
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, bandStatsTex);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, statRead);
    const row = (src, y) => {
      if (!src) return;
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, src, 0);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, y, 0, 0, 8, 1);
    };
    row(chop, 0);
    row(fine, 1);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prev);
    gl.activeTexture(unit);
  }
  // Derivatives give a one-pixel-ish flag outline at any distance. Without them the
  // outline is a fixed band in attribute space.
  const derivatives = webgl2 || gl.getExtension("OES_standard_derivatives");
  const islandDraw = typeof location !== "undefined" && new URLSearchParams(location.search).get("island") === "1";
  const seaSrc = seaFragmentSources(!!derivatives, islandDraw);
  const fragBody = seaSrc.fragBody;
  const cpuText = webgl2 ? seaSrc.cpu : seaSrc.fragBody;
  const fragShader = compile(gl, gl.FRAGMENT_SHADER, cpuText);
  const gpuText = seaSrc.gpu;
  const gpuFragShader = webgl2 ? compile(gl, gl.FRAGMENT_SHADER, gpuText) : null;
  function link(vertSource, frag = fragShader) {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertSource));
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "program link failed");
    }
    return program;
  }
  // Two programs, one fragment shader: per-vertex values streamed from the CPU physics,
  // or read from the GPU physics textures.
  const cpuProgram = link(webgl2 ? toEs3(VERT, "vertex") : VERT);
  const gpuProgram = webgl2 ? link(VERT_GPU, gpuFragShader) : null;
  // v28 FFT sea: its own vertex shader and mesh; the fragment shader is the GPU one with the
  // lod-lost slope variance added to the sub-grid roughness (so it goes into every lobe).
  const fftText = seaSrc.fft;
  const fftFragShader = webgl2 ? compile(gl, gl.FRAGMENT_SHADER, fftText) : null;
  const fftProgram = webgl2 ? link(VERT_FFT, fftFragShader) : null;
  const samplerReports = {};
  samplerReports.cpu = assertFragmentSamplers(gl, cpuProgram, cpuText, "cpu");
  if (gpuProgram) samplerReports.gpu = assertFragmentSamplers(gl, gpuProgram, gpuText, "gpu");
  if (fftProgram) samplerReports.fft = assertFragmentSamplers(gl, fftProgram, fftText, islandDraw ? "fft+island" : "fft");
  globalThis.__samplerBudget = samplerReports;
  const spray = webgl2 && SEA_PARAMS.get("spray") !== "0" ? createSpray(gl) : null;
  // The plunging curl sheet and the coarse lip ribbon are gone. Breaking stays
  // in the foam field, on the spectral surface.
  const curlSheet = null;
  const lip = null;
  // v29.1 fix5 review check (?edgecheck=1, off by default): runs the SAME vertex shader with the
  // same uniforms and textures over the surface tile vertices and the wall-top vertices through
  // transform feedback and reports the largest surface-edge to wall-top distance and how far any
  // surface vertex reaches past a wall (window.__edgeGap, running max in window.__edgeGapMax).
  const edgeCheck = webgl2 && new URLSearchParams(typeof location !== "undefined" ? location.search : "").get("edgecheck") === "1";
  const edgeTf = { prog: null, n: 0 };
  function edgeCheckRun(L) {
    if (!fftMesh.cpu || !blockMesh.cpu) return;
    if (!edgeTf.prog) {
      const pr = gl.createProgram();
      gl.attachShader(pr, compile(gl, gl.VERTEX_SHADER, VERT_FFT)); gl.attachShader(pr, fftFragShader);
      gl.transformFeedbackVaryings(pr, ["vWorld"], gl.INTERLEAVED_ATTRIBS); gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr) || "edgecheck link failed");
      edgeTf.prog = pr; edgeTf.vao = gl.createVertexArray(); edgeTf.bx = gl.createBuffer(); edgeTf.bm = gl.createBuffer(); edgeTf.out = gl.createBuffer(); edgeTf.tf = gl.createTransformFeedback();
    }
    if (!edgeTf.n) {
      const xs = [], ms = [], kind = [];
      const add = (xz, meta, w) => { for (let i = 0; i < xz.length / 2; i += 1) if (Math.abs(meta[4 * i + 3] - w) < 0.1) { xs.push(xz[2 * i], xz[2 * i + 1]); ms.push(0, 0, 0, w); kind.push(w); } };
      add(fftMesh.cpu.xz, fftMesh.cpu.meta, 1); add(blockMesh.cpu.xz, blockMesh.cpu.meta, 2);
      edgeTf.xs = xs; edgeTf.kind = kind; edgeTf.n = kind.length;
      gl.bindVertexArray(edgeTf.vao);
      const lx = gl.getAttribLocation(edgeTf.prog, "xz"), lm = gl.getAttribLocation(edgeTf.prog, "meta");
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeTf.bx); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(xs), gl.STATIC_DRAW); gl.enableVertexAttribArray(lx); gl.vertexAttribPointer(lx, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeTf.bm); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ms), gl.STATIC_DRAW); gl.enableVertexAttribArray(lm); gl.vertexAttribPointer(lm, 4, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeTf.out); gl.bufferData(gl.ARRAY_BUFFER, edgeTf.n * 12, gl.STREAM_READ);
      gl.bindVertexArray(vao);
    }
    const src = fftProgram, dst = edgeTf.prog;
    gl.useProgram(dst);
    const nU = gl.getProgramParameter(src, gl.ACTIVE_UNIFORMS);
    for (let u = 0; u < nU; u += 1) {
      const info = gl.getActiveUniform(src, u), base = info.name.replace(/\[0\]$/, "");
      for (let k = 0; k < info.size; k += 1) {
        const name = info.size > 1 ? `${base}[${k}]` : info.name;
        const ls = gl.getUniformLocation(src, name), ld = gl.getUniformLocation(dst, name);
        if (!ls || !ld) continue;
        const v = gl.getUniform(src, ls), t = info.type;
        if (t === gl.FLOAT) gl.uniform1f(ld, v);
        else if (t === gl.FLOAT_VEC2) gl.uniform2fv(ld, v); else if (t === gl.FLOAT_VEC3) gl.uniform3fv(ld, v); else if (t === gl.FLOAT_VEC4) gl.uniform4fv(ld, v);
        else if (t === gl.FLOAT_MAT4) gl.uniformMatrix4fv(ld, false, v); else if (t === gl.FLOAT_MAT3) gl.uniformMatrix3fv(ld, false, v);
        else if (t === gl.INT_VEC2) gl.uniform2iv(ld, v); else if (t === gl.INT_VEC3) gl.uniform3iv(ld, v); else if (t === gl.INT_VEC4) gl.uniform4iv(ld, v);
        else gl.uniform1i(ld, Number(v));
      }
    }
    gl.bindVertexArray(edgeTf.vao);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, edgeTf.tf); gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, edgeTf.out);
    gl.beginTransformFeedback(gl.POINTS); gl.drawArrays(gl.POINTS, 0, edgeTf.n); gl.endTransformFeedback();
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null); gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.disable(gl.RASTERIZER_DISCARD);
    const r = new Float32Array(edgeTf.n * 3);
    gl.bindBuffer(gl.COPY_READ_BUFFER, edgeTf.out); gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, r); gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindVertexArray(vao); gl.useProgram(program);
    const half = L / 2, key = (i) => `${Math.round(edgeTf.xs[2 * i] * 1000)},${Math.round(edgeTf.xs[2 * i + 1] * 1000)}`, surf = new Map();
    let over = -Infinity, gap = 0, pairs = 0, unmatched = 0;
    for (let i = 0; i < edgeTf.n; i += 1) if (edgeTf.kind[i] === 1) { surf.set(key(i), i); over = Math.max(over, Math.abs(r[3 * i]) - half, Math.abs(r[3 * i + 2]) - half); }
    for (let i = 0; i < edgeTf.n; i += 1) if (edgeTf.kind[i] === 2) {
      const j = surf.get(key(i)); if (j === undefined) { unmatched += 1; continue; }
      pairs += 1; gap = Math.max(gap, Math.hypot(r[3 * i] - r[3 * j], r[3 * i + 1] - r[3 * j + 1], r[3 * i + 2] - r[3 * j + 2]));
    }
    // fix6: adjacent surface vertices: largest vertical step |dy| and slope |dy| / |dxz| (displaced),
    // for edges within 3 cells of a wall vs the interior
    const N = Math.round(Math.sqrt(surf.size)) - 1, sp = L / N, grid = new Int32Array((N + 1) * (N + 1)).fill(-1);
    for (const [, i] of surf) { const gi = Math.round((edgeTf.xs[2 * i] + half) / sp), gj = Math.round((edgeTf.xs[2 * i + 1] + half) / sp); if (gi >= 0 && gj >= 0 && gi <= N && gj <= N) grid[gj * (N + 1) + gi] = i; }
    let dyE = 0, dyI = 0, slE = 0, slI = 0;
    for (let gj = 0; gj <= N; gj += 1) for (let gi = 0; gi <= N; gi += 1) for (const [di, dj] of [[1, 0], [0, 1]]) {
      if (gi + di > N || gj + dj > N) continue;
      const a = grid[gj * (N + 1) + gi], b = grid[(gj + dj) * (N + 1) + gi + di]; if (a < 0 || b < 0) continue;
      const dy = Math.abs(r[3 * a + 1] - r[3 * b + 1]), dh = Math.hypot(r[3 * a] - r[3 * b], r[3 * a + 2] - r[3 * b + 2]), sl = dy / Math.max(dh, 1e-3);
      const near = Math.min(gi, gj, N - gi - di, N - gj - dj) < 3;
      if (gi + di === N || gj + dj === N) window.__dyWrap = Math.max(window.__dyWrap || 0, dy); else if (near) window.__dyNear = Math.max(window.__dyNear || 0, dy);
      if (near) { dyE = Math.max(dyE, dy); slE = Math.max(slE, sl); } else { dyI = Math.max(dyI, dy); slI = Math.max(slI, sl); }
    }
    const res = { gap, overhang: over, pairs, unmatched, dyEdge: dyE, dyIn: dyI, slopeEdge: slE, slopeIn: slI };
    window.__edgeGap = res;
    const m = window.__edgeGapMax || { gap: 0, overhang: -Infinity, frames: 0, unmatched: 0 };
    window.__edgeGapMax = { gap: Math.max(m.gap, gap), overhang: Math.max(m.overhang, over), frames: m.frames + 1, unmatched: Math.max(m.unmatched, unmatched), pairs,
      dyEdge: Math.max(m.dyEdge || 0, dyE), dyIn: Math.max(m.dyIn || 0, dyI), slopeEdge: Math.max(m.slopeEdge || 0, slE), slopeIn: Math.max(m.slopeIn || 0, slI) };
  }
  const vao = webgl2 ? gl.createVertexArray() : null;
  if (vao) gl.bindVertexArray(vao);
  gl.enable(gl.DEPTH_TEST);
  const cpuLocs = locate(cpuProgram);
  const gpuLocs = gpuProgram ? locate(gpuProgram) : null;
  const fftLocs = fftProgram ? locate(fftProgram) : null;
  if (fftLocs) {
    for (const name of ["physA", "physB", "physL", "physF", "physG", "physK", "foamLace", "foamT", "physS", "strandsOn", "crestLift", "physSF", "fineWin", "fineCam", "foamOff", "waterTurb", "waterSc", "sceneTex", "sceneOn", "sfRaw", "physN", "foamDrift", "curlU", "physC", "dayTint", "bandStats", "fftR0", "fftR1", "fftR2", "fftR3", "fftStats", "fftLevels", "fftL", "choppiness", "pixAngle", "lodBias", "minLod", "farHalve", "oneCopyFar", "tileThin", "ringLodAdd", "antiTile", "block", "xCount", "x0", "x1"]) {
      fftLocs[name] = gl.getUniformLocation(fftProgram, name);
    }
    fftLocs.subMss = gl.getUniformLocation(fftProgram, "subMssU");
    fftLocs.meta = gl.getAttribLocation(fftProgram, "meta");
    fftLocs.clusterTex = gl.getUniformLocation(fftProgram, "clusterTex");
    fftLocs.clusterGrid = gl.getUniformLocation(fftProgram, "clusterGrid");
    fftLocs.seabedLut = gl.getUniformLocation(fftProgram, "seabedLut");
  }
  const fftMesh = { xz: gl.createBuffer(), meta: gl.createBuffer(), index: gl.createBuffer(), count: 0, key: "" };
  const fftParams = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
  const lodBias = Number(fftParams.get("lodBias") || 1);
  const minLod = Number(fftParams.get("minLod") || 0);
  const oneCopyFar = fftParams.has("onecopy") ? Number(fftParams.get("onecopy")) : ONE_COPY_DEFAULT;
  const tileThin = fftParams.has("tilethin") ? Number(fftParams.get("tilethin")) : TILE_THIN_DEFAULT;
  const frontToBack = (fftParams.has("f2b") ? Number(fftParams.get("f2b")) : 1) > 0.5; // v28-lod: nearest chunks first (?f2b=0 off)
  // v28-lod: ?farhalve=1 quarters the ring density past 840 m (default set below after the estimate)
  const farHalve = fftParams.has("farhalve") ? Number(fftParams.get("farhalve")) : FAR_HALVE_DEFAULT;
  const antiTile = fftParams.get("antiTile") === "0" ? 0 : 1;
  const chunkOnly = (fftParams.has("chunk") ? Number(fftParams.get("chunk")) : CHUNK_DEFAULT) > 0.5;
  // v29: the cut-away block (walls + floor) on the chunk; ?block=0 = the v28 chunk alone;
  // ?seafloordebug=1 leaves the surface out (floor colour checks, GAD spec section 4).
  const blockOn = (fftParams.has("block") ? Number(fftParams.get("block")) : 1) > 0.5;
  const floorDebug = fftParams.get("seafloordebug") === "1";
  const blockMesh = { xz: gl.createBuffer(), meta: gl.createBuffer(), index: gl.createBuffer(), count: 0, key: "" };
  // Four walls (top row = the surface edge vertices, meta.w 2; bottom row at the floor, 3) and the
  // floor (meta.w 4), at the tile's own vertex spacing so the wall top meets every surface vertex.
  function buildBlockMesh(N, L) {
    const xz = [], meta = [], idx = [], h = L / 2, s = L / N;
    // QM corner fix: one perimeter loop of 4N top/bottom vertex pairs, so each corner vertex is
    // shared by both walls (one vertex, one displaced height and horizontal shift = the corner texel).
    const P = 4 * N;
    for (let k = 0; k < P; k += 1) {
      const side = Math.floor(k / N), i = k % N;
      const x = [-h + i * s, h, h - i * s, -h][side], z = [-h, -h + i * s, h, h - i * s][side];
      xz.push(x, z, x, z); meta.push(0, 0, 0, 2, 0, 0, 0, 3);
    }
    for (let k = 0; k < P; k += 1) { const a = 2 * k, b = 2 * ((k + 1) % P); idx.push(a, a + 1, b, b, a + 1, b + 1); }
    // v29.1 fix2: the floor is a fan from the centre to the walls' (displaced) bottom vertices
    const f = xz.length / 2;
    // v29.1 fix3: the fan's rim has its OWN floor vertices (meta.w 5: floor kind, shifted like the wall
    // bottom at the same perimeter point). fix2 reused the wall-bottom vertices (kind 1), so vKind was
    // interpolated 2 -> 1 across every fan triangle and the centre square / corner diagonals switched
    // between the floor and the wall shader.
    if (fftParams.get("island") !== "1") {
      xz.push(0, 0); meta.push(0, 0, 0, 4);
      for (let k = 0; k < P; k += 1) { xz.push(xz[2 * (2 * k + 1)], xz[2 * (2 * k + 1) + 1]); meta.push(0, 0, 0, 5); }
      for (let k = 0; k < P; k += 1) idx.push(f, f + 1 + k, f + 1 + ((k + 1) % P));
    } else {
      // A fan from the tile centre draws a straight edge across the island.
      // The floor is a grid instead, so that edge is not there.
      const cols = N + 1;
      const base = xz.length / 2;
      for (let j = 0; j <= N; j += 1) for (let i = 0; i <= N; i += 1) {
        xz.push(-h + i * s, -h + j * s);
        meta.push(0, 0, 0, 6);
      }
      for (let j = 0; j < N; j += 1) for (let i = 0; i < N; i += 1) {
        const a = base + j * cols + i;
        const b = a + 1, c = a + cols, d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, blockMesh.xz); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(xz), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, blockMesh.meta); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meta), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, blockMesh.index); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
    blockMesh.count = idx.length; blockMesh.wallCount = 6 * P; blockMesh.cpu = { xz, meta }; edgeTf.n = 0; blockMesh.key = `${N}/${L}`;
  }
  // v29 coral / rock clusters (GAD spec 2.3): ~3 per 100 m^2 (a 5.4 m jittered cell, 12% empty, so
  // 2-4 locally), 1-3 m across, 3-8 heads 0.3-1.0 m high, instanced low domes. Seeded, so the same
  // size always gives the same reef. Visual only: nothing is read back by the physics.
  const CLUSTER_CELL = 9.9; // AO table cell: holds at most one reef patch centre
  const clusters = { key: "", count: 0, grid: [0, CLUSTER_CELL, 0], tex: gl.createTexture(), program: null, vao: null, inst: null, indexCount: 0 };
  if (webgl2 && blockOn) {
    const vs = `#version 300 es
layout(location=0) in vec3 pos; layout(location=1) in vec4 inst0; layout(location=2) in vec4 inst1; layout(location=3) in vec4 inst2;
uniform mat4 viewProj; uniform float depth;
out vec3 vW; out vec3 vN; out vec3 vCol; out float vY; out float vKind;
void main() {
  float r = inst0.z, h = min(1.8 * inst0.w, 0.8 * depth), a = r * inst1.z;
  float c = cos(inst1.w), s = sin(inst1.w);
  vec2 l = vec2(pos.x * a, pos.z * r); l = vec2(c * l.x - s * l.y, s * l.x + c * l.y);
  vec3 nl = vec3(pos.x / a, pos.y / h, pos.z / r); nl.xz = vec2(c * nl.x - s * nl.z, s * nl.x + c * nl.z);
  // inst2.x is the seat. Spawn sets it so this mesh's lowest vertex
  // (pos.y * h + 0.02) meets the rendered sand and sinks a few centimetres.
  float fy = inst2.x;
  // (x, height, z). A vertex above still water is pulled back under it: a tip
  // at y > 0 is drawn at full colour on top of the sea (the floating reef).
  float y = min(fy + pos.y * h + 0.02, -0.04);
  // Same visual compression as the drawn floor, so a deep head is not left
  // hanging under the shortened cut. Shallow heads stay on the real bank.
  if (y < -4.0) y = mix(0.0, y, 0.38);
  vW = vec3(inst0.x + l.x, y, inst0.y + l.y);
  vN = normalize(nl); vCol = inst2.yzw * inst1.y; vY = pos.y; vKind = inst1.x;
  gl_Position = viewProj * vec4(vW, 1.0);
}`;
    const fs = `#version 300 es
precision highp float;
in vec3 vW; in vec3 vN; in vec3 vCol; in float vY; in float vKind;
out vec4 fragColor;
uniform sampler2D seabedLut;
${lightSampleGlsl()}
uniform mat4 uShVP;
uniform vec3 uSun;
uniform float uCausDay;
uniform float uCausDeep;
uniform float uSeaL;
uniform vec2 uCaustO;
uniform float uCaustTile;
${shadowGlsl()}
uniform float raw; uniform float depth; uniform float uClarK;
const vec3 SF_KD = ${v3(SEAFLOOR.kd)};
const float SF_GAIN = ${Number(SEAFLOOR.gain).toFixed(6)};
const vec3 SF_RINF = ${v3(SEAFLOOR.rInf)};
vec3 sfEncode(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  if (vW.y > -0.02) discard; // above still water: not a surface decal
  vec3 rho = vCol;
  // Face normal of this triangle, so a rock reads as a solid instead of a flat card.
  vec3 nFace = normalize(cross(dFdx(vW), dFdy(vW)));
  vec3 Ls = uSun.y > 0.02 ? normalize(uSun) : vec3(0.2, 0.9, 0.15);
  float nd = clamp(dot(nFace, Ls), 0.0, 1.0);
  // Rocks read as stone: dark, a little algae low down, and the base sits in a contact shadow.
  float rock = step(0.5, vKind) * (1.0 - step(1.5, vKind));
  rho = mix(rho, rho * vec3(0.42, 0.52, 0.38), rock * smoothstep(0.65, 0.12, vY));
  float lit = mix(0.16, 1.35, nd);
  lit *= mix(0.45, 1.0, mix(smoothstep(0.0, 0.18, vY), 1.0, 1.0 - rock));
  // Grass keeps its colour. Coral fades only at the contact; rock only at the very foot.
  float edge = vKind > 1.5 && vKind < 2.5 ? 0.95 : mix(smoothstep(0.0, 0.2, vY), smoothstep(0.0, 0.07, vY), rock);
  rho = mix(${v3(SEAFLOOR.sand)} * 0.7, rho * lit, edge);
  float z = max(-vW.y, 0.0) * uClarK; // v30.24: the clarity's equivalent depth
  vec2 uA = vec2((clamp(z / ${SEABED_LUT.dz}, 0.0, ${SEABED_LUT.n - 1}.0) + 0.5) / ${SEABED_LUT.n}.0, 0.5 / 3.0);
  {
    // Normal offset toward the sun side, so a rock face is not shadowed by
    // its own shadow texel. Neighbouring rocks and the boat still block it.
    vec3 nSun = dot(nFace, Ls) < 0.0 ? -nFace : nFace;
    float sh = shadowFactorFast(vW + nSun * 0.3);
    float col = max(0.05, -vW.y);
    vec3 Rf = refract(-normalize(uSun), vec3(0.0, 1.0, 0.0), 1.0 / 1.333);
    if (Rf.y > -0.02) Rf = vec3(0.0, -1.0, 0.0);
    vec2 xz = vW.xz - vW.y * Rf.xz / Rf.y;
    vec3 ratio = causNetRGB(xz, col, vec2(0.0), causDisp(uSun, col), 1.0); // v32: the same fringe as the sand
    float clar = max(uClarK, 0.35);
    float near = exp(-col * clar / 2.6);
    float deep = uCausDeep * 0.08 * exp(-col * clar / 16.0);
    float strength = clamp(near + deep, 0.0, 1.0) * uCausDay;
    rho *= mix(0.5, 1.0, sh);
    rho *= mix(vec3(1.0), ratio, strength);
  }
  if (raw > 0.5) { fragColor = vec4(sfEncode(clamp(rho, 0.0, 1.0)), clamp((vW.y + depth) / 4.0, 0.0, 1.0)); return; } // v30.5 scene pass
  float gain = SF_GAIN * mix(1.65, 1.35, rock);
  fragColor = vec4(sfEncode(${UPWELL.toFixed(4)} * (texture(seabedLut, uA).rgb + gain * rho * texture(seabedLut, uA + vec2(0.0, 1.0 / 3.0)).rgb)), 1.0);
}`;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) || "cluster link failed");
    samplerReports.reef = assertFragmentSamplers(gl, prog, fs, "reef");
    clusters.program = prog;
    clusters.viewProj = gl.getUniformLocation(prog, "viewProj"); clusters.lutLoc = gl.getUniformLocation(prog, "seabedLut"); clusters.depth = gl.getUniformLocation(prog, "depth"); clusters.raw = gl.getUniformLocation(prog, "raw");
    clusters.sun = gl.getUniformLocation(prog, "uSun");
    clusters.light = gl.getUniformLocation(prog, "uLight");
    clusters.shVP = gl.getUniformLocation(prog, "uShVP");
    clusters.causDay = gl.getUniformLocation(prog, "uCausDay");
    clusters.causDeep = gl.getUniformLocation(prog, "uCausDeep");
    clusters.seaL = gl.getUniformLocation(prog, "uSeaL");
    clusters.caustO = gl.getUniformLocation(prog, "uCaustO");
    clusters.caustTile = gl.getUniformLocation(prog, "uCaustTile");
    clusters.causT = gl.getUniformLocation(prog, "uCausT");
    clusters.causRough = gl.getUniformLocation(prog, "uCausRough");
    // unit dome: 10 segments x 3 rings + apex, base ring slightly flared, flattened top
    const SEG = 10, RINGS = 3, P = [], I = [];
    for (let k = 0; k < RINGS; k += 1) {
      const el = (k / RINGS) * Math.PI / 2, rr = Math.cos(el) * (k === 0 ? 1.08 : 1), y = Math.pow(Math.sin(el), 0.8);
      for (let i = 0; i < SEG; i += 1) { const t = (i / SEG) * 2 * Math.PI; P.push(rr * Math.cos(t), y, rr * Math.sin(t)); }
    }
    P.push(0, 1, 0);
    for (let k = 0; k < RINGS - 1; k += 1) for (let i = 0; i < SEG; i += 1) {
      const a = k * SEG + i, b = k * SEG + (i + 1) % SEG; I.push(a, a + SEG, b, b, a + SEG, b + SEG);
    }
    const apex = RINGS * SEG;
    for (let i = 0; i < SEG; i += 1) I.push((RINGS - 1) * SEG + i, apex, (RINGS - 1) * SEG + (i + 1) % SEG);
    clusters.indexCount = I.length;
    clusters.vao = gl.createVertexArray(); gl.bindVertexArray(clusters.vao);
    const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.STATIC_DRAW);
    const lp = gl.getAttribLocation(prog, "pos"); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(I), gl.STATIC_DRAW);
    clusters.inst = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, clusters.inst);
    for (const [name, off] of [["inst0", 0], ["inst1", 16], ["inst2", 32]]) {
      const l = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, 4, gl.FLOAT, false, 48, off); gl.vertexAttribDivisor(l, 1);
    }
    const bindInst = (vao, buf) => {
      gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      for (const [name, off] of [["inst0", 0], ["inst1", 16], ["inst2", 32]]) {
        const l = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(l);
        gl.vertexAttribPointer(l, 4, gl.FLOAT, false, 48, off); gl.vertexAttribDivisor(l, 1);
      }
    };
    bindInst(clusters.vao, clusters.inst);
    function partMesh(positions, indices) {
      const vaoP = gl.createVertexArray(); gl.bindVertexArray(vaoP);
      const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
      const lp = gl.getAttribLocation(prog, "pos"); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
      const inst = gl.createBuffer();
      bindInst(vaoP, inst);
      return { vao: vaoP, inst, indexCount: indices.length, count: 0 };
    }
    // A tapered cone with a closed top, so a head reads from above.
    const cone = (pos, idx, x0, z0, x1, z1, y0, y1, rad, seg = 6) => {
      const base = pos.length / 3;
      for (let i = 0; i < seg; i += 1) {
        const a = (i / seg) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
        pos.push(x0 + c * rad, y0, z0 + s * rad, x1 + c * rad * 0.45, y1, z1 + s * rad * 0.45);
      }
      const tip = pos.length / 3;
      pos.push(x1, y1 + rad * 0.15, z1);
      for (let i = 0; i < seg; i += 1) {
        const a = base + i * 2, b = base + ((i + 1) % seg) * 2;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
        idx.push(a + 1, b + 1, tip);
      }
    };
    const tubeP = [], tubeI = [];
    [[0, 0, 0, 0, 0.2, 1], [0.34, 0.06, 0.34, 0.06, 0.12, 0.7], [-0.28, 0.18, -0.28, 0.18, 0.11, 0.55], [0.08, -0.32, 0.08, -0.32, 0.13, 0.82], [-0.1, -0.16, -0.1, -0.16, 0.09, 0.42]].forEach(([x, z, x1, z1, rad, h]) => cone(tubeP, tubeI, x, z, x1, z1, 0, h, rad, 6));
    const branchP = [], branchI = [];
    [[0.85, 0.2, 1.15, 0.16], [-0.75, 0.55, 0.95, 0.12], [0.35, -0.9, 1.05, 0.13], [-0.25, -0.7, 0.8, 0.1], [0.15, 0.85, 0.7, 0.09]].forEach(([x, z, h, rad]) => cone(branchP, branchI, 0, 0, x, z, 0.02, h, rad, 5));
    const tableP = [], tableI = [];
    cone(tableP, tableI, 0, 0, 0, 0, 0, 0.35, 0.14, 6);
    {
      const base = tableP.length / 3;
      const seg = 12;
      tableP.push(0, 0.4, 0);
      for (let i = 0; i < seg; i += 1) {
        const a = (i / seg) * Math.PI * 2;
        tableP.push(Math.cos(a) * 1.55, 0.34, Math.sin(a) * 1.55);
      }
      const hub = base;
      for (let i = 0; i < seg; i += 1) tableI.push(hub, hub + 1 + i, hub + 1 + (i + 1) % seg);
    }
    const fanP = [], fanI = [];
    {
      const seg = 8;
      const hubL = fanP.length / 3; fanP.push(-0.06, 0.08, 0);
      const hubR = fanP.length / 3; fanP.push(0.06, 0.08, 0);
      for (let i = 0; i <= seg; i += 1) {
        const a = -Math.PI * 0.15 + (i / seg) * Math.PI * 1.15;
        fanP.push(0, 0.08 + Math.sin(a) * 1.05, Math.cos(a) * 0.85);
      }
      for (let i = 0; i < seg; i += 1) {
        fanI.push(hubL, 2 + i, 3 + i);
        fanI.push(hubR, 3 + i, 2 + i);
      }
    }
    const rockP = [], rockI = [];
    {
      const seg = 8, rings = 4;
      for (let k = 0; k <= rings; k += 1) {
        const t = k / rings, el = (t - 0.5) * Math.PI, rr = Math.cos(el);
        const y = Math.sin(el) * 0.42 + 0.48;
        for (let i = 0; i < seg; i += 1) {
          const a = (i / seg) * Math.PI * 2;
          const wob = 0.82 + 0.18 * Math.sin(2 * a + k);
          rockP.push(Math.cos(a) * rr * wob, y, Math.sin(a) * rr * wob * 0.9);
        }
      }
      for (let k = 0; k < rings; k += 1) for (let i = 0; i < seg; i += 1) {
        const a = k * seg + i, b = k * seg + (i + 1) % seg;
        rockI.push(a, a + seg, b, b, a + seg, b + seg);
      }
    }
    const grassP = [], grassI = [];
    for (let b = 0; b < 5; b += 1) {
      const a = (b / 5) * Math.PI, c = Math.cos(a), s = Math.sin(a);
      const o = grassP.length / 3;
      grassP.push((c - s) * 0.08, 0, (s + c) * 0.08, (c + s) * 0.08, 0, (s - c) * 0.08, c * 0.35, 1, s * 0.35);
      grassI.push(o, o + 1, o + 2);
    }
    const shellP = [], shellI = [];
    for (let i = 0; i < 8; i += 1) {
      const a = (i / 8) * Math.PI * 2;
      shellP.push(Math.cos(a) * 0.7, 0.08 + 0.12 * Math.sin(a * 2), Math.sin(a) * 0.45);
    }
    shellP.push(0, 0.22, 0);
    for (let i = 0; i < 8; i += 1) shellI.push(i, (i + 1) % 8, 8);
    clusters.parts = {
      brain: { vao: clusters.vao, inst: clusters.inst, indexCount: clusters.indexCount, count: 0 },
      branch: partMesh(branchP, branchI),
      table: partMesh(tableP, tableI),
      fan: partMesh(fanP, fanI),
      tube: partMesh(tubeP, tubeI),
      rock: partMesh(rockP, rockI),
      grass: partMesh(grassP, grassI),
      shell: partMesh(shellP, shellI),
    };
    gl.bindVertexArray(vao);
  }
  const REEF_DENSITY = fftParams.has("reefdensity") ? Number(fftParams.get("reefdensity")) : 0.5;
  const latt = (i, j) => { let t = (Math.imul(i, 374761393) + Math.imul(j, 668265263)) >>> 0; t = Math.imul(t ^ (t >>> 13), 1274126177); return ((t ^ (t >>> 16)) >>> 0) / 4294967296; };
  const lowNoise = (x, y) => { const i = Math.floor(x), j = Math.floor(y); let u = x - i, v = y - j; u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    return (latt(i, j) * (1 - u) + latt(i + 1, j) * u) * (1 - v) + (latt(i, j + 1) * (1 - u) + latt(i + 1, j + 1) * u) * v; };
  // GAD 09:52 reef: clusters come in patches of 3-8 with >= 10 m of bare sand between patches
  // (dart throwing, seeded), patch radius 2-6 m and cluster radius 0.5-1.5 m (3x each), so the
  // floor reads as scattered reef patches, not a dot field. ?reefdensity= scales the count
  // relative to the spec's 3 per 100 m^2 (default 0.5). The AO table holds one patch per 9.9 m cell.
  const seabedTex = gl.createTexture();
  if (webgl2) {
    const n = SEABED_LUT.n, d = new Float32Array(n * 3 * 4);
    ["A", "T", "Td"].forEach((k, r) => SEABED_LUT[k].forEach((c, i) => { d.set([c[0], c[1], c[2], 1], 4 * (r * n + i)); }));
    gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, seabedTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, n, 3, 0, gl.RGBA, gl.FLOAT, d);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    gl.activeTexture(gl.TEXTURE0);
  }
  function buildClusters(L, N) {
    const C = CLUSTER_CELL, n = Math.ceil(L / C), x0 = -n * C / 2, half = L / 2, GAP = 10;
    const cells = new Uint8Array(n * n * 4), heads = [];
    let st = 0x5eaf1009 >>> 0;
    const rng = () => { st = (st + 0x6d2b79f5) >>> 0; let t = st; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const depth = Number(document.querySelector("#depth") && document.querySelector("#depth").value) || 40;
    const islandOn = fftParams.get("island") === "1";
    const shore = islandOn ? useIslandSize(L) : null;
    const sc = shore ? Math.min(1.7, Math.max(0.6, shore.scale)) : 1;
    // Lowest local y of each mesh. The shader puts that vertex at
    // seat + baseY * h + 0.02, so the seat has to cancel baseY and the 0.02.
    const MESH_BASE = {
      brain: 0,
      branch: 0.02,
      table: 0,
      fan: 0.08 + Math.sin(-Math.PI * 0.15) * 1.05,
      tube: 0,
      rock: Math.sin(-Math.PI / 2) * 0.42 + 0.48,
      grass: 0,
      shell: 0.08 - 0.12,
    };
    const SINK = 0.04; // metres the lowest vertex is set into the sand
    const SLOPE_MAX = 0.5; // rise/run; steeper than this and a head cannot rest
    const smooth = (e0, e1, x) => { let t = (x - e0) / (e1 - e0); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };
    // One floor vertex, the same y the block mesh writes.
    // Island on: open relief, then the bank lift in fftView (yBed - 0.08,
    // held at -0.14 where the analytic bed is still under 0.2 m).
    // Island off: that vertex is only the centre and the rim; the span is the fan.
    const vertBed = (x, z) => {
      const yOpen = -depth + bedRelief(x, z, L, depth);
      if (!shore) return yOpen;
      const yBed = sandbarBed(x, z, depth, shore);
      const lift = smooth(-depth + 0.2, -depth + 2.2, yBed);
      let yF = Math.max(yOpen, yBed - 0.08);
      if (yBed < 0.2) yF = Math.min(yF, -0.14);
      return yOpen * (1 - lift) + yF * lift;
    };
    const rimXZ = (k) => {
      const hh = L / 2, s = L / N, P = 4 * N;
      k = ((k % P) + P) % P;
      const side = Math.floor(k / N), i = k % N;
      return [[-hh + i * s, -hh], [hh, -hh + i * s], [hh - i * s, hh], [-hh, hh - i * s]][side];
    };
    // Height of the triangle that is actually drawn under (x, z).
    const renderedBed = (x, z) => {
      const hh = L / 2;
      if (!shore) {
        const yC = vertBed(0, 0);
        const ax = Math.abs(x), az = Math.abs(z);
        if (ax < 1e-8 && az < 1e-8) return yC;
        const tHit = hh / Math.max(ax, az);
        const hx = x * tHit, hz = z * tHit, s = L / N;
        let u;
        if (hz <= -hh + 1e-3) u = (hx + hh) / s;
        else if (hx >= hh - 1e-3) u = N + (hz + hh) / s;
        else if (hz >= hh - 1e-3) u = 2 * N + (hh - hx) / s;
        else u = 3 * N + (hh - hz) / s;
        const u0 = Math.floor(u), f = Math.min(1, Math.max(0, u - u0));
        const P = 4 * N, k0 = ((u0 % P) + P) % P, k1 = (k0 + 1) % P;
        const r0 = rimXZ(k0), r1 = rimXZ(k1);
        const yRim = vertBed(r0[0], r0[1]) * (1 - f) + vertBed(r1[0], r1[1]) * f;
        const a = Math.max(ax, az) / hh;
        return yC * (1 - a) + yRim * a;
      }
      const s = L / N;
      let u = (x + hh) / s, v = (z + hh) / s;
      u = Math.min(N - 1e-6, Math.max(0, u));
      v = Math.min(N - 1e-6, Math.max(0, v));
      const i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j;
      const yAt = (ii, jj) => vertBed(-hh + ii * s, -hh + jj * s);
      const y00 = yAt(i, j), y10 = yAt(i + 1, j), y01 = yAt(i, j + 1), y11 = yAt(i + 1, j + 1);
      // Same split as the floor grid: (i,j) (i+1,j) (i,j+1) and the other half.
      if (fu + fv <= 1) return y00 * (1 - fu - fv) + y10 * fu + y01 * fv;
      return y10 * (1 - fv) + y11 * (fu + fv - 1) + y01 * (1 - fu);
    };
    // Disc of the instance, in the same ellipse the shader scales and rotates.
    const footprint = (x, z, rad, aspect, rot) => {
      const pts = [[x, z]];
      const a = rad * aspect * 1.08, b = rad * 1.08, c = Math.cos(rot), sn = Math.sin(rot);
      for (let i = 0; i < 8; i += 1) {
        const t = (i / 8) * Math.PI * 2, lx = Math.cos(t) * a, lz = Math.sin(t) * b;
        pts.push([x + c * lx - sn * lz, z + sn * lx + c * lz]);
      }
      return pts;
    };
    // Seat on the lowest rendered sample in the footprint, a few centimetres
    // into that sand. Reject dry ground, anything above still water, and a
    // slope a head cannot rest on. hShader is min(1.8 * stored, 0.8 * depth).
    const fitHead = (x, z, radWorld, aspect, rot, hWant, baseY) => {
      const pts = footprint(x, z, radWorld, aspect, rot);
      let lowest = Infinity, steep = 0;
      const bedC = renderedBed(x, z);
      if (!(bedC < 0)) return null;
      if (shore && sandbarBed(x, z, depth, shore) >= 0) return null;
      for (let i = 0; i < pts.length; i += 1) {
        const b = renderedBed(pts[i][0], pts[i][1]);
        if (!(b < 0)) return null;
        if (shore && sandbarBed(pts[i][0], pts[i][1], depth, shore) >= 0) return null;
        if (b < lowest) lowest = b;
        if (i > 0) {
          const dist = Math.hypot(pts[i][0] - x, pts[i][1] - z);
          if (dist > 1e-3) steep = Math.max(steep, Math.abs(b - bedC) / dist);
        }
      }
      if (steep > SLOPE_MAX) return null;
      const rise = Math.max(1.25 - baseY, 0.2);
      const hCap = (-0.06 - lowest + SINK) / rise;
      let hShader = Math.min(1.8 * hWant * sc, 0.8 * depth, hCap);
      if (!(hShader >= 0.22)) return null;
      const stored = hShader / 1.8;
      const seat = lowest - SINK - baseY * hShader - 0.02;
      return { bed: lowest, bedC, seat, stored, hShader, steep };
    };
    const stillAt = (x, z) => {
      if (shore && sandbarBed(x, z, depth, shore) >= 0) return 0;
      return Math.max(0, -renderedBed(x, z));
    };
    const CORAL = {
      brain: [[0.55, 0.28, 0.42], [0.62, 0.34, 0.26], [0.48, 0.30, 0.22], [0.72, 0.48, 0.36], [0.70, 0.32, 0.38], [0.80, 0.76, 0.68]],
      branch: [[0.50, 0.24, 0.40], [0.66, 0.38, 0.24], [0.34, 0.46, 0.28], [0.58, 0.30, 0.22], [0.46, 0.28, 0.48], [0.84, 0.80, 0.72]],
      table: [[0.58, 0.32, 0.36], [0.42, 0.48, 0.30], [0.64, 0.42, 0.26], [0.52, 0.26, 0.44], [0.76, 0.55, 0.38], [0.82, 0.78, 0.70]],
      fan: [[0.56, 0.26, 0.42], [0.60, 0.32, 0.24], [0.36, 0.44, 0.30], [0.68, 0.40, 0.32], [0.48, 0.28, 0.36], [0.86, 0.82, 0.74]],
      tube: [[0.46, 0.26, 0.44], [0.58, 0.36, 0.26], [0.38, 0.46, 0.30], [0.64, 0.38, 0.30], [0.52, 0.30, 0.28], [0.80, 0.76, 0.68]],
    };
    const ROCKC = [[0.28, 0.26, 0.23], [0.22, 0.24, 0.2], [0.34, 0.28, 0.22], [0.18, 0.17, 0.16], [0.30, 0.27, 0.22]];
    const GRASSC = [[0.24, 0.40, 0.18], [0.30, 0.42, 0.16], [0.20, 0.34, 0.16]];
    const SHELLC = [[0.70, 0.62, 0.50], [0.62, 0.48, 0.40], [0.78, 0.72, 0.64], [0.58, 0.42, 0.38]];
    const bags = {};
    for (const name of Object.keys(clusters.parts || {})) bags[name] = [];
    const audit = [];
    const put = (name, x, z, rad, h, kind, aspect, rot, rgb) => {
      const bag = bags[name]; if (!bag) return;
      const baseY = MESH_BASE[name] ?? 0;
      const fit = fitHead(x, z, rad * sc, aspect, rot, h, baseY);
      if (!fit) return;
      bag.push(x, z, rad * sc, fit.stored, kind, 0.85 + 0.3 * rng(), aspect, rot, fit.seat, rgb[0], rgb[1], rgb[2]);
      // Drawn base: the shader's lowest vertex, including the 2 cm lift and the under-water cap.
      const base = Math.min(fit.seat + baseY * fit.hShader + 0.02, -0.04);
      const top = Math.min(fit.seat + 1.25 * fit.hShader + 0.02, -0.04);
      audit.push({ x, z, kind, name, bed: fit.bed, base, gap: base - fit.bed, top });
    };
    const pick = (list) => list[Math.floor(rng() * list.length)];
    const want = L * L * 0.03 * REEF_DENSITY, patches = [];
    let nClusters = 0;
    for (let tries = 0; tries < 120000 && nClusters < want; tries += 1) {
      const rp = 2 * Math.pow(3, rng());                         // 2-6 m (3x)
      const px = (rng() - 0.5) * L, pz = (rng() - 0.5) * L;
      // reef zones: patches gather where an ~80 m seeded noise is high (Clark-Evans R ~1), leaving wide bare sand between
      if (rng() > Math.min(1, Math.max(0, (lowNoise(px / 80 + 31.7, pz / 80 + 11.3) - 0.7) / 0.05))) continue;
      if (Math.abs(px) + rp + 1.5 + 0.015 * L > half || Math.abs(pz) + rp + 1.5 + 0.015 * L > half) continue; // fix4: clear of the edge shift
      if (stillAt(px, pz) < 0.05) continue; // dry sand: no coral, no rock
      const ci = Math.floor((px - x0) / C), cj = Math.floor((pz - x0) / C);
      if (cells[4 * (cj * n + ci) + 3]) continue;
      let ok = true;
      for (const q of patches) { const dx = q[0] - px, dz = q[1] - pz, m = q[2] + rp + GAP; if (dx * dx + dz * dz < m * m) { ok = false; break; } }
      if (!ok) continue;
      patches.push([px, pz, rp]);
      const o = 4 * (cj * n + ci);
      cells[o] = Math.round(((px - x0) / C - ci) * 255); cells[o + 1] = Math.round(((pz - x0) / C - cj) * 255);
      cells[o + 2] = Math.min(255, Math.round((rp + 1) / 8 * 255)); cells[o + 3] = 255;
      const k = 3 + Math.floor(rng() * 6), placed = [];
      for (let c = 0, a = 0; c < k && a < 60; a += 1) {
        const rad = 0.5 * Math.pow(3, rng());                       // 0.5-1.5 m (1-3 m across)
        const t = rng() * 2 * Math.PI, d = Math.sqrt(rng()) * Math.max(rp - rad, 0.2);
        const cx = px + d * Math.cos(t), cz = pz + d * Math.sin(t);
        if (placed.some((q) => Math.hypot(q[0] - cx, q[1] - cz) < 0.7 * (q[2] + rad))) continue;
        placed.push([cx, cz, rad]); c += 1; nClusters += 1;
        const stillC = stillAt(cx, cz);
        const H = (0.35 + 0.75 * rng()) * (0.7 + 0.5 * rng());
        if (stillC >= 1) {
          const shape = pick(["brain", "branch", "table", "fan", "tube"]);
          const nHeads = 1 + Math.floor(rng() * 3);
          for (let h = 0; h < nHeads; h += 1) {
            const th = rng() * 2 * Math.PI, dd = Math.sqrt(rng()) * rad * 0.55;
            put(shape, cx + dd * Math.cos(th), cz + dd * Math.sin(th), rad * (0.45 + 0.4 * rng()), h === 0 ? H : H * (0.55 + 0.4 * rng()), 0, 0.7 + 0.6 * rng(), rng() * Math.PI * 2, pick(CORAL[shape]));
          }
          if (stillC < 10 && rng() < 0.4) put("grass", cx, cz, 0.7 + rng() * 0.6, 0.35 + 0.45 * rng(), 2, 1, rng() * Math.PI, pick(GRASSC));
        }
        if (stillC > 0.05 && rng() < 0.7) {
          const nR = 1 + Math.floor(rng() * 2);
          for (let h = 0; h < nR; h += 1) {
            const th = rng() * 2 * Math.PI, dd = Math.sqrt(rng()) * Math.max(rp * 0.8, 0.4);
            const rr = (0.45 + rng() * 0.85) * (rng() < 0.22 ? 1.6 : 1);
            put("rock", cx + dd * Math.cos(th), cz + dd * Math.sin(th), rr, rr * (0.55 + 0.35 * rng()), 1, 0.75 + 0.5 * rng(), rng() * Math.PI * 2, pick(ROCKC));
          }
        }
        if (stillC > 0.3 && rng() < 0.25) put("shell", cx + (rng() - 0.5) * rp, cz + (rng() - 0.5) * rp, 0.18 + 0.2 * rng(), 0.08, 3, 1, rng() * Math.PI * 2, pick(SHELLC));
      }
    }
    gl.activeTexture(gl.TEXTURE14); gl.bindTexture(gl.TEXTURE_2D, clusters.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, cells);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    gl.activeTexture(gl.TEXTURE0);
    let headN = 0;
    for (const [name, part] of Object.entries(clusters.parts)) {
      const data = bags[name] || [];
      gl.bindBuffer(gl.ARRAY_BUFFER, part.inst);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      part.count = data.length / 12;
      headN += part.count;
    }
    clusters.count = headN; clusters.grid = [x0, C, n]; clusters.key = `${L}/${N}/${depth}/${islandOn ? 1 : 0}`;
    let maxGap = -Infinity, above = 0, nan = 0;
    for (const a of audit) {
      if (!Number.isFinite(a.gap) || !Number.isFinite(a.base) || !Number.isFinite(a.bed)) nan += 1;
      else if (a.gap > maxGap) maxGap = a.gap;
      if (!(a.base < 0) || !(a.top < 0) || !(a.bed < 0)) above += 1;
    }
    if (typeof window !== "undefined") {
      window.__reefAudit = audit;
      window.__reefSeat = { n: audit.length, maxGap: audit.length ? maxGap : 0, above, nan, sink: SINK };
    }
    if (typeof window !== "undefined") window.__reefPatches = patches; window.__clusters = { patches: patches.length, clusters: nClusters, heads: clusters.count, perHundredM2: nClusters / (L * L) * 100, minGapM: GAP, parts: Object.fromEntries(Object.entries(clusters.parts).map(([k, p]) => [k, p.count])) };
  }
  // ?top=1: straight down from high up, for the tiling map (review only).
  const topView = fftParams.get("top") === "1";
  const topHeight = Number(fftParams.get("topHeight") || 4200);
  if (gpuLocs) {
    gpuLocs.physA = gl.getUniformLocation(gpuProgram, "physA");
    gpuLocs.physB = gl.getUniformLocation(gpuProgram, "physB");
    gpuLocs.physL = gl.getUniformLocation(gpuProgram, "physL");
    gpuLocs.physF = gl.getUniformLocation(gpuProgram, "physF");
    gpuLocs.physG = gl.getUniformLocation(gpuProgram, "physG");
    gpuLocs.physN = gl.getUniformLocation(gpuProgram, "physN");
    gpuLocs.bandStats = gl.getUniformLocation(gpuProgram, "bandStats");
  }
  let program = cpuProgram;
  let locs = cpuLocs;
  gl.useProgram(program);

  function locate(program) { return {
    xz: gl.getAttribLocation(program, "xz"),
    height: gl.getAttribLocation(program, "height"),
    normal: gl.getAttribLocation(program, "normal"),
    shift: gl.getAttribLocation(program, "shift"),
    broken: gl.getAttribLocation(program, "broken"),
    lip: gl.getAttribLocation(program, "lip"),
    foam: gl.getAttribLocation(program, "foam"),
    flag: gl.getAttribLocation(program, "flag"),
    recent: gl.getAttribLocation(program, "recent"),
    fade: gl.getAttribLocation(program, "fade"),
    age: gl.getAttribLocation(program, "age"),
    ahead: gl.getAttribLocation(program, "ahead"),
    run: gl.getAttribLocation(program, "run"),
    thin: gl.getAttribLocation(program, "thin"),
    viewProj: gl.getUniformLocation(program, "viewProj"),
    lightDir: gl.getUniformLocation(program, "lightDir"),
    eye: gl.getUniformLocation(program, "eye"),
    depth: gl.getUniformLocation(program, "depth"),
    showFlags: gl.getUniformLocation(program, "showFlags"),
    travel: gl.getUniformLocation(program, "travel"),
    chopTex: gl.getUniformLocation(program, "chopTex"),
    chopSize: gl.getUniformLocation(program, "chopSize"),
    chopScale: gl.getUniformLocation(program, "chopScale"),
    chopOn: gl.getUniformLocation(program, "chopOn"),
    fineTex: gl.getUniformLocation(program, "fineTex"),
    fineSize: gl.getUniformLocation(program, "fineSize"),
    fineScale: gl.getUniformLocation(program, "fineScale"),
    fineOn: gl.getUniformLocation(program, "fineOn"),
    subMss: gl.getUniformLocation(program, "subMss"),
    chopMss: gl.getUniformLocation(program, "chopMss"),
    fineMss: gl.getUniformLocation(program, "fineMss"),
    chopKept: gl.getUniformLocation(program, "chopKept"),
    fineKept: gl.getUniformLocation(program, "fineKept"),
    chopN: gl.getUniformLocation(program, "chopN"),
    fineN: gl.getUniformLocation(program, "fineN"),
    anisoMax: gl.getUniformLocation(program, "anisoMax"),
  }; }

  // Fine chop texture: re-uploaded each step, repeating, mipmapped so distant pixels
  // average the ripples instead of aliasing; anisotropic where available so ripples
  // stay sharp across the view at a glancing angle.
  // The finer patch has its own texture on unit 1, set up the same way and with the
  // same range (on Elfouhaily its slopes reach about 0.45 at 25 m/s).
  const CHOP_SCALE = 0.6;
  const FINE_SCALE = 0.6;
  const aniso = gl.getExtension("EXT_texture_filter_anisotropic") || gl.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
  let anisoUsed = 1;
  function makePatchTexture(unit) {
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    if (aniso) {
      const most = gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) || 1;
      gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(16, most));
      anisoUsed = Math.min(16, most);
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 128]));
    gl.generateMipmap(gl.TEXTURE_2D);
    return { tex, unit, version: -1, bytes: null, kept: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]), n: 1 };
  }
  // Share of a patch's slope variance each mip level keeps: the levels are built here the
  // way generateMipmap builds them (2x2 box averages of the level below), on the patch's
  // own slopes, so the shader knows how much slope a mip-averaged pixel no longer shows.
  function keptPerLevel(slopes, n, kept) {
    let level = Float64Array.from(slopes);
    let size = n;
    let base = 0;
    for (let L = 0; L < 8; L += 1) {
      let v = 0;
      for (let i = 0; i < level.length; i += 1) v += level[i] * level[i];
      v /= size * size;
      if (L === 0) base = v || 1;
      kept[L] = size >= 1 ? v / base : 0;
      if (size === 1) { for (let r = L + 1; r < 8; r += 1) kept[r] = 0; break; }
      const half = size >> 1;
      const next = new Float64Array(half * half * 4);
      for (let z = 0; z < half; z += 1) for (let x = 0; x < half; x += 1) for (let c = 0; c < 4; c += 1) {
        const a = ((2 * z) * size + 2 * x) * 4 + c;
        next[(z * half + x) * 4 + c] = 0.25 * (level[a] + level[a + 4] + level[a + size * 4] + level[a + size * 4 + 4]);
      }
      level = next;
      size = half;
    }
  }
  const chopTex = makePatchTexture(0);
  const fineTex = makePatchTexture(1);
  function uploadPatch(target, patch, range) {
    if (!patch || patch.version === target.version) return;
    const count = patch.n * patch.n * 4;
    if (!target.bytes || target.bytes.length !== count) target.bytes = new Uint8Array(count);
    const k = 127 / range;
    for (let i = 0; i < count; i += 1) {
      const v = Math.round(patch.slopes[i] * k) + 128;
      target.bytes[i] = v < 1 ? 1 : v > 255 ? 255 : v;
    }
    gl.activeTexture(gl.TEXTURE0 + target.unit);
    gl.bindTexture(gl.TEXTURE_2D, target.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, patch.n, patch.n, 0, gl.RGBA, gl.UNSIGNED_BYTE, target.bytes);
    gl.generateMipmap(gl.TEXTURE_2D);
    target.version = patch.version;
    target.n = patch.n;
    keptPerLevel(patch.slopes, patch.n, target.kept);
  }

  const buffers = {
    xz: gl.createBuffer(),
    height: gl.createBuffer(),
    normal: gl.createBuffer(),
    shift: gl.createBuffer(),
    broken: gl.createBuffer(),
    lip: gl.createBuffer(),
    foam: gl.createBuffer(),
    flag: gl.createBuffer(),
    recent: gl.createBuffer(),
    fade: gl.createBuffer(),
    age: gl.createBuffer(),
    ahead: gl.createBuffer(),
    run: gl.createBuffer(),
    thin: gl.createBuffer(),
    index: gl.createBuffer(),
  };
  let indexCount = 0;
  let builtFor = 0;
  let builtSize = -1;
  // v30.6 foam lace texture (offline, tools/make_lace.py): grey until loaded, then mipmapped + aniso
  let laceT = null;
  function laceTex() {
    if (laceT) return laceT;
    laceT = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, laceT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([28, 28, 128, 100]));
    const img = new Image();
    img.onload = () => {
      gl.activeTexture(gl.TEXTURE17); gl.bindTexture(gl.TEXTURE_2D, laceT);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      const an = gl.getExtension("EXT_texture_filter_anisotropic");
      if (an) gl.texParameterf(gl.TEXTURE_2D, an.TEXTURE_MAX_ANISOTROPY_EXT, 8);
      gl.activeTexture(gl.TEXTURE0); window.__laceReady = true;
    };
    img.src = "assets/foam-lace.png?v=30.22";
    return laceT;
  }
  // Editing frame. 560 left the block small under the sky; 320 starts to clip a
  // handle. 460 keeps the fixed controls on screen at 1440 and 1024.
  const OPEN_DIST = 460;
  const camera = { yaw: 0.75, pitch: 0.5, distance: OPEN_DIST };
  // v30.22e: the curl size boost (curlx) follows the live camera distance, also when physics steps run without a draw (?nodraw=1)
  globalThis.__oceanCamDistFn = () => camera.distance * lastSizeK;
  let lastSizeK = 1;
  // v29 review camera from the address (?camdist=, ?campitch=, ?camyaw=); the size slider resets the
  // distance as before unless camdist is given.
  const camFixed = fftParams.has("camdist");
  // v29.1 fix4: a Size change keeps the user's yaw, pitch and zoom: the orbit distance scales by
  // new / old size (first build: the v29 default 0.62 x size). It used to reset to 0.62 x size.
  // v30.20 (Michael): the block keeps one displayed size. camera.distance is in DISPLAY units of a
  // 250 m block; the eye sits at distance x size / 250 metres, which is exactly the sea scaled by
  // 250 / size (same FOV), so a bigger Size fits more ocean in the same block (boat, foam, lace and
  // bumps all scale together; physics stays in metres). Zoom limits are display units, so the whole
  // block is reachable at every Size.
  const DISPLAY_SIZE = 250, CAM_MIN = 40, CAM_MAX = 1400;
  // Opening distance frames the whole block plus the left-hand controls. 155 sat the
  // eye inside the 250 m footprint, so those controls filled the settings column.
  // Wheel zoom is unchanged (40..1400).
  const sizeDistance = (oldSize, newSize) => (oldSize > 0 ? camera.distance : OPEN_DIST);
  globalThis.__oceanViewCmd = {
    reset() { camera.yaw = 0.75; camera.pitch = 0.5; camera.distance = OPEN_DIST; },
    dist(v) { if (Number.isFinite(v)) camera.distance = Math.max(CAM_MIN, Math.min(CAM_MAX, v)); },
    fit() {
      const targetOf = () => {
        const V = globalThis.__oceanView, P = globalThis.__oceanUI23 && globalThis.__oceanUI23.proj;
        if (!V || !P) return 0;
        const h = (V.size || 250) / 2;
        const pts = [[-h, 0, -h], [-h, 0, h], [h, 0, -h], [h, 0, h]].map((p) => P(p[0], p[1], p[2])).filter(Boolean);
        if (pts.length < 2) return 0;
        const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
        return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      };
      let guard = 0;
      const step = () => {
        const V = globalThis.__oceanView, span = targetOf();
        if (!V || !span || guard++ > 8) return;
        const ratio = span / (Math.min(V.cw, V.ch) * 0.58);
        if (Math.abs(ratio - 1) < 0.05) return;
        camera.distance = Math.max(CAM_MIN, Math.min(CAM_MAX, camera.distance * ratio));
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    },
  };
  if (camFixed) camera.distance = Number(fftParams.get("camdist"));
  if (fftParams.has("campitch")) camera.pitch = Number(fftParams.get("campitch"));
  if (fftParams.has("camyaw")) camera.yaw = Number(fftParams.get("camyaw"));
  const camTarget = (fftParams.get("camtarget") || "0,0,0").split(",").map(Number); // review only
  globalThis.__oceanCamCtl = { camTarget, camera }; // v30.22c review shots (aim at a curl)
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    camera.yaw -= (event.clientX - lastX) * 0.005;
    camera.pitch = Math.max(0.15, Math.min(1.2, camera.pitch + (event.clientY - lastY) * 0.004));
    lastX = event.clientX;
    lastY = event.clientY;
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    camera.distance = Math.max(CAM_MIN, Math.min(CAM_MAX, camera.distance * (event.deltaY > 0 ? 1.08 : 0.92)));
  }, { passive: false });

  function buildGrid(n, size) {
    const xz = new Float32Array(n * n * 2);
    const half = size / 2;
    const denom = Math.max(n - 1, 1);
    for (let iz = 0; iz < n; iz += 1) {
      for (let ix = 0; ix < n; ix += 1) {
        const index = (iz * n + ix) * 2;
        xz[index] = -half + (ix / denom) * size;
        xz[index + 1] = -half + (iz / denom) * size;
      }
    }
    const indices = [];
    for (let iz = 0; iz < n - 1; iz += 1) {
      for (let ix = 0; ix < n - 1; ix += 1) {
        const a = iz * n + ix;
        indices.push(a, a + 1, a + n, a + 1, a + n + 1, a + n);
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.xz);
    gl.bufferData(gl.ARRAY_BUFFER, xz, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);
    const ext = webgl2 || gl.getExtension("OES_element_index_uint");
    if (!ext) throw new Error("Need OES_element_index_uint");
    indexCount = indices.length;
    builtFor = n;
  }

  function stream(location, buffer, data, size) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
  }

  const noCull = typeof location !== "undefined" && /[?&]nocull=1/.test(location.search);
  let buoyView = null; let boatView = null;
  let lightMaps = null;
  // Menu ink probe: up to 16 exact screen pixels, every 10th frame. A downscale blit of the
  // default framebuffer came back black on this GL, so these are direct reads (a few hundred bytes).
  const lumProbe = { buf: new Uint8Array(4), n: 0 };
  // Synchronous GL readbacks (__oceanLum probe, __sceneErr) run only with ?gldebug=1 or globalThis.__glDebug = true.
  const glDebugUrl = typeof location !== "undefined" && /[?&]gldebug=1/.test(location.search);
  const glDebug = () => glDebugUrl || globalThis.__glDebug === true;
  function sampleBackdrop() {
    if (!glDebug()) return;
    try { sampleBackdropInner(); } catch (e) { globalThis.__oceanLumErr = String(e && e.message || e); }
  }
  function sampleBackdropInner() {
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight; if (w < 8 || h < 8) return;
    const cw = canvas.clientWidth || w, ch = canvas.clientHeight || h;
    if ((lumProbe.n++ % 10) !== 0 && globalThis.__oceanLum) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    while (gl.getError()) { /* drop a stale error so this probe is not blamed for one */ }
    const toL = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const pix = lumProbe.buf;
    // 4x4 screen grid, straight from the frame. A downscale blit came back black on this GL, so these
    // are 16 single pixels, and only every 10th frame.
    const want = Array.isArray(globalThis.__oceanLumWant) ? globalThis.__oceanLumWant : null;
    const pts = [];
    if (want && want.length) for (const q of want) { if (q && pts.length < 16) pts.push(q); }
    else for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) pts.push([((i + 0.5) / 4) * cw, ((j + 0.5) / 4) * ch]);
    const samples = [];
    for (const q of pts) {
      const x = q[0], y = q[1];
      const px = Math.max(0, Math.min(w - 1, Math.round(x * w / cw))), py = Math.max(0, Math.min(h - 1, h - 1 - Math.round(y * h / ch)));
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
      const r = toL(pix[0] / 255), g = toL(pix[1] / 255), b = toL(pix[2] / 255);
      samples.push({ x, y, L: 0.2126 * r + 0.7152 * g + 0.0722 * b });
    }
    globalThis.__oceanLum = { w: cw, h: ch, samples };
    const err = gl.getError(); if (err) globalThis.__oceanLumErr = err;
  }
  const scene = { fb: null, tex: null, rb: null, w: 0, h: 0 };
  let islandDummyTex = null;
  let coastTex = null;
  function islandDummy() {
    if (islandDummyTex) return islandDummyTex;
    islandDummyTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, islandDummyTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return islandDummyTex;
  }
  const ripDbg = fftParams.get("ripdebug") === "1";
  const ripEase = { dir: [0, 1], wind: [0, 1], hs: 0.8, tp: 6, t: 0, ready: false };
  const seaAgeNow = () => {
    const el = document.querySelector("#age");
    const v = el ? Number(el.value) : 0.28;
    return Number.isFinite(v) ? v : 0.28;
  };
  function bindClusterLight() {
    if (!lightMaps || !clusters.sun) return;
    const I = lightMaps.info;
    gl.uniform3f(clusters.sun, I.sun[0], I.sun[1], I.sun[2]);
    gl.uniformMatrix4fv(clusters.shVP, false, I.vp);
    gl.uniform1f(clusters.causDay, I.day);
    if (clusters.causT) gl.uniform1f(clusters.causT, I.t || 0);
    if (clusters.causRough) gl.uniform1f(clusters.causRough, I.rough || 0);
    gl.uniform1f(clusters.causDeep, I.deep);
    gl.uniform1f(clusters.seaL, I.seaL);
    if (clusters.caustO) gl.uniform2f(clusters.caustO, I.origin[0], I.origin[1]);
    if (clusters.caustTile) gl.uniform1f(clusters.caustTile, I.tile);
    gl.activeTexture(gl.TEXTURE20); gl.bindTexture(gl.TEXTURE_2D, lightMaps.tex);
    if (clusters.light) gl.uniform1i(clusters.light, 20);
    gl.activeTexture(gl.TEXTURE0);
  }
  function seaFoot(eye, target, aspect) {
    const fy = Math.tan(Math.PI / 8);
    const fx = fy * Math.max(0.2, aspect);
    const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    const zl = Math.hypot(zx, zy, zz) || 1;
    const z = [zx / zl, zy / zl, zz / zl];
    const xl = Math.hypot(z[2], z[0]) || 1;
    const x = [z[2] / xl, 0, -z[0] / xl];
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity, n = 0;
    for (const su of [-1, 0, 1]) for (const sv of [-1, 0, 1]) {
      const dx = -z[0] + x[0] * su * fx + y[0] * sv * fy;
      const dy = -z[1] + x[1] * su * fx + y[1] * sv * fy;
      const dz = -z[2] + x[2] * su * fx + y[2] * sv * fy;
      const dl = Math.hypot(dx, dy, dz) || 1;
      const diry = dy / dl;
      if (diry > -0.04) continue;
      const t = -eye[1] / diry;
      if (!(t > 0 && t < 900)) continue;
      const hx = eye[0] + dx / dl * t, hz = eye[2] + dz / dl * t;
      if (hx < x0) x0 = hx;
      if (hx > x1) x1 = hx;
      if (hz < z0) z0 = hz;
      if (hz > z1) z1 = hz;
      n += 1;
    }
    if (!n) return null;
    return { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, half: Math.max(36, Math.max(x1 - x0, z1 - z0) * 0.5 * 1.7) };
  }
  function draw(state, showFlags) {
    // The GPU physics shares this context: take back the program, vertex state, target
    // and depth test every frame.
    const fromTextures = !!(state.gpu && gpuProgram);
    const fft = fromTextures && state.gpu.fft && fftProgram ? state.gpu.fft : null;
    program = fft ? fftProgram : fromTextures ? gpuProgram : cpuProgram;
    locs = fft ? fftLocs : fromTextures ? gpuLocs : cpuLocs;
    gl.useProgram(program);
    {
      const iu = gl.getUniformLocation(program, "uIsland");
      if (iu) {
        const on = state.island && state.island.on ? 1 : 0;
        gl.uniform1f(iu, on);
        const C = state.island || {};
        const shore = useIslandSize(state.size || 250);
        gl.uniform2f(gl.getUniformLocation(program, "uIslandC"), Number.isFinite(C.cx) ? C.cx : shore.cx, Number.isFinite(C.cz) ? C.cz : shore.cz);
        const uf = (name, v) => { const loc = gl.getUniformLocation(program, name); if (loc) gl.uniform1f(loc, v); };
        uf("uIRb", shore.Rberm); uf("uIRf", shore.Rface); uf("uIRs", shore.Rshelf); uf("uIRbank", shore.Rbank);
        uf("uIEase", shore.ease); uf("uIPeak", shore.peak); uf("uIYb", shore.yBerm); uf("uIYf", shore.yFace);
        uf("uIYs", shore.yShelf); uf("uIFace", shore.faceSlope); uf("uIShelf", shore.shelfSlope);
        gl.uniform2f(gl.getUniformLocation(program, "uMapOrigin"), Number.isFinite(C.ox) ? C.ox : -200, Number.isFinite(C.oz) ? C.oz : -200);
        gl.uniform1f(gl.getUniformLocation(program, "uMapSpan"), C.mapSpan || 400);
        gl.uniform1f(gl.getUniformLocation(program, "uHm"), C.mapN || 512);
        gl.uniform1f(gl.getUniformLocation(program, "uAtlasH"), C.atlasH || 576);
        gl.uniform1f(gl.getUniformLocation(program, "uAtlasW"), C.atlasW || C.mapN || 512);
        gl.uniform1f(gl.getUniformLocation(program, "uR2"), C.r2 > 0 ? C.r2 : 0.6);
        gl.uniform1f(gl.getUniformLocation(program, "uBeach"), BEACH_MODE);
        gl.uniform1f(gl.getUniformLocation(program, "uDLo"), C.dLo || 2);
        gl.uniform1f(gl.getUniformLocation(program, "uDHi"), C.dHi || 6);
        gl.uniform1f(gl.getUniformLocation(program, "uSweN"), C.sweN || 256);
        gl.uniform2f(gl.getUniformLocation(program, "uSweOrigin"), C.sweOx || -90, C.sweOz || -90);
        gl.uniform1f(gl.getUniformLocation(program, "uSweSpan"), C.sweSpan || 180);
        gl.uniform1f(gl.getUniformLocation(program, "uHs"), C.hs > 0 ? C.hs : 0.2);
        const steep = window.__steep;
        gl.uniform1f(gl.getUniformLocation(program, "uSteep"), steep && Number.isFinite(steep.S) ? steep.S : 0.56);
        const cs = gl.getUniformLocation(program, "coastState");
        if (cs) {
          // Unit 15, not 21. The vertex stage on a Mac only has 16 units
          // (0..15). Sampling 21 from the vertex shader returned a huge or
          // NaN height and the sea sheet tore. The seabed lookup is fragment
          // only, so it moves to 21 for this program. Reef draws rebind 15.
          coastTex = (on && C.tex) || islandDummy();
          gl.activeTexture(gl.TEXTURE15);
          gl.bindTexture(gl.TEXTURE_2D, coastTex);
          gl.uniform1i(cs, 15);
          gl.activeTexture(gl.TEXTURE0);
        } else coastTex = null;
      }
    }
    if (vao) gl.bindVertexArray(vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    const width = canvas.clientWidth || 1;
    const height = canvas.clientHeight || 1;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    if (fft && fftMesh.key !== `${fft.N}/${fft.L}`) {
      const m = buildFftMesh(fft.N, fft.L, chunkOnly || blockOn ? 0 : undefined, farHalve > 0.5); // chunk: reach 0 = no rings
      fftMesh.cpu = m; edgeTf.n = 0;
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.xz);
      gl.bufferData(gl.ARRAY_BUFFER, m.xz, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.meta);
      gl.bufferData(gl.ARRAY_BUFFER, m.meta, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, fftMesh.index);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.index, gl.STATIC_DRAW);
      fftMesh.count = m.fullCount ?? m.index.length; // v28-lod: the thin tile lists sit after fullCount
      fftMesh.chunks = m.chunks;
      fftMesh.key = `${fft.N}/${fft.L}`;
    }
    if (!fft && (builtFor !== state.n || builtSize !== state.size)) {
      buildGrid(state.n, state.size);
      if (builtSize !== state.size) camera.distance = sizeDistance(builtSize, state.size);
      builtSize = state.size;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    const dayT = globalThis.__oceanDay || null; // v30.22b time of day (null = 11:00, untouched)
    if (dayT) gl.clearColor(dayT[0], dayT[1], dayT[2], 1); else
    gl.clearColor(1, 1, 1, 1); // v29.2.1: white backdrop (Michael); sky reflection and lighting still use SKY_HORIZON
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (fft && builtSize !== state.size) {
      if (!camFixed) camera.distance = sizeDistance(builtSize, state.size);
      builtSize = state.size;
    }
    const sizeK = Math.max(0.1, (state.size || DISPLAY_SIZE) / DISPLAY_SIZE); // v30.20 display scale
    const camDist = camera.distance * sizeK; lastSizeK = sizeK;
    globalThis.__oceanCamDist = camDist; // v30.22e: the curl size boost (curlx) follows the camera distance
    const eye = topView ? [0, topHeight, 0] : [
      camTarget[0] + Math.cos(camera.pitch) * Math.sin(camera.yaw) * camDist,
      camTarget[1] + Math.sin(camera.pitch) * camDist,
      camTarget[2] + Math.cos(camera.pitch) * Math.cos(camera.yaw) * camDist,
    ];
    const view = lookAt(eye, topView ? [0, 0, 0] : camTarget, topView ? [0, 0, -1] : [0, 1, 0]);
    globalThis.__oceanCam = topView ? [0, 0] : [camTarget[0], camTarget[2]]; // v30.14 fine foam window follows the camera
    // v28: the FFT sea reaches past 2 km.
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, (fft ? 1 : 0.5) * Math.max(1, sizeK * 0.5), Math.max(fft ? (topView ? topHeight + 200 : 4500) : 2000, camDist + 1.5 * (state.size || 0) + 500)); // v30.20 far plane follows the zoom
    const vpMat = multiply(proj, view);
    if (drawSkyDome(gl, vpMat)) { gl.useProgram(program); if (vao) gl.bindVertexArray(vao); } // v30.23 sky dome backdrop
    gl.uniformMatrix4fv(locs.viewProj, false, vpMat);
    const sunU = globalThis.__oceanSun || sun; // v30.21: time-of-day slider (identity at 11:00)
    gl.uniform3f(locs.lightDir, sunU[0], sunU[1], sunU[2]);
    {
      const bu = gl.getUniformLocation(program, "uBoat");
      const b = state && state.boat;
      if (bu) gl.uniform4f(bu, b ? b.x : 0, b ? b.z : 0, b ? Math.max(0.2, b.y + 1.2) : 0, b ? 1 : 0);
      const su = gl.getUniformLocation(program, "uShadeSun");
      const sd = globalThis.__oceanShadeSun;
      if (su) gl.uniform3f(su, sd ? sd[0] : sunU[0], sd ? sd[1] : sunU[1], sd ? sd[2] : sunU[2]);
    }
    globalThis.__oceanView = { vp: vpMat, eye, w: canvas.width, h: canvas.height, cw: canvas.clientWidth, ch: canvas.clientHeight, size: state.size, sun };
    gl.uniform3f(locs.eye, eye[0], eye[1], eye[2]);
    gl.uniform1f(locs.depth, state.depth);
    if (locs.sceneOn) gl.uniform1f(locs.sceneOn, 0);
    gl.uniform1f(locs.showFlags, showFlags ? 1 : 0);
    { const af = gl.getUniformLocation(program, "uAirOn"); if (af) gl.uniform1f(af, globalThis.__airVis === 0 ? 0 : 1); }
    gl.uniform2f(locs.travel, state.travelX || 0, state.travelZ || 1);
    {
      // Wind-sea band eases over 8 s. The bed follows that, not the slider.
      const now = performance.now() / 1000;
      const forced = Number(globalThis.__rippleDt);
      const dt = Number.isFinite(forced) ? forced : (ripEase.t ? Math.min(0.25, now - ripEase.t) : 0);
      if (Number.isFinite(forced)) globalThis.__rippleDt = undefined;
      ripEase.t = now;
      const follow = ripEase.ready ? 1 - Math.exp(-dt / 8) : 1;
      const tx = state.travelX || 0, tz = state.travelZ || 1;
      const tl = Math.hypot(tx, tz) || 1;
      const target = [tx / tl, tz / tl];
      ripEase.dir[0] += (target[0] - ripEase.dir[0]) * follow;
      ripEase.dir[1] += (target[1] - ripEase.dir[1]) * follow;
      const dl = Math.hypot(ripEase.dir[0], ripEase.dir[1]) || 1;
      ripEase.dir[0] /= dl; ripEase.dir[1] /= dl;
      let hs = 0.4, tp = 6;
      if (state.bands) {
        for (const b of state.bands) hs += (b.significantHeight || 0) ** 2;
        hs = Math.sqrt(hs) || 0.4;
        const w = state.bands.find((b) => b.id === "wind-sea");
        const om = w && Number.isFinite(w.omegaMin) ? (w.omegaMin + w.omegaMax) * 0.5 : 1.3;
        tp = (6.2831853 / om) * (0.75 + seaAgeNow());
      }
      ripEase.hs += (hs - ripEase.hs) * follow;
      ripEase.tp += (tp - ripEase.tp) * follow;
      ripEase.ready = true;
      const windA = state.fx && Number.isFinite(state.fx.dir) ? state.fx.dir : Math.atan2(target[1], target[0]);
      const windT = [Math.cos(windA), Math.sin(windA)];
      ripEase.wind[0] += (windT[0] - ripEase.wind[0]) * follow;
      ripEase.wind[1] += (windT[1] - ripEase.wind[1]) * follow;
      const wl = Math.hypot(ripEase.wind[0], ripEase.wind[1]) || 1;
      ripEase.wind[0] /= wl; ripEase.wind[1] /= wl;
      const curEl = document.querySelector("#cur");
      const curSpd = curEl ? Number(curEl.value) : 0;
      const curDir = windA + (curSpd < 0 ? Math.PI : 0);
      const wsp0 = state.fx && Number.isFinite(state.fx.speed) ? state.fx.speed : 0;
      const calm0 = wsp0 <= 0.4 ? 0 : Math.min(1, (wsp0 - 0.4) / 2.6);
      setBedRippleDir(ripEase.dir[0], ripEase.dir[1], calm0);
      const uf = (name, v) => { const loc = gl.getUniformLocation(program, name); if (loc) gl.uniform1f(loc, v); };
      const u2 = (name, x, y) => { const loc = gl.getUniformLocation(program, name); if (loc) gl.uniform2f(loc, x, y); };
      u2("uRipDir", ripEase.dir[0], ripEase.dir[1]);
      uf("uBedRip", calm0);
      u2("uWindDir", ripEase.wind[0], ripEase.wind[1]);
      u2("uCurDir", Math.cos(curDir), Math.sin(curDir));
      uf("uCurSpd", Math.abs(curSpd) || 0);
      uf("uTp", Math.max(ripEase.tp, 1));
      uf("uHsRip", Math.max(ripEase.hs, 0.0) * calm0);
      uf("uRipDbg", ripDbg ? 1 : 0);
      const shore = useIslandSize(state.size || 250);
      const islandOn = state.island && state.island.on ? 1 : 0;
      uf("uHasIsland", islandOn);
      u2("uRipC", shore.cx, shore.cz);
      uf("uRipBank", shore.Rbank);
      uf("uRipFace", shore.Rface);
      const ang = Math.atan2(ripEase.dir[1], ripEase.dir[0]);
      const tang = Math.atan2(target[1], target[0]);
      globalThis.__rippleNow = { dir: ang, target: tang, hs: ripEase.hs, tp: ripEase.tp, follow };
    }
    // GPU physics: the card already wrote the patch textures (gpuPatch.js); otherwise
    // quantise and upload the CPU patches as before.
    const gpuChop = fromTextures && state.chopPatch && state.chopPatch.gpu;
    const gpuFine = fromTextures && state.finePatch && state.finePatch.gpu;
    if (!gpuChop) uploadPatch(chopTex, state.chopPatch, CHOP_SCALE);
    if (!gpuFine) uploadPatch(fineTex, state.finePatch, FINE_SCALE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gpuChop ? gpuChop.slopeTex : chopTex.tex);
    gl.uniform1i(locs.chopTex, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, gpuFine ? gpuFine.slopeTex : fineTex.tex);
    gl.uniform1i(locs.fineTex, 1);
    if (fromTextures && locs.bandStats) {
      copyBandStats(gpuChop && gpuChop.statsTex, gpuFine && gpuFine.statsTex);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, bandStatsTex);
      gl.uniform1i(locs.bandStats, 7);
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(locs.chopSize, state.chopPatch ? state.chopPatch.size : 1);
    gl.uniform1f(locs.chopScale, CHOP_SCALE);
    gl.uniform1f(locs.chopOn, state.chopPatch && !chopOff ? 1 : 0);
    gl.uniform1f(locs.fineSize, state.finePatch ? state.finePatch.size : 1);
    gl.uniform1f(locs.fineScale, FINE_SCALE);
    gl.uniform1f(locs.fineOn, state.finePatch && state.finePatch.modes && !chopOff && !fineOff ? 1 : 0);
    const rough = state.roughness || { subGrid: 0, coarse: 0, fine: 0 };
    gl.uniform1f(locs.subMss, rough.subGrid);
    gl.uniform1f(locs.chopMss, state.chopPatch ? rough.coarse : 0);
    gl.uniform1f(locs.fineMss, state.finePatch ? rough.fine : 0);
    if (!fromTextures) {
      gl.uniform1fv(locs.chopKept, chopTex.kept);
      gl.uniform1fv(locs.fineKept, fineTex.kept);
    }
    gl.uniform1f(locs.chopN, gpuChop ? state.chopPatch.n : chopTex.n);
    gl.uniform1f(locs.fineN, gpuFine ? state.finePatch.n : fineTex.n);
    gl.uniform1f(locs.anisoMax, anisoUsed);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.xz);
    // One vertex array for both programs: only the attributes this program reads stay on.
    for (let i = 0; i < 16; i += 1) gl.disableVertexAttribArray(i);
    gl.enableVertexAttribArray(locs.xz);
    gl.vertexAttribPointer(locs.xz, 2, gl.FLOAT, false, 0, 0);
    if (fft) {
      const g = state.gpu;
      gl.disableVertexAttribArray(locs.xz);
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.xz);
      gl.enableVertexAttribArray(locs.xz);
      gl.vertexAttribPointer(locs.xz, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.meta);
      gl.enableVertexAttribArray(locs.meta);
      gl.vertexAttribPointer(locs.meta, 4, gl.FLOAT, false, 0, 0);
      const units = [["physA", g.A], ["physB", g.B], ["physL", g.L], ["physF", g.F], ["physG", g.K || g.G],
        ["fftR0", fft.R[0]], ["fftR1", fft.R[1]], ["fftR2", fft.R[2]], ["fftR3", fft.R[3]], ["fftStats", fft.stats]];
      const unitOf = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13];
      units.forEach(([name, tex], i) => {
        gl.activeTexture(gl.TEXTURE0 + unitOf[i]);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(locs[name], unitOf[i]);
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(locs.physN, fft.N);
      if (locs.dayTint) { const d = globalThis.__oceanDay; gl.uniform3f(locs.dayTint, d ? d[0] : 1, d ? d[1] : 1, d ? d[2] : 1); }
      setV23Uniforms(gl, program, state);
      if (state.fx) { // v30b foam drift over a foam life: ~3.5% of the wind + 0.4 m/s Stokes drift, times the foam window
        const win = 2 * (1 + 2 * Math.min(1, Math.max(0, (state.fx.speed - 10) / 15)));
        const v = (0.035 * state.fx.speed + 0.4) * win;
        gl.uniform2f(locs.foamDrift, v * Math.cos(state.fx.dir), v * Math.sin(state.fx.dir));
      }
      if (locs.curlU) gl.uniform1f(locs.curlU, state.fx ? state.fx.speed : 0);
      if (locs.physC) { gl.activeTexture(gl.TEXTURE19); gl.bindTexture(gl.TEXTURE_2D, g.CU || null); gl.activeTexture(gl.TEXTURE0); gl.uniform1i(locs.physC, 19); } // v30.22c curl field (vertex only)
      if (locs.waterTurb) { const w = waterFromZsd(state.clarity ?? 22.3024); gl.uniform1f(locs.waterTurb, w.t); if (locs.waterSc) gl.uniform2f(locs.waterSc, w.sa, w.sb); }
      if (locs.foamLace) { gl.activeTexture(gl.TEXTURE17); gl.bindTexture(gl.TEXTURE_2D, laceTex()); gl.activeTexture(gl.TEXTURE0); gl.uniform1i(locs.foamLace, 17); }
      if (state.fx && locs.foamT) gl.uniform1f(locs.foamT, state.fx.t);
      if (locs.physS) { gl.activeTexture(gl.TEXTURE18); gl.bindTexture(gl.TEXTURE_2D_ARRAY, g.STR || null); gl.activeTexture(gl.TEXTURE0); gl.uniform1i(locs.physS, 18); }
      if (locs.strandsOn) gl.uniform1f(locs.strandsOn, g.STR ? 1 : 0);
      if (locs.fineWin) gl.uniform1f(locs.fineWin, g.fineOn ? g.fineWin : 0);
      if (locs.fineCam) { const c = globalThis.__oceanCam || [0, 0]; gl.uniform2f(locs.fineCam, c[0], c[1]); }
      if (locs.crestLift) gl.uniform1f(locs.crestLift, g.STR && state.fx && PHYS_Q.get("lift") !== "0" ? Math.min(0.8, Math.max(0.3, 0.032 * state.fx.speed)) : 0);
      if (locs.physK) gl.uniform1i(locs.physK, 6); // v30c: coverage shares unit 6 (the old flag field is no longer drawn)
      if (state.fx && locs.foamOff) { const dr = 0.03 * state.fx.speed + 0.25; gl.uniform2f(locs.foamOff, dr * Math.cos(state.fx.dir) * state.fx.t, dr * Math.sin(state.fx.dir) * state.fx.t); }
      gl.uniform1f(locs.fftLevels, fft.levels);
      gl.uniform1f(locs.fftL, fft.L);
      gl.uniform1f(locs.choppiness, fft.choppiness);
      gl.uniform1f(locs.pixAngle, (Math.PI / 4) / canvas.height);
      gl.uniform1f(locs.lodBias, lodBias);
      gl.uniform1f(locs.minLod, minLod);
      gl.uniform1f(locs.farHalve, farHalve);
      gl.uniform1f(locs.tileThin, tileThin);
      gl.uniform1f(locs.ringLodAdd, fftParams.has("ringlodadd") ? Number(fftParams.get("ringlodadd")) : 0);
      gl.uniform1f(locs.oneCopyFar, oneCopyFar);
      gl.uniform1f(locs.antiTile, antiTile);
      const count = Math.min(fft.explicitCount, 64);
      gl.uniform1i(locs.xCount, count);
      if (count > 0) {
        gl.uniform4fv(locs.x0, fft.explicit0.subarray(0, count * 4));
        gl.uniform4fv(locs.x1, fft.explicit1.subarray(0, count * 4));
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, fftMesh.index);
      // v28 perf: only the chunks that can reach the screen (?nocull=1 draws all, for checks)
      const ranges = noCull ? [[0, fftMesh.count]] : visibleRanges(fftMesh.chunks, vpMat, 60, tileThin > 0.5 || frontToBack ? eye : null, tileThin > 0.5 ? TILE_THIN_D1 + 1 : Infinity, frontToBack);
      fftMesh.drawn = ranges.reduce((a, r) => a + r[1], 0);
      if (typeof window !== "undefined") window.__fftDrawn = [fftMesh.drawn / 3, fftMesh.count / 3];
      gl.uniform1f(locs.block, blockOn ? 1 : 0);
      if (blockOn) {
        // v29: floor and walls first (opaque), then the surface blended over the floor by its
        // see-through share (alpha written stays 1 so the canvas never shows the page through).
        if (blockMesh.key !== `${fft.N}/${fft.L}`) buildBlockMesh(fft.N, fft.L);
        gl.bindBuffer(gl.ARRAY_BUFFER, blockMesh.xz);
        gl.vertexAttribPointer(locs.xz, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, blockMesh.meta);
        gl.vertexAttribPointer(locs.meta, 4, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, blockMesh.index);
        const reefKey = `${fft.L}/${fft.N}/${state.depth}/${state.island && state.island.on ? 1 : 0}`;
        if (clusters.program && clusters.key !== reefKey) buildClusters(fft.L, fft.N);
        gl.activeTexture(gl.TEXTURE14); gl.bindTexture(gl.TEXTURE_2D, clusters.tex); gl.activeTexture(gl.TEXTURE0);
        gl.uniform1i(locs.clusterTex, 14);
        gl.activeTexture(gl.TEXTURE21); gl.bindTexture(gl.TEXTURE_2D, seabedTex); gl.activeTexture(gl.TEXTURE0); gl.uniform1i(locs.seabedLut, 21);
        gl.uniform3f(locs.clusterGrid, clusters.grid[0], clusters.grid[1], clusters.program ? clusters.grid[2] : 0);
        if (webgl2 && !lightMaps) lightMaps = createLightMaps(gl);
        if (lightMaps && state.gpu) {
          if (!boatView) boatView = createBoatView(gl);
          if (globalThis.__oceanBuoy && !buoyView) buoyView = createBuoyView(gl);
          const tLight = performance.now();
          lightMaps.update({
            physA: state.gpu.A, physB: state.gpu.B, physN: fft.N, L: fft.L, depth: state.depth,
            sun: sunU, boatView, buoyView, boat: state.boat, buoy: globalThis.__oceanBuoy,
            clusters: clusters.program ? clusters : null,
            island: state.island && state.island.on ? useIslandSize(state.size || fft.L) : null,
            bedS: state.size || fft.L,
            rip: ripEase.dir,
            bedRip: state.fx && state.fx.speed > 0.4 ? Math.min(1, (state.fx.speed - 0.4) / 2.6) : 0,
            camXZ: [camTarget[0] || 0, camTarget[2] || 0],
            camDist: camera.distance,
            foot: seaFoot(eye, camTarget, canvas.width / Math.max(1, canvas.height)),
            bisect: fftParams.get("bisect") === "off" ? 1 : fftParams.get("bisect") === "read1" ? 2 : fftParams.get("bisect") === "noshadow" ? 3 : 0,
            t: state.fx ? state.fx.t : 0,
            wind: state.fx ? state.fx.speed : 0,
          });
          lightMaps.clar = (state.clarity ?? 22.3) > 1 ? 22.3 / (state.clarity ?? 22.3) : 4;
          globalThis.__lightMaps = lightMaps;
          lightMaps.bindSea(program);
          globalThis.__lightCost.frameAt = performance.now() - tLight;
        }
        // The caustic pass binds the height field on units 0 and 1. The chop
        // patches live there. Put them back before the floor, the walls and
        // the surface. This is the same call for island on and island off.
        // On a flat sea the height field is (0, 0, 0, 1); read as a patch it
        // is a constant slope, and the patch's phase turn paints that into
        // the pale islands.
        const bindChop = () => {
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, gpuChop ? gpuChop.slopeTex : chopTex.tex);
          gl.uniform1i(locs.chopTex, 0);
          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, gpuFine ? gpuFine.slopeTex : fineTex.tex);
          gl.uniform1i(locs.fineTex, 1);
          gl.activeTexture(gl.TEXTURE0);
        };
        bindChop();
        // v30.5 scene pass: floor + rocks/corals as bare albedo (+ rock height in alpha) into sceneTex,
        // read by the surface at its refracted ray's hit point.
        timed("render:scene", () => {
          const w = canvas.width, h = canvas.height;
          if (!scene.fb || scene.w !== w || scene.h !== h) {
            if (scene.fb) { gl.deleteFramebuffer(scene.fb); gl.deleteTexture(scene.tex); gl.deleteRenderbuffer(scene.rb); }
            // Unit 16, not 0. Creating the target on the active unit replaced
            // the chop patch, and the next draws sampled the scene colour as a slope.
            gl.activeTexture(gl.TEXTURE16);
            scene.tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, scene.tex);
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.activeTexture(gl.TEXTURE0);
            scene.rb = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, scene.rb); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
            scene.fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, scene.fb);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, scene.tex, 0);
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, scene.rb);
            scene.w = w; scene.h = h;
          }
          gl.bindFramebuffer(gl.FRAMEBUFFER, scene.fb);
          gl.viewport(0, 0, w, h);
          // Black, not pale sand. A miss must not read as a sand albedo.
          gl.clearColor(0.0, 0.0, 0.0, 0.0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
          const blendWas = gl.isEnabled(gl.BLEND); gl.disable(gl.BLEND); // v30.6 fix: raw albedo has alpha 0 (rock height); blending dropped the floor
          gl.uniform1f(locs.sfRaw, 1);
          // The sea shader samples sceneTex. Binding that unit to null makes
          // the floor draw illegal, so the pass stayed black and the water
          // never saw the caustic-lit sand. A 1x1 stand-in keeps the draw
          // legal without sampling the texture we are writing.
          gl.activeTexture(gl.TEXTURE16); gl.bindTexture(gl.TEXTURE_2D, islandDummy()); gl.activeTexture(gl.TEXTURE0);
          gl.enable(gl.POLYGON_OFFSET_FILL);
          gl.polygonOffset(1.5, 2.0);
          if (vao) gl.bindVertexArray(vao);
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, blockMesh.index);
          gl.drawElements(gl.TRIANGLES, blockMesh.count - blockMesh.wallCount, gl.UNSIGNED_INT, blockMesh.wallCount * 4);
          if (glDebug()) globalThis.__sceneErr = gl.getError();
          gl.disable(gl.POLYGON_OFFSET_FILL);
          if (clusters.program && clusters.count > 0) {
            gl.useProgram(clusters.program);
            gl.uniformMatrix4fv(clusters.viewProj, false, vpMat); gl.uniform1f(clusters.depth, state.depth); setBedOnly(gl, clusters.program, state.size); bindClusterLight(); gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, seabedTex);
            gl.uniform1i(clusters.lutLoc, 15); gl.uniform1f(clusters.raw, 1);
            for (const part of Object.values(clusters.parts)) if (part.count) { gl.bindVertexArray(part.vao); gl.drawElementsInstanced(gl.TRIANGLES, part.indexCount, gl.UNSIGNED_SHORT, 0, part.count); }
            gl.uniform1f(clusters.raw, 0);
            gl.bindVertexArray(vao); gl.useProgram(program);
            if (coastTex) { gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, coastTex); gl.activeTexture(gl.TEXTURE0); }
          }
          gl.uniform1f(locs.sfRaw, 0); if (blendWas) gl.enable(gl.BLEND);
          gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h);
          gl.clearColor(1, 1, 1, 1);
          gl.activeTexture(gl.TEXTURE16); gl.bindTexture(gl.TEXTURE_2D, scene.tex); gl.activeTexture(gl.TEXTURE0);
          gl.uniform1i(locs.sceneTex, 16); gl.uniform1f(locs.sceneOn, 1);
          bindChop();
        });
        // v29.1 fix4: opaque floor first (the wall quads are the first 6P indices, the floor fan after)
        timed("render:block", () => {
          gl.enable(gl.POLYGON_OFFSET_FILL);
          gl.polygonOffset(1.5, 2.0);
          if (vao) gl.bindVertexArray(vao);
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, blockMesh.index);
          gl.drawElements(gl.TRIANGLES, blockMesh.count - blockMesh.wallCount, gl.UNSIGNED_INT, blockMesh.wallCount * 4);
          gl.disable(gl.POLYGON_OFFSET_FILL);
        });
        if (clusters.program && clusters.count > 0) {
          timed("render:clusters", () => {
            gl.useProgram(clusters.program);
            gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, seabedTex);
            gl.uniformMatrix4fv(clusters.viewProj, false, vpMat); gl.uniform1f(clusters.depth, state.depth); setBedOnly(gl, clusters.program, state.size); bindClusterLight(); gl.uniform1i(clusters.lutLoc, 15);
            for (const part of Object.values(clusters.parts)) if (part.count) { gl.bindVertexArray(part.vao); gl.drawElementsInstanced(gl.TRIANGLES, part.indexCount, gl.UNSIGNED_SHORT, 0, part.count); }
            gl.bindVertexArray(vao); gl.useProgram(program);
            if (coastTex) { gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, coastTex); gl.activeTexture(gl.TEXTURE0); }
          });
        }
        // v29.1 fix4: walls after the floor and coral. Inner faces (front-facing, seen through the top)
        // opaque; outer faces (back-facing by the perimeter winding) blended over the real floor + coral.
        timed("render:walls", () => {
          gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
          gl.drawElements(gl.TRIANGLES, blockMesh.wallCount, gl.UNSIGNED_INT, 0);
          gl.cullFace(gl.FRONT); gl.enable(gl.BLEND);
          gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
          gl.drawElements(gl.TRIANGLES, blockMesh.wallCount, gl.UNSIGNED_INT, 0);
          gl.disable(gl.BLEND); gl.cullFace(gl.BACK); gl.disable(gl.CULL_FACE);
        });
        gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.xz);
        gl.vertexAttribPointer(locs.xz, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.meta);
        gl.vertexAttribPointer(locs.meta, 4, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, fftMesh.index);
        if (floorDebug) return;
        // Clusters and the scene target touch texture units. The surface
        // samples the chop patches, island on or off, so bind them again
        // immediately before that draw.
        bindChop();
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      }
      timed("render:fftMesh", () => { for (const [first, count] of ranges) gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, first * 4); });
      if (spray && state.fx) /* v30.19: Break Flags only adds outlines */ timed("render:spray", () => { spray.draw(state, g, fft, vpMat, eye, (Math.PI / 4) / canvas.height); gl.bindVertexArray(vao); gl.useProgram(program); });
      if (lip && state.fx) timed("render:lip", () => { lip.draw(state, g, fft, vpMat, eye); gl.bindVertexArray(vao); gl.useProgram(program); });
      if (curlSheet && state.fx) timed("render:curl", () => { curlSheet.draw(state, g, fft, vpMat, eye, sunU, blockOn); gl.bindVertexArray(vao); gl.useProgram(program); });
      if (state.boat && webgl2) { if (!boatView) boatView = createBoatView(gl); boatView.draw(state.boat, vpMat, sunU, state.fx || null); gl.bindVertexArray(vao); gl.useProgram(program); }
      if (globalThis.__oceanBuoy && webgl2) { if (!buoyView) buoyView = createBuoyView(gl); buoyView.draw(globalThis.__oceanBuoy, vpMat, sunU); gl.bindVertexArray(vao); gl.useProgram(program); } // v30.21
      if (edgeCheck && blockOn) { gl.disable(gl.BLEND); edgeCheckRun(fft.L); }
      gl.disable(gl.BLEND);
      // The shallow-water step renders into this atlas. Leave it unbound so
      // that pass is not a feedback loop on unit 15.
      if (coastTex) { gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, null); gl.activeTexture(gl.TEXTURE0); }
      sampleBackdrop();
      return;
    }
    if (fromTextures) {
      const g = state.gpu;
      [["physA", g.A], ["physB", g.B], ["physL", g.L], ["physF", g.F], ["physG", g.G]].forEach(([name, tex], i) => {
        gl.activeTexture(gl.TEXTURE2 + i);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(locs[name], 2 + i);
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(locs.physN, g.n);
      if (locs.dayTint) { const d = globalThis.__oceanDay; gl.uniform3f(locs.dayTint, d ? d[0] : 1, d ? d[1] : 1, d ? d[2] : 1); }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.index);
      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
      return;
    }
    stream(locs.height, buffers.height, state.heights, 1);
    stream(locs.normal, buffers.normal, state.normals, 3);
    stream(locs.shift, buffers.shift, state.shifts, 2);
    stream(locs.broken, buffers.broken, state.breaks, 1);
    stream(locs.lip, buffers.lip, state.lips, 1);
    // The shader takes the grade foam was laid with and its fade separately.
    stream(locs.foam, buffers.foam, state.foamGrade, 1);
    stream(locs.flag, buffers.flag, state.flags, 1);
    stream(locs.recent, buffers.recent, state.recent, 1);
    stream(locs.fade, buffers.fade, state.foamFade, 1);
    stream(locs.age, buffers.age, state.foamAge, 1);
    stream(locs.ahead, buffers.ahead, state.ahead, 1);
    stream(locs.run, buffers.run, state.runs, 1);
    stream(locs.thin, buffers.thin, state.thins, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.index);
    gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
  }

  // The sun: the standard one (24.8 degrees up), or with ?sun=low a review sun at the same
  // azimuth only 10 degrees up, for judging ripple relief under low raking light.
  const lowSun = typeof location !== "undefined" && /[?&]sun=low\b/.test(location.search);
  const sun = lowSun ? [0.947, 0.174, 0.267] : [0.78, 0.38, 0.22];
  // ?chop=0 in the address turns the fine chop off, for side-by-side comparison only.
  const chopOff = typeof location !== "undefined" && /[?&]chop=0\b/.test(location.search);
  // ?fine=0 turns only the finer patch off, likewise for comparison only.
  const fineOff = typeof location !== "undefined" && /[?&]fine=0\b/.test(location.search);

  function cameraHeight() {
    return Math.sin(camera.pitch) * camera.distance * lastSizeK; // v30.20 metres
  }

  return { draw, cameraHeight, gl, webgl2 };
}

// v30b spray: GPU points thrown from breaking crests. Stateless: point i lives a 0.7-1.5 s cycle;
// each cycle it picks a random physics cell, and only cells breaking now (break flag, or flagged
// foam) launch it: up 2-6 m/s and downwind, under gravity, fading as it falls. The share of
// points in use climbs with the wind (none below 8 m/s, all at 25). ?spray=0 turns it off.
const SPRAY_SHARE = Number(SEA_PARAMS.get("sprayshare") ?? 1);
const SPRAY_LAND = SEA_PARAMS.get("sland") !== "0";
const SPRAY_N = Number(SEA_PARAMS.get("sprayn") ?? 3072); // v30.19 mist: was 24576 chunky points
// v30.17 visible crash: a plunging lip sheet over the heightfield. Each riding (active) strand segment
// is the crest line of a breaker; from it a thin sheet is thrown forward along the wind and curls down
// onto the face (a parabola: ~0.07 U m forward, falling ~0.05 U m), translucent turquoise at the root
// (light through the thin sheet), aerated white toward the tip. It grows over the first 0.6 s of the
// break and falls/fades as the chain stops riding. The fresh whitecap trail (strand G) and its bubble
// plume, already laid along the crest, are where it lands and trail behind.
function createLip(gl) {
  // v30.17b: driven by the breaking detector on the physics grid (physL: x breaks, y lips), every
  // 2nd cell: where the crest is breaking, a lip segment ~2 cells long along the crest is thrown
  // forward along the wind and curls down. Strength follows the detector, so it grows and fades
  // with the break itself (no dependence on riding strand chains; starburst fix untouched).
  const CURL = SEA_PARAMS.get("curl") === "lip"; // v30.22b: the lip-sheet curl is replaced by the mesh curl (fftView); ?curl=lip for the old one. v30.22: pre-made curl on the biggest breakers (?curl=0 off)
  const SUB = CURL ? 8 : 4, STRIDE = 2;
  const CURL_K = Number(SEA_PARAMS.get("curlk") ?? 0.2); // crest height / Hs(PM) where curls start (tuned on the box)
  const vs = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D physA; uniform highp sampler2D physL;
uniform int physN; uniform float fftL, uU, uT; uniform vec2 uDir; uniform mat4 viewProj; uniform vec3 eye;
out float vS; out float vA; out vec3 vW; out vec3 vN; out float vCurl;
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec3 at(ivec2 c, float sp) { c = (c % physN + physN) % physN; vec4 a = texelFetch(physA, c, 0);
  vec2 xz = (vec2(c) + 0.5) * sp - 0.5 * fftL + a.yz; return vec3(xz.x, a.x, xz.y); }
void main() {
  int v = gl_VertexID % 6, q = gl_VertexID / 6; int sub = q % ${SUB}, cell = q / ${SUB};
  int nS = physN / ${STRIDE};
  ivec2 c = ivec2(cell % nS, cell / nS) * ${STRIDE};
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vCurl = 0.0; vA = 0.0; vS = 0.0; vW = vec3(0.0); vN = vec3(0.0, 1.0, 0.0);
  // v30.19 lip SHEET: one connected ribbon along the breaking stretch of the crest. Only crest-line cells
  // (height max along the wind within 2 cells) carry it; each segment joins this crest cell to the next
  // one along the crest, and every vertex takes its shape from its OWN cell's break strength (no
  // per-segment random), so neighbouring segments share edges and the sheet tapers to nothing where the
  // break ends. Deep water mostly spills: a short throw (~0.02-0.045 U m) that curls down onto the face.
  float sp = fftL / float(physN);
  vec2 cr = vec2(-uDir.y, uDir.x);
  ivec2 o = ivec2(round(cr * float(${STRIDE})));
  ivec2 wv = ivec2(round(uDir)); if (wv == ivec2(0)) wv = ivec2(1, 0);
  #define HAT(q) texelFetch(physA, ((q) % physN + physN) % physN, 0).x
  float hc = HAT(c);
  if (hc <= 0.0 || hc < HAT(c + wv) || hc < HAT(c - wv) || hc < HAT(c + 2 * wv) || hc < HAT(c - 2 * wv)) return; // crest line only
  int side = (v == 1 || v == 2 || v == 4) ? 1 : 0; int up = (v == 2 || v == 4 || v == 5) ? 1 : 0;
  float s = float(sub + up) / float(${SUB});
  ivec2 ce = side == 0 ? c : c + o;
  vec4 l = texelFetch(physL, (ce % physN + physN) % physN, 0);
  float brk = clamp(max(4.0 * l.x, l.y), 0.0, 1.0);
  float g = smoothstep(0.35, 0.8, brk);
  vec4 l0 = texelFetch(physL, c, 0); if (max(4.0 * l0.x, l0.y) < 0.35) return;
  vec3 P0 = at(ce, sp);
  vec3 fw = vec3(uDir.x, 0.0, uDir.y);
  P0 += fw * min(0.35 * max(P0.y, 0.0), 1.2 * sp) * smoothstep(0.1, 0.6, brk);   // starts on the leaned crest (same lean as the mesh)
  float Hc = max(P0.y, 0.0); float Lf = min((0.25 + 0.35 * g) * g * Hc, 0.045 * uU), Hf = min(0.3 * g * Hc, 0.03 * uU); // v30.19: sheet size follows the breaker (crest height), small waves -> tiny spill
  vec3 P = P0 + vec3(0.0, 0.05, 0.0) + fw * Lf * s + vec3(0.0, Hf * (0.6 * s - 1.6 * s * s), 0.0);
  vec3 T = fw * Lf + vec3(0.0, Hf * (0.6 - 3.2 * s), 0.0);
#define CURL_K0 ${(CURL_K).toFixed(3)}
#define CURL_K1 ${(CURL_K * 1.3).toFixed(3)}
#if ${CURL ? 1 : 0}
  // v30.22 pre-made plunging curl (Michael): only the biggest breakers. Gate: the crest stands above
  // CURL_K x the wind's significant height (PM: Hs ~ 0.0246 U^2, so the bar rises with wind and small
  // waves never curl), the break is strong, U >= 11 m/s, and ~half of such crests (per-crest hash).
  // Each crest (cells grouped 8 along the crest x 6 along the wind, so a crest's segments share one
  // timing) runs its own 1.2-2.0 s cycle: rise -> pitch forward -> fold over the trough -> land and fade;
  // the landing area is where the existing lip foam spawns. Size 0.75-1.3x per crest.
  {
    float HsW = 0.0246 * uU * uU;
    vec2 cid = floor(vec2(dot(vec2(c), cr), dot(vec2(c), vec2(wv))) / vec2(8.0, 6.0));
    float r1 = h21(cid), r2 = h21(cid + 17.3), r3 = h21(cid + 41.7);
    float Hcen = max(at(c, sp).y, 0.0);
    float big = smoothstep(CURL_K0 * HsW, CURL_K1 * HsW, Hcen) * smoothstep(10.5, 12.5, uU) * step(r1, 0.55) * smoothstep(0.5, 0.85, max(4.0 * l0.x, l0.y));
    if (big > 0.0) {
      float per = mix(1.2, 2.0, r2); float ph = fract(uT / per + r3);           // 0..1 through one curl
      float sz = mix(0.75, 1.3, r1 / 0.55) * big;
      float Rr = 0.8 * Hc * sz;                                                  // curl radius ~ crest height
      float th = mix(0.25, 3.6, smoothstep(0.0, 0.8, ph));                       // rise (small) -> pitch -> fold past vertical-down
      float a = s * th;
      vec3 Pc = P0 + vec3(0.0, 0.05, 0.0) + fw * (Rr * (1.0 - cos(a)) + 0.35 * Rr * s) + vec3(0.0, Rr * sin(a), 0.0);
      Pc.y -= smoothstep(0.55, 1.0, ph) * s * s * 1.1 * Hc * sz;                 // the lip falls onto the trough
      vec3 Tc = fw * (Rr * th * sin(a) + 0.35 * Rr) + vec3(0.0, Rr * th * cos(a) - 2.2 * smoothstep(0.55, 1.0, ph) * s * Hc * sz, 0.0);
      float land = 1.0 - smoothstep(0.85, 1.0, ph);                              // collapse: fades as it lands (foam takes over)
      P = mix(P, Pc, big); T = normalize(mix(T, Tc, big)); g = mix(g, g * land, big) + 0.0 * land;
      g = max(g, big * land * 0.9); vCurl = big;
    }
  }
#endif
  vec3 B = vec3(cr.x, 0.0, cr.y);
  vN = normalize(cross(B, T)); if (vN.y < 0.0) vN = -vN;
  vW = P; vS = s; vA = g;
  gl_Position = viewProj * vec4(P, 1.0);
}`;
  const fs = `#version 300 es
precision highp float;
uniform vec3 eye;
in float vS; in float vA; in vec3 vW; in vec3 vN; in float vCurl; out vec4 o;
float hn(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  if (vA <= 0.0) discard;
  float br = hn(floor(vW.xz * 3.0));
  float tipA = 1.0 - smoothstep(0.7, 1.0, vS) * (0.6 + 0.4 * br);  // v30.19: the tip thins out softly (no blobs)
  vec3 L = normalize(vec3(0.4, 0.8, 0.3)); vec3 V = normalize(eye - vW);
  float sun = 0.55 + 0.45 * clamp(dot(vN, L), 0.0, 1.0);
  vec3 turq = vec3(0.30, 0.78, 0.74), white = vec3(0.93, 0.97, 0.98);
  vec3 col = mix(turq, white, smoothstep(0.35, 0.9, vS) * smoothstep(0.2, 0.7, vA)) * sun;   // turquoise sheet, white where thin/broken
  col += vec3(0.05, 0.18, 0.15) * pow(max(dot(V, -L), 0.0), 3.0) * (1.0 - vS);   // light through the thin root
  o = vec4(col, vA * tipA * mix(0.35, 0.75, smoothstep(0.1, 0.7, vS)));
  if (${SEA_PARAMS.get("curldbg") === "1" ? 1 : 0} == 1 && vCurl > 0.0) o = vec4(1.0, 0.0, 0.0, 1.0);
  if (${SEA_PARAMS.get("curldbg") === "2" ? 1 : 0} == 1) o = vCurl > 0.0 ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
}`;
  const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) console.error("lip compile fail", gl.getShaderInfoLog(x)); return x; };
  const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error("lip link fail", gl.getProgramInfoLog(prog)); return null; }
  const U = {}; for (const n of ["physA", "physL", "physN", "fftL", "uU", "uT", "uDir", "viewProj", "eye"]) U[n] = gl.getUniformLocation(prog, n);
  const vao = gl.createVertexArray();
  return {
    draw(state, g, fft, vp, eye) {
      if (state.fx.speed < 9) return;
      gl.useProgram(prog); gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0 + 2); gl.bindTexture(gl.TEXTURE_2D, g.A); gl.uniform1i(U.physA, 2);
      gl.activeTexture(gl.TEXTURE0 + 4); gl.bindTexture(gl.TEXTURE_2D, g.L); gl.uniform1i(U.physL, 4);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(U.physN, fft.N); gl.uniform1f(U.fftL, fft.L); gl.uniform1f(U.uU, state.fx.speed); if (U.uT) gl.uniform1f(U.uT, state.fx.t || 0);
      gl.uniform2f(U.uDir, Math.cos(state.fx.dir), Math.sin(state.fx.dir)); gl.uniformMatrix4fv(U.viewProj, false, vp); gl.uniform3f(U.eye, eye[0], eye[1], eye[2]);
      gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE); gl.disable(gl.CULL_FACE); if (SEA_PARAMS.get("lipdbg") === "1") gl.disable(gl.DEPTH_TEST);
      const n = (fft.N / STRIDE) * (fft.N / STRIDE);
      gl.drawArrays(gl.TRIANGLES, 0, n * SUB * 6);
      gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST);
    },
  };
}
// v30.22c plunging curl (Michael: "realistic curls"): a finely tessellated sheet along each curling
// crest, driven by the physics curl field (gpuSurface CURL_*). One instance per (column across travel,
// curling crest in that column); each joins its crest point to the next column's crest, so one breaker
// is one continuous lip. The profile (curlAt) throws forward from the crest top and comes down over
// the trough as the break ages, then slumps into whitewater. Outer side: water that thins to lighter
// turquoise toward the tip; inner side (under the lip): a smooth dark-turquoise gradient; the leading
// edge is whitewater. ?curl=0 off, ?curldbg=4 paints the sheets red.
function createCurl(gl) {
  const SUB = 16, SLOTS = CURL_SLOTS;
  const vs = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D physA; uniform highp sampler2D physL; uniform highp sampler2D curlF; uniform highp sampler2D curlList;
uniform int physN; uniform float fftL; uniform int uAxisX; uniform ivec2 uAx; uniform vec2 uDir; uniform mat4 viewProj; uniform float uBlock; uniform float uCurlX;
out float vS; out float vAmp; out float vPh; out vec3 vW; out vec3 vN; out float vU; out float vH; out float vClear; out float vOpen; flat out vec3 vDbg;
${CURL_SHAPE_GLSL}
ivec2 wrapc(ivec2 c) { return (c % physN + physN) % physN; }
ivec2 cellOf(int u, int v) { return uAxisX == 1 ? ivec2(v, u) : ivec2(u, v); }
vec2 wallShift(vec2 p0, vec2 shift) { // the block's wall taper of the horizontal shift, exactly as the sea mesh (fftView v29.1 fix5)
  if (uBlock < 0.5) return shift;
  float half_ = 0.5 * fftL, band = 0.6 * half_, lim = 0.03 * half_;
  vec2 wN = smoothstep(vec2(0.0), vec2(band), half_ - abs(p0));
  float kx = min(1.0, lim / max(abs(shift.x), 1e-4)), kz = min(1.0, lim / max(abs(shift.y), 1e-4));
  shift.x *= wN.x * mix(kx, 1.0, wN.y); shift.y *= wN.y * mix(kz, 1.0, wN.x);
  return shift;
}
float surfH(vec2 xz, float sp) { // sea height under a world point (one fixed-point step for the chop shift, bilinear)
  vec2 g = (xz + 0.5 * fftL) / sp;
  vec2 sh = wallShift(xz, texelFetch(physA, wrapc(ivec2(floor(g + 0.5))), 0).yz);
  g = (xz - sh + 0.5 * fftL) / sp;
  ivec2 i = ivec2(floor(g)); vec2 f = g - floor(g);
  float h00 = texelFetch(physA, wrapc(i), 0).x, h10 = texelFetch(physA, wrapc(i + ivec2(1, 0)), 0).x;
  float h01 = texelFetch(physA, wrapc(i + ivec2(0, 1)), 0).x, h11 = texelFetch(physA, wrapc(i + ivec2(1, 1)), 0).x;
  return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}
vec3 crestPos(ivec2 c, float sub, float sp) {
  { float k = floor(sub + 0.5); c += int(k) * uAx; sub -= k; } // smoothed crest offset may reach past the neighbour cell
  vec4 a0 = texelFetch(physA, wrapc(c), 0), am = texelFetch(physA, wrapc(c - uAx), 0), ap = texelFetch(physA, wrapc(c + uAx), 0);
  float hpk = a0.x - 0.25 * (am.x - ap.x) * sub;
  vec2 shift = sub >= 0.0 ? mix(a0.yz, ap.yz, sub) : mix(a0.yz, am.yz, -sub);
  vec2 p0 = (vec2(c) + sub * vec2(uAx)) * sp - 0.5 * fftL;
  shift = wallShift(p0, shift);
  vec2 xz = p0 + shift;
  vec4 l = texelFetch(physL, wrapc(c), 0);
  float lean = smoothstep(0.1, 0.6, clamp(max(4.0 * l.x, l.y), 0.0, 1.0));   // same lean as the sea mesh
  xz += uDir * lean * min(0.35 * max(hpk, 0.0), 1.2 * sp);
  return vec3(xz.x, hpk + 0.08 * lean * max(hpk, 0.0), xz.y);
}
void main() {
  int j = gl_VertexID / 2, side = gl_VertexID % 2;
  int k = gl_InstanceID % ${SLOTS}, u = gl_InstanceID / ${SLOTS};
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vS = 0.0; vAmp = 0.0; vPh = 0.0; vW = vec3(0.0); vN = vec3(0.0, 1.0, 0.0); vU = 0.0; vH = 1.0; vClear = 1.0; vOpen = 0.0; vDbg = vec3(0.0);
  float vf = texelFetch(curlList, ivec2(k, u), 0).x;
  if (vf < 0.0) return;
  if (uBlock > 0.5 && u >= physN - 1) return;
  float sp = fftL / float(physN);
  int v = int(vf);
  ivec2 c0 = cellOf(u, v);
  vec4 f0 = texelFetch(curlF, c0, 0);
  ivec2 c1 = c0; vec4 f1 = f0; bool found = false;
  int du = 1;
  for (int e = 1; e <= 2 && !found; e += 1) { // the next column's crest; across a one-column gap, the one after
    if (uBlock > 0.5 && u + e > physN - 1) break;
    for (int d = 0; d < 11; d += 1) { // +-5 cells: an oblique crest line steps more than 3 cells per column
      int dd = (d + 1) / 2 * (d % 2 == 1 ? 1 : -1);
      ivec2 t = wrapc(cellOf(u + e, v + dd)); vec4 ft = texelFetch(curlF, t, 0);
      if (ft.x >= 0.0) { c1 = t; f1 = ft; found = true; du = e; break; }
    }
  }
  vec3 C; vec4 f;
  if (side == 0) { C = crestPos(c0, f0.w, sp); f = f0; }
  else if (found) { C = crestPos(c1, f1.w, sp); f = f1; }
  else { C = crestPos(c0, f0.w, sp); ivec2 bx = cellOf(1, 0); C.xz += vec2(bx) * sp; f = vec4(f0.x * 0.5, f0.yzw); } // it24: no neighbour found -> continue half a column instead of collapsing to a wedge (the saw-tooth)
  float amp = f.x, ph = f.y, Hc = f.z;
  if (uBlock > 0.5) amp *= smoothstep(4.0, 14.0, 0.5 * fftL - max(abs(C.x), abs(C.z))); // no curls hugging the walls
  float s = float(j) / float(${SUB});
  float g = smoothstep(0.0, 0.3, amp);
  vec3 fw = vec3(uDir.x, 0.0, uDir.y);
  // v30.22d it28: the lip LANDS on the water where its tip comes down: the falling (below-crest) part of the
  // profile is scaled so the tip ends just above the water there (lips bigger than their wave no longer get
  // flattened onto it and vanish). Water height at the tip, 2 taps +-1.5 m along the crest (10 fetches/vertex).
  float tg = curlTight(uCurlX, ph); // boosted lips curl tighter; same as the cavity field's reach (CURL_FIELD_FS)
  vec2 qt = curlAtT(1.0, ph, Hc, amp, tg) * g;
  vec2 acr = vec2(-uDir.y, uDir.x);
  vec2 tipXZ = C.xz + fw.xz * qt.x;
  float ht = 0.5 * (surfH(tipXZ - 1.5 * acr, sp) + surfH(tipXZ + 1.5 * acr, sp));
  vec2 q = curlAtT(s, ph, Hc, amp, tg) * g;
  vec2 dq = curlDsT(s, ph, Hc, amp, tg);
  float kd = qt.y < -1e-3 ? clamp((ht + 0.1 * Hc - C.y) / qt.y, 0.0, 1.6) : 1.0;
  kd = max(kd, 0.5 * (1.0 - smoothstep(1.2, 2.6, uCurlX))); // it34: true size may land on the water too (kd >= 0.5: no blobs); kd >= 1 buried the small 25 m/s lips (invisible close)
  kd = mix(1.0, kd, smoothstep(0.15, 0.5, ph));
  if (q.y < 0.0) { q.y *= kd; dq.y *= kd; }
  q += vec2(-0.06, -0.05) * Hc * (1.0 - s) * (1.0 - s); // the root sits just inside the crest (no gap)
  vec3 P = C + fw * q.x + vec3(0.0, q.y, 0.0);
  if (s > 0.15) { // it35: the sheet stays above the face it spans (a small true-size lip on a convex face was hidden under it)
    float hs = surfH(P.xz, sp) + 0.12 * Hc;
    float lift = max(hs - P.y, 0.0) * smoothstep(0.15, 0.35, s);
    P.y += lift; if (lift > 0.0) dq.y = mix(dq.y, 0.0, smoothstep(0.0, 0.1 * Hc, lift));
  }
  vH = mix(1.0, (P.y - ht) / max(Hc, 0.3), smoothstep(0.35, 0.7, s)); // height above the landing water in lip heights
  // Metres above the sea under this vertex. The drawn mesh (and the foam on it) does not
  // match that sample, so a depth test against it used to slice the opaque sheet into a
  // straight-edged facet a few cells wide.
  vClear = P.y - surfH(P.xz, sp);
  float prevA = -1.0;
  if (u > 0) {
    prevA = texelFetch(curlF, wrapc(cellOf(u - 1, v)), 0).x;
    prevA = max(prevA, texelFetch(curlF, wrapc(cellOf(u - 1, v - 1)), 0).x);
    prevA = max(prevA, texelFetch(curlF, wrapc(cellOf(u - 1, v + 1)), 0).x);
  }
  vOpen = side == 1 ? (found ? 0.0 : 1.0) : (prevA > 0.02 ? 0.0 : 1.0);
  vec3 N = fw * (-dq.y) + vec3(0.0, dq.x, 0.0);
  vN = length(N) > 1e-6 ? normalize(N) : vec3(0.0, 1.0, 0.0);
  vDbg = vec3(float(u), found ? float(du) : 0.0, float(k));
  vS = s; vAmp = amp; vPh = ph; vW = P; vU = dot(P.xz, vec2(-uDir.y, uDir.x));
  gl_Position = viewProj * vec4(P, 1.0);
}`;
  const fs = `#version 300 es
precision highp float;
uniform vec3 eye; uniform vec3 uSun; uniform vec3 dayTint; uniform float uU;
in float vS; in float vAmp; in float vPh; in vec3 vW; in vec3 vN; in float vU; in float vH; in float vClear; in float vOpen; flat in vec3 vDbg; out vec4 o;
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 x) { vec2 i = floor(x), f = x - i; f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
void main() {
  if (vAmp <= 0.002) discard;
  vec3 N = normalize(vN), V = normalize(eye - vW), L = normalize(uSun);
  bool outer = dot(N, V) > 0.0; if (!outer) N = -N;
  float thin = smoothstep(0.2, 1.0, vS);
  float dl = max(dot(N, L), 0.0), back = pow(max(dot(V, -L), 0.0), 2.0);
  float F = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
  // outer side: the face of the lip (seen from in front it is the falling curtain): water that turns a
  // translucent, lighter turquoise where the lip is thin (light through it), faint streaks along the flow
  vec3 sea = vec3(${PQ("curlr", 0.10)}, ${PQ("curlg", 0.35)}, ${PQ("curlb", 0.40)});
  vec3 glow = vec3(0.16, 0.50, 0.50), dark = vec3(0.03, 0.15, 0.17), sky = vec3(0.72, 0.81, 0.85);
  float streak = 0.975 + 0.05 * vn(vec2(vU * 0.5 + 2.0 * vS, vS * 1.5 - 0.4 * vPh)); // subtle, no ribs
  vec3 col;
  if (outer) {
    // over the crest the lip is thin and lit through (lighter, greener); down the falling curtain it is
    // the colour of the wave face, a little darker toward the bottom (its own shadow)
    float lit = smoothstep(0.0, 0.25, vS) * (1.0 - smoothstep(0.35, 0.8, vS));
    col = mix(sea, glow, 0.65 * lit) * (0.88 + 0.22 * dl) * mix(1.0, 0.8, smoothstep(0.5, 1.0, vS)) * streak;
    col += vec3(0.03, 0.12, 0.10) * lit * back;
    col = mix(col, sky, 0.55 * F);
  } else {
    col = mix(dark, 0.8 * glow, 0.6 * thin * thin) * streak;    // under the lip: dark root, lighter thin tip
  }
  // v30.22d storm lips (17-24 m/s): a darker curl (curtain and underside) so the lip reads against the foam
  float storm = smoothstep(17.0, 24.0, uU);
  col *= mix(1.0, outer ? 0.62 : 0.5, storm * (1.0 - 0.5 * smoothstep(0.75, 1.0, vS)));
  // whitewater: a feathered white lip edge from the start, spreading back up the curtain as it falls and
  // lands, plus streaks of white on the curtain (colour only: the sheet edge itself stays smooth)
  float nz = 0.6 * vn(vec2(vU * 0.9 + 1.7 * vS, vS * 4.0 + 0.7 * vPh)) + 0.4 * vn(vec2(vU * 2.6 - 3.1 * vS, vS * 9.0 + 3.0)); // blotchy, not banded across the lip
  float wwW = min(mix(0.14, 0.6, smoothstep(0.35, 1.2, vPh)) - 0.06 * storm, 0.95); // v30.22d: storm lips no longer mostly white (white on white foam did not read)
  float ww = smoothstep(1.0 - wwW, 1.0 - wwW + 0.25, vS + 0.35 * (nz - 0.5));
  float nearW = 1.0 - smoothstep(0.0, 0.35, vH + 0.15 * (nz - 0.5)); // where the lip comes down onto the water it is all whitewater
  ww = max(ww, nearW * smoothstep(0.4, 0.7, vS));
  ww = max(ww, (1.0 - smoothstep(0.15, 0.6, vAmp)) * smoothstep(0.3, 0.6, nz + 0.3 * vS)); // the lip's ends break up into whitewater (spilling), not a clean shell edge
  // the curling top of the lip (the part that reads from a distance) is a broken white rim once it throws
  float rim = smoothstep(0.22, 0.38, vS) * (1.0 - smoothstep(0.55, 0.78, vS + 0.2 * (nz - 0.5))) * smoothstep(0.15, 0.5, vPh);
  ww = max(ww, mix(0.6, 1.0, storm) * rim * smoothstep(mix(0.35, 0.3, storm), mix(0.7, 0.55, storm), nz + 0.2)); // broken, not a solid white shell; brighter on storm lips
  col = mix(col, vec3(0.90, 0.94, 0.95) * (0.82 + 0.18 * dl), 0.95 * ww);
  // v31.9: alpha is a hump across the profile, not a plateau of 1. A plateau ran into
  // the depth test and the end of a short crest and left a flat parallelogram with
  // straight edges (turquoise where the sheet is still water, white once it breaks).
  // ?curlfade=0 is the old plateau, for comparison only.
  float a = smoothstep(0.0, 0.12, vAmp);
  if (${SEA_PARAMS.get("curlfade") === "0" ? "false" : "true"}) {
    float along = clamp(vS + 0.2 * (nz - 0.5), 0.0, 1.0);
    float hump = sin(along * 3.14159265);
    a *= pow(max(hump, 0.0), 0.4);
    a *= smoothstep(0.0, 1.15, vClear + 0.4 * (nz - 0.5));
    a *= 1.0 - smoothstep(0.0, 0.85, vOpen);
    a = min(a, 0.62);
  } else {
    a *= (1.0 - smoothstep(0.7, 1.0, vS + 0.3 * (nz - 0.5)));
    a *= smoothstep(0.0, 0.18, vS);
    a *= mix(1.0, smoothstep(0.12, 0.5, vH + 0.12 * (nz - 0.5)), smoothstep(0.3, 0.6, vS));
  }
  a *= 1.0 - smoothstep(0.95, 1.35, vPh + 0.35 * (nz - 0.5) + 0.25 * vS);
  a *= mix(1.0, 0.85, thin * (1.0 - ww));
  o = vec4(col * dayTint, a);
  if (${SEA_PARAMS.get("curldbg") === "4" || SEA_PARAMS.get("curldbg") === "5" ? 1 : 0} == 1) o = vec4(1.0, 0.0, 0.0, 1.0);
  if (${SEA_PARAMS.get("curldbg") === "7" || SEA_PARAMS.get("curldbg") === "8" ? 1 : 0} == 1) { // per-column colour; white = no neighbour found, magenta = bridged a gap
    float hc = fract(sin(vDbg.x * 12.9898 + vDbg.z * 4.1) * 43758.5453);
    o = vec4(0.5 + 0.5 * cos(6.2832 * (hc + vec3(0.0, 0.33, 0.67))), 1.0);
    if (vDbg.y < 0.5) o = vec4(1.0); else if (vDbg.y > 1.5) o = vec4(1.0, 0.0, 1.0, 1.0);
  }
  if (${SEA_PARAMS.get("curldbg") === "6" ? 1 : 0} == 1) o = vec4(clamp(3.0 * vH, 0.0, 1.0), vS, outer ? 1.0 : 0.0, 1.0); // red: height above the water, green: s, blue: outer side
}`;
  const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) console.error("curl compile fail", gl.getShaderInfoLog(x)); return x; };
  const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error("curl link fail", gl.getProgramInfoLog(prog)); return null; }
  const U = {}; for (const n of ["physA", "physL", "curlF", "curlList", "physN", "fftL", "uAxisX", "uAx", "uDir", "viewProj", "uBlock", "eye", "uSun", "dayTint", "uU", "uCurlX"]) U[n] = gl.getUniformLocation(prog, n);
  const vao = gl.createVertexArray();
  return {
    draw(state, g, fft, vp, eye, sun, block) {
      if (state.fx.speed < 11.5 || !g.CU || !g.CL) return;
      gl.useProgram(prog); gl.bindVertexArray(vao);
      [["physA", g.A, 2], ["physL", g.L, 4], ["curlF", g.CU, 7], ["curlList", g.CL, 8]].forEach(([n, t, u]) => { gl.activeTexture(gl.TEXTURE0 + u); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(U[n], u); });
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(U.physN, fft.N); gl.uniform1f(U.fftL, fft.L); gl.uniform1i(U.uAxisX, g.curlAxis.x); gl.uniform2i(U.uAx, g.curlAxis.ax[0], g.curlAxis.ax[1]);
      gl.uniform2f(U.uDir, Math.cos(state.fx.dir), Math.sin(state.fx.dir)); gl.uniformMatrix4fv(U.viewProj, false, vp); gl.uniform3f(U.eye, eye[0], eye[1], eye[2]);
      gl.uniform1f(U.uBlock, block ? 1 : 0); gl.uniform1f(U.uCurlX, curlX(SEA_PARAMS, state.fx.speed)); gl.uniform3f(U.uSun, sun[0], sun[1], sun[2]); gl.uniform1f(U.uU, state.fx.speed);
      { const d = globalThis.__oceanDay; gl.uniform3f(U.dayTint, d ? d[0] : 1, d ? d[1] : 1, d ? d[2] : 1); }
      gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE); gl.disable(gl.CULL_FACE);
      if (SEA_PARAMS.get("curldbg") === "5" || SEA_PARAMS.get("curldbg") === "8") gl.disable(gl.DEPTH_TEST);
      // Don't let the sheet's own triangles clip each other into facets.
      gl.depthMask(false);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 2 * (SUB + 1), SLOTS * fft.N);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
    },
  };
}
function createSpray(gl) {
  const vs = `#version 300 es
uniform highp sampler2D physA; uniform highp sampler2D physL; uniform highp sampler2D physG; uniform highp sampler2DArray physS; uniform float uStr; uniform float uU;
uniform int physN; uniform float fftL; uniform float uT; uniform float uFrac; uniform float pixAngle;
uniform vec2 uDir; uniform vec3 eye; uniform mat4 viewProj; uniform float uLand;
out float vA; out vec3 vS;
float hsh(float n) { return fract(sin(n * 12.9898 + 78.233) * 43758.5453); }
void main() {
  float id = float(gl_VertexID);
  float h1 = hsh(id * 0.731 + 1.0), h2 = hsh(id * 1.913 + 2.0), h3 = hsh(id * 0.377 + 3.0);
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vA = 0.0;
  if (h3 > uFrac) return;
  float wk = clamp((uU - 8.0) / 17.0, 0.0, 1.0);   // v30.13: 0 at 8 m/s .. 1 at 25
  float life = (0.3 + 0.25 * h2) * (1.0 + 1.6 * wk * wk); // short droplets at low wind, streaming spume (to ~1.4 s) at 25
  float ph = uT / life + h1; float cyc = floor(ph); float tau = fract(ph) * life;
  vec2 r = vec2(hsh(cyc * 13.17 + id * 0.71), hsh(cyc * 7.31 + id * 1.37 + 5.0));
  // look for a breaking cell in a 4x4 block (2-cell stride) around the random pick
  ivec2 c0 = ivec2(r * float(physN)); ivec2 cell = c0; float emit = 0.0;
  for (int k = 0; k < 16; k += 1) {
    ivec2 c = (c0 + (uStr > 0.5 ? 5 : 2) * ivec2(k % 4, k / 4)) % physN; // v30.13: wider search (whitecap bands are sparse lines)
    // lips of actively breaking crests only: a break cell, or a flagged-now cell on a lip
    vec4 lc = texelFetch(physL, c, 0);
    float e = smoothstep(0.5, 0.9, clamp(max(4.0 * lc.x, lc.y), 0.0, 1.0));   // v30.19: actually-breaking lips only
    if (e > emit) { emit = e; cell = c; }
  }
  if (emit < 0.12) return;
  vec4 a = texelFetch(physA, cell, 0);
  float sp = fftL / float(physN);
  float h4 = hsh(cyc * 3.7 + id * 2.1), h5 = hsh(cyc * 5.3 + id * 0.9), h6 = hsh(cyc * 9.1 + id * 0.3);
  vec2 p0 = (vec2(cell) + vec2(h4, h5) - 0.5) * sp - 0.5 * fftL + a.yz;
  vec2 side = vec2(-uDir.y, uDir.x);
  vec2 hv = uDir * (1.0 + uU * (0.1 + 0.15 * h4)) + side * (h5 - 0.5) * 1.0; // v30.19 mist: drifts downwind (~0.1-0.25 U)
  float vy = 0.4 + 0.8 * h6;
  // v30.16 ballistic flight ends where it lands (back at launch height), not at a timed fade
  float tLand = (vy + sqrt(vy * vy + 2.0 * 4.9 * 0.25)) / 9.8;
  float tEnd = min(life, tLand);
  if (tau > tEnd) return;
  vec3 pos = vec3(p0.x + hv.x * tau, a.x + 0.25 + vy * tau - 4.9 * tau * tau, p0.y + hv.y * tau);
  // v30.16 real size in metres: droplet clumps 3-12 cm at 10 m/s, spume blobs to ~0.5 m at 25 m/s
  float h7 = hsh(cyc * 2.3 + id * 3.1);
  float sizeM = mix(0.3, 0.7, h7);   // v30.19: a soft haze puff (droplets far below a pixel), never a chunk
  if (uLand > 0.5) {
    // landing pass: in the last 0.12 s of flight, splat a small fresh-foam mark into the strand buffer
    // (G = whitecap trail): it shows as a little white patch with its bubble plume and decays like a trail
    if (tau < tLand - 0.12 || tLand > life) return;
    vec3 pl = vec3(p0.x + hv.x * tLand, 0.0, p0.y + hv.y * tLand);
    vec2 uv = fract((pl.xz + 0.5 * fftL) / fftL);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = clamp(2.5 * sizeM / (fftL / 1024.0), 1.0, 6.0);
    vA = 0.04 * emit; vS = vec3(h7, 1.0, 0.0);
    return;
  }
  float d = length(pos - eye);
  float px = sizeM / (d * pixAngle);
  vec3 vel = vec3(hv.x, vy - 9.8 * tau, hv.y);
  vec4 q0 = viewProj * vec4(pos, 1.0), q1 = viewProj * vec4(pos + 0.05 * vel, 1.0);
  vec2 sd = q1.xy / q1.w - q0.xy / q0.w;
  float elong = 1.0;
  gl_PointSize = clamp(px * (1.0 + tau), 1.0, 24.0);                 // the puff spreads as it drifts
  float cover = clamp(px * px / 4.0, 0.0, 1.0);
  vA = 0.07 * emit * cover * smoothstep(0.0, 0.1, tau) * (1.0 - smoothstep(0.5 * tEnd, tEnd, tau)); // faint
  vS = vec3(h7, elong, atan(sd.y, sd.x));
  gl_Position = q0;
}`;
  const fs = `#version 300 es
precision mediump float;
#define SPRAY_MASK ${SEA_PARAMS.get("spraymask") === "1" ? "true" : "false"}
in float vA; in vec3 vS; out vec4 o;
void main() { vec2 c = gl_PointCoord * 2.0 - 1.0;
  // v30.16 uneven shapes: rotated to the motion, squashed by the elongation, lumpy rim from the seed
  float ca = cos(vS.z), sa = sin(vS.z); c = vec2(ca * c.x - sa * c.y, sa * c.x + ca * c.y); c.y *= vS.y;
  float r2 = dot(c, c); if (r2 > 1.0) discard;   // v30.19: round soft puff
  if (SPRAY_MASK) { if (vA * (1.0 - r2 * r2) < 0.02) discard; o = vec4(0.0, 1.0, 0.0, 1.0); return; } // QM spray mask: pure green
  vec3 col = mix(vec3(0.66, 0.74, 0.82), vec3(0.94, 0.96, 0.96), 0.45 + 0.55 * clamp(0.5 - 0.5 * c.y, 0.0, 1.0)); // sunlit top, sky-lit underside (as the foam)
  o = vec4(col, vA * exp(-3.0 * r2) * (1.0 - r2)); } // peak 0.96 x 255 = 245`;
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.warn("spray", gl.getProgramInfoLog(prog)); return null; }
  const U = {}; for (const n of ["physA", "physL", "physG", "physS", "uStr", "uU", "physN", "fftL", "uT", "uFrac", "pixAngle", "uDir", "eye", "viewProj", "uLand"]) U[n] = gl.getUniformLocation(prog, n);
  const vao = gl.createVertexArray(); let landFb = null;
  return {
    draw(state, g, fft, vp, eye, pixAngle) {
      const frac = SPRAY_SHARE * Math.min(1, Math.max(0, (state.fx.speed - 8) / 17)) ** 1.5; // v30.13: a little from 8-10 m/s, all points at 25
      if (frac <= 0) return;
      gl.useProgram(prog); gl.bindVertexArray(vao);
      [["physA", g.A, 2], ["physL", g.L, 4], ["physG", g.G, 6]].forEach(([n, tex, u]) => { gl.activeTexture(gl.TEXTURE0 + u); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(U[n], u); });
      gl.activeTexture(gl.TEXTURE0 + 18); gl.bindTexture(gl.TEXTURE_2D_ARRAY, g.STR || null); gl.uniform1i(U.physS, 18); gl.uniform1f(U.uStr, g.STR ? 1 : 0); gl.uniform1f(U.uU, state.fx.speed); if (U.uT) gl.uniform1f(U.uT, state.fx.t || 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(U.physN, fft.N); gl.uniform1f(U.fftL, fft.L); gl.uniform1f(U.uT, state.fx.t); gl.uniform1f(U.uFrac, frac);
      gl.uniform1f(U.pixAngle, pixAngle); gl.uniform2f(U.uDir, Math.cos(state.fx.dir), Math.sin(state.fx.dir));
      gl.uniform3f(U.eye, eye[0], eye[1], eye[2]); gl.uniformMatrix4fv(U.viewProj, false, vp);
      gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE); gl.depthMask(false);
      gl.uniform1f(U.uLand, 0);
      gl.drawArrays(gl.POINTS, 0, SPRAY_N);
      gl.depthMask(true); gl.disable(gl.BLEND);
      // v30.16 landing: splat into the strand array layer 0 (level 0) while sampling only levels >= 2
      if (g.STR && SPRAY_LAND) {
        const prevFb = gl.getParameter(gl.FRAMEBUFFER_BINDING), vpw = gl.getParameter(gl.VIEWPORT);
        if (!landFb) { landFb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, landFb); gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, g.STR, 0, 0); }
        gl.bindFramebuffer(gl.FRAMEBUFFER, landFb); gl.viewport(0, 0, 1024, 1024);
        gl.activeTexture(gl.TEXTURE0 + 18); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_BASE_LEVEL, 2);
        gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.colorMask(false, true, false, false);
        gl.uniform1f(U.uLand, 1); gl.drawArrays(gl.POINTS, 0, SPRAY_N);
        gl.colorMask(true, true, true, true); gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_BASE_LEVEL, 0); gl.activeTexture(gl.TEXTURE0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, prevFb); gl.viewport(vpw[0], vpw[1], vpw[2], vpw[3]);
      }
    },
  };
}

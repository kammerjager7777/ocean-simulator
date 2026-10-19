import { VERT_FFT, FRAG_FFT_NORMAL, buildFftMesh, visibleRanges } from "./fftView.js?v=28";
import { LOBE_GLSL } from "../physics/roughness.js?v=28";
import { timed } from "../physics/passTimer.js?v=28";

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
const REFLECT_DEFS = "#define SKY_ZENITH 0.36, 0.47, 0.57\n#define SKY_SEEN_LO -0.04\n#define SKY_SEEN_HI 0.14\n#define REFLECT_GAIN 1.0\n";
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
const LIGHT_DEFS = "#define SUN_IRRADIANCE 0.25\n#define WATER_KNEE 0.4314\n#define WATER_MAX 0.5294\n#define WATER_KNEE_FAR 0.3922\n#define SHOULDER_BLEND_NEAR 50.0\n#define SHOULDER_BLEND_FAR 70.0\n#define SHOULDER_FAR_N 3.0\n#define LUMA 0.2126, 0.7152, 0.0722\n#define GLINT_FADE_REF 0.03\n#define GLINT_FADE_NEAR 15.0\n#define GLINT_FADE_FAR 30.0\n";
// Translucency (subsurface light through thin crests), kept together. vThin is the
// physics' transmission through the crest (surface.js, TRANSLUCENCY): 0 on flat or thick
// water. Light reaching the eye through it: the sun (TRANSL_SCATTER x irradiance x a
// Henyey-Greenstein phase of (V . -L), g = TRANSL_G, strongest with the sun behind the
// wave) and the bright sky behind the crest (TRANSL_SKY). Same constants at every wind.
// The water is pulled toward a lighter green-teal (TRANSL_COLOR, luminance about 130),
// saturating softly at TRANSL_MAX, so it
// never turns white and the body stays teal; glint and foam are drawn on top as before.
const TRANSL_DEFS = "#define TRANSL_COLOR 0.22, 0.60, 0.50\n#define TRANSL_G 0.7\n#define TRANSL_SCATTER 80.0\n#define TRANSL_SKY 6.5\n#define TRANSL_TIP 0.8\n#define TRANSL_MAX 0.85\n";
const LACE_DEFS = "#define DIM_LACE 0.3\n#define DIM_DENSE 0.64\n#define LACE_HOLE_BASE 0.06\n#define LACE_HOLE_DENSE 0.08\n#define LACE_HOLE_SOFT 0.08\n#define LACE_CAP 0.62\n#define LACE_WARP 0.5\n#define LACE_EDGE 0.3\n#define LACE_RIDGE_LO 0.65\n#define LACE_RIDGE_HI 0.9\n#define LACE_MASK_LO 0.6\n#define LACE_SHIFT_X 2.5\n#define LACE_SHIFT_Y 1.5\n#define LACE_DASH 0.45\n#define LACE_SHOULDER 0.8\n#define LACE_FINE_LINE 0.35\n#define LACE_SH_LO 0.45\n#define LACE_SH_HI 0.65\n#define LACE_MASK_HI 0.95\n";

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
float valueNoise(vec2 p) {
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
vec2 patchSlope(sampler2D tex, float size, float scale, vec2 p, float s, float o) {
  float psiA = 6.2832 * (valueNoise(p / (41.0 * s) + (11.0 + o)) + 0.6 * valueNoise(p / (17.0 * s) + (23.0 + o)));
  float psiB = 6.2832 * (valueNoise(p / (37.0 * s) + (41.0 + o)) + 0.6 * valueNoise(p / (15.0 * s) + (53.0 + o)));
  float psiC = 6.2832 * (valueNoise(p / (33.0 * s) + (71.0 + o)) + 0.6 * valueNoise(p / (13.0 * s) + (83.0 + o)));
  float wA = valueNoise(p / (23.0 * s) + (101.0 + o)) + 0.02;
  float wB = valueNoise(p / (23.0 * s) + (117.0 + o)) + 0.02;
  float wC = valueNoise(p / (23.0 * s) + (131.0 + o)) + 0.02;
  float total = wA + wB + wC;
  vec2 a = chopTurn(patchFetch(tex, size, scale, p), psiA);
  vec2 b = chopTurn(patchFetch(tex, size, scale, p + vec2(0.383, 0.617) * size), psiB);
  vec2 c = chopTurn(patchFetch(tex, size, scale, p + vec2(0.711, 0.229) * size), psiC);
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
  vec3 skyHorizon = vec3(0.58, 0.66, 0.70);
  vec3 skyZenith = vec3(SKY_ZENITH);
  float cosV = clamp(dot(nn, V), 0.0, 1.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
  vec3 R = reflect(-V, nn);
  vec3 skyR = mix(skyHorizon, skyZenith, sqrt(clamp(R.y, 0.0, 1.0)));
  w = fresnel * smoothstep(SKY_SEEN_LO, SKY_SEEN_HI, R.y) * REFLECT_GAIN;
  return skyR * w;
}
${LOBE_GLSL}
void main() {
  vec3 nBig = normalize(vNormal);
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
  vec3 trough = vec3(0.012, 0.04, 0.07);
  vec3 face = vec3(0.06, 0.30, 0.34);
  vec3 color = mix(trough, face, shade);

  float lip = smoothstep(0.45, 0.85, vLip);
  float depthFade = clamp(10.0 / max(depth, 2.0), 0.2, 1.0);
  color = mix(color, vec3(0.14, 0.44, 0.46), lip * depthFade);

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
  float reflShare = 0.25 * (w0 + w1 + w2 + w3);
  color = color * (1.0 - reflShare) + 0.25 * sky;

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
  color += vec3(glintRadiance);
  float waterY = dot(color, vec3(LUMA));
  float overKnee = max(waterY - WATER_KNEE, 0.0);
  float yNear = waterY > WATER_KNEE ? WATER_KNEE + (WATER_MAX - WATER_KNEE) * (1.0 - exp(-overKnee / (WATER_MAX - WATER_KNEE))) : waterY;
  float spanFar = WATER_MAX - WATER_KNEE_FAR;
  float overFar = max(waterY - WATER_KNEE_FAR, 0.0);
  float yFar = waterY > WATER_KNEE_FAR ? WATER_KNEE_FAR + spanFar * (1.0 - pow(1.0 + overFar / (SHOULDER_FAR_N * spanFar), -SHOULDER_FAR_N)) : waterY;
  float rolledY = mix(yNear, yFar, smoothstep(SHOULDER_BLEND_NEAR, SHOULDER_BLEND_FAR, distEye));
  color *= waterY > 1e-5 ? rolledY / waterY : 1.0;

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
  float fine = 1.0 - smoothstep(25.0, 70.0, dist);
  float grain = valueNoise(q * vec2(1.0 / 0.9, 1.0 / 2.6));
  grain = mix(grain, 0.6 * grain + 0.4 * valueNoise(q * vec2(1.0 / 0.35, 1.0 / 0.9) + 17.0), fine);
  // Noise only removes foam: alpha never exceeds amount, and the core near the crest
  // (amount close to 1) stays solid while the edges and ends break up.
  float alphaFar = amount * clamp((amount - 0.6 * grain - fadeCut) / 0.3, 0.0, 1.0) * fadeStrength;

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
  float alphaNearFresh = amountNear * smoothstep(0.0, 0.3, amountNear - 0.5 * tear - fadeCut) * fadeStrength;

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
#ifdef HAS_DERIV
  vec2 qw = fwidth(q);
#else
  vec2 qw = vec2(dist * 0.0012, dist * 0.0012);
#endif
  // 1 where a streak octave is several pixels wide, 0 where it is under a pixel.
  float detailC = 1.0 - smoothstep(0.5, 1.0, max(qw.x / 2.8, qw.y / 0.45));
  float detailF = 1.0 - smoothstep(0.5, 1.0, max(qw.x / 0.9, qw.y / 0.18));
  float streak = 0.65 * mix(0.5, valueNoise(q * vec2(1.0 / 2.8, 1.0 / 0.45) + 53.0), detailC)
               + 0.35 * mix(0.5, valueNoise(q * vec2(1.0 / 0.9, 1.0 / 0.18) + 71.0), detailF);
  float oldAmount = smoothstep(0.6, 1.0, marked) * smoothstep(0.1, 0.6, vFoam);
  float cover = 0.3 + 0.55 * (1.0 - fadeV);
  float strengthOld = mix(0.6, 0.8, fadeV);
  float alphaSharp = smoothstep(cover, cover + 0.18, streak * oldAmount);
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
  float laceThin = valueNoise(vec2(q.x / 1.7, yJ / 0.08) + 217.0);
  float laceMid = valueNoise(vec2(q.x / 2.3, yJ / 0.16) + 219.0);
  float laceWide = valueNoise(vec2(q.x / 3.2, yJ / 0.3) + 221.0);
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
  float markedW = marked + dot(gM, warpQ) * detailL;
  float foamW = vFoam + dot(gF, warpQ) * detailL;
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
  oldness = mix(oldness, oldnessW, detailL);
  // Never past the flag outline (drawn at marked 0.5), whatever the displacement.
  float oldAmountL = smoothstep(0.5, 0.58, marked)
                   * smoothstep(edgeAt - 0.05, edgeAt + 0.05, markedW) * smoothstep(gradeAt - 0.08, gradeAt + 0.08, foamW);
  oldAmount = mix(oldAmount, oldAmountL, detailL);
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
  float coverJ = cover + LACE_EDGE * (valueNoise(q * vec2(1.0 / 0.7, 1.0 / 0.4) + 233.0) - 0.58) * detailL;
  alphaSharp = smoothstep(coverJ, coverJ + 0.18, streak * oldAmount);
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
  // GAD (c): fading foam keeps a dim lace (alpha up to DIM_LACE, peak under 130), only
  // inside the flag outline (edgeAt >= LACE_MASK_LO 0.6).
  // GAD 06:27: no floor. The lace fades to 0 with age (fadeV) and its edge is broken up by
  // world-space noise (edgeAt / gradeAt thresholds at ~1 m and 0.35 m, as in v25), so it never
  // shows as a flat sheet with a cell outline; inside, sub-pixel lace is replaced by ~1 m noise.
  float dimLace = DIM_LACE * fadeV * smoothstep(edgeAt - 0.08, edgeAt + 0.08, marked) * smoothstep(gradeAt - 0.1, gradeAt + 0.1, vFoam)
                * mix(0.25 + 0.75 * holeN, lace, detailL);
  alpha = max(alpha, dimLace);
  // Foam takes the fine chop's shading only where it darkens: never brighter than the
  // big grid's lighting alone would make it.
  vec3 foamColor = vec3(0.75, 0.81, 0.82) * (0.92 + 0.08 * min(lightFoam, lightBig));
  // Old streaks also take a little shade on ripple sides turned from the sun
  // (up to 15% darker). Fresh foam is not touched.
  float awayFromSun = smoothstep(0.0, 0.12, dot(fs, normalize(L.xz)));
  foamColor *= 1.0 - 0.15 * awayFromSun * oldness;
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
  color += transl * (1.0 - reflShare) * max(vec3(TRANSL_COLOR) - color, 0.0) * (1.0 - smoothstep(0.0, 0.05, alpha));
  // GAD 04:15 over-130: dense foam is dimmed, not cut: where alpha nears solid the foam
  // colour drops to DIM_DENSE of itself (about 125 instead of 190), so the foam area and
  // the lace (alpha <= DIM_LACE) are unchanged.
  foamColor *= mix(1.0, DIM_DENSE, smoothstep(0.35, 0.8, alpha));
  color = mix(color, foamColor, alpha);

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
    color = mix(color, amber, max(insideMarked * (1.0 - inside) * 0.18, edgeMarked * 0.9));
    color = mix(color, red, max(max(inside * 0.22, breakSample * 0.45), edge * 0.9));
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

export function createView(canvas) {
  // WebGL2 when available (GPU physics needs it); WebGL1 still draws the CPU physics.
  const gl = canvas.getContext("webgl2", { antialias: true }) || canvas.getContext("webgl", { antialias: true });
  if (!gl) throw new Error("WebGL is not available");
  const webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  // Derivatives give a one-pixel-ish flag outline at any distance. Without them the
  // outline is a fixed band in attribute space.
  const derivatives = webgl2 || gl.getExtension("OES_standard_derivatives");
  const fragBody = (derivatives ? "#extension GL_OES_standard_derivatives : enable\n#define HAS_DERIV 1\n" : "") + REFLECT_DEFS + FINE_DEFS + LIGHT_DEFS + TRANSL_DEFS + LACE_DEFS + FRAG;
  const fragShader = compile(gl, gl.FRAGMENT_SHADER, webgl2 ? toEs3(fragBody, "fragment") : fragBody);
  // GPU physics: the patches' slope variance per mip level comes from their stats textures
  // (level L's variance at texel L; kept = level L / level 0, as keptPerLevel computes),
  // so nothing is read back. The rest of the shader is the same text.
  const gpuFragShader = webgl2 ? compile(gl, gl.FRAGMENT_SHADER, toEs3(fragBody
    .replace("uniform float chopKept[8];\nuniform float fineKept[8];", "float chopKept[8];\nfloat fineKept[8];\nuniform highp sampler2D chopStats;\nuniform highp sampler2D fineStats;")
    .replace("void main() {", `void main() {
  {
    float c0 = texelFetch(chopStats, ivec2(0, 0), 0).x;
    float f0 = texelFetch(fineStats, ivec2(0, 0), 0).x;
    for (int i = 0; i < 8; i += 1) {
      chopKept[i] = texelFetch(chopStats, ivec2(i, 0), 0).x / (c0 > 0.0 ? c0 : 1.0);
      fineKept[i] = texelFetch(fineStats, ivec2(i, 0), 0).x / (f0 > 0.0 ? f0 : 1.0);
    }
  }`), "fragment")) : null;
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
  const fftFragShader = webgl2 ? compile(gl, gl.FRAGMENT_SHADER, toEs3(fragBody
    .replace("uniform float chopKept[8];\nuniform float fineKept[8];", "float chopKept[8];\nfloat fineKept[8];\nuniform highp sampler2D chopStats;\nuniform highp sampler2D fineStats;")
    .replace("uniform float subMss;", "uniform float subMssU;\nvarying float vLodLost;\nfloat subMss;")
    // GAD glow: the FFT sea's vThin is interpolated over 2.2 m triangles, so a lip's ramp
    // (vThin 0.05-0.15) glowed around it; transmission now starts at the lip itself
    // (vThin 0.14-0.18) with gain 4.2 so the tips keep v26's 94-102.
    .replace("float transl = TRANSL_MAX * (1.0 - exp(-vThin * TRANSL_SKY));", "float thinE = min(1.0, 4.2 * vThin * smoothstep(0.14, 0.18, vThin)); float transl = TRANSL_MAX * (1.0 - exp(-thinE * TRANSL_SKY));")
    .replace("float translSun = vThin * TRANSL_SCATTER * sunIn;", "float translSun = thinE * TRANSL_SCATTER * sunIn;")
    .replace("vec3 nBig = normalize(vNormal);", "vec3 nBig = normalize(mix(normalize(vNormal), seaNormalPix(vSea.xy, vSea.z), clamp(vSea.w, 0.0, 1.0)));")
    // GAD per-pixel foam (see foamPix in fftView.js): the foam varyings become globals set per pixel.
    .replace("varying float vFoam;", "varying float vFoam;\nfloat fFoam; float rawFoam() { return vFoam; }\n#define vFoam fFoam\n")
    .replace("varying float vFlag;", "varying float vFlag;\nfloat fFlag; float rawFlag() { return vFlag; }\n#define vFlag fFlag\n")
    .replace("varying float vRecent;", "varying float vRecent;\nfloat fRecent; float rawRecent() { return vRecent; }\n#define vRecent fRecent\n")
    .replace("varying float vFade;", "varying float vFade;\nfloat fFade; float rawFade() { return vFade; }\n#define vFade fFade\n")
    .replace("varying float vAge;", "varying float vAge;\nfloat fAge; float rawAge() { return vAge; }\n#define vAge fAge\n")
    .replace("varying float vAhead;", "varying float vAhead;\nfloat fAhead; float rawAhead() { return vAhead; }\n#define vAhead fAhead\n")
    .replace("varying float vRun;", "varying float vRun;\nfloat fRun; float rawRun() { return vRun; }\n#define vRun fRun\n")
    .replace("void main() {", "#define FOAM_PIX " + ((typeof location !== "undefined" && new URLSearchParams(location.search).get("foampix") === "0") ? "false" : "true") + "\n" + FRAG_FFT_NORMAL + `void main() {
  if (FOAM_PIX) {
    vec4 pf, pg; foamPix(vSea.xy, pf, pg);
    float kp = clamp(vFoamKeep, 0.0, 1.0);
    vFoam = pf.x * kp; vFade = pf.y * kp; vAge = pf.z; vAhead = pg.x; vRun = pg.y; vRecent = pg.z * kp; vFlag = pg.w * kp;
  } else { vFoam = rawFoam(); vFlag = rawFlag(); vRecent = rawRecent(); vFade = rawFade(); vAge = rawAge(); vAhead = rawAhead(); vRun = rawRun(); }
  subMss = subMssU + vLodLost;
  {
    float c0 = texelFetch(chopStats, ivec2(0, 0), 0).x;
    float f0 = texelFetch(fineStats, ivec2(0, 0), 0).x;
    for (int i = 0; i < 8; i += 1) {
      chopKept[i] = texelFetch(chopStats, ivec2(i, 0), 0).x / (c0 > 0.0 ? c0 : 1.0);
      fineKept[i] = texelFetch(fineStats, ivec2(i, 0), 0).x / (f0 > 0.0 ? f0 : 1.0);
    }
  }`), "fragment")) : null;
  const fftProgram = webgl2 ? link(VERT_FFT, fftFragShader) : null;
  const vao = webgl2 ? gl.createVertexArray() : null;
  if (vao) gl.bindVertexArray(vao);
  gl.enable(gl.DEPTH_TEST);
  const cpuLocs = locate(cpuProgram);
  const gpuLocs = gpuProgram ? locate(gpuProgram) : null;
  const fftLocs = fftProgram ? locate(fftProgram) : null;
  if (fftLocs) {
    for (const name of ["physA", "physB", "physL", "physF", "physG", "physN", "chopStats", "fineStats", "fftR0", "fftR1", "fftR2", "fftR3", "fftStats", "fftLevels", "fftL", "choppiness", "pixAngle", "lodBias", "minLod", "antiTile", "xCount", "x0", "x1"]) {
      fftLocs[name] = gl.getUniformLocation(fftProgram, name);
    }
    fftLocs.subMss = gl.getUniformLocation(fftProgram, "subMssU");
    fftLocs.meta = gl.getAttribLocation(fftProgram, "meta");
  }
  const fftMesh = { xz: gl.createBuffer(), meta: gl.createBuffer(), index: gl.createBuffer(), count: 0, key: "" };
  const fftParams = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
  const lodBias = Number(fftParams.get("lodBias") || 1);
  const minLod = Number(fftParams.get("minLod") || 0);
  const antiTile = fftParams.get("antiTile") === "0" ? 0 : 1;
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
    gpuLocs.chopStats = gl.getUniformLocation(gpuProgram, "chopStats");
    gpuLocs.fineStats = gl.getUniformLocation(gpuProgram, "fineStats");
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
  const camera = { yaw: 0.75, pitch: 0.5, distance: 155 };
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
    camera.distance = Math.max(40, Math.min(1400, camera.distance * (event.deltaY > 0 ? 1.08 : 0.92)));
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
  function draw(state, showFlags) {
    // The GPU physics shares this context: take back the program, vertex state, target
    // and depth test every frame.
    const fromTextures = !!(state.gpu && gpuProgram);
    const fft = fromTextures && state.gpu.fft && fftProgram ? state.gpu.fft : null;
    program = fft ? fftProgram : fromTextures ? gpuProgram : cpuProgram;
    locs = fft ? fftLocs : fromTextures ? gpuLocs : cpuLocs;
    gl.useProgram(program);
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
      const m = buildFftMesh(fft.N, fft.L);
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.xz);
      gl.bufferData(gl.ARRAY_BUFFER, m.xz, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, fftMesh.meta);
      gl.bufferData(gl.ARRAY_BUFFER, m.meta, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, fftMesh.index);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.index, gl.STATIC_DRAW);
      fftMesh.count = m.index.length;
      fftMesh.chunks = m.chunks;
      fftMesh.key = `${fft.N}/${fft.L}`;
    }
    if (!fft && (builtFor !== state.n || builtSize !== state.size)) {
      buildGrid(state.n, state.size);
      if (builtSize !== state.size) {
        camera.distance = Math.max(40, Math.min(1400, state.size * 0.62));
      }
      builtSize = state.size;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.58, 0.66, 0.70, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (fft && builtSize !== state.size) {
      camera.distance = Math.max(40, Math.min(1400, state.size * 0.62));
      builtSize = state.size;
    }
    const eye = topView ? [0, topHeight, 0] : [
      Math.cos(camera.pitch) * Math.sin(camera.yaw) * camera.distance,
      Math.sin(camera.pitch) * camera.distance,
      Math.cos(camera.pitch) * Math.cos(camera.yaw) * camera.distance,
    ];
    const view = lookAt(eye, [0, 0, 0], topView ? [0, 0, -1] : [0, 1, 0]);
    // v28: the FFT sea reaches past 2 km.
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, fft ? 1 : 0.5, fft ? (topView ? topHeight + 200 : 4500) : 2000);
    const vpMat = multiply(proj, view);
    gl.uniformMatrix4fv(locs.viewProj, false, vpMat);
    gl.uniform3f(locs.lightDir, sun[0], sun[1], sun[2]);
    gl.uniform3f(locs.eye, eye[0], eye[1], eye[2]);
    gl.uniform1f(locs.depth, state.depth);
    gl.uniform1f(locs.showFlags, showFlags ? 1 : 0);
    gl.uniform2f(locs.travel, state.travelX || 0, state.travelZ || 1);
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
    if (fromTextures) {
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, gpuChop ? gpuChop.statsTex : null);
      gl.uniform1i(locs.chopStats, 7);
      gl.activeTexture(gl.TEXTURE8);
      gl.bindTexture(gl.TEXTURE_2D, gpuFine ? gpuFine.statsTex : null);
      gl.uniform1i(locs.fineStats, 8);
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
      const units = [["physA", g.A], ["physB", g.B], ["physL", g.L], ["physF", g.F], ["physG", g.G],
        ["fftR0", fft.R[0]], ["fftR1", fft.R[1]], ["fftR2", fft.R[2]], ["fftR3", fft.R[3]], ["fftStats", fft.stats]];
      const unitOf = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13];
      units.forEach(([name, tex], i) => {
        gl.activeTexture(gl.TEXTURE0 + unitOf[i]);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(locs[name], unitOf[i]);
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(locs.physN, fft.N);
      gl.uniform1f(locs.fftLevels, fft.levels);
      gl.uniform1f(locs.fftL, fft.L);
      gl.uniform1f(locs.choppiness, fft.choppiness);
      gl.uniform1f(locs.pixAngle, (Math.PI / 4) / canvas.height);
      gl.uniform1f(locs.lodBias, lodBias);
      gl.uniform1f(locs.minLod, minLod);
      gl.uniform1f(locs.antiTile, antiTile);
      const count = Math.min(fft.explicitCount, 64);
      gl.uniform1i(locs.xCount, count);
      if (count > 0) {
        gl.uniform4fv(locs.x0, fft.explicit0.subarray(0, count * 4));
        gl.uniform4fv(locs.x1, fft.explicit1.subarray(0, count * 4));
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, fftMesh.index);
      // v28 perf: only the chunks that can reach the screen (?nocull=1 draws all, for checks)
      const ranges = noCull ? [[0, fftMesh.count]] : visibleRanges(fftMesh.chunks, vpMat, 60);
      fftMesh.drawn = ranges.reduce((a, r) => a + r[1], 0);
      if (typeof window !== "undefined") window.__fftDrawn = [fftMesh.drawn / 3, fftMesh.count / 3];
      timed("render:fftMesh", () => { for (const [first, count] of ranges) gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, first * 4); });
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
    return Math.sin(camera.pitch) * camera.distance;
  }

  return { draw, cameraHeight, gl, webgl2 };
}

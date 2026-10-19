// v28: drawing the FFT sea (gpuSurface.js + fftSea.js) out to the horizon.
//
// One mesh, built once per (N, L): the physics tile, (N+1)^2 vertices at the physics grid
// spacing L/N, then square rings, each twice as wide with twice the spacing, out past 2 km
// (4 rings). Where a ring meets the finer region inside it, the finer side's odd edge
// vertices take the average of their two neighbours (no T-junction cracks).
//
// Per vertex (VERT_FFT):
//  - in the tile, x and z in [-L/2, L/2): the physics fields (height, shift, normal, and the
//    break/foam values), exactly as v27 drew them;
//  - elsewhere (the +L/2 edges included): the FFT sea from its mipmapped half-float copies,
//    sampled three times (offsets and phase turns, the one-sided fields turned by e^(i a)),
//    blended with unit-energy weights from a slow world mask; the weight is all on the plain
//    copy at the tile edge and ramps to the mix over 0.6 L, so the repeat does not line up.
//    The explicit long waves (too long for the tile) are added analytically everywhere
//    outside the tile.
//  - Distance fade: the copies are sampled at a mip level from the pixel footprint (and
//    never finer than the ring's own spacing); in the tile, the slope detail finer than
//    that level is taken off the physics normal. The slope variance so removed (from the
//    per-level stats) goes to the fragment shader as vLodLost and into the glint lobe, so
//    the roughness is conserved and the detail turns into sheen, not sparkle.
export const FFT_RINGS_TO = 2000; // metres, at least

export const FRAG_FFT_NORMAL = `
in vec4 vSea;
in vec4 vXs; // explicit waves at the vertex: slope xz, squeeze xx zz
in float vXq; // squeeze xz
uniform highp sampler2D fftR1;
uniform highp sampler2D fftR2;
uniform highp sampler2D fftR3;
uniform highp float fftL;
uniform highp int physN;
uniform highp float choppiness;
uniform highp float antiTile;
uniform float oneCopyFar; // v28-lod: 1 = one FFT copy per pixel past 600 m (faded in over 550-650 m)
uniform highp sampler2D physF;
uniform highp sampler2D physG;
in float vFoamKeep;
// GAD per-pixel foam: grade, fade, age, ahead, run, recent and flag read bilinearly from the
// physics textures at this pixel's undisplaced position (vSea.xy), not interpolated from the
// mesh vertices, so single-cell foam shows on the thinned rings too. The values are the same
// physics fields (CPU parity untouched); only where they are sampled changes.
void foamPix(vec2 p, out vec4 F, out vec4 G) {
  float sp = fftL / float(physN);
  vec2 u = (p + 0.5 * fftL) / sp;
  vec2 i0 = floor(u); vec2 fr = u - i0;
  ivec2 c00 = (ivec2(i0) % physN + physN) % physN;
  ivec2 c10 = ((c00 + ivec2(1, 0)) % physN), c01 = ((c00 + ivec2(0, 1)) % physN), c11 = ((c00 + ivec2(1, 1)) % physN);
  F = mix(mix(texelFetch(physF, c00, 0), texelFetch(physF, c10, 0), fr.x), mix(texelFetch(physF, c01, 0), texelFetch(physF, c11, 0), fr.x), fr.y);
  G = mix(mix(texelFetch(physG, c00, 0), texelFetch(physG, c10, 0), fr.x), mix(texelFetch(physG, c01, 0), texelFetch(physG, c11, 0), fr.x), fr.y);
}
const float TAUF = 6.283185307179586;
// GAD ring edge: the sea's normal per pixel from the FFT slope texture (analytic slopes),
// not interpolated between vertices: ring 2 (8.8 m) has ~1.2 vertices per resolved texel
// at 420 m and its interpolated normals came out flatter than ring 1's (a contrast step).
vec3 seaNormalPix(vec2 p, float lod) {
  float half_ = 0.5 * fftL;
  float ramp = antiTile * smoothstep(half_, half_ + 0.6 * fftL, max(abs(p.x), abs(p.y)));
  vec2 q = p / fftL;
  float ang = TAUF * (0.45 * q.x + 0.31 * q.y) + 2.0 * sin(TAUF * (0.19 * q.x - 0.37 * q.y) + 0.7);
  vec3 w = max(vec3(cos(ang), cos(ang - TAUF / 3.0), cos(ang + TAUF / 3.0)), 0.0) + 1e-3;
  w = mix(vec3(1.0, 0.0, 0.0), w, ramp); w /= length(w);
  float one = oneCopyFar * smoothstep(550.0, 650.0, length(eye - vWorld));
  if (one > 0.0) { w = mix(w, vec3(1.0, 0.0, 0.0), one); w /= length(w); if (one >= 1.0) w = vec3(1.0, 0.0, 0.0); }
  vec3 drift = TAUF * vec3(ramp * dot(q, vec2(0.26, -0.24)), dot(q, vec2(0.23, 0.27)), dot(q, vec2(-0.27, 0.21)));
  vec2 offs[3] = vec2[3](vec2(0.0), vec2(0.37, 0.71), vec2(0.83, 0.19));
  float turns[3] = float[3](0.0, 2.1, 4.2);
  vec2 sl = vec2(0.0); vec3 sq = vec3(0.0);
  for (int j = 0; j < 3; j += 1) {
    if (w[j] < 1e-4) continue;
    vec2 uv = (p + half_) / fftL + offs[j] + 0.5 / float(physN);
    vec4 r1 = textureLod(fftR1, uv, lod);
    vec4 r2 = textureLod(fftR2, uv, lod);
    float cs = cos(turns[j] + drift[j]), sn = sin(turns[j] + drift[j]);
    sl += w[j] * vec2(cs * r1.z - sn * r1.w, cs * r2.x - sn * r2.y);
    if (j == 0) sq += w[j] * vec3(r2.z, r2.w, textureLod(fftR3, uv, lod).x);
  }
  sl += vXs.xy; sq += vec3(vXs.zw, vXq);
  float jxx = 1.0 - choppiness * sq.x, jzz = 1.0 - choppiness * sq.y, jxz = -choppiness * sq.z;
  return normalize(vec3(sl.y * jxz - sl.x * jzz, jxx * jzz - jxz * jxz, jxz * sl.x - sl.y * jxx));
}
`;

export const VERT_FFT = `#version 300 es
#define MAXX 64
in vec2 xz;
in vec4 meta; // ring index (unused by the shader), morph dx, morph dz, 1 on tile vertices
uniform mat4 viewProj;
uniform highp sampler2D physA;
uniform highp sampler2D physB;
uniform highp sampler2D physL;
uniform highp sampler2D physF;
uniform highp sampler2D physG;
uniform highp int physN;
uniform highp sampler2D fftR0; // Zh, Zdx      (one-sided: Re/Im)
uniform highp sampler2D fftR1; // Zdz, ZsxN
uniform highp sampler2D fftR2; // ZszN, squeeze xx, zz (real)
uniform highp sampler2D fftR3; // squeeze xz (real)
uniform highp sampler2D fftStats; // per-level shading-slope variance (texel L)
uniform float fftLevels;
uniform highp float fftL;
uniform highp float choppiness;
uniform vec3 eye;
uniform float pixAngle;  // radians per pixel
uniform float lodBias;
uniform float farHalve; // v28-lod: 1 = rings past 840 m at a quarter of the v28 density
uniform float ringLodAdd; // v28-lod: 1 = the half-density rings sample a level coarser (?ringlodadd=0 for tests)
uniform float tileThin; // v28-lod (GAD 01:08): 1 = tile mesh at half density past TILE_THIN_D1 from the camera
const float TILE_THIN_D0 = 60.0;  // odd tile vertices start to morph (horizontal metres from the camera)
const float TILE_THIN_D1 = 75.0; // fully morphed; chunks wholly past this draw at half density
uniform float minLod;    // shading detail never finer than this level (near-Nyquist waves blink on the mesh)   // footprint multiplier (> 1 fades sooner)
uniform float antiTile;  // 1: mixed copies away from the tile; 0: plain repeat (review only)
uniform highp int xCount;
uniform vec4 x0[MAXX];   // amplitude, k, cos dir, sin dir
uniform vec4 x1[MAXX];   // phase at the origin in turns, 1 for the chop band
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
out float vLodLost;
out float vFoamKeep; // GAD per-pixel foam: tile weight (0 off the tile, 1-edgeW on it)
out vec3 vNormal;
out vec4 vSea;
out vec4 vXs;
out float vXq; // undisplaced x, z, lod, per-pixel normal weight
out vec3 vWorld;
const float TAU = 6.283185307179586;
const bool PIXN = true;
float levelVar(float lod) {
  float l0 = floor(lod);
  float a = texelFetch(fftStats, ivec2(int(min(l0, fftLevels - 1.0)), 0), 0).x;
  float b = texelFetch(fftStats, ivec2(int(min(l0 + 1.0, fftLevels - 1.0)), 0), 0).x;
  return mix(a, b, lod - l0);
}
// Explicit waves at p: height, shift (before choppiness), shading slope, squeeze xx zz xz.
// GAD ring edge: each explicit wave is filtered like the FFT mips at this lod (box of one
// texel, spacing 2^lod: Gaussian equivalent exp(-(k texel)^2 / 24)); unfiltered, its
// slope was sampled exactly by ring 1 and smoothed by ring 2's coarser mesh (a contrast step).
void explicitAt(vec2 p, float lod, inout float h, inout vec2 sh, inout vec2 sl, inout vec3 sq) {
  float texel = fftL / float(physN) * exp2(lod);
  for (int i = 0; i < MAXX; i += 1) {
    if (i >= xCount) break;
    vec4 a = x0[i];
    vec4 b = x1[i];
    float kt = a.y * texel;
    a.x *= exp(-kt * kt / 24.0);
    float t = fract(a.y * (a.z * p.x + a.w * p.y) / TAU + b.x);
    float s = sin(TAU * t), c = cos(TAU * t);
    h += a.x * s;
    sh += a.x * c * a.zw;
    vec2 g = a.x * a.y * c * a.zw;
    sl += g * (b.y > 0.5 ? 2.0 : 1.0);
    sq += a.x * a.y * s * vec3(a.z * a.z, a.w * a.w, a.z * a.w);
  }
}
// v28 perf: the same waves at a second lod (lodB, the per-pixel lod) in the same loop: one
// sin/cos per wave instead of two. The lod-A sums are the same operations as explicitAt.
void explicitAt2(vec2 p, float lod, float lodB, inout float h, inout vec2 sh, inout vec2 sl, inout vec3 sq, inout vec2 slB, inout vec3 sqB) {
  float texel = fftL / float(physN) * exp2(lod);
  float texelB = fftL / float(physN) * exp2(lodB);
  for (int i = 0; i < MAXX; i += 1) {
    if (i >= xCount) break;
    vec4 a = x0[i];
    vec4 b = x1[i];
    float amp = a.x;
    float kt = a.y * texel;
    a.x *= exp(-kt * kt / 24.0);
    float t = fract(a.y * (a.z * p.x + a.w * p.y) / TAU + b.x);
    float s = sin(TAU * t), c = cos(TAU * t);
    h += a.x * s;
    sh += a.x * c * a.zw;
    vec2 g = a.x * a.y * c * a.zw;
    sl += g * (b.y > 0.5 ? 2.0 : 1.0);
    sq += a.x * a.y * s * vec3(a.z * a.z, a.w * a.w, a.z * a.w);
    float ktB = a.y * texelB;
    float aB = amp * exp(-ktB * ktB / 24.0);
    vec2 gB = aB * a.y * c * a.zw;
    slB += gB * (b.y > 0.5 ? 2.0 : 1.0);
    sqB += aB * a.y * s * vec3(a.z * a.z, a.w * a.w, a.z * a.w);
  }
}
// The FFT sea outside the tile: three turned, offset samplings, unit-energy weights.
float rampAt(vec2 p) {
  float half_ = 0.5 * fftL;
  return antiTile * smoothstep(half_, half_ + 0.6 * fftL, max(abs(p.x), abs(p.y)));
}
vec3 weightsAt(vec2 p) {
  float a = rampAt(p);
  vec2 q = p / fftL;
  // A one-tile shift turns the mask by 0.45 turn in x, 0.31 in z (0.76 diagonally) plus a
  // slow wobble, so a point and its repeats get different copy mixes.
  float ang = TAU * (0.45 * q.x + 0.31 * q.y) + 2.0 * sin(TAU * (0.19 * q.x - 0.37 * q.y) + 0.7);
  vec3 c = max(vec3(cos(ang), cos(ang - TAU / 3.0), cos(ang + TAU / 3.0)), 0.0) + 1e-3;
  c = mix(vec3(1.0, 0.0, 0.0), c, a);
  return c / length(c);
}
void seaCore(vec2 p, float lod, out float h, out vec2 sh, out vec2 sl, out vec3 sq) {
  vec3 w = weightsAt(p);
  // Each copy's phase turn also drifts across the sea (about a quarter turn per tile), so
  // a point and its repeat one tile away see different turns of the same waves. The
  // drift is a wavevector shift of under 0.3 dk: no visible change to the waves.
  vec2 q = p / fftL;
  float ramp = rampAt(p);
  vec3 drift = TAU * vec3(ramp * dot(q, vec2(0.26, -0.24)), dot(q, vec2(0.23, 0.27)), dot(q, vec2(-0.27, 0.21)));
  vec2 offs[3] = vec2[3](vec2(0.0), vec2(0.37, 0.71), vec2(0.83, 0.19));
  float turns[3] = float[3](0.0, 2.1, 4.2);
  h = 0.0; sh = vec2(0.0); sl = vec2(0.0); sq = vec3(0.0);
  for (int j = 0; j < 3; j += 1) {
    if (w[j] < 1e-4) continue;
    // texel centres: field sample n sits at x = -L/2 + n L/N
    vec2 uv = (p + 0.5 * fftL) / fftL + offs[j] + 0.5 / float(physN);
    vec4 r0 = textureLod(fftR0, uv, lod);
    vec4 r1 = textureLod(fftR1, uv, lod);
    vec4 r2 = textureLod(fftR2, uv, lod);
    float cs = cos(turns[j] + drift[j]), sn = sin(turns[j] + drift[j]);
    // Re(e^(ia) z) = cs re - sn im; Im(e^(ia) z) = sn re + cs im
    h += w[j] * (sn * r0.x + cs * r0.y);
    sh += w[j] * vec2(cs * r0.z - sn * r0.w, cs * r1.x - sn * r1.y);
    sl += w[j] * vec2(cs * r1.z - sn * r1.w, cs * r2.x - sn * r2.y);
    if (j == 0) sq += w[j] * vec3(r2.z, r2.w, textureLod(fftR3, uv, lod).x);
  }
}
void seaAt(vec2 p, float lod, out float h, out vec2 sh, out vec2 sl, out vec3 sq) {
  seaCore(p, lod, h, sh, sl, sq);
  explicitAt(p, lod, h, sh, sl, sq);
}
// seaAt plus the explicit waves' slope and squeeze at the per-pixel lod (into slB, sqB).
void seaAtPix(vec2 p, float lod, float lodB, out float h, out vec2 sh, out vec2 sl, out vec3 sq, inout vec2 slB, inout vec3 sqB) {
  seaCore(p, lod, h, sh, sl, sq);
  explicitAt2(p, lod, lodB, h, sh, sl, sq, slB, sqB);
}
vec3 normalFrom(vec2 sl, vec3 sq) {
  float jxx = 1.0 - choppiness * sq.x;
  float jzz = 1.0 - choppiness * sq.y;
  float jxz = -choppiness * sq.z;
  vec3 n = vec3(sl.y * jxz - sl.x * jzz, jxx * jzz - jxz * jxz, jxz * sl.x - sl.y * jxx);
  return normalize(n);
}
// v28-lod tile thinning: the tile vertex at p exactly as main() computes it (lod, edge
// crossfade, physics cell, foam fields, explicit-wave varyings), so an odd vertex can morph to
// the average of its two coarse-grid neighbours (CDLOD) with no crack where a half-density
// chunk starts.
void tileVert(vec2 p, out float h, out vec2 sh, out vec3 n, out vec4 L, out vec4 F, out vec4 G, out vec2 sx, out vec3 qx, out float edgeW) {
  float spacing = fftL / float(physN);
  float half_ = 0.5 * fftL;
  float d = length(eye - vec3(p.x, 0.0, p.y));
  float graze = max(eye.y / max(d, 1e-3), 0.05);
  float foot = lodBias * d * pixAngle / (spacing * sqrt(graze));
  float c = max(abs(p.x), abs(p.y)) / half_;
  float ringLod = log2(max(c, 1.0)) + 2.0 * clamp((c - 0.1) / 2.9, 0.0, 1.0);
  ringLod += ringLodAdd * (clamp((c - 1.0) / 0.5, 0.0, 1.0) + farHalve * clamp((c - 4.0) / 2.0, 0.0, 1.0));
  float lod = clamp(max(max(ringLod, log2(max(foot, 1.0))), minLod), 0.0, fftLevels - 1.0);
  float lodPix = clamp(max(log2(max(foot, 1.0)), minLod), 0.0, fftLevels - 1.0);
  edgeW = clamp((max(abs(p.x), abs(p.y)) - 10.0) / (half_ - 10.0 - spacing), 0.0, 1.0);
  ivec2 cell = ivec2(floor((p + half_) / spacing + 0.5));
  cell = (cell % physN + physN) % physN;
  vec4 a = texelFetch(physA, cell, 0);
  vec4 b = texelFetch(physB, cell, 0);
  h = a.x;
  sh = a.yz / max(choppiness, 1e-6);
  n = b.xyz;
  if (lod > 0.0) {
    vec2 uv = (p + half_) / fftL + 0.5 / float(physN);
    vec2 s0 = vec2(textureLod(fftR1, uv, 0.0).z, textureLod(fftR2, uv, 0.0).x);
    vec2 sL = vec2(textureLod(fftR1, uv, lod).z, textureLod(fftR2, uv, lod).x);
    vec2 slope = -n.xz / max(n.y, 1e-3) - (s0 - sL);
    n = normalize(vec3(-slope.x, 1.0, -slope.y));
  }
  sx = vec2(0.0); qx = vec3(0.0);
  if (edgeW > 0.0) {
    float h2; vec2 sh2, sl2; vec3 sq2;
    seaAtPix(p, lod, lodPix, h2, sh2, sl2, sq2, sx, qx);
    h = mix(h, h2, edgeW);
    sh = mix(sh, sh2, edgeW);
    if (!PIXN) n = normalize(mix(n, normalFrom(sl2, sq2), edgeW));
  } else {
    float hx = 0.0; vec2 shx = vec2(0.0);
    explicitAt(p, lodPix, hx, shx, sx, qx);
  }
  L = texelFetch(physL, cell, 0);
  F = texelFetch(physF, cell, 0);
  G = texelFetch(physG, cell, 0);
}
void main() {
  float spacing = fftL / float(physN);
  float half_ = 0.5 * fftL;
  vec2 p = xz;
  // footprint lod (the same function of position on both sides of every seam)
  float d = length(eye - vec3(p.x, 0.0, p.y));
  float graze = max(eye.y / max(d, 1e-3), 0.05);
  float foot = lodBias * d * pixAngle / (spacing * sqrt(graze));
  // Never finer than the mesh, and a continuous function of position (no seam steps).
  float c = max(abs(p.x), abs(p.y)) / half_;
    // Ring r (spacing 2^r cells, from c = 2^(r-1)) reaches level r + 1 by its inner edge:
  // the sampled detail stays a level coarser than the mesh, so no ring shows as a square.
  // GAD edge: starts inside the tile and rises gently, so no ring shows (>=100 m both sides).
  float ringLod = log2(max(c, 1.0)) + 2.0 * clamp((c - 0.1) / 2.9, 0.0, 1.0);
  // v28-lod: rings at half density (a quarter past 840 m with farHalve): the vertex samples a
  // level coarser again, ramped over the first part of the ring so the lod stays continuous.
  ringLod += ringLodAdd * (clamp((c - 1.0) / 0.5, 0.0, 1.0) + farHalve * clamp((c - 4.0) / 2.0, 0.0, 1.0));
  float lod = clamp(max(max(ringLod, log2(max(foot, 1.0))), minLod), 0.0, fftLevels - 1.0);
  float edgeW = clamp((max(abs(p.x), abs(p.y)) - 10.0) / (half_ - 10.0 - spacing), 0.0, 1.0);
  bool tile = p.x >= -half_ - 1e-3 && p.y >= -half_ - 1e-3 && p.x < half_ - 0.5 * spacing && p.y < half_ - 0.5 * spacing;
  float h; vec2 sh; vec2 sl; vec3 sq; vec3 n;
  vec2 slx = vec2(0.0); vec3 sqx = vec3(0.0); bool pixDone = false;
  // per-pixel normals: their detail is limited by the pixel footprint only (the mesh no
  // longer carries them), so no ring lod ramp and no contrast falloff tied to the rings
  float lodPix = clamp(max(log2(max(foot, 1.0)), minLod), 0.0, fftLevels - 1.0);
  float lost = max(levelVar(0.0) - levelVar(PIXN ? lodPix : lod), 0.0);
  bool thinDone = false;
  vec4 tL = vec4(0.0), tF = vec4(0.0), tG = vec4(0.0);
  float mt = tileThin * smoothstep(TILE_THIN_D0, TILE_THIN_D1, length(eye.xz - p));
  vec2 oddT = mod(floor((p + half_) / spacing + 0.5), 2.0);
  if (tile && meta.y == 0.0 && meta.z == 0.0 && mt > 0.0 && oddT.x + oddT.y > 0.5) {
    // both odd: the half-density quad's diagonal is b-c, (+1,-1) (a,b,c / b,d,c split)
    vec2 dv = (oddT.x > 0.5 && oddT.y > 0.5) ? vec2(spacing, -spacing) : oddT * spacing;
    float h0, h1, h2, e0, e1, e2; vec2 a0, a1, a2, x0, x1, x2; vec3 n0, n1, n2, q0, q1, q2;
    vec4 L0, L1, L2, F0, F1, F2, G0, G1, G2;
    tileVert(p, h0, a0, n0, L0, F0, G0, x0, q0, e0);
    tileVert(p - dv, h1, a1, n1, L1, F1, G1, x1, q1, e1);
    tileVert(p + dv, h2, a2, n2, L2, F2, G2, x2, q2, e2);
    h = mix(h0, 0.5 * (h1 + h2), mt); sh = mix(a0, 0.5 * (a1 + a2), mt);
    n = normalize(mix(n0, normalize(n1 + n2), mt));
    slx = mix(x0, 0.5 * (x1 + x2), mt); sqx = mix(q0, 0.5 * (q1 + q2), mt);
    // foam fields keep this vertex's (with the edge fade), only geometry and smooth fields morph
    float k0 = 1.0 - e0, k1 = 1.0 - e1, k2 = 1.0 - e2;
    tL = mix(L0 * k0, 0.5 * (L1 * k1 + L2 * k2), mt); tF = F0; tG = G0;
    pixDone = true; thinDone = true;
  } else if (tile && meta.y == 0.0 && meta.z == 0.0) {
    ivec2 cell = ivec2(floor((p + half_) / spacing + 0.5));
    cell = (cell % physN + physN) % physN;
    vec4 a = texelFetch(physA, cell, 0);
    vec4 b = texelFetch(physB, cell, 0);
    h = a.x;
    sh = a.yz / max(choppiness, 1e-6);
    n = b.xyz;
    if (lod > 0.0) {
      // take off the slope detail finer than lod (plain copy: the tile is copy 0)
      vec2 uv = (p + half_) / fftL + 0.5 / float(physN);
      vec2 s0 = vec2(textureLod(fftR1, uv, 0.0).z, textureLod(fftR2, uv, 0.0).x);
      vec2 sL = vec2(textureLod(fftR1, uv, lod).z, textureLod(fftR2, uv, lod).x);
      vec2 slope = -n.xz / max(n.y, 1e-3) - (s0 - sL);
      n = normalize(vec3(-slope.x, 1.0, -slope.y));
    }
    // GAD edge: over the last XFADE_M inside the tile edge the physics fields hand over to the
    // open-sea sampling the rings use, so there is no line where the tile meets the rings.
    if (edgeW > 0.0) {
      float h2; vec2 sh2, sl2; vec3 sq2;
      seaAtPix(p, lod, lodPix, h2, sh2, sl2, sq2, slx, sqx); pixDone = true;
      h = mix(h, h2, edgeW);
      sh = mix(sh, sh2, edgeW);
      if (!PIXN) n = normalize(mix(n, normalFrom(sl2, sq2), edgeW));
    }
  } else if (meta.y != 0.0 || meta.z != 0.0) {
    // odd vertex on a finer region's outer edge: the average of its two neighbours
    vec2 m = meta.yz;
    float h1, h2; vec2 a1, a2, s1, s2; vec3 q1, q2;
    seaAt(p - m, lod, h1, a1, s1, q1);
    seaAt(p + m, lod, h2, a2, s2, q2);
    h = 0.5 * (h1 + h2); sh = 0.5 * (a1 + a2); sl = 0.5 * (s1 + s2); sq = 0.5 * (q1 + q2);
    n = normalFrom(sl, sq);
  } else {
    seaAtPix(p, lod, lodPix, h, sh, sl, sq, slx, sqx); pixDone = true;
    // GAD ring edge: geomorph (CDLOD). Over the outer 40% of ring r, odd vertices blend to the
    // average of their coarse-grid neighbours, so by the edge where the spacing doubles the
    // ring already carries only the next ring's detail: no square line.
    float r = max(ceil(log2(max(c, 1.0) - 1e-4)), 1.0);
    float inner = exp2(r - 1.0);
    float mk = smoothstep(0.2, 1.0, (c - inner) / inner);
    if (mk > 0.0) {
      float sp = spacing * exp2(meta.x); // v28-lod: the region's own spacing (in meta.x)
      vec2 idx = floor((p + half_) / sp + 0.5);
      vec2 odd = mod(idx, 2.0);
      if (odd.x + odd.y > 0.5) {
        // both odd: the coarse cell's diagonal is b-c, (+1,-1) (buildFftMesh splits a,b,c / b,d,c)
        vec2 dv = (odd.x > 0.5 && odd.y > 0.5) ? vec2(sp, -sp) : vec2(odd.x, odd.y) * sp;
        float h1, h2; vec2 a1, a2, s1, s2; vec3 q1, q2;
        seaAt(p - dv, lod, h1, a1, s1, q1);
        seaAt(p + dv, lod, h2, a2, s2, q2);
        h = mix(h, 0.5 * (h1 + h2), mk); sh = mix(sh, 0.5 * (a1 + a2), mk);
        sl = mix(sl, 0.5 * (s1 + s2), mk); sq = mix(sq, 0.5 * (q1 + q2), mk);
      }
    }
    n = normalFrom(sl, sq);
  }
  vLodLost = lost;
  vBroken = 0.0; vLip = 0.0; vFoam = 0.0; vFlag = 0.0; vRecent = 0.0; vFade = 0.0;
  vAge = 0.0; vAhead = 0.0; vRun = 0.0; vThin = 0.0; vFoamKeep = 0.0;
  if (meta.w > 0.5 && thinDone) {
    // v28-lod morphed tile vertex: lips/thin morphed with the geometry (tL already faded),
    // foam fields of the vertex itself (the fragment reads foam per pixel anyway)
    float keep = 1.0 - edgeW;
    vThin = tL.z; vBroken = tL.x; vLip = tL.y;
    vAge = tF.z; vFade = tF.y * keep; vFoam = tF.x * keep; vAhead = tG.x; vRun = tG.y;
    vFlag = tG.w * keep; vRecent = tG.z * keep;
    vFoamKeep = keep;
  } else if (meta.w > 0.5) {
    ivec2 cell = ivec2(floor((p + half_) / spacing + 0.5));
    cell = (cell % physN + physN) % physN;
    vec4 l = texelFetch(physL, cell, 0);
    vec4 f = texelFetch(physF, cell, 0);
    vec4 g = texelFetch(physG, cell, 0);
    vThin = l.z; vAge = f.z; vFlag = g.w; vRecent = g.z; vFade = f.y;
    vAhead = g.x; vRun = g.y; vBroken = l.x; vLip = l.y; vFoam = f.x;
    // foam, lips and glow fade out over the same band, so where they stop shows no line
    float keep = 1.0 - edgeW;
    vFoamKeep = keep;
    vThin *= keep; vBroken *= keep; vLip *= keep; vFoam *= keep; vFade *= keep; vFlag *= keep; vRecent *= keep;
  }
  vNormal = n;
  { float hx = 0.0; vec2 shx = vec2(0.0);
    if (PIXN && !pixDone) explicitAt(p, lodPix, hx, shx, slx, sqx);
    vXs = vec4(slx, sqx.xy); vXq = sqx.z; }
  vSea = vec4(p, PIXN ? lodPix : lod, tile && meta.y == 0.0 && meta.z == 0.0 ? edgeW : 1.0);
  vWorld = vec3(p.x + choppiness * sh.x, h, p.y + choppiness * sh.y);
  if (meta.w < -0.5) {
    // skirt (see buildFftMesh): reaches outward under the coarser ring and a little down, so
    // the T-junction slivers are covered seen from above too (a vertical skirt has no area
    // there); elsewhere the coarser surface hides it.
    vec2 out_ = abs(p.x) >= abs(p.y) ? vec2(sign(p.x), 0.0) : vec2(0.0, sign(p.y));
    float rr = max(ceil(log2(max(c, 1.0) - 1e-4)), 0.0);
    vWorld.xz += out_ * 2.0 * spacing * exp2(rr + 1.0);
    vWorld.y -= 0.3 + 0.0005 * d;
  }
  gl_Position = viewProj * vec4(vWorld, 1.0);
}
`;

// The mesh: positions (x, z) and meta (ring lod, morph dx, morph dz, tile flag).
export function buildFftMesh(N, L, reach = FFT_RINGS_TO, farHalve = false) {
  const pos = [];
  const meta = [];
  const index = [];
  const s0 = L / N;
  // v28 perf: triangles are bucketed into square chunks (CHUNKS x CHUNKS per region) with
  // their xz bounds, so the view can skip chunks outside the frustum (no change to any
  // drawn pixel: a culled chunk has no triangle on screen).
  const CHUNKS = 16;
  const buckets = new Map();
  const put = (key, bx, bz, tri) => {
    let b = buckets.get(key);
    if (!b) { b = { idx: [], x0: Infinity, x1: -Infinity, z0: Infinity, z1: -Infinity, lod: 0 }; buckets.set(key, b); }
    for (const v of tri) { b.idx.push(v); const x = pos[v * 2], z = pos[v * 2 + 1]; if (x < b.x0) b.x0 = x; if (x > b.x1) b.x1 = x; if (z < b.z0) b.z0 = z; if (z > b.z1) b.z1 = z; }
    b.lod = Math.max(b.lod, bx);
  };
  // A square grid of (cells+1)^2 vertices at spacing s centred on 0, skipping the cells
  // inside |x|,|z| < inner; odd vertices on the outer edge morph to their neighbours when
  // a coarser ring follows.
  // v28-lod: meta.x = log2(own spacing / tile spacing); outerRatio = next region's spacing / s.
  function region(cells, s, inner, lod, outerMorph, tile, outerRatio = 2) {
    const h = (cells * s) / 2;
    const id = new Int32Array((cells + 1) * (cells + 1)).fill(-1);
    const inside = (i, j) => {
      const x0 = -h + i * s, z0 = -h + j * s;
      return x0 >= -inner - 1e-6 && x0 + s <= inner + 1e-6 && z0 >= -inner - 1e-6 && z0 + s <= inner + 1e-6;
    };
    const vert = (i, j) => {
      const k = j * (cells + 1) + i;
      if (id[k] >= 0) return id[k];
      const x = -h + i * s, z = -h + j * s;
      let mx = 0, mz = 0;
      const onOuter = i === 0 || j === 0 || i === cells || j === cells;
      const l = Math.log2(s / s0);
      if (outerMorph && onOuter) {
        const alongX = j === 0 || j === cells;
        const idx = alongX ? i : j;
        const rem = idx % outerRatio;
        if (rem !== 0) { const m = Math.min(rem, outerRatio - rem) * s; if (alongX) mx = m; else mz = m; }
      }
      id[k] = pos.length / 2;
      pos.push(x, z);
      meta.push(l, mx, mz, tile ? 1 : 0);
      return id[k];
    };
    for (let j = 0; j < cells; j += 1) for (let i = 0; i < cells; i += 1) {
      if (inner > 0 && inside(i, j)) continue;
      const a = vert(i, j), b = vert(i + 1, j), c = vert(i, j + 1), d = vert(i + 1, j + 1);
      const key = lod + ":" + Math.floor(i * CHUNKS / cells) + ":" + Math.floor(j * CHUNKS / cells);
      put(key, lod, 0, [a, b, c, b, d, c]);
    }
    // v28-lod tile thinning: each tile chunk also gets a half-density triangle list over its
    // even vertices (same vertex ids), drawn when the chunk lies wholly past TILE_THIN_D1.
    if (tile) for (let j = 0; j < cells; j += 2) for (let i = 0; i < cells; i += 2) {
      const a = vert(i, j), b = vert(i + 2, j), c = vert(i, j + 2), d = vert(i + 2, j + 2);
      const key = "T" + lod + ":" + Math.floor(i * CHUNKS / cells) + ":" + Math.floor(j * CHUNKS / cells);
      put(key, lod, 0, [a, b, c, b, d, c]);
    }
    // GAD ring edge: a skirt hangs below this region's outer edge so the T-junction where a
    // coarser ring starts can never open a one-pixel crack (sky dots along the square).
    if (outerMorph) {
      const skirt = (i, j) => { const k = vert(i, j) * 4; pos.push(pos[vert(i, j) * 2], pos[vert(i, j) * 2 + 1]); meta.push(meta[k], meta[k + 1], meta[k + 2], -1); return pos.length / 2 - 1; };
      const edge = (i0, j0, i1, j1) => { const a = vert(i0, j0), b = vert(i1, j1), a2 = skirt(i0, j0), b2 = skirt(i1, j1);
        const key = lod + ":" + Math.min(Math.floor(i0 * CHUNKS / cells), CHUNKS - 1) + ":" + Math.min(Math.floor(j0 * CHUNKS / cells), CHUNKS - 1);
        put(key, lod, 0, [a, a2, b, b, a2, b2]);
        if (tile) put("T" + key, lod, 0, [a, a2, b, b, a2, b2]); };
      for (let t = 0; t < cells; t += 1) { edge(t, 0, t + 1, 0); edge(t + 1, cells, t, cells); edge(0, t + 1, 0, t); edge(cells, t, cells, t + 1); }
    }
  }
  let half = L / 2;
  let rings = 0;
  while (half < reach) { half *= 2; rings += 1; }
  // v28-lod: every ring at half the v28 density (spacing 2^(r+1) s0, N/2 cells); with farHalve
  // the rings past 840 m (r >= 3) at a quarter (2^(r+2) s0, N/4 cells). The tile is unchanged.
  const dens = (r) => (r === 0 ? 1 : farHalve && r >= 3 ? 4 : 2);
  const sp = (r) => s0 * 2 ** r * dens(r);
  region(N, s0, 0, 0, rings > 0, true, sp(1) / sp(0));
  half = L / 2;
  for (let r = 1; r <= rings; r += 1) {
    region(N / dens(r), sp(r), half, r, r < rings, false, r < rings ? sp(r + 1) / sp(r) : 2);
    half *= 2;
  }
  const chunks = [];
  const byKey = new Map();
  let fullCount = 0;
  for (const [key, b] of buckets) {
    if (key[0] === "T") continue;
    const ch = { first: index.length, count: b.idx.length, x0: b.x0, x1: b.x1, z0: b.z0, z1: b.z1, spacing: s0 * 2 ** b.lod };
    chunks.push(ch); byKey.set(key, ch); for (const v of b.idx) index.push(v);
  }
  fullCount = index.length;
  // half-density lists of the tile chunks, after all full lists, in the same chunk order so
  // neighbouring thin chunks still merge into one draw range
  for (const [key, ch] of byKey) {
    const b = buckets.get("T" + key);
    if (b) { ch.thin = { first: index.length, count: b.idx.length }; for (const v of b.idx) index.push(v); }
  }
  return { xz: new Float32Array(pos), meta: new Float32Array(meta), index: new Uint32Array(index), rings, extent: half, chunks, fullCount };
}

// v28 perf: index ranges of the chunks that can reach the screen. A chunk is dropped only when
// its box (grown by margin m for the displacement, the skirt and the waves' height) lies wholly
// outside one clip plane. Adjacent visible chunks merge into one draw range.
export function visibleRanges(chunks, vp, m, eye = null, thinD = Infinity, f2b = false) {
  const out = [];
  const vis = [];
  for (const c of chunks) {
    const g = m + 4 * c.spacing;
    const xs = [c.x0 - g, c.x1 + g], zs = [c.z0 - g, c.z1 + g], ys = [-m, m];
    let outMask = 63;
    for (const x of xs) for (const y of ys) for (const z of zs) {
      const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
      const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
      const cz = vp[2] * x + vp[6] * y + vp[10] * z + vp[14];
      const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      outMask &= (cx < -cw ? 1 : 0) | (cx > cw ? 2 : 0) | (cy < -cw ? 4 : 0) | (cy > cw ? 8 : 0) | (cz < -cw ? 16 : 0) | (cz > cw ? 32 : 0);
    }
    if (outMask) continue;
    // v28-lod: a tile chunk wholly past thinD (horizontal, from the camera) draws its half-density list
    let first = c.first, count = c.count;
    if (c.thin && eye) {
      const dx = Math.max(c.x0 - eye[0], 0, eye[0] - c.x1), dz = Math.max(c.z0 - eye[2], 0, eye[2] - c.z1);
      if (Math.hypot(dx, dz) >= thinD) { first = c.thin.first; count = c.thin.count; }
    }
    if (f2b) {
      const dx = Math.max(c.x0 - eye[0], 0, eye[0] - c.x1), dz = Math.max(c.z0 - eye[2], 0, eye[2] - c.z1);
      vis.push([Math.hypot(dx, dz), first, count]);
      continue;
    }
    const last = out[out.length - 1];
    if (last && last[0] + last[1] === first) last[1] += count; else out.push([first, count]);
  }
  if (f2b) {
    // v28-lod: nearest chunks first, so the depth test rejects hidden water before it is shaded
    vis.sort((a, b) => a[0] - b[0]);
    for (const [, first, count] of vis) {
      const last = out[out.length - 1];
      if (last && last[0] + last[1] === first) last[1] += count; else out.push([first, count]);
    }
  }
  return out;
}

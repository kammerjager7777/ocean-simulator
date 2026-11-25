// Caustics and sun shadows for the underwater scene.
// Caustics: Evan Wallace. A coarse grid of the live surface is refracted along
// the clock sun and drawn into a small map; triangles that shrink are brighter.
// Shadows: the boat, the buoy and the reef cast into one depth map. Underwater
// the shadow fills back in with depth (scattered light).

const GRID = 256;
const CAUST = 1024;
// 2048 over the ~360 m light window is ~0.18 m a texel. At 512 (0.7 m) a
// hull was four texels wide and a rock's noon shadow, 0.2-0.4 m long, was
// smaller than one texel, so neither read on the sand.
const SHADOW = 2048;
// One draw. The grid is a little larger than the shallow window so a ray
// from just outside can still land inside it. 256² is the ceiling.
const SPAN = 1.12;
// One map across the tank. A 64 m tile sat on the dry crown and the bank,
// where the net has to read, only saw the wrapped edge.
const TILE = 280;
// One texture for both maps. The caustic block is unused (the net is
// procedural). Shadow depth is packed into RGB under it, because an 8-bit
// red channel over a kilometre of range cannot see a boat two metres off
// the sand. Two samplers would push the sea past the 16-unit Mac limit.
export const LIGHT_ATLAS = { w: Math.max(CAUST, SHADOW), h: CAUST + SHADOW, cau: CAUST, sh: SHADOW };

export function lightSampleGlsl() {
  const { w, h, cau, sh } = LIGHT_ATLAS;
  const yScale = (cau / h).toFixed(6);
  const xScale = (cau / w).toFixed(6);
  const xInset = (0.5 / w).toFixed(6);
  const yInset = (0.5 / h).toFixed(6);
  const yMax = ((cau - 12) / h).toFixed(6);
  const grad = (6 / w).toFixed(6);
  return `
uniform sampler2D uLight;
float cauEdge(vec2 uv) {
  return smoothstep(0.0, 0.22, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
}
float lightCau(vec2 uv) {
  vec2 f = clamp(uv, vec2(0.0), vec2(1.0));
  vec2 a = vec2(f.x * ${xScale}, f.y * ${yScale});
  a = clamp(a, vec2(${xInset}, ${yInset}), vec2(${((cau - 0.5) / w).toFixed(6)}, ${yMax}));
  vec2 gx = clamp(dFdx(a), vec2(-${grad}), vec2(${grad}));
  vec2 gy = clamp(dFdy(a), vec2(-${grad}), vec2(${grad}));
  return textureGrad(uLight, a, gx, gy).r;
}
float lightSh(vec2 uv) {
  vec2 f = clamp(uv, vec2(0.0), vec2(0.999));
  ivec2 p = ivec2(f * ${sh}.0);
  return texelFetch(uLight, ivec2(p.x, ${cau} + p.y), 0).r;
}
uniform float uCausT;
float causLine(vec2 p) {
  float s = abs(sin(p.x) * sin(p.y));
  return pow(1.0 - s, 7.0);
}
// Three abs-sine layers. Cells grow with depth. The live surface slope
// warps the domain, so the lines bend as the waves move. No texture.
float causNet(vec2 xz, float depthM, vec2 slope) {
  // Finest in ankle-to-waist-deep water, then the cells grow with depth.
  float d = max(depthM, 0.0);
  float cell = 0.22 + smoothstep(0.3, 3.0, d) * 0.38 + max(d - 1.2, 0.0) * 0.40;
  vec2 p = xz / cell;
  vec2 warp = slope * 6.0;
  float t = uCausT;
  vec2 a = p + warp;
  a += vec2(sin(a.y * 0.9 + t * 0.55), cos(a.x * 0.8 - t * 0.42)) * 0.5;
  float l0 = causLine(a + vec2(t * 0.22, -t * 0.16));
  vec2 b = p * 2.15 + warp * 1.8;
  b += vec2(sin(b.y * 0.65 - t * 0.48), cos(b.x * 0.6 + t * 0.36)) * 0.34;
  float l1 = causLine(b + vec2(-t * 0.14, t * 0.19));
  vec2 c = p * 0.48 + warp * 0.55;
  c += vec2(sin(c.y * 0.42 + t * 0.27), cos(c.x * 0.4 - t * 0.2)) * 0.6;
  float l2 = causLine(c + vec2(t * 0.1, t * 0.08));
  float net = l0 * 0.72 + l1 * 0.36 + l2 * 0.3;
  // Light on sand. The lines stay brighter than the cells, not white wires.
  return 0.84 + net * 0.8;
}
// v32 (C1, C4): the rainbow fringe. Water disperses sunlight (n ~1.331 red, 1.337 green, 1.343
// blue), so each colour's refracted ray meets the floor a little apart along the sun's refracted
// azimuth, by depth x (tan theta_red - tan theta_blue): ~8 mm per metre at a 45 deg sun. That real
// split is exaggerated (CAUS_DISP_GAIN) so it reads at 0.5-5 m, with a small floor so a high sun
// still shows a faint edge. It widens with depth and turns with the clock sun, as the shadows do.
#define CAUS_DISP_GAIN 5.0
#define CAUS_DISP_MIN 0.006
#define CAUS_CHROMA 0.8
float causTanRefr(float s, float n) { float q = s / n; return q / sqrt(max(1.0 - q * q, 1e-4)); }
// Half the red-blue split at the floor (m): red samples the surface at +d, blue at -d.
vec2 causDisp(vec3 sunDir, float depthM) {
  vec3 Ls = normalize(sunDir);
  float s = clamp(length(Ls.xz), 0.0, 1.0);          // sine of the sun's zenith angle (in air)
  vec2 az = s > 1e-3 ? Ls.xz / s : vec2(0.6, 0.8);  // toward the sun
  float half_ = max(0.5 * CAUS_DISP_GAIN * (causTanRefr(s, 1.331) - causTanRefr(s, 1.343)), CAUS_DISP_MIN);
  return az * half_ * max(depthM, 0.0);
}
// causNet per colour. Only the finest layer (l0, where the sharp lines and so the fringe live) is
// sampled three times; the two coarser layers are shared. Kept close to grey (CAUS_CHROMA):
// a Subnautica-like edge, not a prism.
float causL0(vec2 a, float t) {
  a += vec2(sin(a.y * 0.9 + t * 0.55), cos(a.x * 0.8 - t * 0.42)) * 0.5;
  return causLine(a + vec2(t * 0.22, -t * 0.16));
}
vec3 causNetRGB(vec2 xz, float depthM, vec2 slope, vec2 disp) {
  float d = max(depthM, 0.0);
  float cell = 0.22 + smoothstep(0.3, 3.0, d) * 0.38 + max(d - 1.2, 0.0) * 0.40;
  vec2 p = xz / cell;
  vec2 dp = disp / cell;
  vec2 warp = slope * 6.0;
  float t = uCausT;
  vec3 l0 = vec3(causL0(p + warp + dp, t), causL0(p + warp, t), causL0(p + warp - dp, t)); // red +disp, blue -disp
  vec2 b = p * 2.15 + warp * 1.8;
  b += vec2(sin(b.y * 0.65 - t * 0.48), cos(b.x * 0.6 + t * 0.36)) * 0.34;
  float l1 = causLine(b + vec2(-t * 0.14, t * 0.19));
  vec2 c2 = p * 0.48 + warp * 0.55;
  c2 += vec2(sin(c2.y * 0.42 + t * 0.27), cos(c2.x * 0.4 - t * 0.2)) * 0.6;
  float l2 = causLine(c2 + vec2(t * 0.1, t * 0.08));
  vec3 net = l0 * 0.72 + vec3(l1 * 0.36 + l2 * 0.3);
  net = mix(vec3(dot(net, vec3(1.0 / 3.0))), net, CAUS_CHROMA);
  return 0.84 + net * 0.8;
}
`;
}

export function shadowGlsl() {
  const { cau, sh } = LIGHT_ATLAS;
  return `
// Depth of one shadow texel. Clamped inside the shadow block, so a tap at the
// edge cannot read the caustic rows above it.
float shadowDepthAt(ivec2 ip) {
  ip = clamp(ip, ivec2(0), ivec2(${sh - 1}));
  vec3 e = texelFetch(uLight, ivec2(ip.x, ${cau} + ip.y), 0).rgb;
  return dot(e, vec3(1.0, 0.0039215686, 0.0000153787));
}
// Percentage-closer, by hand. The atlas is RGB-packed depth, so linear
// filtering cannot compare. Each of the four texels is a smooth 0..1 test (a
// thin ramp past the bias, not a hard step), then the four are blended with
// hermite weights. One unit of depth is the whole light range (about 1 km).
// The sea floor is not a caster, so the bias only covers depth precision:
// ~2 cm, ramp ~5 cm. Any occluder blocks fully, down to the low rim of a rock
// a few centimetres off the sand (a wider ramp let that rim leak light, and a
// rock's shadow read as a faint grey). The boat and the reef offset along
// their normals before the test, so they do not shadow themselves.
float shadowBilin(vec2 uv, float z) {
  vec2 g = uv * ${sh}.0 - 0.5;
  vec2 f = fract(g);
  f = f * f * (3.0 - 2.0 * f);
  ivec2 i0 = ivec2(floor(g));
  const float bias = 0.00002;
  const float soft = 0.00005;
  float a = smoothstep(bias, bias + soft, z - shadowDepthAt(i0));
  float b = smoothstep(bias, bias + soft, z - shadowDepthAt(i0 + ivec2(1, 0)));
  float c = smoothstep(bias, bias + soft, z - shadowDepthAt(i0 + ivec2(0, 1)));
  float d = smoothstep(bias, bias + soft, z - shadowDepthAt(i0 + ivec2(1, 1)));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// Blocker search tap, smooth in position: the four texels of the bilinear
// footprint each add their depth, weighted by the bilinear weight and by how
// surely they block. A raw texel test here switched the penumbra width at
// every texel edge and drew squares in the soft edge.
void shadowBlk(vec2 t, float z, inout float sumD, inout float sumW) {
  vec2 g = t - 0.5;
  vec2 f = fract(g);
  ivec2 i0 = ivec2(floor(g));
  float d0 = shadowDepthAt(i0), d1 = shadowDepthAt(i0 + ivec2(1, 0));
  float d2 = shadowDepthAt(i0 + ivec2(0, 1)), d3 = shadowDepthAt(i0 + ivec2(1, 1));
  float w0 = (1.0 - f.x) * (1.0 - f.y) * smoothstep(0.00002, 0.00007, z - d0);
  float w1 = f.x * (1.0 - f.y) * smoothstep(0.00002, 0.00007, z - d1);
  float w2 = (1.0 - f.x) * f.y * smoothstep(0.00002, 0.00007, z - d2);
  float w3 = f.x * f.y * smoothstep(0.00002, 0.00007, z - d3);
  sumD += w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3;
  sumW += w0 + w1 + w2 + w3;
}
// Light position and the in-water fill. Returns false off the map.
bool shadowSetup(vec3 w, out vec3 p, out float fill) {
  vec4 lp = uShVP * vec4(w, 1.0);
  p = lp.xyz / max(lp.w, 1e-4) * 0.5 + 0.5;
  fill = clamp(1.0 - exp(-max(0.0, -w.y) * 0.08), 0.0, 0.6);
  return !(p.x < 0.02 || p.x > 0.98 || p.y < 0.02 || p.y > 0.98);
}
// Sea floor. Under water the sun arrives as a forward-scattered cone, not a
// point, so the penumbra grows with the gap between occluder and sand
// (percentage-closer soft shadows). A rock half a metre tall gets a soft
// contact shadow round its foot even at noon, when its hard shadow is only
// 0.2 m long and mostly under it; a hull 2-3 m up keeps a dark core and a
// ~0.7 m soft edge.
float shadowFactor(vec3 w) {
  if (uCausDay < 0.01) return 1.0;
  vec3 p; float fill;
  if (!shadowSetup(w, p, fill)) return 1.0;
  // Metres per shadow texel and per unit of stored depth, read off the
  // orthographic light matrix, so they follow the sea size.
  float texM = 2.0 / (length(vec3(uShVP[0][0], uShVP[1][0], uShVP[2][0])) * ${sh}.0);
  float zM = 2.0 / length(vec3(uShVP[0][2], uShVP[1][2], uShVP[2][2]));
  vec2 tc = p.xy * ${sh}.0;
  float sr = clamp(0.5 / texM, 1.5, 6.0);
  float sumD = 0.0, sumW = 0.0;
  shadowBlk(tc, p.z, sumD, sumW);
  shadowBlk(tc + vec2(sr, 0.3 * sr), p.z, sumD, sumW);
  shadowBlk(tc + vec2(-0.76 * sr, 0.7 * sr), p.z, sumD, sumW);
  shadowBlk(tc + vec2(-0.24 * sr, -sr), p.z, sumD, sumW);
  float distM = sumW > 1e-4 ? max(p.z - sumD / sumW, 0.0) * zM : 0.0;
  distM *= smoothstep(0.0, 0.3, sumW);
  float penM = clamp(0.06 + distM * 0.8, 0.06, 0.7);
  float rad = max(penM / texM, 1.0);
  float blocked = 0.0;
  for (int i = 0; i < 8; i++) {
    float r = sqrt((float(i) + 0.5) / 8.0) * rad;
    float a = float(i) * 2.3999632 + 0.5;
    blocked += shadowBilin((tc + vec2(cos(a), sin(a)) * r) / ${sh}.0, p.z);
  }
  // The cone is about half blocked at a rock's foot, so a linear average
  // read as a pale grey. Square the lit part: full light and full shadow
  // stay put, the contact penumbra deepens.
  float lit = 1.0 - blocked * 0.125;
  return mix(lit * lit, 1.0, fill);
}
// Four filtered taps, fixed ~1.5 texel spread. For the water surface, the
// boat and the reef heads, where a wide penumbra is not needed.
float shadowFactorFast(vec3 w) {
  if (uCausDay < 0.01) return 1.0;
  vec3 p; float fill;
  if (!shadowSetup(w, p, fill)) return 1.0;
  float rad = 1.5 / ${sh}.0;
  float blocked = 0.25 * (
      shadowBilin(p.xy + vec2(-0.8, -0.3) * rad, p.z)
    + shadowBilin(p.xy + vec2(0.3, -0.8) * rad, p.z)
    + shadowBilin(p.xy + vec2(0.8, 0.3) * rad, p.z)
    + shadowBilin(p.xy + vec2(-0.3, 0.8) * rad, p.z));
  return mix(1.0 - blocked, 1.0, fill);
}
`;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
function lookAt(eye, at, up) {
  const z = norm(sub(eye, at));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  // Column-major: the rows of the rotation are x, y, z. Written as columns,
  // the rotation was transposed, so the map looked along the wrong axis and
  // the boat's shadow never fell under the boat.
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}
function ortho(l, r, b, t, n, f) {
  return new Float32Array([
    2 / (r - l), 0, 0, 0,
    0, 2 / (t - b), 0, 0,
    0, 0, -2 / (f - n), 0,
    -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1,
  ]);
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) for (let r = 0; r < 4; r += 1) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return o;
}

const CAUST_VS = `#version 300 es
layout(location=0) in vec2 ij;
uniform sampler2D physA;
uniform sampler2D physB;
uniform float uN;
uniform float uL;
uniform vec3 uSun;
uniform float uDepth;
uniform float uSpan;
uniform vec2 uShift;
uniform vec2 uWinC;
uniform float uWin;
uniform float uHasIsland;
uniform vec2 uIslandC;
uniform float uIRb, uIRf, uIRs, uIRbank, uIEase;
uniform float uIPeak, uIYb, uIYf, uIYs, uIFace, uIShelf;
uniform float uBedS;
uniform vec2 uRipDir;
uniform float uBedRip;
out vec3 vOld;
out vec3 vNew;
float iHermite(float s, float y0, float y1, float m0, float m1) {
  float s2 = s * s, s3 = s2 * s;
  return (2.0 * s3 - 3.0 * s2 + 1.0) * y0 + (s3 - 2.0 * s2 + s) * m0
       + (-2.0 * s3 + 3.0 * s2) * y1 + (s3 - s2) * m1;
}
float islandBed(vec2 xz) {
  vec2 rel = xz - uIslandC;
  float th = atan(rel.y, rel.x);
  float lump = 0.055 * sin(3.0 * th) + 0.032 * sin(5.0 * th + 1.7) + 0.018 * sin(8.0 * th + 0.4);
  float r = length(rel) / max(1.0 + lump, 0.75);
  float yFloor = -max(uDepth, 0.5);
  if (r >= uIRbank) return yFloor;
  float ease = uIEase > 0.0 ? uIEase : 8.0;
  float join0 = uIRf - ease, join1 = uIRf + ease, y;
  if (r <= uIRb) y = iHermite(r / max(uIRb, 1.0), uIPeak, uIYb, 0.0, -uIFace * uIRb);
  else if (r <= join0) y = uIYb - uIFace * (r - uIRb);
  else if (r < join1) {
    float y0 = uIYb - uIFace * (join0 - uIRb);
    float y1 = uIYf - uIShelf * (join1 - uIRf);
    float span = max(join1 - join0, 1.0);
    y = iHermite((r - join0) / span, y0, y1, -uIFace * span, -uIShelf * span);
  } else if (r <= uIRs) y = uIYf - uIShelf * (r - uIRf);
  else {
    float span = max(uIRbank - uIRs, 1.0);
    y = iHermite((r - uIRs) / span, uIYs, yFloor, -uIShelf * span, 0.0);
  }
  return clamp(y, yFloor, -0.05);
}
float bedRelief(vec2 xz, float D) {
  if (uBedS < 1.0) return 0.0;
  float S = uBedS, amp = min(0.22 * D, 5.0);
  float r = amp * 0.45 * (0.6 * xz.x - 0.8 * xz.y) / S;
  float a = 0.0105 * (xz.x * 0.83 + xz.y * 0.56);
  float b = 0.0172 * (xz.x * -0.41 + xz.y * 0.91);
  r += amp * 0.32 * sin(6.2831853 * a + 0.7 * sin(6.2831853 * 0.004 * xz.y));
  r += amp * 0.14 * sin(6.2831853 * b + 1.3);
  float along = dot(xz, uRipDir);
  r += uBedRip * min(0.1, 0.02 * D) * sin(6.2831853 * along / 2.4);
  return min(r, D - 0.3);
}
float bedAt(vec2 xz) {
  float y = -uDepth + bedRelief(xz, uDepth);
  if (uHasIsland > 0.5) y = max(y, islandBed(xz));
  return min(y, -0.12);
}
vec3 projectBed(vec3 origin, vec3 dir) {
  float bedY = -uDepth;
  vec3 hit = origin;
  for (int i = 0; i < 2; i++) {
    float t = (bedY - origin.y) / min(dir.y, -0.05);
    hit = origin + dir * max(t, 0.0);
    bedY = bedAt(hit.xz);
  }
  float t = (bedY - origin.y) / min(dir.y, -0.05);
  return origin + dir * max(t, 0.0);
}
void main() {
  float span = uWin * uSpan;
  vec2 p = (ij / float(${GRID - 1})) * span - 0.5 * span + uWinC + uShift;
  vec2 sea = mod(p + 0.5 * uL, uL) - 0.5 * uL;
  vec2 uv = fract((sea + 0.5 * uL) / uL);
  ivec2 cell = ivec2(uv * uN);
  vec4 a = texelFetch(physA, cell, 0);
  vec3 n = texelFetch(physB, cell, 0).xyz;
  if (n.y < 0.2) n = vec3(0.0, 1.0, 0.0);
  n = normalize(n);
  vec3 sun = normalize(uSun);
  vec3 flatRay = refract(-sun, vec3(0.0, 1.0, 0.0), 1.0 / 1.333);
  vec3 ray = refract(-sun, n, 1.0 / 1.333);
  if (flatRay.y > -0.02) flatRay = vec3(0.0, -1.0, 0.0);
  if (ray.y > -0.02) ray = vec3(0.0, -1.0, 0.0);
  vec3 flatPos = vec3(p.x, 0.0, p.y);
  vec3 dispPos = vec3(p.x, a.x, p.y);
  vOld = projectBed(flatPos, flatRay);
  vNew = projectBed(dispPos, ray);
  vec2 tex = vNew.xz - vNew.y * flatRay.xz / flatRay.y;
  vec2 hitUv = (tex - (uWinC - 0.5 * uWin)) / max(uWin, 1.0);
  gl_Position = vec4(hitUv * 2.0 - 1.0, 0.0, 1.0);
}`;
export const CAUST_FS = `#version 300 es
precision highp float;
in vec3 vOld;
in vec3 vNew;
uniform float uL;
out vec4 o;
void main() {
  vec3 dX = dFdx(vNew), dY = dFdy(vNew);
  // A triangle that stretches across the window is a fold, not a filament.
  if (max(length(dX), length(dY)) > 6.0) discard;
  float oldArea = length(dFdx(vOld)) * length(dFdy(vOld));
  float newArea = max(length(dX) * length(dY), 1e-4);
  float f = min(oldArea / newArea, 8.0);
  // Only a real focus is written. An ordinary triangle stays 0, so it cannot
  // paint a dark block or a tone change at the edge of the window.
  o = vec4(smoothstep(2.0, 4.8, f), 0.0, 0.0, 1.0);
}`;

const SH_VS = `#version 300 es
layout(location=0) in vec3 pos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(pos, 1.0); }`;
export const SH_FS = `#version 300 es
precision highp float;
out vec4 o;
void main() {
  vec3 enc = fract(vec3(1.0, 255.0, 65025.0) * gl_FragCoord.z);
  enc -= enc.yzz * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
  o = vec4(enc, 1.0);
}`;

const REEF_SHADOW_VS = `#version 300 es
layout(location=0) in vec3 pos;
layout(location=1) in vec4 inst0;
layout(location=2) in vec4 inst1;
layout(location=3) in vec4 inst2;
uniform mat4 uMVP;
uniform float depth;
void main() {
  float r = inst0.z;
  float h = min(1.8 * inst0.w, 0.8 * depth);
  float aspect = inst1.z;
  float c = cos(inst1.w), s = sin(inst1.w);
  vec2 l = vec2(pos.x * aspect * r, pos.z * r);
  l = vec2(c * l.x - s * l.y, s * l.x + c * l.y);
  float y = min(inst2.x + pos.y * h + 0.02, -0.04);
  gl_Position = uMVP * vec4(inst0.x + l.x, y, inst0.y + l.y, 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) || "light shader");
  return sh;
}
function link(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || "light link");
  return p;
}

export function createLightMaps(gl) {
  const caustProg = link(gl, CAUST_VS, CAUST_FS);
  const shProg = link(gl, SH_VS, SH_FS);
  const reefProg = link(gl, REEF_SHADOW_VS, SH_FS);
  const ij = new Float32Array(GRID * GRID * 2);
  const idx = [];
  for (let j = 0; j < GRID; j += 1) for (let i = 0; i < GRID; i += 1) {
    ij[(j * GRID + i) * 2] = i;
    ij[(j * GRID + i) * 2 + 1] = j;
  }
  for (let j = 0; j < GRID - 1; j += 1) for (let i = 0; i < GRID - 1; i += 1) {
    const a = j * GRID + i;
    idx.push(a, a + 1, a + GRID, a + 1, a + GRID + 1, a + GRID);
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, ij, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  const indexCount = idx.length;

  function atlasTarget() {
    const { w, h } = LIGHT_ATLAS;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb };
  }
  const atlas = atlasTarget();
  function plate() {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, CAUST, CAUST, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb };
  }
  const fresh = plate();
  const history = plate();
  const compProg = link(gl, `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`, `#version 300 es
precision highp float;
uniform sampler2D uNew;
uniform sampler2D uOld;
uniform float uNewW;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(${CAUST}.0);
  float n = texture(uNew, uv).r;
  float p = texture(uOld, uv).r;
  float g = mix(p, n, uNewW);
  o = vec4(g, 0.0, 0.0, 1.0);
}`);
  const compNew = gl.getUniformLocation(compProg, "uNew");
  const compOld = gl.getUniformLocation(compProg, "uOld");
  const compW = gl.getUniformLocation(compProg, "uNewW");
  const compVao = gl.createVertexArray();
  gl.bindFramebuffer(gl.FRAMEBUFFER, history.fb);
  gl.viewport(0, 0, CAUST, CAUST);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const loc = {
    ca: { a: gl.getUniformLocation(caustProg, "physA"), b: gl.getUniformLocation(caustProg, "physB"), n: gl.getUniformLocation(caustProg, "uN"), L: gl.getUniformLocation(caustProg, "uL"), sun: gl.getUniformLocation(caustProg, "uSun"), depth: gl.getUniformLocation(caustProg, "uDepth"), span: gl.getUniformLocation(caustProg, "uSpan"), shift: gl.getUniformLocation(caustProg, "uShift"), win: gl.getUniformLocation(caustProg, "uWin"), winC: gl.getUniformLocation(caustProg, "uWinC"), island: gl.getUniformLocation(caustProg, "uHasIsland"), iC: gl.getUniformLocation(caustProg, "uIslandC"), bedS: gl.getUniformLocation(caustProg, "uBedS"), rip: gl.getUniformLocation(caustProg, "uRipDir"), bedRip: gl.getUniformLocation(caustProg, "uBedRip") },
    sh: { mvp: gl.getUniformLocation(shProg, "uMVP") },
    reef: { mvp: gl.getUniformLocation(reefProg, "uMVP"), depth: gl.getUniformLocation(reefProg, "depth") },
  };
  const info = { causMs: 0, shMs: 0, causGpu: 0, costFrames: 0, deep: 1, day: 1, sun: [0, 1, 0], vp: new Float32Array(16), seaL: 250, origin: [0, 0], tile: TILE };
  const timer = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  let pendingQuery = null;

  function sunLit(sun) {
    // The clock sun, not the art-directed light. The art vector sits near
    // 25° even at 11:57 and was stretching every rock into a long smear.
    const tod = globalThis.__oceanTodMin;
    let dir = sun;
    if (Number.isFinite(tod)) {
      const e = Math.asin(Math.cos(20 * Math.PI / 180) * Math.cos(((tod / 60 - 12) * 15) * Math.PI / 180));
      const a = -Math.PI / 2 - (tod - 720) * Math.PI / 720;
      dir = [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
    }
    const elev = dir ? dir[1] : 0;
    const on = elev > 0.02 ? 1 : 0;
    const day = on ? Math.min(1, Math.max(0, (elev - 0.02) / 0.28)) : 0;
    return { dir: dir || [0, 1, 0], day };
  }

  function lightVP(dir, L) {
    const s = norm(dir);
    const eye = [s[0] * L, Math.max(0.2, s[1]) * L, s[2] * L];
    const view = lookAt(eye, [0, -8, 0], Math.abs(s[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0]);
    const r = L * 0.72;
    const proj = ortho(-r, r, -r, r, 1, L * 4);
    return mul(proj, view);
  }

  function update(o) {
    const { physA, physB, physN, L, depth, boatView, buoyView, boat, buoy, clusters } = o;
    const lit = sunLit(o.sun);
    info.day = lit.day;
    info.bisect = o.bisect || 0;
    info.sun = lit.dir;
    info.seaL = L;
    info.t = o.t || 0;
    info.vp = lightVP(lit.dir, L);
    // The map is the shallow bank, not the whole FFT tile. Half a bank
    // radius past the shelf still covers the water that is clear enough
    // to show a net. Open sea uses a smaller window on the view centre.
    const foot = o.foot;
    const cx = foot ? foot.cx : 0;
    const cz = foot ? foot.cz : 0;
    const half = foot ? Math.min(L * 0.5, Math.max(40, foot.half)) : L * 0.5;
    info.origin = [cx - half, cz - half];
    info.tile = half * 2;
    info.cx = cx;
    info.cz = cz;
    const fbo0 = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const vpRaw = gl.getParameter(gl.VIEWPORT);
    const vx = vpRaw[0], vy = vpRaw[1], vw = vpRaw[2], vh = vpRaw[3];
    const scRaw = gl.getParameter(gl.SCISSOR_BOX);
    const sx = scRaw[0], sy = scRaw[1], sw = scRaw[2], sh = scRaw[3];
    const prog0 = gl.getParameter(gl.CURRENT_PROGRAM);
    const vao0 = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
    const blend = gl.isEnabled(gl.BLEND);
    const scissor = gl.isEnabled(gl.SCISSOR_TEST);
    if (timer && pendingQuery) {
      const disjoint = gl.getParameter(timer.GPU_DISJOINT_EXT);
      if (gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT_AVAILABLE) && !disjoint) {
        info.causGpu = gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT) / 1e6;
      }
      gl.deleteQuery(pendingQuery);
      pendingQuery = null;
    }
    const gpuQuery = timer ? gl.createQuery() : null;
    if (gpuQuery) gl.beginQuery(timer.TIME_ELAPSED_EXT, gpuQuery);
    // The cost-measure flush is a synchronous readback: only with ?gldebug=1 or globalThis.__glDebug = true.
    const measuring = info.costFrames < 2 && (globalThis.__glDebug === true || (typeof location !== "undefined" && /[?&]gldebug=1/.test(location.search)));
    if (measuring) {
      const flush = new Uint8Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, flush);
    }
    const t0 = performance.now();
    // Procedural caustic: the pattern is computed in the floor shader.
    // This pass does not draw a caustic grid and does not take a sampler.
    info.causMs = 0;
    info.causGpu = 0;
    info.costFrames = Math.min(2, info.costFrames + 1);
    if (gpuQuery) { gl.endQuery(timer.TIME_ELAPSED_EXT); gl.deleteQuery(gpuQuery); }
    const t1 = performance.now();
    gl.bindFramebuffer(gl.FRAMEBUFFER, atlas.fb);
    gl.viewport(0, CAUST, SHADOW, SHADOW);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(0, CAUST, SHADOW, SHADOW);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(1, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (lit.day > 0.01 && !(info.bisect > 2.5)) {
      gl.useProgram(shProg);
      if (boatView && boat && boat.matrix) boatView.castShadow(shProg, loc.sh.mvp, mul(info.vp, boat.matrix()));
      if (buoyView && buoy && buoy.matrix) buoyView.castShadow(shProg, loc.sh.mvp, mul(info.vp, buoy.matrix()));
      if (clusters && clusters.count > 0 && clusters.parts) {
        gl.useProgram(reefProg);
        gl.uniformMatrix4fv(loc.reef.mvp, false, info.vp);
        gl.uniform1f(loc.reef.depth, depth);
        for (const part of Object.values(clusters.parts)) {
          if (!part.count) continue;
          gl.bindVertexArray(part.vao);
          gl.drawElementsInstanced(gl.TRIANGLES, part.indexCount, gl.UNSIGNED_SHORT, 0, part.count);
        }
        gl.bindVertexArray(null);
      }
    }
    const t2 = performance.now();
    if (!(info.causGpu > 0)) info.causMs = t1 - t0;
    info.shMs = t2 - t1;
    info.deep = 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo0);
    gl.viewport(vx, vy, vw, vh);
    gl.scissor(sx, sy, sw, sh);
    gl.bindVertexArray(vao0);
    gl.useProgram(prog0);
    if (blend) gl.enable(gl.BLEND); else gl.disable(gl.BLEND);
    if (scissor) gl.enable(gl.SCISSOR_TEST); else gl.disable(gl.SCISSOR_TEST);
    gl.activeTexture(gl.TEXTURE0);
    globalThis.__lightCost = { causMs: info.causMs, shMs: info.shMs, causGpu: info.causGpu, deep: info.deep, day: info.day, grid: 0, span: 0, caust: 0, tris: 0, tile: 0, mode: "procedural" };
    return info;
  }

  function bindSea(program) {
    const u = (n) => gl.getUniformLocation(program, n);
    gl.activeTexture(gl.TEXTURE20);
    gl.bindTexture(gl.TEXTURE_2D, atlas.tex);
    gl.uniform1i(u("uLight"), 20);
    gl.uniformMatrix4fv(u("uShVP"), false, info.vp);
    gl.uniform1f(u("uCausDay"), info.bisect > 0.5 && info.bisect < 1.5 ? 0.0 : info.day);
    gl.uniform1f(u("uCausT"), info.t || 0);
    gl.uniform1f(u("uBisect"), info.bisect || 0);
    gl.uniform1f(u("uCausDeep"), info.deep);
    gl.uniform1f(u("uSeaL"), info.seaL);
    gl.uniform2f(u("uCaustO"), info.origin[0], info.origin[1]);
    gl.uniform1f(u("uCaustTile"), info.tile);
    gl.activeTexture(gl.TEXTURE0);
  }

  return { update, bindSea, info, tex: atlas.tex, caust: atlas.tex, shadow: atlas.tex };
}

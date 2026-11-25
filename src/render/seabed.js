// v30.23 seabed relief (Michael, Nov 1): gentle slopes, dunes, ripple marks and a few rocks instead of a
// flat bed. One formula, in JS (buoy readings, cross-section, depth gauge) and GLSL (walls, floor rim,
// the wall sand band), so every view agrees. The depth setting stays the MEAN depth: the slope and the
// dunes are zero-mean sinusoids/linear terms over the block; rocks add a small positive bias (< 1% of D).
// Relief scales with depth (amp = min(0.22 D, 5 m)) so a 2 m sea never gets a bed above the water.
const ROCKS = [[0.31, -0.22, 1.0], [-0.27, 0.18, 0.8], [0.08, 0.37, 0.6], [-0.38, -0.31, 0.9], [0.42, 0.29, 0.7]]; // (x/S, z/S, height share)
export const BED_ON = typeof location === "undefined" || new URLSearchParams(location.search).get("bed") !== "flat";
// Lagged wave direction. The 95 m and 58 m dunes stay put (they are the bathymetry
// the reef is seated on). The ~2 m megaripple turns with the sea.
let ripDir = [0, 1];
let ripAmp = 1;
export function setBedRippleDir(x, z, amp = ripAmp) {
  const l = Math.hypot(x, z) || 1;
  ripDir = [x / l, z / l];
  ripAmp = Math.max(0, Math.min(1, Number.isFinite(amp) ? amp : 1));
}
export function bedRelief(x, z, S, D) {
  if (!BED_ON) return 0;
  const amp = Math.min(0.22 * D, 5);
  const u = x / S, v = z / S;
  let r = amp * 0.45 * (0.6 * u - 0.8 * v);                                   // a gentle slope across the block
  const a = 0.0105 * (x * 0.83 + z * 0.56), b = 0.0172 * (x * -0.41 + z * 0.91);
  r += amp * 0.32 * Math.sin(2 * Math.PI * a + 0.7 * Math.sin(2 * Math.PI * 0.004 * z)); // dunes (~95 m crests, meandering)
  r += amp * 0.14 * Math.sin(2 * Math.PI * b + 1.3);                           // a cross set of smaller dunes (~58 m)
  const along = x * ripDir[0] + z * ripDir[1];
  // The 2.4 m bedform is cut by the sea. A calm sea has not left a bullseye of it.
  r += ripAmp * Math.min(0.1, 0.02 * D) * Math.sin(2 * Math.PI * along / 2.4);
  for (const [rx, rz, h] of ROCKS) { const dx = x - rx * S, dz = z - rz * S, w = 0.035 * S; r += amp * 0.55 * h * Math.exp(-(dx * dx + dz * dz) / (w * w)); }
  return Math.min(r, D - 0.3);
}
export const bedY = (x, z, S, D) => -D + bedRelief(x, z, S, D);
// GLSL: float bedRelief(vec2 xz) with uniforms uBedS (block size) and depth (already declared by callers).
export const BED_GLSL = `
uniform float uBedS;
uniform vec2 uRipDir;
uniform float uBedRip;
float bedRelief(vec2 xz, float D) {
  if (${BED_ON ? "false" : "true"} || uBedS < 1.0) return 0.0;
  float S = uBedS, amp = min(0.22 * D, 5.0);
  vec2 uv = xz / S;
  float r = amp * 0.45 * (0.6 * uv.x - 0.8 * uv.y);
  float a = 0.0105 * (xz.x * 0.83 + xz.y * 0.56), b = 0.0172 * (xz.x * -0.41 + xz.y * 0.91);
  r += amp * 0.32 * sin(6.2831853 * a + 0.7 * sin(6.2831853 * 0.004 * xz.y));
  r += amp * 0.14 * sin(6.2831853 * b + 1.3);
  float along = xz.x * uRipDir.x + xz.y * uRipDir.y;
  r += uBedRip * min(0.1, 0.02 * D) * sin(6.2831853 * along / 2.4);
${ROCKS.map(([rx, rz, h]) => `  { vec2 d = xz - vec2(${rx.toFixed(3)}, ${rz.toFixed(3)}) * S; float w = 0.035 * S; r += amp * ${(0.55 * h).toFixed(4)} * exp(-dot(d, d) / (w * w)); }`).join("\n")}
  return min(r, D - 0.3);
}
`;

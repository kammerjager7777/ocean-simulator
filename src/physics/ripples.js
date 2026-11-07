// v30.23 ripple engine (Michael: a REAL ripple, not a decal). Each source is a dispersive deep-water
// wave packet: a ring of crests that spreads at the group speed cg = cp/2 while its crests run through it
// at the phase speed, widens as it goes (dispersion) and fades (viscous + geometric 1/sqrt(r) spreading).
// The height is added to the drawn surface (vertex height + per-pixel normal) and to the boat's and the
// buoy's height samples, so they bob on it. It is NOT part of the physics state, so the break test and
// the foam never see it (foam delta 0). Sim time drives it: Pause freezes the rings.
const G = 9.81, MAX = 16;
const P = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
// v30.27: 2 cm sat inside the wind sea at every zoom. 0.22 m at size 250, growing with
// wavelength so the steepness stays put, capped so a close zoom is a splash and not a wall.
const A_BASE = Number(P.get("ripA") ?? 0.22);
const A_CAP = Number(P.get("ripCap") ?? 0.62);
const LAM = Number(P.get("ripL") ?? 3.0);           // carrier wavelength at size 250 (m); scaled with size
const TAU = 6.0;                                    // e-fold life (s)
const src = [];
let lam = LAM, k0 = 2 * Math.PI / LAM, ampK = 1;
export function setRippleScale(size) { ampK = Math.max(1, size / 250); lam = LAM * ampK; k0 = 2 * Math.PI / lam; }
export function addRipple(x, z, t, amp = 1) {
  const a = Math.min(A_CAP, Math.min(1, Math.max(0, amp)) * A_BASE * ampK);
  src.push({ x, z, t0: t, a }); if (src.length > MAX) src.shift();
}
export function rippleCount() { return src.length; }
function packet(r, age, a) {
  const cp = Math.sqrt(G / k0), cg = 0.5 * cp, w = 0.6 * lam + 0.35 * cg * age;
  const rc = cg * age, env = Math.exp(-((r - rc) * (r - rc)) / (w * w)) * Math.exp(-age / TAU) * Math.sqrt(lam / (lam + r)) * Math.min(1, 0.55 + age / 0.08);
  return [a * env, k0 * (r - cp * age)];
}
export function rippleHeight(x, z, t) {
  let h = 0; for (const s of src) { const age = t - s.t0; if (age < 0 || age > 5 * TAU) continue; const r = Math.hypot(x - s.x, z - s.z); const [e, ph] = packet(r, age, s.a); h += e * Math.cos(ph); }
  return h;
}
export function prune(t) { for (let i = src.length - 1; i >= 0; i--) if (t - src[i].t0 > 5 * TAU) src.splice(i, 1); }
// uniforms: uRip[16] = (x, z, age, amp), uRipP = (k0, lam, TAU, 0)
const pack = new Float32Array(MAX * 4);
const splashPts = new Float32Array(64 * 4), splashN = new Float32Array(64 * 2);
// A short ring of foam on the packet, gone within about 1.5 s. Reuses the buoy trail shader.
export function publishSplash(t) {
  let n = 0;
  const cp = Math.sqrt(G / k0), cg = 0.5 * cp, w = Math.min(3.2, Math.max(0.8, lam * 0.28));
  for (const s of src) {
    const age = t - s.t0;
    if (!(age >= 0) || age > 1.5 || n > 52) continue;
    const R = Math.max(1.1, cg * age);
    const steps = 12;
    for (let i = 0; i < steps && n < 60; i++) {
      const a = (i / steps) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
      splashPts[n * 4] = s.x + c * R; splashPts[n * 4 + 1] = s.z + sn * R;
      splashPts[n * 4 + 2] = age + 0.18; splashPts[n * 4 + 3] = w * (0.85 + 0.15 * s.a / A_CAP);
      splashN[n * 2] = c; splashN[n * 2 + 1] = sn; n++;
    }
  }
  globalThis.__oceanRippleSplash = n ? { pts: splashPts, nrm: splashN, n } : null;
  return n;
}
export function rippleRings(t) {
  const cp = Math.sqrt(G / k0), cg = 0.5 * cp, out = [];
  for (const s of src) {
    const age = t - s.t0;
    if (age < 0 || age > 2.4) continue;
    out.push({ x: s.x, z: s.z, r: Math.max(1.3, cg * age), age, a: Math.max(0, 1 - age / 2.4) });
  }
  return out;
}
export function rippleUniforms(t) {
  pack.fill(0); src.forEach((s, i) => { pack[i * 4] = s.x; pack[i * 4 + 1] = s.z; pack[i * 4 + 2] = t - s.t0; pack[i * 4 + 3] = s.a; });
  return { pack, p: [k0, lam, TAU, src.length] };
}
export const RIPPLE_GLSL = `
uniform vec4 uRip[16];
uniform vec4 uRipP;
// returns (height, d/dx, d/dz)
vec3 ripple(vec2 xz) {
  vec3 o = vec3(0.0);
  if (uRipP.w < 0.5) return o;
  float k0 = uRipP.x, lam = uRipP.y, cp = sqrt(9.81 / k0), cg = 0.5 * cp;
  for (int i = 0; i < 16; i += 1) {
    vec4 s = uRip[i]; if (s.w <= 0.0 || s.z < 0.0) continue;
    vec2 d = xz - s.xy; float r = length(d); float age = s.z;
    float w = 0.6 * lam + 0.35 * cg * age, rc = cg * age;
    float env = s.w * exp(-((r - rc) * (r - rc)) / (w * w)) * exp(-age / uRipP.z) * sqrt(lam / (lam + r)) * min(1.0, 0.55 + age / 0.08);
    if (env < 1e-5) continue;
    float ph = k0 * (r - cp * age);
    vec2 dir = r > 1e-3 ? d / r : vec2(0.0);
    o.x += env * cos(ph);
    o.yz += dir * (-env * k0 * sin(ph) - env * 2.0 * (r - rc) / (w * w) * cos(ph));
  }
  return o;
}
`;

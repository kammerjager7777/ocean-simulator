// Pierson–Moskowitz wind spectrum. S(ω) is variance density in m² / (rad/s).
import { peakScale } from "./steepness.js";

export function peakAngularFrequency(windSpeed, gravity) {
  const u = Math.max(windSpeed, 0.4);
  return (0.877 * gravity) / u;
}

export function piersonMoskowitz(omega, omegaPeak, gravity) {
  const alpha = 0.0081;
  const beta = 1.25;
  const w = Math.max(omega, 0.05);
  const ratio = omegaPeak / w;
  return ((alpha * gravity * gravity) / w ** 5) * Math.exp(-beta * ratio ** 4);
}

// Finite-depth dispersion: ω² = g k tanh(k h), solved by Newton on f(k) = g k tanh(kh) − ω²,
// f'(k) = g (tanh(kh) + kh sech²(kh)), sech² = 1 − tanh². Eckart's seed k∞ / √tanh(k∞ h),
// k∞ = ω²/g, then three steps reach float precision at any depth. WAVENUMBER_GLSL below
// is the same method for the shaders; keep the two in step.
export function wavenumber(omega, depth, gravity) {
  if (!(omega > 0) || !(gravity > 0)) return 0;
  const w2 = omega * omega;
  const kInf = w2 / gravity;
  if (!(depth > 0)) return kInf;
  let k = kInf / Math.sqrt(Math.max(Math.tanh(kInf * depth), 1e-30));
  for (let i = 0; i < 3; i += 1) {
    const kh = k * depth;
    const t = Math.tanh(kh);
    const df = Math.max(gravity * (t + kh * (1 - t * t)), 1e-30);
    k = Math.max(k - (gravity * k * t - w2) / df, 0);
  }
  return k;
}
// The same Newton solve as wavenumber() above, in GLSL. Float only and no extensions, so
// WebGL1 shaders can include it too; tanh is written with exp because GLSL ES 1.00 has none.
export const WAVENUMBER_GLSL = `
float wnTanh(float x) { return 1.0 - 2.0 / (exp(2.0 * min(x, 40.0)) + 1.0); }
float waveNumber(float omega, float h, float gravity) {
  if (omega <= 0.0 || gravity <= 0.0) return 0.0;
  float w2 = omega * omega;
  float kInf = w2 / gravity;
  if (h <= 0.0) return kInf;
  float k = kInf / sqrt(max(wnTanh(kInf * h), 1e-30));
  for (int i = 0; i < 3; i++) {
    float kh = k * h;
    float t = wnTanh(kh);
    float df = max(gravity * (t + kh * (1.0 - t * t)), 1e-30);
    k = max(k - (gravity * k * t - w2) / df, 0.0);
  }
  return k;
}
`;

// Elfouhaily et al. 1997 (J. Geophys. Res. 102, 15781), the unified directional spectrum
// fitted to measured long and short wind waves, including Cox and Munk's slopes. Used
// here as an omnidirectional spectrum; each band keeps its own direction spread.
// Curvature spectrum B(k) = k^3 S(k) = Bl + Bh:
//   long-wave part  Bl = 1/2 alpha_p (c_p / c) F_p, F_p = L_PM J_p exp(-Omega/sqrt(10) (sqrt(k/k_p) - 1))
//   short-wave part Bh = 1/2 alpha_m (c_m / c) F_m, F_m = L_PM J_p exp(-1/4 (k/k_m - 1)^2)
// Standard parameters: fully developed Omega = U10/c_p = 0.84, k_p = g Omega^2 / U10^2,
// alpha_p = 6e-3 sqrt(Omega), gamma = 1.7, sigma = 0.08 (1 + 4 Omega^-3), k_m = 370 rad/m,
// c_m = 0.23 m/s, alpha_m = 0.01 (1 + ln(u*/c_m)) below c_m, 0.01 (1 + 3 ln(u*/c_m)) above,
// u* from the paper's roughness length z0 = 3.7e-5 (U10^2/g) (U10/c_p)^0.9.
// c(k) = sqrt(g/k (1 + (k/k_m)^2)) is the paper's own phase speed, capillarity included,
// because the fit is defined with it. It only matters near k_m (1.7 cm waves); at our
// shortest modelled wave (0.25 m) it differs from gravity-only by 0.2%. The waves
// themselves still move by gravity and finite-depth dispersion (see chopPatch.js).
export const ELFOUHAILY = { omega: 0.84, km: 370, cm: 0.23 };

export function elfouhailyParams(windSpeed, gravity) {
  const U = Math.max(windSpeed, 0.4);
  const { omega: Om, km, cm } = ELFOUHAILY;
  const kp = (gravity * Om * Om) / (U * U);
  const cp = Math.sqrt((gravity / kp) * (1 + (kp / km) ** 2));
  const alphaP = 6e-3 * Math.sqrt(Om);
  const gamma = Om <= 1 ? 1.7 : 1.7 + 6 * Math.log10(Om);
  const sigma = 0.08 * (1 + 4 * Om ** -3);
  const z0 = 3.7e-5 * ((U * U) / gravity) * (U / cp) ** 0.9;
  const uStar = (0.41 * U) / Math.log(10 / z0);
  const alphaM = uStar < cm ? 0.01 * (1 + Math.log(uStar / cm)) : 0.01 * (1 + 3 * Math.log(uStar / cm));
  return { U, Om, km, cm, kp, cp, alphaP, gamma, sigma, uStar, alphaM, gravity };
}

// The two curvature parts at wavenumber k (deep water).
export function elfouhailyParts(k, p) {
  const c = Math.sqrt((p.gravity / k) * (1 + (k / p.km) ** 2));
  const r = Math.sqrt(k / p.kp);
  const lpm = Math.exp(-1.25 * (p.kp / k) ** 2);
  const jp = p.gamma ** Math.exp(-((r - 1) ** 2) / (2 * p.sigma * p.sigma));
  const bl = 0.5 * p.alphaP * (p.cp / c) * lpm * jp * Math.exp((-p.Om / Math.sqrt(10)) * (r - 1));
  const bh = 0.5 * p.alphaM * (p.cm / c) * lpm * jp * Math.exp(-0.25 * (k / p.km - 1) ** 2);
  return { bl, bh };
}

// Frequency spectrum S(omega) in m^2/(rad/s), like piersonMoskowitz: the spectrum is taken
// as defined in deep water (k0 = omega^2/g, dk0/domega = 2 omega/g), and each band's
// shoaling turns it into finite depth exactly as it does for PM. longScale multiplies
// the long-wave part only (see spectrumFor).
export function elfouhaily(omega, p, longScale = 1) {
  const w = Math.max(omega, 0.05);
  const k = (w * w) / p.gravity;
  const { bl, bh } = elfouhailyParts(k, p);
  return ((longScale * bl + bh) / (k * k * k)) * ((2 * w) / p.gravity);
}

// A band's spectrum for one wind sample, as a function of omega. spec.spectrum picks the
// model ("pm" by default). For Elfouhaily, spec.matchBelow = "pm" sets the level of the
// long-wave part from Pierson-Moskowitz at the band's own lower edge (omegaMin x the PM
// peak), so the band hands over from a PM band below without a step in energy. That
// scale is one constant for the whole band (no splice inside the band) and the short-wave
// part keeps its standard level. The band reads only its own spec and wind sample.
function swellEnergy(spec, age) {
  if (spec.id !== "swell" || !Number.isFinite(age)) return 1;
  const g = 0.12 + 0.88 * Math.min(1, Math.max(0, age));
  return g * g;
}

export function spectrumFor(spec, windSpeed, gravity, age) {
  // Age omitted: the historical fully developed peak (self-checks).
  const omegaPeak = peakAngularFrequency(windSpeed, gravity) * peakScale(age);
  const swell = swellEnergy(spec, age);
  if (spec.spectrum !== "elfouhaily") {
    return { model: "pm", omegaPeak, longScale: swell, density: (w) => swell * piersonMoskowitz(w, omegaPeak, gravity) };
  }
  const p = elfouhailyParams(windSpeed, gravity);
  let longScale = 1;
  if (spec.matchBelow === "pm") {
    const wb = spec.omegaMin * omegaPeak;
    const kb = (wb * wb) / gravity;
    const { bl, bh } = elfouhailyParts(kb, p);
    const want = (piersonMoskowitz(wb, omegaPeak, gravity) * (gravity / (2 * wb))) * kb * kb * kb;
    longScale = Math.max(0, (want - bh) / bl);
  }
  return { model: "elfouhaily", omegaPeak, params: p, longScale, density: (w) => swell * elfouhaily(w, p, longScale) };
}

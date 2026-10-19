// Pierson–Moskowitz wind spectrum. S(ω) is variance density in m² / (rad/s).

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

// Finite-depth dispersion: ω² = g k tanh(k h). Iterate k from the deep-water guess.
export function wavenumber(omega, depth, gravity) {
  const h = Math.max(depth, 0.5);
  let k = (omega * omega) / gravity;
  for (let i = 0; i < 6; i += 1) {
    const tanhKh = Math.tanh(k * h);
    k = (omega * omega) / (gravity * Math.max(tanhKh, 1e-4));
  }
  return k;
}

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
export function spectrumFor(spec, windSpeed, gravity) {
  const omegaPeak = peakAngularFrequency(windSpeed, gravity);
  if (spec.spectrum !== "elfouhaily") {
    return { model: "pm", omegaPeak, density: (w) => piersonMoskowitz(w, omegaPeak, gravity) };
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
  return { model: "elfouhaily", omegaPeak, params: p, longScale, density: (w) => elfouhaily(w, p, longScale) };
}

import { piersonMoskowitz, spectrumFor, wavenumber } from "./spectrum.js?v=28";

function unitPhase(n) {
  const x = Math.sin(n * 127.1) * 43758.5453;
  return (x - Math.floor(x)) * Math.PI * 2;
}

export function createBand(band, index) {
  const components = [];
  const nFreq = band.frequencies;
  const nDir = band.directions;
  let n = index * 100;
  for (let i = 0; i < nFreq; i += 1) {
    const u = nFreq === 1 ? 0.5 : i / (nFreq - 1);
    const frac = band.omegaMin + (band.omegaMax - band.omegaMin) * u;
    for (let d = 0; d < nDir; d += 1) {
      const tv = nDir === 1 ? 0 : d / (nDir - 1) - 0.5;
      components.push({
        frac,
        dirOffset: tv * band.spread,
        phase: unitPhase(n),
        omega: 0,
        k: 0,
        amplitude: 0,
        breakAmplitude: 0,
        direction: 0,
      });
      n += 1;
    }
  }
  return { spec: band, components };
}


export function shoal(amplitude, omega, k, depth, gravity) {
  const kh = k * Math.max(depth, 0.5);
  if (kh > 2.5) return amplitude;
  const sinh2 = Math.sinh(Math.min(2 * kh, 20));
  const group = 0.5 * (1 + (2 * kh) / Math.max(sinh2, 1e-4));
  const cg = (omega / Math.max(k, 1e-4)) * group;
  const kDeep = (omega * omega) / gravity;
  const cgDeep = omega / (2 * Math.max(kDeep, 1e-4));
  const gain = Math.sqrt(Math.max(cgDeep, 1e-4) / Math.max(cg, 1e-4));
  return amplitude * Math.min(gain, 2.5);
}

// Retune this band's slice from the wind it was given. Does not read other bands.
export function retuneBand(state, windSample, depth, gravity, dt, spacing) {
  const { spec, components } = state;
  // The band's own spectrum model (scenario data: spec.spectrum), for its own wind sample.
  const spectrum = spectrumFor(spec, windSample.speed, gravity);
  const omegaPeak = spectrum.omegaPeak;
  const dFrac = (spec.omegaMax - spec.omegaMin) / spec.frequencies;
  const dOmega = Math.max(dFrac * omegaPeak, 1e-4);
  const calm = windSample.speed < 0.05;
  // Two cells cannot hold a wave. Shorter than that aliases into fake steepness.
  const maxK = (2 * Math.PI) / Math.max((spacing || 1) * 2.4, 0.5);
  for (const component of components) {
    const omega = component.frac * omegaPeak;
    component.omega = omega;
    component.k = wavenumber(omega, depth, gravity);
    component.direction = windSample.direction + component.dirOffset;
    component.phase -= omega * Math.max(dt, 0);
    if (calm || component.k > maxK) {
      component.amplitude = 0;
      component.breakAmplitude = 0;
      continue;
    }
    const density = spectrum.density(omega);
    const variance = Math.max(0, 2 * density * dOmega / spec.directions);
    component.amplitude = shoal(Math.sqrt(variance), omega, component.k, depth, gravity);
    // What the break test sees of this wave (scenario data: spec.breakTest). By default the
    // wave as drawn. "pm": the same wave at its Pierson-Moskowitz level, whichever spectrum
    // is drawn, so the spectrum switch never changes breaking. false: left out.
    if (spec.breakTest === false) component.breakAmplitude = 0;
    else if (spec.breakTest === "pm" && spectrum.model !== "pm") {
      const pmVariance = Math.max(0, 2 * piersonMoskowitz(omega, omegaPeak, gravity) * dOmega / spec.directions);
      component.breakAmplitude = shoal(Math.sqrt(pmVariance), omega, component.k, depth, gravity);
    } else component.breakAmplitude = component.amplitude;
  }
  // What this band holds on the big grid, for its own fine patch (if it has one): the
  // components stand for consecutive slices of width dOmega from omegaMin up, so the
  // grid holds [omegaMin, omegaMin + kept * dFrac] x peak. Everything above belongs to
  // the patch. Nothing here changes the components themselves.
  let kept = 0;
  let keptOmegaMax = 0;
  let keptKMax = 0;
  for (const component of components) {
    if (component.amplitude === 0) continue;
    keptOmegaMax = Math.max(keptOmegaMax, component.omega);
    keptKMax = Math.max(keptKMax, component.k);
  }
  for (let i = 0; i < spec.frequencies; i += 1) {
    if (components.slice(i * spec.directions, (i + 1) * spec.directions).some((c) => c.amplitude !== 0)) kept += 1;
  }
  state.omegaPeak = omegaPeak;
  state.spectrum = spectrum;
  state.heldOmegaEdge = (spec.omegaMin + kept * dFrac) * omegaPeak;
  state.keptOmegaMax = keptOmegaMax;
  state.keptKMax = keptKMax;
}

export function sampleBand(state, x, z) {
  let height = 0;
  let dhdx = 0;
  let dhdz = 0;
  let shiftX = 0;
  let shiftZ = 0;
  let squeezeX = 0;
  let squeezeZ = 0;
  let squeezeXZ = 0;
  let breakHeight = 0;
  let breakDx = 0;
  let breakDz = 0;
  for (const component of state.components) {
    if (component.amplitude === 0) continue;
    const cx = Math.cos(component.direction);
    const cz = Math.sin(component.direction);
    const phase = component.k * (x * cx + z * cz) + component.phase;
    const s = Math.sin(phase);
    const c = Math.cos(phase);
    const amplitude = component.amplitude;
    const slopeScale = amplitude * component.k;
    height += amplitude * s;
    dhdx += slopeScale * cx * c;
    dhdz += slopeScale * cz * c;
    // Same wave, a quarter turn out of phase. Crests bunch sideways instead of only rising.
    shiftX += amplitude * cx * c;
    shiftZ += amplitude * cz * c;
    squeezeX += slopeScale * cx * cx * s;
    squeezeZ += slopeScale * cz * cz * s;
    squeezeXZ += slopeScale * cx * cz * s;
    const ba = component.breakAmplitude;
    breakHeight += ba * s;
    breakDx += ba * component.k * cx * c;
    breakDz += ba * component.k * cz * c;
  }
  return { height, dhdx, dhdz, shiftX, shiftZ, squeezeX, squeezeZ, squeezeXZ, breakHeight, breakDx, breakDz };
}

export function bandVariance(state) {
  let variance = 0;
  for (const component of state.components) {
    variance += (component.amplitude * component.amplitude) / 2;
  }
  return variance;
}

// Wind-sea steepness from the local wind, with swell underneath.
// Age does not turn breaking off. It sets the size: a young sea is short,
// an old sea is a large fully developed wind-sea, and it carries more swell.
// S is the wind-sea steepness, 0..1. The chop display is S * 2.5 unless overridden.
// 12 m/s, no current, 40 m → chop about 1.4. A light wind stays a low swell.

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// 1 = the fully developed peak for this wind. A young sea peaks higher
// (shorter waves). Callers that omit age keep the historical peak.
export function peakScale(age) {
  if (!Number.isFinite(age)) return 1;
  return 1 + 0.7 * (1 - clamp(age, 0, 1));
}

export function seaSteepness({
  windMs = 0,
  age = 0.28,
  currentMs = 0,
  currentDir = 0,
  windDir = Math.PI / 2,
  depthM = 40,
} = {}) {
  const ageT = clamp(age, 0, 1);
  // Local wind sets the wind-sea steepness. Storm wind stays near 1.
  const windK = clamp((Math.max(windMs, 0) / 18) ** 1.4, 0, 1);
  const follow = currentMs * Math.cos(currentDir - windDir);
  // Following eases steepness a little. Opposing raises it a little.
  const curK = clamp(1 - follow / 8, 0.85, 1.15);
  const shoalK = depthM >= 28 ? 1 : clamp((28 / Math.max(depthM, 2)) ** 0.3, 1, 1.6);
  const windSea = clamp(windK * curK * shoalK, 0, 1);
  const swell = ageT;
  const S = windSea;
  const chop = clamp(windSea * 2.5, 0, 2.5);
  let kind = "Building sea";
  if (ageT < 0.4) kind = "Young wind-sea";
  else if (ageT >= 0.72 && windMs < 8) kind = "Old swell";
  else if (ageT >= 0.72) kind = "Wind over swell";
  return { chop, S, windSea, swell, follow, age: ageT, windMs, depthM, curK, shoalK, kind };
}

export function whySteepness(st, override = false) {
  if (override) return "overridden — click to follow again";
  const n = st.chop.toFixed(1);
  const old = st.age >= 0.72;
  const young = st.age < 0.4;
  const storm = st.windMs >= 18;
  // The storm line is already as long as the box allows, so current stays on its own control.
  if (storm && old) return `${n} — storm over old swell, breaking`;
  let core;
  if (old && st.windMs < 8) core = `${n} — old swell, light wind`;
  else if (old) core = `${n} — wind over old swell`;
  else if (young) core = `${n} — young wind-sea`;
  else core = `${n} — building sea`;
  if (st.follow <= -0.35) core += ", against";
  else if (st.follow >= 0.35) core += ", with current";
  if (st.shoalK >= 1.12) core += ", shoaling";
  return core;
}

// 1 at the reference steepness. Near 0 for an old swell, so open-ocean foam dies back.
export function foamGain(S) {
  const x = Math.max(0, Number.isFinite(S) ? S : 0.56) / 0.56;
  if (x < 0.45) return x * x * 0.35;
  return Math.min(2.2, x ** 1.2);
}

// Multiplier on the break threshold ak. Higher means fewer breaks.
// Unset S keeps the historical threshold (self-checks that never set it).
export function breakScale(S) {
  if (!Number.isFinite(S)) return 1;
  return clamp(0.56 / Math.max(S, 0.04), 0.55, 6);
}

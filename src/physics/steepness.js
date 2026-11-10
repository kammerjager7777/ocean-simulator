// Open-ocean steepness from the sea, not from the chop slider.
// S is 0..1. The chop display is S * 2.5 unless the user overrides it.
// Reference (scale 1 on the old break threshold): 12 m/s, a young sea,
// no current, 40 m depth → chop about 1.4.

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function seaSteepness({
  windMs = 0,
  age = 0.28,
  currentMs = 0,
  currentDir = 0,
  windDir = Math.PI / 2,
  depthM = 40,
} = {}) {
  const ageT = clamp(age, 0, 1);
  const windK = clamp(windMs / 17, 0, 1.45);
  // Young and building seas are steep. Old swell that has travelled is smooth.
  const ageK = 1.55 * (1 - ageT) + 0.38 * ageT;
  const follow = currentMs * Math.cos(currentDir - windDir);
  const curK = clamp(1 - follow / 3, 0.62, 1.45);
  const shoalK = depthM >= 28 ? 1 : clamp((28 / Math.max(depthM, 2)) ** 0.3, 1, 1.6);
  const chop = clamp(2.5 * windK * (ageK / 1.55) * curK * shoalK, 0, 2.5);
  const S = chop / 2.5;
  let kind = "Building sea";
  if (ageT < 0.4) kind = "Young wind-sea";
  else if (ageT >= 0.72) kind = "Old swell";
  return { chop, S, follow, age: ageT, windMs, depthM, curK, shoalK, kind };
}

export function whySteepness(st, override = false) {
  if (override) return "overridden — click to follow again";
  const chop = st.chop.toFixed(1);
  if (st.age >= 0.72) return `${chop} — old swell`;
  const kn = Math.round(st.windMs * 1.94384);
  const hours = Math.max(1, Math.round(2 + st.age * 12));
  const name = st.age < 0.4 ? "young sea" : "building sea";
  const parts = [`${chop} — ${name}`, `${hours} h at ${kn} kn`];
  if (st.follow <= -0.35) parts.push("opposing current");
  else if (st.follow >= 0.35) parts.push("following current");
  if (st.shoalK >= 1.12) parts.push("shoaling");
  return parts.join(", ");
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

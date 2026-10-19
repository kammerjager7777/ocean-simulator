import { scenario } from "./scenario.js";
import { createWind } from "./wind.js";
import { createSurface, NO_FOAM_AHEAD, crestReachAt, foamWindowAt } from "./surface.js";
import { breakFieldCheck } from "./parity.js";
import { createBand, retuneBand } from "./band.js?v=28";
import { createChopPatch } from "./chopPatch.js?v=28";
import { piersonMoskowitz, spectrumFor, elfouhailyParts, wavenumber } from "./spectrum.js?v=28";
import { coxMunkMss } from "./roughness.js?v=28";
import { GRAVITY } from "./gravity.js";

function run(speed, depth, seconds) {
  const wind = createWind({ direction: 0.4, speed, gust: "continuous" });
  const surface = createSurface(scenario, wind);
  let state = null;
  const steps = 12;
  const dt = seconds / steps;
  for (let i = 0; i < steps; i += 1) state = surface.step(i * dt, dt, depth);
  let maxAbs = 0;
  for (const h of state.heights) maxAbs = Math.max(maxAbs, Math.abs(h));
  let mismatches = 0;
  const n = state.n;
  const spacing = scenario.size / Math.max(n - 1, 1);
  // v28 foam fix: every break flag follows the total-height steepness rule (surface.js
  // breakRuleAt), recomputed here from the heights (float ties / |ak - threshold| <= 1e-3 allowed).
  if (state.breakRule && state.breakRule.mode > 0.5) mismatches = breakFieldCheck(state.heights, state.breaks, n, state.breakRule, spacing, { h: state.xHeights, sx: state.xSlopeX, sz: state.xSlopeZ }).offEdge;
  // This step's foam (fresh) is tied to break flags: every fresh foam sample is within
  // four cells of a break, and none sits upwave of every break near it (behind the crest).
  // Lingering foam from earlier steps is checked separately below.
  const breakCells = [];
  for (let i = 0; i < state.sampleCount; i += 1) if (state.breaks[i] > 0) breakCells.push([i % n, Math.floor(i / n)]);
  let strayFoam = 0;
  let behindFoam = 0;
  // Michael 20:49: a break paints faceSteps (0.1 x peak wavelength) down its face; joins as before (4).
  const reachF = Math.max(4, state.faceSteps || 3);
  let faceFoam = 0;
  let faceSlope = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    if (!(state.freshFoam[i] > 0)) continue;
    const ix = i % n;
    const iz = Math.floor(i / n);
    let nearest = Infinity;
    let notBehind = false;
    for (const [bx, bz] of breakCells) {
      const d = Math.hypot(ix - bx, iz - bz);
      if (d < nearest) nearest = d;
      if (d <= reachF && (ix - bx) * state.travelX + (iz - bz) * state.travelZ >= -0.5) notBehind = true;
    }
    if (nearest > reachF + 1e-6) strayFoam += 1;
    if (!notBehind) behindFoam += 1;
    if (state.breaks[i] > 0) continue;
    faceFoam += 1;
    faceSlope += state.slopeX[i] * state.travelX + state.slopeZ[i] * state.travelZ;
  }
  const meanFaceSlope = faceFoam ? faceSlope / faceFoam : 0;
  // Foam is graded: 1 on every break, never above 1, and lower down the front face.
  // Each foam sample sits at or ahead of the crest it was painted from (allowing a
  // little for the sideways chop shift), and samples without foam hold the far value.
  let behindCrest = 0;
  let runStray = 0;
  let maxAhead = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    if (state.foam[i] > 0) {
      if (state.ahead[i] < -0.5) behindCrest += 1;
      maxAhead = Math.max(maxAhead, state.ahead[i]);
    } else if (state.ahead[i] !== NO_FOAM_AHEAD) behindCrest += 1;
    if (!(state.foam[i] > 0) && state.runs[i] !== 0) runStray += 1;
    if (state.foam[i] > 0 && !(state.runs[i] >= 0 && state.runs[i] <= 1)) runStray += 1;
  }
  let breakNotFull = 0;
  let overOne = 0;
  let graded = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    if (state.breaks[i] > 0 && state.foam[i] !== 1) breakNotFull += 1;
    if (state.foam[i] > 1) overOne += 1;
    if (state.foam[i] > 0 && state.foam[i] < 1) graded += 1;
  }
  // This step's foam is exactly the current flags. What is drawn is the current flags
  // plus recent ones: current flags ⊆ foam ⊆ current ∪ recent flags.
  let flagged = 0;
  let foamNoFlag = 0;
  let flagNoFoam = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    if (state.flags[i] > 0) flagged += 1;
    if (state.freshFoam[i] > 0 && !(state.flags[i] > 0)) foamNoFlag += 1;
    if (state.flags[i] > 0 && !(state.freshFoam[i] > 0)) flagNoFoam += 1;
  }
  let maxShift = 0;
  const shifts = state.shifts;
  for (let i = 0; i < shifts.length; i += 2) {
    maxShift = Math.max(maxShift, Math.hypot(shifts[i], shifts[i + 1]));
  }
  // Lingering foam: only inside the window, current flags at full strength, never
  // brighter than when it was laid, and the sets nest as described above.
  // Foam carried from earlier steps is weaker than foam flagged now, on average.
  let freshSum = 0;
  let freshN = 0;
  let agedSum = 0;
  let agedN = 0;
  let ageStray = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    if (state.flags[i] > 0) {
      freshSum += state.foam[i];
      freshN += 1;
      if (state.foamAge[i] !== 0) ageStray += 1;
    } else if (state.foam[i] > 0) {
      agedSum += state.foam[i];
      agedN += 1;
      if (!(state.foamAge[i] > 0 && state.foamAge[i] < 1)) ageStray += 1;
    } else if (state.foamAge[i] !== 0) ageStray += 1;
  }
  const freshMean = freshN ? freshSum / freshN : 0;
  const agedMean = agedN ? agedSum / agedN : 0;
  let pastWindow = 0;
  let currentNotFull = 0;
  let brighter = 0;
  let currentMissing = 0;
  let outsideFlags = 0;
  let bothFlags = 0;
  for (let i = 0; i < state.sampleCount; i += 1) {
    const f = state.foam[i];
    if (f > 0 && !(state.ages[i] < state.foamWindow)) pastWindow += 1;
    if (state.flags[i] > 0 && f !== state.freshFoam[i]) currentNotFull += 1;
    if (f > state.foamGrade[i] + 1e-7 || f > 1) brighter += 1;
    if (state.flags[i] > 0 && !(f > 0)) currentMissing += 1;
    if (f > 0 && !(state.flags[i] > 0 || state.recent[i] > 0)) outsideFlags += 1;
    if (state.flags[i] > 0 && state.recent[i] > 0) bothFlags += 1;
  }
  return { freshMean, agedMean, agedN, ageStray, pastWindow, currentNotFull, brighter, currentMissing, outsideFlags, bothFlags, maxAbs, mismatches, maxShift, strayFoam, behindFoam, faceFoam, meanFaceSlope, flagged, foamNoFlag, flagNoFoam, breakNotFull, overOne, graded, behindCrest, maxAhead, runStray, state };
}

// Which way do the waves really move? Step the sea a little and compare how the
// height changed with the slope along the travel direction the surface reports.
// A wave moving toward +travel rises where the surface drops ahead (its front face).
// Also check that foam beyond the break (front face) is rising more often than the
// same distance behind the break (back face).
function motion(speed, direction) {
  const wind = createWind({ direction, speed, gust: "continuous" });
  const surface = createSurface(scenario, wind);
  let state = null;
  for (let i = 0; i < 12; i += 1) state = surface.step(i / 3, 1 / 3, 40);
  const n = state.n;
  const tx = state.travelX;
  const tz = state.travelZ;
  const h0 = Float32Array.from(state.heights);
  const gx = Float32Array.from(state.slopeX);
  const gz = Float32Array.from(state.slopeZ);
  const foam = Float32Array.from(state.freshFoam);
  const breaks = Float32Array.from(state.breaks);
  const travelError = Math.acos(Math.max(-1, Math.min(1, tx * Math.cos(direction) + tz * Math.sin(direction))));
  const next = surface.step(4, 0.05, 40).heights;
  let cross = 0;
  let a = 0;
  let b = 0;
  for (let i = 0; i < n * n; i += 1) {
    const dh = next[i] - h0[i];
    const ahead = -(gx[i] * tx + gz[i] * tz);
    cross += dh * ahead;
    a += dh * dh;
    b += ahead * ahead;
  }
  let frontUp = 0;
  let front = 0;
  let backUp = 0;
  let back = 0;
  for (let i = 0; i < n * n; i += 1) {
    if (foam[i] > 0 && breaks[i] === 0) {
      front += 1;
      if (next[i] > h0[i]) frontUp += 1;
    }
    if (breaks[i] === 0) continue;
    const ix = i % n;
    const iz = Math.floor(i / n);
    for (let s = 1; s <= 3; s += 1) {
      const x = ix - Math.round(tx * s);
      const z = iz - Math.round(tz * s);
      if (x < 0 || z < 0 || x >= n || z >= n) break;
      back += 1;
      if (next[z * n + x] > h0[z * n + x]) backUp += 1;
    }
  }
  return {
    travelError,
    correlation: cross / Math.sqrt(a * b || 1),
    frontRising: front ? frontUp / front : 0,
    backRising: back ? backUp / back : 0,
  };
}

const calm = run(0, 40, 4);
const live = run(12, 40, 4);
const shallow = run(12, 3, 4);
const problems = [];
if (calm.maxAbs > 0.05) problems.push(`calm sea moved (${calm.maxAbs.toFixed(3)} m)`);
if (live.maxAbs < 0.3) problems.push(`wind did not raise the sea (${live.maxAbs.toFixed(3)} m)`);
if (calm.maxShift > 0.02) problems.push(`calm sea shifted sideways (${calm.maxShift.toFixed(3)} m)`);
if (live.maxShift < 0.2) problems.push(`wind did not lean the crests (${live.maxShift.toFixed(3)} m)`);
const sum = (state) => state.lips.reduce((a, v) => a + v, 0);
if (sum(calm.state) > 0.01) problems.push("calm sea glowed");
if (sum(live.state) < 1) problems.push("crests did not glow");
if (calm.state.breakingCount !== 0) problems.push("calm sea flagged breaks");
if (calm.state.foamCount !== 0) problems.push("calm sea drew foam");
if (calm.state.recentCount !== 0) problems.push("calm sea has recent flags");

// Stop all breaking after a strong sea: foam must fade without moving or growing and
// be gone within the fade window plus one step.
function stopBreaking() {
  const sc = { ...scenario };
  const wind = createWind({ direction: 0.4, speed: 18, gust: "continuous" });
  const surface = createSurface(sc, wind);
  let state = null;
  for (let i = 0; i < 12; i += 1) state = surface.step(i / 3, 1 / 3, 3);
  const startFoam = state.foamCount;
  const startSet = new Uint8Array(state.sampleCount);
  for (let i = 0; i < state.sampleCount; i += 1) startSet[i] = state.foam[i] > 0 ? 1 : 0;
  sc.breakSlope = Infinity;
  sc.depthBreakRatio = Infinity;
  const dt = 0.25;
  const window = foamWindowAt(sc, 18);
  const trace = [];
  let grew = 0;
  let moved = 0;
  let breaksAfter = 0;
  let prev = Float32Array.from(state.foam);
  const steps = Math.ceil(window / dt) + 1;
  for (let k = 1; k <= steps; k += 1) {
    state = surface.step(4 + k * dt, dt, 3);
    breaksAfter += state.breakingCount;
    let total = 0;
    for (let i = 0; i < state.sampleCount; i += 1) {
      const f = state.foam[i];
      total += f;
      if (f > prev[i] + 1e-7) grew += 1;
      if (f > 0 && !startSet[i]) moved += 1;
    }
    prev = Float32Array.from(state.foam);
    trace.push({ t: k * dt, count: state.foamCount, total });
  }
  return { startFoam, trace, grew, moved, breaksAfter, endCount: state.foamCount, steps, dt };
}
const stop = stopBreaking();
if (stop.startFoam === 0) problems.push("stop test: the strong sea made no foam to fade");
if (stop.breaksAfter !== 0) problems.push("stop test: breaks were still flagged after break detection was switched off");
if (stop.grew > 0) problems.push(`stop test: lingering foam grew (${stop.grew})`);
if (stop.moved > 0) problems.push(`stop test: foam appeared where none was laid (${stop.moved})`);
if (stop.endCount !== 0) problems.push(`stop test: foam still there after the window plus one step (${stop.endCount})`);
const mid = stop.trace.find((r) => r.t === 1);
if (!mid || mid.count === 0) problems.push("stop test: foam vanished at once instead of fading");
if (calm.state.flags.some((v) => v > 0)) problems.push("calm sea showed break flags");
if (live.mismatches || calm.mismatches || shallow.mismatches) problems.push("break flag disagreed with the slope test");
for (const band of live.state.bands) {
  if (band.significantHeight < 0.05) problems.push(`${band.id} contributed almost nothing`);
}
const light = run(5, 40, 4);
const strong = run(18, 40, 4);
for (const [name, r] of [["live", live], ["shallow", shallow], ["light", light], ["strong", strong]]) {
  if (r.strayFoam > 0) problems.push(`${name}: foam more than 4 cells from any break (${r.strayFoam})`);
  if (r.behindFoam > 0) problems.push(`${name}: foam behind its break, upwave of the crest (${r.behindFoam})`);
  if (r.state.freshCount > r.state.breakingCount * ((r.state.faceSteps || 3) + 3)) problems.push(`${name}: foam spread far past the breaks (${r.state.freshCount} for ${r.state.breakingCount} breaks)`);
  // GAD 04:15: at 18 m/s foam now lives (U/10)^2 x 2 s = 6.5 s, longer than this 4 s run, so
  // aged foam has barely faded and its mean can sit at the fresh (graded) mean; per-sample
  // fading is checked by 'brighter than laid' (must be 0). Allow 0.05 on the average.
  if (r.agedN > 0 && !(r.agedMean < r.freshMean + 0.05)) problems.push(`${name}: aged foam not weaker than fresh on average (${r.agedMean.toFixed(2)} vs ${r.freshMean.toFixed(2)})`);
  if (r.ageStray > 0) problems.push(`${name}: foam age set where it should not be (${r.ageStray})`);
  if (r.pastWindow > 0) problems.push(`${name}: foam older than the fade window (${r.pastWindow})`);
  if (r.currentNotFull > 0) problems.push(`${name}: flagged samples without full-strength foam (${r.currentNotFull})`);
  if (r.brighter > 0) problems.push(`${name}: lingering foam brighter than when it was laid (${r.brighter})`);
  if (r.currentMissing > 0) problems.push(`${name}: current flags without foam (${r.currentMissing})`);
  if (r.outsideFlags > 0) problems.push(`${name}: foam outside current and recent flags (${r.outsideFlags})`);
  if (r.bothFlags > 0) problems.push(`${name}: samples marked both current and recent (${r.bothFlags})`);
  if (r.breakNotFull > 0) problems.push(`${name}: break samples without full foam (${r.breakNotFull})`);
  if (r.runStray > 0) problems.push(`${name}: lengthwise taper set outside foam or out of range (${r.runStray})`);
  if (r.behindCrest > 0) problems.push(`${name}: foam behind the crest it came from (${r.behindCrest})`);
  if (r.overOne > 0) problems.push(`${name}: foam above 1 (${r.overOne})`);
  if (r.state.breakingCount > 0 && r.graded === 0) problems.push(`${name}: foam is not graded down the face`);
  if (r.foamNoFlag > 0) problems.push(`${name}: foam without a flag (${r.foamNoFlag} samples)`);
  if (r.flagNoFoam > 0) problems.push(`${name}: flag without foam (${r.flagNoFoam} samples)`);
  if (r.faceFoam > 0 && r.meanFaceSlope >= 0) problems.push(`${name}: foam beyond the breaks is not on a downslope in the travel direction`);
}
if (live.faceFoam < live.state.breakingCount * 0.5) problems.push("front faces barely got foam");
const moves = [motion(12, 0.4), motion(12, 0.4 + Math.PI), motion(18, 2.2)];
for (const m of moves) {
  if (m.travelError > 0.05) problems.push(`travel direction drifted from the wind (${m.travelError.toFixed(3)} rad)`);
  if (m.correlation < 0.5) problems.push(`waves do not move toward the reported travel direction (${m.correlation.toFixed(2)})`);
  if (m.frontRising <= m.backRising + 0.15) problems.push(`foam is not on the rising front face (${m.frontRising.toFixed(2)} vs ${m.backRising.toFixed(2)} behind)`);
}
if (light.state.breakingCount > live.state.sampleCount * 0.02) problems.push("light wind foamed too much");
if (live.state.breakingCount < 5) problems.push("developed sea never flagged a crest");
if (shallow.state.breakingCount <= live.state.breakingCount) problems.push("shallow water did not break more than deep water");

// The chop band's fine patch. Built from the bands alone (no foam), as surface.step does.
// which = "fine": the band's second, finer patch, stepped as surface.step does, with its
// low edge at the first patch's cutoff. p is the patch under test; coarse is the first.
function patchSea(speed, depth, size, steps = 2, dt = 0.05, direction = 0.4, which = "coarse") {
  const bands = scenario.bands.map(createBand);
  const chopIndex = bands.findIndex((b) => b.spec.patch);
  const patch = createChopPatch(bands[chopIndex].spec);
  const finer = createChopPatch(bands[chopIndex].spec, bands[chopIndex].spec.finePatch);
  const wind = createWind({ direction, speed, gust: "continuous" });
  const spacing = size / Math.max(scenario.resolution - 1, 1);
  let p = null;
  let coarse = null;
  const snaps = [];
  for (let i = 0; i < steps; i += 1) {
    const winds = bands.map((b) => wind.sample(b.spec, i * dt, dt));
    bands.forEach((b, j) => retuneBand(b, winds[j], depth, GRAVITY, dt, spacing));
    coarse = patch.step(bands[chopIndex], winds[chopIndex], depth, GRAVITY, dt);
    p = which === "fine" ? finer.step(bands[chopIndex], winds[chopIndex], depth, GRAVITY, dt, coarse.kCut) : coarse;
    snaps.push({ re: Float64Array.from(p.heights), im: Float64Array.from(p.heightsQuarter) });
  }
  const gridKMax = Math.max(...bands.map((b) => b.keptKMax));
  return { p, coarse, bands, chop: bands[chopIndex], gridKMax, fft2: (which === "fine" ? finer : patch).fft2, snaps };
}
function patchChecks(which = "coarse", range = 0.6) {
  const out = { overlap: [], calmMax: 0, shells: [], total: null, dispersionErr: 0, clampShare: 0 };
  // 1. No wavelength in two places, across winds, sea sizes and depths. The first patch
  // starts above the grid; the finer patch starts above everything the first can hold
  // (its cutoff) and above anything the first actually holds or the grid holds.
  for (const size of [250, 420]) {
    for (const depth of [40, 3]) {
      for (const speed of [3, 8, 12, 17.5, 25]) {
        const r = patchSea(speed, depth, size, 2, 0.05, 0.4, which);
        const p = r.p;
        const ok = which === "fine"
          ? p.modes === 0 || (p.kMin > r.coarse.kCut && (r.coarse.modes === 0 || p.kMin > r.coarse.kMax) && p.kMin > r.gridKMax && p.kMax <= p.kCut + 1e-9)
          : p.modes === 0 || (p.kMin > r.gridKMax && p.omegaMin > r.chop.keptOmegaMax && p.kMax <= p.kCut + 1e-9);
        out.overlap.push({ size, depth, speed, ok, gridLongestKept: r.gridKMax ? (2 * Math.PI) / r.gridKMax : 0, gridShortest: r.chop.keptKMax ? (2 * Math.PI) / r.chop.keptKMax : 0, coarseShortest: r.coarse.modes ? r.coarse.shortest : 0, coarseCut: (2 * Math.PI) / r.coarse.kCut, patchLongest: p.longest, patchShortest: p.shortest, modes: p.modes });
      }
    }
  }
  // 2. Flat on a calm sea.
  const calmPatch = patchSea(0, 40, 420, 3, 0.05, 0.4, which).p;
  for (const v of calmPatch.heights) out.calmMax = Math.max(out.calmMax, Math.abs(v));
  for (const v of calmPatch.slopes) out.calmMax = Math.max(out.calmMax, Math.abs(v));
  // 3. Height spectrum of the field itself (forward FFT) against the PM chop band.
  const r = patchSea(17.5, 40, 420, 3, 0.05, 0.4, which);
  const p = r.p;
  const N = p.n;
  const re = Float64Array.from(p.heights);
  const im = new Float64Array(N * N);
  r.fft2(re, im, -1);
  const norm = 1 / (N * N);
  const power = new Float64Array(N * N);
  for (let i = 0; i < N * N; i += 1) power[i] = (re[i] * norm) ** 2 + (im[i] * norm) ** 2;
  const omegaOf = (k) => Math.sqrt(GRAVITY * k * Math.tanh(k * 40));
  const pm = (w1, w2) => {
    let sum = 0;
    const steps = 4000;
    for (let j = 0; j < steps; j += 1) {
      const w = w1 + ((j + 0.5) / steps) * (w2 - w1);
      // The band's own spectrum (Elfouhaily or PM, as the scenario says).
      sum += r.chop.spectrum.density(w);
    }
    return (sum * (w2 - w1)) / steps;
  };
  const kLo = p.kMin * 0.999;
  const kHi = p.kMax * 1.001;
  const edges = [0, 1, 2, 3, 4].map((j) => kLo * (kHi / kLo) ** (j / 4));
  let gotAll = 0;
  for (let s = 0; s < 4; s += 1) {
    let got = 0;
    for (let i = 0; i < N * N; i += 1) {
      const k = Math.hypot(p.kx[i], p.kz[i]);
      if (k >= edges[s] && k < edges[s + 1]) got += power[i];
    }
    const w1 = Math.max(omegaOf(edges[s]), p.omegaLow);
    const want = pm(w1, omegaOf(edges[s + 1]));
    gotAll += got;
    out.shells.push({ from: (2 * Math.PI) / edges[s + 1], to: (2 * Math.PI) / edges[s], got, want, ratio: got / want });
  }
  const wantAll = pm(Math.max(omegaOf(kLo), p.omegaLow), omegaOf(kHi));
  out.total = { got: gotAll, want: wantAll, ratio: gotAll / wantAll, hs: 4 * Math.sqrt(gotAll) };
  // 4. It moves by the dispersion relation: each wave's phase between two steps,
  // measured from the field, gives omega; invert omega to k and compare.
  const [a, b] = r.snaps.slice(-2);
  const fa = { re: Float64Array.from(a.re), im: Float64Array.from(a.im) };
  const fb = { re: Float64Array.from(b.re), im: Float64Array.from(b.im) };
  r.fft2(fa.re, fa.im, -1);
  r.fft2(fb.re, fb.im, -1);
  for (let i = 0; i < N * N; i += 1) {
    if (!(p.amplitudes[i] > 0)) continue;
    // z = sum a e^{i(k.x + phase)}, so the forward transform at k is a e^{i phase}.
    const da = Math.atan2(fb.im[i] * fa.re[i] - fb.re[i] * fa.im[i], fb.re[i] * fa.re[i] + fb.im[i] * fa.im[i]);
    const w = -da / 0.05;
    const k = Math.hypot(p.kx[i], p.kz[i]);
    out.dispersionErr = Math.max(out.dispersionErr, Math.abs(wavenumber(w, 40, GRAVITY) - k) / k);
  }
  let clamped = 0;
  for (const v of p.slopes) if (Math.abs(v) > range) clamped += 1;
  out.clampShare = clamped / p.slopes.length;
  out.live = p;
  out.model = r.chop.spectrum.model;
  return out;
}
const pc = patchChecks();
for (const o of pc.overlap) if (!o.ok) problems.push(`chop patch overlaps the big grid at ${o.speed} m/s, size ${o.size} m, depth ${o.depth} m`);
if (pc.overlap.filter((o) => o.speed >= 8).some((o) => o.modes === 0)) problems.push("chop patch empty in a developed sea");
if (pc.calmMax > 0) problems.push(`chop patch moved on a calm sea (${pc.calmMax})`);
for (const sh of pc.shells) if (Math.abs(sh.ratio - 1) > 0.15) problems.push(`chop patch spectrum off its band spectrum in ${sh.from.toFixed(2)}-${sh.to.toFixed(2)} m (${sh.ratio.toFixed(3)})`);
if (Math.abs(pc.total.ratio - 1) > 0.05) problems.push(`chop patch variance off its band spectrum (${pc.total.ratio.toFixed(3)})`);
if (pc.dispersionErr > 1e-3) problems.push(`chop patch waves do not move by the dispersion relation (${pc.dispersionErr})`);
if (pc.clampShare > 1e-4) problems.push(`chop patch slopes exceed the texture range (${pc.clampShare})`);
// The finer patch (about 1.05 to 0.25 m): same four checks, its texture range is 0.6.
const fc = patchChecks("fine", 0.6);
for (const o of fc.overlap) if (!o.ok) problems.push(`finer patch overlaps the first patch or the grid at ${o.speed} m/s, size ${o.size} m, depth ${o.depth} m`);
if (fc.overlap.filter((o) => o.speed >= 8).some((o) => o.modes === 0)) problems.push("finer patch empty in a developed sea");
if (fc.calmMax > 0) problems.push(`finer patch moved on a calm sea (${fc.calmMax})`);
for (const sh of fc.shells) if (Math.abs(sh.ratio - 1) > 0.15) problems.push(`finer patch spectrum off its band spectrum in ${sh.from.toFixed(2)}-${sh.to.toFixed(2)} m (${sh.ratio.toFixed(3)})`);
if (Math.abs(fc.total.ratio - 1) > 0.05) problems.push(`finer patch variance off its band spectrum (${fc.total.ratio.toFixed(3)})`);
if (fc.dispersionErr > 1e-3) problems.push(`finer patch waves do not move by the dispersion relation (${fc.dispersionErr})`);
if (fc.clampShare > 1e-4) problems.push(`finer patch slopes exceed the texture range (${fc.clampShare})`);
// Handover between the wind-sea band (below) and the chop band (above): the energy
// density each band uses at the shared frequency, and so the slope density k^2 S there,
// must agree; a step would be a visible change in roughness at that wavelength. Checked
// on the formulas at the edge and on the grid waves' own variance per rad/s.
function handover(speed, depth = 40) {
  const bands = scenario.bands.map(createBand);
  const wind = createWind({ direction: 0.4, speed, gust: "continuous" });
  const spacing = 420 / Math.max(scenario.resolution - 1, 1);
  const winds = bands.map((b) => wind.sample(b.spec, 0, 0.05));
  bands.forEach((b, j) => retuneBand(b, winds[j], depth, GRAVITY, 0.05, spacing));
  const out = [];
  for (let j = 0; j + 1 < bands.length; j += 1) {
    const lo = bands[j];
    const hi = bands[j + 1];
    if (Math.abs(lo.spec.omegaMax - hi.spec.omegaMin) > 1e-9) continue;
    const w = hi.spec.omegaMin * hi.omegaPeak;
    const below = lo.spectrum.density(w);
    const above = hi.spectrum.density(w);
    // Grid waves at the edge: variance per rad/s of the last frequency below and the first above.
    const perOmega = (b, first) => {
      const dO = ((b.spec.omegaMax - b.spec.omegaMin) / b.spec.frequencies) * b.omegaPeak;
      const row = first ? b.components.slice(0, b.spec.directions) : b.components.slice(-b.spec.directions);
      return row.reduce((sum, c) => sum + (c.amplitude * c.amplitude) / 2, 0) / dO;
    };
    const gLo = perOmega(lo, false);
    const gHi = perOmega(hi, true);
    out.push({ speed, from: lo.spec.id, to: hi.spec.id, lambda: (2 * Math.PI * GRAVITY) / (w * w), ratio: above / below, gridRatio: gLo > 0 && gHi > 0 ? gHi / gLo : 1, models: `${lo.spectrum.model}->${hi.spectrum.model}` });
  }
  return out;
}
const handovers = [3, 5, 8, 10, 12, 17.5, 25].flatMap((u) => handover(u));
for (const h of handovers) {
  if (Math.abs(h.ratio - 1) > 0.01) problems.push(`energy steps by ${h.ratio.toFixed(3)} from ${h.from} to ${h.to} at ${h.speed} m/s`);
  if (Math.abs(h.gridRatio - 1) > 0.01) problems.push(`grid waves step by ${h.gridRatio.toFixed(3)} from ${h.from} to ${h.to} at ${h.speed} m/s`);
}

// Total mean square slope against Cox and Munk's clean-surface fit, mss = 0.003 + 5.12e-3 U.
// Resolved: every band's grid waves plus both patches, as modelled. Not modelled: waves
// shorter than the finer patch's cutoff (about 0.25 m), taken from the chop band's own
// spectrum out to 20 x k_m (well past where it has died away).
function slopeBudget(speed) {
  const wind = createWind({ direction: 0.4, speed, gust: "continuous" });
  const sea = createSurface({ ...scenario, size: 420, choppiness: 2.5 }, wind);
  // Bands and patches only: read them off one step (foam does not matter here).
  const st = sea.step(0.05, 0.05, 40);
  const bands = scenario.bands.map(createBand);
  const winds = bands.map((b) => wind.sample(b.spec, 0.05, 0.05));
  bands.forEach((b, j) => retuneBand(b, winds[j], 40, GRAVITY, 0.05, 420 / Math.max(scenario.resolution - 1, 1)));
  // The shared slope budget (roughness.js), the same numbers the renderer gets.
  const { grid, coarse, fine } = st.roughness;
  const chop = bands.find((b) => b.spec.patch);
  const k0 = st.finePatch.kCut;
  let unresolved = 0;
  if (chop.spectrum.model === "elfouhaily") {
    const p = chop.spectrum.params;
    const steps = 20000;
    const l0 = Math.log(k0);
    const l1 = Math.log(20 * p.km);
    for (let i = 0; i < steps; i += 1) {
      const k = Math.exp(l0 + ((i + 0.5) / steps) * (l1 - l0));
      const { bl, bh } = elfouhailyParts(k, p);
      unresolved += (chop.spectrum.longScale * bl + bh) * ((l1 - l0) / steps); // B dk/k
    }
  }
  const resolved = grid + coarse + fine;
  const cm = st.roughness.coxMunk;
  return { subGrid: st.roughness.subGrid, speed, grid, coarse, fine, resolved, unresolved, total: resolved + unresolved, cm, ratio: (resolved + unresolved) / cm, resolvedShare: resolved / cm, fineRms: st.finePatch.rmsSlope, coarseRms: st.chopPatch.rmsSlope };
}
const budgets = [5, 10, 17.5, 25].map(slopeBudget);
for (const b of budgets) if (Math.abs(b.ratio - 1) > 0.25) problems.push(`total slope ${b.total.toFixed(4)} is off Cox and Munk ${b.cm.toFixed(4)} at ${b.speed} m/s (${b.ratio.toFixed(2)})`);
for (let i = 1; i < budgets.length; i += 1) {
  if (!(budgets[i].fineRms > budgets[i - 1].fineRms && budgets[i].resolved > budgets[i - 1].resolved)) problems.push(`short-wave roughness does not rise with the wind (${budgets[i - 1].speed} to ${budgets[i].speed} m/s)`);
}

// Glint and reflection roughness (roughness.js): the sub-grid mss the renderer spreads
// light by is the Cox and Munk total minus the resolved slope, so the budget closes
// exactly, stays positive at every wind on the slider and beyond, and rises with wind.
const roughWinds = [0, 3, 5, 10, 17.5, 25, 34].map((speed) => {
  const st = createSurface({ ...scenario, size: 420, choppiness: 2.5 }, createWind({ direction: 0.4, speed, gust: "continuous" })).step(0.05, 0.05, 40);
  return { speed, r: st.roughness };
});
for (const { speed, r } of roughWinds) {
  if (Math.abs(r.subGrid + r.resolved - coxMunkMss(speed)) > 1e-12) problems.push(`roughness budget does not close at ${speed} m/s`);
  if (!(r.subGrid > 0)) problems.push(`no sub-grid roughness left at ${speed} m/s (resolved ${r.resolved.toFixed(4)} vs Cox and Munk ${r.coxMunk.toFixed(4)})`);
}
for (let i = 1; i < roughWinds.length; i += 1) if (!(roughWinds[i].r.subGrid > roughWinds[i - 1].r.subGrid)) problems.push(`sub-grid roughness does not rise from ${roughWinds[i - 1].speed} to ${roughWinds[i].speed} m/s`);
// It follows the live wind, gusts included: step by step it is Cox and Munk at the chop
// band's lagged wind (replayed here from the wind model alone) minus the resolved slope.
function roughGust() {
  const sea = createSurface({ ...scenario, size: 420, choppiness: 2.5 }, createWind({ direction: Math.PI / 2, speed: 17.5, gust: "gusty" }));
  const replay = createWind({ direction: Math.PI / 2, speed: 17.5, gust: "gusty" });
  const chopSpec = scenario.bands.find((b) => b.patch);
  let worst = 0;
  let lo = Infinity;
  let hi = 0;
  for (let i = 1; i <= 480; i += 1) {
    const r = sea.step(i * 0.05, 0.05, 40).roughness;
    const u = replay.sample(chopSpec, i * 0.05, 0.05).speed;
    worst = Math.max(worst, Math.abs(r.windSpeed - u), Math.abs(r.subGrid - (coxMunkMss(u) - r.resolved)));
    lo = Math.min(lo, r.subGrid);
    hi = Math.max(hi, r.subGrid);
  }
  return { worst, lo, hi };
}
const roughG = roughGust();
if (roughG.worst > 1e-9) problems.push(`roughness does not follow the live (lagged, gusty) wind: off by ${roughG.worst}`);
if (!(roughG.hi / roughG.lo > 1.2)) problems.push(`roughness barely moves with gusts (${roughG.lo.toFixed(4)} to ${roughG.hi.toFixed(4)})`);

// The chop band's spectrum choice must not change breaking: the same sea with the chop
// band on PM and on Elfouhaily flags exactly the same samples, step for step.
function breakInvariance(speed, steps) {
  const make = (model) => createSurface({ ...scenario, size: 420, choppiness: 2.5, bands: scenario.bands.map((b) => (b.patch ? { ...b, spectrum: model } : b)) }, createWind({ direction: Math.PI / 2, speed, gust: "gusty" }));
  const a = make("pm");
  const b = make("elfouhaily");
  let differ = 0;
  let counts = [];
  for (let i = 1; i <= steps; i += 1) {
    const sa = a.step(i * 0.05, 0.05, 40);
    const sb = b.step(i * 0.05, 0.05, 40);
    if (sa.breakingCount !== sb.breakingCount) differ += 1;
    for (let j = 0; j < sa.flags.length; j += 1) if (sa.flags[j] !== sb.flags[j] || sa.foam[j] !== sb.foam[j]) { differ += 1; break; }
    if (i === steps) counts = [sa.breakingCount, sb.breakingCount];
  }
  return { speed, steps, differ, counts };
}
// Texture range holds up to the strongest gust (25 m/s slider x 1.35).
const gustRange = ["coarse", "fine"].map((which) => {
  const p = patchSea(34, 40, 420, 1, 0.05, 0.4, which).p;
  let over = 0;
  for (const v of p.slopes) if (Math.abs(v) > 0.6) over += 1;
  return { which, share: over / p.slopes.length, rms: p.rmsSlope };
});
for (const g of gustRange) if (g.share > 1e-4) problems.push(`${g.which} patch slopes exceed the texture range at 34 m/s (${g.share})`);
const invariance = [breakInvariance(17.5, 30), breakInvariance(25, 30)];
// v28 foam fix (GAD): the break test now reads the total height, chop included, so the chop
// spectrum choice changes breaking by design; reported, not a failure (counts printed below).

// Translucency (thin-crest transmission, shading only): exactly 0 on flat water and on
// thick water (a long rounded swell alone), within [0, 1], and present on a live sea.
function translucency() {
  const flatMax = Math.max(0, ...calm.state.thins);
  const swellOnly = createSurface({ ...scenario, bands: scenario.bands.filter((b) => b.id === "swell") }, createWind({ direction: 0.4, speed: 10, gust: "continuous" }));
  let thick = null;
  for (let i = 0; i < 12; i += 1) thick = swellOnly.step(4 + i * 0.05, 0.05, 40);
  const thickMax = Math.max(0, ...thick.thins);
  let liveMax = 0;
  let liveMin = 0;
  let liveLit = 0;
  for (const v of live.state.thins) {
    liveMax = Math.max(liveMax, v);
    liveMin = Math.min(liveMin, v);
    if (v > 0) liveLit += 1;
  }
  return { flatMax, thickMax, swellHs: thick.bands[0].significantHeight, liveMax, liveMin, liveShare: liveLit / live.state.thins.length };
}
const transl = translucency();
if (transl.flatMax !== 0) problems.push(`translucency on flat water (${transl.flatMax})`);
if (transl.thickMax !== 0) problems.push(`translucency on thick water (swell only, ${transl.thickMax})`);
if (transl.liveMax > 1 || transl.liveMin < 0) problems.push("translucency outside [0, 1]");
if (!(transl.liveMax > 0)) problems.push("no translucency on the live sea");
// CPU-vs-GPU parity (gpuSurface.js against this CPU physics). The card needs a WebGL2
// browser, so this serves the app on a local port and runs ?parity=1 in headless Chrome
// (puppeteer-core): heights within 1 mm, break and painted flags on >= 99.5% of points with
// every mismatch at a threshold edge and none in calm water, GPU foam exactly on its flags,
// panel counts equal to the fields. Without a headless browser it is reported as skipped;
// OCEAN_GPU_PARITY=require makes that a failure, =off skips it.
async function gpuParity() {
  const mode = process.env.OCEAN_GPU_PARITY || "auto";
  if (mode === "off") return { skipped: "OCEAN_GPU_PARITY=off" };
  const { createRequire } = await import("node:module");
  const { pathToFileURL, fileURLToPath } = await import("node:url");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const http = await import("node:http");
  let puppeteer = null;
  for (const from of [process.env.PUPPETEER_FROM, path.dirname(fileURLToPath(import.meta.url)), process.cwd()].filter(Boolean)) {
    try {
      const resolved = createRequire(path.join(from, "noop.js")).resolve("puppeteer-core");
      puppeteer = (await import(pathToFileURL(resolved).href)).default;
      break;
    } catch { /* try the next place */ }
  }
  if (!puppeteer) return { skipped: "puppeteer-core not found (set PUPPETEER_FROM to a folder that has it)" };
  const chrome = process.env.CHROME || ["/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/chromium"].find((p) => fs.existsSync(p));
  if (!chrome) return { skipped: "no Chrome found (set CHROME)" };
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
  const types = { ".js": "application/javascript", ".html": "text/html", ".css": "text/css", ".json": "application/json" };
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      if (file === root + "/" || file === root) { res.writeHead(200, { "content-type": "text/html" }); res.end(fs.readFileSync(path.join(root, "index.html"))); return; }
      res.writeHead(404); res.end(); return;
    }
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
    res.end(fs.readFileSync(file));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const args = process.env.CHROME_ARGS ? process.env.CHROME_ARGS.split(" ")
    : process.platform === "linux" ? ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox", "--ignore-gpu-blocklist"] : [];
  const browser = await puppeteer.launch({ executablePath: chrome, headless: "new", args, protocolTimeout: 3600000 });
  try {
    const page = await browser.newPage();
    const seconds = process.env.OCEAN_PARITY_SECONDS || "3";
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html?parity=1&speeds=0,5,10,17.5,25&seconds=${seconds}`);
    await page.waitForFunction(() => window.__parity, { timeout: 3600000, polling: 500 });
    return await page.evaluate(() => window.__parity);
  } finally {
    await browser.close();
    server.close();
  }
}
const parity = await gpuParity().catch((error) => ({ error: String(error && error.message || error) }));
if (parity.error) problems.push("GPU parity could not run: " + parity.error);
else if (parity.skipped) {
  if (process.env.OCEAN_GPU_PARITY === "require") problems.push("GPU parity skipped: " + parity.skipped);
} else if (!parity.ok) problems.push(...parity.problems.map((p) => "GPU parity: " + p));

if (problems.length) {
  console.error(problems.join("\n"));
  console.error(JSON.stringify(live.state.bands, null, 2));
  console.error("live breaks", live.state.breakingCount, "max slope", live.state.maxSlope.toFixed(3));
  console.error("shallow breaks", shallow.state.breakingCount, "max slope", shallow.state.maxSlope.toFixed(3));
  process.exit(1);
}
console.log("selfcheck ok");
console.log(parity.skipped ? `GPU parity: skipped (${parity.skipped})` : parity.summary);
console.log(`translucency (thin crests): flat max ${transl.flatMax}, thick (swell only, Hs ${transl.swellHs.toFixed(2)} m) max ${transl.thickMax}, live max ${transl.liveMax.toFixed(3)}, live crests lit ${(100 * transl.liveShare).toFixed(1)}% of samples`);
console.log(live.state.bands.map((b) => `${b.id} Hs ${b.significantHeight.toFixed(2)} m`).join(", "));
console.log(`live breaks ${live.state.breakingCount}/${live.state.sampleCount}, foam now ${live.state.freshCount} (${live.faceFoam} ahead of or between breaks), lingering ${live.state.lingerCount}, max slope ${live.state.maxSlope.toFixed(3)}`);
console.log(`stop test: ${stop.startFoam} foam samples, then breaking off: ` + stop.trace.map((r) => `${r.t.toFixed(2)} s ${r.count}`).join(", ") + ` (grew ${stop.grew}, appeared ${stop.moved})`);
console.log([["live", live], ["shallow", shallow], ["light", light], ["strong", strong], ["calm", calm]].map(([name, r]) => `${name}: past window ${r.pastWindow}, flagged not full ${r.currentNotFull}, brighter than laid ${r.brighter}, current flag without foam ${r.currentMissing}, foam outside current+recent ${r.outsideFlags}, mean strength fresh ${r.freshMean.toFixed(2)} vs aged ${r.agedMean.toFixed(2)}, max foam ${Math.max(0, ...r.state.foam).toFixed(2)}`).join("\n"));
console.log([["live", live], ["shallow", shallow], ["light", light], ["strong", strong], ["calm", calm]].map(([name, r]) => `${name}: foam now ${r.state.freshCount}, flagged ${r.flagged}, foam without flag ${r.foamNoFlag}, flag without foam ${r.flagNoFoam}, graded below 1 ${r.graded}, behind its crest ${r.behindCrest}, furthest ahead ${r.maxAhead.toFixed(1)} m`).join("\n"));
console.log(`foam now per break: live ${(live.state.freshCount / live.state.breakingCount).toFixed(2)}, shallow ${(shallow.state.freshCount / shallow.state.breakingCount).toFixed(2)}, strong ${(strong.state.freshCount / Math.max(1, strong.state.breakingCount)).toFixed(2)} (limit face steps + 3: live ${live.state.faceSteps + 3}, strong ${strong.state.faceSteps + 3})`);
console.log(moves.map((m) => `travel check: height change vs front-face slope r=${m.correlation.toFixed(2)}, front-face foam rising ${(100 * m.frontRising).toFixed(0)}% vs ${(100 * m.backRising).toFixed(0)}% behind the break`).join("\n"));
console.log(`light breaks ${light.state.breakingCount}, strong breaks ${strong.state.breakingCount}`);
console.log(`shallow breaks ${shallow.state.breakingCount}/${shallow.state.sampleCount}, max slope ${shallow.state.maxSlope.toFixed(3)}`);
console.log(`chop patch: ${pc.live.modes} waves, ${pc.live.longest.toFixed(2)}-${pc.live.shortest.toFixed(2)} m on a ${pc.live.size} m patch at ${pc.live.n} (${pc.live.spacing.toFixed(3)} m), Hs ${pc.live.significantHeight.toFixed(3)} m, rms slope ${pc.live.rmsSlope.toFixed(3)}, slopes past texture range ${(pc.clampShare * 100).toFixed(4)}%`);
console.log("chop patch vs big grid (no overlap): " + pc.overlap.map((o) => `${o.speed} m/s ${o.size} m d${o.depth}: grid chop to ${o.gridShortest.toFixed(2)} m, patch ${o.patchLongest.toFixed(2)}-${o.patchShortest.toFixed(2)} m ${o.ok ? "ok" : "OVERLAP"}`).join("; "));
console.log(`chop patch calm: max |height or slope| ${pc.calmMax}`);
console.log(`chop patch spectrum vs its band spectrum (${pc.model}): ` + pc.shells.map((sh) => `${sh.from.toFixed(2)}-${sh.to.toFixed(2)} m ${sh.ratio.toFixed(3)}`).join(", ") + `; total ${pc.total.ratio.toFixed(3)} (Hs ${pc.total.hs.toFixed(3)} m)`);
console.log(`chop patch dispersion: worst k error from measured phase speed ${(pc.dispersionErr * 100).toFixed(5)}%`);
console.log(`finer patch: ${fc.live.modes} waves, ${fc.live.longest.toFixed(3)}-${fc.live.shortest.toFixed(3)} m on a ${fc.live.size} m patch at ${fc.live.n} (${fc.live.spacing.toFixed(3)} m), Hs ${fc.live.significantHeight.toFixed(4)} m, rms slope ${fc.live.rmsSlope.toFixed(3)}, slopes past texture range ${(fc.clampShare * 100).toFixed(4)}%`);
console.log("finer patch vs first patch (no overlap): " + fc.overlap.map((o) => `${o.speed} m/s ${o.size} m d${o.depth}: first patch to ${o.coarseShortest.toFixed(3)} m (cut ${o.coarseCut.toFixed(3)} m), finer ${o.patchLongest.toFixed(3)}-${o.patchShortest.toFixed(3)} m ${o.ok ? "ok" : "OVERLAP"}`).join("; "));
console.log(`finer patch calm: max |height or slope| ${fc.calmMax}`);
console.log(`finer patch spectrum vs its band spectrum (${fc.model}): ` + fc.shells.map((sh) => `${sh.from.toFixed(3)}-${sh.to.toFixed(3)} m ${sh.ratio.toFixed(3)}`).join(", ") + `; total ${fc.total.ratio.toFixed(3)} (Hs ${fc.total.hs.toFixed(4)} m)`);
console.log(`finer patch dispersion: worst k error from measured phase speed ${(fc.dispersionErr * 100).toFixed(5)}%`);
console.log("handover (energy and slope density, above/below): " + handovers.map((h) => `${h.speed} m/s ${h.from}->${h.to} (${h.models}) at ${h.lambda.toFixed(1)} m: formula ${h.ratio.toFixed(4)}, grid waves ${h.gridRatio.toFixed(4)}`).join("; "));
console.log("slope vs Cox and Munk: " + budgets.map((b) => `${b.speed} m/s: grid ${b.grid.toFixed(4)} + patch ${b.coarse.toFixed(4)} + finer ${b.fine.toFixed(4)} = resolved ${b.resolved.toFixed(4)} (rms ${Math.sqrt(b.resolved).toFixed(3)}, ${(100 * b.resolvedShare).toFixed(0)}% of CM), shorter than 0.25 m ${b.unresolved.toFixed(4)}, total ${b.total.toFixed(4)} vs CM ${b.cm.toFixed(4)} (${b.ratio.toFixed(2)})`).join("; "));
console.log("chop spectrum switch vs breaking: " + invariance.map((v) => `${v.speed} m/s gusty, ${v.steps} steps: ${v.differ} differ, last breaks PM ${v.counts[0]} / Elfouhaily ${v.counts[1]}`).join("; "));
console.log("texture range at 34 m/s: " + gustRange.map((g) => `${g.which} rms ${g.rms.toFixed(3)}, past range ${(100 * g.share).toFixed(4)}%`).join("; "));
console.log(`glint/reflection roughness (Cox and Munk minus resolved): ${roughWinds.map(({ speed, r }) => `${speed} m/s ${r.subGrid.toFixed(4)} of ${r.coxMunk.toFixed(4)} (lobe rms ${Math.sqrt(r.subGrid).toFixed(3)})`).join(", ")}; 17.5 m/s gusty over 24 s: ${roughG.lo.toFixed(4)} to ${roughG.hi.toFixed(4)}, follows the lagged wind to ${roughG.worst.toExponential(1)}`);

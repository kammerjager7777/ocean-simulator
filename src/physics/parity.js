// CPU-vs-GPU parity (tests only; reads whole fields back, which the app never does).
// The v26 CPU physics (surface.js) is the yardstick: the same scenario, the same wind, the
// same steps, side by side. Per checkpoint:
//  - heights within 1 mm at every grid point;
//  - break flags (the break test, and the painted flags the panel and foam use) matching on
//    at least 99.5% of points, every mismatch at a threshold edge (some comparison that
//    decides it within EDGE of flipping), none in calm water (no CPU break within 6 cells);
//  - foam matching the flags exactly on the card: fresh foam == flagged, foam only where
//    flagged now or in the window, and the panel counts (reduction) == the fields.
import { scenario as baseScenario } from "./scenario.js?v=28";
import { createWind } from "./wind.js";
import { createSurface, crestReachAt, breakRuleAt, foamWindowAt } from "./surface.js?v=28";
import { createGpuSurface } from "./gpuSurface.js?v=28";
import { createChopPatch } from "./chopPatch.js?v=28";
import { GRAVITY } from "./gravity.js";

const EDGE = 2e-3; // metres (heights) or slope units: an order above float32 noise here
const HEIGHT_TOL = 1e-3;

function edgeMargin(cpu, sc, index) {
  // How close the break test at this sample comes to flipping, on the CPU's own fields.
  const { n, breakHeights: bh, breakSlopeX, breakSlopeZ } = cpu;
  const ix = index % n;
  const iz = (index - ix) / n;
  const at = (x, z) => bh[z * n + x];
  const spacing = sc.size / Math.max(n - 1, 1);
  const reach = Math.max(1, Math.round((sc.reachM ?? sc.crestReach) / spacing));
  const depthLimit = sc.depthBreakRatio * cpu.depth;
  const h = at(ix, iz);
  const steps = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  let m = Infinity;
  const falls = steps.map(([sx, sz]) => h - at(ix + sx, iz + sz));
  for (let a = 0; a < 4; a += 1) {
    m = Math.min(m, Math.abs(falls[a]));
    for (let b = a + 1; b < 4; b += 1) m = Math.min(m, Math.abs(falls[a] - falls[b]));
  }
  for (const [sx, sz] of steps) {
    m = Math.min(m, Math.abs(h - at(ix - sx, iz - sz)));
    for (let s = 0; s <= reach; s += 1) {
      const fx = ix + sx * s;
      const fz = iz + sz * s;
      if (fx < 0 || fz < 0 || fx >= n || fz >= n) break;
      const q = fz * n + fx;
      m = Math.min(m, Math.abs(Math.hypot(breakSlopeX[q], breakSlopeZ[q]) - sc.breakSlope));
      if (bh[q] > 0) m = Math.min(m, Math.abs(bh[q] - depthLimit));
    }
  }
  return m;
}

// Calm water: nowhere near breaking. The steepest break-field slope within the face reach
// (in a square around the sample) is under 3/4 of the break slope, and nothing is
// depth-limited there.
function calmAt(cpu, sc, index) {
  const { n, breakHeights: bh, breakSlopeX, breakSlopeZ } = cpu;
  const ix = index % n;
  const iz = (index - ix) / n;
  const spacing = sc.size / Math.max(n - 1, 1);
  const reach = Math.max(1, Math.round((sc.reachM ?? sc.crestReach) / spacing)) + 3;
  const depthLimit = sc.depthBreakRatio * cpu.depth;
  for (let dz = -reach; dz <= reach; dz += 1) for (let dx = -reach; dx <= reach; dx += 1) {
    const x = ix + dx, z = iz + dz;
    if (x < 0 || z < 0 || x >= n || z >= n) continue;
    const q = z * n + x;
    if (Math.hypot(breakSlopeX[q], breakSlopeZ[q]) >= 0.75 * sc.breakSlope) return false;
    if (bh[q] > 0 && bh[q] >= depthLimit) return false;
  }
  return true;
}

function paintMargin(cpu, index, reachCells) {
  // Smallest break-field height gap between this sample and its neighbours along any
  // cell step nearby: the comparisons the paint (face drop, travel crest, join snap) uses.
  const { n, breakHeights: bh } = cpu;
  const ix = index % n;
  const iz = (index - ix) / n;
  let m = Infinity;
  for (let dz = -reachCells; dz <= reachCells; dz += 1) {
    for (let dx = -reachCells; dx <= reachCells; dx += 1) {
      const x = ix + dx, z = iz + dz;
      if (x < 1 || z < 1 || x >= n - 1 || z >= n - 1) continue;
      const h = bh[z * n + x];
      for (const [sx, sz] of [[1, 0], [0, 1], [1, 1], [1, -1]]) m = Math.min(m, Math.abs(h - bh[(z + sz) * n + x + sx]));
      m = Math.min(m, Math.abs(h));
    }
  }
  return m;
}

function compare(cpu, gpuFields, gpuCounts, sc, history, time) {
  const { n } = cpu;
  const N = n * n;
  const g = gpuFields;
  let maxHeight = 0, maxShift = 0, maxSlope = 0, maxNormal = 0;
  for (let i = 0; i < N; i += 1) {
    maxHeight = Math.max(maxHeight, Math.abs(cpu.heights[i] - g.heights[i]));
    maxShift = Math.max(maxShift, Math.abs(cpu.shifts[2 * i] - g.shiftX[i]), Math.abs(cpu.shifts[2 * i + 1] - g.shiftZ[i]));
    maxSlope = Math.max(maxSlope, Math.abs(cpu.slopeX[i] - g.slopeX[i]), Math.abs(cpu.slopeZ[i] - g.slopeZ[i]));
    maxNormal = Math.max(maxNormal, Math.abs(cpu.normals[3 * i] - g.normalX[i]), Math.abs(cpu.normals[3 * i + 1] - g.normalY[i]), Math.abs(cpu.normals[3 * i + 2] - g.normalZ[i]));
  }
  // Break test.
  const cpuBreakAt = [];
  for (let i = 0; i < N; i += 1) if (cpu.breaks[i] > 0) cpuBreakAt.push(i);
  const nearBreak = new Uint8Array(N);
  for (const i of cpuBreakAt) {
    const ix = i % n, iz = (i - ix) / n;
    for (let dz = -6; dz <= 6; dz += 1) for (let dx = -6; dx <= 6; dx += 1) {
      const x = ix + dx, z = iz + dz;
      if (x >= 0 && z >= 0 && x < n && z < n) nearBreak[z * n + x] = 1;
    }
  }
  let breakMismatch = 0, breakEdgeWorst = 0, breakOffEdge = 0, breakCalm = 0;
  const breakMis = [];
  for (let i = 0; i < N; i += 1) {
    if ((cpu.breaks[i] > 0) === (g.breaks[i] > 0)) continue;
    breakMismatch += 1;
    breakMis.push(i);
    const m = edgeMargin(cpu, sc, i);
    breakEdgeWorst = Math.max(breakEdgeWorst, m);
    if (!(m <= EDGE)) breakOffEdge += 1;
    if (calmAt(cpu, sc, i)) breakCalm += 1;
  }
  // Painted flags (what the panel, Show break flags and the foam use).
  const nearBreakMis = new Uint8Array(N);
  for (const i of breakMis) {
    const ix = i % n, iz = (i - ix) / n;
    for (let dz = -10; dz <= 10; dz += 1) for (let dx = -10; dx <= 10; dx += 1) {
      const x = ix + dx, z = iz + dz;
      if (x >= 0 && z >= 0 && x < n && z < n) nearBreakMis[z * n + x] = 1;
    }
  }
  let flagMismatch = 0, flagOffEdge = 0, flagCalm = 0, flagEdgeWorst = 0, gradeMismatch = 0;
  let aheadMax = 0;
  for (let i = 0; i < N; i += 1) {
    const cf = cpu.flags[i] > 0, gf = g.flags[i] > 0;
    if (cf !== gf || (cf && cpu.freshFoam[i] !== g.freshFoam[i])) history[i] = time;
    if (cf !== gf) {
      flagMismatch += 1;
      if (calmAt(cpu, sc, i)) flagCalm += 1;
      if (!nearBreakMis[i]) {
        const m = paintMargin(cpu, i, 4);
        flagEdgeWorst = Math.max(flagEdgeWorst, m);
        if (!(m <= EDGE)) flagOffEdge += 1;
      }
    } else if (cf) {
      if (cpu.freshFoam[i] !== g.freshFoam[i]) gradeMismatch += 1;
      else aheadMax = Math.max(aheadMax, Math.abs(cpu.ahead[i] - g.freshAhead[i]));
    }
  }
  // Foam on the card against its own flags, exactly.
  let foamNotFlag = 0, flagNotFoam = 0, foamOutside = 0, foamAgree = 0, foamCells = 0;
  let foamMisMax = 0, foamDiffMax = 0, ageDiffMax = 0, foamUnexplained = 0;
  let gBreaks = 0, gFlags = 0, gFoam = 0, gLinger = 0, gRecent = 0;
  for (let i = 0; i < N; i += 1) {
    const flagged = g.flags[i] > 0;
    if (g.freshFoam[i] > 0 && !flagged) foamNotFlag += 1;
    if (flagged && !(g.freshFoam[i] > 0)) flagNotFoam += 1;
    if (g.foam[i] > 0 && !flagged && !(g.recent[i] > 0)) foamOutside += 1;
    const cpuFoam = cpu.foam[i] > 0, gpuFoam = g.foam[i] > 0;
    if (cpuFoam || gpuFoam) {
      foamCells += 1;
      if (cpuFoam === gpuFoam) foamAgree += 1;
      else {
        foamMisMax = Math.max(foamMisMax, cpu.foam[i], g.foam[i]);
        // Explained: this sample's flags disagreed within the foam window (the foam laid
        // then lingers), or it sits at the window's end (fade below 1e-6).
        const flipped = time - history[i] <= foamWindowAt(sc, sc.paritySpeed ?? 25) + 1e-6;
        if (!flipped && Math.max(cpu.foam[i], g.foam[i]) > 1e-6) foamUnexplained += 1;
      }
      foamDiffMax = Math.max(foamDiffMax, Math.abs(cpu.foam[i] - g.foam[i]));
    }
    if (cpu.ages[i] < 1e29 && g.ages[i] < 1e29) ageDiffMax = Math.max(ageDiffMax, Math.abs(cpu.ages[i] - g.ages[i]));
    if (g.breaks[i] > 0) gBreaks += 1;
    if (flagged) gFlags += 1;
    if (g.foam[i] > 0) { gFoam += 1; if (!flagged) gLinger += 1; }
    if (g.recent[i] > 0) gRecent += 1;
  }
  const countsMatch = gpuCounts.breakingCount === gBreaks && gpuCounts.freshCount === gFlags && gpuCounts.foamCount === gFoam
    && gpuCounts.lingerCount === gLinger && gpuCounts.recentCount === gRecent;
  return {
    maxHeight, maxShift, maxSlope, maxNormal,
    cpuBreaks: cpu.breakingCount, gpuBreaks: gpuCounts.breakingCount,
    breakAgree: 1 - breakMismatch / N, breakMismatch, breakOffEdge, breakCalm, breakEdgeWorst,
    cpuFlags: cpu.freshCount, gpuFlags: gFlags,
    flagAgree: 1 - flagMismatch / N, flagMismatch, flagOffEdge, flagCalm, flagEdgeWorst, gradeMismatch, aheadMax,
    foamNotFlag, flagNotFoam, foamOutside, foamAgree: foamCells ? foamAgree / foamCells : 1,
    foamMisMax, foamDiffMax, ageDiffMax, foamUnexplained,
    countsMatch, countDetail: { panel: [gpuCounts.breakingCount, gpuCounts.freshCount, gpuCounts.foamCount, gpuCounts.lingerCount, gpuCounts.recentCount], fields: [gBreaks, gFlags, gFoam, gLinger, gRecent] },
  };
}

// The ripple patches: GPU (gpuPatch.js) against chopPatch.js, as the renderer sees them.
function keptPerLevel(slopes, n) {
  // view.js's keptPerLevel, verbatim in effect: slope variance each 2 x 2 box mip level keeps.
  const kept = new Float64Array(8);
  let level = Float64Array.from(slopes);
  let size = n;
  let base = 0;
  for (let L = 0; L < 8; L += 1) {
    let v = 0;
    for (let i = 0; i < level.length; i += 1) v += level[i] * level[i];
    v /= size * size;
    if (L === 0) base = v || 1;
    kept[L] = size >= 1 ? v / base : 0;
    if (size === 1) break;
    const half = size >> 1;
    const next = new Float64Array(half * half * 4);
    for (let z = 0; z < half; z += 1) for (let x = 0; x < half; x += 1) for (let c = 0; c < 4; c += 1) {
      const a = ((2 * z) * size + 2 * x) * 4 + c;
      next[(z * half + x) * 4 + c] = 0.25 * (level[a] + level[a + 4] + level[a + size * 4] + level[a + size * 4 + 4]);
    }
    level = next;
    size = half;
  }
  return kept;
}
function comparePatch(cpu, gpuState, gpuRead) {
  if (!cpu || !gpuRead) return null;
  const n = cpu.n;
  let slopeMax = 0, sq = 0, byteDiff = 0, byteMax = 0;
  const k = 127 / 0.6;
  for (let i = 0; i < n * n * 4; i += 1) {
    slopeMax = Math.max(slopeMax, Math.abs(cpu.slopes[i] - gpuRead.slopes[i]));
    sq += cpu.slopes[i] * cpu.slopes[i];
    let v = Math.round(cpu.slopes[i] * k) + 128;
    v = v < 1 ? 1 : v > 255 ? 255 : v;
    const d = Math.abs(v - gpuRead.bytes[i]);
    if (d) { byteDiff += 1; byteMax = Math.max(byteMax, d); }
  }
  const keptCpu = keptPerLevel(cpu.slopes, n);
  const base = gpuRead.levelVar[0] || 1;
  let keptMax = 0;
  for (let L = 0; L < 8; L += 1) keptMax = Math.max(keptMax, Math.abs(keptCpu[L] - gpuRead.levelVar[L] / base));
  return {
    slopeMax, slopeRms: Math.sqrt(sq / (n * n * 4)), byteDiffShare: byteDiff / (n * n * 4), byteMax, keptMax,
    rmsRel: Math.abs(gpuState.rmsSlope - cpu.rmsSlope) / (cpu.rmsSlope || 1),
    modesSame: gpuState.modes === cpu.modes, varianceRel: Math.abs(gpuState.variance - cpu.variance) / (cpu.variance || 1),
  };
}

async function runParityCpuV27(gl, { speeds = [0, 5, 10, 17.5, 25], seconds = 6, dt = 0.05, checkEvery = 0.05, gust = "continuous", size = 420, choppiness = 2.5, depth = 40 } = {}) {
  const runs = [];
  const problems = [];
  for (const speed of speeds) {
    const sc = JSON.parse(JSON.stringify(baseScenario));
    sc.size = size;
    sc.choppiness = choppiness;
    sc.reachM = crestReachAt(sc, speed); sc.paritySpeed = speed; // the face reach both surfaces use at this wind
    const cpuWind = createWind({ direction: Math.PI / 2, speed, gust });
    const gpuWind = createWind({ direction: Math.PI / 2, speed, gust });
    const cpu = createSurface(sc, cpuWind);
    const gpu = createGpuSurface(gl, sc, gpuWind);
    const checkpoints = [];
    const history = new Float64Array(sc.resolution * sc.resolution).fill(-Infinity);
    let windJump = 0;
    let prevWind = null;
    const steps = Math.round(seconds / dt);
    const every = Math.max(1, Math.round(checkEvery / dt));
    let t = 0;
    for (let s = 1; s <= steps; s += 1) {
      t += dt;
      const cs = cpu.step(t, dt, depth);
      const gs = gpu.step(t, dt, depth);
      // Wind continuity: the bands' lagged winds (the band heights follow them) per step.
      const hs = gs.bands.map((b) => b.significantHeight);
      if (prevWind) windJump = Math.max(windJump, ...hs.map((h, k) => Math.abs(h - prevWind[k])));
      prevWind = hs;
      if (s % every !== 0) continue;
      const counts = gpu.countsNow();
      const c = compare(cs, gpu.readFields(), counts, sc, history, t);
      const rp = gpu.readPatches();
      c.patches = { chop: comparePatch(cs.chopPatch, gs.chopPatch, rp.chop), fine: comparePatch(cs.finePatch, gs.finePatch, rp.fine) };
      c.time = Math.round(t * 100) / 100;
      checkpoints.push(c);
      await new Promise((r) => setTimeout(r, 0));
    }
    const worst = (key, f = Math.max) => checkpoints.reduce((a, c) => f(a, c[key]), f === Math.max ? -Infinity : Infinity);
    const sum = (key) => checkpoints.reduce((a, c) => a + c[key], 0);
    const r = {
      speed, gust, checkpoints: checkpoints.length, seconds,
      maxHeightMm: worst("maxHeight") * 1000, maxShiftMm: worst("maxShift") * 1000, maxSlope: worst("maxSlope"), maxNormal: worst("maxNormal"),
      breakAgreeMin: worst("breakAgree", Math.min), breakMismatch: sum("breakMismatch"), breakOffEdge: sum("breakOffEdge"), breakCalm: sum("breakCalm"), breakEdgeWorst: worst("breakEdgeWorst"),
      flagAgreeMin: worst("flagAgree", Math.min), flagMismatch: sum("flagMismatch"), flagOffEdge: sum("flagOffEdge"), flagCalm: sum("flagCalm"), flagEdgeWorst: worst("flagEdgeWorst"), gradeMismatch: sum("gradeMismatch"), aheadMax: worst("aheadMax"),
      foamUnexplained: sum("foamUnexplained"), foamMisMax: worst("foamMisMax"), foamDiffMax: worst("foamDiffMax"), ageDiffMax: worst("ageDiffMax"),
      foamNotFlag: sum("foamNotFlag"), flagNotFoam: sum("flagNotFoam"), foamOutside: sum("foamOutside"), foamAgreeMin: worst("foamAgree", Math.min),
      countsMatch: checkpoints.every((c) => c.countsMatch),
      countBad: checkpoints.filter((c) => !c.countsMatch).slice(0, 3).map((c) => ({ t: c.time, ...c.countDetail })), countBadN: checkpoints.filter((c) => !c.countsMatch).length,
      maxHsStepM: windJump,
      patch: ["chop", "fine"].map((which) => {
        const all = checkpoints.map((c) => c.patches[which]).filter(Boolean);
        const w = (key) => Math.max(...all.map((p) => p[key]));
        return { which, slopeMax: w("slopeMax"), slopeRms: w("slopeRms"), byteDiffShare: w("byteDiffShare"), byteMax: w("byteMax"), keptMax: w("keptMax"), rmsRel: w("rmsRel"), varianceRel: w("varianceRel"), modesSame: all.every((p) => p.modesSame) };
      }),
      last: checkpoints[checkpoints.length - 1],
    };
    const tag = `${speed} m/s`;
    if (!(r.maxHeightMm <= HEIGHT_TOL * 1000)) problems.push(`${tag}: heights differ by ${r.maxHeightMm.toFixed(3)} mm`);
    if (!(r.breakAgreeMin >= 0.995)) problems.push(`${tag}: break flags agree on ${(100 * r.breakAgreeMin).toFixed(3)}%`);
    if (!(r.flagAgreeMin >= 0.995)) problems.push(`${tag}: painted flags agree on ${(100 * r.flagAgreeMin).toFixed(3)}%`);
    if (r.breakOffEdge || r.flagOffEdge) problems.push(`${tag}: ${r.breakOffEdge + r.flagOffEdge} mismatches not at a threshold edge`);
    if (r.breakCalm || r.flagCalm) problems.push(`${tag}: ${r.breakCalm + r.flagCalm} mismatches in calm water`);
    if (r.foamNotFlag || r.flagNotFoam || r.foamOutside) problems.push(`${tag}: GPU foam off its flags (${r.foamNotFlag}/${r.flagNotFoam}/${r.foamOutside})`);
    if (r.foamUnexplained) problems.push(`${tag}: ${r.foamUnexplained} foam mismatches without a flag mismatch`);
    for (const p of r.patch) {
      if (!(p.slopeMax <= 1e-4)) problems.push(`${tag}: ${p.which} patch slopes differ by ${p.slopeMax.toExponential(2)}`);
      if (!(p.byteMax <= 1)) problems.push(`${tag}: ${p.which} patch texture differs by ${p.byteMax} levels`);
      if (!(p.keptMax <= 1e-4) || !(p.rmsRel <= 1e-9) || !(p.varianceRel <= 1e-12) || !p.modesSame) problems.push(`${tag}: ${p.which} patch stats differ`);
    }
    if (!r.countsMatch) problems.push(`${tag}: panel counts differ from the fields`);
    r.async = await asyncCounts(gl, sc, speed, gust, dt, depth);
    if (r.async.compared === 0) problems.push(`${tag}: async panel counts never landed`);
    if (!r.async.heldCaughtUp) problems.push(`${tag}: held page did not reach its own step's counts`);
    if (r.async.mismatch) problems.push(`${tag}: async panel counts differ from the synchronous counts at their step (${r.async.mismatch} of ${r.async.compared})`);
    runs.push(r);
  }
  const summary = (problems.length ? "GPU parity FAILED: " + problems.join("; ") : "GPU parity ok") + " | " + runs.map((r) =>
    `${r.speed} m/s: dh ${r.maxHeightMm.toFixed(4)} mm, breaks ${(100 * r.breakAgreeMin).toFixed(3)}% (${r.breakMismatch} mism, worst edge ${r.breakMismatch ? r.breakEdgeWorst.toExponential(1) : "-"}), flags ${(100 * r.flagAgreeMin).toFixed(3)}% (${r.flagMismatch}), last ${r.last.cpuBreaks}/${r.last.gpuBreaks}`).join("; ");
  return { ok: problems.length === 0, problems, runs, summary };
}

// ---------------------------------------------------------------------------------------
// v28 parity. The GPU sea is now a spectral (FFT) sea, which the v26 CPU physics does not
// have, so the old field-by-field CPU yardstick (runParityCpuV27, kept for reference) no
// longer applies. Per checkpoint this checks, on the whole 256^2 GPU grid (scenario.fft):
//  - the FFT against a CPU reference of the same FFT: the card's own spectrum (amplitude and
//    phase per lattice wave, read back) summed directly at sample cells, in double
//    precision, against the transformed height texture; plus the explicit long waves (the
//    step's own components, double precision) against the height the wave pass wrote;
//  - the break test at those cells: the break-slope and break-height fields (FFT part read
//    from its textures, explicit part summed on the CPU) through the threshold, against the
//    card's steep flag; any mismatch must sit within EDGE of the threshold;
//  - foam exactly on the flags and the panel counts (reduction) == the fields, over 256^2;
//  - the ripple patches against chopPatch.js built with the same band limits (the band
//    objects the card stepped with, so the same held edge where the FFT hands over).
const FFT_TOL = 2e-3; // metres: float32 FFT of a 5-15 m sea, about 1e-4 relative
// v28 async-count parity (ported from v27.2): the panel counts arrive a few steps late, tagged
// with countsStep. v28's GPU sea is not the CPU's, so the reference is a second GPU surface
// stepped in lockstep whose counts are read synchronously (countsNow) at every step.
async function asyncCounts(gl, sc, speed, gust, dt, depth, seconds = 2) {
  const ref = createGpuSurface(gl, sc, createWind({ direction: Math.PI / 2, speed, gust }));
  const gpu = createGpuSurface(gl, sc, createWind({ direction: Math.PI / 2, speed, gust }));
  const keys = ["breakingCount", "freshCount", "foamCount", "lingerCount", "recentCount"];
  const refAt = new Map(); const lags = [];
  const frame = () => new Promise((r) => (typeof requestAnimationFrame !== "undefined" ? requestAnimationFrame(() => r()) : setTimeout(r, 16)));
  let compared = 0, mismatch = 0, maxDiff = 0, gs = null, t = 0, step = 0;
  const check = (state) => {
    const c = state.countsStep >= 1 && refAt.get(state.countsStep);
    if (!c) return;
    compared += 1;
    const d = Math.max(...keys.map((k) => Math.abs(c[k] - state[k])));
    if (d) mismatch += 1;
    maxDiff = Math.max(maxDiff, d);
  };
  for (step = 1; step <= Math.round(seconds / dt); step += 1) {
    t += dt;
    ref.step(t, dt, depth);
    const rc = ref.countsNow();
    refAt.set(step, Object.fromEntries(keys.map((k) => [k, rc[k]])));
    gs = gpu.step(t, dt, depth);
    lags.push(gs.stepIndex - gs.countsStep);
    check(gs);
    await frame();
    while (gpu.backlog() > 2) await frame();
  }
  const last = step - 1, t0 = performance.now();
  while (gs.countsStep < last && performance.now() - t0 < 10000) { await new Promise((r) => setTimeout(r, 16)); gpu.refresh(gs); }
  const heldCaughtUp = gs.countsStep === last;
  check(gs);
  lags.sort((a, b) => a - b);
  ref.dispose?.(); gpu.dispose?.();
  return { compared, mismatch, maxDiff, heldCaughtUp, lagMin: lags[0], lagMedian: lags[lags.length >> 1], lagMax: lags[lags.length - 1], steps: last };
}

// v28 foam fix: the CPU break formula (surface.js mode 1) over a whole read-back height field,
// against the GPU break flags. A mismatch is "on the edge" when the cell's steepness is within
// AK_EDGE of the threshold, or its local-max test is a float tie (|fall| or |h - opposite| < 1e-5 m).
export const AK_EDGE = 1e-3;
export function breakFieldCheck(heightsIn, breaks, n, rule, spacing, long = null) {
  // mode 2 (v28 long-slope rule): the short-wave height (total minus the explicit long waves,
  // in float32 as the GPU) and the long wave's local slope added to the steepness.
  const useLong = rule.mode > 1.5 && long && long.h;
  let heights = heightsIn;
  if (useLong) { heights = new Float32Array(n * n); for (let i = 0; i < n * n; i += 1) heights[i] = Math.fround(heightsIn[i] - long.h[i]); }
  const HS = new Float32Array(n * n);
  for (let iz = 1; iz < n - 1; iz += 1) for (let ix = 1; ix < n - 1; ix += 1) {
    let t = 0;
    for (let dz = -1; dz <= 1; dz += 1) for (let dx = -1; dx <= 1; dx += 1) t += heights[(iz + dz) * n + ix + dx];
    HS[iz * n + ix] = t / 9;
  }
  const rowMean = new Float32Array(n * n), trough = new Float32Array(n * n);
  for (let iz = 0; iz < n; iz += 1) for (let ix = 0; ix < n; ix += 1) {
    const x0 = Math.max(ix - rule.rb, 0), x1 = Math.min(ix + rule.rb, n - 1);
    let t = 0; for (let x = x0; x <= x1; x += 1) t += heights[iz * n + x];
    rowMean[iz * n + ix] = t / (x1 - x0 + 1);
  }
  for (let iz = 0; iz < n; iz += 1) for (let ix = 0; ix < n; ix += 1) {
    const z0 = Math.max(iz - rule.rb, 0), z1 = Math.min(iz + rule.rb, n - 1);
    let t = 0; for (let z = z0; z <= z1; z += 1) t += rowMean[z * n + ix];
    trough[iz * n + ix] = t / (z1 - z0 + 1);
  }
  const steps = [[1, 0], [-1, 0], [0, 1], [0, -1]], m = rule.ts + 1;
  let cpu = 0, gpu = 0, mis = 0, onEdge = 0, offEdge = 0;
  for (let iz = 0; iz < n; iz += 1) for (let ix = 0; ix < n; ix += 1) {
    const i = iz * n + ix;
    let b = false, near = Infinity;
    if (ix >= m && iz >= m && ix < n - m && iz < n - m) {
      const h = heights[i];
      let drop = 0, dx = 0, dz = 0, minFall = Infinity;
      for (const [sx, sz] of steps) { const fl = h - heights[(iz + sz) * n + ix + sx]; minFall = Math.min(minFall, Math.abs(fl)); if (fl > drop) { drop = fl; dx = sx; dz = sz; } }
      const opp = h - heights[(iz - dz) * n + ix - dx];
      const tie = minFall < 1e-5 || Math.abs(opp) < 1e-5;
      const t = rule.ts, d = t * spacing, hs = HS[i];
      const lap = (HS[i - t] + HS[i + t] + HS[i - t * n] + HS[i + t * n] - 4 * hs) / (d * d);
      const eta = hs - trough[i];
      const ak = eta > 0 && lap < 0 ? Math.sqrt(-lap * eta) * (useLong ? 1 + (rule.M || 0) * Math.max(0, long.h[i] * (rule.kL || 0)) : 1) + (useLong ? (rule.lg ?? 1) * Math.hypot(long.sx[i], long.sz[i]) : 0) : 0;
      b = drop > 0 && opp >= 0 && ak >= rule.ak;
      near = tie ? 0 : Math.abs(ak - rule.ak);
    }
    const g = breaks[i] > 0;
    if (b) cpu += 1; if (g) gpu += 1;
    if (b !== g) { mis += 1; if (near <= AK_EDGE) onEdge += 1; else offEdge += 1; }
  }
  return { cells: n * n, cpu, gpu, mis, onEdge, offEdge, onEdgePct: (100 * onEdge) / (n * n) };
}

export async function runParity(gl, { speeds = [0, 5, 10, 17.5, 25], seconds = 3, dt = 0.05, checkEvery = 0.5, gust = "continuous", size = 420, choppiness = 2.5, depth = 40, samples = 192 } = {}) {
  const runs = [];
  const problems = [];
  for (const speed of speeds) {
    const sc = JSON.parse(JSON.stringify(baseScenario));
    sc.size = size;
    sc.choppiness = choppiness; sc.paritySpeed = speed;
    const gpu = createGpuSurface(gl, sc, createWind({ direction: Math.PI / 2, speed, gust }));
    const N = gpu.N;
    const chopSpec = sc.bands.find((b) => b.patch);
    const cpuChop = chopSpec ? createChopPatch(chopSpec) : null;
    const cpuFine = chopSpec && chopSpec.finePatch ? createChopPatch(chopSpec, chopSpec.finePatch) : null;
    const steps = Math.round(seconds / dt);
    const every = Math.max(1, Math.round(checkEvery / dt));
    const cps = [];
    let t = 0;
    for (let s = 1; s <= steps; s += 1) {
      t += dt;
      const gs = gpu.step(t, dt, depth);
      const ts = gpu.testState();
      const ci = ts.bands.findIndex((b) => b.spec.patch);
      const cc = cpuChop ? cpuChop.step(ts.bands[ci], ts.winds[ci], depth, GRAVITY, dt) : null;
      const cf = cpuFine ? cpuFine.step(ts.bands[ci], ts.winds[ci], depth, GRAVITY, dt, cc.kCut) : null;
      if (s % every !== 0) continue;
      const g = gpu.readFields();
      const counts = gpu.countsNow();
      const spec = gpu.readSea("spectrum");
      const T0 = gpu.readSea(0), T3 = gpu.readSea(3), T4 = gpu.readSea(4);
      const c = { time: Math.round(t * 100) / 100 };
      // FFT vs direct lattice sum, and the full height, at sample cells.
      let seed = 12345 + s;
      const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
      const modes = [];
      for (let p = 0; p < N * N; p += 1) {
        const re = spec[p * 4], im = spec[p * 4 + 1];
        if (re === 0 && im === 0) continue;
        const px = p % N, pz = (p - px) / N;
        modes.push([px < N / 2 ? px : px - N, pz < N / 2 ? pz : pz - N, re, im]);
      }
      let fftMax = 0, hMax = 0, brkMis = 0, brkOff = 0, brkEdge = 0, steepN = 0;
      const limit = sc.depthBreakRatio * depth;
      for (let k = 0; k < samples; k += 1) {
        const ix = Math.floor(rnd() * N), iz = Math.floor(rnd() * N), idx = iz * N + ix;
        let h = 0;
        for (const [lx, lz, re, im] of modes) {
          const th = (2 * Math.PI * (lx * ix + lz * iz)) / N;
          h += re * Math.sin(th) + im * Math.cos(th);
        }
        fftMax = Math.max(fftMax, Math.abs(h - T0[idx * 4 + 1]));
        let hx = 0, bh = 0, bx = 0, bz = 0;
        for (const q of ts.comps) {
          const x = -q.half + ix * q.spacing, z = -q.half + iz * q.spacing;
          const ph = q.k * (q.cx * x + q.cz * z) + q.phase;
          const sn = Math.sin(ph), co = Math.cos(ph);
          hx += q.a * sn; bh += q.ba * sn; bx += q.ba * q.k * q.cx * co; bz += q.ba * q.k * q.cz * co;
        }
        hMax = Math.max(hMax, Math.abs(T0[idx * 4 + 1] + hx - g.heights[idx]));
        const slope = Math.hypot(T4[idx * 4] + bx, T4[idx * 4 + 1] + bz);
        const bhT = T3[idx * 4 + 3] + bh;
        const steep = slope >= sc.breakSlope || (bhT > 0 && bhT >= limit);
        if (steep) steepN += 1;
        if (steep !== g.steep[idx] > 0.5) {
          brkMis += 1;
          const m = Math.min(Math.abs(slope - sc.breakSlope), bhT > 0 ? Math.abs(bhT - limit) : Infinity);
          brkEdge = Math.max(brkEdge, m);
          if (m > EDGE) brkOff += 1;
        }
      }
      Object.assign(c, { fftMaxMm: fftMax * 1000, heightMaxMm: hMax * 1000, brkMis, brkOff, brkEdge, steepN });
      // v28 foam fix: the break rule itself, CPU formula over the full N^2 read-back heights
      { const sp = sc.size / N; const fb = breakFieldCheck(g.heights, g.breaks, N, breakRuleAt(sc, speed, sp), sp, { h: g.xHeights, sx: g.xSlopeX, sz: g.xSlopeZ });
        Object.assign(c, { fullMis: fb.mis, fullOff: fb.offEdge, fullEdgePct: fb.onEdgePct, fullBreaks: fb.gpu }); }
      // Foam on flags, counts == fields, over the whole grid.
      let foamNotFlag = 0, flagNotFoam = 0, foamOutside = 0, gB = 0, gF = 0, gFoam = 0, gL = 0, gR = 0;
      for (let i = 0; i < N * N; i += 1) {
        const flagged = g.flags[i] > 0;
        if (g.freshFoam[i] > 0 && !flagged) foamNotFlag += 1;
        if (flagged && !(g.freshFoam[i] > 0)) flagNotFoam += 1;
        if (g.foam[i] > 0 && !flagged && !(g.recent[i] > 0)) foamOutside += 1;
        if (g.breaks[i] > 0) gB += 1;
        if (flagged) gF += 1;
        if (g.foam[i] > 0) { gFoam += 1; if (!flagged) gL += 1; }
        if (g.recent[i] > 0) gR += 1;
      }
      Object.assign(c, { foamNotFlag, flagNotFoam, foamOutside,
        countsMatch: counts.breakingCount === gB && counts.freshCount === gF && counts.foamCount === gFoam && counts.lingerCount === gL && counts.recentCount === gR,
        counts: [counts.breakingCount, counts.freshCount, counts.foamCount], fields: [gB, gF, gFoam] });
      const rp = gpu.readPatches();
      c.patches = { chop: comparePatch(cc, gs.chopPatch, rp.chop), fine: comparePatch(cf, gs.finePatch, rp.fine) };
      cps.push(c);
      await new Promise((r) => setTimeout(r, 0));
    }
    const mx = (k) => Math.max(...cps.map((c) => c[k]));
    const sm = (k) => cps.reduce((a, c) => a + c[k], 0);
    const r = { speed, checkpoints: cps.length, N, fftMaxMm: mx("fftMaxMm"), heightMaxMm: mx("heightMaxMm"), brkMis: sm("brkMis"), brkOff: sm("brkOff"), brkEdge: mx("brkEdge"), steepSampled: sm("steepN"), fullMis: sm("fullMis"), fullOff: sm("fullOff"), fullEdgePct: mx("fullEdgePct"), fullBreaks: sm("fullBreaks"), samples: samples * cps.length,
      foamNotFlag: sm("foamNotFlag"), flagNotFoam: sm("flagNotFoam"), foamOutside: sm("foamOutside"), countsMatch: cps.every((c) => c.countsMatch), last: cps[cps.length - 1],
      patch: ["chop", "fine"].map((which) => {
        const all = cps.map((c) => c.patches[which]).filter(Boolean);
        const w = (key) => Math.max(...all.map((p) => p[key]));
        return all.length ? { which, slopeMax: w("slopeMax"), byteMax: w("byteMax"), keptMax: w("keptMax"), rmsRel: w("rmsRel"), varianceRel: w("varianceRel"), modesSame: all.every((p) => p.modesSame) } : { which, none: true };
      }) };
    const tag = `${speed} m/s`;
    if (!(r.fftMaxMm <= FFT_TOL * 1000)) problems.push(`${tag}: FFT vs direct sum ${r.fftMaxMm.toFixed(3)} mm`);
    if (!(r.heightMaxMm <= HEIGHT_TOL * 1000)) problems.push(`${tag}: explicit waves on the FFT differ by ${r.heightMaxMm.toFixed(3)} mm`);
    if (r.brkOff) problems.push(`${tag}: ${r.brkOff} break-test mismatches not at the threshold`);
    if (r.fullOff) problems.push(`${tag}: ${r.fullOff} break-rule mismatches (full grid) off the threshold edge`);
    if (r.fullEdgePct > 0.5) problems.push(`${tag}: break-rule mismatches on the edge ${r.fullEdgePct.toFixed(3)}% > 0.5%`);
    if (r.foamNotFlag || r.flagNotFoam || r.foamOutside) problems.push(`${tag}: GPU foam off its flags (${r.foamNotFlag}/${r.flagNotFoam}/${r.foamOutside})`);
    if (!r.countsMatch) problems.push(`${tag}: panel counts differ from the fields`);
    for (const p of r.patch) {
      if (p.none) continue;
      if (!(p.slopeMax <= 1e-4)) problems.push(`${tag}: ${p.which} patch slopes differ by ${p.slopeMax.toExponential(2)}`);
      if (!(p.byteMax <= 1)) problems.push(`${tag}: ${p.which} patch texture differs by ${p.byteMax} levels`);
      if (!(p.keptMax <= 1e-4) || !(p.rmsRel <= 1e-6) || !(p.varianceRel <= 1e-9) || !p.modesSame) problems.push(`${tag}: ${p.which} patch stats differ`);
    }
    r.async = await asyncCounts(gl, sc, speed, gust, dt, depth);
    if (r.async.compared === 0) problems.push(`${tag}: async panel counts never landed`);
    if (!r.async.heldCaughtUp) problems.push(`${tag}: held page did not reach its own step's counts`);
    if (r.async.mismatch) problems.push(`${tag}: async panel counts differ from the synchronous counts at their step (${r.async.mismatch} of ${r.async.compared})`);
    runs.push(r);
  }
  const summary = (problems.length ? "GPU parity FAILED: " + problems.join("; ") : "GPU parity ok") + " | " + runs.map((r) =>
    `${r.speed} m/s: FFT ${r.fftMaxMm.toFixed(4)} mm, heights ${r.heightMaxMm.toFixed(4)} mm, break test ${r.brkMis} mism (${r.brkOff} off edge) of ${r.samples} (${r.steepSampled} steep), break rule full grid ${r.fullMis} mism (${r.fullOff} off edge) over ${r.checkpoints} x ${r.N}^2 (${r.fullBreaks} breaks), counts ${r.countsMatch ? "=" : "!="} fields (${r.last.counts.join("/")}), async ${r.async.compared - r.async.mismatch}/${r.async.compared} (lag ${r.async.lagMin}-${r.async.lagMax}, held ${r.async.heldCaughtUp ? "caught up" : "STALE"}), chop patch ${r.patch[0].slopeMax?.toExponential(1)}`).join("; ");
  return { ok: problems.length === 0, problems, runs, summary };
}

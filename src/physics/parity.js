// CPU-vs-GPU parity (tests only; reads whole fields back, which the app never does).
// The v26 CPU physics (surface.js) is the yardstick: the same scenario, the same wind, the
// same steps, side by side. Per checkpoint:
//  - heights within 1 mm at every grid point;
//  - break flags (the break test, and the painted flags the panel and foam use) matching on
//    at least 99.5% of points, every mismatch at a threshold edge (some comparison that
//    decides it within EDGE of flipping), none in calm water (no CPU break within 6 cells);
//  - foam matching the flags exactly on the card: fresh foam == flagged, foam only where
//    flagged now or in the window, and the panel counts (reduction) == the fields.
import { scenario as baseScenario } from "./scenario.js?v=27.2";
import { createWind } from "./wind.js";
import { createSurface } from "./surface.js?v=27.2";
import { createGpuSurface } from "./gpuSurface.js?v=27.2";

const EDGE = 2e-3; // metres (heights) or slope units: an order above float32 noise here
const HEIGHT_TOL = 1e-3;

function edgeMargin(cpu, sc, index) {
  // How close the break test at this sample comes to flipping, on the CPU's own fields.
  const { n, breakHeights: bh, breakSlopeX, breakSlopeZ } = cpu;
  const ix = index % n;
  const iz = (index - ix) / n;
  const at = (x, z) => bh[z * n + x];
  const spacing = sc.size / Math.max(n - 1, 1);
  const reach = Math.max(1, Math.round(sc.crestReach / spacing));
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
  const reach = Math.max(1, Math.round(sc.crestReach / spacing)) + 3;
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
        const flipped = time - history[i] <= sc.foamFadeSeconds + 1e-6;
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
    countsMatch,
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

// v27.1: the panel's own path (pixel-pack buffer + fence, no countsNow): every state's
// counts are those of step countsStep; compare them with the CPU's counts at that step.
// Steps are paced like frames (a macrotask between them). Then hold (no steps, refresh()
// as a paused page does) until the counts reach the last step.
async function asyncCounts(gl, sc, speed, gust, dt, depth, seconds = 2) {
  const cpu = createSurface(sc, createWind({ direction: Math.PI / 2, speed, gust }));
  const gpu = createGpuSurface(gl, sc, createWind({ direction: Math.PI / 2, speed, gust }));
  const keys = ["breakingCount", "freshCount", "foamCount", "lingerCount", "recentCount"];
  const cpuAt = new Map();
  const lags = [];
  const lagMs = [];
  const issued = new Map();
  const frame = () => new Promise((r) => (typeof requestAnimationFrame !== "undefined" ? requestAnimationFrame(() => r()) : setTimeout(r, 16)));
  let compared = 0, mismatch = 0, maxDiff = 0, gs = null, t = 0, step = 0;
  const check = (state) => {
    if (state.countsStep < 1) return;
    const c = cpuAt.get(state.countsStep);
    if (!c) return;
    compared += 1;
    const d = Math.max(...keys.map((k) => Math.abs(c[k] - state[k])));
    if (d) mismatch += 1;
    maxDiff = Math.max(maxDiff, d);
  };
  for (step = 1; step <= Math.round(seconds / dt); step += 1) {
    t += dt;
    const cs = cpu.step(t, dt, depth);
    cpuAt.set(step, Object.fromEntries(keys.map((k) => [k, cs[k]])));
    issued.set(step, performance.now());
    gs = gpu.step(t, dt, depth);
    lags.push(gs.stepIndex - gs.countsStep);
    if (issued.has(gs.countsStep)) lagMs.push(performance.now() - issued.get(gs.countsStep));
    check(gs);
    await frame(); // paced like the app (one step per display frame)
    while (gpu.backlog() > 2) await frame(); // and held back like the app (main.js MAX_BACKLOG)
  }
  const last = step - 1;
  const t0 = performance.now();
  while (gs.countsStep < last && performance.now() - t0 < 10000) {
    await new Promise((r) => setTimeout(r, 16));
    gpu.refresh(gs);
  }
  const heldCaughtUp = gs.countsStep === last;
  check(gs);
  lags.sort((a, b) => a - b);
  lagMs.sort((a, b) => a - b);
  // The break test may differ at a threshold edge (see the field parity): allow those.
  return { compared, mismatch, maxDiff, edgeAllowance: Math.ceil(compared * 0.05), heldCaughtUp, lagMin: lags[0], lagMedian: lags[lags.length >> 1], lagMax: lags[lags.length - 1], lagMsMedian: Math.round(lagMs[lagMs.length >> 1]), lagMsMax: Math.round(lagMs[lagMs.length - 1]), steps: last };
}

export async function runParity(gl, { speeds = [0, 5, 10, 17.5, 25], seconds = 6, dt = 0.05, checkEvery = 0.05, gust = "continuous", size = 420, choppiness = 2.5, depth = 40 } = {}) {
  const runs = [];
  const problems = [];
  for (const speed of speeds) {
    const sc = JSON.parse(JSON.stringify(baseScenario));
    sc.size = size;
    sc.choppiness = choppiness;
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
      maxHsStepM: windJump,
      patch: ["chop", "fine"].map((which) => {
        const all = checkpoints.map((c) => c.patches[which]).filter(Boolean);
        const w = (key) => Math.max(...all.map((p) => p[key]));
        return { which, slopeMax: w("slopeMax"), slopeRms: w("slopeRms"), byteDiffShare: w("byteDiffShare"), byteMax: w("byteMax"), keptMax: w("keptMax"), rmsRel: w("rmsRel"), varianceRel: w("varianceRel"), modesSame: all.every((p) => p.modesSame) };
      }),
      last: checkpoints[checkpoints.length - 1],
    };
    r.async = await asyncCounts(gl, sc, speed, gust, dt, depth);
    const tag = `${speed} m/s`;
    if (r.async.compared === 0) problems.push(`${tag}: async panel counts never landed`);
    if (!r.async.heldCaughtUp) problems.push(`${tag}: held page did not reach its own step's counts`);
    if (r.async.mismatch > r.async.edgeAllowance) problems.push(`${tag}: async panel counts differ from the CPU's at their step (${r.async.mismatch} of ${r.async.compared})`);
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
    runs.push(r);
  }
  const summary = (problems.length ? "GPU parity FAILED: " + problems.join("; ") : "GPU parity ok") + " | " + runs.map((r) =>
    `${r.speed} m/s: dh ${r.maxHeightMm.toFixed(4)} mm, breaks ${(100 * r.breakAgreeMin).toFixed(3)}% (${r.breakMismatch} mism, worst edge ${r.breakMismatch ? r.breakEdgeWorst.toExponential(1) : "-"}), flags ${(100 * r.flagAgreeMin).toFixed(3)}% (${r.flagMismatch}), last ${r.last.cpuBreaks}/${r.last.gpuBreaks}, async counts ${r.async.compared - r.async.mismatch}/${r.async.compared} = CPU (max diff ${r.async.maxDiff}), lag ${r.async.lagMin}-${r.async.lagMax} steps (median ${r.async.lagMedian}), held ${r.async.heldCaughtUp ? "caught up" : "STALE"}`).join("; ");
  return { ok: problems.length === 0, problems, runs, summary };
}

import { GRAVITY } from "./gravity.js";
import { createBand, retuneBand, sampleBand, bandVariance } from "./band.js?v=28";
import { createChopPatch } from "./chopPatch.js?v=28";
import { slopeBudget } from "./roughness.js?v=28";
import { fftShare, FFT_SPLIT } from "./fftSea.js?v=28";

// Energy-weighted mean travel direction of every wave component, as a unit (x, z) vector.
// A calm sea has no energy, so it falls back to the wind direction.
export function travelDirection(bands, fallback) {
  let vx = 0;
  let vz = 0;
  for (const band of bands) {
    for (const c of band.components) {
      const energy = c.amplitude * c.amplitude;
      vx += energy * Math.cos(c.direction);
      vz += energy * Math.sin(c.direction);
    }
  }
  const len = Math.hypot(vx, vz);
  if (len < 1e-12) return [Math.cos(fallback), Math.sin(fallback)];
  return [vx / len, vz / len];
}

// Value held by samples without foam: a few metres, so the near band tapers over
// roughly half a cell toward them (torn, tapered ends) instead of stopping square.
export const NO_FOAM_AHEAD = 3;

// Translucency of thin crests (metres): how far below the tip the crest thickness is
// taken, the light's attenuation length through it, and the radius of the local mean the
// crest must stand above. Shading only; breaking and foam never see it.
// Each sample's depth is measured below its crest's apex (slope^2 / (2 kappa) for the local
// parabola), plus the lip's own skin; the crest's horizontal thickness there is
// t = 2 sqrt(2 depth / kappa) and light keeps exp(-t / length) (Beer-Lambert), cut to 0
// by maxThickness. So only the top lip and thin, steep upper parts pass light; a big
// smooth face, metres below its crest and thick, passes none.
export const TRANSLUCENCY = { skin: 0.05, length: 1.0, maxThickness: 2.0, meanRadius: 8 };

// v28 break fix: the face reach (crest to steep face) follows the breaking waves' size.
// A fixed 4 m reach (v26-v27) suits 17.5 m/s; at 25 m/s the crests are broad and the steep
// face lies further below them, so almost no crest found its face (6 breaks at 25 vs 182 at
// 17.5). Above 17.5 m/s the reach grows like the wind sea's wavelength (U^2); at or below
// 17.5 m/s it is exactly the scenario's crestReach (unchanged).
export function crestReachAt(scenario, speed) {
  return scenario.crestReach * Math.max(1, (speed / 17.5) ** 2);
}

// v28 foam fix (GAD): break test on the total height against each wave's own steepness.
// mode 1: a crest (local max of the height) breaks when a k >= ak, with a = elevation of the
// 3x3-averaged height above the mean over the trough window and k the cell's dominant
// wavenumber from the crest curvature (k^2 = -h''/a, curvature over +-ts cells).
// The trough window is half of the wind sea's peak wavelength (PM: kp = 0.77 g / U^2),
// 8-320 m; the curvature stencil round(window / 20 m), 2-4 cells. mode 0 = the v27 rule (break-band height + steep face within reach).
export const BREAK_RULE = { M: 1, ak: 0.18, minWindow: 8, maxWindow: 320, windowFraction: 0.5 };
// GAD 04:15: foam lives about as long as its breaker's period. The breaking crests of a
// fully developed sea are about equally steep at every wind (Hs kp / 2 is wind-independent),
// so they get sparser per area as the waves grow (~1 / lambda_p^2); bigger breakers leave
// foam that lasts longer: foamFadeSeconds (2 s) up to 10 m/s, rising linearly to 3x (6 s)
// at 25 m/s (GAD 06:27). ?foamLife=0 keeps the fixed window.
export function foamWindowAt(scenario, speed) {
  const q = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
  const base = Math.max(scenario.foamFadeSeconds || 0, 1e-6);
  if (q.get("foamLife") === "0" || !(base > 1e-6)) return base;
  // GAD 06:27: 2 s up to 10 m/s, linear to 6 s at 25 m/s (3x), held above 25.
  return base * (1 + 2 * Math.min(1, Math.max(0, (speed - 10) / 15)));
}
// Michael 20:49: each break's foam runs forward down its front face for 0.1 x the full-spectrum
// peak wavelength (PM: 2 pi U^2 / (0.77 g)), in cells; it still stops where the face stops
// dropping. Clamped to 1..FACE_MAX (40 cells, ~66 m at 420/256). ?faceScale overrides 0.1.
export const FACE_MAX = 40;
export function faceStepsAt(scenario, speed, spacing) {
  const q = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
  const lambdaP = speed > 0.05 ? (2 * Math.PI * speed * speed) / (0.77 * 9.81) : 0;
  return Math.min(FACE_MAX, Math.max(1, Math.round((Number(q.get("faceScale") ?? 0.1) * lambdaP) / spacing)));
}
// Grade down the face: the old 4-step shape (1, 0.5, 0.2, 0.06) stretched over F steps.
export function faceGrade(s, F) {
  const G = [1, 0.5, 0.2, 0.06];
  const u = (3 * s) / Math.max(F, 1);
  const i = Math.floor(u);
  if (i >= 3) return G[3];
  return G[i] + (G[i + 1] - G[i]) * (u - i);
}
export function breakRuleAt(scenario, speed, spacing) {
  const q = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
  const lambdaP = speed > 0.05 ? (2 * Math.PI * speed * speed) / (0.77 * 9.81) : 0;
  const mode = Number(q.get("brk") ?? 2);
  // GAD 20:05 (b): mode 2 tests the short field (the FFT tile's waves), so the window and
  // stencil follow the short field's own dominant length: the peak wavelength, but never
  // longer than the FFT cutoff (the middle of the explicit -> FFT hand-over, size / 4).
  const lambdaCut = scenario.size / ((FFT_SPLIT[0] + FFT_SPLIT[1]) / 2);
  const lambdaD = mode > 1.5 ? Math.min(lambdaP, lambdaCut) : lambdaP;
  const win = Number(q.get("brkWin") ?? Math.min(BREAK_RULE.maxWindow, Math.max(BREAK_RULE.minWindow, lambdaD * Number(q.get("brkWf") ?? BREAK_RULE.windowFraction))));
  // curvature stencil grows with the window (the dominant wave's scale): 2-4 cells
  const ts = Number(q.get("brkTs") ?? Math.min(4, Math.max(1, Math.round(win / 20))));
  // The scenario's break slope scales the threshold (0.2 = BREAK_RULE.ak; Infinity switches breaking off).
  return { lg: Number(q.get("brkLg") ?? 1), M: Number(q.get("brkM") ?? BREAK_RULE.M), kL: lambdaP > 0 ? (2 * Math.PI) / lambdaP : 0, mode, ak: Number(q.get("brkAk") ?? BREAK_RULE.ak) * (scenario.breakSlope / 0.2), rb: Math.max(1, Math.round(win / spacing)), ts };
}

export function createSurface(scenario, wind) {
  const n = scenario.resolution;
  const heights = new Float32Array(n * n);
  const slopeX = new Float32Array(n * n);
  const slopeZ = new Float32Array(n * n);
  // What the break test sees: each band's waves as its spec.breakTest says (see band.js).
  // The chop band takes part at its Pierson-Moskowitz level, so a steeper chop spectrum
  // does not spread foam across the sea and the spectrum switch leaves breaking alone.
  // Breaking, the crest and face search and foam all use these; the drawn surface,
  // its normals and the crest glow (lips) use the full sum above.
  const breakHeights = new Float32Array(n * n);
  const breakSlopeX = new Float32Array(n * n);
  const breakSlopeZ = new Float32Array(n * n);
  // v28 long-slope rule: the explicit long waves alone (the part the GPU tile cannot hold,
  // amplitude x sqrt(1 - fftShare)), their slope, and the short-wave height (total - long).
  const xHeights = new Float32Array(n * n);
  const xSlopeX = new Float32Array(n * n);
  const xSlopeZ = new Float32Array(n * n);
  const shortHeights = new Float32Array(n * n);
  const breaks = new Float32Array(n * n);
  const lips = new Float32Array(n * n);
  const freshFoam = new Float32Array(n * n);
  const flags = new Float32Array(n * n);
  // Metres each foam sample sits ahead of the crest it was painted from, measured
  // along the travel direction between the displaced positions. Samples without foam
  // hold NO_FOAM_AHEAD, so interpolating toward them fades the band out. The shader
  // uses this to hold the band at a fixed width in metres up close.
  const freshAhead = new Float32Array(n * n);
  // Lengthwise position in a foam run along the crest: 1 with foam on both sides along
  // the crest, 0 at the last sample of a run of three or more, 0.5 for a run of two or
  // a lone sample, 0 without foam. Interpolated, it fades the near band out inside the last foam cell.
  const freshRuns = new Float32Array(n * n);
  // Lingering foam. The arrays above hold this step's foam (fresh). These keep, per
  // sample, the seconds since it was last flagged and what it carried at that moment,
  // so foam stays where it was laid and fades over scenario.foamFadeSeconds.
  const ages = new Float32Array(n * n).fill(Infinity);
  const keptGrade = new Float32Array(n * n);
  const keptAhead = new Float32Array(n * n).fill(NO_FOAM_AHEAD);
  const keptRun = new Float32Array(n * n);
  // What is drawn: strength (grade x fade), the kept grade and the fade on their own,
  // and the kept ahead/run values. recent = flagged in the window but not now.
  const foam = new Float32Array(n * n);
  const foamGrade = new Float32Array(n * n);
  const foamFade = new Float32Array(n * n);
  // Age as a fraction of the window for foam carried from earlier steps (0 for foam
  // flagged now and where there is none). The shader uses it to stop drawing old foam
  // as a crest band: once the crest has moved on there is no lip left to draw.
  const foamAge = new Float32Array(n * n);
  const ahead = new Float32Array(n * n);
  const runs = new Float32Array(n * n);
  const recent = new Float32Array(n * n);
  const steep = new Uint8Array(n * n);
  const shifts = new Float32Array(n * n * 2);
  // Jacobian of the horizontal displacement (below 1 where choppy crests are squeezed)
  // and the light-transmission weight of thin crests (see TRANSLUCENCY).
  const jacobians = new Float32Array(n * n).fill(1);
  const thins = new Float32Array(n * n);
  const normals = new Float32Array(n * n * 3);
  const bands = scenario.bands.map(createBand);
  // A band may own a fine patch for the wavelengths the big grid cannot hold (the
  // short chop does). It is fed only that band's own state and wind sample.
  const patches = bands.map((band) => (band.spec.patch ? createChopPatch(band.spec) : null));
  // And a second, finer patch that starts where the first one's cutoff stops.
  const finePatches = bands.map((band) => (band.spec.patch && band.spec.finePatch ? createChopPatch(band.spec, band.spec.finePatch) : null));

  function step(time, dt, depth) {
    const winds = bands.map((band) => wind.sample(band.spec, time, dt));
    bands.forEach((band, i) => retuneBand(band, winds[i], depth, GRAVITY, dt, scenario.size / Math.max(n - 1, 1)));
    // Fine patches are for shading only. Break detection and foam stay on the big grid.
    const patchStates = patches.map((patch, i) => (patch ? patch.step(bands[i], winds[i], depth, GRAVITY, dt) : null));
    const chopPatch = patchStates.find((p) => p) || null;
    const fineStates = finePatches.map((patch, i) => (patch ? patch.step(bands[i], winds[i], depth, GRAVITY, dt, patchStates[i].kCut) : null));
    const finePatch = fineStates.find((p) => p) || null;
    // Slope budget at the live wind (the chop band's lagged sample, gusts included): the
    // renderer spreads the reflected light by the sub-grid part (roughness.js).
    const chopIndex = bands.findIndex((band) => band.spec.patch);
    const roughness = slopeBudget(bands, chopPatch, finePatch, chopIndex >= 0 ? winds[chopIndex].speed : wind.speed);

    const half = scenario.size / 2;
    const denom = Math.max(n - 1, 1);
    const spacingGrid = scenario.size / denom;
    let breakingCount = 0;
    const dkTile = (2 * Math.PI) / scenario.size;
    const longComps = [];
    for (const band of bands) for (const c of band.components) {
      const sh = Math.sqrt(Math.max(0, 1 - fftShare(c.k, dkTile)));
      if (c.amplitude === 0 || sh === 0) continue;
      longComps.push({ a: c.amplitude * sh, k: c.k, cx: Math.cos(c.direction), cz: Math.sin(c.direction), phase: c.phase });
    }
    let maxSlope = 0;

    for (let iz = 0; iz < n; iz += 1) {
      const z = -half + (iz / denom) * scenario.size;
      for (let ix = 0; ix < n; ix += 1) {
        const x = -half + (ix / denom) * scenario.size;
        let height = 0;
        let dhdx = 0;
        let dhdz = 0;
        let shiftX = 0;
        let shiftZ = 0;
        let squeezeX = 0;
        let squeezeZ = 0;
        let squeezeXZ = 0;
        let chopDx = 0;
        let chopDz = 0;
        let bh = 0;
        let bdx = 0;
        let bdz = 0;
        for (const band of bands) {
          const sample = sampleBand(band, x, z);
          height += sample.height;
          dhdx += sample.dhdx;
          dhdz += sample.dhdz;
          bh += sample.breakHeight;
          bdx += sample.breakDx;
          bdz += sample.breakDz;
          shiftX += sample.shiftX;
          shiftZ += sample.shiftZ;
          squeezeX += sample.squeezeX;
          squeezeZ += sample.squeezeZ;
          squeezeXZ += sample.squeezeXZ;
          if (band.spec.id === "chop") {
            chopDx += sample.dhdx;
            chopDz += sample.dhdz;
          }
        }
        const chop = scenario.choppiness;
        const jxx = 1 - chop * squeezeX;
        const jzz = 1 - chop * squeezeZ;
        const jxz = -chop * squeezeXZ;
        // Ripples count again in the shading normal only, so they catch the sun
        // without becoming a second surface.
        const sx = dhdx + chopDx;
        const sz = dhdz + chopDz;
        const nx = sz * jxz - sx * jzz;
        const ny = jxx * jzz - jxz * jxz;
        const nz = jxz * sx - sz * jxx;
        const len = Math.hypot(nx, ny, nz) || 1;
        const index = iz * n + ix;
        jacobians[index] = ny;
        shifts[index * 2] = chop * shiftX;
        shifts[index * 2 + 1] = chop * shiftZ;
        normals[index * 3] = nx / len;
        normals[index * 3 + 1] = ny / len;
        normals[index * 3 + 2] = nz / len;
        const slope = Math.hypot(bdx, bdz);
        const depthLimited = bh > 0 && bh >= scenario.depthBreakRatio * depth;
        heights[index] = height;
        { let xh = 0, xdx = 0, xdz = 0;
          for (const c of longComps) {
            const ph = c.k * (x * c.cx + z * c.cz) + c.phase;
            const sn = Math.sin(ph), cs = Math.cos(ph);
            xh += c.a * sn; xdx += c.a * c.k * c.cx * cs; xdz += c.a * c.k * c.cz * cs;
          }
          xHeights[index] = xh; xSlopeX[index] = xdx; xSlopeZ[index] = xdz;
          shortHeights[index] = height - xh; }
        slopeX[index] = dhdx;
        slopeZ[index] = dhdz;
        breakHeights[index] = bh;
        breakSlopeX[index] = bdx;
        breakSlopeZ[index] = bdz;
        steep[index] = (slope >= scenario.breakSlope || depthLimited) ? 1 : 0;
        if (slope > maxSlope) maxSlope = slope;
      }
    }

    // Thin crests let light through (translucency, shading only). Near its tip a crest is
    // close to a parabola h = top - kappa x^2 / 2, with kappa the curvature across it: on
    // the displaced surface distances across the crest are squeezed by the Jacobian J, so
    // the second derivative grows as 1 / J^2 and kappa = -lap(h) / J^2. A sample lying
    // depth = slope^2 / (2 kappa) + skin below the crest's apex sees the crest
    // t = 2 sqrt(2 depth / kappa) thick (absolute metres), and light crossing it keeps
    // exp(-t / length), faded to exactly 0 by maxThickness. Only water above the local
    // mean counts. Flat water (kappa 0), rounded swell and big faces give exactly 0.
    thins.fill(0);
    {
      const r = Math.max(1, Math.round(TRANSLUCENCY.meanRadius / spacingGrid));
      const tMax = TRANSLUCENCY.maxThickness;
      for (let iz = r; iz < n - r; iz += 1) {
        for (let ix = r; ix < n - r; ix += 1) {
          const index = iz * n + ix;
          const h = heights[index];
          const lap = (heights[index - 1] + heights[index + 1] + heights[index - n] + heights[index + n] - 4 * h) / (spacingGrid * spacingGrid);
          const jac = Math.max(jacobians[index], 0.2);
          const kappa = -lap / (jac * jac);
          if (!(kappa > 1e-6)) continue;
          let mean = 0;
          for (let dz = -r; dz <= r; dz += 1) for (let dx = -r; dx <= r; dx += 1) mean += heights[index + dz * n + dx];
          mean /= (2 * r + 1) * (2 * r + 1);
          if (!(h - mean > 0)) continue;
          // Depth below the crest's own apex, continuous in time: locally the crest is the
          // parabola above, so its apex lies slope^2 / (2 kappa) above this sample (slope on
          // the displaced surface, the grid slope / J). Steep water far down a gently curved
          // big face is deep below its apex, so thick, and passes nothing.
          const slopeD = Math.hypot(slopeX[index], slopeZ[index]) / jac;
          const depthBelow = (slopeD * slopeD) / (2 * kappa) + TRANSLUCENCY.skin;
          const thickness = 2 * Math.sqrt((2 * depthBelow) / kappa);
          if (!(thickness < tMax)) continue;
          const cut = 1 - thickness / tMax;
          thins[index] = Math.exp(-thickness / TRANSLUCENCY.length) * cut * cut * (3 - 2 * cut);
        }
      }
    }

    // Flags are for this step only. A crest that has passed must not stay painted.
    breaks.fill(0);
    lips.fill(0);
    freshFoam.fill(0);
    // Slope peaks on the face, not the crest. Look downhill a fixed distance in metres,
    // so a smaller patch does not hide the face by making each cell shorter.
    const spacing = scenario.size / denom;
    const reach = Math.max(1, Math.round(crestReachAt(scenario, wind.speed) / spacing));
    const steps = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    // v28 foam fix: the total-height steepness rule (breakRuleAt; GPU LOCAL_FS does the same).
    const rule = breakRuleAt(scenario, wind.speed, spacing);
    let HS = null, troughMean = null;
    const HB = rule.mode > 1.5 ? shortHeights : heights;
    if (rule.mode > 0.5) {
      HS = new Float32Array(n * n);
      for (let iz = 1; iz < n - 1; iz += 1) for (let ix = 1; ix < n - 1; ix += 1) {
        let t = 0;
        for (let dz = -1; dz <= 1; dz += 1) for (let dx = -1; dx <= 1; dx += 1) t += HB[(iz + dz) * n + ix + dx];
        HS[iz * n + ix] = t / 9;
      }
      // mean of the row means (each row clamped at the edge), rows clamped too: as the GPU
      const rowMean = new Float32Array(n * n);
      for (let iz = 0; iz < n; iz += 1) for (let ix = 0; ix < n; ix += 1) {
        const x0 = Math.max(ix - rule.rb, 0), x1 = Math.min(ix + rule.rb, n - 1);
        let t = 0;
        for (let x = x0; x <= x1; x += 1) t += HB[iz * n + x];
        rowMean[iz * n + ix] = t / (x1 - x0 + 1);
      }
      troughMean = new Float32Array(n * n);
      for (let iz = 0; iz < n; iz += 1) for (let ix = 0; ix < n; ix += 1) {
        const z0 = Math.max(iz - rule.rb, 0), z1 = Math.min(iz + rule.rb, n - 1);
        let t = 0;
        for (let z = z0; z <= z1; z += 1) t += rowMean[z * n + ix];
        troughMean[iz * n + ix] = t / (z1 - z0 + 1);
      }
    }
    const brkMargin = rule.ts + 1;
    for (let iz = 1; iz < n - 1; iz += 1) {
      for (let ix = 1; ix < n - 1; ix += 1) {
        const index = iz * n + ix;
        if (rule.mode > 0.5) {
          if (ix >= brkMargin && iz >= brkMargin && ix < n - brkMargin && iz < n - brkMargin) {
            const h = HB[index];
            let drop = 0, dx = 0, dz = 0;
            for (const [sx, sz] of steps) {
              const fall = h - HB[(iz + sz) * n + (ix + sx)];
              if (fall > drop) { drop = fall; dx = sx; dz = sz; }
            }
            if (drop > 0 && h >= HB[(iz - dz) * n + (ix - dx)]) {
              const t = rule.ts, d = t * spacing, hs = HS[index];
              const lap = (HS[index - t] + HS[index + t] + HS[index - t * n] + HS[index + t * n] - 4 * hs) / (d * d);
              const eta = hs - troughMean[index];
              if (eta > 0 && lap < 0 && Math.sqrt(-lap * eta) * (rule.mode > 1.5 ? 1 + (rule.M || 0) * Math.max(0, xHeights[index] * (rule.kL || 0)) : 1) + (rule.mode > 1.5 ? (rule.lg ?? 1) * Math.hypot(xSlopeX[index], xSlopeZ[index]) : 0) >= rule.ak) { breaks[index] = 0.25; breakingCount += 1; }
            }
          }
        } else {
        const h = breakHeights[index];
        let drop = 0;
        let dx = 0;
        let dz = 0;
        for (const [sx, sz] of steps) {
          const fall = h - breakHeights[(iz + sz) * n + (ix + sx)];
          if (fall > drop) {
            drop = fall;
            dx = sx;
            dz = sz;
          }
        }
        const opposite = breakHeights[(iz - dz) * n + (ix - dx)];
        let face = false;
        for (let step = 0; step <= reach; step += 1) {
          const sx = ix + dx * step;
          const sz = iz + dz * step;
          if (sx < 0 || sz < 0 || sx >= n || sz >= n) break;
          if (steep[sz * n + sx]) {
            face = true;
            break;
          }
        }
        if (drop > 0 && h >= opposite && face) {
          breaks[index] = 0.25;
          breakingCount += 1;
        }
        }
        // Crest glow (lips, shading only): on the drawn surface, as before.
        const hl = heights[index];
        let dropL = 0;
        let lx = 0;
        let lz = 0;
        for (const [sx, sz] of steps) {
          const fall = hl - heights[(iz + sz) * n + (ix + sx)];
          if (fall > dropL) {
            dropL = fall;
            lx = sx;
            lz = sz;
          }
        }
        const oppositeL = heights[(iz - lz) * n + (ix - lx)];
        if (dropL > 0 && hl >= oppositeL && hl > 0) {
          let steepest = 0;
          for (let step = 0; step <= reach; step += 1) {
            const fx = ix + lx * step;
            const fz = iz + lz * step;
            if (fx < 0 || fz < 0 || fx >= n || fz >= n) break;
            const faceSlope = Math.hypot(slopeX[fz * n + fx], slopeZ[fz * n + fx]);
            if (faceSlope > steepest) steepest = faceSlope;
          }
          if (steepest >= 0.12) {
            const thin = Math.min(1, (steepest - 0.12) / 0.28);
            const high = Math.min(1, hl / 0.8);
            lips[index] = thin * high;
          }
        }
      }
    }

    // Foam goes on the front face of a breaking crest: the side the wave is moving
    // toward, where real water spills. The direction comes from the spectrum (each
    // component travels toward +(cos dir, sin dir) as its phase falls), never from
    // the camera. Nothing is painted unless the physics flagged a break nearby.
    const [tx, tz] = travelDirection(bands, wind.direction);
    const crestX = -tz;
    const crestZ = tx;
    const FACE_STEPS = faceStepsAt(scenario, wind.speed, scenario.size / denom);
    const JOIN_STEPS = Math.min(3, FACE_STEPS); // joined gap cells: their own face as before (<= 3)
    const JOIN_GAP = 4;
    const MAX_FROM_BREAK = 4;
    // Graded foam down the front face: full at the break or crest, then falling fast,
    // so the visible band hugs the top of the face. Every painted sample stays above 0,
    // so the foam set and the flag set are still the same samples.
    const inside = (x, z) => x >= 0 && z >= 0 && x < n && z < n;
    const along = (ix, iz, ux, uz, s) => [ix + Math.round(ux * s), iz + Math.round(uz * s)];
    // A crest along the travel axis: higher than the samples just behind and just ahead.
    function travelCrest(ix, iz) {
      if (!inside(ix, iz)) return false;
      const h = breakHeights[iz * n + ix];
      const [bx, bz] = along(ix, iz, tx, tz, -1);
      const [fx, fz] = along(ix, iz, tx, tz, 1);
      if (!inside(bx, bz) || !inside(fx, fz)) return false;
      return h > 0 && h >= breakHeights[bz * n + bx] && h >= breakHeights[fz * n + fx];
    }
    let freshCount = 0;
    // Every foam sample is also a flag sample, so Show break flags covers exactly
    // the foam and the two can be compared mark for mark.
    const spacingM = scenario.size / denom;
    // Displaced position along the travel direction, in metres.
    const travelPos = (index) => {
      const ix = index % n;
      const iz = (index - ix) / n;
      return (-half + ix * spacingM + shifts[index * 2]) * tx + (-half + iz * spacingM + shifts[index * 2 + 1]) * tz;
    };
    function mark(index, grade, crestPos) {
      if (freshFoam[index] === 0) freshCount += 1;
      if (grade > freshFoam[index]) {
        freshFoam[index] = grade;
        freshAhead[index] = travelPos(index) - crestPos;
      }
      flags[index] = 1;
    }
    // Mark a crest sample and the samples ahead of it while the surface keeps dropping.
    // near() limits how far from a break the face may run.
    function paintFace(ix, iz, near, steps = FACE_STEPS) {
      const crestPos = travelPos(iz * n + ix);
      mark(iz * n + ix, faceGrade(0, FACE_STEPS), crestPos);
      let prev = breakHeights[iz * n + ix];
      for (let s = 1; s <= steps; s += 1) {
        const [fx, fz] = along(ix, iz, tx, tz, s);
        if (!inside(fx, fz) || !near(fx, fz)) break;
        const h = breakHeights[fz * n + fx];
        if (h >= prev) break;
        mark(fz * n + fx, faceGrade(s, FACE_STEPS), crestPos);
        prev = h;
      }
    }
    const anywhere = () => true;
    flags.fill(0);
    freshAhead.fill(NO_FOAM_AHEAD);
    const crestBreaks = [];
    for (let index = 0; index < n * n; index += 1) {
      if (breaks[index] <= 0) continue;
      const ix = index % n;
      const iz = (index - ix) / n;
      paintFace(ix, iz, anywhere);
      if (travelCrest(ix, iz)) crestBreaks.push(index);
    }
    // Join breaks into a line along the crest, which runs across the travel direction.
    // Only between two breaks that are both crests along the travel axis, at most
    // JOIN_GAP empty cells apart, and only if every sample in the gap is itself a crest.
    // A trough or open water between them leaves the gap empty.
    for (const index of crestBreaks) {
      const ix = index % n;
      const iz = (index - ix) / n;
      let joined = false;
      for (let s = 1; s <= JOIN_GAP + 1 && !joined; s += 1) {
        for (const w of [0, -1, 1]) {
          const [cx, cz] = along(ix, iz, crestX, crestZ, s);
          const [px, pz] = along(cx, cz, tx, tz, w);
          if (!inside(px, pz)) continue;
          const partner = pz * n + px;
          if (breaks[partner] <= 0 || !travelCrest(px, pz)) continue;
          joined = true;
          if (s === 1) break;
          const gap = [];
          let ok = true;
          for (let j = 1; j < s && ok; j += 1) {
            const [gx, gz] = along(ix, iz, crestX, crestZ, j);
            const [qx, qz] = along(gx, gz, tx, tz, (w * j) / s);
            let best = null;
            for (const m of [0, -1, 1]) {
              const [sx, sz] = along(qx, qz, tx, tz, m);
              if (!travelCrest(sx, sz)) continue;
              // Never snap behind both ends of the gap: that would be the back of the crest.
              const offB = (sx - ix) * tx + (sz - iz) * tz;
              const offP = (sx - px) * tx + (sz - pz) * tz;
              // ...and only snap where the cell is ahead of an end it is near (within
              // MAX_FROM_BREAK), so no fresh foam lands upwave of every break around it.
              const nearB = (sx - ix) ** 2 + (sz - iz) ** 2 <= MAX_FROM_BREAK * MAX_FROM_BREAK;
              const nearP = (sx - px) ** 2 + (sz - pz) ** 2 <= MAX_FROM_BREAK * MAX_FROM_BREAK;
              if (!(offB >= -0.5 && nearB) && !(offP >= -0.5 && nearP)) continue;
              if (!best || breakHeights[sz * n + sx] > breakHeights[best[1] * n + best[0]]) best = [sx, sz];
            }
            if (!best) ok = false;
            else gap.push(best);
          }
          if (!ok) break;
          const near = (x, z) =>
            Math.hypot(x - ix, z - iz) <= MAX_FROM_BREAK || Math.hypot(x - px, z - pz) <= MAX_FROM_BREAK;
          for (const [gx, gz] of gap) if (near(gx, gz)) paintFace(gx, gz, near, JOIN_STEPS);
          break;
        }
      }
    }
    freshRuns.fill(0);
    for (let index = 0; index < n * n; index += 1) {
      if (!(freshFoam[index] > 0)) continue;
      const ix = index % n;
      const iz = (index - ix) / n;
      // Foam samples in a row along the crest on each side, counted up to two. A crest
      // that wanders one cell along the travel direction still counts as one run.
      const count = (dir) => {
        let c = 0;
        let cx = ix;
        let cz = iz;
        for (let s = 1; s <= 2; s += 1) {
          const [x, z] = along(cx, cz, crestX, crestZ, dir);
          let next = null;
          for (const w of [0, -1, 1]) {
            const [wx, wz] = along(x, z, tx, tz, w);
            if (inside(wx, wz) && freshFoam[wz * n + wx] > 0) {
              next = [wx, wz];
              break;
            }
          }
          if (!next) break;
          [cx, cz] = next;
          c += 1;
        }
        return c;
      };
      const a = count(1);
      const b = count(-1);
      // Inside a run: full. End of a run of three or more: 0, so the band tapers out
      // inside the last cell. A run of two or a lone sample: 0.5, so short runs stay visible.
      freshRuns[index] = a > 0 && b > 0 ? 1 : Math.max(a, b) >= 2 ? 0 : 0.5;
    }

    // Age every sample by the sim step. A sample flagged now refreshes to full strength
    // and keeps this step's grade, distance ahead and lengthwise value. Older foam keeps
    // what it had and fades smoothly to 0 at the end of the window; it never moves and
    // never grows.
    const window = foamWindowAt(scenario, wind.speed);
    const stepDt = Math.max(dt, 0);
    let lingerCount = 0;
    let recentCount = 0;
    let foamCount = 0;
    for (let i = 0; i < n * n; i += 1) {
      if (flags[i] > 0) {
        ages[i] = 0;
        keptGrade[i] = freshFoam[i];
        keptAhead[i] = freshAhead[i];
        keptRun[i] = freshRuns[i];
      } else {
        ages[i] += stepDt;
      }
      const age = ages[i];
      if (age < window) {
        const u = age / window;
        const fade = flags[i] > 0 ? 1 : 1 - u * u * (3 - 2 * u);
        foamGrade[i] = keptGrade[i];
        foamFade[i] = fade;
        foam[i] = keptGrade[i] * fade;
        foamAge[i] = flags[i] > 0 ? 0 : age / window;
        ahead[i] = keptAhead[i];
        runs[i] = keptRun[i];
        recent[i] = flags[i] > 0 ? 0 : 1;
      } else {
        foamGrade[i] = 0;
        // Neutral where there is no foam (nothing draws there anyway), so the fade
        // never thins fresh foam at its edge through interpolation.
        foamFade[i] = 1;
        foamAge[i] = 0;
        foam[i] = 0;
        ahead[i] = NO_FOAM_AHEAD;
        runs[i] = 0;
        recent[i] = 0;
      }
      if (foam[i] > 0) {
        foamCount += 1;
        if (!(flags[i] > 0)) lingerCount += 1;
      }
      if (recent[i] > 0) recentCount += 1;
    }

    return {
      heights,
      slopeX,
      slopeZ,
      breakHeights,
      breakSlopeX,
      breakSlopeZ,
      breaks,
      lips,
      foam,
      foamGrade,
      foamFade,
      foamAge,
      flags,
      recent,
      ahead,
      runs,
      ages,
      freshFoam,
      freshCount,
      foamCount,
      lingerCount,
      recentCount,
      foamWindow: window,
      travelX: tx,
      travelZ: tz,
      shifts,
      normals,
      jacobians,
      thins,
      n,
      size: scenario.size,
      breakingCount,
      breakRule: rule, faceSteps: FACE_STEPS,
      xHeights, xSlopeX, xSlopeZ,
      sampleCount: n * n,
      maxSlope,
      depth,
      chopPatch,
      finePatch,
      roughness,
      bands: bands.map((band) => ({
        id: band.spec.id,
        label: band.spec.label,
        significantHeight: 4 * Math.sqrt(bandVariance(band)),
        spectrum: band.spectrum ? band.spectrum.model : "pm",
      })),
    };
  }

  return { step };
}

import { GRAVITY } from "./gravity.js";
import { createBand, retuneBand, sampleBand, bandVariance } from "./band.js?v=27.2";
import { createChopPatch } from "./chopPatch.js?v=27.2";
import { slopeBudget } from "./roughness.js?v=27.2";

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
    const reach = Math.max(1, Math.round(scenario.crestReach / spacing));
    const steps = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (let iz = 1; iz < n - 1; iz += 1) {
      for (let ix = 1; ix < n - 1; ix += 1) {
        const index = iz * n + ix;
        // Breaking: on the break-test field.
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
    const FACE_STEPS = 3;
    const JOIN_GAP = 4;
    const MAX_FROM_BREAK = 4;
    // Graded foam down the front face: full at the break or crest, then falling fast,
    // so the visible band hugs the top of the face. Every painted sample stays above 0,
    // so the foam set and the flag set are still the same samples.
    const FACE_GRADE = [1, 0.5, 0.2, 0.06];
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
    function paintFace(ix, iz, near) {
      const crestPos = travelPos(iz * n + ix);
      mark(iz * n + ix, FACE_GRADE[0], crestPos);
      let prev = breakHeights[iz * n + ix];
      for (let s = 1; s <= FACE_STEPS; s += 1) {
        const [fx, fz] = along(ix, iz, tx, tz, s);
        if (!inside(fx, fz) || !near(fx, fz)) break;
        const h = breakHeights[fz * n + fx];
        if (h >= prev) break;
        mark(fz * n + fx, FACE_GRADE[s], crestPos);
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
              if (offB < -0.5 && offP < -0.5) continue;
              if (!best || breakHeights[sz * n + sx] > breakHeights[best[1] * n + best[0]]) best = [sx, sz];
            }
            if (!best) ok = false;
            else gap.push(best);
          }
          if (!ok) break;
          const near = (x, z) =>
            Math.hypot(x - ix, z - iz) <= MAX_FROM_BREAK || Math.hypot(x - px, z - pz) <= MAX_FROM_BREAK;
          for (const [gx, gz] of gap) if (near(gx, gz)) paintFace(gx, gz, near);
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
    const window = Math.max(scenario.foamFadeSeconds || 0, 1e-6);
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

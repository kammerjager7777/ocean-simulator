import { shoal } from "./band.js?v=28";

// The short wind-chop band's own fine patch. The big grid (about 2.2 m spacing at
// 420 m) cannot hold waves shorter than 2.4 cells, and the chop band's grid
// components stop at omegaMax x peak anyway. The patch is a small square of sea,
// tens of metres across and finely sampled, that holds the rest of the chop band:
// every wavelength from just above what the grid holds down to 2.4 patch cells.
// Same spectrum as the band (scenario data: Elfouhaily or PM), same wind sample (the chop band's own lagged wind),
// same direction spread, gravity and finite-depth dispersion as the chop band.
// Each wavelength lives in exactly one place: the grid or the patch, never both.
//
// The patch is periodic (every wave fits a whole number of times across it), so it
// tiles. It is the sum of lattice waves a*cos(k.x + phase) with each phase falling at
// its own omega(k), evaluated by inverse FFT as a complex field z, whose real part is
// the height. The imaginary part is the same waves a quarter period on; the renderer
// uses it to rotate the phase slowly over the sea so the tiles do not repeat.
//
// The same module builds the band's second, finer patch (patchSpec = bandSpec.finePatch):
// a smaller square, more finely sampled, that carries the band on down from where the
// first patch stops. Its low edge is the first patch's cutoff wavenumber (kLow): a wave
// lives in the fine patch only if it is shorter than anything the first patch can hold.
// Surface tension is left out on purpose: at the shortest wave here (about 0.25 m,
// k about 25 rad/m) it adds sigma k^2 / (rho g) = 7.4e-5 * 625 / 9.81, about 0.5%, to
// omega^2 (0.24% to omega). Capillary effects only matter below a few centimetres.

function makeFFT(N) {
  const bits = Math.round(Math.log2(N));
  if (1 << bits !== N) throw new Error("patch resolution must be a power of two");
  const rev = new Uint32Array(N);
  for (let i = 0; i < N; i += 1) {
    let r = 0;
    for (let b = 0; b < bits; b += 1) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cosT = new Float64Array(N / 2);
  const sinT = new Float64Array(N / 2);
  for (let i = 0; i < N / 2; i += 1) {
    cosT[i] = Math.cos((2 * Math.PI * i) / N);
    sinT[i] = Math.sin((2 * Math.PI * i) / N);
  }
  const tr = new Float64Array(N);
  const ti = new Float64Array(N);
  // In place on a strided line; sign +1 is the inverse transform (no scaling).
  function line(re, im, offset, stride, sign) {
    for (let i = 0; i < N; i += 1) {
      tr[rev[i]] = re[offset + i * stride];
      ti[rev[i]] = im[offset + i * stride];
    }
    for (let size = 2; size <= N; size *= 2) {
      const half = size / 2;
      const step = N / size;
      for (let start = 0; start < N; start += size) {
        for (let j = 0; j < half; j += 1) {
          const wr = cosT[j * step];
          const wi = sign * sinT[j * step];
          const a = start + j;
          const b = a + half;
          const xr = tr[b] * wr - ti[b] * wi;
          const xi = tr[b] * wi + ti[b] * wr;
          tr[b] = tr[a] - xr;
          ti[b] = ti[a] - xi;
          tr[a] += xr;
          ti[a] += xi;
        }
      }
    }
    for (let i = 0; i < N; i += 1) {
      re[offset + i * stride] = tr[i];
      im[offset + i * stride] = ti[i];
    }
  }
  return function fft2(re, im, sign) {
    for (let row = 0; row < N; row += 1) line(re, im, row * N, 1, sign);
    for (let col = 0; col < N; col += 1) line(re, im, col, N, sign);
  };
}

function unitPhase(n) {
  const x = Math.sin(n * 91.7 + 3.1) * 43758.5453;
  return (x - Math.floor(x)) * Math.PI * 2;
}

export function createChopPatch(bandSpec, patchSpec = bandSpec.patch) {
  const patch = patchSpec;
  const N = patch.resolution;
  const L = patch.size;
  const spacing = L / N;
  const dk = (2 * Math.PI) / L;
  // Same rule as the big grid: nothing shorter than 2.4 cells.
  const kCut = (2 * Math.PI) / (2.4 * spacing);
  const count = N * N;
  const kx = new Float64Array(count);
  const kz = new Float64Array(count);
  const kk = new Float64Array(count);
  const phase = new Float64Array(count);
  const omega = new Float64Array(count);
  const amp = new Float64Array(count);
  for (let row = 0; row < N; row += 1) {
    const nz = row < N / 2 ? row : row - N;
    for (let col = 0; col < N; col += 1) {
      const nx = col < N / 2 ? col : col - N;
      const i = row * N + col;
      kx[i] = nx * dk;
      kz[i] = nz * dk;
      kk[i] = Math.hypot(kx[i], kz[i]);
      phase[i] = unitPhase(i);
    }
  }
  const fft2 = makeFFT(N);
  const hRe = new Float64Array(count);
  const hIm = new Float64Array(count);
  const xRe = new Float64Array(count);
  const xIm = new Float64Array(count);
  const zRe = new Float64Array(count);
  const zIm = new Float64Array(count);
  // Slopes for the renderer: (dh/dx, dh/dx quarter-on, dh/dz, dh/dz quarter-on).
  const slopes = new Float32Array(count * 4);
  const heights = new Float32Array(count);
  let version = 0;

  // band: the chop band's state after retuneBand (its own wind, peak and what the grid holds).
  // kLow: for the finer patch, the coarser patch's cutoff; only shorter waves are kept.
  function step(band, windSample, depth, gravity, dt, kLow = 0) {
    const h = Math.max(depth, 0.5);
    // The band's own spectrum for this wind sample, as retuneBand set it: the patch is
    // the same band, so it uses exactly the same formula (no splice at the grid/patch edge).
    const spectrum = band.spectrum;
    const omegaLow = band.heldOmegaEdge;
    const calm = windSample.speed < 0.05;
    const halfSpread = bandSpec.spread / 2;
    const dirDensity = 1 / bandSpec.spread;
    let modes = 0;
    let variance = 0;
    let kMin = Infinity;
    let kMax = 0;
    let omegaMin = Infinity;
    for (let i = 0; i < count; i += 1) {
      const k = kk[i];
      if (k === 0) {
        amp[i] = 0;
        continue;
      }
      const t = Math.tanh(k * h);
      const w = Math.sqrt(gravity * k * t);
      omega[i] = w;
      phase[i] -= w * Math.max(dt, 0);
      amp[i] = 0;
      if (calm || k > kCut || w <= omegaLow || k <= kLow) continue;
      let off = Math.atan2(kz[i], kx[i]) - windSample.direction;
      off = Math.atan2(Math.sin(off), Math.cos(off));
      if (Math.abs(off) > halfSpread) continue;
      // Variance per lattice wave: S(omega) d(omega)/dk D(theta) / k * dk^2.
      const cg = (gravity * t + gravity * k * h * (1 - t * t)) / (2 * w);
      const e = spectrum.density(w) * cg * dirDensity / k * dk * dk;
      const a = shoal(Math.sqrt(2 * e), w, k, depth, gravity);
      amp[i] = a;
      modes += 1;
      variance += (a * a) / 2;
      kMin = Math.min(kMin, k);
      kMax = Math.max(kMax, k);
      omegaMin = Math.min(omegaMin, w);
    }
    for (let i = 0; i < count; i += 1) {
      const a = amp[i];
      const c = a * Math.cos(phase[i]);
      const s = a * Math.sin(phase[i]);
      hRe[i] = c;
      hIm[i] = s;
      // i k C for the slopes.
      xRe[i] = -kx[i] * s;
      xIm[i] = kx[i] * c;
      zRe[i] = -kz[i] * s;
      zIm[i] = kz[i] * c;
    }
    fft2(hRe, hIm, 1);
    fft2(xRe, xIm, 1);
    fft2(zRe, zIm, 1);
    let slopeVar = 0;
    for (let i = 0; i < count; i += 1) {
      heights[i] = hRe[i];
      slopes[i * 4] = xRe[i];
      slopes[i * 4 + 1] = xIm[i];
      slopes[i * 4 + 2] = zRe[i];
      slopes[i * 4 + 3] = zIm[i];
      slopeVar += xRe[i] * xRe[i] + zRe[i] * zRe[i];
    }
    version += 1;
    return {
      size: L,
      n: N,
      spacing,
      heights,
      heightsQuarter: hIm,
      slopes,
      version,
      modes,
      variance,
      significantHeight: 4 * Math.sqrt(variance),
      rmsSlope: Math.sqrt(slopeVar / count),
      kMin: modes ? kMin : 0,
      kMax,
      kCut,
      kLow,
      omegaLow,
      omegaMin: modes ? omegaMin : 0,
      longest: modes ? (2 * Math.PI) / kMin : 0,
      shortest: modes ? (2 * Math.PI) / kMax : 0,
      amplitudes: amp,
      omegas: omega,
      phases: phase,
      kx,
      kz,
    };
  }

  return { step, fft2 };
}

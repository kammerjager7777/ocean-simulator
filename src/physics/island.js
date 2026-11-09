// Fixed sandy island. The ocean's depth slider is the deep-water depth.
// From the centre: a dry top, a beach down to a 2 m shelf, then a bank out
// to the open-ocean depth. The same formula is in the shaders.

export const ISLAND = Object.freeze({
  cx: 70,
  cz: 0,
  span: 80,
  nx: 64,
  ny: 64,
  Rbeach: 22,
  Rbank: 78,
  peak: 1.6,
  shelf: -2,
  dLo: 2,
  dHi: 6,
});

export function islandBed(x, z, depth, I = ISLAND) {
  const r = Math.hypot(x - I.cx, z - I.cz);
  if (r >= I.Rbank) return -depth;
  if (r >= I.Rbeach) {
    const t = (r - I.Rbeach) / (I.Rbank - I.Rbeach);
    return I.shelf + (-depth - I.shelf) * t;
  }
  return I.peak + (I.shelf - I.peak) * (r / I.Rbeach);
}

export function coastWeight(depthStill, I = ISLAND) {
  if (depthStill >= I.dHi) return 0;
  if (depthStill <= I.dLo) return 1;
  const t = (depthStill - I.dLo) / (I.dHi - I.dLo);
  const s = t * t * (3 - 2 * t);
  return 1 - s;
}

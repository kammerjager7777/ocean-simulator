// Sea-surface roughness, in one place, for both the physics and the renderer.
//
// The physics part: Cox and Munk (1954) measured the total mean square slope (mss) of
// the open sea from sun glitter: 0.003 + 5.12e-3 U (U = wind speed at 12.5 m, m/s;
// up- plus cross-wind). The modelled sea resolves part of that slope: the big grid's
// wave components and the chop band's two patches. Everything else (waves shorter than
// the finer patch, about 0.25 m, down to capillaries) is not drawn, so its slope reaches
// the eye only as a spread of the reflected light. That sub-grid mss is the Cox and Munk
// total minus the resolved slope, at the live wind. There is no other constant.
//
// The wind used is the chop band's lagged wind (the same sample its waves and patches
// were retuned with this step, gusts included): the short waves follow the wind within
// seconds, and using the same sample for both terms keeps the subtraction consistent.

export function coxMunkMss(windSpeed) {
  return 0.003 + 5.12e-3 * Math.max(0, windSpeed);
}

// bands: the surface's band states (each with components {amplitude, k}); chopPatch,
// finePatch: this step's patch states (rmsSlope); windSpeed: the chop band's lagged wind.
export function slopeBudget(bands, chopPatch, finePatch, windSpeed) {
  let grid = 0;
  for (const band of bands) for (const c of band.components) grid += (c.amplitude * c.k) ** 2 / 2;
  const coarse = chopPatch ? chopPatch.rmsSlope ** 2 : 0;
  const fine = finePatch ? finePatch.rmsSlope ** 2 : 0;
  const resolved = grid + coarse + fine;
  const coxMunk = coxMunkMss(windSpeed);
  return { windSpeed, coxMunk, grid, coarse, fine, resolved, subGrid: Math.max(0, coxMunk - resolved) };
}

// The renderer's part, as GLSL so the shader carries no roughness numbers of its own.
// The slope distribution under one pixel is taken as Gaussian (as Cox and Munk found)
// with mss = lobeVariance(...): the sub-grid mss from the physics, plus every resolved
// slope the pixel does not show through its normal:
//  - a patch faded out with distance (the shader multiplies its slopes by fade, so a
//    fraction 1 - fade^2 of its variance leaves the normal),
//  - a patch averaged away by its texture's mip level (retained = variance kept at that
//    level, measured on the patch itself when it is uploaded),
//  - the big grid's slope changing inside the pixel (its screen derivatives over a
//    one-pixel box filter, variance 1/12 per pixel width squared).
// So every pixel's slope spread adds back up to the same Cox and Munk total: what the
// pixel resolves comes through its normal, the rest through the width of the lobe.
export const LOBE_GLSL = `
float lobeVariance(float subMss, float coarseMss, float coarseKept, float fineMss, float fineKept, vec2 gridDx, vec2 gridDy) {
  return subMss
       + coarseMss * (1.0 - coarseKept)
       + fineMss * (1.0 - fineKept)
       + (dot(gridDx, gridDx) + dot(gridDy, gridDy)) / 12.0;
}
`;

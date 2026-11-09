// Same-box contrast for the nine-point buoy deposit. Not a Mac timing run.
// Page: ?fixtest=1&fixdt=0.05&nodraw=1&camdist=160&layers=wind,buoy
// Viewport 640x400. After Hs is flat, aim (36, 0, -25), yaw 0.75, pitch 0.5, distance 48.
// Capture a paused background, then push these records with birth = current sim time:
//   x = 30 + i*1.6, z = -25, nx = 0, nz = 0.82
//   w = [4.27, 4.42, 3.83, 4.29, 3.82, 4.05, 4.99, 4.52, 3.74]
// Park the buoy at (90, 90). Unpause with fixdt 0.05 and pause at ages 1, 4, 6, 8.
// Count pixels in the projected trail box (80 px pad in x, 60 in y) whose
// luma delta vs the background is > 8 and whose red and green deltas are > 1.
// On this software renderer, v31.2 was 6518 / 1340 / 646 / 582 and v31.3 was
// 7358 / 1962 / 1871 / 878. Quartermaster's SwiftShader polygon counts for v31
// were 7684 / 2182 / 1881 / 725 and for v31.1 were 6795 / 1001 / 89 / 8.
export const WIDTHS = [4.27, 4.42, 3.83, 4.29, 3.82, 4.05, 4.99, 4.52, 3.74];

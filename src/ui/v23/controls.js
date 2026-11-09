// Fixed world slots. Orbit and zoom do not move them. A domain change moves an
// anchor only so it stays just outside that domain.
//
//   size     +Z face, outside the slice. A dimension line runs the true edge,
//            with T-shaped caps, and is not a drag target. A short slider sits
//            further out, centered, and that is what sets the size. The label
//            is centered on the slider and sits further out again, so it does
//            not cover the track or the dimension line.
//   chop     middle of the +X face, the side opposite the wind. Not a corner.
//   wind     middle of the −X face, flat on the seabed y = -D
//   clarity  corner F. An illustrative Secchi disk on its own line, not the seabed.
//   depth    corner R, the real seabed, in raw metres. The column sits just
//            outside that corner, with a small display-metre gap, and it still
//            runs the full depth.
//
// Glyphs are sized in display metres (S / 250). The eye already backs up with
// the domain, so a display length keeps its apparent size. The size edge and
// the depth column stay in raw metres: they are the thing being measured.
// The visibility disk does not. A shallow bed must not shrink that gauge.
import { ui } from "./core.js?v=31.9";
import { plane, showM } from "./geometry.js?v=30.29";

const LMAX = 18 + 25 * 3.5;
const BAR0 = 76;
// Wind on the −X face. Chop is fixed on the opposite face. Neither follows the camera.
const WDIR = [-1, 0];
const CHOP_FACE = [1, 0];
// Apparent size of the arrow, as a fraction of the old 105 m ruler on a 250 m sea.
// Multiplied by S/250 so a larger sea does not shrink the glyph and a smaller
// sea does not grow it.
export const windScale = (S) => 0.48 * (S / 250);
// Illustrative Secchi gauge, in display metres. Not the ocean depth. The column
// is at least this long, and the quartered disk at least this radius, so a 2 m
// bed does not collapse them. A deeper bed may lengthen the column; the seabed
// control is still the only thing that shows the real depth.
export const CLAR_COL_DM = 24;
export const CLAR_DISK_DM = 12;
export function clarityGauge(S, D) {
  const m = showM(S);
  const col = Math.max(0.42 * D, CLAR_COL_DM * m);
  const disk = CLAR_DISK_DM * m;
  return { col, disk, m };
}

function windRadius(S, D = 40) {
  const h = S / 2, ux = WDIR[0], uz = WDIR[1], reach = LMAX * windScale(S);
  const gap = (need) => {
    let lo = 0, hi = h * 8 + reach * 4;
    for (let i = 0; i < 48; i++) {
      const mid = (lo + hi) / 2;
      if (need(ux * mid, uz * mid)) hi = mid; else lo = mid;
    }
    return hi;
  };
  const square = gap((x, z) => {
    const dx = Math.max(0, Math.abs(x) - h), dz = Math.max(0, Math.abs(z) - h);
    const d = dx === 0 && dz === 0 ? -Math.min(h - Math.abs(x), h - Math.abs(z)) : Math.hypot(dx, dz);
    return d >= reach + 8;
  });
  const from = (px, pz, extra) => gap((x, z) => Math.hypot(x - px, z - pz) >= reach + extra);
  const m = S / 250, g = 0.1 * S, label = 0.12 * S;
  const planes = sizePlanes(S);
  const sizeClear = gap((x, z) => {
    const t = Math.max(-planes.sizeHalf, Math.min(planes.sizeHalf, x));
    const u = Math.max(-planes.labelHalf, Math.min(planes.labelHalf, x));
    return Math.hypot(x - t, z - planes.sliderZ) >= reach + label
      && Math.hypot(x - u, z - planes.labelZ) >= reach + 8 * m;
  });
  const chopClear = from(h + 14 * m, 0, 24 * m);
  const cg = clarityGauge(S, D);
  const clarClear = from(-h - 0.08 * S, -h - 0.08 * S, cg.disk + 10 * m);
  return Math.max(square, from(-h - g, -h - g, label), from(h + g, -h - g, label), chopClear, sizeClear, clarClear);
}

// Dimension line hugs the measured +Z edge. The short slider sits further out,
// and the size label sits further out again, so the three do not share a line.
function sizePlanes(S) {
  const h = S / 2, m = showM(S);
  const sliderZ = h + 34 * m, labelZ = sliderZ + 22 * m;
  return { h, m, sliderZ, labelZ, dimZ: h + 8 * m, sizeHalf: 22 * m, labelHalf: 40 * m };
}

export function worldLayout(S = ui.S(), D = ui.D()) {
  const h = S / 2, m = showM(S);
  const windR = windRadius(S, D);
  const wScale = windScale(S);
  const gauge = 0.08 * S;
  const sp = sizePlanes(S);
  const clar = clarityGauge(S, D);
  // Seabed column: full depth, a short step outside corner R. 0.08 S was twenty
  // display metres off both faces. Ten keeps a visible gap and stays off the water.
  const depthGap = 10 * m;
  return {
    adaptive: false,
    size: {
      edge: "+Z", y: 0,
      // One groove on the block's +Z edge. The grip slides along that edge.
      a: [-h, 0, h + 2 * m], b: [h, 0, h + 2 * m],
    },
    chop: {
      edge: "+X", face: "+X",
      origin: [h + 14 * m, 0, 0], span: 42 * m, along: [0, 0, 1],
    },
    wind: {
      // Surface, outside corner F. The ring radius is the 25 m/s arrow length,
      // so the whole ring stays off the water.
      tail: [-h - LMAX * wScale - 8 * m, 0, -h - LMAX * wScale - 8 * m],
      y: 0, lift: 0, lane: 16 * m,
      outward: [-1, 0, -1], scale: wScale, bar: 0, ring: LMAX * wScale,
    },
    depth: { corner: "+X", wx: h + 2 * m, wz: 0, y0: 0, y1: -D },
  };
}

function normPoint(p, S, D) {
  const h = S / 2 || 1;
  return [p[0] / h, p[1] / (D || 1), p[2] / h];
}

function outside(x, z, h) {
  return Math.abs(x) >= h - 0.05 || Math.abs(z) >= h - 0.05;
}

export function layoutReport(L = worldLayout(), S = ui.S(), D = ui.D()) {
  const h = S / 2;
  let windInside = false;
  const tail = L.wind.tail;
  const reach = LMAX * (L.wind.scale || windScale(S));
  for (let k = 0; k < 16; k++) {
    const a = (k / 16) * Math.PI * 2;
    const x = tail[0] + Math.cos(a) * reach, z = tail[2] + Math.sin(a) * reach;
    if (Math.abs(x) <= h && Math.abs(z) <= h) windInside = true;
  }
  const sizeLen = Math.hypot(L.size.b[0] - L.size.a[0], L.size.b[2] - L.size.a[2]);
  const depthLen = Math.abs(L.depth.y0 - L.depth.y1);
  const chopToWind = Math.hypot(L.chop.origin[0] - tail[0], L.chop.origin[1] - tail[1], L.chop.origin[2] - tail[2]);
  return {
    norm: {
      size: [normPoint(L.size.a, S, D), normPoint(L.size.b, S, D)],
      chop: normPoint(L.chop.origin, S, D),
      chopSpan: L.chop.span / S,
      wind: normPoint(tail, S, D),
      depth: L.depth.corner,
    },
    sizeLen, depthLen, windY: tail[1], windInside,
    chopOutside: L.chop.origin[0] > h && Math.abs(L.chop.origin[2]) < h * 0.2,
    chopToWind,
  };
}

export function arrange() {
  const S = ui.S(), D = ui.D();
  const L = worldLayout(S, D);
  const report = layoutReport(L, S, D);
  ui.debug.slots = L;
  ui.debug.adaptive = false;
  ui.debug.separation = report;
  if (!ui.V) return { ok: false, objects: {}, anchors: report.norm };
  const objects = {};
  const m = showM(S);
  const sizeU = [1, 0, 0];
  const sizeO = [(L.size.a[0] + L.size.b[0]) / 2, 0, L.size.a[2]];
  const sizePlane = plane(sizeO, sizeU, [0, 0, 1]);
  objects.size = {
    kind: "size", edge: "+Z",
    a: ui.P3(L.size.a), b: ui.P3(L.size.b),
    wa: L.size.a, wb: L.size.b,
    len: Math.hypot(L.size.b[0] - L.size.a[0], L.size.b[2] - L.size.a[2]),
    world: sizePlane, corner: "+Z",
  };
  const d = L.depth;
  const depthPlane = plane([d.wx, 0, d.wz], [0, 0, 1], [0, 1, 0]);
  const depthLabel = plane([d.wx + 14 * m, (d.y0 + d.y1) / 2, d.wz], [0, 0, 1], [0, 1, 0]);
  objects.depth = {
    kind: "depth", corner: "R", wx: d.wx, wz: d.wz,
    a: ui.P3([d.wx, d.y0, d.wz]), b: ui.P3([d.wx, d.y1, d.wz]),
    wa: [d.wx, d.y0, d.wz], wb: [d.wx, d.y1, d.wz],
    world: depthPlane, label: depthLabel,
  };
  const w = L.wind;
  objects.wind = {
    kind: "wind", corner: "-X", tail: w.tail.slice(), lift: 0, lane: w.lane, y: w.y,
    outward: w.outward, parallel: true, scale: w.scale, bar: w.bar,
    label: plane(w.tail, [0, 0, 1], [0, 1, 0]),
  };
  ui.debug.anchors = {
    adaptive: false,
    corners: { size: "+Z", chop: "+X", clarity: "F", depth: "R", wind: "-X" },
    placed: Object.keys(objects),
    norm: report.norm,
  };
  return { ok: true, objects, anchors: ui.debug.anchors, duration: 0 };
}

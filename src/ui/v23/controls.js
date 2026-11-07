// v30.27 fixed world slots. v30.26 picked a screen target and retargeted the world
// edge when the camera moved. That is gone. Orbit and zoom do not change these
// endpoints or their lengths. A domain size or depth change does, in proportion.
//
//   size    full +Z edge, just outside the footprint, y = 0
//   chop    outside the -X face, standing on y = 0
//   wind    fixed ray on the −X face, flat on the seabed y = -D, scaled with the domain
//   clarity corner F, depth corner R (opposite corners, depth is the full column)
import { ui } from "./core.js?v=30.27";
import { plane } from "./geometry.js?v=30.27";

const LMAX = 18 + 25 * 3.5;
const BAR0 = 76;
// Chop stays outside corner A. That diagonal is screen-left at yaw 0.75, and it is
// a constant, not the live camera.
const LEFT = [-Math.SQRT1_2, Math.SQRT1_2];
// Middle of the −X face, not the clarity corner. A slot beside corner F puts the
// heading disk across the clarity column once the bar and the arrow are drawn.
// The ray is a constant. Orbit does not retarget it.
const WDIR = [-1, 0];
// On a small sea the full 105.5 m arrow is larger than the gaps between the
// other controls, so it cannot stay on screen and clear of them. The widget
// scales with the domain and is full size from 400 m up.
export const windScale = (S) => Math.min(1, S / 400);

function windRadius(S) {
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
  const g = 0.1 * S, diag = h * Math.SQRT2, chopR = diag + 0.08 * S, sizeZ = h + 0.08 * S, label = 0.16 * S;
  const sizeClear = gap((x, z) => {
    const t = Math.max(-h, Math.min(h, x));
    return Math.hypot(x - t, z - sizeZ) >= reach + label;
  });
  return Math.max(square, from(-h - g, -h - g, label), from(h + g, -h - g, label), from(LEFT[0] * chopR, LEFT[1] * chopR, label), sizeClear);
}

export function worldLayout(S = ui.S(), D = ui.D()) {
  const h = S / 2;
  const diag = h * Math.SQRT2;
  const chopR = diag + 0.08 * S;
  const windR = windRadius(S);
  const wScale = windScale(S);
  const sizeOut = 0.08 * S;
  const gauge = 0.1 * S;
  // The camera pulls back with the domain, so a fixed 6 m lift collapses the bar
  // onto the arrow. Grow the lift with the sea so the two shapes stay apart.
  const lift = Math.max(6, 0.16 * D, 0.07 * S);
  const at = (r, y) => [LEFT[0] * r, y, LEFT[1] * r];
  return {
    adaptive: false,
    size: { edge: "AN", y: 0, a: [-h, 0, h + sizeOut], b: [h, 0, h + sizeOut] },
    chop: { edge: "AF", origin: at(chopR, 0), span: 0.18 * S, along: [LEFT[1], 0, -LEFT[0]] },
    wind: { tail: [WDIR[0] * windR, -D, WDIR[1] * windR], y: -D, lift, outward: [WDIR[0], 0, WDIR[1]], scale: wScale, bar: BAR0 * wScale },
    clarity: { corner: "F", wx: -h - gauge, wz: -h - gauge, y0: 0, y1: -0.42 * D },
    depth: { corner: "R", wx: h + gauge, wz: -h - gauge, y0: 0, y1: -D },
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
      clarity: L.clarity.corner,
      depth: L.depth.corner,
    },
    sizeLen, depthLen, windY: tail[1], windInside,
    chopOutside: outside(L.chop.origin[0], L.chop.origin[2], h) && L.chop.origin[0] < -h,
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
  const sizeU = [1, 0, 0];
  const sizeO = [(L.size.a[0] + L.size.b[0]) / 2, 0, L.size.a[2]];
  const sizePlane = plane(sizeO, sizeU, [0, 0, 1]);
  const sizeLabel = plane([sizeO[0], 0, sizeO[2] + 0.035 * S], sizeU, [0, 1, 0]);
  objects.size = {
    kind: "size", edge: "AN",
    a: ui.P3(L.size.a), b: ui.P3(L.size.b),
    wa: L.size.a, wb: L.size.b, len: S,
    world: sizePlane, label: sizeLabel, corner: "AN",
  };
  const chop = plane(L.chop.origin, L.chop.along, [0, 1, 0]);
  const chopLabel = plane(
    [L.chop.origin[0] + LEFT[0] * 0.06 * S, 0, L.chop.origin[2] + LEFT[1] * 0.06 * S],
    [LEFT[0], 0, LEFT[1]], [0, 1, 0],
  );
  objects.chop = {
    kind: "chop", edge: "AF", span: L.chop.span, world: chop, label: chopLabel,
    corner: "AF", hitWave: [], len: L.chop.span,
  };
  const c = L.clarity;
  const cU = [0, 0, 1];
  const clarityPlane = plane([c.wx, 0, c.wz], cU, [0, 1, 0]);
  objects.clarity = {
    kind: "clarity", corner: "F", wx: c.wx, wz: c.wz,
    a: ui.P3([c.wx, c.y0, c.wz]), b: ui.P3([c.wx, c.y1, c.wz]),
    wa: [c.wx, c.y0, c.wz], wb: [c.wx, c.y1, c.wz],
    world: clarityPlane, label: clarityPlane, icon: clarityPlane,
  };
  const d = L.depth;
  const depthPlane = plane([d.wx, 0, d.wz], [0, 0, 1], [0, 1, 0]);
  objects.depth = {
    kind: "depth", corner: "R", wx: d.wx, wz: d.wz,
    a: ui.P3([d.wx, d.y0, d.wz]), b: ui.P3([d.wx, d.y1, d.wz]),
    wa: [d.wx, d.y0, d.wz], wb: [d.wx, d.y1, d.wz],
    world: depthPlane, label: depthPlane,
  };
  const w = L.wind;
  // Tangent with +X, so the words run beside the arrow rather than further
  // out along the ray that already reaches the heading disk.
  const tangent = [-w.outward[2], w.outward[0]];
  const labelO = [w.tail[0] + w.outward[0] * 10, w.y, w.tail[2] + w.outward[2] * 10];
  objects.wind = {
    kind: "wind", corner: "AF", tail: w.tail.slice(), lift: w.lift, y: w.y,
    outward: w.outward, parallel: true, scale: w.scale, bar: w.bar,
    label: plane(labelO, [tangent[0], 0, tangent[1]], [0, 1, 0]),
  };
  ui.debug.anchors = {
    adaptive: false,
    corners: { size: "AN", chop: "AF", clarity: "F", depth: "R", wind: "AF" },
    placed: Object.keys(objects),
    norm: report.norm,
  };
  return { ok: true, objects, anchors: ui.debug.anchors, duration: 0 };
}

// Fixed world geometry and exact projected collision polygons.
// Nothing in this module rotates toward the camera. Screen is only the view of XZ/Y objects.
import { ui } from "./core.js?v=30.27";
export const add = (a, b, t = 1) => a.map((v, i) => v + b[i] * t);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const norm = (v) => { const l = Math.hypot(...v) || 1; return v.map((n) => n / l); };
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const bounds = (ps) => [Math.min(...ps.map((q) => q[0])), Math.min(...ps.map((q) => q[1])), Math.max(...ps.map((q) => q[0])), Math.max(...ps.map((q) => q[1]))];
export function inside(p, h) { let c = false; if (!h?.length) return false; for (let i = 0, j = h.length - 1; i < h.length; j = i++) { const a = h[i], b = h[j]; if (!a || !b) return false; if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; }
const cross = (a, b, c) => (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0]);
export function intersect(a, b) {
  if (!a || !b || a.length < 2 || b.length < 2) return false;
  if (a.some((p) => !p) || b.some((p) => !p)) return false;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) {
    const c = a[i], d = a[(i + 1) % a.length], e = b[j], f = b[(j + 1) % b.length];
    if (cross(c, d, e) * cross(c, d, f) < 0 && cross(e, f, c) * cross(e, f, d) < 0) return true;
  }
  return inside(a[0], b) || inside(b[0], a);
}
export function plane(o, u, v = [0, 1, 0]) {
  const p = { o: o.slice(), u: norm(u), v: norm(v) };
  p.project = (x = 0, y = 0) => add(add(p.o, p.u, x), p.v, y);
  p.p = (x = 0, y = 0) => ui.P3(p.project(x, y));
  p.poly = (pts) => pts.map((q) => p.p(...q));
  return p;
}
// Pixels per world metre along a plane axis (0 = u, 1 = v).
export function ppm(g, axis = 1) {
  const a = g.p(0, 0), b = axis ? g.p(0, 1) : g.p(1, 0);
  if (!a || !b) return 2;
  return Math.max(0.2, Math.hypot(b[0] - a[0], b[1] - a[1]));
}
export function pxMetres(g, px, axis = 1) { return px / ppm(g, axis); }
// Raster size follows how wide the sea reads, then stops. World anchors are not involved.
// A far cube uses about 68% of the requested size (still a short readable caption);
// a close view never grows past the request.
export function readPx(size = 14) {
  if (!ui.V) return size;
  const h = ui.S() / 2;
  const a = ui.proj(-h, 0, 0), b = ui.proj(h, 0, 0);
  const span = a && b ? Math.hypot(b[0] - a[0], b[1] - a[1]) : 560;
  const k = Math.max(0.68, Math.min(1, span / 620));
  return Math.round(size * k * 10) / 10;
}
// Local-x shift that moves a label toward mid-screen, still inside the world plane.
export function towardCenter(g, px = 48) {
  const a = g.p(0, 0), b = g.p(1, 0);
  if (!a || !b || !ui.V) return 0;
  const want = a[0] > ui.V.cw / 2 ? -1 : 1;
  const sign = (b[0] - a[0]) * want >= 0 ? 1 : -1;
  return sign * pxMetres(g, px, 0);
}
// The plane is not turned. Local +u stays the line the letters run along, and local +v
// stays the glyph's up-axis inside that plane. Only the sign of each axis is chosen:
// canvas +y (the bottom of a letter) points down the screen, and canvas +x runs so the
// basis is not mirrored. A grazing view still collapses the short axis.
export function text(ctx, g, s, x, y, size = 14, color = "#E8F5FC") {
  if (s == null || s === "") return;
  const a = g.p(x, y), b = g.p(x + 1, y), c = g.p(x, y + 1);
  if (!a || !b || !c) return;
  const U = [b[0] - a[0], b[1] - a[1]], V = [c[0] - a[0], c[1] - a[1]];
  const uLen = Math.hypot(U[0], U[1]), vLen = Math.hypot(V[0], V[1]);
  if (uLen < 0.15 && vLen < 0.15) return;
  const vSign = V[1] < 0 ? -1 : 1;
  const down = [V[0] * vSign, V[1] * vSign];
  const uSign = (U[0] * down[1] - U[1] * down[0]) < 0 ? -1 : 1;
  const read = [U[0] * uSign, U[1] * uSign];
  const dom = Math.max(Math.hypot(read[0], read[1]), Math.hypot(down[0], down[1]), 1e-3);
  const px = readPx(size);
  ctx.save();
  ctx.font = `600 ${px}px Barlow,sans-serif`;
  ctx.fillStyle = color;
  // The call point stays the start of local +u. When that direction points left on
  // screen, the letters run back along the same ray so they still occupy local +u.
  const pen = uSign < 0 ? -ctx.measureText(s).width : 0;
  ctx.transform(read[0] / dom, read[1] / dom, down[0] / dom, down[1] / dom, a[0], a[1]);
  ctx.fillText(s, pen, 0);
  ctx.restore();
  const det = read[0] * down[1] - read[1] * down[0];
  const along = (p, q) => Math.abs(p[0] * q[1] - p[1] * q[0]);
  if (ui.debug._textFrame !== ui.now) { ui.debug.textOrient = []; ui.debug._textFrame = ui.now; }
  ui.debug.textOrient.push({
    s, at: [Math.round(a[0]), Math.round(a[1])],
    readX: read[0], downY: down[1], det, px, uSign, vSign,
    parallel: along(read, U) < 1e-3 && along(down, V) < 1e-3,
  });
  ui.debug.lastText = { asked: size, px, dominantPx: px, foreshorten: Math.min(uLen, vLen) / dom, readX: read[0], downY: down[1], det };
}

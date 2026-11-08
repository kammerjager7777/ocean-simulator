// Fixed world geometry and exact projected collision polygons.
// Nothing in this module rotates toward the camera. Screen is only the view of XZ/Y objects.
import { ui } from "./core.js?v=30.29";
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
// World metres that match one metre of a 250 m sea at the same display zoom.
// The eye already backs up with the domain, so a length written in these units
// keeps its apparent size. A true ruler stays in raw metres instead.
export const showM = (S = ui.S()) => S / 250;
// True when the water column sits between the eye and this point. A point beside
// the slice, or on the near side of it, is not hidden. The test is the box the
// slice occupies, not a camera-angle cutoff.
export function seaHides(p) {
  const eye = ui.V && ui.V.eye;
  if (!eye || !p) return false;
  const h = ui.S() / 2, D = Math.max(ui.D(), 0.5);
  const min = [-h, -D, -h], max = [h, 0, h];
  const dir = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 3; i++) {
    const d = dir[i];
    if (Math.abs(d) < 1e-8) {
      if (eye[i] < min[i] - 1e-4 || eye[i] > max[i] + 1e-4) return false;
      continue;
    }
    let ta = (min[i] - eye[i]) / d, tb = (max[i] - eye[i]) / d;
    if (ta > tb) { const s = ta; ta = tb; tb = s; }
    t0 = Math.max(t0, ta);
    t1 = Math.min(t1, tb);
    if (t0 > t1) return false;
  }
  return t0 < 0.97 && t1 >= t0;
}
// How wide the sea's x-axis reads, in pixels. Handles do not use this.
export function seaSpan() {
  if (!ui.V) return 560;
  const h = ui.S() / 2;
  const a = ui.proj(-h, 0, 0), b = ui.proj(h, 0, 0);
  return a && b ? Math.hypot(b[0] - a[0], b[1] - a[1]) : 560;
}
// A value stays at the size it was asked for inside the editing frame.
// A name may ease down when the sea is small, and never grows past the request.
// Close views do not enlarge type, and handles are not part of this scale.
export function readPx(size = 14, keep = false) {
  if (!ui.V || keep) return size;
  const k = Math.max(0.85, Math.min(1, seaSpan() / 480));
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
function basisAt(g, x, y) {
  const a = g.p(x, y), b = g.p(x + 1, y), c = g.p(x, y + 1);
  if (!a || !b || !c) return null;
  const U = [b[0] - a[0], b[1] - a[1]], V = [c[0] - a[0], c[1] - a[1]];
  const uLen = Math.hypot(U[0], U[1]), vLen = Math.hypot(V[0], V[1]);
  if (uLen < 0.15 && vLen < 0.15) return null;
  const vSign = V[1] < 0 ? -1 : 1;
  const down = [V[0] * vSign, V[1] * vSign];
  const uSign = (U[0] * down[1] - U[1] * down[0]) < 0 ? -1 : 1;
  const read = [U[0] * uSign, U[1] * uSign];
  const dom = Math.max(Math.hypot(read[0], read[1]), Math.hypot(down[0], down[1]), 1e-3);
  return { a, U, V, uLen, vLen, uSign, vSign, read, down, dom };
}
function noteText(s, B, px) {
  const det = B.read[0] * B.down[1] - B.read[1] * B.down[0];
  const along = (p, q) => Math.abs(p[0] * q[1] - p[1] * q[0]);
  if (ui.debug._textFrame !== ui.now) { ui.debug.textOrient = []; ui.debug.labels = []; ui.debug._textFrame = ui.now; }
  ui.debug.textOrient.push({
    s, at: [Math.round(B.a[0]), Math.round(B.a[1])],
    readX: B.read[0], downY: B.down[1], det, px, uSign: B.uSign, vSign: B.vSign,
    parallel: along(B.read, B.U) < 1e-3 && along(B.down, B.V) < 1e-3,
  });
  ui.debug.lastText = { asked: px, px, foreshorten: Math.min(B.uLen, B.vLen) / B.dom, readX: B.read[0], downY: B.down[1], det };
}
function mapGlyph(B, gx, gy) {
  return [B.a[0] + B.read[0] / B.dom * gx + B.down[0] / B.dom * gy, B.a[1] + B.read[1] / B.dom * gx + B.down[1] / B.dom * gy];
}
function roundPlate(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
// One plate behind a whole run, then the glyphs. The plate is the backdrop, so a
// bright patch in the sea cannot flip the letters. There is no stroke and no shadow.
function paintRun(ctx, B, runs, hot, center) {
  const gap = 7;
  const total = runs.reduce((s, m) => s + m.w, 0) + gap * (runs.length - 1);
  const pen0 = center ? -total / 2 : B.uSign < 0 ? -total : 0;
  const maxPx = Math.max(...runs.map((m) => m.px));
  const top = -maxPx * 0.84, bot = maxPx * 0.26, padX = 5, padY = 3;
  ctx.save();
  ctx.transform(B.read[0] / B.dom, B.read[1] / B.dom, B.down[0] / B.dom, B.down[1] / B.dom, B.a[0], B.a[1]);
  roundPlate(ctx, pen0 - padX, top - padY, total + padX * 2, bot - top + padY * 2, 4);
  ctx.fillStyle = hot ? "rgba(8, 16, 28, 0.9)" : "rgba(8, 16, 28, 0.74)";
  ctx.fill();
  ctx.fillStyle = "#F4F7FB";
  ctx.textBaseline = "alphabetic";
  let cursor = pen0;
  for (const m of runs) {
    ctx.font = `${m.keep ? 600 : 500} ${m.px}px Barlow, sans-serif`;
    ctx.fillText(m.text, cursor, 0);
    cursor += m.w + gap;
  }
  ctx.restore();
  const box = [[pen0 - padX, top - padY], [pen0 + total + padX, top - padY], [pen0 + total + padX, bot + padY], [pen0 - padX, bot + padY]].map((p) => mapGlyph(B, p[0], p[1]));
  return { box, px: maxPx };
}
// The plane is not turned. Local +u stays the line the letters run along, and local +v
// stays the glyph's up-axis inside that plane. Only the sign of each axis is chosen:
// canvas +y (the bottom of a letter) points down the screen, and canvas +x runs so the
// basis is not mirrored. A grazing view still collapses the short axis.
export function text(ctx, g, s, x, y, size = 14, color = "#F4F7FB", keep = false) {
  if (s == null || s === "") return null;
  const B = basisAt(g, x, y);
  if (!B) return null;
  const px = readPx(size, keep);
  ctx.save();
  ctx.font = `${keep ? 600 : 500} ${px}px Barlow, sans-serif`;
  const w = ctx.measureText(String(s)).width;
  ctx.restore();
  const painted = paintRun(ctx, B, [{ text: String(s), px, w, keep }], false, false);
  noteText(String(s), B, px);
  ui.debug.labels.push({ s: String(s), box: painted.box, keep });
  return painted.box;
}
// Name then value, in one plate, along local +u. Names drop when the reading axis is
// edge-on or the sea is far; the value stays. The anchor does not follow the camera.
export function textRow(ctx, g, x, y, parts, hot = false, center = false) {
  const B = basisAt(g, x, y);
  if (!B) return null;
  const far = seaSpan() < 300;
  const flat = B.uLen < 0.4;
  const shown = parts.filter((p) => p && p.text != null && p.text !== "" && (p.keep || (!far && !flat)));
  if (!shown.length) return null;
  ctx.save();
  const runs = shown.map((p) => {
    const px = readPx(p.size || 13, !!p.keep);
    ctx.font = `${p.keep ? 600 : 500} ${px}px Barlow, sans-serif`;
    return { text: String(p.text), px, w: ctx.measureText(String(p.text)).width, keep: !!p.keep };
  });
  ctx.restore();
  const painted = paintRun(ctx, B, runs, hot, center);
  for (const m of runs) noteText(m.text, B, m.px);
  ui.debug.labels.push({ s: runs.map((m) => m.text).join(" "), box: painted.box, keep: runs.some((m) => m.keep), parts: runs.map((m) => m.text) });
  return painted.box;
}

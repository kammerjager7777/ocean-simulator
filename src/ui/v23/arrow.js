// Wind arrow: fixed outside the footprint, flat on the seabed (y = -depth).
// The power bar sits above that plane. Dragging the arrow updates direction and
// speed on every move. Camera orbit does not move the tail.
import { ui } from "./core.js?v=30.27";
import { arrange } from "./controls.js?v=30.27";
import { inside, plane, text } from "./geometry.js?v=30.27";
const LO = [0, .3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 36.9];
export const bfLevel = (u) => { let i = 0; while (i < 12 && u >= LO[i + 1] - .05) i++; return Math.min(12, i + Math.min(.999, (u - LO[i]) / (LO[i + 1] - LO[i]))); };
const cols = ["#8FD8EA", "#8FD8EA", "#86DCCB", "#86DCCB", "#9BE07A", "#C4E35E", "#E9DE4E", "#FFC93F", "#FFA53A", "#FF8536", "#FF6233", "#F24432", "#E0263A"];
const ARROW = "#226789", HW = 3.3, BAR = 76;
// Grab slop around the drawn stroke only. The segment ends at the drawn tip.
const GRAB = 6;
const U = () => Number(ui.inputs.spd.value), D = () => Number(ui.inputs.dir.value), T = ui.theme;
const lenOf = (u) => 18 + Math.max(0, Math.min(25, u)) * 3.5;
const spdOf = (l) => Math.max(0, Math.min(25, (l - 18) / 3.5));
let g = null, drag = false;
ui.slider("Wind speed", () => [U(), `${U()} metres per second`], (e) => { if (!/Arrow(Up|Down)/.test(e.key)) return false; ui.set("spd", Math.max(0, Math.min(25, U() + (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 5 : .5)))); return true; });
ui.slider("Wind direction", () => [D(), `${D()} degrees`], (e) => { if (!/Arrow(Left|Right)/.test(e.key)) return false; ui.set("dir", ((D() + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 15 : 1)) % 360 + 360) % 360); return true; });

function build(tail, lift, scale = 1, barLen = BAR) {
  const y = tail[1], rad = D() * Math.PI / 180, dir = [Math.cos(rad), 0, Math.sin(rad)], perp = [-dir[2], 0, dir[0]];
  const L = lenOf(U()) * scale;
  const ag = plane([tail[0], y, tail[2]], dir, perp);
  const neck = Math.max(4 * scale, L - 12 * scale);
  const shaft = ag.poly([[0, -HW * scale], [neck, -HW * scale], [neck, -(HW + 3.4) * scale], [L + 4 * scale, 0], [neck, (HW + 3.4) * scale], [neck, HW * scale]]);
  const bar = plane([tail[0], y + lift, tail[2]], dir, perp);
  const bp = bar.poly([[0, -1.8 * scale], [barLen, -1.8 * scale], [barLen, 1.8 * scale], [0, 1.8 * scale]]);
  const head = [tail[0] + dir[0] * L, y, tail[2] + dir[2] * L];
  return { dir, perp, L, ag, shaft, bar, bp, head, a: ag.p(0), b: ag.p(L) };
}

const arrow = ui.register({
  layer: "wind",
  hit(x, y) {
    if (!g) return null;
    const p = [x, y];
    // Same endpoints the draw uses. BAR is the unscaled 76 m ruler; g.barLen is the drawn one.
    const barD = g.barA && g.barB ? ui.segDist(p, g.barA, g.barB) : Infinity;
    const shaftD = g.a && g.b ? ui.segDist(p, g.a, g.b) : Infinity;
    const shown = !!ui.layers.beaufort;
    const inBarPoly = !!(shown && g.bp?.every(Boolean) && inside(p, g.bp));
    const inShaftPoly = !!(g.shaft?.every(Boolean) && inside(p, g.shaft));
    // The drawn arrow, including its head, is the direction target even where
    // the raised bar projects nearby. The bar is speed only on its own shape,
    // and only while that shape is actually drawn.
    if (inShaftPoly || shaftD <= GRAB) return "head";
    if (shown && (inBarPoly || barD <= GRAB)) return "bar";
    return null;
  },
  body(x, y) {
    if (!g) return null;
    const p = [x, y];
    if (g.shaft?.every(Boolean) && inside(p, g.shaft)) return "head";
    if (ui.layers.beaufort && g.bp?.every(Boolean) && inside(p, g.bp)) return "bar";
    return null;
  },
  down(h) { drag = true; ui.drag.key = h; },
  move(h, x, y, e) {
    if (!drag || !g) return;
    if (h === "bar" && g.bar) {
      const distance = (t) => { const q = g.bar.p((g.barLen || BAR) * t, 0); return q ? Math.hypot(x - q[0], y - q[1]) : Infinity; };
      let best = 0, error = Infinity;
      for (let i = 0; i <= 80; i++) { const t = i / 80, err = distance(t); if (err < error) { best = t; error = err; } }
      ui.set("spd", Math.round(Math.max(0, Math.min(25, best * 25)) * 2) / 2);
      return;
    }
    const raw = ui.toPlane(x, y, g.tail[1]); if (!raw) return;
    const dx = raw[0] - g.tail[0], dz = raw[2] - g.tail[2], l = Math.hypot(dx, dz);
    if (l < 3) return;
    let dd = Math.atan2(dz, dx) * 180 / Math.PI;
    if (e.shiftKey) dd = Math.round(dd / 15) * 15;
    ui.set("dir", ((Math.round(dd) % 360) + 360) % 360);
    ui.set("spd", Math.round(spdOf(l / (g.scale || 1)) * 2) / 2);
  },
  up() { drag = false; },
  cursor: (h) => (h === "bar" ? "ew-resize" : "grab"),
  draw() {
    const q = arrange().objects.wind;
    if (!q?.tail) { g = null; ui.debug.arrow = null; return; }
    const built = build(q.tail, q.lift || 6, q.scale || 1, q.bar || BAR);
    const barA = built.bar.p(0, 0), barB = built.bar.p(q.bar || BAR, 0);
    g = { ...built, tail: q.tail.slice(), barLen: q.bar || BAR, scale: q.scale || 1, barA, barB };
    if (built.shaft?.every(Boolean)) {
      T.fill(ui.ctx, built.shaft, ARROW, drag ? 1 : 0.94);
      T.line(ui.ctx, [...built.shaft, built.shaft[0]], { color: "#E7F3F8", width: 1.7, alpha: 0.9 });
    }
    if (ui.layers.beaufort && built.bp?.every(Boolean)) {
      const f = bfLevel(U());
      for (let i = 0; i < 12; i++) {
        const bl = q.bar || BAR, sc = q.scale || 1;
        const seg = built.bar.poly([[i / 12 * bl, -1.8 * sc], [(i + 1) / 12 * bl, -1.8 * sc], [(i + 1) / 12 * bl, 1.8 * sc], [i / 12 * bl, 1.8 * sc]]);
        if (seg.every(Boolean)) T.fill(ui.ctx, seg, cols[i + 1], f > i ? 0.9 : 0.16);
      }
    }
    const lift = q.lift || 6;
    // Baselines are one glyph apart along this same plane. A few fixed metres
    // collapse into one blob once the sea is far, because the type stays a
    // screen size. The tail does not move.
    const a0 = q.label.p(0, 0), au = q.label.p(1, 0), av = q.label.p(0, 1);
    const dom = a0 && au && av ? Math.max(Math.hypot(au[0] - a0[0], au[1] - a0[1]), Math.hypot(av[0] - a0[0], av[1] - a0[1]), 0.2) : 1;
    const step = 1.7 * 15 / dom;
    const y0 = lift + step;
    const lines = [[`${U()} m/s`, y0 + step * 2, 15], [`${String(Math.round(D())).padStart(3, "0")}°`, y0 + step, 12], [`BFT ${Math.floor(bfLevel(U()))}`, y0, 11]];
    for (const [s, y, px] of lines) text(ui.ctx, q.label, s, 0, y, px, "#E8F5FC");
    ui.debug.windLines = lines.map(([s, y]) => ({ s, at: q.label.p(0, y) }));
    ui.debug.arrow = {
      tail: built.a, head: built.b, worldTail: q.tail, worldHead: built.head,
      d: built.dir, physicalY: q.tail[1], worldLength: built.L, color: ARROW, widthMetres: HW * 2,
      parallel: true, corner: q.corner, live: drag, barY: q.tail[1] + (q.lift || 0), layoutClear: !ui.debug.separation?.windInside,
      barA: built.bar.p(0, 0), barB: built.bar.p(q.bar || BAR, 0), scale: q.scale || 1,
    };
  },
  hide() { g = null; },
});
ui.debug.windHit = (x, y) => arrow.hit(x, y);
window.__oceanArrow = () => g;
void arrow;

// v30.27 ripple cursor plus the click's own ring. The ring is the packet on the water,
// a few metres across, not a screen-space splash. Height and foam come from ripples.js.
import { ui } from "./core.js?v=32.10";
import { rippleRings } from "../../physics/ripples.js?v=30.29";
const T = ui.theme;
function ringAt(x, z, r, y) {
  const pts = [];
  for (let i = 0; i < 28; i++) {
    const an = (i / 28) * 2 * Math.PI;
    const q = ui.proj(x + r * Math.cos(an), y, z + r * Math.sin(an));
    if (q) pts.push(q);
  }
  return pts;
}
ui.register({
  layer: "ripple",
  draw() {
    const simT = ui.simT();
    for (const s of rippleRings(simT)) {
      const pts = ringAt(s.x, s.z, s.r, 0.05);
      if (pts.length > 8) T.line(ui.ctx, pts, { color: "#F4FBFF", width: 1.6, alpha: 0.28 + 0.45 * s.a, close: true });
      if (s.age < 0.7) {
        const inner = ringAt(s.x, s.z, Math.max(0.6, s.r * 0.45), 0.08);
        if (inner.length > 8) T.line(ui.ctx, inner, { color: "#FFFFFF", width: 1.2, alpha: 0.35 * (1 - s.age / 0.7), close: true });
      }
    }
    if (!ui.mouse || ui.hover || ui.drag) return;
    const p = ui.onWater(ui.mouse[0], ui.mouse[1]); if (!p) return;
    const c = ui.proj(p[0], 0, p[2]); if (!c) return;
    const e = ui.proj(p[0] + 1, 0, p[2]), f = ui.proj(p[0], 0, p[2] + 1);
    const ppm = Math.max(1e-3, e ? Math.hypot(e[0] - c[0], e[1] - c[1]) : 1, f ? Math.hypot(f[0] - c[0], f[1] - c[1]) : 1), r = 12 / ppm;
    const ring = ringAt(p[0], p[2], r, 0);
    const fl = ui.now - (ui.rippleFlash || -1e9) < 160;
    T.line(ui.ctx, ring, { color: T.accent, width: 1.4, alpha: fl ? 1 : 0.75, close: true });
    T.circle(ui.ctx, c[0], c[1], 1.6, { fill: T.accent });
  },
});

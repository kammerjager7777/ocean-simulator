// Measurement handles. Chop is only the wave glyph: dragging that stroke changes choppiness live.
// Clarity and depth stay on the two different corners arrange() assigned.
import { ui } from "./core.js?v=30.27";
import { arrange } from "./controls.js?v=30.27";
import { text, inside } from "./geometry.js?v=30.27";
import { sample, release } from "./coast.js?v=30.27";
const num = (k) => Number(ui.inputs[k].value), clamp = (v) => Math.min(1, Math.max(0, v)), T = ui.theme;
export const zsdBand = (z) => z < 6 ? "Murky" : z < 15 ? "Coastal" : z < 30 ? "Open ocean" : "Clear tropical";
export const zsdOfV = (v) => 3 * Math.pow(40 / 3, v), vOfZsd = (z) => Math.log(z / 3) / Math.log(40 / 3);
const CHOP_PX = 34;

function nearLine(x, y, pts, r) {
  if (!pts || pts.length < 2) return false;
  for (let i = 1; i < pts.length; i++) if (pts[i - 1] && pts[i] && ui.segDist([x, y], pts[i - 1], pts[i]) <= r) return true;
  return false;
}
function wavePoints(g, span, chop) {
  const S = ui.S();
  const amp = (0.006 + 0.016 * chop) * S;
  const ph = ui.reduced ? 0.4 : (ui.now / 1000) * 1.15;
  const pts = [];
  for (let i = 0; i <= 36; i++) {
    const t = i / 36;
    const y = 0.008 * S + amp * (0.5 - 0.5 * Math.cos(t * Math.PI * 4 + ph));
    const p = g.p((t - 0.5) * span, y);
    if (p) pts.push(p);
  }
  return pts;
}

function make(o) {
  const st = { g: null };
  const kb = ui.slider(o.aria, () => [o.get(), o.vt(o.get())], (e) => {
    const v = o.keys(e); if (v == null) return false; o.set(v); return true;
  });
  const obj = ui.register({
    layer: "tools",
    key: o.key,
    body(x, y) {
      const g = st.g; if (!g) return null;
      // The painted mark only. The knob disk is about 5.5 px; 8 px covers that disk.
      if (o.key === "chop") return nearLine(x, y, g.hitWave, 4) ? "wave" : null;
      if (o.key === "clarity" && g.icon) {
        const icon = g.icon.poly([[-1, 4], [8, 4], [8, 16], [-1, 16]]);
        if (icon && icon.every(Boolean) && inside([x, y], icon)) return "icon";
      }
      if (g.k && Math.hypot(x - g.k[0], y - g.k[1]) < 8) return "k";
      return null;
    },
    hit(x, y) {
      const g = st.g; if (!g) return null;
      if (o.key === "chop") return nearLine(x, y, g.hitWave, 18) ? "wave" : null;
      if (o.key === "clarity" && g.icon) {
        const icon = g.icon.poly([[-1, 4], [8, 4], [8, 16], [-1, 16]]);
        if (icon && icon.every(Boolean) && inside([x, y], icon)) return "icon";
      }
      if (!g.a || !g.b) return null;
      return (g.k && Math.hypot(x - g.k[0], y - g.k[1]) < 16) || ui.segDist([x, y], g.a, g.b) < 10 ? "k" : null;
    },
    down(h, x, y) {
      ui.drag.key = o.key; st.dragging = h; st.value = o.get(); st.y = y; st.x = x;
      if (o.key === "size") sample("size", st.value, performance.now());
    },
    move(h, x, y) {
      if (o.key === "chop" && (h === "wave" || h === "k")) {
        o.set(st.value + (st.y - y) / CHOP_PX); kb.upd(); return;
      }
      const g = st.g; if (!g || !g.a || !g.b) return;
      if (o.key === "depth") {
        let best = num("depth"), err = Infinity;
        for (let i = 0; i <= 156; i++) {
          const d = 2 + (78 * i) / 156, p = ui.proj(g.wx, -d, g.wz);
          if (!p) continue;
          const e = Math.hypot(p[0] - x, p[1] - y);
          if (e < err) { err = e; best = d; }
        }
        ui.set("depth", Math.max(2, Math.min(80, Math.round(best))));
      } else if (o.key === "clarity") ui.set("clarity", Math.round(zsdOfV(clamp(ui.segT([x, y], g.a, g.b)))));
      else if (o.key === "size") {
        ui.set("size", Math.round((100 + 900 * clamp(ui.segT([x, y], g.a, g.b))) / 10) * 10);
        sample("size", num("size"), performance.now());
      }
      kb.upd();
    },
    up() {
      if (o.key === "size" && st.dragging) {
        release("size", {
          min: 100, max: 1000, maxAbs: 420, ignore: 80, tau: 0.16, stop: 18,
          on: (v) => ui.set("size", Math.max(100, Math.min(1000, Math.round(v / 10) * 10))),
        });
      }
      st.dragging = null;
    },
    cursor: () => (o.key === "chop" ? "ns-resize" : "grab"),
    draw() {
      const laid = arrange();
      const q = laid.objects[o.key];
      if (!q) { st.g = null; delete ui.debug[o.key]; return; }
      st.g = q;
      const value = o.get();
      const hot = st.dragging ? "drag" : "idle";
      const S = ui.S();
      if (o.key === "chop") {
        q.hitWave = wavePoints(q.world, q.span, num("chop"));
        if (q.hitWave.length > 1) T.line(ui.ctx, q.hitWave, { color: hot === "drag" ? T.inkAc : T.white, width: hot === "drag" ? 3.4 : 2.8, alpha: 0.96 });
        const yText = 0.02 * S;
        text(ui.ctx, q.label, num("chop").toFixed(1), 0, yText + 0.016 * S, 15, "#E8F5FC", false);
        text(ui.ctx, q.label, "CHOPPINESS", 0, yText, 11, "#E8F5FC", false);
        ui.debug.chop = { v: value, hitWave: q.hitWave, corner: q.corner, fixedPlane: true, target: "wave", span: q.span, k: q.hitWave[Math.floor(q.hitWave.length / 2)] || null };
        return;
      }
      if (!q.a || !q.b) {
        ui.debug[o.key] = { corner: q.corner, wa: q.wa, wb: q.wb, fixedPlane: true, v: value };
        return;
      }
      const t = o.key === "size" ? (num("size") - 100) / 900 : o.key === "clarity" ? vOfZsd(num("clarity")) : 1;
      q.k = [q.a[0] + (q.b[0] - q.a[0]) * t, q.a[1] + (q.b[1] - q.a[1]) * t];
      T.track(ui.ctx, [q.a, q.b], hot, hot === "drag" ? T.inkAc : T.white, o.key === "depth" ? 6 : 5);
      T.knob(ui.ctx, ...q.k, hot, 0, T.white);
      const gap = o.key === "size" ? 0.03 * S : 0.08 * ui.D();
      text(ui.ctx, q.label, o.val(), 0, gap + 0.025 * S, 15);
      text(ui.ctx, q.label, o.key === "size" ? "SIZE" : o.key === "depth" ? "SEABED" : `CLARITY · ${zsdBand(value).toUpperCase()}`, 0, gap, 11);
      if (o.key === "depth") {
        for (const d of [0, 20, 40, 60, 80]) {
          if (d > num("depth") + 0.1) continue;
          const p = ui.proj(q.wx, -d, q.wz);
          if (p) T.circle(ui.ctx, p[0], p[1], 1.7, { fill: T.white });
        }
      }
      if (o.key === "clarity" && q.icon) {
        const ig = q.icon, col = [[-1, 3], [8, 3], [8, 15], [-1, 15]].map((v) => ig.p(...v));
        if (col.every(Boolean)) {
          T.fill(ui.ctx, col, `rgb(${Math.round(80 + 45 * (1 - vOfZsd(num("clarity"))))},145,145)`, 0.82);
          T.line(ui.ctx, [...col, col[0]], { color: T.white, width: 1, alpha: 0.85 });
        }
      }
      ui.debug[o.key] = { a: q.a, b: q.b, k: q.k, v: value, corner: q.corner, fixedPlane: true, world: { o: q.world.o, u: q.world.u, v: q.world.v } };
    },
  });
  return obj;
}
make({ key: "size", label: "Size", aria: "Ocean size", get: () => num("size"), vt: (v) => `${v} metres`, val: () => `${num("size")} m`, set: (v) => ui.set("size", Math.min(1000, Math.max(100, v))), keys: (e) => /Arrow(Left|Right)/.test(e.key) ? num("size") + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 100 : 10) : null });
make({ key: "chop", label: "Choppiness", aria: "Choppiness", get: () => num("chop"), vt: (v) => `${Number(v).toFixed(1)}`, val: () => num("chop").toFixed(1), set: (v) => ui.set("chop", Math.min(2.5, Math.max(0, Math.round(v * 10) / 10)).toFixed(1)), keys: (e) => /Arrow(Left|Right)/.test(e.key) ? num("chop") + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 0.5 : 0.1) : null });
make({ key: "depth", label: "Seabed", aria: "Seabed depth", get: () => num("depth"), vt: (v) => `${v} metres`, val: () => `${num("depth")} m`, set: (v) => ui.set("depth", Math.min(80, Math.max(2, v))), keys: (e) => /Arrow(Up|Down)/.test(e.key) ? num("depth") + (e.key === "ArrowDown" ? 1 : -1) * (e.shiftKey ? 10 : 1) : null });
make({ key: "clarity", label: "Clarity", aria: "Water clarity", get: () => num("clarity"), vt: (z) => `${Math.round(z)} metres visibility, ${zsdBand(z).toLowerCase()}`, val: () => `${Math.round(num("clarity"))} m`, set: (z) => ui.set("clarity", Math.min(40, Math.max(3, Math.round(z)))), keys: (e) => /Arrow(Up|Down)/.test(e.key) ? Math.round(num("clarity")) + (e.key === "ArrowDown" ? 1 : -1) * (e.shiftKey ? 5 : 1) : null });

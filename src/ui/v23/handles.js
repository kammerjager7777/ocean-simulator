// Measurement handles. Chop is only the wave glyph: dragging that stroke changes choppiness live.
// Clarity and depth stay on the two different corners arrange() assigned.
import { ui } from "./core.js?v=30.29";
import { arrange } from "./controls.js?v=30.29";
import { textRow, inside, seaHides, showM } from "./geometry.js?v=30.29";
import { cancel } from "./coast.js?v=30.29";
const num = (k) => Number(ui.inputs[k].value), clamp = (v) => Math.min(1, Math.max(0, v)), T = ui.theme;
export const zsdBand = (z) => z < 6 ? "Murky" : z < 15 ? "Coastal" : z < 30 ? "Open ocean" : "Clear tropical";
export const zsdOfV = (v) => 3 * Math.pow(40 / 3, v), vOfZsd = (z) => Math.log(z / 3) / Math.log(40 / 3);
const CHOP_PX = 34;
// Full Secchi range, in pixels along the projected column. A large sea backs the
// camera up, so the real column can be only a few pixels. The drawn disk stays
// on that column. The drag still travels this far along the same axis, so the
// value does not collapse into one step. Longer columns keep a 1:1 slide.
const CLAR_DRAG_PX = 72;

function nearLine(x, y, pts, r) {
  if (!pts || pts.length < 2) return false;
  for (let i = 1; i < pts.length; i++) if (pts[i - 1] && pts[i] && ui.segDist([x, y], pts[i - 1], pts[i]) <= r) return true;
  return false;
}
function drawRuns(pts, style) {
  let run = [];
  const flush = () => { if (run.length > 1) T.line(ui.ctx, run, style); run = []; };
  for (const p of pts) { if (p) run.push(p); else flush(); }
  flush();
}
function wavePoints(g, span, chop) {
  const m = showM();
  const amp = (1.5 + 4 * chop) * m;
  const ph = ui.reduced ? 0.4 : (ui.now / 1000) * 1.15;
  const pts = [];
  for (let i = 0; i <= 36; i++) {
    const t = i / 36;
    const y = 2 * m + amp * (0.5 - 0.5 * Math.cos(t * Math.PI * 4 + ph));
    const w = g.project((t - 0.5) * span, y);
    const p = w && !seaHides(w) ? g.p((t - 0.5) * span, y) : null;
    pts.push(p);
  }
  return pts;
}
function worldAt(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
function drawDimension(a, b) {
  const samples = [];
  const ends = [];
  for (let i = 0; i <= 32; i++) {
    const w = worldAt(a, b, i / 32);
    const p = !seaHides(w) ? ui.proj(w[0], w[1], w[2]) : null;
    samples.push(p);
    if (i === 0 || i === 32) ends.push(p);
  }
  drawRuns(samples, { color: "#F4F7FB", width: 1.25, alpha: 0.88 });
  const vis = samples.filter(Boolean);
  if (vis.length < 2) return;
  const dx = vis[vis.length - 1][0] - vis[0][0], dy = vis[vis.length - 1][1] - vis[0][1];
  const l = Math.hypot(dx, dy) || 1, nx = -dy / l * 8, ny = dx / l * 8;
  for (const p of ends) if (p) T.line(ui.ctx, [[p[0] - nx, p[1] - ny], [p[0] + nx, p[1] + ny]], { color: "#F4F7FB", width: 1.15, alpha: 0.9 });
}
function seam(ctx, pts, hot) {
  let run = [];
  const flush = () => {
    if (run.length < 2) { run = []; return; }
    ctx.save();
    ctx.lineCap = "butt";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(run[0][0], run[0][1]);
    for (let i = 1; i < run.length; i++) ctx.lineTo(run[i][0], run[i][1]);
    ctx.strokeStyle = "rgba(4, 8, 12, 0.92)";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = hot === "idle" ? "rgba(118, 130, 146, 0.42)" : "rgba(214, 222, 232, 0.55)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
    run = [];
  };
  for (const p of pts) { if (p) run.push(p); else flush(); }
  flush();
}
function flatGrip(ctx, x, y, tx, ty, hot) {
  const l = Math.hypot(tx, ty) || 1, px = -ty / l, py = tx / l, hl = 7;
  ctx.save();
  ctx.lineCap = "butt";
  ctx.lineWidth = hot === "drag" ? 4.5 : hot === "hover" ? 4 : 3.5;
  ctx.strokeStyle = hot === "drag" ? "#FFFFFF" : hot === "hover" ? "#F5F7FB" : "#C5CED8";
  ctx.beginPath();
  ctx.moveTo(x + px * hl, y + py * hl);
  ctx.lineTo(x - px * hl, y - py * hl);
  ctx.stroke();
  ctx.restore();
}
function dragValue(ctx, x, y, text, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.font = "500 12px Geist Mono, ui-monospace, monospace";
  const w = ctx.measureText(text).width;
  const padX = 5, h = 16;
  const left = x + 10, top = y - 20;
  ctx.fillStyle = "#101820";
  ctx.beginPath();
  ctx.roundRect(left, top, w + padX * 2, h, 3);
  ctx.fill();
  ctx.fillStyle = "#F5F7FB";
  ctx.textBaseline = "middle";
  ctx.fillText(text, left + padX, top + h / 2);
  ctx.restore();
}
function screenTag(ctx, x, y, text) {
  ctx.save();
  ctx.font = "600 13px Geist Mono, ui-monospace, monospace";
  const w = ctx.measureText(text).width;
  const left = x - w - 8;
  ctx.fillStyle = "rgba(15,20,28,0.82)";
  ctx.beginPath();
  ctx.roundRect(left - 6, y - 11, w + 12, 22, 6);
  ctx.fill();
  ctx.fillStyle = "#F5F7FB";
  ctx.textBaseline = "middle";
  ctx.fillText(text, left, y);
  ctx.restore();
}
function diskCircle(r, y, n = 24) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push([Math.cos(a) * r, y + Math.sin(a) * r]);
  }
  return pts;
}
function diskParts(r, y) {
  const parts = [];
  for (let q = 0; q < 4; q++) {
    const a0 = q * Math.PI / 2;
    const pts = [[0, y]];
    for (let i = 0; i <= 10; i++) {
      const a = a0 + (Math.PI / 2) * (i / 10);
      pts.push([Math.cos(a) * r, y + Math.sin(a) * r]);
    }
    parts.push(pts);
  }
  return parts;
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
      if (o.key === "clarity" || o.key === "chop") return null;
      const g = st.g; if (!g) return null;
      // The painted mark only. The knob disk is about 5.5 px; 8 px covers that disk.
      if (o.key === "chop") return nearLine(x, y, g.hitWave, 4) ? "wave" : null;
      if (o.key === "clarity" && g.diskScreen) {
        if (g.diskWorld && seaHides(g.diskWorld)) return null;
        if (g.diskScreen.some((poly) => poly.every(Boolean) && inside([x, y], poly))) return "icon";
        if (g.diskAt && Math.hypot(x - g.diskAt[0], y - g.diskAt[1]) < 8) return "icon";
      }
      if (g.k && g.kw && !seaHides(g.kw) && Math.hypot(x - g.k[0], y - g.k[1]) < 12) return "k";
      return null;
    },
    hit(x, y) {
      if (o.key === "clarity" || o.key === "chop") return null;
      const g = st.g; if (!g) return null;
      if (o.key === "chop") return nearLine(x, y, g.hitWave, 18) ? "wave" : null;
      if (o.key === "clarity") {
        if (!g.diskScreen || (g.diskWorld && seaHides(g.diskWorld))) return null;
        if (g.diskScreen.some((poly) => poly.every(Boolean) && inside([x, y], poly))) return "icon";
        if (g.diskAt && Math.hypot(x - g.diskAt[0], y - g.diskAt[1]) < 14) return "icon";
        return null;
      }
      if (!g.a || !g.b || !g.wa || !g.wb) return null;
      const nearKnob = g.k && Math.hypot(x - g.k[0], y - g.k[1]) < 12;
      const nearTrack = ui.segDist([x, y], g.a, g.b) < 10;
      if (!nearKnob && !nearTrack) return null;
      const t = Math.min(1, Math.max(0, nearKnob && g.k ? ui.segT(g.k, g.a, g.b) : ui.segT([x, y], g.a, g.b)));
      if (seaHides(worldAt(g.wa, g.wb, t))) return null;
      return "k";
    },
    down(h, x, y) {
      ui.drag.key = o.key; st.dragging = h; st.value = o.get(); st.y = y; st.x = x;
      st.clar = null;
      if (o.key === "clarity" && st.g && st.g.a && st.g.b) {
        const dx = st.g.b[0] - st.g.a[0], dy = st.g.b[1] - st.g.a[1];
        const span = Math.hypot(dx, dy);
        // Under a pixel the column is edge-on and has no direction. Do not invent one.
        if (span >= 1) st.clar = { ux: dx / span, uy: dy / span, span, v0: vOfZsd(st.value) };
      }
    },
    move(h, x, y) {
      if (o.key === "chop" && (h === "wave" || h === "k")) {
        o.set(st.value + (st.y - y) / CHOP_PX); kb.upd(); return;
      }
      const g = st.g; if (!g || !g.a || !g.b) return;
      if (o.key === "depth") {
        let best = num("depth"), err = Infinity;
        for (let i = 0; i <= 156; i++) {
          const d = 2 + (78 * i) / 156, w = [g.wx, -d, g.wz];
          if (seaHides(w)) continue;
          const p = ui.proj(w[0], w[1], w[2]);
          if (!p) continue;
          const e = Math.hypot(p[0] - x, p[1] - y);
          if (e < err) { err = e; best = d; }
        }
        ui.set("depth", Math.max(2, Math.min(80, Math.round(best))));
      } else if (o.key === "clarity") {
        // Same projected axis the column draws, surface toward the deep end.
        // Screen-up is not a direction. A short column uses CLAR_DRAG_PX for the
        // full range so the value can still be set; the disk stays on the column.
        const c = st.clar; if (!c) return;
        const along = (x - st.x) * c.ux + (y - st.y) * c.uy;
        ui.set("clarity", Math.round(zsdOfV(clamp(c.v0 + along / Math.max(c.span, CLAR_DRAG_PX)))));
      }
      else if (o.key === "size") {
        ui.set("size", Math.round((100 + 900 * clamp(ui.segT([x, y], g.a, g.b))) / 10) * 10);
      }
      kb.upd();
    },
    up() {
      if (o.key === "size" || o.key === "depth") cancel(o.key);
      if (st.dragging) st.fade = 1;
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
      if (o.key === "chop" || o.key === "clarity") { st.g = null; delete ui.debug[o.key]; return; }
      if (!q.a || !q.b) {
        ui.debug[o.key] = { corner: q.corner, wa: q.wa, wb: q.wb, fixedPlane: true, v: value };
        return;
      }
      const t = o.key === "size" ? (num("size") - 100) / 900 : 1;
      q.kw = worldAt(q.wa, q.wb, t);
      q.k = ui.proj(q.kw[0], q.kw[1], q.kw[2]);
      const samples = [];
      for (let i = 0; i <= 24; i++) {
        const w = worldAt(q.wa, q.wb, i / 24);
        samples.push(!seaHides(w) ? ui.proj(w[0], w[1], w[2]) : null);
      }
      const live = st.dragging ? "drag" : (ui.hover && ui.hover.obj === obj) ? "hover" : "idle";
      if (live === "drag") st.fade = 1;
      else if (ui.reduced) st.fade = 0;
      else if (st.fade > 0) st.fade *= Math.exp(-(ui.dt || 0.016) / 0.28);
      seam(ui.ctx, samples, live);
      if (q.k && q.a && q.b && !seaHides(q.kw)) {
        flatGrip(ui.ctx, q.k[0], q.k[1], q.b[0] - q.a[0], q.b[1] - q.a[1], live);
        if (st.fade > 0.04) dragValue(ui.ctx, q.k[0], q.k[1], o.val(), st.fade);
      }
      ui.debug[o.key] = {
        a: q.a, b: q.b, k: q.k, v: value, corner: q.corner, fixedPlane: true,
        world: { o: q.world.o, u: q.world.u, v: q.world.v },
        icon: o.key === "clarity" ? q.diskScreen && q.diskScreen.flat() : null,
        disk: q.diskAt || null, hidden: q.kw ? seaHides(q.kw) : false,
        dimA: q.dimA && !seaHides(q.dimA) ? ui.proj(q.dimA[0], q.dimA[1], q.dimA[2]) : null,
        dimB: q.dimB && !seaHides(q.dimB) ? ui.proj(q.dimB[0], q.dimB[1], q.dimB[2]) : null,
        live, fade: st.fade || 0,
      };
    },
  });
  return obj;
}
make({ key: "size", label: "Size", aria: "Ocean size", get: () => num("size"), vt: (v) => `${v} metres`, val: () => `${num("size")} m`, set: (v) => ui.set("size", Math.min(1000, Math.max(100, v))), keys: (e) => /Arrow(Left|Right)/.test(e.key) ? num("size") + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 100 : 10) : null });
make({ key: "depth", label: "Depth", aria: "Seabed depth", get: () => num("depth"), vt: (v) => `${v} metres`, val: () => `${num("depth")} m`, set: (v) => ui.set("depth", Math.min(80, Math.max(2, v))), keys: (e) => /Arrow(Up|Down)/.test(e.key) ? num("depth") + (e.key === "ArrowDown" ? 1 : -1) * (e.shiftKey ? 10 : 1) : null });

// Measurement handles. Chop is only the wave glyph: dragging that stroke changes choppiness live.
// Clarity and depth stay on the two different corners arrange() assigned.
import { ui } from "./core.js?v=32.10";
import { arrange } from "./controls.js?v=30.29";
import { textRow, inside, seaHides, showM } from "./geometry.js?v=30.29";
import { cancel } from "./coast.js?v=30.29";
import { bedY } from "../../render/seabed.js?v=30.29";
import { rippleHeight } from "../../physics/ripples.js?v=30.29";
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

// v31.8: the size and depth grooves are gone. The four top edges of the domain are the
// size affordance (a corner sets the one square size, 100..1000 m). The four vertical
// corners are the depth affordance (2..80 m). Both follow the displaced surface.
const EDGE_N = 32;
const PROBE_EDGE = 84;
let edgeCache = null, edgeFrame = -1;
function cornerPx() { return ui.HIT > 18 ? 26 : 16; }
function shaftPx() { return ui.HIT > 18 ? 18 : 11; }
function domainEdges() {
  if (edgeCache && edgeFrame === ui.now) return edgeCache;
  edgeFrame = ui.now;
  const S = ui.S(), D = ui.D(), h = S / 2;
  const st = globalThis.__oceanState;
  const pd = st && st.probe && st.probe.data;
  const have = pd && pd.length >= PROBE_EDGE * 4 + 4 * EDGE_N;
  const t = ui.simT();
  const heightAt = (edge, i, x, z) => {
    let y = 0;
    if (have) y = pd[PROBE_EDGE * 4 + edge * EDGE_N + i];
    y += rippleHeight(x, z, t);
    return Math.max(y, bedY(x, z, S, D) + 0.05);
  };
  const place = (edge, i) => {
    const u = i / (EDGE_N - 1);
    let x = 0, z = 0;
    if (edge === 0) { x = -h + 2 * h * u; z = h; }
    else if (edge === 1) { x = h - 2 * h * u; z = -h; }
    else if (edge === 2) { x = -h; z = h - 2 * h * u; }
    else { x = h; z = -h + 2 * h * u; }
    const y = heightAt(edge, i, x, z);
    const w = [x, y, z];
    return { w, p: seaHides(w) ? null : ui.proj(x, y, z), x, z, y };
  };
  const tops = [0, 1, 2, 3].map((e) => Array.from({ length: EDGE_N }, (_, i) => place(e, i)));
  const pick = (edge, i) => tops[edge][i];
  const corners = [[0, 0, -1, 1], [0, EDGE_N - 1, 1, 1], [1, 0, 1, -1], [1, EDGE_N - 1, -1, -1]]
    .map(([edge, i, sx, sz]) => ({ ...pick(edge, i), sx, sz }));
  const verts = corners.map((c) => {
    const bed = bedY(c.x, c.z, S, D);
    const w = [c.x, bed, c.z];
    // The bed point sits inside the water column, so the hull test hides every
    // wall. If the surface corner is on screen, the wall under it is the one to drag.
    return { top: c, bed, bot: c.p ? ui.proj(c.x, bed, c.z) : null, w };
  });
  edgeCache = { tops, corners, verts, S, D };
  ui.debug.edges = { have: !!have, corners: corners.map((c) => c.p), S, D };
  return edgeCache;
}
function paintChain(ctx, pts, cursor, base, hot, width) {
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1].p, b = pts[i].p;
    if (!a || !b) continue;
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    let w = 0;
    if (cursor) { const d = Math.hypot(mx - cursor[0], my - cursor[1]); w = Math.exp(-(d * d) / (2 * 72 * 72)); }
    const alpha = base + (hot - base) * w;
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = `rgba(186, 192, 198, ${alpha})`;
    ctx.lineWidth = width + 1.1 * w;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
  }
}
const SIZE_MIN = 100, SIZE_MAX = 1000;
function snapSize(v, step) {
  return Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round(v / step) * step));
}
function rayOnPlane(sx, sy, level = 0) {
  if (!ui.VI || !ui.V) return null;
  const nx = (sx / ui.V.cw) * 2 - 1, ny = 1 - (sy / ui.V.ch) * 2, m = ui.VI;
  const un = (z) => {
    const w = m[3] * nx + m[7] * ny + m[11] * z + m[15];
    return [(m[0] * nx + m[4] * ny + m[8] * z + m[12]) / w, (m[1] * nx + m[5] * ny + m[9] * z + m[13]) / w, (m[2] * nx + m[6] * ny + m[10] * z + m[14]) / w];
  };
  const a = un(-1), b = un(1);
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const slope = Math.abs(dy) / Math.max(Math.hypot(dx, dz), 1e-6);
  if (Math.abs(dy) < 1e-6) return { p: null, slope: 0 };
  const t = (level - a[1]) / dy;
  if (!(t >= 0)) return { p: null, slope };
  return { p: [a[0] + dx * t, level, a[2] + dz * t], slope };
}
function paintDoubleArrow(ctx, a, b) {
  if (!a || !b) return;
  const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy);
  if (l < 8) return;
  const ux = dx / l, uy = dy / l;
  ctx.save();
  ctx.strokeStyle = "rgba(226, 230, 234, 0.92)";
  ctx.fillStyle = "rgba(226, 230, 234, 0.92)";
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  ctx.moveTo(a[0] + ux * 6, a[1] + uy * 6);
  ctx.lineTo(b[0] - ux * 6, b[1] - uy * 6);
  ctx.stroke();
  const head = (tip, dirx, diry) => {
    const px = -diry, py = dirx;
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - dirx * 7 + px * 3.1, tip[1] - diry * 7 + py * 3.1);
    ctx.lineTo(tip[0] - dirx * 7 - px * 3.1, tip[1] - diry * 7 - py * 3.1);
    ctx.closePath();
    ctx.fill();
  };
  head(b, ux, uy);
  head(a, -ux, -uy);
  ctx.restore();
}
function sizeCursor(sign) {
  const h = ui.S() / 2;
  const p0 = ui.proj(sign[0] * h, 0, sign[1] * h);
  const p1 = ui.proj(sign[0] * (h + 24), 0, sign[1] * (h + 24));
  if (!p0 || !p1) return "nwse-resize";
  return (p1[0] - p0[0]) * (p1[1] - p0[1]) >= 0 ? "nwse-resize" : "nesw-resize";
}
function diagArrow(ctx, sign, half, y) {
  const m = showM();
  const reach = 16 * m;
  const a = ui.proj(sign[0] * half - sign[0] * reach, y, sign[1] * half - sign[1] * reach);
  const b = ui.proj(sign[0] * half + sign[0] * reach, y, sign[1] * half + sign[1] * reach);
  paintDoubleArrow(ctx, a, b);
}
function ghostSquare(ctx, half) {
  const w = [[-half, 0, -half], [half, 0, -half], [half, 0, half], [-half, 0, half]];
  const s = w.map((p) => ui.proj(p[0], p[1], p[2]));
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "rgba(236, 239, 242, 0.78)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < 4; i++) {
    const a = s[i], b = s[(i + 1) % 4];
    if (!a || !b) continue;
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
  ctx.restore();
}
function make(o) {
  const st = { g: null };
  const kb = ui.slider(o.aria, () => [o.get(), o.vt(o.get())], (e) => {
    const v = o.keys(e); if (v == null) return false; o.set(v); return true;
  });
  const obj = ui.register({
    layer: "tools",
    key: o.key,
    body(x, y) { return o.key === "size" || o.key === "depth" ? obj.hit(x, y) : null; },
    hit(x, y) {
      if (o.key !== "size" && o.key !== "depth") return null;
      const g = domainEdges();
      if (o.key === "size") {
        for (let i = 0; i < g.corners.length; i++) {
          const p = g.corners[i].p;
          if (p && Math.hypot(x - p[0], y - p[1]) <= cornerPx()) return "c" + i;
        }
        return null;
      }
      for (let i = 0; i < g.verts.length; i++) {
        const v = g.verts[i];
        if (!v.top.p || !v.bot) continue;
        if (ui.segDist([x, y], v.top.p, v.bot) > shaftPx()) continue;
        const t = Math.min(1, Math.max(0, ui.segT([x, y], v.top.p, v.bot)));
        if (t < 0.22 && Math.hypot(x - v.top.p[0], y - v.top.p[1]) <= cornerPx()) continue;
        return "v" + i;
      }
      return null;
    },
    down(h, x, y) {
      ui.drag.key = o.key; st.dragging = h; st.value = o.get(); st.y = y; st.x = x;
      const g = domainEdges();
      if (o.key === "size") {
        const i = Number(String(h).slice(1));
        const c = g.corners[i];
        st.sign = c ? [c.sx, c.sz] : [1, 1];
        st.shown = o.get();
        st.target = o.get();
        st.grabSize = o.get();
        st.grabX = x; st.grabY = y;
        const half = o.get() / 2;
        const p0 = ui.proj(st.sign[0] * half, 0, st.sign[1] * half);
        const p1 = ui.proj(st.sign[0] * (half + 40), 0, st.sign[1] * (half + 40));
        let ux = 1, uy = 0, mPerPx = 0.8;
        if (p0 && p1) {
          const dx = p1[0] - p0[0], dy = p1[1] - p0[1], l = Math.hypot(dx, dy) || 1;
          ux = dx / l; uy = dy / l; mPerPx = 40 / l;
        }
        st.screen = [ux, uy];
        st.mPerPx = mPerPx;
        const hit = rayOnPlane(x, y, 0);
        st.mode = hit && hit.p && hit.slope >= 0.12 ? "plane" : "pixel";
      } else if (o.key === "depth") {
        const i = Number(String(h).slice(1));
        const v = g.verts[i];
        st.axis = null;
        if (v && v.top.p && v.bot) {
          const dx = v.bot[0] - v.top.p[0], dy = v.bot[1] - v.top.p[1];
          const span = Math.hypot(dx, dy);
          st.axis = { ux: span >= 4 ? dx / span : 0, uy: span >= 4 ? dy / span : 1, px: Math.max(span, 78) / Math.max(st.value, 2) };
        }
      }
    },
    move(h, x, y, e) {
      if (o.key === "size" && st.sign) {
        const step = e && e.shiftKey ? 1 : 10;
        st.fine = step === 1;
        const hit = rayOnPlane(x, y, 0);
        let raw;
        if (hit && hit.p && hit.slope >= 0.12) {
          const half = (st.sign[0] * hit.p[0] + st.sign[1] * hit.p[2]) / 2;
          raw = half * 2;
          st.mode = "plane";
        } else {
          const along = (x - st.grabX) * st.screen[0] + (y - st.grabY) * st.screen[1];
          const delta = Math.max(-140, Math.min(140, along * st.mPerPx * 0.35));
          raw = st.grabSize + delta;
          st.mode = "pixel";
        }
        st.target = snapSize(raw, step);
      } else if (o.key === "depth" && st.axis) {
        const along = (x - st.x) * st.axis.ux + (y - st.y) * st.axis.uy;
        const next = st.value + along / st.axis.px;
        ui.set("depth", Math.max(2, Math.min(80, Math.round(next))));
      }
      kb.upd();
    },
    up() {
      if (o.key === "size" && st.target != null) {
        ui.set("size", snapSize(st.target, st.fine ? 1 : 10));
        st.shown = st.target;
      }
      if (o.key === "size" || o.key === "depth") cancel(o.key);
      if (st.dragging) st.fade = 1;
      st.dragging = null;
    },
    cursor: (h) => {
      if (o.key === "depth" || (h && String(h)[0] === "v")) return "ns-resize";
      if (o.key === "size" && h && String(h)[0] === "c") {
        const c = domainEdges().corners[Number(String(h).slice(1))];
        if (c) return sizeCursor([c.sx, c.sz]);
      }
      return "grab";
    },
    draw() {
      if (o.key !== "size" && o.key !== "depth") { delete ui.debug[o.key]; return; }
      const g = domainEdges();
      const live = st.dragging ? "drag" : (ui.hover && ui.hover.obj === obj) ? "hover" : "idle";
      if (live === "drag") st.fade = 1;
      else if (ui.reduced) st.fade = 0;
      else if (st.fade > 0) st.fade *= Math.exp(-(ui.dt || 0.016) / 0.28);
      const cursor = ui.mouse || null;
      const ctx = ui.ctx;
      if (o.key === "size" && st.dragging && st.target != null) {
        const dt = ui.dt || 0.016;
        if (ui.reduced) st.shown = st.target;
        else st.shown += (st.target - st.shown) * (1 - Math.exp(-dt / 0.05));
        const applied = snapSize(st.shown, st.fine ? 1 : 10);
        if (applied !== num("size")) ui.set("size", applied);
      }
      if (o.key === "size") {
        const base = live === "drag" ? 0.34 : 0.16;
        for (const chain of g.tops) paintChain(ctx, chain, cursor, base, live === "drag" ? 0.78 : 0.72, 1.15);
        let idx = null;
        if (st.dragging && String(st.dragging)[0] === "c") idx = Number(String(st.dragging).slice(1));
        else if (live === "hover" && ui.hover && ui.hover.h && ui.hover.h[0] === "c") idx = Number(ui.hover.h.slice(1));
        if (idx != null && g.corners[idx]) {
          const c = g.corners[idx];
          diagArrow(ctx, [c.sx, c.sz], num("size") / 2, c.y || 0);
        }
        if (st.dragging && st.target != null) {
          ghostSquare(ctx, st.target / 2);
          if (cursor) dragValue(ctx, cursor[0], cursor[1], `${Math.round(st.target)} m`, 1);
        }
        const half = num("size") / 2;
        const cp = st.sign ? ui.proj(st.sign[0] * half, 0, st.sign[1] * half) : null;
        const err = cp && cursor ? Math.hypot(cursor[0] - cp[0], cursor[1] - cp[1]) : null;
        ui.debug.sizeDrag = { mode: st.mode || "idle", target: st.target, shown: st.shown, err, step: st.fine ? 1 : 10, live };
      } else {
        g.verts.forEach((v, i) => {
          if (!v.top.p || !v.bot) return;
          const mine = st.dragging === "v" + i || (live === "hover" && ui.hover && ui.hover.h === "v" + i);
          const mid = [(v.top.p[0] + v.bot[0]) / 2, (v.top.p[1] + v.bot[1]) / 2];
          let w = 0;
          if (cursor) { const d = Math.hypot(mid[0] - cursor[0], mid[1] - cursor[1]); w = Math.exp(-(d * d) / (2 * 64 * 64)); }
          const alpha = (mine ? 0.5 : 0.14) + (0.8 - (mine ? 0.5 : 0.14)) * w;
          ctx.save();
          ctx.lineCap = "round";
          ctx.strokeStyle = `rgba(176, 182, 188, ${alpha})`;
          ctx.lineWidth = (mine ? 1.8 : 1.1) + w;
          ctx.beginPath();
          ctx.moveTo(v.top.p[0], v.top.p[1]);
          ctx.lineTo(v.bot[0], v.bot[1]);
          ctx.stroke();
          ctx.restore();
          if (mine) {
            const my = (v.top.y + v.bed) / 2;
            const span = 7 * showM();
            paintDoubleArrow(ctx, ui.proj(v.top.x, my + span, v.top.z), ui.proj(v.top.x, my - span, v.top.z));
          }
        });
      }
      if (o.key !== "size" && st.fade > 0.04 && cursor) dragValue(ctx, cursor[0], cursor[1], o.val(), st.fade);
      ui.debug[o.key] = { v: o.get(), live, mode: o.key === "size" ? (st.mode || "edge") : "edge" };
    },
  });
  return obj;
}
make({ key: "size", label: "Size", aria: "Ocean size", get: () => num("size"), vt: (v) => `${v} metres`, val: () => `${num("size")} m`, set: (v) => ui.set("size", Math.min(1000, Math.max(100, v))), keys: (e) => /Arrow(Left|Right)/.test(e.key) ? num("size") + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 100 : 10) : null });
make({ key: "depth", label: "Depth", aria: "Seabed depth", get: () => num("depth"), vt: (v) => `${v} metres`, val: () => `${num("depth")} m`, set: (v) => ui.set("depth", Math.min(80, Math.max(2, v))), keys: (e) => /Arrow(Up|Down)/.test(e.key) ? num("depth") + (e.key === "ArrowDown" ? 1 : -1) * (e.shiftKey ? 10 : 1) : null });

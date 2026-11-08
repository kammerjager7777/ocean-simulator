// v30.23 on-sea UI core (behaviour; the look comes from theme.js). Layers v4 + migration, keys 1-7 / U /
// Space, clean view, the pause button, the overlay canvas, world<->screen geometry, the pointer router
// (on-object hits first, misses fall through to the camera) and the per-frame loop the objects hang on.
import { theme, inkForLum } from "./theme.js?v=30.29";
import { publish as publishSky } from "./sky.js?v=30.29";
import { addRipple, publishSplash } from "../../physics/ripples.js?v=30.29";
import { tick as coastTick, cancelAll } from "./coast.js?v=30.29";
export const P = new URLSearchParams(location.search);
export const $ = (s) => document.querySelector(s);
export const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
// [json key, layer name, default, key digit]. Beaufort is not a layer: the wind card always shows it.
export const LAYERS = [["wind", "Wind", true, 1], ["buoy", "Buoy", false, 2], ["section", "Cross-section", false, 3],
  ["ripple", "Ripple touch", false, 4], ["tod", "Sky & time", true, 5], ["tools", "Block handles", true, 6]];
export function migrateLayers(raw) {
  const d = Object.fromEntries(LAYERS.map(([k, , v]) => [k, v]));
  let j = null; try { j = JSON.parse(raw); } catch { return d; }
  if (!j || typeof j !== "object") return d;
  if (j.v === 1) { const v2 = { v: 2 }; for (const k of ["wind", "section", "ripple", "tod"]) if (typeof j[k] === "boolean") v2[k] = j[k]; v2.buoy = j.probe === true || j.buoy === true; v2.tools = true; j = v2; }
  if (j.v === 2) { j = { ...j, v: 3 }; }
  if (j.v === 3) { const { streaks, ...rest } = j; j = { ...rest, v: 4 }; }
  if (j.v === 4) j = { ...j, v: 5, section: false };
  if (j.v === 5 || j.v === 6) { for (const k of Object.keys(d)) if (typeof j[k] === "boolean") d[k] = j[k]; }
  return d;
}
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const UI_OFF = P.get("ui") === "0";
const layers = migrateLayers(store.get("ocean.ui.layers", ""));
const LAYER_URL = P.get("layers"); // ?layers=all|none|wind,tod,... (timing and review only; not saved)
if (LAYER_URL) { const want = LAYER_URL === "all" ? LAYERS.map((l) => l[0]) : LAYER_URL === "none" ? [] : LAYER_URL.split(","); for (const [k] of LAYERS) layers[k] = want.includes(k); }
const saveLayers = () => { if (!LAYER_URL) store.set("ocean.ui.layers", JSON.stringify({ v: 6, ...layers })); };
saveLayers();
let clean = store.get("ocean.ui.clean", "off") === "on";
const vis = {}; for (const [k] of LAYERS) vis[k] = layers[k] && !clean ? 1 : 0;
const inputs = { dir: $("#dir"), spd: $("#spd"), chop: $("#chop"), size: $("#size"), clarity: $("#clarity"), depth: $("#depth"), gust: $("#gust"), pause: $("#pause"), flags: $("#flags"), boat: $("#boat") };

// ---------- geometry ----------
function inv4(a) {
  const o = new Float64Array(16);
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4], b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12], b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06; if (!det) return null; det = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det; o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det; o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det; o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det; o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det; o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det; o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det; o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det; o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det; o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det; o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det; o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det; o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
  return o;
}
export const ui = {
  theme, layers, vis, inputs, reduced, P, store, debug: {}, V: null, VI: null, ctx: null, dpr: 1, dt: 1 / 60, now: 0, hover: null, drag: null, objs: [],
  get clean() { return clean; }, get paused() { return inputs.pause.checked; },
  // S: the size the block is DRAWN at (a Size change rebuilds the sim, so the input can lead it); D: the set depth
  S: () => (ui.V && ui.V.size) || Number(inputs.size.value), D: () => Number(inputs.depth.value),
  simT: () => { const s = globalThis.__oceanState; return s && s.fx ? s.fx.t : 0; },
  idle: () => document.body.classList.contains("ui-idle"),
  proj(x, y, z) {
    const m = ui.V.vp; const cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13], cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (cw <= 1e-6) return null; return [(cx / cw * 0.5 + 0.5) * ui.V.cw, (1 - (cy / cw * 0.5 + 0.5)) * ui.V.ch];
  },
  P3: (p) => ui.proj(p[0], p[1], p[2]),
  toPlane(sx, sy, level = 0) { // screen px -> the point on the plane y = level
    if (!ui.VI) return null; const nx = (sx / ui.V.cw) * 2 - 1, ny = 1 - (sy / ui.V.ch) * 2, m = ui.VI;
    const un = (z) => { const x = m[0] * nx + m[4] * ny + m[8] * z + m[12], y = m[1] * nx + m[5] * ny + m[9] * z + m[13], zz = m[2] * nx + m[6] * ny + m[10] * z + m[14], w = m[3] * nx + m[7] * ny + m[11] * z + m[15]; return [x / w, y / w, zz / w]; };
    const a = un(-1), b = un(1), dy = b[1] - a[1]; if (Math.abs(dy) < 1e-9) return null; const t = (level - a[1]) / dy; if (t < 0) return null;
    return [a[0] + (b[0] - a[0]) * t, level, a[2] + (b[2] - a[2]) * t];
  },
  // block corners (Michael's camera looks from +x,+z): N near, A left, R right, F back; y = 0 top or -D bottom
  corner(name, bottom = false) { const h = ui.S() / 2, y = bottom ? -ui.D() : 0; return { N: [h, y, h], A: [-h, y, h], R: [h, y, -h], F: [-h, y, -h] }[name]; },
  lerp3: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  B() { const a = ui.P3(ui.corner("N")), b = ui.P3(ui.corner("R")); return a && b ? Math.hypot(b[0] - a[0], b[1] - a[1]) : 420; },
  // parameter of screen point q projected onto the screen segment a-b
  segT(q, a, b) { const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1; return ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2; },
  segDist(q, a, b) { const t = Math.min(1, Math.max(0, ui.segT(q, a, b))); return Math.hypot(q[0] - (a[0] + (b[0] - a[0]) * t), q[1] - (a[1] + (b[1] - a[1]) * t)); },
  set(name, v) { const el = inputs[name]; const s = String(v); if (el.value !== s) { el.value = s; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); } },
  register(o) { ui.objs.push(o); return o; },
  state(o, h = null) { return ui.drag && ui.drag.obj === o && (h === null || ui.drag.h === h) ? "drag" : ui.hover && ui.hover.obj === o && (h === null || ui.hover.h === h) ? "hover" : "idle"; },
  pageRect() { const r = $("#panel").getBoundingClientRect(); return r.width ? r : null; },
  HIT: matchMedia("(pointer: coarse)").matches ? 22 : 16,
  // a hidden keyboard slider (spec: Tab reaches a role=slider per value)
  slider(label, get, keys) {
    const el = document.createElement("div"); el.className = "vh v23-kbd"; el.tabIndex = 0; el.setAttribute("role", "slider"); el.setAttribute("aria-label", label);
    el.style.pointerEvents = "none"; el.addEventListener("keydown", (e) => { if (keys(e)) { e.preventDefault(); e.stopPropagation(); } upd(); });
    const upd = () => { const [now, text] = get(); el.setAttribute("aria-valuenow", String(now)); el.setAttribute("aria-valuetext", text); };
    kbd.appendChild(el); upd(); return { el, upd };
  },
};
// v30.24: the block's screen hull (8 corners, convex), once per frame: chips over it take the light ink
ui.hull = null;
function hullOf(P) { P = P.filter(Boolean).sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (P.length < 3) return null; const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); const lo = [], up = [];
  for (const p of P) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = P.length - 1; i >= 0; i--) { const p = P[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1)); }
ui.overBlock = (x, y) => { const H = ui.hull; if (!H) return false; for (let i = 0, j = H.length - 1; i < H.length; j = i++) { const a = H[j], b = H[i]; if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) return false; } return true; };
// Linear luminance of the sea behind a CSS pixel, from the 32x18 probe view.js publishes. Null until the first read.
ui.lumAt = (x, y) => {
  const g = globalThis.__oceanLum; if (!g) return null;
  if (g.samples) {
    let best = null, bd = 1e18;
    for (const s of g.samples) { const d = (s.x - x) * (s.x - x) + (s.y - y) * (s.y - y); if (d < bd) { bd = d; best = s; } }
    return best ? best.L : null;
  }
  if (!(g.w > 0) || !g.L) return null;
  const i = Math.min(g.gw - 1, Math.max(0, Math.floor((x / g.w) * g.gw)));
  const jTop = Math.min(g.gh - 1, Math.max(0, Math.floor((y / g.h) * g.gh)));
  return g.L[(g.gh - 1 - jTop) * g.gw + i];
};
ui.inkAt = (pts) => inkForLum(pts.map((p) => ui.lumAt(p[0], p[1])));
ui.probe = { pts: null, sec: null }; const ZERO_PTS = new Float32Array(32);
const kbd = document.createElement("div"); kbd.id = "v23-kbd"; document.body.appendChild(kbd);

// ---------- overlay ----------
const sea = $("#sea");
const ov = document.createElement("canvas"); ov.id = "onsea"; ov.setAttribute("aria-hidden", "true"); sea.after(ov);
ui.ctx = ov.getContext("2d"); ui.root = document.createElement("div"); ui.root.id = "onsea-ui"; ov.after(ui.root);
if (UI_OFF) { ov.style.display = "none"; ui.root.style.display = "none"; }
// a chip (§7): DOM text positioned by transform only; set() writes only changed html, place() only changed px
const chipInv = []; document.fonts && document.fonts.addEventListener("loadingdone", () => chipInv.forEach((f) => f()));
const onWater = (x, y) => { const p = ui.toPlane(x, y), h = ui.S() / 2; return p && Math.abs(p[0]) < h && Math.abs(p[2]) < h ? p : null; }; ui.onWater = onWater;
ui.chip = (cls = "") => {
  const el = document.createElement("div"); el.className = `v23c ${cls}`; ui.root.appendChild(el); let h = null, px = null, on = false, cw = 0, ch = 0;
  return { el, set(html) { if (html !== h) { h = html; el.innerHTML = html; cw = ch = 0; } return this; },
    place(x, y) { const k = `${Math.round(x)},${Math.round(y)}`; if (k !== px) { px = k; el.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`; }
      ui._chipPts = ui._chipPts || []; ui._chipPts.push([x + (cw || 40) / 2, y + (ch || 14) / 2]); return this; },
    show(v) { if (v !== on) { on = v; el.classList.toggle("on", v); } return this; }, get w() { return cw || (cw = el.offsetWidth); }, get h() { return ch || (ch = el.offsetHeight); } };
};

ui.toast = () => {}; ui.syncHud = () => {}; // the HUD module (hud.js) replaces these
const LNAME = Object.fromEntries(LAYERS.map(([k, n]) => [k, n]));
ui.LNAME = LNAME;
function syncLayerUI() { ui.root.classList.toggle("clean", clean); ui.syncHud(); }
function setLayer(k, on) { if (clean) { clean = false; store.set("ocean.ui.clean", "off"); } layers[k] = on; saveLayers(); syncLayerUI(); ui.toast(LNAME[k], on); }
function setClean(on) { clean = on; store.set("ocean.ui.clean", on ? "on" : "off"); syncLayerUI(); ui.toast("Clean view", on); }
ui.setLayer = setLayer; ui.setClean = setClean; ui.syncLayerUI = syncLayerUI;
ui.togglePause = () => { inputs.pause.checked = !inputs.pause.checked; inputs.pause.dispatchEvent(new Event("change", { bubbles: true })); ui.syncHud(); };

// ---------- keys ----------
const DIGIT = Object.fromEntries(LAYERS.map(([k, , , d]) => [d, k]));
addEventListener("keydown", (e) => {
  const t = e.target;
  const typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && !/range|checkbox/.test(t.type)) || t.isContentEditable);
  if (typing) return;
  if (e.code === "Space") {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // a focused button / switch / slider gets Space itself (never double-fires)
    if (t && t !== document.body && (t.tagName === "BUTTON" || t.tagName === "INPUT" || t.tagName === "SELECT" || t.getAttribute("role") === "slider" || t.getAttribute("role") === "switch")) return;
    ui.togglePause(); e.preventDefault(); return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const m = /^(?:Digit|Numpad)([1-7])$/.exec(e.code);
  if (m) { const k = DIGIT[Number(m[1])]; if (k) setLayer(k, !layers[k]); e.preventDefault(); return; } // in clean view: exit clean view first, then toggle
  if (e.key === "u" || e.key === "U") { setClean(!clean); e.preventDefault(); return; }
  if (e.key === "h" || e.key === "H") { ui.hudH && ui.hudH(); e.preventDefault(); e.stopImmediatePropagation(); return; }
  if (e.key === "Escape") { if (ui.hudEsc && ui.hudEsc()) return; for (const o of ui.objs) o.escape && o.escape(); }
});

// ---------- pointer routing ----------
let down = null;
const pos = (e) => [e.offsetX, e.offsetY];
function hitAny(x, y) {
  // Drawn marks first, front to back, then proximity pads. A visible knob is not
  // stolen by a neighbour's slop, and a hidden shape is not in either pass.
  const objs = [];
  for (let i = ui.objs.length - 1; i >= 0; i--) { const o = ui.objs[i]; if ((vis[o.layer] > 0.5) && o.hit) objs.push(o); }
  for (const o of objs) { const h = o.body && o.body(x, y); if (h) return { obj: o, h }; }
  for (const o of objs) { const h = o.hit(x, y); if (h) return { obj: o, h }; }
  return null;
}
sea.addEventListener("pointerdown", (e) => {
  if (!ui.V || UI_OFF) return; if (e.pointerType === "touch" && !e.isPrimary) return;
  cancelAll();
  const [x, y] = pos(e); const hit = hitAny(x, y);
  if (!hit) { down = { kind: "click", x, y, t: performance.now() }; return; } // the camera keeps the event
  ui.drag = { ...hit, x0: x, y0: y, moved: false }; down = { kind: "obj" };
  hit.obj.down && hit.obj.down(hit.h, x, y, e);
  e.stopImmediatePropagation(); e.preventDefault(); sea.setPointerCapture(e.pointerId);
}, { capture: true });
sea.addEventListener("pointermove", (e) => {
  if (!ui.V || UI_OFF) return; const [x, y] = pos(e);
  if (down && down.kind === "obj" && ui.drag) {
    e.stopImmediatePropagation(); if (Math.hypot(x - ui.drag.x0, y - ui.drag.y0) >= 4) ui.drag.moved = true;
    if (ui.drag.moved && ui.drag.obj.move) ui.drag.obj.move(ui.drag.h, x, y, e); return;
  }
  ui.hover = hitAny(x, y); ui.mouse = [x, y];
  sea.style.cursor = ui.hover ? (ui.hover.obj.cursor ? ui.hover.obj.cursor(ui.hover.h) : "grab") : vis.ripple > 0.5 && onWater(x, y) ? "none" : "";
}, { capture: true });
const up = (e) => {
  if (!down) return; const d = down; down = null; const [x, y] = pos(e);
  if (d.kind === "obj" && ui.drag) { const g = ui.drag; ui.drag = null; if (!g.moved && g.obj.click) g.obj.click(g.h, x, y, e); g.obj.up && g.obj.up(g.h, x, y, e); e.stopImmediatePropagation(); return; }
  if (Math.hypot(x - d.x, y - d.y) >= 4) return;
  if (vis.ripple > 0.5) { const p = onWater(d.x, d.y); if (p) { addRipple(p[0], p[2], ui.simT(), 1); ui.lastRipple = p; ui.rippleFlash = ui.now; } }
};
sea.addEventListener("pointerup", up, { capture: true });
sea.addEventListener("pointerleave", () => { if (!down) ui.hover = null; });
sea.addEventListener("pointercancel", () => { down = null; ui.drag = null; }, { capture: true });
sea.addEventListener("dblclick", (e) => { const [x, y] = pos(e); const hit = hitAny(x, y); if (hit && hit.obj.dblclick) { hit.obj.dblclick(hit.h, x, y, e); e.stopImmediatePropagation(); e.preventDefault(); } }, { capture: true });
sea.addEventListener("wheel", (e) => { const [x, y] = pos(e); const hit = hitAny(x, y); if (hit && hit.obj.wheel) { hit.obj.wheel(hit.h, e); e.stopImmediatePropagation(); e.preventDefault(); } }, { capture: true, passive: false });

// ---------- frame ----------
let last = 0; ui.timing = { n: 0, sum: 0, last: 0 };
function frame(t) {
  requestAnimationFrame(frame);
  const t0 = performance.now();
  ui.dt = last ? Math.min(0.1, (t - last) / 1000) : 1 / 60; last = t; ui.now = t;
  ui.V = globalThis.__oceanView || null; ui.dpr = devicePixelRatio || 1;
  const step = reduced ? 1 : ui.dt / 0.2;
  for (const [k] of LAYERS) { const tg = layers[k] && !clean && !UI_OFF ? 1 : 0; vis[k] = vis[k] < tg ? Math.min(tg, vis[k] + step) : Math.max(tg, vis[k] - step); }
  { const m = Math.round(ui.todMin ?? 660); if (m !== ui.themeTod) { ui.themeTod = m; theme.applyTod(m); } } // only when t changes
  const ctx = ui.ctx;
  if (ui.V) { const w = Math.round(ui.V.cw * ui.dpr), h = Math.round(ui.V.ch * ui.dpr); if (ov.width !== w || ov.height !== h) { ov.width = w; ov.height = h; } ui.VI = inv4(ui.V.vp); }
  publishSky(ui.todMin ?? 660, true, ui.VI);
  coastTick(ui.dt, reduced);
  if ((vis.ripple || 0) > 0.5) publishSplash(ui.simT()); else globalThis.__oceanRippleSplash = null;
  { // Each visible HUD string takes its ink from the pixels behind that string.
    // "dark" is black type (light backdrop). The band between 0.22 and 0.38 holds the
    // previous choice so a wave crest does not flicker the glyph. No outline is added.
    const hud = document.getElementById("v23-hud");
    if (hud && ui.V) {
      const nodes = [];
      const consider = (el) => {
        if (!el || nodes.length >= 16) return;
        const st = getComputedStyle(el);
        if (st.display === "none" || st.visibility === "hidden" || Number(st.opacity) < 0.05) return;
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) return;
        nodes.push({ el, x: r.left + Math.min(28, r.width * 0.45), y: r.top + r.height * 0.5 });
      };
      for (const sel of ["#hb-set", "#hud-title", "#hb-pause", "#hud-pl", "#hud-stats", "#hud-ver"]) consider(hud.querySelector(sel));
      hud.querySelectorAll("#hud-rows .hrow, #hud-sheet .hrow").forEach((el) => {
        if (el.closest(".hidden-col, .sheet-off, .rows-left")) return;
        consider(el);
      });
      globalThis.__oceanLumWant = nodes.map((n) => [n.x, n.y]);
      // The menu sits on its own plate, so one bright pixel must not flip a whole row.
      // Ink stays the light colour written in v23.css. The probe is kept for checks.
      const applied = [];
      for (const n of nodes) {
        const L = ui.lumAt(n.x, n.y);
        n.el.classList.remove("ink-on-light", "ink-on-dark");
        applied.push({ id: n.el.id || n.el.className, mode: "plate", L: Number.isFinite(L) ? Math.round(L * 1000) / 1000 : null });
      }
      ui.debug.hudInk = applied;
    }
  }
  if (ui.V) { const P = []; for (const n of ["N", "A", "R", "F"]) { P.push(ui.P3(ui.corner(n))); P.push(ui.P3(ui.corner(n, true))); } ui.hull = hullOf(P); ui.debug.hull = ui.hull; }
  { const S = globalThis.__oceanSky; if (S) { const u = Number(inputs.spd.value), th = (Number(inputs.dir.value) * Math.PI) / 180, k = 0.0025 * (0.4 + u / 10); S.wind = [Math.cos(th) * k, Math.sin(th) * k]; } } // v30.24 cloud drift
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, ov.width, ov.height);
  for (const o of ui.objs) if (o.update) o.update(vis[o.layer] || 0);
  if (ui.doLayout) ui.doLayout();
  { // UI height probes (async readback in gpuSurface: buoy rim k16-19, the 256-sample section): only while needed
    const sf = globalThis.__surface; const need = !UI_OFF && (ui.probe.pts || ui.probe.sec);
    if (sf && sf.setUiProbe) { if (need) sf.setUiProbe(ui.probe.pts || ZERO_PTS, ui.probe.sec); else sf.setUiProbe(null, null); }
    const st = globalThis.__oceanState; const pd = st && st.probe && st.probe.data && st.probe.data.length >= 84 * 4 ? st.probe.data : null;
    ui.probeData = need ? pd : null;
  }
  if (ui.V && !UI_OFF) { ctx.setTransform(ui.dpr, 0, 0, ui.dpr, 0, 0); for (const o of ui.objs) { const a = vis[o.layer] || 0; if (a > 0.004 && o.draw) { ctx.save(); ctx.globalAlpha = a; o.draw(a); ctx.restore(); } else if (o.hide) o.hide(); } }
  const dt = performance.now() - t0; ui.timing.n++; ui.timing.sum += dt; ui.timing.last = dt;
  ui.timing.buf = ui.timing.buf || []; ui.timing.buf.push(dt); if (ui.timing.buf.length > 600) ui.timing.buf.shift();
  if (!(ui.timing.n % 30)) { const q = ui.timing.buf.slice().sort((x, y) => x - y); ui.timing.med = q[q.length >> 1]; ui.timing.p99 = q[Math.floor(0.99 * (q.length - 1))]; }
  globalThis.__oceanUI23 = { layers: { ...layers }, clean, vis: { ...vis }, paused: inputs.pause.checked, tod: ui.todMin, uiMs: dt, uiMed: ui.timing.med, uiP99: ui.timing.p99, debug: ui.debug || {} };
  globalThis.__oceanUI23.proj = (x, y, z) => ui.proj(x, y, z); // review shots only
}
syncLayerUI();
requestAnimationFrame(frame);

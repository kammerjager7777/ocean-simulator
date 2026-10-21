// v29.2 glass UI (GAD ui-style-spec v1). Drives the panel, wind dial, stats strip, fade and
// H/S keys. It only reads the sea (window.__oceanState) and drives the existing v29 inputs
// (#dir, #spd, #chop, #size, #gust, #depth, #flags, #pause) through their own events.
const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const body = document.body;
const panel = $("#panel");
const strip = $("#strip");
const panelBtn = $("#panel-toggle");
const stripBtn = $("#strip-toggle");
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ?ui=0 hides panel and strip for this load only (timing without the UI); not saved.
const uiOff = params.get("ui") === "0";
const noFade = params.get("nofade") === "1";

// ---------- panel minimise (H, "ocean.ui.panel" open|min) ----------
function setPanel(state, save = true) {
  const min = state === "min";
  panel.classList.toggle("min", min);
  panelBtn.setAttribute("aria-expanded", String(!min));
  panelBtn.setAttribute("aria-label", min ? "Expand controls" : "Minimise controls");
  if (save) store.set("ocean.ui.panel", min ? "min" : "open");
  wake();
}
panelBtn.addEventListener("click", () => setPanel(panel.classList.contains("min") ? "open" : "min"));

// ---------- stats strip (S hides, "ocean.ui.strip" shown|hidden; chevron expands) ----------
function setStripHidden(hidden, save = true) {
  body.classList.toggle("ui-hidden-strip", hidden);
  if (save) store.set("ocean.ui.strip", hidden ? "hidden" : "shown");
}
function setStripOpen(open) {
  strip.classList.toggle("open", open);
  stripBtn.setAttribute("aria-expanded", String(open));
  stripBtn.setAttribute("aria-label", open ? "Collapse stats" : "Expand stats");
}
stripBtn.addEventListener("click", (e) => { e.stopPropagation(); setStripOpen(!strip.classList.contains("open")); });
strip.addEventListener("click", (e) => { if (matchMedia("(max-width: 899px)").matches && e.target.closest(".srowline") && !e.target.closest("#strip-toggle")) setStripOpen(!strip.classList.contains("open")); });

addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && !/range|checkbox/.test(t.type)) || t.isContentEditable)) return;
  if (e.key === "h" || e.key === "H") { setPanel(panel.classList.contains("min") ? "open" : "min"); e.preventDefault(); }
  else if (e.key === "s" || e.key === "S") { setStripHidden(!body.classList.contains("ui-hidden-strip")); e.preventDefault(); }
});

// ---------- sliders: accent fill ----------
function fill(el) { const p = ((el.value - el.min) / (el.max - el.min)) * 100; el.style.setProperty("--p", p + "%"); }
for (const el of document.querySelectorAll('#panel input[type="range"]:not(.vh)')) { fill(el); el.addEventListener("input", () => fill(el)); }

// ---------- gusts segment toggle -> #gust select ----------
const gust = $("#gust");
const segs = [...document.querySelectorAll(".seg button")];
function showGust() { for (const b of segs) { const on = b.dataset.v === gust.value; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; } }
for (const b of segs) b.addEventListener("click", () => {
  if (gust.value !== b.dataset.v) { gust.value = b.dataset.v; gust.dispatchEvent(new Event("change", { bubbles: true })); }
  showGust();
});
$(".seg").addEventListener("keydown", (e) => {
  if (!/Arrow(Left|Right|Up|Down)/.test(e.key)) return;
  e.preventDefault(); const i = segs.findIndex((b) => b.dataset.v === gust.value); const n = segs[(i + 1) % 2]; n.click(); n.focus();
});
gust.addEventListener("change", showGust);
showGust();

// ---------- wind dial -> #dir range ----------
const dir = $("#dir");
const spd = $("#spd");
const dial = $("#dial");
const arrow = $("#dial-arrow");
function showDir() {
  arrow.setAttribute("transform", `rotate(${Number(dir.value)} 60 60)`);
  dial.setAttribute("aria-valuenow", dir.value);
  dial.setAttribute("aria-valuetext", `${dir.value} degrees`);
}
function showSpd() { $("#dial-spd").textContent = `${Number(spd.value).toFixed(1)} m/s`; }
function setDir(deg) {
  const v = ((Math.round(deg) % 360) + 360) % 360;
  if (String(v) === dir.value) return;
  dir.value = String(v);
  dir.dispatchEvent(new Event("input", { bubbles: true }));
}
dir.addEventListener("input", showDir);
spd.addEventListener("input", showSpd);
let dragging = false;
function dialAngle(e) {
  const r = dial.getBoundingClientRect();
  const a = (Math.atan2(e.clientX - (r.left + r.width / 2), -(e.clientY - (r.top + r.height / 2))) * 180) / Math.PI;
  const snap = e.shiftKey ? 15 : 1;
  setDir(Math.round(a / snap) * snap);
}
dial.addEventListener("pointerdown", (e) => { dragging = true; dial.classList.add("drag"); dial.setPointerCapture(e.pointerId); dialAngle(e); dial.focus({ preventScroll: true }); e.preventDefault(); });
dial.addEventListener("pointermove", (e) => { if (dragging) dialAngle(e); });
const endDrag = () => { dragging = false; dial.classList.remove("drag"); };
dial.addEventListener("pointerup", endDrag);
dial.addEventListener("pointercancel", endDrag);
dial.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 15 : 1;
  let d = 0;
  if (e.key === "ArrowRight" || e.key === "ArrowUp") d = step;
  else if (e.key === "ArrowLeft" || e.key === "ArrowDown") d = -step;
  else if (e.key === "PageUp") d = 15; else if (e.key === "PageDown") d = -15;
  if (!d) return;
  e.preventDefault(); setDir(Number(dir.value) + d);
});
// Shift+arrow on sliders: 10x step.
for (const el of document.querySelectorAll('#panel input[type="range"]:not(.vh)')) el.addEventListener("keydown", (e) => {
  if (!e.shiftKey || !/Arrow(Left|Right|Up|Down)/.test(e.key)) return;
  e.preventDefault();
  const s = Number(el.step || 1) * 10 * (/Right|Up/.test(e.key) ? 1 : -1);
  el.value = String(Math.min(Number(el.max), Math.max(Number(el.min), Number(el.value) + s)));
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
showDir(); showSpd();

// ---------- idle fade (4.0 s; out 600 ms, wake 150 ms) ----------
const IDLE_MS = 4000;
let idleTimer = 0;
let lastX = null, lastY = null;
function canFade() {
  if (noFade || dragging || panel.classList.contains("min")) return false;
  if (panel.matches(":hover") || strip.matches(":hover")) return false;
  const a = document.activeElement;
  if (a && a !== document.body && (panel.contains(a) || strip.contains(a))) return false;
  return true;
}
function fade() { if (canFade()) { panel.classList.add("faded"); strip.classList.add("faded"); } else arm(); }
function arm() { clearTimeout(idleTimer); idleTimer = setTimeout(fade, IDLE_MS); }
function wake() { panel.classList.remove("faded"); strip.classList.remove("faded"); arm(); }
addEventListener("pointermove", (e) => {
  if (lastX !== null && Math.hypot(e.clientX - lastX, e.clientY - lastY) < 4) return;
  lastX = e.clientX; lastY = e.clientY; wake();
}, { passive: true });
for (const t of ["touchstart", "keydown", "wheel", "pointerdown"]) addEventListener(t, wake, { passive: true, capture: true });
window.__ui = { fade, wake, setPanel, setStripHidden, setStripOpen };

// ---------- stats ----------
const N = 120;
const ft = new Float64Array(N);
let fi = 0, fn = 0, prevT = null;
const g = 9.81;
function beaufort(u) { const t = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8]; let b = 0; while (b < t.length && u >= t[b]) b++; return b; }
function wavelength(T, h) { // finite-depth dispersion, fixed-point on k
  const w = (2 * Math.PI) / T; let k = (w * w) / g;
  for (let i = 0; i < 30; i++) k = (w * w) / (g * Math.tanh(Math.max(1e-6, k * h)));
  return (2 * Math.PI) / k;
}
function tick(now) {
  requestAnimationFrame(tick);
  if (prevT !== null) { ft[fi] = now - prevT; fi = (fi + 1) % N; fn = Math.min(N, fn + 1); }
  prevT = now;
}
const spark = $("#spark");
const sctx = spark.getContext("2d");
const css = getComputedStyle(document.documentElement);
const ACC = css.getPropertyValue("--accent").trim() || "#5FD3C8";
const WARN = css.getPropertyValue("--warn").trim() || "#FFC56B";
function set(id, text, warn) { const el = document.getElementById(id); if (el.textContent !== text) el.textContent = text; if (warn !== undefined) el.classList.toggle("warn", !!warn); }
function update() {
  if (body.classList.contains("ui-hidden-strip") || document.hidden) return;
  if (fn > 0) {
    const a = Array.from(fn === N ? ft : ft.subarray(0, fn)).sort((x, y) => x - y);
    const med = a[Math.floor((a.length - 1) / 2)];
    const p99 = a[Math.min(a.length - 1, Math.ceil(0.99 * a.length) - 1)];
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const fps = 1000 / mean;
    set("st-fps", String(Math.round(fps)), fps < 55);
    set("st-med", med.toFixed(1), med > 8.5);
    set("st-p99", p99.toFixed(1), p99 > 16.7);
  }
  const u = Number(spd.value);
  set("st-wind", u.toFixed(1));
  set("st-sea", String(beaufort(u)));
  const st = window.__oceanState;
  if (st && st.sampleCount) set("st-foam", ((100 * (st.foamCount || 0)) / st.sampleCount).toFixed(1));
  if (!strip.classList.contains("open")) return;
  const depth = Number($("#depth").value);
  set("st-depth", String(depth));
  set("st-held", String(window.__oceanHeld || 0));
  if (st && st.bands) {
    let s2 = 0; for (const b of st.bands) s2 += b.significantHeight ** 2;
    for (const p of [st.chopPatch, st.finePatch]) if (p && p.modes) s2 += p.significantHeight ** 2;
    set("st-hs", Math.sqrt(s2).toFixed(2));
  }
  if (u > 0.1) { const Tp = (2 * Math.PI * u) / (0.877 * g); set("st-tp", Tp.toFixed(1)); set("st-lam", wavelength(Tp, depth).toFixed(0)); }
  else { set("st-tp", "--"); set("st-lam", "--"); }
  // sparkline: last 120 frames, 0-25 ms, 16.7 ms guide
  const W = spark.width, H = spark.height, top = 25;
  sctx.clearRect(0, 0, W, H);
  sctx.globalAlpha = 0.5; sctx.strokeStyle = WARN; sctx.lineWidth = 1;
  const gy = H - (16.7 / top) * H; sctx.beginPath(); sctx.moveTo(0, gy); sctx.lineTo(W, gy); sctx.stroke();
  sctx.globalAlpha = 1; sctx.strokeStyle = ACC; sctx.lineWidth = 1.5; sctx.beginPath();
  for (let i = 0; i < fn; i++) { const v = ft[(fi - fn + i + N) % N]; const x = (i / (N - 1)) * W; const y = H - Math.min(1, v / top) * (H - 1); i ? sctx.lineTo(x, y) : sctx.moveTo(x, y); }
  sctx.stroke();
}

// ---------- init ----------
if (uiOff) { body.classList.add("ui-hidden-panel", "ui-hidden-strip"); }
else {
  setPanel(store.get("ocean.ui.panel", "open") === "min" ? "min" : "open", false);
  setStripHidden(store.get("ocean.ui.strip", "shown") === "hidden", false);
  requestAnimationFrame(tick);
  setInterval(update, 250);
  arm();
}

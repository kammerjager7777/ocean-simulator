// v30.21 on-sea UI (GAD on-sea spec v1 + Michael's decisions, Oct 30): wind arrow, depth probe,
// wave cross-section, Beaufort dial, buoy, ripple touch, time of day; Layers block, keys 1-7, U.
// Drawn on one 2D overlay canvas (no WebGL texture units used) plus DOM cards; the buoy model is a
// WebGL mesh drawn by view.js (globalThis.__oceanBuoy). Nothing here touches the foam or the physics.
import { BEAUFORT_LO, beaufort } from "./ui.js?v=30.22";

const $ = (s) => document.querySelector(s);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const LAYERS = [
  ["wind", "Wind arrow", true], ["probe", "Depth probe", false], ["section", "Wave cross-section", false],
  ["beaufort", "Beaufort dial", false], ["buoy", "Buoy", false], ["ripple", "Ripple touch", false], ["tod", "Time of day", true],
];
const BF_NAMES = ["Calm", "Light air", "Light breeze", "Gentle breeze", "Moderate breeze", "Fresh breeze", "Strong breeze", "Near gale", "Gale", "Strong gale", "Storm", "Violent storm", "Hurricane"];
const BF_DESC = ["Sea like a mirror.", "Ripples like scales, no foam crests.", "Small wavelets, glassy crests that do not break.", "Large wavelets, crests begin to break, scattered whitecaps.", "Small waves growing longer, fairly frequent whitecaps.", "Moderate, longer waves, many whitecaps, some spray.", "Large waves form, white foam crests everywhere, more spray.", "Sea heaps up, foam from breaking waves blows in streaks.", "Moderately high, long waves, crests break into spindrift, well-marked foam streaks.", "High waves, dense foam streaks, crests topple and roll over, spray cuts visibility.", "Very high waves with overhanging crests, sea looks white, heavy tumbling.", "Exceptionally high waves, sea covered in long white foam patches.", "Air filled with foam and spray, sea completely white, visibility very poor."];
const C = { text: "#EAF6F4", muted: "#A9C9C6", accent: "#5FD3C8", deep: "#008A94", warn: "#FFC56B", shadow: "rgba(4,18,48,0.55)" };
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const params = new URLSearchParams(location.search);

// ---------- state + persistence ----------
function loadLayers() {
  const d = Object.fromEntries(LAYERS.map(([k, , v]) => [k, v]));
  try { const j = JSON.parse(store.get("ocean.ui.layers", "")); if (j && j.v === 1) for (const k of Object.keys(d)) if (typeof j[k] === "boolean") d[k] = j[k]; } catch { /* defaults */ }
  return d;
}
const layers = loadLayers();
// ?layers=all|none|wind,tod,... (timing and review only; not saved)
const LAYER_URL = params.get("layers");
if (LAYER_URL) { const want = LAYER_URL === "all" ? LAYERS.map((l) => l[0]) : LAYER_URL === "none" ? [] : LAYER_URL.split(","); for (const [k] of LAYERS) layers[k] = want.includes(k); }
let clean = store.get("ocean.ui.clean", "off") === "on";
const saveLayers = () => LAYER_URL ? null : store.set("ocean.ui.layers", JSON.stringify({ v: 1, ...layers }));
const vis = {}; for (const [k] of LAYERS) vis[k] = layers[k] && !clean ? 1 : 0; // drawn opacity factor (200 ms fades)
const nowMs = () => performance.now();

// ---------- geometry ----------
function inv4(m) {
  const a = m, o = new Float64Array(16);
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4], b03 = a[1] * a[6] - a[2] * a[5];
  const b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6], b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12];
  const b08 = a[8] * a[15] - a[11] * a[12], b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06; if (!det) return null; det = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det; o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det; o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det; o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det; o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det; o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det; o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det; o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det; o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det; o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det; o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det; o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det; o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
  return o;
}
let V = null, VI = null;
function proj(x, y, z) {
  const m = V.vp; const cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13], cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (cw <= 1e-6) return null;
  return [(cx / cw * 0.5 + 0.5) * V.cw, (1 - (cy / cw * 0.5 + 0.5)) * V.ch];
}
function toWater(sx, sy, level = 0) { // screen px -> point on the plane y = level (null above the horizon)
  if (!VI) return null;
  const nx = (sx / V.cw) * 2 - 1, ny = 1 - (sy / V.ch) * 2;
  const un = (z) => { const m = VI; const x = m[0] * nx + m[4] * ny + m[8] * z + m[12], y = m[1] * nx + m[5] * ny + m[9] * z + m[13], zz = m[2] * nx + m[6] * ny + m[10] * z + m[14], w = m[3] * nx + m[7] * ny + m[11] * z + m[15]; return [x / w, y / w, zz / w]; };
  const a = un(-1), b = un(1); const dy = b[1] - a[1]; if (Math.abs(dy) < 1e-9) return null;
  const t = (level - a[1]) / dy; if (t < 0) return null;
  return [a[0] + (b[0] - a[0]) * t, level, a[2] + (b[2] - a[2]) * t];
}
// pixel basis of the water plane at a point: screen px per metre along x and z
function basis(p) { const o = proj(p[0], p[1], p[2]), ax = proj(p[0] + 1, p[1], p[2]), az = proj(p[0], p[1], p[2] + 1); if (!o || !ax || !az) return null; return { o, ex: [ax[0] - o[0], ax[1] - o[1]], ez: [az[0] - o[0], az[1] - o[1]] }; }

// ---------- DOM ----------
const sea = $("#sea");
const ov = document.createElement("canvas"); ov.id = "onsea"; ov.setAttribute("aria-hidden", "true"); sea.after(ov);
const ctx = ov.getContext("2d");
const root = document.createElement("div"); root.id = "onsea-ui"; ov.after(root);
// v30.22b: ?ui=0 (timing without the UI, this load only) hides the on-sea UI too: nothing drawn, no
// buoy mesh, no UI probe readback, 11:00 light.
const UI_OFF = params.get("ui") === "0";
if (UI_OFF) { ov.style.display = "none"; root.style.display = "none"; }
const toast = document.createElement("div"); toast.className = "os-toast glass-card mono"; root.appendChild(toast);
let toastT = 0;
function showToast(t) { toast.textContent = t; toast.classList.add("on"); clearTimeout(toastT); toastT = setTimeout(() => toast.classList.remove("on"), 1200); }
const idle = () => document.body.classList.contains("ui-idle");

// Layers block (panel, after Pause)
const lb = document.createElement("div"); lb.className = "layers";
lb.innerHTML = `<div class="lh"><span class="lbl" id="layers-h">Layers</span><button type="button" class="hideall" id="hideall">HIDE ALL&nbsp;&nbsp;U</button></div>` +
  LAYERS.map(([k, name], i) => `<label class="srow lrow7" for="ly-${k}"><span class="lname">${name}</span><span class="mono key">${i + 1}</span><input id="ly-${k}" class="sw" type="checkbox" role="switch"></label>`).join("");
$("#panel-body").appendChild(lb);
const lhead = $("#layers-h"), hideBtn = $("#hideall");
function syncLayerUI() {
  for (const [k] of LAYERS) $("#ly-" + k).checked = layers[k];
  lhead.textContent = clean ? "Clean view · U" : "Layers"; lhead.classList.toggle("clean", clean);
  hideBtn.textContent = clean ? "SHOW  U" : "HIDE ALL  U";
  root.classList.toggle("clean", clean); ov.style.opacity = clean ? "0" : "1";
  for (const [k] of LAYERS) root.classList.toggle("ly-" + k, layers[k] && !clean);
}
function setLayer(k, on, why = true) {
  if (clean) { clean = false; store.set("ocean.ui.clean", "off"); }
  layers[k] = on; saveLayers(); syncLayerUI();
  if (why) showToast(`${LAYERS.find((l) => l[0] === k)[1].toUpperCase()}  ${on ? "ON" : "OFF"}`);
  if (k === "tod") applyTod();
  if (k === "beaufort") bfShow();
}
function setClean(on) { clean = on; store.set("ocean.ui.clean", on ? "on" : "off"); syncLayerUI(); applyTod(); bfShow(); showToast(on ? "CLEAN VIEW  ON" : "CLEAN VIEW  OFF"); }
for (const [k] of LAYERS) $("#ly-" + k).addEventListener("change", (e) => setLayer(k, e.target.checked));
hideBtn.addEventListener("click", () => setClean(!clean));
addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && !/range|checkbox/.test(t.type)) || t.isContentEditable)) return;
  const m = /^(?:Digit|Numpad)([1-7])$/.exec(e.code);
  if (m) { const k = LAYERS[Number(m[1]) - 1][0]; setLayer(k, clean ? true : !layers[k]); e.preventDefault(); return; }
  if (e.key === "u" || e.key === "U") { setClean(!clean); e.preventDefault(); return; }
  if (e.key === "Escape" && pins.length) { pins.splice(0).forEach((p) => p.card.remove()); }
});

// ---------- inputs shared with the panel ----------
const dirIn = $("#dir"), spdIn = $("#spd"), clarityIn = $("#clarity"), depthIn = $("#depth"), gustIn = $("#gust");
const fire = (el) => el.dispatchEvent(new Event("input", { bubbles: true }));
function setWind(u, deg) {
  if (u !== null) { const v = Math.min(25, Math.max(0, Math.round(u * 2) / 2)); if (Number(spdIn.value) !== v) { spdIn.value = String(v); fire(spdIn); } }
  if (deg !== null) { const d = ((Math.round(deg) % 360) + 360) % 360; if (Number(dirIn.value) !== d) { dirIn.value = String(d); fire(dirIn); } }
}

// ---------- 1. wind arrow ----------
const ARROW = { L0: 48, K: 6.88 };
const arrow = { anchor: null, tip: null, drag: null, hover: false, shownL: null, shownA: null, labelUntil: 0 };
const arrowL = (u) => ARROW.L0 + ARROW.K * u;
function arrowGeom() {
  const cx = V.cw / 2, cy = V.ch / 2;
  let a = toWater(cx, cy);
  let s = a ? proj(a[0], 0, a[2]) : null;
  if (!s || s[1] < 0 || s[1] > V.ch) { const y = V.ch * 0.6; a = toWater(cx, y); s = a ? [cx, y] : null; }
  if (!a) return null;
  const th = (Number(dirIn.value) * Math.PI) / 180;
  let p1 = proj(a[0] + Math.cos(th) * 5, 0, a[2] + Math.sin(th) * 5);
  if (!p1) return null;
  let ang = Math.atan2(p1[1] - s[1], p1[0] - s[0]);
  let L = arrowL(Number(spdIn.value));
  if (MOBILE.matches) { // fit the whole arrow (anchor..tip, +-14 px) into the clear band above the sheet / cross-section
    let top = 92 + 14, left = 12 + 14; const bot = clearBottom() - 14, right = V.cw - 12 - 14;
    if (vis.buoy > 0) { // the buoy card also lives in this band: go beside it (or below it if there is room)
      const br = bcard.getBoundingClientRect();
      if (br.width && Number(bcard.style.opacity || 1) > 0 && br.bottom > top && br.top < bot) {
        if (bot - (br.bottom + 22) >= 60) top = br.bottom + 22; else left = Math.max(left, br.right + 22);
      }
    }
    const ux = Math.cos(ang), uy = Math.sin(ang);
    const fit = Math.min(1, (bot - top) / Math.max(1, Math.abs(uy) * L), (right - left) / Math.max(1, Math.abs(ux) * L));
    L *= Math.max(0.35, fit);
    let x = s[0], y = s[1];
    x = Math.max(left - Math.min(0, ux * L), Math.min(right - Math.max(0, ux * L), x));
    y = Math.max(top - Math.min(0, uy * L), Math.min(bot - Math.max(0, uy * L), y));
    if (x !== s[0] || y !== s[1]) {
      const a2 = toWater(x, y);
      if (a2) { a = a2; s = [x, y]; p1 = proj(a[0] + Math.cos(th) * 5, 0, a[2] + Math.sin(th) * 5) || p1; ang = Math.atan2(p1[1] - s[1], p1[0] - s[0]); }
    }
  }
  if (gustIn && gustIn.value === "gusty") L *= 1 + 0.04 * Math.sin((nowMs() / 1000) * 2 * Math.PI / 8);
  // 200 ms ease-out when changed elsewhere; no smoothing while dragging
  if (arrow.drag || arrow.shownL === null || reduced) { arrow.shownL = L; arrow.shownA = ang; }
  else { const k = 1 - Math.exp(-frameDt / 0.07); arrow.shownL += (L - arrow.shownL) * k; let d = ang - arrow.shownA; d = Math.atan2(Math.sin(d), Math.cos(d)); arrow.shownA += d * k; }
  return { a, s, ang: arrow.shownA, L: arrow.shownL };
}
function drawArrow(g, alpha, active) {
  const { s, ang, L } = g; const tip = [s[0] + Math.cos(ang) * L, s[1] + Math.sin(ang) * L];
  arrow.anchor = s; arrow.tip = tip; arrow.ang = ang; arrow.L = L;
  ctx.save(); ctx.globalAlpha = alpha; ctx.lineCap = "round";
  const col = active ? C.accent : C.text;
  ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke();
  const ux = Math.cos(ang), uy = Math.sin(ang), nx = -uy, ny = ux;
  ctx.beginPath(); ctx.moveTo(tip[0] - ux * 12 + nx * 7, tip[1] - uy * 12 + ny * 7); ctx.lineTo(tip[0], tip[1]); ctx.lineTo(tip[0] - ux * 12 - nx * 7, tip[1] - uy * 12 - ny * 7);
  if (active) { ctx.fillStyle = C.accent; ctx.closePath(); ctx.fill(); } else ctx.stroke();
  ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s[0], s[1], 5, 0, 2 * Math.PI); ctx.stroke();
  if (active) {
    ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5;
    for (let u = 5; u < 25.01 && arrowL(u) < L - 10; u += 5) { const q = [s[0] + ux * arrowL(u), s[1] + uy * arrowL(u)]; ctx.beginPath(); ctx.moveTo(q[0] + nx * 3, q[1] + ny * 3); ctx.lineTo(q[0] - nx * 3, q[1] - ny * 3); ctx.stroke(); }
    if (arrow.drag) { ctx.fillStyle = "rgba(95,211,200,0.25)"; ctx.beginPath(); ctx.arc(tip[0], tip[1], 14, 0, 2 * Math.PI); ctx.fill(); }
    ctx.fillStyle = C.text; ctx.strokeStyle = C.deep; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(tip[0], tip[1], 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  if (active || nowMs() < arrow.labelUntil) {
    const u = Number(spdIn.value);
    ctx.globalAlpha = alpha * (active ? 1 : Math.min(1, (arrow.labelUntil - nowMs()) / 400));
    textOnWater(`${u.toFixed(1)} m/s · ${String(dirIn.value).padStart(3, "0")}° · F${beaufort(u)}`, tip[0] + 12, tip[1] - 12);
  }
  ctx.restore();
}
function textOnWater(t, x, y, font = "500 12px 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace", col = C.text) {
  ctx.font = font; ctx.textBaseline = "middle";
  ctx.shadowColor = C.shadow; ctx.shadowOffsetY = 1; ctx.shadowBlur = 2;
  // contrast over white foam: a soft dark halo (text shadow) under the glyphs
  ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(4,18,48,0.8)"; ctx.strokeText(t, x, y); // halo: >= 4.5:1 even over #FFFFFF foam
  ctx.fillStyle = col; ctx.fillText(t, x, y); ctx.shadowColor = "transparent";
}

// ---------- 2. depth probe ----------
const pins = [];
function kdOf(clarity) { const T = 1 - clarity; const aW = [0.35 + 0.03 * T, 0.06 + 0.05 * T, 0.02 + 0.12 * T]; const s = 0.0035 + 0.012 * T; const bb = [0.0005 + s * 0.75, 0.0011 + s * 1.75, 0.0024 + s * 0.9]; return aW.map((a, i) => 1.15 * a + bb[i]); }
function bedDepthAt(/* x, z */) { return Number(depthIn.value); } // flat bed today; v31 seabed plugs in here
function dropPin(p) {
  const card = document.createElement("div"); card.className = "os-card probe glass-card";
  card.innerHTML = `<div class="ct">Depth</div><div class="mono dv">— m</div><div class="bars"><i style="--c:#E5735C"></i><i style="--c:#6FD08A"></i><i style="--c:#5C9BE5"></i></div><button class="x" type="button" aria-label="Remove pin">×</button>`;
  root.appendChild(card);
  const pin = { x: p[0], z: p[2], card, born: nowMs(), awake: nowMs() + 2000 };
  card.querySelector(".x").addEventListener("click", (e) => { e.stopPropagation(); card.remove(); pins.splice(pins.indexOf(pin), 1); });
  card.addEventListener("pointerenter", () => { pin.awake = nowMs() + 2000; });
  pins.push(pin); if (pins.length > 3) pins.shift().card.remove();
}
function updatePin(pin, alpha) {
  const s = proj(pin.x, 0, pin.z);
  if (!s) { pin.card.style.display = "none"; return; }
  // min/max within a 3 m radius (straight down from the clicked point to still-water level)
  let lo = Infinity, hi = -Infinity; for (const [dx, dz] of [[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3]]) { const d = bedDepthAt(pin.x + dx, pin.z + dz); lo = Math.min(lo, d); hi = Math.max(hi, d); }
  const K = kdOf(Number(clarityIn.value)); const surv = K.map((k) => Math.exp(-2 * k * lo));
  const dv = Math.abs(hi - lo) < 0.05 ? `${lo.toFixed(1)} m` : `${lo.toFixed(1)}–${hi.toFixed(1)} m`;
  const dvEl = pin.card.querySelector(".dv"); if (dvEl.textContent !== dv) dvEl.textContent = dv;
  pin.card.querySelectorAll(".bars i").forEach((b, i) => b.style.setProperty("--f", (surv[i] * 100).toFixed(1) + "%"));
  const awake = nowMs() < pin.awake || pin.card.matches(":hover");
  const t = Math.min(1, (nowMs() - pin.born) / 220); const drop = (1 - t) * 12;
  // pin (canvas)
  ctx.save(); ctx.globalAlpha = alpha * (awake ? 1 : idle() ? 0.35 : 0.7);
  ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(s[0], s[1] - drop); ctx.lineTo(s[0], s[1] - 28 - drop); ctx.stroke();
  ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.arc(s[0], s[1] - 33 - drop, 5, 0, 2 * Math.PI); ctx.stroke();
  ctx.globalAlpha *= 0.6; ctx.fillStyle = C.text; ctx.beginPath(); ctx.arc(s[0], s[1], 3, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
  // card: 16 px up-right of the pin head, flipped to keep 24 px from the edges and off the panel
  const cw = pin.card.offsetWidth, ch = pin.card.offsetHeight; let x = s[0] + 16, y = s[1] - 33 - 16 - ch;
  const right = panelLeft() - 24; if (x + cw > right) x = s[0] - 16 - cw; if (y < 24) y = s[1] + 16;
  x = Math.max(24, Math.min(right - cw, x)); y = Math.max(24, Math.min(V.ch - 24 - ch, y));
  pin.card.style.display = ""; pin.card.style.transform = `translate(${x}px, ${y}px)`;
  pin.card.style.opacity = String(alpha * (awake ? 1 : idle() ? 0.25 : 0.7) * Math.min(1, Math.max(0, (nowMs() - pin.born - 60) / 180)));
}
const panelEl = $("#panel"), stripEl = $("#strip");
function panelLeft() { const r = panelEl.getBoundingClientRect(); return r.width && r.left > V.cw / 2 ? r.left : V.cw; }
// v30.22d (spec §1/§3 mobile placement): on phone/tablet widths the panel is a bottom sheet; the wave
// cross-section sits just above the sheet and the wind arrow is kept in the clear band between the top
// strip/time slider and the cross-section (nothing drawn under the sheet).
const MOBILE = matchMedia("(max-width: 899px)");
function sheetTop() {
  if (!MOBILE.matches) return V.ch;
  const r = panelEl.getBoundingClientRect(); const op = Number(getComputedStyle(panelEl).opacity);
  return r.width && r.height && op > 0 && r.top > V.ch * 0.2 ? r.top : V.ch;
}
function clearBottom() { // lowest y the on-sea overlays may use on mobile
  const t = sheetTop(); return t - 8 - (vis.section > 0 ? 48 + 8 : 0);
}

// ---------- 3. wave cross-section ----------
const sec = (() => { try { const j = JSON.parse(store.get("ocean.ui.section", "")); if (Array.isArray(j) && j.length === 4 && j.every(Number.isFinite)) return j; } catch { /* default */ } return [-100, 0, 100, 0]; })();
const stripBox = document.createElement("div"); stripBox.className = "os-section"; stripBox.innerHTML = `<canvas></canvas>`; root.appendChild(stripBox);
const sc = stripBox.querySelector("canvas"), sx = sc.getContext("2d");
// v30.22g perf: the strip's size is tracked by a ResizeObserver and its per-frame style writes skip unchanged values
const secRect = { width: 0, height: 0 };
if (typeof ResizeObserver !== "undefined") new ResizeObserver((es) => { for (const e of es) { secRect.width = e.contentRect.width; secRect.height = e.contentRect.height; } }).observe(stripBox);
else Object.defineProperties(secRect, { width: { get: () => stripBox.getBoundingClientRect().width }, height: { get: () => stripBox.getBoundingClientRect().height } });
const styCache = new WeakMap();
function setSty(el, k, v) { let c = styCache.get(el); if (!c) { c = {}; styCache.set(el, c); } if (c[k] !== v) { c[k] = v; el.style[k] = v; } }
const secS = { drag: null, hover: false, cursor: null, scale: 1, brkT: new Float32Array(256), data: null };
let lastSecT = -1;
function drawSection(alpha, sdat) {
  const r = secRect; const dpr = devicePixelRatio || 1; // v30.22g perf: size from a ResizeObserver (no forced layout every frame)
  if (sc.width !== Math.round(r.width * dpr) || sc.height !== Math.round(r.height * dpr)) { sc.width = Math.round(r.width * dpr); sc.height = Math.round(r.height * dpr); }
  sx.setTransform(dpr, 0, 0, dpr, 0, 0); sx.clearRect(0, 0, r.width, r.height);
  setSty(stripBox, "opacity", String(alpha * (secS.hover || secS.drag ? 1 : idle() ? 0.3 : 0.7)));
  setSty(stripBox, "bottom", MOBILE.matches ? `${Math.round(V.ch - sheetTop() + 8)}px` : "");
  const W = r.width, H = r.height, mid = H / 2;
  const bands = (globalThis.__oceanState && globalThis.__oceanState.bands) || []; const hs = Math.max(0.2, Math.sqrt(bands.reduce((a, b) => a + (b.significantHeight || 0) ** 2, 0)) || 1); const tgt = (H / 2 - 6) / (hs * 1.2);
  secS.scale += (tgt - secS.scale) * (1 - Math.exp(-frameDt / 0.1));
  const k = secS.scale;
  sx.setLineDash([2, 3]); sx.strokeStyle = "rgba(169,201,198,0.5)"; sx.lineWidth = 1; sx.beginPath(); sx.moveTo(56, mid); sx.lineTo(W - 56, mid); sx.stroke(); sx.setLineDash([]);
  const depth = Number(depthIn.value);
  if (depth * k < H / 2) { const y = mid + depth * k; sx.fillStyle = "rgba(108,163,155,0.15)"; sx.fillRect(56, y, W - 112, H - y); sx.strokeStyle = "#6CA39B"; sx.beginPath(); sx.moveTo(56, y); sx.lineTo(W - 56, y); sx.stroke(); }
  if (sdat) {
    const x0 = 56, x1 = W - 56; let mx = -Infinity, mn = Infinity;
    sx.strokeStyle = C.text; sx.lineWidth = 1.5; sx.beginPath();
    for (let i = 0; i < 256; i++) { const h = sdat[i * 4]; mx = Math.max(mx, h); mn = Math.min(mn, h); const x = x0 + (x1 - x0) * i / 255, y = mid - h * k; if (i) sx.lineTo(x, y); else sx.moveTo(x, y); }
    sx.stroke();
    // breaking: 3 px warn bar along the crest + 6 px tick; appear 80 ms, fade 600 ms after the flag clears
    for (let i = 0; i < 256; i++) { const f = sdat[i * 4 + 3] > 0; secS.brkT[i] = f ? Math.min(1, secS.brkT[i] + frameDt / 0.08) : Math.max(0, secS.brkT[i] - frameDt / 0.6); }
    sx.strokeStyle = C.warn; sx.lineWidth = 3;
    for (let i = 0; i < 256; i++) if (secS.brkT[i] > 0.01) { sx.globalAlpha = secS.brkT[i]; const x = x0 + (x1 - x0) * i / 255, y = mid - sdat[i * 4] * k; sx.beginPath(); sx.moveTo(x - 1.5, y); sx.lineTo(x + 1.5, y); sx.stroke(); sx.lineWidth = 1; sx.beginPath(); sx.moveTo(x, y - 4); sx.lineTo(x, y - 10); sx.stroke(); sx.lineWidth = 3; }
    sx.globalAlpha = 1;
    sx.font = "500 11px 'JetBrains Mono', ui-monospace, monospace"; sx.fillStyle = C.text; sx.textBaseline = "middle"; sx.lineWidth = 3; sx.lineJoin = "round"; sx.strokeStyle = "rgba(4,18,48,0.8)";
    const lab = (t, x, y) => { sx.strokeText(t, x, y); sx.fillText(t, x, y); };
    lab(`η ${mx >= 0 ? "+" : ""}${mx.toFixed(1)} m`, 2, 8); lab(`${mn < 0 ? "−" : "+"}${Math.abs(mn).toFixed(1)} m`, 2, H - 8);
    const span = Math.hypot(sec[2] - sec[0], sec[3] - sec[1]); sx.textAlign = "right"; lab(`${span.toFixed(0)} m`, W - 2, mid); sx.textAlign = "left";
    if (secS.cursor !== null) {
      const i = Math.round((Math.min(x1, Math.max(x0, secS.cursor)) - x0) / (x1 - x0) * 255); const x = x0 + (x1 - x0) * i / 255; const h = sdat[i * 4];
      sx.strokeStyle = C.text; sx.lineWidth = 1; sx.beginPath(); sx.moveTo(x, 0); sx.lineTo(x, H); sx.stroke();
      lab(`η ${h >= 0 ? "+" : "−"}${Math.abs(h).toFixed(2)} m`, Math.min(x + 6, W - 120), 8);
      secS.cursorI = i; globalThis.__oceanSecCursor = { i, h, x: sec[0] + (sec[2] - sec[0]) * i / 255, z: sec[1] + (sec[3] - sec[1]) * i / 255 };
    }
  }
}
function drawSectionLine(alpha) {
  const a = proj(sec[0], 0, sec[1]), b = proj(sec[2], 0, sec[3]); if (!a || !b) return;
  const act = secS.hover || secS.drag;
  ctx.save(); ctx.globalAlpha = alpha * (act ? 1 : idle() ? 0.3 : 0.7);
  ctx.strokeStyle = act ? "rgba(95,211,200,0.9)" : "rgba(234,246,244,0.35)"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = act ? C.accent : C.text; ctx.lineWidth = 2; for (const p of [a, b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI); ctx.stroke(); }
  if (secS.cursorI !== undefined && secS.cursor !== null) { const t = secS.cursorI / 255; const p = proj(sec[0] + (sec[2] - sec[0]) * t, 0, sec[1] + (sec[3] - sec[1]) * t); if (p) { ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); ctx.fill(); } }
  ctx.restore(); secS.a = a; secS.b = b;
}
stripBox.addEventListener("pointermove", (e) => { const r = stripBox.getBoundingClientRect(); secS.cursor = e.clientX - r.left; secS.hover = true; });
stripBox.addEventListener("pointerleave", () => { secS.cursor = null; secS.hover = false; });

// ---------- 4. Beaufort dial (ring around the panel wind dial) ----------
const dialEl = $("#dial");
const bf = document.createElement("div"); bf.className = "bf"; bf.tabIndex = 0; bf.setAttribute("role", "slider"); bf.setAttribute("aria-label", "Beaufort force"); bf.setAttribute("aria-valuemin", "0"); bf.setAttribute("aria-valuemax", "10");
const BF_COL = (f) => (f <= 3 ? "rgba(95,211,200,A)" : f <= 6 ? "rgba(169,201,198,A)" : f <= 9 ? "rgba(255,197,107,A)" : "rgba(255,138,107,A)");
const BF_A = (f) => (f <= 3 ? 0.5 : 0.6);
const DEG_PER = 300 / 32.7, START = 210; // 300° sweep, gap at the bottom; angles clockwise from north
function arcPath(r, a0, a1) { const p = (a) => { const t = ((a - 90) * Math.PI) / 180; return [58 + r * Math.cos(t), 58 + r * Math.sin(t)]; }; const [x0, y0] = p(a0), [x1, y1] = p(a1); return `M${x0.toFixed(2)},${y0.toFixed(2)} A${r},${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)},${y1.toFixed(2)}`; }
const bfSeg = (f) => { const lo = BEAUFORT_LO[f], hi = f < 12 ? BEAUFORT_LO[f + 1] : 32.7 + 10 / DEG_PER; return [START + lo * DEG_PER, START + Math.min(hi, 32.7) * DEG_PER + (f === 12 ? 10 : 0)]; };
bf.innerHTML = `<svg viewBox="0 0 116 116" width="116" height="116"><defs><pattern id="bfh" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" stroke="#A9C9C6" stroke-width="1.5"/></pattern></defs>` +
  BF_LO_SEGS() + `</svg><div class="bfr"><div class="bfn"></div><div class="bfd"></div></div><div class="bft glass-card"></div>`;
function BF_LO_SEGS() { let s = ""; for (let f = 0; f <= 12; f++) { const [a0, a1] = bfSeg(f); const over = BEAUFORT_LO[f] > 25; s += `<path data-f="${f}" class="seg${over ? " over" : ""}" d="${arcPath(54, a0 + 0.6, a1 - 0.6)}" stroke="${over ? "url(#bfh)" : BF_COL(f).replace("A", BF_A(f))}" stroke-opacity="${over ? 0.25 : 1}"/>`; } return s; }
dialEl.parentElement.insertBefore(bf, dialEl); bf.appendChild(dialEl); // the 84 px dial sits inside the ring
const bfTip = bf.querySelector(".bft");
function bfShow() { const on = layers.beaufort && !clean; document.body.classList.toggle("bf-on", on); }
function bfUpdate() {
  const f = beaufort(Number(spdIn.value));
  if (bf.dataset.f !== String(f)) {
    bf.dataset.f = String(f); bf.setAttribute("aria-valuenow", String(f)); bf.setAttribute("aria-valuetext", `Force ${f}, ${BF_NAMES[f]}`);
    bf.querySelectorAll("path.seg").forEach((p) => { const cur = Number(p.dataset.f) === f; p.classList.toggle("cur", cur); if (!p.classList.contains("over")) p.setAttribute("stroke", BF_COL(Number(p.dataset.f)).replace("A", cur ? 1 : BF_A(Number(p.dataset.f)))); });
    bf.querySelector(".bfn").textContent = `F${f} ${BF_NAMES[f].toUpperCase()}`; bf.querySelector(".bfd").textContent = BF_DESC[f];
  }
}
const bfMid = (f) => Math.min(25, f < 12 ? (BEAUFORT_LO[f] + (f < 12 ? BEAUFORT_LO[f + 1] - 0.1 : 32.7)) / 2 : 25);
let bfHoverT = 0;
bf.querySelectorAll("path.seg").forEach((p) => {
  const f = Number(p.dataset.f);
  p.addEventListener("click", (e) => { e.stopPropagation(); if (BEAUFORT_LO[f] <= 25) setWind(f === 10 ? 25 : bfMid(f), null); });
  p.addEventListener("pointerenter", () => { clearTimeout(bfHoverT); bfHoverT = setTimeout(() => { const hi = f < 12 ? `${(BEAUFORT_LO[f + 1] - 0.1).toFixed(1)}` : "+"; bfTip.innerHTML = `<b>F${f} ${BF_NAMES[f].toUpperCase()}</b> <span class="mono">${BEAUFORT_LO[f].toFixed(1)}–${hi} m/s</span><br>${BF_DESC[f]}`; bfTip.classList.add("on"); }, 250); });
  p.addEventListener("pointerleave", () => { clearTimeout(bfHoverT); bfTip.classList.remove("on"); });
});
bf.addEventListener("keydown", (e) => { const f = beaufort(Number(spdIn.value)); if (e.key === "PageUp") { setWind(bfMid(Math.min(10, f + 1)), null); e.preventDefault(); } else if (e.key === "PageDown") { setWind(bfMid(Math.max(0, f - 1)), null); e.preventDefault(); } });

// ---------- 5. buoy (NDBC-style 3 m discus, fixed spot) ----------
const BUOY = (() => { try { const j = JSON.parse(store.get("ocean.ui.buoy", "")); if (Array.isArray(j) && j.length === 2 && j.every(Number.isFinite)) return j; } catch { /* default */ } return [38, -26]; })();
const buoy = { y: 0, pitch: 0, roll: 0, ring: new Float32Array(7200), rt: new Float64Array(7200), n: 0, i: 0, lastT: -1, Hs: NaN, Tz: NaN, Tp: NaN, shown: { Hs: NaN, Tz: NaN, Tp: NaN }, nextStat: 0, pinned: false, scale: 1 };
const bcard = document.createElement("div"); bcard.className = "os-card buoy glass-card";
bcard.innerHTML = `<div class="ct">Buoy 01 <span class="warm">warming up</span></div><div class="mono rows"><div><span>Hs</span><b class="hs">—</b></div><div><span>Tz</span><b class="tz">—</b></div><div><span>Tp</span><b class="tp">—</b></div></div><canvas class="spark" width="152" height="24"></canvas><div class="barrow"></div>`;
root.appendChild(bcard);
bcard.addEventListener("click", () => { buoy.pinned = !buoy.pinned; });
const bsp = bcard.querySelector(".spark").getContext("2d");
function buoyStats() {
  const n = buoy.n; if (n < 32) return;
  const idx = (k) => (buoy.i - n + k + 7200) % 7200; // oldest -> newest
  const t0 = buoy.rt[idx(0)], t1 = buoy.rt[idx(n - 1)]; if (t1 - t0 < 30) { buoy.Hs = NaN; return; }
  let m = 0; for (let k = 0; k < n; k++) m += buoy.ring[idx(k)]; m /= n;
  let v = 0; for (let k = 0; k < n; k++) { const d = buoy.ring[idx(k)] - m; v += d * d; } v /= n;
  buoy.Hs = 4 * Math.sqrt(v);
  let up = 0, first = -1, last = -1; for (let k = 1; k < n; k++) { const a = buoy.ring[idx(k - 1)] - m, b = buoy.ring[idx(k)] - m; if (a < 0 && b >= 0) { const t = buoy.rt[idx(k)]; if (first < 0) first = t; last = t; up++; } }
  buoy.Tz = up > 1 ? (last - first) / (up - 1) : NaN;
  // Tp: periodogram peak on a uniform 4 Hz resample (DFT over 0.03-0.5 Hz)
  const fs = 4, M = Math.min(480, Math.floor((t1 - t0) * fs)); const xs = new Float32Array(M); let k = 0;
  for (let j = 0; j < M; j++) { const t = t1 - (M - 1 - j) / fs; while (k < n - 1 && buoy.rt[idx(k + 1)] < t) k++; xs[j] = buoy.ring[idx(k)] - m; }
  let best = 0, bf0 = 0; for (let f = 0.03; f <= 0.5; f += 0.0025) { let re = 0, im = 0; const w = 2 * Math.PI * f / fs; for (let j = 0; j < M; j++) { re += xs[j] * Math.cos(w * j); im += xs[j] * Math.sin(w * j); } const p = re * re + im * im; if (p > best) { best = p; bf0 = f; } }
  buoy.Tp = bf0 ? 1 / bf0 : NaN;
  globalThis.__oceanBuoyStats = { Hs: buoy.Hs, Tz: buoy.Tz, Tp: buoy.Tp, n, span: t1 - t0 };
}
function buoyMatrix() {
  const s = buoy.scale, cp = Math.cos(buoy.pitch), sp = Math.sin(buoy.pitch), cr = Math.cos(buoy.roll), sr = Math.sin(buoy.roll);
  // R = Rz(roll about x... ) simple: tilt about z by pitch (x slope), about x by roll (z slope)
  const r = [cp, sp, 0, -sp * cr, cp * cr, sr, sp * sr, -cp * sr, cr]; // columns
  return new Float32Array([r[0] * s, r[1] * s, r[2] * s, 0, r[3] * s, r[4] * s, r[5] * s, 0, r[6] * s, r[7] * s, r[8] * s, 0, BUOY[0], buoy.y, BUOY[1], 1]);
}
const buoyObj = { matrix: buoyMatrix };
function drawBuoy(alpha, pd, simT) {
  if (pd && Number.isFinite(pd[16 * 4])) {
    const h = pd[16 * 4], hx = pd[17 * 4], hz = pd[18 * 4];
    buoy.y = h; buoy.pitch = Math.atan((hx - h) / 1.5); buoy.roll = Math.atan((hz - h) / 1.5);
    if (simT > buoy.lastT + 1e-6) { buoy.lastT = simT; buoy.ring[buoy.i] = h; buoy.rt[buoy.i] = simT; buoy.i = (buoy.i + 1) % 7200; buoy.n = Math.min(7200, buoy.n + 1); while (buoy.n > 2 && simT - buoy.rt[(buoy.i - buoy.n + 7200) % 7200] > 120) buoy.n--; }
  }
  // exaggerated for readability: at least 28 px tall on screen (mast 4.8 m)
  const base = proj(BUOY[0], buoy.y, BUOY[1]), top = proj(BUOY[0], buoy.y + 4.8, BUOY[1]);
  buoy.scale = base && top ? Math.max(1, Math.min(12, 28 / Math.max(1e-3, Math.hypot(top[0] - base[0], top[1] - base[1])))) : 1;
  globalThis.__oceanBuoy = alpha > 0.01 ? buoyObj : null;
  if (nowMs() > buoy.nextStat) { buoy.nextStat = nowMs() + 2000; buoyStats(); }
  const s = base;
  // lamp: yellow, 1 flash every 4 s (0.5 s), glow grows at dusk / night
  if (s) {
    const lamp = proj(BUOY[0], buoy.y + 4.4 * buoy.scale, BUOY[1]);
    const ph = ((simT || nowMs() / 1000) % 4) / 4; const on = ph < 0.125 ? Math.sin(ph / 0.125 * Math.PI) : 0;
    const dark = 1 - Math.min(1, Math.max(0, (sunElev - 2) / 18));
    if (lamp && on > 0) { const r = 6 + 22 * dark; const g = ctx.createRadialGradient(lamp[0], lamp[1], 0, lamp[0], lamp[1], r); g.addColorStop(0, `rgba(255,236,150,${(0.55 + 0.45 * dark) * on * alpha})`); g.addColorStop(1, "rgba(255,200,60,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(lamp[0], lamp[1], r, 0, 2 * Math.PI); ctx.fill(); }
  }
  // card
  const act = buoy.pinned || bcard.matches(":hover") || buoyHover;
  bcard.classList.toggle("act", act);
  const warm = !Number.isFinite(buoy.Hs); bcard.querySelector(".warm").style.display = warm ? "" : "none";
  for (const [k, cls, u] of [["Hs", ".hs", " m"], ["Tz", ".tz", " s"], ["Tp", ".tp", " s"]]) {
    const v = buoy[k]; if (Number.isFinite(v)) { const sv = buoy.shown[k]; buoy.shown[k] = Number.isFinite(sv) ? sv + (v - sv) * Math.min(1, frameDt / 0.3) : v; }
    const t = warm || !Number.isFinite(buoy.shown[k]) ? "—" : buoy.shown[k].toFixed(k === "Hs" ? 2 : 1) + u; const el = bcard.querySelector(cls); if (el.textContent !== t) el.textContent = t;
  }
  if (act) { // 30 s heave sparkline
    bsp.clearRect(0, 0, 152, 24); bsp.strokeStyle = "rgba(169,201,198,0.4)"; bsp.lineWidth = 1; bsp.beginPath(); bsp.moveTo(0, 12); bsp.lineTo(152, 12); bsp.stroke();
    const n = buoy.n, tEnd = buoy.lastT, sc = 10 / Math.max(0.3, (buoy.Hs || 1) / 2);
    bsp.strokeStyle = C.accent; bsp.lineWidth = 1.5; bsp.beginPath(); let st = false;
    for (let k = 0; k < n; k++) { const j = (buoy.i - n + k + 7200) % 7200; const dt = tEnd - buoy.rt[j]; if (dt > 30) continue; const x = 152 * (1 - dt / 30), y = 12 - buoy.ring[j] * sc; if (st) bsp.lineTo(x, y); else { bsp.moveTo(x, y); st = true; } }
    bsp.stroke();
  }
  const cw = 176, ch = bcard.offsetHeight || 104; let x, y; const arrowEl = bcard.querySelector(".barrow");
  const topMin = matchMedia("(max-width: 899px)").matches ? 92 : 24; // mobile: below the top strip and the time slider
  const onScreen = s && s[0] > 0 && s[0] < V.cw && s[1] > 0 && s[1] < V.ch;
  if (onScreen) { x = Math.max(24, Math.min(panelLeft() - 24 - cw, s[0] + 20)); y = Math.max(topMin, Math.min(Math.min(V.ch - 24 - ch - 76, clearBottom() - ch), s[1] - 40 - ch)); arrowEl.style.display = "none"; }
  else { x = 24; y = topMin; arrowEl.style.display = ""; const tgt = s || [V.cw / 2, V.ch]; arrowEl.style.transform = `rotate(${Math.atan2(tgt[1] - (y + ch / 2), tgt[0] - (x + cw / 2))}rad)`; }
  bcard.style.transform = `translate(${x}px, ${y}px)`; bcard.style.opacity = String(alpha * (act ? 1 : idle() ? 0.3 : 0.7));
  buoy.screen = s;
}
let buoyHover = false;

// ---------- 6. ripples ----------
const ripples = [];
function addRipple(p) { ripples.push({ x: p[0], z: p[2], t0: nowMs() }); if (ripples.length > 6) ripples.shift(); }
function drawRipples(alpha) {
  const now = nowMs(); const dur = reduced ? 300 : 900;
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i]; const B = basis([r.x, 0, r.z]); if (!B) continue;
    const echoes = reduced ? [[0, 1]] : [[0, 1], [140, 0.6], [280, 0.35]]; let alive = false;
    for (const [d, a] of echoes) {
      const t = (now - r.t0 - d) / dur; if (t < 0 || t > 1) { if (t < 0) alive = true; continue; } alive = true;
      const e = cubic(t); const rad = 6 + 90 * e; const ex = Math.hypot(...B.ex), ez = Math.hypot(...B.ez); const m = Math.max(ex, ez);
      ctx.save(); ctx.globalAlpha = alpha * a * 0.55 * (1 - t); ctx.strokeStyle = "rgb(206,236,232)"; ctx.lineWidth = 2 - t;
      ctx.setTransform(B.ex[0] / m * dpr, B.ex[1] / m * dpr, B.ez[0] / m * dpr, B.ez[1] / m * dpr, B.o[0] * dpr, B.o[1] * dpr);
      ctx.beginPath(); ctx.arc(0, 0, rad, 0, 2 * Math.PI); ctx.restore(); ctx.save(); ctx.globalAlpha = alpha * a * 0.55 * (1 - t); ctx.strokeStyle = "rgb(206,236,232)"; ctx.lineWidth = 2 - t; ctx.stroke(); ctx.restore();
    }
    if (!alive) ripples.splice(i, 1);
  }
}
function cubic(t) { // cubic-bezier(0.2, 0.7, 0.3, 1) by bisection on x
  let lo = 0, hi = 1, u = t; for (let k = 0; k < 20; k++) { u = (lo + hi) / 2; const x = 3 * (1 - u) * (1 - u) * u * 0.2 + 3 * (1 - u) * u * u * 0.3 + u * u * u; if (x < t) lo = u; else hi = u; }
  return 3 * (1 - u) * (1 - u) * u * 0.7 + 3 * (1 - u) * u * u + u * u * u;
}

// ---------- 7. time of day ----------
const tod = document.createElement("div"); tod.className = "os-tod";
tod.innerHTML = `<div class="track" role="slider" tabindex="0" aria-label="Time of day" aria-valuemin="300" aria-valuemax="1260"><div class="fill"></div><div class="thumb"></div></div><div class="tl mono"></div>`;
root.appendChild(tod);
const todTrack = tod.querySelector(".track"), todThumb = tod.querySelector(".thumb"), todLabel = tod.querySelector(".tl");
let todMin = (() => { const v = Number(store.get("ocean.ui.tod", "660")); return Number.isFinite(v) && v >= 300 && v <= 1260 ? v : 660; })();
if (params.has("tod")) todMin = Math.min(1260, Math.max(300, Number(params.get("tod")) || 660));
let todShown = todMin, todDrag = false, todLabelUntil = 0, sunElev = 0;
const LAT = (20 * Math.PI) / 180;
const elevAt = (min) => Math.asin(Math.cos(LAT) * Math.cos(((min / 60 - 12) * 15 * Math.PI) / 180)) - 0; // equinox, declination 0 (radians)
const azAt = (min) => { const h = ((min / 60 - 12) * 15 * Math.PI) / 180, e = elevAt(min); return Math.atan2(Math.sin(h), Math.cos(h) * Math.sin(LAT)) + 0 * e; };
// v30.22b: colour multiplier for the sea (reflected sky + light into the water) and the backdrop sky.
// Identity (null) for any sun at or above the 11:00 elevation, so 11:00 is exactly the approved look;
// lower sun: dimmer, warm near the horizon (dusk/dawn), dark blue at night.
function dayTint(m) {
  const e = (elevAt(m) * 180) / Math.PI, e0 = (elevAt(660) * 180) / Math.PI;
  if (Math.abs(m - 660) < 0.5 || e >= e0) return null;
  const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const b = 0.1 + 0.9 * (0.35 * sm(-8, 6, e) + 0.65 * sm(2, e0, e)); // brightness: 1 at 11:00, ~0.42 at sunset, 0.1 at night
  const w = Math.exp(-(((e - 3) / 9) ** 2)) * sm(-7, -1, e); // warmth peaks a few degrees above the horizon
  const n = 1 - sm(-7, 0, e); // night blue
  return [b * (1 + 0.45 * w) * (1 - 0.35 * n), b * (1 + 0.02 * w) * (1 - 0.1 * n), b * (1 - 0.35 * w) * (1 + 0.45 * n)];
}
function applyTod() {
  const on = layers.tod && !clean && !UI_OFF; const m = on ? todShown : 660; // hidden layer keeps 11:00 (the approved look)
  sunElev = (elevAt(m) * 180) / Math.PI;
  globalThis.__oceanDay = dayTint(m); // v30.22b: sky, reflected sky and light into the water (null = 11:00 look)
  if (Math.abs(m - 660) < 0.5) { globalThis.__oceanSun = null; return; }
  const V0 = globalThis.__oceanView && globalThis.__oceanView.sun; if (!V0) return;
  // keep the approved 11:00 sun exactly; scale its elevation by elev(t)/elev(11:00) and turn its azimuth by az(t)-az(11:00)
  const l = Math.hypot(V0[0], V0[1], V0[2]); const e0 = Math.asin(V0[1] / l), a0 = Math.atan2(V0[2], V0[0]);
  const e = Math.max(-0.05, e0 * elevAt(m) / elevAt(660)); const a = a0 + (azAt(m) - azAt(660));
  globalThis.__oceanSun = [Math.cos(e) * Math.cos(a) * l, Math.sin(e) * l, Math.cos(e) * Math.sin(a) * l];
}
const fmtT = (m) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;
function setTod(m, tween = false) {
  todMin = Math.min(1260, Math.max(300, Math.round(m / 5) * 5)); store.set("ocean.ui.tod", String(todMin));
  if (!tween || reduced) todShown = todMin; todLabelUntil = nowMs() + 1000;
  todTrack.setAttribute("aria-valuenow", String(todMin)); todTrack.setAttribute("aria-valuetext", `${fmtT(todMin)}, sun elevation ${Math.round((elevAt(todMin) * 180) / Math.PI)} degrees`);
}
const todFromY = (y) => { const r = todTrack.getBoundingClientRect(); const horiz = r.width > r.height; const f = horiz ? (y.x - r.left) / r.width : 1 - (y.y - r.top) / r.height; return 300 + Math.min(1, Math.max(0, f)) * 960; };
todTrack.addEventListener("pointerdown", (e) => { todDrag = true; todTrack.setPointerCapture(e.pointerId); setTod(todFromY({ x: e.clientX, y: e.clientY })); e.preventDefault(); e.stopPropagation(); });
todTrack.addEventListener("pointermove", (e) => { if (todDrag) setTod(todFromY({ x: e.clientX, y: e.clientY })); });
todTrack.addEventListener("pointerup", () => { todDrag = false; todLabelUntil = nowMs() + 1000; });
todTrack.addEventListener("dblclick", () => setTod(660, true));
todTrack.addEventListener("wheel", (e) => { e.preventDefault(); e.stopPropagation(); setTod(todMin + (e.deltaY < 0 ? 15 : -15), true); }, { passive: false });
todTrack.addEventListener("keydown", (e) => { const s = e.shiftKey ? 60 : 5; if (e.key === "ArrowUp" || e.key === "ArrowRight") { setTod(todMin + s, true); e.preventDefault(); } else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { setTod(todMin - s, true); e.preventDefault(); } });
function drawTod(alpha) {
  if (todShown !== todMin) { const k = 1 - Math.exp(-frameDt / 0.08); todShown += (todMin - todShown) * k; if (Math.abs(todShown - todMin) < 0.5) todShown = todMin; }
  applyTod();
  const f = (todShown - 300) / 960; tod.style.setProperty("--f", (f * 100).toFixed(2) + "%");
  const act = todDrag || tod.matches(":hover") || document.activeElement === todTrack;
  tod.style.opacity = String(alpha * (todDrag ? 1 : act ? 0.9 : idle() ? 0.3 : 0.7));
  const showL = todDrag || nowMs() < todLabelUntil || act;
  todLabel.textContent = `${fmtT(todShown)} · SUN ${Math.round(sunElev)}°`; todLabel.style.opacity = showL ? "1" : "0";
}

// ---------- pointer routing (on-sea hit test first, misses fall through to the camera) ----------
let down = null; const HIT = matchMedia("(pointer: coarse)").matches ? 22 : 16;
function hitArrow(x, y) {
  if (!(vis.wind > 0.5) || !arrow.tip) return null;
  if (Math.hypot(x - arrow.tip[0], y - arrow.tip[1]) <= HIT) return "tip";
  const ux = Math.cos(arrow.ang), uy = Math.sin(arrow.ang); const dx = x - arrow.anchor[0], dy = y - arrow.anchor[1]; const t = dx * ux + dy * uy, n = Math.abs(-dx * uy + dy * ux);
  return t > 0 && t < arrow.L && n < HIT * 0.6 ? "shaft" : null;
}
function hitSection(x, y) {
  if (!(vis.section > 0.5) || !secS.a) return null;
  if (Math.hypot(x - secS.a[0], y - secS.a[1]) <= HIT) return "a"; if (Math.hypot(x - secS.b[0], y - secS.b[1]) <= HIT) return "b";
  const ax = secS.b[0] - secS.a[0], ay = secS.b[1] - secS.a[1], l = Math.hypot(ax, ay) || 1; const t = ((x - secS.a[0]) * ax + (y - secS.a[1]) * ay) / (l * l); const n = Math.abs((x - secS.a[0]) * ay - (y - secS.a[1]) * ax) / l;
  return t > 0.1 && t < 0.9 && n < 8 ? "mid" : null;
}
sea.addEventListener("pointerdown", (e) => {
  if (!V) return; if (e.pointerType === "touch" && !e.isPrimary) return;
  const x = e.offsetX, y = e.offsetY; const ha = hitArrow(x, y), hs = hitSection(x, y);
  if (ha) { arrow.drag = ha; down = { kind: "arrow", id: e.pointerId }; }
  else if (hs) { secS.drag = hs; down = { kind: "sec", id: e.pointerId, p: toWater(x, y), s0: sec.slice() }; }
  else { down = { kind: "click", x, y, id: e.pointerId, t: nowMs() }; return; } // camera keeps the event
  e.stopImmediatePropagation(); e.preventDefault(); sea.setPointerCapture(e.pointerId);
}, { capture: true });
sea.addEventListener("pointermove", (e) => {
  if (!V) return; const x = e.offsetX, y = e.offsetY;
  if (down && down.kind === "arrow") {
    e.stopImmediatePropagation();
    const s = arrow.anchor; const p = toWater(x, y), a = toWater(s[0], s[1]);
    let deg = Number(dirIn.value);
    if (p && a) { deg = (Math.atan2(p[2] - a[2], p[0] - a[0]) * 180) / Math.PI; const sn = e.shiftKey ? 15 : 1; deg = Math.round(deg / sn) * sn; }
    const L = Math.hypot(x - s[0], y - s[1]);
    setWind(arrow.drag === "tip" ? (L - ARROW.L0) / ARROW.K : null, deg);
    return;
  }
  if (down && down.kind === "sec") {
    e.stopImmediatePropagation(); const p = toWater(x, y); if (!p) return;
    if (secS.drag === "a") { sec[0] = p[0]; sec[1] = p[2]; } else if (secS.drag === "b") { sec[2] = p[0]; sec[3] = p[2]; }
    else if (down.p) { const dx = p[0] - down.p[0], dz = p[2] - down.p[2]; sec[0] = down.s0[0] + dx; sec[1] = down.s0[1] + dz; sec[2] = down.s0[2] + dx; sec[3] = down.s0[3] + dz; }
    return;
  }
  arrow.hover = !!hitArrow(x, y); secS.hoverLine = !!hitSection(x, y);
  buoyHover = vis.buoy > 0.5 && buoy.screen && Math.hypot(x - buoy.screen[0], y - buoy.screen[1] + 14) < 24;
  sea.style.cursor = arrow.hover || secS.hoverLine ? "grab" : "";
}, { capture: true });
const up = (e) => {
  if (!down) return; const d = down; down = null;
  if (d.kind === "arrow") { arrow.drag = null; arrow.labelUntil = nowMs() + 1400; e.stopImmediatePropagation(); return; }
  if (d.kind === "sec") { secS.drag = null; store.set("ocean.ui.section", JSON.stringify(sec.map((v) => Math.round(v * 10) / 10))); e.stopImmediatePropagation(); return; }
  if (Math.hypot(e.offsetX - d.x, e.offsetY - d.y) >= 4) return;
  if (e.pointerType === "touch" && vis.probe > 0.5 && nowMs() - d.t < 350 && !(vis.ripple > 0.5)) return; // touch: long-press drops a pin
  const p = toWater(d.x, d.y); if (!p) return;
  if (vis.probe > 0.5 && (e.pointerType !== "touch" || nowMs() - d.t >= 350)) dropPin(p);
  if (vis.ripple > 0.5) addRipple(p);
};
sea.addEventListener("pointerup", up, { capture: true });
sea.addEventListener("pointercancel", () => { down = null; arrow.drag = null; secS.drag = null; }, { capture: true });
// wind arrow keyboard (Tab to the canvas while the arrow is on)
sea.tabIndex = 0; sea.setAttribute("role", "slider"); sea.setAttribute("aria-label", "Wind direction and speed on the sea");
sea.addEventListener("keydown", (e) => {
  if (!(vis.wind > 0.5)) return; const sh = e.shiftKey;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") { setWind(null, Number(dirIn.value) + (e.key === "ArrowRight" ? 1 : -1) * (sh ? 15 : 1)); arrow.labelUntil = nowMs() + 1400; e.preventDefault(); }
  else if (e.key === "ArrowUp" || e.key === "ArrowDown") { setWind(Number(spdIn.value) + (e.key === "ArrowUp" ? 1 : -1) * (sh ? 5 : 0.5), null); arrow.labelUntil = nowMs() + 1400; e.preventDefault(); }
});

// ---------- frame ----------
let dpr = 1, lastFrame = 0, frameDt = 1 / 60; const secUnpack = new Float32Array(256 * 4); const uiProbePts = new Float32Array(32);
function frame(t) {
  frameDt = lastFrame ? Math.min(0.1, (t - lastFrame) / 1000) : 1 / 60; lastFrame = t;
  V = globalThis.__oceanView || null;
  const target = {}; for (const [k] of LAYERS) target[k] = layers[k] && !clean && !UI_OFF ? 1 : 0;
  if (UI_OFF) for (const k in vis) vis[k] = 0;
  const step = reduced ? 1 : frameDt / 0.2; for (const k in target) vis[k] = vis[k] < target[k] ? Math.min(target[k], vis[k] + step) : Math.max(target[k], vis[k] - step);
  dpr = devicePixelRatio || 1;
  if (V) {
    if (ov.width !== Math.round(V.cw * dpr) || ov.height !== Math.round(V.ch * dpr)) { ov.width = Math.round(V.cw * dpr); ov.height = Math.round(V.ch * dpr); }
    VI = inv4(V.vp);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, ov.width, ov.height);
  const st = globalThis.__oceanState; const pd = st && st.probe && st.probe.data && st.probe.data.length >= 84 * 4 ? st.probe.data : null;
  // v30.22b: unpack the 1 KB section (64 texels x 4 heights, +1000 = break flag) into 256 x (h, -, -, flag)
  let secD = null; if (pd) { secD = secUnpack; for (let i = 0; i < 256; i++) { const v = pd[20 * 4 + i]; const f = v > 500; secD[i * 4] = f ? v - 1000 : v; secD[i * 4 + 3] = f ? 1 : 0; } }
  // UI probe points for the GPU readback (buoy: centre, +1.5 m x, +1.5 m z)
  const surf = globalThis.__surface;
  if (surf && surf.setUiProbe) {
    const need = vis.buoy > 0 || vis.section > 0;
    if (need) { uiProbePts[0] = BUOY[0]; uiProbePts[1] = BUOY[1]; uiProbePts[2] = BUOY[0] + 1.5; uiProbePts[3] = BUOY[1]; uiProbePts[4] = BUOY[0]; uiProbePts[5] = BUOY[1] + 1.5; } // v30.22g: reused array
    surf.setUiProbe(need ? uiProbePts : null, need ? sec : null);
    if (!need) surf.setUiProbe(null, null);
  }
  if (V) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const act = !!arrow.drag;
    if (vis.wind > 0) { const g = arrowGeom(); if (g) drawArrow(g, vis.wind * (act ? 1 : arrow.hover ? 0.9 : idle() ? 0.3 : 0.7), act || arrow.hover); }
    if (vis.section > 0) { drawSectionLine(vis.section); drawSection(vis.section, secD); } else setSty(stripBox, "opacity", "0");
    if (vis.probe > 0 || pins.length) for (const p of pins) updatePin(p, vis.probe);
    if (vis.ripple > 0) drawRipples(vis.ripple);
    drawBuoy(vis.buoy, pd, st && st.fx ? st.fx.t : 0); if (!(vis.buoy > 0)) bcard.style.opacity = "0";
  }
  drawTod(vis.tod); bfUpdate();
  globalThis.__oceanUI = { layers: { ...layers }, clean, vis: { ...vis }, arrow: { L: arrow.L, ang: arrow.ang, anchor: arrow.anchor }, sunElev, tod: todMin };
  requestAnimationFrame(frame);
}
syncLayerUI(); bfShow(); setTod(todMin); applyTod();
requestAnimationFrame(frame);

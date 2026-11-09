// Two fixed screen instruments above the time dial. They are not on the ocean,
// so a drag cannot orbit the camera, and they do not change size with the sea.
import { ui } from "./core.js?v=31.9";
import { skyAt } from "./sky.js?v=30.29";

const W = 400, H = 176;
const chop = instrument("chop-inst", "Choppiness. Drag the wave up to sharpen, down to soften.");
const clar = instrument("clar-inst", "Water clarity. Lower the disk for clearer water, raise it for murkier water. Secchi depth in metres.");
chop.el.style.cursor = "ns-resize";
clar.el.style.cursor = "ns-resize";

function instrument(id, label) {
  const el = document.createElement("div");
  el.id = id;
  el.className = "sea-inst";
  el.tabIndex = 0;
  el.setAttribute("role", "slider");
  el.setAttribute("aria-label", label);
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.setAttribute("aria-hidden", "true");
  el.appendChild(canvas);
  ui.root.appendChild(el);
  return { el, ctx: canvas.getContext("2d") };
}
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function srgb(c) {
  const f = (v) => {
    const s = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
    return Math.round(clamp(s, 0, 1) * 255);
  };
  return `rgb(${f(c[0])},${f(c[1])},${f(c[2])})`;
}
function accent() {
  return srgb(skyAt(ui.todMin ?? 660).key.disc);
}
function caption(ctx, name, value) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.font = "500 20px Barlow, sans-serif";
  ctx.fillStyle = (ui.theme && ui.theme.hudFace) || "#F6F8FF";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(name, 18, 32);
  ctx.strokeStyle = accent();
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(18, 42);
  ctx.lineTo(18 + Math.min(86, ctx.measureText(name).width), 42);
  ctx.stroke();
  ctx.font = "400 24px Geist Mono, ui-monospace, monospace";
  ctx.fillStyle = (ui.theme && ui.theme.hudFace) || "#F6F8FF";
  ctx.textAlign = "right";
  ctx.fillText(value, W - 18, 34);
}
function chopK() { return clamp(Number(ui.inputs.chop.value) / 2.5, 0, 1); }
// One crest. Steepness squeezes that crest toward a cusp and leaves the
// trough broad. Crest and trough heights stay put, and there is no second crest.
function chopWave(k) {
  const s = 0.93 * k;
  const pts = [];
  for (let i = 0; i <= 140; i++) {
    const theta = -Math.PI + (2 * Math.PI * i) / 140;
    pts.push([theta - s * Math.sin(theta), Math.cos(theta)]);
  }
  return pts;
}
const CHOP_AMP = 34, CHOP_MID = 112;
const chopGlow = { v: 0 }, clarGlow = { v: 0 };
function present(active, store) {
  const now = performance.now();
  if (active) { store.v = 1; store.t = now; return 1; }
  if (ui.reduced) { store.v = 0; store.t = now; return 0; }
  const dt = Math.min(0.1, Math.max(0, (now - (store.t || now)) / 1000));
  store.t = now;
  store.v *= Math.exp(-dt / 0.14);
  if (store.v < 0.04) store.v = 0;
  return store.v;
}
function drawChop() {
  const ctx = chop.ctx, k = chopK(), v = Number(ui.inputs.chop.value);
  caption(ctx, "CHOP", v.toFixed(1));
  const raw = chopWave(k);
  const x0 = 28, x1 = W - 28;
  const pts = raw.map(([x, y]) => [x0 + ((x + Math.PI) / (2 * Math.PI)) * (x1 - x0), CHOP_MID - y * CHOP_AMP]);
  const stroke = (list, style, width) => {
    if (list.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(list[0][0], list[0][1]);
    for (let i = 1; i < list.length; i++) ctx.lineTo(list[i][0], list[i][1]);
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.lineJoin = "miter";
    ctx.miterLimit = 12;
    ctx.lineCap = "butt";
    ctx.stroke();
  };
  const hot = present(!!chopDrag, chopGlow);
  stroke(pts, "rgba(214,218,224,0.9)", 2.2 + hot * 0.6);
  stroke(pts.filter((_, i) => raw[i][1] > 0.86), accent(), 2.4 + hot * 1.1);
  let half = 0;
  const above = pts.filter((_, i) => raw[i][1] >= 0.5);
  if (above.length > 1) half = (above[above.length - 1][0] - above[0][0]) / 2;
  ui.debug.chopInst = {
    v, k, amp: CHOP_AMP, crest: 1, trough: -1, crestHalfWidth: half,
    crestY: CHOP_MID - CHOP_AMP, troughY: CHOP_MID + CHOP_AMP, hot,
  };
  chop.el.setAttribute("aria-valuemin", "0");
  chop.el.setAttribute("aria-valuemax", "2.5");
  chop.el.setAttribute("aria-valuenow", v.toFixed(1));
  chop.el.setAttribute("aria-valuetext", `${v.toFixed(1)} choppiness`);
  const why = (window.__steep && window.__steep.why) || "";
  ctx.font = "400 15px Geist Mono, ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillStyle = (ui.theme && ui.theme.hudFace) || "#F6F8FF";
  ctx.fillText(why, 18, 166);
}
const COL = { x: 156, w: 88, top: 52, bot: 162 };
function clarT(z = Number(ui.inputs.clarity.value)) { return clamp((z - 3) / 37, 0, 1); }
function drawClar() {
  const ctx = clar.ctx, z = Number(ui.inputs.clarity.value), t = clarT(z);
  caption(ctx, "CLARITY", `${Math.round(z)} m`);
  const { x, w, top, bot } = COL;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, top, w, bot - top);
  ctx.clip();
  ctx.fillStyle = "#141c18";
  ctx.fillRect(x, top, w, bot - top);
  const yDisk = top + 16 + t * (bot - top - 32);
  const veil = ctx.createLinearGradient(0, top, 0, yDisk);
  veil.addColorStop(0, `rgba(176, 214, 224, ${0.16 + 0.62 * t})`);
  veil.addColorStop(1, `rgba(46, 96, 112, ${0.04 + 0.28 * t})`);
  ctx.fillStyle = veil;
  ctx.fillRect(x, top, w, Math.max(0, yDisk - top));
  ctx.restore();
  ctx.strokeStyle = accent();
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x + w, top);
  ctx.stroke();
  const cx = x + w / 2, r = 15, fade = 1 - 0.8 * t;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, yDisk, r, 0, Math.PI * 2);
  ctx.clip();
  ctx.globalAlpha = fade;
  for (let q = 0; q < 4; q++) {
    ctx.fillStyle = q % 2 ? "#161616" : "#F4F7FB";
    ctx.beginPath();
    ctx.moveTo(cx, yDisk);
    ctx.arc(cx, yDisk, r, q * Math.PI / 2, (q + 1) * Math.PI / 2);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
  const hot = present(clarDrag, clarGlow);
  ctx.strokeStyle = hot > 0.04 ? accent() : "rgba(246,248,255,0.96)";
  ctx.lineWidth = 1.7 + hot * 1.1;
  ctx.beginPath();
  ctx.arc(cx, yDisk, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - 24, yDisk);
  ctx.lineTo(cx + 24, yDisk);
  ctx.stroke();
  ui.debug.clarInst = {
    z, t, diskY: yDisk, diskAlpha: fade, gripAlpha: 0.96,
    columnTop: top, columnBot: bot, columnH: bot - top, hot,
  };
  clar.el.setAttribute("aria-valuemin", "3");
  clar.el.setAttribute("aria-valuemax", "40");
  clar.el.setAttribute("aria-valuenow", String(Math.round(z)));
  clar.el.setAttribute("aria-valuetext", `${Math.round(z)} metres Secchi`);
}
let chopDrag = null;
chop.el.addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  e.preventDefault();
  chop.el.focus({ preventScroll: true });
  chop.el.setPointerCapture(e.pointerId);
  chopDrag = { y: e.clientY, v: Number(ui.inputs.chop.value) };
});
chop.el.addEventListener("pointermove", (e) => {
  if (!chopDrag) return;
  e.stopPropagation();
  const v = clamp(Math.round((chopDrag.v + (chopDrag.y - e.clientY) / 64 * 2.5) * 10) / 10, 0, 2.5);
  ui.set("chop", v.toFixed(1));
});
const chopUp = (e) => { chopDrag = null; if (e) e.stopPropagation(); };
chop.el.addEventListener("pointerup", chopUp);
chop.el.addEventListener("pointercancel", chopUp);
chop.el.addEventListener("lostpointercapture", () => { chopDrag = null; });

function clarityAt(clientY) {
  const r = clar.el.getBoundingClientRect();
  const y = (clientY - r.top) * (H / r.height);
  const y0 = COL.top + 16, y1 = COL.bot - 16;
  return 3 + clamp((y - y0) / (y1 - y0), 0, 1) * 37;
}
let clarDrag = false;
clar.el.addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  e.preventDefault();
  clar.el.focus({ preventScroll: true });
  clar.el.setPointerCapture(e.pointerId);
  clarDrag = true;
  ui.set("clarity", Math.round(clarityAt(e.clientY) * 10) / 10);
});
clar.el.addEventListener("pointermove", (e) => {
  if (!clarDrag) return;
  e.stopPropagation();
  ui.set("clarity", Math.round(clarityAt(e.clientY) * 10) / 10);
});
const clarUp = (e) => { clarDrag = false; e.stopPropagation(); };
clar.el.addEventListener("pointerup", clarUp);
clar.el.addEventListener("pointercancel", clarUp);

ui.slider("Water clarity", () => [Number(ui.inputs.clarity.value), `${Math.round(Number(ui.inputs.clarity.value))} metres Secchi depth`], (e) => {
  if (!/Arrow(Up|Down)/.test(e.key)) return false;
  ui.set("clarity", Math.min(40, Math.max(3, Math.round(Number(ui.inputs.clarity.value) + (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 5 : 1)))));
  return true;
});
ui.slider("Choppiness", () => [Number(ui.inputs.chop.value), Number(ui.inputs.chop.value).toFixed(1)], (e) => {
  if (!/Arrow(Left|Right|Up|Down)/.test(e.key)) return false;
  const up = e.key === "ArrowUp" || e.key === "ArrowRight";
  ui.set("chop", Math.min(2.5, Math.max(0, Math.round((Number(ui.inputs.chop.value) + (up ? 1 : -1) * (e.shiftKey ? 0.5 : 0.1)) * 10) / 10)).toFixed(1));
  return true;
});
ui.register({
  layer: "tod",
  draw(a) {
    const on = a > 0.5 && !ui.clean;
    if (!on) { chopDrag = null; clarDrag = false; }
    chop.el.classList.toggle("on", on);
    clar.el.classList.toggle("on", on);
    drawChop();
    drawClar();
  },
  hide() {
    chop.el.classList.remove("on"); clar.el.classList.remove("on");
    chopDrag = null; clarDrag = false;
  },
});

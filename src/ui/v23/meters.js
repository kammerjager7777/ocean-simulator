// Two fixed screen instruments above the time dial. They are not on the ocean,
// so a drag cannot orbit the camera, and they do not change size with the sea.
import { ui } from "./core.js?v=32.5";
import { skyAt } from "./sky.js?v=30.29";

const W = 400, H = 176, CHOP_H = 340, CHOP_W = 400;
const chop = instrument("chop-inst", "Choppiness. Drag the wave up to sharpen, down to soften.", CHOP_H, CHOP_W);
const clar = instrument("clar-inst", "Water clarity. Lower the disk for clearer water, raise it for murkier water. Secchi depth in metres.");
chop.el.style.cursor = "ns-resize";
clar.el.style.cursor = "ns-resize";

function instrument(id, label, h = H, w = W) {
  const el = document.createElement("div");
  el.id = id;
  el.className = "sea-inst";
  el.tabIndex = 0;
  el.setAttribute("role", "slider");
  el.setAttribute("aria-label", label);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.setAttribute("aria-hidden", "true");
  el.appendChild(canvas);
  ui.root.appendChild(el);
  return { el, ctx: canvas.getContext("2d"), h };
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
function caption(ctx, name, value, h = H, w = W) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
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
  ctx.fillText(value, w - 18, 34);
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
const CHOP_AMP = 24, CHOP_MID = 86;
const AGE_BOX = { x0: 12, x1: 190, y0: 148, y1: 276 };
const CUR_BOX = { x0: 200, x1: 388, y0: 148, y1: 276 };
const WHY_Y = 286;
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
  caption(ctx, "CHOP", v.toFixed(1), CHOP_H, CHOP_W);
  const raw = chopWave(k);
  const x0 = 28, x1 = CHOP_W - 28;
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
  drawAgeGlyph(ctx, present(!!ageDrag, ageGlow));
  drawCurGlyph(ctx, present(!!curDrag, curGlow));
  ctx.fillStyle = "#000";
  ctx.fillRect(12, WHY_Y, CHOP_W - 24, CHOP_H - WHY_Y);
  const why = (window.__steep && window.__steep.why) || "";
  ctx.font = "400 18px Geist Mono, ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = (ui.theme && ui.theme.hudFace) || "#F6F8FF";
  ctx.fillText(why, 18, (WHY_Y + CHOP_H) / 2);
}
function ageWord(age) { return age >= 0.72 ? "old" : age < 0.4 ? "young" : "building"; }
function curWord(cur) { return Math.abs(cur) < 0.15 ? "still" : `${cur < 0 ? "against" : "with"} ${Math.abs(cur).toFixed(1)}`; }
function glyphLabel(ctx, text, x, y) {
  ctx.font = "500 22px Barlow, sans-serif";
  ctx.fillStyle = (ui.theme && ui.theme.hudFace) || "#F6F8FF";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}
function drawAgeGlyph(ctx, hot) {
  const age = ui.inputs.age ? Number(ui.inputs.age.value) : 0.28;
  const cx = 52, cy = 214, r = 28;
  ctx.strokeStyle = "rgba(214,218,224,0.55)";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI * 0.8, Math.PI * 2.2, false);
  ctx.stroke();
  const a = Math.PI * 0.8 + age * Math.PI * 1.4;
  ctx.strokeStyle = hot > 0.04 ? accent() : "rgba(246,248,255,0.95)";
  ctx.lineWidth = 2.2 + hot;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.cos(a) * (r - 6), cy + Math.sin(a) * (r - 6));
  ctx.stroke();
  ctx.fillStyle = accent();
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fill();
  glyphLabel(ctx, ageWord(age), 100, cy);
}
function drawCurGlyph(ctx, hot) {
  const cur = ui.inputs.cur ? Number(ui.inputs.cur.value) : 0;
  const still = Math.abs(cur) < 0.15;
  const dir = cur < 0 ? -1 : 1;
  const n = still ? 0 : Math.min(3, 1 + Math.round(Math.abs(cur) / 1.2));
  const cy = 228, cx = (CUR_BOX.x0 + CUR_BOX.x1) / 2;
  glyphLabel(ctx, curWord(cur), CUR_BOX.x0, 176);
  ctx.strokeStyle = hot > 0.04 ? accent() : "rgba(214,218,224,0.9)";
  ctx.lineWidth = 2.2 + hot * 0.6;
  ctx.lineJoin = "miter";
  ctx.lineCap = "butt";
  if (still) {
    ctx.beginPath();
    ctx.moveTo(cx - 26, cy);
    ctx.lineTo(cx + 26, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    for (let i = 0; i < n; i++) {
      const x = cx + dir * (i - (n - 1) / 2) * 26;
      ctx.beginPath();
      ctx.moveTo(x - dir * 8, cy - 10);
      ctx.lineTo(x + dir * 8, cy);
      ctx.lineTo(x - dir * 8, cy + 10);
      ctx.stroke();
    }
  }
}
function chopCanvasPos(e) {
  const r = chop.el.getBoundingClientRect();
  const canvas = chop.el.querySelector("canvas");
  return {
    x: ((e.clientX - r.left) / r.width) * canvas.width,
    y: ((e.clientY - r.top) / r.height) * canvas.height,
  };
}
function chopHit(e) {
  const p = chopCanvasPos(e);
  if (p.y >= WHY_Y) return "why";
  if (p.x >= AGE_BOX.x0 && p.x <= AGE_BOX.x1 && p.y >= AGE_BOX.y0 && p.y <= AGE_BOX.y1) return "age";
  if (p.x >= CUR_BOX.x0 && p.x <= CUR_BOX.x1 && p.y >= CUR_BOX.y0 && p.y <= CUR_BOX.y1) return "cur";
  return "wave";
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
let chopDrag = null, ageDrag = null, curDrag = null, chopFocus = "wave", holdChop = 0;
const ageGlow = { v: 0 }, curGlow = { v: 0 };
function followSea() { if (typeof ui.clearChopOverride === "function") ui.clearChopOverride(); }
chop.el.addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  e.preventDefault();
  chop.el.focus({ preventScroll: true });
  chop.el.setPointerCapture(e.pointerId);
  const hit = chopHit(e);
  chopFocus = hit === "why" ? chopFocus : hit;
  if (hit === "why") return;
  if (hit === "age") { ageDrag = { x: e.clientX, v: Number(ui.inputs.age.value) }; return; }
  if (hit === "cur") { curDrag = { x: e.clientX, v: Number(ui.inputs.cur.value) }; return; }
  if (performance.now() < holdChop) return;
  chopDrag = { y: e.clientY, v: Number(ui.inputs.chop.value), moved: false };
});
chop.el.addEventListener("pointermove", (e) => {
  if (!ageDrag && !curDrag && !chopDrag) {
    const hit = chopHit(e);
    chop.el.style.cursor = hit === "why" ? "pointer" : hit === "wave" ? "ns-resize" : "ew-resize";
    return;
  }
  e.stopPropagation();
  if (ageDrag) {
    const v = clamp(Math.round((ageDrag.v + (e.clientX - ageDrag.x) / 160) * 100) / 100, 0, 1);
    ui.set("age", String(v));
    return;
  }
  if (curDrag) {
    const v = clamp(Math.round((curDrag.v + (e.clientX - curDrag.x) / 36) * 10) / 10, -3, 3);
    ui.set("cur", String(v));
    return;
  }
  if (performance.now() < holdChop) return;
  if (Math.abs(e.clientY - chopDrag.y) < 4) return;
  chopDrag.moved = true;
  const v = clamp(Math.round((chopDrag.v + (chopDrag.y - e.clientY) / 64 * 2.5) * 10) / 10, 0, 2.5);
  ui.set("chop", v.toFixed(1));
});
chop.el.addEventListener("pointerup", (e) => {
  if (chopHit(e) === "why" && window.__steep && window.__steep.override) followSea();
  chopDrag = null; ageDrag = null; curDrag = null;
  e.stopPropagation();
});
chop.el.addEventListener("pointercancel", () => { chopDrag = null; ageDrag = null; curDrag = null; });
chop.el.addEventListener("lostpointercapture", () => { chopDrag = null; ageDrag = null; curDrag = null; });
chop.el.addEventListener("dblclick", (e) => {
  if (chopHit(e) !== "wave") return;
  e.preventDefault();
  e.stopPropagation();
  holdChop = performance.now() + 450;
  chopDrag = null;
  followSea();
});
chop.el.addEventListener("keydown", (e) => {
  const up = e.key === "ArrowUp" || e.key === "ArrowRight";
  const down = e.key === "ArrowDown" || e.key === "ArrowLeft";
  if (!up && !down) return;
  if (chopFocus === "age") {
    const step = e.shiftKey ? 0.1 : 0.02;
    const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : -1;
    if (!dir) return;
    ui.set("age", String(clamp(Math.round((Number(ui.inputs.age.value) + dir * step) * 100) / 100, 0, 1)));
  } else if (chopFocus === "cur") {
    const step = e.shiftKey ? 0.5 : 0.1;
    const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : -1;
    ui.set("cur", String(clamp(Math.round((Number(ui.inputs.cur.value) + dir * step) * 10) / 10, -3, 3)));
  } else return;
  e.preventDefault();
  e.stopPropagation();
});

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
    if (!on) { chopDrag = null; ageDrag = null; curDrag = null; clarDrag = false; }
    chop.el.classList.toggle("on", on);
    clar.el.classList.toggle("on", on);
    drawChop();
    drawClar();
  },
  hide() {
    chop.el.classList.remove("on"); clar.el.classList.remove("on");
    chopDrag = null; ageDrag = null; curDrag = null; clarDrag = false;
  },
});

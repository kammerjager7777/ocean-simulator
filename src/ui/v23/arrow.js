// Wind. A compass ring circles the ocean in world bearings: 0 is +X and is
// labeled N, 90 is +Z and is labeled E, matching the sea's (cos θ, sin θ) wind.
// Dragging the ring sets direction only. The fixed card sets speed only.
import { ui } from "./core.js?v=31.9";
import { skyAt } from "./sky.js?v=30.29";
import { sample, release, cancel, velocity } from "./coast.js?v=30.29";
const LO = [0, .3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 36.9];
export const bfLevel = (u) => { let i = 0; while (i < 12 && u >= LO[i + 1] - .05) i++; return Math.min(12, i + Math.min(.999, (u - LO[i]) / (LO[i + 1] - LO[i]))); };
const BF = ["Calm", "Light air", "Light breeze", "Gentle breeze", "Moderate breeze", "Fresh breeze", "Strong breeze", "Near gale", "Gale", "Strong gale", "Storm", "Violent storm", "Hurricane"];
const U = () => Number(ui.inputs.spd.value), D = () => Number(ui.inputs.dir.value);
const CARD_W = 400, CARD_H = 224;
const SPD_MIN = 0, SPD_MAX = 25, SPD_STEP = 0.5;
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function wrap(deg) { return ((Math.round(deg) % 360) + 360) % 360; }
function bearing(deg) { return `${String(wrap(deg)).padStart(3, "0")}°`; }
function snapSpd(v) {
  const s = Math.round(clamp(v, SPD_MIN, SPD_MAX) / SPD_STEP) * SPD_STEP;
  return Math.round(s * 2) / 2;
}
function srgb(c) {
  const f = (v) => {
    const s = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
    return Math.round(clamp(s, 0, 1) * 255);
  };
  return `rgb(${f(c[0])},${f(c[1])},${f(c[2])})`;
}
function accent() { return srgb(skyAt(ui.todMin ?? 660).key.disc); }
function skyLuma() {
  const c = skyAt(ui.todMin ?? 660).key.hor;
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
// 1 on a bright horizon (black ink), 0 once the sky has gone dark (light ink).
function dayInk() { return clamp((skyLuma() - 0.045) / 0.16, 0, 1); }
function hexRgb(hex) {
  const h = String(hex || "").replace("#", "");
  if (h.length < 6) return [246, 248, 255];
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function ringInk(alpha = 1) {
  const t = dayInk();
  const face = ui.theme && (t > 0.55 ? ui.theme.skyInk : ui.theme.hudFace);
  const [r, g, b] = hexRgb(face || (t > 0.55 ? "#0C1724" : "#F6F8FF"));
  return `rgba(${r},${g},${b},${alpha})`;
}
function faceInk() { return (ui.theme && ui.theme.hudFace) || "#F6F8FF"; }
function textGlow(ctx) {
  const a = ui.theme && ui.theme.glowA || 0;
  ctx.shadowColor = (ui.theme && ui.theme.accent) || "#F6F8FF";
  ctx.shadowBlur = a > 0.28 ? 4 : 2;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;
}
function ringRadius() {
  const S = ui.S();
  // A little outside the block's diagonal. v31.8 is slightly larger than the 0.06 S + 8 m ring.
  return (S / 2) * Math.SQRT2 + S * 0.1 + 18;
}
function worldAt(deg, radius = ringRadius()) {
  const a = deg * Math.PI / 180;
  return [Math.cos(a) * radius, 0, Math.sin(a) * radius];
}
// The ring is a 2D overlay drawn after the sea, so the water cannot cover it in the
// depth buffer. v31.7 still dropped a sample when seaHides said the water box sat on
// the ray, or when the projected point fell inside the block's screen hull. That is
// why the compass vanished wherever the ocean's projection overlapped it, including
// grazing views. Points behind the camera still fail ui.proj.
function project(w) {
  if (!w) return null;
  return ui.proj(w[0], w[1], w[2]);
}

const el = document.createElement("div");
el.id = "wind-inst";
el.className = "sea-inst";
el.tabIndex = 0;
el.setAttribute("role", "slider");
el.setAttribute("aria-label", "Wind speed wheel. Drag or flick up and down. Direction is the compass ring.");
const card = document.createElement("canvas");
card.width = CARD_W;
card.height = CARD_H;
card.setAttribute("aria-hidden", "true");
el.appendChild(card);
ui.root.appendChild(el);
const cctx = card.getContext("2d");
let spdDrag = null, spdGlow = 0, spdLive = 12;
function publishSpd(v) {
  spdLive = clamp(v, SPD_MIN, SPD_MAX);
  ui.set("spd", snapSpd(spdLive));
}
el.addEventListener("pointerdown", (e) => {
  e.stopPropagation();
  e.preventDefault();
  el.focus({ preventScroll: true });
  el.setPointerCapture(e.pointerId);
  cancel("spd");
  spdLive = U();
  spdDrag = { y: e.clientY, v: spdLive };
  sample("spd", spdLive, performance.now());
});
el.addEventListener("pointermove", (e) => {
  if (!spdDrag) return;
  e.stopPropagation();
  publishSpd(spdDrag.v + (spdDrag.y - e.clientY) * 0.045);
  sample("spd", spdLive, performance.now());
});
const spdEnd = (e) => {
  if (spdDrag) release("spd", { min: SPD_MIN, max: SPD_MAX, maxAbs: 14, ignore: 3, tau: 0.3, stop: 0.35, on: publishSpd });
  spdDrag = null;
  if (e) e.stopPropagation();
};
el.addEventListener("pointerup", spdEnd);
el.addEventListener("pointercancel", () => { spdDrag = null; cancel("spd"); });
el.addEventListener("wheel", (e) => {
  e.preventDefault();
  e.stopPropagation();
  cancel("spd");
  // Scroll up (deltaY < 0) decreases speed, so the drum turns with the wheel.
  publishSpd(snapSpd(U() + (e.deltaY < 0 ? -SPD_STEP : SPD_STEP)));
}, { passive: false });
ui.slider("Wind speed", () => [U(), `${U()} metres per second`], (e) => { if (!/Arrow(Up|Down)/.test(e.key)) return false; ui.set("spd", Math.max(0, Math.min(25, U() + (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 5 : .5)))); return true; });
ui.slider("Wind direction", () => [D(), `${D()} degrees`], (e) => { if (!/Arrow(Left|Right)/.test(e.key)) return false; ui.set("dir", wrap(D() + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 15 : 1))); return true; });

function drawCard() {
  if (!spdDrag && velocity("spd") === 0) {
    if (Math.abs(spdLive - snapSpd(spdLive)) > 1e-3) publishSpd(snapSpd(spdLive));
    spdLive = U();
  }
  const spd = U(), deg = D();
  const force = Math.min(12, Math.floor(bfLevel(spd)));
  const bf = `B${force} · ${BF[force]}`;
  cctx.clearRect(0, 0, CARD_W, CARD_H);
  cctx.fillStyle = "#000";
  cctx.fillRect(0, 0, CARD_W, CARD_H);
  cctx.font = "500 20px Barlow, sans-serif";
  cctx.fillStyle = faceInk();
  cctx.textAlign = "left";
  cctx.textBaseline = "alphabetic";
  textGlow(cctx);
  cctx.fillText("WIND", 18, 32);
  cctx.shadowBlur = 0;
  cctx.strokeStyle = accent();
  cctx.lineWidth = 1.5;
  cctx.beginPath();
  cctx.moveTo(18, 42);
  cctx.lineTo(18 + cctx.measureText("WIND").width, 42);
  cctx.stroke();
  cctx.font = "400 22px Geist Mono, ui-monospace, monospace";
  cctx.fillStyle = faceInk();
  cctx.textAlign = "right";
  textGlow(cctx);
  cctx.fillText(bearing(deg), CARD_W - 18, 34);
  cctx.shadowBlur = 0;
  if (spdDrag || velocity("spd")) spdGlow = 1;
  else spdGlow = ui.reduced ? 0 : spdGlow * Math.exp(-(ui.dt || 0.016) / 0.14);
  const bodyTop = 52, bodyBot = 176;
  const mid = (bodyTop + bodyBot) / 2, row = 36, cx = CARD_W / 2;
  cctx.strokeStyle = "rgba(246,248,255,0.35)";
  cctx.lineWidth = 1;
  cctx.beginPath();
  cctx.moveTo(cx - 86, mid - 16);
  cctx.lineTo(cx + 86, mid - 16);
  cctx.moveTo(cx - 86, mid + 16);
  cctx.lineTo(cx + 86, mid + 16);
  cctx.stroke();
  cctx.textBaseline = "middle";
  for (let s = SPD_MIN; s <= SPD_MAX + 1e-6; s += SPD_STEP) {
    const dy = (spdLive - s) * row;
    if (Math.abs(dy) > 74) continue;
    const near = 1 - Math.min(1, Math.abs(dy) / 74);
    const selected = Math.abs(s - spd) < 0.01;
    const num = s.toFixed(1);
    cctx.globalAlpha = selected ? 1 : 0.16 + 0.5 * near * near;
    cctx.fillStyle = selected && spdGlow > 0.08 ? accent() : faceInk();
    if (selected) {
      cctx.font = "400 26px Geist Mono, ui-monospace, monospace";
      const nw = cctx.measureText(num).width;
      cctx.font = "500 16px Barlow, sans-serif";
      const uw = cctx.measureText("m/s").width;
      const gap = 10, left = cx - (nw + gap + uw) / 2;
      cctx.font = "400 26px Geist Mono, ui-monospace, monospace";
      cctx.textAlign = "left";
      cctx.fillText(num, left, mid + dy);
      cctx.font = "500 16px Barlow, sans-serif";
      cctx.fillStyle = faceInk();
      cctx.fillText("m/s", left + nw + gap, mid + dy);
    } else {
      cctx.font = "400 15px Geist Mono, ui-monospace, monospace";
      cctx.textAlign = "center";
      cctx.fillText(num, cx, mid + dy);
    }
  }
  cctx.globalAlpha = 1;
  cctx.textAlign = "left";
  cctx.strokeStyle = "rgba(246,248,255,0.28)";
  cctx.lineWidth = 1;
  cctx.beginPath();
  cctx.moveTo(16, 176);
  cctx.lineTo(CARD_W - 16, 176);
  cctx.stroke();
  cctx.font = "500 16px Barlow, sans-serif";
  cctx.fillStyle = faceInk();
  cctx.textBaseline = "alphabetic";
  cctx.fillText(bf, 18, 206);
  el.setAttribute("aria-valuemin", "0");
  el.setAttribute("aria-valuemax", "25");
  el.setAttribute("aria-valuenow", spd.toFixed(1));
  el.setAttribute("aria-valuetext", `${spd.toFixed(1)} metres per second, ${bearing(deg)}, ${bf}`);
  ui.debug.windInst = { deg, spd, spdLive, bf, fixed: true, directionOnCard: false, beaufort: true, coast: velocity("spd"), t: performance.now() };
}

const HUD = ["hud-rows", "hud-sheet", "hud-head", "hb-set", "hb-pause", "hud-pl", "hud-stats", "hud-ver", "wind-inst", "chop-inst", "clar-inst", "solar-dial"];
function covered(x, y) {
  for (const id of HUD) {
    const node = document.getElementById(id);
    if (!node || getComputedStyle(node).opacity === "0") continue;
    const r = node.getBoundingClientRect();
    if (r.width < 2) continue;
    if (x >= r.left - 8 && x <= r.right + 8 && y >= r.top - 8 && y <= r.bottom + 8) return true;
  }
  return false;
}
function outward(deg, radius = ringRadius()) {
  // The contact is the same world point the ring stroke uses. Its screen
  // direction is the rail's own normal, not the projected world radius:
  // that radius is not perpendicular to the projected ring, so the marker
  // swiveled as the camera orbited and the stroke cut across the glyph.
  const at = project(worldAt(deg, radius));
  if (!at) return null;
  const step = 2.5;
  const prev = project(worldAt(deg - step, radius));
  const next = project(worldAt(deg + step, radius));
  let tx, ty;
  if (prev && next) { tx = next[0] - prev[0]; ty = next[1] - prev[1]; }
  else if (next) { tx = next[0] - at[0]; ty = next[1] - at[1]; }
  else if (prev) { tx = at[0] - prev[0]; ty = at[1] - prev[1]; }
  else return null;
  const tl = Math.hypot(tx, ty);
  if (tl < 0.35) return null;
  tx /= tl; ty /= tl;
  let nx = -ty, ny = tx;
  const bump = Math.max(6, radius * 0.04);
  const far = project(worldAt(deg, radius + bump));
  const inn = project(worldAt(deg, Math.max(0, radius - bump)));
  let sx = 0, sy = 0;
  if (far) { sx += far[0] - at[0]; sy += far[1] - at[1]; }
  if (inn) { sx += at[0] - inn[0]; sy += at[1] - inn[1]; }
  if (Math.hypot(sx, sy) < 0.35) {
    const c = ui.proj(0, 0, 0);
    if (c) { sx = at[0] - c[0]; sy = at[1] - c[1]; }
  }
  if (nx * sx + ny * sy < 0) { nx = -nx; ny = -ny; }
  return { at, ux: nx, uy: ny };
}
function ringSamples() {
  const radius = ringRadius();
  const pts = [];
  for (let i = 0; i <= 120; i++) pts.push(project(worldAt((i / 120) * 360, radius)));
  const vis = pts.filter(Boolean);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of vis) { minX = Math.min(minX, p[0]); minY = Math.min(minY, p[1]); maxX = Math.max(maxX, p[0]); maxY = Math.max(maxY, p[1]); }
  const major = vis.length ? Math.max(maxX - minX, maxY - minY) : 0;
  const minor = vis.length ? Math.min(maxX - minX, maxY - minY) : 0;
  return { pts, radius, major, minor };
}
function drawRing(pose) {
  const ctx = ui.ctx;
  const { pts, radius, major, minor } = ringSamples();
  ctx.save();
  ctx.lineCap = "butt";
  ctx.lineJoin = "round";
  ctx.strokeStyle = ringInk(0.92);
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    if (!p) { pen = false; continue; }
    if (!pen) { ctx.moveTo(p[0], p[1]); pen = true; }
    else ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
  const labelOk = minor >= 28 && major >= 70;
  const mid = labelOk && major >= 180;
  const numbers = labelOk && major >= 280;
  const kept = [];
  const nearKept = (p) => kept.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 18);
  if (labelOk) {
    for (let deg = 0; deg < 360; deg += 10) {
      const o = outward(deg, radius);
      if (!o) continue;
      const cardinal = deg % 90 === 0;
      const thirty = deg % 30 === 0;
      const len = cardinal ? 13 : thirty ? 9 : 6;
      ctx.strokeStyle = ringInk(cardinal ? 1 : 0.78);
      ctx.lineWidth = cardinal ? 1.9 : 1.25;
      ctx.beginPath();
      ctx.moveTo(o.at[0], o.at[1]);
      ctx.lineTo(o.at[0] + o.ux * len, o.at[1] + o.uy * len);
      ctx.stroke();
    }
    const place = (deg, text, font, fill) => {
      const o = outward(deg, radius);
      if (!o) return;
      const x = o.at[0] + o.ux * 24, y = o.at[1] + o.uy * 24;
      if (covered(x, y) || nearKept([x, y])) return;
      kept.push([x, y]);
      ctx.font = font;
      ctx.fillStyle = fill;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      textGlow(ctx);
      ctx.fillText(text, x, y);
      ctx.shadowBlur = 0;
    };
    for (const [deg, text] of [[0, "N"], [90, "E"], [180, "S"], [270, "W"]]) place(deg, text, "600 15px Barlow, sans-serif", ringInk(1));
    if (mid) for (const [deg, text] of [[45, "NE"], [135, "SE"], [225, "SW"], [315, "NW"]]) place(deg, text, "500 12px Barlow, sans-serif", ringInk(0.9));
    if (numbers) for (const deg of [30, 60, 120, 150, 210, 240, 300, 330]) place(deg, String(deg), "400 11px Geist Mono, ui-monospace, monospace", ringInk(0.82));
  }
  const grown = pose === "drag" ? 18 : pose === "hover" ? 16 : 14;
  const mark = outward(D(), radius);
  if (mark) {
    const ux = mark.ux, uy = mark.uy, px = -uy, py = ux;
    const half = pose === "idle" ? 5 : 5.6;
    // Base midpoint is the rail sample. The glyph stays on the outer side of
    // that one point while dragging, orbiting, and zooming.
    const tip = [mark.at[0] + ux * grown, mark.at[1] + uy * grown];
    const poly = [
      tip,
      [mark.at[0] + px * half, mark.at[1] + py * half],
      [mark.at[0] - px * half, mark.at[1] - py * half],
    ];
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (const p of poly) ctx.lineTo(p[0], p[1]);
    ctx.closePath();
    ctx.fillStyle = pose === "drag" ? "#FFE08A" : pose === "hover" ? "#FFC14D" : "#E8941A";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = ringInk(1);
    ctx.stroke();
  }
  ctx.restore();
  const screen = {};
  for (const [name, deg] of [["n", 0], ["e", 90], ["s", 180], ["w", 270]]) screen[name] = project(worldAt(deg));
  ui.debug.compass = {
    deg: D(), spd: U(), radius, major, minor, labelOk, dayInk: dayInk(), ink: ringInk(1),
    world: { n: worldAt(0), e: worldAt(90) },
    screen, marker: mark && mark.at, axis: mark && [mark.ux, mark.uy], hiddenMark: !mark, pose, coast: velocity("dir"),
  };
}
function nearestVisible(x, y, limit) {
  let best = Infinity, at = null;
  for (let i = 0; i < 72; i++) {
    const p = project(worldAt(i * 5));
    if (!p) continue;
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d < best) { best = d; at = p; }
  }
  return best <= limit ? at : null;
}

let ringDrag = false;
const ring = ui.register({
  layer: "wind",
  hit(x, y) {
    const { minor } = ringSamples();
    if (minor < 18) return null;
    const mark = project(worldAt(D()));
    if (mark && Math.hypot(mark[0] - x, mark[1] - y) <= 18) return "mark";
    if (nearestVisible(x, y, 14)) return "ring";
    return null;
  },
  body(x, y) {
    const mark = project(worldAt(D()));
    if (!mark || ringSamples().minor < 18) return null;
    return Math.hypot(mark[0] - x, mark[1] - y) <= 12 ? "mark" : null;
  },
  down() {
    cancel("dir");
    ringDrag = true;
    ui.drag.key = "wind";
    sample("dir", D(), performance.now());
  },
  move(h, x, y, e) {
    if (!ringDrag) return;
    const p = ui.toPlane(x, y, 0);
    if (!p) return;
    let deg = Math.atan2(p[2], p[0]) * 180 / Math.PI;
    if (e.shiftKey) deg = Math.round(deg / 15) * 15;
    const next = wrap(deg);
    ui.set("dir", next);
    sample("dir", next, performance.now());
  },
  up() {
    if (ringDrag) release("dir", { wrap: 360, maxAbs: 220, ignore: 45, tau: 0.28, stop: 12, on: (v) => ui.set("dir", wrap(v)) });
    ringDrag = false;
  },
  cursor: () => "grab",
  draw(a) {
    const on = a > 0.5 && !ui.clean;
    if (!on) { spdDrag = null; ringDrag = false; cancel("dir"); cancel("spd"); }
    el.classList.toggle("on", on);
    if (!on) return;
    drawCard();
    const pose = ringDrag ? "drag" : (ui.hover && ui.hover.obj === ring) ? "hover" : "idle";
    drawRing(pose);
  },
  hide() { el.classList.remove("on"); spdDrag = null; ringDrag = false; cancel("dir"); cancel("spd"); },
});
ui.debug.windHit = (x, y) => ring.hit(x, y);
void ring;

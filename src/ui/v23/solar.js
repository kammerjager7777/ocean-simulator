// v30.27 Apple-Watch-style Solar control. The full 24 h sine is centered in the card:
// midnight at both padded ends, noon on the card center, 06:00 and 18:00 symmetric.
// A short coast continues a flick, wraps at midnight, and stops at the next pointer.
import { ui } from "./core.js?v=32.10";
import { elevAt, isSunUp } from "./sky.js?v=30.29";
import { sample, release, cancel, velocity } from "./coast.js?v=30.29";
const D2R = Math.PI / 180, URL_T = ui.P.get("tod");
const X0 = 28, X1 = 372, HY = 101, AMP = 64;
ui.todMin = (() => { if (URL_T && /^\d{1,2}:\d{2}$/.test(URL_T)) { const [h, m] = URL_T.split(":").map(Number); return h * 60 + m; } const v = Number(ui.store.get("ocean.ui.tod", "660")); return Number.isFinite(v) && v >= 0 && v < 1440 ? v : 660; })();
const fmt = (m) => `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(Math.round(m % 60) % 60).padStart(2, "0")}`;
function setT(m) { m = ((Math.round(m) % 1440) + 1440) % 1440; ui.todMin = m; if (!URL_T) ui.store.set("ocean.ui.tod", String(m)); drawDial(); kb.upd(); }
let m0 = 660, x0 = 0, drag = null, settle = 0;
const el = document.createElement("div"); el.id = "solar-dial"; el.setAttribute("role", "slider"); el.tabIndex = 0; el.setAttribute("aria-label", "Solar time: drag left or right through a full day");
el.setAttribute("aria-valuemin", "0"); el.setAttribute("aria-valuemax", "1440");
el.innerHTML = `<canvas width="400" height="312" aria-hidden="true"></canvas><div class="solar-time"><span class="solar-clock">11:00</span><span class="solar-phase">SUN 65°</span></div><div class="solar-caption">SOLAR · DRAG ↔ THROUGH 24 H</div>`;
ui.root.appendChild(el);
const crib = el.querySelector("canvas"), ctx = crib.getContext("2d"), clock = el.querySelector(".solar-clock"), phase = el.querySelector(".solar-phase");
const kb = { upd() { /* the dial itself is the single keyboard slider */ } };
function path(m, end = false) {
  const wrapped = ((m % 1440) + 1440) % 1440;
  const u = end ? 1 : wrapped / 1440;
  const s = Math.sin(elevAt(end ? 0 : wrapped)) / Math.sin(elevAt(720));
  return [X0 + u * (X1 - X0), HY - s * AMP];
}
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// Continuous altitude weights. Inferred from the four Graph stills, not from an Apple shader.
function glowState(e) {
  const high = smooth(8, 32, e);
  const horizon = smooth(-3, 6, e) * (1 - smooth(5, 18, e));
  const below = smooth(-16, -2, e) * (1 - smooth(-2, 8, e));
  const deep = 1 - smooth(-22, -8, e);
  return { high, horizon, below, deep };
}
function radial(x, y, r0, r1, stops) {
  const g = ctx.createRadialGradient(x, y, r0, x, y, r1);
  for (const [t, c] of stops) g.addColorStop(t, c);
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r1, 0, Math.PI * 2);
  ctx.fill();
}
function paintSunLight(x, y, e) {
  const s = glowState(e);
  // Elevated: white core, broad cyan halo. Corners stay black because the radius ends.
  if (s.high > 0.02) {
    radial(x, y, 2, 150, [
      [0, `rgba(255,255,255,${0.9 * s.high})`],
      [0.08, `rgba(186,226,255,${0.55 * s.high})`],
      [0.35, `rgba(70,150,255,${0.28 * s.high})`],
      [1, "rgba(40,80,180,0)"],
    ]);
  }
  // Horizon: warm bloom stretched sideways, mostly above the horizon line, violet haze above it.
  const warm = Math.max(s.horizon, s.below * 0.85);
  if (warm > 0.02) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, 400, HY + 28);
    ctx.clip();
    ctx.translate(x, Math.min(y, HY));
    ctx.scale(1.85, 0.55);
    radial(0, -18, 4, 150, [
      [0, `rgba(120,90,190,${0.22 * warm})`],
      [0.45, `rgba(70,50,140,${0.1 * warm})`],
      [1, "rgba(40,20,80,0)"],
    ]);
    radial(0, 8, 2, 130, [
      [0, `rgba(255,214,170,${0.75 * warm})`],
      [0.18, `rgba(255,120,90,${0.55 * warm})`],
      [0.42, `rgba(255,80,120,${0.28 * warm})`],
      [1, "rgba(180,40,60,0)"],
    ]);
    ctx.restore();
    // A little of the warm light spills under the line, then dies. No hard cut.
    ctx.save();
    ctx.globalAlpha = 0.35 * warm * (1 - s.deep);
    ctx.translate(x, HY + 6);
    ctx.scale(1.4, 0.35);
    radial(0, 0, 2, 90, [[0, "rgba(255,110,80,0.45)"], [1, "rgba(255,80,70,0)"]]);
    ctx.restore();
  }
  return s;
}
function sunDot(x, y, sun, e) {
  ctx.save();
  if (!sun) {
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = "#C5D0DE";
    ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#6E7688";
    ctx.beginPath(); ctx.arc(x + 2.4, y - 1.6, 4.4, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    return;
  }
  // Filled disc above the horizon, dark centre and bright rim once it has set. The mix is altitude, not a mode switch.
  const rim = smooth(2, -8, e);
  const body = 1 - rim;
  if (body > 0.02) {
    ctx.globalAlpha = body;
    ctx.fillStyle = "#FFF6E4";
    ctx.beginPath(); ctx.arc(x, y, 6.2, 0, Math.PI * 2); ctx.fill();
  }
  if (rim > 0.02) {
    ctx.globalAlpha = rim;
    ctx.fillStyle = "#0A0A0C";
    ctx.beginPath(); ctx.arc(x, y, 6.2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(255,244,230,0.95)";
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(x, y, 5.4, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}
function drawDial() {
  const m = ui.todMin, e = elevAt(m) / D2R, g = glowState(e);
  el.style.setProperty("--solar-sky", "#000");
  ctx.clearRect(0, 0, 400, 312);
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 400, 312);
  const up = isSunUp(m), other = (m + 720) % 1440, pOther = path(other), p = path(m);
  paintSunLight(p[0], p[1], e);
  ctx.lineWidth = 1.15; ctx.strokeStyle = "rgba(186,190,198,.45)"; ctx.beginPath(); ctx.moveTo(X0, HY); ctx.lineTo(X1, HY); ctx.stroke();
  ctx.strokeStyle = "rgba(214,218,224,.72)"; ctx.lineWidth = 2.1; ctx.beginPath();
  for (let n = 0; n <= 240; n++) { const q = path(n * 6, n === 240); if (!n) ctx.moveTo(...q); else ctx.lineTo(...q); } ctx.stroke();
  for (const [hour, label] of [[6, "6"], [12, "12"], [18, "18"]]) { const q = path(hour * 60); ctx.fillStyle = hour === 12 ? "rgba(244,247,251,.92)" : "rgba(220,224,230,.8)"; ctx.font = "500 22px Barlow, sans-serif"; ctx.textAlign = "center"; ctx.fillText(label, q[0], 132); }
  sunDot(...pOther, false, e);
  sunDot(...p, true, e);
  if (drag) settle = 1;
  else settle = ui.reduced ? 0 : settle * Math.exp(-(ui.dt || 0.016) / 0.32);
  if (settle > 0.04) {
    ctx.save();
    ctx.strokeStyle = drag ? "rgba(255,246,228,0.95)" : `rgba(255,246,228,${0.55 * settle})`;
    ctx.lineWidth = drag ? 2.4 : 1.6;
    ctx.beginPath();
    ctx.arc(p[0], p[1], drag ? 11 : 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
  el.classList.toggle("grab", !!drag);
  el.classList.toggle("settle", !drag && settle > 0.2);
  clock.textContent = fmt(m); phase.textContent = Math.abs(e) < 4 ? (m < 720 ? "SUNRISE" : "SUNSET") : e >= 4 ? `SUN ${Math.round(e)}°` : m === 0 ? "NIGHT · MIDNIGHT" : `MOON ${Math.round(-e)}°`;
  el.setAttribute("aria-valuenow", String(Math.round(m))); el.setAttribute("aria-valuetext", `${fmt(m)}, ${e >= 0 ? "sun" : "night"}`);
  const p6 = path(360), p12 = path(720), p18 = path(1080), p0 = path(0), p24 = path(0, true);
  ui.debug.solar = {
    m, x: p[0], y: p[1], sun: up, e, dial: true, inSkyDrag: false, coast: velocity("tod"), grab: !!drag, settle,
    noonX: p12[0], x6: p6[0], x18: p18[0], x0: p0[0], x24: p24[0], mid: 200,
    glow: g,
    dialRect: [el.offsetLeft, el.offsetTop, el.offsetWidth, el.offsetHeight],
  };
}
el.addEventListener("pointerdown", (ev) => {
  cancel("tod");
  ev.stopPropagation(); ev.preventDefault(); el.focus({ preventScroll: true }); el.setPointerCapture(ev.pointerId);
  const r = el.getBoundingClientRect(); x0 = ev.clientX; m0 = ui.todMin; drag = { id: ev.pointerId };
  const cx = (ev.clientX - r.left) * 400 / r.width, cy = (ev.clientY - r.top) * 312 / r.height;
  const dot = path(ui.todMin);
  if (Math.hypot(cx - dot[0], cy - dot[1]) < 28) m0 = ((cx - X0) / (X1 - X0)) * 1440;
  setT(m0); sample("tod", m0, performance.now()); ev.stopImmediatePropagation();
});
el.addEventListener("pointermove", (ev) => {
  if (!drag) return; ev.stopPropagation(); ev.preventDefault();
  const m = m0 + (ev.clientX - x0) * 2.6;
  setT(m); sample("tod", m, performance.now());
});
const stop = (ev) => {
  // A little more carry than tau 0.22: the same speed cap and the same
  // 30 min/s floor, so a slow drag still stops and a flick does not spin the day.
  if (drag) release("tod", { wrap: 1440, maxAbs: 220, ignore: 30, tau: 0.34, stop: 8, on: setT });
  drag = null; ev.stopPropagation();
};
el.addEventListener("pointerup", stop); el.addEventListener("pointercancel", stop);
el.addEventListener("wheel", (ev) => { cancel("tod"); ev.preventDefault(); ev.stopPropagation(); setT(ui.todMin + (ev.deltaY > 0 ? 30 : -30)); }, { passive: false });
el.addEventListener("keydown", (ev) => {
  if (/Arrow(Left|Right)/.test(ev.key)) { cancel("tod"); ev.preventDefault(); ev.stopPropagation(); setT(ui.todMin + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 60 : 10)); }
});
addEventListener("keydown", (ev) => { if ((ev.key === "9" || ev.key === "0") && ev.target.tagName !== "INPUT") { cancel("tod"); ev.preventDefault(); setT(ui.todMin + (ev.key === "0" ? 30 : -30)); } });
const solar = ui.register({
  layer: "tod",
  draw(a) {
    const on = a > 0.5 && !ui.clean;
    if (!on && drag) { drag = null; cancel("tod"); }
    el.classList.toggle("on", on);
    drawDial();
    kb.upd();
  },
  hide() { el.classList.remove("on"); drag = null; cancel("tod"); },
});
window.__setTod = (m) => { cancel("tod"); setT(m); };
void solar;

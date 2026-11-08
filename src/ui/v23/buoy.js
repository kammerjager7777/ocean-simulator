// v30.23 buoy depth tool (key 2, spec §8.7 + GAD buoy notes Nov 1): one NOAA-style 3 m discus buoy drawn
// v30.27: scale 1.45 (hull about 4.4 m) so the 3 m discus reads smaller than the 8 m boat.
// It drags on the water only (clamped inside the block), heaves / pitches / rolls from 4 rim samples
// through a 2nd-order follower, flashes Fl(4) Y 20s on sim time, highlights on hover (Living Highlight halo and a
// 2x hull-radius ring on the water), and a click opens the §7.3 card (208 x 128, leader 28 px; shifts sideways off
// the HUD regions, then flips below). A drag leaves wake ripples;
// a fast drag (> 2 m/s) also lays a foam trail into the drawn foam field (gpuSurface foam-trail pass).
import { ui, $ } from "./core.js?v=30.29";
import { bedRelief } from "../../render/seabed.js?v=30.29";
import { rippleHeight, addRipple } from "../../physics/ripples.js?v=30.29";
import { elevAt } from "./sky.js?v=30.29";
const T = ui.theme, D2R = Math.PI / 180;
const SCALE = Number(ui.P.get("buoyx") || 1.45), MASTH = 4.8, HULL_R = 1.5; // mesh: 3 m hull, 4.8 m mast; displayed hull ~4.4 m
const pos = (() => { try { const j = JSON.parse(ui.store.get("ocean.ui.buoy", "")); if (j && Number.isFinite(j.x) && Number.isFinite(j.z)) return [j.x, j.z]; } catch { /* default */ } return [38, -26]; })();
const tgt = pos.slice(), vel = [0, 0];
const b = { s: SCALE, y: 0, vy: 0, pitch: 0, vp: 0, roll: 0, vr: 0, lastT: -1, ring: new Float32Array(7200), rt: new Float64Array(7200), n: 0, i: 0, Hs: NaN, Tz: NaN, Tp: NaN, shown: {}, nextStat: 0, open: false };
const trail = []; let lastWake = 0, lastDragT = 0, lastDragP = null, dragSpeed = 0;
const pts = new Float32Array(32);
// ---------- card (modern-ui-spec §7.3) ----------
const pop = document.createElement("div"); pop.className = "v23-card"; pop.setAttribute("role", "dialog"); pop.setAttribute("aria-label", "Buoy depth");
pop.innerHTML = `<div class="r r1"><span class="l2 ac">Depth</span><span class="l2 mu bl">Buoy</span><button class="x hit" type="button" aria-label="Close"><svg width="10" height="10"><path d="M0 0L10 10M10 0L0 10"/></svg></button></div>
<div class="r r2"><span class="n1 tx dv">—</span><span class="n2 mu"> m</span></div>
<div class="r r3"><span class="l2 mu">Shallow </span><span class="n3 tx lo">—</span><span class="l2 mu"> · Deep </span><span class="n3 tx hi">—</span><span class="l2 mu"> m</span></div>
<div class="r r4 l2 mu">Within 10 m</div><div class="div"></div><div class="r r5 wv"></div>`;
ui.root.appendChild(pop); pop.querySelector(".x").addEventListener("click", (e) => { e.stopPropagation(); b.open = false; });
pop.addEventListener("pointerdown", (e) => e.stopPropagation());
// HUD regions the card must not cover (settings column, pause cluster, toast); cached 500 ms, read only while the card is open
let hudR = [], hudT = -1e9;
function hudRegions() {
  if (ui.now - hudT < 500) return hudR; hudT = ui.now; hudR = [];
  for (const id of ["hb-set", "hud-title", "hud-rows", "hud-sheet", "hb-pause", "hud-pl", "hud-toast"]) { const el = document.getElementById(id); if (!el) continue; const r = el.getBoundingClientRect(); if (r.width && getComputedStyle(el).opacity !== "0") hudR.push([r.left - 8, r.top - 8, r.right + 8, r.bottom + 8]); }
  return hudR;
}
const hitsHud = (x, y, w, h) => hudRegions().some((r) => x < r[2] && x + w > r[0] && y < r[3] && y + h > r[1]);
const q = (s) => pop.querySelector(s);
// v30.24 (Michael item 5): kept on the block AND clear of the boat (the 8 m boat's swing radius + the buoy hull + 3 m)
const BOAT_CLEAR = 4.2 + 3;
function clampPos(p) {
  const S = ui.S(), m = S / 2 - (HULL_R * b.s + 2); let x = Math.max(-m, Math.min(m, p[0])), z = Math.max(-m, Math.min(m, p[1]));
  const bt = globalThis.__oceanState && globalThis.__oceanState.boat;
  if (bt) { const R = BOAT_CLEAR + HULL_R * b.s, dx = x - bt.x, dz = z - bt.z, d = Math.hypot(dx, dz); if (d < R) { const ux = d > 1e-3 ? dx / d : 0.6, uz = d > 1e-3 ? dz / d : -0.8; x = bt.x + ux * R; z = bt.z + uz * R; x = Math.max(-m, Math.min(m, x)); z = Math.max(-m, Math.min(m, z)); } }
  return [x, z];
}
let lastS = 0;
function depthAt(x, z) { return ui.D() - bedRelief(x, z, ui.S(), ui.D()); }
function stats() {
  const n = b.n; if (n < 32) return; const idx = (k) => (b.i - n + k + 7200) % 7200;
  const t0 = b.rt[idx(0)], t1 = b.rt[idx(n - 1)]; if (t1 - t0 < 30) { b.Hs = NaN; return; }
  let m = 0; for (let k = 0; k < n; k++) m += b.ring[idx(k)]; m /= n; let v = 0; for (let k = 0; k < n; k++) { const d = b.ring[idx(k)] - m; v += d * d; } v /= n;
  b.Hs = 4 * Math.sqrt(v);
  let up = 0, first = -1, last = -1; for (let k = 1; k < n; k++) { const a0 = b.ring[idx(k - 1)] - m, a1 = b.ring[idx(k)] - m; if (a0 < 0 && a1 >= 0) { const t = b.rt[idx(k)]; if (first < 0) first = t; last = t; up++; } }
  b.Tz = up > 1 ? (last - first) / (up - 1) : NaN;
  const fs = 4, M = Math.min(480, Math.floor((t1 - t0) * fs)), xs = new Float32Array(M); let k = 0;
  for (let j = 0; j < M; j++) { const t = t1 - (M - 1 - j) / fs; while (k < n - 1 && b.rt[idx(k + 1)] < t) k++; xs[j] = b.ring[idx(k)] - m; }
  let best = 0, bf = 0; for (let f = 0.03; f <= 0.5; f += 0.0025) { let re = 0, im = 0; const w = 2 * Math.PI * f / fs; for (let j = 0; j < M; j++) { re += xs[j] * Math.cos(w * j); im += xs[j] * Math.sin(w * j); } const p = re * re + im * im; if (p > best) { best = p; bf = f; } }
  b.Tp = bf ? 1 / bf : NaN; globalThis.__oceanBuoyStats = { Hs: b.Hs, Tz: b.Tz, Tp: b.Tp, n, span: t1 - t0 };
}
const resetBuffer = () => { b.n = 0; b.i = 0; b.Hs = NaN; b.shown = {}; };
function follower(y, v, target, dt) { const wn = 2 * Math.PI / 2.5, z = 0.4; const acc = wn * wn * (target - y) - 2 * z * wn * v; v += acc * dt; y += v * dt; return [y, v]; }
// v30.24: Fl Y 4s (Michael: a light that blinks like a real buoy so it is easy to find): 0.8 s on, 3.2 s dark
export const lightOn = (t) => { const p = ((t % 4) + 4) % 4; return p < 0.8; };
const matrixObj = { matrix() {
  const s = b.s, cp = Math.cos(b.pitch), sp = Math.sin(b.pitch), cr = Math.cos(b.roll), sr = Math.sin(b.roll);
  const r = [cp, sp, 0, -sp * cr, cp * cr, sr, sp * sr, -cp * sr, cr];
  return new Float32Array([r[0] * s, r[1] * s, r[2] * s, 0, r[3] * s, r[4] * s, r[5] * s, 0, r[6] * s, r[7] * s, r[8] * s, 0, pos[0], b.y, pos[1], 1]);
} };
const g = {};
const buoy = ui.register({
  layer: "buoy",
  update(a) {
    if (!(a > 0)) { ui.probe.pts = null; globalThis.__oceanBuoy = null; pop.classList.remove("on"); publishTrail(); return; }
    // follow the drag target (120 ms critically damped spring), on wall-clock time: the UI stays live when paused
    const w = 2 / 0.12, dt = ui.dt, ex = Math.exp(-w * dt); for (let k = 0; k < 2; k++) { const e0 = pos[k] - tgt[k], c1 = vel[k] + w * e0; pos[k] = tgt[k] + (e0 + c1 * dt) * ex; vel[k] = (c1 - w * (e0 + c1 * dt)) * ex; } // exact solution: stable at any frame time
    // v30.27: world-scaled like the boat, modestly under the 8 m hull (x1.45). It no longer grows when the
    // camera zooms out (the old 18 px floor did that), and it shrinks with the ocean Size exactly like the boat.
    // A Size change keeps its place as a FRACTION of the block (so it never ends up off the block), then clamps.
    b.s = SCALE; { const S = ui.S(); if (lastS && Math.abs(S - lastS) > 1e-6) { const k = S / lastS; for (const v of [pos, tgt]) { v[0] *= k; v[1] *= k; } vel[0] = vel[1] = 0; } lastS = S; }
    { const c = clampPos(tgt); tgt[0] = c[0]; tgt[1] = c[1]; const c2 = clampPos(pos); if (c2[0] !== pos[0] || c2[1] !== pos[1]) { pos[0] = c2[0]; pos[1] = c2[1]; } }
    const r = HULL_R * b.s; pts.fill(0); [[r, 0], [-r, 0], [0, r], [0, -r]].forEach(([dx, dz], k) => { pts[k * 2] = pos[0] + dx; pts[k * 2 + 1] = pos[1] + dz; });
    ui.probe.pts = pts;
    const pd = ui.probeData, simT = ui.simT();
    if (pd && Number.isFinite(pd[16 * 4]) && simT > b.lastT + 1e-6) {
      const h = [0, 1, 2, 3].map((k) => pd[(16 + k) * 4] + rippleHeight(pts[k * 2], pts[k * 2 + 1], simT));
      const heave = (h[0] + h[1] + h[2] + h[3]) / 4, cl = 25 * D2R;
      const pitch = Math.max(-cl, Math.min(cl, Math.atan((h[0] - h[1]) / (2 * r)))), roll = Math.max(-cl, Math.min(cl, Math.atan((h[2] - h[3]) / (2 * r))));
      const sdt = b.lastT < 0 ? 0 : Math.min(0.1, simT - b.lastT); b.lastT = simT;
      if (sdt > 0) { [b.y, b.vy] = follower(b.y, b.vy, heave, sdt); [b.pitch, b.vp] = follower(b.pitch, b.vp, pitch, sdt); [b.roll, b.vr] = follower(b.roll, b.vr, roll, sdt); } else { b.y = heave; }
      if (!ui.drag || ui.drag.obj !== buoy) { b.ring[b.i] = heave; b.rt[b.i] = simT; b.i = (b.i + 1) % 7200; b.n = Math.min(7200, b.n + 1); while (b.n > 2 && simT - b.rt[(b.i - b.n + 7200) % 7200] > 120) b.n--; }
    }
    if (ui.now > b.nextStat) { b.nextStat = ui.now + 2000; stats(); }
    globalThis.__oceanBuoy = matrixObj;
    publishTrail();
  },
  hit(x, y) { if (!g.box) return null; const [x0, y0, x1, y1] = g.box; return x > x0 && x < x1 && y > y0 && y < y1 ? "buoy" : null; },
  down() { lastDragP = null; lastDragT = 0; },
  move(h, x, y, e) {
    // every coalesced sample counts (a busy frame folds many moves into one event, which would hide a fast drag)
    const evs = e && e.getCoalescedEvents ? e.getCoalescedEvents() : []; const list = evs.length > 1 ? evs : [null];
    const r = evs.length > 1 ? e.target.getBoundingClientRect() : null; // coalesced samples carry client coords only
    for (const ev of list) {
      const p = ev ? ui.toPlane(ev.clientX - r.left, ev.clientY - r.top, 0) : ui.toPlane(x, y, 0); if (!p) continue; const c = clampPos([p[0], p[2]]); tgt[0] = c[0]; tgt[1] = c[1];
      const now = ((ev || e) && (ev || e).timeStamp ? (ev || e).timeStamp : performance.now()) / 1000; // event time, so a busy frame does not fake a slow drag
      if (lastDragP) { const dt = now - lastDragT; if (dt > 0) { const v = Math.hypot(c[0] - lastDragP[0], c[1] - lastDragP[1]) / dt; dragSpeed = dragSpeed ? dragSpeed + (v - dragSpeed) * Math.min(1, dt / 0.1) : v; } }
      lastDragP = c; lastDragT = now;
    }
  },
  up() { ui.store.set("ocean.ui.buoy", JSON.stringify({ x: Math.round(tgt[0] * 10) / 10, z: Math.round(tgt[1] * 10) / 10 })); resetBuffer(); dragSpeed = 0; },
  click() { b.open = !b.open; },
  escape() { b.open = false; },
  cursor: () => (ui.drag && ui.drag.obj === buoy ? "grabbing" : "grab"),
  draw(a) {
    const ctx = ui.ctx, s = b.s, simT = ui.simT();
    const base = ui.proj(pos[0], b.y, pos[1]), mast = ui.proj(pos[0], b.y + MASTH * s, pos[1]); if (!base || !mast) { g.box = null; pop.classList.remove("on"); return; }
    const rr = ui.proj(pos[0] + HULL_R * s, b.y, pos[1]), rz = ui.proj(pos[0], b.y, pos[1] + HULL_R * s);
    const rad = (p) => (p ? Math.hypot(p[0] - base[0], p[1] - base[1]) : 0);
    const hr = Math.max(rad(rr), rad(rz)); // true projected hull radius in px: world metres, no screen-size floor
    const hit = Math.max(hr, 12); // grab target only; the mesh itself is not enlarged
    let x0 = Math.min(base[0] - hit, mast[0]) - 8, x1 = Math.max(base[0] + hit, mast[0]) + 8, y0 = mast[1] - 8, y1 = base[1] + hit * 0.6 + 8;
    g.box = [x0, y0, x1, y1];
    // wake ripples (any drag) + the foam trail (fast drag)
    const dragging = ui.drag && ui.drag.obj === buoy && ui.drag.moved;
    if (dragging && performance.now() / 1000 - lastDragT > 0.15) dragSpeed *= Math.exp(-ui.dt / 0.15); // held still: the speed falls away
    if (dragging && ui.now - lastWake > 150) { const sp = Math.min(1, dragSpeed / 8); if (dragSpeed > 0.3) { addRipple(pos[0], pos[1], simT, 0.25 + 0.75 * sp); lastWake = ui.now; } }
    if (dragging && dragSpeed > 2) layTrail(simT);
    // v30.24 item 9: no constant glow; only a slight shine (and a thin ring on the water) while hovered or dragged
    const st = ui.state(buoy), held = st === "drag", hc = [(base[0] + mast[0]) / 2, (base[1] + mast[1]) / 2];
    if (st !== "idle") { T.shine(ctx, hc[0], hc[1], Math.max(4, hr * 1.6), held ? 0.2 : 0.13); const ring = []; for (let i = 0; i < 32; i++) { const an = (i / 32) * 2 * Math.PI; const p = ui.proj(pos[0] + 2 * HULL_R * s * Math.cos(an), 0, pos[1] + 2 * HULL_R * s * Math.sin(an)); if (p) ring.push(p); } T.line(ctx, ring, { color: T.white, width: 1, alpha: held ? 0.8 : 0.5, close: true }); }
    // v30.24: the navigation light, Fl Y 4s (one 0.8 s yellow flash every 4 s, wall clock so it also blinks while paused);
    // its halo grows from day to dusk to night so the buoy is easy to find
    const e = elevAt(ui.todMin ?? 660) / D2R; const k = e >= 10 ? 0 : e <= -6 ? 1 : (10 - e) / 16;
    const lamp = ui.proj(pos[0], b.y + 4.4 * s, pos[1]);
    // The halo follows the drawn mast, then stays inside a few pixels to a few dozen.
    // A fixed 22 px night halo sat on a tiny far buoy like a flood, and a close view
    // must not grow without a bound. The old night stroke under the hull is not drawn.
    const mastPx = Math.hypot(mast[0] - base[0], mast[1] - base[1]);
    const fit = Math.max(4, Math.min(28, mastPx * 0.42));
    const R = fit * (0.45 + 0.55 * k);
    const A = k < 0.5 ? 0.25 + (0.6 - 0.25) * (k / 0.5) : 0.6 + (0.9 - 0.6) * ((k - 0.5) / 0.5);
    const inView = lamp && lamp[0] > -80 && lamp[0] < ui.V.cw + 80 && lamp[1] > -80 && lamp[1] < ui.V.ch + 80;
    if (inView && lightOn(ui.now / 1000) && R > 0) {
      const gr = ctx.createRadialGradient(lamp[0], lamp[1], 0, lamp[0], lamp[1], R); gr.addColorStop(0, `rgba(255,226,122,${A})`); gr.addColorStop(1, "rgba(255,226,122,0)"); ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(lamp[0], lamp[1], R, 0, Math.PI * 2); ctx.fill();
      const dot = Math.max(1.2, Math.min(3.2, mastPx * 0.05));
      ctx.fillStyle = k > 0.5 ? "#FFE27A" : "#FFF2B0"; ctx.beginPath(); ctx.arc(lamp[0], lamp[1], dot, 0, Math.PI * 2); ctx.fill();
    }
    // pop-up
    pop.classList.toggle("on", b.open);
    if (b.open) {
      const d = depthAt(pos[0], pos[1]); let lo = Infinity, hi = -Infinity;
      for (let dx = -10; dx <= 10; dx += 2) for (let dz = -10; dz <= 10; dz += 2) { if (dx * dx + dz * dz > 100) continue; const v = depthAt(pos[0] + dx, pos[1] + dz); lo = Math.min(lo, v); hi = Math.max(hi, v); }
      q(".dv").textContent = d.toFixed(1); q(".lo").textContent = lo.toFixed(1); q(".hi").textContent = hi.toFixed(1);
      // Depth is immediate; no stale "warming" state. Wave estimates are only
      // printed after a real rolling history yields finite samples, never as zeros.
      const ready=Number.isFinite(b.Hs)&&Number.isFinite(b.Tz)&&Number.isFinite(b.Tp),wv=q(".wv");
      for(const kk of ["Hs","Tz","Tp"]){const v=b[kk],sv=b.shown[kk];if(Number.isFinite(v))b.shown[kk]=Number.isFinite(sv)?sv+(v-sv)*Math.min(1,ui.dt/.18):v;}
      wv.setAttribute("title",ready?"Two-minute rolling wave history":"Pause for immediate depth; resume to collect two-minute wave history for Hs, Tz, and Tp");
      const f=(n,d=1)=>Number.isFinite(b.shown[n])?b.shown[n].toFixed(d):"—";
      const html=`<span class="l2 mu nocaps">Hs</span> <span class="n3 tx">${f("Hs",2)}</span><span class="l2 mu nocaps"> m · Tz</span> <span class="n3 tx">${f("Tz")}</span><span class="l2 mu nocaps"> s · Tp</span> <span class="n3 tx">${f("Tp")}</span><span class="l2 mu nocaps"> s</span>`;
      if(wv._h!==html){wv._h=html;wv.innerHTML=html;}
      // placement: above (leader 28 px from a dot 6 px over the lantern); else shift sideways up to 120 px; else flip below
      const W = 208, H = 128, Wc = ui.V.cw, lt = lamp || mast, dot = [lt[0], lt[1] - 6]; let x = dot[0] - W / 2, y = dot[1] - 28 - H, flipped = false, shift = 0;
      const clampX = (v) => Math.max(24, Math.min(Wc - 24 - W, v)); x = clampX(x);
      if (y < 24 || hitsHud(x, y, W, H)) {
        let ok = false; if (y >= 24) for (const d of [20, -20, 40, -40, 60, -60, 80, -80, 100, -100, 120, -120]) { const xx = clampX(dot[0] - W / 2 + d); if (!hitsHud(xx, y, W, H)) { x = xx; shift = d; ok = true; break; } }
        if (!ok) { flipped = true; shift = 0; y = base[1] + hr * 0.6 + 28; x = clampX(base[0] - W / 2); }
      }
      pop.classList.toggle("below", flipped); pop.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`;
      if (flipped) { const hb = [base[0], base[1] + hr * 0.6]; T.line(ctx, [hb, [Math.min(x + W - 12, Math.max(x + 12, hb[0])), y]], { color: T.accent, width: 1, alpha: 0.9 }); }
      else { const jx = Math.min(x + W - 12, Math.max(x + 12, dot[0])); T.line(ctx, [dot, [jx, y + H]], { color: T.accent, width: 1, alpha: 0.9 }); T.circle(ctx, dot[0], dot[1], 1.8, { fill: T.accent }); }
      ui.debug.buoyPop = { x, y, flipped, shift, depth: d, lo, hi };
    }
    ui.debug.buoy = { c: [(g.box[0] + g.box[2]) / 2, (g.box[1] + g.box[3]) / 2], base, lamp, pos: pos.slice(), s: b.s, y: b.y, pitch: b.pitch / D2R, roll: b.roll / D2R, hullPx: 2 * hr, hullM: 2 * HULL_R * b.s, glowR: inView && lightOn(ui.now / 1000) ? R : 0, mastPx, dragSpeed, trail: trail.length, Hs: b.Hs };
  },
  hide() { pop.classList.remove("on"); },
});
// ---------- foam trail: points along the fast-drag path, read by the gpuSurface foam-trail pass ----------
function layTrail(t) {
  const hullD = 2 * HULL_R * b.s, last = trail[trail.length - 1];
  const sp = 0.35 * hullD, dist = last ? Math.hypot(pos[0] - last.x, pos[1] - last.z) : sp;
  if (dist < sp) return;
  const dir = last ? [pos[0] - last.x, pos[1] - last.z] : [1, 0]; const l = Math.hypot(...dir) || 1;
  // a slow frame moves the hull far: fill the gap at the fixed spacing (ages spread over the frame) so the trail has no holes
  const n = Math.min(24, Math.floor(dist / sp)), x0 = last ? last.x : pos[0], z0 = last ? last.z : pos[1], t0 = last ? last.t : t;
  for (let k = 1; k <= n; k++) { const f = last ? (k * sp) / dist : 1; trail.push({ x: x0 + (pos[0] - x0) * f, z: z0 + (pos[1] - z0) * f, t: t0 + (t - t0) * f, nx: -dir[1] / l, nz: dir[0] / l, w: hullD * (1.0 + 0.5 * Math.random()) }); }
  // keep at most 8 hull lengths of path
  let len = 0; for (let i = trail.length - 1; i > 0; i--) { len += Math.hypot(trail[i].x - trail[i - 1].x, trail[i].z - trail[i - 1].z); if (len > 8 * hullD) { trail.splice(0, i); break; } }
  if (trail.length > 64) trail.splice(0, trail.length - 64);
}
const trailBuf = new Float32Array(64 * 4), trailN = new Float32Array(64 * 2);
function publishTrail() {
  const t = ui.simT(); for (let i = trail.length - 1; i >= 0; i--) if (t - trail[i].t > 12) trail.splice(i, 1);
  if (!trail.length) { globalThis.__oceanFoamTrail = null; return; }
  trail.forEach((p, i) => { trailBuf[i * 4] = p.x; trailBuf[i * 4 + 1] = p.z; trailBuf[i * 4 + 2] = t - p.t; trailBuf[i * 4 + 3] = p.w; trailN[i * 2] = p.nx; trailN[i * 2 + 1] = p.nz; });
  globalThis.__oceanFoamTrail = { pts: trailBuf, nrm: trailN, n: trail.length };
}
ui.slider("Buoy position", () => [Math.round(pos[0]), `x ${pos[0].toFixed(0)} m, z ${pos[1].toFixed(0)} m`], (e) => { if (e.key === "Enter") { b.open = !b.open; return true; } const st = e.shiftKey ? 10 : 1, k = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, -st], ArrowDown: [0, st] }[e.key]; if (!k) return false; const c = clampPos([tgt[0] + k[0], tgt[1] + k[1]]); tgt[0] = c[0]; tgt[1] = c[1]; return true; });
ui.buoyApi = { pos, tgt, b, trail, open: (v) => { b.open = v; } };

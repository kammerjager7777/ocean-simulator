// v30.23 cross-section wall (key 3; behaviour sketch spec §8.10, look modern-ui-spec §5.7): a see-through vertical
// wall cut from the surface to the seabed between two Living Highlight knobs that slide along the block's top edges. The water column is
// filled, the wave profile (256 samples, async GPU readback, 1 frame late, no smoothing) runs along its
// top at true scale, the still-water line is dashed, the bed follows the seabed relief, and breaking
// spans (physics break flags only) are marked.
import { ui } from "./core.js?v=30.27";
import { bedY } from "../../render/seabed.js?v=30.27";
const T = ui.theme;
const sec = (() => { try { const j = JSON.parse(ui.store.get("ocean.ui.section", "")); if (Array.isArray(j) && j.length === 4 && j.every(Number.isFinite)) { if (j.some((v) => Math.abs(v) > 0.5001)) { const q = j.map((v) => v / 250); const a = toPerimeter(q[0], q[1]), b = toPerimeter(q[2], q[3]); return [...a, ...b]; } return j; } } catch { /* default */ } return [-0.4, 0.5, 0.5, -0.3]; })(); // fractions of S (resize-proof)
const save = () => ui.store.set("ocean.ui.section", JSON.stringify(sec.map((v) => Math.round(v * 1000) / 1000)));
function toPerimeter(x, z) { // nearest point on the square |x|,|z| = 0.5 (fractions)
  const cx = Math.max(-0.5, Math.min(0.5, x)), cz = Math.max(-0.5, Math.min(0.5, z));
  const d = [0.5 - cx, cx + 0.5, 0.5 - cz, cz + 0.5]; const m = Math.min(...d);
  return m === d[0] ? [0.5, cz] : m === d[1] ? [-0.5, cz] : m === d[2] ? [cx, 0.5] : [cx, -0.5];
}
const chip = ui.chip("sec-chip"), cur = ui.chip("sec-cur");
const g = {}; const prof = new Float32Array(256), flag = new Uint8Array(256); let have = false, down0 = null;
const wall = ui.register({
  layer: "section",
  update(a) { const S = ui.S(); ui.probe.sec = a > 0 ? [sec[0] * S, sec[1] * S, sec[2] * S, sec[3] * S] : null;
    const pd = ui.probeData; if (a > 0 && pd) { for (let i = 0; i < 256; i++) { const v = pd[20 * 4 + i]; const f = v > 500; prof[i] = f ? v - 1000 : v; flag[i] = f ? 1 : 0; } have = true; } },
  hit(x, y) {
    if (!g.a) return null;
    if (Math.hypot(x - g.a[0], y - g.a[1]) < ui.HIT) return "a"; if (Math.hypot(x - g.b[0], y - g.b[1]) < ui.HIT) return "b";
    if (g.poly && (pointIn([x, y], g.poly) || g.poly.some((q, i) => i % 4 === 0 && ui.segDist([x, y], q, g.poly[(i + 4) % g.poly.length]) < 6))) return "mid"; // fill or near its outline
    return null;
  },
  down(h, x, y) { const p = ui.toPlane(x, y); down0 = p ? { p, s: sec.slice() } : null; },
  move(h, x, y) {
    const S = ui.S(), p = ui.toPlane(x, y); if (!p) return;
    if (h === "a" || h === "b") { const q = toPerimeter(p[0] / S, p[2] / S); const o = h === "a" ? 0 : 2; sec[o] = q[0]; sec[o + 1] = q[1]; }
    else if (down0) { const dx = (p[0] - down0.p[0]) / S, dz = (p[2] - down0.p[2]) / S; const qa = toPerimeter(down0.s[0] + dx, down0.s[1] + dz), qb = toPerimeter(down0.s[2] + dx, down0.s[3] + dz); sec[0] = qa[0]; sec[1] = qa[1]; sec[2] = qb[0]; sec[3] = qb[1]; }
  },
  up() { save(); },
  draw(a) {
    const ctx = ui.ctx, S = ui.S(), D = ui.D();
    const x0 = sec[0] * S, z0 = sec[1] * S, x1 = sec[2] * S, z1 = sec[3] * S;
    const top = [], bed = [], still = [];
    for (let i = 0; i < 256; i += 2) { const t = i / 255, x = x0 + (x1 - x0) * t, z = z0 + (z1 - z0) * t;
      const pt = ui.proj(x, have ? prof[i] : 0, z), pb = ui.proj(x, bedY(x, z, S, D), z), ps = ui.proj(x, 0, z); if (!pt || !pb || !ps) { g.a = null; return; } top.push(pt); bed.push(pb); still.push(ps); }
    const st = ui.state(wall); const act = st !== "idle", la = act ? 1 : 0.35;
    const poly = [...top, ...bed.slice().reverse()]; g.poly = poly;
    // idle coverage <= 4% of the frame (§5.7): if the wall is bigger, idle draws only the profile and bed lines
    let ar = 0; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) ar += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]); ar = Math.abs(ar) / 2;
    const cov = ar / (ui.V.cw * ui.V.ch); if (act || cov <= 0.04) T.fill(ctx, poly, "#1F5566", act ? 0.7 : 0.45); // the slice's water (scene-coloured)
    T.line(ctx, still, { color: T.muted, width: 1, dash: [2, 4], alpha: 0.5 * la });
    T.line(ctx, top, { color: "#000814", width: 5, alpha: 0.6 * 0.35 * la }); T.line(ctx, top, { color: T.white, width: 1.5, alpha: 0.9 * la }); // profile + shadow halo
    T.line(ctx, bed, { color: T.white, width: 1, alpha: 0.6 * la });
    // breaking spans (physics flags only): 3 px accent along the crest plus a 6 px tick above
    for (let i = 0; i < 256; i += 2) if (flag[i]) { const j = i >> 1; if (top[j + 1]) { T.line(ctx, [top[j], top[j + 1]], { color: T.accent, width: 3, alpha: la }); if (!flag[i - 2]) T.line(ctx, [[top[j][0], top[j][1] - 4], [top[j][0], top[j][1] - 10]], { color: T.accent, width: 1.2, alpha: la }); } }
    g.a = ui.proj(x0, 0, z0); g.b = ui.proj(x1, 0, z1);
    for (const [p, h] of [[g.a, "a"], [g.b, "b"]]) if (p) T.knob(ctx, p[0], p[1], ui.state(wall, h) === "idle" ? st : ui.state(wall, h));
    let cr = -Infinity, tr = Infinity; for (let i = 0; i < 256; i++) { cr = Math.max(cr, prof[i]); tr = Math.min(tr, prof[i]); }
    const span = Math.hypot(x1 - x0, z1 - z0), sg = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
    const show = act && g.b && a > 0.5 && !ui.clean;
    if (show) {
      chip.set(`<span class="plate"><span class="row">${have ? `<span class="l2 tx">Crest </span><span class="n3 tx">${sg(cr)}</span><span class="l2 tx"> · Trough </span><span class="n3 tx">${sg(tr)}</span><span class="l2 tx"> m</span>` : '<span class="l2 tx">Waves reading…</span>'}</span><span class="row"><span class="l2 tx">Span </span><span class="n3 tx">${span.toFixed(0)}</span><span class="l2 tx"> m</span></span></span>`);
      const w = chip.w || 170, x = g.b[0] + 20 + w > ui.V.cw - 24 ? g.b[0] - 20 - w : g.b[0] + 20; chip.place(x, Math.max(24, Math.min(ui.V.ch - 24 - 32, g.b[1] - 14)));
    }
    chip.show(!!show);
    let curOn = false;
    if (st === "hover" && ui.mouse && have) { // vertical cursor + height at x
      const t = Math.min(1, Math.max(0, ui.segT(ui.mouse, still[0], still[still.length - 1]))); const j = Math.round(t * 127), i = Math.round(t * 255);
      if (top[j] && bed[j]) { T.line(ctx, [top[j], bed[j]], { color: T.white, width: 1 }); cur.set(`<span class="plate"><span class="n3 tx">${prof[i] >= 0 ? "+" : "−"}${Math.abs(prof[i]).toFixed(2)}</span><span class="l2 tx"> m</span></span>`).place(top[j][0] + 8, top[j][1] - 22); curOn = true; }
    }
    cur.show(curOn);
    ui.debug.section = { cov, a: g.a, b: g.b, mid: top[64], have, crest: cr, trough: tr, span, poly };
  },
  hide() { chip.show(false); cur.show(false); },
});
function pointIn(p, poly) { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]) c = !c; } return c; }

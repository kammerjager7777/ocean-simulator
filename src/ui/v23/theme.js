// v30.23 skin: the MODERN UI (GAD modern-ui-spec.md, approved by Michael). All colour, type and glow rules live
// here and in v23.css, so the skin stays one swappable module pair; behaviour modules only call these helpers.
//   accent(t)   §2.3: four held keys mixed in OKLab by the sun elevation (test vectors in the spec)
//   neutrals    §2.4: day/night tokens mixed in OKLab by the night weight n; vignette + text-plate alphas
//   applyTod    writes the CSS custom properties (only when a quantised value changes) and the vignette image
//   glow*/knob  §5.1 Living Highlight for canvas-drawn world marks: a soft accent glow copy, then the sharp ink copy
const D2R = Math.PI / 180, LAT = 20 * D2R;
export const elevDeg = (min) => Math.asin(Math.cos(LAT) * Math.cos(((min / 60 - 12) * 15) * D2R)) / D2R;
const s2l = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const l2s = (c) => { c = Math.max(0, Math.min(1, c)); return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)); };
const hx = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (c) => "#" + c.map((v) => v.toString(16).padStart(2, "0").toUpperCase()).join("");
function lab(h) {
  const [r, g, b] = hx(h).map(s2l);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b), m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b), s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function unlab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return toHex([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map(l2s));
}
export const mixOK = (A, B, w) => { if (w <= 0) return A.toUpperCase(); if (w >= 1) return B.toUpperCase(); const a = lab(A), b = lab(B); return unlab(a.map((v, i) => v + (b[i] - v) * w)); };
const ss = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
const DIFF = typeof location !== "undefined" && new URLSearchParams(location.search).get("ink") === "diff";
if (DIFF) document.documentElement.classList.add("ink-diff");
export const KEY = { night: "#DDE6FF", dawn: "#FFD36B", dusk: "#FF8A4C", noon: "#BFE3F7" };
// weights shared by the accent, the neutrals, and the scrim
export function todWeights(min) {
  const e = elevDeg(min), warm = min < 720 ? "dawn" : "dusk";
  const n = e <= -10 ? 1 : e < 0 ? 1 - ss((e + 10) / 10) : 0; // night weight
  const d = e <= 4 ? 0 : e < 35 ? ss((e - 4) / 31) : 1; // warm -> noon weight (above the horizon)
  return { e, warm, n, d };
}
export function accent(min) {
  const { e, warm, n, d } = todWeights(min); const W = KEY[warm];
  if (e <= -10) return KEY.night; if (e < 0) return mixOK(KEY.night, W, 1 - n); if (e <= 4) return W; if (e < 35) return mixOK(W, KEY.noon, d); return KEY.noon;
}
// v30.24 menu ink: no plates, no glow. The sky colour is only the fallback before the first probe.
// core.js then picks dark or light ink from the sea pixels behind the words (bright close-up water included).
// A 1 px hairline in the opposite ink is the only edge. ?ink=diff keeps the inversion look.
import { skyAt } from "./sky.js?v=30.27";
export const INK = { light: "#F5F7FA", dark: "#0C1724" };
const lumL = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
const lumH = (h) => lumL(hx(h).map(s2l));
export const contrast = (la, lb) => (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
const LUM_LIGHT = lumH(INK.light), LUM_DARK = lumH(INK.dark);
// Which ink reads on these backdrop luminances (linear, 0..1). The score is the worst sample, so a
// label that crosses a bright patch and a dark one takes the ink that still holds on both.
export function inkForLum(samples) {
  const ls = (Array.isArray(samples) ? samples : [samples]).filter((v) => Number.isFinite(v));
  if (!ls.length) return null;
  const score = (ink) => Math.min(...ls.map((L) => contrast(L, ink)));
  return score(LUM_DARK) > score(LUM_LIGHT) ? "dark" : "light";
}
export function backdropLum(min) { const k = skyAt(min).key; const c = k.hor.map((v, i) => v * 0.58 + k.zen[i] * 0.14); return lumL(c); }
const NEU = { text: ["#F5F7FA", "#E8EEF6"], muted: ["#C9D3DD", "#7F93AA"], card: ["#0D1B26", "#050B14"], scrim: ["#0A1622", "#03070C"] };
const rgb = (h) => hx(h).join(",");
export const theme = {
  name: "modern",
  white: "#FFFFFF", buoyHull: "#F2C230", lantern: "#FFF2B0", flash: "#FFE27A",
  accent: KEY.noon, text: NEU.text[0], muted: NEU.muted[0], card: NEU.card[0], scrim: NEU.scrim[0], night: 0,
  state: { idle: 0.35, hover: 0.7, drag: 1 },
  _last: {},
  // §2: everything time-keyed; CSS custom properties are written only when the quantised value changes
  applyTod(min) {
    const { n, d, e } = todWeights(min);
    const ac = accent(min), tx = mixOK(NEU.text[0], NEU.text[1], n), mu = mixOK(NEU.muted[0], NEU.muted[1], n), cd = mixOK(NEU.card[0], NEU.card[1], n), sc = mixOK(NEU.scrim[0], NEU.scrim[1], n);
    // vignette peak A and plate alpha: noon 58/65, warm 50/55, night 55/40 (noon<->warm by d above the horizon, warm<->night by n)
    const warmA = 0.5, warmP = 0.55, A = e >= 0 ? warmA + (0.58 - warmA) * d : warmA + (0.55 - warmA) * n, P = e >= 0 ? warmP + (0.65 - warmP) * d : warmP + (0.4 - warmP) * n;
    const bl = backdropLum(min), dark = contrast(bl, lumH(INK.dark)) > contrast(bl, lumH(INK.light)) && !DIFF;
    const ink = dark ? INK.dark : INK.light, halo = dark ? "#FFFFFF" : "#000814", inkAc = dark ? mixOK(ac, "#140C02", 0.72) : ac;
    Object.assign(this, { accent: ac, text: tx, muted: mu, card: cd, scrim: sc, night: n, plateA: P, vigA: A, ink, halo, inkAc, inkDark: dark, backLum: bl });
    const vars = { "--accent": ac, "--accent-rgb": rgb(ac), "--hud-text": ink, "--hud-muted": ink, "--hud-muted-rgb": rgb(ink), "--ink-ac": inkAc, "--halo-rgb": rgb(halo), "--card-rgb": rgb(cd), "--scrim-rgb": rgb(sc), "--plate-a": P.toFixed(2) };
    document.documentElement.classList.toggle("ink-dark", dark);
    const r = document.documentElement.style;
    for (const [k, v] of Object.entries(vars)) if (this._last[k] !== v) { this._last[k] = v; r.setProperty(k, v); }
    const vA = A * 0.35; const vk = sc + vA.toFixed(2); if (this._last.vig !== vk) { this._last.vig = vk; this.vignette(sc, vA); } // v30.24: the vignette only frames (it was a text scrim)
  },
  // §2.4 vignette: a static image (re-made only when its colour/alpha changes), composited by CSS: no per-frame pass
  vignette(col, A) {
    const el = document.getElementById("v23-vignette"); if (!el) return;
    const w = 256, h = 144, c = document.createElement("canvas"); c.width = w; c.height = h; const x = c.getContext("2d"); const im = x.createImageData(w, h); const [R, G, B] = hx(col);
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const dx = Math.abs((i + 0.5) - w / 2) / (w / 2), dy = Math.abs((j + 0.5) - h / 2) / (h / 2);
      const rr = Math.pow(Math.min(1, Math.max(0, (Math.max(0.85 * dx, dy) - 0.55) / 0.45)), 1.6), cc = Math.min(1, Math.max(0, (Math.hypot(dx, dy) - 0.7) / 0.55));
      const o = (j * w + i) * 4; im.data[o] = R; im.data[o + 1] = G; im.data[o + 2] = B; im.data[o + 3] = Math.round(255 * Math.max(rr, cc) * A);
    }
    x.putImageData(im, 0, 0); el.style.backgroundImage = `url(${c.toDataURL()})`;
  },
  // ---------- canvas marks (§4, §5.1) ----------
  rgba(h, a) { const [r, g, b] = hx(h); return `rgba(${r},${g},${b},${a})`; },
  path(ctx, pts, close) { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); if (close) ctx.closePath(); },
  // soft glow copy of a polyline: stacked wide strokes approximate the σ 7 px falloff (cheap, no blur filter)
  glowLine(ctx, pts, width, alpha, color = this.accent) {
    if (!pts || pts.length < 2 || alpha <= 0) return; ctx.save(); ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = color;
    for (const [k, a] of [[3.2, 0.18], [2.0, 0.32], [1.0, 0.5]]) { ctx.globalAlpha = alpha * a; ctx.lineWidth = width * k; this.path(ctx, pts); ctx.stroke(); }
    ctx.restore();
  },
  line(ctx, pts, { color = this.white, width = 1, alpha = 1, dash = null, close = false } = {}) {
    if (!pts || pts.length < 2 || alpha <= 0) return; ctx.save(); ctx.globalAlpha *= alpha; ctx.lineCap = "round"; ctx.lineJoin = "round"; if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = color; ctx.lineWidth = width; this.path(ctx, pts, close); ctx.stroke(); ctx.restore();
  },
  fill(ctx, pts, color, alpha = 1) { if (!pts || pts.length < 3) return; ctx.save(); ctx.globalAlpha *= alpha; ctx.fillStyle = color; this.path(ctx, pts, true); ctx.fill(); ctx.restore(); },
  glowDot(ctx, x, y, r, alpha, color = this.accent) { // radial falloff, σ ≈ 7 px beyond r
    if (alpha <= 0) return; const R = r + 14; const g = ctx.createRadialGradient(x, y, 0, x, y, R); const c = this.rgba(color, 1), z = this.rgba(color, 0);
    g.addColorStop(0, c); g.addColorStop(Math.min(0.95, r / R), this.rgba(color, 0.6)); g.addColorStop(1, z);
    ctx.save(); ctx.globalAlpha *= alpha; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
  },
  circle(ctx, x, y, r, { fill = null, fillA = 1, stroke = null, width = 1.2, strokeA = 1 } = {}) {
    ctx.save(); ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    if (fill) { ctx.globalAlpha = fillA; ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.globalAlpha = strokeA; ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); }
    ctx.restore();
  },
  // v30.24 item 9 (Michael: no glow): marks are flat ink; the only light is a very slight shine under the cursor on hover / drag
  shine(ctx, x, y, r, alpha) { if (alpha <= 0) return; const g = ctx.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, `rgba(255,255,255,${alpha})`); g.addColorStop(1, "rgba(255,255,255,0)"); ctx.save(); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.restore(); },
  // knobs (size, chop, depth, clarity, cross-section): st "idle" | "hover" | "drag"; ink = the ruler ink
  knob(ctx, x, y, st = "idle", pulse = 0, ink = this.ink) {
    const T = this, halo = ink === INK.dark ? "#FFFFFF" : "#000814";
    if (st !== "idle") T.shine(ctx, x, y, 18, st === "drag" ? 0.10 : 0.05);
    T.circle(ctx, x, y, st === "drag" ? 7.5 : st === "hover" ? 6.5 : 5.5, { fill: halo, fillA: 0.25 });
    T.circle(ctx, x, y, st === "drag" ? 6 : st === "hover" ? 5.2 : 4.4, { fill: st === "drag" ? T.inkAc : ink, fillA: 1 });
  },
  // a ruler: the full measured length, 1 px ink, end ticks across it (len px), a faint halo copy under it
  track(ctx, pts, st = "idle", ink = this.ink, ticks = 6) {
    const halo = ink === INK.dark ? "#FFFFFF" : "#000814", al = st === "drag" ? 0.95 : st === "hover" ? 0.85 : 0.6;
    this.line(ctx, pts, { color: halo, width: 3, alpha: 0.12 }); this.line(ctx, pts, { color: ink, width: 1, alpha: al });
    const [a, b] = pts, dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1, nx = -dy / l, ny = dx / l;
    for (const p of [a, b]) this.line(ctx, [[p[0] - nx * ticks, p[1] - ny * ticks], [p[0] + nx * ticks, p[1] + ny * ticks]], { color: ink, width: 1, alpha: al });
  },
};

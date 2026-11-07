// v30.25 sky & time (sketch-ui-spec §7, §8.6). The time model and the colour keys; publishes, every frame,
// globalThis.__oceanSky (dome + reflection + glitter keys), __oceanSun (the light: sun by day, the full
// moon opposite it by night) and __oceanDay (the v30.22b sea tint). 11:00 is the approved look: identity.
const LAT = (20 * Math.PI) / 180, D2R = Math.PI / 180;
export const elevAt = (min) => Math.asin(Math.cos(LAT) * Math.cos(((min / 60 - 12) * 15) * D2R)); // equinox (radians)
export const azAt = (min) => { const h = ((min / 60 - 12) * 15) * D2R; return Math.atan2(Math.sin(h), Math.cos(h) * Math.sin(LAT)); };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const lin = (c) => c.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
// GAD §7 keys: zenith, horizon, disc, glitter tint, glitter intensity
const K = {
  dawn: { zen: "#2F4673", hor: "#F4B183", disc: "#FFD49A", glit: "#FFC27A", gi: 0.65 },
  noon: { zen: "#3F78B5", hor: "#C8DDEB", disc: "#FFF7E8", glit: "#FFF4DC", gi: 1.0 },
  dusk: { zen: "#283766", hor: "#F08A4B", disc: "#FFA24A", glit: "#FF9E5A", gi: 0.75 },
  night: { zen: "#060B18", hor: "#1A2740", disc: "#F2EEDF", glit: "#CFDAEA", gi: 0.08 },
};
for (const k of Object.values(K)) for (const c of ["zen", "hor", "disc", "glit"]) k[c] = lin(hex(k[c]));
const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
// v30.22b sea tint, kept (approved): identity at or above the 11:00 elevation
function dayTint(m) {
  const e = elevAt(m) / D2R, e0 = elevAt(660) / D2R;
  if (Math.abs(m - 660) < 0.5 || e >= e0) return null;
  // Twilight and the day curve are the approved sunrise / sunset brightness. Deep night used to sit on a
  // 0.10 floor of the tonemapped colour, which crushed a full-moon sea to about RGB 3/6/7. The moon
  // is still a light: hold a 0.55 floor instead of 0.10, and let it meet the existing
  // curve by -3°, so every elevation above -3° — the whole warm sunrise and sunset — is unchanged.
  const b0 = 0.1 + 0.9 * (0.5 * sm(-8, 4, e) + 0.5 * sm(2, e0, e));
  const b = Math.max(b0, 0.55 * (1 - sm(-12, -3, e)));
  const n = 1 - sm(-7, 6, e);
  // v30.23 (GAD 17:50 hue 110° olive): this tint multiplies the WHOLE lit colour, water body included, so it must stay
  // (near-)neutral: brightness only, with a slight cool shift at night. The warm dawn/dusk colour now arrives only
  // through the sky reflection (uSkyHorM / uSkyZenM, keyed by skyAt) and the glitter (uGlitM).
  return [b * (1 - 0.08 * n), b * (1 - 0.03 * n), b * (1 + 0.06 * n)];
}
export function skyAt(m) {
  const e = elevAt(m) / D2R, morning = m < 720;
  const tw = morning ? K.dawn : K.dusk;
  let key = {};
  const blend = (a, b, t) => { for (const c of ["zen", "hor", "disc", "glit"]) key[c] = mix(a[c], b[c], t); key.gi = a.gi + (b.gi - a.gi) * t; };
  if (e <= -4) blend(K.night, tw, sm(-12, -4, e)); else if (e <= 4) blend(tw, tw, 0); else blend(tw, K.noon, sm(4, 35, e));
  const night = elevAt(m) < 0; // one shared crossing for path, dial, dome and light
  key.disc = night ? K.night.disc : key.disc;
  if (night) { key.glit = K.night.glit; key.gi = K.night.gi; }
  return { e, key, night, morning };
}
// v30.24 (Michael, item 2): the sun / moon is a real object in the background sky, dragged along a set path that spans
// the WHOLE sky. Default (Michael's pick): FIXED IN THE WORLD. The arc runs from the east horizon (sunrise, 06:00) over
// the south to the west horizon (sunset, 18:00) at the sun's true azimuth, so it swings round when the camera orbits,
// like the real sun; the moon rides the same arc 12 h apart. The orbit camera looks down ~30 deg, so the true sky is
// above the frame: the backdrop shows the sky as a level wide-angle view, azimuth relative to the camera across the
// width (+-90 deg = the edges, world-fixed), elevation up the height from a horizon at 0.66 H. Facing south, the whole
// east-to-west arc spans the screen; facing east, sunrise is mid-screen and the arc climbs away behind you.
// ?sunarc=screen (fallback): the screen-wide arc, left edge to right edge whatever the camera azimuth.
// Either way the dome draws the disc on the view ray through the arc point; the light takes the world-fixed azimuth
// (world) or that ray's azimuth (screen).
export const SUNARC = typeof location !== "undefined" && new URLSearchParams(location.search).get("sunarc") === "screen" ? "screen" : "world"; // v30.24: Michael picked world (default)
const EMAX = elevAt(720);

export const arcH = (m) => { const h = m / 60; return elevAt(m) >= 0 ? h : h < 12 ? h + 12 : h - 12; }; // hours on the day arc 6..18
export const isSunUp = (m) => elevAt(m) >= 0;
export function arcGeo(W, H) { return { x0: 0.035 * W, x1: 0.965 * W, yb: 0.66 * H, yt: 0.1 * H }; }
export function arcScreen(h, W, H) { const g = arcGeo(W, H), u = (h - 6) / 12; return [g.x0 + (g.x1 - g.x0) * u, g.yb - (g.yb - g.yt) * Math.max(0, Math.sin(elevAt(h * 60)) / Math.sin(EMAX))]; }
// world arc: screen px of arc hour h (6..18), or null when its azimuth is more than 100 deg off the camera's heading
export function worldArcScreen(m, W, H, V, VI) {
 const d=trueDir(m); if(!d||!VI)return null;
 const f=rayDir(W/2,H/2,V,VI), r=rayDir(W*.75,H/2,V,VI), af=Math.atan2(f[2],f[0]);
 const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x)),sgn=Math.sign(wrap(Math.atan2(r[2],r[0])-af))||1;
 const rel=sgn*wrap(Math.atan2(d[2],d[0])-af);if(Math.abs(rel)>Math.PI*.58)return null;
 const g=arcGeo(W,H),hw=(g.x1-g.x0)/2,yn=elevAt(m)/EMAX;
 // A full 24 h ellipse: midnight is bottom; noon top. Its x coordinate follows
 // azimuth, not the old hour-folded day path, so there is no midnight azimuth jump.
 const horizon=elevAt(m);
 return [W/2+(rel/(Math.PI/2))*hw,g.yb-(g.yb-g.yt)*Math.abs(yn)];
}
export function rayDir(x, y, V, VI) { // screen px -> unit world view-ray direction
  const nx = (x / V.cw) * 2 - 1, ny = 1 - (y / V.ch) * 2, m = VI;
  const un = (z) => { const w = m[3] * nx + m[7] * ny + m[11] * z + m[15]; return [(m[0] * nx + m[4] * ny + m[8] * z + m[12]) / w, (m[1] * nx + m[5] * ny + m[9] * z + m[13]) / w, (m[2] * nx + m[6] * ny + m[10] * z + m[14]) / w]; };
  const a = un(-1), b = un(1), d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], l = Math.hypot(...d) || 1; return [d[0] / l, d[1] / l, d[2] / l];
}
// warm (sunrise / sunset) weight: full from the horizon to 6 deg, gone by 24 deg; fades out into the night by -8 deg
export const warmW = (e) => (e >= 6 ? 1 - sm(6, 24, e) : e >= -1 ? 1 : sm(-8, -1, e));
let V0 = null;
// World azimuth of the sun at clock min. Noon is the block's -Z face normal, not the
// default orbit heading. view.js yaw 0.75 looks at the near corner (about 43 degrees
// off -Z), so tying noon to that heading put the crest over the diagonal: you had to
// turn onto the corner to center it. yaw 0 looks along -Z and now centers the crest.
// The hour sweep is unchanged, so sunrise stays on the same side of that axis.
// The ?sunarc=screen fallback keeps the v30.23 screen-relative azimuths.
const A_NOON = -Math.PI / 2;
// Continuous equinox equatorial sweep. The old hour-angle azimuth flipped
// as it crossed the south pole at midnight, so the background dome had a seam.
const azW = (min) => (SUNARC === "world" ? A_NOON - (min-720) * Math.PI / 720 : V0 ? Math.atan2(V0[2], V0[0]) - ((min-660)*Math.PI/720) : 0);
export function trueDir(min) { if (!V0) return null; const e = elevAt(min), a = azW(min); return [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)]; }
export function publish(m, enabled = true, VI = null) {
  if (!enabled) m = 660;
  const { e, key, night } = skyAt(m);
  const tint = dayTint(m); globalThis.__oceanDay = tint;
  const t3 = tint || [1, 1, 1];
  const noon = K.noon;
  const refl = (c, n) => c.map((v, i) => Math.min(3, v / n[i] / t3[i]));
  const ident = Math.abs(m - 660) < 0.5;
  const glit = key.glit.map((v, i) => Math.min(4, (v / noon.glit[i]) * key.gi / t3[i]));
  if (!V0 && globalThis.__oceanView) V0 = globalThis.__oceanView.sun;
  let L = null, Ldir = null; const V = globalThis.__oceanView, scr = SUNARC === "screen" && V && VI && V.cw > 0;
  if (V0) {
    const l = Math.hypot(...V0), e0 = Math.asin(V0[1] / l), a0 = Math.atan2(V0[2], V0[0]);
    const lm = night ? m + 720 : m; // full moon opposite the sun
    // the dome draws the true sky position (the light keeps the art-directed lower elevation)
    const de = elevAt(m), da = azW(m), me = elevAt(m + 720), ma = azW(m + 720);
    Ldir = { sun: [Math.cos(de) * Math.cos(da), Math.sin(de), Math.cos(de) * Math.sin(da)], moon: [Math.cos(me) * Math.cos(ma), Math.sin(me), Math.cos(me) * Math.sin(ma)], sunOn: de > -0.02 ? 1 : 0, moonOn: me > -0.02 ? 1 : 0 };
    let a = azW(lm); // the light keeps the arc azimuth (11:00 included: the visible sun lights the sea)
    if (scr) { // the disc sits on the screen arc: the dome draws it on that pixel's view ray, the light takes its azimuth
      const p = arcScreen(arcH(m), V.cw, V.ch), d = rayDir(p[0], p[1], V, VI); a = Math.atan2(d[2], d[0]);
      if (isSunUp(m)) Ldir.sun = d; else Ldir.moon = d; Ldir.screen = p;
    } else if (V && VI && V.cw > 0) {
      const sp=worldArcScreen(m,V.cw,V.ch,V,VI),mp=worldArcScreen((m+720)%1440,V.cw,V.ch,V,VI);
      if(sp)Ldir.sun=rayDir(...sp,V,VI);
      if(mp)Ldir.moon=rayDir(...mp,V,VI);
      // Exactly one rendered disc, on the same full-day path for each body.
      // It must never flip on before the shared geometric horizon crossing.
      const sunUp=elevAt(m)>=0;Ldir.sunOn=sunUp&&sp?1:0;Ldir.moonOn=!sunUp&&mp?1:0;
      Ldir.screen=(sunUp?sp:mp)||null;
    }
    const el = Math.max(0.02, e0 * elevAt(lm) / elevAt(660));
    L = [Math.cos(el) * Math.cos(a) * l, Math.sin(el) * l, Math.cos(el) * Math.sin(a) * l];
  }
  globalThis.__oceanSun = L;
  globalThis.__oceanSky = {
    t: m, e, night, zen: key.zen, hor: key.hor, disc: key.disc,
    reflHor: ident ? [1, 1, 1] : refl(key.hor, noon.hor), reflZen: ident ? [1, 1, 1] : refl(key.zen, noon.zen),
    glit: ident ? [1, 1, 1] : glit, dirs: Ldir, stars: 1 - sm(-14, -6, e), moonDisc: K.night.disc,
    // v30.24 item 11: the warm light reaching the water. sunC: the light colour relative to noon (diffuse, foam, glitter);
    // warm: the horizon sky colour relative to noon and its weight, which the sea mixes into its body and reflection.
    sunC: ident ? [1, 1, 1] : night ? [0.75, 0.82, 1.0] : key.disc.map((v, i) => Math.min(1.6, v / noon.disc[i])),
    warm: [...key.hor.map((v, i) => Math.min(3, v / noon.hor[i])), night ? 0 : warmW(e)],
    arc: { noonAz: A_NOON, axis: "-Z", worldFixed: SUNARC !== "screen" },
  };
  return e;
}

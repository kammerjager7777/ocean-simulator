// v30.4 boat: a generic ~8 m fishing boat floating on the sea (true scale, a size reference).
// Real buoyancy: the sea height is sampled at 12 hull points (GPU probe, read back
// asynchronously, ~1-3 frames old). Each point is a column of the waterplane: Archimedes force
// rho g A_i d_i (d_i = immersion, capped at the hull depth) plus a heave damping on its vertical
// speed, so the hull heaves, pitches and rolls. Wind windage and a weak current drift it, and a
// soft mooring spring keeps it near the centre of the sea.
import { GRAVITY } from "./gravity.js";
import { clampToWater } from "./island.js";
const BOAT_DRIFT = typeof location !== "undefined" && new URLSearchParams(location.search).get("boatdrift") === "1";
const RHO = 1025, MASS = 6500, LEN = 8.0, BEAM = 2.8, HULL_D = 1.1;
const IX = MASS * (BEAM * BEAM + HULL_D * HULL_D) / 12 * 1.4; // roll
const IZ = MASS * (LEN * LEN) / 12 * 0.9;                     // pitch
const IY = MASS * (LEN * LEN + BEAM * BEAM) / 12;             // yaw
// body frame: x along the boat (bow +), z across (starboard +), y up (keel at y = -0.8)
const PTS = [];
for (const fx of [-0.42, -0.14, 0.14, 0.40]) for (const fz of [-0.42, 0, 0.42]) {
  const narrow = fx > 0.3 ? 0.55 : 1; // the bow narrows
  PTS.push([fx * LEN, -0.8, fz * BEAM * narrow]);
}
const AREA = (LEN * BEAM * 0.78) / PTS.length;
// Just off the beach, on a 250 m slice (the wall is at 125 m). The old pin
// (158, 25) sat in the void. place() pulls this inside the current sea.
const HOME_X = 76, HOME_Z = 12;
// Still-water depth the boat is parked in when the island is on.
export const BANK_DEPTH = 2.2;
export function createBoat() {
  const b = { x: HOME_X, y: -0.2, z: HOME_Z, vx: 0, vy: 0, vz: 0, yaw: 0.6, pitch: 0, roll: 0, wy: 0, wp: 0, wr: 0, length: LEN, homeX: HOME_X, homeZ: HOME_Z };
  // Stay on the slice: half the sea, minus 15 m so an 8 m hull does not hang
  // off the wall. Then off the sandbar, unless that water is already outside
  // the slice (a very small sea is mostly island).
  b.place = (size, depth, island) => {
    const half = Math.max(Number(size) || 250, 40) / 2;
    const maxR = Math.max(24, half - 15);
    // On the bank's outer edge on a 250 m sea, about (82, 13) in 2.2 m of
    // water: the 1.1 m hull floats clear and the sand under it still reads
    // tan, so its shadow has sand to fall on. 3 m was already the drop-off,
    // where the depth colour is deep blue. Other sizes keep that relationship
    // instead of pinning a 250 m home against the wall.
    const s = (Number(size) || 250) / 250;
    let x = HOME_X * s, z = HOME_Z * s;
    const pr = Math.hypot(x, z) || 1;
    if (pr > maxR) { x *= maxR / pr; z *= maxR / pr; }
    if (island) {
      const w = clampToWater(x, z, depth > 0 ? depth : 40, BANK_DEPTH);
      if (Math.hypot(w[0], w[1]) <= maxR + 0.5) { x = w[0]; z = w[1]; }
    }
    b.homeX = x; b.homeZ = z;
    if (!BOAT_DRIFT) { b.x = x; b.z = z; b.vx = 0; b.vz = 0; }
    else {
      const r = Math.hypot(b.x, b.z);
      if (r > maxR && r > 1e-3) { b.x *= maxR / r; b.z *= maxR / r; }
      if (island) {
        const hull = clampToWater(b.x, b.z, depth > 0 ? depth : 40, 2.0);
        if (Math.hypot(hull[0], hull[1]) <= maxR + 0.5) { b.x = hull[0]; b.z = hull[1]; }
      }
    }
  };
  function world(p) {
    const [px, py, pz] = p;
    const cp = Math.cos(b.pitch), sp = Math.sin(b.pitch), cr = Math.cos(b.roll), sr = Math.sin(b.roll), cy = Math.cos(b.yaw), sy = Math.sin(b.yaw);
    // roll about x, pitch about z, then yaw about y
    let x = px, y = py * cr - pz * sr, z = py * sr + pz * cr;
    const x2 = x * cp - y * sp, y2 = x * sp + y * cp; x = x2; y = y2;
    return [b.x + x * cy + z * sy, b.y + y, b.z - x * sy + z * cy];
  }
  b.probePoints = () => { const a = new Float32Array(32); PTS.forEach((p, i) => { const w = world(p); a[i * 2] = w[0]; a[i * 2 + 1] = w[2]; }); return a; };
  b.step = (dt, heights, wind) => {
    if (!(dt > 0)) return;
    dt = Math.min(dt, 0.05);
    let fy = -MASS * GRAVITY, tp = 0, tr = 0;
    if (heights) PTS.forEach((p, i) => {
      const w = world(p); const h = heights[i * 4];
      const d = Math.min(Math.max(h - w[1], 0), HULL_D * 1.6);
      // vertical speed of this hull point (heave + pitch/roll rates, small angles)
      const vp = b.vy + b.wp * p[0] - b.wr * p[2];
      const f = RHO * GRAVITY * AREA * d - (d > 0 ? 2600 * vp : 0);
      fy += f; tp += f * p[0] * Math.cos(b.pitch); tr -= f * p[2] * Math.cos(b.roll);
    });
    else fy = 0;
    b.vy += (fy / MASS) * dt; b.vy *= Math.exp(-0.3 * dt);
    b.wp += (tp / IZ - 0.8 * b.wp) * dt; b.wr += (tr / IX - 0.6 * b.wr) * dt;
    b.y += b.vy * dt; b.pitch += b.wp * dt; b.roll += b.wr * dt;
    b.pitch = Math.max(-0.6, Math.min(0.6, b.pitch)); b.roll = Math.max(-0.8, Math.min(0.8, b.roll));
    // drift: windage ~2.5% of the wind + 0.1 m/s current, soft mooring to the centre, water drag
    const wdx = Math.cos(wind.direction), wdz = Math.sin(wind.direction);
    const ax = 0.012 * wind.speed * wdx + 0.05 - 0.004 * (b.x - b.homeX) - 0.6 * b.vx;
    const az = 0.012 * wind.speed * wdz - 0.004 * (b.z - b.homeZ) - 0.6 * b.vz;
    b.vx += ax * dt; b.vz += az * dt; b.x += b.vx * dt; b.z += b.vz * dt;
    // Pinned just off the beach unless ?boatdrift=1 (heave, pitch, roll, yaw still free).
    // place() owns the pin, including the slice clamp, so a size change moves it.
    // weathervanes slowly bow-to-wind-ish
    const tgt = Math.atan2(-wdz, wdx) + Math.PI; let dy = tgt - b.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    b.wy += (0.02 * dy - 0.5 * b.wy) * dt; b.yaw += b.wy * dt;
  };
  b.matrix = () => { // column-major model matrix
    const o = world([0, 0, 0]), ex = world([1, 0, 0]), ey = world([0, 1, 0]), ez = world([0, 0, 1]);
    const X = ex.map((v, i) => v - o[i]), Y = ey.map((v, i) => v - o[i]), Z = ez.map((v, i) => v - o[i]);
    return new Float32Array([...X, 0, ...Y, 0, ...Z, 0, ...o, 1]);
  };
  return b;
}

// Heightmap island. Ground height is meters relative to sea level (positive is
// land). The sandbar is centered on the slice origin. On a 250 m sea the dry
// sand is about 38% of the slice width, and the bank keeps falling until it
// meets the open floor inside the map. A border of that open floor is stored
// with shadow 1, so the texture edge is the same water as outside it.
// 512x512 over 320 m is 0.625 m cells, finer than the shoreline mesh.
// Ocean size does not scale this grid; a smaller sea crops it and a larger
// one shows more open floor.

const N = 512;
const MAP_SPAN = 400;
const CX = 0;
const CZ = 0;
const BASE_SIZE = 250;
// A 1000 m sea must not grow an 9 m dune. Below the cap, heights scale with
// the radii, so the beach slope stays 0.08. Above it the slope only gets gentler.
const PEAK_CAP = 4;

// The 250 m island, scaled by L/250. Radii scale all the way, so the bank
// stays the same fraction inside the wall (about 3 m of open floor at 250 m,
// more on a larger sea). Heights scale until PEAK_CAP, which is why a 100 m
// sea gets a low sandbar that a developed wave can flood, and a 1000 m sea
// gets a wider island whose crown stops at 4 m.
export function islandAt(size) {
  const L = Math.max(40, Number(size) || BASE_SIZE);
  const s = L / BASE_SIZE;
  const hScale = Math.min(s, PEAK_CAP / 2.2);
  const slopeK = hScale / s;
  const mapSpan = MAP_SPAN * s;
  const slope = 0.08 * slopeK;
  // From just offshore of the waterline to inland of a storm run-up.
  const cross = 3.2 / Math.max(slope, 0.02);
  const rOut = 48 * s + cross * 0.62;
  const sweSpan = Math.max(96, 2 * rOut);
  const DX = 0.35;
  let sweN = Math.ceil(sweSpan / DX);
  sweN = Math.max(160, Math.min(1600, sweN));
  const sweDx = sweSpan / sweN;
  return {
    id: "sandbar",
    size: L,
    scale: s,
    hScale,
    cx: CX,
    cz: CZ,
    // Beach band only, at about 0.35 m, including when the sea is 1000 m.
    // The old 360² grid stretched to 3.6 m cells and drew the film as stairs.
    span: sweSpan,
    nx: sweN,
    ny: sweN,
    sweDx,
    mapN: N,
    mapSpan,
    mapOriginX: CX - mapSpan / 2,
    mapOriginZ: CZ - mapSpan / 2,
    Rbeach: 48 * s,
    Rberm: 30.5 * s,
    yBerm: 1.4 * hScale,
    faceSlope: 0.08 * slopeK,
    Rface: 66.75 * s,
    yFace: -1.5 * hScale,
    shelfSlope: 0.03 * slopeK,
    // Shorter reef flat than the old 82 m shelf, so the outer drop starts sooner.
    Rshelf: 74 * s,
    yShelf: -1.7175 * hScale,
    Rbank: 122 * s,
    peak: 2.2 * hScale,
    shelf: -2 * hScale,
    dLo: 2,
    dHi: 6,
    shoreRefine: 82 * s,
    ease: 8 * s,
    bedFloor: -200,
  };
}

// Stockdon et al. 2006, the 2% run-up. Vertical metres above still water.
export function stockdonR2(hs, tp, slope) {
  const H = Math.max(0.05, Number(hs) || 0.05);
  const T = Math.max(2, Number(tp) || 5);
  const beta = Math.max(0.02, Math.min(0.2, Number(slope) || 0.08));
  const L0 = 9.81 * T * T / (2 * Math.PI);
  const hl = Math.sqrt(H * L0);
  const setup = 0.35 * beta * hl;
  const swash = 0.5 * Math.sqrt(H * L0 * (0.563 * beta * beta + 0.004));
  return 1.1 * (setup + swash);
}

// Frozen 250 m profile. Live drawing uses currentIsland(), which follows the sea size.
export const ISLAND = Object.freeze(islandAt(BASE_SIZE));
let live = ISLAND;

export function useIslandSize(size) {
  const L = Math.max(40, Number(size) || BASE_SIZE);
  if (live.size !== L) live = islandAt(L);
  return live;
}

export function currentIsland() {
  return live;
}

function hermite(s, y0, y1, m0, m1) {
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
}

// The round sandbar the heightmap is filled with. Same curve the shaders used
// before the grid, so a bilinear sample stays on that surface.
// Keep a hull in water. If the point is on land or shallower than minDepth,
// slide it out from the island centre to the nearest deep-enough contour.
export function clampToWater(x, z, depth, minDepth = 1.2, I = currentIsland()) {
  if (sandbarBed(x, z, depth, I) <= -minDepth) return [x, z];
  let dx = x - I.cx;
  let dz = z - I.cz;
  let len = Math.hypot(dx, dz);
  if (len < 1e-3) { dx = 1; dz = 0; len = 1; }
  const ux = dx / len;
  const uz = dz / len;
  let lo = len;
  let hi = Math.max(len + 1, I.Rbank);
  for (let i = 0; i < 18; i += 1) {
    const mid = (lo + hi) * 0.5;
    const bed = sandbarBed(I.cx + ux * mid, I.cz + uz * mid, depth, I);
    if (bed > -minDepth) lo = mid;
    else hi = mid;
  }
  return [I.cx + ux * hi, I.cz + uz * hi];
}

// Still-water depth at a point. Positive offshore, 0 on dry sand. The buoy,
// the section and the handles use this when the island is on, instead of the
// open-ocean relief, so they agree with the clamp.
export function stillDepth(x, z, depth, I = currentIsland()) {
  return Math.max(0, -sandbarBed(x, z, depth, I));
}

export function sandbarBed(x, z, depth, I = currentIsland()) {
  const dx = x - I.cx, dz = z - I.cz;
  const r0 = Math.hypot(dx, dz);
  const th = Math.atan2(dz, dx);
  // Lumpy outline. The same angles are in both islandBed shaders.
  const lump = 0.055 * Math.sin(3 * th) + 0.032 * Math.sin(5 * th + 1.7) + 0.018 * Math.sin(8 * th + 0.4);
  const r = r0 / Math.max(1 + lump, 0.75);
  const yFloor = -Math.max(depth, 0.5);
  if (r >= I.Rbank) return yFloor;
  const Rb = I.Rberm;
  const Rf = I.Rface;
  const Rs = I.Rshelf;
  const ease = I.ease > 0 ? I.ease : 8;
  const join0 = Rf - ease;
  const join1 = Rf + ease;
  let y;
  if (r <= Rb) {
    // Flat crown into the berm. Slope is 0 at the centre and the face slope
    // where the beach starts, so the join is not a kink.
    y = hermite(r / Rb, I.peak, I.yBerm, 0, -I.faceSlope * Rb);
  } else if (r <= join0) {
    y = I.yBerm - I.faceSlope * (r - Rb);
  } else if (r < join1) {
    // Ease 0.08 down to 0.03 so the face does not meet the shelf in a corner.
    const y0 = I.yBerm - I.faceSlope * (join0 - Rb);
    const y1 = I.yFace - I.shelfSlope * (join1 - Rf);
    const span = join1 - join0;
    y = hermite((r - join0) / span, y0, y1, -I.faceSlope * span, -I.shelfSlope * span);
  } else if (r <= Rs) {
    y = I.yFace - I.shelfSlope * (r - Rf);
  } else {
    const span = Math.max(I.Rbank - Rs, 1);
    const s = (r - Rs) / span;
    y = hermite(s, I.yShelf, yFloor, -I.shelfSlope * span, 0);
  }
  // Berm crest and low crown dunes on the dry sand. Spurs and grooves on the
  // outer drop, so it is not a circular cliff. Heights follow the peak scale.
  const scH = I.peak;
  y += 0.16 * scH * Math.exp(-Math.pow((r - Rb) / 4, 2));
  if (r < Rb) y += 0.1 * scH * Math.sin(0.42 * r + 2 * th) * Math.sin(0.28 * r - 3 * th);
  const spanB = Math.max(I.Rbank - Rs, 1);
  const uB = Math.min(1, Math.max(0, (r - Rs) / spanB));
  y += 0.55 * scH * Math.sin(7 * th + 0.8) * Math.sin(Math.PI * uB);
  if (y < yFloor) return yFloor;
  return y;
}

export function coastWeight(depthStill, I = currentIsland()) {
  if (depthStill >= I.dHi) return 0;
  if (depthStill <= I.dLo) return 1;
  const t = (depthStill - I.dLo) / (I.dHi - I.dLo);
  const s = t * t * (3 - 2 * t);
  return 1 - s;
}

function roundSandbar(depth, I = currentIsland()) {
  const n = I.mapN;
  const h = new Float32Array(n * n);
  const ox = I.mapOriginX;
  const oz = I.mapOriginZ;
  const cell = I.mapSpan / n;
  for (let j = 0; j < n; j += 1) {
    for (let i = 0; i < n; i += 1) {
      const x = ox + (i + 0.5) * cell;
      const z = oz + (j + 0.5) * cell;
      let b = sandbarBed(x, z, depth, I);
      // Open-ocean border. Clamp-to-edge then repeats the same depth and a
      // fully lit texel, so the map boundary is not a line in the water.
      if (i < 2 || j < 2 || i >= n - 2 || j >= n - 2) b = -depth;
      h[j * n + i] = b;
    }
  }
  return h;
}

const PRESETS = {
  sandbar: roundSandbar,
};

export function islandById(id, depth, I = currentIsland()) {
  const make = PRESETS[id] || PRESETS.sandbar;
  const heights = make(depth, I);
  return {
    v: 1,
    id: PRESETS[id] ? id : "sandbar",
    meters: I.mapSpan,
    origin: [I.mapOriginX, I.mapOriginZ],
    nx: I.mapN,
    ny: I.mapN,
    heights,
  };
}

// Versioned document. height is little-endian float32, base64. Not for the URL.
export function serializeIsland(island) {
  const bytes = new Uint8Array(island.heights.buffer, island.heights.byteOffset, island.heights.byteLength);
  let raw = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) raw += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
  return {
    v: 1,
    id: island.id,
    meters: island.meters,
    origin: [island.origin[0], island.origin[1]],
    nx: island.nx,
    ny: island.ny,
    height: btoa(raw),
  };
}

export function deserializeIsland(doc) {
  if (!doc || doc.v !== 1 || !doc.height) throw new Error("island document is not v1");
  const bin = atob(doc.height);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const heights = new Float32Array(bytes.buffer);
  if (heights.length !== doc.nx * doc.ny) throw new Error("island heightmap size does not match");
  return {
    v: 1,
    id: doc.id || "sandbar",
    meters: doc.meters,
    origin: [doc.origin[0], doc.origin[1]],
    nx: doc.nx,
    ny: doc.ny,
    heights,
  };
}

export function sampleIsland(island, x, z, depth) {
  const [ox, oz] = island.origin;
  const fx = ((x - ox) / island.meters) * island.nx - 0.5;
  const fz = ((z - oz) / island.meters) * island.ny - 0.5;
  if (fx < -0.5 || fz < -0.5 || fx > island.nx - 0.5 || fz > island.ny - 0.5) return -depth;
  const x0 = Math.max(0, Math.min(island.nx - 1, Math.floor(fx)));
  const z0 = Math.max(0, Math.min(island.ny - 1, Math.floor(fz)));
  const x1 = Math.min(island.nx - 1, x0 + 1);
  const z1 = Math.min(island.ny - 1, z0 + 1);
  const tx = Math.min(1, Math.max(0, fx - x0));
  const tz = Math.min(1, Math.max(0, fz - z0));
  const h00 = island.heights[z0 * island.nx + x0];
  const h10 = island.heights[z0 * island.nx + x1];
  const h01 = island.heights[z1 * island.nx + x0];
  const h11 = island.heights[z1 * island.nx + x1];
  const h = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  return h < -190 ? -depth : h;
}

// 0 = blocked, 1 = lit. A few rays so the edge is soft. Rebaked when the sun moves.
export function bakeShadow(heights, nx, ny, span, sun, depth = 40) {
  const out = new Float32Array(nx * ny);
  const cell = span / nx;
  const sx0 = sun[0];
  const sy = sun[1];
  const sz0 = sun[2];
  const horiz = Math.hypot(sx0, sz0);
  // No light above the horizon: leave the map lit. A flat night fill on the
  // land was a dark disc, and a fill of the whole grid was a dark square.
  // The caller passes the moon once the sun has set, so the dome still shades.
  if (!(sy > 0.03) || horiz < 1e-4) {
    out.fill(1);
    return out;
  }
  const base = Math.atan2(sz0, sx0);
  const rise = sy / horiz;
  const steps = 24;
  for (let j = 0; j < ny; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const h0 = heights[j * nx + i];
      let blocked = 0;
      for (let k = -1; k <= 1; k += 1) {
        const ang = base + k * 0.12;
        const dx = Math.cos(ang);
        const dz = Math.sin(ang);
        for (let s = 1; s <= steps; s += 1) {
          const dist = s * cell * 1.6;
          const ix = Math.round(i + (dx * dist) / cell);
          const iy = Math.round(j + (dz * dist) / cell);
          if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) break;
          if (heights[iy * nx + ix] > h0 + dist * rise + 0.2) {
            blocked += 1;
            break;
          }
        }
      }
      let lit = 1 - blocked / 3;
      // The open floor is not in the island's shadow. A hard cutoff at the
      // stored map was the dark square around the bank.
      if (h0 <= -depth + 0.75) lit = 1;
      const u = (i + 0.5) / nx;
      const v = (j + 0.5) / ny;
      const edge = Math.min(u, 1 - u, v, 1 - v) / 0.05;
      const f = Math.min(1, Math.max(0, edge));
      out[j * nx + i] = lit * f + (1 - f);
    }
  }
  return out;
}

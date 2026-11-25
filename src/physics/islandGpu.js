// Island shallow-water pass on the ocean's WebGL2 context.
// Deep cells are replaced by the live spectral surface. The blend band
// (still depth 2 m to 6 m) mixes the two, which is where the shallow-water
// speed and the dispersive speed nearly agree. Shallow cells integrate the
// shallow-water equations. The spectral elevation is not added on again.

import { GRAVITY } from "./gravity.js";
import { ISLAND, islandById, sampleIsland, bakeShadow, useIslandSize, stockdonR2 } from "./island.js?v=32.85";

const VERT = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D st;
uniform sampler2D ocean;
uniform float uDt, uDx, uDy, uG, uEps, uClip;
uniform vec2 uOrigin;
uniform vec2 uWind;
uniform float uL;
uniform float uDepth;
uniform vec2 uMapOrigin;
uniform float uMapSpan;
uniform float uHm;
uniform float uAtlasH;
uniform float uAtlasW;
uniform float uSweY;
uniform float uR2;
uniform float uDLo, uDHi;
uniform float uOmega;
uniform int uN;
uniform float uK;
out vec4 o;

float bedAt(vec2 p) {
  vec2 uv = (p - uMapOrigin) / uMapSpan;
  if (uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0) return -uDepth;
  float x = (clamp(uv.x, 0.0, 1.0) * (uHm - 1.0) + 0.5) / uAtlasW;
  float y = (clamp(uv.y, 0.0, 1.0) * (uHm - 1.0) + 0.5) / uAtlasH;
  float h = texture(st, vec2(x, y)).r;
  return h < -190.0 ? -uDepth : h;
}

float sweW(float D) {
  float t = clamp((D - uDLo) / (uDHi - uDLo), 0.0, 1.0);
  float s = t * t * (3.0 - 2.0 * t);
  return 1.0 - s;
}

struct Cell { float h; float b; float u; float v; };

Cell loadAt(int i, int j) {
  Cell c;
  i = clamp(i, 0, uN - 1);
  j = clamp(j, 0, uN - 1);
  vec4 s = texelFetch(st, ivec2(i, j + int(uSweY)), 0);
  vec2 world = uOrigin + (vec2(i, j) + 0.5) * vec2(uDx, uDy);
  c.b = bedAt(world);
  c.h = max(s.r, 0.0);
  // Desingularize. A raw hu/h on a wet-dry cell is huge and blows the front apart.
  float denom = c.h * c.h + max(c.h, 0.04) * max(c.h, 0.04);
  c.u = 2.0 * c.h * s.g / denom;
  c.v = 2.0 * c.h * s.b / denom;
  return c;
}

float minmod(float a, float b) {
  if (a * b <= 0.0) return 0.0;
  return abs(a) < abs(b) ? a : b;
}

// Second order in a wet cell, first order on a wet-dry face so the front stays stable.
float recEdge(float vm, float v, float vp, float h, float side) {
  if (h < 0.05) return v;
  return v + side * 0.5 * minmod(v - vm, vp - v);
}

// Kurganov–Petrova central-upwind flux on the hydrostatically reconstructed states.
// At rest (u = 0 and equal reconstructed depths) the mass flux is 0 and the
// pressure flux is 1/2 g h*^2, which the bed source cancels.
void kp(float hL, float uL, float vL, float hR, float uR, float vR,
        out float fh, out float fn, out float ft) {
  float cL = sqrt(uG * hL);
  float cR = sqrt(uG * hR);
  float ap = max(max(uL + cL, uR + cR), 0.0);
  float am = min(min(uL - cL, uR - cR), 0.0);
  float span = ap - am;
  float qL = hL * uL;
  float qR = hR * uR;
  if (span < 1e-5) {
    fh = 0.0;
    fn = 0.5 * uG * hL * hL;
    ft = 0.0;
    return;
  }
  float pL = qL * uL + 0.5 * uG * hL * hL;
  float pR = qR * uR + 0.5 * uG * hR * hR;
  fh = (ap * qL - am * qR + ap * am * (hR - hL)) / span;
  fn = (ap * pL - am * pR + ap * am * (qR - qL)) / span;
  ft = (ap * qL * vL - am * qR * vR + ap * am * (hR * vR - hL * vL)) / span;
}

void face(Cell L, Cell R, float etaL, float etaR, float uL, float vL, float uR, float vR,
          out float fh, out float fn, out float ft, out float hKeep, bool keepLeft) {
  float bs = max(L.b, R.b);
  float hl = max(0.0, etaL - bs);
  float hr = max(0.0, etaR - bs);
  float unL = hl > 0.01 ? uL : 0.0;
  float unR = hr > 0.01 ? uR : 0.0;
  float utL = hl > 0.01 ? vL : 0.0;
  float utR = hr > 0.01 ? vR : 0.0;
  kp(hl, unL, utL, hr, unR, utR, fh, fn, ft);
  hKeep = keepLeft ? hl : hr;
}

Cell specAt(vec2 world) {
  Cell c;
  c.b = bedAt(world);
  vec2 uv = fract(world / uL + 0.5);
  vec4 oc = texture(ocean, uv);
  float etaS = oc.r;
  float hS = max(0.0, etaS - c.b);
  float Ds = max(0.0, -c.b);
  vec2 inland = length(world) > 2.0 ? -normalize(world) : uWind;
  float windward = dot(length(world) > 2.0 ? normalize(world) : uWind, -uWind);
  float faceW = mix(0.18, 1.0, smoothstep(-0.25, 0.55, windward));
  float kh = max(uOmega, 0.35) * sqrt(max(Ds, 0.3) / uG);
  float uAmp = clamp(etaS * max(uOmega, 0.35) / max(tanh(max(kh, 0.12)), 0.15), -1.6, 1.6);
  c.h = hS;
  c.u = inland.x * uAmp * faceW;
  c.v = inland.y * uAmp * faceW;
  return c;
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.x, gl_FragCoord.y - uSweY);
  vec2 world = uOrigin + vec2(p) * vec2(uDx, uDy);
  float b = bedAt(world);
  float D = max(0.0, -b);
  float w = sweW(D);
  vec2 uv = fract(world / uL + 0.5);
  vec4 oc = texture(ocean, uv);
  float etaS = oc.r;
  float hS = max(0.0, etaS - b);
  // The live FFT elevation already carries the groups. The period sets the
  // orbital speed, so one crest is one uprush and the trough is the backwash.
  // Onshore is toward the island. The lee face only gets a fraction of it.
  vec2 inland = length(world) > 2.0 ? -normalize(world) : uWind;
  float windward = dot(length(world) > 2.0 ? normalize(world) : uWind, -uWind);
  float faceW = mix(0.18, 1.0, smoothstep(-0.25, 0.55, windward));
  float kh = max(uOmega, 0.35) * sqrt(max(D, 0.3) / uG);
  float uAmp = clamp(etaS * max(uOmega, 0.35) / max(tanh(max(kh, 0.12)), 0.15), -1.6, 1.6);
  vec2 vel = inland * uAmp * faceW;
  vec4 spec = vec4(hS, hS * vel.x, hS * vel.y, 0.0);
  if (w < 0.02) { o = spec; return; }
  Cell C = loadAt(p.x, p.y);
  Cell L = loadAt(p.x - 1, p.y);
  Cell R = loadAt(p.x + 1, p.y);
  Cell LL = loadAt(p.x - 2, p.y);
  Cell RR = loadAt(p.x + 2, p.y);
  Cell B = loadAt(p.x, p.y - 1);
  Cell T = loadAt(p.x, p.y + 1);
  Cell BB = loadAt(p.x, p.y - 2);
  Cell TT = loadAt(p.x, p.y + 2);
  // Open edge. A mirrored velocity sent the wave back onto the shelf.
  // The outside cell is the spectral sea, so what leaves does not return.
  if (p.x == 0) { LL = L; L = specAt(world - vec2(uDx, 0.0)); }
  if (p.x == uN - 1) { RR = R; R = specAt(world + vec2(uDx, 0.0)); }
  if (p.y == 0) { BB = B; B = specAt(world - vec2(0.0, uDy)); }
  if (p.y == uN - 1) { TT = T; T = specAt(world + vec2(0.0, uDy)); }
  float eLL = LL.h + LL.b, eL = L.h + L.b, eC = C.h + C.b, eR = R.h + R.b, eRR = RR.h + RR.b;
  float eBB = BB.h + BB.b, eB = B.h + B.b, eT = T.h + T.b, eTT = TT.h + TT.b;
  float fhR, fnR, ftR, hRkeep, fhL, fnL, ftL, hLkeep;
  float ghT, gnT, gtT, hTkeep, ghB, gnB, gtB, hBkeep;
  face(C, R,
    recEdge(eL, eC, eR, C.h, 1.0), recEdge(eC, eR, eRR, R.h, -1.0),
    recEdge(L.u, C.u, R.u, C.h, 1.0), recEdge(L.v, C.v, R.v, C.h, 1.0),
    recEdge(C.u, R.u, RR.u, R.h, -1.0), recEdge(C.v, R.v, RR.v, R.h, -1.0),
    fhR, fnR, ftR, hRkeep, true);
  face(L, C,
    recEdge(eLL, eL, eC, L.h, 1.0), recEdge(eL, eC, eR, C.h, -1.0),
    recEdge(LL.u, L.u, C.u, L.h, 1.0), recEdge(LL.v, L.v, C.v, L.h, 1.0),
    recEdge(L.u, C.u, R.u, C.h, -1.0), recEdge(L.v, C.v, R.v, C.h, -1.0),
    fhL, fnL, ftL, hLkeep, false);
  face(C, T,
    recEdge(eB, eC, eT, C.h, 1.0), recEdge(eC, eT, eTT, T.h, -1.0),
    recEdge(B.v, C.v, T.v, C.h, 1.0), recEdge(B.u, C.u, T.u, C.h, 1.0),
    recEdge(C.v, T.v, TT.v, T.h, -1.0), recEdge(C.u, T.u, TT.u, T.h, -1.0),
    ghT, gnT, gtT, hTkeep, true);
  face(B, C,
    recEdge(eBB, eB, eC, B.h, 1.0), recEdge(eB, eC, eT, C.h, -1.0),
    recEdge(BB.v, B.v, C.v, B.h, 1.0), recEdge(BB.u, B.u, C.u, B.h, 1.0),
    recEdge(B.v, C.v, T.v, C.h, -1.0), recEdge(B.u, C.u, T.u, C.h, -1.0),
    ghB, gnB, gtB, hBkeep, false);
  float sx = 0.5 * uG * (hRkeep * hRkeep - hLkeep * hLkeep) / uDx;
  float sy = 0.5 * uG * (hTkeep * hTkeep - hBkeep * hBkeep) / uDy;
  vec4 cur = texelFetch(st, ivec2(p.x, p.y + int(uSweY)), 0);
  float hn = cur.r - uDt * ((fhR - fhL) / uDx + (ghT - ghB) / uDy);
  float hun = cur.g - uDt * ((fnR - fnL) / uDx + (gtT - gtB) / uDy - sx);
  float hvn = cur.b - uDt * ((ftR - ftL) / uDx + (gnT - gnB) / uDy - sy);
  if (hn < 0.0 && hn >= uClip) hn = 0.0;
  if (hn < uEps) { hun = 0.0; hvn = 0.0; }
  // Manning friction on the thin film, so the edge stops and drains
  // instead of sloshing. Deeper water keeps the flux as integrated.
  // Only the very thin sheet. A 0.3 m bore must not be rubbed out on the face.
  if (hn > uEps && hn < 0.12) {
    float spd = length(vec2(hun, hvn)) / max(hn, 0.02);
    float drag = uG * 0.0004 * spd / pow(max(hn, 0.03), 1.333);
    float damp = 1.0 / (1.0 + uDt * drag);
    hun *= damp;
    hvn *= damp;
  }
  // Wet sand keeps percolating, including once the cell reads saturated.
  // Above the Stockdon cap there is no water at all.
  if (b > uR2) {
    hn = 0.0;
    hun = 0.0;
    hvn = 0.0;
  } else if (b > 0.0 && hn > 0.0) {
    float perc = uK * (0.4 + 0.6 * (1.0 - cur.a));
    if (b > 0.25) perc += 0.06;
    hn = max(0.0, hn - perc * uDt);
  }
  if (hn < uEps) { hun = 0.0; hvn = 0.0; }
  // Rusanov damps a wave over the long shelf. Keep the water deeper than
  // about a metre on the spectral sea (capped by the breaking depth), and
  // leave the last metre plus the dry beach to the solver so it can run up
  // and drain. Deeper than the blend the snap above already is the FFT.
  // Hand the spectral wave in above 1 m. Cap the crest, not the whole column:
  // min(h, 0.78 D) sat below still water and drained the shelf.
  if (D > 1.0) {
    // Just outside the swash the shelf has to follow each crest and trough,
    // or the beach never sees a group arrive and leave. Farther out the
    // slower blend stays.
    float tau = D < 2.4 ? 0.22 : (D > uDHi ? 0.35 : 1.2);
    float a = 1.0 - exp(-uDt / tau);
    float crest = min(max(etaS, -0.9 * D), 0.45 * max(D, 0.3));
    float hCap = max(0.0, crest - b);
    hn = mix(hn, hCap, a);
    hun = mix(hun, hCap * vel.x, a);
    hvn = mix(hvn, hCap * vel.y, a);
  }
  float wet = cur.a;
  if (hn > 0.012) wet = min(1.0, wet + uDt * 1.4);
  else wet *= exp(-uDt / 3.0);
  if (b > uR2) wet *= exp(-uDt / 1.5);
  vec4 swe = vec4(hn, hun, hvn, wet);
  o = mix(spec, swe, w);
  o.a = wet;
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) || "island shader");
  return sh;
}

const HM = ISLAND.mapN;

export function createIslandGpu(gl, depthIn, sizeIn) {
  let profile = useIslandSize(sizeIn || 250);
  let nx = profile.nx;
  let ny = profile.ny;
  let atlasW = Math.max(HM, nx);
  let atlasH = HM + ny;
  let dx = profile.span / nx;
  let dy = profile.span / ny;
  let originX = profile.cx - profile.span / 2;
  let originZ = profile.cz - profile.span / 2;
  let lastR2 = 0.6;
  let bedDepth = depthIn;
  let doc = islandById(profile.id, bedDepth, profile);
  let shadow = bakeShadow(doc.heights, doc.nx, doc.ny, doc.meters, [0.4, 0.8, 0.2], bedDepth);
  let init = new Float32Array(atlasW * atlasH * 4);
  let stillShallow = 0;
  let stillRegion = 0;
  let stillEnergy = 0;
  function fillInit() {
    init.fill(0);
    stillShallow = 0;
    stillRegion = 0;
    stillEnergy = 0;
    for (let j = 0; j < HM; j += 1) {
      for (let i = 0; i < HM; i += 1) {
        const o = (j * atlasW + i) * 4;
        init[o] = doc.heights[j * HM + i];
        init[o + 1] = shadow[j * HM + i];
      }
    }
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const x = originX + (i + 0.5) * dx;
        const z = originZ + (j + 0.5) * dy;
        const b = sampleIsland(doc, x, z, bedDepth);
        const h = Math.max(0, -b);
        init[((HM + j) * atlasW + i) * 4] = h;
        const D = Math.max(0, -b);
        if (D > 0 && D < profile.dLo) stillShallow += h;
        if (D < profile.dHi) {
          stillRegion += h;
          stillEnergy += 0.5 * GRAVITY * h * h + GRAVITY * h * b;
        }
      }
    }
    stillShallow *= dx * dy;
    stillRegion *= dx * dy;
    stillEnergy *= dx * dy;
  }
  fillInit();
  const tex = (data) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, atlasW, atlasH, 0, gl.RGBA, gl.FLOAT, data);
    // The solver reads this with texelFetch, which ignores the filter.
    // The sea mesh samples it with texture(), and nearest made the beach
    // a ring of 1.25 m terraces.
    // Linear float filtering is optional. Without the extension the atlas is
    // incomplete and a vertex fetch of it is undefined (the torn sheet).
    const filter = gl.getExtension("OES_texture_float_linear") ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  const state = [tex(init), tex(init)];
  const fbo = state.map((t) => {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return fb;
  });
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) || "island link");
  const loc = {};
  for (const name of ["st", "ocean", "uDt", "uDx", "uDy", "uG", "uEps", "uClip", "uOrigin", "uWind", "uL", "uDepth", "uMapOrigin", "uMapSpan", "uHm", "uAtlasH", "uAtlasW", "uSweY", "uDLo", "uDHi", "uOmega", "uR2", "uN", "uK"]) {
    loc[name] = gl.getUniformLocation(prog, name);
  }
  let cur = 0;
  const scratch = gl.createFramebuffer();

  function guard(fn) {
    const fbo0 = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const vp = gl.getParameter(gl.VIEWPORT);
    const prog0 = gl.getParameter(gl.CURRENT_PROGRAM);
    try { return fn(); }
    finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo0);
      gl.viewport(vp[0], vp[1], vp[2], vp[3]);
      gl.useProgram(prog0);
      gl.activeTexture(gl.TEXTURE0);
    }
  }

  function step(dt, oceanTex, L, windX, windZ, windSpeed, hs, tp) {
    const period = Math.max(2, Number(tp) || 5);
    lastR2 = stockdonR2(hs || 0.8, period, profile.faceSlope);
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - cur]);
      gl.viewport(0, HM, nx, ny);
      gl.useProgram(prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, state[cur]);
      gl.uniform1i(loc.st, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, oceanTex);
      gl.uniform1i(loc.ocean, 1);
      gl.uniform1f(loc.uDt, dt);
      gl.uniform1f(loc.uDx, dx);
      gl.uniform1f(loc.uDy, dy);
      gl.uniform1f(loc.uG, GRAVITY);
      gl.uniform1f(loc.uEps, 1e-4);
      gl.uniform1f(loc.uClip, -1e-4);
      gl.uniform2f(loc.uOrigin, originX, originZ);
      gl.uniform2f(loc.uWind, windX, windZ);
      gl.uniform1f(loc.uL, L);
      gl.uniform1f(loc.uDepth, bedDepth);
      gl.uniform2f(loc.uMapOrigin, profile.mapOriginX, profile.mapOriginZ);
      gl.uniform1f(loc.uMapSpan, profile.mapSpan);
      gl.uniform1f(loc.uHm, HM);
      gl.uniform1f(loc.uAtlasH, atlasH);
      gl.uniform1f(loc.uAtlasW, atlasW);
      gl.uniform1f(loc.uSweY, HM);
      gl.uniform1f(loc.uDLo, profile.dLo);
      gl.uniform1f(loc.uDHi, profile.dHi);
      gl.uniform1f(loc.uOmega, 6.2831853 / period);
      gl.uniform1f(loc.uR2, lastR2);
      gl.uniform1i(loc.uN, nx);
      gl.uniform1f(loc.uK, 0.012);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      cur = 1 - cur;
    });
  }

  function read() {
    const data = new Float32Array(nx * ny * 4);
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[cur]);
      gl.readPixels(0, HM, nx, ny, gl.RGBA, gl.FLOAT, data);
    });
    return data;
  }

  function updateShadow(sun) {
    shadow = bakeShadow(doc.heights, doc.nx, doc.ny, doc.meters, sun, bedDepth);
    const row = new Float32Array(HM * HM * 4);
    for (let i = 0; i < HM * HM; i += 1) {
      row[i * 4] = doc.heights[i];
      row[i * 4 + 1] = shadow[i];
    }
    guard(() => {
      for (const t of state) {
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, HM, HM, gl.RGBA, gl.FLOAT, row);
      }
    });
  }

  function measures(oceanTex, L, oceanN, windX, windZ) {
    const data = read();
    let shallow = 0;
    let region = 0;
    let energy = 0;
    let minH = 0;
    let expE = -1e9;
    let shE = -1e9;
    let seam = 0;
    let seamN = 0;
    let wetMax = 0;
    let runup = 0;
    let runX = 0;
    let runZ = 0;
    // Bin 0 is downwind (lee). Bin 4 is the shore the waves run into.
    const azInland = Array(8).fill(-1e9);
    const azBed = Array(8).fill(-1e9);
    const wlen = Math.hypot(windX || 0, windZ || 1) || 1;
    const wx = (windX || 0) / wlen;
    const wz = (windZ || 1) / wlen;
    const expX = profile.cx - wx * 16;
    const expZ = profile.cz - wz * 16;
    const shX = profile.cx + wx * 16;
    const shZ = profile.cz + wz * 16;
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, scratch);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oceanTex, 0);
    });
    const one = new Float32Array(4);
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const x = originX + (i + 0.5) * dx;
        const z = originZ + (j + 0.5) * dy;
        const b = sampleIsland(doc, x, z, bedDepth);
        const h = data[(j * nx + i) * 4];
        const hu = data[(j * nx + i) * 4 + 1];
        const hv = data[(j * nx + i) * 4 + 2];
        if (h < minH) minH = h;
        const D = Math.max(0, -b);
        if (D < profile.dLo && D > 0) shallow += h;
        if (D < profile.dHi && h > 1e-6) {
          region += h;
          const u = hu / h;
          const v = hv / h;
          energy += 0.5 * h * (u * u + v * v) + 0.5 * GRAVITY * h * h + GRAVITY * h * b;
        }
        const wet = data[(j * nx + i) * 4 + 3];
        if (wet > wetMax) wetMax = wet;
        if (h > 0.008 && b > runup) {
          runup = b;
          runX = x;
          runZ = z;
        }
        if (h > 0.02) {
          const ox = x - profile.cx;
          const oz = z - profile.cz;
          const rad = Math.hypot(ox, oz) || 1;
          const rel = Math.atan2(oz, ox) - Math.atan2(wz, wx);
          const norm = Math.atan2(Math.sin(rel), Math.cos(rel));
          const idx = ((Math.round(norm / (Math.PI / 4)) % 8) + 8) % 8;
          const inland = profile.Rbeach - rad;
          if (inland > azInland[idx]) { azInland[idx] = inland; azBed[idx] = b; }
        }
        const eta = h + b;
        if (Math.abs(x - expX) < dx && Math.abs(z - expZ) < dy) expE = Math.max(expE, eta);
        if (Math.abs(x - shX) < dx && Math.abs(z - shZ) < dy) shE = Math.max(shE, eta);
        if (D > 3 && D < 5 && seamN < 12) {
          const u = Math.min(oceanN - 1, Math.max(0, Math.floor(((x / L + 0.5) % 1) * oceanN)));
          const v = Math.min(oceanN - 1, Math.max(0, Math.floor(((z / L + 0.5) % 1) * oceanN)));
          guard(() => { gl.readPixels(u, v, 1, 1, gl.RGBA, gl.FLOAT, one); });
          seam = Math.max(seam, Math.abs(eta - one[0]));
          seamN += 1;
        }
      }
    }
    return {
      shallow: shallow * dx * dy,
      region: region * dx * dy,
      energy: energy * dx * dy,
      stillShallow,
      stillRegion,
      stillEnergy,
      minH,
      exposed: expE,
      sheltered: shE,
      wetMax,
      runup,
      runX,
      runZ,
      inland: profile.Rbeach - Math.hypot(runX - profile.cx, runZ - profile.cz),
      az: azInland.map((inlandM, i) => ({ bin: i, inlandM: inlandM < -1e8 ? null : Math.round(inlandM * 100) / 100, bed: azBed[i] < -1e8 ? null : Math.round(azBed[i] * 100) / 100 })),
      leeInlandM: azInland[0] < -1e8 ? null : Math.round(azInland[0] * 100) / 100,
      windwardInlandM: azInland[4] < -1e8 ? null : Math.round(azInland[4] * 100) / 100,
      seam,
      r2: lastR2,
      tex: state[cur],
    };
  }

  function uploadAll() {
    guard(() => {
      for (const t of state) {
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, atlasW, atlasH, gl.RGBA, gl.FLOAT, init);
      }
    });
    cur = 0;
  }

  // Depth rebuilds the bed. Size also moves the solver domain, so the film
  // is put back to still water; a 100 m state must not be read as a 1000 m one.
  function sync(depth, size) {
    const d = Math.max(0.5, Number(depth) || bedDepth);
    const next = useIslandSize(size || profile.size);
    const depthMoved = Math.abs(d - bedDepth) >= 0.05;
    const sizeMoved = next.size !== profile.size;
    if (!depthMoved && !sizeMoved) return;
    bedDepth = d;
    if (sizeMoved) {
      profile = next;
      const grew = next.nx !== nx || next.ny !== ny;
      nx = next.nx;
      ny = next.ny;
      dx = profile.span / nx;
      dy = profile.span / ny;
      originX = profile.cx - profile.span / 2;
      originZ = profile.cz - profile.span / 2;
      if (grew) {
        atlasW = Math.max(HM, nx);
        atlasH = HM + ny;
        guard(() => {
          for (const t of state) gl.deleteTexture(t);
          for (const fb of fbo) gl.deleteFramebuffer(fb);
        });
        // Recreate below, after the bed is rebuilt.
      }
    }
    doc = islandById(profile.id, bedDepth, profile);
    if (sizeMoved) {
      shadow = bakeShadow(doc.heights, doc.nx, doc.ny, doc.meters, globalThis.__oceanShadeSun || [0.4, 0.8, 0.2], bedDepth);
      init = new Float32Array(atlasW * atlasH * 4);
      fillInit();
      if (state[0] && gl.getTexParameter) {
        // Textures were deleted when the grid count changed.
      }
      guard(() => {
        const alive = gl.isTexture(state[0]);
        if (!alive) {
          state[0] = tex(init);
          state[1] = tex(init);
          fbo[0] = gl.createFramebuffer();
          fbo[1] = gl.createFramebuffer();
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[0]);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, state[0], 0);
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1]);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, state[1], 0);
        }
      });
      uploadAll();
    } else {
      updateShadow(globalThis.__oceanShadeSun || [0.4, 0.8, 0.2]);
    }
  }

  function setDepth(next) {
    sync(next, profile.size);
  }

  return {
    step, read, measures, updateShadow, setDepth, sync,
    get depth() { return bedDepth; },
    get dx() { return dx; },
    get dy() { return dy; },
    get profile() { return profile; },
    texture: () => state[cur],
    get island() { return doc; },
    get stillShallow() { return stillShallow; },
    get stillRegion() { return stillRegion; },
    get stillEnergy() { return stillEnergy; },
    get nx() { return nx; },
    get ny() { return ny; },
    get r2() { return lastR2; },
    get atlasW() { return atlasW; },
    get atlasH() { return atlasH; },
  };
}

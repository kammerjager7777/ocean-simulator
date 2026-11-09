// Island shallow-water pass on the ocean's WebGL2 context.
// Deep cells are replaced by the live spectral surface. The blend band
// (still depth 2 m to 6 m) mixes the two, which is where the shallow-water
// speed and the dispersive speed nearly agree. Shallow cells integrate the
// shallow-water equations. The spectral elevation is not added on again.

import { GRAVITY } from "./gravity.js";
import { ISLAND, islandBed } from "./island.js";

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
uniform vec2 uCenter;
uniform float uRbeach, uRbank, uPeak, uShelf, uDLo, uDHi;
out vec4 o;

float bedAt(vec2 p) {
  float r = length(p - uCenter);
  if (r >= uRbank) return -uDepth;
  if (r >= uRbeach) {
    float t = (r - uRbeach) / (uRbank - uRbeach);
    return uShelf + (-uDepth - uShelf) * t;
  }
  return uPeak + (uShelf - uPeak) * (r / uRbeach);
}

float sweW(float D) {
  float t = clamp((D - uDLo) / (uDHi - uDLo), 0.0, 1.0);
  float s = t * t * (3.0 - 2.0 * t);
  return 1.0 - s;
}

struct Cell { float h; float b; float u; float v; };

Cell loadAt(int i, int j) {
  ivec2 sz = textureSize(st, 0);
  Cell c;
  i = clamp(i, 0, sz.x - 1);
  j = clamp(j, 0, sz.y - 1);
  vec4 s = texelFetch(st, ivec2(i, j), 0);
  vec2 world = uOrigin + (vec2(i, j) + 0.5) * vec2(uDx, uDy);
  c.b = bedAt(world);
  c.h = s.r;
  c.u = s.r > uEps ? s.g / s.r : 0.0;
  c.v = s.r > uEps ? s.b / s.r : 0.0;
  return c;
}

void face(Cell L, Cell R, float unL, float utL, float unR, float utR,
          out float fh, out float fn, out float ft, out float hKeep, bool keepLeft) {
  float bs = max(L.b, R.b);
  float hl = max(0.0, L.h + L.b - bs);
  float hr = max(0.0, R.h + R.b - bs);
  float smax = max(abs(unL) + sqrt(uG * hl), abs(unR) + sqrt(uG * hr));
  float qL = hl * unL, qR = hr * unR;
  fh = 0.5 * (qL + qR) - 0.5 * smax * (hr - hl);
  fn = 0.5 * (qL * unL + 0.5 * uG * hl * hl + qR * unR + 0.5 * uG * hr * hr) - 0.5 * smax * (qR - qL);
  ft = 0.5 * (qL * utL + qR * utR) - 0.5 * smax * (hr * utR - hl * utL);
  hKeep = keepLeft ? hl : hr;
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 world = uOrigin + gl_FragCoord.xy * vec2(uDx, uDy);
  float b = bedAt(world);
  float D = max(0.0, -b);
  float w = sweW(D);
  vec2 uv = fract(world / uL + 0.5);
  float etaS = texture(ocean, uv).r;
  float hS = max(0.0, etaS - b);
  float hStill = max(D, 0.05);
  float speed = 2.0 * (sqrt(uG * max(hS, 1e-4)) - sqrt(uG * hStill));
  vec2 vel = uWind * speed;
  vec4 spec = vec4(hS, hS * vel.x, hS * vel.y, 0.0);
  if (w < 0.02) { o = spec; return; }
  Cell C = loadAt(p.x, p.y);
  Cell L = loadAt(p.x - 1, p.y);
  Cell R = loadAt(p.x + 1, p.y);
  Cell B = loadAt(p.x, p.y - 1);
  Cell T = loadAt(p.x, p.y + 1);
  if (p.x == 0) L.u = -C.u;
  if (p.x == textureSize(st, 0).x - 1) R.u = -C.u;
  if (p.y == 0) B.v = -C.v;
  if (p.y == textureSize(st, 0).y - 1) T.v = -C.v;
  float fhR, fnR, ftR, hRkeep, fhL, fnL, ftL, hLkeep;
  float ghT, gnT, gtT, hTkeep, ghB, gnB, gtB, hBkeep;
  face(C, R, C.u, C.v, R.u, R.v, fhR, fnR, ftR, hRkeep, true);
  face(L, C, L.u, L.v, C.u, C.v, fhL, fnL, ftL, hLkeep, false);
  face(C, T, C.v, C.u, T.v, T.u, ghT, gnT, gtT, hTkeep, true);
  face(B, C, B.v, B.u, C.v, C.u, ghB, gnB, gtB, hBkeep, false);
  float sx = 0.5 * uG * (hRkeep * hRkeep - hLkeep * hLkeep) / uDx;
  float sy = 0.5 * uG * (hTkeep * hTkeep - hBkeep * hBkeep) / uDy;
  vec4 cur = texelFetch(st, p, 0);
  float hn = cur.r - uDt * ((fhR - fhL) / uDx + (ghT - ghB) / uDy);
  float hun = cur.g - uDt * ((fnR - fnL) / uDx + (gtT - gtB) / uDy - sx);
  float hvn = cur.b - uDt * ((ftR - ftL) / uDx + (gnT - gnB) / uDy - sy);
  if (hn < 0.0 && hn >= uClip) hn = 0.0;
  if (hn < uEps) { hun = 0.0; hvn = 0.0; }
  float wet = cur.a;
  if (hn > 0.02) wet = min(1.0, wet + uDt * 0.5);
  else wet *= exp(-uDt / 20.0);
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

export function createIslandGpu(gl, depth) {
  const I = ISLAND;
  const nx = I.nx;
  const ny = I.ny;
  const dx = I.span / nx;
  const dy = I.span / ny;
  const originX = I.cx - I.span / 2;
  const originZ = I.cz - I.span / 2;
  const init = new Float32Array(nx * ny * 4);
  let stillShallow = 0;
  let stillRegion = 0;
  let stillEnergy = 0;
  for (let j = 0; j < ny; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const x = originX + (i + 0.5) * dx;
      const z = originZ + (j + 0.5) * dy;
      const b = islandBed(x, z, depth);
      const h = Math.max(0, -b);
      init[(j * nx + i) * 4] = h;
      const D = Math.max(0, -b);
      if (D > 0 && D < I.dLo) stillShallow += h;
      if (D < I.dHi) {
        stillRegion += h;
        stillEnergy += 0.5 * GRAVITY * h * h + GRAVITY * h * b;
      }
    }
  }
  stillShallow *= dx * dy;
  stillRegion *= dx * dy;
  stillEnergy *= dx * dy;
  const tex = (data) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, nx, ny, 0, gl.RGBA, gl.FLOAT, data);
    // The solver reads this with texelFetch, which ignores the filter.
    // The sea mesh samples it with texture(), and nearest made the beach
    // a ring of 1.25 m terraces.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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
  for (const name of ["st", "ocean", "uDt", "uDx", "uDy", "uG", "uEps", "uClip", "uOrigin", "uWind", "uL", "uDepth", "uCenter", "uRbeach", "uRbank", "uPeak", "uShelf", "uDLo", "uDHi"]) {
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

  function step(dt, oceanTex, L, windX, windZ) {
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - cur]);
      gl.viewport(0, 0, nx, ny);
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
      gl.uniform1f(loc.uDepth, depth);
      gl.uniform2f(loc.uCenter, I.cx, I.cz);
      gl.uniform1f(loc.uRbeach, I.Rbeach);
      gl.uniform1f(loc.uRbank, I.Rbank);
      gl.uniform1f(loc.uPeak, I.peak);
      gl.uniform1f(loc.uShelf, I.shelf);
      gl.uniform1f(loc.uDLo, I.dLo);
      gl.uniform1f(loc.uDHi, I.dHi);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      cur = 1 - cur;
    });
  }

  function read() {
    const data = new Float32Array(nx * ny * 4);
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[cur]);
      gl.readPixels(0, 0, nx, ny, gl.RGBA, gl.FLOAT, data);
    });
    return data;
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
    const wlen = Math.hypot(windX || 0, windZ || 1) || 1;
    const wx = (windX || 0) / wlen;
    const wz = (windZ || 1) / wlen;
    const expX = I.cx - wx * 16;
    const expZ = I.cz - wz * 16;
    const shX = I.cx + wx * 16;
    const shZ = I.cz + wz * 16;
    guard(() => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, scratch);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oceanTex, 0);
    });
    const one = new Float32Array(4);
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const x = originX + (i + 0.5) * dx;
        const z = originZ + (j + 0.5) * dy;
        const b = islandBed(x, z, depth);
        const h = data[(j * nx + i) * 4];
        const hu = data[(j * nx + i) * 4 + 1];
        const hv = data[(j * nx + i) * 4 + 2];
        if (h < minH) minH = h;
        const D = Math.max(0, -b);
        if (D < I.dLo && D > 0) shallow += h;
        if (D < I.dHi && h > 1e-6) {
          region += h;
          const u = hu / h;
          const v = hv / h;
          energy += 0.5 * h * (u * u + v * v) + 0.5 * GRAVITY * h * h + GRAVITY * h * b;
        }
        const wet = data[(j * nx + i) * 4 + 3];
        if (wet > wetMax) wetMax = wet;
        if (h > 0.02 && b > runup) runup = b;
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
      seam,
      tex: state[cur],
    };
  }

  return {
    step, read, measures,
    texture: () => state[cur],
    stillShallow, stillRegion, stillEnergy,
    nx, ny, dx, dy,
  };
}

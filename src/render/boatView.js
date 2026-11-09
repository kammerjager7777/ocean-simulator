// v31.8 boat: one procedural sloop, about 8 m, the same scale the floater was built for.
// The hull, cabin, mast and rigging are rigid. Only sail vertices (sail.x > 0) move, in
// the vertex shader, from the apparent wind. The buoy mesh does not set that attribute.
// Float, drift and yaw stay in physics/boat.js.
const VS = `#version 300 es
in vec3 pos; in vec3 nrm; in vec3 col; in vec3 sail;
uniform mat4 vp; uniform mat4 model;
uniform vec4 uSail; // cross m/s (+starboard), apparent speed, fill 0..1, camber metres
uniform float uSailT; uniform float uFlap;
out vec3 vN; out vec3 vC;
void main() {
  vec3 p = pos; vec3 n = nrm;
  if (sail.x > 0.5) {
    float u = clamp(sail.y, 0.0, 1.0), v = clamp(sail.z, 0.0, 1.0);
    float crossW = uSail.x, aw = max(uSail.y, 0.0), fill = clamp(uSail.z, 0.0, 1.0), camber = max(uSail.w, 0.0);
    float lee = crossW >= 0.0 ? 1.0 : -1.0;
    float breeze = 1.0 - exp(-aw / 3.2);
    float luff = (1.0 - fill) * breeze;
    float amp = uFlap * (luff * (0.05 + 0.32 * clamp(aw / 9.0, 0.0, 1.0)) + fill * 0.018 * clamp(aw / 14.0, 0.0, 1.0));
    float freq = mix(1.8, 8.5, 1.0 - fill) * (0.4 + clamp(aw / 11.0, 0.0, 1.15));
    float leech = u * u;
    float wave = sin(uSailT * freq + v * 7.2 + u * 3.4);
    wave += (1.0 - fill) * 0.42 * sin(uSailT * freq * 1.67 + v * 11.0 - u);
    p.z += lee * (camber * sin(3.14159265 * u) * (0.28 + 0.72 * v) + amp * wave * leech);
    p.y -= 0.04 * leech + luff * 0.2 * leech * (0.3 + 0.7 * v);
    n.z += lee * camber * 3.14159265 * cos(3.14159265 * u) * 0.35;
    n = normalize(n);
  }
  vN = mat3(model) * n; vC = col; gl_Position = vp * model * vec4(p, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vC; uniform vec3 lightDir; out vec4 o;
void main() { vec3 n = normalize(vN); float d = max(dot(n, normalize(lightDir)), 0.0);
  float sky = 0.5 + 0.5 * n.y; vec3 c = vC * (0.32 * sky + 0.78 * d);
  o = vec4(pow(c, vec3(1.0 / 2.2)), 1.0); }`;
const smooth01 = (x) => { const t = Math.min(1, Math.max(0, x)); return t * t * (3 - 2 * t); };
// Apparent wind in the boat frame. fill is 0 head-to-wind and in a calm, and rises once
// the wind is far enough off the bow to hold a sail. camber is the billow in metres.
// This does not push the hull.
export function sailPose(boat, fx) {
  const spd = fx && Number.isFinite(fx.speed) ? fx.speed : 0;
  const dir = fx && Number.isFinite(fx.dir) ? fx.dir : 0;
  const yaw = boat && Number.isFinite(boat.yaw) ? boat.yaw : null;
  if (yaw == null) return { cross: 0, along: 0, aw: 0, fill: 0, camber: 0, t: fx && fx.t || 0, off: 0 };
  const ax = Math.cos(dir) * spd - (boat.vx || 0);
  const az = Math.sin(dir) * spd - (boat.vz || 0);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const along = ax * cy - az * sy;
  const cross = ax * sy + az * cy;
  const aw = Math.hypot(along, cross);
  const from = Math.atan2(-cross, -along);
  const off = Math.abs(from);
  const point = smooth01((off - 0.5) / 0.7);
  const breeze = 1 - Math.exp(-aw / 3.2);
  const fill = point * breeze;
  const reach = Math.sin(Math.min(off, Math.PI));
  const camber = fill * (0.18 + 0.62 * Math.min(1, aw / 14)) * (0.62 + 0.38 * reach);
  return { cross, along, aw, fill, camber, t: fx && Number.isFinite(fx.t) ? fx.t : 0, off };
}
function buildMesh() {
  const P = [], N = [], C = [], S = [];
  const push = (p, n, col, sail) => { P.push(...p); N.push(...n); C.push(...col); S.push(...sail); };
  const tri = (a, b, c, col, sa = [0, 0, 0], sb = [0, 0, 0], sc = [0, 0, 0]) => {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1; n = n.map((x) => x / l);
    const cc = typeof col === "function" ? col((a[1] + b[1] + c[1]) / 3) : col;
    push(a, n, cc, sa); push(b, n, cc, sb); push(c, n, cc, sc);
  };
  const quad = (a, b, c, d, col) => { tri(a, b, c, col); tri(a, c, d, col); };
  const hullCol = (y) => (y < -0.02 ? [0.10, 0.13, 0.20] : y < 0.07 ? [0.16, 0.17, 0.18] : [0.93, 0.94, 0.93]);
  // sections along x (stern -4 .. bow +4): [x, half beam, deck y, keel y]
  const SEC = [[-4, 0.95, 0.58, -0.42], [-2.4, 1.38, 0.64, -0.82], [-0.4, 1.42, 0.7, -0.86], [1.4, 1.15, 0.74, -0.7], [2.8, 0.62, 0.82, -0.38], [3.7, 0.16, 0.96, -0.08], [4.05, 0.02, 1.05, 0.2]];
  const ring = ([x, hb, dy, ky]) => [[x, dy, -hb], [x, -0.2, -hb * 0.9], [x, ky, 0], [x, -0.2, hb * 0.9], [x, dy, hb]];
  for (let i = 0; i + 1 < SEC.length; i++) {
    const a = ring(SEC[i]), b = ring(SEC[i + 1]);
    for (let k = 0; k < 4; k++) quad(a[k], b[k], b[k + 1], a[k + 1], hullCol);
    quad(a[4], b[4], b[0], a[0], [0.48, 0.34, 0.2]);
  }
  const st = ring(SEC[0]); for (let k = 1; k < 4; k++) tri(st[0], st[k + 1], st[k], hullCol);
  const box = (x0, x1, y0, y1, z0, z1, col, top) => {
    const v = (x, y, z) => [x, y, z];
    quad(v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1), col);
    quad(v(x1, y0, z0), v(x0, y0, z0), v(x0, y1, z0), v(x1, y1, z0), col);
    quad(v(x1, y0, z1), v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1), col);
    quad(v(x0, y0, z0), v(x0, y0, z1), v(x0, y1, z1), v(x0, y1, z0), col);
    quad(v(x0, y1, z1), v(x1, y1, z1), v(x1, y1, z0), v(x0, y1, z0), top || col);
    quad(v(x0, y0, z0), v(x1, y0, z0), v(x1, y0, z1), v(x0, y0, z1), col);
  };
  const cloth = [0.94, 0.91, 0.82];
  const spar = [0.72, 0.73, 0.7];
  box(-0.15, 1.35, 0.68, 1.48, -0.62, 0.62, [0.95, 0.95, 0.93], [0.86, 0.87, 0.84]); // cabin
  box(0.55, 0.62, 1.48, 1.72, -0.28, 0.28, [0.22, 0.28, 0.34]); // companion hatch
  box(0.22, 0.32, 0.72, 7.15, -0.04, 0.04, spar); // mast
  box(-2.45, 0.28, 1.16, 1.26, -0.03, 0.03, spar); // boom
  box(-0.15, 0.85, -1.2, -0.5, -0.035, 0.035, [0.12, 0.13, 0.15]); // keel
  box(-3.85, -3.7, -0.65, 0.45, -0.02, 0.02, [0.2, 0.21, 0.22]); // rudder
  const stay = (a, b, w, col) => {
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l = Math.hypot(...d) || 1;
    let s = Math.abs(d[1]) > 0.9 * l ? [1, 0, 0] : [0, 1, 0];
    let px = d[1] * s[2] - d[2] * s[1], py = d[2] * s[0] - d[0] * s[2], pz = d[0] * s[1] - d[1] * s[0];
    const pl = Math.hypot(px, py, pz) || 1; px = px / pl * w; py = py / pl * w; pz = pz / pl * w;
    quad([a[0] - px, a[1] - py, a[2] - pz], [b[0] - px, b[1] - py, b[2] - pz], [b[0] + px, b[1] + py, b[2] + pz], [a[0] + px, a[1] + py, a[2] + pz], col);
  };
  stay([0.27, 7.05, 0], [-3.6, 0.62, 0], 0.015, spar); // backstay
  stay([0.27, 7.05, 0], [0.2, 0.7, 1.15], 0.015, spar);
  stay([0.27, 7.05, 0], [0.2, 0.7, -1.15], 0.015, spar);
  const sheet = (id, nu, nv, luff0, luff1, leech0, leech1) => {
    const at = (u, v) => {
      const a = [luff0[0] + (luff1[0] - luff0[0]) * v, luff0[1] + (luff1[1] - luff0[1]) * v, 0];
      const b = [leech0[0] + (leech1[0] - leech0[0]) * v, leech0[1] + (leech1[1] - leech0[1]) * v, 0];
      return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, 0];
    };
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const u0 = i / nu, u1 = (i + 1) / nu, v0 = j / nv, v1 = (j + 1) / nv;
      const p00 = at(u0, v0), p10 = at(u1, v0), p11 = at(u1, v1), p01 = at(u0, v1);
      const s00 = [id, u0, v0], s10 = [id, u1, v0], s11 = [id, u1, v1], s01 = [id, u0, v1];
      tri(p00, p10, p11, cloth, s00, s10, s11); tri(p00, p11, p01, cloth, s00, s11, s01);
      tri(p00, p11, p10, cloth, s00, s11, s10); tri(p00, p01, p11, cloth, s00, s01, s11);
    }
  };
  sheet(1, 7, 8, [0.26, 1.28, 0], [0.26, 6.95, 0], [-2.4, 1.2, 0], [-0.05, 6.7, 0]); // main
  sheet(2, 6, 7, [3.45, 0.95, 0], [0.4, 6.55, 0], [0.85, 1.15, 0], [0.4, 6.55, 0]); // jib
  return { P: new Float32Array(P), N: new Float32Array(N), C: new Float32Array(C), S: new Float32Array(S), count: P.length / 3 };
}
// v30.21 buoy (Michael): NOAA/NDBC-style 3 m discus weather buoy, classic yellow #F2C230, mast with
// instruments and a top light (the light's glow is drawn by the on-sea UI overlay). Origin at the waterline.
function buildBuoyMesh() {
  const P = [], N = [], C = [];
  const tri = (a, b, c, col) => {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1; n = n.map((x) => x / l);
    for (const p of [a, b, c]) { P.push(...p); N.push(...n); C.push(...col); }
  };
  const Y = [0.90, 0.58, 0.05], YD = [0.62, 0.38, 0.03], G = [0.55, 0.56, 0.58], DK = [0.12, 0.13, 0.15], W = [0.9, 0.9, 0.88];
  const cyl = (r0, r1, y0, y1, col, seg = 20, capTop = true) => {
    for (let i = 0; i < seg; i++) {
      const a0 = (i / seg) * 2 * Math.PI, a1 = ((i + 1) / seg) * 2 * Math.PI;
      const p = (r, y, a) => [r * Math.cos(a), y, r * Math.sin(a)];
      tri(p(r0, y0, a0), p(r1, y1, a1), p(r1, y1, a0), col); tri(p(r0, y0, a0), p(r0, y0, a1), p(r1, y1, a1), col);
      if (capTop) tri([0, y1, 0], p(r1, y1, a0), p(r1, y1, a1), col);
    }
  };
  cyl(1.2, 1.5, -0.9, -0.2, YD);   // hull below/at the waterline
  cyl(1.5, 1.45, -0.2, 0.35, Y);   // discus rim
  cyl(1.45, 0.9, 0.35, 0.5, Y);    // deck slope
  cyl(0.55, 0.5, 0.5, 1.1, W, 12); // electronics well
  for (const [x, z] of [[0.35, 0.35], [-0.35, 0.35], [0.35, -0.35], [-0.35, -0.35]]) { // 4-leg mast tower
    const q = (y, k) => [x * k, y, z * k];
    tri(q(1.1, 1), q(4.2, 0.25), [x * 0.25 + 0.06, 4.2, z * 0.25], G); tri(q(1.1, 1), [x + 0.06, 1.1, z], [x * 0.25 + 0.06, 4.2, z * 0.25], G);
  }
  cyl(0.25, 0.25, 2.4, 2.9, Y, 10);  // instrument housing (anemometer/temp)
  cyl(0.05, 0.05, 2.9, 3.5, G, 6);   // instrument arm
  cyl(0.18, 0.18, 3.5, 3.6, DK, 8);  // anemometer cups (disc)
  cyl(0.12, 0.10, 4.2, 4.55, [1.0, 0.92, 0.55], 8); // lantern
  cyl(0.05, 0.05, 4.55, 4.8, G, 6);   // finial
  return { P: new Float32Array(P), N: new Float32Array(N), C: new Float32Array(C), count: P.length / 3 };
}
export const BUOY_LAMP_Y = 4.4;
export function createBuoyView(gl) { return createBoatView(gl, buildBuoyMesh); }
export function createBoatView(gl, build = buildMesh) {
  const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error("boat shader: " + gl.getShaderInfoLog(o)); return o; };
  const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
  const m = build(); const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  [["pos", m.P], ["nrm", m.N], ["col", m.C]].forEach(([n, d]) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW);
    const l = gl.getAttribLocation(prog, n); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 3, gl.FLOAT, false, 0, 0); });
  if (m.S) {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, m.S, gl.STATIC_DRAW);
    const l = gl.getAttribLocation(prog, "sail"); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 3, gl.FLOAT, false, 0, 0);
  }
  gl.bindVertexArray(null);
  const U = { vp: gl.getUniformLocation(prog, "vp"), model: gl.getUniformLocation(prog, "model"), lightDir: gl.getUniformLocation(prog, "lightDir"), sail: gl.getUniformLocation(prog, "uSail"), sailT: gl.getUniformLocation(prog, "uSailT"), flap: gl.getUniformLocation(prog, "uFlap") };
  const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  return { draw(boat, vp, sun, fx) {
    const pose = sailPose(boat, fx);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.enable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(U.vp, false, vp); gl.uniformMatrix4fv(U.model, false, boat.matrix()); gl.uniform3f(U.lightDir, sun[0], sun[1], sun[2]);
    if (U.sail) gl.uniform4f(U.sail, pose.cross, pose.aw, pose.fill, pose.camber);
    if (U.sailT) gl.uniform1f(U.sailT, pose.t);
    if (U.flap) gl.uniform1f(U.flap, reduce ? 0 : 1);
    gl.drawArrays(gl.TRIANGLES, 0, m.count); gl.bindVertexArray(null);
    if (m.S && boat && Number.isFinite(boat.yaw)) globalThis.__oceanSail = { ...pose, yaw: boat.yaw, flap: reduce ? 0 : 1 };
  } };
}

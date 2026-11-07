// v30.4 boat: low-poly generic fishing boat (hull, cabin, mast), true scale (8 m), flat shaded.
const VS = `#version 300 es
in vec3 pos; in vec3 nrm; in vec3 col;
uniform mat4 vp; uniform mat4 model;
out vec3 vN; out vec3 vC;
void main() { vN = mat3(model) * nrm; vC = col; gl_Position = vp * model * vec4(pos, 1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vC; uniform vec3 lightDir; out vec4 o;
void main() { vec3 n = normalize(vN); float d = max(dot(n, normalize(lightDir)), 0.0);
  float sky = 0.5 + 0.5 * n.y; vec3 c = vC * (0.32 * sky + 0.78 * d);
  o = vec4(pow(c, vec3(1.0 / 2.2)), 1.0); }`;
function buildMesh() {
  const P = [], N = [], C = [];
  const tri = (a, b, c, col) => {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1; n = n.map((x) => x / l);
    const cc = typeof col === "function" ? col((a[1] + b[1] + c[1]) / 3) : col;
    for (const p of [a, b, c]) { P.push(...p); N.push(...n); C.push(...cc); }
  };
  const quad = (a, b, c, d, col) => { tri(a, b, c, col); tri(a, c, d, col); };
  const hullCol = (y) => (y < -0.15 ? [0.42, 0.07, 0.06] : [0.86, 0.86, 0.84]);
  // sections along x (stern -4 .. bow +4): [x, half beam, deck y, keel y]
  const S = [[-4, 1.25, 0.65, -0.7], [-2, 1.4, 0.65, -0.8], [0, 1.4, 0.7, -0.8], [2, 1.15, 0.8, -0.7], [3.3, 0.6, 0.9, -0.45], [4, 0.04, 1.0, -0.1]];
  const ring = ([x, hb, dy, ky]) => [[x, dy, -hb], [x, -0.25, -hb * 0.88], [x, ky, 0], [x, -0.25, hb * 0.88], [x, dy, hb]];
  for (let i = 0; i + 1 < S.length; i++) {
    const a = ring(S[i]), b = ring(S[i + 1]);
    for (let k = 0; k < 4; k++) quad(a[k], b[k], b[k + 1], a[k + 1], hullCol);
    quad(a[4], b[4], b[0], a[0], [0.55, 0.45, 0.33]); // deck (wood)
  }
  const st = ring(S[0]); for (let k = 1; k < 4; k++) tri(st[0], st[k + 1], st[k], hullCol); // transom
  const box = (x0, x1, y0, y1, z0, z1, col, top) => {
    const v = (x, y, z) => [x, y, z];
    quad(v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1), col);
    quad(v(x1, y0, z0), v(x0, y0, z0), v(x0, y1, z0), v(x1, y1, z0), col);
    quad(v(x1, y0, z1), v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1), col);
    quad(v(x0, y0, z0), v(x0, y0, z1), v(x0, y1, z1), v(x0, y1, z0), col);
    quad(v(x0, y1, z1), v(x1, y1, z1), v(x1, y1, z0), v(x0, y1, z0), top || col);
  };
  box(-2.4, 0.4, 0.65, 2.3, -0.95, 0.95, [0.9, 0.9, 0.88], [0.18, 0.32, 0.55]); // wheelhouse
  box(0.39, 0.42, 1.5, 2.0, -0.8, 0.8, [0.1, 0.13, 0.16]);                      // windscreen
  box(1.0, 1.16, 0.7, 5.6, -0.08, 0.08, [0.55, 0.56, 0.58]);                    // mast
  box(-0.3, 2.6, 4.1, 4.2, -0.04, 0.04, [0.55, 0.56, 0.58]);                    // boom
  return { P: new Float32Array(P), N: new Float32Array(N), C: new Float32Array(C), count: P.length / 3 };
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
  gl.bindVertexArray(null);
  const U = { vp: gl.getUniformLocation(prog, "vp"), model: gl.getUniformLocation(prog, "model"), lightDir: gl.getUniformLocation(prog, "lightDir") };
  return { draw(boat, vp, sun) {
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.enable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(U.vp, false, vp); gl.uniformMatrix4fv(U.model, false, boat.matrix()); gl.uniform3f(U.lightDir, sun[0], sun[1], sun[2]);
    gl.drawArrays(gl.TRIANGLES, 0, m.count); gl.bindVertexArray(null);
  } };
}

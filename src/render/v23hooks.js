// v30.23 render hooks (kept out of view.js so the v30.22g rebase stays small): the sky dome backdrop,
// and the per-frame uniforms for the sky-keyed reflection/glitter, the ripples and the seabed relief.
import { rippleUniforms, setRippleScale } from "../physics/ripples.js?v=30.29";
const SKY_VS = `#version 300 es
out vec2 vNdc;
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0; vNdc = p; gl_Position = vec4(p, 0.9999, 1.0); }`;
const SKY_FS = `#version 300 es
precision highp float;
in vec2 vNdc; out vec4 o;
uniform mat4 uInv; uniform vec3 uZen, uHor, uDisc, uMoon, uSunD, uMoonD; uniform float uStars, uSunOn, uMoonOn, uT, uAsp; uniform vec2 uDrift;
vec3 toS(vec3 c) { return mix(c * 12.92, 1.055 * pow(max(c, 0.0), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vn(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p + 3.1; a *= 0.5; } return s; }
void main() {
  vec4 a = uInv * vec4(vNdc, -1.0, 1.0), b = uInv * vec4(vNdc, 1.0, 1.0);
  vec3 d = normalize(b.xyz / b.w - a.xyz / a.w);
  float y = d.y;
  vec3 c = y >= 0.0 ? mix(uHor, uZen, pow(clamp(y, 0.0, 1.0), 0.55)) : mix(uHor, uHor * 0.55 + uZen * 0.15, smoothstep(0.0, 0.35, -y));
  float cs = dot(d, normalize(uSunD)), cm = dot(d, normalize(uMoonD));
  // v30.24: stars everywhere in the backdrop (the camera mostly looks below the true horizon), faint, slowly twinkling
  float night = uStars;
  if (night > 0.0) {
    vec2 g = vec2(atan(d.z, d.x) * 180.0, asin(clamp(y, -1.0, 1.0)) * 180.0); vec2 id = floor(g); float s = h21(id);
    float tw = 0.65 + 0.35 * sin(uT * (1.5 + 3.0 * h21(id + 7.0)) + 6.2831 * s);
    float st = step(0.987, s) * smoothstep(0.42, 0.0, length(fract(g) - 0.5)) * night * tw * (0.5 + 0.5 * h21(id + 3.0));
    c += vec3(0.9, 0.93, 1.0) * st * 0.55;
    // shooting stars: one slot every 7 s, about half of them used; a 0.7 s streak with a fading tail (screen space)
    float slot = floor(uT / 7.0), ft = (uT - slot * 7.0) / 0.7;
    if (ft < 1.0 && h21(vec2(slot, 1.7)) > 0.45) {
      vec2 p0 = vec2(mix(-0.8, 0.8, h21(vec2(slot, 2.3))), mix(0.2, 0.85, h21(vec2(slot, 4.1))));
      float an = mix(3.6, 4.2, h21(vec2(slot, 5.9))) + (h21(vec2(slot, 8.8)) > 0.5 ? 0.0 : -1.2); vec2 dv = vec2(cos(an), sin(an));
      vec2 q = vec2(vNdc.x * uAsp, vNdc.y), hd = vec2(p0.x * uAsp, p0.y) + dv * 0.5 * ft;
      float along = dot(hd - q, dv), across = abs(dot(hd - q, vec2(-dv.y, dv.x)));
      float tail = 0.22 * (1.0 - 0.5 * ft);
      float m = step(0.0, along) * (1.0 - smoothstep(0.0, tail, along)) * (1.0 - smoothstep(0.0012, 0.0035, across)) * sin(3.14159 * ft);
      c += vec3(0.95, 0.97, 1.0) * m * 0.8 * night;
    }
  }
  // v30.24: soft drifting clouds (cheap 4-octave value noise on the view direction), tinted by the time of day:
  // lit by the sky colour, warm near the sun, dark grey-blue at night (they hide the stars behind them)
  {
    vec2 cp = d.xz / (0.35 + abs(y)) * 1.6 + uDrift;
    float n = fbm(cp + 0.25 * vec2(fbm(cp * 0.7 + 5.0), fbm(cp * 0.7 + 9.0)));
    float cov = smoothstep(0.52, 0.8, n) * (0.55 + 0.45 * smoothstep(-0.9, -0.2, y));
    vec3 lit = mix(uHor * 1.08 + 0.05, vec3(0.92, 0.94, 0.97), 0.35 * (1.0 - night));
    lit = mix(lit, uDisc * 1.1, uSunOn * 0.55 * pow(max(cs, 0.0), 6.0));
    vec3 cc = mix(lit, uHor * 0.6 + uZen * 0.3, night * 0.7);
    c = mix(c, cc, cov * mix(0.32, 0.5, night));
  }
  if (uSunOn > 0.5) { // v30.25: ONE crisp disc, a dark rim and a tight local sky falloff (not a glow).
    float halo = (1.0 - smoothstep(0.9980, 0.99945, cs));
    c *= 1.0 - 0.13 * halo;
    float sd = smoothstep(0.99955, 0.99964, cs);
    vec3 sunc = mix(vec3(1.0, 0.70, 0.28), uDisc * 1.18, 0.36); // readable warm rim even on a pale noon sky
    c = mix(c, vec3(0.39, 0.26, 0.13), sd * (1.0 - smoothstep(0.99962, 0.99965, cs)));
    c = mix(c, sunc, smoothstep(0.99962, 0.99965, cs));
  }
  if (uMoonOn > 0.5) {
    c += vec3(0.0);
    float md = smoothstep(0.99962, 0.99972, cm);
    c = mix(c, uMoon * (0.92 + 0.08 * h21(floor(d.xz * 4000.0))), md);
  }
  o = vec4(toS(c), 1.0);
}`;
function inv4(m) {
  const a = m, o = new Float32Array(16);
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4], b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12], b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06; if (!det) return o; det = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det; o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det; o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det; o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det; o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det; o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det; o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det; o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det; o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det; o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det; o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det; o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det; o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
  return o;
}
let sky = null;
function mk(gl) {
  const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error("sky: " + gl.getShaderInfoLog(o)); return o; };
  const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, SKY_VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, SKY_FS)); gl.linkProgram(p);
  const U = {}; for (const n of ["uInv", "uZen", "uHor", "uDisc", "uMoon", "uSunD", "uMoonD", "uStars", "uSunOn", "uMoonOn", "uT", "uAsp", "uDrift"]) U[n] = gl.getUniformLocation(p, n);
  return { p, U, vao: gl.createVertexArray() };
}
const SKY_OFF = typeof location !== "undefined" && new URLSearchParams(location.search).get("dome") === "0";
// Draw the dome over the cleared frame (before the sea). Returns false when it did not draw.
export function drawSkyDome(gl, vp) {
  const s = globalThis.__oceanSky; if (!s || SKY_OFF || !(gl instanceof WebGL2RenderingContext)) return false;
  if (!sky) sky = mk(gl);
  const dt = gl.isEnabled(gl.DEPTH_TEST), bl = gl.isEnabled(gl.BLEND);
  gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.depthMask(false);
  gl.useProgram(sky.p); gl.bindVertexArray(sky.vao);
  gl.uniformMatrix4fv(sky.U.uInv, false, inv4(vp));
  gl.uniform3fv(sky.U.uZen, s.zen); gl.uniform3fv(sky.U.uHor, s.hor); gl.uniform3fv(sky.U.uDisc, s.disc); gl.uniform3fv(sky.U.uMoon, s.moonDisc);
  const d = s.dirs || { sun: [0.4, 0.9, 0.2], moon: [-0.4, -0.9, -0.2] };
  gl.uniform3fv(sky.U.uSunD, d.sun); gl.uniform3fv(sky.U.uMoonD, d.moon); gl.uniform1f(sky.U.uStars, s.stars);
  gl.uniform1f(sky.U.uSunOn, (d.sunOn ?? (d.sun[1] > -0.05 ? 1 : 0)) * (!new URLSearchParams(location.search).has("nodisc") ? 1 : 0)); gl.uniform1f(sky.U.uMoonOn, d.moonOn ?? (d.moon[1] > -0.05 ? 1 : 0));
   
  const T = (typeof performance !== "undefined" ? performance.now() : 0) / 1000; gl.uniform1f(sky.U.uT, T);
  const vw = gl.drawingBufferWidth, vh = gl.drawingBufferHeight; gl.uniform1f(sky.U.uAsp, vh ? vw / vh : 1.6);
  const w = s.wind || [0.004, 0.0015]; gl.uniform2f(sky.U.uDrift, w[0] * T, w[1] * T);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.depthMask(true); if (dt) gl.enable(gl.DEPTH_TEST); if (bl) gl.enable(gl.BLEND);
  return true;
}
// Per-frame uniforms on a sea/block program (looked up once per program).
const cache = new WeakMap();
export function setV23Uniforms(gl, program, state) {
  let L = cache.get(program);
  if (!L) { L = {}; for (const n of ["uSkyHorM", "uSkyZenM", "uGlitM", "uBedS", "uRip", "uRipP", "uSunC", "uWarm", "uClarK"]) L[n] = gl.getUniformLocation(program, n); cache.set(program, L); }
  const s = globalThis.__oceanSky;
  if (L.uSkyHorM) gl.uniform3fv(L.uSkyHorM, s ? s.reflHor : [1, 1, 1]);
  if (L.uSkyZenM) gl.uniform3fv(L.uSkyZenM, s ? s.reflZen : [1, 1, 1]);
  if (L.uGlitM) gl.uniform3fv(L.uGlitM, s ? s.glit : [1, 1, 1]);
  if (L.uBedS) gl.uniform1f(L.uBedS, state.size || 250);
  if (L.uSunC) gl.uniform3fv(L.uSunC, s && s.sunC ? s.sunC : [1, 1, 1]);
  if (L.uWarm) gl.uniform4fv(L.uWarm, s && s.warm ? s.warm : [1, 1, 1, 0]);
  clarK = 22.3024 / Math.min(40, Math.max(3, state.clarity ?? 22.3024)); if (L.uClarK) gl.uniform1f(L.uClarK, clarK);
  if (L.uRip) { setRippleScale(state.size || 250); const r = rippleUniforms(state.fx ? state.fx.t : 0); gl.uniform4fv(L.uRip, r.pack); gl.uniform4f(L.uRipP, r.p[0], r.p[1], r.p[2], r.p[3]); }
}
let clarK = 1;
export function setBedOnly(gl, program, size) { let L = cache.get(program); if (!L) { L = { uBedS: gl.getUniformLocation(program, "uBedS"), uClarK: gl.getUniformLocation(program, "uClarK") }; cache.set(program, L); } if (L.uBedS) gl.uniform1f(L.uBedS, size || 250); if (L.uClarK) gl.uniform1f(L.uClarK, clarK); }

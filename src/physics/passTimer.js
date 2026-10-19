// Per-pass GPU timing (v27.2), with ?passTimes=1. EXT_disjoint_timer_query_webgl2 when the browser has it (real
// GPU ms, read back a few frames later, never blocking). Otherwise, only with ?passTimes=1,
// gl.finish() around each pass (wall ms; stalls the pipeline, for diagnosis only).
// Results: passTimes() -> { name: { ms (mean), n, mode } }, reset on read.
const stats = new Map();
let ext = null, gl = null, mode = "off";
const pending = [];
export function initPassTimer(context) {
  if (gl) return;
  gl = context;
  ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  const forced = typeof location !== "undefined" && /[?&]passTimes=(1|finish)\b/.test(location.search);
  // v28: ?passTimes=finish forces the wall-clock mode even with the timer extension. With the
  // GPU saturated (by the render), a TIME_ELAPSED query can include queueing behind earlier
  // work; finish drains the pipeline before and after each pass, so each number is that
  // pass alone (plus a small fixed sync cost; subtract the "empty" row).
  const finishForced = typeof location !== "undefined" && /[?&]passTimes=finish\b/.test(location.search);
  // Only with ?passTimes=1: perf.js times the whole step with the same query type otherwise,
  // and WebGL allows one TIME_ELAPSED query at a time.
  mode = !forced ? "off" : ext && !finishForced ? "timer" : "finish";
}
function add(name, ms) { const s = stats.get(name) || { sum: 0, n: 0 }; s.sum += ms; s.n += 1; stats.set(name, s); }
function poll() {
  while (pending.length) {
    const [q, name] = pending[0];
    if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
    pending.shift();
    if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) add(name, gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
    gl.deleteQuery(q);
  }
}
// v28: gl.finish() does not wait in Chrome; a 1x1 readPixels from a scratch target does (it
// cannot return before every earlier command has run). The caller's framebuffer is restored.
let syncFb = null; const px = new Uint8Array(4);
function drain() {
  if (!syncFb) {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, 1);
    syncFb = gl.createFramebuffer();
    const prevR = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, syncFb);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prevR);
  }
  const prev = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, syncFb);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prev);
}
export function timed(name, draw) {
  if (mode === "off") return draw();
  if (mode === "timer") {
    if (pending.length > 512) return draw();
    const q = gl.createQuery();
    gl.beginQuery(ext.TIME_ELAPSED_EXT, q); draw(); gl.endQuery(ext.TIME_ELAPSED_EXT);
    pending.push([q, name]); return;
  }
  drain(); const t = performance.now(); draw(); drain(); add(name, performance.now() - t);
}
// The sync cost of an empty timed() call in finish mode (the floor under every pass).
export function timeEmpty() { if (mode === "finish") timed("empty", () => {}); }
export function passTimes() {
  if (mode === "timer") poll();
  const out = { mode };
  for (const [k, s] of stats) out[k] = { ms: +(s.sum / s.n).toFixed(3), n: s.n };
  stats.clear();
  return out;
}

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
  const forced = typeof location !== "undefined" && /[?&]passTimes=1/.test(location.search);
  // Only with ?passTimes=1: perf.js times the whole step with the same query type otherwise,
  // and WebGL allows one TIME_ELAPSED query at a time.
  mode = !forced ? "off" : ext ? "timer" : "finish";
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
export function timed(name, draw) {
  if (mode === "off") return draw();
  if (mode === "timer") {
    if (pending.length > 512) return draw();
    const q = gl.createQuery();
    gl.beginQuery(ext.TIME_ELAPSED_EXT, q); draw(); gl.endQuery(ext.TIME_ELAPSED_EXT);
    pending.push([q, name]); return;
  }
  gl.finish(); const t = performance.now(); draw(); gl.finish(); add(name, performance.now() - t);
}
export function passTimes() {
  if (mode === "timer") poll();
  const out = { mode };
  for (const [k, s] of stats) out[k] = { ms: +(s.sum / s.n).toFixed(3), n: s.n };
  stats.clear();
  return out;
}

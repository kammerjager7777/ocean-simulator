// ?perf=1 frame timer: for 10 s records, per frame, the frame time (rAF to rAF), the
// physics time on the main thread (the step call) and the whole frame callback's main-
// thread time, plus the physics' own time on the card where the browser offers a GPU timer
// (EXT_disjoint_timer_query_webgl2). Then shows median, 95th percentile and max on screen
// and as window.__perf, for a headless browser to read.
const SECONDS = 10;
// v27.1: the first second after the first frame (shader compiles, texture uploads, first
// fences) is startup, not steady state: not recorded.
const WARMUP = 1;
// v27.1: a run always ends with a __perf line. If the 10 s window has not closed after
// WATCHDOG s of wall clock (loop stopped, tab throttled, page error), it prints what it has
// with done:false and the reason; page errors are counted and the first ones kept.
const WATCHDOG = 30;

function stats(values) {
  if (!values.length) return null;
  const sorted = Float64Array.from(values).sort();
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))];
  const round = (v) => Math.round(v * 1000) / 1000;
  return { median: round(at(0.5)), p95: round(at(0.95)), max: round(sorted[sorted.length - 1]), count: sorted.length };
}

export function createPerf(gl, physicsMode) {
  const frameMs = [];
  const physicsMs = [];
  const mainMs = [];
  const gpuMs = [];
  const lagMs = [];
  let passFn = null;
  const PASS_TIMES = typeof location !== "undefined" && /[?&]passTimes=(1|finish)\b/.test(location.search);
  let heldFrames = 0;
  let simFirst = null;
  let simLast = null;
  const timer = gl && gl.getExtension ? gl.getExtension("EXT_disjoint_timer_query_webgl2") : null;
  const pending = [];
  let open = null;
  let first = null;
  let start = null;
  let previous = null;
  let done = false;
  const box = document.createElement("pre");
  box.id = "perf";
  box.style.cssText = "position:fixed;top:8px;right:8px;margin:0;padding:8px 10px;background:rgba(0,0,0,.72);color:#fff;font:12px/1.4 monospace;z-index:10;pointer-events:none";
  box.textContent = `perf: recording ${SECONDS} s (${physicsMode} physics)...`;
  document.body.appendChild(box);
  window.__perf = { done: false, physics: physicsMode, seconds: SECONDS };
  const errors = [];
  const onError = (message) => { if (errors.length < 5) errors.push(String(message).slice(0, 300)); errors.count = (errors.count || 0) + 1; };
  addEventListener("error", (e) => onError(e.error && e.error.stack ? e.error.stack : e.message));
  addEventListener("unhandledrejection", (e) => onError(e.reason && e.reason.stack ? e.reason.stack : e.reason));
  let frames = 0;
  setTimeout(() => { if (!done) finish(`watchdog: window not closed after ${WATCHDOG} s wall clock (${frames} frames seen)`); }, WATCHDOG * 1000);
  document.addEventListener("visibilitychange", () => { if (document.hidden) onError("page hidden during the run (rAF throttled)"); });

  function collect() {
    while (pending.length) {
      const q = pending[0];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      const disjoint = gl.getParameter(timer.GPU_DISJOINT_EXT);
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT);
      if (!disjoint) gpuMs.push(ns / 1e6);
      gl.deleteQuery(q);
      pending.shift();
    }
  }
  return {
    beginPhysics() {
      if (PASS_TIMES) return; // per-pass queries own the timer (passTimer.js)
      if (!timer || done || open) return;
      open = gl.createQuery();
      gl.beginQuery(timer.TIME_ELAPSED_EXT, open);
    },
    endPhysics() {
      if (!open) return;
      gl.endQuery(timer.TIME_ELAPSED_EXT);
      pending.push(open);
      open = null;
    },
    record(now, physics, main, extra = {}) {
      if (done) return;
      if (extra.passTimes) passFn = extra.passTimes;
      frames += 1;
      if (timer) collect();
      if (start === null) start = now;
      if ((now - start) / 1000 < WARMUP) { gpuMs.length = 0; previous = now; if (passFn) passFn(); return; }
      if (first === null) { first = now; simFirst = extra.simTime; }
      simLast = extra.simTime;
      if (extra.held) heldFrames += 1;
      if (extra.lags) lagMs.push(...extra.lags);
      if (previous !== null) frameMs.push(now - previous);
      previous = now;
      physicsMs.push(physics);
      mainMs.push(main);
      const elapsed = (now - first) / 1000;
      window.__perf.elapsed = elapsed;
      if (elapsed < SECONDS) return;
      finish(null);
    },
  };
  function finish(reason) {
      done = true;
      const result = {
        done: !reason,
        reason: reason || undefined,
        warmupSeconds: WARMUP,
        errors: errors.length ? { count: errors.count, first: errors.slice() } : undefined,
        physics: physicsMode,
        seconds: SECONDS,
        frames: frameMs.length,
        frameMs: stats(frameMs),
        physicsMs: stats(physicsMs),
        mainThreadMs: stats(mainMs),
        gpuPhysicsMs: timer ? stats(gpuMs) : null,
        // v27.1: frames the backlog cap skipped step(); wall time minus sim time over the
        // window (what the 0.05 s dt cap and held frames dropped); panel-count lag.
        heldFrames,
        lostSimMs: first !== null && simFirst !== undefined && simFirst !== null ? Math.round((previous - first) - 1000 * (simLast - simFirst)) : null,
        lagMs: stats(lagMs),
        // v27.2: mean ms per GPU pass per step (?passTimes=1; mode timer = GPU timer, finish = wall)
        gpuPasses: passFn ? passFn() : null,
        series: { frameMs: frameMs.map((v) => Math.round(v * 100) / 100), physicsMs: physicsMs.map((v) => Math.round(v * 100) / 100), mainThreadMs: mainMs.map((v) => Math.round(v * 100) / 100) },
      };
      window.__perf = result;
      // One line a shell can catch (Chrome --enable-logging=stderr), no driver needed.
      console.log("__perf " + JSON.stringify({ ...result, series: undefined }));
      const line = (name, s) => (s ? `${name.padEnd(16)} median ${s.median.toFixed(2)}  p95 ${s.p95.toFixed(2)}  max ${s.max.toFixed(2)} ms` : `${name.padEnd(16)} n/a`);
      box.textContent = [
        `perf over ${SECONDS} s, ${result.frames} frames, ${physicsMode} physics`,
        line("frame", result.frameMs),
        line("physics (CPU)", result.physicsMs),
        line("main thread", result.mainThreadMs),
        line("physics (GPU)", result.gpuPhysicsMs),
        line("panel lag", result.lagMs),
        `held frames ${heldFrames}, sim time lost ${result.lostSimMs} ms`,
        ...(result.gpuPasses && result.gpuPasses.mode !== "off" ? ["GPU passes (" + result.gpuPasses.mode + "): " + Object.entries(result.gpuPasses).filter(([k]) => k !== "mode").map(([k, v]) => `${k} ${v.ms}`).join(", ")] : []),
        ...(reason ? ["NOT COMPLETE: " + reason] : []),
        ...(errors.length ? [`page errors: ${errors.count} (first: ${errors[0].split("\n")[0]})`] : []),
      ].join("\n");
  }
}

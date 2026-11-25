import { GRAVITY_NOTE } from "./physics/gravity.js";
import { scenario } from "./physics/scenario.js";
import { createWind } from "./physics/wind.js";
import { createSurface } from "./physics/surface.js";
import { createView } from "./render/view.js";
import { createIslandGpu } from "./physics/islandGpu.js";
import { useIslandSize, stockdonR2 } from "./physics/island.js";
import { parseShare, applyShareToInputs, islandRedirect } from "./ui/v23/share.js";
import { sunVector } from "./ui/v23/sky.js";
import { seaSteepness, whySteepness } from "./physics/steepness.js";
import { ui } from "./ui/v23/core.js";
import { createGpuSurface, gpuPhysicsProblem } from "./physics/gpuSurface.js";
import { createBoat } from "./physics/boat.js";
import { createPerf } from "./perf.js";
import { rippleHeight, prune as pruneRipples } from "./physics/ripples.js";

// ?chopSpectrum=pm (or =elfouhaily) picks the chop band's spectrum for A/B comparison.
// It only changes that band's scenario setting; nothing else reads it.
if (typeof location !== "undefined") {
  const next = islandRedirect(location.href);
  if (next && next !== location.href) location.replace(next);
}
const shareBoot = typeof location !== "undefined" ? parseShare(location.search) : null;
const chopSpectrum = typeof location !== "undefined" ? new URLSearchParams(location.search).get("chopSpectrum") : null;
if (chopSpectrum === "pm" || chopSpectrum === "elfouhaily") {
  for (const band of scenario.bands) if (band.id === "chop") band.spectrum = chopSpectrum;
}
const params = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const wind = createWind({ direction: Math.PI / 2, speed: 12, gust: "continuous" });
globalThis.__waves = shareBoot && shareBoot.waves === "hybrid" ? "hybrid" : "spectral";
if (shareBoot) {
  applyShareToInputs(shareBoot);
  wind.speed = shareBoot.wind;
  wind.direction = (shareBoot.dir * Math.PI) / 180;
  scenario.size = shareBoot.size;
  scenario.seaAge = shareBoot.age;
}
// ?fixtest=1: developer fixture only. The first step is already flat (no wind-12 startup).
const FIXTEST = typeof location !== "undefined" && new URLSearchParams(location.search).get("fixtest") === "1";
if (FIXTEST) { wind.speed = 0; scenario.choppiness = 0; }
const canvas = document.querySelector("#sea");
const view = createView(canvas);
// Physics on the graphics card by default (gpuSurface.js); ?physics=cpu runs the v26 CPU
// physics (surface.js). If the card cannot run it, say so and use the CPU.
let physicsMode = params.get("physics") === "cpu" ? "cpu" : "gpu";
let physicsNote = "";
let surface = null;
if (physicsMode === "gpu") {
  const problem = gpuPhysicsProblem(view.gl);
  if (!problem) {
    try {
      surface = createGpuSurface(view.gl, scenario, wind);
    } catch (error) {
      physicsNote = String(error.message || error);
    }
  } else physicsNote = "GPU physics unavailable: " + problem;
  if (!surface) {
    physicsMode = "cpu";
    physicsNote += "; running the CPU physics instead.";
    console.error(physicsNote);
  }
}
if (!surface) surface = createSurface(scenario, wind);
if (typeof window !== "undefined") window.__surface = surface;
if (typeof window !== "undefined") { window.__physics = { mode: physicsMode, note: physicsNote }; window.__surface = surface; }
const perf = params.get("perf") === "1" ? createPerf(view.gl, physicsMode) : null;

const dir = document.querySelector("#dir");
const spd = document.querySelector("#spd");
const gust = document.querySelector("#gust");
const depthInput = document.querySelector("#depth");
const chopInput = document.querySelector("#chop");
if (FIXTEST) { spd.value = "0"; chopInput.value = "0"; wind.speed = 0; scenario.choppiness = 0; }
const sizeInput = document.querySelector("#size");
const flags = document.querySelector("#flags");
const pause = document.querySelector("#pause");
const boatOn = document.querySelector("#boat");
// v30.4 water clarity slider (1 clear tropical .. 0 murky coastal); ?clarity= sets its start
const clarityIn = document.querySelector("#clarity");
// v30.23: #clarity holds the Secchi depth Zsd in metres (3..40; default 1.7 / 0.076225 = 22.30 m, the v30.8 K).
// ?clarity= in 0..1 is the legacy slider (1 clear .. 0 murky) and maps through the v30.8 K; > 1 is read as Zsd metres.
if (clarityIn && params.get("clarity") !== null && Number.isFinite(Number(params.get("clarity")))) { const c = Number(params.get("clarity")); clarityIn.value = String(c <= 1 ? 1.7 / (0.076225 + 0.0785 * (1 - Math.max(0, c))) : Math.min(40, Math.max(3, c))); }
const clarityVal = document.querySelector("#clarity-val");
const clarityLabel = () => { if (clarityVal) { const c = Number(clarityIn.value); clarityVal.textContent = c >= 30 ? "clear tropical" : c >= 15 ? "open ocean" : c >= 6 ? "coastal" : "murky"; } };
if (clarityIn) { clarityIn.addEventListener("input", clarityLabel); clarityLabel(); }
const boat = createBoat();
document.querySelector("#gravity").textContent = GRAVITY_NOTE;

let depth = scenario.depth;
let last = performance.now();
// Sim time is the sum of the steps taken, not the wall clock, so pausing holds one
// moment: the same sea, redrawn, while flags or the camera change.
let simTime = 0;
let state = null;
const rippled = new WeakSet(); let pts0 = null; // v30.23 ripple heights already added to these readbacks

function label(id, text) {
  document.querySelector(id).textContent = text;
}

dir.addEventListener("input", () => {
  wind.direction = (Number(dir.value) * Math.PI) / 180;
  label("#dir-val", `${dir.value}°`);
});
spd.addEventListener("input", () => {
  wind.speed = Number(spd.value);
  label("#spd-val", `${Number(spd.value).toFixed(1)} m/s`);
});
gust.addEventListener("change", () => {
  wind.gust = gust.value;
});
depthInput.addEventListener("input", () => {
  depth = Number(depthInput.value);
  label("#depth-val", `${depthInput.value} m`);
});
const ageInput = document.querySelector("#age");
const curInput = document.querySelector("#cur");
let chopOverride = params.has("chop") || !!(shareBoot && shareBoot.override);
function setCurrentAlong(signed) {
  scenario.current = Math.abs(signed);
  scenario.currentDir = wind.direction + (signed < 0 ? Math.PI : 0);
}
function applySteepness() {
  if (ageInput) scenario.seaAge = Number(ageInput.value);
  if (curInput) setCurrentAlong(Number(curInput.value));
  const st = seaSteepness({
    windMs: wind.speed,
    age: scenario.seaAge,
    currentMs: scenario.current,
    currentDir: scenario.currentDir,
    windDir: wind.direction,
    depthM: depth,
  });
  scenario.label = st.kind;
  if (!chopOverride) {
    const v = st.chop.toFixed(1);
    scenario.choppiness = st.chop;
    scenario.steepnessS = st.S;
    if (chopInput.value !== v) chopInput.value = v;
    label("#chop-val", v);
  } else {
    scenario.choppiness = Number(chopInput.value);
    scenario.steepnessS = scenario.choppiness / 2.5;
  }
  scenario.steepnessWhy = whySteepness(st, chopOverride);
  window.__steep = { ...st, override: chopOverride, S: scenario.steepnessS, why: scenario.steepnessWhy, chop: scenario.choppiness };
}
chopInput.addEventListener("input", () => {
  chopOverride = true;
  scenario.choppiness = Number(chopInput.value);
  scenario.steepnessS = scenario.choppiness / 2.5;
  label("#chop-val", Number(chopInput.value).toFixed(1));
});
ui.clearChopOverride = () => {
  chopOverride = false;
  applySteepness();
};
if (ageInput) ageInput.addEventListener("input", () => { scenario.seaAge = Number(ageInput.value); });
if (curInput) curInput.addEventListener("input", () => { setCurrentAlong(Number(curInput.value)); });
sizeInput.addEventListener("input", () => {
  scenario.size = Number(sizeInput.value);
  label("#size-val", `${sizeInput.value} m`);
});

function readout(state) {
  const heading = Math.round(((Math.atan2(state.travelZ, state.travelX) * 180) / Math.PI + 360) % 360);
  label("#readout", `${state.breakingCount} samples breaking now; ${state.foamCount} carry foam from breaks in the last ${Number(state.foamWindow).toFixed(1)} s. Waves travel toward ${heading}°. Foam sits on the front of each breaking wave and stays where it was laid, fading out over ${Number(state.foamWindow).toFixed(1)} s.`);
}
// GPU counts arrive a moment after their step: the panel takes them when they land.
if (surface.kind === "gpu") surface.onCounts = (s) => { if (s === state) readout(s); };

// v28 perf (Mac, 120 Hz): a step's fence signals only once Chrome's GPU process has run that frame,
// 2-3 display frames later at 120 Hz, so a backlog limit of 2 steps held frames on a card that was
// keeping up. 6 steps (50 ms at 120 Hz, 100 ms at 60 Hz) still stops a slow card's queue growing.
const MAX_BACKLOG = Number(params.get("maxbacklog") || 6);
const noDraw = params.get("nodraw") === "1";
// ?fixdt=<s> (headless measurement): fixed physics step, so sim time hits exact sample times
const fixDt = Number(params.get("fixdt") || 0);
// ?island=1 only. The shallow-water island is not built otherwise, and the
// sea shader keeps uIsland at 0 so the open ocean is the V31.9 surface.
function islandGlProblem(gl) {
  if (!gl) return "no WebGL2 context";
  if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) return "not WebGL2";
  if (!gl.getExtension("EXT_color_buffer_float")) return "no float color buffer";
  return "";
}
const islandOn = params.get("island") === "1";
let islandGpu = null;
if (islandOn && view.gl && !islandGlProblem(view.gl)) {
  try { islandGpu = createIslandGpu(view.gl, scenario.depth, scenario.size); }
  catch (err) { console.error(err); }
}
function frame(now) {
  // Next frame first: an exception in this one must not stop the loop (v27.1; a stopped
  // loop is why a ?perf=1 run could end without its __perf line).
  requestAnimationFrame(frame);
  const dt = fixDt > 0 ? fixDt : Math.min(0.05, (now - last) / 1000);
  applySteepness();
  const waitCard = surface.backlog && state && !pause.checked && surface.backlog() > MAX_BACKLOG;
  if (!waitCard) last = now;
  const t0 = performance.now();
  if (perf) perf.beginPhysics();
  // v27.1: never queue more than MAX_BACKLOG steps ahead of the card. A GPU slower than the
  // display would otherwise build an ever longer command queue (counts seconds late, and
  // the browser stalls the page when it finally blocks). Waiting a frame keeps last's time,
  // so the next step takes it (dt still capped at 0.05 s).
  const held = waitCard;
  if (held) {
    // wait for the card; 'last' is not advanced, so this frame's time goes to the next step
  } else if (!state || !pause.checked) {
    simTime += dt;
    state = surface.step(simTime, dt, depth);
  } else if (surface.refresh) {
    // Paused: the sea holds; the panel picks up counts the card has finished since.
    surface.refresh(state);
  }
  if (perf) perf.endPhysics();
  const t1 = performance.now();
  // ?nodraw=1 (headless capture): step the physics without drawing; draw only while paused
  // or on the frame a capture harness holds at (window.__holdTargets). The physics never
  // reads the drawing, and each step takes a fixed dt, so the state at a sim time is the
  // same as when every frame is drawn.
  const capT = window.__holdTargets;
  // v30b spray/foam drift: the view needs the sim time and the wind (drawing only).
  if (state) state.fx = { t: simTime, speed: wind.speed, dir: wind.direction };
  // v30.4 boat: buoyancy from the GPU height probe (a step or two old), drawn at true 8 m scale
  if (state && surface.setProbe) {
    const on = !boatOn || boatOn.checked;
    // v30.23: the boat bobs on the ripple-touch / buoy-wake rings too (added once to each new readback)
    const pdat = state.probe && state.probe.data;
    if (pdat && !rippled.has(pdat) && pts0) { rippled.add(pdat); for (let k = 0; k < 16; k++) pdat[k * 4] += rippleHeight(pts0[k * 2], pts0[k * 2 + 1], simTime); }
    if (!pause.checked) pruneRipples(simTime);
    if (on && !held && !pause.checked) boat.step(dt, state.probe && state.probe.data, wind);
    if (islandOn) useIslandSize(scenario.size);
    boat.place(scenario.size, depth, islandOn);
    pts0 = on ? boat.probePoints() : null;
    surface.setProbe(on ? boat.probePoints() : null);
    state.boat = on ? boat : null;
    state.clarity = clarityIn ? Number(clarityIn.value) : 22.3024; // Secchi depth, m
  }
  if (islandGpu && state) {
    if (islandGpu.sync) islandGpu.sync(depth, scenario.size);
    const todMin = globalThis.__oceanTodMin;
    // Geometric sun by day. At night that vector is below the horizon and the
    // bake would flatten the island, so the moon (opposite the sun) casts it.
    let sun = Number.isFinite(todMin) ? sunVector(todMin) : globalThis.__oceanSun;
    if (sun && sun[1] <= 0.03 && Number.isFinite(todMin)) sun = sunVector(todMin + 720);
    if (sun) globalThis.__oceanShadeSun = sun;
    if (sun) {
      const prev = islandGpu._sun;
      if (!prev || Math.hypot(sun[0] - prev[0], sun[1] - prev[1], sun[2] - prev[2]) > 0.05) {
        islandGpu.updateShadow(sun);
        islandGpu._sun = [sun[0], sun[1], sun[2]];
      }
    }
    if (state.gpu && state.gpu.A && !held && !pause.checked) {
      // Advance the same simulation step as the sea, in CFL-stable pieces.
      // At h ≈ 8 m, c ≈ 12 m/s and dx ≈ 0.89 m, a 0.03 s step is the limit.
      // A 60 fps frame needs one; a long capture frame (dt capped at 0.05 s) needs two.
      const seaDt = fixDt > 0 ? fixDt : Math.min(0.05, dt);
      const cMax = Math.sqrt(9.81 * 8) + 3;
      const dtCfl = 0.4 * islandGpu.dx / cMax;
      const nCfl = Math.max(1, Math.ceil(seaDt / Math.max(dtCfl, 1e-4)));
      const asked = Number(params.get("swashsteps"));
      const nSub = Number.isFinite(asked) && asked > 0 ? Math.max(nCfl, Math.round(asked)) : nCfl;
      const sub = seaDt / nSub;
      globalThis.__swashCfl = { seaDt, dtCfl, nSub, sub, dx: islandGpu.dx };
      const dir = state.fx ? state.fx.dir : Math.PI / 2;
      let hsNow = 0;
      if (state.bands) for (const band of state.bands) hsNow += (band.significantHeight || 0) ** 2;
      hsNow = Math.sqrt(hsNow) || 0.4;
      let tp = 5;
      const wb = scenario.bands.find((b) => b.id === "wind-sea");
      if (wb) tp = 6.2831853 / ((wb.omegaMin + wb.omegaMax) * 0.5);
      tp *= 0.75 + (Number(scenario.seaAge) || 0);
      for (let s = 0; s < nSub; s += 1) islandGpu.step(sub, state.gpu.A, state.gpu.fft.L, Math.cos(dir), Math.sin(dir), state.fx ? state.fx.speed : 12, hsNow, tp);
    }
    let hs = 0;
    if (state.bands) for (const band of state.bands) hs += (band.significantHeight || 0) ** 2;
    hs = Math.sqrt(hs);
    const shore = islandGpu.profile || useIslandSize(scenario.size);
    let tpR = 5;
    const wR = scenario.bands.find((b) => b.id === "wind-sea");
    if (wR) tpR = 6.2831853 / ((wR.omegaMin + wR.omegaMax) * 0.5);
    tpR *= 0.75 + (Number(scenario.seaAge) || 0);
    const r2 = stockdonR2(hs || 0.4, tpR, shore.faceSlope);
    globalThis.__swashCap = { r2, hs, tp: tpR, dx: islandGpu.dx, n: islandGpu.nx, slope: shore.faceSlope };
    state.island = {
      on: 1, tex: islandGpu.texture(),
      cx: shore.cx, cz: shore.cz,
      ox: shore.mapOriginX, oz: shore.mapOriginZ,
      mapSpan: shore.mapSpan, mapN: shore.mapN, atlasH: islandGpu.atlasH || (shore.mapN + islandGpu.ny), atlasW: islandGpu.atlasW || shore.mapN, r2,
      dLo: shore.dLo, dHi: shore.dHi,
      sweN: islandGpu.nx, sweSpan: shore.span,
      sweOx: shore.cx - shore.span / 2, sweOz: shore.cz - shore.span / 2,
      hs,
      Rbeach: shore.Rbeach, Rbank: shore.Rbank, peak: shore.peak,
    };
    globalThis.__islandGpu = islandGpu;
    if (params.get("shore") === "1" && (!islandGpu._mt || performance.now() > islandGpu._mt)) {
      islandGpu._mt = performance.now() + 2500;
      const g = state.gpu;
      if (g && g.A && g.fft) {
        try {
          const m = islandGpu.measures(g.A, g.fft.L, g.fft.N, Math.cos(state.fx ? state.fx.dir : 0), Math.sin(state.fx ? state.fx.dir : 0));
          globalThis.__shore = { runup: m.runup, inland: m.inland, wetMax: m.wetMax, hs, t: simTime, dx: islandGpu.dx, n: islandGpu.nx, az: m.az, leeInlandM: m.leeInlandM, windwardInlandM: m.windwardInlandM };
        } catch (err) { console.error(err); }
      }
    }
  }
  if (!noDraw || pause.checked || (capT && capT.length && simTime >= capT[0] - 1e-6)) view.draw(state, flags.checked);
  const fine = state.chopPatch && state.chopPatch.modes
    ? ` · Fine chop on its own ${state.chopPatch.size} m patch (${state.chopPatch.longest.toFixed(1)} to ${state.chopPatch.shortest.toFixed(1)} m waves): ${state.chopPatch.significantHeight.toFixed(2)} m`
    : "";
  const finer = state.finePatch && state.finePatch.modes
    ? ` · Finer chop on its own ${state.finePatch.size} m patch (${state.finePatch.longest.toFixed(2)} to ${state.finePatch.shortest.toFixed(2)} m waves): ${state.finePatch.significantHeight.toFixed(2)} m`
    : "";
  label("#bands", state.bands.map((band) => `${band.label}${band.spectrum === "elfouhaily" ? " (Elfouhaily)" : ""}: ${band.significantHeight.toFixed(2)} m`).join(" · ") + fine + finer);
  readout(state);
  label("#clock", `Sim time ${simTime.toFixed(2)} s${pause.checked ? " (paused)" : ""}`);
  label("#camera", `Camera height ${view.cameraHeight().toFixed(1)} m above the sea`);
  // v29.2 glass UI: the stats strip reads the latest state and held-frame count (read-only).
  window.__oceanState = state; if (held) window.__oceanHeld = (window.__oceanHeld || 0) + 1;
  if (perf) perf.record(now, t1 - t0, performance.now() - t0, { held: !!waitCard, simTime, lags: surface.takeLags ? surface.takeLags() : [], passTimes: surface.passTimes });
}

if (physicsNote) {
  const note = document.createElement("p");
  note.id = "physics-note";
  note.textContent = physicsNote;
  note.style.cssText = "color:#a33;font-weight:600";
  document.querySelector("#readout").before(note);
}

// ?parity=1: CPU-vs-GPU parity test on this page's context (tests only; see parity.js).
if (params.get("parity") === "1") {
  import("./physics/parity.js").then(({ runParity }) => runParity(view.gl, {
    speeds: (params.get("speeds") || "0,5,10,17.5,25").split(",").map(Number),
    seconds: Number(params.get("seconds") || 6),
    gust: params.get("gust") || "continuous",
  })).then((result) => {
    window.__parity = result;
    document.querySelector("#readout").textContent = result.summary;
  }, (error) => {
    window.__parity = { ok: false, error: String(error && error.stack || error) };
    document.querySelector("#readout").textContent = "parity failed: " + error;
  });
} else {
  requestAnimationFrame(frame);
}

// ?wind=17.5 sets the wind slider before the first step (for ?perf=1 runs from a shell).
if (params.get("wind") !== null && Number.isFinite(Number(params.get("wind")))) {
  const el = document.querySelector("#spd");
  el.value = String(Number(params.get("wind")));
  el.dispatchEvent(new Event("input", { bubbles: true }));
}
if (params.get("age") !== null && ageInput && Number.isFinite(Number(params.get("age")))) ageInput.value = String(Number(params.get("age")));
if (params.get("cur") !== null && curInput && Number.isFinite(Number(params.get("cur")))) curInput.value = String(Number(params.get("cur")));
if (params.get("chop") !== null && Number.isFinite(Number(params.get("chop")))) {
  chopOverride = true;
  chopInput.value = String(Number(params.get("chop")));
}

import { GRAVITY_NOTE } from "./physics/gravity.js";
import { scenario } from "./physics/scenario.js?v=30.22";
import { createWind } from "./physics/wind.js";
import { createSurface } from "./physics/surface.js?v=30.22";
import { createView } from "./render/view.js?v=32.1";
import { createIslandGpu } from "./physics/islandGpu.js?v=32";
import { ISLAND } from "./physics/island.js?v=32";
import { seaSteepness, whySteepness } from "./physics/steepness.js";
import { createGpuSurface, gpuPhysicsProblem } from "./physics/gpuSurface.js?v=31.8";
import { createBoat } from "./physics/boat.js?v=30.22";
import { createPerf } from "./perf.js?v=30.22";
import { rippleHeight, prune as pruneRipples } from "./physics/ripples.js?v=30.29";

// ?chopSpectrum=pm (or =elfouhaily) picks the chop band's spectrum for A/B comparison.
// It only changes that band's scenario setting; nothing else reads it.
const chopSpectrum = typeof location !== "undefined" ? new URLSearchParams(location.search).get("chopSpectrum") : null;
if (chopSpectrum === "pm" || chopSpectrum === "elfouhaily") {
  for (const band of scenario.bands) if (band.id === "chop") band.spectrum = chopSpectrum;
}
const params = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const wind = createWind({ direction: Math.PI / 2, speed: 12, gust: "continuous" });
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
let chopOverride = params.has("chop");
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
  try { islandGpu = createIslandGpu(view.gl, scenario.depth); }
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
    pts0 = on ? boat.probePoints() : null;
    surface.setProbe(on ? boat.probePoints() : null);
    state.boat = on ? boat : null;
    state.clarity = clarityIn ? Number(clarityIn.value) : 22.3024; // Secchi depth, m
  }
  if (islandGpu && state && state.gpu && state.gpu.A && !held && !pause.checked) {
    const step = fixDt > 0 ? fixDt : Math.min(0.02, dt);
    const nSub = Math.max(1, Math.round(step / 0.02));
    const sub = step / nSub;
    const dir = state.fx ? state.fx.dir : Math.PI / 2;
    for (let s = 0; s < nSub; s += 1) islandGpu.step(sub, state.gpu.A, state.gpu.fft.L, Math.cos(dir), Math.sin(dir));
    state.island = {
      on: 1, tex: islandGpu.texture(), cx: ISLAND.cx, cz: ISLAND.cz, span: ISLAND.span,
      Rbeach: ISLAND.Rbeach, Rbank: ISLAND.Rbank, peak: ISLAND.peak, shelf: ISLAND.shelf,
      dLo: ISLAND.dLo, dHi: ISLAND.dHi,
    };
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
  import("./physics/parity.js?v=30.22").then(({ runParity }) => runParity(view.gl, {
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

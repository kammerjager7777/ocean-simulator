import { GRAVITY_NOTE } from "./physics/gravity.js";
import { scenario } from "./physics/scenario.js?v=28";
import { createWind } from "./physics/wind.js";
import { createSurface } from "./physics/surface.js?v=28";
import { createView } from "./render/view.js?v=28";
import { createGpuSurface, gpuPhysicsProblem } from "./physics/gpuSurface.js?v=28";
import { createPerf } from "./perf.js?v=28";

// ?chopSpectrum=pm (or =elfouhaily) picks the chop band's spectrum for A/B comparison.
// It only changes that band's scenario setting; nothing else reads it.
const chopSpectrum = typeof location !== "undefined" ? new URLSearchParams(location.search).get("chopSpectrum") : null;
if (chopSpectrum === "pm" || chopSpectrum === "elfouhaily") {
  for (const band of scenario.bands) if (band.id === "chop") band.spectrum = chopSpectrum;
}
const params = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const wind = createWind({ direction: Math.PI / 2, speed: 12, gust: "continuous" });
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
const sizeInput = document.querySelector("#size");
const flags = document.querySelector("#flags");
const pause = document.querySelector("#pause");
document.querySelector("#gravity").textContent = GRAVITY_NOTE;

let depth = scenario.depth;
let last = performance.now();
// Sim time is the sum of the steps taken, not the wall clock, so pausing holds one
// moment: the same sea, redrawn, while flags or the camera change.
let simTime = 0;
let state = null;

function label(id, text) {
  document.querySelector(id).textContent = text;
}

dir.addEventListener("input", () => {
  wind.direction = (Number(dir.value) * Math.PI) / 180;
  label("#dir-val", `${dir.value}°`);
});
spd.addEventListener("input", () => {
  wind.speed = Number(spd.value);
  label("#spd-val", `${spd.value} m/s`);
});
gust.addEventListener("change", () => {
  wind.gust = gust.value;
});
depthInput.addEventListener("input", () => {
  depth = Number(depthInput.value);
  label("#depth-val", `${depthInput.value} m`);
});
chopInput.addEventListener("input", () => {
  scenario.choppiness = Number(chopInput.value);
  label("#chop-val", Number(chopInput.value).toFixed(1));
});
sizeInput.addEventListener("input", () => {
  scenario.size = Number(sizeInput.value);
  label("#size-val", `${sizeInput.value} m`);
});

function readout(state) {
  const heading = Math.round(((Math.atan2(state.travelZ, state.travelX) * 180) / Math.PI + 360) % 360);
  label("#readout", `${state.breakingCount} samples breaking now; ${state.foamCount} carry foam from breaks in the last ${state.foamWindow} s. Waves travel toward ${heading}°. Foam sits on the front of each breaking wave and stays where it was laid, fading out over ${state.foamWindow} s.`);
}
// GPU counts arrive a moment after their step: the panel takes them when they land.
if (surface.kind === "gpu") surface.onCounts = (s) => { if (s === state) readout(s); };

const MAX_BACKLOG = 2;
const noDraw = params.get("nodraw") === "1";
// ?fixdt=<s> (headless measurement): fixed physics step, so sim time hits exact sample times
const fixDt = Number(params.get("fixdt") || 0);
function frame(now) {
  // Next frame first: an exception in this one must not stop the loop (v27.1; a stopped
  // loop is why a ?perf=1 run could end without its __perf line).
  requestAnimationFrame(frame);
  const dt = fixDt > 0 ? fixDt : Math.min(0.05, (now - last) / 1000);
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
  import("./physics/parity.js?v=28").then(({ runParity }) => runParity(view.gl, {
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

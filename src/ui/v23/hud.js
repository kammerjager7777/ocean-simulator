// v30.23 HUD, the Hairline layer (modern-ui-spec §6): settings column (Open / Minimised / Layers sheet; H, Esc),
// pause button (Space), stats line (S, 2 Hz), version, toast. DOM only; the look is v23.css. The old glass panel
// and strip stay hidden in the DOM because their inputs (#boat, #flags, #pause) and stat ids stay the source of truth.
import { ui, LAYERS, store, $ } from "./core.js?v=31.9";
const hud = document.createElement("div"); hud.id = "v23-hud"; document.body.appendChild(hud);
const vg = document.createElement("div"); vg.id = "v23-vignette"; document.body.appendChild(vg);
const SVG_SET = `<svg width="20" height="20" viewBox="-10 -10 20 20" aria-hidden="true"><path class="ic" d="M-7 -4H7M-7 0H7M-7 4H7"/><circle class="dot" cx="-3" cy="-4" r="1.35"/><circle class="dot" cx="3" cy="0" r="1.35"/><circle class="dot" cx="-1" cy="4" r="1.35"/></svg>`;
const SVG_PAUSE = `<svg width="20" height="20" viewBox="-10 -10 20 20" aria-hidden="true"><path class="bars" d="M-3.2 -5.2V5.2M3.2 -5.2V5.2"/><path class="play" d="M-3.6 -5.4L6 0L-3.6 5.4Z"/></svg>`;
const CHEV = `<svg class="chev" viewBox="0 0 8 10" aria-hidden="true"><path d="M2 1.5L6 5L2 8.5"/></svg>`;
const BACK = `<svg class="chev" viewBox="0 0 8 10" aria-hidden="true" style="margin:0 6px 0 0"><path d="M6 1.5L2 5L6 8.5"/></svg>`;
const row = (id, name, right, extra = "") => `<div class="hrow" id="${id}" role="switch" tabindex="0" ${extra}><span class="nm l1"><span class="plate">${name}</span></span>${right}</div>`;
hud.innerHTML = `
  <button class="hb" id="hb-set" type="button" aria-label="Settings" title="Settings" aria-expanded="true">${SVG_SET}</button>
  <div id="hud-head"><div id="hud-title">Ocean Simulator</div><div id="hud-ver">V32 (dev)</div></div>
  <div id="hud-rows">
    ${row("hr-boat", "Boat", '<span class="tg"></span>')}
    ${row("hr-flags", "Break flags", '<span class="tg"></span>')}
    <div class="hrow" id="hr-layers" role="button" tabindex="0" aria-label="Layers"><span class="nm l1"><span class="plate">Layers</span></span><span class="cnt n4 tx plate" id="hr-cnt">4/7</span>${CHEV}</div>
    ${row("hr-clean", "Clean view", '<span class="tg"></span>')}
    ${row("hr-stats", "Stats", '<span class="tg"></span>')}
    <div class="hrow" id="hr-fit" role="button" tabindex="0"><span class="nm l1">Fit ocean</span></div>
    <div class="hrow" id="hr-reset" role="button" tabindex="0"><span class="nm l1">Reset view</span></div>
    <div class="hrows-end"></div>
  </div>
  <div id="hud-sheet" class="sheet-off">
    <div class="hrow" id="hs-back" role="button" tabindex="0" aria-label="Back"><span class="nm l1"><span class="plate">${BACK}Layers</span></span><span class="l2 plate" id="hs-hide">Hide all <span class="n4">U</span></span></div>
    ${LAYERS.map(([k, n, , d]) => `<div class="hrow" id="hl-${k}" role="switch" tabindex="0"><span class="dg n4 tx plate">${d}</span><span class="nm l1"><span class="plate">${n}</span></span><span class="tg"></span></div>`).join("")}
    <div class="hrows-end"></div>
  </div>
  <button class="hb" id="hb-pause" type="button">${SVG_PAUSE}</button>
  <div id="hud-pl"><span class="l1 plate" id="hud-pause-l">Pause</span><span class="n4 plate">Space</span></div>
  <div id="hud-stats" class="plate"><span class="l1 mu">FPS</span> <span class="n3" id="hs-fps">--</span><span class="sep">·</span><span class="l1 mu">Foam</span> <span class="n3" id="hs-foam">--</span><span class="sep">·</span><span class="n3 ac" id="hs-f">F-</span></div>
  <div id="hud-toast" class="plate" role="status" aria-live="polite"><span class="l1 tx" id="ht-n"></span><span class="st" id="ht-s"></span></div>`;
const q = (s) => hud.querySelector(s);
// ---------- panel state: open | min, plus the Layers sheet ----------
let panel = store.get("ocean.ui.panel", matchMedia("(max-width: 899px), (pointer: coarse)").matches ? "min" : "open") === "min" ? "min" : "open", sheet = false;
function syncPanel() {
  q("#hud-rows").classList.toggle("hidden-col", panel === "min"); q("#hud-rows").classList.toggle("rows-left", panel !== "min" && sheet);
  q("#hud-sheet").classList.toggle("sheet-off", panel === "min" || !sheet); q("#hb-set").setAttribute("aria-expanded", String(panel !== "min"));
}
const setPanel = (p) => { panel = p; store.set("ocean.ui.panel", p); if (p === "min") sheet = false; syncPanel(); };
q("#hb-set").addEventListener("click", () => setPanel(panel === "min" ? "open" : "min"));
ui.hudH = () => setPanel(panel === "min" ? "open" : "min");
ui.hudEsc = () => { if (sheet) { sheet = false; syncPanel(); return true; } return false; };
const act = (el, fn) => { el.addEventListener("click", fn); el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { fn(); e.preventDefault(); e.stopPropagation(); } }); };
act(q("#hr-layers"), () => { sheet = true; syncPanel(); });
act(q("#hs-back"), (e) => { sheet = false; syncPanel(); });
q("#hs-hide").addEventListener("click", (e) => { e.stopPropagation(); ui.setClean(!ui.clean); });
const flip = (inp) => { inp.checked = !inp.checked; inp.dispatchEvent(new Event("change", { bubbles: true })); syncHud(); };
act(q("#hr-boat"), () => flip(ui.inputs.boat)); act(q("#hr-flags"), () => flip(ui.inputs.flags));
act(q("#hr-clean"), () => ui.setClean(!ui.clean));
act(q("#hr-stats"), () => { window.__ui && window.__ui.setStripHidden(!document.body.classList.contains("ui-hidden-strip")); syncHud(); });
act(q("#hr-fit"), () => { const cmd = globalThis.__oceanViewCmd; if (cmd) cmd.fit(); });
act(q("#hr-reset"), () => { const cmd = globalThis.__oceanViewCmd; if (cmd) cmd.reset(); });
for (const [k] of LAYERS) act(q("#hl-" + k), () => ui.setLayer(k, !ui.layers[k]));
q("#hb-pause").addEventListener("click", () => ui.togglePause());
// ---------- sync ----------
const setOn = (el, on) => { el.classList.toggle("on", on); el.setAttribute("aria-checked", String(on)); };
function syncHud() {
  setOn(q("#hr-boat"), ui.inputs.boat.checked); setOn(q("#hr-flags"), ui.inputs.flags.checked); setOn(q("#hr-clean"), ui.clean);
  const statsOn = !document.body.classList.contains("ui-hidden-strip"); setOn(q("#hr-stats"), statsOn); hud.classList.toggle("nostats", !statsOn);
  const n = LAYERS.filter(([k]) => ui.layers[k]).length, cnt = q("#hr-cnt"); cnt.textContent = ui.clean ? "CLEAN" : `${n}/${LAYERS.length}`; // the count is always --hud-text (§9.3; accent "CLEAN" measured 4.09:1 at dusk)
  for (const [k] of LAYERS) setOn(q("#hl-" + k), ui.layers[k]);
  const p = ui.inputs.pause.checked; hud.classList.toggle("paused", p); const pb = q("#hb-pause"); pb.setAttribute("aria-pressed", String(p)); pb.setAttribute("aria-label", p ? "Resume simulation" : "Pause simulation");
  q("#hud-pause-l").textContent = p ? "Paused" : "Pause";
}
ui.syncHud = syncHud; ui.inputs.pause.addEventListener("change", syncHud);
new MutationObserver(syncHud).observe(document.body, { attributes: true, attributeFilter: ["class"] });
// ---------- toast (§6.5) ----------
let tt = 0;
ui.toast = (name, on) => {
  q("#ht-n").textContent = name; const s = q("#ht-s"); s.textContent = on ? "ON" : "OFF"; s.style.color = "var(--hud-text)"; // §9.2/9.3: accent / muted words measured < 4.5:1 at dusk; the accent tick under the toast carries the accent
  const t = q("#hud-toast"); t.classList.add("on"); clearTimeout(tt); tt = setTimeout(() => t.classList.remove("on"), 1200);
};
// ---------- stats (2 Hz) from the hidden strip's values ----------
const txt = (id) => { const e = document.getElementById(id); return e ? e.textContent : "--"; };
setInterval(() => {
  if (hud.classList.contains("nostats")) return;
  const put = (id, v) => { const e = q(id); if (e.textContent !== v) e.textContent = v; };
  put("#hs-fps", txt("st-fps")); const f = txt("st-foam"); put("#hs-foam", f === "--" ? "--" : `${f}%`); put("#hs-f", txt("st-sea"));
}, 500);
syncPanel(); syncHud();
if (new URLSearchParams(location.search).get("notext") === "1") document.documentElement.classList.add("notext");

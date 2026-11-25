// The right-hand cards are one column, anchored at the bottom. A layer
// toggle pops a card out, then the cards above it slide down into the gap.
// Turning it back on opens the gap first, then the card pops in.
import { ui } from "./core.js";

const BASE = 58;
const POP_OUT = 180;
const SLIDE = 250;
const POP_IN = 200;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

const slots = [
  { key: "solar", ids: ["solar-dial"], h: 156, gap: 8, layer: "tod" },
  { key: "clar", ids: ["clar-inst"], h: 88, gap: 10, layer: null },
  { key: "row", ids: ["age-inst", "cur-inst"], h: 88, gap: 6, layer: null },
  { key: "chop", ids: ["chop-inst"], h: 88, gap: 10, layer: null },
  { key: "wind", ids: ["wind-inst"], h: 112, gap: 0, layer: "wind" },
].map((s) => ({
  ...s,
  els: s.ids.map((id) => document.getElementById(id)).filter(Boolean),
  want: false,
  present: false,
  phase: "gone",
  bottom: BASE,
  settled: true,
  gen: 0,
  timer: 0,
}));

for (const s of slots) for (const el of s.els) el.classList.add("slot");

function wanted(s) {
  if (ui.clean) return false;
  if (!s.layer) return true;
  return !!ui.layers[s.layer];
}

function layout() {
  let y = BASE;
  for (const s of slots) {
    if (!s.present) continue;
    s.bottom = y;
    y += s.h + s.gap;
  }
}

function paint(s) {
  for (const el of s.els) {
    el.style.bottom = `${s.bottom}px`;
    el.classList.toggle("slot-hold", s.phase === "hold");
    el.classList.toggle("slot-gone", s.phase === "gone");
    el.classList.toggle("slot-pop-out", s.phase === "pop-out");
    el.classList.toggle("slot-pop-in", s.phase === "pop-in");
    const hidden = s.phase === "gone" || s.phase === "hold";
    el.setAttribute("aria-hidden", hidden ? "true" : "false");
  }
}

function paintAll() {
  layout();
  for (const s of slots) paint(s);
}

function later(s, gen, ms, fn) {
  s.timer = setTimeout(() => { if (s.gen === gen) fn(); }, ms);
}

function finishShown(s) {
  s.present = true;
  s.phase = "shown";
  s.settled = true;
  paint(s);
}

function finishGone(s) {
  s.present = false;
  s.phase = "gone";
  s.settled = true;
  paintAll();
}

function show(s, gen) {
  if (s.present && (s.phase === "pop-out" || s.phase === "shown" || s.phase === "pop-in")) {
    finishShown(s);
    return;
  }
  s.present = true;
  s.phase = "hold";
  s.settled = false;
  paintAll();
  later(s, gen, reduced.matches ? 0 : SLIDE, () => {
    if (!s.want) { hide(s, gen); return; }
    s.phase = "pop-in";
    paint(s);
    later(s, gen, reduced.matches ? 0 : POP_IN, () => {
      if (!s.want) { hide(s, gen); return; }
      finishShown(s);
    });
  });
}

function hide(s, gen) {
  if (!s.present || s.phase === "hold" || s.phase === "gone") {
    finishGone(s);
    return;
  }
  s.phase = "pop-out";
  s.present = true;
  s.settled = false;
  paint(s);
  later(s, gen, reduced.matches ? 0 : POP_OUT, () => {
    if (s.want) { show(s, gen); return; }
    finishGone(s);
  });
}

function sync() {
  let changed = false;
  for (const s of slots) {
    const want = wanted(s);
    // An in-flight pop keeps its timer. Restarting every frame cleared that
    // timer before the pop could finish, so the column never slid.
    if (want === s.want) continue;
    s.want = want;
    s.gen += 1;
    clearTimeout(s.timer);
    changed = true;
    if (reduced.matches) {
      s.present = want;
      s.phase = want ? "shown" : "gone";
      s.settled = true;
      continue;
    }
    if (want) show(s, s.gen);
    else hide(s, s.gen);
  }
  if (changed && reduced.matches) paintAll();
}

for (const s of slots) {
  s.want = wanted(s);
  s.present = s.want;
  s.phase = s.want ? "shown" : "gone";
}
paintAll();

ui.register({ update() { sync(); } });

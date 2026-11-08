// Short, interruptible coast after a scalar drag. Depth, clarity, chop, wind
// speed and placement never register here. Direction may, on its own 360° circle.
const jobs = new Map();
const wrapDelta = (d, period) => {
  const half = period / 2;
  let x = d;
  while (x <= -half) x += period;
  while (x > half) x -= period;
  return x;
};

export function sample(id, value, now) {
  let j = jobs.get(id);
  if (!j) jobs.set(id, j = { samples: [] });
  j.holding = true;
  j.v = 0;
  j.value = value;
  j.samples.push({ t: now, v: value });
  const cut = now - 140;
  if (j.samples.length > 8) j.samples.splice(0, j.samples.length - 8);
  j.samples = j.samples.filter((s) => s.t >= cut);
}

export function release(id, opt) {
  const j = jobs.get(id);
  if (!j) return;
  j.holding = false;
  j.samples = j.samples || [];
  const a = j.samples[0], b = j.samples[j.samples.length - 1];
  j.samples = [];
  if (!a || !b || !(b.t > a.t)) { j.v = 0; return; }
  const dt = (b.t - a.t) / 1000;
  let vel = dt > 0.02 ? (b.v - a.v) / dt : 0;
  if (opt.wrap) vel = wrapDelta(b.v - a.v, opt.wrap) / dt;
  const cap = opt.maxAbs ?? 1e9;
  vel = Math.max(-cap, Math.min(cap, vel));
  const floor = opt.ignore ?? 0;
  j.v = Math.abs(vel) < floor ? 0 : vel;
  j.value = b.v;
  j.min = opt.min; j.max = opt.max; j.wrap = opt.wrap || 0;
  j.tau = opt.tau ?? 0.2; j.stop = opt.stop ?? 0.5;
  j.on = opt.on;
}

export function cancel(id) {
  const j = jobs.get(id);
  if (!j) return;
  j.v = 0; j.holding = false; j.samples = [];
}

export function cancelAll() { for (const id of jobs.keys()) cancel(id); }

export function velocity(id) { return jobs.get(id)?.v || 0; }

export function tick(dt, reduced) {
  if (!(dt > 0)) return;
  for (const j of jobs.values()) {
    if (reduced || j.holding || !j.v || !j.on) { if (reduced) j.v = 0; continue; }
    j.value += j.v * dt;
    if (j.wrap) {
      j.value = ((j.value % j.wrap) + j.wrap) % j.wrap;
    } else {
      if (j.value < j.min) { j.value = j.min; j.v = 0; }
      else if (j.value > j.max) { j.value = j.max; j.v = 0; }
    }
    j.on(j.value);
    j.v *= Math.exp(-dt / j.tau);
    if (Math.abs(j.v) < j.stop) j.v = 0;
  }
}

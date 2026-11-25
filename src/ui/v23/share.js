// Short versioned scene link. The heightmap stays out of the URL; only a preset id.
// Old links (wind, island, cam..., no s=) are left alone.

const PRESET = "sandbar";

export function parseShare(search) {
  const q = new URLSearchParams(search);
  const raw = q.get("s");
  if (!raw) return null;
  const p = raw.split(".");
  if (p[0] !== "1" || p.length < 11) return null;
  const num = (i) => Number(p[i]);
  return {
    wind: num(1) / 10,
    dir: num(2),
    size: num(3),
    age: num(4) / 100,
    cur: (num(5) - 30) / 10,
    override: num(6) === 1,
    chop: num(7) / 10,
    clarity: num(8) / 10,
    island: num(9) === 1,
    preset: p[10] || PRESET,
    waves: p[11] === "h" ? "hybrid" : "spectral",
  };
}

export function encodeShare(s) {
  const n = (v, d = 0) => String(Math.round(v));
  return [
    "1",
    n(s.wind * 10),
    n(s.dir),
    n(s.size),
    n(s.age * 100),
    n(s.cur * 10 + 30),
    s.override ? "1" : "0",
    n(s.chop * 10),
    n(s.clarity * 10),
    s.island ? "1" : "0",
    s.preset || PRESET,
    s.waves === "hybrid" ? "h" : "s",
  ].join(".");
}

// If the share says the island is on, the shader is only built when island=1
// is its own query. Add it and reload once.
export function islandRedirect(href) {
  const u = new URL(href);
  const s = parseShare(u.search);
  if (!s) return null;
  const has = u.searchParams.get("island") === "1";
  if (s.island && !has) {
    u.searchParams.set("island", "1");
    return u.href;
  }
  if (!s.island && has) {
    u.searchParams.delete("island");
    return u.href;
  }
  return null;
}

export function shareSettingsFromPage() {
  const q = (id) => document.getElementById(id);
  const steep = globalThis.__steep || {};
  return {
    wind: Number(q("spd").value),
    dir: Number(q("dir").value),
    size: Number(q("size").value),
    age: Number(q("age").value),
    cur: Number(q("cur").value),
    override: !!steep.override,
    chop: Number(q("chop").value),
    clarity: Number(q("clarity").value),
    island: new URLSearchParams(location.search).get("island") === "1",
    preset: PRESET,
    waves: globalThis.__waves === "hybrid" ? "hybrid" : "spectral",
  };
}

export function shareUrl(settings, href) {
  const u = new URL(href);
  u.searchParams.set("s", encodeShare(settings));
  if (settings.island) u.searchParams.set("island", "1");
  else u.searchParams.delete("island");
  return u.href;
}

export function applyShareToInputs(share) {
  if (!share) return;
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = String(v);
  };
  set("spd", share.wind);
  set("dir", share.dir);
  set("size", share.size);
  set("age", share.age);
  set("cur", share.cur);
  set("clarity", share.clarity);
  if (share.override) set("chop", share.chop);
}

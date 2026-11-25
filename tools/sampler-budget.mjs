// Counts fragment samplers the way a 16-unit implementation does, and fails
// if any sea program would use more than 15 (one unit kept free).
import { register } from "node:module";
import { pathToFileURL } from "node:url";

register("./query-loader.mjs", { parentURL: import.meta.url });

const { seaFragmentSources } = await import("../src/render/view.js");
const { fragmentSamplerNames } = await import("../src/render/samplerBudget.js");
const { BOAT_FS } = await import("../src/render/boatView.js");
const { CAUST_FS, SH_FS } = await import("../src/render/light.js");

const HEADROOM = 15;
const rows = [];

function check(name, source) {
  const names = fragmentSamplerNames(source);
  rows.push({ name, count: names.length, names });
  if (names.length > HEADROOM) {
    console.error(`${name} uses ${names.length} fragment samplers: ${names.join(", ")}`);
    process.exitCode = 1;
  }
}

for (const island of [false, true]) {
  const src = seaFragmentSources(true, island);
  const tag = island ? "+island" : "";
  check(`cpu${tag}`, src.cpu);
  check(`gpu${tag}`, src.gpu);
  check(`fft${tag}`, src.fft);
}
check("boat", BOAT_FS);
check("caust", CAUST_FS);
check("shadow", SH_FS);

console.log("fragment samplers (limit 16, headroom 15):");
for (const row of rows) {
  console.log(`${row.count}\t${row.name}\t${row.names.join(", ")}`);
}
if (process.exitCode) {
  console.error("sampler budget exceeded");
} else {
  console.log("sampler budget ok");
}

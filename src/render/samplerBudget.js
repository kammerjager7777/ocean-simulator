// Mac ANGLE Metal rejects a fragment shader with more than 16 samplers.
// The sea keeps one unit free. The count is the samplers the fragment
// actually references, which is what the linker charges.

const SAMPLER = new Set([
  "sampler2D", "sampler2DArray", "samplerCube", "sampler3D",
  "isampler2D", "usampler2D", "sampler2DShadow", "samplerCubeShadow",
]);

export function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/.*$/gm, "");
}

// Enough of the preprocessor for the sea shaders: #define NAME, #ifdef, #ifndef, #else, #endif.
export function preprocess(source, initial = {}) {
  const defines = { ...initial };
  const lines = stripComments(source).split("\n");
  const out = [];
  const stack = [];
  const live = () => stack.every((s) => s);
  for (const line of lines) {
    const t = line.trim();
    const def = t.match(/^#define\s+([A-Za-z_]\w*)(?:\s+(\S+))?/);
    if (def && live()) { defines[def[1]] = def[2] ?? "1"; out.push(line); continue; }
    if (t.startsWith("#ifdef") || t.startsWith("#ifndef")) {
      const name = t.split(/\s+/)[1];
      const on = Object.prototype.hasOwnProperty.call(defines, name);
      stack.push(live() && (t.startsWith("#ifdef") ? on : !on));
      continue;
    }
    if (t.startsWith("#else")) { if (stack.length) stack[stack.length - 1] = liveParent(stack) && !stack[stack.length - 1]; continue; }
    if (t.startsWith("#endif")) { stack.pop(); continue; }
    if (t.startsWith("#if")) { stack.push(live() && !/^#if\s+0\b/.test(t)); continue; }
    if (live()) out.push(line);
  }
  return out.join("\n");
  function liveParent(s) { return s.slice(0, -1).every(Boolean); }
}

export function fragmentSamplerNames(source) {
  const body = preprocess(source);
  const re = /\buniform\s+(?:(?:lowp|mediump|highp)\s+)?(\w+)\s+(\w+)/g;
  const names = [];
  let m;
  while ((m = re.exec(body))) {
    if (!SAMPLER.has(m[1])) continue;
    const name = m[2];
    const rest = body.slice(0, m.index) + " " + body.slice(m.index + m[0].length);
    if (new RegExp(`\\b${name}\\b`).test(rest)) names.push(name);
  }
  return names;
}

const GL_SAMPLER = new Set();

export function activeSamplerNames(gl, program) {
  if (!GL_SAMPLER.size) {
    for (const n of ["SAMPLER_2D", "SAMPLER_2D_ARRAY", "SAMPLER_CUBE", "SAMPLER_3D", "INT_SAMPLER_2D", "UNSIGNED_INT_SAMPLER_2D"]) {
      if (gl[n] != null) GL_SAMPLER.add(gl[n]);
    }
  }
  const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  const names = [];
  for (let i = 0; i < n; i += 1) {
    const info = gl.getActiveUniform(program, i);
    if (info && GL_SAMPLER.has(info.type)) names.push(info.name.replace(/\[0\]$/, ""));
  }
  return names;
}

// Fragment samplers are the active program samplers that the fragment source references.
// Vertex-only samplers share the program but do not count toward MAX_TEXTURE_IMAGE_UNITS.
export function fragmentSamplerReport(gl, program, fragSource) {
  const referenced = new Set(fragmentSamplerNames(fragSource));
  const active = activeSamplerNames(gl, program).filter((name) => referenced.has(name));
  return { count: active.length, names: active };
}

export function assertFragmentSamplers(gl, program, fragSource, label) {
  const report = fragmentSamplerReport(gl, program, fragSource);
  if (report.count > 16) {
    throw new Error(`${label} fragment shader uses ${report.count} samplers (${report.names.join(", ")}); MAX_TEXTURE_IMAGE_UNITS is 16`);
  }
  return report;
}

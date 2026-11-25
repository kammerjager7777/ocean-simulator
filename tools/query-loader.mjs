// Browser modules import "./file.js?v=33". Node does not. Strip the query.
export async function resolve(specifier, context, nextResolve) {
  const q = specifier.indexOf("?");
  if (q !== -1) specifier = specifier.slice(0, q);
  return nextResolve(specifier, context);
}

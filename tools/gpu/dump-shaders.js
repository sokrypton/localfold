// The WGSL another tool compiles, by pipeline label: wraps a tool's `main`, records every shader module's text and
// the labels of the pipelines built from it, and returns the modules whose label matches --match=<regex>.
//   node tools/gpu-chrome.mjs tools/gpu/dump-shaders.js --tool=bench-head --match=ffw-out [the tool's own flags]
// The text is what the device was handed, so in Firefox it is after webgpu/runtime/wgsl-compat.js's rewrite.
export async function main(device, args) {
  const option = (name) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
  const tool = option("tool");
  const match = new RegExp(option("match") ?? ".");
  const modules = new Map();
  const create = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor) => {
    const module = create(descriptor);
    modules.set(module, { labels: new Set([descriptor.label ?? ""]), code: descriptor.code });
    return module;
  };
  for (const name of ["createComputePipeline", "createComputePipelineAsync"]) {
    const original = device[name].bind(device);
    device[name] = (descriptor) => {
      modules.get(descriptor.compute?.module)?.labels.add(descriptor.label ?? "");
      return original(descriptor);
    };
  }
  const { main: run } = await import(`./${tool}.js`);
  // (a tool that dies - on a shader the browser refuses, say - still returns what it compiled, and why it died)
  let failure = null;
  try {
    await run(device, args.filter((a) => !a.startsWith("--tool=") && !a.startsWith("--match=")));
  } catch (error) { failure = String(error?.message ?? error); }
  // ...and the compiler's own messages for each module, which a refusal does not carry
  for (const [module, entry] of modules) {
    const info = await module.getCompilationInfo().catch(() => null);
    entry.errors = (info?.messages ?? []).filter((m) => m.type === "error")
      .map((m) => `line ${m.lineNum}: ${m.message} | ${(entry.code.split("\n")[m.lineNum - 1] ?? "").trim().slice(0, 200)}`);
  }
  const shaders = [...modules.values()].map((m) => ({ labels: [...m.labels].filter(Boolean), errors: m.errors, code: m.code }))
    .filter((m) => m.labels.some((label) => match.test(label)));
  return { modules: modules.size, matched: shaders.length, failure, shaders };
}

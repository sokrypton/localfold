// One kernel's GPU time on its own: a WGSL file compiled as this device's compiler is handed it, bound to buffers of
// deterministic noise and dispatched alone under a pair of timestamps, so no neighbouring pass shares its clock.
//   node tools/gpu-chrome.mjs tools/gpu/time-shader.js --wgsl=/oracle-dumps/kernel.wgsl --sizes=409600,300000,409600 --groups=200
// `--sizes=` is the element count of each binding of group 0 in order (4 bytes an element; `h` after a count for
// f16), `--groups=x[,y[,z]]` the dispatch. tools/gpu/dump-shaders.js writes the text; Firefox's pass timestamps blur
// between neighbours inside one submit (docs/FIREFOX.md), which is what this is for. `--raw` skips the rewrite.
import { compilerText } from "../../webgpu/runtime/wgsl-compat.js";

export async function main(device, args) {
  const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
  const source = await (await fetch(option("wgsl"))).text();
  const code = args.includes("--raw") ? source : compilerText(device, source);
  const module = device.createShaderModule({ code });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const writable = [...source.matchAll(/@binding\((\d+)\)\s*var<storage,\s*read_write>/g)].map((m) => Number(m[1]));
  const entries = option("sizes", "").split(",").filter(Boolean).map((text, binding) => {
    const half = text.endsWith("h");
    const count = Number(half ? text.slice(0, -1) : text);
    const buffer = device.createBuffer({ size: Math.ceil(count * (half ? 2 : 4) / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (!writable.includes(binding)) {
      const values = half ? new Float16Array(count) : new Float32Array(count);
      for (let i = 0; i < count; i += 1) values[i] = ((Math.imul(i + 1, 2654435761) >>> 8) % 2048) / 1024 - 1;
      device.queue.writeBuffer(buffer, 0, values.buffer, 0, buffer.size);
    }
    return { binding, resource: { buffer } };
  });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const groups = option("groups", "1").split(",").map(Number);
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const stamps = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const once = async () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(...groups);
    pass.end();
    encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, stamps, 0, 16);
    device.queue.submit([encoder.finish()]);
    await stamps.mapAsync(GPUMapMode.READ);
    const [begin, end] = new BigUint64Array(stamps.getMappedRange().slice(0));
    stamps.unmap();
    return Number(end - begin) / 1e6;
  };
  await once();
  const times = [];
  for (let round = 0; round < Number(option("rounds", "9")); round += 1) times.push(await once());
  times.sort((a, b) => a - b);
  return { groups, minMs: times[0], medianMs: times[times.length >> 1], maxMs: times[times.length - 1] };
}

// What a workgroup pays for the workgroup memory it DECLARES, whatever it uses: one kernel that touches 64 floats of
// a `var<workgroup>` array, with the array declared at 64 to 8192 floats, timed on the GPU's clock.
//   node tools/gpu-chrome.mjs tools/gpu/probe-workgroup-init.js
//   LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/probe-workgroup-init.js
// WGSL zeroes workgroup memory at entry. A compiler that has ONE lane do it while the rest wait makes every workgroup
// cost its declared bytes (Firefox; docs/FIREFOX.md); one that spreads it over the lanes does not (Chrome).
// `shape` is how the floats are declared: "f32" one array<f32>, "vec4" one array<vec4<f32>>, "two" two arrays of half.
const shader = (floats, shape) => {
  if (shape === "nested") {
    // the same floats as rows of 2048: one variable, no array of it over 8 KiB
    return `
@group(0) @binding(0) var<storage, read_write> result: array<f32>;
var<workgroup> tile: array<array<f32, 2048>, ${floats / 2048}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  tile[local >> 11u][local & 2047u] = f32(local + group.x);
  workgroupBarrier();
  result[group.x * 64u + local] = tile[(63u - local) >> 11u][(63u - local) & 2047u] + tile[${floats / 2048 - 1}u][2047u];
}`;
  }
  const declare = shape === "vec4" ? `var<workgroup> tile: array<vec4<f32>, ${floats / 4}>;`
    : shape === "two" ? `var<workgroup> tile: array<f32, ${floats / 2}>;\nvar<workgroup> other: array<f32, ${floats / 2}>;`
      : `var<workgroup> tile: array<f32, ${floats}>;`;
  const at = (index) => (shape === "vec4" ? `tile[(${index}) / 4u][(${index}) % 4u]` : `tile[${index}]`);
  const last = shape === "vec4" ? `tile[${floats / 4 - 1}u].w` : shape === "two" ? `other[${floats / 2 - 1}u]` : `tile[${floats - 1}u]`;
  return `
@group(0) @binding(0) var<storage, read_write> result: array<f32>;
${declare}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  ${at("local")} = f32(local + group.x);
  workgroupBarrier();
  result[group.x * 64u + local] = ${at("63u - local")} + ${last};
}`;
};

export async function main(device, args) {
  const groups = Number(args.find((a) => a.startsWith("--groups="))?.slice(9) ?? 4096);
  const result = device.createBuffer({ size: groups * 64 * 4, usage: GPUBufferUsage.STORAGE });
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const stamps = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const arms = [];
  const list = (args.find((a) => a.startsWith("--floats="))?.slice(9) ?? "64,512,2048,4096,8192").split(",").map(Number);
  const shape = args.find((a) => a.startsWith("--shape="))?.slice(8) ?? "f32";
  for (const floats of list) {
    const module = device.createShaderModule({ code: shader(floats, shape) });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: result } }] });
    const once = async () => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups);
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
    for (let round = 0; round < 7; round += 1) times.push(await once());
    const ms = Math.min(...times);
    arms.push({ floats, ms: Number(ms.toFixed(3)), microsecondsAGroup: Number((ms * 1000 / groups).toFixed(2)) });
  }
  return { groups, shape, arms };
}

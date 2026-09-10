/**
 * How many workgroups this device runs at once, measured.
 *
 * 🔴 THE DIFFUSION GEOMETRY KNOBS ARE ALL ONE QUESTION AND IT IS THIS ONE.
 * `--no-prior=<knob>` prices AF3's prior at 2650 ms of sampler on this card,
 * and 1792 of it is `diffusionSplitK` - which exists because a small token
 * count launches too few workgroups to fill the device, so the kernel splits
 * its K to make more. The right split is "how many more do I need", and the
 * only unknown in that is the device's own parallel width. `crossover` is the
 * same fact from the other side: past a few hundred tokens the kernel fills the
 * device by itself and the split is pure cost.
 *
 * A table cannot know that number for a GPU nobody has run. A probe can, in
 * about ten milliseconds, and it measures the physical quantity rather than
 * standing in for it.
 *
 * The kernel is a long dependent chain in registers, so its time is fixed per
 * workgroup and independent of memory: N workgroups take one workgroup's time
 * until the device is full, and grow linearly after. The width is the last
 * count still flat.
 */
const SHADER = `
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // A dependent chain, so nothing here is elided and nothing is memory bound.
  var value = f32(id.x) * 1e-6 + 1.0;
  for (var step = 0u; step < 65536u; step += 1u) {
    value = fma(value, 0.9999, 1e-7);
  }
  // Written by one invocation only, so the store is not the measurement.
  if (id.x == 0u) { sink[0] = value; }
}`;

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args = []) {
  const rounds = Number(option(args, "rounds", "5"));
  const sink = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: sink } }],
  });

  // 🔴 FINER THAN POWERS OF TWO. The width is a plateau edge and a doubling
  // step can only place it within a factor of two, which is not enough to
  // choose a split count from.
  const counts = [];
  for (let n = 1; n <= 8192; n = Math.max(n + 1, Math.round(n * 1.4))) counts.push(n);
  const timings = new Map(counts.map((n) => [n, Infinity]));
  for (let round = 0; round < rounds; round += 1) {
    for (const count of counts) {
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(count);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - started;
      if (ms < timings.get(count)) timings.set(count, ms);
    }
  }

  // The floor is the submit round trip plus one workgroup's chain; the width is
  // the largest count still within a margin of it.
  // 🔴 THE FLOOR IS ONE WORKGROUP'S OWN TIME, not the submit round trip. The
  // chain is long enough that a single workgroup is milliseconds and the round
  // trip is a fifth of one, which is what makes the plateau readable at all -
  // at 4096 iterations every count below 1024 measured the same 0.2-0.3 ms of
  // queue and the edge was invisible.
  // 🔴 THE FLOOR IS THE PLATEAU, AND IT DOES NOT INCLUDE ONE WORKGROUP.
  // Measured here, a single workgroup is 1.4 ms where two through four thousand
  // are all 2.3 - one workgroup is not a scaled-down device, it is a device
  // running one workgroup, and taking it as the baseline puts the plateau at
  // 1.64x its own floor and finds an edge everywhere.
  const early = counts.filter((n) => n >= 2 && n <= 8);
  const floor = Math.min(...early.map((n) => timings.get(n)));
  const margin = Number(option(args, "margin", "1.35"));
  let width = 1;
  for (const count of counts) {
    if (timings.get(count) <= floor * margin) width = count;
  }
  // 🔴 AND A SECOND ESTIMATE THAT DOES NOT DEPEND ON THE MARGIN. Past
  // saturation the device is a fixed number of workgroups wide, so a dispatch
  // of N takes ceil(N / width) plateaus: width is about N * floor / time. The
  // two agreeing is what says the plateau was read and not invented.
  const largest = counts[counts.length - 1];
  const fromSlope = Math.round(largest * floor / timings.get(largest));
  const rows = counts.map((count) => ({
    workgroups: count, ms: Number(timings.get(count).toFixed(3)),
    ratio: Number((timings.get(count) / floor).toFixed(2)),
  }));
  for (const row of rows) console.log(`${String(row.workgroups).padStart(5)}  ${row.ms.toFixed(3)}  x${row.ratio}`);
  return { floorMs: Number(floor.toFixed(3)), margin,
           saturationWorkgroups: width, lanes: width * 64,
           fromSlope, agrees: Math.max(width, fromSlope) / Math.min(width, fromSlope) < 2,
           rows };
}

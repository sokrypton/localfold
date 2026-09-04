/**
 * What a dispatch costs before it computes anything.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-dispatch.js
 *
 * WHY IT EXISTS. An AF3 trunk pass spends nearly all of its wall clock waiting
 * on a GPU whose labelled compute passes add up to well under it, while the
 * encoding that queues the work costs one or two milliseconds - so the gap is
 * neither the host's nor inside any pass. It looked like it had to be BETWEEN
 * the dispatches, of which a pass has hundreds, each small enough at these
 * token counts that whatever a dispatch costs to start and drain could be the
 * larger term. Nothing measured that, and every conclusion about where the
 * trunk's time goes depended on the number.
 *
 * It is not the dispatches. See the answer in AF3.md - the gap is the profiler,
 * which adds 30% to the clock it is measured against and quantises its
 * timestamps to about 100 us across a great many short passes.
 *
 * Three arms, all issuing the same trivial kernel:
 *   - `one-pass`    N dispatches inside a single compute pass
 *   - `pass-each`   N dispatches, each in its own compute pass
 *   - `encoder-each` N dispatches, each in its own encoder and submission
 * The first two differ by what a compute pass costs; the first and last by
 * what a submission costs. The kernel writes one float so it cannot be
 * eliminated, and `groups` sizes the dispatch so the cost can be read against
 * a dispatch that actually fills the machine.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// 🔴 THE REAL LOOP CHANGES PIPELINE AND BIND GROUP EVERY DISPATCH, and a probe
// that reuses one of each prices neither. `variants` cycles that many distinct
// pipelines (identical work, different constants so they cannot be shared) and
// a bind group per dispatch, which is the shape a pairformer block actually
// encodes.
const variantShader = (index) => `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  out[id.x] = out[id.x] * 1.000001 + ${index}e-7;
}`;

const SHADER = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  out[id.x] = out[id.x] * 1.000001 + 1e-7;
}`;

export async function main(device, args) {
  const dispatches = Number(option(args, "dispatches", "512"));
  const rounds = Number(option(args, "rounds", "7"));
  const groups = option(args, "groups", "1,8,64,512").split(",").map(Number);

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
  });
  const out = device.createBuffer({
    size: 512 * 64 * 4, usage: GPUBufferUsage.STORAGE });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: out } }],
  });

  const variants = Number(option(args, "variants", "8"));
  const variantPipelines = [];
  for (let index = 0; index < variants; index += 1) {
    variantPipelines.push(await device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: variantShader(index + 1) }), entryPoint: "main",
      },
    }));
  }

  const shapes = {
    "one-pass": (count, x) => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let i = 0; i < count; i += 1) pass.dispatchWorkgroups(x);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
    "pass-each": (count, x) => {
      const encoder = device.createCommandEncoder();
      for (let i = 0; i < count; i += 1) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(x);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
    },
    "encoder-each": (count, x) => {
      for (let i = 0; i < count; i += 1) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(x);
        pass.end();
        device.queue.submit([encoder.finish()]);
      }
    },
  };

  shapes["cycle-pipelines"] = (count, x) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let i = 0; i < count; i += 1) {
      const pipeline = variantPipelines[i % variantPipelines.length];
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: out } }],
      }));
      pass.dispatchWorkgroups(x);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  // The same cycling with the bind group made once, so the two costs separate.
  const cycledGroups = variantPipelines.map((pipeline) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: out } }],
  }));
  shapes["cycle-pipelines-cached"] = (count, x) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let i = 0; i < count; i += 1) {
      pass.setPipeline(variantPipelines[i % variantPipelines.length]);
      pass.setBindGroup(0, cycledGroups[i % cycledGroups.length]);
      pass.dispatchWorkgroups(x);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  // 🔴 AND WHAT A ROUND TRIP COSTS, which is the thing a pipeline of separate
  // GPU modules pays between every pair of them. Each iteration submits, drains
  // the queue and maps a buffer back - the shape of `run()` returning a
  // Float32Array. `bytes` sizes the copy so the drain and the copy separate.
  const roundTripBytes = Number(option(args, "readback-bytes", String(59 * 768 * 4)));
  const staging = device.createBuffer({
    size: roundTripBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const source = device.createBuffer({
    size: roundTripBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const roundTrip = async (count, x) => {
    for (let i = 0; i < count; i += 1) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(x);
      pass.end();
      encoder.copyBufferToBuffer(source, 0, staging, 0, roundTripBytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      staging.getMappedRange().slice(0);
      staging.unmap();
    }
  };

  const time = async (shape, count, x) => {
    const start = performance.now();
    shapes[shape](count, x);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  const results = [];
  {
    const trips = Number(option(args, "round-trips", "32"));
    await roundTrip(4, 8);
    const times = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      await roundTrip(trips, 8);
      times.push(performance.now() - start);
    }
    const ms = median(times);
    results.push({
      shape: "round-trip", groups: 8, dispatches: trips,
      bytes: roundTripBytes,
      ms: Number(ms.toFixed(2)),
      microsecondsEach: Number((ms * 1000 / trips).toFixed(1)),
    });
  }
  for (const x of groups) {
    for (const shape of Object.keys(shapes)) {
      await time(shape, 8, x);
      const times = [];
      for (let round = 0; round < rounds; round += 1) times.push(await time(shape, dispatches, x));
      const ms = median(times);
      results.push({
        shape, groups: x, dispatches,
        ms: Number(ms.toFixed(2)),
        microsecondsEach: Number((ms * 1000 / dispatches).toFixed(1)),
      });
    }
  }
  return { dispatches, rounds, results };
}

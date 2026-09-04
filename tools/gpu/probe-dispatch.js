/**
 * What a dispatch costs before it computes anything.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-dispatch.js
 *
 * WHY IT EXISTS. An AF3 trunk pass at 59 tokens spends 338 ms waiting on a GPU
 * whose labelled compute passes add up to about 120 - and the encoding that
 * queues it costs 5 ms, so the gap is neither the host's nor inside any pass.
 * It is BETWEEN the dispatches: 1,521 of them over three passes, each one
 * small enough at 59 tokens that whatever a dispatch costs to start and drain
 * is the larger term. Nothing measured that, and every conclusion about where
 * the trunk's time goes depends on the number.
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

  const time = async (shape, count, x) => {
    const start = performance.now();
    shapes[shape](count, x);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  const results = [];
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

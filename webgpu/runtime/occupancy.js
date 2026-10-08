/**
 * How many workgroups this device runs at once, measured rather than tabled.
 *
 * 🔴 AF3's GEOMETRY PRIOR IS FIVE KNOBS AND THEY SHARE ONE UNKNOWN.
 * `diffusionSplitK` splits a kernel's K to make more workgroups, because a
 * small token count launches too few to fill the card; `crossover` turns it off
 * again where the kernel fills the card by itself; `atomRowTile`'s note in
 * device-profile.js reads "408 workgroups of 64 lanes, 12% of this device". All
 * three sentences need the same number, and a table can only hold it for
 * architectures somebody has run - which is two.
 *
 * Priced with `--no-prior=<knob>` on `fold.js --folds=2`, this card: the
 * sampler is 1869 ms with the prior and 4524 without, and `diffusionSplitK`
 * alone is 1792 of the 2655.
 *
 * The kernel here is a long dependent chain in registers - fixed time per
 * workgroup, no memory traffic - so N workgroups take one workgroup's time
 * until the device is full and grow linearly after. The width is the last count
 * still flat. tools/gpu/probe-occupancy.js is the same measurement with its
 * whole table printed; this is the version a fold can afford to run.
 */
const MEASURED = new WeakMap();
const DETAIL = new WeakMap();
const RUNNING = new WeakMap();

const SHADER = `
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var value = f32(id.x) * 1e-6 + 1.0;
  for (var step = 0u; step < 65536u; step += 1u) {
    value = fma(value, 0.9999, 1e-7);
  }
  if (id.x == 0u) { sink[0] = value; }
}`;

// 🔴 SIX COUNTS, NOT TWENTY-SIX. The probe tool sweeps 1.4x steps for a table a
// reader looks at; a fold wants the number and nothing else, and the answer is
// only ever used to pick among a handful of power-of-two split counts. Six
// doublings from 8 bracket every device this could run on and cost six
// dispatches of about one workgroup's time each.
const COUNTS = [4, 8, 64, 512, 2048, 8192];

/**
 * Answer as a device of this width, without measuring one.
 *
 * 🔴 THE DERIVATIONS ARE A DEFAULT FOR EVERY GPU NOBODY HAS RUN, AND THEY WERE
 * CHECKED ON ONE. A fill rule that is right at 4542 workgroups can be wrong at
 * 64, and the devices it would be wrong on are exactly the ones with no prior
 * to fall back to. This is how a machine with one width asks what a machine
 * with another would get; `--occupancy=<n>` on any GPU tool reaches it.
 */
export function setDeviceOccupancy(device, workgroups) {
  MEASURED.set(device, workgroups);
  DETAIL.set(device, [{ workgroups, ms: 0, forced: true }]);
  return device;
}

/** What the measurement saw, for a probe or a log to print. */
export const deviceOccupancyDetail = (device) => DETAIL.get(device) ?? null;

/** The measured width, or null where nothing has measured it yet. */
export function deviceSaturationWorkgroups(device) {
  return MEASURED.get(device) ?? null;
}

/**
 * Measure it, once per device. Concurrent callers share one measurement.
 *
 * 🔴 IT IS NOT FREE AND IT IS NOT ON THE FOLD'S CRITICAL PATH EITHER. Started
 * beside a weight download that is seconds long, it costs nothing a user sees;
 * started in front of one it would cost more than the knobs it sets are worth
 * on a small structure. `requestAlphaFoldDevice` starts it and does not await
 * it, and every consumer treats "not measured yet" as "use the default".
 */
export function measureDeviceOccupancy(device) {
  const done = MEASURED.get(device);
  if (done !== undefined) return Promise.resolve(done);
  const running = RUNNING.get(device);
  if (running !== undefined) return running;
  const promise = (async () => {
    const sink = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    try {
      const pipeline = await device.createComputePipelineAsync({
        label: "occupancy.chain",
        layout: "auto",
        compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
      });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: sink } }],
      });
      const run = async (count) => {
        const started = performance.now();
        const encoder = device.createCommandEncoder({ label: "occupancy" });
        const pass = encoder.beginComputePass({ label: "occupancy" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(count);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - started;
      };
      // 🔴 A WARM-UP AND TWO ROUNDS, BECAUSE ONE TIMING EACH READ 4 WORKGROUPS
      // ON A CARD THE TOOL MEASURES AT 4542. The first dispatch after a
      // pipeline is built is not the pipeline's steady time, and this probe
      // reads a RATIO between counts, so a single slow first reading does not
      // add noise - it moves the floor and collapses the plateau.
      await run(COUNTS[0]);
      const timings = new Map();
      for (let round = 0; round < 2; round += 1) {
        for (const count of COUNTS) {
          const ms = await run(count);
          if (ms < (timings.get(count) ?? Infinity)) timings.set(count, ms);
        }
      }
      DETAIL.set(device, [...timings.entries()]
        .map(([count, ms]) => ({ workgroups: count, ms: Number(ms.toFixed(3)) })));
      // 🔴 THE FLOOR EXCLUDES ONE WORKGROUP, and the tool records why: a single
      // workgroup measured 1.4 ms where two through four thousand all measured
      // 2.3, so taking it as the baseline puts the whole plateau at 1.64x its
      // own floor and finds an edge everywhere. COUNTS starts at four.
      const floor = Math.min(timings.get(COUNTS[0]), timings.get(COUNTS[1]));
      let width = COUNTS[0];
      for (const count of COUNTS) {
        if (timings.get(count) <= floor * 1.35) width = count;
      }
      MEASURED.set(device, width);
      return width;
    } finally {
      sink.destroy();
      RUNNING.delete(device);
    }
  })();
  RUNNING.set(device, promise);
  return promise;
}

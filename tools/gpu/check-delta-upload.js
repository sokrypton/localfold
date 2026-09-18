/**
 * Does the SHIPPED uploader ADD a delta to weights that are already resident?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-delta-upload.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-delta-upload.js \
 *       --base=/model --delta=/model-mono-3-delta --target=/model-mono-3
 *
 * AlphaFold 2 is five models and a bundle is 97 MiB; a delta is 43 (see
 * tools/pack_delta_model.py, and docs/AF2.md for the fold that says three bits
 * is free). Applying one means decoding int3 codes and ADDING them into the f16
 * weights a fold already holds, which is `planBlockUpload(..., { accumulate:
 * true })` - the same planner, the same shader factory, one extra read.
 *
 * 🔴 IT DRIVES THE SHIPPED PLANNER ON REAL PARAMETERS, not a copy of it on
 * random ones. `check-quantised-upload.js` sweeps the codec over a synthetic
 * shard; this one takes AlphaFold's own tensors out of two bundles, because
 * what a delta exercises is the pairing of two DIFFERENT bundles' records - the
 * base's group and the delta's are not the same size, and nothing else here
 * ever reads two at once.
 *
 * 🔴 THE BAR IS ZERO DIFFERING f16 RESULTS. `code * scale + zero` is f64 on the
 * host and f32 in WGSL, and the sum with the resident weight is another place
 * the two could part; the point of the check is that they do not.
 *
 * 🔴 AND TWO CONTROLS, BECAUSE A DELTA THAT ARRIVED AND A DELTA THAT DID NOT
 * BOTH LOOK LIKE A BUFFER FULL OF PLAUSIBLE WEIGHTS. The first runs the same
 * plan with `accumulate` OFF and requires the result to DIFFER - a flag that
 * does not reach the kernel is a flag that overwrites the model it was meant to
 * add to, silently. The second, when `--target=` names the model's own bundle,
 * requires the reconstruction to be nearer that model than the base is; without
 * it a delta of zeros would pass everything above.
 */
import { HttpTensorStore } from "../../src/bundles/http-tensor-store.js";
import { planBlockUpload, runBlockUpload } from "../../src/weights/quantised-upload.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** The f16 the device would hold, from a store that may be int8 or float. */
const heldByTheDevice = async (store, name) => new Float16Array(await store.tensorAsFloat16(name));

export async function main(device, args) {
  const basePath = option(args, "base", "/model");
  const deltaPath = option(args, "delta", "/model-mono-3-delta");
  const targetPath = option(args, "target", "");
  const wanted = Number(option(args, "tensors", "6"));

  const base = await HttpTensorStore.open(`${basePath}/manifest.json`);
  const delta = await HttpTensorStore.open(`${deltaPath}/manifest.json`);
  const header = delta.manifest.delta;
  if (header === undefined) throw new Error(`${deltaPath} is not a delta bundle`);
  const target = targetPath === ""
    ? null : await HttpTensorStore.open(`${targetPath}/manifest.json`);

  // The largest delta'd tensors, because a small one exercises one group.
  const names = header.addTo
    .filter((name) => base.manifest.tensors[name] !== undefined)
    .sort((a, b) => delta.manifest.tensors[b].shape.reduce((x, y) => x * y, 1)
      - delta.manifest.tensors[a].shape.reduce((x, y) => x * y, 1))
    .slice(0, wanted);
  if (names.length === 0) throw new Error("no delta'd tensor is in both bundles");

  const rows = [];
  let failed = 0;
  for (const name of names) {
    const held = await heldByTheDevice(base, name);
    const added = await delta.tensor(name);
    if (added.length !== held.length) {
      rows.push({ name, failed: `${added.length} against the base's ${held.length}` });
      failed += 1;
      continue;
    }
    const elements = held.length;
    const words = Math.ceil(elements / 2);
    await delta.open(name);
    const entries = [{ thunk: { store: delta, tensorName: name, first: 0, count: elements },
                       offset: 0, length: elements }];

    const run = async(accumulate) => {
      const plan = planBlockUpload(entries, "f16", { accumulate });
      if (plan === undefined || plan.gpu.params.length !== 1) return null;
      const buffer = device.createBuffer({
        size: words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
          | GPUBufferUsage.COPY_DST });
      // ...the weights a fold already holds, in the layout the kernel writes:
      // two f16 elements a word, which is what a Float16Array's bytes are.
      device.queue.writeBuffer(buffer, 0, held.buffer, held.byteOffset, words * 4);
      device.pushErrorScope("validation");
      const release = await runBlockUpload(device, plan.gpu, buffer);
      const readback = device.createBuffer({
        size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, words * 4);
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.mapAsync(GPUMapMode.READ);
      const got = new Float16Array(readback.getMappedRange().slice(0));
      readback.unmap();
      release();
      buffer.destroy();
      return got;
    };

    const got = await run(true);
    if (got === null) { rows.push({ name, failed: "the planner refused it" }); failed += 1; continue; }
    const overwritten = await run(false);

    let differing = 0;
    let worst = 0;
    let moved = 0;
    for (let index = 0; index < elements; index += 1) {
      const want = new Float16Array([held[index] + added[index]])[0];
      if (!Object.is(got[index], want)) {
        differing += 1;
        worst = Math.max(worst, Math.abs(got[index] - want));
      }
      if (!Object.is(overwritten[index], got[index])) moved += 1;
    }
    const row = { name, elements, differing, worst: Number(worst.toPrecision(3)),
                  // 🔴 THE CONTROL: with the flag off the same plan OVERWRITES.
                  // A run where these agree everywhere is a run where the flag
                  // reached nothing.
                  changedByTheFlag: moved };
    if (target !== null) {
      const truth = await heldByTheDevice(target, name);
      const rms = (a, b) => {
        let sum = 0;
        for (let i = 0; i < elements; i += 1) sum += (a[i] - b[i]) ** 2;
        return Math.sqrt(sum / elements);
      };
      row.fromTargetBefore = Number(rms(held, truth).toPrecision(3));
      row.fromTargetAfter = Number(rms(got, truth).toPrecision(3));
    }
    if (differing !== 0 || moved === 0
      || (row.fromTargetAfter !== undefined && row.fromTargetAfter >= row.fromTargetBefore)) {
      failed += 1;
      row.failed = differing !== 0 ? "the device and the host disagree"
        : moved === 0 ? "accumulate changed nothing - the flag reached no kernel"
          : "the delta did not move the weights towards the model it names";
    }
    rows.push(row);
  }
  for (const row of rows) console.log(JSON.stringify(row));
  if (failed !== 0) throw new Error(`${failed} of ${rows.length} delta uploads failed`);
  return { base: basePath, delta: deltaPath, baseModel: header.baseModel,
           tensors: rows.length, elements: rows.reduce((s, r) => s + (r.elements ?? 0), 0),
           differing: 0, rows };
}

/**
 * Does `execution.roundToHalf` actually round, and is a fold's pair track
 * already f16-valued?
 *
 * 🔴 WRITTEN BECAUSE AN ACCURACY ARM CAME BACK BIT-IDENTICAL. Rounding AF2's
 * pair to f16 after all 312 of its writes moved neither the checksum nor pLDDT,
 * which has exactly two explanations - the pass writes nothing, or the values
 * were f16 already - and a fold cannot tell them apart. This asks the kernel
 * directly, on values chosen to be unrepresentable in f16.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";

export async function main(device) {
  const execution = new WebGpuExecution(device);
  const count = 4096;
  const values = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    // 🔴 NOT ROUND NUMBERS. 0.5, 1.0 and 2.0 are exact in f16, so a kernel that
    // does nothing passes a test built from them. These have eleven mantissa
    // bits of information and f16 keeps ten.
    values[i] = Math.fround((i + 1) * 0.10000000149011612 + 1e-4 * Math.sin(i));
  }
  const tensor = execution.upload("probe.values", values,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const encoder = device.createCommandEncoder({ label: "probe.round-to-half" });
  device.pushErrorScope("validation");
  await execution.roundToHalf(encoder, tensor, "probe.round");
  const readback = execution.createReadback("probe.readback", tensor, encoder);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  const out = await execution.mapFloat32(readback);
  let changed = 0;
  let worst = 0;
  let notHalf = 0;
  const half = (x) => {
    const buffer = new ArrayBuffer(4);
    new Float32Array(buffer)[0] = x;
    // what an f16 store keeps, on the host, for comparison
    return Math.fround(new Float16Array([x])[0]);
  };
  for (let i = 0; i < count; i += 1) {
    if (out[i] !== values[i]) changed += 1;
    worst = Math.max(worst, Math.abs(out[i] - values[i]));
    if (out[i] !== half(values[i])) notHalf += 1;
  }
  const result = {
    validationError: error === null ? null : error.message,
    elements: count,
    changed,
    disagreeingWithHostF16: notHalf,
    worstDelta: worst,
    sample: [values[0], out[0], values[1000], out[1000]],
  };
  // 🔴 IT THROWS, BECAUSE THE FAILURE IT EXISTS FOR IS SILENCE. A rounding pass
  // the compiler has folded away reports a perfect run - no error, no changed
  // element - and every arm that uses it comes back byte-identical, which reads
  // as "the precision costs nothing". Measured on the one-dispatch form: 0 of
  // 4096, worst delta exactly 0. Anything that reintroduces that shape fails
  // here instead of in a conclusion.
  if (error !== null) throw new Error(`roundToHalf raised: ${error.message}`);
  if (changed === 0) {
    throw new Error("roundToHalf changed nothing over 4096 values that are not"
      + " representable in f16 - the pass is inert, most likely folded away by"
      + " the shader compiler; see PACK_TO_HALF_SHADER");
  }
  if (notHalf !== 0) {
    throw new Error(`${notHalf} of ${count} values disagree with this platform's`
      + " own Float16Array rounding");
  }
  return result;
}

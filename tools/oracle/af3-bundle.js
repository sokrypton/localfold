/**
 * Reading an AF3 export and an oracle dump, for the block checkers.
 *
 * Shared by check_af3_block.js and check_af3_msa_block.js, which differ only in
 * which stack they slice weights out of and which block they run.
 */
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { readTensor } from "../../src/reference/dtype.js";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** Every tensor of a model directory, by name, already widened to float32. */
export async function loadTensors(directory) {
  const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"));
  const shards = new Map();
  const tensors = new Map();
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const bytes = await readFile(join(directory, record.file));
      shards.set(record.file,
                 bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    }
    tensors.set(name, {
      shape: record.shape,
      data: readTensor(record, shards.get(record.file), record.byteOffset, true),
    });
  }
  return { manifest, tensors };
}

/**
 * One layer out of a stacked tensor.
 *
 * 🔴 THE STACK AXIS IS FIRST, so layer `index` is a contiguous slice - but only
 * because the export keeps AF3's own (num_layer, ...) layout. Reading it as the
 * last axis would still produce a correctly shaped tensor of the wrong numbers.
 */
export function layer(tensors, name, index) {
  const tensor = tensors.get(name);
  if (tensor === undefined) throw new Error(`no tensor named ${name}`);
  const stride = tensor.data.length / tensor.shape[0];
  return tensor.data.subarray(index * stride, (index + 1) * stride);
}

/**
 * Relative RMS of a difference, and the reference's own RMS beside it.
 *
 * 🔴 A RELATIVE ERROR IS UNREADABLE WITHOUT ITS DENOMINATOR. The multimer work
 * lost an afternoon to a block that looked four times worse than its neighbour
 * and was not: its reference values were four times smaller.
 */
export function compare(reference, ours) {
  let error = 0;
  let magnitude = 0;
  let worst = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const difference = ours[index] - reference[index];
    error += difference * difference;
    magnitude += reference[index] * reference[index];
    if (Math.abs(difference) > worst) worst = Math.abs(difference);
  }
  return {
    relative: Math.sqrt(error / magnitude),
    rms: Math.sqrt(magnitude / reference.length),
    worst,
  };
}

export const loadDump = async(name) =>
  JSON.parse(await readFile(join(ROOT, name), "utf8"));

/** The dump's captured arrays, by call site, with a message that says what to run. */
export function captures(dump, hint) {
  return (key) => {
    const record = dump.outputs[key];
    if (record === undefined) {
      throw new Error(`${key} is not in the dump; re-run ${hint}`);
    }
    return Float32Array.from(record.data);
  };
}

/** Report one comparison the way both checkers do. */
export function report(name, reference, ours) {
  const { relative, rms, worst } = compare(reference, ours);
  console.log(`  ${name.padEnd(7)} relRMS ${relative.toExponential(3)}`
    + `   worst ${worst.toExponential(2)}   reference RMS ${rms.toFixed(3)}`);
}

/**
 * A packed weight buffer built by decoding its tensors on the DEVICE.
 *
 * 🔴 AF2's WEIGHTS ARE 815 ms OF HOST WORK BEFORE ITS FIRST FOLD COMPUTES
 * ANYTHING. Its 98 MiB bundle is int8 - 283 tensors of 337, 93.1 M elements -
 * and widening them is 445 ms that `weightLoadMs` reports and the fold's own
 * clock does not cover, then concatenating them into the buffers the kernels
 * bind is another 370 inside the fold. Neither needs a host: the codes are in
 * memory and the arithmetic is `code * scale`.
 *
 * This is `src/af3/pair-track-device-weights.js` generalised to "an ordered
 * list of a descriptor's properties", which is the shape every AF2 packer has -
 * `packTransitionWeights` and `packAttentionWeights` both build a list and
 * concatenate it, and the offsets are the running sum of the lengths.
 *
 * It returns undefined for anything it cannot take - a store with no
 * `tensorSource`, a float32 bundle, a codec the planner refuses - and the
 * caller packs on the host as it always did.
 */
import { planBlockUpload, runBlockUpload } from "./quantised-upload.js";
import { residentWeightBufferFilled } from "./resident.js";

/**
 * The device packer could not take this buffer, and why.
 *
 * 🔴 A THROW AND NOT AN `undefined`, BECAUSE A SILENT FALLBACK IS A BUG THAT
 * LOOKS LIKE A SLOW MACHINE. The attention pack refused for a whole afternoon
 * over one descriptor that copied its tensors into a literal instead of
 * deriving them, and the only symptom was a fold 300 ms slower than the one
 * before it - no error, no wrong number, nothing a checker could see. Every
 * refusal now says which tensor and which reason, and the caller decides
 * whether that is survivable.
 */
export class DeviceWeightRefusal extends Error {
  constructor(reason) {
    super(`the device weight packer refused: ${reason}`);
    this.name = "DeviceWeightRefusal";
    this.reason = reason;
  }
}

/**
 * How many bytes this pack would hold on the device, without building it.
 *
 * 🔴 SO A CEILING CAN BE ANSWERED BEFORE THE FIRST ALLOCATION. `noteAllocation`
 * refuses rather than evicts, which is why this path may run under a budget at
 * all - but AF2's stack has no restart, and letting it fill a 200 MiB ceiling
 * with resident weights pushes the POOLED allocator into evicting a buffer an
 * in-flight submit still names. That is a recorded bug and this is not the
 * commit that fixes it; see uploadResident.
 */
export function packedBytesOf(order, sources, precision) {
  let total = 0;
  for (const item of order) {
    for (const source of sourcesOfItem(item, sources)) {
      if (source === undefined) return undefined;
      total += source.count;
    }
  }
  return precision === "f16" ? Math.ceil(total / 2) * 4 : total * 4;
}

/**
 * The tensors one order entry concatenates, in destination part order.
 *
 * 🔴 AN ENTRY IS NOT ALWAYS ONE TENSOR. AF2's triangle multiplication packs its
 * four projection matrices as ONE interleaved block - the four roles of channel
 * `h` as columns `4h..4h+3` - so a `names` entry names up to four sources and
 * the decoder alternates between them element by element. See `partRun` in
 * src/runtime/quantised-upload.js.
 */
function sourcesOfItem(item, sources) {
  if (typeof item === "string") return [sources?.[item]];
  const map = item.sources ?? sources;
  return (item.names ?? [item.name]).map((name) => map?.[name]);
}

/**
 * @param {object} options
 * @param {object} options.sources the descriptor's SOURCES map
 * @param {string[]} options.order the property names, in packed order
 * @param {"f32"|"f16"} options.precision the destination element
 * @returns {Promise<{buffer: GPUBuffer, offsets: number[]} | undefined>}
 */
export async function residentPackFromSources(device, options) {
  const { key, label, variant = "", sources, order, precision } = options;
  const destination = precision === "f16" ? "f16" : "f32";
  if (destination === "f16" && typeof globalThis.Float16Array !== "function") {
    throw new DeviceWeightRefusal(`${label} wants an f16 destination and this runtime `
      + "has no Float16Array");
  }

  // 🔴 AN ENTRY MAY NAME ITS OWN HOLDER. The attention packer concatenates a
  // descriptor's weights AND its pair-bias tensors, which are two objects with
  // two SOURCES maps - so an order entry is either a property of the default
  // map or a `{sources, name}` pair.
  const offsets = [];
  const offsetsByName = {};
  const entries = [];
  let total = 0;
  for (const item of order) {
    const name = typeof item === "string" ? item : item.name;
    const parts = sourcesOfItem(item, sources);
    for (let index = 0; index < parts.length; index += 1) {
      if (parts[index] !== undefined) continue;
      const missing = typeof item === "string" ? item : (item.names ?? [item.name])[index];
      throw new DeviceWeightRefusal(`${label} has no source for ${missing} - the descriptor `
        + "holding it was built from values rather than gathered");
    }
    offsets.push(total);
    offsetsByName[name] = total;
    // 🔴 EVERY OFFSET EVEN FOR AN f16 DESTINATION, because one invocation owns
    // one word. The planner refuses an odd one rather than letting two
    // dispatches share it; this refuses earlier so the caller falls back whole.
    if (destination === "f16" && total % 2 !== 0) {
      throw new DeviceWeightRefusal(`${label} puts ${name} at the odd offset ${total}, and one `
        + "invocation owns one word");
    }
    const length = parts.reduce((sum, source) => sum + source.count, 0);
    const bound = parts.map((source) => ({
      store: { tensorSource: () => source }, tensorName: source.tensorName,
      first: source.first, count: source.count, bias: source.first,
    }));
    // 🔴 THE ENTRY'S MAPPING WINS OVER THE SOURCE'S, BECAUSE A PACK CAN CANCEL
    // ONE. AF2's triangle projections are stored `[in][out]` and the fixture's
    // getter transposes them to `[out][in]`; the matrix layout's interleave
    // then transposes them BACK, so the composition of the two is the raw
    // tensor and a device path that applied `source.transpose` would produce a
    // correctly shaped, entirely wrong matrix. An order entry that knows the
    // final layout says so with `map`, and only an entry that says nothing
    // inherits what the getter would have done.
    // A transposed tensor is `out[c * rows + r] = in[r * columns + c]`, which is
    // the decoder's stride triple; see quantised-upload.js.
    const shape = typeof item === "string" ? undefined : item.map;
    const perPart = length / parts.length;
    if (parts.length > 1 && shape === undefined) {
      throw new DeviceWeightRefusal(`${label}: ${name} concatenates ${parts.length} tensors and `
        + "says no layout - a multi-part entry must state its map");
    }
    const mapped = shape === "contiguous"
      ? { inner: perPart, innerStride: 1, outerStride: 0 }
      : shape !== undefined && shape !== "source"
        ? { inner: shape.rows, innerStride: shape.columns, outerStride: 1 }
        : parts[0].transpose === undefined
          ? { inner: perPart, innerStride: 1, outerStride: 0 }
          : { inner: parts[0].transpose.rows, innerStride: parts[0].transpose.columns,
              outerStride: 1 };
    const partRun = typeof item === "string" ? 1 : (item.partRun ?? 1);
    entries.push({ name, offset: total, length, sources: bound, partRun, ...mapped });
    total += length;
  }

  const planned = planBlockUpload(entries, destination);
  if (planned === undefined) {
    throw new DeviceWeightRefusal(`${label}: the planner refused this set of records`);
  }
  if (planned.gpu.params.length === 0) {
    throw new DeviceWeightRefusal(`${label}: not one of its tensors is quantised`);
  }
  // 🔴 A HOST ENTRY IS A REFUSAL HERE. `host` means "not quantised, write it
  // yourself", and a transposed one is a reshape nothing else in this path
  // implements - so rather than a second implementation of the mapping, the
  // whole pack goes back to the host packer, which is one code path and always
  // right.
  if (planned.host.length > 0) {
    throw new DeviceWeightRefusal(`${label}: ${planned.host.length} of ${entries.length} `
      + `tensors are not quantised (${planned.host.map((e) => e.name).join(", ")})`);
  }

  const bytes = destination === "f32" ? total * 4 : Math.ceil(total / 2) * 4;
  const buffer = await residentWeightBufferFilled(device, key, label, bytes, async (target) => {
    const release = await runBlockUpload(device, planned.gpu, target);
    void device.queue.onSubmittedWorkDone().then(release);
  }, variant);
  return { buffer, offsets, offsetsByName };
}

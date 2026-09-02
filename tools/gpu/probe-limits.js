/**
 * What is the largest protein this device can hold, and where does it stop?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-limits.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-limits.js --tokens=1000
 *
 * 🔴 THE PAIR REPRESENTATION IS THE WHOLE STORY AND IT IS QUADRATIC. Every
 * other tensor in the trunk is linear in length; the pair stack is L x L x 128,
 * so a length that doubles costs four times the memory and the triangle
 * kernels need several of them live at once. A device limit is a cliff rather
 * than a slowdown, so what matters is which limit is hit first and at what
 * length - which is what this reports, from the device's own numbers rather
 * than from a specification sheet.
 */

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const MiB = 1048576;

export async function main(device, args) {
  const limits = device.limits;
  const named = {};
  for (const key of ["maxBufferSize", "maxStorageBufferBindingSize",
                     "maxComputeWorkgroupsPerDimension", "maxComputeInvocationsPerWorkgroup",
                     "maxBindGroups", "maxStorageBuffersPerShaderStage"]) {
    named[key] = Number(limits[key]);
  }

  // What one pair tensor costs at a given length, in the trunk's own layout:
  // L x L x 128 channels, float32.
  const PAIR_CHANNELS = 128;
  const pairBytes = (length) => length * length * PAIR_CHANNELS * 4;

  const lengths = (option(args, "tokens", null) !== null
    ? [Number(option(args, "tokens", "1000"))]
    : [59, 128, 256, 384, 512, 768, 1000, 1500, 2000]);

  const table = lengths.map((length) => ({
    length,
    pairMiB: Number((pairBytes(length) / MiB).toFixed(1)),
    // 🔴 THE BINDING LIMIT BITES BEFORE THE BUFFER LIMIT. A tensor may be
    // allocatable and still be unusable, because a shader binds it whole.
    fitsBinding: pairBytes(length) <= named.maxStorageBufferBindingSize,
    fitsBuffer: pairBytes(length) <= named.maxBufferSize,
  }));

  // Where the cliff is, to the residue, by bisection on the binding limit.
  let low = 1;
  let high = 8192;
  while (low + 1 < high) {
    const mid = Math.floor((low + high) / 2);
    if (pairBytes(mid) <= named.maxStorageBufferBindingSize) low = mid; else high = mid;
  }
  const maxByBinding = low;

  // ...and whether the device will actually hand over a buffer that size, which
  // is a different question from whether the limit permits it.
  const tryAllocate = async (bytes) => {
    let buffer = null;
    device.pushErrorScope("out-of-memory");
    try {
      buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    } catch (error) {
      return `threw: ${error.message.slice(0, 80)}`;
    }
    const failure = await device.popErrorScope();
    buffer?.destroy();
    return failure === null ? "ok" : `refused: ${failure.message.slice(0, 80)}`;
  };

  const allocations = {};
  for (const length of [512, 768, 1000]) {
    const bytes = pairBytes(length);
    allocations[length] = bytes > named.maxBufferSize
      ? "over maxBufferSize"
      : await tryAllocate(bytes);
  }

  // How many such tensors fit at once, which is what the trunk actually needs.
  const live = {};
  for (const length of [512, 768, 1000]) {
    const bytes = pairBytes(length);
    if (bytes > named.maxBufferSize) { live[length] = 0; continue; }
    const held = [];
    device.pushErrorScope("out-of-memory");
    for (let index = 0; index < 12; index += 1) {
      held.push(device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE }));
    }
    const failure = await device.popErrorScope();
    live[length] = failure === null ? "12+" : `<12 (${failure.message.slice(0, 60)})`;
    for (const buffer of held) buffer.destroy();
  }

  return {
    limits: named,
    limitsMiB: {
      maxBufferSize: Number((named.maxBufferSize / MiB).toFixed(0)),
      maxStorageBufferBindingSize: Number((named.maxStorageBufferBindingSize / MiB).toFixed(0)),
    },
    pairTensor: table,
    maxLengthByBindingLimit: maxByBinding,
    allocatesOnePairTensor: allocations,
    twelveLivePairTensors: live,
  };
}

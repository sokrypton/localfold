/**
 * At what chain length does this fold stop BINDING, and which dispatch decides?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-binding-ceiling.js \
 *       --tool=fold-af2 --lengths=59,118
 *
 * 🔴 WHY IT EXISTS. `maxStorageBufferBindingSize` is 2 GiB on this A100 and on
 * most cards, and no request raises it - it is Vulkan's `maxStorageBufferRange`.
 * A pair-shaped tensor is `L^2 * channels * 4` bytes, so it crosses that at a
 * length that has nothing to do with how much memory the card has: 40 GB free
 * and a 2,048-residue pair still will not bind. `addInPlace` hit exactly this
 * and now windows; the question this answers is whether it was the ONLY one,
 * which grepping cannot say and one fold cannot either.
 *
 * 🔴 TWO LENGTHS, BECAUSE ONE CANNOT SEE THE EXPONENT. A binding that grows as
 * `L` and one that grows as `L^2` look identical in a single run and reach the
 * limit at completely different lengths. Running the wrapped tool at two
 * lengths gives each label a growth ratio, and `log2(ratio)/log2(lengthRatio)`
 * is the exponent - about 2 for the pair, 1 for the single track, 0 for a
 * weight. The extrapolation then says where each one crosses.
 *
 * It is an ESTIMATE and says so: a binding whose size steps rather than scales
 * smoothly - anything already chunked against a budget - will be reported with
 * a fractional exponent and should be read as "already windowed", not as a
 * prediction. What it is for is finding the lowest ceiling and HOW MANY labels
 * sit on it - `lowestCeilingGroup`. A ceiling shared by twenty-eight dispatches
 * is not a bug in whichever one gets named first.
 *
 * 🔴 AND BOTH SAMPLED LENGTHS MUST SIT ABOVE EVERY CHUNK THRESHOLD, OR A
 * HANDLED LABEL IS REPORTED AS A CEILING. A transition chunks against
 * `TRANSITION_CHUNK_TARGET_BYTES` (32 MiB) and the outer product mean against
 * its pair block, so below those a label grows as `L^2` with a clean exponent
 * of 2 and the extrapolation sails straight through a threshold it cannot see.
 * At 59 and 118 residues the four pair transitions read a ceiling of 1,448 and
 * the multimer's read 1,023; at 200 and 400, where they have started chunking,
 * they drop out of the ranking entirely. `sampleFraction` is how far the
 * largest sample actually is from the limit - a row extrapolating from under a
 * sixteenth of it is a guess across more than an order of magnitude, and
 * `caveat` says so.
 *
 * The idea of computing what a prediction binds and checking it against a
 * device is @milot-mirdita's, from martin-steinegger/alphafold2-webgpu; that
 * repository is public and unlicensed, so this was written here rather than
 * taken.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";
import { setDeviceTuning } from "../../src/runtime/device-profile.js";

/**
 * The largest BINDING each labelled compute pass makes, watched at the WebGPU
 * objects rather than at a repository helper.
 *
 * 🔴 BECAUSE `WebGpuExecution.prototype.dispatch` IS AF2's SEAM AND NOT AF3's.
 * The AF3 stacks call `dispatchWorkgroups` from six of their own modules, so
 * the dispatch watcher below recorded ZERO labels for `--tool=fold` and the
 * probe returned `lowestCeiling: null` - which reads as "nothing is near the
 * limit" and means "the instrument was pointed elsewhere".
 *
 * Patching `createBindGroup` and `beginComputePass` catches every caller in
 * either family, because they are the API and not a convention. And the
 * quantity is the BINDING's size - `resource.size`, or the buffer's remainder
 * past `resource.offset` - which is what `maxStorageBufferBindingSize` limits,
 * not the buffer's whole length.
 *
 * 🔴 IT NEEDS `batchComputePasses` OFF, for the reason profile.js needs it: a
 * batched pass is one label over many dispatches, so the largest binding in it
 * would be attributed to whichever kernel happens to name the pass, and the
 * whole point here is WHICH label sits at the bottom.
 */
function watchBindings(device) {
  const sizes = new WeakMap();
  const largest = new Map();
  const createBindGroup = device.createBindGroup.bind(device);
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  device.createBindGroup = (descriptor) => {
    const group = createBindGroup(descriptor);
    let most = 0;
    for (const entry of descriptor?.entries ?? []) {
      const resource = entry?.resource;
      if (resource?.buffer === undefined) continue;
      const bytes = resource.size ?? (resource.buffer.size - (resource.offset ?? 0));
      if (bytes > most) most = bytes;
    }
    sizes.set(group, most);
    return group;
  };
  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);
    const begin = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (pass = {}) => {
      const label = pass.label ?? "(unlabelled)";
      const actual = begin(pass);
      const setBindGroup = actual.setBindGroup.bind(actual);
      actual.setBindGroup = (index, group, ...rest) => {
        const bytes = sizes.get(group) ?? 0;
        if (bytes > (largest.get(label) ?? 0)) largest.set(label, bytes);
        return setBindGroup(index, group, ...rest);
      };
      return actual;
    };
    return encoder;
  };
  return {
    largest,
    stop() {
      device.createBindGroup = createBindGroup;
      device.createCommandEncoder = createCommandEncoder;
    },
  };
}

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const BYTES = { f32: 4, f16: 2 };

export async function main(device, args) {
  const tool = option(args, "tool", "fold-af2");
  const lengths = option(args, "lengths", "59,118").split(",").map(Number);
  if (lengths.length !== 2 || lengths.some((n) => !Number.isSafeInteger(n) || n < 2)) {
    throw new RangeError("--lengths wants exactly two integers, e.g. 59,118");
  }
  const rest = args.filter((a) => !a.startsWith("--tool=") && !a.startsWith("--lengths=")
    && !a.startsWith("--sequence="));
  const limit = device.limits.maxStorageBufferBindingSize;
  const module = await import(`./${tool}.js`);

  // The largest byte size each labelled dispatch ever bound, per length.
  const seen = lengths.map(() => new Map());
  const original = WebGpuExecution.prototype.dispatch;
  // One label per pass, so a binding is attributed to the kernel that made it.
  setDeviceTuning(device, { batchComputePasses: false });
  const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
  try {
    for (let at = 0; at < lengths.length; at += 1) {
      const into = seen[at];
      WebGpuExecution.prototype.dispatch = function watched(encoder, pipeline, tensors,
                                                            x, y, z, label) {
        for (const tensor of tensors ?? []) {
          const bytes = (tensor?.elements ?? 0) * (BYTES[tensor?.storage ?? "f32"] ?? 4);
          const name = String(label ?? "(unlabelled)");
          if (bytes > (into.get(name) ?? 0)) into.set(name, bytes);
        }
        return original.call(this, encoder, pipeline, tensors, x, y, z, label);
      };
      const sequence = Array.from({ length: lengths[at] },
        (_, index) => ALPHABET[index % ALPHABET.length]).join("");
      // 🔴 BOTH SEAMS, MERGED. AF2 goes through `WebGpuExecution.dispatch` and
      // AF3 does not; whichever the wrapped tool uses, the larger number for a
      // label wins. A family that uses both is counted once.
      const bindings = watchBindings(device);
      try {
        await module.main(device, [...rest, `--sequence=${sequence}`]);
      } finally {
        bindings.stop();
      }
      for (const [name, bytes] of bindings.largest) {
        if (bytes > (into.get(name) ?? 0)) into.set(name, bytes);
      }
    }
  } finally {
    WebGpuExecution.prototype.dispatch = original;
  }

  // 🔴 A PROBE THAT SAW NOTHING MUST SAY SO, NOT RETURN A CLEAN BILL. This
  // patches `WebGpuExecution.prototype.dispatch`, which is AF2's seam and NOT
  // AF3's: the AF3 stacks call `dispatchWorkgroups` from six of their own
  // modules, so `--tool=fold` recorded **zero** labels and returned
  // `lowestCeiling: null` with an empty `labels` list. Read quickly that is
  // "no label is near the limit"; it actually means the instrument was pointed
  // at the wrong place. Same shape as this repository's `NaN > bound` and its
  // unguarded relRMS over a non-array - a comparison over nothing that passes.
  //
  // Extending it to AF3 means watching the BIND GROUP rather than the dispatch,
  // because AF3 has no single dispatch seam. Until someone does, this refuses.
  const watched = seen.reduce((sum, into) => sum + into.size, 0);
  if (watched === 0) {
    throw new Error(`${tool} issued no dispatch this probe could see: it patches`
      + " WebGpuExecution.prototype.dispatch, which the AF2 path uses and the AF3"
      + " stacks do not - they call dispatchWorkgroups directly. A ceiling"
      + " computed from no observations is not a ceiling. See the note above.");
  }

  const ratio = lengths[1] / lengths[0];
  const rows = [];
  for (const [label, big] of seen[1]) {
    const small = seen[0].get(label);
    if (small === undefined || small === 0 || big === 0) continue;
    const exponent = Math.log(big / small) / Math.log(ratio);
    // Where this label's largest binding reaches the limit, at that exponent.
    const ceiling = exponent < 0.05 ? Infinity
      : lengths[1] * (limit / big) ** (1 / exponent);
    rows.push({ label,
                bytesAt: { [lengths[0]]: small, [lengths[1]]: big },
                exponent: Math.round(exponent * 100) / 100,
                // How far the LARGEST sample is from the limit. A small number
                // means the ceiling below is extrapolated across orders of
                // magnitude, and any budget in between is invisible.
                sampleFraction: Math.round((big / limit) * 1000) / 1000,
                ceilingResidues: Number.isFinite(ceiling) ? Math.floor(ceiling) : null });
  }
  rows.sort((a, b) => (a.ceilingResidues ?? Infinity) - (b.ceilingResidues ?? Infinity));
  const binding = rows.find((row) => row.ceilingResidues !== null);
  // 🔴 THE COUNT IS THE POINT, NOT THE NAME. This used to return one label and
  // that is how it read as "window the triangle and the fold gets longer": the
  // triangle is simply first alphabetically among the dispatches that bind a
  // pair-shaped tensor, and on AF2 **twenty-eight labels reach the limit at the
  // same length**, because they all bind `L^2 * cZ` f32. Windowing any subset
  // of a tie moves the fold's ceiling by nothing - the next member of the group
  // refuses at the same residue count. So the groups are reported.
  const groups = new Map();
  for (const row of rows) {
    if (row.ceilingResidues === null) continue;
    const found = groups.get(row.ceilingResidues) ?? [];
    found.push(row.label);
    groups.set(row.ceilingResidues, found);
  }
  return {
    tool, lengths, maxStorageBufferBindingSize: limit,
    // The label that gives out first, and at what length.
    lowestCeiling: binding === undefined ? null
      : { label: binding.label, residues: binding.ceilingResidues },
    // ...and everything that gives out WITH it. A one-label fix is worth
    // something only where `labels` here is 1.
    lowestCeilingGroup: binding === undefined ? null
      : { residues: binding.ceilingResidues,
          labels: groups.get(binding.ceilingResidues).length,
          members: groups.get(binding.ceilingResidues) },
    ceilingGroups: [...groups.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([residues, members]) => ({ residues, labels: members.length, members })),
    labels: rows.length,
    // Named rather than left to be noticed: the lowest group is only a finding
    // if its largest sample is within reach of the limit.
    caveat: binding === undefined || binding.sampleFraction >= 1 / 16 ? null
      : `the lowest group extrapolates from ${Math.round(binding.sampleFraction * 1000) / 10}% `
        + `of the limit; re-run at lengths where it is closer, or a label that CHUNKS `
        + `above ${lengths[1]} residues will read as a ceiling`,
    tightest: rows.slice(0, 40),
  };
}

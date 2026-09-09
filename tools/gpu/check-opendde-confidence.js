/**
 * OpenDDE's per-pair confidence readouts: the GPU against the host reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-opendde-confidence.js \
 *       --model=/model-opendde-int5/manifest.json
 *
 * 🔴 THIS HEAD HAD NO DIFFERENTIAL GATE AT ALL. Its only check was a whole
 * OpenDDE fold, and a fold's gate is the chain's geometry - which the pLDDT and
 * the PAE do not touch. So the two readouts that are 1.6 seconds of that fold
 * could be moved to the GPU with nothing able to say whether they still
 * computed the same numbers.
 *
 * It is differential, not oracle: it says the GPU agrees with
 * `hostPairReadouts`, which is the code the fold used before.
 *
 * 🔴 AND THE FIXTURE IS THE BUNDLE'S WIDTHS, not typed-in ones - see
 * check-af3-diffusion-conditioning.js, where 128 typed in for OpenDDE's 384 hid
 * ten seconds a fold behind a residual of NaN.
 */
import { confidencePairInit, hostAtomReadouts, hostPairReadouts }
  from "../../src/af3/opendde-confidence.js";
import { openddeAtomReadouts, openddePairInit, openddePairReadouts }
  from "../../src/af3/opendde-confidence-webgpu.js";
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { openAf3Store, openddeConfidenceWeights } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const store = await openAf3Store(option(args, "model", "/model-opendde-int5/manifest.json"));
  // Only the readout tensors are used, but the loader is the loader: a bundle
  // whose widths were read wrongly fails here.
  const weights = await openddeConfidenceWeights(store, 0);
  const channels = weights.pairChannels;

  const results = {};
  // 🔴 TWO TOKEN COUNTS, ONE ODD. The kernel is a workgroup a row with a lane a
  // bin, and PDE reads the TRANSPOSED row - an index the square shape of the
  // fixture would let a symmetric mistake survive at one size.
  for (const tokens of (option(args, "n", "17,40")).split(",").map(Number)) {
    const pair = deterministic(tokens * tokens * channels, 909 + tokens);
    const expected = hostPairReadouts(pair, tokens, channels, weights);
    const actual = await openddePairReadouts(device, { pair, tokens, channels }, weights);
    const arm = {
      pae: relativeRms(actual.pae, expected.pae),
      pde: relativeRms(actual.pde, expected.pde),
    };
    // 🔴 AND THE TWO ARMS MUST DIFFER, because PDE is PAE on the symmetrised
    // pair and a kernel that ignored `symmetrise` would pass both against a
    // reference that... also ignored it. They are computed independently here,
    // so the separation says the flag reached the shader.
    arm.separation = relativeRms(expected.pde, expected.pae);
    results[tokens] = arm;
    console.log(`${tokens} tokens\tpae ${arm.pae.toExponential(2)}`
      + `\tpde ${arm.pde.toExponential(2)}\tpae-vs-pde ${arm.separation.toExponential(2)}`);
    for (const name of ["pae", "pde"]) {
      if (!(arm[name] <= 1e-5)) {
        const bad = (values) => {
          let count = 0;
          for (let i = 0; i < values.length; i += 1) if (!Number.isFinite(values[i])) count += 1;
          return `${count}/${values.length} non-finite`;
        };
        throw new Error(`${name} at ${tokens} tokens is ${arm[name]}, over 1e-5. `
          + `GPU: ${bad(actual[name])}; reference: ${bad(expected[name])}`);
      }
    }
    if (!(arm.separation > 1e-3)) {
      throw new Error(`PAE and PDE are ${arm.separation.toExponential(2)} apart at `
        + `${tokens} tokens: the symmetrisation reached neither, so neither arm `
        + "was checked against anything");
    }
  }
  // 🔴 AND THE PAIR THE HEAD STARTS FROM, which is the one tensor here that is
  // built on the device and never read back - so nothing downstream can notice
  // it being wrong except the fold's own geometry, which no confidence number
  // touches.
  const initResults = {};
  for (const tokens of [17, 40]) {
    const pairs = tokens * tokens;
    const inputWidth = weights.singleInputChannels;
    const pair = deterministic(pairs * channels, 313 + tokens);
    const singleInputs = deterministic(tokens * inputWidth, 314 + tokens);
    // 🔴 COORDINATES SPREAD OVER THE WHOLE BIN RANGE ON PURPOSE. The distance
    // embedding is a gather of one of 39 rows, with everything under 3.25 A
    // taking none and everything over 52 taking the last - so a fixture whose
    // atoms all sit in one bin would check one row and the clamp at neither
    // end. These run from 0 to about 70 A apart.
    const coordinates = new Float32Array(tokens * 3);
    for (let t = 0; t < tokens; t += 1) coordinates[t * 3] = t * (70 / Math.max(1, tokens - 1));
    const expected = confidencePairInit(pair, singleInputs, coordinates, tokens, weights);
    const allocator = new GpuBufferAllocator(device);
    const built = await openddePairInit(device, {
      tokens, channels, pair,
      s1: linear(singleInputs, tokens, inputWidth, channels, weights.s1),
      s2: linear(singleInputs, tokens, inputWidth, channels, weights.s2),
      coordinates, binStart: 3.25, binStep: 1.25,
    }, weights, allocator);
    const bytes = pairs * channels * 4;
    const readback = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(built.allocation.buffer, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    built.release();
    const residual = relativeRms(actual, expected);
    // The control: without the distance embedding this would still be pair +
    // s1 + s2, so the arm has to show that the embedding reached it.
    const withoutDistance = new Float32Array(expected.length);
    const s1 = linear(singleInputs, tokens, inputWidth, channels, weights.s1);
    const s2 = linear(singleInputs, tokens, inputWidth, channels, weights.s2);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        for (let d = 0; d < channels; d += 1) {
          withoutDistance[(i * tokens + j) * channels + d] =
            pair[(i * tokens + j) * channels + d] + s1[j * channels + d] + s2[i * channels + d];
        }
      }
    }
    const distanceShare = relativeRms(withoutDistance, expected);
    initResults[tokens] = { residual, distanceShare: Number(distanceShare.toFixed(4)) };
    console.log(`${tokens} tokens\tpair-init ${residual.toExponential(2)}`
      + `\tthe distance terms are ${distanceShare.toFixed(3)} of it`);
    if (!(residual <= 1e-6)) {
      throw new Error(`the device pair init is ${residual} against confidencePairInit`);
    }
    if (!(distanceShare > 1e-2)) {
      throw new Error(`dropping the distance embedding moves this fixture by only `
        + `${distanceShare.toExponential(2)}, so the arm would pass without it`);
    }
  }

  // 🔴 AND THE TWO PER-ATOM READOUTS, whose weight is chosen by the atom's dense
  // SLOT. A kernel that read `plddt_weight` as [c_s, bins] and broadcast it
  // would give every atom of a token the same score - which is plausible on a
  // backbone and wrong everywhere - so the fixture gives consecutive atoms
  // different slots and the control below says the slot reached the shader.
  const atomResults = {};
  const slots = weights.denseSlots;
  const cs = weights.singleChannels;
  for (const tokens of [17, 40]) {
    const atoms = tokens * slots;
    const single = deterministic(tokens * cs, 4242 + tokens);
    const atomToToken = new Int32Array(atoms);
    const atomToSlot = new Int32Array(atoms);
    for (let atom = 0; atom < atoms; atom += 1) {
      atomToToken[atom] = Math.floor(atom / slots);
      atomToSlot[atom] = atom % slots;
    }
    const input = { atomToToken, atomToSlot };
    const expected = hostAtomReadouts(single, atoms, cs, input, weights);
    const actual = await openddeAtomReadouts(device, {
      single, atoms, channels: cs, atomToToken, atomToSlot }, weights);
    const arm = {
      plddt: relativeRms(actual.plddt, expected.plddt),
      resolved: relativeRms(actual.resolved, expected.resolved),
    };
    // The slots' own separation: the first token's 24 atoms share a single
    // representation and differ only by which matrix they took.
    let low = Infinity;
    let high = -Infinity;
    for (let slot = 0; slot < slots; slot += 1) {
      low = Math.min(low, expected.plddt[slot]);
      high = Math.max(high, expected.plddt[slot]);
    }
    arm.slotSpread = Number((high - low).toFixed(4));
    atomResults[tokens] = arm;
    console.log(`${tokens} tokens\tplddt ${arm.plddt.toExponential(2)}`
      + `\tresolved ${arm.resolved.toExponential(2)}`
      + `\tone token's slots span ${arm.slotSpread} pLDDT`);
    for (const name of ["plddt", "resolved"]) {
      if (!(arm[name] <= 1e-5)) {
        throw new Error(`${name} at ${tokens} tokens is ${arm[name]}, over 1e-5`);
      }
    }
    if (!(arm.slotSpread > 1)) {
      throw new Error(`the 24 atoms of one token span ${arm.slotSpread} pLDDT, so the `
        + "dense slot did not choose a matrix and neither side is reading the head "
        + "this checkpoint has");
    }
  }

  return { channels, bins: { pae: weights.paeBins, pde: weights.pdeBins },
           slots, results, atomResults, initResults };
}

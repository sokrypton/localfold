/**
 * AF2's whole structure module in WebGL2 against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-structure-module.js [--falsify]
 *
 * 🔴 THIS IS THE COMPOSITION GATE FOR THE STRUCTURE SIDE, AND THE FIVE
 * PER-STAGE GATES CANNOT STAND IN FOR IT. Each of them hands its stage an input
 * it built itself, so none can see an error in what feeds what - and this module
 * has three such seams, every one of which produces a plausible structure when
 * wrong:
 *
 *   - the pair LayerNorm is HOISTED out of the eight iterations, because every
 *     iteration reads the same pair with the same weights. Normalising inside
 *     the loop is correct and eight times the work; normalising the WRONG thing
 *     once is not visible at the operation level.
 *   - the sidechain head reads the FIRST representation and the eighth, not the
 *     eighth twice. The initialiser's LayerNorm output has to survive all eight
 *     iterations to get there.
 *   - each iteration's attention reads the frames the PREVIOUS iteration wrote,
 *     and the frame advance rotates its translation by the OLD quaternion. Eight
 *     of them compound, so an error here is a structure that drifts rather than
 *     one that breaks.
 *
 * Six arms, because an error confined to one output would otherwise hide behind
 * the others: the two coordinate sets, the final representation, the frames and
 * both angle tensors. The frames are the sharp one - they are what the
 * coordinates are built from, and they are the only output that accumulates over
 * all eight iterations.
 *
 * 🔴 AND THE AFFINE UPDATE'S WEIGHT IS DELIBERATELY SMALL. At the usual
 * `1/sqrt(channels)` scale the update winds the frames around several times over
 * eight iterations, and the comparison then measures how two implementations
 * diverge along a chaotic trajectory rather than whether they compute the same
 * thing. `--affine-scale=` is the handle; the checkpoint's own weight is small
 * for the same reason, and reading a large-scale run as a port error would be a
 * day lost.
 *
 * The bars are measured floors, not constants: one f32 step of jitter on every
 * weight this module reads, randomly signed, which is what two implementations
 * whose inputs agree to f32 rounding are. Eight iterations of a transition and
 * an attention amplify that, and `--iterations=` is what shows by how much.
 */
import { StructureModuleGpu } from "../../webgpu/af2/structure/module.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  STRUCTURE_SHAPE, releaseGeometryTables, releaseStructureModuleWeights, structureModule,
  uploadGeometryTables, uploadStructureModuleWeights,
} from "../../webgl/af2/structure/module.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { syntheticGeometryTables, syntheticStructureWeights } from "./af2-synthetic-structure.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const length = Number(option(args, "length", "7"));
  const iterations = Number(option(args, "iterations", "8"));
  const affineScale = Number(option(args, "affine-scale", "0.05"));
  const shape = { ...STRUCTURE_SHAPE, iterations };
  const {
    msaChannels, structureChannels, pairChannels, sidechainHidden,
    heads, scalarQk, scalarV, pointQk, pointV,
  } = shape;

  const weights = syntheticStructureWeights({ length, ...shape });
  // `--affine-scale=` rescales the update's weight, which is the one knob that
  // decides whether eight iterations stay near their start or wander.
  if (affineScale !== 0.05) {
    const factor = affineScale / 0.05;
    weights.postAttention.affineWeight = Float32Array.from(
      weights.postAttention.affineWeight, (v) => v * factor);
  }
  const { tables, aatype, atom37ToAtom14, atom37Mask } = syntheticGeometryTables(length);

  const random = generator(2718);
  const noise = (count, spread) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * spread;
    return out;
  };
  const msaFirstRow = noise(length * msaChannels, 1.5);
  const pair = noise(length * length * pairChannels, 1.5);
  // One masked residue, because the attention's logits and the geometry both
  // read the mask and a fold's own mask is not all ones at the ends.
  const mask = new Float32Array(length).fill(1);
  if (length > 2) mask[length - 2] = 0;

  // The floor: one f32 step on every weight, randomly signed. Eight iterations
  // of a LayerNorm, four projections and an attention amplify it, and that
  // amplification IS what two correct f32 implementations differ by.
  const ULP = 1.2e-7;
  const jitter = generator(31337);
  const jitterTensor = (values) => Float32Array.from(
    values, (v) => v * (1 + ULP * (jitter() * 2 - 1)));
  const jitterGroup = (group) => Object.fromEntries(Object.entries(group).map(
    ([name, value]) => [name, Array.isArray(value)
      ? value.map(jitterTensor) : jitterTensor(value)]));
  const jitteredWeights = Object.fromEntries(Object.entries(weights).map(
    ([section, group]) => [section, jitterGroup(group)]));

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const upload = (values, label) => execution.upload(values, label);
  // 🔴 THE PERTURBATION IS THE PAIR, WHICH IS THE ONE INPUT EVERY OUTPUT READS -
  // AND IT SKEWS HALF OF EACH ROW'S CHANNELS, NOT ALL OF THEM. The MSA row
  // reaches the coordinates only through the activations and would leave the
  // frames' own arm weak; the pair goes into every iteration's attention and so
  // into the frames, the angles and both coordinate sets.
  //
  // 🔴 AND `v * 1.01 + 0.01` OVER THE WHOLE ROW REPORTED "0 of 6 arms noticed",
  // which is this repository's documented vacuous-control trap met for the
  // second time in this campaign: the pair's ONLY path into this module is
  // `normalisePair`, and a whole-row affine change is invisible to a LayerNorm -
  // the scale cancels in the variance and the shift cancels in the mean. Half a
  // row is not affine over the row, so it survives the normalise. A control has
  // to be checked against the FIRST operation its tensor meets.
  const inputs = {
    msaFirstRow: upload(msaFirstRow, "msa-row"),
    pair: upload(falsify
      ? Float32Array.from(pair,
        (v, i) => (i % pairChannels < pairChannels / 2 ? v * 1.01 + 0.01 : v))
      : pair, "pair"),
    mask: upload(mask, "mask"),
    aatype: upload(aatype, "aatype"),
    atom37ToAtom14: upload(atom37ToAtom14, "atom37-to-14"),
    atom37Mask: upload(atom37Mask, "atom37-mask"),
  };
  const uploadedTables = uploadGeometryTables(execution, tables);

  const run = (weightSet) => {
    const uploaded = uploadStructureModuleWeights(execution, weightSet);
    const result = structureModule(execution, {
      ...inputs, length, weights: uploaded, tables: uploadedTables, shape,
    });
    /** @type {Record<string, Float32Array>} */
    const values = {};
    for (const [name, tensor] of Object.entries(result)) {
      values[name] = Float32Array.from(execution.read(tensor));
      execution.release(tensor);
    }
    releaseStructureModuleWeights(execution, uploaded);
    return values;
  };
  const ours = run(weights);
  const shifted = run(jitteredWeights);

  const theirs = await new StructureModuleGpu(device).run({
    msaFirstRow, pair, mask, aatype, atom37ToAtom14, atom37Mask, length,
    weights, geometry: tables,
  });

  const override = option(args, "bar", null);
  const rows = [];
  const failures = [];
  /** @type {[string, string][]} */
  const ARMS = [
    ["atom14", "atom14"],
    ["atom37", "atom37"],
    ["finalRepresentation", "finalRepresentation"],
    ["affine", "affine"],
    ["angles", "angles"],
    ["unnormalizedAngles", "unnormalizedAngles"],
  ];
  for (const [name, key] of ARMS) {
    const floor = relativeRms(shifted[key], ours[key], `floor.${name}`);
    const bar = override !== null
      ? Number(override)
      : Number(Math.max(2e-7, floor * 3).toPrecision(3));
    const relative = relativeRms(ours[key], theirs[key], name);
    const ok = relative <= bar;
    rows.push({
      arm: name,
      relativeRms: Number(relative.toPrecision(4)),
      oneUlpFloor: Number(floor.toPrecision(3)),
      bar,
      ok,
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  releaseGeometryTables(execution, uploadedTables);
  for (const tensor of Object.values(inputs)) execution.release(tensor);
  const red = rows.filter((row) => !row.ok).length;
  return {
    shape: { length, ...shape, affineScale },
    falsify,
    barRule: override !== null ? `--bar=${override}` : "max(2e-7, oneUlpFloor * 3)",
    maskedResidue: length > 2 ? length - 2 : null,
    arms: rows,
    memory: execution.stats(),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation of the pair`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed the pair moving`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms, each at its own measured floor`
        : `FAILED: ${failures.join("; ")}`),
  };
}

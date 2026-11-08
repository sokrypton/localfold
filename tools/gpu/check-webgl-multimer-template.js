/**
 * AF2-MULTIMER's template embedder in WebGL2 against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-multimer-template.js [--falsify]
 *
 * Nine projections summed, where the monomer has one Linear over an 88-column
 * concatenation - and three of this stage's conventions produce a plausible
 * number when wrong, so each is its own arm:
 *
 *   - **a masked template still contributes.** Multimer's config has templates
 *     enabled and the wrapper adds this term unconditionally: the biases, a
 *     one-hot of aatype zero and the query embedding all survive the mask. The
 *     reference records what omitting it cost - "the pair track 30% out from the
 *     first evoformer block... shattered backbones at six copies".
 *   - **`templates` is the DIVISOR of the mean, not a count of real ones**, and
 *     it is inside the relu. The padded slot count is what divides, which is the
 *     same trap `templateFeatureMeanOnePass` records for rosettafold3: a stage
 *     exact at one slot and a quarter strength at four. The `templates=4` arm is
 *     the one that can see it.
 *   - **the pair blocks run their five operations in the OPPOSITE order to the
 *     monomer template stack's.** The reference says so: "AF2 monomer's
 *     TemplatePairStack attends first and multiplies second; multimer's
 *     TemplateEmbeddingIteration multiplies first... running one order with the
 *     other's weights is the kind of mistake that produces a plausible number."
 *     The `orderMoves` arm asserts the flag is not inert before any residual is
 *     believed.
 *
 * 🔴 AND THE NINE INDICES ARE NOT THE PACKED FEATURES' ORDER.
 * `template_pair_embedding_0..8` are the distogram, its mask, the aatype along
 * each axis, three unit-vector components, the backbone mask and the query
 * pair, while the packed geometry is bin, mask, three components, backbone
 * mask - so the geometry embeddings are 0, 1, 4, 5, 6, 7 and the aatype ones
 * are 2 and 3. A mis-mapped index is a term of the right magnitude in the wrong
 * place, which only a cross-check can see.
 *
 * The WGSL side has no class, only `encodeTemplateEmbedding`, and it adds its
 * update into the pair IN PLACE - so the comparison is this port's update plus
 * the original pair against the reference's updated one.
 */
import { encodeTemplateEmbedding } from "../../webgpu/af2/multimer/template.js";
import { WebGpuExecution } from "../../webgpu/runtime/execution.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  multimerTemplateEmbedding, multimerTemplateFeatures, releaseMultimerTemplateWeights,
  uploadMultimerTemplateWeights,
} from "../../webgl/af2/multimer/template.js";
import { uploadTemplatePairBlockWeights } from "../../webgl/af2/evoformer/block.js";
import { releaseEvoformerBlockWeights } from "../../webgl/af2/evoformer/block.js";
import { add } from "../../webgl/runtime/elementwise.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator, syntheticBlockWeights } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const TEMPLATE_CHANNELS = 64;
const DGRAM_BINS = 39;
const AATYPE_CLASSES = 22;

/** The WGSL embedder, which has no class of its own. */
async function runWgsl(device, input, weights, pairValues, pairMaskValues) {
  const execution = new WebGpuExecution(device);
  try {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const pair = execution.upload("mt.pair", pairValues, usage);
    const pairMask = execution.upload("mt.pair-mask", pairMaskValues);
    const encoder = device.createCommandEncoder({ label: "multimer-template" });
    device.pushErrorScope("validation");
    await encodeTemplateEmbedding(execution, encoder, input, weights, pair, pairMask);
    const readback = execution.createReadback("mt.readback", pair, encoder);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    return await execution.mapFloat32(readback);
  } finally {
    execution.release();
  }
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const length = Number(option(args, "length", "7"));
  const pairChannels = Number(option(args, "pair-channels", "128"));
  const blocks = Number(option(args, "blocks", "2"));
  const heads = 4;
  const triangleHidden = 32;
  const pairHidden = 128;
  const pairs = length * length;

  const random = generator(1492);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // The nine embeddings, in the reference's own order and widths.
  const rowsOf = (index) => (index === 0 ? DGRAM_BINS
    : index === 2 || index === 3 ? AATYPE_CLASSES
      : index === 8 ? pairChannels : 1);
  const pairEmbeddings = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((index) => ({
    weight: noise(rowsOf(index) * TEMPLATE_CHANNELS, 1 / Math.sqrt(rowsOf(index))),
    bias: noise(TEMPLATE_CHANNELS, 0.05),
  }));
  const stack = [];
  for (let block = 0; block < blocks; block += 1) {
    const all = syntheticBlockWeights({
      sequences: 1, length, cM: TEMPLATE_CHANNELS, cZ: TEMPLATE_CHANNELS, cOuter: 8,
      heads, triangleHidden, msaHidden: TEMPLATE_CHANNELS, pairHidden, seed: 71 + block,
    });
    stack.push({
      triangleMultiplicationOutgoing: all.triangleMultiplicationOutgoing,
      triangleMultiplicationIncoming: all.triangleMultiplicationIncoming,
      triangleAttentionStarting: all.triangleAttentionStarting,
      triangleAttentionEnding: all.triangleAttentionEnding,
      pairTransition: all.pairTransition,
    });
  }
  const weights = {
    stack,
    pairEmbeddings,
    queryNormScale: noise(pairChannels, 0.2, 1),
    queryNormOffset: noise(pairChannels, 0.1),
    outputNormScale: noise(TEMPLATE_CHANNELS, 0.2, 1),
    outputNormOffset: noise(TEMPLATE_CHANNELS, 0.1),
    outputWeight: noise(TEMPLATE_CHANNELS * pairChannels, 1 / Math.sqrt(TEMPLATE_CHANNELS)),
    outputBias: noise(pairChannels, 0.05),
  };

  // A template with both mask branches planted, as the monomer gate does: one
  // residue with no backbone, one with no CB.
  const aatype = new Int32Array(length);
  for (let residue = 0; residue < length; residue += 1) {
    aatype[residue] = Math.floor(random() * 20);
  }
  const atomMask = new Float32Array(length * 37).fill(1);
  const backboneGone = 2 % length;
  const betaGone = (length - 2 + length) % length;
  for (const slot of [0, 1, 2]) atomMask[backboneGone * 37 + slot] = 0;
  atomMask[betaGone * 37 + 3] = 0;
  const template = { aatype, atomMask, atomPositions: noise(length * 37 * 3, 11) };
  // 🔴 TWO CHAINS, BECAUSE A TEMPLATE WITHOUT `asymId` IS REFUSED - by the
  // reference and now by this port too. The refusal is the point: AF2-multimer
  // masks a template's geometry across chains, so assuming one chain lets it
  // speak about pairs it has never seen in one frame. Two chains also makes the
  // mask non-trivial, which an all-ones mask would not.
  const asymId = Float32Array.from(
    { length }, (_, residue) => (residue < Math.ceil(length / 2) ? 0 : 1));

  const pairValues = noise(pairs * pairChannels, 1.5);
  const pairMaskValues = new Float32Array(pairs).fill(1);
  pairMaskValues[1] = 0;

  const ULP = 1.2e-7;
  const jitter = generator(1066);
  const jitterValues = (values) => Float32Array.from(
    values, (v) => v * (1 + ULP * (jitter() * 2 - 1)));
  const jitterSet = (value) => {
    if (value instanceof Float32Array) return jitterValues(value);
    if (Array.isArray(value)) return value.map(jitterSet);
    if (typeof value === "number") return value;
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jitterSet(v)]));
  };

  setDeviceTuning(device, {
    allowHostWeightPacking: true,
    opmVectorArithmetic: "f32",
    triangleContractArithmetic: "f32",
  });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const upload = (values, label) => execution.upload(values, label);
  // The control skews half of each pair row's channels: the pair's first stop
  // here is the query LayerNorm, and a whole-row affine change is invisible to
  // one - the trap that read "0 of 6" in the structure module's gate.
  const pairTensor = upload(falsify
    ? Float32Array.from(pairValues,
      (v, i) => (i % pairChannels < pairChannels / 2 ? v * 1.01 + 0.01 : v))
    : pairValues, "mt.pair");
  const pairMask = upload(pairMaskValues, "mt.pair-mask");

  /** This port's whole stage, with the knobs this gate varies. */
  const run = (weightSet, { withTemplate, templates, order, dither }) => {
    const features = multimerTemplateFeatures(
      { template: withTemplate ? template : undefined, length, asymId });
    const geometry = upload(features.geometry, "mt.geometry");
    // The aatype itself, which the shader indexes - the reference gathers the
    // two weight rows on the host instead, so this arm compares a GPU gather
    // against a host one as well as the arithmetic around it.
    const aatypeTensor = upload(features.aatype, "mt.aatype");
    const uploaded = uploadMultimerTemplateWeights(execution, weightSet);
    const blockWeights = weightSet.stack.map(
      (block) => uploadTemplatePairBlockWeights(execution, block));
    execution.halfDither = dither;
    const update = multimerTemplateEmbedding(execution, {
      pair: pairTensor, geometry, aatype: aatypeTensor, pairMask,
      weights: uploaded, blockWeights,
      length, pairChannels, templateChannels: TEMPLATE_CHANNELS,
      triangleHidden, pairHidden, templates,
      ...(order === "monomer" ? { templateOrder: "monomer" } : {}),
    });
    execution.halfDither = 0;
    const updated = add(execution, pairTensor, update);
    const values = Float32Array.from(execution.read(updated));
    const updateValues = Float32Array.from(execution.read(update));
    for (const tensor of [geometry, aatypeTensor, update, updated]) {
      execution.release(tensor);
    }
    releaseMultimerTemplateWeights(execution, uploaded);
    for (const block of blockWeights) releaseEvoformerBlockWeights(execution, block);
    return { values, update: updateValues };
  };

  const rows = [];
  const failures = [];
  const arm = (name, got, want, floor) => {
    const bar = Number(Math.max(2e-7, floor * 3).toPrecision(3));
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({
      arm: name, relativeRms: Number(relative.toPrecision(4)),
      oneUlpFloor: Number(floor.toPrecision(3)), bar, ok,
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  /** @type {{name: string, withTemplate: boolean, templates: number}[]} */
  const CASES = [
    { name: "a real template", withTemplate: true, templates: 1 },
    { name: "a masked template", withTemplate: false, templates: 1 },
    { name: "four padded slots", withTemplate: true, templates: 4 },
  ];
  let orderMoves;
  for (const { name, withTemplate, templates } of CASES) {
    const options = { withTemplate, templates, order: "multimer" };
    const ours = run(weights, { ...options, dither: 0 });
    const shifted = run(jitterSet(weights), { ...options, dither: 0 });
    const dithered = run(weights, { ...options, dither: ULP });
    const floor = Math.max(
      relativeRms(shifted.values, ours.values, `floor.jitter.${name}`),
      relativeRms(dithered.values, ours.values, `floor.dither.${name}`));
    const theirs = await runWgsl(device, {
      length, pairChannels, templates, asymId,
      template: withTemplate ? template : undefined,
    }, weights, pairValues, pairMaskValues);
    arm(name, ours.values, theirs, floor);
    if (orderMoves === undefined) {
      // 🔴 THE ORDER FLAG MUST MOVE THE ANSWER. Running the monomer order with
      // these weights has to differ, or both sides could be ignoring
      // `templateOrder` and every arm above would be a monomer comparison.
      // 🔴 MEASURED ON THE UPDATE, NOT ON THE UPDATED PAIR. The arms compare
      // `pair + update` because that is what the reference writes, and the pair
      // is the larger of the two by more than an order of magnitude - so the
      // same order change reads 2.6e-5 on the sum and far more on the update
      // itself. A control measured through a dilution is a control that reports
      // "inert" about something that is not.
      const monomerOrder = run(weights, { ...options, order: "monomer", dither: 0 });
      orderMoves = Number(
        relativeRms(monomerOrder.update, ours.update, "order.moves").toPrecision(3));
    }
  }
  const ORDER_FLOOR = 1e-4;
  if (orderMoves < ORDER_FLOOR) {
    failures.push(`templateOrder is INERT: the monomer order agrees to ${orderMoves}`);
  }

  execution.release(pairTensor);
  execution.release(pairMask);
  const red = rows.filter((row) => !row.ok).length;
  return {
    shape: { length, pairChannels, templateChannels: TEMPLATE_CHANNELS, blocks, heads },
    falsify,
    chains: [Math.ceil(length / 2), length - Math.ceil(length / 2)],
    backboneMaskedAt: backboneGone,
    pseudoBetaMaskedAt: betaGone,
    arms: rows,
    orderMoves,
    orderFloor: ORDER_FLOOR,
    memory: execution.stats(),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation of the pair`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed the pair moving`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms at their measured floors, and the block order moves the answer`
        : `FAILED: ${failures.join("; ")}`),
  };
}

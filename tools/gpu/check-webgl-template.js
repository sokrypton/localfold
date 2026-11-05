/**
 * AF2-monomer's template stack in WebGL2 against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-template.js [--falsify]
 *
 * Five arms, and the first one is the only genuinely new arithmetic in the
 * stage: the 88-column `embedding2d`, compared against an f64 reference that
 * MATERIALISES the concatenation and multiplies all 88 columns where the shader
 * gathers only the nonzero ones. The other four compose that kernel with the
 * pair blocks and the pointwise projections and compare against
 * `QueryOnlyTemplateGpu`.
 *
 * 🔴 THE BLOCK ORDER IS THE THING THIS GATE EXISTS FOR. A template pair block is
 * the pair half of an Evoformer block without the outer product mean, and it is
 * NOT that half with the mean deleted: `encodeTemplatePairBlock` runs both
 * triangle ATTENTIONS first and the two multiplications after, where
 * `encodeEvoformerPairBlock` runs the multiplications first. Each sub-layer
 * reads what the previous one wrote, so the two orders compute different
 * tensors from the same weights - and nothing at the operation level can see
 * it. Two blocks rather than one, because a single block cannot distinguish an
 * error in the block from an error in carrying its output into the next.
 *
 * 🔴 AND THE ARMS REACH BOTH SIDES OF THREE CONVENTIONS THAT A PLAIN TEMPLATE
 * NEVER TOUCHES:
 *
 *   - a residue with its BACKBONE missing: the shader's whole 88-column sum is
 *     skipped and the embedding is exactly the bias, which is AF2's `act *=
 *     template_mask_2d` with the bias added afterwards. One residue's N, CA and
 *     C are masked out so that branch runs.
 *   - a residue with its CB missing but its backbone present: the pseudo-beta
 *     mask is zero and the DISTOGRAM STILL CONTRIBUTES (`maskDistogram: false`
 *     for the monomer, the opposite of AF3 and the multimer). Planted
 *     separately, because masking the backbone would hide it.
 *   - `use_template_unit_vector`, which is false in every shipped monomer
 *     config and so leaves columns 84..86 dead. It is a baked constant in the
 *     WebGL2 source, so the TRUE path is its own arm rather than an untested
 *     branch.
 *
 * And a no-template arm, because a masked template does NOT produce a zero
 * update here: the bias survives the mask, runs the whole stack and both
 * projections, and a port that short-circuited to zeros would look right to
 * every other arm.
 *
 * 🔴 THE PERTURBATION IS THE EMBEDDING BIAS, WHICH IS THE ONE TENSOR EVERY ARM
 * READS. The weight reaches four of the five - with no template the backbone
 * mask is zero everywhere and the weight contributes nothing at all, so an arm
 * that matters most for the mask convention would have come back identical and
 * the inverse verdict would have read "4 of 5". That is the fourth vacuous
 * control found in this campaign and the reason every gate here states which
 * tensor its control moves.
 *
 * The host half is compared too, not shared: `templateFeatures` here and
 * `chainMaskFor` plus the aatype fill inside `QueryOnlyTemplateGpu.run` are two
 * transcriptions of the same two conventions - an uncovered position is the GAP
 * type (21), and a monomer with no `asymId` is one chain - and the two drivers
 * are what this compares. The shared `templateGeometry` below them is one
 * implementation on both sides and is NOT under test here.
 */
import { QueryOnlyTemplateGpu } from "../../webgpu/af2/evoformer/template.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  FEATURE_COLUMNS, releaseTemplateWeights, templateEmbedSource, templateEmbedding,
  templateFeatures, uploadTemplateWeights,
} from "../../webgl/af2/evoformer/template.js";
import {
  releaseEvoformerBlockWeights, uploadTemplatePairBlockWeights,
} from "../../webgl/af2/evoformer/block.js";
import { GEOMETRY_STRIDE } from "../../shared/af3/featurise/template-features.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator, syntheticBlockWeights } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * `embedding2d` in f64, written from the operation rather than from the shader.
 *
 * The 88 columns are built as a vector and every one of them is multiplied,
 * where both GPU paths gather the handful that can be nonzero. Column 87 is the
 * backbone mask ITSELF rather than a one - which agrees with both of them for
 * any 0/1 mask, and `backboneMask2d` is a product of atom masks and a chain
 * mask, so it cannot be anything else.
 */
function embedReference(geometry, aatype, weight, bias, length, channels, unitVectors) {
  const out = new Float64Array(length * length * channels);
  const columns = new Float64Array(FEATURE_COLUMNS);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const entry = i * length + j;
      const g = entry * GEOMETRY_STRIDE;
      const backbone = geometry[g + 5];
      for (let channel = 0; channel < channels; channel += 1) {
        out[entry * channels + channel] = bias[channel];
      }
      if (backbone === 0) continue;
      columns.fill(0);
      const bin = geometry[g];       // stored plus one: 0 is "no bin at all"
      if (bin > 0) columns[bin - 1] = 1;
      columns[39] = geometry[g + 1];
      columns[40 + Math.min(Math.max(aatype[j], 0), 21)] = 1;
      columns[62 + Math.min(Math.max(aatype[i], 0), 21)] = 1;
      if (unitVectors) {
        columns[84] = geometry[g + 2];
        columns[85] = geometry[g + 3];
        columns[86] = geometry[g + 4];
      }
      columns[87] = backbone;
      for (let channel = 0; channel < channels; channel += 1) {
        let inner = 0;
        for (let column = 0; column < FEATURE_COLUMNS; column += 1) {
          inner += columns[column] * weight[column * channels + channel];
        }
        out[entry * channels + channel] += backbone * inner;
      }
    }
  }
  return out;
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const length = Number(option(args, "length", "9"));
  const blocks = Number(option(args, "blocks", "2"));
  // Two bisect handles: `--tri-att-f32` and `--tri-f32` lift this port's f16
  // intermediates in the triangle attentions and the triangle multiplications,
  // which is how the shipped block's own unswitchable choices were found.
  const triangleAttentionPrecision = flag(args, "tri-att-f32") ? "f32" : undefined;
  const trianglePrecision = flag(args, "tri-f32") ? "f32" : undefined;
  const templateChannels = 64;
  const pairChannels = 128;
  const heads = 4;
  const triangleHidden = 32;
  const pairHidden = 128;
  const projected = 64;
  const shape = {
    length, templateChannels, pairChannels, heads, triangleHidden, pairHidden, projected, blocks,
  };

  const random = generator(5150);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // A template with both mask branches planted. `atomMask` is atom37, so slots
  // 0, 1 and 2 are N, CA and C - the backbone frame - and slot 3 is CB.
  const aatype = new Int32Array(length);
  for (let residue = 0; residue < length; residue += 1) {
    aatype[residue] = Math.floor(random() * 20);
  }
  const backboneGone = 2 % length;
  const betaGone = (length - 2 + length) % length;
  const atomMask = new Float32Array(length * 37).fill(1);
  for (const slot of [0, 1, 2]) atomMask[backboneGone * 37 + slot] = 0;
  atomMask[betaGone * 37 + 3] = 0;
  const template = {
    aatype,
    atomMask,
    // Spread wide enough that the 39 distogram bins are genuinely used, and
    // close enough that some pairs fall under the first edge at 3.25 A, which
    // lands in no bin at all.
    atomPositions: noise(length * 37 * 3, 11),
  };

  const pairMask = new Float32Array(length * length).fill(1);
  pairMask[1] = 0;
  pairMask[length * length - 2] = 0;

  const blockSource = [];
  for (let block = 0; block < blocks; block += 1) {
    const all = syntheticBlockWeights({
      sequences: 1, length, cM: templateChannels, cZ: templateChannels, cOuter: 8,
      heads, triangleHidden, msaHidden: templateChannels, pairHidden, seed: 31 + block,
    });
    blockSource.push({
      triangleAttentionStarting: all.triangleAttentionStarting,
      triangleAttentionEnding: all.triangleAttentionEnding,
      triangleMultiplicationOutgoing: all.triangleMultiplicationOutgoing,
      triangleMultiplicationIncoming: all.triangleMultiplicationIncoming,
      pairTransition: all.pairTransition,
    });
  }

  const stageWeights = {
    embeddingWeight: noise(FEATURE_COLUMNS * templateChannels,
      1 / Math.sqrt(FEATURE_COLUMNS)),
    embeddingBias: noise(templateChannels, 0.1),
    outputNormScale: noise(templateChannels, 0.2, 1),
    outputNormOffset: noise(templateChannels, 0.1),
    valueWeight: noise(templateChannels * projected, 1 / Math.sqrt(templateChannels)),
    outputWeight: noise(projected * pairChannels, 1 / Math.sqrt(projected)),
    outputBias: noise(pairChannels, 0.05),
  };
  // The control moves the embedding BIAS, the one tensor every arm reads.
  const ourWeights = falsify
    ? { ...stageWeights, embeddingBias: Float32Array.from(stageWeights.embeddingBias,
      (v) => v * 1.01 + 0.01) }
    : stageWeights;

  // 🔴 THE BAR IS A MEASURED FLOOR, AND A CONDITION NUMBER IS THE WRONG
  // INSTRUMENT FOR THIS CHAIN - which cost a run to learn. A 1e-5 nudge of the
  // embedding bias gives a tidy ratio (0.42 on the no-template arm), and that
  // ratio says the chain is CONTRACTIVE while two blocks disagree 27x more than
  // one. The reason is that a bias nudge is UNIFORM over the pairs and a
  // rounding difference is not: with no template every pair carries the same
  // value, so the uniform mode is all such a probe can excite, and the modes
  // that actually amplify are invisible to it.
  //
  // `execution.halfDither` is the instrument that fits, and it is the same one
  // `check-webgl-evoformer-stack.js` had to reach for: a relative dither of one
  // f32 step applied INSIDE every f16 rounding. Two f32 values 1.2e-7 apart that
  // straddle an f16 boundary round 5e-4 apart, so the agreement floor of a chain
  // that quantises is set by HOW MANY elements straddle - a discrete count, not
  // a smooth function of the perturbation, and nothing applied to the chain's
  // input can reach it because the LayerNorms absorb that.
  //
  // 🔴 AND THE f16 IS THE REFERENCE'S CHOICE, MEASURED NOT ASSUMED.
  // `--tri-att-f32` lifts this port's rounding in the triangle attentions and
  // the no-template arm goes from 6.8e-6 to **5.4e-4**, eighty times WORSE - so
  // the WGSL template block stores its attentions in f16 exactly as the main
  // block does, and matching it is what the agreement rests on. `--tri-f32`
  // moves nothing, which is the control: the triangle multiplications are f32 on
  // both sides already.
  //
  // Beside the dither, one f32 step of jitter on every weight this stage reads,
  // randomly signed - "two implementations whose inputs agree to f32 rounding",
  // which is what the two sides are. The pair blocks' own weights are not
  // jittered because the dither already perturbs inside them at every rounding
  // point, which is where their differences live.
  const ULP = 1.2e-7;
  const jitter = generator(4077);
  const jitteredWeights = Object.fromEntries(Object.entries(ourWeights).map(
    ([name, values]) => [name, Float32Array.from(values,
      (v) => v * (1 + ULP * (jitter() * 2 - 1)))]));

  setDeviceTuning(device, { allowHostWeightPacking: true });
  // ...and f32 throughout, or this compares an f32 chain against an f16 one
  // everywhere except where the reference's own unswitchable choices put f16.
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadTemplateWeights(execution, ourWeights);
  const jittered = uploadTemplateWeights(execution, jitteredWeights);
  const uploadedBlocks = blockSource.map(
    (block) => uploadTemplatePairBlockWeights(execution, block));
  const maskTensor = execution.upload(pairMask, "template.pair-mask");

  const override = option(args, "bar", null);
  const barFor = (floor) => (override !== null
    ? Number(override)
    : Number(Math.max(2e-7, floor * 3).toPrecision(3)));
  const rows = [];
  const failures = [];
  const arm = (name, got, want, floor) => {
    const armBar = barFor(floor);
    const relative = relativeRms(got, want, name);
    const ok = relative <= armBar;
    rows.push({
      arm: name,
      relativeRms: Number(relative.toPrecision(4)),
      oneUlpFloor: Number(floor.toPrecision(3)),
      bar: armBar,
      ok,
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${armBar}`);
  };

  /** A run, its jittered twin and its dithered twin: the arm and its floor. */
  const withFloor = (run) => {
    const once = (weights, dither) => {
      const tensor = run(weights, dither);
      const values = Float32Array.from(execution.read(tensor));
      execution.release(tensor);
      return values;
    };
    const ours = once(uploaded, 0);
    const shifted = once(jittered, 0);
    const dithered = once(uploaded, ULP);
    return {
      values: ours,
      floor: Math.max(
        relativeRms(shifted, ours, "floor.jitter"),
        relativeRms(dithered, ours, "floor.dither")),
    };
  };

  // Arm 1: the embedding kernel alone, against the f64 reference.
  {
    const features = templateFeatures({ template, length });
    const geometry = execution.upload(features.geometry, "embed.geometry");
    const aatypeTensor = execution.upload(features.aatype, "embed.aatype");
    const { values, floor } = withFloor((weights, dither) => {
      execution.halfDither = dither;
      const output = execution.allocate(length * length * templateChannels, "embed.output");
      execution.dispatch({
        source: templateEmbedSource({ length, templateChannels, unitVectors: false }),
        output,
        inputs: {
          biasTex: weights.embeddingBias, weightTex: weights.embeddingWeight,
          geometryTex: geometry, aatypeTex: aatypeTensor,
        },
        label: "embed",
      });
      execution.halfDither = 0;
      return output;
    });
    arm("embedding (f64 reference)", values, embedReference(
      features.geometry, features.aatype, stageWeights.embeddingWeight,
      stageWeights.embeddingBias, length, templateChannels, false), floor);
    execution.release(geometry);
    execution.release(aatypeTensor);
  }

  /** One whole-stage comparison against the WGSL class, with its own floor. */
  const stage = async (name, options) => {
    const { blockCount = blocks, useTemplateUnitVector = false, withTemplate = true } = options;
    const slot = withTemplate ? template : undefined;
    const features = templateFeatures({ template: slot, length });
    const geometry = execution.upload(features.geometry, `${name}.geometry`);
    const aatypeTensor = execution.upload(features.aatype, `${name}.aatype`);
    const { values, floor } = withFloor((weights, dither) => {
      execution.halfDither = dither;
      const output = templateEmbedding(execution, {
        geometry, aatype: aatypeTensor, pairMask: maskTensor, weights,
        blockWeights: uploadedBlocks.slice(0, blockCount),
        length, templateChannels, pairChannels, triangleHidden, pairHidden,
        useTemplateUnitVector, triangleAttentionPrecision, trianglePrecision,
      });
      execution.halfDither = 0;
      return output;
    });
    const theirs = await new QueryOnlyTemplateGpu(device).run({
      length, templateChannels, pairChannels, pairMask,
      weights: { ...stageWeights, blockWeights: blockSource.slice(0, blockCount) },
      template: slot, useTemplateUnitVector,
    });
    arm(name, values, theirs.pairUpdate, floor);
    execution.release(geometry);
    execution.release(aatypeTensor);
  };

  await stage("pointwise only (0 blocks)", { blockCount: 0 });
  await stage(`${blocks} blocks`, {});
  await stage("unit vectors on", { useTemplateUnitVector: true });
  await stage("no template", { withTemplate: false });

  // 🔴 AND THE SAME STAGE TWICE IN ONE PAGE, BIT FOR BIT. `bench-af2-warm.js`
  // asks this of the WebGPU fold and CLAUDE.md gives it a row of its own: a
  // difference between two runs of one graph over one input is a MEMORY bug - a
  // texture handed back to the pool and read before it is written - and not a
  // precision question, so it is counted exactly rather than compared to a bar.
  // It is cheap here and it is what rules out the pooled-texture explanation for
  // this stage's floors.
  let repeatDiffering = 0;
  {
    const features = templateFeatures({ template, length });
    const geometry = execution.upload(features.geometry, "repeat.geometry");
    const aatypeTensor = execution.upload(features.aatype, "repeat.aatype");
    const once = () => {
      const output = templateEmbedding(execution, {
        geometry, aatype: aatypeTensor, pairMask: maskTensor, weights: uploaded,
        blockWeights: uploadedBlocks,
        length, templateChannels, pairChannels, triangleHidden, pairHidden,
        triangleAttentionPrecision, trianglePrecision,
      });
      const values = Float32Array.from(execution.read(output));
      execution.release(output);
      return values;
    };
    const first = once();
    const second = once();
    for (let i = 0; i < first.length; i += 1) {
      if (first[i] !== second[i]) repeatDiffering += 1;
    }
    if (repeatDiffering !== 0) {
      failures.push(`repeat: ${repeatDiffering} of ${first.length} elements differ between`
        + " two runs of the same stage over the same input");
    }
    execution.release(geometry);
    execution.release(aatypeTensor);
  }

  releaseTemplateWeights(execution, uploaded);
  releaseTemplateWeights(execution, jittered);
  for (const block of uploadedBlocks) releaseEvoformerBlockWeights(execution, block);
  execution.release(maskTensor);
  const red = rows.filter((row) => !row.ok).length;
  return {
    shape,
    falsify,
    precision: { triangleAttentionPrecision, trianglePrecision },
    barRule: override !== null ? `--bar=${override}` : "max(2e-7, oneUlpFloor * 3)",
    backboneMaskedAt: backboneGone,
    pseudoBetaMaskedAt: betaGone,
    arms: rows,
    repeatDiffering,
    memory: execution.stats(),
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation of the embedding bias`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed the embedding bias moving`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms, each at its own measured floor`
        : `FAILED: ${failures.join("; ")}`),
  };
}

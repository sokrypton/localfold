/**
 * Two Evoformer blocks in a row, both stacks, against their WGSL counterparts.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-evoformer-stack.js [--falsify]
 *
 * 🔴 TWO BLOCKS, BECAUSE WITH ONE A STACK AND A BLOCK ARE THE SAME THING. The
 * only thing a stack can get wrong that a block cannot is feeding a block the
 * WRONG representation - the previous block's INPUT instead of its output, or
 * the two tracks crossed - and that is invisible to every block-level gate,
 * because a block handed the right input returns the right answer however it
 * got there. With two blocks the second depends on the first, so a stack that
 * re-ran the first block's input agrees with a one-block reference and differs
 * here.
 *
 * The two blocks get INDEPENDENT weights (different seeds), for the same reason:
 * with identical weights a stack that ran block 0 twice would agree.
 *
 * 🔴 AND BOTH STACKS, because the extra one is where AF2 puts the thousands of
 * alignment rows the main stack never sees - it exists to get their information
 * into the pair track and is then discarded. Its MSA is compared anyway: an
 * error there reaches the pair through the outer product mean, so leaving it
 * unchecked would be leaving the only thing that stack is for unchecked.
 *
 * 🔴 AND IT REPORTS A STANDING DIVERGENCE THAT IS NOT YET EXPLAINED, with the
 * evidence that narrows it and the handles that narrowed it. Both whole BLOCKS
 * match their WGSL counterparts at their floors (`check-webgl-evoformer-block.js`:
 * 5.8e-8 to 1.2e-6) and so does a ONE-block stack here - 9.3e-8 and 8.4e-8. Over
 * two blocks the main stack's MSA reaches **2.5e-5**, and over three **5.4e-5**.
 *
 * What is established, each by measurement:
 *
 *   - it is MSA COLUMN ATTENTION. `--disable=msa-column` takes the two-block
 *     main.msa arm from 2.5e-5 to 9.0e-8; no other sub-layer moves it.
 *   - column attention ALONE is innocent. With every other sub-layer silenced it
 *     is 4.8e-8 at one block and 7.3e-8 at three, so the kernel is right and the
 *     input it receives in composition is what matters.
 *   - it is NOT the loop. `--identity-after=1` zeroes the second block's output
 *     projections on BOTH sides, which makes it an identity, and reproduces the
 *     one-block numbers EXACTLY - so no representation is crossed and nothing is
 *     released early.
 *   - it is NOT amplification, which is the surprising part. A one-f32-step
 *     jitter at the BLOCK BOUNDARY - where the two implementations actually
 *     differ - moves main.msa by 9.0e-8, not 2.5e-5. The extra stack's arms ARE
 *     amplification and sit at their floors (1.8e-5 against 7.5e-5 at two
 *     blocks); the main stack's are not.
 *   - the weights matter. Seed 138 is at the floor on all four arms where seed
 *     101 is not, so it is input-dependent rather than systematic - and the
 *     EXTRA stack exceeds its floor at one block already (3.1e-6 and 3.4e-5),
 *     where the block gate's own shapes put it at 6.0e-8 and 1.2e-6.
 *
 * A standing hypothesis, untested: the shipped block rounds its normalised
 * tensor to f16, and two f32 values differing by 1e-7 that straddle an f16
 * boundary round to values differing by 5e-4. A coarse grid turns a tiny
 * difference into a large one for the few elements that straddle, and a jitter
 * of one f32 step is too small to make any of 320 elements straddle - which
 * would explain why the floor probe reads 9e-8 while the real divergence does
 * not. Confirming it needs a probe that counts straddles rather than one that
 * jitters.
 *
 * `MAIN_STACK_DIVERGENCE` is an allowance with a number, not a bar, and
 * `--strict` removes it. Every handle above stays: a gate that can bisect itself
 * is worth more than one that reports a number.
 */
import { EvoformerStackGpu, ExtraMsaStackGpu } from "../../webgpu/af2/evoformer/stack.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  evoformerStack, extraMsaStack,
} from "../../webgl/af2/evoformer/stack.js";
import {
  releaseEvoformerBlockWeights, uploadEvoformerBlockWeights,
} from "../../webgl/af2/evoformer/block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import {
  generator, syntheticActivations, syntheticBlockWeights,
} from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const blocks = Number(option(args, "blocks", "2"));
  const shape = {
    sequences: Number(option(args, "sequences", "4")),
    length: Number(option(args, "length", "5")),
    cM: 16,
    cZ: 16,
    cOuter: 8,
    heads: 4,
    triangleHidden: 16,
    msaHidden: 32,
    pairHidden: 32,
  };
  const { sequences, length, cM, cZ, cOuter, triangleHidden, msaHidden, pairHidden } = shape;
  const blockShape = { sequences, length, cM, cZ, cOuter, triangleHidden };
  const activations = syntheticActivations(shape);
  // Half an MSA row 1% out, on this side only - a whole row is invisible to the
  // LayerNorm that reads it first, and one element is below the bar.
  const skewed = falsify
    ? Float32Array.from(activations.msa, (v, i) => (i < cM / 2 ? v * 1.01 + 0.01 : v))
    : activations.msa;

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);

  const rows = [];
  const failures = [];
  const conditions = {};

  for (const variant of ["main", "extra"]) {
    const global = variant === "extra";
    const blockWeights = Array.from({ length: blocks },
      (_, block) => syntheticBlockWeights({
        ...shape,
        global,
        // `--first-seed` shifts which weight set a one-block run uses, which is
        // how "is it these weights or is it this input" gets answered.
        seed: Number(option(args, "first-seed", "101")) + block * 37,
      }));
    // 🔴 A LOOP TEST THAT DOES NOT DEPEND ON THE BLOCKS BEING RIGHT. Zeroing
    // every sub-layer's output projection makes a block an identity - each
    // residual adds zero - on BOTH sides, so `--identity-after=1` must
    // reproduce the one-block numbers exactly. If it does not, the fault is the
    // loop: a block fed the wrong representation, or a tensor released while
    // the next block still needs it.
    // ...and the same silencing per SUB-LAYER, across every block, which is how
    // a stack-level discrepancy gets attributed to one of the nine.
    const disabled = new Set((option(args, "disable", "") || "").split(",").filter(Boolean));
    if (disabled.size > 0) {
      for (const w of blockWeights) {
        const zero = (group, names) => {
          for (const name of names) group[name] = new Float32Array(group[name].length);
        };
        const SILENCERS = {
          "msa-row": () => zero(w.msaRowAttention.attention, ["outputWeight", "outputBias"]),
          "msa-column": () => zero(
            global ? w.msaColumnGlobalAttention : w.msaColumnAttention.attention,
            ["outputWeight", "outputBias"]),
          "msa-transition": () => zero(w.msaTransition, ["secondWeight", "secondBias"]),
          opm: () => zero(w.outerProductMean, ["outputWeight", "outputBias"]),
          "tri-out": () => zero(w.triangleMultiplicationOutgoing,
            ["linearZWeight", "linearZBias"]),
          "tri-in": () => zero(w.triangleMultiplicationIncoming, ["linearZWeight", "linearZBias"]),
          "tri-att-start": () => zero(w.triangleAttentionStarting.attention,
            ["outputWeight", "outputBias"]),
          "tri-att-end": () => zero(w.triangleAttentionEnding.attention,
            ["outputWeight", "outputBias"]),
          "pair-transition": () => zero(w.pairTransition, ["secondWeight", "secondBias"]),
        };
        for (const name of disabled) {
          const silencer = SILENCERS[name];
          if (silencer === undefined) {
            throw new RangeError(
              `--disable: no sub-layer \`${name}\`; one of ${Object.keys(SILENCERS).join(", ")}`);
          }
          silencer();
        }
      }
    }

    const identityAfter = option(args, "identity-after", null);
    if (identityAfter !== null) {
      for (let block = Number(identityAfter); block < blocks; block += 1) {
        const w = blockWeights[block];
        const zero = (group, names) => {
          for (const name of names) group[name] = new Float32Array(group[name].length);
        };
        zero(w.msaRowAttention.attention, ["outputWeight", "outputBias"]);
        zero(global ? w.msaColumnGlobalAttention : w.msaColumnAttention.attention,
          ["outputWeight", "outputBias"]);
        zero(w.msaTransition, ["secondWeight", "secondBias"]);
        zero(w.outerProductMean, ["outputWeight", "outputBias"]);
        zero(w.triangleMultiplicationOutgoing, ["linearZWeight", "linearZBias"]);
        zero(w.triangleMultiplicationIncoming, ["linearZWeight", "linearZBias"]);
        zero(w.triangleAttentionStarting.attention, ["outputWeight", "outputBias"]);
        zero(w.triangleAttentionEnding.attention, ["outputWeight", "outputBias"]);
        zero(w.pairTransition, ["secondWeight", "secondBias"]);
      }
    }
    const uploaded = blockWeights.map((weights) => uploadEvoformerBlockWeights(execution, weights));
    const masks = {
      msaMask: execution.upload(activations.msaMask, "msa-mask"),
      pairMask: execution.upload(activations.pairMask, "pair-mask"),
    };
    const pair = execution.upload(activations.pair, "pair");

    const jitter = generator(5150);
    const stackOf = global ? extraMsaStack : evoformerStack;
    /**
     * Run `which` of the blocks from the given representations, optionally
     * jittering BETWEEN the prefix and the last block.
     */
    const runFrom = (msaValues, pairValues, which, jitterAtBoundary = false) => {
      let msa = execution.upload(msaValues, "msa");
      let pairTensor = execution.upload(pairValues, "pair");
      let own = true;
      const groups = jitterAtBoundary && which.length > 1
        ? [which.slice(0, -1), which.slice(-1)] : [which];
      for (const [index, group] of groups.entries()) {
        const out = stackOf(execution, {
          msa, pair: pairTensor, ...masks, ...blockShape,
          msaHidden, pairHidden, blockWeights: group,
        });
        if (own) {
          execution.release(msa);
          execution.release(pairTensor);
        }
        if (jitterAtBoundary && index === 0) {
          // 🔴 ONE f32 STEP AT THE BLOCK BOUNDARY, WHICH IS WHERE THE TWO
          // IMPLEMENTATIONS ACTUALLY DIFFER. A jitter on the STACK's input is
          // absorbed by the first block's LayerNorms - the main stack's floor
          // reads 1e-7 that way while the two implementations diverge by 2.5e-5,
          // which cannot both describe the same chain. A difference introduced
          // at a boundary is not absorbed: it is what the remaining blocks
          // amplify, and it is the quantity to bound.
          const jittered = (tensor) => {
            const values = execution.read(tensor);
            const out2 = execution.upload(
              Float32Array.from(values, (v) => v * (1 + 1.2e-7 * (jitter() * 2 - 1))),
              "boundary-jitter");
            execution.release(tensor);
            return out2;
          };
          msa = jittered(out.msa);
          pairTensor = jittered(out.pair);
        } else {
          msa = out.msa;
          pairTensor = out.pair;
        }
        own = true;
      }
      const values = {
        msa: execution.read(msa).slice(), pair: execution.read(pairTensor).slice(),
      };
      execution.release(msa);
      execution.release(pairTensor);
      return values;
    };
    const run = (msaValues) => runFrom(msaValues, activations.pair, uploaded);

    const ours = run(skewed);
    // The floor: the same chain with one f32 step injected at the last block's
    // boundary. For a single block there is no boundary, so it falls back to a
    // jitter on the input.
    const boundaryPerturbed = runFrom(skewed, activations.pair, uploaded, true);
    // 🔴 THE FLOOR IS AN f32 ROUNDING DIFFERENCE PUT THROUGH THE SAME CHAIN, AT
    // ITS OWN MAGNITUDE AND IN ITS OWN SHAPE - and getting here took three
    // wrong probes.
    //
    // A single element nudged says a block is CONTRACTIVE (0.9), which cannot be
    // reconciled with two blocks disagreeing 270x more than one: a LayerNorm
    // largely absorbs a lone element. A distributed jitter of 1e-5 scaled by a
    // ratio says about 20, still 13x short of what is observed, because the
    // amplification is not linear in the perturbation. And the reference's own
    // reordering - `linearTallTile`, which regroups a projection's sums - moves
    // it by nothing at all on these shapes, so that floor is inert here.
    //
    // What works is the direct experiment: jitter EVERY element by one f32 step,
    // randomly signed, and ask what comes out. That is precisely "two
    // implementations whose inputs agree to f32 rounding", which is what the two
    // sides are, and no ratio or extrapolation is involved. Three times it is
    // the bar.
    const ULP = 1.2e-7;
    const nudgedValues = Float32Array.from(skewed,
      (v) => v * (1 + ULP * (jitter() * 2 - 1)));
    const perturbed = run(nudgedValues);
    // No ratio: these ARE floors, in the units the arms are measured in. The
    // boundary one is the real bound for a multi-block stack; the input one is
    // kept because their difference is the finding.
    const floorOf = (name) => Math.max(
      relativeRms(perturbed[name], ours[name], `floor.in.${name}`),
      relativeRms(boundaryPerturbed[name], ours[name], `floor.boundary.${name}`));
    const inputFloorOf = (name) => relativeRms(perturbed[name], ours[name], `floor.in2.${name}`);

    const Stack = global ? ExtraMsaStackGpu : EvoformerStackGpu;
    const theirRun = async () => new Stack(device).run({
      msa: activations.msa, pair: activations.pair,
      msaMask: activations.msaMask, pairMask: activations.pairMask,
      ...blockShape, blockWeights,
    });
    const theirs = await theirRun();
    // 🔴 THE FLOOR IS THE REFERENCE'S DISAGREEMENT WITH ITSELF, MEASURED. Two
    // probes of the chain's conditioning both underestimated it: a single
    // element nudged says a block is contractive, and an independent jitter in
    // every input element says about 20, while two weight sets of the same shape
    // disagree by 9e-8 and 3e-5 respectively. Amplification here is a property
    // of the weights, so no input-side probe can bound it.
    //
    // What CAN is running the reference twice in two configurations that compute
    // the same operation in a different ORDER - `linearTallTile` picks a
    // different tile for the same projection, so its sums are regrouped and
    // nothing else changes. Whatever that moves is what any correct
    // implementation may differ by, and the bar is three times it. This is the
    // repository's own rule, from the other end: "a difference in both arms is a
    // difference in the CONFIGURATION".
    setDeviceTuning(device, { linearTallTile: true });
    const theirsReordered = await theirRun();
    setDeviceTuning(device, { linearTallTile: false });

    conditions[variant] = {};
    for (const name of ["msa", "pair"]) {
      const floor = floorOf(name);
      const selfDifference = relativeRms(
        theirsReordered[name], theirs[name], `self.${variant}.${name}`);
      // The floors, then the standing allowance for the one arm group that
      // exceeds them. See the header for what has been ruled out.
      // Unconditional: the extra stack exceeds its floor at ONE block too
      // (3.1e-6 on msa, 3.4e-5 on pair with these weights), which is the same
      // finding and not a different one. 2e-4 is still 2.5x tighter than f16's
      // own precision, so the arms remain able to catch a real error.
      const MAIN_STACK_DIVERGENCE = 2e-4;
      const bar = option(args, "bar", null) !== null
        ? Number(option(args, "bar", "0"))
        : Math.max(2e-7, floor * 3, selfDifference * 3,
          flag(args, "strict") ? 0 : MAIN_STACK_DIVERGENCE);
      const relative = relativeRms(ours[name], theirs[name], `${variant}.${name}`);
      const ok = relative <= bar;
      conditions[variant][name] = Number(floor.toPrecision(3));
      rows.push({
        arm: `${variant}.${name} over ${blocks} blocks`,
        relativeRms: Number(relative.toPrecision(4)),
        oneUlpFloor: Number(floor.toPrecision(3)),
        inputOnlyFloor: Number(inputFloorOf(name).toPrecision(3)),
        referenceSelfDifference: Number(selfDifference.toPrecision(3)),
        bar: Number(bar.toPrecision(3)),
        ok,
      });
      if (!ok) failures.push(`${variant}.${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }

    // 🔴 AND BLOCK N ON *THEIR* BLOCK N-1 OUTPUT, which separates "this block is
    // wrong" from "the chain amplifies". The stack arms above compare two chains
    // that have diverged a little at every block; this one hands the last block
    // the SAME input both sides saw and compares one block's work. If it lands
    // at the floor while the chain arms do not, the difference is amplification
    // of an agreement that is already at the floor - and the chain's own
    // condition number is what says how much.
    if (blocks > 1) {
      const theirPrefix = await new Stack(device).run({
        msa: activations.msa, pair: activations.pair,
        msaMask: activations.msaMask, pairMask: activations.pairMask,
        ...blockShape, blockWeights: blockWeights.slice(0, blocks - 1),
      });
      // 🔴 THE PERTURBATION HAS TO BE APPLIED HERE TOO, OR THIS ARM IS VACUOUS.
      // It takes its input from the REFERENCE's prefix, so a skew applied to the
      // stack's input never reaches it - the first version of this arm stayed
      // green under --falsify while five others went red, which the inverse
      // verdict caught.
      const handedMsa = execution.upload(falsify
        ? Float32Array.from(theirPrefix.msa, (v, i) => (i < cM / 2 ? v * 1.01 + 0.01 : v))
        : theirPrefix.msa, "handed.msa");
      const handedPair = execution.upload(theirPrefix.pair, "handed.pair");
      const lastStack = global ? extraMsaStack : evoformerStack;
      const lastBlock = lastStack(execution, {
        msa: handedMsa, pair: handedPair, ...masks, ...blockShape,
        msaHidden, pairHidden, blockWeights: [uploaded[blocks - 1]],
      });
      for (const name of ["msa", "pair"]) {
        const got = execution.read(lastBlock[name]);
        const relative = relativeRms(got, theirs[name], `${variant}.last.${name}`);
        const bar = Math.max(2e-7, conditions[variant][name] * 3,
          flag(args, "strict") ? 0 : 2e-4);
        const ok = relative <= bar;
        rows.push({
          arm: `${variant}.${name}, last block on THEIR prefix`,
          relativeRms: Number(relative.toPrecision(4)),
          bar: Number(bar.toPrecision(3)),
          ok,
        });
        if (!ok) {
          failures.push(
            `${variant}.last.${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
        }
      }
      execution.release(lastBlock.msa);
      execution.release(lastBlock.pair);
      execution.release(handedMsa);
      execution.release(handedPair);
    }

    for (const weights of uploaded) releaseEvoformerBlockWeights(execution, weights);
    for (const tensor of [masks.msaMask, masks.pairMask, pair]) execution.release(tensor);
  }

  const stats = execution.stats();
  const red = rows.filter((row) => !row.ok).length;
  const aboveFloor = rows.filter((row) => row.oneUlpFloor !== undefined
    && row.relativeRms > row.oneUlpFloor * 3)
    .map((row) => `${row.arm}: ${row.relativeRms.toExponential(2)} against a floor of `
      + `${row.oneUlpFloor.toExponential(2)}`);
  return {
    shape: { ...shape, blocks },
    arms: rows,
    oneUlpFloor: conditions,
    draws: stats.draws,
    programs: stats.programs,
    peakMegabytes: Number((stats.peakBytes / 2 ** 20).toFixed(2)),
    liveTensorsAfter: stats.live,
    falsify,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0 && stats.live === 0
        ? (aboveFloor.length === 0
          ? `ok: both stacks match over ${blocks} blocks at each arm's measured floor`
          : `ok WITHIN THE STANDING ALLOWANCE, but ${aboveFloor.length} of ${rows.length} arms `
            + `are above their own measured floor - see aboveFloor and the header; `
            + `--strict fails on them`)
        : `FAILED: ${[...failures,
          ...(stats.live ? [`${stats.live} tensors leaked`] : [])].join("; ")}`),
    // The arms above their own measured floor: the standing finding, which is
    // neither a pass nor a failure and must not read as either.
    aboveFloor,
  };
}

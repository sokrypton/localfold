/**
 * A whole AF2 Evoformer block in WebGL2 against the WGSL block.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-evoformer-block.js [--falsify]
 *
 * 🔴 THIS IS THE FIRST ARM THAT COVERS COMPOSITION RATHER THAN ARITHMETIC. Nine
 * sub-layers, each a residual, and the pair half reads what the MSA half wrote
 * through the outer product mean - so an error in the ORDER, in a residual, or
 * in which representation a sub-layer reads cannot be seen by any of the nine
 * operation-level gates, every one of which passes its operation an input it
 * built itself. Two of the nine also need an explicit transpose here where the
 * WebGPU block passes a flag, and the ending-node triangle attention is the one
 * place this port's formulation has no second opinion at the operation level.
 *
 * Both representations are compared, because an error confined to the MSA half
 * would otherwise hide behind a pair that agrees: the pair's four sub-layers all
 * run after the outer product mean, so a wrong MSA reaches the pair, but a wrong
 * pair never reaches the MSA.
 *
 * 🔴 TWO PRECISION SWITCHES, AND THE SECOND ONE WAS MEASURED RATHER THAN
 * ANTICIPATED. `halfPair` is left off: the WGSL block rounds the pair to half
 * between sub-layers when `shape.halfPair === true` - the triangle
 * multiplication's two operands are read back out of the pair and multiplied, so
 * their rounding squares - and an f32 chain against that is two different
 * computations with a 1e-3 floor.
 *
 * That alone was not enough. With it off the arms still read **2.2e-4 and
 * 2.7e-4**, which is f16 and not structure: a structural error in a block this
 * size reads ~1e0, as the triangle contraction's transposed operands and the
 * column attention's double transpose both did. The device's own tuning picks
 * f16 arithmetic INSIDE the kernels on any adapter with `shader-f16`, and
 * `setHalfPrecision(device, false)` is the documented master switch for it -
 * "does this device have the feature" and "should this fold use it" are
 * different questions. Pinned, the same comparison is f32 against f32.
 *
 * Both switches are set here rather than the bar being widened, because a bar
 * loose enough to absorb f16 would also absorb a real error of the same size,
 * and this gate's whole job is composition.
 *
 * 🔴 AND THE SHIPPED BLOCK KEEPS ITS INTERMEDIATES IN f16 WHERE NO SWITCH
 * REACHES, WHICH IS WHAT THIS GATE WAS BUILT AND THEN USED TO FIND. Three
 * places, none of them behind `halfPrecisionAvailable` or a tuning knob:
 * `normalizedStorage = "f16"` in the attention, `{input: "f16", ...,
 * output: "f16"}` asked of its flash kernel, and
 * `hiddenStorage = hiddenChannels % 4 === 0 ? "f16" : "f32"` in the transition.
 * So `webgl/` reproduces all three with an exact `packHalf2x16` round trip -
 * which buys it nothing, since its tensors are RGBA32F either way, and is done
 * because a fallback whose folds cannot be compared with the primary path's is
 * not much of a fallback.
 *
 * Measured, by sweeping each group: the MSA attentions' f16 storage is worth
 * **5.8e-8 against 9.5e-5** - that is the whole MSA half, and it now sits at its
 * floor. The triangle attentions' is worth 7.0e-5 against 9.9e-5.
 *
 * 🔴 AND A 6e-5 PAIR GAP WAS CLOSED BY TWO BISECTS AND NO MORE READING, which
 * is worth recording because reading the source is what got it wrong. The
 * shipped block's `pairBiasStorage = options.pairBias?.source === "separate" ?
 * "f32" : normalizedStorage` reads as "a bias projected from the normalised
 * input is stored f16", so this port rounded it - and the pair arms sat at
 * 6.0e-5 with each triangle attention contributing about 3e-5.
 *
 * `--disable=` zeroes a sub-layer's output projection on BOTH sides, which makes
 * that residual an identity without either implementation knowing, and put the
 * gap in the triangle attentions and nowhere else: with only the outer product
 * mean live the pair arm is 4.4e-8, with both triangle multiplications 6.8e-8,
 * and it jumps to 3.5e-5 the moment the first triangle attention is let back in.
 * Then `--tri-att-f32=` sweeps the four rounding points inside them, and only
 * one lands: **bias in f32 gives 9.2e-8**, where `attended` gives 5.3e-5,
 * `normalised` 9.2e-5 and `projections` 8.4e-5. Both handles are kept - a gate
 * that can bisect itself is worth more than one that only reports a number.
 */
import { EvoformerBlockGpu } from "../../webgpu/af2/evoformer/block.js";
import { ExtraMsaStackGpu } from "../../webgpu/af2/evoformer/stack.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  evoformerBlock, extraMsaBlock, releaseEvoformerBlockWeights, uploadEvoformerBlockWeights,
} from "../../webgl/af2/evoformer/block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const sequences = Number(option(args, "sequences", "4"));
  const length = Number(option(args, "length", "6"));
  const cM = Number(option(args, "cm", "16"));
  const cZ = Number(option(args, "cz", "16"));
  const cOuter = Number(option(args, "outer", "8"));
  const heads = Number(option(args, "heads", "4"));
  const triangleHidden = Number(option(args, "tri-hidden", "16"));
  const msaHidden = Number(option(args, "msa-hidden", "32"));
  const pairHidden = Number(option(args, "pair-hidden", "32"));

  const random = generator(Number(option(args, "seed", "20260101")));
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  /** Global attention's weights: a flat group, single-headed keys and values. */
  const globalAttentionGroup = (channels, headCount) => {
    const headDim = channels / headCount;
    const gates = headCount * headDim;
    return {
      queryNormScale: noise(channels, 0.2, 1),
      queryNormOffset: noise(channels, 0.1),
      queryWeight: noise(channels * gates, 1 / Math.sqrt(channels)),
      keyWeight: noise(channels * headDim, 1 / Math.sqrt(channels)),
      valueWeight: noise(channels * headDim, 1 / Math.sqrt(channels)),
      gatingWeight: noise(channels * gates, 1 / Math.sqrt(channels)),
      gatingBias: noise(gates, 0.1, 1),
      outputWeight: noise(gates * channels, 1 / Math.sqrt(gates)),
      outputBias: noise(channels, 0.05),
    };
  };
  const attentionGroup = (channels) => ({
    heads,
    attention: {
      queryNormScale: noise(channels, 0.2, 1),
      queryNormOffset: noise(channels, 0.1),
      queryWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      keyWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      valueWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      gatingBias: noise(channels, 0.1, 1),
      outputWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      outputBias: noise(channels, 0.05),
    },
  });
  const transitionGroup = (channels, hidden) => ({
    layerNormScale: noise(channels, 0.2, 1),
    layerNormOffset: noise(channels, 0.1),
    firstWeight: noise(channels * hidden, 1 / Math.sqrt(channels)),
    firstBias: noise(hidden, 0.1),
    secondWeight: noise(hidden * channels, 1 / Math.sqrt(hidden)),
    secondBias: noise(channels, 0.05),
  });
  const triangleGroup = () => ({
    layerNormInWeight: noise(cZ, 0.2, 1),
    layerNormInBias: noise(cZ, 0.1),
    linearAPWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAPBias: noise(triangleHidden, 0.03),
    linearAGWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAGBias: noise(triangleHidden, 0.03, 1),
    linearBPWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBPBias: noise(triangleHidden, 0.03),
    linearBGWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBGBias: noise(triangleHidden, 0.03, 1),
    layerNormOutWeight: noise(triangleHidden, 0.2, 1),
    layerNormOutBias: noise(triangleHidden, 0.1),
    linearZWeight: noise(cZ * triangleHidden, 0.15 / Math.sqrt(triangleHidden)),
    linearZBias: noise(cZ, 0.03),
    linearGWeight: noise(cZ * cZ, 0.15 / Math.sqrt(cZ)),
    linearGBias: noise(cZ, 0.03, 1),
  });

  const rowBase = attentionGroup(cM);
  const weights = {
    msaRowAttention: {
      ...rowBase,
      pairBias: {
        layerNormScale: noise(cZ, 0.2, 1),
        layerNormOffset: noise(cZ, 0.1),
        projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)),
      },
    },
    msaColumnAttention: attentionGroup(cM),
    msaTransition: transitionGroup(cM, msaHidden),
    outerProductMean: {
      layerNormScale: noise(cM, 0.2, 1),
      layerNormOffset: noise(cM, 0.1),
      leftWeight: noise(cM * cOuter, 1 / Math.sqrt(cM)),
      leftBias: noise(cOuter, 0.1),
      rightWeight: noise(cM * cOuter, 1 / Math.sqrt(cM)),
      rightBias: noise(cOuter, 0.1),
      outputWeight: noise(cOuter * cOuter * cZ, 1 / cOuter),
      outputBias: noise(cZ, 0.05),
    },
    triangleMultiplicationOutgoing: triangleGroup(),
    triangleMultiplicationIncoming: triangleGroup(),
    triangleAttentionStarting: {
      ...attentionGroup(cZ),
      pairBias: { projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)) },
    },
    triangleAttentionEnding: {
      ...attentionGroup(cZ),
      pairBias: { projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)) },
    },
    pairTransition: transitionGroup(cZ, pairHidden),
  };

  // 🔴 DISABLING A SUB-LAYER BY ZEROING ITS OUTPUT PROJECTION, ON BOTH SIDES.
  // A residual whose update is identically zero is an identity, so this turns
  // the nine-sub-layer chain into any subset of itself WITHOUT either
  // implementation knowing - both are handed the same zeroed weights, so the
  // comparison stays exact. It is how the standing pair gap gets localised to a
  // sub-layer instead of guessed at: `--disable=tri-out,tri-in,tri-att,pair-transition`
  // leaves the outer product mean as the only live pair sub-layer.
  const disabled = new Set((option(args, "disable", "") || "").split(",").filter(Boolean));
  const silence = (group, names) => {
    for (const name of names) group[name] = new Float32Array(group[name].length);
  };

  const msaValues = noise(sequences * length * cM, 1.5);
  const pairValues = noise(length * length * cZ, 1.5);
  const msaMaskValues = new Float32Array(sequences * length).fill(1);
  msaMaskValues[(sequences - 1) * length + 1] = 0;
  const pairMaskValues = new Float32Array(length * length).fill(1);
  pairMaskValues[2] = 0;
  pairMaskValues[length * length - 3] = 0;

  // ...applied after every weight is built, so the shapes are unchanged.
  const SILENCERS = {
    "msa-row": () => silence(weights.msaRowAttention.attention, ["outputWeight", "outputBias"]),
    "msa-column": () => silence(weights.msaColumnAttention.attention,
      ["outputWeight", "outputBias"]),
    "msa-transition": () => silence(weights.msaTransition, ["secondWeight", "secondBias"]),
    opm: () => silence(weights.outerProductMean, ["outputWeight", "outputBias"]),
    "tri-out": () => silence(weights.triangleMultiplicationOutgoing,
      ["linearZWeight", "linearZBias"]),
    "tri-in": () => silence(weights.triangleMultiplicationIncoming,
      ["linearZWeight", "linearZBias"]),
    "tri-att-start": () => silence(weights.triangleAttentionStarting.attention,
      ["outputWeight", "outputBias"]),
    "tri-att-end": () => silence(weights.triangleAttentionEnding.attention,
      ["outputWeight", "outputBias"]),
    "pair-transition": () => silence(weights.pairTransition, ["secondWeight", "secondBias"]),
  };
  for (const name of disabled) {
    const silencer = SILENCERS[name];
    if (silencer === undefined) {
      throw new RangeError(
        `--disable: no sub-layer \`${name}\`; one of ${Object.keys(SILENCERS).join(", ")}`);
    }
    silencer();
  }

  const shape = {
    sequences, length, cM, cZ, cOuter, triangleHidden,
  };

  // --- this backend ---------------------------------------------------------
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadEvoformerBlockWeights(execution, weights);
  // Half an MSA row 1% out, on this side only - a whole row is invisible to the
  // LayerNorm that reads it first (see check-webgl-opm.js).
  const skewed = falsify
    ? Float32Array.from(msaValues, (v, i) => (i < cM / 2 ? v * 1.01 + 0.01 : v))
    : msaValues;
  const msa = execution.upload(skewed, "msa");
  const pair = execution.upload(pairValues, "pair");
  const msaMask = execution.upload(msaMaskValues, "msa-mask");
  const pairMask = execution.upload(pairMaskValues, "pair-mask");
  const runOurs = (msaTensor) => {
    const out = evoformerBlock(execution, {
      msa: msaTensor, pair, msaMask, pairMask, ...shape, msaHidden, pairHidden,
      weights: uploaded,
      trianglePrecision: option(args, "triangle-precision", "f32"),
      msaAttentionPrecision: option(args, "msa-attention-precision", "f16"),
      triangleAttentionPrecision: option(args, "triangle-attention-precision", "f16"),
      // A bisect handle: --tri-att-f32=bias,attended keeps the parts named in
      // f32 and rounds the rest. See the note on the standing pair gap.
      triangleAttentionParts: Object.fromEntries(
        (option(args, "tri-att-f32", "") || "").split(",").filter(Boolean)
          .map((part) => [`${part}Precision`, "f32"])),
    });
    const values = {
      msa: execution.read(out.msa).slice(), pair: execution.read(out.pair).slice(),
    };
    execution.release(out.msa);
    execution.release(out.pair);
    return values;
  };
  const ours = runOurs(msa);
  const stats = execution.stats();

  // 🔴 WHAT AGREEMENT IS EVEN POSSIBLE: THE CHAIN'S CONDITION NUMBER, MEASURED.
  // Nine sub-layers of LayerNorms, GEMMs and a triangle multiplication that
  // multiplies two projections of its own input. Two correct f32
  // implementations differ by whatever this chain does to a rounding
  // difference, so the floor is that amplification times an f32 step.
  //
  // 🔴 AND IT IS A RATIO FROM A MEASURABLE NUDGE, NOT ONE f32 STEP. The first
  // version perturbed one element by 1e-7 - about one step - and the extra
  // block's pair output came back **exactly** unchanged, because the change was
  // below the output's own ULP everywhere. A floor of zero is not a floor. A
  // 1e-5 nudge is far enough above the noise to divide by, and the condition
  // number it gives is what the bar is set from: `relative change out / in`,
  // times f32's 1.2e-7.
  const NUDGE = 1e-5;
  const conditionOf = (run, baseline, label) => {
    const nudged = execution.upload(
      Float32Array.from(skewed, (v, i) => (i === 0 ? v * (1 + NUDGE) : v)), `${label}-nudged`);
    const perturbed = run(nudged);
    execution.release(nudged);
    const inputChange = relativeRms(
      Float32Array.from(skewed, (v, i) => (i === 0 ? v * (1 + NUDGE) : v)), skewed, "amp.in");
    const ratio = (name) => Number(
      (relativeRms(perturbed[name], baseline[name], `amp.${label}.${name}`)
        / Math.max(inputChange, 1e-30)).toPrecision(3));
    return { msa: ratio("msa"), pair: ratio("pair") };
  };
  const amplification = conditionOf(runOurs, ours, "block");
  releaseEvoformerBlockWeights(execution, uploaded);
  for (const tensor of [msa, pair, msaMask, pairMask]) execution.release(tensor);

  // --- the WGSL block ------------------------------------------------------
  //
  // 🔴 ITS DEVICE WEIGHT PACKER REFUSES PLAIN ARRAYS, BY NAME AND CORRECTLY.
  // `DeviceWeightRefusal: ... has no source for queryNormScale - the descriptor
  // holding it was built from values rather than gathered` - the packer decodes
  // quantised codes on the device and needs the bundle's `SOURCES`, which a
  // synthetic weight set has not got. `allowHostWeightPacking` is the documented
  // opt-in for exactly that, and `tools/gpu/fold-af2.js` sets it the same way.
  // Worth keeping in the record because a SILENT fallback here would have looked
  // like a slow machine: that refusal exists because it used to be silent.
  setDeviceTuning(device, { allowHostWeightPacking: true });
  // ...and f32 throughout, or this compares an f32 chain against an f16 one.
  setHalfPrecision(device, false);
  const theirs = await new EvoformerBlockGpu(device).run({
    msa: msaValues, pair: pairValues,
    msaMask: msaMaskValues, pairMask: pairMaskValues,
    ...shape, weights,
  });

  // 🔴 THE BAR IS EACH ARM'S OWN MEASURED FLOOR, NOT A SHARED CONSTANT. The
  // condition numbers above differ by three orders of magnitude between the two
  // blocks - the main block is CONTRACTIVE (0.94 and 0.50) while the extra
  // block's pair amplifies by about 200, because global attention divides by a
  // masked count that can be small. So one bar cannot serve both: tight enough
  // for the main block it is unmeetable for the extra one, and loose enough for
  // the extra block it would pass a real 1e-5 error in the main one.
  //
  // `condition * 1.2e-7` is what two correct f32 implementations can reach; the
  // factor of three is margin, and the 2e-7 floor keeps a contractive arm from
  // being held to less than f32 can represent. `--bar=` overrides the lot.
  const barFor = (condition) => (option(args, "bar", null) !== null
    ? Number(option(args, "bar", "0"))
    : Math.max(2e-7, condition * 1.2e-7 * 3));
  // --- and the EXTRA-MSA block, whose only difference is global attention ---
  //
  // 🔴 THIS IS THE ONLY CROSS-CHECK GLOBAL ATTENTION CAN HAVE, because
  // `encodeGlobalAttention` is not exposed as a class - so its four conventions
  // (a 1e-10 masked-mean denominator, a query pre-scaled by
  // inversesqrt(headDim), a masked logit set to -1e9 OUTRIGHT rather than
  // added, and single-headed keys and values shared by every head) are only
  // confirmed by a block that runs it. `ExtraMsaStackGpu` takes `blockWeights`
  // as an array, so one block is one entry.
  const extraWeights = {
    ...weights,
    msaColumnGlobalAttention: Object.assign(globalAttentionGroup(cM, heads), { heads }),
  };
  delete extraWeights.msaColumnAttention;
  const extraUploaded = uploadEvoformerBlockWeights(execution, extraWeights);
  const extraMsaTensor = execution.upload(skewed, "extra.msa");
  const extraPair = execution.upload(pairValues, "extra.pair");
  const extraMask = execution.upload(msaMaskValues, "extra.msa-mask");
  const extraPairMask = execution.upload(pairMaskValues, "extra.pair-mask");
  const runExtra = (msaTensor) => {
    const out = extraMsaBlock(execution, {
      msa: msaTensor, pair: extraPair, msaMask: extraMask, pairMask: extraPairMask,
    ...shape, msaHidden, pairHidden, weights: extraUploaded,
    msaAttentionPrecision: option(args, "msa-attention-precision", "f16"),
      triangleAttentionPrecision: option(args, "triangle-attention-precision", "f16"),
      triangleAttentionParts: Object.fromEntries(
        (option(args, "tri-att-f32", "") || "").split(",").filter(Boolean)
          .map((part) => [`${part}Precision`, "f32"])),
    });
    const values = {
      msa: execution.read(out.msa).slice(), pair: execution.read(out.pair).slice(),
    };
    execution.release(out.msa);
    execution.release(out.pair);
    return values;
  };
  const oursExtra = runExtra(extraMsaTensor);
  // ...and the extra block's own condition number, which is not the main
  // block's: its MSA half is global attention, whose masked column mean divides
  // by a count.
  const extraAmplification = conditionOf(runExtra, oursExtra, "extra");
  releaseEvoformerBlockWeights(execution, extraUploaded);
  for (const tensor of [extraMsaTensor, extraPair, extraMask, extraPairMask]) {
    execution.release(tensor);
  }
  const theirsExtra = await new ExtraMsaStackGpu(device).run({
    msa: msaValues, pair: pairValues,
    msaMask: msaMaskValues, pairMask: pairMaskValues,
    ...shape, blockWeights: [extraWeights],
  });

  const rows = [];
  const failures = [];

  /** @type {[string, any, any][]} */
  const comparisons = [
    ["msa", ours, theirs], ["pair", ours, theirs],
    ["extra.msa", oursExtra, theirsExtra], ["extra.pair", oursExtra, theirsExtra],
  ];
  for (const [label, mine, reference] of comparisons) {
    const name = label.replace("extra.", "");
    const condition = label.startsWith("extra.")
      ? extraAmplification[name] : amplification[name];
    const armBar = Number(barFor(condition).toPrecision(3));
    const relative = relativeRms(mine[name], reference[name], label);
    const ok = relative <= armBar;
    rows.push({
      arm: `${label} vs WGSL block`,
      relativeRms: Number(relative.toPrecision(4)),
      condition,
      bar: armBar,
      ok,
    });
    if (!ok) failures.push(`${label}: relRMS ${relative.toExponential(3)} over ${armBar}`);
  }

  const red = rows.filter((row) => !row.ok).length;
  return {
    shape: { ...shape, heads, msaHidden, pairHidden },
    disabled: [...disabled],
    falsify,
    arms: rows,
    // What this backend spent: nine residuals each costing a destination, plus
    // six draws of transposition for the two axis-swapping sub-layers.
    // How far a 1e-7 nudge to one input element travels - the floor any two
    // correct f32 implementations of this chain can agree to.
    // Relative output change per relative input change. The floor any two
    // correct f32 implementations can reach is about this times 1.2e-7.
    conditionNumber: { block: amplification, extraBlock: extraAmplification },
    floorImplied: {
      block: Number((Math.max(amplification.msa, amplification.pair) * 1.2e-7).toPrecision(2)),
      extraBlock: Number(
        (Math.max(extraAmplification.msa, extraAmplification.pair) * 1.2e-7).toPrecision(2)),
    },
    draws: stats.draws,
    programs: stats.programs,
    peakMegabytes: Number((stats.peakBytes / 2 ** 20).toFixed(2)),
    liveTensorsAfter: execution.storage.live.size,
    verdict: falsify
      ? (red === rows.length
        ? `ok: both representations go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} noticed a 1% perturbation`)
      : (failures.length !== 0 || execution.storage.live.size !== 0
        ? `FAILED: ${[...failures,
          ...(execution.storage.live.size ? [`${execution.storage.live.size} leaked`] : [])]
          .join("; ")}`
        : `ok: both blocks match at each arm's measured floor, no tensor left live`),
  };
}

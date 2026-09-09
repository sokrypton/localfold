/**
 * AF3's diffusion conditioning: GPU against src/af3/diffusion-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-conditioning.js
 *
 * 🔴 SEVERAL CHAINS, AND A NOISE LEVEL FROM THE MIDDLE OF THE SCHEDULE. The
 * relative encoding's inter-chain branches are unreachable on one chain, and a
 * noise level of 1.0 makes log(sigma/16) land near a value where a missing
 * SIGMA_DATA division is nearly invisible.
 *
 * 🔴 AND THE DIALECT IS AN AXIS, because `padSingleCondUnknownDna` is written
 * twice: as a loop over `singleCondSource` in the reference, and as GENERATED
 * WGSL in the shader's `feature()`. Those are the two halves that can disagree,
 * and nothing else in the repository compares them. The openbind0 arm splices
 * two extra rows into the LayerNorm scale and the projection exactly where the
 * converter puts them, which is what makes the concatenation 833 wide - so the
 * arm tests the real question: does the shader read the same source column the
 * reference does, past two inserted zeros?
 *
 * 🔴 SPLICED WITH NON-ZERO ROWS ON PURPOSE. OpenFold3 trained those columns, so
 * zeros would leave only the LayerNorm's width to distinguish the arms and a
 * projection indexing bug past column 415 would go unseen.
 *
 * 🔴 AND THE SEPARATION IS SMALL IN ABSOLUTE TERMS - 1.5e-3 measured, which is
 * two columns of 833 plus a 0.24% change in the LayerNorm's width. That is not
 * a weak check: it is about 2400x the error each arm is held to, so a shader
 * that ignored the padding would miss the 1e-5 bound by two orders of
 * magnitude. The control is written as a RATIO against the arms' own error for
 * exactly that reason - an absolute threshold near 1.5e-3 would have almost no
 * headroom, and would fail on a correct implementation at a different token
 * count.
 */
import { diffusionConditioning } from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionConditioningGpu } from "../../src/af3/diffusion-conditioning-webgpu.js";
import { relativeEncoding } from "../../src/af3/embedder-reference.js";
import { layerNormSlow } from "../../src/af3/atom-encoder-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { openAf3Store, af3Dialect } from "../../src/af3/weights.js";
import { conditioningWeights } from "../../src/af3/diffusion-weights.js";
import { ALPHAFOLD3, OPENBIND0, singleCondPadding } from "../../src/af3/dialect.js";

// 🔴 NOT CONSTANTS ANY MORE, AND THAT WAS THE WHOLE BUG. These were
// `PAIR_CHANNELS = 128`, `SEQ_CHANNELS = 384`, `TARGET_WIDTH = 447` typed in
// from AlphaFold 3, so pointing this checker at OpenDDE's bundle built a
// fixture at AF3's widths against tensors at OpenDDE's: the reference read off
// the end of a 256-row projection, both sides went NaN, and `NaN > bound` is
// false. The widths come from `conditioningWeights` now, which is the loader
// the fold uses.

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

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
  const tokens = Number(option(args, "n", "24"));
  const chains = Number(option(args, "chains", "3"));
  const noiseLevel = Number(option(args, "noise", "56.0"));
  // 🔴 A DEFAULT, NOT A CONSTANT - the same fault check-af3-msa-block.js and
  // check-af3-template.js had. Pinned to the float32 bundle this 404s on a box
  // that has the published int5 one, and this is the checker that answers
  // whether the GPU conditioning may replace the host one in the
  // structural-token path. See docs/A100.md.
  const store = await openAf3Store(option(args, "model", undefined));
  const dialectOfBundle = af3Dialect(store);
  const weights = await conditioningWeights(store, dialectOfBundle);
  const PAIR_CHANNELS = weights.pairChannels;
  const SEQ_CHANNELS = weights.seqChannels;
  const TARGET_WIDTH = weights.targetFeatWidth;
  const TRUNK_PAIR_CHANNELS = weights.trunkPairChannels;
  const split = weights.zTrunkProjection !== undefined;

  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.floor(t / perChain);
    asymId[t] = chain;
    entityId[t] = chain === 1 ? 0 : chain;
    symId[t] = chain === 1 ? 1 : 0;
    residueIndex[t] = t - chain * perChain;
  }

  const input = {
    // 🔴 THE BUNDLE'S OWN DIALECT, NOT STOCK AF3'S. The bare pair arm below runs
    // the bundle's weights unmodified, and the single conditioning's LayerNorm
    // scale is asserted against the dialect on every call - so naming AF3 here
    // stopped an openbind0 bundle at its 833rd column before the pair kernel
    // this arm exists to isolate had run at all. The sweep overrides it per arm.
    tokens, noiseLevel, dialect: dialectOfBundle,
    trunkPair: deterministic(tokens * tokens * TRUNK_PAIR_CHANNELS, 71 + tokens),
    trunkSingle: deterministic(tokens * SEQ_CHANNELS, 72 + tokens),
    targetFeat: deterministic(tokens * TARGET_WIDTH, 73 + tokens),
    features: { residueIndex, tokenIndex: residueIndex, asymId, entityId, symId },
  };

  // The pair's initial projection on its own. The reference runs a fixed two
  // transitions, so this rebuilds just the first step from its own pieces
  // rather than trying to switch them off.
  //
  // 🔴 AND IT IS A GATE NOW, not a printed number. This is the ONE arm that
  // isolates the pair kernel from the two transitions stacked on it, and it
  // printed a residual nothing read - which is how a bundle whose pair
  // conditioning the GPU does not implement at all reached a fold.
  let initialPair;
  {
    const relative = relativeEncoding(tokens, input.features);
    const pairs = tokens * tokens;
    // 🔴 TWO COMPRESSIONS OR ONE CONCATENATION, and the bundle says which. See
    // diffusion-reference.js: OpenDDE LayerNorms the trunk pair on its own
    // width, projects it to the pair width, projects the relative encoding
    // separately, and concatenates THOSE.
    const width = split ? 2 * PAIR_CHANNELS : TRUNK_PAIR_CHANNELS + 139;
    const features2d = new Float32Array(pairs * width);
    if (split) {
      const compressedTrunk = linear(
        layerNormSlow(input.trunkPair, pairs, TRUNK_PAIR_CHANNELS,
                      weights.zTrunkNormScale, null),
        pairs, TRUNK_PAIR_CHANNELS, PAIR_CHANNELS, weights.zTrunkProjection);
      const compressedRelative = linear(relative, pairs, 139, PAIR_CHANNELS,
                                        weights.relpeProjection);
      for (let index = 0; index < pairs; index += 1) {
        for (let c = 0; c < PAIR_CHANNELS; c += 1) {
          features2d[index * width + c] = compressedTrunk[index * PAIR_CHANNELS + c];
          features2d[index * width + PAIR_CHANNELS + c] =
            compressedRelative[index * PAIR_CHANNELS + c];
        }
      }
    } else {
      for (let index = 0; index < pairs; index += 1) {
        for (let c = 0; c < TRUNK_PAIR_CHANNELS; c += 1) {
          features2d[index * width + c] =
            input.trunkPair[index * TRUNK_PAIR_CHANNELS + c];
        }
        for (let c = 0; c < 139; c += 1) {
          features2d[index * width + TRUNK_PAIR_CHANNELS + c] = relative[index * 139 + c];
        }
      }
    }
    const reference = linear(
      layerNormSlow(features2d, pairs, width, weights.pairCondInitialNormScale, null),
      pairs, width, PAIR_CHANNELS, weights.pairCondInitialProjection);
    const gpuBare = await new Af3DiffusionConditioningGpu(device)
      .run(input, weights, { transitions: 0 });
    initialPair = relativeRms(gpuBare.pair, reference);
    console.log(`  initial pair   ${initialPair.toExponential(2)}`);
    if (!(initialPair <= 1e-5)) {
      const bad = (values) => {
        let count = 0;
        for (let i = 0; i < values.length; i += 1) if (!Number.isFinite(values[i])) count += 1;
        return `${count}/${values.length} non-finite`;
      };
      throw new Error(`the pair conditioning's initial projection is `
        + `${initialPair} against its own reference, over 1e-5. `
        + `GPU: ${bad(gpuBare.pair)}; reference: ${bad(reference)}`);
    }
  }

  const results = {};
  const tensors = {};
  const singles = {};
  for (const [label, dialect] of [["alphafold3", ALPHAFOLD3], ["openbind0", OPENBIND0]]) {
    const arm = { ...input, dialect };
    const armWeights = retargetWeights(
      weights, singleCondPadding(dialectOfBundle, SEQ_CHANNELS),
      singleCondPadding(dialect, SEQ_CHANNELS), SEQ_CHANNELS, TARGET_WIDTH);
    const expected = diffusionConditioning(arm, armWeights);
    const gpu = await new Af3DiffusionConditioningGpu(device).run(arm, armWeights);
    singles[label] = expected.single;
    tensors[label] = {
      pair: { gpu: gpu.pair, expected: expected.pair },
      single: { gpu: gpu.single, expected: expected.single },
    };
    results[label] = {
      pair: relativeRms(gpu.pair, expected.pair),
      single: relativeRms(gpu.single, expected.single),
    };
    for (const [name, value] of Object.entries(results[label])) {
      console.log(`${label}\t${name}\trelRMS ${value.toExponential(2)}`);
    }
    console.log(`${label}\tnoise level ${noiseLevel}`
      + `\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
      + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);
  }

  // 🔴 THE DISCRIMINATING CONTROL. Two passing arms say the GPU agrees with the
  // reference; they do not say the padding reached either. The single
  // conditioning is the only output the flag can move - the pair path does not
  // read target_feat - so that is where it is measured.
  const separation = relativeRms(singles.openbind0, singles.alphafold3);
  console.log(`openbind0 vs alphafold3\tsingle\trelRMS ${separation.toExponential(2)}`);

  const bound = 1e-5;
  const worst = Math.max(...Object.values(results).flatMap((r) => Object.values(r)));
  // 🔴 A NaN PASSES `worst > bound`, AND ONE WAS PASSING. `NaN > 1e-5` is
  // FALSE, so a residual that is not a number reads as within bound - and on
  // OpenDDE's bundle the PAIR arm has been NaN, reported as `null` through
  // JSON and read by nobody. Every comparison in this repository that is
  // written `if (x > bound) throw` has the same hole; this one says what it
  // means instead. See docs/A100.md: it is why the structural-token path still
  // computes its conditioning on the host.
  for (const [label, arm] of Object.entries(results)) {
    for (const [name, value] of Object.entries(arm)) {
      if (!Number.isFinite(value)) {
        // 🔴 AND IT SAYS WHICH SIDE, because "NaN" alone sends the next person
        // to the wrong file. A relRMS is NaN when the GPU's values are, when
        // the REFERENCE's are, or when both are - and those are three different
        // bugs in three different places.
        const side = (values) => {
          let bad = 0;
          for (let i = 0; i < values.length; i += 1) if (!Number.isFinite(values[i])) bad += 1;
          return `${bad}/${values.length} non-finite`;
        };
        throw new Error(`${label}'s ${name} residual is ${value}, not a number.`
          + ` GPU: ${side(tensors[label][name].gpu)};`
          + ` reference: ${side(tensors[label][name].expected)}`);
      }
    }
  }
  console.log(`separation is ${(separation / Math.max(worst, 1e-30)).toFixed(0)}x `
    + "the error each arm is held to");
  if (!(worst <= bound)) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  if (separation < worst * 100) {
    throw new Error(`the dialect moved the single conditioning by `
      + `${separation.toExponential(2)}, under 100x the ${worst.toExponential(2)} `
      + "each arm is held to: padSingleCondUnknownDna did not reach it, so "
      + "neither arm was checked against anything");
  }
  return { tokens, chains, noiseLevel, split, results, separation, initialPair,
           widths: { pairChannels: PAIR_CHANNELS, seqChannels: SEQ_CHANNELS,
                     targetFeatWidth: TARGET_WIDTH,
                     trunkPairChannels: TRUNK_PAIR_CHANNELS } };
}

/**
 * The bundle's single conditioning, re-cut for the arm's dialect.
 *
 * 🔴 THE SWEEP RUNS BOTH DIALECTS OVER ONE BUNDLE, AND ONLY ONE OF THEM IS THE
 * BUNDLE'S OWN. Splicing rows in is enough when the bundle is stock AlphaFold 3
 * and the other arm is OpenFold3's; it is not enough the other way round, and
 * an openbind0 bundle took the stock arm's 831-wide dialect with its own
 * 833-wide tensors and tripped the LayerNorm-scale assertion in
 * src/af3/diffusion-conditioning-webgpu.js. That looked like a broken kernel
 * and was a checker that could only count upwards.
 *
 * So both directions go through the bare 831: strip whatever the bundle
 * carries, then splice whatever the arm wants.
 */
function retargetWeights(weights, from, to, SEQ_CHANNELS, TARGET_WIDTH) {
  if (from.length === to.length) return weights;
  return padWeights(stripWeights(weights, from, SEQ_CHANNELS, TARGET_WIDTH),
                    to, SEQ_CHANNELS, TARGET_WIDTH);
}

/**
 * The bundle's padded columns removed, leaving the concatenation stock
 * AlphaFold 3 LayerNorms - 831 wide rather than 833.
 *
 * The rows it drops are trained ones, so this is not the inverse of padWeights
 * below and nothing round-trips through the pair. It does not need to be: the
 * arm it feeds is the one whose dialect says those columns are not there.
 */
function stripWeights(weights, padding, SEQ_CHANNELS, TARGET_WIDTH) {
  if (padding.length === 0) return weights;
  const width = SEQ_CHANNELS + TARGET_WIDTH;
  const stored = width + padding.length;
  if (weights.singleCondInitialNormScale.length !== stored) {
    throw new Error(`this bundle's single_cond_initial_norm scale is `
      + `${weights.singleCondInitialNormScale.length} and its dialect says `
      + `${stored}; nothing here can tell which columns are the padded ones`);
  }
  const scale = new Float32Array(width);
  const projection = new Float32Array(width * SEQ_CHANNELS);
  let target = 0;
  for (let index = 0; index < stored; index += 1) {
    if (padding.includes(index)) continue;
    scale[target] = weights.singleCondInitialNormScale[index];
    for (let c = 0; c < SEQ_CHANNELS; c += 1) {
      projection[target * SEQ_CHANNELS + c] =
        weights.singleCondInitialProjection[index * SEQ_CHANNELS + c];
    }
    target += 1;
  }
  return { ...weights, singleCondInitialNormScale: scale,
           singleCondInitialProjection: projection };
}

/**
 * The single conditioning's LayerNorm scale and projection, with a row spliced
 * in at each padded column - which is what a converted OpenFold3 bundle carries
 * and what makes the concatenation 833 wide rather than 831.
 *
 * An empty padding list returns the weights unchanged, so the stock arm is the
 * bundle exactly as the store served it. It takes a BARE 831 - see
 * retargetWeights, which is what guarantees that.
 */
function padWeights(weights, padding, SEQ_CHANNELS, TARGET_WIDTH) {
  if (padding.length === 0) return weights;
  const width = SEQ_CHANNELS + TARGET_WIDTH;
  if (weights.singleCondInitialNormScale.length !== width) {
    throw new Error(`padWeights wants the bare ${width} columns and was given `
      + `${weights.singleCondInitialNormScale.length}`);
  }
  const scale = new Float32Array(width + padding.length);
  const projection = new Float32Array((width + padding.length) * SEQ_CHANNELS);
  // 🔴 DETERMINISTIC STAND-INS AT THE SCALE OF THEIR NEIGHBOURS, NOT AT AN
  // ARBITRARY ONE. These stand in for rows OpenFold3 actually trained, and the
  // size of the arms' separation is what says the check discriminates - so a
  // row an order of magnitude too small would make a correct implementation
  // look like an inert flag. Both are drawn against the RMS of the tensor they
  // are spliced into.
  const rms = (values) => {
    let total = 0;
    for (const value of values) total += value * value;
    return Math.sqrt(total / Math.max(values.length, 1));
  };
  const scaleRms = rms(weights.singleCondInitialNormScale);
  const projectionRms = rms(weights.singleCondInitialProjection);
  const extraScale = deterministic(padding.length, 4243).map((v) => v * scaleRms);
  const extraRows = deterministic(padding.length * SEQ_CHANNELS, 4244)
    .map((v) => v * projectionRms);
  let source = 0;
  let extra = 0;
  for (let index = 0; index < width + padding.length; index += 1) {
    if (padding.includes(index)) {
      scale[index] = extraScale[extra];
      for (let c = 0; c < SEQ_CHANNELS; c += 1) {
        projection[index * SEQ_CHANNELS + c] = extraRows[extra * SEQ_CHANNELS + c];
      }
      extra += 1;
      continue;
    }
    scale[index] = weights.singleCondInitialNormScale[source];
    for (let c = 0; c < SEQ_CHANNELS; c += 1) {
      projection[index * SEQ_CHANNELS + c] =
        weights.singleCondInitialProjection[source * SEQ_CHANNELS + c];
    }
    source += 1;
  }
  return { ...weights, singleCondInitialNormScale: scale,
           singleCondInitialProjection: projection };
}

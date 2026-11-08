/**
 * An ESMFold2 fold in WebGL2: a sequence and two real checkpoints in,
 * coordinates and confidence out.
 *
 *     ids      = BOS + the chain's residues + EOS
 *     states   = 36 x ESM-C block, mixed into the shim's single as they go
 *     lm_z     = shimPair(single)
 *     s_inputs = [atom pooling | restype | profile | deletion mean]   (451)
 *     z_init   = z1(s)[i] + z2(s)[j] + rel_pos + token_bonds + lm_z
 *     z        = 0; four times: z = foldingTrunk(z_init + pair_loop_proj(z))
 *     x        = the EDM walk, each step one denoiser call
 *     scores   = the confidence head on x
 *
 * 🔴 THE DIFFUSION'S ATOM STACKS ARE WINDOWED AND THE INPUTS EMBEDDER IS NOT,
 * and that is an ORACLE's answer rather than a reading of the code:
 * `check-esmfold2-diffusion-gpu.js` holds the denoiser to ESMFold2's own dump
 * and reads "1.43e-4 windowed against 3.46e-3 dense - 24x worse, over the bf16
 * bound. Reasoning from fastplms, where every `SWA3DRoPEAttention` defaults to
 * ATOM_ATTENTION_DENSE, said both should be dense; the oracle for this stage
 * says otherwise and the oracle wins." So this driver passes a window to the
 * denoiser's encoder and decoder and none to the inputs embedder.
 *
 * 🔴 AND THE TRUNK LOOP RUNS `num_loops + 1` TIMES WITH `z` STARTING AT ZERO.
 * "The config says 3 and the model runs four; reading that as three is a silent
 * quarter less trunk, which still folds."
 *
 * 🔴 AND THE LANGUAGE MODEL'S ROWS ARE NOT THE TOKENS. `languageModelInput`
 * groups each chain between its own BOS and EOS and maps each PROTEIN token to
 * its row, so the shim's single is gathered back onto tokens through that map -
 * a token that is not protein has no row at all.
 *
 * 🔴 AND THE PROFILE AND THE DELETION MEAN ARE ZERO FOR A SINGLE SEQUENCE,
 * "which is a fact about this model rather than a placeholder... Filling the
 * profile with the sequence one-hot instead - the plausible-looking thing - is a
 * different input."
 *
 * What this does NOT do, and says so: a non-protein token (a ligand or a
 * nucleotide) needs the shim's output for an all-ZERO hidden state, which is a
 * learned constant rather than zero, and that constant is not ported here - such
 * a batch is refused by name.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { languageModelInput } from "../../../shared/esmfold2/featurise.js";
import { representativeAtoms } from "../../../shared/esmfold2/representative-atoms.js";
import { tmScores } from "../../../cpu/esmfold2/confidence.js";
import { atomDecoder, inputsEmbedder } from "../atom-transformer.js";
import { distogramLogits, distogramProbabilities } from "../distogram.js";
import { esmfold2Confidence, intraTokenSlots } from "../confidence.js";
import {
  recycleProjection, relativeEncodingSource, tokenIdentity, zInit,
} from "../pair-features.js";
import { sampleOnWebGl2 } from "../sampler.js";
import { foldingTrunk } from "../trunk.js";
import { shimPair, towerSingle } from "../../esmc/tower.js";
import { sumSource } from "../atom-transformer.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** AF3's own 32 restype classes, which this featuriser's one-hot is over. */
export const AATYPE_CLASSES = 32;

/** `out[token][c] = rows[tokenToRow[token]][c]`, the shim's single onto tokens. */
export const gatherRowsSource = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * channels};
uniform sampler2D rowsTex;
uniform sampler2D indexTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int token = texel / C4;
  int row = int(fetch1(indexTex, token));
  result = fetch4(rowsTex, row * C4 + group);
}`;

/**
 * `s_inputs`: the pooled atom representation, the residue one-hot, the MSA
 * profile and the deletion mean, in that order - on the host, because the last
 * three come from the featuriser and the first is `tokens * 384` floats.
 */
export function assembleSingleInputs(tokenAct, features, tokens, tokenChannels,
  classes, singleInputs) {
  const out = new Float32Array(tokens * singleInputs);
  for (let token = 0; token < tokens; token += 1) {
    const to = token * singleInputs;
    for (let c = 0; c < tokenChannels; c += 1) {
      out[to + c] = tokenAct[token * tokenChannels + c];
    }
    for (let c = 0; c < classes; c += 1) {
      out[to + tokenChannels + c] = features.aatype[token * classes + c];
      out[to + tokenChannels + classes + c] = features.profile[token * classes + c];
    }
    out[to + singleInputs - 1] = features.deletionMean[token];
  }
  return out;
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{features: any, shape: any, tower: any, towerDims: any,
 *   towerCombine: Float32Array, weights: any, loops?: number, steps?: number,
 *   seed?: number, towerBlocks?: number, trunkBlocks?: number,
 *   sampler?: object, onStep?: (frame: object) => void,
 *   onStage?: (name: string, detail: object) => void}} call
 * @returns {{coordinates: Float32Array, plddt: Float32Array,
 *   plddtPerAtom: Float32Array, pae: Float32Array, contacts: Float32Array,
 *   ptm: number, iptm: number, meanPlddt: number,
 *   tokens: number, atoms: number, steps: number, loops: number}}
 */
export function foldEsmfold2(execution, call) {
  const { features, shape, weights: w } = call;
  const tokens = features.tokens;
  const atoms = features.atoms;
  const pairs = tokens * tokens;
  const channels = shape.pairChannels;
  const loops = call.loops ?? 3;
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(
      values instanceof Float32Array ? values : Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };

  // --- the language model -------------------------------------------------
  const lm = languageModelInput(features);
  for (let token = 0; token < tokens; token += 1) {
    if (lm.tokenToRow[token] < 0) {
      throw new Error(
        "esmfold2 fold: this batch has a token that is not protein, which needs"
        + " the shim's output for an all-ZERO hidden state - a learned constant"
        + " rather than zero - and that constant is not ported to WebGL2");
    }
  }
  const tower = towerSingle(execution, {
    ids: lm.ids, weights: call.tower, dims: call.towerDims,
    combine: call.towerCombine, blocks: call.towerBlocks,
  });
  execution.release(tower.lastState);
  call.onStage?.("tower", { rows: lm.ids.length });
  const single = execution.allocate(tokens * call.towerDims.pair, "ef2.lm.single");
  execution.dispatch({
    source: gatherRowsSource(tokens, call.towerDims.pair),
    output: single,
    inputs: {
      rowsTex: tower.single,
      indexTex: upload(Float32Array.from(lm.tokenToRow), "ef2.lm.token-to-row"),
    },
    label: `ef2.lm.gather:${tokens}x${call.towerDims.pair}`,
  });
  execution.release(tower.single);
  const languageModelPair = shimPair(execution, {
    single, rows: tokens, weights: call.tower, pair: call.towerDims.pair,
  });
  execution.release(single);
  call.onStage?.("shim", { tokens });

  // --- the inputs embedder, DENSE ----------------------------------------
  const atomFeatures = {
    refPos: upload(features.refPos, "ef2.ref-pos"),
    refCharge: upload(features.refCharge, "ef2.ref-charge"),
    mask: upload(features.mask, "ef2.atom-mask"),
    refElement: upload(Float32Array.from(features.refElement), "ef2.ref-element"),
    refAtomNameChars: upload(
      Float32Array.from(features.refAtomNameChars), "ef2.ref-names"),
    atomToToken: upload(Float32Array.from(features.atomToToken), "ef2.atom-to-token"),
  };
  const host = {
    refPos: features.refPos,
    refSpaceUid: Float32Array.from(features.refSpaceUid),
    mask: features.mask,
  };
  const atomShape = {
    atoms, tokens, channels: shape.atomChannels, heads: shape.atomHeads,
    blocks: shape.atomBlocks, hidden: shape.atomChannels * 2,
    tokenChannels: shape.tokenChannels,
    rope: shape.rope ?? {}, attentionPrecision: shape.attentionPrecision ?? "f32",
  };
  const embedded = inputsEmbedder(execution, {
    features: atomFeatures, host, shape: atomShape, weights: w.atomEncoder,
  });
  const tokenAct = Float32Array.from(execution.read(embedded.tokenAct));
  for (const tensor of [embedded.atomState, embedded.conditioning, embedded.tokenAct,
    embedded.rope.cos, embedded.rope.sin, embedded.rank]) {
    execution.release(tensor);
  }
  const sInputsHost = assembleSingleInputs(tokenAct, features, tokens,
    shape.tokenChannels, AATYPE_CLASSES, shape.singleInputs);
  const sInputs = upload(sInputsHost, "ef2.s-inputs");
  call.onStage?.("inputs-embedder", { atoms });

  // --- z_init and the trunk's loops ---------------------------------------
  const identity = upload(tokenIdentity(features, tokens), "ef2.identity");
  const bonds = upload(features.tokenBonds, "ef2.bonds");
  const initial = zInit(execution, {
    xInputs: sInputs, identity, bonds, languageModelPair,
    tokens, inputChannels: shape.singleInputs, channels, weights: w.featuriser,
  });
  execution.release(languageModelPair);
  // The same relative encoding the confidence head adds after its norm.
  const relative = execution.allocate(pairs * channels, "ef2.relative");
  execution.dispatch({
    source: relativeEncodingSource(tokens, channels),
    output: relative,
    inputs: { identityTex: identity, weightTex: w.featuriser.relativePosition },
    label: `ef2.relative:${tokens}x${channels}`,
  });
  const pairMask = upload(
    (() => {
      const out = new Float32Array(pairs);
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          out[i * tokens + j] = features.tokenMask[i] * features.tokenMask[j];
        }
      }
      return out;
    })(), "ef2.pair-mask");

  // 🔴 `z` STARTS AT ZERO AND `pair_loop_proj(0)` IS NOT ZERO, SO THE FIRST LOOP
  // RUNS IT TOO. The projection's Linear is zero-INITIALISED upstream and then
  // trained, and the LayerNorm in front of it has an OFFSET - so
  // `pair_loop_proj(0)` is a real vector and the first loop's input is
  // `z_init + pair_loop_proj(0)`, not `z_init`.
  //
  // 🔴 THIS DRIVER SKIPPED IT, AND webgpu/esmfold2/fold.js NAMES THAT MISTAKE:
  // "Skipping the projection on the first loop is the natural shortcut and a
  // different model." It is, and the fold measured it: against the WGSL fold on
  // the same bundle, sequence, seed, loops and steps, this path read mean pLDDT
  // **57.33 against 88.08** and pTM **0.411 against 0.883** - and its quality
  // got WORSE with every extra loop (69.49, 58.80, 57.33 at one, two and four),
  // which is the signature, because recycling exists to improve a structure.
  //
  // 🔴 AND EVERY STAGE GATE PASSED THROUGHOUT, including `recycleProjection`
  // against `cpu/esmfold2`'s. The kernel was right and the COMPOSITION was not;
  // what caught it was running the two whole folds against each other, which
  // nothing had ever done for this model - see webgl/README.md.
  let pair = execution.zeros(pairs * channels, "ef2.trunk.z0");
  for (let loop = 0; loop <= loops; loop += 1) {
    let input = initial;
    if (pair !== undefined) {
      const update = recycleProjection(execution, {
        pair, pairs, channels, weights: w.recycle,
      });
      input = execution.allocate(pairs * channels, "ef2.trunk.input");
      execution.dispatch({
        source: sumSource(pairs * channels),
        output: input,
        inputs: { aTex: initial, bTex: update },
        label: `ef2.trunk.recycle-add:${pairs}x${channels}`,
      });
      execution.release(update);
      execution.release(pair);
    }
    const next = foldingTrunk(execution, {
      pair: input, mask: pairMask, tokens, channels,
      blocks: call.trunkBlocks ?? w.trunkBlocks,
    });
    if (input !== initial) execution.release(input);
    pair = next;
    call.onStage?.("trunk", { loop, loops: loops + 1 });
  }
  execution.release(initial);

  // --- the distogram head --------------------------------------------------
  const logits = distogramLogits(execution, {
    pair, tokens, channels, bins: shape.distogramBins,
    weights: w.distogram.weights, bias: w.distogram.bias,
  });
  const probabilities = distogramProbabilities(execution, {
    logits, pairs, bins: shape.distogramBins,
  });
  const contacts = Float32Array.from(execution.read(probabilities));
  execution.release(probabilities);
  execution.release(logits);
  call.onStage?.("distogram", { bins: shape.distogramBins });

  // --- the sampler ---------------------------------------------------------
  // 🔴 WINDOWED HERE AND DENSE ABOVE - see the note at the top.
  const denoiserAtomShape = {
    ...atomShape, windowSize: shape.atomWindow,
    channels: shape.atomChannels, heads: shape.atomHeads,
    blocks: shape.atomBlocks, hidden: shape.atomChannels * 2,
    tokenChannels: shape.tokenChannels2,
  };
  const live = [];
  const embed = (scaled) => {
    const out = inputsEmbedder(execution, {
      features: { ...atomFeatures, noisyCoordinates: scaled },
      host, shape: denoiserAtomShape, weights: w.diffusionEncoder,
    });
    live.push(out);
    return out;
  };
  const decode = (act, out) => atomDecoder(execution, {
    tokenAct: act, skip: out.atomState, conditioning: out.conditioning,
    rope: out.rope, valid: out.valid, rank: out.rank, index: out.index,
    shape: denoiserAtomShape, weights: w.diffusionDecoder,
  });
  const releaseEmbedded = () => {
    for (const out of live.splice(0)) {
      for (const tensor of [out.atomState, out.conditioning, out.tokenAct,
        out.rope.cos, out.rope.sin, out.rank]) {
        execution.release(tensor);
      }
    }
  };
  const sampled = sampleOnWebGl2(execution, {
    shape: {
      tokens, atoms, pairChannels: channels, singleInputs: shape.singleInputs,
      tokenChannels: shape.tokenChannels2, multiplier: shape.transitionMultiplier,
      tokenHeads: shape.tokenHeads, sigmaData: shape.sigmaData,
    },
    weights: w.denoiser,
    sInputs, zTrunk: pair, relPos: relative,
    hostFourier: { weights: w.hostFourier.weights, offsets: w.hostFourier.offsets },
    embed: (scaled) => embed(scaled),
    decode: (act, out) => {
      const update = decode(act, out);
      return update;
    },
    release: releaseEmbedded,
    atomMask: features.mask, atoms,
    steps: call.steps ?? 15, seed: call.seed ?? 42,
    ...(call.sampler ?? {}),
    onStep: (frame) => {
      releaseEmbedded();
      call.onStep?.(frame);
    },
  });
  releaseEmbedded();
  call.onStage?.("sample", { steps: sampled.steps });

  // --- the confidence head -------------------------------------------------
  // One atom a token: CB where there is one, else CA, else the token's first -
  // and a nucleotide's base atom. `representativeAtoms` returns the Int32Array.
  const repAtom = representativeAtoms(features, tokens);
  const coordinates = upload(sampled.coordinates, "ef2.coordinates");
  const scores = esmfold2Confidence(execution, {
    tokens, atoms,
    sInputs, pair, pairBias: relative, coordinates,
    repAtom: upload(Float32Array.from(repAtom), "ef2.rep-atom"),
    atomToToken: atomFeatures.atomToToken,
    tokenMask: upload(features.tokenMask, "ef2.token-mask"),
    atomMask: atomFeatures.mask,
    slots: upload(
      intraTokenSlots(features.atomToToken, w.confidence.maxAtomsPerToken),
      "ef2.slots"),
    weights: w.confidence, blocks: w.confidenceBlocks,
  });
  const plddt = Float32Array.from(execution.read(scores.plddt));
  const plddtPerAtom = Float32Array.from(execution.read(scores.plddtPerAtom));
  const pae = Float32Array.from(execution.read(scores.pae));
  const paeLogits = Float32Array.from(execution.read(scores.paeLogits));
  for (const tensor of Object.values(scores)) execution.release(tensor);
  const tm = tmScores(paeLogits, features.tokenMask,
    features.asymId ?? new Int32Array(tokens), tokens, w.confidence.paeBins);
  call.onStage?.("confidence", {});

  execution.release(relative);
  execution.release(pair);
  for (const tensor of held) execution.release(tensor);

  let weighted = 0;
  let total = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    weighted += plddtPerAtom[atom] * features.mask[atom];
    total += features.mask[atom];
  }
  return {
    coordinates: sampled.coordinates, plddt, plddtPerAtom, pae, contacts,
    ptm: tm.ptm, iptm: tm.iptm,
    meanPlddt: total === 0 ? NaN : (weighted / total) * 100,
    tokens, atoms, steps: sampled.steps, loops: loops + 1,
  };
}

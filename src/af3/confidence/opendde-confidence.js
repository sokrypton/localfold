/**
 * OpenDDE's confidence head: pLDDT, PAE, PDE and experimentally-resolved.
 *
 * 🔴 IT SHARES NOT ONE TENSOR NAME WITH AlphaFold 3's, and the shape of the
 * computation differs too. AF3 embeds target_feat and the trunk's DISTOGRAM
 * into a fresh pair; OpenDDE initialises its pair from `s_inputs` as a row and
 * a column, then adds a distance embedding of the structure the sampler
 * actually produced - so this head reads the PREDICTION, and cannot run before
 * the sampler has.
 *
 * 🔴 AND pLDDT IS A PER-ATOM EINSUM AGAINST A WEIGHT CHOSEN BY THE ATOM'S
 * SLOT. `plddt_weight` is [24, c_s, 50]: an atom takes its TOKEN's single
 * representation and the matrix belonging to its dense slot within that token,
 * which is how one head gives 24 atoms of a residue different answers from one
 * vector. Reading it as [c_s, 50] and broadcasting gives every atom of a token
 * the same pLDDT, which looks plausible on a backbone and is wrong everywhere.
 *
 * 🔴 AND WHAT COMES OUT IS A SCORE, NOT LOGITS. Upstream records the trap: the
 * head returned raw logits under the same key the shared head uses for a 0-100
 * number, so every consumer read a (n_atom, 50) tensor as a pLDDT and the mean
 * came out at -0.1 on a good fold. The bins are OpenDDE's own - pLDDT 50 over
 * [0, 1] scaled by 100, PAE and PDE 64 over [0, 32] - and the reduction is a
 * softmax against bin CENTRES.
 */
import { layerNorm, linear } from "../trunk/pairformer-reference.js";
import { tmAdjustedPae } from "./ptm-reference.js";
import { Af3PairformerStackGpu } from "../trunk/pairformer-block-webgpu.js";
import { openddeAtomReadouts, openddePairInit, openddePairReadouts }
  from "./opendde-confidence-webgpu.js";

/** OpenDDE's distance bins: 3.25 to 52.0 in steps of 1.25, the last open. */
const BIN_START = 3.25;
const BIN_END = 52.0;
const BIN_STEP = 1.25;

/** softmax over bins, weighted by their centres. */
function expectedFromLogits(logits, rows, bins, minBin, maxBin) {
  const width = (maxBin - minBin) / bins;
  const out = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) largest = Math.max(largest, logits[base + b]);
    let total = 0;
    let weighted = 0;
    for (let b = 0; b < bins; b += 1) {
      const value = Math.exp(logits[base + b] - largest);
      total += value;
      weighted += value * (minBin + width * (b + 0.5));
    }
    out[row] = weighted / total;
  }
  return out;
}

/**
 * The pair this head starts from: the refined pair, plus s_inputs as a row and
 * a column, plus two embeddings of the predicted structure's distances.
 *
 * Host-side because it is the cheap half - the per-token projections are n, not
 * n^2, and only the distance embeddings are per pair.
 */
export function confidencePairInit(pair, singleInputs, coordinates, tokens, weights) {
  const c = weights.pairChannels;
  const inputWidth = weights.singleInputChannels;
  // 🔴 s1 IS THE COLUMN AND s2 IS THE ROW. Upstream writes `s1[None, :, :]`
  // and `s2[:, None, :]`, so s1 is indexed by j and s2 by i. Swapping them
  // transposes a term of the initialisation and changes nothing about a shape.
  const s1 = linear(singleInputs, tokens, inputWidth, c, weights.s1);
  const s2 = linear(singleInputs, tokens, inputWidth, c, weights.s2);

  const out = new Float32Array(tokens * tokens * c);
  const bins = weights.distanceBins;
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const at = (i * tokens + j) * c;
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const d = coordinates[i * 3 + axis] - coordinates[j * 3 + axis];
        squared += d * d;
      }
      const distance = Math.sqrt(Math.max(1e-10, squared));
      // The one-hot bin, which is a single row of the embedding - so this is a
      // gather rather than the 39-wide matrix multiply it is written as.
      let bin = Math.floor((distance - BIN_START) / BIN_STEP);
      if (distance < BIN_START) bin = -1;
      if (bin >= bins) bin = bins - 1;      // ...the last bin is open-ended.
      for (let d = 0; d < c; d += 1) {
        out[at + d] = pair[at + d] + s1[j * c + d] + s2[i * c + d]
          + (bin >= 0 ? weights.distance[bin * c + d] : 0)
          + distance * weights.distanceRaw[d];
      }
    }
  }
  return out;
}

/**
 * @param {GPUDevice} device
 * @param {object} input {tokens, singleInputs, single, pair, coordinates,
 *   seqMask, atomToToken, atomToSlot, atomCount, extraPairBias}
 * @param {object} weights from openddeConfidenceWeights
 * @param {object} dialect
 */
export async function openddeConfidence(device, input, weights, dialect, options = {}) {
  const { tokens } = input;
  const c = weights.pairChannels;
  const cs = weights.singleChannels;

  // 🔴 THE TRUNK SINGLE IS CLAMPED BEFORE IT IS NORMALISED. Upstream clips to
  // +/-512 and then LayerNorms; the clip is not decoration on a representation
  // that has been through a trunk and a refiner.
  const clamped = Float32Array.from(input.single,
    (value) => Math.min(512, Math.max(-512, value)));
  const single = layerNorm(clamped, tokens, cs,
                           weights.inputStrunkLnScale, weights.inputStrunkLnOffset);

  const seqMask = input.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // 🔴 THE PER-PAIR HALF ON THE DEVICE, AND ITS RESULT STAYS THERE. Only the
  // four-block stack reads it, and the stack takes a buffer now - so a
  // `tokens^2 x 384` tensor is neither read back nor uploaded. The per-TOKEN
  // projections stay on the host: they are n and not n^2, which is what the
  // note above this function meant and is true of them.
  // 🔴 THE FOUR PINS AlphaFold 3's HEAD CARRIES, WHICH THIS ONE NEVER HAD.
  // `Af3ConfidenceHeadGpu` runs its four blocks in f32 and with the matrix pair
  // kernels off, because pLDDT and PAE are softmaxes over 50 and 64 bins - the
  // most amplifying thing the model emits - and it measured all four outputs
  // FAILING with those kernels on. This stack took only the bundle's weight
  // precision, so OpenDDE's confidence ran at the trunk's defaults: f16 staging
  // wherever the device has it, and f16 matrix kernels wherever it has matrix
  // units, which no precision option reaches.
  //
  // Against af3-any-model on 6MRR (check-opendde-confidence-oracle.js), M2:
  //
  //                 pLDDT      PAE        PDE
  //   before        4.57e-4    1.04e-2    1.43e-2
  //   pinned        1.18e-7    9.11e-7    1.20e-6
  //
  // The A100 read 4.68e-3 on PAE "identical with --f16=off" and concluded it
  // was not precision. It was: --f16=off cannot reach the matrix kernels, so on
  // a device with matrix units the flag changes nothing, and on the M2 - which
  // has none at these widths - the same flag removed all of it.
  const stack = new Af3PairformerStackGpu(device, {
    pairWeightPrecision: weights.weightPrecision,
    stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32",
    pairMatrixKernels: false,
  });
  const onHost = options.hostReadouts === true;
  const built = onHost ? undefined : await openddePairInit(device, {
    tokens, channels: c, pair: input.pair, pairBuffer: input.pairBuffer,
    // s1 and s2 are projected on the device; see openddePairInit.
    singleInputs: input.singleInputs, singleInputChannels: weights.singleInputChannels,
    coordinates: input.coordinates, binStart: BIN_START, binStep: BIN_STEP,
  }, weights, stack.allocator);
  // 🔴 THE HEAD'S INPUT PAIR IS DEAD ONCE THE PAIR INIT HAS BEEN SUBMITTED: the
  // blocks run on the init's own buffer. Handing it back here rather than after
  // the head takes `tokens^2 x 384` floats off the fold's peak - 359 MiB at 495
  // structural tokens - which is this head, four blocks at the structural
  // token count. Submitted work keeps what it binds.
  if (built !== undefined) input.releasePairInput?.();
  const pair = onHost
    ? confidencePairInit(input.pair, input.singleInputs, input.coordinates, tokens, weights)
    : new Float32Array(0);

  // 🔴 AND THE REFINED PAIR STAYS ON THE DEVICE TOO. PAE and PDE are the only
  // readers and both are kernels now, so reading `tokens^2 x 384` back to
  // upload it again was the last round trip in this head. The stack updates
  // `built.allocation` in place - a pair track is a residual chain - so the
  // pair init's buffer is what the readouts bind and what is released below.
  const refined = await stack.run(
    { tokens, pair, single, pairMask, seqMask }, weights.blocks, dialect,
    { extraPairBias: input.extraPairBias,
      ...(built === undefined ? {} : { pairBuffer: built.allocation.buffer, keepPair: true }) });

  const pairs = tokens * tokens;
  // PAE from the pair; PDE from the SYMMETRISED pair - a distance error is
  // symmetric and an aligned error is not, which is the whole difference.
  //
  // 🔴 ON THE GPU, AND THE HOST PATH IS THE REFERENCE. These two were 1.6
  // seconds of a 13.5-second OpenDDE fold - 138 million multiply-accumulates
  // each at 130 structural tokens - under a comment calling this head's host
  // half "the cheap half". `--host-readouts` is what
  // check-opendde-confidence.js compares against.
  const pairReadouts = options.hostReadouts === true
    ? hostPairReadouts(refined.pair, tokens, c, weights, input.tmTokens)
    : await openddePairReadouts(device, { pairBuffer: built.allocation.buffer,
                                          tokens, channels: c,
                                          tmTokens: input.tmTokens }, weights);
  built?.release();

  // pLDDT and resolved: per ATOM, against the matrix its dense SLOT names.
  //
  // 🔴 ON THE GPU, AND THE HOST PATH IS THE REFERENCE. At 130 structural tokens
  // and 24 dense slots these two are 3120 atoms x 384 channels x {50, 2} bins
  // with a LayerNorm each, and they measured 148 ms inside a fold.
  const atoms = input.atomCount;
  const atomReadouts = options.hostReadouts === true
    ? hostAtomReadouts(refined.single, atoms, cs, input, weights)
    : await openddeAtomReadouts(device, {
      single: refined.single, atoms, channels: cs,
      atomToToken: input.atomToToken, atomToSlot: input.atomToSlot,
    }, weights);

  // 🔴 THE REDUCTIONS ARE OpenDDE's OWN BINS. pLDDT is 50 bins over [0, 1] and
  // is scaled by 100; PAE and PDE are 64 over [0, 32]. Reading any of them on
  // AlphaFold 3's grid gives a number in the right range and the wrong place.
  const { pae, pde, tm } = pairReadouts;
  const { plddt, resolved } = atomReadouts;

  // 🔴 AND THE TM-ADJUSTED PAE WHERE THE CALLER ASKED FOR ONE, which is what
  // pTM and ipTM are reduced from. It is absent rather than zero when nobody
  // passed a token count: a pTM of 0 is a confident failure, and "this head
  // reports none" is a different statement.
  return { plddt, pae, pde, resolved, ...(tm === undefined ? {} : { tm }) };
}

/**
 * The PAE and PDE readouts on the host: what the GPU pair readouts replaced,
 * kept as the reference check-opendde-confidence.js holds them to.
 */
export function hostPairReadouts(pair, tokens, c, weights, tmTokens) {
  const pairs = tokens * tokens;
  const paeLogits = linear(
    layerNorm(pair, pairs, c, weights.paeLnScale, weights.paeLnOffset),
    pairs, c, weights.paeBins, weights.pae);
  const symmetric = new Float32Array(pairs * c);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      for (let d = 0; d < c; d += 1) {
        symmetric[(i * tokens + j) * c + d] =
          pair[(i * tokens + j) * c + d] + pair[(j * tokens + i) * c + d];
      }
    }
  }
  const pdeLogits = linear(
    layerNorm(symmetric, pairs, c, weights.pdeLnScale, weights.pdeLnOffset),
    pairs, c, weights.pdeBins, weights.pde);
  // 🔴 AND NEITHER EXPECTATION IS MASKED, WHICH RESTS ON THIS PIPELINE NOT
  // PADDING. The reference multiplies `full_pae` and `full_pde` by
  // `seq_mask[:, None] * seq_mask[None, :]` (model.py), and `featurise.js`
  // writes `seqMask[token] = 1` on every one of the `tokens` it sizes the
  // arrays from - four branches, no gaps - so that mask is all ones and the
  // product is the identity. A bucketed or padded token axis would make this
  // wrong here AND make `tmTokens` (which is the mask's SUM in the reference)
  // wrong at every call site.
  return {
    pae: expectedFromLogits(paeLogits, pairs, weights.paeBins, 0, 32),
    pde: expectedFromLogits(pdeLogits, pairs, weights.pdeBins, 0, 32),
    // ...and the TM-adjusted PAE, which is the head's own arithmetic over the
    // same logits - see the shader. `tmAdjustedPae` is AlphaFold 3's, written
    // out once and applied to whichever head produced the distribution.
    ...(tmTokens === undefined ? {} : { tm: tmAdjustedPae(
      paeLogits, tokens, binCentresFor(weights.paeBins, 0, 32),
      tmTokenCounts(tokens, tmTokens)) }),
  };
}

/** The bin centres a readout reduces over, as the host path already assumes. */
function binCentresFor(bins, minBin, maxBin) {
  const width = (maxBin - minBin) / bins;
  return Float32Array.from({ length: bins },
    (unused, bin) => minBin + width * (bin + 0.5));
}

/**
 * 🔴 d0 IS THE CALLER'S, NOT THIS HEAD'S TOKEN COUNT. `tmAdjustedPae` takes a
 * per-pair token count because AlphaFold 3's INTERFACE form varies it by the
 * two chains; the GLOBAL form - the one pTM, ipTM and the per-chain breakdown
 * all read - is one number, and it is the count pTM is REPORTED over. That is
 * the residue tokens, which is not this head's structural count the moment a
 * ligand or a modified residue expands it.
 */
function tmTokenCounts(tokens, tmTokens) {
  return new Float32Array(tokens * tokens).fill(tmTokens);
}

/**
 * pLDDT and "resolved" on the host: what the GPU atom readouts replaced, kept
 * as the reference check-opendde-confidence.js holds them to.
 */
export function hostAtomReadouts(single, atoms, cs, input, weights) {
  const gathered = new Float32Array(atoms * cs);
  for (let atom = 0; atom < atoms; atom += 1) {
    const from = input.atomToToken[atom] * cs;
    for (let d = 0; d < cs; d += 1) gathered[atom * cs + d] = single[from + d];
  }
  const readout = (scale, offset, table, bins) => {
    const normalised = layerNorm(gathered, atoms, cs, scale, offset);
    const logits = new Float32Array(atoms * bins);
    for (let atom = 0; atom < atoms; atom += 1) {
      const matrix = input.atomToSlot[atom] * cs * bins;
      for (let b = 0; b < bins; b += 1) {
        let total = 0;
        for (let d = 0; d < cs; d += 1) {
          total += normalised[atom * cs + d] * table[matrix + d * bins + b];
        }
        logits[atom * bins + b] = total;
      }
    }
    return logits;
  };
  const plddtLogits = readout(weights.plddtLnScale, weights.plddtLnOffset,
                              weights.plddtWeight, weights.plddtBins);
  const resolvedLogits = readout(weights.resolvedLnScale, weights.resolvedLnOffset,
                                 weights.resolvedWeight, weights.resolvedBins);
  const plddt = expectedFromLogits(plddtLogits, atoms, weights.plddtBins, 0, 1);
  for (let index = 0; index < plddt.length; index += 1) plddt[index] *= 100;
  const resolved = new Float32Array(atoms);
  for (let atom = 0; atom < atoms; atom += 1) {
    const a = resolvedLogits[atom * 2];
    const b = resolvedLogits[atom * 2 + 1];
    const largest = Math.max(a, b);
    resolved[atom] = Math.exp(b - largest) / (Math.exp(a - largest) + Math.exp(b - largest));
  }
  return { plddt, resolved };
}

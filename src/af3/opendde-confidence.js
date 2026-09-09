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
import { layerNorm, linear } from "./pairformer-reference.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
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
  const stack = new Af3PairformerStackGpu(
    device, { pairWeightPrecision: weights.weightPrecision });
  const onHost = options.hostReadouts === true;
  const built = onHost ? undefined : await openddePairInit(device, {
    tokens, channels: c, pair: input.pair, pairBuffer: input.pairBuffer,
    s1: linear(input.singleInputs, tokens, weights.singleInputChannels, c, weights.s1),
    s2: linear(input.singleInputs, tokens, weights.singleInputChannels, c, weights.s2),
    coordinates: input.coordinates, binStart: BIN_START, binStep: BIN_STEP,
  }, weights, stack.allocator);
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
    ? hostPairReadouts(refined.pair, tokens, c, weights)
    : await openddePairReadouts(device, { pairBuffer: built.allocation.buffer,
                                          tokens, channels: c }, weights);
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
  const { pae, pde } = pairReadouts;
  const { plddt, resolved } = atomReadouts;

  return { plddt, pae, pde, resolved };
}

/**
 * The PAE and PDE readouts on the host: what the GPU pair readouts replaced,
 * kept as the reference check-opendde-confidence.js holds them to.
 */
export function hostPairReadouts(pair, tokens, c, weights) {
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
  return {
    pae: expectedFromLogits(paeLogits, pairs, weights.paeBins, 0, 32),
    pde: expectedFromLogits(pdeLogits, pairs, weights.pdeBins, 0, 32),
  };
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

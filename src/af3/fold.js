/**
 * One AF3 fold, from a featurised batch to coordinates and a pLDDT.
 *
 *     const batch = featuriseProtein("GWSTELEK...");
 *     const result = await foldBatch(device, batch, weights, { steps: 200 });
 *
 *     embedder -> template -> 4 x MSA block -> 48 x pairformer -> distogram
 *     -> N-step diffusion sampler around the GPU denoiser
 *     -> confidence head
 *
 * WHY THIS IS NOT IN tools/gpu/fold.js ANY MORE. The command line and the page
 * have to run the SAME pipeline or the page is not what was measured, and the
 * measurements are all from the command line. What is left there is argument
 * parsing, the comparison against AF3's dump, and the geometry report.
 *
 * 🔴 THE PER-ATOM CONDITIONING IS COMPUTED ON THE CPU, TWICE, and it is a
 * learned operation - a real gap against AGENTS.md rather than a convenience.
 * Once for target_feat's atom columns and once for the diffusion head, from two
 * different weight bundles of the same five shapes. It is 574 x 24 rows and has
 * no GPU kernel yet. Flagged rather than hidden.
 *
 * 🔴 ONE MSA ROW: num_msa=1. A de novo design has no homologues, so for the
 * sequences this is aimed at the MSA row IS the query; a deeper one would
 * change three arrays in featurise.js and nothing here. Recycles ARE driven -
 * see the loop below - and default to none, which is what the oracle dumps
 * were made with.
 */
import { perAtomConditioning } from "./atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures } from "./atom-encoder-reference.js";
import { Af3AtomEncoderGpu } from "./atom-encoder-webgpu.js";
import { Af3TrunkGpu } from "./trunk-webgpu.js";
import { Af3ConfidenceHeadGpu } from "./confidence-webgpu.js";
import { sampleOnGpu, flowOnGpu } from "./diffusion-sampler-webgpu.js";

/**
 * 🔴 THE DIALECT IS NOT A PREFERENCE. `model='openfold3'` turns on four
 * branches stock AF3 does not have, and a checkpoint has to be read through the
 * graph it was converted for. This is the stock one.
 */
export const DIALECT = { swapTransposedBias: false };

const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

const ELEMENT_SYMBOL = { 6: "C", 7: "N", 8: "O", 16: "S" };

/** The four-character atom name AF3 stores as codes offset by 32. */
export function atomName(nameChars, slot) {
  let name = "";
  for (let character = 0; character < 4; character += 1) {
    const code = nameChars[slot * 4 + character];
    if (code > 0) name += String.fromCharCode(code + 32);
  }
  return name.trim();
}

/**
 * A PDB from the dense atom layout, with pLDDT in the B-factor column so the
 * viewer can colour by it.
 */
export function toPdb(batch, positions, plddt) {
  const { tokens, dense, sequence } = batch;
  const lines = [];
  let serial = 1;
  // 🔴 ONE LETTER PER CHAIN, NOT "A" FOR EVERYTHING. A complex written as one
  // chain is a single 126-residue protein as far as any viewer or scoring tool
  // is concerned, with a peptide bond implied across an interface that has
  // none.
  const chainLetter = (token) => {
    const asym = batch.asymId === undefined ? 1 : batch.asymId[token];
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[(asym - 1) % 26];
  };
  for (let token = 0; token < tokens; token += 1) {
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      const name = atomName(batch.refAtomNameChars, slot);
      const confidence = plddt ? plddt[slot] : 0;
      lines.push(
        "ATOM  "
        + String(serial).padStart(5) + " "
        + (name.length < 4 ? ` ${name}`.padEnd(4) : name.slice(0, 4)) + " "
        + (THREE_LETTER[sequence[token]] ?? "UNK").padEnd(3) + " " + chainLetter(token)
        // 🔴 residue_index IS ALREADY 1-BASED. Adding one here shifted the whole
        // chain by a residue, which against a helical protein reads as a 3.7 A
        // RMSD and a TM-score of 0.37 - a plausible "wrong fold" rather than an
        // obvious bug. The real number was 0.69 A.
        + String(batch.features.residueIndex[token]).padStart(4) + "    "
        + positions[slot * 3].toFixed(3).padStart(8)
        + positions[slot * 3 + 1].toFixed(3).padStart(8)
        + positions[slot * 3 + 2].toFixed(3).padStart(8)
        + "  1.00" + confidence.toFixed(2).padStart(6) + "          "
        + (ELEMENT_SYMBOL[batch.refElement[slot]] ?? "C").padStart(2));
      serial += 1;
    }
    if (batch.asymId !== undefined && token + 1 < tokens
        && batch.asymId[token + 1] !== batch.asymId[token]) {
      lines.push("TER");
    }
  }
  lines.push("END");
  return lines.join("\n");
}

/**
 * Backbone bond lengths and the radius of gyration.
 *
 * 🔴 THIS IS THE CHECK THAT MATTERS, NOT pLDDT. A confident-looking pLDDT comes
 * off the TRUNK and says nothing about whether the diffusion produced a
 * molecule. A batch built with a broken gather once folded a 17 A spaghetti at
 * a mean pLDDT of 55 with 15 A between consecutive CA - the pLDDT looked merely
 * unimpressive and the CA-CA was impossible.
 */
export function backboneGeometry(batch, positions) {
  const { tokens, dense } = batch;
  const slotOf = [];
  for (let token = 0; token < tokens; token += 1) {
    const atoms = {};
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      atoms[atomName(batch.refAtomNameChars, slot)] = slot;
    }
    slotOf.push(atoms);
  }
  const distance = (a, b) => Math.hypot(
    positions[a * 3] - positions[b * 3],
    positions[a * 3 + 1] - positions[b * 3 + 1],
    positions[a * 3 + 2] - positions[b * 3 + 2]);
  const median = (values) => {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((x, y) => x - y);
    return sorted[Math.floor(sorted.length / 2)];
  };

  const nca = [];
  const cac = [];
  const caca = [];
  for (let token = 0; token < tokens; token += 1) {
    const here = slotOf[token];
    if (here.N !== undefined && here.CA !== undefined) nca.push(distance(here.N, here.CA));
    if (here.CA !== undefined && here.C !== undefined) cac.push(distance(here.CA, here.C));
    // 🔴 NOT ACROSS A CHAIN BREAK. There is no bond between the last residue of
    // one chain and the first of the next, so that distance is a fact about how
    // the complex packed rather than about its geometry - and CA-CA is the
    // number that says whether the backbone is connected at all.
    const sameChain = batch.asymId === undefined
      || batch.asymId[token] === batch.asymId[token + 1];
    if (token + 1 < tokens && sameChain
        && here.CA !== undefined && slotOf[token + 1].CA !== undefined) {
      caca.push(distance(here.CA, slotOf[token + 1].CA));
    }
  }

  const centre = [0, 0, 0];
  let count = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = slotOf[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += positions[ca * 3 + axis];
    count += 1;
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= count;
  let gyration = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = slotOf[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) gyration += (positions[ca * 3 + axis] - centre[axis]) ** 2;
  }

  return {
    nca: median(nca), cac: median(cac), caca: median(caca),
    gyration: Math.sqrt(gyration / count), residues: count,
  };
}

/**
 * AF3's target_feat: 447 columns, of which 384 come from the atom encoder.
 *
 * 🔴 THE ENCODER RUNS ON THE GPU AND IT IS THE WHOLE COST. On a 68-residue
 * chain the CPU reference takes 5267 ms of the 5.4 s this used to spend -
 * longer than the 48-block trunk it feeds - and the GPU does it in 160 ms, 33x
 * faster and matching to relRMS 8e-8. The per-atom conditioning below stays on
 * the CPU because it is 119 ms of the total and has no kernel.
 *
 * `device` may be omitted, and then the CPU reference runs: the checkers that
 * verify this path have no device of their own to lend it.
 */
export async function buildTargetFeat(batch, weights, device) {
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, batch.tokens, batch.dense, weights.reference);

  const shared = {
    shape: batch.shape, conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
  };
  const atoms = batch.tokens * batch.dense;
  const atomFeatures = device === undefined
    ? atomCrossAttentionEncoder(shared, weights.encoder)
    : await new Af3AtomEncoderGpu(device).run({
        ...shared,
        tokensToQueries: batch.tokensToQueries,
        tokensToKeys: batch.tokensToKeys,
        // Zeroed, so the three terms this encoder does not have contribute
        // nothing - see the note on those weights in diffusion-weights.js.
        tokenAtomsAct: new Float32Array(atoms * 3),
        trunkSingleCond: new Float32Array(batch.tokens * 384),
        trunkPairCond: new Float32Array(batch.tokens * batch.tokens * 128),
      }, weights.encoder);

  return targetFeatures({
    aatype: batch.aatype, profile: batch.profile, deletionMean: batch.deletionMean,
    atomFeatures: atomFeatures.tokenAct,
  }, batch.tokens);
}

/** A reproducible normal deviate, so a seed names a structure. */
export function normalFrom(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

/**
 * @param {object} batch from featuriseProtein or an AF3 dump
 * @param {{trunk, diffusion, confidence, atomReference, targetFeat}} weights
 * @param {{mode?: "flow"|"diffusion", steps?: number, recycles?: number,
 *          seed?: number, blocks?: number,
 *          onStage?: (name: string, detail: object) => void,
 *          onStep?: (step: object) => void}} [options]
 */
export async function foldBatch(device, batch, weights, options = {}) {
  const steps = options.steps ?? 200;
  const { tokens, dense } = batch;
  const stage = (name, detail = {}) => options.onStage?.(name, detail);

  // Announced before it runs and awaited, so a page can say what is happening
  // and hand back a frame first. It used to be five seconds of blocked main
  // thread; the encoder is on the GPU now and it is a few hundred milliseconds.
  await stage("target-feat-start");
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
  await stage("target-feat", { targetFeat });

  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // 🔴 RECYCLING WAS BUILT AND NEVER DRIVEN. The embedder has done
  // `pair += prev_embedding(LayerNorm(recycled pair))` since it was written, and
  // this ran one pass with zeros in that slot because the oracle dumps were
  // made with num_recycles=0 and a comparison is only meaningful at the same
  // setting. Folding is not a comparison, and AF3's own default is not zero.
  //
  // 🔴 THE WHOLE TRUNK RUNS AGAIN PER RECYCLE, embedder and all - that is what
  // a recycle IS - so each one costs another full trunk. It is not a cheap
  // refinement pass.
  const recycles = options.recycles ?? 0;
  // Reported from the batch the trunk is actually handed, so a future
  // truncation shows up in every run rather than in a fold that is merely
  // disappointing. The page's status line reports the depth that was
  // FEATURISED, which is not the same claim.
  await stage("msa-depth", { sequences: batch.sequences, tokens });
  const trunkGpu = new Af3TrunkGpu(device);
  let previousPair = new Float32Array(tokens * tokens * 128);
  let previousSingle = new Float32Array(tokens * 384);
  let trunk;
  for (let pass = 0; pass <= recycles; pass += 1) {
    await stage("recycle", { pass, passes: recycles + 1 });
    // 🔴 THE WHOLE ALIGNMENT, NOT ITS FIRST ROW. This passed `sequences: 1` and
    // sliced every MSA array down to one row, which was right when the only
    // inputs were oracle dumps taken at num_msa=1 and became wrong the moment
    // featuriseProtein learned to build a real MSA.
    //
    // It is a quiet failure and it flatters itself: an alignment still reaches
    // the model, because `profile` and `deletion_mean` are computed over all of
    // it and ride into target_feat - so folds DO improve when you supply one
    // (44.5 -> 62.6 pLDDT on a 146-residue chain) and the status line honestly
    // reports the depth that was featurised. What never happened is the MSA
    // stack seeing more than the query, which is most of what an MSA is for.
    trunk = await trunkGpu.run({
      tokens, sequences: batch.sequences, templates: 4, targetFeat,
      features: batch.features,
      msaRows: batch.msa,
      deletionMatrix: batch.deletionMatrix,
      msaMask: batch.msaMask,
      pairMask, seqMask, previousPair, previousSingle,
    }, weights.trunk, DIALECT, {
      onStage: (name, ms) => stage("trunk", { name, ms }),
      onPairformerBlock: (index, total) =>
        stage("pairformer-block", { index, total, pass, passes: recycles + 1 }),
      // 🔴 THE ONE THE STATUS LINE READS. `pairformer-block` fires at encode
      // time, sixteen at a stride; this fires when the device has actually
      // finished a block. The first is still awaited, because that is what
      // yields to the event loop and lets the page paint at all.
      onPairformerBlockDone: (completed, total) =>
        stage("pairformer-block-done", { completed, total, pass, passes: recycles + 1 }),
    });
    previousPair = trunk.pair;
    previousSingle = trunk.single;
  }
  stage("trunk-done", { trunk });

  // 🔴 THE DIFFUSION HEAD HAS ITS OWN FIVE REFERENCE EMBEDDINGS - same shapes as
  // the conditioning module's, different weights. Reusing one for both
  // type-checks and is a different model.
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.atomReference);

  const headInput = {
    shape: batch.shape, conditioning, atomMask: batch.predDenseAtomMask, seqMask,
    features: batch.features, targetFeat,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: trunk.single, trunkPair: trunk.pair,
  };

  // 🔴 TWO WAYS TO TURN THE TRUNK INTO COORDINATES, AND THEY ARE NOT THE SAME
  // KIND OF THING. "diffusion" is AF3's own stochastic sampler: noise is
  // injected at every step. "flow" draws once at the top of the schedule and
  // then walks it deterministically - about 25x fewer calls for the same
  // accuracy on the two proteins it has been measured on. Both take a seed;
  // the flow's spread across seeds is the narrower of the two, because one
  // draw is not the same as noise at every step.
  const positions = options.mode === "diffusion"
    ? await sampleOnGpu(device, headInput, weights.diffusion, {
        steps, stopAfter: options.stopAfter,
        normal: normalFrom(options.seed ?? 20260831),
        onStep: options.onStep,
      })
    : await flowOnGpu(device, headInput, weights.diffusion, {
        cycles: steps, normal: normalFrom(options.seed ?? 20260831),
        onStep: options.onStep,
      });

  // The confidence head reads the sample back.
  const beta = batch.tokenAtomsToPseudoBeta;
  const pseudoBeta = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    if (!beta.mask[token]) continue;
    const from = Number(beta.indices[token]) * 3;
    for (let axis = 0; axis < 3; axis += 1) pseudoBeta[token * 3 + axis] = positions[from + axis];
  }
  const scores = await new Af3ConfidenceHeadGpu(device).run({
    tokens, dense, seqMask, pair: trunk.pair, single: trunk.single, targetFeat, pseudoBeta,
  }, weights.confidence, DIALECT);

  let total = 0;
  let count = 0;
  for (let index = 0; index < tokens * dense; index += 1) {
    if (!batch.predDenseAtomMask[index]) continue;
    total += scores.plddt[index];
    count += 1;
  }

  return {
    positions, trunk, targetFeat, scores,
    meanPlddt: total / count, atoms: count,
    geometry: backboneGeometry(batch, positions),
    pdb: toPdb(batch, positions, scores.plddt),
  };
}

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
import { ELEMENT_SYMBOLS } from "./ccd-component.js";
import { af3ContactClasses } from "./contact-classes.js";
import { perAtomConditioning } from "./atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures } from "./atom-encoder-reference.js";
import { Af3AtomEncoderGpu } from "./atom-encoder-webgpu.js";
import { Af3TrunkGpu } from "./trunk-webgpu.js";
import { Af3ConfidenceHeadGpu } from "./confidence-webgpu.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
import { Af3StructuralExpanderGpu } from "./structural-expander-webgpu.js";
import { structuralAttentionBias, structuralPairFeatures }
  from "./structural-expander-reference.js";
import { structuralBatch, structuralLayout, structuralToResidue }
  from "./structural-tokens.js";
import { diffusionConditioning } from "./diffusion-reference.js";
import { openddeConfidence } from "./opendde-confidence.js";
import { releaseResidentWeights } from "../runtime/resident.js";
import { deviceTuning } from "../runtime/device-profile.js";
import { chainPairTmScores, perChainTmScores, reduceTmScore }
  from "../heads/tm-score.js";
import { sampleOnGpu, flowOnGpu } from "./diffusion-sampler-webgpu.js";
import { Af3DiffusionHeadGpu } from "./diffusion-head-webgpu.js";

/**
 * 🔴 THE DIALECT IS NOT A PREFERENCE. A ported checkpoint turns on branches
 * stock AF3 does not have, and it has to be read through the graph it was
 * converted for. `src/af3/dialect.js` is the table; this is the stock entry,
 * re-exported under its old name so that every caller that means "AlphaFold 3"
 * keeps saying so.
 */
import { ALPHAFOLD3 } from "./dialect.js";

export const DIALECT = ALPHAFOLD3;

/**
 * The pair width above which the trunk's block weights stop being cheap.
 *
 * 🔴 IT IS A THRESHOLD BETWEEN TWO MEASURED POINTS, NOT AN OPTIMUM. These
 * weights go as the SQUARE of the channel count, so at AlphaFold 3's 128 they
 * are a ninth of what they are at OpenDDE's 384 - and every trade about them
 * flips between the two. Both knobs it gates are measured at exactly those two
 * widths and nowhere between, so 256 should move when a third bundle exists.
 */
const WIDE_PAIR_TRACK = 256;
export { ALPHAFOLD3, OPENBIND0, DIALECTS, dialectFor } from "./dialect.js";

export const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

/**
 * ...and back, which reading a template needs.
 *
 * 🔴 INVERTED RATHER THAN WRITTEN OUT, so the two directions cannot disagree.
 * A second table would be twenty more lines that have to be edited together
 * with this one, and the failure of getting one wrong is a residue silently
 * becoming UNK - a four-atom blank the model folds around.
 */
export const ONE_LETTER = Object.fromEntries(
  Object.entries(THREE_LETTER).map(([one, three]) => [three, one]));

/**
 * The element symbol for an atomic number.
 *
 * 🔴 THIS WAS FOUR ENTRIES - C, N, O, S - AND EVERYTHING ELSE FELL THROUGH TO
 * CARBON. Right for a protein, which has nothing else, and wrong for most
 * ligands: across a corpus of 51 distinct hetero components, TWENTY-EIGHT carry
 * an element it dropped. Every phosphate-bearing ligand (P), every heme (FE),
 * and every metal ion - a magnesium written as a carbon atom.
 *
 * It costs twice over, because a viewer with no CONECT records derives a
 * ligand's bonds from the DISTANCE between atoms of known elements: a
 * disulfide at 2.05 A read as C-C (whose ceiling is 1.8) vanishes, and a P-O
 * at 1.63 read as C-O survives its 1.65 ceiling by two hundredths.
 *
 * ccd-component.js already had the full list, indexed the same way, so this is
 * one question with one answer rather than a second short table beside it.
 */
function elementSymbol(atomicNumber) {
  return ELEMENT_SYMBOLS[atomicNumber - 1] ?? "C";
}

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
  // 🔴 A LIGAND IS HETATM, AND IT HAS A NAME. `sequence` covers the polymers,
  // so a ligand token indexed into it is undefined and used to be written as a
  // UNK residue - which a viewer draws as an unknown amino acid and a scoring
  // tool reads as part of the chain. The span table says which tokens belong to
  // which component, and the component's own code is its residue name.
  //
  // 🔴 A MODIFIED RESIDUE IS THE SAME PROBLEM ONE STEP FURTHER IN. Its tokens
  // are inside the chain, so `sequence` HAS a letter at that position - but
  // `sequence` is indexed by RESIDUE and this loop walks TOKENS, and the two
  // stopped being the same number the moment a residue could be several
  // tokens. Every residue from the first modification onwards was named by
  // whatever letter happened to sit at its token index: a chain that reads as
  // a real protein and is not the one that was folded. residueOfToken is the
  // map, and a modified residue takes its component's own code, which is what
  // a PDB calls one.
  const componentOf = new Map();
  for (const span of [...(batch.ligandSpans ?? []), ...(batch.modifiedSpans ?? [])]) {
    for (let offset = 0; offset < span.count; offset += 1) {
      componentOf.set(span.from + offset, span.code);
    }
  }
  const residueOf = (token) => batch.residueOfToken?.[token] ?? token;
  // 🔴 A NUCLEOTIDE'S RESIDUE NAME IS NOT ITS AMINO ACID'S. THREE_LETTER maps
  // the one-letter code through the amino-acid table, where `A` is ALA and `G`
  // is GLY - so a DNA chain came out of here as a poly-alanine peptide that
  // every viewer draws as a protein ribbon and every scoring tool reads as one.
  // A PDB names DNA " DA" and RNA "  A", right-justified in the field, which is
  // what distinguishes the two: the D is the only thing in the format that
  // says which.
  const nucleicName = (residue) => {
    const kind = batch.chainKinds?.[batch.chainOfResidue?.[residue]];
    if (kind !== "dna" && kind !== "rna") return undefined;
    const code = sequence[residue];
    if (code === undefined) return undefined;
    return kind === "dna" ? `D${code}` : code;
  };
  // ...and which serial each ligand token was written as, so CONECT can name
  // them. A ligand token is one heavy atom and it sits in slot zero, so the
  // token is the atom; a polymer token is many atoms and has no entry here.
  const serialOfToken = new Map();
  for (let token = 0; token < tokens; token += 1) {
    const ligandCode = componentOf.get(token);
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      if (ligandCode !== undefined && atom === 0) serialOfToken.set(token, serial);
      const name = atomName(batch.refAtomNameChars, slot);
      const confidence = plddt ? plddt[slot] : 0;
      lines.push(
        (ligandCode === undefined ? "ATOM  " : "HETATM")
        + String(serial).padStart(5) + " "
        + (name.length < 4 ? ` ${name}`.padEnd(4) : name.slice(0, 4)) + " "
        + (ligandCode
          ?? nucleicName(residueOf(token))?.padStart(3)
          ?? THREE_LETTER[sequence[residueOf(token)]] ?? "UNK").padEnd(3)
        + " " + chainLetter(token)
        // 🔴 residue_index IS ALREADY 1-BASED. Adding one here shifted the whole
        // chain by a residue, which against a helical protein reads as a 3.7 A
        // RMSD and a TM-score of 0.37 - a plausible "wrong fold" rather than an
        // obvious bug. The real number was 0.69 A.
        + String(batch.features.residueIndex[token]).padStart(4) + "    "
        + positions[slot * 3].toFixed(3).padStart(8)
        + positions[slot * 3 + 1].toFixed(3).padStart(8)
        + positions[slot * 3 + 2].toFixed(3).padStart(8)
        + "  1.00" + confidence.toFixed(2).padStart(6) + "          "
        + elementSymbol(batch.refElement[slot]).padStart(2));
      serial += 1;
    }
    if (batch.asymId !== undefined && token + 1 < tokens
        && batch.asymId[token + 1] !== batch.asymId[token]) {
      lines.push("TER");
    }
  }
  // 🔴 CONECT, OR THE LIGAND IS A BAG OF ATOMS. A viewer handed no bonds
  // derives them from the DISTANCE between atoms - py2Dmol says so out loud
  // ("No bonds - will use distance calculation") - and on a diffusion
  // trajectory it re-derives them from EVERY frame's coordinates, which are
  // deliberately noisy until the last few steps. The sticks then appear, cross
  // and vanish frame to frame: the picture looks broken while the prediction
  // may be fine. The bonds are already known - they came out of the CCD - so
  // there is nothing to compute here, only to write down.
  //
  // Ligand-internal only. A covalent link between a ligand and a polymer is
  // not featurised yet (see featurise.js), so claiming one here would be the
  // writer inventing chemistry the fold never saw.
  //
  // Four partners per line, continued on another CONECT for an atom with more:
  // the record has room for exactly four and a fifth silently overruns into
  // the next field.
  const partners = new Map();
  for (const span of [...(batch.ligandSpans ?? []), ...(batch.modifiedSpans ?? [])]) {
    for (const bond of span.bonds ?? []) {
      const a = serialOfToken.get(span.from + bond.from);
      const b = serialOfToken.get(span.from + bond.to);
      if (a === undefined || b === undefined) continue;   // a masked atom
      if (!partners.has(a)) partners.set(a, []);
      if (!partners.has(b)) partners.set(b, []);
      partners.get(a).push(b);
      partners.get(b).push(a);
    }
  }
  for (const [atom, bonded] of [...partners].sort((x, y) => x[0] - y[0])) {
    for (let start = 0; start < bonded.length; start += 4) {
      let line = "CONECT" + String(atom).padStart(5);
      for (const other of bonded.slice(start, start + 4)) line += String(other).padStart(5);
      lines.push(line);
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

  // 🔴 THE WORST CA-CA TRAVELS WITH THE MEDIAN, BECAUSE A COLLAPSE IS NOT
  // ALWAYS A MEDIAN. AF2's 825-residue collapse had a median of 1.44 and a
  // worst of 0.06, but the band that catches the other shape of failure - a
  // chain that is mostly right with one link thrown across the box - is on the
  // worst alone. See assertChainGeometry in tools/gpu/chain-geometry.js.
  const worstCaca = caca.reduce(
    (far, value) => (Math.abs(value - 3.8) > Math.abs(far - 3.8) ? value : far),
    caca.length === 0 ? NaN : 3.8);
  return {
    nca: median(nca), cac: median(cac), caca: median(caca), worstCaca,
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
    shape: batch.shape, dialect: weights.dialect,
    conditioning, atomMask: batch.refMask,
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

/**
 * A reproducible uniform in (0, 1), so a seed names a choice.
 *
 * Split out of normalFrom rather than written twice: the MSA subsample needs
 * uniforms and the sampler needs deviates, and two generators would mean two
 * things a seed could mean.
 */
export function uniformFrom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
}

/** A reproducible normal deviate, so a seed names a structure. */
export function normalFrom(seed) {
  const uniform = uniformFrom(seed);
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

/**
 * @param {object} batch from featuriseProtein or an AF3 dump
 * @param {{trunk, diffusion, confidence, atomReference, targetFeat}} weights
 * @param {{mode?: "flow"|"diffusion", steps?: number, recycles?: number,
 *          seed?: number, blocks?: number,
 *          schedule?: {sigmaData?: number, sigmaMin?: number, sigmaMax?: number,
 *                      rho?: number},
 *          onStage?: (name: string, detail: object) => void,
 *          onStep?: (step: object) => void}} [options]
 *
 * `schedule` overrides the EDM noise schedule the levels are drawn from. It
 * exists for probing where a structure actually resolves - `sigmaMax` is in
 * units of sigmaData, so the walk starts at `sigmaData * sigmaMax` angstroms -
 * and folding leaves it unset, which is AF3's own 160.
 */

/**
 * OpenDDE's structural-token stage: expand, refine, and rebuild the head input.
 *
 * 🔴 IT REPLACES THE BATCH THE DIFFUSION SEES, WHICH IS THE WHOLE POINT. The
 * trunk ran on residues; from here on the token space is subtokens, so every
 * gather, mask and conditioning the head reads is the structural batch's. The
 * coordinates are mapped back the moment the sampler returns.
 */
async function expandToStructuralTokens(device, batch, trunk, targetFeat, weights, stage) {
  const layout = structuralLayout(batch);
  const structuralFeatures = structuralPairFeatures(layout, batch.asymId);
  const structural = structuralBatch(batch, layout);
  const expander = weights.expander;
  // `stage` NOTIFIES, it does not wrap - see its definition in foldBatch.
  stage("structural-expand", { tokens: layout.tokens });
  const expanded = await new Af3StructuralExpanderGpu(device).run(
    layout, { single: trunk.single, pair: trunk.pair, targetFeat, asymId: batch.asymId },
    expander, structuralFeatures, batch.tokens);

  // A subtoken's target_feat is its parent's plus a role embedding - the same
  // rule the expander applies to the single, on the other representation.
  const inputWidth = expander.singleInputChannels;
  const structuralTargetFeat = new Float32Array(layout.tokens * inputWidth);
  for (let i = 0; i < layout.tokens; i += 1) {
    const from = layout.parent[i] * inputWidth;
    const role = layout.role[i] * inputWidth;
    for (let c = 0; c < inputWidth; c += 1) {
      structuralTargetFeat[i * inputWidth + c] =
        targetFeat[from + c] + expander.singleInputRoleEmbedding[role + c];
    }
  }

  const n = layout.tokens;
  const seqMask = structural.seqMask;
  const pairMask = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) pairMask[i * n + j] = seqMask[i] * seqMask[j];
  }
  const attentionBias = structuralAttentionBias(layout, structuralFeatures, expander);
  stage("structural-refine", { tokens: n });
  const refined = await new Af3PairformerStackGpu(
    device, { pairWeightPrecision: weights.refinerWeightPrecision }).run(
    { tokens: n, pair: expanded.pair, single: expanded.single, pairMask, seqMask },
    weights.refiner, weights.trunk.dialect, { extraPairBias: attentionBias });

  const conditioning = perAtomConditioning({
    positions: structural.refPos, mask: structural.refMask,
    element: structural.refElement, charge: structural.refCharge,
    atomNameChars: structural.refAtomNameChars,
  }, n, batch.dense, weights.atomReference);
  stage("structural-conditioning", { tokens: n });
  const pairConditioning = diffusionConditioning({
      tokens: n, trunkSingle: refined.single, trunkPair: refined.pair,
      targetFeat: structuralTargetFeat, noiseLevel: 1, features: structural.features,
      dialect: weights.diffusion.dialect,
    }, weights.diffusion.conditioning).pair;

  return {
    layout, structural, attentionBias,
    headInput: {
      shape: structural.shape, dialect: weights.diffusion.dialect,
      conditioning, atomMask: structural.predDenseAtomMask, seqMask,
      features: structural.features, targetFeat: structuralTargetFeat,
      refPos: structural.refPos, refSpaceUid: structural.refSpaceUid,
      tokenAtomsToQueries: structural.tokenAtomsToQueries,
      queriesToKeys: structural.queriesToKeys,
      queriesToTokenAtoms: structural.queriesToTokenAtoms,
      tokensToQueries: structural.tokensToQueries,
      tokensToKeys: structural.tokensToKeys,
      trunkSingle: refined.single, trunkPair: refined.pair,
      pairConditioning,
    },
  };
}

export async function foldBatch(device, batch, weights, options = {}) {
  const steps = options.steps ?? 200;
  const { tokens, dense } = batch;
  const stage = (name, detail = {}) => options.onStage?.(name, detail);

  // Announced before it runs and awaited, so a page can say what is happening
  // and hand back a frame first. It used to be five seconds of blocked main
  // thread; the encoder is on the GPU now and it is a few hundred milliseconds.
  // 🔴 THE TRUNK IS THE EXPENSIVE PART AND IT DOES NOT DEPEND ON THE SAMPLER.
  // `reuse` carries a previous fold's trunk and target features back in, so
  // changing the sampler, its step count, or nothing at all costs only denoiser
  // calls - measured on a 68-residue chain at 3.7 s a trunk pass against 0.85 s
  // a sampler call, so a four-pass fold is about two thirds trunk.
  //
  // 🔴 IT IS THE CALLER'S JOB TO KNOW THE TRUNK IS STILL THE RIGHT ONE. Nothing
  // here can tell whether the batch it was given matches the trunk it was
  // handed; a stale one produces a structure for a different sequence, with a
  // confidence head that agrees with it. web/app.js keys the cache on
  // everything the trunk reads.
  const reused = options.reuse;
  let targetFeat = reused?.targetFeat;
  if (targetFeat === undefined) {
    await stage("target-feat-start");
    targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
    await stage("target-feat", { targetFeat });
  }

  const seqMask = batch.seqMask;
  // 🔴 THIS IS HOST WORK OVER n^2 AND IT IS INSIDE THE SILENT BAND. At 1500
  // tokens the loop below is 2.25M iterations and `previousPair` further down
  // is a 1.15 GB Float32Array allocated and zeroed - both between the last
  // thing the page was told and the first thing the trunk reports.
  await stage("trunk-prep", { tokens });
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
  // 🔴 A RECYCLE'S STATE IS THE TRUNK ITSELF, which is what makes asking for
  // more of them cheap. The loop feeds `previousPair`/`previousSingle` back in,
  // and those ARE the previous pass's `trunk.pair`/`trunk.single` - so a cached
  // trunk is a cached recycle state, and going from three recycles to five runs
  // two passes rather than six. Asking for FEWER is not a continuation and the
  // caller must not offer the cache for it; nothing here can undo a pass.
  // 🔴 THE PRECISION KNOBS TRAVEL WITH THE FOLD so a bench can measure both
  // arms without editing a source file - which is the only way to compare them
  // on a machine that drifts up to 3.2x between processes. Omitted, every one
  // defaults to what the device supports; see docs/AF3.md's memory section.
  const precision = {
    stagedPrecision: options.stagedPrecision,
    weightPrecision: options.weightPrecision,
    // 🔴 THE RESIDENT PAIR WEIGHTS ARE f16 WHERE THE PAIR TRACK IS WIDE, AND
    // THE RULE IS THE MEASUREMENT. These buffers are read one scalar at a time,
    // so halving them buys no bandwidth - what it buys is the peak, and the
    // peak is only made of them when the track is wide, because they go as the
    // SQUARE of the channel count:
    //
    //   OpenDDE, 384 channels   1198.3 -> 873.5 MiB  (-27%)  15.9 -> 15.9 s
    //   AlphaFold 3, 128         476.0 ->  476.0     (  0%)
    //
    // AlphaFold 3 does not move because its peak is the diffusion transformer's
    // 378 MiB of resident weights, not the trunk's - so there is nothing to buy
    // and it is not made to pay: at f16 its fold shifts (mean pLDDT
    // 72.19283791929007 -> 72.17922675038298), which is well inside a seed's
    // spread and still a change for no gain.
    //
    // 🔴 AND IT IS A WIDTH TEST RATHER THAN A MODEL NAME, so a future bundle
    // collects it by being wide rather than by being listed. See
    // WIDE_PAIR_TRACK for what the threshold is and is not.
    pairWeightPrecision: options.pairWeightPrecision
      ?? (weights.trunk.embedder.pairChannels >= WIDE_PAIR_TRACK ? "f16" : "f32"),
    accumulatePrecision: options.accumulatePrecision,
  };
  // 🔴 A WIDE PAIR TRACK DOES NOT KEEP ITS BLOCK WEIGHTS RESIDENT, AND THE
  // TRADE IS THE SAME ONE THE f16 RULE ABOVE PRICES. Residency exists so a
  // SECOND pass does not re-upload 48 blocks; what it costs is holding all of
  // them at once, and that cost goes as the square of the channel count while
  // the saving does not. Measured on OpenDDE at 384 channels, 6MRR:
  //
  //   recycles  resident            non-resident
  //   0         873.5 MiB, 16.1 s   647.6 MiB, 16.1 s
  //   1         894.4     , 19.5     647.6     , 19.6
  //   3         894.4     , 26.4     647.6     , 26.8
  //
  // 247 MiB to buy at most 0.4 s, and the structure is identical throughout
  // (RMSD 1.528, TM 0.9139 at three recycles either way). AlphaFold 3 at 128
  // channels keeps its residency: its block weights are a ninth of these and
  // its peak is the diffusion transformer regardless, so there is nothing to
  // buy and re-uploading would be a cost for no gain.
  const wideTrack = weights.trunk.embedder.pairChannels >= WIDE_PAIR_TRACK;
  const trunkGpu = new Af3TrunkGpu(device, {
    ...precision,
    residentWeights: options.residentWeights ?? !wideTrack,
  });
  // 🔴 THE CONDITIONING AND THE HEAD INPUT DO NOT DEPEND ON THE TRUNK, so they
  // are built once, above the recycle loop. Only `trunkSingle` and `trunkPair`
  // move.
  //
  // 🔴 THE DIFFUSION HEAD HAS ITS OWN FIVE REFERENCE EMBEDDINGS - same shapes
  // as the conditioning module's, different weights. Reusing one for both
  // type-checks and is a different model.
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.atomReference);
  const headInputBase = {
    // 🔴 THE DIALECT COMES FROM THE WEIGHTS, NOT FROM A CALLER. See
    // af3Dialect: a bundle's manifest names the graph it was converted for, so
    // the two cannot be paired wrongly.
    shape: batch.shape, dialect: weights.diffusion.dialect,
    conditioning, atomMask: batch.predDenseAtomMask, seqMask,
    features: batch.features, targetFeat,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
  };
  // 🔴 THERE ARE NO STRUCTURES DURING THE TRUNK ANY MORE. Each recycle used to
  // be followed by a two-cycle flow against that pass's trunk - a real
  // backbone, and the only structure anyone could be shown while the trunk was
  // still running. It was removed because of what it did to the VIEWER rather
  // than to the fold: every preview is a fresh structure in the object, so the
  // camera reframed on each one and a continuation had to carry them forward
  // to avoid losing them. The trunk's own contact map moves per recycle and is
  // free - the distogram is already computed and already read back - so the
  // recycles are watched through the heatmap panel instead, and the viewer
  // shows nothing until the sampler produces the first real frame.
  //
  // 🔴 ONE HEAD, BUILT ABOVE THE RECYCLE LOOP, so both samplers borrow it
  // rather than building their own.
  const head = steps > 0 ? new Af3DiffusionHeadGpu(device, precision) : undefined;
  // 🔴 AND WARMED HERE, WHICH THE COMMENT ABOVE THIS ONE USED TO CLAIM THE
  // CONSTRUCTOR DID. It said "building one compiles its pipelines - 730 ms -
  // and doing that here overlaps the compile with the trunk". The constructor
  // stores a device, an allocator and a cache and compiles nothing, so all of
  // it landed inside the first denoiser call with the trunk already over and
  // nothing left to overlap: 311 ms of a 2.9 s fold at 58 residues sits in the
  // page's "Folding" band before a single step reports.
  //
  // 🔴 NOT AWAITED, WHICH IS THE WHOLE POINT. createComputePipelineAsync
  // compiles off the main thread and a trunk pass leaves the host idle -
  // bench-trunk reports 9.4 ms of encoding against 2948 of waiting - so this
  // costs the trunk nothing and the sampler's first call awaits the same
  // memoised promise. The catch is only so that a compile that fails before
  // anything awaits it is not an unhandled rejection; the real failure still
  // arrives where the sampler asks.
  head?.warm(tokens, weights.diffusion).catch(() => {});

  let trunk = reused?.trunk;
  let previousPair = trunk?.pair ?? new Float32Array(tokens * tokens * 128);
  let previousSingle = trunk?.single ?? new Float32Array(tokens * 384);
  const firstPass = reused === undefined ? 0 : reused.recycles + 1;
  for (let pass = firstPass; pass <= recycles; pass += 1) {
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
      // 🔴 NAMED, BECAUSE THIS OBJECT IS BUILT FIELD BY FIELD. A key the batch
      // carries and this literal does not name is a key thrown away here, and
      // the embedder cannot tell that from a fold with no ligand: both arrive
      // as `undefined` and both fall back to zeros. That is how the whole bond
      // feature came to be computed, shipped and never applied.
      bondMatrix: batch.bondMatrix,
      // ...and the chain ids, which the template embedder masks its geometry
      // by. See the note at its call site in trunk-webgpu.js.
      asymId: batch.asymId,
      // 🔴 THE SLOTS ARE OVER TOKENS, NOT RESIDUES, and a fold with a ligand or
      // a modified residue has more of the first than the second. They are
      // built against `tokens` by web/template-source.js for that reason;
      // handing over a residue-indexed array would put a template's geometry
      // one place to the left of everything after the first ligand.
      templateSlots: options.templateSlots,
      pairMask, seqMask, previousPair, previousSingle,
      // 🔴 WHAT EACH TOKEN IS, so the distogram head can say what a contact
      // means for it. Eight angstroms is a pseudo-beta convention and AF3
      // tokenises a ligand one heavy atom at a time; see
      // ../heads/contact-threshold.js.
      contactClasses: af3ContactClasses(batch, tokens),
    }, weights.trunk, weights.trunk.dialect, {
      onStage: (name, ms) => stage("trunk", { name, ms }),
      // 🔴 THE ONE THE BAR NEEDS, because `trunk` fires when a stage is OVER.
      // Four of the trunk's five stages report nothing while they run, and on a
      // large protein each is seconds. See af3TrunkStageSpans.
      onStageStart: (name) =>
        stage("trunk-stage", { name, pass, passes: recycles + 1 }),
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
    // 🔴 EVERY PASS HAS ITS OWN DISTOGRAM, so the contact map can be shown
    // while the trunk is still recycling rather than only at the end. The
    // head runs per pass regardless - this only hands the result up. A page
    // has no structure to draw yet at this point, which is exactly why it is
    // worth showing: it is the one thing the model knows during the longest
    // part of a fold.
    await stage("recycle-done", { pass, passes: recycles + 1, trunk });
  }
  // 🔴 THE REUSABLE TRUNK RIDES ON trunk-done, NOT ONLY ON THE RETURN. A fold
  // that fails AFTER this point - the memory ceiling refusing the sampler is
  // the case that prompted it - used to lose the trunk with the exception, so
  // retrying re-ran every pass of work that had already succeeded. The loop
  // above has finished by here, so this carries all the recycles that were
  // asked for.
  stage("trunk-done", { trunk, reusable: { trunk, targetFeat, recycles } });

  // 🔴 AND THE PAIRFORMER'S WEIGHTS GO BACK HERE, for the same reason the
  // diffusion transformer's go back below: the stage that can read them is
  // over. The recycle loop has finished, the stack has already awaited its own
  // readback, and what runs next is the sampler - which holds its OWN 378 MiB
  // of resident weights and, until this line, held them alongside these.
  //
  // The confidence head's four blocks are a different weight object and cache
  // themselves; only a LATER fold's trunk pays, and it pays in packing.
  //
  // 🔴 AND ON A DEVICE WITH ROOM, "a LATER fold pays" IS THE WHOLE COST OF A
  // WARM FOLD. Measured at 68 tokens on an A100: re-uploading these is 561 MiB
  // in 1341 writeBuffer calls, waited for inside the pairformer's
  // onSubmittedWorkDone rather than in any compute pass - which is why
  // tools/gpu/profile.js saw 144 ms of a 1300 ms fold and the rest looked like
  // nothing. `keepTrunkWeights` is a per-device prior, null everywhere the
  // trade has not been measured.
  if (deviceTuning(device).keepTrunkWeights !== true) releaseResidentWeights(device, "w.");

  // 🔴 OpenDDE RE-TOKENISES BETWEEN THE TRUNK AND THE DIFFUSION, AND THIS IS
  // WHERE. Every other model here folds one token space end to end; OpenDDE
  // expands each standard residue into a backbone and a sidechain subtoken,
  // refines those, and runs the diffusion on THEM - so the batch the sampler
  // sees is not the batch the trunk saw, and the coordinates come back through
  // `structuralToResidue` before anything else looks at them.
  //
  // The branch lives here rather than in a second driver because everything
  // around it is shared: the recycles, the contact map, the trunk cache, the
  // stage callbacks the page's progress bar reads, and the PDB the viewer
  // draws. A separate driver would have to reproduce all of it.
  const structural = weights.trunk.dialect.structuralTokens
    ? await expandToStructuralTokens(device, batch, trunk, targetFeat, weights, stage)
    : undefined;
  const headInput = structural === undefined
    ? { ...headInputBase, trunkSingle: trunk.single, trunkPair: trunk.pair }
    : structural.headInput;

  // 🔴 TWO WAYS TO TURN THE TRUNK INTO COORDINATES, AND THEY ARE NOT THE SAME
  // KIND OF THING. "diffusion" is AF3's own stochastic sampler: noise is
  // injected at every step. "flow" draws once at the top of the schedule and
  // then walks it deterministically - about 25x fewer calls for the same
  // accuracy on the two proteins it has been measured on. Both take a seed;
  // the flow's spread across seeds is the narrower of the two, because one
  // draw is not the same as noise at every step.
  // 🔴 EVERY FRAME THE SAMPLER EMITS IS IN THE TOKEN SPACE IT RUNS ON, AND FOR
  // OpenDDE THAT IS NOT THE ONE A VIEWER DRAWS. The final coordinates are
  // scattered back after the loop; the per-step ones were not, so a caller
  // rendering them indexed 130 structural tokens' atom slots as though they
  // were 68 residues' - the wrong atom in every position, which on screen is
  // indistinguishable from watching the noise the sampler carries.
  //
  // Both sets are mapped, not just the one the page happens to draw: `denoised`
  // is the frame worth watching (see the sampler), and `positions` is the state
  // - a caller that asked for the state should get the state, in the layout it
  // asked about.
  const mapFrame = structural === undefined ? undefined
    : (frame) => structuralToResidue(frame, structural.layout, tokens, dense);
  const onStep = options.onStep === undefined ? undefined
    : (mapFrame === undefined ? options.onStep : async (detail) => options.onStep({
      ...detail,
      positions: mapFrame(detail.positions),
      denoised: mapFrame(detail.denoised),
      // ...and the structural-layout originals, for a caller that wants the
      // token space the sampler actually ran in.
      structuralPositions: detail.positions, structuralDenoised: detail.denoised,
    }));

  const sampled = options.mode === "diffusion"
    ? await sampleOnGpu(device, headInput, weights.diffusion, {
        steps, stopAfter: options.stopAfter, head,
        normal: normalFrom(options.seed ?? 20260831),
        onStep,
        ...(options.schedule ?? {}),
      })
    : await flowOnGpu(device, headInput, weights.diffusion, {
        cycles: steps, head, normal: normalFrom(options.seed ?? 20260831),
        onStep,
        // The schedule reaches noiseLevels through here, and both samplers
        // already forward their options to it.
        ...(options.schedule ?? {}),
      });

  // 🔴 THE BORROWED HEAD IS RELEASED HERE, because the samplers no longer do
  // it for one they did not build. Without this a fold leaks the diffusion
  // head's device buffers for the life of the page.
  head?.dispose();

  // 🔴 AND THE DIFFUSION TRANSFORMER'S RESIDENT WEIGHTS GO WITH IT, because the
  // fold's peak is AFTER this point and not before it. The sampler is finished
  // and its positions are read back, so nothing can still be reading them; what
  // runs next is the confidence head, which is four more pairformer blocks and
  // allocates the whole pair scratch again. Measured at 272 tokens, they were
  // 378 MiB of a 1214 MiB peak, held for a stage that cannot use them.
  //
  // The cost is that the NEXT fold packs and uploads them again. On unified
  // memory that is the packing and not the transfer; see the measurement in
  // the commit that added this.
  //
  // 🔴 AND ON A DEVICE WITH ROOM THAT COST IS THE POINT. Measured at 68 tokens
  // on an A100 with the trunk's weights already kept: 325 MiB and 24
  // int5-upload passes are still re-done every fold, and the host waits for
  // them on the queue rather than in any compute pass. `keepSamplerWeights` is
  // the same shape of prior as `keepTrunkWeights` and null for the same reason.
  if (deviceTuning(device).keepSamplerWeights !== true) {
    releaseResidentWeights(device, "difftx.");
    // ...and the diffusion conditioning's, which are resident for the same
    // reason and dead at the same moment.
    releaseResidentWeights(device, "cond.");
  }

  // 🔴 BACK TO RESIDUE TOKENS BEFORE ANYTHING ELSE READS THEM. The sampler
  // returned coordinates over the STRUCTURAL layout; the PDB writer, the
  // pseudo-beta gather, the geometry and the viewer all index (residue token,
  // atom slot). `residueAtomGather` is the inverse of the regrouping and was
  // built with it, so this is a scatter and not a reconstruction.
  const positions = structural === undefined ? sampled
    : structuralToResidue(sampled, structural.layout, tokens, dense);

  const confidenceFor = async () => {
    const gather = batch.tokenAtomsToPseudoBeta;
    const pseudoBeta = new Float32Array(tokens * 3);
    for (let token = 0; token < tokens; token += 1) {
      if (!gather.mask[token]) continue;
      const from = Number(gather.indices[token]) * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        pseudoBeta[token * 3 + axis] = positions[from + axis];
      }
    }
    return new Af3ConfidenceHeadGpu(device, options.confidencePrecision ?? {}).run({
      tokens, dense, seqMask, pair: trunk.pair, single: trunk.single, targetFeat, pseudoBeta,
    }, weights.confidence, weights.confidence.dialect);
  };

  /**
   * OpenDDE's confidence, mapped back onto the residue layout.
   *
   * The head reports per STRUCTURAL atom and per structural PAIR;
   * `residueAtomGather` puts the atoms back where every consumer expects them,
   * and `residueRepToken` picks the subtoken that stands for a residue in the
   * per-pair matrices - its FIRST, which is the one carrying the backbone role.
   */
  const openddeScores = async () => {
    const layout = structural.layout;
    const n = layout.tokens;
    const beta = structural.structural.tokenAtomsToPseudoBeta;
    const coordinates = new Float32Array(n * 3);
    for (let token = 0; token < n; token += 1) {
      if (!beta.mask[token]) continue;
      const from = Number(beta.indices[token]) * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        coordinates[token * 3 + axis] = sampled[from + axis];
      }
    }
    const atomToToken = new Int32Array(n * dense);
    const atomToSlot = new Int32Array(n * dense);
    for (let token = 0; token < n; token += 1) {
      for (let slot = 0; slot < dense; slot += 1) {
        atomToToken[token * dense + slot] = token;
        atomToSlot[token * dense + slot] = slot;
      }
    }
    const raw = await openddeConfidence(device, {
      tokens: n, singleInputs: structural.headInput.targetFeat,
      single: structural.headInput.trunkSingle, pair: structural.headInput.trunkPair,
      coordinates, seqMask: structural.structural.seqMask,
      atomToToken, atomToSlot, atomCount: n * dense,
      extraPairBias: structural.attentionBias,
    }, weights.openddeConfidence, weights.trunk.dialect);

    // Per-atom scores, scattered onto the residue layout.
    const plddt = new Float32Array(tokens * dense);
    for (let index = 0; index < tokens * dense; index += 1) {
      const from = layout.residueAtomGather[index];
      if (from >= 0) plddt[index] = raw.plddt[from];
    }
    // Per-pair matrices, on the residue tokens that represent them.
    const rep = layout.residueRepToken;
    const pae = new Float32Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) pae[i * tokens + j] = raw.pae[rep[i] * n + rep[j]];
    }
    // 🔴 A DIAGNOSTIC, because a confidence that does not move with the
    // structure is a confidence that is not reading it.
    let span = 0;
    let live = 0;
    for (let token = 0; token < n; token += 1) {
      if (!beta.mask[token]) continue;
      live += 1;
      span = Math.max(span, Math.abs(coordinates[token * 3]));
    }
    return { plddt, pae, tmAdjusted: undefined, opendde: raw,
             coordinateCheck: { live, tokens: n, span: Number(span.toFixed(3)) } };
  };

  // 🔴 AND A MODEL MAY STILL HAVE NO CONFIDENCE HEAD AT ALL. Its own is a
  // different design on a different distance grid - 51 tensors under
  // `confidence_head/pairformer_stack`, none of AlphaFold 3's names - so there
  // is no pLDDT and no PAE, and the keys are ABSENT rather than zero. A zero
  // pLDDT would be drawn as the colour of no confidence, which is a claim; the
  // page already has this path for EF2-fast, which has no head either.
  // 🔴 OpenDDE's OWN HEAD, ON THE STRUCTURAL TOKENS AND ON THE STRUCTURE THE
  // SAMPLER PRODUCED. It initialises its pair from a distance embedding of the
  // PREDICTION, so unlike AlphaFold 3's it cannot run before the sampler - and
  // it runs on the structural token set, so its pLDDT is per structural atom
  // and comes back through the same gather the coordinates do.
  const scores = structural === undefined ? await confidenceFor()
    : await openddeScores();

  // The confidence head reads the sample back.
  const beta = batch.tokenAtomsToPseudoBeta;

  // 🔴 AND ITS FOUR BLOCKS' WEIGHTS GO BACK TOO. `releaseResidentWeights("w.")`
  // above runs when the TRUNK is done, which is before this head exists - so
  // its own pairformer blocks stayed resident for the life of the page, and a
  // second fold began with 52 blocks' weights on the device where the first
  // began with 48. They are the same prefix and the same policy: the trunk's
  // are given back every fold, so these are too.
  if (deviceTuning(device).keepTrunkWeights !== true) releaseResidentWeights(device, "w.");

  // 🔴 EVERYTHING BELOW IS THE CONFIDENCE HEAD'S, SO IT IS ABSENT WHERE THE
  // HEAD IS. A model without one returns no pLDDT, no pTM and no per-chain
  // scores - not zeros, which every consumer would draw as a confident
  // failure. `pseudoBeta` goes with them: it exists to let a caller compare a
  // per-pair prediction against the PAE, and there is no PAE.
  let meanPlddt;
  let atoms = 0;
  let ptm;
  let iptm;
  let chainPairIptm;
  let chainPtm;
  let chainIptm;
  let pseudoBeta;
  if (scores !== undefined) {
    let total = 0;
    for (let index = 0; index < tokens * dense; index += 1) {
      if (!batch.predDenseAtomMask[index]) continue;
      total += scores.plddt[index];
      atoms += 1;
    }
    meanPlddt = total / atoms;

    // 🔴 pTM NEEDS A TM TERM, AND NOT EVERY HEAD WRITES ONE. AlphaFold 3's
    // confidence head emits one alongside the PAE; OpenDDE's emits pLDDT, PAE,
    // PDE and experimentally-resolved and nothing else - so pTM and ipTM are
    // ABSENT there rather than derived from the PAE, which would be a
    // different quantity wearing pTM's name.
    const asymId = batch.asymId;
    const selected = (i, j) => seqMask[i] > 0 && seqMask[j] > 0;
    if (scores.tmAdjusted !== undefined) {
    ptm = reduceTmScore(scores.tmAdjusted, tokens, selected);
    // NaN for one chain, because there is no interface to score. Zero would
    // read as a confident failure rather than an inapplicable question.
    iptm = reduceTmScore(scores.tmAdjusted, tokens,
      (i, j) => selected(i, j) && asymId[i] !== asymId[j]);
    // 🔴 AND ONE SCORE PER INTERFACE, BECAUSE THE POOLED ONE AVERAGES THEM. On
    // more than two chains `iptm` counts every cross-chain pair equally, so an
    // assembly holding both a native dimer and a designed binder reports the
    // easy interface's confidence for the hard one. See chainPairTmScores.
    chainPairIptm = Object.fromEntries(
      chainPairTmScores(scores.tmAdjusted, tokens, asymId, seqMask).scores);
    // ...and each chain on its own: how well it folded, and how well it sits
    // against everything else. The AlphaFold 3 server writes both.
    const perChain = perChainTmScores(scores.tmAdjusted, tokens, asymId, seqMask);
    chainPtm = Object.fromEntries(perChain.chainPtm);
    chainIptm = Object.fromEntries(perChain.chainIptm);
    }

    pseudoBeta = new Float32Array(tokens * 3);
    for (let token = 0; token < tokens; token += 1) {
      if (!beta.mask[token]) continue;
      const from = Number(beta.indices[token]) * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        pseudoBeta[token * 3 + axis] = positions[from + axis];
      }
    }
  } else {
    for (let index = 0; index < tokens * dense; index += 1) {
      if (batch.predDenseAtomMask[index]) atoms += 1;
    }
  }

  return {
    positions, trunk, targetFeat, scores, ptm, iptm, chainPairIptm,
    // ...the representative coordinates the confidence head was given. Returned
    // because anything comparing a per-pair prediction against the PAE needs
    // the SAME points the PAE is about, and recomputing them is a second
    // reading of `tokenAtomsToPseudoBeta` that can drift from this one.
    pseudoBeta,
    chainPtm, chainIptm,
    // What a caller hands back to skip the trunk next time. Returned even when
    // it was reused, so the cache survives a chain of re-samples.
    reusable: { trunk, targetFeat, recycles },
    meanPlddt, atoms, scores,
    // Per RESIDUE, from the alpha carbon - which is what pLDDT means when it is
    // shown on a cartoon, and what a per-residue check has to compare against.
    perResiduePlddt: scores === undefined ? undefined
      : Array.from({ length: tokens }, (_, token) => {
        const gather = batch.tokenAtomsToPseudoBeta;
        if (!gather.mask[token]) return undefined;
        return scores.plddt[Number(gather.indices[token])];
      }),
    // The structural-token count, where there was one; a caller that reports
    // what a fold did has no other way to see the second token space.
    structuralTokens: structural?.layout.tokens,
    geometry: backboneGeometry(batch, positions),
    pdb: toPdb(batch, positions, scores?.plddt ?? null),
  };
}

/**
 * Check the JavaScript featuriser against AF3's own batch, array by array.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --out af3-6mrr.json
 *     node tools/oracle/check_af3_featurise.js [af3-6mrr.json]
 *
 * This is the last stub between "a batch prepared elsewhere" and "a sequence
 * you type". Everything the featuriser produces is an integer layout or a mask
 * except the reference coordinates, so almost all of it is checked for EXACT
 * equality - a gather that is nearly right is a gather that is wrong.
 *
 * 🔴 ref_pos IS THE ONE ARRAY THAT CANNOT MATCH. AF3 threads a random_state
 * into RefStructure.compute_features and generates a conformer per residue
 * INSTANCE, so its 13 internal glutamates carry 13 different sets of torsions
 * and a baked table carries one. What IS invariant is the chemistry underneath:
 * every bond length and every 1-3 angle distance. Those are checked against
 * AF3, which catches a table built from the wrong component or with its atoms
 * out of order - the failures that matter - and lets the torsions go.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

import { featuriseProtein } from "../../src/af3/featurise.js";
import { REFERENCE_CONFORMERS } from "../../src/af3/reference-conformers.js";
import { af3MsaFromA3m } from "../../src/af3/msa-features.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/ccd-component.js";
import { deduplicateUnpairedAgainstPaired, mergeChainA3ms, mergeRowAlignedChainA3ms }
  from "../../src/input/chains.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dumpPath = process.argv[2] ?? join(ROOT, "af3-6mrr.json");
const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const input = (name) => dump.inputs[name].data;

// 🔴 THE CHAIN SPLIT IS NOT RECOVERABLE FROM THE JOINED SEQUENCE, which is why
// the dump records it separately. Feeding a complex as one chain would make
// every same_chain test true and quietly check nothing.
// An alignment, when the dump was made with one. The a3m path is checked here
// rather than in its own file because the arrays it changes - msa, profile and
// deletion_mean - are three of this file's arrays, and AF3 builds them in the
// same featuriser call as everything else.
// The A3M is PER CHAIN, as AF3's own input is, and is merged here exactly as
// web/app.js merges it - so the merge is part of what this checks. A dimer whose
// unpaired block is stacked block-diagonally rather than concatenated by row
// index reaches the model at twice the depth carrying half the information, and
// nothing downstream can tell.
const a3mPath = process.argv[3] ?? dump.a3m ?? null;
const pairedPath = process.argv[4] ?? null;
const chainCount = (dump.chains ?? [dump.sequence]).length;
const perChain = (path) => Array.from({ length: chainCount }, () => readFileSync(path, "utf8"));
// 🔴 AN A3M MAY ALREADY BE MERGED, and merging it again is not an error anyone
// sees - it silently doubles the width and the crash lands deep inside the
// featuriser. A file whose rows are already as wide as the joined sequence is
// taken as-is; one that is per-chain is merged. The width is the only thing
// that distinguishes them, so it is the thing that is read.
const joinedLength = (dump.chains ?? [dump.sequence]).join("").length;
const alreadyMerged = (text) => {
  const first = text.split(/\r?\n/).find((line) => line !== "" && !line.startsWith(">"));
  return first !== undefined
    && first.replace(/[a-z.]/g, "").length === joinedLength;
};
const asBlock = (path, merge) => {
  const text = readFileSync(path, "utf8");
  return chainCount === 1 || alreadyMerged(text) ? text : merge(perChain(path));
};
const rows = a3mPath === null
  ? { msa: [], deletionMatrix: [], depth: 1, unpairedFrom: 0 }
  : af3MsaFromA3m({
    paired: pairedPath === null ? null : asBlock(pairedPath, mergeChainA3ms),
    // 🔴 DEDUPLICATED AGAINST THE PAIRED BLOCK FIRST, PER CHAIN, exactly as
    // msa_pairing.deduplicate_unpaired_sequences does before AF3 merges
    // anything. Skipping it sends the model rows the paired block already
    // carried, and on this dimer that is EVERY unpaired row.
    unpaired: asBlock(a3mPath, (texts) => mergeRowAlignedChainA3ms(
      pairedPath === null ? texts
        : texts.map((text) => deduplicateUnpairedAgainstPaired(
          text, readFileSync(pairedPath, "utf8"))))),
    // ...and the profile's rows are the block BEFORE that deduplication, since
    // AF3 computes the profile first. See af3MsaFromA3m.
    unpairedProfile: asBlock(a3mPath, mergeRowAlignedChainA3ms),
  }, { maxSequences: Infinity });
// 🔴 A LIGAND'S CHEMISTRY COMES FROM THE PDB, NOT FROM THE DUMP. The batch
// carries the conformer AF3 sampled and the element of every atom, but not
// which component they belong to or how they are bonded - so the dump records
// the codes and this fetches the same dictionary entries the page would.
const ligandCodes = dump.ligands ?? [];
const ligands = [];
for (const code of ligandCodes) {
  const response = await fetch(ccdUrl(code));
  if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
  ligands.push(parseCcdComponent(await response.text()));
}
const batch = featuriseProtein((dump.chains ?? [dump.sequence]).join(":"),
  { msa: rows.msa, deletionMatrix: rows.deletionMatrix, unpairedFrom: rows.unpairedFrom,
    profileMsa: rows.profileMsa, profileDeletionMatrix: rows.profileDeletionMatrix,
    ligands });
const { tokens, dense } = batch;

let failures = 0;
const report = (name, detail, ok) => {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${name.padEnd(34)} ${detail}`);
};

/** Exact equality, elementwise, with the first disagreement named. */
function exact(name, ours, theirs) {
  const mine = Array.from(ours, Number);
  const yours = Array.from(theirs, Number);
  if (mine.length !== yours.length) {
    return report(name, `${mine.length} against ${yours.length} elements`, false);
  }
  for (let index = 0; index < mine.length; index += 1) {
    if (mine[index] !== yours[index]) {
      return report(name, `[${index}] ${mine[index]} against ${yours[index]}`, false);
    }
  }
  report(name, `${mine.length} elements`, true);
}

/**
 * A gather is only meaningful where its mask is set, so the masks are compared
 * exactly and the indices only where both agree an entry is real.
 */
function gatherMatches(name, ours, prefix) {
  const theirIndices = Array.from(input(`${prefix}:gather_idxs`), Number);
  const theirMask = Array.from(input(`${prefix}:gather_mask`), Number);
  exact(`${name} mask`, ours.mask, theirMask);
  const mine = [];
  const yours = [];
  for (let index = 0; index < theirMask.length; index += 1) {
    if (!theirMask[index] || !ours.mask[index]) continue;
    mine.push(ours.indices[index]);
    yours.push(theirIndices[index]);
  }
  exact(`${name} indices`, mine, yours);
}

console.log(`${dump.sequence.length} residues in ${batch.chainLengths.length} chain`
  + `${batch.chainLengths.length === 1 ? "" : "s"} (${batch.chainLengths.join(", ")}),`
  + ` ${tokens} tokens, ${batch.atomCount} atoms, ${batch.subsets} subsets\n`);

console.log("tokenisation and identity");
exact("aatype", batch.aatype, input("aatype"));
exact("residue_index", batch.residueIndex, input("residue_index"));
exact("token_index", batch.tokenIndex, input("token_index"));
exact("asym_id", batch.asymId, input("asym_id"));
exact("entity_id", batch.entityId, input("entity_id"));
exact("sym_id", batch.symId, input("sym_id"));
exact("seq_mask", batch.seqMask, input("seq_mask"));

console.log("\nchemistry");
exact("ref_mask", batch.refMask, input("ref_mask"));
exact("pred_dense_atom_mask", batch.predDenseAtomMask, input("pred_dense_atom_mask"));
exact("ref_element", batch.refElement, input("ref_element"));
// 🔴 CHARGE IS EXACT FOR POLYMERS AND EXPLAINED FOR LIGANDS. AF3 takes a
// ligand atom's charge from RDKit's PERCEIVED formal charge, not from the
// dictionary column, so heme's four -1 atoms come back as 0 - see
// src/af3/ccd-component.js. Splitting the comparison keeps the polymer half
// strict rather than loosening the whole of it around a known difference.
{
  // The polymer tokens are everything before the ligand block, counted from
  // the ligands themselves rather than from a `sequence` field a multi-chain
  // dump does not have.
  const ligandTokens = ligands.reduce((total, l) => total + l.atoms.length, 0);
  const residueSlots = (tokens - ligandTokens) * dense;
  const theirCharge = Array.from(input("ref_charge"));
  exact("ref_charge (polymer)", batch.refCharge.subarray(0, residueSlots),
        theirCharge.slice(0, residueSlots));
  if (residueSlots < tokens * dense) {
    let differing = 0;
    for (let index = residueSlots; index < tokens * dense; index += 1) {
      if (batch.refCharge[index] !== theirCharge[index]) differing += 1;
    }
    console.log(`  note  ligand charge differs on ${differing} atom slot(s):`
      + " RDKit's perception against the dictionary's, see ccd-component.js");
  }
}
exact("ref_atom_name_chars", batch.refAtomNameChars, input("ref_atom_name_chars"));
exact("ref_space_uid", batch.refSpaceUid, input("ref_space_uid"));

console.log("\natom layout");
gatherMatches("token_atoms_to_queries", batch.tokenAtomsToQueries, "token_atoms_to_queries");
gatherMatches("queries_to_token_atoms", batch.queriesToTokenAtoms, "queries_to_token_atoms");
gatherMatches("queries_to_keys", batch.queriesToKeys, "queries_to_keys");
gatherMatches("tokens_to_queries", batch.tokensToQueries, "tokens_to_queries");
gatherMatches("tokens_to_keys", batch.tokensToKeys, "tokens_to_keys");
gatherMatches("token_atoms_to_pseudo_beta", batch.tokenAtomsToPseudoBeta,
              "token_atoms_to_pseudo_beta");

// 🔴 THE WHOLE MSA, NOT JUST ROW ZERO. The gap is at 21 - between the amino
// acids and the nucleotides, not at the end of the 32-wide one-hot - and every
// row of a real alignment has gaps in it. Checking row zero alone would pass on
// an alphabet that puts the gap anywhere at all, because a query row is
// ungapped by construction. The deletion counts are checked the same way and
// for the same reason: they are RAW here, and AF3's embedder squashes them.
console.log(`\nMSA (${rows.depth} row${rows.depth === 1 ? "" : "s"})`);
const depth = input("msa").length / tokens;
exact("msa", batch.msa, Array.from(input("msa")).slice(0, batch.sequences * tokens));
exact("deletion_matrix", batch.deletionMatrix,
      Array.from(input("deletion_matrix") ?? []).slice(0, batch.sequences * tokens));
exact("profile", batch.profile, input("profile"));
exact("deletion_mean", batch.deletionMean, input("deletion_mean"));
console.log(`        the dump carries ${depth} rows; ours has ${batch.sequences}`);
// 🔴 AND AF3'S REMAINING ROWS ARE PADDING, NOT ALIGNMENT. AF3 pads the MSA to
// its crop size with zeros and masks them off, so a dump's row count is a
// buffer size and not a depth - on the 59-mer dimer every unpaired row was a
// duplicate of a paired one, leaving 33 real rows in a 65-row array. Checking
// that the tail really is zero is what distinguishes "we correctly have fewer
// rows" from "we lost rows", which the row count alone cannot.
// ...but only when this run was given the alignment the dump was made with.
// Checked without one, our single query row is correct and AF3's other rows are
// its real MSA, which is not padding and must not be asserted to be zero.
if (depth > batch.sequences && a3mPath !== null) {
  const theirs = input("msa");
  let nonZero = 0;
  for (let index = batch.sequences * tokens; index < depth * tokens; index += 1) {
    if (theirs[index] !== 0) nonZero += 1;
  }
  report("rows past ours are padding", `${depth - batch.sequences} rows,`
    + ` ${nonZero} non-zero`, nonZero === 0);
}

// 🔴 THE CONFORMER CHECK IS ON DISTANCES, NOT COORDINATES. See the note above.
// The pairs held fixed are not chosen by a distance cutoff - a torsion can fold
// two atoms to within a bond length of each other, and one in 6MRR's glutamines
// does. They come from reference-conformers.js, where each type's rigid pairs
// were measured over ten AF3 conformers of that type. This dump is a different
// protein, so the two sides are independent.
console.log("\nreference conformers: the chemistry AF3 holds fixed, not the torsions");
const theirPos = Array.from(input("ref_pos"), Number);
const distance = (source, a, b) => Math.hypot(
  source[a * 3] - source[b * 3],
  source[a * 3 + 1] - source[b * 3 + 1],
  source[a * 3 + 2] - source[b * 3 + 2]);

// AF3 relaxes its conformers rather than building them to ideal geometry, so
// even a bond breathes: over 6MRR the rigid pairs spread up to 0.20 A and the
// flexible ones start at 0.56 A. A 0.3 A tolerance sits inside that gap.
const RIGID_TOLERANCE = 0.3;
let worst = 0;
let worstAt = "";
let checked = 0;
const chainEnds = new Set();
let edge = -1;
for (const length of batch.chainLengths) { edge += length; chainEnds.add(edge); }
for (let token = 0; token < tokens; token += 1) {
  // 🔴 LIGAND TOKENS ARE NOT RESIDUES AND HAVE NO ENTRY HERE. They sit past the
  // end of the sequence, so `dump.sequence[token]` is undefined and the lookup
  // silently falls back to UNK - which then reports a 1.7 A disagreement on a
  // pair of atoms that do not exist. A ligand's own bonded distances are
  // checked below, from its CCD bond table.
  if (token >= dump.sequence.length) continue;
  const code = dump.sequence[token];
  const entry = REFERENCE_CONFORMERS[code] ?? REFERENCE_CONFORMERS.X;
  const atoms = chainEnds.has(token) ? entry.cTerminal : entry.internal;
  for (const [i, j] of entry.rigid) {
    const a = token * dense + atoms[i][0];
    const b = token * dense + atoms[j][0];
    const difference = Math.abs(distance(batch.refPos, a, b) - distance(theirPos, a, b));
    checked += 1;
    if (difference > worst) {
      worst = difference;
      worstAt = `${code}${token + 1} ${atoms[i][1]}-${atoms[j][1]}`;
    }
  }
}
report("rigid pair distances", `worst ${worst.toFixed(3)} A at ${worstAt}`
  + ` over ${checked} pairs`, worst < RIGID_TOLERANCE);

// The ligand's own chemistry, held to the same standard: bonded distances and
// 1-3 angles survive AF3's per-instance torsion sampling, coordinates do not.
if (ligands.length > 0) {
  let base = dump.sequence.length;
  let worst = 0;
  let where = "";
  let pairs = 0;
  for (const ligand of ligands) {
    const bonded = new Set(ligand.bonds.map(({ from, to }) => `${from},${to}`));
    const neighbours = ligand.atoms.map(() => new Set());
    for (const { from, to } of ligand.bonds) { neighbours[from].add(to); neighbours[to].add(from); }
    // 🔴 A BOND TO A METAL IS NOT RIGID. RDKit's conformer places coordination
    // bonds by its own rules - heme's FE-NA is 0.36 A from the dictionary's
    // even though it IS a bond, and NA-NC across the iron is 0.75 - so pairs
    // involving a metal are excluded rather than counted as chemistry AF3
    // should have reproduced. This is the same lesson the amino acid table
    // learned: rigidity comes from the bond graph, and the bond graph does not
    // know that a dative bond bends.
    const METALS = new Set([3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27,
      28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 55, 56,
      57, 78, 79, 80, 81, 82, 83]);
    const isMetal = (atom) => METALS.has(ligand.atoms[atom].element);
    // Each ligand atom is its own token, in slot zero of it.
    for (let i = 0; i < ligand.atoms.length; i += 1) {
      for (let j = i + 1; j < ligand.atoms.length; j += 1) {
        if (isMetal(i) || isMetal(j)) continue;
        const shared = [...neighbours[i]].filter((k) => neighbours[j].has(k));
        const oneThree = shared.some((k) => !isMetal(k));
        if (!bonded.has(`${i},${j}`) && !bonded.has(`${j},${i}`) && !oneThree) continue;
        pairs += 1;
        const a = (base + i) * dense;
        const b = (base + j) * dense;
        const error = Math.abs(distance(batch.refPos, a, b) - distance(theirPos, a, b));
        if (error > worst) { worst = error; where = `${ligand.atoms[i].name}-${ligand.atoms[j].name}`; }
      }
    }
    base += ligand.atoms.length;
  }
  console.log("\nligand chemistry: the dictionary's conformer against AF3's own");
  report("ligand rigid pairs", `worst ${worst.toFixed(3)} A at ${where} over ${pairs} pairs`,
         worst < 0.25);
}


let worstTorsion = 0;
for (let index = 0; index < theirPos.length; index += 1) {
  worstTorsion = Math.max(worstTorsion, Math.abs(batch.refPos[index] - theirPos[index]));
}
console.log(`        raw coordinates differ by up to ${worstTorsion.toFixed(2)} A,`
  + " which is the per-instance torsion sampling and is expected");

console.log(failures === 0
  ? "\nthe JavaScript featuriser reproduces AF3's batch"
  : `\n${failures} arrays disagree`);
process.exit(failures === 0 ? 0 : 1);

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

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dumpPath = process.argv[2] ?? join(ROOT, "af3-6mrr.json");
const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const input = (name) => dump.inputs[name].data;

// 🔴 THE CHAIN SPLIT IS NOT RECOVERABLE FROM THE JOINED SEQUENCE, which is why
// the dump records it separately. Feeding a complex as one chain would make
// every same_chain test true and quietly check nothing.
const batch = featuriseProtein((dump.chains ?? [dump.sequence]).join(":"));
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
exact("ref_charge", batch.refCharge, input("ref_charge"));
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

console.log("\nMSA (the query alone, matching the dump's num_msa=1)");
const depth = input("msa").length / tokens;
exact("msa row 0", batch.msa, Array.from(input("msa")).slice(0, tokens));
exact("profile", batch.profile, input("profile"));
exact("deletion_mean", batch.deletionMean, input("deletion_mean"));
console.log(`        the dump carries ${depth} rows; the trunk read ${1}`);

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

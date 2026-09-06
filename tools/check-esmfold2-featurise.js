// Does LocalFold's featuriser produce the features ESMFold2 was handed?
//
//     node tools/check-esmfold2-featurise.js
//
// 🔴 EVERY DISCRETE FEATURE IS CHECKED EXACTLY, NOT TO A TOLERANCE. Elements,
// four-character atom names, the atom-to-token map, the reference-space ids,
// the atom mask, the formal charges, the language model's token ids, the
// molecule types, all five index features, the bond matrix and the 33-class
// residue one-hot are integers or zero-one, so "identical" is the only
// standard that means anything. The one feature that is NOT identical is
// `ref_pos`, and it cannot be: both models draw a fresh conformer per residue
// INSTANCE, so bond lengths and angles agree and torsions do not.
//
// 🔴 AND THE FEATURISER IS AF3's, WITH AN ADAPTER. src/esmfold2/featurise.js is
// three differences from src/af3/featurise.js - a ragged atom layout, two
// different residue alphabets, and no terminal atom - so what this really
// checks is that those three are the only ones. A second featuriser written
// from scratch would pass this and would still be a second place for a CCD
// component to be read wrongly.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { featuriseForEsmfold2, languageModelInput } from "../src/esmfold2/featurise.js";

const ROOT = new URL("..", import.meta.url).pathname;
const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const dumpPath = positional[0]
  ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40-lm.json");
const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const features = featuriseForEsmfold2(dump.sequence);
const recorded = dump.features;

let failures = 0;
const identical = (label, got, want) => {
  let differ = 0, first = -1;
  const length = Math.max(got.length, want.length);
  for (let i = 0; i < length; i += 1) {
    const wanted = typeof want[i] === "boolean" ? (want[i] ? 1 : 0) : want[i];
    if (got[i] !== wanted) { differ += 1; if (first < 0) first = i; }
  }
  if (got.length !== want.length) differ = Math.max(differ, 1);
  if (differ !== 0) failures += 1;
  console.log(`  ${label.padEnd(24)}${differ === 0 ? "identical"
    : `DIFFER on ${differ}, first at ${first}: ${got[first]} against ${want[first]}`}`);
};

console.log(`${dump.esmfold2}: ${dump.sequence.length} residues, `
  + `${features.atoms} atom slots, ${features.liveAtoms} live\n`);

identical("ref_element", features.refElement, recorded.ref_element.values);
identical("ref_atom_name_chars", features.refAtomNameChars,
          recorded.ref_atom_name_chars.values);
identical("ref_charge", features.refCharge, recorded.ref_charge.values);
identical("ref_space_uid", features.refSpaceUid, recorded.ref_space_uid.values);
identical("atom_to_token", features.atomToToken, recorded.atom_to_token.values);
identical("atom_attention_mask", features.mask, recorded.atom_attention_mask.values);
identical("mol_type", features.molType, recorded.mol_type.values);
identical("residue_index", features.residueIndex, recorded.residue_index.values);
identical("token_index", features.tokenIndex, recorded.token_index.values);
identical("asym_id", features.asymId, recorded.asym_id.values);
identical("entity_id", features.entityId, recorded.entity_id.values);
identical("sym_id", features.symId, recorded.sym_id.values);
identical("token_bonds", features.tokenBonds, recorded.token_bonds.values);
identical("input_ids", features.inputIds, recorded.input_ids.values);
identical("aatype", features.aatype,
          dump.block0.inputs_embedder.arguments.aatype.values);

// 🔴 ref_pos IS THE ONE THAT CANNOT MATCH, AND SAYING SO IS PART OF THE CHECK.
// A conformer is sampled per residue instance, so the two are the same molecule
// at different torsions. What IS comparable is the bonded geometry: the N-CA
// distance of the first residue, which no torsion can change.
const bond = (values, a, b) => Math.hypot(
  values[a * 3] - values[b * 3], values[a * 3 + 1] - values[b * 3 + 1],
  values[a * 3 + 2] - values[b * 3 + 2]);
const theirs = Float32Array.from(recorded.ref_pos.values);
const ours = bond(features.refPos, 0, 1);
const native = bond(theirs, 0, 1);
console.log(`\n  N-CA of residue 0        ${ours.toFixed(4)} A against `
  + `${native.toFixed(4)} - the same molecule, a different torsion`);
if (Math.abs(ours - native) > 0.05) {
  failures += 1;
  console.log("  ...which is too far apart to be a torsion");
}

// The language model's own input, which is a rearrangement rather than a feature.
const lm = languageModelInput(features);
console.log(`\n  language model           ${lm.ids.length} tokens over ${lm.chains} chain(s), `
  + `${lm.rows} residues`);
const expected = features.tokens + 2 * lm.chains;
if (lm.ids.length !== expected) {
  failures += 1;
  console.log(`  ...expected ${expected}: one row a residue plus a BOS and an EOS a chain`);
}

console.log(failures === 0
  ? "\nLocalFold builds the features ESMFold2 was handed"
  : `\n${failures} feature(s) differ`);
process.exit(failures === 0 ? 0 : 1);

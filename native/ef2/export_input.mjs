// One ESMFold2 input for the native CUDA port: the page's own features (src/esmfold2/featurise.js,
// what src/esmfold2/fold.js folds with), in native/af3's model.idx/model.bin format.
//
//   node --js-float16array native/ef2/export_input.mjs <out dir> --sequence=<SEQ>[:<SEQ>...]
//
// Entries (i int32, t float32):
//   per token   i residue_index token_index asym_id entity_id sym_id mol_type res_type input_ids
//               i distogram_atom_idx    t aatype [T, 33] profile [T, 33] deletion_mean token_bonds [T, T]
//   per atom    t ref_pos [A, 3] ref_charge atom_mask    i ref_element ref_atom_name_chars [A, 4]
//               i ref_space_uid atom_to_token
//   the tower   i lm/ids lm/sequence_id (BOS/EOS per chain) lm/token_to_row (-1: not a protein token)
//   m meta/tokens, meta/atoms, meta/lm_rows, meta/classes
// and pdb.template beside them: the page's PDB records, each atom's index where its coordinates go
import { writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { featuriseForEsmfold2, languageModelInput } from "../../src/esmfold2/featurise.js";
import { representativeAtoms } from "../../src/esmfold2/fold.js";
import { toDensePositions } from "../../src/esmfold2/featurise.js";
import { toPdb } from "../../src/af3/fold.js";

const args = process.argv.slice(2);
const out = args[0];
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const sequence = option("sequence", "").trim().toUpperCase();
if (!out || out.startsWith("--") || sequence === "") {
  console.error("usage: export_input.mjs <out dir> --sequence=<SEQ>[:<SEQ>...]"); process.exit(1);
}
const f = featuriseForEsmfold2(sequence);
const lm = languageModelInput(f);
const T = f.tokens, A = f.atoms;

const entries = [];
const int = (name, v) => entries.push(["i", name, v instanceof Int32Array ? v : Int32Array.from(v)]);
const flt = (name, v) => entries.push(["t", name, v instanceof Float32Array ? v : Float32Array.from(v)]);
int("residue_index", f.residueIndex); int("token_index", f.tokenIndex); int("asym_id", f.asymId);
int("entity_id", f.entityId); int("sym_id", f.symId); int("mol_type", f.molType);
int("res_type", f.residueType); int("input_ids", f.inputIds);
int("distogram_atom_idx", representativeAtoms(f, T));
flt("aatype", f.aatype); flt("profile", f.profile); flt("deletion_mean", f.deletionMean);
flt("token_bonds", f.tokenBonds);
flt("ref_pos", f.refPos); flt("ref_charge", f.refCharge); flt("atom_mask", f.mask);
int("ref_element", f.refElement); int("ref_atom_name_chars", f.refAtomNameChars);
int("ref_space_uid", f.refSpaceUid); int("atom_to_token", f.atomToToken);
int("lm/ids", lm.ids); int("lm/sequence_id", lm.sequenceId); int("lm/token_to_row", lm.tokenToRow);
entries.push(["m", "meta/tokens", T], ["m", "meta/atoms", A], ["m", "meta/lm_rows", lm.ids.length],
             ["m", "meta/classes", f.aatype.length / T]);

mkdirSync(out, { recursive: true });
const fd = openSync(`${out}/model.bin`, "w");
const lines = [];
let offset = 0;
for (const [kind, name, value] of entries) {
  if (kind === "m") { lines.push(`m ${name} ${value}`); continue; }
  lines.push(`${kind} ${name} ${offset} ${value.length}`);
  writeSync(fd, Buffer.from(value.buffer, value.byteOffset, value.byteLength));
  offset += value.length;
}
closeSync(fd);
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
// the structure's records, from the page's own writer (residue and atom names, chains, HETATM, CONECT),
// with each atom's INDEX where its coordinates go - x = index / 1000, y = index % 1000 - for the
// native fold to substitute (pdb.template)
{
  const coordinates = new Float32Array(A * 3);
  for (let atom = 0; atom < A; atom += 1) { coordinates[atom * 3] = Math.floor(atom / 1000); coordinates[atom * 3 + 1] = atom % 1000; }
  writeFileSync(`${out}/pdb.template`, toPdb(f.batch, toDensePositions(f, coordinates)) + "\n");
}
console.log(`${T} tokens, ${A} atoms (${f.liveAtoms} live), ${lm.ids.length} tower rows -> ${out}`);

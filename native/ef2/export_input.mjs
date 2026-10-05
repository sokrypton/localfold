// One ESMFold2 input for the native CUDA port: the page's own features (src/esmfold2/featurise.js,
// what src/esmfold2/fold.js folds with), in native/af3's model.idx/model.bin format.
//
//   node --js-float16array native/ef2/export_input.mjs <out dir> --sequence=<SEQ>[:<SEQ>...] [--kinds=protein,dna]
//        [--ligands=GOL,ATP] [--smiles=OCC(O)CO|...] [--modify=SEP@3[@chain]]   |   --job=<AF3 job.json>
//
// Entries (i int32, t float32):
//   per token   i residue_index token_index asym_id entity_id sym_id mol_type res_type input_ids
//               i distogram_atom_idx    t aatype [T, 33] profile [T, 33] deletion_mean token_bonds [T, T]
//   per atom    t ref_pos [A, 3] ref_charge atom_mask    i ref_element ref_atom_name_chars [A, 4]
//               i ref_space_uid atom_to_token
//   the tower   i lm/ids lm/sequence_id (BOS/EOS per chain) lm/token_to_row (-1: not a protein token)
//   m meta/tokens, meta/atoms, meta/lm_rows, meta/classes
// and pdb.template beside them: the page's PDB records, each atom's index where its coordinates go
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { featuriseForEsmfold2, languageModelInput } from "../../src/esmfold2/featurise.js";
import { representativeAtoms } from "../../src/esmfold2/fold.js";
import { toDensePositions } from "../../src/esmfold2/featurise.js";
import { toPdb } from "../../src/af3/fold.js";

const args = process.argv.slice(2);
const out = args[0];
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
let sequence = option("sequence", "").trim().toUpperCase();
if (!out || out.startsWith("--") || (sequence === "" && option("job", "") === "")) {
  console.error("usage: export_input.mjs <out dir> --sequence=<SEQ>[:<SEQ>...] [--kinds=protein,dna,...]"
    + " [--ligands=GOL,ATP] [--smiles=OCC(O)CO|...] [--modify=SEP@3[@chain]] | --job=<AF3 job.json>"); process.exit(1);
}
// what the page folds besides one protein chain, resolved as native/af3's exporter and the page do:
// --kinds (one per ":"-chain: protein, dna, rna), --ligands (CCD codes, from the RCSB), --smiles
// (built by src/chem), --modify (CODE@position[@chain], chain index from 0), or an AF3 job file read
// by the page's own reader (web/job-json.js, web/entities.js: its ligands, glycans, modified residues
// and bases, declared bonds and userCCD)
const { ccdUrl, parseCcdComponent, ligandChain } = await import("../../src/af3/featurise/ccd-component.js");
const { nameSmilesLigands, smilesComponent } = await import("../../src/chem/component.js");
let kinds = option("kinds", ""), jobRequest = null;
const userComponents = new Map();
if (option("job", "") !== "") {
  if (sequence !== "") throw new Error("--job and --sequence both name the input");
  const { jobFromJson } = await import("../../web/job-json.js");
  const { expandEntities } = await import("../../web/entities.js");
  const job = jobFromJson(readFileSync(option("job", ""), "utf8"));
  for (const note of job.notes) console.log(`job: ${note}`);
  jobRequest = expandEntities(job.entities);
  sequence = jobRequest.sequence;
  kinds = jobRequest.chainKinds.join(",");
  for (const block of (job.userCcd ?? "").split(/^(?=data_)/m).filter((b) => b.trim().startsWith("data_"))) {
    const component = parseCcdComponent(block);
    userComponents.set(component.code.toUpperCase(), component);
  }
}
const ccd = async (code) => {
  if (userComponents.has(code.toUpperCase())) return userComponents.get(code.toUpperCase());
  const response = await fetch(ccdUrl(code));
  if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
  return parseCcdComponent(await response.text());
};
const ligands = [], modifications = [];
for (const entry of jobRequest?.ligandCodes ?? []) {
  ligands.push(typeof entry === "string" ? await ccd(entry)
    : entry.codes ? ligandChain(await Promise.all(entry.codes.map(ccd)))
    : await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
}
for (const m of jobRequest?.modifications ?? []) modifications.push({ chain: m.chain, position: m.position, ...(await ccd(m.code)) });
for (const code of option("ligands", "").split(",").filter(Boolean)) ligands.push(await ccd(code));
const smiles = option("smiles", "").split("|").filter(Boolean);
const smilesNames = nameSmilesLigands(smiles);
for (let i = 0; i < smiles.length; i += 1) ligands.push(await smilesComponent(smiles[i], { code: smilesNames[i] }));
for (const spec of option("modify", "").split(",").filter(Boolean)) {
  const [code, at, chain] = spec.split("@");
  modifications.push({ chain: Number(chain ?? 0), position: Number(at), ...(await ccd(code)) });
}
const request = { sequence,
  ...(kinds === "" ? {} : { chainKinds: kinds.split(",") }),
  ...(ligands.length === 0 ? {} : { ligands }),
  ...(modifications.length === 0 ? {} : { modifications }),
  ...(jobRequest?.bonds === undefined ? {} : { bonds: jobRequest.bonds }) };
const f = featuriseForEsmfold2(request);
const lm = languageModelInput(f);
const T = f.tokens, A = f.atoms;

const entries = [];
const int = (name, v) => entries.push(["i", name, v instanceof Int32Array ? v : Int32Array.from(v)]);
const flt = (name, v) => entries.push(["t", name, v instanceof Float32Array ? v : Float32Array.from(v)]);
int("residue_index", f.residueIndex); int("token_index", f.tokenIndex); int("asym_id", f.asymId);
int("entity_id", f.entityId); int("sym_id", f.symId); int("mol_type", f.molType);
int("res_type", f.residueType); int("input_ids", f.inputIds);
// the page's contact rule (src/esmfold2/distogram-webgpu.js): per pair, how many distogram bins lie under
// its threshold - for the 128-bin distogram both published checkpoints carry, stamped, so the binary
// refuses them against any other
// (--fold-bundle=<dir>: the released models' 64-bin distogram is AlphaFold 3's grid, counted by AF3's rule)
{
  const { contactBinCountsByPair, contactBinCountsByPairBreaks } = await import("../../src/esmfold2/distogram-webgpu.js");
  const bundleArg = process.argv.slice(2).find((a) => a.startsWith("--fold-bundle="))?.slice(14);
  const bins = bundleArg ? JSON.parse(readFileSync(`${bundleArg}/manifest.json`, "utf8")).trunk?.distogramBins ?? 128 : 128;
  if (bins !== 128 && bins !== 64) throw new Error(`a ${bins}-bin distogram: only 128 (2-52) and 64 (AF3's) are known`);
  int("contact_bins", bins === 128 ? contactBinCountsByPair(f.molType, f.residueType, f.tokens, 128)
                                   : contactBinCountsByPairBreaks(f.molType, f.residueType, f.tokens));
  entries.push(["m", "meta/contactBinsFor", bins]);
}
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

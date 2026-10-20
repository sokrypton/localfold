// One BioEmu2 input for the native CUDA port: shared/bioemu2/featurise.js's features (the transcription of
// af3-any-model's pipeline/bioemu2_features.py), in cuda/af3's model.idx/model.bin format.
//
//   node cuda/bioemu2/export_input.mjs <out dir> --job=<AF3 job.json> | --sequence=<SEQ>[:<SEQ>...]
//        [--mode=structure|equilibrium] [--contacts=<contacts.json>]
//        [--propagate-from=<pdb|cif>] [--lag-ns=1] [--propagation-steps=1]
//        [--a3m=<merged.a3m> | --a3m=<a.a3m>,<b.a3m>,...] [--paired-a3m=...]
//
// An alignment as the page hands one to its featuriser (web/bioemu2-model.js bioemu2Features): ONE path is the merged
// A3M over the job's protein chains, split back a chain (shared/bioemu2/inputs.js splitA3mByChains) and given to every
// chain of that sequence; several paths are one a protein chain, in job order. A SMILES ligand's template is the
// page's too (inputs.js smilesTemplates). cuda/featurise's bioemu2-featurise is this file in C++, byte for byte.
//
// Entries (i int32, t float32), the vendor's own names (BioEmu2's atoms padded to a multiple of 32):
//   per token   i res_type residue_index asym_id entity_id sym_id token_center_atom_index token_pad_mask
//               i method_feature pocket_feature    t deletion_mean profile [T, 33]
//   per pair    t token_bonds contact_threshold    i contact_class (the contact_conditioning one-hot's index)
//   the MSA     i msa (an index, [S, T]) msa_paired has_deletion msa_mask    t deletion_value
//   per atom    t template_coords [N, 3] charge coords_cond [N, 3]    i element name_encoding [N, 4]
//               i token_idx_at_atom ref_space_uid atom_pad_mask atom_mask
//   propagation t x0 [N, 3] (the starting frame, uncentred)
//   steering    (--steering[=<stereo_chemistry.yaml>], for the edm_gd / edm_fk samplers) every table
//               shared/bioemu2/steering-tables.js builds, before potentials.js's hostTables (which depends on the
//               potentials, so the sampler applies it): i steer/<table> (a (k, n) table row-major), t steer/<table>,
//               and the bond and angle references as i steer/bond_ref.bits / steer/angle_ref.bits (each double's
//               low and high words: the angle bounds are taken from the DOUBLE in degrees).
//               The stereochemistry is CCP4 MON_LIB's (LGPL), fetched from BioEmu2's repository and never shipped.
//   m meta/tokens, meta/atoms (padded), meta/real_atoms, meta/msa_rows, meta/n_valid_rows, meta/mode (0 structure,
//     1 equilibrium), meta/dt, meta/is_dynamics_conditioned, meta/propagation_steps
// and beside them pdb.template (the structure's records in the JOB's chain order, each atom's index where its
// coordinates go - x = index / 1000, y = index % 1000 - as cuda/ef2's) and confidence.template.json (the output's
// tokens in job order, as BioEmu2 tokens, with their chain and residue ids).
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { featuriseBioemu2, bioemu2JobFromAf3Json, parseContacts, addPropagation, METHOD } from "../../shared/bioemu2/featurise.js";
import { ccdUrl, ELEMENT_SYMBOLS } from "../../shared/af3/featurise/ccd-component.js";
import { STANDARD_COMPONENTS } from "../../shared/af3/featurise/standard-components.js";
import { smilesTemplates as pageSmilesTemplates, splitA3mByChains } from "../../shared/bioemu2/inputs.js";
import { steeringTables, parseStereoTables, STEREO_URL, STEREO_COMMIT } from "../../shared/bioemu2/steering-tables.js";

const args = process.argv.slice(2);
const out = args[0];
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (!out || out.startsWith("--") || (option("job", "") === "" && option("sequence", "") === "")) {
  console.error("usage: export_input.mjs <out dir> --job=<job.json> | --sequence=<SEQ>[:<SEQ>...] [--mode=structure|equilibrium]"
    + " [--contacts=<json>] [--propagate-from=<pdb|cif>] [--lag-ns=1] [--propagation-steps=1]");
  process.exit(1);
}
let json;
if (option("job", "") !== "") json = JSON.parse(readFileSync(option("job", ""), "utf8"));
else {
  json = { sequences: option("sequence", "").split(":").map((s, i) => ({ protein: { id: String.fromCharCode(65 + i), sequence: s.toUpperCase() } })) };
}
const job = bioemu2JobFromAf3Json(json);
const mode = option("mode", "structure");
if (!(mode in METHOD)) throw new Error(`--mode=${mode}: structure or equilibrium`);

// ---- the components the job names, resolved before the (synchronous) featuriser runs: the job's userCCD, the
// fixtures, the per-component cache ~/.cache/localfold/ccd (what the native binaries keep), else the RCSB
const userComponents = new Map();
for (const block of (json.userCCD ?? "").split(/^(?=data_)/m).filter((b) => b.trim().startsWith("data_"))) {
  const code = block.match(/_chem_comp\.id\s+(\S+)/)?.[1]?.replace(/['"]/g, "").toUpperCase();
  if (code) userComponents.set(code, block);
}
const needed = new Set();
for (const chain of job.chains) {
  for (const m of chain.modifications ?? []) needed.add(m.code.toUpperCase());
  for (const code of chain.ccdCodes ?? []) needed.add(code.toUpperCase());
  if (chain.kind === "protein" && chain.sequence.includes("X")) needed.add("UNK");
  if (chain.kind === "rna" && chain.sequence.includes("N")) needed.add("N");
  if (chain.kind === "dna" && chain.sequence.includes("N")) needed.add("DN");
  if (chain.kind === "protein" && /[^ARNDCEQGHILKMFPSTWYV]/.test(chain.sequence)) needed.add("ALA");
}
const here = dirname(new URL(import.meta.url).pathname);
const cacheDir = join(homedir(), ".cache", "localfold", "ccd");
const texts = new Map();
for (const code of needed) {
  if (STANDARD_COMPONENTS[code]) continue;
  if (userComponents.has(code)) { texts.set(code, userComponents.get(code)); continue; }
  const fixture = join(here, "..", "..", "tools", "fixtures", "ccd", `${code}.cif`);
  const cached = join(cacheDir, `${code}.cif`);
  if (existsSync(fixture)) { texts.set(code, readFileSync(fixture, "utf8")); continue; }
  if (existsSync(cached)) { texts.set(code, readFileSync(cached, "utf8")); continue; }
  const response = await fetch(ccdUrl(code));
  if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
  const text = await response.text();
  mkdirSync(cacheDir, { recursive: true });
  writeFileSync(cached, text);
  texts.set(code, text);
}
const ccd = (code) => texts.get(code.toUpperCase()) ?? userComponents.get(code.toUpperCase());

// ---- a SMILES ligand's template: shared/chem's conformer (see the note in shared/bioemu2/featurise.js)
const smilesTemplates = {};
// (--smiles-from=<dump>: a tools/oracle/dump_bioemu2.py dump's own templates - af3-any-model's seeded RDKit draw - so
// the oracle compares one conformer on both sides)
if (option("smiles-from", "") !== "") {
  const { readBioemu2Dump, smilesTemplatesOf } = await import("../../tools/bioemu2-dump.js");
  Object.assign(smilesTemplates, smilesTemplatesOf(readBioemu2Dump(option("smiles-from", "")).stages));
}
if (Object.keys(smilesTemplates).length === 0) Object.assign(smilesTemplates, await pageSmilesTemplates(job));

// ---- the alignments, as the page hands them over (see the header)
const proteins = job.chains.filter((c) => c.kind === "protein");
const alignmentsOf = (arg) => {
  if (arg === "") return null;
  const paths = arg.split(",");
  if (paths.length === 1) return splitA3mByChains(readFileSync(paths[0], "utf8"), proteins.map((c) => c.sequence.length));
  if (paths.length !== proteins.length) {
    throw new Error(`${paths.length} alignments for ${proteins.length} protein chain(s): name one a chain, or one merged`);
  }
  return paths.map((path) => readFileSync(path, "utf8"));
};
const unpairedA3ms = alignmentsOf(option("a3m", ""));
const pairedA3ms = alignmentsOf(option("paired-a3m", ""));
if (unpairedA3ms !== null) {
  if (option("a3m", "").split(",").length === 1) {
    const bySequence = new Map(proteins.map((c, at) => [c.sequence.toUpperCase(),
      { unpaired: unpairedA3ms[at], paired: pairedA3ms?.[at] ?? null }]));
    for (const chain of proteins) {
      const found = bySequence.get(chain.sequence.toUpperCase());
      chain.unpairedMsa = found.unpaired;
      chain.pairedMsa = found.paired;
    }
  } else {
    proteins.forEach((chain, at) => { chain.unpairedMsa = unpairedA3ms[at]; chain.pairedMsa = pairedA3ms?.[at] ?? null; });
  }
}
const contacts = option("contacts", "") ? parseContacts(JSON.parse(readFileSync(option("contacts", ""), "utf8"))) : null;
let { features: f, atomKeys, tokens: T, atoms: realAtoms, atomsPadded: N, topology } =
  featuriseBioemu2(job, { ccd, mode, contacts, smilesTemplates });

// ---- propagation: the starting frame, every atom read by (chain, residue, name) from a PDB or an mmCIF
const propagateFrom = option("propagate-from", "");
const propagationSteps = Number(option("propagation-steps", "1"));
let x0 = null;
if (propagateFrom !== "") {
  if (mode !== "structure" && mode !== "equilibrium") throw new Error("unreachable");
  const text = readFileSync(propagateFrom, "utf8");
  const where = new Map();
  if (/^data_/m.test(text) && /_atom_site\./.test(text)) {
    const columns = [...text.matchAll(/^_atom_site\.(\S+)/gm)].map((m) => m[1]);
    const col = (name) => columns.indexOf(name);
    for (const line of text.split("\n")) {
      if (!/^(ATOM|HETATM)\s/.test(line)) continue;
      const v = line.trim().split(/\s+/);
      const chain = v[col("auth_asym_id") >= 0 ? col("auth_asym_id") : col("label_asym_id")];
      const seq = v[col("label_seq_id")] !== "." && col("label_seq_id") >= 0 ? v[col("label_seq_id")] : v[col("auth_seq_id")];
      const name = v[col("label_atom_id")].replace(/^"|"$/g, "");
      where.set(`${chain}|${Number(seq)}|${name}`, [Number(v[col("Cartn_x")]), Number(v[col("Cartn_y")]), Number(v[col("Cartn_z")])]);
    }
  } else {
    for (const line of text.split("\n")) {
      if (!/^(ATOM  |HETATM)/.test(line)) continue;
      where.set(`${line[21]}|${Number(line.slice(22, 26))}|${line.slice(12, 16).trim()}`,
                [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
    }
  }
  const missing = atomKeys.filter((k) => !where.has(k.join("|")));
  if (missing.length) throw new Error(`${propagateFrom} lacks ${missing.length} atom(s) of this input, e.g. ${JSON.stringify(missing.slice(0, 3))}`);
  const flat = Float32Array.from(atomKeys.flatMap((k) => where.get(k.join("|"))));
  f = addPropagation(f, flat, Number(option("lag-ns", "1")));
  x0 = new Float32Array(N * 3); x0.set(flat);
}

// ---- the entries
const STEER_INTS = ["atom_pad_mask", "atom_chain_id", "chain_sizes", "symmetric_chain_index", "interchain_steric_atom_mask",
  "interchain_bonded_pair_key", "polymer_steric_atom_mask", "polymer_bonded_pair_key", "polymer_local_clash_index",
  "bond_index", "angle_index", "rdkit_bounds_index", "rdkit_bounds_bond_mask", "rdkit_bounds_angle_mask",
  "chiral_atom_index", "chiral_atom_orientations", "stereo_bond_index", "stereo_bond_orientations", "planar_bond_index"];
const STEER_FLOATS = ["atom_vdw_radius", "rdkit_upper_bounds", "rdkit_lower_bounds"];
const entries = [];
const int = (name, v) => entries.push(["i", name, v instanceof Int32Array ? v : Int32Array.from(v)]);
const flt = (name, v) => entries.push(["t", name, v instanceof Float32Array ? v : Float32Array.from(v)]);
for (const name of ["res_type", "residue_index", "asym_id", "entity_id", "sym_id", "token_center_atom_index",
                    "token_pad_mask", "method_feature", "pocket_feature", "msa", "msa_paired", "has_deletion", "msa_mask",
                    "element", "name_encoding", "token_idx_at_atom", "ref_space_uid", "atom_pad_mask", "atom_mask"]) {
  int(name, f[name]);
}
for (const name of ["deletion_mean", "profile", "token_bonds", "contact_threshold", "deletion_value",
                    "template_coords", "charge", "coords_cond"]) {
  flt(name, f[name]);
}
const cls = new Int32Array(T * T);
for (let p = 0; p < T * T; p += 1) for (let k = 0; k < 5; k += 1) if (f.contact_conditioning[p * 5 + k]) cls[p] = k;
int("contact_class", cls);
if (x0) flt("x0", x0);
// ---- the steering potentials' tables (see the header)
const steering = args.find((a) => a === "--steering" || a.startsWith("--steering="));
if (steering !== undefined) {
  let path = steering.startsWith("--steering=") ? steering.slice(11) : "";
  if (path === "") {
    path = join(homedir(), ".cache", "alphafold3", "bioemu2", `stereo_chemistry.${STEREO_COMMIT}.yaml`);
    if (!existsSync(path)) {
      const response = await fetch(STEREO_URL);
      if (!response.ok) throw new Error(`could not fetch BioEmu2's stereochemistry tables: ${response.status}`);
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, await response.text());
    }
  }
  const t = steeringTables(topology, parseStereoTables(readFileSync(path, "utf8")),
                           { asymId: f.asym_id, entityId: f.entity_id }, { ccd });
  const flat = (v) => (Array.isArray(v) && v.length && (Array.isArray(v[0]) || ArrayBuffer.isView(v[0]))
    ? v.flatMap((r) => Array.from(r)) : Array.from(v, (x) => (x === true ? 1 : x === false ? 0 : x)));
  for (const name of STEER_INTS) int(`steer/${name}`, flat(t[name]));
  for (const name of STEER_FLOATS) flt(`steer/${name}`, flat(t[name]));
  for (const name of ["bond_ref", "angle_ref"]) {
    const words = new Int32Array(Float64Array.from(t[name].flat()).buffer);
    int(`steer/${name}.bits`, words);
  }
}
const S = f.msa.length / T;
let nValid = 0;
for (let s = 0; s < S; s += 1) {
  let any = false; for (let t = 0; t < T; t += 1) any = any || f.msa_mask[s * T + t] !== 0;
  nValid += any ? 1 : 0;
}
entries.push(["m", "meta/tokens", T], ["m", "meta/atoms", N], ["m", "meta/real_atoms", realAtoms],
             ["m", "meta/msa_rows", S], ["m", "meta/n_valid_rows", Math.max(nValid, 1)],
             ["m", "meta/mode", mode === "equilibrium" ? 1 : 0], ["m", "meta/dt", f.dt],
             ["m", "meta/is_dynamics_conditioned", f.is_dynamics_conditioned ? 1 : 0],
             ["m", "meta/propagation_steps", x0 ? propagationSteps : 0]);

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

// ---- the structure's records and the output's tokens, in the job's chain order
const chainOrder = job.chains.map((c) => c.id);
const tokenChain = new Int32Array(T);            // BioEmu2 token -> its chain (an index into chainOrder)
const atomsOfChain = new Map(chainOrder.map((id) => [id, []]));
atomKeys.forEach((key, atom) => atomsOfChain.get(key[0]).push(atom));
const records = [];
const outTokens = [], tokenChainIds = [], tokenResIds = [];
const polymer = new Map(job.chains.map((c) => [c.id, ["protein", "rna", "dna"].includes(c.kind)]));
const modified = new Map(job.chains.map((c) => [c.id, new Set((c.modifications ?? []).map((m) => m.position))]));
let serial = 1;
const CHAIN_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
chainOrder.forEach((id, c) => {
  const atoms = atomsOfChain.get(id);
  const letter = id.length === 1 ? id : CHAIN_CHARACTERS[c % CHAIN_CHARACTERS.length];
  let lastToken = -1;
  for (const atom of atoms) {
    const [, residue, name] = atomKeys[atom];
    const token = topology.tokenIdxAtAtom[atom];
    tokenChain[token] = c;
    if (token !== lastToken) { outTokens.push(token); tokenChainIds.push(id); tokenResIds.push(residue); lastToken = token; }
    const het = !polymer.get(id) || modified.get(id).has(residue);
    let resName = topology.labels[atom];
    const kind = job.chains[c].kind;
    if (!het && (kind === "dna" || kind === "rna") && resName !== "UNK") resName = resName.padStart(3);
    const element = topology.element[atom] === 0 ? "X" : ELEMENT_SYMBOLS[topology.element[atom] - 1] ?? "X";
    const symbol = element.length === 1 ? element : element[0] + element.slice(1).toLowerCase();
    const field = name.length < 4 && symbol.length === 1 ? ` ${name}`.padEnd(4) : name.padEnd(4).slice(0, 4);
    records.push((het ? "HETATM" : "ATOM  ") + String(serial % 100000).padStart(5) + " " + field + " "
      + resName.slice(0, 3).padStart(3) + " " + letter + String(residue).padStart(4) + "    "
      + Math.floor(atom / 1000).toFixed(3).padStart(8) + (atom % 1000).toFixed(3).padStart(8) + (0).toFixed(3).padStart(8)
      + "  1.00  0.00          " + symbol.toUpperCase().padStart(2));
    serial += 1;
  }
  records.push("TER");
});
const pdb = records.join("\n") + "\nEND\n";
writeFileSync(`${out}/pdb.template`, pdb);
writeFileSync(`${out}/confidence.template.json`, JSON.stringify({
  tokens: outTokens, token_chain_ids: tokenChainIds, token_res_ids: tokenResIds,
}));
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${T} tokens, ${realAtoms} atoms (${N} padded), ${S} MSA rows, ${mode}${x0 ? `, propagating ${propagationSteps} step(s)` : ""} -> ${out}`);

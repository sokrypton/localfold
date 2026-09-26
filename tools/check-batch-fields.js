/**
 * OUR WHOLE FEATURISED BATCH against af3-any-model's own, field by field, for
 * every dumped model.
 *
 *     node tools/check-batch-fields.js [--model=rosettafold3] [--verbose]
 *
 * 🔴 THE GENERALISATION OF check-atom-windows.js, AND THE REASON IS THAT ITS
 * TWO SPOT CHECKS KEPT FINDING THINGS. That tool compares `queries_to_keys` and
 * rf3's chiral centres and nothing else; between them they found the atom key
 * window wrong in four of seven models and a chirality term twice recorded as a
 * no-op. The dump carries SIXTY fields. Comparing two of them and reasoning
 * about the other fifty-eight is how both of those survived, so this compares
 * all of them it can name.
 *
 * 🔴 A FIELD THIS PORT DOES NOT BUILD IS REPORTED, NOT SKIPPED SILENTLY. The
 * whole failure mode here is a convention nobody looked for, so an unmapped
 * field prints as `unmapped` and is counted - `--verbose` lists them. Several
 * are genuinely absent by design (the reference emits ligand bond gathers on a
 * batch with no ligands); the point is that the list is visible rather than
 * implied.
 *
 * 🔴 AND IT COMPARES INTEGERS WHERE IT CAN. `ref_pos` is the one field with a
 * known, deliberate floor - this port ships ONE idealised conformer set shared
 * across models where the reference featurises CCD geometry, which is the
 * user's own decision ("ideal conformer shared across all models is worth it")
 * - so it is compared at a loose bound and labelled, never silently passed.
 *
 * 🔴 AND EVERY PER-ATOM FIELD IS COMPARED ONLY WHERE `ref_mask` IS LIVE, WHICH
 * IS THE THIRD TIME THIS PORT HAS LEARNED THAT RULE. The first version of this
 * tool reported `ref_element` differing in one slot of 1632 across the four
 * families that drop the terminal atom, and it looked like a real defect in
 * four shipped models: the reference drops OXT by MASKING it while leaving its
 * element (8), its name ("OXT") and its conformer position in place, where this
 * port never creates the atom and leaves the dense slot zeroed. Both sides then
 * compute a per-atom conditioning row for it - `rows = tokens * dense` - and
 * neither ever reads it: no gather on EITHER side references dense slot 1617
 * with a live mask, checked for all four. So it is a padded slot the reference
 * fills and this port zeroes, which is precisely what docs/OPENDDE.md's note
 * about `mask_mean` describes and what an earlier encoder checker was already
 * corrected for. Compare where the data is real, or spend a night on a slot
 * that computes nothing.
 *
 * 🔴 AND IT HAS BEEN WATCHED FAILING, WHICH IS THE ONLY THING THAT MAKES IT A
 * GATE. `--falsify=<flag>` inverts one dialect convention before featurising:
 *
 *     SHIPPED                  exit 0
 *     paddedAtomKeys           exit 1   28 red lines
 *     qblockAtomKeys           exit 1   12
 *     dropTerminalAtoms        exit 1   70
 *     centreRefConformers      exit 1    8
 *     symmetriseBonds          exit 1    7
 *     atomizedElementNames     exit 1    7
 *     atomizedUnknownRestype   exit 1    8
 *     atomizedUnknownMsa       exit 1    2
 *     atomizedBackboneBonds    exit 1    7
 *     dedupeSelfMsa            exit 0   🔴 NOT OBSERVABLE HERE
 *
 * The last row is the honest limitation. `dedupeSelfMsa` decides whether the
 * QUERY appears twice in the alignment, and the alignment is the one input this
 * port takes from its caller rather than from the reference - so `msa` is not
 * compared and the convention cannot be seen from here. It has evidence
 * elsewhere (AF3 6MRR 83.084 -> 83.169); it does not have it in this gate.
 *
 * 🔴 AND IT GOES THROUGH `af3BatchFromA3m`, NOT STRAIGHT INTO THE
 * FEATURISER. src/af3/featurise/batch.js forwards the dialect to `featuriseProtein` field
 * by field, and the first version of this tool skipped that step - so a
 * convention dropped from THAT list would have left this gate green while the
 * page and every fold tool silently featurised with another model's
 * conventions, which is CLAUDE.md's stale-allow-list trap for a third time.
 * Verified: deleting `atomizedBackboneBonds` from batch.js's forwarding turns
 * this red with "4 bonds the reference has and this port does not".
 *
 * 🔴 AND `centreRefConformers` NEEDED ITS OWN ARM BECAUSE IT IS A
 * TRANSLATION. It is invisible to intra-token distances (invariant) and buried
 * inside ref_pos's floor bucket elementwise, so it was the one convention that
 * stayed green under falsification until the centroid comparison was added. A
 * convention nothing can see is a convention nobody is checking.
 */
import { readFileSync } from "node:fs";
import { af3BatchFromA3m } from "../src/af3/featurise/batch.js";
import { parseCcdComponent } from "../src/af3/featurise/ccd-component.js";
import { dialectFor, DIALECTS, featuriserDialect } from "../src/af3/dialect.js";
import { structuralBatch, structuralLayout } from "../src/af3/featurise/structural-tokens.js";

const MODELS = ["alphafold3", "openbind0", "opendde", "boltz2", "protenix2",
                "intellifold2", "rosettafold3"];

// 🔴 TWO TARGETS, BECAUSE ONE OF THEM CANNOT REACH FOUR CONVENTIONS. 6MRR is a
// plain 68-residue protein: it has no ligand and no modified residue, so
// `symmetriseBonds`, `atomized_element_names`, `atomized_backbone_bonds` and
// `atomized_unknown_restype` are all inert on it, and a gate built only on that
// target reports them green by never reaching them. `gol-sep3` is the same
// sequence with GLYCEROL as a second chain and a PHOSPHOSERINE at position 3 -
// 83 tokens rather than 68, the ligand at 77-82 - dumped by
// tools/oracle/dump_af3_batch.py --ligand GOL --ptm SEP@3.
const ccd = (code) => parseCcdComponent(readFileSync(
  new URL(`fixtures/ccd/${code}.cif`, import.meta.url), "utf8"));
const TARGETS = {
  "6mrr": { suffix: "", extra: () => ({}) },
  "gol-sep3": {
    suffix: "-gol-sep3",
    extra: () => ({
      modifications: [{ chain: 0, position: 3, ...ccd("SEP") }],
      ligands: [ccd("GOL")],
    }),
  },
};

// theirs -> how to get ours. A gather is three fields and is handled below.
const FIELDS = {
  aatype: (b) => b.aatype,
  asym_id: (b) => b.asymId,
  entity_id: (b) => b.entityId,
  sym_id: (b) => b.symId,
  residue_index: (b) => b.residueIndex,
  token_index: (b) => b.tokenIndex,
  seq_mask: (b) => b.seqMask,
  deletion_mean: (b) => b.deletionMean,
  profile: (b) => b.profile,
  ref_mask: (b) => b.refMask,
  ref_element: (b) => b.refElement,
  ref_charge: (b) => b.refCharge,
  ref_atom_name_chars: (b) => b.refAtomNameChars,
  ref_space_uid: (b) => b.refSpaceUid,
  ref_pos: (b) => b.refPos,
  pred_dense_atom_mask: (b) => b.predDenseAtomMask,
};
const GATHERS = {
  "queries_to_keys": (b) => b.queriesToKeys,
  "queries_to_token_atoms": (b) => b.queriesToTokenAtoms,
  "tokens_to_queries": (b) => b.tokensToQueries,
  "tokens_to_keys": (b) => b.tokensToKeys,
  "token_atoms_to_queries": (b) => b.tokenAtomsToQueries,
  "token_atoms_to_pseudo_beta": (b) => b.tokenAtomsToPseudoBeta,
};
// Fields this port has no equivalent for, each with the reason. Listed rather
// than omitted, so "we do not build it" stays a claim someone can check.
const ABSENT = {
  msa: "the dump's MSA is the reference's own search; ours is the caller's",
  msa_mask: "as msa",
  deletion_matrix: "as msa",
  is_protein: "derived from aatype at use, not stored",
  is_dna: "as is_protein", is_rna: "as is_protein", is_ligand: "as is_protein",
  is_water: "as is_protein", is_nonstandard_polymer_chain: "as is_protein",
  is_modified: "boltz2 only; derived from the modification list at use",
  frames_mask: "the frame set is built in the confidence head, not the batch",
  residue_center_index: "derived from the dense layout at use",
  chiral_angles: "gated in check-atom-windows.js",
  chiral_centers: "gated in check-atom-windows.js",
  ligand_ligand_bond_order: "compared beside the bond gathers, as an ORDER per pair",
  template_aatype: "the template slot is built by buildTemplate, not featurise",
  template_atom_mask: "as template_aatype",
  template_atom_positions: "as template_aatype",
};
// A bound per field. Integers are exact; ref_pos carries the shared-conformer
// floor this repository has chosen deliberately and documents everywhere.
const BOUNDS = { ref_pos: 2.5, ref_charge: 1e-6, profile: 1e-6,
                 deletion_mean: 1e-6, default: 0 };
// Per-atom fields live on the dense `tokens * dense` grid, where a slot is real
// only if `ref_mask` says so. See the note above on why this is not optional.
const PER_ATOM = new Set(["ref_element", "ref_charge", "ref_atom_name_chars",
                          "ref_space_uid", "ref_pos", "ref_mask",
                          "pred_dense_atom_mask"]);
const WIDTH = { ref_pos: 3, ref_atom_name_chars: 4 };
// The reference's bond gathers. On a ligand-free target they are entirely dead
// and this says so; on `gol-sep3` they carry real bonds and are compared
// against this port's bond matrix as an undirected edge set - see below.
const BOND_GATHERS = ["token_atoms_to_polymer_ligand_bonds",
                      "tokens_to_ligand_ligand_bonds",
                      "tokens_to_polymer_ligand_bonds"];

const flat = (v) => (Array.isArray(v?.data) ? v.data : []);
const args = process.argv.slice(2);
const only = (args.find((a) => a.startsWith("--model=")) ?? "").slice(8);
const verbose = args.includes("--verbose");
// 🔴 AND IT MUST BE ABLE TO FAIL. `--falsify=<flag>` inverts one dialect flag
// before featurising, and every one of them must turn some model red - a gate
// nobody has watched fail is a gate that may be comparing nothing. See the
// bottom of this file for the run that proves it.
const falsify = (args.find((a) => a.startsWith("--falsify=")) ?? "").slice(10);
const onlyTarget = (args.find((a) => a.startsWith("--target=")) ?? "").slice(9);

/** The dump for one (model, target), or null when it has not been generated. */
function dumpFor(model, target) {
  const file = `af3-batch-${model}-6mrr${TARGETS[target].suffix}.json`;
  try {
    return JSON.parse(readFileSync(new URL(`../oracle-dumps/${file}`, import.meta.url)));
  } catch { return null; }
}

// One place that turns a model into a batch, because there are two loops below
// and the first version let `--falsify` reach only one of them - so the
// centroid arm silently measured the SHIPPED options while claiming to measure
// the falsified ones, and read as "this convention is unobservable".
function batchFor(dialect, target) {
  // 🔴 THROUGH `af3BatchFromA3m` AND `featuriserDialect`, NOT STRAIGHT INTO THE
  // FEATURISER. This called `featuriseProtein` with a hand-picked option list
  // at first, and that is the same allow-list shape it was written to catch:
  // src/af3/featurise/batch.js forwards the dialect to the featuriser field by field, and
  // a gate that skips it would stay green while the PAGE and every fold tool
  // silently dropped a convention. The one path the shipped fold takes is the
  // one to measure.
  const options = { ...featuriserDialect(dialect), ...TARGETS[target].extra() };
  if (falsify !== "") {
    if (!(falsify in options)) {
      throw new Error(`--falsify=${falsify} is not a dialect field the featuriser reads`);
    }
    options[falsify] = !options[falsify];
  }
  return options;
}
let failures = 0;
let missing = 0;
for (const target of Object.keys(TARGETS)) {
  if (onlyTarget !== "" && target !== onlyTarget) continue;
  console.log(`\n=== target ${target}`);
  for (const model of MODELS) {
  if (only !== "" && model !== only) continue;
  const dump = dumpFor(model, target);
  if (dump === null) { missing += 1; console.log(`${model.padEnd(14)} no dump`); continue; }
  const dialect = dialectFor(model);
  const batch = af3BatchFromA3m(dump.sequence, null, batchFor(dialect, target)).batch;

  const bad = [];
  const floor = [];   // deliberate deviations: reported every run, never failed
  const notes = [];
  const seen = new Set();
  for (const [name, pick] of Object.entries(FIELDS)) {
    const theirs = flat(dump.inputs[name]).map(Number);
    if (theirs.length === 0) continue;
    seen.add(name);
    const ours = pick(batch);
    if (ours === undefined) { notes.push(`${name}: unmapped`); continue; }
    // 🔴 COMPARE THE OVERLAP AND SAY SO. Their atom axis is the dense grid and
    // ours is compacted for some models; a length difference is reported as
    // its own line rather than silently truncating the comparison to nothing.
    const n = Math.min(ours.length, theirs.length);
    const bound = BOUNDS[name] ?? BOUNDS.default;
    const width = WIDTH[name] ?? 1;
    const perAtom = PER_ATOM.has(name);
    const theirLive = flat(dump.inputs.ref_mask).map(Number);
    let differ = 0;
    let worst = 0;
    let skipped = 0;
    for (let at = 0; at < n; at += 1) {
      // Live on BOTH sides, since a slot either side calls padding carries
      // whatever each happens to leave there.
      if (perAtom) {
        const slot = Math.floor(at / width);
        if (!(theirLive[slot] > 0.5) || !(batch.refMask[slot] > 0.5)) { skipped += 1; continue; }
      }
      const delta = Math.abs(Number(ours[at]) - theirs[at]);
      if (delta > worst) worst = delta;
      if (delta > bound) differ += 1;
    }
    const lengths = ours.length === theirs.length
      ? "" : ` [ours ${ours.length} theirs ${theirs.length}]`;
    const pad = skipped === 0 ? "" : ` (${skipped} padded slots not compared)`;
    if (differ !== 0) {
      // 🔴 ref_pos IS THE ONE DELIBERATE DEVIATION AND MUST NOT TURN THIS GATE
      // RED FOREVER. A gate that is always failing is a gate nobody reads. It
      // is reported on every run with its size, and the frame-free section
      // below is what says whether it is still only a frame-and-rotamer
      // difference. Anything else here is a defect.
      const line = `${name}: ${differ}/${n - skipped} worst ${worst.toPrecision(3)}${lengths}${pad}`;
      (name === "ref_pos" ? floor : bad).push(line);
    } else if (verbose) {
      notes.push(`${name}: ${n - skipped} ok worst ${worst.toPrecision(3)}${lengths}${pad}`);
    }
  }
  for (const [name, pick] of Object.entries(GATHERS)) {
    const idx = flat(dump.inputs[`${name}:gather_idxs`]).map(Number);
    const msk = flat(dump.inputs[`${name}:gather_mask`]).map(Number);
    if (idx.length === 0) continue;
    seen.add(`${name}:gather_idxs`); seen.add(`${name}:gather_mask`);
    seen.add(`${name}:input_shape`);
    const ours = pick(batch);
    if (ours === undefined) { notes.push(`${name}: unmapped gather`); continue; }
    let maskDiff = 0;
    let indexDiff = 0;
    let live = 0;
    const n = Math.min(ours.indices.length, idx.length);
    for (let at = 0; at < n; at += 1) {
      const mine = ours.mask[at] > 0.5;
      const theirs = msk[at] > 0.5;
      if (mine !== theirs) { maskDiff += 1; continue; }
      if (!mine) continue;
      live += 1;
      if (Number(ours.indices[at]) !== idx[at]) indexDiff += 1;
    }
    if (maskDiff !== 0 || indexDiff !== 0) {
      bad.push(`${name}: mask ${maskDiff} index ${indexDiff} of ${live} live`);
    } else if (verbose) notes.push(`${name}: ${live} live ok`);
  }
  // 🔴 THE BOND GATHERS ARE COMPARED BY CONTENT, NOT BY REPRESENTATION. This
  // port carries bonds as a dense `tokens x tokens` matrix and the reference as
  // a list of token pairs, so a gate that asks "did you build this field"
  // answers no for both a port that has the bonds and one that does not. The
  // first version did exactly that and reported all seven models missing the
  // ligand bonds, when six of them have every bond the reference has. Compare
  // the SET of bonded token pairs.
  for (const name of BOND_GATHERS) {
    const idx = flat(dump.inputs[`${name}:gather_idxs`]).map(Number);
    const msk = flat(dump.inputs[`${name}:gather_mask`]).map(Number);
    if (idx.length === 0) continue;
    seen.add(`${name}:gather_idxs`); seen.add(`${name}:gather_mask`);
    seen.add(`${name}:input_shape`);
    // Each entry is a (row, column) token pair; the mask is per element.
    const theirs = new Set();
    for (let k = 0; k * 2 + 1 < idx.length; k += 1) {
      if (!(msk[k * 2] > 0.5)) continue;
      theirs.add(`${idx[k * 2]}-${idx[k * 2 + 1]}`);
    }
    if (theirs.size === 0) {
      if (verbose) notes.push(`${name}: dead on both sides`);
      continue;
    }
    // 🔴 AND DIRECTION IS PART OF THE QUESTION, WHICH THE FIRST VERSION THREW
    // AWAY. It compared undirected edges, on the reasoning that the reference
    // lists most bonds one way round while this port's matrix is symmetric -
    // and that reasoning discards `symmetriseBonds`, the convention this whole
    // field exists for. Measured on gol-sep3: AlphaFold 3's DIRECTED set is
    // ours exactly, 14 against 14 with nothing either way, and the symmetrising
    // dialects have exactly 14 more, which is each internal bond's reverse. So
    // the expectation is the reference's own pairs plus their reverses where
    // the dialect symmetrises - exact, and it makes the convention visible.
    const expected = new Set(theirs);
    if (dialect.symmetriseBonds === true) {
      for (const pair of theirs) {
        const [a, b] = pair.split("-").map(Number);
        expected.add(`${b}-${a}`);
      }
    }
    const ours = new Set();
    if (batch.bondMatrix !== undefined) {
      for (let i = 0; i < batch.tokens; i += 1) {
        for (let j = 0; j < batch.tokens; j += 1) {
          if (batch.bondMatrix[i * batch.tokens + j]) ours.add(`${i}-${j}`);
        }
      }
    }
    const missing = [...expected].filter((p) => !ours.has(p));
    const extra = [...ours].filter((p) => !expected.has(p));
    // 🔴 AND THE BOND ORDER, WHICH SAT IN THE "ABSENT" LIST SAYING "no ligand on
    // this target" - true of 6MRR and false of gol-sep3, so the moment a target
    // had bonds the gate stopped comparing the one channel that was empty.
    // boltz2 reads it as the second plane of its z-init bond feature and was
    // getting zeros; see the note in featurise.js.
    const orders = flat(dump.inputs.ligand_ligand_bond_order).map(Number);
    let orderDiff = 0;
    if (orders.length > 0 && batch.bondOrderMatrix !== undefined) {
      seen.add("ligand_ligand_bond_order");
      for (let k = 0; k * 2 + 1 < idx.length; k += 1) {
        if (!(msk[k * 2] > 0.5)) continue;
        const a = idx[k * 2];
        const b = idx[k * 2 + 1];
        if (a === b) continue;
        if (batch.bondOrderMatrix[a * batch.tokens + b] !== orders[k]) orderDiff += 1;
      }
      if (orderDiff !== 0) {
        bad.push(`ligand_ligand_bond_order: ${orderDiff} pairs with the wrong order`);
      } else if (verbose) {
        notes.push(`ligand_ligand_bond_order: exact over ${theirs.size} pairs`);
      }
    }
    if (missing.length !== 0 || extra.length !== 0) {
      bad.push(`${name}: ${missing.length} bonds the reference has and this port`
        + ` does not${missing.length === 0 ? "" : ` (${missing.slice(0, 6).join(" ")})`}`
        + `, ${extra.length} the other way`
        + `${extra.length === 0 ? "" : ` (${extra.slice(0, 6).join(" ")})`}`);
    } else if (verbose) {
      notes.push(`${name}: ${expected.size} directed bonded token pairs, exact`);
    }
  }
  // 🔴 OpenDDE's SECOND TOKEN SPACE, which this port builds in
  // src/af3/featurise/structural-tokens.js and which nothing compared for as long as it
  // existed. `structbook/*` is the mapping that DEFINES the space - which
  // parent residue each subtoken belongs to, its role, its twin - so if it is
  // wrong every stage after it is wrong on a shipped model.
  const structural = Object.keys(dump.inputs)
    .filter((k) => k.startsWith("struct/") || k.startsWith("structbook/"));
  // 🔴 EVERY `struct/*` KEY IS MARKED SEEN WHOLESALE, so the UNMAPPED check
  // below cannot see a structural field nothing compares - which is how
  // `struct/token_atoms_to_pseudo_beta` sat in the dump uncompared.
  for (const k of structural) seen.add(k);
  if (structural.length > 0) {
    const found = compareStructural(dump, batch);
    for (const line of found.bad) bad.push(line);
    for (const line of found.floor) floor.push(line);
    for (const line of found.notes) notes.push(line);
  }

  const unmapped = Object.keys(dump.inputs)
    .filter((k) => !seen.has(k) && ABSENT[k] === undefined);
  if (unmapped.length > 0) bad.push(`UNMAPPED: ${unmapped.join(" ")}`);

  const status = bad.length === 0
    ? (floor.length === 0 ? "EXACT" : "EXACT but for the conformer floor") : "DIFFERS";
  console.log(`${model.padEnd(14)} ${String(seen.size).padStart(2)} fields compared`
    + `  ${status}`);
  for (const line of bad) console.log(`    🔴 ${line}`);
  for (const line of floor) console.log(`    floor  ${line}`);
  for (const line of notes) console.log(`       ${line}`);
  if (bad.length !== 0) failures += 1;
  }
}
const targets = Object.keys(TARGETS).filter((t) => onlyTarget === "" || t === onlyTarget);
const checked = MODELS.filter((m) => only === "" || m === only).length * targets.length;
console.log(`\n${checked - missing - failures} model/target pairs exact on the compared`
  + ` fields, ${failures} differing, ${missing} without a dump`);
const unlisted = Object.keys(DIALECTS).filter((m) => !MODELS.includes(m));
if (unlisted.length > 0) {
  console.log(`🔴 a dialect has no dump here, so its batch is unchecked: ${unlisted.join(", ")}`);
}

// 🔴 AND `ref_pos` DESERVES A FRAME-FREE QUESTION, BECAUSE THE ELEMENTWISE ONE
// IS PARTLY MEANINGLESS. An idealised conformer has an ARBITRARY rigid frame:
// the reference reads CCD geometry in the CCD's own orientation and this port
// generates its own, so two identical molecules can differ by 9 A elementwise
// while being the same shape. The elementwise number above is what the network
// actually consumes (`embed_ref_pos` is a linear layer, so orientation reaches
// it) and is the honest size of this port's documented conformer floor. This
// asks the other half: are they the same MOLECULE? Intra-residue pairwise
// distances are invariant to rotation and translation, so a small number here
// with a large number above means "same geometry, different frame" - a
// deliberate deviation - and a large number here would mean a wrong conformer,
// which is a defect. The distinction is not visible in either number alone.

/**
 * OpenDDE's second token space, against the reference's `struct/` and
 * `structbook/`.
 *
 * 🔴 THE REFERENCE PADS AND THIS PORT DOES NOT, so every comparison is over the
 * first `tokens` entries: 130 real subtokens against a batch bucketed to 160,
 * with `struct/seq_mask` zero across the tail. Comparing the whole array would
 * report thirty tokens of padding as a disagreement.
 *
 * 🔴 AND `ref_space_uid` IS COMPARED AS A PARTITION, NOT ELEMENTWISE. It exists
 * to answer "are these two atoms in the same space", so the LABELS are
 * arbitrary and only the grouping is meaningful: this port numbers densely from
 * zero and the reference skips a number, which differs in 3116 of 3120 slots
 * and means nothing. Both give 68 spaces containing exactly the same atoms.
 * Elementwise it looks like a defect; as a partition it is exact.
 */
function compareStructural(dump, batch) {
  const bad = [];
  const floor = [];
  const notes = [];
  const layout = structuralLayout(batch);
  const structural = structuralBatch(batch, layout);
  const tokens = layout.tokens;
  const flatten = (name) => flat(dump.inputs[name]).map(Number);

  const live = flatten("struct/seq_mask");
  if (live.length > tokens && live.slice(tokens).some((v) => v > 0.5)) {
    bad.push(`struct/: the reference has live tokens past our ${tokens}`);
    return { bad, floor, notes };
  }

  // The mapping that defines the space. `structbook` is the reference's name.
  const BOOK = {
    parent: "structbook/parent_residue_idx", role: "structbook/subtoken_role_id",
    twin: "structbook/twin_token_idx", prevParent: "structbook/prev_parent_residue_idx",
    nextParent: "structbook/next_parent_residue_idx",
    residueAtomGather: "structbook/residue_atom_gather",
    residueRepToken: "structbook/residue_rep_token",
  };
  for (const [ours, theirs] of Object.entries(BOOK)) {
    const mine = layout[ours];
    const yours = flatten(theirs);
    if (yours.length === 0) continue;
    if (mine === undefined) { bad.push(`${theirs}: this port does not build it`); continue; }
    let differ = 0;
    for (let at = 0; at < Math.min(mine.length, yours.length); at += 1) {
      if (Number(mine[at]) !== yours[at]) differ += 1;
    }
    if (differ !== 0) bad.push(`${theirs}: ${differ} of ${mine.length} differ`);
  }

  // The per-token and per-atom fields, over the real tokens only.
  const FIELDS = {
    aatype: "struct/aatype", asymId: "struct/asym_id", entityId: "struct/entity_id",
    symId: "struct/sym_id", residueIndex: "struct/residue_index",
    tokenIndex: "struct/token_index", seqMask: "struct/seq_mask",
    profile: "struct/profile", deletionMean: "struct/deletion_mean",
    refMask: "struct/ref_mask", refElement: "struct/ref_element",
    refCharge: "struct/ref_charge", refAtomNameChars: "struct/ref_atom_name_chars",
    predDenseAtomMask: "struct/pred_dense_atom_mask", refPos: "struct/ref_pos",
  };
  for (const [ours, theirs] of Object.entries(FIELDS)) {
    const mine = structural[ours];
    const yours = flatten(theirs);
    if (yours.length === 0 || mine === undefined) continue;
    const bound = theirs === "struct/ref_pos" ? 2.5 : 1e-6;
    let differ = 0;
    let worst = 0;
    for (let at = 0; at < mine.length && at < yours.length; at += 1) {
      const delta = Math.abs(Number(mine[at]) - yours[at]);
      if (delta > worst) worst = delta;
      if (delta > bound) differ += 1;
    }
    if (differ === 0) continue;
    const line = `${theirs}: ${differ}/${mine.length} worst ${worst.toPrecision(3)}`;
    (theirs === "struct/ref_pos" ? floor : bad).push(line);
  }

  // 🔴 THE STRUCTURAL PSEUDO-BETA GATHER, WHICH NOTHING COMPARED. `GATHERS`
  // above carries `token_atoms_to_pseudo_beta` - the RESIDUE-space one - and
  // the structural-space twin sits in the dump beside it, uncompared. It is
  // the ONE geometric input OpenDDE's confidence head takes: the head adds a
  // binned AND a raw embedding of the distances between these atoms, so a
  // gather that picked the wrong atom would move the PAE and nothing here
  // would object. Same shape as the atom-key window, which was wrong in four
  // of seven models because no gate asked.
  {
    const idx = flat(dump.inputs["struct/token_atoms_to_pseudo_beta:gather_idxs"]).map(Number);
    const msk = flat(dump.inputs["struct/token_atoms_to_pseudo_beta:gather_mask"]).map(Number);
    const mine = structural.tokenAtomsToPseudoBeta;
    // Never a silent skip: a dump that carries the field and a port that does
    // not build it is the finding, not a reason to compare nothing.
    if (idx.length > 0 && mine === undefined) {
      bad.push("struct/token_atoms_to_pseudo_beta: this port does not build it");
    } else if (idx.length > 0) {
      let differ = 0;
      for (let at = 0; at < idx.length && at < mine.indices.length; at += 1) {
        const live = msk[at] > 0.5;
        if (live !== (Number(mine.mask[at]) > 0.5)) { differ += 1; continue; }
        if (live && Number(mine.indices[at]) !== idx[at]) differ += 1;
      }
      if (differ > 0) {
        bad.push(`struct/token_atoms_to_pseudo_beta: ${differ}/${idx.length} differ`);
      }
    }
  }

  // ref_space_uid as a PARTITION - see the note above.
  {
    const theirs = flatten("struct/ref_space_uid");
    const mask = flatten("struct/pred_dense_atom_mask");
    const group = (values) => {
      const spaces = new Map();
      for (let at = 0; at < structural.refSpaceUid.length && at < theirs.length; at += 1) {
        if (!(mask[at] > 0.5)) continue;
        const key = values[at];
        if (!spaces.has(key)) spaces.set(key, []);
        spaces.get(key).push(at);
      }
      return [...spaces.values()].map((slots) => slots.join(" ")).sort();
    };
    const mine = group(structural.refSpaceUid);
    const yours = group(theirs);
    if (JSON.stringify(mine) !== JSON.stringify(yours)) {
      bad.push(`struct/ref_space_uid: ${mine.length} spaces here against ${yours.length}`
        + " - the atoms are grouped differently, which changes what may be compared");
    } else if (verbose) {
      notes.push(`struct/ref_space_uid: ${mine.length} spaces, the same partition`
        + " (the labels differ and are arbitrary)");
    }
  }

  notes.push(`${tokens} structural subtokens compared`
    + ` against the reference's ${live.length} padded`);
  return { bad, floor, notes };
}

let centroidFailures = 0;
{
  console.log("\nref_pos, frame-free: intra-token pairwise distances");
  for (const target of Object.keys(TARGETS)) {
    if (onlyTarget !== "" && target !== onlyTarget) continue;
    console.log(`  target ${target}`);
    for (const model of MODELS) {
    if (only !== "" && model !== only) continue;
    const dump = dumpFor(model, target);
    if (dump === null) continue;
    const dialect = dialectFor(model);
    const batch = af3BatchFromA3m(dump.sequence, null, batchFor(dialect, target)).batch;
    const theirPos = flat(dump.inputs.ref_pos).map(Number);
    const theirMask = flat(dump.inputs.ref_mask).map(Number);
    const dense = batch.dense;
    let worst = 0;
    let sum = 0;
    let pairs = 0;
    let worstToken = -1;
    for (let token = 0; token < batch.tokens; token += 1) {
      const live = [];
      for (let slot = 0; slot < dense; slot += 1) {
        const at = token * dense + slot;
        if (theirMask[at] > 0.5 && batch.refMask[at] > 0.5) live.push(at);
      }
      for (let a = 0; a < live.length; a += 1) {
        for (let b = a + 1; b < live.length; b += 1) {
          const d = (pos, i, j) => Math.hypot(
            pos[i * 3] - pos[j * 3], pos[i * 3 + 1] - pos[j * 3 + 1],
            pos[i * 3 + 2] - pos[j * 3 + 2]);
          const delta = Math.abs(d(batch.refPos, live[a], live[b])
                                 - d(theirPos, live[a], live[b]));
          sum += delta * delta;
          pairs += 1;
          if (delta > worst) { worst = delta; worstToken = token; }
        }
      }
    }
    const rms = Math.sqrt(sum / Math.max(pairs, 1));
    // 🔴 AND THE CENTROID, WHICH IS THE ONE THING NEITHER OTHER TEST CAN SEE.
    // `centreRefConformers` is a pure TRANSLATION: invariant to intra-token
    // distances, and buried in ref_pos's floor bucket elementwise - so
    // `--falsify=centreRefConformers` left this whole gate green until this
    // was added. A centred conformer's token centroid is 0 and an uncentred
    // one's is not, so comparing the two centroids' magnitudes names the
    // convention directly.
    let ourCentroid = 0;
    let theirCentroid = 0;
    let counted = 0;
    for (let token = 0; token < batch.tokens; token += 1) {
      const live = [];
      for (let slot = 0; slot < dense; slot += 1) {
        const at = token * dense + slot;
        if (theirMask[at] > 0.5 && batch.refMask[at] > 0.5) live.push(at);
      }
      if (live.length === 0) continue;
      const mean = (pos) => {
        let x = 0; let y = 0; let z = 0;
        for (const at of live) { x += pos[at * 3]; y += pos[at * 3 + 1]; z += pos[at * 3 + 2]; }
        return Math.hypot(x, y, z) / live.length;
      };
      ourCentroid += mean(batch.refPos);
      theirCentroid += mean(theirPos);
      counted += 1;
    }
    ourCentroid /= Math.max(counted, 1);
    theirCentroid /= Math.max(counted, 1);
    // 🔴 COMPARE THE CONVENTION, NOT THE MAGNITUDE. Two UNCENTRED conformer
    // sets have centroids of 1.480 and 1.580 - a 0.1 A difference that is just
    // the rotamer difference above showing up in a mean - and the first version
    // of this line called that a disagreement. What is being asked is binary:
    // does this side subtract the token centroid or not.
    const CENTRED = 0.05;   // A: a centred conformer's centroid is exactly 0
    const oursCentred = ourCentroid < CENTRED;
    const theirsCentred = theirCentroid < CENTRED;
    const centred = oursCentred === theirsCentred
      ? `both ${oursCentred ? "centred" : "uncentred"}`
      : `🔴 DIFFER (ours ${oursCentred ? "centred" : "uncentred"},`
        + ` theirs ${theirsCentred ? "centred" : "uncentred"})`;
    console.log(`  ${model.padEnd(14)} ${String(pairs).padStart(5)} intra-token pairs`
      + `  rms ${rms.toFixed(4)} A  worst ${worst.toFixed(4)} A`
      + (worstToken < 0 ? "" : ` (token ${worstToken}, aatype ${batch.aatype[worstToken]})`)
      + `  centroid ${ourCentroid.toFixed(3)}/${theirCentroid.toFixed(3)} ${centred}`);
    if (centred.startsWith("🔴")) centroidFailures += 1;
    }
  }
}

// The verdict covers BOTH sections. Without this the centroid arm could fail
// while the line above still read "7 models exact", which is how a gate starts
// lying about itself.
if (centroidFailures > 0) {
  console.log(`🔴 ${centroidFailures} models disagree on the conformer CENTRING convention`);
}
if (failures > 0 || centroidFailures > 0) process.exitCode = 1;
else console.log("every dumped model's batch matches the reference's, "
  + "but for the shared-conformer floor");

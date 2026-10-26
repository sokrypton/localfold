/**
 * The design page's condition editor: residues picked in the viewer, set as any of RFD4-Proteina's
 * conditions - what AtomWorks Studio's editor does, written as the condition spec a job carries.
 *
 * 🔴 IT WRITES THE SPEC, NOT THE BATCH. Every row becomes `conditions: {C_X: {VALUE: [{select}]}}`
 * with a `sel('<chain>/*\/[<residues>]/[<atoms>]')` path, the form upstream's own examples use, so
 * what the page runs, what the saved job.json says and what `localfold-rfd4p --job=` reads are one
 * thing - and the pipeline (shared/rfd4p/pipeline/spec.js) refuses a bad value in upstream's words
 * rather than this module second-guessing it. tools/check-rfd4p-batch.py holds that pipeline's
 * batch to upstream's for every condition offered here.
 *
 * 🔴 AND THE NAMES ARE THE INPUT STRUCTURE'S, AS THE SELECTION LANGUAGE READS THEM: the label chain
 * id and `res_id` (label_seq_id on a polymer, auth_seq_id otherwise), which is what
 * `parseCifToAtomArray` gives. The viewer draws a PDB written from that same array
 * (`inputStructurePdb`), so a click maps back to exactly those names (`inputResidueMap`), never to
 * an author numbering the selection would not find.
 */

import { toPdbString } from "../shared/rfd4p/output.js";

/**
 * The conditions Studio's editor offers, by upstream alias. `values` is the enum's names in
 * upstream's own spelling (rfproteina/conditions.py, atomworks' definitions), or the kind of free
 * value the condition takes. `level` says what a selection must cover: `atom` conditions may be
 * narrowed to named atoms, `chain` conditions take whole chains.
 */
export const CONDITION_KINDS = Object.freeze([
  { key: "C_HOT", label: "Hotspot", values: ["true"], level: "atom" },
  { key: "C_CRD", label: "Fix coordinates", values: ["true", "false"], level: "atom" },
  { key: "C_SEQ", label: "Fix sequence", values: ["true", "false"], level: "residue" },
  { key: "C_SEC", label: "Secondary structure", values: ["HELIX", "SHEET", "LOOP"], level: "residue" },
  { key: "C_RAS", label: "Burial (RASA)", values: ["BURIED", "INTERMEDIATE", "EXPOSED"], level: "atom" },
  { key: "C_HB", label: "H-bond", values: ["DONOR", "ACCEPTOR", "BOTH"], level: "atom" },
  { key: "C_SALT", label: "Salt bridge", values: ["POSITIVE", "NEGATIVE"], level: "atom" },
  { key: "C_MCOORD", label: "Metal coordination", values: ["METAL", "COORDINATING"], level: "atom" },
  { key: "C_PISTACK", label: "Pi stack", values: ["true"], level: "atom" },
  { key: "C_PICATION", label: "Pi-cation", values: ["AROMATIC", "CATION", "BOTH"], level: "atom" },
  { key: "C_HPHOB", label: "Hydrophobic", values: ["true"], level: "atom" },
  { key: "C_CRDGRP", label: "Coordinate group", values: "group", level: "atom" },
  { key: "C_IDX", label: "Unindexed motif", values: "label", level: "atom" },
  { key: "C_CENTROID", label: "Chain centroid", values: "xyz", level: "chain" },
  { key: "C_CATH", label: "CATH label", values: "cath", level: "residue" },
]);

const KIND = new Map(CONDITION_KINDS.map((kind) => [kind.key, kind]));

/** What a free-valued condition's box asks for, and how it is checked before it is written. */
const FREE = {
  group: { placeholder: "1", ok: (v) => /^[1-9]\d*$/.test(v), want: "a positive integer group id" },
  label: { placeholder: "motif", ok: (v) => /^\S+$/.test(v), want: "a group label with no spaces" },
  xyz: { placeholder: "0,0,0", ok: (v) => v.split(",").length === 3
    && v.split(",").every((x) => x.trim() !== "" && Number.isFinite(Number(x))), want: "x,y,z in angstroms" },
  cath: { placeholder: "1.10.238", ok: (v) => /^(\d+|x)(\.(\d+|x)){0,3}$/.test(v), want: "a C.A.T label like 1.10.238" },
};

/** Sorted residue numbers as `16-20,25`. */
export function residueRanges(numbers) {
  const sorted = [...new Set(numbers)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < sorted.length; i += 1) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j += 1;
    out.push(i === j ? `${sorted[i]}` : `${sorted[i]}-${sorted[j]}`);
    i = j;
  }
  return out.join(",");
}

/**
 * The selection for picked residues, one `sel()` a chain joined by `or`.
 *
 * @param {{chain: string, resId: number}[]} picks
 * @param {string} [atoms] comma-separated atom names to narrow to, e.g. "OD1,OD2"; empty is all
 * @param {"atom"|"residue"|"chain"} [level]
 */
export function selectionOf(picks, atoms = "", level = "atom") {
  if (picks.length === 0) throw new Error("pick residues in the viewer first");
  const names = atoms.split(",").map((a) => a.trim().toUpperCase()).filter((a) => a !== "");
  if (names.length > 0 && level !== "atom") {
    throw new Error("this condition applies to whole residues, so it takes no atom names");
  }
  const byChain = new Map();
  for (const { chain, resId } of picks) {
    if (!byChain.has(chain)) byChain.set(chain, []);
    byChain.get(chain).push(resId);
  }
  return [...byChain.entries()].map(([chain, residues]) => {
    if (level === "chain") return `sel('${chain}')`;
    const atomPart = names.length === 0 ? "" : `/[${names.join(",")}]`;
    return `sel('${chain}/*/[${residueRanges(residues)}]${atomPart}')`;
  }).join(" or ");
}

/**
 * One editor row as a spec arm. Refuses what the condition cannot take, by name.
 *
 * @param {{key: string, value: string, picks: {chain: string, resId: number}[], atoms?: string}} row
 * @returns {{key: string, value: string, arm: {select: string, group?: string}}}
 */
export function armOf(row) {
  const kind = KIND.get(row.key);
  if (kind === undefined) throw new Error(`no condition ${row.key}`);
  const value = String(row.value ?? "").trim();
  if (Array.isArray(kind.values)) {
    if (!kind.values.includes(value)) {
      throw new Error(`${kind.label} takes ${kind.values.join(", ")}, not "${value}"`);
    }
  } else if (!FREE[kind.values].ok(value)) {
    throw new Error(`${kind.label} takes ${FREE[kind.values].want}, not "${value}"`);
  }
  const select = selectionOf(row.picks, row.atoms ?? "", kind.level);
  // An unindexed motif is upstream's `C_IDX: {false: [{select, group}]}`: the label groups blocks.
  if (row.key === "C_IDX") return { key: "C_IDX", value: "false", arm: { select, group: value } };
  return { key: row.key, value, arm: { select } };
}

/**
 * Conditions merged in order: `base` first, then each row's arm appended under its key and value.
 * Upstream applies arms in declaration order and the last wins an atom two of them select, so the
 * editor's rows - what the user set last - come after a preset's or the spans'.
 */
export function mergeConditions(base, rows) {
  const out = {};
  for (const [key, arms] of Object.entries(base ?? {})) {
    out[key] = {};
    for (const [value, list] of Object.entries(arms)) out[key][value] = [...list];
  }
  for (const row of rows) {
    const { key, value, arm } = armOf(row);
    out[key] ??= {};
    out[key][value] ??= [];
    out[key][value].push(arm);
  }
  return out;
}

/**
 * The parsed input structure as a PDB the viewer draws, with its chain ids made single letters
 * (a label chain id may be two), and the map from those letters back.
 *
 * @param {object} atoms `parseCifToAtomArray`'s array
 * @returns {{pdb: string, chainOf: Map<string, string>}} `chainOf` maps the PDB letter to the
 *   structure's chain id
 */
export function inputStructurePdb(atoms) {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const letterOf = new Map();
  const chainOf = new Map();
  const count = atoms?.nAtoms ?? atoms?.chain_id?.length ?? 0;
  const out = [];
  for (let i = 0; i < count; i += 1) {
    // Waters are drawn as nothing a condition could mean; hydrogens add nothing to a cartoon.
    if (String(atoms.res_name[i]) === "HOH" || String(atoms.element[i]).toUpperCase() === "H") continue;
    const chain = String(atoms.chain_id[i]);
    if (!letterOf.has(chain)) {
      if (letterOf.size >= letters.length) throw new Error(`more than ${letters.length} chains to draw`);
      const letter = chain.length === 1 && !chainOf.has(chain) ? chain : letters[letterOf.size];
      letterOf.set(chain, letter);
      chainOf.set(letter, chain);
    }
    const [x, y, z] = atoms.coord[i];
    if (![x, y, z].every(Number.isFinite)) continue;   // an atom the preparation added unplaced
    out.push({ chainId: letterOf.get(chain), resId: atoms.res_id[i], resName: atoms.res_name[i],
      atomName: atoms.atom_name[i], element: atoms.element[i], x, y, z, isHetero: atoms.hetero[i] });
  }
  return { pdb: toPdbString(out, { title: "input structure" }), chainOf };
}

/**
 * For each drawn position of a py2Dmol renderer, the input residue it is: `{chain, resId}` in the
 * structure's own names, through the letters `inputStructurePdb` wrote.
 */
export function inputResidueMap(renderer, chainOf) {
  const chains = renderer?.chains ?? [];
  const numbers = renderer?.residueNumbers ?? [];
  return Array.from(chains, (letter, i) => ({
    chain: chainOf.get(String(letter)) ?? String(letter), resId: Number(numbers[i]),
  }));
}

/**
 * Picked residues as the span editor's `From structure` rows: one segment a contiguous run, chains
 * in the order first picked, a chain break between two chains. What a motif scaffolding job is
 * built from, so a motif can be dragged out of the sequence strip rather than typed.
 *
 * @param {{chain: string, resId: number}[]} picks
 */
export function segmentsOf(picks) {
  const byChain = new Map();
  for (const { chain, resId } of picks) {
    if (!byChain.has(chain)) byChain.set(chain, []);
    byChain.get(chain).push(resId);
  }
  const spans = [];
  for (const [chain, residues] of byChain) {
    if (spans.length > 0) spans.push({ kind: "break" });
    for (const run of residueRanges(residues).split(",")) {
      const [first, last = first] = run.split(/(?<=\d)-/).map(Number);
      spans.push({ kind: "segment", chain, first, last, hotspots: "" });
    }
  }
  return spans;
}

/** Distinct residues out of drawn positions (a ligand is one position an atom). */
export function picksOf(positions, map) {
  const seen = new Map();
  for (const p of positions) {
    const residue = map[p];
    if (residue === undefined) continue;
    seen.set(`${residue.chain}:${residue.resId}`, residue);
  }
  return [...seen.values()];
}

/**
 * The editor panel. It holds rows; the page owns the viewer, and hands the current picks in.
 *
 * @param {HTMLElement} host
 * @param {object} hooks
 * @param {() => {chain: string, resId: number}[]} hooks.picks the residues selected in the viewer
 * @param {() => void} hooks.onChange the rows changed (repaint, re-describe the job)
 */
export function createConditionEditor(host, { picks, onChange }) {
  let rows = [];
  const panel = document.createElement("div");
  panel.className = "condition-editor";
  panel.id = "condition-editor";

  const kind = document.createElement("select");
  kind.id = "condition-kind";
  for (const k of CONDITION_KINDS) kind.append(new Option(k.label, k.key));
  const valueSelect = document.createElement("select");
  valueSelect.id = "condition-value";
  const valueText = document.createElement("input");
  valueText.id = "condition-value-text";
  valueText.type = "text";
  valueText.size = 10;
  const atoms = document.createElement("input");
  atoms.id = "condition-atoms";
  atoms.type = "text";
  atoms.size = 10;
  atoms.placeholder = "all atoms";
  atoms.title = "Atom names to narrow the condition to, e.g. OD1,OD2 or NZ. Empty is every atom of each residue.";
  const add = document.createElement("button");
  add.type = "button";
  add.id = "condition-add";
  add.className = "btn btn-grey btn-small";
  add.textContent = "Add selection";
  const note = document.createElement("div");
  note.className = "option-note";
  note.id = "condition-note";
  const list = document.createElement("ul");
  list.id = "condition-rows";
  list.className = "condition-rows";

  const paintKind = () => {
    const k = KIND.get(kind.value);
    const fixed = Array.isArray(k.values);
    valueSelect.hidden = !fixed;
    valueText.hidden = fixed;
    if (fixed) valueSelect.replaceChildren(...k.values.map((v) => new Option(v, v)));
    else { valueText.value = ""; valueText.placeholder = FREE[k.values].placeholder; }
    atoms.disabled = k.level !== "atom";
    if (atoms.disabled) atoms.value = "";
  };
  kind.addEventListener("change", paintKind);
  paintKind();

  const paintRows = () => {
    list.replaceChildren(...rows.map((row, index) => {
      const k = KIND.get(row.key);
      const item = document.createElement("li");
      const { arm } = armOf(row);
      item.append(`${k.label} = ${row.value} · ${arm.select}`);
      const drop = document.createElement("button");
      drop.type = "button";
      drop.className = "condition-drop";
      drop.setAttribute("aria-label", `remove ${k.label}`);
      drop.textContent = "×";
      drop.addEventListener("click", () => { rows.splice(index, 1); paintRows(); onChange(); });
      item.append(" ", drop);
      return item;
    }));
  };

  add.addEventListener("click", () => {
    const k = KIND.get(kind.value);
    const row = { key: k.key, value: Array.isArray(k.values) ? valueSelect.value : valueText.value.trim(),
      picks: picks(), atoms: atoms.value };
    try {
      armOf(row);
    } catch (error) {
      note.textContent = error.message;
      note.classList.add("warn");
      return;
    }
    note.textContent = "";
    note.classList.remove("warn");
    rows.push(row);
    paintRows();
    onChange();
  });

  const controls = document.createElement("div");
  controls.className = "condition-controls";
  controls.append(kind, valueSelect, valueText, atoms, add);
  const title = document.createElement("span");
  title.className = "fetch-option-title";
  title.textContent = "Conditions";
  const hint = document.createElement("div");
  hint.className = "option-note";
  hint.textContent = "Pick residues of the loaded structure - click them in the viewer or drag along"
    + " the sequence - then add them as a condition.";
  panel.append(title, controls, hint, note, list);
  host.append(panel);

  return {
    get rows() { return rows.map((row) => ({ ...row, picks: [...row.picks] })); },
    /** The residues picked right now, through the page's hook (which may refuse). */
    picksNow() { return picks(); },
    set(next) { rows = next.map((row) => ({ ...row, picks: [...row.picks] })); paintRows(); onChange(); },
    /** The rows over `base`, as the job's `conditions`. */
    conditions(base) { return mergeConditions(base, rows); },
    /** Grey the whole panel with a reason, or bring it back with `null`. */
    disable(reason) {
      for (const c of panel.querySelectorAll("select, input, button")) c.disabled = reason !== null;
      if (reason === null) paintKind();
      panel.title = reason ?? "";
    },
  };
}

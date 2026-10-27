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
  const { atoms: out, chainOf } = inputStructureAtoms(atoms);
  return { pdb: toPdbString(out, { title: "input structure" }), chainOf };
}

/** `inputStructurePdb`'s atoms before they are written, for a page that adds to them. */
export function inputStructureAtoms(atoms) {
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
  return { atoms: out, chainOf };
}

/**
 * For each drawn position of a py2Dmol renderer, the input residue it is: `{chain, resId}` in the
 * structure's own names, through the letters `inputStructurePdb` wrote.
 */
export function inputResidueMap(renderer, chainOf, ghosts = new Map()) {
  const chains = renderer?.chains ?? [];
  const numbers = renderer?.residueNumbers ?? [];
  // 🔴 A SKETCHED PIECE IS NO RESIDUE OF THE STRUCTURE (web/job-sketch.js): undefined, which
  // `picksOf` skips, so dragging over a ghost picks nothing a condition could name.
  return Array.from(chains, (letter, i) => (ghosts.has(String(letter)) ? undefined : {
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

/** Picked residues as a reader names them: `A86–95, A101` (chain and residue numbers). */
export function describePicks(picks) {
  const byChain = new Map();
  for (const { chain, resId } of picks) {
    if (!byChain.has(chain)) byChain.set(chain, []);
    byChain.get(chain).push(resId);
  }
  return [...byChain.entries()]
    .map(([chain, residues]) => residueRanges(residues).split(",").map((r) => `${chain}${r.replace(/(?<=\d)-/, "\u2013")}`).join(", "))
    .join(", ");
}

/** The constraints a selection is most often given, one press each: [key, value, label]. */
const QUICK = [
  ["C_HOT", "true", "Mark as hotspot", "A hotspot is a residue of the target the design should touch: the site to bind."],
  ["C_CRD", "true", "Fix coordinates", "Hold these residues exactly where they are."],
  ["C_SEQ", "true", "Fix sequence", "Keep these residues' amino acids."],
];

/**
 * The editor panel. It holds rows; the page owns the viewer, and hands the current picks in.
 *
 * 🔴 SELECTION FIRST, FORM SECOND. A newcomer selects residues and then asks "what can I do with
 * these?", so the panel answers that: what is selected, the three things most often done with a
 * selection (and keeping it in the design, which is the page's), and every other condition behind
 * "More constraints". Each constraint is a chip in the colour the viewer paints it, named as the
 * reader would name it; the selection syntax it writes is its tooltip and the saved job.
 *
 * @param {HTMLElement} host
 * @param {object} hooks
 * @param {() => {chain: string, resId: number}[]} hooks.picks the residues selected (may refuse)
 * @param {() => {chain: string, resId: number}[] | null} [hooks.peek] the same without side effects,
 *   null when there is no structure to pick from
 * @param {(index: number, row: object) => string} [hooks.colourOf] a row's colour in the viewer
 * @param {() => void} [hooks.onKeep] "Keep in the design": the page turns the selection into spans
 * @param {() => void} hooks.onChange the rows changed (repaint, re-describe the job)
 */
export function createConditionEditor(host, { picks, peek = () => null, colourOf = () => "#64748b", onKeep, onChange }) {
  let rows = [];
  const panel = document.createElement("div");
  panel.className = "condition-editor";
  panel.id = "condition-editor";
  const button = (label, title, id) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn btn-grey btn-small";
    b.textContent = label;
    b.title = title;
    if (id) b.id = id;
    return b;
  };

  const selection = document.createElement("div");
  selection.className = "condition-selection";
  selection.id = "condition-selection";
  const quick = document.createElement("div");
  quick.className = "condition-quick";
  const note = document.createElement("div");
  note.className = "option-note";
  note.id = "condition-note";
  const list = document.createElement("ul");
  list.id = "condition-rows";
  list.className = "condition-rows";

  const add = (row) => {
    try {
      armOf(row);
    } catch (error) {
      note.textContent = error.message;
      note.classList.add("warn");
      return false;
    }
    note.textContent = "";
    note.classList.remove("warn");
    rows.push(row);
    paintRows();
    onChange();
    return true;
  };
  const tryPicks = () => {
    try {
      return picks();
    } catch (error) {
      note.textContent = error.message;
      note.classList.add("warn");
      return null;
    }
  };

  if (onKeep) {
    const keep = button("Keep in the design", "Add the selected residues to step 3 as pieces kept from the structure.",
      "selectionToSpans");
    keep.addEventListener("click", () => onKeep());
    quick.append(keep);
  }
  for (const [key, value, label, title] of QUICK) {
    const b = button(label, title);
    b.dataset.condition = key;
    b.addEventListener("click", () => {
      const p = tryPicks();
      if (p !== null) add({ key, value, picks: p, atoms: "" });
    });
    quick.append(b);
  }

  // Every condition, behind one disclosure.
  const more = document.createElement("details");
  more.className = "condition-more";
  const summary = document.createElement("summary");
  summary.textContent = "More constraints\u2026";
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
  const addButton = button("Add", "Set the chosen condition on the selected residues.", "condition-add");
  const controls = document.createElement("div");
  controls.className = "condition-controls";
  controls.append(kind, valueSelect, valueText, atoms, addButton);
  more.append(summary, controls);

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
  addButton.addEventListener("click", () => {
    const k = KIND.get(kind.value);
    const p = tryPicks();
    if (p === null) return;
    add({ key: k.key, value: Array.isArray(k.values) ? valueSelect.value : valueText.value.trim(), picks: p,
      atoms: atoms.value });
  });

  const paintRows = () => {
    list.replaceChildren(...rows.map((row, index) => {
      const k = KIND.get(row.key);
      const item = document.createElement("li");
      item.className = "condition-chip";
      const { arm } = armOf(row);
      item.title = arm.select;
      const swatch = document.createElement("span");
      swatch.className = "role-swatch";
      swatch.style.setProperty("--chain-colour", colourOf(index, row));
      const value = Array.isArray(k.values) && k.values.length === 1 ? "" : ` ${row.value.toLowerCase()}`;
      const narrowed = row.atoms ? ` (${row.atoms.toUpperCase()})` : "";
      item.append(swatch, `${k.label}${value} \u00b7 ${describePicks(row.picks)}${narrowed}`);
      const drop = document.createElement("button");
      drop.type = "button";
      drop.className = "condition-drop";
      drop.setAttribute("aria-label", `remove ${k.label}`);
      drop.textContent = "\u00d7";
      drop.addEventListener("click", () => { rows.splice(index, 1); paintRows(); onChange(); });
      item.append(drop);
      return item;
    }));
  };

  /** What is selected now, said once, with the actions live only when there is something to act on. */
  const paintSelection = () => {
    const now = peek();
    const any = Array.isArray(now) && now.length > 0;
    selection.textContent = now === null ? "Load a structure in step 2, then select residues to constrain."
      : any ? `Selected: ${describePicks(now)} (${now.length} residue${now.length === 1 ? "" : "s"})`
        : "Select residues: click them in the viewer or drag along the sequence.";
    selection.classList.toggle("has-selection", any);
    // Nothing to pick from yet: the actions would be four dead buttons.
    quick.hidden = now === null;
    more.hidden = now === null;
    for (const b of quick.querySelectorAll("button")) b.disabled = !any || panel.dataset.off === "1";
    addButton.disabled = !any || panel.dataset.off === "1";
  };

  panel.append(selection, quick, more, note, list);
  host.append(panel);
  paintSelection();

  return {
    get rows() { return rows.map((row) => ({ ...row, picks: [...row.picks] })); },
    set(next) { rows = next.map((row) => ({ ...row, picks: [...row.picks] })); paintRows(); onChange(); },
    /** The residues picked right now, through the page's hook (which may refuse). */
    picksNow() { return picks(); },
    /** The rows over `base`, as the job's `conditions`. */
    conditions(base) { return mergeConditions(base, rows); },
    /** Re-read the selection (the page calls this when the viewer's changes). */
    refreshSelection: paintSelection,
    /** Grey the whole panel with a reason, or bring it back with `null`. */
    disable(reason) {
      panel.dataset.off = reason === null ? "" : "1";
      for (const c of panel.querySelectorAll("select, input")) c.disabled = reason !== null;
      if (reason === null) paintKind();
      panel.title = reason ?? "";
      paintSelection();
    },
  };
}

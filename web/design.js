/**
 * design.js
 * ---------
 * AI Context: THE PROTEIN HUNTER PAGE'S CONTROLS
 *
 * 🔴 THE LOOP IS NOT WIRED. This module makes the controls real and says so
 * when Hunt is pressed. What IS real here is everything that can be real
 * without a GPU run: the target list, the designer's automatic choice (through
 * `chooseDesigner`, the same rule the loop will use), the folder count advice,
 * and whether this browser can run a hunt at all.
 *
 * 🔴 WHY THE AUTOMATIC DESIGNER WRITES ITS REASON OUT. A picker that silently
 * changes its own value is worse than one that cannot - `cpu/design/designers.js`
 * says so in those words, and returns a `why` for exactly this. The note under
 * the picker is that `why`.
 *
 * See docs/DESIGN.md for the plan this page is step one of.
 */
import { createEntityList } from "./entity-ui.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";
import { MAX_ALANINE } from "../webgpu/design/hunter-loop.js";
import { hunt } from "./hunt.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

/** The two counts `chooseDesigner` reads, from the entity list's own rows. */
const contentsOf = (entities) => ({
  ligands: entities.filter((e) => e.type === "ligand" || e.type === "smiles").length,
  nucleic: entities.filter((e) => e.type === "dna" || e.type === "rna").length,
});

const designerSelect = element("designer");
const designerNote = element("designer-note");

/**
 * What the Designer picker says underneath itself.
 *
 * On Auto this is the rule's own reason. On an explicit choice it is the
 * checkpoint's description, plus a warning where that choice cannot see part of
 * the target - picking ProteinMPNN against a ligand is legal and loses the
 * ligand, and the page should say which.
 */
const paintDesignerNote = (entities) => {
  const contents = contentsOf(entities);
  if (designerSelect.value === "auto") {
    const { name, why } = chooseDesigner(contents);
    designerNote.textContent = `Auto → ${DESIGNERS[name].label}, because ${why}.`;
    designerNote.classList.remove("warn");
    return;
  }
  const chosen = DESIGNERS[designerSelect.value];
  const blind = [];
  // 🔴 NAMED, NOT IMPLIED. These are the cases where an explicit pick drops
  // something that is in the target: only LigandMPNN reads heteroatoms, and
  // only NA-MPNN has a nucleic alphabet.
  if (contents.ligands > 0 && designerSelect.value !== "ligand") blind.push("the ligand");
  if (contents.nucleic > 0 && designerSelect.value !== "na" && designerSelect.value !== "ligand") {
    blind.push("the DNA or RNA chain");
  }
  // The checkpoint's own description, authored beside the registry rather than
  // written a second time here, plus whatever this choice cannot see.
  designerNote.textContent = blind.length === 0
    ? `${chosen.label}. ${chosen.note}`
    : `${chosen.label} — it will not see ${blind.join(" or ")}. ${chosen.note}`;
  designerNote.classList.toggle("warn", blind.length > 0);
};

const folders = element("folders");
const folderNote = element("folder-note");

/**
 * What the Folders picker says underneath itself.
 *
 * 🔴 THE ADVICE IS SOMEBODY ELSE'S MEASUREMENT AND IS LABELLED AS SUCH. Yehlin
 * measured two or three folders beating one, and five or six doing worse than
 * two, on her stack with her judge (docs/DESIGN.md 3C). Nothing here has
 * reproduced it, so the page advises and does not enforce.
 */
const paintFolderNote = () => {
  const n = Array.from(folders.selectedOptions).length;
  if (n === 0) {
    folderNote.textContent = "Pick at least one model to fold with.";
    folderNote.classList.add("warn");
    return;
  }
  const rotation = n === 1 ? "One model folds every cycle." : `${n} models, rotating one per cycle.`;
  let advice = "";
  if (n === 1) advice = " Two or three usually beat one: a design has to satisfy all of them.";
  else if (n >= 5) advice = " Five or more did worse than two in the measurement this follows — weak members get an equal vote.";
  folderNote.textContent = rotation + advice;
  folderNote.classList.toggle("warn", n >= 5);
};

// 🔴 `let`, AND THE CALLBACK READS IT DEFENSIVELY, BECAUSE createEntityList
// CALLS onChange WHILE IT IS STILL CONSTRUCTING. It renders its initial rows
// inside the constructor and notifies from there, so a callback that reads the
// binding being initialised throws "Cannot access 'entities' before
// initialization" and the whole module fails to load - which is how this was
// found, by loading the page rather than by resolving its imports.
let entities;
entities = createEntityList(
  element("entity-rows"), element("add-entity"),
  {
    // Something to design against on arrival, as index.html ships a foldable
    // sequence: PD-L1's binding domain, the target Yehlin's ensemble numbers
    // were measured on.
    initial: [{ type: "protein", copies: 1,
      value: "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA" }],
    onChange: () => paintDesignerNote(entities?.read() ?? []),
    // This page has no MSA control, so an automatic template has nothing to
    // draw a hit from. Answer honestly rather than letting the list assume.
    msaIsSearch: () => false,
  },
);

// 🔴 THE TARGET LIST, REACHABLE FROM A DRIVER, as index.html exposes
// `window.__entityList` for the same reason: tools/check-design-page.py sets a
// target and then requires the automatic designer to MOVE, which is the only
// form of that check that can fail for the right reason.
window.__designEntities = entities;

designerSelect.addEventListener("change", () => paintDesignerNote(entities.read()));
folders.addEventListener("change", paintFolderNote);
paintDesignerNote(entities.read());
paintFolderNote();

/**
 * WHICH CONTROLS THE CHOSEN METHOD ACTUALLY HONOURS.
 *
 * 🔴 DISABLED WITH A REASON, NEVER REMOVED. A control that vanishes when the
 * method changes teaches the reader nothing and makes them hunt for it again;
 * one that greys out and says why teaches them what the two methods are. Each
 * entry below is a control Protein Hunter uses and RFD4-Proteina does not,
 * because RFD4 generates structure directly rather than folding and redesigning
 * in a loop - so there is no cycle, no designer, and no folding model.
 */
const HUNTER_ONLY = {
  designerGroup: "RFD4-Proteina writes its own sequence; there is no separate designer.",
  folderGroup: "RFD4-Proteina generates structure directly, so nothing folds it in a loop.",
  scheduleGroup: "RFD4-Proteina has no fold-and-redesign cycle to repeat.",
};

const method = element("method");
const methodNote = element("method-note");

const paintMethod = () => {
  const hunter = method.value === "hunter";
  methodNote.textContent = hunter
    ? "Folds a mostly-X sequence, redesigns on the hallucinated backbone, and re-folds. Needs a folding model."
    : "Generates all-atom structure by flow matching, in one pass. Not wired into this page yet.";
  methodNote.classList.toggle("warn", !hunter);
  for (const [id, why] of Object.entries(HUNTER_ONLY)) {
    const group = document.getElementById(id);
    if (group === null) continue;
    group.classList.toggle("unavailable", !hunter);
    for (const control of group.querySelectorAll("select, input, button")) control.disabled = !hunter;
    let reason = group.querySelector(".unavailable-reason");
    if (reason === null) {
      reason = document.createElement("div");
      reason.className = "option-note unavailable-reason";
      group.append(reason);
    }
    reason.textContent = hunter ? "" : why;
  }
};

method.addEventListener("change", paintMethod);
paintMethod();

// 🔴 WHETHER A HUNT CAN RUN HERE AT ALL, SAID BEFORE IT IS TRIED. A hunt is
// many folds; finding out on cycle one that this browser has no WebGPU is the
// worst time to find out.
const gpuSummary = element("gpu-summary");
if (navigator.gpu === undefined) {
  gpuSummary.textContent = "WebGPU unavailable — this page needs Chrome or Edge with WebGPU to fold.";
  gpuSummary.classList.add("warn");
} else {
  navigator.gpu.requestAdapter().then((adapter) => {
    const info = adapter?.info;
    gpuSummary.textContent = adapter === null || adapter === undefined
      ? "WebGPU present but no adapter answered."
      : `WebGPU · ${[info?.vendor, info?.architecture].filter(Boolean).join(" / ") || "adapter available"}`;
  }).catch((error) => {
    gpuSummary.textContent = `WebGPU could not be queried: ${error.message}`;
  });
}

const status = element("status-message");
const progress = element("progress");
const huntButton = element("hunt");
const resultsBox = element("results");
const resultsBody = element("results-body");
const trackBox = element("cycle-track-container");
const track = element("cycle-track");

const say = (text, bad = false) => {
  status.textContent = text;
  status.classList.toggle("error", bad);
};

const number = (id, fallback) => {
  const raw = Number(element(id).value);
  return Number.isFinite(raw) ? raw : fallback;
};

/**
 * One cycle, as a row and as a line of sequence.
 *
 * 🔴 APPENDED AS IT ARRIVES, NOT COLLECTED AND DRAWN AT THE END. A hunt is
 * minutes long; a page that shows nothing until it finishes is the shape this
 * one exists not to be.
 */
const drawCycle = (record) => {
  resultsBox.style.display = "";
  trackBox.style.display = "";

  const row = document.createElement("tr");
  // 🔴 THE PREVIOUS BEST ROW HAS TO BE UN-MARKED, because the loop decides
  // `best` by MUTATING the record it had marked before - and this page drew
  // that row when it arrived and cannot see the change. Without this, every
  // cycle that was ever the leader keeps the marker and a four-cycle run ends
  // with three "best" rows, which the page gate caught.
  if (record.best === true) {
    for (const marked of resultsBody.querySelectorAll("tr.best")) marked.classList.remove("best");
    row.className = "best";
  }
  const objective = Number.isFinite(record.score) ? record.score.toFixed(3) : "—";
  // 🔴 THE VALIDITY CELL CARRIES A MEASUREMENT, NOT A VERDICT. Only one check
  // exists so far - the loop's alanine ceiling, which is why a cycle can be
  // ineligible - so that is what it reports, as the fraction it actually is.
  // Ring closure and base pairing are step 4 of docs/DESIGN.md.
  const alanine = `${(record.alanine * 100).toFixed(0)}% Ala`;
  const vetoed = record.cycle > 0 && record.alanine > MAX_ALANINE;
  if (vetoed) row.classList.add("rejected");
  for (const [text, className] of [
    [String(record.run + 1), ""], [String(record.cycle), ""],
    [record.folder ?? "—", ""], [objective, ""],
    [vetoed ? `${alanine} — over the ${Math.round(MAX_ALANINE * 100)}% ceiling` : alanine, "validity"],
  ]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }
  resultsBody.append(row);

  const line = document.createElement("div");
  line.textContent = `${record.cycle}  ${record.sequence}`;
  // ...and the same for the track, for the same reason.
  if (record.best === true) {
    for (const marked of track.querySelectorAll(".best-line")) marked.classList.remove("best-line");
    line.className = "best-line";
  }
  track.append(line);
};

let running = null;

const startHunt = async () => {
  const entities = window.__designEntities.read();
  // The designed chain is a new one appended after the target's, which is what
  // chainIndex names; the target's own chains keep their order.
  const target = entities.filter((e) => e.type === "protein").map((e) => e.value);
  const chains = ["", ...target];
  const controller = new AbortController();
  running = controller;
  huntButton.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  say("starting");
  progress.value = 0;
  progress.dataset.state = "running";

  try {
    const { best } = await hunt({
      chains,
      chainIndex: 0,
      folders: Array.from(folders.selectedOptions, (option) => option.value),
      designer: designerSelect.value,
      contents: contentsOf(entities),
      runs: number("runs", 1),
      cycles: number("cycles", 5),
      length: number("binder-length", 75),
      temperature: number("temperature", 0.1),
      omit: element("omit").value,
      seed: number("seed", 0),
      signal: controller.signal,
      report: drawCycle,
      status: (text) => say(text),
      progress: (fraction) => { progress.value = fraction; },
    });
    say(best === null
      ? "no cycle was eligible — every one was over the alanine ceiling"
      : `done · best run ${best.run + 1} cycle ${best.cycle}, objective ${best.score.toFixed(3)}`);
    progress.dataset.state = "done";
  } catch (error) {
    if (controller.signal.aborted) {
      say("stopped");
      progress.dataset.state = "idle";
    } else {
      say(error.message, true);
      progress.dataset.state = "error";
    }
  } finally {
    running = null;
    huntButton.textContent = "Hunt";
  }
};

huntButton.addEventListener("click", () => {
  if (running !== null) {
    running.abort();
    return;
  }
  if (method.value !== "hunter") {
    say("RFD4-Proteina's engine is in the tree but not wired to this page yet. See docs/DESIGN.md.", true);
    return;
  }
  startHunt();
});

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
import {
  chainColour, chainOfSpans, chainsOf, CONDITION_COLOURS, conditionsOf, contigOf, createContigList,
  drawLengths, hotspotsOf, INPUT_COLOUR, LIGAND_COLOUR, shade, spansFromContig,
} from "./contig-ui.js";
import { DEFAULT_STEPS, generate } from "./generate.js";
import { writeZip } from "./zip.js";
import { validateDesign } from "./validate-design.js";
import { getDevice } from "./model.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { THREE_TO_ONE } from "../cpu/design/mpnn/constants.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";
import { CHAIN_IDS, MAX_ALANINE } from "../webgpu/design/hunter-loop.js";

import { hunt } from "./hunt.js";
import { createStructureViewer } from "./viewer.js";
import {
  createConditionEditor, inputResidueMap, inputStructurePdb, picksOf, segmentsOf,
} from "./condition-editor.js";
import { alignToReference } from "../cpu/design/align.js";
import { translateSpans } from "../cpu/design/source-numbering.js";
import { HUNTER_CANNOT, presetSuits } from "../cpu/design/preset-methods.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

/**
 * The two counts `chooseDesigner` reads, from the spans.
 *
 * 🔴 BOTH ARE REAL NOW: a `Given sequence` span says whether it is protein, DNA
 * or RNA, so a nucleic target picks NA-MPNN and a ligand picks LigandMPNN
 * without anyone choosing by hand. That is the rule the reference uses, and
 * `cpu/design/designers.js` owns it.
 */
const NUCLEIC = new Set(["dna", "rna"]);

const contentsOf = (spans) => ({
  ligands: spans.filter((span) => span.kind === "ligand").length,
  nucleic: spans.filter((span) => span.kind === "sequence" && NUCLEIC.has(span.polymer ?? ""))
    .length,
});

// 🔴 THE METHOD PICKER IS LOOKED UP FIRST, before anything that reads it. The
// span list calls its `method` callback while drawing its initial rows inside
// the constructor, so a `const method` declared further down throws "Cannot
// access 'method' before initialization" and the module never loads. The same
// trap caught the entity list's onChange; a module's declaration order is part
// of its behaviour when constructors call back.
// 🔴 EVERY ELEMENT THIS MODULE TOUCHES IS LOOKED UP HERE, AT THE TOP, and the
// reason is a bug that happened three times: a callback firing while something
// is still CONSTRUCTING - the span list draws its initial rows and notifies from
// inside createContigList - reads a binding declared further down and throws
// "Cannot access X before initialization", and the whole module fails to load.
// The page then renders perfectly and does nothing, which no static check sees.
// Declaration order is part of this file's behaviour, so the lookups do not
// move below the functions that read them.
const method = element("method");
const status = element("status-message");
const progress = element("progress");
const huntButton = element("hunt");
// 🔴 THE LABEL IS A <span> INSIDE THE BUTTON, beside its icon: writing to the
// button's own textContent would delete the icon with it.
const huntLabel = huntButton.querySelector("span");
const resultsBox = element("results");
const resultsBody = element("results-body");
const track = element("cycle-track");
const advancedButton = element("show-advanced");
const stepsTitle = element("steps-title");
// The two methods' step defaults: a hunt's fold diffusion and RFD4's flow.
const HUNTER_STEPS = 25;
const RFD4_STEPS = DEFAULT_STEPS;
const unusedNote = element("unused-note");
const roleList = element("role-list");
// 🔴 HOISTED HERE WITH THE REST, and not left beside the preset code, because
// `paintMethod` rebuilds the example list and runs during construction - long
// before that code. Declared below, it throws "Cannot access presetSelect
// before initialization" and the whole module fails to load, which is the
// fifth time this file has been bitten by exactly that.
const presetSelect = element("preset");
const presetNote = element("preset-note");

// 🔴 EVERY PIECE OF MUTABLE STATE A PAINT FUNCTION READS LIVES HERE TOO, for the
// same reason as the lookups above: `paintJob` runs while the span list is still
// constructing, and it reaches the run state and the loaded structure. Five
// bindings on this page have thrown "Cannot access X before initialization" by
// being declared below the callback that reads them. The rule that actually
// holds is: shared state at the top, helpers as function DECLARATIONS.
let running = null;
// 🔴 DECLARED HERE, MADE BELOW: `paintMethod` runs at load, before the viewer the editor reads
// exists, and a `const` it reached first would throw - this page's fifth binding of that kind.
let conditionEditor = null;
let loaded = null;
// 🔴 THE FIRST CYCLE OF THE RUN, KEPT AS THE FRAME EVERY LATER ONE IS FITTED
// ONTO, and the designed chain's held positions, kept so the viewer can paint
// them. Both belong to a RUN rather than to a record, so `drawCycle` - which
// sees one record at a time - cannot derive either.
// 🔴 EVERY FLOW STEP IS A FRAME. It was a budget of 16, so a 60-step run drew one step in
// four and the play bar skipped most of the trajectory; each frame is an all-atom decoder
// pass, which is the price of seeing all of it.

let reference = null;
let held = [];

/**
 * A chain's own residue numbering, as py2Dmol's global residue indices.
 *
 * 🔴 THE TWO NUMBERINGS ARE NOT THE SAME AND LOOK IDENTICAL WHEN THEY AGREE.
 * `chainsOf` gives positions inside the DESIGNED CHAIN; py2Dmol's `positions`
 * index the whole object, ligands and target chains included. They coincide
 * exactly when the designed chain is written first and nothing precedes it -
 * a monomer design - which is the case a binder run is not.
 *
 * Counted off the PDB rather than off the chain lengths, because a ligand's
 * HETATM residues are in that count too and the span list does not say where.
 */
function globalResidues(pdb, chain, positions) {
  if (positions.length === 0) return [];
  const wanted = new Set(positions);
  const out = [];
  let index = -1;
  let within = 0;
  let previous = null;
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    // Chain, residue number and insertion code together identify a residue.
    const key = line.slice(21, 27);
    if (key === previous) continue;
    previous = key;
    index += 1;
    if (line[21] !== chain) continue;
    if (wanted.has(within)) out.push(index);
    within += 1;
  }
  return out;
}

const say = (text, bad = false) => {
  status.textContent = text;
  status.classList.toggle("error", bad);
};

const number = (id, fallback) => {
  const raw = Number(element(id).value);
  return Number.isFinite(raw) ? raw : fallback;
};


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
const paintDesignerNote = (spans) => {
  const contents = contentsOf(spans);
  if (designerSelect.value === "auto") {
    const { name, why } = chooseDesigner(contents);
    // 🔴 THE ANSWER, NOT THE MECHANISM. With the picker hidden there is no
    // "Auto" for the reader to relate this to, so the line names the checkpoint
    // and the reason it was chosen, and leaves the arrow out of it.
    designerNote.textContent = `${DESIGNERS[name].label} designs the sequence, because ${why}.`;
    designerNote.classList.remove("warn");
    return;
  }
  // ...and the explicit branch below stays live: the select is still settable by
  // a driver, and re-showing the group is one attribute.
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
const beam = element("beam");
const validateSelect = element("validate");

/**
 * What the schedule will actually cost, in folds.
 *
 * 🔴 BECAUSE THE BEAM MULTIPLIES IT AND NOTHING ELSE SAYS SO. Sequences × cycles
 * × runs is the number of folds a hunt runs, and at four sequences over five
 * cycles that is twenty - minutes of work chosen by a control that looks like
 * any other number field.
 */
const paintCost = () => {
  const sequences = Math.max(1, number("beam", 1));
  const folds = sequences * (Math.max(1, number("cycles", 5)) + 1) * Math.max(1, number("runs", 1));
  // 🔴 A GENERATION HAS NO FOLDS TO COUNT. This line is Protein Hunter's
  // arithmetic and said nothing true on RFD4, where it sat under a method that
  // folds nothing at all.
  if (method.value !== "hunter") {
    folderNote.textContent = "";
    folderNote.classList.remove("warn");
    return;
  }
  folderNote.textContent = sequences === 1
    ? `${folds} folds: one sequence a cycle, the reference's method.`
    : `${folds} folds: ${sequences} sequences a cycle, and the best carries forward.`;
  folderNote.classList.toggle("warn", folds > 40);
};

// 🔴 THE JOB, AS A LIST OF SPANS. One editor for both methods: `chainsOf`
// turns these rows into Protein Hunter's chains and `contigOf` into
// RFD4-Proteina's contig, so neither method has an input screen of its own.
// See web/contig-ui.js.
//
// `let`, and the callback reads it defensively, because createContigList draws
// its initial rows inside the constructor and notifies from there - a callback
// reading the binding being initialised throws and the module never loads,
// which is how this was found when the list was an entity list.
let spanList;
spanList = createContigList(element("contig-rows"), element("add-span"), {
  // Something runnable on arrival: a target to design against and a binder,
  // which is Protein Hunter's default case and the page's default method.
  initial: [
    { kind: "sequence",
      value: "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA" },
    { kind: "break" },
    { kind: "generated", min: 75, max: 75 },
  ],
  method: () => method.value,
  onChange: () => paintJob(),
});

const contigLine = element("contig-line");
const contigProblems = element("contig-problems");

/**
 * What the spans amount to, under the method that is selected.
 *
 * 🔴 BOTH DERIVATIONS ARE SHOWN, not just the one the method uses: the contig
 * line is what RFD4 would be asked, and the token count is what either method
 * would produce. A reader switching method sees the same job described two ways,
 * which is the clearest way to convey that the spans are the job and the method
 * is a choice about how to realise it.
 */
/**
 * The colour key, built from the job rather than written into the page.
 *
 * 🔴 IT NAMES THE CHAINS THAT EXIST, NOT A FIXED SET OF ROLES. The static
 * version listed five - designed, held, tied, motif, target - and nothing
 * painted `tied` or `motif`: a legend for a picture the page could not draw.
 * What the viewer actually shows is one colour a chain, so that is what this
 * says, and it comes from the same `chainOfSpans` walk the rows use.
 */
function paintLegend(spans) {
  const chains = [...new Set(chainOfSpans(spans).filter((at) => at !== null))]
    .sort((a, b) => a - b);
  const { chainIndex } = chainsOf(spans, { sequenceOf: resolver() });
  const entry = (colour, text) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "role-swatch";
    swatch.style.setProperty("--chain-colour", colour);
    item.append(swatch, text);
    return item;
  };
  const items = chains.map((at) => entry(chainColour(at),
    `chain ${CHAIN_IDS[at] ?? "?"}${at === chainIndex ? " — designed" : ""}`));
  // Only worth a line when there is something held to explain.
  if (held.length > 0 && chainIndex >= 0) {
    items.push(entry(shade(chainColour(chainIndex), "#374151", 0.55), "held"));
  }
  if (spans.some((span) => span.kind === "ligand")) {
    items.push(entry(LIGAND_COLOUR, "ligand"));
  }
  roleList.replaceChildren(...items);
}

const paintJob = () => {
  const spans = spanList?.read() ?? [];
  paintLegend(spans);
  const hunter = method.value === "hunter";
  const { contig, problems, tokens } = contigOf(spans, method.value);
  const span = tokens.min === tokens.max ? `${tokens.min}` : `${tokens.min}\u2013${tokens.max}`;

  // 🔴 EACH METHOD IS SHOWN WHAT IT WILL ACTUALLY BE ASKED. A contig is RFD4's
  // notation, and under Hunter it is actively misleading - a given SEQUENCE has
  // no contig notation, so the line read "/0,75-75" for a job whose target is
  // 114 residues long. Hunter is told about its chains instead.
  if (hunter) {
    const { chains, chainIndex, length } = chainsOf(spans, { sequenceOf: resolver() });
    const { hold } = chainsOf(spans, { sequenceOf: resolver() });
    // 🔴 A RANGE IS REPORTED AS A RANGE. Hunter resolves one length per run now,
    // and `chainsOf` takes the minimum - so a contig asking for 5-20 residues
    // described itself as "design 5", which is one draw out of sixteen and not
    // what the job says. Both ends are taken by pinning the spans each way.
    const pinned = (end) => chainsOf(spans.map((span) => (span.kind === "generated"
      ? { ...span, min: span[end], max: span[end] } : span)),
    { sequenceOf: resolver() }).length;
    const shortest = pinned("min");
    const longest = pinned("max");
    const described = chains.map((chain, index) => {
      if (index !== chainIndex) return `given ${chain.length}`;
      // 🔴 THE DESIGNED CHAIN IS HANDED OVER EMPTY when every residue of it comes
      // from a Design span - that is what makes the loop draw its own opening
      // sequence - so its SIZE is the spans' total and not the string's length,
      // which is zero. Reading the string here printed "design 0".
      const total = chain.length === 0 ? length : chain.length;
      if (longest !== shortest) {
        const span = `${shortest}\u2013${longest}`;
        return hold.length === 0 ? `design ${span}`
          : `design ${span} with ${hold.length} held`;
      }
      // 🔴 AND HELD RESIDUES ARE PART OF THE DESCRIPTION: a 100-residue chain
      // with 80 held is a 20-residue design, and "design 100" would be wrong by
      // a factor of five.
      return hold.length === 0 ? `design ${total}`
        : `design ${total - hold.length} of ${total}, ${hold.length} held`;
    }).join("  ·  ");
    contigLine.textContent = chains.length === 0 ? "" : described;
  } else {
    contigLine.textContent = contig.length === 0 ? "" : `${contig}    (${span} tokens)`;
  }

  const theirs = hunter ? chainsOf(spans, { sequenceOf: resolver() }).problems : problems;
  contigProblems.textContent = theirs.join(" · ");
  contigProblems.classList.toggle("warn", theirs.length > 0);
  paintDesignerNote(spans);
};

window.__designSpans = spanList;

designerSelect.addEventListener("change", () => paintDesignerNote(spanList.read()));
for (const id of ["beam", "cycles", "runs"]) element(id).addEventListener("input", paintCost);
paintCost();
paintAdvanced();
// 🔴 AN ARROW, NOT THE FUNCTION ITSELF: `paintMethod` is declared below this
// line, and handing the binding over directly reads it before it is
// initialised. Deferring the lookup to call time is the whole fix.
folders.addEventListener("change", () => paintMethod());

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
const HUNTER_NO_CONDITIONS = "Protein Hunter folds sequences; these conditions are RFD4-Proteina's.";
const UNUSED_BY = {
  hunter: {
    thinkingGroup: "Protein Hunter folds with a diffusion model; Thinking is RFD4-Proteina's recycle budget.",
    backboneKGroup: "Backbone K is RFD4-Proteina's sampler; a hunt's fold has no such setting.",
    latentKGroup: "Latents K is RFD4-Proteina's sampler; a hunt's fold has no such setting.",
    partialGroup: "Partial diffusion re-denoises a structure with RFD4-Proteina; a hunt starts from a sequence.",
  },
  rfd4p: {
    folderGroup: "RFD4-Proteina generates structure directly, so nothing folds it in a loop.",
    cyclesGroup: "RFD4-Proteina has no fold-and-redesign cycle to repeat.",
    runsGroup: "RFD4-Proteina generates once; run it again for another sample.",
    temperatureGroup: "RFD4-Proteina does not sample a sequence with a temperature.",
    omitGroup: "RFD4-Proteina has no designer to forbid letters to.",
  },
};

/**
 * 🔴 THE METHOD'S DESCRIPTION IS THE RESTING STATUS LINE, not a note under the
 * picker. index.html does the same with "Ready. Paste a sequence and press
 * Fold." - the one place a reader already looks for what the page is about to
 * do, and one fewer block of prose in the panel.
 */
const RESTING = {
  hunter: "Folds a mostly-X sequence, redesigns it on the hallucinated backbone, and folds again."
    + " Press Hunt.",
  rfd4p: "Generates all-atom structure in one pass. The first run downloads ~420 MB of weights."
    + " Press Generate.",
};

/**
 * The tuning knobs, folded away until asked for.
 *
 * 🔴 ELEVEN CONTROLS IS A FORM, NOT A SETTINGS STRIP. The five behind this are
 * the ones nobody changes on a first run; the six left describe the job.
 */
function paintAdvanced() {
  const open = advancedButton.getAttribute("aria-expanded") === "true";
  const groups = document.querySelectorAll("#designOptions .advanced");
  for (const group of groups) {
    group.hidden = !open;
  }
  // Counted, not typed: the label said "5 more" for five, and nine are behind it now.
  advancedButton.textContent = open ? "fewer" : `${groups.length} more`;
}

advancedButton.addEventListener("click", () => {
  const open = advancedButton.getAttribute("aria-expanded") === "true";
  advancedButton.setAttribute("aria-expanded", open ? "false" : "true");
  paintAdvanced();
});

const paintMethod = () => {
  const hunter = method.value === "hunter";
  // 🔴 ONE CONTROL, TWO MEANINGS, AND THE LABEL SAYS WHICH. Both methods sample
  // a structure over steps; a hunt's are the fold's diffusion steps and RFD4's
  // are its flow steps.
  stepsTitle.textContent = hunter ? "Diffusion steps" : "Flow steps";
  // 🔴 AND ONE DEFAULT EACH. The box said 25 for both, which is the fold's
  // diffusion count and NOT RFD4-Proteina's: upstream samples 60 flow steps
  // (`configs/generation/inference_base.yaml`), and the native port records 25
  // as under-converged - one seed in three came out at CA-CA 4.50 A. A value the
  // user typed is theirs; only the other method's untouched default is swapped.
  const steps = element("steps");
  const mine = hunter ? HUNTER_STEPS : RFD4_STEPS;
  const theirs = hunter ? RFD4_STEPS : HUNTER_STEPS;
  if (Number(steps.value) === theirs) steps.value = String(mine);
  if (running === null) say(RESTING[method.value] ?? "");
  // 🔴 THE BUTTON SAYS WHICH MACHINE IT STARTS. "Hunt" is the loop; RFD4
  // generates in one pass and calling that a hunt would misdescribe it.
  if (running === null) huntLabel.textContent = hunter ? "Hunt" : "Generate";
  // The examples are the method's: four of the ten are RFD4-only.
  paintPresets();
  paintJob();
  paintCost();
  // 🔴 BOTH DIRECTIONS, from one map. Every group either method cannot use is
  // greyed WITH ITS REASON and never removed - a control that vanishes when the
  // method changes teaches nothing and makes the reader hunt for it again.
  const unusable = UNUSED_BY[method.value] ?? {};
  conditionEditor?.disable(hunter ? HUNTER_NO_CONDITIONS : null);
  for (const id of new Set([...Object.keys(UNUSED_BY.hunter), ...Object.keys(UNUSED_BY.rfd4p)])) {
    const group = document.getElementById(id);
    if (group === null) continue;
    const why = unusable[id];
    group.classList.toggle("unavailable", why !== undefined);
    for (const control of group.querySelectorAll("select, input, button")) {
      control.disabled = why !== undefined;
    }
    for (const chip of group.querySelectorAll(".design-chip")) {
      chip.disabled = why !== undefined;
    }
    // 🔴 THE REASON DOES NOT GO ON THE GROUP. Six italic sentences threaded
    // between the controls is the ragged-paragraph problem the notes column was
    // made to fix, and switching to RFD4 reintroduced it in one step. The greyed
    // control says THAT it cannot be used; one line below says what and why.
    group.querySelector(".unavailable-reason")?.remove();
  }

  const names = Object.keys(unusable)
    .map((id) => document.getElementById(id)?.querySelector(".fetch-option-title")?.textContent)
    .filter((name) => name !== undefined && name.trim().length > 0)
    .map((name) => name.toLowerCase());
  unusedNote.textContent = names.length === 0 ? ""
    : `${hunter ? "Protein Hunter" : "RFD4-Proteina"} does not use ${names.length === 1
      ? names[0] : `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}`}.`;
};

method.addEventListener("change", paintMethod);
paintMethod();

// 🔴 WHETHER A HUNT CAN RUN HERE AT ALL, SAID BEFORE IT IS TRIED. A hunt is
// many folds; finding out on cycle one that this browser has no WebGPU is the
// worst time to find out.
const gpuSummary = element("gpu-summary");
if (navigator.gpu === undefined) {
  gpuSummary.textContent = "WebGPU unavailable — this page needs Chrome or Edge 113+, or Safari 26+, to fold.";
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


/**
 * One cycle, as a row and as a line of sequence.
 *
 * 🔴 APPENDED AS IT ARRIVES, NOT COLLECTED AND DRAWN AT THE END. A hunt is
 * minutes long; a page that shows nothing until it finishes is the shape this
 * one exists not to be.
 */
const drawCycle = (record) => {
  resultsBox.style.display = "";

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
  const overAlanine = record.cycle > 0 && record.alanine > MAX_ALANINE;
  // 🔴 THE VETO'S OWN WORDS, WHICH ALREADY CARRY ITS MEASUREMENT - "touches 0/4
  // hotspots (nearest 23.7 A, needs 8 A)". Re-describing it here would be a
  // second place for the number to go stale.
  const vetoed = overAlanine || typeof record.vetoed === "string";
  if (vetoed) row.classList.add("rejected");
  for (const [text, className] of [
    [String(record.run + 1), ""], [String(record.cycle), ""],
    [record.folder ?? "—", ""], [objective, ""],
    [overAlanine ? `${alanine} — over the ${Math.round(MAX_ALANINE * 100)}% ceiling`
      : (record.vetoed ?? alanine), "validity"],
  ]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }
  resultsBody.append(row);

  // 🔴 PUSHED BEFORE THE ROW IS DRAWN would be wrong: the viewer is built on
  // its FIRST frame, and orienting it on that frame is what gives every later
  // one a sensible camera. So push, then orient once, then draw.
  // 🔴 EVERY CYCLE IS FITTED ONTO THE FIRST, OR THE PLAY BAR SHOWS THE CAMERA
  // MOVING. Each cycle is an independent fold, so its coordinates sit wherever
  // the model put them: played back raw, the whole complex tumbles and the one
  // thing that changed - a loop closing, a helix straightening - is invisible
  // underneath the tumbling. `alignToReference` holds the chains the design is
  // NOT, so the designed chain is the only thing that moves.
  if (reference === null) reference = record.pdb;
  const placed = reference === record.pdb ? record.pdb
    : alignToReference(record.pdb, reference, record.chain);

  viewerBox.style.display = "";
  viewer.push(placed, (built) => { if (built) viewer.orient(); });
  // 🔴 COLOURED BY ROLE, NOT BY CONFIDENCE. The page's job is saying what is
  // being DECIDED, and pLDDT answers a different question - so the designed
  // chain takes the "free" colour and everything else the "target" one, the
  // same two swatches the legend names. These are the only two roles a Hunter
  // run has today; fixed, tied and motif arrive with the RunSpec's positions
  // reaching the page.
  // 🔴 HUE IS THE CHAIN, TREATMENT IS THE ROLE. A reader needs both answers -
  // which chain is this, and is it being decided - and there is one set of
  // pixels to give them in. Colouring by role alone, which this page did first,
  // cannot say which chain a held motif belongs to; colouring by chain alone
  // cannot say what is being designed. So every chain keeps its own colour, the
  // one being designed at full strength and the rest faded toward the
  // background - and the fade is a BLEND rather than a swap, so a recessive
  // chain is still the colour of its row.
  const designed = CHAIN_IDS.indexOf(record.chain);
  viewer.colourChains({
    [record.chain]: chainColour(designed),
    ...Object.fromEntries(CHAIN_IDS.split("").map((id, at) => [id, at])
      .filter(([id]) => id !== record.chain)
      .map(([id, at]) => [id, shade(chainColour(at), "#ffffff", 0.62)])),
  });
  // 🔴 AND THE HELD RESIDUES OVER THE TOP, which is what makes the picture
  // answer "which parts are kept": a scaffolded motif is held residues inside
  // a designed chain, so colouring by chain alone paints the motif as though
  // the designer were free to rewrite it. The legend has named `held` since the
  // page was built and nothing painted it until now.
  // ...and the held residues darker rather than grey, so they still read as
  // part of their own chain while saying they are not being rewritten.
  viewer.colourResidues(globalResidues(placed, record.chain, held),
    shade(chainColour(designed), "#374151", 0.55));

  const line = document.createElement("div");
  line.textContent = `${record.cycle}  ${record.sequence}`;
  // ...and the same for the track, for the same reason.
  if (record.best === true) {
    for (const marked of track.querySelectorAll(".best-line")) marked.classList.remove("best-line");
    line.className = "best-line";
  }
  track.append(line);
};

// 🔴 EVERY CYCLE IS A FRAME OF ONE OBJECT, so the play bar walks the whole hunt
// and a design can be WATCHED arriving rather than inspected afterwards. v1 did
// this and it is the single best thing about it.
//
// web/viewer.js is reused as it is: it owns building py2Dmol on the first frame,
// appending the rest, and keeping the camera across rebuilds, and it
// deliberately owns nothing about the prediction. (It was written for the two
// withheld pages and nothing has loaded it since; this is the first page to.)
const viewerBox = element("viewer-container");
const viewer = createStructureViewer({
  container: viewerBox,
  frameLabel: "cycle",
  // 🔴 NO SIDECHAINS, AND NO pLDDT RAMP. The default paint colours sidechains by
  // hydrophobicity, which is right for ONE finished structure and wrong for
  // watching a design move: every cycle redraws a different set of them and the
  // backbone - the thing being designed - disappears behind them. Setting this
  // at CONSTRUCTION rather than after the first frame is what stops the ramp
  // flashing on every rebuild before the roles overwrite it.
  // 🔴 ASKED FOR AT CONSTRUCTION, NOT HIDDEN AFTER. py2Dmol rebuilds side-chain
  // atoms on every frame load, so hiding them in `onPaint` - which runs on the
  // FIRST frame only - had them back by cycle two and every cycle after, with
  // nothing throwing. Not asking for them is the only fix that is not a race
  // with the per-frame rebuild.
  sidechains: false,
  onPaint: (v) => v.paintRoles({}),
  onFail: (message) => say(`the viewer could not start: ${message}`, true),
});

// ---- the condition editor ----------------------------------------------------------------------
// 🔴 THE VIEWER SHOWS THE LOADED STRUCTURE UNTIL A RUN TAKES IT, and only then can a click name one
// of its residues: a design's positions are not the input's. `inputView` is set while it does.
let inputView = null;
const conditionHost = element("condition-host");

/** Paint the drawn input: grey, with each condition row's residues in its own colour. */
const paintInput = () => {
  if (inputView === null || viewer.renderer === undefined) return;
  viewer.colourChains(Object.fromEntries([...inputView.chainOf.keys()].map((letter) => [letter, INPUT_COLOUR])));
  const map = inputResidueMap(viewer.renderer, inputView.chainOf);
  conditionEditor.rows.forEach((row, index) => {
    const wanted = new Set(row.picks.map((pick) => `${pick.chain}:${pick.resId}`));
    const positions = map.flatMap((residue, i) => (wanted.has(`${residue.chain}:${residue.resId}`) ? [i] : []));
    viewer.colourResidues(positions, CONDITION_COLOURS[index % CONDITION_COLOURS.length]);
  });
};

/** Draw the loaded structure in the viewer, for picking. */
function showInput() {
  if (loaded === null || running !== null) return;
  const { pdb, chainOf } = inputStructurePdb(loaded.atoms);
  viewer.cancelAnimations();
  viewer.reset();
  viewer.forgetCamera();
  viewerBox.style.display = "";
  inputView = { chainOf };
  viewer.push(pdb, (built) => { if (built) viewer.orient(); });
  conditionHost.style.display = "";
  sequenceBox.style.display = "";
  paintInput();
  window.SEQ?.clear?.();
  window.SEQ?.buildView?.();
}

// 🔴 THE FOLD PAGE'S SEQUENCE STRIP, REUSED AS IT IS. py2Dmol's panel finds its renderer through
// these callbacks, and the full bundle's own app set them to a viewer this page does not have. It is
// handed this page's viewer only while the INPUT is drawn: a design's positions are not the input's
// residues, and a strip that answered for a design would let a drag pick residues no condition can
// name. Its selection is the viewer's `residueSelection`, so the editor reads strip and clicks alike.
const sequenceBox = element("sequence-viewer-container");
window.SEQ?.setCallbacks?.({
  getRenderer: () => (inputView === null ? null : viewer.renderer ?? null),
  getObjectSelect: () => null,
  applySelection: () => {},
});
document.addEventListener("py2dmol-residue-selection-change", () => {
  if (inputView !== null) window.SEQ?.updateSelection?.();
});
/** Hide the strip while a run owns the viewer. */
const hideSequence = () => { sequenceBox.style.display = "none"; window.SEQ?.clear?.(); };
// The strip wires its own Select all / Unselect / Invert (py2Dmol's panels/seq.js, since
// sokrypton/py2Dmol#32), against the renderer `getRenderer` above names.
// 🔴 A MOTIF DRAGGED OUT OF THE SEQUENCE, not typed: the selection's contiguous runs appended to the
// job as `From structure` spans, in the structure's own numbering, which is what a contig names.
element("selectionToSpans").addEventListener("click", () => {
  let picks;
  try {
    picks = conditionEditor.picksNow();
  } catch (error) {
    say(error.message, true);
    return;
  }
  if (picks.length === 0) { say("select residues first - drag along the sequence or click them", true); return; }
  const added = segmentsOf(picks);
  spanList.set([...spanList.read(), ...added]);
  say(`added ${added.filter((span) => span.kind === "segment").length} span(s) from the selection`);
});

conditionEditor = createConditionEditor(conditionHost, {
  picks: () => {
    if (inputView === null || viewer.renderer === undefined) {
      showInput();
      throw new Error("the viewer was showing a design - the structure is drawn again; pick its residues");
    }
    const selected = viewer.renderer.residueSelection;
    return picksOf(selected instanceof Set ? [...selected] : [],
      inputResidueMap(viewer.renderer, inputView.chainOf));
  },
  onChange: () => { paintInput(); paintJob(); },
});
conditionEditor.disable(method.value === "hunter" ? HUNTER_NO_CONDITIONS : null);
/**
 * The conditions a generation is handed: a preset's, then the spans' hotspots over them (what the
 * user just typed wins), then the editor's rows appended after both - upstream applies arms in
 * order and the last wins an atom two of them select, so what was set by hand is set last.
 */
function runConditions(spans) {
  const fromSpans = conditionsOf(spans);
  const base = fromSpans === undefined ? presetExtras.conditions : { ...presetExtras.conditions, ...fromSpans };
  return conditionEditor.rows.length === 0 ? base : conditionEditor.conditions(base);
}
// A driver's door, as `__designSpans` is: the conditions a generation would be handed now.
window.__designConditions = () => runConditions(spanList.read()) ?? null;
window.__designShowInput = showInput;

const startHunt = async () => {
  const spans = spanList.read();
  // 🔴 THE SPANS ARE THE JOB; `chainsOf` is how Hunter reads them. It also says
  // what Hunter cannot express - a motif inside a designed chain, two designed
  // chains - and a hunt that cannot express the job must not start.
  const { chains, chainKinds, ligandCodes, modifications, chainIndex, length, hold, problems }
    = chainsOf(spans, { sequenceOf: resolver() });
  if (problems.length > 0) {
    say(problems.join(" · "), true);
    return;
  }
  const controller = new AbortController();
  running = controller;
  huntLabel.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  // A new hunt is a new object: the atom count changes with the binder's
  // length, so frames cannot be shared with the last one.
  viewer.cancelAnimations();
  viewer.reset();
  inputView = null;
  hideSequence();
  // A new run is a new frame of reference and a new held set.
  reference = null;
  held = hold;
  say("starting");
  progress.value = 0;
  progress.dataset.state = "running";

  try {
    const { best, cycles } = await hunt({
      // 🔴 RE-READ PER RUN, so a length RANGE becomes a search over lengths
      // rather than a notation the hunt ignores. `drawLengths` is deterministic
      // in the seed and the run, so the same hunt twice is the same hunt.
      job: (run) => chainsOf(drawLengths(spans, { run, seed: number("seed", 0) }),
        { sequenceOf: resolver() }),
      chains,
      chainKinds,
      ligandCodes,
      modifications,
      chainIndex,
      hold,
      // 🔴 THE SAME HOTSPOTS RFD4 CONDITIONS ON, used the only way Hunter can:
      // fold, then reject a cycle whose binder is not touching them.
      // 🔴 `window.__designHotspots` IS A DRIVER'S DOOR, as __designSpans is:
      // hotspots normally come from a `From structure` span, which needs a
      // structure loaded, and tools/check-design-page.py has to be able to force
      // an epitope that cannot be satisfied to prove the veto fires.
      hotspots: window.__designHotspots ?? hotspotsOf(spans),
      // 🔴 STILL A LIST, because web/hunt.js rotates one per cycle and that
      // capability is intact; the page asks for one.
      folders: [folders.value],
      designer: designerSelect.value,
      contents: contentsOf(spans),
      runs: number("runs", 1),
      cycles: number("cycles", 5),
      beam: number("beam", 1),
      validate: validateSelect.value,
      // 🔴 THE FOLD'S DIFFUSION STEPS, which ran at the sampler's default and
      // were not reachable from this page at all.
      steps: number("steps", HUNTER_STEPS),
      length,
      temperature: number("temperature", 0.1),
      omit: element("omit").value,
      seed: number("seed", 0),
      signal: controller.signal,
      report: drawCycle,
      status: (text) => say(text),
      progress: (fraction) => { progress.value = fraction; },
    });
    // 🔴 WHY NO CYCLE WON, FROM THE CYCLES THEMSELVES. This used to blame the
    // alanine ceiling unconditionally, which became a lie the moment a second
    // veto existed: a run rejected for touching nothing was reported as
    // alanine-rich.
    if (best === null) {
      const reasons = [...new Set(cycles.filter((record) => record.cycle > 0)
        .map((record) => record.vetoed
          ?? (record.alanine > MAX_ALANINE
            ? `over the ${Math.round(MAX_ALANINE * 100)}% alanine ceiling` : "not scored")))];
      // 🔴 NOT MARKED AS AN ERROR, because the run COMPLETED - it just has no
      // candidate, which is a result and sometimes the right one (a 24-mer
      // against a lone ligand really is alanine-rich). Red is for a page that
      // could not proceed, and painting this red made the gate read a finished
      // run as a crash.
      say(reasons.length === 0 ? "no cycle completed"
        : `no cycle was eligible — ${reasons.join("; ")}`);
    } else {
      say(`done · best run ${best.run + 1} cycle ${best.cycle},`
        + ` objective ${best.score.toFixed(3)}`);
    }
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
    huntLabel.textContent = method.value === "hunter" ? "Hunt" : "Generate";
  }
};

/**
 * Load one of the engine's worked examples into the spans.
 *
 * 🔴 INTO THE ROWS, NOT INTO A HIDDEN FIELD. A preset is a starting point to
 * edit, so loading one has to leave the user looking at something they can
 * change - which means parsing its contig back into spans. Anything the parser
 * cannot read is reported rather than dropped, because a preset that loaded
 * "most of" its contig would run a different job from the one it names.
 */

/**
 * The examples the selected method can actually run.
 *
 * 🔴 OFFERING ONE THE METHOD WILL REFUSE IS A CONTROL THAT DOES NOTHING. The
 * examples are RFD4-Proteina's and Protein Hunter can take six of the ten, so
 * under Hunter the other four are listed as unavailable WITH their reason
 * rather than silently dropped - a reader who came for the ligand-pocket
 * example needs to know it is the method and not the page.
 *
 * 🔴 AND A LOADED EXAMPLE THE NEW METHOD CANNOT RUN IS CLEARED, because
 * changing the method would otherwise leave the rows describing a job the
 * button refuses, with the picker still naming it.
 */
function paintPresets() {
  const hunter = method.value === "hunter";
  // 🔴 READ THE SELECTION BEFORE CLEARING IT. `replaceChildren` removes the
  // options, and a select with no options reports "" - so comparing against
  // `presetSelect.value` further down compares against nothing and the loaded
  // example loses its place in the picker on every method change.
  const wasChosen = presetSelect.value;
  presetSelect.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Start from\u2026";
  presetSelect.append(blank);
  for (const preset of RFD4P_PRESETS) {
    const suits = presetSuits(preset.id, method.value);
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = suits ? (preset.title ?? preset.id)
      : `${preset.title ?? preset.id} — ${HUNTER_CANNOT[preset.id]}`;
    option.disabled = !suits;
    option.selected = preset.id === wasChosen && suits;
    presetSelect.append(option);
  }
  if (!hunter) return;
  // ...and if the example already loaded is one Hunter cannot run, it goes.
  const chosen = RFD4P_PRESETS.find((entry) => entry.id === wasChosen);
  if (chosen !== undefined && !presetSuits(chosen.id, "hunter")) presetSelect.value = "";
}
paintPresets();


// What a preset carries beyond its contig: the conditions, the template
// structure, and the per-modality options that make partial diffusion and
// inverse folding what they are.
let presetExtras = {};

presetSelect.addEventListener("change", () => {
  const preset = RFD4P_PRESETS.find((entry) => entry.id === presetSelect.value);
  if (preset === undefined) {
    presetExtras = {};
    // 🔴 AND THE EXAMPLE'S SAMPLER GOES WITH IT, back to upstream's defaults: a Partial % left
    // from the partial-diffusion example turned the next job into partial diffusion, which is what
    // dropping `presetExtras` did for these two before they were controls.
    element("thinking").value = "medium";
    element("partial").value = "";
    presetNote.textContent = "";
    presetNote.classList.remove("warn");
    return;
  }
  const spans = spansFromContig(preset.contig);
  if (spans === null) {
    presetNote.textContent = `${preset.title} uses labelled spans this editor has no row for yet`
      + ` (${preset.contig}) — it would run a different job from the one it names.`;
    presetNote.classList.add("warn");
    return;
  }
  // 🔴 AND THE SECONDARY STRUCTURE IS PUT BACK ON THE SPANS, so a loaded preset
  // shows what it asks for instead of an unexplained label. C_SEC's arms name
  // the spans by `sel('*/h1/*/*')`, which is where these come from.
  for (const [wanted, arms] of Object.entries(preset.conditions?.C_SEC ?? {})) {
    for (const arm of arms) {
      for (const match of String(arm.select ?? "").matchAll(/sel\('\*\/([A-Za-z0-9]+)\//g)) {
        for (const span of spans) {
          if (span.name === match[1]) span.secondary = wanted;
        }
      }
    }
  }
  presetExtras = {
    conditions: preset.conditions,
    cifUrl: preset.cifUrl,
    // 🔴 A PRESET WITHOUT `specMode` IS DRIVEN BY ITS STRUCTURE'S ANNOTATIONS,
    // not by its contig - see the note in web/generate.js. Its contig string is
    // written in the original entry's numbering and does not match the cropped
    // example, so running it from the spec fails outright.
    fromAnnotations: preset.specMode !== true && preset.cifUrl !== null
      && preset.cifUrl !== undefined,
    inverseFolding: preset.inverseFolding,
  };
  // 🔴 THE PRESET'S RECYCLE BUDGET AND PARTIAL FRACTION GO INTO THE CONTROLS,
  // not into this object: the budget was carried here once and never shown (the
  // page ran `medium` under a preset that says `low` before that), and a value
  // kept beside a control that shows another is two answers to one question. A
  // bare budget and `thinking_<budget>` are one schedule (shared/rfd4p/flow.js).
  if (preset.recycleSchedule !== undefined) {
    element("thinking").value = String(preset.recycleSchedule).replace(/^thinking_/, "");
  }
  element("partial").value = preset.partialTrajectoryFractionSteps === undefined
    ? "" : String(Math.round(preset.partialTrajectoryFractionSteps * 100));
  if (preset.nsteps !== undefined) element("steps").value = String(preset.nsteps);
  spanList.set(spans);

  // 🔴 AND THE PRESET'S STRUCTURE IS LOADED, NOT JUST NAMED. `cifUrl` was
  // handed to the generation and never fetched by the page, so the structure
  // existed only inside a run - which is fine for RFD4, whose engine fetches it
  // itself, and leaves PROTEIN HUNTER with nothing: Hunter reaches a structure
  // by folding a SEQUENCE, so a `From structure` span needs the file here to
  // read its residues from. Without this, switching a motif preset to Hunter
  // refused with "needs the structure loaded" for a structure the preset had
  // already named.
  const chosen = presetSelect.value;
  // 🔴 ALREADY LOADED IS NOT ALREADY TRANSLATED. The spans were just re-set from
  // the preset, in the ENTRY's numbering, so skipping the fetch must not also
  // skip the move onto the structure - re-picking a preset left its contig
  // naming residues the loaded file does not contain, which is the exact state
  // this whole path exists to prevent.
  if (preset.cifUrl !== undefined && preset.cifUrl !== null && loaded?.name === preset.cifUrl) {
    const { spans: moved, translated } = translateSpans(spans, loaded.atoms);
    if (translated > 0) spanList.set(moved);
  } else if (preset.cifUrl !== null && preset.cifUrl !== undefined) {
    // 🔴 THE OLD STRUCTURE GOES FIRST, SYNCHRONOUSLY. `startGenerate` prefers a
    // loaded structure over the preset's own `cifUrl` - a user's file is the
    // one they mean - and the fetch below takes a moment, so a Generate pressed
    // in that window would run THIS preset's contig against the PREVIOUS
    // preset's structure. It is not hypothetical: it shifted every result in
    // the preset sweep by one, each preset silently generating the job before
    // it, with no error anywhere. Cleared, the engine falls back to the
    // preset's own cifUrl, which is the right answer for the whole window.
    loaded = null;
    describeStructure();
    fetch(preset.cifUrl).then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      // The shipped examples are zstd, which the reader unwraps like any other.
      const bytes = new Uint8Array(await response.arrayBuffer());
      // 🔴 THE PRESET MAY HAVE CHANGED WHILE THIS WAS IN FLIGHT. A fetch that
      // lands after the reader has picked something else would take THIS
      // structure and apply it to THOSE spans - which is not a hypothetical:
      // it shifted every result in the preset sweep by one and translated one
      // preset's contig against another's crop, producing `A17-131`, a range
      // that matched no atoms and still looked like a plausible answer.
      if (presetSelect.value !== chosen) return;
      takeStructure(bytes, preset.cifUrl, false);
      // 🔴 AND THE CONTIG IS PUT INTO THE CROP'S OWN NUMBERING. A shipped
      // example is a CROP, and its contig is written in the numbering of the
      // entry it was cut from - `A16-35` against a file whose chain A runs
      // 0-42. RFD4 never notices, because a preset without `specMode` runs from
      // its structure's annotations rather than its contig; Protein Hunter
      // reads the contig literally and refused, correctly, for a structure the
      // preset had already named. Each atom records where it came from, so this
      // is a lookup rather than an alignment - see cpu/design/source-numbering.js.
      // ...and the spans translated are the ones this PRESET set, not whatever
      // is in the editor now, for the same reason.
      const { spans: moved, translated } = translateSpans(spans, loaded.atoms);
      if (translated > 0) spanList.set(moved);
      paintJob();
    }).catch((error) => {
      // Not fatal: RFD4 fetches it again inside the run, and only Hunter needs
      // it here - so this says so rather than refusing the preset.
      structureNote.textContent = `could not load this example's structure: ${error.message}`;
    });
  }
  // 🔴 A PRESET IS RFD4's, so loading one picks that method rather than leaving
  // a contig the selected method cannot run.
  method.value = "rfd4p";
  paintMethod();
  // 🔴 AND THE ROWS ARE A VIEW OF IT, NOT THE INPUT, for those presets. Editing
  // them switches to the contig path, where the numbering has to match the
  // structure - and for the 1PRW examples it does not. Said plainly rather than
  // letting an edit fail with "matched no atoms".
  presetNote.textContent = (preset.description ?? "")
    + (presetExtras.fromAnnotations
      ? "  \u2014 this preset is driven by its structure's own annotations;"
        + " the spans show what it asks for, and editing them switches to the contig,"
        + " whose numbering may not match the example."
      : "");
  presetNote.classList.remove("warn");
});

/**
 * THE LOADED STRUCTURE, which half the modalities are conditioned on.
 *
 * 🔴 WHAT IT UNLOCKS, so the control is not mistaken for a convenience: a
 * `From structure` span names residues of THIS; partial diffusion starts the
 * flow partway along from it; inverse folding holds its backbone. Without one,
 * only the unconditional cases can run.
 *
 * 🔴 AND IT SERVES BOTH METHODS, differently, which is the geometry axis again.
 * RFD4 takes its coordinates. Protein Hunter cannot - it folds sequences - but
 * it CAN read the sequence of a given segment out of it, which is how a
 * structure-derived target reaches a hunt.
 */
const structureNote = element("structure-note");

/**
 * The one-letter sequence of a loaded structure's residues, for Hunter.
 *
 * 🔴 ONE LETTER PER RESIDUE, NOT PER ATOM, and in residue order. An atom array
 * repeats its residue name once per atom, so walking it naively gives a
 * sequence several times too long - which folds, and folds something else.
 */
/**
 * The resolver, or nothing when no structure is loaded.
 *
 * 🔴 `undefined` IS A DIFFERENT ANSWER FROM "not found", and conflating them
 * produced a wrong message: with the resolver always supplied, a segment span
 * on a page with no structure reported "chain A 1-40 is not in the loaded
 * structure" - naming a structure that does not exist - instead of saying to
 * load one. chainsOf distinguishes the two, so this has to as well.
 *
 * 🔴 AND A DECLARATION, for the fifth time on this page: written as a `const`
 * arrow it threw "Cannot access 'resolver' before initialization" in the same
 * breath as the comment below warning about exactly that. Helpers a
 * constructor-time callback can reach are declarations here, without exception.
 */
function resolver() {
  return loaded === null ? undefined : sequenceOf;
}

// 🔴 A FUNCTION DECLARATION, NOT A `const` ARROW, AND THAT IS THE POINT. This
// is the FOURTH binding on this page that a constructor-time callback reached
// before its initialiser ran - the span list draws and notifies from inside
// createContigList, which calls paintJob, which calls this. A declaration is
// hoisted AND initialised, so the whole class of bug cannot happen to it.
// Anything a callback might reach during construction should be declared this
// way rather than hoisted by hand one at a time.
function sequenceOf(chain, first, last) {
  if (loaded === null) return null;
  const atoms = loaded.atoms;
  const count = atoms?.nAtoms ?? atoms?.chain_id?.length ?? 0;
  const seen = new Map();
  for (let index = 0; index < count; index += 1) {
    if (String(atoms.chain_id[index]) !== chain) continue;
    const residue = Number(atoms.res_id[index]);
    if (residue < first || residue > last) continue;
    if (!seen.has(residue)) seen.set(residue, String(atoms.res_name[index]).trim().toUpperCase());
  }
  if (seen.size === 0) return null;
  return [...seen.entries()].sort((a, b) => a[0] - b[0])
    // 🔴 AN UNKNOWN RESIDUE BECOMES X RATHER THAN DISAPPEARING. Dropping it
    // would shift every residue number after it, and the span's own numbering is
    // what the user typed.
    .map(([, name]) => THREE_TO_ONE[name] ?? "X").join("");
}

const describeStructure = () => {
  if (loaded === null) {
    structureNote.textContent = "";
    structureNote.classList.remove("warn");
    return;
  }
  const parts = [...loaded.chains.entries()]
    .map(([chain, range]) => `${chain} ${range.first}\u2013${range.last}`);
  structureNote.textContent = `${loaded.name}: ${parts.join(", ")}`;
  structureNote.classList.remove("warn");
};

/** The chains a loaded structure has, and the residue range of each. */
const chainRanges = (atoms) => {
  const chains = new Map();
  const count = atoms?.nAtoms ?? atoms?.chain_id?.length ?? 0;
  for (let index = 0; index < count; index += 1) {
    const chain = String(atoms.chain_id[index]);
    const residue = Number(atoms.res_id[index]);
    const seen = chains.get(chain);
    if (seen === undefined) chains.set(chain, { first: residue, last: residue });
    else {
      seen.first = Math.min(seen.first, residue);
      seen.last = Math.max(seen.last, residue);
    }
  }
  return chains;
};

const takeStructure = (data, name, offerChain = true) => {
  // 🔴 PARSED ON LOAD, NOT ON RUN. A file that is not mmCIF, or an entry with no
  // atom records, is a thing to say now - while the user is looking at the
  // control they just used - rather than minutes later inside a generation.
  // `data` is text or the file's BYTES - mmCIF or BinaryCIF, plain, gzip or
  // Zstandard, as upstream takes a structure - and it is handed on as given.
  const atoms = parseCifToAtomArray(data);
  const chains = chainRanges(atoms);
  if (chains.size === 0) throw new Error("no atoms in that file");
  loaded = { data, name, atoms, chains };
  describeStructure();
  // A new structure's residues are new names: rows that named the old one would select nothing.
  if (conditionEditor.rows.length > 0) conditionEditor.set([]);
  showInput();
  // Offer its first chain as a span, since that is almost always the next move.
  // 🔴 NOT WHEN A PRESET LOADED IT. The preset has just written the spans it
  // means, and a preset with no `From structure` span - an unconditional
  // design - would have one PREPENDED here, silently turning it into a
  // different job from the one it names.
  const [first] = [...chains.entries()];
  const spans = spanList.read();
  if (offerChain && !spans.some((span) => span.kind === "segment")) {
    spanList.set([{ kind: "segment", chain: first[0], first: first[1].first, last: first[1].last },
      { kind: "break" }, ...spans.filter((span) => span.kind !== "sequence")]);
  }
  paintJob();
};

element("load-structure").addEventListener("click", async () => {
  const id = element("structure-id").value.trim();
  if (id.length === 0) {
    structureNote.textContent = "give a PDB entry, like 1PRW";
    structureNote.classList.add("warn");
    return;
  }
  structureNote.textContent = `fetching ${id.toUpperCase()}\u2026`;
  structureNote.classList.remove("warn");
  try {
    const response = await fetch(`https://files.rcsb.org/download/${id.toUpperCase()}.cif`);
    if (!response.ok) throw new Error(`RCSB answered ${response.status}`);
    takeStructure(await response.text(), id.toUpperCase());
  } catch (error) {
    structureNote.textContent = `could not load ${id.toUpperCase()}: ${error.message}`;
    structureNote.classList.add("warn");
  }
});

element("structure-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (file === undefined) return;
  try {
    takeStructure(new Uint8Array(await file.arrayBuffer()), file.name);
  } catch (error) {
    structureNote.textContent = `could not read ${file.name}: ${error.message}`;
    structureNote.classList.add("warn");
  }
});

/**
 * How good a generated candidate is, for picking between them.
 *
 * 🔴 A GENERATION SCORES NOTHING OF ITS OWN, which is the whole difficulty: the
 * flow produces coordinates and no opinion about them. So the ranking is the
 * VALIDATION fold's - the confidence a model that did not make it assigns, and
 * whether it reproduces the backbone - and without a validation model there is
 * nothing to rank by and the first candidate stands.
 *
 * 🔴 AND RMSD IS A PENALTY, NOT THE SCORE. A design the validator folds
 * confidently but somewhere else is not the design that was asked for; 2 A is
 * the threshold the field holds a self-consistent design to, so beyond it the
 * confidence is discounted rather than trusted.
 */
function rankOf(candidate) {
  const validation = candidate.validation;
  if (validation == null || validation.failed !== undefined) return 0;
  const confidence = validation.confidence ?? {};
  const base = confidence.iptm ?? (confidence.meanPlddt ?? 0) / 100;
  if (validation.comparable !== true || !Number.isFinite(validation.rmsd)) return base;
  return base - Math.max(0, validation.rmsd - 2) * 0.1;
}

/** One finished candidate, as a row and as the structure on screen. */
function drawGenerated(candidate, total) {
  resultsBox.style.display = "";
  viewerBox.style.display = "";
  // The finished structure is the last frame, in the same frame of reference as
  // the trajectory that led to it.
  viewer.push(reference === null ? candidate.pdb
    : alignToReference(candidate.pdb, reference, undefined),
  (built) => { if (built) viewer.orient(); });

  const row = document.createElement("tr");
  // 🔴 ONE "best" MARKER, CLEARED FIRST - the same rule the hunt table has, and
  // for the same reason: the leader changes as candidates arrive and every row
  // that was ever the leader would otherwise keep the marker.
  for (const marked of resultsBody.querySelectorAll("tr.best")) marked.classList.remove("best");
  row.className = "best";
  const validation = candidate.validation;
  for (const [text, className] of [
    [String(candidate.candidate + 1), ""],
    [total === 1 ? "\u2014" : `of ${total}`, ""],
    ["rfd4-proteina", ""],
    [`${candidate.tokens} tokens`, ""],
    [validation?.described ?? `${candidate.steps} steps`, "validity"],
  ]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }
  resultsBody.append(row);
}

/**
 * THE GENERATIONS OF THE LAST RUN, AS FILES. `generate` returned each design's PDB and mmCIF and
 * nothing offered them, so a design could be looked at and not kept. Each candidate gets its two
 * files, and one archive holds them all with the job that made them - the structure it named
 * beside it - so `localfold-rfd4p --job=job.json` makes the same design again.
 */
const downloadsBox = element("downloads");
let generatedRun = [];

function saveFile(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** The input structure's file name in the archive, by what its bytes are. */
function inputFileName(data) {
  if (typeof data === "string") return "input.cif";
  if (data[0] === 0x28 && data[1] === 0xb5 && data[2] === 0x2f && data[3] === 0xfd) return "input.cif.zst";
  if (data[0] === 0x1f && data[1] === 0x8b) return "input.cif.gz";
  if ((data[0] >= 0x80 && data[0] <= 0x8f) || data[0] === 0xde || data[0] === 0xdf) return "input.bcif";
  return "input.cif";
}

/** The archive's members: the input, the job, a README and each candidate's two files. */
function archiveEntries() {
  const first = generatedRun[0];
  const entries = [];
  const job = { ...first.job };
  if (first.inputCif !== undefined) {
    job.input = inputFileName(first.inputCif);
    entries.push([job.input, first.inputCif]);
  }
  entries.push(["job.json", `${JSON.stringify(job, null, 2)}\n`]);
  entries.push(["README.txt", "RFD4-Proteina designs from the LocalFold design page.\n\n"
    + "To make one again with the native port (seed as listed):\n"
    + "  localfold-rfd4p --job=job.json --int8=<bundle> --seed=<seed> --out=design.pdb\n\n"
    + generatedRun.map((c) => `candidate ${c.candidate + 1}: seed ${c.seed}`
      + (c.validation?.described ? ` - ${c.validation.described}` : "")).join("\n") + "\n"]);
  for (const c of generatedRun) {
    entries.push([`candidate-${c.candidate + 1}.pdb`, c.pdb]);
    entries.push([`candidate-${c.candidate + 1}.cif`, c.cif]);
    // Upstream's design metrics and pre-oracle verdict, keyed as upstream keys them (NaN, which
    // JSON has no number for, as null).
    entries.push([`candidate-${c.candidate + 1}.metrics.json`,
      `${JSON.stringify({ metrics: c.metrics, pre_oracle: c.preOracle }, null, 2)}\n`]);
  }
  return entries;
}

/**
 * One line of what upstream's evaluation would say about a design: the pre-oracle verdict (a
 * failed filter by name), and the metrics a reader acts on, where the job's tags computed them.
 */
function metricsSummary(c) {
  const m = c.metrics ?? {};
  const failed = Object.entries(c.preOracle?.filters ?? {}).filter(([, f]) => !f.pass).map(([k]) => k);
  const pct = (v) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : null);
  const parts = [failed.length === 0 ? "passes the pre-oracle filters" : `fails ${failed.join(", ")}`];
  const clash = pct(m["clash_metrics.frac_atoms_clashing_1p5A"]);
  if (clash !== null) parts.push(`clashing atoms ${clash}`);
  const bonds = pct(m["bond_length_metrics.bond_length.frac_unreasonable"]);
  if (bonds !== null) parts.push(`long bonds ${bonds}`);
  const helix = pct(m["sse_distribution.helix_fraction"]);
  const sheet = pct(m["sse_distribution.sheet_fraction"]);
  if (helix !== null && sheet !== null) parts.push(`helix ${helix}, sheet ${sheet}`);
  const sse = pct(m["sse_accuracy.overall_accuracy"]);
  if (sse !== null) parts.push(`secondary structure as asked ${sse}`);
  const hot = pct(m["hotspot_obedience.hotspot_obedience"]);
  if (hot !== null) parts.push(`hotspots contacted ${hot}`);
  const angstrom = (v) => (Number.isFinite(v) ? `${v.toFixed(2)} Å` : null);
  const motif = angstrom(m["design_motif_allatom_rmsd.rmsd"]);
  if (motif !== null) parts.push(`motif RMSD ${motif}`);
  const toInput = angstrom(m["ca_rmsd_to_input.rmsd"]);
  if (toInput !== null) parts.push(`CA RMSD to the input ${toInput}`);
  return parts.join(" · ");
}
// tools/check-design-page.py reads the members it would download and folds the job natively.
window.__designArchive = archiveEntries;

async function saveArchive() {
  const seeds = generatedRun.map((c) => c.seed);
  saveFile(`rfd4p-design-seed${Math.min(...seeds)}.zip`, await writeZip(archiveEntries()), "application/zip");
}

function paintDownloads() {
  downloadsBox.replaceChildren();
  if (generatedRun.length === 0) {
    downloadsBox.style.display = "none";
    return;
  }
  downloadsBox.style.display = "";
  const heading = document.createElement("h3");
  heading.textContent = "Downloads";
  downloadsBox.append(heading);
  const button = (label, onClick) => {
    const b = document.createElement("button");
    b.className = "btn btn-grey btn-small";
    b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
  };
  for (const c of generatedRun) {
    const row = document.createElement("div");
    row.append(`candidate ${c.candidate + 1} (seed ${c.seed}) `,
      button("PDB", () => saveFile(`candidate-${c.candidate + 1}.pdb`, c.pdb, "chemical/x-pdb")),
      " ",
      button("mmCIF", () => saveFile(`candidate-${c.candidate + 1}.cif`, c.cif, "chemical/x-cif")));
    const metrics = document.createElement("div");
    metrics.className = "design-metrics";
    metrics.textContent = metricsSummary(c);
    row.append(metrics);
    downloadsBox.append(row);
  }
  downloadsBox.append(button("All, with the job (.zip)", () => { saveArchive(); }));
}

/**
 * RFD4-Proteina's sampler, as AtomWorks Studio exposes it: the recycle budget, the two
 * samplers' `tsr_k`, and partial diffusion as a percentage of the trajectory.
 *
 * Returned as `generate()`'s options, which write the same fields into the saved job, so
 * `localfold-rfd4p --job=` runs what the page ran. A K at its released value is left out
 * rather than restated, so a job that never touched it reads as one that did not. Out of
 * Studio's own ranges is refused by name; an empty Partial % is off.
 */
const SAMPLER_DEFAULT_K = { bb_ca: 16, local_latents: 4 };
const rfd4Sampler = () => {
  const out = { recycleSchedule: `thinking_${element("thinking").value}` };
  const sampling = {};
  for (const [id, modality, max, label] of [["backbone-k", "bb_ca", 20, "Backbone K"],
    ["latent-k", "local_latents", 6, "Latents K"]]) {
    const k = Number(element(id).value);
    if (!(k >= 1 && k <= max)) throw new Error(`${label} takes 1 to ${max}, not "${element(id).value}"`);
    if (k !== SAMPLER_DEFAULT_K[modality]) sampling[modality] = { simulation_step_params: { tsr_k: k } };
  }
  if (Object.keys(sampling).length > 0) out.sampling = sampling;
  const partial = element("partial").value.trim();
  if (partial !== "") {
    const pct = Number(partial);
    if (!(pct >= 1 && pct <= 100)) throw new Error(`Partial % takes 1 to 100, not "${partial}"`);
    out.partialTrajectoryFractionSteps = pct / 100;
  }
  return out;
};

/** RFD4-Proteina: one generation, straight into the viewer. */
const startGenerate = async () => {
  const spans = spanList.read();
  const { contig, problems } = contigOf(spans, "rfd4p");
  if (problems.length > 0) {
    say(problems.join(" · "), true);
    return;
  }
  let sampler;
  try {
    sampler = rfd4Sampler();
  } catch (error) {
    say(error.message, true);
    return;
  }
  // 🔴 AN EXAMPLE DRIVEN BY ITS STRUCTURE'S OWN ANNOTATIONS READS NO SPEC, so rows added here would
  // be dropped without a word (generate()'s `fromAnnotations`). Refused instead, with the way out.
  if (conditionEditor.rows.length > 0 && presetExtras.fromAnnotations) {
    say("this example is driven by its structure's own annotations, which take no added conditions"
      + " - clear the example (or load the structure yourself) to add them", true);
    return;
  }
  const controller = new AbortController();
  running = controller;
  inputView = null;
  hideSequence();
  huntLabel.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  viewer.cancelAnimations();
  viewer.reset();
  // A generation is one frame, so there is nothing to fit it onto and nothing
  // held: RFD4 conditions on given coordinates rather than holding a sequence.
  reference = null;
  held = [];
  progress.value = 0;
  progress.dataset.state = "running";
  say("starting");

  // 🔴 THE BEAM IS CANDIDATES, NOT CYCLES. A generation is one pass, so there
  // is nothing to carry forward the way a hunt carries a sequence between
  // cycles - drawing N of them and keeping the best is the only thing "try more
  // than one" can mean here. Each is a whole generation, so the cost is linear
  // and the control says so by being the same control Hunter labels Sequences.
  const candidates = Math.max(1, Math.floor(number("beam", 1)));
  const wanted = validateSelect.value;
  let best = null;
  generatedRun = [];
  paintDownloads();

  try {
    for (let candidate = 0; candidate < candidates; candidate += 1) {
    if (controller.signal.aborted) break;
    // Each candidate is its own trajectory, so it gets its own frame of
    // reference and its own run of the viewer.
    if (candidate > 0) { viewer.cancelAnimations(); viewer.reset(); reference = null; }
    const result = await generate({
      contig,
      ...presetExtras,
      ...(runConditions(spans) === undefined ? {} : { conditions: runConditions(spans) }),
      // A structure the user loaded is the one they mean, over a preset's.
      ...(loaded === null ? {} : { inputCif: loaded.data }),
      steps: number("steps", RFD4_STEPS),
      ...sampler,
      // 🔴 A GENERATION IS A TRAJECTORY AND WAS DRAWN AS ONE PICTURE. The flow
      // runs 60 steps and the page pushed only the finished structure, so the
      // play bar - which hides itself below two frames - never appeared and
      // RFD4 looked static beside a hunt that animates. Asking for a FRAME
      // COUNT rather than a cadence is what keeps the cost bounded: each frame
      // is an all-atom decoder pass on top of the flow, so this is ~16 decodes
      // whether the run is 20 steps or 200.
      frames: number("steps", RFD4_STEPS),
      onStep: (step) => {
        if (step.pdb === undefined) return;
        viewerBox.style.display = "";
        // 🔴 FITTED ONTO THE FIRST FRAME, for the reason a hunt is: a flow step
        // moves the molecule as well as folding it, so played back raw the
        // trajectory tumbles. There is no designed chain to exclude here - the
        // whole thing is being generated - so the fit is over everything.
        if (reference === null) reference = step.pdb;
        const placed = reference === step.pdb ? step.pdb
          : alignToReference(step.pdb, reference, undefined);
        viewer.push(placed, (built) => { if (built) viewer.orient(); });
      },
      // 🔴 A DIFFERENT SEED A CANDIDATE, or the beam draws the same structure
      // N times: the flow is deterministic given its seed, so N identical
      // generations is N times the cost for one answer.
      seed: number("seed", 0) + candidate,
      signal: controller.signal,
      status: (text) => say(candidates === 1 ? text
        : `candidate ${candidate + 1}/${candidates} · ${text}`),
      progress: (fraction) => {
        progress.value = (candidate + fraction) / candidates;
      },
    });

    // 🔴 THE CHECK THE GENERATE PATH DID NOT HAVE. `Validate with` was on this
    // page for both methods and only the hunt honoured it - a control that
    // silently does nothing, which is worse than an absent one. It matters MORE
    // here than in a hunt: a hunt at least optimises a folding model's opinion,
    // where a generation scores nothing at all, so this is the only number the
    // run produces.
    const validation = await validateDesign({
      pdb: result.pdb,
      model: wanted,
      device: await getDevice(),
      seed: number("seed", 0) + candidate,
      // 🔴 NO `steps`: this is the REFOLD's diffusion count, and the box beside
      // RFD4 holds RFD4's FLOW steps - two different samplers' budgets. Handing
      // one to the other ran AlphaFold 3 at 60 because RFD4 runs 60. The
      // validator takes the fold page's own default.
      status: (text) => say(candidates === 1 ? text
        : `candidate ${candidate + 1}/${candidates} · ${text}`),
      signal: controller.signal,
    });

    const scored = { ...result, validation, candidate };
    if (best === null || rankOf(scored) > rankOf(best)) best = scored;
    drawGenerated(scored, candidates);
    generatedRun.push(scored);
    paintDownloads();
    }

    if (best === null) throw new Error("no candidate completed");
    const result = best;
    say(`done · ${result.tokens} tokens over ${result.steps} steps`
      + (candidates === 1 ? "" : `, best of ${candidates}`)
      + (result.validation?.described === undefined ? ""
        : ` · ${result.validation.described}`));
    progress.value = 1;
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
    huntLabel.textContent = method.value === "hunter" ? "Hunt" : "Generate";
  }
};

huntButton.addEventListener("click", () => {
  if (running !== null) {
    running.abort();
    return;
  }
  if (method.value === "hunter") startHunt();
  else startGenerate();
});

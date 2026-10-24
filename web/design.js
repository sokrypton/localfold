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
import { chainsOf, contigOf, createContigList } from "./contig-ui.js";
import { generate } from "./generate.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";
import { CHAIN_IDS, MAX_ALANINE } from "../webgpu/design/hunter-loop.js";

// 🔴 THE SAME VALUES AS THE .role-* SWATCHES IN web/localfold.css. Two places,
// because one is a stylesheet and the other reaches a WebGL renderer that
// cannot read a CSS class - so they are written out here with the rule that
// they have to match, and the page gate compares them.
export const ROLE_COLOURS = {
  free: "#ffffff",
  fixed: "#6b7280",
  tied: "#3b82f6",
  motif: "#a855f7",
  target: "#e5e7eb",
};
import { hunt } from "./hunt.js";
import { createStructureViewer } from "./viewer.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

/**
 * The two counts `chooseDesigner` reads, from the spans.
 *
 * 🔴 NUCLEIC IS ALWAYS ZERO FOR NOW, and that is a gap rather than a fact: a
 * span does not yet say whether a given chain is protein or DNA, so a nucleic
 * target cannot be expressed and NA-MPNN can only be chosen by hand. Marking
 * the span's polymer type is the next piece of the editor.
 */
const contentsOf = (spans) => ({
  ligands: spans.filter((span) => span.kind === "ligand").length,
  nucleic: 0,
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
const methodNote = element("method-note");
const status = element("status-message");
const progress = element("progress");
const huntButton = element("hunt");
const resultsBox = element("results");
const resultsBody = element("results-body");
const trackBox = element("cycle-track-container");
const track = element("cycle-track");

let running = null;

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
const paintJob = () => {
  const spans = spanList?.read() ?? [];
  const hunter = method.value === "hunter";
  const { contig, problems, tokens } = contigOf(spans, method.value);
  const span = tokens.min === tokens.max ? `${tokens.min}` : `${tokens.min}\u2013${tokens.max}`;

  // 🔴 EACH METHOD IS SHOWN WHAT IT WILL ACTUALLY BE ASKED. A contig is RFD4's
  // notation, and under Hunter it is actively misleading - a given SEQUENCE has
  // no contig notation, so the line read "/0,75-75" for a job whose target is
  // 114 residues long. Hunter is told about its chains instead.
  if (hunter) {
    const { chains, chainIndex, length } = chainsOf(spans);
    const described = chains.map((chain, index) => (index === chainIndex
      ? `design ${length}` : `given ${chain.length}`)).join("  ·  ");
    contigLine.textContent = chains.length === 0 ? "" : described;
  } else {
    contigLine.textContent = contig.length === 0 ? "" : `${contig}    (${span} tokens)`;
  }

  const theirs = hunter ? chainsOf(spans).problems : problems;
  contigProblems.textContent = theirs.join(" · ");
  contigProblems.classList.toggle("warn", theirs.length > 0);
  paintDesignerNote(spans);
};

window.__designSpans = spanList;

designerSelect.addEventListener("change", () => paintDesignerNote(spanList.read()));
folders.addEventListener("change", paintFolderNote);
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

const paintMethod = () => {
  const hunter = method.value === "hunter";
  methodNote.textContent = hunter
    ? "Folds a mostly-X sequence, redesigns on the hallucinated backbone, and re-folds. Needs a folding model."
    : "Generates all-atom structure by flow matching, in one pass. Its weights are ~420 MB and"
      + " download on the first generation.";
  methodNote.classList.toggle("warn", !hunter);
  // 🔴 THE BUTTON SAYS WHICH MACHINE IT STARTS. "Hunt" is the loop; RFD4
  // generates in one pass and calling that a hunt would misdescribe it.
  if (running === null) huntButton.textContent = hunter ? "Hunt" : "Generate";
  paintJob();
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

  // 🔴 PUSHED BEFORE THE ROW IS DRAWN would be wrong: the viewer is built on
  // its FIRST frame, and orienting it on that frame is what gives every later
  // one a sensible camera. So push, then orient once, then draw.
  viewerBox.style.display = "";
  viewer.push(record.pdb, (built) => { if (built) viewer.orient(); });
  // 🔴 COLOURED BY ROLE, NOT BY CONFIDENCE. The page's job is saying what is
  // being DECIDED, and pLDDT answers a different question - so the designed
  // chain takes the "free" colour and everything else the "target" one, the
  // same two swatches the legend names. These are the only two roles a Hunter
  // run has today; fixed, tied and motif arrive with the RunSpec's positions
  // reaching the page.
  viewer.colourChains({
    [record.chain]: ROLE_COLOURS.free,
    ...Object.fromEntries(CHAIN_IDS.split("").filter((id) => id !== record.chain)
      .map((id) => [id, ROLE_COLOURS.target])),
  });

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
  onFail: (message) => say(`the viewer could not start: ${message}`, true),
});

const startHunt = async () => {
  const spans = spanList.read();
  // 🔴 THE SPANS ARE THE JOB; `chainsOf` is how Hunter reads them. It also says
  // what Hunter cannot express - a motif inside a designed chain, two designed
  // chains - and a hunt that cannot express the job must not start.
  const { chains, chainIndex, length, problems } = chainsOf(spans);
  if (problems.length > 0) {
    say(problems.join(" · "), true);
    return;
  }
  const controller = new AbortController();
  running = controller;
  huntButton.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  // A new hunt is a new object: the atom count changes with the binder's
  // length, so frames cannot be shared with the last one.
  viewer.cancelAnimations();
  viewer.reset();
  say("starting");
  progress.value = 0;
  progress.dataset.state = "running";

  try {
    const { best } = await hunt({
      chains,
      chainIndex,
      folders: Array.from(folders.selectedOptions, (option) => option.value),
      designer: designerSelect.value,
      contents: contentsOf(spans),
      runs: number("runs", 1),
      cycles: number("cycles", 5),
      length,
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
    huntButton.textContent = method.value === "hunter" ? "Hunt" : "Generate";
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
const presetSelect = element("preset");
const presetNote = element("preset-note");

for (const preset of RFD4P_PRESETS) {
  const option = document.createElement("option");
  option.value = preset.id;
  option.textContent = preset.title ?? preset.id;
  presetSelect.append(option);
}

/** One contig string back into spans, or null for a piece this editor has no row for. */
export const spansFromContig = (contig) => {
  const spans = [];
  for (const raw of String(contig).split(",")) {
    const piece = raw.trim();
    if (piece.length === 0) continue;
    if (piece === "/0" || /^\/\d+$/.test(piece)) { spans.push({ kind: "break" }); continue; }
    const segment = /^([A-Za-z][A-Za-z0-9]*)(\d+)-(\d+)$/.exec(piece);
    if (segment !== null) {
      spans.push({ kind: "segment", chain: segment[1],
        first: Number(segment[2]), last: Number(segment[3]) });
      continue;
    }
    const generated = /^(\d+)-(\d+)$/.exec(piece);
    if (generated !== null) {
      spans.push({ kind: "generated", min: Number(generated[1]), max: Number(generated[2]) });
      continue;
    }
    if (/^\d+$/.test(piece)) {
      spans.push({ kind: "generated", min: Number(piece), max: Number(piece) });
      continue;
    }
    // 🔴 A LABELLED SPAN LIKE `h1:16-16P` IS NOT SOMETHING THIS EDITOR HAS A ROW
    // FOR. It carries a name and a polymer type and drives the CATH and
    // secondary-structure conditions, which is a whole feature; saying so beats
    // loading a job that silently lost them.
    if (/^[A-Za-z0-9]+:/.test(piece)) return null;
    if (/^[A-Z0-9]{1,5}$/.test(piece)) { spans.push({ kind: "ligand", code: piece }); continue; }
    return null;
  }
  return spans;
};

let presetConditions;
let presetCif;

presetSelect.addEventListener("change", () => {
  const preset = RFD4P_PRESETS.find((entry) => entry.id === presetSelect.value);
  if (preset === undefined) {
    presetConditions = undefined;
    presetCif = undefined;
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
  presetConditions = preset.conditions;
  presetCif = preset.cifUrl;
  if (preset.nsteps !== undefined) element("steps").value = String(preset.nsteps);
  spanList.set(spans);
  // 🔴 A PRESET IS RFD4's, so loading one picks that method rather than leaving
  // a contig the selected method cannot run.
  method.value = "rfd4p";
  paintMethod();
  presetNote.textContent = preset.description ?? "";
  presetNote.classList.remove("warn");
});

/** RFD4-Proteina: one generation, straight into the viewer. */
const startGenerate = async () => {
  const spans = spanList.read();
  const { contig, problems } = contigOf(spans, "rfd4p");
  if (problems.length > 0) {
    say(problems.join(" · "), true);
    return;
  }
  const controller = new AbortController();
  running = controller;
  huntButton.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  viewer.cancelAnimations();
  viewer.reset();
  progress.value = 0;
  progress.dataset.state = "running";
  say("starting");

  try {
    const result = await generate({
      contig,
      conditions: presetConditions,
      cifUrl: presetCif,
      steps: number("steps", 60),
      seed: number("seed", 0),
      signal: controller.signal,
      status: (text) => say(text),
      progress: (fraction) => { progress.value = fraction; },
    });
    viewerBox.style.display = "";
    viewer.push(result.pdb, (built) => { if (built) viewer.orient(); });
    resultsBox.style.display = "";
    const row = document.createElement("tr");
    row.className = "best";
    for (const text of ["1", "\u2014", "rfd4-proteina", `${result.tokens} tokens`,
      `${result.steps} steps`]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    resultsBody.append(row);
    say(`done · ${result.tokens} tokens over ${result.steps} steps`);
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
    huntButton.textContent = method.value === "hunter" ? "Hunt" : "Generate";
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

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
import { chainsOf, conditionsOf, contigOf, createContigList, hotspotsOf } from "./contig-ui.js";
import { generate } from "./generate.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { THREE_TO_ONE } from "../cpu/design/mpnn/constants.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
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
const methodNote = element("method-note");
const status = element("status-message");
const progress = element("progress");
const huntButton = element("hunt");
const resultsBox = element("results");
const resultsBody = element("results-body");
const trackBox = element("cycle-track-container");
const track = element("cycle-track");

// 🔴 EVERY PIECE OF MUTABLE STATE A PAINT FUNCTION READS LIVES HERE TOO, for the
// same reason as the lookups above: `paintJob` runs while the span list is still
// constructing, and it reaches the run state and the loaded structure. Five
// bindings on this page have thrown "Cannot access X before initialization" by
// being declared below the callback that reads them. The rule that actually
// holds is: shared state at the top, helpers as function DECLARATIONS.
let running = null;
let loaded = null;

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
    const { chains, chainIndex, length } = chainsOf(spans, { sequenceOf: resolver() });
    const { hold } = chainsOf(spans, { sequenceOf: resolver() });
    const described = chains.map((chain, index) => {
      if (index !== chainIndex) return `given ${chain.length}`;
      // 🔴 THE DESIGNED CHAIN IS HANDED OVER EMPTY when every residue of it comes
      // from a Design span - that is what makes the loop draw its own opening
      // sequence - so its SIZE is the spans' total and not the string's length,
      // which is zero. Reading the string here printed "design 0".
      const total = chain.length === 0 ? length : chain.length;
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
  const { chains, chainKinds, ligandCodes, modifications, chainIndex, length, hold, problems }
    = chainsOf(spans, { sequenceOf: resolver() });
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
    const { best, cycles } = await hunt({
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
    // A labelled span - `h1:16-16P` - carries a name and a polymer type, and the
    // name is the handle a secondary-structure or CATH condition selects it by.
    const labelled = /^([A-Za-z][A-Za-z0-9]*):(\d+)-(\d+)([PRD])?$/.exec(piece);
    if (labelled !== null) {
      spans.push({
        kind: "generated", name: labelled[1],
        min: Number(labelled[2]), max: Number(labelled[3]),
        polymer: { P: "protein", R: "rna", D: "dna" }[labelled[4] ?? "P"],
        // 🔴 THE SECONDARY STRUCTURE IS RECOVERED FROM THE CONDITIONS, NOT THE
        // CONTIG, because the contig carries only the name. Filled in below.
        secondary: "",
      });
      continue;
    }
    if (/^[A-Za-z0-9]+:/.test(piece)) return null;
    if (/^[A-Z0-9]{1,5}$/.test(piece)) { spans.push({ kind: "ligand", code: piece }); continue; }
    return null;
  }
  return spans;
};

// What a preset carries beyond its contig: the conditions, the template
// structure, and the per-modality options that make partial diffusion and
// inverse folding what they are.
let presetExtras = {};

presetSelect.addEventListener("change", () => {
  const preset = RFD4P_PRESETS.find((entry) => entry.id === presetSelect.value);
  if (preset === undefined) {
    presetExtras = {};
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
    partialTrajectoryFractionSteps: preset.partialTrajectoryFractionSteps,
    inverseFolding: preset.inverseFolding,
  };
  if (preset.nsteps !== undefined) element("steps").value = String(preset.nsteps);
  spanList.set(spans);
  // 🔴 A PRESET IS RFD4's, so loading one picks that method rather than leaving
  // a contig the selected method cannot run.
  method.value = "rfd4p";
  paintMethod();
  presetNote.textContent = preset.description ?? "";
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

const takeStructure = (text, name) => {
  // 🔴 PARSED ON LOAD, NOT ON RUN. A file that is not mmCIF, or an entry with no
  // atom records, is a thing to say now - while the user is looking at the
  // control they just used - rather than minutes later inside a generation.
  const atoms = parseCifToAtomArray(text);
  const chains = chainRanges(atoms);
  if (chains.size === 0) throw new Error("no atoms in that file");
  loaded = { text, name, atoms, chains };
  describeStructure();
  // Offer its first chain as a span, since that is almost always the next move.
  const [first] = [...chains.entries()];
  const spans = spanList.read();
  if (!spans.some((span) => span.kind === "segment")) {
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
    const bytes = new Uint8Array(await file.arrayBuffer());
    const zstd = bytes[0] === 0x28 && bytes[1] === 0xb5 && bytes[2] === 0x2f && bytes[3] === 0xfd;
    takeStructure(new TextDecoder().decode(zstd ? decompress(bytes) : bytes), file.name);
  } catch (error) {
    structureNote.textContent = `could not read ${file.name}: ${error.message}`;
    structureNote.classList.add("warn");
  }
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
      ...presetExtras,
      // 🔴 HOTSPOTS NAMED IN THE SPANS WIN OVER A PRESET'S, because they are
      // what the user just typed. A preset's other conditions stay.
      ...(conditionsOf(spans) === undefined ? {} : {
        conditions: { ...presetExtras.conditions, ...conditionsOf(spans) },
      }),
      // A structure the user loaded is the one they mean, over a preset's.
      ...(loaded === null ? {} : { inputCif: loaded.text }),
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

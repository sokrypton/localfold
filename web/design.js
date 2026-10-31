/**
 * design.js
 * ---------
 * AI Context: THE DESIGN PAGE - Protein Hunter's loop (the default) and RFD4-Proteina's generation
 *
 * The page's five steps, in design.html's order: the method and its models, an example to start
 * from, the segments (one card each, web/contig-ui.js), what is selected in the viewer, and the run.
 *
 * 🔴 WHY THE AUTOMATIC DESIGNER WRITES ITS REASON OUT. A picker that silently
 * changes its own value is worse than one that cannot - `cpu/design/designers.js`
 * says so in those words, and returns a `why` for exactly this. The note under
 * the picker is that `why`.
 *
 * See docs/DESIGN.md for the plan this page is step one of.
 */
import {
  chainColour, chainLetters, chainOfSpans, chainsOf, CONDITION_COLOURS, conditionsOf, contigOf, createContigList,
  drawChainDiagram, drawLengths, HOTSPOT_COLOUR, hotspotsOf, INPUT_COLOUR, LIGAND_COLOUR, shade,
} from "./contig-ui.js";
import { DEFAULT_STEPS, generate } from "./generate.js";
import { writeZip } from "./zip.js";
import { refoldDesign } from "./validate-design.js";
import { getDevice } from "./model.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { THREE_TO_ONE } from "../cpu/design/mpnn/constants.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";
import { CHAIN_IDS, MAX_ALANINE } from "../webgpu/design/hunter-loop.js";

import { hunt } from "./hunt.js";
import { search } from "./search.js";
import { protocolFor } from "../cpu/design/protocol.js";
import { heldCoordinates } from "../cpu/design/motif.js";
import { createStructureViewer } from "./viewer.js";
import {
  createConditionEditor, inputResidueMap, inputStructureAtoms, picksOf, residueRanges, segmentsOf,
} from "./condition-editor.js";
import { sketchJob } from "./job-sketch.js";
import { exampleSpans } from "./example-spans.js";
import { toPdbString } from "../shared/rfd4p/output.js";
import { alignToReference } from "../cpu/design/align.js";
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
/** Deoxy- and ribonucleotides by their base, and which polymer each is. */
const NUCLEOTIDE = { DA: "A", DC: "C", DG: "G", DT: "T", A: "A", C: "C", G: "G", U: "U" };
const DEOXY = new Set(["DA", "DC", "DG", "DT"]);

function contentsOf(spans) {
  // what the designed chain is made of: a Designed segment's own, or a given one marked Redesign
  const designed = spans.find((span) => span.kind === "generated" || span.redesign === true);
  return {
    designs: designed === undefined ? "protein" : designed.kind === "segment"
      ? polymerOf(String(designed.chain ?? "")) : designed.polymer ?? "protein",
    ligands: spans.filter((span) => span.kind === "ligand").length,
    // a kept strand of a loaded structure is nucleic too (the DNA-binder example's two)
    nucleic: spans.filter((span) => (span.kind === "sequence" && NUCLEIC.has(span.polymer ?? ""))
      || (span.kind === "segment" && NUCLEIC.has(polymerOf(String(span.chain ?? ""))))).length,
  };
}

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
const methodNote = element("method-note");

/** Results' columns, by method: a hunt's cycles, or a generation's designs. */
function paintResultsHead() {
  const names = method.value === "hunter" ? ["Run", "Cycle", "Folded by", "Objective", "Moved", "Validity"]
    : ["Design", "Seed", "Size", "Refold"];
  element("results-head").replaceChildren(...names.map((name) => {
    const th = document.createElement("th");
    th.textContent = name;
    return th;
  }));
}

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
// What a loaded example carries beyond its contig (its conditions, structure and modality), and
// the spans it set as JSON - what `warnIgnoredEdit` compares an edit against.
let presetExtras = {};
let presetSpans = null;
// 🔴 THE VIEWER SHOWS THE LOADED STRUCTURE UNTIL A RUN TAKES IT, and only then can a click name one
// of its residues: a design's positions are not the input's. Set while it does.
let inputView = null;
// 🔴 FALSE UNTIL THE VIEWER EXISTS: `paintJob` runs while the span list constructs, which is before
// the viewer is made, and a redraw it asked for then would read a `const` still in its dead zone.
let viewerReady = false;
let jobRedraw = 0;
// 🔴 THE RUN'S PICTURE, KEPT so the refold can take the viewer and give it back: every frame pushed
// (`pushRunFrame`), and the paint of the last. `designs` are the finished ones the Refold step
// offers; `refolding` is a refold in flight, and `refoldShown` says the viewer holds one.
let runFrames = [];
let runPaint = () => {};
let designs = [];
let refolding = null;
let refoldShown = false;
// a refold residue where it sits on the design, and where it has moved REFOLD_FAR_AT or more
const REFOLD_NEAR = "#94a3b8";
const REFOLD_FAR = "#dc2626";
const REFOLD_FAR_AT = 4;
// the refold's viewer object, drawn over the design's (drawRefold)
const REFOLD_OBJECT = "refold";
// 🔴 A FUNCTION DECLARATION, called while the span list constructs.
function BLANK_JOB() { return [{ kind: "generated", min: 100, max: 100, name: "", secondary: "", polymer: "protein" }]; }

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
const searchSelect = element("search");
const holdSelect = element("hold-coordinates");
const searchNote = element("search-note");

/**
 * What the Designer picker says underneath itself.
 *
 * On Auto this is the rule's own reason. On an explicit choice it is the
 * checkpoint's description, plus a warning where that choice cannot see part of
 * the target - picking ProteinMPNN against a ligand is legal and loses the
 * ligand, and the page should say which.
 */
/**
 * Which recipe a search would run for this job, and why - or why none applies. The recipe's own numbers replace
 * Cycles, Temperature, Omit and the designer, and the note says so, because a control the run then ignores is a
 * control that lies.
 */
const searchRecipe = (spans) => {
  const { chains, ligandCodes, chainIndex, structureParts } = chainsOf(spans, readers());
  const motif = structureParts.filter((part) => part.held && part.index === chainIndex)
    .reduce((n, part) => n + part.last - part.first + 1, 0);
  return protocolFor({ ...contentsOf(spans), partners: Math.max(0, chains.length - 1) + (ligandCodes?.length ?? 0),
    motif, fixed: holdSelect.value === "held" });
};
/**
 * What Held coordinates holds for one run's spans: null when Free. 🔴 REFUSED, NOT SKIPPED, when there is nothing to
 * hold - a reader who asked for the DNA held and has no structure loaded gets a sentence, not a free fold.
 */
const heldFor = (job) => {
  if (holdSelect.value !== "held") return null;
  if (loaded === null) throw new Error("Held coordinates needs a loaded structure to read them from");
  const held = heldCoordinates({ atoms: loaded.atoms, chains: job.chains, chainIndex: job.chainIndex,
    ligandCodes: job.ligandCodes, structureParts: job.structureParts });
  if (held.atoms(job.length).length < 3) {
    throw new Error("Held coordinates: the loaded structure has no ligand, kept DNA or motif residue of this job to hold");
  }
  return held;
};
const paintSearchNote = (spans) => {
  const on = method.value === "hunter" && searchSelect.value === "recipe";
  searchNote.hidden = !on;
  if (!on) return;
  const { protocol, why } = searchRecipe(spans);
  searchNote.classList.toggle("warn", protocol === null);
  searchNote.textContent = protocol === null ? `No benchmark recipe fits this job: ${why}.`
    : `The ${{ ppi: "protein-binder", sm: "small-molecule", dna: "DNA", ame: "enzyme (AME)" }[protocol.criterion]}`
      + ` recipe (${protocol.tag}), because ${why}: up to ${protocol.restarts} restarts of ${protocol.cycles} cycles`
      + `${protocol.guide ? `, every design fold held to the structure${protocol.guideUntil === undefined ? ""
        : ` until its last steps`}` : ""}${protocol.probe ? ", each design refolded unguided and its motif found in it" : ""},`
      + ` the best cycles refolded at 200 steps as a check${protocol.polish === undefined ? "" : ", near misses polished"}`
      + `${protocol.climb === undefined ? "" : ", a climb on the motif when nothing passes"}.`
      + " Its own cycles, temperature, omitted letters and designer replace the ones set here.";
};

const paintDesignerNote = (spans) => {
  const contents = contentsOf(spans);
  if (method.value !== "hunter") { designerNote.textContent = ""; return; }
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

/** A distance setting in angstroms, or null when it is blank - the reference's method. */
function angstromsOf(id, label) {
  const text = document.getElementById(id)?.value.trim() ?? "";
  if (text === "") return null;
  const value = Number(text);
  if (!(value > 0)) throw new Error(`${label} must be a distance in angstroms, not ${text}`);
  return value;
}
/** The "Steer to hotspots" strength, or null. */
function steerWeight() {
  const text = document.getElementById("steer")?.value.trim() ?? "";
  if (text === "") return null;
  const value = Number(text);
  if (!(value > 0)) throw new Error(`Steer to hotspots must be a positive strength, not ${text}`);
  return value;
}
/** The "Stay within" distance, or null. */
function stayWithin() { return angstromsOf("stay-within", "Stay within"); }

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
  const within = stayWithin();
  const warm = angstromsOf("warm-start", "Warm start");
  // 🔴 "THE REFERENCE'S METHOD" ONLY WHEN IT IS ONE: a warm start (on by default) is not
  const reference = warm === null && within === null;
  folderNote.textContent = (within !== null
    ? `Up to ${folds} folds: up to ${sequences} sequences a cycle, keeping the first that stays within`
      + ` ${within} Å of the cycle before - it stops sooner when one does.`
    : sequences === 1
      ? `${folds} folds: one sequence a cycle${reference ? ", the reference's method" : ""}.`
      : `${folds} folds: ${sequences} sequences a cycle, and the best carries forward.`)
    + (warm === null ? "" : ` Each fold starts from the cycle before, ${warm} Å of noise down.`);
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
  // 🔴 THE BLANK JOB ON ARRIVAL: a new 100-residue protein, which a newcomer can read at a glance
  // and run as it is. It was a 114-residue target sequence nobody recognised and a binder.
  initial: BLANK_JOB(),
  method: () => method.value,
  onChange: () => paintJob(),
  hotspotsFromSelection: (span) => selectedWithin(span),
  claimed: (spans) => ligandChains(spans),
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
  const { chainIndex } = chainsOf(spans, readers());
  const entry = (colour, text) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "role-swatch";
    swatch.style.setProperty("--chain-colour", colour);
    item.append(swatch, text);
    return item;
  };
  // the letters the chosen method's output has (chainLetters), as the cards and the diagram say
  const letters = chainLetters(spans, method.value, ligandChains(spans));
  const items = chains.map((at) => entry(chainColour(at),
    `chain ${letters[at] ?? "?"}${at === chainIndex ? " — designed" : ""}`));
  // Only worth a line when there is something held to explain.
  if (held.length > 0 && chainIndex >= 0) {
    items.push(entry(shade(chainColour(chainIndex), "#374151", 0.55), "held"));
  }
  if (spans.some((span) => span.kind === "ligand")) {
    items.push(entry(LIGAND_COLOUR, "ligand"));
  }
  roleList.replaceChildren(...items);
}

/**
 * 🔴 EVERY HOTSPOT THE JOB HAS: a kept segment's hotspot field - where an example's file puts its own
 * (web/example-spans.js), and where Mark as hotspot writes - and any Hotspot condition row added under
 * More constraints. One place to read them, so the viewer, the diagram and the run agree.
 * @returns {{chain: string, resId: number, from: string}[]}
 */
function hotspotResidues(spans) {
  const out = new Map();
  const add = (chain, resId, from) => {
    const key = `${chain}:${resId}`;
    if (!out.has(key)) out.set(key, { chain, resId, from });
  };
  for (const span of spans) {
    if (span.kind !== "segment") continue;
    const chain = String(span.chain ?? "").trim();
    for (const piece of String(span.hotspots ?? "").split(/[^0-9-]+/).filter(Boolean)) {
      const [first, last = first] = piece.split(/(?<=\d)-/).map(Number);
      for (let r = first; r <= last; r += 1) add(chain, r, "segment");
    }
  }
  for (const row of conditionEditor?.rows ?? []) {
    if (row.key !== "C_HOT" || row.value !== "true") continue;
    for (const pick of row.picks) add(pick.chain, pick.resId, "constraint");
  }
  return [...out.values()];
}

/**
 * The loaded structure's chains that hold one of the job's ligands - letters RFD4-Proteina keeps.
 * 🔴 A FUNCTION DECLARATION, read while the span list constructs.
 */
function ligandChains(spans) {
  if (loaded === null) return [];
  const codes = new Set(spans.filter((span) => span.kind === "ligand").map((span) => String(span.code).toUpperCase()));
  const out = new Set();
  for (let i = 0; i < loaded.atoms.chain_id.length; i += 1) {
    if (codes.has(String(loaded.atoms.res_name[i]).trim().toUpperCase())) out.add(String(loaded.atoms.chain_id[i]));
  }
  return [...out];
}

/**
 * A generation's frame in the job's colours: each chain the colour its card and the sketch gave it.
 *
 * 🔴 IT WAS LEFT TO py2Dmol, whose palette goes by the output's own letters - and RFD4-Proteina
 * letters chains its own way (`chainLetters`), so the 5O45 binder's design was drawn in its target's
 * colour and the target in the design's. 🔴 AND A CHAIN THE RULE DID NOT PREDICT IS AN ERROR, not
 * a colour guessed: it means the page's account of the output is wrong.
 */
function paintGenerated(pdb) {
  const spans = spanList.read();
  const claimed = ligandChains(spans);
  const letters = chainLetters(spans, "rfd4p", claimed);
  const polymer = new Set();
  for (const line of pdb.split("\n")) {
    if (line.startsWith("ATOM  ")) polymer.add(line.slice(21, 22).trim());
  }
  const unknown = [...polymer].filter((letter) => !letters.includes(letter) && !claimed.includes(letter));
  if (unknown.length > 0) {
    throw new Error(`RFD4-Proteina wrote chain ${unknown.join(", ")}, which the job's lettering`
      + ` (${letters.join(", ")}) does not have`);
  }
  viewer.colourChains(Object.fromEntries(letters.map((letter, at) => [letter, chainColour(at)])));
}

/** What chainsOf reads off the loaded structure: a segment's sequence, and what its chain is made of. */
function readers() {
  return { sequenceOf: resolver(), polymerOf };
}

const paintJob = () => {
  const spans = spanList?.read() ?? [];
  paintLegend(spans);
  drawChainDiagram(element("chain-diagram"), spans, { onPick: (index) => spanList.reveal(index),
    hotspots: hotspotResidues(spans), selected: spanList?.selected ?? -1,
    letters: chainLetters(spans, method.value, ligandChains(spans)) });
  const hunter = method.value === "hunter";
  const { contig, problems, tokens } = contigOf(spans, method.value);
  const span = tokens.min === tokens.max ? `${tokens.min}` : `${tokens.min}\u2013${tokens.max}`;

  // 🔴 EACH METHOD IS SHOWN WHAT IT WILL ACTUALLY BE ASKED. A contig is RFD4's
  // notation, and under Hunter it is actively misleading - a given SEQUENCE has
  // no contig notation, so the line read "/0,75-75" for a job whose target is
  // 114 residues long. Hunter is told about its chains instead.
  if (hunter) {
    const { chains, chainIndex, length } = chainsOf(spans, readers());
    const { hold } = chainsOf(spans, readers());
    // 🔴 A RANGE IS REPORTED AS A RANGE. Hunter resolves one length per run now,
    // and `chainsOf` takes the minimum - so a contig asking for 5-20 residues
    // described itself as "design 5", which is one draw out of sixteen and not
    // what the job says. Both ends are taken by pinning the spans each way.
    const pinned = (end) => chainsOf(spans.map((span) => (span.kind === "generated"
      ? { ...span, min: span[end], max: span[end] } : span)),
    readers()).length;
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
    // The contig is the expert's view; the count is what a newcomer reads.
    contigLine.textContent = contig.length === 0 ? "" : `${span} residues`;
    contigLine.title = contig;
  }

  const theirs = hunter ? chainsOf(spans, readers()).problems : problems;
  contigProblems.textContent = theirs.join(" · ");
  contigProblems.classList.toggle("warn", theirs.length > 0);
  paintDesignerNote(spans);
  paintSearchNote(spans);
  warnIgnoredEdit(spans);
  scheduleJobView();
};

window.__designSpans = spanList;

designerSelect.addEventListener("change", () => paintDesignerNote(spanList.read()));
searchSelect.addEventListener("change", () => paintSearchNote(spanList.read()));
holdSelect.addEventListener("change", () => paintSearchNote(spanList.read()));
// A random seed, drawn on request rather than behind the reader's back: the box shows it, so the
// same design can be made again.
element("random-seed").addEventListener("click", () => {
  const draw = new Uint32Array(1);
  crypto.getRandomValues(draw);
  element("seed").value = String(draw[0] % 1000000);
  say(`seed ${element("seed").value} - press Design for a new design`);
});
for (const id of ["beam", "cycles", "runs", "stay-within", "warm-start"]) element(id).addEventListener("input", paintCost);
paintCost();
paintAdvanced();
// 🔴 AN ARROW, NOT THE FUNCTION ITSELF: `paintMethod` is declared below this
// line, and handing the binding over directly reads it before it is
// initialised. Deferring the lookup to call time is the whole fix.
folders.addEventListener("change", () => paintMethod());

/**
 * WHICH CONTROLS THE CHOSEN METHOD ACTUALLY HONOURS.
 *
 * 🔴 HIDDEN NOW, NOT GREYED WITH A REASON (2026-10-27). Greying every control the other method
 * uses, with a sentence saying why, was meant to teach what the two methods are; to a newcomer it
 * read as half the page being broken. The method is an Advanced choice now, so the controls follow
 * it and nothing explains an absence. The reasons stay as each group's tooltip.
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
    searchGroup: "The search wraps Protein Hunter's hunt; RFD4-Proteina generates once.",
    holdGroup: "RFD4-Proteina keeps a structure's coordinates by its own conditioning.",
    stayGroup: "RFD4-Proteina has no cycle to hold to the one before.",
    warmGroup: "RFD4-Proteina has no previous cycle to start a fold from.",
    steerGroup: "RFD4-Proteina conditions on hotspots itself.",
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
  hunter: "Ready. Press Design to start the hunt.",
  rfd4p: "Ready. The first run downloads about 420 MB of model weights.",
};

/**
 * The tuning knobs, folded away until asked for.
 *
 * 🔴 ELEVEN CONTROLS IS A FORM, NOT A SETTINGS STRIP. The five behind this are
 * the ones nobody changes on a first run; the six left describe the job.
 */
function paintAdvanced() {
  const open = advancedButton.getAttribute("aria-expanded") === "true";
  element("advancedOptions").hidden = !open;
  advancedButton.textContent = open ? "Hide advanced" : "Advanced";
  // Protein Hunter's two notes are about its settings, so they live with them.
  for (const id of ["designer-note", "folder-note"]) element(id).hidden = method.value !== "hunter";
}

advancedButton.addEventListener("click", () => {
  const open = advancedButton.getAttribute("aria-expanded") === "true";
  advancedButton.setAttribute("aria-expanded", open ? "false" : "true");
  paintAdvanced();
});

/** What each method is, in one sentence under the picker. */
const METHOD_NOTES = {
  hunter: "Folds a sequence, redesigns it and folds it again, cycle by cycle - you watch the design"
    + " converge. The folding model's confidence is what it optimises.",
  rfd4p: "Generates the structure directly, conditioned on what you keep and constrain."
    + " The first run downloads about 420 MB of weights.",
};

function paintMethod() {
  const hunter = method.value === "hunter";
  // 🔴 ONE CONTROL, TWO MEANINGS, AND THE LABEL SAYS WHICH. Both methods sample a structure over
  // steps; a hunt's are the fold's diffusion steps and RFD4's are its flow steps.
  stepsTitle.textContent = hunter ? "Diffusion steps" : "Flow steps";
  // 🔴 AND ONE DEFAULT EACH: upstream samples 60 flow steps, the fold 25. A value the user typed is
  // theirs; only the other method's untouched default is swapped.
  const steps = element("steps");
  const mine = hunter ? HUNTER_STEPS : RFD4_STEPS;
  const theirs = hunter ? RFD4_STEPS : HUNTER_STEPS;
  if (Number(steps.value) === theirs) steps.value = String(mine);
  methodNote.textContent = METHOD_NOTES[method.value] ?? "";
  if (running === null) say(RESTING[method.value] ?? "");
  // The examples are the method's: four of the ten are RFD4-only.
  paintPresets();
  // the cards draw only the controls the method uses (secondary structure is RFD4-Proteina's)
  spanList.repaint();
  paintJob();
  paintCost();
  // 🔴 HIDDEN, FROM ONE MAP: every group the method cannot use, and nothing explains the absence.
  const unusable = UNUSED_BY[method.value] ?? {};
  conditionEditor?.disable(hunter ? HUNTER_NO_CONDITIONS : null);
  for (const id of new Set([...Object.keys(UNUSED_BY.hunter), ...Object.keys(UNUSED_BY.rfd4p)])) {
    const group = document.getElementById(id);
    if (group === null) continue;
    const why = unusable[id];
    group.classList.toggle("not-this-method", why !== undefined);
    if (why !== undefined) group.dataset.why = why;
  }
  unusedNote.textContent = "";
  // The selection step: keep and aim under either method, and constrain under RFD4-Proteina.
  element("constraints-title").textContent = hunter ? "Selection" : "Selection and constraints";
  // The Designs count is a hunt's sequences a cycle.
  element("beam-title").textContent = hunter ? "Sequences" : "Designs";
  paintResultsHead();
  paintAdvanced();
  paintSelectionStep();
}

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
  sizeStage();

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
  // ...and for a DNA or RNA chain, which has no alanine ceiling (webgpu/design/hunter-loop.js), its GC
  // share - a number about the strand rather than an alanine count of its adenines
  const alanine = record.alanine === null
    ? `${Math.round(100 * [...record.sequence].filter((base) => base === "G" || base === "C").length
      / Math.max(1, record.sequence.length))}% GC`
    : `${(record.alanine * 100).toFixed(0)}% Ala`;
  const overAlanine = record.cycle > 0 && record.alanine !== null && record.alanine > MAX_ALANINE;
  // 🔴 THE VETO'S OWN WORDS, WHICH ALREADY CARRY ITS MEASUREMENT - "touches 0/4
  // hotspots (nearest 23.7 A, needs 8 A)". Re-describing it here would be a
  // second place for the number to go stale.
  const vetoed = overAlanine || typeof record.vetoed === "string";
  if (vetoed) row.classList.add("rejected");
  // How far the cycle's designed chain moved from the previous cycle's, and - holding a run to its
  // history - how many sequences it took and whether any stayed within (web/hunt.js `accept`)
  const moved = !Number.isFinite(record.drift) ? "—"
    : `${record.drift.toFixed(1)} Å`
      + (record.drawn > 1 ? ` · ${record.drawn} tries` : "")
      + (record.accepted === false ? " · none within" : "");
  for (const [text, className] of [
    [String(record.run + 1), ""], [String(record.cycle), ""],
    [record.folder ?? "—", ""], [objective, ""], [moved, ""],
    [overAlanine ? `${alanine} — over the ${Math.round(MAX_ALANINE * 100)}% ceiling`
      : (record.vetoed ?? alanine), "validity"],
  ]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }
  row.dataset.key = `${record.run}:${record.cycle}`;
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
  if (reference === null) { beginRunPicture(); reference = record.pdb; }
  const placed = reference === record.pdb ? record.pdb
    : alignToReference(record.pdb, reference, record.chain);

  const paint = () => {
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
  };
  pushRunFrame(placed, paint);
  // a cycle a refold can check: scored, and not vetoed
  if (record.cycle > 0 && !vetoed) {
    designs.push({ label: `Run ${record.run + 1} · cycle ${record.cycle}`, pdb: placed, chain: record.chain,
      record, paint });
    paintRefoldStep();
  }

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
// 🔴 NOT `#viewer-container`: py2Dmol's website app (the full bundle's main.js) starts itself on any page
// that has one, and took this page's box for its own viewer - writing a 600px size into it and
// throwing on the controls it looked for next, on every load.
const viewerBox = element("design-viewer");
/** The viewer replaces its placeholder the first time anything is drawn. */
function showViewer() {
  viewerBox.style.display = "";
  element("viewer-empty").hidden = true;
  paintViewerBar();
}
const viewer = createStructureViewer({
  container: viewerBox,
  // the stylesheet sizes the box to the window (--viewer-height) and py2Dmol follows it; this is
  // only the first frame's size, asked near it so the first mesh is not built twice
  canvasHeight: Math.max(420, Math.round(window.innerHeight - 220)),
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
const conditionHost = element("condition-host");

/** Paint the drawn input: grey, with each condition row's residues in its own colour. */
/** A constraint's colour: hotspots their own, every other row the next of the palette. */
function constraintColour(index, row) {
  return row?.key === "C_HOT" ? HOTSPOT_COLOUR : CONDITION_COLOURS[index % CONDITION_COLOURS.length];
}

function paintInput() {
  if (inputView === null || viewer.renderer === undefined) return;
  const renderer = viewer.renderer;
  viewer.colourChains(Object.fromEntries([...inputView.chainOf.keys()].map((letter) => [letter, INPUT_COLOUR])));
  const map = inputResidueMap(renderer, inputView.chainOf, inputView.ghosts);
  const where = (test) => map.flatMap((residue, i) => (residue !== undefined && test(residue) ? [i] : []));
  // 🔴 WHAT STEP 3 KEEPS, IN ITS CHAIN'S COLOUR, so the picture says which residues of the structure
  // are in the design (the same colour as the piece's block and row); constraints paint over it.
  const spans = spanList.read();
  const chainOf = chainOfSpans(spans);
  spans.forEach((span, index) => {
    if (span.kind !== "segment" || chainOf[index] === null) return;
    viewer.colourResidues(where((residue) => residue.chain === String(span.chain).trim()
      && residue.resId >= Number(span.first) && residue.resId <= Number(span.last)), chainColour(chainOf[index]));
  });
  // ...and every piece still to be designed, in its chain's colour and faded: a sketch, not atoms.
  const letters = Array.from(renderer.chains ?? [], String);
  for (const [letter, { chain }] of inputView.ghosts) {
    const positions = letters.flatMap((l, i) => (l === letter ? [i] : []));
    // both of py2Dmol's painters see through a faded piece (the 2D one since sokrypton/py2Dmol#33),
    // so the structure behind a sketch stays readable and the sketch keeps its chain's own colour
    viewer.colourResidues(positions, chainColour(chain));
    renderer.setOpacity({ positions }, GHOST_OPACITY);
  }
  conditionEditor.rows.forEach((row, index) => {
    const wanted = new Set(row.picks.map((pick) => `${pick.chain}:${pick.resId}`));
    viewer.colourResidues(where((residue) => wanted.has(`${residue.chain}:${residue.resId}`)), constraintColour(index, row));
  });
  // ...and every hotspot last, in its own colour, whichever of the three places set it.
  const hot = new Set(hotspotResidues(spans).map((h) => `${h.chain}:${h.resId}`));
  viewer.colourResidues(where((residue) => hot.has(`${residue.chain}:${residue.resId}`)), HOTSPOT_COLOUR);
  renderer.render?.("job");
  paintViewerBar();
}

const GHOST_OPACITY = 0.45;
/** A sketched piece's colour as it reads on paper - its chain's, faded - for the legend's swatch. */
function ghostColour(chain) { return shade(chainColour(chain), "#ffffff", 1 - GHOST_OPACITY); }

/** The job as the viewer draws it: the structure's atoms, then a sketch of each designed piece. */
function jobDrawing(spans) {
  const { atoms, chainOf } = loaded === null ? { atoms: [], chainOf: new Map() } : inputStructureAtoms(loaded.atoms);
  const sketch = sketchJob(atoms, chainOf, spans, { hotspots: hotspotResidues(spans) });
  return { pdb: sketch.atoms.length === 0 ? null : toPdbString(sketch.atoms, { title: "the job" }),
    chainOf, ghosts: sketch.ghosts };
}

/**
 * Draw the job in the viewer: the loaded structure, and a faded sketch of every piece to be designed.
 *
 * 🔴 DRAWN FROM THE MOMENT THERE IS A JOB, not from the moment a structure is loaded: the blank job
 * is a 100-residue sketch on arrival, and every edit in step 3 redraws it. The camera is kept across
 * an edit of the same structure, so lengthening a linker does not throw the view away, and the
 * selection is kept with it - the structure's residues come first, so their positions do not move.
 */
function showInput() {
  if (running !== null || !viewerReady) return;
  const drawing = jobDrawing(spanList.read());
  const same = inputView !== null && inputView.loaded === loaded && loaded !== null;
  if (same && drawing.pdb === inputView.pdb) { paintInput(); return; }
  const selected = same && viewer.renderer?.residueSelection instanceof Set ? [...viewer.renderer.residueSelection] : [];
  viewer.cancelAnimations();
  viewer.reset();
  if (!same) viewer.forgetCamera();
  if (drawing.pdb === null) {
    inputView = null;
    viewerBox.style.display = "none";
    element("viewer-empty").hidden = false;
    hideSequence();
    paintViewerBar();
    return;
  }
  showViewer();
  inputView = { chainOf: drawing.chainOf, ghosts: drawing.ghosts, pdb: drawing.pdb, loaded };
  viewer.push(drawing.pdb, (built) => { if (built && !same) viewer.orient(); });
  if (selected.length > 0) {
    const real = new Set(inputResidueMap(viewer.renderer, inputView.chainOf, inputView.ghosts)
      .flatMap((residue, i) => (residue === undefined ? [] : [i])));
    viewer.renderer?.setResidueSelection?.(new Set(selected.filter((i) => real.has(i))));
  }
  conditionHost.style.display = "";
  // the strip is for picking the STRUCTURE's residues, so it is shown only when there is one
  if (!hasPickable()) hideSequence();
  else {
    showSequenceBox(true);
    sequenceHint.textContent = "Drag along the sequence to select residues.";
  }
  paintInput();
  window.SEQ?.clear?.();
  if (hasPickable()) window.SEQ?.buildView?.();
  conditionEditor?.refreshSelection();
}

/** A redraw for an edit, once the edits of one moment have landed. */
function scheduleJobView() {
  if (!viewerReady || running !== null) return;
  // a design on screen is not taken away by an edit; "Show the job" brings the job back
  if (inputView === null && viewer.built) { paintViewerBar(); return; }
  const ticket = ++jobRedraw;
  setTimeout(() => { if (ticket === jobRedraw) showInput(); }, 120);
}

/**
 * Whether any of the loaded structure is left out of the job - a residue no "Keep" segment covers,
 * or a ligand no ligand segment names - which is what the legend's "not used" grey means. 🔴 Keyed
 * on a structure being loaded, it named a colour nothing on screen had (SAM's pocket, the
 * unconditional example's placeholder).
 */
function unusedInput(spans) {
  if (loaded === null) return false;
  const placeholders = placeholderChains();
  const codes = new Set(spans.filter((span) => span.kind === "ligand").map((span) => String(span.code).toUpperCase()));
  const atoms = loaded.atoms;
  for (let i = 0; i < atoms.chain_id.length; i += 1) {
    const chain = String(atoms.chain_id[i]);
    if (placeholders.has(chain)) continue;
    const name = String(atoms.res_name[i]).trim().toUpperCase();
    if (THREE_TO_ONE[name] === undefined && NUCLEOTIDE[name] === undefined) {
      if (!codes.has(name)) return true;
      continue;
    }
    const residue = Number(atoms.res_id[i]);
    const kept = spans.some((span) => span.kind === "segment" && String(span.chain).trim() === chain
      && residue >= Number(span.first) && residue <= Number(span.last));
    if (!kept) return true;
  }
  return false;
}

/** The line over the viewer: what its colours mean, and the way back to the job from a design. */
function paintViewerBar() {
  if (!viewerReady) return;
  const showJob = element("show-job");
  const idle = running === null && refolding === null;
  showJob.hidden = !(inputView === null && viewer.built && idle);
  // the way back to the run from a refold, or from the job drawn after it
  element("show-run").hidden = !(runFrames.length > 0 && (refoldShown || inputView !== null) && idle);
  const legend = element("viewer-legend");
  legend.replaceChildren();
  const key = (colour, text, className = "") => {
    const item = document.createElement("span");
    item.className = `viewer-key ${className}`.trim();
    const swatch = document.createElement("i");
    swatch.style.setProperty("--key-colour", colour);
    item.append(swatch, text);
    legend.append(item);
  };
  if (inputView === null && refoldShown) {
    key(REFOLD_NEAR, "refold, where it agrees with the design");
    key(REFOLD_FAR, `where it moved ${REFOLD_FAR_AT} Å or more`);
    return;
  }
  if (inputView === null) return;
  const spans = spanList.read();
  const chainOf = chainOfSpans(spans);
  const first = (kind) => spans.findIndex((span, i) => span.kind === kind && chainOf[i] !== null);
  // 🔴 IN THE SEGMENTS' OWN NAMES (2026-10-28) - Designed, Sequence, Structure - and a sketch is
  // either of the first two, so each kind that is sketched gets its own key
  for (const [kind, name] of [["generated", "designed"], ["sequence", "sequence"]]) {
    const ghost = [...inputView.ghosts.values()].find((g) => spans[g.span]?.kind === kind);
    if (ghost !== undefined) key(ghostColour(ghost.chain), `${name} (a sketch of its length)`);
  }
  if (first("segment") >= 0) key(chainColour(chainOf[first("segment")]), "structure");
  if (unusedInput(spans)) key(INPUT_COLOUR, "not used");
  if (hotspotResidues(spans).length > 0) key(HOTSPOT_COLOUR, "hotspot");
}

// 🔴 THE FOLD PAGE'S SEQUENCE STRIP, REUSED AS IT IS. py2Dmol's panel finds its renderer through
// these callbacks, and the full bundle's own app set them to a viewer this page does not have. Its
// selection is the viewer's `residueSelection`, so the editor reads strip and clicks alike - and
// only while the INPUT is drawn can a pick name a residue (`inputView`, read by the editor's hooks).
// 🔴 AND IT STAYS THROUGH A RUN (2026-10-27), showing the design's sequence and changing with it:
// each cycle of a hunt, each frame of a generation, is a frame of the viewer, and the strip
// rebuilds on py2Dmol's frame change. It used to be hidden the moment a run began.
const sequenceBox = element("sequence-viewer-container");
const sequenceHint = element("sequence-hint");
const stage = document.querySelector(".design-stage");
/**
 * 🔴 THE PICTURE CANNOT BE SCROLLED AWAY (2026-10-27). The stage used to scroll as a whole once a
 * run's table and downloads made it taller than the window - and scrolled back up, the canvas
 * reached the pointer and took the wheel as a zoom, so the viewer could not be scrolled back into
 * view. The table and downloads scroll in their own box under it now, and while either is up the
 * viewer gives that box a share of the window. Looked up here rather than through the constants
 * above, which a run's first row can be drawn before.
 */
function sizeStage() {
  const shown = (id) => document.getElementById(id)?.style.display !== "none";
  document.querySelector(".design-stage")?.classList.toggle("has-results", shown("results") || shown("downloads"));
}
/**
 * Show or hide the strip. 🔴 THE VIEWER GIVES IT ROOM: the viewer is as tall as the window, and the
 * strip under it was below the fold of a laptop screen - present, and invisible, through every run.
 * The stage's class shortens the viewer by the strip's height (web/localfold.css).
 */
function showSequenceBox(on) {
  sequenceBox.style.display = on ? "" : "none";
  stage.classList.toggle("has-sequence", on);
}
window.SEQ?.setCallbacks?.({
  getRenderer: () => (sequenceBox.style.display === "none" ? null : viewer.renderer ?? null),
  getObjectSelect: () => null,
  applySelection: () => {},
  // a sketched piece is no sequence, so the strip leaves its chain out (and its Select all and
  // Invert with it); showInput clears the strip whenever the sketch changes, which re-asks this
  listsChain: (name, chain) => inputView === null || !inputView.ghosts.has(String(chain)),
});
document.addEventListener("py2dmol-residue-selection-change", () => {
  if (inputView !== null && takeGhostPick()) return;
  if (sequenceBox.style.display !== "none") window.SEQ?.updateSelection?.();
  conditionEditor?.refreshSelection();
});
// ...synchronously: the deferred build waits for the next animation frame, so the strip trailed the
// picture by a frame, and through a fast flow by several.
document.addEventListener("py2dmol-frame-change", () => {
  if (sequenceBox.style.display !== "none") window.SEQ?.buildView?.();
});
/**
 * A run's first frame takes the viewer: the job's drawing goes (a new object - a design's atom count
 * is not the job's), and the strip turns to the design's sequence.
 */
function beginRunPicture() {
  viewer.cancelAnimations();
  viewer.reset();
  inputView = null;
  runFrames = [];
  refoldShown = false;
  showRunSequence();
}

/**
 * The strip, rebuilt for the frame just pushed. A frame that BUILDS the viewer (a run's first) raises
 * no frame change, so without this the strip stayed empty through a hunt's whole first cycle.
 */
function followSequence() {
  if (sequenceBox.style.display !== "none") window.SEQ?.buildView?.();
}

/** The strip over a design: its sequence, rebuilt as frames arrive (`py2dmol-frame-change`). */
function showRunSequence() {
  showSequenceBox(true);
  sequenceHint.textContent = "The design's sequence, as it changes.";
  window.SEQ?.clear?.();
  window.SEQ?.buildViewDeferred?.();
}
/**
 * A click on a sketched piece opens its row in step 3, rather than selecting it: a sketch is no
 * residue a constraint could name, and its row is where its length is changed.
 */
function takeGhostPick() {
  const renderer = viewer.renderer;
  const selected = renderer?.residueSelection;
  if (!(selected instanceof Set) || selected.size === 0) return false;
  const letters = renderer.chains ?? [];
  const ghost = [...selected].find((i) => inputView.ghosts.has(String(letters[i])));
  if (ghost === undefined) return false;
  // a click on the sketch alone; a drag across both keeps only what it took of the structure
  if ([...selected].every((i) => inputView.ghosts.has(String(letters[i])))) {
    spanList.reveal(inputView.ghosts.get(String(letters[ghost])).span);
  }
  // the event this raises lands here again, with nothing sketched left in it
  renderer.setResidueSelection(new Set([...selected].filter((i) => !inputView.ghosts.has(String(letters[i])))));
  return true;
}
/** Hide the strip: nothing with a sequence is drawn. */
const hideSequence = () => {
  showSequenceBox(false);
  window.SEQ?.clear?.();
  conditionEditor?.refreshSelection();
};
// The strip wires its own Select all / Unselect / Invert (py2Dmol's panels/seq.js, since
// sokrypton/py2Dmol#32), against the renderer `getRenderer` above names.
// 🔴 A MOTIF DRAGGED OUT OF THE SEQUENCE, not typed: the selection's contiguous runs appended to the
// job as `Structure` spans, in the structure's own numbering, which is what a contig names.
function keepSelection() {
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
  say(`kept ${added.filter((span) => span.kind === "segment").length} piece(s) of the structure in the design`);
}

/**
 * 🔴 MARK AS HOTSPOT WRITES ONTO THE SEGMENTS, for either method: each picked residue joins the
 * hotspots of the kept segment that holds it, so the card shows it, both methods read it, and the
 * reader edits it there. A residue in no kept segment is said, not dropped in silence.
 */
function markHotspots(picks) {
  if (picks.length === 0) throw new Error("select residues first - click them in the viewer or drag along the sequence");
  const spans = spanList.read();
  const placed = new Set();
  const next = spans.map((span) => {
    if (span.kind !== "segment") return span;
    const chain = String(span.chain ?? "").trim();
    const inside = picks.filter((p) => p.chain === chain && p.resId >= Number(span.first) && p.resId <= Number(span.last));
    if (inside.length === 0) return span;
    for (const p of inside) placed.add(`${p.chain}:${p.resId}`);
    const own = hotspotResidues([span]).map((h) => h.resId);
    return { ...span, hotspots: residueRanges([...own, ...inside.map((p) => p.resId)]).replaceAll(",", " ") };
  });
  if (placed.size === 0) {
    throw new Error("those residues are in no kept segment - keep them in the design first, then mark them");
  }
  spanList.set(next);
  const missed = picks.length - placed.size;
  say(`marked ${placed.size} hotspot${placed.size === 1 ? "" : "s"}`
    + (missed > 0 ? `; ${missed} selected residue${missed === 1 ? " is" : "s are"} in no kept segment` : ""));
}

/** A segment card's "Use selection": the selected residues inside that segment, as its hotspot field. */
function selectedWithin(span) {
  const chain = String(span.chain ?? "").trim();
  const inside = conditionEditor.picksNow()
    .filter((p) => p.chain === chain && p.resId >= Number(span.first) && p.resId <= Number(span.last));
  if (inside.length === 0) {
    throw new Error(`select residues of chain ${chain} ${span.first}\u2013${span.last} first,`
      + " in the viewer or along the sequence");
  }
  return residueRanges(inside.map((p) => p.resId)).replaceAll(",", " ");
}

/** The selection step is about a structure's residues, so it is shown while there is one. */
function paintSelectionStep() {
  element("step-constraints").hidden = !hasPickable() && (conditionEditor?.rows.length ?? 0) === 0;
}

conditionEditor = createConditionEditor(conditionHost, {
  peek: () => {
    if (loaded === null) return null;
    if (inputView === null || viewer.renderer === undefined) return [];
    const selected = viewer.renderer.residueSelection;
    return picksOf(selected instanceof Set ? [...selected] : [], inputResidueMap(viewer.renderer, inputView.chainOf, inputView.ghosts));
  },
  colourOf: (index, row) => constraintColour(index, row),
  onKeep: () => keepSelection(),
  onHotspot: (picks) => markHotspots(picks),
  picks: () => {
    if (inputView === null || viewer.renderer === undefined) {
      showInput();
      throw new Error("the viewer was showing a design - the structure is drawn again; pick its residues");
    }
    const selected = viewer.renderer.residueSelection;
    return picksOf(selected instanceof Set ? [...selected] : [],
      inputResidueMap(viewer.renderer, inputView.chainOf, inputView.ghosts));
  },
  onChange: () => { paintInput(); paintJob(); },
});
conditionEditor.disable(method.value === "hunter" ? HUNTER_NO_CONDITIONS : null);
// The viewer and the editor exist: the job can be drawn, and is, from arrival.
viewerReady = true;
element("refold-button").addEventListener("click", () => { startRefold(); });
element("refold-design").addEventListener("change", (event) => { event.target.dataset.chosen = "1"; });
element("show-run").addEventListener("click", () => showRun());
element("show-job").addEventListener("click", () => { inputView = null; viewer.forgetCamera(); showInput(); });
showInput();
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
// ...and what the viewer is drawing: the structure's name and the sketch's chain letters, or null
// while a design is on screen.
window.__designDrawn = () => (inputView === null ? null : {
  structure: inputView.loaded?.name ?? null, ghosts: [...inputView.ghosts.keys()],
  positions: viewer.renderer?.chains?.length ?? 0,
});

/**
 * A search's answer on the page: its design marked best in the table, the track and the refold picker. A polished
 * design was never a hunt's cycle, so it joins the designs list here.
 */
function finishSearch({ best, passed, checks, runs, polished }) {
  if (best === null) {
    say(`no design: ${runs} restarts found no eligible cycle`);
    return;
  }
  for (const design of designs) if (design.record !== undefined) design.record.best = false;
  for (const marked of resultsBody.querySelectorAll("tr.best")) marked.classList.remove("best");
  let found = designs.find((design) => design.record?.run === best.restart && design.record?.cycle === best.cycle
    && design.record?.sequence === best.sequence);
  if (found === undefined) {
    const chain = CHAIN_IDS[0];
    found = { label: `Polish ${best.cycle + 1}`, pdb: reference === null ? best.pdb : alignToReference(best.pdb, reference, chain),
      chain, record: { run: best.restart, cycle: best.cycle, sequence: best.sequence, chain } };
    designs.push(found);
  }
  found.record.best = true;
  resultsBody.querySelector(`tr[data-key="${best.restart}:${best.cycle}"]`)?.classList.add("best");
  paintRefoldStep();
  const check = best.check;
  say(`done · ${passed ? "passed its check" : "no check passed; the nearest miss"} · ${found.label}`
    + (check === undefined ? "" : ` · check iPAE ${check.ipae.toFixed(2)}, RMSD ${check.rmsd.toFixed(1)} Å`)
    + ` · ${runs} restart${runs === 1 ? "" : "s"}, ${checks.length} check${checks.length === 1 ? "" : "s"}`
    + (polished > 0 ? `, ${polished} polished` : ""));
}

const startHunt = async () => {
  const spans = spanList.read();
  // 🔴 THE SPANS ARE THE JOB; `chainsOf` is how Hunter reads them. It also says
  // what Hunter cannot express - a motif inside a designed chain, two designed
  // chains - and a hunt that cannot express the job must not start.
  const { chains, chainKinds, ligandCodes, modifications, chainIndex, length, hold, fromStructure, problems }
    = chainsOf(spans, readers());
  if (problems.length > 0) {
    say(problems.join(" · "), true);
    return;
  }
  // 🔴 THE SEARCH TAKES THE SAME JOB AND THE SAME HUNT SETTINGS, with the recipe's numbers over the few it owns, and
  // answers with ONE design: the first whose check passed, else the nearest miss (web/search.js)
  const recipe = searchSelect.value === "recipe" ? searchRecipe(spans) : null;
  if (recipe !== null && recipe.protocol === null) {
    say(`no benchmark recipe fits this job: ${recipe.why}`, true);
    return;
  }
  // ...and Held coordinates is resolved now, so nothing to hold is said before any weights are fetched
  try {
    heldFor(chainsOf(spans, readers()));
  } catch (error) {
    say(error.message, true);
    return;
  }
  const controller = new AbortController();
  running = controller;
  huntLabel.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  designs = [];
  paintRefoldStep();
  // 🔴 THE JOB STAYS ON SCREEN UNTIL THE FIRST CYCLE REPLACES IT (`beginRunPicture`): loading the
  // folding model and the first fold take seconds to minutes, and wiping the viewer and the strip
  // at the click left two empty boxes for all of it.
  // A new run is a new frame of reference and a new held set.
  reference = null;
  held = hold;
  say("starting");
  progress.value = 0;
  progress.dataset.state = "running";

  const runner = recipe === null ? hunt : (options) => search({
    ...options, protocol: recipe.protocol,
    // a search's restarts are its runs: the table numbers them, and a length range draws one per restart
    report: (record) => { record.run = record.restart; options.report(record); },
    onCheck: (check) => say(`restart ${check.restart + 1} · cycle ${check.cycle} checked: iPAE ${check.ipae.toFixed(2)},`
      + ` RMSD ${check.rmsd.toFixed(1)} Å — ${check.pass ? "passes" : "misses"}`),
  });

  try {
    const outcome = await runner({
      // 🔴 RE-READ PER RUN, so a length RANGE becomes a search over lengths
      // rather than a notation the hunt ignores. `drawLengths` is deterministic
      // in the seed and the run, so the same hunt twice is the same hunt.
      job: (run) => {
        const drawn = drawLengths(spans, { run, seed: number("seed", 0) });
        // ...and its hotspots in that run's fold numbering (a drawn length moves what follows it)
        const job = { ...chainsOf(drawn, readers()), hotspots: window.__designHotspots ?? hotspotsOf(drawn) };
        // ...and what it holds in place, in that run's tokens: a search takes it as `held` and guides by its recipe,
        // a plain hunt guides every design fold to the last step
        const held = heldFor(job);
        return held === null ? job : { ...job, held, ...(recipe === null ? { motif: { atoms: held.atoms } } : {}) };
      },
      chains,
      chainKinds,
      ligandCodes,
      modifications,
      chainIndex,
      hold,
      // 🔴 EVERY KEPT CHAIN COPIED FROM THE LOADED STRUCTURE IS TEMPLATED BY IT (2026-10-29): the refold
      // templates it and so does RFD4-Proteina's benchmark, and a target folded from its sequence alone
      // is not the target the design is scored against (docs/DESIGN-TODO.md, the binder benchmark)
      templates: fromStructure.map(({ index, chain, first, last }) =>
        ({ text: structurePdb(chain, first, last), chainId: "A", chain: index })),
      // 🔴 THE SAME HOTSPOTS RFD4 CONDITIONS ON, used the only way Hunter can:
      // fold, then reject a cycle whose binder is not touching them.
      // 🔴 `window.__designHotspots` IS A DRIVER'S DOOR, as __designSpans is:
      // hotspots normally come from a `From structure` span, which needs a
      // structure loaded, and tools/check-design-page.py has to be able to force
      // an epitope that cannot be satisfied to prove the veto fires.
      hotspots: window.__designHotspots ?? hotspotsOf(drawLengths(spans, { run: 0, seed: number("seed", 0) })),
      // 🔴 STILL A LIST, because web/hunt.js rotates one per cycle and that
      // capability is intact; the page asks for one.
      folders: [folders.value],
      designer: designerSelect.value,
      contents: contentsOf(spans),
      runs: number("runs", 1),
      cycles: number("cycles", 5),
      // 🔴 "STAY WITHIN" TURNS THE SEQUENCES COUNT INTO A BUDGET: up to that many a cycle, one at a
      // time, the first that keeps its fold near the cycle before winning - not the best of all.
      ...(stayWithin() === null ? { beam: number("beam", 1) }
        : { beam: 1, accept: { within: stayWithin(), tries: Math.max(1, number("beam", 1)) } }),
      // 🔴 "WARM START" STARTS EACH CYCLE'S DIFFUSION FROM THE CYCLE BEFORE (shared/af3/structure/warm-start.js)
      ...(angstromsOf("warm-start", "Warm start") === null ? {}
        : { warm: { level: angstromsOf("warm-start", "Warm start") } }),
      // 🔴 "STEER TO HOTSPOTS" PULLS EVERY FOLD TOWARD THEM (cpu/af3/diffusion/steering.js)
      ...(steerWeight() === null ? {} : { steer: { weight: steerWeight() } }),
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
    if (recipe !== null) {
      finishSearch(outcome);
      progress.dataset.state = "done";
      return;
    }
    const { best, cycles } = outcome;
    // 🔴 WHY NO CYCLE WON, FROM THE CYCLES THEMSELVES. This used to blame the
    // alanine ceiling unconditionally, which became a lie the moment a second
    // veto existed: a run rejected for touching nothing was reported as
    // alanine-rich.
    if (best === null) {
      const reasons = [...new Set(cycles.filter((record) => record.cycle > 0)
        .map((record) => record.vetoed
          ?? (record.alanine !== null && record.alanine > MAX_ALANINE
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
        + ` objective ${best.score.toFixed(3)} · refold it in step 6 to check it`);
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
    paintViewerBar();
    paintRefoldStep();
    huntLabel.textContent = "Design";
  }
};

/**
 * 🔴 THE EXAMPLES, AS ONE DROPDOWN (2026-10-27), grouped by what they make and named as a reader
 * would ask for them; the engine's own title is each option's tooltip. It was eight goal cards and a
 * second "More examples" picker beside them - two controls for one choice. Each one fills in every
 * setting below it: the segments, the structure (drawn in the viewer), and its sampler settings.
 * A function declaration, because `paintMethod` reaches it while the module constructs.
 */
function exampleGroups() {
  // [id, what it is called here, what it does - in the words of someone choosing it]
  return [
    ["Start from scratch", [
      ["", "A new protein (100 residues)", "One chain of 100 residues, designed from nothing. Change its length in its card."],
      ["uncond_short", "Unconditional monomer", "One chain of 100 residues from the RFD4-Proteina example file."]]],
    ["Binders", [
      ["ppi_pdl1", "Binder to PD-L1, at its hotspots",
        "A new chain of 50\u2013120 residues against PD-L1 (chain B), aimed at the three hotspots on B's card."],
      ["yaml_5o45_binder", "Binder to a 5O45 domain", "A new 70-residue chain against chain A of 5O45."]]],
    ["Motifs", [
      ["motif_idx_1prw", "Scaffold an EF-hand motif (1PRW)",
        "Keeps two calcium-binding loops of 1PRW where they are and designs the protein around them."],
      ["motif_unidx_1prw", "Scaffold a motif, order free (1PRW)",
        "The same two loops, which the design may place anywhere in its chain."]]],
    ["Ligands and nucleic acids", [
      ["sm_sam", "A pocket for SAM", "A protein that holds S-adenosylmethionine, designed around the ligand."],
      ["dna_7m5w", "Bind double-stranded DNA", "A new chain against the two DNA strands of 7M5W."]]],
    ["A given fold", [
      ["custom_cath_monomer", "A Rossmann fold, helix by strand",
        "Six segments, each asked for a helix, a loop or a strand, in a Rossmann (CATH 3.40.50) order."]]],
    ["An existing protein", [
      ["partial_diffusion_1prw", "Diversify 1PRW", "Variants of 1PRW: half of the way back from its own structure."],
      ["inverse_folding_1prw", "New sequence for 1PRW's backbone", "A new sequence for 1PRW's backbone, the backbone held."]]],
  ];
}

/**
 * The examples dropdown, under the selected method. 🔴 ONE THE METHOD CANNOT RUN STAYS IN THE LIST,
 * disabled WITH its reason: a reader who came for the ligand pocket needs to know it is the method
 * and not the page. One already loaded keeps its place, and its card says why it will not run.
 */
function paintPresets() {
  const wasChosen = presetSelect.value;
  presetSelect.replaceChildren();
  const listed = new Set();
  for (const [label, entries] of exampleGroups()) {
    const group = document.createElement("optgroup");
    group.label = label;
    for (const [id, text] of entries) {
      const preset = RFD4P_PRESETS.find((entry) => entry.id === id);
      if (id !== "" && preset === undefined) continue;
      listed.add(id);
      const suits = id === "" || presetSuits(id, method.value);
      const option = new Option(suits ? text : `${text} — ${HUNTER_CANNOT[id]}`, id);
      option.title = preset?.title ?? "A designed chain of 100 residues, nothing else";
      option.disabled = !suits && id !== wasChosen;
      group.append(option);
    }
    presetSelect.append(group);
  }
  // ...and any example the groups above do not name, so a new one is never unreachable
  const rest = RFD4P_PRESETS.filter((preset) => !listed.has(preset.id));
  if (rest.length > 0) {
    const group = document.createElement("optgroup");
    group.label = "More";
    for (const preset of rest) group.append(new Option(preset.title ?? preset.id, preset.id));
    presetSelect.append(group);
  }
  presetSelect.value = wasChosen;
  paintPresetNote();
}

/**
 * The chosen example's one-line description - in the page's words, the engine's own being the
 * option's tooltip - or why the method cannot run it.
 */
function paintPresetNote() {
  const preset = RFD4P_PRESETS.find((entry) => entry.id === presetSelect.value);
  const cannot = preset !== undefined && !presetSuits(preset.id, method.value);
  const said = exampleGroups().flatMap(([, entries]) => entries).find(([id]) => id === presetSelect.value)?.[2];
  presetNote.textContent = cannot
    ? `Protein Hunter cannot run this example: ${HUNTER_CANNOT[preset.id]}. Switch the method to`
      + " RFD4-Proteina, or pick another."
    : said ?? preset?.description ?? "";
  presetNote.classList.toggle("warn", cannot);
  // 🔴 AND DESIGN SAYS SO TOO: a note reading "cannot run" over a button that runs anyway is two
  // answers. Never while running, where the button is Stop.
  if (running === null) {
    huntButton.disabled = cannot;
    huntButton.title = cannot ? `Protein Hunter cannot run this example: ${HUNTER_CANNOT[preset.id]}.` : "";
  }
}

// What a preset carries beyond its contig: the conditions, the template structure, and the
// per-modality options (presetExtras and presetSpans are declared with the shared state at the top).
presetSelect.addEventListener("change", () => chooseExample(presetSelect.value));

/**
 * Load an example: its segments, its structure, its settings. "" is the blank job, which also drops a
 * loaded structure.
 */
function chooseExample(id) {
  presetSelect.value = id;
  const preset = RFD4P_PRESETS.find((entry) => entry.id === id);
  // 🔴 AND THE LAST EXAMPLE'S SAMPLER GOES WITH IT, back to upstream's defaults: a Partial % left
  // from the partial-diffusion example turned the next job into partial diffusion.
  element("thinking").value = "medium";
  element("partial").value = "";
  if (preset === undefined) {
    presetExtras = {};
    presetSpans = null;
    clearStructure();
    spanList.set(BLANK_JOB());
    paintPresetNote();
    return;
  }
  const spans = exampleSpans(preset, null);
  if (spans === null) {
    presetNote.textContent = `${preset.title} uses labelled spans this editor has no row for yet`
      + ` (${preset.contig}) — it would run a different job from the one it names.`;
    presetNote.classList.add("warn");
    return;
  }
  // 🔴 AND THE SECONDARY STRUCTURE IS PUT BACK ON THE SPANS, so a loaded preset shows what it asks
  // for instead of an unexplained label. C_SEC's arms name the spans by `sel('*/h1/*/*')`.
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
    // 🔴 A PRESET WITHOUT `specMode` IS DRIVEN BY ITS STRUCTURE'S ANNOTATIONS, not by its contig -
    // see the note in web/generate.js.
    fromAnnotations: preset.specMode !== true && preset.cifUrl !== null && preset.cifUrl !== undefined,
    inverseFolding: preset.inverseFolding,
  };
  // 🔴 THE PRESET'S RECYCLE BUDGET AND PARTIAL FRACTION GO INTO THE CONTROLS, not into this object:
  // a value kept beside a control that shows another is two answers to one question.
  if (preset.recycleSchedule !== undefined) {
    element("thinking").value = String(preset.recycleSchedule).replace(/^thinking_/, "");
  }
  element("partial").value = preset.partialTrajectoryFractionSteps === undefined
    ? "" : String(Math.round(preset.partialTrajectoryFractionSteps * 100));
  if (preset.nsteps !== undefined && method.value === "rfd4p") element("steps").value = String(preset.nsteps);
  presetSpans = null;
  spanList.set(spans);
  presetSpans = JSON.stringify(spanList.read());
  paintPresetNote();

  // 🔴 AND THE PRESET'S STRUCTURE IS LOADED AND DRAWN, NOT JUST NAMED: Protein Hunter reads a kept
  // segment's sequence out of it, and the reader sees what the example is about.
  if (preset.cifUrl === undefined || preset.cifUrl === null) {
    clearStructure();
    return;
  }
  if (loaded?.name === preset.cifUrl) {
    examplePieces(preset);
    return;
  }
  // 🔴 THE OLD STRUCTURE GOES FIRST, SYNCHRONOUSLY. `startGenerate` prefers a loaded structure over
  // the preset's own `cifUrl`, and a run started while this fetch is in flight would otherwise run
  // THIS preset's contig against the PREVIOUS preset's structure - which shifted every result in the
  // preset sweep by one, with no error anywhere.
  loaded = null;
  describeStructure();
  fetch(preset.cifUrl).then(async (response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    // 🔴 THE PRESET MAY HAVE CHANGED WHILE THIS WAS IN FLIGHT.
    if (presetSelect.value !== id) return;
    takeStructure(bytes, preset.cifUrl, false);
    examplePieces(preset);
    paintJob();
  }).catch((error) => {
    // Not fatal: RFD4 fetches it again inside the run; only Hunter needs it here.
    structureNote.textContent = `could not load this example's structure: ${error.message}`;
    structureNote.classList.add("warn");
  });
}

/**
 * An example's segments once its structure is here (web/example-spans.js): an annotation-driven one
 * as its FILE describes it, hotspots and all; any other with its contig moved into the crop's own
 * numbering. Its sampler settings and its note were set when it was chosen.
 */
function examplePieces(preset) {
  const spans = exampleSpans(preset, loaded.atoms);
  // 🔴 NOT COMPARED WHILE IT LOADS: against the last example's spans, an example sharing its file
  // (the three 1PRW ones) read its own loading as an edit and said so.
  presetSpans = null;
  if (spans !== null) spanList.set(spans);
  presetSpans = JSON.stringify(spanList.read());
  warnIgnoredEdit(spanList.read());
}

/**
 * 🔴 SAID WHEN IT APPLIES: an example that runs from its structure's own annotations ignores the
 * spans, so an edit to them changes nothing that runs. Compared against the spans the example set
 * (and translated), so the example's own loading is not mistaken for an edit.
 */
function warnIgnoredEdit(spans) {
  // Protein Hunter reads the segments as they are, so only RFD4-Proteina's edits can go unused.
  if (method.value !== "rfd4p" || !presetExtras.fromAnnotations || presetSpans === null) return;
  const edited = JSON.stringify(spans) !== presetSpans;
  presetNote.classList.toggle("warn", edited);
  if (!edited) { paintPresetNote(); return; }
  presetNote.textContent = "This example runs from its structure's own annotations, so edits here are not used"
    + " by RFD4-Proteina. Choose \u201cA new protein\u201d, or load a structure, to build your own.";
}

/** Forget the loaded structure: the viewer goes back to its placeholder. */
function clearStructure() {
  if (running !== null) return;
  loaded = null;
  describeStructure();
  inputView = null;
  hideSequence();
  if (conditionEditor.rows.length > 0) conditionEditor.set([]);
  // ...and the job is drawn again without it: what is left to design, sketched
  showInput();
}

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
/**
 * One loaded chain's residues `first`-`last`, as a PDB with chain letter A: a template for the chain the
 * job keeps from it. A function DECLARATION, as `sequenceOf` below is.
 */
function structurePdb(chain, first, last) {
  const atoms = loaded.atoms;
  const lines = [];
  for (let index = 0; index < atoms.chain_id.length; index += 1) {
    if (String(atoms.chain_id[index]) !== chain) continue;
    const residue = Number(atoms.res_id[index]);
    if (residue < first || residue > last || atoms.coord[index] === undefined) continue;
    const [x, y, z] = atoms.coord[index];
    if (![x, y, z].every(Number.isFinite)) continue;
    const name = String(atoms.atom_name[index]);
    const element = String(atoms.element[index] ?? "").toUpperCase();
    lines.push(`ATOM  ${String(lines.length + 1).padStart(5)} ${(name.length === 4 ? name : ` ${name}`).padEnd(4)}`
      + ` ${String(atoms.res_name[index]).padStart(3)} A${String(residue).padStart(4)}    `
      + `${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00`
      + `          ${element.padStart(2)}`);
  }
  if (lines.length === 0) throw new Error(`chain ${chain} ${first}-${last} has no coordinates to template`);
  return `${lines.join("\n")}\nEND\n`;
}

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
    // what the user typed. A nucleotide is its base's letter: DG is G.
    .map(([, name]) => THREE_TO_ONE[name] ?? NUCLEOTIDE[name] ?? "X").join("");
}

/**
 * What a loaded chain is made of, from its residue names: "dna", "rna" or "protein". A chain mostly
 * of deoxynucleotides is DNA, of ribonucleotides RNA - so the DNA-binder example's kept strands fold
 * as DNA under Protein Hunter, and pick NA-MPNN, rather than as a protein of unknowns.
 */
function polymerOf(chain) {
  if (loaded === null) return "protein";
  const atoms = loaded.atoms;
  const count = atoms?.nAtoms ?? atoms?.chain_id?.length ?? 0;
  let deoxy = 0;
  let ribo = 0;
  let other = 0;
  let last = null;
  for (let index = 0; index < count; index += 1) {
    if (String(atoms.chain_id[index]) !== chain) continue;
    const residue = Number(atoms.res_id[index]);
    if (residue === last) continue;
    last = residue;
    const name = String(atoms.res_name[index]).trim().toUpperCase();
    if (DEOXY.has(name)) deoxy += 1;
    else if (NUCLEOTIDE[name] !== undefined) ribo += 1;
    else other += 1;
  }
  if (deoxy > other && deoxy >= ribo) return "dna";
  if (ribo > other) return "rna";
  return "protein";
}

/**
 * 🔴 A CHAIN THAT IS ONLY AN EXAMPLE'S PLACEHOLDER FOR WHAT IS DESIGNED (an expandable segment,
 * one residue in the file) is not "chain A 1-1": it is where the design goes, and nothing in it can
 * be kept, picked or aimed at.
 */
function placeholderChains() {
  const placeholders = new Set();
  if (loaded === null) return placeholders;
  const ann = loaded.atoms?.annotations ?? {};
  const sentinel = ann.annotation_expsegmin_1_residue ?? loaded.atoms?.annotation_expsegmin_1_residue;
  if (!sentinel) return placeholders;
  const real = new Set();
  for (let i = 0; i < loaded.atoms.chain_id.length; i += 1) {
    const chain = String(loaded.atoms.chain_id[i]);
    if (Number(sentinel[i]) >= 0) placeholders.add(chain); else real.add(chain);
  }
  for (const chain of real) placeholders.delete(chain);
  return placeholders;
}

/** Whether the loaded structure has anything to pick: a chain or ligand that is not a placeholder. */
function hasPickable() {
  if (loaded === null) return false;
  const placeholders = placeholderChains();
  return [...loaded.chains.keys()].some((chain) => !placeholders.has(chain));
}

const describeStructure = () => {
  element("clear-structure").hidden = loaded === null;
  paintSelectionStep();
  if (loaded === null) {
    structureNote.textContent = "";
    structureNote.classList.remove("warn");
    return;
  }
  const placeholders = placeholderChains();
  // 🔴 AND A CHAIN THAT IS ONLY A LIGAND IS NAMED AS ONE: "chain B 1-1" for SAM read as a broken file.
  const ligandOf = (chain) => {
    const names = new Set();
    const atoms = loaded.atoms;
    for (let index = 0; index < atoms.chain_id.length; index += 1) {
      if (String(atoms.chain_id[index]) !== chain) continue;
      const name = String(atoms.res_name[index]).trim().toUpperCase();
      if (THREE_TO_ONE[name] !== undefined || NUCLEOTIDE[name] !== undefined) return null;
      names.add(name);
    }
    return [...names].join(" ");
  };
  const chains = [...loaded.chains.entries()].filter(([chain]) => !placeholders.has(chain));
  const parts = chains.filter(([chain]) => ligandOf(chain) === null)
    .map(([chain, range]) => `${chain} ${range.first}\u2013${range.last}`);
  const ligands = chains.map(([chain]) => ligandOf(chain)).filter((name) => name !== null && name !== "");
  // The entry or file's name, not the path an example is served from.
  const name = String(loaded.name).split("/").pop().replace(/\.(b?cif|mmcif)(\.(zst|gz))?$/i, "");
  const pieces = [];
  if (parts.length > 0) pieces.push(`chain${parts.length === 1 ? "" : "s"} ${parts.join(", ")}`);
  if (ligands.length > 0) pieces.push(`ligand${ligands.length === 1 ? "" : "s"} ${ligands.join(", ")}`);
  if (pieces.length === 0) pieces.push("only a placeholder for the design, nothing to keep");
  structureNote.textContent = [name, ...pieces].join(" \u00b7 ");
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

element("clear-structure").addEventListener("click", () => { clearStructure(); paintJob(); });

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

/** One finished candidate, as a row and as the structure on screen. */
function drawGenerated(candidate, total, designed) {
  resultsBox.style.display = "";
  sizeStage();
  if (reference === null) beginRunPicture();
  // The finished structure is the last frame, in the same frame of reference as
  // the trajectory that led to it.
  const placed = reference === null ? candidate.pdb : alignToReference(candidate.pdb, reference, undefined);
  const paint = () => paintGenerated(placed);
  pushRunFrame(placed, paint);
  // 🔴 AND FRAMED ON THE FINISHED DESIGN: the viewer oriented on the trajectory's FIRST frame, which
  // is close to noise at another scale, so the design came out zoomed far into one corner.
  viewer.orient();

  // 🔴 NO "best" ROW: a generation scores nothing of its own, so with the automatic check gone there
  // is nothing to rank candidates by - the Refold step is how one is judged, and its verdict lands
  // in this row's last cell.
  const row = document.createElement("tr");
  const refoldCell = document.createElement("td");
  refoldCell.className = "validity";
  refoldCell.textContent = "not refolded";
  for (const text of [total === 1 ? "1" : `${candidate.candidate + 1} of ${total}`,
    String(candidate.seed), `${candidate.tokens} tokens`]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.append(cell);
  }
  row.append(refoldCell);
  resultsBody.append(row);
  designs.push({ label: `Design ${candidate.candidate + 1} · seed ${candidate.seed}`, pdb: placed,
    chain: designed, paint, candidate, cell: refoldCell });
  paintRefoldStep();
}

/** A run's frame: pushed, painted, and kept so the run can be drawn again after a refold. */
function pushRunFrame(pdb, paint) {
  showViewer();
  runFrames.push(pdb);
  runPaint = paint;
  viewer.push(pdb, (built) => { if (built) viewer.orient(); });
  paint();
  followSequence();
}

/** The run again, in place of a refold or the job: every kept frame, painted as the last was. */
function showRun() {
  viewer.cancelAnimations();
  viewer.reset();
  inputView = null;
  refoldShown = false;
  showViewer();
  showRunSequence();
  for (const pdb of runFrames) viewer.push(pdb);
  runPaint();
  viewer.orient();
  followSequence();
  paintViewerBar();
}

/**
 * The design and its refold, overlaid: two viewer objects drawn together, the design in the run's
 * colours and the refold over it in grey that turns red where it has moved - so a misfold shows
 * where it happens, not only as an RMSD.
 *
 * 🔴 TWO OBJECTS, NOT ONE OF TWICE THE CHAINS. It was drawn as one structure with the refold under
 * spare chain letters, which put chains C and D on the strip for molecules that are A and B;
 * as its own object the refold keeps the design's letters, and the viewer's object list can switch
 * either off.
 */
function drawRefold(design, result) {
  viewer.cancelAnimations();
  viewer.reset();
  inputView = null;
  refoldShown = true;
  showViewer();
  showRunSequence();
  sequenceHint.textContent = `The design and its ${result.model} refold, overlaid.`;
  viewer.push(design.pdb);
  design.paint();
  const designObject = viewer.object;
  viewer.addObject(REFOLD_OBJECT, result.refold);
  viewer.showObjects([designObject, REFOLD_OBJECT]);
  const letters = new Set(result.refold.split("\n").filter((line) => line.startsWith("ATOM") || line.startsWith("HETATM"))
    .map((line) => line[21]));
  viewer.colourChains(Object.fromEntries([...letters].map((letter) => [letter, REFOLD_NEAR])), REFOLD_OBJECT);
  // five steps from "agrees" (1 A or less) to "moved" (REFOLD_FAR_AT or more)
  const steps = new Map();
  for (const { chain, residue, distance } of result.deviations) {
    const step = Math.round(5 * Math.min(1, Math.max(0, (distance - 1) / (REFOLD_FAR_AT - 1))));
    if (step === 0) continue;
    const key = `${step}:${chain}`;
    if (!steps.has(key)) steps.set(key, { step, chain, residues: [] });
    steps.get(key).residues.push(residue);
  }
  for (const { step, chain, residues } of steps.values()) {
    viewer.colourNumbered(REFOLD_OBJECT, chain, residues, shade(REFOLD_NEAR, REFOLD_FAR, step / 5));
  }
  viewer.orient();
  followSequence();
  paintViewerBar();
}

/** Step 6: the finished designs to choose from, and whether a refold can start now. */
function paintRefoldStep() {
  const select = element("refold-design");
  const button = element("refold-button");
  // the reader's pick stays; until there is one, the pick follows the best design (a hunt's best
  // cycle moves as cycles arrive), else the latest
  if (designs.length === 0) delete select.dataset.chosen;
  const kept = select.dataset.chosen === undefined ? undefined
    : select.selectedOptions[0]?.textContent.replace(/ \(best\)$/, "");
  select.replaceChildren(...designs.map((design, at) => new Option(
    `${design.label}${design.record?.best === true ? " (best)" : ""}`, String(at))));
  const again = designs.findIndex((design) => design.label === kept);
  const best = designs.findIndex((design) => design.record?.best === true);
  select.value = String(again >= 0 ? again : best >= 0 ? best : designs.length - 1);
  const busy = running !== null;
  select.disabled = designs.length === 0 || busy || refolding !== null;
  element("refold-model").disabled = refolding !== null;
  button.textContent = refolding === null ? "Refold" : "Stop";
  button.disabled = refolding === null && (designs.length === 0 || busy);
  if (designs.length === 0) {
    element("refold-note").textContent = "Design something first; each finished design can then be refolded here.";
    element("refold-save").hidden = true;
  }
}

/** Refold the chosen design with the chosen model, and draw the two together. */
async function startRefold() {
  if (refolding !== null) {
    refolding.abort();
    return;
  }
  const design = designs[Number(element("refold-design").value)];
  const model = element("refold-model").value;
  const note = element("refold-note");
  const controller = new AbortController();
  refolding = controller;
  paintRefoldStep();
  paintViewerBar();
  note.textContent = `Refolding ${design.label} with ${model}…`;
  element("refold-save").hidden = true;
  progress.value = 0;
  progress.dataset.state = "running";
  try {
    const result = await refoldDesign({
      pdb: design.pdb,
      chain: design.chain,
      model,
      device: await getDevice(),
      // 🔴 NOT THE RUN'S SEED: refolding with the one that made the design invites the two to share
      // whatever the seed decided, the opposite of an independent check.
      seed: number("seed", 0) + 1,
      status: (text) => say(text),
      signal: controller.signal,
    });
    design.refold = result;
    if (design.candidate !== undefined) design.candidate.refold = result;
    if (design.cell !== undefined) design.cell.textContent = result.described;
    drawRefold(design, result);
    note.textContent = `${design.label}: ${result.described}`;
    say(`refolded · ${result.described}`);
    const save = element("refold-save");
    save.hidden = false;
    save.onclick = () => saveFile(`${design.label.replace(/[^A-Za-z0-9]+/g, "-").toLowerCase()}-refold-${model}.pdb`,
      result.refold, "chemical/x-pdb");
    progress.value = 1;
    progress.dataset.state = "done";
  } catch (error) {
    if (controller.signal.aborted) {
      say("refold stopped");
      note.textContent = "Refold stopped.";
      progress.dataset.state = "idle";
    } else {
      say(error.message, true);
      note.textContent = `Could not refold: ${error.message}`;
      progress.dataset.state = "error";
    }
  } finally {
    refolding = null;
    paintRefoldStep();
    paintViewerBar();
  }
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
      + (c.refold?.described ? ` - refolded, ${c.refold.described}` : "")).join("\n") + "\n"]);
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
    sizeStage();
    return;
  }
  downloadsBox.style.display = "";
  sizeStage();
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
    row.append(`Design ${c.candidate + 1} \u00b7 seed ${c.seed}`,
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
  // (the job stays drawn until the flow's first frame replaces it - see `beginRunPicture`)
  huntLabel.textContent = "Stop";
  resultsBody.replaceChildren();
  track.replaceChildren();
  designs = [];
  paintRefoldStep();
  // A generation is one frame, so there is nothing to fit it onto and nothing
  // held: RFD4 conditions on given coordinates rather than holding a sequence.
  reference = null;
  held = [];
  progress.value = 0;
  progress.dataset.state = "running";
  say("starting");

  // 🔴 THE BEAM IS CANDIDATES, NOT CYCLES. A generation is one pass, so there
  // is nothing to carry forward the way a hunt carries a sequence between
  // cycles - drawing N of them is the only thing "try more than one" can mean
  // here, and the Refold step is how one is told from another. Each is a whole
  // generation, so the cost is linear and the control says so by being the
  // same control Hunter labels Sequences.
  const candidates = Math.max(1, Math.floor(number("beam", 1)));
  // the chains RFD4 writes for what it generates, held out when a refold is fitted onto the design -
  // so the refold is laid on what was kept, and a binder that lands elsewhere shows as moved
  const runLetters = chainLetters(spans, "rfd4p", ligandChains(spans));
  const chainOfRun = chainOfSpans(spans);
  const designedLetters = [...new Set(spans.map((span, at) => (span.kind === "generated" && chainOfRun[at] !== null
    ? runLetters[chainOfRun[at]] : null)).filter((letter) => letter != null))];
  generatedRun = [];
  paintDownloads();

  try {
    for (let candidate = 0; candidate < candidates; candidate += 1) {
    if (controller.signal.aborted) break;
    // Each candidate is its own trajectory, so it gets its own frame of
    // reference and its own run of the viewer.
    if (candidate > 0) reference = null;
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
        showViewer();
        // 🔴 FITTED ONTO THE FIRST FRAME, for the reason a hunt is: a flow step
        // moves the molecule as well as folding it, so played back raw the
        // trajectory tumbles. There is no designed chain to exclude here - the
        // whole thing is being generated - so the fit is over everything.
        if (reference === null) { beginRunPicture(); reference = step.pdb; }
        const placed = reference === step.pdb ? step.pdb
          : alignToReference(step.pdb, reference, undefined);
        pushRunFrame(placed, () => paintGenerated(placed));
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

    const made = { ...result, candidate };
    drawGenerated(made, candidates, designedLetters);
    generatedRun.push(made);
    paintDownloads();
    }

    if (generatedRun.length === 0) throw new Error("no candidate completed");
    const last = generatedRun[generatedRun.length - 1];
    say(`done · ${last.tokens} tokens over ${last.steps} steps`
      + (candidates === 1 ? "" : `, ${generatedRun.length} designs`) + " · refold one in step 6 to check it");
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
    paintViewerBar();
    paintRefoldStep();
    huntLabel.textContent = "Design";
  }
};

huntButton.addEventListener("click", () => {
  if (running !== null) {
    running.abort();
    return;
  }
  if (refolding !== null) {
    say("a refold is running - stop it in step 6 first", true);
    return;
  }
  if (method.value === "hunter") startHunt();
  else startGenerate();
});

/**
 * contig-ui.js
 * ------------
 * AI Context: THE DESIGN JOB AS A LIST OF SPANS YOU ADD TO
 *
 * The design page's input, built the way `web/entity-ui.js` builds a fold's:
 * rows you add, reorder and remove, with the thing the engine actually wants
 * derived from them rather than typed. Here that thing is a **contig** -
 * RFdiffusion's notation - and the rows are its spans.
 *
 * 🔴 WHY A LIST OF SPANS IS THE RIGHT INPUT FOR BOTH METHODS. A span has two
 * independent axes, geometry and sequence, and every modality is a combination
 * of them (docs/DESIGN.md section 5): a target chain is given/given, a binder is
 * generated/free, a motif is given/given inside a designed chain, inverse
 * folding is given geometry with a free sequence. So one editor expresses all of
 * them and there is no per-modality screen to build.
 *
 * THE FOUR KINDS, and what each emits:
 *
 *   generated   a length, or a range     ->  `100-100`, `50-120`
 *   segment     a chain and its residues ->  `A18-132`
 *   ligand      a CCD code               ->  `SAM`
 *   break       a new chain starts here  ->  `/0`
 *
 * 🔴 AND WHAT DIFFERS PER METHOD IS WHICH KINDS ARE LEGAL, NOT WHAT A SPAN IS.
 * Protein Hunter cannot generate geometry - it folds a mostly-X sequence and
 * redesigns it - so a length RANGE is not something it can express, and it has
 * no use for a ligand span's pocket conditioning. The list takes a `method` and
 * marks the rows that method cannot honour, with the reason on the row, rather
 * than hiding them: a control that vanishes teaches nothing.
 *
 * 🔴 THE ROW IS A GRID WITH A FIXED NUMBER OF CHILDREN, three, whatever the
 * kind. `web/localfold.css`'s note on `.entity-row` is the reason: a grid with
 * more children than tracks does not overflow, it WRAPS onto an implicit second
 * row, and `tools/mobile-layout.py` reports that as a row broken across two
 * lines. The per-kind fields therefore live inside one middle child.
 */

/** The kinds a span can be, in the order the picker lists them. */
export const SPAN_KINDS = ["generated", "sequence", "segment", "ligand", "break"];

/** What a contig calls each polymer, from SUFFIX_TO_CHAIN_TYPE in the pipeline. */
export const SPAN_POLYMER_SUFFIX = { protein: "P", rna: "R", dna: "D" };

/**
 * The secondary structures a named span can ask for, as C_SEC's own keys - upstream's `SSEnum`
 * says SHEET, and a span written STRAND was a job upstream refuses. The label stays "strand".
 */
export const SPAN_STRUCTURES = { "": "any", HELIX: "helix", LOOP: "loop", SHEET: "strand" };

/** The chain letters the loop and the fold agree on, mirroring hunter-loop's. */
export const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/**
 * One colour per chain, shared by the span rows and the viewer.
 *
 * 🔴 THE SWATCH AND THE STRUCTURE MUST BE THE SAME COLOUR OR NEITHER IS WORTH
 * DRAWING. That is the whole point of the indicator: a reader looks at a row,
 * looks at the viewer, and knows which thing moved. So the palette is exported
 * from here - beside `chainsOf`, which is what decides where one chain ends and
 * the next begins - rather than written out again in the page and in the CSS.
 *
 * 🔴 AND HUE IS THE CHAIN, NEVER THE ROLE. Designed, held and target are drawn
 * as TREATMENTS of the chain's own colour (see `shade`), because a reader needs
 * both answers at once and there is only one set of pixels to give them in.
 * Colouring by role alone - which this page did first - cannot say which chain a
 * held motif belongs to, and colouring by chain alone cannot say what is being
 * decided.
 *
 * Chosen to stay apart on a white background and to survive the fade below:
 * blue, green, magenta, orange, teal, olive. A seventh chain wraps, which is
 * honest - the page is not going to pretend 26 distinguishable hues exist.
 */
export const CHAIN_COLOURS = ["#2563eb", "#16a34a", "#d946ef", "#ea580c", "#0891b2", "#ca8a04"];

/** A ligand belongs to no chain, so it takes the one neutral in the set. */
export const LIGAND_COLOUR = "#78716c";

/**
 * The loaded structure while residues are picked from it for conditions (web/condition-editor.js):
 * drawn pale, each condition row's residues in one of these. Hues the chain palette does not use,
 * because the view is about which residue carries which condition and not about chains.
 */
export const INPUT_COLOUR = "#cbd5e1";
export const CONDITION_COLOURS = ["#e11d48", "#7c3aed", "#0ea5e9", "#84cc16", "#be185d", "#1e293b"];
/** Hotspots have one colour of their own, wherever they were set: on the structure, in a chip, in the line. */
export const HOTSPOT_COLOUR = "#f59e0b";

/** The colour of the nth chain, wrapping rather than running out. */
export function chainColour(index) {
  return CHAIN_COLOURS[((index % CHAIN_COLOURS.length) + CHAIN_COLOURS.length) % CHAIN_COLOURS.length];
}

/**
 * A chain's colour, moved toward another - how a ROLE is drawn.
 *
 * 🔴 BLENDED RATHER THAN SWAPPED, so the hue survives and the chain is still
 * identifiable when it is recessive. A target chain painted flat grey is a chain
 * the reader can no longer match to its row, which is the thing the swatch was
 * added for.
 *
 * @param {string} colour a #rrggbb
 * @param {string} towards a #rrggbb
 * @param {number} fraction 0 keeps `colour`, 1 reaches `towards`
 */
export function shade(colour, towards, fraction) {
  const read = (hex) => [1, 3, 5].map((at) => Number.parseInt(hex.slice(at, at + 2), 16));
  const [r, g, b] = read(colour);
  const [tr, tg, tb] = read(towards);
  const mix = (from, to) => Math.round(from + (to - from) * fraction)
    .toString(16).padStart(2, "0");
  return `#${mix(r, tr)}${mix(g, tg)}${mix(b, tb)}`;
}

/**
 * Which chain each span belongs to, as an index per span.
 *
 * 🔴 THE SAME WALK `chainsOf` DOES, AND IT HAS TO STAY THE SAME WALK: a break
 * starts a chain, a ligand belongs to none, and everything else joins the chain
 * in progress. If this drifted from `chainsOf`, the row would be labelled one
 * colour and the structure painted another, which is worse than no indicator at
 * all because it reads as an answer.
 *
 * @param {object[]} spans
 * @returns {(number|null)[]} the chain index per span, or null where the span is
 *   not part of a chain - a break, or a ligand.
 */
export function chainOfSpans(spans) {
  const out = [];
  let index = 0;
  let used = false;
  for (const span of spans) {
    if (span.kind === "break") {
      // 🔴 A BREAK ONLY ADVANCES A CHAIN THAT HAS SOMETHING IN IT. Two breaks in
      // a row, or one before any span, would otherwise skip a colour and the
      // rows would disagree with `chainsOf`, which starts a chain when a part
      // arrives rather than when a break is seen.
      if (used) { index += 1; used = false; }
      out.push(null);
      continue;
    }
    if (span.kind === "ligand") { out.push(null); continue; }
    used = true;
    out.push(index);
  }
  return out;
}

// 🔴 NAMED FOR WHAT A NEWCOMER WANTS DONE TO THE PIECE (2026-10-27): "From structure" and
// "New chain" described where a span comes from and what follows it, not what it is.
export const SPAN_LABELS = {
  generated: "Design new",
  sequence: "Known sequence",
  segment: "Keep from structure",
  ligand: "Ligand",
  break: "Chain break",
};

/** What each kind is, shown as the picker's title. */
export const SPAN_NOTES = {
  generated: "Residues to design. Give one length, or a range for the method to choose from."
    + " Name it to ask for a secondary structure.",
  sequence: "A chain you know the sequence of - protein, DNA or RNA. Protein Hunter folds it;"
    + " RFD4-Proteina needs its coordinates, so use \u201cKeep from structure\u201d there.",
  segment: "Residues taken from a loaded structure, held as they are.",
  ligand: "A small molecule by its CCD code, as context to design against.",
  break: "Everything after this is a separate chain.",
};

/**
 * Which kinds a method can honour, and why not when it cannot.
 *
 * 🔴 PROTEIN HUNTER'S LIMIT IS NOT A MISSING FEATURE, IT IS THE METHOD. It
 * reaches a structure by folding a sequence, so it can only ask for ONE length
 * and see what comes back; a range implies a search over lengths, which is a
 * different loop. And no folder here takes a pocket constraint, so a ligand in
 * the list changes the fold and the designer but is not designed *against* in
 * the way RFD4 means - a gap in this port rather than in the method.
 */
export const METHOD_LIMITS = {
  hunter: {
    // 🔴 A RANGE IS NO LONGER REFUSED HERE. It used to read "Protein Hunter
    // folds one length at a time; a range needs a search over lengths" - which
    // is true of a single fold and describes a search the page already has.
    // `Runs` IS that search: each run draws its own length out of the range
    // (see `drawLengths`), so a range means "try several sizes" rather than
    // "this method cannot". Four of the ten shipped examples were refused for
    // this reason alone.
    // 🔴 THIS IS A PORT GAP, NOT A PROPERTY OF THE METHOD, and the wording used
    // to claim the second. Genuine Boltz-2 takes a `pocket` constraint - a
    // binder chain plus the residues it should contact - and Hunter folds with
    // whatever is in FOLDERS, so a folder that honoured one WOULD condition the
    // fold on the pocket. What is missing is in this tree: the AF3-lineage
    // featuriser has no constraint path at all (nothing in shared/af3/featurise
    // builds one), and boltz2 is not in FOLDERS, though it is published and
    // shares `foldAf3` with the three that are.
    ligand: "This port's folding models do not take a pocket constraint yet — a ligand"
      + " changes the fold and the designer, but is not conditioned on.",
    // 🔴 A LABEL IS ONLY USEFUL TO SOMETHING THAT CONDITIONS ON IT. Hunter
    // reaches a structure by folding a sequence; it has no way to be told "this
    // stretch should be a helix", so a named span would be a control that did
    // nothing.
    labelled: "Protein Hunter cannot be asked for a secondary structure; it folds whatever"
      + " the sequence gives.",
  },
  rfd4p: {
    // 🔴 THE GEOMETRY AXIS, AS A LIMIT. RFD4 generates coordinates AROUND given
    // coordinates, so a chain known only by its sequence is not something it can
    // hold fixed - where Hunter, which reaches structure by folding, takes
    // exactly that.
    sequence: "RFD4-Proteina conditions on coordinates, not on a sequence — load the structure"
      + " and use \u201cKeep from structure\u201d.",
  },
};

const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className !== undefined) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const number = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

/** A fresh span of `kind`, with defaults a reader would expect. */
export function emptySpan(kind = "generated") {
  // (a given span is HELD by default; `redesign` is the opt-in)
  if (kind === "generated") return { kind, min: 75, max: 75, name: "", secondary: "", polymer: "protein" };
  if (kind === "sequence") return { kind, value: "", polymer: "protein" };
  if (kind === "segment") return { kind, chain: "A", first: 1, last: 100, hotspots: "" };
  if (kind === "ligand") return { kind, code: "" };
  return { kind };
}

/**
 * One span as its piece of contig, or null when it cannot be written yet.
 *
 * 🔴 null RATHER THAN A GUESS. An empty ligand code or a backwards range is a
 * row the user is still filling in, not a span to emit something plausible for -
 * and a contig assembled around a guess fails deep inside the pipeline with a
 * message about a token count.
 */
export function spanToContig(span) {
  if (span.kind === "break") return "/0";
  // 🔴 A GIVEN SEQUENCE HAS NO CONTIG NOTATION, because a contig names
  // coordinates. Hunter reads these spans through `chainsOf` instead; for RFD4
  // the row is marked unavailable and this returns null so nothing is emitted.
  if (span.kind === "sequence") return null;
  if (span.kind === "ligand") {
    const code = String(span.code ?? "").trim().toUpperCase();
    return code.length === 0 ? null : code;
  }
  if (span.kind === "segment") {
    const chain = String(span.chain ?? "").trim();
    const first = number(span.first, NaN);
    const last = number(span.last, NaN);
    if (chain.length === 0 || !Number.isFinite(first) || !Number.isFinite(last)) return null;
    if (last < first) return null;
    return `${chain}${first}-${last}`;
  }
  const min = number(span.min, NaN);
  const max = number(span.max, min);
  if (!Number.isFinite(min) || min < 1) return null;
  if (!Number.isFinite(max) || max < min) return null;
  // 🔴 A NAMED SPAN BECOMES A RESIDUE NAME, which is how a secondary-structure
  // or CATH condition selects it: `sel('*/h1/*/*')` reads field two of the path,
  // the res_name. So the label is not decoration - it is the only handle a
  // condition has on a stretch of residues that does not exist yet.
  const name = String(span.name ?? "").trim();
  const suffix = SPAN_POLYMER_SUFFIX[span.polymer ?? "protein"] ?? "P";
  if (name.length === 0) return `${min}-${max}`;
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(name)) return null;
  return `${name}:${min}-${max}${suffix}`;
}

/**
 * The whole list as a contig string, and what is wrong with it.
 *
 * @param {object[]} spans
 * @param {string} [method]
 * @returns {{contig: string, problems: string[], tokens: {min: number, max: number}}}
 */
export function contigOf(spans, method = "rfd4p") {
  const limits = METHOD_LIMITS[method] ?? {};
  const pieces = [];
  const problems = [];
  let min = 0;
  let max = 0;
  for (const [index, span] of spans.entries()) {
    if (span.kind === "sequence") {
      if (limits.sequence !== undefined) problems.push(limits.sequence);
      else min += String(span.value ?? "").trim().length;
      if (limits.sequence === undefined) max = min;
      continue;
    }
    const piece = spanToContig(span);
    if (piece === null) {
      problems.push(`span ${index + 1} is not filled in yet`);
      continue;
    }
    pieces.push(piece);
    if (span.kind === "generated") {
      const low = number(span.min, 0);
      const high = number(span.max, low);
      min += low;
      max += high;
      if (high !== low && limits.range !== undefined) problems.push(limits.range);
      if (String(span.name ?? "").trim().length > 0 && limits.labelled !== undefined) {
        problems.push(limits.labelled);
      }
    } else if (span.kind === "segment") {
      if (String(span.hotspots ?? "").trim().length > 0 && limits.hotspots !== undefined) {
        problems.push(limits.hotspots);
      }
      const span0 = number(span.first, 0);
      const span1 = number(span.last, span0);
      min += span1 - span0 + 1;
      max += span1 - span0 + 1;
    } else if (span.kind === "ligand" && limits.ligand !== undefined) {
      problems.push(limits.ligand);
    }
  }
  if (pieces.length === 0) problems.push("add at least one span");
  // 🔴 A LIST THAT IS NOTHING BUT BREAKS IS NOT A JOB. It writes a contig the
  // pipeline accepts and that asks for no residues at all.
  if (pieces.length > 0 && pieces.every((piece) => piece === "/0")) {
    problems.push("every span is a chain break, so nothing would be designed");
  }
  return { contig: pieces.join(","), problems: [...new Set(problems)], tokens: { min, max } };
}

/**
 * The modified residues a span list names, as the fold wants them.
 *
 * 🔴 POSITION IS WITHIN THE CHAIN AND 1-BASED, because that is how a PDB and a
 * paper number residues, and this field is read by a person. `chainsOf` turns it
 * into the chain letters the fold uses.
 *
 * @param {object[]} spans
 * @returns {{chainOffset: number, position: number, code: string}[]} positions
 *   relative to the span's own chain, for `chainsOf` to place
 */
export function modificationsOf(spans) {
  const out = [];
  let chainOffset = 0;
  let within = 0;
  for (const span of spans) {
    if (span.kind === "break") { chainOffset += 1; within = 0; continue; }
    if (span.kind === "ligand") continue;
    const length = span.kind === "generated"
      ? number(span.min, 0)
      : (span.kind === "sequence"
        ? String(span.value ?? "").trim().length
        : number(span.last, 0) - number(span.first, 0) + 1);
    for (const entry of String(span.modifications ?? "").split(/[\s,]+/).filter(Boolean)) {
      const [position, code] = entry.split(":");
      const at = Number(position);
      if (!Number.isFinite(at) || code === undefined || code.length === 0) continue;
      out.push({ chainOffset, position: within + at, code: code.toUpperCase() });
    }
    within += length;
  }
  return out;
}

/** A hotspot field's residues, ranges expanded: `40 60-62` is 40, 60, 61, 62. */
export function hotspotNumbers(text) {
  return String(text ?? "").split(/[^0-9-]+/).filter(Boolean).flatMap((piece) => {
    const [first, last = first] = piece.split(/(?<=\d)-/).map(Number);
    return Array.from({ length: Math.max(0, last - first + 1) }, (_, k) => first + k);
  });
}

/**
 * The hotspots a span list names, as Protein Hunter's fold numbers them: per fold chain letter, by
 * position along that chain from 1.
 *
 * 🔴 THE FOLD'S NUMBERING, NOT THE STRUCTURE'S. Hunter folds the job and checks the design touches
 * these residues in THAT prediction, whose chains are lettered by job order (CHAIN_IDS, as `chainsOf`
 * builds them) and numbered from 1. This returned the structure's chain id and residue numbers, which
 * agree only when the target is the job's chain of that letter and numbered from 1 - true of the
 * PD-L1 example and nothing else: a target kept as B5-50 behind a designed chain was asked for B10,
 * the fold's B10 being the structure's B14. A kept segment's residue r sits at its chain's running
 * length plus r - first + 1; a sequence's hotspots are already positions along it.
 *
 * (RFD4-Proteina reads the structure's own numbering, through `conditionsOf`.)
 *
 * @param {object[]} spans with any length ranges already drawn (`drawLengths`): a designed range
 *   before a hotspot in the same chain moves it
 * @returns {Record<string, number[]>|undefined}
 */
export function hotspotsOf(spans) {
  const chainOf = chainOfSpans(spans);
  const running = new Map();
  const hot = new Map();
  spans.forEach((span, index) => {
    const chain = chainOf[index];
    if (chain === null) return;
    const before = running.get(chain) ?? 0;
    const length = span.kind === "generated" ? number(span.min, 0)
      : span.kind === "sequence" ? String(span.value ?? "").trim().length
        : span.kind === "segment" ? Math.max(0, number(span.last, 0) - number(span.first, 0) + 1) : 0;
    running.set(chain, before + length);
    if (span.kind !== "segment" && span.kind !== "sequence") return;
    const at = hotspotNumbers(span.hotspots).map((r) => (span.kind === "segment" ? r - number(span.first, 0) + 1 : r))
      .filter((p) => p >= 1 && p <= length).map((p) => before + p);
    if (at.length === 0) return;
    const letter = CHAIN_IDS[chain];
    hot.set(letter, [...(hot.get(letter) ?? []), ...at]);
  });
  return hot.size === 0 ? undefined : Object.fromEntries(hot);
}

/**
 * The conditions a span list asks RFD4-Proteina for.
 *
 * 🔴 TODAY THAT IS HOTSPOTS AND NOTHING ELSE. `C_HOT` marks residues of a GIVEN
 * chain that the designed chain should sit against - an epitope - and it is the
 * one condition a user can reasonably name by hand. The rest (`C_CRD`, `C_SEQ`,
 * `C_CRDGRP`) are structural bookkeeping a preset sets as a whole.
 *
 * 🔴 THE RESIDUE LIST IS BRACKETED, AND THAT IS NOT COSMETIC.
 * `evalPathSelection` splits a selection on commas before it splits a path on
 * slashes, so a bare list conditions on its first residue alone and says
 * nothing - which is what two shipped presets were doing. See
 * test/rfd4p-selection.test.js.
 *
 * @param {object[]} spans
 * @returns {object|undefined} a condition spec, or nothing to condition on
 */
export function conditionsOf(spans) {
  const hot = new Map();
  for (const span of spans) {
    if (span.kind !== "segment" && span.kind !== "sequence") continue;
    const residues = String(span.hotspots ?? "").split(/[^0-9-]+/).filter(Boolean);
    if (residues.length === 0) continue;
    // A sequence span has no chain id of its own in the structure; only a
    // `From structure` span can name one.
    if (span.kind !== "segment") continue;
    const chain = String(span.chain ?? "").trim();
    if (chain.length === 0) continue;
    const seen = hot.get(chain) ?? [];
    hot.set(chain, [...seen, ...residues]);
  }
  // 🔴 SECONDARY STRUCTURE IS ASKED FOR BY NAME, grouped by what was asked. The
  // selections are OR-ed with `|` because that is what the shipped
  // fold-conditioned preset does, and a condition takes one expression per arm.
  const structures = new Map();
  for (const span of spans) {
    if (span.kind !== "generated") continue;
    const name = String(span.name ?? "").trim();
    const wanted = String(span.secondary ?? "").trim();
    if (name.length === 0 || wanted.length === 0) continue;
    structures.set(wanted, [...(structures.get(wanted) ?? []), name]);
  }

  const conditions = {};
  if (hot.size > 0) {
    conditions.C_HOT = {
      true: [...hot.entries()].map(([chain, residues]) => ({
        select: `sel('${chain}/*/[${residues.join(",")}]/*')`,
      })),
    };
  }
  if (structures.size > 0) {
    conditions.C_SEC = Object.fromEntries([...structures.entries()].map(([wanted, names]) => [
      wanted, [{ select: names.map((name) => `sel('*/${name}/*/*')`).join(" | ") }],
    ]));
  }
  return Object.keys(conditions).length === 0 ? undefined : conditions;
}

/** One contig string back into spans, or null for a piece this editor has no row for. */
export const spansFromContig = (contig) => {
  const spans = [];
  for (const raw of String(contig).split(",")) {
    const piece = raw.trim();
    if (piece.length === 0) continue;
    if (piece === "/0" || /^\/\d+$/.test(piece)) { spans.push({ kind: "break" }); continue; }
    // 🔴 THE CHAIN IS LETTERS ONLY, AND THE GREEDY VERSION SPLIT `A16-35` AS
    // CHAIN "A1", FIRST 6. This is `parseContigToken`'s own expression in
    // shared/rfd4p/pipeline/spec.js, which is the parser the engine actually
    // uses; ours read `[A-Za-z][A-Za-z0-9]*`, which can absorb digits and then
    // backtracks to leave the first residue a digit short.
    //
    // 🔴 AND A ROUND TRIP CANNOT SEE IT, which is why it survived a check
    // against six shipped presets byte for byte: chain "A1" with first 6 joins
    // back to exactly "A16-35". The only thing that catches a mis-split is
    // comparing the FIELDS, so test/design-contig-spans.test.js does that.
    //
    // 🔴 AND THE RANGE IS OPTIONAL: `A16` is one residue, which the engine
    // accepts and this editor used to refuse outright - the whole preset fell
    // through to "no row for it yet".
    const segment = /^([A-Za-z]+)(\d+)(?:-(\d+))?$/.exec(piece);
    if (segment !== null) {
      const first = Number(segment[2]);
      spans.push({ kind: "segment", chain: segment[1],
        first, last: segment[3] === undefined ? first : Number(segment[3]) });
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

/**
 * Resolve every length RANGE to a concrete length, once per run.
 *
 * 🔴 THIS IS WHAT "A SEARCH OVER LENGTHS" MEANS, AND THE PAGE ALREADY HAD THE
 * SEARCH. A contig writes `5-20` for "somewhere between 5 and 20 residues".
 * Protein Hunter folds one length at a time, so a range used to be refused
 * outright - but `Runs` repeats the whole hunt, so giving each run its own draw
 * turns the range into exactly the search the refusal said was missing.
 *
 * 🔴 DETERMINISTIC, NOT RANDOM. Two hunts with the same seed have to be the
 * same hunt, and a draw taken from `Math.random` would make a run impossible to
 * reproduce or to report. The length comes from a hash of the seed, the run and
 * WHICH span it is - the last of those because two ranged spans in one contig
 * are independent choices, and spreading them together would only ever try
 * "all short" and "all long".
 *
 * A span whose min and max agree is returned untouched, so a job with no ranges
 * is unaffected and `spans === drawLengths(spans, ...)` holds for it elementwise.
 *
 * @param {object[]} spans
 * @param {object} [options]
 * @param {number} [options.run=0] which run this is
 * @param {number} [options.seed=0]
 * @returns {object[]} the same spans with each generated range collapsed
 */
export function drawLengths(spans, options = {}) {
  const run = Math.max(0, Math.floor(options.run ?? 0));
  const seed = Math.floor(options.seed ?? 0);
  return spans.map((span, index) => {
    if (span.kind !== "generated") return span;
    const low = Math.min(number(span.min, 0), number(span.max, 0));
    const high = Math.max(number(span.min, 0), number(span.max, 0));
    if (high <= low) return span;
    // A small integer hash: enough to decorrelate the spans and the runs, and
    // reproducible anywhere this runs.
    let bits = (seed * 0x9e3779b1 + run * 0x85ebca6b + index * 0xc2b2ae35) >>> 0;
    bits ^= bits >>> 15;
    bits = Math.imul(bits, 0x2545f491) >>> 0;
    bits ^= bits >>> 13;
    const length = low + (bits >>> 0) % (high - low + 1);
    return { ...span, min: length, max: length };
  });
}

/**
 * The same spans as Protein Hunter's chains.
 *
 * 🔴 TWO DERIVATIONS OF ONE LIST, WHICH IS THE WHOLE POINT. `contigOf` writes
 * RFD4-Proteina's notation; this writes what the hunt loop takes - an array of
 * chain sequences with one of them empty, the one being designed. Neither method
 * gets its own input screen.
 *
 * A `break` ends a chain. A `generated` span is the designed chain and its
 * length; a `sequence` span is a chain given whole. 🔴 A chain mixing the two -
 * a motif inside a designed chain - is not something Hunter can express, and is
 * reported rather than flattened.
 *
 * 🔴 A `From structure` SPAN REACHES HUNTER THROUGH ITS SEQUENCE, not its
 * coordinates - that is the geometry axis once more. The caller supplies
 * `sequenceOf`, backed by whatever structure is loaded; without one the span is
 * reported rather than skipped, because a hunt that silently dropped the target
 * would fold the binder alone and score it against nothing.
 *
 * @param {object[]} spans
 * @param {object} [options]
 * @param {(chain: string, first: number, last: number) => string|null} [options.sequenceOf]
 * @param {(chain: string) => string} [options.polymerOf] what a loaded structure's chain is made of
 *   ("protein" | "dna" | "rna"); a kept DNA strand folds as DNA, not as a protein of unknowns
 * @returns {{chains: string[], chainKinds: string[], ligandCodes: string[],
 *            chainIndex: number, length: number, hold: number[],
 *            problems: string[]}} `hold` is positions within the designed chain
 *   the designer must not rewrite - see designChain() in cpu/design/mpnn-bridge.js
 */
export function chainsOf(spans, options = {}) {
  const sequenceOf = options.sequenceOf;
  const polymerOf = options.polymerOf ?? (() => "protein");
  const chains = [];
  const kinds = [];
  const ligands = [];
  const modifications = [];
  const problems = [];
  let hold = [];
  let designedIndex = -1;
  let designedLength = 0;
  let current = { parts: [], polymer: "protein", designed: false, length: 0, given: false };

  const finish = () => {
    if (current.parts.length === 0) return;
    const index = chains.length;
    // 🔴 THE CHAIN IS BUILT PART BY PART, AND THE HELD POSITIONS ARE THE GIVEN
    // PARTS' INDICES INSIDE IT. That is what lets one chain mix given and
    // designed residues - a scaffolded motif, an antibody's framework around its
    // CDRs - which Hunter could not express at all before: it designed whole
    // chains only.
    let text = "";
    const held = [];
    for (const part of current.parts) {
      if (part.held) {
        for (let at = 0; at < part.text.length; at += 1) held.push(text.length + at);
      }
      text += part.text;
    }
    // 🔴 A WHOLLY DESIGNED CHAIN IS HANDED OVER EMPTY, AND THAT IS NOT A
    // DETAIL: the loop draws its own starting sequence when the chain is empty -
    // the percent-X ramp, the first cycle's proline omission, the seeded draw -
    // and all of that is Protein Hunter's METHOD, held to the reference in
    // cpu/design/sample-sequence.js. Pre-filling it with X would silently
    // replace the method's own opening move with a constant.
    //
    // 🔴 A MIXED CHAIN CANNOT BE, because the loop has no way to know where the
    // held residues go. So a scaffolded chain arrives pre-shaped, and the
    // percent-X ramp does NOT apply to it - a deviation from the reference's
    // schedule that only arises in a case the reference cannot express at all.
    // 🔴 EMPTIED ONLY WHEN EVERY RESIDUE CAME FROM A `Design` SPAN. A chain that
    // is wholly FREED - a given span marked redesign, which is inverse folding -
    // keeps its text, because that sequence IS the starting point the method
    // works from. Emptying it would throw away the thing being inverse-folded
    // and draw a random mostly-X chain of the same length instead.
    const wholly = current.designed && held.length === 0 && !current.given;
    chains.push(wholly ? "" : text);
    kinds.push(current.designed ? "protein" : current.polymer);
    if (current.designed) {
      if (designedIndex >= 0) problems.push("Protein Hunter designs one chain at a time");
      designedIndex = index;
      designedLength = current.length;
      hold = held;
    }
    current = { parts: [], polymer: "protein", designed: false, length: 0, given: false };
  };

  for (const span of spans) {
    if (span.kind === "break") { finish(); continue; }
    if (span.kind === "ligand") {
      const code = String(span.code ?? "").trim().toUpperCase();
      if (code.length > 0) ligands.push(code);
      continue;
    }
    if (span.kind === "generated") {
      current.designed = true;
      const length = number(span.min, 0);
      current.length += length;
      // 🔴 X IS WHAT THE LOOP FOLDS AND WHAT MPNN REWRITES. A designed span is
      // that many unknowns in the starting sequence; the loop only draws its own
      // when the whole chain is empty, so a mixed chain arrives already shaped.
      current.parts.push({ text: "X".repeat(length), held: false });
      continue;
    }
    // ...a given span: its residues, held unless the user asked for them back
    let text = "";
    const polymer = span.kind === "sequence" ? span.polymer ?? "protein" : polymerOf(String(span.chain ?? ""));
    if (current.parts.length === 0) current.polymer = polymer;
    else if (polymer !== current.polymer) {
      problems.push("a chain cannot be part protein and part nucleic;"
        + " put a chain break between them");
    }
    if (span.kind === "sequence") {
      text = String(span.value ?? "").trim().toUpperCase();
      if (text.length === 0) problems.push("a given sequence is empty");
    } else {
      if (sequenceOf === undefined) {
        problems.push("Protein Hunter folds sequences, so a \u201cKeep from structure\u201d span needs"
          + " the structure loaded to read its sequence from");
        continue;
      }
      const found = sequenceOf(String(span.chain ?? ""), number(span.first, 0), number(span.last, 0));
      if (found === null || found.length === 0) {
        problems.push(`chain ${span.chain} ${span.first}-${span.last} is not in the loaded structure`);
        continue;
      }
      text = found;
    }
    // 🔴 `redesign` IS THE SEQUENCE AXIS, MADE EDITABLE. A given span is held by
    // default; asked to be redesigned it keeps its place and its length and lets
    // the designer rewrite its letters - which is inverse folding when it is the
    // whole chain, and an antibody's CDR when it is part of one.
    const redesign = span.redesign === true;
    if (redesign) current.designed = true;
    current.given = true;
    current.parts.push({ text, held: !redesign });
  }
  finish();

  if (designedIndex < 0) {
    problems.push("nothing is being designed; add a Design span or mark a given one Redesign");
  }
  // 🔴 A CHAIN THAT IS ENTIRELY HELD IS NOT A DESIGN, and the designer says so
  // too - but saying it here names the spans rather than the residue count.
  if (designedIndex >= 0 && hold.length > 0 && hold.length === chains[designedIndex]?.length) {
    problems.push("every residue of the designed chain is held, so there is nothing to design");
  }

  // 🔴 A MODIFICATION'S CHAIN IS THE LETTER THE FOLD USES, resolved here because
  // a span knows only its own offset: the editor counts chains by breaks, and
  // CHAIN_IDS is the alphabet the loop and the fold agree on.
  const placed = modificationsOf(spans).map((entry) => ({
    chain: CHAIN_IDS[entry.chainOffset],
    position: entry.position,
    code: entry.code,
  })).filter((entry) => entry.chain !== undefined);
  for (const entry of placed) {
    const chain = chains[CHAIN_IDS.indexOf(entry.chain)];
    if (chain !== undefined && chain.length > 0 && entry.position > chain.length) {
      problems.push(`chain ${entry.chain} has ${chain.length} residues,`
        + ` so position ${entry.position} cannot be modified`);
    }
  }

  return {
    chains, chainKinds: kinds, ligandCodes: ligands, chainIndex: designedIndex,
    length: designedLength, hold, modifications: placed,
    problems: [...new Set(problems)],
  };
}

/**
 * A list of spans, as rows.
 *
 * @param {HTMLElement} rowsContainer
 * @param {HTMLElement} addButton
 * @param {object} [options]
 * @param {object[]} [options.initial]
 * @param {() => string} [options.method] which method is selected, read when
 *   the list repaints - so changing method re-marks the rows
 * @param {(spans: object[]) => void} [options.onChange]
 * @returns {{read: () => object[], set: (spans: object[]) => void,
 *            contig: () => object}}
 */
/**
 * The pieces an ANNOTATED structure is, read off it - for the examples that run from their file's
 * own annotations rather than from their contig.
 *
 * 🔴 THE CONTIG THOSE EXAMPLES CARRY IS A LABEL IN ANOTHER NUMBERING. The binder example's says
 * "keep A18-132, design 76" while its file - which is what runs - designs chain A at 50-120 residues
 * against chain B 1-115, and its hotspots are on B; the page drew the label as the job. Here every
 * run of residues is a piece in file order: an expandable-segment sentinel (`annotation_expsegmin`/
 * `max`) a designed span of that length range, polymer residues a kept span per contiguous run, a
 * non-polymer residue a ligand, and a chain change a break.
 *
 * @param {object} atoms `parseCifToAtomArray`'s array
 * @returns {object[] | null} the spans, or null when the file carries no expandable segments
 */
export function spansFromAnnotatedAtoms(atoms) {
  const ann = atoms?.annotations ?? {};
  const min = ann.annotation_expsegmin_1_residue ?? atoms?.annotation_expsegmin_1_residue;
  const max = ann.annotation_expsegmax_1_residue ?? atoms?.annotation_expsegmax_1_residue;
  if (!min || !max) return null;
  const n = atoms.chain_id.length;
  const spans = [];
  let chain = null;
  let last = null;     // the residue the current span ended on, "chain:res"
  for (let i = 0; i < n; i += 1) {
    const c = String(atoms.chain_id[i]);
    const r = Number(atoms.res_id[i]);
    if (last === `${c}:${r}`) continue;
    if (chain !== null && c !== chain) spans.push({ kind: "break" });
    const previous = spans[spans.length - 1];
    if (Number(min[i]) >= 0) {
      spans.push({ kind: "generated", min: Number(min[i]), max: Number(max[i]), name: "", secondary: "", polymer: "protein" });
    } else if (atoms.is_polymer?.[i] === false || (atoms.is_polymer === undefined && atoms.hetero?.[i])) {
      spans.push({ kind: "ligand", code: String(atoms.res_name[i]).trim() });
    } else if (previous?.kind === "segment" && previous.chain === c && r === previous.last + 1) {
      previous.last = r;
    } else {
      spans.push({ kind: "segment", chain: c, first: r, last: r, hotspots: "" });
    }
    chain = c;
    last = `${c}:${r}`;
  }
  return spans.some((span) => span.kind === "generated") ? spans : null;
}

/**
 * The job at a glance: a row per chain, a block per span - kept from the structure (solid, in the
 * chain's colour), designed (striped), a known sequence (outlined) - and ligands as chips beside
 * them. A block's width follows its residue count (a designed range by its middle), so a 20-residue
 * motif inside a 100-residue design looks it. `onPick(index)` is called with the clicked span.
 *
 * @param {HTMLElement} container
 * @param {object[]} spans
 * @param {{ onPick?: (index: number) => void, selected?: number,
 *   hotspots?: {chain: string, resId: number}[] }} [options] `hotspots` are counted on the kept
 *   piece that holds them, so the block says where the design is aimed
 */
export function drawChainDiagram(container, spans, { onPick = () => {}, selected = -1, hotspots = [] } = {}) {
  const chainOf = chainOfSpans(spans);
  const rows = new Map();
  const ligands = [];
  const sizeOf = (span) => {
    if (span.kind === "generated") return (number(span.min, 1) + number(span.max, number(span.min, 1))) / 2;
    if (span.kind === "segment") return Math.max(1, number(span.last, 1) - number(span.first, 1) + 1);
    if (span.kind === "sequence") return Math.max(1, String(span.value ?? "").length);
    return 0;
  };
  const labelOf = (span) => {
    if (span.kind === "generated") {
      const min = number(span.min, 0);
      const max = number(span.max, min);
      // the stripes say "designed"; the block says how many
      return min === max ? `${min}` : `${min}\u2013${max}`;
    }
    if (span.kind === "segment") return `${span.chain ?? "?"}${span.first ?? ""}\u2013${span.last ?? ""}`;
    if (span.kind === "sequence") return `sequence (${String(span.value ?? "").length})`;
    return "";
  };
  spans.forEach((span, index) => {
    if (span.kind === "ligand") { ligands.push(index); return; }
    const chain = chainOf[index];
    if (chain === null) return;
    if (!rows.has(chain)) rows.set(chain, []);
    rows.get(chain).push(index);
  });
  const out = [];
  for (const [chain, indices] of rows) {
    const row = element("div", "chain-row");
    row.style.setProperty("--chain-colour", chainColour(chain));
    row.append(element("span", "chain-name", CHAIN_IDS[chain] ?? "?"));
    const track = element("div", "chain-track");
    for (const index of indices) {
      const span = spans[index];
      const block = element("button", `chain-block chain-block-${span.kind}`, labelOf(span));
      block.type = "button";
      block.style.flexGrow = String(sizeOf(span));
      block.title = `${SPAN_LABELS[span.kind]}: ${labelOf(span)} - click to edit`;
      if (span.kind === "segment") {
        const held = hotspots.filter((h) => h.chain === String(span.chain).trim()
          && h.resId >= Number(span.first) && h.resId <= Number(span.last)).length;
        if (held > 0) {
          const mark = element("span", "chain-block-hot", `${held} hotspot${held === 1 ? "" : "s"}`);
          block.append(" \u00b7 ", mark);
          block.title += ` - ${held} hotspot${held === 1 ? "" : "s"} on it`;
        }
      }
      if (index === selected) block.classList.add("selected");
      block.addEventListener("click", () => onPick(index));
      track.append(block);
    }
    row.append(track);
    out.push(row);
  }
  if (ligands.length > 0) {
    const row = element("div", "chain-row chain-row-ligands");
    row.append(element("span", "chain-name", ""));
    const track = element("div", "chain-track");
    for (const index of ligands) {
      const block = element("button", "chain-block chain-block-ligand", String(spans[index].code ?? "").trim() || "ligand");
      block.type = "button";
      block.title = "Ligand - click to edit";
      block.addEventListener("click", () => onPick(index));
      track.append(block);
    }
    row.append(track);
    out.push(row);
  }
  container.replaceChildren(...out);
}

/**
 * What "Add a segment" offers, in order: [value, label, the span it adds]. The three sequence
 * entries are one kind with its polymer set, because a reader adds "a DNA strand", not "a known
 * sequence" and then finds the polymer.
 */
export const ADD_CHOICES = [
  ["generated", "Design new residues", () => emptySpan("generated")],
  ["protein", "Protein sequence", () => ({ ...emptySpan("sequence"), polymer: "protein" })],
  ["dna", "DNA sequence", () => ({ ...emptySpan("sequence"), polymer: "dna" })],
  ["rna", "RNA sequence", () => ({ ...emptySpan("sequence"), polymer: "rna" })],
  ["segment", "Keep from the structure", () => emptySpan("segment")],
  ["ligand", "Ligand", () => emptySpan("ligand")],
  ["break", "Chain break", () => emptySpan("break")],
];

/**
 * The job as a list of segment cards, each holding every setting that segment has.
 *
 * 🔴 A CARD, NOT A ROW AND A HIDDEN DISCLOSURE (2026-10-27). A segment's settings lived behind a
 * "⋯" button - a binder's hotspots among them - so the thing that makes a binder a binder was one
 * click away from where its target was named, and the page drew a separate hotspot line to make up
 * for it. Each card now shows what that segment is: a length, a chain and its residues and the
 * hotspots on them, a sequence and what it is made of, a ligand's code - as a two-column form, a
 * label beside each control, one setting a line.
 *
 * 🔴 AND A CONTROL THE METHOD CANNOT USE IS NOT DRAWN. Secondary structure and a designed span's
 * polymer are RFD4-Proteina's; under Protein Hunter they are absent, and a value an example set is
 * named in a warning on the card rather than silently kept.
 *
 * @param {HTMLElement} rowsContainer
 * @param {HTMLSelectElement} addControl the "Add a segment" picker; choosing appends that segment
 * @param {object} [options]
 * @param {object[]} [options.initial]
 * @param {() => string} [options.method] the selected method, read on every draw
 * @param {(spans: object[]) => void} [options.onChange]
 * @param {(span: object) => string} [options.hotspotsFromSelection] the viewer's selected residues
 *   inside this segment, as the hotspot field writes them; throws a sentence when there are none
 * @returns {{read: () => object[], set: (spans: object[]) => void, contig: () => object,
 *            repaint: () => void, reveal: (index: number) => void}}
 */
export function createContigList(rowsContainer, addControl, options = {}) {
  const notify = options.onChange ?? (() => {});
  const methodOf = options.method ?? (() => "rfd4p");
  let spans = (options.initial ?? [emptySpan("generated")]).map((span) => ({ ...span }));
  // Which card a click elsewhere (the diagram, the viewer) pointed at. Kept outside `spans`: it is
  // about looking, not about the job.
  let selected = -1;
  let uid = 0;

  const draw = () => {
    const method = methodOf();
    const limits = METHOD_LIMITS[method] ?? {};
    const rfd4 = method === "rfd4p";
    rowsContainer.replaceChildren();
    const chainOf = chainOfSpans(spans);

    for (const [index, span] of spans.entries()) {
      const row = element("div", `contig-row contig-row-${span.kind}`);
      row.setAttribute("role", "group");
      if (index === selected) row.classList.add("selected");
      const chain = chainOf[index];
      if (chain !== null) {
        row.style.setProperty("--chain-colour", chainColour(chain));
        row.classList.add("contig-row-chained");
      } else if (span.kind === "ligand") {
        row.style.setProperty("--chain-colour", LIGAND_COLOUR);
        row.classList.add("contig-row-chained");
      }
      const where = chain !== null ? `chain ${CHAIN_IDS[chain] ?? "?"}` : span.kind === "ligand" ? "ligand" : "";
      row.setAttribute("aria-label", `Segment ${index + 1}: ${SPAN_LABELS[span.kind]}${where ? `, ${where}` : ""}`);

      // ---- the head: what the segment is, which chain, remove -------------------------------
      const head = element("div", "contig-head");
      const picker = element("select", "entity-picker contig-kind");
      picker.setAttribute("aria-label", `Segment ${index + 1} type`);
      picker.title = SPAN_NOTES[span.kind] ?? "";
      for (const kind of SPAN_KINDS) {
        const option = element("option", undefined, SPAN_LABELS[kind]);
        option.value = kind;
        option.selected = kind === span.kind;
        picker.append(option);
      }
      picker.addEventListener("change", () => {
        // 🔴 A NEW SPAN, NOT THE OLD ONE RELABELLED: a segment's `first` is not a length.
        spans[index] = emptySpan(picker.value);
        changed();
      });
      head.append(picker);
      if (where) head.append(element("span", "contig-chain-tag", where));
      // a break has nothing to fill in: its card is one line saying what it does
      if (span.kind === "break") head.append(element("span", "contig-break", "the next segment starts a new chain"));
      const remove = element("button", "btn btn-grey btn-small entity-remove");
      remove.type = "button";
      remove.innerHTML = '<i class="fa-solid fa-xmark" aria-hidden="true"></i>';
      remove.title = "Remove this segment";
      remove.setAttribute("aria-label", `Remove segment ${index + 1}`);
      // The last one cannot go: an empty list is a job that cannot run.
      remove.disabled = spans.length === 1;
      remove.addEventListener("click", () => {
        spans.splice(index, 1);
        selected = -1;
        changed();
      });
      head.append(remove);
      row.append(head);

      // ---- the fields: a two-column form, one setting a line ----------------------------------
      // 🔴 LABEL LEFT, CONTROL RIGHT, ONE LINE EACH (2026-10-27, third pass). The fields flowed and
      // wrapped where they happened to fit, with a "More" disclosure pushing more of them into the same
      // flow - so a label could sit at the end of one line and its box at the start of the next, and a
      // card read as a jumble. A grid keeps every label beside its control at any width.
      const fields = element("div", "contig-fields");
      const notes = [];
      const id = () => `contig-${index}-${uid += 1}`;
      /** One line of the form: its label, and the control(s) it names. */
      const line = (labelText, controls, { className = "", hint } = {}) => {
        const label = element("label", "contig-label", labelText);
        const first = controls.find((c) => c instanceof HTMLInputElement || c instanceof HTMLSelectElement
          || c instanceof HTMLTextAreaElement);
        if (first) {
          first.id = first.id || id();
          label.htmlFor = first.id;
        }
        if (className === "contig-hot") label.classList.add("contig-label-hot");
        const cell = element("div", `contig-control ${className}`.trim());
        cell.append(...controls);
        if (hint) cell.append(element("div", "contig-hint", hint));
        fields.append(label, cell);
        return cell;
      };
      const bind = (input, key, transform = (value) => value) => {
        input.addEventListener("input", () => {
          spans[index] = { ...spans[index], [key]: transform(input.value) };
          // The summary updates on every keystroke; the cards are not redrawn, which would steal
          // focus mid-number.
          notify(read());
        });
      };
      const pick = (title, choices, key, value) => {
        const control = element("select", "entity-picker");
        control.title = title;
        for (const [v, label] of choices) {
          const option = element("option", undefined, label);
          option.value = v;
          option.selected = v === value;
          control.append(option);
        }
        control.addEventListener("change", () => {
          spans[index] = { ...spans[index], [key]: control.value };
          changed();
        });
        return control;
      };
      const text = (className, value, placeholder, title, key, transform) => {
        const control = element("input", `entity-value ${className}`);
        control.value = String(value ?? "");
        if (placeholder !== undefined) control.placeholder = placeholder;
        control.title = title;
        bind(control, key, transform);
        return control;
      };
      const numberBox = (value, title, key, label) => {
        const control = text("contig-number", value, undefined, title, key, (v) => number(v, 1));
        control.type = "number";
        control.min = "1";
        control.inputMode = "numeric";
        if (label) control.setAttribute("aria-label", label);
        return control;
      };
      const tick = (labelText, title, key) => {
        const box = element("input");
        box.type = "checkbox";
        box.checked = span[key] === true;
        box.title = title;
        box.addEventListener("change", () => {
          spans[index] = { ...spans[index], [key]: box.checked };
          changed();
        });
        return line(labelText, [box, element("span", "contig-unit", title)], { className: "contig-redesign" });
      };
      const word = (t) => element("span", "contig-unit", t);
      /** A segment's hotspots: a box, Use selection, and what a hotspot is - the same on both given kinds. */
      const hotspots = (what) => {
        const hot = text("contig-hotspots", span.hotspots, "e.g. 54 56 60-64",
          `Hotspots: residues of this segment the design should touch (the site to bind), ${what}.`,
          "hotspots", (v) => v.trim());
        const controls = [hot];
        if (options.hotspotsFromSelection && span.kind === "segment") {
          const use = element("button", "btn btn-grey btn-small contig-use-selection", "Use selection");
          use.type = "button";
          use.title = "Set this segment's hotspots to the residues selected in the viewer or the sequence.";
          use.addEventListener("click", () => {
            try {
              const value = options.hotspotsFromSelection(spans[index]);
              spans[index] = { ...spans[index], hotspots: value };
              changed();
            } catch (error) {
              const note = row.querySelector(".contig-hot-note") ?? element("div", "option-note warn contig-hot-note");
              note.textContent = error.message;
              row.append(note);
            }
          });
          controls.push(use);
        }
        line("Hotspots", controls, { className: "contig-hot",
          hint: `Where the design should bind, ${what}.` });
      };

      if (span.kind === "generated") {
        line("Length", [
          numberBox(span.min ?? 75, "How many residues to design - the shortest, if you give a range.", "min"),
          word("to"),
          numberBox(span.max ?? span.min ?? 75, "The longest. Leave it equal to design exactly one length.",
            "max", "longest length"),
          word("residues")]);
        if (rfd4) {
          const secondary = element("select", "entity-picker");
          secondary.title = "Ask for this segment's secondary structure.";
          for (const [v, label] of Object.entries(SPAN_STRUCTURES)) {
            const option = element("option", undefined, label);
            option.value = v;
            option.selected = v === (span.secondary ?? "");
            secondary.append(option);
          }
          secondary.addEventListener("change", () => {
            // 🔴 A SECONDARY STRUCTURE NEEDS A LABEL TO BE ASKED FOR (C_SEC selects a segment by the name
            // the contig gives it), so one is made here rather than left for the reader to find.
            let name = String(spans[index].name ?? "").trim();
            if (secondary.value !== "" && name === "") {
              const taken = new Set(spans.map((other) => String(other.name ?? "").trim()));
              const letter = { HELIX: "h", SHEET: "s", LOOP: "l" }[secondary.value] ?? "x";
              let n = 1;
              while (taken.has(`${letter}${n}`)) n += 1;
              name = `${letter}${n}`;
            }
            spans[index] = { ...spans[index], secondary: secondary.value, name };
            changed();
          });
          line("Structure", [secondary]);
          line("Made of", [pick("What this segment is made of.",
            [["protein", "protein"], ["rna", "RNA"], ["dna", "DNA"]], "polymer", span.polymer ?? "protein")]);
        }
        if (number(span.max, 0) !== number(span.min, 0) && limits.range !== undefined) notes.push(limits.range);
        if ((String(span.name ?? "").trim() || String(span.secondary ?? "").trim()) && limits.labelled !== undefined) {
          notes.push(limits.labelled);
        }
      } else if (span.kind === "sequence") {
        line("Made of", [pick("What this chain is. A nucleic chain folds one token per base.",
          [["protein", "protein"], ["dna", "DNA"], ["rna", "RNA"]], "polymer", span.polymer ?? "protein")]);
        const value = element("textarea", "entity-value contig-sequence");
        value.value = String(span.value ?? "");
        value.rows = 2;
        value.spellcheck = false;
        value.placeholder = span.polymer === "dna" ? "e.g. ATGCATGC" : span.polymer === "rna" ? "e.g. AUGCAUGC"
          : "one-letter sequence, e.g. MKTAYIAK";
        value.title = "A chain whose sequence you know.";
        bind(value, "value", (raw) => raw.replace(/\s+/g, "").toUpperCase());
        line("Sequence", [value]);
        // A target given by its sequence can be aimed at too, under Protein Hunter (RFD4 needs coordinates).
        if ((span.polymer ?? "protein") === "protein" && !rfd4) hotspots("numbered from 1 along this sequence");
        tick("Redesign", "let the designer rewrite these residues", "redesign");
        line("Modified", [text("contig-modified", span.modifications, "e.g. 16:SEP",
          "Modified residues, as position:CCD code. Positions are 1-based within the segment.",
          "modifications", (v) => v.trim())]);
        if (limits.sequence !== undefined) notes.push(limits.sequence);
      } else if (span.kind === "segment") {
        const chainBox = text("contig-chain", span.chain ?? "A", undefined,
          "Which chain of the loaded structure.", "chain", (v) => v.trim());
        chainBox.maxLength = 4;
        line("Chain", [chainBox, word("of the loaded structure")]);
        line("Residues", [
          numberBox(span.first ?? 1, "First residue, by the structure's own numbering.", "first"),
          word("to"),
          numberBox(span.last ?? 100, "Last residue, inclusive.", "last", "last residue")]);
        // 🔴 THE HOTSPOTS ARE THIS SEGMENT'S, SO THEY ARE ON ITS CARD, for both methods (RFD4 conditions
        // on them; Protein Hunter rejects a cycle whose design does not touch them).
        hotspots("by the structure's numbering");
        tick("Redesign", "let the designer rewrite these residues", "redesign");
        line("Modified", [text("contig-modified", span.modifications, "e.g. 16:SEP",
          "Modified residues, as position:CCD code. Positions are 1-based within the segment.",
          "modifications", (v) => v.trim())]);
        if (String(span.hotspots ?? "").trim().length > 0 && limits.hotspots !== undefined) notes.push(limits.hotspots);
      } else if (span.kind === "ligand") {
        const code = text("contig-code", span.code, "e.g. SAM",
          "The ligand's CCD code, as the PDB names it.", "code", (v) => v.trim().toUpperCase());
        code.maxLength = 5;
        line("CCD code", [code]);
        if (limits.ligand !== undefined) notes.push(limits.ligand);
      }
      if (fields.childElementCount > 0) row.append(fields);
      for (const reason of notes) {
        row.classList.add("contig-row-unavailable");
        row.append(element("div", "option-note warn contig-reason", reason));
      }
      row.addEventListener("focusin", () => {
        if (selected === index) return;
        selected = index;
        for (const card of rowsContainer.children) card.classList.toggle("selected", card === row);
      });
      rowsContainer.append(row);
    }
  };

  const read = () => spans.map((span) => ({ ...span }));
  const changed = () => { draw(); notify(read()); };

  // The picker's first entry is its prompt, and it goes back to it after each choice.
  addControl.replaceChildren(new Option("Add a segment…", ""),
    ...ADD_CHOICES.map(([value, label]) => new Option(label, value)));
  addControl.addEventListener("change", () => {
    const choice = ADD_CHOICES.find(([value]) => value === addControl.value);
    addControl.value = "";
    if (choice === undefined) return;
    // 🔴 A SEQUENCE OR A LIGAND ADDED IS A CHAIN OF ITS OWN, so it comes with the break before it:
    // appended to the last chain, a DNA strand made "a chain part protein and part nucleic".
    const own = ["protein", "dna", "rna", "ligand"].includes(choice[0]);
    const last = spans[spans.length - 1];
    if (own && last !== undefined && last.kind !== "break") spans.push(emptySpan("break"));
    spans.push(choice[2]());
    selected = spans.length - 1;
    changed();
    rowsContainer.lastElementChild?.querySelector("input, textarea, select:not(.contig-kind)")?.focus();
  });

  draw();

  return {
    read,
    set(next) {
      spans = (next ?? []).map((span) => ({ ...span }));
      if (spans.length === 0) spans = [emptySpan("generated")];
      selected = -1;
      changed();
    },
    contig() { return contigOf(read(), methodOf()); },
    repaint: draw,
    /** Bring a segment's card into view and mark it, for the chain diagram's blocks and the viewer. */
    reveal(index) {
      if (index < 0 || index >= spans.length) return;
      selected = index;
      draw();
      const row = rowsContainer.children[index];
      if (row === undefined) return;
      row.classList.add("contig-row-flash");
      row.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
      setTimeout(() => row.classList.remove("contig-row-flash"), 1200);
    },
    /** Which card is marked, or -1. */
    get selected() { return selected; },
  };
}

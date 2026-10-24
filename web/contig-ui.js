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

export const SPAN_LABELS = {
  generated: "Design",
  sequence: "Given sequence",
  segment: "From structure",
  ligand: "Ligand",
  break: "New chain",
};

/** What each kind is, shown as the picker's title. */
export const SPAN_NOTES = {
  generated: "Residues to design. Give one length, or a range for the method to choose from.",
  sequence: "A chain you know the sequence of - protein, DNA or RNA. Protein Hunter folds it;"
    + " RFD4-Proteina needs its coordinates, so use \u201cFrom structure\u201d there.",
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
 * different loop. And it has no pocket conditioning, so a ligand in the list
 * would change the fold and the designer but not be designed *against* in the
 * way RFD4 means.
 */
export const METHOD_LIMITS = {
  hunter: {
    range: "Protein Hunter folds one length at a time; a range needs a search over lengths.",
    ligand: "Protein Hunter has no pocket conditioning — a ligand changes the fold and the"
      + " designer, but is not conditioned on.",
  },
  rfd4p: {
    // 🔴 THE GEOMETRY AXIS, AS A LIMIT. RFD4 generates coordinates AROUND given
    // coordinates, so a chain known only by its sequence is not something it can
    // hold fixed - where Hunter, which reaches structure by folding, takes
    // exactly that.
    sequence: "RFD4-Proteina conditions on coordinates, not on a sequence — load the structure"
      + " and use \u201cFrom structure\u201d.",
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
  if (kind === "generated") return { kind, min: 75, max: 75 };
  if (kind === "sequence") return { kind, value: "", polymer: "protein" };
  if (kind === "segment") return { kind, chain: "A", first: 1, last: 100 };
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
  return `${min}-${max}`;
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
    } else if (span.kind === "segment") {
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
 * @returns {{chains: string[], chainKinds: string[], ligandCodes: string[],
 *            chainIndex: number, length: number, problems: string[]}}
 */
export function chainsOf(spans, options = {}) {
  const sequenceOf = options.sequenceOf;
  // 🔴 WHAT EACH CHAIN IS MADE OF, AND THE LIGANDS BESIDE THEM, because the FOLD
  // needs both: the featuriser gives a ligand a chain of its own and reads a
  // nucleic chain one token per base, so a DNA target folded as protein is not a
  // slower answer, it is a different one. They travel with the chains.
  const kinds = [];
  const ligands = [];
  const chains = [];
  const problems = [];
  let current = { parts: [], designed: false, length: 0, polymer: "protein" };
  const finish = () => {
    if (current.parts.length === 0 && !current.designed) return;
    if (current.designed && current.parts.length > 0) {
      problems.push("Protein Hunter cannot mix designed and given residues in one chain;"
        + " put a chain break between them");
    }
    chains.push(current.designed ? "" : current.parts.join(""));
    // A designed chain is always protein: nothing here designs nucleic acids.
    kinds.push(current.designed ? "protein" : current.polymer);
    if (current.designed) lengths.push(current.length);
    current = { parts: [], designed: false, length: 0, polymer: "protein" };
  };
  const lengths = [];
  for (const span of spans) {
    if (span.kind === "break") { finish(); continue; }
    if (span.kind === "sequence") {
      const value = String(span.value ?? "").trim().toUpperCase();
      if (value.length === 0) problems.push("a given sequence is empty");
      current.parts.push(value);
      // 🔴 THE FIRST SPAN OF A CHAIN DECIDES ITS TYPE, and mixing two inside one
      // chain is reported rather than resolved - a chain is one polymer.
      if (current.parts.length === 1) current.polymer = span.polymer ?? "protein";
      else if ((span.polymer ?? "protein") !== current.polymer) {
        problems.push("a chain cannot be part protein and part nucleic;"
          + " put a chain break between them");
      }
      continue;
    }
    if (span.kind === "generated") {
      current.designed = true;
      current.length = number(span.min, 0);
      continue;
    }
    if (span.kind === "segment") {
      if (sequenceOf === undefined) {
        problems.push("Protein Hunter folds sequences, so a \u201cFrom structure\u201d span needs"
          + " the structure loaded to read its sequence from");
        continue;
      }
      const sequence = sequenceOf(String(span.chain ?? ""), number(span.first, 0), number(span.last, 0));
      if (sequence === null || sequence.length === 0) {
        problems.push(`chain ${span.chain} ${span.first}-${span.last} is not in the loaded structure`);
        continue;
      }
      current.parts.push(sequence);
      continue;
    }
    if (span.kind === "ligand") {
      const code = String(span.code ?? "").trim().toUpperCase();
      if (code.length > 0) ligands.push(code);
      continue;
    }
  }
  finish();

  const chainIndex = chains.findIndex((chain) => chain.length === 0);
  if (chainIndex < 0) problems.push("nothing is being designed; add a Design span");
  if (lengths.length > 1) {
    problems.push("Protein Hunter designs one chain at a time, not " + lengths.length);
  }
  return {
    chains, chainKinds: kinds, ligandCodes: ligands, chainIndex,
    length: lengths[0] ?? 0, problems: [...new Set(problems)],
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
export function createContigList(rowsContainer, addButton, options = {}) {
  const notify = options.onChange ?? (() => {});
  const methodOf = options.method ?? (() => "rfd4p");
  let spans = (options.initial ?? [emptySpan("generated")]).map((span) => ({ ...span }));

  const draw = () => {
    const limits = METHOD_LIMITS[methodOf()] ?? {};
    rowsContainer.replaceChildren();

    for (const [index, span] of spans.entries()) {
      const row = element("div", `contig-row contig-row-${span.kind}`);

      const picker = element("select", "entity-picker");
      picker.title = SPAN_NOTES[span.kind] ?? "";
      for (const kind of SPAN_KINDS) {
        const option = element("option", undefined, SPAN_LABELS[kind]);
        option.value = kind;
        option.selected = kind === span.kind;
        picker.append(option);
      }
      picker.addEventListener("change", () => {
        // 🔴 A NEW SPAN, NOT THE OLD ONE RELABELLED. The fields of one kind mean
        // nothing in another - a segment's `first` is not a length - and
        // carrying them over leaves a row that looks filled in and is not.
        spans[index] = emptySpan(picker.value);
        changed();
      });
      row.append(picker);

      const fields = element("div", "contig-fields");
      const bind = (input, key, transform = (value) => value) => {
        input.addEventListener("input", () => {
          spans[index] = { ...spans[index], [key]: transform(input.value) };
          // 🔴 THE CONTIG LINE UPDATES ON EVERY KEYSTROKE, but the rows are NOT
          // redrawn - redrawing steals focus mid-number and the caret jumps to
          // the end of the field.
          notify(read());
        });
      };

      if (span.kind === "generated") {
        const from = element("input", "entity-value contig-number");
        from.type = "number";
        from.min = "1";
        from.value = String(span.min ?? 75);
        from.title = "How many residues to design. The shortest, if you give a range.";
        bind(from, "min", (value) => number(value, 1));
        const to = element("input", "entity-value contig-number");
        to.type = "number";
        to.min = "1";
        to.value = String(span.max ?? span.min ?? 75);
        to.title = "The longest. Leave it equal to design exactly one length.";
        bind(to, "max", (value) => number(value, 1));
        fields.append(element("span", "contig-label", "residues"), from,
          element("span", "contig-label", "to"), to);
        if (number(span.max, 0) !== number(span.min, 0) && limits.range !== undefined) {
          row.classList.add("contig-row-unavailable");
          fields.append(element("div", "option-note warn contig-reason", limits.range));
        }
      } else if (span.kind === "segment") {
        const chain = element("input", "entity-value contig-chain");
        chain.value = String(span.chain ?? "A");
        chain.maxLength = 4;
        chain.title = "Which chain of the loaded structure.";
        bind(chain, "chain", (value) => value.trim());
        const first = element("input", "entity-value contig-number");
        first.type = "number";
        first.value = String(span.first ?? 1);
        first.title = "First residue, by the structure's own numbering.";
        bind(first, "first", (value) => number(value, 1));
        const last = element("input", "entity-value contig-number");
        last.type = "number";
        last.value = String(span.last ?? 100);
        last.title = "Last residue, inclusive.";
        bind(last, "last", (value) => number(value, 1));
        fields.append(element("span", "contig-label", "chain"), chain,
          element("span", "contig-label", "residues"), first,
          element("span", "contig-label", "to"), last);
      } else if (span.kind === "sequence") {
        const value = element("textarea", "entity-value contig-sequence");
        value.value = String(span.value ?? "");
        value.rows = 2;
        value.spellcheck = false;
        value.placeholder = "the chain's one-letter sequence";
        value.title = "A chain whose sequence you know. Protein Hunter folds it.";
        bind(value, "value", (text) => text.replace(/\s+/g, "").toUpperCase());
        const polymer = element("select", "entity-picker");
        polymer.title = "What this chain is. A nucleic chain folds one token per base.";
        for (const [key, label] of [["protein", "protein"], ["dna", "DNA"], ["rna", "RNA"]]) {
          const option = element("option", undefined, label);
          option.value = key;
          option.selected = (span.polymer ?? "protein") === key;
          polymer.append(option);
        }
        polymer.addEventListener("change", () => {
          spans[index] = { ...spans[index], polymer: polymer.value };
          changed();
        });
        fields.append(polymer, value);
        if (limits.sequence !== undefined) {
          row.classList.add("contig-row-unavailable");
          fields.append(element("div", "option-note warn contig-reason", limits.sequence));
        }
      } else if (span.kind === "ligand") {
        const code = element("input", "entity-value contig-code");
        code.value = String(span.code ?? "");
        code.maxLength = 5;
        code.placeholder = "SAM";
        code.title = "The ligand's CCD code, as the PDB names it.";
        bind(code, "code", (value) => value.trim().toUpperCase());
        fields.append(element("span", "contig-label", "code"), code);
        if (limits.ligand !== undefined) {
          row.classList.add("contig-row-unavailable");
          fields.append(element("div", "option-note warn contig-reason", limits.ligand));
        }
      } else {
        fields.append(element("span", "contig-label contig-break",
          "everything below is a separate chain"));
      }
      row.append(fields);

      const remove = element("button", "entity-remove", "×");
      remove.type = "button";
      remove.title = "Remove this span";
      // 🔴 THE LAST SPAN CANNOT BE REMOVED. An empty list is a job that cannot
      // run, and a list with no rows also has no way to add the first one back
      // that is obvious from looking at it.
      remove.disabled = spans.length === 1;
      remove.addEventListener("click", () => {
        spans.splice(index, 1);
        changed();
      });
      row.append(remove);

      rowsContainer.append(row);
    }
  };

  const read = () => spans.map((span) => ({ ...span }));
  const changed = () => { draw(); notify(read()); };

  addButton.addEventListener("click", () => {
    // A break is almost always what you want next after a given segment, and a
    // design span after a break. Guessing that saves two clicks and is easy to
    // change.
    const last = spans[spans.length - 1];
    spans.push(emptySpan(last?.kind === "segment" ? "break" : "generated"));
    changed();
  });

  draw();

  return {
    read,
    set(next) {
      spans = (next ?? []).map((span) => ({ ...span }));
      if (spans.length === 0) spans = [emptySpan("generated")];
      changed();
    },
    contig() { return contigOf(read(), methodOf()); },
    repaint: draw,
  };
}

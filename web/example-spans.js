/**
 * example-spans.js
 * ----------------
 * The segments a worked example loads as, once its structure is here - one rule for the design page
 * and for test/design-preset-methods.test.js, which says which examples each method can run.
 *
 * 🔴 TWO KINDS OF EXAMPLE, TWO SOURCES OF TRUTH. An example without `specMode` runs from its file's
 * own annotations (generate()'s `fromAnnotations`), so its segments are read off the FILE
 * (`spansFromAnnotatedAtoms`); its contig is a label in the source entry's numbering and drawing it
 * showed a job that never runs. Any other example's contig is moved into the crop's own numbering
 * (cpu/design/source-numbering.js).
 *
 * 🔴 AND THE FILE'S HOTSPOTS GO ON THE SEGMENT THAT HOLDS THEM. The binder example carries its
 * epitope as a per-atom annotation; left there, the page could draw it but nothing a reader could
 * see or edit said so, and Protein Hunter - which reads hotspots off the segments - ran the binder
 * against no site at all. On the segment, both methods read the same field and the reader edits it.
 */
import { spansFromAnnotatedAtoms, spansFromContig } from "./contig-ui.js";
import { residueRanges } from "./condition-editor.js";
import { translateSpans } from "../cpu/design/source-numbering.js";

/** The residues a structure's own hotspot annotation marks, by chain. */
export function annotatedHotspots(atoms) {
  const column = atoms?.condition_hotspot_1_atom ?? atoms?.annotations?.condition_hotspot_1_atom;
  const out = new Map();
  if (!column) return out;
  for (let i = 0; i < column.length; i += 1) {
    if (!column[i]) continue;
    const chain = String(atoms.chain_id[i]);
    if (!out.has(chain)) out.set(chain, new Set());
    out.get(chain).add(Number(atoms.res_id[i]));
  }
  return out;
}

/**
 * An example's segments.
 *
 * @param {{contig: string, specMode?: boolean, cifUrl?: string|null}} preset
 * @param {object|null} atoms its structure, parsed, or null when it names none
 * @returns {object[]|null} the spans, or null when the contig has a piece the editor has no row for
 */
export function exampleSpans(preset, atoms) {
  const parsed = spansFromContig(preset.contig);
  if (atoms === null || atoms === undefined) return parsed;
  const fromAnnotations = preset.specMode !== true && preset.cifUrl !== null && preset.cifUrl !== undefined;
  let spans = fromAnnotations ? spansFromAnnotatedAtoms(atoms) : null;
  if (spans === null) {
    if (parsed === null) return null;
    spans = translateSpans(parsed, atoms).spans;
  }
  // 🔴 A LIGAND IS NAMED BY ITS CCD CODE. The SAM example's own file calls it `S;M` - which the
  // engine reads and a reader takes for a typo - so a name that is no CCD code takes the contig's.
  const codes = (parsed ?? []).filter((span) => span.kind === "ligand").map((span) => span.code);
  spans = spans.map((span) => (span.kind === "ligand" && !/^[A-Z0-9]{1,5}$/.test(String(span.code))
    && codes.length > 0 ? { ...span, code: codes.shift() } : span));
  const hot = annotatedHotspots(atoms);
  return spans.map((span) => {
    if (span.kind !== "segment") return span;
    const chain = String(span.chain ?? "").trim();
    const inside = [...(hot.get(chain) ?? [])].filter((r) => r >= Number(span.first) && r <= Number(span.last));
    if (inside.length === 0) return span;
    // the residues the file marks, merged with any the contig's own span named
    const own = String(span.hotspots ?? "").split(/[^0-9-]+/).filter(Boolean).flatMap((piece) => {
      const [first, last = first] = piece.split(/(?<=\d)-/).map(Number);
      return Array.from({ length: Math.max(0, last - first + 1) }, (_, k) => first + k);
    });
    return { ...span, hotspots: residueRanges([...own, ...inside]).replaceAll(",", " ") };
  });
}

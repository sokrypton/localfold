/**
 * source-numbering.js
 * -------------------
 * Translate a contig written in a structure's ORIGINAL numbering into the
 * numbering of the cropped copy actually being folded.
 *
 * 🔴 THE TWO NUMBERINGS DISAGREE AND BOTH LOOK LIKE REAL RESIDUE IDS. The
 * shipped examples are crops: `motif_idx_1prw.cif.zst` carries chain A at 0-42,
 * while the preset's contig asks for `A16-35` and `A52-71` - the numbering of
 * the deposited entry those residues came from. Read literally against the
 * file, half of that contig names residues it does not contain and the other
 * half names the WRONG ones, which is worse, because it resolves.
 *
 * 🔴 RFD4-PROTEINA NEVER NOTICES, which is why this went unseen. A preset
 * without `specMode` is driven by its structure's own condition annotations
 * rather than by its contig, so the mismatched string is never read. Protein
 * Hunter has no such path - it reaches a structure by folding a SEQUENCE, so it
 * reads the contig literally and is right to refuse.
 *
 * 🔴 AND THE ANSWER IS IN THE FILE. Each atom carries
 * `annotation_src-component_1_atom`, the chain and residue it came FROM -
 * "A16" for the atom now sitting at A1 - so the crop records its own
 * provenance and the translation is a lookup rather than an alignment.
 *
 * Pure: a parsed atom array in, spans out. `test/design-source-numbering.test.js`
 * holds it.
 */

/** The annotation each atom carries naming where it came from. */
const SOURCE = "annotation_src-component_1_atom";

/**
 * Where every original residue ended up in the cropped structure.
 *
 * @param {object} atoms a `parseCifToAtomArray` result
 * @returns {Map<string, {chain: string, residue: number}>} keyed by the
 *   original label exactly as the file writes it, e.g. "A16". Empty when the
 *   structure carries no provenance, which is every ordinary PDB entry - such a
 *   file IS its own source and needs no translation.
 */
export function sourceResidues(atoms) {
  const out = new Map();
  const labels = atoms?.annotations?.[SOURCE];
  if (labels === undefined) return out;
  const chains = atoms.chain_id ?? [];
  const residues = atoms.res_id ?? [];
  for (let at = 0; at < labels.length; at += 1) {
    const label = labels[at];
    // An expandable segment placeholder has no source: it is the part being
    // generated, and `<M>` or "" is how the file says so.
    if (typeof label !== "string" || label.length === 0 || label === "<M>") continue;
    if (out.has(label)) continue;
    out.set(label, { chain: String(chains[at]), residue: Number(residues[at]) });
  }
  return out;
}

/**
 * One `chain` + `first`..`last` range, in the cropped structure's numbering.
 *
 * 🔴 CONTIGUITY IS CHECKED, NOT ASSUMED. A contig range is one SEGMENT - a
 * stretch the model is told to keep together - so if the residues it names
 * landed in the crop out of order, in two pieces, or on different chains, there
 * is no single range that means the same thing. Returning the ends anyway would
 * quietly include whatever fell between them.
 *
 * @returns {{chain: string, first: number, last: number}|null} null when the
 *   range cannot be expressed as one range in the crop
 */
function translateRange(map, chain, first, last) {
  const found = [];
  for (let residue = first; residue <= last; residue += 1) {
    const hit = map.get(`${chain}${residue}`);
    if (hit === undefined) return null;
    found.push(hit);
  }
  if (found.length === 0) return null;
  const onto = found[0].chain;
  if (found.some((hit) => hit.chain !== onto)) return null;
  for (let at = 1; at < found.length; at += 1) {
    if (found[at].residue !== found[at - 1].residue + 1) return null;
  }
  return { chain: onto, first: found[0].residue, last: found[found.length - 1].residue };
}

/**
 * Rewrite every `segment` span into the loaded structure's own numbering.
 *
 * 🔴 A SPAN THAT CANNOT BE TRANSLATED IS LEFT EXACTLY AS IT IS, so the page's
 * existing refusal - "chain A 52-71 is not in the loaded structure" - still
 * fires and says the true thing. Guessing a range here would turn a visible
 * refusal into a silent wrong answer, which is the trade this whole file exists
 * to avoid.
 *
 * 🔴 AND A STRUCTURE WITH NO PROVENANCE IS A NO-OP. An ordinary entry loaded by
 * the user IS its own source, so there is nothing to translate and the spans
 * come back identical.
 *
 * @param {object[]} spans
 * @param {object} atoms a `parseCifToAtomArray` result
 * @returns {{spans: object[], translated: number}}
 */
export function translateSpans(spans, atoms) {
  const map = sourceResidues(atoms);
  if (map.size === 0) return { spans, translated: 0 };
  let translated = 0;
  const out = spans.map((span) => {
    if (span.kind !== "segment") return span;
    const moved = translateRange(map, String(span.chain ?? ""),
      Number(span.first), Number(span.last));
    if (moved === null) return span;
    // Already in the crop's own numbering: nothing to say and nothing to count.
    if (moved.chain === span.chain && moved.first === span.first
      && moved.last === span.last) return span;
    translated += 1;
    return { ...span, chain: moved.chain, first: moved.first, last: moved.last };
  });
  return { spans: translated === 0 ? spans : out, translated };
}

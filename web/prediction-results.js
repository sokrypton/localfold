import { validatedChainLengths } from "../src/input/chains.js";

const RESIDUE_NAMES = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS",
  I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP",
  Y: "TYR", V: "VAL", X: "UNK",
};

// AlphaFold's atom37 order from residue_constants.py.
const ATOM_NAMES = [
  "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
  "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
  "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];

function field(value, width, decimals) {
  return value.toFixed(decimals).padStart(width);
}

/** The ATOM records for one structure. Serial numbering restarts per model, as in an NMR ensemble. */
/**
 * The chain ids this writer gives a complex, in order.
 *
 * Exported because the heatmap needs the same ones BEFORE there is a structure
 * to read them off: see the trunk's contact map in web/app.js.
 */
export const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

function atomLines(sequence, structure, plddt, chainLengths) {
  if (structure.atom37.length !== sequence.length * 37 * 3 || structure.atom37Mask.length !== sequence.length * 37) {
    throw new RangeError("atom37 output does not match the sequence length");
  }
  if (plddt.length !== sequence.length) throw new RangeError("pLDDT output does not match the sequence length");
  const lengths = validatedChainLengths(sequence.length, chainLengths);
  if (lengths.length > CHAIN_IDS.length) throw new RangeError(`PDB output supports at most ${CHAIN_IDS.length} chains`);
  const lines = [];
  let serial = 1;
  let residue = 0;
  for (let chain = 0; chain < lengths.length; chain += 1) {
    const chainId = CHAIN_IDS[chain];
    for (let within = 0; within < lengths[chain]; within += 1, residue += 1) {
      const residueName = RESIDUE_NAMES[sequence[residue]] ?? "UNK";
      for (let atom = 0; atom < ATOM_NAMES.length; atom += 1) {
        if (structure.atom37Mask[residue * 37 + atom] < 0.5) continue;
        const offset = (residue * 37 + atom) * 3;
        const atomName = ATOM_NAMES[atom];
        const element = atomName[0];
        lines.push(
          // 🔴 THE NAME IS LEFT-JUSTIFIED FROM COLUMN 14, NOT RIGHT-JUSTIFIED
          // INTO 16. The PDB format gives the atom name columns 13-16 and
          // starts a one-character element's name at 14, so alpha carbon is
          // " CA " and only a two-character ELEMENT - iron, " FE " as "FE  " -
          // begins at 13. padStart wrote "  CA", which every lenient parser
          // trims back to the right name and every strict one reads by column:
          // the backbone is then not where N, CA and C are looked for, and a
          // viewer draws a structure with no backbone rather than refusing to
          // open it. src/af3/fold.js has always written it the other way, which
          // is why only AlphaFold 2's files were wrong.
          `ATOM  ${String(serial).padStart(5)}  ${atomName.padEnd(3)} ${residueName} ${chainId}${String(within + 1).padStart(4)}    `
          + `${field(structure.atom37[offset], 8, 3)}${field(structure.atom37[offset + 1], 8, 3)}`
          + `${field(structure.atom37[offset + 2], 8, 3)}  1.00${field(plddt[residue], 6, 2)}`
          + `          ${element.padStart(2)}`,
        );
        serial += 1;
      }
    }
    lines.push("TER");
  }
  return lines;
}

/** Serializes an AlphaFold atom37 result as PDB chains with pLDDT in the B-factor field. */
export function predictionToPdb(
  sequence,
  structure,
  plddt,
  chainLengths = undefined,
) {
  const lines = ["REMARK   1 ALPHAFOLD2 WEBGPU PREDICTION"];
  lines.push(...atomLines(sequence, structure, plddt, chainLengths), "END");
  return `${lines.join("\n")}\n`;
}

/**
 * Every recycle as one multi-model PDB, oldest first.
 *
 * This is what makes the viewer animate. py2Dmol reads MODEL/ENDMDL the way it
 * reads an NMR ensemble - more than one frame and it puts up its own play strip
 * - so the recycling loop becomes something you can scrub through rather than a
 * number in a table. The B-factor column carries each pass's OWN pLDDT, so the
 * colouring moves with the structure and you watch confidence arrive.
 *
 * @param {string} sequence
 * @param {readonly {structure: object, confidence: {plddt: Float32Array}}[]} recycles
 */
export function recyclesToPdb(sequence, recycles, chainLengths = undefined) {
  if (recycles.length === 0) throw new RangeError("a prediction must have at least one recycle");
  const lines = ["REMARK   1 ALPHAFOLD2 WEBGPU PREDICTION",
    `REMARK   2 ${recycles.length} RECYCLE${recycles.length === 1 ? "" : "S"}, MODEL n IS RECYCLE n-1`];
  recycles.forEach((recycle, index) => {
    lines.push(`MODEL     ${String(index + 1).padStart(4)}`);
    lines.push(...atomLines(sequence, recycle.structure, recycle.confidence.plddt, chainLengths), "ENDMDL");
  });
  lines.push("END");
  return `${lines.join("\n")}\n`;
}

/**
 * The trunk's contact map, as the heatmap panel's byte format.
 *
 * 🔴 IT IS A RESHAPE, NOT A COMPUTATION. The distogram head already sums its
 * bins up to 8 A into P(d <= 8 A) for every pair and the result is already
 * read back to the host, so this costs one pass over tokens^2 bytes and no
 * GPU work at all.
 *
 * 🔴 AND IT NEEDS NO COLOURS OR BOUNDS FROM HERE. `contact` is a scale the
 * panel knows - 0 to 1, white to a dark blue - and a map that states its own
 * would override exactly the thing that makes it read correctly: white is
 * zero and the ink is the signal, which is the opposite of PAE's reading.
 * `vmin` and `vmax` are given because the BYTES are encoded against them and
 * a map that does not say so is trusting two tables to agree.
 *
 * 🔴 IT GOES ON FRAME 0, NOT THE LAST ONE. The panel resolves each map by
 * searching BACKWARD from the frame being drawn, and the contact map is a
 * property of the trunk rather than of any sampler step - fixed for the whole
 * fold - so one copy at the start is on screen for every frame. The PAE stays
 * where it is, on the final frame, because it only exists there.
 *
 * 🔴 AND THE SECOND ARGUMENT IS NOT OPTIONAL, because the call that forgot it
 * is the whole reason this moved here. There are four of these - two on the
 * AF3 path, one on AF2's distogram, one on ESMFold2's - and exactly ONE is
 * reached by a fold carrying a modified residue. That one did not collapse,
 * and the symptom was a **13-wide PAE beside a 22-wide contact map**, measured
 * on a real fold of GWSTELEKHRSVQ + SEP@3. A default of `undefined` made the
 * omission silent; asking for the argument makes a new call site state which
 * space it is in, and a path with nothing to collapse says so by passing
 * `undefined`.
 */
export function contactMapFor(contactProbs, keep) {
    if (arguments.length < 2) {
        throw new TypeError("contactMapFor needs the viewer's tokens (or an explicit"
            + " undefined): a token matrix and a residue picture are not the same width");
    }
    const n = Math.round(Math.sqrt(contactProbs.length));
    if (n * n !== contactProbs.length) return undefined;
    // 🔴 IN THE VIEWER'S INDEX SPACE, NOT THE MODEL'S. A modified residue is
    // several TOKENS and one POSITION, so a fold carrying one hands the panel
    // a matrix wider than the structure beside it unless it is collapsed - and
    // only where it is WIDER: a matrix narrower than the positions is not a
    // token space this can read, and it passes through as it always did.
    const rows = keep === undefined || keep.length >= n
        ? undefined : matrixForViewer(contactProbs, keep);
    const width = rows === undefined ? n : rows.length;
    const data = new Uint8Array(width * width);
    for (let index = 0; index < data.length; index += 1) {
        const value = rows === undefined
            ? contactProbs[index]
            : rows[Math.floor(index / width)][index % width];
        data[index] = Math.max(0, Math.min(255, Math.round(value * 255)));
    }
    return { data, n: width, vmin: 0, vmax: 1 };
}

/**
 * WHICH TOKENS THE VIEWER DRAWS, one per position it makes - and the whole
 * reason the two index spaces are not the same one.
 *
 * 🔴 A MODIFIED RESIDUE IS ONE POSITION AND SEVERAL TOKENS. Every family but
 * boltz2 ATOMISES one: a phosphoserine is ten tokens carrying one atom each,
 * and py2Dmol draws it as ONE residue, because `toPdb` writes those atoms
 * under one residue number with a backbone among them and the parser keeps the
 * alpha carbon. Measured on a twelve-residue chain with SEP at position 3:
 * AF3 says 21 tokens, the viewer says 12 positions - so every residue after
 * the modification was reading somebody else's row, and nine rows addressed
 * nothing at all. Reported as the PAE being arranged wrongly on a fold with a
 * modified amino acid.
 *
 * 🔴 AND A LIGAND IS THE OPPOSITE, WHICH IS WHY THIS IS NOT A RESIDUE MAP.
 * A ligand's heavy atoms are one token each AND one position each - the parser
 * has no backbone to collapse them onto - so they pass through untouched. The
 * rule is the parser's own: a span that carries a representative atom is a
 * RESIDUE and collapses to it; anything else is atoms and stays.
 *
 * The representative is the alpha carbon for a protein residue and C1' for a
 * nucleotide, which is the atom py2Dmol keeps; falling back to the span's
 * first token means a span with neither is still one position rather than an
 * off-by-n for everything after it.
 *
 * Absent a batch (AlphaFold 2, ESMFold2 without modifications) there is
 * nothing to collapse and the caller passes none: tokens are residues there.
 */
// 🔴 WHAT MAKES A RESIDUE ONE POSITION IS THE PARSER'S OWN TEST, AND IT IS
// NOT "has a name": py2Dmol keeps a residue whole when it carries a backbone
// - N and CA and C for a protein, C4' and O4' and C1' for a nucleotide - and
// draws it at that backbone's atom (the CA, or the C4'). Anything else is a
// LIGAND to it, and a ligand is one position per heavy atom.
//
// Measured in the viewer, three modifications inside a six-residue chain:
// SEP with a full backbone is ONE position of type P; a bare phosphate
// (P, O1P, O2P, O3P) is FOUR positions of type L; a modified nucleotide
// carrying a ribose is ONE position of type R. So a rule that collapsed
// every modified span would be wrong by three on the second of those - the
// same fault as the one this exists to fix, pointing the other way.
//
// What it assumes is that the modification is ATTACHED: the parser also
// requires the residue to be within bonding distance of its neighbours, and
// a fold that flung it off the chain would be drawn as a ligand while this
// still collapsed it. That is a broken fold rather than a shape this can
// serve, and tools/gpu/probe-modified.js is what measures it.
const BACKBONES = [
    { needs: ["N", "CA", "C"], drawnAt: "CA" },
    { needs: ["C4'", "O4'", "C1'"], drawnAt: "C4'" },
];

/**
 * WHICH TOKENS THE VIEWER DRAWS, one per position it makes - and the whole
 * reason the two index spaces are not the same one.
 *
 * 🔴 A MODIFIED RESIDUE IS ONE POSITION AND SEVERAL TOKENS. Every family but
 * boltz2 ATOMISES one: a phosphoserine is ten tokens carrying one atom each,
 * and py2Dmol draws it as ONE residue. Measured on a twelve-residue chain
 * with SEP at position 3: AF3 says 21 tokens, the viewer says 12 positions -
 * so every residue after the modification was reading somebody else's row,
 * and nine rows addressed nothing at all. Reported as the PAE being arranged
 * wrongly on a fold with a modified amino acid.
 *
 * 🔴 AND A LIGAND IS THE OPPOSITE, WHICH IS WHY THIS IS NOT A RESIDUE MAP.
 * A ligand's heavy atoms are one token each AND one position each - the
 * parser has no backbone to collapse them onto - so they pass through
 * untouched, and so does a modification that carries no backbone either.
 *
 * Absent a batch (AlphaFold 2, ESMFold2 without modifications) there is
 * nothing to collapse and the caller passes none: tokens are residues there.
 */
export function viewerTokens(batch) {
    const tokens = batch?.tokens ?? 0;
    if (!(tokens > 0)) return [];
    const collapsed = new Map();          // token -> the one the viewer draws
    for (const span of batch.modifiedSpans ?? []) {
        if (!(span.count > 1)) continue;  // boltz2 keeps it in one token already
        const names = (span.atoms ?? []).map((atom) => atom.name);
        const backbone = BACKBONES.find(
            (kind) => kind.needs.every((name) => names.indexOf(name) >= 0));
        if (backbone === undefined) continue;   // the viewer draws these as atoms
        const keeps = span.from + names.indexOf(backbone.drawnAt);
        for (let offset = 0; offset < span.count; offset += 1) {
            collapsed.set(span.from + offset, keeps);
        }
    }
    const keep = [];
    for (let token = 0; token < tokens; token += 1) {
        const draws = collapsed.get(token);
        if (draws === undefined || draws === token) keep.push(token);
    }
    return keep;
}

/**
 * WHERE THE MODIFIED RESIDUES ARE, as positions the viewer can address.
 *
 * A modification is invisible in a cartoon: the ribbon runs through its alpha
 * carbon exactly as it runs through the serine it was made from, and the
 * phosphate - the whole reason the residue is in the job - is a side-chain
 * atom that nothing draws until it is asked for. So the page asks, for these
 * residues and no others: `showSidechains` takes a selector, and this is the
 * `positions` for it.
 *
 * A modification the viewer draws as ATOMS rather than as a residue (see
 * BACKBONES) is left out: its atoms are already on screen, and there is no
 * side chain to ask for.
 */
export function modifiedPositions(batch, keep) {
    const at = new Map();
    for (let position = 0; position < keep.length; position += 1) at.set(keep[position], position);
    const positions = [];
    for (const span of batch?.modifiedSpans ?? []) {
        // ...the one token of it the viewer kept, which exists only where the
        // span collapsed; a span drawn as atoms has ALL of its tokens here and
        // is not a side chain to show.
        const kept = [];
        for (let offset = 0; offset < span.count; offset += 1) {
            const position = at.get(span.from + offset);
            if (position !== undefined) kept.push(position);
        }
        if (kept.length === 1) positions.push(kept[0]);
    }
    return positions;
}

/**
 * A token-by-token matrix read in the viewer's own index space.
 *
 * The rows and columns a modified residue's other atoms contributed are
 * DROPPED rather than averaged: the value kept is the one at the atom the
 * viewer draws, which is the residue's own frame - the same reading the
 * matrix has for every unmodified residue beside it. Averaging would mix the
 * phosphate's error into the backbone's and make the residue's row mean
 * something no other row means.
 */
export function matrixForViewer(values, keep) {
    const stride = Math.round(Math.sqrt(values.length));
    if (stride * stride !== values.length) {
        throw new RangeError(`a token matrix has ${values.length} entries, which is not square`);
    }
    if (keep.length === 0 || keep[keep.length - 1] >= stride) {
        throw new RangeError(`a ${stride}-wide matrix cannot serve ${keep.length}`
            + ` positions ending at token ${keep[keep.length - 1]}`);
    }
    const rows = [];
    for (const row of keep) {
        const out = new Array(keep.length);
        for (let at = 0; at < keep.length; at += 1) out[at] = values[row * stride + keep[at]];
        rows.push(out);
    }
    return rows;
}

/** The flat per-pair errors as rows, which is how the format is written. */
export function paeMatrix(values, length) {
  // 🔴 THE STRIDE IS NOT ALWAYS THE LENGTH. AlphaFold 3 scores TOKENS, and a
  // ligand contributes one token per heavy atom - so a fold with a ligand in it
  // returns a matrix wider than the polymer the viewer draws. Reading it at the
  // residue stride walks diagonally through somebody else's rows and produces a
  // PAE that is scrambled rather than obviously wrong, which is why the stride
  // is recovered from the data instead of assumed.
  const stride = Math.round(Math.sqrt(values.length));
  if (stride * stride !== values.length) {
    throw new RangeError(`predicted aligned error has ${values.length} entries, which is not square`);
  }
  if (stride < length) {
    throw new RangeError(`predicted aligned error is ${stride} wide for ${length} residues`);
  }
  // `length` rows and columns of it, from the top-left, in order.
  //
  // 🔴 THE CALLER ASKS FOR THE WHOLE THING NOW. This used to be handed the
  // RESIDUE count on an AlphaFold 3 fold, which cropped a mixed fold's matrix
  // to its polymer block and dropped every ligand row - reported as the PAE
  // missing the ligand part. It was never necessary: py2Dmol carries one
  // position per ligand heavy atom too, in the same order, so the token matrix
  // indexes exactly what is drawn (see the note at `paeSize` in web/app.js).
  // The parameter stays because the AlphaFold 2 path passes a residue count
  // that happens to equal the stride, and because a caller that genuinely
  // wants a block should be able to say so.
  const rows = [];
  for (let row = 0; row < length; row += 1) {
    rows.push(Array.from(values.subarray(row * stride, row * stride + length)));
  }
  return rows;
}

export function confidenceJson(sequence, confidence) {
  const result = {
    sequence,
    plddt: Array.from(confidence.plddt),
    mean_plddt: confidence.meanPlddt,
    ptm: confidence.ptm,
  };
  // 🔴 NaN IS "NOT APPLICABLE" HERE, NOT A MISSING FIELD. AlphaFold 3 reports a
  // monomer's ipTM as NaN - there is no interface to score - and JSON.stringify
  // turns that into `null`, so a bare `!== undefined` writes `"iptm": null` and
  // a ranking_confidence of null beside it. Both are treated as absent.
  if (confidence.iptm !== undefined && !Number.isNaN(Number(confidence.iptm))) {
    result.iptm = confidence.iptm;
    result.ranking_confidence = confidence.multimerScore ?? (0.8 * confidence.iptm + 0.2 * confidence.ptm);
  }
  // 🔴 A NESTED L x L MATRIX, NOT A FLAT ARRAY. This is what AlphaFold and
  // ColabFold write, and what every consumer expects - py2Dmol's PAE panel
  // reads a nested array as angstroms and scales it by 8 into its byte
  // encoding, but reads a FLAT array as bytes that are already scaled. Handed
  // flat angstroms it draws a matrix eight times too small, silently.
  result.predicted_aligned_error = paeMatrix(confidence.predictedAlignedError, sequence.length);
  result.max_predicted_aligned_error = confidence.maxPredictedAlignedError;
  // 🔴 THE CONTACTS ARE NESTED THE SAME WAY THE PAE IS, AND FOR THE SAME
  // REASON. Both are token-by-token matrices whose stride is not the residue
  // count once a ligand is in the fold, so both recover it from the data -
  // which is exactly what paeMatrix does, and why this reuses it rather than
  // flattening a second convention into the file. A consumer that can read one
  // can read the other.
  //
  // Optional because it is optional in the models: AF3 returns it from the
  // trunk and AF2 computes it from the distogram, but a fold that was aborted
  // before the trunk finished has a structure and no contacts.
  if (confidence.contactProbs !== undefined) {
    result.contact_probs = paeMatrix(confidence.contactProbs, sequence.length);
  }
  return JSON.stringify(result, null, 2);
}

export function safeJobName(value) {
  return value.trim().replace(/[^a-zA-Z0-9_.-]+/g, "_").replace(/^[_\.]+|[_\.]+$/g, "").slice(0, 80) || "prediction";
}

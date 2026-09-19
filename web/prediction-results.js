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
const RESIDUE_ANCHORS = ["CA", "C1'"];
export function viewerTokens(batch) {
    const tokens = batch?.tokens ?? 0;
    if (!(tokens > 0)) return [];
    const collapsed = new Map();          // token -> the one the viewer draws
    for (const span of batch.modifiedSpans ?? []) {
        if (!(span.count > 1)) continue;  // boltz2 keeps it in one token already
        const names = (span.atoms ?? []).map((atom) => atom.name);
        let at = -1;
        for (const anchor of RESIDUE_ANCHORS) {
            at = names.indexOf(anchor);
            if (at >= 0) break;
        }
        const keeps = span.from + (at >= 0 ? at : 0);
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
 */
export function modifiedPositions(batch, keep) {
    const at = new Map();
    for (let position = 0; position < keep.length; position += 1) at.set(keep[position], position);
    const positions = [];
    for (const span of batch?.modifiedSpans ?? []) {
        for (let offset = 0; offset < span.count; offset += 1) {
            const position = at.get(span.from + offset);
            if (position !== undefined) { positions.push(position); break; }
        }
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

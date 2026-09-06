/**
 * ESMFold2's input features, built on AF3's featuriser.
 *
 * 🔴 THIS IS AN ADAPTER, NOT A SECOND FEATURISER, AND THAT IS THE WHOLE POINT.
 * ESMFold2 uses AF3's all-atom representation term for term - `ref_pos`,
 * `ref_element`, `ref_charge`, `ref_atom_name_chars`, `ref_space_uid`,
 * `atom_to_token`, `token_bonds` - so src/af3/featurise.js already knows how to
 * tokenise a complex, a nucleic chain, a ligand at one token per heavy atom and
 * a modified residue at one token per atom. Writing that again for this model
 * would be a thousand lines and a second place for a CCD component to be read
 * wrongly. What is genuinely different is three things, and they are all here:
 *
 *   1. THE ATOM LAYOUT IS RAGGED, not AF3's 24 dense slots a token. A residue's
 *      atoms are packed contiguously in conformer order and the array is padded
 *      at the END to a multiple of 32. Reading AF3's slots in increasing order
 *      reproduces conformer order exactly - checked for every conformer in both
 *      tables, protein and nucleic: no entry's slots are non-monotonic.
 *
 *   2. THE ALPHABETS DIFFER, and both are one-hots over residues so neither
 *      complains. AF3's `restype` is 31 wide, one-letter alphabetical with UNK
 *      at 20 and a gap at 21; ESMFold2's is 33 wide, the THREE-letter codes
 *      alphabetically from 2 (ALA 2 ... VAL 21, UNK 22), then RNA at 23-27 and
 *      DNA at 28-32. And the language model's alphabet is a third one again.
 *
 *   3. THERE IS NO TERMINAL ATOM. ESMFold2's `DNA_HEAVY_ATOMS` has no OP3 and
 *      its dumps carry no OXT, so every residue takes the internal form -
 *      which is what `terminalAtoms: false` asks AF3's featuriser for.
 *
 * 🔴 AND ONLY PROTEIN TOKENS REACH THE LANGUAGE MODEL. `protein_mask =
 * (mol_type == 0) & token_mask` upstream, and a non-protein token's hidden
 * state stays ZERO - which is not the same as absent, because the shim's
 * biases make its response to a zero state non-zero. `lmZeroTokens` is the list
 * of tokens that need that constant instead of a tower output.
 */
import { featuriseProtein } from "../af3/featurise.js";

/** ESM-C's alphabet, which is what the language model is tokenised in. */
export const ESM_ALPHABET = ["<cls>", "<pad>", "<eos>", "<unk>",
  "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y",
  "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"];
export const ESM_BOS = 0, ESM_PAD = 1, ESM_EOS = 2, ESM_UNK = 3;
/**
 * 🔴 A NON-PROTEIN TOKEN'S STRUCTURE ID IS 24 AND IT NEVER REACHES THE TOWER.
 * `DNA_RNA_LIGAND_INPUT_ID = 24` is carried in `input_ids` so the feature has a
 * value, and `compute_lm_hidden_states` then masks those tokens out entirely.
 * Feeding them to ESM-C as `X` - which is what 24 spells in this alphabet -
 * runs and is a different model.
 */
export const ESM_NON_PROTEIN = 24;

export const AATYPE_CLASSES = 33;
/** ESMFold2's `PROTEIN_UNK_RES_TYPE`, and where AF3's gap column lands too. */
export const UNKNOWN_AATYPE = 22;
export const MOL_PROTEIN = 0, MOL_DNA = 1, MOL_RNA = 2, MOL_NONPOLYMER = 3;
/** The atom array is padded to a multiple of this, as ESMFold2's dumps are. */
export const ATOM_ALIGNMENT = 32;

/**
 * AF3's 31-class restype into ESMFold2's 33-class one.
 *
 * 🔴 BOTH ARE ONE-HOTS OVER RESIDUES AND NEITHER IS THE OTHER. AF3 is
 * one-letter alphabetical (A 0 ... Y 19, X 20, gap 21, then RNA and DNA);
 * ESMFold2 is three-letter alphabetical offset by two (ALA 2, ARG 3, ASN 4,
 * ASP 5 ...), which permutes every amino acid. Passing either for the other
 * conforms in shape and folds something.
 */
export const AF3_TO_ESMFOLD2_AATYPE = (() => {
  // 🔴 AND THE PROTEIN HALF IS `+ 2`, WHICH IS NOT A COINCIDENCE AND IS NOT
  // SOMETHING TO RELY ON WITHOUT SAYING WHY. AF3's `ARNDCQEGHILKMFPSTWYV` is
  // alphabetical by THREE-letter code - ALA, ARG, ASN, ASP, CYS ... - and so is
  // ESMFold2's, from index 2. So the twenty amino acids are the same order
  // offset by two, and everything after them is not: AF3's gap at 21 has no
  // ESMFold2 slot, its RNA runs 22-25 against 23-26, and its DNA 26-29 against
  // 28-31 because ESMFold2 puts an RNA unknown at 27 where AF3 puts nothing.
  const table = new Int32Array(31).fill(UNKNOWN_AATYPE);
  for (let restype = 0; restype <= 19; restype += 1) table[restype] = restype + 2;
  table[20] = UNKNOWN_AATYPE;         // X
  table[21] = UNKNOWN_AATYPE;         // the alignment gap, which is not a residue
  for (let rna = 0; rna < 4; rna += 1) table[22 + rna] = 23 + rna;   // A G C U
  for (let dna = 0; dna < 4; dna += 1) table[26 + dna] = 28 + dna;   // DA DG DC DT
  return table;
})();

/** ESMFold2's residue type into ESM-C's token id; anything non-protein is 24. */
export const AATYPE_TO_ESM_ID = (() => {
  const letters = { 2: "A", 3: "R", 4: "N", 5: "D", 6: "C", 7: "Q", 8: "E", 9: "G",
    10: "H", 11: "I", 12: "L", 13: "K", 14: "M", 15: "F", 16: "P", 17: "S",
    18: "T", 19: "W", 20: "Y", 21: "V" };
  const index = new Map(ESM_ALPHABET.map((token, at) => [token, at]));
  const table = new Int32Array(AATYPE_CLASSES).fill(ESM_NON_PROTEIN);
  for (const [aatype, letter] of Object.entries(letters)) {
    table[Number(aatype)] = index.get(letter);
  }
  // 🔴 AN UNKNOWN RESIDUE IS `<unk>` AND NOT `X`. ESMFold2's own table has
  // `"X": 3`, which is `<unk>`, while the ESM alphabet's letter `X` is 24 - the
  // same 24 it uses for a ligand. The two are a token apart and both are
  // "unknown"; one of them is the token the tower was trained to see there.
  table[UNKNOWN_AATYPE] = ESM_UNK;
  return table;
})();

/** ESM-C's token ids for a plain protein sequence, with no BOS or EOS. */
export function tokeniseForEsmc(sequence) {
  const index = new Map(ESM_ALPHABET.map((token, at) => [token, at]));
  const ids = new Int32Array(sequence.length);
  for (let i = 0; i < sequence.length; i += 1) {
    ids[i] = index.get(sequence[i]) ?? ESM_UNK;
  }
  return ids;
}

const MOL_TYPE_OF_KIND = { protein: MOL_PROTEIN, dna: MOL_DNA, rna: MOL_RNA };

/**
 * @param {string|object} input a colon-joined sequence, or AF3's own options
 *   object `{ sequence, chainKinds, ligands, modifications }`
 * @param {{profile?: Float32Array, deletionMean?: Float32Array}} [options]
 *   the MSA-derived halves of `s_inputs`. ESMFold2 folds from a single
 *   sequence - `disable_msa_features` is true in this checkpoint - so both are
 *   ZERO unless a caller has an alignment, and the dumps record them as zeros.
 *   Filling the profile with the sequence one-hot instead, which is the
 *   plausible-looking thing, is a different input.
 */
export function featuriseForEsmfold2(input, options = {}) {
  const request = typeof input === "string" ? { sequence: input } : input;
  const batch = featuriseProtein(request.sequence, {
    ...request,
    // 🔴 NO OXT AND NO OP3. See the note at the top; measured, this is what
    // takes ubiquitin's first 40 residues from 312 atoms to the 311 the model
    // was handed.
    terminalAtoms: false,
    // 🔴 AND THE BOND MATRIX IS SYMMETRIC HERE. `compute_token_bonds` writes
    // `bonds[i, j] = bonds[j, i] = 1` for every edge; AF3's featuriser writes
    // one triangle unless asked, because stock AF3 wants one.
    symmetriseBonds: true,
  });
  const tokens = batch.tokens;
  const dense = batch.dense;

  // ---- molecule type per token, which decides the alphabet and the LM mask.
  const molType = new Int32Array(tokens);
  const ligandToken = new Uint8Array(tokens);
  // 🔴 THE SPAN'S FIELDS ARE `from` AND `count`, NOT `start` AND `length`, and
  // the first version read the other pair: `undefined` starts a loop that never
  // runs, so every ligand token came back as PROTEIN with a protein alphabet
  // and a language-model id. Nothing threw and every shape agreed.
  for (const span of batch.ligandSpans ?? []) {
    for (let token = span.from; token < span.from + span.count; token += 1) {
      ligandToken[token] = 1;
    }
  }
  for (let token = 0; token < tokens; token += 1) {
    if (ligandToken[token]) { molType[token] = MOL_NONPOLYMER; continue; }
    const kind = batch.chainKinds[batch.chainOfResidue[batch.residueOfToken[token]]];
    molType[token] = MOL_TYPE_OF_KIND[kind] ?? MOL_PROTEIN;
  }

  // ---- the dense atom slots, read in slot order, packed and padded.
  let live = 0;
  for (let slot = 0; slot < tokens * dense; slot += 1) live += batch.refMask[slot] !== 0 ? 1 : 0;
  const atoms = Math.ceil(live / ATOM_ALIGNMENT) * ATOM_ALIGNMENT;
  const refPos = new Float32Array(atoms * 3);
  const refCharge = new Float32Array(atoms);
  const refElement = new Int32Array(atoms);
  const refAtomNameChars = new Int32Array(atoms * 4);
  const refSpaceUid = new Int32Array(atoms);
  const atomToToken = new Int32Array(atoms);
  const mask = new Float32Array(atoms);
  // 🔴 THE RAGGED-TO-DENSE MAP IS RECORDED HERE OR IT IS RE-DERIVED LATER. AF3's
  // `toPdb` writes a DENSE batch, so a structure this model predicts has to go
  // back through this permutation to be written - and re-deriving it at the far
  // end means a second copy of the packing rule, which is exactly the kind of
  // thing that agrees for a monomer and not for a ligand.
  const denseSlot = new Int32Array(atoms).fill(-1);
  let at = 0;
  for (let token = 0; token < tokens; token += 1) {
    for (let slot = 0; slot < dense; slot += 1) {
      const from = token * dense + slot;
      if (batch.refMask[from] === 0) continue;
      refPos[at * 3] = batch.refPos[from * 3];
      refPos[at * 3 + 1] = batch.refPos[from * 3 + 1];
      refPos[at * 3 + 2] = batch.refPos[from * 3 + 2];
      refCharge[at] = batch.refCharge[from];
      refElement[at] = batch.refElement[from];
      for (let i = 0; i < 4; i += 1) {
        refAtomNameChars[at * 4 + i] = batch.refAtomNameChars[from * 4 + i];
      }
      // 🔴 THE SPACE ID IS A ROTARY FREQUENCY HERE, NOT A LABEL.
      // `ref_space_uid` drives ten of the sixteen rope pairs, so a constant
      // offset on it is a phase shift on every atom rather than a renaming -
      // which is why it is checked against the dump rather than assumed to be
      // free. AF3 counts these from zero, and so does this model.
      refSpaceUid[at] = batch.refSpaceUid[from];
      atomToToken[at] = token;
      denseSlot[at] = from;
      mask[at] = 1;
      at += 1;
    }
  }

  // ---- the alphabets.
  const aatype = new Float32Array(tokens * AATYPE_CLASSES);
  const residueType = new Int32Array(tokens);
  const inputIds = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    const code = molType[token] === MOL_NONPOLYMER
      ? UNKNOWN_AATYPE
      : (AF3_TO_ESMFOLD2_AATYPE[batch.aatype[token]] ?? UNKNOWN_AATYPE);
    residueType[token] = code;
    aatype[token * AATYPE_CLASSES + code] = 1;
    inputIds[token] = molType[token] === MOL_PROTEIN
      ? AATYPE_TO_ESM_ID[code] : ESM_NON_PROTEIN;
  }

  const zeroBase = (values) => {
    const out = new Int32Array(values.length);
    for (let i = 0; i < values.length; i += 1) out[i] = values[i] - 1;
    return out;
  };
  return {
    tokens, atoms, liveAtoms: live, batch, dense, denseSlot,
    refPos, refCharge, refElement, refAtomNameChars, refSpaceUid, atomToToken, mask,
    aatype, residueType, molType, inputIds,
    profile: options.profile ?? new Float32Array(tokens * AATYPE_CLASSES),
    deletionMean: options.deletionMean ?? new Float32Array(tokens),
    // AF3 numbers chains and residues from one; this model's features are
    // zero-based, and `relativePositionBins` only ever takes differences - but
    // `asym_id` is compared for EQUALITY, so an off-by-one that cancels in the
    // differences would still be right. They are converted anyway, because a
    // feature that means the same thing should look the same.
    residueIndex: zeroBase(batch.residueIndex),
    tokenIndex: zeroBase(batch.tokenIndex),
    asymId: zeroBase(batch.asymId),
    entityId: zeroBase(batch.entityId),
    symId: zeroBase(batch.symId),
    tokenBonds: batch.bondMatrix ?? new Float32Array(tokens * tokens),
    tokenMask: new Float32Array(tokens).fill(1),
    sequence: batch.sequence, chains: batch.chains, chainKinds: batch.chainKinds,
  };
}

/**
 * The language model's input, and where its answers go.
 *
 *     [BOS] chain1 [EOS BOS] chain2 ... [EOS]
 *
 * 🔴 EVERY CHAIN GETS ITS OWN BOS AND EOS, AND ONE `sequence_id` PER CHAIN. The
 * tower attends within a chain and not across, which for a single chain is no
 * constraint at all and for a complex is the difference between two chains and
 * one long one.
 *
 * 🔴 AND AN ATOM-TOKENISED RESIDUE COLLAPSES TO ONE LM TOKEN. A modified
 * residue is several structure tokens sharing one `(asym_id, residue_index)`;
 * the language model was trained on per-residue inputs, so upstream keeps the
 * FIRST token of each such key, runs the tower over that, and scatters the
 * answer back to every token of the residue.
 *
 * @returns {{ids: Int32Array, sequenceId: Int32Array, tokenToRow: Int32Array}}
 *   `tokenToRow[token]` indexes the tower's output, or -1 for a token the
 *   tower never saw - which is every non-protein one.
 */
export function languageModelInput(features) {
  const { tokens, molType, asymId, residueIndex, inputIds } = features;
  const rows = [];
  const rowOfKey = new Map();
  const tokenToRow = new Int32Array(tokens).fill(-1);
  for (let token = 0; token < tokens; token += 1) {
    if (molType[token] !== MOL_PROTEIN) continue;
    const key = `${asymId[token]}:${residueIndex[token]}`;
    let row = rowOfKey.get(key);
    if (row === undefined) {
      row = rows.length;
      rowOfKey.set(key, row);
      rows.push({ id: inputIds[token], chain: asymId[token] });
    }
    tokenToRow[token] = row;
  }
  if (rows.length === 0) {
    return { ids: new Int32Array(0), sequenceId: new Int32Array(0), tokenToRow,
             rows: 0, chains: 0 };
  }
  // ...grouped by chain, in the order the chains first appear.
  const chainOrder = [...new Set(rows.map((row) => row.chain))].sort((a, b) => a - b);
  const ids = [];
  const sequenceId = [];
  const positionOfRow = new Int32Array(rows.length);
  chainOrder.forEach((chain, index) => {
    ids.push(ESM_BOS);
    sequenceId.push(index);
    for (let row = 0; row < rows.length; row += 1) {
      if (rows[row].chain !== chain) continue;
      positionOfRow[row] = ids.length;
      ids.push(rows[row].id);
      sequenceId.push(index);
    }
    ids.push(ESM_EOS);
    sequenceId.push(index);
  });
  for (let token = 0; token < tokens; token += 1) {
    if (tokenToRow[token] >= 0) tokenToRow[token] = positionOfRow[tokenToRow[token]];
  }
  return { ids: Int32Array.from(ids), sequenceId: Int32Array.from(sequenceId),
           tokenToRow, rows: rows.length, chains: chainOrder.length };
}

/**
 * Ragged coordinates back into AF3's dense layout, so `toPdb` can write them.
 *
 * 🔴 A STRUCTURE WRITER IS NOT WORTH HAVING TWICE. src/af3/fold.js's `toPdb`
 * already knows that a ligand is HETATM and carries its component's code, that
 * a modified residue takes its own code rather than the letter at its token
 * index, that a nucleotide is " DA" and not "ALA", and that a complex needs one
 * chain letter per asym id - four things that each returned a plausible PDB
 * when they were wrong. Permuting 3n floats is cheaper than any of that.
 */
export function toDensePositions(features, coordinates) {
  const dense = new Float32Array(features.tokens * features.dense * 3);
  for (let atom = 0; atom < features.atoms; atom += 1) {
    const slot = features.denseSlot[atom];
    if (slot < 0) continue;
    dense[slot * 3] = coordinates[atom * 3];
    dense[slot * 3 + 1] = coordinates[atom * 3 + 1];
    dense[slot * 3 + 2] = coordinates[atom * 3 + 2];
  }
  return dense;
}

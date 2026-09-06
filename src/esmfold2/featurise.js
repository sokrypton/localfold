/**
 * A sequence into the features ESMFold2's inputs embedder wants.
 *
 * 🔴 THE ATOM LAYOUT IS RAGGED, NOT AF3's DENSE 24 SLOTS A TOKEN. A residue's
 * heavy atoms are packed contiguously in conformer order and the array is
 * padded at the END to a multiple of 32; `ref_space_uid` is the token index and
 * `atom_to_token` is what says which atoms are whose. AF3's featuriser writes
 * 24 slots a token with holes in them, so its arrays cannot be handed to this
 * model even though every per-atom field has the same name.
 *
 * 🔴 AND THE REFERENCE CONFORMERS ARE AF3's, WHICH IS ADMISSIBLE AND WORTH
 * SAYING OUT LOUD. Both models sample a fresh conformer per residue INSTANCE -
 * bond lengths and angles fixed, torsions drawn - so neither has a canonical
 * one, and src/af3/reference-conformers.js is one sample of the same
 * distribution. Checked against ESMFold2's own draw for ubiquitin's first 40
 * residues: the ATOM SETS, their order, their elements, their charges and their
 * four-character names agree exactly, and the N-CA distance is 1.4738 A against
 * 1.4655 - the same molecule, a different torsion.
 *
 * 🔴 AND THERE IS NO OXT. ESMFold2's C-terminal residue carries the internal
 * atom set, so `conformerFor` is asked for the internal form at every position
 * - including the last, where AF3 would add one.
 *
 * 🔴 AND `aatype` IS NOT ESM-C's ALPHABET, THOUGH BOTH ARE 33 WIDE. The
 * language model is fed the ESM alphabet (`L` is 4, `M` is 20); the embedder's
 * one-hot is the THREE-LETTER code in alphabetical order offset by two (ALA 2,
 * ARG 3, ASN 4 ... VAL 21). Two 33-class one-hots over the same residues that
 * disagree on every index, and passing either for the other conforms in shape.
 */
import { RESIDUE_TYPES, conformerFor } from "../af3/reference-conformers.js";

/** ESM-C / ESM-2's alphabet, which is what the language model is tokenised in. */
export const ESM_ALPHABET = ["<cls>", "<pad>", "<eos>", "<unk>",
  "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y",
  "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"];

/** The embedder's own alphabet: three-letter codes alphabetically, from 2. */
const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};
export const AATYPE_CLASSES = 33;
const AATYPE = (() => {
  const codes = Object.entries(THREE_LETTER).sort((a, b) => (a[1] < b[1] ? -1 : 1));
  const table = {};
  codes.forEach(([letter], index) => { table[letter] = index + 2; });
  return table;
})();
/** Everything unrecognised, which is `UNK`'s slot. */
export const UNKNOWN_AATYPE = 22;

/** The atom array is padded to a multiple of this, as the dumps are. */
export const ATOM_ALIGNMENT = 32;

const esmIndex = (() => {
  const table = new Map();
  ESM_ALPHABET.forEach((token, index) => table.set(token, index));
  return table;
})();

/** ESM-C's token ids for a sequence, with no BOS or EOS: the tower attaches those. */
export function tokeniseForEsmc(sequence) {
  const ids = new Int32Array(sequence.length);
  for (let i = 0; i < sequence.length; i += 1) {
    ids[i] = esmIndex.get(sequence[i]) ?? esmIndex.get("X");
  }
  return ids;
}

/**
 * @param {string} sequence one-letter codes
 * @param {{profile?: Float32Array, deletionMean?: Float32Array}} options
 *   the MSA-derived halves of `s_inputs`. ESMFold2 folds from a single
 *   sequence, so both are ZERO unless a caller has an alignment - which is what
 *   the dumps record, and not the sequence one-hot repeated.
 */
export function featuriseForEsmfold2(sequence, options = {}) {
  const tokens = sequence.length;
  if (tokens === 0) throw new Error("an empty sequence");
  const conformers = [];
  let atomCount = 0;
  for (const letter of sequence) {
    const code = RESIDUE_TYPES.includes(letter) ? letter : "X";
    const conformer = conformerFor(code, false);
    conformers.push(conformer);
    atomCount += conformer.length;
  }
  const atoms = Math.ceil(atomCount / ATOM_ALIGNMENT) * ATOM_ALIGNMENT;

  const refPos = new Float32Array(atoms * 3);
  const refCharge = new Float32Array(atoms);
  const refElement = new Int32Array(atoms);
  const refAtomNameChars = new Int32Array(atoms * 4);
  const refSpaceUid = new Int32Array(atoms);
  const atomToToken = new Int32Array(atoms);
  const mask = new Float32Array(atoms);

  let at = 0;
  for (let token = 0; token < tokens; token += 1) {
    for (const [, name, element, charge, x, y, z] of conformers[token]) {
      refPos[at * 3] = x;
      refPos[at * 3 + 1] = y;
      refPos[at * 3 + 2] = z;
      refCharge[at] = charge;
      refElement[at] = element;
      // 🔴 `chr(code + 32)`, so a name character is its ASCII code less 32 and
      // an absent one is ZERO - which decodes to a space, not to a null.
      for (let i = 0; i < 4; i += 1) {
        refAtomNameChars[at * 4 + i] = i < name.length ? name.charCodeAt(i) - 32 : 0;
      }
      refSpaceUid[at] = token;
      atomToToken[at] = token;
      mask[at] = 1;
      at += 1;
    }
  }

  const aatype = new Float32Array(tokens * AATYPE_CLASSES);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const zeros = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    aatype[token * AATYPE_CLASSES + (AATYPE[sequence[token]] ?? UNKNOWN_AATYPE)] = 1;
    residueIndex[token] = token;
    tokenIndex[token] = token;
  }

  return {
    tokens, atoms, liveAtoms: atomCount,
    refPos, refCharge, refElement, refAtomNameChars, refSpaceUid, atomToToken, mask,
    aatype,
    profile: options.profile ?? new Float32Array(tokens * AATYPE_CLASSES),
    deletionMean: options.deletionMean ?? new Float32Array(tokens),
    residueIndex, tokenIndex,
    asymId: zeros, symId: zeros.slice(), entityId: zeros.slice(),
    tokenBonds: new Float32Array(tokens * tokens),
    tokenMask: new Float32Array(tokens).fill(1),
    inputIds: tokeniseForEsmc(sequence),
  };
}

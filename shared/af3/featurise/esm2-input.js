/**
 * ESM2's input for Chai-1, from a featurised AF3-lineage batch: each protein chain's token ids and where each
 * token's embedding row comes from.
 *
 * chai-1's token features are mostly ESM2 3B's last hidden state (af3-any-model esm.py, model_features.py
 * `_attach_esm`): each PROTEIN chain run alone as [BOS, residues, EOS], BOS and EOS stripped, the rows concatenated
 * in chain order; a token takes its RESIDUE's row (so an atomised modified residue's atom tokens all take their
 * parent's), and a non-protein token takes zeros. The residue letters are the parent residues' (a modified residue
 * reads as its parent, as the job's sequence writes it).
 *
 * @param {object} batch a featurised batch (shared/af3/featurise/featurise.js): aatype, residueOfToken,
 *   chainOfResidue, chainKinds
 * @returns {{ids: Int32Array, chainLengths: Int32Array, tokenRow: Int32Array}} `ids` is every protein chain's
 *   [BOS, residues, EOS] back to back; `tokenRow[t]` the token's row in the concatenated residue rows, or -1
 */
export const ESM2_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
  "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"];
const BOS = 0, EOS = 2, UNK = 3;
const AF3_LETTERS = "ARNDCQEGHILKMFPSTWYV";

export function esm2Inputs(batch) {
  const { aatype, residueOfToken, chainOfResidue, chainKinds } = batch;
  if (!aatype || !residueOfToken || !chainOfResidue || !chainKinds) {
    throw new Error("esm2Inputs needs a batch's aatype, residueOfToken, chainOfResidue and chainKinds");
  }
  const tokens = aatype.length;
  const index = new Map(ESM2_VOCAB.map((token, at) => [token, at]));
  // each residue's letter, off its first token's restype
  const letterOf = new Map();
  for (let t = 0; t < tokens; t += 1) {
    const r = residueOfToken[t];
    if (r < 0 || letterOf.has(r)) continue;
    letterOf.set(r, aatype[t] >= 0 && aatype[t] < 20 ? AF3_LETTERS[aatype[t]] : "X");
  }
  const ids = [], chainLengths = [], rowOfResidue = new Map();
  let row = 0;
  for (let chain = 0; chain < chainKinds.length; chain += 1) {
    if (chainKinds[chain] !== "protein") continue;
    const residues = [];
    for (let r = 0; r < chainOfResidue.length; r += 1) if (chainOfResidue[r] === chain) residues.push(r);
    if (residues.length === 0) continue;
    ids.push(BOS);
    for (const r of residues) { ids.push(index.get(letterOf.get(r) ?? "X") ?? UNK); rowOfResidue.set(r, row++); }
    ids.push(EOS);
    chainLengths.push(residues.length);
  }
  const tokenRow = new Int32Array(tokens).fill(-1);
  for (let t = 0; t < tokens; t += 1) {
    const r = residueOfToken[t];
    if (r >= 0 && rowOfResidue.has(r)) tokenRow[t] = rowOfResidue.get(r);
  }
  return { ids: Int32Array.from(ids), chainLengths: Int32Array.from(chainLengths), tokenRow };
}

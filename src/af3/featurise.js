/**
 * AF3's batch for a single protein chain, built from a sequence in JavaScript.
 *
 *     const batch = featuriseProtein("GWSTELEK...");
 *
 * This is the piece that turns "the model runs on the GPU against a batch
 * prepared elsewhere" into "the browser folds a sequence you type". AF3's own
 * featuriser is a 515 MB chemical component dictionary and a tokeniser, and
 * neither is going into a web page; what a protein chain actually reaches of
 * that dictionary is 21 components, baked into reference-conformers.js.
 *
 * Every array here is checked element-for-element against AF3's own batch for
 * 6MRR by tools/oracle/check_af3_featurise.js. Two are checked differently and
 * the difference is the point:
 *
 * 🔴 ref_pos CANNOT MATCH AND DOES NOT NEED TO. AF3 samples a fresh conformer
 * for every residue INSTANCE - fixed bond lengths and angles, random torsions -
 * so the 13 internal glutamates in a 6MRR batch have 13 different side chains.
 * A baked table gives them all the same one. Measured end to end, that moves
 * the trunk's pair representation by relRMS 2.7e-2 and the folded structure by
 * 0.01 A RMSD. The checker holds the bonded geometry to AF3 and lets the
 * torsions go.
 *
 * 🔴 THE MSA IS THE QUERY ALONE. A de novo design has no homologues, and the
 * dump this was built against ran num_msa=1. `msa`, `profile` and
 * `deletionMatrix` are all one row of the query, which is what AF3 produces for
 * a single-sequence input - not a stub. A real MSA changes only these three
 * arrays; nothing else here depends on depth.
 *
 * 🔴 A COLON SEPARATES CHAINS, as it does everywhere else on the page. The
 * chain identity comes from src/input/chains.js - the same chainIdentity() and
 * residueIndexPerChain() AlphaFold-multimer uses - because AF3 wants exactly
 * what AF2-multimer wants and writing a second copy of it is how the two drift.
 * The only difference is the base: AF2 counts asym, entity and sym from zero
 * and AF3 from one. The relative encoding reads DIFFERENCES and EQUALITY, so
 * the offset changes nothing the model sees; it is applied so this matches
 * AF3's own batch element for element, which is what makes the checker strict.
 *
 * WHAT IS NOT HERE, deliberately: ligands, nucleic acids, covalent bonds
 * between chains, and templates. Each is a token type this tokeniser does not
 * create, and each would need its own CCD entries.
 */
import { conformerFor, aatypeFor } from "./reference-conformers.js";
import { chainIdentity, residueIndexPerChain } from "../input/chains.js";

const DENSE = 24;
const QUERIES = 32;
const KEYS = 128;
/** AF3's restype alphabet is 31 wide: 20 amino acids, UNK, and the nucleic acids. */
const RESTYPES = 31;
/** Every ligand token carries UNK, whatever the atom is. */
const UNK_AATYPE = 20;
/** ...and a gap in the MSA, which sits between the amino acids and the nucleotides. */
const MSA_GAP = 21;

/**
 * A gather in AF3's form: indices into a flattened source, and a mask marking
 * which of them are real.
 */
function gather(count) {
  return { indices: new Int32Array(count), mask: new Float32Array(count), count };
}

/**
 * @param {string} sequence one-letter codes; anything unrecognised becomes UNK
 * @param {{msa?: number[][], deletionMatrix?: number[][], unpairedFrom?: number}}
 *   [options] extra MSA rows, each already tokenised to AF3 aatypes and the
 *   same length as the
 *   sequence. The query is always row zero and is prepended here.
 */
export function featuriseProtein(sequence, options = {}) {
  const chains = sequence.split(":").filter((chain) => chain.length > 0);
  const joined = chains.join("");
  const residueTokens = joined.length;
  // 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, not one token. Sixty-eight
  // residues plus glycerol is seventy-four tokens, plus heme is a hundred and
  // eleven - checked against AF3's own batch. Every array below is sized from
  // this total, and a ligand counted as a single token produces a batch whose
  // shapes all agree with each other and with nothing else.
  const ligands = options.ligands ?? [];
  const ligandTokens = ligands.reduce((sum, ligand) => sum + ligand.atoms.length, 0);
  const tokens = residueTokens + ligandTokens;
  if (tokens === 0) throw new Error("featuriseProtein: empty sequence");
  const subsets = Math.ceil((tokens * DENSE) / QUERIES);
  const chainLengths = chains.map((chain) => chain.length);
  // 🔴 NOT COMPUTED WHEN THERE ARE NO RESIDUES. A ligand on its own is a valid
  // fold - AF3 accepts one - and both of these reject a zero-length sequence,
  // rightly, because a zero-length CHAIN is a bug. There is no chain here to be
  // wrong about: they are read only inside the polymer loop below, which does
  // not run.
  const identity = residueTokens === 0
    ? null : chainIdentity(residueTokens, chainLengths, chains);
  const withinChain = residueTokens === 0
    ? null : residueIndexPerChain(residueTokens, chainLengths);
  // 🔴 THE LAST RESIDUE OF EVERY CHAIN TAKES AN OXT, not the last token of the
  // batch. Checked against AF3's own complex: a three-chain A/A/B dump carries
  // it on tokens 20, 41 and 62. Getting this wrong is one missing oxygen and
  // one spurious one per extra chain, both of which land in a token's pooled
  // atom representation and in the atom-pair window around it.
  const lastOfChain = new Set();
  let edge = -1;
  for (const length of chainLengths) { edge += length; lastOfChain.add(edge); }

  const aatype = new Int32Array(tokens);
  const refPos = new Float32Array(tokens * DENSE * 3);
  const refMask = new Float32Array(tokens * DENSE);
  const refElement = new Int32Array(tokens * DENSE);
  const refCharge = new Float32Array(tokens * DENSE);
  const refAtomNameChars = new Int32Array(tokens * DENSE * 4);
  const refSpaceUid = new Int32Array(tokens * DENSE);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  const seqMask = new Float32Array(tokens);

  // The flat index of every real atom, in token-then-slot order. AF3's query
  // layout is this list compacted, so it is built once and read four times.
  const realAtoms = [];
  const pseudoBetaSlot = new Int32Array(tokens).fill(-1);

  for (let token = 0; token < residueTokens; token += 1) {
    const code = joined[token];
    aatype[token] = aatypeFor(code);
    // AF3 counts these from one; chains.js counts from zero. See the note above.
    residueIndex[token] = withinChain[token] + 1;
    tokenIndex[token] = token + 1;
    asymId[token] = identity.asymId[token] + 1;
    entityId[token] = identity.entityId[token] + 1;
    symId[token] = identity.symId[token] + 1;
    seqMask[token] = 1;

    const atoms = conformerFor(code, lastOfChain.has(token));
    for (const [slot, name, element, charge, x, y, z] of atoms) {
      const flat = token * DENSE + slot;
      refMask[flat] = 1;
      refElement[flat] = element;
      refCharge[flat] = charge;
      refPos[flat * 3] = x;
      refPos[flat * 3 + 1] = y;
      refPos[flat * 3 + 2] = z;
      // AF3 stores an atom name as four characters offset by 32, zero-padded.
      for (let character = 0; character < 4; character += 1) {
        refAtomNameChars[flat * 4 + character] =
          character < name.length ? name.charCodeAt(character) - 32 : 0;
      }
      realAtoms.push(flat);
    }
    // 🔴 CB, FALLING BACK TO CA. Glycine has no CB, and taking its slot anyway
    // would read a masked slot sitting at the origin - the pseudo-beta feeds
    // the confidence head's distance bins, so every glycine would come out
    // tens of angstroms from everything.
    const beta = atoms.find((atom) => atom[1] === "CB")
      ?? atoms.find((atom) => atom[1] === "CA");
    if (beta) pseudoBetaSlot[token] = beta[0];
    // 🔴 ONE REFERENCE SPACE PER TOKEN. Every atom of a residue shares its uid,
    // and the atom encoder uses the uid only to decide whether a PAIR of atoms
    // may compare reference coordinates at all - which is what makes the random
    // per-residue orientation harmless.
    for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = token;
  }

  // 🔴 THE LIGANDS, AFTER EVERY POLYMER CHAIN. Their tokens continue the token
  // index, take their own asym_id, and share ONE ref_space_uid across the whole
  // component - the six atoms of a glycerol are one rigid conformer, not six
  // independent ones, and giving each its own uid tells the atom encoder they
  // may not be compared, which is the opposite of true.
  let asym = chainLengths.length;
  const entityOfCode = new Map();
  const copiesOfEntity = new Map();
  let ligandToken = residueTokens;
  // Where each ligand's tokens sit, so a writer can name them: the batch's
  // `sequence` covers the polymers only, and a ligand token indexed into it
  // comes back undefined and is written as UNK.
  const ligandSpans = [];
  for (const ligand of ligands) {
    asym += 1;
    ligandSpans.push({ from: ligandToken, count: ligand.atoms.length, code: ligand.code });
    // Identical codes are one entity, and each occurrence is a copy of it -
    // the same rule chainIdentity applies to repeated sequences.
    if (!entityOfCode.has(ligand.code)) entityOfCode.set(ligand.code, entityOfCode.size);
    const entity = entityOfCode.get(ligand.code);
    const copy = (copiesOfEntity.get(entity) ?? 0) + 1;
    copiesOfEntity.set(entity, copy);
    const uid = ligandToken;
    for (let atom = 0; atom < ligand.atoms.length; atom += 1) {
      const token = ligandToken + atom;
      const source = ligand.atoms[atom];
      aatype[token] = UNK_AATYPE;
      // Every atom of the component is the same residue, so they share its
      // number - AF3 writes 1 for a single-residue ligand.
      residueIndex[token] = 1;
      tokenIndex[token] = token + 1;
      // `asym` is already one past the last polymer chain, and AF3 counts from
      // one, so the two cancel: no further +1 here.
      asymId[token] = asym;
      entityId[token] = chains.length + entity + 1;
      symId[token] = copy;
      seqMask[token] = 1;

      const flat = token * DENSE;          // one atom, and it sits in slot zero
      refMask[flat] = 1;
      refElement[flat] = source.element;
      refCharge[flat] = source.charge;
      refPos[flat * 3] = source.x;
      refPos[flat * 3 + 1] = source.y;
      refPos[flat * 3 + 2] = source.z;
      for (let character = 0; character < 4; character += 1) {
        refAtomNameChars[flat * 4 + character] =
          character < source.name.length ? source.name.charCodeAt(character) - 32 : 0;
      }
      realAtoms.push(flat);
      // ...and it is its own centre, where a residue's is CB.
      pseudoBetaSlot[token] = 0;
      for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = uid;
    }
    ligandToken += ligand.atoms.length;
  }

  // 🔴 ONE DIRECTION PER BOND, AND [0,0] CLEARED. AF3 sets contact[i][j] from
  // the CCD's bond table and does NOT set [j][i]; only the OpenFold3 dialect
  // symmetrises. It then clears [0,0] explicitly, because its padded gather
  // rows are zeros and would otherwise mark token 0 as bonded to itself.
  // Neither is cosmetic: this matrix goes through a learned linear straight
  // into the pair representation.
  let bondMatrix;
  if (ligands.some((ligand) => ligand.bonds.length > 0)) {
    bondMatrix = new Float32Array(tokens * tokens);
    let base = residueTokens;
    for (const ligand of ligands) {
      for (const bond of ligand.bonds) {
        bondMatrix[(base + bond.from) * tokens + (base + bond.to)] = 1;
        if (options.symmetriseBonds) bondMatrix[(base + bond.to) * tokens + (base + bond.from)] = 1;
      }
      base += ligand.atoms.length;
    }
    bondMatrix[0] = 0;
  }

  const atomCount = realAtoms.length;

  // token_atoms_to_queries: query slot -> flat token-atom, the compacted list.
  const tokenAtomsToQueries = gather(subsets * QUERIES);
  for (let query = 0; query < atomCount; query += 1) {
    tokenAtomsToQueries.indices[query] = realAtoms[query];
    tokenAtomsToQueries.mask[query] = 1;
  }

  // queries_to_token_atoms: its inverse, laid out over the dense (token, slot)
  // grid, masked exactly where an atom is real.
  const queriesToTokenAtoms = gather(tokens * DENSE);
  for (let query = 0; query < atomCount; query += 1) {
    queriesToTokenAtoms.indices[realAtoms[query]] = query;
    queriesToTokenAtoms.mask[realAtoms[query]] = 1;
  }

  // queries_to_keys: a contiguous 128-wide window per subset of 32 queries,
  // centred on it and SHIFTED IN-BOUNDS at the ends rather than truncated -
  // every subset sees exactly 128 keys.
  //
  // 🔴 THE WINDOW IS CLAMPED AGAINST THE REAL ATOM COUNT, NOT subsets * 32.
  // The query layout is padded out to the dense grid (51 subsets for 574
  // atoms here), so clamping against the padded length would slide the last
  // windows off the end of the molecule and into masked slots.
  const queriesToKeys = gather(subsets * KEYS);
  const tokensToQueries = gather(subsets * QUERIES);
  const tokensToKeys = gather(subsets * KEYS);
  const tokenOfQuery = new Int32Array(atomCount);
  for (let query = 0; query < atomCount; query += 1) {
    tokenOfQuery[query] = (realAtoms[query] / DENSE) | 0;
  }
  const lastStart = Math.max(0, atomCount - KEYS);
  for (let subset = 0; subset < subsets; subset += 1) {
    const start = Math.min(Math.max(subset * QUERIES - (KEYS - QUERIES) / 2, 0), lastStart);
    for (let key = 0; key < KEYS; key += 1) {
      const query = start + key;
      const at = subset * KEYS + key;
      if (query >= atomCount) continue;
      queriesToKeys.indices[at] = query;
      queriesToKeys.mask[at] = 1;
      tokensToKeys.indices[at] = tokenOfQuery[query];
      tokensToKeys.mask[at] = 1;
    }
    for (let slot = 0; slot < QUERIES; slot += 1) {
      const query = subset * QUERIES + slot;
      if (query >= atomCount) continue;
      const at = subset * QUERIES + slot;
      tokensToQueries.indices[at] = tokenOfQuery[query];
      tokensToQueries.mask[at] = 1;
    }
  }

  const tokenAtomsToPseudoBeta = gather(tokens);
  for (let token = 0; token < tokens; token += 1) {
    tokenAtomsToPseudoBeta.indices[token] = token * DENSE + pseudoBetaSlot[token];
    tokenAtomsToPseudoBeta.mask[token] = pseudoBetaSlot[token] >= 0 ? 1 : 0;
  }

  // The MSA. Row zero is the query; anything the caller supplies follows.
  const extra = options.msa ?? [];
  const sequences = 1 + extra.length;
  const msa = new Int32Array(sequences * tokens);
  const msaMask = new Float32Array(sequences * tokens).fill(1);
  const deletionMatrix = new Float32Array(sequences * tokens);
  // 🔴 A LIGAND TOKEN IS A GAP IN THE MSA, NOT AN UNKNOWN RESIDUE. Its aatype
  // is UNK (20) but AF3 writes MSA_GAP (21) in the alignment and in the
  // profile - an alignment has nothing to say about an atom. Copying aatype
  // across, which is right for every polymer token, puts a 20 there instead and
  // tells the model the ligand is a row of unknown amino acids.
  const queryRow = Int32Array.from(aatype);
  for (let token = residueTokens; token < tokens; token += 1) queryRow[token] = MSA_GAP;
  msa.set(queryRow, 0);
  for (let row = 0; row < extra.length; row += 1) {
    msa.set(extra[row], (row + 1) * tokens);
    if (options.deletionMatrix?.[row]) {
      deletionMatrix.set(options.deletionMatrix[row], (row + 1) * tokens);
    }
  }

  // profile: the column-wise restype frequency, and deletionMean the mean
  // deletion count.
  //
  // 🔴 THESE ARE NOT AVERAGES OVER THE WHOLE ARRAY ABOVE. AF3 computes them in
  // its data pipeline, per chain, over the UNPAIRED alignment alone - before
  // the paired rows are prepended and before the query row joins them. So a
  // 32-row A3M gives a 33-row `msa` and a profile over 32, and averaging the 33
  // instead double-counts the query in every column. `unpairedFrom` names where
  // that block starts; 0 means there is no alignment and the profile is the
  // query's own one-hot, which is what AF3 produces for a single sequence.
  //
  // 🔴 AND THEY ARE COMPUTED BEFORE DEDUPLICATION, so their rows are not always
  // the rows of `msa`. AF3 calls get_profile_features on each chain and only
  // then runs deduplicate_unpaired_sequences, which drops every unpaired row
  // the paired block already had - rows that still counted towards the profile.
  // `profileMsa` carries that original block when the caller has one; without
  // it the slice of `msa` is the same thing, which is every case with no
  // pairing.
  const unpairedFrom = options.unpairedFrom ?? (extra.length === 0 ? 0 : 1);
  // An empty block is no block: the profile then falls back to the query's own
  // one-hot through the slice below, which is what AF3 gives a single sequence.
  const profileRows = (options.profileMsa?.length ?? 0) > 0 ? options.profileMsa : null;
  const profile = new Float32Array(tokens * RESTYPES);
  const deletionMean = new Float32Array(tokens);
  const profileDepth = profileRows === null
    ? Math.max(1, sequences - unpairedFrom)
    : Math.max(1, profileRows.length);
  const codeAt = profileRows === null
    ? (row, token) => msa[(unpairedFrom + row) * tokens + token]
    : (row, token) => (profileRows[row] === undefined ? -1 : profileRows[row][token]);
  const deletionAt = profileRows === null
    ? (row, token) => deletionMatrix[(unpairedFrom + row) * tokens + token]
    : (row, token) => (options.profileDeletionMatrix?.[row]?.[token] ?? 0);
  for (let row = 0; row < profileDepth; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      const code = codeAt(row, token);
      if (code >= 0 && code < RESTYPES) profile[token * RESTYPES + code] += 1 / profileDepth;
      deletionMean[token] += deletionAt(row, token) / profileDepth;
    }
  }

  return {
    sequence: joined, chains, chainLengths,
    tokens, dense: DENSE, subsets, atomCount, sequences,
    shape: { tokens, dense: DENSE, subsets, queries: QUERIES, keys: KEYS },
    aatype, profile, deletionMean,
    msa, msaMask, deletionMatrix,
    residueIndex, tokenIndex, asymId, entityId, symId, seqMask,
    refPos, refMask, refElement, refCharge, refAtomNameChars, refSpaceUid,
    // AF3 keeps these separate and they are equal for a protein-only chain:
    // every atom the model predicts is one it has a reference conformer for.
    predDenseAtomMask: refMask,
    bondMatrix, ligandSpans,
    tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
    tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
  };
}

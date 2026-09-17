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
import { polymerResidue, ELEMENT_SYMBOLS } from "./ccd-component.js";
import { nucleicAatypeFor, nucleicConformerFor }
  from "./reference-conformers-nucleic.js";
import { chainIdentity, residueIndexPerChain } from "../../input/chains.js";

const DENSE = 24;
const QUERIES = 32;
const KEYS = 128;
/** AF3's restype alphabet is 31 wide: 20 amino acids, UNK, and the nucleic acids. */
const RESTYPES = 31;
/** Every ligand token carries UNK, whatever the atom is. */
const UNK_AATYPE = 20;

/**
 * The four characters of an ATOMISED atom's name, as AF3 stores them - the code
 * minus 32, zero-padded.
 *
 * 🔴 rosettafold3 RENAMES EVERY ATOMISED ATOM TO ITS ELEMENT SYMBOL, which is
 * `atomized_element_names` in its convention set: a phosphoserine's "CA", "CB",
 * "OG", "O1P" become "C", "C", "O", "O", and a glycerol's "C1" and "O1" become
 * "C" and "O". Every other family keeps the component's own names. Measured
 * against the reference's own batch for 6MRR + GOL + SEP@3: 15 characters over
 * 12 atoms, and rf3 alone.
 *
 * A residue's own atoms are NOT touched by this - only atoms that have been
 * atomised into one-atom tokens, which is what "atomized" names.
 */
function writeAtomName(target, flat, name, element, elementNames) {
  const text = elementNames === true
    ? (ELEMENT_SYMBOLS[element - 1] ?? "C") : name;
  for (let character = 0; character < 4; character += 1) {
    target[flat * 4 + character] =
      character < text.length ? text.charCodeAt(character) - 32 : 0;
  }
}
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
 * The six atom gathers, from a token layout and its compacted atom list.
 *
 * 🔴 EXTRACTED SO THE STRUCTURAL TOKENISER CAN REUSE IT RATHER THAN COPY IT.
 * OpenDDE folds a REGROUPING of these same atoms (see structural-tokens.js), so
 * it needs exactly this construction over a different (token, slot) layout -
 * and a second copy of a windowing rule this delicate is how the two would
 * drift. Everything here is a function of its arguments; nothing reads the
 * enclosing featuriser.
 *
 * @param {{tokens: number, dense: number, realAtoms: ArrayLike<number>,
 *          pseudoBetaSlot: ArrayLike<number>}} layout
 */
export function atomGathers({ tokens, dense, realAtoms, pseudoBetaSlot,
                              paddedKeys = false, qblockKeys = false }) {
  const atomCount = realAtoms.length;
  // 🔴 SUBSETS COUNT THE REAL ATOMS, NOT THE PADDED GRID. The query layout is
  // the compacted list of real atoms, so a subset past `atomCount / 32` holds
  // 32 masked queries, gathers nothing, scatters nowhere and computes a full
  // 32 x `keys` attention over padding. Sizing it from `tokens * dense` made
  // that the common case: 51 subsets for 574 atoms on a 68-residue chain where
  // 18 carry an atom, and OpenDDE's structural layout - the same atoms in
  // twice the tokens - took 98. The atom encoder and decoder run once per
  // diffusion step and every buffer they hold is `subsets * queries * keys`
  // wide, so the padding was the majority of both.
  const subsets = Math.max(1, Math.ceil(atomCount / QUERIES));
  // token_atoms_to_queries: query slot -> flat token-atom, the compacted list.
  const tokenAtomsToQueries = gather(subsets * QUERIES);
  for (let query = 0; query < atomCount; query += 1) {
    tokenAtomsToQueries.indices[query] = realAtoms[query];
    tokenAtomsToQueries.mask[query] = 1;
  }

  // queries_to_token_atoms: its inverse, laid out over the dense (token, slot)
  // grid, masked exactly where an atom is real.
  const queriesToTokenAtoms = gather(tokens * dense);
  for (let query = 0; query < atomCount; query += 1) {
    queriesToTokenAtoms.indices[realAtoms[query]] = query;
    queriesToTokenAtoms.mask[realAtoms[query]] = 1;
  }

  // queries_to_keys: a contiguous window per subset of 32 queries, centred on
  // it and SHIFTED IN-BOUNDS at the ends rather than truncated - every subset
  // sees exactly `keys` of them.
  //
  // 🔴 THE WINDOW IS CLAMPED AGAINST THE REAL ATOM COUNT, NOT subsets * 32.
  // The query layout is padded out to the dense grid (51 subsets for 574
  // atoms here), so clamping against the padded length would slide the last
  // windows off the end of the molecule and into masked slots.
  //
  // 🔴 AND THE WINDOW NEVER EXCEEDS THE MOLECULE, so there is no such thing as
  // a padded KEY. AlphaFold 3 pads because JAX wants static shapes; nothing
  // here does - the kernels are generated per shape and size their workgroup
  // storage from `keys`. Above 128 atoms the clamp already guaranteed this and
  // measured it: `tools/gpu/probe-ablate.js` reports zero padded keys at 143
  // atoms and above. BELOW 128 the window could not fit, and 288 of a
  // 4-residue chain's 384 key slots were padding, each gathering reference
  // space ZERO - which collides with the first conformer's own uid and makes
  // every one of them a valid neighbour of residue 0. That is the released
  // AF3 bug OpenFold3 trained around, and it cannot occur here at any size
  // now, so `maskPaddedKeys` has nothing left to switch off.
  const keys = Math.min(KEYS, atomCount);
  const queriesToKeys = gather(subsets * keys);
  const tokensToQueries = gather(subsets * QUERIES);
  const tokensToKeys = gather(subsets * keys);
  const tokenOfQuery = new Int32Array(atomCount);
  for (let query = 0; query < atomCount; query += 1) {
    tokenOfQuery[query] = (realAtoms[query] / dense) | 0;
  }
  // 🔴 AND THREE FAMILIES DO NOT SLIDE IT AT ALL: THEY CLAMP AND MASK. rf3,
  // opendde and protenix centre the window on `subset * 32 + 16` and take a
  // FIXED offset range around it - `clamp(index, 0, L - 1)` on the gather so
  // the read is in bounds, and `-1e9 * (maskQ | maskK)` on the attention so the
  // out-of-range slots contribute NOTHING. The end subsets therefore see FEWER
  // real keys, where sliding gives them a full window of real ones.
  //
  // It is an END EFFECT and it is not small: af3-any-model measured its own
  // version of this fix at 6MRR 0.767 -> 0.737 on opendde, and here the atom
  // encoder's skip connection reads relRMS 1.17e+0 against the oracle with our
  // rms 43.5 where native's is 30.5 - too BIG, which is what a window that sees
  // 128 real keys instead of 80 does. Upstream missed it on rf3 for a reason
  // worth keeping: rf3 was already in KEY_MASKED_ATOM_ATTENTION, and that list
  // is about the MASK, not about where the window SITS.
  // 🔴 AND IntelliFold-2 SLIDES AGAINST A PADDED EDGE, which is a THIRD rule.
  // It reshapes the flat atom axis into windows and pads to a whole query block
  // first, so its last window starts at `ceil(atoms / 32) * 32 - keys`: 448 on
  // 6MRR where the plain slide gives 446. Its last two subsets therefore take
  // keys 448..575 and reach past the last real atom, which the mask handles.
  const edge = qblockKeys ? Math.ceil(atomCount / QUERIES) * QUERIES : atomCount;
  const lastStart = Math.max(0, edge - keys);
  for (let subset = 0; subset < subsets; subset += 1) {
    const start = paddedKeys
      ? subset * QUERIES + (QUERIES >> 1) - (keys >> 1)
      : Math.min(Math.max(subset * QUERIES - (keys - QUERIES) / 2, 0), lastStart);
    for (let key = 0; key < keys; key += 1) {
      const query = start + key;
      const at = subset * keys + key;
      // ...clamped so the GATHER is in bounds and left MASKED so it counts for
      // nothing. `continue` is the same thing for a slot whose index and mask
      // both start at zero, and index 0 is a real atom - so the clamp is what
      // the reference writes and the mask is what makes either safe.
      if (query < 0 || query >= atomCount) continue;
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
    tokenAtomsToPseudoBeta.indices[token] = token * dense + pseudoBetaSlot[token];
    tokenAtomsToPseudoBeta.mask[token] = pseudoBetaSlot[token] >= 0 ? 1 : 0;
  }

  return { subsets, keys, atomCount, tokenAtomsToQueries, queriesToTokenAtoms,
           queriesToKeys, tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta };
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
  // How many RESIDUES the polymers hold. It used to be the token count too,
  // and stopped being when a modified residue became several tokens.
  const residueCount = joined.length;
  // 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, not one token. Sixty-eight
  // residues plus glycerol is seventy-four tokens, plus heme is a hundred and
  // eleven - checked against AF3's own batch. Every array below is sized from
  // this total, and a ligand counted as a single token produces a batch whose
  // shapes all agree with each other and with nothing else.
  const ligands = options.ligands ?? [];
  const ligandTokens = ligands.reduce((sum, ligand) => sum + ligand.atoms.length, 0);

  // 🔴 A MODIFIED RESIDUE IS ONE TOKEN PER HEAVY ATOM, LIKE A LIGAND, BUT IT
  // LIVES IN THE CHAIN. Measured against AF3: a ten-residue chain with a
  // phosphoserine at position 3 is NINETEEN tokens, the ten belonging to SEP
  // each carrying one atom, all of them holding the PARENT residue's aatype
  // (serine), all sharing that residue's index, and all keeping the chain's
  // asym, entity and sym. Only MSE is different - AF3 folds selenomethionine
  // into methionine's alphabet slot and leaves it as one token - which is why
  // `component` being null is the standard path and not an exception.
  //
  // So a chain's tokens are no longer one per letter, and everything below that
  // used to index the sequence by token now indexes the RESIDUE list instead.
  // 🔴 A CHAIN'S KIND DECIDES WHAT ITS LETTERS MEAN. `A` is alanine in a protein
  // chain and adenine in a nucleic one, and the two take different aatypes,
  // different conformers and a different terminal rule - so a kind is carried
  // per chain rather than inferred from the letters, which cannot distinguish
  // them. Absent, every chain is protein, which is what every caller before
  // nucleic acids meant.
  const chainKinds = chains.map((_, index) => options.chainKinds?.[index] ?? "protein");
  const modificationOf = new Map();
  for (const modification of options.modifications ?? []) {
    modificationOf.set(`${modification.chain}:${modification.position}`, modification);
  }
  // Which chain each residue came from, so a reader can ask what its letters
  // mean without re-deriving the split.
  const chainOfResidue = [];
  const residues = [];
  chains.forEach((chain, chainIndex) => {
    for (let at = 0; at < chain.length; at += 1) {
      const asked = modificationOf.get(`${chainIndex}:${at + 1}`) ?? null;
      // 🔴 RESOLVED AGAINST THE END OF THE CHAIN HERE, ONCE. The dictionary
      // describes a free amino acid, so a modified residue arrives carrying the
      // OXT it loses on forming a peptide bond: ten atoms in the middle of a
      // chain and eleven at the C-terminus, which is what AF3 counts. See
      // polymerResidue.
      const modification = asked === null
        ? null : polymerResidue(asked, at === chain.length - 1);
      residues.push({
        code: chain[at],
        kind: chainKinds[chainIndex],
        // 🔴 THE TERMINAL RULE IS AT THE OTHER END FOR A NUCLEOTIDE. A protein
        // residue takes its extra atom (OXT) at the chain's LAST residue; a
        // nucleotide takes its extra atom (OP3) at the FIRST.
        //
        // 🔴 AND NOT EVERY MODEL HAS ONE. ESMFold2's featuriser uses the
        // internal atom set at both ends of every chain - its DNA_HEAVY_ATOMS
        // list has no OP3 and its dumps carry no OXT - so `terminalAtoms:
        // false` is what a caller building for it passes. It is off only when
        // asked, because every AF3-shaped caller means the AF3 rule.
        terminal: options.terminalAtoms === false ? false
          : (chainKinds[chainIndex] === "protein"
            ? at === chain.length - 1
            : at === 0),
        modification,
        // 🔴 boltz2 KEEPS A MODIFIED RESIDUE IN ONE TOKEN. Every other family
        // atomises it - one token per atom - and giving boltz2 the atomised
        // form tears the residue apart: SEP's own bonds at ratio 1.813 where
        // the others are 0.73-1.16. See `modifiedAsOneToken` in dialect.js.
        tokens: modification === null || options.modifiedAsOneToken === true
          ? 1 : modification.atoms.length,
      });
      chainOfResidue.push(chainIndex);
    }
  });
  // 🔴 AN ALIGNMENT COLUMN IS NOT A TOKEN INDEX, AND USED TO BE ASSUMED TO BE
  // ONE. An A3M has one column per RESIDUE of the PROTEIN chains, and a batch
  // has one token per residue only in the simplest case: a modified residue is
  // several tokens, a ligand is several more, and a nucleic chain has no
  // columns in a protein alignment at all. Copying a row in flat - which is
  // what `msa.set(row, ...)` did - lines the alignment up with the tokens for
  // as long as those agree and shifts it silently from the first modified
  // residue onward, laying one residue's homologs over another's.
  //
  // Measured against AF3, twice, because the two cases differ:
  //   - a modified residue's tokens ALL REPEAT the parent residue's column
  //     (SEP at position 3 gives ten tokens, every one of them reading the
  //     alignment's third column - 15 in every row, the gapped one included)
  //   - a nucleic token takes MSA_GAP in a protein row, and its own aatype in
  //     the query row.
  const msaColumnOfResidue = new Int32Array(residueCount).fill(-1);
  {
    // Only the protein chains are in the alignment, in chain order, which is
    // the order its columns are in.
    let column = 0;
    for (let residue = 0; residue < residueCount; residue += 1) {
      if (residues[residue].kind !== "protein") continue;
      msaColumnOfResidue[residue] = column;
      column += 1;
    }
  }
  const polymerTokens = residues.reduce((sum, residue) => sum + residue.tokens, 0);
  const tokens = polymerTokens + ligandTokens;
  if (tokens === 0) throw new Error("featuriseProtein: empty sequence");
  const chainLengths = chains.map((chain) => chain.length);
  // 🔴 NOT COMPUTED WHEN THERE ARE NO RESIDUES. A ligand on its own is a valid
  // fold - AF3 accepts one - and both of these reject a zero-length sequence,
  // rightly, because a zero-length CHAIN is a bug. There is no chain here to be
  // wrong about: they are read only inside the polymer loop below, which does
  // not run.
  // ...over RESIDUES, not tokens: chain identity is a property of the residue,
  // and every token of a modified one repeats it.
  const identity = residueCount === 0
    ? null : chainIdentity(residueCount, chainLengths, chains);
  const withinChain = residueCount === 0
    ? null : residueIndexPerChain(residueCount, chainLengths);
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

  // Where each modified residue's tokens sit, so a writer can name its atoms:
  // the batch's `sequence` covers one letter per residue and a modified
  // residue's tokens index into it as one letter for all of them.
  const modifiedSpans = [];
  // 🔴 WHICH RESIDUE A TOKEN BELONGS TO, WHICH USED TO BE THE TOKEN'S OWN INDEX.
  // Anything that reads `sequence[token]` - a PDB writer naming an atom, a
  // checker looking up a conformer - is wrong from the first modified residue
  // onwards without this, and wrong in a way that still finds A residue and so
  // reports a plausible disagreement rather than an error. -1 for a ligand,
  // which is not a residue at all.
  const residueOfToken = new Int32Array(tokens).fill(-1);
  let token = 0;
  // 🔴 ONE REFERENCE SPACE PER RESIDUE, COUNTED SEPARATELY FROM THE TOKENS.
  // They used to be the same number because a residue was a token; a modified
  // residue makes them diverge, and AF3 keeps counting SPACES - a chain with a
  // phosphoserine at position 3 gives the residue after it uid 3, not uid 12.
  // The uid decides only whether a PAIR of atoms may compare reference
  // coordinates, so a wrong one silently lets the atom encoder compare two
  // residues that were oriented independently.
  let space = 0;
  for (let residue = 0; residue < residueCount; residue += 1) {
    const { code, kind, terminal, modification } = residues[residue];
    // 🔴 lastOfChain IS THE PROTEIN RULE AND ONLY THE PROTEIN RULE. `terminal`
    // above already knows which end this residue's kind cares about; keeping
    // the old set here would put an OXT on a nucleotide's last base.
    const isCTerminal = terminal;
    void lastOfChain;
    const asym = identity.asymId[residue] + 1;
    const entity = identity.entityId[residue] + 1;
    const sym = identity.symId[residue] + 1;
    // AF3 counts these from one; chains.js counts from zero. See the note above.
    const number = withinChain[residue] + 1;

    if (modification === null) {
      // 🔴 THE KIND PICKS THE ALPHABET, NOT THE LETTER. A in a protein chain is
      // aatype 0 and in an RNA chain 22; there is no way to tell from the
      // letter, and getting it wrong folds a different molecule that looks
      // entirely reasonable.
      aatype[token] = kind === "protein"
        ? aatypeFor(code) : (nucleicAatypeFor(kind, code) ?? UNK_AATYPE);
      residueIndex[token] = number;
      tokenIndex[token] = token + 1;
      asymId[token] = asym;
      entityId[token] = entity;
      symId[token] = sym;
      seqMask[token] = 1;
      residueOfToken[token] = residue;

      const atoms = kind === "protein"
        ? conformerFor(code, isCTerminal)
        : (nucleicConformerFor(kind, code, isCTerminal) ?? conformerFor("X", false));
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
      // 🔴 A NUCLEOTIDE HAS NEITHER CB NOR CA, AND ITS CENTRE IS IN THE BASE.
      // Read out of AF3's own gather rather than guessed: C4 for a purine and
      // C2 for a pyrimidine - ring atoms, not the primed sugar carbons, and not
      // the C1' the sugar hangs the base off, which was the plausible wrong
      // answer. The distinction is by base, since both names exist in both:
      // a pyrimidine has a C4 as well, so matching on the name alone would take
      // the wrong atom in three of the five components.
      const purine = code === "A" || code === "G";
      const beta = kind === "protein"
        ? (atoms.find((atom) => atom[1] === "CB") ?? atoms.find((atom) => atom[1] === "CA"))
        : atoms.find((atom) => atom[1] === (purine ? "C4" : "C2"));
      if (beta) pseudoBetaSlot[token] = beta[0];
      // 🔴 ONE REFERENCE SPACE PER TOKEN. Every atom of a residue shares its uid,
      // and the atom encoder uses the uid only to decide whether a PAIR of atoms
      // may compare reference coordinates at all - which is what makes the random
      // per-residue orientation harmless.
      for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = space;
      space += 1;
      token += 1;
      continue;
    }

    // 🔴 A MODIFIED RESIDUE, ONE TOKEN PER ATOM, AND ITS REFERENCE SPACE IS
    // SHARED. Ten tokens carrying one atom each are still ONE rigid conformer,
    // so they take a single uid between them - exactly the reasoning the
    // ligand loop below spells out. Giving each its own would tell the atom
    // encoder that the phosphate may not be compared with the backbone.
    const uid = space;
    space += 1;
    // 🔴 boltz2's SINGLE TOKEN, WHICH IS A DIFFERENT SHAPE AND NOT A SPECIAL
    // CASE OF THE LOOP BELOW: all of the residue's atoms go into ONE token's
    // dense slots, where the atomised form puts each in slot zero of its own
    // token. The restype is the UNKNOWN one (`atomizedUnknownRestype`, already
    // true for boltz2) and `modifiedSpans` still describes the residue so a
    // writer names it - it is one token wide now rather than ten.
    if (options.modifiedAsOneToken === true) {
      modifiedSpans.push({ from: token, count: 1, code: modification.code,
                           residue, atoms: modification.atoms,
                           bonds: modification.bonds, oneToken: true });
      aatype[token] = options.atomizedUnknownRestype === true
        ? UNK_AATYPE : aatypeFor(code);
      residueIndex[token] = number;
      tokenIndex[token] = token + 1;
      asymId[token] = asym;
      entityId[token] = entity;
      symId[token] = sym;
      seqMask[token] = 1;
      residueOfToken[token] = residue;
      for (let atom = 0; atom < modification.atoms.length; atom += 1) {
        const source = modification.atoms[atom];
        // 🔴 THE COMPONENT'S OWN SLOT, NOT A COMPACTED ONE. A removed leaving
        // atom leaves its dense slot EMPTY - a mid-chain phosphoserine is
        // N,CA,CB,OG,C,O,_,P,O1P,O2P,O3P with a hole at 6 where the OXT was -
        // and compacting shifts the phosphate and its three oxygens down one.
        const flat = token * DENSE + (source.componentSlot ?? atom);
        refMask[flat] = 1;
        refElement[flat] = source.element;
        refCharge[flat] = source.charge;
        refPos[flat * 3] = source.x;
        refPos[flat * 3 + 1] = source.y;
        refPos[flat * 3 + 2] = source.z;
        writeAtomName(refAtomNameChars, flat, source.name, source.element,
                      options.atomizedElementNames);
        realAtoms.push(flat);
      }
      // 🔴 THE PSEUDO-BETA IS CB, THE SAME RULE A STANDARD RESIDUE TAKES - and
      // slot zero was wrong. A one-token modified residue is a residue, so its
      // representative atom is its beta carbon (its alpha carbon if it has
      // none), not whichever atom the dictionary happens to list first, which
      // for a phosphoserine is N. Caught by `check-batch-fields.js` against the
      // reference's own gather: token 2 wants slot 2 (CB) where this wrote 0.
      const slotOfName = (name) => {
        const found = modification.atoms.find((a) => a.name === name);
        return found === undefined ? -1
          : (found.componentSlot ?? modification.atoms.indexOf(found));
      };
      const betaAt = slotOfName("CB");
      const alphaAt = slotOfName("CA");
      pseudoBetaSlot[token] = betaAt >= 0 ? betaAt : (alphaAt >= 0 ? alphaAt : 0);
      for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = uid;
      token += 1;
      continue;
    }
    modifiedSpans.push({ from: token, count: modification.atoms.length,
                         code: modification.code, residue,
                         atoms: modification.atoms, bonds: modification.bonds });
    for (let atom = 0; atom < modification.atoms.length; atom += 1) {
      const source = modification.atoms[atom];
      // ...the PARENT's aatype, on every one of them. AF3 writes serine for all
      // ten tokens of a phosphoserine, in `aatype` and in the MSA alike.
      //
      // 🔴 EXCEPT boltz2 AND rosettafold3, which write the UNKNOWN restype
      // instead - `atomized_unknown_restype`. An atomised token is not a
      // serine to them, it is one atom of something, so it takes X. Measured
      // against both references on 6MRR + GOL + SEP@3: all ten of the
      // phosphoserine's tokens, 15 -> 20, and it carries into `profile`, which
      // is a one-hot over the same alphabet.
      aatype[token] = options.atomizedUnknownRestype === true
        ? UNK_AATYPE : aatypeFor(code);
      residueIndex[token] = number;
      tokenIndex[token] = token + 1;
      asymId[token] = asym;
      entityId[token] = entity;
      symId[token] = sym;
      seqMask[token] = 1;
      residueOfToken[token] = residue;

      const flat = token * DENSE;          // one atom, and it sits in slot zero
      refMask[flat] = 1;
      refElement[flat] = source.element;
      refCharge[flat] = source.charge;
      refPos[flat * 3] = source.x;
      refPos[flat * 3 + 1] = source.y;
      refPos[flat * 3 + 2] = source.z;
      writeAtomName(refAtomNameChars, flat, source.name, source.element,
                    options.atomizedElementNames);
      realAtoms.push(flat);
      // Each token holds exactly one atom, so that atom is its pseudo-beta.
      pseudoBetaSlot[token] = 0;
      for (let slot = 0; slot < DENSE; slot += 1) refSpaceUid[token * DENSE + slot] = uid;
      token += 1;
    }
  }

  // 🔴 THE LIGANDS, AFTER EVERY POLYMER CHAIN. Their tokens continue the token
  // index, take their own asym_id, and share ONE ref_space_uid across the whole
  // component - the six atoms of a glycerol are one rigid conformer, not six
  // independent ones, and giving each its own uid tells the atom encoder they
  // may not be compared, which is the opposite of true.
  let asym = chainLengths.length;
  // Keyed on what the ligand IS, not on what it is called; see below.
  const entityOfLigand = new Map();
  const copiesOfEntity = new Map();
  let ligandToken = polymerTokens;
  // Where each ligand's tokens sit, so a writer can name them: the batch's
  // `sequence` covers the polymers only, and a ligand token indexed into it
  // comes back undefined and is written as UNK.
  const ligandSpans = [];
  for (const ligand of ligands) {
    asym += 1;
    // ...AND ITS BONDS TRAVEL WITH IT, because a writer needs them too. The
    // bond matrix beside this is token x token and one direction only, which is
    // the shape the MODEL wants; a PDB's CONECT records want the pairs, and
    // scanning L^2 cells per trajectory frame to recover them is the wrong way
    // round when the list is right here.
    ligandSpans.push({ from: ligandToken, count: ligand.atoms.length, code: ligand.code,
                       bonds: ligand.bonds });
    // Identical codes are one entity, and each occurrence is a copy of it -
    // the same rule chainIdentity applies to repeated sequences.
    //
    // 🔴 AND "IDENTICAL" IS THE MOLECULE, NOT THE CODE, WHICH ONLY MATTERS
    // SINCE A LIGAND CAN ARRIVE AS A STRUCTURE. A CCD code identifies its
    // contents - two ATPs really are one entity - so for every fold this port
    // has ever done the two keys agree. A SMILES ligand has no code at all and
    // is given one, and two DIFFERENT structures in one job both arrived as
    // `LIG`: benzene and glycerol came out sharing an entity_id, with the
    // model told that six carbons and a glycerol are two copies of one thing.
    // Keying on what the component IS rather than on what it is called fixes
    // that and cannot change a dictionary fold, where the two are equivalent.
    const identity = `${ligand.code}|${ligand.atoms.length}|`
      + `${ligand.atoms.map((atom) => `${atom.element}:${atom.charge}`).join(",")}|`
      + `${ligand.bonds.map((bond) => `${bond.from}-${bond.to}:${bond.order}`).join(",")}`;
    if (!entityOfLigand.has(identity)) entityOfLigand.set(identity, entityOfLigand.size);
    const entity = entityOfLigand.get(identity);
    const copy = (copiesOfEntity.get(entity) ?? 0) + 1;
    copiesOfEntity.set(entity, copy);
    // ...and its space continues the residues' count, not the token index. The
    // two are the same number until a modified residue makes them differ.
    const uid = space;
    space += 1;
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
      writeAtomName(refAtomNameChars, flat, source.name, source.element,
                    options.atomizedElementNames);
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
  //
  // 🔴 A MODIFIED RESIDUE'S OWN BONDS GO HERE TOO, and its peptide bonds to its
  // neighbours do NOT. AF3 puts a phosphoserine's nine internal bonds through
  // the same ligand-bond machinery as a ligand's, and leaves the backbone
  // connectivity implicit in residue_index, exactly as it is for an
  // unmodified chain - `polymer_ligand_bonds` comes back empty for one.
  //
  // 🔴 EXCEPT rosettafold3, WHICH BONDS IT BACK INTO THE CHAIN. That is
  // `atomized_backbone_bonds`, and it is rf3's alone: on 6MRR + GOL + SEP@3
  // the reference lists 18 bonded token pairs for rf3 and 14 for every other
  // family, and the four extra are the peptide bonds either side of the
  // atomised residue - the preceding residue to its N, and its C to the
  // following residue, each way round. Without them the phosphoserine is a
  // ligand floating beside the chain as far as the pair track is concerned.
  const bondedGroups = [
    // 🔴 A ONE-TOKEN MODIFIED RESIDUE CONTRIBUTES NO TOKEN-TOKEN BONDS. Its
    // bonds are INSIDE the token, and `base + bond.from` only means a token
    // while each atom is its own; under boltz2's convention it walks off the
    // residue and bonds the neighbours instead - measured as 18 pairs the
    // reference does not have, `2-3 3-2 3-4 3-6 4-3 4-5`, where 2 is the
    // phosphoserine and 3 and 4 are the residues after it. The atoms are still
    // bonded to each other; the pair track just is not where that is said.
    ...modifiedSpans.filter((span) => span.oneToken !== true)
      .map((span) => ({ base: span.from, bonds: span.bonds })),
    ...ligands.map((ligand, index) => ({
      base: polymerTokens + ligands.slice(0, index)
        .reduce((sum, earlier) => sum + earlier.atoms.length, 0),
      bonds: ligand.bonds,
    })),
  ];
  // 🔴 AND THE BOND ORDER IS A SECOND PLANE, WHICH NOTHING HERE BUILT. boltz2's
  // z-init reads TWO planes - the contact flag and the bond ORDER - and it is
  // the only family with `tokenBondsTypeEmbed`, so it is the only one that
  // notices. `embedder-webgpu.js` and `confidence-webgpu.js` both read
  // `input.bondOrderMatrix`, `fold.js` forwards `batch.bondOrderMatrix`, and
  // the featuriser never set it: five consumers and no producer, so every bond
  // reached boltz2 with order 0 - "unspecified" - where the CCD says 1 or 2.
  //
  // Measured by tools/check-ligand-path.mjs, which is what found it: glycerol
  // folded into boltz2 came apart at bond rms 3.602 A, C1-O1 at 6.97 A against
  // a 1.43 ideal, while its pLDDT read 92.38. Every other family was 0.044 to
  // 0.069. The order is in the component's own bond table and has been all
  // along - `parseCcdComponent` returns it - so this is a channel that was
  // parsed, forwarded and never filled.
  let bondMatrix;
  let bondOrderMatrix;
  if (bondedGroups.some((group) => group.bonds.length > 0)) {
    bondMatrix = new Float32Array(tokens * tokens);
    bondOrderMatrix = new Float32Array(tokens * tokens);
    for (const { base, bonds } of bondedGroups) {
      for (const bond of bonds) {
        bondMatrix[(base + bond.from) * tokens + (base + bond.to)] = 1;
        bondOrderMatrix[(base + bond.from) * tokens + (base + bond.to)] = bond.order ?? 1;
        if (options.symmetriseBonds) {
          bondMatrix[(base + bond.to) * tokens + (base + bond.from)] = 1;
          bondOrderMatrix[(base + bond.to) * tokens + (base + bond.from)] = bond.order ?? 1;
        }
      }
    }
    if (options.atomizedBackboneBonds === true) {
      for (const span of modifiedSpans) {
        // The span's own N and C, by name - the atom ORDER is the component's
        // and is not something to count on.
        const slotOf = (name) => span.atoms.findIndex((atom) => atom.name === name);
        const nitrogen = slotOf("N");
        const carbon = slotOf("C");
        const sameChain = (residue) => residue >= 0 && residue < residueCount
          && chainOfResidue[residue] === chainOfResidue[span.residue];
        // 🔴 BOTH DIRECTIONS, unlike the internal bonds, which the reference
        // lists one way round. Its own gather carries 1-2 AND 2-1, 6-12 AND
        // 12-6, where the nine internal SEP bonds appear once each.
        const link = (a, b) => {
          if (a < 0 || b < 0 || a >= tokens || b >= tokens) return;
          bondMatrix[a * tokens + b] = 1;
          bondMatrix[b * tokens + a] = 1;
          // A peptide bond is a single bond. rf3 has no bond-order embedding,
          // so this changes nothing for it and is right rather than blank.
          bondOrderMatrix[a * tokens + b] = 1;
          bondOrderMatrix[b * tokens + a] = 1;
        };
        // The neighbouring residue's token is the one adjacent to the span,
        // which holds while the neighbour is not itself atomised - the case
        // this port has a reference dump for.
        if (nitrogen >= 0 && sameChain(span.residue - 1)) link(span.from - 1, span.from + nitrogen);
        if (carbon >= 0 && sameChain(span.residue + 1)) link(span.from + carbon, span.from + span.count);
      }
    }
    bondMatrix[0] = 0;
    bondOrderMatrix[0] = 0;
  }

  const atomCount = realAtoms.length;

  // 🔴 `subsets` COMES FROM HERE AND NOWHERE ELSE. It used to be computed a
  // second time, three hundred lines above, from `tokens * DENSE` - so the
  // batch's `shape.subsets` and the gathers sized beside it were two
  // derivations of one number, which is the failure this repository keeps
  // meeting: a dispatch sized for one width against buffers built for another.
  // They agreed only while both were wrong.
  const {
    subsets, keys, tokenAtomsToQueries, queriesToTokenAtoms, queriesToKeys,
    tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta,
  } = atomGathers({ tokens, dense: DENSE, realAtoms, pseudoBetaSlot,
                    // rf3, opendde and protenix clamp the key window and mask
                    // its out-of-range slots; see the note on the rule.
                    paddedKeys: options.paddedAtomKeys === true,
                    qblockKeys: options.qblockAtomKeys === true });

  // The MSA. Row zero is the query; anything the caller supplies follows.
  const extra = options.msa ?? [];
  // 🔴 WITH NO ALIGNMENT, THREE FAMILIES GET THE QUERY TWICE AND FOUR GET IT
  // ONCE, and this port gave all seven one row. AlphaFold 3 concatenates a
  // PAIRED and an UNPAIRED block, so a chain with no homologs contributes its
  // own sequence to each and the model sees depth 2; boltz's featuriser - which
  // protenix, IntelliFold-2 and RoseTTAFold3 all fork or match - emits a
  // depth-1 `dummy_msa` and finds nothing to pair. Counted in the reference's
  // own batches on 6MRR: alphafold3, openbind0 and opendde have TWO live rows
  // and boltz2, protenix2, intellifold2 and rosettafold3 have ONE.
  //
  // 🔴 AND IT IS NOT COSMETIC. The outer product mean over two identical rows
  // is unchanged, but the pair-weighted averaging and the row transition are
  // DEPTH-sensitive - upstream measured the same convention at 4.3% of
  // esmfold2's MSA injection - so the three that want two rows have been
  // folding single sequences one row short. Only where the alignment is EMPTY:
  // with homologs the query's second copy is what
  // `deduplicateUnpairedAgainstPaired` already removes.
  const duplicateQuery = extra.length === 0 && options.duplicateQueryRow === true;
  const sequences = 1 + extra.length + (duplicateQuery ? 1 : 0);
  const msa = new Int32Array(sequences * tokens);
  const msaMask = new Float32Array(sequences * tokens).fill(1);
  const deletionMatrix = new Float32Array(sequences * tokens);
  // 🔴 A LIGAND TOKEN IS A GAP IN THE MSA, NOT AN UNKNOWN RESIDUE. Its aatype
  // is UNK (20) but AF3 writes MSA_GAP (21) in the alignment and in the
  // profile - an alignment has nothing to say about an atom. Copying aatype
  // across, which is right for every polymer token, puts a 20 there instead and
  // tells the model the ligand is a row of unknown amino acids.
  const queryRow = Int32Array.from(aatype);
  // 🔴 boltz2 MOVES `aatype` TO UNKNOWN AND LEAVES THE ALIGNMENT ALONE, WHERE
  // rosettafold3 MOVES BOTH. Two conventions, and one flag reported them as
  // one: with `atomizedUnknownRestype` alone, boltz2's profile came out 20
  // where the reference has 15, because the query row of the MSA is built from
  // `aatype` and the profile is built from the MSA. Read off the references'
  // own query rows at 6MRR + GOL + SEP@3, tokens 2..11:
  //     alphafold3   15 15 15 ...   aatype 15
  //     boltz2       15 15 15 ...   aatype 20
  //     rosettafold3 20 20 20 ...   aatype 20
  // So the alignment keeps the PARENT residue for everyone but rf3, whose
  // atomised token is unknown wherever it appears.
  if (options.atomizedUnknownRestype === true
      && options.atomizedUnknownMsa !== true) {
    for (const span of modifiedSpans) {
      const parent = aatypeFor(residues[span.residue].code);
      for (let at = 0; at < span.count; at += 1) queryRow[span.from + at] = parent;
    }
  }
  for (let token = polymerTokens; token < tokens; token += 1) queryRow[token] = MSA_GAP;
  msa.set(queryRow, 0);
  // ...and again, for the families whose two blocks each contribute it.
  if (duplicateQuery) msa.set(queryRow, tokens);
  // Which column of the alignment each TOKEN reads, or -1 for a token the
  // alignment does not describe: a ligand's atom, or any token of a nucleic
  // chain.
  const msaColumnOfToken = new Int32Array(tokens).fill(-1);
  const nucleicToken = new Uint8Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    const residue = residueOfToken[token];
    if (residue < 0) continue;
    msaColumnOfToken[token] = msaColumnOfResidue[residue];
    if (residues[residue].kind !== "protein") nucleicToken[token] = 1;
  }
  // Where the unpaired block starts in the finished array, which is where a
  // chain's own alignment begins. Read before `unpairedFrom` is declared below,
  // so it is computed here by the same rule.
  const unpairedFromRow = () => options.unpairedFrom ?? (extra.length === 0 ? 0 : 1);
  // 🔴 A NUCLEIC CHAIN CONTRIBUTES ONE ROW OF ITS OWN, AND IT IS NOT ROW ZERO.
  // AF3 stacks each chain's unpaired block side by side and pads the short ones
  // with gaps, so a DNA chain - whose alignment is only itself - puts its own
  // sequence in the FIRST ROW OF THE UNPAIRED BLOCK and gaps below it. Measured:
  // in a protein+DNA batch rows 0 and 1 both read 26 28 27 29 at the DNA
  // columns and rows 2 and 3 read 21. Gapping every row but the query is the
  // natural reading and is one row short, which is a row the model reads as the
  // chain being absent from its own alignment.
  const nucleicRow = unpairedFromRow();
  for (let row = 0; row < extra.length; row += 1) {
    const base = (row + 1) * tokens;
    const deletions = options.deletionMatrix?.[row];
    const nucleicHere = row + 1 === nucleicRow ? "own" : MSA_GAP;
    for (let token = 0; token < tokens; token += 1) {
      const column = msaColumnOfToken[token];
      if (column < 0 && nucleicToken[token]) {
        msa[base + token] = nucleicHere === "own" ? aatype[token] : MSA_GAP;
        continue;
      }
      msa[base + token] = column < 0 ? MSA_GAP : (extra[row][column] ?? MSA_GAP);
      if (deletions !== undefined) {
        deletionMatrix[base + token] = column < 0 ? 0 : (deletions[column] ?? 0);
      }
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
  //
  // 🔴 AND THE PROFILE'S ROWS ARE ALIGNMENT COLUMNS TOO. `profileMsa` comes
  // straight from the A3M and was indexed here by token, which is the same
  // off-by-a-modified-residue as the block above.
  const codeAt = profileRows === null
    ? (row, token) => msa[(unpairedFrom + row) * tokens + token]
    : (row, token) => {
      const column = msaColumnOfToken[token];
      if (column < 0) return -1;
      return profileRows[row] === undefined ? -1 : profileRows[row][column];
    };
  const deletionAt = profileRows === null
    ? (row, token) => deletionMatrix[(unpairedFrom + row) * tokens + token]
    : (row, token) => {
      const column = msaColumnOfToken[token];
      return column < 0 ? 0 : (options.profileDeletionMatrix?.[row]?.[column] ?? 0);
    };
  // 🔴 A NUCLEIC TOKEN'S PROFILE IS ONE-HOT AT ITS OWN AATYPE, NOT AT THE GAP.
  // AF3 runs get_profile_features PER CHAIN, so a DNA chain - which has no
  // alignment beyond itself - gets a profile of its own sequence: measured as
  // exactly 1.0 at restype 26 for a leading A. Letting it fall out of the loop
  // below would put that 1.0 at MSA_GAP instead, which says the chain is absent.
  for (let row = 0; row < profileDepth; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      if (nucleicToken[token]) continue;
      const code = codeAt(row, token);
      if (code >= 0 && code < RESTYPES) profile[token * RESTYPES + code] += 1 / profileDepth;
      deletionMean[token] += deletionAt(row, token) / profileDepth;
    }
  }

  for (let token = 0; token < tokens; token += 1) {
    if (nucleicToken[token]) profile[token * RESTYPES + aatype[token]] = 1;
  }

  // 🔴 THE NAME THE MODEL READS AND THE NAME A PDB CARRIES ARE NOT THE SAME
  // NAME, once `atomizedElementNames` is on. rf3 renames every atomised atom to
  // its element symbol, which is right for the FEATURE and wrong for the FILE:
  // the PDB writer reads the same array, so a glycerol came out as six atoms
  // called C, O, C, O, C, O in one residue - names that are not unique within a
  // residue, which the format does not allow and a viewer keying on them
  // collapses. The reference renames its batch feature; what it writes out is
  // its own business. So `displayAtomNameChars` keeps the component's real
  // names and everything that produces OUTPUT reads it.
  //
  // It is the same array unless the rename is on, so this costs nothing for the
  // other six families.
  let displayAtomNameChars = refAtomNameChars;
  if (options.atomizedElementNames === true) {
    displayAtomNameChars = Int32Array.from(refAtomNameChars);
    const restore = (base, atoms) => {
      for (let at = 0; at < atoms.length; at += 1) {
        writeAtomName(displayAtomNameChars, (base + at) * DENSE, atoms[at].name,
                      atoms[at].element, false);
      }
    };
    for (const span of modifiedSpans) restore(span.from, span.atoms);
    for (let index = 0; index < ligands.length; index += 1) {
      const base = polymerTokens + ligands.slice(0, index)
        .reduce((sum, earlier) => sum + earlier.atoms.length, 0);
      restore(base, ligands[index].atoms);
    }
  }

  // 🔴 THE REFERENCE CONFORMERS ARE CENTRED PER `ref_space_uid`, AND FIVE OF
  // THE SIX FAMILIES EXPECT IT. `CENTRE_REF_CONFORMERS` in af3-any-model is
  // ('boltz2', 'openfold3', 'openbind0', protenix*, 'opendde') - everything but
  // stock AlphaFold 3, which is the reference implementation, and intellifold2,
  // which passes centering=False. Their featurisers all subtract the group mean
  // (boltz2 `centering=True` per ref_space_uid; openbind0
  // `pos_centered = xl - mean_xl`; protenix/opendde `random_transform(
  // centralize=True)`), and this port did not - measured, glycine's four atoms
  // mean to exactly (0,0,0) in the reference's batch and to (1.31, -0.02, 0.58)
  // here.
  //
  // 🔴 IT IS THE RAW `ref_pos` CHANNEL ONLY, and that is why it hid: the atom
  // encoder reads the positions BOTH raw and through a translation-invariant
  // pairwise difference, so half the module cannot see a translation at all.
  // What it cost is the floor under every sequence-featurised oracle
  // comparison - openbind0's `target_feat` 2.89e-2 from a sequence against
  // 4.93e-8 from the reference's own batch.
  //
  // Centring is applied to every family here rather than behind a dialect flag
  // ONLY IF the dialect says so; stock AF3 must keep its uncentred CCD ideals.
  if (options.centreRefConformers === true) {
    const sums = new Map();
    for (let slot = 0; slot < tokens * DENSE; slot += 1) {
      if (refMask[slot] === 0) continue;
      const uid = refSpaceUid[slot];
      const entry = sums.get(uid) ?? [0, 0, 0, 0];
      entry[0] += refPos[slot * 3]; entry[1] += refPos[slot * 3 + 1];
      entry[2] += refPos[slot * 3 + 2]; entry[3] += 1;
      sums.set(uid, entry);
    }
    for (let slot = 0; slot < tokens * DENSE; slot += 1) {
      if (refMask[slot] === 0) continue;
      const entry = sums.get(refSpaceUid[slot]);
      if (entry === undefined || entry[3] === 0) continue;
      refPos[slot * 3] -= entry[0] / entry[3];
      refPos[slot * 3 + 1] -= entry[1] / entry[3];
      refPos[slot * 3 + 2] -= entry[2] / entry[3];
    }
  }

  return {
    sequence: joined, chains, chainLengths,
    tokens, dense: DENSE, subsets, atomCount, sequences,
    shape: { tokens, dense: DENSE, subsets, queries: QUERIES, keys },
    aatype, profile, deletionMean,
    msa, msaMask, deletionMatrix,
    residueIndex, tokenIndex, asymId, entityId, symId, seqMask,
    refPos, refMask, refElement, refCharge, refAtomNameChars, refSpaceUid,
    displayAtomNameChars,
    // AF3 keeps these separate and they are equal for a protein-only chain:
    // every atom the model predicts is one it has a reference conformer for.
    predDenseAtomMask: refMask,
    bondMatrix, bondOrderMatrix, ligandSpans, modifiedSpans, residueOfToken,
    chainKinds, chainOfResidue,
    tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
    tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
  };
}

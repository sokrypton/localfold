/**
 * The join between the two models: an AF3 prediction in, a designed sequence
 * out.
 *
 * 🔴 THE INTERFACE IS A PDB STRING, WHICH IS NOT A COMPROMISE. `foldAf3`
 * already writes one - `toPdb` in src/af3/fold.js, one letter per chain, real
 * residue names - and `structureFromText` in the MPNN port already parses one
 * into exactly the `{X, mask, residueIdx, chainLabels}` that `Model.encode`
 * takes. Reaching into AF3's dense atom layout instead would mean a second
 * implementation of the residue/token distinction that toPdb already got
 * right, and the two would drift.
 *
 * 🔴 THE FIXED CHAINS' OWN SEQUENCE HAS TO BE HANDED TO THE SAMPLER. MPNN
 * decodes autoregressively over the whole structure and reads the amino acid
 * at every position it has already seen; `Model.sample` starts from
 * `opts.S ?? fill(20)`, which is all-X, and never decodes a position the chain
 * mask excludes. So without `S` the target chain is conditioned on as a
 * poly-unknown, and the binder is designed against a target the model cannot
 * read. `structure.S` is the parsed structure's real sequence and is what goes
 * in.
 *
 * The MPNN modules under ./mpnn/ are a mirror of a separate checkout; see
 * SOURCE.md there and tools/sync-mpnn.py.
 */
import { ALPHABET } from "./mpnn/constants.js";
import { Model, sequenceToString } from "./mpnn/model.js";
import { NA_ALPHABET, naDisplaySequence } from "./mpnn/na.js";
import { structureFromText } from "./mpnn/pdb.js";
import { Weights } from "./mpnn/weights.js";
import { enableAcceleration } from "./mpnn/accel.js";
import { DESIGNERS } from "./designers.js";
import { designBias } from "./sample-sequence.js";

/** Where the page serves the mirror from. */
export const CHECKPOINT_BASE = "./web/public/mpnn/";
export const DEFAULT_KERNELS = "./web/vendor/mpnn/kernels.wasm";
export const DEFAULT_DESIGNER = "soluble";

/** One promise per family, so switching back and forth re-reads nothing. */
const loaded = new Map();

/**
 * Read a checkpoint and install the SIMD kernel.
 *
 * 🔴 ON DEMAND AND MEMOISED PER FAMILY. The four checkpoints are 16 MB
 * together and a page that read all of them on load would spend that before
 * anyone had chosen anything - while a page that re-read one per design cycle
 * would spend it eighteen times in a five-cycle hunt. Keyed by name rather
 * than by URL because the URL is derived from the name; two spellings of one
 * file would be two entries and two reads.
 *
 * 🔴 THE ACCELERATOR IS BEST EFFORT AND ITS ABSENCE IS NOT AN ERROR. It is a
 * 26 KB WebAssembly module holding one dense kernel; a runtime without SIMD
 * fails to validate it, `enableAcceleration` returns null, and the JS kernel
 * serves every call. It installs into `ops.linear` globally, so it is loaded
 * once and every family after the first gets it for free.
 *
 * @param {{name?: keyof typeof DESIGNERS, base?: string, kernels?: string,
 *          signal?: AbortSignal,
 *          onProgress?: (received: number, total: number) => void}} [options]
 * @returns {Promise<{model: Model, name: string, accelerated: boolean}>}
 */
export function loadDesigner(options = {}) {
  const name = options.name ?? DEFAULT_DESIGNER;
  const designer = DESIGNERS[name];
  if (designer === undefined) {
    throw new Error(`no designer ${name} (have ${Object.keys(DESIGNERS).join(", ")})`);
  }
  if (!loaded.has(name)) {
    loaded.set(name, (async () => {
      const weights = await Weights.fetch(
        `${options.base ?? CHECKPOINT_BASE}${designer.file}`,
        { signal: options.signal, onProgress: options.onProgress });
      const accelerator = await enableAcceleration(options.kernels ?? DEFAULT_KERNELS);
      return { model: new Model(weights), name, accelerated: accelerator !== null };
    })().catch((error) => {
      // ...a failed read is not cached: an aborted or offline first attempt
      // would otherwise make the family permanently unavailable to the page.
      loaded.delete(name);
      throw error;
    }));
  }
  return loaded.get(name);
}

/**
 * Which residues of a parsed structure belong to a chain.
 *
 * @param {{chainIds: string[]}} structure
 * @param {string} chain a PDB chain letter
 * @returns {Float32Array} 1 on that chain, 0 elsewhere
 */
export function chainMaskFor(structure, chain) {
  const mask = new Float32Array(structure.chainIds.length);
  for (let index = 0; index < structure.chainIds.length; index += 1) {
    if (structure.chainIds[index] === chain) mask[index] = 1;
  }
  return mask;
}

/**
 * Redesign one chain of a predicted structure.
 *
 * @param {Model} model
 * @param {object} options
 * @param {string} options.pdb the prediction, as `foldAf3` writes it
 * @param {string} options.chain the chain to design; every other chain is held
 * @param {number} [options.temperature] the reference's 0.1
 * @param {string} [options.omit] letters this cycle may not choose
 * @param {number} [options.alanineBias] added to alanine's logit; negative
 *   discourages it. See alanineBias() in sample-sequence.js.
 * @param {() => number} [options.random] pass `uniformFrom(seed)` to make the
 *   design reproducible
 * @returns {{sequence: string, full: string, score: number,
 *            designed: number, structure: object}} `sequence` is the designed
 *   chain alone, which is what goes back into the fold; `full` is every chain,
 *   colon-separated, in the order the structure lists them.
 */
export function designChain(model, options) {
  // 🔴 THE PARSE DEPENDS ON THE FAMILY, AND PARSING WRONG IS SILENT. Three
  // readings of one file:
  //
  //   * `ligands: true` keeps heteroatoms as an atom cloud. Only LigandMPNN's
  //     encoder looks at them; for anyone else they are memory and a longer
  //     parse.
  //   * `nucleicAsResidues: true` promotes DNA and RNA to model POSITIONS with
  //     their own backbone - and switches `S` and `sequence` to the 33-letter
  //     alphabet, so the structure object belongs to ONE family at a time.
  //   * neither, which is the protein-only reading.
  //
  // Handing NA-MPNN the protein-only parse is the quiet one: the nucleic chain
  // simply is not in the structure, the designed chain is graphed against
  // nothing, and every number downstream looks reasonable.
  const structure = structureFromText(options.pdb, {
    ligands: model.isLigand, nucleicAsResidues: model.isNA,
  });
  const alphabet = model.isNA ? NA_ALPHABET : ALPHABET;
  const chainMask = chainMaskFor(structure, options.chain);
  let designed = 0;
  for (const value of chainMask) designed += value;
  if (designed === 0) {
    throw new Error(`the prediction has no chain ${options.chain}`
      + ` (it has ${structure.chainList.join(", ")})`);
  }

  const result = model.sample(model.encode({
    X: structure.X,
    mask: structure.mask,
    residueIdx: structure.residueIdx,
    chainLabels: structure.chainLabels,
    // NA-MPNN graphs 18 atom slots over protein and nucleic alike and reads
    // the polymer type as a node label; the parse above filled all three.
    X16: structure.X16,
    X16Mask: structure.X16Mask,
    polytype: structure.polytype,
    // LigandMPNN's atom context. Empty arrays for everyone else, which its
    // encoder is the only one to look at anyway.
    ligandXyz: structure.ligandXyz,
    ligandType: structure.ligandType,
    ligandMask: structure.ligandMask,
  }), {
    batch: 1,
    temperature: options.temperature ?? 0.1,
    rng: options.random ?? Math.random,
    // See the note at the top: the fixed chains are conditioned on their own
    // amino acids, not on the all-X default.
    S: Int32Array.from(structure.S),
    chainMask,
    bias: designBias(structure.chainIds.length, {
      omit: options.omit, alanineBias: options.alanineBias, alphabet,
    }),
  });

  // 🔴 `result.seq` IS 21-LETTER, WHATEVER THE MODEL IS, AND READING NA-MPNN
  // WITH IT PRODUCES A PERFECTLY GOOD PROTEIN SEQUENCE THAT IS NOT THE ONE THE
  // MODEL CHOSE. `sequenceToString` in the mirror indexes `ALPHABET` -
  // `ACDEFGHIKLMNPQRSTVWYX` - while NA-MPNN's tokens are `NA_ALPHABET`,
  // `ARNDCQEGHILKMFPSTWYVX` then the nucleic letters. Same 21 letters, a
  // DIFFERENT ORDER, so token 11 is `N` in one and `K` in the other. Measured
  // on the Top7 fixture: `SKKITVTIKSKDKTKTITYEV...` read the wrong way is
  // `SNNLTYTLNSNENTNTLTWHY...` - twenty amino acids, right length, wrong
  // protein, and nothing downstream can tell. `naDisplaySequence` is the
  // mirror's own reader for that alphabet, and it also undoes the shared-token
  // trick that stores an RNA base as the corresponding DNA one.
  const letters = [...(model.isNA
    ? naDisplaySequence(result.S[0], structure.isRNA)
    : sequenceToString(result.S[0]))];
  const chainLetters = [];
  for (let index = 0; index < chainMask.length; index += 1) {
    if (chainMask[index] === 1) chainLetters.push(letters[index]);
  }
  // 🔴 THE CHAINS COME BACK IN THE STRUCTURE'S ORDER, NOT THE REQUEST'S. The
  // reference splits its designer's output on ":" and indexes it by a chain
  // letter through a fixed A=0, B=1 table, which is right only while the PDB
  // lists chains alphabetically. `chainList` is the order the file actually
  // used, so the caller can line these up itself.
  const byChain = new Map(structure.chainList.map((id) => [id, []]));
  for (let index = 0; index < structure.chainIds.length; index += 1) {
    byChain.get(structure.chainIds[index]).push(letters[index]);
  }
  return {
    sequence: chainLetters.join(""),
    full: structure.chainList.map((id) => byChain.get(id).join("")).join(":"),
    score: result.score[0],
    designed,
    structure,
    // 🔴 THE TOKENS AND THE ALPHABET THEY MEAN SOMETHING IN, TOGETHER. A
    // sequence handed over without its alphabet is what the note above is
    // about: twenty amino acids under either reading, and no way to tell which
    // was used. test/mpnn-bridge.test.js pins the branch with these, and a
    // caller wanting a profile or a logo needs them anyway.
    tokens: result.S[0],
    alphabet,
  };
}

/** MPNN's alphabet, re-exported so callers need not reach into the mirror. */
export { ALPHABET };

/**
 * hunt.js
 * -------
 * AI Context: WHAT MAKES THE DESIGN LOOP REAL ON THIS PAGE
 *
 * `webgpu/design/hunter-loop.js` is the method and takes its `fold` and
 * `design` as arguments, so it tests without a GPU. This module is the half
 * that supplies them: a real AlphaFold 3 fold and a real MPNN redesign, plus
 * the progress reporting a minutes-long run needs.
 *
 * 🔴 IT IS NOT THE PAGE. No DOM here beyond what the caller passes in: the page
 * hands over a `report` callback and gets one call per cycle, so the loop, the
 * wiring and the drawing are three separable things. The retired `web/hunter.js`
 * was all three at once in 795 lines.
 *
 * 🔴 THE FOLDER ROTATES PER CYCLE, which is the measured win (docs/DESIGN.md
 * 3C): a design then has to satisfy several models instead of overfitting one.
 * Only the AF3-lineage families are wired - they share `foldAf3` - so ESMFold2
 * and the rest are not in FOLDERS yet.
 */
import { getDevice } from "./model.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { CHAIN_IDS, runDesign } from "../webgpu/design/hunter-loop.js";
import { designChain, loadDesigner } from "../cpu/design/mpnn-bridge.js";
import { chooseDesigner } from "../cpu/design/designers.js";
import { contactProblem } from "../cpu/design/contact.js";
import { FOLDERS } from "./validate-design.js";

/** The folding models this driver can actually run, and what each is in af3-model's terms. */
export { FOLDERS } from "./validate-design.js";

/**
 * Run one hunt.
 *
 * @param {object} options
 * @param {string[]} options.chains every chain of the complex; the designed
 *   one's entry is replaced each cycle and starts empty
 * @param {string[]} [options.chainKinds] what each chain is made of
 *   (`"protein" | "dna" | "rna"`), which changes the FOLD and not only the
 *   designer: the featuriser reads a nucleic chain one token per base
 * @param {string[]} [options.ligandCodes] ligands beside the chains, each given
 *   a chain of its own by the featuriser
 * @param {{chain: string, position: number, code: string}[]} [options.modifications]
 *   modified residues - a phosphoserine, a methylated base - swapped in by CCD
 *   code at one position of one chain
 * @param {number} [options.chainIndex=0] which chain is designed
 * @param {string[]} [options.folders=["af3"]] which models fold, rotating one per cycle
 * @param {string} [options.designer="auto"] a key of DESIGNERS, or "auto"
 * @param {{ligands?: number, nucleic?: number}} [options.contents] what the
 *   target carries, for the automatic designer
 * @param {number} [options.runs=1]
 * @param {(run: number) => {chains: string[], length: number, hold: number[],
 *   hotspots?: Record<string, number[]>}} [options.job]
 *   re-read the job for this run, so a contig carrying a length RANGE draws a
 *   different length each time. Omitted means the fixed `chains` are used for
 *   every run, which is right whenever nothing is ranged.
 * @param {number} [options.cycles=5]
 * @param {number} [options.length=75] the designed chain's length
 * @param {number} [options.beam=1] how many sequences each cycle tries
 * @param {Iterable<number>} [options.hold] positions within the designed chain
 *   to keep as they are - a framework, a motif - every cycle
 * @param {Record<string, number[]>} [options.hotspots] per chain, residues the
 *   design must touch. A cycle that does not is rejected with the distance.
 * @param {number} [options.hotspotsNeeded] how many of them must be touched
 * @param {(record: object) => void} options.report called once per cycle
 * @param {(text: string) => void} [options.status]
 * @param {(fraction: number) => void} [options.progress]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{best: object|null, cycles: object[]}>}
 */
export async function hunt(options) {
  const {
    chains, chainKinds, ligandCodes, modifications, chainIndex = 0, folders = ["af3"],
    designer = "auto", contents = {},
    runs = 1, cycles = 5, length = 75, beam = 1, report,
    status = () => {}, progress = () => {}, signal,
  } = options;

  if (typeof report !== "function") throw new TypeError("hunt needs a report callback");
  if (!Array.isArray(chains) || chains.length === 0) throw new TypeError("hunt needs at least one chain");
  if (folders.length === 0) throw new Error("hunt needs at least one folding model");
  for (const name of folders) {
    if (FOLDERS[name] === undefined) {
      throw new Error(`no folder ${name} (have ${Object.keys(FOLDERS).join(", ")})`);
    }
  }

  // 🔴 THE SEQUENCE IS CHECKED BEFORE ANY WEIGHTS ARE FETCHED. A hunt is many
  // folds, and finding out after a 300 MB download that a chain has a letter
  // the model has no token for is the worst moment to find out.
  //
  // 🔴 PROTEIN CHAINS ONLY. af3SequenceProblem reads the protein alphabet, so a
  // DNA chain's A, T, G and C would be judged against it - three of the four are
  // legal amino acids and T is not, which makes it fail for the wrong reason and
  // pass for worse ones.
  for (const [index, chain] of chains.entries()) {
    if (index === chainIndex || chain.length === 0) continue;
    if ((chainKinds?.[index] ?? "protein") !== "protein") continue;
    const problem = af3SequenceProblem(chain);
    if (problem !== null) throw new Error(`chain ${index + 1}: ${problem}`);
  }

  const chosen = designer === "auto" ? chooseDesigner(contents).name : designer;

  // 🔴 TWO DIFFERENT `onProgress` CONTRACTS, SAME NAME. `foldAf3` calls its one
  // with a FRACTION; the weight loaders call theirs with an OBJECT
  // (`{loadedBytes, totalBytes, ...}`, see HttpTensorStore#reportProgress).
  // Treating the object as a number put NaN into the page's progress element,
  // which threw where a bar would simply have looked wrong - found by loading
  // the page, not by reading either signature.
  const fractionOf = (value) => {
    if (typeof value === "number") return value;
    const total = value?.totalBytes || value?.totalTensors || 0;
    const loaded = value?.loadedBytes ?? value?.loadedTensors ?? 0;
    return total > 0 ? loaded / total : 0;
  };
  // ...and nothing non-finite ever reaches the caller, whatever a loader does.
  const toBar = (fraction) => {
    if (Number.isFinite(fraction)) progress(Math.min(1, Math.max(0, fraction)));
  };

  // 🔴 EVERY FOLDER IN THE ROTATION IS LOADED BEFORE THE FIRST CYCLE, not on the
  // cycle that first needs it. Loading mid-run would put a 300 MB download in
  // the middle of a progress bar that is already counting cycles, and the
  // rotation visits all of them anyway.
  status(folders.length === 1 ? "loading the folding model"
    : `loading ${folders.length} folding models`);
  const device = await getDevice(signal);
  const loaded = new Map();
  for (const [index, name] of folders.entries()) {
    const share = (step) => 0.05 * ((index + step) / folders.length);
    loaded.set(name, await loadAf3Weights(
      (value) => toBar(share(fractionOf(value))), FOLDERS[name]));
  }

  status("loading the designer");
  const { model: mpnn } = await loadDesigner({
    name: chosen, signal,
    onProgress: (value) => toBar(0.05 + 0.05 * fractionOf(value)),
  });

  // The bar spans the WHOLE hunt, so a cycle's own fraction is a fraction of
  // one cell of it. Weight loading took the first tenth.
  // 🔴 THE BEAM MULTIPLIES THE WORK, so the bar has to count the folds and not
  // the cycles: at four sequences a cycle a bar counting cycles sits still for
  // four folds and then jumps.
  const total = runs * (1 + cycles * beam);
  const spanOf = (done, fraction) => 0.1 + 0.9 * ((done + fractionOf(fraction)) / total);

  const all = [];
  let best = null;
  // Which folder the cycle currently being yielded was folded by. The loop
  // calls `fold` and then yields, so this is set before the record exists.
  let foldedBy = folders[0];

  for (let run = 0; run < runs; run += 1) {
    // 🔴 THE JOB IS RE-READ PER RUN, because a contig may carry a length RANGE
    // and each run draws its own length out of it - which is what makes `Runs`
    // the search over lengths that a range asks for. Without this the chains
    // are built once and every run folds the same size, so the range would be
    // honoured in the notation and ignored in the work. A caller that passes no
    // `job` gets the fixed chains it handed in, which is every caller with no
    // ranges to resolve.
    const drawn = options.job === undefined ? null : options.job(run);
    const runChains = drawn?.chains ?? chains;
    const runLength = drawn?.length ?? length;
    const runHold = drawn?.hold ?? options.hold;

    const stream = runDesign({
      chains: runChains.slice(), chainIndex, cycles, length: runLength, run, signal,
      temperature: options.temperature,
      omit: options.omit,
      hold: runHold,
      beam,

      // 🔴 THE CONTACT CHECK, WHICH IS HOW AN EPITOPE REACHES PROTEIN HUNTER AT
      // ALL. It cannot condition on hotspots the way RFD4 does, so the only way
      // to ask is to fold and then reject a cycle whose binder is somewhere
      // else. With no hotspots named it still refuses a binder that drifted off
      // the target entirely, which ipTM alone will not catch - the model can be
      // confident about a chain that is nowhere near.
      veto: ({ pdb }) => contactProblem(pdb, {
        chain: CHAIN_IDS[chainIndex],
        // a run's own, where its job carries them: a drawn length moves the residues after it
        ...((drawn?.hotspots ?? options.hotspots) === undefined ? {}
          : { residues: drawn?.hotspots ?? options.hotspots }),
        ...(options.hotspotsNeeded === undefined ? {} : { minimum: options.hotspotsNeeded }),
      }),
      percentX: options.percentX,
      alanineBias: options.alanineBias,
      seed: (options.seed ?? 0) + run,

      fold: async (sequence, context) => {
        const cycleStart = context.cycle === 0 ? 0 : 1 + (context.cycle - 1) * beam;
        const done = context.run * (1 + cycles * beam) + cycleStart + (context.draw ?? 0);
        // 🔴 WHICH MODEL FOLDS THIS CYCLE, AND WHY IT ROTATES BY CYCLE RATHER
        // THAN BY RUN. A design has to satisfy several models instead of
        // overfitting one, which only happens if the model CHANGES UNDER the
        // sequence being refined - rotating per run would give each run one
        // model and overfit each of them separately. Yehlin measured two or
        // three beating one and five or six doing worse than two
        // (docs/DESIGN.md 3C, her stack, not reproduced here).
        const folder = folders[(context.run * (cycles + 1) + context.cycle) % folders.length];
        foldedBy = folder;
        return foldAf3({
          sequence, weights: loaded.get(folder), device, signal,
          // A ligand and a nucleic chain change the FOLD as well as the
          // designer: the featuriser gives each ligand a chain of its own and
          // reads a nucleic chain one token per base.
          ...(chainKinds === undefined ? {} : { chainKinds }),
          ...(ligandCodes === undefined || ligandCodes.length === 0 ? {} : { ligandCodes }),
          // A modified residue is a CCD component swapped in at one position:
          // the fold fetches it and the token count does not change.
          ...(modifications === undefined || modifications.length === 0
            ? {} : { modifications }),
          seed: (options.seed ?? 0) + context.run,
          recycles: options.recycles,
          // 🔴 foldAf3 HAS NO DEFAULT FOR `calls` - it destructures it and the
          // caller owns it - so passing `undefined` folds anyway and writes
          // "Folding 16/undefined" into the status line. The sampler's own
          // preferred count is the right default here.
          calls: options.steps ?? AF3_COUNTS.diffusion.preferred,
          onStatus: (text) => status(
            `run ${context.run + 1}/${runs} · cycle ${context.cycle}/${cycles}`
            + (context.draws > 1 ? ` · sequence ${(context.draw ?? 0) + 1}/${context.draws}` : "")
            + `${folders.length > 1 ? ` · ${folder}` : ""} · ${text}`),
          onProgress: (fraction) => toBar(spanOf(done, fraction)),
        });
      },

      design: async (pdb, context) => {
        status(`run ${context.run + 1}/${runs} · cycle ${context.cycle}/${cycles} · designing`);
        return designChain(mpnn, {
          pdb,
          chain: context.chain,
          temperature: context.temperature,
          omit: context.omit,
          alanineBias: context.alanineBias,
          random: context.random,
          hold: context.hold,
        });
      },
    });

    for await (const record of stream) {
      // 🔴 WHICH MODEL FOLDED IT TRAVELS WITH THE CYCLE. With one folder it is
      // the same answer every time; with a rotation it is the difference
      // between a readable table and a meaningless one, so the field exists now
      // rather than being added when ensembles land.
      // 🔴 NOT SPREAD INTO A NEW OBJECT. The loop OWNS `best` and un-sets it on
      // the record it previously marked when a better cycle arrives, by
      // mutating that object - so a copy would freeze a `best: true` that has
      // since become false. The two fields this driver adds are written onto
      // the loop's own record.
      record.folder = foldedBy;
      record.designer = chosen;
      all.push(record);
      report(record);
    }
  }

  // 🔴 AND THE WINNER IS THE LOOP'S, NOT A MAX OVER SCORES. It refuses to let
  // cycle 0 win - that sequence is the mostly-X draw the run STARTED from, so
  // naming it best would be reporting the question as the answer - and refuses
  // a cycle over the alanine ceiling. Re-deriving the best by score here would
  // quietly undo both rules, which is what the first version of this file did.
  for (const record of all) {
    if (record.best === true && (best === null || record.score > best.score)) best = record;
  }

  toBar(1);
  status(best === null ? "no cycle completed" : `best ${best.score.toFixed(3)}`);
  return { best, cycles: all };
}

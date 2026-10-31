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
import { chooseDesigner, DESIGNERS, NUCLEIC_KINDS } from "../cpu/design/designers.js";
import { contactProblem, hotspotPocket } from "../cpu/design/contact.js";
import { CONSTRAINT_FAMILIES } from "../shared/af3/featurise/constraints.js";
import { designedDrift } from "../cpu/design/align.js";
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
 * @param {{within: number, tries: number, from?: number}} [options.accept] hold each cycle
 *   to the previous one's fold: see runDesign. Its `tries` is the budget `beam` would be.
 * @param {{level: number, from?: number}} [options.warm] start each cycle's diffusion from the
 *   previous cycle's structure, `level` angstroms of noise down the schedule, from cycle `from` (2:
 *   cycle 1 folds its first design, which should not be held to cycle 0's mostly-X guess). See
 *   shared/af3/structure/warm-start.js.
 * @param {Iterable<number>} [options.hold] positions within the designed chain
 *   to keep as they are - a framework, a motif - every cycle
 * @param {Record<string, number[]>} [options.hotspots] per chain, residues the
 *   design must touch. A folder trained on pocket constraints (boltz2) is handed
 *   them as one, a prior; any other folder's cycle that misses them is rejected
 *   with the distance, unless `steer` aims it.
 * @param {{weight: number, cutoff?: number}} [options.steer] pull the designed chain
 *   toward the hotspots during every fold's diffusion (cpu/af3/diffusion/steering.js),
 *   which makes them a prior for every folder rather than a filter
 * @param {number} [options.hotspotsNeeded] how many of them must be touched
 * @param {(sequence: string) => object} [options.alignmentOf] each fold's alignment, from the complex's
 *   sequence as foldAf3 takes it (chains joined by ":") - the binder changes every cycle, so a kept
 *   chain's MSA is laid into a fresh alignment per fold (shared/input/chains.js mergeJobAlignments)
 * @param {{text: string, chainId: string, chain: number}[]} [options.templates] structures for
 *   the KEPT chains, as foldAf3 takes them, handed to every fold - a target folded from its
 *   sequence alone is a target nobody designed against
 * @param {{atoms: (length: number) => {token: number, atom: string, xyz: number[]}[], until?: number}} [options.motif]
 *   atoms held in every fold (cpu/af3/diffusion/inpaint.js, foldAf3's `motif`): a function of the run's
 *   designed-chain LENGTH, because a token after that chain moves with it - a catalytic site's tips, a fixed
 *   ligand's conformer, a given duplex. `until` releases the hold below that noise level (angstroms)
 * @param {number} [options.foldTries=1] a fold that leaves the designed chain touching nothing is folded again at the
 *   next seed, up to this many times, so MPNN designs from a docked backbone rather than a binder floating free
 *   (task-bench's `contactTries`; the DNA recipe's 5)
 * @param {boolean} [options.stick] a cycle that will not dock hands the NEXT design the run's last docked fold instead
 *   of its own, marked `stuck` - MPNN redesigning an undocked binder writes a soluble protein that never comes back.
 *   A stuck cycle's structure is not its sequence's, so a caller must never choose it (cpu/design/protocol.js)
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

  // 🔴 WHAT THE DESIGNED CHAIN IS DECIDES WHO CAN DESIGN IT: a DNA or RNA chain has NA-MPNN alone,
  // and an explicit pick of another family is refused here, before any weights are fetched.
  const kind = chainKinds?.[chainIndex] ?? "protein";
  const chosen = designer === "auto" ? chooseDesigner({ ...contents, designs: kind }).name : designer;
  if (NUCLEIC_KINDS.has(kind) && chosen !== "na") {
    throw new Error(`${DESIGNERS[chosen]?.label ?? chosen} cannot design a ${kind.toUpperCase()} chain;`
      + " only NA-MPNN can");
  }

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
  // ...and under `accept` the most a cycle can draw is its tries (it usually stops sooner)
  const perCycle = options.accept === undefined ? beam : options.accept.tries;
  const total = runs * (1 + cycles * perCycle);
  const spanOf = (done, fraction) => 0.1 + 0.9 * ((done + fractionOf(fraction)) / total);

  const all = [];
  let best = null;
  // Which folder the cycle currently being yielded was folded by. The loop
  // calls `fold` and then yields, so this is set before the record exists.
  let foldedBy = folders[0];
  // Which folder folds a run's cycle - the fold and its veto must agree on it
  const folderOf = (run, cycle) => folders[(run * (cycles + 1) + cycle) % folders.length];
  const readsPocket = (folder) => CONSTRAINT_FAMILIES[FOLDERS[folder]] !== undefined;

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
    const runHotspots = drawn?.hotspots ?? options.hotspots;
    // 🔴 THE HOTSPOTS AS A POCKET, for the folders that read one (cpu/design/contact.js hotspotPocket);
    // the rest of a rotation gets the check alone, as ProteinHunter's other models do
    const pocket = hotspotPocket(runHotspots, chainIndex, CHAIN_IDS);
    // 🔴 ...AND STEERED TOWARD BY ANY FOLDER, when asked: the same chains and residues, pulled on
    // during the diffusion instead of read by the network
    const steering = options.steer !== undefined && pocket.length > 0;
    // the run's last fold that touched its target, for `stick`
    let lastDocked = null;

    const stream = runDesign({
      chains: runChains.slice(), chainIndex, cycles, length: runLength, run, signal,
      temperature: options.temperature,
      omit: options.omit,
      hold: runHold,
      beam,
      kind,
      // 🔴 HOW FAR A CYCLE'S DESIGNED CHAIN MOVED, IN THE TARGET'S FRAME (designedDrift): fitted on
      // everything else, so a binder sliding round its target counts - superposed on itself it read
      // 1-2 A while the viewer showed it wandering. A monomer is measured on itself.
      drift: (previous, folded) => designedDrift(previous, folded, CHAIN_IDS[chainIndex]),
      ...(options.accept === undefined ? {} : { accept: options.accept }),

      // 🔴 THE CONTACT CHECK, WHICH IS HOW AN EPITOPE REACHES PROTEIN HUNTER AT
      // ALL. It cannot condition on hotspots the way RFD4 does, so the only way
      // to ask is to fold and then reject a cycle whose binder is somewhere
      // else. With no hotspots named it still refuses a binder that drifted off
      // the target entirely, which ipTM alone will not catch - the model can be
      // confident about a chain that is nowhere near.
      //
      // 🔴 AND FOR A FOLDER THAT TOOK THE HOTSPOTS AS A POCKET THEY ARE A PRIOR, NOT A FILTER
      // (2026-10-28): the fold was asked to put the binder there, so its cycle is not rejected for
      // missing one - only for touching nothing at all. A folder that cannot read a pocket keeps
      // the hotspot filter, which is the only way it is aimed.
      veto: ({ pdb, cycle }) => {
        const prior = pocket.length > 0 && (steering || readsPocket(folderOf(run, cycle)));
        return contactProblem(pdb, {
          chain: CHAIN_IDS[chainIndex],
          // a run's own, where its job carries them: a drawn length moves the residues after it
          ...(runHotspots === undefined || prior ? {} : { residues: runHotspots }),
          ...(options.hotspotsNeeded === undefined || prior ? {} : { minimum: options.hotspotsNeeded }),
        });
      },
      percentX: options.percentX,
      alanineBias: options.alanineBias,
      seed: (options.seed ?? 0) + run,

      fold: async (sequence, context) => {
        const cycleStart = context.cycle === 0 ? 0 : 1 + (context.cycle - 1) * perCycle;
        const done = context.run * (1 + cycles * perCycle) + cycleStart + (context.draw ?? 0);
        // 🔴 WHICH MODEL FOLDS THIS CYCLE, AND WHY IT ROTATES BY CYCLE RATHER
        // THAN BY RUN. A design has to satisfy several models instead of
        // overfitting one, which only happens if the model CHANGES UNDER the
        // sequence being refined - rotating per run would give each run one
        // model and overfit each of them separately. Yehlin measured two or
        // three beating one and five or six doing worse than two
        // (docs/DESIGN.md 3C, her stack, not reproduced here).
        const folder = folderOf(context.run, context.cycle);
        foldedBy = folder;
        // 🔴 HISTORY IN THE FOLD ITSELF: the diffusion joins part-way down, from the last cycle's
        // structure, instead of from noise (shared/af3/structure/warm-start.js)
        const warm = options.warm !== undefined && context.previous !== undefined
          && context.cycle >= (options.warm.from ?? 2);
        const foldAt = (seed) => foldAf3({
          sequence, weights: loaded.get(folder), device, signal,
          // 🔴 THE TARGET AS ITS OWN TEMPLATE, as the refold (validate-design.js) and RFD4-Proteina's
          // AF3 benchmark both fold it (2026-10-29): designing against a target AF3 refolds elsewhere
          // optimises a complex nobody will score
          ...(options.templates === undefined ? {} : { templates: options.templates }),
          // ...or its own MSA, which is how the original Protein Hunter gave the target (cycle by cycle,
          // since the alignment's query row carries the binder this cycle designed)
          ...(options.alignmentOf === undefined ? {} : { alignment: options.alignmentOf(sequence) }),
          ...(warm ? { start: { pdb: context.previous, level: options.warm.level } } : {}),
          // A ligand and a nucleic chain change the FOLD as well as the
          // designer: the featuriser gives each ligand a chain of its own and
          // reads a nucleic chain one token per base.
          ...(chainKinds === undefined ? {} : { chainKinds }),
          ...(ligandCodes === undefined || ligandCodes.length === 0 ? {} : { ligandCodes }),
          // A modified residue is a CCD component swapped in at one position:
          // the fold fetches it and the token count does not change.
          ...(modifications === undefined || modifications.length === 0
            ? {} : { modifications }),
          ...(pocket.length > 0 && readsPocket(folder) ? { constraints: pocket } : {}),
          ...(steering ? { steer: { binder: chainIndex, contacts: pocket[0].contacts, ...options.steer } } : {}),
          ...(options.motif === undefined ? {} : { motif: { atoms: options.motif.atoms(runLength),
            ...(options.motif.until === undefined ? {} : { until: options.motif.until }) } }),
          seed,
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
        // 🔴 `foldTries` AND `stick` ARE OFF UNLESS ASKED, so a plain hunt folds exactly as before: one fold a cycle
        // at the run's seed. A retry's seed is a fixed offset from it, so a hunt is still named by its seed.
        const tries = options.foldTries ?? 1;
        let folded;
        let docked = false;
        for (let k = 0; k < tries; k += 1) {
          folded = await foldAt((options.seed ?? 0) + context.run + 7919 * k);
          if (tries === 1 && !options.stick) return folded;
          if (contactProblem(folded.pdb, { chain: CHAIN_IDS[chainIndex] }) === null) { docked = true; break; }
        }
        if (docked) lastDocked = folded;
        else if (options.stick && lastDocked !== null) return { ...lastDocked, stuck: true };
        return folded;
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
          kind: context.kind,
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

/**
 * search.js
 * ---------
 * The search around a Protein Hunter hunt that beat RFD4-Proteina on its benchmarks: restarts that stop early on a
 * confident cycle, the most promising cycles refolded at the evaluation's settings as a check, and a polish of the
 * nearest misses - the loop `tools/ppi-bench/bench.js` and `task-bench.mjs` ran around `hunt()`, here where the
 * page can run it. The recipe is data (cpu/design/protocol.js); this is the loop.
 *
 * 🔴 IT SUBMITS ONE DESIGN, as RFD4-Proteina's Search pipeline does: everything before that - every restart, every
 * check fold, every polish draw - is search, and the answer is the first design whose check passed the paper's own
 * criterion, else the nearest miss. That is the like-for-like comparison with RFD4 + Search, not with one-shot.
 *
 * 🔴 THE SEEDS ARE bench.js's (a restart at `seed + 101 * restart`, a check at `7 * seed + cycle + 1000 * restart`),
 * so a PPI search here IS bench.js's NC recipe and reproduces its rows at the same seeds.
 *
 * 🔴 THE CHECK FOLDS FREE: no template steering, no motif, no pocket - single sequence plus the kept chains'
 * templates, 200 steps, 5 recycles, as the benchmark's evaluation folds. Only the design folds are guided.
 */
import { CHAIN_IDS, withChain } from "../webgpu/design/hunter-loop.js";
import { contactProblem } from "../cpu/design/contact.js";
import { alanineFraction } from "../cpu/design/sample-sequence.js";
import { confident, footprint, interfaceTerms, verdict } from "../cpu/design/protocol.js";
import { motifMatch, motifTerms } from "../cpu/design/motif.js";
import { uniformFrom } from "../shared/af3/random.js";
import { designChain, loadDesigner } from "../cpu/design/mpnn-bridge.js";
import { chooseDesigner } from "../cpu/design/designers.js";

/** The check's settings: the benchmark's evaluation fold. */
export const CHECK_STEPS = 200;
export const CHECK_RECYCLES = 5;

/**
 * Run a protocol's search over a hunt.
 *
 * @param {object} options `hunt()`'s options (chains, chainKinds, ligandCodes, modifications, chainIndex, templates,
 *   hotspots, hold, motif, folders, designer, contents, length, job, seed, alignmentOf, warm, steer) plus:
 * @param {object} options.protocol a cpu/design/protocol.js recipe; its cycles, steps, recycles, temperature, omit,
 *   designer, foldTries and stick override the hunt's own
 * @param {{atoms: (length: number) => object[], motif: object[]}} [options.held] what a `guide` recipe holds in
 *   place (cpu/design/motif.js heldCoordinates) - or a run's job returns its own as `held`, where a drawn length
 *   moves it
 * @param {(record: object) => void} [options.report] every design cycle, with its `restart` and interface `terms`
 * @param {(check: object) => void} [options.onCheck] every check fold, as it finishes
 * @param {object} [deps] the folds and the designer, injectable so the loop tests without a GPU:
 *   `hunt`, `fold(sequence, {seed, steps, recycles, signal})`, `redesign(pdb, {temperature, omit, random, hold})`
 * @returns {Promise<{best: object|null, passed: boolean, checks: object[], records: object[], runs: number,
 *   polished: number, climbed: number}>}
 */
export async function search(options, deps = {}) {
  const { protocol, chainIndex = 0, report = () => {}, onCheck = () => {}, status = () => {},
    progress = () => {}, signal } = options;
  if (protocol === undefined || protocol === null) throw new TypeError("search needs a protocol (cpu/design/protocol.js)");
  const criterion = protocol.criterion;
  const chain = CHAIN_IDS[chainIndex];
  const seed = options.seed ?? 0;
  const hunt = deps.hunt ?? (await import("./hunt.js")).hunt;
  const fold = deps.fold ?? defaultFold(options);
  const redesign = deps.redesign ?? defaultRedesign(options);
  const restarts = protocol.restarts ?? 1;

  const footprintOk = (pdb) => protocol.minFootprint === undefined || footprint(pdb, chain) >= protocol.minFootprint;
  const looksDone = (e) => protocol.stop !== undefined && confident(e, protocol.stop, criterion);
  const promising = (e) => confident(e, protocol.check, criterion) && footprintOk(e.pdb);
  // AME: how far a candidate is from passing on the motif, which orders the climb and the fallback
  const motifScore = (e) => e.motifRmsd + (e.clashes > 0 ? 5 : 0) + Math.max(0, e.ipae - 1.5);
  if (protocol.guide && options.held === undefined && options.job === undefined) {
    throw new Error(`the ${protocol.tag} recipe holds coordinates, and none were given`);
  }
  if (criterion === "ame" && !protocol.probe) throw new Error("the AME recipe judges an unguided refold (`probe`)");
  // the terms and structure a design fold is judged by: AME's unguided probe where there is one, else its own
  const judged = (record) => record.folded?.examined
    ?? (record.folded?.confidence === undefined ? null : { terms: interfaceTerms(record.folded, chainIndex), pdb: record.pdb });

  // one candidate through the check: every seed must pass, and the first that fails ends it
  const checkAt = async (entry, from) => {
    let result;
    for (let k = 0; k < (protocol.check.seeds ?? 1); k += 1) {
      const refold = await fold(withChain(entry.chains, chainIndex, entry.sequence),
        { seed: from + 104729 * k, steps: CHECK_STEPS, recycles: CHECK_RECYCLES, signal });
      const terms = interfaceTerms(refold, chainIndex);
      const held = criterion === "ame" ? { motif: entry.motif, positions: entry.positions } : undefined;
      result = { ...terms, ...verdict(criterion, entry.pdb, refold.pdb, terms, chain, held), refoldPdb: refold.pdb, seeds: k + 1 };
      if (!result.pass) break;
    }
    const check = { restart: entry.restart, cycle: entry.cycle, sequence: entry.sequence, ...result };
    onCheck(check);
    return result;
  };

  const records = [];
  const eligible = [];
  const checked = [];
  let runs = 0;
  for (let restart = 0; restart < restarts; restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const combined = signal === undefined ? controller.signal : AbortSignal.any([signal, controller.signal]);
    const mine = [];
    let runChains = options.chains;
    let runHeld = options.held;
    const guide = () => ({ atoms: runHeld.atoms, ...(protocol.guideUntil === undefined ? {} : { until: protocol.guideUntil }) });
    // 🔴 THE PROBE (AME): every design fold's sequence folded again UNGUIDED at the design settings, and the motif
    // found in it unindexed - what the cycle is judged and checked by; MPNN still designs from the guided backbone
    const probe = async ({ sequence, context }) => {
      const free = await fold(sequence, { seed: seed + 101 * restart + 13 * context.cycle + 4999,
        steps: protocol.steps, recycles: protocol.recycles, signal: combined });
      const match = motifMatch(runHeld.motif, free.pdb, chain);
      const positions = match?.positions ?? runHeld.motif.map((m) => m.position);
      return { pdb: free.pdb, positions,
        terms: { ...interfaceTerms(free, chainIndex), ...motifTerms(runHeld.motif, positions, free.pdb, free.pdb, chain) } };
    };
    try {
      await hunt({
        ...options,
        runs: 1,
        cycles: protocol.cycles ?? options.cycles,
        steps: protocol.steps ?? options.steps,
        recycles: protocol.recycles ?? options.recycles,
        ...(protocol.temperature === undefined ? {} : { temperature: protocol.temperature }),
        ...(protocol.omit === undefined ? {} : { omit: protocol.omit }),
        ...(protocol.designer === undefined ? {} : { designer: protocol.designer }),
        ...(protocol.foldTries === undefined ? {} : { foldTries: protocol.foldTries }),
        ...(protocol.stick === undefined ? {} : { stick: protocol.stick }),
        seed: seed + 101 * restart,
        signal: combined,
        ...(protocol.guide && options.job === undefined ? { motif: guide() } : {}),
        ...(protocol.probe ? { examine: probe } : {}),
        // 🔴 A LENGTH RANGE DRAWS PER RESTART, as it drew per run: each restart is hunt's run 0, so the job is read
        // at the restart's index instead
        ...(options.job === undefined ? {} : {
          job: () => {
            const drawn = options.job(restart);
            runChains = drawn.chains ?? options.chains;
            runHeld = drawn.held ?? options.held;
            if (!protocol.guide) return drawn;
            if (runHeld === undefined) throw new Error(`the ${protocol.tag} recipe holds coordinates, and none were given`);
            return { ...drawn, motif: guide() };
          },
        }),
        status: (text) => status(`restart ${restart + 1}/${restarts} · ${text}`),
        progress: (fraction) => progress(Math.min(1, (restart + fraction) / restarts)),
        report: (record) => {
          const seen = judged(record);
          const terms = seen?.terms ?? null;
          const stuck = record.folded?.stuck === true;
          const row = { restart, cycle: record.cycle, score: record.score, vetoed: record.vetoed ?? null,
            alanine: record.alanine, stuck, ...(terms ?? {}) };
          records.push(row);
          report({ ...record, restart, terms, stuck });
          if (terms === null || stuck || record.cycle === 0 || (record.vetoed ?? null) !== null) return;
          if (record.alanine !== null && record.alanine > 0.2) return;
          const entry = { ...terms, restart, cycle: record.cycle, sequence: record.sequence, pdb: seen.pdb,
            score: record.score, chains: runChains,
            ...(criterion === "ame" ? { positions: seen.positions, motif: runHeld.motif } : {}) };
          eligible.push(entry);
          mine.push(entry);
          if (looksDone(entry)) controller.abort();
        },
      });
    } catch (error) {
      if (!controller.signal.aborted || signal?.aborted) throw error;
    }
    if (protocol.check !== undefined) {
      const order = criterion === "ame" ? (a, b) => motifScore(a) - motifScore(b) : (a, b) => a.ipae - b.ipae;
      for (const entry of mine.filter(promising).sort(order).slice(0, protocol.check.top)) {
        status(`restart ${restart + 1}/${restarts} · checking cycle ${entry.cycle}`);
        const result = await checkAt(entry, seed * 7 + entry.cycle + 1000 * restart);
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }

  // 🔴 THE CLIMB (AME), ONLY WHEN NOTHING PASSED: hill-climb on the graded quantity itself - from the `from` probe
  // structures nearest the motif, `gens` generations of `draws` MPNN sequences (the matched residues held), each
  // folded UNGUIDED and the motif found again; a generation's best replaces the current one when it is closer
  let climbed = 0;
  const climb = protocol.climb;
  if (climb !== undefined && !checked.some((c) => c.result.pass)) {
    const starts = eligible.filter((e) => e.motifRmsd !== undefined).sort((a, b) => motifScore(a) - motifScore(b))
      .slice(0, climb.from);
    for (const [n, start] of starts.entries()) {
      let current = start;
      for (let g = 0; g < climb.gens; g += 1) {
        let bestDraw = null;
        for (let k = 0; k < climb.draws; k += 1) {
          status(`climbing from ${n + 1}/${starts.length} · generation ${g + 1}/${climb.gens} · draw ${k + 1}/${climb.draws}`);
          const draw = seed + 70000 + 1000 * n + 37 * g + k;
          const designed = await redesign(current.pdb, { chain, temperature: climb.temperature ?? 0.1,
            omit: protocol.omit ?? "C", random: uniformFrom(draw), hold: current.positions });
          const refold = await fold(withChain(current.chains, chainIndex, designed.sequence),
            { seed: draw, steps: protocol.steps, recycles: protocol.recycles, signal });
          climbed += 1;
          const match = motifMatch(current.motif, refold.pdb, chain);
          const here = motifTerms(current.motif, current.positions, refold.pdb, refold.pdb, chain);
          const at = match !== null && match.rmsd < here.motifRmsd ? match.positions : current.positions;
          const e = { ...interfaceTerms(refold, chainIndex), ...motifTerms(current.motif, at, refold.pdb, refold.pdb, chain),
            restart: -2, cycle: 10 * n + g, sequence: designed.sequence, pdb: refold.pdb, positions: [...at],
            chains: current.chains, motif: current.motif };
          records.push({ restart: -2, cycle: e.cycle, climb: k, ipae: e.ipae, motifRmsd: e.motifRmsd, clashes: e.clashes });
          if (alanineFraction(designed.sequence) > 0.2 || contactProblem(refold.pdb, { chain }) !== null) continue;
          eligible.push(e);
          if (bestDraw === null || motifScore(e) < motifScore(bestDraw)) bestDraw = e;
        }
        if (bestDraw !== null && motifScore(bestDraw) < motifScore(current)) current = bestDraw;
      }
    }
    const climbs = eligible.filter((e) => e.restart === -2 && promising(e)).sort((a, b) => motifScore(a) - motifScore(b));
    for (const entry of climbs.slice(0, climb.top ?? 3)) {
      const result = await checkAt(entry, seed * 7 + 555 + checked.length);
      checked.push({ entry, result });
      if (result.pass) break;
    }
  }

  // 🔴 POLISH ONLY WHEN NOTHING PASSED: redesign around the NEAREST checked misses instead of from scratch - `draws`
  // MPNN sequences on each one's design backbone, folded at the design settings, the best `top` checked in turn
  let polished = 0;
  const polish = protocol.polish;
  if (polish !== undefined && checked.length > 0 && !checked.some((c) => c.result.pass)) {
    const near = [...checked].sort((a, b) => a.result.margin - b.result.margin).slice(0, polish.from);
    const pool = [];
    for (const [n, c] of near.entries()) {
      for (let k = 0; k < polish.draws; k += 1) {
        status(`polishing miss ${n + 1}/${near.length} · draw ${k + 1}/${polish.draws}`);
        const designed = await redesign(c.entry.pdb, { chain, temperature: polish.temperature ?? 0.1,
          omit: protocol.omit ?? "C", random: uniformFrom(seed * 31 + 50000 + 97 * n + k), hold: options.hold });
        const refold = await fold(withChain(c.entry.chains, chainIndex, designed.sequence),
          { seed: seed + 50000 + 97 * n + k, steps: protocol.steps, recycles: protocol.recycles, signal });
        polished += 1;
        const terms = interfaceTerms(refold, chainIndex);
        records.push({ restart: -1, cycle: n, polish: k, ...terms });
        if (alanineFraction(designed.sequence) > 0.2) continue;
        if (contactProblem(refold.pdb, { chain }) !== null || !footprintOk(refold.pdb)) continue;
        pool.push({ ...terms, restart: -1, cycle: n, sequence: designed.sequence, pdb: refold.pdb, chains: c.entry.chains });
      }
    }
    for (const entry of pool.filter(promising).sort((a, b) => a.ipae - b.ipae).slice(0, polish.top)) {
      const result = await checkAt(entry, seed * 7 + 777 + checked.length);
      checked.push({ entry, result });
      eligible.push(entry);
      if (result.pass) break;
    }
  }

  // 🔴 THE ANSWER: a design whose check passed; else the checked one nearest passing; else, by the design folds' own
  // terms, the lowest interface PAE among confident cycles, else the most confident
  const passedCheck = checked.find((c) => c.result.pass) ?? null;
  const nearest = checked.length === 0 ? null : checked.reduce((a, b) => (b.result.margin < a.result.margin ? b : a));
  const sure = eligible.filter((e) => confident(e, { confidence: 0.8, ipae: Infinity }, criterion));
  const chosen = passedCheck ?? nearest;
  // (AME's fallback is the nearest to passing on the motif, as task-bench's)
  const key = criterion === "ame" ? motifScore : (e) => e.ipae;
  const best = chosen !== null ? { ...chosen.entry, check: chosen.result }
    : sure.length > 0 ? sure.reduce((a, b) => (key(b) < key(a) ? b : a))
    : eligible.length > 0 ? eligible.reduce((a, b) => (key(b) < key(a) ? b : a)) : null;
  progress(1);
  status(passedCheck !== null ? `a design passed its check (restart ${passedCheck.entry.restart + 1}, cycle ${passedCheck.entry.cycle})`
    : best === null ? "no eligible cycle" : "no check passed: the nearest miss");
  return { best, passed: passedCheck !== null, checks: checked.map((c) => ({ restart: c.entry.restart, cycle: c.entry.cycle,
    ...c.result })), records, runs, polished, climbed };
}

/** The check's and the polish's fold: the hunt's first folder, the kept chains templated, nothing guided. */
function defaultFold(options) {
  let weights;
  return async (sequence, { seed, steps, recycles, signal }) => {
    const { foldAf3, loadAf3Weights } = await import("./af3-model.js");
    const { getDevice } = await import("./model.js");
    const { FOLDERS } = await import("./validate-design.js");
    weights ??= await loadAf3Weights(() => {}, FOLDERS[(options.folders ?? ["af3"])[0]]);
    return foldAf3({
      sequence, weights, device: await getDevice(signal), signal, seed, recycles, calls: steps,
      ...(options.templates === undefined ? {} : { templates: options.templates }),
      ...(options.chainKinds === undefined ? {} : { chainKinds: options.chainKinds }),
      ...(options.ligandCodes === undefined || options.ligandCodes.length === 0 ? {} : { ligandCodes: options.ligandCodes }),
      ...(options.modifications === undefined || options.modifications.length === 0 ? {} : { modifications: options.modifications }),
      onStatus: () => {}, onProgress: () => {},
    });
  };
}

/** The polish's designer: the protocol's, else the hunt's, resolved as hunt resolves "auto". */
function defaultRedesign(options) {
  let model;
  return async (pdb, { chain, temperature, omit, random, hold }) => {
    const asked = options.protocol.designer ?? options.designer ?? "auto";
    const name = asked === "auto" ? chooseDesigner({ ...(options.contents ?? {}), designs: "protein" }).name : asked;
    model ??= (await loadDesigner({ name })).model;
    return designChain(model, { pdb, chain, temperature, omit, random, hold, kind: "protein" });
  };
}

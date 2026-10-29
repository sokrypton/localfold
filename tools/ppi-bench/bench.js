// In-page: one Protein Hunter attempt on an RFD4-Proteina PPI benchmark target, then the benchmark's own
// AF3 check (single sequence, target templated, 5 recycles, one sample, a fresh seed) and its pass criteria:
// binder-chain pTM >= 0.80, min interface PAE <= 1.5 A, target-aligned binder backbone RMSD <= 2.5 A.
window.__bench = async (cfg) => {
  const { hunt } = await import("/web/hunt.js");
  const { foldAf3, loadAf3Weights } = await import("/web/af3-model.js");
  const { getDevice } = await import("/web/model.js");
  const { weightedRigidAlign } = await import("/cpu/af3/diffusion/rigid-align.js");
  const t = cfg.target;
  const templates = t.sequences.map((_, i) => ({ text: t.pdb, chainId: "BCDEFG"[i], chain: i + 1 }));
  const records = [];

  // the benchmark's two confidence terms, for any fold of [binder, ...target]
  const confidenceTerms = (fold) => {
    const asym = Array.from(fold.batch.asymId);
    const n = asym.length;
    const pae = fold.confidence.predictedAlignedError;
    const minPae = (a, b) => {
      let m = Infinity;
      for (let i = 0; i < n; i += 1) if (asym[i] === a) for (let j = 0; j < n; j += 1) if (asym[j] === b) m = Math.min(m, pae[i * n + j]);
      return m;
    };
    const coi = asym[0];
    const ipae = Math.min(...[...new Set(asym)].filter((c) => c !== coi).map((c) => (minPae(coi, c) + minPae(c, coi)) / 2));
    return { coi, ipae, binderPtm: fold.confidence.chainPtm[String(coi)] };
  };
  // target-backbone-aligned binder backbone RMSD between two complexes laid out alike
  const atomsOf = (pdb) => {
    const map = new Map();
    for (const line of pdb.split("\n")) {
      if (!line.startsWith("ATOM")) continue;
      const name = line.slice(12, 16).trim();
      if (!["N", "CA", "C", "O"].includes(name)) continue;
      map.set(`${line[21]}:${line.slice(22, 26).trim()}:${name}`,
        [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)]);
    }
    return map;
  };
  const binderRmsd = (designPdb, refoldPdb) => {
    const design = atomsOf(designPdb);
    const refold = atomsOf(refoldPdb);
    const keys = [...design.keys()].filter((k) => refold.has(k));
    const x = new Float32Array(keys.length * 3);
    const y = new Float32Array(keys.length * 3);
    const w = new Float32Array(keys.length);
    keys.forEach((k, i) => { x.set(refold.get(k), i * 3); y.set(design.get(k), i * 3); w[i] = k[0] === "A" ? 0 : 1; });
    const fitted = weightedRigidAlign(x, y, w, keys.length);
    let sum = 0;
    let count = 0;
    keys.forEach((k, i) => {
      if (k[0] !== "A") return;
      for (let a = 0; a < 3; a += 1) sum += (fitted[i * 3 + a] - y[i * 3 + a]) ** 2;
      count += 1;
    });
    return Math.sqrt(sum / count);
  };
  // 🔴 THE BENCHMARK'S CHECK, as one function: the final evaluation and (cfg.check) the pipeline's own
  // internal filter run exactly this, with different seeds
  const weights = await loadAf3Weights(() => {}, "af3");
  const assess = async (binder, designPdb, seed) => {
    const r = await foldAf3({
      sequence: [binder, ...t.sequences].join(":"), templates, weights, device: await getDevice(),
      seed, recycles: 5, calls: cfg.evalSteps ?? 200, onStatus: () => {}, onProgress: () => {},
    });
    const terms = confidenceTerms(r);
    const rmsd = binderRmsd(designPdb, r.pdb);
    return { ...terms, rmsd, pass: terms.binderPtm >= 0.8 && terms.ipae <= 1.5 && rmsd <= 2.5,
      chainPtm: r.confidence.chainPtm, iptm: r.confidence.iptm, ptm: r.confidence.ptm, refoldPdb: r.pdb };
  };

  const started = performance.now();
  const eligible = [];
  const checked = [];
  let checkSeconds = 0;
  // 🔴 RESTARTS AND AN EARLY STOP: a run ends the moment a cycle looks like a pass (cfg.stop), and an
  // attempt that has not found one restarts from a new seed, up to cfg.restarts runs - one design out
  const looksDone = (e) => cfg.stop !== undefined && e.binderPtm >= cfg.stop.ptm && e.ipae <= cfg.stop.ipae;
  let runs = 0;
  for (let restart = 0; restart < (cfg.restarts ?? 1); restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const mine = [];
    try {
      await hunt({
        chains: ["", ...t.sequences], chainIndex: 0, length: cfg.length, cycles: cfg.cycles,
        folders: cfg.folders ?? ["af3"], steps: cfg.steps, recycles: cfg.recycles, seed: cfg.seed + 101 * restart,
        templates: cfg.templateDesign === false ? undefined : templates,
        signal: controller.signal,
        ...(cfg.hunt ?? {}),
        report: (record) => {
          const terms = record.folded?.confidence === undefined ? {} : confidenceTerms(record.folded);
          records.push({ restart, cycle: record.cycle, score: record.score, iptm: record.iptm, vetoed: record.vetoed ?? null,
            alanine: record.alanine, ipae: terms.ipae, binderPtm: terms.binderPtm });
          // 🔴 GIVING UP ON A RUN THAT HAS NOT DOCKED (cfg.giveUp = {cycle, ipae}): a binder still this far
          // off its target by that cycle rarely comes back, and the restart is the cheaper chance
          if (cfg.giveUp !== undefined && record.cycle >= cfg.giveUp.cycle && terms.ipae > cfg.giveUp.ipae) {
            controller.abort();
          }
          if (record.cycle > 0 && record.alanine <= 0.2 && (record.vetoed ?? null) === null) {
            const entry = { ...terms, restart, cycle: record.cycle, sequence: record.sequence, pdb: record.pdb, score: record.score };
            eligible.push(entry);
            mine.push(entry);
            if (looksDone(entry)) controller.abort();
          }
        },
      });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    }
    // 🔴 THE INTERNAL CHECK (cfg.check = {top, ptm, ipae}): this run's most promising cycles go through
    // the benchmark's own check with a seed of the pipeline's, and the first that passes is the design
    if (cfg.check !== undefined) {
      const promising = mine.filter((e) => e.binderPtm >= cfg.check.ptm && e.ipae <= cfg.check.ipae)
        .sort((a, b) => a.ipae - b.ipae).slice(0, cfg.check.top);
      for (const entry of promising) {
        const t2 = performance.now();
        const result = await assess(entry.sequence, entry.pdb, cfg.seed * 7 + entry.cycle + 1000 * restart);
        checkSeconds += (performance.now() - t2) / 1000;
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }
  const huntSeconds = (performance.now() - started) / 1000;

  // 🔴 THE WINNER: a candidate that passed the internal check; else the checked one closest to passing;
  // else, by the design folds' own terms, lowest interface PAE at binder pTM >= 0.8, else highest pTM
  const margin = (c) => Math.max(c.result.ipae / 1.5, c.result.rmsd / 2.5, 0.8 / Math.max(c.result.binderPtm, 1e-3));
  const confident = eligible.filter((e) => e.binderPtm >= 0.8);
  const best = checked.length > 0 ? checked.reduce((a, b) => (margin(b) < margin(a) ? b : a)).entry
    : confident.length > 0 ? confident.reduce((a, b) => (b.ipae < a.ipae ? b : a))
    : eligible.length > 0 ? eligible.reduce((a, b) => (b.binderPtm > a.binderPtm ? b : a)) : null;
  const summary = { huntSeconds, checkSeconds, records, runs, checks: checked.map((c) => ({ restart: c.entry.restart,
    cycle: c.entry.cycle, binderPtm: c.result.binderPtm, ipae: c.result.ipae, rmsd: c.result.rmsd })) };
  if (best === null) return { ...summary, best: null };

  const t1 = performance.now();
  const final = await assess(best.sequence, best.pdb, cfg.evalSeed);
  const evalSeconds = (performance.now() - t1) / 1000;
  return { ...summary, evalSeconds, binder: best.sequence, bestCycle: best.cycle, bestRestart: best.restart,
    bestScore: best.score, chainPtm: final.chainPtm, coi: final.coi, ipae: final.ipae, rmsd: final.rmsd,
    designPdb: best.pdb, refoldPdb: final.refoldPdb, iptm: final.iptm, ptm: final.ptm };
};
1;

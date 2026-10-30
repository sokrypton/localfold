// Protein Hunter on RFD4-Proteina's DNA- and small-molecule-binder benchmarks (tasks.json, from prep-tasks.py),
// every fold on the native CUDA port - native-bench.mjs's loop with the target given by an AlphaFold 3 job (DNA
// by sequence, a ligand by SMILES or CCD code), MPNN that sees it (NA-MPNN for DNA, LigandMPNN for a ligand), and
// each task's own pass criteria (the paper's S8.2):
//   dna: min interface PAE <= 1.5 and binder backbone RMSD <= 8 A after aligning all DNA atoms, design to refold
//   sm:  protein-ligand ipTM >= 0.80, min interface PAE <= 1.5, protein backbone RMSD <= 1.5 A and ligand RMSD
//        <= 5 A after protein-backbone alignment
// The interface terms are the paper's oracle's: chain_pair_* averaged both ways, min (PAE) or mean (ipTM) over the
// binder's partners.
// usage: node task-bench.mjs <out.jsonl> <config.json> <target>:<attempt> ...
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash } from "node:crypto";

const run = promisify(execFile);
const HERE = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(HERE, "../..");
const { runDesign } = await import(`${ROOT}/webgpu/design/hunter-loop.js`);
const { designChain } = await import(`${ROOT}/cpu/design/mpnn-bridge.js`);
const { Weights } = await import(`${ROOT}/cpu/design/mpnn/weights.js`);
const { Model } = await import(`${ROOT}/cpu/design/mpnn/model.js`);
const { enableAcceleration } = await import(`${ROOT}/cpu/design/mpnn/accel.js`);
const { contactProblem } = await import(`${ROOT}/cpu/design/contact.js`);
const { weightedRigidAlign } = await import(`${ROOT}/cpu/af3/diffusion/rigid-align.js`);

const [out, configPath, ...jobs] = process.argv.slice(2);
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
const tasks = JSON.parse(fs.readFileSync(path.join(HERE, "tasks.json"), "utf8"));
const done = new Set(fs.existsSync(out)
  ? fs.readFileSync(out, "utf8").split("\n").filter(Boolean).map((l) => { const d = JSON.parse(l); return `${d.target}:${d.attempt}`; })
  : []);
const bytes = (file) => { const b = fs.readFileSync(file); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
if ((await enableAcceleration(bytes(`${ROOT}/web/vendor/mpnn/kernels.wasm`))) === null) throw new Error("MPNN's accelerator did not load");
const checkpoint = { dna: "na_mpnn_design.mpnn", sm: "ligandmpnn_v_32_020_25.mpnn" };
const mpnns = {};
const mpnnFor = (task) => (mpnns[task] ??= new Model(Weights.fromArrayBuffer(bytes(`${ROOT}/web/public/mpnn/${config.mpnn?.[task] ?? checkpoint[task]}`))));
const work = fs.mkdtempSync(path.join(os.tmpdir(), "task-native-"));

function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const seedOf = (text) => createHash("sha1").update(text).digest().readUInt32LE(0);
const LETTERS = "ABCDEFGHIJ";

/** One native fold of [binder, ...the task's chains] from an AF3 job; the structure and the paper's terms. */
async function fold(t, binder, { steps, recycles, seed }) {
  const file = path.join(work, `f${process.pid}-${Date.now()}.pdb`);
  const job = file.replace(".pdb", ".json");
  const sequences = [{ protein: { id: "A", sequence: binder } }, ...t.chains.map((c, i) => {
    const id = LETTERS[i + 1];
    if (c.kind === "dna") return { dna: { id, sequence: c.sequence } };
    return { ligand: c.ccd !== undefined ? { id, ccdCodes: [c.ccd] } : { id, smiles: c.smiles } };
  })];
  fs.writeFileSync(job, JSON.stringify({ dialect: "alphafold3", version: 2, name: "j", modelSeeds: [seed], sequences }));
  await run(`${ROOT}/cuda/af3/fold`, [file, `--job=${job}`, "--", `--steps=${steps}`, `--recycles=${recycles}`, `--seed=${seed}`],
    { maxBuffer: 1 << 26 });
  const pdb = fs.readFileSync(file, "utf8");
  const s = JSON.parse(fs.readFileSync(file.replace(".pdb", "_summary_confidences.json"), "utf8"));
  for (const f of fs.readdirSync(work)) if (f.startsWith(path.basename(file, ".pdb"))) fs.rmSync(path.join(work, f));
  const k = s.chain_pair_pae_min.length;
  const both = (m, j) => (m[0][j] + m[j][0]) / 2;
  let ipae = Infinity, iptm = 0;
  for (let j = 1; j < k; j += 1) { ipae = Math.min(ipae, both(s.chain_pair_pae_min, j)); iptm += both(s.chain_pair_iptm, j) / (k - 1); }
  const chainPairIptm = {};
  for (let i = 0; i < k; i += 1) for (let j = i + 1; j < k; j += 1) chainPairIptm[`${i + 1}|${j + 1}`] = Math.max(s.chain_pair_iptm[i][j], s.chain_pair_iptm[j][i]);
  return { pdb, binderPtm: s.chain_ptm[0], ipae, iptm, confidence: { iptm: s.iptm, ptm: s.ptm, meanPlddt: s.mean_plddt, chainPairIptm } };
}

/** One ESMFold2 fold (cuda/ef2, a fresh process - 1.2 s with its start-up, so no second resident server) of the same
 * job, its terms read as AF3's are: the binder's chain pTM, ipTM and interface PAE from the chain-level summary
 * cuda/ef2 writes since 2026-10-30. */
async function foldEf2(t, binder, { seed }) {
  const file = path.join(work, `e${process.pid}-${Date.now()}.pdb`);
  const job = file.replace(".pdb", ".json");
  const sequences = [{ protein: { id: "A", sequence: binder } }, ...t.chains.map((c, i) => {
    const id = LETTERS[i + 1];
    if (c.kind === "dna") return { dna: { id, sequence: c.sequence } };
    return { ligand: c.ccd !== undefined ? { id, ccdCodes: [c.ccd] } : { id, smiles: c.smiles } };
  })];
  fs.writeFileSync(job, JSON.stringify({ dialect: "alphafold3", version: 2, name: "j", modelSeeds: [seed], sequences }));
  await run(`${ROOT}/cuda/ef2/fold`, [file, `--job=${job}`, "--", `--seed=${seed}`], { maxBuffer: 1 << 26 });
  const pdb = fs.readFileSync(file, "utf8");
  const s = JSON.parse(fs.readFileSync(file.replace(".pdb", "_summary_confidences.json"), "utf8"));
  for (const f of fs.readdirSync(work)) if (f.startsWith(path.basename(file, ".pdb"))) fs.rmSync(path.join(work, f));
  if (s.chain_pair_iptm === undefined) throw new Error("cuda/ef2 wrote no chain-level summary - rebuild it");
  const k = s.chain_pair_pae_min.length;
  const both = (m, j) => (m[0][j] + m[j][0]) / 2;
  let ipae = Infinity, iptm = 0;
  for (let j = 1; j < k; j += 1) { ipae = Math.min(ipae, both(s.chain_pair_pae_min, j)); iptm += both(s.chain_pair_iptm, j) / (k - 1); }
  const chainPairIptm = {};
  for (let i = 0; i < k; i += 1) for (let j = i + 1; j < k; j += 1) chainPairIptm[`${i + 1}|${j + 1}`] = Math.max(s.chain_pair_iptm[i][j], s.chain_pair_iptm[j][i]);
  return { pdb, binderPtm: s.chain_ptm[0], ipae, iptm, confidence: { iptm: s.iptm, ptm: s.ptm, meanPlddt: s.mean_plddt, chainPairIptm } };
}

/** Atoms by chain:residue:name: the binder's backbone, and every atom of the other chains. */
function atoms(pdb) {
  const binder = new Map(), rest = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const name = line.slice(12, 16).trim(), chain = line[21];
    const key = `${chain}:${line.slice(22, 26).trim()}:${name}`;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (chain === "A") { if (["N", "CA", "C", "O"].includes(name)) binder.set(key, xyz); } else rest.set(key, xyz);
  }
  return { binder, rest };
}
/** RMSD of `measure` after fitting `fit` (design onto refold; both maps keyed alike). */
function fittedRmsd(fitD, fitR, measureD, measureR) {
  const fk = [...fitD.keys()].filter((k) => fitR.has(k)), mk = [...measureD.keys()].filter((k) => measureR.has(k));
  const keys = [...fk, ...mk], n = keys.length;
  const x = new Float32Array(n * 3), y = new Float32Array(n * 3), w = new Float32Array(n);
  keys.forEach((k, i) => { const fit = i < fk.length; x.set((fit ? fitR : measureR).get(k), i * 3); y.set((fit ? fitD : measureD).get(k), i * 3); w[i] = fit ? 1 : 0; });
  const fitted = weightedRigidAlign(x, y, w, n);
  let sum = 0;
  for (let i = fk.length; i < n; i += 1) for (let a = 0; a < 3; a += 1) sum += (fitted[i * 3 + a] - y[i * 3 + a]) ** 2;
  return Math.sqrt(sum / mk.length);
}
function verdict(task, designPdb, r) {
  const d = atoms(designPdb), f = atoms(r.pdb);
  if (task === "dna") {
    const rmsd = fittedRmsd(d.rest, f.rest, d.binder, f.binder);
    return { rmsd, pass: r.ipae <= 1.5 && rmsd <= 8 };
  }
  const rmsd = fittedRmsd(d.binder, f.binder, d.binder, f.binder);
  const ligandRmsd = fittedRmsd(d.binder, f.binder, d.rest, f.rest);
  return { rmsd, ligandRmsd, pass: r.iptm >= 0.8 && r.ipae <= 1.5 && rmsd <= 1.5 && ligandRmsd <= 5 };
}

/** `minFootprint`: a design must touch at least that many residues of the other chains (heavy atoms within 4.5 A) -
 * on a 10-bp duplex a binder that grips one end or three bases moves 15-40 A on refolding, one that spans both
 * strands holds its pose. */
function footprintProblem(pdb) {
  if (config.minFootprint === undefined) return null;
  const binder = [], other = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(76, 78).trim() === "H") continue;
    const xyz = [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)];
    if (line[21] === "A") binder.push(xyz);
    else { const k = line.slice(21, 26); if (!other.has(k)) other.set(k, []); other.get(k).push(xyz); }
  }
  let touched = 0;
  for (const atoms of other.values()) {
    if (atoms.some((a) => binder.some((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 < 4.5 ** 2))) touched += 1;
  }
  return touched >= config.minFootprint ? null : `footprint ${touched} residues, needs ${config.minFootprint}`;
}

async function attempt(name, index) {
  const t = tasks[name];
  const cfg = config;
  const rng = mulberry(seedOf(`${cfg.pair ?? cfg.tag}:${name}:${index}`));
  // `lengths`: one of a list (the paper's 150/200/250 for a small molecule); `length`: a range; else the file's range
  const [lo, hi] = cfg.length?.[t.task] ?? t.binder;
  const draw = rng();
  const length = cfg.lengths?.[t.task] !== undefined ? cfg.lengths[t.task][Math.floor(draw * cfg.lengths[t.task].length)]
    : lo + Math.floor(draw * (hi - lo + 1));
  const evalSeed = 1 + Math.floor(rng() * 2 ** 30);
  const seed = 1000 * index + 7;
  const mpnn = mpnnFor(t.task);
  const assess = async (binder, designPdb, foldSeed) => {
    const r = await fold(t, binder, { steps: 200, recycles: 5, seed: foldSeed });
    return { ...r, ...verdict(t.task, designPdb, r) };
  };
  // `check.seeds`: a candidate passes the internal check only if it passes at that many seeds (the first that
  // fails ends it) - a small molecule's interface PAE sits on the 1.5 line, and one passing seed is 64% likely to
  // mean a passing evaluation
  const checkAt = async (entry, s0) => {
    let result;
    for (let k = 0; k < (cfg.check?.seeds ?? 1); k += 1) {
      result = await assess(entry.sequence, entry.pdb, s0 + 104729 * k);
      result.seedsPassed = k + (result.pass ? 1 : 0);
      if (!result.pass) break;
    }
    return result;
  };
  // the design-time stop: confident about the terms the task is judged on
  const looksDone = (e) => cfg.stop !== undefined && e.ipae <= cfg.stop.ipae && (t.task === "dna" || e.iptm >= cfg.stop.iptm);
  // 🔴 THE FOOTPRINT SCREENS WHAT IS CHECKED, NOT WHAT THE LOOP DESIGNS FROM: as a per-cycle veto it failed most
  // docked folds, and a docked cycle was followed by a docked one 2 times in 68 against 92 in 290 without it
  const promisingOf = (e) => e.ipae <= cfg.check.ipae && (t.task === "dna" || e.iptm >= cfg.check.iptm) && footprintProblem(e.pdb) === null;
  const records = [], eligible = [], checked = [];
  let runs = 0, foldSeconds = 0;
  const started = Date.now();
  for (let restart = 0; restart < (cfg.restarts ?? 1); restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const mine = [];
    let lastDocked = null, undockedRun = 0;
    try {
      const stream = runDesign({
        chains: [""], chainIndex: 0, cycles: cfg.cycles, length, run: restart,
        seed: seed + 101 * restart, signal: controller.signal, kind: "protein", beam: cfg.beam ?? 1,
        ...(cfg.temperature === undefined ? {} : { temperature: cfg.temperature }),
        // `omit`: letters MPNN may never write, per task (a DNA binder without D/E is basic) - the loop's own default is "C"
        ...(cfg.omit?.[t.task] === undefined ? {} : { omit: cfg.omit[t.task] }),
        veto: ({ pdb }) => contactProblem(pdb, { chain: "A" }),
        fold: async (sequence, context) => {
          const t0 = Date.now();
          // `contactTries`: a fold that leaves the binder off the target is folded again with the next seed, up to
          // that many times - so MPNN designs from a docked backbone rather than from a binder floating free
          let r, docked = false;
          for (let k = 0; k < (cfg.contactTries ?? 1); k += 1) {
            const s = seed + 101 * restart + 13 * context.cycle + 7919 * k;
            // `designEngine`: "ef2" designs with ESMFold2 (the check and the evaluation stay AF3)
            // `ef2Cycles`: ESMFold2 for cycles before that one, AF3 after - ESMFold2 docks a DNA binder on every cycle
            // where AF3 docks ~5%, and AF3 then refines a docked start into one AF3 believes
            const ef2 = cfg.designEngine === "ef2" || (cfg.ef2Cycles !== undefined && context.cycle < cfg.ef2Cycles);
            r = ef2 ? await foldEf2(t, sequence, { seed: s })
              : await fold(t, sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: s });
            if (contactProblem(r.pdb, { chain: "A" }) === null) { docked = true; break; }
          }
          foldSeconds += (Date.now() - t0) / 1000;
          // `stick`: a cycle that will not dock hands the NEXT design the restart's last docked structure instead of
          // its own - MPNN redesigning an undocked binder writes a soluble protein that never comes back. The cycle
          // itself can never be chosen (its terms are infinite; its structure is not its sequence's)
          if (docked) lastDocked = r;
          else if (cfg.stick && lastDocked !== null) {
            return { pdb: lastDocked.pdb, confidence: lastDocked.confidence, terms: { binderPtm: 0, ipae: Infinity, iptm: 0, stuck: true } };
          }
          return { pdb: r.pdb, confidence: r.confidence, terms: { binderPtm: r.binderPtm, ipae: r.ipae, iptm: r.iptm } };
        },
        design: (pdb, context) => designChain(mpnn, { pdb, chain: context.chain, temperature: context.temperature,
          omit: context.omit, alanineBias: context.alanineBias, random: context.random, hold: context.hold, kind: context.kind,
          // `residueBias`: per task, MPNN logits added per letter (a DNA binder pushed toward K/R)
          ...(cfg.residueBias?.[t.task] === undefined ? {} : { residueBias: cfg.residueBias[t.task] }) }),
      });
      for await (const record of stream) {
        const terms = record.folded.terms;
        records.push({ restart, cycle: record.cycle, score: record.score, vetoed: record.vetoed ?? null,
          alanine: record.alanine, ...terms });
        // `giveUpStuck`: a restart whose last N design cycles all failed to dock is abandoned for a fresh one -
        // on DNA most restarts undock after cycle 0 and never come back, and each stuck cycle still costs its
        // `contactTries` folds
        if (record.cycle > 0) undockedRun = (record.vetoed ?? null) !== null || terms.stuck ? undockedRun + 1 : 0;
        if (cfg.giveUpStuck !== undefined && undockedRun >= cfg.giveUpStuck) controller.abort();
        // a restart that has not found the interface by `giveUp.cycle` is abandoned for a fresh one
        if (cfg.giveUp !== undefined && record.cycle >= cfg.giveUp.cycle && terms.ipae > cfg.giveUp.ipae) controller.abort();
        if (record.cycle > 0 && record.alanine <= 0.2 && (record.vetoed ?? null) === null) {
          const entry = { ...terms, restart, cycle: record.cycle, sequence: record.sequence, pdb: record.pdb, score: record.score };
          eligible.push(entry); mine.push(entry);
          if (looksDone(entry)) controller.abort();
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    }
    if (cfg.check !== undefined) {
      const promising = mine.filter(promisingOf).sort((a, b) => a.ipae - b.ipae).slice(0, cfg.check.top);
      for (const entry of promising) {
        const result = await checkAt(entry, seed * 7 + entry.cycle + 1000 * restart);
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }
  // `polish`: when no restart produced a passing check, redesign around the NEAREST misses instead of from scratch -
  // `draws` MPNN sequences on each of the `from` closest checked candidates' design backbones, each folded at the
  // design settings, and the best `top` of them (by design iPAE, under the check's thresholds) checked in turn
  const margin = (r) => Math.max(r.ipae / 1.5, t.task === "dna" ? r.rmsd / 8 : Math.max(r.rmsd / 1.5, r.ligandRmsd / 5, 0.8 / Math.max(r.iptm, 1e-3)));
  let polished = 0;
  if (cfg.polish !== undefined && !checked.some((c) => c.result.pass)) {
    const near = [...checked].sort((a, b) => margin(a.result) - margin(b.result)).slice(0, cfg.polish.from);
    const pool = [];
    for (const [n, c] of near.entries()) {
      for (let k = 0; k < cfg.polish.draws; k += 1) {
        const random = mulberry(seedOf(`polish:${name}:${index}:${n}:${k}`));
        const designed = designChain(mpnn, { pdb: c.entry.pdb, chain: "A", temperature: cfg.polish.temperature ?? 0.1,
          omit: cfg.omit?.[t.task] ?? "C", random, kind: "protein" });
        const r = await fold(t, designed.sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: seed + 50000 + 97 * n + k });
        polished += 1;
        records.push({ restart: -1, cycle: n, polish: k, ipae: r.ipae, iptm: r.iptm, binderPtm: r.binderPtm });
        const alanine = [...designed.sequence].filter((x) => x === "A").length / designed.sequence.length;
        if (alanine <= 0.2 && (contactProblem(r.pdb, { chain: "A" }) ?? footprintProblem(r.pdb)) === null) pool.push({ ipae: r.ipae, iptm: r.iptm, binderPtm: r.binderPtm, restart: -1, cycle: n, sequence: designed.sequence, pdb: r.pdb });
      }
    }
    for (const entry of pool.filter(promisingOf).sort((a, b) => a.ipae - b.ipae).slice(0, cfg.polish.top)) {
      const result = await checkAt(entry, seed * 7 + 777 + checked.length);
      checked.push({ entry, result });
      eligible.push(entry);
      if (result.pass) break;
    }
  }
  const chosen = checked.find((c) => c.result.pass) ?? null;
  const best = chosen !== null ? chosen.entry
    : eligible.length > 0 ? eligible.reduce((a, b) => (b.ipae < a.ipae ? b : a)) : null;
  const row = { target: name, task: t.task, attempt: index, length, config: cfg.tag, runs, polished, records, foldSeconds,
    checks: checked.map((c) => ({ restart: c.entry.restart, cycle: c.entry.cycle, ipae: c.result.ipae, iptm: c.result.iptm,
      rmsd: c.result.rmsd, ligandRmsd: c.result.ligandRmsd, pass: c.result.pass, seedsPassed: c.result.seedsPassed })) };
  if (best !== null) {
    // 🔴 THE EVALUATION IS ALWAYS THE NATIVE AF3 at the benchmark's settings, with its own seed
    const final = await assess(best.sequence, best.pdb, evalSeed);
    Object.assign(row, { binder: best.sequence, bestCycle: best.cycle, bestRestart: best.restart,
      binderPtm: final.binderPtm, ipae: final.ipae, iptm: final.iptm, rmsd: final.rmsd, ligandRmsd: final.ligandRmsd,
      pass: final.pass, designPdb: best.pdb, refoldPdb: final.pdb });
  } else {
    row.pass = false;
  }
  row.wall = (Date.now() - started) / 1000;
  return row;
}

// --rescore <rows.jsonl>: every submitted design evaluated again at its own evaluation seed (the attempt's seeded
// draw) - for rows whose evaluation ran on a server that ignored the job's seed (fixed 2026-10-30; see af3.cu)
async function rescore(rowsPath) {
  for (const line of fs.readFileSync(rowsPath, "utf8").split("\n").filter(Boolean)) {
    const d = JSON.parse(line);
    if (done.has(`${d.target}:${d.attempt}`) || d.error) continue;
    const t = tasks[d.target];
    const rng = mulberry(seedOf(`${config.pair ?? config.tag}:${d.target}:${d.attempt}`));
    rng();
    const evalSeed = 1 + Math.floor(rng() * 2 ** 30);
    const row = { target: d.target, task: d.task, attempt: d.attempt, config: d.config, rescored: true,
      served: { ipae: d.ipae, iptm: d.iptm, rmsd: d.rmsd, ligandRmsd: d.ligandRmsd, pass: d.pass } };
    if (d.binder) {
      const r = await fold(t, d.binder, { steps: 200, recycles: 5, seed: evalSeed });
      Object.assign(row, { binder: d.binder, binderPtm: r.binderPtm, ipae: r.ipae, iptm: r.iptm, ...verdict(t.task, d.designPdb, r) });
    } else {
      row.pass = false;
    }
    fs.appendFileSync(out, JSON.stringify(row) + "\n");
    console.log(d.target, d.attempt, d.pass, "->", row.pass, row.ipae?.toFixed?.(2), row.rmsd?.toFixed?.(2));
  }
}

try {
  if (jobs[0] === "--rescore") { await rescore(jobs[1]); jobs.length = 0; }
  for (const job of jobs) {
    if (done.has(job)) continue;
    const [name, index] = job.split(":");
    let row;
    try {
      row = await attempt(name, Number(index));
    } catch (error) {
      row = { target: name, attempt: Number(index), config: config.tag, error: String(error?.stack ?? error).slice(0, 2000) };
    }
    fs.appendFileSync(out, JSON.stringify(row) + "\n");
    console.log(name, index, row.pass, row.iptm?.toFixed?.(2), row.ipae?.toFixed?.(2), row.rmsd?.toFixed?.(2), row.ligandRmsd?.toFixed?.(2), row.wall?.toFixed?.(0), row.error?.slice(0, 160) ?? "");
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}

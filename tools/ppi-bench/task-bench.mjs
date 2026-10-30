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
  // the design-time stop: confident about the terms the task is judged on
  const looksDone = (e) => cfg.stop !== undefined && e.ipae <= cfg.stop.ipae && (t.task === "dna" || e.iptm >= cfg.stop.iptm);
  const promisingOf = (e) => e.ipae <= cfg.check.ipae && (t.task === "dna" || e.iptm >= cfg.check.iptm);
  const records = [], eligible = [], checked = [];
  let runs = 0, foldSeconds = 0;
  const started = Date.now();
  for (let restart = 0; restart < (cfg.restarts ?? 1); restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const mine = [];
    try {
      const stream = runDesign({
        chains: [""], chainIndex: 0, cycles: cfg.cycles, length, run: restart,
        seed: seed + 101 * restart, signal: controller.signal, kind: "protein", beam: cfg.beam ?? 1,
        // `omit`: letters MPNN may never write, per task (a DNA binder without D/E is basic) - the loop's own default is "C"
        ...(cfg.omit?.[t.task] === undefined ? {} : { omit: cfg.omit[t.task] }),
        veto: ({ pdb }) => contactProblem(pdb, { chain: "A" }),
        fold: async (sequence, context) => {
          const t0 = Date.now();
          // `contactTries`: a fold that leaves the binder off the target is folded again with the next seed, up to
          // that many times - so MPNN designs from a docked backbone rather than from a binder floating free
          let r;
          for (let k = 0; k < (cfg.contactTries ?? 1); k += 1) {
            r = await fold(t, sequence, { steps: cfg.steps, recycles: cfg.recycles, seed: seed + 101 * restart + 13 * context.cycle + 7919 * k });
            if (contactProblem(r.pdb, { chain: "A" }) === null) break;
          }
          foldSeconds += (Date.now() - t0) / 1000;
          return { pdb: r.pdb, confidence: r.confidence, terms: { binderPtm: r.binderPtm, ipae: r.ipae, iptm: r.iptm } };
        },
        design: (pdb, context) => designChain(mpnn, { pdb, chain: context.chain, temperature: context.temperature,
          omit: context.omit, alanineBias: context.alanineBias, random: context.random, hold: context.hold, kind: context.kind }),
      });
      for await (const record of stream) {
        const terms = record.folded.terms;
        records.push({ restart, cycle: record.cycle, score: record.score, vetoed: record.vetoed ?? null,
          alanine: record.alanine, ...terms });
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
        const result = await assess(entry.sequence, entry.pdb, seed * 7 + entry.cycle + 1000 * restart);
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }
  const chosen = checked.find((c) => c.result.pass) ?? null;
  const best = chosen !== null ? chosen.entry
    : eligible.length > 0 ? eligible.reduce((a, b) => (b.ipae < a.ipae ? b : a)) : null;
  const row = { target: name, task: t.task, attempt: index, length, config: cfg.tag, runs, records, foldSeconds,
    checks: checked.map((c) => ({ restart: c.entry.restart, cycle: c.entry.cycle, ipae: c.result.ipae, iptm: c.result.iptm,
      rmsd: c.result.rmsd, ligandRmsd: c.result.ligandRmsd, pass: c.result.pass })) };
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

try {
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

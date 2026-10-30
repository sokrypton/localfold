// Protein Hunter on the RFD4-Proteina PPI benchmark with every fold on the native CUDA port (cuda/af3/fold --serve):
// the page's own design loop (webgpu/design/hunter-loop.js runDesign), MPNN (cpu/design/mpnn-bridge.js) and contact
// check (cpu/design/contact.js) under Node, the folds - design, internal check and evaluation - handed to the resident
// server. bench.js is the same attempt in a browser; this one is ~8x the folds an hour.
// usage: node native-bench.mjs <out.jsonl> <config.json> <target>:<attempt> ...   (start `cuda/af3/fold --serve
//   [--model=<m>]` for every model the config names first; run several workers in parallel)
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
const targets = JSON.parse(fs.readFileSync(path.join(HERE, "targets.json"), "utf8"));
const done = new Set(fs.existsSync(out)
  ? fs.readFileSync(out, "utf8").split("\n").filter(Boolean).map((l) => { const d = JSON.parse(l); return `${d.target}:${d.attempt}`; })
  : []);
const bytes = (file) => { const b = fs.readFileSync(file); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
if ((await enableAcceleration(bytes(`${ROOT}/web/vendor/mpnn/kernels.wasm`))) === null) throw new Error("MPNN's accelerator did not load");
const mpnn = new Model(Weights.fromArrayBuffer(bytes(`${ROOT}/web/public/mpnn/solublempnn_v_48_020.mpnn`)));
const work = fs.mkdtempSync(path.join(os.tmpdir(), "ppi-native-"));

// seeded, as the page's hunt is: one stream per attempt
function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const seedOf = (text) => createHash("sha1").update(text).digest().readUInt32LE(0);

/** One native fold of [binder, ...target], the target templated: structure and the benchmark's terms. */
async function fold(t, templateFile, sequence, { model = "af3", steps, recycles, seed, pocket, msas }) {
  const file = path.join(work, `f${process.pid}-${Date.now()}.pdb`);
  const slots = t.sequences.map((_, i) => `${templateFile}:${"BCDEFG"[i]}@${i + 1}`).join(",");
  // 🔴 `msas`: each target chain's own alignment (and the binder its query alone) INSTEAD of the template - the
  // original Protein Hunter's input for the target
  const binderA3m = file.replace(".pdb", "-binder.a3m");
  if (msas !== undefined) fs.writeFileSync(binderA3m, `>A\n${sequence.split(":")[0]}\n`);
  const alignment = msas === undefined ? [] : [`--a3m=${[binderA3m, ...msas].join(",")}`];
  const templated = msas === undefined ? [`--template=${slots}`] : [];
  // 🔴 A POCKET GOES IN A JOB FILE - the native featuriser reads constraints there and nowhere else - beside
  // the same template flags (the binder is chain A, its target chains B, C, ...)
  let input = [`--sequence=${sequence}`];
  if (pocket !== undefined && pocket.length > 0) {
    const job = file.replace(".pdb", ".json");
    fs.writeFileSync(job, JSON.stringify({ dialect: "alphafold3", version: 2, name: "j", modelSeeds: [seed],
      sequences: sequence.split(":").map((s, i) => ({ protein: { id: "ABCDEFG"[i], sequence: s,
        ...(msas === undefined ? {} : { unpairedMsa: i === 0 ? `>A\n${s}\n` : fs.readFileSync(msas[i - 1], "utf8"), pairedMsa: "" }) } })),
      constraints: [{ pocket: { binder: "A", contacts: pocket, maxDistance: 6.0 } }] }));
    input = [`--job=${job}`];
  } else {
    input.push(...alignment);
  }
  await run(`${ROOT}/cuda/af3/fold`, [file, ...input, ...templated,
    ...(model === "af3" ? [] : [`--model=${model}`]), "--", `--steps=${steps}`, `--recycles=${recycles}`, `--seed=${seed}`],
    { maxBuffer: 1 << 26 });
  const pdb = fs.readFileSync(file, "utf8");
  const s = JSON.parse(fs.readFileSync(file.replace(".pdb", "_summary_confidences.json"), "utf8"));
  for (const f of fs.readdirSync(work)) if (f.startsWith(path.basename(file, ".pdb"))) fs.rmSync(path.join(work, f));
  const k = s.chain_pair_pae_min.length;
  const chainPairIptm = {};
  for (let i = 0; i < k; i += 1) for (let j = i + 1; j < k; j += 1) {
    chainPairIptm[`${i + 1}|${j + 1}`] = Math.max(s.chain_pair_iptm[i][j], s.chain_pair_iptm[j][i]);
  }
  let ipae = Infinity;
  for (let j = 1; j < k; j += 1) ipae = Math.min(ipae, (s.chain_pair_pae_min[0][j] + s.chain_pair_pae_min[j][0]) / 2);
  return { pdb, binderPtm: s.chain_ptm[0], ipae,
    confidence: { iptm: s.iptm, ptm: s.ptm, meanPlddt: s.mean_plddt, chainPairIptm } };
}

function backbone(pdb) {
  const map = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM")) continue;
    const name = line.slice(12, 16).trim();
    if (["N", "CA", "C", "O"].includes(name)) {
      map.set(`${line[21]}:${line.slice(22, 26).trim()}:${name}`, [+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)]);
    }
  }
  return map;
}
function binderRmsd(designPdb, refoldPdb) {
  const design = backbone(designPdb), refold = backbone(refoldPdb);
  const keys = [...design.keys()].filter((k) => refold.has(k));
  const x = new Float32Array(keys.length * 3), y = new Float32Array(keys.length * 3), w = new Float32Array(keys.length);
  keys.forEach((k, i) => { x.set(refold.get(k), i * 3); y.set(design.get(k), i * 3); w[i] = k[0] === "A" ? 0 : 1; });
  const fitted = weightedRigidAlign(x, y, w, keys.length);
  let sum = 0, count = 0;
  keys.forEach((k, i) => { if (k[0] !== "A") return; for (let a = 0; a < 3; a += 1) sum += (fitted[i * 3 + a] - y[i * 3 + a]) ** 2; count += 1; });
  return Math.sqrt(sum / count);
}

async function attempt(name, index) {
  const t = targets[name];
  const cfg = config;
  // `pair`: arms of one A/B draw the same lengths and evaluation seeds
  const rng = mulberry(seedOf(`${cfg.pair ?? cfg.tag}:${name}:${index}`));
  const [lo, hi0] = t.binder;
  const hi = Math.min(hi0, lo + (cfg.lengthSpan ?? 1e9));
  const length = lo + Math.floor(rng() * (hi - lo + 1));
  const evalSeed = 1 + Math.floor(rng() * 2 ** 30);
  const seed = 1000 * index + 7;
  const templateFile = path.join(work, `${name}.pdb`);
  if (!fs.existsSync(templateFile)) fs.writeFileSync(templateFile, t.pdb);
  const designModel = cfg.folder ?? "af3";
  const targetMsas = cfg.msaDir === undefined ? undefined
    : t.sequences.map((s) => path.join(cfg.msaDir, `${createHash("sha1").update(s).digest("hex").slice(0, 12)}.a3m`));
  // the benchmark's hotspots in the fold's numbering (each target chain's residues in order), handed as a pocket
  // to a design engine trained on one (boltz2, chai1) - the original Protein Hunter's use of them
  const pocket = [];
  if (cfg.hotspots && (designModel === "boltz2" || designModel === "chai1")) {
    const order = {};
    for (const line of t.pdb.split("\n")) {
      if (!line.startsWith("ATOM")) continue;
      const list = (order[line[21]] ??= []);
      const residue = Number(line.slice(22, 26));
      if (list[list.length - 1] !== residue) list.push(residue);
    }
    for (const [letter, residue] of t.hotspots) {
      const at = (order[letter] ?? []).indexOf(residue);
      if (at >= 0) pocket.push([letter, at + 1]);
    }
  }
  const checkModel = cfg.checkModel ?? "af3";
  const assess = async (binder, designPdb, foldSeed, model) => {
    const r = await fold(t, templateFile, [binder, ...t.sequences].join(":"), { model, steps: 200, recycles: 5, seed: foldSeed });
    const rmsd = binderRmsd(designPdb, r.pdb);
    return { binderPtm: r.binderPtm, ipae: r.ipae, rmsd, pdb: r.pdb, pass: r.binderPtm >= 0.8 && r.ipae <= 1.5 && rmsd <= 2.5 };
  };
  const looksDone = (e) => cfg.stop !== undefined && e.binderPtm >= cfg.stop.ptm && e.ipae <= cfg.stop.ipae;
  const records = [], eligible = [], checked = [];
  let runs = 0, foldSeconds = 0;
  const started = Date.now();
  for (let restart = 0; restart < (cfg.restarts ?? 1); restart += 1) {
    runs += 1;
    const controller = new AbortController();
    const mine = [];
    try {
      const stream = runDesign({
        chains: ["", ...t.sequences], chainIndex: 0, cycles: cfg.cycles, length, run: restart,
        seed: seed + 101 * restart, signal: controller.signal, kind: "protein",
        veto: ({ pdb }) => contactProblem(pdb, { chain: "A" }),
        fold: async (sequence, context) => {
          const t0 = Date.now();
          const r = await fold(t, templateFile, sequence, { model: designModel, steps: cfg.steps, recycles: cfg.recycles,
            // `sameSeed`: every cycle of a run folds at ONE diffusion seed, as web/hunt.js does (a control for the page)
            seed: seed + 101 * restart + (cfg.sameSeed ? 0 : 13 * context.cycle), pocket, msas: targetMsas });
          foldSeconds += (Date.now() - t0) / 1000;
          return { pdb: r.pdb, confidence: r.confidence, terms: { binderPtm: r.binderPtm, ipae: r.ipae } };
        },
        design: (pdb, context) => designChain(mpnn, { pdb, chain: context.chain, temperature: context.temperature,
          omit: context.omit, alanineBias: context.alanineBias, random: context.random, hold: context.hold, kind: context.kind }),
      });
      for await (const record of stream) {
        const terms = record.folded.terms;
        records.push({ restart, cycle: record.cycle, score: record.score, vetoed: record.vetoed ?? null,
          alanine: record.alanine, ipae: terms.ipae, binderPtm: terms.binderPtm });
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
      const promising = mine.filter((e) => e.binderPtm >= cfg.check.ptm && e.ipae <= cfg.check.ipae)
        .sort((a, b) => a.ipae - b.ipae).slice(0, cfg.check.top);
      for (const entry of promising) {
        const result = await assess(entry.sequence, entry.pdb, seed * 7 + entry.cycle + 1000 * restart, checkModel);
        checked.push({ entry, result });
        if (result.pass) break;
      }
      if (checked.some((c) => c.result.pass)) break;
    } else if (eligible.some(looksDone)) {
      break;
    }
  }
  const margin = (c) => Math.max(c.result.ipae / 1.5, c.result.rmsd / 2.5, 0.8 / Math.max(c.result.binderPtm, 1e-3));
  const confident = eligible.filter((e) => e.binderPtm >= 0.8);
  const chosen = checked.length > 0 ? checked.reduce((a, b) => (margin(b) < margin(a) ? b : a)) : null;
  const best = cfg.select === "iptm" ? (eligible.length > 0 ? eligible.reduce((a, b) => (b.score > a.score ? b : a)) : null)
    : chosen !== null ? (cfg.submitChecked ? { ...chosen.entry, pdb: chosen.result.pdb } : chosen.entry)
    : confident.length > 0 ? confident.reduce((a, b) => (b.ipae < a.ipae ? b : a))
    : eligible.length > 0 ? eligible.reduce((a, b) => (b.binderPtm > a.binderPtm ? b : a)) : null;
  const row = { target: name, attempt: index, length, config: cfg.tag, runs, records, foldSeconds,
    checks: checked.map((c) => ({ restart: c.entry.restart, cycle: c.entry.cycle, binderPtm: c.result.binderPtm, ipae: c.result.ipae, rmsd: c.result.rmsd })) };
  if (best !== null) {
    // 🔴 THE EVALUATION IS ALWAYS AF3 (the native port), whatever designed or checked
    const final = await assess(best.sequence, best.pdb, evalSeed, "af3");
    Object.assign(row, { binder: best.sequence, bestCycle: best.cycle, bestRestart: best.restart,
      binderPtm: final.binderPtm, ipae: final.ipae, rmsd: final.rmsd, designPdb: best.pdb, refoldPdb: final.pdb });
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
    console.log(name, index, row.binderPtm?.toFixed?.(2), row.ipae?.toFixed?.(2), row.rmsd?.toFixed?.(2), row.wall?.toFixed?.(0), row.error?.slice(0, 120) ?? "");
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}

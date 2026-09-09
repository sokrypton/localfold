/**
 * OpenDDE, sequence in and a structure out.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-opendde.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-opendde.js --steps=50 --target=1qys
 *
 * 🔴 SCORED AGAINST THE CRYSTAL, WHICH IS THE ONLY ORACLE THIS PORT HAS - and
 * the geometry is the gate before the fold is. A CA-CA spacing of 3.8 A is a
 * peptide bond: a port with the arithmetic subtly wrong produces a plausible
 * cloud at the wrong scale, and RMSD alone would not say which. The sequence
 * and the geometry come from the SAME deposition so they cannot disagree.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch, toPdb, backboneGeometry, warmTrunkPipelines }
  from "../../src/af3/fold.js";
import { structuralLayout } from "../../src/af3/structural-tokens.js";
import { STRUCTURAL_REFINER } from "../../src/af3/weights.js";
import { assertChainGeometry } from "./chain-geometry.js";
import { memorySnapshot } from "../../src/runtime/device-memory.js";
import { setDeviceTuning } from "../../src/runtime/device-profile.js";
import { profileDevice } from "./profile.js";
import { profileBuffers } from "./buffer-profile.js";
import { setMemoryBudget } from "../../src/runtime/device-memory.js";
import {
  confidenceWeights, openAf3Store, openddeConfidenceWeights,
  structuralExpanderWeights, structuralRefinerWeights, trunkWeights,
} from "../../src/af3/weights.js";
import { atomReference, diffusionWeights, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V", MSE: "M",
};

/** One CA per (chain, residue), altLoc " " or "A"; see trunk-opendde.js. */
function readChain(text, wanted) {
  const residues = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A") continue;
    if (line[21] !== wanted) continue;
    const code = THREE_TO_ONE[line.slice(17, 20).trim()];
    if (code === undefined) continue;
    const number = Number(line.slice(22, 26));
    if (!residues.has(number)) residues.set(number, { code, ca: null });
    if (line.slice(12, 16).trim() === "CA") {
      residues.get(number).ca = [Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                                 Number(line.slice(46, 54))];
    }
  }
  const ordered = [...residues.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  return { sequence: ordered.map((r) => r.code).join(""),
           alphaCarbons: ordered.map((r) => r.ca) };
}

/** Kabsch RMSD after superposition, and a TM-score. */
function superpose(model, truth) {
  const pairs = model.map((p, i) => [p, truth[i]]).filter(([a, b]) => a && b);
  const n = pairs.length;
  const centre = (which) => {
    const c = [0, 0, 0];
    for (const pair of pairs) for (let d = 0; d < 3; d += 1) c[d] += pair[which][d] / n;
    return c;
  };
  const cm = centre(0);
  const ct = centre(1);
  const covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const [a, b] of pairs) {
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) covariance[i][j] += (a[i] - cm[i]) * (b[j] - ct[j]);
    }
  }
  // Rotation by iterative polar decomposition - enough for a score, and it
  // avoids a second SVD in the tree.
  let rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const multiply = (x, y) => x.map((row, i) => y[0].map((_, j) =>
    row.reduce((s, v, k) => s + v * y[k][j], 0)));
  const transpose = (m) => m[0].map((_, j) => m.map((row) => row[j]));
  const inverse3 = (m) => {
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const c = (i, j) => {
      const rows = [0, 1, 2].filter((r) => r !== i);
      const cols = [0, 1, 2].filter((cc) => cc !== j);
      return ((i + j) % 2 ? -1 : 1)
        * (m[rows[0]][cols[0]] * m[rows[1]][cols[1]] - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]);
    };
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / det));
  };
  rotation = covariance;
  for (let iteration = 0; iteration < 40; iteration += 1) {
    const next = transpose(inverse3(rotation)).map((row, i) => row.map((v, j) =>
      0.5 * (rotation[i][j] + v)));
    rotation = next;
  }
  let squared = 0;
  const d0 = 1.24 * Math.cbrt(Math.max(n - 15, 1)) - 1.8;
  let tm = 0;
  const deviations = [];
  for (const [a, b] of pairs) {
    const moved = [0, 1, 2].map((i) =>
      [0, 1, 2].reduce((s, k) => s + (a[k] - cm[k]) * rotation[k][i], 0) + ct[i]);
    const d2 = [0, 1, 2].reduce((s, i) => s + (moved[i] - b[i]) ** 2, 0);
    squared += d2;
    deviations.push(Math.sqrt(d2));
    tm += 1 / (1 + d2 / (d0 * d0));
  }
  return { rmsd: Math.sqrt(squared / n), tm: tm / n, pairs: n, deviations };
}

export async function main(device, args) {
  // 🔴 `--tune=key=value`, THE SAME FLAG fold.js CARRIES. A knob no gate enters
  // is a knob nobody has checked, and this file's kernel choices -
  // `gridAttendMatrix`, `pairTransitionSplit`, `triangleProjectMatrix` - are all
  // device-profile knobs.
  for (const pair of (args ?? []).filter((a) => a.startsWith("--tune="))
       .flatMap((a) => a.slice("--tune=".length).split(",")).filter(Boolean)) {
    const at = pair.indexOf("=");
    if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
    const raw = pair.slice(at + 1);
    let value;
    try { value = JSON.parse(raw); } catch { value = raw; }
    setDeviceTuning(device, { [pair.slice(0, at)]: value });
  }
  // 🔴 A LENGTH ARM, because every measurement in this port so far is 68-92
  // residues - the regime where the resident weights dominate. The pair scratch
  // is quadratic in STRUCTURAL tokens and OpenDDE has about two per residue, so
  // whatever is true at 68 need not be true at 300.
  const synth = Number(option(args, "length", "0"));
  const target = option(args, "target", "6mrr");
  const crystalText = await (await fetch(`/tools/fixtures/${target}-crystal.pdb`)).text();
  const crystal = readChain(crystalText, option(args, "chain", "A"));
  const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
  const sequence = synth > 0
    ? Array.from({ length: synth }, (_, i) => ALPHABET[i % ALPHABET.length]).join("")
    : option(args, "sequence", crystal.sequence);
  const steps = Number(option(args, "steps", "200"));
  const recycles = Number(option(args, "recycles", "0"));
  const manifest = option(args, "model", "/model-opendde-int5/manifest.json");

  const batch = featuriseProtein(sequence, {});
  const openedAt = performance.now();
  const store = await openAf3Store(manifest);
  // 🔴 EVERY SHARD AT ONCE, WHICH IS WHAT THE PAGE DOES. `prefetch` is opt-in
  // because a bench that reads four blocks should not pull the whole manifest -
  // but this tool loads a whole model, so a run without it measures a download
  // pattern no user has: shards arrive as tensors are asked for, which leaves
  // most of the connection idle most of the time. Measured on the ESMFold2
  // tool, which had the same hole: a fold 2.25 s -> 1.75.
  store.prefetch();
  // 🔴 AND THE PAIRFORMER'S SHADERS, WHILE THE SHARDS ARE STILL ARRIVING.
  // OpenDDE's fold is 1.23 s of shader compilation out of 2.93, with the
  // compiler pool saturated while it runs and completely idle through the 1.74
  // s of weight load in front of it. It runs the stack at TWO token counts -
  // the residues and the structural tokens the expander produces - so both are
  // warmed. Not awaited; see warmTrunkPipelines.
  const structuralTokens = structuralLayout(batch).tokens;
  // 🔴 `--no-warm` IS THE ARM. A warm with no control beside it is a warm
  // nobody has priced, and this one is speculative by construction: it compiles
  // against a stand-in and can only be checked by counting what the fold
  // compiled with it and without.
  if (!args.includes("--no-warm")) void Promise.all([
    warmTrunkPipelines(device, store, batch.tokens),
    // ...and the refiner, which is a different root at a different token count
    // with a pair bias its kernels bind. See warmTrunkPipelines.
    warmTrunkPipelines(device, store, structuralTokens, {
      root: STRUCTURAL_REFINER, stack: { pairWeightPrecision: undefined },
      run: { extraPairBias: new Float32Array(0), keepPair: true },
    }),
  ]).catch(() => {});
  const storeMs = Math.round(performance.now() - openedAt);
  const trunk = await trunkWeights(store, Number(option(args, "blocks", "48")), 4);
  const trunkMs = Math.round(performance.now() - openedAt) - storeMs;
  const weights = {
    trunk,
    targetFeat: await targetFeatureWeights(store),
    // The structural stacks exist only in a bundle whose dialect says so; AF3
    // and OpenBind-0 run this same tool as the control and have none.
    ...(trunk.dialect.structuralTokens ? {
      expander: await structuralExpanderWeights(store),
      refiner: await structuralRefinerWeights(store),
      openddeConfidence: {
        ...await openddeConfidenceWeights(store),
        weightPrecision: option(args, "confidence-weights", undefined),
      },
    } : { confidence: await confidenceWeights(store) }),
    diffusion: await diffusionWeights(store),
    refinerWeightPrecision: option(args, "refiner-weights", undefined),
    atomReference: await atomReference(store),
  };
  // 🔴 AND IT IS THE DOWNLOAD, NOT A DECODE. `openAf3Store` is 12 ms - the
  // manifest - and the loaders are 1.75 s, of which `trunkWeights` is 1.54. The
  // AF3 loaders are lazy: they build thunks and OPEN the shards, and 495 MB at
  // the **371 MB/s this browser's fetch reaches** is 1.33 s of it.
  //
  // 🔴 AND 371 MB/s IS THE BROWSER, NOT THE SERVER. The same twelve shards read
  // with python from the same server are 2129 MB/s; `probe-shard-read.js`
  // measures both paths inside Chrome and they agree with each other -
  // `arrayBuffer()` 371, the streamed read the store uses 371 - so the chunk
  // loop costs nothing and the cap is six HTTP/1.1 connections at about 62 MB/s
  // each. The lever is fewer bytes or more connections, not fewer instructions;
  // see docs/HOSTING.md.
  const weightSplit = { store: storeMs, trunk: trunkMs,
                        loaders: Math.round(performance.now() - openedAt) - storeMs };

  // 🔴 THE TRAJECTORY IS MEASURED, NOT LOOKED AT. A frame drawn from the wrong
  // token space has the right ATOM COUNT and the wrong atoms, so it renders as
  // a plausible cloud - and the radius of gyration is what separates that from
  // a protein: a 68-residue chain is about 11 A, and scrambled atoms are not.
  const closeTimings = () => {
    // 🔴 THE LAST STAGE NEVER CLOSES ITSELF. `onStage` attributes the gap
    // between two notifications to the earlier one, so whatever runs after the
    // final stage - the confidence head, the readbacks - was never counted at
    // all, and the timings summed to 7.2 s of a 23.5 s fold.
    if (lastStage !== null) {
      timings[lastStage.name] = (timings[lastStage.name] ?? 0)
        + Math.round(performance.now() - lastStage.at);
      lastStage = null;
    }
  };
  const frames = [];
  const unmapped = [];
  let lastStage = null;
  const timings = {};
  // 🔴 THE FOLD'S CLOCK STARTS AFTER THE WEIGHTS ARE LOADED, AND A USER'S DOES
  // NOT. See fold-esmfold2.js, where that gap was 1.4 seconds nobody was
  // measuring. `weightSeconds` is everything above this line.
  const weightSeconds = Number(((performance.now() - openedAt) / 1000).toFixed(3));
  // 🔴 `--profile` SO THE FOLD'S GPU TIME IS ATTRIBUTABLE AT ALL. The stage
  // clock says which STAGE the wall time is in; it cannot say whether a stage
  // is compute or the host waiting, and two of this fold's five biggest stages
  // turned out to be host arithmetic. The query set holds 2048 passes, so a
  // whole fold overflows it - read `gpuDropped`, and drop the step count to
  // profile the sampler.
  const profile = args.includes("--profile") ? profileDevice(device) : null;
  // 🔴 AND `--buffers` ANSWERS WHAT `--profile` CANNOT: how much of the wall is
  // the host on the bus or waiting on a drain. An OpenDDE fold's pairformer is
  // 4.4 seconds of wall against a few hundred milliseconds of labelled compute,
  // and only this says where the difference is.
  const buffers = args.includes("--buffers") ? profileBuffers(device) : null;
  // 🔴 A CEILING, SO THE WIDE TRACK'S RESIDENCY REFUSAL CAN BE MADE TO FIRE.
  // The rule in foldBatch keeps 48 blocks of a 384-channel pair track on the
  // device when the device has room and declines when it does not; a fallback
  // nothing has taken is a fallback nobody has checked.
  const budgetMiB = Number(option(args, "budget", "0"));
  if (budgetMiB > 0) setMemoryBudget(device, budgetMiB * 1024 * 1024);
  const started = performance.now();
  // 🔴 THE SAME DRIVER EVERY OTHER AlphaFold 3-graph MODEL USES. The
  // structural-token stage is a branch inside foldBatch gated on the dialect,
  // not a second driver - so the recycles, the contact map, the trunk cache
  // and the stage callbacks are shared rather than reproduced.
  const fold = await foldBatch(device, batch, weights, {
    // The resident trunk weights' element; see the measurement in docs.
    weightPrecision: option(args, "weights", undefined),
    pairWeightPrecision: option(args, "pair-weights", undefined),
    // ...undefined unless asked, so the width rule in foldBatch decides.
    residentWeights: args.includes("--no-resident") ? false
      : args.includes("--resident") ? true : undefined,
    steps, recycles, seed: Number(option(args, "seed", "20260831")),
    mode: option(args, "mode", "diffusion"),
    onStep: ({ step, denoised, structuralDenoised }) => {
      // 🔴 THE CONTROL: the SAME frame read the way the page read it before -
      // structural-layout coordinates indexed by the residue mask. It has the
      // right atom count and the wrong atoms.
      const rgOf = (coordinates) => {
        let cx = 0; let cy = 0; let cz = 0; let n = 0;
        for (let index = 0; index < batch.tokens * batch.dense; index += 1) {
          if (!batch.predDenseAtomMask[index]) continue;
          cx += coordinates[index * 3]; cy += coordinates[index * 3 + 1];
          cz += coordinates[index * 3 + 2]; n += 1;
        }
        cx /= n; cy /= n; cz /= n;
        let s = 0;
        for (let index = 0; index < batch.tokens * batch.dense; index += 1) {
          if (!batch.predDenseAtomMask[index]) continue;
          s += (coordinates[index * 3] - cx) ** 2 + (coordinates[index * 3 + 1] - cy) ** 2
            + (coordinates[index * 3 + 2] - cz) ** 2;
        }
        return Number(Math.sqrt(s / n).toFixed(2));
      };
      // ...absent for a model with one token space, which is every other one.
      if (structuralDenoised !== undefined) {
        unmapped.push({ step, rg: rgOf(structuralDenoised) });
      }
      let cx = 0;
      let cy = 0;
      let cz = 0;
      let n = 0;
      for (let index = 0; index < batch.tokens * batch.dense; index += 1) {
        if (!batch.predDenseAtomMask[index]) continue;
        cx += denoised[index * 3]; cy += denoised[index * 3 + 1]; cz += denoised[index * 3 + 2];
        n += 1;
      }
      cx /= n; cy /= n; cz /= n;
      let squared = 0;
      for (let index = 0; index < batch.tokens * batch.dense; index += 1) {
        if (!batch.predDenseAtomMask[index]) continue;
        squared += (denoised[index * 3] - cx) ** 2 + (denoised[index * 3 + 1] - cy) ** 2
          + (denoised[index * 3 + 2] - cz) ** 2;
      }
      frames.push({ step, rg: Number(Math.sqrt(squared / n).toFixed(2)) });
    },
    // 🔴 foldBatch's `onStage` NOTIFIES, it does not time - so the clock is
    // here. Each stage's cost is the gap between its announcement and the next.
    onStage: (name) => {
      const now = performance.now();
      if (lastStage !== null) {
        timings[lastStage.name] = (timings[lastStage.name] ?? 0)
          + Math.round(now - lastStage.at);
      }
      lastStage = { name, at: now };
    },
  });
  closeTimings();
  const wholeMs = Math.round(performance.now() - started);

  // 🔴 FOLD THE WHOLE THING AGAIN, REUSING NOTHING. A first fold's numbers are
  // all pipeline compilation and first touch - measured here, `pairformer-block`
  // is 4.4 seconds of wall against about 250 ms of labelled GPU and does not
  // move when the recycles go from 3 to 0, which is what a one-time cost looks
  // like. The row a user sees is the SECOND fold, and nothing was measuring it.
  //
  // The atoms are compared to the first fold's: a repeat exists to price the
  // residency, and residency that returned a different structure would be
  // invisible in a stopwatch.
  const repeat = Number(option(args, "repeat", "1"));
  // Snapshotted before the repeats, which reuse the same clock.
  const firstTimings = { ...timings };
  const repeats = [];
  for (let again = 1; again < repeat; again += 1) {
    for (const key of Object.keys(timings)) delete timings[key];
    lastStage = null;
    const at = performance.now();
    const other = await foldBatch(device, batch, weights, {
      weightPrecision: option(args, "weights", undefined),
      pairWeightPrecision: option(args, "pair-weights", undefined),
      residentWeights: args.includes("--no-resident") ? false
        : args.includes("--resident") ? true : undefined,
      steps, recycles, seed: Number(option(args, "seed", "20260831")),
      mode: option(args, "mode", "diffusion"),
      onStage: (name) => {
        const now = performance.now();
        if (lastStage !== null) {
          timings[lastStage.name] = (timings[lastStage.name] ?? 0)
            + Math.round(now - lastStage.at);
        }
        lastStage = { name, at: now };
      },
    });
    closeTimings();
    let worst = 0;
    for (let index = 0; index < fold.positions.length; index += 1) {
      worst = Math.max(worst, Math.abs(fold.positions[index] - other.positions[index]));
    }
    repeats.push({
      wholeMs: Math.round(performance.now() - at),
      worstDisplacement: Number(worst.toFixed(6)),
      sameFold: worst === 0,
      timings: Object.fromEntries(Object.entries(timings)
        .filter(([, ms]) => ms >= 20).sort((a, b) => b[1] - a[1])),
    });
  }

  const geometry = fold.geometry;
  // 🔴 AND IT GATES NOW. This file's own opening line is "the geometry is the
  // gate before the fold is", and until this it was the gate before nothing:
  // the number was computed, reported and never asserted on. See
  // tools/gpu/chain-geometry.js for what that cost AF2.
  assertChainGeometry(geometry, {
    plddt: fold.meanPlddt, doc: "docs/OPENDDE.md and docs/AF2.md",
    allow: args.includes("--allow-broken-geometry"),
  });
  // The model's alpha carbons, in residue order.
  const modelCa = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    let slot = -1;
    for (let s = 0; s < batch.dense; s += 1) {
      const base = (token * batch.dense + s) * 4;
      const name = [0, 1, 2, 3].map((c) => {
        const v = batch.refAtomNameChars[base + c];
        return v > 0 ? String.fromCharCode(v + 32) : "";
      }).join("");
      if (name === "CA") { slot = s; break; }
    }
    modelCa.push(slot < 0 ? null : [
      fold.positions[(token * batch.dense + slot) * 3],
      fold.positions[(token * batch.dense + slot) * 3 + 1],
      fold.positions[(token * batch.dense + slot) * 3 + 2]]);
  }
  const scored = sequence === crystal.sequence
    ? superpose(modelCa, crystal.alphaCarbons) : undefined;

  // 🔴 A pLDDT IN THE RIGHT RANGE IS NOT A pLDDT THAT MEANS ANYTHING. What it
  // claims is that a residue is placed well, so the check is whether it tracks
  // the residue's actual deviation from the deposition - a head read at the
  // wrong bin grid, or with `plddt_weight` broadcast instead of selected per
  // atom, still lands in 0-100 and still averages to something plausible.
  // Negative Spearman is the expectation: high pLDDT, low error.
  let plddtVsError;
  if (scored !== undefined && fold.perResiduePlddt !== undefined) {
    const pairs = [];
    for (let i = 0; i < batch.tokens; i += 1) {
      if (scored.deviations[i] === undefined || fold.perResiduePlddt[i] === undefined) continue;
      pairs.push([fold.perResiduePlddt[i], scored.deviations[i]]);
    }
    const rank = (values) => {
      const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
      const out = new Array(values.length);
      order.forEach(([, i], at) => { out[i] = at; });
      return out;
    };
    const a = rank(pairs.map((p) => p[0]));
    const b = rank(pairs.map((p) => p[1]));
    const n = pairs.length;
    const mean = (n - 1) / 2;
    let num = 0;
    let da = 0;
    let db = 0;
    for (let i = 0; i < n; i += 1) {
      num += (a[i] - mean) * (b[i] - mean);
      da += (a[i] - mean) ** 2;
      db += (b[i] - mean) ** 2;
    }
    plddtVsError = Number((num / Math.sqrt(da * db)).toFixed(4));
  }

  const gpuSummary = profile === null ? undefined : await profile.summary();
  const profiled = profile === null ? undefined : await profile.report();
  profile?.restore();
  if (buffers !== null) {
    const traffic = buffers.report();
    console.log(`buffer traffic: ${traffic.totalMs.toFixed(0)} ms of ${wholeMs} ms wall`
      + ` (${(100 * traffic.totalMs / wholeMs).toFixed(0)}%)`);
    console.log(`  queue drain: ${traffic.queueWait.unionMs.toFixed(0)} ms`
      + ` (${(100 * traffic.queueWait.unionMs / wholeMs).toFixed(0)}% of wall)`
      + ` over ${traffic.queueWait.calls} onSubmittedWorkDone, union not sum`);
    for (const row of traffic.byKind) {
      console.log(`  ${row.ms.toFixed(0).padStart(6)} ms`
        + ` ${(100 * row.ms / wholeMs).toFixed(1).padStart(5)}%`
        + ` x${String(row.calls).padEnd(6)}`
        + ` ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(9)} MiB  ${row.kind}`);
    }
    for (const row of traffic.rows.slice(0, 14)) {
      console.log(`    ${row.ms.toFixed(0).padStart(6)} ms x${String(row.calls).padEnd(6)}`
        + ` ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(9)} MiB`
        + `  ${row.kind} ${row.label}`);
    }
    buffers.restore();
  }

  return {
    target, sequence: sequence.length,
    ...(profiled === undefined ? {} : {
      gpuSummary,
      gpuTotalMs: Number(profiled.reduce((t, e) => t + e.ms, 0).toFixed(1)),
      gpuPasses: profiled.slice(0, 24),
      gpuLabels: profiled.length,
      gpuDispatches: profiled.reduce((t, e) => t + e.passes, 0),
    }),
    residueTokens: batch.tokens, structuralTokens: fold.structuralTokens,
    meanPlddt: fold.meanPlddt ?? null,
    // ...undefined unless asked, so the width rule in foldBatch decides.
    residentWeights: args.includes("--no-resident") ? false
      : args.includes("--resident") ? true : undefined,
    steps, recycles, wholeMs, budgetMiB, weightSeconds, weightSplit,
    ...(repeats.length === 0 ? {} : { repeats }),
    timings: Object.fromEntries(Object.entries(firstTimings)
      .filter(([, ms]) => ms >= 20).sort((a, b) => b[1] - a[1])),
    plddtSpread: fold.perResiduePlddt === undefined ? undefined : (() => {
      const v = fold.perResiduePlddt.filter((x) => x !== undefined);
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length);
      return { min: Number(Math.min(...v).toFixed(2)), max: Number(Math.max(...v).toFixed(2)),
               sd: Number(sd.toFixed(3)) };
    })(),
    coordinateCheck: fold.scores?.coordinateCheck,
    peakMiB: Number((memorySnapshot(device).peakBytes / 2 ** 20).toFixed(1)),
    peakRows: memorySnapshot(device).peakByLabel.slice(0, 6)
      .map((r) => ({ label: r.label, MiB: Number((r.bytes / 2 ** 20).toFixed(1)),
                     count: r.count })),
    geometry,
    frameGyration: frames.filter((f, i) => i % 6 === 0 || i === frames.length - 1),
    frameGyrationUnmapped: unmapped.filter((f, i) => i % 6 === 0 || i === unmapped.length - 1),
    scored: scored && { rmsd: Number(scored.rmsd.toFixed(3)),
                        tm: Number(scored.tm.toFixed(4)), pairs: scored.pairs },
    // Spearman of per-residue pLDDT against per-residue deviation. NEGATIVE is
    // the model working: high confidence where the error is small.
    plddtVsError,
    finite: fold.positions.every(Number.isFinite),
    pdb: args.includes("--pdb") ? toPdb(batch, fold.positions, null) : undefined,
    deviceMemory: memorySnapshot(device),
  };
}

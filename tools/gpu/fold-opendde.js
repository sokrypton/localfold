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
import { foldBatch, toPdb, backboneGeometry } from "../../src/af3/fold.js";
import { memorySnapshot } from "../../src/runtime/device-memory.js";
import {
  openAf3Store, structuralExpanderWeights, structuralRefinerWeights, trunkWeights,
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
  for (const [a, b] of pairs) {
    const moved = [0, 1, 2].map((i) =>
      [0, 1, 2].reduce((s, k) => s + (a[k] - cm[k]) * rotation[k][i], 0) + ct[i]);
    const d2 = [0, 1, 2].reduce((s, i) => s + (moved[i] - b[i]) ** 2, 0);
    squared += d2;
    tm += 1 / (1 + d2 / (d0 * d0));
  }
  return { rmsd: Math.sqrt(squared / n), tm: tm / n, pairs: n };
}

export async function main(device, args) {
  const target = option(args, "target", "6mrr");
  const crystalText = await (await fetch(`/tools/fixtures/${target}-crystal.pdb`)).text();
  const crystal = readChain(crystalText, option(args, "chain", "A"));
  const sequence = option(args, "sequence", crystal.sequence);
  const steps = Number(option(args, "steps", "200"));
  const recycles = Number(option(args, "recycles", "0"));
  const manifest = option(args, "model", "/model-opendde-int5/manifest.json");

  const batch = featuriseProtein(sequence, {});
  const store = await openAf3Store(manifest);
  const weights = {
    trunk: await trunkWeights(store, Number(option(args, "blocks", "48")), 4),
    targetFeat: await targetFeatureWeights(store),
    expander: await structuralExpanderWeights(store),
    refiner: await structuralRefinerWeights(store),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
  };

  const timings = {};
  const started = performance.now();
  // 🔴 THE SAME DRIVER EVERY OTHER AlphaFold 3-graph MODEL USES. The
  // structural-token stage is a branch inside foldBatch gated on the dialect,
  // not a second driver - so the recycles, the contact map, the trunk cache
  // and the stage callbacks are shared rather than reproduced.
  const fold = await foldBatch(device, batch, weights, {
    steps, recycles, seed: Number(option(args, "seed", "20260831")),
    mode: option(args, "mode", "diffusion"),
    onStage: (name, detail) => {
      timings[name] = typeof detail === "number" ? Math.round(detail)
        : Math.round(detail?.ms ?? 0);
    },
  });
  const wholeMs = Math.round(performance.now() - started);

  const geometry = fold.geometry;
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

  return {
    target, sequence: sequence.length,
    residueTokens: batch.tokens, structuralTokens: fold.structuralTokens,
    meanPlddt: fold.meanPlddt ?? null,
    steps, recycles, wholeMs, timings,
    geometry,
    scored: scored && { rmsd: Number(scored.rmsd.toFixed(3)),
                        tm: Number(scored.tm.toFixed(4)), pairs: scored.pairs },
    finite: fold.positions.every(Number.isFinite),
    pdb: args.includes("--pdb") ? toPdb(batch, fold.positions, null) : undefined,
    deviceMemory: memorySnapshot(device),
  };
}

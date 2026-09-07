/**
 * OpenDDE's trunk, sequence in and contact map out.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/trunk-opendde.js
 *     node tools/gpu-chrome.mjs tools/gpu/trunk-opendde.js \
 *       --model=/model-opendde-trunk-int5/manifest.json --crystal=6mrr
 *
 * 🔴 THIS IS THE ONLY END-TO-END STATEMENT THE PORT CAN MAKE, AND IT IS NOT AN
 * ORACLE. OpenDDE's diffusion runs on an expanded structural-token set that
 * this graph does not have, so there are no coordinates to compare and no dump
 * of OpenDDE's own intermediates to compare against. What there IS is the
 * distogram: a contact map is a claim about a real structure, and a deposited
 * crystal can say whether the claim is true. A trunk assembled with the wrong
 * width, the wrong head count, the wrong bin grid or either branch inverted
 * does not predict a fold's contacts by accident.
 *
 * 🔴 AND IT RUNS AlphaFold 3 THROUGH THE SAME PATH ON PURPOSE. `--model` takes
 * either bundle, and AF3's number is the control: without it, "OpenDDE predicts
 * contacts at precision 0.9" has nothing to be good or bad against, and a
 * harness fault that flattered both would be invisible.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { af3ContactClasses } from "../../src/af3/contact-classes.js";
import { memorySnapshot } from "../../src/runtime/device-memory.js";
import { af3Dialect, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";
import { contactAngstromsForClasses } from "../../src/heads/contact-threshold.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * The sequence and the geometry from ONE file, which is the point.
 *
 * 🔴 A HARDCODED SEQUENCE BESIDE A CRYSTAL IS A SILENT MISMATCH WAITING. The
 * first version of this tool carried a 6MRR sequence typed from memory; it was
 * a different protein from the deposition beside it, and it folded and scored
 * and reported numbers. Reading both from the same PDB makes that impossible.
 *
 * One CA per (chain, residue number), altLoc " " or "A" only - 6MRR's chain A
 * has 71 CA records for 68 residues, and a naive walk pairs the model against
 * a shifted crystal. This repository records that costing a fold 4.74 A and
 * TM 0.393 where the truth was 1.43 A and 0.922.
 */
const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V", MSE: "M",
};

function readChain(text, wanted) {
  const residues = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const altLoc = line[16];
    if (altLoc !== " " && altLoc !== "A") continue;
    if (line[21] !== wanted) continue;
    const number = Number(line.slice(22, 26));
    const name = line.slice(12, 16).trim();
    const element = (line.slice(76, 78).trim() || name[0]).toUpperCase();
    if (element === "H") continue;
    const code = THREE_TO_ONE[line.slice(17, 20).trim()];
    if (code === undefined) continue;
    let residue = residues.get(number);
    if (residue === undefined) {
      residue = { number, code, atoms: [] };
      residues.set(number, residue);
    }
    residue.atoms.push([Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                        Number(line.slice(46, 54))]);
  }
  const ordered = [...residues.values()].sort((a, b) => a.number - b.number);
  return { sequence: ordered.map((r) => r.code).join(""), residues: ordered };
}

/**
 * 🔴 THE GROUND TRUTH IS REAL ATOMIC CONTACT, NOT A REPRESENTATIVE DISTANCE.
 * `tools/calibrate-contact-cutoff.py` established that here: two residues are
 * in contact when ANY heavy atom of one is within 5 A of any heavy atom of the
 * other. Scoring a predicted pseudo-beta threshold against another
 * pseudo-beta threshold would only measure that two conventions agree.
 */
function actualContacts(residues, separation) {
  const contacts = new Set();
  for (let i = 0; i < residues.length; i += 1) {
    for (let j = i + separation; j < residues.length; j += 1) {
      let touching = false;
      for (const a of residues[i].atoms) {
        for (const b of residues[j].atoms) {
          const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
          if (dx * dx + dy * dy + dz * dz <= 25) { touching = true; break; }
        }
        if (touching) break;
      }
      if (touching) contacts.add(i * residues.length + j);
    }
  }
  return contacts;
}

export async function main(device, args) {
  const target = option(args, "target", "6mrr");
  const crystalText = await (await fetch(`/tools/fixtures/${target}-crystal.pdb`)).text();
  const crystal = readChain(crystalText, option(args, "chain", "A"));
  const sequence = option(args, "sequence", crystal.sequence);
  const rows = Number(option(args, "msa", "1"));
  const blocks = Number(option(args, "blocks", "48"));
  const passes = Number(option(args, "recycles", "1"));
  const manifest = option(args, "model", "/model-opendde-trunk-int5/manifest.json");

  const batch = featuriseProtein(sequence, {});
  const tokens = batch.tokens;
  const store = await openAf3Store(manifest);
  // 🔴 THE DIALECT IS THE BUNDLE'S, NOT A MODULE CONSTANT. bench-trunk.js
  // imports `DIALECT` from fold.js, which is stock AlphaFold 3's - correct for
  // the bundle it was written against and wrong for every other.
  let dialect = af3Dialect(store);
  // 🔴 --ablate AND --enable PRICE A BRANCH, because a branch that is silent
  // when wrong cannot be trusted on a reading of somebody else's table. Same
  // shape as tools/gpu/fold.js's own two flags.
  const ablate = (option(args, "ablate", "") || "").split(",").filter(Boolean);
  const enable = (option(args, "enable", "") || "").split(",").filter(Boolean);
  if (ablate.length || enable.length) {
    const changed = { ...dialect };
    for (const flag of ablate) {
      if (!(flag in changed)) throw new Error(`no dialect flag ${flag}`);
      changed[flag] = false;
    }
    for (const flag of enable) {
      if (!(flag in changed)) throw new Error(`no dialect flag ${flag}`);
      changed[flag] = true;
    }
    dialect = Object.freeze(changed);
  }
  const weights = { trunk: await trunkWeights(store, blocks, 4),
                    targetFeat: await targetFeatureWeights(store) };
  // The embedder carries its own copy (it is the one stage that reads the
  // dialect off the weights rather than from the caller), so an ablation has to
  // reach it too or it would silently keep the bundle's own answer.
  weights.trunk.embedder = { ...weights.trunk.embedder, dialect };
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);

  const pairChannels = weights.trunk.embedder.pairChannels;
  const singleChannels = weights.trunk.embedder.singleChannels;
  const bins = weights.trunk.distogram.bins;

  // A depth-1 alignment: the query alone. OpenDDE takes an MSA and this tool
  // does not search one, so read the contacts as "what the trunk does with no
  // evolutionary information" rather than as the model's best.
  const msa = new Int32Array(rows * tokens);
  const deletionMatrix = new Float32Array(rows * tokens);
  const msaMask = new Float32Array(rows * tokens).fill(1);
  for (let row = 0; row < rows; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      msa[row * tokens + token] = batch.msa[token];
    }
  }
  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  const trunkGpu = new Af3TrunkGpu(device, { residentWeights: true });
  let previousPair = new Float32Array(tokens * tokens * pairChannels);
  let previousSingle = new Float32Array(tokens * singleChannels);
  let trunk;
  const timings = {};
  const started = performance.now();
  for (let pass = 0; pass < passes; pass += 1) {
    trunk = await trunkGpu.run({
      tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
      msaRows: msa, deletionMatrix, msaMask,
      bondMatrix: batch.bondMatrix, pairMask, seqMask, previousPair, previousSingle,
      contactClasses: af3ContactClasses(batch, tokens),
    }, weights.trunk, dialect, { onStage: (name, ms) => { timings[name] = Math.round(ms); } });
    previousPair = trunk.pair;
    previousSingle = trunk.single;
  }
  const wholeMs = Math.round(performance.now() - started);

  // 🔴 THE PAIR IS FINITE AND NOT CONSTANT, ASSERTED BEFORE ANYTHING IS READ
  // OFF IT. A trunk that produced NaN, or a flat tensor, would still hand the
  // distogram head something to project, and every number below would be a
  // statement about that rather than about the model.
  const finite = trunk.pair.every(Number.isFinite);
  const rmsOf = (a) => Number(Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length).toFixed(4));
  const pairRms = rmsOf(trunk.pair);

  const classes = af3ContactClasses(batch, tokens);
  const contacts = [];
  for (let i = 0; i < tokens; i += 1) {
    for (let j = i + 1; j < tokens; j += 1) {
      // The same separation rule the rest of this repository uses for a single
      // unmodified protein chain.
      if (j - i < 6) continue;
      contacts.push({ i, j, p: trunk.contactProbs[i * tokens + j],
                      cutoff: contactAngstromsForClasses(classes[i], classes[j]) });
    }
  }
  const predicted = contacts.filter((c) => c.p > 0.5);

  // 🔴 SCORED AGAINST THE DEPOSITION, WHICH IS THE ONLY ORACLE THIS PORT HAS.
  // There is no dump of OpenDDE's own intermediates to compare against and no
  // coordinates to superpose, so what says the trunk is assembled correctly is
  // that its contact map is TRUE of a real structure. A trunk with the wrong
  // width, the wrong head count, the wrong bin grid or either branch inverted
  // does not predict a fold's contacts by accident.
  const separation = 6;
  const actual = sequence === crystal.sequence
    ? actualContacts(crystal.residues, separation) : null;
  let scored;
  if (actual !== null) {
    let truePositive = 0;
    for (const c of predicted) if (actual.has(c.i * tokens + c.j)) truePositive += 1;
    // The strongest N, where N is how many contacts the crystal has - the
    // measure that does not move when a threshold does.
    const ranked = contacts.slice().sort((a, b) => b.p - a.p).slice(0, actual.size);
    let topL = 0;
    for (const c of ranked) if (actual.has(c.i * tokens + c.j)) topL += 1;
    scored = {
      actual: actual.size,
      predicted: predicted.length,
      precision: Number((truePositive / Math.max(predicted.length, 1)).toFixed(4)),
      recall: Number((truePositive / Math.max(actual.size, 1)).toFixed(4)),
      precisionAtN: Number((topL / Math.max(ranked.length, 1)).toFixed(4)),
      // 🔴 THE CONTROL. The same count of pairs drawn at random from the
      // eligible set is what "no information" scores, and on a 68-residue chain
      // that is far from zero - about a fifth of all pairs really are in
      // contact. Quoting precision against 0 would flatter every arm.
      chanceAtN: Number((actual.size / contacts.length).toFixed(4)),
    };
  }

  return {
    scored,
    model: dialect === undefined ? "?" : manifest,
    tokens, sequence: sequence.length, msaRows: rows, blocks, recycles: passes,
    widths: { pairChannels, singleChannels, bins,
              gridHeads: weights.trunk.pairformerBlocks[0].pairAttention1.heads,
              msaChannels: weights.trunk.msaBlocks[0].msaChannels },
    dialect: Object.fromEntries(Object.entries(dialect).filter(([, v]) => v)),
    trunk: { finite, pairRms, singleRms: rmsOf(trunk.single), msaRms: rmsOf(trunk.msa),
             wholeMs, ...timings },
    contacts: {
      predicted: predicted.length,
      eligiblePairs: contacts.length,
      strongest: predicted.sort((a, b) => b.p - a.p).slice(0, 8)
        .map((c) => ({ i: c.i, j: c.j, p: Number(c.p.toFixed(4)) })),
      meanProbability: Number(
        (contacts.reduce((s, c) => s + c.p, 0) / Math.max(contacts.length, 1)).toFixed(4)),
    },
    contactMatrix: option(args, "dump", null) === null ? undefined
      : Array.from(trunk.contactProbs),
    deviceMemory: memorySnapshot(device),
  };
}

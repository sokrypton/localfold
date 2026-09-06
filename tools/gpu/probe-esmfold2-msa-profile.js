/**
 * Can EF2-fast use an MSA?
 *
 * `s_inputs` is `[atomPooled 384 | aatype 33 | profile 33 | deletionMean 1]`,
 * and this checkpoint sets `disable_msa_features: true` - which does NOT mean
 * "no alignment was given". Upstream's `experimental.py` falls back to the
 * QUERY ONE-HOT when there is no MSA and only then zeroes both blocks, so the
 * profile channels are identically zero in training as at inference. The
 * weights say what that cost them: the profile columns of `zInit1` have a
 * coefficient of variation of 0.029 against the aatype block's 0.289, BELOW the
 * 0.044 that 256 columns of noise would give, and they are IDENTICAL between
 * the 600M and 300M checkpoints while the aatype columns beside them differ by
 * 0.075. Two training runs cannot agree on a shared weight unless no gradient
 * ever reached it.
 *
 * So the prediction is that a profile is a random projection rather than an
 * input, and this measures it rather than asserting it. Five arms, ONE process
 * because this machine drifts:
 *
 *   zeros      what ships
 *   onehot     the query one-hot - what upstream computes with the flag off
 *   msa        a real PSSM from a real alignment, with its deletion mean
 *   scrambled  that PSSM with its RESIDUE AXIS permuted (the control)
 *   seed       zeros again at a different sampler seed (the yardstick)
 *
 * 🔴 THE SCRAMBLED ARM IS THE WHOLE EXPERIMENT. A fold that moves when a
 * profile is supplied proves only that the numbers reached the matrix. If the
 * REAL profile and a permuted one move it by the same distance, nothing about
 * the alignment was used - and "it changed" would otherwise read as "it
 * worked".
 *
 * 🔴 AND THE YARDSTICK IS THE SAMPLER'S OWN SPREAD, NOT ZERO. Two seeds of
 * identical weights move a structure several angstroms; a displacement is only
 * meaningful against that.
 */
import {
  readTensor, readTensorAsFloat16, tensorByteLength,
} from "../../src/reference/dtype.js";
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";
import { foldEsmfold2 } from "../../src/esmfold2/fold.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../../src/esmfold2/language-pair-webgpu.js";
import { weightedRigidAlign } from "../../src/esmfold2/sampler-reference.js";
import { AATYPE_CLASSES, UNKNOWN_AATYPE } from "../../src/esmfold2/featurise.js";
import { parseA3m } from "../../src/input/a3m.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
const NARROW_LEAVES = new Set(["qkv/weights", "attn_out/weights",
  "fc1/weights", "fc2/weights"]);
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

function reader(bundle) {
  let table; const shards = new Map();
  const manifest = async () => {
    if (table === undefined) table = await (await fetch(`${bundle}/manifest.json`)).json();
    return table;
  };
  const read = async (name, half = false) => {
    const loaded = await manifest();
    const record = loaded.tensors[name];
    if (record === undefined) throw new Error(`${bundle} has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return half
      ? readTensorAsFloat16(record, shards.get(record.file), record.byteOffset ?? 0)
      : readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  };
  return { manifest, read };
}

/**
 * One-letter code to ESMFold2 restype. Its alphabet is three-letter
 * alphabetical from 2, which is why this is a table and not `charCode - 63`.
 */
const LETTER_TO_AATYPE = (() => {
  const order = "ARNDCQEGHILKMFPSTWYV";   // alphabetical by THREE-letter code
  const table = new Int32Array(128).fill(UNKNOWN_AATYPE);
  for (let i = 0; i < order.length; i += 1) table[order.charCodeAt(i)] = i + 2;
  return table;
})();

/**
 * The profile and the deletion mean, exactly as `experimental.py` computes
 * them: a masked mean of the one-hot over depth, and the mean count of
 * lower-case insertions preceding each column.
 *
 * 🔴 A GAP IS NOT A RESIDUE AND IS NOT A CLASS. Upstream one-hots the MSA
 * token, and its alphabet has no gap - so a gapped row contributes nothing at
 * that column and the denominator is the rows that are present there, which is
 * what `msa_attention_mask` does.
 */
function profileFromA3m(text, tokens) {
  // `parseA3m` has already removed the insertions and counted them, so this
  // reads its aligned rows and its deletion matrix rather than re-parsing.
  const { sequences, deletionMatrix } = parseA3m(text);
  const profile = new Float32Array(tokens * AATYPE_CLASSES);
  const deletionMean = new Float32Array(tokens);
  const counts = new Float32Array(tokens);
  for (let row = 0; row < sequences.length; row += 1) {
    const aligned = sequences[row];
    const deletions = deletionMatrix[row];
    for (let column = 0; column < tokens && column < aligned.length; column += 1) {
      if (aligned[column] !== "-") {
        profile[column * AATYPE_CLASSES + LETTER_TO_AATYPE[aligned.charCodeAt(column)]] += 1;
        counts[column] += 1;
      }
      deletionMean[column] += deletions?.[column] ?? 0;
    }
  }
  const rows = sequences.length;
  for (let token = 0; token < tokens; token += 1) {
    const present = Math.max(1, counts[token]);
    for (let c = 0; c < AATYPE_CLASSES; c += 1) profile[token * AATYPE_CLASSES + c] /= present;
    deletionMean[token] /= Math.max(1, rows);
  }
  return { profile, deletionMean, rows };
}

/** The same profile with its RESIDUE axis permuted: same statistics, no meaning. */
function scramble(profile, tokens, seed = 12345) {
  const order = [...Array(AATYPE_CLASSES).keys()];
  let state = seed >>> 0;
  for (let i = order.length - 1; i > 0; i -= 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const j = state % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const out = new Float32Array(profile.length);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < AATYPE_CLASSES; c += 1) {
      out[token * AATYPE_CLASSES + order[c]] = profile[token * AATYPE_CLASSES + c];
    }
  }
  return out;
}

function alphaCarbons(features) {
  const names = features.refAtomNameChars;
  const out = [];
  for (let atom = 0; atom < features.atoms; atom += 1) {
    if (features.mask[atom] !== 0 && names[atom * 4] === 35 && names[atom * 4 + 1] === 33
        && names[atom * 4 + 2] === 0) out.push(atom);
  }
  return out;
}

/** RMSD after superposition - a sampler draws a fresh frame every step. */
function displacement(a, b, mask, atoms) {
  const fitted = weightedRigidAlign(a, b, mask, atoms);
  let squared = 0, live = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    live += 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const d = fitted[atom * 3 + axis] - b[atom * 3 + axis];
      squared += d * d;
    }
  }
  return Math.sqrt(squared / live);
}

function caSpacing(x, alphas) {
  const gaps = [];
  for (let i = 1; i < alphas.length; i += 1) {
    const p = alphas[i - 1] * 3, q = alphas[i] * 3;
    gaps.push(Math.hypot(x[q] - x[p], x[q + 1] - x[p + 1], x[q + 2] - x[p + 2]));
  }
  return gaps.reduce((s, v) => s + v, 0) / gaps.length;
}

export async function main(device, args = []) {
  const sequence = option(args, "sequence",
    "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK");
  const alignment = option(args, "a3m", "/tools/fixtures/test.a3m");
  const seed = Number(option(args, "seed", "1"));
  const seeds = Number(option(args, "seeds", "1"));
  const sampler = option(args, "sampler", "diffusion-15");
  const foldBundle = option(args, "bundle", "/model-esmfold2-trunk-f32");
  const towerBundle = option(args, "esmc", "/model-esmc-600m-int3");

  const fold = reader(foldBundle);
  const tower = reader(towerBundle);
  const manifest = await fold.manifest();
  const towerTable = await tower.manifest();
  const towerManifest = towerTable.languageModel ?? {};
  const M = manifest.trunk;

  const shim = {};
  for (const name of [...SHIM_PAIR_TENSORS, "lm/norm/offset", "lm/projection/weights",
                      "lm/downproject/weights", "lm/downproject/bias"]) {
    shim[name] = await tower.read(name);
  }
  const [featuriser, inputsEmbedder, denoiser, encoder, decoder] = await Promise.all([
    featuriserWeights(fold.read),
    atomEncoderWeights(fold.read, "atom", M.atomBlocks),
    denoiserWeights(fold.read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(fold.read, "diffusionAtomEncoder", M.atomBlocks,
                       { withCoordinates: true }),
    atomDecoderWeights(fold.read, "diffusionAtomDecoder", M.atomBlocks),
  ]);
  denoiser.encoder = encoder;
  denoiser.decoder = decoder;
  const trunkBlocks = [];
  for (let layer = 0; layer < M.blocks; layer += 1) {
    trunkBlocks.push(await trunkBlockWeights(fold.read, layer));
  }
  const towerShared = {};
  for (const name of TOWER_SHARED) towerShared[name] = await tower.read(name);
  const allocator = new GpuBufferAllocator(device);
  const runTower = async (ids, sequenceId, onBlock) => {
    const engine = new EsmcTowerGpu(device, allocator);
    const result = await engine.run(ids, {
      rows: ids.length, model: towerManifest.width,
      heads: towerManifest.heads ?? towerManifest.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: towerManifest.layers, pair: M.pairChannels,
      residualScale: towerManifest.residualScale ?? 1,
    }, async (layer) => {
      const weights = {};
      for (const leaf of BLOCK_LEAVES) {
        weights[leaf] = await tower.read(`blocks/${layer}/${leaf}`, NARROW_LEAVES.has(leaf));
      }
      return weights;
    }, towerShared, { sequenceId, onBlock });
    return result.single;
  };

  const tokens = sequence.length;
  const text = await (await fetch(alignment)).text();
  const { profile, deletionMean, rows } = profileFromA3m(text, tokens);

  // ...the query one-hot, which is what upstream builds when there is no MSA
  // and the flag is off. Built here from the sequence rather than read out of
  // the features, so the arm is what it says it is.
  const onehot = new Float32Array(tokens * AATYPE_CLASSES);
  for (let token = 0; token < tokens; token += 1) {
    onehot[token * AATYPE_CLASSES + LETTER_TO_AATYPE[sequence.charCodeAt(token)]] = 1;
  }

  // 🔴 THE PROFILE HAS TO BE A PROFILE, or this measures a broken fixture. It
  // is a distribution per column, so each sums to one; and it must AGREE with
  // the query, or the alignment is not this sequence's.
  let columnSums = 0, agreeing = 0, peak = 0;
  for (let token = 0; token < tokens; token += 1) {
    let sum = 0, best = 0, at = 0;
    for (let c = 0; c < AATYPE_CLASSES; c += 1) {
      const v = profile[token * AATYPE_CLASSES + c];
      sum += v;
      if (v > best) { best = v; at = c; }
    }
    columnSums += sum;
    peak += best;
    if (at === LETTER_TO_AATYPE[sequence.charCodeAt(token)]) agreeing += 1;
  }
  const profileCheck = {
    meanColumnSum: Number((columnSums / tokens).toFixed(4)),
    meanPeak: Number((peak / tokens).toFixed(4)),
    modeIsTheQuery: `${agreeing} / ${tokens}`,
  };

  const armsFor = (armSeed) => [
    { name: "zeros", features: undefined, seed: armSeed },
    { name: "onehot", features: { profile: onehot }, seed: armSeed },
    { name: "msa", features: { profile, deletionMean }, seed: armSeed },
    { name: "scrambled", features: { profile: scramble(profile, tokens), deletionMean },
      seed: armSeed },
    { name: "seed", features: undefined, seed: armSeed + 100 },
  ];

  const rowsOut = [];
  for (let trial = 0; trial < seeds; trial += 1) {
    const armSeed = seed + trial;
    const folded = [];
    for (const arm of armsFor(armSeed)) {
      const out = await foldEsmfold2(device, {
        sequence, allocator, seed: arm.seed, sampler,
        shape: M,
        weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
        tower: runTower,
        ...(arm.features === undefined ? {} : { features: arm.features }),
      });
      folded.push({ arm: arm.name, out });
    }
    const base = folded[0].out;
    const alphas = alphaCarbons(base.features);
    for (const { arm, out } of folded) {
      const live = out.certainty === undefined ? []
        : [...out.certainty].filter((v) => v >= 0);
      rowsOut.push({
        seed: armSeed, arm,
        caCa: Number(caSpacing(out.coordinates, alphas).toFixed(3)),
        certainty: live.length === 0 ? null
          : Number((live.reduce((s, v) => s + v, 0) / live.length).toFixed(4)),
        movedFromZeros: arm === "zeros" ? 0
          : Number(displacement(out.coordinates, base.coordinates,
                                base.features.mask, base.atoms).toFixed(3)),
      });
    }
  }

  const meanFor = (name) => {
    const v = rowsOut.filter((r) => r.arm === name).map((r) => r.movedFromZeros);
    return Number((v.reduce((s, x) => s + x, 0) / v.length).toFixed(3));
  };
  const summary = {
    onehot: meanFor("onehot"), msa: meanFor("msa"),
    scrambled: meanFor("scrambled"), seedChange: meanFor("seed"),
  };
  // 🔴 THE PLUMBING CONTROL. Same seed, so a profile that never reached the
  // model would move the fold by EXACTLY zero on all three arms - which is also
  // what "the alignment does nothing" looks like if you only read the means.
  // The two are told apart here and nowhere else.
  const reached = summary.onehot > 0 && summary.msa > 0 && summary.scrambled > 0;
  // ...and the verdict: is a REAL alignment distinguishable from a permuted one?
  const informative = Math.abs(summary.msa - summary.scrambled)
    > 0.25 * summary.seedChange;

  // ...and how big the profile term even is, from the weights themselves.
  const zInit1 = featuriser.zInit1;
  const width = M.pairChannels;
  const columnRms = (from, to) => {
    let total = 0;
    for (let row = from; row < to; row += 1) {
      let squared = 0;
      for (let c = 0; c < width; c += 1) squared += zInit1[row * width + c] ** 2;
      total += Math.sqrt(squared / width);
    }
    return total / (to - from);
  };

  return {
    ok: true, sequence: tokens, alignmentRows: rows, sampler, seeds,
    profileCheck,
    rows: rowsOut,
    meanDisplacement: summary,
    profileReachedTheModel: reached,
    alignmentIsDistinguishableFromNoise: informative,
    zInit1: {
      aatypeRms: Number(columnRms(384, 417).toFixed(6)),
      profileRms: Number(columnRms(417, 450).toFixed(6)),
      ratio: Number((columnRms(417, 450) / columnRms(384, 417)).toFixed(4)),
    },
  };
}

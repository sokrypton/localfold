/**
 * The same per-pair features as probe-pae-from-distogram.js, off EF2-fast.
 *
 * 🔴 THE TWO GRIDS ARE DIFFERENT AND THE FEATURES MUST NOT BE. AF3's distogram
 * is 64 bins over 2-22 A; this one is 128 over a BORROWED 2-52 (the disabled
 * confidence head's range - see CONTACT_EDGES). A sigma read in BIN INDICES
 * would differ by a factor of two between them for the same physical spread,
 * so every moment here is computed in ANGSTROMS from that model's own bin
 * centres. That is what makes one fitted estimator applicable to both.
 *
 * 🔴 AND THE BIAS IS NOT IN THE LOGITS BUFFER. The projection has none and the
 * contact pass adds it as it reads, so a caller taking the logits away is
 * handed the bias separately and must add it - a softmax over the 128 numbers
 * without it is still a distribution, just the wrong one.
 *
 * There is no PAE to score against here: this checkpoint has no confidence
 * head, which is the whole reason for the exercise. The dump carries `pae` as
 * zeros and the analysis takes its target from the matching AF3 run.
 */
import {
  readTensor, readTensorAsFloat16,
} from "../../src/reference/dtype.js";
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";
import { foldEsmfold2 } from "../../src/esmfold2/fold.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../../src/esmfold2/language-pair-webgpu.js";
import { CONTACT_EDGES } from "../../src/esmfold2/distogram-webgpu.js";
import { representativeAtoms } from "../../src/esmfold2/fold.js";

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

/** Moments in ANGSTROMS, from this model's own uniform bin centres. */
function moments(logits, bias, tokens, bins, edges) {
  const width = (edges.maximum - edges.minimum) / bins;
  const centre = (b) => edges.minimum + width * (b + 0.5);
  const pairs = tokens * tokens;
  const mean = new Float32Array(pairs);
  const sigma = new Float32Array(pairs);
  const entropy = new Float32Array(pairs);
  const p = new Float64Array(bins);
  for (let index = 0; index < pairs; index += 1) {
    const base = index * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) {
      const v = logits[base + b] + bias[b];
      if (v > largest) largest = v;
    }
    let total = 0;
    for (let b = 0; b < bins; b += 1) {
      p[b] = Math.exp(logits[base + b] + bias[b] - largest);
      total += p[b];
    }
    let m = 0, m2 = 0, h = 0;
    for (let b = 0; b < bins; b += 1) {
      const q = p[b] / total, c = centre(b);
      m += q * c; m2 += q * c * c;
      if (q > 1e-12) h -= q * Math.log(q);
    }
    mean[index] = m;
    sigma[index] = Math.sqrt(Math.max(0, m2 - m * m));
    entropy[index] = h;
  }
  return { mean, sigma, entropy };
}

export async function main(device, args = []) {
  // 🔴 EVERY TARGET IN ONE INVOCATION, because the weights are the cost. A
  // fold here is seconds and loading 856 tensors is most of a minute, so a
  // shell loop spends nearly all of its time re-reading the same bundle - and
  // `node tools/gpu-chrome.mjs` sometimes does not exit, which in a loop stalls
  // every arm behind it and leaves orphaned browsers accumulating. One process,
  // one load, one browser.
  //
  //     --targets='mono|GWSTE...;split|GWST...:ELEK...'
  //
  // A chain break is ":" as everywhere else here, so targets are separated by
  // ";" and a name from its sequence by "|".
  const spec = option(args, "targets", "");
  const sequence = option(args, "sequence", "");
  if (spec === "" && sequence === "") {
    throw new Error("--targets= or --sequence= is required");
  }
  const seed = Number(option(args, "seed", "1"));
  // 🔴 THE LANGUAGE MODEL IS THIS MODEL'S "SINGLE SEQUENCE". AF2 and AF3 fold
  // without their alignment; ESM-C is where EF2-fast's evolutionary information
  // comes from, so switching it off is the same ablation - and it is the case a
  // confidence estimate has to survive, because the fold really does collapse.
  const noPlm = args.includes("--no-plm");
  // ...matrices are tokens^2 and a calibration run wants none of them.
  const summaryOnly = args.includes("--summary");
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

  const requested = spec === ""
    ? [["single", sequence]]
    : spec.split(";").filter((entry) => entry !== "")
        .map((entry) => {
          const at = entry.indexOf("|");
          if (at < 0) throw new Error(`--targets entry has no name: ${entry}`);
          return [entry.slice(0, at), entry.slice(at + 1)];
        });

  const out = [];
  for (const [name, text] of requested) {
    const result = await foldEsmfold2(device, {
      sequence: text, allocator, seed, sampler,
      entities: text,
      shape: M,
      weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
      tower: runTower,
      languageModel: !noPlm,
      alignedError: true,
      distogramLogits: !summaryOnly,
    });

    const tokens = result.tokens;

    // 🔴 A CALIBRATION RUN WANTS NO MATRICES. Summaries only, so a dozen folds
    // fit in one process and one stdout - and `distogramLogits` is left off,
    // which is what makes the fold cheap rather than what makes it different.
    if (summaryOnly) {
      const pae = result.alignedError;
      const off = [];
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) if (i !== j) off.push(pae[i * tokens + j]);
      }
      off.sort((a, b) => a - b);
      const live = [...result.certainty].filter((v) => v >= 0);
      let contacts = 0;
      for (let i = 0; i < tokens; i += 1) {
        for (let j = i + 6; j < tokens; j += 1) {
          if (result.contacts[i * tokens + j] > 0.5) contacts += 1;
        }
      }
      const summary = {
        name, tokens, languageModel: !noPlm,
        paeMean: Number((off.reduce((a, v) => a + v, 0) / off.length).toFixed(3)),
        paeMedian: Number(off[off.length >> 1].toFixed(3)),
        paeMin: Number(off[0].toFixed(3)),
        paeMax: Number(off[off.length - 1].toFixed(3)),
        certainty: live.length === 0 ? null
          : Number((live.reduce((a, v) => a + v, 0) / live.length).toFixed(4)),
        contacts,
      };
      out.push(summary);
      console.log(`${name}: ${tokens} tokens, pAE ${summary.paeMean}`
        + `, certainty ${summary.certainty}, ${contacts} contacts`);
      continue;
    }

    const bins = M.distogramBins;
    const { mean, sigma, entropy } = moments(result.distogram.logits,
                                             result.distogram.bias, tokens, bins,
                                             CONTACT_EDGES);

    // ...the same representative atoms the distogram is defined on.
    const rep = representativeAtoms(result.features, tokens);
    const x = result.coordinates;
    const observed = new Float32Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const a = rep[i] * 3, b = rep[j] * 3;
        observed[i * tokens + j] = Math.hypot(x[a] - x[b], x[a + 1] - x[b + 1],
                                              x[a + 2] - x[b + 2]);
      }
    }

    out.push({
      name, model: "ef2-fast", sequence: text.replace(/:/g, "").length,
      tokens, bins, seed, sampler,
      data: {
        pae: new Array(tokens * tokens).fill(0),   // no confidence head; see above
        sigma: [...sigma].map((v) => Number(v.toFixed(3))),
        mean: [...mean].map((v) => Number(v.toFixed(3))),
        entropy: [...entropy].map((v) => Number(v.toFixed(4))),
        observed: [...observed].map((v) => Number(v.toFixed(3))),
        asymId: [...result.features.asymId.slice(0, tokens)],
        residue: [...result.features.residueIndex.slice(0, tokens)],
      },
    });
    console.log(`${name}: ${tokens} tokens`);
  }
  return { ok: true, model: "ef2-fast", targets: out };
}

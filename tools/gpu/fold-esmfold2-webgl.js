/**
 * Fold ESMFold2 in WebGL2, from the two real weight bundles.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2-webgl.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2-webgl.js --tower-blocks=36 --steps=15
 *
 * 🔴 THE DEFAULTS ARE A TRUNCATED FOLD AND THE REPORT SAYS SO. A full ESMFold2
 * fold is 36 ESM-C blocks, four trunk loops of 24 blocks and eleven denoiser
 * calls, and this backend is about 17x slower than the WGSL one - so the
 * defaults run a PREFIX of the tower and fewer loops, which is enough to ask
 * whether the composition is right and not enough to ask whether the structure
 * is good. `--tower-blocks=36 --loops=3 --steps=15` is the real thing.
 *
 * 🔴 AND A CHAIN THAT IS NOT A CHAIN IS THE GATE, NOT pLDDT. The consecutive
 * CA-CA distances are asserted at a converged step count and pLDDT is reported
 * without being gated on - `fold-af2.js` learned that from a collapsed
 * 825-residue fold that climbed to pLDDT 69.
 *
 * 🔴 AND IT IS THE FIRST THING HERE TO READ TWO CHECKPOINTS AT ONCE: the tower's
 * shim is per folding model, so a bundle pair that was not exported together is
 * refused by name rather than measured.
 */
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { featuriseForEsmfold2 } from "../../shared/esmfold2/featurise.js";
import {
  atomDecoderWeights, atomEncoderWeights, confidenceHeadWeights,
  conditioningWeights, denoiserWeights, featuriserWeights, trunkBlockWeights,
} from "../../shared/esmfold2/weights.js";
import { webglExecution, webglProfile } from "./webgl-execution-options.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { foldEsmfold2 } from "../../webgl/esmfold2/model/fold.js";
import { SAMPLER_DEFAULTS } from "../../webgl/esmfold2/sampler.js";
import { SAMPLER_DEFAULTS as WGSL_SAMPLER_DEFAULTS } from "../../webgpu/esmfold2/fold.js";
import { uploadTriangleWeights, releaseTriangleWeights }
  from "../../webgl/af3/trunk/triangle.js";
import { uploadTransitionWeights, releaseTransitionWeights }
  from "../../webgl/af3/trunk/transition.js";

const DEFAULT_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Consecutive CA-CA distances, which is what says a chain is a chain. */
function backboneSpacing(coordinates, features) {
  const spacings = [];
  let previous = null;
  for (let atom = 0; atom < features.atoms; atom += 1) {
    if (!(features.mask[atom] > 0)) continue;
    let isAlpha = true;
    for (const [index, code] of [..."CA"].entries()) {
      if (features.refAtomNameChars[atom * 4 + index] !== code.charCodeAt(0) - 32) {
        isAlpha = false;
      }
    }
    if (features.refAtomNameChars[atom * 4 + 2] !== 0) isAlpha = false;
    if (!isAlpha) continue;
    const at = [coordinates[atom * 3], coordinates[atom * 3 + 1], coordinates[atom * 3 + 2]];
    if (previous !== null) {
      spacings.push(Math.hypot(at[0] - previous[0], at[1] - previous[1], at[2] - previous[2]));
    }
    previous = at;
  }
  spacings.sort((a, b) => a - b);
  return spacings;
}

export async function main(device, args) {
  const foldBundle = option(args, "bundle", "/model-esmfold2-int5").replace(/\/$/, "");
  const towerBundle = option(args, "tower", "/model-esmc-600m-int3").replace(/\/$/, "");
  const length = Number(option(args, "length", "20"));
  const sequence = option(args, "sequence", DEFAULT_SEQUENCE.slice(0, length));
  const towerBlocks = Number(option(args, "tower-blocks", "2"));
  const trunkBlocks = Number(option(args, "trunk-blocks", "4"));
  const loops = Number(option(args, "loops", "0"));
  const steps = Number(option(args, "steps", "15"));
  const seed = Number(option(args, "seed", "42"));

  // 🔴 THE TWO BACKENDS' SAMPLER CONSTANTS ARE HELD TOGETHER HERE, because they
  // are stated twice and every one of them differs from the AF3-shaped defaults
  // a port reaches for - sMax 160, rho 8, gamma_min 1.107, noise scale 0.901,
  // step scale 1.638. Nothing errors on the wrong set.
  for (const [key, value] of Object.entries(WGSL_SAMPLER_DEFAULTS)) {
    if (SAMPLER_DEFAULTS[key] !== value) {
      throw new Error(
        `the two backends disagree about the sampler's ${key}:`
        + ` WebGL2 ${SAMPLER_DEFAULTS[key]} against WebGPU ${value}`);
    }
  }

  const loadStart = performance.now();
  const fold = await HttpTensorStore.open(`${foldBundle}/manifest.json`);
  const tower = await HttpTensorStore.open(`${towerBundle}/manifest.json`);
  const manifest = fold.manifest;
  const towerTable = tower.manifest;
  const towerManifest = towerTable.languageModel ?? {};
  // 🔴 THE SHIM IS PER FOLDING MODEL, so a pair that was not exported together
  // is refused rather than measured.
  if (towerManifest.shim !== manifest.trunk?.source) {
    throw new Error(
      `the tower's shim is for ${towerManifest.shim} and the folding bundle is`
      + ` ${manifest.trunk?.source}; the shim is per model`);
  }
  const M = manifest.trunk;
  const read = (name) => fold.tensor(name);
  const [featuriser, atomEncoder, denoiser, diffusionEncoder, diffusionDecoder,
    conditioning] = await Promise.all([
    featuriserWeights(read),
    atomEncoderWeights(read, "atom", M.atomBlocks),
    denoiserWeights(read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(read, "diffusionAtomEncoder", M.atomBlocks,
      { withCoordinates: true }),
    atomDecoderWeights(read, "diffusionAtomDecoder", M.atomBlocks),
    conditioningWeights(read),
  ]);
  const trunk = [];
  for (let layer = 0; layer < Math.min(trunkBlocks, M.blocks); layer += 1) {
    trunk.push(await trunkBlockWeights(read, layer));
  }
  const confidence = await confidenceHeadWeights(read, manifest);
  if (confidence === null) {
    throw new Error(
      `${foldBundle} was exported without --confidence, so there is no head to run`
      + " - which is what biohub ships; this driver wants Synthyra's");
  }
  const towerNames = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
    "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
    "lm/downproject/bias", "lm/pair_mlp_1/weights", "lm/pair_mlp_1/bias",
    "lm/pair_mlp_2/weights", "lm/pair_mlp_2/bias", "lm/pair_norm/scale",
    "lm/pair_norm/offset"];
  const towerHost = {};
  for (const name of towerNames) towerHost[name] = await tower.tensor(name);
  for (let layer = 0; layer < towerBlocks; layer += 1) {
    for (const leaf of ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
      "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
      "ffn_norm/offset", "fc1/weights", "fc2/weights"]) {
      towerHost[`blocks/${layer}/${leaf}`] = await tower.tensor(`blocks/${layer}/${leaf}`);
    }
  }
  const loadMs = Math.round(performance.now() - loadStart);

  const features = featuriseForEsmfold2(sequence);

  const gl = createWebGL2Context();
  const execution = webglExecution(gl, args ?? []);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value === undefined || value === null) return undefined;
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    // 🔴 A STRING IS NOT A TREE. `confidenceHeadWeights` carries
    // `source` and `unusedInCheckpoint` beside the tensors, and
    // `Object.entries("abc")` is three single-character strings - so a generic
    // walk recurses on a string for ever and dies with "Maximum call stack size
    // exceeded", naming nothing. Anything that is not an array, a typed array
    // or a plain object is passed through.
    if (typeof value !== "object") return value;
    return Object.fromEntries(Object.entries(value)
      .map(([key, v]) => [key, uploadTree(v, `${label}.${key}`)])
      .filter(([, v]) => v !== undefined));
  };
  const uploadStart = performance.now();
  const uploadedTower = {};
  for (const [name, values] of Object.entries(towerHost)) {
    if (name === "lm/combine") continue;
    uploadedTower[name] = upload(values, `w.esmc.${name}`);
  }
  // 🔴 THE LOADERS' NAMES AND THE MODULES' NAMES ARE MAPPED HERE, EXPLICITLY.
  // `featuriserWeights` returns `relPos`, `recycleScale` and `distogramWeights`
  // where the modules take `relativePosition`, `recycle.scale` and
  // `distogram.weights` - one literal, in the one place that knows both.
  const weights = {
    featuriser: {
      relativePosition: upload(featuriser.relPos, "w.ef2.rel-pos"),
      tokenBonds: upload(featuriser.tokenBonds, "w.ef2.token-bonds"),
      zInit1: upload(featuriser.zInit1, "w.ef2.z-init-1"),
      zInit2: upload(featuriser.zInit2, "w.ef2.z-init-2"),
    },
    recycle: {
      scale: upload(featuriser.recycleScale, "w.ef2.recycle-scale"),
      offset: upload(featuriser.recycleOffset, "w.ef2.recycle-offset"),
      projection: upload(featuriser.recycleProjection, "w.ef2.recycle-projection"),
    },
    distogram: {
      weights: upload(featuriser.distogramWeights, "w.ef2.distogram"),
      bias: upload(featuriser.distogramBias, "w.ef2.distogram-bias"),
    },
    atomEncoder: uploadTree(atomEncoder, "w.ef2.atom"),
    diffusionEncoder: uploadTree(diffusionEncoder, "w.ef2.diffusion-encoder"),
    diffusionDecoder: uploadTree(diffusionDecoder, "w.ef2.diffusion-decoder"),
    denoiser: {
      ...uploadTree({ ...denoiser, conditioning: undefined }, "w.ef2.denoiser"),
      conditioning: uploadTree(conditioning, "w.ef2.conditioning"),
    },
    hostFourier: {
      weights: conditioning.fourierWeights,
      offsets: conditioning.fourierOffsets,
    },
    trunkBlocks: trunk.map((block) => ({
      triangleMultiplicationOutgoing: uploadTriangleWeights(
        execution, block.triangleMultiplicationOutgoing),
      triangleMultiplicationIncoming: uploadTriangleWeights(
        execution, block.triangleMultiplicationIncoming),
      pairTransition: uploadTransitionWeights(execution, block.pairTransition),
    })),
    confidence: {
      ...uploadTree({ ...confidence, blocks: undefined }, "w.ef2.confidence"),
      pairChannels: confidence.pairChannels,
      singleChannels: confidence.singleChannels,
      singleInputs: confidence.singleInputs,
      plddtBins: confidence.plddtBins,
      paeBins: confidence.paeBins,
      maxAtomsPerToken: confidence.maxAtomsPerToken,
    },
    confidenceBlocks: confidence.blocks.map((block) => ({
      triangleMultiplicationOutgoing: uploadTriangleWeights(
        execution, block.triangleMultiplicationOutgoing),
      triangleMultiplicationIncoming: uploadTriangleWeights(
        execution, block.triangleMultiplicationIncoming),
      pairTransition: uploadTransitionWeights(execution, block.pairTransition),
    })),
  };
  const uploadMs = Math.round(performance.now() - uploadStart);
  const weightMegabytes = Number((execution.stats().bytes / (1 << 20)).toFixed(1));

  const stages = [];
  const foldStart = performance.now();
  const result = foldEsmfold2(execution, {
    features,
    shape: {
      pairChannels: M.pairChannels, singleInputs: M.singleInputs,
      tokenChannels: M.tokenChannels, tokenChannels2: M.tokenChannels2,
      tokenHeads: M.tokenHeads, transitionMultiplier: M.transitionMultiplier,
      atomChannels: M.atomChannels, atomHeads: M.atomHeads,
      atomBlocks: M.atomBlocks, atomWindow: M.atomWindow,
      distogramBins: M.distogramBins, sigmaData: M.sigmaData,
      rope: {}, attentionPrecision: "f32",
    },
    tower: uploadedTower,
    towerDims: {
      model: towerManifest.width,
      heads: towerManifest.heads ?? towerManifest.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: towerManifest.layers,
      pair: M.pairChannels,
    },
    towerCombine: towerHost["lm/combine"],
    towerBlocks,
    weights,
    loops, steps, seed,
    onStage: (name, detail) => stages.push({
      name, ...detail, at: Math.round(performance.now() - foldStart),
    }),
  });
  const foldMs = Math.round(performance.now() - foldStart);
  const peak = execution.stats();
  for (const tensor of held) execution.release(tensor);
  for (const group of [weights.trunkBlocks, weights.confidenceBlocks]) {
    for (const block of group) {
      releaseTriangleWeights(execution, block.triangleMultiplicationOutgoing);
      releaseTriangleWeights(execution, block.triangleMultiplicationIncoming);
      releaseTransitionWeights(execution, block.pairTransition);
    }
  }
  const memory = { ...execution.stats(), peakBytes: peak.peakBytes };

  const spacings = backboneSpacing(result.coordinates, features);
  const median = spacings.length === 0 ? NaN : spacings[spacings.length >> 1];
  let contacts = 0;
  for (let index = 0; index < result.contacts.length; index += 1) {
    if (result.contacts[index] > 0.5) contacts += 1;
  }
  const failures = [];
  if (!(spacings.length > 0)) failures.push("no alpha carbons found to measure");
  // At a converged step count the backbone is a backbone.
  if (steps >= 11 && !(median > 3.4 && median < 4.2)) {
    failures.push(`the chain is not a chain: median CA-CA ${median.toFixed(3)} A, not ~3.8`);
  }
  if (result.coordinates.some((value) => !Number.isFinite(value))) {
    failures.push("the coordinates are not all finite");
  }
  if (!Number.isFinite(result.meanPlddt)) failures.push("pLDDT is not finite");
  if (memory.live !== 0) failures.push(`${memory.live} tensors left live`);

  return {
    bundles: { fold: foldBundle, tower: towerBundle },
    model: manifest.trunk?.source ?? null,
    sequence: sequence.length,
    tokens: result.tokens, atoms: result.atoms,
    configuration: {
      towerBlocks, towerLayers: towerManifest.layers,
      trunkBlocks: trunk.length, trunkTotal: M.blocks,
      loops: result.loops, steps: result.steps, seed,
      fullTower: towerBlocks >= towerManifest.layers,
      fullTrunk: trunk.length >= M.blocks,
    },
    milliseconds: { load: loadMs, upload: uploadMs, fold: foldMs },
    weightMegabytes,
    caSpacing: spacings.length === 0 ? null : {
      median: Number(median.toFixed(3)),
      lowest: Number(spacings[0].toFixed(3)),
      highest: Number(spacings[spacings.length - 1].toFixed(3)),
      count: spacings.length,
    },
    meanPlddt: Number.isFinite(result.meanPlddt) ? Number(result.meanPlddt.toFixed(3)) : null,
    // 🔴 THE RANGE AND NOT ONLY THE MEAN, because a mean cannot tell a
    // confident fold from a kernel returning a constant. `fold-esmfold2.js`
    // reports min and max and this did not, which is why "mean pLDDT 57.4" sat
    // in webgl/README.md for a whole campaign without anyone asking whether the
    // per-residue values varied at all.
    minPlddt: result.plddt.length === 0 ? null
      : Number(Math.min(...result.plddt).toFixed(3)),
    maxPlddt: result.plddt.length === 0 ? null
      : Number(Math.max(...result.plddt).toFixed(3)),
    ptm: Number.isFinite(result.ptm) ? Number(result.ptm.toFixed(4)) : null,
    iptm: Number.isFinite(result.iptm) ? Number(result.iptm.toFixed(4)) : null,
    contactsOverHalf: contacts,
    stages,
    memory,
    // 🔴 `--profile` BUILT THE TABLE AND NOBODY PRINTED IT. `webglExecution`
    // honours the flag, so every dispatch in this tool has been paying the
    // one-texel readback barrier under `--profile` and reporting nothing - and
    // this is the model whose trunk is 53 s of a 62 s fold, so it is the one
    // most worth ranking. Read it with the barrier in mind: it is about 0.9 ms
    // a call here, so a kernel's share must be corrected for its CALL COUNT
    // before it is believed (see webgl/README.md's note on `linear.repack`).
    profile: webglProfile(execution, args ?? []),
    verdict: failures.length === 0
      ? `ok: ${result.tokens} tokens folded in WebGL2`
        + `${towerBlocks >= towerManifest.layers ? "" : " (a tower PREFIX)"}`
        + `${trunk.length >= M.blocks ? "" : " (a trunk PREFIX)"}`
        + ` at ${result.steps} steps`
      : `FAILED: ${failures.join("; ")}`,
  };
}

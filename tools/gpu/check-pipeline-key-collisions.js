/**
 * Do two models' trunks collide in ONE pipeline cache - the page's case?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-pipeline-key-collisions.js \
 *       --models=/model-af3-int5/manifest.json,/model-rosettafold3-int5/manifest.json
 *
 * 🔴 EVERY MODEL GATE HERE SPAWNS A FRESH BROWSER PER MODEL, SO NOTHING TESTED
 * THIS. `check-template-path.mjs`, `check-ligand-path.mjs`, `check-stock-flags`
 * and `check-portable-limits` all loop models by launching `gpu-chrome.mjs`
 * again - one process, one device, one `ComputePipelineCache` each. The PAGE
 * does the opposite: a visitor who folds with AlphaFold 3 and then switches to
 * RoseTTAFold3 reuses the cache, so a key that does not name its shader hands
 * the second model the first model's kernel. That is `LOCALFOLD_STOCK_FLAGS`'s
 * shape again - the configuration every gate checks was not the one that ships.
 *
 * It found one on its first run, reported from a real session:
 *
 *     af3-msa:59:128:64:128:0.00001:fast:false:msa:keyMask
 *     line 8 of 35: "const DIMENSION: u32 = 8u;" against "...= 32u;"
 *
 * Every kernel in msa-attention-webgpu.js is built from one `common` preamble,
 * so all of them embed HEADS and DIMENSION - even `keyMask`, thirteen lines
 * that read neither - while the stack keyed on the channel widths alone.
 * AlphaFold 3 and openbind0 are msaChannels 64 with 8 heads of dimension 8;
 * RoseTTAFold3 and boltz2 are the same 64 and 8 with dimension 32. Identical
 * key, different text.
 *
 * 🔴 THE ORDER MATTERS AND BOTH ARE RUN. A collision is only reported by the
 * SECOND compile, so a pair that is clean one way round is not evidence; this
 * runs the list forwards and then backwards in the same cache.
 *
 * 🔴 AND IT IS A COMPILE TEST, NOT A FOLD. One short trunk pass per model at 59
 * tokens is enough to build every trunk pipeline, which is what a key can
 * collide in; it says nothing about whether either model folds well, and
 * `test:template` and `test:ligand` are what say that.
 */
import { featuriseProtein } from "../../src/af3/featurise/featurise.js";
import { buildTargetFeat } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk/trunk-webgpu.js";
import { af3Dialect, openAf3Store, trunkWeights } from "../../src/af3/weights/weights.js";
import { targetFeatureWeights } from "../../src/af3/weights/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "ACDEFGHIKLMNPQRSTVWY";

async function trunkPass(device, model, tokens, rows) {
  const sequence = Array.from({ length: tokens },
    (_, i) => ALPHABET[i % ALPHABET.length]).join("");
  const batch = featuriseProtein(sequence, {});
  const store = await openAf3Store(model);
  const trunk = await trunkWeights(store, undefined, undefined, { allowPrefix: true });
  const targetFeat = await buildTargetFeat(batch, await targetFeatureWeights(store), device);
  const msa = new Int32Array(rows * tokens);
  const deletionMatrix = new Float32Array(rows * tokens);
  const msaMask = new Float32Array(rows * tokens).fill(1);
  for (let row = 0; row < rows; row += 1) {
    for (let t = 0; t < tokens; t += 1) msa[row * tokens + t] = (batch.msa[t] + row) % 20;
  }
  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const gpu = new Af3TrunkGpu(device, {});
  // The MSA stack's own first block - `msaBlocks`, not `pairformerBlocks`: the
  // head shape that collided is the MSA attention's, and the two stacks do not
  // share it (af3 is msa 8x8 against a pairformer 4x32).
  const sample = trunk.msaBlocks[0];
  const pairChannels = sample.pairAttention1.heads * sample.pairAttention1.dimension;
  await gpu.run({
    tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
    msaRows: msa, deletionMatrix, msaMask, bondMatrix: batch.bondMatrix,
    pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * pairChannels),
    previousSingle: new Float32Array(tokens * 384),
    contactClasses: new Int32Array(tokens),
  }, trunk, af3Dialect(store));
  return { msaHeads: sample.msaAttention1.heads,
           msaDimension: sample.msaAttention1.dimension, pairChannels };
}

export async function main(device, args) {
  const models = option(args, "models",
    "/model-af3-int5/manifest.json,/model-rosettafold3-int5/manifest.json")
    .split(",").map((m) => m.trim()).filter((m) => m !== "");
  const tokens = Number(option(args, "tokens", "59"));
  const rows = Number(option(args, "msa", "128"));
  if (models.length < 2) {
    throw new Error("this gate needs at least two models in ONE process; "
      + "with one it can never see a collision and would pass by finding nothing");
  }

  const shapes = {};
  const order = [...models, ...[...models].reverse()];
  for (const model of order) {
    const name = model.replace(/^\/model-|\/manifest\.json$/g, "");
    try {
      shapes[name] = await trunkPass(device, model, tokens, rows);
    } catch (error) {
      // A collision is what this is for; anything else is the run failing.
      if (String(error?.message ?? error).includes("pipeline cache key collision")) {
        throw new Error(`${name} collided in a cache ${order.indexOf(model)} model(s) `
          + `deep: ${error.message}`);
      }
      throw error;
    }
  }
  console.log(`${models.length} models, both orders, one pipeline cache: no collision`);
  for (const [name, s] of Object.entries(shapes)) {
    console.log(`  ${name.padEnd(24)} msa ${s.msaHeads}x${s.msaDimension}`
      + `  pair ${s.pairChannels}`);
  }
  return { models, tokens, rows, shapes, collision: null };
}

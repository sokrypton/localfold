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
import { buildTargetFeat, foldBatch } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk/trunk-webgpu.js";
import { af3BatchFromA3m } from "../../src/af3/featurise/batch.js";
import {
  af3Dialect, confidenceWeights, openAf3Store, openddeConfidenceWeights,
  structuralExpanderWeights, structuralRefinerWeights, trunkDepths, trunkWeights,
} from "../../src/af3/weights/weights.js";
import { atomReference, diffusionWeights, targetFeatureWeights }
  from "../../src/af3/weights/diffusion-weights.js";
import { dialectFor, featuriserDialect } from "../../src/af3/dialect.js";

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

/**
 * A whole fold, which is the stage set the PAGE compiles.
 *
 * 🔴 THE TRUNK IS NOT THE WHOLE CACHE. `--stage=trunk` compiles the embedder,
 * template, MSA stack, pairformer and distogram and nothing else; a page also
 * builds the diffusion conditioning, the atom encoder and decoder, the token
 * transformer and the confidence head, and every one of those is keyed the same
 * way and can collide the same way. Four sampler steps is enough - a pipeline
 * is compiled on the first one - so this costs a fold's weights, not a fold.
 */
async function foldPass(device, model, sequence, steps) {
  const store = await openAf3Store(model);
  const dialect = af3Dialect(store);
  const batch = af3BatchFromA3m(sequence, null,
    { ...featuriserDialect(dialectFor(dialect.model ?? dialect.name ?? "alphafold3")) }).batch;
  const depths = await trunkDepths(store);
  const trunk = await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks,
                                   { allowPrefix: true });
  const weights = {
    trunk,
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
    // 🔴 OpenDDE's SECOND TOKEN SPACE, WITHOUT WHICH IT CANNOT BE SWEPT AT ALL.
    // It runs a structural-token expander and refiner and its OWN confidence
    // head, so `foldBatch` with AF3's weight set dies before compiling any of
    // them - which is why the first version of this gate covered opendde at
    // `--stage=trunk` only and left its diffusion and confidence stacks
    // unswept. The dialect says which set a bundle wants; AF3 and OpenBind-0
    // take the other branch and are the control that this branch is not simply
    // never taken.
    ...(trunk.dialect.structuralTokens
      ? { expander: await structuralExpanderWeights(store),
          refiner: await structuralRefinerWeights(store),
          openddeConfidence: await openddeConfidenceWeights(store) }
      // 🔴 NO `.catch`. This was written to tolerate a head the loader cannot
      // read, on the strength of a note about boltz2 keeping its confidence
      // under its own scope - and boltz2 loads here and scores, so the
      // tolerance was protecting nothing while silently turning a missing head
      // into an unswept one. A bundle whose head will not load should fail.
      : { confidence: await confidenceWeights(store) }),
  };
  const sample = trunk.msaBlocks[0];
  // 🔴 DIFFUSION, NOT FLOW: rosettafold3 REFUSES flow outright - see
  // `noFlowSampler` - and this gate must run every checkpoint. Four steps
  // will not converge and does not need to: a pipeline is compiled on the
  // first one, and nothing here reads the structure.
  const result = await foldBatch(device, batch, weights,
                                 { steps, mode: "diffusion", recycles: 0 });
  return { msaHeads: sample.msaAttention1.heads,
           msaDimension: sample.msaAttention1.dimension,
           pairChannels: sample.pairAttention1.heads * sample.pairAttention1.dimension,
           // 🔴 REPORTED SO THE COVERAGE IS STATED AND NOT IMPLIED. `structural`
           // says the expander, refiner and OpenDDE's OWN confidence head were
           // the stacks that ran; `scored` says a head produced a number, which
           // is how a silently skipped head is told from a swept one.
           structural: trunk.dialect.structuralTokens === true,
           scored: Number.isFinite(result.meanPlddt) };
}

export async function main(device, args) {
  // 🔴 THE WHOLE LINEAGE BY DEFAULT, BECAUSE A PAIR PROVES NOTHING ABOUT A
  // THIRD. af3 and rosettafold3 alone were clean on the confidence head; adding
  // intellifold2 found `af3-confidence:...:embedProject` at "C_Z 128 against
  // 512". A bundle this box lacks is a SKIP and not a failure - that is
  // test:ligand's convention, and openbind0 is float32 here.
  const requested = option(args, "models", [
    "/model-af3-int5/manifest.json",
    "/model-protenix2-int5/manifest.json",
    "/model-boltz2-int5/manifest.json",
    "/model-intellifold2-int5/manifest.json",
    "/model-rosettafold3-int5/manifest.json",
    "/model-openbind0-f32/manifest.json",
    "/model-opendde-int5/manifest.json",
  ].join(",")).split(",").map((m) => m.trim()).filter((m) => m !== "");
  const models = [];
  for (const model of requested) {
    const head = await fetch(model, { method: "GET" });
    if (head.ok) models.push(model);
    else console.log(`  skipped ${model} - not on this box (${head.status})`);
  }
  const tokens = Number(option(args, "tokens", "59"));
  const rows = Number(option(args, "msa", "128"));
  // 🔴 `fold` IS THE DEFAULT BECAUSE `trunk` CANNOT SEE HALF OF IT. The
  // confidence collision this gate found is in a stage no trunk pass
  // compiles: af3 and intellifold2 were clean at stage=trunk and collided
  // on `af3-confidence:...:embedProject` at stage=fold. A gate that cannot
  // reach the bug it was built for is not a gate. `--stage=trunk` is the
  // fast arm for bisecting, not the one to run.
  const stage = option(args, "stage", "fold");
  const steps = Number(option(args, "steps", "4"));
  if (models.length < 2) {
    throw new Error(`this gate needs at least two models in ONE process and this `
      + `box has ${models.length} of ${requested.length}; with one it can never see `
      + `a collision and would pass by finding nothing`);
  }
  const sequence = Array.from({ length: tokens },
    (_, i) => ALPHABET[i % ALPHABET.length]).join("");

  const shapes = {};
  const order = [...models, ...[...models].reverse()];
  for (const model of order) {
    const name = model.replace(/^\/model-|\/manifest\.json$/g, "");
    try {
      shapes[name] = stage === "fold"
        ? await foldPass(device, model, sequence, steps)
        : await trunkPass(device, model, tokens, rows);
    } catch (error) {
      // A collision is what this is for; anything else is the run failing.
      if (String(error?.message ?? error).includes("pipeline cache key collision")) {
        throw new Error(`${name} collided in a cache ${order.indexOf(model)} model(s) `
          + `deep: ${error.message}`);
      }
      throw error;
    }
  }
  // 🔴 A STACK THAT DID NOT RUN IS NOT SWEPT, AND THAT REGRESSES SILENTLY.
  // `foldBatch` skips a head whose weights are absent and returns a structure
  // all the same, which is exactly how OpenDDE's diffusion and confidence sat
  // outside this gate: its weight set needs an expander, a refiner and its own
  // head, and without them the fold still finished. So the fold arm insists
  // every model produced a confidence number.
  if (stage === "fold") {
    const silent = Object.entries(shapes).filter(([, s]) => !s.scored).map(([n]) => n);
    if (silent.length > 0) {
      throw new Error(`no confidence number from ${silent.join(", ")} - that head did `
        + `not run, so its pipelines were never compiled and nothing here swept them`);
    }
  }
  console.log(`${models.length} models, both orders, one pipeline cache, `
    + `stage=${stage}: no collision`);
  for (const [name, s] of Object.entries(shapes)) {
    console.log(`  ${name.padEnd(24)} msa ${s.msaHeads}x${s.msaDimension}`
      + `  pair ${String(s.pairChannels).padEnd(4)}`
      + (stage === "fold"
        ? `  ${s.structural ? "structural + own confidence" : "af3 confidence   "}`
          + `  ${s.scored ? "scored" : "NO SCORE"}`
        : ""));
  }
  return { models, tokens, rows, stage, shapes, collision: null };
}

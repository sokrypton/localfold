/**
 * Does a fold that FAILS still leave its trunk behind for the retry?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-trunk-reuse-after-failure.js
 *
 * WHY IT EXISTS. The page offers a "Fold anyway" button when its memory ceiling
 * refuses a fold, and the ceiling refuses at the point of allocation - which
 * for a large structure is in the SAMPLER, after the trunk has already run. The
 * trunk was cached only when foldAf3 RESOLVED, so the exception threw it away
 * and the retry re-ran every pass that had already succeeded.
 *
 * 🔴 THE FIX IS A CALLBACK, NOT A CATCH. fold.js emits the reusable trunk on
 * its `trunk-done` stage, so a caller holds it before anything downstream has
 * had the chance to fail - which covers an abort and any other later error, not
 * just the memory one that prompted it.
 *
 * WHAT IS CHECKED: that `onTrunk` fires at all, that it fires BEFORE the
 * sampler runs, that a fold thrown away mid-sampler still yielded one, and that
 * feeding it back skips the trunk entirely - counted in recycle stages, not
 * guessed from a clock.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const DEFAULT = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";

export async function main(device, args) {
  const recycles = Number(option(args, "recycles", "2"));
  const steps = Number(option(args, "steps", "4"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };
  const batch = featuriseProtein(option(args, "sequence", DEFAULT), {});

  /** One fold, counting the stages that mean real trunk work. */
  const run = async (reuse, failAtStep) => {
    let recycleStages = 0;
    let handed;
    let handedBeforeSampler = true;
    let sawSampler = false;
    let threw = null;
    const started = performance.now();
    try {
      await foldBatch(device, batch, weights, {
        mode: "flow", steps, recycles, seed: 5, reuse,
        onStage: (name, detail) => {
          if (name === "recycle") recycleStages += 1;
          if (name === "trunk-done") {
            handed = detail.reusable;
            if (sawSampler) handedBeforeSampler = false;
          }
        },
        onStep: ({ step }) => {
          sawSampler = true;
          // 🔴 A THROW FROM INSIDE THE SAMPLER, which is where the memory
          // ceiling actually refuses. Failing before the trunk would prove
          // nothing: there would be no trunk to keep.
          if (failAtStep !== undefined && step >= failAtStep) {
            throw new Error("simulated memory ceiling");
          }
        },
      });
    } catch (cause) {
      threw = String(cause.message ?? cause);
    }
    return {
      recycleStages,
      handedTrunk: handed !== undefined,
      handedBeforeSampler,
      trunkRecycles: handed?.recycles ?? null,
      threw,
      seconds: Number(((performance.now() - started) / 1000).toFixed(2)),
      reusable: handed,
    };
  };

  // 1. A fold that dies inside the sampler, as the ceiling makes it.
  const failed = await run(undefined, 2);
  // 2. The retry, handed what the failed one left behind.
  const retried = await run(failed.reusable, undefined);
  // 3. The control: the same retry with nothing to reuse.
  const fromScratch = await run(undefined, undefined);

  return {
    recycles,
    expectedRecycleStages: recycles + 1,
    failed: { ...failed, reusable: undefined },
    retried: { ...retried, reusable: undefined },
    fromScratch: { ...fromScratch, reusable: undefined },
    verdict: {
      failedStillHandedTrunk: failed.handedTrunk && failed.threw !== null,
      handedBeforeSampler: failed.handedBeforeSampler,
      failedRanTheWholeTrunk: failed.recycleStages === recycles + 1,
      // The whole point: the retry does NO trunk work.
      retryRanNoTrunk: retried.recycleStages === 0,
      controlRanTheWholeTrunk: fromScratch.recycleStages === recycles + 1,
      secondsSaved: Number((fromScratch.seconds - retried.seconds).toFixed(2)),
    },
  };
}

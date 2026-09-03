/**
 * Do recycles help a COMPLEX, or entrench a wrong interface?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-recycles-on-complexes.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-recycles-on-complexes.js --steps=160
 *
 * WHY IT EXISTS. The page's recycle dial now defaults to 3, which is what
 * AlphaFold itself ships and what a monomer wants. A report from a real
 * two-chain target went the other way: 0 recycles, diffusion at 160 steps and
 * an MSA capped at 128 rows was what made it come out right. One target is not
 * a default, but a default this repo changed on convention rather than on
 * measurement is exactly the kind that should be checked.
 *
 * 🔴 THERE IS A MECHANISM FOR IT GOING EITHER WAY, WHICH IS WHY IT NEEDS
 * MEASURING. A recycle feeds the trunk's own pair representation back in, so it
 * sharpens whatever the trunk already believes. On a monomer that is more
 * evidence about a fold it is mostly right about. On a complex it can be more
 * evidence about an interface it placed wrongly on the first pass, and the
 * failure mode of recycling is a CONFIDENT wrong answer rather than an
 * uncertain one.
 *
 * 🔴 IT MUST RUN AT A CONVERGED STEP COUNT. probe-flow-sigma-by-size.js found
 * AF3's sampler badly under-stepped at the page's 20, so a recycle sweep there
 * would be measuring the sampler's noise floor and calling it recycling. The
 * default here is 80, which is where that sweep found the sampler landing.
 *
 * WHAT IT FOUND: RECYCLING HELPS, AND THE DEFAULT OF 3 SURVIVES. Diffusion at
 * 80 steps, two seeds, ipTM by recycle count:
 *
 *     GCN4 dimer    0.726  0.710  0.731      flat, inside the seed range
 *     Top7 dimer    0.735  0.780  0.797
 *     MRR dimer     0.115  0.128  0.173
 *     Top7+DES58    0.338  0.601  0.584
 *     Top7 monomer  pTM 0.707  0.768  0.768  (the control)
 *
 * Four of five improve and the fifth does not move; pLDDT and pTM go the same
 * way. Most of the gain is in the FIRST recycle - Top7+DES58 goes 0.338 to
 * 0.601 on one - and 1 to 3 is worth little.
 *
 * 🔴 IT IS SINGLE-SEQUENCE, WHICH IS THE LIMIT OF WHAT IT SAYS. The mechanism
 * for recycling HURTING a complex needs the trunk to have a wrong belief worth
 * entrenching, and with no alignment there is little for it to entrench. The
 * aligned version of this question is
 * tools/gpu/probe-msa-depth-on-complexes.js, which crosses depth with recycles
 * on a real alignment - and there the recycle effect is inside the seed range
 * at every depth.
 *
 * 🔴 AND IT REPORTS ipTM AND pTM SEPARATELY. The whole question is whether
 * recycling helps the CHAINS while hurting their arrangement, and a combined
 * score would average exactly that away.
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

const GCN4 = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER";
const MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";
const DES58 = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const TOP7 = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE";

/** A complex that works, two that do not, and a monomer as the control. */
const PANEL = [
  ["GCN4 dimer", `${GCN4}:${GCN4}`],
  ["Top7 dimer", `${TOP7}:${TOP7}`],
  ["MRR dimer", `${MRR}:${MRR}`],
  ["Top7+DES58", `${TOP7}:${DES58}`],
  // 🔴 THE CONTROL. If recycling helps here and hurts the complexes, that is
  // the result; if it does nothing anywhere, the panel is too easy.
  ["Top7 monomer", TOP7],
];

const RECYCLES = [0, 1, 3];
const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

export async function main(device, args) {
  const steps = Number(option(args, "steps", "80"));
  const mode = option(args, "mode", "diffusion");
  const seeds = option(args, "seeds", "1,2").split(",").map(Number);
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  const rows = [];
  for (const [name, sequence] of PANEL) {
    const batch = featuriseProtein(sequence, {});
    const row = { name, tokens: batch.tokens, arms: {} };
    // 🔴 THE TRUNK IS CARRIED FORWARD, NOT REBUILT. `reuse` continues a fold
    // from a previous one's trunk when the recycle count only goes UP, so
    // recycles 0, 1 and 3 cost four trunk passes between them rather than
    // eight - and, more importantly, arm 3 is arm 0 recycled rather than an
    // independent run that might differ for another reason.
    let reuse;
    for (const recycles of RECYCLES) {
      const runs = [];
      for (const seed of seeds) {
        const result = await foldBatch(device, batch, weights, {
          mode, steps, recycles, seed, reuse,
        });
        // Only the first seed's trunk is kept: the others reuse it, and the
        // trunk does not depend on the sampler's seed anyway.
        if (seed === seeds[0]) reuse = result.reusable;
        runs.push(result);
      }
      const iptm = runs.map((r) => r.iptm).filter(Number.isFinite);
      row.arms[`recycles ${recycles}`] = {
        iptm: iptm.length === 0 ? null : Number(mean(iptm).toFixed(3)),
        iptmRange: iptm.length === 0 ? null
          : [Number(Math.min(...iptm).toFixed(3)), Number(Math.max(...iptm).toFixed(3))],
        ptm: Number(mean(runs.map((r) => r.ptm)).toFixed(3)),
        meanPlddt: Number(mean(runs.map((r) => r.meanPlddt)).toFixed(1)),
      };
    }
    rows.push(row);
  }
  return { mode, steps, seeds, rows };
}

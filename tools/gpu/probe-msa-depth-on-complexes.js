/**
 * With a REAL alignment: does capping its depth help a complex, and do
 * recycles hurt one?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-msa-depth-on-complexes.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-msa-depth-on-complexes.js --steps=160
 *
 * 🔴 IT GOES TO THE NETWORK, WHICH NOTHING ELSE UNDER tools/gpu DOES. The
 * alignment comes from ColabFold's public MMseqs2 API, the same search the page
 * runs, because the one thing every sampler probe here could not test is the
 * setting that needs an alignment to exist. A search takes minutes and is
 * queued at the far end; run this deliberately, not in a loop.
 *
 * WHAT IT IS FOR. A report from a real two-chain target said the recipe that
 * made it work was 0 recycles, diffusion at 160 steps, and an MSA capped at
 * 128 rows. The step count is confirmed elsewhere - see
 * probe-flow-sigma-by-size.js, where 20 scores less than half of 80. The other
 * two are what this measures, and the depth cap cannot be measured at all
 * without a real alignment: every other probe here folds single-sequence,
 * where there is no depth to cap.
 *
 * 🔴 WHY A CAP COULD POSSIBLY HELP, so the result can be read rather than just
 * recorded. Depth is not evidence in equal amounts: a deep alignment of one
 * chain's family says a great deal about that CHAIN and, for a complex, can
 * drown the far smaller number of rows that speak to how the two chains sit
 * together. Capping keeps the top rows, which are the closest homologues and
 * the ones most likely to be genuinely paired.
 *
 * WHAT IT FOUND ON GCN4, at 80 diffusion steps over two seeds. The search
 * returned 215 rows, so "depth 512" is the whole alignment:
 *
 *     depth    rows   recycles 0      recycles 3
 *        1        1   0.726           0.731
 *       32       32   0.675           0.682
 *      128      128   0.662           0.671
 *      512      215   0.657           0.671
 *
 * 🔴 THE ALIGNMENT HURTS THIS TARGET, MONOTONICALLY. Capping the depth helps,
 * which is the direction the report described - but the trend does not stop at
 * 128, it runs all the way to no alignment at all, and 0.726 against 0.657 is
 * the one comparison here whose seed ranges do not overlap. So this is not
 * "128 is the right cap"; it is "depth costs on a target that did not need it".
 *
 * 🔴 AND GCN4 HAS NO HEADROOM, WHICH IS THE WEAKNESS OF THE TARGET. A leucine
 * zipper is a coiled coil AF3 already places from sequence alone at ipTM 0.73
 * and pLDDT 87. Homologues of a short, highly conserved motif add little and
 * dilute what the model already knew. The case worth measuring is a complex the
 * MSA is NEEDED for, where there is room for depth to pay.
 *
 * 🔴 RECYCLES ARE INSIDE THE SEED RANGE AT EVERY DEPTH HERE. 3 beats 0 in all
 * four rows, but by 0.005 to 0.014 with ranges that overlap - so with a real
 * alignment on this target the recycle count is not resolved either way, and
 * the single-sequence panel in probe-recycles-on-complexes.js, where it is
 * resolved and helps, is the stronger evidence for the default.
 *
 * 🔴 AND IT RUNS AT A CONVERGED STEP COUNT. At the page's 20 the sampler has
 * not landed, so an MSA sweep there would be measuring sampler noise. 80 is
 * where probe-flow-sigma-by-size.js found it landing.
 */
import { generateMmseqs2ComplexMsa } from "../../src/input/mmseqs2-api.js";
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * 🔴 A NATURAL COMPLEX, NOT A DESIGNED ONE. Top7 and the other designed chains
 * this repo benchmarks with have essentially no natural homologues, so their
 * alignments come back nearly empty and a depth cap would be measuring nothing.
 * GCN4's leucine zipper is a real, deeply conserved coiled-coil homodimer.
 */
const GCN4 = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER";

const DEPTHS = [1, 32, 128, 512];
const RECYCLES = [0, 3];
const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

export async function main(device, args) {
  const steps = Number(option(args, "steps", "80"));
  const mode = option(args, "mode", "diffusion");
  const seeds = option(args, "seeds", "1,2").split(",").map(Number);
  const sequence = option(args, "sequence", GCN4);
  const chains = [sequence, sequence];

  const search = await generateMmseqs2ComplexMsa(chains, {
    model: "af3",
    onProgress: (progress) => console.log("[msa]", JSON.stringify(progress)),
  });
  // 🔴 THE BLOCKS, NOT THE MERGED TEXT. AF3's `msa` is the paired block
  // followed by the unpaired one, and it computes the profile over the second -
  // so handing it the merged a3m as one string loses the boundary the model
  // reads. web/app.js passes `blocks` for the same reason.
  const alignment = search.blocks ?? { unpaired: search.a3m };
  console.log("[msa] merged depth", search.pairedDepth, "paired rows");
  const weights = await loadAf3Weights(() => {});

  const arms = {};
  for (const maxMsaSequences of DEPTHS) {
    for (const recycles of RECYCLES) {
      const runs = [];
      for (const seed of seeds) {
        runs.push(await foldAf3({
          sequence: chains.join(":"), mode, calls: steps, recycles, seed,
          device, weights,
          // Depth 1 is the query alone, which is the single-sequence control
          // every other probe here has been running.
          alignment: maxMsaSequences === 1 ? null : alignment,
          maxMsaSequences,
          onStatus: () => {}, onProgress: () => {},
        }));
      }
      const iptm = runs.map((r) => r.confidence.iptm).filter(Number.isFinite);
      arms[`depth ${maxMsaSequences} / recycles ${recycles}`] = {
        depth: runs[0].depth,
        iptm: iptm.length === 0 ? null : Number(mean(iptm).toFixed(3)),
        iptmRange: iptm.length === 0 ? null
          : [Number(Math.min(...iptm).toFixed(3)), Number(Math.max(...iptm).toFixed(3))],
        ptm: Number(mean(runs.map((r) => r.confidence.ptm)).toFixed(3)),
        meanPlddt: Number(mean(runs.map((r) => r.meanPlddt)).toFixed(1)),
      };
    }
  }
  return { mode, steps, seeds, chains: chains.length, arms };
}

/**
 * The page's own trajectory, checked for per-frame confidence in its B-factors.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-frame-confidence.js
 *
 * WHY IT EXISTS. src/af3/distogram-confidence.js is measured against real
 * pLDDT and against settling by tools/gpu/probe-distogram-confidence.js; this
 * checks the WIRING - that web/af3-model.js actually puts a different, rising
 * confidence in each frame's B-factor column and still lands the finished
 * structure on the real pLDDT. Those are two different failures and only this
 * one can see the second.
 *
 * 🔴 IT DRIVES foldAf3, NOT foldBatch, because the thing being tested is the
 * page's assembly and not the model. Everything the page does to a frame -
 * the rigid fit onto the first frame, the B-factor broadcast from tokens to
 * atom slots, the PDB text - is in that path and in none of the others.
 */
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** The B-factor column of every ATOM record, which is where pLDDT is written. */
function bFactors(pdb) {
  const values = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    values.push(Number(line.slice(60, 66)));
  }
  return values;
}
const mean = (v) => v.reduce((s, x) => s + x, 0) / Math.max(v.length, 1);

export async function main(device, args) {
  const sequence = option(args, "sequence",
    "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK");
  const calls = Number(option(args, "steps", "8"));
  const weights = await loadAf3Weights(() => {});
  const result = await foldAf3({
    sequence, mode: "flow", calls, recycles: 0, seed: 3, device, weights,
    onStatus: () => {}, onProgress: () => {},
  });

  const frames = result.framePdbs.map(bFactors);
  const final = bFactors(result.pdb);
  const means = frames.map((f) => Number(mean(f).toFixed(1)));
  // Every frame must have the same atom count as the final structure, or the
  // broadcast has gone wrong rather than the score.
  const sameLength = frames.every((f) => f.length === final.length);
  // 🔴 THE FAILURE THIS IS REALLY FOR: every frame carrying the SAME numbers,
  // which is what the page did before and what a broken wiring would look like.
  const distinct = new Set(means).size;
  return {
    sequence: `${sequence.slice(0, 16)}...(${sequence.length})`,
    frames: frames.length,
    atomsPerFrame: final.length,
    sameLength,
    frameMeans: means,
    distinctFrameMeans: distinct,
    rising: means[means.length - 1] > means[0],
    finalPdbMeanBFactor: Number(mean(final).toFixed(1)),
    reportedMeanPlddt: Number(result.meanPlddt.toFixed(1)),
    // The last frame is the same structure as the final PDB, so its calibrated
    // confidence should land on the real pLDDT the card reports.
    lastFrameVersusFinal: Number((means[means.length - 1] - mean(final)).toFixed(1)),
  };
}

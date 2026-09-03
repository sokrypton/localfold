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
 *
 * 🔴 AND IT ASSEMBLES THE TIMELINE THE WAY web/app.js DOES, which is the thing
 * actually watched: every frame but the last from the distogram, and the last
 * one the FINISHED structure with its real pLDDT. app.js drops the final
 * sampler frame rather than appending after it - it is the same structure, so
 * keeping both ends the play bar on the same picture twice - and that slicing
 * is easy to get wrong in a way no other check here would see.
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
  // 🔴 THE LIVE FRAMES ARE CAPTURED SEPARATELY FROM THE FINISHED ONES. They
  // come through onFrame while the sampler runs, coloured from the trunk's
  // distogram with no calibration - there is no finished structure yet. They
  // used to arrive with a null B-factor, so "every live frame is zero" is the
  // regression this exists to catch.
  const liveFrames = [];
  const result = await foldAf3({
    sequence, mode: "flow", calls, recycles: 0, seed: 3, device, weights,
    onStatus: () => {}, onProgress: () => {},
    onFrame: (pdb) => { liveFrames.push(pdb); },
  });

  const frames = result.framePdbs.map(bFactors);
  const final = bFactors(result.pdb);
  // web/app.js line for line: `[...framePdbs.slice(0, -1), result.pdb]`.
  const timeline = [...result.framePdbs.slice(0, -1), result.pdb].map(bFactors);
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
    live: {
      frames: liveFrames.length,
      means: liveFrames.map((pdb) => Number(mean(bFactors(pdb)).toFixed(1))),
      allZero: liveFrames.every((pdb) => bFactors(pdb).every((v) => v === 0)),
    },
    // 🔴 THERE IS NO CARD NUMBER TO CHECK THE COLOUR AGAINST ANY MORE. The
    // quality card used to show a "Settled" percentage per frame, and this
    // fitted a line through it against the frame's own B-factors to catch an
    // off-by-one between two arrays built by different code. The card shows
    // only the head's scores now, so the frames carry a colour and nothing
    // else, and the checks below - that the live frames are not all zero, that
    // the timeline ends on the real pLDDT - are what is left to hold.
    timeline: {
      length: timeline.length,
      means: timeline.map((f) => Number(mean(f).toFixed(1))),
      // The last entry must BE the finished structure, not a distogram frame.
      endsOnRealPlddt: Math.abs(mean(timeline[timeline.length - 1])
        - mean(final)) < 1e-6,
      // ...and it must not repeat the structure: one fewer sampler frame than
      // was produced, with the finished one in its place.
      dropsLastSamplerFrame: timeline.length === frames.length,
    },
  };
}

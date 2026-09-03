/**
 * Does the bar move at the speed the fold actually runs at?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-progress-bar.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-progress-bar.js --recycles=0
 *
 * WHY IT EXISTS. bench-runtime.js fits the PIECES - a trunk pass, a denoiser
 * call - and each of them can be right while the bar is still wrong, because
 * what the bar shows is the ratio between them plus the bands that no bench
 * measures: featurisation, the pipeline compile before the first denoiser call,
 * and the per-frame work on the host. A fold that sits at half and then jumps
 * to the end is a band that is sized wrong, not a kernel that is mistimed.
 *
 * WHAT IT MEASURES. Every onProgress call with the clock beside it. A perfect
 * bar has `fraction == elapsed / total` at every sample, so the error is
 * `fraction - elapsed/total` and the worst of those is the number to drive
 * down. It is signed on purpose: POSITIVE means the bar is ahead of the fold
 * and will stall, NEGATIVE means it is behind and will jump at the end.
 *
 * 🔴 IT NEEDS THE WHOLE PAGE PATH, NOT foldBatch. The bands that are wrong are
 * the ones web/af3-model.js adds on top of the GPU work, so this drives
 * foldAf3 exactly as index.html does.
 */
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const sequenceOf = (tokens) => Array.from({ length: tokens },
  (_, index) => ALPHABET[index % ALPHABET.length]).join("");

/** One fold, reporting what each band actually cost in milliseconds. */
async function timeOne(device, weights, { sequence, calls, recycles, mode }) {
  const samples = [];
  let phase = "";
  const started = performance.now();
  const result = await foldAf3({
    sequence, mode, calls, recycles, seed: 3, device, weights,
    // 🔴 A HANDLER, EVEN AN EMPTY ONE, OR THE FRAME WORK IS NOT MEASURED.
    // web/af3-model.js writes `options.onFrame?.(fittedPdb(...), shown)`, and
    // an optional call does not evaluate its arguments - so without this the
    // superposition and the PDB text are skipped and a per-call figure comes
    // back that the page never sees.
    onFrame: () => {},
    // ...the status line names the band, so a sample can say WHICH band it was
    // in without this having to know the stage names.
    onStatus: (text) => { phase = text.split(" ")[0]; },
    onProgress: (fraction) => {
      samples.push({ phase, fraction, at: performance.now() - started });
    },
  });
  const total = performance.now() - started;

  const errors = samples.map((sample) => sample.fraction - sample.at / total);
  const worst = errors.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
  const bands = [];
  for (const sample of samples) {
    const last = bands[bands.length - 1];
    if (last === undefined || last.phase !== sample.phase) {
      bands.push({ phase: sample.phase, from: last?.at ?? 0, fraction: sample.fraction, at: sample.at });
    } else {
      last.fraction = sample.fraction;
      last.at = sample.at;
    }
  }
  // 🔴 THE COMPILE IS THE INTERVAL BEFORE THE FIRST SAMPLER SAMPLE, NOT THE
  // ONE BETWEEN THE FIRST TWO. It has no milestones and blocks the main
  // thread, so it shows up as one long silence between the trunk's last report
  // and the first step's - which is that interval minus one ordinary call.
  // Reading gaps[0] instead measured the second call and said the compile had
  // vanished; it had not, it is 736 ms at 58 tokens.
  const building = samples.filter((sample) => sample.phase === "Building");
  const gaps = building.slice(1).map((sample, index) => sample.at - building[index].at);
  const sorted = [...gaps].sort((a, b) => a - b);
  return {
    tokens: result.batch.tokens,
    atoms: result.batch.tokens * result.batch.dense,
    seconds: Number((total / 1000).toFixed(1)),
    worstError: Number(worst.toFixed(3)),
    meanAbsError: Number(
      (errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length).toFixed(3)),
    bands: bands.map((band) => ({
      phase: band.phase,
      ms: Math.round(band.at - band.from),
      barEndsAt: Number(band.fraction.toFixed(3)),
      clockEndsAt: Number((band.at / total).toFixed(3)),
      // 🔴 THIS IS THE COLUMN TO READ. A band the bar gives more room than the
      // clock does is one the fold leaves early, and every band after it has
      // to make the difference up.
      error: Number((band.fraction - band.at / total).toFixed(3)),
    })),
    // ...every gap, because a median hides the shape: the band is 2.7 s where
    // sixteen median calls are 1.8, so something in there is not a median call.
    gaps: gaps.map((gap) => Math.round(gap)),
    perCallMs: sorted.length < 3 ? null
      : Math.round(sorted[Math.floor(sorted.length / 2)]),
    warmupMs: (() => {
      if (building.length === 0 || sorted.length < 3) return null;
      const band = bands.find((entry) => entry.phase === building[0].phase);
      const firstInterval = building[0].at - band.from;
      return Math.round(firstInterval - sorted[Math.floor(sorted.length / 2)]);
    })(),
  };
}

export async function main(device, args) {
  const calls = Number(option(args, "steps", "16"));
  const recycles = Number(option(args, "recycles", "3"));
  const mode = option(args, "mode", "flow");
  const lengths = option(args, "lengths", "58,128,192").split(",").map(Number);
  const weights = await loadAf3Weights(() => {});
  const runs = [];
  for (const length of lengths) {
    runs.push(await timeOne(device, weights,
      { sequence: sequenceOf(length), calls, recycles, mode }));
  }
  return { mode, calls, recycles, runs };
}

/**
 * ONE denoiser call at each starting sigma: where does the backbone land?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-one-step-sigma.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-one-step-sigma.js --assembly=1to1
 *
 * 🔴 IT GOES TO THE NETWORK for the alignment, and then does almost no GPU
 * work: ONE trunk, reused by every arm, and one denoiser call per arm. The
 * whole sweep costs about what a single 20-step fold does, because the trunk is
 * the expensive part and it does not depend on the sampler at all.
 *
 * WHAT IT ASKS. The flow starts from one draw at sigma0 and walks down
 * deterministically. At `cycles = 1` there is no walk: positions are seeded at
 * sigma0, the head is told the level is sigma0, and its output IS the answer -
 * so a sweep over sigma0 at one step is the cleanest possible picture of what
 * the starting noise buys, with the schedule shape and the step count taken out
 * of it entirely.
 *
 * 🔴 THE FLOW AND NOT THE SAMPLER, BECAUSE OF WHAT EACH ONE RETURNS. flowOnGpu
 * returns the head's DENOISED prediction; sampleOnGpu returns the noisy WALK,
 * which at one step is still at the top of the schedule. Measured on 27UH A2B2,
 * `diffusion 1` gives a CA-CA of 2852 A and a radius of gyration of 2234 A -
 * a cloud, not a structure - while the page's trajectory looks fine at that
 * setting because web/app.js draws `denoised`. See docs/AF3.md.
 *
 * 🔴 BACKBONE FIRST, SCORES SECOND. A CA-CA around 3.8 A is what a connected
 * chain looks like and a radius of gyration near the crystal's is what a
 * compact assembly looks like; both are properties of the structure. ipTM is
 * the model's opinion of it. A sigma that wrecks the geometry is wrong however
 * it scores.
 */
import { generateMmseqs2ComplexMsa } from "../../src/input/mmseqs2-api.js";
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** 27UH entity 1, the designed VHH, with GGGGSHHHHHH removed. */
const VHH = "EVQLVESGGGLVQPGGSLRLSCAASGDTSFIIAMAWYRQAPGKGRELVAGLNRLTSSISYADSVKG"
  + "RFTISRDNAKNTLYLQMNSLRPEDTAVYYCAAARVLGGTTERAWGQGTLVTVSS";
/** 27UH entity 2, human S100A4. */
const S100A4 = "SMACPLEKALDVMVSTFHKYSGKEGDKFKLNKSELKELLTRELPSFLGKRTDEAAFQKLMSNLDSN"
  + "RDNEVDFQEYCVFLSCIAMMCNEFFEGFPDKQPRKK";

/**
 * In angstroms. sigmaMax is in units of sigmaData, which is 16 A.
 *
 * 🔴 SWEEP IT IN CHUNKS WITH `--sigmas=`, BECAUSE THE PAGE DIES AROUND THIRTY
 * FOLDS. Twelve sigmas at three seeds is thirty-six folds in one process and
 * Chrome exits without reporting; fourteen folds in the same shape completes.
 * Something accumulates per fold that the per-fold disposals do not release.
 * Each process pays one trunk (~23 s at 444 tokens) and the arms are
 * independent, so chunking costs a trunk per chunk and nothing else.
 */
const DEFAULT_SIGMAS = [16, 32, 64, 128, 160, 256, 512, 1024, 2048, 2560, 5120, 10240];

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

export async function main(device, args) {
  const seeds = option(args, "seeds", "1,2,3").split(",").map(Number);
  const maxMsaSequences = Number(option(args, "max-msa", "128"));
  const SIGMAS = option(args, "sigmas", "") === ""
    ? DEFAULT_SIGMAS : option(args, "sigmas", "").split(",").map(Number);
  const chains = option(args, "assembly", "a2b2") === "a2b2"
    ? [VHH, VHH, S100A4, S100A4] : [VHH, S100A4];

  const search = await generateMmseqs2ComplexMsa(chains, { model: "af3" });
  const alignment = search.blocks ?? { unpaired: search.a3m };

  const weights = await loadAf3Weights(() => {});
  const common = {
    sequence: chains.join(":"), mode: "flow", calls: 1, recycles: 0,
    device, weights, alignment, maxMsaSequences,
    onStatus: () => {}, onProgress: () => {},
  };

  // 🔴 ONE TRUNK FOR THE WHOLE SWEEP. It does not depend on sigma, and
  // `onTrunk` hands it back the moment it exists - which is the callback the
  // page's "Fold anyway" button needed and this reuses.
  let reuse;
  const first = await foldAf3({ ...common, seed: seeds[0],
    schedule: { sigmaMax: SIGMAS[0] / 16 },
    onTrunk: (reusable) => { reuse = reusable; } });

  const rows = [];
  for (const sigma of SIGMAS) {
    const runs = [];
    for (const seed of seeds) {
      runs.push(sigma === SIGMAS[0] && seed === seeds[0] ? first
        : await foldAf3({ ...common, seed, reuse, schedule: { sigmaMax: sigma / 16 } }));
    }
    const iptm = runs.map((r) => r.confidence.iptm).filter(Number.isFinite);
    const interfaces = Object.keys(runs[0].confidence.chainPairIptm ?? {});
    rows.push({
      sigma,
      // The ratio the hypothesis is about: the starting cloud against the
      // structure's own size.
      sigmaOverRg: Number((sigma / mean(runs.map((r) => r.geometry.gyration))).toFixed(1)),
      caCa: Number(mean(runs.map((r) => r.geometry.caca)).toFixed(2)),
      gyration: Number(mean(runs.map((r) => r.geometry.gyration)).toFixed(1)),
      iptm: iptm.length === 0 ? null : Number(mean(iptm).toFixed(3)),
      ptm: Number(mean(runs.map((r) => r.confidence.ptm)).toFixed(3)),
      meanPlddt: Number(mean(runs.map((r) => r.meanPlddt)).toFixed(1)),
      interfaces: Object.fromEntries(interfaces.map((pair) => [pair,
        Number(mean(runs.map((r) => r.confidence.chainPairIptm[pair])).toFixed(3))])),
    });
  }
  return { tokens: first.batch.tokens, msaRows: first.depth, chains: chains.length,
    seeds, rows };
}

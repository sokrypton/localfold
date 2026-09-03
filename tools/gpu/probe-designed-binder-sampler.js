/**
 * Which sampler configuration folds a DESIGNED BINDER onto its target?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-designed-binder-sampler.js
 *
 * 🔴 IT GOES TO THE NETWORK. The alignment comes from ColabFold's public
 * MMseqs2 API, the same search the page runs. A search takes minutes and is
 * queued at the far end; run this deliberately.
 *
 * THE TARGET IS PDB 27UH: a de novo-designed VHH bound to human S100A4. That
 * pairing is the point, and it is a regime none of the other complex probes
 * here contain. They fold homodimers, where chain pairing is trivially correct
 * because both chains ARE the same sequence, or two designed chains together,
 * where neither side has an alignment. Neither can produce a WRONG pairing, so
 * neither could ever show the failure this target shows.
 *
 * 🔴 WHY THIS COMPLEX IS DIFFERENT. S100A4 is a conserved human protein with a
 * deep alignment. The VHH is synthetic - its framework is a standard
 * immunoglobulin V-domain and will pull plenty of nanobody hits, but its CDRs,
 * which are the whole of the binding, were designed and have no homologues at
 * all. And PAIRING between the two is meaningless in principle: pairing works
 * by co-evolution across orthologous species pairs, and there is no species in
 * which a designed binder co-evolved with S100A4. So the paired rows are noise
 * about precisely the interface, which is a reason for both a depth cap and
 * few recycles to help - recycling feeds the pair representation back into
 * itself and sharpens whatever it already believes, including a wrong pairing.
 *
 * WHAT IS HELD AND WHAT MOVES. Recycles 0 and an MSA capped at 128 rows are
 * held, because a real fold of this target is already known to work at those.
 * The sampler is what moves: AF3's own stochastic sampler across step counts,
 * and the flow beside it. probe-flow-sigma-by-size.js found the sampler badly
 * under-stepped at the page's 20 on a synthetic complex; this asks the same
 * question where the answer matters.
 *
 * WHAT IT FOUND: THE SAMPLER DOES NOT MATTER HERE AND THE SEED DOES. At
 * recycles 0 and 128 MSA rows, two seeds, ipTM as mean [range]:
 *
 *     diffusion  20   0.505 [0.352-0.659]      flow 16   0.548 [0.454-0.641]
 *     diffusion  40   0.573 [0.497-0.649]      flow 32   0.545 [0.460-0.631]
 *     diffusion  80   0.551 [0.482-0.619]      flow 64   0.529 [0.448-0.611]
 *     diffusion 160   0.557 [0.450-0.664]
 *     diffusion 320   0.529 [0.417-0.641]
 *
 * Every range overlaps every other range. pTM sits between 0.697 and 0.726 and
 * pLDDT between 81.6 and 82.4 across the whole table, and 320 steps buys
 * nothing over 20 while costing nine times as much.
 *
 * 🔴 THE SEED SPREAD IS 0.3 WIDE AND SWAMPS EVERY SETTING. One configuration
 * spans 0.352 to 0.659 across two seeds; the widest gap between configurations
 * is 0.068 between their means. So a fold or two per setting - which is how
 * anyone tunes by hand - will show a "setting that fixed it" that is the seed.
 *
 * 🔴 AND IT NARROWS A RESULT MEASURED ELSEWHERE. probe-flow-sigma-by-size.js
 * found AF3's sampler badly under-stepped at 20, scoring less than half of 80
 * on Top7 x3. That does NOT hold here: 20 is as good as 320. The difference
 * between the panels is the alignment - Top7 x3 is single-sequence and
 * synthetic - so the honest form of that finding is "20 is too few when the
 * trunk has little to go on", not "20 is too few". The diffusion default was
 * not moved on it, which was the right call.
 *
 * 🔴 WHAT THIS TARGET WANTS IS MORE SEEDS, NOT MORE STEPS. Five seeds at
 * diffusion 20 cost less than one at 320 and cover a range the step count
 * cannot reach.
 *
 * 🔴 THE His-TAG IS STRIPPED. The deposited VHH ends GGGGSHHHHHH - a Gly-Ser
 * linker and a purification tag, disordered and outside the interface. Twelve
 * disordered tokens drag pLDDT and pTM down globally and would be charged to
 * whichever sampler happened to place them.
 *
 * 🔴 THE BIOLOGICAL ASSEMBLY IS A2B2 AND `--assembly=a2b2` FOLDS IT. RCSB
 * reports 27UH-1 as tetrameric with C2 global symmetry, stoichiometry A2B2:
 * the S100A4 homodimer with a VHH on each protomer. The 1:1 default is the
 * cheap version of the interface question and the sweep above was run on it.
 *
 * 🔴 AND ON A2B2 THE POOLED ipTM IS THE WRONG NUMBER. Its selector counts every
 * cross-chain pair equally, so the native S100A4 dimer - an interface the model
 * places well - is averaged in with the designed binder's, and the score reads
 * better while saying less about the only interface anyone is asking about.
 * The per-interface breakdown comes from chainPairTmScores; chains 0 and 1 are
 * the VHHs and 2 and 3 the S100A4s, so 2|3 is the native dimer and 0|2, 0|3,
 * 1|2, 1|3 are the four binder interfaces.
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

/** The sampler arms: AF3's own across step counts, and the flow beside it. */
const ARMS = [
  ["diffusion", 20], ["diffusion", 40], ["diffusion", 80],
  ["diffusion", 160], ["diffusion", 320],
  ["flow", 16], ["flow", 32], ["flow", 64],
];

/** `--arms=diffusion:20,flow:16` to run a subset, for bisecting a failure. */
const chosenArms = (args) => {
  const wanted = args.find((a) => a.startsWith("--arms="))?.slice(7);
  if (wanted === undefined) return ARMS;
  const keep = new Set(wanted.split(","));
  return ARMS.filter(([mode, calls]) => keep.has(`${mode}:${calls}`));
};

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

export async function main(device, args) {
  const seeds = option(args, "seeds", "1,2").split(",").map(Number);
  const recycles = Number(option(args, "recycles", "0"));
  const maxMsaSequences = Number(option(args, "max-msa", "128"));
  // Chains in blocks, so an interface can be named: 0,1 are VHH and 2,3 S100A4.
  const chains = option(args, "assembly", "1to1") === "a2b2"
    ? [VHH, VHH, S100A4, S100A4] : [VHH, S100A4];

  const search = await generateMmseqs2ComplexMsa(chains, {
    model: "af3",
    onProgress: (progress) => console.log("[msa]", JSON.stringify(progress)),
  });
  const alignment = search.blocks ?? { unpaired: search.a3m };
  console.log("[msa] paired depth", search.pairedDepth);

  const weights = await loadAf3Weights(() => {});
  const arms = {};
  let tokens = 0;
  let depth = 0;
  // 🔴 REPORTED, NOT THROWN. A rejection out of main reaches the harness as
  // "Chrome exited before reporting", which names nothing and cost a run.
  const failures = [];
  for (const [mode, calls] of chosenArms(args)) {
    try {
    const runs = [];
    for (const seed of seeds) {
      const result = await foldAf3({
        sequence: chains.join(":"), mode, calls, recycles, seed,
        device, weights, alignment, maxMsaSequences,
        onStatus: () => {}, onProgress: () => {},
      });
      tokens = result.batch.tokens;
      depth = result.depth;
      runs.push(result);
    }
    const iptm = runs.map((r) => r.confidence.iptm).filter(Number.isFinite);
    arms[`${mode} ${calls}`] = {
      iptm: iptm.length === 0 ? null : Number(mean(iptm).toFixed(3)),
      iptmRange: iptm.length === 0 ? null
        : [Number(Math.min(...iptm).toFixed(3)), Number(Math.max(...iptm).toFixed(3))],
      ptm: Number(mean(runs.map((r) => r.confidence.ptm)).toFixed(3)),
      meanPlddt: Number(mean(runs.map((r) => r.meanPlddt)).toFixed(1)),
      seconds: Number(mean(runs.map((r) => r.seconds)).toFixed(1)),
      // ...and every interface on its own, meaned over the seeds.
      interfaces: Object.fromEntries(
        Object.keys(runs[0].confidence.chainPairIptm ?? {}).map((pair) => [pair,
          Number(mean(runs.map((r) => r.confidence.chainPairIptm[pair])).toFixed(3))])),
    };
    } catch (cause) {
      failures.push({ arm: `${mode} ${calls}`, error: String(cause?.message ?? cause),
        stack: String(cause?.stack ?? "").split("\n").slice(0, 5) });
    }
  }
  return { target: "27UH", tokens, msaRows: depth, recycles, maxMsaSequences,
    seeds, arms, failures };
}

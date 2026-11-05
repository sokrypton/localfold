/**
 * AF3's confidence head in WebGL2 against `cpu/af3/confidence/confidence.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-confidence.js [--falsify]
 *
 * 🔴 ALL SIX OUTPUTS, PER DIALECT, BECAUSE FIVE OF THEM SHARE A BUG BUDGET AND
 * ONE DOES NOT. pLDDT and the resolved logits come off the SINGLE track and PAE,
 * PDE and both logit tensors off the PAIR track, so a check that reads only the
 * numbers a page shows (pLDDT and PAE) leaves the resolved head and the
 * distance logits compared against nothing - and the distance logits are what
 * pTM is computed from downstream.
 *
 * 🔴 pLDDT IS PER ATOM SLOT, NOT PER TOKEN - the projection is (384, 24, 50) -
 * so the comparison is over `tokens * dense` rows. "Reading it as (384, 50) and
 * broadcasting gives plausible per-residue numbers, so a check that compares
 * token-wise means would pass on it."
 *
 * 🔴 FIVE DIALECTS, EACH OF WHICH CHANGES THE ANSWER WITHOUT CHANGING A SHAPE:
 * stock AF3, AF3 with no `input_single_norm` (its own branch, and the one a
 * clamp bug lived in), rf3 (a masked GLOBAL norm over every detached trunk input, 40
 * distogram BOUNDARIES rather than 39 edges, two absent s_inputs columns, and
 * the per-atom heads read by RANK), protenix2 (a second, UNBINNED distance term
 * beside the binned one) and boltz2's pair heads (split intra/inter, and
 * symmetrised BEFORE the projection rather than after). The gate asserts each
 * moves the answer, because a dialect this layout cannot see is a dialect
 * nobody has checked - which on a MONOMER the split heads genuinely are, so it
 * runs TWO CHAINS.
 *
 * 🔴 AND pTM IS COMPARED AS A NUMBER. It is pure host arithmetic over the PAE
 * logits, shared with AF2 through `shared/heads/tm-score.js`, so what is at
 * stake is whether this port's logits reach it intact: a 1e-7 tensor that
 * produces a different pTM would mean the layout is wrong rather than the
 * arithmetic.
 *
 * 🔴 AND ONE FLAG IS REFUSED RATHER THAN APPROXIMATED. boltz2's
 * `reembedConfidencePair` rebuilds z from nine terms instead of adding two to
 * the trunk's; AF3's two-term embedding under that flag would return a finite,
 * plausible PAE for a model this backend has not ported, which is the failure
 * mode this repository keeps finding.
 */
import {
  confidenceHead as cpuConfidenceHead,
} from "../../cpu/af3/confidence/confidence.js";
import { plddtByRank } from "../../cpu/af3/confidence/confidence.js";
import { predictedTmScores } from "../../cpu/af3/confidence/ptm.js";
import { pairformerBlock as cpuPairformerBlock } from "../../cpu/af3/trunk/pairformer.js";
import {
  confidenceHead, errorBinCentres, MAX_ERROR_BIN, NUM_BINS,
  releaseConfidenceHeadWeights, uploadConfidenceHeadWeights,
} from "../../webgl/af3/confidence/confidence.js";
import {
  releasePairformerBlockWeights, uploadPairformerBlockWeights,
} from "../../webgl/af3/trunk/pairformer-block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { pairformerBlockWeights } from "./af3-synthetic-pairformer.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const tokens = Number(option(args, "tokens", "6"));
  const dense = Number(option(args, "dense", "4"));
  const pairChannels = Number(option(args, "pair-channels", "32"));
  const singleChannels = Number(option(args, "single-channels", "64"));
  const targetFeatWidth = Number(option(args, "target-feat-width", "40"));
  const blockCount = Number(option(args, "blocks", "2"));
  const pairs = tokens * tokens;

  const random = generator(1732050);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // 🔴 TWO CHAINS AND A PADDED TAIL. The split pair heads are masked HARD by
  // asym_id, so on a monomer the inter head never fires and the arm would read
  // inert; and the masked global norm is padding-sensitive in a way nothing else
  // here is - upstream measured a chain padded into a bucket at "PAE ~28 A
  // everywhere and pTM 0.04 against 0.89 for the same fold".
  const seqMask = new Float32Array(tokens).fill(1);
  if (tokens > 3) seqMask[tokens - 1] = 0;
  const asymId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) asymId[t] = t < Math.ceil(tokens / 2) ? 0 : 1;
  // A slot is real when its name is not empty, and the rank heads only differ
  // from the slot heads where one is EMPTY - so some tokens have holes.
  const atomNameChars = new Float32Array(tokens * dense * 4);
  for (let token = 0; token < tokens; token += 1) {
    for (let slot = 0; slot < dense; slot += 1) {
      const empty = (token + slot) % 3 === 0;
      for (let c = 0; c < 4; c += 1) {
        atomNameChars[((token * dense + slot) * 4) + c] = empty ? 0 : 65 + slot;
      }
    }
  }

  const pair = noise(pairs * pairChannels, 1);
  // 🔴 THE SINGLE AT THE TRUNK'S OWN MAGNITUDE, NOT AT NOISE SCALE. AF3's trunk
  // single "reaches 1.7e5" and enters this head at a standard deviation of 211,
  // and the head's two input branches differ by a CLAMP at +/-512 - which is a
  // no-op on values of order one. Driven at noise scale, a port that clamped
  // where it should copy agreed with the reference to 3e-6 here and took a real
  // fold's pLDDT from 75 to 6, with PAE and the coordinates still in band
  // because the pair half of a pairformer block never reads the single.
  // 🔴 AND WITH A TAIL PAST 512, WHICH A UNIFORM DRAW AT std 200 DOES NOT HAVE.
  // The first version of this arm scaled the single to 200 and the clamp was
  // STILL a no-op, because nothing a uniform +/-200 produces reaches the
  // threshold: the gate stayed green with the bug restored. A real single's
  // distribution is heavy-tailed, so every eighth channel here is six times the
  // bulk - which is what makes the two branches differ at all.
  const singleScale = Number(option(args, "single-scale", "200"));
  const single = Float32Array.from(noise(tokens * singleChannels, singleScale),
    (value, index) => (index % 8 === 0 ? value * 6 : value));
  const targetFeat = noise(tokens * targetFeatWidth, 1);
  // Coordinates on a realistic scale, so the 39 distogram bins are exercised
  // rather than all falling in the catch-all.
  const pseudoBeta = noise(tokens * 3, 20);

  // 🔴 THE TRANSITION FACTOR IS **4** AND IT IS NOT A FREE PARAMETER OF THE
  // GATE. `cpu/af3/trunk/pairformer.js`'s `transition` takes it as an argument
  // defaulting to 4 and the pairformer block never passes one, so factor-2
  // weights are read at the wrong stride and every output comes back NaN - which
  // is what this gate did on its first run. The webgl transition DERIVES the
  // factor from the weight, so it would have computed a different, finite answer:
  // a shape neither side admits is a shape to refuse, not to pick.
  const blocks = Array.from({ length: blockCount }, () => pairformerBlockWeights(
    noise, { pairChannels, singleChannels, factor: 4 }));
  const base = {
    pairChannels, singleChannels, targetFeatWidth,
    leftTargetFeatProject: noise(
      targetFeatWidth * pairChannels, 1 / Math.sqrt(targetFeatWidth)),
    rightTargetFeatProject: noise(
      targetFeatWidth * pairChannels, 1 / Math.sqrt(targetFeatWidth)),
    inputSingleNormScale: noise(singleChannels, 0.2, 1),
    inputSingleNormOffset: noise(singleChannels, 0.1),
    logitsLnScale: noise(pairChannels, 0.2, 1),
    logitsLnOffset: noise(pairChannels, 0.1),
    leftHalfDistanceLogits: noise(pairChannels * NUM_BINS, 1 / Math.sqrt(pairChannels)),
    paeLogitsLnScale: noise(pairChannels, 0.2, 1),
    paeLogitsLnOffset: noise(pairChannels, 0.1),
    paeLogits: noise(pairChannels * NUM_BINS, 1 / Math.sqrt(pairChannels)),
    plddtLnScale: noise(singleChannels, 0.2, 1),
    plddtLnOffset: noise(singleChannels, 0.1),
    plddtLogits: noise(singleChannels * dense * 50, 1 / Math.sqrt(singleChannels)),
    resolvedLnScale: noise(singleChannels, 0.2, 1),
    resolvedLnOffset: noise(singleChannels, 0.1),
    experimentallyResolvedLogits: noise(
      singleChannels * dense * 2, 1 / Math.sqrt(singleChannels)),
    blocks,
  };
  const af3Dgram = noise(39 * pairChannels, 0.3);
  const caDgram = noise(40 * pairChannels, 0.3);

  /** @type {{name: string, weights: object, dialect: object}[]} */
  const CASES = [
    {
      name: "stock AF3",
      weights: { ...base, distogramFeatProject: af3Dgram },
      dialect: {},
    },
    {
      name: "rosettafold3 (global norm, 40 boundaries, absent columns, by rank)",
      weights: {
        ...base, distogramFeatProject: caDgram,
        leftTargetFeatProjectAbsent: noise(2 * pairChannels, 0.3),
        rightTargetFeatProjectAbsent: noise(2 * pairChannels, 0.3),
      },
      dialect: {
        confidenceGlobalNorm: true, confidenceCaDgram: true, perAtomHeadsByRank: true,
      },
    },
    {
      // 🔴 AF3's OWN BRANCH: no `input_single_norm` at all, so the trunk's single
      // is used RAW. That tensor is protenix2's - "every other model gets the
      // trunk's single unchanged, which is what `undefined` means" - and this is
      // the arm the clamp bug lived in.
      name: "AF3 with no input_single_norm (the trunk's single raw)",
      weights: Object.fromEntries(Object.entries({
        ...base, distogramFeatProject: af3Dgram,
      }).filter(([key]) => !key.startsWith("inputSingleNorm"))),
      dialect: {},
    },
    {
      name: "protenix2 (a second, unbinned distance term)",
      weights: {
        ...base, distogramFeatProject: af3Dgram,
        distanceFeatProject: noise(pairChannels, 0.3),
      },
      dialect: {},
    },
    {
      name: "boltz2's pair heads (split intra/inter, symmetrised first)",
      weights: {
        ...base, distogramFeatProject: af3Dgram,
        interHalfDistanceLogits: noise(
          pairChannels * NUM_BINS, 1 / Math.sqrt(pairChannels)),
        paeInterLogits: noise(pairChannels * NUM_BINS, 1 / Math.sqrt(pairChannels)),
      },
      dialect: { preSymmetrisedPde: true },
    },
  ];

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE, and it has to reach BOTH tracks: the
  // trunk pair feeds PAE, PDE and both logit tensors while the trunk single
  // feeds pLDDT and the resolved head, so one control would leave half the arms
  // green on a broken port. The coordinates get one too, since the distogram
  // term is the only path they take.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const inputs = {
    pair: upload(falsify ? skew(pair, pairChannels) : pair, "af3.conf.pair-in"),
    single: upload(
      falsify ? skew(single, singleChannels) : single, "af3.conf.single-in"),
    targetFeat: upload(
      falsify ? skew(targetFeat, targetFeatWidth) : targetFeat, "af3.conf.target"),
    pseudoBeta: upload(
      falsify ? Float32Array.from(pseudoBeta, (v) => v * 1.01 + 0.01) : pseudoBeta,
      "af3.conf.pseudo-beta"),
    seqMask: upload(seqMask, "af3.conf.seq-mask"),
    asymId: upload(Float32Array.from(asymId), "af3.conf.asym"),
    atomNameChars: upload(atomNameChars, "af3.conf.names"),
  };

  const arms = [];
  const failures = [];
  const refusals = [];
  const moves = {};
  const byDialect = {};
  const centres = errorBinCentres(NUM_BINS, MAX_ERROR_BIN);
  for (const { name, weights, dialect } of CASES) {
    const uploadedBlocks = blocks.map((block) => uploadPairformerBlockWeights(execution, block));
    const uploaded = uploadConfidenceHeadWeights(execution, weights);
    const out = confidenceHead(execution, {
      ...inputs,
      tokens, dense, weights: uploaded, blocks: uploadedBlocks,
      ...dialect,
    });
    const mine = {};
    for (const key of ["plddt", "pae", "pde", "distanceLogits", "paeLogits",
      "resolvedLogits"]) {
      mine[key] = Float32Array.from(execution.read(out[key]));
    }
    for (const tensor of Object.values(out)) execution.release(tensor);
    releaseConfidenceHeadWeights(execution, uploaded);
    for (const block of uploadedBlocks) releasePairformerBlockWeights(execution, block);

    // 🔴 THE REFERENCE GETS THE UNPERTURBED INPUTS, because a control that feeds
    // both arms is not a control - this repository's "comparing something with
    // itself" trap wearing the clothes of a control.
    let expected = cpuConfidenceHead({
      tokens, dense, seqMask, pair, single, targetFeat, pseudoBeta,
      features: { asymId },
    }, weights, cpuPairformerBlock, dialect);
    if (dialect.perAtomHeadsByRank === true) {
      // The per-atom heads read by the atom's RANK among its token's real atoms.
      expected = {
        ...expected,
        plddt: plddtByRank(expected.plddt, atomNameChars, tokens, dense),
      };
    }
    byDialect[name] = mine;
    for (const [key, rows] of [["plddt", tokens * dense], ["pae", pairs], ["pde", pairs],
      ["distanceLogits", pairs * NUM_BINS], ["paeLogits", pairs * NUM_BINS],
      ["resolvedLogits", tokens * dense * 2]]) {
      const armName = `${name}, ${key}`;
      const relative = relativeRms(mine[key], expected[key], armName);
      const ok = relative <= bar;
      arms.push({
        arm: armName, rows, relativeRms: Number(relative.toPrecision(4)), bar, ok,
      });
      if (!ok) failures.push(`${armName}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }

    // pTM and ipTM from both sides' PAE logits, as numbers.
    const scoresOf = (logits) => predictedTmScores({
      paeLogits: logits, tokens, binCentres: centres, seqMask, asymId,
    });
    const ours = scoresOf(mine.paeLogits);
    const theirs = scoresOf(expected.paeLogits);
    const worst = Math.max(
      Math.abs(ours.ptm - theirs.ptm), Math.abs(ours.iptm - theirs.iptm));
    // 🔴 THE pTM BAR IS THE MEASURED FLOOR AND NOT A ROUND NUMBER, because pTM
    // ATTENUATES: it is a max over anchors of a masked row mean of
    // probabilities, so a 1% perturbation of the logits reaches it as single
    // digits of 1e-6 - which a 1e-5 bar swallows. Measured here at 1e-9 to
    // 1e-10; ESMFold2's own confidence gate had exactly that control come back
    // green at 1e-5.
    const tmOk = worst <= 1e-8;
    arms.push({
      arm: `${name}, pTM and ipTM`,
      ptm: Number(ours.ptm.toPrecision(6)), iptm: Number(ours.iptm.toPrecision(6)),
      relativeRms: Number(worst.toPrecision(3)), bar: 1e-8, ok: tmOk,
    });
    if (!tmOk) {
      failures.push(`${name}: pTM differs by ${worst.toExponential(3)}`);
    }
  }

  // 🔴 AND EACH DIALECT MUST MOVE THE ANSWER, or this layout cannot see it.
  for (const { name } of CASES.slice(1)) {
    const changed = Math.max(...["pae", "pde", "plddt"].map(
      (key) => relativeRms(byDialect[name][key], byDialect["stock AF3"][key], `moves.${key}`)));
    moves[name] = Number(changed.toPrecision(3));
    if (!(changed > 1e-4)) {
      failures.push(`${name} is INERT on this layout: ${moves[name]}`);
    }
  }

  const refuses = (what, pattern, run) => {
    let message = null;
    try {
      run();
    } catch (error) {
      message = String(error.message);
    }
    const ok = message !== null && message.includes(pattern);
    refusals.push({ refusal: what, ok, said: message ?? "did not throw" });
    if (!ok) failures.push(`${what} was NOT refused: ${message ?? "no throw"}`);
  };
  const runWith = (weights, dialect, extra = {}) => {
    const uploadedBlocks = blocks.map((b) => uploadPairformerBlockWeights(execution, b));
    const uploaded = uploadConfidenceHeadWeights(execution, weights);
    try {
      const out = confidenceHead(execution, {
        ...inputs, ...extra, tokens, dense,
        weights: uploaded, blocks: uploadedBlocks, ...dialect,
      });
      for (const tensor of Object.values(out)) execution.release(tensor);
    } finally {
      releaseConfidenceHeadWeights(execution, uploaded);
      for (const block of uploadedBlocks) releasePairformerBlockWeights(execution, block);
    }
  };
  refuses("boltz2's nine-term re-embedding of z", "has not ported",
    () => runWith({ ...base, distogramFeatProject: af3Dgram },
      { reembedConfidencePair: true }));
  refuses("a 39-bin weight under rf3's 40-boundary dialect", "bins",
    () => runWith({ ...base, distogramFeatProject: af3Dgram },
      { confidenceCaDgram: true }));
  refuses("a 40-bin weight under AF3's 39-edge dialect", "bins",
    () => runWith({ ...base, distogramFeatProject: caDgram }, {}));
  refuses("split pair heads with no asym_id", "asymId",
    () => runWith({
      ...base, distogramFeatProject: af3Dgram,
      interHalfDistanceLogits: noise(pairChannels * NUM_BINS, 0.1),
    }, {}, { asymId: undefined }));
  refuses("the rank heads with no atom names", "atomNameChars",
    () => runWith({ ...base, distogramFeatProject: af3Dgram },
      { perAtomHeadsByRank: true }, { atomNameChars: undefined }));

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    const labels = Array.from(execution.storage.live ?? [], (t) => t.label);
    failures.push(`${execution.stats().live} tensors left live: ${labels.join(", ")}`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, dense, pairChannels, singleChannels, targetFeatWidth,
      blocks: blockCount, errorBins: NUM_BINS, plddtBins: 50,
    },
    chains: 2,
    paddedTokens: tokens - seqMask.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    dialectMoves: moves,
    refusals,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over ${CASES.length} dialects,`
          + ` ${refusals.length} refusals`
        : `FAILED: ${failures.join("; ")}`),
  };
}

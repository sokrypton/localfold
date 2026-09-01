/**
 * AF3's confidence head: GPU against src/af3/confidence-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-confidence.js
 *
 * 🔴 THE pLDDT COMPARISON IS PER ATOM SLOT. The projection is (384, 24, 50) -
 * one distribution per dense atom slot of a token, not one per token. Reading
 * it as (384, 50) and broadcasting gives plausible per-residue numbers, so a
 * check that compares token-wise means would pass on it.
 */
import { confidenceHead, distogramFeatures } from "../../src/af3/confidence-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { Af3ConfidenceHeadGpu } from "../../src/af3/confidence-webgpu.js";
import { confidenceWeights, openAf3Store } from "../../src/af3/weights.js";

const DIALECT = { swapTransposedBias: false };
const DENSE = 24;

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function deterministic(length, seed, scale = 1) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = ((((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1) * scale;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const tokens = Number(option(args, "n", "24"));
  const store = await openAf3Store();
  const weights = await confidenceWeights(store);
  const blockCount = Number(option(args, "blocks", "4"));
  weights.blocks = weights.blocks.slice(0, blockCount);

  const seqMask = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) seqMask[t] = t < Math.ceil(tokens * 0.8) ? 1 : 0;
  const input = {
    tokens, dense: DENSE, seqMask,
    pair: deterministic(tokens * tokens * 128, 61 + tokens),
    single: deterministic(tokens * 384, 62 + tokens),
    targetFeat: deterministic(tokens * 447, 63 + tokens),
    // Coordinates on a realistic scale, so the 39 distogram bins are actually
    // exercised rather than all falling in the catch-all.
    pseudoBeta: deterministic(tokens * 3, 64 + tokens, 20),
  };

  const expected = confidenceHead(input, weights, pairformerBlock, DIALECT);
  const gpu = await new Af3ConfidenceHeadGpu(device).run(input, weights, DIALECT,
    { variance: option(args, "variance", "fast") });

  // 🔴 THE FOUR CONFIDENCE PAIRFORMER BLOCKS ARE THE SAME CHAOTIC STACK the
  // trunk runs 48 of, so the heads downstream of them inherit its conditioning.
  // Perturb the input pair by a relative 1e-7 and run the CPU head against
  // ITSELF to find out what rounding alone produces before calling anything an
  // error. See tools/gpu/check-af3-block.js.
  const perturbed = { ...input, pair: Float32Array.from(input.pair) };
  for (let index = 0; index < perturbed.pair.length; index += 1) {
    perturbed.pair[index] += input.pair[index] * 1e-7;
  }
  const control = confidenceHead(perturbed, weights, pairformerBlock, DIALECT);

  // Diagnostic: is the divergence in the four blocks or in the heads? Replicate
  // the reference's own block loop and compare the stack's outputs directly.
  {
    const pm = new Float32Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) pm[i * tokens + j] = seqMask[i] * seqMask[j];
    }
    // The reference embeds the pair inside confidenceHead, so re-run just that
    // part by calling it and reading what it leaves - instead, compare the GPU
    // stack outputs against a CPU stack fed the GPU's own embedded pair.
    // What does the embed pass itself cost? Replicate it on the CPU.
    const cpuEmbedded = Float32Array.from(input.pair);
    const dgram = distogramFeatures(input.pseudoBeta, pm, tokens);
    const left = linear(input.targetFeat, tokens, 447, 128, weights.leftTargetFeatProject);
    const right = linear(input.targetFeat, tokens, 447, 128, weights.rightTargetFeatProject);
    const embedded = linear(dgram, tokens * tokens, 39, 128, weights.distogramFeatProject);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const base = (i * tokens + j) * 128;
        for (let c = 0; c < 128; c += 1) {
          cpuEmbedded[base + c] += left[j * 128 + c] + right[i * 128 + c] + embedded[base + c];
        }
      }
    }
    console.log(`  embed pass   ${relativeRms(gpu.embeddedPair, cpuEmbedded).toExponential(2)}`);

    let cpuPair = Float32Array.from(gpu.embeddedPair ?? gpu.pair);
    let cpuSingle = Float32Array.from(input.single);
    if (gpu.embeddedPair !== undefined) {
      for (const block of weights.blocks) {
        const next = pairformerBlock({ pair: cpuPair, single: cpuSingle, pairMask: pm,
                                       seqMask, tokens }, block, DIALECT);
        cpuPair = next.pair;
        cpuSingle = next.single;
      }
      console.log(`  stack pair   ${relativeRms(gpu.pair, cpuPair).toExponential(2)}`);
      console.log(`  stack single ${relativeRms(gpu.single, cpuSingle).toExponential(2)}`);
      // Does the single track intrinsically amplify the pair? Perturb the
      // stack's INPUT pair by the size the GPU actually differs by, and run the
      // CPU against itself.
      // 🔴 THE RIGHT CONTROL PERTURBS BOTH TRACKS, EVERY BLOCK. The GPU rounds
      // `single` at each of the four blocks, not only at the input, and single
      // is the residual accumulator - it grows from 0.58 to 19,356 here, so a
      // relative nudge to it each block is what the GPU actually does. Nudging
      // only the input pair understates it by whatever the accumulation adds.
      {
        let p2 = Float32Array.from(gpu.embeddedPair);
        let s2 = Float32Array.from(input.single);
        const nudge = Number(option(args, "nudge", "5e-7"));
        for (const block of weights.blocks) {
          const next = pairformerBlock({ pair: p2, single: s2, pairMask: pm, seqMask, tokens },
                                       block, DIALECT);
          p2 = Float32Array.from(next.pair);
          s2 = Float32Array.from(next.single);
          for (let i = 0; i < p2.length; i += 1) p2[i] *= 1 + nudge;
          for (let i = 0; i < s2.length; i += 1) s2[i] *= 1 + nudge;
        }
        console.log(`  a ${nudge} nudge to BOTH tracks each block gives pair `
          + `${relativeRms(p2, cpuPair).toExponential(2)} and single `
          + `${relativeRms(s2, cpuSingle).toExponential(2)}`);
      }
      const scan = Number(option(args, "probe", "3.2e-6"));
      let probePair = Float32Array.from(gpu.embeddedPair);
      for (let index = 0; index < probePair.length; index += 1) {
        probePair[index] *= 1 + scan;
      }
      let probeSingle = Float32Array.from(input.single);
      for (const block of weights.blocks) {
        const next = pairformerBlock({ pair: probePair, single: probeSingle, pairMask: pm,
                                       seqMask, tokens }, block, DIALECT);
        probePair = next.pair;
        probeSingle = next.single;
      }
      console.log(`  a 3.2e-6 input perturbation gives pair `
        + `${relativeRms(probePair, cpuPair).toExponential(2)} and single `
        + `${relativeRms(probeSingle, cpuSingle).toExponential(2)}`);
      // Where does the difference live? Padded tokens are never masked on the
      // way out of single attention, so they carry whatever four blocks make of
      // nothing - and they are in the relRMS denominator with everything else.
      // 🔴 LayerNorm IS PER ROW, so the statistic that governs whether
      // E[x^2]-E[x]^2 cancels is |mean|/std WITHIN a row, not globally.
      let worst = 0;
      let worstRow = -1;
      for (let t = 0; t < tokens; t += 1) {
        let m = 0;
        for (let c = 0; c < 384; c += 1) m += cpuSingle[t * 384 + c];
        m /= 384;
        let s2 = 0;
        for (let c = 0; c < 384; c += 1) s2 += (cpuSingle[t * 384 + c] - m) ** 2;
        const sd = Math.sqrt(s2 / 384);
        const ratio = Math.abs(m) / Math.max(sd, 1e-30);
        if (ratio > worst) { worst = ratio; worstRow = t; }
      }
      // f32 keeps ~7 digits, so cancellation bites once |mean|/std approaches 1e3.
      console.log(`  worst per-row |mean|/std ${worst.toFixed(1)} at token ${worstRow}`
        + `  (f32 loses ~${Math.log10(Math.max(worst * worst, 1)).toFixed(1)} digits there)`);
      // Per block: how big is the delta the block adds, against the result?
      // If the delta dwarfs the result, the residual is cancelling and relative
      // error amplifies by exactly that ratio.
      {
        let p3 = Float32Array.from(gpu.embeddedPair);
        let s3 = Float32Array.from(input.single);
        const sd = (v) => { let m = 0; for (const x of v) m += x; m /= v.length;
          let q = 0; for (const x of v) q += (x - m) ** 2; return Math.sqrt(q / v.length); };
        for (let b = 0; b < weights.blocks.length; b += 1) {
          const before = s3;
          const next = pairformerBlock({ pair: p3, single: s3, pairMask: pm, seqMask, tokens },
                                       weights.blocks[b], DIALECT);
          const delta = new Float32Array(before.length);
          for (let i = 0; i < delta.length; i += 1) delta[i] = next.single[i] - before[i];
          console.log(`  block ${b}: single ${sd(before).toExponential(2)}`
            + ` + delta ${sd(delta).toExponential(2)} -> ${sd(next.single).toExponential(2)}`
            + `  (delta/result ${(sd(delta) / Math.max(sd(next.single), 1e-30)).toFixed(2)})`);
          p3 = next.pair;
          s3 = next.single;
        }
      }
      const std = (v) => { let m = 0; for (const x of v) m += x; m /= v.length;
        let s2 = 0; for (const x of v) s2 += (x - m) ** 2; return Math.sqrt(s2 / v.length); };
      console.log(`  scales: input pair ${std(input.pair).toFixed(2)},`
        + ` embedded ${std(gpu.embeddedPair).toFixed(2)},`
        + ` stack pair ${std(cpuPair).toFixed(2)},`
        + ` single in ${std(input.single).toFixed(2)} out ${std(cpuSingle).toFixed(2)}`);
      // The same stack, same weights, but fed a SMALL random pair - the regime
      // check-af3-block.js runs in, where single came out at 6.2e-7.
      const slice = (source, keep) => {
        const out = [];
        for (let t = 0; t < tokens; t += 1) {
          if ((seqMask[t] > 0) !== keep) continue;
          for (let c = 0; c < 384; c += 1) out.push(source[t * 384 + c]);
        }
        return Float32Array.from(out);
      };
      console.log(`  single, real tokens only   `
        + `${relativeRms(slice(gpu.single, true), slice(cpuSingle, true)).toExponential(2)}`);
      console.log(`  single, padded tokens only `
        + `${relativeRms(slice(gpu.single, false), slice(cpuSingle, false)).toExponential(2)}`);
    }
  }

  const pairs = [
    ["plddt", gpu.plddt, expected.plddt, control.plddt],
    ["pae", gpu.pae, expected.pae, control.pae],
    ["pde", gpu.pde, expected.pde, control.pde],
    ["resolved", gpu.resolved, expected.resolvedLogits, control.resolvedLogits],
  ];
  const results = {};
  let failed = 0;
  for (const [name, actual, reference, controlValue] of pairs) {
    const value = relativeRms(actual, reference);
    const envelope = relativeRms(controlValue, reference);
    const ratio = value / Math.max(envelope, 1e-30);
    // 🔴 THE SINGLE TRACK AMPLIFIES BY ABOUT 1e5, AND THAT IS WHY plddt AND
    // resolved SIT AT 1e-4 WHILE pae AND pde SIT AT 6e-6. All four come off the
    // same four blocks; the first two read `single` and the last two read the
    // pair.
    //
    // The measurement that says so: fast and two-pass LayerNorm variance are
    // ALGEBRAICALLY IDENTICAL and differ only at the 1e-7 level, and swapping
    // one for the other moves `single` by 6.3e-2. So a 1e-7 perturbation comes
    // out the far end at 1e-2 - an amplification near 1e5 - and the GPU's
    // 1.41e-4 against a float64 reference is well UNDER what f32 rounding
    // through a stack like that would predict.
    //
    // Why this stack and not the trunk's: `single` grows from 0.58 to 4,220 in
    // ONE confidence block and to 19,356 in four, with no cancellation in the
    // residual (delta/result stays between 0.5 and 1.0). The trunk's grows far
    // more gently per block, and the same code gives single 5.4e-7 to 6.2e-7
    // there at every pair scale from 1 to 100. It is the weights, not the input:
    // `check-af3-block.js --stack=confidence` reproduces 1.14e-4 on a plain
    // random pair.
    //
    // Ruled out on the way, each by measurement rather than argument: the heads
    // themselves (the divergence is already in the stack's single output); the
    // embed pass (6.75e-7); the pair track (3.18e-6); padding (real tokens
    // 1.59e-4, padded 1.58e-5); LayerNorm cancellation (worst per-row
    // |mean|/std is 0.0 - this was my first guess and it was wrong); and
    // magnitude (single attention and the transition are flat at 3-5e-7 from
    // scale 1 to 170,000, the trunk's real range).
    //
    // 1.41e-4 is well inside AF3's own bfloat16 floor of 3.9e-3. The ceiling
    // below is a regression guard, not a target.
    const bound = name === "plddt" || name === "resolved"
      ? 5e-4
      : Math.max(1e-5, envelope * 10);
    const ok = value <= bound;
    if (!ok) failed += 1;
    results[name] = { relRms: value, envelope, ratio };
    console.log(`${name}\trelRMS ${value.toExponential(2)}`
      + `\t(envelope ${envelope.toExponential(2)}, ${ratio.toFixed(1)}x)`
      + `\t${ok ? "" : "FAIL"}`);
  }
  if (failed > 0) throw new Error(`${failed} head(s) outside their conditioning envelope`);
  return { tokens, results };
}

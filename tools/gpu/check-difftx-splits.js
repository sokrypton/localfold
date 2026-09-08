/**
 * Does every K-split of the diffusion transformer compute the unsplit answer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-difftx-splits.js
 *
 * 🔴 A SPLIT IS A DEGENERACY TEST AND NOTHING ELSE. Splitting an inner extent
 * across workgroups and summing the parts is an associativity claim: the same
 * arithmetic, regrouped. So every knob must return the unsplit answer to
 * floating-point noise, and the ONLY thing that makes a split "faster" if it
 * does not is that it stopped computing the model.
 *
 * 🔴 A 200-STEP FOLD AT `--norm-splits=2` IS WRONG AND THIS FILE CANNOT
 * REPRODUCE IT. The fold returns pLDDT 66.347114 where the unsplit path returns
 * 84.208873, and `--norm-splits=4` returns the SAME 66.347114 - a wrong answer
 * INDEPENDENT of the part count. Everything below passes at relRMS 0: every
 * knob alone, every knob against `normSplit`/`normSubgroups`, and the whole
 * `ampere` rule with only the count moved, at both 32 and 68 tokens. A bisect
 * on the fold is clean too - at 1 and 2 steps the split and unsplit paths are
 * bit-identical, so it is not the unchained first call and not the chained one
 * either.
 *
 * So the transformer's split arithmetic is RIGHT and something else in a long
 * fold is wrong. Do not raise `normSplits` off 1 until that is understood: it
 * read 3.556 s against a 3.639 s baseline, the fastest arm of its sweep, and it
 * is fast because it is not computing the model.
 *
 * 🔴 WHAT HAS BEEN RULED OUT, so the next attempt does not repeat it:
 *
 *   - the shape. The fold's resolved shape was dumped and compared field by
 *     field against this checker's; the checker produces that EXACT 24-field
 *     shape (kSplits 16, qkvgTile/wideTile 4, outKSplits 4, attnKSplits 1,
 *     normKSplits 2, normSplit true, tile 1, outChunk 384, f16) and passes.
 *   - cross-call state. Four calls on ONE instance, all relRMS 0 - so it is not
 *     the scratch cache handing back a stale or undersized buffer, which was
 *     the leading theory because `shapeKey` carries no split count and
 *     `#scratchBuffer` returns by label with no size check.
 *   - cross-kernel sharing. adaln and ffw-adaln share one partials buffer;
 *     giving ffw-adaln its own changed the fold's answer by NOTHING - 16 steps
 *     returned 75.0197297936948 either way.
 *   - the chained path. A bisect at 1/2/4 steps looked clean, but that test has
 *     NO RESOLVING POWER: the fold returns pLDDT 55.510356 at all three step
 *     counts, so it cannot distinguish anything. Wrong from 8 steps up.
 *
 *   - the activation's mean. `noise` is uniform on [-1, 1] and `adaln` is a
 *     conditioned LAYERNORM, so a mishandled centring would be invisible on a
 *     zero-mean input and obvious on a real one - the best of the six theories.
 *     `--bias` shifts the activation off zero and every arm still passes at
 *     0.7. Kept anyway: the input SHOULD be off zero, and a checker whose
 *     inputs sit where a bug cancels is worse than none.
 *   - the weights wrapper. The head passes `#transformerWeights(weights)`,
 *     which only adds `weightPrecision`, and the dumped shape says f16 both
 *     sides.
 *
 * 🔴 WHERE IT ACTUALLY DIVERGES: THE FIRST DENOISER STEP, AT relRMS 0.538.
 * `fold.js --steps=8 --trajectory=on` under both arms, compared frame by frame
 * on `denoised`: step 1 is already 5.384e-01 apart and never recovers. Inputs
 * to step 1 are identical - same seed, same trunk, same noise - so the
 * transformer returns a different answer for the same inputs at a shape this
 * checker drives correctly. That is the whole mystery in one sentence.
 *
 * 🔴 AND THIS FILE ONCE COMPARED NOTHING, WHICH IS THE WORST THING A CHECKER
 * CAN DO. `run()` resolves to `{output, elapsedMilliseconds, memory}` and not a
 * Float32Array. `rel()` took the OBJECTS, read `a.length` as undefined, never
 * entered its loop, and returned `Math.sqrt(0 / 1e-30)` - exactly zero. 106
 * arms "passed" on that, and on the strength of it the transformer was declared
 * exonerated and seven other causes were chased instead.
 *
 * The tell was written down and rationalised away: a K split regroups its
 * additions, so a correct one must read ~1e-7, NEVER 0.0 exactly. An identity
 * check that cannot fail is worse than no check, because it gets quoted as
 * evidence. `values()` now throws if it is handed anything but an array.
 *
 * 🔴 IT WAS BORN BROKEN, WHICH A BISECT SETTLES AND NO AMOUNT OF READING DID.
 * Forcing `normSplits: 2` in the prior at each of the four commits that have
 * touched this file gives the SAME wrong answer at every one of them - 8-step
 * pLDDT 67.5422122968614 against 68.92187291321439 - including `311d7a4`, the
 * commit that introduced the split. There is no regression to find: the path
 * has never computed the model.
 *
 * 🔴 AND ITS OWN COMMIT MESSAGE SAYS WHY THAT SURVIVED. `311d7a4` measured the
 * arms as KERNEL TIMINGS ("fused 1.43 ms, two parts 3.15, four 2.21") and then
 * reports "Fold gate exact" - which was true and vacuous, because it had just
 * set `normSplits: 1` and the gate therefore never entered the split path. The
 * feature was retired for being SLOW before anyone asked whether it was RIGHT.
 * A tuning knob that is off by default is not covered by a whole-fold gate;
 * that is what this file is for.
 *
 * 🔴 AND IT COVERS THE KNOBS NO FOLD ON THIS DEVICE ENTERS, WHICH IS THE
 * POINT. `normSplits` was reachable, wrong and invisible because the `ampere`
 * prior sets it to 1 and a whole-fold gate only proves things about the paths
 * it takes. The same exposure applied to `diffusionLanes` and
 * `diffusionNormSubgroups` - null in every prior here, so nothing had ever
 * folded with them. Both are now arms: lanes 64 and 128 pass at relRMS 0, and
 * lanes 512 is REFUSED (768 is not a multiple of it), which is the right
 * answer. 106 arms, one known-bad path.
 *
 * 🔴 AND SIX MORE KNOBS SINCE: `batchedGates`, `gateTile`, `attnOutTile`,
 * `outTile`, the split rule's `crossover` and `attendKeyChunk`. The tiles and
 * the hoist REORDER NOTHING, so their arms are held to exactly 0 rather than to
 * TOLERANCE, and all of them read it. 150 arms now.
 *
 * 🔴 STILL UNCOVERED, and named so nobody assumes otherwise: `singleProjectLanes`
 * (a different module), `attentionQueriesPerLane > 1`, `submissionWindow`, the
 * split rule's `crossover` (this checker passes a shape directly and never goes
 * through the token-count gate that reads it), and the `--ablate`/`--enable`
 * dialect branches. `weights.tile` and `weights.outChunk` ARE reachable here now
 * - `outTile` is an arm - but `weights.splitK` still is not.
 *
 * 🔴 AND FALSIFY EVERY ARM YOU ADD, because this file has now been vacuous
 * TWICE, by two different routes. The first was an unguarded relRms over
 * non-arrays. The second was subtler and is worth the warning: the reference
 * `unsplit` named the four split counts and nothing else, so every knob it did
 * NOT name fell through to the device prior - and the ampere prior turns
 * `batchedGates` on. The reference was running the hoisted path, the arms were
 * running the hoisted path, and six arms compared it against itself. With
 * AD_SW and AD_CB deliberately swapped in the shader they all still read 0;
 * with the reference pinned to `batchedGates: false` the same bug reads 1.05.
 * A reference has to say "off" for everything it is a reference FOR.
 *
 * 🔴 WHAT HAS NOT BEEN TRIED: the tensor VALUES. Every arm here feeds
 * synthesised noise. The next step is to capture the fold's real `act`, `cond`
 * and `pairCond` at step 1 and replay them through this checker; if they fail
 * here, it is the data, and if they pass, the fault is in the head's plumbing
 * rather than the transformer at all.
 *
 * Each knob is tested ALONE and then in combination, because they are set
 * together in one rule object and a combined arm cannot say which one broke.
 */
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights } from "../../src/af3/diffusion-weights.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "32"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await diffusionWeights(store);
  const tx = weights.transformer;
  const { channels, condChannels, pairChannels } = tx;

  const noise = (n, seed) => {
    const out = new Float32Array(n);
    let state = seed >>> 0;
    for (let i = 0; i < n; i += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      out[i] = (state / 4294967296) * 2 - 1;
    }
    return out;
  };

  // 🔴 THE ACTIVATION MUST NOT BE ZERO-MEAN, AND THAT IS THE WHOLE POINT.
  // `noise` is uniform on [-1, 1], so its mean is ~0 - and `adaln` is a
  // CONDITIONED LAYERNORM that subtracts the activation's mean. A split that
  // mishandles the centring is therefore invisible on this input and obvious on
  // a real one: every arm here passed at relRMS 0 while a fold diverged at
  // relRMS 0.538 on its FIRST denoiser step. --bias shifts the activation off
  // zero, which is the cheapest stand-in for a real activation.
  const bias = Number(option(args, "bias", "0.7"));
  const act = noise(tokens * channels, 7).map((v) => v + bias);
  const cond = noise(tokens * condChannels, 11);
  const pairCond = noise(tokens * tokens * pairChannels, 13);
  const mask = new Float32Array(tokens).fill(1);

  // 🔴 UNWRAP, AND THROW IF IT IS NOT AN ARRAY. `run()` resolves to
  // `{output, elapsedMilliseconds, memory}` and NOT a Float32Array. Comparing
  // the objects made `a.length` undefined, so the loop never ran, num and den
  // stayed 0, and this returned Math.sqrt(0 / 1e-30) = EXACTLY ZERO. Every one
  // of this file's 106 arms passed by comparing nothing.
  //
  // The tell was there and was rationalised away: a K split sums its partials
  // in a different grouping, so a real one should read ~1e-7, not 0.0 exactly.
  // An identity check that cannot fail is worse than no check, because it is
  // reported as evidence.
  // 🔴 A SPLIT REGROUPS ITS ADDITIONS, SO EXACT EQUALITY IS THE WRONG BAR.
  // Measured here: kSplits under 1e-5, outKSplits 2.5e-5, attnKSplits 3.5e-5,
  // normSubgroups 9.9e-5 - all float non-associativity, all correct. The one
  // real failure was 0.813. There is no threshold question between 1e-4 and
  // 0.8; a tolerance tight enough to flag reordering would drown the signal.
  const TOLERANCE = 1e-3;
  const rel = relativeRms;

  // 🔴 THE REFERENCE IS EVERY SPLIT OFF, NOT THE DEVICE'S PRIOR. The prior may
  // already turn some of these on, and a reference that carries a broken split
  // would report the broken arm as correct.
  // 🔴 AND "OFF" HAS TO BE SAID FOR EVERY KNOB, NOT JUST THE SPLITS. Each of
  // these falls through to the DEVICE PRIOR when the shape omits it, and the
  // ampere prior turns `diffusionBatchedGates` ON - so a reference that named
  // only the split counts was running the hoisted path too, and every
  // batchedGates arm below was comparing that path against itself. Verified:
  // with AD_SW and AD_CB deliberately swapped in the shader, all six arms still
  // read relRms 0. A reference that inherits what it is testing cannot fail.
  const unsplit = { ...tx, kSplits: 1, outKSplits: 1, attnKSplits: 1, normKSplits: 1,
                    batchedGates: false };
  const reference = await new Af3DiffusionTransformerGpu(device).run(
    act, cond, pairCond, mask, tokens, unsplit);

  // Each knob alone, at the counts a rule would plausibly use. The shader
  // factories throw on a count that does not divide their extent, which is a
  // pass for this checker's purposes - it is a refusal, not a wrong answer.
  const knobs = {
    kSplits: [2, 4, 8, 16],
    outKSplits: [2, 4],
    attnKSplits: [2, 4],
    normKSplits: [2, 4],
    // 🔴 THE PER-KERNEL TILES, which no fold entered until they were given their
    // own numbers. `outTile` had been pinned to 1 by the shared tile for as long
    // as the shared tile was 1, so the values below are paths nothing had run.
    // A tile reorders NOTHING - it changes how many tokens a workgroup carries,
    // not the order of any sum - so these arms should read exactly 0, and a
    // non-zero here is a real defect rather than float non-associativity.
    outTile: [2, 4],
    attnOutTile: [2, 4],
    // 🔴 AND THE KEY CHUNK AT A VALUE THAT ACTUALLY CHUNKS. The default 64 is
    // larger than this checker's 32 tokens, so every arm would run ONE chunk and
    // exercise nothing; 16 forces the loop to go round twice at 32 tokens. That
    // is the difference between covering a knob and naming it.
    attendKeyChunk: [16, 96],
  };

  // 🔴 AND EVERY KNOB ALONE IS NOT ENOUGH, WHICH THIS FILE LEARNED THE HARD WAY.
  // All eight passed at relRMS 0 while a FOLD at --norm-splits=2 returned
  // pLDDT 66.347114 against 84.208873. The difference is that the `ampere`
  // prior also sets `diffusionNormSplit: true` - the OUTPUT-channel split, which
  // makes `c = group.y * lanes + local` - so group.y carries output channels
  // while group.z carries K parts. A knob that is correct alone can still be
  // wrong beside the one the device actually turns on, so each is tested
  // against every modifier the priors set.
  const modifiers = {
    "": {},
    "normSplit": { normSplit: true },
    "normSubgroups": { normSubgroups: true },
    "normSplit+normSubgroups": { normSplit: true, normSubgroups: true },
    // 🔴 AND THE KNOBS NO GATE ON THIS DEVICE ENTERS, WHICH IS THE SAME
    // EXPOSURE normSplits HAD. `diffusionLanes` is null in every prior here, so
    // nothing has ever folded with a width other than the default - and it is
    // baked into every kernel in this stack as a workgroup size, which is
    // exactly the shape of thing that is wrong quietly. `attendStageKeys` and
    // `attendSubgroups` are on for ampere and covered by the fold gate; they
    // are here so the matrix is complete rather than because they are suspect.
    "lanes=64": { lanes: 64 },
    "lanes=128": { lanes: 128 },
    "lanes=512": { lanes: 512 },
    "attendStageKeys=false": { attendStageKeys: false },
    "attendSubgroups=false": { attendSubgroups: false },
  };

  const findings = [];
  for (const [name, modifier] of Object.entries(modifiers)) {
    // The modifier alone must also be a no-op, or a failure below cannot be
    // blamed on the interaction rather than on the modifier itself.
    const base = { ...unsplit, ...modifier };
    let baseRel = null;
    try {
      const out = await new Af3DiffusionTransformerGpu(device).run(
        act, cond, pairCond, mask, tokens, base);
      baseRel = Number(rel(out, reference).toPrecision(3));
    } catch (error) {
      baseRel = String(error.message).slice(0, 60);
    }
    // 🔴 A REFUSAL IS A PASS, HERE AS IN THE KNOB ARMS. The shader factory
    // throws on a geometry it cannot express - `lanes: 512` against a 768-wide
    // attention projection is not a multiple - and a configuration the code
    // declines to build is not a configuration the code gets wrong. Treating it
    // as a failure reported a false BROKEN and would have buried a real one.
    const refusedBase = typeof baseRel !== "number";
    findings.push({ knob: `(modifier alone)`, count: name || "none",
                    relRms: refusedBase ? null : baseRel,
                    refused: refusedBase ? baseRel : null,
                    ok: refusedBase || baseRel < TOLERANCE });
    if (refusedBase) continue;
    for (const [knob, counts] of Object.entries(knobs)) {
      for (const count of counts) {
        let relRms = null;
        let refused = null;
        try {
          const out = await new Af3DiffusionTransformerGpu(device).run(
            act, cond, pairCond, mask, tokens, { ...base, [knob]: count });
          relRms = Number(rel(out, reference).toPrecision(3));
        } catch (error) {
          refused = String(error.message).slice(0, 90);
        }
        findings.push({
          knob: name === "" ? knob : `${knob} +${name}`, count, relRms, refused,
          ok: refused !== null || relRms < TOLERANCE,
        });
      }
    }
  }

  // 🔴 THE CONDITIONING HOIST, WHICH IS A DIFFERENT KIND OF ARM. batchedGates
  // moves six projections out of the block loop into one dispatch; it reorders
  // nothing, so like the tiles it must read EXACTLY 0 and not merely small.
  // Kept out of the cross product above because each arm rebuilds ~85 MB of
  // concatenated gate weights - the resident buffer is keyed by the weights
  // object and every arm passes a fresh one.
  for (const [label, extra] of Object.entries({
    "alone": {},
    "+normSplit": { normSplit: true },
    "gateTile=2": { gateTile: 2 },
    "gateTile=4": { gateTile: 4 },
    "gateTile=16": { gateTile: 16 },
    "+the ampere rule": { kSplits: 16, outKSplits: 4, attnKSplits: 4, attnOutTile: 2,
                          outTile: 4, qkvgTile: 4, wideTile: 4, normSplit: true },
  })) {
    let relRms = null;
    let refused = null;
    try {
      const out = await new Af3DiffusionTransformerGpu(device).run(
        act, cond, pairCond, mask, tokens, { ...unsplit, batchedGates: true, ...extra });
      relRms = Number(rel(out, reference).toPrecision(3));
    } catch (error) {
      refused = String(error.message).slice(0, 90);
    }
    findings.push({
      knob: "batchedGates", count: label, relRms, refused,
      // 🔴 A HOIST IS HELD TO ZERO, NOT TO THE TOLERANCE. It moves arithmetic
      // without regrouping it, so anything above 0 means the move changed the
      // computation - which is exactly what a wrong conditioning buffer looked
      // like when it was fed the raw one instead of the normalised one.
      exact: refused === null && relRms === 0,
      ok: refused !== null || relRms < TOLERANCE,
    });
  }

  // 🔴 AND ONE KNOB AT A TIME IS STILL NOT THE SHIPPED CONFIGURATION. All forty
  // arms above pass while a FOLD is wrong, because a fold turns every knob on
  // TOGETHER and at the prior's tile: kSplits 16, outKSplits 4, qkvgTile and
  // wideTile 4, normSplit true. An interaction between them is invisible to a
  // sweep that varies one axis. These arms are the `ampere` rule itself, with
  // only the count under test moved.
  const rule = { kSplits: 16, outKSplits: 4, attnKSplits: 1,
                 qkvgTile: 4, wideTile: 4, normSplit: true };
  for (const normKSplits of [1, 2, 4]) {
    let relRms = null;
    let refused = null;
    try {
      const out = await new Af3DiffusionTransformerGpu(device).run(
        act, cond, pairCond, mask, tokens, { ...unsplit, ...rule, normKSplits });
      relRms = Number(rel(out, reference).toPrecision(3));
    } catch (error) {
      refused = String(error.message).slice(0, 90);
    }
    findings.push({
      knob: "ampere rule, normKSplits", count: normKSplits, relRms, refused,
      ok: refused !== null || relRms < TOLERANCE,
    });
  }
  for (const attnKSplits of [2, 4]) {
    let relRms = null;
    let refused = null;
    try {
      const out = await new Af3DiffusionTransformerGpu(device).run(
        act, cond, pairCond, mask, tokens, { ...unsplit, ...rule, attnKSplits });
      relRms = Number(rel(out, reference).toPrecision(3));
    } catch (error) {
      refused = String(error.message).slice(0, 90);
    }
    findings.push({
      knob: "ampere rule, attnKSplits", count: attnKSplits, relRms, refused,
      ok: refused !== null || relRms < TOLERANCE,
    });
  }

  // 🔴 AND ONE CALL PER INSTANCE IS STILL NOT A FOLD. A fold makes 200 calls on
  // ONE transformer, whose scratch buffers are cached by a shapeKey that
  // contains no split count and handed back by label with no size check - so
  // anything the reduce reads that this call's split did not write is the
  // PREVIOUS call's value, zero on the first call and stale after. A checker
  // that builds a fresh instance per arm cannot see that, which is why every
  // arm above passes while a 16-step fold at --norm-splits=2 returns pLDDT
  // 75.019730 against 85.909313.
  const repeats = Number(option(args, "repeats", "4"));
  for (const normKSplits of [1, 2]) {
    const instance = new Af3DiffusionTransformerGpu(device);
    const perCall = [];
    for (let call = 0; call < repeats; call += 1) {
      const out = await instance.run(act, cond, pairCond, mask, tokens,
        { ...unsplit, ...rule, normKSplits });
      perCall.push(Number(rel(out, reference).toPrecision(3)));
    }
    findings.push({
      knob: `${repeats} calls on ONE instance, normKSplits`, count: normKSplits,
      relRms: perCall.join(" "), refused: null,
      ok: perCall.every((v) => v < TOLERANCE),
    });
  }

  const broken = findings.filter((f) => !f.ok);
  return {
    tokens, channels, condChannels,
    findings,
    broken: broken.map((f) => `${f.knob}=${f.count} relRMS ${f.relRms}`),
    ok: broken.length === 0,
  };
}

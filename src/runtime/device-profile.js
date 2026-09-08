/**
 * Which device is this, and what has been MEASURED about it.
 *
 * 🔴 WHY THIS IS NOT JUST `if (vendor === "apple")`. Three knobs in this
 * repository have a different best answer on different hardware, and the
 * vendor predicts none of them on its own:
 *
 *   - Queries per invocation in the flash attention. The M2 measures **0.21x**
 *     at q2 (docs/PERF.md), upstream's M4 Pro 0.45x, upstream's GB10
 *     **1.17-1.42x**, and this A100 **1.000x** - free, neither way. Two Apple
 *     parts agree and two NVIDIA parts do not.
 *   - `shader-f16`. Present on the M2. **Absent** on an A100 under NVIDIA's
 *     535 Linux driver, because Dawn wants `storageInputOutput16` and that
 *     driver reports it false - while the same silicon under a newer driver
 *     has it. So the vendor does not tell you, and the FEATURE does.
 *   - The dense projection's tile. The M2 wants 32 rows and 64 threads and
 *     docs/PERF.md records taller tiles losing "at every shape measured"; the
 *     A100 wants 128 rows and 256 threads, at 1.55x. Here the two vendors do
 *     disagree - but that is a fact about occupancy, not about the badge.
 *
 * So this resolves in three layers, most trustworthy first:
 *
 *   1. **Capabilities.** `device.features`, `device.limits`. The API states
 *      these; never infer them from a name. Existing code already does this
 *      and should keep doing it - `chooseLinearKernel`'s `shader-f16` test is
 *      the model.
 *   2. **A measurement**, from `tools/gpu/probe-tuning.js` or from a caller
 *      that has run one. `setDeviceTuning` is how it arrives.
 *   3. **An architecture prior**, below, for a device nobody has probed. Every
 *      entry names the measurement it came from. A device not in the table
 *      gets `DEFAULT_TUNING`, which is **exactly what shipped before this file
 *      existed** - so an unrecognised device cannot regress, and adding a
 *      prior is always a deliberate act backed by a number.
 *
 * 🔴 AND `adapter.info` IS TWO FIELDS, NOT FOUR. Outside a browser started with
 * `--enable-webgpu-developer-features`, `device` and `description` are empty
 * strings for fingerprinting reasons - measured, not assumed: this A100
 * reports `{vendor: "nvidia", architecture: "ampere", device: "", description:
 * ""}` normally and fills the other two in only under that flag. So the key is
 * `vendor` and `architecture` and nothing finer, and a prior cannot single out
 * one card.
 */

/** @typedef {{linearTallTile: boolean, attentionGroup: number,
 *             attentionVectorScore: boolean, attentionQueriesPerLane: number}} Tuning */

/**
 * 🔴 THE DEFAULT IS TODAY'S SHIPPED PATH, AND THAT IS THE POINT. Every knob
 * here is the value the code used before it was a knob. A device that reaches
 * none of the priors below therefore computes and schedules exactly what it
 * did, which is what makes adding a prior a safe change to one device rather
 * than a risky change to all of them.
 * @type {Tuning}
 */
export const DEFAULT_TUNING = Object.freeze({
  // 🔴 ONE SWITCH FOR THE WHOLE f16 PATH. "auto" is the feature test every
  // call site used to make for itself, in eleven places across three models;
  // `false` forces the f32 path on a device that HAS the feature. It exists
  // for three reasons and all three have come up: to A/B the f16 path in one
  // process rather than by relaunching the browser, which is what this
  // repository's own measurement rules ask for; to switch it off if Dawn's
  // NVIDIA CTS worry ever turns out to bite (see docs/A100.md); and because
  // "does this device have the feature" and "should this fold use it" are
  // different questions that were the same expression.
  //
  // It gates AUTOMATIC selection only. A caller that explicitly asks for f16 -
  // every bench arm and every differential checker that forces a precision -
  // still gets it, so the suite can test a kernel the default is not using.
  halfPrecision: "auto",
  // The 32-row tile, via chooseLinearTile's wide/narrow pair.
  linearTallTile: false,
  // 🔴 THE MATRIX PROJECTION'S GEOMETRY, OR null FOR ITS DEFAULT. This is a
  // knob rather than a constant because the best block is a property of the
  // device's tile and its workgroup storage, not of the kernel - and because a
  // sweep needs somewhere to put its answer. `chooseMatrixLinear` reads it and
  // `tools/gpu/profile-af2-block.js --matrix=` writes it, which is how the
  // shipped 128x128x16x1x8 was picked in situ rather than from a bench.
  // 🔴 null MEANS OFF, NOT "DEFAULT GEOMETRY", AND THAT DISTINCTION IS THE
  // WHOLE POINT. The matrix projection is measured on ONE device. An M2 also
  // reports subgroup matrix configs - f32 AND f16, both 8x8x8 - so a gate that
  // only asked "does this device have matrix units" would have switched Apple
  // onto an unmeasured kernel whose block is sized for a 16x16x16 tile: at
  // 8x8x8 the same 128x128 block is 32 accumulators a subgroup, and
  // tools/gpu/gemm-matrix.js records that shape spilling to 122-172 GFLOP/s
  // against 1082-1466. Opting in per architecture keeps an unmeasured device on
  // the kernel its numbers were taken with.
  matrixLinear: null,
  // 🔴 SPLITTING THE DIFFUSION PROJECTIONS' INNER EXTENT, AND RAISING THE TOKEN
  // TILE BEHIND IT. null is the shipped behaviour: no split, and the tile rule
  // above. The two are ONE knob because neither works alone - the tile raises
  // arithmetic intensity from 0.5 to TILE/2 flop a byte and pays in workgroups,
  // the split buys the workgroups back and pays in partial traffic. Measured on
  // this shape at 68 tokens: plain 1.97 TFLOP/s, split alone 2.63, tile alone
  // 2.18, both 4.54. And it inverts with size, which is what `crossover` is
  // for: past a few hundred tokens the tile no longer starves the device and
  // the split is pure cost. See docs/A100.md.
  diffusionSplitK: null,
  // One key per softmax rescale, scalar q.k reduction.
  attentionGroup: 1,
  attentionVectorScore: false,
  // One query per invocation.
  attentionQueriesPerLane: 1,
  // 🔴 THE DIFFUSION TRANSFORMER'S TOKEN TILE, null MEANING "the model's own
  // rule". That rule is `min(4, fits(channels))` and its comment records the
  // M2 sweep that chose it, ending "the ceiling was never what bound this,
  // occupancy was" - which is exactly why the answer inverts on a machine with
  // ten times the cores. A smaller tile is more workgroups.
  diffusionTokenTile: null,
  // 🔴 SUBGROUP REDUCTIONS IN THE DIFFUSION ATTENTION. Needs the `subgroups`
  // feature and a fixed subgroup size; null is the portable barrier tree.
  diffusionAttendSubgroups: null,
  // 🔴 THE SAME CAPABILITY FOR THE ADAPTIVE LAYER NORM'S REDUCTIONS, asked for
  // separately because they are a different kernel with different numbers. It
  // runs four barrier-tree reductions per token and the tile sweep says those
  // dominate it - adaln gets 1.85x SLOWER at tile 2, which is what a
  // barrier-bound kernel does and the opposite of what a weight-bound one does.
  diffusionNormSubgroups: null,
  // 🔴 SPLITTING THE ADAPTIVE LAYER NORM ACROSS OUTPUT CHANNELS. One workgroup
  // per token is 68 of them at a 68-token fold, 8% of an A100's cores; a split
  // multiplies that by C/lanes and pays only for recomputing the norms.
  diffusionNormSplit: null,
  // 🔴 HOISTING THE ZERO-INIT GATE OUT OF THE BLOCK LOOP AND BATCHING IT OVER
  // THE BLOCKS. The gate is `bias + cond @ W_zero`, and the conditioning does
  // not move with the block, so all twenty-four can be one dispatch. It costs a
  // concatenated copy of those weights - ~14 MB at f16 across 24 blocks - and
  // needs them resident, so it is opt-in rather than a default.
  diffusionBatchedGates: null,
  // 🔴 THE BATCHED PROJECTION'S OWN TOKEN TILE, null meaning its default of 8.
  // It is the one kernel here that wants a BIG tile: the block axis already
  // gives it workgroups, so what binds it is re-reading the same six weight
  // slices once per token group - 5.8 GB a step at a tile of 1 against 0.76 at
  // 8, at 68 tokens.
  // 🔴 THE DIFFUSION ATTENTION'S KEY CHUNK, null meaning the model's 64. That 64
  // is a hardcoded default sized against a 16 KiB workgroup-storage limit, and a
  // device reporting 48 KiB can stage more keys at once - fewer barriers and
  // more lanes in the dot loop, traded against workgroups resident per SM.
  diffusionAttendKeyChunk: null,
  diffusionGateTile: null,
  // The diffusion transformer's workgroup width. null is 256, its shipped value.
  diffusionLanes: null,
  // 🔴 THE ATOM STACKS' ROW TILE, DIRECTLY, BECAUSE A WORKGROUP TARGET CANNOT
  // EXPRESS WHAT THIS DEVICE WANTS. outputRowTileFor takes the largest tile
  // leaving 256 workgroups, and on an A100 the measured optima are tile 1 at
  // 1632 atoms (project-* 1.34 ms against 4.37 for the shipped tile 4) and tile
  // 8 at 5760 (1.73 against 3.58 for tile 1). Tile 1 needs a target above 816
  // and tile 8 needs one at or below 720, so no single number gives both.
  //
  // 🔴 AND TILE 4 AT 5760 ROWS IS A CLIFF, NOT A POINT ON A CURVE: 15.79 ms
  // against 1.73 for tile 8 and 3.58 for tile 1, nine times worse than either
  // neighbour. That is unexplained and it is why this rule names tiles rather
  // than a target - a target of 1024 lands on it exactly, and shipping that
  // would have been a 1.26x REGRESSION at 240 tokens while looking like a
  // 1.18x win at 68.
  //
  // null is the shipped rule, unchanged.
  atomRowTile: null,
  // 🔴 WHETHER A FOLD KEEPS THE TRUNK'S WEIGHTS BETWEEN FOLDS. `fold.js` calls
  // releaseResidentWeights("w.") when the trunk is done so the pairformer's
  // weights are not on the device beside the sampler's own 378 MiB - which is
  // right on a phone and on unified memory, and is a pure loss on a card with
  // tens of gigabytes.
  //
  // 🔴 IT IS NOT FREE TO GET WRONG IN EITHER DIRECTION, WHICH IS WHY IT IS A
  // PRIOR AND NOT A HEURISTIC. Measured at 68 tokens on an A100: a warm fold
  // re-uploads 561 MiB in 1341 writeBuffer calls, and the host waits for those
  // transfers inside the pairformer's onSubmittedWorkDone - invisible to
  // tools/gpu/profile.js, which only times compute passes and so reported
  // 144 ms of a 1300 ms fold. Keeping them costs device memory that a small
  // device does not have; the whole trunk is ~567 MiB resident.
  //
  // null is the shipped rule: give them back every fold.
  keepTrunkWeights: null,
  // 🔴 AND THE SAMPLER'S, WHICH ARE RELEASED FOR THE SAME REASON AND MEASURED
  // SEPARATELY. `fold.js` gives back "difftx." and "cond." once the sampler is
  // done, because the fold's PEAK is after that point - the confidence head is
  // four more pairformer blocks and allocates the whole pair scratch again, and
  // at 272 tokens these were 378 MiB of a 1214 MiB peak held for a stage that
  // cannot use them. The cost, as that comment says, is that the next fold
  // packs and uploads them again.
  //
  // Two priors rather than one because the trade is not the same: the trunk's
  // weights are re-uploaded before the pairformer, the sampler's before the
  // denoiser, and only measuring them apart says which is worth the memory.
  //
  // null is the shipped rule: give them back every fold.
  keepSamplerWeights: null,
  // 🔴 HOW MANY PAIRFORMER BLOCKS GO OUT BEFORE THE HOST WAITS. Each wait is a
  // full pipeline drain; the window exists to bound the queue and let an abort
  // land, not because the memory needs it. 16 is measured - but it was measured
  // over a 59-token stack BEFORE the weights were kept resident, when every
  // block was also re-uploading them, so the shape of the curve it was fitted
  // to no longer exists. null keeps the 16 every caller had.
  pairformerSubmissionWindow: null,
  // 🔴 STAGE THE ATTENTION'S KEYS COALESCED. See the note in the kernel.
  diffusionAttendStageKeys: null,
  // 🔴 HOW MANY WORKGROUPS THE SINGLE PROJECTION AIMS AT. See
  // singleProjectSplits: 110 and a candidate list stopping at 3 are an M2's
  // answer, and the constant is written into the rule rather than measured per
  // device.
  singleProjectWorkgroupTarget: 110,
  singleProjectMaxSplits: 3,
  // 🔴 THE TRIANGLE'S PROJECTION TILE, null meaning src/triangle/shaders.js's
  // 32x16. Note this one goes the OTHER way from the diffusion token tile: it
  // wants a BIGGER tile, because its dispatch already has tens of thousands of
  // workgroups and occupancy is long since saturated, so what is left to win
  // is traffic. Which direction a tile wants to move is a question about
  // whether the dispatch already fills the device, not about the device.
  trianglePairProjectTile: null,
  // 🔴 HOW MANY THREADS A TRANSITION SHOULD AIM TO HAVE IN FLIGHT. The
  // transition's dispatch is rows-only, so a short track cannot fill a large
  // device at any tile; the workgroup WIDTH is the only axis left. null keeps
  // the 128 every caller had. See transitionWidth.
  transitionThreadTarget: null,
  // 🔴 THE SINGLE ATTENTION'S TWO WORKGROUP WIDTHS. Both were 64 everywhere;
  // null keeps that. See createSingleAttentionShaders.
  singleProjectLanes: null,
  singleProjectOutLanes: null,
});

/**
 * Priors by `adapter.info.architecture`, then by `vendor` as the coarser
 * fallback. Keep the measurement in the comment or the entry is a guess.
 */
const PRIORS = new Map([
  // A100-SXM4-40GB, Chrome 152, Dawn/Vulkan, driver 535.129.03, clocks locked
  // at 1410 MHz. Every entry names the measurement it came from; docs/A100.md
  // has the tables.
  //
  //   linearTallTile   16x16x8x4 (128x64, 256 lanes) 1.067 ms against the
  //                    shipped 32x64's 1.655 - 1.551x, relRMS 0, and a whole
  //                    AF2 fold bit identical (fold-af2.js checksum -1397134).
  //
  //   attentionGroup   auto/g4v 1.494 ms against auto's 1.831 - 1.226x in f32
  //                    and 1.078x once f16 staging is on. relRMS 3e-7: it
  //                    reassociates the online softmax, so it is gated on
  //                    check-evoformer-attention.js and not on a stopwatch.
  //
  //   queriesPerLane   q2 measures 1.000x here and q4 0.773x, so there is
  //                    nothing to take; left at 1. An M2 measures 0.21x and
  //                    upstream's GB10 1.17x-1.42x, which is why this is a
  //                    per-device knob at all.
  //
  //   diffusionTokenTile  the transformer's dense passes cost the same at 59
  //                    tokens as at 400 - 6.8x the work for 1.07x the time -
  //                    so at small sizes they underfill a 108-SM machine and
  //                    want the smallest tile there is. At large sizes they do
  //                    not, and the tile's weight traffic (proportional to
  //                    tokens/tile) binds instead. Swept, tile 1 / 2 / the
  //                    model's 4, on bench-diffusion-transformer.js:
  //
  //                      59 tokens   **24** / 29 / 36
  //                      150         **44** / 45 / 52
  //                      200         62 / **60** / 67
  //                      240         77 / **71** / 77
  //                      300        129 / **114** / 112
  //                      400        157 / 157 / 162
  //
  //                    🔴 AND A SINGLE VALUE HERE WAS A REGRESSION. This was
  //                    plain `1` for one revision - right to 150 tokens, and
  //                    making a 300-token fold **129 ms against the shipped
  //                    112**, 15% slower than the M2's rule at exactly the
  //                    sizes where a fold is slow enough to care. Measure a
  //                    knob across the range its caller varies, not at one
  //                    shape.
  //
  //   singleProject...Target/MaxSplits  NOT TAKEN. bench-single-project.js
  //                    measures 1.46x-2.50x for a 1200-workgroup target
  //                    against the rule's M2-sized 110, and
  //                    `singleProjectSplits` duly returns 6 where it returned
  //                    1 - but `single.project` in a real trunk does not move
  //                    by 0.3 ms either way. The bench's kernel is not the
  //                    pipeline's. The rule keeps its new parameters so the
  //                    axis stays expressible.
  //
  //   singleProject...Lanes  THE REWRITE, AND IT MOSTLY DID NOT PAY. The three
  //                    single-attention shaders hardcoded workgroup_size(64)
  //                    in thirteen places including a reduction tree; they
  //                    take a width now. Trunk GPU total: 64/64 **368.8,
  //                    369.1**; 64/256 **365.9, 365.8**; 128/128 367.4;
  //                    128/256 366.4; 128/64 369.4. So 0.85%, not the 5%
  //                    predicted - and `project` at 128 lanes is WORSE (15.60
  //                    -> 16.09 ms). The transition gained 4.19x from the same
  //                    move because ONE workgroup walked all its channels
  //                    sequentially; `project` already blocks perThread
  //                    outputs a lane so the normalised token is read once for
  //                    all of them, and widening buys those reads back. Ask
  //                    what a kernel already does per lane before widening it.
  //
  //   trianglePairProjectTile  32x32 against the shipped 32x16, contractTile
  //                    left at its 32x32 default. `tri.project-out` **30.22 ->
  //                    24.77 ms (1.22x)** and `tri.project` 37.90 -> 36.29,
  //                    contract untouched, trunk total 376.1 -> 368.9,
  //                    relRMS 0.
  //
  //                    🔴 AND IT WAS REJECTED ONCE ON A BAD MEASUREMENT. A/B'd
  //                    through fold.js's pairformer stage timer it read 815 ms
  //                    against 766 - the wrong way round - because that timer
  //                    spans 758 to 994 ms across repeats of ONE
  //                    configuration. Only bench-trunk.js --profile, whose
  //                    pass timings reproduce to 0.2 ms, can see 2%.
  //
  //   diffusionAttendSubgroups / ...StageKeys  the diffusion attention owns
  //                    one (token, head) a workgroup: at 240 tokens 3840
  //                    workgroups of 256 lanes, 983,040 threads where the
  //                    device holds 221,184. It is oversubscribed, not
  //                    starved, and still ran at 2.5% of the arithmetic
  //                    ceiling. Three things were tried and only the third is
  //                    the answer:
  //
  //                      workgroup width  64, 128 and 256 lanes measure the
  //                                       SAME; only 32 differs and is worse.
  //                                       So the idle lanes in `for (d = local;
  //                                       d < DIMENSION; ...)` - 48 of 256 -
  //                                       are not the problem.
  //                      subgroup reduce  the two barrier trees are 16 of
  //                                       about 20 barriers, and removing them
  //                                       is **5%** of the pass.
  //                      staged keys      the logit loop gave a lane a whole
  //                                       KEY, 48 contiguous floats with 3072
  //                                       bytes between lanes. Staging the
  //                                       chunk with a (key, channel)-indexed
  //                                       loop makes the global read coalesced
  //                                       and the dot product read workgroup
  //                                       memory: **1.30x**.
  //
  //                    Together, `attend` 9.32 -> 6.38 ms at 240 tokens and
  //                    24.19 -> 16.43 at 400, both **1.46x**; the whole
  //                    denoiser step 49.42 -> 46.14 and 79.41 -> 71.66. Key
  //                    chunk 64 against 128 (6.38 against 6.90 at 240) and 32
  //                    (16.98 against 16.43 at 400); 256 does not fit the
  //                    workgroup storage.
  //   transitionThreadTarget  the trunk's SINGLE transition is `n` rows, not
  //                    n^2, and the transition kernel dispatches rows ONLY -
  //                    so at 200 tokens it runs 200 workgroups of the default
  //                    128 threads, 25,600 where this device holds 221,184. It
  //                    measured 2.3% of the arithmetic ceiling for 13.6% of
  //                    the trunk's GPU time. Width 512: **54.83 -> 13.13 ms,
  //                    4.19x**, relRMS 2.8e-7. The same rule leaves the PAIR
  //                    transition at 128 - 40,000 rows is already 5000
  //                    workgroups, and 512 costs it 0.70x.
  ["ampere", {
    linearTallTile: true,
    attentionGroup: 4,
    attentionVectorScore: true,
    diffusionTokenTile: { below: 1, atOrAbove: 2, crossover: 175 },
    singleProjectOutLanes: 256,
    trianglePairProjectTile: { rows: 32, columns: 32 },
    diffusionAttendSubgroups: true,
    diffusionAttendStageKeys: true,
    // Swept in situ at 68 tokens as a whole denoiser step, repeated to
    // separate the answer from this box's ~2% drift: off 29.68 ms, 4/2 26.82,
    // 8/4 26.27 and 26.33, **16/4 26.08 and 26.06**, 16/8 26.20 and 26.13.
    // Sixteen parts is where qkvg stops improving (4.34 -> 1.24 ms) and the
    // partials, 13 MiB shared between the two split kernels, still fit.
    // 🔴 normSplits IS 4 ON THE FOLD, AND THE MODULE BENCH CANNOT SEE IT.
    // Paired folds, arms in separate processes with the order alternating:
    // at 200 steps nine pairs, eight favouring, **+0.098 s of 3.70 (2.6%)**;
    // at 20 steps eight pairs, seven favouring, **+0.0079 s**, which is 80% of
    // the 0.0098 a per-step effect predicts. It SCALES WITH STEP COUNT, which
    // is what rules out a fixed per-process offset. pLDDT is 84.208873 in every
    // one of those folds and the module relRMS is 1.3e-4.
    //
    // 🔴 AND THE KERNEL'S OWN PASS TIME IS A WASH, WHICH IS NOT A CONTRADICTION
    // BUT IS NOT EXPLAINED EITHER. Per block from --profile: fused
    // adaln + ffw-adaln 0.113 ms, split 0.118 including both reduces. So the
    // fold's win is not the adaln kernels running faster, and no mechanism
    // here has been measured - do not quote occupancy for it, which is what
    // this comment said before the profile was read.
    //
    // 🔴 bench-difftx-splits.js DISAGREES AND IS THE WRONG INSTRUMENT. It read
    // 1.025 over eleven pairs and **0.982 over fifteen** - an instrument that
    // contradicts itself by 4% has no resolving power at this size. It also
    // awaits onSubmittedWorkDone after every call, so it measures the LATENCY
    // of one transformer call where a fold measures the throughput of two
    // hundred, and the split trades launch count for occupancy.
    //
    // 🔴 AND IT WAS BROKEN WHEN 311d7a4 MEASURED IT A LOSS. One ternary handed
    // the bias to the reduce epilogue as well as the fused path, which added it
    // again - a doubled bias on a scale feeding a sigmoid, relRMS 0.813 and
    // pLDDT 66.3 against 84.2. Every figure for this knob from before that fix
    // describes a kernel that was not computing the model.
    // 🔴 attnSplits IS 4 AND ITS TILE IS 2, AND NEITHER WORKS WITHOUT THE OTHER.
    // The split alone measured **+1.1% and non-monotonic** (2 beat 4 in one
    // round of three) - because a K split divides the work among parts and each
    // part still re-reads its weight slice once per TOKEN GROUP, so it moves
    // occupancy and leaves bandwidth exactly where it was. The tile is what
    // divides the traffic, and the split is what pays for the workgroups the
    // tile costs. Together, against the previous behaviour:
    //
    //   out-tile=1 (was)   3.1440 s      -
    //   out-tile=2         3.1107     +1.07%
    //   out-tile=4         3.1315     +0.40%
    //   attn 4 / tile 4    3.0293     +3.8%
    //   attn 4 / tile 2    2.9570     **+6.3%**
    //
    // 🔴 AND THE GAIN IS SMALLER THAN THE TRAFFIC ARITHMETIC PREDICTS, for a
    // reason worth keeping: this card has **40 MB of L2** and
    // attention-output's whole weight set is 24 x 768 x 768 x 2 = 28 MB, so most
    // of the re-reading was already being absorbed. The bandwidth model says
    // 1.28 ms and the kernel measured 1.43 at a tile of 1 - close, but the
    // model's 4x at a tile of 4 does not arrive.
    // 🔴 AND attnSplits CHANGES THE STRUCTURE, WHERE THE TILE DOES NOT. It
    // regroups a 768-term sum, so it is a reordering like every other K split
    // here: `tools/diff-fold-coords.py` measures **33 of 574 atoms moved, by
    // 0.001 A** - the same class as the kSplits: 16 and outSplits: 4 this rule
    // has always shipped, and not the bit-exactness the conditioning hoist has.
    // meanPlddt is 84.20887255253277 either way, which is exactly why that
    // number is not the gate it looks like.
    // 🔴 THE CROSSOVER WAS 175 AND THAT NUMBER EXPIRED. It came from a
    // measurement at 240 tokens - "the shipped rule disengages and the step is
    // 46.63 ms against 46.68 before, unchanged" - which was true of those
    // kernels and stopped being true when the conditioning was hoisted and the
    // tiles became per kernel. Above it EVERY split disengages at once, so the
    // whole mechanism this rule describes was switched off exactly where the
    // gap to native JAX is worst.
    //
    // Re-measured by length, 60 steps, splits off against splits on:
    //
    //     tokens   splits off   splits on   benefit
    //        272     4.9185 s     4.6895     +4.9%
    //        408     8.7013       8.5645     +1.6%
    //        544    13.7658      13.7410     +0.2%  (a wash)
    //
    //   the linear trend crosses zero at ~536 tokens; 512 sits above the last
    //   clearly-positive point and below the wash.
    //
    // 🔴 AND IT DECAYS FOR A REASON THE ARITHMETIC GIVES: the partials are
    // read and written once a block, so their traffic grows with the token
    // count - about 4.8 GB a step at 256 tokens and 10.3 at 544, against a
    // 1.5 TB/s card. A crossover SHOULD exist; 175 was simply the wrong one.
    diffusionSplitK: { splits: 16, tile: 4, crossover: 512, outSplits: 4,
                       attnSplits: 4, attnTile: 2, normSplits: 4 },
    // On, with the geometry derived from this device's tile. See the note on
    // matrixLinear in DEFAULT_TUNING for why this is opt-in.
    matrixLinear: {},
    diffusionNormSplit: true,
    // 🔴 1024, WHICH RESOLVES TO A ROW TILE OF 1 AT THE SIZES A FOLD SEES. The
    // shipped 256 picks a tile of four at 1632 atoms - 408 workgroups of 64
    // lanes, 12% of this device - and the atom projections were the single
    // largest item in a step because of it. Swept at 68 tokens: target 256
    // gives project-* 4.37 ms, 512 gives 2.04, 1024 gives 1.34 and 2048 gives
    // 1.33, so it saturates where the tile reaches 1. The rule still grows the
    // tile on a big enough structure, which is what it is for.
    atomRowTile: { below: 1, atOrAbove: 8, crossover: 3000 },
    // 🔴 THE ZERO GATES, HOISTED AND BATCHED. Paired folds at 200 steps, order
    // alternating: **6/6 favour it, median 1.0635, -0.213 s of 3.60, t=27.3**,
    // and pLDDT is 84.208873 in all twelve. The profile predicted 1.17 ms a
    // step and the fold returned 1.06, which is the agreement normSplits above
    // does NOT have - the mechanism here is measured, not inferred.
    // Affordable on this card: ~14 MB of duplicated f16 weights and 10 MB of
    // gate buffers at 68 tokens, against 40 GB.
    diffusionBatchedGates: true,
    keepTrunkWeights: true,
    keepSamplerWeights: true,
    transitionThreadTarget: 100000,
  }],
], );

const VENDOR_PRIORS = new Map([
  // 🔴 NOTHING FOR "apple" ON PURPOSE. Its measurements ARE the defaults above,
  // and an entry that restated them would be a second place for them to drift.
]);

const RECORDED = new WeakMap();
const OVERRIDES = new WeakMap();
const CACHE = new WeakMap();

/**
 * Remembers what the adapter said about itself, so code holding only a
 * `GPUDevice` can still ask. Called by `requestAlphaFoldDevice`; a device
 * built any other way simply has no identity and gets the defaults.
 */
export function recordAdapter(device, adapter) {
  const info = adapter?.info ?? {};
  RECORDED.set(device, {
    vendor: String(info.vendor ?? "").toLowerCase(),
    architecture: String(info.architecture ?? "").toLowerCase(),
    // 🔴 THE SHAPES ARE ONLY ON THE ADAPTER, and a kernel that wants them holds
    // a device. They are also WebIDL interfaces rather than plain objects, so
    // spreading or JSON-ing one yields `{}` - the fields have to be named. See
    // tools/gpu/probe-subgroup-matrix.js, which learned that the hard way.
    matrixConfigs: [...(info.subgroupMatrixConfigs ?? [])].map((c) => ({
      componentType: c.componentType,
      resultComponentType: c.resultComponentType,
      M: c.M, N: c.N, K: c.K,
    })),
  });
  CACHE.delete(device);
  return device;
}

/**
 * A measured answer, which beats any prior. `tools/gpu/probe-tuning.js` is the
 * intended caller; a bench forcing one arm is the other.
 * @param {Partial<Tuning>} tuning
 */
export function setDeviceTuning(device, tuning) {
  OVERRIDES.set(device, { ...(OVERRIDES.get(device) ?? {}), ...tuning });
  CACHE.delete(device);
}

/**
 * @returns {{vendor: string, architecture: string, software: boolean,
 *            tuning: Tuning}}
 */
export function deviceProfile(device) {
  const cached = CACHE.get(device);
  if (cached !== undefined) return cached;
  const { vendor = "", architecture = "", matrixConfigs = [] } = RECORDED.get(device) ?? {};
  // 🔴 SwiftShader ANSWERS `requestAdapter` ON LINUX/NVIDIA BY DEFAULT and does
  // not announce it - see docs/A100.md. Naming it here means a caller can
  // refuse to believe a benchmark rather than quietly reporting one taken on a
  // CPU at a thousandth of the speed.
  const software = vendor === "google" || architecture === "swiftshader"
    || architecture === "software" || vendor === "mesa";
  const prior = PRIORS.get(architecture) ?? VENDOR_PRIORS.get(vendor) ?? {};
  const profile = Object.freeze({
    vendor,
    architecture,
    software,
    matrixConfigs: Object.freeze(matrixConfigs),
    tuning: Object.freeze({
      ...DEFAULT_TUNING,
      ...(software ? {} : prior),
      ...(OVERRIDES.get(device) ?? {}),
    }),
  });
  CACHE.set(device, profile);
  return profile;
}

/** Shorthand, because every caller wants one knob and not the object. */
export const deviceTuning = (device) => deviceProfile(device).tuning;

/**
 * The matrix tile this device would rather a GEMM used, or null if it has no
 * matrix units - or has them and the caller cannot feed them.
 *
 * 🔴 THE COMPONENT TYPE IS NOT NEGOTIABLE AND IS NOT THE SAME EVERYWHERE. An
 * M2 offers f32 tiles at 8x8x8; this A100 offers **no f32 component type at
 * all**, only f16 and 8-bit integer, and asking for the f32 one does not fall
 * back - `createComputePipeline` rejects the shader with "Unknown configuration
 * is M(8), N(8), K(0), f32". So a caller states what it can feed and gets null
 * rather than a shape that will not compile. See docs/A100.md.
 *
 * 🔴 AND f32 ACCUMULATION IS NOT A CONCESSION. Measured at 309.7 TFLOP/s
 * against 310.9 for f16 accumulation on the same units, so the default prefers
 * it: the multiply takes halves either way and there is no reason to also sum
 * in one.
 *
 * @param {GPUDevice} device
 * @param {object} [options]
 * @param {"f16"|"f32"} [options.element] what the caller can hand the units.
 * @param {boolean} [options.preferWideResult] prefer an f32 accumulator.
 *   Default true, and free on every device measured.
 */
export function deviceMatrixConfig(device, options = {}) {
  if (!device?.features?.has?.("chromium-experimental-subgroup-matrix")) return null;
  const element = options.element ?? "f16";
  const preferWideResult = options.preferWideResult ?? true;
  const candidates = deviceProfile(device).matrixConfigs
    .filter((c) => c.componentType === element);
  if (candidates.length === 0) return null;
  const rank = (c) => (
    // Widest tile first - it is the most arithmetic per load - then the wider
    // accumulator, which costs nothing.
    c.M * c.N * c.K * 4
    + (preferWideResult && c.resultComponentType === "f32" ? 2 : 0)
  );
  return candidates.slice().sort((a, b) => rank(b) - rank(a))[0] ?? null;
}

/**
 * Should this device's AUTOMATIC kernel selection use half precision?
 *
 * The one place `device.features.has("shader-f16")` is asked on behalf of a
 * shipping path. Returns false when the device lacks the feature OR when the
 * switch above is off; a caller forcing a precision must test the feature
 * itself, because forcing is exactly the case the switch is not about.
 */
export function halfPrecisionAvailable(device) {
  if (device?.features?.has("shader-f16") !== true) return false;
  return deviceTuning(device).halfPrecision !== false;
}

/** Turn the whole f16 path on or off for this device. */
export const setHalfPrecision = (device, enabled) =>
  setDeviceTuning(device, { halfPrecision: enabled ? "auto" : false });

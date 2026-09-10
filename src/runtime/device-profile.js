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
  // 🔴 A STACK'S DISPATCHES SHARE ONE COMPUTE PASS. WebGPU orders them and
  // makes each one's writes visible to the next, so this is a pure encoding
  // change - AF3 at 200 steps is 3.03-3.11 s against 3.12-3.16, pLDDT identical.
  // `profileDevice` sets it false, because one pass a dispatch is the only
  // shape profile.js can attribute; the profiled number is the slower one.
  batchComputePasses: true,
  // 🔴 A DEVICE PACK THAT REFUSES STOPS THE FOLD, unless this says otherwise.
  // The alternative is what it used to do: fall back to packing on the host,
  // which is correct, silent, and 300 ms slower - a bug with no symptom but a
  // number. A bundle that genuinely cannot be decoded on the device (float32,
  // or a fixture built over plain arrays rather than a store) sets this once;
  // a descriptor that merely lost its sources on the way through a spread
  // should fail loudly instead. See DeviceWeightRefusal.
  allowHostWeightPacking: null,
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
  // 🔴 HOW MANY BYTES OF PAIRS THE OUTER PRODUCT MEAN'S FAST PATH HOLDS AT ONCE.
  // null is OPM_PAIR_BLOCK_BYTES. It is a WORKING SET, not a limit: the path
  // runs at every length on every device whatever this says, and the number
  // only decides how many blocks it takes. Clamped down by what the device can
  // actually bind, so a phone gets more blocks rather than a failed fold. It is
  // a knob because the answer trades dispatch count against occupancy and that
  // trade is a property of the device - see docs/AF2.md for the A100 sweep.
  opmPairBlockBytes: null,
  // 🔴 HOW MANY PAIRS ONE OUTER-PRODUCT-MEAN OUTPUT WORKGROUP CARRIES. null is
  // OPM_PROJECT_OUTPUT_PAIRS. It is here rather than a constant because the
  // measurement that fixed the constant at 2 was taken on an M2, where 4 costs
  // more occupancy than it saves traffic - and this kernel is bound by the
  // WEIGHT read, which every pair in a workgroup shares, so the trade is
  // entirely a property of the device's workgroup storage against its L2.
  opmProjectOutputPairs: null,
  // 🔴 FLASH ATTENTION ON THE MATRIX UNITS. The four flash kernels are 38% of an
  // AF2 block at 825 residues and the units do the query-key reduction in
  // hardware. See src/evoformer/attention-matrix.js - and note that a GEMM
  // benchmark at K = head_dim predicts the opposite and asks a different
  // question: a flash attention's reuse is in its loop, not in K.
  attentionMatrix: null,
  // Subgroups a workgroup and keys a tile for that kernel; null takes its
  // default. Both are fixed costs the tile amortises - see attention-matrix.js.
  attentionMatrixTile: null,
  // 🔴 AND THE SAME UNITS ON AF3's `grid.attend`, WHICH IS A DIFFERENT KERNEL
  // AND A DIFFERENT KNOB. It is the largest pass in the pairformer and the only
  // CUBIC one, so it leads by more on every longer chain; OpenDDE runs the same
  // track and gets it too. Separate from `attentionMatrix` because the two
  // bodies differ - no gate, no uniform, a bias that is always present - and
  // because the geometry that suits one is 5-6% wrong for the other. See
  // src/af3/grid-attention-matrix.js.
  gridAttendMatrix: null,
  // Its geometry, "subgroupsXkeys"; null takes GRID_ATTEND_MATRIX_DEFAULT_TILE.
  gridAttendMatrixTile: null,
  // 🔴 THE PAIR TRANSITION AS THREE PASSES INSTEAD OF ONE, ON THE MATRIX UNITS.
  // Whether it pays is a CHANNEL WIDTH question and not only a device one: the
  // fused kernel holds the widened row in workgroup memory, so its row tile
  // halves each time the channels double. Measured at 200 tokens against the
  // fused kernel's own best tile - 1.13x at AF3's 128 channels, 2.77x at
  // ESMFold2's 256, 3.71x at OpenDDE's 384. It brings back the widened tensor
  // the fusion exists to avoid, chunked over rows. See
  // src/af3/transition-webgpu.js.
  pairTransitionSplit: null,
  // 🔴 HOW BIG THE SPLIT TRANSITION'S WIDENED ACTIVATION MAY GET, in MiB; null
  // is 64. It is a SPEED knob as well as a memory one - a chunk is its own
  // dispatch, and one that does not fill the device leaves it idle - so a card
  // with room should raise it. See transitionSplitChunkRows.
  pairTransitionChunkBytes: null,
  // 🔴 AND THE TRIANGLE PROJECTION ON THE SAME UNITS, which with the transition
  // split is the LARGEST kernel left in an ESMFold2 trunk - 130.6 ms of 494.
  // Unlike the transition it needs no new memory: its source and both its
  // outputs are pair-sized scratch the track already holds. It does need the
  // four projection matrices interleaved, which is a reshape at pack time and
  // costs no bytes. See src/triangle/project-matrix.js.
  triangleProjectMatrix: null,
  // 🔴 THE BLOCK THE STAGED MATRIX GEMMs SHARE, "BMxBNxBKxSRxSC"; null is
  // 128x128x32x2x4. Four kernels use it now - the transition's two halves and
  // the triangle's two projections - and after the split and the projections
  // landed they are the top four of an ESMFold2 trunk within 8% of each other,
  // so the geometry is the next thing to move rather than another kernel. It is
  // ONE knob until a sweep shows the four want different answers.
  stagedMatrixBlock: null,
  // 🔴 READ THE RIGHT OPERAND OUT OF THE WEIGHT BUFFER INSTEAD OF STAGING IT.
  // With `subgroupRows` 1 - which the swept block above has - every subgroup
  // owns its own columns, so the staged weight panel is read exactly once and
  // the workgroup memory is a detour; the source panel, which eight subgroups
  // share, still pays for itself. It is two thirds of the staging, and
  // tools/gpu/probe-staged-gemm-parts.js prices the staging loop at 3.10 ms of
  // a 4.58 ms kernel. It needs the weight BUFFER to hold halves, so it comes
  // with `weightPrecision: "f16"` or it does nothing at all - see
  // directWeightsAllowed in src/runtime/matrix-linear.js, which is what turns
  // this request into an answer per kernel.
  stagedMatrixDirectWeights: null,
  // 🔴 THE ACCUMULATOR'S WIDTH, null meaning the device config's own. It is not
  // a throughput knob - probe-matrix-ceiling.js measures 309.7 TFLOP/s into f32
  // against 310.9 into f16, so the units accumulate in f32 for free - it is a
  // REGISTER knob. Four f32 results of 16x16 are 32 registers a lane before
  // anything else, and eight workgroups of 256 threads need 32 registers a
  // thread in total to fill this card. Halving the accumulator is the only
  // lever left on the occupancy this kernel is bound by.
  stagedMatrixResult: null,
  // 🔴 DOUBLE-BUFFER THE STAGING: issue the next K panel's global reads BEFORE
  // this panel's multiplies, so their latency is covered by work the workgroup
  // already has rather than by whatever else the SM happens to hold. Bit-exact
  // - the same reads in the same order, held in registers for one panel - and
  // 1.43x on the standalone GEMM with staged weights, 1.33x on top of the
  // direct read. It needs neither f16 weights nor a narrower accumulator, so it
  // is the one of these three that costs nothing at all.
  stagedMatrixPrefetch: null,
  // 🔴 THE TWO WIDTH RULES, AS KNOBS, BECAUSE THEY WERE CALIBRATED AGAINST A
  // SLOWER GEMM. TRANSITION_SPLIT_MIN_CHANNELS and
  // TRIANGLE_PROJECT_MATRIX_MIN_CHANNELS are 192 because at AF3's 128 the split
  // was 1.8% of a trunk for 72 MiB - measured when the staged matrix kernel ran
  // at 20.6 TFLOP/s. It runs at 39.3 now, so the width at which the matrix path
  // starts to pay has moved and nothing could re-measure it without these.
  // null means the constant in the module that owns the rule.
  pairTransitionSplitMinChannels: null,
  triangleProjectMatrixMinChannels: null,
  // 🔴 GRID ATTENTION'S q/k/v/gate PROJECTION ON THE UNITS. It is the biggest
  // single pass in an OpenDDE trunk - 386 ms of 1876 at 256 tokens, ahead of
  // grid.attend - and 58 of AF3's 463. Unlike the triangle's, its weights are
  // ALREADY interleaved [k][4w + role], because the vector kernel wanted one
  // vec4 a cell; so this is a shader and not a layout migration. It refuses a
  // PACKED q/k/v/gate, which is two channels to a word and needs the vector
  // kernel's ownership rule. See src/af3/grid-project-matrix.js.
  gridProjectMatrix: null,
  // 🔴 AF2's q/k/v/gate PROJECTION ON THE UNITS. It is 20% of an evoformer
  // block at 400 residues and 512 sequences - 12.73 ms of 78.01, the largest
  // thing in the block still on the vector path - and as one packed GEMM the
  // same work prices at 3.659 ms against 6.364. It reaches the four matrices
  // through `weightIndex` rather than a repack, because that buffer is bound by
  // five shaders. See src/evoformer/attention-project-matrix.js.
  attentionProjectMatrix: null,
  // 🔴 THE OUTER PRODUCT MEAN'S CONTRACTION IN f16, null meaning f32. Only the
  // STAGED TILE and a per-chunk accumulator narrow; the running total stays
  // f32, so the sum a half carries is bounded by the chunk however deep the MSA
  // gets. That is the distinction alphafold2-webgpu's own source draws, and
  // theirs is on the wrong side of it: a whole-contraction f16 accumulator
  // takes a 508-row prediction from 96.80 pLDDT to 69.94. Needs `shader-f16`.
  // 🔴 AND IT DOES NOTHING ON THE MATRIX PATH, WHICH IS WHERE AMPERE IS. Swept
  // in an 825-residue block: opm.contract is 29.878 ms at null and 29.954 at
  // "f16", and the block 262.96 against 262.80. The staged matrix contraction
  // already narrows its tile to halves because the units take nothing else, so
  // this knob only reaches the hand-tiled vector kernel.
  opmContractPrecision: null,
  // 🔴 THE OUTER PRODUCT MEAN'S CONTRACTION ON THE MATRIX UNITS. It is the
  // biggest kernel in an AF2 block and it is a plain GEMM with the deepest K in
  // the model, which is docs/A100.md's own rule for when the units pay. Opt-in
  // per architecture, like matrixLinear and for the same reason.
  opmMatrixContract: null,
  // ...and its OUTPUT PROJECTION, which is a second GEMM with the same operands
  // one step on. Separate from the contraction so either can be measured, or
  // bisected, without the other.
  opmMatrixOutput: null,
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
    // 🔴 THE FLASH ATTENTIONS RUN ON THE MATRIX UNITS HERE. Swept in situ in
    // an 825-residue, 512-row block, which is the only place the answer is
    // legible: the four attentions go 116.9 -> 69.3 ms (1.69x) and the block
    // 310.9 -> 263.5 (1.18x). The tile is a joint optimum between the scalar
    // softmax per key and the workgroup bytes per lane, and it MOVED when the
    // softmax got cheaper - 6x16 before the hoists, 4x32 after. Re-sweep it
    // before trusting it on another part; see src/evoformer/attention-matrix.js.
    attentionMatrix: true,
    attentionMatrixTile: "4x32",
    // 🔴 AND AF3's `grid.attend` ON THE SAME UNITS, SWEPT IN THE TRUNK. It is
    // the largest pass in the pairformer and the only cubic one; measured as
    // GPU pass time over eight blocks, at 32 MSA rows, medians reproducible
    // across processes to 0.15%:
    //
    //     tokens      200    300    384    400    512    640
    //     scalar    17.16  45.24  90.71 108.02 208.06 405.85
    //     2x16      11.23  34.31  71.01  74.57 165.36 321.44
    //     4x32      11.41  31.29  64.79  81.77 148.60 289.39
    //
    // 1.40x to 1.53x, and 4x32 wins four of the six. 400 is where it loses:
    // 400/64 is 6.25, so the row tile costs a seventh workgroup where 384
    // needs six, and the key tile is ragged on top of that.
    //
    // 🔴 AND AF2's TILE RULE DOES NOT GOVERN HERE, WHICH IS THE FINDING. That
    // kernel's five geometries ranked EXACTLY by workgroup bytes a lane. These
    // do not rank by it at all - at 512 the order is 4x32 (169 bytes a lane)
    // 148.6, 4x16 (136) 162.4, 2x16 (154) 165.4, 6x16 (130) 166.1 - and the
    // winner is the one that takes the MOST. AF2's kernel was occupancy-starved
    // and this one is not: a pairformer pass launches n x heads workgroups per
    // block, 16384 of them at 512 tokens over eight blocks, so the device is
    // full either way and the fixed costs a bigger tile amortises are what is
    // left to win. Re-sweep on any device that is not this one.
    gridAttendMatrix: true,
    gridAttendMatrixTile: "4x32",
    pairTransitionSplit: true,
    triangleProjectMatrix: true,
    // 🔴 SWEPT IN THE TRUNK, WHERE THE FOUR KERNELS THAT SHARE IT RUN. Total
    // GPU time of an ESMFold2 trunk at 300 tokens, 24 blocks:
    //
    //     64x128x16x1x8    349.8      128x128x16x2x4   386.7
    //     64x128x16x1x4    364.8      64x128x32x1x8    388.0
    //     64x64x16x1x4     373.8      128x128x32x2x4   405.4  (the old default)
    //     128x128x16x1x8   386.7      32x128x16x1x8    416.4
    //                                 256x128x16x1x8   503.2
    //
    // 1.16x for a knob. Two things move it and they pull the same way: a
    // smaller K panel (16 beat 32 and 64 at every block, which is what
    // matrix-linear.js's own note already said) and fewer accumulators a lane -
    // 64x128 with eight subgroups gives each lane FOUR results where 128x128
    // with 2x4 gives it eight, and this kernel is occupancy-bound the way
    // `grid.attend` is not. Re-sweep on another device.
    stagedMatrixBlock: "64x128x16x1x8",
    // 🔴 ON BY DEFAULT BECAUSE IT IS BIT-EXACT AND FREE. Double-buffering the
    // staging is the same reads in the same order, held in named registers for
    // one panel, so check-staged-matrix.js reads relRMS 0 across all 384 cases
    // and AF2's fold returns the same atom checksum to the integer. Measured
    // here: ESMFold2's trunk 332.3 -> 276.0 ms, OpenDDE's 2122.4 -> 1876.3,
    // AF2 at 400 residues 4.418 -> 4.120 s warm. The other two staged-matrix
    // knobs are trades and stay off; this one is not.
    stagedMatrixPrefetch: true,
    // 🔴 THE TWO WIDTH RULES MOVED WHEN THE GEMM GOT FASTER, AND THIS IS WHERE
    // THAT LANDS. Both constants are 192 because at AF3's 128 pair channels the
    // matrix path was worth 1.8% for 72 MiB - measured against a staged kernel
    // running at 20.6 TFLOP/s. It runs at 39.3 now. Re-measured on the AF3
    // trunk at 256 tokens: 529.6 ms of GPU at 192/192, 493.6 with the split,
    // 500.6 with the projection, 463.1 with both - 1.14x, and the warm pass
    // 1040 -> 966 ms.
    //
    // 🔴 AND IT IS MORE ACCURATE, NOT LESS. check-af3-block-any.js on that
    // bundle reads pair 1.15e-1 against a 1.5e-1 bound with the vector kernels
    // and 3.21e-2 with the matrix ones, single 1.47e-3 against 2.42e-4: the
    // staged path accumulates in f32 where the vector triangle accumulates in
    // f16. AF3's default was within 24% of failing its own gate.
    //
    // It costs 63.5 MiB at 256 tokens - 545.3 -> 608.8 peak - which is why it
    // is a PRIOR and not a new constant. A device that has the room says so.
    pairTransitionSplitMinChannels: 128,
    triangleProjectMatrixMinChannels: 128,
    // 🔴 THE BIGGEST SINGLE PASS IN AN OpenDDE TRUNK, HALVED. grid.project
    // 385.9 -> 192.0 ms there (2.01x) and 58.5 -> 36.0 on AF3 (1.63x); the
    // trunks 1877.1 -> 1685.9 and 463.9 -> 441.1. It costs no memory and the
    // differential does not move - AF3's pair 3.21e-2 -> 3.09e-2, OpenDDE's
    // 1.23e-2 -> 1.30e-2 against a 1.5e-1 bound.
    gridProjectMatrix: true,
    // 🔴 AF2's LARGEST REMAINING VECTOR KERNEL. The two q/k/v/gate projections
    // are 12.75 ms of a 72.31 ms block at 400 residues and 512 sequences; on
    // the units they are 8.33, so the block is 67.29 and the stack 3471 -> 3230
    // ms. A warm fold at that shape is 4.123 -> 3.880 s. pLDDT 57.28 -> 57.29
    // and the first alpha carbon moves 0.13 A, which is the f16 multiply the
    // matrix units do and the same order as every other kernel here that made
    // that trade.
    attentionProjectMatrix: true,
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
    // ...and the outer product mean's contraction, which is the biggest kernel
    // in an AF2 block and the deepest K in the model. See opmMatrixContract.
    opmMatrixContract: true,
    // 🔴 FOUR PAIRS AN OUTPUT WORKGROUP, AGAINST THE M2'S TWO. The kernel is
    // bound by a weight read every pair in the workgroup shares, and what
    // limits P is workgroup storage: 4 KiB a pair against 32 KiB on an M2 and
    // 48 here. Swept at 825 residues and 512 rows, interleaved, as this
    // kernel's own time - 1: 71.1 ms, 2: 40.9, **4: 26.2**, 8: 28.6. Eight
    // turns back up because 32 KiB of staged cells is one workgroup a core.
    // Bit-exact in P: fold-af2.js returns checksum -1805925 at all four.
    //
    // 🔴 AND IT STOPPED MATTERING WHEN THE OUTPUT PROJECTION WENT ON THE UNITS.
    // Re-swept at the same shape with opmMatrixContract on - and so
    // opmMatrixOutput with it, since that follows unless turned off by itself -
    // the kernel is 12.46, 12.40 and 12.47 ms at P = 2, 4 and 8. The matrix
    // output kernel has its own geometry and does not read P at all, so this
    // knob now only steers a path this device no longer takes. Left at 4 for
    // the devices that do, and for the day the matrix path is bisected out.
    opmProjectOutputPairs: 4,
  }],
  // Apple M2, 10 cores, macOS 13.2, Chrome 152 - the machine docs/PERF.md is
  // measured on, reporting {vendor: "apple", architecture: "metal-3"}.
  //
  // 🔴 ONE KNOB, AND THE REST OF THIS DEVICE'S ANSWERS REMAIN THE DEFAULTS.
  // This entry exists because "the M2's answers ARE the defaults" was true when
  // every default was measured here and stopped being true when the matrix
  // path arrived: the units are REACHABLE on this part - it reports
  // `chromium-experimental-subgroup-matrix`, f32 and f16, both 8x8x8 - and the
  // outer product mean's contraction is the one kernel whose shape suits them
  // at that size.
  ["metal-3", {
    // 🔴 THE CAPABILITY LAYER'S ANSWER IS WRONG ON 8x8 UNITS, AND A PRIOR IS
    // WHERE THAT GETS SAID. `matrixCapabilityTuning` turns this on for any
    // device announcing subgroup matrices with an f16 configuration, which is
    // true of this M2 - and AF2's q/k/v/gate projection on 8x8 units is a LOSS
    // here where it is 1.53x on the A100's 16x16. Measured on a 59-residue
    // fold, two rounds, the block stack: 1.37/1.38 s with it against 1.19/1.18
    // without, and it is the WHOLE of the capability layer's cost on this part
    // - `matrixLinear` and `stagedMatrixPrefetch` move neither the time nor the
    // checksum at this shape, so neither is named here.
    //
    // It also changes the arithmetic, which is how it was isolated: on it the
    // fold is -1876396 and pLDDT 57.249, off it -1848346 and 57.213, which is
    // this repository's answer before the capability layer existed.
    attentionProjectMatrix: false,
    // 🔴 SWEPT IN SITU WITH tools/gpu/profile-af2-block.js --sweep, WHICH
    // INTERLEAVES ITS ARMS - this machine drifts up to 3.2x between runs and a
    // sweep is exactly the shape that hides it. Block milliseconds, false
    // against true, and the output projection that follows the contraction:
    //
    //   length x rows   block off   block on   speedup   opm.project-output
    //   59  x 128           22.87      21.55     1.06x    2.017 ->  0.927
    //   128 x  64           49.50      44.45     1.11x    9.023 ->  4.171
    //   200 x 128          152.35     136.52     1.12x   22.272 -> 10.120
    //   400 x 256          766.50     665.71     1.15x   94.242 -> 40.115
    //
    // Monotone across a 34x range of block cost and it never inverts, which is
    // why it is a prior rather than a size rule. The contraction itself is
    // 1.32x at 200x128; most of the block win is the output projection at
    // 2.2-2.4x, which follows this knob unless `opmMatrixOutput` turns it off.
    //
    // 🔴 CORRECTNESS, NOT ONLY SPEED. tools/gpu/check-opm-paths.js --length=400
    // --sequences=512 --cz=128 passes here with the blocked arm at relRMS
    // exactly 0 and the f16 arm finite at the depth upstream records
    // overflowing; fold-af2.js at 200 residues holds its CA-CA gate and moves
    // mean pLDDT by 0.084.
    //
    // 🔴 AND THIS IS ONE M2. "metal-3" spans parts with very different core
    // counts, and this repository's own attentionQueriesPerLane spread - M2
    // 0.21x, M4 Pro 0.45x, GB10 1.17-1.42x - is the standing warning that the
    // badge does not predict the number. The direction here is mechanism (a
    // GEMM with the model's deepest K onto units that exist) rather than a
    // tuned constant, but re-sweep before trusting it on another Apple part.
    opmMatrixContract: true,
  }],
], );

const VENDOR_PRIORS = new Map([
  // 🔴 NOTHING FOR "apple" ON PURPOSE. Its measurements ARE the defaults above,
  // and an entry that restated them would be a second place for them to drift.
  // The one knob an Apple part does NOT want at its default is in PRIORS under
  // "metal-3", because it was measured on a part and not on a vendor.
]);

const RECORDED = new WeakMap();
const OVERRIDES = new WeakMap();
const CACHE = new WeakMap();
const UNRECOGNISED = new WeakSet();
const KEPT = new WeakMap();

/**
 * Treat this device as one no prior has ever been measured on.
 *
 * 🔴 IT IS NOT A TEST HOOK, IT IS THE ONLY WAY TO SEE WHAT MOST USERS GET.
 * PRIORS has two entries - `ampere` and `metal-3` - and every other GPU in the
 * world takes DEFAULT_TUNING, which is one M2's answers. Nothing in this
 * repository could measure what that costs, because the two machines that run
 * it both HAVE priors. This makes either of them answer as an unrecognised
 * device does, which is what a probe has to beat and what a probe has to be
 * measured against.
 *
 * `--no-prior` on any GPU tool reaches it; see tools/gpu-chrome.mjs.
 */
export function ignoreDevicePrior(device, keep = []) {
  // 🔴 THE KEPT KNOBS TAKE THEIR VALUES FROM THE PRIOR, NOT FROM THE CALLER,
  // which is the only way to sweep the object-valued ones. `--tune` splits its
  // argument on commas and `diffusionTokenTile` is `{below, atOrAbove,
  // crossover}`, so a knob like that cannot be written on a command line at
  // all. Naming it here says "restore whatever the prior says" and the value
  // never has to be spelled.
  UNRECOGNISED.add(device);
  KEPT.set(device, new Set(keep));
  CACHE.delete(device);
  return device;
}

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
/**
 * The knobs a device's own CAPABILITIES answer, for a device no prior names.
 *
 * 🔴 THE PRIORS TABLE WAS DOING TWO DIFFERENT JOBS AND ONLY ONE OF THEM IS A
 * TABLE. Measured with `--no-prior=<knob>`, which restores one knob at a time
 * from the prior, on this A100. An AF2 fold's warm repeat is 421 ms with the
 * prior and 639 without, and the 218 ms splits like this:
 *
 *   matrixLinear             95 ms      attentionGroup            18
 *   attentionMatrix          51         opmMatrixContract         12
 *   attentionProjectMatrix   39         stagedMatrixPrefetch       7
 *   linearTallTile, triangleProjectMatrix, opmProjectOutputPairs:  0
 *
 * Every significant one is the same question - "does this device have subgroup
 * matrix units, and can this kernel feed them" - which the API ANSWERS. It is
 * not an architecture secret and it never needed a table.
 *
 * 🔴 AND AF3's SIDE IS NOT LIKE THAT, WHICH IS WHY THIS STOPS HERE. The same
 * sweep on an AF3 fold: `sample-start` is 1867 ms with the prior and 4517
 * without, and the 2650 ms is `diffusionSplitK` 1792, `diffusionTokenTile`
 * 1103, `diffusionBatchedGates` 700, `diffusionNormSplit` 526, `atomRowTile`
 * 316 - tiles and split counts, every one of them a measurement about how many
 * workgroups fill this card and how many registers a kernel may hold. No
 * capability states those, and guessing them from a limit would be a table
 * again with worse provenance. They stay a prior, and they are where a runtime
 * calibration would have to go.
 *
 * So a GPU nobody has measured now gets the matrix paths and not the
 * geometries, which is most of AF2's gap and none of AF3's.
 *
 * 🔴 THE KERNELS STILL CHECK THEIR OWN FIT. `deviceMatrixConfig` returns null
 * where there is no configuration of the right element type, and each caller
 * refuses a geometry past the device's workgroup storage or invocation limit.
 * So this switch says "try", not "assume": a device advertising the feature it
 * cannot actually feed declines per kernel exactly as it did before.
 */
export function matrixCapabilityTuning(device, matrixConfigs = []) {
  if (device?.features?.has?.("chromium-experimental-subgroup-matrix") !== true) return {};
  if (device?.features?.has?.("shader-f16") !== true) return {};
  if (!matrixConfigs.some((c) => c.componentType === "f16")) return {};
  return {
    matrixLinear: true,
    attentionMatrix: true,
    attentionProjectMatrix: true,
    opmMatrixContract: true,
    stagedMatrixPrefetch: true,
  };
}

/**
 * Devices that must answer as if they had no matrix units at all.
 *
 * `--default-tuning` on any GPU tool reaches it, and it is the ONLY way to
 * measure what DEFAULT_TUNING alone is worth now that a capability layer sits
 * above it. `--no-prior` is a different question - an unrecognised device WITH
 * whatever units it has, which is what most users actually are.
 */
const CAPABILITY_REFUSED = new WeakSet();

/**
 * Whether this device may answer a knob from a MEASUREMENT rather than a table.
 *
 * 🔴 A DERIVATION IS NOT A PRIOR AND `--no-prior` MUST NOT SILENCE IT, because
 * measuring is exactly what an unrecognised device does. But `--default-tuning`
 * has to silence it, or the arm that prices DEFAULT_TUNING alone stops being
 * reproducible - it read 4525 ms before the derivations existed and 3765 after,
 * measuring something that no longer had a name.
 */
export const deviceDerivationsAllowed = (device) => !CAPABILITY_REFUSED.has(device);

/** Answer as a device with no usable matrix units. See CAPABILITY_REFUSED. */
export function ignoreDeviceCapabilities(device) {
  CAPABILITY_REFUSED.add(device);
  CACHE.delete(device);
  return device;
}

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
  const measured = PRIORS.get(architecture) ?? VENDOR_PRIORS.get(vendor) ?? {};
  const capability = CAPABILITY_REFUSED.has(device)
    ? {} : matrixCapabilityTuning(device, matrixConfigs);
  const kept = KEPT.get(device);
  const prior = !UNRECOGNISED.has(device) ? measured
    : Object.fromEntries(Object.entries(measured).filter(([key]) => kept?.has(key)));
  const profile = Object.freeze({
    vendor,
    architecture,
    software,
    matrixConfigs: Object.freeze(matrixConfigs),
    tuning: Object.freeze({
      ...DEFAULT_TUNING,
      // 🔴 CAPABILITY UNDER PRIOR, so a measured architecture always wins. The
      // capability layer is what a device NOBODY has measured gets; a prior is
      // what a device somebody has.
      ...(software ? {} : capability),
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

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
  // 🔴 THE ALIGNMENT SIZE AT WHICH THE NEAREST-CENTRE SEARCH IS WORTH A
  // DISPATCH, in bytes of a3m text - which is rows x length to within the
  // headers, and costs nothing to ask where counting the rows would cost a
  // parse. The device path is FLAT and the host path is LINEAR in depth, so
  // this is a crossover and not a preference. Measured on an M2 at 59
  // residues, the `features` phase of a fold:
  //
  //     rows    device    host
  //      128    0.060 s   0.010 s
  //      512    0.080     0.020
  //     1024    0.080     0.050
  //     2048    0.040     0.140
  //
  // - host by 50 ms at the shipped default and device by 100 ms at 2048, which
  // puts the crossover near 100,000 cells. That agrees with the other datum
  // there is: src/af2/model/monomer.js records the device path saving 640 ms of
  // 1072 at 825 residues, and 825 x 128 is 105,600 - the same side of the line.
  // Both paths return the SAME features, checksum for checksum at every size
  // measured, so this only ever chooses what it costs.
  //
  // null routes everything to the device, which is what this branch did before
  // the rule existed.
  deviceFeaturisationMinBytes: 100000,
  // 🔴 A DEVICE PACK THAT REFUSES STOPS THE FOLD, unless this says otherwise.
  // The alternative is what it used to do: fall back to packing on the host,
  // which is correct, silent, and 300 ms slower - a bug with no symptom but a
  // number. A bundle that genuinely cannot be decoded on the device (float32,
  // or a fixture built over plain arrays rather than a store) sets this once;
  // a descriptor that merely lost its sources on the way through a spread
  // should fail loudly instead. See DeviceWeightRefusal.
  allowHostWeightPacking: null,
  // 🔴 THE PAIR BIAS OWNS ITS LAYER NORM, WHERE ITS NORMALISED TENSOR HAS ONE
  // READER. The MSA row attention's bias comes from the pair while the
  // attention runs over the MSA, so `<label>.pair-normalized` is written by one
  // dispatch, read by one, and is `L * L * 128` floats - 348 MiB at 825
  // residues. Fusing the two removes the tensor and a dispatch. Off until it
  // is measured on this card; see createFusedPairBiasShader, and docs/AF2.md
  // for what it is worth.
  // 🔴 THE 64x64 LINEAR TILE, WHICH THE CHOOSER COULD NOT REACH. Measured on
  // this card under stock flags it is the fastest arm of eight - 12798 GFLOP/s
  // against the tall tile's 9599 - and the three tiles chooseLinearTile could
  // return are the three slowest. Off until it is measured IN A BLOCK, because
  // a standalone GEMM bench has named the wrong tile here before (see
  // stagedMatrixBlock, 16% off in the trunk). See LINEAR_TILE_SQUARE.
  linearSquareTile: false,
  fusedPairBias: false,
  // 🔴 THE FLASH KERNEL'S KEY CHUNK, WHICH WAS A FORMULA AND NEVER A SWEEP.
  // `max(8, floor(512 / (vectors * 2)))` gives 64 keys where the operands are
  // f16 and 32 where they are f32 - it holds the staged tile at 8 KiB either
  // way, which is a memory rule rather than a measured one. A stock browser on
  // NVIDIA has no `shader-f16` at all, so the f32 arm is the one a visitor
  // runs and the one nobody had swept. `null` keeps the formula.
  attentionKeyChunk: null,
  // One key per softmax rescale, scalar q.k reduction.
  attentionGroup: 1,
  attentionVectorScore: false,
  // One query per invocation.
  //
  // 🔴 DECLARED, MEASURED, AND NOT WIRED - which is why `attention.js` now
  // REFUSES any value but 1 rather than ignoring it. Nothing reads this knob:
  // `createAttentionRegisterFlashShader` takes `options.queriesPerLane`
  // (attention.js:866) and no call site ever filled it from tuning, so
  // `--tune=attentionQueriesPerLane=2` silently did nothing and
  // `audit-knobs.py` could only report it "moved nothing" - indistinguishable
  // from a knob that is merely inert on the workload.
  //
  // It is NOT deleted, because it has numbers: the spread recorded further down
  // this file is M2 **0.21x**, M4 Pro **0.45x**, GB10 **1.17-1.42x**, so it is a
  // real win on at least one part and a rout on another. A GB10 prior would
  // want it.
  //
  // Wiring it is a multi-site change, which is the other half of why it is not
  // wired yet: the value bakes constants into the WGSL (`group.x *
  // ${64 * queriesPerLane}u`, and `perQuery` unrolls the body) AND changes the
  // dispatch through `attentionFlashQueriesPerGroup`, while `registerKey`
  // (attention.js:1757) names neither. Connecting it without extending the key
  // is exactly the pipeline collision that killed AF3 twice - once for
  // `triangleProjectMatrix=false`, once for `singleProjectWorkgroupTarget=`.
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
  // 🔴 A fold whose largest pair tensor is 64 MiB or more releases both sets of
  // weights whatever the two above say; null means it does, `false` opts out.
  // See foldHolding in src/af3/fold.js.
  largeFoldReleasesWeights: null,
  // ...and streams its trunk weights from their codes inside the fold; `false`
  // keeps them decoded for the fold's passes. See setStreamedWeights.
  largeFoldStreamsWeights: null,
  // ...and a small fold streams its trunk's, keeping the codes between folds;
  // false keeps them decoded. See foldHolding.
  streamTrunkWeights: null,
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
  // 🔴 THE TRIANGLE'S PROJECTION TILE, null meaning src/kernels/triangle/shaders.js's
  // 32x16. Note this one goes the OTHER way from the diffusion token tile: it
  // wants a BIGGER tile, because its dispatch already has tens of thousands of
  // workgroups and occupancy is long since saturated, so what is left to win
  // is traffic. Which direction a tile wants to move is a question about
  // whether the dispatch already fills the device, not about the device.
  trianglePairProjectTile: null,
  // 🔴 THE TRIANGLE'S OUTPUT PROJECTION'S COLUMN TILE, null meaning the input
  // projection's. Its accumulator is a vec2 where the input's is a vec4, so it
  // wants twice the columns at the same register cost. AF3's pair track reads
  // it where that kernel accumulates in f32 - see compilePairTrack - and so do
  // AF2's two blocks, whose triangle always does. See
  // src/kernels/triangle/shaders.js.
  triangleProjectOutColumns: null,
  // 🔴 AF2's f32 q/k/v/gate projection's rows a lane, null meaning 4 - an M2's
  // register budget. See selectAttentionProjectKernel.
  attentionProjectRowsPerLane: null,
  // 🔴 AF2's OPM output projection and sequence contraction as vector GEMMs
  // where there are no matrix units - null meaning YES, `false` the old
  // per-pair kernels. Bit-identical either way, and on by default because what
  // they remove (each workgroup re-reading the whole weight matrix, or each
  // residue's slice L times) costs on any device: as an unrecognised GPU on an
  // A100, AF2 at 255 residues 1.60 -> 1.49 s. See
  // createOuterProductMeanVectorOutputShader / ...VectorContractShader.
  opmVectorOutput: null,
  opmVectorContract: null,
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
  // hardware. See src/kernels/attention-matrix.js - and note that a GEMM
  // benchmark at K = head_dim predicts the opposite and asks a different
  // question: a flash attention's reuse is in its loop, not in K.
  attentionMatrix: null,
  // Subgroups a workgroup and keys a tile for that kernel; null takes its
  // default. Both are fixed costs the tile amortises - see attention-matrix.js.
  attentionMatrixTile: null,
  // 🔴 THE KEY TILE'S GLOBAL READS AHEAD OF THE BARRIER, so a tile's memory
  // latency sits underneath the tail of the previous tile's compute instead of
  // in front of its own. The reads go to registers, the barrier moves between
  // the read and the write, and nothing else changes. docs/A100.md recorded
  // this as reverted on AF2 for a race; the race was in the BISECTION - see
  // src/kernels/attention-matrix.js on why the staging loop's trip count is
  // not uniform at a head of eight.
  // 🔴 HOW MANY BYTES ONE TRANSITION CHUNK MAY BIND, overriding
  // TRANSITION_CHUNK_TARGET_BYTES. That constant's 32 MiB knee was measured on a
  // 59-RESIDUE fold as a memory trade - device peak against wall - and the
  // transitions are 263 of an 825-residue block's 339 dispatches, which is a
  // different question. null takes the constant.
  transitionChunkBytes: null,
  // 🔴 HOW MANY BYTES OF CACHED PAIR ATTENTION BIAS A SCHEDULE MAY HOLD,
  // overriding PAIR_LOGITS_CACHE_BYTES. That constant's 64 MiB was measured at
  // 200 TOKENS, where it covers all twenty-four blocks; the cache is
  // `64 * tokens^2` bytes a block, so it covers six at 400 and about three at
  // 512 - the same shape as transitionChunkBytes, a cap fitted where it
  // happened to cover the whole workload. null takes the constant.
  pairLogitsCacheBytes: null,
  // 🔴 AF3's OUTER PRODUCT MEAN, whose OPM_BLOCK_I and OPM_CELL_CHUNK were both
  // measured at 59 and 150 TOKENS. The block's own note says "the workgroup
  // count still wins at these sizes", which is a claim about a SIZE - see
  // docs/AF3.md. null takes the constants.
  opmBlockI: null,
  opmCellChunk: null,
  // Above this many tokens the outer product mean takes a block of ONE. null
  // means "never" - see OPM_BLOCK_I_TOKENS for why this is a prior and not a
  // derivation.
  opmBlockITokens: null,
  attentionMatrixPrefetch: null,
  // 🔴 AND THE SAME UNITS ON AF3's `grid.attend`, WHICH IS A DIFFERENT KERNEL
  // AND A DIFFERENT KNOB. It is the largest pass in the pairformer and the only
  // CUBIC one, so it leads by more on every longer chain; OpenDDE runs the same
  // track and gets it too. Separate from `attentionMatrix` because the two
  // bodies differ - no gate, no uniform, a bias that is always present - and
  // because the geometry that suits one is 5-6% wrong for the other. See
  // src/af3/trunk/grid-attention-matrix.js.
  gridAttendMatrix: null,
  esmfold2TokenRowTile: null,
  esmcRowTile: null,
  runtimeLoopBounds: null,
  // Its geometry, "subgroupsXkeys"; null takes GRID_ATTEND_MATRIX_DEFAULT_TILE.
  gridAttendMatrixTile: null,
  // 🔴 THE PAIR TRANSITION AS THREE PASSES INSTEAD OF ONE, ON THE MATRIX UNITS.
  // Whether it pays is a CHANNEL WIDTH question and not only a device one: the
  // fused kernel holds the widened row in workgroup memory, so its row tile
  // halves each time the channels double. Measured at 200 tokens against the
  // fused kernel's own best tile - 1.13x at AF3's 128 channels, 2.77x at
  // ESMFold2's 256, 3.71x at OpenDDE's 384. It brings back the widened tensor
  // the fusion exists to avoid, chunked over rows. See
  // src/af3/trunk/transition-webgpu.js.
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
  // costs no bytes. See src/kernels/triangle/project-matrix.js.
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
  // directWeightsAllowed in src/kernels/matrix-linear.js, which is what turns
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
  // kernel's ownership rule. See src/af3/trunk/grid-project-matrix.js.
  gridProjectMatrix: null,
  // 🔴 AF2's q/k/v/gate PROJECTION ON THE UNITS. It is 20% of an evoformer
  // block at 400 residues and 512 sequences - 12.73 ms of 78.01, the largest
  // thing in the block still on the vector path - and as one packed GEMM the
  // same work prices at 3.659 ms against 6.364. It reaches the four matrices
  // through `weightIndex` rather than a repack, because that buffer is bound by
  // five shaders. See src/kernels/attention-project-matrix.js.
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
  //   singleProject...Target/MaxSplits  🔴 TAKEN AFTER ALL, AND THIS ENTRY WAS
  //                    WRONG. It said "single.project in a real trunk does not
  //                    move by 0.3 ms either way". It moves by 11.3 ms of a
  //                    104.3 ms trunk. bench-trunk.js --profile, gpuTotalMs
  //                    (every pass, not the listed rows), three rounds an arm:
  //
  //                      n=68    old 104.3 104.3 104.3   new 93.0 92.8 93.1
  //                      n=300   old 607.8 609.3 608.4   new 599.0 597.7 596.9
  //
  //                    and `single.project` itself 15.00 -> 3.69 at 68 tokens,
  //                    15.67 -> 4.99 at 300. The rounds agree to 0.3 ms.
  //
  //                    🔴 WHY BOTH EARLIER READINGS SAID NO: EACH KNOB WAS
  //                    SWEPT ALONE, AND ALONE NEITHER PAYS. The entry below
  //                    measured lanes at the DEFAULT split and found 128 WORSE
  //                    - which reproduces exactly here, 15.48 -> 16.24 - and
  //                    this entry measured the split with maxSplits 6, whose
  //                    perSplit is 64 and which therefore cannot use a wider
  //                    lane at all. The two compose and nothing else does:
  //                    split three ways the output is 128 wide, one lane an
  //                    output, and 7.49 becomes 3.69. A sweep of either axis on
  //                    its own says the knob does not pay.
  //
  //                    Which leaves one number unexplained: the split ALONE is
  //                    7.5 ms of 104.3 here and was read as 0.3 there. Recorded
  //                    as a disagreement rather than resolved - the metric
  //                    above is named exactly so the next person can repeat it.
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
    // before trusting it on another part; see src/kernels/attention-matrix.js.
    attentionMatrix: true,
    attentionMatrixTile: "4x32",
    // 🔴 THE TRANSITION CHUNK, RE-SWEPT AT 825 RESIDUES. TRANSITION_CHUNK_TARGET_BYTES
    // is 32 MiB and its knee was measured on a 59-residue fold as a MEMORY
    // trade. At 825 with 512 rows the transitions are 263 of a block's 339
    // dispatches, and the same sweep as time is monotone: 16 MiB 202.95 ms,
    // 32 192.12, 64 188.56, 128 184.24, **256 181.71**, then 512 181.18, 1024
    // 180.51 and 2047 180.04 - so 256 takes 5.4% of the block and eight times
    // the memory past it takes 0.9% more. A fold at 825/512/1024: warm
    // 11195 -> 10742 ms for 6803 -> 6971 MiB, and at 59 residues 414 -> 398 for
    // 540 -> 575. **Bit-identical at both** (-121844157 and -329598), because
    // chunking splits rows and reorders no sum. Ampere only: the cost is 168 MiB
    // and a laptop keeps the 32 MiB constant.
    transitionChunkBytes: 256 * 1024 * 1024,
    // 🔴 THE PAIR-LOGITS CACHE, RE-FITTED AT 400 TOKENS. PAIR_LOGITS_CACHE_BYTES
    // is 64 MiB and was measured at 200 tokens, where it covers all twenty-four
    // blocks; the cache is `64 * tokens^2` a block, so it covers six at 400.
    // A denoiser call at 400 tokens, median of eight steady calls, two rounds:
    // 64 MiB 51.5 and 51.5 ms, 128 MiB 50.0 and 50.5, **256 MiB 47.0 and 47.5**,
    // 512 MiB 47.0 and 47.0 - 256 caches all twenty-four at this length and 512
    // is the same arm. **8.7% of a denoiser call for 166 MiB.** Ampere only,
    // like the two above it.
    pairLogitsCacheBytes: 256 * 1024 * 1024,
    // See OPM_BLOCK_I_TOKENS: measured on THIS card, so only this card takes it.
    opmBlockITokens: 256,
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
    // `tri.project-out` at 32 x 64, stock flags, 255 tokens: 4.34 -> 3.26 ms at
    // 384 channels and 0.56 -> 0.44 at 128, relRMS 0 (bench-triangle-project.js
    // --arms=32x32@32x32,32x64@32x32).
    triangleProjectOutColumns: 64,
    // 🔴 THE LINEARS' ROW TILE OF TWO IS THIS CARD'S, NOT A DEFAULT. A smaller
    // tile re-reads the weights once per row tile, which a 40 MB L2 at 1.5
    // TB/s absorbs (ESMFold2's sampler 618 -> 437 ms at 255 residues here) and
    // a T4's 4 MB at 320 GB/s does not: there the same change cost its
    // language model 211 -> 290 ms and its sampler ~60 ms at 255. Elsewhere
    // the tile stays the shared linear's eight.
    esmfold2TokenRowTile: 2,
    esmcRowTile: 2,
    // AF2's f32 attention projection at 8 rows a lane, stock flags, 128 x 255:
    // 1.912 -> 1.375 ms, bitwise identical (bench-attention-project.js).
    attentionProjectRowsPerLane: 8,
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
    // 🔴 THE SINGLE PROJECTION WAS RUNNING AN M2's CONSTANT, AND IT IS 4x.
    // singleProjectSplits' own comment says "110 is an M2's number, and the
    // shape of the rule is not... an A100 wants about 1200", gives a
    // standalone bench table showing 1.46x-2.50x, and leaves both as
    // parameters for a device to set. No device ever set them. Measured in the
    // TRUNK, which is where the answer counts, at 2 passes and msa 128, two
    // rounds agreeing to 0.1 ms:
    //
    //     n     single.project        trunk GPU total
    //     68    15.00 -> 3.70  4.05x  104.3 -> 93.0   -10.8%
    //     150   15.78 -> 3.98  3.97x  196.1 -> 184.3   -6.0%
    //     300   15.50 -> 4.98  3.11x  608.4 -> 598.6   -1.6%
    //     512   11.47 -> 5.16  2.22x  1374  -> 1367    -0.5%
    //
    // The win is largest at the SHORT chains a page actually folds, because
    // the starvation is: `project` runs one workgroup a token and a split, so
    // 68 tokens is 136 workgroups on a card that holds 4542.
    //
    // 🔴 AND THE TWO KNOBS ONLY WORK TOGETHER. The target alone is 7.87 ms at
    // n=300 and the lanes alone are 16.24 - WORSE than the 15.50 default -
    // because 128 lanes over the unsplit 384-wide output leaves each lane
    // three outputs deep and adds nothing. Split three ways the output is 128
    // wide, one lane an output, and the two compose: 4.98. A sweep of either
    // on its own says the knob does not pay.
    //
    // 🔴 AND IT IS A REORDERING, NOT A REPACKING. Three workgroups normalise
    // the same row where one did, and 128 lanes reduce it in a different tree,
    // so `check-af3-block-any` moves its single residual 2.0899e-4 -> 2.0899e-4
    // in the eighth figure and the pair not at all. On a fold: max |dx| 0.001 A
    // with 531 of 574 atoms identical - the same class as `attnSplits` 1 -> 4,
    // which this prior already ships at 541 of 574.
    //
    // 🔴 ampere ONLY. The table in singleProjectSplits shows 6 splits LOSING on
    // an M2 at every n it was measured at, which is the whole reason these were
    // left as parameters. DEFAULTS_ARE_MEASUREMENTS.
    // 🔴 THE TARGET AND THE SPLIT CEILING ARE A PAIR, AND ONLY ONE WAS EVER SET.
    // This prior carried `singleProjectWorkgroupTarget: 2048` and left
    // `singleProjectMaxSplits` at its module default of 3, so
    // `singleProjectSplits`' candidate list `[1,2,3,6].filter(<= 3)` could never
    // reach the 6 the comment beside it says an A100 wants - and at 2048 the
    // "reaches the target" loop never fires at a realistic token count either,
    // so every fold fell through to "take the most workgroups available" and
    // got 3, at every length.
    //
    // Re-measured on bench-single-project.js at 31 rounds and 64 iterations,
    // reproducible across two reps (width 384, heads 16 x 24):
    //
    //     n      splits 1   2        3        6
    //     59     0.1422   0.1422   0.0906   0.0578   <- 6, and 3 was shipping
    //     128    0.1422   0.1594   0.0922   0.0594   <- 6
    //     200    0.1563   0.1578   0.1078   0.0734   <- 6
    //     400    0.1578   0.1578   0.1078   0.1578   <- 3
    //     512    0.1578   0.1594   0.1422   0.2094   <- 3
    //
    // 1200 with a ceiling of 6 picks the best arm at all five; 2048 WITH 6 would
    // pick 6 at 400 and 512 and be 46% worse there, which is why the target
    // moves with the ceiling rather than the ceiling alone.
    // 🔴 AND THE BENCH'S DEFAULT 11 ROUNDS OF 16 CANNOT SEE THIS: every arm
    // reads 0.15-0.22 there and 3 looks like the winner everywhere, which is a
    // false negative that nearly kept the bug.
    // 🔴 AND IT IS 4.1% OF THE TRUNK'S GPU TIME, NOT A FOLD'S. Profiled at 68
    // tokens: `single.project` is 3.79 ms of 93.1, so 1.57x on it is ~2.3% of
    // the trunk and the wall clock does not move (68 tokens: 190/191 ms before,
    // 194/184 after). Taken because the prior should express its own
    // measurement, not because a fold gets faster.
    singleProjectWorkgroupTarget: 1200,
    singleProjectMaxSplits: 6,
    singleProjectLanes: 128,
    // ...and the outer product mean's contraction, which is the biggest kernel
    // in an AF2 block and the deepest K in the model. See opmMatrixContract.
    opmMatrixContract: true,
    // 🔴 THE OUTER PRODUCT MEAN'S WORKING SET, RAISED FROM THE SHIPPED 64 MiB.
    // It is a working set and not a limit - the path runs at every length
    // whatever this says, and the number only decides how many blocks it takes
    // - so a card with room should hold more pairs at once and dispatch fewer
    // times. Swept in situ at 825 residues and 512 sequences, which is where
    // the blocking actually bites, block milliseconds and `opm.contract`:
    //
    //    32 MiB  211.20  24.809      256 MiB  195.43  17.972
    //    64      199.81  19.436      512      194.01  17.527
    //   128      198.25  18.571     1024      193.53  17.196
    //
    // 256 is the knee: 64 -> 256 is 2.2% of a whole block and 1.08x on the
    // contraction, and 256 -> 1024 buys 1.9 ms more for four times the memory.
    // 🔴 AND IT REORDERS NO SUM, so this is free of any accuracy question:
    // check-opm-paths.js holds the blocked arm to relRMS EXACTLY 0 because a
    // pair's contraction is untouched and only where it lands moves.
    opmPairBlockBytes: 256 * 1024 * 1024,
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
  // Tesla T4 (Turing, 40 SMs, 48 KiB of workgroup storage), which is what
  // Colab hands out - so this is the part most people who fold from a notebook
  // are on. Measured there; see docs/PERF.md.
  //
  // 🔴 ONE KNOB, AND THE REST ARE THE DEFAULTS ON PURPOSE. Ampere's entry is
  // the neighbour this architecture would inherit from if priors were keyed by
  // vendor, and it is a **40% REGRESSION** here - `bench-triangle` at L=300
  // goes 127.0 ms to 173.4 - with each of its triangle knobs losing on its own
  // too. Turing has 64 FP32 lanes per SM against Ampere's 128 and half the
  // workgroup storage, so its shapes do not transfer, and `16x4` on the kernel
  // below will not even build here (65536 bytes against a 49152 limit).
  //
  // 🔴 WHAT DOES TRANSFER IS NOTHING; WHAT WAS MEASURED IS THIS. The outer
  // product mean is the trunk's second-largest kernel, and this part wants
  // FEWER, FATTER workgroups than any device measured before it -
  // `bench-opm.js --rows=1024`, ms, the shipped block of two against eight:
  //
  //     tokens      150     256     300     400
  //     blockI 1   67.2   231.3   341.7   649.8
  //     blockI 2   52.4   164.5   208.2   386.4   <- ships
  //     blockI 4   55.5   111.4   181.1   310.7
  //     blockI 8   25.2    81.4    95.6   173.8   <- 2.0-2.2x, every size
  //
  // It is not a band and it is not a crossover: eight wins from 150 tokens to
  // 400 and the ordering is monotone in the block. `blockJ` stays 4 - 8x2 is
  // 280.9 ms and 8x1 is 440.1 against 8x4's 206.6 at 400 tokens - and the cell
  // chunk stays at its default, where shrinking it is catastrophic (8x4@8 is
  // 3372.9 ms, 8x4@2 is 13366.7). Every arm agrees numerically (relRms 0).
  //
  // 🔴 AND `opmBlockITokens` IS DELIBERATELY ABSENT. Ampere's "above 256
  // tokens take a block of ONE" is that card's memory system turning over;
  // here block ONE is the WORST arm at every size measured, so the threshold
  // would be exactly backwards.
  // 🔴 AND THE GRID ATTENTION WANTS THE MATRIX PATH, WHICH IS 8-12% OF THE
  // WHOLE TRUNK. `grid.attend` is the largest kernel in an AF3 pairformer
  // (15.4% at 300 tokens and 512 rows) and this part has the units for it -
  // the adapter reports `shader-f16` AND
  // `chromium-experimental-subgroup-matrix`. Measured on the trunk with the
  // arms ALTERNATED against a baseline, because this card throttles (below):
  //
  //     round 1   base 5383.3   matrix 5105.5   base 5713.3    -8.0%
  //     round 2   base 6173.3   matrix 5633.6   base 6591.7   -11.7%
  //
  // `steady.pairformer` agrees independently, -7.3% and -13.7%. The tile is
  // ampere's 4x32 and was NOT swept here - it is the value that was measured,
  // not the value that was chosen.
  //
  // 🔴 `pairTransitionSplit` IS NOT TAKEN, measured in the same two rounds at
  // -1.2% and -1.3%, which is this box's noise.
  //
  // 🔴 AND A T4 IN COLAB THROTTLES HARD, WHICH IS WHY EVERY ARM HERE IS
  // MEASURED NEXT TO A BASELINE. Under sustained load it sat at 81 C, **585
  // MHz against a 1590 MHz boost clock**, 71.3 W against a 70 W limit - and
  // by the end of an eight-run sweep it was at 360 MHz. The baseline rose
  // 5383 -> 6592 (+22%) across four minutes of that. A sweep that compares an
  // arm with a baseline taken ten minutes earlier is measuring the
  // temperature; nothing in this entry was taken that way.
  //
  // 🔴 AND THE THREE KNOBS THE MEASURED WIDTH DERIVES, PINNED, BECAUSE ON A
  // COLAB T4 THAT MEASUREMENT IS A COIN TOSS. Logged at the diffusion
  // transformer's compile across six identical runs of bench-head at 255
  // tokens, the width read 2048, 512, 2048, 8, 4, 512 - the card idles at 585
  // MHz and boosts under load, and the probe times six dispatches while the
  // fold's own compiles and uploads share the device - and the denoiser call
  // came out 132, 192, 132, 158, 158, 198 ms. Two of the three derived knobs
  // moved it:
  //
  //   diffusionSplitK  width 2048 -> 8 splits (fast); 512 -> no split, qkvg
  //                    30 ms against 8, ffw-wide 30 against 10
  //   atomRowTile      bench-head, ms of atom stack a call, two rounds:
  //                        tile      1      2      4      8
  //                        255 tok  16.0   19.5   47.0   22.4
  //                        510 tok  30.7   38.4   93.1   42.4
  //
  // A CORRECT width would not help either: a T4's ~640 workgroups say "no
  // split" at 255 tokens, and that is the 192 ms arm - these GEMMs stream their
  // weights and want the extra workgroups for latency, not to fill the SMs.
  ["turing", {
    opmBlockI: 8,
    gridAttendMatrix: true,
    gridAttendMatrixTile: "4x32",
    // Split rules at tile 4, bench-head GPU ms a denoiser call, two rounds
    // each, 255 / 68 tokens: splits 16 171/140 and 34/36, splits 8 166/135 and
    // 27/31, splits 4 131/170 and 27/30; tile 2 at splits 8 is 187 and 39. The
    // K-split and tile carry it (qkvg 8 ms against 30); the out/attn splits
    // and the whole-call totals are inside this card's run-to-run spread.
    diffusionSplitK: { splits: 8, tile: 4, crossover: 512, outSplits: 4,
                       attnSplits: 4, attnTile: 2, normSplits: 4 },
    atomRowTile: { below: 1, atOrAbove: 1, crossover: 1 << 30 },
    // ...and the third, which moves nothing measurable (tiles 1/2/4, with the
    // two above pinned: 126.6/128.6/131.1 ms at 255 tokens, 28-30 at 68) - so
    // pinned for the determinism alone, at the arm that was never worst.
    diffusionTokenTile: { below: 1, atOrAbove: 1, crossover: 1 << 30 },
    // 🔴 A FRESH COLAB VM'S FIRST FOLD IS MOSTLY THE DRIVER COMPILING, AND
    // CONSTANT LOOP BOUNDS ARE WHAT IT SPENDS THAT ON: NVIDIA's compiler
    // unrolls them. "tiered" compiles every kernel with opaque bounds first and
    // the unrolled one behind it (see withRuntimeLoopBounds in
    // pipeline-cache.js). AF3 on this T4, driver cache cleared, s:
    //
    //                          first fold      folds 2..5
    //   68 residues   off      14.0 / 13.7     1.2 each
    //                 tiered    8.6 / 11.2     2.4, 2.4, 2.0, 2.0 (-> 1.4)
    //   255 residues  off      22.5            7.6, 7.5
    //                 tiered   18.2            7.9, 7.8
    //
    // Bit-identical (the arithmetic is untouched). A user who folds once saves
    // 3-5 s; one who folds five times comes out even.
    runtimeLoopBounds: "tiered",
    // 🔴 AND AF2's FLASH ATTENTION OFF THE MATRIX UNITS, FOR THE SAME REASON:
    // its three variants (heads of 8, 16 and 32) are ~1.25 s of driver compile
    // EACH on this card, 3.8 of AF2's 9.7 s of compile, for a warm kernel that
    // is level at 59 residues and ~9% faster at 255. Driver cache cleared,
    // fold-af2 at 59: first run 11.7 / 8.4 s -> 4.4 / 5.5, repeats 0.44-0.49 s
    // either way; at 255 repeats 4.2 -> 4.6 s. Turning off EVERY AF2 matrix
    // knob saves 1-2 s more cold and costs 60-70% warm, so only this one goes.
    attentionMatrix: false,
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
    // 🔴 AND THE TRANSITION'S PROJECTIONS FOR THE SAME REASON. The capability
    // layer turns `matrixLinear` on for any device announcing an f16 matrix
    // configuration, and src/kernels/transition.js then DERIVES the block
    // from this part's 8x8x8 tile rather than assuming 16x16x16 - so it runs,
    // correctly, and slower. It is the whole of AF2's regression on this M2:
    // warm folds 1198-1226 ms with it against 1124-1133 without, which is what
    // this repository folded before the capability layer existed, at the same
    // checksum -1848346.
    //
    // It took seven wrong hypotheses to find because the knob had no OFF: the
    // gate tested only null and undefined, so every arm measured with
    // `matrixLinear=false` was measured with it ON. Fixed at that gate too.
    matrixLinear: false,
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

// NVIDIA L4 (Ada, "lovelace"), a Colab Pro runtime with 12 vCPUs and the two
// developer flags, 2026-09-26. Nothing measured it before, so it took
// DEFAULT_TUNING. Two rounds interleaved, driver cache cleared, seconds:
//
//                          default   ampere prior   ampere + the two below
//   AF3 68, first fold     3.7       4.0-4.2        2.56
//   AF3 68, folds 2-8      0.61      0.59-0.65      1.06 -> 0.67 as upgrades land
//   AF3 255, first fold    6.0       6.1-6.3        5.35
//   AF3 255, later folds   3.3-3.4   2.9-3.0        3.97, 3.75, then 2.97-3.0
//   AF2, whole run         3.7-3.9   3.85-3.9       2.43-2.47 (repeats +5%)
//   ESMFold2, repeats      0.80      0.62-0.64      0.62-0.64
//
// So the ampere prior's warm settings, plus the T4's tiered loop bounds (the
// upgrade queue runs a quarter of the CPU's threads wide, three here) and
// AF2's flash attention off the matrix units, which halves AF2's cold run as
// it did on the T4.
PRIORS.set("lovelace", {
  ...PRIORS.get("ampere"),
  runtimeLoopBounds: "tiered",
  attentionMatrix: false,
});

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
/**
 * Vendors whose measurements ARE `DEFAULT_TUNING`, and which therefore have
 * nothing to derive.
 *
 * 🔴 THE DERIVATION LAYER READS SILENCE AS "NOBODY MEASURED THIS DEVICE", AND
 * FOR EXACTLY ONE VENDOR THAT IS BACKWARDS. See VENDOR_PRIORS: there is no
 * `apple` entry ON PURPOSE, because this repository was tuned on an M2 and its
 * answers are the defaults themselves - so a knob metal-3 does not name is not
 * an unanswered question, it is an answered one whose answer lives upstairs.
 * Deriving over it replaces a measurement with an estimate.
 *
 * Priced on that M2, AF3 at 68 tokens and 200 steps, warm fold and peak:
 * 4.01 s and 476 MiB with the derivations suppressed, 5.60 s and 978 MiB with
 * them - 1.40x slower for 2.05x the memory, and the resident weights alone go
 * 13.9 MiB to 874. The same mechanism is 6032 -> 3365 ms on an A100, where the
 * silence it reads is real.
 *
 * 🔴 AND `--no-prior` MUST STILL DERIVE. That switch asks what an UNRECOGNISED
 * device gets, and measuring is precisely what such a device does - so this
 * yields to it rather than compounding with it.
 */
const DEFAULTS_ARE_MEASUREMENTS = new Set(["apple"]);

const measurementsAreDefaults = (device) => {
  if (UNRECOGNISED.has(device)) return false;
  const { vendor = "" } = RECORDED.get(device) ?? {};
  return DEFAULTS_ARE_MEASUREMENTS.has(vendor);
};

export const deviceDerivationsAllowed = (device) =>
  !CAPABILITY_REFUSED.has(device) && !measurementsAreDefaults(device);

/** Answer as a device with no usable matrix units. See CAPABILITY_REFUSED. */
export function ignoreDeviceCapabilities(device) {
  CAPABILITY_REFUSED.add(device);
  CACHE.delete(device);
  return device;
}

/** Which architecture's prior a device answers with instead of its own. */
const PRIOR_AS = new WeakMap();

/**
 * Answer with another architecture's prior - a GPU nothing has measured, asked
 * whether the nearest measured one's settings suit it. `--prior=` on any GPU
 * tool reaches it; see tools/gpu-chrome.mjs.
 */
export function useDevicePrior(device, architecture) {
  if (!PRIORS.has(architecture)) {
    throw new Error(`no prior named ${architecture}; known: ${[...PRIORS.keys()].join(", ")}`);
  }
  PRIOR_AS.set(device, architecture);
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
  const measured = PRIORS.get(PRIOR_AS.get(device) ?? architecture) ?? VENDOR_PRIORS.get(vendor) ?? {};
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
  // 🔴 WIRED NOW, AND THIS IS WHERE THE REFUSAL USED TO BE. The knob was
  // declared with numbers from three other devices (M2 0.21x, M4 Pro 0.45x,
  // GB10 1.17-1.42x) and read by nothing, so `deviceProfile` threw on any value
  // but 1 rather than letting it silently change nothing - the right answer for
  // a dead knob, and the note is kept because the SHAPE of that mistake
  // recurred three times in one session afterwards.
  //
  // What it needed was the pipeline KEY and the DISPATCH, not just a
  // destructure: `selectAttentionFlashKernel` now reads it, puts `-q<n>` in the
  // register key and returns `queryTile: 64 * n`, which is the grid both block
  // files divide by. The shader has carried `options.queriesPerLane` and
  // `attentionFlashQueriesPerGroup` all along.
  CACHE.set(device, profile);
  return profile;
}

/**
 * A knob's value where `false` means "unset", for a knob that takes a SHAPE.
 *
 * 🔴 `?? undefined` DOES NOT TREAT `false` AS OFF, AND THAT HAS BITTEN TWICE.
 * `matrixLinear: false` fell straight through into the matrix path, so every
 * arm ever measured with that knob off was measured ON; and
 * `trianglePairProjectTile: false` reaches the tile resolver as a boolean,
 * where `false.rows` is undefined and AF2 dies with "projectTile
 * undefinedxundefined is not a multiple of the 8x8 workgroup" - an error that
 * names the symptom and not the cause. Both were found by
 * tools/audit-knobs.py, which could not express either knob until --tune-json
 * existed.
 *
 * A knob whose value is a tile, an object or a count has no meaningful `false`,
 * so this reads it as "nobody set one". A BOOLEAN knob is the opposite -
 * `attentionMatrix: false` means do not use the matrix kernel - and must not go
 * through here.
 */
export const shapedKnob = (value) =>
  (value === false || value === null ? undefined : value);

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

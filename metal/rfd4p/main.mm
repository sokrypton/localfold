// RFD4-Proteina on Metal: a job file in, a design out, in one process with no JavaScript.
//
//   bash metal/build.sh rfd4p
//   ./metal/rfd4p/localfold-rfd4p --job=<spec.json> --bundle=model-rfd4p-int8 [--out=design.pdb|.cif]
//       [--designs=N] [--seed=N] [--f32]
//
// The spec is read and featurised by `cuda/rfd4p/featurise/featurise.cpp` LINKED IN (the same object cuda/rfd4p
// links, and byte for byte what `rfd4p-featurise` writes - measured: the whole input directory identical and all
// 44 tensors of the generation export exact against `cuda/rfd4p/export_input.mjs`, the replayed Box-Muller
// stream among them). Then the trunk, the autoencoder and the flow sampler run on this Mac's GPU, and the design
// is written - mmCIF where the name ends `.cif`, which is the format RFD4-Proteina itself writes.
//
// SPEED, on an M5 (8-core GPU), a whole job including featurisation: a 54-residue design at 60 steps is
// **3.7 s**, a 67-token motif 3.5 s, a 165-token small molecule 17.6 s, a 183-token interface 21.0 s.
// 🔴 AND THIS ROUND IS **10.5% AT 165 TOKENS AND 12.9% AT 400**, end to end, against its own starting commit
// (`4b92ef8c`, built in a worktree and interleaved against the current binary): four pairs at 165 giving -9.5,
// -10.5, -10.1, -11.8%, and eight at 400 giving -15.6, -12.7, -12.8, -14.0, -12.5, -12.1, -11.9, -12.3%. The
// longer input gains MORE, because everything this round removed scales with the pair track - the attention's
// bias, the four gate planes, the normed pair copy - which is also why the 400-token case needed no separate
// work. Measured in short runs (4 steps) so the laptop's 19% thermal drift over a long pair cannot swamp it.
// 🔴 ONE SAMPLER STEP AT 165 TOKENS IS **108.7 ms** AND EVERY ITEM LEFT IN IT IS AT A MEASURED CEILING. This
// round took three: metal/core's flash attention for this port's own scalar kernel (4.4%), the triangle's gate
// fused into its projection through `gemmTriGate` (5.2%), and the pair bias's norm folded into its weight
// (0.9%). What remains, with what each is against: the fused triangle gate 25.0% (the matrix units' 12.22
// TFLOP/s, which the plain projection already reached at 97%), the gated add 15.7% (106% of
// `GFLOP / 12.22 + MB / 131`, so at the roofline once its 1.17 GB epilogue is paid), the pair transition 10.1%
// (10.2 TFLOP/s, and its tile is swept too: six matrix arms interleaved on its exact shape give 1.411 ms for
// 64 x 128 against a 1.418 default and 1.482 / 1.606 / 1.810 / 1.817 for the rest, so the default is the best
// there as well, and the trunk's 1.573 ms a call is the 0.16 ms its gated-residual epilogue costs - three
// extra pair-sized planes whose 62.7 MB would be 0.48 ms unoverlapped, so two thirds of it is already hidden),
// and then nothing above 3.6% - the centre norm, the two contractions, the pair write, the
// bias projection, the triangle's doubled input norm. The two things that would move it further are named
// above and below, and neither is an opportunity: the norm-emitting epilogue is ALREADY firing on every pair
// update, and a second triangle shape is what the 400-token scaling says is not needed.
// 🔴 HALF IS THE DEFAULT AND `--f32` IS THE CONTROL, which is 3.6x slower: 12.6 s for that first design against
// 3.5. One sampler step at 183 tokens went **681 ms of GPU time to 151, and 618 dispatches to 419**, over this
// port's optimisation round: the weights, the activations and the PAIR TRACK in half; the triangle's projections
// writing channel-major so nothing transposes; its gate, its residual and the NEXT stage's LayerNorm all folded
// into the GEMM that writes the track; the four projections batched into one; every layer's adaLN conditioning
// precomputed in a single GEMM, since none of it depends on the activations; and a pair row's index taken from
// its threadgroup rather than from a software integer division a thread.
// 🔴 AND THE ATTENTION IS metal/core's FLASH KERNEL NOW, ON THE MATRIX UNITS. This port had its own scalar one
// - a simdgroup a query, a Kahan dot product a key - which is a correct softmax and nowhere near the machine:
// at 165 tokens with 12 heads of 64, through `metal/bench attn` with its arms interleaved, core's is **0.108 ms
// against 0.571**. The packing is free, because core wants q | k | v | gate at 0, W, 2W, 3W of each position's
// row with W = heads * dimHead, and W IS the token width here - so `to_qkv`'s three contiguous output blocks
// already ARE q | k | v, and all it takes is an output row stride of 4 D with the gate's own product written at
// 3 D. The q and k norms then run in place.
// 🔴 THE BIAS IS WHERE IT NEARLY WENT WRONG, AND THE FIRST ANSWER WAS WORSE THAN THE PROBLEM. core reads the
// pair bias head-major, [H][n][n], where the projection writes it pair-major with the head innermost; the
// obvious transpose kernel reads `bias[(q n + k) H + h]`, a gather at a stride of twelve floats, and it cost
// **22.6 ms a step against the 8.0 ms of attention it was there to save** - the step went 124 ms to 140. The
// projection can simply be written the other way round instead, swapping which operand is the weight
// (`Y[rows][out] = X[rows][in] W[in][out]` does not care), which gives [H][n n] contiguous for +2.2 ms on that
// GEMM and carries the log2(e) core's `scale * log2(e)` expects in `alpha` for nothing.
// Measured at 165 tokens: one step **124.4 ms of GPU time to 115.2**, and a whole 60-step interface design
// 15.9 s to 15.2 - four paired rounds, ABAB, flash ahead in all four by 0.50 to 0.82 s, which the arms have to
// be interleaved to see at all because this laptop's clock drifts 9% upward over the eight runs. It costs
// 0.009 A RMS on the design (worst 0.071, every atom matched), and `--f32` keeps the scalar kernel.
// 🔴 AND WHAT IS LEFT IS AT THE ROOFLINE, WHICH IS THE ANSWER TO "WHAT NEXT" AND NOT AN EXCUSE. Measured on
// this M5: the matrix units reach **12.22 TFLOP/s** on the projection's own shape (`metal/bench 1024 27225 384
// 0,1000000` - and 3.03 without them, so a kernel off the matrix units is the thing to look for), and a copy
// reaches **131 GB/s** (`metal/bench bw 256`). Against those two numbers, the four kernels that are 54% of a
// step at 165 tokens:
//
//   kernel                  ms    GFLOP     MB   floor   of roofline
//   tri-project          25.32    299.7      0   24.53      97%   (now fused with the gate; see below)
//   gated add            17.00    112.4    878   15.90      94%
//   pair-transition-out  10.78    112.4    439   12.55     116%
//   tri_ab_hm             8.83      0.0   1171    8.94     101%
//
// (floor = GFLOP / 12.22 + MB / 131, the arithmetic and the bytes each at their own ceiling and not overlapped,
// which is why one row reads over 100%.) `tri_ab_hm` moves its 1171 MB at **133 GB/s**, i.e. at the copy
// ceiling: it is a gate and a transpose of four pair-sized planes and there is nothing in it to speed up.
// 🔴 AND THE BATCHED PROJECTION - THE BIGGEST ITEM, 22% - IS THE FASTEST FORM OF THAT PRODUCT ON THIS
// MACHINE, which is worth setting out at length because "22% of a step" invites one more attempt at it. Every
// lever, measured:
//   - THE KERNEL: the matrix units' GEMM is already what runs. The vector one is **3.03 TFLOP/s** against
//     **11.7** on this exact shape, so the choice is not in doubt.
//   - THE TILE: six matrix tiles interleaved in one run - 1.834 ms for the default, 1.829 for 64x128 (which is
//     what it picks), 1.830 for 32x256, and 1.858, 2.233, 2.305 for the rest. Nothing beats the default.
//   - THE TRUNK AGAINST THE BENCH: 25.32 ms over 14 calls is **1.81 ms a call** against the best isolated
//     arm's **1.829**. There is no gap between this port's use of the GEMM and the GEMM.
//   - THE ORIENTATION: both measured equal (167.1 ms a step against 166.6), recorded on the projection itself.
//   - THE OPERAND REUSE: doubling M from 1024 to 2048 at the same K gives **11.77 TFLOP/s against 11.73** -
//     the same. So the complaint that 64-row tiles re-read the whole 25.7 MB of `zn` sixteen times costs
//     nothing measurable, and a formulation with a larger M would buy nothing.
//   - FUSING THE TWO DIRECTIONS into one GEMM of 8T: IMPOSSIBLE, and for a reason worth writing down rather
//     than measuring. They look like two affine functions of one normalised pair track, which would fold each
//     norm's scale into its weight and share the standardisation - but the two triangles are SEQUENTIAL and
//     the second reads what the first wrote. There is no shared input. The norm that genuinely could be shared
//     already is: the GEMM that writes the pair track emits the next stage's norm in its epilogue (`znPre`).
//   - WHAT IS LEFT IS THE CHECKPOINT'S OWN WIDTH: at K = 384 the kernel reaches ~11.7 TFLOP/s where K >= 768
//     reaches ~12.2 (12.21 at 1536, 12.10 at 4096). That 4% is a property of a 384-channel pair track, not of
//     this port, and P is not ours to change.
// 🔴 AND ONE MORE PASS IS GONE, WORTH 0.9% AND HONESTLY LESS THAN HOPED. The attention's pair bias normed the
// whole pair track into a buffer and projected that; RMSNorm is a per-row SCALE, so
// `sum_c (z[r][c] / rms[r] * w[c]) W[h][c]` is `(1 / rms[r]) sum_c z[r][c] (w[c] W[h][c])` exactly - the
// channel scale folds into the weight once at load (`weightTimesNorm`) and the row scale comes out of the
// product after (`rfd4p_bias_scale`). That removes a read AND a write of `n*n x P` halves (20.9 MB each at
// 165 tokens, 123 MB at 400) in exchange for one float a row, and removes the `tl.znh` buffer with them.
// Measured ABAB at 165 tokens over three rounds: 13.04 / 13.45 / 13.82 s against 13.22 / 13.55 / 13.90, ahead
// in all three but by only 0.08 to 0.18 - a mean 0.12 of 13.5, **0.9%**, against a roofline estimate of 1.8%,
// so half of that write was already overlapped.
// 🔴 AND AT 400 TOKENS IT CANNOT BE MEASURED ON THIS MACHINE, which is worth saying rather than rounding up.
// Eight paired rounds of a 43 s job gave deltas +1.41, +2.05, +0.23, -0.56, -0.75, -0.85, -0.81, -2.01: a mean
// of -0.34 s inside a spread of +/-2, because the laptop drifts **19%** over those sixteen runs (the control
// alone goes 38.65 to 46.57 s). The ordering puts the fused arm second and so on the hotter machine, which
// biases against it, and the mean still comes out negative - but a 1% effect under a 19% drift is not a
// measurement. It is kept for the 165-token number, the exactness of the identity, and the buffer.
//
// 🔴 AND THE LAST PASS I THOUGHT WAS LEFT DOES NOT EXIST: THE FUSION IS ALREADY THERE. I read 14 `layernorm`
// dispatches a step against 7 pair updates and concluded the triangle's input norm ran twice a layer, with
// core's `lnEmit` never firing for the gated add. Both halves of that were wrong.
//   * `lnEmit` DOES fire. The `e` number in a kernel's name is not the epilogue field - it is a packed
//     instance key, `tri|1 plain|2 hacc|8 triReg|16 swigluReg|32 lnEmit|64 vecEp|128` - so the gated add's
//     `e192` reads as **vecEp | lnEmit**, which is the opposite of what I took it for. Instrumented: both
//     handoffs return true on all 147 pair updates of a run (`haveZnIn=1 haveTn=1`), so the first triangle's
//     gated add emits the second's norm AND the second's emits the pair transition's. The fallback
//     `layerNorm` calls in `triangle` and in `pairUpdate` never run at all.
//   * THE 14 CALLS ARE THE TOKEN TRACK'S, one per token layer over [n][D] - there are 14 layers - and each is
//     necessary. Their 2.58 ms is 0.18 ms a call for 0.5 MB of traffic, which is command-buffer overhead on a
//     165-row dispatch under a profiling mode that gives every dispatch its own buffer, not work.
// 🔴 AND THE MEASUREMENT I QUOTED FOR IT WAS VACUOUS, which is the part worth keeping. The probe skipped
// `triangle`'s fallback norm - the branch that never executes - so "all fourteen skipped" skipped nothing, and
// the -0.07 / -0.16 / -0.15 / +0.27 s it produced was noise, as its own sign disagreement should have said.
// A control arm that cannot vary the thing under test is not a control: CLAUDE.md's own rule, earned again.
//
// 🔴 AND THE STEP IS 46% PER-DISPATCH OVERHEAD, WHICH IS NOW MEASURED RATHER THAN INFERRED. A step's wall is
// 230.6 ms while the per-dispatch profile sums 123.2 ms over 419 dispatches, and two earlier explanations of
// that gap were both wrong: `Model::hostI` does `sync()` so three loop-invariant reads were draining the queue
// sixty times a generation (hoisted out - correct, and the wall did not move), and the profile does NOT miss
// unlabelled work (`dispatch` defaults the label to the kernel's own name, so all 419 are timed).
// What settles it is a measurement core could not make: `LF_GPUBUSY` sums every COMMITTED command buffer's own
// `GPUEndTime - GPUStartTime` with batching left ON, which the per-dispatch profile cannot do because it gives
// each dispatch a buffer of its own. It reports **228.1 ms a step over 11 buffers** against a 230.6 ms wall.
// So the GPU is busy ~100% of the step and the port is GPU-BOUND - and the 105 ms between 228.1 and 123.2 is
// GPU-side overhead spread over 419 dispatches, about **0.25 ms each**.
// 🔴 AND THE 105 ms IS NOT THE DISPATCH COUNT - FOUR EXPERIMENTS, AND THE LAST TWO ARE THE ones THAT SETTLE
// IT. The GPU-busy figure had to be fixed first: summing every buffer's `GPUEndTime - GPUStartTime`
// DOUBLE-COUNTS, because several are in flight at once, and the tell was that the sum came out the same per
// step whether the port ran 6 buffers a step or 744. Taken as the UNION of the spans it reads **13542 ms of a
// 13668 ms span - 99% busy, 225.7 ms a step** against a 228 ms wall. So the port is GPU-bound and there is no
// host-side lever; that part is solid.
// Then what the 105 ms is NOT:
//   * not a spurious `sync` - three loop-invariant `M.hostI` reads were draining the queue sixty times a
//     generation (each is `i(name); sync(); host(p)`), hoisted out, and the wall did not move;
//   * not unlabelled work - `dispatch` defaults a label to the kernel's own name, so all 391 are timed;
//   * not the dispatch COUNT, measured twice. Merging two pairs that share an input took 419 dispatches a step
//     to 391 with the kernel sum 123.2 ms to 119.4, and the wall did not move. Then **36,000 extra trivial
//     dispatches** were added to price one directly: the slope puts a dispatch at **0.02 to 0.07 ms**, so all
//     391 are 8 to 27 ms a step - a fifth of the gap at most, which is exactly why removing 28 of them was
//     invisible.
//   * and it cannot be attributed per dispatch in situ at all on this GPU: core.mm records the probe - an M5
//     supports counter sampling only `AtStageBoundary`, which a compute encoder cannot use.
// 🔴 SO THE GAP SCALES WITH THE DATA, NOT WITH THE COUNT, and that is the conclusion this file already drew
// from the other end: the lever is FEWER PASSES OVER THE PAIR TRACK. ~90 ms a step of GPU time that no kernel
// timestamp attributes, over a 20 MB pair track touched by every stage, is most consistent with memory-system
// work between dependent dispatches - and it is why the gate's fusion paid 5.2% while removing 28 dispatches
// paid nothing: the fusion removed two whole PASSES over four pair-sized planes.
// 🔴 THE MERGES ARE KEPT, AND NOT AS A SPEED WIN: one GEMM where there were two (`to_qkv` [3D][D] and `to_g`
// [D][D] stacked into [4D][D], whose natural layout IS core's packed q | k | v | g row, so the output stride
// the two-GEMM form needed went with it), one dispatch where there were two (q's and k's RMSNorms, the only
// independent pair), a stride trick gone, 3.8 ms less kernel time. Facts about the code, not about this
// laptop's temperature. What cannot merge: the two adaLNs a layer are sequential (the transition's reads `s`
// after the attention), the node norm precedes q and k, and the triangle's `linear_z` and `linear_g` read
// different tensors.
// 🔴 AND MEASUREMENT HAS A FLOOR HERE. After a long session the job that runs in 12.8 s cool takes 22.4 s, and
// three interleaved rounds of a **12%** effect (those 36,000 dispatches) came out -0.23, +1.95, +0.23 s. Every
// number in this file above 4% was taken interleaved and survived both orderings; nothing smaller should be
// believed from here without a cold machine.
//
// So no kernel here is worth another tile sweep. The lever left is FEWER PASSES OVER THE PAIR TRACK - fusing an
// elementwise pass into the epilogue of the GEMM that produced it, as the gate, the residual and the next
// LayerNorm already are - and `gemmGatedAdd`'s own 878 MB is the measure of what one such pass costs. The
// attention no longer reaches the profile's top twenty-four.
//
// 🔴 AND THE ONE WORTH TAKING WAS THE TRIANGLE'S GATE INTO ITS PROJECTION, WHICH IS TAKEN: **5.2%**. The
// two-step form wrote four pair-sized planes and read all four back to write two - 139.5 MB a call at 165
// tokens against the 27.9 MB the fused form writes - and metal/core already had the epilogue, because
// `gemmTriGate` is what af3's and bioemu2's triangles run. One thing had kept this port out of it: the host
// wrapper set the contraction's depth from the channel count (`g.m = 4 * C; g.k = C; g.ldb = C` out of one
// `C`), and this triangle is 384 in and 256 out. `k`, `m` and `tgC` are separate GemmArgs fields and the
// epilogue reads only `tgC` and `m / 4`, so the depth was free all along: `gemmTriGate` takes an `in` now, 0
// meaning C, which is every AF3-lineage call and leaves them expression-for-expression as they were.
// The weight is the other half. The epilogue wants columns in blocks of 8 - eight channels' pa, then their ga,
// pb and gb - so a simdgroup's four accumulators hold one channel's four values at the same lane and gate them
// with no shuffle; read backwards out of `lf_tri_quarters` that is `[in][C / 8][4][8]`, where this port's
// `weightStack` builds `[4T][P]`. `triGateWeight` builds it once and quarters it once rather than per dispatch.
// Measured, ABAB over three rounds at 165 tokens and 60 steps: **13.06 / 13.51 / 13.92 s against 13.94 / 14.20
// / 14.55**, fused ahead in all three by 0.63 to 0.88 s, mean 0.73 of 14.2. The ceiling was 7.6% (the gate's
// dispatch skipped entirely) and the fused GEMM costs some of it back: 2.02 ms a call against the plain
// projection's 1.81, against the 0.63 ms the gate pass took. `rfd4p_tri_ab_hm` is gone from the profile and
// the design is the same structure, 1315 of 1315 atoms matched at 0.0095 A rms.
// 🔴 AND THE TWO PORTS THAT SHARE THAT EPILOGUE ARE UNMOVED, shown rather than argued, and BOTH were run.
// With `in` 0 every expression reduces to what it was; metal/af3's gate folds its nineteen cases to the same
// pLDDT, pTM and RMSD with the change and without it, and metal/bioemu2's stage comparison against its own
// webgpu reference is IDENTICAL on every line - the trunk's four passes, the distogram, the context and all
// four denoise sigmas - built from `ab42d072^`'s core and from this one.
// 🔴 AND "NO LANE FOR THE bioemu2 REFERENCE" WAS WRONG TWICE OVER. `tools/dump-bioemu2-webgpu.mjs` runs in
// this Mac's own Chrome (adapter apple / metal-3) and wrote 31 entries in under two minutes; the input it is
// compared against comes from `cuda/featurise/bioemu2-featurise`, which is built here - the port's own
// `--job=` featurises into a temp directory it then REMOVES, which is what made the input look unobtainable.
// `LOCALFOLD_RFD4P_TRI_FUSED=0` is the two-step control, kept because it is what the `--f32` arm and a
// checkpoint with `tri_mult_bias` run: the epilogue's operands are half, and its bias wants a [4T] f32 in the
// blocks-of-8 order that no released checkpoint has.
//
// 🔴 AND IT DOES NOT NEED cuda/rfd4p's `longForm`, WHICH IS THE ONE THING ITS SOURCE HAS THAT THIS PORT DOES
// NOT. That is a second triangle shape there, taken above ~40,000 pairs because the short form degrades with
// length; this port has one shape - channel-major, now fused with its gate - and it is the long form's
// equivalent already. Measured rather than assumed: at 400 tokens (160,000 pairs, 5.88x the 165-token case)
// the step is 626.7 ms against 113.5, which is 5.52x, SUB-linear in pairs, and the mix is flat to a tenth of a
// point - triangle gate 24.8% against 24.9%, gated add 15.6% against 15.8%, the pair transition 10.7% against
// 9.9%. There is no crossover here to put a second shape on either side of.
//
// WHAT RUNS: unconditional, motif scaffolding, protein-protein interfaces, small molecules (whose chirality is
// re-aligned every step), coordinate groups (whose labels are permuted every step), partial diffusion and
// inverse folding. Measured, consecutive CA-CA within a chain at 60 steps: unconditional 3.81 A over
// [3.69, 3.92], motif 3.80, interface 3.80, small molecule 3.83, partial 3.74 - and inverse folding returns the
// input's own backbone to 0.0005 A, the nm-to-angstrom rounding.
// 🔴 AND THE ARITHMETIC IS HELD TO THE REFERENCE AT LENGTH NOW, NOT ONLY AT 54 AND 67 TOKENS. An oracle was
// built at **183 tokens** - 33,489 pairs, **11.5x** the 54-token case's 2,916 - which took the CPU reference
// 39 minutes in one Node process (the triangle is n^3, so it is 40x that case's work) and 841 MB of stages.
// Every arm passes there: **86 stage comparisons**, 100 with the sampler's per-step trajectory beside them,
// and 86 again under `--f32`, with step 0's sequence features at 1.9e-9 and its pair features at exactly 0.
// So the fused gate, the flash attention and the folded pair-bias norm are all held to the reference at a
// length where each of them is doing an order of magnitude more work than when they were written.
// 🔴 AND AT 400 TOKENS TOO, which is 160,000 pairs and 5.9x the largest case any oracle here covers: the
// converged backbone is **3.85 A over [3.76, 3.94]**, the same band as 54 residues, and three runs are
// byte-identical. That is the length check that matters for the fused kernels - a tile edge case at 160,000
// pairs would show up as a broken chain or as a run that differs from itself, and neither happens. (The
// half-against-f32 differential is NOT the check to use there: at 8 steps the trajectory is chaotic, CA-CA
// median 20 A, and the two arithmetics pick different residues at a few positions - 2598 of 2612 atoms
// matched at 0.19 A rms. That is precision sensitivity on an unconverged sample, not a disagreement.)
// 🔴 READ THOSE AT A CONVERGED STEP COUNT: the same code at 2 steps gives 11 A and at 60 gives 3.81. The
// trajectory gate below runs at 2 steps and says nothing about geometry; they are different questions.
// 🔴 AND HALF COSTS 0.022 A: the same seed through both arithmetics gives designs 0.022 A RMS apart (worst
// 0.059) with 53 of 54 residues identical, against 13 to 23 A between seeds.
//
// 🔴 AND THIS LIST USED TO READ AS TWO UNPORTED FEATURES AND WAS NEITHER. It said "REFUSED BY NAME: a bias on
// the triangle's projections (the channel-major projection cannot carry one), and a `coord_group_seq` block
// whose place in the 897 sequence columns cannot be located" - and a reader (human or otherwise) takes that as
// two things this port cannot do. Both entries were wrong, in opposite directions, and a doc that misdescribes
// its own coverage costs more than a missing one.
//   - `coord_group_seq` IS PORTED AND EXERCISED. The 37 columns are LOCATED (`coordGroupOffset`) from the
//     labels the featuriser wrote at step 0, and the group labels are permuted every step; two gate jobs carry
//     three coordinate groups each and both are green. The `die` is a GUARD - no match means the feature's
//     place cannot be assumed, and a guessed offset would overwrite some other feature - and it does not fire
//     on an export this featuriser wrote.
//   - THE TRIANGLE'S BIASES NOW COME THROUGH, in the gate rather than the GEMM. The old reason was false of
//     the triangle: core's bias is per output COLUMN, which the channel-major product cannot use, but a
//     per-CHANNEL bias in that layout is a per-ROW constant and the gate already reads every element of all
//     four planes. `rfd4p_tri_ab_hmb` adds them; `opt` returning a tensor picks it, so the common path is
//     byte-identical and pays nothing for a feature no released checkpoint has.
// 🔴 VERIFIED WITHOUT A CHECKPOINT THAT HAS THEM, because none does (`tri_mult_bias: false`) and an
// unexercised path is a claim rather than a feature. Synthetic per-channel biases were injected, the gate's
// operands and its result dumped, and `aT` held to `(aP + baP[c]) * sigmoid(aG + baG[c])`: the channel-major
// mapping `c = t / (n * n)` matches to **4.9e-4** relative, which is half-precision storage of the result,
// while `t % T`, `(t / T) % T` and "no bias at all" are out by **19x, 34x and 34x**. So the one thing that
// could be wrong in six lines - which plane an element is in - is the thing the check discriminates.
//
// REFUSED BY NAME rather than run as something else: nothing, in this checkpoint's configuration.
//
// AGAINST cuda/rfd4p, FLAG BY FLAG, because "fully ported" is a claim and a claim should be checkable. Every
// flag that port's parser accepts is here now, and this is what each maps to:
//   --job= --work-dir= --out= --seed= --designs= --oracle= --encoder= --metrics= --dump-sample=
//                      the same, and `--out=x.cif` picks mmCIF in both
//   --int8= --weights=  `--bundle=` here: one bundle, the page's published int8, where cuda/rfd4p takes either
//                       that or a float32 checkpoint directory
//   --generate=         `--gen=`
//   --refold[=af3|af2] --recycles= --repeat= --no-insertion
//                       the same, and all exercised: `--recycles=2` reports two recycles, `--repeat=3`
//                       prices a trunk pass, `--no-insertion` leaves the 40 unindexed tokens of
//                       `motif_unidx_1prw` ungrafted and writes a different file
//   --config= --length= --step= --spec=
//                       forwarded to the featuriser, which is what cuda/rfd4p's own FEATURISER_ONLY list does
//   --fast              this port's default IS the fast arithmetic - half weights, activations and pair track -
//                       and `--f32` is the control, so the axis is here with its sense reversed
//   --gen-oracle= --enc-oracle=
//                       the generation's and the encoder's oracle directories, which this port reads through
//                       `--gen=` and `--encoder=` beside `--oracle=`
// 🔴 AND `--refold[=af3|af2]` IS HERE TOO, which this file said twice that it could not be. Upstream's
// REFOLDING ORACLE: each design that passes the pre-oracle filters refolded by this tree's own AlphaFold 3 or
// AlphaFold 2 and scored as upstream scores it. The claim that it needed "a job interface in two other ports"
// was simply wrong - metal/af3, metal/af2 and metal/ef2 all take `--job=` through `lf::standalone::main` (the
// dispatch is one line: an argv[1] starting `--`), and `--model=`, `--msa-mask=` and `--template-pad=` are
// featuriser flags that layer already forwards. Reading a port's usage line and stopping there cost two rounds.
// `oracle.h` is 1491 lines of host-only C++ that derive.sh repoints like the others, so nothing was rewritten.
// What it really needed was two small things: `/proc/self/exe` does not exist on macOS, so the sibling
// binary's path is found with `_NSGetExecutablePath` in metrics.mm (the relative arithmetic is identical -
// metal/rfd4p/localfold-rfd4p sits the same distance from metal/af3/localfold-af3 as its cuda/ twin does), and
// AF2's oracle reads `--raw-confidences`, which metal/af2 did not write and now does.
// Both run, on `gate-oracle-motif-60` at 60 steps: af3 gives oracle pLDDT 86.9, a refolded backbone RMSD of
// 2.21 A and a motif all-atom RMSD of 1.10; af2 gives 73.28 and 5.27 / 4.90, with `model_1_ptm` named in the
// output as upstream labels it. `--refold` without `--metrics=` is refused by name, because the oracle runs on
// what the filters pass and that file is what computes them.
//
// THE GATE, over an exported input directory rather than a job:
//   localfold-rfd4p <input dir> --bundle=<model-rfd4p-int8> [--oracle=<dir>] [--encoder=<dir>] [--gen=<dir>-gen]
//     [--out=design.pdb] [--f16]
// built by `cuda/rfd4p/export_input.mjs <in> --oracle=<o> [--encoder] [--generate --gen-oracle] --steps=N
// --checkpoint=$PWD/model-rfd4p-int8/model.safetensors` (that file from `tools/rfd4p-bundle-to-safetensors.mjs`:
// the int8 shards dequantised, so the oracle is built from the weights this port runs and the quantisation
// cancels). Run BOTH recycle counts - `--recycles=2` is the only arm that exercises the cycle projections.
//
// FOUR ARMS, and the first two are the ones that hold the kernels:
//   * every trunk stage FROM THE ORACLE'S OWN INPUT at 1e-6 - each kernel answering for its own arithmetic;
//   * every autoencoder layer the same way at 1e-5 (it amplifies: see the bounds below);
//   * the whole trunk and the decoder CHAINED, reported at 2e-6 and 1e-4;
//   * the sampler against the reference's own per-step trajectory at 1e-4.
#include "rfd4p.h"
#include <cmath>
#include <cstring>
#include <map>
#include <vector>
#include <fstream>
#include <filesystem>

extern const char* PORT_SOURCE;

// RFD4-Proteina's own featuriser, linked in rather than run as a second process
// (cuda/rfd4p/featurise/featurise.cpp, the same object cuda/rfd4p links).
namespace rfd4p { namespace featurise { int run(int argc, char** argv); } }

namespace {

double WORST_ABS = 0;
double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i];
    if (!std::isfinite(d)) return INFINITY;
    num += d * d;
    den += (double)b[i] * b[i];
    worst = std::max(worst, d < 0 ? -d : d);
  }
  WORST_ABS = worst;
  return sqrt(num / std::max(den, 1e-30));
}

int FAILED = 0;
// 🔴 AND HOW MANY COMPARISONS ACTUALLY HAPPENED, because "all green" was printed by a run that compared
// NOTHING. `--job=` sets the generation directory, which is what the final line tests, while the stage arms
// need `--oracle=` and the trajectory arm needs a gen-oracle - so a job run printed "all green" with no
// reference on disk and no `check` ever reaching a bound. It was reported as a verified fold through this whole
// optimisation round, and the probe that found it skipped the triangle's gate entirely: the design came out at
// CA-CA median 5.26 A against 3.80, min 0.54, and the line still read "all green". A gate that cannot fail is
// not a gate, and a gate that says it passed when it did not compare is worse than one that is absent.
int CHECKED = 0;
// ours (on the device, n values) against the oracle's "o/<name>". `bound` 0 or less REPORTS the number without
// gating on it - for a stage whose residual is its INPUT's, measured, and which has its own arm that is gated.
double check(const char* label, const std::string& name, const float* ours, size_t n, double bound) {
  // a name with its own prefix (the decoder's "d/...") is taken as it is; a bare one is the trunk's "o/..."
  const std::string key = name.find('/') == std::string::npos ? "o/" + name : name;
  if (!M.has(key)) { printf("  %-24s (not in the oracle)\n", label); return -1; }
  CHECKED += 1;
  if (M.len(key) != n) die("the oracle's %s has %zu values, this port's %zu", key.c_str(), M.len(key), n);
  sync();
  const double r = relRms((const float*)host(ours), M.hostF(key), n);
  // maxAbs beside it, because that is what `tools/gpu/check-decoder.js` holds the page's own decoder to (1e-4 on
  // the float32 arm) and a number nobody can compare with the other ports' is a number that says less.
  if (bound <= 0) { printf("  %-24s relRMS %.3e  maxAbs %.2e  (reported)\n", label, r, WORST_ABS); return r; }
  const bool ok = r <= bound;
  if (!ok) FAILED += 1;
  printf("  %-24s relRMS %.3e  maxAbs %.2e  %s\n", label, r, WORST_ABS, ok ? "ok" : "FAILED");
  return r;
}

double checkFn(const char* label, const std::string& key, const float* ours, size_t n, double bound) {
  return check(label, key, ours, n, bound);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, bundle, oracle, encDir, genDir, outPath, job, workDir, seed, metricsPath, dumpSample;
  std::vector<std::string> featuriserArgs;
  int recycles = 0, repeat = 0;
  std::string refold;
  int designs = 1;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundle = a + 9;
    else if (!strncmp(a, "--oracle=", 9)) oracle = a + 9;
    else if (!strncmp(a, "--encoder=", 10)) encDir = a + 10;
    else if (!strncmp(a, "--gen=", 6)) genDir = a + 6;
    else if (!strncmp(a, "--out=", 6)) outPath = a + 6;
    else if (!strncmp(a, "--job=", 6)) job = a + 6;
    else if (!strncmp(a, "--work-dir=", 11)) workDir = a + 11;
    else if (!strncmp(a, "--seed=", 7)) seed = a + 7;
    else if (!strncmp(a, "--designs=", 10)) designs = atoi(a + 10);
    else if (!strncmp(a, "--metrics=", 10)) metricsPath = a + 10;
    else if (!strncmp(a, "--dump-sample=", 14)) dumpSample = a + 14;
    // 🔴 `--refold[=af3|af2]`: UPSTREAM'S REFOLDING ORACLE, which needs no change to the oracle ports at all -
    // metal/af3, metal/af2 and metal/ef2 all take `--job=` through `lf::standalone::main`, and `--model=`,
    // `--msa-mask=` and `--template-pad=` are featuriser flags that layer already forwards. `--refold=af2`
    // additionally wants `--raw-confidences` - the PAE head's logits at full precision, which the oracle
    // reduces per chain pair - and metal/af2 has that now, so both oracles run.
    else if (!strcmp(a, "--refold")) refold = "af3";
    else if (!strncmp(a, "--refold=", 9)) refold = a + 9;
    // cuda/rfd4p's own flags, so a command that drives one port drives the other
    else if (!strncmp(a, "--recycles=", 11)) recycles = atoi(a + 11);
    else if (!strncmp(a, "--repeat=", 9)) repeat = atoi(a + 9);
    else if (!strcmp(a, "--no-insertion")) NO_INSERTION = true;
    // 🔴 ...AND THE FEATURISER'S OWN, FORWARDED RATHER THAN REFUSED. `--config=`, `--length=` and `--step=`
    // are read by the featuriser and by nothing in this port, which is why cuda/rfd4p keeps a
    // FEATURISER_ONLY list: a flag this parser rejects is a job that cannot be expressed here at all, and a
    // flag it silently swallows is worse.
    else if (!strncmp(a, "--config=", 9) || !strncmp(a, "--length=", 9) || !strncmp(a, "--step=", 7))
      featuriserArgs.push_back(a);
    else if (!strcmp(a, "--f16")) HALF = true;
    else if (!strcmp(a, "--f32")) HALF = false;
    else if (a[0] != '-') input = a;
    else die("unknown flag %s", a);
  }
  // 🔴 `--job=<spec>` IS ONE PROCESS AND NO JAVASCRIPT. The featuriser is linked in, so a condition spec becomes
  // this port's input here - by the same code and the same bytes `rfd4p-featurise` writes, which
  // `npm run test:featurise`-style gates hold to the JavaScript's to the last place. The directory is still
  // written to disk, because that is what the port reads and what a second run reuses; what is gone is the
  // second process and the Node it needed. cuda/rfd4p's `jobMain` is the reading.
  if (!refold.empty()) {
    if (refold != "af3" && refold != "af2") {
      fprintf(stderr, "--refold=%s: the oracles are af3 and af2\n", refold.c_str());
      return 2;
    }
    if (metricsPath.empty()) {
      fprintf(stderr, "--refold needs --metrics=<file.json>: the oracle runs on the designs that pass the "
                      "pre-oracle filters, which is what that file computes\n");
      return 2;
    }
  }
  if (!job.empty()) {
    if (bundle.empty()) { fprintf(stderr, "--job= needs --bundle=<model-rfd4p-int8>\n"); return 2; }
    if (workDir.empty()) workDir = job + ".featurised";
    // 🔴 THE FEATURISER READS THE PUBLISHED LAYOUT (`<dir>/shards/<file>`) AND A LOCAL EXPORT IS FLAT.
    // `tools/export_rfd4p_model.py` writes manifest.json beside its shards; the deployed bundle the port fetches
    // keeps them under `shards/`. Rather than teach one side the other's layout, a run stages a directory of
    // symlinks when it needs to - the bytes are not copied, and the bundle this port reads is untouched.
    std::string fbundle = bundle;
    {
      std::error_code ec;
      if (!std::filesystem::exists(bundle + "/shards", ec) && std::filesystem::exists(bundle + "/shard_000.bin", ec)) {
        const std::string stage = workDir + "-weights";
        std::filesystem::create_directories(stage + "/shards", ec);
        const std::filesystem::path src = std::filesystem::absolute(bundle);
        for (const char* f : {"manifest.json", "config.json"}) {
          std::filesystem::remove(stage + "/" + f, ec);
          std::filesystem::create_symlink(src / f, stage + "/" + f, ec);
        }
        for (const auto& e : std::filesystem::directory_iterator(src, ec)) {
          const std::string name = e.path().filename().string();
          if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin")) continue;
          std::filesystem::remove(stage + "/shards/" + name, ec);
          std::filesystem::create_symlink(e.path(), stage + "/shards/" + name, ec);
        }
        fbundle = stage;
      }
    }
    std::vector<std::string> fargs = {"rfd4p-featurise", workDir, "--spec=" + job, "--generate", "--int8=" + fbundle};
    if (!seed.empty()) fargs.push_back("--seed=" + seed);
    for (const std::string& f : featuriserArgs) fargs.push_back(f);
    std::vector<char*> fargv;
    for (std::string& f : fargs) fargv.push_back(f.data());
    const double tf = now();
    if (const int rc = rfd4p::featurise::run((int)fargv.size(), fargv.data())) return rc;
    printf("featurised %s in %.2f s\n", job.c_str(), now() - tf);
    input = workDir;
    genDir = workDir + "-gen";
    if (outPath.empty()) outPath = "design.pdb";
    // 🔴 THE ENCODER'S INPUTS FOR A PARTIAL RUN AND ONLY FOR ONE: the featuriser writes `<work>-enc` when the
    // job is one and says so in the generation's metadata, and handing `--encoder=` to an ordinary run would
    // read whatever a previous job left in the same work directory.
    std::ifstream idx(genDir + "/model.idx");
    std::string kind, name;
    while (idx >> kind >> name) {
      if (kind != "m") { std::string rest; std::getline(idx, rest); continue; }
      double v = 0;
      idx >> v;
      if (name == "meta/partial" && v != 0) encDir = workDir + "-enc";
    }
  }

  if (input.empty() || bundle.empty()) {
    fprintf(stderr, "usage: localfold-rfd4p --job=<spec.json|.yaml> --bundle=<model-rfd4p-int8> [--out=design.pdb|.cif]\n"
                    "         [--designs=N] [--seed=N] [--metrics=<file.json>]\n"
                    "         [--dump-sample=<dir>] [--refold[=af3|af2]] [--recycles=N] [--repeat=N]\n"
                    "         [--no-insertion] [--f32]\n"
                    "       localfold-rfd4p <input dir> --bundle=<model-rfd4p-int8> [--oracle=<dir>] "
                    "[--encoder=<dir>] [--gen=<dir>-gen] [--out=design.pdb] [--designs=N] [--f32]\n");
    return 2;
  }

  // LOCALFOLD_RFD4P_CHURN=<MB>: host memory dirtied and freed before anything else, which is what the linked-in
  // featuriser does to this process before the fold starts. A fold that is bit-identical with and without it does
  // not depend on a fresh allocation's contents; one that is not, does - and that is a bug in the fold.
  if (const char* mb = getenv("LOCALFOLD_RFD4P_CHURN")) {
    const size_t bytes = (size_t)atof(mb) * (1u << 20);
    std::vector<unsigned char> churn(bytes, 0xa5);
    volatile unsigned char sink = 0;
    for (size_t k = 0; k < bytes; k += 4096) sink = (unsigned char)(sink + churn[k]);
    (void)sink;
  }

  setSource("rfd4p", PORT_SOURCE);
  const double t0 = now();
  loadWeights(bundle);
  checkDims();
  M.loadInput(input);
  if (!oracle.empty()) M.loadInput(oracle);
  // the encoder's own input directory and the reference's answer to it (`export_input.mjs --encoder`)
  // the encoder's input, and the reference's answer to it only where that was written (a job run has none)
  if (!encDir.empty()) {
    M.loadInput(encDir);
    if (FILE* probe = fopen((encDir + "-oracle/model.idx").c_str(), "r")) { fclose(probe); M.loadInput(encDir + "-oracle"); }
  }
  // a generation's exported inputs, and the reference's own trajectory over them where it was written
  if (!genDir.empty()) {
    M.loadInput(genDir);
    const std::string go = genDir + "-oracle";
    FILE* probe = fopen((go + "/model.idx").c_str(), "r");
    if (probe) { fclose(probe); M.loadInput(go); }
  }
  D_.n = (int)M.meta("meta/n");
  // `--recycles=` overrides what the input was exported with, as cuda/rfd4p's does.
  D_.nRecycle = recycles > 0 ? recycles : (int)M.meta("meta/nRecycle", 1);
  const int n = D_.n;
  const size_t nn = (size_t)n * n;
  printf("rfd4p: %d tokens, %d recycle%s, bundle %s (%.2f GB of weights in %.2f s)\n", n, D_.nRecycle,
         D_.nRecycle == 1 ? "" : "s", bundle.c_str(), M.weightBytes() / 1e9, now() - t0);

  // 🔴 THE FEATURES ARE THE ORACLE'S OWN, which is what makes a residual this port's. The featuriser is
  // `shared/rfd4p/features.js` through the input directory; nothing about a molecule is decided here yet.
  const float* seqF = M.f("in/seq");
  const float* cond0 = M.f("in/cond0");
  const float* pairF = M.f("in/pair");
  const float* pairCondF = M.f("in/pairCond");

  // the attended tokens, as indices: `attn = padding_mask & ~is_in_context_conditioning`, which the input dir
  // carries ready-made. The autoencoder runs over these rows and nothing else.
  int attnCount = 0;
  int* attnIdx = nullptr;
  if (M.has("in/attnMask")) {
    const float* am = M.hostF("in/attnMask");
    std::vector<int> idx;
    for (int i = 0; i < n; ++i) if (am[i] != 0) idx.push_back(i);
    attnCount = (int)idx.size();
    attnIdx = uploadNew(idx.data(), idx.size());
  }

  float* c = allocT<float>((size_t)n * D_.C);
  float* sInit = allocT<float>((size_t)n * D_.D);
  // 🔴 THE GATE KEEPS THE TRACK IN THE ARM'S OWN PRECISION, so what it holds is what ships: under `--f16` the
  // pair track is half here too, and the comparisons go through a float copy rather than a float track.
  Z zInit;
  if (HALF) zInit.h = allocT<half>(nn * D_.P); else zInit.f = allocT<float>(nn * D_.P);
  float* zCmp = allocT<float>(nn * D_.P);
  auto asFloat = [&](Z zz) {
    if (!zz.h) return zz.f;
    toFloat(zz.h, zCmp, nn * D_.P);
    return zCmp;
  };
  float* c1 = allocT<float>((size_t)n * D_.C);
  float* bbCaS = allocT<float>((size_t)n * 3);
  float* latentsS = allocT<float>((size_t)n * 32);
  float* logitsS = allocT<float>((size_t)n * 147);
  float* insertionS = allocT<float>(nn);

  condC(cond0, c, n);
  initSingle(seqF, sInit);
  initPair(pairF, pairCondF, zInit);

  // 🔴 `--repeat=N`: THE TRUNK ON A CLOCK, as cuda/rfd4p's own arm is - it compares nothing and gates nothing,
  // it prices a pass. The state is re-zeroed every time because the first cycle's input IS the init product
  // (trunk.mm says why), so a pass that inherited the last one's `s` would be timing a different computation.
  if (repeat > 0) {
    float* rs = allocT<float>((size_t)n * D_.D);
    Z rz;
    if (HALF) rz.h = allocT<half>(nn * D_.P); else rz.f = allocT<float>(nn * D_.P);
    float* rbb = allocT<float>((size_t)n * 3);
    float* rll = allocT<float>((size_t)n * 32);
    float* rlg = allocT<float>((size_t)n * 147);
    double ms = 0;
    for (int i = 0; i < repeat; ++i) {
      fill(rs, 0, (size_t)n * D_.D * 4);
      fill(rz.h ? (void*)rz.h : (void*)rz.f, 0, nn * D_.P * (rz.h ? 2 : 4));
      sync();
      const double t0 = now();
      trunkForward(seqF, cond0, M.has("in/cond1") ? M.f("in/cond1") : cond0, pairF, pairCondF, D_.nRecycle,
                   rs, rz, rbb, rll, rlg, nullptr);
      sync();
      ms += (now() - t0) * 1e3;
    }
    printf("trunk: %.3f ms a pass (%d tokens, %d recycle%s, %d passes)\n", ms / repeat, n, D_.nRecycle,
           D_.nRecycle == 1 ? "" : "s", repeat);
  }

  float* s = allocT<float>((size_t)n * D_.D);

  if (!oracle.empty()) {
    printf("held to %s (cpu/rfd4p/reference.js over these same weights):\n", oracle.c_str());
    // 🔴 1e-6 FOR A FLOAT32 ARM, WHICH IS A REORDERING BOUND AND NOT A TOLERANCE ANYBODY PICKED: the reference
    // sums a row in source order on the host and the GEMM sums it in tiles, so the two differ by float
    // non-associativity and nothing else. `--f16` is a different question and will carry its own bound.
    const double bound = HALF ? 3e-3 : 1e-6;
    // 🔴 THE CHAINED ARM'S BOUND IS THE MEASURED ACCUMULATION, AND IT IS NOT A WIDER TOLERANCE FOR THE SAME
    // QUESTION. Arm 1 holds every kernel at `bound` on the reference's own input; this arm asks a different
    // thing - does fourteen layers of float32 drift stay small - and the answer is monotone in depth (1.0e-6 at
    // z10 over both recycle counts). 2e-6 is that with a factor of two, and a failure here with arm 1 green is
    // drift, where a failure in arm 1 is a kernel.
    const double chainedBound = HALF ? 6e-3 : 2e-6;
    // 🔴 THE AUTOENCODER'S PER-STAGE BOUND IS THE TRUNK'S, AND ITS CHAINED ONE IS NOT - BECAUSE THE DECODER
    // AMPLIFIES. Measured over 54 tokens, every decoder layer fed the reference's own input reads 7.4e-8 to
    // 9.9e-7 and is FLAT in depth, exactly as the trunk's are; chained, the same layers read 6.9e-7 at s0,
    // 6.7e-6 at s2 and 3.4e-5 by s4, then settle back to 4.5e-6 by s11. So the stack's own gain is above one on
    // this input and nothing is wrong with it: the per-stage arm is the gate and the chained arm is the report.
    // 🔴 AND THE AMPLIFICATION IS INPUT-DEPENDENT, which is the trap this nearly became: the same code on the
    // one-recycle input chains to 2.9e-6 and on the two-recycle input to 3.4e-5, from latents that differ by
    // 0.2 and a C_SEQ that differs in most positions while every magnitude stays the same to three figures. A
    // bound fitted on one input would have called the other a regression. Only the per-stage arm is stable
    // enough to hold a kernel to.
    // 🔴 AND ONE SUM IS WORTH COMPENSATING, measured rather than assumed: the attention's score dot product
    // runs over q and k that carry a LayerNorm's learned BIAS, so it is a product of vectors with a large
    // common component and what the softmax reads is the cancellation. Kahan-compensating that one sum took the
    // attention output from 3.43e-6 to 1.53e-6 and every stage below it with it (s0 1.26e-6 -> 5.9e-7, coors
    // 1.83e-6 -> 9.0e-7), while the same compensation on the softmax and the context sums moved NOTHING. The
    // trunk's q and k are RMSNormed and carry no bias, which is why its attention never needed it.
    // 🔴 1e-5 AND NOT THE TRUNK'S 1e-6, FROM TWO INPUTS RATHER THAN ONE. Over 54 tokens every decoder layer on
    // the reference's own input reads 7.4e-8 to 9.9e-7; over the 67-token partial-diffusion input the same
    // layers read ~4e-7 except LAYER 2, which reads 3.4e-6 - and layer 2 is the largest of the 54-token set
    // too (9.9e-7), so what is near-degenerate is that layer's input and not its code. A bug in a layer would
    // move every layer, since there is one `aeLayer`; this moves one, on both inputs, by the same ratio.
    const double decBound = HALF ? 3e-3 : 1e-5;
    // the chained decoder, reported at the level measured above with room for another machine and another input
    const double decChained = HALF ? 1e-2 : 1e-4;
    check("c", "c", c, (size_t)n * D_.C, bound);
    check("s_init", "sInit", sInit, (size_t)n * D_.D, bound);
    check("z_init", "zInit", asFloat(zInit), nn * D_.P, bound);
    // 🔴 EVERY STAGE COMPARED THE MOMENT IT IS PRODUCED, so the first one that moves is the one named - and the
    // pair track is compared after each update rather than only at the end, which is how `cuda/rfd4p` found three
    // of its bugs (its README: "o/s0 rather than a finished trunk that was merely wrong").
    Z z;
    if (HALF) z.h = allocT<half>(nn * D_.P); else z.f = allocT<float>(nn * D_.P);
    auto loadZ = [&](const std::string& key) {
      if (z.h) toHalf(M.f(key), z.h, nn * D_.P);
      else copy(z.f, M.f(key), nn * D_.P * 4);
    };

    // ---- arm 1: EVERY STAGE FROM THE ORACLE'S OWN INPUT ----------------------------------------------------
    // 🔴 THIS IS THE ARM THAT HOLDS THE KERNELS, AND THE CHAINED ONE BELOW CANNOT. Chained, a stage is handed
    // what the stage before it computed, so what reaches layer 13 is fourteen layers of accumulated float
    // reordering - the reference accumulates every dot product in float64 (`linearLayer` sums into a JS number
    // and stores to a Float32Array) where the GEMM accumulates in float32 and in tiles, so the two drift apart
    // monotonically with depth: measured 2.9e-7 after layer 0 and 1.0e-6 on the pair track after layer 10.
    // Fed the reference's own input, each stage answers for its own arithmetic alone - which is how a defect in
    // layer 11 stops being indistinguishable from drift, and is what metal/bioemu2's oracle arm does.
    {
      const float* cc = D_.nRecycle > 1 ? (condC(M.f("in/cond1"), c1, n), c1) : c;
      const char* zFrom = "zInit";
      // 🔴 LAYER 0 IS HELD ONLY AT ONE RECYCLE, AND THE REASON IS THE ORACLE'S AND NOT THIS PORT'S: the stages
      // it records are the LAST cycle's, and that cycle's layer 0 reads s_init + Lin(RMSNorm(s of the cycle
      // before)) rather than s_init itself. Nothing in the dump is that tensor, so feeding this arm `o/sInit` at
      // two recycles compares the right kernel against the wrong input - measured 1.8e-1, which is what said so.
      // The chained arm holds it instead, and s1 onwards are held here either way.
      const int first = D_.nRecycle > 1 ? 1 : 0;
      if (first) printf("  %-24s (the oracle records the last cycle only - held by the chained arm)\n", "stage s0, z0");
      if (first) { zFrom = "z0"; }
      for (int i = first; i < D_.layers; ++i) {
        const std::string sFrom = i == 0 ? "o/sInit" : "o/s" + std::to_string(i - 1);
        copy(s, M.f(sFrom), (size_t)n * D_.D * 4);
        loadZ(std::string("o/") + zFrom);
        tokenLayer(i, s, z, cc);
        check(("stage s" + std::to_string(i)).c_str(), "s" + std::to_string(i), s, (size_t)n * D_.D, bound);
        if (i < D_.layers - 1 && i % 2 == 0) {
          copy(s, M.f("o/s" + std::to_string(i)), (size_t)n * D_.D * 4);
          pairUpdate(i, s, z);
          check(("stage z" + std::to_string(i)).c_str(), "z" + std::to_string(i), asFloat(z), nn * D_.P, bound);
          zFrom = strdup(("z" + std::to_string(i)).c_str());
        }
      }
      copy(s, M.f("o/s"), (size_t)n * D_.D * 4);
      heads(s, bbCaS, latentsS, logitsS, insertionS);
      check("stage bb_ca", "bbCa", bbCaS, (size_t)n * 3, bound);
      check("stage local_latents", "localLatents", latentsS, (size_t)n * 32, bound);
      check("stage logits", "logits", logitsS, (size_t)n * 147, bound);
      check("stage insertion", "insertion", insertionS, nn, bound);
    }

    // ---- arm 2: the whole trunk, chained -------------------------------------------------------------------
    // 🔴 EVERY CYCLE BUT THE LAST IS RUN WITHOUT COMPARING ANYTHING, because the oracle records the LAST cycle's
    // stages: a two-recycle run writes one s0..s13 and it is the second pass's. So the stages of the earlier
    // cycles are held only by the one they produce - which is why `--recycles=2` is the arm that exercises the
    // cycle projections and the second conditioning at all, and why cuda/rfd4p runs both.
    // 🔴 ITS OWN BUFFERS, ZEROED BY THE ALLOCATOR, AND NOT ARM 1'S ZEROED AGAIN. The first cycle's answer is a
    // property of the state being zero (RMSNorm(0) is 0 and the projections have no bias, so the cycle's input
    // IS the init product), and arm 1 leaves its own s and z in those buffers - which is how this read relRMS
    // 1.9e-1 once, a plausible structure with every stage wrong and nothing in the recycle code touched.
    // 🔴 AND ZEROING THEM WITH `fill` IS NOT THE FIX, WHICH COST AN HOUR TO SEE: `fill` memsets on the HOST
    // when the device looks idle, so it can land while the dispatches that wrote those buffers are still in
    // flight - the GPU's writes then arrive AFTER the zeroing and the state is not zero at all. It showed as a
    // chained trunk that was right seven runs in eight and 7.2e-5 in the eighth, from the same bytes: a
    // difference between runs is a race, never a precision question. A fresh allocation cannot be raced.
    float* sc = allocT<float>((size_t)n * D_.D);
    s = sc;
    if (z.h) z.h = allocT<half>(nn * D_.P); else z.f = allocT<float>(nn * D_.P);
    for (int cycle = 0; cycle < D_.nRecycle; ++cycle) {
      recycleIn(sInit, zInit, s, z);
      // the conditioning of every cycle after the first is built from the `recycleRemaining` features
      const float* cc = c;
      if (cycle > 0) {
        if (!M.has("in/cond1")) die("a %d-recycle input with no in/cond1", D_.nRecycle);
        condC(M.f("in/cond1"), c1, n);
        cc = c1;
      }
      const bool last = cycle + 1 == D_.nRecycle;
      for (int i = 0; i < D_.layers; ++i) {
        tokenLayer(i, s, z, cc);
        if (last) check(("chained s" + std::to_string(i)).c_str(), "s" + std::to_string(i), s, (size_t)n * D_.D, chainedBound);
        if (i < D_.layers - 1 && i % 2 == 0) {
          pairUpdate(i, s, z);
          if (last) check(("chained z" + std::to_string(i)).c_str(), "z" + std::to_string(i), asFloat(z), nn * D_.P, chainedBound);
        }
      }
    }
    float* bbCa = allocT<float>((size_t)n * 3);
    float* latents = allocT<float>((size_t)n * 32);
    float* logits = allocT<float>((size_t)n * 147);
    float* insertion = allocT<float>(nn);
    heads(s, bbCa, latents, logits, insertion);
    check("chained bb_ca", "bbCa", bbCa, (size_t)n * 3, chainedBound);
    check("chained local_latents", "localLatents", latents, (size_t)n * 32, chainedBound);
    check("chained logits", "logits", logits, (size_t)n * 147, chainedBound);
    // 🔴 REPORTED, NOT GATED, AND THE ARM BELOW IS WHAT GATES IT. Measured: fed the reference's own single
    // track this head reads 4.8e-7, and fed ours it reads 1.5e-6 - a dot product over 768 channels of a track
    // carrying 6.3e-7, with the cancellation a difference of large sums brings, so it amplifies its input about
    // 2.4x. Holding the end-to-end number to the per-stage bound would be asking this kernel to be more exact
    // than its input, and widening the bound for every stage to let it through would hide the stages that
    // SHOULD hold. So the amplification is stated and the head is gated where it answers for itself.
    check("chained insertion", "insertion", insertion, nn, 0);
    check("chained s", "s", s, (size_t)n * D_.D, chainedBound);
    check("chained z", "z", asFloat(z), nn * D_.P, chainedBound);
    // ---- the autoencoder's decoder ------------------------------------------------------------------------
    // 🔴 FED THE ORACLE'S OWN LATENTS AND ALPHA CARBONS (`in/zLatent`, `in/caNm`), not the trunk's, for the same
    // reason every stage above is: the decoder is a separate network and its residual should be its own. The
    // trunk's own latents reach it in a generation, where the sampler closes the loop.
    if (M.has("d/feats") && M.has("in/zLatent")) {
      checkDecoderDims();
      const int na = attnCount;
      float* feats = allocT<float>((size_t)n * DD_.feat);
      decoderFeatures(M.hostF("in/caNm"), M.hostF("in/zLatent"), M.hostI("in/tokenType"), M.hostI("in/decSeq"),
                      M.hostF("in/mask"), feats);
      check("decoder feats", "d/feats", feats, (size_t)n * DD_.feat, decBound);

      float* sd = allocT<float>((size_t)na * DD_.D);
      float* sInitD = allocT<float>((size_t)na * DD_.D);
      float* perLayer = allocT<float>((size_t)DD_.layers * na * DD_.D);
      float* ada0 = allocT<float>((size_t)na * DD_.D);
      float* attn0 = allocT<float>((size_t)na * DD_.D);
      float* dLogits = allocT<float>((size_t)na * DD_.types);
      float* dCoors = allocT<float>((size_t)na * 111);
      float* dPresence = allocT<float>((size_t)na * 37);
      DEC_TAP_ADA = ada0;
      DEC_TAP_ATTN = attn0;
      decode(feats, attnIdx, na, sd, dLogits, dCoors, dPresence, sInitD, perLayer);
      DEC_TAP_ADA = DEC_TAP_ATTN = nullptr;

      // 🔴 SCATTERED BACK BEFORE COMPARING, because the oracle's rows are the INPUT's and this port's are the
      // ATTENDED ones. They coincide whenever nothing is in context - which is every case this preset has - and
      // a gate that only ever ran there would not notice the day they stop.
      // 🔴 AND EVERY DECODER LAYER FROM THE ORACLE'S OWN INPUT, for the reason the trunk has the same arm: fed
      // what the stage before it computed, a layer's residual is its own plus everything above it, and this
      // stack amplifies. Measured here, it is what separates the two.
      {
        float* sl = scratch<float>("gate.sl", (size_t)na * DD_.D);
        float* cz = scratch<float>("gate.cz", (size_t)na * DD_.C);
        fill(cz, 0, (size_t)na * DD_.C * 4);
        for (int l = 0; l < DD_.layers; ++l) {
          const std::string from = l == 0 ? "d/sInit" : "d/s" + std::to_string(l - 1);
          gatherRows(M.f(from), sl, attnIdx, na, DD_.D);
          decoderLayerOne(l, sl, cz, na);
          float* out = scratch<float>("gate.scatter2", (size_t)n * DD_.D);
          fill(out, 0, (size_t)n * DD_.D * 4);
          scatterRows(sl, out, attnIdx, na, DD_.D);
          check(("decoder stage s" + std::to_string(l)).c_str(), "d/s" + std::to_string(l), out,
                (size_t)n * DD_.D, decBound);
        }
      }

      float* full = allocT<float>((size_t)n * DD_.D);
      auto scattered = [&](const float* src, int width) {
        float* out = scratch<float>("gate.scatter", (size_t)n * width);
        fill(out, 0, (size_t)n * width * 4);
        scatterRows(src, out, attnIdx, na, width);
        return out;
      };
      (void)full;
      check("decoder s_init", "d/sInit", scattered(sInitD, DD_.D), (size_t)n * DD_.D, decBound);
      // layer 0's two internals: its input is the exact s_init, so these are per-stage numbers too
      check("decoder ada0", "d/ada0", scattered(ada0, DD_.D), (size_t)n * DD_.D, decBound);
      check("decoder attn0", "d/attn0", scattered(attn0, DD_.D), (size_t)n * DD_.D, decChained);
      for (int l = 0; l < DD_.layers; ++l)
        check(("decoder chained s" + std::to_string(l)).c_str(), "d/s" + std::to_string(l),
              scattered(perLayer + (size_t)l * na * DD_.D, DD_.D), (size_t)n * DD_.D, decChained);
      check("decoder chained logits", "d/logits", scattered(dLogits, DD_.types), (size_t)n * DD_.types, decChained);
      check("decoder chained coors", "d/coors", scattered(dCoors, 111), (size_t)n * 111, decChained);
      check("decoder chained presence", "d/presence", scattered(dPresence, 37), (size_t)n * 37, decChained);
    }

    // ---- the autoencoder's encoder ------------------------------------------------------------------------
    if (M.has("e/feats")) {
      checkEncoderDims();
      const int na = attnCount;
      float* ef = allocT<float>((size_t)n * ED_.feat);
      encoderFeatures(n, M.hostF("e/coorsNm"), M.hostF("e/atomMask"), M.hostF("e/chainIid"), M.hostF("e/bondCounts"),
                      M.hostI("e/element"), M.hostI("e/charge"), ef);
      check("encoder feats", "e/feats", ef, (size_t)n * ED_.feat, bound);
      float* mean = allocT<float>((size_t)na * ED_.latent);
      float* logScale = allocT<float>((size_t)na * ED_.latent);
      encode(ef, attnIdx, na, mean, logScale);
      // 🔴 THE LATENTS ARE EMITTED ROWS ONLY, so the comparison is against the oracle's own emit mask and not
      // the attention's: `emit = attn & supports_local_latent`, and a token that supports no latent has zeros
      // there in the reference. Scattering by `emit` is what makes the two tables the same table.
      const float* emit = M.hostF("e/emit");
      std::vector<int> eidx;
      for (int i = 0; i < n; ++i) if (emit[i] != 0) eidx.push_back(i);
      int* ei = uploadNew(eidx.data(), eidx.size());
      auto scatterEmitC = [&](const float* src, const char* name) {
        float* out = scratch<float>(std::string("gate.") + name, (size_t)n * ED_.latent);
        fill(out, 0, (size_t)n * ED_.latent * 4);
        // the emitted rows are a subset of the attended ones, in the same order
        float* gathered = scratch<float>("gate.eg", eidx.size() * ED_.latent);
        std::vector<int> within;
        {
          const float* am = M.hostF("in/attnMask");
          int k = 0;
          for (int i = 0; i < n; ++i) { if (am[i] == 0) continue; if (emit[i] != 0) within.push_back(k); ++k; }
        }
        int* wi = uploadNew(within.data(), within.size());
        gatherRows(src, gathered, wi, within.size(), ED_.latent);
        scatterRows(gathered, out, ei, eidx.size(), ED_.latent);
        return out;
      };
      // 🔴 THE ENCODER'S OUTPUT IS A TWELVE-LAYER CHAIN AND IS HELD AS ONE. There are no per-layer encoder
      // stages in the dump, and there do not need to be: its layers ARE the decoder's (one `aeLayer`, the same
      // prefix away), which the decoder's per-stage arm gates layer by layer. What is left for this arm to say
      // is whether the chain and the head are wired right, at the chained level the decoder measures.
      check("encoder mean", "e/mean", scatterEmitC(mean, "mean"), (size_t)n * ED_.latent, decChained);
      check("encoder logScale", "e/logScale", scatterEmitC(logScale, "ls"), (size_t)n * ED_.latent, decChained);
      // the deterministic latent IS the mean, which is what `e/zLatent` holds in this arm
      check("encoder zLatent", "e/zLatent", scatterEmitC(mean, "zl"), (size_t)n * ED_.latent, decChained);
    }

    printf("%s so far (%.2f s)\n", FAILED ? "FAILED" : "every stage in band", now() - t0);
  }

  // ---- the flow sampler ---------------------------------------------------------------------------------
  if (!genDir.empty()) {
    checkDecoderDims();
    // only where the reference's own trajectory was exported beside the generation: without it there is nothing
    // to hold the steps to, and printing "(not in the oracle)" sixty times says nothing
    if (M.has("gen/x0Bb")) GATE_CHECK = checkFn;
    // 🔴 THE TRAJECTORY'S BOUND IS NOT A STAGE'S. Every step feeds the next, so what a stage holds to 1e-6
    // arrives here multiplied; the per-step record is what says whether that is amplification (the numbers grow
    // smoothly) or a wiring fault (one step jumps).
    GATE_TRAJ_BOUND = HALF ? 1e-3 : 1e-4;
    // 🔴 THE NOISED STATE'S OWN BOUND IN HALF, from the arithmetic and then from the designs. A step's score
    // divides by (1 - t), which at the LAST step is 0.091 over a two-step run - the worst-conditioned point a
    // trajectory has - and there the half arm reads 1.5e-2 where x1, which divides by nothing, reads 1.4e-5.
    // What that is worth was measured end to end rather than argued: at sixty steps on one seed the two
    // arithmetics' designs are 0.022 A RMS apart (worst 0.059) with 53 of 54 residues identical, against 13 to
    // 23 A between seeds. 3e-2 is that with room; the `--f32` arm holds every row at 1e-4.
    GATE_TRAJ_XT_BOUND = HALF ? 3e-2 : 1e-4;
    const int ran = generate(genDir, outPath, designs, seed.empty() ? 0 : strtoull(seed.c_str(), nullptr, 10),
                             metricsPath, dumpSample, refold);
    printf("  the sampler ran %d steps\n", ran);
    GATE_CHECK = nullptr;
  }


  if (!oracle.empty() || !genDir.empty()) {
    if (FAILED) printf("FAILED (%.2f s)\n", now() - t0);
    else if (CHECKED) printf("all green: %d comparisons (%.2f s)\n", CHECKED, now() - t0);
    else {
      // Nothing was compared, so there is nothing to call green. `--job=` alone folds and writes; the arms need
      // a reference built by `cuda/rfd4p/export_input.mjs --oracle= [--generate --gen-oracle]`.
      printf("folded; NOTHING COMPARED - no oracle on disk, so no arm ran (%.2f s)\n", now() - t0);
    }
    return FAILED ? 1 : 0;
  }
  printf("the trunk's init products built in %.2f s (no --oracle= to hold them to)\n", now() - t0);
  return 0;
}

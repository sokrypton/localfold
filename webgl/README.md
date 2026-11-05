# LocalFold on WebGL2

The fallback backend, for the browsers and machines that have no WebGPU. `webgpu/` is the page's
backend and needs Chrome/Edge 113+ or Safari 26+ - which the page says in those words
(`web/design.js`: "WebGPU unavailable — this page needs Chrome or Edge 113+, or Safari 26+, to
fold"). WebGL2 has been in Safari since 15 and in Chrome since 56, so it reaches machines and
operating systems that the WebGPU path refuses outright.

🔴 **THIS TREE IS TWO KERNELS AND A RUNTIME, AND NOTHING SELECTS IT.** It is not a working
backend and this file is not a plan to pretend otherwise. What is here is the spike that answered
*can WebGL2 do this work at all* - the two kernels that decide the question, moved out of
`webgpu/kernels/` where they sat with `webgl2` in their names and no importer - plus the texture
layout and the limit arithmetic they shared by copy. The honest summary is in the sections below:
the numerics are **now checked and correct**, the speed is **better than anyone here expected**,
and the shapes are **worse**.

🔴 **AND THEY WERE UNGATED FOR AS LONG AS THEY EXISTED, WHICH IS WHY THE FIRST THING THIS TREE
GOT WAS A DIFFERENTIAL.** AGENTS.md requires every GPU kernel to have "a deterministic differential
test against an independent reference implementation" and these two were the standing exception:
written as spikes, run by hand once, quoted in two documents ever since. Nothing could have told a
reader whether they still computed anything. `tools/gpu/check-webgl-kernels.js` is that gate now -
see **Verified** below for what it covers and what it does not.

```
webgl/
  index.js          the backend's one door, as webgpu/index.js is for that tree
  runtime/
    context.js      a WebGL2 context used as a compute device; the texture layout every kernel
                    shares (a power-of-two row width, four floats to an RGBA32F texel), the
                    full-framebuffer triangle that stands in for a dispatch, program linking
    limits.js       what a context MUST provide, and the ceilings that follow from what it
                    reports; `pairTokenCeiling` bounds a 2D operand, `tensorCapacity` in
                    storage.js bounds a linear one, and they differ eightfold
    storage.js      tensors in textures, addressed LINEARLY - allocate, upload, read, release,
                    every allocation counted so peak memory is measurable
    execution.js    what stands in for a dispatch: one draw, one fragment per output texel,
                    banded when a viewport cannot cover the target; the shared GLSL prelude
                    and the program cache, keyed by source
    elementwise.js  residuals: add, multiply, scale, and the per-row mask broadcast
    transpose.js    swapping the two leading axes - which is what MSA column attention is
  kernels/
    triangle.js     the triangle contraction (22% of an Evoformer block at 221 residues)
    attention.js    the original spike's flash attention, 32-channel heads, as a benchmark
    attention-flash.js          gated attention for ANY head width: `min(D/4, MAX_DRAW_BUFFERS)`
                                planes an invocation, drawn more than once when a head is too
                                wide, plus the gather back into one interleaved tensor
    attention-block.js          the whole block - LayerNorm, q/k/v/gate projections, the flash
                                core, the output projection; what an Evoformer layer calls
    pair-bias.js                the pair track's logit bias, head-major: the only place the
                                MSA and pair tracks meet before the outer product mean
    transition.js               LayerNorm, expand with a ReLU, contract back
    outer-product-mean.js       the MSA's path INTO the pair track, in two draws
    layer-norm.js   row-wise LayerNorm, about ten times a block
    linear.js       the dense projection, with bias and activation folded in - where the FLOPs are
    triangle-contract.js        the contraction, one source shared by the stage and the benchmark
    triangle-multiplication.js  the whole operation, nine draws: LayerNorm, four projections,
                                two gates, the mask, the contraction, the output gate
    global-attention.js         AF2's global column attention: one query a residue, the masked
                                mean of its column, with single-headed keys and values
  af2/
    evoformer/block.js          a whole Evoformer block and the extra-MSA block, nine sub-layers
                                each, sharing the six that make up the pair half
    evoformer/stack.js          the loop over blocks, for both kinds
    evoformer/input-embedder.js the MSA, extra-MSA and pair embeddings, and the recycle
    evoformer/template.js       the template stack: one Linear over 88 geometry columns, the pair
                                blocks, and a pointwise attention that is two projections
    structure/initialize.js     the single representation and the identity frame
    structure/sidechain.js      seven torsions a residue, as normalised (x, y) pairs
    structure/geometry.js       atom coordinates from frames and torsions - atom14 and atom37
    structure/ipa.js            invariant point attention: the one attention whose logits are
                                a distance - eight draws plus five projections
    structure/iteration.js      the IPA update folded in, a transition, and the frame advanced
    structure/module.js         the four stages composed, eight iterations, one hoisted pair norm
  heads/
    distogram.js                P(d < 8 A) for every pair - the contact map the page draws
    confidence.js               pLDDT, PAE and the pTM term
```

## AF2, which is what this tree is being built for

`webgpu/`'s AF2 path is **75 `@compute` entry points** (32 in `webgpu/af2/`, 34 in
`webgpu/kernels/`, 2 in `webgpu/heads/`, 1 in `webgpu/input/`, 6 in `webgpu/runtime/`) over about
18,000 lines. The primitives ported so far are deliberately the load-bearing ones rather than the
easy ones: the triangle contraction and flash attention are 80% of an Evoformer block's GPU time
between them, and LayerNorm, the dense projection, the residual add and the row mask are what the
other 70 kernels are mostly made of.

**Triangle multiplication is complete and checked in both directions** - the first whole AlphaFold
operation here rather than a primitive. It is nine draws where the WGSL side fuses into fewer, and
every intermediate is released as its last reader finishes, so a 400-token block does not hold nine
pair tensors at once.

**Attention is generalised off the fixed 32-channel head**, which was the other thing blocking
every model: `targets = min(D/4, MAX_DRAW_BUFFERS)` is derived, a head too wide for one draw is
drawn in several (D = 64 takes two, each recomputing its own dot products, and the redundancy is
reported rather than hidden), and the `1/sqrt(head_dim)` scaling is now explicit - the original
applied none, so it was correct only for a caller that had pre-scaled its queries and nothing said
so. Checked at D = 4, 8, 16, 32 and 64, with and without a pair bias, with both mask layouts, and
against the kernel it replaces at that kernel's own convention.

**The whole attention block is done and cross-checked against the WGSL one** at 2.3e-7, which is
the comparison that matters: `AttentionGpu` is the block, so the flash core alone had nothing on
that side to be measured against. It confirmed four conventions rather than assuming them - weights
`[in][out]` (the triangle's are `[out][in]`, same model), q/k/v with no bias against a gate with
one, the `1/sqrt(head_dim)` scale on the logit here against the query there, and the plane gather.

**AF2's MSA row attention with pair bias is complete and verified end to end** - the attention
block driven with a bias projected out of a separate pair tensor, which is the only place the two
tracks meet before the outer product mean sends information the other way. So are the transition
and the outer product mean, each cross-checked against its WGSL twin and an f64 reference.

**MSA column attention works too** - the same block over the other axis, which here is an explicit
transpose draw plus the flash core's mask flag, where the WGSL side folds the transpose into its
normalise. So an Evoformer block's MSA half is complete: row attention with pair bias, column
attention, and the transition.

**Both whole blocks match**, at every arm's own measured floor:

| arm | relRMS | condition | floor-derived bar |
|---|---:|---:|---:|
| main block, msa | 5.8e-8 | 0.94 | 3.4e-7 |
| main block, pair | 9.1e-8 | 0.50 | 2.0e-7 |
| extra block, msa | 6.0e-8 | 0.97 | 3.5e-7 |
| extra block, pair | 1.2e-6 | 198 | 7.1e-5 |

🔴 **AND THE BAR IS EACH ARM'S CONDITION NUMBER TIMES AN f32 STEP, BECAUSE ONE CONSTANT CANNOT
SERVE BOTH BLOCKS.** The main block is CONTRACTIVE - a relative nudge comes out smaller than it went
in - while the extra block's pair amplifies by about **200**, because global attention divides by a
masked column count that can be small. A bar tight enough for the main block is unmeetable for the
extra one; a bar loose enough for the extra block would pass a real 1e-5 error in the main one. The
gate measures the ratio with a 1e-5 nudge and derives both. (Its first version used a single f32
step and the extra block's pair came back EXACTLY unchanged - below its own ULP everywhere. A floor
of zero is not a floor.)

🔴 **AND THAT GATE FOUND SOMETHING NO OPERATION GATE COULD: THE SHIPPED AF2 BLOCK KEEPS ITS
INTERMEDIATES IN f16 WHERE NO SWITCH REACHES.** `normalizedStorage = "f16"` in the attention,
`{input: "f16", ..., output: "f16"}` asked of its flash kernel, and
`hiddenStorage = hiddenChannels % 4 === 0 ? "f16" : "f32"` in the transition - none of it behind
`halfPrecisionAvailable` or a tuning knob. The STANDALONE classes every operation gate compares
against default to f32, so each of those gates was passing at 2e-7 on a configuration AF2 never
runs: CLAUDE.md's "checking a configuration nothing runs", from the other side. This tree now
reproduces all three with an exact `packHalf2x16` round trip, which buys it nothing (its tensors are
RGBA32F either way) and is done because a fallback whose folds cannot be compared with the primary
path's is not much of a fallback. Worth 5.8e-8 against 9.5e-5 on the MSA half.

**Both block kinds run**: the main Evoformer block and the extra-MSA block, which differ only in
their column attention - the extra stack runs thousands of alignment rows, so attending over the
sequences pairwise is unaffordable and its query is a masked column mean instead. Global attention's
four conventions are confirmed at the floor (`extra.msa` at 6.0e-8), and that block is the ONLY
cross-check it can have, because `encodeGlobalAttention` is not exposed as a class.

🔴 **A 6e-5 PAIR GAP WAS CLOSED BY TWO BISECTS AND NO MORE READING - AND READING THE SOURCE IS
WHAT GOT IT WRONG.** `pairBiasStorage = options.pairBias?.source === "separate" ? "f32" :
normalizedStorage` reads as "a bias projected from the normalised input is stored f16", so this port
rounded it; both pair arms sat at ~6e-5 with each triangle attention contributing about 3e-5.

`--disable=` zeroes a sub-layer's output projection on BOTH sides, which makes that residual an
identity without either implementation knowing: with only the outer product mean live the pair arm
is 4.4e-8, with both triangle multiplications 6.8e-8, and it jumps to 3.5e-5 the moment the first
triangle attention is let back in. Then `--tri-att-f32=` sweeps the four rounding points inside
them, and exactly one lands: **the bias in f32 gives 9.2e-8**, where `attended` gives 5.3e-5,
`normalised` 9.2e-5 and `projections` 8.4e-5. So the f16 choices are per TENSOR, not per group, and
`DEFAULT_PART` in `kernels/attention-block.js` records each one as a measurement.

Both handles are kept. A gate that can bisect itself is worth more than one that only reports a
number.

🔴 **AND THE STACK HAS A DIVERGENCE THAT IS RECORDED AND NOT YET EXPLAINED.** Both whole
blocks match at their floors, and so does a ONE-block stack (9.3e-8). Over two blocks the main
stack's MSA reaches **2.5e-5** and over three **5.4e-5**.
`tools/gpu/check-webgl-evoformer-stack.js` carries the evidence and the handles that produced it:

- it is **MSA column attention** - `--disable=msa-column` takes the two-block arm from 2.5e-5 to
  9.0e-8, and no other sub-layer moves it
- column attention **alone** is innocent: with everything else silenced it is 4.8e-8 at one block
  and 7.3e-8 at three, so the kernel is right and what it receives in composition is what matters
- it is **not the loop**: `--identity-after=1` zeroes the second block's output projections on BOTH
  sides, making it an identity, and reproduces the one-block numbers exactly
- it is **not amplification**, which is the surprising part: a one-f32-step jitter at the block
  BOUNDARY - where the two implementations actually differ - moves it by 9.0e-8. The extra stack's
  arms ARE amplification and sit at their floors; the main stack's are not
- it is **weight-dependent**: one weight set is at the floor on all four arms where another is not

🔴 **AND THE EXPLANATION IS f16 BOUNDARY STRADDLING, WHICH MAKES THE FLOOR DISCRETE RATHER THAN
SMALL.** Two f32 values 1e-7 apart that fall either side of an f16 boundary round **5e-4** apart, so
a chain that quantises cannot be bounded by perturbing its input: the floor is set by how many
elements straddle. The arithmetic matches at both ends - this port in f32 against their f16 should
differ by the full quantisation error of ~1.4e-4 and measures 9.5e-5; both in f16 with f32 inputs
1e-7 apart gives `sqrt(P) * h` per rounding point, about 7e-6, and six points a block gives ~1.7e-5
against 2.5e-5 measured over two blocks and 5.4e-5 over three, growing linearly in the block count.

`execution.halfDither` settles it - a relative dither applied INSIDE every f16 rounding. At one f32
step it reads **exactly 0** on three of four arms and 3.7e-6 on the fourth: a jump, not a gradient,
which is what a Bernoulli straddle predicts and what no smooth floor model can produce. With ~2000
rounded elements and a straddle probability of 2.4e-4, one probe run expects less than one crossing
and usually sees none, while the real divergence accumulates them across every block.

So both implementations are correct and the floor is simply higher than any single-perturbation
probe reports.

The gate asserts a 2e-4 allowance - still 2.5x tighter than f16's own precision - and REPORTS every
arm above its own measured floor, so neither a pass nor a failure can be read off it by mistake.
`--strict` fails on them.

**The contact map head is done**, at 1.0e-7 to 1.7e-7 against `shared/heads/distogram.js` - the
page's own host implementation, which keeps f64 accumulators and was written before any of this, so
it is a second implementation rather than a transcription. It pins three conventions that would each
produce a plausible map: the pair is symmetrised, the bias is therefore counted twice, and the bin
straddling the threshold is excluded.

**The confidence head is done too** - pLDDT at 1.1e-7, PAE **bit-identical**, the pTM term at
2.4e-7 against the shared host implementation, and pTM itself exact against the WGSL head and 1.4e-9
against the host reduction. It pins two bin-centre rules that are different and neither obvious:
pLDDT's are `(bin + 0.5) / bins * 100` while PAE's come from the model's own breaks as
`breaks[bin] + step / 2`, with the last bin CLOSED rather than open.

🔴 **AND AF2's 50 pLDDT BINS FORCED A SECOND LINEAR KERNEL.** A width that is not a multiple
of four means a texel straddles two rows - row r occupies elements `[50r, 50r + 50)` - so a
fragment's four consecutive outputs can belong to two rows with two different input vectors. The
aligned kernel shares one input sweep across its four outputs and cannot. `linear` compiles a ragged
variant that resolves each component separately: four times the reads, on a projection whose whole
output is `rows * 50` floats, and the alternative was refusing a shape the model actually has.

**The structure module is under way**: its starting point (the single representation and the
identity frame, `single` bit-exact) and its sidechain torsion head (1.0e-7 to 1.1e-7) are done and
cross-checked. Both pin conventions a fold could not reveal - an affine is a quaternion THEN a
translation with the identity at `(1, 0, 0, 0)`, the torsion head reads the INITIAL normalised
representation rather than the activations, and the angle normalisation guards at 1e-12 so a zero
torsion comes out `(0, 0)` rather than NaN.

**Atom geometry is done and BIT-IDENTICAL** - atom14 and atom37 both at relRMS 0 against the WGSL
stage, which matters because this stage is almost all convention and almost no arithmetic. Five
choices decide whether the output is a structure or a plausible cloud, and every one of them
produces finite coordinates when wrong: a default frame is a 4x4 ROW-MAJOR matrix whose translation
is the fourth column; the quaternion is `(w, x, y, z)` and is used UNNORMALISED; the backbone
translation carries a factor of **10**; only groups 5, 6 and 7 chain; and the torsion mixes columns
1 and 2 with `(sine, cosine)` in that order.

**Invariant point attention matches at 2.14e-7** - the hardest kernel group in AF2 and the stage
with the most ways to be plausibly wrong, since every one of its conventions produces a finite
update of the right shape when broken:

- the point term is **subtracted** and is a **squared** distance, scaled by a **softplus** of a
  trained per-head parameter, so the weight is positive however it trained and the logit falls as
  the points separate
- the three scale factors are `sqrt(1/(3*scalarQk))`, `sqrt(1/(3*pointQk*4.5))` and `sqrt(1/3)`,
  and the **4.5** cannot be guessed
- the local point layout is `[residue][component][head][point]` - the three components a plane
  apart, not adjacent - while the global one is `[residue][head][point][xyz]`
- the point feature is summed in global space and rotated back into the **query's** frame, which is
  what makes it invariant, with a fourth plane holding the vector's norm guarded at 1e-8

Its three feature blocks get their own tensors and a gather assembles the row, because they occupy
disjoint ranges whose boundaries are not texel-aligned.

**AF2's structure module is complete** - all five stages ported and cross-checked: the starting
point, invariant point attention, the iteration that folds its update in and advances the frame, the
sidechain torsion head, and atom geometry.

🔴 **AND THE FRAME UPDATE IS THE ERROR THAT WOULD HAVE DEGRADED A FOLD RATHER THAN BREAKING
IT.** The update's first three floats are a VECTOR, and the increment is the product of `q` with
`(1, v)` less `q` - `(-dot(q.yzw, v), q.x * v + cross(q.yzw, v))` - added and then NORMALISED.
Reading those three as a rotation, or skipping the normalise, gives frames that drift off the unit
sphere over eight iterations: a structure that gets worse with length rather than one that fails.
The gate asserts the quaternion IS a unit as well as comparing it, because a comparison would not
catch both sides drifting the same way.

**The input embedder is done** - msa 6.5e-8, extra msa bit-exact, pair 6.4e-8 - with four
conventions pinned that each produce a plausible embedding when wrong: the recycled MSA enters on
row ZERO only; the pseudo-beta is CB except for glycine, which uses CA; a distance below the first
bin's lower edge lands in NO bin while the last bin's upper edge is `maxBin + binWidth`; and the
relative position is CLAMPED, so every separation past 32 shares one bin. Its arms plant a glycine
and an index jump of 100, because neither branch is reached by a plain sequence.

🔴 **AND THE CROSS-CHECK CAUGHT A MISSING STAGE BY REFUSING TO RUN.** Both recycled tensors
are LayerNormalised before being added - four weights with nowhere else to go - and this port did
not do it. `packWeights` reads all eighteen tensors and threw on the four that were absent. Adding
the recycled tensors raw is a plausible embedding whose scale drifts with every pass: the kind of
error that degrades a long fold and leaves a short one looking fine, and one that three kernels each
looking complete would never reveal.

**The template stack is done** - the 88-column `embedding2d` at **4.6e-8** against an f64
reference, and the whole stage at its measured floor against `QueryOnlyTemplateGpu` on five arms.
Three conventions are pinned, and none of them is reached by a plain template: the whole 88 is
multiplied by the BACKBONE mask rather than each feature by its own, so the monomer's distogram
survives at a pair whose pseudo-beta is missing - the opposite of what AF3 and the multimer do;
the bias is OUTSIDE that mask, so a masked template leaves the bias to run the whole stack and both
projections rather than producing a zero update; and `use_template_unit_vector` is false in every
shipped monomer config, which this port bakes into the source and gates as its own arm rather than
leaving as an untested branch.

🔴 **AND A TEMPLATE PAIR BLOCK IS NOT THE EVOFORMER'S PAIR HALF WITH THE OUTER PRODUCT MEAN
DELETED.** It runs both triangle ATTENTIONS first and the two multiplications after, where the
Evoformer block runs the multiplications first. Every sub-layer reads what the previous one wrote,
so the two orders compute different tensors from the same weights and nothing at the operation
level can see it. Swapping them in this port measures **relRMS 0.47 to 0.88** against a bar of
6e-4, which is what the gate was built for; two blocks rather than one, because a single block
cannot separate an error in the block from an error in carrying its output into the next.

🔴 **AND THE FLOOR HERE IS f16 STRADDLING, NOT f32 AMPLIFICATION - A CONDITION NUMBER IS THE WRONG
INSTRUMENT AND SAID SO CONFIDENTLY.** Nudging the embedding bias gives a tidy ratio of 0.42 on the
no-template arm, i.e. "contractive", while two blocks disagree 27x more than one: a bias nudge is
UNIFORM over the pairs and a rounding difference is not, and with no template every pair carries the
same value, so the uniform mode is all such a probe can excite. `execution.halfDither` is the
instrument that fits, the same one the stack gate had to reach for, and the no-template arm then
sits at **7.98e-6 against a measured floor of 7.9e-6**. That the floor is f16 at all is measured
rather than assumed: `--tri-att-f32` lifts this port's rounding in the triangle attentions and the
arm gets eighty times WORSE, 6.8e-6 to 5.4e-4, so the WGSL template block stores its attentions in
f16 exactly as the main block does. `--tri-f32` moves nothing, which is the control.

**The structure module composes** - all six outputs against `StructureModuleGpu` at their measured
floors (coordinates 3.3e-7, representation 3.0e-7, frames 1.6e-7, angles 4-6e-7), which is the
first arm that covers the structure side's three seams. None of the five per-stage gates can see
them, because each hands its stage an input it built itself: the pair LayerNorm is HOISTED out of
the eight iterations, the sidechain head reads the FIRST representation and the eighth rather than
the eighth twice, and each iteration's attention reads the frames the previous one wrote. Handing
the sidechain head the current activations twice measures **relRMS 0.50 to 0.77** on four arms
while the representation and the frames stay green at 3e-7 - which is why there are six arms and
not one.

🔴 **AND THE AFFINE UPDATE'S WEIGHT HAS TO BE SMALL FOR THE GATE TO MEAN ANYTHING.** At the usual
`1/sqrt(channels)` scale the update winds the frames around several times over eight iterations,
and the comparison then measures how two implementations diverge along a chaotic trajectory rather
than whether they compute the same thing. `--affine-scale=` is the handle; the checkpoint's own
weight is small for the same reason.

🔴 **AND ITS FIRST CONTROL WAS VACUOUS FOR A REASON THIS FILE ALREADY RECORDS.** `v * 1.01 + 0.01`
over the whole pair reported "0 of 6 arms noticed": the pair's ONLY path into this module is
`normalisePair`, and a whole-row affine change is invisible to a LayerNorm - the scale cancels in
the variance and the shift cancels in the mean. Half of each row's channels is not affine over the
row and all six arms go red. **A control has to be checked against the FIRST operation its tensor
meets**, which is now the second time in this campaign that cost a run.

Still to come in AF2: the multimer variant, and reading a weight bundle. After
that: AF3's pairformer and diffusion, ESMFold2, bioemu2.

Every operation so far is checked against TWO references - an f64 CPU one written from the
operation, and the WGSL kernel run in the same page - because a single reference cannot say which
side is wrong. That has already paid for itself twice: the triangle contraction's incoming
direction was the pair-transpose of itself and only the WGSL arm could see it, and the attention
block read 1e-3 against WGSL while reading 2.5e-7 against f64, which was an unpinned precision axis
on the WGSL side rather than a defect here. Nothing is wired to a weight
bundle yet - every kernel takes its operands as plain arrays, and reading
`model-af2-monomer-int5` is its own stage with its own differential, because a transposed weight is
a plausible-looking wrong answer.

## No compute shaders, and everything follows from that

The work is done by a **fragment** shader: one draw covers a rectangle of output texels, every
fragment computes one output element, and the "dispatch" is a triangle that covers the
framebuffer. There are no storage buffers, so tensors live in textures and every linear index is
decoded into a texel coordinate. There is no workgroup shared memory, so the tiling the WGSL
kernels use - `var<workgroup>` staging with two barriers a step - has no equivalent at all; these
read straight from texture memory and lean on the texture cache instead.

That is the honest shape of the comparison: not the same algorithm in two languages, but the same
maths under two different sets of constraints.

🔴 **AND IT CUTS BOTH WAYS, WHICH IS WHY THE SPIKE WAS WORTH IT.** Writing the attention kernel
without barriers is **what exposed how much the WGSL kernel was paying for them - the 11.4x**. A
fragment has no one to synchronise with: the dot product is a loop in registers and the online
softmax's running maximum and sum never leave them, where the compute kernel spent a 32-lane tree
reduction with a barrier at every step. On a fair fight the two WebGL2 kernels are about **1.4x
behind on the contraction and about 1.55x ahead on attention**, computing the same numbers.

## The speed, which has been wrong three times

From docs/A100.md, measured on an A100 through Chrome. Each earlier answer was wrong by measuring
the easier thing: **165x** (jax-js's WebGL backend, which its own authors offer "on a best-effort
basis"), then **4x** (a naive fragment GEMM), then **3.3x** (a properly blocked one, with a tidy
proof that 21.3 flops a fetch was "the ceiling of the API" - sound proof, wrong premise: it is the
ceiling of the FRAGMENT path).

| | TFLOP/s | of 18.1 |
|---|---:|---:|
| LocalFold, WebGPU compute | 13.22 | 73% |
| **WebGL2, transform feedback** | **8.64** | **47.8%** |
| jax-js, WebGPU | 4.29 | 23.7% |
| WebGL2, fragment + MRT | 3.99 | 22.0% |
| WebGL2, naive fragment | 1.01 | 5.6% |
| jax-js, WebGL | 0.08 | 0.4% |

The vertex path is 2.2x the fragment path and the whole difference is **output budget** - a
fragment writes `MAX_DRAW_BUFFERS` x RGBA = 32 floats, a vertex shader with rasterisation
discarded writes up to `MAX_VERTEX_OUTPUT_COMPONENTS`, and more outputs an invocation is more
arithmetic per fetch. This is the vertex shader used as a compute kernel, which is what people did
before compute shaders existed. See `vertexOutputFloats` in `runtime/limits.js` for why the
advertised limit is not the real one.

🔴 **AND EVERY NUMBER ABOVE IS AN A100's, WHICH IS NOT WHO THIS BACKEND IS FOR.** The machines
without WebGPU are older machines, and no speed here has been measured on one. Old drivers are
exactly where transform feedback gets flaky, so the 47.8% is the figure most at risk of not
transferring - and it is the figure the whole case rests on. The two kernels have now been run on a
second machine, and it is not an old one either: an M5 through ANGLE's Metal backend.

## Verified

`node tools/gpu-chrome.mjs tools/gpu/check-webgl-kernels.js` - **18 arms, all green**, on an M5
through ANGLE Metal. Both kernels against CPU references written from the shaders' own index
arithmetic and sharing no code with them: the contraction in both directions over four shapes
(L=8/13/24/31, CH=4/8/32/12) at **1.0e-7**, gated attention with and without a pair bias over three
(batch, queries, heads) at **1.2e-7**, and the `channel-major` relayout **bit-exact** against
`channel-minor`, which is the sharp arm - a relayout reorders no sum, so a 1e-6 bar there would pass
an off-by-a-group index.

`--falsify` puts one element 1% out in the operand handed to the kernel and **every one of the 18
arms goes red**. That matters because the first version of the flag perturbed the SHARED input,
which moves both sides equally: it reported "only 4 of 18 arms noticed", the 4 being the layout arms
that compare two kernel runs and were already one-sided. A control that feeds both arms is this
repository's "comparing something with itself" trap wearing the clothes of a control.

`node tools/gpu-chrome.mjs tools/gpu/check-webgl-triangle.js` - **12 arms**, the whole operation
against TWO references over four shapes: the f64 CPU reference, and the WGSL kernel driven in the
same page at its own f32/two-pass defaults. 2e-7 against the first, 3-5e-7 against the second.

🔴 **AND THAT SECOND REFERENCE IS WHY THE INCOMING DIRECTION IS NOT STILL WRONG.** The
contraction's incoming arm kept `a` on the `i` side in both directions, where the WGSL kernel -
gated against OpenFold and AlphaFold fixtures - swaps the operands: `sum_k b[k,i] a[k,j]`, not
`sum_k a[k,i] b[k,j]`. That is the pair-TRANSPOSE of the right answer and it measured **relRMS
1.3**, no signal at all, on all four shapes. It survived because the CPU reference it had been
checked against was written FROM the shader's own index arithmetic, so it encoded the same
convention and agreed to 1e-7 while both were wrong. The operation's own CPU reference covers
outgoing only, so nothing else here could have seen it. **Independence has to mean a second
implementation of the operation, not a second transcription of the code.**

🔴 **AND PINNING THE WGSL SIDE TOOK FOUR TRIES, WHICH IS WORTH KNOWING BEFORE WRITING THE NEXT
CROSS-CHECK.** `AttentionGpu` has FOUR precision axes - `flashPrecision`, `projectPrecision`,
`outputPrecision` and `storage` - all defaulting to `"auto"`, and `"auto"` on a device with
`shader-f16` is f16, a 1e-3 floor. Worse, `storage` is READ as `{input, value, output}` and
REPORTED back as `{projected, value, weighted}`, so passing the reported spelling sets nothing and
raises nothing: the run answered `projected: "f32"` - which it already was - and the number did not
move. The tell throughout was that two references disagreed by four orders of magnitude about the
same tensor (2.5e-7 against f64, 1e-3 against WGSL); when that happens the instrument is wrong, not
the kernel. Every cross-API arm here prints the `storage` and `shaders` keys the run reports, so
"the option worked" and "the option was ignored and the default agreed" are distinguishable.

**What these do NOT verify.** Numerics at small shapes, nothing else. It does
not check the banded output path (a framebuffer is at most `MAX_TEXTURE_SIZE` on a side, so long
tensors are drawn in several passes with a row offset, and no arm here is large enough to need a
second band); it does not touch transform feedback, which is where the 47.8% comes from; and it says
nothing about the 1.4x/1.55x speed comparison, which remains a single hand-run.

## The walls, which are shapes and not speed

🔴 **THE "181 TOKENS" CEILING IS A PROPERTY OF ONE LAYOUT AND NOT OF THIS BACKEND - CORRECTED
HERE, HAVING BEEN WRITTEN DOWN TWICE.** docs/A100.md measures a GEMM whose operand is a 2D texture
of `(K/4) x M` with `M = tokens^2`, so the pair COUNT must fit one texture axis and stops at
`sqrt(MAX_TEXTURE_SIZE)` - 181 on that card, 128 on the M5 here. Real, for that layout.

Addressed **linearly** - what `runtime/storage.js` does, and what the triangle kernel has always
done - the bound is the total element count, `2048 * MAX_TEXTURE_SIZE * 4`: **134,217,728 floats on
this M5, which is 1024 tokens** at AF2's 128-channel pair width. Measured, not argued: a 400-token
pair tensor of 20,480,000 floats round-trips with 0 differing. So the wall this file described in
its first version is eight times further out than it said, and AF3's pair-transition is
expressible after all. The 2,047-residue storage-binding ceiling the WebGPU port hits is the next
wall, not this one.

What remains true: WebGPU has no shape-dependent limit at all (a buffer is one-dimensional and
`maxStorageBufferBindingSize` is 2 GB), and a 2D-operand kernel written for speed would still meet
the axis limit - so the layout is a real constraint on how a fast GEMM gets written here, just not
on whether a fold is expressible.

    MAX_TEXTURE_SIZE 32768 -> 181 tokens      16384 -> 128      8192 -> 90      2048 -> 45

🔴 **AND THE SECOND MACHINE MEASURED REPORTS HALF THE FIRST'S, SO 181 IS NOT THE NUMBER TO
PLAN AGAINST.** An M5 through ANGLE Metal reports `MAX_TEXTURE_SIZE` **16,384** - a pair-track
ceiling of **128 tokens** - against the A100's 32,768 and 181. Its output budgets are smaller too:
32 floats a fragment (8 draw buffers, as the A100) but **120** a vertex invocation against that
card's 124. A current Apple GPU, not an old one, and already at half the ceiling every number in
docs/A100.md was taken at.

`pairTokenCeiling` in `runtime/limits.js` is that arithmetic, because the ceiling is a property of
the context and never a constant - and **the specification floor is 2,048**, so a machine that
needs this backend may bring a ceiling of 45 residues. Past it every pair-track kernel wants its
row index tiled across two texture dimensions, with the address arithmetic that implies. Work, not
an impossibility; work nothing here has done.

**On the shapes this repository actually runs it is 1.6x to 2.1x**, not the 1.5x of a square
matmul: AF2's transition (30208x256x1024) 9.05 against 14.70, the diffusion's `qkvg`
(240x768x768) 1.51 against 3.18.

**And `grid.attend`, the transition and the diffusion attention all depend on workgroup memory**,
which does not exist here at all.

## What it would take, and what to measure before starting

`webgpu/` is **250 `@compute` entry points**. Two are ported. Before the third is written, three
measurements decide whether any of it is worth doing, and each can kill it cheaply:

1. **Coverage.** What fraction of the no-WebGPU population actually has WebGL2 *and*
   `EXT_color_buffer_float` *and* `MAX_TEXTURE_SIZE` large enough *and* the vertex output budget?
   `tools/gpu/probe-webgl2-limits.js` asks exactly this and has only ever run on development
   machines. Nobody here has the number.
2. **The 47.8% on old hardware.** Re-run `tools/gpu/bench-webgl2-tf-matmul.js` on an actual old
   integrated GPU rather than the A100. This is the measurement the case rests on and the one
   least likely to transfer.
3. **Whether the pair track can dodge the texture axis** - tiling `tokens^2` across array layers
   (`MAX_ARRAY_TEXTURE_LAYERS`, which the limits probe already reads). If it can, the ceiling
   moves and a length-generous goal is possible; if not, this backend is capped near a small
   domain and should say so in its first sentence.

And **weights are a separate wall from kernels**: the bundles run 73 MiB (`model-af2-monomer-int5`)
to 2.1 GiB (`model-rfd4p-int8`), and an old machine with 4-8 GiB of RAM is not folding from the
large end whatever the kernels do. Only the small end is browser-shaped on the hardware this tree
exists for.

## Three measurement traps, each of which reported a confident wrong number

Recorded in docs/A100.md and worth repeating where the kernels are:

- **`gl.finish()` does not block here.** Four draws of a 2048-cube measured **0 ms** until
  `readPixels` replaced it.
- **A buffer bound to `TRANSFORM_FEEDBACK_BUFFER` may not also be bound to `ARRAY_BUFFER`.**
  `getBufferSubData` then returns zeros behind an `INVALID_OPERATION` nobody reads, and both the
  base and the transform feedback OBJECT have to be released.
  `tools/gpu/probe-webgl2-tf.js` is the smallest case that works.
- **A surgery arm that prices a fetch by replacing an operand with `vec4(1.0)` is useless here**,
  because with a = 1 the eight accumulators become one expression and collapse. That arm reported
  **250 TFLOP/s**.

## The tools

```
node tools/gpu-chrome.mjs tools/gpu/probe-webgl2.js            what this device exposes, and whether any of it is compute
node tools/gpu-chrome.mjs tools/gpu/probe-webgl2-limits.js     the limits that bound a GPGPU kernel, fragment and vertex
node tools/gpu-chrome.mjs tools/gpu/probe-webgl2-tf.js         the smallest transform feedback that works
node tools/gpu-chrome.mjs tools/gpu/bench-webgl2-matmul.js     the fragment path: fetch-rate bound, intensity capped at 21.3
node tools/gpu-chrome.mjs tools/gpu/bench-webgl2-tf-matmul.js  the vertex path: 2.2x, and where it stops being fetch-bound
node tools/gpu-chrome.mjs tools/gpu/check-webgl-kernels.js     both kernels against independent CPU references; --falsify
```

## Making it selectable

Two lines, neither of which should move before a fold actually runs here: `tools/build_site.py`'s
`DIRECTORIES` does not copy this tree, so it is not on the site; and the page requires WebGPU
rather than choosing a backend. A backend on the site that cannot fold is worse than no backend,
because the page would offer it.

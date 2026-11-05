# LocalFold on WebGL2

The fallback backend, for the browsers and machines that have no WebGPU. `webgpu/` is the page's
backend and needs Chrome/Edge 113+ or Safari 26+ - which the page says in those words
(`web/design.js`: "WebGPU unavailable — this page needs Chrome or Edge 113+, or Safari 26+, to
fold"). WebGL2 has been in Safari since 15 and in Chrome since 56, so it reaches machines and
operating systems that the WebGPU path refuses outright.

🔴 **ALPHAFOLD 2 FOLDS HERE NOW - BOTH GRAPHS - AND NOTHING SELECTS IT YET.** 59 residues from
`model-af2-monomer-int5`, 4 extra-MSA blocks and 48 Evoformer blocks, the template stack, eight
structure iterations and both confidence heads: **consecutive alpha carbons 3.834 A, pLDDT 57.937
against the WGSL fold's 57.938, and the worst atom 0.0565 A away** against a measured floor of
0.0355 A - that residual and that floor are both float16 weights' (see "Closing the gap"); at
`--no-half-weights` they are 0.0291 and 0.0973. The MULTIMER folds from `model-multimer` on the same driver, and agrees far more closely
- **0.0005 A at the worst atom** - because that graph is f32 throughout.
`tools/gpu/fold-af2-webgl.js` is both folds and both comparisons. The SPEED is **1.25 s where the
WGSL fold of the same weights is 0.27 s**, a 4.6x gap of which 1.1 s is shader compilation - see
"Closing the gap".

🔴 **THIS LINE USED TO READ "7.1 s where WebGPU is 0.42 s", A 17x GAP, AND THE 7.1 WAS ALMOST
CERTAINLY A COLD CLOCK.** The identical command measured **6256 ms on the first run of a session
and 1446-1465 ms on every run after it**, on an idle Apple GPU that ramps its clocks under load.
Nothing about the tree changed between those runs. So the first number any of these tools prints
on a quiet machine is not the number - take the second.

This file began as the spike that answered *can WebGL2 do this work at all*: the two kernels that
decide it, moved out of `webgpu/kernels/` where they sat with `webgl2` in their names and no
importer, plus the texture layout and the limit arithmetic they shared by copy. The honest summary
of the sections below: the numerics are **checked and correct**, the speed is **4.6x off the WebGPU
path on AF2 and 15x on bioemu2** - it was 17x and 246x - and the shapes are **worse**.

🔴 **AND ON MEMORY IT IS NOW THE EFFICIENT ONE.** A 68-token bioemu2 fold holds **409 MiB** here
against the WGSL fold's 815 MiB of weights plus 135 of activations, because this path keeps the
checkpoint in the five-bit form the bundle ships in and decodes it in the kernel while the WGSL
side expands it to float16. The same fold was 1656 MiB and 968 s at the start of that work and is
**57.6 s** now. Nothing about the sections below is a projection: every figure is a fold that ran,
and the ones that are losses are in there too.

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
  af3/
    trunk/transition.js         AF3's SwiGLU transition, a BLOCKED split
    trunk/triangle.js           AF3's triangle multiplication, an INTERLEAVED split
    trunk/grid-attention.js     triangle (grid) self-attention, both directions
    trunk/single-attention.js   the single track, biased by the pair
    trunk/pairformer-block.js   seven sub-layers threaded, not seven parallel deltas
    trunk/msa-attention.js      pair-weighted averaging: no queries, no keys
    trunk/msa-block.js          ...and the two crossing lines, with no default
    trunk/embedder.js           the trunk's inputs; 139 one-hots as four weight rows
    trunk/template.js           the module everyone skips: an empty slot is not a no-op
    trunk/trunk.js              the assembly, and the distogram head's SUM symmetry
    diffusion/atom-attention.js the AdaLN block that is 98% of the diffusion head
    diffusion/atom-conditioning.js  chemistry in: the reference conformer, and arcsinh
    diffusion/transformer.js    the 24 blocks, nested six by four
    diffusion/atom-encoder.js   the five gathers, the pair, and seven stage taps
    diffusion/conditioning.js   three pair shapes, and a refusal where the width disagrees
    diffusion/head.js           the decoder and the whole denoise step: a BLEND, not a prediction
    diffusion/sampler.js        the walk: every constant imported, never restated
    confidence/confidence.js    pLDDT per ATOM SLOT, PAE, PDE, and rf3's masked global norm
    model/fold.js               the whole AF3 fold: target_feat, trunk, the walk, confidence
  af2/
    multimer/template.js        the multimer's nine-projection template embedder
    evoformer/block.js          a whole Evoformer block and the extra-MSA block, nine sub-layers
                                each, sharing the six that make up the pair half
    evoformer/stack.js          the loop over blocks, for both kinds
    evoformer/input-embedder.js the MSA, extra-MSA and pair embeddings, and the recycle
    evoformer/template.js       the template stack: one Linear over 88 geometry columns, the pair
                                blocks, and a pointwise attention that is two projections
    evoformer/input-embedder.js ...and the multimer's 73-row relative encoding, which is a superset
                                of the monomer's 65 and is held bit-identical to it on one chain
    structure/initialize.js     the single representation and the identity frame
    structure/sidechain.js      seven torsions a residue, as normalised (x, y) pairs
    structure/geometry.js       atom coordinates from frames and torsions - atom14 and atom37
    structure/ipa.js            invariant point attention: the one attention whose logits are
                                a distance - eight draws plus five projections
    structure/iteration.js      the IPA update folded in, a transition, and the frame advanced
    structure/module.js         the four stages composed, eight iterations, one hoisted pair norm
    model/fold.js               the fold, both graphs: the stages in order, with recycling,
                                from a real bundle
  esmc/
    tower.js                    ESM-C's 36 blocks and ESMFold2's shim, mixed as they are produced
  esmfold2/
    pair-features.js            z_init's five terms, and the chain block's inverted polarity
    trunk.js                    24 blocks of the pair track alone - AF3's block, less three parts
    distogram.js                the head whose symmetrisation IS the head
    atom-transformer.js         the windowed atom transformer with 3D RoPE - no AF3 analogue
    diffusion.js                the conditioning, twelve token blocks, and the EDM step
    sampler.js                  the walk, with the pair conditioning cached across it
    confidence.js               pLDDT, PAE, pTM - and the stack added to its own input
    model/fold.js               the whole ESMFold2 fold, from two checkpoints
  bioemu2/
    denoiser.js                 the dual stream: 16 layers coupled through every centre atom
    embedder.js                 three windowed atom layers and a token mean, giving s_inputs
    sequence.js                 z_init, s_init and m0 - four relative blocks, four conditions
    trunk.js                    the PairMixer: four MSA layers and 48 pair-only blocks
    context.js                  the denoiser's static context - the pairwise conditioner, q0 and p
    model/fold.js               the whole BioEmu2 fold, 1.6 GiB of f32 denoiser weights
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
arm gets seventy times WORSE, 7.98e-6 to 5.40e-4, so the WGSL template block stores its attentions
in f16 exactly as the main block does. `--tri-f32` moves nothing, which is the control.

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

## It folds

```
node tools/gpu-chrome.mjs tools/gpu/fold-af2-webgl.js
node tools/gpu-chrome.mjs tools/gpu/fold-af2-webgl.js --recycles=1 --no-reference
```

`webgl/af2/model/monomer.js` is the driver: the template stack once, then every pass through the
input embedder, the extra-MSA stack, the Evoformer stack, the structure module, the confidence
heads and the contact map, carrying the MSA's first row, the pair and the atom37 forward. 59
residues, one pass, from `model-af2-monomer-int5`:

| | WebGL2 | WGSL |
|---|---|---|
| consecutive CA, median | 3.834 A | - |
| mean pLDDT | 57.941 | 57.938 |
| atom37, relRMS | 2.45e-3 | - |
| worst atom apart | 0.0705 A | - |
| f16 dither floor, worst atom | 0.0554 A | - |
| fold | 7111 ms | 417 ms |
| weights resident | 446.4 MiB | - |
| draws | 5187 | - |

🔴 **AND "0.07 A APART" IS MEANINGLESS WITHOUT ITS FLOOR, WHICH IS WHY THE TOOL MEASURES ONE.** The
trunk keeps three intermediates in f16 with no switch to reach them, so both backends quantise, and
two f32 values one step apart that straddle an f16 boundary round 5e-4 apart - compounded over 48
blocks and eight structure iterations that reaches the coordinates. `execution.halfDither`
perturbs by one f32 step INSIDE every one of those roundings and the fold is run again: **0.0554 A
at the worst atom**, the same order as the cross-backend 0.0705, so the disagreement is
quantisation and not a defect. `atFloor` is that comparison and the verdict fails on it. With a
recycle at 24 residues the two agree to **0.0154 A against a 0.0347 A floor** and pLDDT to all
three decimals.

🔴 **AND THE CHAIN GEOMETRY IS A GATE HERE TOO, FOR THE REASON `fold-af2.js` HAS ONE.** pLDDT reads
**92.50 on a fold truncated to one Evoformer block whose chain has collapsed to 0.774 A between
consecutive alpha carbons** - measured, in this tool, on the way to the working fold. Both backends
produce that same collapsed structure from one block, agreeing to 0.0365 A, so it is the model at
depth one and not a port error; what it is NOT is detectable from the confidence number. The bands
are `fold-af2.js`'s own.

🔴 **`--main-blocks=` AND `--extra-blocks=` TRUNCATE BOTH SIDES**, which is how a divergence is told
from an accumulation: one block agreeing and forty-eight not is an accumulating difference, where a
disagreement at the first block is a defect in the block.

## The multimer graph, so far

Both block kinds run in multimer order, at **8.2e-8 / 1.1e-7** (main) and **8.0e-8 / 1.1e-7**
(extra) against the WGSL multimer block, each at its own measured floor, and the same in monomer
order under `--order=monomer`. `outerProductMeanFirst` is the flag: the mean runs BEFORE the MSA
row attention, so the pair receives the mean of the MSA as the block received it where the monomer
gets the mean of an MSA already updated three times.

🔴 **IT IS GATED IN BOTH BLOCK KINDS BECAUSE THE REFERENCE SHIPPED WITH IT WORKING IN ONE.**
`webgpu/af2/multimer/block.js` says so at `encodeExtraMsaBlock`: "Only the main evoformer honoured
the flag before; this stack read it and did nothing, which left every multimer fold running one of
its two stacks in the monomer order." This port writes the branch twice because the two blocks do
not share their MSA half, so both are compared.

🔴 **AND THE FLAG IS PROVED NOT TO BE INERT BEFORE ANY RESIDUAL IS BELIEVED.** CLAUDE.md's rule -
"a control arm that cannot vary the thing under test is not a control" - so four `orderMoves` arms
run this port with the flag off and assert the answer moves: **0.054 and 0.008 on the main block's
pair and MSA, 0.043 and 0.007 on the extra block's**, against a 1e-3 bar. Without them, both sides
ignoring the flag would read as perfect agreement.

🔴 **AND THE TWO GRAPHS DIFFER IN THREE PRECISION DECISIONS BESIDES THE ORDER, NONE OF THEM A
KNOB - WHICH COST THREE ROUNDS OF 1e-4.** The multimer block's transitions keep their hidden row in
**f32** where the monomer's round it to f16 whenever the width is a multiple of four; its attention
targets are **all f32** ("AND THE MULTIMER'S TARGETS ARE ALL f32, which is the SIMPLER of the two",
in its own words) where the monomer's store the normalised input, the projections and the output in
f16; and its outer product mean and triangle contraction take f16 ARITHMETIC from device tuning,
which `setHalfPrecision` does not reach - `opmVectorArithmetic` and `triangleContractArithmetic`
are the controls. Each one was found by the residual refusing to move, and each is a property of
the graph rather than of the device: `transitionPrecision`, `msaAttentionPrecision` and
`triangleAttentionPrecision` are how this port states them.

**The 73-row relative encoding runs** - 6.8e-8 on the pair against the WGSL multimer embedder with
a real three-chain complex, two chains being copies of one entity at different symmetry ids, so
every branch is reached: same chain, same entity across chains, different entity, and symmetry
deltas of 0, +1 and -1. The monomer looks up ONE row; the multimer sums three - the clamped offset
(or row 65, "a different chain entirely"), row 66 when the two share an ENTITY, and one of six
relative-chain rows.

🔴 **AND THE REFERENCE'S CLAIM THAT THE MULTIMER FORM IS A SUPERSET IS NOW MEASURED RATHER THAN
QUOTED.** It says "THE MULTIMER GRAPH IS THE ONE GRAPH, and monomer is its single-chain case...
zero rows there reproduce the 65-row result EXACTLY, and the shader needs no monomer branch." One
arm runs a monomer's features through the CHAINS kernel with a widened weight table and holds it to
**relRMS exactly 0** against the monomer kernel, and to 6.9e-8 against the WGSL monomer embedder.
That is the claim that would justify deleting the monomer form, so it is measured before anyone
acts on it - and it is declared `falsifiable: false`, because it compares two of this port's own
kernels on one input and no perturbation of that input can move one side without the other.

**`position_scale` is a parameter now**, not the 10.0 that was written into the atom14 shader: ten
for the monomer checkpoints and twenty for multimer, because the weights were trained against their
own value. Both are **bit-exact** (relRMS 0) against `webgpu/af2/multimer/geometry.js`, which
generates its shader per scale for the same reason, and a `scaleMoves` arm asserts the two differ
(0.491) so that neither side can be ignoring it. A monomer scale on multimer weights halves every
coordinate from the origin: a structure, just not the model's, and nothing downstream would raise.

**The nine-projection template embedder runs** - 1.3e-7 with a real template, 1.5e-7 with a masked
one and 4.9e-8 at four padded slots, each at its measured floor against the WGSL multimer embedder.
The monomer has one Linear over an 88-column concatenation; the multimer sums nine projections, and
three of its conventions produce a plausible number when wrong, so each is its own arm:

- **a masked template still contributes.** The wrapper adds this term unconditionally and the
  biases, a one-hot of aatype zero and the query embedding all survive the mask. The reference
  records what omitting it cost: "the pair track 30% out from the first evoformer block... shattered
  backbones at six copies".
- **`templates` is the DIVISOR of the mean and sits inside the relu**, so the PADDED slot count is
  what divides - the same trap `templateFeatureMeanOnePass` records for rosettafold3, a stage exact
  at one slot and a quarter strength at four.
- **the pair blocks run their five operations in the OPPOSITE order to the monomer template
  stack's.** "AF2 monomer's TemplatePairStack attends first and multiplies second; multimer's
  TemplateEmbeddingIteration multiplies first... running one order with the other's weights is the
  kind of mistake that produces a plausible number." `orderMoves` asserts the flag is live at
  **0.084** before any residual is believed.

🔴 **AND THIS PORT WAS MORE PERMISSIVE THAN THE REFERENCE, WHICH IS HOW THAT WAS CAUGHT.** The WGSL
embedder REFUSES a template with no `asymId` - "AF2-multimer masks the template's geometry across
chains, and assuming one chain lets it speak about pairs it has never seen in one coordinate frame"
- where this port defaulted to an all-ones mask, copied from the monomer path where one chain IS
the model. That is the exact mistake the monomer template's own comment warns about, and it
surfaced only because the reference threw where this port would have folded. The refusal is
matched, and the gate runs two chains.

🔴 **AND THE ORDER CONTROL READ "INERT" THROUGH A DILUTION BEFORE IT READ 0.084.** The arms compare
`pair + update`, because that is what the reference writes, and the pair is more than an order of
magnitude larger than the update - so the same order change reads **2.6e-5 on the sum** and 0.084
on the update itself. A control measured through a dilution reports "inert" about something that is
not.

AF2 is done, both graphs, and so is AF3 - and AF3 FOLDS: a 59-residue chain through the whole
48-block trunk and 25 diffusion steps is **21.0 s** here against the WGSL fold's 1.26 s, with a
median CA-CA of **3.855 A**, mean pLDDT 57.076 against the WGSL fold's 57.048, and the worst atom
**0.078 A** from it. ESMFold2 folds (below), and **so does bioemu2**: all five of its stages are
ported, and 6MRR at 20 steps lands on **RMSD 1.595 A, TM 0.9083** - the WGSL fold's own numbers to
four digits, from the same seed. So all four models in the goal fold in WebGL2.

🔴 **AND THE FOLD FOUND A BUG NO STAGE GATE COULD.** The confidence head CLAMPED the trunk's single
to +/-512 on the branch where AF3 uses it raw - protenix2 is the model that normalises there - which
is invisible at the noise scale every stage gate drives it with and took a real fold's pLDDT from
75 to 6 while PAE and the coordinates stayed in band, because the pair half of a pairformer block
never reads the single. The confidence gate now drives a single at the trunk's own magnitude WITH A
HEAVY TAIL, since a uniform draw at std 200 never reaches 512 and left the arm green with the bug
restored.

**AND ESMFold2 FOLDS TOO**: ubiquitin's 75 residues through the whole 36-block ESM-C tower, four
trunk loops of 24 blocks and eleven denoiser calls is **240 s** here, median CA-CA **3.637 A**, mean
pLDDT 57.4, pTM 0.413, 126 contacts over 0.5, peak 2978 MiB. At 20 residues it is 4.2 s.

🔴 **AND IT TAKES ALL FIVE FLAGS, BECAUSE EVERY DEFAULT IN THAT TOOL IS A TRUNCATION:**

```
node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2-webgl.js \
  --length=75 --tower-blocks=36 --trunk-blocks=24 --loops=3 --steps=15
```

Re-measured at this commit: **pTM 0.4132, 126 contacts, 65.0 s** - the 253 s line above at
**3.9x**, with the same contact count and a pTM a ten-thousandth away, which is the float16
weights; at `--no-half-weights` it is 0.4133 exactly. See "Closing the gap". Leave a flag out and the tool folds a prefix of the model and the geometry gate rightly
calls it "not a chain": `--loops=0 --trunk-blocks=4` (the defaults) reads **2.478 A** at the full
tower, and a 3-block tower reads 3.292. All three of those were read as a REGRESSION here before the
flags were checked, twice from a log line that had truncated `--tower-blocks=36` to `3`. A number
quoted without its configuration is a number the next reader cannot reproduce.

🔴 **AND 92% OF THAT IS THE TRUNK**: 4 x 55 s against the tower's 1.7, the shim's 0.3 and the atom
transformer's 1.1 (the stage clock is in the fold tool's own report). The pair track's triangle
contraction is O(L^3) a block and there are 24 of them a loop, so the trunk is the lever for this
model exactly as the 24-block transformer is for AF3's denoiser.

🔴 **AND THE SAMPLER'S CONSTANTS ARE THE MODEL'S, NOT AF3's.** ESMFold2 wants sMax 160, rho 8,
gamma_min 1.107, noise scale 0.901 and step scale 1.638; with the AF3-shaped sMax 16 the 256 cap
drops NOTHING, so "15 scheduled run 11" silently became 16 - the fold tool holds the two backends'
constants against each other for that reason.

🔴 **AND 61% OF A DENOISER CALL IS THE 24-BLOCK TRANSFORMER**: 425 ms of 695 at 59 tokens
(`profile-webgl-af3-head.js`), with the atom encoder's stack at 55 and the pair conditioning at 51.
Hoisting that pair conditioning out of the walk - it does not depend on the noise level, so 25 steps
recomputed one tensor 25 times - is correct and measures **2%** (436 against 445 ms a step), which
is worth knowing before anyone spends a day on it: the transformer is the lever.

Every operation so far is checked against TWO references - an f64 CPU one written from the
operation, and the WGSL kernel run in the same page - because a single reference cannot say which
side is wrong. That has already paid for itself twice: the triangle contraction's incoming
direction was the pair-transpose of itself and only the WGSL arm could see it, and the attention
block read 1e-3 against WGSL while reading 2.5e-7 against f64, which was an unpinned precision axis
on the WGSL side rather than a defect here. And since `model/monomer.js` there is a third
reference that none of them is: **a real checkpoint**. docs/AF2.md records why that matters - "a
packing that moved the real model 4e-4 read 6e-7 on noise" - because real weights are not uniform,
so a layout or a transpose that noise forgives shows up in a fold.

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

The vertex path is 2.2x the fragment path there and the whole difference was read as **output
budget** - a fragment writes `MAX_DRAW_BUFFERS` x RGBA = 32 floats, a vertex shader with
rasterisation discarded writes up to `MAX_VERTEX_OUTPUT_COMPONENTS`, and more outputs an invocation
is more arithmetic per fetch.

🔴 **THAT READING IS WRONG ON THIS DEVICE, AND THE 2.2x IS NOT THE BUDGET.**
`tools/gpu/bench-webgl-tf-linear.js` runs one real projection both ways: at the same 32-float
budget transform feedback is **1.00x** the fragment kernel and bit-identical (`relRms` 0), and at
112 floats - the budget that was supposed to be the point - it is **1.85x slower**, because that
many accumulators spill out of the register file. The table's own winning arm (`8x1`) writes 32
floats, the same as a fragment, so the 2.2x was never evidence for the budget; it is a property of
a square-2048 GEMM, whose operands stream perfectly where a projection's do not. **The binding
constraint is registers.** This is the vertex shader used as a compute kernel, which is what people did
before compute shaders existed. See `vertexOutputFloats` in `runtime/limits.js` for why the
advertised limit is not the real one.

🔴 **AND EVERY NUMBER ABOVE IS AN A100's, WHICH IS NOT WHO THIS BACKEND IS FOR.** The machines
without WebGPU are older machines, and no speed here has been measured on one. Old drivers are
exactly where transform feedback gets flaky, so the 47.8% is the figure most at risk of not
transferring - and it is the figure the whole case rests on. The two kernels have now been run on a
second machine, and it is not an old one either: an M5 through ANGLE's Metal backend.

## Closing the gap, and the two instruments that lied about where it was

The question was whether the tricks in `sokrypton/rfd4-proteina-webgpu`'s own WebGL2 port close the
distance to WGSL here. It has **four** - a blocked matmul into several render targets, operands as
float16 packed eight to a texel, narrow texture rows (384 texels, where this tree uses 2048), and
the fragment ORDER its `cmLimit` decides - and this file first looked for three of them, which is
why the fourth arrives at the bottom of this section by a different route.

- **The blocked matmul** is worth **6.3x to 7.9x** on the projections bioemu2 is made of.
- **Packed half-precision operands** are taken twice over: float16 in RGBA16F (**1.4x and nearly
  half the peak**) and then the bundle's own five-bit codes decoded in the kernel (**another 2.08x
  of memory**, and the only optimisation here that is more accurate than what it replaced).
- **The fragment order** is **1.23x** where it applies, and **5.1x against** where it does not -
  see "Which operand streams" below, which is also the reason the int5 kernel was two and a half
  times off its own model.
- **Narrow texture rows** are measured and are **a loss on this device**, which is the one place
  this port and that one genuinely disagree.

🔴 **NARROW ROWS ARE 1.6x FOR THEM AND 3.8x AGAINST US, AND THE REASON IS THE ADDRESSING.** Their
measurement is a trunk pass at 2.35 s with 3072-texel rows and 1.47 s with 384. Here, with the same
fold and the same answer to every digit (`caca` 2801.7916 at all three widths, since this changes
only where a texel lives), `TEXTURE_WIDTH` costs:

| `TEXTURE_WIDTH` | bioemu2 cheap fold | its trunk |
|---|---:|---:|
| **2048** (shipped) | **8.41 s** | **0.75 s** |
| 1024 | 12.89 s | 2.82 s |
| 512 | 11.98 s | 2.77 s |

Their kernels walk a row with a FIXED texture row - the width is chosen so a row never wraps, so a
tile of fragments covers nearby rows of its operands and a narrow texture keeps them together. This
tree addresses linearly (`Storage`'s first paragraph): a tensor is an index, `fetch4` is
`(t % W, t / W)`, and nothing about a logical row reaches the texture. A fragment's consecutive
reads are consecutive INDICES, so a wide row is what makes them one cache line - and narrowing it
breaks every sweep into more rows than the cache can follow. The trick is a property of their
layout, not of the API, and `W` must also stay a power of two here (their 384 is not) because
`% W` and `/ W` are in every fetch. 🔴 **AND IT COSTS CAPACITY**: the largest tensor is
`W * MAX_TEXTURE_SIZE * 4` floats, so 512 would cut this backend's reach by four.

And the two biggest wins were not on that list at all - one of them a kernel doing sixty-four times
the work it needed to - while the first two attempts to find them both reported confident wrong
numbers.

🔴 **A FRAGMENT PROFILE ON THIS API MEASURES THE INSTRUMENT.** WebGL2 has no timestamp query, so
`Execution.profile` times a kernel by WAITING for it - a one-texel `readPixels` of its output,
which cannot return until that draw has run. Over an AF2 fold's 5,187 draws the wait is most of
what it measures: the profiled fold is **3862 ms where the fold is 1410**, and every label's share
came out within a few points of its share of the DISPATCH COUNT. `linear` read 49.7% for that
reason and not because it is half the arithmetic.

🔴 **AND THE OBVIOUS FIX IS DEFEATED BY HIDDEN-SURFACE REMOVAL.** `Execution.doubleLabels` issues
every matching draw twice: the kernels are pure, so nothing it computes changes, and the fold's
wall should grow by exactly that family's GPU time. It grows by **2%** on a bioemu2 fold - and the
same fold with `--skip-draws`, which issues no draws at all, is **895 ms against 28.7 s**. The
draws are 97% of that fold and doubling them is free, because an Apple GPU is tile-based and does
hidden-surface removal: the same full-viewport triangle drawn twice into one target with no read
between has its earlier fragments discarded before they are shaded. The same trap had
`tools/gpu/bench-webgl-linear.js` reading `linear.block:576x896x1792` at 11.25 ms a call where the
fold's own profile says 87 - a factor of eight, which was the iteration count. **A ring of distinct
outputs is the fix**, and `--skip-draws` is the arm that settles any argument between the other
two.

### What the gap turned out to be

**A big projection is bound by streaming its weight from DRAM, and the model is exact.** bioemu2's
atom projection is 576 rows x 896 in x 1792 out. One output texel a fragment reads the whole
6.4 MiB weight once per ROW - 3.7 GB - and this context moves about 10 GB/s: 438 ms measured
against 370 predicted. A block of R rows reads it once per R rows, so both the traffic and the time
divide by R. Measured at R = 8, which is `MAX_DRAW_BUFFERS` here:

| projection | one texel a fragment | 8 rows a fragment | |
|---|---:|---:|---:|
| bioemu2 atom 576 x 896 -> 1792 | 444.8 ms | 68.3 ms | **6.5x** |
| bioemu2 atom 576 x 768 -> 1920 | 423.9 | 67.4 | **6.3x** |
| bioemu2 token 68 x 768 -> 4608 | 158.3 | 22.7 | **7.0x** |
| bioemu2 token 68 x 3072 -> 768 | 56.7 | 7.2 | **7.9x** |
| af2 pair 3481 x 128 -> 128, `[out][in]` | 2.16 | 0.86 | 2.5x |
| af2 pair 3481 x 128 -> 128, `[in][out]` | 0.65 | 1.06 | **0.6x** |

🔴 **AND FOR `[in][out]` THE RULE IS EMPIRICAL AND NON-MONOTONIC, WHICH AN ATTEMPT TO SIMPLIFY IT
PROVED.** The gain is however much weight traffic there is to divide, and the three in-major sizes
these models use do not line up:

| `[in][out]` weight | | blocked |
|---|---:|---|
| 128 x 128 | 64 KiB | **0.44x to 0.65x** - a loss, repeatably |
| 128 x 512 | 256 KiB | **a win** - 1.18x to 1.38x on the bench, and removing it made an ESMFold2 fold **9.5% slower** (63.6 s to 69.7) |
| 256 x 512 | 512 KiB | **a loss** - removing it took a bioemu2 trunk **5245 ms to 4152** and `linear.repack` from 9.2% of that fold to nothing |

Loss, win, loss. 🔴 **SO "IN-MAJOR NEVER BLOCKS" WAS TRIED AND IS A NET LOSS ACROSS THE MODELS**: it
bought bioemu2 2.5% of a fold and cost ESMFold2 9.5% of one, so it was reverted. That is the whole
argument for folding four models on every verification pass rather than the one being optimised -
and the second time in this section that a rule fitted to one model regressed another. The
65,536-float threshold is fitted to those three points and is **not a model of anything**; a fourth
shape could fall either way and would need measuring. `[out][in]` wins at every size these models
use, because its four weight rows are read per OUTPUT GROUP rather than shared across one, so there
is traffic to divide even at 64 KiB.

**And then the whole thing was reading both operands' rows for every pair.** The split contraction
runs at **four flops a fetch where the projection reaches twenty-three**: a fragment owning four
channels of one pair reads `a[i][*][c]` and `b[j][*][c]` for each of those four channels, and in
this trunk's channel-major layout those four channels live in four different planes - 136 texel
reads for 272 multiply-adds. Measured inside a real 48-block trunk the contraction is about half of
it, at roughly **1.8 GFLOP/s** where the same device's projections reach 70.

Nothing about the sweep can be made cheaper; the operands can be REUSED. A block of **two i and
four j** computes eight pairs from two `a` rows and four `b` rows, so a k-step costs six fetches
instead of eight for four times the arithmetic - **10.7 flops a fetch**. Eight pairs of four
channels is exactly `MAX_DRAW_BUFFERS` targets. Measured at 48 mixer blocks, which is the
configuration a fold uses: **the trunk goes 8.49 s to 5.31 s, 1.60x**, with the fold's `caca`
identical to eight digits. `--no-contract-block` is the control arm, and a context with only the
four draw buffers the specification guarantees keeps the one-pair kernel.

🔴 **AND IT NEEDS A TOKEN COUNT DIVISIBLE BY FOUR, WHICH ONE OF THE TWO MODELS THAT USE THIS TRUNK
DOES NOT HAVE.** The block is two i and four j, so L must be even and a multiple of four -
bioemu2's 68 is, and ESMFold2's ubiquitin at **75** is not, which is why its full configuration
reads 64.9 s and 67.4 s either side of this change: unchanged within the variance, because it never
takes the path. A j block of two would reach it at any even length and is not written.

**And its outgoing direction asked for the same texel four times.** Four consecutive k of
`a[c][i][k]` are four consecutive floats, which is one RGBA texel, and `fetch1` fetches a texel to
select one component - so the loop issued four fetches where one would do. The texture cache
absorbs the BYTES, but this kernel is bound by the fetch RATE (~10 G a second) and not by
bandwidth, so the instruction count is the cost: one `dot` per four k is worth **1.25x on the
bioemu2 trunk stage** at four mixer blocks (1026 ms to 822), and **1.64x on the real trunk** -
78.4 s to 47.9 at 48 blocks and four passes, the largest single factor in this section after the
row block itself. The INCOMING direction
cannot do this directly - it walks a stride of L floats, so no two reads share a texel - **so it is
run as an outgoing contraction over transposed planes**, which is an identity rather than an
approximation: `sum_k a[c][k][j] * b[c][k][i]` is `sum_k aT[c][j][k] * bT[c][i][k]`, the outgoing
kernel handed bT and aT. Two transposes are two passes over the tensor where the strided form pays
L passes, and the incoming direction - the slower half of this stage once outgoing was coalesced -
goes from 126 ms a call to 64.

🔴 **AND THE FOUR-BLOCK MEASUREMENT UNDERSTATED THIS BADLY, WHICH IS WORTH RECORDING.** Read at
`--mixer-blocks=4` the coalescing and the transpose together are worth 1.15x on the trunk stage
(710 ms to 617); read at **48**, which is what a fold runs, they are worth **2.08x** (17.58 s to
8.47). A trunk at a twelfth of its blocks is not a twelfth of a trunk - the contraction scales with
the block count where the embedding and the sampler's share do not - so a cheap configuration
flatters whatever is NOT the pair track. Every trunk figure in this section is a 48-block one now. `--no-triangle-coalesce` and
`--no-triangle-transpose` are the control arms, the second of which also covers the same trick in
AF2's own contraction (`webgl/kernels/triangle-contract.js`).

**AF3's triangle contraction was doing sixty-four times the work it needed to.** Its fragment owns
four channels of one pair, the centre norm needs that pair's whole channel axis, so the fragment
contracted ALL 256 channels and kept four - and the 64 fragments of one pair each contracted the
same 256 channels again. That is `C * L * 2` = 34,816 texel reads a fragment over 295,936
fragments, **10.3 billion fetches**, which at ~10 G fetches a second is the **1055 ms a call** a
bioemu2 fold's profile measures: the most expensive kernel in that fold, 35% of it. The kernel's
own comment defended the fused form as what let "the centre norm see the channel axis at all
without a second pass over a channel-major intermediate" - and the second pass costs one read and
one write per texel. Split, the contraction writes a pair-major intermediate and the centre norm
reads it: bit-identical, because the statistics accumulate in the same order, and
`--no-triangle-split` is the control arm. The bioemu2 trunk-and-distogram phase goes **10.9 s to
3.6 s**.

**A baked shape is a shader, and a shader is a compile.** Every kernel here is generated with its
sizes as `const int`, so `linear` at 576x896x1792 and `linear` at 68x768x3072 were two programs -
and bioemu2 has eleven distinct projection shapes, so it compiled **168**. With the sizes in
uniforms one program serves every size and only the STRUCTURE forks the source: **109** programs
and `programMs` from 3927 ms to 478, at no measurable cost to the kernels, which are fetch-bound
rather than issue-bound.

🔴 **AND IT MOVED THE bioemu2 FOLD BY NOTHING, WHICH IS WORTH RECORDING BECAUSE IT WAS THE
SUSPECT.** Compilation looked like the answer precisely because the draw-doubling instrument said
the GPU was 2%; once `--skip-draws` put the draws at 97%, 59 fewer compiles was below noise. It is
where the SMALL models' time is, though: **1122 ms of an AF2 fold's 1400 is its 120 programs**, so
a first fold there is a compile queue and nothing else - the same thing docs/PERF.md records of the
WGSL side.

**`getError` after every draw is a synchronous round trip** - 4% of an AF2 fold - and
`checkFramebufferStatus` per dispatch is another. A framebuffer per attachment SET, bound rather
than rebuilt, removes the second; a fold reuses 65 of them because the textures are pooled.

### What it is worth, end to end

Every fold is the same job, the same seed and the same settings either side; `--no-linear-block` and
`--no-triangle-split` are the control arms, and every one of the 47 stage gates is green both ways.

| fold | before | after | |
|---|---:|---:|---:|
| bioemu2, 6MRR 68 tokens, 4 passes, 20 steps | 968 s | **57.6 s** | **16.8x** |
| ...its trunk | 277.7 s | **21.6 s** | 12.8x |
| ...its 20 denoiser calls | 658 s | **35.2 s** | 18.7x |
| ESMFold2, 75 residues, 36-block tower, 3 loops, 15 steps | 253 s | **65.0 s** | **3.9x** |
| ESMFold2 at the tool's defaults | 2.82 s | **1.37 s** | 2.1x |
| AF3, 59 tokens | 4.68 s | **3.39 s** | 1.4x |
| AF2 monomer, 59 residues | 1.40 s | **1.29 s** | 1.1x |
| peak, bioemu2 | 1656 MiB | **409 MiB** | **4.05x** |

🔴 **AF2 MOVES LEAST BECAUSE IT IS NOT DOING THE ARITHMETIC.** Its pair weights are 64 KiB and sit
in cache, so there is no traffic to divide, and 1122 ms of its 1400 is shader compilation. The
models that gained are the ones with big weights - which is the same sentence from the other end.

### Float16 weights, which need no kernel change at all

The sibling port packs float16 operands eight to an RGBA32UI texel and unpacks them in the shader.
That is not necessary here: **RGBA16F holds the same four values a texel in the same texel order**,
and a `sampler2D` returns them as a `vec4` of floats because the hardware converts on the read. So
`fetch4` and every kernel built on it are untouched, and only the internal format and the bytes on
the bus change. A big projection's time is its weight traffic, so halving the bytes halves the
traffic.

| fold | f32 weights | f16 weights | | peak |
|---|---:|---:|---:|---|
| bioemu2, 6MRR 68 tokens, 20 steps | 186 s | **133 s** | 1.40x | 1656 -> **851 MiB** |
| ...its 20 denoiser calls | 104.1 s | **50.9 s** | 2.05x | |
| AF3, 59 tokens | 4.68 s | **3.39 s** | 1.38x | 937 -> **509 MiB** |
| AF2 monomer, 59 residues | 1.31 s | 1.27 s | 1.03x | 469 -> **332 MiB** |
| ESMFold2, 75 residues, full config | 65.1 s | 64.4 s | 1.01x | |

🔴 **AND IT PAYS EXACTLY WHERE THE WEIGHTS ARE BIG, WHICH IS THE ROW BLOCK'S SENTENCE AGAIN.**
ESMFold2 and AF2 gain 1% and 3% because their per-block weights stay in cache, so there is no
traffic to halve; bioemu2's structure module is 424.6 M parameters and gains 1.4x. Two
optimisations, one rule - and it is why neither of them is a default anybody could have guessed.

🔴 **AND bioemu2's FOLD DOES NOT MOVE: rmsd 1.595, TM 0.9083, CA-CA 3.854, worst 3.904, bond rms
0.042/0.056/0.028, 201 contacts** - every reported quantity identical to the f32 fold and to the
WGSL fold. AF3's pTM is identical to four digits. So at these depths the models are insensitive to
eleven bits of weight mantissa, which is the same conclusion the WGSL path reached when it chose to
hold the same tensors in f16.

🔴 **IT IS STILL THE FIRST THING HERE THAT IS NOT BIT-IDENTICAL, AND AF2 IS WHERE THAT SHOWS.** Its
differential against the WGSL fold of the same weights - forced to f32 on the reference side, as
that driver has always done - goes from **0.0291 A at the worst atom to 0.0565**, against a
measured floor that itself moves 0.0973 to 0.0355. Both arms pass the gate's own `3 x floor` rule,
but the margin is five times tighter and the difference is a real precision change rather than
noise, so `--no-half-weights` is the arm that compares f32 with f32 and is the one to quote when
the question is correctness.

🔴 **AND THE DEFAULT IS THE FOLD DRIVERS', NOT `Execution`'s.** The 47 stage gates build their
weights through the same `upload*Weights` helpers and hold the kernels to f32 references at bars
down to 1e-7; a backend-wide default would have to re-bar every one of them. So a FOLD holds large
weights in f16 and a GATE does not, which is a split worth being explicit about: making f16 the
backend's precision rather than the folding one is work this has not done.

Two guards it needed. The conversion happens on the HOST through `float32ToFloat16Array` (round to
nearest, ties to even) rather than by handing f32 bytes to an RGBA16F texture, because the sibling
port records that ANGLE's Metal backend TRUNCATES on the way in and that truncation's bias
accumulated over a trunk pass cost it 6e-3 relative where rounding cost 4e-4. And a tensor whose
largest magnitude is past float16's 65504 stays in f32 and says so (`halfRefusals`): this entry
point is named by what the caller calls a weight, and an index table is not a weight in that sense
- ESMFold2's `atomToToken` goes through a helper that looks identical and must never be converted,
since f16 represents integers exactly only to 2048.

### The bundle's five-bit codes, decoded in the shader

Float16 weights were two bytes where the bundle ships **five-bit codes plus a group's scale and
zero - 0.875 bytes a weight**. Decoding them in the kernel instead of on the host is the only
optimisation in this section that costs nothing at all in accuracy: the value decoded is the
float32 number the host would have produced, so it is faster *and* more accurate than the float16
it replaces.

| | f16 weights | int5 codes | |
|---|---:|---:|---:|
| bioemu2, 6MRR 68 tokens, 20 steps | 89.7 s | **84.8 s** | 1.06x |
| ...peak | 851 MiB | **409 MiB** | **2.08x** |
| bioemu2's cheap fold (4 blocks, 2 steps) | 7.94 s | **6.83 s** | 1.16x |

🔴 **409 MiB IS LESS THAN THE WGSL BACKEND HOLDS**, which is 815 MiB of weights plus 135 of
activations for the same fold. This backend began the session at 1656 MiB and four times slower per
byte; it is now the more memory-efficient of the two, because it keeps the checkpoint in the form
the checkpoint ships in and the WGSL side expands it to float16.

🔴 **AND THE FIRST VERSION WAS 23% SLOWER THAN THE FLOAT16 IT REPLACED, WHICH IS THE WHOLE LESSON.**
It moved 2.3x fewer weight bytes and still lost, because a decode is about **fifteen integer
operations of address arithmetic for two texel reads** against float16's four for one - and the
blocked projection calls it once per output row per k-step, so at four codes a call that arithmetic
is paid eight times a group per row. The kernel is not purely traffic-bound, and quadrupling its
integer work spent more than the bytes saved. Decoding **eight codes at a time** halves it and is
where the win comes from: forty bits is exactly five bytes, every chunk's offset within a group is a
multiple of forty, so the shift is always 0, 8, 16 or 24 and the span is always exactly two words -
no running bit counter and no third read.

🔴 **TWO SHAPES CANNOT TAKE THE FAST PATH AND bioemu2 HAS BOTH.** The inner loop needs the codes to
run along the axis a fragment sweeps, which means `[out][in]` with an input width that is a whole
number of groups - and bioemu2's single conditioner is **839** wide (384 trunk channels plus 455
featuriser columns, a number with no reason to be a multiple of anything) while its pair
projections are read `[in][out]`. The uploader cannot tell which way a caller will read a tensor, so
`linear` decodes such a weight ONCE into float16 and memoises it on the weight object. That is why
the trunk stage grows slightly (619 ms to 843 at four mixer blocks) while the sampler, which is all
`[out][in]`, is where the gain lands.

🔴 **AND THE GATE IS EQUALITY, NOT A TOLERANCE.** `tools/gpu/check-webgl-int5-decode.js` holds the
shader to `readTensor` element by element on the largest int5 tensors bioemu2 ships: **0 of
10,616,832 elements differ**, not even by the fused multiply-add the compiler is allowed. It is a
differential rather than a floor because a quantised decoder that is wrong stays plausible -
`check-quantised-upload.js` once found the WGSL device decoder and the host one disagreeing on
**104,170 of 131,072 elements** at int5 group 64, and it was the HOST that was wrong. Its
`--falsify` arm reads a code as a multiple of the scale rather than an offset from the zero, which
is that same real fault, and turns 7,077,888 elements red.

### Which operand streams, which stays in cache - and three wrong guesses first

The int5 projection sat at **26 ms a call where its traffic model said 12 and its fetch model said
10**, and the three obvious explanations were all wrong:

- not bandwidth: float16 at **2.3x the bytes** took the same 27 ms;
- not the x reads: replacing them with a held value is worth **14%**;
- not occupancy: holding eight codes as two uints instead of eight floats, which frees 24
  registers, measured **2% worse**, and a four-row block measured 1.5x worse than the eight-row one.

The diagnostic that found it: make every chunk read **the same four code texels**, so they are
always cache hits, and the fold goes **6664 ms to 3476**. The kernel was bound by code-fetch
MISSES.

🔴 **AND THE FIX IS THE FRAGMENT ORDER, WHICH DECIDES WHICH OPERAND STREAMS.** A fragment's texel
index decodes into a row block and an output group, and which one varies fastest decides what its
neighbours share. Ordered by output group - `q = texel / OUT4`, what every kernel here did -
neighbours share x and walk DIFFERENT weight rows, so the weight streams once per row block, which
for a 576-row projection is seventy-two times. Ordered row block fastest they share the weight rows
and x streams instead.

So the right order is whichever operand is smaller, and getting it the wrong way round is worth
five times:

| projection | x | weight | row-block-fastest |
|---|---:|---:|---|
| bioemu2 atom, 576 x 896 -> 1792 | 2.0 MiB | 1.4 MiB (int5) | **1.23x** |
| trunk pair, 4624 x 256 -> 512 | **4.7 MiB** | 262 KiB | **5.1x WORSE** |

`ROW_FASTEST_X_FLOATS` is a million floats, which is the same number
`sokrypton/rfd4-proteina-webgpu` picks for the same decision (its `cmLimit`) - and this is the
third of its three tricks, arriving by a different route than the one this file first looked for.
Its own note on getting it wrong: ordered by columns "each row block streamed the whole weight
matrix again... about 0.2 TFLOP/s, 350 ms of a trunk pass". `--no-row-fastest` is the control arm,
and the fold's answer is identical either way - this moves where a texel is read, not what is
computed.

🔴 **AND IT IS THE INT5 PATH ONLY, WHICH THE AF2 ARM OF THE VERIFICATION CAUGHT.** Applied to the
float paths as well it cost AF2 **3.7x - 1264 ms to 4698** - and the reason is the REPACK rather
than the matmul: ordered row block fastest, the repack's source index strides by the row-block
count, which is 72 for bioemu2's atom projection and **436** for AF2's pair track, so every output
texel lands on its own cache line. The order is worth having only where the block count is small
enough that the repack survives it, so the threshold above is a necessary condition and not a
sufficient one. A generalisation measured on one model is a regression on another, and the only
reason this is a sentence rather than a shipped defect is that the AF2 differential runs on every
pass.

### How close can this get, and why not closer

The campaign above took a bioemu2 fold from **968 s to 57.6 s** and its device peak from 1656 MiB
to 409. The gap to the WGSL fold of the same weights is **15x** - it was 246x - and it decomposes:

| | WebGL2 | WGSL | |
|---|---:|---:|---:|
| trunk, 4 passes x 52 blocks | 21.6 s | 2.3 s | 9.4x |
| 20 denoiser calls | 35.1 s | 1.2 s | 29x |
| peak memory | **409 MiB** | 950 MiB | **0.43x** |

🔴 **AND THE FIRST EXPLANATION WRITTEN HERE WAS WRONG, FROM EXACTLY THE MISTAKE THIS FILE WARNS
ABOUT EVERYWHERE ELSE.** It said the gap was the subgroup MATRIX units, on the evidence that
`tools/gpu/probe-subgroup-matrix.js` reports them present (f32 and f16 at 8x8x8). That is a
CAPABILITY PROBE, not a measurement, and `--tune=matrixLinear=false` on the WGSL fold settles it:

```
default                     foldMs 3941   diffusion 1178
--tune=matrixLinear=false   foldMs 3934   diffusion 1179
```

**Taking the matrix units away changes that fold by nothing.** It never used them. A probe that
shows a feature exists says nothing about whether the code under test reaches it - which is this
repository's own rule ("a control arm that cannot vary the thing under test is not a control"), and
the `--tune` hook on `fold-bioemu2.js` exists now because nothing could reach that fold's knobs.

🔴 **THE GAP IS AN ARITHMETIC RATE, AND THE CAP IS THE OUTPUT BUDGET.** Summed over the shapes one
denoiser call dispatches, a call is about **143 GFLOP**. WGSL does it in 59 ms - **2.4 TFLOP/s** -
and this backend in 1.76 s, **81 GFLOP/s**. A compute kernel reaches that rate by staging a tile in
workgroup memory so a whole workgroup reads it at L1 speed; a fragment shader has no workgroup
memory, so its only reuse is what ONE invocation holds in registers, and that is capped by
`MAX_DRAW_BUFFERS x 4 = 32 output floats`. Every optimisation in this section is a way of spending
those 32 floats better, and the row block spends all of them: 23 flops a fetch against the 21.3
docs/A100.md measures as an eight-target fragment GEMM's ceiling.

So the floor follows from two measured numbers rather than from a feature list. The best fetch rate
observed here is **37 G/s**, on the one kernel whose whole weight (64 KiB) sits in L1; at the
fragment path's 21.3 flops a fetch that is **790 GFLOP/s**, which is **3x below** the 2.4 TFLOP/s
WGSL reaches on the same fold. The vertex path's four-times-larger output budget would double the
intensity and halve that to ~1.5x. **Neither is parity**, and the 3x is a lower bound that assumes
L1 residency this fold's 409 MiB of weights cannot have.

So the ceiling is structural, and the measurements say where it is:

- **The weight-reuse headroom is gone.** Forcing every weight read in the dominant kernel to hit
  cache - the diagnostic that found the fragment order - now changes a fold by **2%** (4757 ms
  against 4856). There is no more traffic to save.
- **The kernel is at the fragment path's intensity ceiling.** 512 flops per 22 fetches is **23
  flops a fetch**, against the 21.3 that docs/A100.md measures as the ceiling of an eight-target
  fragment GEMM. More outputs an invocation is the only way up and there are no more targets.
- **It is not at this device's fetch rate**, which looked like the remaining headroom: 79.5 M
  fetches in 19.5 ms is **4.1 G/s**, an L2 rate, against the 37 G/s the L1-resident kernel reaches.
  With the fragment order fixed the operand that streams is **x**, two thirds of the kernel's
  fetches, so packing it as halves eight to a texel should have cut 22 fetches a chunk to 14.

🔴 **IT WAS BUILT AND IT IS A LOSS TWICE OVER, SO THE "1.5x STILL AVAILABLE" THAT USED TO BE WRITTEN
HERE IS WITHDRAWN.** A conversion pass over x before each projection - one read and one write of
2 MiB, against 8 fetches a chunk saved - measures **7% SLOWER** (5763 ms against 5384 on the cheap
fold): the pass is paid per MATMUL, and the denoiser dispatches many with their own x, so it never
amortises the way a per-output-group saving suggests. 🔴 **AND IT OVERFLOWS float16 BEFORE IT IS
EVEN SLOW**: a transition's pre-activation at `576 x 1792 -> 128` packs to **-Infinity**, which
`--nan-watch` named on the first run. Weights are safe in f16 and were measured so; ACTIVATIONS are
not, and holding them there needs a per-tensor scale - a reduction pass, which would cost more than
the fetches it saves. The code is not kept, because a measured loss with an accuracy hazard is not
a control arm worth maintaining; the measurement is kept so the next reader does not rebuild it.
- **The vertex path's bigger budget cannot be spent, and that is measured now.**
  `tools/gpu/bench-webgl-tf-linear.js` runs the same projection both ways at 576 x 896 -> 1792: at
  the SAME 32-float budget a fragment has, transform feedback is **1.00x** - 70.2 ms against 70.3,
  with `relRms` **exactly 0** - and at **112** floats, the budget that was supposed to be the
  advantage, it is **1.85x SLOWER**, because 112 accumulators do not fit in a register file and
  spill. The binding constraint is REGISTERS, not outputs.

🔴 **SO "AS FAST AS WebGPU" IS NOT AVAILABLE - BY A MEASURED 3x AT BEST - AND THE USEFUL QUESTION
IS WHETHER 15x IS SHIPPABLE.**
A 68-token fold is 58 s where WebGPU is 3.9; a 59-residue AF2 fold is 1.3 s where WebGPU is 0.28,
and most of that 1.3 s is shader compilation rather than arithmetic. For a visitor whose browser
refuses the WebGPU path outright, those are folds instead of refusals, on less memory than the other
backend needs. That is the trade this tree exists to make, and it is a product question rather than
a kernel one - see "Making it selectable".

### A texture pool, and the bug it found

Pooling released textures by rectangle is worth **1%** of an AF2 fold and costs **4% more peak**
(465.9 MiB to 483.8), because a pooled texture is held rather than deleted - texture creation is
simply not what this backend waits for. On the fold that matters it is worse on both counts:
bioemu2 at 68 tokens is **185.8 s and 1656 MiB without it against 188.5 s and 2158 MiB with**. So
it is **off by default** and `--pool` turns it on, for a driver that allocates more slowly than
this one.

It is kept at all because it is an INSTRUMENT, and it found a real defect in TWO models the first
times it ran: the AF2 fold's answer changed, pLDDT 57.941 to 54.774, and bioemu2's TM moved 0.9083
to 0.9081. That is a
kernel reading texels nothing wrote, and `texImage2D` with a null pointer leaves the contents
UNDEFINED: Chrome on Metal happens to hand back zeros, and this backend's whole audience is drivers
nobody has tested. `Storage.fillFresh` fills every fresh texture with NaN and `Execution.nanWatch`
reads every output back and refuses at the first non-finite value, which named it in one run: the
first recycle pass has no previous activations and `webgl/af2/model/fold.js` expressed that by
ALLOCATING the three tensors and handing them to the embedder. `Storage.zeros` is what a caller
that wants zeros asks for now.

🔴 **AND THE SECOND MODEL HAD THE SAME FAULT IN THE SAME SHAPE.** bioemu2's first trunk pass has no
previous activations either, and `webgl/bioemu2/model/fold.js` allocated its `z_prev` and `s_prev`
and handed them to a pass that LayerNorms both - `--nan-watch` named `layer-norm:68x384` inside the
first pass. Two models, one habit: a first recycle is where this mistake lives, so the other two
drivers are checked under the same arm.

🔴 **A BISECT OVER WHICH ALLOCATION LABEL MAY REUSE A TEXTURE FOUND NOTHING**, because every label
was clean on its own and only the whole set together moved the answer - an allocation-order
coincidence, not a property of any one kernel. The NaN fill is what made the question local.

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

## bioemu2, all five stages

The DENOISER, the INPUT EMBEDDER, the SEQUENCE EMBEDDER, the TRUNK PASS and the DISTOGRAM are
ported and gated, and `tools/gpu/fold-bioemu2-webgl.js` strings them into a fold.

🔴 **THE SEQUENCE EMBEDDER'S THREE OUTPUTS READ EXACTLY ZERO AGAINST THE WGSL ONE, AND THAT IS NOT
AN INSTRUMENT FAULT.** `check-webgl-bioemu2-sequence.js` compares `z_init`, `s_init` and `m0` and
all three are bit-identical at 8 tokens and at 12: there is no accumulation here and no regrouping -
both backends add the same five terms of `z_init` in the same order, from the same host projections -
so the two land on the same f32 bits. This repository's own rule is that a suspiciously perfect
number is a reason to check the instrument, so the gate carries four SEPARATIONS through the same
`relativeRms`, each flattening one input group: the contact conditioning moves `z_init` by **0.83**,
the chain and entity blocks by **0.16**, the token bonds by **0.098** and the method by **0.027**. A
1% skew of `s_inputs` reads 1.6e-2 / 6.6e-2 / 3.8e-2. The bar is 1e-6 all the same, because another
device may contract a multiply and an add into one fma.

🔴 **AND THE STAGE'S FEATURES HAVE TO REACH BRANCHES A PLAIN FOLD DOES NOT.** The relative encoding's
four blocks fall back on four DIFFERENT conditions - a different CHAIN for the residue block, not the
same chain AND residue for the token block, a different ENTITY for the chain block, and the same
entity for `embed_same_entity` - so the gate runs two chains and two entities. And the contact
conditioning has three kinds: class 0 (`encoding_unspecified`, which every pair of a restraint-free
fold gets, so leaving it out drops a constant the model was trained with), class 1, and a real
contact, which runs a fourier embedding of its THRESHOLD and an encoder per pair. The gate REFUSES to
run if no pair reached class 2.

🔴 **AND THE SAMPLER MOVED RATHER THAN BEING WRITTEN TWICE**, the second time this family has done
it: `cpu/bioemu2/sampler.js` now holds the schedule, the random rigid augmentation, the churn, the
realignment and the Euler step, and `webgpu/bioemu2/fold.js` re-exports them so its behaviour cannot
change - the same treatment `host.js` got, and verified the same way (`moved[n] === reexported[n]`
for all five). A walk re-derived for the second backend is how two backends come to anneal
differently, and nothing errors on a wrong constant: the walk simply returns a plausible structure.

🔴 **AND THE TRUNK'S BOUND IS LOOSER THAN A STAGE GATE'S, FOR A REASON WORTH KNOWING.** Two f32
implementations through 52 pair-track blocks - each an O(L) contraction summed in a different order,
this port's fragment loops against the reference's GEMMs - diverge by accumulation: **3.6e-4** on
pass one at 8 tokens, 3.8e-4 on pass two, 7.7e-4 at 12 tokens, while the SINGLE reads **6.1e-8** at
every shape because nothing but the recycle touches it. Upstream's own gate holds a trunk pass to
5e-4 where it holds the stages either side to 1e-4.

🔴 **AND THE HOST HALF MOVED RATHER THAN BEING WRITTEN TWICE.** The 455 atom features, the atom-pair
tail, the window biases and the `s_inputs` assembly were in `webgpu/bioemu2/host.js` -
backend-agnostic host arithmetic inside one backend - and `webgl/` must not import `webgpu/`. They
live in `cpu/bioemu2/host.js` now, with `webgpu/bioemu2/host.js` RE-EXPORTING them, so that
backend's behaviour cannot change: every caller there gets the same function objects it had before.
That is the one move whose correctness needs no gate, which matters because the only thing checking
those stages anywhere is `check-bioemu2-network.js` against `dump_bioemu2.py`'s dump and **that dump
is not on this box**.

🔴 **AND THE EMBEDDER'S REFERENCE IS THE OTHER BACKEND, IN THE SAME PAGE.** `cpu/bioemu2/` holds the
denoiser and its tape and nothing else, so `check-webgl-bioemu2-embedder.js` drives the WGSL
embedder - f16 weights off, same int5 bundle, same features - and reads **4.2e-7** on `s_inputs`.
Splitting that arm is what found the one bug: the host's `windowBias` is **head-MAJOR** and this
port's attention reads head-MINOR, which showed as relRMS 0.118 on the pooled half with the 71
featuriser columns exact.

The trunk is composition over kernels this tree already has - the pair-only block is ESMFold2's
`foldingTrunk` and the MSA layer's outer product mean is AF2's generalised one - and its reference
is the same WGSL path in the same page.

### The fold, and why it is a differential and not a geometry floor

`fold-bioemu2-webgl.js` on `6mrr-68.json`, 68 tokens and 573 atoms, four trunk passes, all 52 pair
blocks, 20 steps, seed 1, against the 6MRR crystal:

```
                     WebGL2 (ported)  WebGL2 (now)  WGSL (same job, seed, steps)
rmsd / TM            1.595 / 0.9083   1.595 / 0.9083   1.595 / 0.9083
CA-CA median         3.8540558        3.8540558        3.8539739   (a relative 2.1e-5)
worst CA-CA          3.904            3.904            3.904
bond rms  main       0.042            0.042            0.042
          side       0.056            0.056            0.056
          peptide    0.028            0.028            0.028
contacts (P > 0.5)   201              201
wall                 968 s            57.6 s           3.9 s
  trunk              277.7 s          21.6 s           2.3 s
  distogram                           0.7 s
  20 denoiser calls  658 s            35.2 s           1.2 s
weights/activations  1656 MiB peak    409 MiB peak     815 MiB weights + 135 activations
```

🔴 **968 s TO 57.6 s ON A QUARTER OF THE MEMORY, AND NOT ONE REPORTED QUANTITY MOVES.** The 16.8x
is the row block and its fragment order, the contraction split, coalesced, transposed and
pair-blocked, and the weights held as the bundle's own five-bit codes from "Closing the gap" above, and every figure in
the middle column is the left column's - the same rmsd, the same TM, the same CA-CA median to eight
digits, the same 201 contacts - while the peak HALVES. Four kernels reordered or re-quantised and
the fold does not notice; the weights are within 36 MiB of what the WGSL fold holds.

🔴 **AND A MIDDLE COLUMN THAT DID NOT MATCH IS WHAT FOUND A BUG.** Before `Storage.zeros` landed
this read TM 0.9081 and **209** contacts, which looked like a tolerable consequence of a reordered
sum and was not: bioemu2's first trunk pass was reading texels nothing had written. The 201 is the
evidence that it is fixed - see "A texture pool, and the bug it found".

🔴 **THE SAME SEED THROUGH BOTH BACKENDS LANDS ON THE SAME STRUCTURE, which makes this a WHOLE-FOLD
DIFFERENTIAL and not a plausibility check.** The sampler's draws come from one generator consumed in
one order, so the walk is identical given a denoiser that computes the same function - and the two
agree on every scored quantity to the digits reported, with the CA-CA median differing by 2.1e-5
relative. A geometry floor would have accepted a port that was wrong by 10%; this does not. It is
also why the AF3-style "worst atom" figure is not quoted here: it needs both PDBs, and the WebGL2
side is sixteen minutes a fold.

🔴 **AND IT IS 15x THE WGSL FOLD - WAS 246x - ON LESS MEMORY THAN THAT FOLD USES.** The trunk is
9x and the diffusion 30x, and the diffusion is where the remaining asymmetry lives: the WGSL denoiser
batches S samples through one compiled program, which this one does not. **The half-precision
weights this file once called "the next multiplicative factor" are taken** - see "Float16 weights"
above - so the denoiser's 1.6 GiB is 0.8 and a call is **2.5 s where it was 32.9**. That is also
why the tool's default is 20 steps where the model's is 200, and why a fold at 20 steps is a
plumbing answer that happens to score well rather than a converged sample.

🔴 **AND THE DEFAULT ORDER OF THE WEIGHT UPLOADS IS LOAD-BEARING.** `structure_module` is 424.6 M of
the checkpoint's 511.7 M parameters - 1.62 GiB in f32, ~101 MiB a layer over 16 layers - where the
trunk's 52 pair blocks are 288 MiB. The driver releases the trunk's weights before it reads the
denoiser's, which costs nothing because no sampler step touches the trunk, and keeps the peak at
**1656 MiB** instead of 2.3 GiB - and at **851 MiB** since the weights went to float16, which is
within 36 MiB of what the WGSL fold holds. The row block and the split contraction each hold an
intermediate while they run and pushed it to 2158 MiB in between, which is why that figure appears
in this file's history; the release ordering is what stops it being 3 GiB either way.

🔴 **AND THE COMPOSITION FOUND A BUG FIVE STAGE GATES COULD NOT**, which is the same argument the AF3
driver made: `concatSource` had only an ALIGNED path, so the real single conditioning - `[s_trunk |
s_inputs]`, 384 + 455 - compiled `const int R4 = 113.75` and failed with "cannot convert from const
float to const highp int". 455 is the featuriser's 384 pooled channels plus 71 one-hots, a number
with no reason to be a multiple of anything, and every gate in this family drives the stage at
synthetic widths that are multiples of four. It has a ragged path now. The ragged/aligned split is
the fourth kernel in this tree to need one.

🔴 **AND TWO TRAPS COST A FOURTEEN-MINUTE RUN EACH, BOTH RECORDED IN THE TOOL'S HEADER.**
`gpu-chrome.mjs` gives a module **ten minutes** (`LOCALFOLD_GPU_TIMEOUT_MS`), so the first fold died
with no JSON at all; and piping the run through `tail` reports that timeout as a SUCCESS, because the
pipe's exit status is tail's. Then `6mrr.json` is 71 residues where the shipped crystal resolves 68,
so the second fold finished and died in the scorer - the crystal is counted BEFORE the fold now,
since `alphaCarbonSpacings`'s centres depend on the atom keys and the topology rather than on the
coordinates.

## What it would take to ship it, and what is still unmeasured

🔴 **THIS SECTION IS THE QUESTION AS IT WAS ASKED BEFORE ANY OF IT WAS WRITTEN, AND ITS FIRST
SENTENCE HAS GONE STALE ONCE ALREADY.** It said "`webgpu/` is 250 `@compute` entry points. Two are
ported", which was true when two were; all four models in the goal fold now, and this tree does not
mirror those entry points one for one - a fragment shader replaces whole families of them, and one
68-token bioemu2 fold compiles **110** programs in total (it was 144, then 168 as more shapes
arrived, and the projection's shapes are uniforms now - see "Closing the gap"). What the three measurements below decide is
no longer whether to start but whether to SHIP, and all three are still open:

1. **Coverage.** What fraction of the no-WebGPU population actually has WebGL2 *and*
   `EXT_color_buffer_float` *and* `MAX_TEXTURE_SIZE` large enough *and* the vertex output budget?
   `tools/gpu/probe-webgl2-limits.js` asks exactly this and has only ever run on development
   machines. Nobody here has the number.
2. **The 47.8% on old hardware.** Re-run `tools/gpu/bench-webgl2-tf-matmul.js` on an actual old
   integrated GPU rather than the A100. This is the measurement the case rests on and the one
   least likely to transfer. 🔴 **AND IT IS NOW THE ONLY REMAINING SPEED LEVER**, because the
   fragment path is at its intensity ceiling - see "How close can this get". Transform feedback is
   65% of WebGPU where the fragment path is 30%, so it is the difference between a 1.5x gap and a
   3.3x one; it is also the path old drivers are worst at, which is the whole tension.
3. **Whether the pair track can dodge the texture axis** - tiling `tokens^2` across array layers
   (`MAX_ARRAY_TEXTURE_LAYERS`, which the limits probe already reads). If it can, the ceiling
   moves and a length-generous goal is possible; if not, this backend is capped near a small
   domain and should say so in its first sentence.

And **weights are a separate wall from kernels**: the bundles run 73 MiB (`model-af2-monomer-int5`)
to 2.1 GiB (`model-rfd4p-int8`), and an old machine with 4-8 GiB of RAM is not folding from the
large end whatever the kernels do. Only the small end is browser-shaped on the hardware this tree
exists for.

🔴 **THAT WALL MOVED FURTHEST OF ANYTHING HERE.** A bioemu2 fold's device peak was 1656 MiB; the
float16 weights took it to 851, which is within 36 MiB of what the WGSL fold holds, and the
bundle's own five-bit codes decoded in the kernel take it to **409 MiB - less than half what the
WGSL backend uses for the same fold**. The memory question is no longer the binding one for that
model, and on this axis the fallback backend is now the efficient one.
The next factor is the bundle's own quantisation, and it is worth writing out because the format is
already settled and the plumbing already exists. `model-bioemu2-int5` ships five-bit CODES; this
backend decodes them to f32 on the host and uploads the result (two bytes a weight since the
float16 work above). Decoding them in the shader instead would be **bit-identical** - the decoded
value is the same f32 number the host computes - and would cut the fetch count that bounds every
big projection again:

- the format is **group 32, asymmetric**: 32 codes in 20 bytes (32 x 5 bits is exactly 160), one
  float16 `scale` and one float16 `zero` a group, and a value is `zero + code * scale`. A code
  never spans more than two bytes. `readTensor`'s int5 branch in `shared/weights/dtype.js` is the
  reference, and its own comments record that a bundle at any other group decodes into "a finite,
  plausible, wrong tensor".
- `HttpTensorStore.tensorSource(name)` exists for exactly this - "THIS IS WHAT LETS A DECODER RUN
  SOMEWHERE ELSE... a caller that wants to decode on the GPU needs the CODES, and the scale and
  zero tables beside them".
- the arithmetic favours it. A weight's codes run along its CONTIGUOUS axis, which for an
  `[out][in]` weight is the k direction a fragment sweeps, so a k-step of 32 reads one group: two
  RGBA32UI texels (five 32-bit words padded to eight) plus one for the scale and zero, against
  float16's **eight** texels for the same 32 values. **2.7x fewer fetches**, and this kernel is
  fetch-bound.
- it needs a differential before it needs a benchmark. `tools/gpu/check-quantised-upload.js` found
  the WGSL device decoder and the host one disagreeing on **104,170 of 131,072 elements** at int5
  group 64, and it was the HOST that was wrong - so the gate is "every element equal to
  `readTensor`'s", not a tolerance.

That is what the WGSL side's device packer already does, and it is the largest
measured-but-untaken item in this file.

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
node tools/gpu-chrome.mjs tools/gpu/bench-webgl-linear.js      the dense projection at the shapes four models fold at, one texel a fragment against eight rows
node tools/gpu-chrome.mjs tools/gpu/bench-webgl-upload.js      what weight bytes cost on the way into a texture, and a readback the other way
node tools/gpu-chrome.mjs tools/gpu/check-webgl-int5-decode.js the in-shader five-bit decoder against readTensor, element by element; --falsify
```

Every fold driver here takes the diagnostic switches in `tools/gpu/webgl-execution-options.js`,
which is where they are documented: `--skip-draws` (everything but the drawing - the arm that
settles what the GPU is costing), `--double=<regex>` (a family's draws issued twice; **read its
header before trusting it**), `--nan-watch` (names the kernel that reads a texel nothing wrote),
`--host` (where the host time inside a dispatch goes), `--no-pool`, `--no-check-errors`, and the
three control arms for the optimisations above: `--no-linear-block`, `--no-triangle-split`,
`--no-triangle-coalesce`, `--no-triangle-transpose`, `--no-half-weights` and `--no-int5-weights`
(which nest: dropping the codes falls back to float16, dropping both to float32).

## Making it selectable

Two lines: `tools/build_site.py`'s `DIRECTORIES` does not copy this tree, so it is not on the site;
and the page requires WebGPU rather than choosing a backend.

🔴 **THE PRECONDITION IS NOW MET AND THE DECISION IS STILL NOT AUTOMATIC.** This section used to say
neither line should move "before a fold actually runs here", and one does. What has not been
settled is whether a backend **4.6x slower on AF2 and 23x on bioemu2** should be offered: a
59-residue AF2 fold is 1.29 s against 0.28 s, 68-token bioemu2 is 58 s against 3.9 s on 409 MiB
against 950, and a
255-residue fold has not been measured at all. A visitor with no WebGPU gets a fold
instead of a refusal, which is the whole point of the tree; a visitor who waits four minutes for
something the other path does in fifteen seconds may not thank anyone. The multimer is also still
absent, so the page would have to offer a monomer-only backend. Those are product questions, so
they are written down here rather than decided in a commit.

# LocalFold on WebGL2

The fallback backend, for the browsers and machines that have no WebGPU. `webgpu/` is the page's
backend and needs Chrome/Edge 113+ or Safari 26+ - which the page says in those words
(`web/design.js`: "WebGPU unavailable — this page needs Chrome or Edge 113+, or Safari 26+, to
fold"). WebGL2 has been in Safari since 15 and in Chrome since 56, so it reaches machines and
operating systems that the WebGPU path refuses outright.

🔴 **THE PAGE SELECTS THIS BACKEND NOW, FOR af3, openbind0 AND intellifold2** - `foldBackend`
in web/model.js, `WEBGL2_FAMILIES` in shared/bundles/manifests/index.js, measured through
`tools/fold-in-page.py --no-webgpu` (AlphaFold 3 pLDDT 68.0 in 11 s against WebGPU's 68.2 in 1 s).
The four fused-template bundles and opendde are refused by name before the download; see
webgl/index.js. **AlphaFold 2 folds here - both graphs - and the PAGE still does not select it**,
because web/ has no AF2 entry point that takes a backend.

🔴 **ALPHAFOLD 2 FOLDS HERE NOW - BOTH GRAPHS.** 59 residues from
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
path on AF2 and 12x on bioemu2** - it was 17x and 246x - and the shapes are **worse**.

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
WGSL side. 🔴 **READ THAT `programMs` AS A FIRST VISIT AND NOT AS A PROPERTY OF THE FOLD**: the same
measurement on a machine that has built those shaders before is 70 ms, for the reason the next
section opens with.

### Four models' worth of programs, and what a link costs

🔴 **EVERY COMPILE NUMBER IN THIS FILE TAKEN AFTER THE FIRST RUN OF A SHADER ON THIS MACHINE IS A
RETURNING VISITOR'S, AND THAT IS A FACTOR OF THIRTY.** An AF2 WebGL fold compiled its 53 programs
in **1507 ms** the first time those shaders had ever been built here and in **70 ms** on every run
after - through a FRESH Chrome profile each time, because `gpu-chrome.mjs` makes one per run and
throws it away. Metal keeps its compiled shaders per USER, outside that profile, so the profile
wipe this repository relies on to measure a first visit does not reach the thing that matters.
`tools/gpu/probe-webgl-programs.js --cold` is the honest number: it salts every shader text so no
cache can answer. 🔴 **AND THE SALT HAS TO BE SEMANTIC** - the first version prepended a COMMENT
and measured exactly the warm number, because ANGLE translates GLSL to MSL and a comment does not
survive the translator. A `discard` guarded by a unique constant against `gl_FragCoord.x` does, and
never runs. `probe-webgl-uber.js` had the same hole: its salts were `rep * 0.001`, so its second
run asked for shaders its first run had built - 20 kernels in one program read **50 ms on the first
run of the probe and 2 ms on the next**, with nothing changed.

**A link costs ~41 ms a program on this M2's Chrome, and almost nothing extra for what is in it.**
Cold, through `probe-webgl-uber.js`: 4 small kernels as 4 programs are 151 ms and as one program
41; 8 are 306 and 43; **20 are 762 and 51**. So the shader cost of a first fold is the program
COUNT times a constant, which is the same shape Safari 16.3 shows at ~19 ms a link - and it is why
the lever is combining kernels rather than shortening them.

**So every model's single-target kernels now share programs, and every fold tool asks for them up
front.**
Two mechanisms, both already here for AF3: a dry run on virtual tensors says which programs a fold
draws with (`Execution.collectPrograms`: no textures, no draws, reads are zeros), and
`Execution.prepare` combines the parts among them (`Execution.combine`) and links what is left.
Measured cold, with the fold's programs linked ONE AT A TIME in a fresh context - which is what a
browser that links on its own thread pays, and what Chrome's `KHR_parallel_shader_compile` hides:

| fold | programs | linked one at a time | and combined | | |
|---|---:|---:|---:|---:|---:|
| AlphaFold 2 monomer, 59 residues | 53 | 2178 ms | **16** | **807 ms** | 2.70x |
| AlphaFold 3 / OpenBind-0, 59 tokens | 83 | 3291 ms | **23** | **1108 ms** | 2.97x |
| ESMFold2, 59 residues | 82 | 3247 ms | **10** | **535 ms** | 6.07x |
| bioemu2, 68 tokens | 114 | 4707 ms | **14** | **938 ms** | 5.02x |

Both arms, two interleaved rounds, the minimum of each; the spread is under 7% on every row but
AF2's own 807/862. A first reading had bioemu2 at 4866 ms and 5.28x from one run.

What is left in every one of them is the kernels a part cannot be: the heavy projections and
attentions, which write SEVERAL render targets where a part writes `result` (8, 11, 3 and 9 of
them), and AF2's `geometry.frames`, which declares a struct. Everything else is in a combined
program. The draws are cheaper too - **6 to 16%** off each fold's own clock (AF2 1909 ms to 1784,
AF3 3031 to 2843, ESMFold2 1986 to 1675, bioemu2 2759 to 2421), from the program switches a
dispatch no longer makes.

Nothing any of them computes moves. AF2's worst atom against the WGSL fold is **0.0353 A** either
way (its own dither floor is 0.0308); AF3's whole comparison is identical to the digit in both arms
- worst atom 0.0301 A, relative rms 0.002689, trunk pair 4.72e-4 - at pLDDT **66.525**; ESMFold2 is
**83.632** with identical CA spacings; and a 20-step bioemu2 fold is **rmsd 1.595, TM 0.9083, 201
contacts** at a 411 MiB peak, every digit the same. `--no-combine` is the control arm in all four,
and `check-af3-webgl-page.js` - the page's own WebGL2 fold against its WebGPU one - reads the same
0.1371 A either way.

🔴 **AND THE BLOCKER WAS NEVER THE SHAPE OF `part`, IT WAS THAT PARTS SHARE ONE NAMESPACE.**
Written as a part a kernel can share a program; this tree has ~150 kernels and seven of them were
parts, so the round looked like 150 hand conversions. What actually stood in the way was that a
finished kernel carries its own names: the WIDTHS `sized` deliberately leaves as `const int` so a
loop keeps its compile-time bound (in 37 of AF2's 53 programs and 69 of ESMFold2's 82), and its own
helper functions - and two kernels cannot both declare `CH4`, or `coordinateOf`, in one program. So
`partFromSource` reads a finished kernel back as a part and RENAMES its own declarations by a
digest of its source, with every reference to them in its uniforms, helpers and body; the uniforms
keep their names, which is what lets two kernels share a sampler unit. **Not one kernel was
rewritten and none lost a bound.** 🔴 The first version moved the consts into the body instead,
which is also correct and also keeps them compile-time - and it cannot touch a kernel with a
HELPER, which is most of the interesting ones: AF2 went to 36 programs that way and to **16** with
the names renamed. What it refuses now is a kernel with SEVERAL render targets, since a part writes
`result`, and one whose module scope holds anything else at all - a struct, a global.

🔴 **AND A COMBINED DRAW'S SPARE SAMPLERS NEED A PLACEHOLDER PER TYPE.** The samplers a combined
program has but this draw does not use are pointed at a one-texel texture, because one left on a
unit holding this draw's render target is a feedback loop. Two samplers of DIFFERENT types on one
texture unit is `GL_INVALID_OPERATION`, and the moment a bin held an int5 kernel's `usampler2D`
beside a float kernel's `sampler2D` the fold died at `bioemu2.fourier-add:68x768: GL error 0x502
after the draw` - the same error as the pool bug, and again from a binding rather than from
arithmetic. There is a float placeholder and a words one now, on their own units, which is also why
`combine` keeps two sampler units back from `MAX_TEXTURE_IMAGE_UNITS`.

🔴 **AND A KERNEL THAT LEANED ON A UNIFORM'S ZERO WOULD HAVE READ ANOTHER KERNEL'S VALUE.** On its
own a program's scalar uniforms start at zero and a kernel may rely on that: `triangle.contract`
ends `result = sum + bias` and **no caller has ever set `bias`**. Sharing a program takes that zero
away - the uniform keeps whatever the last draw of that program left in it - and the symptom is a
wrong number with nothing to point at. Measured rather than assumed: **112 to 1159 draws a fold**
leave one of their own declared scalars unset (mostly `scale`, which the shared elementwise part reads in
one branch of five, and `bias`), so `dispatch` gives every one of them its zero back and `stats`
reports the count by kernel and name. The folds are identical either way, which says this was
LATENT and not live - `bias` is the only `bias` in its bin, so nothing had ever written it - and
latent by luck of which kernels happened to share a program.

🔴 **AND HALF-COMBINING WAS A LOSS ON CHROME, WHICH IS THE PARALLEL COMPILE AND WORTH KNOWING.**
`prepare` starts every link at once, and with `KHR_parallel_shader_compile` Chrome runs them off
the thread: AF2's 53 programs come back in **519 ms**, under 10 ms each. At 36 programs - the
consts-into-the-body version, which left every helper-declaring kernel on its own - it took
**1226**, because merging reduces the parallelism and lengthens the longest job while saving a
fixed cost the parallelism had already hidden. At **16** it is **292 ms**, so the count had to fall
far enough; every model now improves on both paths (ESMFold2 740 ms to 194, AF3 720 to 383,
bioemu2 1200 to 595). The serial column above is still the one this backend is for - a visitor on Chrome
has WebGPU, and the browsers that bring anyone here link inside `linkProgram` on the page's thread,
which is where a 6x shows up as a tab that does not freeze. `--no-combine` is the control arm.

Two arms were measured and rejected on the way: `--combine-bytes=<n>`, a cap on how much GLSL one
program may hold, because AF2's compile rose monotonically with every cap swept (514 ms at 53
programs, 739 at 38, 968 at 37, 1223 at 36) while ESMFold2 was best with no cap at all; and leaving
the LOOPING kernels out, on the theory that a dozen unrolled bodies in one program is what costs
the driver, because it was worse serially in both (AF2 1758 ms against 1557, ESMFold2 1443 against
1023). Both were measuring the half-combined state, and what the second round says is that neither
was the mechanism: the cost is per program, and the cure is fewer of them.

**`getError` after every draw is a synchronous round trip** - 4% of an AF2 fold - and
`checkFramebufferStatus` per dispatch is another. A framebuffer per attachment SET, bound rather
than rebuilt, removes the second; a fold reuses 65 of them because the textures are pooled.

### What it is worth, end to end

Every fold is the same job, the same seed and the same settings either side; `--no-linear-block` and
`--no-triangle-split` are the control arms, and every one of the 47 stage gates is green both ways.

| fold | before | after | |
|---|---:|---:|---:|
| bioemu2, 6MRR 68 tokens, 4 passes, 20 steps | 968 s | **20.7 s** | **46.8x** |
| ...its trunk | 277.7 s | **21.6 s** | 12.8x |
| ...its 20 denoiser calls | 658 s | **23.3 s** | 28.2x |
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

### A row that wraps, which was the largest factor in the whole campaign

🔴 **A LOGICAL ROW MUST START WHERE A TEXTURE ROW DOES, AND NOTHING HERE MADE IT.** A row of `IN4`
texels begins at `row * IN4`, and `fetch4` puts that at `(t % 2048, t / 2048)` - so unless `IN4`
divides 2048 the row WRAPS onto the next texture row at a different offset for every row. A texture
is stored swizzled for two-dimensional locality, so a row that wraps is a row whose neighbours are
not neighbours, and the twelve streams a blocked projection walks defeat the cache together.

Swept at bioemu2's dominant shape, 576 x 896 -> 1792, with the answer identical in every arm:

| `in` | `IN4` | GFLOP/s |
|---|---:|---:|
| 768 | 192 | 31.5 |
| 896 | 224 | **25.0** |
| **1024** | **256** | **95.2** |
| 1152 | 288 | 25.6 |
| 1280 | 320 | 23.0 |
| 1792 | 448 | 21.1 |
| **2048** | **512** | **113.3** |

**Only the powers of two are fast**, 2048 being one, and `in = 1024` does MORE arithmetic than 896
while running four times faster. So the fix is a **stride and not a width**: lay the row at 256
texels and keep the loop at 224, and not one extra multiply-add is done. **14% more memory for
4.08x**, bit-identical - `webgl/runtime/pad-stride.js`, `--no-stride-pad`.

🔴 **AND IT IS x, NOT THE WEIGHT, WHICH IS WHY IT IS CHEAP.** Padding x alone is the whole **4.08x**
and padding the weight alone is **0.94x** - so it is one conversion pass over the activation, 0.2 ms
against 60 of cache misses, and it cascades into nothing. (The same plumbing packed as float16
halves was a 0.93x loss; the transform is what differs, not the machinery.)

🔴 **AND IT IS THE SIBLING PORT'S `TW = 384` FROM THE OTHER SIDE, WHICH THIS FILE HAD ALREADY READ
AND MISUNDERSTOOD.** Its width is 384 "because the row widths the fast paths walk (24, 48, 64, 96,
128, 192, 384 texels) divide it, so no such row wraps onto the next texture row". It chose the WIDTH
to suit its rows; this chooses the ROW to suit the width, and they are the same property. The
narrow-row experiment above measured a 3.8x LOSS and was read as closing the question - but
narrowing `W` makes MORE rows wrap, not fewer. The condition was never the width's size, and that
negative result had the answer in it.

🔴 **AND THE CONTRACTION WALKS A ROW TOO AND DOES NOT WANT IT - MEASURED, AND DECLINED.** AF3's
triangle contraction reads operands laid `[c][i][j]` with j contiguous, so at 68 tokens a row is
**17 texels**, which is exactly the condition above and does not divide 2048. Padding both operands
to 32 texels (a pad pass each, the transposed incoming direction absorbing it into its own pass) is
a **LOSS at both scales**, on the bioemu2 trunk at 68 tokens with the fold bit-identical in every
arm:

| mixer blocks | padded | at its own stride |
|---|---:|---:|
| 4 | 3090 / 3074 ms | **2978 / 3004 ms** |
| 48 | 21918 / 21896 ms | **21511 / 21580 ms** |

**So the condition is necessary and not sufficient, and the difference is HOW MANY STREAMS walk the
rows.** The projection's blocked kernel walks twelve rows at once and they evict each other, which
is what made a wrap catastrophic there. The contraction's fragment walks its six rows (two `a` and
four `b`) ALONG k, contiguously, so a wrapped row costs one miss when the row is entered rather than
one per fetch - and against that, padding 17 texels to 32 is **1.88x the operand bytes** plus two
passes over 4.7 MiB. The 4.08x and this 0.98x are the same addressing property read at two
arithmetic intensities.

🔴 **AND THE CHECKER WAS GREEN ON THREE KERNELS IT HAD NEVER RUN, WHICH IS THE REAL FINDING HERE.**
`check-webgl-af3-triangle.js` swept lengths **5 and 11** - chosen for the odd-width bug that the
WGSL triangle once had - and the WebGL2 contraction selects its kernel on `L % 4 === 0`: the
pair-blocked one, the coalesced `fetch4` along k, and the transposed incoming path are all
unreachable at an odd length, so twenty-four green arms had only ever exercised the
one-float-at-a-time fallback. The padding above is what exposed it, by reddening 12 and 68 at relRMS
1.4 while 5, 11, 16 and 32 stayed green - **a gate that passes at every length the gate drives and
fails at every length a fold uses**. The default is `5,11,12,68` now and the shipped kernels are
green at all four and under all five control arms. The lesson is the one this tree keeps relearning
in a new place: a gate whose inputs cannot reach the code path is not a gate, and here the
unreachability was arithmetic rather than a flag.

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

### The fold's whole arithmetic, and why 670 GFLOP/s is not a ceiling it can be held to

**The one calculation that decides whether the remaining gap is structural, and it needed counting
rather than arguing.** Summed from a bioemu2 fold's own profile - every `linear*` label's
`2 * rows * in * out` times its call count, plus the contraction's `2 * L^3 * channels`:

| label | GFLOP | achieved |
|---|---:|---:|
| `linear.block.int5:576x896x1792` out-major | 1184 | **175 GFLOP/s** |
| `linear:4624x256x512` in-major, one-texel | 1008 | 323 |
| `linear.block:4624x256x2048` in-major | 1008 | **670** |
| `linear.block.int5:576x768x1920` | 544 | 169 |
| `linear.block:4624x1024x256` in-major | 504 | 186 |
| **the countable total** | **5081** | **167 average, in 30.4 s** |

🔴 **THIS TABLE IS OF ITS DATE AND THE FIRST TWO ROWS ARE SUPERSEDED** - `576x896x1792` is **375
GFLOP/s** now and `576x768x1920` **394**, both tiled and on a float16 x. The rates here were also
taken under `--profile`, whose per-call wait is most of what it measures, so they compare within a run
and not across them.

🔴 **SO ONE SHAPE REACHES 670 GFLOP/s AND THE FOLD AVERAGES 167, AND THE FIRST READING OF THAT WAS
WRONG.** Dividing 5081 by 670 gives 7.6 s and the appearance of a 4x of headroom - and the fragment
counts seemed to explain it, since the 670 shape has **295,936** fragments where
`576 x 896 -> 1792` has **32,256**, a 9.2x deficit at a 3.8x deficit in rate.

**Built and measured: a k split, folded into the fragment index so it costs no extra draw, which
multiplies the fragments by four or eight. It is 1.5x SLOWER** - the cheap bioemu2 fold 6073 ms to
9265, the sample stage 3824 to 6245. The reason is what the split must buy the fragments with: more
fragments in ONE draw needs more output slots, so `splits` times the intermediate planes, and at
`576 x 896 -> 1792` that is **132 MB of partials written and read back per call** against an output
of 4 MB. Accumulating across draws with `EXT_float_blend` instead does not help, because each of
those draws still has the same 32,256 fragments - the parallelism only rises if the splits are
concurrent, and then they need the planes.

🔴 **AND THAT SETTLES WHAT 670 MEANS: IT IS A SHAPE'S RATE, NOT THE BACKEND'S.** The 670 kernel is
`[in][out]` float16 with an aspect ratio of 8, which is exactly the case the blocked geometry suits;
the int5 kernels are balanced shapes whose cost includes the DECODE - about fifteen integer
operations of address arithmetic per eight weights, which this file records as the reason the first
int5 projection was 23% slower than the float16 it replaced.

🔴 **AND THE TWO CONCLUSIONS THAT USED TO FOLLOW HERE ARE BOTH RETRACTED, BY MEASUREMENT.** They
read: *"Those kernels are ALU-limited on the decode, not fetch-limited, so no fragment count reaches
670 for them"* and *"5081 GFLOP in 30.4 s is at the floor for this mix of kernels"*.

**The decode is not the limit.** `probe-webgl-linear-operand.js` aliases the weight codes so every
fragment reads the SAME four code texels - removing every code-fetch miss while leaving every decode
instruction in place - and the kernel does not move: **1.09x, and 1.01 and 0.85 on other runs**, all
inside its own spread. Meanwhile the arm that aliases **x**, doing the identical decode, goes from
10.06 ms to 2.93 and from 184 GFLOP/s to **632**. A kernel that reaches 632 GFLOP/s with the same
decode arithmetic is not ALU-limited on that decode; it is limited by x's fetch locality, and the
inference from "the decode is expensive" to "the decode is the limit" was never measured.

**And 30.4 s was not the floor**: the same mix of kernels, same weights, same step count, folds in
**21.7 s** - 22.4 s with the projection work alone, then the 4x8x1 contraction block - with
`576x896x1792` going **175 to 375 GFLOP/s** and `576x768x1920` to 394. See "The dense projection is bound by x's fetch locality" below
for all of it. What was right in the retracted paragraph is narrower and still holds: **670 is a
shape's rate rather than the backend's**, and the honest floor is the sum over labels of each label's
own achievable rate - it is just that two of those rates were 2.1x higher than this section assumed,
and a floor computed from unmeasured rates is an estimate wearing a floor's clothes.

### 🔴 A UNIFORM BLOCK READ IS 2.1x A `texelFetch`, SO THE CEILING IS A TEXTURE CEILING

**This corrects the section below, which says the texture unit is the only path. It is not.** GLSL ES
300 has UNIFORM BLOCKS, they are dynamically indexable, and they are a different memory path - a
constant cache rather than the texture unit. Nothing in this backend has ever used one: every
operand of every kernel here arrives through `texelFetch`.

`tools/gpu/probe-webgl-uniform-reads.js` runs the identical loop over the identical data both ways,
with its own GL setup so none of the runtime is in the way. Three consecutive runs, eight
interleaved rounds each, minimum per arm:

| | G reads/s | spread |
|---|---:|---:|
| `texelFetch` | 83.2 / 83.6 / 84.9 | 1.04-1.07 |
| **uniform block** | **179.0 / 177.5 / 179.0** | 1.09-1.35 |
| | **2.11x to 2.15x** | |

So **21.33 flops a fetch is the ceiling for a kernel whose operands all come through the TEXTURE
unit**, which is every kernel here, and not an absolute one. A projection whose shared operand sat in
uniforms would fetch only the streaming one.

🔴 **AND THE CONSTRAINTS ARE WHAT MAKE IT MODEST RATHER THAN DECISIVE, so they are stated before the
mechanism is believed.** This context reports `MAX_UNIFORM_BLOCK_SIZE` 16384 bytes and
`MAX_FRAGMENT_UNIFORM_BLOCKS` 16, so **65,536 floats** can sit in uniforms for one draw - and the
operand has to be CONSTANT across that draw, because a UBO update costs a draw, and this backend's
host cost is about 0.57 ms a draw. So:

- bioemu2's `[in][out]` projections (13.1% of its fold) have weights of 131,072 floats: they fit in
  **two** draws, and per k-step the kernel would do eight texture fetches and four uniform reads
  instead of twelve fetches - about **1.2x** on those kernels, **~2.5% of the fold**;
- an operand that changes per row block does NOT work: x would need 72 UBO updates a projection,
  which is 41 ms of draw overhead against the 10 ms the projection takes;
- and the **hot int5 weights do not fit at all** - `576 x 896 -> 1792` is 1.6 million codes - so the
  48.6% of a bioemu2 fold that matters most cannot use this.

🔴 **AND PRICED AGAINST THE DRAW IT COSTS, IT IS NOT WORTH BUILDING - WHICH IS WHY IT IS WRITTEN
DOWN RATHER THAN BUILT.** A UBO that does not fit in one draw costs a draw per chunk, and `--host`
puts this backend's draw at **0.504 ms** (466 ms of `bindMs` plus 2554 of `drawMs` over 5,989
draws). So:

| | fetches a k-step | extra draws | saved a call | cost a call | net |
|---|---|---:|---:|---:|---:|
| `4624x256x512` in-major, 13.05 ms a call | 8 texture + 4 uniform against 12 | +1 | 2.2 ms | 0.50 ms | **+1.7 ms** |
| `576x896x1792` int5, 10.0 ms a call | 16 texture + 5 uniform against 21 | **+3** | 1.2 ms | 1.5 ms | **-0.3 ms** |

The in-major case nets 1.7 ms over 208 calls - **340 ms of a 30.4 s fold, 1.1%** - and the int5 case,
which is 48.6% of the fold, is a **net LOSS** because its 1.6 million codes need four draws to pass
through 65,536 floats of uniforms and three extra draws cost more than the fetches save. So the
honest figure for this path is **about one percent**, not the "~2.5%" a per-read ratio suggested
before the draws were counted.

**And untested in a kernel even so**: the packed-halves weight had an equally sound mechanism (32
flops a fetch against 21.33, measured, at no accuracy cost) and still lost 0.78x because the
`unpackHalf2x16` ALU exceeded the fetches it saved. A per-read ratio is not a kernel result, and a
per-read ratio minus a draw count is still not one.

What is established is narrower and worth keeping: **the bound below is a texture-unit bound and not
an absolute one**, there is exactly one other path, it is 2.1x per read, it holds 65,536 floats a
draw, and once the draws are priced it is worth about a percent on the one fold that can use it.

### 🔴 RETRACTED: A FRAGMENT GEMM REACHES 2 TFLOP/s HERE, SO THE API IS NOT THE LIMIT

**Everything below this heading that calls the remaining gap structural is wrong, and this is the
measurement that settles it.** `tools/gpu/probe-webgl-gemm-ceiling.js` runs a GEMM in the best case
this backend admits - the optimal 8-by-4 tile into 8 render targets, 32 output floats, L1-resident
operands, power-of-two strides, a long k loop, 262,144 fragments - and measures what the fragment
pipeline can actually do:

| | |
|---|---:|
| **best-case fragment GEMM** | **1958 - 2003 GFLOP/s** (spread 1.02 - 1.18 over three runs and k = 256, 512, 1024) |
| its fetch rate | **91.8 G fetch4/s** at exactly 21.33 flops a fetch |
| what parity needs | 5,081 GFLOP of bioemu2 arithmetic in the WGSL fold's 3.9 s = **1,303 GFLOP/s** |
| what the fold averages | **167 GFLOP/s** |

**So the output budget is not the binding constraint, the fetch issue rate is not the binding
constraint, and parity is not ruled out by anything in the API.** The hardware will do two teraflops
through a fragment shader with the 32-float tile. The fold does 167. **The entire remaining gap is
LOCALITY** - a real GEMM's fragments must read disjoint rows, and that is worth a factor of twelve
here.

🔴 **AND THIS IS THE THIRD BOUND IN THIS FILE TO BE RETRACTED, EACH FOR THE SAME REASON: THE NUMBER
MULTIPLIED BY THE ARITHMETIC WAS NEVER MEASURED.** The first inferred the fetch rate from the
shipped kernels (6x too low). The second measured it under locality no GEMM can have and read the
difference as headroom (over-read). The third - mine, above - concluded from the same figures that
the gap was a property of WebGL2 and said so repeatedly. **The arithmetic was right every time and
the factor it was multiplied by was guessed every time.** The probe is cheap; it should have been the
first thing written, not the last.

What this redirects the work to: the two changes that moved these kernels most were already
locality - the contraction's fragment order (1.49x) and its tile-aware mapping (3.29x) - and the
fetch-rate probe says scattered reads get **19 to 32 G fetch4/s** where this best case gets **92**.

🔴 **AND THE 2D FRAGMENT ORDER WAS THEN BUILT FOR THE PROJECTION AND DOES NOT PAY.** A
`TILE_BLOCK x TILE_BLOCK` block of fragments covering that many row blocks and output groups, the
repack inverting it (the fold's CA-CA identical throughout, so the mapping is right), swept over the
block size against the fragments the tiling rounds up:

| | waste | sample stage | fold |
|---|---:|---:|---:|
| flat (shipped) | 1.00x | **3824 ms** | **6523 ms** |
| `TILE_BLOCK` 8 | **1.02x** | 4981 | 8291 |
| `TILE_BLOCK` 16 | 2.03x | **3725** | 6716 |
| `TILE_BLOCK` 32 | 2.03x | 4083 | 7163 |

**The locality gain is real and small**: at 16 the sample stage beats the flat order by 1.03x while
paying 2x the plane memory, and the fold loses anyway. At 8, where the waste is almost nothing, it is
much WORSE - which says the rasterisation tile is bigger than eight fragments square, so a block that
small buys no sharing and pays the indexing. Reverted.

🔴 **SO THE SYNTHESIS, STATED AS AN INFERENCE AND NOT A BOUND THIS TIME.** The 2 TFLOP/s ceiling is
reached only because **every fragment reads the same operands**, and a GEMM's fragments must read
disjoint rows. Four mechanisms for recovering that have now been built and measured - the fragment
ORDER in one dimension (1.49x on the contraction, a 1.28x loss on the projection), the TILE-AWARE
mapping (3.29x on the contraction), the 2D BLOCK (above), and a k SPLIT to raise the fragment count
(1.5x slower) - and what is left after them is a factor of about twelve. **The fragment pipeline has
no mechanism for sharing a loaded operand between invocations**, which is exactly what workgroup
memory is, and the measured cost of not having it is that twelve.

That is a claim about four tried mechanisms, not a proof that no fifth exists - the last three
"bounds" in this file were each overturned by one measurement, and this paragraph should be read with
that history in front of it.

### What the remaining gap IS, from the WGSL kernel's own notes

**The distance is arithmetic intensity per GLOBAL fetch, and the mechanism that buys it is the one
thing WebGL2 does not have.** This is no longer inference: `webgpu/bioemu2/ops.js` says what its
GEMM does and what it measured.

| | WGSL `gemm2Source` | this backend's best |
|---|---|---|
| tile | **64 x 128**, staged in WORKGROUP MEMORY, 256 lanes of 4 x 8 | 32 output floats an invocation |
| its own figure | *"three workgroup reads for 32 multiply-adds"* | **~23 flops a `fetch4`** with int5 weights |
| achieved | ~2.4 TFLOP/s | **185 GFLOP/s** at **8 G fetch4/s** |

A 64-by-128 block loads its two panels from global memory ONCE and then reads them from workgroup
memory for every one of 8,192 outputs. A fragment shader has no workgroup memory, so a fragment's
only reuse is what it holds in registers, and that is capped by `MAX_DRAW_BUFFERS * 4`. **Every
global fetch here feeds about 23 flops; every global fetch there feeds hundreds.** That ratio is the
gap, and it is a property of the API.

🔴 **AND THE PRECISION SHORTCUT THAT WOULD PARTLY SUBSTITUTE WAS MEASURED AND DECLINED BY THE
REFERENCE PORT ITSELF.** The obvious way past 23 flops a fetch is to pack `x` as float16 halves -
eight k-values a texel instead of four, which would cut this kernel's dominant term almost in half.
`webgpu/bioemu2/ops.js` built exactly that on the WGSL side and rejected it: *"FLOAT32 ON PURPOSE.
f16 staging... was built and measured on the normalised-input GEMMs: promoted every 32 products, 266
-> 237 ms at FIVE TIMES the shipped arm's error against a float32 reference (sigma 16: 2.8e-3 ->
1.4e-2)... This arm keeps float32 activations and only the weights in half, which is why it is the
more accurate of the two ports."* So it is worth 1.12x there and costs 5x the error - and this
backend's own earlier attempt at it overflowed f16 outright and lost 7% when clamped. **Both ports
have now measured it and both decline it.**

🔴 **SO THE INT5 PROJECTIONS - 48.6% OF A bioemu2 FOLD - ARE AT THIS BACKEND'S PRACTICAL CEILING,
and that is a measurement rather than the two bounds this file got wrong before it.** `576 x 896 ->
1792` runs at **185 GFLOP/s on about 8 G fetch4 a second**, which is the HIGHEST rate any real kernel
here reaches - the shaped arms of `probe-webgl-fetch-rate.js` span 4 to 9 - and its 23 flops a fetch
is already past the 21.33 that f32 operands allow, because an RGBA32UI texel carries about 25
five-bit codes where an RGBA32F texel carries four floats. There is no further operand to compress
without changing the answer.

What that leaves, honestly: the remaining distance on bioemu2 is **not** a kernel underperforming
its hardware. It is 23 flops a global fetch against several hundred, and closing it needs workgroup
memory. Every control arm in this file was re-measured on the finished tree and every shipped
default is still the faster one (`--no-row-fastest` 11.6 s against 6.1, `--no-int5-weights` 13.1,
`--no-contract-block` 9.7, `--no-stride-pad` 8.5, `--no-half-weights` 6.4), so there is no knob left
set the wrong way either.

### 🔴 OPEN DEFECT: ESMFold2's RECYCLING MAKES ITS FOLD WORSE, AND NOTHING GATED THE SEAM

**This is a correctness finding, not a speed one, and it was surfaced by being asked for a
comparison this file had never made.** Every ESMFold2 number here - 253 s, then 65 s, then 9.5 s -
was taken against ITSELF. The WGSL fold of the same weights was never run, on the belief that its
bundle was absent; it is not. `fold-esmfold2.js` merely DEFAULTS to `/model-esmfold2-trunk-f32`,
which this box lacks, and takes `--bundle=` - so the comparison was one flag away the whole time.

Same bundle (`/model-esmfold2-int5`), same tower (`/model-esmc-600m-int3`), same model
(`esmfold2-fast-600m`), same 75-residue sequence, same seed 0, same four trunk loops, same eleven
sampler steps:

| | WebGL2 | WGSL |
|---|---:|---:|
| mean pLDDT | **57.33** | **88.08** |
| pTM | **0.411** | **0.883** |
| median CA-CA | 3.636 A | 3.826 A |
| fold | 9.53 s | 1.53 s |

🔴 **AND THE DECISIVE EVIDENCE NEEDS NO SECOND BACKEND AT ALL: MORE RECYCLING MAKES IT WORSE,
MONOTONICALLY.**

| trunk loops | 1 | 2 | 4 |
|---|---:|---:|---:|
| mean pLDDT | **69.49** | 58.80 | **57.33** |
| pTM | **0.548** | 0.424 | **0.411** |

Recycling exists to improve the structure. A fold that gets worse every time it recycles is wrong
however fast it is, and the one-loop number being the best says the seam - not the trunk - is where
to look.

🔴 **AND EVERY STAGE GATE PASSES, INCLUDING THE RECYCLE PROJECTION ITSELF.**
`check-webgl-esmfold2-trunk.js` compares `recycleProjection` against `cpu/esmfold2`'s and is green,
the layout matches the reference's `(in, out)` default, and the loop runs `num_loops + 1` times with
`z` starting at zero as the driver's own comment requires. So the kernel is right and the
composition is not - which is this file's own argument for whole-fold drivers, turned on a model
where the driver was only ever compared with itself.

**bioemu2 does not have this problem and that is why**: `check-webgl-bioemu2-trunk.js` drives TWO
PASSES against the WGSL trunk, so its recycle seam is gated across backends. ESMFold2's is gated
only per-kernel against the CPU reference. One gate's inputs, again.

**What this costs the speed claims above**: ESMFold2's 62.9 s to 9.53 s is real and measured, but it
is 6.6x on a fold whose answer does not match the WGSL one's, so it is NOT a like-for-like narrowing
of the gap and must not be read as one. bioemu2's and AF3's numbers are unaffected - both are gated
against the WGSL path across passes.

🔴 **ONE CONTRIBUTING BUG IS FIXED AND IT IS NOT THE MAIN ONE.** This driver SKIPPED the recycle
projection on the first loop, on the comment "z starts at zero and the projection is
zero-initialised upstream, so the first iteration is exactly `folding_trunk(z_init)`" - and
`webgpu/esmfold2/fold.js` names exactly that error: *"`pair_loop_proj(0)` IS NOT ZERO. Its Linear is
zero-initialised upstream and then TRAINED, and the LayerNorm in front of it has an offset...
Skipping the projection on the first loop is the natural shortcut and a different model."* Fixed -
`z` is an explicit zero pair now and the projection runs every loop - and it is worth **0.18 of
pLDDT**: 57.33 to 57.50. A real defect, measured, and not the cause.

🔴 **AND THE RESIDUAL IS A SHIFT, NOT A COLLAPSE, WHICH IS WHAT `minPlddt` AND `maxPlddt` ARE FOR.**
This tool reported only the MEAN, so "mean pLDDT 57.4" sat in this file for a whole campaign without
anyone asking whether the per-residue values varied at all. They do:

| | mean | min | max | spread |
|---|---:|---:|---:|---:|
| WebGL2 (rescaled from 0-1) | 57.50 | 34.9 | 73.9 | 39 |
| WGSL | 88.08 | 51.86 | 94.78 | 43 |

**A uniform downward shift of twenty-odd points with the same spread width** - and the structures are
comparable on the measures that do not go through the confidence head: median CA-CA 3.637 against
3.826, long-range contacts 126 against 117. So the trunk is probably not where the rest of it is,
and the calibration or the head is.

🔴 **AND THE COMPOSED LOOP IS NOW GATED AND IT IS CORRECT, WHICH NARROWS THE HUNT A LOT.**
`check-webgl-esmfold2-trunk.js` has a new arm that runs the fold's own composition -
`z = trunk(z_init + pair_loop_proj(z))`, from zeros, `--loops` times - on both the device and the
f64 reference, and it reads **1.281e-07** at two loops (green at one, two and three, red on a 1%
perturbation). So the seam is not the fault either.

What the hunt has now ELIMINATED, each against the f64 reference:

| stage | arm | relRMS |
|---|---|---:|
| ESM-C tower, **36 of 36 blocks** | `check-webgl-esmc-tower.js --blocks=36` | 2.5e-6 |
| `z_init`, every term | the trunk gate | 3.5e-8 |
| the trunk, 2 blocks | the trunk gate | 9.1e-8 |
| `pair_loop_proj` | the trunk gate | 1.5e-7 |
| **the composed loop, 2 loops** | **the trunk gate (new)** | **1.3e-7** |
| the distogram, both ways | the trunk gate | 1.4e-7 |
| the atom encoder, the diffusion, the confidence head | their own gates | all green |

**So no stage and no seam reproduces it at the gates' shapes** - and that is the clue. Every one of
those arms drives 8 tokens and 64 atoms of synthetic input, where a fold is 75 tokens and 608
atoms. This file has already recorded one bug of exactly that kind: "the confidence gate now drives
a single at the trunk's own magnitude WITH A HEAVY TAIL, since a uniform draw at std 200 never
reaches 512 and left the arm green with the bug restored."

🔴 **AND THE SHAPE SWEEP IS DONE, AND SO IS THE STRUCTURAL COMPARISON - THE STRUCTURES AGREE AND THE
CONFIDENCE HEAD IS THE WHOLE OF IT.** `fold-esmfold2-webgl.js` reports `alphaCarbons` now, which is
the one thing two BACKENDS can be compared on without either tool's confidence head in the way.
Superposed against the WGSL fold's own PDB through `tools/gpu/superpose.js`:

| | |
|---|---:|
| CA RMSD, 75 residues | **1.842 A** |
| TM | **0.862** |
| per-residue deviation | mostly 0.3 to 2 A |
| the worst two | the LAST two residues, 7.13 and 10.06 A |

**Two folds agreeing at TM 0.862 cannot differ by thirty points of predicted confidence.** The
trunk, the recycle and the sampler are right; the head's score is not.

🔴 **AND IT IS SYSTEMATIC, NOT A SAMPLE** - which had to be excluded before any of this counted as a
defect, because the two backends' sampler RNGs differ, so these are different draws. Four seeds:
**57.503, 57.563, 57.342, 57.552**. Pinned to a tenth while the structure moves.

🔴 **AND THE RANGE IS COMPRESSED RATHER THAN THE SCORE MERELY LOW.** Varying only the sampler:

| sampler steps | 5 | 15 | 40 |
|---|---:|---:|---:|
| median CA-CA | **227.8 A** (a collapsed chain) | 3.637 | 3.719 |
| mean pLDDT | **51.41** | 57.50 | 57.28 |

A head that sees the structure properly scores a 227-angstrom chain near 25, not 51. So the
structural term is present - pLDDT is not a constant - but its dynamic range is squeezed, and the
whole distribution sits about 0.2 below the WGSL fold's with a similar spread (0.35-0.74 against
0.52-0.95 on the per-residue 0-1 scale).

**What is eliminated, each measured:** the ESM-C tower at 36 of 36 blocks; `z_init`; the trunk; the
composed recycle loop; the distogram; the atom encoder, the diffusion and the confidence head on
their own gates, at 6 tokens AND at the fold's 75; the representative-atom choice (both drivers call
the same `shared/esmfold2/representative-atoms.js`); and the confidence pair bias, whose only WGSL
addend this driver omits is the token-bond encoding - *"absmax 0.0 over a 59-mer"*, zero on a plain
chain.

**One shape-dependent gate failure did surface and is not the cause**:
`check-webgl-esmfold2-confidence.js --tokens=75` fails "pTM and ipTM off the PAE logits" by
**1.034e-8**, which is a bar too tight for 75 tokens rather than a defect - but it is green at the
default 6 and nobody had run it at a fold's shape.

**And four more of the head's inputs are eliminated, each checked against the WGSL driver's or the
reference's own code rather than assumed:**

- **the pLDDT bin convention** - both backends call the identical
  `categoricalMean(logits, atoms, bins, 0, 1)` and both average per-atom over a token weighted by
  `atomMask`, so the 0-1 scale and the per-token reduction are shared;
- **the per-slot pLDDT table** - `intraTokenSlots` matches `cpu/esmfold2/confidence.js`'s running
  count exactly, and `maxAtomsPerToken` is **23** where ubiquitin's largest residue has about
  eleven atoms, so nothing collides at the cap. (The reference's own comment warns about this: *"Indexing
  it by the atom's global position instead reads a different residue's weights and still returns a
  number."*);
- **the representative atoms** - both drivers call the same `shared/esmfold2/representative-atoms.js`;
- **the token mask** - the WGSL driver passes an explicit all-ones array where this one passes
  `features.tokenMask`, which looked like the difference and is not: patched to all ones, the fold
  returns pLDDT **57.503** and pTM **0.4127**, to the digit. That mask is already all ones for a
  plain chain.

🔴 **AND pLDDT IS NOT A FUNCTION OF THE STRUCTURE IN THIS MODEL, WHICH IS WHAT FINALLY LOCALISED
IT.** The WGSL fold at `diffusion-8` produces a **collapsed chain** - median CA-CA **52.3 A** - and
scores it **pLDDT 84.47**, three and a half points below its good fold's 88.08. So the head is
reading the PAIR, almost not the coordinates, and the thirty points cannot be a structural
difference. Which agrees with the superposition: the structures were never the problem.

🔴 **SO THE PAIR BIAS WAS TESTED DIRECTLY, AND PASSING IT MAKES THE FOLD WORSE.** Zeroing the
`pairBias` this driver hands the confidence head:

| | mean pLDDT | pTM |
|---|---:|---:|
| with the relative encoding as `pairBias` | 57.50 | 0.4127 |
| **with it zeroed** | **59.42** | **0.5167** |

`webgpu/esmfold2/fold.js` measures the CORRECT bias as worth **+13.8 pLDDT** ("Measured absent: 13.8
pLDDT and a PAE 2.4x rougher"). A term that should help by fourteen points and hurts by two says
something is wrong here.

🔴 **BUT NOT NECESSARILY THE BIAS, AND SAYING SO WAS AN OVERCLAIM.** The kernel that makes it is
exact - relRMS **0** against the reference, newly gated - the weight is the same
`featuriser.relPos` the WGSL driver uses, the width matches the head's 256, and the sampler reads
the SAME tensor to produce a structure agreeing to 1.84 A. What the zeroing test actually shows is
that the head's **assembled pair** is wrong, and that removing one correct addend happens to improve
the score. The head assembles

```
out = normed + bias + rows[i] + columns[j] + product + distanceTable[bucket(d)]
```

and any of those six being wrong would do it. Adding a correct bias on top of a wrong base need not
help. So the defect is somewhere in that assembly, which is gated against
`cpu/esmfold2/confidence.js` and green - **and a head and a reference that share a mistake agree
perfectly**, which is this file's own recorded failure mode and why the next comparison has to be
against the vendor's forward pass rather than against the CPU side.

🔴 **AND THE KERNEL THAT PRODUCES IT WAS IMPORTED BY ITS GATE AND NEVER DISPATCHED.**
`check-webgl-esmfold2-trunk.js` brought in `relativeEncodingSource` and never ran it: the arms that
mention the relative encoding compare `relativePositionEncoding` - the CPU function - against ITSELF
with one field moved, which is a separation check and not a comparison. `z_init` covered only the
SUM of five terms, where a wrong relative encoding hides under the other four. **Gated now, as its
own arm: relRMS exactly 0.** So the kernel is right, the tensor is right, and the sampler uses the
same tensor to produce a structure that agrees to 1.84 A.

🔴 **SO EACH OF THE SIX ADDENDS WAS ZEROED IN TURN, AND THE BIAS IS THE ONLY ONE THAT HURTS.** One
fold per arm, the term multiplied by zero so the sampler stays referenced (removing it outright trips
`dispatch`'s guard against an input the kernel does not name - the same guard that caught a stale
`biasTex` binding earlier):

| zeroed | mean pLDDT | pTM |
|---|---:|---:|
| nothing (as shipped) | 57.50 | 0.413 |
| **the pair bias** | **59.42** | **0.517** |
| the row term | 48.80 | 0.407 |
| the column term | 53.73 | 0.373 |
| the outer product | 56.58 | 0.420 |
| the distance table | 51.63 | 0.335 |

Five of the six earn their place; the bias is net harmful. **So the first reading was right and the
correction above was too cautious** - though "the bias is wrong" is still not what the evidence says,
because:

🔴 **THE BIAS TENSOR IS NUMERICALLY THE WGSL'S.** `webgpu/esmfold2/fold.js` states its magnitude off
the vendor's own forward pre-hook - *"the relative encoding is 12.97"* - and this backend's is
**absmax 12.9752**. Same weight (`featuriser.relPos`), same width, same placement (added to the
normalised pair, after the norm), and the sampler reads the same tensor to produce a structure
agreeing to 1.84 A. The head's own `zNormScale` / `zNormOffset` exist in the bundle under
`confidence/zNorm/*`, so the base is not being normalised unweighted either.

**So the same tensor, in the same place, helps by 13.8 points there and hurts by 1.9 here** - which
leaves the base it is added to, and that is where the next person should look. Every gate on this
head supplies its OWN synthetic weights, so any mis-wiring between the bundle and the head is
invisible to all of them; the comparison that would find it is the two backends' heads on one input
in one page, which both modules already support because these gates load them together. **That is a
tool, not a research question, and it is the one thing this hunt did not build.**

🔴 **BUILT, AND IT ELIMINATES THE REFERENCES: THE TWO HEADS AGREE.**
`check-webgl-esmfold2-confidence.js` runs `esmfold2ConfidenceFold` from
`webgpu/esmfold2/confidence.js` beside this backend's head, on ONE set of synthetic weights and ONE
set of inputs, at `--tokens=8 --pair-channels=128 --single-channels=128 --single-inputs=128`:

| | pLDDT per token | PAE |
|---|---:|---:|
| WebGL2 against the f64 reference | **5.17e-08** | 1.43e-07 |
| WGSL against the same reference | **1.02e-05** | 4.54e-05 |
| the two heads against each other | **1.01e-05** | 4.54e-05 |

The WGSL figures are that path's own f16/bf16 rounding, not a disagreement - it lands at the same
~1e-5 against the reference and against this head, which is what it means for the difference to be
its own arithmetic and nothing else. **So both references are right, every stage of both heads is
right on synthetic data, and "a head and a reference sharing a mistake" is dead** - the possibility
twelve eliminations could not see, because each was measured against that pair of references. The
fold's divergence is thirty points of pLDDT and these heads differ at 1e-5, three to four orders of
magnitude too small, so what remains is **something the drivers do differently at a fold's real
magnitudes** - which is the base the bias is added to, from the other end.

🔴 **AND IT RUNS ONLY WHERE THE WGSL TRUNK COMPILES, WHICH THIS GATE'S DEFAULTS ARE NOT.** The WGSL
pair transition falls back to chunking its WHOLE intermediate when its preferred chunk does not
divide it, and then refuses a chunk that is not a multiple of its 128-lane workgroup - so 16 pair
channels cannot compile there however right the head is. The arm is skipped on **that refusal by
name** and the verdict says so; every other error is rethrown, because a swallowed exception is how a
WGSL regression would come to read as a skip. Sweeping `--tokens` does not help: the constraint is on
the CHANNEL widths, not the token count, which cost a run to learn.

🔴 **AND THE FIRST VERSION OF BOTH NEW ARMS WAS VACUOUS FOR A REASON WORTH KEEPING.** `--falsify`
skewed the inputs inside the WebGL2 UPLOAD, so the WGSL head read the originals - and "WGSL against
the f64 reference" then compared an unskewed head with an unskewed reference and read 1.015e-05 under
falsification exactly as without it. The skew is hoisted so both heads read the same numbers, and all
ten arms go red. **The head-to-head is reported rather than gated for the same reason inverted**: no
input skew can falsify it, because a skew moves both heads together. It is not lost - each head is
pinned to the f64 reference, which bounds their difference between them, and is the stronger
statement of the two.

🔴 **AND THE NEW ARM WAS VACUOUS ON ITS FIRST TRY, WHICH THE GATE ITSELF CAUGHT.** `--falsify` skews
`xInputs` and the language model pair, and this arm reads neither, so it stayed green under
falsification and the verdict reported **"VACUOUS: only 7 of 8 arms noticed it"**. Skewing the
identity does not reach it either - relRMS stays 0, which says the kernel does not depend on the
identity's magnitudes. It is falsified on the WEIGHT now, which is what it actually consumes. **An
arm is only a gate if the falsification perturbs something it reads**, and a verdict that counts
which arms noticed is how that gets found rather than assumed.

### The dense projection is bound by x's fetch locality, and the output tile is not the lever

Re-profiled after this campaign's wins, **the projection IS the fold**: at 68 tokens and 20 steps
`linear*` is **27.0 s of the 37.1 s** of labelled GPU time in a bioemu2 fold (72.8%) at an aggregate
**219 GFLOP/s**, where `probe-webgl-gemm-ceiling.js` measures a best-case fragment GEMM at 1958-2003.
The contraction that used to dominate is **11.8%**. So the next factor is here, and the spread within
the family is 13x - every fast row is `in-major` and every slow one `out-major` int5, which the fold
cannot separate because the two always coincide there.

`bench-webgl-linear.js --layout=` separates them on the float path, same weights laid out two ways:

| shape | out-major | in-major | ratio |
|---|---:|---:|---:|
| 576x896x1792 | 63.5 ms (29.1 GFLOP/s) | **12.7 ms (145.5)** | **5.0x** |
| 576x768x1920 | 66.9 ms (25.4) | **11.9 ms (142.7)** | **5.6x** |
| 4624x256x2048 | 29.3 ms (165.7) | **9.9 ms (487.9)** | **2.9x** |

🔴 **AND THAT IS NOT A LEVER ON THE FOLD, BECAUSE int5 OUT-MAJOR ALREADY BEATS FLOAT IN-MAJOR.** The
same shape runs at **10.63 ms in the fold** as int5 out-major against float in-major's 12.71 - the
codes are 0.625 bytes a weight against four, and the layout is the float path's problem. Reading the
layout table as a 5x waiting to be taken would have been a wasted kernel rewrite.

🔴 **SO WHICH OPERAND BINDS? x, AT 3.44x, AND THE CODES COST NOTHING AT ALL (1.09x).**
`probe-webgl-linear-operand.js` is the diagnostic `linear.js` already ran once on the other operand -
its own comment records "make every chunk read the SAME four code texels and the fold goes 6664 ms to
3476", which is what chose `orderRowFastest`. The x-side twin had never been run. Three arms, same
fragment count, same fetch count, same decode, same multiply-adds, differing only in which texels a
fragment addresses:

| arm | ms | GFLOP/s | factor |
|---|---:|---:|---:|
| shipped | 10.06 | 184 | - |
| **x-aliased** (every fragment reads row block 0) | **2.93** | **632** | **3.44x** |
| w-aliased (every fragment reads output group 0) | 9.25 | 200 | **1.09x** |
| **half-x** (x in float16) | **7.68** | **241** | **1.31x** |
| **tiled-2** (the fragment order below) | **5.13** | **361** | **1.96x** |
| tiled-4 | 5.73 | 323 | 1.76x |
| tiled-2 + half-x | 4.16 | 444 | 2.42x |
| **tiled-4 + half-x** | **3.94** | **470** | **2.56x** |

🔴 **AND THE FIRST VERSION OF THIS TABLE READ 19.24 / 2.94 / 19.24 ms AND IS RETRACTED, BECAUSE THE
PROBE LAID x OUT UNPADDED WHERE A FOLD DOES NOT.** 576x896 is 224 texels a row and 224 does not
divide 2048, so `linear` runs `padStride` over x - the 4.08x recorded further up this file - and
without it the probe's shipped arm read **18.76 ms where the fold's own profile says 10.63 for the
same shape**, 1.77x apart. Every lever it found was measured against misaligned x and so overstated:
x binding read **6.62x and is 3.44x**, `tiled-2` read **1.87x and is 1.96x** (that one went up), and
the aspect arm `4x8` read 0.84x where it is a **tie at 1.01x**. The probe stride-pads x now and its
shipped arm lands on the fold's own figure. **A probe that does not reproduce the shipped kernel's own
addressing prices every mechanism against a kernel nobody dispatches**, and the lesson cost a
published table.

**`orderRowFastest` finished the weight side.** Every remaining factor in this kernel is on the x
side, and 630 GFLOP/s is what the arithmetic reaches once x is in cache.

🔴 **AND THE OBVIOUS CONCLUSION FROM THAT IS FALSE - FOUR ARMS, THREE OF THEM NULLS.** x is re-read
once per OUTPUT GROUP, so a wider output tile of the same `MAX_DRAW_BUFFERS * 4` budget reads
proportionally less x, and the extra codes it reads were just measured free. Every one is SLOWER,
monotonically:

| tile | ms | GFLOP/s | x MB | code MB | factor |
|---|---:|---:|---:|---:|---:|
| **8x4** (shipped) | **10.06** | **184** | 882 | 68.9 | **1.00** |
| 4x8 | 9.93 | 186 | 441 | 137.8 | 1.01 |
| 2x16 | 16.43 | 113 | 220.5 | 275.6 | 0.61 |
| 1x32 | 33.00 | 56 | 110.3 | 551.3 | 0.30 |

**Quartering x's bytes makes the kernel 1.6x slower and halving them buys nothing**, which is the
whole finding: what binds is not x's VOLUME, read once per output group, but its fetch LOCALITY. Aliasing x does not merely read less - it makes every concurrent
fragment read the SAME eight rows - while a wider output tile keeps the scattered pattern and
multiplies the decode work and the live accumulators. **The shipped 8x4 is the best aspect of its
budget, and the traffic model that predicted 1.9x for 2x16 is retracted.** The lever is a fragment
ORDER that brings nearby x under one rasterisation tile, which is exactly what was worth 3.29x on the
contraction; it is not a reshaped output tile, and it is not the weight layout.

🔴 **AND THE FRAGMENT ORDER IS 1.082x ON THE FOLD, BIT-IDENTICAL - HALF WHAT THE PROBE PROMISED, AND
THE SHORTFALL IS THE PROBE'S OWN BASELINE.** `blockedInt5Source` takes a tile width now: ordered
row-block fastest a TILEW-wide run of texels covers TILEW consecutive row blocks at ONE output group,
where the 1D `texel / RQ` order gave a rasterisation tile every row block there is. In the probe that
reads **1.74x and 1.87x** over the shipped order; in the fold it is **1.12x** on `576x896x1792` and
**1.19x** on `576x768x1920`, and the whole fold goes **30.40 s to 28.10 s** - interleaved, two rounds
each, spreads of 1.0006, and `caca` 3.8540767531771998 on all four runs.

**The probe overstated it because the probe lays x out UNPADDED and a fold does not.** 576x896 is 224
texels a row and 224 does not divide 2048, so `linear` runs `padStride` over x - the 4.08x recorded
further up this file - and the probe's shipped arm reads 18.76 ms where the fold's reads 10.63 for
the same shape. The tiling was recovering an alignment the shipped kernel already has. **A probe that
does not reproduce the shipped kernel's own addressing overstates every lever it finds**, and its
header says so now; read its ratios and never its milliseconds.

The width is the NARROWEST inside a 2.5x plane-padding budget rather than the measured best, because
the mapping pads the output-group axis to `W / TILEW`: TILEW 2 pads bioemu2's 1792 outputs 2.29x and
would pad AF2's 256 outputs **sixteenfold**, 4.7 MiB of planes to 75.8. It applies only where
`rowFastest` already does, which is the order the gain is measured against. `--no-linear-tile` is the
control arm and `--linear-tile-width=` sweeps it. The other three models do not reach it and are
unchanged to the digit: ESMFold2 9497/9671 ms against 9668/9671 at the same pLDDT 57.579 and pTM
0.4147, AF3 2351 ms both ways at pLDDT 66.525, AF2's three recycles 57.938/57.937/57.941 both ways.
`linear.repack` pays 347 ms of the gain back, reading a plane 2.29x sparser.

### x in float16, the second mechanism on the stream that binds

The arm table above prices it at **1.31x on its own and 2.56x beside the tile order**, and it is the
same operand from the other end: the tile order makes concurrent fragments read NEARBY x, and this
makes each of those reads half the bytes. `Storage.uploadWeight`'s comment says why it had never been
offered for an activation - *"an activation is written by a kernel drawing into it, which would need
an RGBA16F render target and the rounding discipline that goes with one"* - and `padStride` **is**
that kernel: it already rewrites every element of x to fix the row stride, and `linear.int5-decoded`
is already an RGBA16F target this backend draws into, so the render target was never the obstacle.

Measured on the fold, three arms interleaved, two rounds:

| | fold | rmsd | CA-CA | contacts |
|---|---:|---:|---:|---:|
| neither | 30.40 / 30.46 s | 1.595 | 3.8540767531771998 | 201 |
| the tile order (at its first width rule) | 28.11 / 28.13 s | 1.595 | 3.8540767531771998 | 201 |
| and x in float16 | 25.70 / 25.69 s | 1.595 | 3.853090267294919 | 201 |
| **and the tile width swept** | **23.80 / 23.87 s** | **1.595** | 3.853090267294919 | **201** |

**1.275x in all** and 416 MiB down to 411, bit-identical through the tile order and **0.00099 A of
CA-CA** at the float16 step, with rmsd and the contact count unmoved on every arm. ESMFold2 is
**1.037x** (9397 ms against 9747) at pLDDT 57.573 against 57.579 and pTM 0.4146 against 0.4147.
AlphaFold 3 (2384 against 2372 ms, pLDDT 66.525 and pTM 0.242 both ways) and AlphaFold 2 are
unchanged to the digit - AF2 to the point that **its worst-atom differential against the WGSL fold is
0.0353 / 0.0308 / 0.0285 A on both arms**, which is the sharpest statement available that the
precision change does not reach it. Neither model's shapes meet `padStride`, which runs only where a
row width does NOT divide the texture width, so a shape already aligned gets no narrowing and no
rounding.

🔴 **AND THE TILE WIDTH IS 4, MEASURED ON THE FOLD, WHERE THE FIRST RULE DERIVED 2.** The planes pad
to `W / TILEW` output groups, so a narrower tile is better locality and worse memory, and the first
rule took the NARROWEST inside a 2.5x padding budget - 2 for a wide output, 8 or 16 for a narrow one.
Swept, two rounds each, every arm bit-identical: **width 4 is 24132/24119 ms**, that rule
25650/25628, width 8 26769/26808, width 16 27256/26902, and forcing 2 everywhere **26641/27180** -
slower than the rule that chose it, because forcing it pads a narrow output eightfold. Uniform 4 is
also the LEANER arm, 411 MiB against 416: the budget rule spent its padding on the big planes to save
it on small ones, which is backwards. **And the resolver preferring 4 beats forcing 4** -
23803/23873 against 24150/24151 - because 4 is illegal where `W / 4` is fewer groups than the shape
has (ESMFold2's 3072 outputs, bioemu2's `68x768x3072`) and it falls to 2 there instead of not tiling
at all. So the order is 4, then 2, then wider, subject to legality and a `TILE_PADDING_CAP` of 8.

🔴 **IT IS THE SECOND OPTIMISATION HERE THAT IS NOT BIT-IDENTICAL, AND IT IS A DEFAULT OF THE FOLD
DRIVERS AND NOT OF `Execution`** - exactly as `halfWeights` is, and for the same reason: the stage
gates build their own `new Execution(gl)` and hold these kernels to f32 references at bars down to
1e-7, so a backend-wide default would re-bar every one of them. All **48 check-webgl gates are ok**
with it in the tree. `--no-half-x` is the arm that compares f32 with f32.

🔴 **AND IT IS WHAT THE PRIMARY PATH ALREADY DOES**, which is the part that makes it a correction
rather than a trade: `webgpu/af2/evoformer/block.js` sets `normalizedStorage` to f16 outright, asks
its flash kernel for f16 input and output, and sets the transition's `hiddenStorage` to f16 for every
width that is a multiple of four - none of it gated by a tuning knob. The PRELUDE in
`webgl/runtime/execution.js` has said so since it was written. A port keeping f32 at those points
disagrees with the path it is matching.

🔴 **AND f16 OVERFLOWS AT 65504, SO THE NaN WATCH IS THE ARM THAT SAYS IT IS SAFE.** `--nan-watch`
fills every fresh texture with NaN and reads every output back, refusing at the first non-finite
value: the fold is **ok in both arms**, so nothing on this activation path reaches f16's ceiling at
this shape. Run it on a new model before trusting the default there - a pre-norm activation is the
one that could.

### Narrowing x where nothing had to move it, and why two models decline it

`padStride` runs only where a row width does NOT divide the texture width, so every already-aligned
shape kept an f32 x - and after the three changes above those were most of what the projection still
cost. Re-profiled: `linear:4624x256x512` at **4752 ms and 212 GFLOP/s** (the largest single row, and
the unblocked path, which had no such pass at all) and `linear.block:4624x1024x256` at **3897 ms and
129 GFLOP/s**, both reading 256 or 64 texels a row that 2048 divides exactly. `strideFor` returns a
row width unchanged when it already divides, so on such a shape `padStride` is a pure copy - and with
`halfActivations`, a pure NARROW. Two conditions widened, and the unblocked path given the pass:

| | fold | against | |
|---|---:|---:|---:|
| **bioemu2** | **22.45 / 22.46 s** | 26.55 s | **1.18x** |
| **ESMFold2** | **8.79 s** | 9.53 s | **1.10x** |
| AlphaFold 3 | 2460-2487 ms | 2365-2393 | **0.96x** |
| AlphaFold 2 | no change | - | **1.00x** |

🔴 **AND THE TWO HALVES WERE MEASURED TOGETHER UNDER ONE FLAG, WHICH IS NOT A MEASUREMENT OF EITHER.**
Widening the blocked path's condition and giving the unblocked path its own pass are different
mechanisms, and combined they read 1.016x - near the run-to-run spread - while the profile had
`linear:4624x256x512` going 212 to 189 GFLOP/s and `pad-stride` from 5284 calls to 9502, which looks
exactly like the second one being a net loss. Split into `narrowAlignedX` and `narrowUnblockedX` and
measured, two rounds each: **neither 23867/23895 ms, blocked-only 22781/22674 (1.05x), unblocked-only
23658/23643 (1.01x), both 22457/22492 (1.06x)**. Both pay, the blocked half by far the more, and the
profile's hint was the `--profile` wait artefact its own header warns about. Two mechanisms behind one
flag is one number that describes neither.

🔴 **AND THE LAST TWO ROWS ARE WHY IT IS A DRIVER DEFAULT AND NOT A BACKEND ONE.** AlphaFold 3 is
**1.04x SLOWER** with its pLDDT moved 66.533 against 66.525, and AlphaFold 2 gains no time at all
while **its worst atom against the WGSL fold goes 0.0353 A to 0.0904** - reproducible to the digit
over two rounds, and past the **0.0355 A dither floor** that comparison measures for itself, so a
real loss of agreement rather than noise. AF2's cross-backend differential is the one quality number
this backend has that is not its own reference, and spending it for no speed would be the worst trade
available.

So `narrowAlignedX` is a default of the two drivers it pays for. `webglExecution` takes driver
defaults now, `--narrow-aligned-x` and `--no-narrow-aligned-x` override either way, and with the
split in place AF2 reads **0.0353 / 0.0308 / 0.0285 A** and AF3 **2365 ms at pLDDT 66.525** - both
exactly their figures before any of this. All 48 check-webgl gates are ok.

🔴 **AND THE TILE ORDER PAST ITS `rowFastest` CAP IS 0.92x - THE REASONING I HAD AND THE ANSWER I DID
NOT.** The two worst rows left are `linear:4624x256x512` at 188.7 GFLOP/s and
`linear.block:4624x1024x256` at **144.5**, both float `in-major` at 4624 rows and both without the
tile order, because `rowFastest` is declined above a million x floats. The reason it is declined is
stated in `linear.js` and is about the REPACK: "the repack's source index strides by the row-block
count, which is 72 for bioemu2's atom projection and **436** for AF2's pair track". Tiled, that index
strides by **TILEW** - 4 rather than 578 at that shape - so the objection really is gone, and the
order was offered to the float path and past the cap to find out.

**0.92x**: a bioemu2 fold **24784/24214 ms against 22510/22382** over two rounds, bit-identical, with
AF2's differential unmoved at 0.0353 A. The tile buys no locality where x is **18.9 MiB** - far past
anything a rasterisation tile holds - and still pays for its padded fragments. Reverted, flag and
all. **Removing the reason a thing was declined is not evidence that it now pays**, and the cap
happens to be right for a second reason nobody had written down: a tile cannot hold a large x
whatever the repack does.

🔴 **AND THE REDUNDANT `padStride` CALLS ARE 29.8% OF THEM, SIZED AND DECLINED.** That pass is 9.4%
of a fold's labelled GPU time across **9502 calls** now, roughly one per projection, so an instrument
counted how many repeat the same `(tensor, rows, inChannels)`: **2832 of 9502 are repeats** - 2000
pairs called twice and 416 three times - which is several projections reading one x, as q, k and v do.
Memoising them would be **about 2.8% of GPU time, call it 1.03x**.

Not taken, and the reason is the hazard rather than the size. The census keys on a tensor OBJECT and
`Storage` POOLS textures, so a recycled object is a different tensor later and a memo keyed that way
would serve the previous tensor's bytes - the exact failure `--pool` exists to detect, bought for 3%.
The safe form is an API change: let a caller pad once and hand the result to several projections,
which is explicit and cannot go stale. That is a change across several model files for 1.03x, so it
is written down with its number instead of built. **The better fix is upstream of it anyway** - three
projections of one x should be one projection of a concatenated weight, which saves the draws too.

🔴 **AND THE LAST TWO BIG ROWS ARE X-BOUND BY THEIR SHAPE, WHICH `--float` ON THE PROBE SETTLES.**
`linear.block:4624x1024x256` runs at 144.5 GFLOP/s in a fold and its transpose `4624x256x2048` at
461.8, at the **same 0.19 fetches a flop** - a 3.2x spread no arithmetic here explains. The probe's
float arms (an f16 weight sampled `in-major`, ordered group fastest, as that path ships):

| shape | shipped | x-aliased | w-aliased |
|---|---|---|---|
| 4624x1024x256 | 23.21 ms, **104 GF/s** | **4.00 ms, 606** | 22.51, 108 |
| 4624x256x2048 | **7.25 ms, 669 GF/s** | 1.04x | 1.03x |

The wide-output shape is **at its arithmetic limit with neither operand binding**; the narrow-output
one is x-bound at **5.76x**. The difference is the **sharer count**, and it is a property of the
shape, not of the kernel: ordered group fastest, the fragments sharing one row block's x are exactly
the output groups, so 2048 outputs give 512 of them and 256 outputs give **64**. The fast shape's x
block is 4 KiB read by 512 fragments; the slow one's is 16 KiB read by 64.

🔴 **AND SPLITTING k IS 1.30x - THE HYPOTHESIS HALF RIGHT AND THE CONCLUSION WRONG.** `--k-splits=`
cuts the per-fragment x block to the fast shape's size: at four splits it is 4 KiB, exactly
`4624x256x2048`'s. **17.90 ms at 135 GF/s against 23.21 at 104**, plateauing there (eight splits is
17.98) and nowhere near 606 - because a split changes the block's SIZE and not the number of
fragments sharing it, which is still 64 - and it would still owe a reduce pass over four sets of
partials. Declined, and it is the second time split-K has been measured and rejected here.

🔴 **AND THE TILE ORDER ON THAT SHAPE ALONE IS 0.94x AT BEST AND 0.37x AT WORST**, which the
fold-level 0.92x could not establish because it bundled every float shape with a forced `rowFastest`.
Probed on `4624x1024x256` by itself: **tiled-4 24.58 ms against the shipped 23.14**, and tiled-8, 16
and 32 at **58.5-62.1 ms**. `tiled-32` wastes the same 1920 fragments the shipped arm does, so it is
not the padding - the mapping itself is worse here. **The tile order is worth 1.96x where the sharer
count is high and a loss where it is low, with nothing in between.**

**So the binding constraint on this kernel is how many fragments share an operand, and at 256 outputs
no fragment order can raise that above 64.** The only remedy is more output channels per fragment,
which `MAX_DRAW_BUFFERS * 4` forbids and the aspect sweep measured as a loss. For this shape the
measured options are exhausted: x binds at 5.9x, making it resident reaches 620 GF/s, and no
ordering, tiling or k split gets there. That is the clearest statement this backend has of where its
remaining gap lives - and it is the one thing a fragment shader cannot express, since sharing an
operand across fragments is what workgroup memory is for and WebGL2 has none.

**This session's four projection changes, end to end:**

| | before | after | |
|---|---:|---:|---:|
| **bioemu2**, 20-step 6MRR-68 | 30.37 s | **21.74 s** | **1.397x** |
| **ESMFold2**, full configuration | 9.63 s | **8.32 s** | **1.157x** |
| AlphaFold 3 | 2.37 s | 2.37 s | 1.00x |
| AlphaFold 2 | unchanged, 0.0353 A | - | 1.00x |

rmsd 1.595 and 201 contacts on every bioemu2 arm; the device peak 416 MiB to 411.

### What binds the triangle contraction: operand locality, at 3.91x - and four arms that missed it

The contraction is the second-largest family in a bioemu2 fold at **17.3% of labelled GPU time**, and
the shipped tiled kernel runs at **4.85 ms and 33.2 GFLOP/s** at 68 tokens and 256 channels where the
dense projection reaches 375. Four arms in `bench-webgl-contract.js`, each removing one candidate:

| arm | factor | what it says |
|---|---|---|
| **`resident-tiled`** - every fragment on pair block 0 | **3.91x** | **locality binds** |
| `aliased-tiled` - both operands one texture | 1.05-1.14x | two operands rather than one: no |
| `half-tiled` - both operands float16 | 0.96-1.08x | bytes: no |
| probe-alu's vec4 ceiling, 5034 GFLOP/s | 142x headroom | arithmetic: no |
| `--lengths=68,128,256`, whose k run is L/4 | 33.2 / 40.6 / **19.3** GF/s | run length: no |
| gemm-ceiling `--side=128`, 16,384 fragments | **1867 GFLOP/s** | occupancy: no |

**`resident-tiled` is 1.26 ms against 4.94 and 127.5 GFLOP/s against 32.6.** So the contraction is
operand-locality-bound, the same shape of answer the dense projection gives at 3.44x.

🔴 **AND THE FIRST VERSION OF THIS SECTION CLAIMED FOUR NULLS AND NO CAUSE, WHICH IS RETRACTED. The
mistake is worth more than the number.** `aliased` hands the kernel ONE texture as both operands,
which halves the **distinct footprint** and leaves every fragment reading its own pair rows - so the
working set stays past the cache, and the arm's 1.08x says only that having two 4.7 MiB operands
rather than one is not the constraint. I read it as "reuse is not the constraint", which is a far
larger claim than the arm supports, and `half-tiled`'s null said the same thing from the byte side:
**neither footprint nor bytes matter, and locality matters by 3.91x.** The projection's equivalent arm
had been right all along because it collapses the WORKING SET rather than the footprint. A diagnostic
arm has to remove the thing you mean to test - and this is the second time in this session that an arm
measured beside the wrong kernel, after the operand probe's straw baseline.

The occupancy line is retracted too, and by its own probe: `probe-webgl-gemm-ceiling.js --side=` was a
literal and had never been swept. Swept, it reads **1867 GFLOP/s on 16,384 fragments**, 1022 on
32,761, 1444 on 65,536, 1347 on 131,044, 1847 on 262,144 and 1903 on 524,176 - flat within noise. A
kernel with 36,992 fragments is not short of them.

🔴 **AND THE WORKING SET IS 148 KiB A TILE BECAUSE THE TILE MAPPING SPANS EVERY CHANNEL GROUP.** With
`TILEW = 2048 / (channels / 4)` = 32 at 256 channels, a 32-wide run of texels is ONE channel group
and 32 pair blocks, so a rasterisation tile covers about every block there is at that group - 136 pair
rows of 4 channels and LP values, **148 KiB**. Collapsing that to six rows is the 3.91x. The mapping
bought its 3.38x by fixing the one-dimensional neighbour and left this untouched, which is where the
contraction's remaining factor lives.

🔴 **AND THE BLOCK SHAPE TAKES 1.6-2.8x OF IT, MEASURED.** The working set is `rows * channels * LP`
and the arithmetic intensity is `8 * I * J / (I + J)`, so **channels a fragment cost locality and buy
no intensity**. The shipped block is 2 i x 4 j x 4 channels: 148 KiB a tile and 10.7 flops a fetch4.
One channel a fragment makes the same 32-float budget buy a **4 i x 8 j** block - 37 KiB a tile and
**21.3 flops a fetch4**, two factors from one change. `wide-block` in the bench:

| L | shipped tiled | wide 4x8x1 | factor | tiled GF/s | wide GF/s |
|---|---:|---:|---:|---:|---:|
| 64 | 3.47 ms | **2.16** | **1.61x** | 38.6 | 62.1 |
| 72 | 6.80 | **2.46** | **2.76x** | 28.1 | 77.6 |
| 128 | 26.65 | **13.34** | **2.00x** | 40.3 | 80.5 |

**On the fold's own shape it is about 2x**: 68 tokens pad to 72 for a 4x8 block, and 2.46 ms there
against the shipped kernel's 4.85 ms at 68 is 1.97x after paying for the 1.19x of extra padded work.

🔴 **AND IT NEEDS THE PAIR AXES PADDED TO A MULTIPLE OF EIGHT, where `padPairPlanesSource` pads to
four.** At 68 tokens `JBLOCKS` is 8.5 and the shader does not compile - "cannot convert from const
float to const highp int", which is what a non-integer in a `const int` looks like - so `JBLOCKS`
asks for its own padding and 68 tokens become 72, which is 1.19x the padded arithmetic and is already
inside the figures above. It also writes its 32 outputs as (i, j) quads rather than channel
components, so it needs its own repack: four `fetch4` where the 2x4x4 repack is one, because an
output texel is four CHANNELS of one pair and a wide fragment holds one channel of thirty-two pairs.
Timing the block alone read **2.76x** at L=72 where the pair reads 2.01x, which is what that repack
costs - and the bench arm times them together for exactly that reason.

🔴 **BUILT AND SHIPPED: 1.03x ON bioemu2 AND 1.05x ON ESMFold2, BIT-IDENTICAL.** `widePairBlockSource`
and `wideRepackSource` are in `webgl/af3/trunk/triangle.js` and `contractInto` prefers them wherever
`padded % 8 === 0` and `W / channels` is at least one tile:

| | wide | 2x4x4 | |
|---|---:|---:|---:|
| **bioemu2**, 20-step 6MRR-68 | **21.74 / 21.74 s** | 22.42 / 22.37 | **1.03x** |
| **ESMFold2**, full configuration | **8.32 s** | 8.75 | **1.05x** |
| AlphaFold 3 (59 tokens) | 2274 / 2321 ms | 2273 / 2305 | 1.00x |
| AlphaFold 2 | unchanged | - | 1.00x |

Every arm bit-identical: bioemu2 at rmsd 1.595, 201 contacts, `caca` 3.8531561311027773 and 411 MiB
on both; ESMFold2 at pLDDT 57.593 and pTM 0.4148; AF3 at 66.525 and 0.242; AF2's worst atom 0.0353 /
0.0308 / 0.0285 A. The 24-arm triangle gate passes on both paths and `--falsify` turns all 16
falsifiable arms red. `--no-contract-wide` is the control arm.

🔴 **AND AlphaFold 3 READ 0.83x ON ONE RUN AND IT WAS AN OUTLIER, WHICH NEARLY BOUGHT A PER-MODEL
RULE NOBODY NEEDED.** 2752 ms against 2292 looked like the eight-padding costing more than the block
saves - 59 tokens pad from 60 to 64, so every pair tensor is 1.14x bigger and that cost spreads
across a trunk whose contraction is a smaller share than bioemu2's. The reasoning was sound and the
measurement was one run: repeated, it is **2274/2321 against 2273/2305**, identical. One run is not a
measurement, and a plausible mechanism is not evidence that the mechanism is what you saw.

🔴 **AND `aliased` HAD ALSO BEEN MEASURED AGAINST THE WRONG KERNEL.** The shipped contraction is
TILE-MAPPED and aliasing had only ever been run against the untiled block, so this file's "aliasing is
1.11x" was a statement about a kernel nothing dispatches. The tiled answer agrees at 1.08x, which is
luck rather than vindication - and the arm was asking the wrong question either way.

🔴 **AND THE BENCH IS RED AT `--lengths=256`, PRE-EXISTING.** `CONTRACT_BLOCK_MAX_PAIR_FLOATS`
declines blocking above 1.57M pair floats, and at L=256 (16.8M) blocking is **1.12x** anyway - 268.52
ms against the split kernel's 301.01 - while at L=128 (4.19M) it is rightly declined, 187.86 against
42.18. **So the relation is not monotone in the pair size and a single size threshold cannot express
it.** Reproduced on the committed bench, so it is not from this work; it is at a length the current
folds do not reach, and it wants a rule over the shape rather than the size.

### The host side is not the lever, and a framebuffer cache that never hit

With the kernels taken as far as the measured arms allow, the remaining named family was everything
that is NOT drawing. `--skip-draws` puts it at **5925 ms of a 22434 ms fold** - 26% - and `--host`
splits what happens inside `dispatch`:

| | ms | per dispatch |
|---|---:|---:|
| `drawMs` | 14947 | 275 us |
| **`bindMs`** | **4398** | **81 us** |
| `readMs`, 24 reads | 2148 | 89 ms a read |
| `programMs` | 266 | 5 us |

🔴 **AND `bindMs` SPLIT THREE WAYS SAYS IT IS ALL ONE CALL**: of 4398 ms, `#bindTargets` is **4334**
and the input and uniform loops together are **64**. The split is in the tool because guessing which
part it was had already cost three wrong hypotheses in this session.

🔴 **THE FRAMEBUFFER CACHE NEVER HIT - 0 HITS AND 54,381 MISSES**, which is every dispatch in the
fold. Its key is the texture OBJECTS, and `Storage` only pools textures under `--pool`, so every
allocation is a fresh texture, a fresh key and a fresh framebuffer: a `createFramebuffer`, its
attachments, a `drawBuffers`, a `checkFramebufferStatus` and a `deleteFramebuffer` at the 256 cap,
per draw. Raising the cap to 4096 changes nothing, because the misses are not evictions. **A cache
whose key can never repeat is pure overhead wearing a cache's name.**

Reusing **one framebuffer per target COUNT** and re-attaching takes `#bindTargets` from **4285 ms to
92** - 47x - bit-identical, with the completeness query kept but asked once per count and rectangle
rather than once per draw. The keyed cache stays for the case it was built for, a fold under `--pool`.

🔴 **AND IT IS NOT A SPEEDUP, WHICH IS THE POINT.** The fold is **22411/22441 ms with it against
22459/22470 without**. Four seconds of host time vanished and the wall did not move, because that
time was OVERLAPPED with the GPU - the host runs ahead and blocks in `drawArrays` when the queue is
full, which is why `drawMs` is 275 us a dispatch. **So `bindMs` being 19.5% of the wall never meant
it was ON the wall**, and the inference from "19.5% of the fold is in binding" to "there is 19.5% to
win" was mine and was wrong. It is kept because it removes 54,381 create/delete pairs and would
matter on a slower CPU, not because it folds faster. `--no-framebuffer-reuse` is the arm.

🔴 **AND CACHING THE REST OF THE GL STATE IS WORTH NOTHING AND BREAKS THE FOLD.** `useProgram` and
the VAO cached is **1.004x** (22355 ms against 22434), which the split above predicts, since those
calls are inside the 64 ms. Caching the texture UNITS fails outright with `GL error 0x502` - a
feedback loop, because a cached binding lets a texture linger on a unit that a later dispatch makes a
render target, and clearing the conflict in `#bindTargets` does not save it: `bioemu2.slice:68x768`
still refuses. Reverted, and with it the hypothesis that 54,381 dispatches of redundant state were
costing anything.

🔴 **AND `--pool` FAILED WITH `bioemu2.m0:1x68: GL error 0x502` - WHICH WAS THE POOL KEY MISSING A
FORMAT, NOT A USE-AFTER-RELEASE.** This section first guessed a pooled texture was handed out while a
kernel still named it as an input. It is not: the error path says so outright now - *"no input shares
a target's texture"* - and so does the arm that checks whether an EARLIER dispatch left a target
bound to a sampler unit, which finds nothing either.

The pool keys a free texture by its RECTANGLE and its FORMAT, and `allocate`'s own comment has said
why since f16 arrived - "a texture comes back only as the format it was made in: handing an RGBA16F
texture to an f32 tensor is a silent halving of everything written into it". Then `allocateWords`
added a **third** format, RGBA32UI for the int5 codes, and the key never learned about it. So
`release` filed every words texture under `f32` and the next float allocation of that rectangle was
handed an integer texture: drawing into RGBA32UI as though it were RGBA32F is `GL_INVALID_OPERATION`,
and reading one through a `sampler2D` rather than a `usampler2D` is undefined - so the quiet version
of this bug is a wrong answer, and `--no-check-errors` turns the loud one into the quiet one. **The
rule was right, stated, and applied to two formats of three.**

🔴 **FIXED, AND POOLING IS BIT-IDENTICAL AND 1.04x.** `poolKey` carries the format for all three and
`allocateWords` takes from the pool rather than only filling it - it had been stranding every words
texture it released, bytes `pooledBytes` counted and nothing could use. A 20-step bioemu2 fold:

| | fold | caca | peak | creates | framebuffer hits |
|---|---:|---:|---:|---:|---:|
| `--pool` | **19.76 s** | 3.8540435712892709 | 799 MiB | **1,606** | **57,142** |
| default | 20.47 s | 3.8540435712892709 | **411 MiB** | 117,366 | 0 |

`caca` agrees to all sixteen digits, which is what pooling demands - a pooled texture carries the
last tensor's bytes, so a fold that agrees bit for bit also says **no kernel here reads a texel
nothing wrote**, which is the diagnostic pooling exists to be and which has never run before. All 48
gates pass with `--pool` as well as without.

**It stays OFF by default on the memory**: 799 MiB against 411 is the "measured loss" this file's
pooling note refers to, and 1.04x is not worth doubling the peak on a device that has WebGL2 because
it has little else. What changes is that the trade is now a choice rather than a crash.

🔴 **AND IT MAKES THE FRAMEBUFFER CACHE WORK, WHICH WAS THE OTHER HALF.** With fresh textures the
keyed cache cannot hit (0 of 57,397) and re-attaching one framebuffer per target COUNT is 47x better;
with pooling the objects repeat, the cache hits **57,142 of 57,397** and is the cheaper of the two
(69 ms against 89). So `#bindTargets` takes the reuse path only when pooling is off - which is also
what stops the keyed cache being dead code under a comment claiming it is kept for exactly this
case.

### The threshold that chose between two projection kernels was fitted with the wrong dtype

🔴 **`bench-webgl-linear.js` UPLOADED ITS WEIGHTS WITH `upload`, AND EVERY FOLD USES
`uploadWeight`.** That is the difference between an f32 weight and a **float16** one above 65,536
elements - and f16 halves the one-texel path's weight traffic, which helps it far more than it helps
the blocked one. So `blockedApplies`' `in * out >= 65536` rule for `[in][out]` was fitted on a
kernel pair that no fold runs. With the bench on `uploadWeight` and four interleaved rounds, the two
largest in-major labels in a bioemu2 fold come out on the **one-texel** side:

| `[in][out]` shape | one-texel | blocked | |
|---|---:|---:|---|
| 4624 x 256 -> 512 | **226-240 GFLOP/s** | 157-159 | **one-texel 1.43-1.53x** |
| 4624 x 256 -> 256 | **239-244** | 186-188 | **one-texel 1.29-1.30x** |
| 4624 x 1024 -> 256 | 153 | **179** | blocked 1.17x |
| 4624 x 256 -> 2048 | 231-234 | **475-481** | blocked 2.03-2.09x |

| | before | after | |
|---|---:|---:|---:|
| **bioemu2**, 20-step 6MRR-68 | 33.0 s | **30.4 s** | **1.08x** |
| ...and again, on the tile-aware fragment order below | 30.4 s | **28.1 s** | **1.08x** |
| ...and on x in float16 beside it | 28.1 s | **25.7 s** | **1.09x** |
| ...and on the tile width, swept on the fold | 25.7 s | **23.8 s** | **1.08x** |
| ...and on narrowing x where nothing had to move it | 23.8 s | **22.4 s** | **1.06x** |
| ...and on the 4x8x1 contraction block | 22.4 s | **21.7 s** | **1.03x** |
| ...and on the LayerNorm statistics in their own pass | 21.7 s | **21.0 s** | **1.03x** |
| ...and on adaLN's, the same mechanism and the same band | 21.0 s | **20.7 s** | **1.02x** |
| bioemu2's trunk | 13.4 s | **10.7 s** | **1.26x** |
| **ESMFold2**, full configuration | 11.2 s | **9.53 s** | **1.17x** |
| AlphaFold 3 | 2.36 s | 2.35 s | 1.00x |

Every reported quantity holds - rmsd 1.595, 201 contacts, bond rms 0.042 / 0.056 / 0.028, 412 MiB;
ESMFold2 at pLDDT 57.405, pTM 0.4132, 126 contacts and median CA-CA 3.637. The two kernels sum in
different orders, so the fold's CA-CA moves in the **seventh** figure (3.8540768 against 3.8540761)
and the bench's cross-path `relRms` is 1.46e-7 against a 1e-6 bar.

🔴 **AND `in * out` IS NOT THE DISCRIMINATOR AT ALL, WHICH IS WHY NO THRESHOLD ON IT COULD BE
RIGHT.** 384x512 and 256x768 are the same 196,608 and disagree; so do 512x512 and 256x1024 at
262,144. What fits ten of eleven measured shapes is the **aspect ratio** - blocking wins when one
dimension is at least four times the other, which is what an 8-by-4 register tile should prefer
while the one-texel kernel's 1-by-4 suits a balanced shape. The eleventh is 256x768, where the
one-texel arm collapses to **68 GFLOP/s** because its weight stride is 192 texels and 192 does not
divide the texture width - `pad-stride.js`'s row-wrap fault, in the weight this time - so that case
takes the block as well.

🔴 **AND THE BENCH NEEDED ROUNDS BEFORE ANY OF IT WAS TRUE.** Run single-shot it read 4624x256x512
at 155.7 GFLOP/s and then 112.6 - the same shape 1.38x apart - which is how a two-parameter rule
very nearly got fitted to noise: an early pass of this table had 256x768 and 384x512 disagreeing in
one direction and 512x512 and 256x1024 in the other, with no artefact to explain either. Each arm's
MINIMUM over four interleaved rounds now, with `spreadOneTexel` and `spreadBlocked` beside it
(1.00 to 1.13 here). **This is the second instrument in this file to have published numbers before it
was checked against itself, and the second threshold fitted to them.** `--no-linear-aspect` restores
the old rule, and the aspect rule is empirical: eleven shapes on one device, confirmed on two whole
folds, which is the evidence it has.

### A rasterisation tile is two-dimensional and a texel index is not

🔴 **THE ANSWER TO THE OPEN QUESTION, AFTER SEVEN NULLS: 3.29x ON THE CONTRACTION, BIT-IDENTICAL.**
Every kernel here decodes its work from `linearTexel()` - `y * W + x` - and reasons about which
*consecutive indices* share an operand. But fragments are not rasterised in index order: a tile is
a SQUARE of (x, y), so its rows are **W apart in the index**. With 578 pair blocks per channel, a
32-row tile therefore spans `32 * 2048 / 578` = **about a hundred channel planes**, each with its
own 37 KiB of operands - and "order the channel slowest" fixes only the one-dimensional neighbour.

The fix is to take the channel group from the fragment's **x range** instead:

```
TILEW = W / C4                   // 32 at 256 channels, and C4 divides W for every power of two
cgroup = (texel % W) / TILEW     // every fragment in an x range shares a channel
block  = (texel / W) * TILEW + texel % TILEW
```

A tile is a range of x crossed with a range of y, so a whole tile now shares one channel's
operands. Measured at 68 tokens and 256 channels, with the cross-arm `relRms` **exactly 0**:

| contraction at 68x256 | |
|---|---:|
| channel fastest | 24.59 ms |
| channel slowest (the previous fix) | 16.21 ms |
| **tile-mapped** | **4.93 ms** |

| | before | after | |
|---|---:|---:|---:|
| **bioemu2**, 20-step 6MRR-68 | 37.8 s | **33.1 s** | **1.14x** |
| bioemu2's trunk | 18.1 s | **13.5 s** | **1.34x** |
| **ESMFold2**, full configuration | 15.1 s | **11.3 s** | **1.34x** |
| AlphaFold 3 (a 4-block trunk prefix) | 2.63 s | 2.58 s | 1.02x |

bioemu2's CA-CA is identical to all sixteen digits and every other quantity holds.
`--no-contract-tile` is the control arm. The price is 5% more fragments, because the tiled grid is
whole texture rows.

🔴 **AND THE PROBE THAT FOUND IT WORKS BY ELIMINATION, WHICH IS WHY IT TOOK SEVEN NULLS TO GET
HERE.** `tools/gpu/probe-webgl-fetch-rate.js` measures the achieved `fetch4` rate as a function of
the fragment's SHAPE with everything else nailed down - one fragment count, one pair of textures,
one k axis, 32 output floats - and the contraction's shape is **not** the slow one:

| arm | fetches a k-step | G fetch4/s |
|---|---:|---:|
| `8x4x1`, the projection's shape | 12 | 2.55 |
| `4x4x2` | 16 | 2.98 |
| **`2x4x4`, the contraction's shape** | 24 | **3.60** |

**So the shape was innocent and the access PATTERN was the whole of it** - which is what turned the
search from "what is wrong with this kernel" to "what does the hardware actually schedule". Its
absolute numbers are not the real kernels' (its own neighbours read different rows on purpose), and
the header says so; what it establishes is a comparison.

🔴 **AND THE SAME MAPPING ON THE PROJECTION IS A 1.28x LOSS, WHICH IS THE USEFUL HALF OF THE
RESULT.** Its decode has the same one-dimensional assumption - neighbouring indices share the eight
x rows, and a tile's rows are W apart, so a tile spans about 512 row blocks rather than one - so it
looked like the same bug. Built and measured on bioemu2's cheap fold, with the answer identical:
**10042 ms against 7870**.

**Tile-awareness pays when the shared operand is SMALL AND FULLY REUSED and costs when the tile is
forced to sweep a large STREAMING one.** The contraction shares one channel's 37 KiB and reuses all
of it; the projection's flat mapping already gives a tile ROW one row block (`OUT4` = 128
consecutive texels covers a 32-wide segment), and forcing a whole tile onto one row block makes that
tile sweep every output group - which streams the entire 0.5 to 2.3 MiB weight inside one tile. The
locality it buys is worth less than the locality it spends. Reverted; the contraction keeps it.

### A short projection is slow and neither lever moves it

🔴 **ESMFold2's SINGLE TRACK IS 436 CALLS OF A FOLD AT 55 GFLOP/s, where the pair track beside it
reaches 198 to 523 - AND TWO PLAUSIBLE EXPLANATIONS ARE BOTH MEASURED WRONG.** The shape is 75 rows
by 768 to 768, so an eight-row block leaves `ceil(75 / 8) * 768 / 4` = **1920 fragments**, which
looks exactly like the starved dispatch CLAUDE.md documents on the WGSL side.

| | ESMFold2, 36 tower blocks, 1 loop, 4 steps |
|---|---:|
| rows a fragment: **8** (shipped) | **5009 ms** |
| 4 | 5214 ms |
| 2 | 5725 ms |
| a floor of 8192 fragments, so 2 for this shape | 6019 ms |

**Monotone the wrong way**: more fragments is strictly worse, so the kernel is not short of work to
issue, and the R = 4 against R = 8 measurement in "How close can this get" holds at a 75-row shape
too. `--block-rows=<n>` is kept as the sweep.

🔴 **AND NEITHER IS IT THE FRAGMENT ORDER, THOUGH THE SHIPPED RULE SAYS IT SHOULD BE.** Row-block
fastest is gated to the int5 path because it once cost AF2 3.7x through the REPACK, whose source
stride is the row-block count - 436 for AF2's pair track. This shape has **ten**, so the repack
would survive it, and `rows * inChannels` is 57,600 floats against the rule's million, so the rule
would select it. Measured: **5253 ms against 5012**, a loss. The traffic model says why, and it is
the one the "whichever operand is smaller" phrasing hides - what matters is the streaming operand
times the number of times it is re-streamed. Output-group fastest re-streams the weight once a row
block, `2.3 MiB * 10` = 23 MiB; row-block fastest re-streams x once an output group, `0.23 MiB *
192` = 44 MiB. The shipped order is already the cheaper one, and the int5 gate was not the proxy it
looked like.

So what limits that kernel is neither its block nor its order, and 55 GFLOP/s at 21.3 flops a fetch
is 2.8 G fetches a second against the 6.8 the long projections reach. It is the same unexplained
factor as the contraction's, in a second kernel, and it is the open question.

### `fetch1` is a whole texelFetch, and five attentions walked their head with it

🔴 **A SCALAR WALK ALONG A CONTIGUOUS AXIS COSTS FOUR TIMES WHAT IT NEEDS TO, AND IT WAS IN EVERY
ATTENTION IN THE TREE.** `fetch1(tex, i)` fetches the texel at `i >> 2` and selects a component, so
four consecutive scalars are four `texelFetch` of one or two distinct texels. Every attention kernel
here computed `q . k` with `for (d) dot += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d)` - and
computed it **twice**, once for the softmax's maximum and once for the weight. At a head dimension
of 64 that is 128 scalar fetches a key where 32 `fetch4` carry the same numbers.

Both bases are a row offset plus `head * HEAD_DIM`, and every head dimension in play is a multiple
of four, so the division is exact. Fixed in five kernels - bioemu2's full and windowed attentions,
ESM-C's tower, ESMFold2's atom transformer and diffusion, and AF3's atom attention:

| | before | after | |
|---|---:|---:|---:|
| bioemu2, the 20-step 6MRR-68 fold | 40.1 s | **37.8 s** | **1.06x** |
| bioemu2's sample stage, per step | 1062 ms | **957 ms** | **1.11x** |
| ESMFold2, the full configuration | 16.2 s | **15.2 s** | **1.06x** |
| AF3 | 2.46 s | 2.47 s | 1.00x |

bioemu2's CA-CA is identical to all sixteen digits and ESMFold2 holds pLDDT 57.405, pTM 0.4132 and
126 contacts; AF3's atom attention is too small a share of its fold to move it. `--no-attend-dot4`
is the control arm for all five, and each kernel's own gate reports identical residuals both ways.

🔴 **AND THE SEARCH THAT FOUND THEM IS WORTH MORE THAN THE FIX.** A dozen lines of Python over
`webgl/` for a `fetch1` whose index is a loop variable plus a base turned up **seventy-odd sites**,
and ranking them against the profiles picked out the five that matter. Two near-misses in the same
sweep: `layer-norm` already dispatches to an aligned `fetch4` form when four divides the channels,
which is every hot case; and `af3.triangle.operands` went from twenty `texelFetch` an output texel
to nine and measured **exactly nothing**, because that kernel is bound by its writes rather than its
reads. A pattern that is wrong in seventy places is worth a script; which seventy are worth fixing
is a question only the profile answers.

### Four does not divide 75, and that was two thirds of an ESMFold2 fold

🔴 **THE BIGGEST SINGLE FACTOR IN THIS WHOLE CAMPAIGN, AND IT WAS A DIVISIBILITY
CONDITION.** The contraction has two fast paths and BOTH need four to divide the pair length: the
coalesced `fetch4` along k, because a row must begin on a texel boundary, and the pair block,
because a fragment owns four j. ESMFold2 folds **75 tokens**, so neither was reachable and it ran
the scalar one-float-at-a-time loop - the fallback written for an odd width. Its own profile, with
`--profile` finally wired into `fold-esmfold2-webgl.js`:

```
af3.triangle.contract-only   15252 ms   56 calls   65.9%   272.35 ms each
linear.block                  3545 ms 1139 calls   15.3%     3.11 ms each
```

**272 ms a call at 0.4 GFLOP/s**, where the same kernel pair-blocked at 68 tokens reaches 10 - and
**15.3 s of a 23.2 s fold in one kernel**. Padding both pair axes to 76 with zeros makes both fast
paths legal:

| | WebGL2 before | after | |
|---|---:|---:|---:|
| **ESMFold2**, 36 tower blocks, 4 loops of 24, 11 steps | 62.9 s | **16.2 s** | **3.88x** |
| **AlphaFold 3**, 59 tokens padded to 60 | 3.67 s | **2.86 s** | **1.28x** |
| bioemu2, 68 tokens - already a multiple of four | 7.21 s | 7.30 s | 1.00x |

Every reported quantity holds: ESMFold2 at pLDDT 57.405, pTM 0.4132, 126 contacts and median CA-CA
3.637 **in both arms**, AF3 at pLDDT 66.525 and pTM 0.242, bioemu2's CA-CA identical to sixteen
digits. `--no-contract-pad` is the control arm.

🔴 **AND THE PADDING IS EXACT, NOT AN APPROXIMATION.** The tail is ZEROS, so the k loop running to
76 adds `0.0 * 0.0` - the same sum, not a nearly-equal one - and the repack reads only the true 75
of the padded grid, so nothing downstream sees a different shape. The cost is `(76/75)^3` = **4%
more multiply-adds** plus two passes over the operands, against a factor of four.

🔴 **AND IT IS NOT THE PADDED ROW STRIDE THAT WAS MEASURED AND DECLINED ABOVE.** That one laid an
ALREADY-ALIGNED row at a wider stride to buy cache locality, and lost. This one makes an UNALIGNED
row aligned so that a different kernel becomes legal. Same machinery, opposite question - and the
reason the first result did not predict the second is that a cache experiment at 68 tokens cannot
see a kernel that 68 tokens already reaches.

🔴 **AND THE GATE FOUND IT READY, BECAUSE ITS LENGTHS WERE WIDENED EARLIER FOR AN UNRELATED REASON.**
`check-webgl-af3-triangle.js` drives 5, 11, 12 and 68 - the first two ODD, so the padded path is
exercised by the default gate from the moment it exists (5 pads to 8, 11 to 12), and `12` and `68`
hold the unpadded one. The commit that widened those lengths was about a gate that had never run
three of its kernels; it paid for itself twice.

🔴 **AND `--profile` WAS BUILT INTO THE ESMFold2 TOOL AND NEVER PRINTED.** `webglExecution` honours
the flag, so every dispatch in that tool had been paying the one-texel readback barrier and
reporting nothing - on the model whose trunk is 53 s of a 62 s fold, which is the one most worth
ranking. One line. **The kernel that was two thirds of a fold was invisible because nobody had asked
that tool where its time went.**

### The contraction is 256 matmuls, and the fragment order decides whether one fits in cache

🔴 **THE SAME PROPERTY AS THE PROJECTION'S ROW ORDER, IN THE KERNEL WHERE IT IS WORTH MORE.** The
pair-blocked contraction was the third-largest kernel in a bioemu2 fold - **1591 ms of 7074, 22.5%,
in only 64 calls** - at **6.2 GFLOP/s** on a device whose padded projections reach 95 to 113. It is
not fetch-RATE bound: `tools/gpu/bench-webgl-contract.js` measures it at **0.58 G fetches a second**
where this context does about ten.

It is bound by TRAFFIC, and the arithmetic says where from. `out[i][j][c] = sum_k a[c][i][k] *
b[c][j][k]` is `channels` INDEPENDENT matmuls, one per channel, and at 68 tokens one channel's two
operands are `2 * 68 * 68 * 4` = **37 KiB** - small enough to stay in cache for that channel's whole
68x68x68 product. With a block of two i and four j, operand `a` is wanted by `L / 4` = 17 j-blocks
and `b` by `L / 2` = 34 i-blocks, so the 9.4 MiB of operands are read about **25 times each, 242 MB
a call** - and the measured 9.3 GB/s says that is exactly what is happening.

🔴 **AND THE FRAGMENT ORDER WAS SPENDING IT.** A fragment's texel index decoded with the CHANNEL
group varying fastest (`cgroup = texel % C4`), so neighbouring fragments worked on different
channels: 64 channel groups in flight together, 256 planes live, and not one of them resident. With
the pair block varying fastest they share one channel's 37 KiB. Measured, the answer identical to
all sixteen digits of the reported CA-CA in every arm:

| | contraction at 68x256 | bioemu2 trunk, 48 blocks | the 20-step fold |
|---|---:|---:|---:|
| channel fastest | 24.7 ms | 21628 / 21577 ms | 45.9 s |
| **pair block fastest** | **16.5 ms** | **18139 / 18129 ms** | **42.4 s** |
| | **1.49x** | **1.19x** | **1.08x** |

`--no-contract-channel-slowest` is the control arm. This is `ROW_FASTEST_X_FLOATS`' finding in a
second kernel, and it was found by writing the bench rather than by reading the fold: the fold's own
profile had the contraction at 14.9% and `linear.repack` at 9.6%, and the repack turned out to be
**barrier and not kernel**.

🔴 **AND IT IS WORTH NOTHING ON ESMFold2 OR AF3, FOR A REASON WORTH WRITING DOWN.** Both are
**exactly** unchanged - ESMFold2's full configuration 62256 ms against 62419, same pTM 0.4132, same
126 contacts; AF3 identical at pLDDT 66.525 - because the pair block needs `L % 4 === 0` and they
fold 75 and 59 tokens. Those lengths take the one-pair-a-fragment fallback, where there is no block
and so no order to get wrong. A knob's worth is a property of the configuration, and here the
configuration is the protein's length.

🔴 **AND THE BLOCK HAD NO CEILING AND WAS THE WRONG KERNEL ABOVE ONE, BY UP TO 4.8x - NOW CLOSED.**
Swept over a grid of length and channels with the two kernels' answers identical, the blocked
kernel's time over the one-pair kernel's (under one means the block LOSES):

| `L * L * channels` | 296K | 590K | 592K | 1.05M | 1.18M | 2.10M | 2.36M | 4.19M |
|---|---|---|---|---|---|---|---|---|
| blocked / one-pair | 5.10 | 1.71 | 4.62 | 1.12 | 3.95 | **0.38** | **0.59** | **0.21** |

Nine points over three lengths and three channel widths, and **`L * L * channels` separates them
completely**: every one at or below 1,183,744 wins and every one at or above 2,097,152 loses, with
no overlap either way - so the threshold is that ONE quantity and not a pair of them, and
`CONTRACT_BLOCK_MAX_PAIR_FLOATS` is set between the two where nothing is measured. The mechanism is
the REPACK ROUND TRIP, which is why it is the tensor and not the length: the block's saving is
fetches, and its cost is writing `L * L * channels` floats across eight planes and reading all of
them back to interleave, which is proportional to the tensor where the saving is not.

**bioemu2 at 68 tokens and 256 channels is 1,183,744** - the largest measured win and the top of the
range - so this costs that fold nothing and is worth **1.18x on a 128-token AF3 fold** (4764 / 4767
ms against 5612 / 5655, pLDDT identical at 70.559). AF3's 128-channel pair track crosses at 128
tokens and bioemu2's 256-channel one at about 86, so it is every input above those.
`--no-contract-ceiling` is the control arm, and `bench-webgl-contract.js` GATES the threshold rather
than only measuring beside it: it reports what the shipped rule would pick at each shape and FAILS
when that is not the faster arm.

🔴 **AND FOUR HYPOTHESES FOR THE CONTRACTION'S REMAINING 20x ARE MEASURED AND DEAD**, which is worth
more than another guess. It sits at 7.6 GFLOP/s in a fold and about 1 G fetches a second against the
projection's 6.8, on only 2x less arithmetic a fetch, and it is **not**:

- **short of fragments** - GFLOP/s FALLS as the channel count rises, 11.7 at 64 channels to 6.5 at
  512, where more channels is strictly more fragments;
- **short of memory-level parallelism** - fusing the four channel blocks into one k loop, which
  quadruples the fetches in flight per thread for the same fetches in total, is **1.25x WORSE** (16.07
  ms at one loop apiece, 17.95 fused in pairs, 20.07 fused in fours), so the register pressure costs
  more than the latency it hides;
- **row alignment** - and this is the sharpest of the four, because it needed no padding code to
  test: pick a length whose row is ALREADY a power of two. L=64 is 16 texels and runs at **8.9
  GFLOP/s**, the slowest of the sweep, where L=68 at 17 texels runs at 12.1. The aligned lengths are
  the slow ones, which is independently why the padded stride above lost;
- **bandwidth** - after the fragment order fix one channel's 37 KiB is resident, and the operands are
  9.4 MiB read against 242 MB before.

What is left is the 2x in arithmetic intensity, and the pair block cannot be widened past 32 output
floats while a vec4 holds four CHANNELS: four channels times eight pairs is already the budget, and
a tile of four i by eight j reaches the projection's 21.3 flops a fetch only by making the output
vec4 hold four j instead, which costs the repack four reads per output texel. That trade is
unmeasured and is the next thing to try.

🔴 **AND ONE RUN IS NOT A MEASUREMENT, ON A MACHINE SOMEBODY ELSE IS ALSO USING.** The first
128-token AF3 pair read **199191 ms against 5557** - the ceiling arm 36x SLOWER, the opposite of
every bench point - and at one trunk block the two arms were 11149 and 11768. Repeated, they are
4764 / 4767 against 5612 / 5655 and the ceiling wins. The tell was that the numbers did not compose:
a one-block fold cannot be slower than a four-block one. Nothing was changed to fix it, because
there was nothing wrong.

### A row normalised once a head, where four heads share the row

🔴 **bioemu2's PAIR BIAS ISSUED SEVEN TIMES THE `texelFetch` IT NEEDED, AND IT WAS 4.4% OF A FOLD.**
`pairBias` normalises a pair row and projects it onto each attention head, and its `at(element)`
recomputed the LayerNorm mean AND spread - two full sweeps of the channel axis - **once per output
element**, four times a fragment. The statistics depend on the ROW only; and with `heads` a multiple
of four all four of a fragment's elements ARE one row, because `base` is a multiple of four and the
head is `base % HEADS`. Its projection loop also read all three operands through `fetch1`, which is
a whole `texelFetch` plus a component select, 256 channels times three operands times four heads.

```
per fragment, at 256 channels and 16 heads:
  4 * (64 + 64 + 768) = 3584 texelFetch   ->   64 + 64 + 384 = 512
```

Measured on the 6MRR-68 fold, with `--no-pair-bias-fast` as the control arm:

| | sample stage, per step | the 20-step fold |
|---|---:|---:|
| a row normalised per head | 1167 - 1176 ms | 42315 ms |
| **once per fragment** | **1052 - 1064 ms** | **40101 ms** |
| | **1.11x** | **1.055x** |

Every reported quantity holds - rmsd 1.595, 201 contacts, bond rms 0.042 / 0.056 / 0.028, peak 412
MiB - and CA-CA moves in the SEVENTH figure (3.8540761 against 3.8540776), which is the reordering
and not the fix.

🔴 **AND "BIT-IDENTICAL" WAS CLAIMED TWICE AND WAS WRONG BOTH TIMES.** The products accumulate in
ascending channel order, one scalar at a time, exactly as the loop over `c` did - deliberately not a
`dot`, which would associate four at a time. That was not enough: written as one `vec4` chain the
normalisation contracts differently, so it was rewritten scalar per component - and it STILL differs,
because the compiler vectorises the scalar adds back at this channel width. The evidence that it is a
reordering rather than a defect is that `check-webgl-bioemu2-denoiser.js` reports **identical
residuals to every digit in both arms** (1.854e-07, 8.898e-08, 7.506e-10 against the f64 reference).
Claim a reordering and prove it with the gate; do not claim bit-identity without reading the digits.

🔴 **AND THE FIRST MEASUREMENT OF IT SAID ZERO, BECAUSE THE CONTROL ARM WAS NOT RUN.** A loop over
two arms built the flag into the printing and not into the command, so both "arms" were the fast one
and the full fold read 40058 against 40105 - a convincing null. The tell was that it contradicted a
reproducible 1.11x at a cheaper configuration; the two configurations differed in two variables at
once, and isolating them (48 blocks at 4 steps, 4 blocks at 20) put the per-step cost at 1052-1064
against 1167-1176 in every cell. **A null that disagrees with a reproducible positive is a bug in the
harness until proven otherwise**, and `--host` ruled out the obvious alternative first: the whole
fold compiles for 186 ms, so no shader-compilation story could have explained 467.

### How close can this get, and why not closer

The campaign above took a bioemu2 fold from **968 s to 21.7 s** and its device peak from 1656 MiB
to 411. The gap to the WGSL fold of the same weights is **5.57x** - it was 246x.

🔴 **THE 40.1 s AND 10.3x THIS SECTION OPENED WITH ARE SUPERSEDED, and so is the decomposition that
followed them** (a trunk at 18.2 s against 2.3, and 20 denoiser calls at 21.0 against 1.2). They were
true when written; the projection work below - a tile-aware fragment order, its width swept on the
fold, x in float16, that narrowing extended to aligned shapes - and the 4x8x1 contraction block took
the fold to 21.7 s. The sections after this one carry the current numbers; this one is the standing
argument about the CEILING, which has not changed.

🔴 **AND THE DECOMPOSITION RE-MEASURED, BECAUSE THE GAP HAS MOVED FROM THE TRUNK TO THE DENOISER.**
Same job, same 68 tokens and 573 atoms, same mode, same 20 steps, one sample - checked field by
field, since this tree has twice recorded a false regression from a mismatched configuration:

| | WebGL2 | WGSL | |
|---|---:|---:|---:|
| trunk | **8.91 s** | 2.31 s | **3.86x** (was 7.9x) |
| 20 denoiser calls | **12.11 s** | 1.18 s | **10.3x** (was 17.5x) |

So this campaign halved the trunk's gap and the denoiser is now **56% of the fold at 10.3x** - which
is where a reader should start, and is NOT where the whole-fold profile points, because that profile
mixes the two. Differencing `--steps=2` against `--steps=12` by label is the attribution: per step
the projections are 31.5% of the denoiser, `pad-stride` **9.9%** over 362 calls and `linear.repack`
**8.8%** over 361, with the remaining 38% spread over 36 kernels.

🔴 **AND THERE IS AN ARITHMETIC FLOOR UNDER THAT NUMBER, WHICH IS WORTH STATING BEFORE THE NEXT
OPTIMISATION RATHER THAN AFTER.** An invocation writes at most `MAX_DRAW_BUFFERS * 4` = **32 output
floats**. Produce those 32 from a k loop as a register tile of R by C, and a k step costs `R + C`
`fetch4` for `32 * 4 * 2` = 256 flops - so the intensity is `256 / (R + C)`, and with `R * C = 32`
the smallest `R + C` is **12** at 4 by 8 or 8 by 4. **21.33 flops a fetch is therefore the ceiling
for ANY kernel in this backend**, not a property of these kernels.

The two dense projections reach exactly that intensity, and the int5 ones now run at **375 to 394
GFLOP/s** where this line used to read 137 to 198 - the tile-aware order and a float16 x, measured in
"The dense projection is bound by x's fetch locality" above. They are **49%** of a bioemu2 fold, not
the 70% recorded here before the contraction was fixed a third time.

🔴 **AND WITH THE ISSUE RATE MEASURED ON ONE SIDE AND THE PROJECTION'S OWN RATE ON THE OTHER, THE
CEILING IS A PRODUCT OF TWO MEASUREMENTS RATHER THAN AN ESTIMATE.** `probe-webgl-linear-operand.js`
counts the fetches its shader issues exactly - `C` scale/zero reads a group of 32 input channels, then
four chunks each decoding `C` channels and reading `R` rows of x twice, with `decodeInt5x8` taking a
second code texel only when its word lands on channel 3, one chunk in four - and reports the rate
beside the flops:

| arm | ms | GFLOP/s | G fetch4/s | flops a fetch4 |
|---|---:|---:|---:|---:|
| the shipped kernel at the start of this work | 10.08 | 184 | 7.9 | 23.3 |
| **the shipped kernel now** (tiled-4, float16 x) | **3.89** | **476** | **20.4** | 23.3 |
| `x-aliased`, perfect operand locality | 2.91 | 635 | **27.3** | 23.3 |

And `probe-webgl-fetch-rate.js --ceiling-window=` puts this device's issue rate at **34.1 G fetch4/s
through a 4 KiB window, 25.7 through 64 KiB, 19.5 through 256 KiB and 19.4 through 1 MiB** (four
rounds each, spreads 1.03-1.14). The projection's working set under its tile mapping is about **75
KiB** - four row blocks of eight rows of float16 x, 57 KiB, plus eight output groups of codes, 18 -
so the rate it is entitled to is the 64 KiB figure, **25.7**.

**It achieves 20.4 of 25.7: 79% of this device's fetch issue rate at its own working set.** The
`x-aliased` arm's 27.3 is the 4 KiB regime, which is what a few KiB of operands buys and what no real
GEMM has.

🔴 **SO THE CEILING FOR THIS KERNEL FAMILY IS 23.3 x 25.7 = ABOUT 600 GFLOP/s, AND PARITY WITH THE
WGSL FOLD NEEDS 1303.** Both factors are measured: the intensity by the output budget this section
derives, the rate by a probe whose loop cannot be hoisted. At 476 the projections have **about 1.26x**
left in them, which on 49% of the fold is 1.12x - and a fold that captured all of it would be **19.5
s against the WGSL fold's 3.9**.

That is the clearest statement this file can make about why the gap does not close here, and it is
not an impossibility proof: it bounds the dominant family on this device at this working set. What it
rules out is closing a 5x gap by making these kernels better, because there is 1.26x in them.

🔴 **AND THE OUTPUT BUDGET ITSELF IS NOT 32 FLOATS EVERYWHERE IN WebGL2 - TRANSFORM FEEDBACK CARRIES
128, WHICH WAS BUILT AND IS WORTH 1.06-1.13x AT THE WORKING SET THAT MATTERS.** Every mechanism this
campaign measured moves a kernel WITHIN the fragment path's ceiling, so the question left over is
whether WebGL2 has a different ceiling anywhere. `probe-webgl-limits.js` says it does:
`MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` is **128** on this device against
`MAX_DRAW_BUFFERS * 4`'s 32, and `MAX_VERTEX_TEXTURE_IMAGE_UNITS` is 16, so a VERTEX shader can read
its operands and write four times as much. At 96 outputs an invocation (8 by 12, inside
`MAX_VERTEX_OUTPUT_COMPONENTS`'s 120) the intensity is **38.4 flops a fetch4** against the fragment
path's 21.33.

`probe-webgl-feedback-gemm.js` runs the SAME k loop in both stages, each at its own best tile, and
sweeps the working set because that is what decided every other question here:

| working set | transform feedback | fragment | ratio |
|---|---|---|---:|
| shared, a few KiB | 1074 GF/s at **28.0** G fetch4/s | 1986 at **93.1** | **0.54x** |
| **64 KiB** | **543-560 GF/s at 14.6** | 494-507 at 23.1 | **1.06-1.13x** |
| 256 KiB | 197 GF/s at 5.1 | 482 at 22.6 | **0.41x** |

**The intensity gain is real and the issue rate is where it goes.** The vertex stage reads 1.6x more
slowly than the fragment stage at 64 KiB and **4.4x** more slowly at 256, so the 1.8x of intensity
only nets out ahead in one band - and that band happens to be the projection's own working set of
about 75 KiB, where feedback reaches **543-560 GFLOP/s against the fragment path's 494-507**. Five
runs read **1.13, 1.06, 1.07, 1.07 and 1.10**, so the figure is 1.06-1.13 and not the 1.13 the first
run alone would have published - the third time in this session a single run nearly became a number.
The fragment arm is the check on this whole section: the shipped projection measures **476** at that
locality against this arm's 494-507.

So the answer to "is there a different ceiling in WebGL2" is **yes, and it is 1.06-1.13x**, before paying
for the thing that makes it unusable: transform feedback writes a BUFFER, nothing in WebGL2 samples a
buffer, and every kernel here feeds the next one through a texture - so each dispatch would owe a
buffer-to-texture copy of its whole output, which at 1.13x of the matmul is not a trade. **The output
budget is the binding constraint, raising it is possible, and raising it does not pay.**

🔴 **AND THE ONE WAY TO BREAK THE INTENSITY FACTOR WAS BUILT AND IS A 3.4x LOSS.** The cap assumes
**four floats a fetch**, and that is an assumption about the FORMAT rather than the API: the int5
weights have packed about twenty-five codes into one RGBA32UI texel since they were written, and x
gets four. Packing x as **eight float16s a texel** and unpacking with `unpackHalf2x16` halves its
side of `R + C`, which by this section's own arithmetic takes the intensity from 23.3 to **36.6** and
the ceiling from ~600 to ~940 GFLOP/s. Measured:

| arm | ms | GFLOP/s | G fetch4/s | flops a fetch4 |
|---|---:|---:|---:|---:|
| the shipped kernel | 10.14 | 182 | 7.8 | 23.3 |
| **`packed-x`** | **35.39** | **52** | **1.4** | **36.6** |
| `tiled-2` | 5.11 | 362 | 15.5 | 23.3 |
| **`tiled-2-packed-x`** | **9.88** | **187** | **5.1** | 36.6 |

**The intensity went up exactly as predicted and the rate fell by 3-11x.** Four `unpackHalf2x16` a
texel cost far more than the fetch they save - the codes get away with it because `decodeInt5x8` pays
its unpack once for EIGHT weights against this paying four unpacks for eight x values, while the x
side is read `R` times a chunk where the weight side is read `C`. So the 23.3 flops a fetch4 is not
merely the API's cap; on this device it is also the profitable maximum, and the ~600 GFLOP/s ceiling
stands with both of its factors measured and the obvious route around one of them closed.

🔴 **AND IT RECONCILES WITH `probe-webgl-gemm-ceiling.js`'s 1958-2003 GFLOP/s, WHICH IS NOT A
CONTRADICTION BUT THE SAME CURVE AT THE OTHER END.** That probe's operands are 32 KiB shared by every
fragment, so its fetches are served from the smallest cache and it reaches **87-89 G fetch4/s** - far
above the windowed probe's 34, because the windowed probe is built not to be hoistable. Three figures,
one curve: ~88 G fetch4/s when every fragment reads the same tiny operands, 34 through 4 KiB
non-hoistably, 25.7 through 64 KiB, 19.5 through 256. **A ceiling is a function of the working set,
and quoting the shared-operand end of it as "the backend's ceiling" is what made this section's
earlier 6x floor wrong in the first place.**

🔴 **AND THE INTENSITY CAP IS EXACTLY WHY THE 4x8x1 CONTRACTION BLOCK PAID.** 21.33 flops a fetch4 at
`R + C = 12` is the ceiling for any kernel here, and the contraction's 2 i x 4 j x 4 channel block
reached only **10.7** - because its four channels a fragment are not part of the register tile at
all, they are four separate k loops. One channel a fragment makes the tile 4 by 8 and the intensity
**21.33**, which is this paragraph's own ceiling, reached at last by the kernel that was furthest
from it.

🔴 **AND THE "6x FLOOR" THIS SECTION USED TO DERIVE FROM THAT IS RETRACTED, BECAUSE THE OTHER HALF
OF THE PRODUCT WAS NEVER MEASURED.** The old text multiplied 21.33 flops a fetch by a fetch rate of
about nine billion a second - inferred from the kernels themselves - and concluded that the
projections were AT the hardware's limit and the fold could not go below about 24 s. The intensity
cap is real. The rate was wrong by a factor of six.

`tools/gpu/probe-webgl-fetch-rate.js --ceiling-window=<texels>` measures the issue rate directly,
with a non-hoistable stride through a window of a chosen size and nothing else in the loop. Four
interleaved rounds, each arm's minimum, and `spreadWithinArm` beside it so a reader can see whether
the number settled:

| window | | issue rate | spread | `x 21.33` |
|---|---|---:|---:|---:|
| 256 texels, 4 KiB | | 82.6 | **1.45** | 1762 |
| 1024, 16 KiB | | **32.5** | 1.07 | **693** |
| 4096, 64 KiB | a realistic tile | **24.3** | 1.07 | **519** |
| 16384, 256 KiB | | 19.0 | 1.12 | 404 |
| 65536, 1 MiB | a streamed weight | 19.2 | 1.13 | 410 |

The two smallest windows do not settle - at 4 KiB the loop is probably being hoisted into registers -
so read the four from 16 KiB down. **The backend's ceiling at a realistic working set is 400 to 700
GFLOP/s, and the projections at 105 to 198 are at a QUARTER TO A HALF of it.** They are not at the
hardware's limit; they are missing. What caps this backend is the output budget TIMES the miss rate,
and only the first factor is fixed - which is why the two changes that moved these kernels most were
both LOCALITY (the fragment order at 1.49x, the tile-aware mapping at 3.29x) and neither was
arithmetic.

🔴 **AND THE INSTRUMENT ITSELF HAD TO BE FIXED TWICE BEFORE ANY OF THAT WAS TRUE.** Its first
version walked `i` modulo 16 through a 64-texel window, which a compiler that unrolls sixteen
iterations can hoist entirely - an "issue rate" that is really a register read. Its second ran each
arm once after two warm-up dispatches and reported **7.54, 16.88 and 17.04 G fetch4/s for the same
arm on three consecutive runs**, because an Apple GPU's clock ramps over a load that lasts fifty
milliseconds. Both numbers were quoted in this file before the rounds existed. A probe is an
instrument, and an instrument that has not been checked against itself is a number generator.

🔴 **AND THE SEARCH FOR WHAT THEY ARE MISSING ON HAS FOUR MORE NULLS.** Neither operand's size is
the limiter - swept at constant arithmetic through `bench-webgl-linear.js --shapes=`, a **16x**
range of weight (1 MiB to 64 KiB) moves the rate by 1.12x and a **16x** range of x (0.56 to 9 MiB)
by 1.12x, both flat at 102 to 140 GFLOP/s. Nor is it the render-target count, which is the opposite
of the obvious guess: eight RGBA32F targets is 128 bytes a pixel of tile memory, and the ceiling arm
measures **the same rate at one target and at eight**. Nor the row block, nor the fragment order,
nor a tile-aware mapping - those three are measured above. What is left is the STREAM COUNT: the
projection reads twelve rows per k-step where the ceiling arm reads four, and the probe's own
`2x4x1` arm (six streams) is 1.3x the rate of `8x4x1` (twelve) - the right sign, not the right size.
That is where the remaining 2 to 4x is, and it is unresolved.

🔴 **AND "2 TO 4x OF HEADROOM" IS ITSELF AN OVER-READING OF THE CEILING, WHICH IS THE THIRD TIME
THIS FILE HAS MADE THAT MISTAKE.** The ceiling arm has **every fragment reading the same 64 KiB
band**, which is the best locality a kernel can have and not a locality a GEMM can have: different
fragments MUST read different rows, or they would be computing the same output. So 24 G fetch4/s
bounds what the hardware can ISSUE, and says nothing about what a kernel whose fragments read
disjoint rows can reach. The measured facts are:

- the output budget caps intensity at 21.33 flops a `fetch4`, arithmetically;
- the issue rate with perfect locality is 19 to 32 G fetch4/s, measured;
- a 12-stream GEMM with per-fragment rows reaches 4 to 9, measured;
- and the distance between the last two is locality that is partly INHERENT to the access pattern
  rather than a defect to be fixed.

How much of it is inherent is not known. **The honest form of this section is therefore: the old 6x
floor was wrong because it multiplied by an unmeasured rate, the projections are not at the
hardware's issue limit, and how much of the remainder is reachable is an open question - not a
quantity this file can state.** Every attempt on it so far (the row block, the fragment order, a
tile-aware mapping, both operand sizes, the target count) has come back null or negative.

The first version of this paragraph is the mistake this file warns about everywhere else, made in
the place where it does the most damage: **an estimate presented as a bound, used to argue that
further work was pointless.** It was also the reason the ESMFold2 contraction sat at 0.4 GFLOP/s
unexamined for a whole campaign. A floor is a measurement or it is nothing.

A compute shader is still past this backend by staging a tile in workgroup memory, where a whole
workgroup reuses one load and no invocation's output count constrains anything; and it has f16
ARITHMETIC, where WebGL2's RGBA16F is storage only. Those remain real advantages - but they are
worth the distance between 198 GFLOP/s and about 2.4 TFLOP/s, not the distance this file claimed.

🔴 **AND THE f16 HALF OF THAT IS MEASURED NOW RATHER THAN ASSERTED.** GLSL ES 300 has no f16 type
but it has PRECISION QUALIFIERS, and a translator over Metal could map `mediump float` to `half` -
real 16-bit arithmetic at half the register pressure, which is exactly the lever the WGSL side has.
Nothing here had ever asked, because `PRELUDE` pins `highp` on float, int and sampler, so every
kernel in this backend is f32 whatever its storage. `tools/gpu/probe-webgl-precision.js` runs the
shipped eight-row projection geometry both ways and the answer is **the qualifier is PROMOTED**:
`relRms` is **exactly 0** over 4624x256->512, so the two arms are one kernel. **Read the relRms and
not the time** - the ratio came out 1.072, 1.068 and 0.992 over three runs, which is ordering noise
on two identical kernels, and reading that 1.07 as a win is the trap the probe's own header warns
about. So half arithmetic is not reachable from GLSL ES here - which is a real difference from the
WGSL side, and the one leg of the old floor argument that survives measurement.

🔴 **AND THE FIRST EXPLANATION WRITTEN HERE WAS WRONG, FROM EXACTLY THE MISTAKE THIS FILE WARNS
ABOUT EVERYWHERE ELSE.** It said the gap was the subgroup MATRIX units, on the evidence that
`tools/gpu/probe-subgroup-matrix.js` reports them present (f32 and f16 at 8x8x8). That is a
CAPABILITY PROBE, not a measurement, and `--tune=matrixLinear=false` on the WGSL fold settles it:

```
default                     foldMs 3941   diffusion 1178
--tune=matrixLinear=false   foldMs 3934   diffusion 1179
```

Re-measured on this tree with every matrix knob off at once - `matrixLinear`,
`attentionMatrix`, `attentionProjectMatrix`, `gridAttendMatrix`, `gridProjectMatrix`,
`opmMatrixContract` - **3928 ms against the default's 4130, at the same atom checksum 9506259**. Still
nothing, and now nothing across all six rather than one.

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

🔴 **AND THAT CLAIM HAS NOW BEEN TESTED DIRECTLY RATHER THAN RESTED ON, BECAUSE WebGL2 HAS A SECOND
OUTPUT BUDGET.** If the cap is `MAX_DRAW_BUFFERS * 4`, then a path with a bigger budget should break
it - and `probe-webgl-limits.js` finds one: `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` is
**128** on this device, four times 32, with `MAX_VERTEX_TEXTURE_IMAGE_UNITS` 16 so a VERTEX shader
can read its operands. `probe-webgl-feedback-gemm.js` runs the same k loop in both stages at 8 by 12
against 8 by 4, and at the projection's own working set it is **1.06-1.13x** over five runs - 543-560
GFLOP/s at 14.6 G fetch4/s against 494-507 at 23.1. **So the claim is confirmed in direction and
small in size**: the budget really is what caps the intensity (38.4 flops a fetch4 against 21.33,
exactly as predicted), and the vertex stage gives back 1.6x of issue rate to get it - 4.4x at a 256
KiB working set, where it is 0.41x. See "How close can this get" above for the sweep, and note that
transform feedback writes a BUFFER that nothing in WebGL2 samples, so each kernel would owe a
buffer-to-texture copy on top.

**Both halves of the cap are therefore measured, not argued**: raising the output budget is possible
and buys about a tenth, and raising the flops a fetch4 by packing the operands tighter is a **3.4x
loss** because the unpack costs more than the fetch it saves. The 32-float budget is not just the
API's limit here; it is the profitable one.

🔴 **AND THE DRAW COUNT IS NOT THE LEVER: A DRAW COSTS 22.9 us AND THE FOLD'S 54,381 OF THEM ARE
5.7%.** The denoiser's per-step profile is **41 kernels and about 1500 draws**, most of them small -
`bioemu2.cond-slice` alone is 192 calls a step - so the obvious suspicion is that the fold pays for
draws rather than for work, which only FUSING would fix. `probe-webgl-draw-floor.js` prices one:

| output texels | us a draw | effective |
|---:|---:|---:|
| 1 | 22.9 | - |
| 1,024 | 23.8 | 1.4 GB/s |
| 16,384 | 23.6 | 22.2 GB/s |
| 65,536 | 23.6 | **88.7 GB/s** |
| 262,144 | 28.9 | 290.2 GB/s |

**A draw of 65,536 texels costs what a draw of ONE texel costs** - up to about that size the work is
free and the cost is entirely fixed, which is the regime those small kernels live in. So the whole
draw bill is `54,381 x 22.9 us` = **1245 ms, 5.7% of a 20.5 s fold**, and fusing every small kernel in
the backend has a ceiling of **1.06x**.

🔴 **AND ELIMINATING 192 DRAWS A STEP CONFIRMS IT DIRECTLY: A WASH.** `bioemu2.cond-slice` is a copy -
the conditioning is one tensor of `6 * channels`, each consumer wanted its own piece, so
`piece(index)` allocated `rows * channels` and copied into it, six times a layer across 32 layers. The
pieces sit at a known offset in a known stride, so a consumer can address them in place; built, that
removed 192 slice draws a step plus the gate's separate product and sum. Four interleaved rounds:
**20529/20460/20474/20468 ms against 20572/20520/20451/20513** - 0.15% apart. Reverted. The draws
overlapped the GPU and the copies were cache-served, which is the same answer the framebuffer reuse
gave from the host side.

🔴 **AND SEVEN HYPOTHESES FOR THE REST CAME BACK NULL, WHICH IS THE EVIDENCE THAT THERE IS NO
EIGHTH IN THIS SHAPE OF KERNEL.** Each was built and measured, and each is a tool or a control arm
in the tree now rather than a note:

| hypothesis | arm | result |
|---|---|---|
| weight bytes | `--no-int5-weights`, `--no-half-weights` | 1.16x, 1.40x - **taken** |
| fragment order | `--no-row-fastest` | 1.23x - **taken** |
| weight cache misses | every chunk reads one texel | **1.02x** - no headroom left |
| x fetch count | x packed eight halves to a texel | **0.93x**, and it overflows f16 |
| accumulator registers | eight codes as two uints | **0.98x** |
| rows a fragment owns | R = 4 against R = 8 | **0.66x** |
| texture layout | `probe-webgl-2d-weight.js`, one matrix row per texture row | **1.01x** |
| too few fragments | the same probe's split-K, 2/4/8 | **1.01x** at 8x the fragments |
| output budget | `bench-webgl-tf-linear.js`, transform feedback | **1.00x** at 32 floats, 0.54x at 112 |

🔴 **AND SIX MORE CAME BACK NULL IN THE CONTRACTION, THE ONE KERNEL THAT IS NOT AT THE CEILING.** It
runs at **7.6 GFLOP/s** in a fold - 16.4% of it - and about 1 G fetches a second against the
projections' 6.8, on only 2x less arithmetic a fetch. The six, each with a measurement rather than
an argument:

| hypothesis | arm | result |
|---|---|---|
| fragment order | `--no-contract-channel-slowest` | **1.49x** - taken |
| the pair block's size | `--no-contract-ceiling` | **up to 4.8x** at the wrong size - taken |
| row alignment | a padded row stride, and lengths already aligned | **0.98x**, and L=64 is the SLOWEST |
| too few fragments | the channel sweep, 64 to 512 | GFLOP/s **FALLS** as fragments rise |
| fetches in flight | the four channel blocks fused into one k loop | **0.80x** - registers cost more |
| the operands pass's fetches | twenty `texelFetch` an output texel cut to nine | **1.00x** - not fetch-bound |

The alignment arm is the one to copy: it needed no padding code at all, because a length whose row is
ALREADY a power of two tests the same thing. **L=64 is 16 texels and runs at 8.9 GFLOP/s, the slowest
of the sweep, where L=68 at 17 texels runs at 12.1** - so the aligned lengths are the slow ones, and
that is independently why padding the contraction's stride lost.

🔴 **AND ITS 10.7 FLOPS A FETCH IS FORCED BY THE OUTPUT LAYOUT, WHICH CLOSES THE RETILING QUESTION.**
Its 32 floats must be four CHANNELS of eight pairs, because every consumer - the repack and the
centre norm - reads a pair's channels contiguously. Four channels of 2 by 4 pairs costs `4 * (2 + 4)`
= 24 fetches a k step where the projection's 8 by 4 costs 12, and 2 by 4 is already the best split of
eight pairs. Reaching 21.3 means one channel's 4 by 8 pairs with the output `vec4` over j - and then
the repack and the centre norm each read four texels an output instead of one, which is 1804 ms of
consumer going to about 7200 against 4400 saved. **So the retiling is a net loss before it is
written**, and the contraction's intensity is as high as this output layout permits.

🔴 **AND THE ONE THING THAT DOES MOVE IT IS THE SHAPE, WHICH IS NOT A LEVER.** The same kernel is
**24 GFLOP/s at 576 x 896 -> 1792 and 254 at 2048 x 2048 x 2048** - a tenfold difference in fetch
rate at identical intensity, identical layout and identical code. bioemu2's projections are the
small shape and the model decides that. It is why `bench-webgl2-tf-matmul.js`'s 1.6 TFLOP/s was
never a figure this backend could inherit: that is a square-2048 GEMM, and its timing is real (12.8
ms at one iteration, 10.7 at sixteen), but no fold dispatches one.

So the floor follows from measured numbers rather than from a feature list. The best fetch rate
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
wall                 968 s            45.9 s           3.9 s
  trunk              277.7 s          21.6 s           2.3 s
  distogram                           0.7 s
  20 denoiser calls  658 s            23.3 s           1.2 s
weights/activations  1656 MiB peak    409 MiB peak     815 MiB weights + 135 activations
```

🔴 **968 s TO 45.9 s ON A QUARTER OF THE MEMORY, AND NOT ONE REPORTED QUANTITY MOVES.** The 21.1x
is the row block, its fragment order, x's padded row stride, the contraction split, coalesced, transposed and
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

🔴 **AND IT IS 12x THE WGSL FOLD - WAS 246x - ON LESS MEMORY THAN THAT FOLD USES.** The trunk is
9x and the diffusion 19x, and the diffusion is where the remaining asymmetry lives: the WGSL denoiser
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
59-residue AF2 fold is 1.29 s against 0.28 s, 68-token bioemu2 is 46 s against 3.9 s on 412 MiB
against 950, and a
255-residue fold has not been measured at all. A visitor with no WebGPU gets a fold
instead of a refusal, which is the whole point of the tree; a visitor who waits four minutes for
something the other path does in fifteen seconds may not thank anyone. The multimer is also still
absent, so the page would have to offer a monomer-only backend. Those are product questions, so
they are written down here rather than decided in a commit.

## 🔴 2026-11-08: an AF3 fold on an M2, from 9.9 s to ~5.3 s - and what bounds it now

Measured with `tools/gpu/fold-af3-webgl.js --a3m=/tools/fixtures/test.a3m` (59 residues, an 8076-row alignment drawn
128 rows a pass as the page draws it, 48 + 4 blocks, 2 passes, 25 steps) in Chrome on an M2, against WebGPU's own warm
fold of the same input: **1.74 s**. Safari 16.3 on the page, ubiquitin with a real search: **34 s / 25 s -> 18 s / 9 s**
first / second fold. In the order they landed, each in its commit with its numbers:

| change | effect |
|---|---|
| a deep alignment drawn a pass at a time (`foldMsa`) | a real search folded at all - it was REFUSED, "(8077 rows)" |
| a bounded, LRU texture pool (`Storage.poolLimitBytes`, 128 MiB) | 9.9 -> 7.7 s; Safari's second fold 25 -> 14 s |
| the outer product mean contracting the sequences first | trunk 2.85 -> ~1.9 s, and a 512/1024-row `max-msa` no longer overflows |
| wide 59-row projections blocked; q/k/v/gate fused; AdaLN conditioning batched across all 24 blocks | diffusion 4.3 -> 3.5 s |
| residuals in the gate draw; pair biases and the atom encoder's conditioning kept for the walk | diffusion 3.5 -> 3.1 s |
| weights from 4,096 floats in float16 (`AF3_HALF_WEIGHT_FLOOR`) | 5.55 -> 5.27 s |
| a narrow output (< 32) never blocks | trunk 1.74 -> ~1.71 s |

🔴 **`--double` READS FREE ON THIS GPU, AND IT SAID THE GPU WAS NOT THE COST.** It was the first attribution here, and
it is the trap this file already records for bioemu2: hidden-surface removal discards the first of two identical
draws. `--skip=<regex>` (`Execution.skipLabels`) removes a family's draws instead, which nothing can discard, and every
number above was found with it. Skipping EVERY draw leaves ~1.5 s of a ~5.5 s fold: the rest is GPU, and the host's
~47 us a draw (`probe-webgl-draw-floor.js`) overlaps it - removing 96 draws a step moved nothing.

🔴 **A KERNEL BENCH'S WIN DID NOT REACH THE FOLD, FOUR TIMES.** Out-major weights transposed to in-major (the bench:
2x), the loop bound baked as a literal, a k-split summed by the BLEND unit (`probe-webgl-ksplit-chain.js`: 1.16x at
59 rows, a loss at 3481), and uniform-block weights in Chrome (1.23x on the kernel) - each was taken through a whole
fold and moved it by nothing measurable. What the dependent chain probe does show is the rate itself: a 59-row
projection ~250 GFLOP/s and a 3481-row one ~400, chain or not, so the diffusion is about 1 GFLOP a block at ~250
GFLOP/s - most of its ~5 ms a block. And `bench-webgl-linear.js` had measured FLOAT32 weights all along, which no fold
holds; `--half-weights` is the configuration a fold runs.

🔴 **SAFARI IS NOT CHROME, AND THREE THINGS ONLY SAFARI SHOWED** (`tools/run-in-safari.py`, `tools/fold-in-safari.py`):

- **`mediump` is promoted there too** (`probe-webgl-precision.js`, relRms exactly 0): no f16 arithmetic from GLSL ES in
  either browser.
- **It refuses `readPixels` as FLOAT from an RGBA16F attachment**, which Chrome takes.
- **It draws with less than 16 KiB of uniform blocks IN TOTAL while reporting 192**
  (`probe-webgl-uniform-budget.js`: 12 x 16,384 reported; one 16,000-byte block draws, one 16,384-byte block or
  4 x 4,096 do not - GL_INVALID_OPERATION at the draw and nothing earlier). So the uniform-block weight this file's
  "A UNIFORM BLOCK READ IS 2.1x" section declined for bioemu2 is out for AF3 in the browser that matters: a 128 x 128
  pair weight is 64 KiB. 🔴 **AND `probe-webgl-ubo-weight.js` FIRST REPORTED 2.22x THERE FROM DRAWS THAT NEVER RAN**: no
  `getError`, a skipped draw is a fast one, and its chain had decayed to ~3e-5 where an unwritten target compares as
  agreement. A probe that cannot tell "faster" from "refused" is not a measurement; both probes check errors now.

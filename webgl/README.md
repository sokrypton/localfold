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

Still to come, in the order they are needed: MSA COLUMN attention (the same block over the sequence
axis); AF2's separate GLOBAL attention for the extra-MSA stack; triangle ATTENTION; then the
Evoformer block that drives all of them, the stack, the structure module, and the confidence and
distogram heads. After AF2: AF3's pairformer and diffusion, ESMFold2, bioemu2.

Every operation so far is checked against TWO references - an f64 CPU one written from the
operation, and the WGSL kernel run in the same page - because a single reference cannot say which
side is wrong. That has already paid for itself twice: the triangle contraction's incoming
direction was the pair-transpose of itself and only the WGSL arm could see it, and the attention
block read 1e-3 against WGSL while reading 2.5e-7 against f64, which was an unpinned precision axis
on the WGSL side rather than a defect here. After AF2: AF3's pairformer and diffusion, ESMFold2, bioemu2. Nothing is wired to a weight
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

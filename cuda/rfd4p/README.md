# RFD4-Proteina on CUDA

The native port of the design model the page runs in `webgpu/rfd4p/`. **It is a trunk and nothing
else so far** — one pass, held to `cpu/rfd4p/reference.js` stage by stage. There is no sampler, no
decoder and no output structure yet, so this does not generate anything.

## What runs, and what it is held to

```
node cuda/rfd4p/export_weights.mjs cuda/rfd4p/weights-f32          # 558 tensors, 714 MB
node cuda/rfd4p/export_input.mjs  cuda/rfd4p/data --oracle=cuda/rfd4p/oracle
bash cuda/build.sh rfd4p
./cuda/rfd4p/localfold-rfd4p cuda/rfd4p/data --weights=cuda/rfd4p/weights-f32 --oracle=cuda/rfd4p/oracle
```

On `custom_cath_monomer` (54 tokens, the one preset that needs no structure), against the float32
checkpoint, all 37 stages — the three init products, the single and pair state after each of the 14
layers, and the five outputs:

| | one recycle | two recycles |
|---|---|---|
| exact (the default) | **5.69e-07** | **5.19e-07** |
| `--fast` (TF32 and f16) | **9.28e-04** | **9.36e-04** |

Each stage is compared the moment it is produced, so the first one that moves is the one named. The
two-recycle arm is the one that exercises the cycle projections and the second conditioning; three of
the bugs below were found by it or by `o/s0` rather than by a finished trunk that was merely wrong.

## Speed

A trunk pass at 100 tokens and 6 recycles, on this A100, against PyTorch on the same card and the
**same captured batch** — hooked out of a real `rfd4-generate` run, so it is their own featurised
input and their own 6 cycles, not something built here:

| | ms a pass | of which features | network | precision |
|---|---:|---:|---:|---|
| PyTorch eager | 265.4 | | | f32 |
| PyTorch compiled | 158.7 | | | f32 |
| **PyTorch compiled, its own default precision** | **65.3** | 8.2 | **57.1** | bf16 matmuls |
| cuda/rfd4p, exact | 157.8 | — | 157.8 | f32 |
| **cuda/rfd4p, `--fast`** | **49.5** | — | **49.5** | TF32 + f16 |

🔴 **THE HONEST NUMBER IS 1.15x, NOT 5x.** Against PyTorch's *production* path - compiled, at the
precision upstream's own config selects - this port's network is 49.5 ms against 57.1. The 5.4x is
against `quickstart`, which turns compilation off and is upstream's first-run preset, not how anyone
generates. A ratio quoted without naming the other side's configuration is not a measurement, and
this file had it wrong for a while.

🔴 **AND THEIR FORWARD BUILDS ITS OWN FEATURES WHERE THIS ONE IS HANDED THEM** - 8.2 ms of their
65.3, measured by wrapping `cond_factory`, `init_repr_factory` and `pair_repr_builder`. That work is
inside their number and outside ours, and a finished port owes it. Counting it against us, the whole
trunk call is 49.5 + (unmeasured) against 65.3.

At matched f32 and both optimised, this port and a compiled PyTorch are the same speed to within a
percent - 157.8 against 158.7. Everything beyond that is arithmetic, which is why `--fast` is a flag
with its own oracle bound.

### How it scales

Both arms at 6 recycles, the `--length=N` unconditioned segment:

| n | exact | `--fast` | ratio |
|---:|---:|---:|---:|
| 54 | 72.5 | 29.5 | 2.46 |
| 100 | 157.4 | 50.0 | 3.15 |
| 200 | 533.0 | 147.2 | 3.62 |
| 400 | 2046.9 | 509.1 | 4.02 |

The reduced-precision arm pulls further ahead with length, which is what a pair track that grows as
n^2 and contracts as n^3 should do: the longer the chain, the more of the pass is GEMM and the less
is launch overhead.

### What the optimisation round moved, in order

Starting point 190.8 ms. Every step was profiled, kept only if it measured, and checked against all
four oracle arms.

| | ms | why |
|---|---:|---|
| the f32 baseline | 190.8 | GEMMs 67% of the pass at 16.0 TFLOP/s against a 19.5 f32 ceiling |
| TF32 behind `--fast` | 98.8 | nothing to win in the kernels, everything in the arithmetic |
| the triangle's four projections into one, in plane layout | 80.8 | `toPlanesK` was 14.8%, larger than any GEMM, and is now nothing |
| residuals and biases fused into their kernels | 75.0 | `addIntoMaskedK` 7.0% and `addBiasK` 5.0% over 7403 launches |
| the f16 flash path | 66.5 | `flashGridF32` is the one-thread-a-query reference kernel |
| a tiled transpose | 65.3 | the naive one read with consecutive threads `pairs` apart |
| the conditioning hoisted out of the recycle loop | 62.3 | 504 GEMMs a pass recomputing the same numbers; 168 will do |
| five projections into one, four into one | 60.6 | everything reading one input is one GEMM |
| the triangle's planes in half | 53.0 | 56 MB written and read back a triangle, 84 triangles a pass |
| the pair transition's hidden in half | 49.5 | the largest buffer left in an update |
| the pair bias fused into one kernel | 49.8 at n=100, **555.9 at n=400** | a wash on a short chain, 3.5% at a long one |
| a long form past 40,000 pairs: the gate transposed and the contraction's output narrowed | **509.1 at n=400**, 50.0 unchanged at n=100 | both choices invert with length |

The GEMMs now run at ~144 TFLOP/s against this card's 156 TF32 ceiling, and the kernel total is
within a few percent of the wall clock, so there is no idle to recover: what is left is bytes.

### Measured and NOT taken

🔴 Each of these looked obviously right and is slower. They are here so nobody spends the afternoon
again.

- **The plane transpose fused into the LayerNorm after it**, 66.5 → 73.4. One thread a pair looping
  the channels makes every load coalesced and leaves ~39 blocks at n = 100, against `layerNormRegK`'s
  warp a row with the row in registers. The coalescing argument was right; occupancy decided it.
- **The contraction writing its output in half AT A SHORT CHAIN**, 50.0 → 51.1 at n=100. At n=400 it
  is 513.5 → 507.6, so it is taken there and not here; it rides the same `longForm` switch as the
  gate.
- **The fused pair-bias kernel at a SHORT chain.** It is a win at the lengths people design at
  (n=200 151.9 → 147.8, n=400 575.9 → 555.9) and exactly nothing at n=100, where the materialised
  norm it removes is small beside the launch overhead it does not. Kept for the lengths that matter;
  measure a kernel at the size it will run, not at the size that is convenient.
- **Avoiding the strided read of the triangle's gate**, either way round: a tiled transpose of the
  gate first is 53.0 → 54.1, and giving `linear_g` its own row-major GEMM off the half-normed input
  is 54.9. A strided element costs a whole transaction whatever its width, and both ways of dodging
  that stride cost more than it does.

## The four things that were written rather than reused

Most of this port is `cuda/ef2/src/ops.cuh`'s GEMMs and LayerNorm, `cuda/af3/src/flash.cuh`'s
attention and `cuda/af3/src/common.cuh`'s allocator. Four pieces had to be its own, and each is a
place where the obvious reading is wrong:

- **RMSNorm.** Nothing in `cuda/` had one. It is `layerNormRegK` with the mean reduction deleted.
- **adaLN.** Not AF3's `SingleCond*` packing and not ESMFold2's DiT-style `(C, 6C)` off `silu(c)`:
  RFD4 has two separate matrices, a sigmoid on the scale, no `+ 1` and no silu.
- **SwiGLU with the halves the other way round.** `ops.cuh`'s `swigluK` is `silu(a) * b`; this model's
  is `silu(b) * a`. Reusing the other one returns the right shape and the wrong answer.
- **The triangle at a hidden width that is not the pair width.** `ef2`'s and `af3`'s both assume
  `projection` is `(C, 2C)` and contract at the pair width; RFD4 goes 384 → 256 → 384.

## The traps this port has already paid for

🔴 **THE QK NORM IS OVER THE WHOLE 768, NOT PER HEAD.** `pair_bias_attn.py` norms the concatenated
projection before it is split into twelve heads of 64.

🔴 **THE TRIANGLE'S CONTRACTION CANNOT READ A CHANNEL-INTERLEAVED TENSOR.** A column-major GEMM
operand's column must be contiguous, and in `[i][k][C]` the elements a channel needs are `C` apart.
Handing cuBLAS `lda = C` reads columns out of the wrong channels and returns a full tensor of
plausible numbers — `o/z0` at relRMS 8.3e-01 while every token stage was exact.

🔴 **`flashGridF32` DEREFERENCES ITS MASK WITHOUT CHECKING IT**, where the f16 kernels take a `MASKED`
template bool — so a null is an illegal access rather than an unmasked attention. It is a KEY mask,
`mask[row * n + j]`; the reference's per-PAIR masking rides in the bias instead.

🔴 **AND THE f16 KERNELS WORK IN THE LOG2 DOMAIN.** They fold `scale * LOG2E` into q and leave the
bias alone, where the f32 kernel is in the natural domain and wants `scale` 1. An unscaled bias on
the half path is a softmax at the wrong temperature: every logit finite, every weight wrong.

🔴 **ITS BIAS PLANE IS `n * biasStride`, NOT `stride * stride`.** Padding the row count as well as the
column count sends every head after the first to the wrong place, quietly, inside the allocation.

🔴 **THE TWO HALVES OF THE PAIR UPDATE'S `linear_x` ARE INDEXED BY DIFFERENT TOKENS.** The first is
broadcast along i, so it is indexed by **j**, and the second by **i**. Swapping them transposes the
write and still produces a plausible pair tensor.

🔴 **A FUSED PROJECTION'S ROW STRIDE IS NOT THE ONE IT REPLACED.** `packVGK` had 3W baked in from when
q/k/v came from one GEMM and the gate from another; at 4W it reads the wrong token's v and returns a
full, finite, wrong attention. And `stacked()` returned a reference to a static string, so asking it
for a weight and then a bias left both names pointing at the bias.

## What is not done

- **The sampler, the decoder and an output structure** — i.e. generating anything. The flow loop is
  `shared/rfd4p/flow.js`, the decoder is the autoencoder's 12 layers, and both want their own
  stage-by-stage bar before they are believed.
- **The int8 bundle.** `model-rfd4p-int8/` has a `manifest.json` and is the shipping path through
  `Model::loadBundle`, as `cuda/bioemu2` does; this reads the float32 export only.
- **A native featuriser.** `export_input.mjs` drives the page's own JS; the C++ twin under
  `cuda/featurise/` comes with the `--job` form and the `lf::standalone::main` dispatch.
- **A gate.** There is no `gate.py` or `gate-baseline.json` yet, so nothing re-runs these numbers.
- **Any case but one, and any length but three.** `custom_cath_monomer` is the only preset that builds
  without fetching a structure — no ligand, no nucleic chain, no binder — and `--length=N` makes an
  unconditioned segment for sizing and timing only.
- **A padded input**, so the masking paths are written and unexercised.

🔴 **AND TWO BRANCHES ONLY RUN PAST 40,000 PAIRS, WHICH NO ORACLE CASE REACHES.** `LOCALFOLD_BIG`
forces the long form at any size - as `cuda/af3`'s big-input paths are forced - and the gate runs
every oracle case both ways for exactly that reason. An untested branch that fires only on inputs
nobody gates is how a port grows a second, wrong implementation.

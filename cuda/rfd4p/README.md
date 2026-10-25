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
checkpoint: **every one of 37 stages inside 6e-7 — worst `o/z10` at 5.97e-07 at one recycle and
5.42e-07 at two**, the second arm being what exercises the recycle projections and the second
conditioning. The pass is 0.47 s wall including loading and uploading 714 MB of weights, against the
JS reference's 64 s.

The stages are the three init products (`c`, `sInit`, `zInit`), the single and pair state after each of
the 14 layers, and the five outputs — so the first stage that moves is the one named, and a later
stage's residual is its own fault plus everything upstream.

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
plausible numbers — `o/z0` at relRMS 8.3e-01 while every token stage was exact. One `n x n` plane a
channel, which is why AF3's pairtrack transposes before it contracts.

🔴 **`flashGridF32` DEREFERENCES ITS MASK WITHOUT CHECKING IT**, where the f16 kernels take a `MASKED`
template bool — so a null is an illegal access rather than an unmasked attention. It is a KEY mask,
`mask[row * n + j]`; the reference's per-PAIR masking rides in the bias instead.

🔴 **ITS BIAS PLANE IS `n * biasStride`, NOT `stride * stride`.** Padding the row count as well as the
column count sends every head after the first to the wrong place, quietly, inside the allocation.

🔴 **THE TWO HALVES OF THE PAIR UPDATE'S `linear_x` ARE INDEXED BY DIFFERENT TOKENS.** The first is
broadcast along i, so it is indexed by **j**, and the second by **i**. Swapping them transposes the
write and still produces a plausible pair tensor.

## What is not done

- **The sampler, the decoder and an output structure** — i.e. generating anything. The flow loop is
  `shared/rfd4p/flow.js`, the decoder is the autoencoder's 12 layers, and both want their own
  stage-by-stage bar before they are believed.
- **The int8 bundle.** `model-rfd4p-int8/` has a `manifest.json` and is the shipping path through
  `Model::loadBundle`, as `cuda/bioemu2` does; this reads the float32 export only. Once both load,
  `--compare-weights=` gives the int8 decode's per-tensor relRMS against f32 for free.
- **A native featuriser.** `export_input.mjs` drives the page's own JS; the C++ twin under
  `cuda/featurise/` (held to these bytes by `tools/check-native-featuriser.py`) comes with the
  `--job` form, and with it the `lf::standalone::main` dispatch and a `fold` wrapper.
- **A gate.** There is no `gate.py` or `gate-baseline.json` yet, so nothing re-runs the numbers above.
- **Any case but one.** `custom_cath_monomer` is the only preset that builds without fetching a
  structure, so the port has seen 54 tokens of pure-spec protein and nothing else - no ligand, no
  nucleic chain, no binder, and no length past 54.
- **A padded input.** Every case here is one full chain, so the masking paths — the bias's `-1e4` for
  an invalid pair, which the reference *replaces* a score with rather than adding to — are written but
  unexercised.
- **Speed.** Nothing has been profiled or tuned. Every linear reads the checkpoint's `[out, in]`
  through `CUBLAS_OP_T` rather than being pre-transposed, the attention is on the f32 reference
  kernel rather than the f16 flash path, and the triangle transposes to planes and back around every
  contraction.

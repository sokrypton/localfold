# RFD4-Proteina on CUDA

The native port of the design model the page runs in `webgpu/rfd4p/`. **It generates**: noise in, a
PDB out, with the trunk, the flow sampler and the autoencoder decoder all held to
`cpu/rfd4p/reference.js` stage by stage.

```
node cuda/rfd4p/export_input.mjs cuda/rfd4p/data --generate
./cuda/rfd4p/localfold-rfd4p cuda/rfd4p/data --weights=cuda/rfd4p/weights-f32 \
    --generate=cuda/rfd4p/data-gen --out=design.pdb --fast
generated 54 residues, 420 atoms over 60 steps (120 trunk passes) -> design.pdb
  sampling 0.52 s, with the decode and the write 0.53 s
```

And it is a protein rather than a ball of atoms: 54 CA, consecutive CA-CA **median 3.81 A with all
53 inside [3.6, 4.0]**, radius of gyration 10.3 A for a 54-mer.

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
| exact (the default) | **5.80e-07** | **5.32e-07** |
| `--fast` (TF32 and f16) | **9.85e-04** | **9.48e-04** |

Each stage is compared the moment it is produced, so the first one that moves is the one named. The
two-recycle arm is the one that exercises the cycle projections and the second conditioning; three of
the bugs below were found by it or by `o/s0` rather than by a finished trunk that was merely wrong.

## Speed

### A whole generation

100 tokens, 60 steps, **180 trunk passes on both sides** (the `medium` recycle schedule, which is
what `rfd4-generate` runs), one design, same A100. Timed from the first trunk call to the end, so
model loading and reading the input file are outside it on both sides, and the per-step feature
assembly, the sampler and the decode are inside it on both:

| | sampling + decode | |
|---|---:|---|
| PyTorch eager | 16.00 s | `inference_experiment=quickstart`, which turns compilation OFF |
| PyTorch compiled, first design | 10.91 s | about half of it is `torch.compile` |
| **PyTorch compiled, warm** | **5.34 s** | the second design of a two-replicate run |
| cuda/rfd4p, exact f32 | 4.80 s | |
| **cuda/rfd4p, `--fast`** | **1.50 s** | **3.6x the warm compiled run** |

🔴 **3.4x IS THE NUMBER, AND THE OTHER THREE ROWS ARE WHY IT TOOK FOUR RUNS TO GET.** Against the
preset upstream ships for first runs it is 10.1x, and that preset says in its own header that it
disables compilation. Against a compiled run whose FIRST design pays for the compiling it is 1.9x.
Only the warm row is production against production. 🔴 And our exact f32 arm at 4.80 s is a TIE
with it, which is the same thing the per-pass trunk comparison below says: the margin is
arithmetic, not kernels.

Whole process, including loading 1.2 GB of weights: **2.08 s against 22.14 s**.

Per pass, the two loops are 8.8 ms (ours) and 29.7 ms (theirs) against isolated trunk costs of 7.8
and 10.9 ms - so our loop adds 13% over its own trunk and theirs adds 170%. That gap is mostly
feature assembly: their factories rebuild every feature each call where this port overwrites the 35
columns that move and the 30 distance bins, which is what `webgpu/rfd4p/generate.js` does too.

### A trunk pass, by itself

A trunk pass at 100 tokens and 6 recycles, on this A100, against PyTorch on the same card and the
**same captured batch** — hooked out of a real `rfd4-generate` run, so it is their own featurised
input and their own 6 cycles, not something built here:

| | ms a pass | of which features | network | precision |
|---|---:|---:|---:|---|
| PyTorch eager | 265.4 | | | f32 |
| PyTorch compiled | 158.7 | | | f32 |
| **PyTorch compiled, its own default precision** | **65.3** | 8.2 | **57.1** | bf16 matmuls |
| cuda/rfd4p, exact | 156.0 | — | 156.0 | f32 |
| **cuda/rfd4p, `--fast`** | **43.9** | — | **43.9** | TF32 + f16 |

🔴 **AND THEIR FORWARD BUILDS ITS OWN FEATURES WHERE THIS ONE IS HANDED THEM** - 8.2 ms of their
65.3, measured by wrapping `cond_factory`, `init_repr_factory` and `pair_repr_builder`. In the
generation above both sides pay it.

At matched f32 and both optimised, this port and a compiled PyTorch are the same speed to within a
percent - 156.0 against 158.7. Everything beyond that is arithmetic, which is why `--fast` is a flag
with its own oracle bound.

### How it scales

Both arms at 6 recycles, the `--length=N` unconditioned segment:

| n | exact | `--fast` | ratio |
|---:|---:|---:|---:|
| 54 | 73.2 | 28.9 | 2.53 |
| 100 | 156.0 | 46.6 | 3.35 |
| 200 | 521.4 | 126.8 | 4.11 |
| 400 | 1994.5 | 431.5 | 4.62 |

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
| the elementwise kernels vectorised | **47.5 at n=100, 468.8 at n=400** | one `half` a thread is a 2-byte access |
| the long form's gate projected row-major instead of transposed | **442.6 at n=400**, 47.5 unchanged at n=100 | a second GEMM beats a transpose once the tensor is big |
| the pair-bias kernel's loads vectorised | 47.3 / 440.4 | four channels a lane, not one |
| the pair tensor itself in half | **46.6 at n=100, 431.5 at n=400** | 2-3%, and it halves the biggest tensor |
| the half pair tensor's kernels vectorised too | **43.9 at n=100, 401.2 at n=400** | narrowing z had silently dropped them to one element a thread |

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

- **An end-to-end trajectory comparison.** Every PIECE is held to the reference - the trunk, the
  decoder, and the sampler's schedules are ported from `shared/rfd4p/flow.js` - but the LOOP that
  joins them is checked only by the structure it produces being a protein. `--gen-oracle` runs
  `generateRfd4pCpu` beside it at a step count somebody can wait for; 🔴 do not run it at the
  preset's 60, which is `depths` JS trunk passes at about a minute each.
- **The int8 bundle.** `model-rfd4p-int8/` has a `manifest.json` and is the shipping path through
  `Model::loadBundle`, as `cuda/bioemu2` does; this reads the float32 export only.
- **A native featuriser.** `export_input.mjs` drives the page's own JS; the C++ twin under
  `cuda/featurise/` comes with the `--job` form and the `lf::standalone::main` dispatch.
- **A gate.** There is no `gate.py` or `gate-baseline.json` yet, so nothing re-runs these numbers.
- **Any case but one, and any length but three.** `custom_cath_monomer` is the only preset that builds
  without fetching a structure — no ligand, no nucleic chain, no binder — and `--length=N` makes an
  unconditioned segment for sizing and timing only.
- **A padded input**, so the masking paths are written and unexercised.
- **Anything but an unconditional monomer through `--generate`.** The export REFUSES an input whose
  per-step features are more than the noised state, rather than silently freezing a feature that
  should move - coordinate groups to permute and chirality frames both move per step and neither is
  handled. That is the same condition `webgpu/rfd4p/generate.js` tests before taking its own
  on-device loop.
- **The C++ random source.** Box-Muller runs through V8's `Math.log`, and `cuda/featurise/jsmath.h`
  carries sin and cos but no log, so the noise is REPLAYED from the exporter rather than redrawn:
  one ulp in one draw diverges a sixty-step trajectory. The draw ORDER is a function of the schedule
  alone, so the exporter reproduces the stream without running the model - the same thing
  `tools/oracle/dump_rfd4p.py` does for the JS sampler. Porting fdlibm's `__ieee754_log` into
  `jsmath.h` is ~40 lines and would make the port's own sampler exact.

🔴 **AND TWO BRANCHES ONLY RUN PAST 40,000 PAIRS, WHICH NO ORACLE CASE REACHES.** `LOCALFOLD_BIG`
forces the long form at any size - as `cuda/af3`'s big-input paths are forced - and the gate runs
every oracle case both ways for exactly that reason. An untested branch that fires only on inputs
nobody gates is how a port grows a second, wrong implementation.

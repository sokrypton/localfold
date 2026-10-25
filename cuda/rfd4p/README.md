# RFD4-Proteina on CUDA

The native port of the design model the page runs in `webgpu/rfd4p/`. **A job file in, a PDB out,
in one process with no JavaScript** - the featuriser, the trunk, the autoencoder's two halves and
the flow sampler are all here, and all held to `cpu/rfd4p/` and `shared/rfd4p/` stage by stage.

```
./cuda/rfd4p/localfold-rfd4p --job=<job.json> --weights=cuda/rfd4p/weights-f32 --out=design.pdb --fast
generated 53 residues, 265 atoms over 25 steps (52 trunk passes) -> design.pdb
  sampling 0.42 s
```

A real job end to end - 100 residues, 60 steps, from the file to the PDB - is **1.74 s of wall
clock**, of which 1.04 is the sampler and the rest is the weights and the featurisation.

All ten of RFD4-Proteina's shipped presets generate: unconditional, fold-conditioned, motif
scaffolding indexed and unindexed, protein-protein interface, DNA binding, a YAML binder job,
small molecule, partial diffusion and inverse folding.

And the result is a protein rather than a ball of atoms: 54 CA, consecutive CA-CA **median 3.81 A
with all 53 inside [3.6, 4.0]**, radius of gyration 10.3 A for a 54-mer.

## What runs, and what it is held to

```
node cuda/rfd4p/export_weights.mjs cuda/rfd4p/weights-f32 --include=all   # 4,371 entries, 1.7 GB (trunk, decoder, encoder)
node cuda/rfd4p/export_input.mjs  cuda/rfd4p/data --oracle=cuda/rfd4p/oracle
bash cuda/build.sh rfd4p
./cuda/rfd4p/localfold-rfd4p cuda/rfd4p/data --weights=cuda/rfd4p/weights-f32 --oracle=cuda/rfd4p/oracle
```

On `custom_cath_monomer` (54 tokens, the one preset that needs no structure), against the float32
checkpoint, all 37 stages — the three init products, the single and pair state after each of the 14
layers, and the five outputs:

`gate.py` is what re-runs all of it - **25 checks, all green on this A100**: the trunk's stages in
five arms at one and two recycles, the encoder, FOUR sampling loops against the reference's own
trajectory (the monomer, partial diffusion, inverse folding and the small molecule's chirality),
the native featuriser against the JavaScript's bytes, its refusals, the one-process `--job` form,
and a timing.

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
| cuda/rfd4p, `--fast`, first design | 1.43 s | pays cuBLAS's heuristics and the lazy f16 mirrors |
| **cuda/rfd4p, `--fast`, warm** | **1.24 s** | **4.3x the warm compiled run** |

🔴 **4.3x IS THE NUMBER, AND THE OTHER ROWS ARE WHY IT TOOK FIVE RUNS TO GET.** Against the
preset upstream ships for first runs it is 10.1x, and that preset says in its own header that it
disables compilation. Against a compiled run whose FIRST design pays for the compiling it is 1.9x.
Only the two warm rows are production against production, and they are matched on purpose:
`--designs=N` generates more than once so the SECOND can be timed, which is how PyTorch's own warm
number is taken. Comparing their warm against our cold would have under-read us by 11%.
🔴 And our exact f32 arm at 4.80 s is a TIE with their warm compiled run, which is the same thing
the per-pass trunk comparison below says: the margin is arithmetic, not kernels.

Whole process, including loading 1.2 GB of weights: **2.00 s against 22.39 s** - **11.2x cold**,
because their cold pays for `torch.compile` (10.2 s of that 22.39) where ours pays only for a read.
The warm row above is the stricter comparison and the one to quote.

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
| **cuda/rfd4p, `--fast`** | **40.0** | — | **40.0** | TF32 + f16 |

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
| eight lanes a pair in the bias kernel, and the contraction's output narrow to its norm | **43.2 at n=100, 370.5 at n=400** | the reductions, not the loads; and a widening transpose nothing needed |
| `xProj` narrowed where the projection has no bias | **42.7 at n=100, 357.3 at n=400** | it was the biggest thing the gated add read |
| the token track on f16 | **40.6 at n=100, 352.1 at n=400** | it was the last thing still running on TF32 |
| each pair stage leaving behind the next one's LayerNorm, past 40,000 pairs | **340.1 at n=400**, 40.7 unchanged at n=100 | z was written and read straight back, three times an update |
| the conditioning projections on f16 | **40.0 at n=100** | small each, 5,040 of them a generation |

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
- **CUDA GRAPHS, which is the one I most expected to pay.** At n=100 a pass is 578 kernel launches
  averaging 12.5 us, 104,130 of them in a generation, and nsys puts the card at 88% busy - 12% of
  the sampling spent between kernels. Captured once per distinct recycle depth and replayed, with
  the conditioning copied into fixed buffers and the sampler's scalars moved behind a device
  pointer so a frozen graph could still see them: **1.46 s with, 1.45 s without.** Nothing. The
  gaps are not launch overhead that a graph can remove.
  🔴 AND THE FIRST ATTEMPT LOOKED 9% SLOWER FOR A REASON THAT WAS MINE: the warm-up that sizes
  scratch before a capture ran one pass A DEPTH, 21 extra trunk passes out of 180, inside the timed
  region. The buffers do not depend on the depth - only whether the second conditioning pack exists
  does - so one pass at the deepest recycle is enough. A setup cost inside the measurement is not a
  slow kernel.
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

## The published int8 bundle

`--int8=model-rfd4p-int8` reads the bundle `sokrypton/rfd4-proteina-webgpu` publishes, as the page
does: 11 core shards, 331 MB, int8 codes with a float16 scale a block of 64, decoded on the device
into the same names the float32 export writes. The architecture scalars come from its `config.json`
where the export writes them as `m meta/...`, so the port sizes itself identically either way.

🔴 **IT IS NOT A SPEED FEATURE AND THE MEASUREMENT SAYS SO: 2.01 s cold against the float32
export's 1.95.** Reading a third of the bytes buys nothing here because the float path already
mmaps and uploads asynchronously, and both land at about half a second of startup. What it buys is
**distribution**: 331 MB instead of 1,226 on disk, and no NVIDIA checkpoint at all - the port runs
from the published bundle alone, so nobody has to accept terms to fold with it.

Quality: against an oracle computed from the FLOAT checkpoint it reads 1.6e-02, which is the
bundle's quantisation and upstream's decision rather than this port's arithmetic. The structure it
generates is **0.15 A of mean CA from the float one at the same seed**, with identical CA-CA and
radius of gyration. The gate runs it as its own arm wherever the shards are on disk.

🔴 **AND THE ENCODER'S FIVE SHARDS ARE NOT FETCHED.** The bundle splits 331 MB of `core` from 138 MB
of `encoder`, the latter read only for partial diffusion and inverse folding, which this port does
not do. A bundle without them is complete rather than broken, and a tensor the trunk or decoder
actually wants is still caught by name at first use.

## The sampling loop, against the reference's own

`--generate=<dir> --gen-oracle=<dir>` runs `generateRfd4pCpu` beside the port and compares the
trajectory at EVERY step, plus the loop's input before it: the replayed noise, the step-0 sequence
and pair features, and the conditioning.

🔴 **THIS IS THE ONLY ARM THAT CHECKS THE WIRING, AND IT FOUND THE LOOP RUNNING A DIFFERENT MODEL.**
Each piece above is held to the reference one call at a time. What is left over is whether they are
joined the way `generateRfd4pCpu` joins them - the order of the noise draws, which features move
between steps, when the emit masks apply, whether the step advances `x` or `x1`, and **which recycle
depth each step gets**. The last of those was wrong: the exporter wrote `g/depths` from the
PRESET's `low` schedule and ran the reference at `generateRfd4pCpu`'s default `medium`, so the two
sides did four recycles against six on identical inputs. It read as a broken trajectory - the
initial noise and the step-0 features were exact to the last bit and the velocity was 8e-2 out -
and no per-stage gate could see it, because every stage was right.

🔴 **AND A SHORT RUN RESAMPLES THE RECYCLE BANDS, IT DOES NOT TAKE THEIR TAIL.** The bands expand to
SIXTY entries whatever `nsteps` is and are then read at `i * 60 / nsteps`, so two steps of `low` are
its first and thirty-first entries, `[4, 1]`. `sampler.cuh`'s own `recycleDepths` took the tail,
`[1, 1]` - the whole short schedule at the cheapest depth the long one ever reaches, which is a
different model and not a cheaper one. It is unused by `--generate` (the depths come from the
export) and would have been the standalone form's first bug.

🔴 **TWO STEPS EXERCISE AS MUCH AS SIXTY AND COST MINUTES INSTEAD OF HOURS** - `generateRfd4pCpu` is
`sum(depths)` JS trunk passes at about a minute each, so the preset's 60 steps is an afternoon.
`--gen-oracle` is opt-in on the exporter for that reason.

## The featuriser in C++, held to the JavaScript's BYTES

`cuda/rfd4p/featurise/rfd4p-featurise` takes a condition spec and writes the port's input
directory with no JavaScript anywhere on the path:

```
cuda/rfd4p/featurise/rfd4p-featurise <out> --weights=cuda/rfd4p/weights-f32 --spec=<job.json>
cuda/rfd4p/featurise/rfd4p-featurise <out> --weights=cuda/rfd4p/weights-f32 --length=100
```

**Or in one process**, which is the form to use: the fold binary links the same object and
featurises before it folds, so a condition spec goes in and a structure comes out with nothing
else on the path.

```
cuda/rfd4p/localfold-rfd4p --job=<job.json> --weights=cuda/rfd4p/weights-f32 --out=design.pdb
```

The gate holds that route to the two-step one byte for byte, because linking the featuriser in is
a second way to reach it and a second way to get it wrong.

`--generate` writes the whole-generation directory too, so **a fold from a condition spec needs no
JavaScript at all** - and its PDB is byte-identical to the one the exporter's own directory
produces.

`gate.py`'s `featurise` arm holds both directories to `export_input.mjs` **field by field, as
bytes** - fifteen cases, 0 differing - over seven lengths, the CATH-plus-secondary-structure
preset, a RANGED contig at four seeds, a THREE-CHAIN one at two, and a job with a protein, a DNA
and an RNA chain in it. That bar is `cuda/featurise`'s and it is the only one worth having: a featuriser that
merely agrees closely is a second, slightly different model.

🔴 **AND A SEGMENT'S LENGTH IS DRAWN EVEN WHEN ITS RANGE IS A POINT.** Skipping the draw when
`min == max` is not an optimisation: it moves every later draw up one, so the chain and molecule
PERMUTATIONS come out different and the chains are relabelled. A one-chain job cannot see it
(`permutation(1)`'s Fisher-Yates loop runs zero times), a two-chain one agreed by coincidence, and
three chains is where it showed - which is why the gate runs three.

🔴 **AND THERE ARE TWO RANDOM SOURCES, WHICH ARE NOT THE SAME ONE.** The SAMPLER's is xorshift128+
with Box-Muller; the PIPELINE's is mulberry32, and it decides how long a ranged contig segment
comes out - the gate's four seeds give 24, 30, 27 and 33 tokens, so a wrong draw order there is a
wrong protein rather than a wrong digit.

**And it is quick**, which matters because every gate here featurises:

| | native | the JavaScript exporter |
|---|---:|---:|
| floor (1 token, 1 step) | **0.01 s** | 3.24 s |
| 100 tokens, 60 steps | **0.13 s** | 3.41 s |
| 400 tokens, 60 steps | **0.81 s** | 4.13 s |

🔴 **READ THE FLOOR ROW FIRST**: most of the JavaScript's time is node starting and
`parseSafetensors` walking a 1.74 GB checkpoint, not featurising. The honest statement is that the
native side has essentially no floor - it reads seven embedding tables out of the exported index -
and is about 4x on the work itself at 400 tokens.

🔴 **THE NOISE IS DRAWN NATIVELY NOW, NOT REPLAYED**, and the gate compares it: `g/noise` is in the
generation directory, so Box-Muller is held to V8's in the same breath as everything else.

🔴 **IT NEEDED V8's `log` AND `exp`, AND glibc's ARE NOT THEM.** `get_time_embedding` is
`exp(-log(2000) / (half - 1) * i)` a channel, and glibc's own log and exp differ from V8's on
**2.6% and 7.2%** of arguments - so a featuriser on the system libm cannot be held to bytes however
right its structure. `cuda/featurise/jsmath.h` carries both now, 0 of 200,000 apart, which also
makes the port's own Box-Muller exact.

🔴 **AND THE FIRST VERSION WAS EXACT AT 53 AND 100 RESIDUES AND ONE ULP OUT AT 1, 2, 7 AND 257** -
the centering scale written `0.2f` where the reference multiplies by JavaScript's `0.2`, the
DOUBLE, so the product rounded twice. A sweep of lengths is what separates "exact" from "exact at
the length I tried", and it is why the gate runs seven.

🔴 **AND FOR A SPEC INPUT ALMOST EVERY BATCH FIELD IS ITS INITIALISATION VALUE** - which is the
reference's behaviour and not an omission. Every token realised from a contig is a `<M>` stub with
no atoms, so `buildAtom37Slots` returns -1 for all 37 slots and the whole per-atom loop of
`convertToProteinaBatch` is a no-op: `element` is -1 and `condition_index` is 1 because nothing
overwrites the fill. A port that helpfully computed them would disagree. What is actually computed
is the chain - `token_bond_path_length[i][j] = |i - j|` and `canonical_polymer_ordering` at the
neighbours - and the conditions.

The selection language is a SUBSET and says so: a union of `sel('*/<segment>/*/*')`, which is what
a contig with no residue numbers and no atoms can express. Anything else is refused BY NAME, never
quietly matched to nothing - a condition silently dropped is a condition the model is told is
absent, which is a different job folded under the right name. The gate has an arm for the
refusals: a preserved segment, an unknown chain type, a condition that needs atoms, a selection
that constrains a chain.

🔴 **AND ONE OF THOSE REFUSALS IS STRICTER THAN THE REFERENCE.** `C_SEQ` on a pure contig reaches
`finalizeAndCanonicalizeSequenceConditioning`, which rewrites every stub back to `<M>`, so the
page folds such a job with the condition having done **nothing** - measured. This refuses instead.
What it costs is a job the page accepts and this does not; what it buys is that a conditioning
this cannot apply is never silently dropped.

`featurise/tables.inc` is generated from `shared/rfd4p/` (the 147-token vocabulary and its atom37
layouts, the condition registry and its enums, the CATH maps), and
`test/rfd4p-native-tables.test.js` fails when it is stale.

## Every preset generates

All ten of RFD4-Proteina's shipped presets now run end to end: unconditional, fold-conditioned,
motif scaffolding (indexed and unindexed), protein-protein interface, DNA binding, a YAML binder
job, small molecule, partial diffusion and inverse folding.

🔴 **THE SMALL-MOLECULE PRESET WAS THE LAST, AND IT NEEDED A KERNEL RATHER THAN A FLAG.** The
export writes the sequence features once and the port overwrites the thirty-five state columns each
step, so a feature that moves for any other reason would silently be frozen at step zero - which is
why it was refused rather than folded with a stale value. `align_chirality` is such a feature: a
chiral centre's packed vectors are written in the CLEAN structure's frame, and at each step the
noisy CA positions define a new one, so the neighbours are re-read out of `x_t`, a proper rotation
is fitted, and the vectors are rotated by it. `src/chirality.cuh` does that on the device - one
thread a frame, a 4x4 Jacobi eigendecomposition in DOUBLE, because the reference runs it in JS
doubles and `sm_sam` has six frames over 165 tokens. What still refuses is a coordinate group.

## Partial diffusion and inverse folding

The two runs that do not start from noise, and the reason the encoder exists.

**Partial diffusion** takes a clean structure, corrupts it to a point part-way along the schedule
(`partial_trajectory_fraction_steps`) and denoises from there, which diversifies a scaffold rather
than inventing one. **Inverse folding** holds the backbone and redesigns the sequence and rotamers:
the target CA is resampled from its forward marginal at every step
(`resampled_forward_marginal`), so the latents are denoised against a backbone that never drifts.

`--generate` takes both when the export says so, and needs `--encoder=<dir>` beside it, because the
latent half of the clean sample is the encoder's output and the other half is the input's own CA.

🔴 **A PARTIAL RUN'S SCHEDULE IS NOT THE ORDINARY ONE WITH ITS HEAD CUT OFF.**
`build_partial_schedule` EXPANDS a schedule until its tail of `nsteps + 1` points starts at the
corruption time, so half of 60 steps is 30 points taken from the 60-step schedule - not the
30-step schedule, which has the right endpoints and the wrong spacing everywhere between. Inverse
folding starts at t = 0 for both modalities, where the expansion degenerates to the ordinary
schedule.

🔴 **AND ITS RECYCLE DEPTHS TAKE THE TAIL WHERE A SHORT RUN RESAMPLES.** `recycleDepths` is passed
`fullNsteps = baseNsteps` always, and when the bands already expand to that it slices rather than
resamples - a different branch from the one a `--steps=2` run takes.

🔴 **AND THE DRAW ORDER IS THE SPECIFICATION, AND A PARTIAL RUN'S IS NOT THE ORDINARY ONE.**
`corrupt_batch` draws the backbone's noise, then the latents', then the backbone's per-molecule
offset - where `sample_noise` draws the backbone, the offset, then the latents. One transposition
shifts every subsequent draw of the whole run: invisible in any single stage and wrong everywhere.
The encoder's own reparameterisation draw comes first of all, because the generator hands it the
rng before `corrupt_batch` touches it.

## The autoencoder ENCODER

`--encoder=<dir> --enc-oracle=<dir>`, on the 1PRW partial-diffusion preset (67 tokens): the input
features **exact**, `mean` and `zLatent` at **1.27e-06**.

It is the DECODER's transformer run the other way round - LayerNorm throughout, no pair bias, a
conditioning that is identically zero - so `aeLayer` and `aeConditioning` serve both and the weight
prefix is the whole of the difference. What is its own is the two ends: 1,587 feature columns built
per ATOM and gated on `atom37_mask`, and a reparameterisation head, `z = mean + eps * exp(logScale)`.

🔴 **ITS ORACLE IS THE DETERMINISTIC ARM.** `encodeLatentsCpu` draws `n * 32` normals out of the
SAMPLER's stream, so comparing the sampled `z` would compare two Box-Muller implementations rather
than the encoder. The gate takes `deterministic: true`, where `z` is the mean.

🔴 **AND IT COULD NOT BE CHECKED AT ALL UNTIL THE EXPORTER READ A STRUCTURE.** The encoder's input
IS a structure, and the exporter refused every preset that has one with "only the pure-spec presets
export offline" - which was never true: the seven examples the page names are
`web/examples/*.cif.zst`, shipped in this checkout, and the page fetches them over its own origin.
Reading them off disk is the whole of what that refusal cost, and it is what makes the motif, PPI,
small-molecule, DNA and partial-diffusion cases exportable.

🔴 **A PRESET WITHOUT `specMode` IS BUILT FROM ITS STRUCTURE'S ANNOTATIONS, NOT ITS CONTIG** -
web/generate.js has the note. Those contigs are written in the original entry's numbering and name
residues the cropped example does not contain, so building one from the spec fails outright.

## What is not done

- **A native featuriser for an input STRUCTURE.** The spec path is done (above); the seven
  `web/examples/*.cif.zst` presets still go through `export_input.mjs`. What that path needs, in
  the order it runs - the map is worth having before anybody starts:
  - `pipeline/zstd.js` + `vendor/fzstd.js` to decompress, and `pipeline/cif.js`'s tokeniser and
    `parseCifToAtomArray` (1,195 lines, and the largest single piece).
  - `buildBondsForAtomArray` with `STD_RESIDUE_BONDS` (28 residues of explicit bond lists),
    `STD_RESIDUE_CHARGES`, `STD_RESIDUE_AROMATIC`, `ELEMENT_TO_ATOMIC_NUMBER` (94) and
    `METAL_ELEMENTS` (61) - all generable into `tables.inc`.
  - `pipeline/chirality.js`: `ensureStereochemistryAnnotations` and `computeChiralityFeatures`,
    with `IDEAL_POLYHEDRA` (4 geometries, 19 unit vectors), `STEREOISOMER_TEMPLATES` (11 entries,
    105 point sets, built at load by brute-force permutation - precompute them) and
    `STD_POLYMER_CHIRAL_CENTERS` (~29 residues). 🔴 The port already has `properRotation` and the
    4x4 Jacobi on the device (`src/chirality.cuh`); what is missing is the TEMPLATES, not the fit.
  - the parts of `transforms.js` a contig skips: `buildAtom37Slots` returning real slots, so the
    per-atom loop of `convertToProteinaBatch` stops being a no-op, and `unindexFlaggedTokens`'
    body, `materializeInternalCoords`' rotations and `centerAndFeaturizeCentroid`' recentering.
  - the selection language in full (`chain/res_name/res_id/atom_name`), which only has anything to
    constrain once there are residues and atoms.

  🔴 **AND THE THING TO CHECK FIRST IS THE RNG ORDER**, not the chemistry: `materializeInternalCoords`
  draws `randomSO3` per internal-coordinate group and `unindexFlaggedTokens` takes its own path,
  both of which a contig never reaches - and this port has already been bitten twice by a draw
  that happens when it looks as though it should not.
- **An input with COORDINATE GROUPS to permute.** It is the last thing `--generate` refuses: the
  group relabelling is a `randperm` draw a step, so it has to be replayed like the noise. No
  shipped preset has one.
- **Writing an mmCIF**, and mapping an insertion back onto the input structure
  (`processSampleToAtoms`'s `inputAtoms` path, with the Hungarian assignment). The port writes a
  PDB from the fallback path.
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

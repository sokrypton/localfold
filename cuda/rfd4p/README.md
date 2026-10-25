# RFD4-Proteina on CUDA

The native port of the design model the page runs in `webgpu/rfd4p/`. **A job file in, a PDB out,
in one process with no JavaScript** - the featuriser, the trunk, the autoencoder's two halves and
the flow sampler are all here, and all held to `cpu/rfd4p/` and `shared/rfd4p/` stage by stage.

```
./cuda/rfd4p/localfold-rfd4p --job=<job.json> --weights=cuda/rfd4p/weights-f32 --out=design.pdb --fast
generated 53 residues, 265 atoms over 25 steps (52 trunk passes) -> design.pdb
  sampling 0.42 s
```

`--out=design.cif` writes mmCIF, which is what RFD4-Proteina itself writes; `--out=design.pdb`
writes PDB. 🔴 **A multi-chain design comes out as more than one chain now** - every atom used to
go into chain A, which for a binder or a complex is a structure no viewer and no scorer can read
correctly, and which no RMSD or bond check would ever have noticed.

**`--designs=N` is a campaign**, not a timing flag: N different proteins out of one process with
the weights already up - 0.43 s for the first and **0.27 s each after it**. Design 1 keeps the
featuriser's recorded noise so the trajectory gate still compares two sides that drew the same
numbers; every later one DRAWS its own, seeded `seed + design`, which the port can do now that
its Box-Muller is bit-identical to the page's. The gate asserts both halves - three structures,
all different (13-15 A of CA apart) and all still chains - because a bug that returned the same
structure N times would pass every geometry check there is.

🔴 **AND IT FOLDS THEM AT SIXTY STEPS, NOT TWENTY-FIVE.** At 25 one seed in three came out at
CA-CA 4.50 A, which is the sampler not being converged rather than the campaign being wrong - the
same lesson `npm run test:modified` records as "it must be folded converged or it measures
nothing".

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

`gate.py` is what re-runs all of it - **30 checks, all green on this A100**: the trunk's stages in
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
cuda/rfd4p/localfold-rfd4p --job=<job.json> --int8=model-rfd4p-int8 --out=design.pdb
```

🔴 **THE SECOND LINE IS A WHOLE DESIGN WITH NO FLOAT CHECKPOINT**, which is the difference between
a run anybody can make and one that needs NVIDIA's weight terms accepted: the published int8
bundle carries every table the featuriser wants as **float32**, by the bundle's own rule that only
the matrices are coded. The gate folds a converged job that way and scores the CHAIN - 52 CA-CA at
a mean of 3.72 A - rather than comparing coordinates with the float arm, because int8 perturbs the
velocity field and the sampler is chaotic, so it lands on a DIFFERENT member of the distribution
(2.1 A of CA at 25 steps). That is not a defect and not a tolerance anybody should pin.

The gate holds that route to the two-step one byte for byte, because linking the featuriser in is
a second way to reach it and a second way to get it wrong.

`--generate` writes the whole-generation directory too, so **a fold from a condition spec needs no
JavaScript at all** - and its PDB is byte-identical to the one the exporter's own directory
produces.

`gate.py`'s `featurise` arm holds both directories to `export_input.mjs` **field by field, as
bytes** - eighteen cases, 0 differing - over seven lengths, the CATH-plus-secondary-structure
preset, a RANGED contig at four seeds, a THREE-CHAIN one at two, a job with a protein, a DNA
and an RNA chain in it, two that name an input STRUCTURE (with a contig and without), and one
with three COORDINATE GROUPS, whose labels are permuted afresh at the top of every step. That bar is `cuda/featurise`'s and it is the only one worth having: a featuriser that
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
that constrains a chain, and a centroid given fewer than three numbers.

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
  - ~~`pipeline/cif.js`'s tokeniser and the `_atom_site` loop~~ **done**: `featurise/cif.h` reads
    a plain mmCIF and `gate.py`'s `cif` arm holds every column that comes STRAIGHT FROM THE FILE
    to `parseCifToAtomArray`'s, over all seven shipped structures - 0 differing, from a 1-atom
    file to a 925-atom complex. 🔴 Two rules that are the obvious guess backwards: `label_*` wins
    over `auth_*` (writing it the other way renumbers every residue a contig names), and the
    reader filters to MODEL ONE and the first altloc, without which a twenty-model NMR entry comes
    through twenty times. Decompression is native now (`cuda/featurise/zstd.h`), so a `.cif.zst`
    goes straight in; the four MAGIC BYTES decide, never the file extension, or a decompressed
    copy under its original name would be refused and a renamed plain file accepted.
  - ~~the chemistry `parseCifToAtomArray` derives from those columns~~ **done**: the charge
    fallbacks, `atomic_number`, the aromatic and leaving-atom flags, `chem_comp_type`,
    `chain_type` and `is_polymer` - all 22 columns, all seven structures, 0 differing. 🔴 The
    file's own `_chem_comp_atom` wins over the built-in tables, which are the fallback for a
    component the file says nothing about; and a NON-POLYMER chain takes `auth_seq_id` as its
    residue number, without which a het group's atoms all collapse into one residue because its
    `label_seq_id` is `.`.
  - ~~`computeIdAndEntityAnnotations`~~ **done**: the pn-unit and molecule components, the
    three-level Weisfeiler-Lehman entity hash and the instance ids - all 30 columns, all seven
    structures, 0 differing. 🔴 Coordination bonds are excluded at every level (a metal touching
    two chains does not make them one molecule), and the sort is NUMERIC where the column is -
    `molecule_id` and `res_id` sorted as text put 10 before 9 and renumber every entity past the
    ninth.
  - ~~the annotation columns, typed~~ **done**, including the legacy `mask_*` columns that
    REWRITE the `condition_*` ones they shadow. With this `parseCifToAtomArray` is ported whole:
    every column, every annotation and every bond, on all seven shipped structures, 0 differing.
  - ~~`buildBondsForAtomArray`~~ **done**: the `_struct_conn` covalent links, each residue's own
    bonds from `STD_RESIDUE_BONDS` overlaid by the file's `_chem_comp_bond`, the polymer backbone,
    and the metal retype - the bond list compared IN ORDER, since later stages index into it. 🔴
    A first bond for a pair wins, so a declared link is not overwritten by the dictionary's idea
    of the same pair; and `_struct_conn`'s partner search keeps the LAST match rather than the
    first, which is the reference's loop and not a slip.
  - ~~`ensureStereochemistryAnnotations`~~ **done** (`featurise/chirality.h`): the standard
    residues looked up, everything else DETECTED - the sp3 test, the phosphate and metal false
    positives, the Weisfeiler-Lehman distinctness of the neighbours, the planarity floor and the
    signed volume that picks CW from CCW; and the stereogenic double bonds. Exact on all seven.
    🔴 The WL colouring removes the CENTRE, because neighbours are only distinct in the graph
    that does not contain the atom joining them. And `STD_POLYMER_CHIRAL_CENTERS` generated into
    `tables.inc` was 21 entries where the source has 49 until the generator's character class
    admitted a quoted atom name - it had silently kept the twenty amino acids and dropped every
    nucleotide.
  - ~~`computeChiralityFeatures`~~ **done**: the Kabsch fit through the eigendecomposition of
    `BᵀB`, the template search, the planar double-bond alignment, the block packing for
    overlapping centres and the cycling fill. All four tensors exact on all seven.
    🔴 Its eigenvector sort comparator reads `S[j][j] - S[i][j]` - an OFF-diagonal where it plainly
    means `S[i][i]` - so it is not a sort by eigenvalue at all. Ported as written, with V8's own
    binary insertion sort, because the order decides the rotation and "fixing" it would disagree
    with every number the page produces.
  - ~~**`transforms.js` with real atoms**~~ **done, all fourteen stages** - the same pipeline
    the spec path already ran, but with every branch a contig's `<M>` stubs made inert now
    live. Held to the reference over twenty-nine cases: the seven shipped structures and twenty-two the
    gate synthesises, because the shipped seven could not reach most of it (see the three
    findings below the table) - nine of the twenty built this session purely to make the mmCIF READER's own filters and fallbacks reachable, four of those chosen by gcov. `convertToProteinaBatch` (1175-1606) is ported too, so a
    structure now reaches the batch the model reads - 46 keys held to the reference over all
    twenty-nine cases. Zstandard is native too (`cuda/featurise/zstd.h`,
    dlopen'd like `common.cuh`'s blob reader), so the page's own `.cif.zst` goes straight into
    the binary and the gate drives all seven that way as well as decompressed. `applyContig` takes PRESERVED segments now
    (`featurise/contig.h`), so a contig against a real structure assembles natively - five
    contigs over each of the twenty-nine cases, compared column by column with the bonds and
    `originalResId`. The selection language reads real atoms too
    (`featurise/selection.h`): `sel('chain/res_name/res_id/atom_name')` with lists and ranges,
    which is what every structure preset writes and what the contig-only subset could not say.
    `applyConditions` writes those selections onto the atoms
    (`featurise/conditions.h`), every dtype the presets use. **A job that names an input
    structure FOLDS natively now** - `localfold-rfd4p --job=<spec with an input>` reads the
    mmCIF, runs the fourteen stages and samples, 67 tokens of 1PRW at CA-CA 3.798 A - and the
    featuriser is byte-exact against `export_input.mjs` on both routes.
    Its output carries the INPUT's own residue numbering and chain letters, so a motif still
    superposes on the target it was built against, and is aligned onto the conditioned frame
    (`src/superpose.h`) - a **0.12 A** refinement, measured, not the rescue it looks like. An unindexed motif is grafted onto a
    generated token (`geometricInsertionMapping` and `matchInsertedAtoms`, on `src/assign.h`)
    and its conditioned coordinates transferred, which is how it reaches the output at all.
    The mapping is the trunk's OWN - `predictedInsertionMapping` off the insertion head's
    logits, validated pair by pair and falling back to the geometric one if any pair cannot be
    grafted. On `motif_unidx_1prw` the head's answer is used and agrees with the geometric one
    atom for atom. The coordinate-group permutation is
    implemented on both sides, which was the last item: **the structure path is complete.**

    🔴 **IT WAS A SILENT WRONG ANSWER ON ONE SIDE AND A REFUSAL ON THE OTHER.**
    `coord_group_seq` permutes its group LABELS at the top of every step - a group's id says
    only which atoms move together, so the model must not learn to read the number.
    `export_input.mjs` refused an input with more than one group; this side wrote the labels
    FROZEN and generated from them. Every shipped example has at most ONE group, where the
    permutation is the identity, so neither the refusal nor the bug was reachable without a
    case built for it (`gate-crdgroups.json`: three groups on `ppi_pdl1` through `C_CRDGRP`).

    🔴 **AND THE DRAW IS RAW UNIFORMS AT THE TOP OF THE STEP, BOTH OF WHICH MOVE THE WHOLE
    RUN.** `randperm` takes `next()` once per swap where every other draw here goes through
    Box-Muller's two-uniforms-per-pair, and it happens before that step's noise because
    `trunkSeqFeatures` is built before the forward. Modelling it as normals moves 49 of 75
    permutation entries; placing it after the noise moves 45. Freezing the labels moves 909 of
    `in/seq`'s 104052. All three watched failing, and the step-0 permutation is in `in/seq`
    itself - which is why the stream is drawn before the features rather than beside the noise.

    🔴 **AND THREE OF THE mmCIF READER'S RULES ARE FILTERS NO SHIPPED STRUCTURE CAN TRIP.**
    Scanned over the seven: every file is **model 1 alone**, every `label_alt_id` is **`.`**,
    and `label_seq_id` equals `auth_seq_id` in **every row of every file**. So the model
    filter, the altloc filter and the `label_*`-wins rule were ported from reading the
    reference and gated by nothing - and each fails silently rather than loudly. Three
    structures now build a file that trips them, each watched failing on itself and on nothing
    else: without the model filter `sm_sam+twomodels` is **132 atoms against 66**, without the
    altloc skip `motif_idx_1prw+altloc` is **330 against 306**, and taking `auth_seq_id` moves
    `res_id` in all 306 of `motif_idx_1prw+authnum`, eight selections, every condition - and
    **refuses six contigs outright** with `Preserved segment 'A1-6' matched no atoms`. That
    last is the shape worth knowing: the wrong column does not make a wrong structure, it
    makes a REFUSAL, which reads as a malformed job rather than as a reader bug.

    🔴 **AND A FOURTH: THE LEGACY `mask_*` COLUMNS, WHERE IGNORING A COLUMN YOU DO NOT KNOW IS
    GREEN ON EVERY FILE AND MEANS THE OPPOSITE.** `mask_index_1_residue`,
    `mask_sequence_1_residue` and `mask_coordinate_1_atom` are BOOLEANS that rewrite the
    `condition_*` annotations - where the sequence mask is set the condition becomes the
    residue's own name and where it is not it becomes `<M>` - so a file written the old way
    conditions the model on everything exactly where a reader that skipped the columns would
    condition it on nothing. No shipped structure carries one. `motif_idx_1prw+legacymask`
    does, and skipping the index rewrite alone is watched failing on the ANNOTATION KEY SET
    and on five contigs' `condition_index`.

    🔴 **AND THEN GCOV SAID WHAT READING COULD NOT: THE WHOLE FALLBACK HALF OF
    `deriveChemistry` WAS DEAD.** Built with `--coverage` and run over the corpus, the guard
    reads **913,128 hits on the `if` and ZERO on the `else`** - "the file's own tables win over
    the standard ones" turned out to mean they win EVERY time, because every structure carries
    `_chem_comp_atom`, `_chem_comp` and `_entity_poly`. Five rules live in that else and none
    had ever run: `STD_RESIDUE_CHARGES`, `STD_RESIDUE_AROMATIC`, the OXT/OP3 leaving-atom rule,
    `chemCompType` from the residue tables, and the chain type from the residue NAME - the last
    of which carries this port's 🔴 that `5-20` and `<M>` must come out POLYPEPTIDE "or a
    contig's own chain is typed NON-POLYMER", a justification read out of the reference and
    sitting on a line that had never executed. `withoutChemTables` strips the tables; all five
    are watched failing now - the chain-type rule at **2 of 27**, the leaving-atom rule 2, the
    aromatic table 2, `chemCompType` 1, the charge table 1.

    🔴 **AND TWO OF THE FIVE STAYED GREEN ON A BRANCH THE STRIP HAD JUST MADE REACHABLE.** The
    leaving-atom rule needs a file that HAS an OXT or an OP3, so stripping a fragment with
    neither reaches the line and cannot see it. And `STD_RESIDUE_CHARGES` sits under
    `if (a.charge[k] == 0)`, which every chargeable atom in this corpus fails because the file
    declares `+1` itself - measured, the guard is entered 913,128 times of 926,116 and the
    ~12,988 that skip it are exactly those atoms - so it needs no `_chem_comp_atom`, NO
    declared formal charge, and one of the three residues the table knows. **Reaching a line
    is not exercising the rule on it**, which is the sharpest form of this session's lesson:
    coverage tells you where to look and is not itself the test.

    The corpus is **29 structures** now, twenty-two of them synthesised, and gcov is how the last
    ten were chosen. Three more gaps it named and the corpus now covers: the contig
    suffixes `R` and `D` (`suffixChainType` maps `P`/`R`/`D` onto protein, RNA and DNA, and
    every contig ever written here asked for a protein - a segment typed as the wrong molecule
    is not a refusal, it is a chain the model reads as the wrong polymer; **24 of 27** with
    both arms returning protein), and the whole `centroid` condition dtype, the only one whose
    value is three numbers rather than a flag or a name (**20 of 27** with the triple
    reversed).

    🔴 **AND THE `centroid` CASE FOUND A REAL DIVERGENCE, NOT JUST A GAP.** Given two numbers
    the reference stores `parts.slice(0, 3)` - a TWO-element entry in a column every stage
    after it reads as a point, silently. This port cannot represent that (its annotation is a
    flat `n * 3`) and would not want to, because padding the third with NaN invents a
    coordinate and a two-number centroid is a typo. It is **refused by name**, a stated
    divergence beside the `sel(a),sel(b)` one; more than three is NOT refused, because both
    sides ignore the extra and the corpus holds them to each other on a four-component value.
    With the refusal in place `parts.resize(3, NAN)` provably cannot do anything, measured at
    0 of 27, and is gone.

    🔴 **AND THE GATE'S OWN `structure` FIXTURE WAS TESTING CONDITIONS IT COULD NOT APPLY.**
    Without `specMode` a job's `input` is a STRUCTURE rather than a spec, so `applyContig` and
    `applyConditions` are never called and the file's own annotations are kept instead - the
    fold is **byte-identical with the fixture's two conditions deleted**. That drop is correct
    and load-bearing rather than a bug: **eight of the ten shipped presets have exactly this
    shape**, their `.cif.zst` already carrying the conditions as annotations, which is why
    `input_as_annotated_cif` is true on that path and why refusing would break them. What was
    wrong was the fixture, which copied a preset's shape and so read as though it tested
    something. `gate-structure-contig.json` and `gate-crdgroups.json` set `specMode` and are
    where a structure's conditions are actually gated.

    🔴 **AND COUNTING FOUND A CAPABILITY THE PORT DID NOT HAVE: NO ATOM OF A DNA OR RNA CHAIN
    COULD BE NAMED IN A SELECTION AT ALL.** `splitTopLevel`'s quote machinery - single quotes,
    the doubled-`''` escape, double quotes, backslash escapes - ran **1758 times over 14,534
    characters and never saw a quote**, because every selection the corpus wrote named a
    PROTEIN atom. That is not a formality: **a sugar atom's name contains a prime**, which
    closes the single quote, so `dna_7m5w`'s eight primed names (`C1'`, `O3'`, `O5'` and the
    rest) can only be written double-quoted - which is how the mmCIF itself writes them. The
    reference accepts `sel("...")` and returns 9 atoms for one name and 18 for a two-name
    list; this port matched `sel('` alone and **refused the whole selection**. A loud refusal
    rather than a wrong answer, and still a real limitation on exactly the modality the port
    advertises.

    🔴 **AND BEHIND IT A SECOND BUG, WHERE BEING CAREFUL WAS THE MISTAKE.** With the double
    quote accepted, a two-name list was still wrong: `expandPathField` split the bracket
    contents with the quote-aware `splitTopLevel` where the reference uses a plain
    `v.slice(1, -1).split(',')`. Quote-awareness looks like the considered choice and is wrong
    here - a name ending in a prime opens a quote that never closes, so `[C1',C2']` comes back
    as the single item `C1',C2'` and matches nothing where the reference matches **18 of 390**
    atoms. The brackets are already stripped and the reference supports no nesting, so there
    is no top level left to respect. Both were watched failing before the fix.

    🔴 **AND E/Z DOUBLE-BOND STEREOCHEMISTRY WAS THE LARGEST UNTESTED THING IN THE
    FEATURISER: 80 LINES OF `chirality.h`, UNREACHED BY ALL TWENTY-SEVEN STRUCTURES.**
    `classifyDoubleBond`, the stereogenic-bond scan and the E/Z labelling. A double bond is
    stereogenic only where each end has one or two neighbours and a two-neighbour end's pair
    are topologically different - and SAM's only non-aromatic `DOUB` is a CARBONYL, whose
    oxygen end has no other neighbour at all, while every other double bond in the corpus is
    aromatic and skipped. So `stereogenic_bond_atoms`, which the model reads, was produced by
    code nothing had run: get it backwards and the model is handed an ISOMER, which no bond
    length, clash score or chain check can see. `sm_sam+stereobond` promotes `C2-C3` to
    `DOUB` and drops the four hydrogens that would give those carbons three neighbours each,
    which leaves one neighbour at each end and makes the rule fire - **2 stereogenic atoms
    where the unmodified file has 0**. Flipping the E/Z side and never setting the flag are
    each 1 of 29.

    🔴 **AND THE PLANARITY CUTOFF NEEDED A BOND THAT IS NOT PLANAR, WHICH A REAL MOLECULE'S IS
    NOT.** `classifyDoubleBond` returns 0 where the substituents' in-plane projections are
    less than 60 degrees apart, because the geometry then says nothing about which side
    anything is on - and dropping that guard was **0 of 28**, since `C2=C3` is cleanly
    determined. `sm_sam+twistedbond` moves one substituent to make the cosine zero, and now
    dropping it is 1 of 29. 🔴 **THE FIRST VERSION ROTATED A FIXED NINETY DEGREES AND DID NOT
    WORK**: the substituents start **49.3 degrees** apart, so a quarter turn lands on 139.3
    and |cosine| goes 0.652 -> **0.758**, still the wrong side of 0.5. The perpendicular is
    CONSTRUCTED now, which is exact whatever the molecule started at. Third time in this
    session that reaching a line was mistaken for exercising the rule on it.

    🔴 **AND THE REFUSAL ARM FOR THE CENTROID NEEDED THREE THINGS AND MISSED TWO IN TURN.** On
    a pure contig a centroid is ALREADY refused for needing a structure, so the first attempt
    reported NOT REFUSED with the rule it was testing never reached - **two refusals on one
    input test the first of them**. The second attempt gave it a structure and no `specMode`,
    so the condition was dropped before it could be refused: the very trap the paragraph above
    describes, walked into while writing that paragraph.

    🔴 **AND TWO OF `superpose.h`'s FIVE CONVENTIONS ARE NOT WHAT THE FILE SAID THEY WERE.**
    The header claimed three things were "the reference's and not the textbook's" and that a
    port writing the textbook version "disagrees on more than half of a random sweep". Mutated
    ONE AT A TIME against the gate's own 240 point sets, they are nothing like each other:

    | mutation of `src/superpose.h` | differing of 240 | worst element | where |
    |---|---:|---:|---|
    | the eigenvectors not forced right-handed | **114** | 1.94 | every n |
    | U's third column as a third singular vector | **99** | 1.92 | every n |
    | `det(H)`'s sign ignored | **90** | 1.92 | every n |
    | the textbook Jacobi (two full matrix products) | **29** | 0.97 | **n = 2 only** |
    | a negative eigenvalue not clamped to zero | **0** | - | nowhere |
    | ...and the `sqrt`-site guard removed as well | **0** | - | nowhere |
    | the three "reference, not textbook" ones together | 125 | | |

    The "more than half" was the three TOGETHER. The partial Jacobi update moves only
    rank-deficient inputs - two centred points are collinear, so the rotation about that line
    is not determined by the data and either answer is as good; it is still ported exactly
    because the two sides are held to the same bytes, but it decides no structure. And the
    eigenvalue clamp is **inert, and cannot be anything else**: a negative eigenvalue does come
    up, 41 times in that sweep, but it lands in slot 2 thirty-seven times - whose vector is
    never read, because U's third column is a cross product, so the two conventions are
    connected and the second kills the first - and in slot 1 four times, where the clamped 0
    and the unclamped NaN both fail `sigma > 1e-9` and take the same complement branch. Three
    of the five are load-bearing at every point count; the file now says which.

    🔴 **AND THE WRITER EMITTED THE UNINDEXED TOKENS, 40 OF THEM AS `UNK`.** They are
    in-context conditioning that the sampler never generates, and the reference writes only the
    INDEXED tokens; `motif_unidx_1prw` came out with 112 residues where it should have 72. The
    fold's own line did not report it - it counts tokens with a known residue type, and those
    have none - so the count said 72 while the file held 112. Found by reading the two against
    each other, not by any gate.

    🔴 **AND THE ARM FOR IT FAILED FIRST AT 25 STEPS, WHICH WAS THE STEP COUNT AND NOT THE
    PORT.** The shortest CA-CA step is 2.76 A there and 3.66 at 60. The int8 arm records this
    exact trap in these exact words - "reading that as int8's fault would be reading the step
    count as a defect" - and this arm was written at 25 anyway.

    🔴 **AND THE FOLD'S OWN COUNT DISAGREED WITH ITS OWN FILE, TWICE.** It counted every token
    where it writes only the emitted ones: "72 residues" over a file holding 112, then "602
    atoms" over 562. That disagreement is what made the residue bug look like it was not
    there, so the gate now holds the printed line to the file it describes - and putting the
    mask back reddens it at `said (72, 602) wrote (72, 562)`.

    🔴 **AND THE VALIDATION OF THE PREDICTED MAPPING IS NOT FALSIFIABLE ON THE ONE UNINDEXED
    CASE THE CORPUS HAS.** The head's mapping is fully matchable there, so applying it
    UNCHECKED gives the same answer and the gate stays green. What would exercise it is an
    input whose motif cannot be grafted onto the token the head picks; there is none.

    🔴 **AND THE OUTPUT USED TO RENUMBER THE MOTIF.** The writer spelled `i + 1` and
    `chainName(chain_iid)`, which is right for a chain this port NAMED and wrong for one a file
    did: `motif_idx_1prw` came out 1..67 where the reference writes the input's 0..66, and a
    two-chain input would come out A and B whatever its letters were. Both are carried across
    now (`g/outResId`, `g/outChain` - the chain as its CHARACTERS, since an input with chains H
    and L would otherwise become A and B), and the exporter writes them too so the
    featurise-then-fold path agrees. The gate compares per TOKEN against the reference
    pipeline's own `res_id` and `chain_id`, over two structures - **one chain cannot falsify
    the chain half**, since `motif_idx_1prw`'s only chain is A and `chainName(0)` is also A;
    with `ppi_pdl1` beside it, naming from the permuted index reddens 183 tokens.

    🔴 **AND A STRUCTURE IS THE JOB UNLESS THE SPEC SAYS OTHERWISE - THE TWO ROUTES DISAGREE BY
    67 TOKENS ON ONE FILE.** Without `specMode`, `export_input.mjs` hands the CIF straight to
    `buildBatch`: annotations KEPT, contig and conditions IGNORED, because a preset's contig is
    written in the ORIGINAL entry's numbering and names residues the cropped example does not
    contain. With `specMode: true` it is a spec - annotations discarded, arginines re-labelled,
    contig and conditions applied. The port implemented the second as the default and gave 116
    tokens where the exporter gives 183; the featuriser gate now drives both.

    🔴 **AND `cuda/build.sh` WITH NO ARGUMENTS DOES NOT BUILD rfd4p** (`ports=(af3 af2 ef2)`),
    while the featuriser-binary block sits OUTSIDE the port loop - so every `bash cuda/build.sh`
    in this campaign rebuilt `rfd4p-featurise` and left `localfold-rfd4p` alone. A structure job
    featurised correctly through the standalone binary and refused through the fold one, with
    nothing in the diff to explain it; the fold binary was eight hours old. **And the fold
    binary's own stamp hashed `cuda/featurise` while rfd4p LINKS
    `cuda/rfd4p/featurise/featurise.o`**, so even naming the port would have relinked the
    previous object. Both fixed - the SECOND stamp in this port to name fewer files than it
    depends on, after the one that hashed five while the structure path lived in three more.

    In the order they run, with what each needs:

    | stage | `transforms.js` | what a structure makes live |
    |---|---|---|
    | ~~`addSrcComponentAnnotation`~~ **done** | 340-358 | nothing new |
    | ~~`ensureChemicalFlags`~~ **done** | 272-335 | the 17 boolean columns actually vary |
    | ~~`computeWithinChainSourceResIdx`~~ **done** | 252-267 | real residues to count |
    | ~~`realizeExpandableSegments`~~ **done** | 363-475 | real residues between the sentinels |
    | ~~`filterUnwantedAtoms`~~ **done** | 480-496 | hydrogens, leaving `OXT` and nucleic `OP3` are dropped |
    | ~~`addGlobalAtomIdAnnotation`~~ **done** | 499-504 | `atom_id` after the filter, not before |
    | ~~`annotateCovalentModificationsAndAtomize`~~ **done** | 510-562 | real peptide bonds, so `atomize` means something |
    | ~~`addGlobalTokenIdAnnotation`~~ **done** | 567-584 | one token a residue, or one an ATOM where it is atomised |
    | ~~`finalizeAndCanonicalizeSequenceConditioning`~~ **done** | 629-708 | a real sequence to canonicalise |
    | ~~`featurizeExpandedSegments`~~ **done** | 713-747 | the inference deletion block fires |
    | ~~`materializeInternalCoords`~~ **done** | 752-817 | 🔴 `prng.randomSO3()` per coordinate group - A DRAW |
    | ~~`unindexFlaggedTokens`~~ **done** | 822-1027 | 🔴 the whole 200-line body, which a contig skips entirely |
    | ~~`centerAndFeaturizeCentroid`~~ **done** | 1032-1108 | the recentering on conditioned coordinates |
    | ~~`addSupportsLocalLatentAnnotation`~~ **done** | 1113-1126 | a residue carries a latent only out of unindexed context |
    | ~~`buildAtom37Slots`~~ **done** | 1131-1145 | 🔴 real slots, so `convertToProteinaBatch`'s per-atom loop (1405-1442, 1475-1486) stops being a no-op |
    | ~~`computeTokenBondFeatures`~~ **done** | `bonds.js` 198-566 | real bonds, so `bond_mask` is not all zero |
    | ~~`computeTokenPathLengths`~~ **done** | `bonds.js` 130-188 | a BFS over a graph with invented nodes in it |

    🔴 **`realizeExpandableSegments` GOES RIGHT TO LEFT**, so an earlier sentinel's index is
    still its index when its turn comes; and the suffix's residue numbers shift only within the
    sentinel's own `chain_iid`, not its `chain_id`, because two chains can share a letter across
    transformations. Exact on all seven - 306 atoms to 330 on 1PRW, bonds included.

    🔴 **AND THE SEVEN SHIPPED STRUCTURES REACH ONE OF `filterUnwantedAtoms`' THREE RULES.**
    Counted: 32 hydrogens in `sm_sam` and **not one `OXT` or `OP3` in the whole corpus**, so
    deleting either of the other two left `check-cif.mjs` green on all seven - and no protein
    structure has a SECOND chain, so a cross-chain peptide bond could not be built at all. The
    gate synthesises four cases on top of the seven (`+OXT`, `+OP3`, `+crosslink` - a declared
    covale across a gap - and `+interchain` - a duplicated chain bonded C to N at adjacent
    numbering), both sides reading the same synthesised text. Each of the five conventions was
    then watched failing on exactly one of them: the `OP3` rule, the `OXT` rule, `resDiff === 1`,
    `sameChain`, and the atomised branch of the token numbering. **Four of the five were
    unreachable before those cases existed.**

    🔴 **AND `finalizeAndCanonicalizeSequenceConditioning`'s BACKBONE EXEMPTION IS INVISIBLE
    WHILE THE SEQUENCE IS GIVEN.** The stage reveals a residue's name only where the atom is not
    ALREADY conditioned, and every shipped motif conditions its sequence - so deleting the
    `is_*_backbone` test changed nothing on eleven cases. A twelfth (`+masked`, the same motif
    with every `condition_sequence_1_residue` set to `<M>`) separates a residue whose conditioned
    coordinates are backbone-only from one with a conditioned side chain, and the mutation moves
    20 atoms there. The deletion of an all-`NaN` probability column is the other live convention
    (`5o45_cropped` keeps **0** of the 28 where the others keep all 28), which is why the gate
    compares the SET of column names and not only their values. One clause genuinely cannot be
    falsified - the `|| !allowed` in `conditioned` - and `atoms.h` says why rather than leaving
    it looking untested.

    🔴 **AND THE DRAW THIS TABLE HAS BEEN WARNING ABOUT SINCE IT WAS WRITTEN IS NOW COVERED, BY
    TWO CASES THAT HAD TO BE BUILT.** Not one shipped structure has a POSITIVE
    `condition_CoordinateGroup_1_atom`, so `materializeInternalCoords` never reached its
    `randomSO3` at all. `+coordgroups` gives a structure two groups (one cannot tell a per-group
    draw from one draw shared by all), and `+icgroups` gives a NON-EXPANDING structure two groups
    with the first's internal coordinates already set, so that group is skipped and the second
    must take the FIRST rotation. Four orderings were then watched failing: drawing before the
    `allFinite` skip, drawing once for every group, filling the nine normals column-major, and
    taking `u2` before `u1`. **Every one of them produces a perfectly valid rotation and a stream
    one draw out** - which is the whole reason the warning was there.

    🔴 **AND `unindexFlaggedTokens` HAS THREE CONVENTIONS THAT ONLY EXIST FOR THE BRANCH THE
    SHIPPED MOTIFS DO NOT TAKE.** Every shipped motif is unindexed at a chain EDGE, which
    collapses; the interesting rules are about an INTERIOR run. `+interior-unindexed` (an
    interior run with ordinary residues both sides, everything in index group 0) turns two of
    them red - a removed run must leave a HOLE in `res_id` rather than renumber, and the
    index-group shift is zero whenever the groups were derived rather than supplied, so only a
    file that supplies them can see it. `+beside-expandable` (an interior run touching the
    expandable residue at position 21 of `motif_idx_1prw`) is the only shape that reaches the
    re-bond at all, since that needs a run which is interior AND collapses. Removing the
    re-bond turns it red; forcing its `collapsedKept` test TRUE does not, because the extra
    pairs are already bonded - the guard saves work rather than changing the answer, and
    `atoms.h` says so rather than implying it was checked.

    🔴 **AND TWO OF `centerAndFeaturizeCentroid`'s CONVENTIONS WERE INVISIBLE BECAUSE THE GATE
    WAS NOT LOOKING, NOT BECAUSE NO CASE REACHED THEM.** The centring shifts
    `condition_coordinate_1_atom` as well as the structure, and nothing compared that column
    after the stage - deleting the shift left all sixteen green. It is compared now and the
    same mutation turns **twelve** of them red. The other was the gate's own recorded mistake
    repeated: the centroid is a `Float32Array` in the reference, the comparison used
    `Math.fround` on BOTH sides, and so a port that kept it in double scored perfectly.
    Compared exactly, with the native side printed at full precision, removing the single
    float32 rounding turns **fifteen** of sixteen red. **A narrowed comparison is how a
    narrowing hides - twice now, in two different columns.** Two clauses here genuinely cannot
    be falsified by any input (the `isfinite` guard on the shift, since `NaN - x` is `NaN`, and
    the `|| !allowed` in the sequence conditioning) and `atoms.h` says so of each.

    🔴 **AND `convertToProteinaBatch` OPENS WITH TWO DRAWS WHOSE ORDER TWO CHAINS CANNOT SEE.**
    It permutes the chains and then the molecules from the stream the realisation and the
    internal coordinates have already advanced; at one element `permutation(1)` draws nothing,
    and at TWO the two draws agree often enough that swapping them left all sixteen cases
    green. `+threechains` (`ppi_pdl1` with its one real chain duplicated twice) turns it red,
    and the failure reads as `chain_iid` and `molecule_iid` with each other's values - which is
    what a swapped draw order looks like and what no geometry check could show. Four more
    conventions were watched failing there: a masked chain taking a FRESH entity after every
    one the input had (16 cases), `element` carrying -1 rather than the atomic number wherever
    the residue is masked (3), `is_resolved_mask` spreading over a token with ALL where the
    context masks spread with ANY (8), and the chain permutation itself (9).

    🔴 **AND `tetrahedral_vertices` WAS STUBBED AT -1 AND THE GATE AGREED WITH IT.** The
    reference fills it only `if (tokenIds)`, the earlier chirality arm passed `{}` on both
    sides, so neither computed anything and the column matched perfectly. Building the batch
    passes real token ids and `sm_sam` - the one structure with stereocentres - went red at
    once. **A comparison where both sides take the same early return is not a comparison**;
    this is the third shape of that mistake recorded here, after the `Math.fround` pair.

    🔴 **AND `applyContig`'s SENTINEL RESIDUE ID IS `-(2**31)`, NOT -1** - guessed as -1 and the
    gate named it on fifteen structures, one entry each. A real mmCIF carries negative residue
    numbers, so -1 would COLLIDE with one, and silently: a selection written against the input's
    own numbering would then match the stub. Four more conventions were watched failing - an
    expandable token before a preserved one inserts BEFORE it (15 cases), a leading
    pure-expandable run is queued and then REORDERED back in front (15), the sentinel's
    inherited annotations are reset to their defaults (2 int, 12 string), and the sentinel takes
    over its neighbours' bond (15). One branch is unreachable from here and says so in
    `contig.h`: the sentinel fits a HOLE in the numbering where there is one, and
    `buildRunPreserved` renumbers every preserved segment contiguously from 1 before anything is
    inserted, so there never is one.

    🔴 **AND ONE OF THOSE COLUMNS READ AS WRONG BECAUSE OF THE COMPARISON, NOT THE PORT.**
    `condition_index_1_residue` is BOOLEANS in the reference and ints in the dump, so `1 ===
    true` is false and all 53 atoms differed. Cast one side only and a correct column reads as
    a total failure; it is the mirror of the `Math.fround` trap, which hides a difference where
    this invents one.

    🔴 **AND THE GATE WAS COMPARING TWO HAND-WRITTEN SEQUENCES, NOT THE PIPELINE.** Both the
    dump and the gate's JavaScript side listed the fourteen stages by hand, so both were
    missing `runValidationPipeline`'s SECOND `addGlobalTokenIdAnnotation` and agreed perfectly.
    The native side has the driver now (`featurise.cpp`'s `runValidationPipeline`) and the gate
    drives the reference's own, so a stage added to either is in the comparison without anyone
    remembering to add it. The missing call turned out to be INERT - that stage changes neither
    `atomize` nor the residue boundaries, and removing it leaves all eighteen green (of the eighteen then in the corpus) - so this
    found no wrong answer, only an instrument that could not have found one.

    🔴 **AND `sel(a),sel(b)` IS NOT `sel(a)|sel(b)`.** The pipe is a union; the COMMA makes the
    reference's parser return after the first term and silently drop the second. The first
    port unioned both and the gate caught it as 8 extra atoms on three structures. This port
    **refuses a comma between terms by name** rather than copying either behaviour: matching
    the reference means silently discarding a condition somebody wrote, and unioning means
    applying one they did not ask for. A deliberate divergence, stated here and in
    `selection.h`, that cannot produce a wrong fold either way. Four conventions were watched
    failing - a descending range read ascending (9 cases), atom names expanding a hyphen as a
    range (17), `*` matched as a literal (17), and a later term overwriting instead of
    unioning (9).

    🔴 **AND TWO OF THOSE FOUR WERE SAID TO BE COVERED AND WERE NOT, BECAUSE THE SELECTIONS
    CAME FROM THE PRESETS AND THE PRESETS ONLY WRITE THE EASY FORMS.** Counted: **117 distinct
    atom names over the shipped structures and not one has a hyphen**, so the 17 cases above
    for "atom names expanding a hyphen" cannot have been that rule - a mutation of a branch no
    input reaches reddens nothing, so whatever that arm changed, it changed something else as
    well (dropping `isResId` entirely also pushes every atom name through `atoi`, which turns
    `CA` into `0` and reddens everything). The count was real and the attribution was reading
    the mutation rather than the corpus. Likewise the residue-number normalisation was written
    down as "provably inert, since every preset writes clean numbers" - true of the presets,
    and an excuse rather than a measurement.

    Nine selection forms nothing could reach are in the corpus now - a bare residue number, a
    bare hyphenated one, a zero-padded one alone and in a spaced list, an atom-name list, one
    whose item carries a hyphen, a chain list, and the transformation field both ways. Four
    rules, each watched failing at **9 of the 18 structures then in the corpus** with a mutation that touches only it: a
    hyphenated ATOM NAME split in two, a BARE `1-3`
    expanded as a range where it is the single value 1, a zero-padded `[0054]` left
    unnormalised, and the `transformation_id` default moved from `"1"` to `"2"` (checked both
    ways - `.../1` is every atom, `.../2` is none). The already-matched skip remains genuinely
    inert and still says so: that loop only ever SETS a bit.

    🔴 **AND `applyConditions`' ARMS ARE APPLIED IN JSON ORDER, WHICH A `std::map` DESTROYS.**
    A `C_CRD` with a `true` arm over a chain and a `false` arm over two of its residues is one
    condition with two arms, and the LAST applied wins; sorted, `"false"` came before `"true"`
    and the false arm was overwritten instead of overwriting - **48 coordinates and 16 groups
    wrong on every structure**. The JSON reader was already order-preserving; the container
    threw it away. Two conventions that read backwards are pinned beside it: `C_SEQ: {true}`
    writes the residue's OWN NAME rather than the string `"true"` (13 cases), and a CATH code
    gains a trailing `.x`, so `3.40.50` is stored `3.40.50.x` - the fourth level unnamed means
    ANY, not missing (13 cases).

    🔴 **AND THE CATH RULE IS NOT "PAD TO FOUR LEVELS", WHICH IS THE ONLY THING ONE CODE CAN
    SAY.** The reference is `if (!cathStr.endsWith('.x')) cathStr += '.x'`, so a COMPLETE
    four-level code gains a FIFTH field - `3.40.50.20` is stored `3.40.50.20.x` - and a
    one-level `3` becomes `3.x`, not `3.x.x.x`. All four shapes are in the corpus now, and the
    pad-to-four reading was watched failing at **14 of those same 18** - naming `cath-4` and `cath-1` and
    agreeing on `3.40.50`, which is the whole point: on the one code the presets write, the
    wrong rule is indistinguishable from the right one.

    🔴 **AND THE GATE COULD NOT SEE THE RESIDUE-NUMBER SWAP, TWICE OVER.** `applyConditions`
    evaluates selections against the INPUT's numbering - `applyContig` renumbers every
    preserved segment from 1 - and puts the assembled numbering back at the end. Running
    conditions with `originalResId` equal to `res_id` makes BOTH identity, so deleting either
    left all eighteen green (of the eighteen then in the corpus). A case in the pipeline's real order (contig, then conditions)
    reddens the swap on 2 structures, and dumping `res_id` reddens the restore on 8. Same root
    cause as the four above: **an input on which the thing under test cannot act**.

    🔴 **AND THE LOAD A JOB DOES IS NOT THE ONE EVERY OTHER ARM USES.**
    `input_as_annotated_cif` defaults to FALSE, so a spec that names a structure DISCARDS the
    file's own conditions - it is an input, not a spec - and re-labels each arginine's NH1/NH2
    by distance from CD, because a deposited arginine is routinely labelled the other way and
    the model reads the labels. Keeping the annotations reddens 16 cases and inverting the
    arginine rule reddens 8 - but DELETING the arginine fix reddens nothing, because on all
    seven shipped structures they are already the right way round. `+swapped-arg` exchanges
    every arginine's two nitrogen names so the fix must fire, and then deleting it reddens
    exactly that case.

    🔴 **CHECK THE RNG ORDER BEFORE THE CHEMISTRY.** `materializeInternalCoords` draws
    `randomSO3` per internal-coordinate group and `unindexFlaggedTokens` takes a path a contig
    never reaches; this port has been bitten twice by a draw that happens where it looks as
    though it should not, and both times the symptom was somewhere else entirely.

    🔴 **AND THE ATOM ARRAY IS ONE OBJECT NOW** (`featurise/atoms.h`), because a reader naturally
    produces four pieces and every stage of `transforms.js` changes several of them TOGETHER -
    `realizeExpandableSegments` inserts atoms, which moves every column, every annotation and
    every bond index at once. Four parallel objects is four places to forget one.

    Its `slice` and `concat` are the only primitives the rest of the pipeline has for inserting
    and dropping atoms, and they walk the columns through ONE list (`forEachColumn`) - the
    reference gets that free by walking its object's own keys, and a C++ struct cannot. A column
    added and not listed keeps its OLD values through an insertion, which is a wrong answer in a
    field nothing else looks at, so the gate round-trips every structure:
    `concat(slice(0, k), slice(k, n))` must reproduce the dump byte for byte, at four split
    points including ones that cut a residue and a bond in half.
  - ~~the selection language in full (`chain/res_name/res_id/atom_name`)~~ **done**
    (`featurise/selection.h`); it only had anything to constrain once there were residues and
    atoms, which is why it waited for the mmCIF reader.

  🔴 **AND THE THING TO CHECK FIRST IS THE RNG ORDER**, not the chemistry: `materializeInternalCoords`
  draws `randomSO3` per internal-coordinate group and `unindexFlaggedTokens` takes its own path,
  both of which a contig never reaches - and this port has already been bitten twice by a draw
  that happens when it looks as though it should not.
- ~~**An input with COORDINATE GROUPS to permute**~~ **done** (`894b684f`): the relabelling is a
  `randperm` draw at the top of every step and is replayed like the noise. No shipped preset has
  one, so `gate-crdgroups.json` builds a three-group case.
- ~~**Mapping an insertion back onto the input structure**~~ **done**: the trunk's own insertion
  head with a geometric CA-distance fallback (`src/assign.h`, `resolveInsertion`), and the output
  aligned onto the conditioned frame.
- ~~**The selection language in full**~~ **done** (`featurise/selection.h`), as a stated SUBSET -
  a union of `sel(...)` joined by `|`, in EITHER quote - a nucleic atom's name carries a
  prime, so it can only be written double-quoted - with a comma between terms refused by
  name.
- ~~**Anything but an unconditional monomer through `--generate`**~~ **done**: both per-step
  features are handled rather than refused - a chirality frame recomputed on the device, a
  coordinate group's labels permuted from a replayed stream.
- **A padded input**, so the masking paths are written and unexercised.
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

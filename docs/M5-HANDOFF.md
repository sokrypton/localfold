# Handoff to the M5: the Metal port on Apple's newest GPU

Written on the M2 (10-core GPU, 16 GB, macOS 13.2) on 2026-10-09, for an agent working on an Apple M5. Everything
here is on local branch **`metal`**. Read `metal/README.md` first (the port's layout, its gates, what won and what
lost, its traps), then this. `CLAUDE.md` is the repository's operational guide; its Metal rows point at the same files.

## Why an M5 pass

Every speed and memory choice in `metal/` was measured on one M2. The M5 differs on the axes that matter:

- **Matrix hardware in every GPU core ("neural accelerators")**, reachable only through Metal 4's tensor operations
  (macOS 26). The port spends most of a fold in matrix multiplies and attention that already run near the M2's
  ceiling (the GEMM at 2.93 TFLOP/s on a 4096 square, about 82% of peak; the transformer's GEMMs at 1.7-2.1). Using
  the new units is the one change that could move a fold by a large factor.
- **A newer GPU generation** (dynamic register and threadgroup-memory allocation since the M3) and **more memory
  bandwidth**. The tile sizes, the K step and the register budgets below were chosen where the M2 runs out of
  registers or occupancy; on the M5 some of them may invert.
- **The WebGPU side's Apple prior** (`metal-3` in `webgpu/runtime/device-profile.js`) was measured on the same M2.

## Rules - the same as on the M2

1. **Never push.** Pushing `main` deploys the website. Commit on local `metal`; the user merges.
2. **Do not accept AlphaFold 3's terms for the user.** Never run `cuda/featurise/fetch-weights af3` (or anything that
   calls it with the terms accepted: `tools/check-native-worker.py` without `--no-af3`, a page fold of AF3 in a test
   profile). The AF3 gate cases read the page bundle `model-af3-int5/`, which the user copies over themselves.
3. **Do not modify `cuda/`** except for portability fixes. It is the CUDA port; `metal/` is written natively.
4. **Measure arms interleaved in one process** (`metal/bench/localfold-bench`), never two runs minutes apart: Apple
   laptops throttle and share the GPU with the window server. On the M2 the same GEMM read 0.22 and 0.42 ms in two runs.
5. **Every change goes through the gates** (below) before it is committed, with the numbers in the commit message.
6. **Use bash for loops over words.** This machine's shell is zsh, which does not split `$var` into words:
   `for x in $list` and `set -- $shape` silently pass one argument. Run such loops under `bash -c '...'`.

## Setup

- **macOS 26 (Tahoe) and Xcode 26's SDK** for Metal 4's tensor operations. Steps 1, 2 and 4 below need only the
  Xcode command-line tools and run on any macOS 13+.
- The checkout on branch `metal`, with the weights the M2 has. Copy them, following symlinks (`model-af3-int5` on the
  M2 is a symlink into a sibling checkout):

  | what | where | for |
  |---|---|---|
  | `model-af3-int5/` | the checkout | AF3's gate cases (copied by the user: AF3 terms) |
  | `af3am-*/` (boltz2, chai1, esm2, intellifold2, openbind0, opendde, protenix2, rosettafold3) | the checkout | the other AF3-lineage models; `af3am-esm2` is chai-1's ESM2 3B (5 GB) |
  | `model-af2-monomer-int5/` | the checkout | AF2's WebGPU profiler (`--bundle=`) |
  | `~/.cache/localfold/` (model, model-multimer, deltas, esmfold2, esmc, ccd) | home | AF2, ESMFold2, the native backend's fetches |

  A bundle that is missing is fetched by the binaries on first use, except AlphaFold 3's (rule 2).

```
bash cuda/build.sh featurise                   # the native featurisers (no GPU needed)
bash metal/build.sh af3 af2 ef2 selftest       # -> metal/<port>/localfold-<port>, metal/selftest/localfold-selftest
bash metal/build.sh bench                      # -> metal/bench/localfold-bench (GEMM, attention and LayerNorm arms)
metal/selftest/localfold-selftest              # the GEMM, attention and LayerNorm against host references
```

The kernels are Metal source compiled at run time. The first fold of each model on a new machine compiles its
kernels and is slow (chai-1's took 146 s on the M2, then 11.7); the GEMM instances used are recorded in
`~/.cache/localfold/metal/*.specs` and compiled up front, in parallel, from then on.

## Step 1 - correctness on the M5 (no new code)

```
python3 metal/af3/gate.py            # 16 cases: eight models, ligands and a phosphoserine, DNA, glycans, templates
python3 metal/af2/gate.py            # 5
python3 metal/ef2/gate.py            # 8
python3 tools/check-native-worker.py --offline --no-af3     # npm run test:native: the local server, page included
python3 tools/check-native-bridge.py                         # npm run test:bridge (no GPU)
npm test                                                     # the CPU suite; 1337/1338 on the M2 (see below)
```

`metal/<port>/gate-baseline.json` is **the M2's**, held to 0.05 A RMSD and 0.5 pLDDT, and the A100's is printed
beside each case. A different GPU rounds differently, so small moves are expected. **Do not re-record (`--write`)
until each moved case is explained**: compare it with the A100 column and the M2 baseline, and fold it at another seed
if it is an unconfident fold (pLDDT under 50 is chaotic in the rounding). The M2's AF3 values, for reference:
af3 6MRR 0.771 A / pLDDT 86.16, protenix2 1.561, boltz2 0.578, intellifold2 1.539, rosettafold3 1.699, openbind0
1.759, opendde 1.457, chai1 0.976, 5CAJ self-template 0.205, 1BRS 0.457.

`npm test`'s one failure on the M2 is `tools/_fold_film.py`, a file git does not track that names the old `src/`
layout; it is not this branch's.

## Step 2 - re-measure the existing choices (no new code)

The M2's numbers, each from an interleaved bench or a warm fold. Run the same on the M5 and keep both columns.

| measurement | command | M2 |
|---|---|---|
| GEMM peak, 4096 square | `metal/bench/localfold-bench 4096 4096 4096 0,1016,1032,164064 hhh` | 2.93 TFLOP/s |
| transformer q/k/v/g, 68 tokens padded to 80 | `metal/bench/localfold-bench 3072 80 768 180032,180064 hhh` | 0.19 ms (80x64 tile) |
| trunk K-128 projection, 261 tokens | `metal/bench/localfold-bench 512 68121 128 0,1032,148064,164032 hhh` | 3.44 ms, 2.60 TFLOP/s |
| grid attention, 261 tokens | `metal/bench/localfold-bench attn 261 4 32 261 0 0` | 6.84 ms, 1.33 TFLOP/s |
| LayerNorm, 68121 x 128 | `metal/bench/localfold-bench ln 68121 128` | 83.5 GB/s (fitted) against 42.6 (widest) |
| AF3 6MRR, 200 steps | `metal/af3/localfold-af3 $D/af3-6mrr.input --fold --fast --bundle=model-af3-int5 --family=af3 --out=/dev/null`, where `D=$(python3 -c "import tempfile,os;print(os.path.join(tempfile.gettempdir(),'metal-af3-gate'))")` (the gate featurises its inputs there) | 6.1 s: trunk 1.17 (4 passes), diffusion 4.90 |
| a diffusion step | the same, `--recycles=0 --steps=100` | 24 ms at 68 tokens, 80 at 261 |
| AF3 trunk pass, 261 tokens | the same on `$D/af3-5caj-template.input`, `--recycles=1 --steps=50` | 3.9 s a pass |
| memory, 1044 tokens, two passes | `AF3_MEM=1 /usr/bin/time -l` on a 1044-token input (`cuda/featurise/af3-featurise <dir> --no-weights --family=af3 --sequence=...`), `--steps=5 --recycles=1` | 4.7 GB footprint |

Bench arms: `0` the default tile; `1000+B` a K step of B; `100000 + R*1000 + C` an R x C tile. The tile rule is
`gemmRun` in `metal/core/core.mm` (n <= 128: rows rounded to 16, columns 64 where that fills the GPU, else 32; larger
n: 64 x 64, or 48 where 64 wastes over a tenth). `LOCALFOLD_PROFILE=1` profiles a fold per kernel (each dispatch its
own command buffer, so an upper bound); `AF3_STAGES=1` times the denoiser's stages; `AF3_MEM=1` prints peaks and the
scratch by name.

What to look for: a tile or K step that now wins (the rule then wants a device condition, not a new constant for
everyone); whether the attention kernel is still register-bound (its direct-from-device K/V loads lost 2x on the M2
for that reason - see the README's "lost" list - and dynamic register allocation may change that); whether LayerNorm
still wants its row array sized to the row.

## Step 3 - the matrix units (the main work)

The goal: the GEMMs (`metal/core/gemm.metal`, chosen and launched by `gemmRun` in `metal/core/core.mm`) and then the
flash attention (`lf_attention<D>` in `metal/core/common.metal`) on the M5's matrix units, through Metal 4's tensor
operations (Metal Performance Primitives' `matmul2d`, called from a kernel in Metal Shading Language 4.0).
**Verify every API name against the macOS 26 SDK's headers** - this note was written on a machine that has neither.

Constraints the port already imposes:

- **The macOS 13 floor stays.** The wheel (`python/build_wheel_macos.sh`) is tagged `macosx_13_0_arm64`, and every
  M1/M2 user runs the existing kernels. The new path is chosen **at run time** - the device supports the family and
  the OS has the language version (`@available(macOS 26, *)` around anything new on the host) - and the old kernels
  stay as they are. The kernels compile at run time with `MTLCompileOptions.languageVersion` set to 3.0 today
  (`compile` in `core.mm`); the tensor kernels want their own library compiled at the newer version.
- **The GEMM's epilogues come with it.** `gemm.metal`'s instances fuse a bias, ReLU, GELU, the gated residual
  (`aux * sigmoid`), SwiGLU in blocks of 8 and the triangle's gate (EP bit 1, channel-major output). A tensor-op GEMM
  either applies the same epilogues on its output tile or is offered only for the plain ones first. The epilogue
  arithmetic must stay the same (the gates hold structures to 0.05 A).
- **Half accumulation is used on purpose** in the all-half GEMMs (EP bit 8); keep the choice per instance, and check
  the matrix units' accumulator types.
- **Ragged tiles are slow** on the M2 (bounds-checked staging and scalar stores), which is why the diffusion
  transformer pads its rows to 16. Re-check what the tensor path wants.

How to hold it to the existing path: the selftest against host references; `localfold-bench`'s arms (it prints each
arm's relRMS against the first - the old kernel - beside its time); then the three gates and `test:native`; then the
Step 2 folds, interleaved against the old path through an environment switch you add (as `LOCALFOLD_LN_WIDE=1` and
`AF3_SCALAR_ATOM_ATTENTION=1` are for theirs).

Where the time is, on the M2, to choose what to do first:
- **AF3 at 68 tokens (200 steps):** diffusion is 4.9 of 6.1 s; within a step the transformer's q/k/v/g and SwiGLU
  GEMMs (3072 x 80 x 768) are about 43%, its two conditioning GEMMs (73728 and 36864 x 80 x 384, once a step) 12%.
- **AF3 at 261 tokens:** the trunk is 3.9 s a pass - attention 18%, the K-128 projections (grid q/k/v/g, triangle
  gate, transition, 512 x 68121 x 128) about 35%, the triangle contractions (batched 264 x 264 x 264) 8%.
- **AF2:** the same pair-track shapes (`metal/af2`, profile with `LOCALFOLD_PROFILE=1`).

## Step 4 - the WebGPU side's Apple prior

`webgpu/runtime/device-profile.js`'s `metal-3` entry sets `attentionProjectMatrix: false`, `matrixLinear: false`
(both: 8x8 matrix units lose to the vector kernels on the M2), `opmMatrixContract: true` and `fusedPairBias: true`.
Each has its measurement beside it. Re-run the arms on the M5 in Chrome:

```
node tools/gpu-chrome.mjs tools/gpu/probe-subgroup-matrix.js       # which matrix shapes Chrome offers here
node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --bundle=/model-af2-monomer-int5 --length=150 --sequences=256 --tune=fusedPairBias=false
node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --bundle=/model-af2-monomer-int5 --length=150 --sequences=256 --tune=fusedPairBias=true
```

Two rounds each, interleaved. A knob that differs on the M5 becomes a condition on the device inside the entry (or a
second Apple entry), never a change to the default for every device. Note that a stock Chrome on the M2 offered **no**
subgroup matrices (`--enable-unsafe-webgpu` is needed); check what a stock Chrome on the M5 offers
(`LOCALFOLD_STOCK_FLAGS=1`). CLAUDE.md's "IF YOU ARE THE M2" section has the WebGPU traps.

## Traps already paid for on the M2

- **An array of simdgroup matrices must be fully unrolled** (`_Pragma("clang loop unroll(full)")` on every loop over
  one), or it spills: the flash attention ran at 0.15 TFLOP/s until it was.
- **Apple GPUs have no integer divider.** `a / d` by a runtime `d` is a software routine; use `lf_udiv`.
- **32 KB of threadgroup memory, and no `double`** in Metal. Check whether the M5 raises the first.
- **A register array sized for the widest case costs occupancy** (LayerNorm's `float4 r[12]` halved its bandwidth at
  128 channels). Template on the size.
- **A released buffer stays resident until the work issued before it is done**, and the host runs far ahead of the
  GPU; the allocator waits past 128 MB of them (`LOCALFOLD_BURIED_MB`).
- **A blob's float32 data is not aligned**: af3-any-model's records have headers of any length.
- **`MTLCreateSystemDefaultDevice()` returns nil to a command-line process sometimes**; `MTLCopyAllDevices()` does not.
  And a pipeline can silently refuse the threads it is dispatched with: no error, the output untouched.
- **A loop bound naming a constant is made opaque for a WebGPU device's first fold** (`runtimeLoopBounds`), and the
  opaque and unrolled kernels can round differently; write literal bounds where the two must agree.
- **Fusing launches did not pay on Metal**: one serial encoder makes a launch nearly free, so the diffusion's fused
  adaLN + gated residual was level (the README's "lost" list).

## Report back

Append a section **"The M5's reply"** at the end of this file: Step 1's gate results (any moved case and its
explanation), Step 2's table with an M5 column, what Step 3 built and what it is worth (each arm against the old path,
interleaved, with the gates' numbers), and Step 4's arms. Update `metal/README.md`'s tables where the M5 changes them,
naming the machine (an M2 number and an M5 number side by side, never one replacing the other). Commit on `metal`
with the numbers in each commit message. Do not push.

## The M5's reply

Measured on an Apple M5 (8-core GPU, macOS 26.6, Xcode 27.0, Metal Toolchain 27A266a), 2026-10-09. Two commits on
`metal`: the GEMM on the matrix units, then the flash attention on them (and a fix to the first). Not pushed.

### Step 1 - correctness: every case passes, nothing re-recorded

All 29 gate cases pass against the M2's baselines, on the old kernels and again on the final build with the matrix
units on: ESMFold2 8/8, AF2 5/5, AF3 lineage 16/16 (chai-1's two included; `model-af3-int5/` fetched from the
registry's pinned Hugging Face remote with the user's permission). Nothing moved past the bars, so `--write` was not
run. The final build, against the M2: AF3 6MRR **0.768 A / 86.15** (M2 0.771 / 86.16), 5CAJ self-template **0.205**
(0.205), 1BRS **0.454** (0.457), protenix2 1.561 (1.561), boltz2 0.577 (0.578), intellifold2 1.539 (1.539),
rosettafold3 1.702 (1.699), openbind0 1.757 (1.759), opendde 1.453 (1.457), chai1 0.979 / 1.743 (0.976 / 1.744). The
largest moves are the unconfident folds the bars expect to move: AF2 5CAJ from its sequence (pLDDT 31) 19.804 ->
20.109 A, AF2 1BRS multimer without a template (pLDDT 39) 15.855 -> 15.798.

The matrix units accumulate wider than lf_gemm's half accumulator even for a half destination - against the host
reference the selftest's SwiGLU goes 2.5e-3 -> 3.9e-4 and the triangle gate 1.2e-3 -> 3.0e-4 - which is why nothing
moved the other way. `npm test`, `test:native` and `test:bridge` were not run in this pass.

### Step 2 - the existing choices, re-measured (old kernels on the M5)

| measurement | M2 | M5, lf_gemm / lf_attention | M5, matrix units |
|---|---|---|---|
| GEMM peak, 4096 square | 2.93 TFLOP/s | 3.05 | **12.0** (64 x 128 tile) |
| transformer q/k/v/g, 3072 x 80 x 768 | 0.19 ms | 0.162 | **0.064** |
| trunk K-128 projection, 512 x 68121 x 128 | 3.44 ms, 2.60 TFLOP/s | 3.08 ms, 2.90 | **0.93 ms, 9.6** (half out); 1.68 (f32 out) |
| grid attention, 261 tokens | 6.84 ms, 1.33 TFLOP/s | 4.68 ms, 1.94 | **4.18 ms, 2.18** |
| LayerNorm, 68121 x 128 | 83.5 GB/s fitted, 42.6 widest | 132.3 / 92.4 | (not a matrix kernel) |
| AF3 6MRR, 200 steps | 6.1 s: trunk 1.17, diffusion 4.90 | 3.79 s: 0.69, 3.07 | **1.93-2.01 s: 0.41-0.49, 1.50** |
| a diffusion step, 68 tokens | 24 ms | 15.3 ms | **7.5 ms** |
| AF3 trunk pass, 255 tokens (5CAJ self-template) | 3.9 s | 2.64 s | **1.65 s** |
| memory, 1044 tokens | 4.7 GB | not measured | not measured |

On the old kernels nothing in the tile rule inverted: the M2's arms (K step 16/32, 64 x 64, 48 x 64, 80 x 64) are
within 2% of each other here, as there. LayerNorm still wants its row array sized to the row (1.43x). The attention is
still register-bound in the sense the README means; it was not re-tried with direct-from-device K/V loads. The M5's
threadgroup memory is still 32 KB.

### Step 3 - the matrix units

`metal/core/gemm_tensor.metal`, compiled at MSL 4.0 in a library of its own and chosen at run time on an Apple10 GPU
under macOS 26 - every other GPU and OS keeps the old kernels, and the wheel's macOS 13 floor stands.

- **The GEMM** (`lf_gemm_tensor`): matmul2d over the GEMM's own pointers (`tensor_inline`, leading dimensions as
  strides, a transpose as the descriptor's flag), every lf_gemm epilogue - plain products stored by the cooperative
  tensor, the rest on a tile staged in threadgroup memory with lf_gemm's arithmetic. Half x half instances only; an
  f32-operand GEMM keeps lf_gemm. Interleaved, old -> new: 4096 square 3.0 -> 12.0 TFLOP/s; 3072 x 80 x 768 0.162 ->
  0.064 ms; 73728 x 80 x 384 1.59 -> 0.77; 512 x 68121 x 128 3.11 -> 0.93; 128 x 68121 x 512 3.07 -> 0.99; 264^3
  0.046 -> 0.030. Ragged tiles need no padding here: matmul2d bounds-checks against the tensor's extents.
- **The attention** (`lf_attention_tensor`): S and P.V on matmul2d, a simdgroup's 16 queries on their own. A smaller
  win, taken only where it measured: heads 32 wide from 256 keys (n 261 1.12x, 512 1.34x, 1044 1.29x); level at D 48
  and 64, worse at D 16 and at 68 tokens.
- **Whole folds**, alternated: AF3 6MRR 3.79 -> 1.97 s; boltz2 6MRR 5.02 -> 2.35; protenix2 6MRR 5.11 -> 2.42;
  boltz2 5CAJ (255 tokens, 2 passes, 50 steps) 9.49 -> 5.17 with the GEMM, 5.04 with the attention as well.
- Controls: `LOCALFOLD_GEMM_TENSOR=0`, `LOCALFOLD_ATTN_TENSOR=0` (=1 forces the attention at any length).
  `metal/bench`: arm 1000000 (+ R*1000 + C a tile) is the tensor GEMM, `attn` arm bit 4 the tensor attention.

Traps paid for, for the next agent:
- **matmul2d's default mode OVERWRITES its destination** (`mode::multiply`), whatever the header's comment
  ("C = A*B + C") says. An accumulation across tiles wants `matmul2d_descriptor::mode::multiply_accumulate`.
- **`reduce_rows` wants a single simdgroup's scope**, so a row-wise softmax splits the queries by simdgroup.
- **The header's own examples do not compile**: there is no `get_mask` (it is `is_valid_element`), `store` wants
  the cooperative tensor's element type exactly, and `get_destination_cooperative_tensor` needs `template` when the
  op's type is dependent.
- **Compiling at run time works**: `newLibraryWithSource` with `MTLLanguageVersion4_0` finds
  `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` with no other setup, ~100 ms an instance.

Not done: f32-operand GEMMs on the units (matmul2d takes float x half and float x float; not measured), the
remaining shapes of the attention (D 8, 16, 24), and the 1044-token memory row.

### Step 4 - the WebGPU side

Chrome 154 on the M5 offers **the same as on the M2**: subgroup matrices of 8 x 8 x 8 only (f16 and f32), and only
with `--enable-unsafe-webgpu` - a stock Chrome offers none (`LOCALFOLD_STOCK_FLAGS=1`: `subgroupMatrixAvailable:
false`). Those are Metal's simdgroup matrices, not the M5's matrix units, which nothing in WebGPU reaches. So the
`metal-3` prior's matrix knobs are no closer to paying here than on the M2.

The arms (`profile-af2-block.js --bundle=/model-af2-monomer-int5 --length=150 --sequences=256`, two rounds each,
alternated; block ms, the stack of 48 in brackets):

| arm | Chrome | round 1 | round 2 |
|---|---|---|---|
| `fusedPairBias=false` | stock | 79.40 (3811) | 79.36 (3809) |
| `fusedPairBias=true` (the prior) | stock | 78.44 (3765) | 78.51 (3769) |
| prior (`matrixLinear` and `attentionProjectMatrix` false) | flagged | 75.84 (3640) | 76.49 (3672) |
| `matrixLinear=true` | flagged | 74.19 (3561) | 75.07 (3603) |
| `attentionProjectMatrix=true` | flagged | 78.10 (3749) | 78.01 (3745) |

- `fusedPairBias` still wins (1.1%): the prior holds.
- `attentionProjectMatrix=true` still loses (2-3%): the prior holds.
- `matrixLinear=true` reads **1.9% faster** on the M5 where it lost on the M2 - but only with
  `--enable-unsafe-webgpu`, which no visitor has, two rounds, and inside the drift this file warns about. **Not
  changed**: a 2% flagged-only gain does not earn a second Apple entry. Worth one more interleaved round if the
  flag ever ships.

### After the report - two more (144d2be2)

A profile of boltz2's 255-token 5CAJ trunk pass found two leftovers:

- **Half-output GEMMs that asked for float accumulation** (`linW`'s half outputs, such as the grid's qkvg) were taking
  a float cooperative tensor, which forced the staged path and a 64 x 64 tile. The matrix units sum a run wider than
  half and round once, so a half output now always takes a half destination. 512 x 65025 x 128 + bias: 1.41 -> 0.86
  ms. In the fold, grid qkvg went from 234 to 115 ms a pass, with CA coordinates identical to the PDB's 0.001 A.
- **Tensor attention's cutoff, 256 -> 168 keys.** 5CAJ's 255 tokens sat just under the old one. n 255 D 32: 3.87 ->
  3.06 ms; n 192: 1.68 -> 1.39; n 168: 1.35 -> 1.20. It still loses at 160 (D 64 1.74 -> 1.87: a 64-query block
  half empty). Wider tiles lose: n 255 D 32 at 64 x 32 takes 3.06 ms, 64 x 64 takes 3.56, 128 x 32 takes 4.43
  (`LOCALFOLD_ATTN_TILE`).

boltz2 5CAJ fold: 5.11 -> 4.83 s from the GEMM alone (interleaved x3). All 29 gates pass. AF3 6MRR in the gate:
1.97 -> 1.77 s.

What the trunk pass spends now (255 tokens, 1.78 s GPU): attention 24%, the triangle gate 12%, layernorm 8%, the
pair transition's SwiGLU 7%, gated add 6%, qkvg 6%. The attention runs at 2.8 TFLOP/s against the GEMMs' 10. A
rewrite of it (softmax in registers rather than threadgroup rows, the bias tile shared) is the next big item.

### The attention rewrite (d45e9e0f)

`lf_attention_tensor2` is the rewrite. Each simdgroup runs on its own and reads Q, K and V straight from the qkvg
buffer through device tensors, so the key loop has no staging and no threadgroup barrier. The bias and mask tile goes
through P's own half tile. That leaves about 5 KB of threadgroup memory, against v1's 13. **Occupancy was the cost,
not the barriers**: dropping only the barriers gained 5%; an extra 8 KB float bias tile made it 40% slower.

| H 4 x n rows | lf_attention | v1 | v2 |
|---|---|---|---|
| n 255 D 32 | 3.88 ms | 3.06 | **1.87** |
| n 255 D 64 | 6.84 | 5.81 | **3.19** |
| n 255 D 16 | 2.51 | 2.73 | **1.68** |
| n 128 D 32 | 0.53 | 0.47 | **0.30** |
| n 64 D 32 | 0.092 | 0.101 | **0.077** |
| n 32 D 32 | **0.032** | 0.044 | 0.038 |

v2 runs from 48 keys at D >= 32 and from 112 at D 16; v1 stays only for a q bias. On boltz2 folds (1 recycle, 50
steps, interleaved), 255 tokens: 4.64 -> 4.30 s, trunk attention 424 -> 260 ms a pass. 510 tokens: 20.5 -> 17.4 s.
All 29 gates pass.

Still open: the diffusion transformer's attention (one row, 16 heads, about 0.1 ms a call) fills only 64
threadgroups and barely moved (124 -> 117 ms over 50 steps). Splitting its keys across simdgroups is the remaining
lever there.

(That last item turned out to be the q bias: the diffusion attention carries one, which kept it on v1. v3 takes it
now, below.)

### Overnight, 2026-10-10: nine more commits

Every one gated (all 29 pass), measured interleaved, pushed. boltz2 5CAJ (255 tokens, 1 recycle, 50 steps):
**4.30 -> 3.78 s**. The README's M5 table has the current headline numbers.

| commit | what | measured |
|---|---|---|
| 78bb6f03 | the triangle gate in registers: the weight permuted to quarters (`lf_tri_quarters`) so every thread holds a channel's pa, ga, pb, gb | gate 214 -> 163 ms a trunk pass |
| 0f109fe3 | attention v3: the online softmax in cooperative tensors (`map_iterator` for the row statistics, P as a cooperative left input), no threadgroup memory | n 255 D 32 1.87 -> 1.63 ms |
| b763aa15 | v3 takes a q bias (Q + bias staged once per simdgroup) | diffusion attention 116 -> 70 ms over 50 steps |
| 893cc30b | the general GEMM epilogues (alpha, beta, bias, GELU, gated residual) in registers, 4 columns at a time | output projections 51 -> 43 ms; 64 x 128 tiles for float outputs |
| f5cc698e, ca6c6dc4 | the triangle zeroes only its planes' padding; the bias layout as a tiled transpose; padZero's index in 32 bits | 31 + 29 ms a pass at 255 tokens; 140 + 104 ms at 510 |
| 0e29920f | v3's output epilogue 4 columns at a time | n 255 D 32 1.66 -> 1.38 ms |
| 274b76ec | a float X allowed half staging goes to the matrix units | ESMFold2's sampler 347 -> 265 ms |
| 878ac60d | fast exp and divide in the hot epilogues | triangle gate 166 -> 131, SwiGLU 132 -> 119 ms |

Also e2ffe387: ESMFold2 crashed on `--out=/dev/null` (wrote through a null `FILE*`); fixed as AF3's main does it.

The layouts these lean on, probed on the M5 (`get_multidimensional_index`, kept in the comments where used):
- A 64 x 128 destination over 4 simdgroups gives a thread 4 adjacent columns, repeated every 32 columns, on rows r, r +
  8, r + 32, r + 40. Element e + 8 is column + 32, and column + 8 lives in lane ^ 8.
- A 64 x 64 destination, and attention's 16 x D output, keep the same 4-adjacent-column groups.
- The half logits are compatible as P.V's left input; logits -> rows and output -> the logits' rows map by iterator.

What **lost** or did nothing, so nobody repeats it blind:
- **SwiGLU in registers** (a shuffle pairs a with b in lane ^ 8): 131 -> 154 ms. Then staging only the gated half
  tile, 8 KB: 146. The SwiGLU GEMM is compute-bound at 9-10 TFLOP/s; its epilogue was never the cost.
- **The triangle gate staged transposed** (8 KB, each channel's 64 rows written as one run): 166 -> 185 ms. Its
  scattered 2-byte stores were cheaper than the staging.
- **Fusing the diffusion transformer's gated residuals and boltz2's up-gate multiply into their GEMMs' epilogues**
  (2400 + 1200 dispatches fewer over 50 steps): level. At 255 rows those GEMMs fill only 24-48 threadgroups, and the
  epilogue's reads sit on their critical path.
- **matmul2d's relaxed precision**: 12.2 TFLOP/s either way on 4096 square.
- **A half exp2 for P in v3**: 1.652 against 1.657 ms. **A runtime branch between two softmax variants in v3**: 2x
  slower. The kernel lives on its registers.
- **v3's bias read as half4 from device** rather than `cooperative_tensor::load`: 2% slower at D 32, 3% faster at D 64.
- **Bigger v3 tiles**: 16 x 64 is best at D >= 32 (32 x 32 is 4% better at D 32 only), 16 x 32 at D 16.
- **Smaller tiles for the diffusion GEMMs** (768-3072 x 256): 64 x 128 is already best or within 3%.
- **A fast `lf_sigmoid` for AF3's small diffusion kernels** (adaLN, the gated residuals): unchanged. They are
  latency-bound, about 20 us a dispatch.
- **A concurrent encoder**: a serial dispatch boundary costs about 2.7 us on the M5 (measured: 1000 tiny dispatches
  take 2.7 ms serial, 0.3 concurrent). That is about 1 ms of a 68-token step's 8.8, and only part of it is free to
  overlap. It would need dependency tracking; not done.
- **The pair LayerNorm emitted by the previous GEMM's epilogue** (its tile covers all 128 channels): about 3% of a
  trunk pass at best, for a second 279 MB buffer at 1044 tokens and every pair update re-plumbed. Not done.

Where the time is now (boltz2 5CAJ, profiled, a trunk pass of 1.27 s GPU): attention 199 ms, the triangle gate 131,
LayerNorm 156 (bandwidth), SwiGLU 119 and qkvg 116 (both near peak), the gated add 114 (bandwidth). At 68 tokens a
diffusion step is weight-bandwidth- and latency-bound: about 370 MB of transformer weights read a step, 381
dispatches. Batching samples would amortize the weights.

**Caution for profiling small folds**: `LOCALFOLD_PROFILE=1` gives every dispatch its own command buffer. At 68
tokens that turns a 1.76 s diffusion into 5.46 s; use it for proportions only.

### Overnight, continued: loading, the MSA kernels, small folds

The gate's folds carry one MSA row and warm weights, which hid a family of bugs: **64-bit integer division**. The
GPU emulates it slowly. Several kernels split a 64-bit thread index into coordinates with `/` and `%` on `ulong`
once per element. Each fix below keeps the 64-bit path for ranges past 2^32 (none at these sizes) and is
byte-identical against HEAD's binary.

| commit | kernel | measured |
|---|---|---|
| 54f7cd24 | `lf_decode` (the bundle decoder) | weights load: AF3 int5 178 -> 96 ms, boltz2 ~360 -> ~200, protenix2 455 -> 175 |
| 4269f04f | `lf_gather` (the weight walk's parts); `lf_copy2d` 16 bytes a thread | ESMFold2 loads in 0.13 s, not 0.36; its sampler's copy2d 25.8 -> 2.4 ms |
| 3197fce0 | `af2_opm_permute`, `af3_msa_v_heads`, `af3_opm_permute`, the OPMs' left operands | AF2 with a 1500-row MSA: 4.79 -> 4.35 s a pass; boltz2's trunk pass with it 1.68 -> 1.43 s |
| 2eadafef | the AF3 MSA attention's head permutes, 16 bytes a thread | 46 -> 26 ms a pass |
| ca6c6dc4 | `lf_pad_zero` | 104 -> 14 ms a pass at 510 tokens |

The MSA numbers come from a synthetic alignment: 1500 random variants of 5CAJ, 30% substituted. Any real a3m will do;
the gate has none offline.

Also:
- **b0dd3f5b, the diffusion transformer's conditioning for K noise levels at once.** It depends on the level
  alone, and the sampler knows its levels, so at a short n the two conditioning GEMMs read their 170 MB of
  weights once for K steps. AF3 6MRR: 1.71 -> 1.64 s, byte-identical.
- **274b76ec, a float X allowed half staging goes to the matrix units.** ESMFold2's sampler: 347 -> 265 ms.
- **fb6c1a71**: the profile now labels each attention by version and shape.

Where a big fold spends its time (boltz2, 1020 tokens, a trunk pass of 30.8 s; the fold 71 s, peak 5.96 GB):
attention 37% (11.4 s, 6.5 TFLOP/s), LayerNorm 8%, the triangle gate 7%, SwiGLU 6%, the gated add 6%, qkvg 6%, the
two triangle contractions 11% at 10.5 TFLOP/s.

More that **lost**:
- **Attention v3 software-pipelined** (the next tile's Q.K issued before this tile's softmax, two logit tensors):
  11-20% slower. **Its row statistics by hand** (two shuffles over lanes ^ 1 and ^ 8, rather than `reduce_rows` and
  `map_iterator`): level. **P written straight into a left-input tensor** (it shares the logits' layout): level.
  The kernel is bound by occupancy and its serial tile chain; the MPP helpers cost nothing extra.
- **LayerNorm, 4 rows a simdgroup** (every load issued before the first reduction): 2-12% on a cold bench, level in
  folds, where the pair arrives from cache.
- **AF3's per-step diffusion kernels in 32-bit division** (`af3_gather_rows`, the encoder and decoder broadcasts):
  level. They are latency-bound, about 20-65 us a dispatch.

Fixed on the way: ESMFold2 crashed on `--out=/dev/null` (e2ffe387).

Later still:
- **a567d5f3**: `singleConditioning` is batched with the look-ahead too: K levels' noise projections, then the
  transitions over K n rows. boltz2 6MRR's diffusion: 1.736 -> 1.556 s. AF3's: 1.383 -> 1.286 s. Byte-identical.
- **d6bc7230**: chai-1's sampler hands the denoiser its plan as well. chai1 6MRR's diffusion: 1.019 -> 0.933 s.
- `npm run test:native` (offline, without AF3 and the page arm) and `test:bridge` pass on the M5 with all of this.

And more that **lost**:
- **The single-conditioning embedding (snProj) in the look-ahead too**: level.
- **adaLN as float4s in registers with a fast sigmoid**: diffusion 3.693 -> 3.669 s at 261 tokens (under 1%), and the
  fast sigmoid moves the numbers. Not kept.
- **v3 with its keys split over the four simdgroups** (flash-decoding's split, for the diffusion's one-row
  attention): correct, and 2x on a 261-key bench row, but level in the fold.
- **A v4 on 4-simdgroup tiles** (64 queries x 64 keys, both products cooperative): MPP refuses it. "Input cooperative
  tensors require a single SIMD group", so P could only go through threadgroup memory, which is v1's design.
- **Concurrent transformer branches**: only `diffusionNoResidual` dialects have independent attention and transition
  branches, and AF3, boltz2 and protenix2 do not.

Measuring at night: the M5's times drift with heat by 5-20% across back-to-back folds (one 261-token trunk ran
4.86 then 5.92 s, unchanged). Every number here alternates the arms; trust no single pair.

### The night, end to end

Start of the night (cddc61ac) against 523aaed3, both binaries built from their trees and alternated: a warm-up round,
then two measured rounds, averaged. Fold times as each port prints them; ESMFold2 wall-clock, because the old binary
crashes on `--out=/dev/null`.

| case | before | after | |
|---|---|---|---|
| boltz2 5CAJ, 255 tokens (1 recycle, 50 steps) | 4.51 s | 3.87 s | -14% |
| boltz2 6MRR, its defaults (4 passes, 200 steps) | 2.30 s | 2.02 s | -12% |
| chai1 6MRR | 2.04 s | 1.84 s | -10% |
| AF3 6MRR (the gate's) | 1.81 s | 1.62 s | -10% |
| AF3 5CAJ with its template, 261 tokens | 10.16 s | 8.86 s | -13% |
| AF2 5CAJ with a 1500-row MSA, 1 pass | 5.86 s | 4.62 s | -21% |
| AF2 1BRS multimer (the gate's, 4 passes) | 3.29 s | 2.68 s | -19% |
| ESMFold2 5CAJ, wall | 4.09 s | 3.41 s | -17% |

Weights load before -> after: AF3 bundle 178 -> 96 ms, boltz2 ~350 -> ~200, chai1 ~480 -> ~385, AF2 0.15 -> 0.06 s,
ESMFold2 0.26 -> 0.11 s.

Near morning:
- **27c4d94d, the triangle's output projection and gated add in one kernel** (`lf_gemm_tensor_dual`: both products
  of a tile in two cooperative tensors, combined in registers). boltz2 5CAJ: 154 -> 126 ms a trunk pass. AF2
  5CAJ: 4.19 -> 4.11 s. Only at C <= 128: at ESMFold2's 256 the two accumulators cost more than the round trip
  saves. Not byte-identical, because two matmuls in one kernel round differently (CA 0.06 A); the gates pass.
- **2f8783be**: the profile reports AF3's confidence head too.
- **Lost: the pair transition as one fused MLP** (`lf_ffn_tensor`: SwiGLU's hidden activations staged through a 16 or
  8 KB threadgroup tile per hidden chunk, never written to memory). 209 ms against SwiGLU 118 + transition2 77 a
  pass. The three accumulators and two barriers a chunk cost more than the 132 MB round trip saved. The raw
  `transition1` would also have had to stay unretired, since `prepareWeights` retires it once interleaved.
- **Lost (again): the diffusion's gated residuals and boltz2's up-gate multiply in their GEMMs' epilogues**, re-tried
  in the small-fold regime. Byte-identical, 72 dispatches a step fewer, and level at 16 and 68 tokens. A step at 16
  tokens costs 6.0 ms against 7.6 at 68, but that floor is not dispatches. It is the transformer's ~311 MB of weights
  read every step by GEMMs too small to saturate the bus (768 x 80 x 768: ~30 us whatever K, 38 GB/s; 3072 x 80 x
  768: 96 GB/s). What would move it: int8-resident weights with a dequantising GEMM (README's Next 3, ESM2's case), or
  more samples a batch (NS > 1 amortises every weight read).
- **Lost: the triangle's product in half** (scaled by 1/n so long chains stay in range, the centre norm's eps by
  1/n^2: the same norm). First it was NaN: the product accumulated into a half destination before alpha, and 255
  terms' sums overflowed. With float accumulation it works, but it saves 13 ms of a 1.21 s pass (the centre norm 65
  -> 58, the contractions 76 -> 69) for 0.07 A of drift. Not kept.
- **Lost: attention at head width 8** on the matrix units. MPP wants K dynamic or a multiple of 16, and AF2's extra
  MSA is 1.6% of an MSA fold.
- **169c0d82, the triangle gate's quartered weight built once per model.** Each port builds it as its own derived
  tensor, instead of core rearranging it every call (that guarded against a stale cache, on a misdiagnosis).
  IntelliFold-2 (C 512): `lf_tri_quarters` was 18 ms of a 547 ms pass at 68 tokens; its trunk is now 2.21 -> 2.15 s.
  Byte-identical.
- **Lost: a register-resident centre norm for C 384-512** (`lf_center_norm_wide` reads the product three times and
  writes rows 32 apart). With 64 floats a lane and a 128-channel transposing tile: 344 -> 329 ms a pass at 255
  tokens, because the registers cost occupancy. Not kept. IntelliFold-2 at 255 tokens is 7.5 s a trunk pass: its
  C 512 GEMMs run at 11-11.5 TFLOP/s, so it is simply big.
- **4f4cb7c1**: chai-1's grouped outer product's permutes in 32-bit index arithmetic. 13.0 -> 1.0 ms a trunk pass at
  68 tokens. Byte-identical.

A sweep of every model's heaviest non-GEMM kernels at 255 tokens (5CAJ, one pass) finds no more 64-bit-division
outliers. What remains, all bandwidth or scalar work:
- the centre norms: 55-111 ms a pass at C 128-256, and 295-370 at OpenDDE's and IntelliFold-2's 384-512 (the wide
  kernel; a register-resident one did not pay, above);
- **chai-1's parallel pairformer copies the pair every block** (`copy(base, pair)`: `lf_copy` 66 ms a pass at 255
  tokens, about 4%). Ping-pong buffers would remove it, but the first update must then run out of place (its last
  GEMM reading C = the input and writing D = the other buffer; `lf_gemm_tensor_dual` reads D today). Not done;
- **ESMFold2's atom windowed attention** (`ef2_swa`: scalar float, 54 ms over the sampler's 11 steps, about 1.6% of a
  fold). Its K and V rows are gathered through `valid[]`, so the matrix units would need them staged. A fast exp
  changed nothing.

### Morning summary (2026-10-10)

- **Head of `metal`**: eb5ccd1c and its docs. All 29 gates, the selftest, `test:native` (offline, no AF3, no page) and
  `test:bridge` pass on it.
- **Folds vs the start of the night**, alternated: 10-21% faster across AF3, boltz2, chai1, AF2 (with and without an
  MSA) and ESMFold2. Weights load 1.7-2.6x faster. The table is under "The night, end to end".
- **What carried it**: attention v3 (the online softmax in cooperative tensors, about 2.8x the original kernel), the
  triangle gate and the GEMM epilogues in registers, fast math in those epilogues, and the 64-bit-division class of
  bug in the loader, MSA and permute kernels. Also the diffusion conditioning batched across noise levels, and the
  triangle's tail as one dual GEMM.
- **What is left**, each needing a real project: int8-resident weights with a dequantising GEMM (the small-fold floor
  is weight reads); the pair LayerNorm emitted by the previous GEMM's epilogue (~3%); and attention itself (6-7
  TFLOP/s; MPP's single-simdgroup rule for cooperative inputs blocks the obvious bigger tiles).

### 2026-10-10, daytime: the emitted LayerNorm, and what the WebGPU side has left

- **fbd35863 (from the M2)**: `MTLGPUFamilyApple10` and `MTLLanguageVersion4_0` by value. The macOS 26 SDK is the only
  one that names them, so the M2's SDK 13.3 could not build a1a0d4f5's tensor path. Checked against this SDK's headers
  (1010, `4 << 16`). Rebuilt here: the selftest and all 29 gates pass. **Host code must build on an old SDK**: a Metal
  name newer than macOS 13 goes by value, and the `@available` check stays.
- **A fresh install's first fold** (the kernel source salted so neither cache could answer, 5CAJ): AF3 9.22 -> 9.52 s
  with the wheel's kernel list and 10.52 without; AF2 4.10 -> 4.35 / 5.07; ESMFold2 3.26 -> 3.56 / 4.39. That is
  0.3 s with the list. A `.metallib` built offline would need the full Xcode at wheel-build time; not done.
- **3e8d4600, AF3: the pair's next LayerNorm emitted by the previous update's GEMM.**
  - The trunk is 4.95 -> 4.78 s at 255 tokens (4 passes, alternated). In a profiled pass, layernorm went from
    161 ms / 551 dispatches to 30 / 215.
  - The emitters are the triangle's fused tail, the grid attention's output projection and the transition's second
    GEMM. The transition writes two norms from the same statistics.
  - The statistics come from shuffles plus one threadgroup exchange, deterministic. The probe:
    - Each pair row of a 64 x 128 float destination is held by 4 lanes (lane ^ 1, ^ 8) in each of two simdgroups
      (sg ^ 1).
    - Element bit 2 is row + 8 and bit 5 is row + 32; bits 3-4 are column + 32.
  - The first version reduced through an 8 KB threadgroup table with 3 barriers. That cost the tail +42 ms against
    -52 saved.
  - The cost is a second pair-sized half buffer (279 MB at 1044 tokens). `LOCALFOLD_LN_EMIT=0` is the control.
- **a1e26132, AF2: the same, plus the triangle attention's duplicate LN.** The bias's LN(pair) was the queries' LN,
  computed twice. Single sequence 1023-1039 -> 963-979 ms (-5.6%). With the 1500-row MSA it is level: that fold's
  LayerNorm is mostly the MSA's 256-wide rows.
- **Lost: the diffusion's gated residuals and boltz2's up-gate in their GEMMs' epilogues, at 255 tokens too.**
  Byte-identical. boltz2 4104/3936 -> 4126/3927 ms, chai1 2390/2386 -> 2342/2355, protenix2 level. Now measured
  at 16, 68 and 255 tokens.
- **ESMFold2 is not covered**: its pair is 256 wide, two 64 x 128 tiles a row, so no epilogue sees a whole row.
  LayerNorm is 8.6% of its trunk (221 ms a pass at 255). A 32 x 256 destination with its own probed layout is the
  way in. Not done.
- **WebGPU on the M5**, stock Chrome (`LOCALFOLD_STOCK_FLAGS=1`, adapter `apple / metal-3`, so the M2's prior):
  - **The baseline**: AF3's 255-token trunk pass is 3.34 s against Metal's ~1.2.
  - **Profiled per pass**: pair-transition 669 ms, grid.attend 638, grid.project 476, tri.project 340, tri.contract
    288, tri.project-out 277. That is 1.4-2 TFLOP/s a kernel.
  - **`probe-alu.js`**: f32 2.77 TFLOP/s scalar and 11.3 vec4; f16 3.66 and 14.3. **f16 is only 1.3x here**, against
    1.7x on the M2, so half arithmetic is a smaller lever than on the part this code was tuned on.
  - **The trunk's precision options**: `--pair-weights`, `--accumulate`, `--staged` and `--weights` at f16 are level
    or slower (two rounds, interleaved).
  - **`exp2` in log2 units** in the tiled grid.attend: 638/666 against 641/682 ms, level (about 1%), so not committed.
    The exponentials are not what bounds it.
  - **What is left is kernel work**: the vector GEMMs' tiling at the M5's instruction rate, and grid.attend's
    barrier-heavy tile.
- **05c066c5, ESMFold2: the emitted LayerNorm on 32 x 256 tiles.**
  - The tile: a 32 x 256 destination over 4 simdgroups. Probed: a row is 4 lanes (^ 1, ^ 8) in each of the four
    simdgroups, with the same 4-adjacent-column groups. A GEMM whose 256-wide output wants its next norm takes this
    tile.
  - The tile's own cost: level at 256 x 65025 x 256 (0.900 against 0.903 ms) and 5% slower at 256 x 43690 x 1024.
  - The epilogue reads the tile's values back from D, where each thread just wrote its own, instead of holding 16
    float4 registers:
    - the gated add, a pass: 419 ms (registers) -> 369 ms;
    - AF3's trunk against the register version: 5.05 / 4.94 -> 4.77 / 4.79 s.
  - The trunk: 2604 -> 2474 ms of GPU time, and the fold's wall time 3.22 / 3.27 -> 3.15 / 3.18 s.
- **ea79ad2e, WebGPU: grid.attend's tiled form with f16 tiles, in the metal-3 prior.**
  - **What**: q, k, v and P are staged in f16, and the q.k dot is taken in f16 four products at a time. The softmax
    statistics and the output stay f32.
  - **Gain**: grid.attend 668 / 690 -> 491 / 505 ms per 255-token trunk pass (-26%), the pass's GPU time -3.5%.
  - **Where it applies**: the tiled form only (an M2 never ran it). Its f32 tiles are level with the untiled kernel,
    and 8x4 tiles are 2.5x worse.
  - **Accuracy**: check-af3-grid-attention (new `--model=` and `--tiled-half`) gives 4.6e-3 / 5.2e-3, the untiled
    kernel's own f16 figures. 6MRR pLDDT is 85.168 against 85.170 with the knob off.
- **The WebGPU ceiling on this part, measured** (`tools/gpu/probe-register-gemm.js`):
  - A WGSL GEMM with workgroup-staged tiles reaches 2.8-2.9 TFLOP/s in f16 and 2.0 in f32.
  - A subgroup-shuffle design reaches 0.7-1.7.
  - The trunk's kernels already run at 1.4-2.1, so a rewrite is worth at most ~1.4x a kernel.
  - **Trap**: lanes writing single components of one workgroup vec4 race on Metal (a read-modify-write of the vector).
- **WebGPU, also level**:
  - AF2's `attentionTiled`: inert on the f16 path.
  - The ampere prior wholesale.
  - `--pair-weights`, `--accumulate`, `--staged` and `--weights` at f16.
  - What remains is per-kernel. In AF2's block (255 x 256): opm.contract 20 ms, then the MSA attentions and their
    projections at 10-12 each.
- **Not done**:
  - 8 x 8 matrix-unit attention: only a flagged browser reaches it, and Apple's units are 8 x 8 where the kernel
    declares 16 x 16.
  - Moving the diffusion sampler onto the GPU: a step's GPU is 93-96% busy already.
- **Lost (WebGPU, the M5): grid.project as a register-blocked GEMM.** 128 rows a workgroup, a lane 8 x 2 channels'
  (q, k, v, gate), the activations staged in f16.
  - **The arithmetic**: with the weights also staged in f16 and multiplied in f16, the transposed direction's f16 arm
    reads relRMS **2.06e-2** against the old kernel's 5.2e-3, deterministic. That is the weights' rounding: staged in
    f32 it is 9.1e-4.
  - **The time**: with f32 weights the arithmetic is f32, and an f32 WGSL GEMM here is ~2.0 TFLOP/s. grid.project went
    474 -> 448 ms a pass (-5%), the trunk inside the drift.
  - **The same ceiling decides the pair transition**: it already runs its 25.6 GFLOP at 2.06 effective, and its f16
    arm was declined on the M2 for accuracy. Not attempted.

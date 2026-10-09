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

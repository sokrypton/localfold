# LocalFold on Metal

The Apple-native port: **AlphaFold 3's lineage (seven models), AlphaFold 2 (monomer and multimer, models 1-5) and
ESMFold2, written for Metal** - no CUDA translation, no patches. Each port reads the same featurised input as
`cuda/<port>` (`cuda/featurise`, linked in) and the same published weights, and folds to the same structures; the
arithmetic is chosen for Apple's GPUs (8 x 8 simdgroup matrices, unified memory, no integer divider).

```
bash metal/build.sh af3 af2 ef2                # -> metal/<port>/localfold-<port>; only the command-line tools needed
metal/af3/localfold-af3 --out=6mrr.pdb --model=boltz2 --sequence=GWSTELEKHREEL...
metal/ef2/fold 6mrr.pdb --sequence=GWSTELEKHREEL...    # cuda/<port>/fold's interface
bash python/build_wheel_macos.sh               # the macosx_13_0_arm64 wheel: `pip install localfold` on a Mac
```

The binaries take the CUDA binaries' flags (`--help`), fetch their weights into `~/.cache/localfold` (or
`--weights-dir=`), and need macOS 13 (Metal 3: a buffer's GPU address) on Apple silicon. Kernels are Metal source
compiled at run time; the GEMM instances a run used are listed in `~/.cache/localfold/metal/native-<port>-<hash>.specs`
(a wheel ships one) and compiled up front, in parallel, the next time.

## Layout

```
metal/
  build.sh       builds metal/<port>/localfold-<port> (and selftest, bench)
  core/          what every port is written against:
    core.h/.mm     the device, memory (shared buffers addressed by GPU address), dispatch, the GEMM and its fused
                   epilogues, the gated flash attention, LayerNorm and elementwise kernels, profiling
    args.h         every kernel's argument struct, one declaration read by C++ and by Metal
    common.metal   the core's kernels (attention, LayerNorm, the bundle decoder, conversions)
    gemm.metal     the GEMM on 8 x 8 simdgroup matrices, compiled per instance
    model.h/.cpp   weights and inputs by name: page bundles and af3-any-model blobs decoded on the device, through
                   cuda/featurise's weight walks (af2_weights.h, af3_weights.h), delta bundles, featurised inputs
    host.h/.cpp    the drivers' shared host pieces: --serve, frame superposition, whole-file writes
  af3/ af2/ ef2/ one per port, flat: the port's kernels (kernels.h, <port>.metal), its stages (*.mm), main.mm,
                 fold, gate.py, gate-baseline.json
  selftest/      the GEMM, attention and LayerNorm against host references
  bench/         GEMM and attention arms, interleaved, each checked against the first
```

A port's Metal source is `core/args.h + <port>/kernels.h + core/common.metal + <port>/*.metal`, embedded in the
binary by `build.sh`; the GEMM's is `args.h + gemm.metal`, compiled per (types, tile, epilogue) instance on demand.

## Is it right? The gates

The CUDA gates' cases, scored against crystals, with this Mac's baseline in `metal/<port>/gate-baseline.json`
(0.05 A and 0.5 pLDDT) and the A100's printed beside each case:

```
python3 metal/af3/gate.py [--write] [--only=<case or model>]   # 14: seven models, ligand + phosphoserine, DNA,
python3 metal/af2/gate.py [--write]                             #   methylated DNA, kitchen sink, glycan, templates
python3 metal/ef2/gate.py [--write]                             # AF2 5, ESMFold2 8
```

On an M2 (10-core GPU), 2026-10-09: every case passes. Against the CUDA-translated port this replaced, case by case:

| | native | translated |
|---|---|---|
| AF3 6MRR | 0.771 A, pLDDT 86.16 | 0.768, 86.15 |
| protenix2 / boltz2 / intellifold2 6MRR | 1.561 / 0.578 / 1.539 A | 1.561 / 0.577 / 1.539 |
| rosettafold3 / openbind0 / opendde 6MRR | 1.699 / 1.759 / 1.457 A | 1.706 / 1.756 / 1.454 |
| AF3 5CAJ / 1BRS self-template | 0.205 / 0.457 A | 0.205 / 0.454 |
| AF2 6MRR / 5CAJ + template / 1BRS multimer + template | 1.899 / 0.216 / 0.272 A | 1.900 / 0.216 / 0.272 |
| ESMFold2 6MRR / 1QYS / 5CAJ / 1BRS | 1.353 / 0.865 / 2.105 / 0.912 A | 1.362 / 0.865 / 2.104 / 0.911 |

The AF3 trunk's pair after one pass agreed with the translated port's at relRMS 8.5e-4 (single 5.6e-4) on the first
build. `LOCALFOLD_SAVE_SEAMS=<dir>` writes AF3's stage outputs as `.f32` files for exactly that kind of comparison.

Not ported: chai-1 (its token features need ESM2 3B, which was not on this machine to check against), and ESMFold2's
released-model extras (the parcae recycle and the MSA encoder). Each refuses by name.

## How fast

On the M2, warm, the fold alone unless said:

| | time | |
|---|---|---|
| AF3 6MRR (68 tokens, 3 recycles, 200 steps) | 6.3 s: trunk 1.3, diffusion ~5 | 6.2 s translated, 12.0 WebGPU |
| AF3 5CAJ self-template (255 tokens) | 31 s with start-up | |
| ESMFold2 6MRR / 5CAJ | 0.3 / 0.8 s | |
| AF2 6MRR / 5CAJ + template | 1.2 / 3.9 s | 14.4 s translated |

Profile any fold with `LOCALFOLD_PROFILE=1` (every labelled dispatch its own command buffer: an upper bound, good for
proportions) and AF3's denoiser by stage with `AF3_STAGES=1` (a sync between stages: real time).

What won, natively:

- **The diffusion transformer's GEMMs on rows rounded up to 16.** A short n (68 tokens) takes one tile row of 80, and a
  ragged tile runs the GEMM's bounds-checked staging and scalar stores: 0.31 against 0.22 ms on 768 x 3072. The
  transformer 33.7 -> 22.6 ms a step at 68 tokens; the padding rows compute values nobody reads.
- **The atom attention on simdgroup matrices** (`af3_atom_attention_mma`: 32 queries against 128 keys a threadgroup,
  the softmax in registers): the encoder and decoder 3.4 -> 1.8 ms a step each. `AF3_SCALAR_ATOM_ATTENTION=1` is the
  control.
- **Every attention on one flash kernel** (`core/common.metal`, heads 8-64 wide): the grid attention's column
  direction, AF2's MSA column and ending-node attentions, ESMFold2's diffusion - read ACROSS a tensor's leading axis
  through strides, so nothing is transposed. 1.22 TFLOP/s at D 32 on the M2 (0.15 before its loops were unrolled).
- **The diffusion transformer's 96 conditioning projections folded into two GEMMs** (the per-block LayerNorm scales
  folded into the weights, the biases as GEMM biases).

What was tried and **lost**, so nobody repeats it blind:

- **Split K for the skinny GEMMs** (partial products a K slice, then one reduction applying the epilogue): 0.395
  against 0.334 ms on 3072 x 68 x 768, and the trunk slower. These GEMMs are not short of threadgroups.
- **Matrix stores for the whole 8-row blocks of a ragged tile**: 0.31 -> 0.30 ms, and one instance hit a Metal
  compiler internal error. Padding the rows (above) does it for free.
- From the translated port's measurements, which hold here: double-buffered GEMM tiles (1.13-1.37x the time), eight
  simdgroups a threadgroup (1.04x), 64 x 128 or 128 x 64 tiles (spilled registers), MPS (level at best, more error),
  a "skinny" GEMM reading W straight into the matrix units (uncoalesced), register prefetch (2.43 -> 1.48 TFLOP/s).

## Memory

Every weight held once, in the precision it is used in: large tensors decoded straight to float16 at load (no float32
copy is made of them; one is built on demand where a kernel reads float, and cached), the rest float32. AF3's int5
bundle is ~1.2 GB on the device. Scratch buffers are named and grown as asked; each stage gives its own back when it
is done.

## Traps, each paid for

🔴 **AN ARRAY OF SIMDGROUP MATRICES MUST BE UNROLLED** (`_Pragma("clang loop unroll(full)")` on every loop over one),
or it spills to memory: the flash attention ran at 0.15 TFLOP/s until it was.

🔴 **APPLE'S GPUs HAVE NO INTEGER DIVIDER.** `a / d` by a runtime `d` is a software routine at any width; a kernel that
divides an element runs at a fraction of its bandwidth. `lf_udiv` (`core/common.metal`) divides through a float
reciprocal with one correction, exact below 2^24.

🔴 **A BLOB'S DATA IS NOT ALIGNED.** af3-any-model's records have headers of any length, so a float32 tensor can start
at any byte; four-byte loads read garbage and protenix2 folded NaN. The decoder assembles words from bytes.

🔴 **32 KB OF THREADGROUP MEMORY, AND NO `double`.** A kernel staging more does not compile on Apple's GPUs (the CUDA
ports assume 48 KB or more); rf3's chirality gradient, double precision on CUDA, is float here with a larger step.

🔴 **THIS MACHINE THROTTLES, AND THE GPU IS SHARED WITH THE WINDOW SERVER.** The same GEMM read 0.22 and 0.42 ms in two
runs minutes apart. Compare arms interleaved in one process (`metal/bench`), never two runs.

🔴 **A FLOAT ATTENTION MASK IS ADDITIVE AND FINITE.** The flash kernel's sentinels are -1e30 past the keys and -1e9 for
a masked one, so a row whose every key is masked is uniform rather than NaN - as the references are.

🔴 **`MTLCreateSystemDefaultDevice()` RETURNS nil TO A COMMAND-LINE PROCESS** sometimes; `MTLCopyAllDevices()` does not.
And a pipeline can silently refuse the threads it is dispatched with (`maxTotalThreadsPerThreadgroup` falls as a kernel
uses more registers): no error, the output untouched.

## Next

1. The skinny GEMMs (a single sample's diffusion): weights read at ~25 GB/s where the M2 has ~100.
2. Fusions the CUDA port has: the gated residual into the next adaptive LayerNorm, the trunk's transition and
   triangle kernels.
3. chai-1 and ESMFold2's released-model extras.
4. An M5: its GPU's matrix hardware through Metal 4's tensor APIs.

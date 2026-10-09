# LocalFold on Metal

The Apple-native port, begun 2026-10-08: **the CUDA ports (`cuda/af3`, `cuda/af2`, `cuda/ef2`) translated to Metal**,
not rewritten - the same host code, the same kernels, the same featuriser, so a fold on a Mac is the fold an NVIDIA
card runs. It exists because WebGPU cannot reach what Apple silicon has: a stock browser exposes no subgroup matrix
units, WebGPU kernels pay index clamps on every access, and an M5's Neural Accelerators are reachable only through
Metal itself.

```
bash metal/build.sh [af3] [af2] [ef2]          # -> metal/<port>/localfold-<port>; only the command-line tools needed
metal/af3/localfold-af3 --out=6mrr.pdb --model=boltz2 --sequence=GWSTELEKHREEL...
metal/ef2/fold 6mrr.pdb --sequence=GWSTELEKHREEL... --fast        # cuda/<port>/fold's interface
bash python/build_wheel_macos.sh               # the macosx_13_0_arm64 wheel: `pip install localfold` on a Mac
```

The binaries take exactly the CUDA binaries' flags (`--help`), fetch their weights into `~/.cache/localfold` (or
`--weights-dir=`), and need macOS 13 (Metal 3: a buffer's GPU address) on Apple silicon. Their kernels are Metal
source compiled on first use and cached by the system; the specialisations a run used are listed in
`~/.cache/localfold/metal/<port>-<hash>.specs` so the next run compiles them all up front, in parallel.

## Is it right? The gates

The CUDA gates' own cases, scored against crystals, with this Mac's baseline in `metal/<port>/gate-baseline.json`
and the A100's printed beside each case:

```
python3 metal/check/gate-af3.py [--write] [--only=<case>]     # 14 cases: seven models, ligand, modified residue, DNA,
python3 metal/check/gate-af2.py [--write]                      #   kitchen sink, glycan, templates; AF2 5; ESMFold2 8
python3 metal/check/gate-ef2.py [--write]
```

| on an M2 (macOS 13.2) | Metal | A100 (cuda/) |
|---|---|---|
| AF2 6MRR | 1.900 A, pLDDT 84.59, pTM 0.6098 | 1.9, 84.58, 0.6097 |
| AF2 1BRS multimer + template | 0.272 A, 95.58 | 0.272, 95.59 |
| ESMFold2 6MRR / 1QYS / 5CAJ / 1BRS | 1.362 / 0.865 / 2.104 / 0.911 A | 1.422 / 0.865 / 2.077 / 0.923 |
| ESMFold2 KRAS-sotorasib | covalent bond 1.52 A, pLDDT 83.75 | 83.7 |
| AF3 6MRR, seven models | 0.58-1.76 A (protenix2 1.56 is a seed: seeds 1-4 give 0.62-0.69) | 0.46-1.77 |
| AF3 5CAJ / 1BRS self-template | 0.205 / 0.454 A | 0.182 / 0.515 |

And seam by seam against WebGPU's f32 path, under `LOCALFOLD_STOCK_FLAGS=1` (see below): AF3 trunk 3.5e-7,
protenix2 8.4e-7, intellifold2 1.1e-6 at `trunk_out_pair`; ESMFold2's language pair 1.5e-4, z_init 4.6e-5 (against
`--atom-windowed=1`, WebGPU's default is unwindowed), one denoiser call 6.8e-6 through the atom encoder.

## How fast

AF3 6MRR (68 tokens, 200 steps) folds in 6.4 s warm against WebGPU's 12.0 (cold 7.4 against 23.3); ESMFold2 6MRR in
3.2 s for the whole job. At 255 tokens an AF3 fold is ~75 s, half trunk and half diffusion. Profile any fold with
`--profile` (every kernel its own command buffer: an upper bound) and count with `LOCALFOLD_METAL_STATS=1`.

🔴 **THIS MACHINE THROTTLES, AND A BENCH CANNOT SEE IT.** The same 4096-cube GEMM read 2.43 TFLOP/s and, hours of load
later, 1.19 - same binary. A ratio taken across that gap is a ratio of clocks. Interleave arms, and re-measure the
control before believing a win.

What was tried for speed and **lost**, so nobody repeats it blind (all `metal/check/bench-gemm`, the runtime's GEMM
alone - `metal/check/build-bench-gemm.sh` builds it):

- **MPS (`MPSMatrixMultiplication`) in place of `lf_gemm`.** Standalone it looked 1.8x on 3072 x 272 x 768; through the
  runtime, timed the same way as ours, it was 1.18 against 1.25 TFLOP/s and its error 5e-4 against 2e-4. On the
  triangle's tall K-128 projections it loses outright. The GEMM is at MPS's level already.
- **A "skinny" GEMM** (n <= 128) reading W straight from device memory into the matrix units, staging only X: 8x8 loads
  at a long stride are uncoalesced, and every configuration lost (1.46 against 1.66 TFLOP/s at best).
- **Register prefetch** (the next k-step's tiles read while this one multiplies): correct and slower everywhere -
  4096-cube 2.43 -> 1.48. Apple GPUs hide latency with occupancy, and the registers cost more of it than they saved.

## How the translation works

`metal/tools/cu2metal.py` reads a port's `cuda/<port>/src` and writes `metal/build/<port>`:

- **Host code** (`.cu`/`.cuh`) compiled as Objective-C++ against `metal/shim/include`: each `__global__` becomes a stub
  that packs its arguments and calls `lf::launch`; `<<<g, b, smem, s>>>` becomes `(lf::setLaunch(...), k(args))`.
- **Device code** into one Metal source with `metal/shim/prelude.metal` (CUDA's builtins, atomics, shuffles, half and
  bf16, `lf_f64` for a double the host reads). Every kernel becomes a template instantiated lazily by name.
- `metal/shim/lfcuda.mm` is the CUDA runtime and cuBLAS on Metal: `cudaMalloc` returns a buffer's GPU address, one
  queue with a serial compute encoder, graph capture as a recorded op list, events, and `lf_gemm`
  (`metal/shim/shim.metal`) on the 8x8 simdgroup matrices.

Where a kernel cannot translate as it is (inline PTX, more than Apple's 32 KB of threadgroup memory) the port names a
replacement, and nothing in `cuda/` changes:

| | replaces |
|---|---|
| `metal/<port>/kernels/<name>.cu` | a kernel's body (e.g. af3's `atomAttentionMMA`, ef2's `swaWindowK` at 96 keys) |
| `metal/<port>/host/<name>.h` | a host function (af3's flash dispatchers onto `flashGridMetal`) |
| `metal/<port>/inject/<path>` | inserted after a file's includes (af3's `flashGridMetal`, `lnHeadsMetal`) |
| `metal/replace/<path>` | a whole file (af3's CUPTI profiler) |
| `metal/<port>/defaults.env` | environment defaults: the CUDA port's own switches to its unfused paths |

## Traps, each paid for

🔴 **`#pragma unroll` MUST BECOME `clang loop unroll(full)`.** Without it an array of `simdgroup_matrix` spills and the
GEMM ran 5x slower; the translator converts CUDA's pragmas.

🔴 **A POINTER TO DEVICE POINTERS TAKES ITS ADDRESS SPACE ON THE OUTER LEVEL.** `cublasGemmBatchedEx`'s arrays were cast
`(device const device T* const*)`, which read the pointers from the wrong memory: relRMS 1.0 on ESMFold2's adaLN
projections, a denoiser 32% off WebGPU's with identical input, and a structure with N-CA bonds of 8 A. The GEMM check
(`LOCALFOLD_CHECK_GEMM=<calls>`) skipped pointer batches, which is why nothing saw it; it samples them now.

🔴 **A HOST `double` IN AN ARGUMENT STRUCT SITS ON 8 BYTES.** `lf_f64` was 4-aligned, so every kernel's double arguments
were read 4 bytes late: ESMFold2's pTM printed 1.0 on every fold and garbage on RNA, while pLDDT was right.

🔴 **A RESIDUAL AGAINST WEBGPU CAN BE WEBGPU'S.** protenix2's trunk read 6.6e-3 off WebGPU's; bisected stage by stage it
was WebGPU's split pair transition and grid attention on the matrix units - f16 by construction, which `--f16=off`
cannot reach - and with `LOCALFOLD_STOCK_FLAGS=1` (no matrix units) the whole trunk agrees to 8.4e-7. The bisection
also found WebGPU's vector split ignoring `--f16=off` (fixed).

🔴 **`MTLCreateSystemDefaultDevice()` RETURNS nil TO A COMMAND-LINE PROCESS**, saying so only on stderr - sometimes.
`MTLCopyAllDevices()` does not. Under the sandbox there is no device at all.

🔴 **A PIPELINE CAN SILENTLY REFUSE THE THREADS IT IS DISPATCHED WITH** (`maxTotalThreadsPerThreadgroup` falls as a kernel
uses more registers): no error, the output untouched.

## The prototype (before the translation)

Two hand-written kernels against WebGPU on an M2 at 256 tokens, which is what started this: `grid.attend` 4.3 ms
against 12.2, `pair-transition` 9.8 against 19.2 (`metal/kernels/`, `metal/check/check-kernels`). Leaving WebGPU is
1.2-1.7x of it; the 8x8 matrix units, with the softmax in registers, the rest.

## Next

1. Metal forms of the CUDA port's fused tensor-core kernels (the triangle, the 128- and 256-channel transitions, the
   grid attention's projections), which run here on their unfused paths (`defaults.env`).
2. The trunk at large inputs: at 255 tokens the triangle's K-128 projections run at ~1 TFLOP/s.
3. Memory: an AF3 fold holds ~2.5 GB of decoded weights before it starts (f32 and f16 copies).
4. An M5: its GPU's matrix hardware through Metal 4's tensor APIs.

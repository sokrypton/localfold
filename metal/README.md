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
python3 metal/af3/gate.py [--write] [--only=<case>]     # 14 cases: seven models, ligand, modified residue, DNA,
python3 metal/af2/gate.py [--write]                      #   kitchen sink, glycan, templates; AF2 5; ESMFold2 8
python3 metal/ef2/gate.py [--write]
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

What **won**: the GEMM's k step at 16 for a whole 64 x 64 tile, where it was 32 - half the threadgroup memory, so more
tiles resident to hide their loads. Interleaved on an M2: 0.73-0.85x the time on the triangle's tall K-128
projections, 0.83-0.89x on the pair track's GEMMs, 0.84x on a 4096-cube in f16, and in f32 (AF2's) 0.62-0.95x. Bit
for bit the same answer: the k order of the accumulation does not change. `LOCALFOLD_GEMM_BK=16|32` is the arm.

And: **the triangle's gated residual in the gate GEMM's epilogue** (`lf::gemmGatedAdd`, applied by
`metal/af3/pairtrack.cuh.patch`): `gatedAddK`'s pass and the gate tensor gone, 19.99 -> 19.74 s of GPU at
255 tokens; `LOCALFOLD_UNFUSED_GATE=1` is the control. The gate is now f32 where the port rounded it to f16, and
**`af3-kitchen-sink` moved 74.30 -> 75.44** - that case is bimodal under any rounding change (a two-pass LayerNorm took it
to 75.37), so its baseline was re-recorded; every other case read its digits. A **patch** is an exact old -> new block
applied to a CUDA source at translation; a block that no longer matches stops the build and names itself.

And: **an all-half GEMM accumulates in half** (`simdgroup_half8x8`). 0.85x the time on the trunk's K-128 projections,
interleaved, for a relRMS of 1.7e-3 where float accumulation gives 2.4e-4 - bfloat16's rounding, which is what AF3
runs at. GPU at 255 tokens 18.04 -> 17.31 s. A GEMM writing float (a residual) keeps float. Accumulating each k step
in half and adding it to a float accumulator (6.6e-4) won nothing: the win is the accumulator's registers. On
rosettafold3-6mrr the two arms agree to 0.01 A seed by seed (1.005/1.006, 1.672/1.671, 1.859/1.854, 1.770/1.782), and
the gate's own seed moved 1.699 -> 1.764; that case and af3-glycan (44.35 -> 45.30, toward the A100's 47.42) were
re-recorded. **AlphaFold 2 keeps float** (`metal/af2/defaults.env`): its no-alignment multimer moved 0.75 pLDDT, and
its reference is a float model. `LOCALFOLD_GEMM_FLOAT_ACC=1` is the control. Extending it to the half-input, float-output GEMMs at k <= 128
(the residual's updates) was 0.94x on them and **crashed** intellifold2, openbind0, opendde and two ESMFold2 cases:
one of those sums overflowed half to inf and host code downstream segfaulted - so a float output keeps float.

And **the unfused grid attention's pair bias in one kernel** (`lnBiasMetal`, `metal/af3/pairtrack.cuh.patch`): a
simdgroup a pair row takes its LayerNorm (layerNormK's arithmetic: `norm` is the same bytes) and the four heads'
logits - reduced together in one butterfly, 6 shuffles where four reductions took 20 - and writes them into the flash
kernel's bias layout. layerNormK + a 4-column GEMM (a 32-column tile, 7/8 padding) + biasLayoutK were 1176 ms at 255
tokens; now 950. GPU 16.98 -> 16.81 s. `LOCALFOLD_UNFUSED_BIAS=1` is the control. In the column direction it writes `norm` transposed
as well, so gatherTransposedK's pass (169 ms) is gone: 16.81 -> 16.61 s, the same digits (`LOCALFOLD_GRID_NORM_T=0`,
the port's own switch, is the control).

And **the diffusion's transitions take the SwiGLU in their GEMM** too (`metal/af3/diffusion.cuh.patch` for the token
transformer, `atom.cuh.patch` for the atom blocks): swigluK's 30 passes a step gone, ~0.8 ms of a 69 ms step.

🔴 **rosettafold3-6mrr, af3-glycan AND af3-kitchen-sink SWING UNDER ANY ROUNDING CHANGE.** rf3's RMSD went 1.699 -> 1.764 -> 1.706 over
two changes that each move the other cases' digits only, and glycan's pLDDT 44.35 -> 45.30 -> 44.63; seed by seed
the arms agree (above). A MOVED on those two alone, with every other case in its digits, is that and not a defect.

And in `flashGridMetal`: K and V staged four halves a load (3109 -> 2630 ms at 255 tokens). center_norm's
statistics summed while loading, by all eight rows of threads where one row did it alone (613 -> 365 ms).
`layerNormK` is at the M2's bandwidth already (~95 GB/s), and a pair-bias read two halves at a time moved nothing.

**Memory: every weight held once.** The CUDA port keeps each weight tensor as float32 *and* its f16 mirror - 2.51 GB
for AF3's int5 bundle before the trunk starts, of which the mirror is 0.85. `metal/af3/af3.cu.patch` drops the float
copy of every tensor of 64 K elements or more once its mirror exists (1.33 GB); the few paths that read one as float -
mostly a concatenation's parts, read once - get a copy rebuilt from the mirror, given back at the next phase boundary.
Not the input embedder's (bisected by prefix: their rounding took IntelliFold-2's pLDDT down 0.8 on every seed while
the structure did not move) and not the confidence head's. At 255 tokens: 2.51 -> 1.18 GB in use before the trunk, the
fold's peak 4.95 -> 3.85 GB. `LOCALFOLD_KEEP_F32=1` is the control.

What was tried for speed and **lost**, so nobody repeats it blind (all `metal/tools/bench-gemm`, the runtime's GEMM
alone - `metal/tools/build-bench-gemm.sh` builds it):

- **`flashGridMetal`'s tiles and accumulator** (255 tokens, its 488 calls): keys 32 at a time 2611 -> 2958 ms, 8 at a
  time 2710, 64 queries a threadgroup (eight simdgroups sharing a staged key tile) 2859, and the output accumulated in
  half 2637. Keys 16 at a time over 32 queries with a float accumulator is where it sits.
- **MPS (`MPSMatrixMultiplication`) in place of `lf_gemm`.** Standalone it looked 1.8x on 3072 x 272 x 768; through the
  runtime, timed the same way as ours, it was 1.18 against 1.25 TFLOP/s and its error 5e-4 against 2e-4. On the
  triangle's tall K-128 projections it loses outright. The GEMM is at MPS's level already.
- **A "skinny" GEMM** (n <= 128) reading W straight from device memory into the matrix units, staging only X: 8x8 loads
  at a long stride are uncoalesced, and every configuration lost (1.46 against 1.66 TFLOP/s at best).
- **Register prefetch** (the next k-step's tiles read while this one multiplies): correct and slower everywhere -
  4096-cube 2.43 -> 1.48. Apple GPUs hide latency with occupancy, and the registers cost more of it than they saved.
- **The epilogue.** Storing a lane's pair as one `half2`, or staging a half tile through threadgroup memory so a row
  leaves in whole 16-byte pieces: 0.99-1.03x and 1.05x SLOWER, interleaved (`AB=1 metal/build/bench-gemm`). 🔴 The
  measurement that sent me there was wrong: skipping the stores of 15 of 16 matrices read 2.5x faster, because the
  compiler then deleted the multiplies feeding them. A K-128 GEMM is slow in its k-loop, not its stores.

## How the translation works

`metal/tools/cu2metal.py` reads a port's `cuda/<port>/src` and writes `metal/build/<port>`:

- **Host code** (`.cu`/`.cuh`) compiled as Objective-C++ against `metal/runtime/include`: each `__global__` becomes a stub
  that packs its arguments and calls `lf::launch`; `<<<g, b, smem, s>>>` becomes `(lf::setLaunch(...), k(args))`.
- **Device code** into one Metal source with `metal/runtime/prelude.metal` (CUDA's builtins, atomics, shuffles, half and
  bf16, `lf_f64` for a double the host reads). Every kernel becomes a template instantiated lazily by name.
- `metal/runtime/lfcuda.mm` is the CUDA runtime and cuBLAS on Metal: `cudaMalloc` returns a buffer's GPU address, one
  queue with a serial compute encoder, graph capture as a recorded op list, events, and `lf_gemm`
  (`metal/runtime/runtime.metal`) on the 8x8 simdgroup matrices.

## Layout

`metal/` sits beside `webgpu/` and `cuda/` and is laid out like them:

```
metal/
  build.sh            builds metal/<port>/localfold-<port>
  runtime/            the CUDA runtime and cuBLAS on Metal (lfcuda.mm), the CUDA vocabulary in Metal (prelude.metal),
                      the runtime's own kernels and GEMM (runtime.metal), the CUDA headers the host code includes
  tools/              cu2metal.py (the translator), msl-check, the GEMM bench
  af3/ af2/ ef2/      one per port, flat, like cuda/<port>/: fold, gate.py, gate-baseline.json, defaults.env,
                      and the port's changes to its CUDA sources
```

A port's changes are named by what they change, and nothing in `cuda/` is edited:

| in `metal/<port>/` | what it does |
|---|---|
| `<file>.patch` | exact `old` -> `new` blocks applied to `cuda/<port>/src/<file>`, and `insert` blocks placed after its includes (af3's `flashGridMetal` and `lnHeadsMetal` kernels, the fused gate and SwiGLU in `pairtrack.cuh`). A block that no longer matches stops the build and names itself |
| `<kernel>.kernel.cu` | a kernel's body (af3's `atomAttentionMMA`, ef2's `swaWindowK` at 96 keys) - where it cannot translate, or needs more than Apple's 32 KB of threadgroup memory |
| `<function>.host.h` | a host function (af3's flash dispatchers onto `flashGridMetal`) |
| `<file>` | a whole file (af3's `profile.cuh`, the CUPTI profiler) |
| `defaults.env` | environment defaults: the CUDA port's own switches to its unfused paths |

A change belongs to the port that owns the CUDA file: cuda/af2 and cuda/ef2 include cuda/af3's headers, and metal/af3's
changes to those apply to them too.

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

🔴 **A GRAPH CAPTURE IS THE CAPTURING THREAD'S - CUDA's `cudaStreamCaptureModeThreadLocal` - AND THIS RUNTIME'S WAS
GLOBAL.** ESMFold2 uploads its weights on a thread while a warm-up fold runs, and the warm-up's sampler captures a
graph; an upload step in that window went into the graph and its event record was dropped, so the wait guarding a
decode buffer returned at once and the next shard overwrote it before its decode ran. 1QYS folded at 13.5 A or to NaN
against 0.865, ~3% of runs, only under load or on a slow first run - and every ruled-out arm before it (NaN slack past
buffers, NaN-poisoned threadgroup memory, shader validation, barrier audits) said, correctly, that no kernel was at
fault. Found with stage checksums and then per-tensor weight sums: 71 of 1183 tensors differed, a shard's worth. Fixed:
0 of 90 under the same two-process stress where 1 in ~35 failed before. **A CUDA API's thread semantics are part of
what the runtime must emulate, not only its arithmetic.**

🔴 **`MTLCreateSystemDefaultDevice()` RETURNS nil TO A COMMAND-LINE PROCESS**, saying so only on stderr - sometimes.
`MTLCopyAllDevices()` does not. Under the sandbox there is no device at all.

🔴 **A PIPELINE CAN SILENTLY REFUSE THE THREADS IT IS DISPATCHED WITH** (`maxTotalThreadsPerThreadgroup` falls as a kernel
uses more registers): no error, the output untouched.

## The prototype (before the translation)

Two hand-written kernels against WebGPU on an M2 at 256 tokens, which is what started this: `grid.attend` 4.3 ms
against 12.2, `pair-transition` 9.8 against 19.2 (removed once the translation folded; commit 278adfde has them). Leaving WebGPU is
1.2-1.7x of it; the 8x8 matrix units, with the softmax in registers, the rest.

## Next

1. Metal forms of the CUDA port's fused tensor-core kernels (the triangle, the 128- and 256-channel transitions, the
   grid attention's projections), which run here on their unfused paths (`defaults.env`).
2. The trunk at large inputs: at 255 tokens the triangle's K-128 projections run at ~1 TFLOP/s.
3. Memory: an AF3 fold holds ~2.5 GB of decoded weights before it starts (f32 and f16 copies).
4. An M5: its GPU's matrix hardware through Metal 4's tensor APIs.
5. Intel Macs are not covered: every fast path here (`lf_gemm`, `flashGridMetal`, `atomAttentionMMA`, `lnHeadsMetal`)
   is built on `simdgroup_matrix`, which Metal offers only on Apple-silicon GPUs, so an Intel Mac's AMD or Intel GPU
   would need scalar fallbacks for each, and the binaries an x86_64 slice. Untestable on the machine this was built on.

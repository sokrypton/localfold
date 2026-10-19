# BindCraft2 on Metal: where to attach, and the three ways in

The ask (2026-10-19): BindCraft2 - `sokrypton/BindCraft2`, forked from `PacesaLab/BindCraft2` - has a native CUDA
implementation, so implement it on Metal.

**The thing to understand first: this is not a kernel port.** Every other port in this repository takes a forward
model written for CUDA and writes it for Metal. BindCraft2's trajectory is **gradient descent on a sequence through
AlphaFold 2**, so what it needs is a BACKWARD, and no backend here has one - not Metal, not CUDA, not WebGPU, not the
CPU references. The forward kernels are reusable as *references for shape and layout*, and not otherwise.

Everything below was measured on the M5 (macOS 26.6.2, 16 GB) on 2026-10-19.

## What BindCraft2 is made of

| | |
| --- | --- |
| the design pipeline | Python, **JAX end to end** - `jax>=0.11,<0.12`, dm-haiku, optax |
| its native half | `native/`, **23,613 lines** of CUDA, **237 `__global__` kernels, 40 of them backward** |
| how the two meet | `jax.ffi` custom calls: `dlopen` of `libnative_evoformer.so`, then `jax.ffi.register_ffi_target(..., platform='CUDA')` (`bindcraft/af/native.py:689`) |
| what the native half owns | the trunk, the extra-MSA stack, the template pair stack and embedding, the embedder, the post-trunk structure module and heads - and since 2026-10-13 the whole design step, forward AND backward, as ONE custom call (`native/design_step.cuh`) |
| what JAX keeps | the loss, and only the loss. The contract is nine arrays in `bindcraft/af/boundary.py`, which marks which carry a gradient |
| extras published | `cuda12`, `cuda13`, `rocm`. **No Metal, no CPU** |
| its floor | sm_80. bf16 tensor cores and `cp.async`, so Volta and Turing are out |

`bindcraft/af/boundary.py` is the useful part of their design for us: a loss sees `atoms, plddt, pae, ptm, iptm,
iptm_per_residue, experimentally_resolved_ca, distogram, sequence`, nothing else, and the file names which are
differentiable. That is the exact surface a Metal design step would have to produce and accept cotangents for.

`native/lf_flash.cuh` is **vendored from this repository** - "sokrypton/localfold, branch opus-55-opt,
native/af3/src/flash.cuh" (the path is `cuda/af3/src/flash.cuh` now). It is our FlashAttention-2 **forward**; their
`flash_bwd_body.cuh` is theirs, written on top of it. So the borrowing already runs in that direction, and the piece
they had to write themselves is precisely the piece we do not have.

## Path 1: run BindCraft2 unmodified on JAX-on-Metal - works, and is slow

Apple's own `jax-metal` is dead (0.1.0, `jax>=0.4.26`). Two newer third-party plugins are not:

- **`metal-pjrt-plugin`** 0.0.2 (Oct 2026, dfm) - treats Metal as an XLA:GPU platform named **`mtl`**, needs macOS 26
  and Python 3.12, claims "wide coverage of the JAX test suite" bar float64.
- **`metaljax`** 0.11.8 (Sep 2026) - pins JAX 0.11.2, macOS 14+.

`metal-pjrt-plugin` installs clean next to `jax==0.11.2`, which is inside BindCraft2's own pin. Measured with
BindCraft2's REAL `EvoformerIteration` (haiku, `use_remat=True` as `bindcraft/af2.py:362` sets it), random weights,
N=192, 8 MSA rows, forward and `jax.grad`:

| | one block, fwd | one block, fwd+bwd | implied design step (48 blocks x 2 states) |
| --- | --- | --- | --- |
| M5 GPU (`mtl`) | 37.5 ms | **91.5 ms** | **~8.8 s** |
| CPU, same Mac | 108.9 ms | 665.8 ms | ~64 s |
| their CUDA number | - | - | **0.755 s** |

So the plugin is **7.3x the CPU** and **~12x slower than their native CUDA step**. Correctness is not the problem:
at N=64 the Metal and CPU runs agree to ~7 digits in f32 (`pair_sum` 107.387070 both sides, gradient norms identical)
and closely in bf16. **This path needs no Metal code at all** - `install()` in `native.py` is a no-op without the
`.so`, and it skips any device whose `platform != 'gpu'`, so BindCraft2 on `mtl` simply takes its own JAX fallback.

The other ceiling on this path is memory: their baselines hold 9.91-14.64 GB at NMAX=512 on a 24 GB card. This
machine has 16 GB, shared with the OS.

## Path 2: port the kernels and attach them to JAX-on-Metal - BLOCKED, measured

This is the shape that would mirror CUDA: keep the Python, port `native/`'s kernels, register them as FFI handlers.
**It does not work, and the reason is not op coverage.**

`metal-pjrt-plugin`'s own integration notes say the FFI registry is static inside its dylib's copy of XLA, and that
"handlers from other libraries (e.g. jaxlib's CUDA ones) aren't visible". Confirmed directly: a minimal typed-FFI
handler in its own dylib, registered from Python exactly as `native.py:689` does it -

```
register platform='mtl'  : OK          # registration always succeeds
  CALL  platform='mtl'   : FAIL  NOT_FOUND: No FFI handler registered
                                  for ffitest_add_one_mtl on a platform METAL (canonical metal)
```

- and the same under `MTL`, `METAL` and `metal`. **The same dylib, same code path, works on `platform='cpu'`**, so
this is the plugin's private registry and not a mistake in the probe. A Metal kernel library cannot be attached from
outside; handlers have to be compiled INTO the plugin (`metal_pjrt/ffi`, Bazel), or upstream has to expose external
registration. Note also `metal_pjrt/compiler/ptx_custom_kernel_emitter_stub.cc`: CUDA-style custom kernels are
stubbed out by design.

Worth asking upstream before anyone forks a PJRT plugin - it is one question with a large answer.

## Path 3: a native Metal design loop, no JAX - CHOSEN (user, 2026-10-19)

Leave JAX. Build the design loop on `metal/`'s own AF2, which means writing the backward. The scale, honestly:

- `metal/af2/` is ~1.8k lines over `metal/core` and **forward only** (37 kernels).
- The repository contains **no reverse-mode anything**. The only `.backward()` in the tree is
  `tools/esmc/distil-esmc.py`, an offline PyTorch distillation of ESM-C weights that never touches a kernel here.
  Everything else matching `grad`/`backward`/`bwd` is an input feature (RoseTTAFold3's chirality term, which
  `shared/af3/diffusion/chiral-gradient.js` is at pains to say is not a loss), a force-field descent in the conformer
  generator, a transposed array index, or a `no_grad()` in an oracle script.
- So the reference for what must exist is their 40 backward kernels, and the work is ours.

**The oracle this gets for free, and should not give up:** the venv built for Path 1 runs BindCraft2's own AF2 forward
AND backward on this machine, on CPU and on `mtl`. Every Metal backward kernel can be checked against `jax.grad` of
the exact haiku module it replaces, at the same shapes, instead of against a hand-derived reference that repeats the
derivation. That is the independent reference AGENTS.md asks for.

### Staging

1. **The primitives** - LayerNorm, a linear layer's two GEMMs and its bias. **DONE**, see below.
2. **The pair stack's cheap halves** - the transitions, the outer-product mean, the gates.
3. **Triangle multiplication**, outgoing and incoming.
4. **Attention** - MSA row (with pair bias), MSA column, and both triangle attentions. The flash forward exists;
   the backward does not, and `native/flash_bwd_body.cuh` is 46 KB of what it takes.
5. **The structure module and the heads** - IPA, frames, side chains, and the seven arrays `boundary.py` marks
   differentiable.
6. **The loop itself** - the losses, the optimiser, the schedules. Python driving native steps, or native throughout.

### Stage 1, landed

`metal/design/` - built by `bash metal/build.sh design`, tested by `metal/design/localfold-design`.

| | |
| --- | --- |
| `kernels.h` | `LnStatsArgs`, `LnBwdArgs`, `ColSumArgs` |
| `design.metal` | `lf_ln_stats`, `lf_ln_bwd`, `lf_col_sums` |
| `grad.h` / `grad.mm` | `layerNormBwd`, `linearBwd`, `columnSum` |
| `main.mm` | the differential test |

A linear's backward needs **no kernel**: `dX = dY W^T` and `dW = X^T dY` are `metal/core`'s GEMM with one operand
read transposed, which it already serves. Only the bias's column sum is new, and it is the same reduction `doffset`
is.

**Two references, on purpose.** A double-precision analytic reference catches a kernel that does not compute what was
derived; it cannot catch a derivation that is wrong, because it repeats it. So every `dx` is ALSO checked by central
difference against the forward alone - `S(x) = sum(dy * forward(x))`, and `dS/dx` must be what the kernel produced.
All 32 cases pass: the analytic comparisons at **7e-8 to 5.8e-7** relRMS (float32 roundoff), and the
central-difference checks at **1.6e-8 to 4.2e-8**, which is the derivation confirmed rather than merely echoed.

Two things decided while writing it, both worth keeping:

- **The row statistics go in a buffer, not recomputed per consumer.** `dscale` needs `n[r][c]` for one channel per
  thread, and recomputing the row's mean and variance there reads the whole row per channel - O(rows * C^2) for the
  tensor instead of O(rows * C). Two floats a row buys it back, and a training forward would save them anyway.
- **No float atomics in the column sums.** Letting every row-block accumulate straight into the result is the obvious
  shape and it sums in whatever order the groups retire, which AGENTS.md's "deterministic differential test" forbids.
  One threadgroup a channel with a fixed-order reduction instead. The two-stage reduction this leaves on the table is
  deliberate - correctness before performance, and no measurement has asked yet.

`lf_col_sums` serialises a channel's rows across 256 threads, which is the right shape while C is 128-384 and rows is
large, and the wrong one for a short tensor. Noted, not fixed.

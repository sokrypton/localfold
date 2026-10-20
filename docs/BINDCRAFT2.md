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

Rebuilding it, which takes a couple of minutes and needs no weights at all (random haiku init is enough - the
question is the derivative, not the checkpoint):

```bash
python3.12 -m venv .venv-mtl                       # 3.12+; macOS 26+ for the plugin
.venv-mtl/bin/pip install jax==0.11.2 jaxlib==0.11.2 metal-pjrt-plugin \
                          dm-haiku optax ml_collections absl-py biotite
git clone --depth=1 https://github.com/sokrypton/BindCraft2   # for bindcraft/af/alphafold
JAX_PLATFORMS=mtl .venv-mtl/bin/python your_probe.py          # or =cpu for the oracle's oracle
```

Two traps found doing it. The plugin loads but does **not** become the default device - without `JAX_PLATFORMS=mtl`
`jax.devices()` returns `CpuDevice` and everything silently runs on the CPU, which looks like a working Metal run.
And AF2's `EvoformerIteration.__call__` takes `use_dropout`, not `is_training`, and needs a real PRNG key passed to
`apply` even with dropout off.

### Staging

1. **The primitives** - LayerNorm, a linear layer's two GEMMs and its bias. **DONE**, see below.
2. **The pair stack's cheap halves** - the transitions, the outer-product mean, the gates. **DONE**, see below.
3. **Triangle multiplication**, outgoing and incoming. **DONE**, see below.
4. **Attention** - the core backward, the Attention module, and the biased self-attention both
   `MSARowAttentionWithPairBias` and `TriangleAttention` are. **DONE**, see below. Left in the tier: the MSA COLUMN
   attentions, of which `MSAColumnGlobalAttention` is a genuinely different shape (mean queries), and the
   `per_column` orientation, which is a transpose the caller does as AF2 does it (`modules.py:703`, `:731`).
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

### Stage 2, landed: the first whole modules

**131 cases, all passing** (`metal/design/localfold-design`). Added: `lf_relu_bwd`, `lf_gate_bwd`, and the seven
`lf_opm_*` kernels; on the host, `reluBwd`, `gateBwd`, `transitionBwd` and `outerProductMeanBwd`.

**`Transition`** (Alg. 9 MSATransition / 15 PairTransition) - `out = relu(LayerNorm(act) W1 + b1) W2 + b2`. All seven
gradients - the input, both LayerNorm parameters, both weights, both biases - agree with central difference at
**1e-8 to 5e-8**, and with the analytic double reference at **7e-8 to 5.3e-7**.

Two things worth knowing about it:

- **AF2's `Transition` builds a mask and never applies it** (`modules.py:277` sets it, `:297` passes only `act`). Not
  a simplification taken here - the reference really does drop it.
- **The pre-relu activation never has to be saved**, because `relu(h) > 0` exactly where `h > 0`, so the post-relu
  value - which the second product needs anyway - carries the mask. The module recomputes its forward rather than
  taking a save set, which is what AF2 runs under in any case (`use_remat=True`).

**`OuterProductMean`** (Alg. 10). The one real tensor contraction of the stage, and the only place so far where the
shape of the backward had to be chosen rather than derived:

- `P[i][j][c][e]` and its cotangent are **[N][N][O][O]** - 1024 floats a pair, **151 MB at N=192, 604 MB at N=384**.
  Neither is materialised whole: the backward runs in **tiles of T rows of i**, holding `[T][N][O][O]`.
- `dWo` and `dright` accumulate across tiles. They do it in **separate dispatches**, which run in stream order, so
  the sum has a fixed order and the test is deterministic. A float atomic would have been the obvious way and is the
  wrong one - same reason as `lf_col_sums`.
- **The alternative that was rejected:** fold `Wo` into `right` first, `V[s][j][c][f] = sum_e Wo[c][e][f]
  right[s][j][e]`, which avoids `dP` altogether. It is **S-dependent** - 25 MB at S=8 but **1.6 GB** at the extra-MSA
  stack's S=512 - and costs `F/O = 4x` the multiplies. The tiled form is independent of the MSA depth, which is the
  axis that varies most between the two stacks.

Checked against central difference only, on all nine gradients, at **8e-8 to 1.9e-6**: for a four-index contraction
an analytic double reference is about as likely to be wrong as the kernel, so the forward is the thing to trust.
Cases include a tile that does not divide N (N=17, T=5) and the reference's real O=32 / F=128.

Every `lf_opm_*` kernel is one thread an output element looping the contracted axes serially. `dP`'s stride in `j`
is `O*O`, so the reads are strided; that and the tile constant are the whole optimisation surface, and nothing has
been measured yet because nothing here is correct-and-slow until the stack runs end to end.

### Stage 3, landed: TriangleMultiplication, both equations

**199 cases, all passing.** Added `lf_mul_sigmoid`, `lf_tri_pack`, `lf_tri_unpack` and
`triangleMultiplicationBwd`; all 17 gradients (the input and sixteen parameters) agree with central difference at
**3.7e-8 to 6.6e-7**, for the outgoing AND the incoming equation.

This is where the Evoformer's **O(N^3)** lives, so the shape of the backward matters more here than anywhere so far:

- **The two contractions are batched GEMMs over the channel**, which means the operands are packed channel-major,
  `[n2][I] -> [I][n2]`. That is the same reason `metal/core`'s forward keeps channel-major planes for the triangle
  (`gemmTriGate` writes them that way): without it each channel's matrix is strided by `I` and the product cannot be
  one GEMM. The packs are O(N^2 I) against the product's O(N^3 I), so they are free.
- **The four products differ between the equations**, and getting them wrong is invisible in a forward test, so both
  are checked. Outgoing is `p_c = A_c B_c^T`, `dA = dP B`, `dB = dP^T A`; incoming is `p_c = B_c^T A_c`,
  `dA = B dP`, `dB = A dP^T`.
- **The order in the forward is mask THEN gate** - `aM = mask * (act0 Wl + bl)` and then `a = aM * sigmoid(...)`
  (`modules.py:961` against `:980`). So on the way back the gate comes off first and the mask second. Reversing them
  passes a forward test and fails a gradient test, which is the kind of thing only the central difference catches.
- `dP` is written over `P`, which is finished with by then, and the whole module recomputes its forward.

**Open, and the next thing to fix here:** this holds about nineteen `[n2][C]`-or-`[n2][I]` scratch buffers, which is
**~360 MB at N=192, C=I=128** and over a gigabyte at N=384. `metal/core` has `shareScratch` for exactly this - the
forward's stage families already share slots - and none of it has been applied here yet. Correctness first, but this
is the ceiling that will bite before the stack runs end to end on 16 GB.

### Stage 4, landed: the attention backward, flash-shaped

**219 cases, all passing.** `attentionFwdSaved` and `attentionBwd`, over five kernels (`lf_attn_ref_fwd`,
`lf_attn_drow`, `lf_attn_bwd_dq`, `lf_attn_bwd_dkv`, `lf_attn_bwd_dbias`). All four gradients - dq, dk, dv and the
non-batched pair bias - agree with central difference at **5.3e-8 to 7.8e-7**, and the reference forward agrees with
a double host forward at **1.0e-7**.

**Nothing here stores a score, and that is forced rather than chosen.** The attention weights are `[B][H][Q][K]`, and
in a triangle attention `b` runs over residues too, so that buffer is **O(N^3)** - 113 MB at N=192 with four heads -
which AGENTS.md forbids outright. So every pass recomputes `s` from q and k. The saved **logsumexp** `L[b][h][q]` is
what makes that cheap: with it a pass needs no row maximum, just `p = exp(s - L)`.

**Three passes, because of who accumulates over what.** This is the part worth understanding before touching it:

| pass | a simdgroup owns | inner loop | why |
| --- | --- | --- | --- |
| `dq` | `(b, h, q)` | k | dq accumulates over k |
| `dk`, `dv` | `(b, h, k)` | q | both accumulate over **q**, so q cannot be the outer axis |
| `dPairBias` | a thread per `(h, q, k)` | b | the bias is **shared across b**, so its gradient sums over b |

The obvious single-pass alternative - write `dS` once and let the other reductions read it - is precisely the
O(N^3) buffer that is not allowed. Three passes recompute the scores three times; fusing `dq` with `dPairBias` under
a deterministic tree reduction over b is the optimisation left on the table, and it is not worth doing before the
stack runs end to end. No pass uses an atomic, so all of it stays deterministic.

**One thing that needs a core change later:** `metal/core`'s tuned `attention()` does not return the logsumexp, so
the backward cannot be paired with it yet - `attentionFwdSaved` here is a correct but untuned stand-in (a simdgroup a
query, no simdgroup matrices, scores recomputed twice in the forward). Saving `L` from the real flash forward is the
cheap fix and the right one; writing a second tuned attention would not be.

Held to `D <= 64` (a lane carries at most two channels), which covers every AF2 head here: 32 for the triangle and
MSA row/column, 8 for the extra-MSA stack.

### Stage 4 continued: the attention modules, and a dead parameter in AF2

**264 cases, all passing.** Added `attentionModuleBwd` (AF2's whole `Attention`: the q/k/v projections with its
`key_dim^-0.5`, the sigmoid gate, the output projection) and `biasedSelfAttentionBwd`, which serves
**both** `MSARowAttentionWithPairBias` and `TriangleAttention` because the two are the same shape:

| | MSARowAttentionWithPairBias | TriangleAttention |
| --- | --- | --- |
| LayerNorms | two (`query_norm`, `feat_2d_norm`) | one (`query_norm`) |
| the bias comes from | a DIFFERENT tensor (the pair) | the SAME normalised tensor it attends over |
| so the bias path's cotangent | goes back through its own LayerNorm | **accumulates into the attention's own** |

Self-attention is the case that needed care: AF2 passes the same activation as `q_data` and `m_data` everywhere, so
**five** paths - Wq, Wk, Wv, the gate and (for the triangle) the bias linear - all accumulate into one cotangent.
`attentionModuleBwd` takes one pointer for both and tracks which path writes first; passing two distinct buffers
gives the cross-attention case, and both are tested.

#### 🔴 feat_2d_norm's OFFSET has an identically zero gradient, and it is a dead parameter in AF2

Found by the central-difference check, which reported a relative error of **7.7e+02** on `dbOffset` while every other
gradient in the same case passed. Not a bug - the gradient is **exactly zero**, and a relative measure cannot judge
that:

- `feat_2d_norm`'s output is used for **nothing but** the non-batched bias (`modules.py:539-552`).
- Shifting its offset by d shifts `biasN[r][c]` by d for every row, so `nb[h][q][k]` shifts by `sum_c d_c W[c][h]` -
  a **constant for a given head, across every (q, k)**.
- A constant added to every logit of a softmax row cancels. So the output does not move at all.

It falls out of the kernel arithmetic too: `sum_k ds[k] = sum_k p dp - D` and `D = sum_k p dp` by definition, so a
softmax row's `ds` sums to zero, and the column sum that `doffset` is sums those zeros.

Consequences worth keeping: **that parameter never moves during training and cannot affect a prediction** - AF2
carries it regardless, so a port must still read it from the checkpoint, and must not "fix" it. Its SCALE is a real
parameter (`dbScale` passes at 3.4e-7); only the offset is dead. And the test now reports such a case as
`ok (zero gradient)` on an absolute floor rather than a relative one, which is the honest way to check it.

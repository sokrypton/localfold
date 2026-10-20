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

0. **The whole EvoformerIteration composes** - all eight residual modules, both tracks. **DONE**, see below.
1. **The primitives** - LayerNorm, a linear layer's two GEMMs and its bias. **DONE**, see below.
2. **The pair stack's cheap halves** - the transitions, the outer-product mean, the gates. **DONE**, see below.
3. **Triangle multiplication**, outgoing and incoming. **DONE**, see below.
4. **Attention - the tier is COMPLETE.** The core backward, the Attention module, the biased self-attention both
   `MSARowAttentionWithPairBias` and `TriangleAttention` are, `MSAColumnAttention`, and
   `MSAColumnGlobalAttention`. See below.
5. **The structure module and the heads.** The five differentiable boundary arrays that come from the PAIR alone -
   the distogram, and pae/ptm/iptm/iptm_per_residue - are **DONE**, see below. What is left is the STRUCTURE
   MODULE: IPA, quaternion rigids, frames and side chains, and with it the two remaining arrays (`plddt` and
   `experimentally_resolved_ca`, both of which read the single representation it produces) and `atoms`.
   **This is the next thing to do, and it is the largest piece left.**
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

**Confirmed against AF2 itself, in float64** - `metal/design/af2-grad-oracle.py`, which differentiates BindCraft2's
own haiku `MSARowAttentionWithPairBias` with `jax.grad`. Three independent implementations now agree it is zero:
the Metal kernel, the central difference on a double host forward, and the reference's own autodiff.

| parameter | max abs gradient (float64) |
| --- | --- |
| `feat_2d_norm/offset` | **1.98e-16** (machine epsilon) |
| `feat_2d_norm/scale` | 2.61e-01 |
| the next smallest live gradient (`attention/query_w`) | 3.93e-02 |

Two traps in confirming it, both of which give a confident wrong answer and are now guarded in the script:

- **`zero_init` is TRUE in AF2's config**, which makes `output_w` zero, so the loss sees only `output_b` and EVERY
  other gradient is trivially zero - which looks exactly like the finding. The first run of this check said every
  parameter was dead. Set it False, and randomise the parameters besides, since haiku always initialises
  `gating_w` to zero.
- **float32 roundoff of this cancellation sits near 1e-8** - small, but not obviously zero, and not obviously
  distinguishable from a real but tiny gradient. In float64 it drops to ~1e-16 while the live gradients do not move
  at all, and that is what makes it a proof rather than a hint.

Consequences worth keeping: **that parameter never moves during training and cannot affect a prediction** - AF2
carries it regardless, so a port must still read it from the checkpoint, and must not "fix" it. Its SCALE is a real
parameter (`dbScale` passes at 3.4e-7); only the offset is dead. And the test now reports such a case as
`ok (zero gradient)` on an absolute floor rather than a relative one, which is the honest way to check it.

### The memory pass: 830 MB of scratch a block down to 436, and a bug the proof caught

`metal/design/localfold-design --mem` reports what each backward holds at AF2's real widths, and `--mem-block [N]`
what a whole block holds with nothing released between modules - which is what an Evoformer block actually does.

**Two separate problems, and `shareScratch` only addresses one of them.** Its granularity is the FAMILY: it aliases
the buffers of modules that run at different times, and can do nothing about buffers co-existing inside one module.
So both were done.

#### Within a module: buffers whose predecessors are provably finished

A pair buffer is 18 MB at N=192, and two modules were each holding three more than they needed.

| | before | after |
| --- | --- | --- |
| `triangleMultiplicationBwd` | 378.3 MB | **324.3 MB** |
| `biasedSelfAttentionBwd` (triangle attn) | 237.8 MB | **183.8 MB** |
| `transitionBwd` (pair track) | 180.3 MB | 180.3 MB |
| `outerProductMeanBwd` | 33.8 MB | 33.8 MB |

In the triangle, `q`, `prow` and `gPre` are spent by the time the product's cotangent, its transpose and the final
LayerNorm's input are wanted, so those three are named rather than allocated. In the attention module, `gpre` and
`dgpre` are read for the last time by the gate, and one of `dwa`/`dwa2` carries the gate stage's spent cotangent.
Dispatches run in issue order on one queue, so a later write cannot race an earlier read - that is what makes this
safe rather than hopeful.

🔴 **AND THE OBVIOUS WAY TO DO IT IS WRONG.** `dqs` was first bound to `dwa2` at declaration, which is *before* the
`std::swap` that decides which of the two carries the live cotangent - so after the swap `dqs` aliased the very
buffer `attentionBwd` READS. The dq pass overwrote its own input and the dk/dv and dBias passes then read rubbish.
**21 cases failed and named exactly the gradients that depend on it** (dWk, dWv, dMData, dPairBias, while dWq and
dWo passed). The swap is gone now and the dead buffer is bound after the branch. A buffer reuse must be bound where
the lifetime is *known*, not where the buffer is declared.

#### Across modules: `designShareScratch()`

Families mirroring `metal/af2/main.mm`'s for the forward - `dtr`, `dopm`, `dtri`, `dbattn`, `dattnm`, `dattn`,
`dln`. The nesting is the part that matters: `attentionModuleBwd` runs INSIDE `biasedSelfAttentionBwd` and
`attentionBwd` inside that, and a LayerNorm's row statistics are live inside every module, so those are declared as
living around each other's turn and are never aliased together.

One block's backward - transition, outer product, two triangle multiplications, two triangle attentions:

| N | pair buffer | sharing off | sharing on | saved |
| --- | --- | --- | --- | --- |
| 192 | 18.0 MB | 721.5 MB | 721.5 MB | **nothing** |
| 192 (floor at 1 MB) | 18.0 MB | 721.5 MB | **436.5 MB** | 39.5% |
| 320 | 52.4 MB | 1986.8 MB | **1236.8 MB** | 37.7% |

🔴 **THE FLOOR IS WHY N=192 GAINS NOTHING.** `shareScratch` only aliases buffers of **32 MB or more**
(`LOCALFOLD_SHARED_MIN`, core.mm:87), and a 128-channel pair tensor does not reach that until about **N=256**. So
the mechanism engages by itself exactly where the pressure is - at N=320 it already saves 38% - and is inert below
that. Lowering the floor to 1 MB buys another 39.5% at N=192; it is a global knob, not a per-tree setting, so it is
left alone rather than set from library code. If a small-N design loop ever needs it, that is the lever, and the
scribble proof below passes at that setting too.

#### Verified, not assumed

`LOCALFOLD_SCRIBBLE=1` NaN-fills every slot but those of a family nested with either side of each change of family,
and the results must not move (docs/CUDA-TO-METAL.md). All 264 cases pass at **every** floor tried - 1 byte, 256,
4096, 65536 and the 32 MB default - and with `LOCALFOLD_SHARED_SCRATCH=0`, the control. At a floor of 1 byte every
buffer in the tree is aliased and scribbled, so that run is the one with teeth.

#### Still on the table, and WHICH of it actually matters

🔴 **THE TRIANGLE MULTIPLICATION IS THE CEILING, NOT THE TRANSITION.** An earlier version of this section named the
transition as the next thing to fix, and that was wrong in a way worth recording. The per-module numbers are
`tri 324 MB > battn 184 ~= transition 180 > opm 34`, and all four scale as N^2, so the ordering is
size-independent. With sharing on, a block holds the MAXIMUM of its modules plus its own running cotangents - so
**chunking the transition cannot lower a block's peak at all** while the triangle holds nearly twice as much. It
would be real work for no change in the number that matters.

So, in order:

1. **The triangle multiplication's twelve simultaneously-live pair buffers.** Cutting these needs its product
   **tiled over rows** - a different kernel structure, not a cleverer alias, because the twelve really are all live
   around the batched GEMM. This is the one that moves the peak.
2. **The block's own temporaries** (`tmpMsa`, `tmpPair`, `tmpPair2`) are live across every module's turn by
   construction, so `shareScratch` can never alias them - they are declared nested with everything. Three
   pair-sized buffers is the price of composing residual modules this way.
3. **The transition's 180 MB**, by chunking over rows as AF2 does (`mapping.inference_subbatch`). Worth doing when
   the triangle is no longer dominating, and not before. Note it also needs accumulating weight gradients across
   chunks, so `linearBwd`, `columnSum` and `layerNormBwd`'s parameter outputs would each need an accumulate mode -
   which is why it is not the cheap win it looks like.

### MSAColumnAttention, and the per-column orientation

**276 cases.** `msaColumnAttentionBwd`, plus `lf_swap01` (swap the two leading axes of an `[A][B][C]` tensor).

AF2's column attention is **the same module as the row one with the first two axes swapped and no pair bias**
(`modules.py:595`, `:614`): the MSA is attended down its columns, so a residue is a batch and the sequences are the
queries. So it is served by transposing in, calling `biasedSelfAttentionBwd` with `feat2dW` null, and transposing
the cotangent back - which is the same swap read the other way round. `biasedSelfAttentionBwd` now drops its whole
bias path when `feat2dW` is null rather than taking a separate code path.

The same `lf_swap01` serves the triangle attention's `per_column` orientation, which AF2 also implements as a swap
before and after (`modules.py:703`, `:731`).

**A trap for whoever writes the next test here:** the first run of this one segfaulted, and the cause was the HOST
REFERENCE, not a kernel - `biasedForward` indexed a `feat2dW` that the no-bias case had passed as an empty vector.
Worth noting because the output looked like a crash deep in an earlier module: stdout is block-buffered when
redirected, so the visible output ended mid-line in `attnmod` while the fault was two modules later. `stdbuf -oL`
is how to find out where a crash in this binary actually is.

### MSAColumnGlobalAttention: the tier is complete

**296 cases.** All 19 gradients agree with central difference at **5.1e-8 to 6.3e-7**, first run. With this the
whole Evoformer's attention tier has a backward: the core, the Attention module, both biased self-attentions, the
column attention and now the global one.

This is the one attention in AF2 that is a **different shape**, not a different configuration
(`modules.py:432-497`):

- **One query a batch**, and it is a *masked mean* over the sequences - `qavg[b][a] = sum_k m[b][k] q[b][k][a] /
  (sum_k m[b][k] + 1e-10)`. So its backward is a SCATTER: every key receives the mean's cotangent weighted by its
  own share of the mask.
- **k and v have no head axis** (`Wk`, `Wv` are `[Cm][D]`, shared across heads), so `dk` and `dv` sum over the
  heads where every other attention here keeps them separate.
- **The gate broadcasts that single query back over every position** (`weighted_avg[:, None] * gate_values`), which
  makes the gate's backward a reduction over the position axis rather than elementwise - a thread per `(b, h, c)`
  walking the positions, writing each one's gate cotangent and summing the activation's.

🔴 **AND IT STORES ITS ATTENTION WEIGHTS, WHICH EVERY OTHER ATTENTION HERE REFUSES TO DO.** That looks like an
inconsistency and is not: the weights are `[B][H][K]` with **no query axis**, so they are kilobytes rather than the
O(N^3) a triangle attention's would be. The softmax backward reads them instead of recomputing a score. The policy
is the same everywhere - never materialise O(N^3) - and the absent axis is what changes the answer.

Gating is required rather than optional here, because AF2's ungated branch has a different output shape
(`modules.py:495` returns `[B][1][O]` broadcast, not `[B][Q][O]`), and its config has gating on.

### The whole EvoformerIteration, composed and checked

**308 cases.** `evoformerIterationBwd` runs all eight of a block's modules in reverse, threading both tracks'
cotangents. Checked against central difference on a double forward of the WHOLE block: both input cotangents and
parameter gradients sampled from four different modules (the MSA row attention, a triangle multiplication, the pair
transition and the outer product), for both orderings of the outer product mean. **5.5e-8 to 6.2e-7.**

Sampling parameters from four modules is deliberate: it is what catches a weight wired to the wrong module, which
threading bugs look like and which `dMsa`/`dPair` alone would often hide.

**Every module is a residual**, because `dropout_wrapper` is `new_act = output_act + dropout(module(input_act))`
(`modules.py:46-76`) and a design trajectory runs with dropout off. So a cotangent BOTH passes straight through and
goes through the module, summed - nothing is a chain of overwrites. Two consequences:

- **The MSA row attention feeds BOTH cotangents**, being the one module that reads both tracks; and the outer
  product mean writes the pair from the MSA, so the pair's cotangent passes it untouched while the MSA's grows.
- The accumulation is a temporary plus one `lf_accum` pass a module, rather than an `accumulate` flag threaded
  through six modules' final LayerNorm. One buffer, one cheap pass; the blast radius decided it.

🔴 **AND THE BLOCK NEEDS A SAVE SET, WHICH IS WHY IT IS AN ARGUMENT.** Each module's backward recomputes its own
forward, but from the activation it was *given* - and in a block that is the output of everything before it. There
are only two honest ways to have those: duplicate AF2's whole forward in this tree, or have the forward hand them
over. `EvoBlockSaved` is the second, ten activations, one per module call. This is the same contract BindCraft2's
`native/GLUE_PORT.md` describes - "carrying the trunk's save set and pout/mout across the boundary as XLA buffers" -
and it reached the same shape independently.

**So the piece that is now missing is on the FORWARD side:** `metal/af2`'s Evoformer must be able to emit these ten
activations per block. That is wiring, not new arithmetic, but it is a real task and it is what stands between this
tree and a block that can actually run forwards and backwards. The other known forward-side gap is the same kind of
thing: `metal/core`'s tuned `attention()` does not return the logsumexp the attention backward needs.

Also worth knowing: the ending node's triangle attention is run on the **transposed** pair, as AF2 does
(`modules.py:703`, `:731`), and the N x N transpose is its own inverse, so the same `lf_swap01` serves both
directions.

### The pair-side heads and the metrics a loss actually reads

**322 cases.** `distogramHeadBwd`, `paeHeadBwd` and `paeMetricsBwd`. **Five of the seven differentiable arrays in
`boundary.py` come from the pair alone**, so they need no structure module: `distogram`, `pae`, `ptm`, `iptm` and
`iptm_per_residue`. ipTM is BindCraft's primary binder objective, so this is the part of the boundary a design loss
leans on hardest.

The heads are plain linears - the distogram's symmetrisation `logits = half + half^T` is self-adjoint, so its
backward is the same operation and the same kernel serves both directions.

The metrics are where the arithmetic is (`confidence.py:113-180`), and three things about the derivative are worth
writing down because they are not obvious from the forward:

- **Only `probs` carries a gradient.** The bin centres, `d0` and the normalisation are all constant with respect to
  the logits - `d0` depends on the residue weights' SUM, not on the logits at all - so everything else is a fixed
  coefficient. That collapses what looks like a long chain into one softmax backward per pair.
- 🔴 **pTM and ipTM are a MAX over the residues**, `(per_alignment * residue_weights).max()`, so their gradient is a
  **one-hot at the argmax**, not a spread. The kernel picks the FIRST maximum, which is what `jnp.max`'s gradient
  does; with ties the two would disagree, and random data has none.
- **pTM and ipTM differ only in their pair mask** (everything, against `asym_id[i] != asym_id[j]`), so one set of
  kernels runs twice and `lf_pae_dlogits` sums both paths into the same logits.

`iptm_per_residue` is the `per_alignment` vector itself, so it costs nothing extra and its cotangent enters the same
`dPerAlign` accumulator as ipTM's one-hot.

### The structure module, stage by stage

The second half of the fold, and where every remaining boundary array of `bindcraft/af/boundary.py` comes from.
The forward is `metal/af2/structure.mm` and the tail of `metal/af2/af2.metal`; the backward is
`metal/design/structure.metal`, `structgrad.h` and `structgrad.mm`. **213 new cases, 535 in all.**

#### 🔴 BindCraft2 does not cut the rigid chain, and upstream AF2 does

`bindcraft/af/alphafold/model/folding.py:379` is

```python
    # affine = affine.apply_rotation_tensor_fn(jax.lax.stop_gradient)
```

**commented out.** Upstream AF2 ends each fold iteration by stopping the gradient on the rotation; BindCraft2 does
not. So the rigid is a RECURRENT STATE whose cotangent flows back through all eight iterations, and no iteration
may treat it as a constant input. That single line decides the shape of this whole module: `dRig` is read as the
output's cotangent and rewritten as the input's, once per iteration, and inside one iteration it takes a
contribution from four places - each of the three point projections, and again where the IPA's point block returns
to the local frame.

#### What is there

| | |
| --- | --- |
| rigids | the backbone update (Alg. 23, quaternion through its normalisation), the point projections into the global frame |
| IPA core | the attention weights (scalar dot, point distance, pair bias, the softplus'd point weight) and the outputs (scalar, point-with-norm, pair) |
| IPA module | all six projections, both cores and the residual output projection - `invariantPointAttentionBwd` |
| the iteration | `foldIterationBwd`: the IPA, both LayerNorms, the three-layer transition with its residual, quat_rigid, the backbone update |
| side chains | `sidechainMlpBwd` (two input projections, two residual blocks, the angle head) and `sidechainGeometryBwd` (angle normalisation, eight group frames, the chain, atom14, atom37) |

#### Three things the shapes forced

**A thread a residue, not a thread a point.** Several (head, point) pairs read one residue's rigid, so a per-point
kernel would need a float atomic to accumulate `dRig` - and `AGENTS.md` wants the differential test DETERMINISTIC,
which an atomic is not. The side-chain geometry has a second reason on top: the frame chain is sequential, so its
backward walks it in reverse and splitting a residue would mean a reduction per link.

**Three passes for the attention weights**, exactly as the flash-shaped attention above: `dqs` and `dqp` accumulate
over keys while `dks` and `dkp` accumulate over queries. `dLogit` is materialised once - the same size as `attn`,
which the forward already holds - so both passes read it rather than recomputing the softmax, and the pass order
cannot change the answer.

**Everything accumulates except `dAct`, and the caller zeroes.** One set of IPA weights runs eight times and one
pair bias feeds all eight. `linearBwd` and `columnSum` gained `accumulateW`, `layerNormBwd` gained
`accumulateParams`. A flag per destination would have been four chances to get the rigid wrong.

#### 🔴 Two MORE dead parameters in AF2, and one is this port's alone

The IPA composition test reported `dbk` as an identically zero gradient on all three shapes. Not a bug - the same
mechanism as `feat_2d_norm`'s offset, and chasing it found a third. **Anything that can only shift a whole softmax
row by a constant is dead.** Held to `jax.grad` of AF2's own `InvariantPointAttention` in float64:

| parameter | max abs gradient | |
| --- | ---: | --- |
| `attention_2d/bias` | **7.49e-16** | the IPA's pair bias is per-head and constant across (query, key) |
| `kv_scalar/bias`, the K half | **2.22e-16** | the key's bias enters only through `sum_c qs[q][c] ks[k][c]`, where its contribution does not depend on k |
| `attention_2d/weights` | 2.12e+00 | the same linear, live |
| `kv_scalar/bias`, the V half | 5.54e+00 | the same tensor, live |
| the smallest LIVE gradient in the module | 9.59e-01 | `q_scalar/bias` |

The second is a property of **this port's parameterisation** and is why nobody would see it upstream: the reference
fuses the key and value scalar projections into one `kv_scalar`, whose gradient is large and unremarkable because
the value half is live. This port SPLITS that tensor (`cuda/featurise/af2_weights.h`), so
`k_scalar_projection/bias` is dead outright. **Monomer only** - the multimer checkpoint carries no scalar bias at
all, which that file already says.

`metal/design/af2-grad-oracle.py` runs all three checks from one command. The design suite holds the first of the
two directly: the pair bias comes from a linear over `act2d`, so that linear's bias gradient is the column sum of
`db2d` over every (i, j), asserted zero on all three shapes.

#### 🔴 Two test defects found here, both of the family "a check that cannot fail"

The dead-parameter assertion built a vector from `download(...).begin()` and `download(...).end()` - two SEPARATE
temporaries, so an invalid range. It read as three passes because the two allocations happened to land on one
address, then threw `std::length_error` the moment anything else allocated first. CLAUDE.md's own rule one layer
down: this one could not even be relied on to crash.

And the fold iteration's three transition weight matrices sit behind a relu, so about half of any sampled element
has an exactly zero derivative on both sides - at which point `reportGrad` says `both ~0, ok` however wrong the
kernel is. **Three probes passed that way.** The probe now draws up to eight times its budget and keeps LIVE
entries by preference, falling back only if a tensor really is dead.

#### Memory: the structure module is not the ceiling, measured

`design.ipa.` is declared to `designShareScratch()` - it runs AFTER the Evoformer, so it may alias with all of it,
and the LayerNorm statistics are nested inside its turn. `metal/design/localfold-design --mem` at N=192:

```
  transitionBwd (pair track)                     scratch    180.3 MB
  outerProductMeanBwd                            scratch     33.8 MB
  triangleMultiplicationBwd                      scratch    324.3 MB
  biasedSelfAttentionBwd (triangle attn)         scratch    183.8 MB
  invariantPointAttentionBwd (structure module)  scratch      6.3 MB
```

**6.3 MB against the triangle's 324.** Its largest own buffer is `dLogit` at `[L][H][L]`, 1.7 MB here. So the
structure module changes nothing about the memory ceiling, which stays exactly where the previous session's
correction left it: the triangle multiplication, tiled over rows.

## Where this stands, and what the next session should pick up

**535 differential cases, all passing**, at every sharing floor and with sharing off. `bash metal/build.sh design &&
metal/design/localfold-design`. The core selftest and the three shipping ports are untouched and still build and
pass.

### Done: the Evoformer's reverse mode AND the structure module's

| | |
| --- | --- |
| primitives | LayerNorm, linear (two GEMMs + bias), column sums, relu, the sigmoid gate |
| pair modules | `Transition`, `OuterProductMean`, `TriangleMultiplication` (both equations) |
| attentions | the flash-shaped core, the `Attention` module, `MSARowAttentionWithPairBias`, `TriangleAttention`, `MSAColumnAttention`, `MSAColumnGlobalAttention` |
| the block | `evoformerIterationBwd` - all eight residual modules, both tracks, both OPM orderings |
| heads | the distogram, the PAE head, and `pae`/`ptm`/`iptm`/`iptm_per_residue` |
| the structure module | the rigids, the IPA core, the whole IPA, `foldIterationBwd`, and MultiRigidSidechain's MLP and geometry |
| memory | within-module reuse and `designShareScratch()`, proved by `LOCALFOLD_SCRIBBLE` |

Which leaves, of the nine boundary arrays, `atoms` reachable (`sidechainGeometryBwd` takes `dPos14` and `dPos37`)
and only `plddt` and `experimentally_resolved_ca` unwritten - both are heads off the single representation, and
both are small next to anything above.

### Next, in the order I would do it

1. **🔴 THE FORWARD SIDE, WHICH IS NOW THE ONLY THING BETWEEN THIS AND A FOLD THAT RUNS BOTH WAYS.** All of it is
   in shipping code, which is why it was left for review rather than done unsupervised. Four items, the last two
   new this session:
   - **`metal/af2`'s Evoformer must emit the save set** - the ten activations per block in `EvoBlockSaved`.
   - **`metal/core`'s tuned `attention()` does not return the logsumexp** the attention backward needs.
     `attentionFwdSaved` here is a correct but untuned stand-in. Saving `L` from the real flash forward is the
     cheap, right fix; writing a second tuned attention is not.
   - **`metal/af2`'s structure module must emit `FoldIterSaved` per iteration** - and keep the eight iterations'
     rigids rather than updating one in place, since the gradient flows through all of them.
   - **and it must keep `projQ`, `projK` and `projV` APART.** The forward reuses one piece of scratch for all
     three point projections; `dRig` needs each one's local coordinates.
2. **The two remaining heads**: `plddt` and `experimentally_resolved_ca`.
3. **The triangle multiplication's memory**, tiled over rows - see the memory section for why this one and not the
   transition.
4. **The design loop**: the losses, the optimiser, the schedules. `bindcraft/af/boundary.py` is the contract to
   hold to, and six of its seven differentiable arrays have a backward here now.

### How to check anything added here

Three references, and the third is the one that settles arguments about AF2 rather than about this port:

```
metal/design/localfold-design                                   # double analytic + central difference, 535 cases
LOCALFOLD_SCRIBBLE=1 LOCALFOLD_SHARED_MIN=1 .../localfold-design   # every buffer aliased and NaN-scribbled
BC2=~/BindCraft2 .venv-mtl/bin/python metal/design/af2-grad-oracle.py   # jax.grad of AF2 itself, in float64
```

Never widen a bound to make a case pass. A gradient that is analytically zero is reported as `ok (zero gradient)`
on an absolute floor - see the `feat_2d_norm` finding for why that distinction had to exist - and the structure module's two test
defects for why a probe must also be checked to be capable of failing at all.

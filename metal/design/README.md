# metal/design: the backward

Every other backend here folds - forward only. Binder design (BindCraft2's trajectory) is gradient descent on a
sequence through the folding network, so it needs the backward of every stage, and no backend in this repository has
one. This tree is that, built up a stage at a time.

```
bash metal/build.sh design && metal/design/localfold-design
```

**Read `docs/BINDCRAFT2.md` before adding to this.** It has why the port takes this shape rather than attaching
kernels to JAX (measured: the Metal PJRT plugin cannot see an external FFI handler), what BindCraft2's own CUDA does,
the staging, and the two decisions already made here that a later kernel should follow.

## What is here

| | |
| --- | --- |
| `kernels.h` | the kernels' argument structs, read by C++ and by Metal |
| `design.metal` | `lf_ln_stats`, `lf_ln_bwd`, `lf_col_sums`, `lf_relu_bwd`, `lf_gate_bwd`, `lf_mul_sigmoid`, `lf_tri_pack`/`_unpack`, the seven `lf_opm_*`, the five `lf_attn_*` |
| `grad.h` / `grad.mm` | the host API: `layerNormBwd`, `linearBwd`, `columnSum`, `reluBwd`, `gateBwd`, `transitionBwd`, `outerProductMeanBwd`, `triangleMultiplicationBwd`, `attentionFwdSaved`/`attentionBwd`, `attentionModuleBwd`, `biasedSelfAttentionBwd` |
| `main.mm` | the differential test - every backward against two independent references, in double |

A linear layer's backward needs no kernel of its own: `dX = dY W^T` and `dW = X^T dY` are `metal/core`'s GEMM with
one operand read transposed, which it already serves.

Whole modules so far: **`Transition`** (Alg. 9 / 15), **`OuterProductMean`** (Alg. 10) and
**`TriangleMultiplication`** (Alg. 11 / 12, both equations). All RECOMPUTE their forward rather than taking a save
set, which is what AF2 runs under anyway (`use_remat=True`). **322 cases pass.**

**What has a backward now:** every module of the AF2 Evoformer, the whole block composed
(`evoformerIterationBwd`), and the pair-side heads with the four metrics a design loss reads off the PAE head
(ipTM among them). **322 cases pass.** `docs/BINDCRAFT2.md` has the staging, the measurements and the findings;
read its last section first - it says what is done, what is next, and the two forward-side gaps that block a block
from running both ways.

Plus the **attention backward** (`attentionBwd`), which is flash-shaped because it has to be: the weights are
[B][H][Q][K] and a triangle attention's b runs over residues, so storing them is the O(N^3) that AGENTS.md forbids.
Scores are recomputed from a saved logsumexp, in three passes - see `docs/BINDCRAFT2.md` for which gradient
accumulates over what, which is what decides the passes.

...and the modules around that core: `attentionModuleBwd` (AF2's whole `Attention`) and
`biasedSelfAttentionBwd`, which serves **both** `MSARowAttentionWithPairBias` and `TriangleAttention` - the two are
the same shape, differing only in whether the pair bias comes from the tensor being attended over.

Still to write: **the structure module** (IPA, rigids, frames, side chains - the largest piece left, and where
`atoms`, `plddt` and `experimentally_resolved_ca` come from) and the design loop.

One finding from here that a reader should not re-derive: **`feat_2d_norm`'s offset has an identically zero
gradient** and is a dead parameter in AF2. See `docs/BINDCRAFT2.md`.

## Memory

`localfold-design --mem` reports what each backward holds at AF2's real widths; `--mem-block [N]` what a whole
block holds with nothing released between modules. A driver must call **`designShareScratch()`** once, or every
module's scratch is its own and a block holds the sum rather than the maximum.

Two levers, and they are different: `shareScratch` aliases across MODULES (its granularity is the family, so it can
do nothing inside one), while buffers that co-exist inside a module are reused by hand where a lifetime is provably
over. Both are done; `docs/BINDCRAFT2.md` has the numbers, the 32 MB floor that makes the sharing inert below about
N=256, and the aliasing bug the proof caught.

Verify any change here with the mechanism's own proof, which must stay `all ok`:

```
LOCALFOLD_SCRIBBLE=1 LOCALFOLD_SHARED_MIN=1 metal/design/localfold-design   # every buffer aliased and NaN-scribbled
LOCALFOLD_SHARED_SCRATCH=0 metal/design/localfold-design                    # the control
```

## The testing rule this tree follows

AGENTS.md: *every GPU kernel must have a deterministic differential test against an independent reference*. For a
backward that means **two** references, because one of them cannot see the mistake that matters most:

- a **double-precision analytic reference** catches a kernel that does not compute what was derived;
- a **central difference against the forward alone** catches a derivation that is wrong. `S(x) = sum(dy *
  forward(x))`, and `dS/dx` must be the cotangent the kernel produced. The analytic reference cannot catch this,
  because it repeats the derivation.

A third is available and should be used for anything the first two leave ambiguous, above all the attentions: the
JAX environment in `docs/BINDCRAFT2.md` runs BindCraft2's own AF2 forward and backward on this machine, so a kernel
can be held to `jax.grad` of the exact haiku module it replaces.

Never widen a bound to make a case pass.

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
| `design.metal` | `lf_ln_stats`, `lf_ln_bwd`, `lf_col_sums`, `lf_relu_bwd`, `lf_gate_bwd`, `lf_mul_sigmoid`, `lf_tri_pack`/`_unpack`, the seven `lf_opm_*`, the five `lf_attn_*`, the seven `lf_gattn_*`, the five `lf_pae_*` |
| `grad.h` / `grad.mm` | the Evoformer's host API: `layerNormBwd`, `linearBwd`, `columnSum`, `reluBwd`, `gateBwd`, `transitionBwd`, `outerProductMeanBwd`, `triangleMultiplicationBwd`, `attentionFwdSaved`/`attentionBwd`, `attentionModuleBwd`, `biasedSelfAttentionBwd`, `msaColumnAttentionBwd`, `msaColumnGlobalAttentionBwd`, `evoformerIterationBwd`, the heads and `paeMetricsBwd` |
| `structure.metal` | the structure module's kernels: `lf_rigid_upd_bwd`, `lf_points_global_bwd`, the five `lf_ipa_*` weight passes, the four `lf_ipa_out_*`, `lf_sidechains_bwd` |
| `structgrad.h` / `structgrad.mm` | its host API: `rigidUpdateBwd`, `pointsGlobalBwd`, `ipaWeightsBwd`, `ipaOutputsBwd`, `invariantPointAttentionBwd`, `foldIterationBwd`, `sidechainMlpBwd`, `sidechainGeometryBwd` |
| `main.mm` | the differential test - every backward against two independent references, in double |

A linear layer's backward needs no kernel of its own: `dX = dY W^T` and `dW = X^T dY` are `metal/core`'s GEMM with
one operand read transposed, which it already serves.

The Evoformer's pair modules - **`Transition`** (Alg. 9 / 15), **`OuterProductMean`** (Alg. 10) and
**`TriangleMultiplication`** (Alg. 11 / 12, both equations) - RECOMPUTE their forward rather than taking a save
set, which is what AF2 runs under anyway (`use_remat=True`). The structure module cannot: its save set is a
`FoldIterSaved` per iteration, because recomputing one iteration means recomputing the rigid chain that led to it.

**What has a backward now:** every module of the AF2 Evoformer, the whole block composed
(`evoformerIterationBwd`), the pair-side heads with the four metrics a design loss reads off the PAE head (ipTM
among them), and **the whole structure module** - the rigids, InvariantPointAttention, the fold iteration and
MultiRigidSidechain. **535 cases pass.** `docs/BINDCRAFT2.md` has the staging, the measurements and the findings;
read its last section first - it says what is done, what is next, and the four forward-side gaps that are now the
only thing between this and a fold that runs both ways.

🔴 **The rigid is a RECURRENT STATE here, because BindCraft2 commented out AF2's `stop_gradient` on the rotation**
(`folding.py:379`). `foldIterationBwd` reads `dRig` as the output's cotangent and rewrites it as the input's; a
caller must zero it once and never treat it as a per-iteration input.

Plus the **attention backward** (`attentionBwd`), which is flash-shaped because it has to be: the weights are
[B][H][Q][K] and a triangle attention's b runs over residues, so storing them is the O(N^3) that AGENTS.md forbids.
Scores are recomputed from a saved logsumexp, in three passes - see `docs/BINDCRAFT2.md` for which gradient
accumulates over what, which is what decides the passes.

...and the modules around that core: `attentionModuleBwd` (AF2's whole `Attention`) and
`biasedSelfAttentionBwd`, which serves **both** `MSARowAttentionWithPairBias` and `TriangleAttention` - the two are
the same shape, differing only in whether the pair bias comes from the tensor being attended over.

Still to write: the two heads off the single representation (`plddt`, `experimentally_resolved_ca`) and the design
loop - plus the forward-side wiring, which is in shipping code and wants review.

Three findings from here a reader should not re-derive: **three DEAD PARAMETERS in AF2**, all the same mechanism -
anything that can only shift a whole softmax row by a constant. `feat_2d_norm`'s offset, the IPA's
`attention_2d/bias`, and the key half of `kv_scalar/bias` (which only this port's split parameterisation can see).
All three are held to `jax.grad` of AF2's own module in float64 by `af2-grad-oracle.py`. See `docs/BINDCRAFT2.md`.

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

Never widen a bound to make a case pass. **And check that a probe is capable of failing at all** - two checks here
passed while proving nothing (one built an iterator range from two separate temporaries; one sampled only
relu-killed entries, where both sides are exactly zero and "both ~0, ok" is unconditional). `docs/BINDCRAFT2.md`
has both.

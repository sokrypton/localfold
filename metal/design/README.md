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
| `design.metal` | `lf_ln_stats`, `lf_ln_bwd`, `lf_col_sums`, `lf_relu_bwd`, `lf_gate_bwd`, the seven `lf_opm_*` |
| `grad.h` / `grad.mm` | the host API: `layerNormBwd`, `linearBwd`, `columnSum`, `reluBwd`, `gateBwd`, `transitionBwd`, `outerProductMeanBwd` |
| `main.mm` | the differential test - every backward against two independent references, in double |

A linear layer's backward needs no kernel of its own: `dX = dY W^T` and `dW = X^T dY` are `metal/core`'s GEMM with
one operand read transposed, which it already serves.

Whole modules so far: **`Transition`** (Alg. 9 / 15) and **`OuterProductMean`** (Alg. 10). Both RECOMPUTE their
forward rather than taking a save set, which is what AF2 runs under anyway (`use_remat=True`). 131 cases pass.

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

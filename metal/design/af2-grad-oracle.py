"""Hold a claim about AF2's OWN gradients to AF2 itself, with jax.grad.

metal/design derives every backward here twice - a double-precision analytic reference and a central difference
against the forward - but neither can settle a question about what the REFERENCE model does. This does: it runs
BindCraft2's haiku modules and differentiates them, in float64, on this machine.

What it checks today: that `feat_2d_norm`'s OFFSET is a dead parameter in MSARowAttentionWithPairBias - an
identically zero gradient, because that LayerNorm feeds nothing but the non-batched bias, so shifting its offset
shifts every logit of a softmax row by the same constant, and that cancels. See docs/BINDCRAFT2.md.

SETUP (a few minutes, no weights needed - random init is enough, because the question is the derivative):

    python3.12 -m venv .venv-mtl
    .venv-mtl/bin/pip install jax==0.11.2 jaxlib==0.11.2 dm-haiku optax ml_collections absl-py biotite
    git clone --depth=1 https://github.com/sokrypton/BindCraft2
    BC2=$PWD/BindCraft2 .venv-mtl/bin/python metal/design/af2-grad-oracle.py

Add `metal-pjrt-plugin` and `JAX_PLATFORMS=mtl` to run it on the GPU instead; it needs no GPU for this, and float64
is CPU-only on that plugin anyway.

TWO TRAPS, both of which produce a confident wrong answer:
  * `zero_init` is TRUE in AF2's config, which makes `output_w` zero, so the loss sees only `output_b` and EVERY
    other gradient is trivially zero - which looks exactly like the finding. It is set False here, and every
    parameter is randomised besides (haiku always initialises `gating_w` to zero).
  * float32 roundoff of this cancellation sits near 1e-8, which is small but not obviously zero. In float64 it drops
    to ~1e-16 while the live gradients do not move at all, and that is what makes it a proof rather than a hint.
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)   # the clincher: a quantity that is EXACTLY zero drops to ~1e-16 here,
                                            # where float32 roundoff of the same cancellation sits near 1e-8
import haiku as hk                          # noqa: E402
import jax.numpy as jnp                     # noqa: E402
import numpy as np                          # noqa: E402

# the BindCraft2 checkout, so `bindcraft.af.alphafold` imports
sys.path.insert(0, os.environ.get('BC2', os.path.expanduser('~/BindCraft2')))

from bindcraft.af.alphafold.model import modules  # noqa: E402
from bindcraft.af.alphafold.model import config as af_config  # noqa: E402

S, N = 5, 7
cfg = af_config.model_config('model_1_ptm')
gc = cfg.model.global_config
gc.use_dgram = False
gc.use_remat = False
gc.bfloat16 = False
gc.subbatch_size = None
gc.zero_init = False     # otherwise output_w is zero, the loss sees only output_b, and EVERY other gradient is
                         # trivially zero - which looks like the finding and is just a degenerate initialisation
c = cfg.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias

rng = np.random.default_rng(0)
Cm = 32
Cz = 16
msa = jnp.asarray(rng.standard_normal((S, N, Cm)) * 0.5, jnp.float64)
pair = jnp.asarray(rng.standard_normal((N, N, Cz)) * 0.5, jnp.float64)
msa_mask = jnp.asarray((rng.random((S, N)) > 0.2).astype(np.float32))
msa_mask = msa_mask.at[:, 0].set(1.0)
cot = jnp.asarray(rng.standard_normal((S, N, Cm)) * 0.5, jnp.float64)


def fwd(msa, msa_mask, pair):
    return modules.MSARowAttentionWithPairBias(c, gc)(msa, msa_mask, pair)


model = hk.transform(fwd)
key = jax.random.PRNGKey(0)
params = model.init(key, msa, msa_mask, pair)
# and randomise every parameter, so nothing is dead merely because haiku initialised it to zero (gating_w always is)
leaves, treedef = jax.tree.flatten(params)
params = jax.tree.unflatten(
    treedef, [jnp.asarray(rng.standard_normal(l.shape) * 0.2, jnp.float64) for l in leaves])
print("parameters:")
for mod, d in params.items():
    for name, v in d.items():
        print(f"  {mod}/{name:<12} {tuple(v.shape)}")


def loss(params):
    out = model.apply(params, key, msa, msa_mask, pair)
    return jnp.sum(cot * out)


g = jax.grad(loss)(params)
print(f"\nloss = {float(loss(params)):.6f}\n")
print("gradient norms, per parameter:")
rows = []
for mod, d in g.items():
    for name, v in d.items():
        rows.append((f"{mod}/{name}", float(jnp.linalg.norm(v)), float(jnp.max(jnp.abs(v)))))
width = max(len(r[0]) for r in rows)
for n, nrm, mx in sorted(rows, key=lambda r: r[1]):
    flag = "   <-- ZERO" if mx < 1e-13 else ""
    print(f"  {n:<{width}}  |g| = {nrm:12.3e}   max|g| = {mx:12.3e}{flag}")

# the two the claim is about, side by side
fn = [k for k in g if 'feat_2d_norm' in k]
if fn:
    k = fn[0]
    off = g[k].get('offset')
    sca = g[k].get('scale')
    print(f"\n{k}:")
    print(f"  offset  max|g| = {float(jnp.max(jnp.abs(off))):.3e}   "
          f"{'DEAD (identically zero)' if float(jnp.max(jnp.abs(off))) < 1e-13 else 'NOT zero'}")
    print(f"  scale   max|g| = {float(jnp.max(jnp.abs(sca))):.3e}   "
          f"{'dead' if float(jnp.max(jnp.abs(sca))) < 1e-13 else 'a real parameter'}")

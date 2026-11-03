"""One AlphaFold 2 Evoformer block's forward AND its gradient, from af3-any-model's JAX AF2, in float64.

    ~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --out cuda/grad/data/af2-block \
        [--extra] [--opm-last] [--L 37 --S 5] [--model model_1_multimer_v3] [--block 0]

The reference every CUDA backward in cuda/grad/af2 is held to. One EvoformerIteration (the main stack's, or with
--extra the extra-MSA stack's, which has global column attention and 64 MSA channels), its parameters sliced out of
the checkpoint at --block, run in float64 on random inputs with a padded tail (masked residues and a masked MSA
row), and a loss  sum(msa_out * ct_msa) + sum(pair_out * ct_pair)  with fixed random cotangents. jax.vjp gives the
gradient of that loss to both inputs and to EVERY parameter, which is what training wants.

Written in cuda/af3's model.idx/model.bin format (float32 on disk; the float64 reference rounds once):
  p/<module>/<param>        the block's parameters, haiku's names under the block
  in/msa in/pair in/msa_mask in/pair_mask   ct/msa ct/pair   out/msa out/pair
  g/msa g/pair g/p/<module>/<param>
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "af2"))
from oracle import write_native  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--model", default="model_1_multimer_v3")
  ap.add_argument("--params", default=os.path.expanduser("~/lfjax/af2_params"))
  ap.add_argument("--reference", default=os.path.expanduser("~/lfjax/branch_src"))
  ap.add_argument("--block", type=int, default=0)
  ap.add_argument("--extra", action="store_true")
  ap.add_argument("--opm-last", action="store_true")
  ap.add_argument("--L", type=int, default=37)
  ap.add_argument("--S", type=int, default=5)
  ap.add_argument("--pad", type=int, default=3, help="masked residues at the tail")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  sys.argv = sys.argv[:1]
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  sys.path[:0] = [os.path.join(args.reference, "src")]

  import jax
  jax.config.update("jax_enable_x64", True)
  jax.config.update("jax_default_matmul_precision", "highest")
  import jax.numpy as jnp
  import haiku as hk
  from alphafold3.af2.model import config as af2config
  from alphafold3.af2.model import modules

  cfg = af2config.model_config(args.model)
  c = cfg.model.embeddings_and_evoformer.evoformer
  c.outer_product_mean.first = not args.opm_last
  gc = cfg.model.global_config
  gc.subbatch_size = None
  gc.bfloat16 = False

  stack = "extra_msa_stack" if args.extra else "evoformer_iteration"
  prefix = "alphafold/alphafold_iteration/evoformer/" + stack + "/"
  raw = np.load(os.path.join(args.params, "params_" + args.model + ".npz"))
  params = {}
  for k in raw.keys():
    if k.startswith(prefix):
      mod, name = k[len(prefix):].split("//")
      params.setdefault(stack + "/" + mod, {})[name] = np.asarray(raw[k][args.block], np.float64)
  if not params:
    sys.exit("no parameters under " + prefix)

  def fn(msa, pair, msa_mask, pair_mask):
    it = modules.EvoformerIteration(c, gc, is_extra_msa=args.extra, name=stack)
    out = it({"msa": msa, "pair": pair}, {"msa": msa_mask, "pair": pair_mask}, use_dropout=False,
             safe_key=modules.prng.SafeKey(jax.random.PRNGKey(0)))
    return out["msa"], out["pair"]

  f = hk.transform(fn)
  L, S = args.L, args.S
  cm = 64 if args.extra else 256
  rng = np.random.default_rng(args.seed)
  msa = rng.normal(size=(S, L, cm))
  pair = rng.normal(size=(L, L, 128))
  rmask = np.ones(L); rmask[L - args.pad:] = 0
  msa_mask = np.tile(rmask, (S, 1)); msa_mask[S - 1, : L // 2] = 0
  pair_mask = np.outer(rmask, rmask)
  ct_msa = rng.normal(size=(S, L, cm))
  ct_pair = rng.normal(size=(L, L, 128))
  # A padded residue's query row is masked in every key (-1e9 on all of them), so its softmax is whatever the
  # arithmetic makes of q.k - 1e9: float64 keeps q.k, float32 keeps nothing. Neither is a defined answer, so the
  # loss reads no padded output: their cotangent is zero, and nothing downstream of a padded row reaches the loss.
  ct_msa[:, L - args.pad:] = 0
  ct_pair[L - args.pad:] = 0
  ct_pair[:, L - args.pad:] = 0

  def loss_parts(p, msa, pair):
    return f.apply(p, None, msa, pair, jnp.asarray(msa_mask), jnp.asarray(pair_mask))

  (om, op), vjp = jax.vjp(loss_parts, params, jnp.asarray(msa), jnp.asarray(pair))
  gp, gm, gz = vjp((jnp.asarray(ct_msa), jnp.asarray(ct_pair)))

  # the same vjp in float32: how far the reference's own float32 lands from its float64 on each tensor, which is
  # the scale a float32 port is judged on (an ill-conditioned gradient is ill-conditioned in JAX too)
  c32 = lambda t: jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), t)
  def parts32(p, msa, pair):
    return f.apply(p, None, msa, pair, jnp.asarray(msa_mask, jnp.float32), jnp.asarray(pair_mask, jnp.float32))
  (om32, op32), vjp32 = jax.vjp(parts32, c32(params), c32(msa), c32(pair))
  gp32, gm32, gz32 = vjp32((c32(ct_msa), c32(ct_pair)))
  def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-300)))

  arrays = {}
  for name, a, b in [("out/msa", om32, om), ("out/pair", op32, op), ("g/msa", gm32, gm), ("g/pair", gz32, gz)]:
    arrays["e32/" + name] = np.array([rel(a, b)], np.float32)
  for mod, d in gp.items():
    for name, v in d.items():
      arrays["e32/g/p/" + mod[len(stack) + 1:] + "/" + name] = np.array([rel(gp32[mod][name], v)], np.float32)
  f32 = lambda x: np.asarray(x, np.float32).ravel()
  for mod, d in params.items():
    for name, v in d.items():
      arrays["p/" + mod[len(stack) + 1:] + "/" + name] = f32(v)
  for name, v in [("in/msa", msa), ("in/pair", pair), ("in/msa_mask", msa_mask), ("in/pair_mask", pair_mask),
                  ("ct/msa", ct_msa), ("ct/pair", ct_pair), ("out/msa", om), ("out/pair", op),
                  ("g/msa", gm), ("g/pair", gz)]:
    arrays[name] = f32(v)
  for mod, d in gp.items():
    for name, v in d.items():
      arrays["g/p/" + mod[len(stack) + 1:] + "/" + name] = f32(v)
  arrays["meta/L"] = np.array([L], np.float32)
  arrays["meta/S"] = np.array([S], np.float32)
  arrays["meta/cm"] = np.array([cm], np.float32)
  arrays["meta/extra"] = np.array([1.0 if args.extra else 0.0], np.float32)
  arrays["meta/opm_first"] = np.array([0.0 if args.opm_last else 1.0], np.float32)
  write_native(args.out, arrays)
  print(args.out, len(arrays), "arrays;", sum(1 for k in arrays if k.startswith("g/p/")), "parameter gradients")


if __name__ == "__main__":
  main()

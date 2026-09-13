"""af3-any-model's whole TRUNK on the real 6MRR batch, stage by stage.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_trunk_taps.py boltz2

🔴 THE TRUNK'S GATE HERE COMPARES THE GPU AGAINST THIS PORT'S OWN CPU, so it
says the two agree and cannot say whether either is the model. boltz2's denoise
step is now exact and its FOLD is still a ball - consecutive CA 6.27 A against
3.80 - which is only possible if the trunk it is fed is wrong, and no checker in
this repository could see that.

`AF3_ESM_TRUNK_TAPS=1` makes the reference's Evoformer record z at each seam:
`z_init` (the embedder), `z_after_template`, `z_after_msa`, `trunk_in_pair` and
`trunk_out_pair` (the pairformer). With the final single and pair beside them
that is the same stage-by-stage bisect `dump_af3_denoise_stages.py` gave the
diffusion side, on the half that is still wrong.

Built by calling `ev.Evoformer` DIRECTLY rather than through `fold_check.fold`,
because recycling runs inside model.py's fori_loop and returns tracers - the
recipe is `dev/oracles/trunk_stage_parity.py`'s, including its scope rename.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["AF3_ESM_TRUNK_TAPS"] = "1"
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np, haiku as hk, jax, jax.numpy as jnp
import fold_check
from alphafold3.model import feat_batch, model as af3_model
from alphafold3.model import params as afp
from alphafold3.model.network import evoformer as ev
from alphafold3.model.components import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else "boltz2"
PASSES = int(os.environ.get("PASSES", "1"))
seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
# 🔴 fp32, NOT the fold path's bfloat16. A port compared against a bfloat16
# reference is being held to the reference's rounding as well as its model.
cfg.global_config.bfloat16 = "none"


@hk.transform
def fwd(b):
    fb = feat_batch.Batch.from_data_dict(b)
    length = fb.token_features.mask.shape[0]
    prev = {"pair": jnp.zeros((length, cfg.evoformer.pair_channel), jnp.float32)
            if False else jnp.zeros((length, length, cfg.evoformer.pair_channel), jnp.float32),
            "single": jnp.zeros((length, cfg.evoformer.seq_channel), jnp.float32)}
    target = af3_model.create_target_feat_embedding(
        batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
    module = ev.Evoformer(cfg.evoformer, cfg.global_config)
    embeddings = None
    for _ in range(PASSES):
        embeddings = module(batch=fb, prev=prev, target_feat=target,
                            key=jax.random.PRNGKey(0))
        prev = {**prev, **{k: v.astype(jnp.float32)
                           for k, v in embeddings.items() if k in prev}}
    return embeddings, target


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
params = afp.get_model_haiku_params(model_dir=model_dir)
# Calling Evoformer directly drops the Model's own `diffuser/` prefix, and the
# blob's ROOT scope becomes haiku's '~'.
params = {("~" if k == "diffuser"
           else k[len("diffuser/"):] if k.startswith("diffuser/") else k): v
          for k, v in params.items()}
out, target = fwd.apply(params, jax.random.PRNGKey(0), b)
taps = {k: [np.asarray(x, np.float32) for x in v] for k, v in ev.ESM_TRUNK_TAPS.items()}
print(MODEL, "passes", PASSES, "| emb", sorted(out.keys()),
      "| taps", {k: len(v) for k, v in taps.items()})

record = {}
def put(name, a):
    a = np.asarray(a, np.float32)
    record[name] = {"shape": list(a.shape), "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
                    "data": a.ravel().tolist()}
    print("  %-22s %-20s rms %9.4f" % (name, list(a.shape), record[name]["rms"]))

put("target_feat", target)
for name, values in taps.items():
    put("tap.%s" % name, values[-1])       # the LAST pass; the dump keeps one cycle
for name in ("single", "pair"):
    if name in out: put(name, out[name])
p = "/tmp/af3-oracle-trunk-%s.json" % MODEL
open(p, "w").write(json.dumps({"model": MODEL, "passes": PASSES, "stages": record}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

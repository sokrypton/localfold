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
from alphafold3.model.network import featurization as _feat
from alphafold3.model import feat_batch, model as af3_model
from alphafold3.model import params as afp
from alphafold3.model.network import evoformer as ev
from alphafold3.model.components import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else "boltz2"
# 🔴 THE MSA SUBSAMPLE IS A DRAW, AND A DRAW IS NOT COMPARABLE. The Evoformer
# gumbel-shuffles the alignment and keeps the first `num_msa` rows, so on a
# 16384-row batch native and this port see different SEQUENCES and their MSA
# stages cannot be compared at all. With this on, the shuffle is the identity
# and both sides take the alignment's first 1024 rows - which is what
# `foldBatch`'s own cap does. Off by default, so a dump without it is still the
# model as it runs.
if os.environ.get("DETERMINISTIC_MSA"):
    _feat.shuffle_msa = lambda key, msa: (msa, key)
PASSES = int(os.environ.get("PASSES", "1"))
seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
# 🔴 fp32, NOT the fold path's bfloat16. A port compared against a bfloat16
# reference is being held to the reference's rounding as well as its model.
cfg.global_config.bfloat16 = "none"
# 🔴 BLOCKS= TRUNCATES THE PAIRFORMER ON BOTH SIDES, which is how a divergence
# that grows with depth is separated from one that is already there at block
# one. `layer_stack` takes its trip count from the config and its weights from
# the array's leading axis, so the params are sliced to match below.
BLOCKS = int(os.environ.get("BLOCKS", "0")) or cfg.evoformer.pairformer.num_layer
cfg.evoformer.pairformer.num_layer = BLOCKS
MSA_BLOCKS = int(os.environ.get("MSA_BLOCKS", "0")) or cfg.evoformer.msa_stack.num_layer
cfg.evoformer.msa_stack.num_layer = MSA_BLOCKS


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
# Slice every stacked weight whose depth the config just changed. Asserted
# rather than inferred: a tensor whose leading axis merely happens to be 48 and
# is not a block stack would be silently truncated.
init = fwd.init(jax.random.PRNGKey(0), b)
sliced, cut, _cut_names = {}, 0, []
for scope, leaves in params.items():
    sliced[scope] = {}
    for leaf, value in leaves.items():
        value = np.asarray(value)
        want = np.asarray(init.get(scope, {}).get(leaf, value)).shape
        if value.shape != want and value.shape[1:] == want[1:] and want[0] <= value.shape[0]:
            _cut_names.append("%s/%s %s -> %s" % (scope, leaf, value.shape, want))
            value = value[:want[0]]
            cut += 1
        sliced[scope][leaf] = value
if cut:
    print("  sliced %d stacked tensors to %d blocks" % (cut, BLOCKS))
    for name in _cut_names[:6]:
        print("    ", name)
# 🔴 AND EVERY MODULE INSIDE THE TARGET-FEAT ENCODER, ON DEMAND. OpenDDE's
# `target_feat` reads 9.65e-2 against this dump at f32 with a bundle that agrees
# with the reference's params 481 of 481 - so the disagreement is in the port's
# code, in 8 of 384 channels, and a whole-tensor residual names none of them.
# `dump_af3_scopes.py` cannot reach these: its forward pass is the denoiser's
# and never runs the trunk's encoder. CAPTURE is a regex over module names,
# e.g. CAPTURE=evoformer_conditioning.
SCOPES = {}
_seen = {}
def _tracer(next_f, targs, tkwargs, context):
    value = next_f(*targs, **tkwargs)
    try:
        name = context.module.module_name
        if hasattr(value, "shape"):
            a = np.asarray(value, np.float32)
            _seen[name] = _seen.get(name, 0) + 1
            # Keyed by name and overwritten so the LAST write - the apply pass -
            # wins over hk.init's random one; `calls` keeps a module genuinely
            # called twice from hiding behind that.
            SCOPES[name] = {"shape": list(a.shape),
                            "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
                            "calls": _seen[name], "data": a.ravel().tolist()}
    except Exception:
        pass
    return value

CAPTURE = os.environ.get("CAPTURE")
if CAPTURE:
    import re as _re
    with hk.intercept_methods(_tracer):
        out, target = fwd.apply(sliced, jax.random.PRNGKey(0), b)
else:
    out, target = fwd.apply(sliced, jax.random.PRNGKey(0), b)
taps = {k: [np.asarray(x, np.float32) for x in v] for k, v in ev.ESM_TRUNK_TAPS.items()}
print(MODEL, "passes", PASSES, "blocks", BLOCKS, "msa", MSA_BLOCKS, "| emb", sorted(out.keys()),
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
if CAPTURE:
    kept = 0
    for _name, _entry in SCOPES.items():
        if _re.search(CAPTURE, _name):
            record["scope.%s" % _name] = _entry
            kept += 1
            print("  scope.%-44s %-18s rms %9.4f  calls %d"
                  % (_name, _entry["shape"], _entry["rms"], _entry["calls"]))
    # A capture that matches nothing is a typo, not agreement.
    assert kept, "CAPTURE=%r matched none of %d modules" % (CAPTURE, len(SCOPES))

p = ("/tmp/af3-oracle-trunk-%s.json" % MODEL if BLOCKS == 48
     else "/tmp/af3-oracle-trunk-%s-b%d.json" % (MODEL, BLOCKS))
open(p, "w").write(json.dumps({"model": MODEL, "passes": PASSES, "stages": record}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

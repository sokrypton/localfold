"""af3-any-model's OpenDDE ConfidenceHead, inputs and outputs.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_opendde_confidence.py

🔴 OPENDDE'S HEAD IS ITS OWN PARAMETRISATION AND `dump_af3_confidence.py`
CANNOT REACH IT. That script indexes
`diffuser/confidence_head/~_embed_features/left_target_feat_project` to learn
the target-feat width, and OpenDDE has no such tensor - it initialises the pair
from s_inputs through `linear_no_bias_s1`/`s2`, adds a binned AND a raw distance
embedding, runs a four-block pairformer, and reads PAE/PDE off the pair with
pLDDT/resolved as a per-token-atom einsum against `[24, c_s, bins]` weights. So
it died with a KeyError, and OpenDDE was the one family whose confidence head
had no oracle at all - its only gate was `check-opendde-confidence.js`, which is
differential against OUR host code and cannot see a shared misreading.

The layout is `dev/oracles/confidence_parity.py`'s `ours_opendde`: a synthesised
`atom_to_token_idx`/`atom_to_tokatom_idx` at a fixed slot count, because the
head's arithmetic is what is under test and the structural tokenisation is
gated elsewhere (check-opendde-expander.js).

🔴 THE COORDINATES MUST SPAN THE DISTANCE BINS. OpenDDE's are 39 bins from 3.25
to 52.0 A; a tight cloud puts every pair in one column of the one-hot and the
embedding stops being tested. POS_SCALE is confidence_parity's own 12.0.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np, haiku as hk, jax, jax.numpy as jnp
import fold_check
from alphafold3.model import params as afp
from alphafold3.model.network import opendde_confidence

MODEL = "opendde"
SCALE = float(os.environ.get("POS_SCALE", "12.0"))
# The token count is the head's, not a protein's: this gates arithmetic, and a
# smaller n keeps an n^2 x 384 pair dump readable. 68 matches every other dump
# in oracle-dumps/ so the shapes line up by eye.
n = int(os.environ.get("TOKENS", "68"))
max_atoms = int(os.environ.get("SLOTS", "24"))

seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
_batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
cfg.global_config.bfloat16 = "none"
c_s, c_z = cfg.evoformer.seq_channel, cfg.evoformer.pair_channel

# 🔴 THE WIDTH COMES OFF A TENSOR OPENDDE ACTUALLY HAS. `linear_no_bias_s1` is
# the projection s_inputs enters through, so its input axis IS the target-feat
# width - the same question the AF3 dumper asks of a tensor OpenDDE lacks.
full = afp.get_model_haiku_params(model_dir=model_dir)
width = int(full["diffuser/confidence_head/linear_no_bias_s1"]["weights"].shape[0])

rng = np.random.default_rng(0)
pos = (rng.normal(size=(n, max_atoms, 3)) * SCALE).astype(np.float32)
s_inputs = (rng.normal(size=(n, width)) * 0.5).astype(np.float32)
s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
a2t = np.repeat(np.arange(n), max_atoms).astype(np.int32)
a2ta = np.tile(np.arange(max_atoms), n).astype(np.int32)
rep = pos.reshape(-1, 3)[np.arange(n) * max_atoms]
seq_mask = np.ones(n, np.float32)

def fwd():
    return opendde_confidence.OpenDDEConfidenceHead(
        c_s, c_z, width, cfg.global_config)(
            jnp.asarray(s_inputs), jnp.asarray(s), jnp.asarray(z), jnp.asarray(rep),
            jnp.asarray(a2t), jnp.asarray(a2ta), jnp.asarray(seq_mask))

# Every module's output too, keyed by name and overwritten so the LAST write -
# the apply pass - wins over hk.init's random one.
SCOPES = {}
def tracer(next_f, targs, tkwargs, context):
    value = next_f(*targs, **tkwargs)
    try:
        if hasattr(value, "shape"):
            a = np.asarray(value, np.float32)
            SCOPES[context.module.module_name] = {
                "shape": list(a.shape),
                "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
                "data": a.ravel().tolist()}
    except Exception:
        pass
    return value

f = hk.transform(fwd)
init = f.init(jax.random.PRNGKey(0))
params, unmapped = {}, []
for scope in init:
    params[scope] = {}
    for leaf in init[scope]:
        src = full.get("diffuser/" + scope)
        value = None if src is None else src.get(leaf)
        if value is None:
            unmapped.append("%s/%s" % (scope, leaf)); params[scope][leaf] = init[scope][leaf]
        else:
            params[scope][leaf] = np.asarray(value, np.float32)
# 🔴 A HEAD PARTLY AT INIT IS NOISE, NOT A REFERENCE - confidence_parity.py's
# own assertion, and the reason a "close enough" comparison here would mean
# nothing at all.
assert not unmapped, unmapped[:4]
with hk.intercept_methods(tracer):
    out = f.apply(params, jax.random.PRNGKey(0))
print(MODEL, "tokens", n, "slots", max_atoms, "target_feat", width,
      "| keys", sorted(out.keys()))

record = {}
def put(name, a):
    a = np.asarray(a, np.float32)
    record[name] = {"shape": list(a.shape),
                    "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
                    "data": a.ravel().tolist()}
    print("  %-34s %-22s rms %10.4f" % (name, list(a.shape), record[name]["rms"]))

put("in.pair", z); put("in.single", s); put("in.targetFeat", s_inputs)
put("in.coordinates", rep)
put("in.atomPositions", pos)
put("in.seqMask", seq_mask)
put("in.atomToToken", a2t)
put("in.atomToSlot", a2ta)
for key in ("predicted_lddt", "full_pae", "full_pde", "average_pde",
            "predicted_experimentally_resolved", "predicted_lddt_logits",
            "predicted_aligned_error_logits", "predicted_distance_error_logits",
            "experimentally_resolved_logits"):
    if key in out: put("out.%s" % key, out[key])
CAPTURE = os.environ.get("CAPTURE")
if CAPTURE:
    import re as _re
    for _name, _entry in SCOPES.items():
        if _re.search(CAPTURE, _name):
            record["scope.%s" % _name] = _entry
            print("  %-44s %-20s rms %10.4f"
                  % (_name, _entry["shape"], _entry["rms"]))
p = "/tmp/af3-oracle-confidence-opendde.json"
open(p, "w").write(json.dumps({"model": MODEL, "tokens": n, "slots": max_atoms,
                               "targetFeatWidth": width, "stages": record}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

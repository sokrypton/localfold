"""af3-any-model's own ConfidenceHead, inputs and outputs, for a model.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_confidence.py boltz2

🔴 THE HEAD'S ONLY GATE HERE COMPARES THE GPU AGAINST THIS PORT'S CPU. Both can
be wrong together, which is how boltz2's head was written against a reading of
the reference rather than against its numbers - and how protenix2 folds ideal
geometry at pLDDT 59 while boltz2 folds the same protein at 93.

The trunk activations are seeded and the LAYOUT is a real featurised batch, the
same split `dev/oracles/confidence_parity.py` uses: both heads consume an atom
layout (which slot of which token each atom is, and which atom represents a
token), and that cannot be synthesised.

🔴 THE COORDINATES MUST SPAN THE DISTANCE BINS. A tight cloud puts every pair
in one column of the one-hot and the distance embedding stops being tested;
POS_SCALE inflates it, and 12.0 is confidence_parity's own default.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np, haiku as hk, jax, jax.numpy as jnp
import fold_check
from alphafold3.model import feat_batch
from alphafold3.model import params as afp
from alphafold3.model.network import confidence_head, evoformer as ev

MODEL = sys.argv[1] if len(sys.argv) > 1 else "boltz2"
SCALE = float(os.environ.get("POS_SCALE", "12.0"))
seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
batch_dict, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
batch = feat_batch.Batch.from_data_dict(batch_dict)
cfg.global_config.bfloat16 = "none"
n = int(np.asarray(batch.token_features.mask).shape[0])
max_atoms = int(np.asarray(batch.predicted_structure_info.atom_mask).shape[1])
rng = np.random.default_rng(0)
pos = (rng.normal(size=(n, max_atoms, 3)) * SCALE).astype(np.float32)
c_s, c_z = cfg.evoformer.seq_channel, cfg.evoformer.pair_channel
# The head's target_feat is whatever the TRUNK produced, so its width is the
# model's own - 447 for AF3 and protenix2, 384 for boltz2.
width = afp.get_model_haiku_params(model_dir=model_dir)[
    "diffuser/confidence_head/~_boltz2_reembed/left_target_feat_project"
    if MODEL == "boltz2" else
    "diffuser/confidence_head/~_embed_features/left_target_feat_project"]["weights"].shape[0]
s_inputs = (rng.normal(size=(n, width)) * 0.5).astype(np.float32)
s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

bond_matrix = ev.token_bond_matrix(batch, symmetrize=True)
bond_type_matrix = ev.token_bond_type_matrix(batch, symmetrize=True)

def fwd():
    return confidence_head.ConfidenceHead(cfg.heads.confidence, cfg.global_config)(
        dense_atom_positions=jnp.asarray(pos),
        embeddings={"pair": jnp.asarray(z), "single": jnp.asarray(s),
                    "target_feat": jnp.asarray(s_inputs)},
        seq_mask=batch.token_features.mask,
        token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
        asym_id=batch.token_features.asym_id,
        token_features=batch.token_features,
        bond_matrix=bond_matrix, bond_type_matrix=bond_type_matrix,
        atom_name_chars=batch.ref_structure.atom_name_chars)

# 🔴 EVERY MODULE'S OUTPUT TOO, keyed by name and overwritten so the LAST write
# - the apply pass - wins over hk.init's random one. The re-embedding is nine
# terms summed into one tensor; without them "z is wrong by 3.32" names nothing.
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
full = afp.get_model_haiku_params(model_dir=model_dir)
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
    print("  %-28s %-22s rms %10.4f" % (name, list(a.shape), record[name]["rms"]))

# The representative atom per token, which is what the head embeds distances of.
from alphafold3.model.atom_layout import atom_layout
rep = atom_layout.convert(batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                          jnp.asarray(pos), layout_axes=(-3, -2))
put("in.pair", z); put("in.single", s); put("in.targetFeat", s_inputs)
put("in.pseudoBeta", rep)
put("in.seqMask", batch.token_features.mask)
put("in.bondMatrix", bond_matrix)
put("in.bondTypeMatrix", bond_type_matrix)
for name in ("residue_index", "token_index", "asym_id", "entity_id", "sym_id"):
    put("in.%s" % name, getattr(batch.token_features, name))
for key in ("predicted_lddt", "full_pde", "full_pae", "average_pde",
            "predicted_experimentally_resolved", "tmscore_adjusted_pae_global"):
    if key in out: put("out.%s" % key, out[key])
CAPTURE = os.environ.get("CAPTURE")
if CAPTURE:
    import re as _re
    for _name, _entry in SCOPES.items():
        if _re.search(CAPTURE, _name):
            record["scope.%s" % _name] = _entry
            print("  %-40s %-20s rms %10.4f"
                  % (_name, _entry["shape"], _entry["rms"]))
p = "/tmp/af3-oracle-confidence-%s.json" % MODEL
open(p, "w").write(json.dumps({"model": MODEL, "tokens": n, "slots": max_atoms,
                               "targetFeatWidth": int(width), "stages": record}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

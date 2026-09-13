"""Dump a model's TEMPLATE EMBEDDER - its features, its inputs and its output.

    # on a machine with af3-any-model and its weights:
    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles \\
      python dump_af3_template.py protenix2

🔴 WHY IT EXISTS. protenix2 runs boltz2's FUSED template module, not AF3's,
and docs/AF3.md carries the whole specification - the forward, the 108-wide
feature order, Boltz's frame convention, the 32-class remap. Writing both halves
of LocalFold's check from that specification would have produced two pieces of
new code agreeing with each other and a checker that cannot fail, which is this
repository's own rule about verifying against the oracle rather than against our
own reference. So the oracle comes first.

🔴 AND IT RECORDS THE SEAM THE REFERENCE'S OWN GATE USES. The features and
the forward are separate jobs: `template_parity.py`'s docstring is explicit that
it hands ITS OWN derived features to the vendor, so any gap it reports is in the
projections or the pairformer stack and NOT in the feature construction. This
dump keeps both halves - `feat:*` for the 108 columns and `output` for what the
module makes of them - so a port can be held to the forward alone before its
featuriser is written, and the two failures can never be confused.

It runs `template_parity.ours`, which is that gate's own entry point, so what is
recorded is exactly what it compares.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/ubuntu/alphafold3/src")
sys.path.insert(0, "/home/ubuntu/alphafold3")
sys.path.insert(0, "/home/ubuntu/alphafold3/dev/oracles")
import numpy as np, jax, haiku as hk
import template_parity as TP
import fold_check
from alphafold3.model import feat_batch
from alphafold3.model.network import template_modules as T

MODEL = sys.argv[1] if len(sys.argv) > 1 else "protenix2"
seq, tmpl = TP._self_template(os.path.expanduser("~/5K9P.cif"), "A")
batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None, templates=[tmpl])
fb = feat_batch.Batch.from_data_dict(batch)
templates = fb.templates
n_tok = int(np.asarray(fb.token_features.mask).shape[0])
asym = np.asarray(fb.token_features.asym_id).astype(np.int64)
multichain = (asym[:, None] == asym[None, :]).astype(np.float32)
single = type(templates)(aatype=np.asarray(templates.aatype)[0],
                         atom_positions=np.asarray(templates.atom_positions)[0],
                         atom_mask=np.asarray(templates.atom_mask)[0])
feats = TP.our_features(MODEL, cfg.evoformer.template, single, multichain)
c_z = cfg.evoformer.pair_channel
rng = np.random.default_rng(0)
z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
pair_mask = np.ones((n_tok, n_tok), np.float32)

print("tokens", n_tok, "c_z", c_z, "feature keys:", sorted(feats.keys()))
out = {"model": MODEL, "tokens": n_tok, "pairChannels": c_z,
       "source": "sokrypton/alphafold3 @ af3-any-model, dev/oracles/template_parity.py",
       "inputs": {}}
def put(name, a):
    a = np.asarray(a)
    out["inputs"][name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                           "data": a.astype(np.float32).ravel().tolist()}
put("pair", z); put("pairMask", pair_mask); put("asymId", asym.astype(np.float32))
put("multichainMask2d", multichain)
for k, v in feats.items():
    if isinstance(v, (np.ndarray, jax.Array)) and np.asarray(v).ndim > 0:
        put("feat:" + k, v)
# ...and the module's OUTPUT with the real weights, which is the half a
# forward can actually be checked against. `ours` is template_parity's own
# entry point, so this records exactly what that gate compares.
one = type(templates)(aatype=np.array(templates.aatype)[:1].copy(),
                      atom_positions=np.array(templates.atom_positions)[:1].copy(),
                      atom_mask=np.array(templates.atom_mask)[:1].copy())
got = TP.ours(MODEL, cfg, model_dir, one, z, pair_mask, multichain)
got = np.asarray(got)
print("output", got.shape, "rms %.4f" % float(np.sqrt((got**2).mean())))
out["output"] = {"shape": list(got.shape), "dtype": str(got.dtype),
                 "data": got.astype(np.float32).ravel().tolist()}
out["slots"] = 1

path = "/tmp/af3-oracle-template-%s.json" % MODEL
open(path, "w").write(json.dumps(out))
print("wrote", path, "%.1f MB" % (os.path.getsize(path) / 2**20))

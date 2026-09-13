"""Dump one denoise step: its inputs and af3-any-model's output."""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import denoise_parity as DP
import fold_check
from alphafold3.model import feat_batch
from atom_parity import flat_atom_features

MODEL = sys.argv[1] if len(sys.argv) > 1 else "protenix2"
NOISE = float(os.environ.get("NOISE", "16.0"))
seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
fb = feat_batch.Batch.from_data_dict(batch)
feats = flat_atom_features(fb)
n_tok = int(np.asarray(fb.token_features.mask).shape[0])
max_atoms = feats['mask'].shape[1]
rng = np.random.default_rng(0)
c_s, c_z = cfg.evoformer.seq_channel, cfg.evoformer.pair_channel
s = (rng.normal(size=(n_tok, c_s)) * 0.5).astype(np.float32)
z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap), [448], np.arange(384)])
s449 = (rng.normal(size=(n_tok, 449)) * 0.5).astype(np.float32)
s449[:, np.setdiff1d(np.arange(449), idx)] = 0.0
s447 = s449[:, idx]
pos_dense = (rng.normal(size=(n_tok, max_atoms, 3)) * NOISE).astype(np.float32) * feats['mask'][..., None]
x_got = np.asarray(DP.ours(MODEL, cfg, model_dir, fb, pos_dense, NOISE, s447, s, z))
print("tokens", n_tok, "max_atoms", max_atoms, "c_s", c_s, "c_z", c_z,
      "out", x_got.shape, "rms %.4f" % float(np.sqrt((x_got**2).mean())))
out = {"model": MODEL, "tokens": n_tok, "maxAtoms": int(max_atoms), "noise": NOISE,
       "seqChannels": int(c_s), "pairChannels": int(c_z),
       "source": "af3-any-model dev/oracles/denoise_parity.ours", "inputs": {}}
def put(n, a):
    a = np.asarray(a)
    out["inputs"][n] = {"shape": list(a.shape), "dtype": str(a.dtype),
                        "data": a.astype(np.float32).ravel().tolist()}
put("single", s); put("pair", z); put("sInputs", s447); put("posNoisy", pos_dense)
put("atomMask", feats['mask'].astype(np.float32))
cross = fb.atom_cross_att
for nm in ("token_atoms_to_queries","queries_to_keys","queries_to_token_atoms",
           "tokens_to_queries","tokens_to_keys"):
    g = getattr(cross, nm)
    put(f"{nm}:gather_idxs", np.asarray(g.gather_idxs))
    put(f"{nm}:gather_mask", np.asarray(g.gather_mask))
# ...and the reference-conformer features, which the per-atom conditioning
# needs. LocalFold takes that conditioning as an INPUT to its diffusion head
# where native computes it inside the module, so a comparison of the two has to
# build it on this side from the same numbers native used.
for _k in ("ref_pos", "ref_space_uid", "ref_mask", "ref_element",
           "ref_charge", "ref_atom_name_chars"):
    put(_k, batch[_k])
put("seq_mask", fb.token_features.mask)
out["output"] = {"shape": list(x_got.shape), "data": x_got.astype(np.float32).ravel().tolist()}
p = "/tmp/af3-oracle-denoise-%s.json" % MODEL
open(p, "w").write(json.dumps(out))
print("wrote", p, "%.1f MB" % (os.path.getsize(p)/2**20))

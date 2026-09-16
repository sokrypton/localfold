"""Dump one denoise step from a REAL fold: real trunk conditioning, real structure.

🔴 WHY THIS EXISTS WHEN `dump_af3_denoise.py` ALREADY DUMPS A DENOISE STEP.
That one builds `pos_dense = rng.normal(...) * NOISE` and `s`, `z`, `s_inputs`
as `rng.normal(...) * 0.5`, so the flagship AF3 denoise oracle has never seen a
structure or a trunk output. It is a WIRING test, and LocalFold passes it at
relRMS 1.3e-5 at sigma 16, 9.9e-6 at 0.2 and 1.1e-5 at 0.02 while its folds
carry side-chain bonds at rms 0.339 A against native's 0.051 - 100% of them
SHORT, progressively with distance from the backbone. A gate fed noise cannot
see a defect that only exists in the presence of real geometry.

This runs native's own fold, captures the diffusion head's actual inputs, then
re-denoises the FINAL structure at a chosen sigma and records native's answer.
`fold_check.fold` does not jit, so a monkeypatch sees concrete arrays.

  ~/venv/bin/python dump_af3_real_denoise.py alphafold3 --noise 2.0
"""
import os, sys, json, argparse
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
for e in ("/home/ubuntu/alphafold3/src", "/home/ubuntu/alphafold3",
          "/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np

# 🔴 ENVIRONMENT, NOT FLAGS. tokamax calls absl FLAGS(sys.argv) from inside the
# gated linear unit, so any argv this script owns kills the fold with
# UnrecognizedFlagError three quarters of the way through a trunk.
class a:
    model = os.environ.get("MODEL", "alphafold3")
    noise = float(os.environ.get("NOISE", "2.0"))
    seed = int(os.environ.get("SEED", "0"))
    out = os.environ.get("OUT", "/tmp/af3-real-denoise-%s-n%s.json")

import fold_check
import denoise_parity as DP
from alphafold3.model import feat_batch
from alphafold3.model.network import diffusion_head
from atom_parity import flat_atom_features

SEQ = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"

captured = {}
_orig = diffusion_head.DiffusionHead.__call__

def patched(self, positions_noisy, noise_level, batch, embeddings, *ar, **kw):
    # The LAST call wins: the sampler walks down, so this is the lowest sigma.
    # Only the embeddings matter here - they are constant across the walk.
    if "single" not in captured:
        captured["single"] = np.asarray(embeddings["single"], np.float32)
        captured["pair"] = np.asarray(embeddings["pair"], np.float32)
        captured["target_feat"] = np.asarray(embeddings["target_feat"], np.float32)
        print("  captured single", captured["single"].shape,
              "pair", captured["pair"].shape,
              "target_feat", captured["target_feat"].shape, flush=True)
    return _orig(self, positions_noisy, noise_level, batch, embeddings, *ar, **kw)

diffusion_head.DiffusionHead.__call__ = patched
print("folding natively...", flush=True)
out, batch = fold_check.fold(a.model, SEQ, seed=a.seed)
diffusion_head.DiffusionHead.__call__ = _orig

fb = feat_batch.Batch.from_data_dict(batch)
feats = flat_atom_features(fb)
mask = feats["mask"]                                   # [tokens, max_atoms]
n_tok, max_atoms = mask.shape
clean = np.asarray(out["diffusion_samples"]["atom_positions"])[0].astype(np.float32)
assert clean.shape == (n_tok, max_atoms, 3), (clean.shape, n_tok, max_atoms)
clean = clean * mask[..., None]

rng = np.random.default_rng(a.seed)
pos_dense = (clean + rng.normal(size=clean.shape) * a.noise).astype(np.float32)
pos_dense *= mask[..., None]

s = captured["single"]; z = captured["pair"]; s447 = captured["target_feat"]
x_got = np.asarray(DP.ours(a.model, *fold_check._fold_setup(a.model, SEQ, None)[1:],
                           fb, pos_dense, a.noise, s447, s, z))
print("tokens", n_tok, "max_atoms", max_atoms, "out", x_got.shape,
      "rms %.4f" % float(np.sqrt((x_got ** 2).mean())), flush=True)

o = {"model": a.model, "tokens": int(n_tok), "maxAtoms": int(max_atoms),
     "noise": a.noise, "seqChannels": int(s.shape[-1]), "pairChannels": int(z.shape[-1]),
     "source": "real fold: native trunk conditioning + native final structure",
     "sequence": SEQ, "inputs": {}}
def put(n, arr):
    arr = np.asarray(arr)
    o["inputs"][n] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                      "data": arr.astype(np.float32).ravel().tolist()}
put("single", s); put("pair", z); put("sInputs", s447); put("posNoisy", pos_dense)
put("atomMask", mask.astype(np.float32))
cross = fb.atom_cross_att
for nm in ("token_atoms_to_queries", "queries_to_keys", "queries_to_token_atoms",
           "tokens_to_queries", "tokens_to_keys"):
    g = getattr(cross, nm)
    put(f"{nm}:gather_idxs", np.asarray(g.gather_idxs))
    put(f"{nm}:gather_mask", np.asarray(g.gather_mask))
for k in ("ref_pos", "ref_space_uid", "ref_mask", "ref_element", "ref_charge",
          "ref_atom_name_chars"):
    put(k, batch[k])
put("seq_mask", fb.token_features.mask)
# The clean structure too, so the comparison can be scored as CHEMISTRY and not
# only as a residual: a denoiser that agrees to 1e-5 on noise is the thing this
# whole dump exists to get past.
put("clean", clean)
o["output"] = {"shape": list(x_got.shape), "data": x_got.astype(np.float32).ravel().tolist()}
p = a.out % (a.model, a.noise)
open(p, "w").write(json.dumps(o))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

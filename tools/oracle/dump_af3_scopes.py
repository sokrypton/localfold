"""Every haiku scope's OUTPUT in one denoise step, as a summary plus (optionally)
the tensor itself.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_scopes.py boltz2 [--capture=<regex>]

🔴 A SEAM DUMP LOCALISES TO A STAGE; THIS LOCALISES TO A MODULE. boltz2's
conditioning single reads 6.76e-1 against the reference with ours 1.65x too
large, which is one of four things - the initial norm, the initial projection,
the noise embedding or the two transitions - and the seam between them has no
name in `dump_af3_denoise_stages.py`.

`hk.intercept_methods` sees every `hk.Module.__call__`, so this needs no
per-model code and no patch that can drift from the module it traces. The
summary is cheap enough to record for all ~105 scopes; --capture= writes the
full tensor for the ones worth comparing element by element.
"""
import os, re, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np, haiku as hk
import denoise_parity as DP
import fold_check
from alphafold3.model import feat_batch
from atom_parity import flat_atom_features

# 🔴 THE CAPTURE REGEX COMES THROUGH THE ENVIRONMENT, NOT argv. Something under
# `fold_check` parses sys.argv with absl, which refuses an unknown flag outright
# ("Unknown command line flag 'capture'") - and stripping it here is too late,
# the import has already run.
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = argv[0] if argv else "boltz2"
CAPTURE = os.environ.get("CAPTURE") or None
NOISE = float(os.environ.get("NOISE", "16.0"))

SCOPES = {}
_seen = {}
def _record(name, a):
    a = np.asarray(a, np.float32)
    # 🔴 KEYED BY NAME AND OVERWRITTEN, because `ours` runs hk.init BEFORE
    # hk.apply and the interceptor sees both - an indexed key put the RANDOM
    # init pass at the bare name and the real one at "#2", which reads as a
    # module whose output is exactly 1.0000 rms (a LayerNorm at init scale).
    # The last write wins and the last write is the apply. `calls` says how many
    # times a name was seen, so a scope genuinely called twice in one pass
    # cannot hide behind this.
    _seen[name] = _seen.get(name, 0) + 1
    key = name
    e = {"shape": list(a.shape), "rms": float(np.sqrt((a.astype(np.float64)**2).mean())),
         "mean": float(a.mean()), "absmax": float(np.abs(a).max()),
         "head": a.ravel()[:16].tolist(), "calls": _seen[name]}
    if CAPTURE and re.search(CAPTURE, key):
        e["data"] = a.ravel().tolist()
    SCOPES[key] = e

def tracer(next_f, args, kwargs, context):
    out = next_f(*args, **kwargs)
    try:
        name = context.module.module_name
        if isinstance(out, (np.ndarray,)) or hasattr(out, "shape"):
            _record(name, out)
    except Exception:
        pass
    return out

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
if MODEL == "boltz2":
    _w = cfg.evoformer.seq_channel
    s449 = s447 = (rng.normal(size=(n_tok, _w)) * 0.5).astype(np.float32)
pos_dense = (rng.normal(size=(n_tok, max_atoms, 3)) * NOISE).astype(np.float32) * feats['mask'][..., None]

_ours = DP.ours
def ours_traced(*a, **k):
    with hk.intercept_methods(tracer):
        return _ours(*a, **k)
x_got = np.asarray(ours_traced(MODEL, cfg, model_dir, fb, pos_dense, NOISE, s447, s, z))
print("tokens", n_tok, "rms %.4f" % float(np.sqrt((x_got**2).mean())),
      "|", len(SCOPES), "scopes")
for k, v in SCOPES.items():
    print("  %-62s %-18s rms %10.4f  mean %9.4f" % (k, v["shape"], v["rms"], v["mean"]))
p = "/tmp/af3-oracle-scopes-%s.json" % MODEL
open(p, "w").write(json.dumps({"model": MODEL, "tokens": n_tok, "scopes": SCOPES}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p)/2**20))

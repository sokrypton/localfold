"""The SAME denoise step as dump_af3_denoise.py, with af3-any-model's own
intermediates recorded alongside the answer.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_denoise_stages.py boltz2

🔴 WHY THIS EXISTS. check-af3-denoise's per-stage arms are all GPU-against-CPU
internal differentials - they say the port agrees with ITSELF, and say nothing
about where it stops agreeing with the reference. boltz2's whole step reads
relRMS 1.60 at an rms ratio of 1.195, which is what two UNCORRELATED tensors
score (sqrt(1 + 1.195^2) = 1.56), so the disagreement is structural and the
per-stage numbers cannot see it.

It monkeypatches the four seams rather than re-deriving them, so what is
recorded is exactly what the shipped head passed between its own stages.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import denoise_parity as DP
import fold_check
from alphafold3.model import feat_batch
from alphafold3.model.network import atom_cross_attention as ACA
from alphafold3.model.network import diffusion_transformer as DT
from alphafold3.model.network import diffusion_head as DH
from atom_parity import flat_atom_features

MODEL = sys.argv[1] if len(sys.argv) > 1 else "boltz2"
NOISE = float(os.environ.get("NOISE", "16.0"))

TRACE = {}
def _put(name, a):
    a = np.asarray(a)
    TRACE[name] = {"shape": list(a.shape), "data": a.astype(np.float32).ravel().tolist()}

# 🔴 THE SEAMS, PATCHED IN PLACE. Each wrapper calls the original and records
# what it returned; none of them recomputes anything, so a wrapper that is
# subtly wrong cannot make the trace disagree with the answer beside it.
_enc, _dec, _tx, _cond = (ACA.atom_cross_att_encoder, ACA.atom_cross_att_decoder,
                          DT.Transformer.__call__, DH.DiffusionHead._conditioning)

def enc(*a, **k):
    out = _enc(*a, **k)
    if k.get("conditioning_only"):
        return out
    _put("encoder.tokenAct", out.token_act)
    _put("encoder.skipConnection", out.skip_connection)
    return out
def dec(*a, **k):
    out = _dec(*a, **k); _put("decoder.update", out); return out
def tx(self, *a, **k):
    out = _tx(self, *a, **k); _put("transformer.out", out)
    # the transformer's own INPUT too - `act` after the encoder and the
    # single-conditioning projection, which is the one tensor no other seam
    # carries and the first place a wrong projection would show.
    _put("transformer.act", k.get("act", a[0] if a else None))
    _put("transformer.singleCond", k["single_cond"])
    _put("transformer.pairCond", k["pair_cond"])
    return out
def cond(self, *a, **k):
    out = _cond(self, *a, **k)
    single, pair = out
    if single is not None: _put("conditioning.single", single)
    if pair is not None: _put("conditioning.pair", pair)
    return out
ACA.atom_cross_att_encoder = enc; ACA.atom_cross_att_decoder = dec
DT.Transformer.__call__ = tx; DH.DiffusionHead._conditioning = cond
# `diffusion_head` imported the two functions by name, so rebinding the module
# attribute alone would leave the head calling the originals.
DH.atom_cross_attention.atom_cross_att_encoder = enc
DH.atom_cross_attention.atom_cross_att_decoder = dec

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
x_got = np.asarray(DP.ours(MODEL, cfg, model_dir, fb, pos_dense, NOISE, s447, s, z))
print("tokens", n_tok, "out", x_got.shape,
      "rms %.4f" % float(np.sqrt((x_got**2).mean())))
for k, v in TRACE.items():
    print("  %-24s %s" % (k, v["shape"]))
out = {"model": MODEL, "tokens": n_tok, "maxAtoms": int(max_atoms), "noise": NOISE,
       "source": "af3-any-model dev/oracles/denoise_parity.ours, seams traced",
       "stages": TRACE}
p = "/tmp/af3-oracle-stages-%s.json" % MODEL
open(p, "w").write(json.dumps(out))
print("wrote", p, "%.1f MB" % (os.path.getsize(p)/2**20))

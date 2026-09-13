"""af3-any-model's token transformer at a TRUNCATED depth, for a bisect.

    cd ~/alphafold3 && SUPERS=1 PYTHONPATH=src:.:dev/oracles \
      ~/venv/bin/python /tmp/dump_af3_tx_supers.py boltz2

🔴 THE ONLY WAY INTO THAT STACK. Its 24 blocks are one `hk.layer_stack`, so the
interceptor `dump_af3_scopes.py` uses fires ONCE for the whole scan and there
are no per-block scopes to compare against - which is exactly the visibility a
transformer 2.15e-1 from the reference needs. Truncating both sides to N super
blocks turns depth into an axis: an error flat in N is outside the block, one
that grows with N is inside it.

The params are sliced on their leading axis to match, because `layer_stack`
takes its trip count from the config and its weights from the array.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import fold_check
from alphafold3.model import feat_batch
from alphafold3.model.network import diffusion_transformer as DT
from alphafold3.model.network import diffusion_head as DH
from atom_parity import flat_atom_features

argv=[a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = argv[0] if argv else "boltz2"
SUPERS = int(os.environ.get("SUPERS", "1"))
NOISE = float(os.environ.get("NOISE", "16.0"))

TRACE = {}
def _put(n, a):
    a = np.asarray(a, np.float32)
    TRACE[n] = {"shape": list(a.shape), "data": a.ravel().tolist()}
_tx = DT.Transformer.__call__
def tx(self, *a, **k):
    out = _tx(self, *a, **k)
    _put("transformer.act", k.get("act", a[0] if a else None))
    _put("transformer.out", out)
    return out
DT.Transformer.__call__ = tx

seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
batch, cfg, model_dir = fold_check._fold_setup(MODEL, seq, None)
full_supers = cfg.heads.diffusion.transformer.num_blocks \
    // cfg.heads.diffusion.transformer.super_block_size
cfg.heads.diffusion.transformer.num_blocks = (
    SUPERS * cfg.heads.diffusion.transformer.super_block_size)
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
    s449 = s447 = (rng.normal(size=(n_tok, c_s)) * 0.5).astype(np.float32)
pos_dense = (rng.normal(size=(n_tok, max_atoms, 3)) * NOISE).astype(np.float32) * feats['mask'][..., None]

# `denoise_parity.ours`, with the transformer's stacked params CUT to SUPERS.
import haiku as hk, jax, jax.numpy as jnp
from alphafold3.model import params as afp
cfg.global_config.bfloat16 = 'none'
full = afp.get_model_haiku_params(model_dir=model_dir)
emb = {'single': jnp.asarray(s), 'pair': jnp.asarray(z), 'target_feat': jnp.asarray(s447)}
def fwd():
    return DH.DiffusionHead(cfg.heads.diffusion, cfg.global_config)(
        positions_noisy=jnp.asarray(pos_dense), noise_level=jnp.asarray(NOISE, jnp.float32),
        batch=fb, embeddings=emb, use_conditioning=True)
f = hk.transform(fwd)
init = f.init(jax.random.PRNGKey(0))
params, unmapped = {}, []
for sc in init:
    params[sc] = {}
    for k in init[sc]:
        tail = sc[len('diffusion_head/'):] if sc.startswith('diffusion_head/') else sc
        key = ('diffuser/~/diffusion_head' if tail in ('~', 'diffusion_head')
               else 'diffuser/~/diffusion_head/' + tail)
        src = full.get(key); v = None if src is None else src.get(k)
        if v is None:
            unmapped.append('%s/%s' % (sc, k)); params[sc][k] = init[sc][k]; continue
        v = np.asarray(v, np.float32)
        want = np.asarray(init[sc][k]).shape
        # 🔴 SLICE, DO NOT RESHAPE. The cut is on the OUTER stack axis only; the
        # inner super_block_size is unchanged, and an assert keeps a silent
        # mis-slice from looking like a depth result.
        if v.shape != want:
            assert v.shape[1:] == want[1:] and want[0] <= v.shape[0], (sc, k, v.shape, want)
            v = v[:want[0]]
        params[sc][k] = v
assert not unmapped, unmapped[:3]
x = np.asarray(f.apply(params, jax.random.PRNGKey(0)))
print("supers %d of %d, out rms %.4f" % (SUPERS, full_supers, float(np.sqrt((x**2).mean()))))
out = {"model": MODEL, "supers": SUPERS, "tokens": n_tok, "maxAtoms": int(max_atoms),
       "noise": NOISE, "stages": TRACE,
       "output": {"shape": list(x.shape), "data": x.astype(np.float32).ravel().tolist()}}
p = "/tmp/af3-oracle-supers%d-%s.json" % (SUPERS, MODEL)
open(p, "w").write(json.dumps(out)); print("wrote", p)

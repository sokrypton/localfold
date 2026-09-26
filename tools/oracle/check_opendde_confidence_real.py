"""af3-any-model's OpenDDE ConfidenceHead on OUR REAL trunk output.

    node tools/gpu-chrome.mjs tools/gpu/fold-opendde.js --target=6mrr \
      --steps=16 --confidence-inputs=1 > /tmp/odd.log
    AF3_SRC=/tmp/af3colab ~/af3_venv/bin/python \
      tools/oracle/check_opendde_confidence_real.py /tmp/odd_inputs.json

🔴 THE ARM `dump_af3_opendde_confidence.py` SAYS IT IS NOT. That dump
SYNTHESISES `atom_to_token_idx`/`atom_to_tokatom_idx` at a fixed slot count and
feeds `s`, `z` as seeded normals - its own header says the head's arithmetic is
what is under test - so the head has never been run on a real trunk's pair
here. It also never passes `extra_pair_bias`, which defaults to None, while a
fold does pass one: an argument neither side supplies is an argument no oracle
can see, which is exactly how ESMFold2's missing `relative_position_encoding`
survived a 1e-7 agreement. See docs/EF2FAST.md.

🔴 AND IT READS OUR INPUTS, NOT ITS OWN. Everything the head takes comes from
the fold: the refined pair read back off the device, the structural-token
target feat and trunk single, the sampled rep coordinates, both atom maps, the
sequence mask and the attention bias.
"""
import os, sys, json
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# 🔴 ONLY THE ORACLE HARNESS GOES ON THE PATH, NOT THE CHECKOUT'S `src`.
# `alphafold3.cpp` is a COMPILED extension: a source tree shadows the installed
# package and the import dies on `No module named 'alphafold3.cpp'`. The venv's
# own alphafold3 already carries `opendde_confidence`; what it lacks is
# `fold_check`, which is the one thing this adds.
ROOT = os.environ.get("AF3_SRC", "/tmp/af3colab")
# ...and the checkout ROOT for its `converters` package, which fold_check
# re-exports `parse_ca` from. Still not `src`.
for entry in (ROOT, ROOT + "/dev/oracles"):
    sys.path.insert(0, entry)
import numpy as np, haiku as hk, jax, jax.numpy as jnp
from alphafold3.model import params as afp
from alphafold3.model.network import opendde_confidence
import fold_check

inputs = json.load(open(sys.argv[1]))
n = inputs["tokens"]
c_z = inputs["channels"]
arr = lambda k, shape, dt=np.float32: np.asarray(inputs[k], dtype=dt).reshape(shape)
s_inputs = arr("singleInputs", (n, -1))
s = arr("single", (n, -1))
z = arr("pair", (n, n, c_z))
rep = arr("coordinates", (n, 3))
a2t = arr("atomToToken", (-1,), np.int32)
a2ta = arr("atomToSlot", (-1,), np.int32)
seq_mask = arr("seqMask", (n,))
bias = arr("extraPairBias", (n, n))
width = s_inputs.shape[-1]
c_s = s.shape[-1]
# 🔴 SAY WHICH BUNDLE, OR THE RESIDUAL CANNOT BE READ. This head is 6.13e-3
# against af3-any-model on int5 and 5.41e-7 on float32 - the same code, the
# same real pair - so a number quoted without its bundle is not a measurement.
bundle = inputs.get("bundle", "UNSTATED - re-run the fold, it carries it now")
print(f"  tokens {n}  c_z {c_z}  c_s {c_s}  target_feat {width}  atoms {a2t.size}")
print(f"  bundle {bundle}")

# ...the harness's own setup, which is where the config and the weight
# directory come from; `_fold_setup` returns both and does it once.
SEQ = os.environ.get("SEQ",
    "GSHMSLFDFFKNKGSAFTPEERSRFSREFHLSREEFIRLLGLSREEFDRLLQEHHHHHH"[:0]
    or "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
_batch, cfg, model_dir = fold_check._fold_setup("opendde", SEQ)
full = afp.get_model_haiku_params(model_dir=model_dir)

def fwd():
    return opendde_confidence.OpenDDEConfidenceHead(c_s, c_z, width, cfg.global_config)(
        jnp.asarray(s_inputs), jnp.asarray(s), jnp.asarray(z), jnp.asarray(rep),
        jnp.asarray(a2t), jnp.asarray(a2ta), jnp.asarray(seq_mask),
        extra_pair_bias=jnp.asarray(bias))

f = hk.transform(fwd)
init = f.init(jax.random.PRNGKey(0))
params, unmapped = {}, []
for scope in init:
    params[scope] = {}
    for leaf in init[scope]:
        src = full.get("diffuser/" + scope)
        value = None if src is None else src.get(leaf)
        if value is None:
            unmapped.append(f"{scope}/{leaf}"); params[scope][leaf] = init[scope][leaf]
        else:
            params[scope][leaf] = np.asarray(value, np.float32)
# 🔴 A HEAD PARTLY AT INIT IS NOISE, NOT A REFERENCE.
assert not unmapped, unmapped[:4]
out = f.apply(params, jax.random.PRNGKey(0))

theirs = {k: np.asarray(v, np.float64) for k, v in out.items() if hasattr(v, "shape")}
for name, value in sorted(theirs.items()):
    print(f"    out.{name:26} {tuple(value.shape)}")
np.savez_compressed(os.environ.get("OUT", "/tmp/opendde_head_real.npz"), **theirs)
# 🔴 AND OURS BESIDE IT, which is the whole point: the head's own oracle agrees
# at 8.46e-7 on SYNTHETIC inputs, and this asks whether the two still agree when
# the pair is a real trunk's.
ours = inputs.get("oursPae")
if ours is not None:
    o = np.asarray(ours, np.float64).reshape(n, n)
    t = theirs["full_pae"]
    rel = float(np.sqrt(((o - t) ** 2).mean()) / np.sqrt((t ** 2).mean()))
    print(f"  PAE, ours vs theirs ON THE SAME REAL PAIR: relRMS {rel:.4e}"
          f"  max|d| {np.abs(o - t).max():.4f}")
    lo = inputs.get("oursPlddt")
    if lo is not None:
        lp = np.asarray(lo, np.float64); tp = theirs["predicted_lddt"]
        m = min(lp.size, tp.size)
        print(f"  pLDDT relRMS {float(np.sqrt(((lp[:m]-tp[:m])**2).mean())/np.sqrt((tp[:m]**2).mean())):.4e}")

p = theirs.get("full_pae")
if p is not None and p.ndim == 2:
    off = ~np.eye(len(p), dtype=bool)
    rough = (np.abs(np.diff(p, axis=0)).mean() + np.abs(np.diff(p, axis=1)).mean()) / 2 / p[off].std()
    ac = np.corrcoef(p[:, :-1].ravel(), p[:, 1:].ravel())[0, 1]
    print(f"  THEIR head on OUR pair: PAE mean {p[off].mean():.3f}"
          f"  roughness {rough:.4f}  lag-1 {ac:.4f}")

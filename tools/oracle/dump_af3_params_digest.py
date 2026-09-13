"""A digest of every tensor af3-any-model would load, for a bundle to be held to.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles ~/venv/bin/python \
      /tmp/dump_af3_params_digest.py boltz2

🔴 A BUNDLE IS NOT THE REFERENCE'S WEIGHTS UNTIL SOMETHING SAYS SO. boltz2's
`diffusion_embed_pair_offsets` is the NEGATIVE of the one af3-any-model loads -
a converter fix that landed after this bundle was exported - and no gate here
could see it: every checker feeds one bundle to both sides, so a sign in the
weights is invisible to a comparison of two forwards. It cost a day's bisection
down to four thousand atom pairs, and it would have been one line of this.

Shapes and values, not a hash: a hash says "different" where a per-tensor rms
and a sign say "this tensor, negated".
"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for e in ("/home/ubuntu/alphafold3/src","/home/ubuntu/alphafold3","/home/ubuntu/alphafold3/dev/oracles"):
    sys.path.insert(0, e)
import numpy as np
import fold_check
from alphafold3.model import params as afp

MODEL = sys.argv[1] if len(sys.argv) > 1 else "boltz2"
# The real fixture sequence: `_fold_setup` featurises before it resolves the
# directory, and a four-residue stand-in falls off the template alphabet.
seq, _ = fold_check.parse_ca(os.path.expanduser("~/6MRR.pdb"))
_, _, model_dir = fold_check._fold_setup(MODEL, seq, None)
full = afp.get_model_haiku_params(model_dir=model_dir)
out = {}
for scope, leaves in full.items():
    for leaf, value in leaves.items():
        a = np.asarray(value, np.float32)
        out["%s/%s" % (scope, leaf)] = {
            "shape": list(a.shape),
            "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
            "sum": float(a.astype(np.float64).sum()),
            "head": a.ravel()[:8].tolist(),
        }
print(MODEL, len(out), "tensors")
p = "/tmp/af3-params-digest-%s.json" % MODEL
open(p, "w").write(json.dumps({"model": MODEL, "tensors": out}))
print("wrote", p, "%.1f MB" % (os.path.getsize(p) / 2 ** 20))

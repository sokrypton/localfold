"""AF2's fused triangle multiplication on the toy pair, as a reference.

Everything else in an evoformer block passes weights through untouched; the
triangle multiplication is the only one this repository RESHAPES, so it is what
is left after the embedder came out exact and the outer-product ordering was
ruled out. This computes what AF2 would produce, for LocalFold's own CPU
reference to be checked against.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, "tools")
from convert_multimer_params import load_params

E = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
P = load_params(pathlib.Path("/Users/mini/Documents/GitHub/af-params/params_model_1_multimer_v3.npz"))
stage = json.load(open("toy-oracle-stages.json"))
L = 16
z = np.array(stage["embedder_pair"], np.float64).reshape(L, L, 128)
mask = np.ones((L, L))

BLOCK, C = 0, 128
base = E + "triangle_multiplication_outgoing/"
def norm(p, x):
    s, o = p["scale"][BLOCK], p["offset"][BLOCK]
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * s + o
def linear(p, x):
    return x @ p["weights"][BLOCK] + p["bias"][BLOCK]
sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = norm(P[base + "left_norm_input"], z)
proj = mask[..., None] * linear(P[base + "projection"], x)
proj = proj * sigmoid(linear(P[base + "gate"], x))
act = np.einsum("ikc,jkc->ijc", proj[:, :, :C], proj[:, :, C:])
act = norm(P[base + "center_norm"], act)
act = linear(P[base + "output_projection"], act)
act = act * sigmoid(linear(P[base + "gating_linear"], x))

print(f"AF2 triangle-outgoing on the toy pair: |x| {np.abs(act).mean():.4f} max {np.abs(act).max():.3f}")
json.dump({"z": z.ravel().tolist(), "expected": act.ravel().tolist(), "length": L},
          open("toy-triangle.json", "w"))
print("wrote toy-triangle.json")

"""Run AF2-multimer on a toy complex, on CPU, and dump inputs and outputs.

The point is a ground truth LocalFold can be fed EXACTLY: not "build the same
features and hope", but the model input dict itself, so any difference that
shows up afterwards is in the forward pass rather than in the featurisation.
"""
import os, sys, json

# 🔴 DERIVED FROM THIS FILE, NOT WRITTEN OUT. The default output path used to be
# the absolute path of one checkout, so renaming the folder sent the dump
# somewhere that did not exist. ORACLE_OUT still overrides it.
REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/Users/mini/Documents/GitHub/ColabDesign2")
import numpy as np
import jax

from colabdesign2 import parse_contigs
from colabdesign2.af2 import featurize, register_losses
register_losses()
from colabdesign2.af2.runner import AF2Runner

PARAMS = "/Users/mini/Documents/GitHub/af-params/oracle"
LENGTH = int(os.environ.get("ORACLE_LENGTH", 8))   # small enough to run on a CPU
COPIES = int(os.environ.get("ORACLE_COPIES", 2))
# ...the same harness serves as the MONOMER control. If our monomer trunk
# differs from JAX by as much as the multimer one does, the fault is in shared
# code and has been there all along.
MODEL = os.environ.get("ORACLE_MODEL", "alphafold2_multimer_v3")
NAME = os.environ.get("ORACLE_NAME", "model_1_multimer_v3")

spec = parse_contigs(":".join([str(LENGTH)] * COPIES)).resolve()
inputs = featurize(spec, chain_break=None)
inputs["opt"] = {"weights": {}, "alpha": 2.0, "temp": 1.0, "soft": 1.0,
                 "hard": 1.0, "dropout": False, "pssm_hard": True,
                 "template": {"rm_ic": False},
                 "con": {"num": 2, "cutoff": 14.0, "binary": False,
                         "seqsep": 9, "num_pos": float("inf")}}

runner = AF2Runner(model_type=MODEL, data_dir=PARAMS,
                   model_names=[NAME], use_bfloat16=False)

rng = np.random.default_rng(0)
seq = np.zeros((1, LENGTH * COPIES, 20), np.float32)
for i in range(LENGTH * COPIES):
    seq[0, i, rng.integers(0, 20)] = 1.0

out = runner.apply({"seq": jax.numpy.asarray(seq)}, inputs, jax.random.PRNGKey(0))

print("output keys:", sorted(out.keys()))
print("representations:", {k: np.asarray(v).shape for k, v in out["representations"].items()})
model_inputs = out["inputs"]
print("\nMODEL INPUTS the forward actually saw:")
def walk(d, prefix=""):
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict): walk(v, prefix + k + "/")
        else:
            a = np.asarray(v)
            if a.ndim > 0: print(f"  {prefix}{k:28s} {str(a.shape):22s} {a.dtype}")
walk(model_inputs)
dg = np.asarray(out["distogram"]["logits"], np.float32)
pl = np.asarray(out["predicted_lddt"]["logits"], np.float32)
ca = np.asarray(out["structure_module"]["final_atom_positions"][:, 1], np.float32)
print("distogram", dg.shape, "plddt", pl.shape, "CA", ca.shape)
bonds = np.linalg.norm(np.diff(ca[:LENGTH], axis=0), axis=-1)
print("chain A CA-CA:", np.round(bonds, 2))
flat = {"seq": seq, "distogram": dg, "plddt": pl, "ca": ca}
def collect(d, prefix=""):
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict): collect(v, prefix + k + ".")
        else:
            a = np.asarray(v)
            if a.ndim > 0 and a.dtype.kind in "fiub": flat["in." + prefix + k] = a
collect(model_inputs)
np.savez("/Users/mini/Documents/GitHub/af-params/oracle/toy_multimer.npz", **flat)
# ...and as JSON, so the browser side can read it without a loader.
wanted = ["msa_feat", "target_feat", "extra_msa_feat", "extra_msa", "extra_has_deletion",
          "extra_deletion_value", "extra_msa_mask", "msa_mask", "residue_index",
          "asym_id", "entity_id", "sym_id", "seq_mask", "residx_atom37_to_atom14",
          "atom37_atom_exists", "msa"]
payload = {"length": LENGTH * COPIES, "copies": COPIES, "chainLength": LENGTH}
for name in wanted:
    payload[name] = np.asarray(model_inputs[name]).astype(np.float64).ravel().tolist()
# ...the trunk's own outputs, which bracket it from the structure module.
reps = out["representations"]
payload["rep_pair"] = np.asarray(reps["pair"], np.float64).ravel().tolist()
payload["rep_msa_first_row"] = np.asarray(reps["msa_first_row"], np.float64).ravel().tolist()
payload["rep_single"] = np.asarray(reps["single"], np.float64).ravel().tolist()
payload["distogram"] = dg.astype(np.float64).ravel().tolist()
payload["plddt_logits"] = pl.astype(np.float64).ravel().tolist()
payload["ca"] = ca.astype(np.float64).ravel().tolist()
payload["shapes"] = {n: list(np.asarray(model_inputs[n]).shape) for n in wanted}
out_path = os.environ.get("ORACLE_OUT", os.path.join(REPO_ROOT, "toy-oracle.json"))
with open(out_path, "w") as fh:
    json.dump(payload, fh)
print("saved toy_multimer.npz and", out_path)

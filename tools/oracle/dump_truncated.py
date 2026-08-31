"""One evoformer block on each side, so a difference has one place to live.

The stacks run under layer_stack, so their per-block outputs are tracers and
cannot be read out. Truncating the model instead makes the FINAL pair the
output of a single block, which is concrete, and the same truncation is easy to
apply on the LocalFold side by handing it one block's weights.
"""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/Users/mini/Documents/GitHub/ColabDesign2")
import numpy as np, jax

from colabdesign2 import parse_contigs
from colabdesign2.af2 import featurize, register_losses
register_losses()
from colabdesign2.af2.runner import AF2Runner

PARAMS = "/Users/mini/Documents/GitHub/af-params/oracle"
LENGTH = int(os.environ.get("ORACLE_LENGTH", 8))
COPIES = int(os.environ.get("ORACLE_COPIES", 2))
MODEL = os.environ.get("ORACLE_MODEL", "alphafold2_multimer_v3")
NAME = os.environ.get("ORACLE_NAME", "model_1_multimer_v3")
MAIN = int(os.environ.get("MAIN_BLOCKS", 1))
EXTRA = int(os.environ.get("EXTRA_BLOCKS", 1))

spec = parse_contigs(":".join([str(LENGTH)] * COPIES)).resolve()
inputs = featurize(spec, chain_break=None)
inputs["opt"] = {"weights": {}, "alpha": 2.0, "temp": 1.0, "soft": 1.0,
                 "hard": 1.0, "dropout": False, "pssm_hard": True,
                 "template": {"rm_ic": False},
                 "con": {"num": 2, "cutoff": 14.0, "binary": False,
                         "seqsep": 9, "num_pos": float("inf")}}

runner = AF2Runner(model_type=MODEL, data_dir=PARAMS,
                   model_names=[NAME], use_bfloat16=False)
cfg = runner._cfg if hasattr(runner, "_cfg") else None
for attr in ("_cfg", "cfg", "config"):
    cfg = getattr(runner, attr, None)
    if cfg is not None:
        break
print("runner config attr:", type(cfg))
ev = cfg.model.embeddings_and_evoformer
ev.evoformer_num_block = MAIN
# ...OPM_FIRST=0 puts the outer product mean back at the end of the block on
# BOTH sides, which separates "our OPM-first path is wrong" from "the converted
# triangle weights are wrong".
# 🔴 MULTIMER HAS template.enabled TRUE. Its template embedder runs even when
# every template is masked off, and its output is ADDED to the pair - so a
# template-free implementation is missing a term the reference always has.
if os.environ.get("TEMPLATES") == "0":
    ev.template.enabled = False
    print("template.enabled = False")
if os.environ.get("OPM_FIRST") == "0":
    ev.evoformer.outer_product_mean.first = False
    print("outer_product_mean.first = False")
ev.extra_msa_stack_num_block = EXTRA
print(f"truncated to main={ev.evoformer_num_block} extra={ev.extra_msa_stack_num_block}")

# ...the block weights are STACKED, and layer_stack checks the stack height
# against the block count, so the params have to be sliced to match.
def slice_blocks(params):
    out = {}
    for name, tensors in params.items():
        keep = MAIN if "evoformer_iteration" in name else (EXTRA if "extra_msa_stack" in name else None)
        if keep is None:
            out[name] = tensors
        else:
            out[name] = {k: v[:keep] for k, v in tensors.items()}
    return out
runner.model_params = [slice_blocks(p) for p in runner.model_params]
print("sliced block weights to match")

rng = np.random.default_rng(0)
seq = np.zeros((1, LENGTH * COPIES, 20), np.float32)
for i in range(LENGTH * COPIES):
    seq[0, i, rng.integers(0, 20)] = 1.0
out = runner.apply({"seq": jax.numpy.asarray(seq)}, inputs, jax.random.PRNGKey(0))
pair = np.asarray(out["representations"]["pair"], np.float64)
print("pair", pair.shape, "|x|", np.abs(pair).mean())
msa_first = np.asarray(out["representations"]["msa_first_row"], np.float64)
print("msa_first_row", msa_first.shape, "|x|", np.abs(msa_first).mean())
json.dump({"truncated_pair": pair.ravel().tolist(),
           "truncated_msa_first_row": msa_first.ravel().tolist(),
           "main": MAIN, "extra": EXTRA},
          open(os.environ.get("TRUNC_OUT",
    "/Users/mini/Documents/GitHub/alphafold2-webgpu/toy-oracle-truncated.json"), "w"))
print("wrote toy-oracle-truncated.json")

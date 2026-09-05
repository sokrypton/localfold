"""The embedder's pair init, computed straight from the inputs and weights.

Interception cannot reach it - it is an inline sum, and the stacks sit inside
remat - but every term is known and every input was dumped, so it can just be
evaluated:

    pair[i,j] = left_single(target)[i] + right_single(target)[j]
              + prev_pos_linear(dgram(prev_pos))       ... prev_pos is zero, so
                                                           no distance bin fires
                                                           and only the bias lands
              + prev_pair_norm(prev_pair)              ... LayerNorm of zeros is
                                                           the offset
              + relative_encoding[i,j]
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, "tools")
from convert_multimer_params import load_params

EVO = "alphafold/alphafold_iteration/evoformer/"
P = load_params(pathlib.Path("/Users/mini/Documents/GitHub/af-params/params_model_1_multimer_v3.npz"))
o = json.load(open("oracle-dumps/toy-oracle.json"))
L = o["length"]

target = np.array(o["target_feat"], np.float64).reshape(L, 20)
target = np.pad(target, [[0, 0], [0, 1]])           # multimer pads trailing
residue = np.array(o["residue_index"], np.float64)
asym = np.array(o["asym_id"], np.float64)
entity = np.array(o["entity_id"], np.float64)
sym = np.array(o["sym_id"], np.float64)

left = target @ P[EVO + "left_single"]["weights"] + P[EVO + "left_single"]["bias"]
right = target @ P[EVO + "right_single"]["weights"] + P[EVO + "right_single"]["bias"]
pair = left[:, None, :] + right[None, :, :]

# prev_pos is all zero: every pseudo-beta distance is 0, below the first edge
# at 3.25 A, so no bin fires and only the bias contributes.
pair = pair + P[EVO + "prev_pos_linear"]["bias"]
# LayerNorm of an all-zero vector is its offset.
pair = pair + P[EVO + "prev_pair_norm"]["offset"]

MAX_REL, MAX_CHAIN = 32, 2
asym_same = asym[:, None] == asym[None, :]
offset = residue[:, None] - residue[None, :]
clipped = np.clip(offset + MAX_REL, 0, 2 * MAX_REL).astype(int)
final_offset = np.where(asym_same, clipped, 2 * MAX_REL + 1)
rel_pos = np.eye(2 * MAX_REL + 2)[final_offset]
entity_same = entity[:, None] == entity[None, :]
rel_sym = sym[:, None] - sym[None, :]
clipped_chain = np.clip(rel_sym + MAX_CHAIN, 0, 2 * MAX_CHAIN).astype(int)
final_chain = np.where(entity_same, clipped_chain, 2 * MAX_CHAIN + 1)
rel_chain = np.eye(2 * MAX_CHAIN + 2)[final_chain]
feat = np.concatenate([rel_pos, entity_same[..., None].astype(float), rel_chain], -1)
W = P[EVO + "~_relative_encoding/position_activations"]
pair = pair + feat @ W["weights"] + W["bias"]

print(f"embedder pair init {pair.shape}  |x| {np.abs(pair).mean():.4f}  max {np.abs(pair).max():.3f}")
json.dump({"embedder_pair": pair.ravel().tolist()},
          open("oracle-dumps/toy-oracle-stages.json", "w"))
print("wrote oracle-dumps/toy-oracle-stages.json")

"""Layer 1: the relative-position encoding, ours against AF2-multimer's.

This is the only place the model is told which chain a residue belongs to, so
if copies come out on top of each other it is the first thing to check. Their
version is a pure function of asym/entity/sym and the residue index, so it can
be compared exactly - real weights, no model, no GPU.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, "tools")
from convert_multimer_params import load_params

EVO = "alphafold/alphafold_iteration/evoformer/"
params = load_params(pathlib.Path("/Users/mini/Documents/GitHub/af-params/params_model_1_multimer_v3.npz"))
W = params[EVO + "~_relative_encoding/position_activations"]["weights"]   # (73, 128)
B = params[EVO + "~_relative_encoding/position_activations"]["bias"]

MAX_REL, MAX_CHAIN = 32, 2

def theirs(residue_index, asym, entity, sym):
    """modules.py _relative_encoding, transcribed."""
    n = len(residue_index)
    asym_same = asym[:, None] == asym[None, :]
    offset = residue_index[:, None] - residue_index[None, :]
    clipped = np.clip(offset + MAX_REL, 0, 2 * MAX_REL)
    final_offset = np.where(asym_same, clipped, 2 * MAX_REL + 1)
    rel_pos = np.eye(2 * MAX_REL + 2)[final_offset]                     # (n, n, 66)
    entity_same = entity[:, None] == entity[None, :]
    rel_sym = sym[:, None] - sym[None, :]
    clipped_chain = np.clip(rel_sym + MAX_CHAIN, 0, 2 * MAX_CHAIN)
    final_chain = np.where(entity_same, clipped_chain, 2 * MAX_CHAIN + 1)
    rel_chain = np.eye(2 * MAX_CHAIN + 2)[final_chain]                  # (n, n, 6)
    feat = np.concatenate([rel_pos, entity_same[..., None].astype(float), rel_chain], -1)
    assert feat.shape[-1] == 73, feat.shape
    return feat @ W + B

def ours(residue_index, asym, entity, sym):
    """A mirror of src/multimer/input-embedder.js PAIR_SHADER's three lookups."""
    n = len(residue_index)
    out = np.empty((n, n, W.shape[1]), np.float64)
    for i in range(n):
        for j in range(n):
            asym_same = asym[i] == asym[j]
            entity_same = entity[i] == entity[j]
            raw = int(residue_index[i]) - int(residue_index[j]) + MAX_REL
            clipped = min(max(raw, 0), 2 * MAX_REL)
            offset_row = clipped if asym_same else 2 * MAX_REL + 1
            entity_row = 2 * MAX_REL + 2
            delta = int(sym[i]) - int(sym[j])
            clipped_chain = min(max(delta + MAX_CHAIN, 0), 2 * MAX_CHAIN)
            chain_row = clipped_chain if entity_same else 2 * MAX_CHAIN + 1
            value = B + W[offset_row] + W[entity_row + 1 + chain_row]
            if entity_same:
                value = value + W[entity_row]
            out[i, j] = value
    return out

def case(name, copies, chain_len=6, plus200=True):
    n = copies * chain_len
    residue = np.zeros(n, int); asym = np.zeros(n, int); entity = np.zeros(n, int); sym = np.zeros(n, int)
    r = 0
    for c in range(copies):
        for k in range(chain_len):
            residue[r] = r + (c * 200 if plus200 else 0)
            asym[r] = c; entity[r] = 0; sym[r] = c
            r += 1
    a, b = theirs(residue, asym, entity, sym), ours(residue, asym, entity, sym)
    delta = np.abs(a - b).max()
    # ...float32 weights summed in a different order: a matmul over 73 one-hot
    # features against three table lookups. 1e-5 is the float32 floor here, not
    # a tolerance for being roughly right.
    print(f"  {name:34s} maxabs {delta:.3e}  {'MATCH' if delta < 1e-5 else '🔴 DIFFER'}")
    return residue, asym, entity, sym

print("relative encoding, ours vs AF2-multimer's, homo-oligomers:")
for copies in (1, 2, 3, 4, 6):
    case(f"{copies} cop{'y' if copies == 1 else 'ies'} (+200 residue breaks)", copies)
print()
print("...and with AF2-multimer's own residue numbering (no +200):")
for copies in (2, 3, 6):
    case(f"{copies} copies (per-chain numbering)", copies, plus200=False)

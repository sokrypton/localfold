#!/usr/bin/env python3
"""AF2-multimer's template embedder, evaluated for a fully-masked template.

This is the term LocalFold omits. With every template masked off it is far
smaller than the module count suggests: the distogram, the unit vectors and the
backbone mask are all zero, so `construct_input` collapses to

    act = sum(biases) + W_aatype_row[0] + W_aatype_col[0]
        + W_query . LayerNorm(pair)

and the rest is two ordinary pair blocks, a layer norm, a relu and a projection.
Writing it here first gives the GPU implementation an exact target instead of a
plausible one.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from convert_multimer_params import load_params

T = ("alphafold/alphafold_iteration/evoformer/template_embedding/")
S = T + "single_template_embedding/"
IT = S + "template_embedding_iteration/"

sigmoid = lambda x: 1 / (1 + np.exp(-x))
def norm(p, x, block=None):
    scale = p["scale"] if block is None else p["scale"][block]
    offset = p["offset"] if block is None else p["offset"][block]
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * scale + offset
def linear(p, x, block=None):
    w = p["weights"] if block is None else p["weights"][block]
    b = p["bias"] if block is None else p["bias"][block]
    return x @ w + b

def triangle(P, base, act, mask, block, equation):
    C = P[base + "projection"]["weights"].shape[-1] // 2
    x = norm(P[base + "left_norm_input"], act, block)
    proj = mask[..., None] * linear(P[base + "projection"], x, block)
    proj = proj * sigmoid(linear(P[base + "gate"], x, block))
    out = np.einsum(equation, proj[:, :, :C], proj[:, :, C:])
    out = norm(P[base + "center_norm"], out, block)
    out = linear(P[base + "output_projection"], out, block)
    return out * sigmoid(linear(P[base + "gating_linear"], x, block))

def triangle_attention(P, base, act, mask, block, ending):
    a = P[base + "/attention"]
    x = norm(P[base + "/query_norm"], act, block)
    if ending:
        x = x.transpose(1, 0, 2)
        m = mask.T
    else:
        m = mask
    q = np.einsum("ijc,chd->ijhd", x, a["query_w"][block])
    k = np.einsum("ijc,chd->ijhd", x, a["key_w"][block])
    v = np.einsum("ijc,chd->ijhd", x, a["value_w"][block])
    bias = np.einsum("ijc,ch->ijh", norm(P[base + "/query_norm"], act, block)
                     if False else x, P[base]["feat_2d_weights"][block])
    logits = np.einsum("iqhd,ikhd->ihqk", q, k) / np.sqrt(q.shape[-1])
    logits = logits + 1e9 * (m[:, None, None, :] - 1.0)
    logits = logits + bias.transpose(0, 2, 1)[:, :, None, :]
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    out = np.einsum("ihqk,ikhd->iqhd", weights, v)
    gate = sigmoid(np.einsum("ijc,chd->ijhd", x, a["gating_w"][block]) + a["gating_b"][block])
    out = out * gate
    out = np.einsum("iqhd,hdc->iqc", out, a["output_w"][block]) + a["output_b"][block]
    return out.transpose(1, 0, 2) if ending else out

def transition(P, base, act, block):
    x = norm(P[base + "/input_layer_norm"], act, block)
    x = np.maximum(linear(P[base + "/transition1"], x, block), 0)
    return linear(P[base + "/transition2"], x, block)

def template_embedding(P, pair, mask, aatype_index=0, num_templates=1):
    L, C = pair.shape[0], 64
    act = np.zeros((L, L, C))
    for i in (0, 1, 4, 5, 6, 7):                      # zero inputs: bias only
        act = act + P[S + f"template_pair_embedding_{i}"]["bias"]
    onehot = np.zeros(22); onehot[aatype_index] = 1
    act = act + onehot @ P[S + "template_pair_embedding_2"]["weights"] \
              + P[S + "template_pair_embedding_2"]["bias"]
    act = act + onehot @ P[S + "template_pair_embedding_3"]["weights"] \
              + P[S + "template_pair_embedding_3"]["bias"]
    act = act + linear(P[S + "template_pair_embedding_8"],
                       norm(P[S + "query_embedding_norm"], pair))

    for block in range(P[IT + "pair_transition/transition1"]["weights"].shape[0]):
        act = act + triangle(P, IT + "triangle_multiplication_outgoing/", act, mask, block, "ikc,jkc->ijc")
        act = act + triangle(P, IT + "triangle_multiplication_incoming/", act, mask, block, "kjc,kic->ijc")
        act = act + triangle_attention(P, IT + "triangle_attention_starting_node", act, mask, block, False)
        act = act + triangle_attention(P, IT + "triangle_attention_ending_node", act, mask, block, True)
        act = act + transition(P, IT + "pair_transition", act, block)

    act = norm(P[S + "output_layer_norm"], act)
    act = np.maximum(act / num_templates, 0)
    return linear(P[T + "output_linear"], act)


def main() -> int:
    P = load_params(pathlib.Path("/Users/mini/Documents/GitHub/af-params/params_model_1_multimer_v3.npz"))
    stage = json.load(open("toy-oracle-stages.json"))
    L = 16
    pair = np.array(stage["embedder_pair"], np.float64).reshape(L, L, 128)
    out = template_embedding(P, pair, np.ones((L, L)))
    print(f"template term {out.shape}  |x| {np.abs(out).mean():.4f}  max {np.abs(out).max():.3f}")
    json.dump({"template_term": out.ravel().tolist(), "length": L},
              open("toy-template.json", "w"))
    print("wrote toy-template.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""MSA Pairformer's weights for the native port (cuda/msap), in cuda/af3's model.idx/model.bin format.

    ~/venv_msap/bin/python tools/export_msap.py [cuda/msap/weights] [--weights=~/msap/weights]

The checkpoint is yakiyama/MSA-Pairformer's (model_cuex.bin, contact.bin, confind_contact.bin; bfloat16 but the
contact heads), every tensor written float32 - the widening is exact. The modules MSA Pairformer shares with
ESMFold2's MSA encoder take ESMFold2's names and layouts (cuda/ef2's kernels read them as they are), so three
conversions happen here and nowhere else:
  - every linear input-major ([in][out]: cuda/ef2's gemm), torch's being [out][in];
  - every SwiGLU's halves swapped: MSA Pairformer gates the SECOND half (x, g = chunk(2); silu(g) x) where cuda/ef2's
    swigluK gates the first (silu(h[:F]) h[F:]);
  - a triangle's projection and gate interleaved (column 2c is a's channel c, 2c + 1 b's: triSplitInterleavedK),
    the incoming one's a and b swapped (its contraction is the transpose of cuda/ef2's).
Entries (f/ prefix, as cuda/ef2's F() reads them):
  msap/relpos [139][256], msap/bond [256], msap/msaInit [28][464]
  msap/blocks/<l>/pwa/...             normScale, normOffset, value, gate, pairNormScale, pairNormOffset, pairLogits, output
  msap/blocks/<l>/msaTransition/...   inputLayerNormScale, inputLayerNormOffset, transition1, transition2
  msap/blocks/<l>/opm/...             the query-biased outer product (normScale/Offset, q, k, qNormScale/Offset,
                                      kNormScale/Offset, lambdaQ1/K1/Q2/K2, left, right, toPair, toPairBias)
  msap/blocks/<l>/triangleMultiplicationOutgoing|Incoming/..., msap/blocks/<l>/pairTransition/...   (cuda/ef2's trunkBlock)
  msap/finalPwa/..., msap/finalMsaTransition/..., msap/lm/..., msap/cbContact/..., msap/confindContact/...
  meta: msaHeads 8, msaHeadWidth 32, depth 22, contactLayer 15, confindContactLayer 18, opmLambdaInit/<l>
"""
import glob, math, os, sys
import numpy as np
import torch

args = [x for x in sys.argv[1:] if not x.startswith("--")]
opts = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
out_dir = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "cuda", "msap", "weights")
src = os.path.expanduser(opts.get("weights", "~/msap/weights"))
sd = {}
for n in ("model_cuex.bin", "contact.bin", "confind_contact.bin"):
    hits = glob.glob(f"{src}/*/snapshots/*/{n}") or glob.glob(f"{src}/{n}")
    if not hits: sys.exit(f"no {n} under {src} (MSAPairformer.from_pretrained(weights_dir=...) fetches it)")
    sd.update(torch.load(hits[0], map_location="cpu", weights_only=True))
used = set()
def t(name):
    used.add(name)
    return sd[name].float().numpy()
def lin(name): return t(name).T                                  # [out][in] -> [in][out]
def swap(w):                                                     # SwiGLU halves along the output axis (last)
    h = w.shape[-1] // 2
    return np.concatenate([w[..., h:], w[..., :h]], axis=-1)

entries = []
def put(name, arr): entries.append(("f/" + name, np.ascontiguousarray(arr, dtype=np.float32)))
meta = {}

C, CM, HEADS, HW = 256, 464, 8, 32
put("msap/relpos", lin("relative_position_encoding.out_embedder.weight"))
put("msap/bond", t("token_bond_to_pairwise_feat.1.weight")[:, 0])
put("msap/msaInit", lin("msa_init_proj.weight"))

def pwa(prefix, out):
    w = t(prefix + "msa_to_values_and_gates.1.weight")              # [2 * heads * d][464]: values, then gates
    put(out + "normScale", t(prefix + "msa_to_values_and_gates.0.weight"))
    put(out + "normOffset", t(prefix + "msa_to_values_and_gates.0.bias"))
    put(out + "value", w[: HEADS * HW].T); put(out + "gate", w[HEADS * HW:].T)
    put(out + "pairNormScale", t(prefix + "pairwise_repr_to_attn.0.weight"))
    put(out + "pairNormOffset", t(prefix + "pairwise_repr_to_attn.0.bias"))
    put(out + "pairLogits", lin(prefix + "pairwise_repr_to_attn.1.weight"))
    put(out + "output", lin(prefix + "to_out.1.weight"))
def transition(prefix, out):                                    # PreLayerNorm(Transition): norm, fn.ff.0/2
    put(out + "inputLayerNormScale", t(prefix + "norm.weight")); put(out + "inputLayerNormOffset", t(prefix + "norm.bias"))
    put(out + "transition1", swap(lin(prefix + "fn.ff.0.weight"))); put(out + "transition2", lin(prefix + "fn.ff.2.weight"))
def triangle(prefix, out, incoming=False):
    # MSA Pairformer's incoming is sum_k a[k,i] b[k,j] where cuda/ef2's is sum_k a[k,j] b[k,i]: a and b swapped
    def inter(w):                                                # [2C][C] (a's rows, then b's) -> [C][2C] interleaved
        a, b = w[:C].T, w[C:].T
        if incoming: a, b = b, a
        x = np.empty((w.shape[1], 2 * C), np.float32); x[:, 0::2] = a; x[:, 1::2] = b
        return x
    put(out + "leftNormInputScale", t(prefix + "norm_in.weight")); put(out + "leftNormInputOffset", t(prefix + "norm_in.bias"))
    put(out + "projection", inter(t(prefix + "p_in.weight"))); put(out + "gate", inter(t(prefix + "g_in.weight")))
    put(out + "centerNormScale", t(prefix + "norm_out.weight")); put(out + "centerNormOffset", t(prefix + "norm_out.bias"))
    put(out + "outputProjection", lin(prefix + "p_out.weight")); put(out + "gatingLinear", lin(prefix + "g_out.weight"))

depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("core_stack.layers."))
for l in range(depth):
    P, O = f"core_stack.layers.{l}.", f"msap/blocks/{l}/"
    pwa(P + "0.", O + "pwa/")
    transition(P + "1.", O + "msaTransition/")
    q = P + "2.opm."
    put(O + "opm/normScale", t(q + "norm.weight")); put(O + "opm/normOffset", t(q + "norm.bias"))
    put(O + "opm/q", lin(q + "q_proj.weight")); put(O + "opm/k", lin(q + "k_proj.weight"))
    for nm in ("q", "k"):
        put(O + f"opm/{nm}NormScale", t(q + f"{nm}_norm.weight")); put(O + f"opm/{nm}NormOffset", t(q + f"{nm}_norm.bias"))
    for nm in ("q1", "k1", "q2", "k2"): put(O + f"opm/lambda{nm.upper()}", t(q + f"lambda_{nm}"))
    put(O + "opm/left", lin(q + "to_left_hidden.weight")); put(O + "opm/right", lin(q + "to_right_hidden.weight"))
    put(O + "opm/toPair", swap(lin(q + "to_pairwise_repr.weight"))); put(O + "opm/toPairBias", swap(t(q + "to_pairwise_repr.bias")))
    # lambda_init: CoreModule's 0.8 - 0.6 exp(-0.3 l), made a bfloat16 tensor there - so rounded as bfloat16 here
    meta[f"opmLambdaInit/{l}"] = float(torch.tensor(0.8 - 0.6 * math.exp(-0.3 * l), dtype=torch.bfloat16).float())
    triangle(P + "3.tri_mult_outgoing.", O + "triangleMultiplicationOutgoing/")
    triangle(P + "3.tri_mult_incoming.", O + "triangleMultiplicationIncoming/", incoming=True)
    transition(P + "3.pairwise_transition.", O + "pairTransition/")
pwa("core_stack.final_msa_pwa.", "msap/finalPwa/")
transition("core_stack.final_msa_transition.", "msap/finalMsaTransition/")
put("msap/lm/initNormScale", t("lm_head.init_ln.weight")); put("msap/lm/initNormOffset", t("lm_head.init_ln.bias"))
put("msap/lm/dense", swap(lin("lm_head.dense.weight"))); put("msap/lm/denseBias", swap(t("lm_head.dense.bias")))
put("msap/lm/preLogitNormScale", t("lm_head.pre_logit_norm.weight")); put("msap/lm/preLogitNormOffset", t("lm_head.pre_logit_norm.bias"))
put("msap/lm/logits", lin("lm_head.weight")); put("msap/lm/logitsBias", t("lm_head.bias"))
for head, out in (("contact_head.", "msap/cbContact/"), ("confind_contact_head.", "msap/confindContact/")):
    put(out + "normScale", t(head + "init_ln.weight")); put(out + "normOffset", t(head + "init_ln.bias"))
    put(out + "weight", t(head + "weight")[0]); put(out + "bias", t(head + "bias"))
left = sorted(set(sd) - used)
if left: sys.exit(f"{len(left)} checkpoint tensors not exported, e.g. {left[:4]}")

meta.update({"msaHeads": HEADS, "msaHeadWidth": HW, "depth": depth, "contactLayer": 15, "confindContactLayer": 18,
             "pairChannels": C, "msaChannels": CM})
os.makedirs(out_dir, exist_ok=True)
lines = []
with open(os.path.join(out_dir, "model.bin"), "wb") as f:
    for name, arr in entries:
        lines.append(f"t {name} {f.tell() // 4} {arr.size}")         # (in 4-byte elements: model.idx's unit)
        for k, dim in enumerate(arr.shape): lines.append(f"m {name}#{k} {dim}")
        lines.append(f"m {name}#r {arr.ndim}")
        f.write(arr.tobytes())
lines += [f"m meta/{k} {v!r}" for k, v in meta.items()]
with open(os.path.join(out_dir, "model.idx"), "w") as f: f.write("\n".join(lines) + "\n")
print(f"{out_dir}: {len(entries)} tensors, {sum(a.size for _, a in entries) / 1e6:.1f} M parameters, depth {depth}")

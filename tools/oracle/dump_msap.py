#!/usr/bin/env python3
"""MSA Pairformer's reference tensors for the native port (cuda/msap), in cuda/af3's model.idx/model.bin format.

    ~/venv_msap/bin/python tools/oracle/dump_msap.py <alignment.a3m> <out dir> [--depth=512] [--chain-breaks=265]
        [--layers=0,1,7,15,18,21] [--hhfilter=~/hhsuite/bin/hhfilter]

The reference is yoakiyama/MSA_Pairformer (pip msa-pairformer, weights yakiyama/MSA-Pairformer) run in float32 with
the plain PyTorch triangle multiplication (cuEquivariance's Triton kernel off), so a native float32 port can be held
to it; the README's own bfloat16 autocast is ~5e-3 from it on contacts (1B70). The alignment is subsampled exactly as
the README does (hhfilter -id 90 -diff N -cov 70 -qid 30 -qsc -20, then greedy to N), and the subsample's TOKENS are
written, so the port reads the same rows rather than repeating the selection.

Entries ('t' f32 tensors, 'i' int32, 'm' numbers), all under o/:
  o/tokens [S][L] (i)          the subsampled, tokenised alignment (28-letter alphabet, aa2tok_d)
  o/chainBreaks [k] (i)        complex_chain_break_indices
  o/pairInit [L][L][256]       the relative-position encoding plus the bond feature
  o/msaInit [S][L][464]        the projected one-hot
  o/layer0/<stage>             every module of layer 0: afterPwa, afterMsaTransition (MSA) and seqWeights [S],
                               afterOpm, afterTriOut, afterTriIn, afterPairTransition (pair)
  o/layer<k>/pair, o/layer<k>/msa   after layer k, for --layers
  o/layer<k>/seqWeights [S]    the outer product's query-biased sequence weights, every layer
  o/final/msaQuery [L][464]    after the final MSA update, the query row
  o/logits [L][26]             the LM head over the query row
  o/cbContacts, o/confindContacts [L][L]
"""
import argparse, os, sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("alignment"); p.add_argument("out")
p.add_argument("--depth", type=int, default=512)
p.add_argument("--chain-breaks", default="")
p.add_argument("--layers", default="0,1,7,15,18,21")
p.add_argument("--hhfilter", default=os.path.expanduser("~/hhsuite/bin/hhfilter"))
p.add_argument("--weights", default=os.path.expanduser("~/msap/weights"))
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d
from MSA_Pairformer.pairwise_operations import TriangleMultiplication

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda:0")
model = MSAPairformer.from_pretrained(weights_dir=a.weights, device=dev).eval().float()
for mod in model.modules():                         # the plain PyTorch triangle, f32 throughout
    if isinstance(mod, TriangleMultiplication): mod.use_cuequivariance = False

np.random.seed(a.seed)
msa = MSA(msa_file_path=a.alignment, max_seqs=a.depth, max_length=10240, max_tokens=np.inf,
          diverse_select_method="hhfilter", hhfilter_kwargs={"binary": a.hhfilter})
tok = msa.diverse_tokenized_msa                     # [S][L]
S, L = tok.shape
breaks = [int(x) for x in a.chain_breaks.split(",") if x != ""]
oh = torch.nn.functional.one_hot(tok, num_classes=len(aa2tok_d)).unsqueeze(0).float().to(dev)
mask, msa_mask, full_mask, pairwise_mask = [m.to(dev) for m in prepare_msa_masks(tok.unsqueeze(0))]

out = {}
def put(name, x):
    out[name] = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

# layer 0, module by module (the stack's own loop, unrolled for its first layer)
layers = [int(x) for x in a.layers.split(",") if x != ""]
cb = model.contact_layer; cf = model.confind_contact_layer
with torch.no_grad():
    m, z = model.init_representations(oh, [breaks] if breaks else None)
    put("o/msaInit", m[0]); put("o/pairInit", z[0])
    stack = model.core_stack
    for li, (pwa, mtr, opm, pblock) in enumerate(stack.layers):
        m = m + pwa(msa=m, pairwise_repr=z, mask=mask, full_mask=full_mask, pairwise_mask=pairwise_mask)
        if li == 0: put("o/layer0/afterPwa", m[0])
        m = m + mtr(m)
        if li == 0: put("o/layer0/afterMsaTransition", m[0])
        upd, w = opm(msa=m, mask=mask, msa_mask=msa_mask, full_mask=full_mask, pairwise_mask=pairwise_mask, seq_weights=None)
        z = z + upd
        put(f"o/layer{li}/seqWeights", w[0])
        if li == 0: put("o/layer0/afterOpm", z[0])
        if li == 0:
            z = pblock.tri_mult_outgoing(z, pairwise_mask=pairwise_mask) + z; put("o/layer0/afterTriOut", z[0])
            z = pblock.tri_mult_incoming(z, pairwise_mask=pairwise_mask) + z; put("o/layer0/afterTriIn", z[0])
            z = pblock.pairwise_transition(z) + z; put("o/layer0/afterPairTransition", z[0])
        else:
            z = pblock(pairwise_repr=z, pairwise_mask=pairwise_mask)
        if li in layers: put(f"o/layer{li}/pair", z[0]); put(f"o/layer{li}/msa", m[0])
        if li == cb: put("o/cbContacts", model.contact_head(z)[0])
        if li == cf: put("o/confindContacts", model.confind_contact_head(z)[0])
    m = m + stack.final_msa_pwa(msa=m, pairwise_repr=z, mask=mask, full_mask=full_mask, pairwise_mask=pairwise_mask)
    m = m + stack.final_msa_transition(m)
    put("o/final/msaQuery", m[0, 0])
    put("o/logits", model.lm_head(m[:, :1])[0, 0])

    # the same through the model's own forward, which the unrolled loop must reproduce
    r = model(msa=oh, mask=mask, msa_mask=msa_mask, full_mask=full_mask, pairwise_mask=pairwise_mask,
              complex_chain_break_indices=[breaks] if breaks else None, return_cb_contacts=True, return_confind_contacts=True)
    for k, mine in (("predicted_cb_contacts", "o/cbContacts"), ("predicted_confind_contacts", "o/confindContacts")):
        d = np.abs(r[k][0].float().cpu().numpy() - out[mine]).max()
        print(f"unrolled vs forward {k}: max |d| {d:.2e}")
        assert d < 1e-5, "the unrolled loop is not the model's forward"

os.makedirs(a.out, exist_ok=True)
lines = []
with open(os.path.join(a.out, "model.bin"), "wb") as f:
    def tensor(name, arr, kind):
        arr = np.ascontiguousarray(arr, dtype=np.int32 if kind == "i" else np.float32)
        lines.append(f"{kind} {name} {f.tell() // 4} {arr.size}")   # (in 4-byte elements: model.idx's unit)
        for k, dim in enumerate(arr.shape): lines.append(f"m {name}#{k} {dim}")
        lines.append(f"m {name}#r {arr.ndim}")
        f.write(arr.tobytes())
    tensor("o/tokens", tok.numpy(), "i")
    tensor("o/chainBreaks", np.array(breaks or [-1]), "i")
    for name in sorted(out): tensor(name, out[name], "t")
lines += [f"m o/meta/depth {S}", f"m o/meta/length {L}"]
with open(os.path.join(a.out, "model.idx"), "w") as f: f.write("\n".join(lines) + "\n")
print(f"{a.out}: {S} rows x {L} columns, {len(out)} tensors, {os.path.getsize(os.path.join(a.out, 'model.bin')) / 1e6:.0f} MB")

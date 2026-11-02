#!/usr/bin/env python3
"""MSA Pairformer's conservation and categorical Jacobian from the reference, for cuda/msap's --conservation and --jacobian.

    ~/venv_msap/bin/python tools/oracle/dump_msap_variants.py <subsample.a3m> <out.npz> [--chain-breaks=i,j]
        [--jacobian=ARNDCEQGHILKMFPSTWYV] [--weights=~/msap/weights] [--bf16]

The rows are read as they are - cuda/msap's own subsample.a3m - so the two sides see the same alignment and the
comparison is of the model, not the subsampler. The procedure is the authors' notebook's (MSA_Pairformer_with_MMseqs2.ipynb,
run_msa_analysis): a first pass with the query biasing on for the per-layer sequence weights, then every variant with the
biasing off and those weights fixed (fix_weights); the query row's logits over the 20 amino acids. Float32 (TF32 off, the
plain PyTorch triangle) unless --bf16 (the notebook's own autocast).
Writes conservation [L][20], jacobian [L][K][L][20] (base minus mutant; the notebook copies the base where the mutant is
the wild type and every amino acid is asked for), jacobian_contacts [L][L] (the notebook's jac_to_con, symm_after).
"""
import argparse, os, re
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("alignment"); p.add_argument("out")
p.add_argument("--chain-breaks", default="")
p.add_argument("--jacobian", default="ARNDCEQGHILKMFPSTWYV")
p.add_argument("--weights", default=os.path.expanduser("~/msap/weights"))
p.add_argument("--bf16", action="store_true")
a = p.parse_args()

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import prepare_msa_masks, aa2tok_d
from MSA_Pairformer.pairwise_operations import TriangleMultiplication

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda:0")
model = MSAPairformer.from_pretrained(weights_dir=a.weights, device=dev).eval()
if a.bf16: model = model.to(torch.bfloat16)
else:
    model = model.float()
    for mod in model.modules():
        if isinstance(mod, TriangleMultiplication): mod.use_cuequivariance = False

rows = [re.sub(r"[a-z.*]", "", l.strip()) for l in open(a.alignment) if l.strip() and not l.startswith(">")]
tok = torch.tensor([[aa2tok_d.get(c, aa2tok_d["X"]) for c in r] for r in rows]).long()
S, L = tok.shape
breaks = [int(x) for x in a.chain_breaks.split(",") if x != ""]
oh = torch.nn.functional.one_hot(tok, num_classes=len(aa2tok_d)).unsqueeze(0).float().to(dev)
mask, msa_mask, full_mask, pairwise_mask = [m.to(dev) for m in prepare_msa_masks(tok.unsqueeze(0))]
kw = dict(mask=mask, msa_mask=msa_mask, full_mask=full_mask, pairwise_mask=pairwise_mask,
          complex_chain_break_indices=[breaks] if breaks else None)
dtype = torch.bfloat16 if a.bf16 else torch.float32
ctx = (lambda: torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda")) if a.bf16 else torch.no_grad

with torch.no_grad():
    with ctx():
        first = model(msa=oh.to(dtype), return_seq_weights=True, **kw)
    model.turn_off_query_biasing()
    kw["seq_weights_dict"] = first["seq_weights_list_d"]
    def f(msa):
        with ctx():
            r = model(msa=msa.to(dtype), return_seq_weights=False, query_only=True, **kw)   # (the notebook also passes return_contacts=False, which pip 1.0.2 lacks: heads only)
        return r["logits"][0, 0, :L, :20].float().cpu().numpy()
    cons = np.zeros((L, 20), np.float32)
    for n in range(L):
        h = oh.clone(); h[0, 0, n, :] = 0; h[0, 0, n, aa2tok_d["<mask>"]] = 1
        x = f(h)[n].astype(np.float64); x = np.exp(x - x.max()); cons[n] = x / x.sum()
        print(f"\rconservation {n + 1}/{L}", end="", flush=True)
    print()
    fx = f(oh)
    muts = [aa2tok_d[c] for c in a.jacobian]
    every = a.jacobian == "ARNDCEQGHILKMFPSTWYV"
    J = np.zeros((L, len(muts), L, 20), np.float32)
    wt = tok[0].numpy()
    for n in range(L):
        for k, t in enumerate(muts):
            if every and t == wt[n]: continue        # (fx itself)
            h = oh.clone(); h[0, 0, n, :] = 0; h[0, 0, n, t] = 1
            J[n, k] = fx - f(h)
        print(f"\rjacobian {n + 1}/{L}", end="", flush=True)
    print()

def jac_to_con(jac, center=True, diag="remove", apc=True, symm_before=True, symm_after=False):   # (the notebook's)
    X = jac.copy()
    if symm_before: X = X + X.transpose(2, 3, 0, 1)
    if center:
        for i in range(4):
            if X.shape[i] > 1: X -= X.mean(i, keepdims=True)
    contacts = np.sqrt(np.square(X).sum((1, 3)))
    if symm_after: contacts = contacts + contacts.T
    if diag == "remove": np.fill_diagonal(contacts, 0)
    if apc:
        ap = contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
        contacts = contacts - ap
    if diag == "remove": np.fill_diagonal(contacts, 0)
    return contacts

np.savez(a.out, conservation=cons, jacobian=J,
         jacobian_contacts=jac_to_con(J.astype(np.float64), symm_before=False, symm_after=True).astype(np.float32),
         cb_contacts=first["predicted_cb_contacts"][0].float().cpu().numpy())
print(f"{a.out}: {S} rows x {L} columns, jacobian over {a.jacobian}")

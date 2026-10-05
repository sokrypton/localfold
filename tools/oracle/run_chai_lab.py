#!/usr/bin/env python3
# Genuine chai-lab 0.6.1 on one sequence, with the ESM2 embedding handed in (af3-any-model's ESM2 3B, int8 - the
# same tower chai downloads as a 5.7 GB fp16 trace) and every exported module's inputs and outputs recorded.
import sys, os, pathlib, numpy as np, torch
seq, emb_path, out_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
recycles = int(sys.argv[4]) if len(sys.argv) > 4 else 3
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
out_dir.mkdir(parents=True, exist_ok=True)
import chai_lab.data.dataset.embeddings.esm as esmmod
from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
def fake(prot_sequences, device):
    e = torch.tensor(np.load(emb_path), dtype=torch.float32)
    out = {}
    for s in prot_sequences:
        assert s == seq, (s, seq); out[s] = EmbeddingContext(esm_embeddings=e)
    return out
esmmod._get_esm_contexts_for_sequences = fake
import chai_lab.chai1 as c1
calls = {}
def to_cpu(x):
    if torch.is_tensor(x):
        for d, n in enumerate(x.shape):          # the MSA axis (16384 rows, padded): its first 8 rows only
            if n == 16384: x = x.narrow(d, 0, 8)
        return x.detach().float().cpu() if x.is_floating_point() else x.detach().cpu()
    if isinstance(x, (list, tuple)): return type(x)(to_cpu(v) for v in x)
    if isinstance(x, dict): return {k: to_cpu(v) for k, v in x.items()}
    return x
orig = c1.load_exported
class Rec:
    def __init__(self, name, m): self.name, self.m = name, m
    def forward(self, **kw):
        out = self.m.forward(**kw)
        n = calls.setdefault(self.name, 0); calls[self.name] = n + 1
        if n < 3 or self.name == 'trunk.pt':
            torch.save({'inputs': to_cpu({k: v for k, v in kw.items() if k not in ('move_to_device',)}), 'outputs': to_cpu(out)},
                       out_dir / ('%s.%d.pt' % (self.name[:-3], n)))
        return out
c1.load_exported = lambda name, device: Rec(name, orig(name, device))
fasta = out_dir / 'in.fasta'
fasta.write_text('>protein|name=A\n%s\n' % seq)
res = c1.run_inference(fasta_file=fasta, output_dir=out_dir / 'out', num_trunk_recycles=recycles,
                       num_diffn_timesteps=steps, seed=42, device='cuda:0', use_esm_embeddings=True)
print('calls', calls)
for i, (p, s) in enumerate(zip(res.cif_paths, res.aggregate_score.tolist() if hasattr(res, 'aggregate_score') else [None]*5)):
    print(i, p, s)

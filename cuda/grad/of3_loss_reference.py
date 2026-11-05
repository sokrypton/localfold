"""OpenFold3's own losses on test-of3loss's case: openfold3/core/loss (aqlaboratory/openfold-3) imported and called -
not a transcription - with autograd for every gradient, against cuda/grad/of3loss.h's values and gradients.

    PYTHONPATH=<openfold-3 checkout> ~/venv_protenix/bin/python cuda/grad/of3_loss_reference.py <case dir>

The case is in the trainer's dense layout; OpenFold3 reads a flat one whose atoms within a standard token are in its
canonical order (token_atom_constants.TOKEN_NAME_TO_ATOM_NAMES), which its restype-indexed offsets assume. So each
standard token is laid out in that order, an atom the featuriser dropped (a leaving atom) a padding atom with every
mask zero; an atomised token is its one atom. Residue types map by name: the batch's aatype is AF3's alphabet.
"""
import sys

import numpy as np
import torch

from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3 as OF3_RES
from openfold3.core.data.resources.token_atom_constants import TOKEN_NAME_TO_ATOM_NAMES
from openfold3.core.loss.confidence import all_atom_experimentally_resolved_loss, all_atom_plddt_loss, pae_loss, pde_loss
from openfold3.core.loss.diffusion import bond_loss, mse_loss, smooth_lddt_loss
from openfold3.core.loss.distogram import all_atom_distogram_loss

torch.set_default_dtype(torch.float64)
D = sys.argv[1]
rd = lambda name: np.fromfile(f"{D}/{name}.bin", dtype=np.float64)
vals = rd("values")
n, dense = int(vals[10]), int(vals[11])
t_noise = vals[9]
exists = rd("exists").astype(bool).reshape(n, dense)
names_raw = rd("names").reshape(n, dense, 4).astype(int)
name = [["".join(chr(c + 32) for c in names_raw[t, k] if c > 0) for k in range(dense)] for t in range(n)]
tok = rd("tok").reshape(n, 7)
is_dna, is_rna, is_lig, is_mod = (tok[:, k] for k in range(4))
asym = tok[:, 4]
aatype = tok[:, 5].astype(int)
# AF3's alphabet (the featuriser's NAMES) -> OpenFold3's
AF3_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
             "THR", "TRP", "TYR", "VAL", "UNK", "GAP", "A", "G", "C", "U", "DA", "DG", "DC", "DT", "N"]
nexist = exists.sum(1)
atomized = ((is_lig + is_mod) > 0) | (nexist == 1)
is_prot = 1 - is_dna - is_rna - is_lig

# ---- the flat layout: (dense slot or -1, OpenFold3's atom) per atom, token by token
slots, tok_of, counts = [], [], []
for t in range(n):
    here = {name[t][k]: t * dense + k for k in range(dense) if exists[t, k]}
    if atomized[t]:
        order = [k for k in here.values()]
    else:
        res = AF3_NAMES[aatype[t]]
        canon = TOKEN_NAME_TO_ATOM_NAMES[res]
        extra = [nm for nm in here if nm not in canon]
        if extra: sys.exit(f"token {t} ({res}): atoms {extra} not in OpenFold3's {res}")
        order = [here.get(nm, -1) for nm in canon]
    slots += order; tok_of += [t] * len(order); counts.append(len(order))
slots = np.array(slots); A = len(slots)
real = slots >= 0
start = np.concatenate([[0], np.cumsum(counts)[:-1]])


def flat(v, width):       # a dense [n*dense, width] array into the flat layout (padding atoms zero)
    v = v.reshape(n * dense, width)
    out = np.zeros((A, width)); out[real] = v[slots[real]]
    return out


x0 = torch.tensor(flat(rd("x"), 3)); xgt = torch.tensor(flat(rd("xt"), 3))
gm = torch.tensor(flat(rd("gm"), 1)[:, 0])
atom_mask = torch.tensor(real.astype(float))
restype = np.zeros((n, 32))
for t in range(n):
    restype[t, OF3_RES.index(AF3_NAMES[aatype[t]]) if AF3_NAMES[aatype[t]] in OF3_RES else OF3_RES.index("UNK")] = 1
B = lambda v: torch.tensor(np.asarray(v, dtype=np.float64))[None]      # a batch of one
batch = {
    "token_mask": B(np.ones(n)), "num_atoms_per_token": B(counts), "start_atom_index": B(start),
    "is_protein": B(is_prot), "is_dna": B(is_dna), "is_rna": B(is_rna), "is_ligand": B(is_lig),
    "is_atomized": B(atomized.astype(float)), "asym_id": B(asym), "restype": B(restype),
    "atom_mask": atom_mask[None], "token_bonds": B(rd("bonds").reshape(n, n)),
    "ground_truth": {"atom_positions": xgt[None], "atom_resolved_mask": gm[None]},
    "loss_weights": {"distogram": torch.ones(1)},
}
EPS = 1e-8

worst = 0.0
def check(what, mine, ref):
    global worst
    e = abs(mine - ref) / max(abs(ref), 1e-12); worst = max(worst, e)
    print(f"  {what:22s} C++ {mine:.10f}  OpenFold3 {ref:.10f}  rel {e:.2e}")
def grad_check(what, mine, ref):
    global worst
    e = float(np.sqrt(np.sum((mine - ref) ** 2) / max(np.sum(ref ** 2), 1e-300))); worst = max(worst, e)
    print(f"  {what:22s} gradient relRMS {e:.2e}")

# ---- the diffusion terms (mse with its weighted alignment, bond, smooth LDDT; diffusion_loss's weighting)
xv = x0.clone().requires_grad_(True)
ones = torch.ones(1, n)
lm = mse_loss(x=xv[None], batch=batch, loss_token_mask=ones, dna_weight=5.0, rna_weight=5.0, ligand_weight=10.0, eps=EPS)[0]
lb = bond_loss(x=xv[None], batch=batch, eps=EPS)[0]
ll = smooth_lddt_loss(x=xv[None], batch=batch, loss_token_mask=ones, eps=EPS)[0]
wt = (t_noise ** 2 + 16 ** 2) / (t_noise * 16) ** 2
total = wt * (4 * lm + 1 * lb) + 4 * ll
total.backward()
check("mse", vals[0], lm.item()); check("bond", vals[1], lb.item()); check("smooth_lddt", vals[2], ll.item())
check("diffusion total", vals[3], total.item())
mine_dx = flat(rd("dx"), 3)
grad_check("d total / d x", mine_dx[real], xv.grad.numpy()[real])

# ---- distogram, pLDDT, PDE, PAE, experimentally resolved
def logits_pair(nm, K): return torch.tensor(rd(nm).reshape(n, n, K), requires_grad=True)
def logits_atom(nm, K): return torch.tensor(flat(rd(nm), K), requires_grad=True)
lg = logits_pair("lDg", 64)
l, _ = all_atom_distogram_loss(batch=batch, logits=lg[None], no_bins=64, bin_min=2.0, bin_max=22.0, eps=EPS)
l.backward(); check("distogram", vals[4], l.item()); grad_check("d distogram / d logits", rd("gDg").reshape(n, n, 64), lg.grad.numpy())
lg = logits_atom("lPl", 50)
l = all_atom_plddt_loss(batch=batch, x=x0[None], logits=lg[None], no_bins=50, bin_min=0.0, bin_max=1.0, eps=EPS)[0]
l.backward(); check("plddt", vals[5], l.item()); grad_check("d plddt / d logits", flat(rd("gPl"), 50)[real], lg.grad.numpy()[real])
lg = logits_pair("lPde", 64)
l = pde_loss(batch=batch, x=x0[None], logits=lg[None], no_bins=64, bin_min=0.0, bin_max=32.0, eps=EPS)[0]
l.backward(); check("pde", vals[6], l.item()); grad_check("d pde / d logits", rd("gPde").reshape(n, n, 64), lg.grad.numpy())
lg = logits_pair("lPae", 64)
l = pae_loss(batch=batch, x=x0[None], logits=lg[None], angle_threshold=25.0, no_bins=64, bin_min=0.0, bin_max=32.0, eps=EPS, inf=1e9)[0]
l.backward(); check("pae", vals[7], l.item()); grad_check("d pae / d logits", rd("gPae").reshape(n, n, 64), lg.grad.numpy())
lg = logits_atom("lRs", 2)
l = all_atom_experimentally_resolved_loss(batch=batch, logits=lg[None], no_bins=2, eps=EPS)[0]
l.backward(); check("resolved", vals[8], l.item()); grad_check("d resolved / d logits", flat(rd("gRs"), 2)[real], lg.grad.numpy()[real])
ok = worst < 1e-12
print("worst", f"{worst:.2e}", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

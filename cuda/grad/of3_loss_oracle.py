"""The oracle for cuda/grad/of3loss.h: OpenFold3's training losses (aqlaboratory/openfold-3, openfold3/core/loss:
diffusion.py, distogram.py, confidence.py, and the atom selections in core/utils/atomize_utils.py) transcribed line by
line into torch, in float64, on the case test_of3loss.cu writes - its values, and its gradients by autograd, against
the C++ side's own.

    ~/venv_ef2/bin/python cuda/grad/of3_loss_oracle.py <case dir>

The one liberty: OpenFold3 finds an atom within its residue through restype tables; here it is found by NAME in the
batch's own per-slot names (the table's purpose), glycine by aatype (AF3's alphabet, G at 7), a purine by having N9.
"""
import math
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
D = sys.argv[1]
rd = lambda name: np.fromfile(f"{D}/{name}.bin", dtype=np.float64)
vals = rd("values")
n, dense = int(vals[10]), int(vals[11])
t_noise = vals[9]
exists = rd("exists").astype(bool)
live = np.nonzero(exists)[0]                       # the flat atoms, token-major as OpenFold3 lays them
A = len(live)
xs = lambda v: torch.tensor(v.reshape(-1, 3)[live])
x0, xgt = xs(rd("x")), xs(rd("xt"))
gm = torch.tensor(rd("gm")[live])
tok = rd("tok").reshape(n, 7)
is_dna, is_rna, is_ligand, is_mod = (torch.tensor(tok[:, k]) for k in range(4))
asym = torch.tensor(tok[:, 4]).long()
aatype = tok[:, 5].astype(int)
is_protein = 1 - is_dna - is_rna - is_ligand
names_raw = rd("names").reshape(-1, 4).astype(int)
token_of = live // dense
num_atoms = torch.tensor(np.bincount(token_of, minlength=n)).double()
start = np.zeros(n, dtype=int)
start[1:] = np.cumsum(num_atoms.numpy().astype(int))[:-1]
start = torch.tensor(start)
is_atomized = ((is_ligand + is_mod) > 0).double()
is_atomized = torch.where(num_atoms == 1, torch.ones_like(is_atomized), is_atomized)
token_bonds = torch.tensor(rd("bonds").reshape(n, n))
name = ["".join(chr(c + 32) for c in names_raw[i] if c > 0) for i in live]
EPS, INF = 1e-8, 1e9


def offset(atom):
    """get_token_atom_index_offset: the atom's index within its token, and whether the token has it"""
    off = torch.zeros(n, dtype=torch.long); has = torch.zeros(n)
    for a in range(A):
        if name[a] == atom:
            t = token_of[a]; off[t] = a - start[t]; has[t] = 1
    return off, has


def bcast(feat):   # broadcast_token_feat_to_atoms
    return feat[torch.tensor(token_of)]


# ---- diffusion.py
def weighted_rigid_align(x, x_gt, w, m):
    m = m.bool()
    ws = torch.sum(w * m)
    mu = torch.sum(x * w[:, None] * m[:, None], 0) / (ws + EPS)
    mu_gt = torch.sum(x_gt * w[:, None] * m[:, None], 0) / (ws + EPS)
    x = x - mu; x_gt = x_gt - mu_gt
    H = (x_gt[:, :, None] * x[:, None, :] * (w * m)[:, None, None]).sum(0)
    U, _, V = torch.linalg.svd(H)
    F = torch.eye(3); F[-1, -1] = torch.sign(torch.linalg.det(U @ V))
    R = U @ F @ V
    return (x @ R.T + mu_gt).detach()


def mse_loss(x):
    w = bcast(1 + is_dna * 5 + is_rna * 5 + is_ligand * 10)
    xa = weighted_rigid_align(xgt, x, w, gm)
    lm = gm
    return (1 / 3.0) * torch.sum(torch.sum((x - xa) ** 2, -1) * w * lm) / (torch.sum(lm) + EPS)


def bond_loss(x):
    dx = torch.sqrt(EPS + torch.sum((x[:, None] - x[None]) ** 2, -1))
    dg = torch.sqrt(EPS + torch.sum((xgt[:, None] - xgt[None]) ** 2, -1))
    is_polymer = is_protein + is_dna + is_rna
    bm = token_bonds * (is_polymer[None, :] * is_ligand[:, None])
    bm = bm[torch.tensor(token_of)][:, torch.tensor(token_of)]
    mask = (bm * (gm[:, None] * gm[None])).bool()
    return torch.sum((dx - dg) ** 2 * mask) / (torch.sum(mask, dtype=torch.float32) + EPS)


def smooth_lddt_loss(x):
    dx = torch.sqrt(EPS + torch.sum((x[:, None] - x[None]) ** 2, -1))
    dg = torch.sqrt(EPS + torch.sum((xgt[:, None] - xgt[None]) ** 2, -1))
    d = torch.abs(dg - dx)
    e = 0.25 * (torch.sigmoid(0.5 - d) + torch.sigmoid(1.0 - d) + torch.sigmoid(2.0 - d) + torch.sigmoid(4.0 - d))
    nuc = bcast(is_dna + is_rna)
    c = (dg < 30) * nuc[:, None] + (dg < 15) * (1 - nuc[:, None])
    mask = ((1 - torch.eye(A)) * (gm[:, None] * gm[None])).bool()
    return 1 - torch.sum(c * e * mask) / (torch.sum(c * mask) + EPS)


# ---- atomize_utils.py
off_cb, has_cb = offset("CB"); off_ca, has_ca = offset("CA"); off_c4, has_c4 = offset("C4"); off_c2, has_c2 = offset("C2")
off_n, has_n = offset("N"); off_c, has_c = offset("C"); off_c3p, has_c3p = offset("C3'"); off_c1p, has_c1p = offset("C1'")
off_c4p, has_c4p = offset("C4'"); _, has_n9 = offset("N9")
std_prot = is_protein * (1 - is_atomized)
gly = std_prot * torch.tensor((aatype == 7).astype(float))
std_nuc = (is_dna + is_rna) * (1 - is_atomized)
purine = std_nuc * has_n9
pyrimidine = std_nuc * (1 - has_n9)


def representative(x, am):
    """get_token_representative_atoms"""
    idx = torch.zeros(n, dtype=torch.long); ok = torch.zeros(n)
    for t in range(n):
        if is_atomized[t]: idx[t], k = start[t], 1
        elif gly[t]: idx[t], k = start[t] + off_ca[t], has_ca[t]
        elif std_prot[t]: idx[t], k = start[t] + off_cb[t], has_cb[t]
        elif purine[t]: idx[t], k = start[t] + off_c4[t], has_c4[t]
        else: idx[t], k = start[t] + off_c2[t], has_c2[t]
        ok[t] = k * am[idx[t]]
    return x[idx], ok


def frame_atoms(x, am, angle=25.0):
    """get_token_frame_atoms"""
    aa = bcast(asym)
    pm = am.bool()[start][:, None] & am.bool()[None] & (aa[start][:, None] == aa[None])
    d = torch.sum(EPS + (x[start][:, None] - x[None]) ** 2, -1) ** 0.5
    d = torch.where(pm, d, torch.full_like(d, INF))
    _, cl = torch.topk(d, k=3, dim=-1, largest=False)
    ai, ci = cl[:, 1], cl[:, 2]
    ia = is_atomized.long()
    sp, sn = std_prot.long(), std_nuc.long()
    index = {"a": ai * ia + (start + off_n) * sp + (start + off_c3p) * sn,
             "b": start * ia + (start + off_ca) * sp + (start + off_c1p) * sn,
             "c": ci * ia + (start + off_c) * sp + (start + off_c4p) * sn}
    tm = {"a": is_atomized + has_n * std_prot + has_c3p * std_nuc,
          "b": is_atomized + has_ca * std_prot + has_c1p * std_nuc,
          "c": is_atomized + has_c * std_prot + has_c4p * std_nuc}
    P = {k: x[index[k]] for k in index}
    AS = {k: aa[index[k]] for k in index}
    M = {k: am[index[k]] * tm[k] for k in index}
    u, v = P["a"] - P["b"], P["c"] - P["b"]
    cosang = (u * v).sum(-1) / ((EPS + (u ** 2).sum(-1)) ** 0.5 * (EPS + (v ** 2).sum(-1)) ** 0.5)
    lo, hi = math.cos((180 - angle) * math.pi / 180), math.cos(angle * math.pi / 180)
    va = ((cosang < hi) * (cosang > lo)).double()
    va = va * is_atomized + (1 - is_atomized)
    valid = va * M["a"] * M["b"] * M["c"] * ((AS["a"] == AS["b"]) * (AS["b"] == AS["c"])).double()
    return (P["a"], P["b"], P["c"]), valid


def express(x, phi):
    a, b, c = phi
    w1, w2 = a - b, c - b
    w1 = w1 / torch.sqrt(EPS + (w1 ** 2).sum(-1, keepdim=True)); w2 = w2 / torch.sqrt(EPS + (w2 ** 2).sum(-1, keepdim=True))
    e1, e2 = w1 + w2, w2 - w1
    e1 = e1 / torch.sqrt(EPS + (e1 ** 2).sum(-1, keepdim=True)); e2 = e2 / torch.sqrt(EPS + (e2 ** 2).sum(-1, keepdim=True))
    e3 = torch.stack([e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1], e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
                      e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]], -1)
    d = x[None] - b[:, None]
    return torch.stack([torch.einsum("ik,ijk->ij", e, d) for e in (e1, e2, e3)], -1)


def binned(v, lo, hi, K):
    w = (hi - lo) / K
    centres = lo + w / 2 + torch.arange(K) * w
    return torch.nn.functional.one_hot(torch.argmin(torch.abs(v[..., None] - centres), -1), K).double()


def ce(logits, labels):
    return -torch.sum(labels * torch.log_softmax(logits, -1), -1)


# ---- the losses
all_atoms = torch.ones(A)


def distogram(lg):
    rx, ok = representative(xgt, gm)
    d = torch.sqrt(torch.sum((rx[:, None] - rx[None]) ** 2, -1))
    pm = (ok[:, None] * ok[None]).bool()
    return torch.sum(ce(lg, binned(d, 2, 22, 64)) * pm) / (torch.sum(pm, dtype=torch.float32) + EPS)


def plddt(x, lg):
    dx = torch.sqrt(EPS + torch.sum((x[:, None] - x[None]) ** 2, -1))
    dg = torch.sqrt(EPS + torch.sum((xgt[:, None] - xgt[None]) ** 2, -1))
    d = torch.abs(dg - dx)
    pm = (dg < 15) * bcast(is_protein)[None, :] + (dg < 30) * bcast(is_dna + is_rna)[None, :]
    rep = torch.zeros(A)
    for t in range(n):
        if std_prot[t] and has_ca[t]: rep[start[t] + off_ca[t]] = 1
        if std_nuc[t] and has_c1p[t]: rep[start[t] + off_c1p[t]] = 1
    rep = rep * gm
    pm = (pm * (gm[:, None] * rep[None]) * (1 - torch.eye(A))).bool()
    lddt = torch.sum(0.25 * ((d < 0.5).int() + (d < 1).int() + (d < 2).int() + (d < 4).int()) * pm, -1) / (torch.sum(pm, -1, dtype=torch.float32) + EPS)
    return torch.sum(ce(lg, binned(lddt, 0, 1, 50)) * gm) / (torch.sum(gm, dtype=torch.float32) + EPS)


def pde(x, lg):
    rx, _ = representative(x, all_atoms)
    rg, okg = representative(xgt, gm)
    d = torch.sqrt(EPS + torch.sum((rx[:, None] - rx[None]) ** 2, -1))
    dg = torch.sqrt(EPS + torch.sum((rg[:, None] - rg[None]) ** 2, -1))
    pm = (okg[:, None] * okg[None]).bool()
    return torch.sum(ce(lg, binned(torch.abs(d - dg), 0, 32, 64)) * pm) / (torch.sum(pm, dtype=torch.float32) + EPS)


def pae(x, lg):
    phi, vf = frame_atoms(x, all_atoms)
    phig, vfg = frame_atoms(xgt, gm)
    rx, ok = representative(x, all_atoms)
    rg, okg = representative(xgt, gm)
    e = torch.sqrt(EPS + torch.sum((express(rx, phi) - express(rg, phig)) ** 2, -1))
    pm = ((vf[:, None] * ok[None]) * (vfg[:, None] * okg[None])).bool()
    return torch.sum(ce(lg, binned(e, 0, 32, 64)) * pm) / (torch.sum(pm) + EPS)


def resolved(lg):
    y = torch.nn.functional.one_hot(gm.long(), 2).double()
    return torch.sum(ce(lg, y)) / (torch.tensor(float(A), dtype=torch.float32) + EPS)


# ---- evaluate, with autograd, against the C++ side
def logits(nm, K, per_atom):
    v = rd(nm)
    if per_atom: v = v.reshape(n * dense, K)[live]
    else: v = v.reshape(n, n, K)
    return torch.tensor(v, requires_grad=True)


worst = 0.0
def check(what, mine, ref):
    global worst
    e = abs(mine - ref) / max(abs(ref), 1e-12)
    worst = max(worst, e)
    print(f"  {what:24s} C++ {mine:.10f}  oracle {ref:.10f}  rel {e:.2e}")


def grad_check(what, mine, ref):
    global worst
    e = float(np.sqrt(np.sum((mine - ref) ** 2) / max(np.sum(ref ** 2), 1e-300)))
    worst = max(worst, e)
    print(f"  {what:24s} gradient relRMS {e:.2e}")


xv = x0.clone().requires_grad_(True)
wt = (t_noise ** 2 + 256) / (t_noise * 16) ** 2
lm, lb, ll = mse_loss(xv), bond_loss(xv), smooth_lddt_loss(xv)
total = wt * (4 * lm + 1 * lb) + 4 * ll
total.backward()
check("mse", vals[0], lm.item()); check("bond", vals[1], lb.item()); check("smooth_lddt", vals[2], ll.item())
check("diffusion total", vals[3], total.item())
dx = np.zeros((n * dense, 3)); dx[live] = xv.grad.numpy()
grad_check("d total / d x", rd("dx").reshape(-1, 3)[live], dx[live])

for nm, fn, K, per_atom, gnm, vi in [("lDg", distogram, 64, False, "gDg", 4), ("lPl", None, 50, True, "gPl", 5),
                                     ("lPde", None, 64, False, "gPde", 6), ("lPae", None, 64, False, "gPae", 7),
                                     ("lRs", resolved, 2, True, "gRs", 8)]:
    lg = logits(nm, K, per_atom)
    f = {"lPl": lambda l: plddt(x0, l), "lPde": lambda l: pde(x0, l), "lPae": lambda l: pae(x0, l)}.get(nm, fn)
    v = f(lg); v.backward()
    check(nm[1:], vals[vi], v.item())
    g = rd(gnm)
    if per_atom: g = g.reshape(n * dense, K)[live]
    else: g = g.reshape(n, n, K)
    grad_check(f"d {nm[1:]} / d logits", g, lg.grad.numpy())
print("worst", f"{worst:.2e}", "PASS" if worst < 1e-9 else "FAIL")
sys.exit(0 if worst < 1e-9 else 1)

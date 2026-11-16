"""ProteinEBM's Langevin dynamics run by the vendor's own code, for the WebGPU port's trajectory statistics.

    python3 tools/oracle/proteinebm_dynamics.py --pdb tools/fixtures/6mrr-crystal.pdb --steps 300 --every 30 --seeds 0,1

The same update as tools/gpu/dynamics-proteinebm.js (the Langevin notebook's; self-conditioning "none" is what
run_dynamics.py runs), its noise from torch, so the two agree in DISTRIBUTION, not step for step - the dump in
dump_proteinebm.py is the step-for-step check. Printed per reported step: the energy, the noisy state's RMSD to the
start, the denoised prediction's RMSD and its median consecutive CA-CA, and its radius of gyration.
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default=str(Path.home() / "Documents/GitHub/ProteinEBM"))
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--pdb", default="tools/fixtures/6mrr-crystal.pdb")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--every", type=int, default=30)
parser.add_argument("--t", type=float, default=0.1)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--temp-scaling", type=float, default=0.85714)
parser.add_argument("--selfcond", default="none", choices=["none", "current"])
parser.add_argument("--seeds", default="0")
args = parser.parse_args()

sys.modules.setdefault("py3Dmol", types.ModuleType("py3Dmol"))
sys.path.insert(0, args.repo)
import yaml  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.Polypeptide import is_aa  # noqa: E402
from protein_ebm.model.r3_diffuser import R3Diffuser  # noqa: E402
from protein_ebm.model.ebm import ProteinEBM  # noqa: E402
from protein_ebm.data.protein_utils import residues_to_features  # noqa: E402

config = ConfigDict(yaml.safe_load(open(Path(args.repo) / "protein_ebm/config/base_pretrain.yaml")))
diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser)
checkpoint = args.checkpoint or str(Path(args.repo) / "weights/model_1_frozen_1m_md.pt")
state = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
model.eval()

chain = next(PDBParser(QUIET=True).get_structure("x", args.pdb)[0].get_chains())
positions, _, aatype, residue_idx = residues_to_features([r for r in chain.get_residues() if is_aa(r)])
aatype = torch.as_tensor(aatype, dtype=torch.long)
residue_idx = torch.as_tensor(residue_idx, dtype=torch.long)
start = positions[:, 1, :].float()
start = start - start.mean(0, keepdim=True)
n = start.shape[0]


def kabsch(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((x @ r - y) ** 2).sum(1).mean()))


def ca_median(x):
    return float(np.median(np.linalg.norm(x[1:] - x[:-1], axis=1)))


g = float(diffuser.diffusion_coef(args.t))
for seed in [int(s) for s in args.seeds.split(",")]:
    torch.manual_seed(seed)
    pos = start.clone()
    print(f"seed {seed}")
    for step in range(args.steps):
        feats = {"r_noisy": pos[None], "aatype": aatype[None], "mask": torch.ones(1, n), "residue_idx": residue_idx[None],
                 "t": torch.tensor([args.t]), "chain_encoding": torch.zeros(1, n, dtype=torch.long),
                 "external_contacts": torch.zeros(1, n, dtype=torch.long)}
        if args.selfcond == "current":
            feats["selfcond_coords"] = pos[None].clone()
        out = model.compute_score(feats)
        score = out["trans_score"][0].detach()
        x0 = out["pred_coords"][0].reshape(n, 3).detach().numpy()
        pos = pos - (diffuser.drift_coef(pos, args.t) - g ** 2 * score * args.temp_scaling / 0.1) * args.dt \
            + g * np.sqrt(args.dt) * torch.randn_like(pos) / 0.1
        pos = pos - pos.mean(0, keepdim=True) + torch.randn(1, 3)
        if (step + 1) % args.every == 0 or step == 0:
            p = pos.numpy()
            rg = float(np.sqrt(((x0 - x0.mean(0)) ** 2).sum(1).mean()))
            print(f"  step {step + 1:4d}  E {float(out['energy'].sum()):8.2f}  noisy rmsd {kabsch(p, start.numpy()):6.3f}"
                  f"  ca {ca_median(p):5.2f} | denoised rmsd {kabsch(x0, start.numpy()):6.3f}  ca {ca_median(x0):5.3f}"
                  f"  rg {rg:5.2f}", flush=True)

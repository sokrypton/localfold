"""ProteinEBM's own energy and score, dumped for the WebGPU port to be held to.

    python3 tools/oracle/dump_proteinebm.py --repo ~/Documents/GitHub/ProteinEBM \
        --checkpoint ~/Documents/GitHub/ProteinEBM/weights/model_1_frozen_1m_md.pt \
        --pdb tools/fixtures/6mrr-crystal.pdb --out oracle-dumps/proteinebm-6mrr.json

It runs jproney/ProteinEBM's `ProteinEBM.compute_score` (torch, float32, CPU) on one chain's CA trace and writes:

  * the inputs: aatype (AlphaFold's ARNDC... order, 20 = unknown), residue_idx (the PDB's own numbering, which is what
    `residues_to_features` hands the model), the centred CA coordinates, t;
  * the outputs: energy, r_update, the score (-dE/dr, in the units of r: Angstroms in, per Angstrom out);
  * the seams the backward pass is bisected on: s (the single conditioner's output), z (the pair conditioner's),
    a (the token transformer's output after a_norm), and dE/ds, dE/dz - the gradient as it reaches each;
  * TWO self-conditioning arms: `zeros` (what run_dynamics.py and proteinebm_reverse_diffusion.py actually run - they
    pass the coordinates as `sc_ca_t`, a key compute_energy never reads, so the channel is zero) and `current` (the
    Langevin notebook's `selfcond_coords = pos_t`);
  * three Langevin steps of the notebook's update with the noise and the recentring translation written down, so a
    port can replay them exactly.

py3Dmol is imported by protein_utils and is not needed: a stub stands in for it.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default=str(Path.home() / "Documents/GitHub/ProteinEBM"))
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--pdb", default="tools/fixtures/6mrr-crystal.pdb")
parser.add_argument("--chain", default=None)
parser.add_argument("--t", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--temp-scaling", type=float, default=0.85714)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="oracle-dumps/proteinebm-6mrr.json")
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

torch.manual_seed(args.seed)
config = ConfigDict(yaml.safe_load(open(Path(args.repo) / "protein_ebm/config/base_pretrain.yaml")))
diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser)
checkpoint = args.checkpoint or str(Path(args.repo) / "weights/model_1_frozen_1m_md.pt")
state = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
model.eval()

structure = PDBParser(QUIET=True).get_structure("x", args.pdb)
chains = list(structure[0].get_chains())
chain = next(c for c in chains if args.chain is None or c.id == args.chain)
positions, atom_mask, aatype, residue_idx = residues_to_features([r for r in chain.get_residues() if is_aa(r)])
aatype = torch.as_tensor(aatype, dtype=torch.long)
residue_idx = torch.as_tensor(residue_idx, dtype=torch.long)
ca = positions[:, 1, :].float()
ca = ca - ca.mean(dim=0, keepdim=True)
n = ca.shape[0]

# the seams: the conditioners' outputs and the transformer's normed output, captured so the gradient can be asked of them
seams = {}
model.single_conditioner.register_forward_hook(lambda m, i, o: seams.__setitem__("s", o[0]))
model.pairwise_conditioner.register_forward_hook(lambda m, i, o: seams.__setitem__("z", o))
model.a_norm.register_forward_hook(lambda m, i, o: seams.__setitem__("a", o))


def feats(r, sc):
    f = {"r_noisy": r[None], "aatype": aatype[None], "mask": torch.ones(1, n), "residue_idx": residue_idx[None],
         "t": torch.tensor([args.t]), "chain_encoding": torch.zeros(1, n, dtype=torch.long),
         "external_contacts": torch.zeros(1, n, dtype=torch.long)}
    if sc is not None:
        f["selfcond_coords"] = sc[None]
    return f


def flat(x):
    return x.detach().double().reshape(-1).tolist()


def arm(sc):
    r = ca.clone().requires_grad_(True)
    out = model.compute_energy(feats(r, sc))
    energy = out["energy"].sum()
    dr, ds, dz = torch.autograd.grad(energy, [r, seams["s"], seams["z"]])
    # compute_score's own path, as the control: the conservative score it returns must be -dr
    score = model.compute_score(feats(ca.clone(), sc))["trans_score"][0]
    assert torch.allclose(score, -dr, atol=1e-5), float((score + dr).abs().max())
    return {"energy": float(energy), "rUpdate": flat(out["r_update"][0]), "score": flat(-dr),
            "s": flat(seams["s"][0]), "z": flat(seams["z"][0]), "a": flat(seams["a"][0]),
            "dEds": flat(ds[0]), "dEdz": flat(dz[0])}


def langevin(sc_mode):
    """The notebook's update, its noise and recentring drawn here and written down."""
    pos = ca.clone()
    g = float(diffuser.diffusion_coef(args.t))
    steps = []
    for _ in range(args.steps):
        sc = None if sc_mode == "zeros" else pos.clone()
        out = model.compute_score(feats(pos.clone(), sc))
        score = out["trans_score"][0].detach()
        noise = torch.randn_like(pos)
        drift = diffuser.drift_coef(pos, args.t)
        pos = pos - (drift - g ** 2 * score * args.temp_scaling / diffuser.config.coordinate_scaling) * args.dt \
            + g * np.sqrt(args.dt) * noise / diffuser.config.coordinate_scaling
        pos = pos - pos.mean(dim=0, keepdim=True)
        shift = torch.randn(1, 3)          # center_random_augmentation(rotate=False)'s translation, std 1
        pos = pos + shift
        steps.append({"energy": float(out["energy"].sum()), "noise": flat(noise), "shift": flat(shift[0]),
                      "position": flat(pos)})
    return steps


dump = {
    "source": "jproney/ProteinEBM compute_score, torch float32 CPU",
    "checkpoint": Path(checkpoint).name, "pdb": Path(args.pdb).name, "length": n,
    "aatype": aatype.tolist(), "residueIdx": residue_idx.tolist(), "ca": flat(ca), "t": args.t,
    "diffuser": {"minB": config.diffuser.min_b, "maxB": config.diffuser.max_b,
                 "coordinateScaling": config.diffuser.coordinate_scaling},
    "arms": {"zeros": arm(None), "current": arm(ca.clone())},
    "langevin": {"dt": args.dt, "tempScaling": args.temp_scaling,
                 "zeros": langevin("zeros"), "current": langevin("current")},
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(dump))
print(f"{args.out}: {n} residues, energy zeros {dump['arms']['zeros']['energy']:.4f} "
      f"current {dump['arms']['current']['energy']:.4f}")

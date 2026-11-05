"""OpenFold3's own training step in PyTorch, timed - the reference cuda/grad/train-of3 is compared against: the full
OpenFold3AllAtom model (48 pairformer blocks) from an openfold-3 checkout in train mode, OpenFold3Loss, the backward,
on random_of3_features (OpenFold3's test batch) - as openfold3/tests/test_of3_model.py runs a step, at a training shape.

    PYTHONPATH=<openfold-3 checkout> ~/venv_protenix/bin/python cuda/grad/bench_of3_torch.py [--tokens=384] [--msa=1024]
        [--templates=4] [--precision=fp32|tf32|bf16] [--steps=5]

Recycles are fixed at none (num_recycles 0, one cycle) so a step is comparable with train-of3 --max-recycles=0;
48 diffusion samples, the 20-step mini rollout and every loss as OpenFold3 configures them. Activation checkpointing
a block at a time (blocks_per_ckpt 1), as the test and OpenFold3's training run it. bf16: torch.autocast bfloat16 over
the step, the parameters float32 (bf16-mixed).
"""
import argparse
import copy
import time

import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.model import OpenFold3 as OpenFold3AllAtom
from openfold3.projects.of3_all_atom.config.model_config import model_config as of3_model_config
from openfold3.tests.utils.data_utils import random_of3_features

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=384)
ap.add_argument("--msa", type=int, default=1024)
ap.add_argument("--templates", type=int, default=4)
ap.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "bf16"])
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--cueq", action="store_true", help="cuEquivariance's triangle kernels (settings.memory.train.use_cueq_triangle_kernels)")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = a.precision == "tf32"
torch.backends.cudnn.allow_tf32 = a.precision == "tf32"
config = copy.deepcopy(of3_model_config)   # (OF3ProjectEntry's default, no presets - without its Lightning runner)
config.settings.blocks_per_ckpt = 1
config.settings.ckpt_intermediate_steps = True
config.architecture.shared.num_recycles = 0
config.settings.memory.train.use_cueq_triangle_kernels = a.cueq
config.architecture.loss_module.diffusion.chunk_size = 16   # (as test_of3_model.py: the diffusion losses a chunk of samples at a time)
import openfold3.projects.of3_all_atom.model as of3_model_module


def identity_alignment(batch, atom_positions_predicted):
    # random_of3_features carries no ground-truth token ids, so OpenFold3's permutation alignment fails on it and
    # zeroes every loss (the step then runs almost no backward); its truth is already in order and already shaped
    # a sample at a time, so it is left as it is
    pass


of3_model_module.safe_multi_chain_permutation_alignment = identity_alignment
model = OpenFold3AllAtom(config).cuda().train()
loss_fn = OpenFold3Loss(config=config.architecture.loss_module)
n_params = sum(p.numel() for p in model.parameters())
batch = random_of3_features(batch_size=1, n_token=a.tokens, n_msa=a.msa, n_templ=a.templates)
batch = tensor_tree_map(lambda t: t.cuda(), batch)
n_atom = int(batch["num_atoms_per_token"].sum().item())
print(f"OpenFold3 (PyTorch {torch.__version__}), {n_params / 1e6:.1f} M parameters; {a.tokens} tokens, {n_atom} atoms, "
      f"{a.msa} MSA rows, {a.templates} templates; {a.precision}{' cueq' if a.cueq else ''}")
times = []
for step in range(a.steps):
    b = tensor_tree_map(lambda t: t.clone(), batch)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
        b, out = model(batch=b)
        loss = loss_fn(batch=b, output=out)
    loss.backward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    times.append(ms)
    print(f"step {step}: {ms:.0f} ms, loss {loss.item():.4f}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
warm = sorted(times[1:]) if len(times) > 1 else times
print(f"median of steps 1..{a.steps - 1}: {warm[len(warm) // 2]:.0f} ms")

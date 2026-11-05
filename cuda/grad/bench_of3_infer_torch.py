"""OpenFold3's own inference in PyTorch, timed - what cuda/af3's localfold-af3 --model=openbind0 is compared against:
the full OpenFold3AllAtom model (48 pairformer blocks) from an openfold-3 checkout in eval mode, no gradients, on
random_of3_features (OpenFold3's test batch, its eval form), with OpenFold3's inference settings: 1 + 3 recycles, a full
rollout of 200 diffusion steps for 5 samples, the confidence heads.

    PYTHONPATH=<openfold-3 checkout> python cuda/grad/bench_of3_infer_torch.py [--tokens=384] [--msa=1024]
        [--templates=0] [--precision=bf16|fp32] [--runs=3] [--recycles=3] [--samples=5] [--steps=200]

bf16: torch.autocast bfloat16 over the forward, the parameters float32 (OpenFold3's bf16-mixed). The eval memory
settings are OpenFold3's own (its Triton triangle kernels where installed). Peak is the allocator's.
"""
import argparse
import copy
import time

import torch

from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.config.model_config import model_config as of3_model_config
from openfold3.projects.of3_all_atom.model import OpenFold3 as OpenFold3AllAtom
from openfold3.tests.utils.data_utils import random_of3_features

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=384)
ap.add_argument("--msa", type=int, default=1024)
ap.add_argument("--templates", type=int, default=0)
ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--recycles", type=int, default=3)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--no-triton", action="store_true", help="the eval settings' Triton triangle kernels off")
a = ap.parse_args()

config = copy.deepcopy(of3_model_config)   # (OF3ProjectEntry's default, no presets - without its Lightning runner)
config.architecture.shared.num_recycles = a.recycles
config.architecture.shared.diffusion.no_full_rollout_samples = a.samples
config.architecture.shared.diffusion.no_full_rollout_steps = a.steps
if a.no_triton: config.settings.memory.eval.use_triton_triangle_kernels = False
import openfold3.projects.of3_all_atom.model as of3_model_module


def identity_alignment(batch, atom_positions_predicted):
    # OpenFold3 aligns the batch's ground truth to the prediction inside forward (for its metrics); random_of3_features
    # has no ground-truth token ids, and inference needs none of it - left as it is
    pass


of3_model_module.safe_multi_chain_permutation_alignment = identity_alignment
model = OpenFold3AllAtom(config).cuda().eval()
batch = random_of3_features(batch_size=1, n_token=a.tokens, n_msa=a.msa, n_templ=a.templates, is_eval=True)
batch = tensor_tree_map(lambda t: t.cuda(), batch)
n_atom = int(batch["num_atoms_per_token"].sum().item())
print(f"OpenFold3 inference (PyTorch {torch.__version__}); {a.tokens} tokens, {n_atom} atoms, {a.msa} MSA rows, "
      f"{a.templates} templates; {a.precision}; 1 + {a.recycles} cycles, {a.samples} samples x {a.steps} steps; "
      f"triton triangle kernels {config.settings.memory.eval.use_triton_triangle_kernels}")
times = []
for run in range(a.runs):
    b = tensor_tree_map(lambda t: t.clone(), batch)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
        _, out = model(batch=b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    times.append(ms)
    print(f"run {run}: {ms:.0f} ms, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
warm = sorted(times[1:]) if len(times) > 1 else times
print(f"median of runs 1..{a.runs - 1}: {warm[len(warm) // 2]:.0f} ms")

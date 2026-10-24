"""Dump deterministic CPU ground-truth oracle for the RFD4-Proteina Autoencoder Encoder
(`EncoderTransformer`, `AutoEncoder.encode`) and Partial Diffusion / Inverse Folding helpers
(`resolve_partial_t`, `build_partial_schedule`, `corrupt_batch`, `sample_noised_target_backbone`).

Runs on CPU only:
    CUDA_VISIBLE_DEVICES="" /home/ubuntu/venv_rfd4p/bin/python /home/ubuntu/localdesign/tools/oracle/dump_encoder_cpu.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import lightning as L  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from rfproteina.datapipes.collate import collate_batch  # noqa: E402
from rfproteina.datapipes.pipeline import build_design_validation_pipeline  # noqa: E402
from rfproteina.flow_matching.product_space_flow_matcher import (  # noqa: E402
    sample_noised_target_backbone,
)
from rfproteina.generate import get_contig_or_design_problem_dataset  # noqa: E402
from rfproteina.proteina import Proteina  # noqa: E402
from rfproteina.search.vanilla import (  # noqa: E402
    build_partial_schedule,
    resolve_inverse_folding_method,
    resolve_partial_t,
)

BATCH_KEYS = (
    "padding_mask",
    "is_in_context_conditioning",
    "supports_local_latent",
    "token_type",
    "atom37_mask",
    "coord_mask",
    "coords_nm",
    "chain_iid",
    "incident_bond_type_counts",
    "ground_truth_element",
    "charge",
    "coord_cond_mask",
    "molecule_iid",
)

DTYPE_RESTORE = {
    "padding_mask": torch.bool,
    "is_in_context_conditioning": torch.bool,
    "supports_local_latent": torch.bool,
    "atom37_mask": torch.bool,
    "coord_mask": torch.bool,
    "coord_cond_mask": torch.bool,
    "token_type": torch.long,
    "chain_iid": torch.long,
    "molecule_iid": torch.long,
    "ground_truth_element": torch.long,
    "charge": torch.long,
    "incident_bond_type_counts": torch.long,
    "coords_nm": torch.float32,
}


def load_batch_from_dump(path: Path) -> dict[str, torch.Tensor]:
    raw = load_file(str(path))
    batch: dict[str, torch.Tensor] = {}
    for k in BATCH_KEYS:
        t = raw[f"batch.{k}"].clone()
        if k in DTYPE_RESTORE:
            t = t.to(DTYPE_RESTORE[k])
        batch[k] = t
    # Ensure unresolved atoms with NaN coords are cleared in atom37_mask and coords_nm
    finite = torch.isfinite(batch["coords_nm"]).all(dim=-1)
    batch["atom37_mask"] = batch["atom37_mask"] & finite
    batch["coord_mask"] = batch["atom37_mask"]
    batch["coords_nm"] = torch.nan_to_num(batch["coords_nm"], nan=0.0) * batch["atom37_mask"][..., None]
    batch["coords"] = batch["coords_nm"] * 10.0
    return batch


def load_full_batch_from_cif(cif_path: Path, seed: int = 0) -> dict[str, torch.Tensor]:
    L.seed_everything(seed, verbose=False)
    pipeline = build_design_validation_pipeline(metrics_tags=[], path_length_search_cutoff=64)
    dataset = get_contig_or_design_problem_dataset(str(cif_path), num_replicates=1, transform=pipeline)
    sample = dataset[0]
    batch = collate_batch([sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]
    out = {k: batch[k].to(DTYPE_RESTORE.get(k, batch[k].dtype)).contiguous().clone() for k in BATCH_KEYS}
    out["coords"] = out["coords_nm"] * 10.0
    return out


def main() -> None:
    torch.set_float32_matmul_precision("highest")
    dump_dir = Path("/home/ubuntu/localdesign/oracle-dumps")
    weights_dir = Path("/home/ubuntu/localdesign/weights")

    model = Proteina.from_pretrained(weights_dir, device="cpu")
    ae = model.autoencoder
    fm = model.fm

    # Mirror released inference sampling config (`DEFAULT_SAMPLING` in shared/rfd4p/flow.js).
    sampling_model_args = {
        "bb_ca": {
            "schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
            "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "vf_tsr",
                "sc_scale_noise": 1.0,
                "t_lim_ode": 0.95,
                "t_lim_ode_below": 0.02,
                "center_every_step": False,
                "tsr_k": 16.0,
                "tsr_sigma": 3.0,
            },
        },
        "local_latents": {
            "schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
            "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "vf_tsr",
                "sc_scale_noise": 1.0,
                "t_lim_ode": 0.95,
                "t_lim_ode_below": 0.02,
                "center_every_step": False,
                "tsr_k": 4.0,
                "tsr_sigma": 12.0,
            },
        },
    }

    tensors: dict[str, torch.Tensor] = {}
    meta: dict = {"cases": [], "schedules": {}}

    def put(name: str, t: torch.Tensor) -> None:
        t = t.detach().cpu()
        if t.dtype == torch.bool:
            t = t.to(torch.uint8)
        elif t.dtype in (torch.float16, torch.bfloat16, torch.float64):
            t = t.float()
        tensors[name] = t.contiguous().clone()

    cases = {
        "motif_idx_1prw": load_batch_from_dump(dump_dir / "motif_idx_1prw.safetensors"),
        "ppi_pdl1": load_batch_from_dump(dump_dir / "ppi_pdl1.safetensors"),
        "ppi_pdl1_full": load_full_batch_from_cif(dump_dir / "ppi_pdl1.cif", seed=0),
    }

    captured_feats: list[torch.Tensor] = []

    def feat_pre_hook(_mod, args):
        captured_feats.append(args[0].detach().cpu().clone())

    hook_handle = ae.encoder.init_repr_factory.linear_out.register_forward_pre_hook(feat_pre_hook)

    orig_randn = torch.randn
    orig_randn_like = torch.randn_like
    recorded_draws: list[torch.Tensor] = []

    def rec_randn(*args, **kwargs):
        out = orig_randn(*args, **kwargs)
        recorded_draws.append(out.detach().cpu().clone())
        return out

    def rec_randn_like(inp, *args, **kwargs):
        out = orig_randn_like(inp, *args, **kwargs)
        recorded_draws.append(out.detach().cpu().clone())
        return out

    torch.randn = rec_randn
    torch.randn_like = rec_randn_like

    try:
        with torch.no_grad():
            for idx, (case_name, batch) in enumerate(cases.items()):
                meta["cases"].append(case_name)
                for k, v in batch.items():
                    put(f"{case_name}.batch.{k}", v)

                torch.manual_seed(100 + idx)
                captured_feats.clear()
                recorded_draws.clear()
                enc_out = ae.encode(batch)
                assert len(captured_feats) == 1 and len(recorded_draws) == 1
                put(f"{case_name}.enc.features", captured_feats[0])
                put(f"{case_name}.enc.eps", recorded_draws[0])
                put(f"{case_name}.enc.mean", enc_out["mean"])
                put(f"{case_name}.enc.log_scale", enc_out["log_scale"])
                put(f"{case_name}.enc.z_latent", enc_out["z_latent"])
                put(f"{case_name}.enc.emit_mask", enc_out["emit_mask"])

                # Also test corrupt_batch and sample_noised_target_backbone on this batch
                batch_c = {**batch}
                batch_c["x_1"] = {
                    "bb_ca": batch["coords_nm"][:, :, 1, :].clone(),
                    "local_latents": enc_out["z_latent"].clone(),
                }
                pt_bb = 0.35
                pt_ll = 0.55
                t_dict = {
                    "bb_ca": torch.full((1,), pt_bb, dtype=torch.float32),
                    "local_latents": torch.full((1,), pt_ll, dtype=torch.float32),
                }
                recorded_draws.clear()
                batch_c = fm.corrupt_batch(batch_c, t=t_dict)
                for di, draw in enumerate(recorded_draws):
                    put(f"{case_name}.corrupt.rng.{di}", draw)
                meta[f"{case_name}_corrupt_n_draws"] = len(recorded_draws)
                meta[f"{case_name}_corrupt_t"] = {"bb_ca": pt_bb, "local_latents": pt_ll}
                for dm in ("bb_ca", "local_latents"):
                    put(f"{case_name}.corrupt.x_0.{dm}", batch_c["x_0"][dm])
                    put(f"{case_name}.corrupt.x_1.{dm}", batch_c["x_1"][dm])
                    put(f"{case_name}.corrupt.x_t.{dm}", batch_c["x_t"][dm])

                recorded_draws.clear()
                t_bb_val = 0.42
                noised_bb = sample_noised_target_backbone(
                    fm,
                    batch_c,
                    batch_c["x_1"]["bb_ca"],
                    torch.full((1,), t_bb_val, dtype=torch.float32),
                )
                for di, draw in enumerate(recorded_draws):
                    put(f"{case_name}.noised_bb.rng.{di}", draw)
                meta[f"{case_name}_noised_bb_n_draws"] = len(recorded_draws)
                meta[f"{case_name}_noised_bb_t"] = t_bb_val
                put(f"{case_name}.noised_bb.out", noised_bb)
    finally:
        torch.randn = orig_randn
        torch.randn_like = orig_randn_like
        hook_handle.remove()

    # Schedule & resolve_partial_t test cases
    sched_configs = {
        "frac_0p5": {"partial_trajectory_fraction_steps": 0.5, "nsteps": 60},
        "explicit_both": {"partial_t_backbone": 0.4, "partial_t_latent": 0.6, "nsteps": 20},
        "coupled_bb": {"partial_t_backbone": 0.35, "couple_partial_t": True, "nsteps": 25},
        "coupled_ll": {"partial_t_latent": 0.5, "couple_partial_t": True, "nsteps": 25},
        "inverse_folding": {"inverse_folding": True, "nsteps": 60},
    }
    for sname, cfg_args in sched_configs.items():
        pt_dict = resolve_partial_t(cfg_args, fm, sampling_model_args)
        if_method = resolve_inverse_folding_method(cfg_args)
        frac = cfg_args.get("partial_trajectory_fraction_steps", None)
        nsteps = cfg_args["nsteps"]
        actual_nsteps = round(frac * nsteps) if pt_dict and frac is not None else nsteps
        ts, gt = build_partial_schedule(
            partial_t_dict=pt_dict,
            nsteps=actual_nsteps,
            fm=fm,
            sampling_model_args=sampling_model_args,
            full_nsteps=nsteps if frac is not None else None,
        )
        for dm in ("bb_ca", "local_latents"):
            put(f"sched.{sname}.ts.{dm}", ts[dm])
            put(f"sched.{sname}.gt.{dm}", gt[dm])
        meta["schedules"][sname] = {
            "cfg_args": cfg_args,
            "partial_t": {k: float(v) for k, v in pt_dict.items()},
            "inverse_folding_method": if_method,
            "actual_nsteps": actual_nsteps,
            "full_nsteps": nsteps if frac is not None else None,
        }

    out_st = dump_dir / "encoder_oracle.safetensors"
    out_json = dump_dir / "encoder_oracle.json"
    save_file(tensors, str(out_st))
    out_json.write_text(json.dumps(meta, indent=2))
    total_mb = sum(v.numel() * v.element_size() for v in tensors.values()) / 1e6
    print(f"wrote {len(tensors)} tensors ({total_mb:.2f} MB) -> {out_st} and {out_json}")


if __name__ == "__main__":
    main()

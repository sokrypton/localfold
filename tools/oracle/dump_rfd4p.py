"""Dump a deterministic RFD4-Proteina oracle from the upstream PyTorch code.

Runs upstream ``rfd4-generate`` unchanged (fp32, TF32 off) with recording hooks:

- every random draw (``torch.randn``/``randn_like``/``randperm``/``rand``) in call order, so the
  JS sampler can replay exactly the same noise;
- on the first trunk call: every tensor in the batch, each feature factory's pre-``linear_out``
  feature matrix, ``c``, ``s_init``, ``z_init`` and the single/pair state after each layer of the
  first recycle cycle;
- per sampling step: ``x_t``, ``t``, recycle depth and the network outputs;
- the decoder's input batch and outputs, and the written CIF.

Writes ``<out>/<case>.safetensors`` (tensors) and ``<out>/<case>.json`` (metadata + RNG log).

Usage (from the RFD4-Proteina checkout, with its venv):
    python /home/ubuntu/localdesign/tools/oracle/dump_rfd4p.py --case uncond_short \
        --inputs data/benchmarks/monomer/monomer-short.cif.zst [--nsteps 60] [hydra overrides...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

STATE: dict = {
    "tensors": {},
    "rng": [],
    "steps": [],
    "recording": False,
    "call": -1,
    "layer_capture": False,
    "meta": {},
}


def put(name: str, t: torch.Tensor) -> None:
    if name in STATE["tensors"]:
        raise KeyError(f"duplicate oracle tensor {name}")
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16 or t.dtype == torch.float64:
        t = t.float()
    STATE["tensors"][name] = t.contiguous().clone()


def install_rng_recorder() -> None:
    """Wrap the torch RNG entry points; record draws while generation runs."""

    def wrap(fn_name: str):
        orig = getattr(torch, fn_name)

        def recorded(*args, **kwargs):
            out = orig(*args, **kwargs)
            if STATE["recording"]:
                import traceback

                frame = traceback.extract_stack(limit=3)[-2]
                idx = len(STATE["rng"])
                STATE["rng"].append(
                    {
                        "i": idx,
                        "fn": fn_name,
                        "shape": list(out.shape),
                        "site": f"{Path(frame.filename).name}:{frame.lineno}",
                        "call": STATE["call"],
                    }
                )
                put(f"rng.{idx}", out)
            return out

        setattr(torch, fn_name, recorded)

    for name in ("randn", "randn_like", "randperm", "rand"):
        wrap(name)


def install_model_hooks() -> None:
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    from rfproteina.proteina import Proteina

    orig_schedule = ProductSpaceFlowMatcher.sample_schedule

    def sample_schedule(self, nsteps, sampling_model_args):
        ts, gt = orig_schedule(self, nsteps=nsteps, sampling_model_args=sampling_model_args)
        for dm in ts:
            put(f"schedule.ts.{dm}", ts[dm])
            put(f"schedule.gt.{dm}", gt[dm])
        STATE["meta"]["sampling_model_args"] = json.loads(json.dumps(sampling_model_args, default=str))
        STATE["recording"] = True
        return ts, gt

    ProductSpaceFlowMatcher.sample_schedule = sample_schedule

    orig_depths = Proteina.recycle_depths

    def recycle_depths(self, nsteps, full_nsteps=None):
        d = orig_depths(self, nsteps, full_nsteps)
        STATE["meta"]["recycle_depths"] = list(d)
        return d

    Proteina.recycle_depths = recycle_depths

    orig_call_nn = Proteina.call_nn
    hooked: set[int] = set()

    def call_nn(self, batch, n_recycle):
        STATE["call"] += 1
        k = STATE["call"]
        if id(self) not in hooked:
            attach_layer_hooks(self.nn)
            hooked.add(id(self))
        prefix = f"step.{k}"
        for dm, v in batch["x_t"].items():
            put(f"{prefix}.x_t.{dm}", v)
        for dm, v in batch["t"].items():
            put(f"{prefix}.t.{dm}", v)
        if k == 0:
            for key, v in batch.items():
                if isinstance(v, torch.Tensor):
                    put(f"batch.{key}", v)
            STATE["meta"]["batch_nontensor_keys"] = sorted(
                key for key, v in batch.items() if not isinstance(v, torch.Tensor)
            )
            STATE["layer_capture"] = True
            STATE["cycle_layer0_calls"] = 0 if STATE["meta"].get("layers", True) else 99
        out = orig_call_nn(self, batch, n_recycle)
        STATE["layer_capture"] = False
        STATE["steps"].append({"call": k, "n_recycle": int(n_recycle)})
        for dm, d in out.items():
            for kind, v in d.items():
                if isinstance(v, torch.Tensor) and not (dm == "insertion" and k > 0):
                    put(f"{prefix}.out.{dm}.{kind}", v)
        return out

    Proteina.call_nn = call_nn

    orig_format = Proteina._format_sample_local_latents

    def format_sample(self, x, ret_mode, batch):
        for dm, v in x.items():
            put(f"final.x.{dm}", v)
        ae = self.autoencoder
        orig_decode = ae.decode

        def decode(decode_batch, required_atom_mask=None):
            for key in ("z_latent", "ca_coors_nm", "token_type", "padding_mask"):
                if isinstance(decode_batch.get(key), torch.Tensor):
                    put(f"decode.in.{key}", decode_batch[key])
            from atomworks.ml.conditions import C_SEQ

            put("decode.in.C_SEQ", decode_batch[C_SEQ.full_name])
            if required_atom_mask is not None:
                put("decode.in.required_atom_mask", required_atom_mask)
            out = orig_decode(decode_batch, required_atom_mask=required_atom_mask)
            for key, v in out.items():
                if isinstance(v, torch.Tensor):
                    put(f"decode.out.{key}", v)
            return out

        ae.decode = decode
        try:
            res = orig_format(self, x, ret_mode, batch)
        finally:
            ae.decode = orig_decode
        for key in ("coors", "atom_mask", "residue_type"):
            put(f"final.{key}", res[key])
        STATE["recording"] = False
        return res

    Proteina._format_sample_local_latents = format_sample


def attach_layer_hooks(nn: torch.nn.Module) -> None:
    """Record factory features and per-layer states for the first cycle of the first call."""

    def factory_hook(name):
        def pre(mod, args):
            if STATE["layer_capture"]:
                key = f"call0.{name}.features"
                if key not in STATE["tensors"]:
                    put(key, args[0])

        return pre

    def out_hook(name):
        def post(mod, args, out):
            if STATE["layer_capture"]:
                key = f"call0.{name}"
                if key not in STATE["tensors"]:
                    put(key, out)

        return post

    nn.cond_factory.linear_out.register_forward_pre_hook(factory_hook("cond_factory"))
    nn.init_repr_factory.linear_out.register_forward_pre_hook(factory_hook("init_repr_factory"))
    nn.pair_repr_builder.init_repr_factory.linear_out.register_forward_pre_hook(
        factory_hook("pair_repr_builder.init_repr_factory")
    )
    nn.pair_repr_builder.cond_factory.linear_out.register_forward_pre_hook(
        factory_hook("pair_repr_builder.cond_factory")
    )
    nn.transition_c_2.register_forward_hook(out_hook("c"))
    nn.init_repr_factory.register_forward_hook(out_hook("s_init"))
    nn.pair_repr_builder.register_forward_hook(out_hook("z_init"))

    def layer_hook(kind, i):
        def post(mod, args, out):
            if not STATE["layer_capture"]:
                return
            if kind == "attn" and i == 0:
                STATE["cycle_layer0_calls"] += 1
                if STATE["cycle_layer0_calls"] == 1:
                    put("call0.cycle0.s_in", args[0])
                    put("call0.cycle0.z_in", args[1])
            if STATE["cycle_layer0_calls"] != 1:
                return
            put(f"call0.cycle0.{kind}.{i}", out)

        return post

    for i, layer in enumerate(nn.transformer_layers):
        layer.register_forward_hook(layer_hook("attn", i))
    for i, layer in enumerate(nn.pair_update_layers):
        if layer is not None:
            layer.register_forward_hook(layer_hook("pair", i))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="/home/ubuntu/localdesign/oracle-dumps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--no-layers", action="store_true", help="skip per-layer state dumps (large for big n)")
    args, hydra_overrides = ap.parse_known_args()

    torch.set_float32_matmul_precision("highest")
    install_rng_recorder()
    install_model_hooks()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.case}.run"
    shutil.rmtree(run_dir, ignore_errors=True)
    overrides = [
        "inference_experiment=quickstart",
        f"inputs={args.inputs}",
        f"out_dir={run_dir}",
        f"seed={args.seed}",
        "precision=32-true",
        "skip_existing=false",
        "hydra.run.dir=" + str(run_dir / "hydra"),
    ]
    if args.nsteps is not None:
        overrides.append(f"generation.args.nsteps={args.nsteps}")
    overrides += hydra_overrides
    STATE["meta"]["layers"] = not args.no_layers
    STATE["meta"].update({"case": args.case, "inputs": args.inputs, "seed": args.seed, "overrides": overrides})

    from rfproteina import generate

    # run_generation resets matmul precision from cfg.precision; keep fp32 exact.
    orig_set = torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision = lambda _p: orig_set("highest")
    sys.argv = ["rfd4-generate", *overrides]
    generate.main_hydra()

    cifs = sorted(run_dir.rglob("*_generated.cif"))
    if cifs:
        shutil.copy(cifs[0], out_dir / f"{args.case}.cif")
    from safetensors.torch import save_file

    tensors = STATE["tensors"]
    for k, v in list(tensors.items()):
        if v.dtype == torch.bool:
            tensors[k] = v.to(torch.uint8)
    save_file(tensors, str(out_dir / f"{args.case}.safetensors"))
    meta = {
        **STATE["meta"],
        "steps": STATE["steps"],
        "rng": STATE["rng"],
        "tensor_index": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
        "torch": torch.__version__,
        "upstream_commit": os.popen("git -C /home/ubuntu/RFD4-Proteina rev-parse HEAD").read().strip(),
    }
    (out_dir / f"{args.case}.json").write_text(json.dumps(meta, indent=1, default=str))
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1e6:.1f} MB, {len(STATE['rng'])} RNG draws, "
          f"{len(STATE['steps'])} trunk calls -> {out_dir}/{args.case}.*")


if __name__ == "__main__":
    main()

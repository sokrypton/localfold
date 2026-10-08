"""ESMFold2's own forward (the `esm` package, biohub's EsmFold2ExperimentalModel) on the native port's
OWN input: the oracle every CUDA stage is held to.

    ~/venv_ef2/bin/python cuda/esmfold2/oracle.py cuda/esmfold2/data-6mrr --out cuda/esmfold2/data-6mrr/oracle

It builds the feature dict the model's forward takes from exactly the arrays export_input.mjs wrote
(no featurisation of its own - the conformer included, so a difference is the network's), runs the
model on the CPU with its own ESM-C (biohub/ESMC-600M-1500000, loaded in float32 here), and records every seam:

  o/lm_hidden      [T, 37, 1152]  the 37 hidden states the shim mixes, per token
  o/lm_z           [T, T, 256]    the language model's pair term
  o/s_inputs       [T, 451]       the inputs embedder's output
  o/atom/linear, norm, block<i>, toToken   [A, ...] inside it
  o/z_init_1 o/z_init_2 o/rel_pos o/token_bonds   the other four terms of z_init
  o/loop<k>/in o/loop<k>/out      the trunk's pair into and out of each of its num_loops + 1 passes
  o/distogram      [T, T, 128]
  o/cond/single o/cond/pair       the diffusion conditioning at the first step
  o/step<k>/x_noisy, x_denoised   [A, 3] every denoiser call; m o/step<k>/t_hat
  o/coords         [A, 3]         the sampler's answer (seeded; the native port draws its own noise)

--float32-attention neutralises the atom attention's unconditional bfloat16 cast (the control that
separates a convention from that rounding).
"""
import argparse
import glob
import os
import pathlib
import sys

import numpy as np


def read_native(path):
    entries = {}
    raw = np.fromfile(os.path.join(path, "model.bin"), dtype=np.uint8)
    for line in open(os.path.join(path, "model.idx")):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "m":
            entries[parts[1]] = parts[2]
            continue
        kind, name, offset, length = parts[0], parts[1], int(parts[2]), int(parts[3])
        dtype = np.int32 if kind == "i" else np.float32
        entries[name] = raw[offset * 4:(offset + length) * 4].view(dtype).copy()
    return entries


def write_native(path, tensors, metas):
    os.makedirs(path, exist_ok=True)
    lines, offset = [], 0
    with open(os.path.join(path, "model.bin"), "wb") as handle:
        for name, value in tensors.items():
            flat = np.ascontiguousarray(np.asarray(value, np.float32)).ravel()
            lines.append(f"t {name} {offset} {flat.size}")
            handle.write(flat.tobytes())
            offset += flat.size
    lines += [f"m {k} {v}" for k, v in metas.items()]
    with open(os.path.join(path, "model.idx"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--esmfold2", default=os.path.join(os.path.dirname(__file__), "..", "..", "esmfold2-fast-600m"))
    parser.add_argument("--float32-attention", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # CPU by default: on CUDA the model runs its language model, inputs embedder and trunk under bf16
    # autocast (use_amp = ref_pos.device.type == "cuda"), which is not the float32 reference
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthyra", default="", help="the Synthyra/ESMFold2-600 snapshot (default: the HF cache's)")
    args = parser.parse_args()

    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.float32_attention:
        torch.Tensor.bfloat16 = lambda self: self
    from esm.models.esmfold2 import EsmFold2ExperimentalModel

    x = read_native(args.input)
    T, A = int(x["meta/tokens"]), int(x["meta/atoms"])
    C = int(x["meta/classes"])
    dev = args.device
    torch.set_num_threads(os.cpu_count() or 8)
    model = EsmFold2ExperimentalModel.from_pretrained(args.esmfold2, load_esmc=False).eval().to(dev)
    # ESM-C loaded HERE, in float32: the experimental model always loads its LM in bf16 (from_pretrained
    # discards esmc_precision), and casting afterwards keeps bf16-rounded weights. The hidden states go
    # through the package's own compute_lm_hidden_states (BOS/EOS per chain, sequence ids, the
    # per-residue collapse) and into the forward as lm_hidden_states
    from esm.models.esmc import EsmcModel
    from esm.models.esmfold2.layers import compute_lm_hidden_states
    esmc = EsmcModel.from_pretrained(model.config.esmc_id, device=dev, dtype=torch.float32).eval()

    L = lambda v: torch.as_tensor(np.asarray(v, np.int64), device=dev).reshape(1, -1)
    res_type = L(x["res_type"])
    features = dict(
        token_index=L(x["token_index"]), residue_index=L(x["residue_index"]), asym_id=L(x["asym_id"]),
        sym_id=L(x["sym_id"]), entity_id=L(x["entity_id"]), mol_type=L(x["mol_type"]), res_type=res_type,
        input_ids=L(x["input_ids"]),
        token_bonds=torch.as_tensor(x["token_bonds"], device=dev).reshape(1, T, T, 1),
        token_attention_mask=torch.ones(1, T, dtype=torch.bool, device=dev),
        ref_pos=torch.as_tensor(x["ref_pos"], device=dev).reshape(1, A, 3),
        ref_element=L(x["ref_element"]),
        ref_charge=torch.as_tensor(x["ref_charge"].astype(np.int8), device=dev).reshape(1, A),
        ref_atom_name_chars=L(x["ref_atom_name_chars"]).reshape(1, A, 4),
        ref_space_uid=L(x["ref_space_uid"]),
        atom_attention_mask=torch.as_tensor(x["atom_mask"] > 0, device=dev).reshape(1, A),
        atom_to_token=L(x["atom_to_token"]), distogram_atom_idx=L(x["distogram_atom_idx"]),
        msa=res_type.reshape(1, 1, T), msa_attention_mask=torch.ones(1, 1, T, dtype=torch.bool, device=dev),
        has_deletion=torch.zeros(1, 1, T, dtype=torch.bool, device=dev),
        deletion_value=torch.zeros(1, 1, T, device=dev), deletion_mean=torch.zeros(1, T, device=dev),
    )

    out, metas = {}, {}
    first = lambda v: v[0] if isinstance(v, (tuple, list)) else v
    keep = lambda name, v: out.setdefault(name, first(v).detach().float().cpu().numpy()[0])

    lm_hidden = compute_lm_hidden_states(esmc, features["input_ids"], features["asym_id"], features["residue_index"],
                                         features["mol_type"], features["token_attention_mask"]).float()
    keep("o/lm_hidden", lm_hidden)
    features["lm_hidden_states"] = lm_hidden
    hooks = []
    def on(module, name, transform=None):
        def hook(_m, inputs, output):
            if name not in out:
                keep(name, output if transform is None else transform(inputs, output))
        hooks.append(module.register_forward_hook(hook))
    on(model.language_model, "o/lm_z")
    on(model.inputs_embedder, "o/s_inputs")
    enc = model.inputs_embedder.atom_attention_encoder
    on(enc.atom_linear, "o/atom/linear")
    on(enc.atom_norm, "o/atom/norm")
    for i, block in enumerate(enc.atom_transformer.blocks):
        on(block, f"o/atom/block{i}")
    on(enc.atom_to_token_linear, "o/atom/toToken")
    for n in ("z_init_1", "z_init_2", "rel_pos", "token_bonds"):
        on(getattr(model, n), "o/" + n)
    loops = []
    def trunk_hook(_m, inputs, output):
        k = len(loops)
        loops.append(k)
        out[f"o/loop{k}/in"] = inputs[0].detach().float().cpu().numpy()[0]
        out[f"o/loop{k}/out"] = first(output).detach().float().cpu().numpy()[0]
    hooks.append(model.folding_trunk.register_forward_hook(trunk_hook))
    diffusion = model.structure_head.diffusion_module
    def cond_hook(_m, inputs, output):
        if "o/cond/single" not in out:
            out["o/cond/single"] = output[0].detach().float().cpu().numpy()[0]
            out["o/cond/pair"] = output[1].detach().float().cpu().numpy()[0]
    hooks.append(diffusion.conditioning.register_forward_hook(cond_hook))
    steps = []
    def step_hook(_m, inputs, keywords, output):
        k = len(steps)
        steps.append(k)
        out[f"o/step{k}/x_noisy"] = keywords["x_noisy"].detach().float().cpu().numpy().reshape(-1, 3)
        out[f"o/step{k}/x_denoised"] = output["x_denoised"].detach().float().cpu().numpy().reshape(-1, 3)
        metas[f"o/step{k}/t_hat"] = float(keywords["t_hat"].reshape(-1)[0])
    hooks.append(diffusion.register_forward_hook(step_hook, with_kwargs=True))

    result = model(**features, num_diffusion_samples=1, seed=args.seed)
    for h in hooks:
        h.remove()
    out["o/distogram"] = result["distogram_logits"].detach().float().cpu().numpy()[0]
    # Synthyra's confidence head (biohub ships none; Synthyra trained one on this frozen trunk), THEIR
    # module out of their bundle, on this fold's own trunk pair, s_inputs, coordinates and encodings
    snap = args.synthyra or max(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Synthyra--ESMFold2-600/snapshots/*/")), default="")
    if snap:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "oracle"))
        from dump_esmfold2_confidence import unpack_runtime
        sys.path.insert(0, str(unpack_runtime(pathlib.Path(snap) / "fastplms_bundle.py")))
        import json
        from safetensors.torch import load_file
        from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead
        head = ConfidenceHead(ESMFold2Config(**json.loads(open(os.path.join(snap, "config.json")).read()))).eval()
        state = {k[len("confidence_head."):]: v.float() for k, v in load_file(os.path.join(snap, "model.safetensors")).items()
                 if k.startswith("confidence_head.")}
        head.load_state_dict(state, strict=False)
        T_ = lambda name: torch.as_tensor(out[name]).unsqueeze(0)
        cout = head(s_inputs=T_("o/s_inputs"), z=T_(f"o/loop{len(loops) - 1}/out"),
                    x_pred=result["sample_atom_coords"].detach().float().reshape(1, A, 3).cpu(),
                    distogram_atom_idx=features["distogram_atom_idx"].cpu(), token_attention_mask=torch.ones(1, T, dtype=torch.long),
                    atom_to_token=features["atom_to_token"].cpu(), atom_attention_mask=features["atom_attention_mask"].long().cpu(),
                    asym_id=features["asym_id"].cpu(), mol_type=features["mol_type"].cpu(),
                    relative_position_encoding=T_("o/rel_pos"), token_bonds_encoding=T_("o/token_bonds"))
        for k, v in cout.items():
            if torch.is_tensor(v):
                a = v.detach().float().cpu().numpy()
                out["o/conf/" + k] = a[0] if a.ndim > 0 and a.shape[0] == 1 else a
        print("confidence:", {k: list(v.shape) for k, v in cout.items() if torch.is_tensor(v)})
    out["o/coords"] = result["sample_atom_coords"].detach().float().cpu().numpy().reshape(-1, 3)
    metas["o/loops"] = len(loops)
    metas["o/steps"] = len(steps)
    n = write_native(args.out, out, metas)
    print(f"{len(loops)} trunk passes, {len(steps)} denoiser calls; {len(out)} tensors, "
          f"{n * 4 / 2**20:.0f} MiB -> {args.out}")
    for k, v in out.items():
        if not k.startswith("o/step"):
            print(f"  {k:18s} {list(v.shape)}")


if __name__ == "__main__":
    main()

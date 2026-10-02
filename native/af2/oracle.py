"""af3-any-model's JAX AlphaFold 2 on the native port's OWN input: the oracle every CUDA stage is held to.

    ~/.venv-lfjax/bin/python native/af2/oracle.py native/af2/data-test59 \
        --weights native/af2/weights-model_1_ptm --out native/af2/data-test59/oracle

It reads the features native/af2/export_input.mjs wrote (pass 0), builds the batch the reference's
multimer graph reads from exactly those arrays - no featurisation of its own, so a difference is the
network's - and runs RunModel.apply in float32 (no bfloat16, XLA attention). Stages are tapped by
running the same network with fewer blocks (the stacked params sliced to match):

  o/embed/...     no extra-MSA block, no Evoformer block: the embedder's msa and pair
  o/extra1/...    after the first extra-MSA block
  o/extra/...     after the whole extra-MSA stack
  o/evo1/...      after the first Evoformer block
  o/full/...      the whole pass: msa_first_row, pair, single, the structure module, every head

Written in native/af3's model.idx/model.bin format, so the CUDA side loads it beside the input.
"""
import argparse
import copy
import os
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
            entries[parts[1]] = float(parts[2])
            continue
        kind, name, offset, length = parts[0], parts[1], int(parts[2]), int(parts[3])
        dtype = np.int32 if kind == "i" else np.float32
        entries[name] = raw[offset * 4:(offset + length) * 4].view(dtype)
    return entries


def write_native(path, arrays):
    os.makedirs(path, exist_ok=True)
    lines, offset = [], 0
    with open(os.path.join(path, "model.bin"), "wb") as handle:
        for name, value in arrays.items():
            flat = np.ascontiguousarray(np.asarray(value, np.float32)).ravel()
            lines.append(f"t {name} {offset} {flat.size}")
            handle.write(flat.tobytes())
            offset += flat.size
    with open(os.path.join(path, "model.idx"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="model_1_ptm")
    parser.add_argument("--params", default=os.path.expanduser("~/lfjax/af2_params"))
    parser.add_argument("--reference", default="/tmp/claude-1000/ref")
    parser.add_argument("--out", required=True)
    parser.add_argument("--taps", default="embed,extra1,extra,evo1,full")
    parser.add_argument("--pass", dest="pass_index", type=int, default=0)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]    # absl, somewhere in the reference, parses the command line as its own flags
    sys.path[:0] = [os.path.join(args.reference, "dev", "oracles")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    # float32 means float32: JAX's default f32 matmul on an A100 is TF32 (a 10-bit mantissa), and
    # the reference picks a Pallas flash attention at TF32 for long axes - both turned off below
    jax.config.update("jax_default_matmul_precision", "highest")
    from alphafold3.af2.runner import AF2Runner
    from alphafold3.af2.model import model as af2_model

    x = read_native(args.input)
    w = read_native(args.weights)
    L = int(x["meta/tokens"])
    N = int(x["meta/msa_rows"])
    E = int(x["meta/extra_rows"])
    k = args.pass_index
    multimer = bool(w["meta/multimer"])

    runner = AF2Runner(model_type="alphafold2_multimer_v3" if multimer else "alphafold2_ptm",
                       use_templates=False, data_dir=args.params, model_names=[args.model],
                       num_recycle=0, use_bfloat16=False, recycle_remat=False)
    params = runner.model_params[0]

    aatype = np.clip(x["aatype"].astype(np.int32), 0, 19)
    extra_codes = x[f"f{k}/extra_msa"].reshape(E, L)
    extra_onehot = np.eye(23, dtype=np.float32)[extra_codes]
    batch = {
        "aatype": jnp.asarray(aatype),
        "residue_index": jnp.asarray(x["residue_index"].astype(np.int32)),
        "seq_mask": jnp.asarray(x["seq_mask"]),
        "asym_id": jnp.asarray(x["asym_id"].astype(np.int32)),
        "entity_id": jnp.asarray(x["entity_id"].astype(np.int32)),
        "sym_id": jnp.asarray(x["sym_id"].astype(np.int32)),
        "target_feat": jnp.asarray(np.eye(20, dtype=np.float32)[aatype]),
        "msa_feat": jnp.asarray(x[f"f{k}/msa_feat"].reshape(N, L, 49)),
        "msa_mask": jnp.asarray(x[f"f{k}/msa_mask"].reshape(N, L)),
        "extra_msa_feat": jnp.asarray(np.concatenate([
            extra_onehot, x[f"f{k}/extra_has_deletion"].reshape(E, L, 1),
            x[f"f{k}/extra_deletion_value"].reshape(E, L, 1)], -1)),
        "extra_msa_mask": jnp.asarray(x[f"f{k}/extra_msa_mask"].reshape(E, L)),
        "use_dropout": False,
        "mask_template_interchain": False,
        "position_scale": jnp.asarray(float(w["meta/position_scale"]), jnp.float32),
        "opm_first": jnp.asarray(float(w["meta/opm_first"]), jnp.float32),
        "prev": {"prev_msa_first_row": jnp.zeros([L, 256], jnp.float32),
                 "prev_pair": jnp.zeros([L, L, 128], jnp.float32),
                 "prev_pos": jnp.zeros([L, 37, 3], jnp.float32)},
    }

    evo = runner.cfg.model.embeddings_and_evoformer
    full_extra, full_evo = int(evo.extra_msa_stack_num_block), int(evo.evoformer_num_block)
    plan = {"embed": (0, 0), "extra1": (1, 0), "extra": (full_extra, 0), "evo1": (full_extra, 1),
            "full": (full_extra, full_evo)}
    out = {}
    for tap in args.taps.split(","):
        n_extra, n_evo = plan[tap]
        cfg = copy.deepcopy(runner.cfg)
        cfg.model.global_config.flash_attention = None
        # layer_stack cannot run zero blocks, so "none" is ONE block whose every residual is zero:
        # each sub-module adds through a final projection (attention output_w/output_b, transition2,
        # the triangle's output_projection, the outer product's output_w/output_b), and zeroing those
        # makes the block an exact identity
        cfg.model.embeddings_and_evoformer.extra_msa_stack_num_block = max(n_extra, 1)
        cfg.model.embeddings_and_evoformer.evoformer_num_block = max(n_evo, 1)
        if tap != "full":
            # only the representations are compared; the heads would need the structure module
            for head in ("structure_module", "predicted_lddt", "predicted_aligned_error",
                         "experimentally_resolved", "masked_msa", "distogram"):
                if head in cfg.model.heads:
                    cfg.model.heads[head].weight = 0.0
        final = ("output_w", "output_b")
        def cut(module, values, n):
            kept = {key: v[:max(n, 1)] for key, v in values.items()}
            if n == 0 and (module.endswith("/attention") or module.endswith("outer_product_mean")
                           or module.endswith("transition2") or module.endswith("output_projection")):
                kept = {key: (np.zeros_like(v) if (key in final or module.endswith("transition2")
                                                   or module.endswith("output_projection")) else v)
                        for key, v in kept.items()}
            return kept
        sliced = {}
        for module, values in params.items():
            if "extra_msa_stack" in module:
                sliced[module] = cut(module, values, n_extra)
            elif "evoformer_iteration" in module:
                sliced[module] = cut(module, values, n_evo)
            else:
                sliced[module] = values
        runner_model = af2_model.RunModel(cfg, use_multimer=True)
        result = runner_model.apply(sliced, jax.random.PRNGKey(0), copy.copy(batch))
        rep = result["representations"]
        for key in ("msa", "pair", "single", "msa_first_row"):
            if key in rep:
                out[f"o/{tap}/{key}"] = np.asarray(rep[key])
        if tap == "full":
            sm = result["structure_module"]
            out["o/full/final_atom_positions"] = np.asarray(sm["final_atom_positions"])
            out["o/full/final_atom14_positions"] = np.asarray(sm["final_atom14_positions"])
            out["o/full/traj"] = np.asarray(sm["traj"])
            out["o/full/structure_act"] = np.asarray(rep["structure_module"])
            out["o/full/angles"] = np.asarray(sm["sidechains"]["angles_sin_cos"])
            out["o/full/plddt_logits"] = np.asarray(result["predicted_lddt"]["logits"])
            out["o/full/pae_logits"] = np.asarray(result["predicted_aligned_error"]["logits"])
            out["o/full/distogram_logits"] = np.asarray(result["distogram"]["logits"])
            out["o/full/masked_msa_logits"] = np.asarray(result["masked_msa"]["logits"])
        print(f"{tap}: extra {n_extra} evo {n_evo} done", flush=True)
    write_native(args.out, out)
    print(f"wrote {len(out)} tensors -> {args.out}")


if __name__ == "__main__":
    main()

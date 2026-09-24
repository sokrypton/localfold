"""af3-any-model's OpenDDE structural expander and refiner, inputs and outputs.

    ~/af3_venv/bin/python tools/oracle/dump_af3_opendde_structural.py

🔴 THE ONE OPENDDE STAGE WITH NO ORACLE, AND THE ONE THE RESIDUAL IS IN.
`check-opendde-expander.js` says so in its own header - "THERE IS NO ORACLE FOR
THIS, SO WHAT IS CHECKED IS CONSERVATION AND SHAPE" - so the expander and the
four-block refiner have been gated on invariants that hold by construction and
on nothing numeric. Measured against a real native fold of 6MRR, the structural
single this stage hands the confidence head is **relRMS 9.6%** from the
reference's (corr 0.9962), which arrives at the head as pLDDT 3.0% and PAE
10.0%. `s_inputs` into the same head matches to six digits, which is the control
that says this is the stage and not a misalignment.

WHAT IS CAPTURED, and why each:

    in.*            the expander's residue-level inputs and its structbook, so
                    our expander can be driven with the REFERENCE's inputs and
                    the stage isolated from the trunk that feeds it
    expander.*      tf_struct / s_struct / z_struct / attn_bias, so the expander
                    and the refiner can be blamed separately
    refiner.block*  the single after each of the four blocks, which is what
                    localises a defect to a block. The pair is NOT captured per
                    block: at 160x160x384 it is 39 MB a copy and the single is
                    where the channel structure shows.
    refiner.*       the single and pair the confidence head actually reads

🔴 THE REFINER IS `hk.experimental.layer_stack`, SO `hk.intercept_methods` SEES
NOTHING INSIDE IT - CLAUDE.md's own note - and every tensor here is a TRACER
besides. Both are why the taps are `jax.debug.callback` and not `np.asarray`,
which raises TracerArrayConversionError.

🔴 AND THE BLOB ON THIS BOX IS A 2026-09-09 CONVERSION (831-row single
conditioning) WHERE `opendde` JOINED `PADDED_SINGLE_COND` ON 2026-09-10 (833).
LocalFold's bundle IS the re-export, so the two tensors are spliced in from it
and the reference runs at the convention the port ships. Without that the dump
varies the convention as well as the stage. Measured: the convention is worth
relRMS 0.0021 on the PAE, i.e. nothing, but a comparison that moves two things
is not a comparison. Set BUNDLE= to point at another export.
"""
import json
import os
import sys

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
REF = os.path.expanduser(os.environ.get("AF3REF", "~/af3ref"))
for entry in (f"{REF}/src", REF, f"{REF}/dev/oracles"):
    sys.path.insert(0, entry)
sys.argv = sys.argv[:1]                       # tokamax parses argv lazily

import numpy as np                            # noqa: E402
import jax                                    # noqa: E402
import jax.numpy as jnp                       # noqa: E402

BUNDLE = os.environ.get("BUNDLE", "model-opendde-full-f32")
TARGET = os.path.expanduser(os.environ.get("TARGET", "~/6MRR.pdb"))
OUT = os.environ.get(
    "OUT", os.path.join(os.path.dirname(__file__), "..", "..", "oracle-dumps",
                        "af3-oracle-structural-opendde.json"))
PAD = ("diffuser/~/diffusion_head/single_cond_initial_norm",
       "diffuser/~/diffusion_head/single_cond_initial_projection")


def padded_tensors():
    """The 833-row scale and projection, read out of LocalFold's own bundle."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", BUNDLE)
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    out = {}
    for name in (f"{PAD[0]}/scale", f"{PAD[1]}/weights"):
        record = manifest["tensors"][name]
        count = int(np.prod(record["shape"]))
        with open(os.path.join(root, record["file"]), "rb") as handle:
            handle.seek(record["byteOffset"])
            out[name] = np.frombuffer(handle.read(count * 4),
                                      dtype="<f4").reshape(record["shape"]).copy()
    return out


GRABBED = {}
WITH_PAIR = os.environ.get("WITH_PAIR", "0") == "1"


def keep(name):
    """A host-side sink. `%d` in the name is the CALL index, which is how the
    refiner's four blocks are told apart: `layer_stack` traces its body once and
    scans it, so there is no trace-time block index to use - only the order the
    callbacks actually fire in."""
    def sink(value):
        key = name % len([k for k in GRABBED if k.startswith(name.split("%d")[0])
                          and k.endswith(name.split("%d")[-1])]) if "%d" in name else name
        if key not in GRABBED:
            GRABBED[key] = np.asarray(value)
    return sink


def tap(name, value, big=False):
    # 🔴 THE TWO STRUCTURAL PAIRS ARE 160x160x384 EACH - 39 MB of float and
    # about 190 MB of JSON apiece, which is most of the dump and none of what
    # localises a defect in the SINGLE. WITH_PAIR=1 asks for them. The RESIDUE
    # pair is not one of these: at 68x68x384 it is a tenth the size and the
    # checker cannot drive our expander without it.
    if big and not WITH_PAIR:
        return
    jax.debug.callback(keep(name), value)


def main():
    from alphafold3.model import params as afp
    from alphafold3.model import model as af3_model
    from alphafold3.model.network import structural_tokens
    from alphafold3.model.network import modules as pairformer_modules
    from alphafold3.model.network import diffusion_transformer
    import haiku as hk
    import fold_check

    pad = padded_tensors()
    original_params = afp.get_model_haiku_params

    def with_padding(model_dir):
        params = original_params(model_dir=model_dir)
        params[PAD[0]]["scale"] = jnp.asarray(pad[f"{PAD[0]}/scale"])
        params[PAD[1]]["weights"] = jnp.asarray(pad[f"{PAD[1]}/weights"])
        return params

    afp.get_model_haiku_params = with_padding

    # The stage under test, re-expressed with taps. It is a copy of
    # `Model._structural_expand_refine` and must stay one: a tap that drifts from
    # the code it taps is a dump of something nobody runs.
    def traced(self, embeddings, residue_batch, struct_data, struct_mask):
        gc = self.global_config
        c_s = self.config.evoformer.seq_channel
        c_z = self.config.evoformer.pair_channel
        c_s_inputs = embeddings["target_feat"].shape[-1]
        book = {k[len("structbook/"):]: v for k, v in struct_data.items()
                if k.startswith("structbook/")}
        batch_struct = {
            "parent_residue_idx": book["parent_residue_idx"],
            "subtoken_role_id": book["subtoken_role_id"],
            "prev_parent_residue_idx": book["prev_parent_residue_idx"],
            "next_parent_residue_idx": book["next_parent_residue_idx"],
            "residue_index": residue_batch.token_features.residue_index,
            "asym_id": residue_batch.token_features.asym_id,
        }
        tap("in.targetFeat", embeddings["target_feat"])
        tap("in.single", embeddings["single"])
        tap("in.pair", embeddings["pair"])
        tap("in.seqMask", struct_mask)
        for field, value in batch_struct.items():
            tap(f"in.book.{field}", value)

        tf_struct, s_struct, z_struct, attn_bias = (
            structural_tokens.StructuralTokenExpander(c_s, c_z, c_s_inputs, gc)(
                batch_struct, embeddings["target_feat"], embeddings["single"],
                embeddings["pair"]))
        tap("expander.targetFeat", tf_struct)
        tap("expander.single", s_struct)
        tap("expander.pair", z_struct, big=True)
        tap("expander.attnBias", attn_bias)

        ref_cfg = pairformer_modules.PairFormerIteration.Config(
            num_layer=1,
            pair_attention=pairformer_modules.GridSelfAttention.Config(
                num_head=c_z // 32),
            single_attention=diffusion_transformer.SelfAttentionConfig(num_head=8),
            pair_transition=pairformer_modules.TransitionBlock.Config(
                num_intermediate_factor=2),
            single_transition=pairformer_modules.TransitionBlock.Config(
                num_intermediate_factor=4))
        ref_cfg.shard_transition_blocks = False
        seq_mask = struct_mask.astype(jnp.float32)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        # 🔴 THE BLOCK IS TRACED ONCE AND SCANNED, so there is no trace-time
        # block index: `%d` in the tap name is filled from the order the
        # callbacks fire at RUN time, which is the only thing that separates
        # the four. A trace-time counter reaches 1 and dumps block 0 alone,
        # which is what the first version of this did.
        def blk(carry):
            zz, ss = carry
            out_z, out_s = pairformer_modules.PairFormerIteration(
                ref_cfg, gc, with_single=True, name="trunk_pairformer")(
                    zz, pair_mask, single_act=ss, seq_mask=seq_mask,
                    extra_pair_bias=attn_bias)
            tap("refiner.block%d.single", out_s)
            return out_z, out_s

        z_struct, s_struct = hk.experimental.layer_stack(
            4, name="structural_token_refiner")(blk)((z_struct, s_struct))
        tap("refiner.single", s_struct)
        tap("refiner.pair", z_struct, big=True)
        return s_struct, z_struct, tf_struct, attn_bias

    af3_model.Model._structural_expand_refine = traced

    sequence, _ = fold_check.parse_ca(TARGET)
    print(f"target {TARGET} · {len(sequence)} residues", flush=True)
    fold_check.fold("opendde", sequence,
                    model_dir=os.path.expanduser("~/ported/opendde"))

    stages = {}
    for name, value in sorted(GRABBED.items()):
        array = np.asarray(value)
        flat = array.reshape(-1).astype(np.float64)
        stages[name] = {
            "shape": list(array.shape),
            "rms": float(np.sqrt((flat ** 2).mean())),
            "data": [float(x) for x in flat],
        }
        print(f"  {name:28s} {array.shape}  rms {stages[name]['rms']:.6f}", flush=True)

    out = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"model": "opendde", "target": os.path.basename(TARGET),
                   "sequence": sequence, "bundle": BUNDLE,
                   "withPair": WITH_PAIR, "stages": stages}, handle)
    print(f"wrote {out} ({os.path.getsize(out) / 2 ** 20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()

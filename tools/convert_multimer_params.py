#!/usr/bin/env python3
"""Normalise AF2 monomer OR multimer parameters onto one graph.

THE MULTIMER GRAPH IS THE SUPERSET, AND IT IS THE ONE TO KEEP.

ColabDesign2 settled this the hard way (../ColabDesign2/colabdesign2/af2/
MERGE_NOTES.md): they made the multimer graph the single graph, converted the
monomer weights up onto it, and deleted the monomer forward code. Their merge is
finished - no *_multimer.py files remain.

The direction matters and it is easy to get backwards. Converting multimer
weights DOWN into the monomer layout produces something that runs and is not
multimer: the monomer graph has no relative-chain encoding, so asym/entity/sym
have nowhere to go and every cross-chain pair reads as a plain offset. Monomer
is the special case of multimer - one chain, one entity - so the superset is
what both should land on.

WHAT EACH SOURCE NEEDS:

    monomer     rel-pos 65 -> 73 (zero-pad the chain columns), alphabet 22 -> 21
                (drop the LEADING row), IPA already fused, OPM last, scale 10
    multimer    rel-pos and alphabet pass through, IPA split -> fused with zero
                scalar bias, OPM first, scale 20

The IPA is the one place LocalFold keeps the MONOMER layout, and that is not a
contradiction: fusing is a memory packing, not a graph difference - the same
arithmetic either way - and the fused form is itself the superset, because it
HAS the scalar bias slot that multimer's split form lacks. That slot is exactly
what the reference merge had to add by hand as its Stage 1; LocalFold gets it
for free by keeping the fused kernels.

WHY THIS IS NUMPY AND NOT WGSL. Every bug that merge hit was a layout mistake -
alphabet pad direction, coord-major versus head-major points, per-head versus
flat kv split - each silent, each costing a session. In numpy they are visible
against a reference; in a shader, on a machine whose GPU test path does not run,
they would not be.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SM = "alphafold/alphafold_iteration/structure_module/fold_iteration/"
IPA = SM + "invariant_point_attention/"

# AF2's IPA geometry, identical in both graphs: 12 heads, 16 scalar channels,
# 4 query/key points and 8 value points, each point carrying 3 coordinates.
HEADS, SCALAR, POINT_QK, POINT_V = 12, 16, 4, 8


def load_params(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Group a haiku npz's flat "module//leaf" keys into modules."""
    raw = np.load(path, allow_pickle=True)
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key in raw.files:
        module, _, leaf = key.rpartition("//")
        grouped.setdefault(module, {})[leaf] = np.asarray(raw[key])
    return grouped


def _fuse_scalar(parts: list[np.ndarray]) -> np.ndarray:
    """(D, head, 16) parts -> (D, head * sum(widths)).

    🔴 THE SPLIT IS PER HEAD, NOT A FLAT OFFSET. Monomer's kv_scalar is
    (head, 32) sliced at 16 within each head, so k and v interleave head by
    head. Concatenating the flat projections instead - k's 192 then v's 192 -
    puts whole heads where half-heads belong and every attention logit is wrong.
    """
    return np.concatenate(parts, axis=2).reshape(parts[0].shape[0], -1)


def _fuse_point(parts: list[np.ndarray]) -> np.ndarray:
    """(D, head, 3*points) parts -> (D, 3 * head * sum(points)).

    🔴 MONOMER PACKS POINTS COORD-MAJOR, MULTIMER HEAD-MAJOR. The flat monomer
    projection is [x(all heads), y(all heads), z(all heads)]; multimer stores
    (head, [x..., y..., z...]). A flat reshape between them transposes the two
    and silently scrambles every point coordinate.
    """
    depth = parts[0].shape[0]
    stacked = [part.reshape(depth, HEADS, 3, -1) for part in parts]
    fused = np.concatenate(stacked, axis=3)
    return fused.transpose(0, 2, 1, 3).reshape(depth, -1)


def _fuse_point_bias(parts: list[np.ndarray]) -> np.ndarray:
    stacked = [part.reshape(HEADS, 3, -1) for part in parts]
    fused = np.concatenate(stacked, axis=2)
    return fused.transpose(1, 0, 2).reshape(-1)


def convert_structure_module(params: dict) -> dict[str, dict[str, np.ndarray]]:
    """The split multimer IPA, fused the way LocalFold's kernels read it."""
    def module(name: str) -> dict[str, np.ndarray]:
        if name not in params:
            raise KeyError(f"multimer parameters have no {name}")
        return params[name]

    q_scalar = module(IPA + "q_scalar_projection")["weights"]
    k_scalar = module(IPA + "k_scalar_projection")["weights"]
    v_scalar = module(IPA + "v_scalar_projection")["weights"]
    q_point = module(IPA + "q_point_projection/point_projection")
    k_point = module(IPA + "k_point_projection/point_projection")
    v_point = module(IPA + "v_point_projection/point_projection")

    return {
        # 🔴 ZERO BIAS IS CORRECT HERE, not a dropped value. Multimer's scalar
        # projections have no bias at all - the npz carries no such key - while
        # monomer's are real and nonzero. Converting the other way loses them
        # (|max| 0.90 and 0.19); converting this way there is nothing to lose.
        IPA + "q_scalar": {
            "weights": _fuse_scalar([q_scalar]),
            "bias": np.zeros(HEADS * SCALAR, np.float32),
        },
        IPA + "kv_scalar": {
            "weights": _fuse_scalar([k_scalar, v_scalar]),
            "bias": np.zeros(HEADS * 2 * SCALAR, np.float32),
        },
        IPA + "q_point_local": {
            "weights": _fuse_point([q_point["weights"]]),
            "bias": _fuse_point_bias([q_point["bias"]]),
        },
        IPA + "kv_point_local": {
            "weights": _fuse_point([k_point["weights"], v_point["weights"]]),
            "bias": _fuse_point_bias([k_point["bias"], v_point["bias"]]),
        },
        # ...a rename. The composition is equivalent: q (+) q (x) update, with
        # the translation rotated. Confirmed bit-exact by the reference merge.
        SM + "affine_update": module(SM + "quat_rigid/rigid"),
    }


TRIANGLE_RENAMES = {"left_norm_input": "layer_norm_input", "center_norm": "center_layer_norm"}
TRIANGLE_SPLITS = {"projection": ("left_projection", "right_projection"),
                   "gate": ("left_gate", "right_gate")}


def convert_triangle_multiplication(params: dict, scope: str) -> dict[str, dict[str, np.ndarray]]:
    """Multimer's fused triangle multiplication, split the way monomer stores it.

    Multimer computes both sides in one projection and one gate, each twice the
    intermediate width, and slices the result:

        left_proj_act  = proj_act[:, :, :c]
        right_proj_act = proj_act[:, :, c:]

    (modules.py, _fused_triangle_multiplication). So the split is FLAT and left
    comes first - there are no heads here, unlike the IPA, where the same-looking
    fusion interleaves per head and a flat slice takes whole heads instead of
    half of each. The two layer norms are renames.

    Weights arrive stacked over blocks, so the channel axis is the last one.
    """
    converted: dict[str, dict[str, np.ndarray]] = {}
    for direction in ("outgoing", "incoming"):
        base = f"{scope}triangle_multiplication_{direction}/"
        for source, target in TRIANGLE_RENAMES.items():
            converted[base + target] = params[base + source]
        for source, (left, right) in TRIANGLE_SPLITS.items():
            fused = params[base + source]
            width = fused["weights"].shape[-1] // 2
            converted[base + left] = {
                "weights": fused["weights"][..., :width], "bias": fused["bias"][..., :width]}
            converted[base + right] = {
                "weights": fused["weights"][..., width:], "bias": fused["bias"][..., width:]}
    return converted


EVO = "alphafold/alphafold_iteration/evoformer/"


def convert_monomer_relative_encoding(params: dict) -> dict[str, np.ndarray]:
    """Monomer's 65-row relative encoding, widened to the multimer graph's 73.

    73 = 66 offset bins (the last meaning "different chain") + 1 entity-same +
    6 relative-chain bins. A single chain never lights any of the 8 trailing
    features, so zero columns there reproduce the monomer contribution exactly.
    """
    source = params[EVO + "pair_activiations"]
    weights = source["weights"]
    padding = np.zeros((73 - weights.shape[0], weights.shape[1]), weights.dtype)
    return {"weights": np.concatenate([weights, padding], axis=0), "bias": source["bias"]}


def convert_monomer_alphabet(params: dict) -> dict[str, dict[str, np.ndarray]]:
    """Monomer's 22-wide target features, narrowed to the multimer graph's 21.

    🔴 DROP THE LEADING ROW, NOT THE TRAILING ONE. The two graphs pad the target
    features differently - monomer [1,1] so the real restypes sit at 1..20,
    multimer [0,1] so they sit at 0..19. Dropping the trailing row instead
    shifts every restype by one, which raises nothing and puts about 1.0 of
    error into the pair init. That is the bug that cost the reference merge a
    session, and it is the reason this is a named function rather than a slice.
    """
    converted = {}
    for name in ("left_single", "right_single", "preprocess_1d"):
        source = params[EVO + name]
        entry = {"weights": source["weights"][1:]}
        if "bias" in source:
            entry["bias"] = source["bias"]
        converted[EVO + name] = entry
    return converted


EVOFORMER_SCOPES = {
    "evoformerStack": EVO + "evoformer_iteration/",
    "extraMsaStack": EVO + "extra_msa_stack/",
    # ...the template embedder's pair stack is an evoformer block like any
    # other, and fuses its triangle multiplication the same way.
    "templateStack": EVO + "template_embedding/single_template_embedding/"
                           "template_embedding_iteration/",
}


def widen_multimer_alphabet(params: dict) -> dict[str, dict[str, np.ndarray]]:
    """Multimer's 21-wide target features, padded to the 22 LocalFold builds.

    🔴 INSERT A LEADING ZERO ROW, mirroring the drop-leading that goes the other
    way. Multimer puts the real restypes at 0..19 and monomer at 1..20, and
    LocalFold's feature builder emits the monomer's 22 channels - so a leading
    zero lands multimer's rows exactly where the features expect them. Padding
    the trailing end instead shifts every restype by one, which raises nothing.
    """
    converted = {}
    for name in ("left_single", "right_single", "preprocess_1d"):
        source = params[EVO + name]
        weights = source["weights"]
        padded = np.concatenate([np.zeros((1, weights.shape[1]), weights.dtype), weights], axis=0)
        entry = {"weights": padded}
        if "bias" in source:
            entry["bias"] = source["bias"]
        converted[EVO + name] = entry
    return converted


def convert_multimer_params(params: dict) -> dict[str, dict[str, np.ndarray]]:
    """Every multimer tensor that has to change shape or name for LocalFold.

    Modules absent from the result pass through unchanged - they are the ~100
    the two graphs already share.
    """
    converted: dict[str, dict[str, np.ndarray]] = {}
    converted.update(convert_structure_module(params))
    for scope in EVOFORMER_SCOPES.values():
        converted.update(convert_triangle_multiplication(params, scope))
    converted.update(widen_multimer_alphabet(params))
    # ...the relative encoding keeps its 73 rows; only the name changes, because
    # LocalFold's packer widens a 65-row table and passes a 73-row one through.
    converted[EVO + "pair_activiations"] = params[
        EVO + "~_relative_encoding/position_activations"]
    return converted


def report_unconverted(params: dict) -> list[str]:
    """What still stands between these weights and a multimer fold.

    Listed rather than guessed. Each needs a decision that a numerical oracle
    should make, and a wrong guess here fails silently rather than loudly.
    """
    notes = []
    notes.append(
        "the MSA pipeline is multimer's own - paired and unpaired blocks, its own clustering - "
        "and none of it is built. This is the largest piece left.")
    notes.append(
        "chain identity (asym/entity/sym) is built by src/input/chains.js but does not yet reach "
        "the model: src/multimer/model.js takes no chainLengths.")
    notes.append("run with outerProductMeanFirst: true and positionScale: 20 - both already exist.")
    notes.append(
        f"{sum(1 for name in params if 'template' in name)} template modules are architecturally "
        "different and are excluded; run template-free, as the reference merge does.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", type=Path, help="params_model_N_multimer_v3.npz")
    parser.add_argument("--output", type=Path, help="write the converted arrays here")
    args = parser.parse_args()

    params = load_params(args.params)
    converted = convert_multimer_params(params)
    print(f"{len(params)} modules in, {len(converted)} converted, "
          f"{len(params) - len(converted)} passed through\n")
    groups = {"structure module": "/structure_module/", "triangle multiplication": "triangle_multiplication",
              "alphabet": "_single", "relative encoding": "pair_activiations"}
    for label, needle in groups.items():
        names = sorted(n for n in converted if needle in n)
        if not names:
            continue
        print(f"  {label}:")
        for name in names[:4]:
            shapes = ", ".join(f"{leaf}{tuple(value.shape)}"
                               for leaf, value in sorted(converted[name].items()))
            print(f"    {name.split('evoformer/')[-1].split('fold_iteration/')[-1]:56s} {shapes}")
        if len(names) > 4:
            print(f"    ... {len(names)} in this group")

    print("\nNOT converted, and why:")
    for note in report_unconverted(params):
        print(f"  - {note}")

    if "preprocess_1d" in "".join(converted):
        pass
    if args.output is not None:
        flat = {f"{module}//{leaf}": value
                for module, tensors in converted.items() for leaf, value in tensors.items()}
        np.savez(args.output, **flat)
        print(f"\nwrote {args.output} ({len(flat)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

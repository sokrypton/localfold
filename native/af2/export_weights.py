"""AlphaFold 2's weights for the native CUDA port, as af3-any-model's JAX AF2 loads them.

    ~/.venv-lfjax/bin/python native/af2/export_weights.py --model model_1_ptm \
        --reference /tmp/claude-1000/ref --out native/af2/weights-model_1_ptm

ONE GRAPH: the reference runs every AF2 checkpoint on the MULTIMER network, converting a monomer's
params at load (alphafold3/af2/convert.py, verified bit-exact against the retired monomer graph), so
this writes the converted set and the CUDA side implements that one graph. The regime a monomer needs
on it - position scale 10, the outer product mean after the MSA stack - is written beside the weights.

Writes <out>/model.idx and <out>/model.bin in native/af3's format (raw float32/int32, one index line
an entry):  t <name> <offset> <length> | i ... | m <name> <value>.
  t w/<module>/<param>        a weight, haiku's own name less 'alphafold/alphafold_iteration/'
  m w/<module>/<param>#<k>    its k-th dimension (#r: the rank)
  t c/<table>                 a residue-constant table the structure module reads
  m meta/...                  the regime
"""
import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_1_ptm")
    parser.add_argument("--params", default=os.path.expanduser("~/lfjax/af2_params"))
    parser.add_argument("--reference", default="/tmp/claude-1000/ref")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path[:0] = [os.path.join(args.reference, "dev", "oracles")]
    from alphafold3.af2.runner import load_params
    from alphafold3.af2.convert import convert_monomer_params
    from alphafold3.af2.common import residue_constants as rc
    from alphafold3.af2.model import all_atom

    multimer = "multimer" in args.model
    params = load_params([args.model], args.params, False, multimer)[0]
    if not multimer:
        params = convert_monomer_params(params)

    entries = []   # (kind, name, array or number)
    prefix = "alphafold/alphafold_iteration/"
    for module in sorted(params):
        for name, value in sorted(params[module].items()):
            value = np.asarray(value, np.float32)
            key = "w/" + module.removeprefix(prefix) + "/" + name
            entries.append(("t", key, value))
            entries.append(("m", key + "#r", value.ndim))
            for k, d in enumerate(value.shape):
                entries.append(("m", f"{key}#{k}", d))
    tables = {
        "rigid_group_default_frame": rc.restype_rigid_group_default_frame,     # (21, 8, 4, 4)
        "atom14_to_rigid_group": rc.restype_atom14_to_rigid_group,             # (21, 14) int
        "atom14_rigid_group_positions": rc.restype_atom14_rigid_group_positions,  # (21, 14, 3)
        "atom14_mask": rc.restype_atom14_mask,                                 # (21, 14)
        "atom37_to_atom14": all_atom.RESTYPE_ATOM37_TO_ATOM14,                 # (21, 37) int
        "atom37_mask": all_atom.RESTYPE_ATOM37_MASK,                           # (21, 37)
        "atom14_to_atom37": all_atom.RESTYPE_ATOM14_TO_ATOM37,                 # (21, 14) int
    }
    for name, value in tables.items():
        value = np.asarray(value)
        kind = "i" if np.issubdtype(value.dtype, np.integer) else "t"
        entries.append((kind, "c/" + name, value.astype(np.int32 if kind == "i" else np.float32)))
    entries.append(("m", "meta/multimer", int(multimer)))
    entries.append(("m", "meta/position_scale", 20.0 if multimer else 10.0))
    entries.append(("m", "meta/opm_first", int(multimer)))

    os.makedirs(args.out, exist_ok=True)
    lines, offset = [], 0
    with open(os.path.join(args.out, "model.bin"), "wb") as handle:
        for kind, name, value in entries:
            if kind == "m":
                lines.append(f"m {name} {value}")
                continue
            flat = np.ascontiguousarray(value).ravel()
            lines.append(f"{kind} {name} {offset} {flat.size}")
            handle.write(flat.tobytes())
            offset += flat.size
    with open(os.path.join(args.out, "model.idx"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{args.model}: {sum(1 for e in entries if e[0] != 'm')} tensors, {offset * 4 / 2**20:.0f} MiB -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add AlphaFold 2's template SINGLE features to the published monomer bundles, leaving every shard
already there byte-identical.

    python3 tools/append_template_single.py [--params ~/lfjax/af2_params]

WHAT WAS MISSING. A monomer template reaches AF2 two ways: the pair stack (templateEmbedding) and
the template's torsion angles as extra MSA rows - `template_single_embedding` (57 -> 256) and
`template_projection` (256 -> 256), which sit at evoformer/ and not under template_embedding/, so
tools/export_monomer_model.py's section scopes never picked them up. Without them 5CAJ with its own
crystal folds to 2.53 A / pLDDT 74.5 where DeepMind's model_1_ptm gives 0.21 / 97.4.

WHAT THIS WRITES. One new float32 shard per bundle (80K parameters, 321 KiB - small enough that a
quantised copy would save nothing worth its error) and a `templateSingle` section naming it:
  model/                monomer: the four tensors, from params_model_1_ptm.npz
  model-mono-2-delta/   model_2_ptm has its own: carried WHOLE (a delta inherits any base tensor
                        its header does not name, which would hand model_2 model_1's values)
  model-mono-3..5-delta model_3/4/5_ptm have no template embedder: the four are `absent`
Then run tools/write_manifest_module.py for the five families and publish the five directories.
Idempotent: a bundle that already has the section is left alone.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E = "alphafold/alphafold_iteration/evoformer/"
LEAVES = [("template_single_embedding", "bias"), ("template_single_embedding", "weights"),
          ("template_projection", "bias"), ("template_projection", "weights")]
NAMES = [f"template_single_haiku_{k:04d}" for k in range(len(LEAVES))]


def section():
    params = {}
    for (module, leaf), name in zip(LEAVES, NAMES):
        params.setdefault(module, {})[leaf] = name
    return {"parameterFormat": "haiku",
            "note": "the templates' torsion features as MSA rows (AF2 monomer); float32",
            "parameters": params}


def append_shard(directory, manifest, values):
    """the tensors into one new float32 shard after the last; returns the shard's name"""
    files = sorted({r["file"] for r in manifest["tensors"].values()})
    index = max(int(f.split("-")[1].split(".")[0]) for f in files) + 1
    shard = f"weights-{index:02d}.f32.bin"
    offset, chunks = 0, []
    for name, v in zip(NAMES, values):
        flat = np.ascontiguousarray(v, dtype="<f4").ravel()
        manifest["tensors"][name] = {"file": shard, "shape": list(v.shape), "byteOffset": offset, "dtype": "float32"}
        chunks.append(flat); offset += flat.nbytes
    (directory / shard).write_bytes(b"".join(c.tobytes() for c in chunks))
    b = manifest["bundle"]
    b["tensors"] = len(manifest["tensors"]); b["shards"] = len(files) + 1; b["bytes"] = b["bytes"] + offset
    return shard


def checkpoint(params_dir, model):
    z = np.load(Path(params_dir) / f"params_{model}.npz")
    return [np.asarray(z[f"{E}{module}//{leaf}"], np.float32) for module, leaf in LEAVES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.expanduser("~/lfjax/af2_params"))
    args = ap.parse_args()

    base = ROOT / "model"
    m = json.loads((base / "manifest.json").read_text())
    if "templateSingle" in m:
        print("model/: already has templateSingle")
    else:
        shard = append_shard(base, m, checkpoint(args.params, "model_1_ptm"))
        m["templateSingle"] = section()
        m["float32Tensors"] = sorted(set(m["float32Tensors"]) | set(NAMES))
        (base / "manifest.json").write_text(json.dumps(m))
        print(f"model/: + {shard}")

    for k in (2, 3, 4, 5):
        d = ROOT / f"model-mono-{k}-delta"
        m = json.loads((d / "manifest.json").read_text())
        h = m["delta"]
        if set(NAMES) <= set(h["whole"]) | set(h["absent"]):
            print(f"{d.name}/: already decided")
            continue
        has = bool(checkpoint_has(args.params, f"model_{k}_ptm"))
        if has:
            shard = append_shard(d, m, checkpoint(args.params, f"model_{k}_ptm"))
            h["whole"] = h["whole"] + NAMES
            m["float32Tensors"] = sorted(set(m.get("float32Tensors", [])) | set(NAMES))
            print(f"{d.name}/: + {shard} (whole)")
        else:
            h["absent"] = h["absent"] + NAMES
            print(f"{d.name}/: absent")
        (d / "manifest.json").write_text(json.dumps(m))


def checkpoint_has(params_dir, model):
    z = np.load(Path(params_dir) / f"params_{model}.npz")
    present = [f"{E}{module}//{leaf}" in z for module, leaf in LEAVES]
    if any(present) and not all(present):
        raise SystemExit(f"{model}: only some of the template single features - not a checkpoint this knows")
    return all(present)


if __name__ == "__main__":
    main()

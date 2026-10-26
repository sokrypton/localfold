"""Dump the RFD4-Proteina *input pipeline* batch on CPU (no model, no GPU).

Runs only the upstream data pipeline used by ``rfd4-generate`` (``build_design_validation_pipeline``
+ ``get_contig_or_design_problem_dataset`` + ``collate_batch`` + ``preprocess_batch_for_generation``)
and writes the model batch as ``batch.<key>`` tensors, the same naming as ``dump_rfd4p.py``.

Extra tensors (prefix ``trace.``) record the upstream random choices so the JS port can replay them:

- ``trace.chain_perm`` / ``trace.molecule_perm``: the ``np.random.permutation`` draws used for
  ``chain_iid`` / ``molecule_iid`` (indexed by sorted unique upstream id);
- the realized expandable-segment lengths are recoverable from the batch itself.

Usage (from the RFD4-Proteina checkout, with its venv):
    CUDA_VISIBLE_DEVICES="" python /home/ubuntu/localdesign/tools/oracle/dump_batch_cpu.py \
        --case binder_spec --inputs docs/examples/inputs/foo.yaml [--seed 0] [--cutoff 64]

Writes ``<out>/<case>.safetensors`` and ``<out>/<case>.json`` (with ``inputs``; ``kind: "batch-only"``).
Use ``--atoms`` to also write ``<out>/<case>.atoms.json`` (the final atom array annotations), handy
for debugging the JS port.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402


class FixedLengths:
    """Stand-in ``np.random.Generator`` returning preset expandable-segment lengths."""

    def __init__(self, lengths: list[int]):
        self.lengths = list(lengths)

    def integers(self, lo: int, hi: int) -> int:
        n = self.lengths.pop(0)
        if not lo <= n < hi:
            raise ValueError(f"segment length {n} outside [{lo}, {hi - 1}]")
        return n


def _ccd_for(inputs: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfd4p_ccd import ccd_entries

    path = Path(inputs)
    if path.suffix in (".yaml", ".yml", ".json"):
        import yaml

        spec = yaml.safe_load(path.read_text())
        files = []
        for k, ex in spec.items():
            if k == "global_args" or not isinstance(ex, dict) or "input" not in ex:
                continue
            for p in ([ex["input"]] if isinstance(ex["input"], str) else ex["input"]):
                q = Path(p) if Path(p).is_absolute() else path.parent / p
                if q.suffix == ".cif" and q.exists():
                    files.append(q)
        return ccd_entries(files)
    return ccd_entries([path]) if path.suffix == ".cif" and path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--inputs", required=True, help="annotated CIF or condition-spec yaml/json")
    ap.add_argument("--out", default="/home/ubuntu/localdesign/oracle-dumps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff", type=int, default=64, help="path_length_search_cutoff (checkpoint: 64)")
    ap.add_argument("--seg-lengths", default=None,
                    help="comma-separated realized expandable-segment lengths, in sentinel (file) order")
    ap.add_argument("--atoms", action="store_true", help="also dump the final atom array as JSON")
    args = ap.parse_args()

    import lightning as L

    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.generate import get_contig_or_design_problem_dataset

    # Record the id permutations drawn by ConvertToAtomLevelProteinaFeatures.
    perms: list[np.ndarray] = []
    orig_perm = np.random.permutation

    def recorded_perm(x, *a, **k):
        out = orig_perm(x, *a, **k)
        perms.append(np.asarray(out).copy())
        return out

    np.random.permutation = recorded_perm

    L.seed_everything(args.seed, verbose=False)
    seg_rng = None
    if args.seg_lengths:
        seg_rng = FixedLengths([int(x) for x in args.seg_lengths.split(",")])
    pipeline = build_design_validation_pipeline(
        metrics_tags=[], path_length_search_cutoff=args.cutoff, expandable_segment_rng=seg_rng
    )
    dataset = get_contig_or_design_problem_dataset(args.inputs, num_replicates=1, transform=pipeline)
    sample = dataset[0]
    np.random.permutation = orig_perm

    batch = collate_batch([sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    # Mirror Proteina.preprocess_batch_for_generation.
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]

    tensors: dict[str, torch.Tensor] = {}
    for key, v in batch.items():
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous().clone()
            if t.dtype == torch.bool:
                t = t.to(torch.uint8)
            tensors[f"batch.{key}"] = t
    for i, p in enumerate(perms):
        tensors[f"trace.perm.{i}"] = torch.as_tensor(p, dtype=torch.int64)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(tensors, str(out_dir / f"{args.case}.safetensors"))
    meta = {
        "case": args.case,
        "kind": "batch-only",
        "inputs": args.inputs,
        "seed": args.seed,
        "path_length_search_cutoff": args.cutoff,
        "n_perms": len(perms),
        "batch_nontensor_keys": sorted(k for k, v in batch.items() if not isinstance(v, torch.Tensor)),
        "tensor_index": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
        # The dictionary entries the page would fetch for the input's components (ccd.js), so a
        # checker replays a raw structure's preparation with what upstream had.
        "ccd": _ccd_for(args.inputs),
    }
    (out_dir / f"{args.case}.json").write_text(json.dumps(meta, indent=1, default=str))

    if args.atoms:
        _dump_atoms(sample["atom_array"], out_dir / f"{args.case}.atoms.json")
        if hasattr(dataset, "file_paths"):  # annotated-CIF input: also dump the freshly parsed array
            parsed = dataset._load_from_path(dataset.file_paths[0])["atom_array"]
            _dump_atoms(parsed, out_dir / f"{args.case}.parsed.json")
    print(f"wrote {len(tensors)} tensors -> {out_dir}/{args.case}.safetensors", file=sys.stderr)


def _dump_atoms(aa, path: Path) -> None:
    """Write every annotation of an atom array, its coordinates and bonds as JSON (debugging aid)."""
    ann = {}
    for name in aa.get_annotation_categories():
        vals = np.asarray(aa.get_annotation(name))
        ann[name] = vals.tolist() if vals.dtype != object else [str(v) for v in vals]
    ann["coord"] = aa.coord.tolist()
    ann["bonds"] = aa.bonds.as_array().tolist() if aa.bonds is not None else []
    path.write_text(json.dumps(ann, default=str))


if __name__ == "__main__":
    main()

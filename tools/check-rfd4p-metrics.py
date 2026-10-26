#!/usr/bin/env python3
"""Hold the RFD4 design metrics to upstream's own metric manager, on the same sample.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-metrics.py [<job.json>:<dump dir> ...]

`localfold-rfd4p --dump-sample=<dir>` writes a design's decoded sample. For each one this builds
the job's batch with upstream's own pipeline, runs upstream's `process_single_sample_to_atom_array`
and `annotate_unindexed_insertion` over the dumped coordinates - which is the array upstream's
metrics read - and runs upstream's metric manager, instantiated from its own
`design_metrics.yaml`, and its pre-oracle filters from `all_pre_oracle_filters.yaml`. The page's
side is cuda/rfd4p/design-metrics.mjs over the same dump. It requires:

- the same atoms in the same order, and the same BONDS - the clash and bond-length metrics read
  them, and upstream's come from three places (the input's, and biotite's `connect_via_residue_names`
  whenever a generated residue added atoms);
- the batch is upstream's own pipeline's, handed the expandable-segment lengths this port
  realized (its draw is its own RNG's, as `dump_batch_cpu.py --seg-lengths` replays them);
- every metric value (integers exactly, fractions to 1e-9, RMSDs to 1e-4 A since upstream's
  superposition is float32) and the pre-oracle verdict, filter by filter, under each job's own
  `metrics_tags` (the page's `jobMetricsTags`).

With no arguments it reads the dumps gate.py's writer and metrics arms leave
(`cuda/rfd4p/gate-writer-*`; `gate-metrics-5o45` from `featurise/gate-metrics.json`, the job
built to reach hotspots, C_SEC and sequence probabilities, which no shipped preset does; and
`gate-groups-5o45` and `gate-partial-1brs`, which reach the motif RMSD over three coordinate
groups and `ca_rmsd_to_input` on a design that keeps the input's CAs).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = Path.home() / "RFD4-Proteina"
METRICS = ["amino_acid_composition", "bond_length_metrics", "clash_metrics", "hotspot_obedience",
           "sequence_adherence", "sequence_probability_adherence", "sse_distribution", "sse_accuracy",
           "unindexed_insertion_mapping", "sequence_similarity_to_input", "design_motif_allatom_rmsd",
           "design_unindexed_motif_allatom_rmsd", "ca_rmsd_to_input"]


class FixedLengths:
    """Upstream's expandable-segment draw, replaced by the lengths this port realized."""

    def __init__(self, lengths: list[int]):
        self.lengths = list(lengths)

    def integers(self, lo: int, hi: int) -> int:
        n = self.lengths.pop(0)
        if not lo <= n < hi:
            raise ValueError(f"segment length {n} outside [{lo}, {hi - 1}]")
        return n


def upstream(job: dict, spec_dir: Path, dump: Path, tmp: Path, segment_lengths: list[int], tags: list[str]) -> dict:
    import lightning as L
    import numpy as np
    import torch
    import yaml
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.generate import get_contig_or_design_problem_dataset
    from rfproteina.metrics.metric import MetricManager
    from rfproteina.transforms.unindexing import annotate_unindexed_insertion
    from rfproteina.utils.structure_export import process_single_sample_to_atom_array

    if job.get("input_as_annotated_cif") and not job.get("contig") and not job.get("conditions"):
        inputs = str((spec_dir / job["input"]).resolve())
    else:
        spec = {k: job[k] for k in ("input", "contig", "conditions", "input_as_annotated_cif") if k in job}
        spec["input"] = str((spec_dir / spec["input"]).resolve())
        inputs = str(tmp / "spec.json")
        Path(inputs).write_text(json.dumps({"design": spec}))  # upstream keys examples by name
    L.seed_everything(0, verbose=False)
    pipe = build_design_validation_pipeline(
        metrics_tags=[], path_length_search_cutoff=64,
        expandable_segment_rng=FixedLengths(segment_lengths) if segment_lengths else None)
    sample = get_contig_or_design_problem_dataset(inputs, num_replicates=1, transform=pipe)[0]
    pad = np.fromfile(dump / "pad.f32", dtype=np.float32)
    n = len(pad)
    ins = dump / "insertion.f32"
    res = process_single_sample_to_atom_array(
        coords=torch.from_numpy(np.fromfile(dump / "coors.f32", dtype=np.float32).reshape(n, 37, 3)),
        residue_type=torch.from_numpy(np.fromfile(dump / "residueType.i32", dtype=np.int32).astype(np.int64)),
        padding_mask=torch.from_numpy(pad > 0.5),
        input_atom_array=sample["atom_array"],
        atom_mask=torch.from_numpy(np.fromfile(dump / "atomMask.u8", dtype=np.uint8).reshape(n, 37).astype(bool)),
        insertion_logits=(torch.from_numpy(np.fromfile(ins, dtype=np.float32).reshape(n, n))
                          if ins.exists() and ins.stat().st_size == n * n * 4 else None),
    )
    mapping = res.get("predicted_insertion_mapping") or {}
    g = annotate_unindexed_insertion(res.get("full_array_with_unindexed") or res["generated_atom_array"], mapping)
    cfg = yaml.safe_load((UPSTREAM / "configs/metrics/inline_metrics/design_metrics.yaml").read_text())
    mm = MetricManager.instantiate_from_hydra({k: cfg[k] for k in METRICS})
    out = mm(network_input=None, network_output=None,
             extra_info={"example_id": "x", "metrics_tags": tags,
                         "full_array_with_unindexed": res.get("full_array_with_unindexed"),
                         "predicted_insertion_mapping": mapping},
             ground_truth_atom_array=res["input_atom_array"], predicted_atom_array=g)
    metrics = {k: v for k, v in out.items() if k != "example_id"}
    fcfg = yaml.safe_load((UPSTREAM / "configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml").read_text())
    verdict = MetricManager.instantiate_from_hydra(fcfg)(
        network_input=None, network_output=None,
        extra_info={"example_id": "x", "metrics_tags": tags, "metrics": metrics})
    bonds = (sorted((*sorted((int(i), int(j))), int(t)) for i, j, t in g.bonds.as_array())
             if g.bonds is not None else [])
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "filters": {k[:-5]: v for k, v in verdict.items() if k.endswith(".pass")},
        "atoms": [[str(g.chain_id[i]), int(g.res_id[i]), str(g.res_name[i]), str(g.atom_name[i])]
                  for i in range(g.array_length())],
        "bonds": sorted(set(bonds)),
    }


def page(spec: Path, dump: Path) -> dict:
    r = subprocess.run(["node", REPO / "cuda/rfd4p/design-metrics.mjs", f"--spec={spec}", "--with-array", dump],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    out = json.loads(r.stdout)
    out["metrics"] = {k: float("nan") if v == "NaN" else float(v) for k, v in out["metrics"].items()}
    out["filters"] = {k: float(v["pass"]) for k, v in out["preOracle"]["filters"].items()}
    out["bonds"] = sorted(tuple(b) for b in out["bonds"])
    return out


def same(key: str, a: float, b: float) -> bool:
    # An RMSD is a superposition, which upstream solves in float32 (biotite's SVD) and this
    # port in float64; every other value is counted or summed the same way on both sides.
    tolerance = 1e-4 if key.endswith(".rmsd") else 1e-9
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tolerance


def main() -> int:
    pairs = [a.split(":", 1) for a in sys.argv[1:]]
    if not pairs:
        for dump in sorted((REPO / "cuda/rfd4p").glob("gate-writer-*")):
            job = REPO / "cuda/rfd4p/featurise" / f"{dump.name}.json"
            if dump.is_dir() and (dump / "coors.f32").exists() and job.exists():
                pairs.append((str(job), str(dump)))
        for jname, dname in (("gate-metrics.json", "gate-metrics-5o45"), ("gate-groups-5o45.json", "gate-groups-5o45"),
                             ("gate-partial-1brs.json", "gate-partial-1brs")):
            crafted = REPO / "cuda/rfd4p" / dname
            if (crafted / "coors.f32").exists():
                pairs.append((str(REPO / "cuda/rfd4p/featurise" / jname), str(crafted)))
    bad = 0
    for spec, dump in pairs:
        spec, dump = Path(spec), Path(dump)
        if "input" not in json.loads(spec.read_text()):
            # A contig with no input structure is this port's convenience; upstream's
            # condition_spec requires `input:` and calls anything else the removed legacy dialect.
            print(f"  {dump.name:42s} skipped: no input, which upstream's condition_spec has no form for")
            continue
        got = page(spec, dump)
        with tempfile.TemporaryDirectory() as tmp:
            up = upstream(json.loads(spec.read_text()), spec.resolve().parent, dump, Path(tmp), got["segmentLengths"],
                          got["metricsTags"])
        problems = []
        if up["atoms"] != got["atoms"]:
            problems.append(f"atoms {len(up['atoms'])} against {len(got['atoms'])}")
        elif up["bonds"] != [tuple(b) for b in got["bonds"]]:
            only_up = sorted(set(up["bonds"]) - set(got["bonds"]))
            only_got = sorted(set(got["bonds"]) - set(up["bonds"]))
            show = lambda b: [f"{up['atoms'][i][2]}{up['atoms'][i][1]}:{up['atoms'][i][3]}" for i in b[:2]] + [b[2]]
            problems.append(f"bonds: {len(only_up)} only upstream {[show(b) for b in only_up[:3]]}, "
                            f"{len(only_got)} only here {[show(b) for b in only_got[:3]]}")
        for k in sorted(set(up["metrics"]) | set(got["metrics"])):
            if k not in up["metrics"] or k not in got["metrics"]:
                problems.append(f"{k}: {'missing here' if k in up['metrics'] else 'not upstream'}")
            elif not same(k, up["metrics"][k], got["metrics"][k]):
                problems.append(f"{k}: upstream {up['metrics'][k]!r}, here {got['metrics'][k]!r}")
        if up["filters"] != got["filters"]:
            problems.append(f"filters: upstream {up['filters']}, here {got['filters']}")
        bad += bool(problems)
        print(f"  {dump.name:42s} {len(up['metrics'])} metrics, {len(up['bonds'])} bonds  "
              f"{'ok' if not problems else 'DIFFERS'}")
        for p in problems[:8]:
            print(f"      {p}")
    print(f"{len(pairs)} designs, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hold the RFD4 refolding oracle (cuda/rfd4p/src/oracle.h) to upstream's, on the same designs.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-oracle.py [<job.json>:<dump dir>:<design stem> ...]

`localfold-rfd4p --refold --dump-sample=<dir> --out=<stem>.cif` refolds each design with our own
AlphaFold 3 and writes, under `<stem>_refold/`, the AF3 job it built and AF3's files. For each
design this rebuilds upstream's designed array from the dump (as tools/check-rfd4p-metrics.py
does) and holds the two halves of the oracle apart:

- THE INPUT. Upstream writes the design as mmCIF and reads it back with AF3's own
  `Input.from_mmcif` (`AF3Oracle._build_manifest`, `af3_runtime.build_fold_input_from_cif`); the
  port writes the job that round trip produces. Both are FEATURISED by AlphaFold 3 itself
  (~/.venv-lfjax, af3-any-model) and every field of the two batches compared - which is what the
  model reads, so a difference that does not reach it is not one.
- THE SCORES. Upstream's own oracle, refolding and pass-criteria managers, instantiated from
  `refolding_oracles/af3.yaml`, `all_refolding_metrics.yaml` and `all_pass_criteria.yaml`, run
  over the same refold - `AF3Oracle._dispatch_fold` handed our AF3's files in place of a
  container's, and upstream's worker's own `_augment_summary_confidences` applied to them - and
  every value is required to match the port's `<stem>.refold.json` (an RMSD to 1e-4 A, since
  upstream superposes in float32; everything else to 1e-9).

The fold in between is our AF3 port, which tools/check-batch-fields.js and cuda/af3's gates hold to
AlphaFold 3 on their own.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = Path.home() / "RFD4-Proteina"
JAX = Path.home() / ".venv-lfjax" / "bin" / "python"


def checker():
    spec = importlib.util.spec_from_file_location("metrics_checker", REPO / "tools" / "check-rfd4p-metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def designed_array(cm, job: dict, spec_dir: Path, dump: Path, segment_lengths: list[int]):
    """Upstream's designed array for a dumped sample (check-rfd4p-metrics.py's `upstream`, kept)."""
    import lightning as L
    import numpy as np
    import torch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.generate import get_contig_or_design_problem_dataset
    from rfproteina.transforms.unindexing import annotate_unindexed_insertion
    from rfproteina.utils.structure_export import process_single_sample_to_atom_array

    with tempfile.TemporaryDirectory() as tmp:
        spec = {k: job[k] for k in ("input", "contig", "conditions", "input_as_annotated_cif") if k in job}
        spec["input"] = str((spec_dir / spec["input"]).resolve())
        inputs = str(Path(tmp) / "spec.json")
        Path(inputs).write_text(json.dumps({"design": spec}))
        L.seed_everything(0, verbose=False)
        pipe = build_design_validation_pipeline(
            metrics_tags=[], path_length_search_cutoff=64,
            expandable_segment_rng=cm.FixedLengths(segment_lengths) if segment_lengths else None)
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
    # 🔴 THE ORACLE SEES THE DESIGN IN ITS INPUT'S FRAME. Upstream's generated-structure worker
    # saves the CIF through `align_atom_array_to_input_frame`, which moves the array IN PLACE, and
    # hands that same array on as `designed_atom_array` (metrics_utils.py) - so a refold's template
    # is cut from the aligned design, as the port's records are.
    from rfproteina.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
    return align_atom_array_to_input_frame(g)


def positional(k: int) -> str:
    """The name our AF3 gives the k-th chain of its output (cuda/af3/src/sampler.cuh `chainId`)."""
    out, a = "", k
    while True:
        out = chr(ord("A") + a % 26) + out
        if a < 26:
            return out
        a = a // 26 - 1


def relabel_cif(text: str, named: dict[str, str]) -> str:
    """An mmCIF with its `_atom_site` chain columns renamed."""
    lines = text.splitlines()
    columns, out = [], []
    for line in lines:
        if line.startswith("_atom_site."):
            columns.append(line.split(".", 1)[1].strip())
        elif columns and (line.startswith("ATOM") or line.startswith("HETATM")):
            fields = line.split()
            for col in ("label_asym_id", "auth_asym_id"):
                if col in columns:
                    i = columns.index(col)
                    fields[i] = named[fields[i]]
            line = " ".join(fields)
        out.append(line)
    return "\n".join(out) + "\n"


def augment_summary_confidences():
    """Upstream's worker's `_augment_summary_confidences`, from its own source: the module that
    holds it imports AlphaFold 3, which this environment has not, so the one function is lifted."""
    import json as _json
    import pathlib as _pathlib

    import numpy as _np
    source = (UPSTREAM / "src" / "rfproteina" / "af3" / "af3_runtime.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "_augment_summary_confidences")
    namespace = {"json": _json, "np": _np, "pathlib": _pathlib}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "af3_runtime.py", "exec"), namespace)
    return namespace["_augment_summary_confidences"]


def managers(tags: list[str]):
    import yaml
    from rfproteina.metrics.metric import MetricManager

    def load(rel):
        return yaml.safe_load((UPSTREAM / "configs/metrics" / rel).read_text())
    oracles = load("refolding_oracles/af3.yaml")
    # The entries this port runs; `oracle_apo` (a pre-oracle transform) is refused by the port.
    oracle_mm = MetricManager.instantiate_from_hydra({k: oracles[k] for k in ("oracle", "oracle_ppi", "oracle_dna")})
    refold_mm = MetricManager.instantiate_from_hydra(load("refolding_metrics/all_refolding_metrics.yaml"))
    pass_cfg = load("pass_criteria/all_pass_criteria.yaml")
    # `internal_coords` names motif_scaffold's thresholds through a hydra interpolation.
    pass_cfg["internal_coords"]["metric_thresholds"] = pass_cfg["motif_scaffold"]["metric_thresholds"]
    pass_mm = MetricManager.instantiate_from_hydra(pass_cfg)
    for mm in (oracle_mm, refold_mm, pass_mm):
        mm.on_validation_epoch_start(val_dataset_tags=[set(tags)])
    return oracle_mm, refold_mm, pass_mm


def upstream_scores(g, example_id: str, folds: Path, tags: list[str], prior: dict, work: Path) -> tuple[dict, dict]:
    """Upstream's oracle + refolding + pass criteria over our AF3's files, and its manifests."""
    from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
    from rfproteina.metrics.oracles.oracle_metric import OracleMetric
    from rfproteina.metrics.utils.metrics_utils import run_oracle_refolding_and_pass_criteria

    augment = augment_summary_confidences()
    manifests: dict[str, dict] = {}

    def dispatch(self, manifest_path, daemon_rank, n_designs):
        for entry in json.loads(Path(manifest_path).read_text()):
            # Our AF3 ran on `<example>_<oracle_name>`; upstream names the job `<example>`.
            ours = folds / f"{example_id}_{self.oracle_name}"
            out = Path(entry["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            name = entry["name"]
            # Our AF3 names its chains by position (A, B, ...) where AlphaFold 3 keeps the job's
            # ids: read back as the job's, as the port's oracle reads them (oracle.h `positionalChain`).
            job = json.loads(Path(f"{ours}_job.json").read_text())
            ids = [next(iter(e.values()))["id"] for e in job["sequences"]]
            named = {positional(k): cid for k, cid in enumerate(ids)}
            (out / f"{name}_model.cif").write_text(relabel_cif(Path(f"{ours}_model.cif").read_text(), named))
            summary = json.loads(Path(f"{ours}_model_summary_confidences.json").read_text())
            (out / f"{name}_summary_confidences.json").write_text(json.dumps(summary))
            full = json.loads(Path(f"{ours}_model_confidences.json").read_text())
            for key in ("atom_chain_ids", "token_chain_ids"):
                full[key] = [named[c] for c in full[key]]
            (out / f"{name}_confidences.json").write_text(json.dumps(full))
            augment(out, name)
            manifests[self.oracle_name] = entry

    AF3Oracle._dispatch_fold = dispatch
    AF3Oracle.setup = OracleMetric.setup        # no daemon: the fold is ours
    oracle_mm, refold_mm, pass_mm = managers(tags)
    result = run_oracle_refolding_and_pass_criteria(
        atom_array=g, example_id=example_id, output_dir=work, oracle_mm=oracle_mm, refold_mm=refold_mm,
        pass_mm=pass_mm, metrics_tags=tags, prior_metrics=prior)
    return result, manifests


def featurised_difference(ours: Path, theirs: dict, tmp: Path) -> list[str]:
    """Both jobs through AlphaFold 3's own featuriser (the jax environment): the fields that differ."""
    script = r'''
import sys, json, datetime, dataclasses
import numpy as np
sys.path.insert(0, sys.argv[3])
import af3_runtime
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation

def batch(fi):
    fi = dataclasses.replace(fi, rng_seeds=(1,))
    ccd = chemical_components.Ccd(user_ccd=fi.user_ccd)
    return featurisation.featurise_input(fold_input=fi, buckets=None, ccd=ccd, verbose=False,
        ref_max_modified_date=datetime.date(2021, 9, 30), conformer_max_iterations=None,
        resolve_msa_overlaps=True, fix_standalone_glycans=True)[0]

ours = folding_input.Input.from_json(open(sys.argv[1]).read())
e = json.load(open(sys.argv[2]))
theirs = af3_runtime.build_fold_input_from_cif(cif_string=e["cif"], cif_name=e["name"],
    paired_msa=e.get("paired_msa"), unpaired_msa=e.get("unpaired_msa"), template=e.get("template"),
    user_ccd=e.get("user_ccd"), rng_seeds=[1])
a, b = batch(ours), batch(theirs)
bad = []
for k in sorted(set(a) | set(b)):
    if k not in a or k not in b:
        bad.append(f"{k}: only {'ours' if k in a else 'upstream'}"); continue
    x, y = a[k], b[k]
    if not isinstance(x, np.ndarray) or x.dtype == object:
        continue
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        bad.append(f"{k}: shape {x.shape} against {y.shape}")
    elif x.dtype.kind in "fc":
        # A template's coordinates are written at three decimals on both sides, from a design
        # each side put in its input's frame with its own superposition (agreeing to ~1e-6): a
        # value on a rounding boundary may land one step apart, and nothing finer is meaningful.
        atol = 1.5e-3 if k == "template_atom_positions" else 1e-5
        if not np.allclose(x, y, atol=atol, rtol=0, equal_nan=True):
            bad.append(f"{k}: max |d| {float(np.nanmax(np.abs(x - y))):.3g}")
    elif not np.array_equal(x, y):
        bad.append(f"{k}: {int((x != y).sum())} of {x.size} differ")
print(json.dumps({"fields": len(a), "bad": bad}))
'''
    entry = tmp / "entry.json"
    entry.write_text(json.dumps(theirs))
    r = subprocess.run([JAX, "-c", script, str(ours), str(entry), str(UPSTREAM / "src" / "rfproteina" / "af3")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return [f"featurising failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out["bad"]


def same(key: str, a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= (1e-4 if key.endswith(".rmsd") else 1e-9)


def main() -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    cm = checker()
    cases = [a.split(":", 2) for a in sys.argv[1:] if not a.startswith("--")]
    bad = 0
    for jpath, dpath, stem in cases:
        jpath, dpath, stem = Path(jpath), Path(dpath), Path(stem)
        job = json.loads(jpath.read_text())
        got = cm.page(jpath, dpath)
        g = designed_array(cm, job, jpath.resolve().parent, dpath, got["segmentLengths"])
        ours = json.loads(Path(f"{stem}.refold.json").read_text())
        prior = {k: (float("nan") if v == "NaN" else v)
                 for k, v in json.loads(Path(f"{stem}.metrics.json").read_text())["metrics"].items()}
        tags = got["metricsTags"]
        problems = []
        with tempfile.TemporaryDirectory() as tmp:
            result, manifests = upstream_scores(g, stem.name, Path(f"{stem}_refold"), tags, prior, Path(tmp))
            for oracle_name, entry in manifests.items():
                for line in featurised_difference(Path(f"{stem}_refold") / f"{stem.name}_{oracle_name}_job.json",
                                                  entry, Path(tmp)):
                    problems.append(f"{oracle_name} input: {line}")
        up = {k: v for k, v in result.items()
              if not k.endswith(("refolded_structure_path", "confidence_json_path")) and k != "example_id"}
        coi = {k: up.pop(k) for k in list(up) if k.endswith(".chain_of_interest")}
        here = {k: (float("nan") if v == "NaN" else v) for k, v in ours["metrics"].items()}
        for k in sorted(set(up) | set(here)):
            if k not in up or k not in here:
                problems.append(f"{k}: {'missing here' if k in up else 'not upstream'}")
            elif not same(k, up[k], here[k]):
                problems.append(f"{k}: upstream {up[k]!r}, here {here[k]!r}")
        for k, v in coi.items():
            if v != ours["chainOfInterest"]:
                problems.append(f"{k}: upstream {v!r}, here {ours['chainOfInterest']!r}")
        bad += bool(problems)
        print(f"  {stem.name:24s} {len(manifests)} oracle(s), {len(up)} values  {'ok' if not problems else 'DIFFERS'}")
        for p in problems[:10]:
            print(f"      {p}")
    print(f"{len(cases)} designs, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

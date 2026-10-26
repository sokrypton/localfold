#!/usr/bin/env python3
"""Hold the RFD4 refolding oracle (cuda/rfd4p/src/oracle.h) to upstream's, on the same designs.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-oracle.py [--oracle=af2] [<job.json>:<dump dir>:<design stem> ...]

`localfold-rfd4p --refold[=af2] --dump-sample=<dir> --out=<stem>.cif` refolds each design with our
own AlphaFold 3 (or 2) and writes, under `<stem>_refold/`, the job it built and the fold's files.
For each design this rebuilds upstream's designed array from the dump (as
tools/check-rfd4p-metrics.py does) and holds the two halves of the oracle apart:

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

With `--oracle=af2` the same two halves for `refolding_oracles/af2.yaml`:

- THE INPUT. Upstream's `AF2Oracle._build_manifest` payload through its own worker's
  `build_raw_features_from_payload` (a ColabFold environment, ~/.venv-colabfold), against our job
  through our AF2 featuriser (cuda/featurise/af2-featurise, as `localfold-af2 --job` featurises):
  the sequence, numbering, chain ids and the assembly's chain order, the template's residues,
  atoms and coordinates, and the alignment's query row, unmasked (`pct_seq_mask` 0).
- THE SCORES. Our AF2's raw confidences, padded to upstream's bucket as its model's are, through
  upstream's own `_chain_summary` - with DeepMind's `predicted_tm_score`
  (tools/oracle/alphafold2_confidence.py), which upstream's image runs - against the summary the
  port wrote; then upstream's managers over that summary and our structure written as DeepMind's
  `protein.to_mmcif` writes it.

The fold in between is our AF3 or AF2 port, which tools/check-batch-fields.js and cuda/af3's and
cuda/af2's gates hold to AlphaFold on their own.
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
COLABFOLD = Path.home() / ".venv-colabfold" / "bin" / "python"
# The AF2 bundles our fold reads (shared/bundles/manifests/index.js `directory`).
AF2_BUNDLES = {"model_1_ptm": REPO / "model", "model_1_multimer_v3": REPO / "model-multimer"}
PAE_HEAD = "__rfproteina_pae_head__"   # af2_runtime's `_PAE_HEAD_KEY`


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


def managers(tags: list[str], oracle: str = "af3"):
    import yaml
    from rfproteina.metrics.metric import MetricManager

    def load(rel):
        return yaml.safe_load((UPSTREAM / "configs/metrics" / rel).read_text())
    oracles = load(f"refolding_oracles/{oracle}.yaml")
    # Every entry the config runs (the off-target and crop transforms are commented out upstream).
    keys = ("oracle", "oracle_ppi", "oracle_dna", "oracle_apo") if oracle == "af3" else ("oracle", "oracle_ppi")
    oracle_mm = MetricManager.instantiate_from_hydra({k: oracles[k] for k in keys})
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
            # Our AF3 ran on `<example>[_<save_id>]_<oracle_name>` (upstream's `output_stem`);
            # upstream names the job `<example>[_<save_id>]`.
            name = entry["name"]
            ours = folds / f"{name}_{self.oracle_name}"
            out = Path(entry["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
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


def chain_summary():
    """Upstream's AF2 worker's `_chain_summary`, from its own source (the module that holds it
    imports AlphaFold 2), scoring with DeepMind's `predicted_tm_score` as upstream's image does."""
    from collections.abc import Mapping
    from typing import Any

    import numpy as _np
    spec = importlib.util.spec_from_file_location("alphafold2_confidence",
                                                  REPO / "tools" / "oracle" / "alphafold2_confidence.py")
    confidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confidence)
    tree = ast.parse((UPSTREAM / "src" / "rfproteina" / "af2" / "af2_runtime.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_chain_summary")
    namespace = {"np": _np, "Mapping": Mapping, "Any": Any, "confidence": confidence, "_PAE_HEAD_KEY": PAE_HEAD}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "af2_runtime.py", "exec"), namespace)
    return namespace["_chain_summary"]


AF2_SIDE = r'''
import sys, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import af2_runtime as rt
from alphafold.common import protein, residue_constants as rc

payload, ours_dir, ours_job, pdb, cif, name = sys.argv[2:8]
raw = rt.build_raw_features_from_payload(json.load(open(payload)))
meta = raw.pop(rt._META_KEY)
L, multimer = meta["num_res"], meta["multimer_mode"]
idx = {}
for line in open(ours_dir + "/model.idx"):
    fields = line.split()
    if fields[0] in ("i", "t"):
        idx[fields[1]] = (fields[0], int(fields[2]), int(fields[3]))
words = np.fromfile(ours_dir + "/model.bin", dtype=np.uint32)
def ours(key):
    kind, at, n = idx[key]
    return words[at:at + n].view(np.int32 if kind == "i" else np.float32)
bad = []
def same(what, a, b, atol=0):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        bad.append(f"{what}: shape {a.shape} against {b.shape}")
    elif (not np.allclose(a, b, atol=atol, rtol=0)) if atol else (not np.array_equal(a, b)):
        bad.append(f"{what}: {int((np.abs(a - b) > atol).sum())} of {a.size} differ")
ids = [next(iter(e.values()))["id"] for e in json.load(open(ours_job))["sequences"]]
lengths = [len(next(iter(e.values()))["sequence"]) for e in json.load(open(ours_job))["sequences"]]
if meta["feature_chain_ids"] != ids or meta["feature_chain_lengths"] != lengths:
    bad.append(f"chains: upstream {meta['feature_chain_ids']} {meta['feature_chain_lengths']}, here {ids} {lengths}")
aatype = raw["aatype"] if multimer else np.argmax(raw["aatype"], -1)
same("aatype", ours("aatype"), aatype[:L])
same("residue_index", ours("residue_index"), raw["residue_index"][:L])
if multimer:
    for k in ("asym_id", "entity_id", "sym_id"):
        same(k, ours(k) + 1, raw[k][:L])     # upstream counts from 1
# the alignment: the query row alone, never masked
k = 0
while f"f{k}/msa_feat" in idx:
    feat = ours(f"f{k}/msa_feat").reshape(-1, L, 49)
    if feat[:, :, 22].any():
        bad.append(f"f{k}/msa_feat: {int(feat[:, :, 22].sum())} masked positions where upstream masks none")
    same(f"f{k}/msa_feat query", np.argmax(feat[0, :, :23], -1), aatype[:L])
    k += 1
if multimer:
    ta, tm, tp = (raw["template_aatype"][0][:L], raw["template_all_atom_mask"][0][:L],
                  raw["template_all_atom_positions"][0][:L])
    has = bool(tm.any())
elif raw["template_aatype"].shape[0]:
    ta = np.asarray(rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)[np.argmax(raw["template_aatype"][0], -1)]
    tm, tp, has = raw["template_all_atom_masks"][0], raw["template_all_atom_positions"][0], True
else:
    has = False
if has != ("t/aatype" in idx):
    bad.append(f"template: {'upstream' if has else 'here'} only")
elif has:
    same("template aatype", ours("t/aatype"), ta)
    mask = ours("t/mask").reshape(L, 37)
    same("template mask", mask, tm)
    # three decimals on both sides, from a design each side put in its input's frame with its
    # own superposition: a value on a rounding boundary may land one step apart
    same("template positions", ours("t/positions").reshape(L, 37, 3) * mask[..., None], tp * tm[..., None], atol=1.5e-3)
# our structure as upstream's worker writes the model's (`protein.to_mmcif`)
prot = protein.from_pdb_string(open(pdb).read())
open(cif, "w").write(protein.to_mmcif(prot, file_id=name, model_type="Multimer" if multimer else "Monomer"))
print(json.dumps({"bad": bad, "meta": meta}))
'''


def upstream_af2_scores(g, example_id: str, folds: Path, tags: list[str], prior: dict, work: Path,
                        problems: list[str]) -> tuple[dict, dict]:
    """Upstream's AF2 oracle + refolding + pass criteria over our AF2's files, its input held to ours."""
    import numpy as np
    from rfproteina.metrics.oracles.af2_oracle import AF2Oracle
    from rfproteina.metrics.oracles.oracle_metric import OracleMetric
    from rfproteina.metrics.utils.metrics_utils import run_oracle_refolding_and_pass_criteria

    summarise = chain_summary()
    manifests: dict[str, dict] = {}

    def dispatch(self, manifest_path, daemon_rank, n_designs):
        for entry in json.loads(Path(manifest_path).read_text()):
            ours = folds / f"{example_id}_{self.oracle_name}"
            out = Path(entry["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            name = entry["name"]
            payload = work / f"{self.oracle_name}_payload.json"
            payload.write_text(json.dumps(entry))
            featurised = work / f"{self.oracle_name}_featurised"
            r = subprocess.run([REPO / "cuda" / "featurise" / "af2-featurise", featurised, f"--job={ours}_job.json",
                                f"--bundle={AF2_BUNDLES[self.model_name]}", "--recycles=3", "--msa-mask=0", "--template-pad=zero", "--seed=0"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"af2-featurise: {r.stderr.strip()}")
            r = subprocess.run([COLABFOLD, "-c", AF2_SIDE, UPSTREAM / "src" / "rfproteina" / "af2", payload, featurised,
                                f"{ours}_job.json", f"{ours}_model.pdb", out / f"{name}_model.cif", name],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"upstream's AF2 features: {r.stderr.strip().splitlines()[-1]}")
            side = json.loads(r.stdout.strip().splitlines()[-1])
            problems.extend(f"{self.oracle_name} input: {line}" for line in side["bad"])
            meta = side["meta"]
            # Our AF2's confidences as upstream's model hands them over: padded to the bucket.
            raw = json.loads(Path(f"{ours}_model_raw_confidences.json").read_text())
            n, bucket = meta["num_res"], meta["bucket"]
            plddt = np.zeros(bucket, np.float32)
            plddt[:n] = raw["plddt"]
            pae = np.zeros((bucket, bucket), np.float32)
            pae[:n, :n] = np.asarray(raw["pae"], np.float32).reshape(n, n)
            logits = np.zeros((bucket, bucket, 64), np.float32)
            logits[:n, :n] = np.fromfile(f"{ours}_model_pae_logits.f32", dtype=np.float32).reshape(n, n, 64)
            prediction = {"plddt": plddt, "predicted_aligned_error": pae, "ptm": np.float32(raw["ptm"]),
                          PAE_HEAD: {"logits": logits, "breaks": np.linspace(0, 31, 63, dtype=np.float32)}}
            if raw.get("iptm") is not None:
                prediction["iptm"] = np.float32(raw["iptm"])
            summary = summarise(prediction, {**meta, "model_name": self.model_name})
            (out / f"{name}_summary_confidences.json").write_text(json.dumps(summary))
            here = json.loads(Path(f"{ours}_summary_confidences.json").read_text())
            for k in sorted(set(summary) | set(here)):
                if summary.get(k) != here.get(k):
                    problems.append(f"{self.oracle_name} summary {k}: upstream {summary.get(k)!r}, here {here.get(k)!r}")
            manifests[self.oracle_name] = entry

    AF2Oracle._dispatch_fold = dispatch
    AF2Oracle.setup = OracleMetric.setup        # no daemon: the fold is ours
    oracle_mm, refold_mm, pass_mm = managers(tags, "af2")
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
    oracle = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--oracle=")), "af3")
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
            if oracle == "af2":
                result, manifests = upstream_af2_scores(g, stem.name, Path(f"{stem}_refold"), tags, prior, Path(tmp),
                                                        problems)
            else:
                result, manifests = upstream_scores(g, stem.name, Path(f"{stem}_refold"), tags, prior, Path(tmp))
            for oracle_name, entry in (manifests.items() if oracle == "af3" else ()):
                for line in featurised_difference(Path(f"{stem}_refold") / f"{entry['name']}_{oracle_name}_job.json",
                                                  entry, Path(tmp)):
                    problems.append(f"{oracle_name} input: {line}")
        up = {k: v for k, v in result.items()
              if not k.endswith(("refolded_structure_path", "confidence_json_path")) and k != "example_id"}
        up = {k: (v.item() if hasattr(v, "item") else v) for k, v in up.items()}
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

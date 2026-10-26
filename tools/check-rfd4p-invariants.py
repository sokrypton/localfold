#!/usr/bin/env python3
"""Hold the RFD4 featurisers' refusals to upstream's: which jobs a batch is refused for, and why.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-invariants.py [--no-native] [NAME ...]

Upstream finishes every batch with `ValidateBatchInvariants` (rfproteina/transforms/invariants.py)
and raises on a violation, and refuses a few jobs earlier still (a sequence probability that
contradicts C_SEQ, a coordinate group over an absolute target). For each job here - the shipped
presets and the gate's own jobs, which must all build, and jobs made to trip one check each - this
runs upstream's pipeline, the page's (`buildBatch`) and the native featuriser, and requires the
same verdict: built, or refused for the same checks (the `[check]` names upstream lists), or
refused with the same sentence.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "web" / "examples"
RCSB = REPO / "tools" / "fixtures" / "rcsb"
FEATURISER = REPO / "cuda" / "rfd4p" / "featurise" / "rfd4p-featurise"
WEIGHTS = REPO / "cuda" / "rfd4p" / "weights-f32"   # the featuriser reads the trunk's config off it


def sel(s: str) -> list[dict]:
    return [{"select": f"sel('{s}')"}]


def crafted() -> dict[str, dict]:
    ppi, brs, dna = str(EXAMPLES / "ppi_pdl1.cif.zst"), str(RCSB / "1brs.cif.zst"), str(RCSB / "7m5w.cif.zst")
    all_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
              "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    return {
        "nan-crd": {"input": ppi, "conditions": {"C_CRD": {"true": sel("B")}}},
        "expseg-hotspot": {"input": ppi, "contig": "a:10-10,B2-40", "conditions": {"C_HOT": {"true": sel("*/a/*/*")}}},
        "centroid-one-chain": {"input": brs, "contig": "A3-110, /0, D1-89",
                               "conditions": {"C_CENTROID": {"1,2,3": sel("A")}}},
        "centroid-part-chain": {"input": brs, "contig": "A3-110, /0, D1-89",
                                "conditions": {"C_CENTROID": {"1,2,3": sel("A/*/[3-50]")}}},
        "unindexed-no-crd": {"input": brs, "contig": "A3-110, /0, D1-89",
                             "conditions": {"C_IDX": {"false": [{"select": "sel('D/*/[5,6]')", "group": "g"}]}}},
        "seqprob-sum-above": {"input": brs, "contig": "A3-110", "conditions": {
            "C_SEQ_PROB_ALA": {"0.7": sel("A/*/[5-9]")}, "C_SEQ_PROB_LEU": {"0.7": sel("A/*/[5-9]")}}},
        "seqprob-sum-below": {"input": brs, "contig": "A3-110", "conditions": {
            f"C_SEQ_PROB_{t}": {"0.01": sel("A/*/[5-9]")} for t in all_aa}},
        "seqprob-wrong-type": {"input": dna, "contig": "A1-20", "conditions": {"C_SEQ_PROB_DA": {"0.5": sel("A")}}},
        "seqprob-against-seq": {"input": brs, "contig": "A3-110", "conditions": {
            "C_SEQ": {"true": sel("A")}, "C_SEQ_PROB_ALA": {"0.5": sel("A/*/[5-9]")}}},
        "contig-joins-a-gap": {"input": brs, "contig": "A3-20,A40-60"},
        "group-over-crd": {"input": brs, "contig": "A3-110", "conditions": {
            "C_CRD": {"true": sel("A")}, "C_CRDGRP": {"1": sel("A/*/[5-9]")}}},
        "sidechain-crd-adds-seq": {"input": ppi, "contig": "B2-40", "conditions": {"C_CRD": {"true": sel("B")}}},
        "two-segments-1brs": {"input": brs, "contig": "A3-110, /0, D1-89"},
    }


def corpus() -> dict[str, dict]:
    jobs = {}
    for path in sorted(p for p in (REPO / "cuda/rfd4p/featurise").glob("gate-*.json")
                       if not p.name.startswith(("gate-writer-", "gate-refusal"))):
        spec = json.loads(path.read_text())
        if "input" not in spec or isinstance(spec["input"], list) or "contig" not in spec and "conditions" not in spec:
            continue
        spec["input"] = str((path.parent / spec["input"]).resolve())
        jobs[path.stem] = {k: spec[k] for k in ("input", "contig", "conditions") if k in spec}
    jobs.update(crafted())
    return jobs


def upstream(spec: dict) -> str:
    import lightning as L
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.generate import get_contig_or_design_problem_dataset

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.json"
        path.write_text(json.dumps({"design": spec}))
        L.seed_everything(0, verbose=False)
        pipe = build_design_validation_pipeline(metrics_tags=[], path_length_search_cutoff=64)
        try:
            get_contig_or_design_problem_dataset(str(path), num_replicates=1, transform=pipe)[0]
            return "built"
        except Exception as e:  # noqa: BLE001 - a refusal is the answer
            return verdict(str(e))


def verdict(message: str) -> str:
    checks = re.findall(r"\[([a-z0-9_]+)\]", message)
    if checks:
        return "refused: " + ", ".join(sorted(set(checks)))
    line = message.strip().splitlines()[0] if message.strip() else ""
    line = re.sub(r"^Transforms failed at stage `\w+`: ", "", line)
    return "refused: " + line


def page(jobs: dict[str, dict]) -> dict[str, str]:
    script = """
import { readFileSync } from "node:fs";
import { buildBatch } from "./shared/rfd4p/pipeline/index.js";
const jobs = JSON.parse(readFileSync(0, "utf8"));
const out = {};
for (const [name, j] of Object.entries(jobs)) {
  try {
    // Bytes, as given: the reader tells mmCIF from BinaryCIF, plain from compressed.
    const inputCif = new Uint8Array(readFileSync(j.input));
    buildBatch({ contig: j.contig, ...(j.conditions ? { conditions: j.conditions } : {}) }, { seed: 0, inputCif });
    out[name] = null;
  } catch (e) { out[name] = String(e.message); }
}
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, capture_output=True, text=True,
                       input=json.dumps(jobs))
    if r.returncode != 0:
        sys.exit(r.stderr[-2000:])
    return {k: "built" if v is None else verdict(v) for k, v in json.loads(r.stdout).items()}


def native(spec: dict) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        job.write_text(json.dumps(spec))
        r = subprocess.run([FEATURISER, Path(tmp) / "out", f"--weights={WEIGHTS}",
                            f"--config={REPO}/model-rfd4p-int8/config.json", f"--spec={job}"],
                           capture_output=True, text=True)
        return "built" if r.returncode == 0 else verdict(r.stderr)


def main() -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_native = "--no-native" not in sys.argv and FEATURISER.exists() and WEIGHTS.exists()
    jobs = corpus()
    if args:
        jobs = {k: v for k, v in jobs.items() if k in args}
    ours = page(jobs)
    bad = refused = 0
    for name, spec in jobs.items():
        want = upstream(spec)
        got = {"page": ours[name]}
        if with_native:
            got["native"] = native(spec)
        wrong = {side: v for side, v in got.items() if v != want}
        refused += want != "built"
        bad += bool(wrong)
        print(f"  {name:28s} {want[:90]}  {'ok' if not wrong else 'DIFFERS'}")
        for side, v in wrong.items():
            print(f"      {side}: {v[:160]}")
    print(f"{len(jobs)} jobs ({refused} refused upstream), {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hold the page's RFD4 batch to upstream's own pipeline on the jobs that set every condition Studio edits.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-batch.py [NAME ...]

The trunk and trajectory oracles hold the batch to upstream only for the shipped presets, and no
preset sets metal coordination, RASA, a salt bridge, an H-bond, a pi-stack or pi-cation, a
hydrophobic contact, a chain centroid, index groups or the terminus flags - the conditions
AtomWorks Studio's editor offers. Each `cuda/rfd4p/featurise/gate-conditions*.json` sets some of
them on raw 1PRW (calmodulin, four calcium ions with their `metalc` bonds); for each this builds
upstream's batch with `tools/oracle/dump_batch_cpu.py` (the pipeline `rfd4-generate` runs, no
model) at fixed segment lengths, and `tools/oracle/check-batch.mjs` requires the page's
`buildBatch` to match it - every integer exactly, every float to 1e-4. gate.py's featurise arm
holds the native featuriser to the page on the same jobs, byte for byte.

🔴 THE LENGTHS ARE GIVEN, NOT INFERRED. check-batch.mjs otherwise reads them off the batch as runs
of unresolved tokens, and an unindexed motif (gate-conditions.json's two index groups) moves its
own tokens, so the runs it finds are not the segments that were drawn.

🔴 AND THE TERMINUS FLAGS DO NOT REACH THIS CHECKPOINT'S BATCH: upstream lists `C_NTR`/`C_CTR` among
its token annotations and its batch carries neither (nor does the page's), so moving them changes
nothing on either side. gate-conditions-termini.json is the case that would show it if a pipeline
started to.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FEATURISE = REPO / "cuda" / "rfd4p" / "featurise"
UPSTREAM = Path.home() / "RFD4-Proteina"
# job name -> the realized expandable-segment lengths, in the order the contig writes them
JOBS = {
    "gate-conditions": "7,10,6",
    "gate-conditions-termini": "7",
    "gate-conditions-interactions": "10",
}


def main() -> int:
    names = sys.argv[1:] or list(JOBS)
    bad = 0
    with tempfile.TemporaryDirectory(prefix="rfd4p-batch-") as tmp:
        tmp = Path(tmp)
        for name in names:
            job = json.loads((FEATURISE / f"{name}.json").read_text())
            job["input"] = str((FEATURISE / job["input"]).resolve())
            job.pop("nsteps", None)   # a sampler field; upstream's spec has none
            case = name.replace("-", "_")
            spec = tmp / f"{case}.spec.json"
            spec.write_text(json.dumps({case: job}))
            r = subprocess.run([sys.executable, REPO / "tools" / "oracle" / "dump_batch_cpu.py",
                                f"--case={case}", f"--inputs={spec}", f"--seg-lengths={JOBS[name]}",
                                f"--out={tmp}"], cwd=UPSTREAM, capture_output=True, text=True,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            if r.returncode != 0:
                print(f"  {name:32s} upstream did not build it:\n{r.stderr[-1500:]}")
                bad += 1
                continue
            r = subprocess.run(["node", REPO / "tools" / "oracle" / "check-batch.mjs",
                                f"--cpu-dir={tmp}", case], cwd=REPO, capture_output=True, text=True)
            line = next((ln for ln in r.stdout.splitlines() if "[cpu-dump]" in ln), "")
            issues = [ln.strip() for ln in r.stdout.splitlines() if ln.startswith("   - ")]
            ok = r.returncode == 0 and line.startswith("PASS")
            m = re.search(r"n=(\d+)", line)
            print(f"  {name:32s} {m.group(1) if m else '?'} tokens  {'ok' if ok else 'DIFFERS'}")
            for issue in issues:
                print(f"      {issue}")
            if not ok:
                bad += 1
                if not line:
                    print(r.stdout[-1000:], r.stderr[-1000:])
    print(f"{len(names)} jobs against upstream's batch, {bad} differing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

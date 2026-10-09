"""The Metal port's ESMFold2 gate: cuda/ef2/gate.py's fold arm - its cases, tolerances and scoring - with its fold
command and baseline pointed at metal/ef2 (metal/ef2/fold runs metal/ef2/localfold-ef2 --fast; the weights are the
checkout's bundles). The oracle arm needs biohub's forward and is not run here. The A100's baseline is printed beside.

    python3 metal/check/gate-ef2.py [--write] [--only=6mrr,...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "cuda", "ef2"))
import gate  # noqa: E402

a100 = json.load(open(gate.BASELINE)) if os.path.exists(gate.BASELINE) else {}
gate.HERE = os.path.join(REPO, "metal", "ef2")
gate.BASELINE = os.path.join(gate.HERE, "gate-baseline.json")
gate.TMP = os.environ.get("GATE_TMP", "/tmp/metal-ef2-gate")
gate.oracle_cases = lambda: []
cases = gate.fold_cases
WEIGHTS = os.environ.get("LOCALFOLD_WEIGHTS", os.path.expanduser("~/.cache/localfold"))   # (the checkout's esmfold2 export has no confidence head)
gate.fold_cases = lambda: [(c[0], c[1] + ["--fast", f"--weights-dir={WEIGHTS}"], *c[2:]) for c in cases()]
print("A100 baseline: " + ", ".join(f"{k} {v.get('rmsd')} / {v.get('plddt')}" for k, v in sorted(a100.items())))
gate.main()

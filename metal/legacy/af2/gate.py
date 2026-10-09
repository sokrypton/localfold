"""The Metal port's AF2 gate: cuda/af2/gate.py itself - its cases, tolerances and scoring - with its fold command and
its baseline pointed at metal/af2 (metal/af2/fold runs metal/af2/localfold-af2; metal/af2/gate-baseline.json is this
Mac's). The A100's baseline (cuda/af2/gate-baseline.json) is printed beside it for reference.

    python3 metal/af2/gate.py [--write] [--only=6mrr,...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "cuda", "af2"))
import gate  # noqa: E402

a100 = json.load(open(gate.BASELINE)) if os.path.exists(gate.BASELINE) else {}
gate.HERE = os.path.join(REPO, "metal", "legacy", "af2")
gate.BASELINE = os.path.join(gate.HERE, "gate-baseline.json")
gate.TMP = os.environ.get("GATE_TMP", "/tmp/metal-af2-gate")
print("A100 baseline: " + ", ".join(f"{k} {v.get('rmsd')} / {v.get('plddt')}" for k, v in sorted(a100.items())))
gate.main()

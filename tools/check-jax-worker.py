"""The JAX backend's worker, folding for real on this machine's GPU.

    npm run test:jax

🔴 `test:colab` NEVER REACHES `Worker.fold`. Its worker is a stub, because that
gate is about the feed - so a name clash in the worker's template bookkeeping
crashed every templated JAX fold and was caught only by a TPU on Colab. This
drives tools/jax_worker.py itself, over its own stdin/stdout protocol, against
af3-any-model installed here exactly as the notebook installs it on Colab:

    uv venv --python 3.13 ~/.venv-lfjax
    VIRTUAL_ENV=~/.venv-lfjax uv pip install pip "jax[cuda12]==0.11.1" dm-tree
    mkdir ~/lfjax && cd ~/lfjax && PATH=~/.venv-lfjax/bin:$PATH \\
      AF3_NB_OVERRIDES='{"model": "af2_multimer"}' python <ColabFold2's install cell>

(the cell is `nb_install.py`, the first cell of the notebook the Colab page
runs). `--python` and `--jax-dir` point elsewhere; with neither present this
FAILS and says so, rather than skipping - a gate that passes on a box without
the thing it tests is not one.

WHAT IT CHECKS, each against 1BRS (barnase-barstar) and its own crystal:
  - the multimer's template, BOTH arms: no template is > 10 A, both chains
    templated is < 1.5 A - one arm alone cannot tell a template from luck;
  - that the result carries the templates it used, one per fold chain with
    the structure chain, and names them on the status line - what the page's
    archive saves and a dropped archive restores;
  - an AF3-lineage template fold (openbind0), < 1.5 A;
  - a ligand job after those, which is the worker's `execv` restart path (a
    new CCD code cannot be added to a running process), and
  - rosettafold3 refusing Flow, which must be a refusal and not a fold.
About five minutes on an A100, most of it compiling.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def alpha_carbons(text, chains):
    points, sequence = [], {}
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A" \
                and line[21] in chains:
            points.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            sequence.setdefault(line[21], []).append(THREE.get(line[17:20], "X"))
    return np.array(points), {chain: "".join(s) for chain, s in sequence.items()}


def rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=os.path.expanduser("~/.venv-lfjax/bin/python"))
    parser.add_argument("--jax-dir", default=os.path.expanduser("~/lfjax"))
    args = parser.parse_args()
    if not (os.path.exists(args.python) and os.path.isdir(args.jax_dir)):
        sys.exit(f"no JAX install at {args.python} / {args.jax_dir} - see this file's header")

    crystal_text = open(os.path.join(ROOT, "tools/fixtures/1brs-crystal.pdb")).read()
    crystal, sequences = alpha_carbons(crystal_text, "AD")
    barnase, barstar = sequences["A"], sequences["D"]
    job = json.dumps({"name": "gate", "modelSeeds": [1], "dialect": "alphafold3", "version": 2,
                      "sequences": [{"protein": {"id": "A", "sequence": barnase}},
                                    {"protein": {"id": "B", "sequence": barstar}}]})
    with_ligand = json.dumps({**json.loads(job), "sequences": json.loads(job)["sequences"]
                              + [{"ligand": {"id": "C", "ccdCodes": ["GOL"]}}]})

    def upload(chain):
        return {"kind": "upload", "text": crystal_text, "filename": "1brs.pdb", "source": chain}

    def rows(template_a=None, template_d=None, ligand=False):
        out = [{"type": "protein", "value": barnase, "copies": 1},
               {"type": "protein", "value": barstar, "copies": 1}]
        if template_a:
            out[0]["template"] = template_a
        if template_d:
            out[1]["template"] = template_d
        if ligand:
            out.append({"type": "ligand", "value": "GOL", "copies": 1})
        return out

    multimer = {"model-family": "multimer", "msa-mode": "none", "recycles": "3", "af2Model": "1"}
    openbind = {"model-family": "openbind0", "msa-mode": "none", "af3-mode": "diffusion"}
    cases = [
        # name, job, entities, controls, check(result, rmsd) -> problem or None
        ("multimer, no template", job, rows(), multimer,
         lambda r, d: None if d > 10 else f"{d:.2f} A without a template - the control must be far"),
        ("multimer, A and D templated", job, rows(upload("A"), upload("D")), multimer,
         lambda r, d: (f"{d:.2f} A with its own crystal" if d > 1.5 else
                       f"templates {r.get('templates')!r:.120}" if sorted(
                           (t["chain"], t["chainId"]) for t in r.get("templates", []))
                       != [(0, "A"), (1, "D")] else
                       "the status line names no template" if r["status"].count("template") != 2
                       else None)),
        ("openbind0, A and D templated", job, rows(upload("A"), upload("D")), openbind,
         lambda r, d: None if d < 1.5 else f"{d:.2f} A with its own crystal"),
        ("openbind0 + GOL (restart)", with_ligand, rows(ligand=True), openbind,
         lambda r, d: None if "GOL" in r["pdb"] else "no GOL in the structure"),
        ("rosettafold3 with Flow", job, rows(), {**openbind, "model-family": "rosettafold3",
                                                 "af3-mode": "flow"}, None),
    ]

    worker = subprocess.Popen([args.python, os.path.join(ROOT, "tools/jax_worker.py")],
                              cwd=args.jax_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True)
    failures = 0
    try:
        for name, spec, entities, controls, check in cases:
            worker.stdin.write(json.dumps({"job": spec, "entities": entities,
                                           "family": controls["model-family"],
                                           "controls": controls}) + "\n")
            worker.stdin.flush()
            started = time.time()
            while True:
                line = worker.stdout.readline()
                if not line:
                    sys.exit("FAIL  the worker exited")
                event = json.loads(line)
                if event["kind"] == "result":
                    break
            result = event["payload"]
            seconds = time.time() - started
            if check is None:
                # A refusal is the pass: it must say why, and fold nothing.
                problem = None if "error" in result and "flow" in result["error"].lower() \
                    else f"folded or failed otherwise: {str(result)[:160]}"
                shown = result.get("error", "")[:70]
            elif "error" in result:
                problem, shown = result["error"][:160], ""
            else:
                predicted, _ = alpha_carbons(result["pdb"], "AB")
                distance = rmsd(predicted, crystal)
                problem = check(result, distance)
                shown = f"{distance:6.2f} A  pLDDT {result['confidence']['meanPlddt']:.1f}"
            failures += problem is not None
            print(f"{'ok  ' if problem is None else 'FAIL'}  {name:30s} {seconds:6.1f}s  {shown}"
                  + ("" if problem is None else f"\n      {problem}"), flush=True)
    finally:
        worker.stdin.close()
        worker.wait(timeout=60)
    if failures:
        sys.exit(f"{failures} of {len(cases)} JAX worker checks failed")
    print("the JAX worker folds, templates and restarts")


if __name__ == "__main__":
    main()

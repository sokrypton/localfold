"""The JAX backend's worker, folding for real on this machine's GPU.

    npm run test:jax

🔴 `test:colab` NEVER REACHES `Worker.fold`. Its worker is a stub, because that
gate is about the feed - so a name clash in the worker's template bookkeeping
crashed every templated JAX fold and was caught only by a TPU on Colab. This
drives jax/worker.py itself, over its own stdin/stdout protocol, against
af3-any-model installed here exactly as the notebook installs it on Colab:

    uv venv --python 3.13 ~/.venv-lfjax
    VIRTUAL_ENV=~/.venv-lfjax uv pip install pip "jax[cuda12]==0.11.1" dm-tree requests
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
  - the page's recycle tolerance stopping a settled fold early, as WebGPU's does;
  - an AF3-lineage template fold (openbind0), < 1.5 A;
  - a ligand job after those, which is the worker's `execv` restart path (a
    new CCD code cannot be added to a running process), its GOL bonds held
    to 0.15 A rms, and
  - rosettafold3 refusing Flow, which must be a refusal and not a fold, and
  - ESMFold2 600M folding 6MRR (< 3 A), which it cannot without its language
    model - the worker once set the flag and not the argument that decides it;
  - and a SECOND and third ESMFold2 fold: barstar (another size, < 3 A) and 6MRR reversed (the
    same size), the latter equal to the same job in a fresh worker - the live sampler once baked
    the first fold's rotary tables into every later one.
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


def glycerol_problem(pdb):
    """GOL's five bonds, as `test:ligand` asserts them on WebGPU: present, and
    within 0.15 A rms of the dictionary - arriving is not the same as holding."""
    atoms = {line[12:16].strip(): np.array([float(line[30:38]), float(line[38:46]),
                                            float(line[46:54])])
             for line in pdb.splitlines() if line.startswith("HETATM") and line[17:20] == "GOL"}
    bonds = [("C1", "O1", 1.43), ("C1", "C2", 1.52), ("C2", "O2", 1.43), ("C2", "C3", 1.52),
             ("C3", "O3", 1.43)]
    if any(a not in atoms or b not in atoms for a, b, _ in bonds):
        return f"GOL atoms {sorted(atoms)}"
    error = np.sqrt(np.mean([(np.linalg.norm(atoms[a] - atoms[b]) - ideal) ** 2
                             for a, b, ideal in bonds]))
    return None if error < 0.15 else f"GOL bond rms {error:.3f} A"


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

    # 6MRR for ESMFold2: a 68-residue designed protein, one chain.
    mrr_text = open(os.path.join(ROOT, "tools/fixtures/6mrr-crystal.pdb")).read()
    mrr, mrr_sequence = alpha_carbons(mrr_text, "A")
    mrr_job = json.dumps({"name": "gate", "modelSeeds": [1], "dialect": "alphafold3", "version": 2,
                          "sequences": [{"protein": {"id": "A", "sequence": mrr_sequence["A"]}}]})
    mrr_rows = [{"type": "protein", "value": mrr_sequence["A"], "copies": 1}]

    ef2 = {"model-family": "ef2-fast-600m", "plm-mode": "esmc-600m", "msa-mode": "none", "af3-mode": "diffusion"}
    def single_chain(sequence):
        return (json.dumps({"name": "gate", "modelSeeds": [1], "dialect": "alphafold3", "version": 2,
                            "sequences": [{"protein": {"id": "A", "sequence": sequence}}]}),
                [{"type": "protein", "value": sequence, "copies": 1}])
    barstar_job, barstar_rows = single_chain(barstar)
    barstar_ca, _ = alpha_carbons(crystal_text, "D")
    reversed_job, reversed_rows = single_chain(mrr_sequence["A"][::-1])
    multimer = {"model-family": "multimer", "msa-mode": "none", "recycles": "3", "af2Model": "1"}
    openbind = {"model-family": "openbind0", "msa-mode": "none", "af3-mode": "diffusion"}
    cases = [
        # name, job, entities, controls, check(result, rmsd) -> problem or None;
        # scored against 1BRS unless the case names another crystal (a 6th item)
        ("multimer, no template", job, rows(), multimer,
         lambda r, d: None if d > 10 else f"{d:.2f} A without a template - the control must be far"),
        ("multimer, A and D templated", job, rows(upload("A"), upload("D")), multimer,
         lambda r, d: (f"{d:.2f} A with its own crystal" if d > 1.5 else
                       f"templates {r.get('templates')!r:.120}" if sorted(
                           (t["chain"], t["chainId"]) for t in r.get("templates", []))
                       != [(0, "A"), (1, "D")] else
                       "the status line names no template" if r["status"].count("template") != 2
                       else None)),
        # 🔴 THE PAGE'S EARLY STOP, WHICH THIS BACKEND ONCE IGNORED: the settled
        # templated complex stops after 2 of 4 passes at 0.5, as WebGPU does.
        ("multimer, tolerance 0.5", job, rows(upload("A"), upload("D")),
         {**multimer, "tolerance": "0.5"},
         lambda r, d: None if "converged at" in r["status"] and d < 1.5
         else f"no early stop: {r['status'][-80:]}"),
        ("openbind0, A and D templated", job, rows(upload("A"), upload("D")), openbind,
         lambda r, d: None if d < 1.5 else f"{d:.2f} A with its own crystal"),
        ("openbind0 + GOL (restart)", with_ligand, rows(ligand=True), openbind,
         lambda r, d: glycerol_problem(r["pdb"])),
        ("rosettafold3 with Flow", job, rows(), {**openbind, "model-family": "rosettafold3",
                                                 "af3-mode": "flow"}, None),
        # 🔴 ESMFold2 IS ITS LANGUAGE MODEL. The worker set the flag and not
        # process_fold_input's `use_esm` argument, so the tower never ran: 6MRR
        # at 15.81 A on the 600M, where the CLI and the WebGPU port give ~1.5.
        ("esmfold2 600M, 6MRR", mrr_job, mrr_rows, ef2,
         lambda r, d: None if d < 3 else f"{d:.2f} A - is the language model reaching it?", mrr),
        # 🔴 AND A SECOND ESMFold2 FOLD, BECAUSE THE FIRST ONE'S ROTARY TABLES WERE BAKED INTO THE
        # LIVE SAMPLER'S CACHED STEP (see patch_staged in jax/worker.py): any later fold of
        # another size died with "mul got incompatible shapes", and one of the SAME size folded
        # silently with the first molecule's conformer - the same-size arm is checked after the loop
        ("esmfold2 600M, barstar (another size)", barstar_job, barstar_rows, ef2,
         lambda r, d: None if d < 3 else f"{d:.2f} A", barstar_ca),
        ("esmfold2 600M, 6MRR reversed (same size)", reversed_job, reversed_rows, ef2,
         lambda r, d: None, mrr),
    ]

    # 🔴 ITS OWN WEIGHT CACHE, as fresh as a Colab VM's. af3-any-model takes any
    # blob already in ~/.cache/alphafold3/weights, and this box had September's
    # there: boltz2, protenix2, rosettafold3, opendde and esmfold2 all died on
    # parameters the published checkpoints have since renamed, where a Colab
    # runtime - which starts empty - folded all five.
    env = {**os.environ, "AF3_WEIGHTS_DIR": os.path.join(args.jax_dir, "weights")}
    def start():
        return subprocess.Popen([args.python, os.path.join(ROOT, "jax/worker.py")],
                                cwd=args.jax_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, env=env)

    def run(proc, spec, entities, controls):
        proc.stdin.write(json.dumps({"job": spec, "entities": entities,
                                     "family": controls["model-family"], "controls": controls}) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                sys.exit("FAIL  the worker exited")
            event = json.loads(line)
            if event["kind"] == "result":
                return event["payload"]
    worker = start()
    folded = {}
    failures = 0
    try:
        for name, spec, entities, controls, check, *truth in cases:
            started = time.time()
            result = run(worker, spec, entities, controls)
            folded[name] = result
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
                distance = rmsd(predicted, truth[0] if truth else crystal) \
                    if len(predicted) == len(truth[0] if truth else crystal) else float("nan")
                problem = check(result, distance)
                shown = f"{distance:6.2f} A  pLDDT {result['confidence']['meanPlddt']:.1f}"
            failures += problem is not None
            print(f"{'ok  ' if problem is None else 'FAIL'}  {name:30s} {seconds:6.1f}s  {shown}"
                  + ("" if problem is None else f"\n      {problem}"), flush=True)
    finally:
        worker.stdin.close()
        worker.wait(timeout=60)
    # the same-size arm: 6MRR reversed, folded AFTER 6MRR above, against the same job in a FRESH worker -
    # one seed, one input, so the coordinates must agree (they were 1.0 A apart with the cached step
    # still holding 6MRR's rotary tables, at a plausible pLDDT)
    name = "esmfold2 600M, 6MRR reversed (same size)"
    after = folded.get(name, {})
    fresh_worker = start()
    try:
        alone = run(fresh_worker, reversed_job, reversed_rows, ef2)
    finally:
        fresh_worker.stdin.close()
        fresh_worker.wait(timeout=60)
    if "pdb" not in after or "pdb" not in alone:
        problem = f"{after.get('error', '')}{alone.get('error', '')}"[:160] or "no structure"
    else:
        moved = float(np.abs(alpha_carbons(after["pdb"], "A")[0] - alpha_carbons(alone["pdb"], "A")[0]).max())
        problem = None if moved < 1e-3 else f"{moved:.3f} A from the same job in a fresh worker"
    failures += problem is not None
    print(f"{'ok  ' if problem is None else 'FAIL'}  {'esmfold2: after another = alone':30s}"
          + ("" if problem is None else f"\n      {problem}"), flush=True)
    if failures:
        sys.exit(f"{failures} of {len(cases) + 1} JAX worker checks failed")
    print("the JAX worker folds, templates and restarts")


if __name__ == "__main__":
    main()

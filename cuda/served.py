"""Every gate case again, on ONE resident server - each port's gate.py's `served` arm.

🔴 A RESIDENT SERVER IS NOT A FRESH PROCESS (CLAUDE.md, "The traps that repeat"), and every case a gate folds runs in a
fresh one, so state one job leaves for the next - a cache keyed on an address, a kernel attribute a warm-up set, a
shared scratch slot - is invisible to it. Four bugs of that class reached the worker in a month, and the last
(1a3b313b: a warm-up then a templated fold, pTM 0.0000) was caught only because test:native happened to fold a
template after a warm-up. This arm starts the port's server as python/localfold/worker.py does (its flags, its
--warm), featurises every case with the port's featuriser binary, folds them in the gate's order and then the first
again, and holds each warm fold to the SAME case's cold fold from the gate's folds arm: the same atoms, within
TOL_RMS / TOL_MAX angstroms. A fold that carried state reads angstroms; one that did not reads ~0.

    served(port, server, cases, tmp) -> problems
      server  the server's argv before --serve (binary, "-", the worker's flags)
      cases   [(name, featurise argv with "{dir}" for the input directory, fold flags, the cold fold's PDB)]
"""
import math
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FEATURISE = os.path.join(HERE, "featurise")
# a warm fold against its cold one: the atoms' RMS and largest move (A). Measured 0.000-0.0xx where nothing is carried
# (a resident server's roomFor choices see a different free memory, which may pick another kernel)
TOL_RMS, TOL_MAX = 0.05, 0.5
# the flags only a featuriser reads (cuda/featurise/standalone.h, inputFlags; and standalone.cpp's BIOEMU2_INPUT)
INPUT_FLAGS = {"job", "sequence", "a3m", "paired-a3m", "search", "search-templates", "template", "template-search-chains",
               "kinds", "ligands", "smiles", "modify", "max-msa", "max-extra", "msa-mask", "template-pad",
               "no-span-chains", "ccd", "mode", "contacts", "propagate-from", "lag-ns", "propagation-steps"}
# the dictionary the standalone hands its featuriser when it is on disk (standalone.cpp, ccdFlag)
CCD = os.path.join(REPO, "ccd", "components.cif")


def weights(model):
    """A model's weight directories, as fetch-weights names them (python/localfold/worker.py, model_weights)."""
    out = subprocess.run([os.path.join(FEATURISE, "fetch-weights"), model], cwd=REPO, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"fetch-weights {model}: {(out.stdout + out.stderr).strip()[-300:]}")
    return [line.split(": ", 1)[1] for line in out.stdout.splitlines() if line.startswith(model + ": /")]


def split(flags):
    """A standalone command's flags as its featuriser's and its fold's, as standalone.cpp splits them: the featuriser is
    handed every flag (it ignores what it does not read) and the CCD, the fold all but the input flags and --model."""
    name = lambda f: f[2:].split("=", 1)[0] if f.startswith("--") else ""
    inputs = [f for f in flags if name(f) != "model"] + ([f"--ccd={CCD}"] if os.path.exists(CCD) else [])
    fold = [f for f in flags if name(f) not in INPUT_FLAGS and name(f) != "model"]
    return inputs, fold


def atoms(pdb):
    out = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            out.append((line[12:27], float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def compare(cold, warm):
    """The warm fold's atoms against the cold one's, in place (no superposition: one seed, one frame): (rms, max) or
    a problem."""
    a, b = atoms(cold), atoms(warm)
    if len(a) != len(b) or any(x[0] != y[0] for x, y in zip(a, b)):
        return f"{len(b)} atoms where the cold fold has {len(a)} (or in another order)"
    d = [math.dist(x[1:], y[1:]) for x, y in zip(a, b)]
    return math.sqrt(sum(v * v for v in d) / max(1, len(d))), max(d, default=0.0)


def served(port, server, cases, tmp):
    problems = []
    root = os.path.join(tmp, f"served-{port}")
    shutil.rmtree(root, ignore_errors=True)
    jobs = os.path.join(root, "jobs")
    os.makedirs(jobs)
    # every input featurised first, so the server folds back to back as it does a reader's jobs
    for name, featurise, _, cold in cases:
        if not os.path.exists(cold):
            problems.append(f"{name}: no cold fold at {cold} (run the folds arm first)")
            continue
        p = subprocess.run([a.replace("{dir}", os.path.join(root, name)) for a in featurise], cwd=REPO,
                           capture_output=True, text=True)
        if p.returncode:
            problems.append(f"{name}: featurising failed: {(p.stdout + p.stderr).strip()[-300:]}")
    if problems:
        return problems
    log = open(os.path.join(root, "server.log"), "w")
    proc = subprocess.Popen([*server, f"--serve={jobs}"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    try:
        while "serving" not in open(log.name).read():
            if proc.poll() is not None:
                return [f"the {port} server exited: {open(log.name).read()[-400:]}"]
            time.sleep(0.05)
        order = list(cases) + [cases[0]]                  # ...and the first again, after everything else
        for k, (name, _, fold, cold) in enumerate(order):
            out = os.path.join(root, f"{name}-{k}.pdb")
            job = os.path.join(jobs, f"{k:03d}")
            with open(job + ".tmp", "w") as f:
                f.write("\n".join([os.path.join(root, name), f"--out={out}", *fold]) + "\n")
            os.rename(job + ".tmp", job + ".job")
            while not os.path.exists(job + ".done"):
                if proc.poll() is not None:
                    said = open(job + ".log").read().strip() if os.path.exists(job + ".log") else ""
                    print(f"  served #{k + 1:<2d} {name:24s} the server died", flush=True)
                    return problems + [f"{name} (served #{k + 1}, after {', '.join(c[0] for c in order[:k]) or 'nothing'}):"
                                       f" the {port} server died: {(said or open(log.name).read())[-400:]}"]
                time.sleep(0.01)
            code = int(open(job + ".done").read())
            said = open(job + ".log").read() if os.path.exists(job + ".log") else ""
            if code or not os.path.exists(out):
                problems.append(f"{name} (served #{k + 1}): exit {code}: {said.strip()[-300:]}")
                print(f"  served #{k + 1:<2d} {name:24s} FAILED", flush=True)
                continue
            got = compare(cold, out)
            if isinstance(got, str):
                problems.append(f"{name} (served #{k + 1}): {got}")
                print(f"  served #{k + 1:<2d} {name:24s} {got}", flush=True)
                continue
            rms, top = got
            ok = rms <= TOL_RMS and top <= TOL_MAX
            print(f"  served #{k + 1:<2d} {name:24s} against its cold fold: rms {rms:.4f} A  max {top:.3f} A  "
                  + ("ok" if ok else "MOVED"), flush=True)
            if not ok:
                problems.append(f"{name} (served #{k + 1}, after {', '.join(c[0] for c in order[:k]) or 'nothing'}):"
                                f" rms {rms:.3f} A, max {top:.2f} A from its cold fold")
    finally:
        if proc.poll() is None:
            with open(os.path.join(jobs, "~quit.tmp"), "w") as f:
                f.write("quit\n")
            os.rename(os.path.join(jobs, "~quit.tmp"), os.path.join(jobs, "~quit.job"))
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
    return problems

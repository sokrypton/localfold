#!/usr/bin/env python3
"""Gate the standalone ports: `cuda/<port>/<port> --job=... --out=...` against the two-step path it replaces.

    python3 tools/check-standalone.py [--network]

🔴 ONE PROCESS MUST FOLD WHAT TWO DID. The standalone mode (cuda/featurise/standalone.h) fetches the weights,
featurises in its own process on a thread and folds; the resident server cuda/worker.py drives still takes the
featuriser binary's directory. Each case runs both ways - `cuda/featurise/<port>-featurise <dir> ...` then
`cuda/<port>/<port> <dir> ...`, and the standalone command - and holds the two PDBs to the SAME BYTES, so a flag the
standalone mode drops or misroutes (an input flag reaching the fold, af2's --recycles reaching only one of the
two) is a difference here rather than a quietly different fold. Beside them: refusals come back as the page's
sentence with a nonzero status (an input flag the port does not read is one of them, never dropped), `--frames=` streams what the page draws, a weights home is honoured
(--weights-dir), and with --network a searched alignment is kept as <out>.a3m.

Needs the GPU and the weights the worker uses (fetched on first use).
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = os.path.join(REPO, "cuda")
FIX = os.path.join(REPO, "tools", "fixtures")
JOBS = os.path.join(FIX, "af3-jobs")
S6 = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"


def run(cmd, env=None):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)


def two_step(port, featuriser_args, fold_args, out, work):
    """The worker's path: the featuriser binary into a directory, then the port over it."""
    d = os.path.join(work, "in")
    shutil.rmtree(d, ignore_errors=True)
    f = run([os.path.join(CUDA, "featurise", f"{port}-featurise"), d, *featuriser_args])
    if f.returncode != 0:
        raise SystemExit(f"{port}-featurise failed: {f.stdout}{f.stderr}")
    r = run([os.path.join(CUDA, port, port), d, *fold_args, f"--out={out}"])
    if r.returncode != 0:
        raise SystemExit(f"{port} over the directory failed: {r.stdout[-800:]}{r.stderr[-800:]}")


def standalone(port, args, out, env=None):
    r = run([os.path.join(CUDA, port, port), *args, f"--out={out}"], env)
    if r.returncode != 0:
        raise SystemExit(f"standalone {port} failed: {r.stdout[-800:]}{r.stderr[-800:]}")
    return r


def main():
    network = "--network" in sys.argv[1:]
    home = REPO
    af3 = lambda m: [f"--bundle={home}/af3am-{m}", f"--family={m}", "--fold", "--fast"]
    cases = [
        # name, port, input flags, model, featuriser extras, fold extras (both two-step)
        ("af3 kras (covalent ligand, job)", "af3", [f"--job={JOBS}/kras_g12c_sotorasib.json"], "af3",
         ["--no-weights", "--family=af3"], af3("af3")),
        ("boltz2 6mrr + GOL + SEP3", "af3", [f"--sequence={S6}", "--ligands=GOL", "--modify=SEP@3"], "boltz2",
         ["--no-weights", "--family=boltz2"], af3("boltz2")),
        ("chai1 streptavidin (SMILES)", "af3", [f"--job={JOBS}/streptavidin_biotin_smiles.json"], "chai1",
         ["--no-weights", "--family=chai1"], af3("chai1") + [f"--esm-bundle={home}/af3am-esm2"]),
        ("af3 barnase-barstar, barnase's crystal as a template, 2 samples", "af3",
         [f"--job={JOBS}/barnase_barstar.json", "--template=" + os.path.join(FIX, "1brs-crystal.pdb") + ":A@0",
          "--samples=2"], "af3", ["--no-weights", "--family=af3"], af3("af3") + ["--samples=2"]),
        ("af2 monomer model 3 (a delta), 1 recycle", "af2", [f"--sequence={S6}", "--recycles=1"], "model_3_ptm",
         [f"--bundle={home}/model", "--recycles=1"], [f"--bundle={home}/model", f"--delta={home}/model-mono-3-delta", "--fast", "--recycles=1"]),
        ("af2 multimer barnase-barstar (job)", "af2", [f"--job={JOBS}/barnase_barstar.json"], "model_1_multimer_v3",
         [f"--bundle={home}/model-multimer"], [f"--bundle={home}/model-multimer", "--fast"]),
        ("esmfold2 600M calmodulin (job, ions)", "esmfold2", [f"--job={JOBS}/calmodulin_4calcium.json"], "esmfold2-fast-600m",
         [f"--fold-bundle={home}/model-esmfold2-int5"], [f"--fold-bundle={home}/model-esmfold2-int5", f"--esmc-bundle={home}/model-esmc-600m-int3", "--fast",
          "--steps=64"]),      # (the per-atom floor the worker applies: four calcium ions are four atom tokens)
        ("esmfold2 300M 6mrr, seed 7", "esmfold2", [f"--sequence={S6}", "--seed=7"], "esmfold2-fast-300m",
         [f"--fold-bundle={home}/model-ef2-fast-300m-int5"], [f"--fold-bundle={home}/model-ef2-fast-300m-int5", f"--esmc-bundle={home}/model-esmc-300m-int3", "--fast", "--seed=7"]),
    ]
    failed = 0
    with tempfile.TemporaryDirectory(dir="/tmp") as work:
        for name, port, inputs, model, fextra, fold_extra in cases:
            # the standalone command first: it fetches whatever weights the two-step path then reads
            new, old = os.path.join(work, "standalone.pdb"), os.path.join(work, "two-step.pdb")
            fold_only = [a for a in inputs if a.split("=")[0] in ("--samples", "--seed")]
            standalone(port, [*inputs, f"--model={model}"], new)
            featuriser_inputs = [a for a in inputs if a not in fold_only or port == "af2"]
            two_step(port, [*featuriser_inputs, *fextra], fold_extra, old, work)
            same = open(new, "rb").read() == open(old, "rb").read()
            failed += not same
            print(f"{'ok  ' if same else 'FAIL'} {name}: {'identical' if same else 'the PDBs differ'}")

        # refusals: the page's sentence, a nonzero status
        for name, port, args, want in [
            ("a ligand on AF2", "af2", [f"--job={JOBS}/kras_g12c_sotorasib.json"], "AlphaFold 2 folds protein chains only"),
            ("an unknown model", "af3", [f"--sequence={S6}", "--model=nope"], "no model nope"),
            ("a template on ESMFold2", "esmfold2", [f"--sequence={S6}", "--template=x.pdb:A"], "esmfold2 takes no --template"),
            ("a SMILES ligand on AF2", "af2", [f"--sequence={S6}", "--smiles=OCC(O)CO"], "af2 takes no --smiles"),
            ("an alignment on fast ESMFold2", "esmfold2", [f"--sequence={S6}", "--a3m=x.a3m"], "reads no alignment"),
        ]:
            r = run([os.path.join(CUDA, port, port), *args, f"--out={work}/x.pdb"])
            said = r.stdout + r.stderr
            ok = r.returncode != 0 and "Error: " in said and (want is None or want in said)
            failed += not ok
            line = next((l for l in said.splitlines() if l.startswith("Error: ")), said.strip()[-200:])
            print(f"{'ok  ' if ok else 'FAIL'} refused: {name}: {line}")

        # --frames: the trunk's contacts and the sampler's frames, as the page draws them
        frames = os.path.join(work, "frames")
        os.makedirs(frames)
        standalone("af3", [f"--sequence={S6}", f"--frames={frames}", "--steps=20"], os.path.join(work, "f.pdb"))
        names = sorted(os.listdir(frames))
        ok = any(n.startswith("frame-") for n in names) and any(n.startswith("contacts-") for n in names)
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} --frames: {len(names)} files ({', '.join(names[:3])}, ...)")

        standalone("esmfold2", [f"--sequence={S6}", "--model=esmfold2-fast-300m"], f"{work}/standalone-300m.pdb")
        # --weights-dir names where the weights live (here: links to this checkout's)
        alt = os.path.join(work, "weights")
        os.makedirs(alt)
        for d in ("model-ef2-fast-300m-int5", "model-esmc-300m-int3"):
            os.symlink(os.path.join(REPO, d), os.path.join(alt, d))
        r = run([os.path.join(CUDA, "esmfold2", "esmfold2"), f"--sequence={S6}", "--model=esmfold2-fast-300m",
                 f"--weights-dir={alt}", f"--out={work}/h.pdb"])
        ok = r.returncode == 0 and open(f"{work}/h.pdb", "rb").read() == open(f"{work}/standalone-300m.pdb", "rb").read()
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} --weights-dir: {'the same fold from ' + alt if ok else (r.stdout + r.stderr)[-300:]}")
        r = run([os.path.join(CUDA, "esmfold2", "esmfold2"), f"--sequence={S6}", "--weights-dir=/nonexistent/localfold-weights",
                 f"--out={work}/h2.pdb"])
        ok = r.returncode != 0 and "Error: " in r.stdout + r.stderr
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} --weights-dir unwritable: refused, not folded from elsewhere")

        if network:
            out = os.path.join(work, "searched.pdb")
            standalone("af3", [f"--sequence={S6}", "--search"], out)
            a3m = os.path.join(work, "searched.a3m")
            ok = os.path.exists(a3m) and open(a3m).read().startswith(">")
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} --search: the alignment kept as {os.path.basename(a3m)}"
                  + (f" ({open(a3m).read().count(chr(62))} sequences)" if ok else ""))
    print("standalone: " + ("ok" if not failed else f"{failed} FAILED"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

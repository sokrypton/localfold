"""How does cuda/ef2 compare with upstream ESMFold2 (biohub's `esm` package) - time, memory and accuracy, same settings.

    <venv with esm>/bin/python tools/esmc/compare-upstream-esmfold2.py --model=fast          # ESMFold2-fast 600M
    <venv with esm>/bin/python tools/esmc/compare-upstream-esmfold2.py --model=full --loops=3,20
    <venv with esm>/bin/python tools/esmc/compare-upstream-esmfold2.py --model=fast-released --loops=3,20   # ESMFold2-Fast

Upstream runs in this process (its fastest kernel backend, `fused`, unless --backends says otherwise) and
`cuda/ef2/localfold-ef2` as a subprocess, on the same targets (the crystals in tools/fixtures) and settings: ONE
diffusion sample, seed 1, the checkpoint's own step count (fast 15 scheduled / 11 run, full 14), and upstream's extra
`lm_dropout` OFF - the port has none; the full model's own per-loop dropout (0.25, part of the architecture) runs on
both. Upstream is timed warm (each case once untimed, then timed, CUDA synchronised); the port reports its own stage
times after the weights are up. GPU memory is the whole PROCESS's as nvidia-smi sees it, for both, each case's own peak - torch's
allocator peak leaves out its context and cache. Each fold is scored by CA RMSD against its crystal (cuda/af3/score.py).

🔴 THE FAST MODEL CANNOT GO THROUGH upstream's fold(): the 600M checkpoint ships without a confidence head and fold()
dies on `KeyError: 'plddt'`, so it is driven through prepare_input + forward and a CA trace is written for scoring.
🔴 AND upstream's defaults are not the port's: fold() samples 16 (fast) or 32 (full) structures and adds 0.3 LM
dropout, and the full checkpoint's config runs 20 loops where the port's bundle says 3 - pass --loops to match.
See docs/EF2FAST.md, "Against upstream ESMFold2".
"""
import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(ROOT, "tools", "fixtures")
THREE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
             MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
TARGETS = [("6mrr", ["A"]), ("1brs", ["A", "D"]), ("5caj", ["A"]), ("1tim", ["A", "B"])]
A3M = {"5caj": ["oracle-dumps/5caj-a.a3m"], "1tim": ["oracle-dumps/1tim-a.a3m"] * 2}


PORT_MODEL = {"fast": "ef2-fast-600m", "full": "ef2", "fast-released": "ef2-fast"}


def chain_seq(pdb, chain):
    seen, s = set(), ""
    for line in open(os.path.join(FIX, f"{pdb}-crystal.pdb")):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[21] == chain and line[22:27] not in seen:
            seen.add(line[22:27]); s += THREE.get(line[17:20], "X")
    return s


class GpuPeak:
    """the process-level GPU memory peak, sampled every 20 ms (above the level when it started)"""
    def __init__(self):
        self.base = self.used(); self.peak = self.base; self.stop = False; self.generation = 0
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()
    @staticmethod
    def used():
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    def run(self):
        while not self.stop:
            try:
                g = self.generation; u = self.used()
                # (a sample begun before a reset() is dropped: nvidia-smi takes tens of ms, and a stale reading -
                # the model load's, the previous case's - written back after the reset was the whole of the peak)
                if g == self.generation: self.peak = max(self.peak, u)
            except Exception: pass
            time.sleep(0.02)
    def reset(self):
        """a new case: the peak restarts from what is held NOW (the weights), still above the starting level"""
        self.generation += 1; self.peak = self.used()
    def gib(self):
        return (self.peak - self.base) / 1024


def rmsd(pdb, name, chains):
    if name.endswith("-x4"):
        return None
    r = subprocess.run([sys.executable, os.path.join(ROOT, "cuda", "af3", "score.py"), pdb,
                        os.path.join(FIX, f"{name.split('+')[0]}-crystal.pdb"), ",".join(chains)], capture_output=True, text=True)
    m = re.search(r"CA RMSD ([0-9.]+)", r.stdout + r.stderr)
    return float(m.group(1)) if m else None


def ca_pdb(coords, chains_out, path):
    lines = []
    for ci, chain in enumerate(chains_out):
        for ri, tok in enumerate(chain.tokens):
            x, y, z = coords[tok.atom_start + 1]          # a standard residue's atoms: N, CA, C, O, ...
            lines.append(f"ATOM  {len(lines)+1:5d}  CA  {tok.residue_name:>3s} {chr(65+ci)}{ri+1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    open(path, "w").write("\n".join(lines) + "\nEND\n")


def upstream(args, cases, work):
    import torch
    from esm.models.esmfold2 import EsmFold2ExperimentalModel, EsmFold2Model
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.msa.msa import MSA
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput
    peak = GpuPeak()
    if args.model == "fast":
        model = EsmFold2ExperimentalModel.from_pretrained(args.fast_checkpoint, load_esmc=True, device="cuda").eval()
        model.configure_lm_dropout(0.0, force_lm_dropout_during_inference=False)
    else:
        model = EsmFold2Model.from_pretrained(args.full_checkpoint if args.model == "full" else args.fast_released_checkpoint,
                                              device="cuda").eval()
    builder = ESMFold2InputBuilder()
    accepted = set(inspect.signature(model.forward).parameters)
    rows = []
    for backend in args.backends.split(","):
        model.set_kernel_backend(None if backend == "none" else backend)
        for loops in args.loop_list:
            for name, chains, seqs, a3ms in cases:
                spi = StructurePredictionInput(sequences=[
                    ProteinInput(id=chr(65 + k), sequence=s, msa=MSA.from_a3m(os.path.join(ROOT, a3ms[k])) if a3ms else None)
                    for k, s in enumerate(seqs)])

                def fold():
                    if args.model != "fast":
                        return builder.fold(model, spi, num_loops=loops, num_sampling_steps=args.steps,
                                            num_diffusion_samples=1, seed=1, lm_dropout=0.0)
                    feats, out_chains = builder.prepare_input(spi, seed=1, device="cuda")
                    with torch.no_grad():
                        out = model.forward(**{k: v for k, v in feats.items() if k in accepted}, num_loops=loops,
                                            num_diffusion_samples=1, num_sampling_steps=args.steps, seed=1)
                    return out, out_chains
                try:
                    torch.cuda.empty_cache(); peak.reset()       # (each case's own peak, not the largest so far)
                    fold(); torch.cuda.synchronize()
                    t = time.time(); res = fold(); torch.cuda.synchronize(); ms = (time.time() - t) * 1000
                    pdb = os.path.join(work, f"up-{name}-{backend}-L{loops}.pdb")
                    if args.model != "fast":
                        import gemmi
                        cif = pdb[:-4] + ".cif"; open(cif, "w").write(res.complex.to_mmcif())
                        st = gemmi.read_structure(cif); st.setup_entities(); st.write_pdb(pdb)
                    else:
                        out, out_chains = res
                        ca_pdb(out["sample_atom_coords"][0].float().cpu().numpy(), out_chains, pdb)
                    rows.append(dict(side=f"upstream {backend}", target=name, loops=loops, ms=ms, gib=peak.gib(),
                                     rmsd=rmsd(pdb, name, chains)))
                except Exception as e:
                    rows.append(dict(side=f"upstream {backend}", target=name, loops=loops, error=f"{type(e).__name__}: {e}"[:160]))
                print(json.dumps(rows[-1]), flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def ours_warm(args, cases, work):
    """the port timed WARM, as upstream is: its resident server (--serve, the Colab worker's mode) folds each case
    three times and the third fold's stage times are read - a one-shot binary since b7118c9 skips its warm-up when the
    input is already written, so its stage times carry every kernel's first launch (6MRR's trunk 128 ms cold, 39 warm)"""
    binary = os.path.join(ROOT, "cuda", "ef2", "localfold-ef2")
    model = PORT_MODEL[args.model]
    if args.model == "fast":
        r = subprocess.run([os.path.join(ROOT, "cuda", "featurise", "fetch-weights"), model], capture_output=True, text=True)
        dirs = re.findall(re.escape(model) + r": (/\S+)", r.stdout + r.stderr)
    else:                       # (the released models' bundles are local exports: cuda/ef2/README.md, "The released models")
        r = subprocess.CompletedProcess([], 1, "", "no local export")
        dirs = [os.path.join(ROOT, "model-esmfold2-f32" if args.model == "full" else "model-esmfold2-fast-f32"),
                os.path.join(ROOT, "model-esmc-6b-int8")]
        dirs = dirs if all(os.path.isdir(d) for d in dirs) else []
    if len(dirs) != 2:
        return [dict(side="localfold-ef2 warm", target=n, loops=l, error="fetch-weights: " + (r.stdout + r.stderr)[-160:])
                for l in args.loop_list for n, *_ in cases]
    serve = os.path.join(work, "serve"); os.makedirs(serve, exist_ok=True)
    log = open(os.path.join(serve, "server.log"), "w")
    proc = subprocess.Popen([binary, "-", f"--fold-bundle={dirs[0]}", f"--esmc-bundle={dirs[1]}", "--fast", "--warm=96,800",
                             f"--serve={serve}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, EF2_PASSES=str(args.loop_list[0] + 1)))
    if len(args.loop_list) > 1:
        print("(the warm server takes the first --loops only)", flush=True)
    rows, count = [], 0
    def job(inputs, flags):
        nonlocal count
        count += 1
        base = os.path.join(serve, f"{count:06d}")
        open(base + ".tmp", "w").write("\n".join([inputs, *flags]) + "\n"); os.rename(base + ".tmp", base + ".job")
        while not os.path.exists(base + ".done"):
            if proc.poll() is not None: raise RuntimeError(open(log.name).read()[-300:])
            time.sleep(0.005)
        return int(open(base + ".done").read().strip() or 1), open(base + ".log").read()
    try:
        for loops in args.loop_list:
            for name, chains, seqs, a3ms in cases:
                inputs = os.path.join(work, f"in-{name}")
                f = subprocess.run([os.path.join(ROOT, "cuda", "featurise", "ef2-featurise"), inputs, "--sequence=" + ":".join(seqs),
                                    f"--fold-bundle={dirs[0]}"], capture_output=True, text=True)
                if f.returncode != 0 or a3ms:
                    rows.append(dict(side="localfold-ef2 warm", target=name, loops=loops, error=(f.stdout + f.stderr)[-160:] or "alignment"))
                    continue
                pdb = os.path.join(work, f"lfw-{name}-L{loops}.pdb")
                peak = GpuPeak()
                for k in range(3):
                    code, said = job(inputs, [f"--out={pdb}", "--seed=1"])
                peak.stop = True
                stages = {m.group(1): float(m.group(2)) for m in re.finditer(r"^([a-z][a-z ]+?) ([0-9.]+) ms", said, re.M)}
                rows.append(dict(side="localfold-ef2 warm", target=name, loops=loops, ms=sum(stages.values()), gib=peak.gib(),
                                 rmsd=rmsd(pdb, name, chains) if code == 0 else None,
                                 **({} if code == 0 else {"error": said.strip()[-160:]})))
                print(json.dumps(rows[-1]), flush=True)
        open(os.path.join(serve, "zzzzzz.tmp"), "w").write("quit\n"); os.rename(os.path.join(serve, "zzzzzz.tmp"), os.path.join(serve, "zzzzzz.job"))
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None: proc.kill()
    return rows


def ours(args, cases, work):
    binary = os.path.join(ROOT, "cuda", "ef2", "localfold-ef2")
    rows = []
    for loops in args.loop_list:
        for name, chains, seqs, a3ms in cases:
            pdb = os.path.join(work, f"lf-{name}-L{loops}.pdb")
            cmd = [binary, f"--model={PORT_MODEL[args.model]}", "--sequence=" + ":".join(seqs),
                   "--seed=1", f"--out={pdb}"] + ([f"--a3m={','.join(os.path.join(ROOT, a) for a in a3ms)}"] if a3ms else [])
            peak = GpuPeak()
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, EF2_PASSES=str(loops + 1)))
            wall = time.time() - t
            peak.stop = True
            said = r.stdout + r.stderr
            stages = {m.group(1): float(m.group(2)) for m in re.finditer(r"^([a-z][a-z ]+?) ([0-9.]+) ms", said, re.M)}
            rows.append(dict(side="localfold-ef2", target=name, loops=loops, ms=sum(stages.values()), gib=peak.gib(),
                             command_s=wall, rmsd=rmsd(pdb, name, chains) if r.returncode == 0 else None,
                             **({} if r.returncode == 0 else {"error": said.strip()[-160:]})))
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["fast", "full", "fast-released"], default="fast",
                   help="fast: the 600M experimental (the site's); full: ESMFold2; fast-released: ESMFold2-Fast (ESM-C 6B)")
    p.add_argument("--loops", default="", help="comma-separated; default 3")
    p.add_argument("--backends", default="fused", help="upstream kernel backends: none, fused, cuequivariance")
    p.add_argument("--fast-checkpoint", default=os.path.join(os.path.dirname(ROOT), "ef2", "esmfold2-fast-600m"))
    p.add_argument("--full-checkpoint", default="biohub/ESMFold2")
    p.add_argument("--fast-released-checkpoint", default="biohub/ESMFold2-Fast")
    p.add_argument("--no-msa", action="store_true", help="skip the alignment cases (the full model only reads them)")
    p.add_argument("--only", default="", help="targets whose name contains this")
    p.add_argument("--ours-only", action="store_true", help="skip upstream (no esm package needed)")
    args = p.parse_args()
    args.loop_list = [int(x) for x in (args.loops or "3").split(",")]
    args.steps = 15 if args.model == "fast" else 14
    cases = [(n, cs, [chain_seq(n, c) for c in cs], None) for n, cs in TARGETS]
    cases.append(("1tim-x4", ["A"] * 4, [chain_seq("1tim", "A")] * 4, None))       # ~1000 tokens, speed only
    if args.model == "full" and not args.no_msa:
        cases += [(n + "+msa", dict(TARGETS)[n], [chain_seq(n, c) for c in dict(TARGETS)[n]], A3M[n]) for n in A3M]
    cases = [c for c in cases if args.only in c[0]]
    work = tempfile.mkdtemp(prefix="compare-ef2-")
    rows = (upstream(args, cases, work) if not args.ours_only else []) + ours(args, cases, work)
    rows += ours_warm(args, cases, work)
    print(f"\n{'target':10s} {'loops':>5s}  " + "  ".join(f"{s:>28s}" for s in sorted({r['side'] for r in rows})))
    for name, *_ in cases:
        for loops in args.loop_list:
            cells = []
            for side in sorted({r["side"] for r in rows}):
                r = next((x for x in rows if x["side"] == side and x["target"] == name and x["loops"] == loops), None)
                cells.append("error" if not r or "error" in r else
                             f"{r['ms']:7.0f} ms {r['gib']:4.1f} GiB {r['rmsd'] if r['rmsd'] is not None else '-':>5} A")
            print(f"{name:10s} {loops:5d}  " + "  ".join(f"{c:>28s}" for c in cells))
    print(f"\n(folds in {work})")


if __name__ == "__main__":
    main()

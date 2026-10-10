#!/usr/bin/env python3
"""One AlphaFold 3 fold across several GPUs (cuda/af3 --gpus=N): correctness against the one-GPU fold, then scaling.

For a rented multi-GPU box (the port's development box has one GPU; the ranks can share it with --gpu-map=0,0 for a
smoke test, which checks correctness and nothing about speed):

  python3 tools/check-multigpu.py                      # build, correctness, scaling over 1..every GPU, a report
  python3 tools/check-multigpu.py --sizes=4,12,24      # complexes of 4, 12, 24 copies of 1TIM's chain (247 a copy)
  python3 tools/check-multigpu.py --big=41             # and one fold of 41 copies (10,127 tokens) on every GPU
  python3 tools/check-multigpu.py --gpu-map=0,0 --gpus=2 --sizes=4 --skip-build   # the smoke test on one GPU

Correctness: 5CAJ chain A and the 1TIM dimer, each self-templated, folded on one GPU and on N; both scored against
their crystal, and the N-GPU fold against the one-GPU fold (the sharded trunk reorders sums - a few thousandths of
an angstrom here, 0.003-0.009 A measured with ranks sharing one A100). Scaling: each size on 1, 2, 4, ... GPUs
(--steps=25 --recycles=0 unless --full), the fold's own stage times and every GPU's peak memory (nvidia-smi, polled).
Writes multigpu-report.md (and .json) under --out.
"""
import argparse, json, math, os, re, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cuda/af3/localfold-af3"
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}
# 5CAJ's construct (the sequence every 5CAJ number in cuda/af3/README.md folds; the crystal's CA records miss its ends)
C5 = "PRGSHMLILISPAKTLDYQSPLTTTRYTLPELLDNSQQLIHEARKLTPPQISTLMRISDKLAGINAARFHDWQPDFTPANARQAILAFKGDVYTGLQAETFSEDDFDFAQQHLRMLSGLYGVLRPLDLMQPYRLEMGIRLENARGKDLYQFWGDIITNKLNEALAAQGDNVVINLASDEYFKSVKPKKLNAEIIKPVFLDEKNGKFKIISFYAKKARGLMSRFIIENRLTKPEQLTGFNSEGYFFDEDSSSNGELVFKRYE"
TIM = ("APRKFFVGGNWKMNGKRKSLGELIHTLDGAKLSADTEVVCGAPSIYLDFARQKLDAKIGVAAQNCYKVPKGAFTGEISPAMIKDIGAAWVILGHSERRHVFGESDELIGQ"
       "KVAHALAEGLGVIACIGEKLDEREAGITEKVVFQETKAIADNVKDWSKVVLAYEPVWAIGTGKTATPQQAQEVHEKLRGWLKTHVSDAVAVQSRIIYGGSVTGGNCKELASQ"
       "HDVDGFLVGGASLKPEFVDIINAKH")


def chains(pdb):
    out = {}
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.setdefault(line[21], []).append(THREE.get(line[17:20], "X"))
    return {k: "".join(v) for k, v in out.items()}


def gpu_count():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v: return len([x for x in v.split(",") if x])
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.startswith("GPU")])


class Peak:
    """every GPU's peak memory in use while it runs (nvidia-smi, polled)"""
    def __init__(self): self.peak = {}; self.stop = False
    def __enter__(self):
        def poll():
            while not self.stop:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
                for line in r.stdout.splitlines():
                    try: i, m = (int(x) for x in line.split(","))
                    except ValueError: continue
                    self.peak[i] = max(self.peak.get(i, 0), m)
                time.sleep(0.5)
        self.t = threading.Thread(target=poll, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self.stop = True; self.t.join()


def fold(args, gpus, flags, out, env_extra=None):
    env = dict(os.environ, LOCALFOLD_ACCEPT_MODEL_TERMS="alphafold3")
    if args.gpu_map and gpus > 1: env["LOCALFOLD_GPU_MAP"] = args.gpu_map
    env.update(env_extra or {})
    cmd = [str(BIN)] + flags + [f"--out={out}"] + ([f"--gpus={gpus}"] if gpus > 1 else [])
    t0 = time.time()
    with Peak() as pk:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    wall = time.time() - t0
    log = r.stdout + r.stderr
    m = re.search(r"fold 1: trunk ([\d.]+) ms.*?diffusion ([\d.]+) ms.*?confidence ([\d.]+) ms, total ([\d.]+) ms", log)
    p = re.search(r"mean pLDDT ([\d.]+)\s+pTM ([\d.]+)", log)
    res = {"gpus": gpus, "ok": r.returncode == 0 and Path(out).exists(), "wall_s": round(wall, 1),
           "peak_mib": dict(sorted(pk.peak.items()))}
    if m: res.update(trunk_ms=float(m[1]), diffusion_ms=float(m[2]), confidence_ms=float(m[3]), total_ms=float(m[4]))
    if p: res.update(plddt=float(p[1]), ptm=float(p[2]))
    if not res["ok"]: res["error"] = log.strip().splitlines()[-5:]
    return res


def rmsd(a, b, chain_ids):
    r = subprocess.run([sys.executable, str(ROOT / "cuda/af3/score.py"), str(a), str(b), chain_ids], capture_output=True, text=True)
    m = re.search(r"CA RMSD ([\d.]+)", r.stdout)
    return float(m[1]) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=0, help="the most GPUs to use (default: every one)")
    ap.add_argument("--gpu-map", default="", help="LOCALFOLD_GPU_MAP, e.g. 0,0 to share one GPU (a smoke test)")
    ap.add_argument("--sizes", default="4,12,24", help="scaling complexes, in copies of 1TIM's 247-residue chain")
    ap.add_argument("--big", type=int, default=0, help="also one fold of this many copies on every GPU, 100 steps")
    ap.add_argument("--full", action="store_true", help="scaling folds at 100 steps and 3 recycles, not 25 and 0")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--out", default="multigpu-out")
    args = ap.parse_args()
    most = args.gpus or gpu_count()
    out = (ROOT / args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        subprocess.run(["bash", "cuda/build.sh", "af3"], cwd=ROOT, check=True)
    report = {"gpus": most, "nvidia_smi": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip(),
              "correctness": [], "scaling": []}
    counts = [1] + [g for g in (2, 4, 8, 16) if g <= most]
    if most not in counts: counts.append(most)
    # correctness: one GPU against every GPU, self-templated targets scored on their crystal
    fx = ROOT / "tools/fixtures"
    cases = [("5CAJ chain A", C5, "tools/fixtures/5caj-crystal.pdb:A@0",
              fx / "5caj-crystal.pdb", "A"),
             ("1TIM dimer", ":".join([chains(fx / "1tim-crystal.pdb")[c] for c in "AB"]),
              "tools/fixtures/1tim-crystal.pdb:A@0+tools/fixtures/1tim-crystal.pdb:B@1", fx / "1tim-crystal.pdb", "A,B")]
    for name, seq, tmpl, crystal, ids in cases:
        flags = [f"--sequence={seq}", "--seed=1", "--steps=100", f"--template={tmpl}"]
        tag = name.split()[0].lower()
        one = fold(args, 1, flags, out / f"{tag}-1.pdb")
        many = fold(args, most, flags, out / f"{tag}-{most}.pdb")
        row = {"case": name, "one": one, "many": many}
        if one["ok"]: row["one_vs_crystal"] = rmsd(out / f"{tag}-1.pdb", crystal, ids)
        if many["ok"]: row["many_vs_crystal"] = rmsd(out / f"{tag}-{most}.pdb", crystal, ids)
        if one["ok"] and many["ok"]: row["many_vs_one"] = rmsd(out / f"{tag}-{most}.pdb", out / f"{tag}-1.pdb", ids)
        report["correctness"].append(row)
        print(json.dumps(row)[:400], flush=True)
    # scaling
    speed = ["--steps=100", "--recycles=3"] if args.full else ["--steps=25", "--recycles=0"]
    for copies in [int(x) for x in args.sizes.split(",") if x]:
        seq = ":".join([TIM] * copies)
        for g in counts:
            r = fold(args, g, [f"--sequence={seq}", "--seed=1"] + speed, out / f"tim{copies}-{g}.pdb")
            r.update(copies=copies, tokens=copies * len(TIM))
            report["scaling"].append(r)
            print(json.dumps(r)[:400], flush=True)
    if args.big:
        seq = ":".join([TIM] * args.big)
        r = fold(args, most, [f"--sequence={seq}", "--seed=1", "--steps=100", "--recycles=0"], out / f"tim{args.big}-{most}.pdb")
        r.update(copies=args.big, tokens=args.big * len(TIM), big=True)
        report["scaling"].append(r)
        print(json.dumps(r)[:400], flush=True)
    (out / "multigpu-report.json").write_text(json.dumps(report, indent=1))
    lines = ["# Several GPUs on one fold", "", "```", report["nvidia_smi"], "```", "", "## Correctness (self-templated)", "",
             "| case | 1 GPU vs crystal | N GPUs vs crystal | N vs 1 | pLDDT 1 / N |", "|---|---:|---:|---:|---|"]
    for c in report["correctness"]:
        lines.append(f"| {c['case']} | {c.get('one_vs_crystal')} | {c.get('many_vs_crystal')} | {c.get('many_vs_one')} | "
                     f"{c['one'].get('plddt')} / {c['many'].get('plddt')} |")
    lines += ["", "## Scaling", "", "| tokens | GPUs | trunk ms | diffusion ms | confidence ms | total ms | peak MiB a GPU | ok |",
              "|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in report["scaling"]:
        pk = max(r["peak_mib"].values()) if r["peak_mib"] else None
        lines.append(f"| {r['tokens']} | {r['gpus']} | {r.get('trunk_ms')} | {r.get('diffusion_ms')} | {r.get('confidence_ms')} | "
                     f"{r.get('total_ms')} | {pk} | {r['ok']} |")
    (out / "multigpu-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

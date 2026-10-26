"""The WebGPU arm: a case folded by the page's own AlphaFold 2 path in headless Chrome.

    python3 cuda/af2/parity/webgpu_fold.py <case.json> <out.pdb> [fold-af2.js flags...]

Merges the case's alignments with LocalFold's own merge (ours_features.mjs, the page's mergeSearchedChains),
serves the result from oracle-dumps/af2-parity/, and folds it through tools/gpu/fold-af2.js - predictA3m on
the published bundle - which returns the PDB web/prediction-results.js writes. Needs node on PATH, and on a
Linux/NVIDIA box Xvfb on DISPLAY=:99 (docs/A100.md: headless Chrome steers the adapter to software).
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    case_path, out = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]
    case = json.load(open(case_path))
    chains = case["sequences"]
    tag = hashlib.sha1(open(case_path, "rb").read()).hexdigest()[:12]
    served = f"oracle-dumps/af2-parity/{tag}.a3m"
    if not os.path.exists(os.path.join(ROOT, served)):
        subprocess.run(["node", os.path.join(ROOT, "cuda/af2/parity/ours_features.mjs"), case_path,
                        os.path.join(ROOT, served + ".features.json"), os.path.join(ROOT, served)],
                       check=True, stdout=subprocess.DEVNULL)
    family = "multimer" if "multimer" in case["model_type"] else "monomer"
    args = ["node", "tools/gpu-chrome.mjs", "tools/gpu/fold-af2.js", f"--family={family}",
            f"--sequence={''.join(chains)}", f"--a3m=/{served}", "--pdb"]
    if len(chains) > 1:
        args.append("--chains=" + ",".join(str(len(c)) for c in chains))
    args += extra
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":99"),
               XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR", "/tmp/xdg"))
    run = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    text = run.stdout
    if "[gpu-chrome] adapter: nvidia" not in text and "adapter: apple" not in text:
        raise SystemExit("not on a real GPU adapter:\n" + text[-2000:] + run.stderr[-2000:])
    result = json.loads(text[text.index("\n{") + 1:])
    open(out, "w").write(result.pop("pdb"))
    print(json.dumps({k: result[k] for k in ("meanPlddt", "ptm", "iptm", "passesRun", "elapsedMilliseconds")
                      if k in result}))


if __name__ == "__main__":
    main()

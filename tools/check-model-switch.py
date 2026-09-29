"""Ten models folded in ONE page, as a reader switching models does.

    npm run test:switch

🔴 EVERY OTHER GATE FOLDS ONE MODEL PER PAGE, so none could see what a reader
sees after trying a few: residency is kept between folds so the same model's
next fold skips its packing, and it was kept across a CHANGE of model too,
because it is keyed on each family's weight objects and the page keeps every
store it has loaded. Four models in one page held all four - 677, 1281, 2171,
2874 MiB live on an A100 - and the fourth fold died in WebGPU validation
("Invalid Buffer ... due to a previous error"); on a Colab L4 the device was
lost outright. `releaseAllWeights` on a change of family is the fix
(web/app.js).

What this asserts, after each fold of 6MRR by each model in turn:
  - the fold FINISHED (the status line is not an error), and
  - the device holds one model's worth, not a running total: live bytes stay
    under LIMIT_MIB. Measured with the fix, the largest single model is
    IntelliFold-2 at 1313 MiB; without it the tenth fold sat at 4165 and six
    of ten failed.
Watched failing with the release removed. Needs the GPU lane
(DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg) and the network for the weights.
"""
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cdp  # noqa: E402

MODELS = ["openbind0", "af3", "boltz2", "protenix2", "intellifold2", "rosettafold3", "opendde",
          "ef2-fast-600m", "monomer", "multimer"]
LIMIT_MIB = 2048
SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
SNAPSHOT = """(async () => {
  const { getDevice } = await import('/web/model.js');
  const { memorySnapshot } = await import('/src/runtime/device-memory.js');
  const s = memorySnapshot(await getDevice());
  const status = document.getElementById('status-message');
  return JSON.stringify({ mib: Math.round(s.residentBytes / 1048576),
    top: s.currentByLabel.slice(0, 3).map((r) => r.label + ' ' + Math.round(r.bytes / 1048576)),
    status: status.textContent, error: status.classList.contains('error') });
})()"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main():
    port = free_port()
    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools/serve.py"), str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc, ws = cdp.launch(free_port(), f"/tmp/localfold-switch-{os.getpid()}")
    failures = 0
    try:
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/index.html")
        cdp.wait_for(ws, "!!window.__entityList", 120, "the page")
        cdp.evaluate(ws, """(() => { for (const k of ['alphafold3','openbind0','opendde','boltz2',
          'protenix2','intellifold2','rosettafold3']) localStorage.setItem('localfold.modelTerms.' + k,
          'accepted'); return 1; })()""")
        rows = json.dumps([{"type": "protein", "value": SEQUENCE, "copies": 1}])
        for model in MODELS:
            controls = {"model-family": model, "af3-mode": "diffusion", "msa-mode": "none",
                        "af2Model": "1", "plm-mode": "esmc-600m"}
            cdp.evaluate(ws, f"""(() => {{ window.__entityList.set({rows});
              window.__foldInputs.apply({{ entities: {rows}, controls: {json.dumps(controls)} }});
              return 1; }})()""")
            time.sleep(1.5)
            cdp.evaluate(ws, "document.getElementById('predict').click()")
            started = time.time()
            while time.time() - started < 900:
                time.sleep(0.5)
                state = cdp.evaluate(ws, "JSON.stringify(window.__foldState ?? {})")
                if '"running":false' in state and time.time() - started > 2:
                    break
            snap = json.loads(cdp.evaluate(ws, SNAPSHOT))
            problem = ("the fold failed" if snap["error"] else
                       f"{snap['mib']} MiB live, over {LIMIT_MIB}" if snap["mib"] > LIMIT_MIB else None)
            failures += problem is not None
            print(f"{'ok  ' if problem is None else 'FAIL'}  {model:14s} {time.time() - started:5.1f}s"
                  f"  live {snap['mib']:5d} MiB  {snap['status'][:70]}"
                  + ("" if problem is None else f"\n      {problem}: {'; '.join(snap['top'])}"),
                  flush=True)
    finally:
        proc.terminate()
        server.terminate()
    if failures:
        sys.exit(f"{failures} of {len(MODELS)} folds in one page failed")
    print("ten models fold in one page, holding one model at a time")


if __name__ == "__main__":
    main()

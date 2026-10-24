"""Does design.html load, run its module, and say true things?

    python3 tools/check-design-page.py

🔴 WHY THIS EXISTS WHEN `test:site` AND `mobile-layout.py` ALREADY TOUCH THE
PAGES. Neither one runs a page's JavaScript. `build_site.py` resolves the module
graph - it proves every import can be fetched - and `mobile-layout.py` measures
where boxes land. design.html's first bug was neither: `createEntityList` calls
its `onChange` while it is still constructing, so a callback reading the binding
being initialised threw `Cannot access 'entities' before initialization` and the
module never loaded. The graph resolved perfectly and every box was in the right
place on a page that did nothing.

🔴 AND IT ASSERTS THE AUTOMATIC DESIGNER RESPONDS TO THE TARGET, not merely that
it printed something. A note reading "Auto -> SolubleMPNN" is also what a broken
wire prints, because soluble is the fallback: the check therefore adds a ligand
and requires the answer to MOVE to LigandMPNN, which is the only version of this
test that can fail for the right reason.

What it does not do: run the loop, which is not wired (docs/DESIGN.md), or touch
a GPU beyond asking whether an adapter answers.
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

# A ligand beside the protein: the one change that must move the automatic
# designer, because only LigandMPNN's encoder reads heteroatoms.
WITH_LIGAND = [{"type": "protein", "copies": 1, "value": "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"},
               {"type": "ligand", "copies": 1, "value": "HEM"}]

READ_PAGE = """(() => {
  const t = (id) => document.getElementById(id)?.textContent?.trim() ?? null;
  const n = (sel) => document.querySelectorAll(sel).length;
  return JSON.stringify({
    designer: t("designer-note"), folder: t("folder-note"), gpu: t("gpu-summary"),
    entityRows: n("#entity-rows > *"), roles: n("#roles-legend .role-list li"),
    columns: n("#results-table thead th"),
    hunt: document.getElementById("hunt")?.textContent?.trim(),
  });
})()"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main():
    port = free_port()
    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools/serve.py"), str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc, ws = cdp.launch(free_port(), f"/tmp/localfold-design-{os.getpid()}")
    bad = []
    try:
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/design.html")
        # 🔴 THE MODULE RAN IFF IT WROTE THE DESIGNER'S REASON. There is no
        # window marker to wait on, and `readyState === "complete"` is true of a
        # page whose module threw.
        try:
            cdp.wait_for(ws, "document.getElementById('designer-note')?.textContent.length > 0",
                         60, "web/design.js to run")
        except RuntimeError:
            threw = cdp.evaluate(ws, """(async () => {
              try { await import("/web/design.js"); return "imported cleanly the second time"; }
              catch (e) { return String((e && (e.stack || e.message)) || e); }
            })()""")
            print(f"FAIL: web/design.js did not run:\n{threw}")
            return 1

        page = json.loads(cdp.evaluate(ws, READ_PAGE))
        for key, value in page.items():
            print(f"  {key}: {value}")
        if not (page["designer"] or "").startswith("Auto"):
            bad.append("the designer note is not the automatic reason")
        if page["roles"] != 5:
            bad.append(f"{page['roles']} role swatches, expected 5")
        if page["entityRows"] == 0:
            bad.append("the target list drew no rows")
        if page["columns"] != 5:
            bad.append(f"{page['columns']} result columns, expected 5")
        if (page["gpu"] or "") == "":
            bad.append("the page did not say whether WebGPU is available")

        # ...and that the automatic choice is wired to the target, not printed.
        cdp.evaluate(ws, f"""(() => {{ window.__designEntities.set({json.dumps(WITH_LIGAND)}); return 1; }})()""")
        time.sleep(0.3)
        withLigand = cdp.evaluate(ws, "document.getElementById('designer-note').textContent")
        print(f"  with a ligand: {withLigand}")
        if "LigandMPNN" not in withLigand:
            bad.append(f"a ligand did not move the automatic designer: {withLigand!r}")

        # Pressing Hunt must explain itself rather than throw or do nothing.
        cdp.evaluate(ws, "document.getElementById('hunt').click()")
        time.sleep(0.3)
        status = cdp.evaluate(ws, "document.getElementById('status-message').textContent")
        print(f"  after Hunt: {status}")
        if "not wired" not in status:
            bad.append("Hunt did not say the loop is missing")
    finally:
        proc.terminate()
        server.terminate()

    print("FAIL: " + "; ".join(bad) if bad else "design.html: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

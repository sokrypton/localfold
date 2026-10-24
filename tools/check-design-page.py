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

        # ...and that switching method disables exactly the controls the other
        # method cannot honour, WITH their reasons - greyed and explained, never
        # removed, which is the page's own rule.
        switched = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById("method");
          pick.value = "rfd4p";
          pick.dispatchEvent(new Event("change"));
          const groups = ["designerGroup", "folderGroup", "scheduleGroup"];
          return JSON.stringify({
            note: document.getElementById("method-note")?.textContent?.trim() ?? "",
            disabled: groups.filter((id) => {
              const g = document.getElementById(id);
              return g !== null && [...g.querySelectorAll("select, input")].every((c) => c.disabled);
            }),
            reasons: groups.filter((id) => (document.getElementById(id)
              ?.querySelector(".unavailable-reason")?.textContent ?? "").length > 0),
            stillThere: groups.filter((id) => {
              const g = document.getElementById(id);
              return g !== null && g.offsetParent !== null;
            }),
          });
        })()"""))
        print(f"  on RFD4-Proteina: {switched}")
        for key in ("disabled", "reasons", "stillThere"):
            if len(switched[key]) != 3:
                bad.append(f"switching method: {key} covered {switched[key]}, expected all three groups")
        if "flow matching" not in switched["note"] or "🔴" in switched["note"]:
            # 🔴 AND THE REPO'S COMMENT MARKER MUST NOT REACH THE PAGE. It is a
            # convention for source, not a thing to show a reader; it leaked into this
            # very note once.
            bad.append(f"the method note is wrong or carries a source marker: {switched['note']!r}")

        # ...and switching back must re-enable them, or the page is a one-way door.
        back = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById("method");
          pick.value = "hunter";
          pick.dispatchEvent(new Event("change"));
          return JSON.stringify({ enabled: [...document.querySelectorAll(
            "#designerGroup select, #folderGroup select, #scheduleGroup input")].every((c) => !c.disabled) });
        })()"""))
        if not back["enabled"]:
            bad.append("switching back to Protein Hunter left controls disabled")

        # 🔴 AND A REAL HUNT, END TO END: two models' weights, a fold, an MPNN
        # redesign, three more folds with the model rotating under the sequence,
        # and a row per cycle. A 31-residue target and a 12-residue binder runs
        # in about thirty seconds on an M5 off the local bundles.
        cdp.evaluate(ws, """(() => {
          for (const k of ['alphafold3','openbind0','opendde','boltz2','protenix2',
                           'intellifold2','rosettafold3'])
            localStorage.setItem('localfold.modelTerms.' + k, 'accepted');
          window.__designEntities.set([{ type: 'protein', copies: 1,
            value: 'GWSTELEKHREELKEFLKKEGITNVEIRIDN' }]);
          document.getElementById('binder-length').value = '12';
          document.getElementById('cycles').value = '3';
          document.getElementById('runs').value = '1';
          // 🔴 TWO FOLDERS, so the run exercises the rotation and not just the
          // loop: a design has to satisfy several models rather than overfit
          // one, and that only happens if the model changes under it.
          for (const option of document.getElementById('folders').options) {
            option.selected = option.value === 'af3' || option.value === 'opendde';
          }
          document.getElementById('folders').dispatchEvent(new Event('change'));
          document.getElementById('hunt').click();
          return 1;
        })()""")
        deadline = time.time() + 300
        hunted = None
        while time.time() < deadline:
            hunted = json.loads(cdp.evaluate(ws, """JSON.stringify({
              status: document.getElementById('status-message').textContent,
              failed: document.getElementById('status-message').classList.contains('error'),
              rows: document.querySelectorAll('#results-body tr').length,
              best: [...document.querySelectorAll('#results-body tr.best td')].map((c) => c.textContent),
              folders: [...document.querySelectorAll('#results-body tr')].map((r) => r.children[2].textContent),
              cycle0Best: (document.querySelector('#results-body tr')?.className ?? '').includes('best'),
              button: document.getElementById('hunt').textContent.trim(),
              progress: document.getElementById('progress').value,
              track: document.querySelectorAll('#cycle-track > *').length,
            })"""))
            if hunted["failed"]:
                break
            if hunted["button"] == "Hunt" and hunted["rows"] > 0:
                break
            time.sleep(2)

        print(f"  hunt: {hunted['status']}")
        print(f"        folded by: {hunted['folders']}")
        print(f"        rows={hunted['rows']} track={hunted['track']} progress={hunted['progress']}"
              f" best={hunted['best']}")
        # 🔴 A MISSING BUNDLE IS NOT A FAILING PAGE, AND IS NOT A SILENT PASS
        # EITHER. This box has model-af3-int5; another checkout may not, and the
        # difference has to be visible in the output rather than inferred.
        if hunted["failed"] and ("404" in hunted["status"] or "manifest" in hunted["status"]):
            print("        skipped the hunt: this checkout has no AF3 bundle to fold with")
        elif hunted["failed"]:
            bad.append(f"the hunt failed: {hunted['status']}")
        else:
            # cycle 0 and cycle 1, so the loop folded, designed, and folded again
            if hunted["rows"] < 4:
                bad.append(f"the hunt drew {hunted['rows']} rows, expected cycle 0 through 3")
            # 🔴 THE ROTATION, READ OFF THE TABLE. Alternating is the whole
            # point; a column reading af3 four times is a rotation that is not
            # rotating, and would look like a working ensemble.
            if hunted["folders"][:4] != ["af3", "opendde", "af3", "opendde"]:
                bad.append(f"the folder did not rotate per cycle: {hunted['folders']}")
            if hunted["track"] < 2:
                bad.append(f"the sequence track has {hunted['track']} lines, expected one per cycle")
            if "done" not in hunted["status"]:
                bad.append(f"the hunt did not finish: {hunted['status']!r}")
            if hunted["progress"] != 1:
                bad.append(f"the bar ended at {hunted['progress']}, not 1")
            # 🔴 EXACTLY ONE BEST ROW, five cells wide. "At least one" passed
            # happily while three rows carried the marker, because each had been
            # the leader when it was drawn.
            if len(hunted["best"]) != 5:
                bad.append(f"expected exactly one best row of 5 cells, got {len(hunted['best'])} cells:"
                           f" {hunted['best']}")
            # 🔴 THE LOOP'S OWN RULE, CHECKED THROUGH THE PAGE: cycle 0 is the
            # mostly-X draw the run STARTED from, so it must never be the best
            # row however well it folded - and on a 12-mer of mostly X it folds
            # deceptively well.
            if hunted["cycle0Best"]:
                bad.append("cycle 0 was marked best; it is the starting draw and cannot win")

        # Pressing Hunt must explain itself rather than throw or do nothing.
        # On RFD4 it must say so rather than pretending: the engine is in the
        # tree, the page is not joined to it.
        cdp.evaluate(ws, """(() => {
          const pick = document.getElementById('method');
          pick.value = 'rfd4p';
          pick.dispatchEvent(new Event('change'));
          document.getElementById('hunt').click();
          return 1;
        })()""")
        time.sleep(0.3)
        onRfd4p = cdp.evaluate(ws, "document.getElementById('status-message').textContent")
        print(f"  Hunt on RFD4-Proteina: {onRfd4p}")
        if "not wired" not in onRfd4p:
            bad.append(f"Hunt on RFD4 did not say it is unwired: {onRfd4p!r}")
    finally:
        proc.terminate()
        server.terminate()

    print("FAIL: " + "; ".join(bad) if bad else "design.html: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

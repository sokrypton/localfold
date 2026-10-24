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

# A ligand span beside a designed one: the change that must move the automatic
# designer, because only LigandMPNN's encoder reads heteroatoms.
WITH_LIGAND = [{"kind": "generated", "min": 60, "max": 60},
               {"kind": "break"},
               {"kind": "ligand", "code": "HEM"}]

# The target and binder a hunt runs on, as spans.
HUNT_SPANS = [{"kind": "sequence", "value": "GWSTELEKHREELKEFLKKEGITNVEIRIDN"},
              {"kind": "break"},
              {"kind": "generated", "min": 12, "max": 12}]

READ_PAGE = """(() => {
  const t = (id) => document.getElementById(id)?.textContent?.trim() ?? null;
  const n = (sel) => document.querySelectorAll(sel).length;
  return JSON.stringify({
    designer: t("designer-note"), folder: t("folder-note"), gpu: t("gpu-summary"),
    spanRows: n("#contig-rows > *"), roles: n("#roles-legend .role-list li"),
    contig: t("contig-line"), problems: t("contig-problems"),
    presets: document.querySelectorAll("#preset option").length,
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
        if page["spanRows"] != 3:
            bad.append(f"the span editor drew {page['spanRows']} rows, expected 3")
        # 🔴 THE LINE DESCRIBES WHAT THE SELECTED METHOD WILL ACTUALLY BE
        # ASKED, which differs by method and must. The page opens on Protein
        # Hunter, whose input is chains - so it reads "given 114 · design 75"
        # for the default job. A contig would be actively wrong here: a given
        # SEQUENCE has no contig notation, so the line read "/0,75-75" for a job
        # whose target is 114 residues long. The RFD4 form is checked below,
        # through a preset.
        if "given 114" not in (page["contig"] or "") or "design 75" not in (page["contig"] or ""):
            bad.append(f"the line does not describe the default job to Hunter: {page['contig']!r}")
        if page["presets"] < 11:
            bad.append(f"the preset picker lists {page['presets']} entries, expected 10 plus the blank")
        if page["columns"] != 5:
            bad.append(f"{page['columns']} result columns, expected 5")
        if (page["gpu"] or "") == "":
            bad.append("the page did not say whether WebGPU is available")

        # ...and that the automatic choice is wired to the target, not printed.
        cdp.evaluate(ws, f"""(() => {{ window.__designSpans.set({json.dumps(WITH_LIGAND)}); return 1; }})()""")
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
          const HUNT_SPANS_JSON = __SPANS__;
          for (const k of ['alphafold3','openbind0','opendde','boltz2','protenix2',
                           'intellifold2','rosettafold3'])
            localStorage.setItem('localfold.modelTerms.' + k, 'accepted');
          window.__designSpans.set(HUNT_SPANS_JSON);
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
        })()""".replace("__SPANS__", json.dumps(HUNT_SPANS)))
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
              // 🔴 THE VIEWER'S OWN FRAME COUNT, from py2Dmol rather than from
              // our bookkeeping: one frame a cycle is what makes the play bar
              // walk the hunt, and a viewer that drew nothing would otherwise
              // pass every other assertion here.
              frames: (() => {
                // 🔴 THE REGISTRY HOLDS A WRAPPER, NOT THE RENDERER. Each entry
                // is { handleIncrementalStateUpdate, handleReplaceFrame,
                // renderer } - reading frames off the wrapper gives 0 for a
                // viewer that is working perfectly, which is what this check
                // did until it was probed.
                const registry = window.py2dmol_viewers ?? {};
                const entry = Object.values(registry)[0];
                const renderer = entry?.renderer ?? entry;
                const name = renderer?.currentObjectName;
                return renderer?.objectsData?.[name]?.frames?.length ?? 0;
              })(),
              viewerShown: document.getElementById('viewer-container').offsetParent !== null,
            })"""))
            if hunted["failed"]:
                break
            if hunted["button"] == "Hunt" and hunted["rows"] > 0:
                break
            time.sleep(2)

        print(f"  hunt: {hunted['status']}")
        print(f"        folded by: {hunted['folders']}")
        print(f"        rows={hunted['rows']} track={hunted['track']} frames={hunted['frames']}"
              f" progress={hunted['progress']}"
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
            if hunted["track"] < 4:
                bad.append(f"the sequence track has {hunted['track']} lines, expected one per cycle")
            if not hunted["viewerShown"]:
                bad.append("the viewer stayed hidden through a finished hunt")
            if hunted["frames"] < 4:
                bad.append(f"the viewer holds {hunted['frames']} frames, expected one per cycle")
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
        # 🔴 ON RFD4, A GIVEN SEQUENCE MUST BE REFUSED FOR THE RIGHT REASON.
        # The spans left over from the hunt describe a target by its SEQUENCE,
        # which is exactly what Hunter takes and exactly what RFD4 cannot hold
        # fixed - it conditions on coordinates. Refusing it with that reason is
        # the geometry axis working; a generation that started anyway would be
        # conditioning on nothing.
        cdp.evaluate(ws, """(() => {
          const pick = document.getElementById('method');
          pick.value = 'rfd4p';
          pick.dispatchEvent(new Event('change'));
          document.getElementById('hunt').click();
          return 1;
        })()""")
        time.sleep(0.3)
        onRfd4p = cdp.evaluate(ws, "document.getElementById('status-message').textContent")
        print(f"  Generate on a sequence span: {onRfd4p}")
        if "coordinates" not in onRfd4p:
            bad.append(f"RFD4 did not refuse a sequence span for the right reason: {onRfd4p!r}")

        # ...and a contig it CAN run is accepted: loading a preset fills the
        # spans, picks RFD4, and leaves no problem to report.
        loaded = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById('preset');
          pick.value = 'motif_idx_1prw';
          pick.dispatchEvent(new Event('change'));
          return JSON.stringify({
            method: document.getElementById('method').value,
            contig: document.getElementById('contig-line').textContent.trim(),
            problems: document.getElementById('contig-problems').textContent.trim(),
            rows: document.querySelectorAll('#contig-rows > *').length,
            button: document.getElementById('hunt').textContent.trim(),
          });
        })()"""))
        print(f"  preset motif_idx_1prw: {loaded}")
        # 🔴 THE PRESET'S OWN CONTIG, REBUILT FROM THE ROWS IT FILLED. Loading a
        # preset into an editor is only honest if the editor can express it: if
        # the round trip lost a span the page would run a different job from the
        # one it names.
        if not loaded["contig"].startswith("5-20,A16-35,10-25,A52-71,5-20"):
            bad.append(f"the preset did not round-trip through the spans: {loaded['contig']!r}")
        if loaded["rows"] != 5:
            bad.append(f"the preset filled {loaded['rows']} rows, expected 5")
        if loaded["method"] != "rfd4p":
            bad.append("loading an RFD4 preset did not select RFD4")
        if loaded["problems"] != "":
            bad.append(f"the preset reported a problem: {loaded['problems']!r}")
        if loaded["button"] != "Generate":
            bad.append(f"the button says {loaded['button']!r} on RFD4, expected Generate")
    finally:
        proc.terminate()
        server.terminate()

    print("FAIL: " + "; ".join(bad) if bad else "design.html: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

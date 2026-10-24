"""Does design.html load, run its module, and say true things?

    python3 tools/check-design-page.py
    python3 tools/check-design-page.py --generate   # ...and a real RFD4-Proteina generation
    python3 tools/check-design-page.py --cases      # ...and a hunt per modality
    python3 tools/check-design-page.py --presets    # ...and a generation per RFD4 preset

🔴 `--cases` RUNS ONE HUNT PER MODALITY - unconditional, a ligand target, a
nucleic target - and asserts the AUTOMATIC DESIGNER follows the job: soluble for
protein alone, LigandMPNN where heteroatoms are present, NA-MPNN for a nucleic
chain. It is the matrix, and it takes a couple of minutes.

🔴 `--presets` GENERATES EVERY ONE OF RFD4-PROTEINA'S TEN PRESETS, which is the
only check that tells a preset that LOADS from a preset that RUNS - and seven of
eight conditioned ones were failing while the editor round-tripped them
perfectly. Several minutes, and it downloads the weights.

🔴 `--generate` IS OPT-IN BECAUSE IT DOWNLOADS ~420 MB. It runs an
unconditional 100-mer through RFD4-Proteina - the second method, the one the
default run only checks the refusals of - and a fresh Chrome profile has no
cache, so every run pays the weights again. About forty seconds on an M5 once
they are down. The default arm checks everything that does not need them.

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

# Every RFD4-Proteina preset, for `--presets`. The ids are the engine's own.
PRESET_TOKENS = ["uncond_short", "ppi_pdl1", "motif_idx_1prw", "motif_unidx_1prw", "sm_sam",
                 "dna_7m5w", "yaml_5o45_binder", "partial_diffusion_1prw",
                 "inverse_folding_1prw", "custom_cath_monomer"]

# 🔴 ONE HUNT PER MODALITY, with the designer each must choose. The rule is the
# reference's and lives in cpu/design/designers.js; what is checked here is that
# the SPANS carry enough for it to fire.
MOTIF = "GWSTELEKHR"
FRAME_LEFT = "GWSTELEKHR"
FRAME_RIGHT = "EGITNVEIRIDN"

MODALITIES = [
    ("unconditional", [{"kind": "generated", "min": 24, "max": 24}], "SolubleMPNN", None),
    ("ligand target", [{"kind": "generated", "min": 24, "max": 24}, {"kind": "break"},
                       {"kind": "ligand", "code": "HEM"}], "LigandMPNN", None),
    ("nucleic target", [{"kind": "sequence", "value": "ATGCATGCATGC", "polymer": "dna"},
                        {"kind": "break"}, {"kind": "generated", "min": 20, "max": 20}],
     "NA-MPNN", None),
    # 🔴 THE THREE CASES HELD RESIDUES UNLOCK, and the fourth element is what
    # must SURVIVE in the designed sequence. That is the only assertion that
    # catches a hold which is not holding: the run looks identical either way,
    # and the first version of this returned the free residues alone - a
    # 22-residue scaffold came back as its 12 designed ones.
    ("motif scaffold", [{"kind": "generated", "min": 6, "max": 6},
                        {"kind": "sequence", "value": MOTIF},
                        {"kind": "generated", "min": 6, "max": 6}], "SolubleMPNN", MOTIF),
    ("inverse folding", [{"kind": "sequence", "value": "GWSTELEKHREELKEFLKKEGITNVEIRIDN",
                          "redesign": True}], "SolubleMPNN", None),
    # 🔴 A PTM TARGET: a CCD component swapped in at one position, which v1
    # could not express at all. The token count does not change.
    ("PTM target", [{"kind": "sequence", "value": "GWSTELEKHREELKEFLKKEGITNVEIRIDN",
                     "modifications": "4:SEP"},
                    {"kind": "break"}, {"kind": "generated", "min": 14, "max": 14}],
     "SolubleMPNN", None),
    ("antibody-style CDR", [{"kind": "sequence", "value": FRAME_LEFT},
                            {"kind": "sequence", "value": "EELKEFLKK", "redesign": True},
                            {"kind": "sequence", "value": FRAME_RIGHT}],
     "SolubleMPNN", FRAME_LEFT + "|" + FRAME_RIGHT),
]

# The target and binder a hunt runs on, as spans.
HUNT_SPANS = [{"kind": "sequence", "value": "GWSTELEKHREELKEFLKKEGITNVEIRIDN"},
              {"kind": "break"},
              {"kind": "generated", "min": 12, "max": 12}]

READ_PAGE = """(() => {
  const t = (id) => document.getElementById(id)?.textContent?.trim() ?? null;
  const n = (sel) => document.querySelectorAll(sel).length;
  return JSON.stringify({
    designer: t("designer-note"), folder: t("folder-note"), gpu: t("gpu-summary"),
    spanRows: n("#contig-rows > *"),
    // 🔴 THE LEGEND IS THE JOB'S NOW, so it is read as TEXT and compared with
    // the chains the span list actually has. Counting a fixed five was a gate
    // on a hard-coded list, and two of that five - tied, motif - were never
    // painted by anything.
    legend: [...document.querySelectorAll("#role-list li")].map((li) => li.textContent.trim()),
    // Every row that belongs to a chain carries that chain's bar; the swatch
    // and the bar have to be the SAME colour or the indicator points nowhere.
    rowColours: [...document.querySelectorAll("#contig-rows .contig-row")]
      .map((row) => row.style.getPropertyValue("--chain-colour") || null),
    legendColours: [...document.querySelectorAll("#role-list .role-swatch")]
      .map((s) => s.style.getPropertyValue("--chain-colour")),
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
        if "because" not in (page["designer"] or ""):
            bad.append(f"the designer note does not say why: {page['designer']!r}")
        # The default job is a given target, a break, and a designed binder - so
        # two chains, and the designed one has to be named as such or the reader
        # cannot tell which colour is the thing being made.
        if len(page["legend"]) != 2:
            bad.append(f"the legend has {page['legend']}, expected one entry a chain")
        if not any("designed" in entry for entry in page["legend"]):
            bad.append(f"the legend does not say which chain is designed: {page['legend']}")
        # 🔴 THE SWATCH AND THE ROW BAR MUST BE THE SAME COLOUR. They come from
        # two different walks of the same spans, and a disagreement between them
        # is an indicator that points at the wrong chain - which reads as an
        # answer rather than as a bug.
        rowsByChain = [c for c in page["rowColours"] if c]
        if sorted(set(rowsByChain)) != sorted(set(page["legendColours"])):
            bad.append("row bars and legend swatches disagree: "
                       f"{sorted(set(rowsByChain))} against {sorted(set(page['legendColours']))}")
        # ...and the break row belongs to no chain, so it carries no bar.
        if len(rowsByChain) != 2:
            bad.append(f"expected two chained rows, got {page['rowColours']}")
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
          // 🔴 NOT designerGroup: it is hidden now, since the automatic rule is
          // right in every measured case and a picker whose answer is already
          // correct is a question nobody needs asked. Its NOTE is still checked
          // above - the answer stays even though the control went.
          const groups = ["folderGroup", "cyclesGroup", "runsGroup"];
          return JSON.stringify({
            // 🔴 THE METHOD'S DESCRIPTION IS THE RESTING STATUS LINE NOW, as
            // index.html does with "Ready. Paste a sequence and press Fold." -
            // one fewer block of prose in the panel, and the one place a reader
            // already looks for what the page is about to do.
            note: document.getElementById("status-message")?.textContent?.trim() ?? "",
            disabled: groups.filter((id) => {
              const g = document.getElementById(id);
              return g !== null && [...g.querySelectorAll("select, input")].every((c) => c.disabled);
            }),
            // 🔴 ONE LINE NAMING ALL OF THEM, NOT A SENTENCE PER GROUP. Six
            // italic reasons threaded between the controls is the ragged-
            // paragraph problem the notes column exists to fix, and the first
            // version of this put it straight back. The greyed control says
            // THAT it cannot be used; this line says which and why.
            unused: document.getElementById("unused-note")?.textContent ?? "",
            // 🔴 STILL IN THE PAGE, NOT NECESSARILY ON SCREEN - and `hidden`
            // cannot tell the two apart. `runsGroup` lives in the advanced
            // strip, which is COLLAPSED by default, so it is legitimately
            // hidden and `offsetParent` is null for a control that is present
            // and correct. What must never happen is a control being REMOVED on
            // a method switch: a disabled control explains itself, a vanished
            // one does not, and taking a control away does not change its value
            // either. So this asks whether the element is in the document, and
            // reports separately whether it is hidden for a reason other than
            // the strip being shut.
            stillThere: groups.filter((id) => document.getElementById(id) !== null),
            wronglyHidden: groups.filter((id) => {
              const g = document.getElementById(id);
              return g !== null && g.hasAttribute("hidden") && !g.classList.contains("advanced");
            }),
          });
        })()"""))
        print(f"  on RFD4-Proteina: {switched}")
        for key in ("disabled", "stillThere"):
            if len(switched[key]) != 3:
                bad.append(f"switching method: {key} covered {switched[key]}, expected all three groups")
        # The one line has to NAME every greyed control, or a reader is left with
        # three dead inputs and no account of why.
        if switched["wronglyHidden"]:
            bad.append("a method switch hid a control that is not advanced: "
                       f"{switched['wronglyHidden']}")
        missing = [word for word in ("folds with", "cycles", "runs")
                   if word not in switched["unused"].lower()]
        if missing:
            bad.append(f"the unused-control line does not name {missing}: {switched['unused']!r}")
        if "one pass" not in switched["note"] or "🔴" in switched["note"]:
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
          // The select is still the state; the chips mirror it.
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
            # 🔴 ONE FOLDER, NAMED ON EVERY ROW. This used to assert an
            # ALTERNATION - af3, opendde, af3, opendde - from when the page
            # offered two folders and rotated them per cycle. There is one
            # folder select now, so the column must read that folder every time;
            # an empty cell or a second name means the row is reporting
            # something other than the model that actually folded it.
            folded = set(hunted["folders"][:4])
            if folded != {"opendde"}:
                bad.append(f"the folder column is not the one folder chosen: {hunted['folders']}")
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

        # 🔴 A STRUCTURE SEGMENT WITHOUT A STRUCTURE MUST BE REFUSED, on both
        # methods and for different reasons - Hunter needs the sequence to fold,
        # RFD4 needs the coordinates to condition on. A run that started anyway
        # would design against nothing.
        for wanted_method, phrase in (("hunter", "needs the structure loaded"),
                                      ("rfd4p", "names a segment of an existing structure")):
            cdp.evaluate(ws, """(() => {
              const pick = document.getElementById('method');
              pick.value = '%s';
              pick.dispatchEvent(new Event('change'));
              document.getElementById('preset').value = '';
              document.getElementById('preset').dispatchEvent(new Event('change'));
              window.__designSpans.set([{ kind: 'segment', chain: 'A', first: 1, last: 40 },
                { kind: 'break' }, { kind: 'generated', min: 30, max: 30 }]);
              document.getElementById('hunt').click();
              return 1;
            })()""" % wanted_method)
            # 🔴 THE REFUSAL IS ASYNCHRONOUS ON RFD4 - it happens inside
            # generate(), after the click returns - so reading the status in the
            # same expression caught "starting" and called it a pass.
            time.sleep(1.5)
            refused = cdp.evaluate(ws, """document.getElementById('status-message').textContent
              + ' || ' + document.getElementById('contig-problems').textContent""")
            print(f"  segment with no structure, {wanted_method}: {refused[:120]}")
            if phrase not in refused:
                bad.append(f"{wanted_method} did not refuse a segment with no structure: {refused!r}")

        # 🔴 A LENGTH RANGE UNDER PROTEIN HUNTER. Four of the ten shipped
        # examples were refused for this reason alone - "Hunter folds one length
        # at a time" - which is true of a single fold and describes a search the
        # page already has: `Runs` repeats the hunt, so each run draws its own
        # length out of the range. What must be visible here is that the job is
        # ACCEPTED, and that the line reports a RANGE rather than the minimum -
        # the minimum is one draw out of sixteen, not what the job says.
        ranged = json.loads(cdp.evaluate(ws, """(() => {
          // 🔴 A RANGED JOB BUILT HERE, NOT A PRESET. A motif preset also
          // carries a SECOND and unrelated problem - its contig is written in
          // the original PDB entry's numbering while the shipped example CIF is
          // a cropped copy - so driving one here would test the range and the
          // numbering together and report whichever failed first.
          document.getElementById('method').value = 'hunter';
          document.getElementById('method').dispatchEvent(new Event('change'));
          window.__designSpans.set([
            { kind: 'sequence', value: 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKL' },
            { kind: 'break' },
            { kind: 'generated', min: 5, max: 20 },
          ]);
          return JSON.stringify({
            contig: document.getElementById('contig-line').textContent.trim(),
            problems: document.getElementById('contig-problems').textContent.trim(),
          });
        })()"""))
        # 🔴 THE PRESET'S STRUCTURE IS FETCHED, SO THIS HAS TO WAIT FOR IT.
        # Hunter reads a `From structure` span's residues out of the loaded
        # file, and reading the moment after the change event fires catches the
        # page mid-fetch and reports a refusal that clears a second later.
        for _ in range(20):
            if not ranged["problems"]:
                break
            time.sleep(1)
            ranged = json.loads(cdp.evaluate(ws, "JSON.stringify({"
                "contig: document.getElementById('contig-line').textContent.trim(),"
                "problems: document.getElementById('contig-problems').textContent.trim(),"
                "note: document.getElementById('structure-note').textContent.trim()})"))
        print(f"  ranged under hunter: {ranged}")
        if ranged["problems"]:
            bad.append("a length range is refused under Protein Hunter: "
                       f"{ranged['problems']!r}")
        if "design 5\u201320" not in ranged["contig"]:
            bad.append(f"the range is not reported as 5-20: {ranged['contig']!r}")

        # 🔴 AND, WHEN ASKED, A REAL GENERATION: the second method end to end.
        if "--generate" in sys.argv:
            cdp.evaluate(ws, """(() => {
              document.getElementById('preset').value = '';
              document.getElementById('preset').dispatchEvent(new Event('change'));
              document.getElementById('method').value = 'rfd4p';
              document.getElementById('method').dispatchEvent(new Event('change'));
              // the simplest job there is: one generated span, nothing given
              window.__designSpans.set([{ kind: 'generated', min: 100, max: 100 }]);
              document.getElementById('steps').value = '30';
              // 🔴 A BEAM OF TWO, because one exercises none of the loop. The
              // candidate loop, the per-candidate seed, the ranking and the
              // single-best marker all only exist above one, and a beam of 1
              // runs the same straight line the page ran before the control was
              // honoured at all.
              document.getElementById('beam').value = '2';
              document.getElementById('hunt').click();
              return 1;
            })()""")
            genDeadline = time.time() + 2400
            generated = None
            while time.time() < genDeadline:
                generated = json.loads(cdp.evaluate(ws, """JSON.stringify({
                  status: document.getElementById('status-message').textContent,
                  failed: document.getElementById('status-message').classList.contains('error'),
                  rows: document.querySelectorAll('#results-body tr').length,
                  best: document.querySelectorAll('#results-body tr.best').length,
                  // The validity cell carries each candidate's own validation
                  // sentence, which is the only per-candidate number the table
                  // shows.
                  verdicts: [...document.querySelectorAll('#results-body tr')]
                    .map((tr) => tr.lastElementChild?.textContent ?? ''),
                  progress: document.getElementById('progress').value,
                  frames: (() => {
                    const entry = Object.values(window.py2dmol_viewers ?? {})[0];
                    const renderer = entry?.renderer ?? entry;
                    return renderer?.objectsData?.[renderer?.currentObjectName]?.frames?.length ?? 0;
                  })(),
                })"""))
                # 🔴 NOT "a row appeared": the first candidate draws its row
                # immediately, so that would stop the clock while the second is
                # still generating and read a half-finished beam as the result.
                if generated["failed"] or "done" in generated["status"]:
                    break
                time.sleep(3)
            print(f"  generate: {generated['status']}")
            print(f"            rows={generated['rows']} best={generated['best']}"
                  f" frames={generated['frames']} progress={generated['progress']}")
            if generated["failed"]:
                bad.append(f"the generation failed: {generated['status']}")
            else:
                if "100 tokens" not in generated["status"]:
                    bad.append(f"the generation did not report its tokens: {generated['status']!r}")
                # 🔴 A GENERATION IS A TRAJECTORY AND MUST BE DRAWN AS ONE.
                # This asked for one frame, which a page that pushes only the
                # finished structure satisfies - and that is exactly what it
                # did, so RFD4 looked static beside a hunt that animates while
                # the gate stayed green. The flow decodes a bounded number of
                # live frames now, so several is the right answer; the play bar
                # hides itself below two.
                if generated["frames"] < 3:
                    bad.append(f"the generation drew {generated['frames']} frames,"
                               " expected a trajectory")
                if generated["frames"] < 1:
                    bad.append("the generation drew no structure")
                if generated["progress"] != 1:
                    bad.append(f"the bar ended at {generated['progress']}, not 1")
                # Both candidates are drawn, and exactly one of them is marked -
                # the same rule the hunt table has, because the leader changes
                # as candidates arrive.
                if generated["rows"] != 2:
                    bad.append(f"a beam of 2 drew {generated['rows']} rows, expected 2")
                if generated["best"] != 1:
                    bad.append(f"{generated['best']} rows marked best, expected exactly 1")
                if "best of 2" not in generated["status"]:
                    bad.append("the status does not say the beam was ranked: "
                               f"{generated['status']!r}")
                # 🔴 AND THE TWO CANDIDATES MUST ACTUALLY DIFFER. The flow is
                # deterministic given its seed, so if the per-candidate seed did
                # not reach the engine the beam would draw ONE structure twice,
                # rank it against itself, and satisfy every assertion above - N
                # times the cost for one answer, silently. Two identical
                # verdicts is what that looks like from here.
                if len(set(generated["verdicts"])) < 2:
                    bad.append("the beam's candidates are identical, so the seed is not"
                               f" reaching the engine: {generated['verdicts']}")

        # 🔴 THE CONTACT VETO, PROVED BY ASKING FOR AN EPITOPE THAT CANNOT BE
        # TOUCHED. A filter that never fires is indistinguishable from no filter,
        # and this is the only check that tells them apart: residue 9999 is not
        # in any prediction, so every cycle must be rejected, the reason must
        # carry its measurement, and the summary must name THAT reason rather
        # than blaming the alanine ceiling - which it did, until a second veto
        # existed to catch it out.
        if "--cases" in sys.argv:
            cdp.evaluate(ws, """(() => {
              document.getElementById('method').value = 'hunter';
              document.getElementById('method').dispatchEvent(new Event('change'));
              document.getElementById('cycles').value = '1';
              window.__designHotspots = { A: [9999] };
              window.__designSpans.set([
                { kind: 'sequence', value: 'GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE' },
                { kind: 'break' }, { kind: 'generated', min: 16, max: 16 }]);
              document.getElementById('hunt').click();
              return 1;
            })()""")
            vetoDeadline = time.time() + 600
            veto = None
            while time.time() < vetoDeadline:
                veto = json.loads(cdp.evaluate(ws, """JSON.stringify({
                  status: document.getElementById('status-message').textContent,
                  rows: document.querySelectorAll('#results-body tr').length,
                  best: document.querySelectorAll('#results-body tr.best').length,
                  // The validity cell carries each candidate's own validation
                  // sentence, which is the only per-candidate number the table
                  // shows.
                  verdicts: [...document.querySelectorAll('#results-body tr')]
                    .map((tr) => tr.lastElementChild?.textContent ?? ''),
                  rejected: document.querySelectorAll('#results-body tr.rejected').length,
                  validity: [...document.querySelectorAll('#results-body td.validity')]
                    .map((n) => n.textContent).join(' | '),
                  button: document.getElementById('hunt').textContent.trim(),
                })"""))
                if veto["button"] == "Hunt" and veto["rows"] > 0:
                    break
                time.sleep(2)
            print(f"  contact veto: rejected={veto['rejected']} | {veto['status'][:70]}")
            print(f"                {veto['validity'][:110]}")
            if veto["rejected"] < 1:
                bad.append("an impossible epitope rejected no cycle; the contact filter never fired")
            if "hotspots" not in veto["validity"]:
                bad.append(f"the veto's reason is missing from the table: {veto['validity']!r}")
            if "needs 8 A" not in veto["validity"]:
                bad.append(f"the veto did not carry its measurement: {veto['validity']!r}")
            if "hotspots" not in veto["status"]:
                bad.append(f"the summary blamed the wrong thing: {veto['status']!r}")
            cdp.evaluate(ws, "delete window.__designHotspots")

        # 🔴 AND, WHEN ASKED, ONE HUNT PER MODALITY. Each asserts the automatic
        # designer followed the job, which is the thing most likely to rot: the
        # rule lives in cpu/design/designers.js and the spans have to carry
        # enough for it to fire.
        if "--cases" in sys.argv:
            cdp.evaluate(ws, """(() => {
              for (const o of document.getElementById('folders').options) {
                o.selected = o.value === 'af3';
              }
              document.getElementById('folders').dispatchEvent(new Event('change'));
              document.getElementById('cycles').value = '1';
              document.getElementById('preset').value = '';
              document.getElementById('preset').dispatchEvent(new Event('change'));
              return 1;
            })()""")
            for label, spans, designer, survives in MODALITIES:
                cdp.evaluate(ws, """(() => {
                  document.getElementById('method').value = 'hunter';
                  document.getElementById('method').dispatchEvent(new Event('change'));
                  window.__designSpans.set(%s);
                  document.getElementById('hunt').click();
                  return 1;
                })()""" % json.dumps(spans))
                caseDeadline = time.time() + 600
                case = None
                while time.time() < caseDeadline:
                    case = json.loads(cdp.evaluate(ws, """JSON.stringify({
                      status: document.getElementById('status-message').textContent,
                      failed: document.getElementById('status-message').classList.contains('error'),
                      rows: document.querySelectorAll('#results-body tr').length,
                  best: document.querySelectorAll('#results-body tr.best').length,
                  // The validity cell carries each candidate's own validation
                  // sentence, which is the only per-candidate number the table
                  // shows.
                  verdicts: [...document.querySelectorAll('#results-body tr')]
                    .map((tr) => tr.lastElementChild?.textContent ?? ''),
                      designer: document.getElementById('designer-note').textContent,
                      button: document.getElementById('hunt').textContent.trim(),
                      last: [...document.querySelectorAll('#cycle-track > *')]
                        .map((n) => n.textContent.trim().split(/\s+/)[1] ?? '').pop() ?? '',
                    })"""))
                    if case["failed"] or (case["button"] == "Hunt" and case["rows"] > 0):
                        break
                    time.sleep(2)
                print(f"  {label}: rows={case['rows']} | {case['status'][:60]}")
                print(f"        {case['designer'][:90]}")
                if case["failed"]:
                    bad.append(f"{label} failed: {case['status']}")
                elif case["rows"] < 2:
                    bad.append(f"{label} drew {case['rows']} rows, expected cycle 0 and 1")
                if designer not in case["designer"]:
                    bad.append(f"{label} chose the wrong designer: {case['designer']!r}")
                if survives is not None and not case["failed"]:
                    print(f"        designed: {case['last']}")
                    for piece in survives.split("|"):
                        if piece not in case["last"]:
                            bad.append(f"{label} did not hold {piece!r}:"
                                       f" the designed sequence is {case['last']!r}")

        # 🔴 EVERY RFD4 PRESET, GENERATED. A preset that loads is not a preset
        # that runs: the editor round-tripped all ten while seven of the eight
        # conditioned ones failed in the pipeline, because a preset without
        # `specMode` is driven by its structure's ANNOTATIONS and its contig is
        # written in the original entry's numbering. Only a generation says so.
        if "--presets" in sys.argv:
            for preset in PRESET_TOKENS:
                cdp.evaluate(ws, """(() => {
                  const pick = document.getElementById('preset');
                  pick.value = '%s';
                  pick.dispatchEvent(new Event('change'));
                  document.getElementById('steps').value = '20';
                  document.getElementById('results-body').replaceChildren();
                  document.getElementById('hunt').click();
                  return 1;
                })()""" % preset)
                presetDeadline = time.time() + 2400
                ran = None
                while time.time() < presetDeadline:
                    ran = json.loads(cdp.evaluate(ws, """JSON.stringify({
                      status: document.getElementById('status-message').textContent,
                      failed: document.getElementById('status-message')
                        .classList.contains('error'),
                      rows: document.querySelectorAll('#results-body tr').length,
                  best: document.querySelectorAll('#results-body tr.best').length,
                  // The validity cell carries each candidate's own validation
                  // sentence, which is the only per-candidate number the table
                  // shows.
                  verdicts: [...document.querySelectorAll('#results-body tr')]
                    .map((tr) => tr.lastElementChild?.textContent ?? ''),
                      button: document.getElementById('hunt').textContent.trim(),
                    })"""))
                    if ran["failed"] or (ran["button"] == "Generate" and ran["rows"] > 0):
                        break
                    time.sleep(3)
                print(f"  {preset}: {ran['status'][:72]}")
                if ran["failed"]:
                    bad.append(f"preset {preset} failed to generate: {ran['status']}")
                elif "tokens" not in ran["status"]:
                    bad.append(f"preset {preset} did not report its tokens: {ran['status']!r}")

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
          // 🔴 ITS OWN SPANS, so this check does not depend on what an earlier
          // one left behind - which it did, and read the wrong refusal.
          window.__designSpans.set([{ kind: 'sequence', value: 'GWSTELEKHREELKEFLKK' },
            { kind: 'break' }, { kind: 'generated', min: 30, max: 30 }]);
          document.getElementById('hunt').click();
          return 1;
        })()""")
        time.sleep(1.5)
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

        # 🔴 AND THE FOLD-CONDITIONED PRESET, which the editor could not express
        # at all until labelled spans existed - it used to refuse to load it
        # rather than lose its labels. Its six named spans drive C_SEC, so the
        # round trip has to rebuild the contig AND show what each asks for.
        cath = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById('preset');
          pick.value = 'custom_cath_monomer';
          pick.dispatchEvent(new Event('change'));
          return JSON.stringify({
            contig: document.getElementById('contig-line').textContent.trim(),
            problems: document.getElementById('contig-problems').textContent.trim(),
            note: document.getElementById('preset-note').textContent.trim(),
            rows: document.querySelectorAll('#contig-rows > *').length,
            // 🔴 READ FROM THE SPANS, NOT THE DOM. A span's secondary structure
            // lives behind its row's disclosure now, so the control is not in
            // the document until that row is opened - and a DOM query finds
            // nothing while the job is perfectly correct.
            structures: window.__designSpans.read().map((span) => span.secondary ?? '')
              .filter((value) => value !== ''),
          });
        })()"""))
        print(f"  preset custom_cath_monomer: rows={cath['rows']}"
              f" structures={cath['structures']}")
        print(f"        {cath['contig'][:80]}")
        if not cath["contig"].startswith("h1:16-16P,l1:4-4P,s1:8-8P,l2:4-4P,h2:14-14P,s2:8-8P"):
            bad.append(f"the labelled preset did not round-trip: {cath['contig']!r}")
        if cath["rows"] != 6:
            bad.append(f"the labelled preset filled {cath['rows']} rows, expected 6")
        if cath["structures"] != ["HELIX", "LOOP", "STRAND", "LOOP", "HELIX", "STRAND"]:
            bad.append(f"the secondary structures were not restored: {cath['structures']}")
        if cath["problems"] != "":
            bad.append(f"the labelled preset reported a problem: {cath['problems']!r}")
    finally:
        proc.terminate()
        server.terminate()

    print("FAIL: " + "; ".join(bad) if bad else "design.html: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

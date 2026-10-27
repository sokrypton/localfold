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
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

# The page's default job until 2026-10-27 - a target sequence and a binder - which the Hunter arm
# still describes, now set by hand since the page opens on a blank RFD4-Proteina design.
OLD_DEFAULT = [{"kind": "sequence",
                "value": "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"},
               {"kind": "break"},
               {"kind": "generated", "min": 75, "max": 75}]

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
    // 🔴 THE EXAMPLES ARE THE METHOD'S. They are RFD4-Proteina's and
    // Protein Hunter can run six of the ten, so under Hunter the other four
    // are listed DISABLED with their reason rather than dropped - a reader
    // who came for the ligand-pocket example needs to know it is the method
    // and not the page. Offering one the method then refuses is the same
    // fault as a control that does nothing.
    presetsOffered: [...document.querySelectorAll("#preset option")]
      .filter((o) => !o.disabled && o.value).length,
    presetsRefused: [...document.querySelectorAll("#preset option")]
      .filter((o) => o.disabled).map((o) => o.textContent),
    goalsDisabled: [...document.querySelectorAll("#goals .goal-card:disabled")].map((c) => c.dataset.goal),
    goalReasons: [...document.querySelectorAll("#goals .goal-card:disabled")].map((c) => c.title.length > 0),
    constraintsHidden: document.getElementById("step-constraints").hidden,
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
    profile = f"/tmp/localfold-design-{os.getpid()}"
    proc, ws = cdp.launch(free_port(), profile)
    bad = []
    try:
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/design.html")
        # 🔴 THE MODULE RAN IFF IT DREW THE GOAL CARDS. There is no window marker to wait on, and
        # `readyState === "complete"` is true of a page whose module threw.
        try:
            cdp.wait_for(ws, "document.querySelectorAll('#goals .goal-card').length > 0",
                         60, "web/design.js to run")
        except RuntimeError:
            threw = cdp.evaluate(ws, """(async () => {
              try { await import("/web/design.js"); return "imported cleanly the second time"; }
              catch (e) { return String((e && (e.stack || e.message)) || e); }
            })()""")
            print(f"FAIL: web/design.js did not run:\n{threw}")
            return 1

        # The GPU line is filled when `requestAdapter()` answers, which is after the module ran.
        try:
            cdp.wait_for(ws, "document.getElementById('gpu-summary')?.textContent.length > 0", 30,
                         "the WebGPU line")
        except RuntimeError:
            pass

        # 🔴 WHAT A NEWCOMER SEES FIRST (2026-10-27): the goals, a blank 100-residue design under
        # RFD4-Proteina, the job drawn as a chain, and no prose - no greyed controls of the other
        # method, no note explaining them, no designer note. Each was a reason the old page read
        # as a form for someone who already knew it.
        first = json.loads(cdp.evaluate(ws, """(() => {
          const visible = (el) => el !== null && el.offsetParent !== null;
          return JSON.stringify({
            goals: [...document.querySelectorAll('#goals .goal-card')].map((c) => c.dataset.goal),
            chosen: document.querySelector('#goals [aria-checked="true"]')?.dataset.goal ?? null,
            method: document.getElementById('method').value,
            rows: document.querySelectorAll('#contig-rows > *').length,
            blocks: [...document.querySelectorAll('#chain-diagram .chain-block')].map((b) => b.textContent),
            contig: document.getElementById('contig-line').textContent,
            columns: document.querySelectorAll('#results-table thead th').length,
            otherMethodShown: ['folderGroup', 'cyclesGroup', 'runsGroup', 'temperatureGroup', 'omitGroup']
              .filter((id) => !document.getElementById(id).classList.contains('not-this-method')),
            notes: [...document.querySelectorAll('.design-notes .option-note, #preset-note')]
              .filter(visible).map((n) => n.textContent.trim()).filter(Boolean),
            placeholder: visible(document.getElementById('viewer-empty')),
            drawn: window.__designDrawn(),
            constraints: visible(document.getElementById('step-constraints')),
            hunt: document.getElementById('hunt').textContent.trim(),
            huntInView: (() => { const b = document.getElementById('hunt').getBoundingClientRect();
              return b.top >= 0 && b.bottom <= innerHeight && b.width > 0; })(),
            gpu: document.getElementById('gpu-summary').textContent.trim(),
          });
        })()"""))
        print(f"  on arrival: {first}")
        if len(first["goals"]) != 8 or first["chosen"] != "blank":
            bad.append(f"the goals are {first['goals']} with {first['chosen']!r} chosen; expected 8, New protein")
        if first["method"] != "rfd4p" or first["hunt"] != "Generate":
            bad.append(f"the page opens on {first['method']!r} / {first['hunt']!r}, expected RFD4-Proteina and Generate")
        if first["rows"] != 1 or first["blocks"] != ["100"] or first["contig"] != "100 residues":
            bad.append(f"the blank job is {first['rows']} rows, blocks {first['blocks']}, {first['contig']!r};"
                       " expected one designed 100-residue chain")
        if first["columns"] != 4:
            bad.append(f"{first['columns']} result columns under RFD4-Proteina, expected 4")
        if first["otherMethodShown"]:
            bad.append(f"Protein Hunter's controls are on screen under RFD4-Proteina: {first['otherMethodShown']}")
        if first["notes"]:
            bad.append(f"the page explains itself before anything happened: {first['notes']}")
        # 🔴 THE JOB IS DRAWN FROM ARRIVAL (web/job-sketch.js): the blank job is one sketched chain of
        # 100 positions and no structure, so there is no empty placeholder to read - and Generate is
        # beside it, in view, rather than at the end of step 5.
        if first["placeholder"] or first["drawn"] is None or first["drawn"]["structure"] is not None \
                or len(first["drawn"]["ghosts"]) != 1 or first["drawn"]["positions"] != 100:
            bad.append(f"the blank job is not drawn as one 100-residue sketch on arrival: {first['drawn']},"
                       f" placeholder {first['placeholder']}")
        if not first["constraints"]:
            bad.append("the constraints step is missing on arrival")
        if not first["huntInView"]:
            bad.append("Generate is not in view on arrival")
        if first["gpu"] == "":
            bad.append("the page did not say whether WebGPU is available")

        # 🔴 A GOAL CARD LOADS ITS EXAMPLE, AND NEW PROTEIN TAKES IT AWAY AGAIN: the motif card picks
        # the 1PRW example, which draws its structure in the viewer, lays out five pieces and names
        # the file without its path; editing that example's pieces says they are not used (it runs
        # from its structure's annotations); and New protein drops the structure and the pieces.
        cdp.evaluate(ws, "(() => { document.querySelector('[data-goal=motif_idx_1prw]').click(); return 1; })()")
        # 🔴 ON THE STRUCTURE AND ITS PIECES, not on "a viewer exists": the blank job's sketch is one
        # already, so that returned before the example had loaded anything.
        cdp.wait_for(ws, "(() => { const d = window.__designDrawn?.(); return d !== null && d.structure !== null"
                         " && d.ghosts.length === 3; })()", 30, "the motif's structure and its three sketched pieces")
        # 🔴 A CLICK ON A SKETCHED PIECE OPENS ITS ROW in step 3 - where its length is changed - and
        # selects nothing: a sketch is no residue a constraint can name. A real click, as below.
        time.sleep(0.5)
        ghost = json.loads(cdp.evaluate(ws, """(() => {
          const entry = Object.values(window.py2dmol_viewers ?? {}).at(-1);
          const r = entry?.renderer ?? entry;
          const canvas = document.querySelector('#design-viewer canvas');
          canvas.scrollIntoView({ block: 'center' });
          const box = canvas.getBoundingClientRect();
          const sketched = new Set(window.__designDrawn().ghosts);
          const steady = (x, y, hit) => [[-2, 0], [2, 0], [0, -2], [0, 2]]
            .every(([dx, dy]) => r.pickResidueAt(x + dx, y + dy) === hit);
          for (let y = Math.ceil(box.top + 20); y < box.bottom - 30; y += 3)
            for (let x = Math.ceil(box.left + 20); x < box.right - 20; x += 3) {
              const hit = r.pickResidueAt(x, y);
              if (hit >= 0 && sketched.has(r.chains[hit]) && steady(x, y, hit))
                return JSON.stringify({ x, y, opened: document.querySelectorAll('#contig-rows .contig-more').length });
            }
          return 'null';
        })()"""))
        if ghost is None:
            bad.append("no sketched piece of the motif job could be found under the pointer")
        else:
            for kind in ("mousePressed", "mouseReleased"):
                ws.call("Input.dispatchMouseEvent", type=kind, x=ghost["x"], y=ghost["y"], button="left", clickCount=1)
                time.sleep(0.05)
            time.sleep(0.4)
            after = json.loads(cdp.evaluate(ws, """JSON.stringify({
              opened: document.querySelectorAll('#contig-rows .contig-more').length,
              selected: (() => { const e = Object.values(window.py2dmol_viewers ?? {}).at(-1);
                return (e?.renderer ?? e).residueSelection?.size ?? 0; })() })"""))
            print(f"  sketch click: {ghost} -> {after}")
            if after["opened"] <= ghost["opened"] or after["selected"] != 0:
                bad.append(f"a click on a sketched piece did not open its row and select nothing: {after}")
        motif = json.loads(cdp.evaluate(ws, """(() => {
          const out = { preset: document.getElementById('preset').value,
            chosen: document.querySelector('#goals [aria-checked="true"]')?.dataset.goal,
            blocks: [...document.querySelectorAll('#chain-diagram .chain-block')].map((b) => b.textContent),
            structure: document.getElementById('structure-note').textContent,
            placeholder: !document.getElementById('viewer-empty').hidden,
            drawn: window.__designDrawn(),
            noteBefore: document.getElementById('preset-note').textContent };
          const spans = window.__designSpans.read();
          spans[0] = { ...spans[0], min: 7, max: 7 };
          window.__designSpans.set(spans);
          const note = document.getElementById('preset-note');
          out.edited = { text: note.textContent, warn: note.classList.contains('warn') };
          document.querySelector('[data-goal=blank]').click();
          out.blank = { rows: document.querySelectorAll('#contig-rows > *').length,
            structure: document.getElementById('structure-note').textContent,
            placeholder: !document.getElementById('viewer-empty').hidden,
            preset: document.getElementById('preset').value };
          return JSON.stringify(out);
        })()"""))
        print(f"  motif card: {motif}")
        if motif["preset"] != "motif_idx_1prw" or motif["chosen"] != "motif_idx_1prw":
            bad.append(f"the motif card did not load its example: {motif}")
        if motif["blocks"] != ["5–20", "A1–20", "10–25", "A22–41", "5–20"]:
            bad.append(f"the motif example is drawn as {motif['blocks']}")
        if motif["structure"] != "motif_idx_1prw · chain A 0–42" or motif["placeholder"]:
            bad.append(f"the motif's structure is not shown and named: {motif['structure']!r}")
        if not motif["edited"]["warn"] or "not used" not in motif["edited"]["text"]:
            bad.append(f"an edit to an annotation-driven example is not said to be unused: {motif['edited']}")
        if motif["blank"] != {"rows": 1, "structure": "", "placeholder": False, "preset": ""}:
            bad.append(f"New protein did not clear the example: {motif['blank']}")

        # 🔴 AND HOTSPOTS ARE NAMED WHEREVER THEY WERE SET. The binder example carries its epitope in
        # its FILE (B40, B99, B107, in the crop's numbering) and runs from that file, so its pieces
        # are the file's - a designed chain A of 50-120 against chain B 1-115 - and B's block counts
        # the hotspots on it. The page used to draw the example's contig (keep A18-132, design 76),
        # a job that never runs, and to show its hotspots nowhere.
        cdp.evaluate(ws, "(() => { document.querySelector('[data-goal=ppi_pdl1]').click(); return 1; })()")
        try:
            cdp.wait_for(ws, "document.querySelector('#chain-diagram .chain-block-hot') !== null", 30,
                         "the binder's hotspots")
        except RuntimeError:
            pass   # reported below, from what IS drawn
        binder = json.loads(cdp.evaluate(ws, """(() => JSON.stringify({
          blocks: [...document.querySelectorAll('#chain-diagram .chain-block')].map((b) => b.textContent),
          line: document.getElementById('hotspot-line').textContent,
        }))()"""))
        print(f"  binder card: {binder}")
        if binder["blocks"] != ["50\u2013120", "B1\u2013115 \u00b7 3 hotspots"]:
            bad.append(f"the binder example is drawn as {binder['blocks']}, not as its file runs")
        if "B40, B99, B107" not in binder["line"] or "should touch" not in binder["line"]:
            bad.append(f"the binder's hotspots are not named and defined: {binder['line']!r}")
        cdp.evaluate(ws, "(() => { document.querySelector('[data-goal=blank]').click(); return 1; })()")

        # 🔴 PROTEIN HUNTER IS AN ADVANCED CHOICE NOW, and under it the page is the one it was: its own
        # controls, its notes, its examples - and the old default job, set here, is described to it.
        cdp.evaluate(ws, """(() => {
          const pick = document.getElementById('method');
          pick.value = 'hunter';
          pick.dispatchEvent(new Event('change'));
          window.__designSpans.set(__OLD_DEFAULT__);
          return 1;
        })()""".replace("__OLD_DEFAULT__", json.dumps(OLD_DEFAULT)))
        page = json.loads(cdp.evaluate(ws, READ_PAGE))
        for key, value in page.items():
            print(f"  {key}: {value}")
        if "because" not in (page["designer"] or ""):
            bad.append(f"the designer note does not say why: {page['designer']!r}")
        # Two chains, and the designed one named as such.
        if len(page["legend"]) != 2:
            bad.append(f"the legend has {page['legend']}, expected one entry a chain")
        if not any("designed" in entry for entry in page["legend"]):
            bad.append(f"the legend does not say which chain is designed: {page['legend']}")
        # 🔴 THE SWATCH AND THE ROW BAR MUST BE THE SAME COLOUR.
        rowsByChain = [c for c in page["rowColours"] if c]
        if sorted(set(rowsByChain)) != sorted(set(page["legendColours"])):
            bad.append("row bars and legend swatches disagree: "
                       f"{sorted(set(rowsByChain))} against {sorted(set(page['legendColours']))}")
        if len(rowsByChain) != 2:
            bad.append(f"expected two chained rows, got {page['rowColours']}")
        # Protein Hunter takes six of the ten examples, and four goals are disabled with a reason.
        if page["presetsOffered"] != 6:
            bad.append(f"{page['presetsOffered']} examples offered under Hunter, expected 6")
        if len(page["presetsRefused"]) != 4:
            bad.append(f"{len(page['presetsRefused'])} examples refused, expected 4")
        for text in page["presetsRefused"]:
            if "—" not in text:
                bad.append(f"an unavailable example gives no reason: {text!r}")
        if sorted(page["goalsDisabled"]) != ["custom_cath_monomer", "dna_7m5w", "ppi_pdl1", "sm_sam"] \
                or not all(page["goalReasons"]):
            bad.append(f"under Hunter the goals disabled are {page['goalsDisabled']}, each with its reason")
        if page["spanRows"] != 3:
            bad.append(f"the span editor drew {page['spanRows']} rows, expected 3")
        # 🔴 THE LINE DESCRIBES WHAT THE SELECTED METHOD WILL ACTUALLY BE ASKED: chains for Hunter.
        if "given 114" not in (page["contig"] or "") or "design 75" not in (page["contig"] or ""):
            bad.append(f"the line does not describe the job to Hunter: {page['contig']!r}")
        if page["presets"] < 11:
            bad.append(f"the preset picker lists {page['presets']} entries, expected 10 plus the blank")
        if page["columns"] != 5:
            bad.append(f"{page['columns']} result columns under Hunter, expected 5")
        if page["constraintsHidden"] is not True:
            bad.append("the constraints step is on screen under Protein Hunter, which takes none")

        # ...and that the automatic choice is wired to the target, not printed.
        cdp.evaluate(ws, f"""(() => {{ window.__designSpans.set({json.dumps(WITH_LIGAND)}); return 1; }})()""")
        time.sleep(0.3)
        withLigand = cdp.evaluate(ws, "document.getElementById('designer-note').textContent")
        print(f"  with a ligand: {withLigand}")
        if "LigandMPNN" not in withLigand:
            bad.append(f"a ligand did not move the automatic designer: {withLigand!r}")

        # ...and that switching method shows exactly the method's controls: the other's are hidden
        # (still in the page, so a driver and the job read them), and nothing explains the absence.
        switched = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById("method");
          pick.value = "rfd4p";
          pick.dispatchEvent(new Event("change"));
          const groups = ["folderGroup", "cyclesGroup", "runsGroup"];
          return JSON.stringify({
            note: document.getElementById("status-message")?.textContent?.trim() ?? "",
            hidden: groups.filter((id) => document.getElementById(id)?.classList.contains("not-this-method")),
            stillThere: groups.filter((id) => document.getElementById(id) !== null),
            unused: document.getElementById("unused-note")?.textContent ?? "",
          });
        })()"""))
        print(f"  on RFD4-Proteina: {switched}")
        for key in ("hidden", "stillThere"):
            if len(switched[key]) != 3:
                bad.append(f"switching method: {key} covered {switched[key]}, expected all three groups")
        if switched["unused"]:
            bad.append(f"the page explains a hidden control: {switched['unused']!r}")
        if "420 MB" not in switched["note"] or "🔴" in switched["note"]:
            # 🔴 AND THE REPO'S COMMENT MARKER MUST NOT REACH THE PAGE.
            bad.append(f"the resting line is wrong or carries a source marker: {switched['note']!r}")

        # 🔴 AND AtomWorks Studio's GENERATION OPTIONS: Thinking, the two K and Partial %, under
        # RFD4-Proteina, filled by a preset that declares them, and an out-of-range K refused by name
        # BEFORE a generation starts (no weights are touched by this arm).
        sampler = json.loads(cdp.evaluate(ws, """(() => {
          const ids = ["thinkingGroup", "backboneKGroup", "latentKGroup", "partialGroup"];
          const live = ids.filter((id) => !document.getElementById(id).classList.contains("not-this-method"));
          const preset = document.getElementById("preset");
          preset.value = "partial_diffusion_1prw";
          preset.dispatchEvent(new Event("change"));
          const filled = { thinking: document.getElementById("thinking").value,
                           partial: document.getElementById("partial").value };
          preset.value = "";
          preset.dispatchEvent(new Event("change"));
          document.getElementById("backbone-k").value = "25";
          document.getElementById("hunt").click();
          const refused = document.getElementById("status-message").textContent;
          document.getElementById("backbone-k").value = "16";
          return JSON.stringify({ live, filled, refused,
            more: document.getElementById("show-advanced").textContent.trim() });
        })()"""))
        print(f"  sampler controls: {sampler}")
        if len(sampler["live"]) != 4:
            bad.append(f"under RFD4-Proteina only {sampler['live']} of the sampler controls are shown")
        if sampler["filled"] != {"thinking": "low", "partial": "50"}:
            bad.append(f"the partial-diffusion preset did not fill Thinking/Partial %: {sampler['filled']}")
        if "Backbone K takes 1 to 20" not in sampler["refused"]:
            bad.append(f"Backbone K 25 was not refused by name: {sampler['refused']!r}")
        if sampler["more"] != "Advanced":
            bad.append(f"the advanced toggle reads {sampler['more']!r}, expected 'Advanced'")

        # ...and switching back shows Hunter's and hides the sampler's, or the page is a one-way door.
        back = json.loads(cdp.evaluate(ws, """(() => {
          const pick = document.getElementById("method");
          pick.value = "hunter";
          pick.dispatchEvent(new Event("change"));
          const shown = (id) => !document.getElementById(id).classList.contains("not-this-method");
          return JSON.stringify({ hunter: ["folderGroup", "cyclesGroup"].every(shown),
            samplerHidden: ["thinkingGroup", "backboneKGroup", "latentKGroup", "partialGroup"].every((id) => !shown(id)) });
        })()"""))
        if not back["hunter"] or not back["samplerHidden"]:
            bad.append(f"switching back to Protein Hunter did not swap the controls: {back}")

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
              viewerShown: document.getElementById('design-viewer').offsetParent !== null,
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
              // and Studio's sampler, off its defaults, so the archived job has to carry them
              document.getElementById('thinking').value = 'high';
              document.getElementById('backbone-k').value = '10';
              document.getElementById('latent-k').value = '3';
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
                # the gate stayed green. 🔴 AND EVERY STEP: a budget of 16 frames
                # drew one step in four of a 60-step run, which a user saw as
                # frames missing. The run here is 30 steps: 30 frames, and the
                # finished design after them (which grafts an unindexed motif the
                # live frames do not).
                if generated["frames"] != 31:
                    bad.append(f"the generation drew {generated['frames']} frames for its"
                               " 30 steps, expected one a step and the finished design")
                if generated["frames"] < 1:
                    bad.append("the generation drew no structure")
                if generated["progress"] != 1:
                    bad.append(f"the bar ended at {generated['progress']}, not 1")
                # 🔴 THE DESIGN STAYS UNTIL ASKED: an edit does not take it off screen, and "Show the
                # job" - offered only while a design is drawn - puts the job's sketch back.
                back = json.loads(cdp.evaluate(ws, """(() => {
                  const button = document.getElementById('show-job');
                  const offered = !button.hidden && button.offsetParent !== null;
                  const spans = window.__designSpans.read();
                  window.__designSpans.set(spans);
                  return new Promise((done) => setTimeout(() => {
                    const kept = window.__designDrawn() === null;
                    button.click();
                    done(JSON.stringify({ offered, kept, drawn: window.__designDrawn(), hidden: button.hidden }));
                  }, 400));
                })()"""))
                print(f"  show the job: {back}")
                if not back["offered"] or not back["kept"] or back["drawn"] is None or not back["hidden"]:
                    bad.append(f"after a generation, Show the job did not bring the job back: {back}")
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
                # 🔴 AND THE DESIGNS CAN BE TAKEN AWAY, AS THE JOB THAT MADE THEM. Nothing
                # offered a generated structure for download; now each candidate has its files
                # and the archive carries the job, which must be one the native port runs to the
                # same token count - a job file that is only decorative is the same bug again.
                archive = json.loads(cdp.evaluate(ws, """JSON.stringify({
                  rows: document.querySelectorAll('#downloads > div').length,
                  metricLines: [...document.querySelectorAll('#downloads .design-metrics')]
                    .map((e) => e.textContent),
                  members: window.__designArchive().map(([name, content]) =>
                    [name, typeof content === 'string' ? content : null]),
                })"""))
                names = [n for n, _ in archive["members"]]
                if archive["rows"] != 2:
                    bad.append(f"the downloads list {archive['rows']} candidates, expected 2")
                for want in ("job.json", "README.txt", "candidate-1.pdb", "candidate-1.cif",
                             "candidate-2.pdb", "candidate-2.cif", "candidate-1.metrics.json"):
                    if want not in names:
                        bad.append(f"the archive has no {want}: {names}")
                # 🔴 AND EACH DESIGN CARRIES UPSTREAM'S METRICS, under the `design` tag the job
                # records: the composition and clash keys present, the verdict beside them, and
                # the line on the page saying it. tools/check-rfd4p-metrics.py holds the values.
                if "candidate-1.metrics.json" in names:
                    m = json.loads(dict(archive["members"])["candidate-1.metrics.json"])
                    for key in ("amino_acid_composition.n_generated_unknown_residues",
                                "clash_metrics.frac_atoms_clashing_1p5A",
                                "sse_distribution.helix_fraction"):
                        if key not in m.get("metrics", {}):
                            bad.append(f"the design's metrics have no {key}: {sorted(m.get('metrics', {}))[:6]}")
                    if "pass" not in m.get("pre_oracle", {}):
                        bad.append(f"the design's metrics carry no pre-oracle verdict: {m.get('pre_oracle')}")
                    if '"metrics_tags"' not in dict(archive["members"])["job.json"]:
                        bad.append("the archived job does not record the metrics tags it was scored under")
                if len(archive["metricLines"]) != 2 or not all("pre-oracle" in t for t in archive["metricLines"]):
                    bad.append(f"the downloads do not show each design's metrics: {archive['metricLines']}")
                else:
                    print(f"            candidate 1: {archive['metricLines'][0]}")
                if "job.json" in names:
                    ran = json.loads(dict(archive["members"])["job.json"])
                    want = {"recycleSchedule": "thinking_high",
                            "sampling": {"bb_ca": {"simulation_step_params": {"tsr_k": 10}},
                                         "local_latents": {"simulation_step_params": {"tsr_k": 3}}}}
                    got = {k: ran.get(k) for k in want}
                    if got != want:
                        bad.append(f"the archived job does not record the sampler that ran: {got}")
                # 🔴 AND K REACHES THE ENGINE, not only the job file: the same seed at two K must be
                # two structures, and at one K twice the same one (the flow is deterministic).
                kdiff = json.loads(cdp.evaluate(ws, """(async () => {
                  const { generate } = await import('/web/generate.js');
                  const run = (k) => generate({ contig: '40-40', steps: 6, seed: 3,
                    ...(k === undefined ? {} : { sampling: { bb_ca: { simulation_step_params: { tsr_k: k } } } }) })
                    .then((r) => r.pdb);
                  const a = await run(undefined), b = await run(4), c = await run(4);
                  return JSON.stringify({ kMoves: a !== b, repeatable: b === c });
                })()"""))
                print(f"            backbone K at the engine: {kdiff}")
                if not kdiff["kMoves"] or not kdiff["repeatable"]:
                    bad.append(f"Backbone K does not reach the engine as a setting: {kdiff}")
                featuriser = Path(ROOT) / "cuda" / "rfd4p" / "featurise" / "rfd4p-featurise"
                if featuriser.exists() and "job.json" in names:
                    with tempfile.TemporaryDirectory() as tmp:
                        job = Path(tmp) / "job.json"
                        job.write_text(dict(archive["members"])["job.json"])
                        r = subprocess.run([str(featuriser), str(Path(tmp) / "f"), f"--spec={job}",
                                            f"--int8={Path(ROOT) / 'model-rfd4p-int8'}"], capture_output=True, text=True)
                        if "100 tokens" not in r.stdout:
                            bad.append(f"the archived job does not featurise natively to 100 tokens: "
                                       f"{(r.stdout + r.stderr)[-300:]}")
                        else:
                            print("            the archived job featurises natively: 100 tokens")

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
            contig: document.getElementById('contig-line').title.trim(),
            problems: document.getElementById('contig-problems').textContent.trim(),
            rows: document.querySelectorAll('#contig-rows > *').length,
            button: document.getElementById('hunt').textContent.trim(),
          });
        })()"""))
        # 🔴 THE PRESET'S STRUCTURE IS FETCHED, AND LOADING IT REWRITES THE
        # CONTIG. A shipped example is a CROP whose contig is written in the
        # numbering of the entry it was cut from, so the spans are translated onto
        # the file once it arrives - which happens a beat after the change event,
        # so reading immediately catches the untranslated form.
        for _ in range(20):
            if "A1-20" in loaded["contig"]:
                break
            time.sleep(1)
            loaded = json.loads(cdp.evaluate(ws, "JSON.stringify({"
                "method: document.getElementById('method').value,"
                "contig: document.getElementById('contig-line').title.trim(),"
                "problems: document.getElementById('contig-problems').textContent.trim(),"
                "rows: document.querySelectorAll('#contig-rows > *').length,"
                "button: document.getElementById('hunt').textContent.trim()})"))
        print(f"  preset motif_idx_1prw: {loaded}")
        # The crop runs chain A from 0-42; the preset asks for A16-35 and A52-71.
        # Translated, that is A1-20 and A22-41, and an UNtranslated contig here
        # means Protein Hunter would refuse a structure the preset just loaded.
        if "A1-20" not in loaded["contig"] or "A22-41" not in loaded["contig"]:
            bad.append("the preset's contig was not moved onto its own structure: "
                       f"{loaded['contig']!r}")
        # 🔴 THE PRESET'S SHAPE, REBUILT FROM THE ROWS IT FILLED. Loading a preset
        # into an editor is only honest if the editor can express it: if the round
        # trip lost a span the page would run a different job from the one it
        # names. This used to assert the preset's contig STRING - which is right
        # until the spans are translated onto the structure, and then asserts the
        # numbering that does not match the file. The five spans and the generated
        # lengths are what must survive; the chain ranges are checked above,
        # against the crop.
        shape = loaded["contig"].split()[0]
        if (not shape.startswith("5-20,") or ",10-25," not in shape
                or not shape.endswith(",5-20")):
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
            contig: document.getElementById('contig-line').title.trim(),
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
        if cath["structures"] != ["HELIX", "LOOP", "SHEET", "LOOP", "HELIX", "SHEET"]:
            bad.append(f"the secondary structures were not restored: {cath['structures']}")
        if cath["problems"] != "":
            bad.append(f"the labelled preset reported a problem: {cath['problems']!r}")

        # 🔴 AND THE CONDITION EDITOR, WITH A REAL CLICK. Raw 1PRW is opened from the file control,
        # which must draw it; a pixel over a residue is found with py2Dmol's own picker and CLICKED
        # through CDP's input events - not `setResidueSelection`, which would test an API the reader
        # never touches - and the row added must name that residue in the selection language's own
        # numbering, narrowed to the atoms typed. Last, because it leaves a structure loaded.
        cdp.evaluate(ws, """(async () => {
          const pick = document.getElementById('preset');
          pick.value = ''; pick.dispatchEvent(new Event('change'));
          const m = document.getElementById('method'); m.value = 'rfd4p'; m.dispatchEvent(new Event('change'));
          const bytes = await (await fetch('/tools/fixtures/rcsb/1prw.cif.zst')).arrayBuffer();
          const dt = new DataTransfer(); dt.items.add(new File([bytes], '1prw.cif.zst'));
          const input = document.getElementById('structure-file'); input.files = dt.files;
          input.dispatchEvent(new Event('change'));
          return 1;
        })()""")
        cdp.wait_for(ws, "(() => { const e = Object.values(window.py2dmol_viewers ?? {})[0];"
                         " return ((e?.renderer ?? e)?.chains?.length ?? 0) > 100; })()", 30, "1PRW drawn")
        time.sleep(1.0)
        target = json.loads(cdp.evaluate(ws, """(() => {
          const entry = Object.values(window.py2dmol_viewers ?? {})[0];
          const r = entry?.renderer ?? entry;
          // 🔴 ON SCREEN FIRST: a click outside the window reaches nothing, and the earlier arms
          // leave the page scrolled - the picker still answers there, so the miss is silent.
          const canvas = document.querySelector('#design-viewer canvas');
          canvas.scrollIntoView({ block: 'center' });
          const box = canvas.getBoundingClientRect();
          // 🔴 WHOLE PIXELS, AND THE SAME RESIDUE AROUND IT: a mouse event's coordinates are integers
          // where this search's were fractional, so a hit found on a residue's EDGE was a miss by the
          // time the click arrived - py2Dmol read it as the background and cleared the selection.
          const steady = (x, y, hit) => [[-2, 0], [2, 0], [0, -2], [0, 2]]
            .every(([dx, dy]) => r.pickResidueAt(x + dx, y + dy) === hit);
          // ...and a residue of the STRUCTURE: the job's sketched pieces are drawn too, and pick
          const sketched = new Set(window.__designDrawn()?.ghosts ?? []);
          for (let y = Math.ceil(box.top + box.height * 0.3); y < box.bottom - 30; y += 4)
            for (let x = Math.ceil(box.left + box.width * 0.3); x < box.right - 10; x += 4) {
              const hit = r.pickResidueAt(x, y);
              if (hit >= 0 && r.positionTypes?.[hit] === 'P' && !sketched.has(r.chains[hit]) && steady(x, y, hit))
                return JSON.stringify({ x, y, chain: r.chains[hit], num: r.residueNumbers[hit] });
            }
          return 'null';
        })()"""))
        editor = None
        if target is None:
            bad.append("no residue of 1PRW could be found under the pointer to click")
        else:
            for kind in ("mousePressed", "mouseReleased"):
                ws.call("Input.dispatchMouseEvent", type=kind, x=target["x"], y=target["y"],
                        button="left", clickCount=1)
                time.sleep(0.05)
            time.sleep(0.4)
            editor = json.loads(cdp.evaluate(ws, """(() => {
              const kind = document.getElementById('condition-kind');
              kind.value = 'C_MCOORD'; kind.dispatchEvent(new Event('change'));
              document.getElementById('condition-value').value = 'COORDINATING';
              document.getElementById('condition-atoms').value = 'od1,OD2';
              document.getElementById('condition-add').click();
              const added = window.__designConditions();
              const entry = Object.values(window.py2dmol_viewers ?? {})[0];
              const chip = document.querySelector('#condition-rows li')?.textContent ?? '';
              (entry?.renderer ?? entry).clearResidueSelection();
              // nothing selected: the actions are disabled, which is the refusal
              const empty = document.getElementById('condition-add').disabled
                && [...document.querySelectorAll('.condition-quick button')].every((b) => b.disabled);
              const m = document.getElementById('method'); m.value = 'hunter'; m.dispatchEvent(new Event('change'));
              const greyed = document.getElementById('step-constraints').hidden;
              m.value = 'rfd4p'; m.dispatchEvent(new Event('change'));
              return JSON.stringify({ added, empty, greyed, chip,
                rows: [...document.querySelectorAll('#condition-rows li')].length,
                regreyed: document.getElementById('step-constraints').hidden });
            })()"""))
            print(f"  condition editor: clicked {target['chain']}{target['num']} -> {editor}")
            want = {"C_MCOORD": {"COORDINATING": [
                {"select": f"sel('{target['chain']}/*/[{target['num']}]/[OD1,OD2]')"}]}}
            # The spans the CATH example left still carry their C_SEC labels, and those are still
            # the job: the editor's row is merged AFTER them, never in their place.
            added = editor["added"] or {}
            if added.get("C_MCOORD") != want["C_MCOORD"] or list(added)[-1:] != ["C_MCOORD"] \
                    or "C_SEC" not in added:
                bad.append(f"the clicked residue became {added}, expected the spans' C_SEC then {want}")
            if editor["rows"] != 1:
                bad.append(f"the editor shows {editor['rows']} rows, expected 1")
            # 🔴 THE CHIP NAMES THE RESIDUE AS A READER WOULD, not as the selection syntax does.
            if editor["chip"] != f"Metal coordination coordinating \u00b7 {target['chain']}{target['num']} (OD1,OD2)\u00d7":
                bad.append(f"the constraint chip reads {editor['chip']!r}")
            if editor["empty"] is not True:
                bad.append("with nothing selected the constraint actions are still live")
            if not editor["greyed"] or editor["regreyed"]:
                bad.append("the constraints step is not hidden under Protein Hunter and back under RFD4-Proteina")
            # 🔴 AND THE FOLD PAGE'S SEQUENCE STRIP, REUSED: a real DRAG along it selects a run (the
            # same `residueSelection` a click sets), and "Add as spans" appends that run to the job as a
            # From structure span in the structure's own numbering.
            strip = json.loads(cdp.evaluate(ws, """(() => {
              const c = document.querySelector('#sequenceView canvas');
              if (!c) return 'null';
              c.scrollIntoView({ block: 'center' });
              const b = c.getBoundingClientRect();
              return JSON.stringify({ x: b.left, y: b.top });
            })()"""))
            if strip is None:
                bad.append("the sequence strip did not draw the loaded structure")
            else:
                y = strip["y"] + 30
                xs = [strip["x"] + 60 + i * 8 for i in range(12)]
                ws.call("Input.dispatchMouseEvent", type="mouseMoved", x=xs[0], y=y)
                ws.call("Input.dispatchMouseEvent", type="mousePressed", x=xs[0], y=y, button="left", clickCount=1)
                for x in xs[1:]:
                    ws.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="left", buttons=1)
                    time.sleep(0.02)
                ws.call("Input.dispatchMouseEvent", type="mouseReleased", x=xs[-1], y=y, button="left", clickCount=1)
                time.sleep(0.4)
                dragged = json.loads(cdp.evaluate(ws, """(() => {
                  const entry = Object.values(window.py2dmol_viewers ?? {})[0];
                  const r = entry?.renderer ?? entry;
                  const sel = [...(r.residueSelection ?? [])].sort((a, b) => a - b);
                  const before = window.__designSpans.read().length;
                  document.getElementById('selectionToSpans').click();
                  const spans = window.__designSpans.read();
                  // the quick action: one press, the selection becomes a named chip
                  document.querySelector('.condition-quick [data-condition=C_HOT]').click();
                  const chips = [...document.querySelectorAll('#condition-rows li')].map((li) => li.textContent);
                  // and a block of the diagram opens its own row
                  const blocks = document.querySelectorAll('#chain-diagram .chain-block');
                  blocks[blocks.length - 1].click();
                  const opened = document.querySelectorAll('#contig-rows .contig-more').length;
                  // the strip's own buttons, wired by py2Dmol's panel (sokrypton/py2Dmol#32), not here
                  const size = () => (r.residueSelection ? r.residueSelection.size : 0);
                  document.getElementById('selectAllResidues').click();
                  const all = size();
                  document.getElementById('invertSelection').click();
                  const inverted = size();
                  document.getElementById('selectAllResidues').click();
                  document.getElementById('clearAllResidues').click();
                  return JSON.stringify({ chains: [...new Set(sel.map((i) => r.chains[i]))],
                    nums: sel.map((i) => r.residueNumbers[i]), added: spans.slice(before), chips, opened,
                    // the structure's positions: Select all takes no sketched piece
                    buttons: { n: r.chains.filter((c) => !(window.__designDrawn()?.ghosts ?? []).includes(c)).length,
                      all, inverted, cleared: size() } });
                })()"""))
                print(f"  sequence strip: dragged {dragged['chains']} {dragged['nums'][:1]}..{dragged['nums'][-1:]}"
                      f" -> {dragged['added']}")
                nums = dragged["nums"]
                if len(nums) < 4 or nums != list(range(nums[0], nums[0] + len(nums))):
                    bad.append(f"a drag along the strip did not select a run: {nums}")
                elif dragged["added"] != [{"kind": "segment", "chain": dragged["chains"][0], "first": nums[0],
                                           "last": nums[-1], "hotspots": ""}]:
                    bad.append(f"Add as spans appended {dragged['added']}, expected the dragged run")
                run = f"{dragged['chains'][0]}{nums[0]}\u2013{nums[-1]}" if nums else "?"
                if not any(c.startswith(f"Hotspot \u00b7 {run}") for c in dragged["chips"]):
                    bad.append(f"the quick Hotspot action did not chip {run}: {dragged['chips']}")
                if dragged["opened"] < 1:
                    bad.append("clicking a block of the chain diagram did not open its row")
                b = dragged["buttons"]
                print(f"  strip buttons: {b}")
                if (b["all"], b["inverted"], b["cleared"]) != (b["n"], 0, 0):
                    bad.append(f"the strip's Select all / Invert / Unselect do not act on the input: {b}")
            # ...and, with --generate, a real generation from it: the archived job must carry the
            # row, merged into the conditions the engine was handed, and the run must finish.
            if "--generate" in sys.argv:
                cdp.evaluate(ws, """(() => {
                  window.__designSpans.set([{ kind: 'generated', min: 6, max: 6 },
                    { kind: 'segment', chain: 'A', first: 60, last: 80 }, { kind: 'generated', min: 6, max: 6 }]);
                  document.getElementById('steps').value = '8';
                  document.getElementById('beam').value = '1';
                  document.getElementById('validate').value = '';
                  document.getElementById('hunt').click();
                  return 1;
                })()""")
                deadline = time.time() + 1200
                state = {}
                while time.time() < deadline:
                    state = json.loads(cdp.evaluate(ws, """JSON.stringify({
                      status: document.getElementById('status-message').textContent,
                      failed: document.getElementById('status-message').classList.contains('error') })"""))
                    if state["failed"] or state["status"].startswith("done"):
                        break
                    time.sleep(3)
                print(f"  conditioned generation: {state.get('status')}")
                hidden = cdp.evaluate(ws, "document.getElementById('sequence-viewer-container').style.display")
                if hidden != "none":
                    bad.append(f"the sequence strip stayed up over a design ({hidden!r}): its residues are not the input's")
                if state.get("failed") or not state.get("status", "").startswith("done"):
                    bad.append(f"a generation with an editor condition did not finish: {state}")
                else:
                    job = json.loads(dict(json.loads(cdp.evaluate(ws,
                        "JSON.stringify(window.__designArchive().map(([n, c]) => [n, typeof c === 'string' ? c : null]))")))
                        ["job.json"])
                    if job.get("conditions", {}).get("C_MCOORD") != want["C_MCOORD"]:
                        bad.append(f"the archived job lost the editor's row: {job.get('conditions')}")
    finally:
        proc.terminate()
        server.terminate()
        proc.wait(timeout=30)
        # The browser's profile goes with it: a run leaves 1-2 GB of cache and weights behind.
        shutil.rmtree(profile, ignore_errors=True)

    print("FAIL: " + "; ".join(bad) if bad else "design.html: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

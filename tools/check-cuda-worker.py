"""The CUDA backend's worker, folding for real: npm run test:cuda.

    python3 tools/check-cuda-worker.py             # every case (two go to api.colabfold.com)
    python3 tools/check-cuda-worker.py --offline   # without the search cases
    python3 tools/check-cuda-worker.py --no-page   # without the page arm (a headless Chrome, a broker)
    python3 tools/check-cuda-worker.py --only=6mrr  # just the cases whose name contains it

cuda/worker.py over its own stdin protocol, one process for every case as the broker runs it,
each job shaped as the page sends one (AlphaFold 3 JSON from the entity rows, the rows themselves, the
form's controls), each fold scored against its deposited structure (cuda/af3/score.py) and held to a
bar, and every result held to the fields the page ingests (web/app.js, jaxPrediction): a PDB, a pLDDT a
token, a PAE of tokens^2, the token layout, the chains. test:colab's CUDA arm is a stub and proves the
broker's routing; this is the half that proves the worker folds.

Needs the native ports built (cuda/colab_setup.sh, or each port's nvcc line) and the bundles on disk
(a missing one is fetched). The refusals are cases too: a sampler, a model or an input the CUDA backend
does not have must come back as a sentence naming it, never as some other fold.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIX = os.path.join(REPO, "tools", "fixtures")
S6 = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
THREE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L",
             LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")


def chain_sequence(pdb, chain):
    seen, out = set(), ""
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[21] == chain and line[22:27] not in seen:
            seen.add(line[22:27])
            out += THREE.get(line[17:20], "X")
    return out


def job(sequences, ligands=(), modifications=None):
    """AlphaFold 3 JSON as web/job-json.js writes it for these rows."""
    entries, ids = [], iter("ABCDEFGHIJ")
    for index, sequence in enumerate(sequences):
        protein = {"id": next(ids), "sequence": sequence}
        if modifications and index in modifications:
            protein["modifications"] = [{"ptmType": code, "ptmPosition": at} for code, at in modifications[index]]
        entries.append({"protein": protein})
    for code in ligands:
        entries.append({"ligand": {"id": next(ids), "ccdCodes": [code]}})
    return json.dumps({"name": "check", "modelSeeds": [1], "sequences": entries, "dialect": "alphafold3", "version": 1})


def controls(**over):
    return {"model-family": "af3", "msa-mode": "none", "max-msa": "128:256", "recycles": "3", "tolerance": "0.1",
            "af3-mode": "diffusion", "af3-count": "25", "random-seed": "1", **over}


def protein(sequence, template=None):
    row = {"type": "protein", "value": sequence, "copies": 1}
    if template:
        row["template"] = template
    return row


def cases(offline):
    s5 = chain_sequence(f"{FIX}/5caj-crystal.pdb", "A")
    sa, sd = chain_sequence(f"{FIX}/1brs-crystal.pdb", "A"), chain_sequence(f"{FIX}/1brs-crystal.pdb", "D")
    brs = open(f"{FIX}/1brs-crystal.pdb").read()
    caj = open(f"{FIX}/5caj-crystal.pdb").read()
    out = [
        # name, payload, (reference, chains) or None, bar (A), expected tokens or None, refusal text or None
        ("af3 6mrr", {"family": "af3", "controls": controls(), "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        ("af3 5caj, its crystal uploaded", {"family": "af3", "controls": controls(),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 0.5, None, None),
        ("protenix2 6mrr", {"family": "protenix2", "controls": controls(**{"model-family": "protenix2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        # (past 80 tokens: protenix2's 256-channel pair track on its fused kernels, cuda/af3/src/fused256.cuh)
        ("protenix2 5caj, its crystal uploaded", {"family": "protenix2", "controls": controls(**{"model-family": "protenix2"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 0.5, None, None),
        ("opendde 6mrr", {"family": "opendde", "controls": controls(**{"model-family": "opendde", "af3-count": "16"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        ("af3 6mrr + GOL + SEP3", {"family": "af3", "controls": controls(),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}],
         "job": job([S6], ["GOL"], {0: [("SEP", 3)]})}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 9 + 6, None),
        ("af2 monomer 6mrr", {"family": "monomer", "controls": controls(**{"model-family": "monomer", "af2Model": "1"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 monomer 6mrr, stopping early at 0.5 A", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "1", "tolerance": "0.5"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 monomer 5caj, its crystal uploaded", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "1", "recycles": "0"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 1.0, None, None),
        ("af2 multimer 1brs, both chains templated", {"family": "multimer",
         "controls": controls(**{"model-family": "multimer", "af2Model": "1"}),
         "entities": [protein(sa, {"kind": "upload", "text": brs, "source": "A", "filename": "1brs.pdb"}),
                      protein(sd, {"kind": "upload", "text": brs, "source": "D", "filename": "1brs.pdb"})],
         "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.0, len(sa) + len(sd), None),
        ("af2 multimer 1brs, an alignment per chain (an archive's)", {"family": "multimer",
         "controls": controls(**{"model-family": "multimer", "af2Model": "1", "msa-mode": "upload"}),
         "entities": [protein(sa), protein(sd)], "job": job([sa, sd]),
         "msas": {"unpaired": [f">101\n{sa}\n", f">101\n{sd}\n"], "paired": ["", ""]}},
         (f"{FIX}/1brs-crystal.pdb", "A,D"), 25.0, len(sa) + len(sd), None),
        ("esmfold2 6mrr", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        # (past 80 tokens: ESMFold2's 256-channel trunk on its fused kernels, cuda/af3/src/fused256.cuh)
        ("esmfold2 5caj", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(s5)], "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 3.0, None, None),
        ("esmfold2 300M 6mrr", {"family": "ef2-fast-300m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        # the page's short schedule (web/af3-model.js diffusionScheduleFor), which the page resolves and sends:
        # an empty step dial is the family's preferred count there and the model's own 200 here
        ("af3 6mrr, the page's short schedule", {"family": "af3", "controls": controls(**{"af3-count": ""}),
         "schedule": {"steps": 20, "sigmaMax": 80}, "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        ("af3 6mrr, the Flow sampler", {"family": "af3", "controls": controls(**{"af3-mode": "flow", "af3-count": "16"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 1.2, 68, None),
        ("chai1 6mrr", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("chai1 5caj, its crystal uploaded", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 3.0, None, None),
        ("chai1 6mrr + GOL + SEP3", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}],
         "job": job([S6], ["GOL"], {0: [("SEP", 3)]})}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 9 + 6, None),
        ("refused: flow on chai1", {"family": "chai1",
         "controls": controls(**{"model-family": "chai1", "af3-mode": "flow"}), "entities": [protein(S6)],
         "job": job([S6])}, None, None, None, "Diffusion"),
        ("refused: flow on rosettafold3", {"family": "rosettafold3",
         "controls": controls(**{"model-family": "rosettafold3", "af3-mode": "flow"}), "entities": [protein(S6)],
         "job": job([S6])}, None, None, None, "Diffusion"),
        ("af2 monomer model 3 (a delta on model 1) 6mrr", {"family": "monomer-3",
         "controls": controls(**{"model-family": "monomer"}), "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 multimer model 2 (a delta) 1brs, templated", {"family": "multimer-2",
         "controls": controls(**{"model-family": "multimer"}),
         "entities": [protein(sa, {"kind": "upload", "text": brs, "source": "A", "filename": "1brs.pdb"}),
                      protein(sd, {"kind": "upload", "text": brs, "source": "D", "filename": "1brs.pdb"})],
         "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.0, len(sa) + len(sd), None),
        ("refused: a template on AF2 model 4", {"family": "monomer-4", "controls": controls(**{"model-family": "monomer"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A"})], "job": job([s5])},
         None, None, None, "no template embedder"),
        ("refused: a template on ESMFold2", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A"})], "job": job([s5])},
         None, None, None, "no template"),
        ("refused: a ligand on AF2", {"family": "monomer", "controls": controls(**{"model-family": "monomer"}),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}], "job": job([S6], ["GOL"])},
         None, None, None, "protein chains only"),
    ]
    if not offline:
        out += [
            ("af3 1brs, MSA searched", {"family": "af3", "controls": controls(**{"msa-mode": "search"}),
             "entities": [protein(sa), protein(sd)], "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.5,
             len(sa) + len(sd), None),
            ("af2 monomer 5caj, MSA and template searched", {"family": "monomer",
             "controls": controls(**{"model-family": "monomer", "af2Model": "1", "msa-mode": "search"}),
             "entities": [protein(s5, {"kind": "search"})], "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 2.5,
             None, None),
        ]
    return out


def score(pdb_text, reference, chains):
    path = "/tmp/localfold-cuda-check.pdb"
    with open(path, "w") as handle:
        handle.write(pdb_text)
    said = subprocess.run([sys.executable, os.path.join(REPO, "cuda", "af3", "score.py"), path, reference, chains],
                          capture_output=True, text=True).stdout
    found = re.findall(r"CA RMSD ([0-9.]+) A", said)
    return float(found[0]) if found else None


def page_arm(bad):
    """The website's half: a real broker (tools/colab_backend.py --cuda), a reader's page on it - no
    backend picker, CUDA is where its folds go - a sequence and Fold - and what the page made of the answer: the status
    line the worker wrote, a prediction in the page's own shape (its PAE a typed array of tokens^2, the
    model labelled CUDA), and no WebGPU device asked for in the reader's browser."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import cdp
    import urllib.request
    port, cdp_port, reader_port, token = 8893, 9395, 9396, "cuda-check"
    broker = subprocess.Popen([sys.executable, "tools/colab_backend.py", "--port", str(port), "--cdp-port", str(cdp_port),
                               "--token", token, "--profile", "/tmp/localfold-cuda-page-runtime", "--cuda"],
                              cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    reader = None
    try:
        deadline = time.time() + 180
        for line in broker.stdout:
            if line.startswith("BACKEND ") or time.time() > deadline:
                break
        health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health?t={token}"))
        if "cuda" not in health.get("backends", []):
            bad.append(f"page: /health offers {health.get('backends')} with --cuda")
            return
        reader, ws = cdp.launch(reader_port, "/tmp/localfold-cuda-page-reader")
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
          window.__readerGpu = 0;
          if (navigator.gpu) {
            const ask = navigator.gpu.requestAdapter.bind(navigator.gpu);
            navigator.gpu.requestAdapter = (...a) => { window.__readerGpu += 1; return ask(...a); };
          }""")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/index.html?backend=colab&t={token}")
        cdp.wait_for(ws, "!!window.__entityList && !!document.querySelector('.colab-live')", 120, "the reader's page")
        cdp.evaluate(ws, """(() => {
          for (const key of ['alphafold3', 'openbind0', 'opendde', 'boltz2', 'protenix2', 'intellifold2', 'rosettafold3'])
            try { localStorage.setItem('localfold.modelTerms.' + key, 'accepted'); } catch (cause) {}
          return true;
        })()""")
        # ...and nothing to choose: the page offers no backend control at all
        if cdp.evaluate(ws, "!!document.querySelector('#colab-status select')"):
            bad.append("page: the badge offers a backend select - CUDA is not a choice")
        # the badge's Live preview: present for CUDA, and off means the finished fold only
        live = cdp.evaluate(ws, "(() => { const l = document.querySelector('.colab-live'); return l ? { shown: !l.hidden, on: l.querySelector('input').checked } : null; })()")
        if not live or not live["shown"] or not live["on"]:
            bad.append(f"page: the Live preview box is {live} - want it shown and on for CUDA")
        for family, tokens, streamed in (("af3", 68, True), ("monomer", 68, True), ("ef2-fast-600m", 68, True), ("af3", 68, False)):
            cdp.evaluate(ws, f"""(() => {{ const b = document.querySelector('.colab-live input');
              if (b.checked !== {'true' if streamed else 'false'}) b.click(); return b.checked; }})()""")
            cdp.evaluate(ws, f"""(() => {{
              const g = (id) => document.getElementById(id);
              g('model-family').value = '{family}';
              g('model-family').dispatchEvent(new Event('change', {{ bubbles: true }}));
              window.__entityList.set([{{ type: 'protein', value: '{S6}', copies: 1, modifications: [] }}]);
              if (g('msa-mode')) {{ g('msa-mode').value = 'none'; g('msa-mode').dispatchEvent(new Event('change', {{ bubbles: true }})); }}
              const a = g('af3-mode'); if (a) {{ a.value = 'diffusion'; a.dispatchEvent(new Event('change', {{ bubbles: true }})); }}
              return true;
            }})()""")
            cdp.wait_for(ws, "!document.getElementById('predict').disabled", 60, "the fold button")
            started = time.time()
            cdp.evaluate(ws, "(document.getElementById('predict').click(), true)")
            cdp.wait_for(ws, """(() => { const t = document.getElementById('status-message').textContent;
              return / on CUDA .* done in |failed|refused|Error|error/.test(t); })()""", 300, f"a CUDA fold of {family}")
            seconds = time.time() - started           # (click to the result's status line, as a reader waits)
            time.sleep(1.0)
            got = cdp.evaluate(ws, """(() => {
              const p = window.__lastPrediction();
              return { status: document.getElementById('status-message').textContent,
                       model: p?.model ?? null, family: p?.family ?? null,
                       pae: p?.confidence?.predictedAlignedError?.length ?? null,
                       paeTyped: p?.confidence?.predictedAlignedError instanceof Float32Array,
                       plddt: p?.confidence?.plddt?.length ?? null,
                       atoms: (p?.pdb ?? '').split(String.fromCharCode(10)).filter((l) => l.startsWith('ATOM')).length,
                       frames: (() => { const v = Object.values(window.py2dmol_viewers ?? {})[0]?.renderer;
                                        return v?.objectsData?.[v?.currentObjectName]?.frames?.length ?? 0; })(),
                       mapped: (() => { const v = Object.values(window.py2dmol_viewers ?? {})[0]?.renderer;
                                        const fs = v?.objectsData?.[v?.currentObjectName]?.frames ?? [];
                                        return { contact: fs.filter((f) => f.maps?.contact).length,
                                                 pae: fs.filter((f) => f.pae).length }; })(),
                       gpu: window.__readerGpu };
            })()""")
            print(f"page: {family:14s} live {'on ' if streamed else 'off'} {seconds:5.2f} s  {got['frames']:2d} frames {got['mapped']}  {got['status'][:70]}")
            problems = []
            if " on CUDA " not in got["status"] or "done in" not in got["status"]:
                problems.append(f"status {got['status']!r}")
            if not (got["model"] or "").endswith("(CUDA)"):
                problems.append(f"model {got['model']!r}")
            if got["pae"] != tokens * tokens or not got["paeTyped"] or got["plddt"] != tokens:
                problems.append(f"PAE {got['pae']} (typed {got['paeTyped']}), pLDDT {got['plddt']}")
            if got["atoms"] < tokens * 4:
                problems.append(f"{got['atoms']} atoms")
            if not streamed:
                if got["frames"] > 1:
                    problems.append(f"Live preview off and the viewer still holds {got['frames']} frames")
            elif family in ("af3", "ef2-fast-600m") and (got["frames"] < 5 or got["mapped"]["contact"] < 5):
                problems.append(f"the viewer holds {got['frames']} frames, {got['mapped']['contact']} with a contact map")
            if streamed and family == "monomer" and (got["mapped"]["pae"] < 1 or got["mapped"]["contact"] < 1):
                problems.append(f"the viewer's AF2 passes carry {got['mapped']} - no streamed PAE or contacts")
            if got["gpu"]:
                problems.append("the reader's browser asked for a WebGPU adapter")
            bad += [f"page {family}{'' if streamed else ' (live off)'}: {p}" for p in problems]
    finally:
        if reader is not None:
            reader.terminate()
        # 🔴 SIGINT, NOT SIGTERM: the broker stops its own headless Chrome on the way out of a
        # KeyboardInterrupt, and a SIGTERM skips that and orphans it on the CDP port the next run wants
        broker.send_signal(signal.SIGINT)
        try:
            broker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            broker.kill()


def main():
    offline = "--offline" in sys.argv
    plan = cases(offline)
    only = next((a[7:] for a in sys.argv if a.startswith("--only=")), None)   # (a substring of the case names)
    if only is not None:
        plan = [case for case in plan if only in case[0]]
    worker = subprocess.Popen([sys.executable, os.path.join(REPO, "cuda", "worker.py")], cwd=REPO,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/localfold-cuda-check.log", "w"),
                              text=True, bufsize=1)
    first = json.loads(worker.stdout.readline())
    bad = [] if first.get("kind") == "cuda-ready" else [f"the worker's first line was {first}, not cuda-ready"]
    for name, payload, reference, bar, tokens, refusal in plan:
        started = time.time()
        worker.stdin.write(json.dumps(payload) + "\n")
        worker.stdin.flush()
        kinds, result, fractions = [], None, []
        for line in worker.stdout:
            event = json.loads(line)
            kinds.append(event["kind"])
            if event["kind"] == "progress":
                fractions.append(event["payload"])
            if event["kind"] == "result":
                result = event["payload"]
                break
        seconds = time.time() - started
        if result is None:
            bad.append(f"{name}: the worker exited")
            break
        if refusal is not None:
            said = result.get("error", "")
            ok = refusal in said and "Error:" not in said.split(" ")[0]
            print(f"{name:44s} {'refused' if ok else 'WRONG'}: {said[:110]}")
            if not ok:
                bad.append(f"{name}: wanted a refusal naming {refusal!r}, got {result.get('error') or result.get('status')}")
            continue
        if "error" in result:
            print(f"{name:44s} FAILED: {result['error'][:200]}")
            bad.append(f"{name}: {result['error'][:200]}")
            continue
        c = result.get("confidence") or {}
        n = len(c.get("plddt") or [])
        problems = []
        if result.get("cuda") is not True:
            problems.append("not marked cuda")
        if len(c.get("predictedAlignedError") or []) != n * n:
            problems.append(f"PAE {len(c.get('predictedAlignedError') or [])} for {n} tokens")
        if len((result.get("tokens") or {}).get("chainIds") or []) != n:
            problems.append("the token layout is not one entry a token")
        if tokens is not None and n != tokens:
            problems.append(f"{n} tokens, not {tokens}")
        if not (0 < (c.get("meanPlddt") or 0) <= 100) or c.get("ptm") is None:
            problems.append(f"pLDDT {c.get('meanPlddt')} / pTM {c.get('ptm')}")
        if "status" not in kinds or kinds.count("progress") < 2:
            problems.append(f"events {kinds}")
        # what each port streams (--frames, through the binaries' AsyncTap), the bar moving with it: the
        # AF3 lineage its sampler and each trunk pass's contact map, AF2 each pass (structure + scores, and
        # contacts where the bundle carries the distogram head), ESMFold2 its trunk's contacts and sampler
        family = payload["family"].split("-")[0]
        want = {"frame": 5, "contacts": 1} if family in ("af3", "openbind0", "opendde", "boltz2", "protenix2",
                                                         "intellifold2", "rosettafold3", "chai1", "ef2") \
            else {"frame": 1, "scores": 1, **({"contacts": 1} if family == "monomer" else {})}
        for kind, least in want.items():
            if kinds.count(kind) < least:
                problems.append(f"{kinds.count(kind)} {kind} events streamed")
        if fractions != sorted(fractions):
            problems.append("the progress bar went backwards")
        if "stopping early" in name and "converged at" not in (result.get("status") or ""):
            problems.append("the page's early stop did not stop it (or did not say so)")
        rmsd = score(result.get("pdb") or "", *reference)
        if rmsd is None or rmsd > bar:
            problems.append(f"RMSD {rmsd} past {bar} A")
        print(f"{name:44s} RMSD {rmsd} A (bar {bar})  {n} tokens  {kinds.count('frame'):2d}f {kinds.count('contacts')}c {kinds.count('scores')}s  {seconds:5.1f} s"
              f"  | {result.get('status')}")
        bad += [f"{name}: {p}" for p in problems]
    worker.stdin.close()
    worker.wait(timeout=60)
    if "--no-page" not in sys.argv:
        page_arm(bad)
    if bad:
        print("\n" + "\n".join(bad) + "\n\ncuda worker: FAILED (its log: /tmp/localfold-cuda-check.log)")
        sys.exit(1)
    print("\ncuda worker: ok")


if __name__ == "__main__":
    main()

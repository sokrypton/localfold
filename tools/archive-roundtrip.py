"""Fold, download the archive, upload it back, fold again - and compare.

    python3 tools/archive-roundtrip.py

🔴 THIS IS THE CHECK THE ARCHIVE EXISTS FOR. `web/app.js` used to say it
outright: "A pasted or uploaded A3M is one text and cannot be split into blocks;
it becomes the unpaired one." AlphaFold 3 reads the paired block first and takes
its profile over the unpaired block ALONE, so a fold restored from its own
downloaded alignment was silently a different fold - a different MSA, a
different structure, and nothing on screen to say so.

The unit tests assert that the per-chain files merge back to the same blocks.
They cannot assert that the PAGE does it: the download button, the file input,
the upload branch of `alignmentText` and the model's own cache key are four
things between the two halves, and each is a place the split can be lost.

🔴 THE STRUCTURES ARE COMPARED, NOT THE DEPTHS. "128 MSA rows" is reported for
almost any alignment - it is the row budget, not the alignment - so two folds
agreeing on it says nothing. Two folds at the same seed agree residue for
residue only if they were given the same MSA, which is the whole question.
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import cdp                                                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fip", os.path.join(os.path.dirname(__file__), "fold-in-page.py"))
fip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fip)

# Two short, unrelated chains: distinct sequences are what make the search
# produce a PAIRED block at all, and a homo-oligomer would not exercise it.
CHAIN_A = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"
CHAIN_B = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
           "KESTLHLVLRLRGG")


def fold_and_wait(ws, timeout):
    cdp.evaluate(ws, "window.__foldClickedAt = performance.now();"
                     " document.getElementById('predict').click()")
    cdp.wait_for(ws, """(() => {
      const s = document.getElementById('status-message');
      return /pLDDT/.test(s ? s.textContent : '')
        && document.getElementById('downloads').style.display !== 'none';
    })()""", timeout=timeout, what="the fold to finish", progress=fip.STATUS_LINE)
    time.sleep(1.0)
    return cdp.evaluate(ws, "(document.getElementById('status-message')||{}).textContent")


def structure_digest(ws, downloads, label):
    """A hash of the PDB the page actually saves.

    🔴 THE SAVED FILE, NOT THE VIEWER'S COPY. py2Dmol's frame objects are its
    own business and have changed shape before; the PDB button is the thing a
    reader gets, and hashing it compares what the two folds would hand over
    rather than what the page happens to be holding.
    """
    before = set(os.listdir(downloads))
    cdp.evaluate(ws, "document.getElementById('download-pdb').click()")
    end = time.time() + 60
    while time.time() < end:
        fresh = [f for f in set(os.listdir(downloads)) - before
                 if f.endswith(".pdb")]
        if fresh:
            path = os.path.join(downloads, fresh[0])
            data = open(path, "rb").read()
            os.rename(path, os.path.join(downloads, "%s.pdb" % label))
            atoms = sum(1 for line in data.decode().splitlines()
                        if line.startswith("ATOM"))
            return hashlib.sha256(data).hexdigest()[:16], atoms
        time.sleep(0.5)
    raise RuntimeError("no structure was downloaded")


def main() -> int:
    # ...outside the checkout, because Chrome writes several files here and a
    # download directory in the repository is one more thing to gitignore.
    downloads = tempfile.mkdtemp(prefix="localfold-roundtrip-")

    httpd = fip.serve(local_weights=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    proc, ws = cdp.launch(fip.DBG, "/tmp/_cdp_roundtrip_profile")
    try:
        ws.call("Page.enable")
        ws.call("DOM.enable")
        # Headless Chrome cancels a download unless told where to put it.
        ws.call("Browser.setDownloadBehavior", behavior="allow",
                downloadPath=downloads)
        ws.call("Page.navigate",
                url="http://127.0.0.1:%d/index.html" % fip.PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     timeout=90, what="the page to load")

        print(cdp.evaluate(ws, """(() => {
          const list = window.__entityList;
          const base = list.read().find((e) => e.type === 'protein')
            ?? { type: 'protein', copies: 1, modifications: [] };
          list.set(%s.map((value) => ({ ...base, value, copies: 1 })));
          return 'chains: ' + JSON.stringify(list.read().map((e) => e.value.length));
        })()""" % json.dumps([CHAIN_A, CHAIN_B])))

        def control(name, value):
            cdp.evaluate(ws, """(() => {
              const el = document.getElementById(%s);
              if (!el) return 'no ' + %s;
              el.value = %s;
              el.dispatchEvent(new Event('change', { bubbles: true }));
              return el.value;
            })()""" % (json.dumps(name), json.dumps(name), json.dumps(value)))

        control("model-family", "af3")
        control("recycles", "0")
        control("af3-count", "4")
        control("random-seed", "7")
        control("msa-mode", "search")

        print("first fold (search)…", flush=True)
        first_status = fold_and_wait(ws, 900)
        print("  ", first_status)
        first_digest, first_atoms = structure_digest(ws, downloads, "first")

        print("downloading the archive…", flush=True)
        cdp.evaluate(ws, "document.getElementById('download-all').click()")
        end = time.time() + 60
        archive = None
        while time.time() < end:
            ready = [f for f in os.listdir(downloads)
                     if f.endswith(".zip") and not f.endswith(".crdownload")]
            if ready:
                archive = os.path.join(downloads, ready[0])
                break
            time.sleep(0.5)
        if archive is None:
            raise RuntimeError("no archive was downloaded")
        print("  ", os.path.basename(archive), os.path.getsize(archive), "bytes")

        # 🔴 THE REAL FILE INPUT, NOT A SYNTHESISED EVENT. The upload branch
        # reads `event.target.files[0]`, so anything that fakes the event tests
        # a path the reader never takes. DOM.setFileInputFiles is Chrome
        # putting a file on the input exactly as a picker would.
        control("msa-mode", "upload")
        root = ws.call("DOM.getDocument")["root"]["nodeId"]
        node = ws.call("DOM.querySelector", nodeId=root,
                       selector="#msa-file")["nodeId"]
        ws.call("DOM.setFileInputFiles", files=[archive], nodeId=node)
        cdp.wait_for(ws, """/archive/.test(
          (document.getElementById('status-message')||{}).textContent || '')""",
                     timeout=30, what="the archive to be read",
                     progress=fip.STATUS_LINE)
        print("  read back:", cdp.evaluate(
            ws, "(document.getElementById('status-message')||{}).textContent"))

        print("second fold (from the archive)…", flush=True)
        second_status = fold_and_wait(ws, 900)
        print("  ", second_status)
        second_digest, second_atoms = structure_digest(ws, downloads, "second")

        print()
        print("archive kept at", downloads)
        print("first  %s  %d atoms" % (first_digest, first_atoms))
        print("second %s  %d atoms" % (second_digest, second_atoms))

        # 🔴 "trunk reused" IS THE ASSERTION, NOT THE MATCHING DIGEST. The
        # trunk cache key hashes the alignment BLOCKS (see foldWithAf3), so the
        # second fold reusing the first one's trunk is the page saying, in its
        # own words, that what came out of the archive is bit-identical to what
        # the search produced. The digest alone is weaker: two folds at one seed
        # would also match if the blocks differed only where the model shrugged.
        reused = "trunk reused" in (second_status or "")
        if not reused:
            print("FAIL: the second fold did not reuse the trunk, so the blocks"
                  " it was given differ from the ones the search produced")
        if first_digest != second_digest:
            print("FAIL: the archive did not restore the alignment it was given")
        if not reused or first_digest != second_digest:
            return 1
        print("OK: the blocks hash the same (trunk reused) and the structures match")
        return 0
    finally:
        proc.kill()
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

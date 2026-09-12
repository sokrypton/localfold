"""Drive a REAL fold in the real page, and report what the frames carry.

    python3 tools/fold-in-page.py                       # AF2 monomer
    python3 tools/fold-in-page.py --model af3 --steps 4
    python3 tools/fold-in-page.py --sequence GWSTELEK... --recycles 1

🔴 IT EXISTS BECAUSE EVERY OTHER CHECK HERE MISSES THE PATH THAT MATTERS.
tools/heatmap-panel.py loads a structure and attaches maps by hand, so it
proves the PANEL works with our format; tools/gpu/probe-af2-contacts.js runs
the head against a fold, so it proves the ARITHMETIC. Neither runs web/app.js's
own wiring, and a contact map failed to appear three times in a row with both
of those passing - once because the panel is not told by render(), once
because `viewer.objects` does not exist, once because the shard was a 404.

So this presses the page's own Fold button and reads the frames back: what
maps each frame has, how many frames there are, and what the heatmap panel is
showing. If it says a frame has no `contact`, the wiring is wrong; if it says
the frame has one and the panel does not list it, the panel is not being told.

🔴 IT NEEDS THE WEIGHTS, so it is slow and it is not part of any suite. AF2
pulls its bundle from the pinned remote unless a local model/ directory is
served, which it is here - the server's root is the repo.
"""
import argparse
import http.server
import re
import json
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

PORT, DBG = 9667, 9230
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"


MANIFESTS = "/src/reference/manifests/index.js"
REMOTE_LINE = re.compile(rb'^\s*remote:\s*"[^"]*",\s*$', re.MULTILINE)


def serve(local_weights=True):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            # 🔴 SOURCE IS NEVER CACHED, because a cached ES module looks exactly
            # like a broken feature - CLAUDE.md's own trap, from the server's
            # side. Not the weights: they are hundreds of megabytes of shards
            # and re-reading them turns a one-minute run into ten.
            if self.path.split("?")[0].endswith((".js", ".html", ".css")):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            # 🔴 AND THE AF3 BUNDLE COMES OFF THE DISK BY DEFAULT. Its manifest
            # names a `remote`, which a deployed page is right to use and a
            # check that runs on demand is not: 150 MB per run. Rewriting the
            # one module as it is served changes nothing on disk.
            if local_weights and self.path.split("?")[0] == MANIFESTS:
                source = open(os.path.join(REPO, MANIFESTS.lstrip("/")), "rb").read()
                body = REMOTE_LINE.sub(b"", source)
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# The page's own status line, polled by `cdp.wait_for` so a long fold narrates
# itself rather than ending in a timeout that names no stage.
def template_entry(text):
    """The entity-list template object a --template argument asks for.

    🔴 THE PAGE NO LONGER GUESSES THE DATABASE, so neither does this. The
    dropdown carries an explicit kind, and `pdb:1abc` / `afdb:P00533` is how a
    run says which one it means. Bare text keeps the old four-characters rule so
    existing invocations still work, which is a convenience here and is exactly
    what stopped being one in the page.
    """
    kind, _, rest = text.partition(":")
    if kind == "upload" and rest:
        # 🔴 THE FILE PICKER IS NOT THE PATH WORTH CHECKING. What an upload can
        # break is everything downstream of it - the text reaching the slot
        # builder, the chain box choosing between a file's chains - and that is
        # reached by setting what the picker would have set. `path` or
        # `path@CHAIN`.
        path, _, chain = rest.partition("@")
        with open(os.path.expanduser(path)) as handle:
            return {"kind": "upload", "text": handle.read(),
                    "filename": os.path.basename(path), "source": chain}
    if kind in ("pdb", "afdb") and rest:
        return {"kind": kind, "source": rest}
    if text == "auto":
        return {"kind": "search"}
    return {"kind": "pdb" if len(text.split("_")[0]) == 4 else "afdb",
            "source": text}


# 🔴 AND THE MODEL DIAL BESIDE IT, because the weights now download while the
# MSA search runs and the status line deliberately says nothing about them.
# Reading only the line would show a gap exactly where the parallel half of the
# work is.
# 🔴 AND THE COLOUR MODE RIDES ALONG, BECAUSE THE END STATE IS NOT THE STORY.
# The mode was set only when the fold FINISHED, so every frame drawn while the
# sampler ran came up in `auto` - which resolves to rainbow - and a probe that
# looked afterwards saw `plddt` and passed. Sampling it from the wait loop is
# what makes "frames added during diffusion still showing rainbow" visible to
# the tool rather than only to the eye.
STATUS_LINE = """(() => {
  const line = (document.getElementById('status-message')||{}).textContent || '';
  const dial = document.getElementById('model-load');
  const loading = dial && !dial.hidden ? dial.getAttribute('aria-label') : '';
  const reg = window.py2dmol_viewers || {};
  const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
  const frames = v && v.objectsData && v.objectsData[v.currentObjectName]
    ? v.objectsData[v.currentObjectName].frames.length : 0;
  const mode = v ? (v.colorMode === 'auto' ? 'auto->' + v.resolvedAutoColor : v.colorMode) : '-';
  // ...and the camera, so "the last frame is a different angle" is a number.
  const r = v && v.viewerState && v.viewerState.rotation;
  const rot = r ? ' ' + r[0].map((x) => x.toFixed(2)).join(',') : '';
  const drawn = frames > 0 ? `  [${frames}f ${mode}${rot}]` : '';
  return (loading ? line + '  ||  ' + loading : line) + drawn;
})()"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT)
    parser.add_argument("--model", default="monomer",
                        help="the value of the #model select: monomer, multimer, af3 or openbind0")
    parser.add_argument("--recycles", default="1")
    parser.add_argument("--steps", default="4", help="AF3 sampler steps")
    parser.add_argument("--url", default=None,
                        help="drive a DEPLOYED page instead of this checkout,"
                             " e.g. https://localfold.org/index.html. The"
                             " weights then come from the bundle's pinned"
                             " remote, which is a ~97 MB download per run"
                             " because each run starts a fresh profile with an"
                             " empty cache.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--throttle", type=float, default=0,
                        help="shape the page's network to this many MB/s "
                             "(0 = unthrottled); Hugging Face measures 8")
    parser.add_argument("--latency", type=float, default=30,
                        help="added round-trip latency in ms, with --throttle")
    parser.add_argument("--keep-profile", action="store_true",
                        help="reuse the Chrome profile instead of wiping it, so"
                             " the HTTP cache and the SHADER cache survive - the"
                             " second visit a real user makes, which nothing"
                             " here had ever measured. The status line is the"
                             " fold alone and `elapsedMs` includes the weights,"
                             " so the two together say which cache paid.")
    # 🔴 SINGLE SEQUENCE BY DEFAULT, because this tool is a wiring check and a
    # search is a minute of somebody else's server. `--msa-mode search` is
    # needed for `--template auto`, which has nothing to draw on without one.
    parser.add_argument("--msa-mode", default="none", choices=["none", "search"])
    # 🔴 EF2-fast's OWN "single sequence". Its evolutionary information comes
    # from a protein language model rather than an alignment, so turning ESM-C
    # off is the same ablation `--msa-mode none` is for the other models.
    parser.add_argument("--plm", default="esmc-600m",
                        choices=["esmc-600m", "esmc-300m", "none"])
    parser.add_argument("--remote-weights", action="store_true",
                        help="fetch the AF3 bundle from its pinned remote"
                             " (~150 MB) instead of ./model-af3-int5/")
    parser.add_argument("--dev-report", action="store_true",
                        help="open the footer's dev panel and print what it says")
    parser.add_argument("--session-hidden", action="store_true",
                        help="hide the tab before reloading, the other save signal")
    parser.add_argument("--session", action="store_true",
                        help="save the session, reload, restore it, read the panels back")
    parser.add_argument("--download", action="store_true",
                        help="press Download all and report the zip it wrote")
    parser.add_argument("--bar", action="store_true",
                        help="print every value the progress bar took, with the"
                             " clock beside it - a bar that stops short is a"
                             " sequence, not a final state")
    parser.add_argument("--timeline", action="store_true",
                        help="print when the model shards and the MSA search"
                             " were each on the wire, and how much they"
                             " overlapped. The two used to be strictly"
                             " sequential.")
    parser.add_argument("--ligand", default="",
                        help="a CCD code folded alongside the sequence, e.g. GOL."
                             " 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, so a fold"
                             " with one has MORE TOKENS THAN RESIDUES - which is"
                             " the case the archive refuses to infer a layout for"
                             " and the sharpest test of a session carrying"
                             " `tokens` back.")
    parser.add_argument("--job-round-trip", action="store_true",
                        help="write the archive, WIPE THE ENTITY ROWS, drop the"
                             " archive back on the upload box, and compare what"
                             " comes back with what folded. 🔴 THE WIPE IS"
                             " THE TEST: the rows are still on screen from the"
                             " fold, so 'they match' is true of a page that read"
                             " nothing at all.")
    parser.add_argument("--modify", default="",
                        help="a modified residue on the protein entity, as"
                             " CODE@POSITION - e.g. SEP@3, counting from 1."
                             " Comma-separate for several. 🔴 A MODIFIED RESIDUE"
                             " IS SEVERAL TOKENS FOR ONE RESIDUE, so like a"
                             " ligand it makes tokens outnumber residues - and"
                             " unlike a ligand it does so INSIDE a chain, which"
                             " is the layout `tokenIdentifiers` refuses to"
                             " infer. It is also INPUT, so a session that drops"
                             " it describes a different job.")
    parser.add_argument("--template", default="",
                        help="a PDB entry (1abc, 1abc_A), a UniProt accession,"
                             " `auto` to use what the MSA search finds, or"
                             " pdb:ID / afdb:ID to name the database outright,"
                             " or upload:PATH[@CHAIN] for a local structure."
                             " Goes to the network either way.")
    parser.add_argument("--then-sequence", default=None,
                        help="fold a SECOND time on this sequence, which is a"
                             " fresh fold rather than a continuation")
    parser.add_argument("--then-recycles", default=None,
                        help="fold a SECOND time at this recycle count, which is"
                             " what the rewind-and-continue path does")
    args = parser.parse_args()

    httpd = serve(local_weights=args.url is None and not args.remote_weights)
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_fold_profile", keep=args.keep_profile)
    try:
        ws.call("Page.enable")
        # 🔴 THE DEV SERVER IS 371 MB/s AND HUGGING FACE IS 8, SO EVERY WEIGHT
        # NUMBER TAKEN HERE IS 5-12x OPTIMISTIC - which docs/PERF.md records and
        # nothing could act on, because there was no way to ask this harness for
        # a user's link. `--throttle=<MB/s>` shapes the whole page's network
        # through CDP, which is the only place a first visit can be measured at
        # all: it is the largest single cost a user pays and the one no local
        # timing can see.
        if args.throttle:
            ws.call("Network.enable")
            ws.call("Network.emulateNetworkConditions",
                    offline=False, latency=float(args.latency),
                    downloadThroughput=float(args.throttle) * 1000 * 1000,
                    uploadThroughput=float(args.throttle) * 1000 * 1000)
        ws.call("Page.navigate",
                url=args.url or ("http://127.0.0.1:%d/index.html" % PORT))
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the page to finish loading")

        # 🔴 ONE ENTITY PER CHAIN, because the page validates one sequence per
        # entity. Typing a colon-joined complex into a single field is rejected
        # with "One sequence per entity - use Add entity for another chain", the
        # Fold click does nothing, and the tool waits out its whole timeout: two
        # 25-minute runs went that way. The entity list has its own API, which
        # is also how the templates below are set, so a complex goes in through
        # that rather than through the DOM.
        #
        # 🔴 THE SEQUENCE FIELD IS CONTENTEDITABLE, NOT AN INPUT, and the list
        # reads it on `input` - so setting textContent alone leaves the entity
        # empty and Fold does nothing. That still applies to the single-chain
        # path, which is left alone because it is what every existing run uses.
        chains = [c for c in args.sequence.split(":") if c.strip()]
        if len(chains) > 1:
            print("chains:", cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              if (!list) return 'no entity list';
              const template = list.read().find((e) => e.type === 'protein')
                ?? { type: 'protein', copies: 1, modifications: [] };
              list.set(%s.map((value) => ({ ...template, value, copies: 1 })));
              return JSON.stringify(list.read().map((e) => e.value.length));
            })()""" % json.dumps(chains)))
        else:
            cdp.evaluate(ws, """(() => {
              const field = document.querySelector('.entity-field [contenteditable],'
                + ' .entity-field textarea, .entity-field input');
              if (!field) return 'no field';
              if ('value' in field && field.tagName !== 'DIV') field.value = %s;
              else field.textContent = %s;
              field.dispatchEvent(new Event('input', { bubbles: true }));
              return field.tagName;
            })()""" % (json.dumps(args.sequence), json.dumps(args.sequence)))

        # 🔴 THE MODIFICATION GOES ON THE ENTITY, like the template below, and
        # not on a control: the popup behind the row's ⋮ writes into the same
        # entity model, so the list's own API is the shape a paste would take.
        if args.modify:
            mods = []
            for piece in args.modify.split(","):
                code, _, position = piece.strip().partition("@")
                mods.append({"code": code.strip().upper(), "position": int(position)})
            print("modify:", cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              if (!list) return 'no entity list';
              const entities = list.read();
              const protein = entities.find((e) => e.type === 'protein');
              if (!protein) return 'no protein entity';
              protein.modifications = %s;
              list.set(entities);
              return JSON.stringify(list.read().map(
                (e) => (e.modifications ?? []).map((m) => m.code + '@' + m.position)));
            })()""" % json.dumps(mods)))

        if args.ligand:
            print("ligand:", cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              if (!list) return 'no entity list';
              const entities = list.read();
              entities.push({ type: 'ligand', value: %s, copies: 1, modifications: [] });
              list.set(entities);
              return JSON.stringify(list.read().map((e) => e.type + ':' + e.value));
            })()""" % json.dumps(args.ligand)))

        # 🔴 THE TEMPLATE GOES ON THE ENTITY, NOT ON A CONTROL. It lives behind
        # the row's ⋮ beside the modified residues, in the entity model that
        # web/entities.js expands - so it is set through the list's own API,
        # the way a paste would set it, rather than by poking at the popup.
        if args.template:
            print("template:", cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              if (!list) return 'no entity list';
              const entities = list.read();
              const protein = entities.find((e) => e.type === 'protein');
              if (!protein) return 'no protein entity';
              protein.template = %s;
              list.set(entities);
              return JSON.stringify(list.read().map((e) => e.template || null));
            })()""" % json.dumps(template_entry(args.template))))

        cdp.evaluate(ws, """(() => {
          const set = (id, value) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          };
          set('model-family', %s);
          set('recycles', %s);
          set('af3-count', %s);
          set('msa-mode', %s);
          set('plm-mode', %s);
          // 🔴 THE TERMS DIALOG WOULD OTHERWISE EAT THE CLICK. AlphaFold 3's
          // parameters are gated behind an acknowledgement, and it opens in
          // front of `predict` - so without this the Fold press opens a modal,
          // nothing folds, and the tool waits out its whole timeout with the
          // status line never moving. That is the exact failure mode the
          // progress printing was added to diagnose, so it is worth naming.
          //
          // A developer driving their own page is not a user being asked, and
          // the deploy-side gate (LOCALFOLD_ACCEPT_MODEL_TERMS) is untouched.
          try {
            localStorage.setItem('localfold.modelTerms.alphafold3', 'accepted');
          } catch (e) { /* asked again, which the dialog check covers */ }
        })()""" % (json.dumps(args.model), json.dumps(args.recycles),
                   json.dumps(args.steps), json.dumps(args.msa_mode),
                   json.dumps(args.plm)))
        time.sleep(0.5)
        print("controls:", cdp.evaluate(ws, """(() => {
          const v = (id) => (document.getElementById(id) || {}).value;
          return JSON.stringify({ model: v('model-family'), recycles: v('recycles'),
            msa: v('msa-mode'), plm: v('plm-mode'), af3count: v('af3-count') });
        })()"""))

        # 🔴 THE STATUS LINE IS SAMPLED, NOT GLANCED AT. A line that alternates
        # between two sentences reads as flicker, and neither a screenshot nor
        # a reading after the fold can see it - only the sequence of values it
        # took while the fold ran. Every distinct value is recorded, with the
        # numbers blanked, so the SHAPES it took can be counted.
        cdp.evaluate(ws, """(() => {
          window.__statusLog = [];
          const el = document.getElementById('status-message');
          const shape = (s) => s.replace(/[0-9]+(\\.[0-9]+)?/g, '#');
          new MutationObserver(() => {
            const now = shape(el.textContent || '');
            const log = window.__statusLog;
            if (log.length === 0 || log[log.length - 1] !== now) log.push(now);
          }).observe(el, { childList: true, characterData: true, subtree: true });
        })()""")
        # 🔴 AND THE BAR IS SAMPLED THE SAME WAY, because "it only goes half
        # way" is a claim about a SEQUENCE of values and nothing that reads the
        # page after the fold can see it - by then the bar is back to idle.
        # Polled rather than observed: `bar.value = x` is an IDL write, and a
        # MutationObserver on the element sees nothing at all.
        cdp.evaluate(ws, """(() => {
          window.__barLog = [];
          const bar = document.getElementById('progress');
          const started = performance.now();
          setInterval(() => {
            const state = bar.dataset.state || '';
            const value = bar.hasAttribute('value') ? bar.value : null;
            // ...and the download dial beside it, which is a second animation
            // with a second calibration and the same way of being wrong.
            const node = document.getElementById('model-load');
            // 🔴 THE ARC AS DRAWN, NOT THE LABEL BESIDE IT. The two can
            // disagree: the fill carries a .2s transition on stroke-dashoffset,
            // so on a fast load the label reads 100% while the arc is still
            // sweeping - and it is the arc that gets hidden mid-sweep.
            const fillNode = node ? node.querySelector('.model-load-fill') : null;
            const arc = fillNode === null || node.hidden ? null : (() => {
              const cs = getComputedStyle(fillNode);
              const array = parseFloat(cs.strokeDasharray);
              const offset = parseFloat(cs.strokeDashoffset);
              return Number.isFinite(array) && array > 0
                ? Number((1 - offset / array).toFixed(2)) : null;
            })();
            const dial = node && !node.hidden
              ? (node.getAttribute('aria-label') || '') + ' [arc ' + arc + ']' : '';
            const log = window.__barLog;
            const last = log[log.length - 1];
            if (last !== undefined && last.value === value
              && last.state === state && last.dial === dial) return;
            log.push({ at: Math.round(performance.now() - started), value, state, dial });
          }, 50);
        })()""")
        # ...stamped before the click so the timeline can tell what the fold
        # caused from what the page had already fetched. See --timeline.
        cdp.evaluate(ws, "window.__foldClickedAt = performance.now();"
                         " document.getElementById('predict').click()")
        cdp.wait_for(ws, """(() => {
          const s = document.getElementById('status-message');
          const text = s ? s.textContent : '';
          return /done|complete|finished|s\\b/i.test(text)
            && document.getElementById('downloads').style.display !== 'none';
        })()""", timeout=args.timeout, what="the fold to finish",
                     progress=STATUS_LINE)
        time.sleep(1.5)

        # 🔴 THE OVERLAP IS MEASURED, NOT ASSERTED. The weights and the MSA
        # search were serialised - the download ran inside the fold, which runs
        # after the alignment - and the fix is invisible from the outside
        # except as a total. Resource timing says it directly: when the model
        # shards were on the wire, and when the search was.
        if args.bar:
            print("bar:", cdp.evaluate(ws, """(() => JSON.stringify(
              (window.__barLog || []).map((e) =>
                [e.at, e.value === null ? e.state : Number(e.value.toFixed(3)),
                 e.dial || undefined])))()"""))

        if args.timeline:
            print("timeline:", cdp.evaluate(ws, r"""(() => {
              // 🔴 ONLY WHAT THE FOLD ASKED FOR, MEASURED FROM THE CLICK.
              // The first version of this filtered the whole resource list by
              // name and reported a 1.2s "overlap" - which the unchanged tree
              // reproduced exactly, because both patterns were matching MODULE
              // fetches from page load: `src/input/mmseqs2-api.js` is a match
              // for /mmseqs/, and the local weights directory is probed before
              // the button is ever pressed. A span that starts at 66ms is not
              // a span of anything a click caused.
              const since = window.__foldClickedAt ?? 0;
              const span = (match) => {
                const hits = performance.getEntriesByType('resource')
                  .filter((entry) => match.test(entry.name) && entry.startTime >= since);
                if (hits.length === 0) return null;
                return {
                  requests: hits.length,
                  start: Math.round(Math.min(...hits.map((e) => e.startTime))),
                  end: Math.round(Math.max(...hits.map((e) => e.responseEnd))),
                };
              };
              // Shard files and the manifest beside them, never a .js module.
              const model = span(/(weights-\d+[^/]*\.bin|manifest\.json)(\?|$)/);
              const search = span(/^https?:\/\/[^/]*colabfold/);
              const overlap = model && search
                ? Math.round(Math.min(model.end, search.end)
                             - Math.max(model.start, search.start))
                : null;
              return JSON.stringify({ model, search, overlapMs: overlap });
            })()"""))

        print("frames:", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const name = v.currentObjectName;
          const frames = v.objectsData[name].frames;
          return JSON.stringify({
            object: name,
            frames: frames.length,
            perFrame: frames.map((f) => ({
              name: f.name,
              maps: f.maps ? Object.keys(f.maps) : [],
              pae: f.pae ? f.pae.length : 0,
            })),
          });
        })()"""))
        # 🔴 THE B-FACTOR IS THE COLOUR, so read it rather than the frame count.
        # A frame nothing has measured must carry zero - the pLDDT ramp paints
        # that red - and only the finished structure may carry real values.
        print("bfactor:", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const frames = v.objectsData[v.currentObjectName].frames;
          // 🔴 A FRAME KEEPS A FLAT `plddts`, NOT PER-ATOM OBJECTS. Reading
          // it wrong reports zero for everything, which is exactly what an
          // uncoloured frame looks like - so the check would have passed
          // whatever the page did.
          return JSON.stringify(frames.map((f) => {
            const b = Array.from(f.plddts || []);
            if (b.length === 0) return { name: f.name, plddts: 'missing' };
            return { name: f.name, n: b.length, min: Math.min(...b).toFixed(1),
              max: Math.max(...b).toFixed(1) };
          }));
        })()"""))
        # 🔴 THE VALUES BEING PRESENT IS NOT THE COLOUR BEING APPLIED. `plddts`
        # on a frame says the B-factors parsed; what decides what is drawn is
        # the renderer's colour SCHEME, and a page can set it and have py2Dmol
        # set it back. Reported as "still not seeing colors, though certainty is
        # showing up in the status" - with the B-factor probe above passing.
        # 🔴 AND THE MODE DURING THE FOLD, NOT ONLY AFTER IT. The end-state
        # probe passed while every frame drawn WHILE the sampler ran was in
        # rainbow, because the page set the mode only once the fold finished.
        # `liveColour` is sampled from the wait loop below.
        print("colour:", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const entry = reg[Object.keys(reg)[0]];
          const v = entry && entry.renderer;
          if (!v) return 'no viewer';
          const keys = Object.keys(v).filter((k) => /colou?r|scheme/i.test(k));
          const state = {};
          for (const k of keys) {
            const value = v[k];
            state[k] = (typeof value === 'object' && value !== null)
              ? Object.keys(value).slice(0, 8) : String(value);
          }
          return JSON.stringify({
            state,
            hasSetColorScheme: typeof v.setColorScheme === 'function',
            hasColorBy: typeof v.colorBy === 'function',
          });
        })()"""))
        # 🔴 AND THE PIXELS, BECAUSE THE TWO PROBES ABOVE CANNOT SEE THE
        # COLOUR. `bfactor` says the values parsed and `colour` says the mode
        # the renderer holds; neither says what reached the canvas, and the
        # note above records this being reported with both of them passing.
        # A pLDDT ramp over a real fold is a SPREAD - blue where the model is
        # sure, orange where it is not - so what is asked is that the drawn
        # colours VARY, and that the frame the page lands on is the one with
        # the confidence head's answer on it.
        print("drawn :", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const cv = v.canvas;
          const c2 = document.createElement('canvas');
          c2.width = cv.width; c2.height = cv.height;
          c2.getContext('2d').drawImage(cv, 0, 0);
          const d = c2.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
          const seen = new Map();
          for (let i = 0; i < d.length; i += 4) {
            if (d[i] > 240 && d[i + 1] > 240 && d[i + 2] > 240) continue;
            const key = (d[i] >> 4) + ',' + (d[i + 1] >> 4) + ',' + (d[i + 2] >> 4);
            seen.set(key, (seen.get(key) || 0) + 1);
          }
          const frames = v.objectsData[v.currentObjectName].frames;
          const at = v.currentFrame;
          const bs = Array.from(frames[at]?.plddts || []);
          // 🔴 AND THE RAMP IS FOLLOWING THE VALUES, WHICH A COLOUR COUNT
          // CANNOT SAY. Shading and outlines make hundreds of buckets out of
          // ONE scheme, so "many colours" is not "coloured by pLDDT". What
          // separates them is that the second-to-last frame is UNMEASURED -
          // written with a zero B-factor on purpose - so under a pLDDT ramp it
          // must not look like the finished structure, whose pLDDT here spans
          // 57 to 81. Under chain colours the two are identical.
          // 🔴 THE CANVAS TRAILS THE RENDER, so one read after one render is
          // not the picture. Measured with CDP screenshots on py2Dmol's own
          // page: stepping 15 -> 14 -> 15, the third capture returns the
          // SECOND frame's bytes while the palette is provably right at every
          // step - so a synchronous read reports the previous frame's colours
          // and "the colour did not change" is indistinguishable from "I read
          // too early". This reads, waits a frame, reads again and keeps the
          // second, which is the settled one.
          const sample = () => {
            const cv2 = v.canvas;
            const t = document.createElement('canvas');
            t.width = cv2.width; t.height = cv2.height;
            t.getContext('2d').drawImage(cv2, 0, 0);
            const q = t.getContext('2d').getImageData(0, 0, cv2.width, cv2.height).data;
            let r = 0, g = 0, b = 0, n = 0;
            for (let i = 0; i < q.length; i += 4) {
              if (q[i] > 240 && q[i + 1] > 240 && q[i + 2] > 240) continue;
              r += q[i]; g += q[i + 1]; b += q[i + 2]; n += 1;
            }
            return n === 0 ? null : [Math.round(r / n), Math.round(g / n), Math.round(b / n), n];
          };
          const mean = () => { sample(); return sample(); };
          const last = frames.length - 1;
          v.setFrame(last); v.render('probe-last');
          const inkLast = mean();
          v.setFrame(Math.max(0, last - 1)); v.render('probe-prev');
          const inkPrev = mean();
          v.setFrame(last); v.render('probe-back');
          const prevB = Array.from(frames[Math.max(0, last - 1)]?.plddts || []);
          return JSON.stringify({
            frame: at, lastFrame: last,
            onLast: at === last,
            plddtOfDrawnFrame: bs.length === 0 ? 'missing'
              : {min: Math.min(...bs).toFixed(1), max: Math.max(...bs).toFixed(1)},
            plddtOfPrev: prevB.length === 0 ? 'missing'
              : {min: Math.min(...prevB).toFixed(1), max: Math.max(...prevB).toFixed(1)},
            inkedBuckets: seen.size,
            meanInkLast: inkLast, meanInkPrev: inkPrev,
            // the unmeasured frame and the finished one must not paint the same
            followsPlddt: !(inkLast && inkPrev
              && Math.abs(inkLast[0] - inkPrev[0]) < 6
              && Math.abs(inkLast[1] - inkPrev[1]) < 6
              && Math.abs(inkLast[2] - inkPrev[2]) < 6),
          });
        })()"""))
        print("panel :", cdp.evaluate(ws, """(() => {
          const c = document.getElementById('heatmapContainer');
          return JSON.stringify({
            visible: c && getComputedStyle(c).display !== 'none',
            tabs: [...document.querySelectorAll('#heatmapContainer [role="tab"]')]
              .map((t) => t.dataset.mapKey),
          });
        })()"""))
        print("status:", cdp.evaluate(ws,
            "(document.getElementById('status-message')||{}).textContent"))
        # 🔴 THE STATUS LINE ROUNDS TO WHOLE SECONDS, which is a fine thing to
        # show a reader and useless for measuring a change: a 300 ms speedup on
        # a 4 s fold moves nothing on it. __foldClickedAt is stamped just before
        # the click for --timeline, so the same stamp gives the fold's own wall.
        print("elapsedMs:", cdp.evaluate(ws,
            "Math.round(performance.now() - (window.__foldClickedAt || 0))"))
        # 🔴 WHERE THE WEIGHT LOAD WENT, which on a RETURNING visit is the whole
        # gap between the click and the fold's own clock - 2.8 s of an OpenDDE
        # page with no shard on the wire. See af3LoadMilliseconds.
        print("weightPhases:", cdp.evaluate(ws, """(async () => {
          try {
            const m = await import('/web/af3-model.js');
            const s = await import('/src/reference/http-tensor-store.js');
            const d = s.tensorDecodeStats || {};
            return JSON.stringify({ ...(m.af3LoadMilliseconds || {}),
              hostDecodeMs: Math.round(d.ms || 0), hostDecodeCalls: d.calls || 0 });
          } catch (error) { return 'unavailable: ' + error.message; }
        })()""", await_promise=True))
        if args.dev_report:
            # 🔴 THROUGH THE BUTTON, NOT THE MODULE. The panel is built the
            # first time it is opened, so calling devReport() directly would
            # check the log and not the thing a reader presses.
            print("dev    :", cdp.evaluate(ws, """(() => {
              const button = document.getElementById('dev-toggle');
              if (button === null) return 'no dev button in the footer';
              button.click();
              const panel = document.getElementById('dev-panel');
              if (panel === null) return 'the button built no panel';
              if (panel.hidden) return 'the panel stayed hidden';
              return panel.querySelector('pre').textContent;
            })()"""))
        shapes = cdp.evaluate(ws, """(() => {
          const log = window.__statusLog || [];
          const seen = [];
          for (const s of log) if (seen.indexOf(s) < 0) seen.push(s);
          return JSON.stringify({ changes: log.length, shapes: seen });
        })()""")
        print("statusln:", shapes)
        print("map1   :", cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const h = v.heatmapRenderer;
              const maps = h && h.maps;
              const sum = (m) => {
                if (!m || !m.data) return null;
                let a = 0;
                for (let i = 0; i < m.data.length; i += 1) a = (a + m.data[i] * (i % 7 + 1)) % 1000000007;
                return a;
              };
              const c = document.getElementById('heatmapContainer');
              return JSON.stringify({
                visible: !!(c && getComputedStyle(c).display !== 'none'),
                keys: maps ? Object.keys(maps) : [],
                contact: maps ? sum(maps.contact) : null,
              });
            })()"""))
        # 🔴 THE CAMERA IS PART OF THE ANSWER. addFrame recentres viewerState on
        # the centroid of every frame the object holds, so a rewind that clears
        # the frames and re-adds them walks the camera - which is what "the view
        # jumps" is. Read before and after, and compare.
        camera_before = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const s = v.viewerState || {};
              const o = v.objectsData[v.currentObjectName] || {};
              const r = (x) => x === null || x === undefined ? null
                : (typeof x === 'number' ? Number(x.toFixed(3)) : x);
              return JSON.stringify({
                zoom: r(s.zoom), focal: r(s.focalLength),
                center: s.center ? [r(s.center.x), r(s.center.y), r(s.center.z)] : null,
                objCenter: (o.center || []).map(r),
                extent: r(o.maxExtent),
                rot0: (s.rotation && s.rotation[0] || []).map(r),
              });
            })()""")
        print("camera1:", camera_before)

        # 🔴 THE DOWNLOAD BUTTON IS A CODE PATH NOTHING RAN. It builds the
        # archive from `lastPrediction`, so every field a fold forgets to store
        # fails HERE and nowhere else - and it failed exactly that way on
        # EF2-fast, whose prediction carries no confidence object, with "Cannot
        # read properties of undefined (reading 'length')". The handler catches
        # its own error and writes it to the status line, so a click that
        # produces no blob and a changed status IS the failure.
        if args.download:
            archive = json.loads(cdp.evaluate(ws, """(async () => {
              // ...the blob is kept and read BACK, because "a zip was written"
              // is not "the fold's numbers are in it". web/zip.js is a reader
              // as well as a writer, which is why it is one file.
              const blobs = [];
              const made = URL.createObjectURL;
              URL.createObjectURL = (blob) => { blobs.push(blob); return made.call(URL, blob); };
              const before = document.getElementById('status')?.textContent ?? '';
              document.getElementById('download-all').click();
              await new Promise((done) => setTimeout(done, 3000));
              URL.createObjectURL = made;
              const after = document.getElementById('status')?.textContent ?? '';
              const out = { size: blobs[0]?.size ?? null,
                            changed: before !== after ? after : null };
              if (blobs[0] !== undefined) {
                const { readZip } = await import('/web/zip.js');
                const files = await readZip(new Uint8Array(await blobs[0].arrayBuffer()));
                out.members = [...files.keys()];
                const full = [...files.keys()].find((k) => k.endsWith('full_data_0.json'));
                const data = full === undefined ? {} : JSON.parse(files.get(full));
                out.fullData = Object.keys(data);
                out.contactRows = data.contact_probs?.length ?? null;
                out.readme = files.get('README.md');
                const req = [...files.keys()].find((k) => k.endsWith('job_request.json'));
                out.jobRequest = req === undefined ? null : files.get(req);
                const sum = [...files.keys()].find((k) => k.endsWith('summary_confidences_0.json'));
                out.summary = sum === undefined ? null : files.get(sum);
              }
              return JSON.stringify(out);
            })()""", await_promise=True))
            print("archive:", archive)
        # 🔴 INDEXEDDB AND py2Dmol'S SESSION EXIST ONLY IN A BROWSER, so this
        # is the whole gate on saving one: fold, read the record back out of
        # the real database, RELOAD, and restore it. "A record was written" is
        # not "the session comes back", the same distinction --download draws
        # for the archive.
        # 🔴 THE ARCHIVE'S OWN PROMISE, CHECKED. Its README tells the reader to
        # drop the .zip back on the page to fold again, and until the job
        # reader existed that restored the ALIGNMENT and nothing else - the
        # sequence, the copies, the ligands, the modifications and the seed all
        # had to be retyped. So: fold, write the archive, wipe the rows, drop
        # the archive back, and compare the entity list with what folded.
        #
        # 🔴 THE ROWS ARE WIPED FIRST, or the check passes on a page that read
        # nothing: the entities are still on screen from the fold that just
        # ran, and "they match afterwards" is true of a no-op.
        if args.job_round_trip:
            print("round trip:", cdp.evaluate(ws, """(async () => {
              const list = window.__entityList;
              const seedInput = document.getElementById('random-seed');
              const before = { entities: list.read(), seed: seedInput?.value ?? null };
              const blobs = [];
              const made = URL.createObjectURL;
              URL.createObjectURL = (blob) => { blobs.push(blob); return made.call(URL, blob); };
              document.getElementById('download-all').click();
              await new Promise((done) => setTimeout(done, 3000));
              URL.createObjectURL = made;
              if (blobs[0] === undefined) return JSON.stringify({ error: 'no archive' });
              const bytes = new Uint8Array(await blobs[0].arrayBuffer());
              list.set([{ type: 'protein', value: 'AAAAAAAA', copies: 1, modifications: [] }]);
              if (seedInput) seedInput.value = '999';
              const input = document.getElementById('msa-file');
              const carrier = new DataTransfer();
              carrier.items.add(new File([bytes], 'fold.zip', { type: 'application/zip' }));
              input.files = carrier.files;
              input.dispatchEvent(new Event('change', { bubbles: true }));
              await new Promise((done) => setTimeout(done, 1500));
              const after = { entities: list.read(), seed: seedInput?.value ?? null };
              // 🔴 COMPARED ON THE FIELDS THE JOB FILE CAN CARRY, not on the
              // whole row: a restored row has no `template.origin` and no
              // coverage status, which are discovered when a template is
              // FETCHED and are not part of the job. Comparing raw objects
              // would fail on things the format never claimed to hold.
              const shape = (entities) => entities.map((e) => ({
                type: e.type, value: e.value, copies: e.copies,
                modifications: (e.modifications ?? []).map(
                  (m) => m.code + '@' + m.position),
                template: e.template?.kind ?? 'none' }));
              return JSON.stringify({
                status: document.getElementById('status-message')?.textContent ?? '',
                zipBytes: bytes.length,
                before: shape(before.entities), after: shape(after.entities),
                same: JSON.stringify(shape(before.entities))
                      === JSON.stringify(shape(after.entities)),
                seedBefore: before.seed, seedAfter: after.seed,
                seedSame: before.seed === after.seed,
              });
            })()""", await_promise=True))

            # 🔴 AND THE OTHER DIALECT, HAND WRITTEN, because the archive only
            # ever exercises the one this page WRITES - a reader that passed
            # the round trip could still be blind to every file an AlphaFold 3
            # pipeline produces, which is half the reason for reading JSON at
            # all. Fed as a .json file to the same box.
            print("open dialect:", cdp.evaluate(ws, """(async () => {
              const list = window.__entityList;
              const seedInput = document.getElementById('random-seed');
              const mode = document.getElementById('msa-mode');
              mode.value = 'search'; mode.dispatchEvent(new Event('change'));
              const drop = async (text, name) => {
                const input = document.getElementById('msa-file');
                const carrier = new DataTransfer();
                carrier.items.add(new File([text], name, { type: 'application/json' }));
                input.files = carrier.files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                await new Promise((done) => setTimeout(done, 900));
                return document.getElementById('status-message')?.textContent ?? '';
              };
              // \U0001f534 BOTH `dialect` AND `version`, which is upstream's own rule
              // and which this fixture broke - it carried a version with no
              // dialect beside it, so it was a file AlphaFold 3 itself would
              // refuse. The same flaw was in test/job-json.test.js's helper.
              const job = { name: 'pipeline', modelSeeds: [1234],
                dialect: 'alphafold3', version: 2,
                sequences: [
                  { ligand: { id: 'C', ccdCodes: ['ATP'] } },
                  { protein: { id: ['A', 'B'], sequence: 'ACDEFGHIKLMNPQRSTVWY',
                               unpairedMsa: '', pairedMsa: '' } }] };
              const loaded = await drop(JSON.stringify(job), 'job.json');
              const after = list.read().map((e) => e.type + ':' + e.value + 'x' + e.copies);
              const out = { loaded, after, msaMode: mode.value,
                            seed: seedInput?.value ?? null };
              // 🔴 AND A REFUSAL LEAVES THE ROWS ALONE. A file naming chemistry
              // this page does not build must not half-load: the sequence in it
              // folds perfectly well without the bonds, which is exactly the
              // silent wrong answer the reader exists to prevent.
              const bad = { ...job, bondedAtomPairs: [[['A', 1, 'CA'], ['B', 1, 'CA']]] };
              out.refusal = await drop(JSON.stringify(bad), 'bad.json');
              out.rowsAfterRefusal =
                list.read().map((e) => e.type + ':' + e.value + 'x' + e.copies);
              out.unchanged = JSON.stringify(out.after) === JSON.stringify(out.rowsAfterRefusal);
              // 🔴 AND ONE OF AlphaFold 3'S OWN FILES, THROUGH THE REAL BOX.
              // test/af3-example-jobs.test.js reads all fourteen, but it calls
              // the reader directly - which says nothing about the file input,
              // the handler, or the rows being repainted. This is the same
              // file arriving the way a person would send it.
              const real = await (await fetch(
                '/tools/fixtures/af3-jobs/tetr_dimer_tetracycline.json')).text();
              out.exampleStatus = await drop(real, 'tetr_dimer_tetracycline.json');
              out.exampleRows =
                list.read().map((e) => e.type + ':' + e.value.length + 'x' + e.copies);
              // 🔴 AND A REAL AlphaFold SERVER ARCHIVE, WHICH DEEPMIND WROTE.
              // tools/fixtures/fold_2026_09_01_10_17.zip is the file this whole
              // format was reverse engineered from, and until now nothing ever
              // fed it BACK to the page - so the reader was checked against the
              // archive we write, which is the same source as the reader. It
              // carries two chains, a job request, and four a3m blocks.
              const zip = await (await fetch(
                '/tools/fixtures/fold_2026_09_01_10_17.zip')).arrayBuffer();
              const box = document.getElementById('msa-file');
              const held = new DataTransfer();
              held.items.add(new File([new Uint8Array(zip)], 'server.zip',
                                      { type: 'application/zip' }));
              box.files = held.files;
              box.dispatchEvent(new Event('change', { bubbles: true }));
              await new Promise((done) => setTimeout(done, 2500));
              out.serverArchive =
                document.getElementById('status-message')?.textContent ?? '';
              out.serverRows =
                list.read().map((e) => e.type + ':' + e.value.length + 'x' + e.copies);
              out.serverSeed = seedInput?.value ?? null;
              return JSON.stringify(out);
            })()""", await_promise=True))

        if args.session:
            probe = cdp.evaluate(ws, """(async () => {
              const out = { build: typeof window.buildViewerState,
                            load: typeof window.loadViewerState,
                            idb: typeof indexedDB };
              try {
                const state = window.buildViewerState();
                out.built = state !== null && state !== undefined;
                out.objects = (state?.objects ?? []).length;
                out.json = JSON.stringify(state).length;
                // 🔴 IndexedDB STORES A STRUCTURED CLONE, NOT JSON. A value that
                // stringifies fine can still be unclonable, and `put` throws
                // synchronously when it is - which is the one failure a
                // save-and-forget wrapper turns into silence.
                try { structuredClone(state); out.cloneable = true; }
                catch (e) { out.cloneable = false; out.cloneError = String(e).slice(0, 200); }
              } catch (e) { out.buildError = String(e).slice(0, 200);
                             out.stack = String(e.stack ?? '').slice(0, 900); }
              return JSON.stringify(out);
            })()""", await_promise=True)
            print("api   :", probe)
            # 🔴 NO visibilitychange IS DRIVEN HERE, ON PURPOSE. Driving one
            # made this gate green while the page was broken: a reader who
            # folds and then presses reload never hides the tab, so the only
            # save that ran was the one at fold completion - and that one
            # captures ONE frame, because the trajectory lands afterwards.
            # The gate reloads the way a reader does. `--session-hidden` is
            # the other arm, for the tab that really is hidden first.
            if args.session_hidden:
                cdp.evaluate(ws, """(() => {
                  Object.defineProperty(document, 'visibilityState',
                    { configurable: true, get: () => 'hidden' });
                  document.dispatchEvent(new Event('visibilitychange'));
                  return 1;
                })()""")
                time.sleep(3)
            saved = json.loads(cdp.evaluate(ws, "(async () => {\n              const { readSession } = await import('/web/fold-session.js');\n              const state = await readSession();\n              if (!state) return JSON.stringify({ saved: false });\n              const objects = state.objects || [];\n              return JSON.stringify({\n                saved: true,\n                version: state.version,\n                objects: objects.map((o) => o.name),\n                // 🔴 THE WHOLE TRAJECTORY, which is the point of reusing\n                // py2Dmol's own session: our archive carried the answer alone.\n                frames: objects[0]?.frames?.length ?? 0,\n                framePae: objects[0]?.frames?.some((f) => f.pae !== undefined),\n                frameMaps: objects[0]?.frames?.some(\n                  (f) => f.maps && Object.keys(f.maps).length > 0),\n                hasCamera: state.viewer_state?.rotation_matrix !== undefined,\n                colorMode: state.viewer_state?.color_mode,\n                // ...and the half py2Dmol does not know about.\n                job: state.localfold ? {\n                  stem: state.localfold.stem,\n                  model: state.localfold.model,\n                  residues: state.localfold.residues,\n                  plddt: state.localfold.confidence?.meanPlddt,\n                  msaOrigin: state.localfold.msaOrigin,\n                  framesAtSave: state.localfold.framesAtSave,\n                } : null,\n                bytes: JSON.stringify(state).length,\n                gzip: await (async () => {\n                  // 🔴 MEASURED, NOT ASSUMED. The payload is rounded decimal\n                  // coordinates repeated over every frame of a trajectory, which\n                  // is about as compressible as text gets - but how much is a\n                  // number, and CompressionStream is in the browser already.\n                  if (typeof CompressionStream !== \'function\') return null;\n                  const json = JSON.stringify(state);\n                  const raw = new TextEncoder().encode(json);\n                  const out = new Blob([raw]).stream()\n                    .pipeThrough(new CompressionStream(\'gzip\'));\n                  const packed = new Uint8Array(await new Response(out).arrayBuffer());\n                  return { raw: raw.length, gzip: packed.length,\n                           ratio: +(raw.length / packed.length).toFixed(2) };\n                })(),\n              });\n            })()", await_promise=True))
            print("saved:", saved)
            cdp.evaluate(ws, "location.reload()")
            time.sleep(5)
            back = json.loads(cdp.evaluate(ws, '(async () => {\n              await new Promise((done) => setTimeout(done, 1200));\n              const row = document.getElementById(\'session\');\n              const offered = row !== null && !row.hidden && row.offsetParent !== null;\n              const text = document.getElementById(\'session-text\')?.textContent ?? \'\';\n              document.getElementById(\'session-restore\')?.click();\n              await new Promise((done) => setTimeout(done, 3000));\n              const reg = window.py2dmol_viewers || {};\n              const renderer = reg[Object.keys(reg)[0]]?.renderer;\n              const name = renderer?.currentObjectName;\n              const frames = renderer?.objectsData?.[name]?.frames ?? [];\n              const heat = document.getElementById(\'heatmapContainer\');\n              return JSON.stringify({\n                offered, offerText: text,\n                object: name,\n                frames: frames.length,\n                pae: frames[0]?.pae_n ?? null,\n                contact: frames[0]?.maps?.contact !== undefined,\n                scoreBox: getComputedStyle(\n                  document.getElementById(\'predictionScoresBox\')).display !== \'none\',\n                plddtCell: document.getElementById(\'metricMeanPlddt\')?.textContent ?? \'\',\n                ptmCell: document.getElementById(\'metricPtm\')?.textContent ?? \'\',\n                panelShown: heat !== null && getComputedStyle(heat).display !== \'none\',\n                panelTabs: [...document.querySelectorAll(\'#heatmapContainer [role="tab"]\')]\n                  .map((t) => t.dataset.mapKey),\n                stillOffering: !document.getElementById(\'session\').hidden,\n                downloads: (() => { const d = document.getElementById(\'downloads\');\n                  return d ? getComputedStyle(d).display : null; })(),\n                // 🔴 BOTH BUTTONS ARE PRESSED, not merely looked at. A restored\n                // session used to show them and fail on click: PDB wrote the word\n                // undefined into a file and All threw inside the archive builder.\n                // The blob is caught and read back, because \'a zip was written\'\n                // is not \'the fold is in it\'.\n                pressed: await (async () => {\n                  const blobs = [];\n                  const made = URL.createObjectURL;\n                  URL.createObjectURL = (b) => { blobs.push(b); return made.call(URL, b); };\n                  document.getElementById(\'download-pdb\')?.click();\n                  await new Promise((r) => setTimeout(r, 600));\n                  document.getElementById(\'download-all\')?.click();\n                  await new Promise((r) => setTimeout(r, 2500));\n                  URL.createObjectURL = made;\n                  const out = { blobs: blobs.length, pdbBytes: blobs[0]?.size ?? null,\n                                zipBytes: blobs[1]?.size ?? null };\n                  if (blobs[0]) { const t = await blobs[0].text();\n                    out.pdbAtoms = (t.match(/^ATOM/gm) || []).length;\n                    out.pdbUndefined = t.includes(\'undefined\'); }\n                  if (blobs[1]) { const { readZip } = await import(\'/web/zip.js\');\n                    const f = await readZip(new Uint8Array(await blobs[1].arrayBuffer()));\n                    out.members = [...f.keys()];\n                    // \U0001f534 AND THE JOB REQUEST NAMES THE MODIFICATION.\n                    // It is INPUT, and the request is the file a reader hands\n                    // back to reproduce the fold - one listing the parent\n                    // sequence alone describes a different job, silently,\n                    // since a modified residue changes no residue COUNT.\n                    const jr = [...f.keys()].find((k) => k.endsWith(\'job_request.json\'));\n                    out.requestMods = jr\n                      ? JSON.parse(f.get(jr))[0]?.sequences?.[0]?.proteinChain?.modifications ?? null\n                      : null;\n                    const fd = [...f.keys()].find((k) => k.endsWith(\'full_data_0.json\'));\n                    out.fullData = fd ? Object.keys(JSON.parse(f.get(fd))) : null;\n                    // 🔴 AND THE VALUES, NOT ONLY THE KEYS. The matrices are\n                    // rebuilt from the frames now, so \'contact_probs exists\' is\n                    // not \'contact_probs is the fold\'s\': a decode with the wrong\n                    // bounds fills the key with plausible nonsense.\n                    if (fd) { const d = JSON.parse(f.get(fd));\n                      const flat = (m) => m ? m.flat() : [];\n                      const pae = flat(d.pae), con = flat(d.contact_probs);\n                      out.paeCheck = { diag: d.pae?.[0]?.[0], far: d.pae?.[0]?.[57],\n                                       max: Math.max(...pae), min: Math.min(...pae) };\n                      out.contactCheck = { diag: d.contact_probs?.[0]?.[0],\n                                           max: Math.max(...con), min: Math.min(...con) };\n                      // 🔴 AND THE TOKEN LAYOUT, which is what a ligand breaks.\n                      // More tokens than residues is the case tokenIdentifiers\n                      // REFUSES to infer, so a session that lost `tokens` throws\n                      // here rather than writing a quietly wrong file.\n                      out.layout = { tokens: d.token_chain_ids?.length,\n                                     chains: [...new Set(d.token_chain_ids || [])],\n                                     lastResId: d.token_res_ids?.[d.token_res_ids.length - 1] };\n                      const sm = [...f.keys()].find((k) => k.endsWith(\'summary_confidences_0.json\'));\n                      if (sm) { const q = JSON.parse(f.get(sm));\n                        out.summaryCheck = { chainPtm: q.chain_ptm,\n                                             pairContact: q.chain_pair_max_contact,\n                                             ptm: q.ptm, meanPlddt: q.mean_plddt }; }\n                      out.plddtCheck = { first: d.atom_plddts?.[0],\n                                         max: Math.max(...(d.atom_plddts || [])) }; }\n                    out.readmeOmits = (f.get(\'README.md\') || \'\').includes(\'not in this archive\');\n                    // 🔴 AND THE TEMPLATES, which are INPUT: an absent array\n                    // means \'this model has no such control\' and drops the\n                    // README line entirely, so a restored fold that used one\n                    // would quietly describe a different job.\n                    out.templateMembers = [...f.keys()].filter(\n                      (k) => k.startsWith(\'templates/\'));\n                    out.readmeTemplates = ((f.get(\'README.md\') || \'\')\n                      .match(/^- templates: .*$/m) || [null])[0]; }\n                  return out; })(),\n                heat: (() => { const reg = window.py2dmol_viewers || {};\n                  const r = reg[Object.keys(reg)[0]]?.renderer;\n                  const box = document.getElementById(\'heatmapContainer\');\n                  const info = { hasRenderer: !!r?.heatmapRenderer,\n                                 hasContainer: !!r?.heatmapContainer,\n                                 rendererMaps: Object.keys(r?.heatmapRenderer?.maps || {}),\n                                 shown: (r?.shownObjects instanceof Set)\n                                   ? [...r.shownObjects] : String(r?.shownObjects),\n                                 heatName: (() => { try {\n                                   return r?.heatmapObjectName ? String(r.heatmapObjectName()) : \'no fn\'; }\n                                   catch (e) { return String(e).slice(0,80); } })(),\n                                 curObj: String(r?.currentObjectName),\n                                 curFrame: String(r?.currentFrame),\n                                 hasDataSays: (() => { try { const o = r?.objectsData?.[r.currentObjectName];\n                                   return window.Heatmap?.hasData?.(o); } catch (e) { return String(e).slice(0,60); } })(),\n                                 mapKeys: (() => { try { const o = r?.objectsData?.[r.currentObjectName];\n                                   return window.Heatmap?.mapKeysOf?.(o) ?? \'no mapKeysOf\'; }\n                                   catch (e) { return String(e).slice(0,80); } })(),\n                                 frame0Maps: (() => { const o = r?.objectsData?.[r.currentObjectName];\n                                   const f = o?.frames?.[0]; if (!f?.maps) return null;\n                                   const k = Object.keys(f.maps)[0];\n                                   const m = f.maps[k];\n                                   return { key: k, type: typeof m,\n                                            hasData: m && m.data !== undefined,\n                                            dataType: typeof m?.data,\n                                            n: m?.n }; })(),\n                                 paeType: (() => { const o = r?.objectsData?.[r.currentObjectName];\n                                   const f = o?.frames?.[0];\n                                   return { pae: Array.isArray(f?.pae) ? \'array\' : typeof f?.pae, n: f?.pae_n }; })(),\n                                 display: box ? getComputedStyle(box).display : null };\n                  try { window.Heatmap?.syncToDrawn(r);\n                        info.afterSync = box ? getComputedStyle(box).display : null; }\n                  catch (e) { info.syncError = String(e).slice(0, 140); }\n                  return info; })(),\n                status: document.getElementById(\'status-message\')?.textContent ?? \'\',\n              });\n            })()', await_promise=True))
            print("restored:", back)

            # 🔴 AND IT COMES BACK AT REST, NOT MERGED. `loadViewerState`
            # clears and then adds one object at a time, and `addObject` joins
            # each to the shown set when there IS one - so while
            # clearAllObjects left an EMPTY SET (which is Multi, on, with
            # everything switched off) every restored object joined it. This
            # gate folds ONE sequence, so a restored session that draws a set
            # rather than the resting `null` has invented a mode: the frames
            # then advance on the merged timeline and the confidence panel goes
            # (a merge of two objects has no PAE by construction). Fixed in
            # py2Dmol's clearAllObjects; asserted here because this is the only
            # place that restores a real session in a real browser.
            # 🔴 AND THE RESTORED STRUCTURE IS STILL COLOURED BY ITS pLDDT.
            # The frames come back with their B-factors, but what a reader sees
            # is the SCHEME applied to the frame the page lands on - and the
            # restored prediction's own `confidence.plddt` is recovered from
            # the frames by `matricesFromFrames`, which walked them forwards
            # and so took frame 0's deliberate zeros. Reported as the pLDDT not
            # being saved on the last frame.
            print("redrawn:", cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
              if (!v) return 'no viewer';
              const frames = v.objectsData[v.currentObjectName].frames;
              const last = frames.length - 1;
              const bs = Array.from(frames[last]?.plddts || []);
              // see the note on `mean` in the fold probe above: the canvas
              // trails the render, so the read is taken twice and the second
              // one is the answer.
              const sample = () => {
                const cv = v.canvas;
                const t = document.createElement('canvas');
                t.width = cv.width; t.height = cv.height;
                t.getContext('2d').drawImage(cv, 0, 0);
                const q = t.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
                let r = 0, g = 0, b = 0, n = 0;
                for (let i = 0; i < q.length; i += 4) {
                  if (q[i] > 240 && q[i + 1] > 240 && q[i + 2] > 240) continue;
                  r += q[i]; g += q[i + 1]; b += q[i + 2]; n += 1;
                }
                return n === 0 ? null : [Math.round(r / n), Math.round(g / n), Math.round(b / n), n];
              };
              const mean = () => { sample(); return sample(); };
              const trace = [];
              const step = (to, tag) => {
                v.setFrame(to);
                trace.push(tag + ' plddtNeedUpdate=' + v.plddtColorsNeedUpdate
                  + ' rendererPlddt=' + (v.plddts && v.plddts.length
                      ? Math.min(...v.plddts).toFixed(0) + '-' + Math.max(...v.plddts).toFixed(0)
                      : 'none')
                  + ' segs=' + (v.segmentIndices || []).length
                  + ' palette=' + (v.plddtColors ? v.plddtColors.length : 'none'));
                v.render('restored-' + tag);
                return mean();
              };
              const extra = {isPlaying: v.isPlaying, useGPU: !!v.useGPU,
                             merged: !!v.multiState?.enabled};
              const inkLast = step(last, 'last');
              const inkPrev = step(Math.max(0, last - 1), 'prev');
              // 🔴 WHICH STAGE IS STUCK. If marking the palette stale by hand
              // makes the picture follow, the invalidation is what the restore
              // lost; if only a mesh invalidate does it, the palette upload is.
              v.plddtColorsNeedUpdate = true;
              v.render('forced-palette');
              const inkForcedPalette = mean();
              if (window.py2dmolCartoonGPU) window.py2dmolCartoonGPU.invalidate();
              v.render('forced-mesh');
              const inkForcedMesh = mean();
              step(last, 'back');
              return JSON.stringify({
                frames: frames.length, colorMode: v.colorMode,
                // ...per FRAME, because "the last frame kept its values" says
                // nothing about whether the frames still differ from each
                // other - and identical ink with different values is a colour
                // that is not being recomputed, which is a different fault
                // from a value that was lost.
                perFrame: frames.map((f, i) => {
                  const a = Array.from(f.plddts || []);
                  return a.length === 0 ? i + ':none'
                    : i + ':' + Math.min(...a).toFixed(0) + '-' + Math.max(...a).toFixed(0);
                }).join(' '),
                plddtOfLast: bs.length === 0 ? 'missing'
                  : {min: Math.min(...bs).toFixed(1), max: Math.max(...bs).toFixed(1)},
                meanInkLast: inkLast, meanInkPrev: inkPrev,
                inkForcedPalette, inkForcedMesh,
                extra, trace,
                followsPlddt: !(inkLast && inkPrev
                  && Math.abs(inkLast[0] - inkPrev[0]) < 6
                  && Math.abs(inkLast[1] - inkPrev[1]) < 6
                  && Math.abs(inkLast[2] - inkPrev[2]) < 6),
              });
            })()"""))
            shown_back = (back.get("heat") or {}).get("shown")
            if shown_back != "null":
                print("FAIL: a restored session came back with an explicit"
                      " shown set (%r) where one fold was saved - the restore"
                      " has put the viewer into Multi" % (shown_back,))

            # 🔴 AND CAN THE PAGE FOLD AGAIN AFTER A RESTORE? loadViewerState
            # calls clearAllObjects, so the viewer a new fold opens into is one
            # this page did not build. The offer also describes a record the
            # next save overwrites - a row left on screen would advertise the
            # old fold while Restore brought back the new one - so it must be
            # gone once what is saved is what is on screen.
            #
            # Clicked here and polled from Python: awaiting a whole fold inside
            # one cdp.evaluate outlives the call and returns nothing at all,
            # which reads as a probe that did not run.
            print("refold:", cdp.evaluate(ws, "(() => {\n              const field = document.querySelector('.entity-field [contenteditable],'\n                + ' .entity-field textarea, .entity-field input');\n              if (!field) return 'no field';\n              const other = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGK';\n              if ('value' in field && field.tagName !== 'DIV') field.value = other;\n              else field.textContent = other;\n              field.dispatchEvent(new Event('input', { bubbles: true }));\n              const button = document.getElementById('predict');\n              if (!button) return 'no button';\n              if (button.disabled) return 'button disabled';\n              button.click();\n              return 'clicked';\n            })()"))
            cdp.wait_for(ws, "/pLDDT|certainty/.test("
                             "(document.getElementById('status-message')||{}).textContent||'')",
                         what="the second fold to finish", timeout=args.timeout)
            time.sleep(5)
            print("after :", json.loads(cdp.evaluate(ws, "(async () => {\n              const { readSession } = await import('/web/fold-session.js');\n              const now = await readSession();\n              const row = document.getElementById('session');\n              return JSON.stringify({\n                offerHidden: row === null ? null : row.hidden,\n                offerText: document.getElementById('session-text')?.textContent ?? '',\n                savedResidues: now?.localfold?.residues,\n                savedStem: now?.localfold?.stem,\n                status: document.getElementById('status-message')?.textContent ?? '',\n              });\n            })()", await_promise=True)))

        first_object = json.loads(cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          return JSON.stringify(reg[Object.keys(reg)[0]].renderer.currentObjectName);
        })()"""))

        # 🔴 THE SECOND FOLD IS THE ONE THAT REWINDS. Asking for more recycles
        # with everything else unchanged should keep the frames the earlier
        # passes already produced and append to them - not start an object over.
        # 🔴 FORGET IS A PATH TOO, AND IT HAD A BUG NO OTHER ARM COULD SEE.
        # The session lives in TWO records now - the fold, and the summary the
        # offer row reads without unpacking it - and `clearSession` deleted only
        # the first. So "Forget" removed the fold, the row stayed on screen
        # advertising it, and pressing Restore found nothing. A reload is what
        # makes it visible: the row is redrawn from the store rather than from
        # whatever the click left in memory.
        if args.session:
            print("forget:", cdp.evaluate(ws, """(async () => {
              const before = !document.getElementById('session').hidden;
              document.getElementById('session-forget').click();
              await new Promise((done) => setTimeout(done, 800));
              const hidden = document.getElementById('session').hidden;
              const rows = await new Promise((resolve) => {
                const open = indexedDB.open('localfold-session');
                open.onsuccess = () => {
                  const db = open.result;
                  const store = db.transaction('session', 'readonly')
                    .objectStore('session');
                  const keys = store.getAllKeys();
                  keys.onsuccess = () => { db.close(); resolve(keys.result); };
                  keys.onerror = () => { db.close(); resolve('error'); };
                };
                open.onerror = () => resolve('no db');
              });
              return JSON.stringify({ offeredBefore: before, hiddenAfter: hidden,
                                      keysLeft: rows });
            })()""", await_promise=True))
            ws.call("Page.reload")
            cdp.wait_for(ws, "typeof window.processFiles === 'function'",
                         what="the page to come back after forgetting")
            print("after forget:", cdp.evaluate(ws, """(async () => {
              await new Promise((done) => setTimeout(done, 1200));
              const row = document.getElementById('session');
              return JSON.stringify({
                // 🔴 THE ROW MUST BE GONE ON A FRESH PAGE, which is the whole
                // assertion: a stale summary redraws it from the store.
                offered: row !== null && !row.hidden && row.offsetParent !== null,
                text: document.getElementById('session-text')?.textContent ?? '' });
            })()""", await_promise=True))

        if args.then_recycles is not None or args.then_sequence is not None:
            if args.then_sequence is not None:
                cdp.evaluate(ws, """(() => {
                  const field = document.querySelector('.entity-field [contenteditable],'
                    + ' .entity-field textarea, .entity-field input');
                  if ('value' in field && field.tagName !== 'DIV') field.value = %s;
                  else field.textContent = %s;
                  field.dispatchEvent(new Event('input', { bubbles: true }));
                })()""" % (json.dumps(args.then_sequence), json.dumps(args.then_sequence)))
                time.sleep(0.5)
            cdp.evaluate(ws, """(() => {
              const el = document.getElementById('recycles');
              if (%s !== null) {
                el.value = %s;
                el.dispatchEvent(new Event('change', { bubbles: true }));
              }
              document.getElementById('predict').click();
            })()""" % (json.dumps(args.then_recycles), json.dumps(args.then_recycles)))
            # 🔴 THE PANEL IS SAMPLED WHILE THE NEW FOLD IS STILL IN ITS TRUNK.
            # A map left over from the PREVIOUS fold is invisible afterwards -
            # by then the new one has replaced it - so the only moment it can
            # be caught is between the click and the first recycle's contacts.
            for _ in range(6):
                print("early  :", cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const h = v.heatmapRenderer;
              const maps = h && h.maps;
              const sum = (m) => {
                if (!m || !m.data) return null;
                let a = 0;
                for (let i = 0; i < m.data.length; i += 1) a = (a + m.data[i] * (i % 7 + 1)) % 1000000007;
                return a;
              };
              const c = document.getElementById('heatmapContainer');
              return JSON.stringify({
                visible: !!(c && getComputedStyle(c).display !== 'none'),
                keys: maps ? Object.keys(maps) : [],
                contact: maps ? sum(maps.contact) : null,
              });
            })()"""))
                time.sleep(0.4)
            cdp.wait_for(ws, """(() => {
              const s = document.getElementById('status-message');
              return /done|complete|finished|s\\b/i.test(s ? s.textContent : '')
                && document.getElementById('predict').disabled === false;
            })()""", timeout=args.timeout, what="the second fold to finish",
                     progress=STATUS_LINE)
            time.sleep(1.5)
            second = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const names = Object.keys(v.objectsData);
              return JSON.stringify({
                objects: names,
                current: v.currentObjectName,
                frames: names.map((n) => [n, v.objectsData[n].frames.length,
                  v.objectsData[n].frames.map((f) => f.name)]),
              });
            })()""")
            print("2nd    :", second)
            camera_after = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const s = v.viewerState || {};
              const o = v.objectsData[v.currentObjectName] || {};
              const r = (x) => x === null || x === undefined ? null
                : (typeof x === 'number' ? Number(x.toFixed(3)) : x);
              return JSON.stringify({
                zoom: r(s.zoom), focal: r(s.focalLength),
                center: s.center ? [r(s.center.x), r(s.center.y), r(s.center.z)] : null,
                objCenter: (o.center || []).map(r),
                extent: r(o.maxExtent),
                rot0: (s.rotation && s.rotation[0] || []).map(r),
              });
            })()""")
            print("camera2:", camera_after)
            # 🔴 A REWIND MUST NOT MOVE THE CAMERA. It continues on the object
            # it already had, so the reader is looking at a view they set - and
            # _switchToObject restores a viewerState that is only ever SAVED
            # when switching AWAY from an object, so asking for the object
            # already current restored its default and reset the rotation to
            # the identity. A fresh fold is a different object and is expected
            # to orient itself, so this only applies when the second fold
            # stayed on the first one's object.
            if json.loads(second).get("current") == first_object:
                was = json.loads(camera_before)
                now = json.loads(camera_after)
                # 🔴 ORIENTATION AND ZOOM ONLY. The centre and the focal
                # length are derived from the structure - a continuation that
                # re-samples really does land somewhere else, and following it
                # is right - but the ROTATION is the reader's, and moving it is
                # what reads as the view jumping.
                for field in ("rot0", "zoom"):
                    if was[field] != now[field]:
                        print("FAIL: a rewind moved the camera's %s: %r -> %r"
                              % (field, was[field], now[field]))
                for field in ("center", "focal"):
                    if was[field] != now[field]:
                        print("note: %s followed the structure: %r -> %r"
                              % (field, was[field], now[field]))
            print("2status:", cdp.evaluate(ws,
                "(document.getElementById('status-message')||{}).textContent"))
    finally:
        proc.kill()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""index.html at real phone widths: it must FIT, not merely not overflow.

    python3 tools/mobile-layout.py            # 320, 360, 390 and the 1200 control
    python3 tools/mobile-layout.py --shot 390 # ...and write /tmp/localfold-390.png

🔴 "NO HORIZONTAL OVERFLOW" IS NOT THE TEST. Under mobile emulation a page that
cannot fit does not overflow: the LAYOUT VIEWPORT GROWS to fit it. So
`scrollWidth == innerWidth` and every overflow check passes while the phone
renders the whole page zoomed out. THE ASSERTION IS `innerWidth == the width
asked for`.

WHAT THIS RUNS ON: tools/cdp.py, because --window-size clamps at 500px and the
whole interesting band is below it. See that file.

🔴 THE 1200px CONTROL IS NOT A FORMALITY. Every rule here could as easily have
made the desktop fluid, and "it fits" passes trivially on a page that has
thrown its layout away. It asserts the 948px shell, the side-by-side columns
and the 600x600 canvas are exactly what they were.

THE PAGE IS LOADED WITH A STRUCTURE AND AN ALIGNMENT, not bare. Half the rows
this measures - the play bar, the sequence strip, the MSA header, the panel
column - are `display: none` until something is loaded, and a probe that
measures a bare page is green while they are broken. The structure is
6mrr-crystal.pdb from the repo root, loaded TWICE so the play bar exists, and
the alignment is synthesised from its own sequence - test.a3m is a different
protein, so py2Dmol matched no chain and drew a warning where the MSA panel
should have been.
"""
import argparse, base64, http.server, os, re, socketserver, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import launch, evaluate, wait_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "_mobile.html")
PORT, DBG = 9664, 9227
PROFILE = "/tmp/localfold-mobile-prof"

MEASURE = r"""(() => {
  const R = {};
  const box = (e) => e ? e.getBoundingClientRect() : null;
  const wide = (e) => e ? Math.round(e.getBoundingClientRect().width) : 0;
  R.viewport = innerWidth;
  R.scrollW = document.documentElement.scrollWidth;
  R.overflow = R.scrollW - innerWidth;
  R.shell = wide(document.querySelector('.page-width'));
  const main = document.getElementById('mainContainer');
  R.mainDir = main ? getComputedStyle(main).flexDirection : 'absent';

  const cc = document.getElementById('canvasContainer');
  const cv = document.getElementById('canvas');
  R.container = [Math.round(box(cc).width), Math.round(box(cc).height)];
  R.content = [cc.clientWidth, cc.clientHeight];
  R.canvasCSS = [Math.round(parseFloat(cv.style.width) || 0),
                 Math.round(parseFloat(cv.style.height) || 0)];
  const viewer = (window.py2dmol_viewers || {})['standalone-viewer-1'];
  R.rendererSize = viewer && viewer.renderer
    ? [viewer.renderer.displayWidth, viewer.renderer.displayHeight] : null;

  // AN OVERLAP COSTS NO WIDTH, so no size check can see one. Two rectangles,
  // intersected. This page's actions (Add entity / Fold) are in the head's own
  // flex row rather than absolutely positioned, which is what py2Dmol's were -
  // asserted anyway, because the CSS it inherits still has that rule.
  const hit = (a, b) => {
    const A = box(a), B = box(b);
    if (!A || !B) return false;
    return !(B.right <= A.left + 1 || B.left >= A.right - 1
          || B.bottom <= A.top + 1 || B.top >= A.bottom - 1);
  };
  R.titleOverlap = hit(document.querySelector('.page-head h1'),
                       document.querySelector('.fold-actions'));

  // WHAT REFUSES TO SHRINK, measured rather than guessed: give each box
  // `width: min-content` for one layout and read what it insists on. This is
  // what names the offender when the viewport has been forced wide.
  // ...against the width ASKED FOR, not innerWidth: by the time this runs the
  // layout viewport has already grown to whatever the stiffest box demanded,
  // so comparing against it finds nothing and names nothing.
  const asked = window.__asked || innerWidth;
  const stiff = [];
  document.querySelectorAll('div,section,footer,header,table,select,input,'
                          + 'button,canvas,textarea,h1,progress').forEach((e) => {
    const cs = getComputedStyle(e);
    if (cs.display === 'none' || cs.position === 'absolute') return;
    const prev = e.style.width;
    e.style.width = 'min-content';
    const mc = e.getBoundingClientRect().width;
    e.style.width = prev;
    if (mc > asked + 1) stiff.push({
      el: (e.tagName + (e.id ? '#' + e.id : '')
           + (typeof e.className === 'string' && e.className
              ? '.' + e.className.trim().split(/\s+/)[0] : '')).slice(0, 46),
      min: Math.round(mc)});
  });
  stiff.sort((a, b) => b.min - a.min);
  R.stiff = stiff.slice(0, 6);

  // WHICH CONTROLS SHARE A LINE, not how many lines there are.
  //
  // 🔴 THREE TRAPS, ONE MEASUREMENT.
  //
  //  * `align-items: center` gives items of different heights different `top`
  //    values on the SAME visual line, so counting distinct tops reports three
  //    lines for two. But `align-items: start` - which is what .entity-row uses
  //    - gives them the same top and different CENTRES, and grouping by centre
  //    reported the 52px sequence box as a line of its own. Neither edge works
  //    for both. Bands are OVERLAPPING SPANS: two boxes share a line when their
  //    vertical extents overlap by more than half of the shorter one, which is
  //    true under either alignment and false for a real wrap.
  //  * An absolutely positioned child is out of flow and belongs to no line.
  //    The modification popup hangs below its button and was read as a third
  //    line of the entity row.
  //  * And a line COUNT cannot tell [type, sequence] / [copies, options,
  //    remove] from [type] / [the rest]: both are "2 lines" and only one of
  //    them is the row that was asked for. It reports the membership.
  const bands = (sel) => {
    const p = document.querySelector(sel);
    if (!p) return null;
    const out = [];
    [...p.children].filter((e) => e.getBoundingClientRect().height > 4
                              && getComputedStyle(e).display !== 'none'
                              && getComputedStyle(e).position !== 'absolute')
                   .forEach((e) => {
      const b = e.getBoundingClientRect();
      let g = out.find((x) => {
        const shared = Math.min(x.bottom, b.bottom) - Math.max(x.top, b.top);
        return shared > 0.5 * Math.min(x.bottom - x.top, b.height);
      });
      if (!g) { g = {top: b.top, bottom: b.bottom, items: []}; out.push(g); }
      g.top = Math.min(g.top, b.top);
      g.bottom = Math.max(g.bottom, b.bottom);
      g.items.push(e.id || (e.className && typeof e.className === 'string'
                            ? '.' + e.className.trim().split(/\s+/)[0]
                            : e.tagName.toLowerCase()));
    });
    return out.map((g) => g.items);
  };
  R.rows = {head: bands('.page-head-fold'), entity: bands('.entity-row'),
            options: bands('#foldOptions'), play: bands('#controlsContainer'),
            footer: bands('footer > div')};

  // A CONTROL'S LABEL IS NOT A PARAGRAPH: a box too small for its own text
  // overflows rather than growing, because these all state a height.
  const over = (sel) => { const e = document.querySelector(sel);
    if (!e) return null;
    return {over: e.scrollHeight - e.clientHeight,
            h: Math.round(e.getBoundingClientRect().height)}; };
  R.overflowing = {fold: over('#predict'), add: over('#add-entity'),
                   type: over('.entity-type'), status: over('#status-message')};

  // 🔴 A BOX ONE CHARACTER WIDE IS NOT AN OVERFLOW AND NOT A WRAP. The entity
  // row is a five-track grid whose third track is `1fr`, so at 320 the sequence
  // field got what four fixed controls and four gaps left it - 16px, with
  // "PIA" set one letter per line - and every check above passed. The row fits,
  // the page fits, and the control the page exists for is unusable.
  R.entityWidths = {};
  const row = document.querySelector('.entity-row');
  if (row) [...row.children].forEach((e) => {
    if (getComputedStyle(e).position === 'absolute') return;
    const name = e.className && typeof e.className === 'string'
      ? e.className.trim().split(/\s+/)[0] : e.tagName.toLowerCase();
    R.entityWidths[name] = Math.round(e.getBoundingClientRect().width);
  });

  // ...and the MSA panel, which is the widest thing this page draws and is
  // another bitmap in a `width: 100%` box.
  const msa = document.getElementById('msaCanvas');
  R.msa = msa ? {css: Math.round(box(msa).width),
                 logical: Math.round(msa.width / (200 / 96)),
                 stretch: +(box(msa).width / (msa.width / (200 / 96))).toFixed(3)}
              : null;

  // ...and the entity popup, which is `min-width: 340px` and hangs off a button
  // at the right edge of the row - wider than a 320px phone all by itself.
  const pop = document.querySelector('.entity-popup');
  R.popup = pop ? {w: Math.round(box(pop).width), left: Math.round(box(pop).left),
                   right: Math.round(box(pop).right)} : null;

  // THE SEQUENCE STRIP IS A BITMAP, undistorted only when its backing store
  // and its CSS box agree; `width: 100%` against a store sized once is a
  // horizontal scale factor and the letters wear it.
  const sq = document.getElementById('sequenceCanvas');
  const sbox = document.getElementById('sequence-viewer-container');
  R.strip = sq ? {css: Math.round(box(sq).width),
                  logical: Math.round(sq.width / (200 / 96)),
                  stretch: +(box(sq).width / (sq.width / (200 / 96))).toFixed(3),
                  boxH: sbox ? Math.round(box(sbox).height) : 0,
                  share: sbox ? +(box(sbox).height / innerHeight).toFixed(2) : 0}
             : null;

  const rh = document.querySelector('#canvasContainer .resize-handle');
  R.resizeHandle = rh ? getComputedStyle(rh).display : 'absent';

  // ...and the panel column, which is 340px of fixed card beside a 600px
  // canvas: on a phone it has to become the page's width like everything else.
  R.canvases = [...document.querySelectorAll('canvas')].map((c) =>
    [c.id || c.className || '(anon)', c.width, Math.round(box(c).width),
     c.parentElement.id || c.parentElement.className || c.parentElement.tagName]);
  R.panelCol = wide(document.getElementById('rightPanelsContainer'));
  R.msaBox = wide(document.getElementById('msa-buttons'));
  return R;
})()"""

# 🔴 THE ALIGNMENT IS SYNTHESISED FROM THE STRUCTURE, not read from test.a3m.
# That file is the 59-residue "PIAQ..." reference and 6mrr is a different
# protein, so py2Dmol matched no chain and put "No chains matched to MSA
# sequences" on the page instead of an MSA panel - a probe that then measured
# every row except the widest one this page has. Four rows is enough for the
# panel to draw its coverage bar and its logo.
LOAD = """(async () => {
    const pdb = await (await fetch('/6mrr-crystal.pdb')).text();
    const AA = {ALA:'A',ARG:'R',ASN:'N',ASP:'D',CYS:'C',GLN:'Q',GLU:'E',GLY:'G',
                HIS:'H',ILE:'I',LEU:'L',LYS:'K',MET:'M',PHE:'F',PRO:'P',SER:'S',
                THR:'T',TRP:'W',TYR:'Y',VAL:'V'};
    const seen = new Set();
    let query = '';
    for (const line of pdb.split('\\n')) {
        if (!line.startsWith('ATOM') || line.slice(12, 16).trim() !== 'CA') continue;
        const key = line.slice(21, 27);
        if (seen.has(key)) continue;
        seen.add(key);
        query += AA[line.slice(17, 20).trim()] || 'X';
    }
    const rows = ['>query', query];
    for (let i = 1; i <= 4; i += 1) {
        rows.push('>hit' + i);
        rows.push([...query].map((c, j) => (j % (i + 3) === 0 ? '-' : c)).join(''));
    }
    const a3m = rows.join('\\n') + '\\n';
    try {
        await window.processFiles([
            {name: 'a.pdb', readAsync: () => Promise.resolve(pdb)},
            {name: 'b.pdb', readAsync: () => Promise.resolve(pdb)},
            {name: 'a.a3m', readAsync: () => Promise.resolve(a3m)}], true);
    } catch (e) { return 'processFiles threw: ' + (e && e.message); }
    return 'query ' + query.length + ' residues, ' + rows.length + ' a3m lines';
    })()"""


def serve():
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
        def log_message(self, *a): pass
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def write_probe(name="index.html", out=None):
    """index.html with a cache token on every local asset.

    🔴 A FRESH PAGE WITH STALE SCRIPTS READS EXACTLY LIKE A FIX THAT DID
    NOTHING. It cost py2Dmol two rounds. The token is the clock.
    """
    src = open(os.path.join(ROOT, name)).read()
    stamp = str(int(time.time() * 1000))
    src = re.sub(r'(<script[^>]* src="(?!https?:)[^"?]+)(")',
                 lambda m: m.group(1) + "?v=" + stamp + m.group(2), src)
    src = re.sub(r'(<link rel="stylesheet" href="(?!https?:)[^"?]+)(")',
                 lambda m: m.group(1) + "?v=" + stamp + m.group(2), src)
    open(out or PROBE, "w").write(src)


def measure(ws, w, h, shot=None):
    ws.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
            deviceScaleFactor=2, mobile=(w < 980))
    ws.call("Page.navigate", url="http://127.0.0.1:%d/_mobile.html" % PORT)
    # 🔴 `typeof processFiles === "function"` IS NOT "the page is ready". The
    # bundle defines it while initializeApp is still running and before
    # web/app.js - a module, so it waits for the 780 KB classic script above it
    # - has run at all. Called in that window, processFiles resolves having
    # loaded NOTHING: no error, an empty viewer, and every wait below then times
    # out naming the strip that was never going to be built. #predict ships
    # disabled in the markup and app.js enables it when its handler is bound,
    # which is the last thing to happen on this page.
    wait_for(ws, "typeof window.processFiles === 'function'"
                 " && !document.getElementById('predict').disabled",
             what="the bundle and web/app.js to finish loading")
    # 🔴 TWO STRUCTURES, OR THE PLAY BAR IS NOT ON THE PAGE. #controlsContainer
    # is display:none with a single model, so a probe that loads one file
    # measures every row except the one the recycle walker lives in.
    print('   load:', evaluate(ws, LOAD))
    try:
        wait_for(ws, "!!document.getElementById('sequenceCanvas')",
                 what="the sequence strip to be built")
    except RuntimeError:
        state = evaluate(ws, """JSON.stringify({
            errors: window.__errors || [],
            viewer: getComputedStyle(document.getElementById('viewer-container')).display,
            strip: getComputedStyle(document.getElementById('sequence-viewer-container')).display,
            objects: Object.keys((window.py2dmol_viewers || {})).map((k) =>
                (window.py2dmol_viewers[k].objects || []).length)})""", False)
        raise RuntimeError("the sequence strip never appeared at %dpx: %s" % (w, state))
    # ...and one settled frame, so the ResizeObserver has driven the canvas.
    evaluate(ws, "new Promise(r => requestAnimationFrame(() => "
                 "requestAnimationFrame(() => setTimeout(() => r(1), 500))))")
    # THE ENTITY POPUP IS OPENED, because it is the one control on this page
    # that is wider than a phone and it does not exist until it is asked for.
    evaluate(ws, "(() => { const b = document.querySelector('.entity-options');"
                 " if (b) b.click(); return 1; })()", False)
    evaluate(ws, "window.__asked = %d" % w, False)
    R = evaluate(ws, MEASURE, False)
    R["asked"] = w
    if shot is not None:
        data = ws.call("Page.captureScreenshot", format="png",
                       captureBeyondViewport=True)["data"]
        open(shot, "wb").write(base64.b64decode(data))
        R["shot"] = shot
    return R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=None,
                        help="also write /tmp/localfold-<w>.png at this width")
    arguments = parser.parse_args()

    write_probe()
    write_probe("single.html", os.path.join(ROOT, "_mobile-single.html"))
    write_probe("proteinhunter.html", os.path.join(ROOT, "_mobile-hunter.html"))
    httpd = serve()
    proc = ws = None
    results = {}
    single = {}
    hunter = {}
    try:
        proc, ws = launch(DBG, PROFILE)
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        # 🔴 A PAGE THAT THROWS WHILE LOADING FAILS QUIETLY, and every wait_for
        # below then times out naming the thing that never appeared rather than
        # the thing that broke. Installed BEFORE any navigation, because by the
        # time a Runtime.evaluate could add a listener the error has happened.
        ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
            window.__errors = [];
            addEventListener('error', (e) => window.__errors.push(
                String(e.message) + ' @ ' + String(e.filename) + ':' + e.lineno));
            addEventListener('unhandledrejection', (e) => window.__errors.push(
                'unhandled rejection: ' + String(e.reason && e.reason.message || e.reason)));
            const realError = console.error.bind(console);
            console.error = (...parts) => { window.__errors.push(parts.map(String).join(' '));
                                            realError(...parts); };
        """)
        for name, w, h in (("320px", 320, 800), ("360px", 360, 800),
                           ("390px", 390, 844), ("desktop", 1200, 1000)):
            shot = ("/tmp/localfold-%d.png" % w) if arguments.shot == w else None
            results[name] = measure(ws, w, h, shot)
            # ===== AND THE TEACHING PAGE, which is the same shell with a
            # different panel and had its own five inline widths. It cannot
            # fold in headless Chrome - there is no WebGPU device here - so
            # what is measured is the page as a reader first meets it: does it
            # fit, and is the viewer box the width of the screen.
            ws.call("Page.navigate", url="http://127.0.0.1:%d/_mobile-single.html" % PORT)
            wait_for(ws, "!!document.getElementById('predict')",
                     what="single.html to load")
            evaluate(ws, "new Promise(r => requestAnimationFrame("
                         "() => setTimeout(() => r(1), 300)))")
            single[name] = evaluate(ws, """(() => ({
                viewport: innerWidth,
                overflow: document.documentElement.scrollWidth - innerWidth,
                shell: Math.round(document.getElementById('viewer-root')
                                          .getBoundingClientRect().width),
                viewer: Math.round(document.getElementById('canvasContainer')
                                           .getBoundingClientRect().width),
                sequence: Math.round(document.getElementById('sequence')
                                             .getBoundingClientRect().width)}))()""", False)
            single[name]["asked"] = w
            # ===== AND THE DESIGN PAGE, which is the same shell again with a
            # grid of controls and a results table under the viewer. The table
            # is the new risk: a designed sequence is 150 monospace characters
            # and there is one per cycle, so without `overflow-x: auto` on its
            # card the widest ROW would set the layout viewport - which reads
            # as the whole page zoomed out rather than as anything overflowing.
            # Measured with the table EMPTY, because that is the page as it is
            # first met; the rule that contains it is on the card either way.
            ws.call("Page.navigate", url="http://127.0.0.1:%d/_mobile-hunter.html" % PORT)
            wait_for(ws, "!!document.getElementById('hunt')",
                     what="proteinhunter.html to load")
            evaluate(ws, "new Promise(r => requestAnimationFrame("
                         "() => setTimeout(() => r(1), 300)))")
            # 🔴 AND WITH A TARGET ENTITY ROW, which is the element that has
            # already done this once: index.html's entity row put its sequence
            # box on a `1fr` grid track and measured 0px at 320 with "PIA" set
            # one letter per line, while every fit check passed. This page
            # reuses that row, so it inherits both the layout and the trap, and
            # it starts EMPTY - so a probe that did not add one would measure a
            # page with no entity row in it at all.
            evaluate(ws, """(() => {
                if (!window.__hunterTargets) return 'no entity list';
                window.__hunterTargets.set([
                    { type: 'protein', value: 'GWSTELEKHREELKEFLKKEGITLGFTNAEK', copies: 2 },
                    { type: 'ligand', value: 'HEM', copies: 1 },
                ]);
                return window.__hunterTargets.read().length;
            })()""", False)
            # 🔴 WITH A ROW IN IT. An empty table cannot squeeze anything,
            # and the row is the whole risk: eight cells of `white-space:
            # nowrap` with a 150-character monospace sequence in the last one.
            # Measured empty this check passed while the populated page pushed
            # the layout viewport to 1100 on a 320px phone.
            evaluate(ws, """(() => {
                const body = document.getElementById('results-body');
                const cells = ['1', '3', '0.812', 'iptm', '87.4', '0.812', '5%',
                    'DEVKKELEEIKEFIKKEKEKDEVKKELEEIKEFIKKEKEKDEVKKELEEIKEFIKKEKEK'
                    + 'DEVKKELEEIKEFIKKEKEKDEVKKELEEIKEFIKKEKEKDEVKKELEEIKEFIKKEKEK'
                    + 'DEVKKELEEIKEFIKKEKEKDEVKKELEEIKEFIKKEKEK'];
                for (let i = 0; i < 4; i += 1) {
                    const row = document.createElement('tr');
                    for (const [index, text] of cells.entries()) {
                        const cell = document.createElement('td');
                        cell.textContent = text;
                        if (index === cells.length - 1) cell.className = 'sequence';
                        row.append(cell);
                    }
                    body.append(row);
                }
                document.getElementById('results').hidden = false;
                document.getElementById('downloads').hidden = false;
                return body.children.length;
            })()""", False)
            evaluate(ws, "new Promise(r => requestAnimationFrame("
                         "() => setTimeout(() => r(1), 200)))")
            hunter[name] = evaluate(ws, """(() => {
                const box = (id) => Math.round(
                    document.getElementById(id).getBoundingClientRect().width);
                return {
                    viewport: innerWidth,
                    overflow: document.documentElement.scrollWidth - innerWidth,
                    shell: box('viewer-root'),
                    viewer: box('canvasContainer'),
                    // The entity row's own sequence box, which is the one that
                    // has measured 0px before. `.entity-field` is entity-ui's.
                    target: Math.min(...[...document.querySelectorAll(
                        '#entity-rows .entity-field')].map((el) =>
                            Math.round(el.getBoundingClientRect().width))),
                    // 🔴 THE NARROWEST FIELD OF THE CONTROL GRID, NOT THE GRID.
                    // A `1fr` track squeezed to nothing is invisible to every
                    // fit check - the entity row's sequence box measured 0px at
                    // 320 while every one of them passed - and an auto-fit grid
                    // is exactly the construct that does it.
                    // ...text and number fields only. A checkbox is 13px on
                    // purpose and would be the minimum at every width, which
                    // is a floor that can never move and so measures nothing.
                    field: Math.min(...[...document.querySelectorAll(
                        '.hunt-field input:not([type=checkbox])')].map((el) =>
                            Math.round(el.getBoundingClientRect().width))),
                };
            })()""", False)
            hunter[name]["asked"] = w
    finally:
        if proc: proc.kill()
        httpd.shutdown()
        for stale in (PROBE, os.path.join(ROOT, "_mobile-single.html"),
                      os.path.join(ROOT, "_mobile-hunter.html")):
            if os.path.exists(stale): os.remove(stale)
        import shutil
        shutil.rmtree(PROFILE, ignore_errors=True)

    bad = []
    for name in ("320px", "360px", "390px", "desktop"):
        R = results[name]
        print("%s: asked %d, innerWidth %d, scrollWidth %d, shell %d, columns %s,"
              " title overlap %s"
              % (name, R["asked"], R["viewport"], R["scrollW"], R["shell"],
                 R["mainDir"], R["titleOverlap"]))
        print("   canvas box %s (content %s), canvas css %s, renderer %s"
              % (R["container"], R["content"], R["canvasCSS"], R["rendererSize"]))
        print("   panel column %s, msa header %s, msa canvas %s, popup %s"
              % (R["panelCol"], R["msaBox"], R["msa"], R["popup"]))
        print("   entity row %s" % R["entityWidths"])
        for c in R["canvases"]:
            print("   canvas %-16s store %-5s css %-5s in %s" % tuple(c))
        for st in R["stiff"]:
            print("   WILL NOT SHRINK below %dpx: %s" % (st["min"], st["el"]))
        for k in ("head", "entity", "options", "play", "footer"):
            rows = R["rows"][k]
            print("   %-8s %s" % (k, "absent" if rows is None else
                  " | ".join("[" + ", ".join(l) + "]" for l in rows)))
        if R["strip"]:
            print("   strip    box %spx (%d%% of viewport), stretch %s"
                  % (R["strip"]["boxH"], round(R["strip"]["share"] * 100),
                     R["strip"]["stretch"]))
        if "shot" in R:
            print("   wrote %s" % R["shot"])

    for name in ("320px", "360px", "390px"):
        R = results[name]
        # 🔴 THE ONE THAT MATTERS: the viewport grew rather than the page
        # overflowing, which is what a squished phone page actually looks like.
        if R["viewport"] > R["asked"] + 8:
            bad.append("%s: asked for a %dpx device and got a %dpx layout viewport -"
                       " the page could not fit, so the viewport grew. It renders"
                       " zoomed out." % (name, R["asked"], R["viewport"]))
        if R["overflow"] > 1:
            bad.append("%s: scrolls sideways by %dpx" % (name, R["overflow"]))
        if R["titleOverlap"]:
            bad.append("%s: Add entity / Fold are drawn ON TOP of the title - it costs"
                       " no width, so no size check can see it" % name)
        if R["mainDir"] != "column":
            bad.append("%s: the viewer and the panels are still %s: 600 + 8 + 340 does"
                       " not fit" % (name, R["mainDir"]))
        if abs(R["canvasCSS"][0] - R["content"][0]) > 2 \
                or abs(R["canvasCSS"][1] - R["content"][1]) > 2:
            bad.append("%s: the canvas is %s inside a %s content box - it did not follow"
                       " its container" % (name, R["canvasCSS"], R["content"]))
        if R["rendererSize"] and abs(R["rendererSize"][0] - R["canvasCSS"][0]) > 2:
            bad.append("%s: the renderer thinks it is drawing into %s while the canvas"
                       " is %s" % (name, R["rendererSize"], R["canvasCSS"]))
        if R["container"][0] < R["asked"] * 0.75:
            bad.append("%s: the canvas is only %dpx wide - it stacked but did not use"
                       " the width, which is most of the point"
                       % (name, R["container"][0]))
        if R["panelCol"] > R["asked"]:
            bad.append("%s: the panel column is %dpx on a %dpx device - it kept its"
                       " desktop 340 and then some" % (name, R["panelCol"], R["asked"]))
        # THE ENTITY ROW IS ONE LINE, which is the whole reason it is a grid:
        # the columns line up down the list as well as across it.
        entity = R["rows"]["entity"]
        if entity is None:
            bad.append("%s: there is no entity row on the page - the probe is checking"
                       " nothing" % name)
        # TWO LINES ON PURPOSE, and which two is the assertion: the four small
        # controls together, then the sequence box alone with the row's whole
        # width. [type, copies, options] / [remove] / [field] is also "the row
        # wrapped" and is the layout this replaced.
        elif len(entity) != 2 or entity[1] != [".entity-field"]:
            bad.append("%s: the entity row should be the four controls and then the"
                       " sequence box on a line of its own; got %s" % (name, entity))
        # 🔴 THE FIELD IS THE CONTROL THE PAGE EXISTS FOR. 120px is about
        # twenty residues at 13px monospace - narrow, but a sequence box rather
        # than a column of single letters.
        field = R["entityWidths"].get(".entity-field")
        if field is not None and field < 120:
            bad.append("%s: the sequence box is %dpx wide - the fixed controls beside it"
                       " took the row and it sets one letter per line" % (name, field))
        if R["msa"] and abs(R["msa"]["stretch"] - 1) > 0.02:
            bad.append("%s: the MSA panel's bitmap is %d logical px in a %dpx box - a"
                       " %.2fx horizontal scale" % (name, R["msa"]["logical"],
                                                    R["msa"]["css"], R["msa"]["stretch"]))
        # THE OPTIONS ARE PAIRED BY MEANING, and a flex row left to itself
        # filled lines in DOM order: Seed beside Sampler, Steps beside MSA -
        # controls from different questions sharing a line while the two halves
        # of one question were split across two. A line COUNT cannot see that;
        # this is the membership.
        options = R["rows"]["options"]
        want = [["modelGroup", "recyclesGroup", "seedGroup"],
                ["af3ModeGroup", "af3CountGroup"],
                ["msaModeGroup", "maxMsaGroup"]]
        if options != want:
            bad.append("%s: the fold options should be Model/Recycles/Seed, then"
                       " Sampler/Steps, then MSA/Max MSA; got %s" % (name, options))
        for what, m in R["overflowing"].items():
            if m and m["over"] > 1:
                bad.append("%s: %s overflows its own box by %dpx - the label wrapped"
                           " inside it" % (name, what, m["over"]))
        if R["popup"] and (R["popup"]["left"] < -1 or R["popup"]["right"] > R["viewport"] + 1):
            bad.append("%s: the modification popup spans %d..%d on a %dpx screen - it"
                       " hangs off the edge" % (name, R["popup"]["left"],
                                                R["popup"]["right"], R["viewport"]))
        if R["strip"] and abs(R["strip"]["stretch"] - 1) > 0.02:
            bad.append("%s: the sequence strip's bitmap is %d logical px in a %dpx box -"
                       " a %.2fx horizontal scale, which is what squishes the letters"
                       % (name, R["strip"]["logical"], R["strip"]["css"],
                          R["strip"]["stretch"]))
        if R["strip"] and R["strip"]["share"] > 0.4:
            bad.append("%s: the sequence strip is %dpx, %d%% of the viewport - the"
                       " structure it belongs to is what the reader came for"
                       % (name, R["strip"]["boxH"], round(R["strip"]["share"] * 100)))
        if R["resizeHandle"] != "none":
            bad.append("%s: the canvas resize handle is showing (display: %s). It is"
                       " revealed on :hover, which a touch screen has not got, and the"
                       " box is resize:none here anyway" % (name, R["resizeHandle"]))
        play = R["rows"]["play"]
        if play and len(play) != 1:
            bad.append("%s: the play bar broke across %d lines: %s"
                       % (name, len(play), play))

    print("single.html:")
    for name in ("320px", "360px", "390px", "desktop"):
        S = single[name]
        print("   %-8s asked %d, innerWidth %d, shell %d, viewer %d, sequence box %d"
              % (name, S["asked"], S["viewport"], S["shell"], S["viewer"], S["sequence"]))
        if name != "desktop":
            if S["viewport"] > S["asked"] + 8:
                bad.append("single.html at %s: the layout viewport is %d - it could not"
                           " fit and renders zoomed out" % (name, S["viewport"]))
            if S["viewer"] > S["asked"]:
                bad.append("single.html at %s: the viewer box is %dpx on a %dpx device"
                           % (name, S["viewer"], S["asked"]))
            if S["sequence"] < 100:
                bad.append("single.html at %s: the sequence box is %dpx wide"
                           % (name, S["sequence"]))
        elif S["shell"] != 948 or S["viewer"] != 948:
            bad.append("single.html on the desktop: shell %d, viewer %d - both were 948"
                       % (S["shell"], S["viewer"]))

    print("proteinhunter.html:")
    for name in ("320px", "360px", "390px", "desktop"):
        H = hunter[name]
        print("   %-8s asked %d, innerWidth %d, shell %d, viewer %d,"
              " narrowest entity box %d, narrowest field %d"
              % (name, H["asked"], H["viewport"], H["shell"], H["viewer"],
                 H["target"], H["field"]))
        if name != "desktop":
            if H["viewport"] > H["asked"] + 8:
                bad.append("proteinhunter.html at %s: the layout viewport is %d - it"
                           " could not fit and renders zoomed out"
                           % (name, H["viewport"]))
            if H["viewer"] > H["asked"]:
                bad.append("proteinhunter.html at %s: the viewer box is %dpx on a %dpx"
                           " device" % (name, H["viewer"], H["asked"]))
            if H["target"] < 100:
                bad.append("proteinhunter.html at %s: an entity row's sequence box"
                           " is %dpx wide" % (name, H["target"]))
            if H["field"] < 40:
                bad.append("proteinhunter.html at %s: a control-grid field is %dpx"
                           " wide - the auto-fit track collapsed"
                           % (name, H["field"]))
        elif H["shell"] != 948 or H["viewer"] != 948:
            bad.append("proteinhunter.html on the desktop: shell %d, viewer %d - both"
                       " were 948" % (H["shell"], H["viewer"]))

    D = results["desktop"]
    if D["overflow"] > 1:
        bad.append("the DESKTOP layout overflows by %dpx" % D["overflow"])
    if D["mainDir"] != "row":
        bad.append("the desktop stacked its columns (%s) - the breakpoint fires too"
                   " wide" % D["mainDir"])
    if D["shell"] != 948:
        bad.append("the desktop shell is %dpx, not the 948 it has always been"
                   % D["shell"])
    if D["content"] != [598, 598]:
        bad.append("the desktop canvas content box is %s, not 598x598 (600 less its"
                   " 1px border)" % D["content"])
    if D["titleOverlap"]:
        bad.append("the desktop title and the fold actions overlap")
    if D["resizeHandle"] == "none":
        bad.append("the desktop canvas resize handle was hidden too - that rule is"
                   " meant to be narrow-only")
    if D["rows"]["options"] is not None and len(D["rows"]["options"]) != 1:
        bad.append("the desktop fold options broke across %d lines - they are one row"
                   " there and the narrow grouping must not reach them"
                   % len(D["rows"]["options"]))
    if D["rows"]["entity"] and len(D["rows"]["entity"]) != 1:
        bad.append("the desktop entity row broke across %d lines"
                   % len(D["rows"]["entity"]))

    for m in bad:
        print("FAIL:", m)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

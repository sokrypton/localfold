# Vendored from the py2Dmol checkout

🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them
upstream and re-run `python3 tools/sync-py2dmol.py`;
`python3 tools/sync-py2dmol.py --check` says whether they have drifted.

- upstream: `/Users/mini/Documents/GitHub/py2Dmol`
- commit: `67cbe73f06f49645291ea3e79a06293b0abaa3a7`

Two of these are built by upstream's `tools/bundle.py build` and two
are source files that ship as they are. `full` is the website plus the
embed API and `embed` is the embed API alone - both are needed,
because index.html loads one and the other two pages load the other.

- `py2Dmol.app.css` — `f7fda445ca6a5629` — index.html
- `py2Dmol.align.js` — `24cc39809ebe3ba4` — index.html (TM-align; upstream cannot bundle it)
- `py2Dmol.full.min.js` — `9d7b8b2b96ae4890` — index.html

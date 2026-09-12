# Vendored from the py2Dmol checkout

🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them
upstream and re-run `python3 tools/sync-py2dmol.py`;
`python3 tools/sync-py2dmol.py --check` says whether they have drifted.

- upstream: `/Users/mini/Documents/GitHub/py2Dmol`
- commit: `e5fbfe99cec67b7a3fbc5e77ae87ac4f9a785811`

Two of these are built by upstream's `tools/bundle.py build` and two
are source files that ship as they are. `full` is the website plus the
embed API and `embed` is the embed API alone - both are needed,
because index.html loads one and the other two pages load the other.

- `py2Dmol.app.css` — `a3ba427f9f7ff30b` — index.html, single.html, proteinhunter.html
- `py2Dmol.align.js` — `24cc39809ebe3ba4` — index.html (TM-align; upstream cannot bundle it)
- `py2Dmol.embed.min.js` — `c2a0c88faa4dc4ff` — single.html, proteinhunter.html
- `py2Dmol.full.min.js` — `e9d4439433afe91f` — index.html

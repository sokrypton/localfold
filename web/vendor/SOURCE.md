# Vendored from the py2Dmol checkout

🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them
upstream and re-run `python3 tools/sync-py2dmol.py`;
`python3 tools/sync-py2dmol.py --check` says whether they have drifted.

- upstream: `/Users/mini/Documents/GitHub/py2Dmol`
- commit: `10c9a561ffcdfe9acbd3981f35b86391dd32d375`

Two of these are built by upstream's `tools/bundle.py build` and two
are source files that ship as they are. `full` is the website plus the
embed API and `embed` is the embed API alone - both are needed,
because index.html loads one and the other two pages load the other.

- `py2Dmol.app.css` — `bc5e5e6917e3c397` — index.html, single.html, proteinhunter.html
- `py2Dmol.align.js` — `24cc39809ebe3ba4` — index.html (TM-align; upstream cannot bundle it)
- `py2Dmol.embed.min.js` — `339997214b2c81e0` — single.html, proteinhunter.html
- `py2Dmol.full.min.js` — `774128d6371bd213` — index.html

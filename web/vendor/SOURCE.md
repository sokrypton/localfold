# Vendored from the py2Dmol checkout

🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them
upstream and re-run `python3 tools/sync-py2dmol.py`;
`python3 tools/sync-py2dmol.py --check` says whether they have drifted.

- upstream: `/home/ubuntu/py2Dmol`
- commit: `bfef031853bbe521991a5857862bcc17413606d7`

Two of these are built by upstream's `tools/bundle.py build` and two
are source files that ship as they are. `full` is the website plus the
embed API and `embed` is the embed API alone - both are needed,
because index.html loads one and the other two pages load the other.

- `py2Dmol.app.css` — `f7fda445ca6a5629` — index.html
- `py2Dmol.align.js` — `24cc39809ebe3ba4` — index.html (TM-align; upstream cannot bundle it)
- `py2Dmol.full.min.js` — `6c3dc2904796595a` — index.html

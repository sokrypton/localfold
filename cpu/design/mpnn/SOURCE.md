# Vendored from the mpnn checkout

🔴 **DO NOT EDIT THESE FILES.** They are a mirror. Change them
upstream and re-run `python3 tools/sync-mpnn.py`;
`python3 tools/sync-mpnn.py --check` says whether the mirror and the
upstream checkout disagree. Nothing runs it for you: it was a node
test and that went red whenever ../mpnn moved on, which says nothing
about this tree.

- upstream: `/Users/mini/Documents/GitHub/mpnn/mpnn`
- commit: `a80942f053a9f35327fc97a6aa25765bbb9fd3e0`

The modules below are the import closure of what
`cpu/design/mpnn-bridge.js` reaches. Upstream also ships `c6d.js`,
`potts.js`, `search.js`, `trmrf.js` and `trmrfaccel.js`, which serve
the Potts and trMRF features and are not mirrored.

- `accel.js` — `d43e0994eb6c1257`
- `arena.js` — `f9e798b5c3d39987`
- `constants.js` — `56801622c35b01ab`
- `features.js` — `7e51673f3e2452d2`
- `layers.js` — `9ba932cf6098cadc`
- `model.js` — `c5362294ae1bdf99`
- `na.js` — `73fe8a2931822466`
- `ops.js` — `e6dba9b79b1907f7`
- `pdb.js` — `272fd67263d50a55`
- `weights.js` — `0f92ee6e89b65966`

## Local changes not in the upstream commit above

🔴 The mirror had already drifted from `a80942f` before this note: `constants.js` re-exports `ATOM37`
from `shared/af3/featurise/template-features.js` (a LocalFold path, so it cannot go upstream),
`model.js` rewords a comment, and `pdb.js` carries `labelSeq`. A plain `sync-mpnn.py` would undo all
three. One more, made here and upstream on 2026-10-28 (sokrypton/mpnn#4):
`pdb.js`'s `NUCLEIC` names AF3's unknown nucleotides `DN` and `N`.

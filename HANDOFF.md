# Where this is, and what it is for

Written mid-task so the next session can pick it up. Delete it when the work
below is finished.

## The problem

**AF3 has no confidence until the fold is over.** Its head runs once, on the
finished sample, so the four trunk previews and fifteen sampler frames drawn
on the way have nothing real to colour by. The page currently paints them with
a distogram-derived estimate whose bias is per target - on Top7 it reads 86.5
against a real 85.3, on a nonsense fusion 83.6 against 54.0 - and then
recolours the whole trajectory once the head has run.

The user's framing, which is the one to build to:

> use af2 dgram and af2 coordinates and af2 plddt, and figure out how to
> convert dgram to plddt

**AF2 is the vehicle, AF3 is the customer.** AF2's confidence head runs on
EVERY recycle, so it hands over (distogram, structure, pLDDT) triples with real
labels for free. AF3 cannot label its own previews, which is the whole gap.

## Running right now

`tools/gpu/probe-af2-dgram-plddt.js`, in the background, 13 chunks:

```
S=/private/tmp/claude-501/-Users-mini-Documents-GitHub-localfold/<session>/scratchpad
for start in 0 12 24 ... 144; do
  node tools/gpu-chrome.mjs tools/gpu/probe-af2-dgram-plddt.js \
    --from=$start --to=$((start+12)) --recycles=3 > $S/af2/chunk-$start.json
done
```

151 single-chain natives, single sequence, 4 recycles each. About 300 MB, ~1
hour. Chunks 0-48 were done at the time of writing; check `$S/af2/`.

Per target per recycle it saves: the AF2 distogram (softmaxed, uint8, upper
triangle, base64), the alpha carbons, AF2's per-residue pLDDT, and lDDT against
the native for context.

## What to do when it lands

1. **Fit distogram -> pLDDT per position**, leave-one-TARGET-out, report RMSE.
   `$S/fit.py` is a working starting point written against the AF3 set; the
   feature builder in it (expected neighbours per distance bin, error
   histograms at several tolerances, the same restricted to pairs inside 15 A,
   and those normalised by inclusion mass - 89 features) transfers directly.
   On the AF3 set against true lDDT it gave RMSE 9.42 where the hand-derived
   estimator gave 21.19.
2. **Cut the features down.** 89 is fine offline and too many to ship. A
   per-position ridge over ~10 features is what could run live.
3. **Replace the live estimator.** `web/af3-model.js` builds `liveConfidence`
   at the first recycle; it currently calls `distogramLddt` +
   `lddtToPlddt` from `src/af3/distogram-lddt.js`. That is the seam.

## Facts already established, so they are not re-derived

- **lDDT's own arithmetic beats a heuristic.** Rewriting the estimator as
  lDDT's definition with the distogram as the reference - four thresholds,
  probabilistic inclusion, every pair rather than the sixteen sharpest - cut
  LOTO RMSE from 10.13 to 8.67 against pLDDT before any parameter moved.
  `src/af3/distogram-lddt.js`.
- **Inclusion radius 18 A, not lDDT's 15.** Swept: 12 -> 9.56, 15 -> 8.67,
  18 -> 8.33, 22 -> 9.30. It wins on 10 of 14 targets and in all 14
  leave-one-out subsets. The reason is real rather than fitted: lDDT decides
  inclusion from a structure it can measure, but here the reference is a
  DISTRIBUTION, so a pair whose expected distance is 17 A still carries mass
  below 15.
- **Thresholds stay lDDT's {0.5, 1, 2, 4}.** Widening to {1, 2, 4, 8} bought
  0.10 RMSE and flipped sign on one of fourteen subsets. Not worth the
  definition.
- **Sequence separation and a contact cap were never swept.** They are wired
  as options (`separation`, `maxContacts`) and default to off.
- **Fitting to pLDDT flatters.** The same hand-derived estimator scores 8.33
  against pLDDT and 21.19 against true lDDT vs crystal. pLDDT is itself an
  approximation; fitting to it caps a predictor at reproducing another model's
  guess. The user has chosen pLDDT as the target deliberately - a live UI
  number should match what the head will say - and both columns are saved so
  this is revisitable without re-folding.
- **AF2's distogram head did not exist in this repo until today.** Neither
  published bundle carried one. `tools/add_distogram_head.py` puts it in the
  manifest as base64 rather than a shard; a shard broke every AF2 fold because
  the manifests are compiled into the page while shards come from a pinned
  remote that had no such file.

## Traps that have already cost time

- **`recycle-done` fires BEFORE that pass's preview** in `src/af3/fold.js`.
  Reading backwards from a captured list pairs each distogram with the PREVIOUS
  pass's structure. Silent - every row still looks well formed. Hold the trunk
  in a variable instead.
- **A plain reload serves cached ES modules.** `http.server` sends no cache
  headers, so a change lands, the page is reloaded, and nothing happens. Ask
  the page what it LOADED, not what is on disk:
  `(await import('/src/af3/fold.js')).foldBatch.toString().includes('...')`
  against a cache-busted `fetch`. Cmd+Shift+R clears it. `tools/fold-in-page.py`
  never sees this because it launches a fresh profile - which is why it can
  pass while the browser in front of you does not. Also in CLAUDE.md.
- **A soft reload keeps CSS cached too**, and a rule that never arrived looks
  exactly like a rule that does not work. Check `document.styleSheets` for the
  selector.
- **`viewer.objects` does not exist on this build.** Several call sites use
  `viewer?.objects?.find(...)`, an optional chain that always yields undefined.
  The objects live in `viewer.objectsData`.
- **Chrome dies somewhere above ~15 folds in one process** at 444 tokens, and
  silently: "Chrome exited (0) before reporting". Chunk long sweeps. The probes
  set no memory budget, unlike the page, so an overrun is a crash rather than a
  message.
- **The entity field is a `<textarea>`.** Setting `textContent` does not change
  its value once rendered, so a test can fold the same sequence repeatedly and
  hit the trunk cache without noticing. Set `.value` and dispatch `input`.
- **A re-fold reuses the cached trunk and runs no recycles**, so it produces no
  previews. Correct, but it makes the trajectory change length with cache
  state.

## Uncommitted here

`src/af3/distogram-lddt.js` (new), `test/distogram-lddt.test.js` (new),
`tools/gpu/probe-plddt-features.js`, `probe-lddt-dataset.js`,
`probe-af2-dgram-plddt.js` (all new), and edits to `web/af3-model.js` pointing
the live estimator at the new file. `natives/` is a symlink to the unpacked
`natives.zip` and is gitignored; the dataset lives in the scratchpad.

## Also outstanding, unrelated to the fitting

- **`../py2Dmol` has ~8 uncommitted changes of mine** on top of heatmap work
  that was already dirty when first vendored: the borrowed-head option, the
  axis-band union, contact captions, tab styling, the outside-the-box strip,
  the `overflow` rule, the margin and four-side inset, and `updateVisibility`
  accepting a renderer-loaded map. That tree should be committed so the
  vendored bundles have a reference point.
- **Nothing is deployed.** `localfold.org` is ~60 commits behind. The AF2
  distogram head is embedded in the manifest specifically so it needs no
  upload.

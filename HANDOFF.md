# Where this is, and what it is for

Written mid-task so the next session can pick it up. Delete it when the work
below is finished.

## The question that is now ANSWERED

**AF3 has no confidence until the fold is over.** Its head runs once, on the
finished sample, so the trunk previews and sampler frames drawn on the way have
nothing real to colour by; the page paints them with a distogram-derived
estimate and recolours the trajectory once the head has run.

The plan was to replace that estimate with something fitted, on the user's
framing:

> use af2 dgram and af2 coordinates and af2 plddt, and figure out how to
> convert dgram to plddt

**That has been done, and the answer is not to ship a fitted predictor.**
`tools/gpu/probe-af2-dgram-plddt.js` collected 108 single-sequence targets over
four recycles - 44,740 residue rows, per-target mean pLDDT from 29.7 to 91.5 -
and `tools/fit-distogram-plddt.py` fitted them leave-one-target-out:

    shipped estimator, uncalibrated   RMSE 10.44   worst target 23.28
    the same + a two-number affine          4.53                 7.88
    ten features, ridge                     4.18                 6.97
    all 89 raw features, ridge              3.91                 6.23

**Almost all of it is calibration.** Two numbers take 10.44 to 4.53; eighty-nine
features buy 0.6 more, and the constants are per-model anyway - AF2 wants
`1.33 + 1.20x` for the same feature where AF3 wants `41.29 + 0.578x`, so a
vector fitted on AF2 cannot be dropped into an AF3 estimator. The lDDT-shaped
estimator in `src/af3/distogram-lddt.js` stays, and `web/af3-model.js` keeps
calling it. For scale: it predicts AF2's pLDDT to 4.53 where AF2's pLDDT
predicts a crystal structure to 13.44.

Two open parameters were closed at the same time, both negative - a sequence
separation floor and a contact cap are a wash at best and much worse past a
floor of three or a cap of thirty-two. The numbers are in the constants block
of `src/af3/distogram-lddt.js`.

🔴 **AND ONE EARLIER CONCLUSION DID NOT REPRODUCE.** The 18 A inclusion radius
was chosen on a fourteen-target AF3 panel where it beat 15 by 0.34 RMSE. On
108 AF2 targets they are level the other way - 15 at 4.97, 18 at 5.06 - while
12 (6.27) and 22 (5.48) are clearly worse on both. The shape of the curve is
real and the peak is not. 18 stays because it is the better of the two on the
model the estimator actually serves, not because the margin was established.

## Still outstanding, unrelated to the fitting

- **Four of thirteen collection chunks crashed** ("Chrome exited (0) before
  reporting", the known ceiling somewhere above fifteen folds in one process),
  so the set is 108 targets rather than 151. Re-running those four would add
  about forty targets; nothing above turned on set size.
- **`../py2Dmol` has ~8 uncommitted changes of mine** on top of heatmap work
  that was already dirty when first vendored: the borrowed-head option, the
  axis-band union, contact captions, tab styling, the outside-the-box strip,
  the `overflow` rule, the margin and four-side inset, and `updateVisibility`
  accepting a renderer-loaded map. That tree should be committed so the
  vendored bundles have a reference point.
- **Nothing is deployed.** `localfold.org` is ~60 commits behind. The AF2
  distogram head is embedded in the manifest specifically so it needs no
  upload.

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
- **The entity field is a `<textarea>`.** Setting `textContent` does not change
  its value once rendered, so a test can fold the same sequence repeatedly and
  hit the trunk cache without noticing. Set `.value` and dispatch `input`.
- **A re-fold reuses the cached trunk and runs no recycles**, so it produces no
  previews. Correct, but it makes the trajectory change length with cache
  state.

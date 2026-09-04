# The pLDDT-from-distogram problem, and why the code for it is gone

Everything that estimated confidence from a distogram has been deleted. The
DATA is kept. This is the record of what was measured, so the next attempt does
not re-derive it.

## What the page does now

**Frames drawn during a fold carry no confidence, and the finished trajectory
takes the final structure's real pLDDT.** That is a constant colour on a moving
structure - honest, and worse to look at than what it replaced.

AF3's confidence head runs ONCE, on the finished sample. The four trunk
previews and fifteen sampler frames drawn on the way have no measured
confidence and cannot have one without running the head per frame, which costs
more than a denoiser call and would roughly double a fold.

## The data that is kept

`plddt-data/` (gitignored, 228 MB), written by
`tools/gpu/probe-af2-dgram-plddt.js`:

- 108 single-chain natives, folded SINGLE SEQUENCE by AF2 monomer, 4 recycles.
- Per target per recycle: the distogram (softmaxed, uint8, upper triangle,
  base64), the alpha carbons, AF2's per-residue pLDDT, and true lDDT vs the
  native.
- 44,740 residue-recycle rows. Per-target mean pLDDT from 29.7 to 91.5, 28
  targets under 40 and 16 over 70 - the range is the point, and single sequence
  is how it was got.
- `features.npz` is a cached design matrix: 89 raw features per row plus the
  pLDDT and true-lDDT columns and the target names, for leave-one-target-out.

🔴 **AF2 IS THE VEHICLE AND AF3 IS THE CUSTOMER.** AF2's confidence head runs on
every recycle, so it hands over (distogram, structure, pLDDT) already paired.
That is the only source of free labels; it is not the model the answer is for.

## Everything that was tried, and what it scored

All leave-one-TARGET-out on the 108 targets, RMSE against AF2's own pLDDT,
single score plus a two-number affine unless said otherwise.

| estimator | RMSE | worst target |
|---|---|---|
| lDDT's arithmetic, distogram as reference | 4.97 | 9.29 |
| contact Jaccard, 12-14 A | 5.93 | 12.39 |
| cross entropy, arithmetic mean, 2 A box | 6.58 | 12.39 |
| cross entropy, arithmetic mean, one-hot | 7.14 | 15.71 |
| contact precision only | 7.06 | 17.62 |
| cross entropy, GEOMETRIC mean (the plain CE) | 11.11 | 55.40 |
| distogram entropy alone, structure unused | 13.84 | 25.32 |

and with fitted weights rather than one affine:

| | RMSE |
|---|---|
| the four lDDT thresholds as separate regressors | 4.53 |
| + contact Jaccard | 4.35 |
| + Jaccard and the CE box term | 4.30 |
| ten hand-picked features, ridge | 4.18 |
| all 89 raw features, ridge | 3.91 |

### What those numbers mean

🔴 **ALMOST ALL OF ANY FIT IS CALIBRATION.** Two numbers take the shipped shape
from 10.44 uncalibrated to 4.53. Eighty-nine features buy 0.6 more. Nothing in
the feature engineering was worth what it cost to justify.

🔴 **AND THE CALIBRATION DOES NOT CROSS MODELS.** AF2 wants `1.33 + 1.20x` for
the same feature where AF3 wants `41.29 + 0.578x`; crossing them costs RMSE
19.17. A ten-feature vector fitted on AF2 and moved to the AF3 panel with only
the affine refitted scored 8.77 where the plain lDDT form scored 8.33 - worse
than the thing it was supposed to improve.

🔴 **THE PLAIN CROSS ENTROPY FAILS BECAUSE OF THE LOG, NOT BECAUSE IT IS THE
WRONG QUANTITY.** Same per-pair number, averaged arithmetically instead of
geometrically: 11.11 to 7.14. One pair the trunk gave 1e-4 to drags the whole
geometric mean, so the score reads the worst neighbour rather than the typical
one. Widening the one-hot to a 2 A box - a bin is 0.32 A, so an exact one-hot
asks the trunk to name a distance to a third of an angstrom - takes it to 6.58.
It was NOT measuring the distogram's sharpness: the entropy-only control
correlates 0.072 with it.

🔴 **AND THE ARITHMETIC MEAN OF THE MASS WITHIN A TOLERANCE IS lDDT'S
NUMERATOR.** So the cross-entropy sweep walks back to the lDDT form by changing
only how the average is taken and how wide the target is. Three independent
starting points - lDDT's definition, ColabDesign's contact loss, and cross
entropy against the frame - converge on the same estimator.

### Parameters, all swept

- **Inclusion radius**: 12 -> 6.27, 15 -> 4.97, 18 -> 5.06, 22 -> 5.48. 15 and
  18 are a coin toss and the ends are clearly worse. An earlier 14-target AF3
  panel put 18 ahead by 0.34; it did not reproduce.
- **Thresholds**: lDDT's {0.5, 1, 2, 4} against {1, 2, 4, 8} was 0.10 apart.
- **Sequence separation**: 0 to 3 is a wash for the lDDT form, 6 and beyond
  worse. For the contact and CE forms 3 to 6 was best. Nothing dramatic
  anywhere; the free marks from chain neighbours are not the problem they look.
- **Contact cap**: every cap is worse than none - top-8 is 12.10 against 5.93
  for every pair. Restricting to the sharpest contacts throws away information.
- **Contact cutoff**: 12 and 14 A level and best, 8, 10 and 16 worse.
- **Match form**: Jaccard beats precision (7.06), recall and F1, because it is
  the only one charging for BOTH a promised contact that never forms and a
  formed one that was never promised.

## What is NOT known

- **Nothing here was ever validated against a large AF3 set.** Every AF3 number
  above comes from a 14-target, 810-token panel. AF2's 108 targets are what the
  conclusions rest on, and AF3 is a different distogram.
- **Whether AF3's own head could be run cheaply on a preview.** It was assumed
  too expensive from the AF2 measurement (47-226 ms against a denoiser call)
  and never actually profiled on a trunk preview.
- **Whether a frame needs a per-residue number at all.** Every attempt was
  aimed at per-residue pLDDT. A single per-frame "how settled is this" scalar,
  which is what the animation actually communicates, was never tried on its own
  terms - the per-residue estimator's mean was measured to rise monotonically
  and saturate on all eight targets of an earlier panel, which is the property
  that was actually wanted.

## Reproducing any of it

`tools/gpu/probe-af2-dgram-plddt.js` is the collector and is kept. The fitting
scripts were scratchpad Python over `plddt-data/`; the design matrix in
`features.npz` is 64 distogram-mass bins, 6 error-histogram counts, the mass
inside 15 A, the same 6 restricted to it, and both sets normalised by that mass.

## Also outstanding, unrelated

- **`../py2Dmol` has ~8 uncommitted changes of mine** on top of heatmap work
  that was already dirty when first vendored. That tree should be committed so
  the vendored bundles have a reference point.
- **Nothing is deployed.** `localfold.org` is ~60 commits behind.
- **Four of thirteen collection chunks crashed** ("Chrome exited (0) before
  reporting", the known ceiling above ~15 folds in one process), so the set is
  108 targets rather than 151.

## Traps that have already cost time

- **`recycle-done` fires BEFORE that pass's preview** in `src/af3/fold.js`.
  Reading backwards from a captured list pairs each distogram with the PREVIOUS
  pass's structure. Silent - every row still looks well formed.
- **A plain reload serves cached ES modules.** `http.server` sends no cache
  headers. Ask the page what it LOADED, not what is on disk. Also in CLAUDE.md.
- **`viewer.objects` does not exist on this build.** The objects live in
  `viewer.objectsData`; `viewer?.objects?.find(...)` is a silent no-op.
- **The entity field is a `<textarea>`.** Setting `textContent` does not change
  its value, so a test can fold the same sequence repeatedly and hit the trunk
  cache without noticing. Set `.value` and dispatch `input`.

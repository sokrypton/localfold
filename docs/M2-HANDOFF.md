# Handoff to the M2: the `opus-55-opt` branch

Written from the A100 on 2026-09-26. Branch `opus-55-opt` is **65 commits on top
of `main`@`ab5bd38`'s parent** (it is one commit behind `origin/main`, the pLDDT
trace tab - merge or rebase before measuring the page). Nothing here is on
`main` or deployed.

Every number below was taken on an **A100 under `LOCALFOLD_STOCK_FLAGS=1`**:
no `shader-f16`, no subgroup matrices, so every model ran its f32 vector
kernels. **An M2 is a different configuration on both axes that matter**: a
stock Chrome there HAS `shader-f16` (round four) and still has no matrix units
at these widths, and its memory is unified. So nothing here has been measured
on the path an M2 runs. That is the job.

🔴 **COMPARE AN ARM AGAINST THE SAME BOX'S OTHER ARM.** Checksums and pLDDTs do
not travel between machines (CLAUDE.md). Every change below has a knob or a
parent commit to be its control; run both arms on the M2, interleaved.

## What the branch does, grouped by what an M2 should ask

### 1. Changes that are defaults on EVERY device - the ones that can regress there

Only two knobs are ampere-prior-only (`triangleProjectOutColumns: 64`,
`attentionProjectRowsPerLane: 8`). Everything else is on for an M2 unless a
knob turns it off. The ones whose value is a DEVICE question:

| change | commit | A100 stock result | knob / control | M2 question |
|---|---|---|---|---|
| AF3 outer product mean as two GEMMs (AF2's vector contraction + a new output GEMM) | `6ab73a7` | `opm.contract` 195.9 -> 46.6 ms, 255 tokens, 1024 MSA rows; bit-exact on 6 models | `--tune=opmVectorContract=false` | does the vector GEMM beat the fused kernel on Apple's memory system at depth? `bench-trunk.js --profile --msa=1024` both arms |
| ESMFold2 token transformer and ESM-C tower linears at row tile 2 (was 8) | `0145163` | sampler 499 -> 309 ms at 59 res, LM 145 -> 114; bit-exact | `--tune=esmfold2TokenRowTile=8`, `--tune=esmcRowTile=8` | the tile was chosen for a 108-SM card's occupancy. An M2 has ~10 cores: **sweep 1/2/4/8**, it may want 8 back |
| small folds stream trunk weights from int5 codes kept on the device | `3fee383` | AF3 5CAJ-255 peak 1980 -> 1530 MiB at equal warm time | `--tune=streamTrunkWeights=false` | on unified memory this is RAM the whole machine shares - probably worth MORE there. Confirm the warm fold does not slow (the replay is one fused dispatch a block) |
| large folds stream / release weights (threshold: a 64 MiB pair tensor) | `c6f9133` `1d73c7d` `d20e773` `1a33d43` | OpenDDE 5CAJ-255 5440 -> 2349 MiB | `largeFoldReleasesWeights`, `largeFoldStreamsWeights` `=false` | the threshold is a memory-pressure guess made on a 40 GB card. On a 16 GB Mac it may want to be LOWER - measure where macOS starts paging (`fold.js --budget=0` prints peaks) |
| ESM-C keeps recorded int3 decodes, not decoded halves | `3fee383` | ESMFold2 59 res peak 1798 -> 957 MiB, repeat +40 ms (then won back by the row tile) | none (compare against `0145163^`) | the replay is 144 decode dispatches on the tower's critical path; Apple's dispatch cost differs |
| the vector split pair transition on every device from 192 channels | `cfead49` `7b032be` `0003df3` | IntelliFold-2 trunk ~2x | `--tune=pairTransitionSplit=false` | it stores its scratch in **f16 where the device has shader-f16** - which an M2 does and the A100 did not. That arm has never been run. Check numerics (`check-transition-split.js`) AND speed |
| AF2 vector OPM GEMMs default without matrix units | `bd6b10c` `bb57b3f` | AF2 `opm.contract` 1.60 -> 1.22 ms a block | `--tune=opmVectorContract=false` | same question as the AF3 one |
| the thread-a-row kernels re-tiled (template output, MSA attention weights, OPM projection) | `fe92b55` `75ede73` `dcfbdea` `0c4e07d` | 62.8 -> 1.7 ms etc., bit-identical | parent commit | tile sizes were occupancy-driven on the A100; likely still wins, cheap to confirm |
| trunk pair, confidence pair, recycle state kept on the device | `6b09846` `8278407` `28f0577` `414b832` ... | 1.4-3.0x a pass, byte-identical | parent commit | on unified memory the copies they remove were nearly free (CLAUDE.md: "the bus is free on an M2") - expect LESS gain, but it should never lose |

### 2. Cold-run work - measure on the page, where a visitor sees it

The user's standing point: **most visitors make ONE cold fold**. Measured on the
A100 page (`fold-in-page.py --model af3 --dev-report`, stock flags):

- stages now request all their pipelines before awaiting any (`6b46cf4`);
- the trunk's embedder/template/MSA stacks compile together at fold start with
  `compileOnly` (`2fc28e7`), and the template during the weight download
  (`d3f52d9`);
- the diffusion transformer fills its resident weights inside the warm that
  runs beside the trunk (`966c134`).

AF3 6MRR first fold 1.40 -> ~1.28 s; page cold trunk pass 1 705 -> ~520 ms.
`tools/cdp.py` now honours `LOCALFOLD_STOCK_FLAGS=1` (on macOS it already drops
only `--enable-unsafe-webgpu`).

**On the M2, ask:** `python3 tools/fold-in-page.py --model af3 --dev-report`
against the same on `main`, and `--keep-profile` for the second-visit number.
CLAUDE.md's round-three notes say an M2's first AF2 fold WAS its compile queue,
so Apple's compiler may make these matter more, not less.

### 2b. Since the handoff: Colab GPUs, and what does NOT reach an M2

Measured on Colab T4s and one L4 (docs/PERF.md, "A Colab T4" and "A Colab L4"):

- `runtimeLoopBounds: "tiered"` (turing and lovelace priors only): each
  kernel's first pipeline compiles with opaque loop bounds and is GENERIC over
  its u32 constants (read from a uniform in bind group 1), so a new protein
  length reuses it; the specific, unrolled kernel compiles behind the fold.
  Bit-identical everywhere checked. 🔴 **IT IS EVERY DEVICE'S DEFAULT NOW, SO
  AN M2 RUNS IT** - measured only on NVIDIA. Its control arm is
  `--tune-json={"runtimeLoopBounds":false}`; compare a first fold and the next
  few on the page, and if Apple's compiler is fast enough that it only costs,
  `metal-3` should set it false.
- a `lovelace` prior (ampere's settings + tiered + `attentionMatrix: false`).
- `gpu-chrome.mjs --prior=NAME` answers with another architecture's prior.
- `tools/gpu/fold.js` now applies `--tune` BEFORE its pipeline warm; earlier
  `--tune` arms that changed a kernel warmed the prior's kernels first.
- Tried on the T4 and not taken: the triangle projection on the matrix units
  (a cold fold pays more than a warm one saves), capping concurrent compiles,
  a tiled distogram (its 900 ms was the stage timer absorbing queued work).

### 3. Correctness to re-check where the backend CLAMPS

Metal clamps an out-of-range write where Vulkan discards it (the 160-residue
race). New or rewritten kernels on this branch:

- `createAf3OpmVectorOutputShader` (webgpu/af3/trunk/outer-product-mean.js):
  rows are guarded; columns and the inner dimension are NOT, and the caller
  refuses shapes that do not divide (`pairChannels % 64`, `C^2 % 16`). Check
  that refusal holds for every bundle you fold.
- the fused int5/int3 decode (webgpu/weights/quantised-upload.js): one dispatch a
  recording, a workgroup per 512 slots, `slot < slots` guarded.
- the deferred replay rides a wrapped `queue.submit` that re-installs itself if
  replaced (`ae36c4a`) - a second store's second fold read pLDDT 5.3 before
  that fix. `tools/gpu/probe-submits.js --tool=fold --folds=2 --repeats=3`
  must fold the same pLDDT three times.

`npm test` (includes `folded-grid-guard.test.js`), then
`probe-grid-overdispatch.js` should still say CLAMPS for the stripped arm and 1
for the shipped kernel.

## Dead ends already paid for on the A100 - do not re-run them there first

All in docs/PERF.md: packed-half diffusion weights without shader-f16 (moot on
an M2, which has f16); a tiled transpose for the decode kernel (~70 G el/s,
transposes scatter); `grid.attend` bias staging and a flash-style chunk
softmax; the single transition as the vector split (slower: few rows, few
workgroups); `mappedAtCreation` uploads; ESMFold2 SwiGLU below tile 4. 🔴 Some of
these were occupancy arguments on a 108-SM card and could come out differently
on ~10 cores - the single-transition split and the SwiGLU tile especially.

## The gates, in the order to run them

```
npm test                                    # 1299/0 on the A100
npm run test:stock                          # macOS: drops --enable-unsafe-webgpu
npm run test:template test:ligand test:modified test:batch test:cache
npm run test:portable test:spec-floor       # 20 min each
npm run test:delta
```

`tools/gate-baseline.json` is keyed on the adapter, so the M2's signatures are
its own; a moved signature on the M2 needs the control arm run before it is
believed or re-recorded.

## What the A100 could not answer

1. Every f16 arm of the above - the M2 is the only box here that runs them
   stock.
2. Whether unified memory changes the release/stream thresholds (1.).
3. Whether the row-tile and occupancy choices (1.) invert on ~10 cores.
4. The cold run on Apple's shader compiler (2.).

Report back in the same shape the earlier rounds did: numbers per arm, same
box, and which knob should become an Apple-prior entry in
`webgpu/runtime/device-profile.js` rather than a default change.

## 🔴 THE M2'S REPLY (2026-09-26)

Apple M2, 10 cores, 16 GB, macOS 13.2, headless Chrome 154, **stock flags**
(`shader-f16` yes, subgroup matrices no). `opus-55-opt` with `origin/main`
merged. Every row is arm against arm on this box, interleaved, two rounds
unless noted. Bundles here: af3-int5, af2-monomer-int5, esmfold2-int5 -
nothing else, so every other model is untested on Apple.

### 🔴 A BUG ONLY A DEVICE WITHOUT THE AMPERE PRIOR COULD SEE: A SECOND FOLD DIED

`fold.js --folds=2` died in fold 2 with `[Buffer "int5-codes"] used in submit
while destroyed` and never finished (the harness hung; the page would have too).
`releaseStreamedWeights` destroys a prefix's recorded codes at a fold's end, but
cannot take them out of `recordings.byKey` - a WeakMap - so the next fold,
streaming again, replayed destroyed buffers. It runs only where
`keepTrunkWeights` is not true, which is every device but the ampere prior:
**every visitor's second fold**. Streaming off (`streamTrunkWeights=false`) was
the only arm that survived; f16 and flags made no difference.

Fixed: `releaseRecording` marks the recording and `residentWeightBufferFilled`
treats a released one as a miss and records again. Three folds in one process
then match the unstreamed arm to every digit (85.83089054918456), and
`probe-submits --folds=2 --repeats=3` is 85.8 / pTM 0.741 three times.
**Gated**: the AF3 row in tools/gate-folds.mjs now folds twice with
`--tune=keepTrunkWeights=false`, so an A100 takes the release path too - watched
failing with the fix reverted.

### Section 1, measured

| change | M2 result | verdict |
|---|---|---|
| AF3 OPM as two GEMMs | 255 tok x 1024 rows: OPM 1312 -> 645 ms (2 passes), steady pass 6850 -> 6240 (**-9%**), +16 MiB. 68 x 128: 15.0 -> 11.8 ms, pass unmoved | keep |
| AF2 vector OPM | warm fold 59 res 1117 -> 1095 ms, 200 res x 256/512 rows 11150 -> 10620 (**-4.8%**), checksum identical | keep |
| ESMFold2 token row tile | sampler (11 steps) tile 1: 271-292, **2: 254-256**, 4: 269-273, 8: 349-353; checksum -133920 in every arm | 2 is right here too |
| ESM-C row tile | LM 178-216 ms across 1/2/4/8, flat | no prior |
| small-fold trunk streaming | 5CAJ-255: peak **952 -> 751 MiB**, warm 15.7-16.1 s both arms, pLDDT identical | keep (after the fix) |
| large-fold release/stream | 400 res: peak 1063 -> 684 MiB, 37.5-38.2 s both arms, ~400 pageouts either way | keep; no case for a lower threshold - nothing here comes near a 16 GB machine's pressure |
| vector split, f16 scratch - numerics | `check-transition-split.js --vector=1` (new arm): relRMS **4.3e-4** f16 / 5e-7 f32 at 256, 384, 512 | sound |
| vector split - speed, isolated | `bench-transition.js` `vsplit` (new arm) vs the fused kernel at its shipped tile: 256 ch **1.02-1.07x**, 384 **2.0x**, 512 **2.5x** | wins where wide |
| vector split - speed, in ESMFold2 (256 ch) | trunk 462 -> 494 ms at 59 res, 11.7-12.1 -> 12.3 s at 200; peak +72 MiB at 200 | **loses at 256** |

🔴 **ONE PRIOR CANDIDATE: `pairTransitionSplitMinChannels: 320` for metal-3**
- keeps 256 fused and 384/512 split. Measured on ESMFold2 only; protenix2 and
boltz2 are also 256 and are not on this box, so it is a recommendation, not
a change.

`check-transition-split.js` without `--vector=1` checks the MATRIX split, which a
stock browser never runs - on this M2 it silently took the 8x8 units under the
developer flag. `bench-transition.js` likewise priced only the matrix split.

### Section 2, the page (AF3, 6MRR, single sequence, `--dev-report`)

| | main | branch |
|---|---:|---:|
| first visit, fold done at | 2762-2849 ms | **2653-2680** |
| returning visit (`--keep-profile`) | 2544-2602 | 2512-2525 |
| sampler start | 183-216 | 130-132 |
| **page peak** | **430 MiB** | **552 MiB** |

-4% on a first fold, -1-2% on a return. 🔴 **The peak rises 121 MiB**: the
diffusion transformer's 378 MiB of resident weights now fill during the trunk
(`966c134`) instead of after it. Not a problem on this machine; worth a line
wherever the page's peak is quoted.

### Section 3, correctness where Metal clamps

`npm test` 1300/0. `probe-grid-overdispatch.js`: shipped 1,1,1; stripped arm
CLAMPS (367, 4117, 273). Gates, stock flags: test:stock AF2/AF3/ESMFold2 ok;
test:template af3 0.241 / af2 2.497 A; test:ligand af3 0.061; test:modified
af3 0.98, esmfold2 0.99. test:cache and test:batch cannot run here (one model,
no dumps). Not run: test:portable, test:spec-floor, test:delta (needs bundles
this box lacks).

🔴 A missing bundle is a FAIL, not a SKIP, in test:stock and test:template
whenever the tool is fold-opendde.js or takes `--sequence=`/`--target=`:
`SyntaxError: Unexpected token 'o', "not found" is not valid JSON` - the
server's 404 body parsed as a manifest. The fold.js `--model=` rows skip
correctly.

## 🔴 ROUND SEVEN ASK: the register-tiled attentions (2026-10-03)

`gridAttendTiled` (AF3's `grid.attend`) and `attentionTiled` (AF2's flash attentions) are the
register-tiled vector kernels in docs/A100.md's last section - 1.32x to 1.46x on the A100's STOCK
path, which is the path an M2 always runs (no matrix units at 16x16). They are set in the ampere prior
only, because webgpu/kernels/attention.js records a kernel on this part that got 4.7x SLOWER when a lane
held two queries' registers, and this one holds sixteen scores and sixteen accumulators a lane.
Worth one run each way:

```
node tools/gpu-chrome.mjs tools/gpu/bench-trunk.js --model=/model-af3-int5/manifest.json --tokens=200 \
  --msa=1 --passes=1 --profile --tune-json='{"gridAttendTiled":true}'      # and false
node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --length=200 --sequences=128 \
  --tune-json='{"attentionTiled":true}'                                     # and false
```

and `check-af3-grid-attention.js --tiled` / `check-evoformer-attention.js --dimension=32
--tune-json='{"attentionTiled":true}'` for the answers. If it wins there it belongs in DEFAULT_TUNING.

## 🔴 ROUND EIGHT ASK: the large-length pass (2026-10-24)

Written from the A100. A pass over every port at 68 to 4,080 tokens (copies of 6MRR, which folds from its sequence
alone, each chain scored against the crystal). Most of what it fixed is cuda/-only (memory fit checks, the scratch
pool's fallback, AF2's launch limits and scratch release). **Three changes reach `metal/`, and one of them has never
been compiled** - this box cannot build Objective-C++. In order:

**1. Build it: e573a0a2 (PDB chain letters past Z).** metal/af2 named a chain `'A' + min(k, 25)` in its PDB and its
confidence JSON, so from the 27th chain on every chain wrote as Z and readers merged them; it now takes the page's 62
letters (`pdbChain` in metal/af2/main.mm), and metal/af3/output.mm's template-free fallback wrapped the same set at 26.
It is cuda/af2's 042f1442 as plain C++, unbuilt. Under 27 chains every letter is unchanged, so the gates should not
move:

```
bash metal/build.sh
python3 metal/af2/gate.py && python3 metal/af3/gate.py && python3 metal/ef2/gate.py
```

**2. AF2's monomer on a complex: 39cab21d (another agent's port) and 26c14ecf (the shared featuriser).** Two faults,
found because 12 copies of 6MRR, single sequence, read pLDDT 50 where ColabFold reads 80.7:
- a cross-chain pair took the multimer's relpos row 65, which a monomer's projection has as zeros (cuda/af2 995ef402;
  metal 39cab21d);
- a single-sequence complex folded on ONE alignment row where ColabFold gives each copy its own (the monomer merge
  dropped each chain's query row). AlphaFold 2 itself collapses many copies on one row: ColabFold's own JAX, patched
  to one row, reads 21.8 at 30 copies, as did we.

The second is in cuda/featurise, which metal links (`standalone.o`), so it reaches the M2 with no metal edit. The A100:
12 copies pLDDT **79.3** native, **79.1** on the page (WebGPU), ColabFold 80.7, ~0.9 A a chain. On the M2:

```
S=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE
metal/af2/fold x12.pdb --sequence=$(python3 -c "print(':'.join(['$S']*12))")
```

Expect ~79 and 12 chains A-L. 🔴 **A metal gate case that folds a single-sequence complex on the monomer WILL move**
(it has more alignment rows now) - that is 26c14ecf, not a regression: re-record it and say which moved, by how much.
cuda/af2's gate gained `6mrr-x12-monomer` for this (69.1 -> 79.3); metal/af2/gate.py could carry the same case.

**3. ESMFold2's long-input step floor: 4fa29d9f.** Past 2,048 residues the checkpoint's 15 scheduled steps (11 run)
leave the backbone SHORT - 45 copies (3,060 residues) had 15.9% of CA-CA outside 3.6-4.0 A, 60 copies 91.5% - so a
long input now takes the per-atom floor, 64 scheduled (45 run), in the page, python/localfold/worker.py and
cuda/featurise/standalone.cpp (which metal links). Euler is kept: the exponential update measured WORSE there at every
count (docs/EF2FAST.md, "Long inputs" - it never passes the denoiser's shrunken answer, where Euler's 1.638 step
overshoots it). Only if the M2 has the memory for it:

```
metal/ef2/fold x45.pdb --sequence=$(python3 -c "print(':'.join(['$S']*45))")
```

should print 45 steps run and keep CA-CA in band. A clean refusal or a clean out-of-memory is also an answer worth
writing down - nothing in this pass sized metal/ at these lengths, so its large-length behaviour is unmeasured.

Not reaching metal/, for completeness: OpenDDE's structural-stage fit check (a71ca0eb), the scratch pool's fallback to
the card (fc66ba51), chai-1's pre-MSA pair room check and AF2's tight-card scratch release (dd61b0d9), AF2's masked
column attention past 8,192 residues and the template scale's int count (995ef402). If metal/ has twins of any of
those - a fit check that sizes only the residue-level trunk, a pool that keeps pages across stages - they are worth a
look at the M2's own ceiling.

## 🔴 THE M2'S REPLY TO ROUND EIGHT (2026-10-24)

On the M2 (Apple8, 10 GPU cores, 16 GB, `recommendedMaxWorkingSetSize` 11.45 GB), at `6ac580e8`.

🔴 **`bash metal/build.sh` WITH NO ARGUMENTS BUILDS ESMFold2 ALONE** (`ports=(ef2)` is its default), so the command
above left `localfold-af2` and `localfold-af3` as they were and the first gate run measured the OLD binaries. Build
what you mean to gate: `bash metal/build.sh ef2 af2 af3 selftest`. Everything below is on binaries rebuilt that way.

**1. e573a0a2 compiles clean (no warnings) and does what it says.** All three gates pass on the rebuilt binaries -
AF2 7/7, AF3 18/18, ESMFold2 8/8, every case at its M2 baseline (AF3's 18 identical to the digit to the pre-rebuild
run). 30 copies of a 13-residue peptide write chains `A..Z a b c d` on both metal/af2 and metal/af3 (boltz2), in the
PDB and in af2's `token_chain_ids`; the old af2 binary wrote `A..Z` and then Z four more times. metal/af3's
template-free PDB fallback took the change and compiled, but no fold here reaches it (the featuriser's records are
used), so that line is built, not exercised.

**2. Already measured here before this round**: `6mrr-x12-monomer` was recorded in `gate-baseline.m2.json` after both
26c14ecf and 39cab21d (pLDDT **79.17**, RMSD 1.006 A, 12 chains A-L), and it reads the same today. The A100's 79.3 and
ColabFold's 80.7 bracket it.

**3. ESMFold2 at 45 copies is refused up front, cleanly**: `this fold needs about 36.1 GB ... past the 11.5 GB this
Mac's GPU can keep resident`, in 0.28 s. 🔴 **SO THE NEW STEP FLOOR NEVER RUNS ON A 16 GB M2**: it starts past 2,048
residues and this machine's ESMFold2 ceiling is ~1,690 (metal/README.md's figure, and the 36.1 GB estimate scaled to
11.45 says the same). It is reachable only on a Mac with more memory.

**4. The twin you asked about exists in metal/, and it was worse here than on the A100: OpenDDE's structural stage
was not in the fit check.** metal/af3's `foldInput` sized only the residue-level trunk. 8 copies of 6MRR (544
residues, **1,040 structural tokens**) estimated **3.66 GB** and peaked at **11.08 GB** - 97% of the working set - in
the confidence head's transition on the structural pair (`tr.g` 3.3 GB, twice the refiner's), with the refiner at
9.38 before it. 9 copies would have passed the old check at ~4 GB and paged. Fixed in metal/af3/main.mm the way
a71ca0eb fixed cuda/: when the dialect folds on structural tokens, the peak is also `allocated + 6 x` the structural
f32 pair (`sbatch.tokens`, read before the trunk runs; measured 5.86 at 544 residues). Now 8 copies estimate
**11.31 GB** against the measured 11.16 (footprint) and fold unchanged (pLDDT 53.21 both ways), and 9 copies are
refused in under a second at 14.0 GB. AF3's gate is identical after the change. The factor is one M2 measurement at
one length; the M5, whose tensor path may hold differently, should re-measure it with `LOCALFOLD_PEAK=1
LOCALFOLD_MEM=1` before trusting it there. Forcing 9 copies past the check (`LOCALFOLD_NO_FOLD_FITS=1`) was not run.

The other items on your not-reaching-metal list were not looked at beyond this one.

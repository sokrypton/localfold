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
  Bit-identical everywhere checked. **Not active on an M2** - `metal-3` does not
  set it. Whether Apple's compiler is slow enough on a first fold to want it is
  a question for `--tune-json={"runtimeLoopBounds":"tiered"}` on the page.
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

- `createAf3OpmVectorOutputShader` (src/af3/trunk/outer-product-mean-webgpu.js):
  rows are guarded; columns and the inner dimension are NOT, and the caller
  refuses shapes that do not divide (`pairChannels % 64`, `C^2 % 16`). Check
  that refusal holds for every bundle you fold.
- the fused int5/int3 decode (src/weights/quantised-upload.js): one dispatch a
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
`src/runtime/device-profile.js` rather than a default change.

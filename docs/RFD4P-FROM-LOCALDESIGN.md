# RFD4-Proteina: what `rfd4-proteina-webgpu` measured, for this copy

**Source:** `sokrypton/rfd4-proteina-webgpu` (the "localdesign" checkout), 2026-10-25/26, Apple M2 (10-core GPU),
Chrome 156, macOS 13. Full record: that repository's `SPEED-HANDOFF.md` and `M2-TO-M5.md`.

**Why it applies here:** `webgpu/rfd4p/engine.js`, `blocks.js`, `decoder.js` and `encoder.js` are byte-identical in
the two repositories as of `bed641ad`. Every kernel number below is a number for this tree.

Stock means `LOCALFOLD_STOCK_FLAGS=1`: no `chromium-experimental-subgroup-matrix`. Developer means
`--enable-unsafe-webgpu`, which adds 8x8x8 subgroup matrices.

## 1. 🔴 This copy loads its weights from that site

`shared/rfd4p/safetensors.js` falls back to `https://sokrypton.github.io/rfd4-proteina-webgpu/weights/int8` when
there is no local `./model-rfd4p-int8`. That site's owner has been told the path is a dependency. If RFD4 stops
loading here, check that URL first: `manifest.json` and `shard_000.bin` should both answer 200.

## 2. Both GPU paths are at the M2's ceiling

| | stock | developer |
|---|---:|---:|
| one trunk pass, n=100 | 286-303 ms | 132 ms |
| one trunk pass, n=208 | 1227 ms | 530 ms |
| `triProjG` rate | ~1.1 TFLOP/s | ~2.7 TFLOP/s |
| measured ceiling for the class | f32 GEMM ~1.1-1.3 | f16 8x8 matrix, 64x16 per subgroup: 2.75 |

- **Subgroup-matrix ceiling.** A synthetic kernel that only loads f16 8x8 tiles from workgroup memory and
  multiply-accumulates them peaks at:
  - 2.75 TFLOP/s for a 64x16 block per subgroup (`gemmSubmatSource`'s layout);
  - 2.98 for 64x32 and 2.92 for 32x64;
  - 2.06 for 16x16.

  Re-blocking to 64x32 could buy at most ~8% on `triProjG`, about 3% of a pass.
- **Triangle multiplications.** `tri-in` + `tri-out` take this share of a pass at n=200 / 400 / 600:
  - stock: 8% / 12% / 17%;
  - developer: 7% / 12% / 15%.

  Their raw rate looks low (stock 0.66-1.06 TFLOP/s), but that is mostly tile padding. 64x128 tiles round n=400 up
  to 448x512 (1.43x the work) and n=600 up to 640x640 (1.14x). Corrected for padding, they sit at each path's
  ceiling. 32x64 tiles pad less but run at 0.8x.
- **The one real gap left.** Past n=512 on stock, the chunked incoming triangle reads its operand by stride
  (`3762ee56`): 952 ms against 732 for outgoing at n=600. That is about 2% of a pass.

## 3. Tried and lost on this kernel set (do not repeat blind)

All variants were measured on stock unless noted, interleaved in one process, inside the real trunk:

| variant | result |
|---|---|
| f16 accumulators in `triProjG` (`vec4<f16>` x8, f32 epilogue) | 1.027x / 1.033x a pass at n=100 / 200; drift twice FP16's existing |
| `subgroupShuffle` broadcast of A inside the K-loop (`shuffleA`) | 0.52x |
| `vec4` A along K (`vecTa`) | 0.64x |
| f32 workgroup staging | 0.73x |
| 8x8 per thread, 128-row tile (`mt8`) | 0.77x |
| f16 FMA with per-k-tile partials | 0.89x, worse numerics |
| register prefetch of the next k-tile | 0.94x |
| small-tile GEMM for row counts up to 128 (token track, decoder at n=100) | 0.80x end to end |
| handing `onStep` off unawaited, so the next pass overlaps frame building | 1.000x |

- **f16 accumulators.** These give localfold 1.43-1.55x elsewhere (`6e996fa8`), because there the accumulators
  lived in a WGSL array and spilled. Here they are scalar `vec4` variables and do not spill.
- **Subgroup shuffles.** Both shuffle variants (`shuffleA`, `vecTa`) lose for the same reason: holding A live
  across the K-loop pushes the kernel past the M2's ~64 registers per thread, which halves occupancy. This is also
  why the developer path cannot be emulated on stock Chrome with plain subgroups. No origin trial for subgroup
  matrices existed as of Chrome 156.

## 4. What a frame costs

On the M2, stock, one all-atom decode per live frame costs:
- 33-37 ms at n=100;
- 68-73 ms at n=208.

That is about 14% of a 60-step design at n=100 when every step is decoded. `web/generate.js` asks for
`TRAJECTORY_FRAMES` by count (`decodeEvery`), so it pays roughly a quarter of that. The localdesign owner chose
every step there; the cost above is what that choice is.

Where a 60-step n=100 design's time goes there (stock, visible window, every frame):
- trunk 17.1 s (82%);
- decodes ~1.9 s (9%);
- viewer ~0.35 s;
- the rest ~1.1 s.

Per-step readback is free: 56.07 s against 55.87 s.

## 5. Habits these runs taught

- **Headless hides what a visitor's window costs.** The same change measured 1.059x in a visible window and 1.02x
  headless (device pixel ratio 1 against 2, compositor contention). `LOCALFOLD_HEADED=1` now runs
  `tools/gpu-chrome.mjs` visibly on macOS. Visible Chrome also ran the stock f16 trunk to n=800 with no device
  loss.
- **A gate that counts frames handed to a viewer stays green when the viewer draws nothing.** localdesign shipped
  blank live frames for a day: Mol*'s FXAA got `{}` params, so its thresholds were undefined. Its page check only
  counted render-hook calls. After any viewer change, sample the canvas: serve the page with WebGL
  `preserveDrawingBuffer: true` patched in and count non-background pixels per frame.
- **Old Safari fails at parse time, on the whole module.** Safari before 16.4 cannot parse regex lookbehind. One
  literal in a bundled dependency (`mdast-util-gfm-autolink-literal`, pulled in by Mol*) stopped the Mol* chunk from
  loading at all. `npm run check:safari` here parses every shipped module with the Mac's own JavaScriptCore; all
  264 passed on Safari 16.3's engine when it was added.
- **A design on the CPU path freezes the page.** With WebGPU hidden, a 2-step design on the CPU fallback blocked
  the main thread for 375 s, with the last status still on screen. localdesign now disables Generate without
  WebGPU. This page already refuses up front (`web/design.js`), which is the right behaviour; keep it that way.

## 6. 🔴 The same idea on the FOLDING models: not worth the flag on an M2 (2026-10-26)

**Question:** subgroup matrices made localdesign's trunk 2.3x faster. Does that carry over to AF2, AF3 and ESMFold2
on WebGPU?

**Answer: no.** Measured on this M2 with the developer flag on, interleaved against the shipped tuning. The
experiment code was reverted. If an M5 or a later chip changes the answer, the notes below say how to rebuild it.

**Why the flag buys localdesign 2.3x and folding almost nothing.**
- RFD4-Proteina's trunk is almost all plain GEMMs. Folding models are not:
  - at 400 residues an AF2 block is ~72% GEMM-shaped and ~21% attention;
  - an AF3 trunk is ~65% GEMM-shaped (fused triangle and grid kernels) and 26% `grid.attend`;
  - small folds are dominated by the diffusion head's latency-bound steps.
- Our stock f16 vector GEMM already runs AF2's MSA transitions at 1.15-1.22 TFLOP/s. That is the same as
  localdesign's stock ~1.1, so the stock path has no slack. The win is only where the units beat it.
- The `metal-3` prior keeps every matrix kernel off. So with the flag on and nothing forced, a fold is identical in
  speed and answer: AF3 6MRR 3.16-3.30 s against 3.14-3.16 s, AF2 59-mer checksum unchanged.

**The staged GEMM (`matrix-linear.js`) on 8x8 units.** AF2's four transition GEMMs, 400 residues by 256 rows:

| arm | ms |
|---|---:|
| vector f16 (shipped) | 129 |
| matrix, the derived 64x64 block (64x8 a subgroup, 8 accumulators) | 103 |
| matrix, localdesign's 64x128 (64x16 a subgroup, 16 accumulators) | **86 (1.5x)** |
| ...with f32 8x8 tiles instead of f16 | 105 |
| ...with the staged A panel padded by 4, 8 or 16 | 86-90 |
| ...with register prefetch | no change |

It levels off at ~1.9 TFLOP/s against localdesign's 2.7. The AF2 fold checksum is identical either way (f32
accumulation reorders sums only).

Whole AF2 folds with `matrixLinear` on and the 64x128 block:
- 1.10x at 59 residues (1097-1121 against 1206-1244 ms);
- 1.10x at 200 (10.31 against 11.38 s);
- 1.05x at 400 (33.6 against 35.3 s).

**The flash attention (`attention-matrix.js`) on 8x8 units.**
- **No f16-in, f32-out config.** An M2 offers only f16 -> f16 and f32 -> f32 at 8x8x8. The kernel was generalised
  so a subgroup's 16 query rows become two 8-row matrix blocks; the softmax pairing is unchanged.
- **Results** (`check-attention-variants.js`):

| shape | shipped register kernel | matrix f32 -> f32 | matrix f16 -> f16 |
|---|---|---|---|
| 59 keys | ~62 ms | 66.6 ms (best tile 4x16) | 55 ms |
| 400 keys | 40.6-42.4 ms | 40.0 ms | 38.4 ms |
| relRMS | | 3-4e-5 | **6-9e-3** |

  f32 breaks even. f16 is ~1.1x but ten times the matrix bar's error, from accumulating the logits in f16.
- **Tiles:** keyTile 16 beat 32 and 64 in both precisions. f32 4x32 and 2x64 do not fit in 32 KiB.

**Everything forced on** (`matrixLinear`, `attentionProjectMatrix`, `opmMatrixContract`, `triangleProjectMatrix`,
`gridAttendMatrix`, `attentionMatrix` f32 4x16, `stagedMatrixPrefetch`), warm folds, two rounds:

| model and length | gain |
|---|---|
| AF2, 59 residues | 1.08x |
| AF2, 200 residues | 1.01x (`matrixLinear` alone was 1.10x there, so another knob gives it back) |
| AF3, 6MRR | 1.04x |
| AF3, 200 residues | 1.04x |
| ESMFold2, 200 residues | none |

All folds stayed sound; pLDDT moved by at most 0.09.

**So on a Mac the flag is not worth asking a visitor for.** The fast Mac path is the native Metal port
(`localfold serve`), which already uses `simdgroup_matrix`. The flag matters on NVIDIA under Linux, where stock
Chrome also lacks `shader-f16` and a fold is ~2x slower (docs/A100.md).

**If an M5 should re-ask this:**
- the GEMM block is `--sweep=matrixLinear={"blockRows":64,"blockColumns":128}` on `profile-af2-block.js`, which
  needs no code change;
- the 8x8 attention needs rebuilding: `ATTENTION_MATRIX_UNIT` stays the row count (16), the unit and the element
  and result types become geometry, and `supportsAttentionMatrix` must check the config with those types.

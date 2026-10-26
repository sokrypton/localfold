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

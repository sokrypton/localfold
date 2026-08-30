# LocalFold

LocalFold is an end-to-end WebGPU implementation of AlphaFold 2 monomer model 1. It accepts either a raw amino-acid sequence or A3M text, runs recycling, the extra-MSA stack, all 48 main Evoformer blocks, the eight-layer structure module, atom geometry, and pLDDT/PAE heads.

All learned model operations execute in WGSL. The CPU is used only for non-neural input preprocessing, scheduling, readback, and confidence aggregation. There is no ONNX runtime and no CPU neural-network fallback.

There is also no build step. `src/` is plain ES modules that the browser and node both load as written, so running the demo is a static file server and nothing else:

```bash
python3 -m http.server 4173     # then open http://127.0.0.1:4173/
```

npm is not required to run the demo, the CPU tests, the benchmarks, or the Pages deployment. It is needed only for the two lanes that genuinely depend on a package: headless GPU tests (Dawn) and browser tests (Playwright).

### Running with no server at all

A page opened as `file://` cannot load ES modules — the origin is `null`, and Chrome allows cross-origin module loads only for http, https, data and extension schemes — and it cannot `fetch()` either. Both are worked around by one artifact:

```bash
python3 tools/bundle.py     # -> localfold-local.html, self-contained
```

That collapses the module graph into a single classic script and inlines the stylesheet and py2Dmol, so the file can be double-clicked, emailed, or carried on a stick.

The weights need a second door, because `fetch()` is unavailable on a file page too. A `file://` page has exactly two ways to reach bytes on disk — a classic `<script src>`, which was never subject to the same-origin read rule, and a `data:` URL, which carries its own bytes rather than asking an origin for them. `tools/export-js-weights.py` uses both:

```bash
python3 tools/export-js-weights.py --check    # model/*.f32.bin -> model/*.js
```

Each shard becomes a script assigning a base64 `data:` URL, and `ScriptTensorStore` injects the script then hands the URL to `fetch`, so the base64 decode happens in the browser's C++ rather than a JS loop. Two shards are in flight at a time and each string is dropped the moment it decodes, because a 48 MiB shard is 64 MiB of base64 and holding all eight would cost half a gigabyte for nothing.

Base64 is four characters per three bytes, so this writes about a third more than it reads: **355 MiB of shards become 474 MiB of script.** That is the price of the doors, and it is why the served site keeps the binaries — `build_site.py` excludes `weights-*.js` from `dist/`, since nothing on an http page can use them and shipping both would take a 356 MiB site to 830 MiB.

To point the page at a model hosted somewhere else, pass `?model=<manifest url>`.

The bundler is a second artifact, not a stage in front of the first — `python3 -m http.server` still runs the checkout exactly as written. Nothing is renamed or minified, so a stack trace from the bundle still names the function you wrote.

## Acknowledgments and citation

LocalFold is an independent browser/WebGPU port of AlphaFold2; it does not originate the protein-structure prediction method or model parameters. We thank the AlphaFold team at Google DeepMind for developing AlphaFold2 and releasing its source code and parameters. The scientific method should be credited to the original publication:

> Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

Please also consult and cite the [official AlphaFold repository](https://github.com/google-deepmind/alphafold) as appropriate. The bundled AlphaFold parameters remain subject to DeepMind's [CC BY 4.0 parameters license](https://github.com/google-deepmind/alphafold/blob/main/WEIGHTS_LICENSE); this repository does not alter their ownership or license.

## Verified reference predictions

The acceptance sequence is:

```text
PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK
```

Against official `alphafold2_ptm_model_1_seed_000` FP32 intermediates:

| Input | Recycle | WebGPU pLDDT | Official pLDDT | WebGPU pTM | Official pTM |
|---|---:|---:|---:|---:|---:|
| Sequence only | 0 | 57.000 | 56.994 | 0.37598 | 0.37594 |
| Sequence only | 3 | 64.517 | 64.511 | 0.43466 | 0.43464 |
| `test.a3m` processed oracle | 0 | 96.615 | 96.625 | 0.75294 | 0.75293 |
| `test.a3m` processed oracle | 3 | 96.049 | 96.063 | 0.75355 | 0.75342 |

The literal 8,076-row `test.a3m`, independently parsed and clustered in JavaScript, produced 96.82 pLDDT and 0.7548 pTM after its first WebGPU recycle.

The weights ship quantised, so the browser does no rounding at all. `tools/quantize_model.py` rewrites the exported shards as **int8 with a float16 scale per 64-weight block**, taking the download from 355 MiB to **97.3 MiB**; `src/reference/dtype.js` dequantises each tensor to `Float32Array` once as it is read, because the shaders take `f32`.

How far that can be pushed was measured rather than assumed - each row is a four-pass fold of the 59-residue reference against 63.4 pLDDT / 0.421 pTM at float16:

| format | size | vs float32 | pLDDT | Δ |
|---|---:|---:|---:|---:|
| float32 | 355.3 MiB | 1.0× | — | — |
| float16 | 181.5 MiB | 2.0× | 63.4 | baseline |
| **int8, block 64** | **97.3 MiB** | **3.7×** | **63.3** | **−0.1** |
| int6, block 32 | 78.3 MiB | 4.5× | 62.9 | −0.5 |
| int5, block 32 | 67.5 MiB | 5.3× | 55.6 | −7.8 |
| int4, block 32 | 56.6 MiB | 6.3× | 52.8 | −10.6 |

Below six bits the model falls off a cliff. The reason is bias rather than magnitude: a block's weights are not centred on zero — the midpoint sits a median 17% of the half-range away — so symmetric quantisation leaves a systematic mean shift in every block, and a mean shift compounds across 48 Evoformer blocks where zero-mean rounding noise averages out. Adding a zero point is worth 6.1 pLDDT at five bits (55.6 → 61.7) and 3.7 at four (52.8 → 56.5), but only 0.1 at eight, where the step is already small enough that it is noise. So int8 ships symmetric, and the note in `tools/quantize_model.py` says to add a zero point if it ever drops below six bits. Getting four bits to work would need error-compensating quantisation of the GPTQ/AWQ kind, which is a different project.

Quantised weights are high-entropy, so gzip only manages 1.06–1.12× on any of these: the on-disk size is the wire size.

🔴 **The structure module and the geometry tables stay float32.** The structure module composes rigid transforms across eight iterations, so an error in one frame is carried into the next and lands in the coordinates - AlphaFold's own capture records it as `float32`, and the geometry tables are not learned weights at all but the residue-constants literals, where rounding an ideal atom position moves an atom by construction. Keeping them costs 3.9 MiB: the main Evoformer stack alone is 94.3% of the parameters, structure and geometry together 2.2%. Verified against the float32 originals, every retained tensor is bit-identical and the worst relative error among the rounded ones is 4.3e-4, which is fp16 precision.

This replaced a runtime pass that rounded all 92.8 million weights on **every fold**, into a freshly allocated 371 MiB tree in the middle of the prediction. The conversion was never the expensive part - it is 210 ms if done in bulk, 961 ms mapping a callback per weight - but the allocation was: measured in Chrome, a single-pass 59-residue fold took 7.1-7.4 s that way against 1.0-1.1 s without it, and four passes went from 11.9 s to 4.3 s. Predictions are unchanged.

Development-host timings on an NVIDIA GB10 were 9.75 and 9.76 seconds for four A3M passes (508 clustered + 1,024 extra rows) when the optional key-parallel subgroup fast path was available. Run `node tools/benchmark-a3m-model.js` to reproduce the full-model measurement. The original correctness-first scheduler took approximately 95 seconds. These are engineering measurements, not cross-device claims.

## Public API

Load a browser-hosted model manifest and predict a sequence:

```js
import { AlphaFoldFixture, AlphaFoldQueryOnlyGpu, HttpTensorStore, requestAlphaFoldDevice } from "./src/index.js";

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("WebGPU unavailable");
const device = await requestAlphaFoldDevice(adapter);

const model = AlphaFoldFixture.fromStore(
  await HttpTensorStore.open(new URL("/model/manifest.json", location.href)),
);
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] =
  await Promise.all([
    model.embeddingWeights(), model.templateWeights(), model.extraPairStackWeights(), model.mainStackWeights(),
    model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
  ]);

const prediction = await new AlphaFoldQueryOnlyGpu(device).predictSequence(
  "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
  { embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry },
  featureTables,
  { recycles: 3, randomSeed: 0 },
  await model.tensor("confidencePaeBreaks"),
);
console.log(prediction.final.confidence.meanPlddt, prediction.final.confidence.ptm);
```

For A3M input, load `extraStackWeights()` and call `AlphaFoldMonomerGpu.predictA3m(...)`. `makeA3mFeatures(...)` is also exported for applications that want preprocessing and inference as separate steps.

The manifest is a JSON tensor table whose values are little-endian float32 binary files. `HttpTensorStore` fetches tensors lazily and caches them. `FileTensorStore` provides the equivalent Node test/development loader.

## Architecture and memory

Implemented components include input/recycling embeddings, mock-template pair embedding for model 1, MSA row and column attention, extra-MSA global column attention, transitions, OuterProductMean, both triangle multiplication directions, both triangle attention orientations, IPA, backbone updates, sidechain torsions, atom14/atom37 geometry, pLDDT, PAE, and pTM.

Attention uses online softmax and never materializes attention-logit cubes. When `subgroups` and `subgroup-size-control` are available, eight 32-lane subgroups process eight queries while sharing a 32-key K/V tile. Within each subgroup, the lanes calculate 32 key scores concurrently, reduce the tile softmax, and shuffle probabilities while accumulating one output channel per lane. Other devices use the portable workgroup kernel. The earlier lane-per-channel 8x16, 8x32, 8x64, 16x64, 32x64, and 64x64 variants remain selectable for differential benchmarking. QKV/gate projections use register-blocked 16x16 tiles, attention output uses 16x32 tiles, and transition GEMMs use 16x64 tiles. Triangle multiplication uses cooperative 16x16 joint tiles for its split A/B projection and gates and for its output projection/gate, inspired by the fused split projection in [steineggerlab/alphafold](https://github.com/steineggerlab/alphafold), and never materializes an `O(L³)` tensor. For deep, short MSAs, OuterProductMean uses AF2's canonical outer-first contraction (13.6 MiB at length 59); a 32-sequence bounded path remains available when that temporary would exceed 64 MiB.

Evoformer blocks are submitted ahead without host waits and alias a pooled set of scratch buffers. WebGPU queue ordering preserves block dependencies while reducing the 48-block A3M trunk from about 21.7 seconds to about 2.3 seconds. Final projections commit directly into residual tensors, reducing the measured trunk scratch peak from 767 MB to approximately 664 MB. Embedding, extra-MSA, and main-stack activations stay device-resident across stage and recycle boundaries; only the first MSA row and pair representation required by the current structure API are read back.

The pair representation is the largest tensor in the model, at `L*L*128` floats: 1.7 MiB at length 59, 25 MiB at 221, and 44 MiB at 300. The single-sequence path used to move it across the bus nineteen times per recycle. The input embedder now writes it into one buffer that the template residual, the extra-MSA stack, and the main stack all mutate in place, and the eight structure-module IPA iterations upload and layer-normalize it once between them rather than once each. Four crossings per recycle remain: the main stack's readback, and the uploads the structure module, the confidence heads, and the next recycle's embedder each make from the host copy.

## Browser demo and GitHub Pages

There are two pages, both on the one implemented parameter set, `model_1_ptm`.

**`index.html` is [py2Dmol](https://github.com/sokrypton/py2Dmol)'s own application**, with one panel swapped: where it fetches a PDB ID or takes an upload, it folds a sequence. Everything else on that page — the viewer, the sequence strip, the MSA and PAE panels, selections, sessions — is py2Dmol's, running its own code, wired by its own `app/main.js`. None of it is reimplemented here, and none of it should be.

A prediction reaches it the way a loaded file does, without being one. py2Dmol's ingestion takes *virtual* files — a name and a reader — because a ZIP entry was never a `File` either, so `web/app.js` hands the fold straight over through `window.py2dmolLoadFiles`:

```js
load([{ name: "prediction.pdb",         readAsync: () => Promise.resolve(pdb) },
      { name: "prediction_scores.json", readAsync: () => Promise.resolve(scores) },
      { name: "prediction.a3m",         readAsync: () => Promise.resolve(a3m) }], true);
```

Nothing is written to disk and no `File` is manufactured. Extensions decide what each one *is*: `.pdb` becomes the structure (one model per recycle, walked by the play bar), `.json` is paired to it as PAE, `.a3m` populates the MSA panel. 🔴 The **names are load-bearing** — a PAE matrix is matched to its structure by a fuzzy basename score that rewards `pae`, `scores`, `full_data` and `aligned_error`, and renaming these leaves the PAE panel empty without complaining. Downloading the raw prediction is separate and explicit, under the PAE panel: py2Dmol's own Save writes a *session*, those two buttons write what the **model** produced.

🔴 **py2Dmol's own fetch panel is still in the page, hidden, and has to stay.** `app/main.js` binds listeners to `#fetch-id`, `#fetch-btn`, `#upload-button` and the option toggles during `initializeApp`, and one missing element throws inside `setupEventListeners` — which silently aborts the *rest* of `initializeApp`. The symptom is not an error message; it is that the MSA panel never wires itself up and a fold appears to hang. Hidden is safe. Deleted is not.

`single.html` is the teaching version and is still LocalFold's own standalone page: one sequence, no alignment, click a residue in the structure to mutate it and re-fold, with the change morphed into the picture. It is next to move onto the same application.

That split is why there are two vendored py2Dmol bundles. `index.html` loads `web/vendor/py2Dmol.full.min.js` (the `full` target: the website, the panels, *and* the embed API, since the morph needs `framesFromText`); `single.html` still loads `py2Dmol.embed.min.js`. Both are vendored rather than fetched from a CDN, so the pages draw with no network access. Confidence colouring is a single `setColor("plddt")` against py2Dmol's own AlphaFold ramp, read out of the B-factor column the PDB writer already fills.

`dev.html` is a kernel diagnostic and is not part of either page.

Styling: `web/localfold.css` holds only what LocalFold adds to py2Dmol's page, and **no rule in it uses a bare element selector**. `web/style.css` — the dark theme for the standalone pages — sets `body` and restyles `textarea, input, select` globally; loaded over py2Dmol's light application it keeps their layout while turning every text box black.

Alignment depth is **508 clustered rows and 1,024 extra rows**, set in `src/input/a3m-features.js`. 508 is AlphaFold's own effective number for a templated `model_1` — it asks for 512 clusters and gives four rows to templates — and matches the left half of ColabFold's `--max-msa 512:1024`. The 1,024 is a deliberate reduction: AlphaFold's `model_1` sets `max_extra_msa: 5120`, and 1,024 is what its template-free models use and what ColabFold's preset selects to make a run cheaper. The extra-MSA stack is the most expensive part of an A3M fold, so 5120 would cost roughly five times that stack. Measured, the reduction is not visibly hurting the reference case: on the 8,076-row `test.a3m` the JavaScript clustering reaches 96.8 pLDDT against the captured AlphaFold reference's 96.625 at the same recycle — though that is a shallow alignment of a 59-residue protein, which is where a smaller `max_extra_msa` is least likely to bite. Neither value is exposed in the page; both are the defaults for every fold.

MSA search runs against the public ColabFold MMseqs2 server (`api.colabfold.com`) through `src/input/mmseqs2-api.js`, ported from [upstream](https://github.com/martin-steinegger/alphafold2-webgpu). It is the only request either page makes off the machine, which is why it is a mode the reader selects rather than a default — the status line names the server while it waits. Pasting or uploading an A3M keeps everything local.

Over http the page reads `./model/manifest.json`; on a `file://` page it reads the base64 weight scripts beside it. `?model=<url>` overrides either.

Nothing has to be built to open the page from a checkout. `tools/build_site.py` exists only to assemble a *deployable* directory — the same files, minus `test/`, `tools/` and the fixtures, which together are far larger than the app:

```bash
python3 tools/build_site.py            # dist/, no model parameters
python3 tools/build_site.py --model    # ...and the exported model/ too
```

It is a copy with an allow-list, not a bundler: the layout is preserved exactly, because `index.html` refers to `./web/app.js` and that refers to `../src/model/...`.

The model exporter copies only the 335 tensors required for inference and discards captured activations and reference outputs. It packs them into eight balanced binary shards to avoid hundreds of HTTP round trips. The resulting model-1 PTM download is approximately 355 MiB, versus 419 MiB for the query reference fixture and 1.1 GiB for the A3M reference fixture. It remains float32 to avoid silently changing predictions. Full-model captures are intentionally excluded from the published source history.

The Pages workflow deploys the UI on pushes to `main`. To keep Git history small, the reduced browser model is stored once in the private repository's `model1-ptm` GitHub Release as `localfold-model1-ptm.tar.gz`; the workflow downloads that asset before constructing the Pages artifact. Because GitHub Pages is normally public even when its source repository is private, the workflow does not publish model parameters unless the repository variable `LOCALFOLD_INCLUDE_MODEL` is explicitly set to `true`. Without the bundled model, enter a CORS-enabled manifest URL in Advanced settings.

From a development checkout containing the full model fixture, prepare the release asset with:

```bash
node tools/export-web-model.js <full-model-manifest>
mkdir -p artifacts
tar -czf artifacts/localfold-model1-ptm.tar.gz model
```

Create a private-repository release tagged `model1-ptm` and attach the archive. The archive contains a top-level `model/` directory, so the Pages workflow can extract it directly into the built site.

After pushing the repository, select **Settings → Pages → Source: GitHub Actions**. To publish the model with the demo, add the Actions variable under **Settings → Secrets and variables → Actions → Variables**. The full Pages artifact is about 356 MiB and remains below GitHub's 1 GiB Pages artifact/site limit.

## Development

```bash
python3 -m http.server 4173                                    # the demo
node --test --test-concurrency=1 test/*.test.js                # CPU tests
python3 tools/build_site.py                                    # dist/
```

Those need nothing installed. The two lanes that do:

```bash
npm install                                                    # Dawn, Playwright
LOCALFOLD_GPU_TESTS=1 node --test --test-concurrency=1 test/*.gpu.test.js
npx playwright test
```

Types live in JSDoc on the public surface. `npx tsc -p tsconfig.json` will check them and is entirely optional; nothing in the repository consumes its output.

GPU tests use Dawn's native Node WebGPU implementation. The suite contains operator-level official AlphaFold differential tests, complete block/stack tests, four-recycle query-only inference, four-recycle A3M inference, and a literal raw-A3M acceptance test.

Small reference fixtures needed by the default test suite are committed. Full Evoformer/model captures can be regenerated with the scripts under `tools/`; they require a ColabFold/AlphaFold JAX environment and official model parameters and are not part of published Git history.

## Current scope

- Monomer `model_1_ptm`, FP32 inference, mock templates, and fixed model channel sizes are supported.
- A3M sampling and clustering are implemented in JavaScript with a deterministic application PRNG. It is distribution-equivalent to AlphaFold preprocessing but does not reproduce TensorFlow's private RNG stream unless exact masked-MSA codes are supplied.
- Multimer models, real template hits, other AF2 parameter sets, weight quantization, remote MSA search, and result relaxation remain outside the current scope.
- Holding the pair representation on the device through the structure module and confidence heads, and across the recycle boundary, would remove the four crossings per recycle that remain.
- Requesting raised device limits would let long inputs skip chunking where the hardware allows it. `requestAlphaFoldDevice` currently asks for none, so a device reports the 128 MiB default storage binding even on an adapter offering 4 GiB. Upstream's `planMonomerDevice` works out what an input needs and requests only that, rounded up by power-of-two tiers; chunking is the safety net either way.
- Caching each block's packed weights on the device across recycles was tried and reverted. It works - 456 uploads on the first pass, 1368 cache hits over the next three, and the packing paid once - but a four-pass 59-residue fold measured 4.3 s either way. The fold is GPU-bound and the uploads already overlap with compute; on Apple Silicon, where host and device share memory, `writeBuffer` is close to free. The cache cost 345 MiB of resident GPU memory against roughly 7 MiB for transient per-block weights, so it was memory spent for no time saved. It may be worth revisiting on a discrete GPU, where the same traffic crosses PCIe. The note is kept in `src/runtime/execution.js`.
- Transitions are row-chunked when their activations cannot be bound whole. `maxBufferSize` and `maxStorageBufferBindingSize` are different limits and the second is smaller, so a tensor can allocate and then fail to bind: a pair transition needs `L*L*512*4` bytes, which passes a 128 MiB binding at **256 residues**, and an MSA transition needs `msaSequences*L*1024*4`, which passes it at about 65. `transitionChunkRows` in `src/evoformer/transition.js` picks a row window aligned to both the 16-row kernel tile and the 256-byte binding-offset requirement (their least common multiple), and `execution.view()` binds each window as its own range rather than reallocating. Ported from [upstream](https://github.com/martin-steinegger/alphafold2-webgpu); the arithmetic matches its pinned values. Short inputs take an unchanged single-dispatch path, and forcing the chunked path on an input that does not need it reproduces the full path **bit for bit**. `outerProductMeanTileCapacity` bounds the outer-product tile the same way.
- Every dispatch that indexes by row or by element folds its grid across two dimensions, because a dispatch may be at most 65535 workgroups wide. A transition layer-norm runs over `msaSequences * length` rows — 147,828 for 508 rows of a 291-residue alignment — so the unfolded form was a live overflow on the MSA path at roughly 130 residues and above. `linearGrid` and `rowGrid` do the folding, the shaders read `group.x + group.y * GRID_WIDTH` (or `id.x + id.y * GRID_WIDTH * 64u`), and `execution.dispatch` throws by name if something slips through. The fold is inert at small sizes, so it changes no short prediction.
- `shader-f16` was tried and rejected. Timestamp profiling puts 80% of a block's GPU time in two kernels at length 221: the pair-track flash attention (58%) and triangle multiplication's contraction (22%). The triangle kernel already has a differentially tested f16 path, so it could be measured directly - isolating compute by scaling `cHidden` while holding the readback fixed, f16 was **13% slower** than f32 (129.7 ms against 114.5 ms) while its error against the CPU reference was 1.89e-4 against f32's 1.04e-7. Apple GPUs run f32 and f16 ALU at the same rate; f16 only saves bandwidth, and these kernels are ALU-bound.
- The subgroup attention fast path cannot run on Chrome-on-Metal at all. All three subgroup kernels declare `enable subgroup_size_control;` and use the `@subgroup_size` attribute, and that extension is not exposed there - so `supportsAttentionSubgroups` correctly falls back to the portable kernel, and the fast-path timings quoted above describe a code path Apple hardware does not reach. The subgroup size measured on this adapter is in fact exactly 32, so a kernel that dropped the `@subgroup_size` attribute and gated on a runtime probe of `subgroup_size` instead could plausibly run here; that would need its own numerical validation, since correctness would then rest on a measured property rather than a pinned one.
- With f16 and subgroups both ruled out, the remaining levers on the flash-attention kernel are algorithmic - tiling, occupancy, and the number of passes over the pair track - rather than precision.

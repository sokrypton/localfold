# Working on LocalFold

`AGENTS.md` has the engineering invariants and `AF3.md` the AF3 port's state,
costs and dead ends. This file is the operational half: how to actually run
things here, and the traps that have cost time more than once.

## Running anything that needs a GPU

```
node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]
```

It serves the repo over HTTP, drives headless Chrome, and calls the module's
`export async function main(device, args)`. Whatever `main` returns is printed
as JSON. Anything under `tools/gpu/` is written to that shape.

🔴 **`npm run test:gpu` DOES NOT WORK ON THIS MACHINE AND NEVER HAS.** The Dawn
node binding fails to load - *"built for macOS 26.0 which is newer than running
OS"* - so every `test/*.gpu.test.js` is unrunnable locally. That is the whole
reason `tools/gpu-chrome.mjs` exists. `npm test` (the CPU suite) does run, and
must pass.

🔴 **AND ONE `.gpu.test.js` NAMES A FIXTURE THAT IS NOT IN THE REPOSITORY.**
`test/evoformer-attention.gpu.test.js` wants
`test/fixtures/evoformer/model1-query-59-block0`, which does not exist; only
`model1-query-59-stack` does. So checking an attention change against official
values means the whole-stack checker, not that file.

## The tools, by what they answer

| Question | Tool |
|---|---|
| Does the AF3 head still match AF3? | `tools/gpu/probe-head-vs-af3-steps.js --dump=/af3-rings20.json` |
| Is a fold still the same fold? | `tools/gpu/probe-sidechains.js --steps=8` |
| Is a MODIFIED residue the right shape? | `tools/gpu/probe-modified.js --code=SEP --at=3` |
| Do the distogram's contact scores track pTM/ipTM? | `tools/gpu/probe-contact-confidence.js` |
| Is the sampler converged at this step count? | `tools/gpu/probe-flow-sigma-by-size.js --panel=churn` |
| Do recycles help a complex? | `tools/gpu/probe-recycles-on-complexes.js` |
| Does MSA depth help a complex? | `tools/gpu/probe-msa-depth-on-complexes.js` (**goes to the network**) |
| Does the sampler setting matter on a real binder? | `tools/gpu/probe-designed-binder-sampler.js` (**network**) |
| Does AF2's stack match AlphaFold? | `tools/gpu/check-evoformer-stack.js` |
| Does AF2 still fold the SAME structure? | `tools/gpu/fold-af2.js` |
| Does AF2's distogram head agree with AF2's structure? | `tools/gpu/probe-af2-contacts.js` |
| Which register tile does AF2's dense projection want? | `tools/gpu/bench-evoformer-linear.js` |
| What does AF2's column attention cost alone? | `tools/gpu/bench-msa-attention.js` |
| What does a sampler step cost besides the denoiser? | `tools/gpu/probe-sampler-overhead.js` |
| Where does a denoiser call's time go? | `tools/gpu/bench-head.js --profile` |
| Where does a trunk pass's time go? | `tools/gpu/bench-trunk.js --profile --msa=1024` |
| Where does an AF2 block's time go? | `tools/gpu/profile-af2-block.js --sequences=512` |
| Just the transformer, in 3 seconds? | `tools/gpu/bench-diffusion-transformer.js` |
| Which attention kernel does this device get? | `tools/gpu/probe-kernel.js` |
| What does the page cost per frame? | `tools/gpu/bench-frame.js` |
| Which tile does a pairformer kernel want? | `tools/gpu/bench-{triangle-project,grid-project,transition,single-project,opm}.js` |
| Does an AF2 kernel still compute AF2? | `tools/gpu/check-evoformer-{transition,opm,attention}.js`, `check-triangle-residual.js` |
| What is this device's actual ceiling? | `tools/gpu/probe-alu.js` |
| Where does the HOST memory go? | `tools/gpu/probe-memory.js` |
| How long does a fold take, by shape? | `tools/gpu/bench-runtime.js` (fits `src/runtime/cost-model.js`) |
| Does the progress bar move at the fold's speed? | `tools/gpu/probe-progress-bar.js` |
| Does a failed fold keep its trunk for the retry? | `tools/gpu/probe-trunk-reuse-after-failure.js` |
| What does a fold hold on the DEVICE? | `tools/gpu/fold.js --budget=0` (prints per stage) |
| Does it still fold on a small device? | `tools/gpu/bench-trunk.js --budget=200` |
| Does the page fit a phone? | `python3 tools/mobile-layout.py` |
| Do the heatmap panel's tabs still work after a vendor bump? | `python3 tools/heatmap-panel.py` |

`tools/gpu/check-af3-*.js` are the per-module AF3 oracle checkers.

🔴 **THE PHONE LAYOUT IS MEASURED, NOT LOOKED AT, AND `--window-size` CLAMPS AT
500px.** 390 and 320 both report an innerWidth of 500; `--headless=old` clamps
identically. `tools/cdp.py` is sixty lines of WebSocket (no new dependency) and
gives `Emulation.setDeviceMetricsOverride`, which is a true viewport at any
width, plus `Page.captureScreenshot`, which `--screenshot` cannot do on a page
with a running rAF loop. `tools/mobile-layout.py` runs 320, 360 and 390 with
1200 as the control, loading a structure and an alignment first - half the rows
it measures are `display: none` on a bare page. It checks `single.html` too.

🔴 **AND "NO HORIZONTAL OVERFLOW" IS NOT THE TEST.** Under mobile emulation a
page that cannot fit does not overflow: the LAYOUT VIEWPORT GROWS, so
`scrollWidth == innerWidth` while the phone renders everything zoomed out. The
assertion is `innerWidth == the width asked for`. Nor can a size check see an
OVERLAP, or a `1fr` grid track squeezed to nothing - the entity row's sequence
box measured 0px at 320 with "PIA" set one letter per line, and every fit check
passed. Nor can it see a page that is ready to be measured: `processFiles` is
defined while `initializeApp` is still running and before `web/app.js` (a
module) has run at all, and called in that window it resolves having loaded
NOTHING. Wait for `#predict` to be enabled, which is the last thing to happen.

🔴 **AN INLINE WIDTH IS ONE NO STYLESHEET CAN OVERRIDE.** Not a media query, not
a container query, not any specificity - only `!important`, and a page whose
responsive rules all need that has no cascade left. There were four in
`index.html`, five in `single.html`, two on the MSA filter sliders, and one that
py2Dmol's own JS writes on the viewer box (turned off with
`data-autosize="css"` on `#canvasContainer`). `max-width` is a DIFFERENT
PROPERTY and beats an inline `width` with no `!important` at all, which is how
the MSA panel is contained and how `single.html` lost three of its own.

🔴 **AND py2Dmol's MSA PANEL IS STILL 948px.** `src/panels/msa.js` has
`const MIN_CANVAS_WIDTH = 948`, clamps every canvas width up to it and writes
the result as `container.style.width`. Our narrow block gives that box
`max-width: 100%; overflow-x: auto`, so it scrolls sideways inside its own card
instead of taking the whole document with it - measured, it was forcing a 972px
layout viewport on a 320px phone. Fixing it properly is an upstream change.

🔴 **AND AF2 NOW HAS AN END-TO-END GATE, WHICH THE DIFFERENTIAL ONES ARE NOT.**
A per-kernel checker says one kernel still computes its own operation. It
cannot say the assembled model still folds, and after three kernel rewrites
that was the whole of AF2's coverage here. `tools/gpu/fold-af2.js` folds a
59-mer through the driver the page uses and prints a checksum over every
coordinate, plus mean pLDDT, pTM and the backbone CA-CA geometry. Run it, stash
the change, run it again: at 128 rows and at 512 rows with a recycle, the tree
before this session's kernel work and the tree after agree to every digit.

It synthesises its alignment from the query, so the 512-row kernels run without
fetching anything, and it opens `./model/` by directory - `loadModel` resolves
the monomer family to Hugging Face, and a regression tool should not pull
227 MB. That makes it a fingerprint, not an oracle: it does not know what
AlphaFold would say.

🔴 **AF2's KERNELS NOW HAVE FOUR DIFFERENTIAL GATES, BECAUSE IT HAD NONE.**
`npm run test:gpu` cannot load Dawn here and `test/fixtures/evoformer/` is
gitignored, so every `test/*.gpu.test.js` covering AF2 is unrunnable - which
left its transition, its outer product mean, its attention projection and the
residual form of its triangle output projection with nothing checking them at
all. Each new checker writes its own CPU reference in its own file, because a
reference that shares code with the thing it checks tests nothing, and each
uses ragged shapes and ragged masks so the bounds checks and the masking are
actually exercised. They are differential, not oracle: they say the kernel
computes the operation, not that AlphaFold agrees.

🔴 **MEMORY HAS TWO HALVES AND THE BENCHES ONLY EVER SHOWED ONE.** The GPU
allocator's snapshot cannot see a `Float32Array`, and until
`src/runtime/device-memory.js` existed nothing counted the buffers created
outside the allocator - which are most of them by size. Host heap comes from
`tools/gpu/probe-memory.js` (it forces a collection first, or the reading
carries 300 MiB of garbage); device memory from `memorySnapshot(device)`,
which `fold.js` and `bench-trunk.js` print. A 31-residue fold holds **305 MiB
of heap and 1390 MiB on the device**; 1190 MiB of the latter is weights kept
resident on purpose, which `--budget` makes the code give up when it must.

🔴 **KNOW THE CEILING BEFORE CHASING IT.** `tools/gpu/probe-alu.js` runs
multiply-adds out of registers with no memory in the way. On this M2 it reports
**1287 GFLOP/s scalar, 2526 vec2, 5034 vec4**, and 396 billion workgroup reads a
second - so a vec4 multiply-add is 4x a scalar one, and every one of those is
about 640 G instructions a second. Read a kernel's number against THAT, not
against a specification sheet: the trunk's kernels sit at 900 GFLOP/s to
1.1 TFLOP/s, which is 70-85% of the scalar ceiling and a quarter of the vector
one. It is an instruction-count machine.

🔴 **THE FOUR KERNEL BENCHES EXIST BECAUSE bench-trunk.js COSTS FORTY SECONDS
AND AVERAGES 48 BLOCKS.** Each synthesises its weights, runs one shader at
several shapes interleaved in one process, and costs about a second an arm - and
each checks every arm's output against the first, because a tile the dispatch
does not match leaves rows unprocessed and reads as a speedup. Tune with those;
confirm with `bench-trunk.js`.

## Measuring, without fooling yourself

🔴 **PROFILE, DO NOT BISECT BY DELETION.** Disabling a pass and re-measuring
attributes scheduling and overlap to whatever was removed and has the bench's
noise for resolution. It has produced wrong answers here twice - once reporting
a *removed* pass as costing negative time, once naming the wrong kernel by 4x.
Two profilers exist and both work:

- `tools/gpu/profile.js` wraps `createCommandEncoder` and times every labelled
  compute pass. AF3 labels all of its passes, so this covers the AF3 side.
- AF2 has its own, older and better: `execution.beginTimestampProfile()` with
  `stack.js`'s `profileBlock` input, driven by `tools/gpu/profile-af2-block.js`.
  It is per *dispatch*, not per pass. `profile.js` cannot see into AF2, which
  batches a block's dispatches into one pass called `localfold.compute`.

Timestamps are quantised by Chrome to about 100 microseconds, so a single short
pass is unmeasurable; totals over many passes are fine.

🔴 **THIS MACHINE DRIFTS BY UP TO 3.2x BETWEEN RUNS.** Interleave A and B in one
process, or take a median of many calls - `bench-head.js` medians nine. A single
run of each is not a comparison.

## Hosting the weights somewhere other than Pages

GitHub Pages publishes at most a gigabyte, and the weights are most of it: AF2
monomer 227 MB, AF3 150 MB, before a third model exists. A page meaning to offer
five keeps its parameters elsewhere.

Everything a bundle needs is one field. In `src/reference/manifests/index.js`:

```js
af3: {
  directory: "./model-af3-int5/",                       // the fallback
  remote: "https://huggingface.co/USER/REPO/resolve/<sha>/",
  ...
}
```

and that is the whole change. Shard URLs are resolved against the bundle's base,
so the store never learns the difference; `build_site.py` and the Pages workflow
both ask `build_site.py --is-remote <family>` and stop publishing a copy.

🔴 **PIN A COMMIT SHA, NOT `main`.** A shard fetched from a moving branch can
change under a manifest that did not, which is the failure the shard-cache token
exists to prevent - and three separate hours have already gone into "<file> has
an invalid byte length", a message that names neither half.

🔴 **AND A TRAILING SLASH, OR THE LAST SEGMENT IS LOST.** `new URL(file, base)`
against ".../resolve/abc123" puts the shard beside `abc123` rather than inside
it. `bundleBaseUrl` adds one; `test/model-bundles.test.js` holds it to that.

Verified against Hugging Face from the browser: CORS passes, the 302 to
`cdn.hf.co` is followed, `?v=` cache tokens survive, ranges answer 206, and the
responses come back `type: "cors"` so the shard cache can store them. What is
NOT verified is a real upload - there were no HF credentials on this machine, so
the repository and the push are still to do.

To upload:

```
pip install huggingface_hub && hf auth login
hf upload USER/REPO model-af3-int5 . --repo-type=model
```

DeepMind's AF3 parameters carry a Prohibited Use Policy - `build_site.py`
already refuses to publish them without `LOCALFOLD_ACCEPT_MODEL_TERMS`. On
Hugging Face the equivalent is a **gated repository**, which is a better fit
than a CI variable because it asks each downloader rather than the deployer.

## Deploying

```
python3 tools/deploy.py          # push main, dispatch the workflow, verify
python3 tools/deploy.py --verify # what is live right now
```

It ends by polling `https://localfold.org/build.json` until the commit it pushed
is the one being served, so "live" is a fact rather than an impression. Pages
builds from the pushed commit, so an uncommitted file cannot reach the site -
and will not be deployed either.

## Oracle dumps

```
python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 0 --diffusion 20 \
  --float32 --sequence <SEQ> \
  --capture 'diffusion_head/__call__$|evoformer/__call__$' \
  --capture-args 'diffusion_head/__call__$' --out <path>.json
```

`--capture-args` is what records the head's *inputs*, without which its answer
cannot be reproduced. Dumps are large; keep them in the scratchpad and symlink
into the repo root only while a checker needs to fetch them.

## Two habits worth keeping

- **Verify against the oracle, not against our own reference.** The side-chain
  bug survived for months because the only checker reaching the diffusion head
  builds its weight dict by hand instead of through the loader, so it passed
  while the shipped pipeline was wrong.
- **When a kernel's shape comes from a device limit, resolve it once and pass it
  down.** Resolving it in two places gave shaders tiling by four under a
  dispatch dividing by eight - half the tokens silently unprocessed, reported by
  the bench as a 30% speedup.

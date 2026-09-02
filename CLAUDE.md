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
| Does AF2's stack match AlphaFold? | `tools/gpu/check-evoformer-stack.js` |
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
| What does a fold hold on the DEVICE? | `tools/gpu/fold.js --budget=0` (prints per stage) |
| Does it still fold on a small device? | `tools/gpu/bench-trunk.js --budget=200` |

`tools/gpu/check-af3-*.js` are the per-module AF3 oracle checkers.

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

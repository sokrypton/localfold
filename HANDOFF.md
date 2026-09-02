# Where this is, and how to pick it up

Written 2026-09-02, after a long day of performance work. `CLAUDE.md` is the
permanent operational file (how to run anything); `AF3.md` is the AF3 port's
state and its dead ends; `AGENTS.md` is the invariants. This file is the
perishable half: what is live, what is open, and the exact commands that
produced the numbers, so none of it has to be re-derived.

## Live right now

`62c81596`, deployed to https://localfold.org and verified by
`python3 tools/deploy.py`, which polls `build.json` until the pushed commit is
the one being served. Driven in the browser at that commit: 31 residues, AF3,
flow-8, pLDDT 68.7, CA-CA 3.89 A, with the device reporting 1390 MiB resident
against its 5461 MiB budget.

🔴 **A LOCAL PAGE SERVED FROM THE SAME PORT WILL RUN YESTERDAY'S MODULES.**
Chrome caches ES modules by URL and `python3 -m http.server` gives it no reason
not to; after an edit the page failed with "does not provide an export named
releaseResidentWeights" while the file on disk plainly had it, and reloads did
not help because the import specifier has no query on it. Serve on a NEW PORT
(`python3 -m http.server 4188`) - a different origin is a different cache.

## The memory work, 2026-09-02

Prompted by reading upstream (`martin-steinegger/alphafold2-webgpu`), which
spent September on exactly this. We had never measured host memory at all.

🔴 **THE DISK WAS FULL, AND IT WAS OUR OWN TOOLING.** `tools/gpu-chrome.mjs`
gave every run a fresh Chrome `--user-data-dir` and never deleted it: 1265 of
them, 428 GB, 20 MiB free on a 926 GiB disk. A full disk does not announce
itself - git fails to write its index, shell redirections produce empty files,
node exits non-zero with no message. `249c068` deletes the profile after the
run. If a session ever looks like "the shell is broken", run `df -h /` first.

| host heap, shipped int5 export | before | now |
|---|---|---|
| after loading the trunk's weights | 732 MiB | **116 MiB** |
| after the diffusion head as well | 1039 MiB | **272 MiB** |
| after confidence and the atom reference | 1050 MiB | **284 MiB** |
| after one trunk pass | 1068 MiB | **305 MiB** |

Three causes, all the same shape - a decoded float32 copy kept alive beside the
same numbers already resident on the device:

- **Stacked tensors decoded whole.** The trunk's 48 pairformer blocks and the
  head's 24 transformer blocks are each one tensor with the block as the
  leading axis; a block took its slice by decoding all of it, and
  `HttpTensorStore` kept every array it ever decoded. `readTensorRange` decodes
  part of a tensor with the block scales indexed by absolute position, the
  store gained `open(name)` (the shard, not the tensor) and `tensorRangeSync`,
  and the block loaders build descriptors of thunks that `bind()` turns into
  properties decoding on first read. Memoised, so the CPU reference paths are
  unaffected; released explicitly by the GPU block encoders, once every buffer
  they need is on the device.
- **Packed arrays cached after they were needed.** Two WeakMaps held the
  concatenated upload buffers - 103 MiB for the pairformer, ~630 MB for the
  diffusion transformer - although the resident device buffers they fill are
  created on the first miss and never again.
- **A no-op quantiser copying the model.** Nine tools passed
  `{ fetchImplementation: fetch }` into `openAf3Store`'s quantisation slot, so
  every tensor went through `Float32Array.from` and then through a quantiser
  that walked no groups. `openAf3Store` now rejects it.

Nothing moved on time or accuracy: trunk vs AF3 6.18e-7, transformer 2.20e-5,
flow-8 bond scale median 1.015, a 59-token recycle ~390 ms, a denoiser call
114 ms.

`tools/gpu/probe-memory.js` is what found all of it and is the only tool here
that reports host memory - the benches print the GPU allocator's snapshot,
which cannot see a Float32Array. It forces a collection before each reading
(`gpu-chrome.mjs` now passes `--expose-gc` and `--enable-precise-memory-info`);
without that the numbers carry uncollected garbage and move by 300 MiB.

## The device memory work, 2026-09-02

The second half, and the first time anything here counted GPU memory.
`GpuBufferAllocator` counts its own allocations, but the buffers that dominate
bypass it - resident weights, the diffusion transformer's blocks and the atom
encoder's statics all call `createBuffer` directly because they must outlive
the run. `src/runtime/device-memory.js` keeps one account per device and all
four sites report to it.

**A 31-residue flow-8 fold holds 1390 MiB on the device**, peaking at 1395:
567 MiB the 48 pairformer blocks, ~630 the 24 transformer blocks, the rest
scratch. Confirmed in the page as well as headless.

**The budget, taken from upstream.** Metal accepts allocations past the point
where macOS pages and a phone's driver takes them and is then killed; WebGPU
reports nothing either way, so the symptom is a frozen machine. An allocation
over the ceiling is now refused BEFORE `createBuffer` with a
`GpuMemoryBudgetError` naming the tensor, the resident total and the ceiling.
The page sets one at a third of `navigator.deviceMemory` - 5461 MiB here, since
this Mac reports 16 GiB; 1365 MiB on a 4 GiB phone, which is under what a fold
needs, and that is the machine this exists for. Benches set none.

**Residency is a measured trade, decided by the budget rather than guessed.**
Keeping the pairformer resident costs 567 MiB and buys 30 ms a recycle (398 ms
against 428 at 59 tokens). Choosing in advance needs an estimate of scratch
against weights, which is exactly the number upstream let drift 3-5x; instead
the fast path is tried and the refusal answers. On refusal the stack is
ABANDONED and re-encoded uploading per block - freeing mid-stack gives
"Buffer w.tri.out used in submit while destroyed", and not freeing leaves the
fallback ten megabytes to work in. The refusal is remembered on the DEVICE:
the trunk builds a fresh stack per pass, so remembering it on the object cost
a whole abandoned stack every pass (654 ms a pass at a 400 MiB ceiling, 421
after). The allocator also drops its pool before giving up, since a pooled
buffer is memory nothing is using.

The diffusion head submits per super-block when it is not resident, which
bounds its uploading path at four blocks; batched into one command buffer it
held all twenty-four, 756 MiB, more than the residency it was avoiding.

A 31-residue fold, 8 flow steps, by ceiling - pLDDT 64.2 at every one:

| ceiling | time | device peak |
|---|---|---|
| none | 2.0 s | 1390 MiB |
| 800 MiB | 3.2 s | 774 MiB |
| 400 MiB | 3.5 s | 398 MiB |
| 150 MiB | 3.2 s | 150 MiB |
| 110 MiB | refused | four blocks and scratch do not fit |

At 32 steps the constrained path is 8.1 s against 3.4, which is the price of
running at all on a machine that could not have.

Exercise it with `fold.js --budget=<MiB>` or `bench-trunk.js --budget=<MiB>`.
Nothing else reaches that code on a Mac.

### Still on the table from upstream

Read but not taken, roughly in order of value:

- **An allocator that can reuse a buffer it did not size exactly.** Ours keys
  the pool on `bytes:usage` and matches both exactly, which is the failure
  upstream describes: each operation grows its own set of chunks. Theirs backs
  any request over a mebibyte with whole mebibytes, reuses any retired buffer
  whose usage covers the request and is at most twice its size, and destroys
  buffers idle for four submissions.
- **Blocking the pair-shaped scratch** so no operation holds a whole one
  (their 8e679b5, worth 100-130 MiB at 384-512 residues, at 3-4% of the run).
- **Optional half-precision activation storage** for the MSA and the triangle
  projection - inexact, off by default, 15-25% of the working set.
- **A memory estimate pinned to measured working sets by a test**, so it cannot
  drift; theirs had gone 3-5x stale unnoticed. We chose the reactive answer
  instead, which needs no estimate - but it is first-come-first-served, and a
  stage that fits takes the budget from every stage after it.
- **Running the pipeline in a worker**, so a long fold does not freeze the page.

We already had their submit-ahead window and their deferred validation, and
their on-demand weight decoding is done and went further.

| | before today | now |
|---|---|---|
| AF3 trunk pass, 59 tokens / 32 MSA rows | 540 ms | **403 ms** |
| ...of which the pairformer | 435 ms | **311 ms** |
| AF3 trunk pass, 150 tokens | 3.38 s | **2.25 s** |
| AF3 trunk pass, 1024 MSA rows | 670 ms | **544 ms** |
| ...of which the MSA stack | 334 ms | **196 ms** |
| AF3 denoiser call | 134 ms | **111 ms** |
| AF3 flow-8 fold, 68-mer | ~7 s | **3.0 s** |
| AF3 diffusion-200 fold | 26.3 s | **25.9 s** |
| AF2 evoformer block, 512 MSA rows | 188.6 ms | **138.6 ms** |
| AF3 cold weight download, 266 MB | ~20 s | **7.8 s** |
| 6MRR, flow 8, seed 1 | - | 0.64 A, TM 0.960 |

Accuracy moved the right way or not at all: the denoiser's worst error against
AF3's own is **6.03e-6** (was 9.22e-6), side-chain bond ratio **1.015** against
AF3's own 1.017, 383 CPU tests, 18 AF3 checkers and 6 AF2 gates all pass.

## Driving the site, which is the check the benches cannot make

Both models were run on localfold.org in Chrome at `ffc670d7`: **AF2 monomer**
through `single.html` in 6.9 s at pLDDT 85.0, and **AF3** through `index.html`
at 3 recycles and flow-8 in 6 s cold, 1 s on a repeat with the trunk reused,
pLDDT 86.9 - with the PAE panel, trajectory scrubber and sequence track working.

The recipe, through the browser tools: navigate to the page, then in the tab

```js
for (const n of await caches.keys()) await caches.delete(n);   // cold-ish
performance.clearResourceTimings();
const ta = document.querySelector('textarea');
const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
set.call(ta, '<SEQUENCE>');
ta.dispatchEvent(new Event('input', { bubbles: true }));
[...document.querySelectorAll('button')].find(b => /^\s*Fold\s*$/i.test(b.textContent)).click();
```

then read the status line out of the DOM and the shard timings out of
`performance.getEntriesByType('resource')`. 🔴 Clearing the Cache API does NOT
give a cold load - Chrome's own HTTP cache still serves the shards. To measure
the network, refetch the shard URLs with a `?bust=` query instead.

## What 2026-09-02 did, in one place

Twenty-one commits, every one deployed and verified. The pattern that paid,
over and over, was not arithmetic - it was **who reads what**:

1. **N lanes issuing N identical global loads.** A workgroup shares a row, a
   head, a pair; the loop runs over an axis they all share; every lane fetches
   the same value. Staging it in workgroup memory was worth 1.9x on AF3's grid
   attention, 1.8x on its outer product, 1.6x on AF2's flash attention, and 3x
   on AF2's output projection.
2. **A quantity recomputed per output that depends only on the input.** AF2's
   outer product recomputed its denominator once per channel - 128 times per
   pair. AF3's MSA projection renormalised a row once per output - 64 times.
   The pairformer's pair-logits renormalised inside the head loop - 16 times.
3. **Four matrices contracted over one activation, read as four scalars.** They
   want to be one vec4: the triangle projection, the grid projection, AF2's
   q/k/v/gate.
4. **A tile of rows so one weight read serves all of them** - and the tile is
   always a function of the row count, because past a point the workgroups are
   worth more than the traffic.
5. **A delta written, then read, then added.** Four of the pair track's five
   updates now write themselves in.

And the thing that made all of it decidable: `tools/gpu/probe-alu.js`, which
asks the device what it can do rather than reading a specification sheet.

## The tools this session added, and what each answers

Everything runs through `node tools/gpu-chrome.mjs tools/gpu/<module>.js`.

| new tool | answers |
|---|---|
| `probe-alu.js` | what this device actually does: 1236 GFLOP/s scalar, 4839 vec4, 394 G workgroup reads/s, 111 G cached global reads/s, and 7 G streamed |
| `probe-latency.js` | why kernels reach 270 G instr/s and the probe says 640 - sweeps chains, lanes, workgroup memory, read:fma ratio, barriers |
| `bench-triangle-project.js` | the triangle projection and contraction tiles, `rows x cols[@contraction]`, with `:barrier`/`:x`/`:w` diagnostic arms |
| `bench-grid-project.js` | the grid attention's projection row tile, and times `attend` beside it |
| `bench-transition.js` | the transition's `tile:chunk[:drop[:width[:lanes]]]` |
| `bench-single-project.js` | the single track's width split |
| `bench-opm.js` | the outer product's `i x j[@cells]` block |
| `check-triangle-residual.js` | AF2-only: the residual form of the triangle output projection |
| `check-evoformer-transition.js` | AF2's transition against a CPU reference |
| `check-evoformer-opm.js` | AF2's outer product, BOTH paths |
| `check-evoformer-attention.js` | AF2's attention against a CPU reference |

🔴 **THE FOUR `check-evoformer-*` / `check-triangle-residual` GATES EXIST
BECAUSE AF2 HAD NONE.** Dawn will not load here and `test/fixtures/evoformer/`
is gitignored, so every AF2 `.gpu.test.js` is unrunnable. Each new checker
writes its own CPU reference in its own file - a reference sharing code with the
thing it checks tests nothing - and each uses ragged shapes and ragged masks so
the bounds checks and the masking are exercised. They are differential, not
oracle: they say the kernel computes the operation, not that AlphaFold agrees.

## The commands that produced those numbers

Everything GPU runs through the Chrome harness; see CLAUDE.md for why
`npm run test:gpu` cannot.

```
node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]
```

Measuring:

```
tools/gpu/bench-head.js --profile                  # a denoiser call, by stage and by pass
tools/gpu/bench-trunk.js --passes=2 --msa=1024 --profile
tools/gpu/bench-diffusion-transformer.js --tile=4 --splits=2   # ~3 s, synthetic weights
tools/gpu/bench-triangle-project.js --arms=16x16,32x16@32x32   # projection@contraction tiles
tools/gpu/bench-grid-project.js --arms=4,8,16                  # grid projection row tile
tools/gpu/bench-transition.js --arms=4,4:256,8:128             # tile:chunk[:diagnostic[:width]]
tools/gpu/bench-single-project.js --tokens=59 --arms=1,2,3     # the single track's width split
tools/gpu/probe-alu.js                             # the device's own ceiling, for the above
tools/gpu/bench-weights.js --family=monomer|multimer           # or --model=<dir>/manifest.json
tools/gpu/bench-blocks.js --repeats=2 --profile    # AF2 against AF3, interleaved
tools/gpu/profile-af2-block.js --sequences=512     # AF2 per DISPATCH, not per pass
tools/gpu/bench-frame.js                           # what a trajectory frame costs the sampler
tools/gpu/probe-kernel.js                          # which attention kernel this device gets
```

Checking:

```
npm test                                           # 377 tests, must pass
tools/gpu/probe-head-vs-af3-steps.js --dump=/af3-rings20.json   # THE oracle gate
tools/gpu/probe-sidechains.js --steps=8            # is it still the same fold
tools/gpu/check-triangle-residual.js               # AF2's ONLY gate on the residual projection
tools/gpu/check-af3-*.js                           # per-module AF3 oracle checkers
tools/gpu/check-attention-variants.js              # every flash kernel, same input
tools/gpu/check-evoformer-stack.js                 # AF2 vs official - NEEDS captures, see below
```

The oracle dumps live in the session scratchpad and are symlinked in only while a
checker needs them:

```
ln -sf <scratchpad>/af3-rings20.json af3-rings20.json   # then --dump=/af3-rings20.json
```

Regenerating one, when a sequence with rings is needed:

```
python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 0 --diffusion 20 \
  --float32 --sequence PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK \
  --capture 'diffusion_head/__call__$|evoformer/__call__$' \
  --capture-args 'diffusion_head/__call__$' --out <scratchpad>/af3-rings20.json
```

## How to A/B without lying to yourself

Two of this session's A/Bs were invalid, both the same way: **the tool moved with
the code.**

- `git stash` reverts the BENCH as well as the thing under test. An arm that lost
  a `--family` flag silently measured a different model and reported it as a 1.7x
  win that did not exist.
- An earlier pair ran with the disk at 100%, where git could not write its index,
  so both arms were the same code.

Revert exactly one file and alternate in one session:

```
cp src/<file>.js /tmp/now.js
<measure>                                   # arm A
git checkout -q HEAD~1 -- src/<file>.js
<measure>                                   # arm B
cp /tmp/now.js src/<file>.js
```

This machine drifts up to 3.2x between runs. `bench-head.js` medians nine calls
and prints a range; trust the range, not a pair.

## Open threads, in the order I would take them

1. **AF2's column attention, 32 ms of its 160 ms block at 512 rows.** It is the
   largest kernel left there and the only one that is quadratic in DEPTH -
   column attention runs over the sequence axis - so it grows 16x for 4x the
   rows while everything else grows 4x. It is already a good kernel: key and
   value tiles staged in workgroup memory, subgroup reductions for the softmax,
   an online maximum. What is left in it is the 32-iteration `subgroupShuffle`
   loop that weights the values, which is about three instructions per useful
   multiply-add. Not attempted.
2. **The MSA stack at real depth is now the untouched half.** At 59 tokens and
   1024 rows the stack is 325 ms against a 315 ms pairformer, and the outer
   product's sequence sweep is most of it - the pair tiling below was worth
   nothing there (312 against 325, adjacent runs) because the sweep scales with
   SEQUENCES and the win was all in the output projection, which scales with
   pairs. Two structural attempts on that sweep have already lost; see AF3.md.
3. **Count instructions, not flops.** `tools/gpu/probe-alu.js` says this device
   does 1287 GFLOP/s scalar, 5034 vec4, and 396 billion workgroup reads a
   second - all of them about **640 billion instructions a second**. The trunk's
   kernels sit at 200-310 billion, so what is left is in the instruction count,
   and the biggest remaining term is global weight loads. Concretely, for
   `pair-transition`: its first matmul issues two scalar weight loads per
   channel per slot. Giving each lane four CONSECUTIVE slots would make those
   two vec4 loads and cut the kernel's instruction count by about 28%.
   Estimated, not measured - the accumulators then need a second vector axis
   (four slots by four rows), which is the intricate part.
4. **`grid.attend` again, at 150 tokens.** Staging its keys was worth 1.9x, but
   it is still the only cubic kernel and it is second-largest at both sizes. Two
   attacks are measured and LOST (more than one query an invocation; a vec4
   score accumulator) - read AF3.md before starting. What has NOT been tried:
   splitting the keys across invocations with a combining pass, which trades a
   second dispatch for occupancy it does not have at small N.
5. **`single-transition`, 27 ms for 59 rows.** Splitting it into two dispatches
   was tried and measured 65 ms against 27; see AF3.md. Untouched and under-occupied: 59
   workgroups, and its row tile cannot rise because there are no rows, so it is
   the one kernel still generating scalar code. The only real fix is
   materialising the widened intermediate (362 KB at this size) and splitting it
   into two dispatches - which the pair track must NOT do, at 1.47 GB for 600
   tokens. Two code paths for one kernel; judged not worth it yet.
6. **Weight loading — done, and here is what it was.** The thread said this was
   blocked on a real cold-load number. Measured on localfold.org rather than
   localhost, where fetch is 22 ms and hides everything:

   | | before | after |
   |---|---|---|
   | mean download concurrency | 0.68 | **7.32** |
   | peak concurrency | 1 | **8** (the cap) |
   | effective throughput | 6.8 MB/s | **34.3 MB/s** |
   | span for 266 MB | ~20 s | **7.8 s** |

   A shard was only requested when a loader reached a tensor inside it, and the
   loaders await tensor by tensor - so a cold load was: fetch a shard, sit on
   the network while it is dequantised, fetch the next. The eight-way limit was
   never once reached because nothing ever asked for eight.
   `HttpTensorStore.prefetch()` schedules them all up front, and the page calls
   it once a load has legitimately begun.

   🔴 **AND STARTING THE TRANSFER EARLIER WAS BUILT AND THEN TAKEN OUT.** Warming
   on the first typed sequence worked - 15 of 26 shards down four seconds after
   typing, with the button untouched - but it has to guess which model the
   sequence is for, and guessing wrong spends 277 MB on a download nobody wanted.
   With a model selector on the page and more models intended, the guess is
   wrong often enough that it is not worth having. If it ever returns it belongs
   on the model SELECTION, not on the sequence.

   **Measured again on 2026-09-02, against Hugging Face, and there is no
   parallelism win left.** Throughput against the number of connections, 4 MB
   apiece on distinct shards:

   | connections | 1 | 2 | 4 | 8 | 16 |
   |---|---|---|---|---|---|
   | MB/s | 3.7 | 6.9 | 8.5 | **9.3** | 9.4 |

   Eight is already 99% of sixteen, so the eight-way fetch is right and raising
   it buys nothing.

   🔴 **AND THE 3.7 AT ONE CONNECTION IS NOT A PER-CONNECTION CAP**, which is
   what it looks like and what sent this down a wrong path. A 4 MB transfer is
   mostly redirect, TLS and TCP slow-start; the same connection fetching a
   41 MiB shard on its own reaches **9.0 MB/s, the whole link**. Splitting that
   shard into four ranges - which Hugging Face serves, 206 and all - therefore
   saves 7%, not 4x: 4.39 s against 4.68 s, alternated. The shards are uneven
   (median 7.9 MiB, max 40.5) and it does not matter.

   So a cold load is bytes divided by the link, and nothing about how they are
   requested changes that: 265 MB at 9.4 MB/s is 28 s against 32.4 measured.

   🔴 **AND ONE BIG FILE IS NOT THE ANSWER, THOUGH IT LOOKS LIKE ONE.** Asked
   directly: 32 MB as eight ranges of a single file against the same bytes as
   twenty-six shard requests, alternated - 9.6 and 10.0 MB/s against 9.9 and
   10.2. Request count does not matter; both saturate the link. What one file
   costs is everything else:

   - **A single request cannot use a fast link.** HANDOFF's own earlier figures
     say so: serial 14 MB/s against four-way 27.5. One file fetched as one
     request would have been half speed there.
   - **Fetching it as parallel ranges gives up the cache.** `Cache.put()`
     rejects a 206 by specification, and every ranged response is a 206 - so a
     repeat visitor would re-download the whole model. The shard cache works
     because a whole shard is a cacheable 200.
   - So one file is either not parallel or not cacheable, and shards are what
     you get back if you fix both.

   **What is left is the bytes, and they are close to irreducible.** 277 MB of
   int5, served gzipped already, and gzip takes 5% off packed integers. The
   link caps at about 27 MB/s (serial 14, four-way 27.5, eight-way 26.3 - so
   four connections already saturate it). Going below 10 s cold means fewer
   bytes: int4 is the only lever anybody has looked at, and its accuracy cost is
   unmeasured.

7. **Other Stop-then-retry paths.** A user hit "mergeSearchedChains is not
   defined" by stopping a fold and folding again. Fixed, and the search-reuse
   decision was extracted to `planSearchReuse` with tests - but `af2Cache` and the
   recycle-continuation logic that reuses `trunkCache` are the same shape of
   stateful code with the same absence of tests.
8. **AF2 has no official-value gate on this machine.** Four differential gates
   now exist - `check-evoformer-{transition,opm,attention}.js` and
   `check-triangle-residual.js` - each with its own CPU reference. They say the
   kernels compute the operations; they do not say AlphaFold agrees. Partly
   closed:
   `tools/gpu/check-triangle-residual.js` now covers the one path only AF2
   reaches - the residual form of the triangle output projection - and fails at
   1.0 relative if the two forms are swapped. The rest still rests on
   differential evidence. Three independent reasons,
   The three reasons, none a bug: Dawn will not load; `test/evoformer-attention.gpu.test.js` names a
   fixture directory that is not in the repository; and `test/fixtures/evoformer/`
   is gitignored, so this checkout has 26 of the stack manifest's 530 tensors.
   `tools/gpu/check-evoformer-stack.js` is ported and will run wherever those
   captures live. Until then AF2 kernel changes rest on differential evidence.

## Where the time is now, in the case that matters

150 tokens with a 512-row alignment - a real fold rather than a corner. A trunk
pass is **2.60 s**: pairformer 1960, MSA stack 531, template 51, distogram 31.
Inside the pairformer, in milliseconds over the measured window:

    pair-transition 449   grid.attend 316   tri.project 291   grid.project 243
    opm.contract 219      tri.project-out 212   tri.contract 115

🔴 **AND EVERY ONE OF THOSE IS NOW AT ABOUT 270 BILLION INSTRUCTIONS A SECOND**,
against the 640 billion tools/gpu/probe-alu.js measures with no memory in the
way and no dependent chains. That gap is latency, not instruction count: the
counts are within about 1.3x of what the arithmetic needs, and eight separate
attempts to cut them further (listed in AF3.md) measured worse or level. The
next real gain is not another tile.

## Measured on AF2 and not kept

- **Widening the transition's column tile from 64 to 128**, so a thread owns
  sixteen columns as four vec4s rather than eight as two. Its two weight reads a
  step are shared by every column, so this should have cut the read-to-
  multiply-add ratio by a third - and the column axis is the safe one to widen,
  since the ROW tile is shared with the structure module, whose linears run over
  59 residues. It measured **22.2 ms against 18.7** on AF2's transition at 512
  MSA rows, and took the structure module's encoded pass from 19.3 ms to 25.8.
- **Unrolling that kernel's staging and writeback loops** at 64 columns, which
  the wider tile needed and which looked free: 19.1 and 19.5 ms against 18.7 and
  18.3, and the structure module unchanged within its noise. The loop with a
  runtime component index compiles to something no worse than the unrolled form.

## Traps that cost time

- **`profile-af2-block.js`'s BLOCK total drifts by 10% between processes**, on
  the same build - 11.0 to 12.7 ms measured four times. Its per-DISPATCH numbers
  are stable to about 1%, so a claim about an AF2 kernel rests on those and a
  claim about the block does not rest on anything.
- **`--profile` costs about a fifth of the trunk.** It writes a timestamp pair
  per compute pass; the same build measures 439 ms without it and 528 with. Rank
  kernels with it; quote totals from runs that do not use it.

- **`fold.js` defaults to `--mode=diffusion`**, where `--steps` is the schedule's
  discretisation and not a budget. Eight steps of it prints an N-CA of 27 A
  against an ideal of 1.46, which reads as corrupted weights. It now warns.
- **A kernel shape that comes from a device limit must be resolved once and
  passed down.** Resolving it in both `run()` and the shader factory gave shaders
  tiling by four under a dispatch dividing by eight - half the tokens never
  projected, reported by the bench as a 30% speedup. Every tile added since is
  RETURNED by the shader factory (`projectTile`, `contractTile`, `normalizeRows`,
  `tiles`, `projectSplits`) so the dispatch cannot read a different one.
- **A string replacement that stops matching throws nothing.** AF2's blocks made
  the residual projection by patching the finished WGSL; rewriting that kernel's
  writeback would have silently left the plain form in place, overwriting the
  pair representation where it means to add. It is generated now, and
  `test/triangle-project-tile.test.js` asserts the two forms differ in exactly
  one line.
- **Pricing a read by substituting a constant overstates it**, because the
  constant lets the compiler hoist the arithmetic that depended on the read.
  A diagnostic arm said 29%; the change it pointed at was worth 5%. Rank with
  it, do not target it.

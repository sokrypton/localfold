# Where this is, and how to pick it up

Written 2026-09-02, after two sessions that were mostly performance work. `CLAUDE.md`
is the permanent operational file (how to run anything); `AF3.md` is the AF3
port's state and its dead ends; `AGENTS.md` is the invariants. This file is the
perishable half: what is live, what is open, and the exact commands that produced
the numbers, so none of it has to be re-derived.

## Live right now

`c057bc63`, deployed to https://localfold.org and verified by
`python3 tools/deploy.py`, which polls `build.json` until the pushed commit is
the one being served. Everything below is live.

| | before today | now |
|---|---|---|
| AF3 denoiser call, 59-mer | 760 ms | **134 ms** |
| AF3 diffusion-200 fold, end to end | ~152 s | **26.3 s** |
| AF3 trunk pass, 32 MSA rows | 756 ms | **439 ms** |
| ...of which the pairformer | 632 ms | **337 ms** |
| AF3 trunk pass, 150 tokens | 3.38 s | **2.54 s** |
| AF3 trunk pass, 1024 MSA rows | 1093 ms | **804 ms** |
| AF3 checkpoint load | 5470 ms | **1364 ms** |
| AF2 monomer / multimer load | 1012 / 874 ms | **417 / 400 ms** |
| AF2 evoformer block, 512 MSA rows | 302.8 ms | **192.0 ms** |
| AF2 evoformer block, 5 MSA rows | 12.6 ms | **10.4 ms** (a 48-block stack, 498) |
| AF3 side-chain bond ratio | 0.927 | **1.015** (AF3 itself: 1.017) |

Accuracy moved the right way: worst relRMS against AF3's own denoiser over
twenty noise levels is **9.22e-6**, where it was 1.19e-5 this morning.

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

1. **Count instructions, not flops.** `tools/gpu/probe-alu.js` says this device
   does 1287 GFLOP/s scalar, 5034 vec4, and 396 billion workgroup reads a
   second - all of them about **640 billion instructions a second**. The trunk's
   kernels sit at 200-310 billion, so what is left is in the instruction count,
   and the biggest remaining term is global weight loads. Concretely, for
   `pair-transition`: its first matmul issues two scalar weight loads per
   channel per slot. Giving each lane four CONSECUTIVE slots would make those
   two vec4 loads and cut the kernel's instruction count by about 28%.
   Estimated, not measured - the accumulators then need a second vector axis
   (four slots by four rows), which is the intricate part.
2. **`grid.attend` again, at 150 tokens.** Staging its keys was worth 1.9x, but
   it is still the only cubic kernel and it is second-largest at both sizes. Two
   attacks are measured and LOST (more than one query an invocation; a vec4
   score accumulator) - read AF3.md before starting. What has NOT been tried:
   splitting the keys across invocations with a combining pass, which trades a
   second dispatch for occupancy it does not have at small N.
3. **`single-transition`, 27 ms for 59 rows.** Untouched and under-occupied: 59
   workgroups, and its row tile cannot rise because there are no rows, so it is
   the one kernel still generating scalar code. The only real fix is
   materialising the widened intermediate (362 KB at this size) and splitting it
   into two dispatches - which the pair track must NOT do, at 1.47 GB for 600
   tokens. Two code paths for one kernel; judged not worth it yet.
4. **The network side of weight loading** — 265 MB over 26 shards, unmeasured
   from a real client; everything here was localhost, where fetch is 31 ms. The
   fork at `martin-steinegger/alphafold2-webgpu` packs to 8 shards and has
   download-throttling and progress commits worth mining. Blocked on a real
   cold-load number.
5. **Other Stop-then-retry paths.** A user hit "mergeSearchedChains is not
   defined" by stopping a fold and folding again. Fixed, and the search-reuse
   decision was extracted to `planSearchReuse` with tests - but `af2Cache` and the
   recycle-continuation logic that reuses `trunkCache` are the same shape of
   stateful code with the same absence of tests.
6. **AF2 has no official-value gate on this machine.** Partly closed:
   `tools/gpu/check-triangle-residual.js` now covers the one path only AF2
   reaches - the residual form of the triangle output projection - and fails at
   1.0 relative if the two forms are swapped. The rest still rests on
   differential evidence. Three independent reasons,
   The three reasons, none a bug: Dawn will not load; `test/evoformer-attention.gpu.test.js` names a
   fixture directory that is not in the repository; and `test/fixtures/evoformer/`
   is gitignored, so this checkout has 26 of the stack manifest's 530 tensors.
   `tools/gpu/check-evoformer-stack.js` is ported and will run wherever those
   captures live. Until then AF2 kernel changes rest on differential evidence.

## Traps that cost time

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

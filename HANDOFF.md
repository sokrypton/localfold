# Where this is, and how to pick it up

Written 2026-09-02, after two sessions that were mostly performance work. `CLAUDE.md`
is the permanent operational file (how to run anything); `AF3.md` is the AF3
port's state and its dead ends; `AGENTS.md` is the invariants. This file is the
perishable half: what is live, what is open, and the exact commands that produced
the numbers, so none of it has to be re-derived.

## Live right now

`60cb9e02`, deployed to https://localfold.org and verified by
`python3 tools/deploy.py`, which polls `build.json` until the pushed commit is
the one being served. Everything below is live.

| | before today | now |
|---|---|---|

| AF3 diffusion-200 fold, end to end | ~152 s | **26.3 s** |
| AF3 trunk pass, 32 MSA rows | 756 ms | **408 ms** |
| ...of which the pairformer | 632 ms | **311 ms** |
| AF3 trunk pass, 150 tokens | 3.38 s | **2.25 s** |
| AF3 trunk pass, 1024 MSA rows | 1093 ms | **544 ms** |
| ...of which the MSA stack | 334 ms | **196 ms** |
| AF3 denoiser call | 760 ms | **111 ms** |
| 6MRR, flow 8, seed 1 | - | 0.64 A, TM 0.960 |

🔴 **AND BOTH MODELS WERE DRIVEN ON THE DEPLOYED SITE, not just on the bench.**
2026-09-02, localfold.org, the 68-residue 6MRR sequence, single sequence, in
Chrome on this M2:

- **AF2 monomer** through `single.html`: done in **6.9 s**, pLDDT 85.0, pTM
  0.593, structure rendered with side chains.
- **AF3** through `index.html` at 3 recycles and flow-8: **6 s** cold and **1 s**
  on a repeat with the trunk reused, pLDDT 86.9, pTM 0.764, CA-CA 3.83 A - with
  the PAE panel, the trajectory scrubber and the sequence track all working.

That is the check the benches cannot make: every number above this line is a
kernel or a stage, and none of them says the page still folds.
| AF3 trunk pass, 1024 MSA rows | 1093 ms | **804 ms** |
| AF3 checkpoint load | 5470 ms | **1364 ms** |
| AF2 monomer / multimer load | 1012 / 874 ms | **417 / 400 ms** |
| AF2 evoformer block, 512 MSA rows | 302.8 ms | **192.0 ms** |
| AF2 evoformer block, 512 MSA rows | 188.6 ms | **160.4 ms** |
| ...its outer product contraction | 28.8 ms | **16.2 ms** |
| ...its transition, both halves | 45.1 ms | **37.5 ms** |
| ...its attention projections, both | 38.5 ms | **33.1 ms** |
| ...its triangle projection | 0.581 ms | **0.422 ms** |
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
6. **The network side of weight loading** — 265 MB over 26 shards, unmeasured
   from a real client; everything here was localhost, where fetch is 31 ms. The
   fork at `martin-steinegger/alphafold2-webgpu` packs to 8 shards and has
   download-throttling and progress commits worth mining. Blocked on a real
   cold-load number.
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

# Where this is, and how to pick it up

Written 2026-09-02, after a session that was mostly performance work. `CLAUDE.md`
is the permanent operational file (how to run anything); `AF3.md` is the AF3
port's state and its dead ends; `AGENTS.md` is the invariants. This file is the
perishable half: what is live, what is open, and the exact commands that produced
the numbers, so none of it has to be re-derived.

## Live right now

`57088515`, deployed to https://localfold.org and verified by
`python3 tools/deploy.py`, which polls `build.json` until the pushed commit is
the one being served.

| | before today | now |
|---|---|---|
| AF3 denoiser call, 59-mer | 760 ms | **134 ms** |
| AF3 diffusion-200 fold, end to end | ~152 s | **26.3 s** |
| AF3 trunk pass, 32 MSA rows | 756 ms | **570 ms** |
| AF3 trunk pass, 1024 MSA rows | 1093 ms | **804 ms** |
| AF3 checkpoint load | 5470 ms | **1364 ms** |
| AF2 monomer / multimer load | 1012 / 874 ms | **417 / 400 ms** |
| AF2 evoformer block, 512 MSA rows | 302.8 ms | **192.0 ms** |
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

1. **Deep kernel work** — the agreed next task. Both hot paths are flat: the
   head's largest kernel is `ffw-out` at 21 ms of 134, the trunk's top six are
   84/63/43/41/40/39 ms with no outlier. What is left is arithmetic intensity:
   these kernels read a weight and use it 2-4 times. The fix is cooperative
   tiling - staging BOTH operands in shared memory - rather than the one-sided
   tiling already there. **Read AF3.md's list of what has already failed first**:
   three structural attempts on the transformer all lost, and six changes that
   looked obvious measured as nothing.
2. **The network side of weight loading** — 265 MB over 26 shards, unmeasured
   from a real client; everything here was localhost, where fetch is 31 ms. The
   fork at `martin-steinegger/alphafold2-webgpu` packs to 8 shards and has
   download-throttling and progress commits worth mining. Blocked on a real
   cold-load number.
3. **Other Stop-then-retry paths.** A user hit "mergeSearchedChains is not
   defined" by stopping a fold and folding again. Fixed, and the search-reuse
   decision was extracted to `planSearchReuse` with tests - but `af2Cache` and the
   recycle-continuation logic that reuses `trunkCache` are the same shape of
   stateful code with the same absence of tests.
4. **AF2 has no official-value gate on this machine.** Three independent reasons,
   none a bug: Dawn will not load; `test/evoformer-attention.gpu.test.js` names a
   fixture directory that is not in the repository; and `test/fixtures/evoformer/`
   is gitignored, so this checkout has 26 of the stack manifest's 530 tensors.
   `tools/gpu/check-evoformer-stack.js` is ported and will run wherever those
   captures live. Until then AF2 kernel changes rest on differential evidence.

## Two traps that cost time today

- **`fold.js` defaults to `--mode=diffusion`**, where `--steps` is the schedule's
  discretisation and not a budget. Eight steps of it prints an N-CA of 27 A
  against an ideal of 1.46, which reads as corrupted weights. It now warns.
- **A kernel shape that comes from a device limit must be resolved once and
  passed down.** Resolving it in both `run()` and the shader factory gave shaders
  tiling by four under a dispatch dividing by eight - half the tokens never
  projected, reported by the bench as a 30% speedup.

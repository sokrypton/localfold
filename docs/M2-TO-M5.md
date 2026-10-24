# M2 to M5: what the M2 found in the M5's pass, what it changed, and what the M5 should check (`opus-55-opt`)

Written on the **Apple M2** (`Apple8`, 10-core GPU, 16 GB, macOS 13.2.1, SDK 13.3, stock Chrome 154) on 2026-10-24,
for whoever next works on the **Apple M5** (`Apple10`, macOS 26). It answers docs/M5-TO-M2.md (whose section 5 is the
short version of this) and docs/M2-HANDOFF.md's round seven. Every number is arm against arm on the M2, interleaved,
after one untimed run of each arm; "base" is `9b61f8b4`, the commit before the M5's pass, built in a worktree beside
HEAD. Checksums and pLDDTs do not travel between machines - compare an M5 arm against the M5's other arm.

| commit | what |
|---|---|
| `5a48c897` | `lf_gemm`'s store loop back to its pre-`5646656b` shape; the extended epilogues off on Apple8 (`gemmExtendedEpilogues()`); the selftest reports a declined epilogue as "not taken here"; M2 baselines recorded |
| `40de1a68` | `lf_gemm`'s plain product decided once a tile again, not inside each block's store |
| `3e3e7f80` | docs/M5-TO-M2.md section 5 |
| (this commit) | `msa.project`'s row tile clamped to the device's workgroup storage (section 2, `test:spec-floor`); this document |

**Nothing here changes what the M5 runs by default.** Every code change is in `lf_gemm` (`metal/core/gemm.metal`),
which the M5 runs only under `LOCALFOLD_GEMM_TENSOR=0`, or in a predicate that is true wherever the matrix units are.
What the M5 can do that the M2 cannot is say what these changes cost the M3/M4 class and the M5's own control arm -
section 3.

---

## 1. What the M2 found

### 1A. Every `lf_gemm` was 15x slower (fixed, `5a48c897`)

Found by the M2's AF2 gate taking 7 minutes, `5caj-msa` alone 345 s. Profiled, the same GEMM labels at the same shapes:

| GEMM (AF2, 261 residues, 512 + 1024 rows, one pass) | base | HEAD before the fix |
|---|---:|---:|
| `attention qkvg hhh_64x64_NN_k16_e8 1024x66816x256` | 1270 ms | 19350 |
| `transition1 hhh_64x64_NN_k16_e8 1024x65536x256` | 1243 | 19019 |
| the same qkvg kernel at `1024x130x256 batched` | 808 | 800 |

Reproduced with no fold around it: `./metal/bench/localfold-bench 1024 66816 256 0 hhh 3` - **12.4 ms at base, 189 at
HEAD**, `1024 1024 256` 0.29 against 2.64. Bisected with that command (`git bisect run`, threshold 60 ms) to
**`5646656b` "Port gemmCouplingCombine to Metal Apple8"**, then file by file (only `gemm.metal`), then hunk by hunk
(only the fifth hunk: the store section).

That hunk replaced "a whole tile in range stores in 8 x 8 blocks with both loops unrolled, and returns" with a per-row
loop carrying its own 8 x 8 store for partial tiles. 🔴 **Putting an early whole-tile return IN FRONT of that loop did
not help** (still 173 ms on a bench where every tile is whole and returns before the loop): the cost is the loop's
existence in the kernel, not its execution. That is what static register allocation does - the M2 allocates for the
worst point of the whole kernel, and the M5 (Apple9+) allocates dynamically, which is the most likely reason the M5
never saw it. It is an inference from that experiment, not something read off the compiler. The parent's shape is
back: whole tiles in 8 x 8 blocks, everything else element by element - 13.4 ms.

**What it gives back:** a partial tile's whole rows no longer take 8 x 8 stores; they take the element-wise path, as
before `5646656b`. Its commit message measures nothing for that store, so nothing measured is lost - but nobody has
measured it, which is ask 3B.

### 1B. The extended epilogues lose on every port on the M2 (off on Apple8, `5a48c897`)

`0de9becb` ported the extended epilogues (EP bit 64: `gemmGatedRows` 262144, `gemmGateResidual` 16384,
`gemmCondPermuted` 524288, `gemmMulH` 1048576, `gemmOpmPermuted` 4096, `opmOut` 8192) from the matrix units to `lf_gemm`,
and `5646656b` the coupling combine (32768). Measured on the M5 under `TENSOR=0`, they won 1-4%. On the M2 the extended
instance itself is slow - the bench's plain product with the EP 64 bit and no epilogue is **21.8 ms against 12.6** in
the plain instance (`localfold-bench 1024 66816 256 0,64 hhh 3`; base 12.6 / 12.7) - and every port is slower with them:

| interleaved on the M2: base / extended off / on | |
|---|---|
| AF2 5CAJ, 261 res, 512 + 1024 rows, one pass | 17.4 s / **15.5** / 26.4 |
| AF2 6MRR, 4 passes | 1105 ms / 888-973 / 888 |
| ESMFold2 sampler, 68 res | 274 ms / 185 / **174** |
| ESMFold2 sampler, 261 res | 765 ms / **595** / 808 |
| Boltz-2 total, 68 res | 7.96 s / **7.50** / 11.87 (diffusion 6.15 / 10.5) |
| Boltz-2 total, 261 res | 39.4 s / **35.0** / 51.3 (diffusion 17.6 / 33.8) |
| BioEmu2 6MRR, 2 samples, sampler | 25.4 s / **21.0** / 28.7 |

The OPM's two fusions were the AF2 deep case on their own: `LOCALFOLD_OPM_PERMUTED=0 LOCALFOLD_OPM_ADD=0` took a pass
from 26.4 to 15.5 s (`opm product, permuted` 6743 ms against the unfused product's 1335 plus the permute's ~165).
Compiling the coupling block out of the e64 instance changed nothing (24.0 ms), so it is the instance, not that block.
The e64 regression bisects to `0de9becb` itself (12.5 -> 22.2 ms on the e64 bench arm).

`gemmExtendedEpilogues()` in `metal/core/core.mm`:

```
tensorWanted()                     -> true    (the M5 by default: nothing changes)
LOCALFOLD_LF_GEMM_EXT=0 / 1        -> forced
[device supportsFamily:1009]       -> Apple9+ (M3, M4, and the M5 under TENSOR=0): true, as the M5 measured
                                      Apple8 and older (M1, M2): false - the pre-0de9becb unfused kernels
```

All seven entry points ask it, and BioEmu2's coupling split asks it in place of what `5646656b` removed
(`gemmTensorOps()`), with its derived `csQ2` weights only built where the split runs. Every caller already had its
fallback, the one that died (`denoise.mm:274`) included, because the split decision upstream of it follows the same
predicate. 🔴 **The Apple9 boundary is an assumption, not a measurement**: no M3 or M4 has run either arm. It rests on
the M5's own numbers under `TENSOR=0` (dynamic registers, wins) against the M2's (static, losses).

### 1C. Another 3% of the plain product (`40de1a68`)

With 1A fixed the bench was still 13.2 ms against base's 12.3. Bisected over `9b61f8b4..5646656b^` (threshold 12.8 ms,
7 repetitions) to `0de9becb`: it moved the plain-product check (no epilogue but bias/ReLU, no beta, the accumulator in
the output's type) inside `LF_STORE_8X8`, so each of a tile's FR x FC unrolled blocks carries the general epilogue
beside its plain store. Decided once for the tile again (`PLAIN_IN_PLACE`): 13.24 -> 12.83 ms, the same rep count. The
remaining ~4% to base is spread through the store's restructure and was not chased.

### 1D. Baselines (M2 only)

`metal/af2/gate-baseline.m2.json`: `5caj-msa` (new). `metal/af3/gate-baseline.m2.json`: `af3-5caj-msa` and
`boltz2-5caj-template` (new); `af3-kitchen-sink` 70.19 -> 78.69 and `af3-glycan` 48.67 -> 47.71 (`cb160ba0`, as
docs/M5-TO-M2.md predicted); `chai1-6mrr` 82.74 -> 84.46 and `chai1-6mrr-gol-sep3` (ligand 0.064 -> 0.054) - **not
your pass**: base reads the same. It is `1e670797`'s chai-1 input work, which the M5 recorded for itself and the M2
never had. The M5's baselines are untouched.

---

## 2. What the M2 confirmed (no change made)

**Metal gates after the fixes:** selftest 123 ok in both `LOCALFOLD_LF_GEMM_EXT` arms (forced on, the e64 kernels stay
correctness-tested on Apple8); AF2 6/6 (the gate 7 min 9 s -> 1 min 24 s); ESMFold2 8/8; AF3 18/18;
`tools/check-native-worker.py --offline --no-page --no-af3` ok on every case. `npm test` 1372/1373 (the one is an
untracked local file on this box).

**Metal kernels, docs/M5-TO-M2.md 3B:**

| kernel | M2 | verdict |
|---|---|---|
| `af3_atom_attention_mma` vs `AF3_SCALAR_ATOM_ATTENTION=1` (Boltz-2, 68 res) | diffusion 6.09-6.18 against 6.74-6.86 s | MMA wins |
| blocked triangle, `LOCALFOLD_TRI_BLOCKS=4` vs 1 (ESMFold2) | 740 / 744 ms at 68, 10.77-10.85 / 10.84-10.91 s at 261 | level, bit-identical |
| `ef2_ada_ln` (ESMFold2 261, `--profile`) | 26 ms over 253 calls; the sampler is faster than base | fine |
| `lf_center_norm_32h1` against base's `lf_center_norm_32` | 552-559 against 347-360 ms in the 261-res trunk | slower, but... |
| ...the half-product path it belongs to, `LOCALFOLD_HALF_PROD=1` vs 0 | ESMFold2 trunk 10.80 vs 11.01 s; Boltz-2 16.77 vs 17.01 s | ...wins overall; kept |

**WebGPU, stock flags (`LOCALFOLD_STOCK_FLAGS=1`), the `metal-3` prior's values:**

| value | M2 | verdict |
|---|---|---|
| `runtimeLoopBounds: false` (cold Metal shader cache each arm, `fold.js --folds=2`) | 6MRR first fold 4.95 / 4.96 s against tiered 6.33 / 7.67; 5CAJ 15.7 against 20.6; second folds level | **holds** |
| `diffusionTokenTile` (the prior) against a flat 4 (`bench-head.js`) | 68 tok 46-47 / 47; 150 tok 76-77 / 84-85; 255 tok 135 / 150-152 ms | **holds** |
| `atomRowTile` 4 against the derived rule | 44 / 44 ms at 68 tokens, 132 / 132 at 255 | level |
| AF2 vector OPM (`opmVectorOutput`/`Contract`) | block 145 against 155 ms (150 res x 256) | holds |
| `gridAttendTiled` (round seven) | `grid.attend` 571 -> 952 ms, pairformer 2287 -> 2549 ms at 200 tokens | **loses**: stays ampere-only |
| `attentionTiled` (round seven) | AF2 block 144-147 -> 150-157 ms (200 res x 128) | **loses**: stays ampere-only |

🔴 One oddity on the tiered arm: its second round's pLDDT read **86.6637** where every other arm read 86.6648, both
folds of that run included - so on the M2 a tiered fold is not always bit-identical to the constant-bound one. Tiered
is off on Apple so it does not ship here, but it is every other device's default and its commit says bit-identical
everywhere checked; ask 3D.

**WebGPU gates:** `check-quantised-upload` 0 differing at every codec; `test:stock` folds AF3, OpenDDE, ESMFold2,
boltz2, protenix2, IntelliFold-2, RF3; `test:template` passes all six AF3-lineage models present (AF3 0.671 / 19.442 A,
boltz2 0.380, protenix2 0.388, OpenDDE 0.266, IntelliFold-2 0.341, RF3 0.124) and AF2's monomer by hand (below);
`test:ligand` 5/5; `test:modified` 7/7 (openbind0 skipped: no f32 bundle here).
`test:cache` - six AF3-lineage models both orders in one pipeline cache, no collision, every confidence scored (its AF2
and ESMFold2 tool arms then stop at the stale AF2 bundle below); `test:portable` 7 of 9 (AF2 and BioEmu2: bundles).

🔴 **`test:spec-floor` FOUND ONE IN YOUR PASS, FIXED HERE (`msa-stack.js`).** `b8169ca8` gave `msa.project` 16 MSA
rows a workgroup, staging `R x (C_M + 128)` floats and R statistics - 16,512 bytes at a 128-channel MSA, 128 over the
16,384 WebGPU guarantees, so protenix2, OpenDDE and IntelliFold-2 failed `While initializing [ComputePipeline
"af3-msa:...:msa:project:8x8:r16"]` (AF3's 64 channels, 12,416 bytes, fit). `msaProjectRows` now halves the row count
until the tile fits `maxComputeWorkgroupStorageSize`; the kernel is bit-identical at any row count, and on any device
reporting 32 KiB (both Apple chips, the A100) nothing changes - at the floor protenix2 reads 84.6931 and OpenDDE
92.0450, their `test:portable` figures to the digit, IntelliFold-2 83.182, AF3 unchanged. Its OpenDDE row also hung
once (node at 0% CPU, Chrome gone, 41 min - CLAUDE.md's "sometimes does not exit"); killed and re-run by hand.

**Not code, this box:** the registry's monomer is int5 at `./model/` and this checkout's `./model/` is an old int8
export, so every tool that opens the monomer by its registry name 404s (`test:stock`'s and `test:template`'s AF2 rows);
`fold-af2.js --bundle=/model-af2-monomer-int5` folds stock at pLDDT 62.912 (repeat identical) and its template arms at
**2.494 A with, 22.162 without**. No AF2 multimer, BioEmu2 WebGPU or openbind0 bundle is here.
`check-evoformer-stack.js` needs fixtures no checkout has.

---

## 3. What the M5 should check

### 3A. The fixes cost the M5 nothing by default - confirm it

```bash
git pull --rebase origin opus-55-opt && bash metal/build.sh selftest bench af3 af2 ef2 bioemu2
./metal/selftest/localfold-selftest                         # matrix units: every epilogue "ok", none "not taken"
LOCALFOLD_GEMM_TENSOR=0 ./metal/selftest/localfold-selftest # Apple10 lf_gemm: the e64 cases still taken (Apple9+)
python3 metal/af2/gate.py && python3 metal/ef2/gate.py && python3 metal/af3/gate.py    # against gate-baseline.m5.json
```

The default path should fold the same as `5e874012` (the commit before the M2's): everything the matrix units run is
untouched. Any GEMM the M5 still sends to `lf_gemm` by default (operands the tensor kernel does not take) goes through
the changed store, which computes the same values in the same order - expected identical, and the gates say. Under
`TENSOR=0` the selftest's e64 cases must still print `ok`, not "not taken here" - if they do not, `supportsFamily(1009)`
answers differently on macOS 26 than assumed, and the predicate is wrong for the M3/M4 too.

### 3B. The M5's control arm, `LOCALFOLD_GEMM_TENSOR=0`, before and after `5a48c897` and `40de1a68`

The M2's fixes revert two M5-measured changes in `lf_gemm`. On the M5 they should cost little - its registers are
dynamic - but nobody has measured it:

```bash
git worktree add ../lf-pre 5e874012 && (cd ../lf-pre && bash metal/build.sh bench af2 af3 ef2 bioemu2)
for r in 1 2; do for root in ../lf-pre .; do (cd $root && LOCALFOLD_GEMM_TENSOR=0 \
  ./metal/bench/localfold-bench 1024 66816 256 0,64 hhh 7 | tail -2); done; done
# partial tiles - the per-row 8 x 8 store 5a48c897 removed: a row count that is not a multiple of 64
for r in 1 2; do for root in ../lf-pre .; do (cd $root && LOCALFOLD_GEMM_TENSOR=0 \
  ./metal/bench/localfold-bench 1024 66850 256 0 hhh 7 | tail -1); done; done
```

then AF2 6MRR, ESMFold2, Boltz-2 and BioEmu2 at 68 and 255 residues under `TENSOR=0`, the commands in
docs/M5-TO-M2.md step 2. If the per-row store was worth something on Apple9+, bring it back **behind a compile-time
condition the host sets only where registers are dynamic** (the GEMM source is compiled at run time, so a `#define`
from `gemmExtendedEpilogues()`'s family check works) - never in the shared path, or the M2 pays 15x again.

### 3C. Make `lf_gemm` regressions visible from the M5

Both regressions were invisible to every gate the M5 ran: the gates check answers, and the answers never moved. A speed
check would have caught them in seconds, and the bench already exists. Two cheap ideas, for the M5 to judge:

1. **A per-chip bench baseline**, the way `gate-baseline.<chip>.json` works for answers: `localfold-bench`'s plain and
   e64 arms at two shapes, recorded per chip, a gate failing at 1.3x. It only fires on the chip that regressed, so it
   needs an M2 run before merging an `lf_gemm` change - which is the honest cost of a kernel two chips run differently.
2. **A register-pressure proxy readable on the M5.** `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` falls
   when a kernel's registers rise on a static-allocation GPU. Whether the M5 reports anything useful there for an
   `lf_gemm` instance is unknown - worth one look, comparing `5e874012` and HEAD's instances. If it moves, it is a way
   to see the M2's problem without an M2.

### 3D. WebGPU: the shared `metal-3` prior

- Round seven's two register-tiled attentions LOSE on the M2 (section 2). If they win on the M5, the prior cannot
  just take them (the two chips report the same `metal-3`): a startup race like `webgpu/af3/trunk/grid-attend-race.js`,
  or leave them off. Commands are in docs/M2-HANDOFF.md, "ROUND SEVEN ASK".
- Tiered's non-bit-identical second fold: `fold.js --folds=2 --tune-json='{"runtimeLoopBounds":"tiered"}'` a few times
  on the M5 with a cold shader cache, comparing `meanPlddt` to the last digit against the constant-bound arm. If it
  differs there too, the generic kernel does not compute the same sum as the specific one, which matters for NVIDIA,
  where tiered is the default.

### 3E. Leave alone

- The M2's baselines (`*.m2.json`) - re-recording them from an M5 is how `a56d94d2`'s problem started (the M5 re-recorded the shared file and the M2's gate failed on its own unchanged fold).
- `runtimeLoopBounds: false` and `diffusionTokenTile` in `metal-3`: both now measured on both chips.

---

## 4. How the M2 measured (to reproduce or extend)

- **Base worktree:** `git worktree add ../lf-base 9b61f8b4`, built with `bash metal/build.sh ...`; the binaries take
  `--weights-dir=<checkout>` so both read one set of weights.
- **Three-arm A/B:** base, HEAD with `LOCALFOLD_LF_GEMM_EXT=0`, HEAD with `=1`; one untimed run of each, then two
  rounds alternated, reading the binary's own stage lines (`trunk`, `sampler`, `fold:`, `pass N/4`).
- **Bisecting a kernel's speed:** `git bisect run` over `-- metal/core` with a script that rebuilds only the bench
  (`bash metal/build.sh bench`, ~10 s) and thresholds its time - five steps, two minutes. A run of the full port per
  step would have been an hour. Then the commit split by file (`git checkout <c>^ -- <file>`) and by hunk (`git diff`
  split into one patch per hunk, applied alone onto the parent).
- **A cold shader cache for WebGPU:** move `$(getconf DARWIN_USER_CACHE_DIR)com.google.Chrome.helper/com.apple.metal`
  aside before each arm, discard what the arm compiled, put it back. The harness runs the regular Google Chrome, so
  that is the right helper's cache; it is also the user's browser's, which is why it goes back.
- **zsh:** an unquoted `$var` holding `1024 66816 256` is ONE argument; the bench then read `EP+3 0.000 ms`. Run
  multi-argument loops under `bash -c`.
- **The harness's stderr carries a `{`** (the adapter line), so a JSON parser fed `2>&1` fails with "Extra data":
  `2>/dev/null | grep -v '^\[gpu-chrome\]'`.

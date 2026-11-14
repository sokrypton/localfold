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

### 3D-2. Five `metal-3` values the M2 lost on, now raced - measured on BOTH parts (2026-11-14)

Section 2 confirmed the `metal-3` values it swept; five it never swept do not hold on the M2. `linearHalfTile`,
`attentionOutputHalfTile`, `attentionProjectHalfTile` (7bb3fe2f), `trianglePairProjectTile` (a6b916c8) and
`vectorGemmBlock` were the M5's, each marked "not measured on an M2". M2, Chrome 157, `profile-af2-block.js --sweep`,
two rounds interleaved, the pinned value against the default, kernel ms:

| shape | msa-transition | attention outputs (row / col) | q/k/v/gate (row / col) | triangle output | opm contract + output |
|---|---|---|---|---|---|
| 59 x 128 | 3.23 + 3.42 -> 2.28 + 2.21 | 1.11 / 1.13 -> 0.64 / 0.64 | 3.06 / 3.06 -> 2.11 / 2.11 | 0.34 -> 0.17 | 0.90 + 0.97 -> 0.71 + 0.72 |
| 256 x 128 | 14.2 + 14.6 -> 9.9 + 9.6 | 4.74 / 4.98 -> 2.72 / 2.79 | 13.2 / 13.3 -> 9.0 / 9.0 | 5.81 -> 2.93 | 16.8 + 16.9 -> 13.4 + 12.7 |

and in an AF3 trunk pass (`bench-trunk.js --profile`, 255 tokens, the race's answers named by hand): the triangle tile
5683 / 5710 -> 5247 / 5261 ms, then the GEMM block 5261 -> 4964 (`pair-transition.wide` 7.46 -> 5.80, `down` 3.94 ->
3.34); protenix2 at 150 tokens 5160 -> 4866, IntelliFold-2 20870 -> 19874.

So the prior leaves all five null and `grid-attend-race.js` answers them: a fifth arm races the transition's f16
projection (8192 rows, 1024 -> 256) on the default tile against 16x16x4x8 and sets the M5's values where the wide arm
wins by the margin; otherwise the half tiles and the GEMM block stay null (the defaults) and the triangle takes 32x32.
The triangle tile and the GEMM block ride that answer by proxy - the same question, a lane's register tile - and the
proxy is right on both parts, measured:

| | race: default / wide tile | answer | `vectorGemmBlock` f16, AF3 trunk pass |
|---|---|---|---|
| M2 | 2.47 / 4.05 ms | narrow | 64x128x32 5261 ms, **64x64x16 4964** |
| M5 | 1.47 / 1.19 ms | wide (all five at their old values) | **64x128x32 2341**, 64x64x16 2410 |

Whole folds, warm, the five pinned by `--tune-json` against the race, alternated:

| fold | M2 pinned | M2 raced | M5 pinned | M5 raced |
|---|---|---|---|---|
| AF2 59 residues, 128 + 128 rows | 1244-1276 ms | **912-922** | 453-456 | 454-457 |
| AF2 5CAJ (261), 128 + 128 rows | 11.9 s | **9.1** | 4.60 | 4.60 |
| AF2 5CAJ, 512 + 1024 rows | 29.7 s | **23.1** | - | - |
| AF3 6MRR (68 tokens), 25 steps, one pass | 1.61 s | 1.57 | - | - |
| AF3 5CAJ (255 tokens), 25 steps, one pass | 9.40 s | **8.65** | 4.85-4.86 | 4.84-4.86 |

The M5's folds are bit-identical either way (AF2 -1349430 / -774811, AF3 pLDDT 26.965896894539252). On the M2 the four
tiles reorder no sum (AF2 -1361689 both ways with those alone); the GEMM block regroups the f16 partial sums, so with
it AF2 reads -1364228 (pLDDT 62.912 -> 62.918) and AF3 5CAJ 26.9758 -> 26.9549 - the f16 bound, not a tile. The race
is ~60 ms longer on the M2 (745 -> 805 ms, once a browser; the memory key is v5, so every browser races once more) and
433 ms on the M5. `npm test` is unchanged (1910 pass; the one failure is `tools/_fold_film.py`'s stale paths, before
and after). The AF3 template stack awaits the race now (it reads the triangle tile and compiles before the pairformer).

🔴 **`--profile` MISLEADS THIS ARM TOO**, as the race's header says of the others: under `bench-trunk.js --profile` the
M2's race answered WIDE (tri.project-out 5.77 ms a pass against 2.87). Name all the answers with `--tune-json` when
profiling. And a run that skips the race with `--tune=gridAttendTiled=...` gets the default tiles unless it names the
five as well.

🔴 **THE M5 IS REACHABLE FROM THE M2: `ssh m5`** (Tailscale). These M5 numbers were taken from a copy of the working
tree in `~/sergey/lf-m2test` there (tracked files by rsync, the bundles and node_modules symlinked from
`~/sergey/localfold`), which leaves that checkout alone.

Not swept on the M2: `singleTransitionTile` and `singleProjectRows` (under 1% of a trunk pass here).

### 3D-3. Two more 256-thread f16 tiles on 64 lanes, and a knob sweep that found nothing - BOTH parts (2026-11-14)

Asked: a speedup for either chip that costs the other nothing. Every 16 x 16-lane f16 tile the M5 work introduced has
lost on the M2 to an 8 x 8-lane twin (3D-2), so the two kernels that still had no twin got one. Both are answered by
the startup race's existing "linear tile" arm (`tilesAnswer` in webgpu/af3/trunk/grid-attend-race.js): no new race.

**1. The f16 triangle contraction, `triangleContractLanes`** (8, `"fit"`, null/16; webgpu/kernels/triangle/shaders.js,
`f16ContractTile`). The tile is `4 * lanes` a side, so 8 lanes is a 32 x 32 tile where 16 is 64 x 64 - which also pads
less whenever the length is not a multiple of 64. `tools/gpu/probe-contract-lanes.js` is the standalone race.

| AF3 trunk, `tri.contract` ms | M2: 16 -> 8 lanes | M5: 16 -> 8 lanes |
|---|---|---|
| 255 tokens (pads to 256 either way) | 494 -> 375 (trunk -2.2%) | **loses 2%** |
| 200 tokens (pads to 224, not 256) | 359 -> 215 (trunk -5%) | 129 -> 114 (trunk -1%) |

So the M2 takes 8 and the M5 takes `"fit"`: 8 only where the 32-tile's padded area is 13% under the 64-tile's.

**2. The f16 vector GEMM on 64 lanes** - the split pair transition's two and the outer product mean's two
(`vectorGemmBlock`, a fourth number naming the lane grid's side: `"32x64x8x8"`). AF3 trunk at 255 tokens, GPU ms:

| block (f16) | M2 `wide` / `down` | M2 trunk | M5 trunk |
|---|---|---|---|
| default 64x64x16 (M2), 64x128x32 (M5) | 602 / 345 | - | - |
| **32x64x8 on 64 lanes** | **455 / 255** (twice: 460/258, 457/256) | **-3.9, -5.5, -4.8%** | +1.9% |
| 32x64x16 on 64 lanes | 568 / 309 | +0.8% | |
| 32x128x8 on 64 lanes | 749 / 408 | +5.2% | +4.4% (x16) |
| 64x64x16 on 64 lanes | 816 / 456 | +8.6% | |
| 64x128 on 64 lanes | 1787 / 928 | +43.5% | +14..16% |

The M2 takes `32x64x8x8`; the M5 keeps its own block, which beats every 64-lane one there. A 32-column block is not
one AF3's OPM output kernel has (it reads the block's columns and depth only; 64 rows on 256 lanes) - its `fits` says
so now, where it compiled `tx * 0.5u`.

**2b. The contraction wants a narrower block than the transition does, so it has its own** (`vectorGemmBlock.f16Contract`,
read through `vectorGemmBlock(..., "Contract")`; absent, the arithmetic's). M2, AF2 block at 255 residues x 128 rows:

| block (f16, 64 lanes) | `opm.contract` | `opm.project-output` | AF3 `pair-transition.wide` / `down` |
|---|---|---|---|
| default 64x64x16 on 256 | 12.9-13.8 ms | 13.0 | 602 / 345 |
| 32x64x8 | 12.6-13.0 | **9.1-9.8** | **456 / 255** |
| 32x32x8 | **10.3-10.4** | 10.0-10.2 | 517 / 311 |
| 32x32x16, 32x32x32, 32x64x32 | 15.0, 16.1, 18.6 | 11.7-15.5 | |

The M2's race answer is `{ f16: "32x64x8x8", f16Contract: "32x32x8x8" }`: 10.4 and 9.1 together, the AF2 block 157 ->
150 ms, the same checksum as one block for both (the k step and the lanes are the same, so no sum is regrouped). It
is ~1% of a 5CAJ AF2 fold (7.93-8.06 s against 8.07-8.10). The M5's answer has no such entry and its fold is the
fold it was (4218 / 4220 ms, checksum unchanged).

**Both together, whole folds, arms interleaved, two rounds** (`fold-af2.js --repeat`, `fold.js --folds=2`, warm fold;
old arm = `--tune-json={"triangleContractLanes":16,"vectorGemmBlock":{}}` on the M2, `{"triangleContractLanes":16}`
on the M5, whose GEMM block did not change):

| | M2 old | M2 new | M5 old | M5 new |
|---|---|---|---|---|
| AF2 59 res | 927, 928 / 905, 891 ms | 895, 900 / 883, 874 | 464, 461 / 481, 482 | 465, 463 / 468, 471 |
| AF2 200 res, 128+128 rows | 5778 / 5653 | 5382 / 5401 (-5.5%) | 2919 / 3087 | 2877 / 3027 (-1.7%) |
| AF2 5CAJ 255, 128+128 rows | 8433 / 8220 | 7909 / 7888 (-5%) | 4315 / 4506 | 4222 / 4453 (same kernels, same checksum) |
| AF3 200 tokens | 8.08 / 8.09 s | 7.82 / 7.72 (-3.9%) | 5.12 / 5.34 | 5.00 / 5.33 |
| AF3 5CAJ 255 tokens | 11.78 / 11.84 s | 11.54 / 11.44 (-2.7%) | 7.30 / 7.72 | 7.03 / 7.39 (same kernels, same pLDDT) |

Every fold is a chain (CA-CA median 3.77-3.90 on both arms) and pLDDT moves in the second decimal. 🔴 **THE M5 DRIFTS
~5% SLOWER OVER SUCH A RUN AND THE M2 UP TO 20% OVER HALF AN HOUR** (heat): read adjacent pairs, never a column.
`probe-grid-attend-race.js` prints what each part resolved: the M2 `vectorGemmBlock {f16: "32x64x8x8"}`,
`triangleContractLanes 8`; the M5 `{f32: "128x128x8", f16: "64x128x32"}`, `"fit"` - what it had.

**3. The knob sweep, a null result on both.** `profile-af2-block.js --sweep` over 12 AF2 knobs and
`bench-trunk.js --profile --tune-json` over 31 AF3 arms, on each part: nothing beyond the above moves either by more
than its noise. The two M5 candidates at ~1% (`linearHalfTile 16x16x8x8`, `attentionProjectHalfTile 16x16x8x2`) are
inside its bimodal run-to-run band and unconfirmed. Both parts are GPU-bound (no host-side gap). 🔴 `bench-trunk.js
--profile` FOOLS THE RACE (the M2 answered wide under it): name the race's answers in `--tune-json` when profiling.

### 3D-4. `grid.project-out` as a register-tiled GEMM - a gain on BOTH parts, in the shared prior (2026-11-14)

Per-kernel throughput in an AF3 trunk pass at 255 tokens (multiply-adds x 2 over GPU time, both parts' own tiles):
the GEMM-shaped kernels sit at 1.7-2.1 TFLOP/s on the M2 and 3.0-3.4 on the M5, and `grid.project-out` at **1.0 and
1.6**. It gave a lane ONE output channel: a scalar weight read and two staged reads bought 8 multiply-adds, in f32.
`gridProjectOutVector` (the rows a tile; `projectOutVector` in webgpu/af3/trunk/grid-attention.js) gives a lane four
adjacent channels and eight rows - one vec4 weight read and two staged reads buy 32 - in f16 where the gated rows are
staged in f16 (eight k summed in f16, then into f32). Rows are cut to what the device stages (width x rows elements).

| AF3 trunk, 255 tokens, `grid.project-out` ms | M2 | M5 |
|---|---|---|
| one channel a lane, 8 rows | 216 | 133 |
| **32 rows** (the prior's) | **181, 182** (trunk -0.6%) | **118, 117** (trunk -0.8%) |
| 16 rows | 181 | 114 |
| 64 rows | 228 | 123 |

16 rows is no slower with half its lanes idle, so the kernel is no longer arithmetic-bound: what is left is the staging
(a logistic an element) and the weight read from global memory. `check-af3-block-any.js` (two blocks against the CPU
reference, int5): pair 0.117040 -> 0.117048, single 1.50e-3 -> 1.49e-3. Whole folds, warm, arms interleaved twice
(old arm `--tune-json={"gridProjectOutVector":0}`):

| | M2 old | M2 new | M5 old | M5 new |
|---|---|---|---|---|
| AF3 200 tokens | 7.81 / 7.82 s | 7.58 / 7.73 | 5.00 / 5.16 | 4.98 / 5.02 |
| AF3 5CAJ 255 | 11.43 / 11.49 | 11.22 / 11.36 | 7.05 / 7.32 | 7.01 / 7.19 |

boltz2, rosettafold3 and openbind0 at 68 tokens take the kernel and fold the same chain at the same speed (pLDDT moves
in the second decimal); intellifold2, protenix2 and OpenDDE do not take it (their folds are identical to the digit).

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

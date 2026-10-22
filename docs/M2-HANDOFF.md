# Handoff to the M2: Verifying & Tuning the 17-Round M5 Pass (`opus-55-opt`)

Written on the Apple M5 (`Apple10`, macOS 26.6) on 2026-10-22 for an agent working on the **Apple M2** (`Apple8`, 10-core GPU, 16 GB).
Everything is committed and pushed on branch **`opus-55-opt`** (`976a298d..0aeeb53b`).

---

## 1. Why This Handoff Exists (M5 vs. M2 Architecture Gap)

Over 17 commits (`976a298d..0aeeb53b`), we optimized all four model families (`af3`, `af2`, `ef2` / `esmc`, `bioemu2`) across both native **Metal** (`metal/`) and **WebGPU** (`webgpu/`) on an Apple M5.

While ~80% of those optimizations reduce memory traffic, dispatch count, or host/device overhead on **all** GPUs, the M5 and M2 differ in three hardware ways that can invert kernel-level choices:

1. **Dynamic Caching (M3+ / `Apple9`+) vs. Static Register File (M2 / `Apple8`)**:
   - On the M5, unified SRAM is dynamically partitioned between registers, L1, and threadgroup memory. Holding `12..16 x vec4<f32>` (48–64 floats per thread) in private registers to stage a full LayerNorm/AdaLN row is essentially free in occupancy.
   - On the M2, registers are statically carved out per simdgroup. Increasing per-thread registers from 32 to 64–96 can drop active simdgroups per core by 2x or spill to device stack memory.
2. **WebGPU `adapter.info` Cannot Distinguish M2 from M5**:
   - Under stock Chrome flags (`LOCALFOLD_STOCK_FLAGS=1`), both M2 and M5 report `{ vendor: "apple", architecture: "metal-3", device: "", description: "" }`.
   - Therefore, any change to `PRIORS.get("metal-3")` in `webgpu/runtime/device-profile.js` or any shader change in `webgpu/` runs identically on M2 and M5 unless gated by a runtime micro-race (like `webgpu/af3/trunk/grid-attend-race.js`).
3. **Metal 4 Tensor Units (`Apple10`) vs. Metal 3 `simdgroup_matrix` (`Apple8`)**:
   - In native `metal/`, `lf_gemm_tensor` (`metal/core/gemm_tensor.metal`, using Metal 4 `matmul2d` on M5's per-core neural accelerators) is gated on `tensorWanted()` (`Apple10` + macOS 26) in `metal/core/core.mm`.
   - On the M2, `tensorWanted()` is `false`, so `gemmRun` uses `lf_gemm` (`metal/core/gemm.metal`) and clears `a.lnOut = 0`. However, all non-tensor kernels in `metal/` (`common.metal`, `af3.metal`, `af2.metal`, `ef2.metal`, `bioemu2.metal`) run directly on the M2.

---

## 2. Engineering Rules & Setup on the M2

1. **Engineering invariants (`AGENTS.md`, `CLAUDE.md`, `metal/README.md`)**:
   - Never widen tolerances or overwrite reference tensors / gate baselines merely to make a failing test pass.
   - Run GPU benchmarks strictly one at a time, and measure speed arms **interleaved (`ABBA` or `BAAB`)** on warm shader/pipeline caches.
   - Keep `src/` and `webgpu/` directly loadable by the browser (plain ES modules, relative `.js` specifiers, no bundler or build step).
   - Never stage or commit `package-lock.json` or `localfold-fixes-for-sergey.md`.
   - When a tuning knob disagrees between M2 and M5:
     - In **Metal (`metal/`)**: branch at runtime using `[device supportsFamily:MTLGPUFamilyApple9]` (M3/M4/M5 with Dynamic Caching) vs `MTLGPUFamilyApple8` (M2), or `tensorWanted()` (`Apple10`).
     - In **WebGPU (`webgpu/`)**: either pick a threshold (e.g. register-staging row width cap) that wins on both, or add the arm to the startup micro-race (`webgpu/af3/trunk/grid-attend-race.js`) rather than regressing either chip.
2. **Environment & build commands**:
   ```bash
   export PATH="$HOME/sergey/.venv-localfold/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
   git checkout opus-55-opt && git pull --rebase origin opus-55-opt

   # Build all native Metal binaries
   bash metal/build.sh selftest bench af3 af2 ef2 bioemu2
   ```
   *(Note: When benchmarking two Metal binaries side-by-side, place them in `metal/<port>/localfold-<port>-a` and `metal/<port>/localfold-<port>-b` rather than running from `/tmp` or overwriting `metal/<port>/localfold-<port>` on each run: `lf::standalone::main` locates repo assets relative to `dirname(argv[0])`, and overwriting a Mach-O executable in place triggers ~45 ms of macOS `amfid` code-signature re-verification.)*

---

## 3. Classification of the 17 M5 Rounds (`976a298d..0aeeb53b`)

### A. Inert on M2 (`Apple10` Tensor-Unit Only)
These changes live inside `metal/core/gemm_tensor.metal` or `gemmTensorRun` in `metal/core/core.mm` (`tensorWanted() == true`):
- `172f226c`, `d1778df5`, `41555140`, `63e12b5c`, `91f35240`, `7d2cbdf7`, `0aeeb53b`:
  - `lf_gemm_tensor` butterfly register SwiGLU (`e40`), `16x256` and `32x256` single/dual emitted LayerNorm epilogues (`e64`, `e192`, `e256`), `yinFloat` (`biasType |= 512`), and `16x256` dual triangle-output GEMM (`lf_gemm_tensor_dual`).
  - **How M2 handles `g.lnOut` / `g.lnOut2` and `g.yinFloat`**:
    - When `gemmRun` runs on M2 (`tensorWanted() == false`), line 788 of `metal/core/core.mm` sets `a.lnOut = 0` and returns `false`. Thus `recordEmitted` is never called, `takeEmitted(dst)` returns `false`, and every caller in `af3`, `af2`, `ef2`, and `bioemu2` executes its standard `layerNorm(...)` path automatically.
    - `g.yinFloat` (`a.biasType |= 512`, adding a `float*` residual `C` to a `half`-output GEMM) is supported in **both** `lf_gemm_tensor` (`metal/core/gemm_tensor.metal`) and `lf_gemm` (`metal/core/gemm.metal`), and has been verified across all four Metal ports (`af3`, `af2`, `ef2`, `bioemu2`) and `localfold-selftest` under `LOCALFOLD_GEMM_TENSOR=0`.

### B. Universally Faster on Both M2 and M5
- **Load-time weight preparation & decode-time SwiGLU interleaving** (`63e12b5c`, `426f0a57`, `7d2cbdf7`, `0aeeb53b`):
  - `isSwiglu` in `loadBundle` (`metal/core/core.mm`), `prepareWeights()` in `metal/af2/evoformer.mm`, `prepareTrunkWeights()` / `prepareAtomWeights()` / `prepareDiffusionWeights()` in `metal/ef2`, and decoding small weights (`>= 2048` elements and 2D AF2/EF2 linear weights) to `half` at load time.
- **Producer kernels writing `half*` directly** (`7d2cbdf7`, `0aeeb53b`):
  - Dual `float* y + half* yh` output in `lf_layernorm` / `lf_layernorm4` (`metal/core/common.metal`), `af2` embedder/template/IPA/structure transitions writing `half*`, and `ef2` `ef2_atom_features`, `ef2_coords_input`, and `ef2_distance_embed` writing `half*`.
- **`ef2` Confidence Head memory & dispatch cuts** (`0aeeb53b`, `metal/ef2/confidence.mm`):
  - Fused `confidence/zNorm` into `ef2_conf_z` in-place on `zTrunk` (1 simdgroup per pair row instead of 256 threads per pair), fused `toHalf` into `ef2_distance_embed`, and replaced `toFloat + add` with `ef2_add_half` (eliminating two `P * C * 4` float buffers — `2.2 GB` saved at 1044 tokens).
- **Step-invariant & pointer-batched diffusion conditioning** (`976a298d`, `3be1a173`, `172f226c`, `426f0a57`, `35dfa517`, `f285d862`):
  - Hoisting step-invariant conditioning projections outside the step loop in `metal/bioemu2`, `metal/ef2`, and `webgpu/esmfold2`.
- **WebGPU contiguous `int5` unpack & bank-conflict-free shared tiles** (`35dfa517`, `befd793d`, `52f8dd56`):
  - Contiguous-pair fast path in `webgpu/weights/quantised-upload.js` (`-65.5%` unpack GPU time), `+1`/`+4` stride padding in `contractF16` (`webgpu/kernels/triangle/shaders.js`), `createTranspose2dShader`, and `createOuterProductMeanShader`, and `tuning.matrixLinear !== false` gating for `splitTransitionConfig` / `pairTransitionSplit`.

### C. Suspects to A/B Test on the M2 (Where M5 and M2 Can Diverge)

These are the exact items tuned on M5 where M2's static register file, 10-core GPU size, or cache hierarchy could behave differently:

| # | Area | Commit & Files | Why It Might Differ on M2 |
|---|---|---|---|
| **1** | **WebGPU `atomRowTile: 4` pinned in `metal-3`** | `ed330437` (`webgpu/runtime/device-profile.js:1345`) | Pinning `atomRowTile = 4` (14 KiB workgroup memory, `vec4<f32>`) beat tile 1/2 (at `<1024` atoms) and tile 8 (at `>2048` atoms) on M5. On M2's 10-core GPU, check if tile 4 at 68 tokens (`~576` atoms = 144 workgroups across 3 splits) under-occupies or beats `DEFAULT_TUNING` (`atomRowTile: null`). |
| **2** | **WebGPU register-staged full-row LayerNorm / AdaLN (`C = 768..1024`)** | `ed330437` (`webgpu/af3/diffusion-conditioning.js`, `diffusion-transformer.js`, `trunk/transition.js`), `f285d862` (`webgpu/bioemu2/ops.js`), `35dfa517` (`webgpu/esmc/block.js`) | Holds up to `16 x vec4<f32>` (64 floats/lane) in private registers to avoid re-reading global memory. Great with M5 Dynamic Caching; on M2 (`Apple8`), `>= 48` floats/lane in private registers can halve simdgroup occupancy. |
| **3** | **WebGPU 4-row x 64-lane (256-thread) workgroup packing** | `ed330437` (`webgpu/af3/atom-attention.js`), `f285d862` (`webgpu/bioemu2/ops.js` `lnFastSource` & `csAtomPrep`) | Packs 4 atom rows of 64 lanes into one 256-lane workgroup. Check both numerical stability and speed on M2. |
| **4** | **WebGPU AF2 Vector OPM (`opmVectorOutput`, `opmVectorContract`)** | `befd793d` (`webgpu/af2/evoformer.js`) | Enabled when `opmMatrixContract === false` (which `metal-3` sets). Won `-8.8%` to `-14.6%` on M5; verify on M2 across `L = 59, 150, 255`. |
| **5** | **Metal unrolled `float4` register staging in AdaLN (`C = 768..1024`)** | `0c3dd2dd` (`metal/bioemu2/bioemu2.metal`), `70248dce` & `dedf5ec0` (`metal/af3/af3.metal`), `d8620d9b` (`metal/ef2/ef2.metal`) | Unrolled `float4` register arrays in `af3_ada_ln*`, `be2_ada_ln*`, `ef2_ada_ln`. Check on M2 with `--profile` that static register allocation does not spill. |
| **6** | **Metal `af3_atom_attention_mma` at `keys <= 128` & single-block `triContract` up to 32 MB** | `41555140` (`metal/af3/atom.mm:164`, `metal/core/core.mm:1205`) | Both run on M2 (`af3_atom_attention_mma` uses Metal 3 `simdgroup_float8x8`). Verify `AF3_SCALAR_ATOM_ATTENTION=0` vs `1` and `LOCALFOLD_TRI_BLOCKS=1` vs `4` on M2. |

---

## 4. Step-by-Step Action Plan on the M2

### Step 1: Run Correctness Gates on the M2 (Metal & WebGPU)
```bash
export PATH="$HOME/sergey/.venv-localfold/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# 1. Native Metal self-test & gates
bash metal/build.sh selftest bench af3 af2 ef2 bioemu2
./metal/selftest/localfold-selftest
python3 metal/af2/gate.py
python3 metal/ef2/gate.py
python3 metal/af3/gate.py
python3 tools/check-native-worker.py --offline --no-af3 --no-page

# 2. WebGPU differential checks & stock-Chrome suite
node tools/gpu-chrome.mjs tools/gpu/check-quantised-upload.js
node tools/gpu-chrome.mjs tools/gpu/check-af3-pairformer-block.js
node tools/gpu-chrome.mjs tools/gpu/check-af2-evoformer-block.js
LOCALFOLD_STOCK_FLAGS=1 npm run test:stock
```

### Step 2: Run the WebGPU A/B Benchmarks on the M2 (`LOCALFOLD_STOCK_FLAGS=1`)

Run each pair interleaved (`ABBA`) on warm caches:

1. **Test `atomRowTile: 4` vs `null` (Suspect #1)**:
   ```bash
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-head.js --tokens=68 --rounds=3 --tune=atomRowTile=null
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-head.js --tokens=68 --rounds=3
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-head.js --tokens=255 --rounds=3 --tune=atomRowTile=null
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-head.js --tokens=255 --rounds=3
   ```
   - **If `atomRowTile=null` wins at 68 tokens on M2**: change `atomRowTile` in `PRIORS.get("metal-3")` (`webgpu/runtime/device-profile.js:1345`) or race it in `grid-attend-race.js`.

2. **Test AF2 Vector OPM on M2 (Suspect #4)**:
   ```bash
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --bundle=/model-af2-monomer-int5 --length=150 --sequences=256 --tune=opmVectorOutput=false,opmVectorContract=false
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --bundle=/model-af2-monomer-int5 --length=150 --sequences=256
   ```

3. **Test Register-Staged LayerNorm/AdaLN & 4-Row Packing on M2 (Suspects #2 & #3)**:
   - Compare `HEAD` against `976a298d~1` (or toggle the register-staging width cap from `1024` down to `512` in `webgpu/af3/diffusion-conditioning.js`, `webgpu/af3/diffusion-transformer.js`, `webgpu/af3/trunk/transition.js`, and `webgpu/bioemu2/ops.js`):
   ```bash
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/fold-esmfold2.js --tokens=68 --repeat=2
   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-trunk.js --tokens=200 --blocks=8
   ```
   - **If `C = 768` / `1024` register staging is slower on M2**: lower the register-staged cutoff to `channels <= 512` (at `<= 512` channels across 64 lanes, each lane holds at most `2 x vec4<f32>` = 8 floats, which never spills on M2 while still covering `C=128`, `256`, `384`, and `512`!).

### Step 3: Run the Native Metal A/B Benchmarks on the M2

1. **Test `af3_atom_attention_mma` (`keys <= 128`) and `LOCALFOLD_TRI_BLOCKS` (Suspect #6)**:
   ```bash
   # Atom attention MMA vs scalar at 68 tokens
   AF3_SCALAR_ATOM_ATTENTION=1 ./metal/af3/localfold-af3 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --recycles=0 --steps=50 --out=/tmp/a.pdb
   AF3_SCALAR_ATOM_ATTENTION=0 ./metal/af3/localfold-af3 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --recycles=0 --steps=50 --out=/tmp/b.pdb

   # Single-block vs 4-block triangle contraction
   LOCALFOLD_TRI_BLOCKS=4 ./metal/ef2/localfold-ef2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --out=/tmp/a.pdb
   LOCALFOLD_TRI_BLOCKS=1 ./metal/ef2/localfold-ef2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --out=/tmp/b.pdb
   ```
2. **Profile AdaLN kernels on M2 (Suspect #5)**:
   ```bash
   ./metal/ef2/localfold-ef2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --profile
   ./metal/bioemu2/localfold-bioemu2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE --samples=5 --profile
   ```
   - If any `ada_ln` kernel regressed vs `976a298d~1` on M2 due to static register pressure, dispatch the unrolled register variant only when `D().family9` (`[device supportsFamily:MTLGPUFamilyApple9]`) is true.

---

## 5. Further M2-Specific Optimization Opportunities (Once Verified)

Once the M5 commits are verified on M2, the highest-leverage areas specifically for M2 are:
1. **Metal `lf_gemm` (`metal/core/gemm.metal`) epilogue fusions for `Apple8`**:
   - **Already ported & verified (`LOCALFOLD_GEMM_TENSOR=0`)**:
     - `Gemm::yinFloat` (`biasType & 512`), `halfProductScale()` (`1/1024` saturating `half` triangle contraction + `lf_center_norm_16h2`/`32h1`), `gemmGatedRows` (`262144`), `gemmGateResidual` (`16384`), `gemmCondPermuted` (`524288`), `gemmMulH` (`1048576` with 8x8 `simdgroup_load`/`simdgroup_store`), `gemmOpmPermuted` (`4096` with 8x8 strided `simdgroup_store`), `gemmOpmOut` (`8192` with 8x8 `simdgroup_load`/`simdgroup_store`), and fast `om`-only 8x8 stores for `accFloat` `half` outputs (`0de9becb`).
     - Emitted LayerNorm (`EP & 128` in `gemm.metal`) for `C = 64, 128, 256` and `gemmOpmOut`: enabled by default on `lf_gemm` at `C = 64` (where `tr = 64, tc = 64` is already `lf_gemm`'s native `4x4`-fragment square tile, cutting 60 dispatches in `af2` `extra_msa_stack` + `templates` and 36 dispatches in `af3` `templatePairStack`), and kept opt-in via `LOCALFOLD_LN_EMIT=2` at `C = 128, 256` (where widening `tc` to `128`/`256` on a 4-simdgroup `8x8` tile shrinks `tr` to `32`/`16` and rereads `X` 2x–4x across `K`).
     - `precompile()` in `metal/core/core.mm` filters out `gemm_hh*` instances when `tensorWanted()` is true and `gemmt_*` instances when `tensorWanted()` is false, preventing cross-mode spec cache pollution (`-28%` to `-62%` startup compile time) (`557fcc3e`).
     - Eliminated `simdgroup_matrix<float, 8, 8> cfm` register spill in `LF_STORE_8X8` (`metal/core/gemm.metal`) by guarding `biasType & 512` (`yinFloat`) with `if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>)` (`3072x80x768 hhh`: `0.382 ms / 0.99 TFLOP/s` -> `0.165 ms / 2.29 TFLOP/s`), restored vectorized 16-byte `lf_stage8` tile staging for pointer-array batched GEMMs (`a.ptrs`, cutting `352.7 ms` from `bioemu2` sampler GPU time and `90.4 ms` from `bioemu2 6MRR 1x50` under `LOCALFOLD_GEMM_TENSOR=0`), and added `8x8` `simdgroup_load`/`simdgroup_store` fast paths for `gemmGateResidual` (`16384`), `gemmGatedRows` (`262144`), and `gemmCondPermuted` (`524288`) (`-25.6 ms` `af3` diffusion GPU time under `LOCALFOLD_GEMM_TENSOR=0`) (`19d31add`).
     - Fused ESM-C self-attention (`esmc.scores` + `ef2_softmax_seq` + `esmc.context`) into a single `simdgroup_float8x8` FlashAttention kernel (`ef2_esmc_attend`, `-72` dispatches per fold), rewrote `ef2_swa` using `simdgroup_float8x8` MMA with a `9 KB` threadgroup tile (`SWA_KT = 32`) and fused `ef2_gate_live` into its output write (`-69` dispatches per fold, `2.29x` faster SWA at 256aa), and added exact `MAXV = 3, 6, 9` specializations (`C = 384, 768, 1152`) to `lf_layernorm4` in `metal/core/common.metal` so common widths use 25% fewer registers per lane on `Apple8`–`Apple10` (`b1d9d3bc`).
     - Rewrote `be2_window_attention_h` (`metal/bioemu2/bioemu2.metal`) using `simdgroup_half8x8` / `simdgroup_float8x8` MMA + `exp2` log2 scaling (`14.32 ms` -> `9.20 ms` on 6MRR 20-step profile, `1.56x` faster, `-13.8 ms` on 50-step sampler / `-55 ms` on 200-step sampler on both `TENSOR=1` and `TENSOR=0`), fused `af2_ipa_outputs` (`metal/af2/af2.metal`) into a single 256-thread threadgroup per residue with `float4` pair/scalar reductions and hoisted invariant query/point registers in `af2_ipa_weights` (`af2_ipa_outputs` `3.93 ms` -> `2.86 ms`, `af2_ipa_weights` `2.23 ms` -> `1.36 ms` at 256aa), and fused `af3` `msaBlock` (`metal/af3/trunk.mm`) `outerProductMean` + `msaAttention` input norms via `layerNormDual` and emitted `msaTransition.inputLayerNorm` from `msaAttention.outputProjection` (`Cm = 64`, active on both `TENSOR=1` and `TENSOR=0`, `-8` LayerNorm dispatches per trunk pass) (`0d73a36c`).
     - Optimized both FlashAttention kernels (`lf_attention` in `metal/core/common.metal` for M1/M2 `Apple8` and `D = 24`, and `lf_attention_tensor3` in `metal/core/gemm_tensor.metal` for `Apple10`): added `if (qa >= n) return;` early simdgroup exit and in-register `qBias` addition (eliminating `12–16 KB` `Qs` threadgroup staging and `simdgroup_barrier`) in `lf_attention_tensor3`; added `NT` (`biasPadInf`) specialization (`lf_attention_nt_*`), shared `Qs`/`Ks`/`Vs` threadgroup memory (`-33%` threadgroup memory at `D = 32`), 128-thread parallel `Ks`/`Vs` staging (`D <= 32`), `fast::exp2`, and `!sgActive` early simdgroup skip in `lf_attention`; and enabled `-INFINITY` key-tile bias padding (`biasPadInf = true`) for `qBias` token attentions in `af3`, `ef2`, and `bioemu2` (`af2` 325aa non-tensor attention `937.5 ms` -> `838.9 ms`, `-98.6 ms`; `af3` 68-token diffusion attention `80.3 ms` -> `62.9 ms` on `TENSOR=1` and `86.3 ms` -> `62.8 ms` on `TENSOR=0`; `ef2` 325aa diffusion attention `37.8 ms` -> `31.3 ms` on `TENSOR=1` and `77.4 ms` -> `67.2 ms` on `TENSOR=0`).
2. **WebGPU `grid-attend-race.js` calibration on M2**:
   - Check `node tools/gpu-chrome.mjs tools/gpu/probe-grid-attend-race.js` on M2 to verify that the 4-way startup race consistently selects M2's winning arms (`gridAttendTiled=false`, `gridProjectArithmetic="f32"`, `attentionGroup=1`, and the fastest `triangleContractArithmetic`).

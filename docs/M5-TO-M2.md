# M5 to M2: checking the M5's Metal and WebGPU pass on an Apple M2 (`opus-55-opt`)

Written on the Apple M5 (`Apple10`, macOS 26) for whoever next works on the **Apple M2** (`Apple8`, 10-core GPU,
16 GB). It covers the M5 pass `976a298d..bed4fb7d` (2026-10-21..23); the per-commit detail is in `git log`, not here.
docs/M2-HANDOFF.md is the A100's earlier round-six handoff, a different pass.

**Corrected since it was written** (the fixes after `bed4fb7d`; do not test the old behaviour):
- `gemmGateResidual` (epilogue 16384) sums its product in a **float** tile again; the half tile is
  `LOCALFOLD_GATE_HALF_ACC=1` (e17ab007: 4.2e-8 against 5.9e-5 of the residual, ~1% of Boltz-2's diffusion).
- AF2's outer product mean merges its left and right projections into one GEMM **only up to 128 rows**; deeper
  alignments take two (a62840d3 - merged, every alignment past 128 rows read l and r interleaved, pLDDT 96.2 -> 47.6).
- The row-grouped vector epilogue (`vecEp`) is **not** taken by 128-row tiles (e17ab007: 128 x 32 was wrong).
- `AF3_SCALAR_ATOM_ATTENTION=0` now means off; it used to turn the scalar control ON whatever its value, so the A/B
  below was the same arm twice.
- ESMFold2 past ~620 residues (6a0bb428) and two-token AF3-lineage folds (3bd04bb6) were broken and are fixed.
- `lf_gemm` (the M2's GEMM) emits the next LayerNorm only at 64 output columns, `lf_gemm_dual` up to 128; the 128/256
  emission mode on `lf_gemm` lost, nothing took it, and it is gone with its `LOCALFOLD_LN_EMIT` switch (89e5e033).
- Measured since on the M5 and kept: the `plain` GEMM rule from 976a298d (reverted, six pairs: every case within
  1.1%), and the `metal-3` WebGPU knobs, each now with its M5 numbers in `webgpu/runtime/device-profile.js` (52e5b69f).

---

## 1. Why the M2 needs its own pass

Everything here was written and measured on the M5. Three differences can turn an M5 win into an M2 loss or break:

1. **The M2 runs different GEMM kernels.** The M5 takes `lf_gemm_tensor` (`metal/core/gemm_tensor.metal`, Metal 4
   tensor ops, `Apple10` and macOS 26 only). The M2 takes `lf_gemm` / `lf_gemm_dual` (`metal/core/gemm.metal`,
   `simdgroup_matrix`). The pass changed `gemm.metal` heavily, and **those kernels have only ever run on the M5 under
   `LOCALFOLD_GEMM_TENSOR=0`**: the same source, but the M5's compiler, register file and core count. See section 3A.
2. **Register pressure.** The M5 (Apple9+) allocates registers dynamically; the M2 (`Apple8`) does not, so kernels that
   hold a whole row in registers (the unrolled `float4` AdaLN / LayerNorm forms) can lose occupancy or spill there.
3. **WebGPU cannot tell the two apart.** Both report `apple / metal-3` with an empty device string under stock Chrome,
   so every `PRIORS.get("metal-3")` value in `webgpu/runtime/device-profile.js` ships to both. A knob the two chips
   disagree on needs a threshold that wins on both, or a startup race like `webgpu/af3/trunk/grid-attend-race.js`.

---

## 2. Rules and setup

- `AGENTS.md` / `CLAUDE.md` / `metal/README.md` hold the invariants: never widen a tolerance or rewrite a baseline to
  pass a gate; one GPU job at a time; speed arms interleaved (ABBA) on warm caches; never commit `package-lock.json` or
  `localfold-fixes-for-sergey.md`.
- **The first run of a freshly built Metal binary compiles its GEMM instances inside the timed stages** (up to 4x
  long). Run every case once, untimed, before timing it.
- In zsh, `env $e` with `e="A=1 B=2"` sets ONE variable `A` to `1 B=2`. Write each assignment out.
- A Metal knob the chips disagree on: branch on the device (`tensorWanted()` for Apple10; for Apple9 use
  `supportsFamily` with the enum **by value** - the M2's SDK 13.3 does not name it; see the note at `core.mm:40`).

```bash
export PATH="$HOME/sergey/.venv-localfold/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
git checkout opus-55-opt && git pull --rebase origin opus-55-opt
bash metal/build.sh selftest bench af3 af2 ef2 bioemu2
```

---

## 3. What to suspect on the M2

### A. `gemm.metal` code written on the M5 (runs on the M2, never run on one)

| Commit | What |
|---|---|
| `887f8d1d` | `yinFloat` (a float residual into a half-output GEMM) in `lf_gemm` |
| `0de9becb`, `19d31add` | fused epilogues, `halfProductScale`, float-accumulated f16 stores in `lf_gemm`; a register spill fix |
| `557fcc3e` | `lf_gemm` emitting the next LayerNorm at 64 columns; the 64-wide dual GEMM |
| `57d4b618`, `0847a49b` | vectorised bias/scale epilogue; blocks-of-2 `gemmCouplingCombine` |
| `5646656b` | `gemmCouplingCombine` ported to Apple8 |
| `84e68ca6` | `lf_gemm_dual` (two GEMMs sharing an A tile), emitting LayerNorm up to 128 |
| `b15f7b6d`, `c2791e57` | `gemmGatedAddDual` at C = 256, then 384/512 |

The M5 check was `LOCALFOLD_GEMM_TENSOR=0`: the selftest passes, and AF2 6/6, ESMFold2 8/8 and AF3 18/18 gates match
before and after the cleanup. Note that an M5 under `TENSOR=0` gives AF3's kitchen-sink different digits from the
tensor baseline (arithmetic, expected) - **the M2's own baselines are the reference, not the M5's.**

### B. Other Metal kernels (run on both chips)

- Unrolled `float4` register staging in AdaLN: `af3_ada_ln*`, `be2_ada_ln*`, `ef2_ada_ln` (`0c3dd2dd`, `70248dce`,
  `dedf5ec0`, `d8620d9b`) - check `--profile` for a regression against `9b61f8b4`.
- `simdgroup_float8x8` FlashAttention for ESM-C and sliding-window attention (`b1d9d3bc`, `0d73a36c`, `376c5a1d`).
- `af3_atom_attention_mma` at keys <= 128 (`41555140`; `AF3_SCALAR_ATOM_ATTENTION=1` is the scalar control) and the
  blocked triangle contraction (`LOCALFOLD_TRI_BLOCKS=1` the control).
- The XOR-swizzled `lf_center_norm` family and `lf_center_norm_wide_h` (`dd66ce1a`, `a8c6006c`).

### C. WebGPU `metal-3` values this pass set (ship to the M2 unmeasured there)

- `atomRowTile` pinned to 4 at every size (`ed330437`). M5: best or level at 68 and 255 tokens (numbers at the value).
- `opmMatrixContract: false` - matters only where `chromium-experimental-subgroup-matrix` is on, not under stock flags.
- `diffusionTokenTile.outCrossover: 96`.
- Register-staged full-row LayerNorm / AdaLN at C = 768-1024 (`ed330437`, `f285d862`, `35dfa517`) and the 4-row x
  64-lane atom packing (`ed330437`, `f285d862`).

The other `metal-3` values (`trianglePairProjectTile`, the f16 half tiles, `singleTransitionTile`,
`singleProjectRows`) predate this pass; their comments say "Not measured on an M2".

---

## 4. The M2 checklist

### Step 0: the build is the first check

`bash metal/build.sh ...` above, on the M2's SDK 13.3. The host code must not name a Metal enum newer than macOS 13
(the M5's SDK 26 hides that). A failure here is a portability fix, not a reason to raise the floor.

### Step 1: correctness

```bash
./metal/selftest/localfold-selftest          # the M2's GEMM is the shipped path here; nothing to force
python3 metal/af2/gate.py
python3 metal/ef2/gate.py
python3 metal/af3/gate.py
python3 tools/check-native-worker.py --offline --no-page   # add --no-af3 without AlphaFold 3's weights
```

- `af2 5caj-msa` and `af3-5caj-msa` have no M2 baseline, so they print NEW and cannot fail. If everything else passes,
  record them: `python3 metal/af2/gate.py --write --only=5caj-msa` and the AF3 one below.
- `af3-kitchen-sink` and `af3-glycan` are expected to read MOVED: cb160ba0 (AF3 drops a glycan's O1, exact against
  AF3's own batch) moved them on the M5 at that commit alone (kitchen sink pLDDT 70.61 -> 77.08, glycan 48.62 ->
  48.25). If those are the only moves: `python3 metal/af3/gate.py --write --only=af3-kitchen-sink,af3-glycan,af3-5caj-msa`.
- Any other MOVED or FAIL is a real finding against section 3A or 3B.

```bash
node tools/gpu-chrome.mjs tools/gpu/check-quantised-upload.js
node tools/gpu-chrome.mjs tools/gpu/check-evoformer-stack.js
npm run test:stock            # every bundle on this box folds under a stock Chrome
```

### Step 2: is the M2 faster than before the pass?

The question the M5 answered with -12..-29% Metal compute (and WebGPU -3..-26%). Build `9b61f8b4` beside HEAD and
alternate them, after one untimed run of each:

```bash
git worktree add ../lf-base 9b61f8b4 && (cd ../lf-base && bash metal/build.sh af2 ef2 af3 bioemu2)
SEQ=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE
for r in 1 2 3; do for root in ../lf-base .; do
  (cd $root && ./metal/ef2/localfold-ef2 --sequence=$SEQ --out=/tmp/x.pdb | grep -E "^(trunk|sampler) ")
done; done
```

The same for `af2` (pass times), `af3 --model=boltz2` (`total`) and `bioemu2
--job=tools/fixtures/bioemu2/6mrr-68.json --samples=2`, at 68 residues and at 5CAJ's 255. A port slower than
`9b61f8b4` is bisected through section 3A/3B's commits, with `LOCALFOLD_GEMM_TENSOR` irrelevant here.

### Step 3: the WebGPU `metal-3` values (stock flags, interleaved)

```bash
# atomRowTile: off = the derived rule; 4 = the pin. Atom output passes in gpuPasses (output-*).
for t in off 4 off 4; do LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-head.js \
  --model=/model-af3-int5/manifest.json --tokens=68 --calls=6 --profile --atomtile=$t; done   # and --tokens=255
# AF2's vector OPM
for v in false true false true; do LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js \
  --bundle=/model-af2-monomer-int5 --length=150 --sequences=256 --tune=opmVectorOutput=$v,opmVectorContract=$v; done
```

Where the M2 disagrees with the M5, do not just flip the prior (the M5 then loses): find a threshold that wins on
both, or race the two forms at startup as `grid-attend-race.js` does.

### Step 4: Metal kernels in 3B

```bash
for v in 1 0 1 0; do AF3_SCALAR_ATOM_ATTENTION=$v ./metal/af3/localfold-af3 --sequence=$SEQ --model=boltz2 --out=/tmp/x.pdb | grep total; done
for v in 1 4 1 4; do LOCALFOLD_TRI_BLOCKS=$v ./metal/ef2/localfold-ef2 --sequence=$SEQ --out=/tmp/x.pdb | grep trunk; done
./metal/ef2/localfold-ef2 --sequence=$SEQ --out=/tmp/x.pdb --profile        # the ada_ln rows, against ../lf-base
./metal/bioemu2/localfold-bioemu2 --sequence=$SEQ --samples=2 --out=/tmp/x.pdb --profile
```

An `ada_ln` kernel slower than `9b61f8b4`'s on the M2 takes its unrolled form only on Apple9+.

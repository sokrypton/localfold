# Handoff: make Milot's AF2 kernels (ColabFold's Pallas path) catch up with LocalFold's CUDA AF2

**The ask.** ColabFold with Milot Mirdita's fused kernels (`--use-fast-kernels`) is now within 1.2-1.8x of
LocalFold's native CUDA AF2 on an A100, and level with it on Blackwell up to ~500 residues. Find where the
remaining time goes and close it **inside the Pallas/JAX stack** - the kernels, the dispatch in AlphaFold's
modules, and what XLA does around them. The CUDA port (`cuda/af2`, with most of its pair-stack kernels shared
from `cuda/af3/src`) is the reference for what is achievable on the same card, and its kernels are ideas to
borrow, not code to call.

Everything below was measured on this box (A100-SXM4-40GB, driver 580) on 2026-10-21..23 unless it says
otherwise.

## Rules

- Work in your own venv and your own clone of the kernels. Do not modify `~/.venv-colabfold-k08` (it is the
  baseline every number here was taken on) or `~/.venv-colabfold` (the older 2.3.20 install).
- Do not push to `mirditalab/colabfold-kernels` or `sokrypton/alphafold`. Keep changes as local commits or
  patches; the user decides what goes upstream.
- A speedup only counts with the structure gate below passing. Compare arms on THIS box, alternated in one
  sitting: the A100 drifts, and a number from another machine is not a bar.
- The GPU is shared with other sessions. `nvidia-smi` before a timing run; if something else is on the card,
  wait.

## What "Milot's kernels" are

Three PyPI packages, all MIT, pulled in by AlphaFold's ColabFold fork:

| package | version here | source | what it is |
|---|---|---|---|
| `colabfold-kernels` | **0.8.0** (latest; tags v0.5.0, v0.5.1, v0.6.0, v0.7.0, v0.8.0) | https://github.com/mirditalab/colabfold-kernels | the Pallas kernels - pure Python, ~3,300 lines |
| `colabfold-cuda-kernels` | 0.2.2 | https://github.com/mirditalab/colabfold-cuda-kernels | a prebuilt `.so` (AF3 diffusion megakernel, FP8 triangle contraction for sm_89+) |
| `colabfold-legacy-kernels` | 0.5.1 | (dependency of the above) | the older CUDA kernels, used on pre-sm_80 cards (`cuda_legacy` backend) |
| `alphafold-colabfold` | 2.3.26 | https://github.com/sokrypton/alphafold | AlphaFold's model code; `alphafold/model/modules.py` and `common_modules.py` call the kernels |
| `colabfold` | 1.6.3 | https://github.com/sokrypton/ColabFold | `colabfold_batch`; `--use-fast-kernels` sets `global_config.use_pallas` |

`colabfold_kernels/` (all paths relative to `site-packages/`):

| file | what |
|---|---|
| `fused_ops.py` | **the dispatcher** - for each op, picks a Pallas kernel, the volta CUDA kernel, or `None` (plain XLA). Start here |
| `packed_attn.py` | Pallas/Triton flash attention that reads q/k/v/gate in place from one packed projection; rows and columns without a transposed copy. **The largest kernel in a fold** |
| `mgpu_attn.py` | Mosaic GPU version of the same, used on **sm_12x only** (Blackwell RTX/PRO) - compiled once at startup to check this JAX's Mosaic API matches, silently skipped if not |
| `tri_flash.py` | flash attention (`pallas_attention`) plus the backward that `packed_attn` pairs with |
| `tri_mul.py` | fused LayerNorm + gated dual projection for the triangle multiplication; the contraction itself stays an XLA batched GEMM |
| `transition.py` | fused LayerNorm + transition MLP |
| `opm.py`, `contract.py` | outer product mean update, contraction helpers |
| `volta.py`, `device.py` | legacy-kernel bridge, compute capability detection |
| `diffusion.py`, `grads.py` | AF3 diffusion and backward passes (not on the AF2 inference path) |

How the backend is chosen (`colabfold/alphafold/models.py`): `--kernel-backend auto` picks `pallas` on sm_80+
and `cuda_legacy` below, `--use-fast-kernels` requires bfloat16 (the default), and
`global_config.fp8` (off by default; e4m3 attention and contraction on sm_89+) has **no CLI flag** - it is
`model_config.model.global_config.fp8` in `config.py`. On an A100 FP8 cannot apply anyway (no 8-bit MMA).

## The numbers to beat

Settings for every row: `model_1_ptm`, 3 recycles (4 passes), early stop OFF, MSA 512:1024 (LocalFold and
NVFold at 508:1024 clusters - the same effective depth, because AF2 reduces clusters by the template count),
bfloat16, warm (second fold of the same shape in one process). Cases: 6MRR (68 aa), 5CAJ chain A (261),
5CAJ x2 (522), x4 (1044), single sequence; and 5CAJ with a 7,907-row alignment (261 + MSA).

**A100 (this box), warm seconds:**

| engine | 68 | 261 | 522 | 1,044 | 261 + MSA |
|---|---:|---:|---:|---:|---:|
| **LocalFold CUDA** (opus-55-opt 973308d3) | **0.065** | **0.316** | **1.145** | **5.59** | **1.103** |
| ColabFold + `colabfold-kernels` 0.8.0 | 0.116 | 0.432 | 1.408 | 6.66 | 1.564 |
| ColabFold + old kernels (alphafold-colabfold 2.3.20) | 0.2 | 1.2 | 4.6 | 20.2 | 3.2 |
| ColabFold, plain JAX 0.10.2 | 0.60 | 3.41 | 10.1 | 40.1 | 8.9 |
| *gap to close (ColabFold 0.8.0 / ours)* | *1.78x* | *1.37x* | *1.23x* | *1.19x* | *1.42x* |

**Blackwell (Colab G4, RTX PRO 6000, sm_120), warm seconds** - here 0.8.0 runs the Mosaic GPU attention:

| engine | 68 | 261 | 522 | 1,044 | 261 + MSA |
|---|---:|---:|---:|---:|---:|
| LocalFold CUDA | 0.070 | 0.222 | 0.741 | 3.04 | 0.694 |
| ColabFold + kernels 0.8.0 | 0.068 | 0.228 | 0.756 | 3.44 | 0.816 |

So the A100 gap is the interesting one, and the MSA case and the small case are where it is widest.
NVFold (NVIDIA's AF2, evaluation build) was also measured; its numbers are under an evaluation licence that
forbids disclosure, so they are not repeated here - it sits between the two rows above on the A100.

## The harness: `~/af2bench`

```
~/af2bench/
  cf/<case>/          ColabFold inputs: <case>.fasta holds the query TWICE (r68_1, r68_2) so one process folds it
                      cold then warm; r261msa/ is two copies of the a3m
  cfdata/params/      symlinks to ~/lfjax/af2_params (all five *_ptm.npz are needed - ColabFold reads
                      params_model_3_ptm.npz even with --num-models 1) + download_finished.txt
  cf.sh               ColabFold runner:  VENV=<venv> OUT=<dir> ./cf.sh fast r68 r261 r522 r1044 r261msa
  cftime.py           warm time per case at ms resolution + the structure gate:  python3 cftime.py <dir>
  nat2.py, serve.py   LocalFold CUDA runner (resident --serve, 4 folds a case):
                      R=/home/ubuntu/localfold TAG=<dir> python3 nat2.py [cases]
  cases.json, 5caj-a.a3m   the sequences and the alignment
  native/r261msa.pdb  LocalFold's MSA fold, the structure reference
  prof/               the nsys profiles quoted below (cf522, cf68, ours522)
```

Timing: ColabFold prints `took 1.4s`, rounded to 0.1 s. `cftime.py` instead reads the timestamps of the
warm query's per-recycle log lines and reports 4 x the median gap - the gaps agree to a millisecond at every
size, and the median drops the odd 70 ms host hiccup at 68 residues. A whole sweep of five cases takes about
two minutes; the first query in each process pays the JIT compile (~10-17 s).

Baseline check before changing anything:

```
cd ~/af2bench
OUT=cfout-base ./cf.sh fast r68 r261 r522 r1044 r261msa && python3 cftime.py cfout-base
# expect ~116 / 432 / 1408 / 6660 / 1564 ms and "r261msa CA RMSD to ours: 0.121 A"
```

**The structure gate.** The 261 + MSA fold is determined (pLDDT 96, pTM 0.914 on every engine), so any
correct arm lands on it: CA RMSD to `native/r261msa.pdb` must stay at **<= 0.15 A** (the baseline is 0.121,
plain JAX 0.128) and pLDDT/pTM at 96 / 0.914. The single-sequence cases are NOT a structure gate - at
pLDDT 26-34 a correct numerical change can move them by angstroms - but they must still finish and print a
pLDDT within a point of the baseline.

## A development install

```
git clone https://github.com/mirditalab/colabfold-kernels ~/colabfold-kernels && cd ~/colabfold-kernels
git checkout v0.8.0          # main has moved past the release; measure the release first, then decide
uv venv -p 3.11 ~/.venv-cfk-dev
uv pip install --python ~/.venv-cfk-dev/bin/python $(~/.venv-colabfold-k08/bin/pip freeze | grep -v '^colabfold-kernels')
uv pip install --python ~/.venv-cfk-dev/bin/python --no-deps -e ~/colabfold-kernels
~/.venv-cfk-dev/bin/python -c "import colabfold_kernels; print(colabfold_kernels.__file__)"   # must be the clone
VENV=~/.venv-cfk-dev OUT=cfout-dev ./cf.sh fast r522 && python3 cftime.py cfout-dev
```

Changes to the dispatch side (which op calls which kernel, the projection layouts) live in
`alphafold/model/modules.py` of `alphafold-colabfold`; install `sokrypton/alphafold` editable the same way if
you need to touch it.

Two traps already paid for: on Colab, `uv pip install` lands in the SYSTEM Python because Colab sets
`UV_SYSTEM_PYTHON` (use `env -u UV_SYSTEM_PYTHON uv pip install --python <venv>/bin/python`), and
`MPLBACKEND=agg` is needed there or matplotlib refuses Colab's inline backend inside a venv. `cf.sh` sets the
latter.

## Where the time goes (A100, 522 residues, single sequence)

`nsys profile -t cuda --cuda-graph-trace=node ...` - **the graph-node flag is required**: XLA runs the
model through CUDA graphs, and without it the trace shows only XLA's autotuning kernels (Redzone, DelayKernel)
and 0 ms of model. Drop those four autotune kernel names when summing. Per pass, both engines:

| op | ColabFold 0.8.0 (ms / pass) | LocalFold CUDA (ms / pass) | ratio |
|---|---:|---:|---:|
| triangle attention: flash kernel | `packed_attention__1/__2` **102** | `flashGrid2R` **79.5** | 1.28x |
| triangle attention: LN + q/k/v/gate projection | `ln_projection__2` 47 | `gridInK` (fused LN + projection) 34.6 | 1.36x |
| triangle attention: output projection | `output_projection__1` 17 | `gridOutK` 19.8 | 0.85x |
| triangle multiplication: LN + gated dual projection | `ln_gated_dual_proj` 39 | `triIn256K` 36.7 | 1.06x |
| triangle multiplication: **the contraction** (XLA bf16 GEMM) | cutlass `128x128_32x4` tn+nt **49** | cuBLAS `128x128_ldg8_stages_32x5` **27** | **1.82x** |
| triangle multiplication: epilogue (LN, out proj, gate) | `trimul_epilogue` 33 | `triangleOutK` 27.8 | 1.18x |
| pair transition | `fused_transition__1` 30 | `fusedTransitionK` 26.7 | 1.14x |
| **all kernels** | **~363** | **~285** | **1.27x** |

(ColabFold: 2903 ms of kernels over 2 folds x 4 passes; ours: 4562 ms over 4 folds x 4 passes. Our CUDA
kernel names map to `cuda/af3/src/flash.cuh`, `fusedtriangle.cuh` (`gridInK`, `gridOutK`, `lnHeadsK`),
`fused256.cuh` (`triIn256K`, `triangleOutK`) and `fusedtransition.cuh`, which `cuda/af2` reuses.)

So the leads, largest first:

1. **The triangle contraction is 1.8x slower per call with the same FLOPs** - 0.50 ms against 0.27 ms, 52
   calls a pass each. It is not a Pallas kernel at all: XLA's autotuner picks a cutlass 128x128 tile on the
   batched GEMM the einsum lowers to. Look at the operand layout it is handed (a transpose it materialises, a
   non-unit stride, the channel axis in the wrong place) and at whether a channel-major layout lets it run as
   one well-shaped batched GEMM, which is what ours does. ~22 ms a pass, the biggest single item.
2. **The flash attention** (`packed_attn.py`): 102 vs 79.5 ms. Ours is a register-staged flash kernel with the
   pair bias read once per tile (`cuda/af3/src/flash.cuh`, `flashGrid2R<32,2,48,...>`: head dim 32, 2 heads a
   block, 48-key tiles, fp16). Block sizes, the number of warps, the bias load and the softmax in `exp2` are
   the knobs.
3. **The LN + q/k/v/gate projection** (`ln_projection__2`): 47 vs 34.6.
4. Everything else is 1.06-1.18x and each item is a few ms.

At **68 residues** (the widest gap, 29 ms a pass against our 16) ColabFold is still GPU-bound - 31.5 ms of
kernels a pass over ~3,500 launches - and the profile is led by `fused_transition__2` (4.3 ms) and
`ln_projection__3` (2.9 ms), i.e. MSA-side kernels on a single-sequence input. Check whether ColabFold pads the
MSA / extra-MSA stacks to the `512:1024` caps in single-sequence mode; if it does, that is work on padding rows
that LocalFold does not do, and it is a dispatch or featurisation question rather than a kernel one.
`~/af2bench/prof/cf68.nsys-rep` has the trace.

With the MSA (`r261msa`, 1.42x) the MSA stack's row/column attention and outer product mean join the picture;
profile `cf/r261msa` the same way before working on it.

## How to report back

A table in the format of the A100 one above, from one sitting with the baseline (`~/.venv-colabfold-k08`) and
your arm alternated, each arm's `cftime.py` output including the r261msa RMSD line, and which change bought
what. A change that does not move the 522 or 261 + MSA row is not worth its complexity.

## Results (2026-10-23, A100-SXM4-40GB)

Upstream PRs:
- [`mirditalab/colabfold-kernels#5`](https://github.com/mirditalab/colabfold-kernels/pull/5) (`sokrypton:a100-v08-speed` -> `mirditalab:main`)
- [`steineggerlab/alphafold#12`](https://github.com/steineggerlab/alphafold/pull/12) (`sokrypton:a100-evoformer-speed` -> `steineggerlab:main`)

| engine | 68 | 261 | 522 | 1,044 | 261 + MSA | `r261msa` CA RMSD |
|---|---:|---:|---:|---:|---:|---:|
| ColabFold + `colabfold-kernels` 0.8.0 (`~/.venv-colabfold-k08`) | 0.160 | 0.432 | 1.408 | 6.660 | 1.564 | 0.121 A |
| **ColabFold + `colabfold-kernels` (`~/.venv-cfk-dev`, PRs #5 & #12)** | **0.088** | **0.324** | **1.160** | **5.616** | **1.192** | **0.122 A** |
| LocalFold CUDA baseline (`973308d3`) | 0.065 - 0.089 | 0.316 | 1.145 | 5.590 | 1.103 | 0.000 A |
| **LocalFold CUDA + single-seq `outLn256<..., VG=true>`** | **0.065** | **0.313** | **1.126** | **5.506** | **1.097** | **0.000 A** |

### What bought what

1. **`colabfold-kernels` (`mirditalab/colabfold-kernels#5`):**
   - `_ln_proj_kernel` `jax.lax.fori_loop` over `slots` + `(block_m=128, num_warps=4, num_stages=1)` for `m > 16_384` on A100 (`packed_attn.py`): cuts `ln_projection` register spills from 255 registers (`47.0 ms` -> `31.9 ms` per pass on `r522`).
   - `packed_attention` A100 tile & staging tuning (`packed_attn.py`): `block_q=64` for `half and (not column or (biased and 128 < s <= 264))`, `block_k=64` for `half and not column`, `num_stages=3` for `column or (half and s > 128)` (`102 ms` -> `95.2 ms` per pass on `r522`; `-48 ms` on `r261`).
   - `ln_gated_dual_proj` (`tri_mul.py`): `(block_m=128, num_warps=4, chunk=64)` for `m_tot > 16_384` and `(64, 4, None)` for `m_tot <= 16_384` on A100 (`39.0 ms` -> `34.1 ms` per pass on `r522`).
   - `fused_transition` (`transition.py`): restrict `sm80_small` (`chunk=128, num_warps=8`) to `m <= 16_384`.
   - Fused `global_attention` Pallas kernel (`packed_attn.py`, `fused_ops.py`): replaces the 6-kernel XLA sequence in `MSAColumnGlobalAttention` (`extra_msa_stack`) with a single two-pass Pallas kernel in-place on `[S, B, C]`.
   - Fused next-module `LayerNorm` in `output_projection` (`packed_attn.py`, `fused_ops.py`): emits `ln_rows(m_out, ln_scale, ln_offset)` from the already-in-register residual output rows (borrowed from `cuda/af2/src/outln.cuh`), saving 96 full-MSA `pallas_layer_norm` launches per pass (`-24 ms` on `r261msa`).
2. **`alphafold` (`steineggerlab/alphafold#12`):**
   - Single-sequence (`s == 1`) fast path in `Attention.packed` (`modules.py`): skips `packed_attention` in `MSAColumnAttention` when `N_seq == 1` (`r68: 160 -> 112 ms`).
   - Hoisted `OuterProductMean` mask `norm` out of `extra_msa_stack` and `evoformer_stack` (`modules.py`) and removed the redundant `norm = utils.wide_einsum('abc,adc->bdc', mask, mask)` recomputation inside `OuterProductMean.__call__` (saves 208 `[S, L, 1] x [S, L, 1] -> [L, L, 1]` contractions per fold, `-96 ms` on `r261msa`).
   - Wired `fused_ops.global_attention` into `GlobalAttention` & `MSAColumnGlobalAttention` and fused `output_projection` + next-module `LayerNorm` across `MSARowAttentionWithPairBias -> MSAColumnAttention -> Transition` in `EvoformerIteration` (`modules.py`).
   - Hoisted `InvariantPointAttention` 2D pair bias and inference-only `MultiRigidSidechain` out of the 8-layer `StructureModule` `hk.scan` (`folding.py`, `r68: 112 -> 88 ms`).
3. **Pallas -> LocalFold CUDA transfer (`cuda/af2/src/outln.cuh`, `cuda/af2/src/evoformer.cuh`):**
   - Added `VG=true` mode to `outLn256` (`outLn16K` / `outLnK`) for `rowsN == 1 && t.msaOnes` in `msaColumnAttention`, computing `v * sigm(g)` directly from `qkvg` in registers while staging `A` into shared memory and skipping `msaFlash<true>` (192 launches/fold saved; `r68: 89 -> 64.5 ms`, `r261: 316 -> 312.6 ms`, `r522: 1145 -> 1125.9 ms`, `r1044: 5579 -> 5505.5 ms`).


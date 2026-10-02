# AlphaFold 3 in CUDA

A native CUDA/cuBLAS AlphaFold 3, transcribed stage by stage from this repository's CPU
references under `src/af3/` (the specification) and checked against af3-any-model's own
oracle dumps in `oracle-dumps/`. Proteins fold end to end from a sequence (and optionally an
A3M), with pLDDT, PAE, PDE and pTM.

## Build and run

```
cd native/af3
nvcc -O3 -std=c++17 -arch=sm_80 --default-stream per-thread --use_fast_math src/af3.cu -lcublas -lcupti -o af3

python3 ../../tools/serve.py 8791 &        # the exporter reads the bundle over HTTP
# a sequence (chains joined by ":") and optionally an alignment, featurised by the repo's own
# af3BatchFromA3m; or, with no --sequence, AF3's own 6MRR batch plus every oracle to check against
node --js-float16array --max-old-space-size=24000 export-model.mjs data-5caj \
  --sequence=<SEQ> --a3m=../../oracle-dumps/5caj-a.a3m
node --js-float16array --max-old-space-size=24000 export-model.mjs data          # oracle checks

./af3 data                                   # f32 path, every stage against AF3
./af3 data-5caj --fold --fast --out=5caj.pdb # fold: PDB with pLDDT in the B-factor column
python3 score.py 5caj.pdb ../../tools/fixtures/5caj-crystal.pdb A
```

`--fold` options: `--steps=200 --recycles=3 --seed=42 --folds=N` (N warm repeats), `--samples=N`
(N diffusion samples off one trunk, AF3 runs five: each scored by the confidence head and ranked by
0.8 ipTM + 0.2 pTM - pTM for one chain; AF3's disorder and clash terms are not computed - the best
written to `--out`, all to `<out>_sample<k>.pdb`). The samples run as ONE batch through the
denoiser - the transformer's GEMMs at 5x the rows, the samples as the flash kernel's batch, what
they share (conditioning, masks, biases) read once - and sample k draws exactly what a one-sample
run seeded `seed + k` draws, so its structure is the same: 5CAJ's five read 1.989 / 2.023 / 1.961 /
1.996 / 2.018 A batched and one at a time. Five samples' diffusion: **1.96 s against 3.61 s** in
sequence (one sample 0.69 s),
`--fast` (f16 trunk and denoiser transformer), `--stages` (per-stage profile),
`--no-graphs`. `pairformer.cu` is the earlier one-file pairformer prototype and benchmark.

## Accuracy against AF3 (6MRR, f32 bundle)

| seam | f32 path | f16 path |
|---|---|---|
| target_feat (atom encoder) | 5.4e-8 | |
| z_after_msa | 1.0e-4 | 1.2e-4 |
| trunk_out_pair | **3.0e-5** | 3.0e-4 |
| single | 7.2e-6 | 1.3e-4 |
| one denoiser call | 1.8e-5 | 1.8e-3 |
| confidence PAE / PDE | 2.8e-6 / 3.7e-6 | |

WebGPU (docs/AF3.md, f32): trunk_out_pair 2.98e-4. `z_init` reads 2.2e-4 because AF3's dump
stores that tap in bfloat16. 🔴 pLDDT on the confidence oracle's RANDOM inputs is
ill-conditioned - a 1e-6 change to the input moves it 7.2e-3 - so it reads ~6e-3 there (the
JS CPU reference 8.8e-3) and is checked on folds. 🔴 Likewise a random N(0,1) block input reads
1-3e-2 in any 16-bit format where AF3's real input reads 3e-5: always check on real inputs.

Folds: 6MRR from its sequence **0.683 A** CA RMSD, pLDDT 85.1, pTM 0.720 (WebGPU 0.65-0.71);
5CAJ (255 residues of chain A) with its MSA **2.04 A**, pLDDT 94.6, pTM 0.940 - the WebGPU
port gives pLDDT 94.5, pTM 0.938 on the same inputs.

## Speed: A100-SXM4-40GB, 200 diffusion steps, `--fast`

| target | WebGPU first / warm (0 recycles) | native first / warm, 0 recycles | native warm, 3 recycles |
|---|---|---|---|
| 6MRR, 68 tokens | 5.1 s / 2.4 s | 0.49 s / **0.41 s** | 0.48 s |
| 5CAJ, 261 tokens, 512 MSA rows | 11.4 s / 7.4 s | 0.85 s / **0.73 s** | 1.14 s |
| 5CAJ x 2, 522 tokens, 512 rows | | 1.50 s / 1.41 s | 2.95 s |
| 5CAJ x 4, 1044 tokens, no MSA | | 4.17 s / 4.11 s | 11.4 s |

WebGPU with the developer flags (`fold.js --folds=2`); a stock-Chrome NVIDIA visitor gets
about half its speed. A whole process - mapping model.bin, the CUDA context, target_feat, the
fold - is 1.7 s for 6MRR and 2.4 s for 5CAJ. No 2 GiB binding ceiling. At 1044 tokens a trunk pass is 2.4 s - grid
attention's flash kernel 40% of it - so recycles dominate there; up to ~300 tokens the 200
denoiser steps do.

Against AlphaFold 3 itself - af3-any-model's JAX (bf16, Triton flash attention) with DeepMind's
weights, on this A100, `tools/oracle/bench_af3_native.py` at matched settings (one sample unless
said, no token bucketing, the same MSA rows), steady-state calls:

| 200 steps | JAX AF3 | native `--fast` | |
|---|---|---|---|
| 6MRR, 68 tokens, 1 pass | 1.67 s | **0.41 s** | 4.1x |
| 5CAJ, 261 tokens, 512 rows, 1 pass | 2.76 s | **0.73 s** | 3.8x |
| 5CAJ, 4 passes (3 recycles) | 3.47 s | **1.13 s** | 3.1x |
| 5CAJ, 1 pass, 5 samples | 5.00 s | **1.57 s** | 3.2x |

and JAX's first call carries ~60 s of compilation where the native first fold is within 15% of
a warm one.

The pairformer alone (48 blocks, FP16): 64 tokens 15 ms, 256 142 ms, 1024 2.95 s - 1.9-3.3x
the WebGPU trunk's pairformer.

## What made it fast

Trunk: tensor cores, FP16 end to end with f32 residuals; the pair track's row-wise work as fused
kernels - each a warp's 16 rows held as MMA fragments in registers, weights streamed through
shared memory - so the wide intermediates never reach HBM: the triangle multiplication's input
side (LN, projection, gate, the output gate's logits; a and b written channel-major), its output
side (center norm, output projection, gated residual) around one cuBLAS batched contraction; the
pair transition (LN, both GEMMs, SwiGLU, residual); grid attention's input (LN, q/k/v/gate and the
pair bias in one pass, the column direction reading its rows transposed in place) and the column
direction's output projection; the single track's pair logits. The grid attention itself is a
FlashAttention-2 kernel on `mma.sync` (S, P, O in registers, cp.async double buffering, ldmatrix,
log2-domain scores) - occupancy-bound at ~65 TFLOP/s, see below. The MSA stack in f16 too.

Diffusion: everything derived from the conditioning computed once per fold (atom pair
conditioning and pair logits, every block's adaptive-LayerNorm scales/shifts and zero-init gates,
the transformer's 24 pair-logit sets as f16 flash biases); the transformer's 144 conditioning
projections as two GEMMs per step with each block's LayerNorm scale folded into its weights
(folded on the device); the whole step replayed as a CUDA graph; the sampler on the device, its
Gaussians a counter-based hash computed where they are used; the atom blocks' keys and values
projected once per atom and gathered; a split-over-keys flash kernel for the transformer's few
blocks at small n; `--samples` batched through the whole denoiser.

Start-up: model.bin mapped and copied to the device in one transfer; every fused weight
concatenated on the device. A first fold is within 5-15% of a warm one.

## Ligands, modified residues, nucleic acids

```
node ... export-model.mjs data-gol --sequence=<SEQ> --ligands=GOL          # CCD codes (RCSB)
node ... export-model.mjs data-sep --sequence=<SEQ> --modify=SEP@3         # as probe-modified.js
node ... export-model.mjs data-dna --sequence=<SEQ>:GCGATCGC:GCGATCGC --kinds=protein,dna,dna
node ... export-model.mjs data-smi --sequence=<SEQ> --smiles='OCC(O)CO'
node ... export-model.mjs data-ab --sequence=<A>:<B> --a3m=a.a3m,b.a3m [--paired-a3m=pa.a3m,pb.a3m]
node bonds.mjs fold.pdb GOL        # bond lengths by class, ideals from the CCD
node ... export-model.mjs data-job --job=../../tools/fixtures/af3-jobs/kras_g12c_sotorasib.json
```

`--job` reads an AlphaFold 3 job JSON (either dialect) with the page's own reader
(`web/job-json.js`, `web/entities.js`): chains and their kinds, CCD and SMILES ligands, modified
residues, `bondedAtomPairs`, and the first model seed (the native run's default `--seed`) - and,
unlike the page, the `unpairedMsa` / `pairedMsa` AF3's data pipeline writes into the job, which
become the alignment (5CAJ with 200 inline rows: 2.02 A, pLDDT 94.2). Seven of
AF3's examples fold single-sequence: ubiquitin, calmodulin + 4 Ca, KRAS G12C + covalent sotorasib
(SG-C25 1.53 A - bonded), ERK2 with two phosphorylations, streptavidin + SMILES biotin, the TetR
dimer on DNA (476 tokens), U1A on an RNA hairpin.

The featuriser is the repository's own; the trunk adds the bond embedding (one column, bias-free).
The PDB is written through the page's own `toPdb` records (exported as `template.pdb`): chains,
HETATM ligands under their codes, modified residues, CONECT, per-atom pLDDT. 6MRR with each, bond
rms (A): GOL ligand **0.020**, SEP **0.051**, a DNA duplex's nucleic bonds **0.040**; protein
mainchain 0.033-0.034 throughout.

## Templates

`--template=<pdb or cif>:<chain>[@<query chain>]` (comma-separated, up to four slots) builds each
slot with the page's own `buildTemplate` and its geometry with the reference's `templateGeometry`;
the trunk adds the distogram, masks and unit vectors per real slot and counts the empty ones.
Parts joined by `+` share one slot, as AF3 puts each chain's k-th template in slot k
(`1brs.pdb:A@0+1brs.pdb:D@1`); that slot may speak across its chains (`--no-span-chains` masks it).

| target | no template | self-template |
|---|---|---|
| 5CAJ, single sequence | 21.75 A, pLDDT 29.2 | **0.139 A**, pLDDT 95.2 (WebGPU AF3 0.281) |
| 1BRS barnase-barstar, single sequence | 16.15 A complex, ipTM 0.07 | **0.493 A** merged slot, ipTM 0.94 (WebGPU AF3 0.475); per-chain slots 0.513 |

`score.py <pred> <ref> A,D` scores a complex: one superposition over all chains, each chain also
alone, residues paired by aligning the sequences (a numbering gap in the reference shifts nothing).

## Not ported yet

The other dialects (OpenDDE, boltz2, protenix2, IntelliFold-2,
RoseTTAFold3 - each raises a named "not ported" error).

## Tried and not taken

- **A split-K "skinny" GEMM for the denoiser's few-row projections** (68 rows x 768 x 3072, where
  cuBLAS reads 4.7 MB of weights in 9 us against a ~3.5 us bandwidth floor). Correct to 5e-7, and
  its main loop plus partial writes ran in 6 us - but the cross-slice reduction (last block of a
  tile sums the slices, deterministic) put it at 11-25 us for every split target swept, slower than
  cuBLAS at every shape. Two traps on the way: a dynamically bounded loop over the accumulator array
  put 192 bytes of it in local memory, and predicated loads into one register serialised the
  reduction (each `LDG` waited on the last).
- **cuBLASLt with per-shape autotuning** (time every heuristic candidate, keep the fastest):
  no change on either fold - cuBLAS's default pick was already the fastest candidate.
- **The trunk's grid attention with two 16-query tiles a warp** (each K/V fragment feeding two
  MMAs, 128-query blocks on 4 warps): 63 against 65.5 TFLOP/s at 1044 tokens, 32.6 against 38.3 at
  261. The kernel is occupancy-bound at 128 registers and four blocks an SM; the variant needs 217.
  Stripping it piece by piece at 1044 tokens: no pair bias 0.86 ms of 1.09, no bias and no exp
  0.81, no PV either 0.75 - the floor is reloading each row's keys and values once per 64-query
  block, not the arithmetic.
- **Grid attention blocks that share the pair-bias tile across rows** (the bias is the same for
  every row of the grid, and per row it is half of the kernel's L2 traffic): exact, and slower in
  every geometry - 10.7 / 14.8 / 12.3 / 16.8 ms against the plain kernel's 8.8 at 1044 tokens for
  1, 2, 4 rows of 4 warps and 4 rows of 2. Not L2 bandwidth, then.


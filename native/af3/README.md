# AlphaFold 3 in CUDA

A native CUDA/cuBLAS AlphaFold 3, transcribed stage by stage from this repository's CPU
references under `src/af3/` (the specification) and checked against af3-any-model's own
oracle dumps in `oracle-dumps/`. Proteins fold end to end from a sequence (and optionally an
A3M), with pLDDT, PAE, PDE and pTM.

## Build and run

```
native/af3/fold kras.pdb --job=tools/fixtures/af3-jobs/kras_g12c_sotorasib.json -- --samples=5
native/af3/fold 5caj.pdb --sequence=<SEQ> --a3m=oracle-dumps/5caj-a.a3m
```

`fold` builds `af3` if it is missing, exports the weights once (`native/af3/weights`, from the
bundle on disk - no server), featurises the input with the repository's own featuriser into a
temporary directory (0.2 s, while `af3` starts) and folds it (`--fold --fast`); everything after
`--` goes to `af3`. A job JSON to a PDB is 1.5 s of wall clock (KRAS with sotorasib), 6MRR from
its sequence 1.0 s, 5CAJ with its alignment 1.7 s. By hand:

```
cd native/af3
nvcc -O3 -std=c++17 -arch=sm_80 --default-stream per-thread --use_fast_math src/af3.cu -lcublas -lcupti -o af3
node --js-float16array --max-old-space-size=24000 export-model.mjs weights --weights-only
node --js-float16array export-model.mjs in --no-weights --sequence=<SEQ> [--a3m=...]
./af3 in --weights=weights --fold --fast --out=fold.pdb
python3 score.py fold.pdb ../../tools/fixtures/5caj-crystal.pdb A
node --js-float16array --max-old-space-size=24000 export-model.mjs data   # + every oracle
./af3 data                                   # f32 path, every stage against AF3
```

`--fold` options: `--steps=200 --recycles=3 --seed=42 --folds=N` (N warm repeats), `--samples=N`
(N diffusion samples off one trunk, AF3 runs five: each scored by the confidence head and ranked by
0.8 ipTM + 0.2 pTM - pTM for one chain; AF3's disorder and clash terms are not computed - the best
written to `--out`, all to `<out>_sample<k>.pdb`). Beside every structure, what the page's archive
writes: `<stem>_confidences.json` (atom_chain_ids and atom_plddts in the PDB's atom order, the
distogram's contact_probs, pae, token_chain_ids, token_res_ids) and
`<stem>_summary_confidences.json` (chain_ids, chain_pair_iptm, chain_pair_max_contact,
chain_plddt, chain_ptm, chain_iptm, chain_pair_pae_min, iptm, ptm, ranking_score,
fraction_disordered, mean_plddt). The samples run as ONE batch through the
denoiser - the transformer's GEMMs at 5x the rows, the samples as the flash kernel's batch, what
they share (conditioning, masks, biases) read once - and sample k draws exactly what a one-sample
run seeded `seed + k` draws, so its structure is the same: 5CAJ's five read 1.989 / 2.023 / 1.961 /
1.996 / 2.018 A batched and one at a time. Five samples' diffusion: **1.33 s against 0.54 s** for
one,
`--fast` (f16 trunk and denoiser transformer), `--stages` (per-stage profile),
`--no-graphs`. `pairformer.cu` is the earlier one-file pairformer prototype and benchmark.

## Accuracy against AF3 (6MRR, f32 bundle)

| seam | f32 path | f16 path |
|---|---|---|
| target_feat (atom encoder) | 5.4e-8 | |
| z_after_msa | 1.0e-4 | 1.2e-4 |
| trunk_out_pair | **3.0e-5** | 4.2e-4 |
| single | 7.2e-6 | 1.3e-4 |
| one denoiser call | 1.8e-5 | 1.8e-3 |
| confidence PAE / PDE | 2.8e-6 / 3.7e-6 | |

WebGPU (docs/AF3.md, f32): trunk_out_pair 2.98e-4. `z_init` reads 2.2e-4 because AF3's dump
stores that tap in bfloat16. 🔴 pLDDT on the confidence oracle's RANDOM inputs is
ill-conditioned - a 1e-6 change to the input moves it 7.2e-3 - so it reads ~6e-3 there (the
JS CPU reference 8.8e-3) and is checked on folds. 🔴 Likewise a random N(0,1) block input reads
1-3e-2 in any 16-bit format where AF3's real input reads 3e-5: always check on real inputs.

Folds: 6MRR from its sequence **0.515 A** CA RMSD, pLDDT 87.6, pTM 0.767 (WebGPU 0.65-0.71);
5CAJ (255 residues of chain A) with its MSA **1.93 A**, pLDDT 95.1, pTM 0.943 - the WebGPU
port gives pLDDT 94.5, pTM 0.938 on the same inputs. `gate.py` holds these and nine more.

## Speed: A100-SXM4-40GB, 200 diffusion steps, `--fast`

| target | WebGPU first / warm (0 recycles) | native first / warm, 0 recycles | native warm, 3 recycles |
|---|---|---|---|
| 6MRR, 68 tokens | 5.1 s / 2.4 s | 0.57 s / **0.39 s** | 0.45 s |
| 5CAJ, 261 tokens, 512 MSA rows | 11.4 s / 7.4 s | 0.88 s / **0.67 s** | 1.02 s |
| 5CAJ x 2, 522 tokens, 512 rows | | 1.42 s / 1.24 s | 2.55 s |
| 5CAJ x 4, 1044 tokens, no MSA | | 3.62 s / 3.48 s | 9.42 s |

WebGPU with the developer flags (`fold.js --folds=2`); a stock-Chrome NVIDIA visitor gets
about half its speed. A whole `fold` command - the export, the CUDA context, the weights, the
fold - is 1.0 s for 6MRR and 1.7 s for 5CAJ. No 2 GiB binding ceiling. At 1044 tokens a trunk
pass is 1.97 s - grid attention's flash kernel 38% of it - so recycles dominate there; up to ~300
tokens the 200 denoiser steps do. 2088 tokens: 11.3 s a trunk pass, 2.9 s of diffusion.

Against AlphaFold 3 itself - af3-any-model's JAX (bf16, Triton flash attention) with DeepMind's
weights, on this A100, `tools/oracle/bench_af3_native.py` at matched settings (one sample unless
said, no token bucketing, the same MSA rows), steady-state calls:

| 200 steps | JAX AF3 | native `--fast` | |
|---|---|---|---|
| 6MRR, 68 tokens, 1 pass | 1.67 s | **0.39 s** | 4.3x |
| 5CAJ, 261 tokens, 512 rows, 1 pass | 2.76 s | **0.67 s** | 4.1x |
| 5CAJ, 4 passes (3 recycles) | 3.47 s | **1.02 s** | 3.4x |
| 5CAJ, 1 pass, 5 samples | 5.00 s | **1.58 s** | 3.2x |
| 5CAJ x 4, 1044 tokens, no MSA, 1 pass | 11.1 s | **3.48 s** | 3.2x |

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
log2-domain scores) - ~85 TFLOP/s at 1044 tokens, see below. The MSA stack in f16 too.

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

## Many inputs

`native/af3/fold --batch=<file>` folds every line of `<file>` (`<out.pdb> <input flags>`) in one
`af3` process (`af3 dir1,dir2,... --out=a.pdb,b.pdb` underneath): the weights load once and every
fold after the first skips CUDA's start-up, the weight upload and the first fold's warm-up, and
each input folds as soon as its export lands - three folds in 2.4 s where three commands take 4.4.
Each input's outputs are byte-identical to folding it alone. A single `fold` command starts `af3`
before the export too (`--wait-input`), so the CUDA context and the weight upload overlap it, and
returns once the outputs are written rather than after the driver has released the process's
device memory (`--detach-output`: a quarter second), and the weights go up through pinned buffers
filled by three threads (77 against 170 ms), and every weight's f16 copy is made in one launch, and a thread pays cuBLAS's first-GEMM cost (~70 ms) meanwhile:
6MRR is 1.00 s
a command, of which the fold is
0.64.

## A resident af3

`native/af3/fold --serve` starts one `af3` (`--serve=DIR`) that keeps the weights on the device and
every kernel and cuBLAS plan warm; until `native/af3/fold --stop`, every ordinary `fold` command
for that model has its input exported by a resident exporter too (`export-model.mjs --serve`,
node's module loading being most of an export) and hands it to the server as a job (per-job
`--samples`, `--steps`, `--recycles`, `--seed`) - 6MRR 0.53 s a command, against 1.00 s starting
both each time. The outputs are byte-identical either way.

## The gate

`python3 native/af3/gate.py` folds 6MRR from its sequence through all seven models, plus 5CAJ and
barnase-barstar with their crystals as templates, scores each against the deposited structure and
holds RMSD and mean pLDDT to `gate-baseline.json` (0.05 A, 0.5 points; `--write` re-records,
`--only=` takes models or case names). Run it after any kernel change; it takes about three
minutes once every model's weights are exported. The baseline is this A100's.

## Other AF3-lineage models

`--model=<name>` on `fold` (or `--bundle=` on the exporter, `--oracle-model=<name>` for its own
oracles) folds the other checkpoints the page offers, through the same dialect flags:

| model | against its own af3-any-model oracles | 6MRR from its sequence | 5CAJ + its crystal |
|---|---|---|---|
| OpenBind-0 | trunk_out_pair 5.4e-7, PAE 4.2e-6 | 1.71 A | |
| protenix2 | trunk_out_pair 1.0e-6, denoise 1.5e-6, PAE 2.4e-7 | 0.453 A | 0.136 A |
| IntelliFold-2 (int5 bundle) | trunk_out_pair 4.5e-2 (the quantisation; z_after_msa 9.1e-3) | 1.551 A | 0.296 A |
| RoseTTAFold3 (int5 bundle) | trunk_out_pair 4.85e-2 (WebGPU on the same bundle 4.90e-2) | 1.621 A, bonds 0.050 A | |
| boltz2 | trunk_out_pair 4.5e-7, PAE 4.5e-7; denoise 2.5e-2 (see below) | 0.460 A (0.466 on its reference batch; WebGPU 0.507) | 0.347 A |
| OpenDDE | trunk_out_pair 9.6e-7, expander 5.2e-9, refiner 1.8e-7, denoise 1.1e-6, PAE 1.8e-6 | 1.499 A at the page's 16 steps (WebGPU 1.501); its reference batch 0.679 (0.485 at 200) | 0.127 A |

Ported for them: the padded single conditioning, per-block atom pair norm, chained atom
LayerNorms, split pair conditioning, per-block atom masking, the fused template embedder (passes
built by the exporter as the page's trunk builds them - empty-slot gap, coverage weights, outer
residual, rf3's one averaged pass), protenix2's confidence head (raw-distance term, normalised
single, PDE symmetrised before its projection), head width 64; for rf3 the pre-trunk query, q/k
LayerNorms in the transformer and the atom blocks, the no-residual block wiring, 35-wide MSA
features, biased outer-product projections, the chirality gradient term, and its confidence head
(whole-tensor masked norms, the CA distogram); for boltz2 the up-gated conditioned transitions,
its 384-wide target_feat (the atom encoder plus six summed projections), the bond-order and
contact-conditioning z-init terms, the MSA update before the outer product and the pre-MSA pair
added back, the re-embedding confidence head with split intra/inter-chain heads and no head
LayerNorms, and its own EDM constants; for OpenDDE its second token space - the expander (49
role-pair projections, one GEMM per matrix over the pairs sorted by it), the four-block refiner
with the expander's bias on every single-attention head, the diffusion on the structural batch
(the exporter writes it as `sbatch.*` and `af3` swaps it in for `batch.*` after the trunk), and
its own confidence head, mapped back onto residues through the layout's gathers.

boltz2's denoise reads 2.5e-2 because its token transformer amplifies its input ~2.2e4x: the seams
before it are 1.2e-6 (`transformer.act`), and the transformer fed the oracle's own input reads
9.4e-4 (`TX_ORACLE_IN=1`; the WebGPU f32 path's whole step is 3.5e-3). The seams are compared
whenever a stage oracle (`oracle-dumps/af3-oracle-stages-<model>.json`) was exported.

After a first fold `af3` lists every weight family it never read; for these models the list is
only what should be there (heads not computed, alternative per-block forms, absent bonds and
template geometry) - rf3's atom-block q/k norms and chirality term were found by it.

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
- **The grid attention's exponentials in f16x2 on their own** (two per special-function
  instruction): 9.2 against 8.6 ms at 1044 tokens. Combined with starting S from the bias and taking
  P's row sums on the tensor cores (P . ones), they did pay - see below. Nor did 8-warp blocks at
  the largest sizes (8.8 / 8.1 against 8.6 / 8.0 ms at 1044 / 2088), a third or fourth cp.async
  stage (10.5 against 8.0 ms: the shared memory halves the blocks an SM holds), or interleaving
  two key fragments' MMAs (the registers). What did pay: no mask when every token is real (3-7%).

- **The trunk pass as a CUDA graph** (a pass is ~1000 launches): 88.8 -> 80.5 ms for four passes
  at 68 tokens and 469 -> 458 at 261 once warm, but the capture costs ~5 ms and a scratch buffer
  that a later phase grows forces another, so a single fold - the usual run - came out slower.
- **Larger blocks for the fused pair transition** (16 warps, halving its weight reads from L2):
  219 against 199 ms at 1044 tokens. Stripping it, neither the residual (-23 ms) nor the second
  GEMM (-27) dominates.

- **Two 16-row tiles a warp in the triangle's input kernel** (each weight fragment feeding two
  MMAs, half the shared-memory reads): 8 warps of 32 rows lost to 16 of 16 - 252 against 230 ms at
  1044 tokens; 16 warps of 32 rows do not fit in shared memory.
- **The atom blocks' transition fused** (the pair transition's kernel with the adaptive LayerNorm's
  output read in and a gated residual): slower at every size - 400 against 371 ms of diffusion at
  68 tokens, 1314 against 1300 at 1044 - few rows make few blocks, and cuBLAS's three launches win.
- **cuBLAS split-K by batching the K slices** for the denoiser's N = 768 projections: 1-4 us a
  block, less than the consumers' extra reads would cost.

- **Hand-written skinny GEMMs for the denoiser's 68-row projections**, three designs: split over K
  with per-slice f32 partials (the consumers to sum them) - 5.2 / 6.7 us against cuBLAS's 8.2 /
  10.8 for the N = 768 projections, but no better for N = 3072, so ~5% of a 68-token fold after
  the consumers' extra reads; the same, pipelined - no faster; no split, every block streaming its
  columns' weights over the whole K - 15 against 9 us, each block re-reading all of X from L2.
  cuBLAS's own tiling and split-K searches (781 configurations) found nothing faster either.
- **The outer product mean's product straight into the output GEMM's layout** (a strided-batched
  GEMM over query tokens, the output weight's rows permuted to match): its m = 32 tiling ran
  0.84 ms a call against 0.5 for the GEMM and permute it replaced.
- **Grid attention without bounds checks** (inputs padded so loads past n read finite values):
  7.22 against 6.89 ms - at 125 registers the kernel is one register from losing a block an SM.

## What the grid attention's time was

Without a profiler on this box, by counting SASS: the kernel issued ~800 integer instructions a
tile (the cp.async addresses, recomputed every tile in a strided loop the compiler could not
unroll) against 36 tensor-core MMAs. Compile-time trip counts and 32-bit offsets, the bias as S's
starting value, f16x2 exponentials straight into the P fragments, P's row sums as one more MMA,
and a tree for the row maxima: 8.07 -> 7.47 ms at 1044 tokens (`--bench-grid=1044`).

The triangle contraction runs in a padded np x np space (np a multiple of 8): cuBLAS's GEMM is
twice as fast on an aligned size (1044: 5.4 against 2.7 ms the pair of them). A multiple of 32 was
the first choice and padded the fused kernels' rows for nothing (68 tokens: 96^2 rows against 72^2).


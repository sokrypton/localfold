# BioEmu2 in CUDA

A native CUDA/cuBLAS BioEmu2 (microsoft/bioemu2, checkpoint `bioemu2-preview`, MIT), held stage by stage to
af3-any-model's port (`network/bioemu2.py` on its colab branch, itself held to the vendor at 0.000 A) on the same
input. It reads af3-any-model's own blob (`diffuser/~/bioemu2`: the vendor's PyTorch names and `(out, in)`
layouts), and its input is `shared/bioemu2/featurise.js` - the page's featuriser - through `export_input.mjs`.

It runs:
- the input embedder: per-atom features and the windowed atom pair, three sliding-window conditioned layers
  (32 queries, 128 centred keys), the mean over each token's atoms -> `s_inputs` (455);
- `z_init` (method conditioning, relative position, token bonds, contact conditioning) and four trunk passes, each
  4 MSA layers (pair-weighted averaging, outer product mean over valid rows, the two Boltz triangle
  multiplications, transitions) then 48 PairMixer blocks (triangle out, triangle in, transition);
- the distogram (Boltz's 64 bins over 2-22 A), whose P(d < 8 A) is the contact map;
- the dual-stream denoiser: 16 layers of a windowed atom transformer and a full 768-wide token transformer, both
  adaLN-Zero on the single conditioning, coupled through every token's centre atom; EDM preconditioning, sigma_data 16;
- Boltz's realigning EDM sampler (gamma 0.8 above sigma 1, noise 1.0, step 1.5 structure / 1.0 equilibrium,
  rho 7, sigma 4e-4..160, 200 steps, a 0 appended), every sample centred on its real atoms.

There is **no confidence head** in this checkpoint: the PDB's B-factors are zero and the confidence files carry
`null` pLDDT, PAE and pTM (and say why) beside the contact probabilities.

## Use

```
bash cuda/build.sh bioemu2                     # the binary for this GPU
cuda/bioemu2/fold out.pdb --job=tools/fixtures/bioemu2/6mrr.json
cuda/bioemu2/fold out.pdb --sequence=GWSTELEKHREEL... -- --samples=8 --steps=200 --seed=3
```

`fold` featurises into a scratch directory and folds with `--fast`; the int8 weights (490 MB) are fetched once into
`<checkout>/af3am-bioemu2` (`cuda/featurise/fetch-weights bioemu2`, pinned in fetch.h) unless `--weights=<dir>`
names a directory holding a `bioemu2*.bin.zst` (the float32 blob for the oracle).

| input flag (export_input.mjs) | |
|---|---|
| `--job=<json>` / `--sequence=A:B` | an AlphaFold 3 job (protein, RNA, DNA, CCD and SMILES ligands, modified residues, declared bonds, inline MSAs) |
| `--mode=equilibrium` | sample the equilibrium ensemble (method token 4, step scale 1.0) instead of a structure (7, 1.5) |
| `--contacts=<json>` | contact conditioning, BioEmu2's `Contact` form (`[{"token1": {"asym_id", "res_id"[, "atom_name"]}, "token2": ..., "max_distance"}]`; a named atom must be a token's centre) |
| `--propagate-from=<pdb\|cif> --lag-ns=1 --propagation-steps=N` | propagation: each sample a trajectory of N lag-time steps from that structure (equilibrium sampling, the frame as `coords_cond`, the endpoint superposed on the previous frame); sample `i * N + k` is trajectory i, step k + 1 |

| binary flag | default | |
|---|---|---|
| `--samples=N` | 5 | samples (all through each denoiser call together) |
| `--steps=N` | 200 | sampler steps (at least 2) |
| `--seed=N` | 0 | this port's own draws (a seeded mt19937 per sample): a different sample from the reference's |
| `--step-scale=S` | the mode's | |
| `--fast` | off in the binary, on in `fold` | f16/bf16 tensor cores; without it every GEMM is float32 (the oracle's arm) |
| `--serve=<dir>` | | stay resident (common.cuh's serveJobs), each `<id>.job` an input dir and flags |
| `--oracle=<dir>` | | `oracle_input.mjs`'s conversion of a `tools/oracle/dump_bioemu2.py` dump: every stage's relRMS |

Outputs for `--out=fold.pdb`: `fold.pdb` (sample 0), `fold_sample-<k>.pdb`, `fold_confidences.json` (contact
probabilities, token chain/residue ids, null confidences), `fold_summary_confidences.json`.

## Held to the reference

`python3 cuda/bioemu2/gate.py` (and `--only=oracles` / `--only=folds`, `--write` to re-record). The oracle arm runs
every fixture in `tools/fixtures/bioemu2` that has a dump in `oracle-dumps/`, the denoiser fed a REAL sample
re-noised at sigma 160 / 16 / 1 / 0.05 plus a propagation call at a 1 ns lag, from our trunk and from the
reference's own. Worst stage relRMS on the float32 blob:

| target | float32 | `--fast` |
|---|---:|---:|
| 6MRR (71 tokens, single sequence) | 2.1e-6 | 2.2e-3 |
| 6MRR + SEP@3 + GOL | 1.6e-6 | 2.9e-3 |
| RNA + DNA | 3.6e-6 | 3.1e-3 |
| 6MRR + a SMILES glycerol | 2.5e-6 | 2.0e-3 |
| 6MRR equilibrium, two contacts | 2.1e-6 | 3.2e-3 |
| heteromer (156 tokens, 41 paired MSA rows; one pass) | 1.6e-6 | 1.7e-3 |

The int8 blob decodes to af3-any-model's own dequantised values (`--compare-weights=<float32 dir>`: the worst
tensor's relRMS against float32 is 9.544e-3 on both sides, to four digits), so the int8 fold's residual - trunk
5e-3 to 7e-3, distogram 2.7e-2, denoiser 2.4e-2 at sigma 160 - is the quantisation, not this port. 🔴 A denoiser
output is compared over the REAL atoms: a padding atom's coordinates are undefined on both sides (under f16 its
state saturates where the reference's grows).

## How it is made fast (`--fast`)

- **The samples go through each denoiser call together**, and everything that depends on sigma alone - the single
  conditioning and every layer's conditioning projections, the atom side's too - is computed once a call on the T
  token rows: the atom stream is conditioned on its token's row broadcast to atoms, so its `cond_2` is the same for
  every atom of a token and every sample (an atom row reads it through a map; a padding atom reads a zero row, as the
  reference conditions it). Exact: the float32 oracle is unchanged by it.
- **The trunk's pair track is cuda/ef2's** fused 256-channel kernels (`triangle256` / `transition256`, the
  ESMFold2 trunk's), each block's weights re-laid once into ef2's layout (`fasttrunk.cuh`; 🔴 the incoming
  triangle's a and b swap, because ef2's incoming contraction is `sum_k a[k][j] b[k][i]` and Boltz's
  `sum_k a[k][i] b[k][j]`): 494 tokens 7.4 -> 1.6 s for four passes.
- **The denoiser's GEMMs are f16** on the tensor cores, their inputs written in f16 by the norm or gate before them,
  fc1 | fc2 one GEMM - except 🔴 **fc3, whose input is a SwiGLU product reaching 1.3e5** (f16 stops at 65,504: the
  first fold came back NaN), which takes bf16. fc1 | fc2's own outputs peak at 1,059 over a whole fold.
- The windowed atom attention splits each query's 128 keys over four lanes, merged by shuffles.

## Timings

The A100 (40 GB SXM4), int8 weights, `--fast`, 5 samples, 200 steps, a process's first fold:

| tokens (atoms) | inputs | trunk (4 passes) | static context | sampler | a step (5 samples) |
|---|---:|---:|---:|---:|---:|
| 71 (599), 6MRR | 43 ms | 226 ms | 47 ms | 3.04 s | 15 ms |
| 255 (2,058), 5CAJ | 40 ms | 569 ms | 49 ms | 6.43 s | 32 ms |
| 494 (3,740), 1TIM A:B | 41 ms | 1,612 ms | 71 ms | 11.54 s | 58 ms |

The first fold of a process also derives the re-laid trunk weights and the f16 mirrors; a resident `--serve`
process pays that once (6MRR's trunk 226 -> 100 ms warm, the warm fold byte-identical to the cold one). Without
`--fast` (float32 everywhere) 494 tokens is trunk 7.4 s and 81 ms a step. At 71 tokens the denoiser keeps the GPU
~90% busy at ~12 us a kernel - short kernels, not launch gaps - so a CUDA graph would win at most ~10% there.

Fold accuracy, 6MRR from its sequence alone (its crystal, CA RMSD, 5 samples): this port 2.78-3.04 A (mean 2.875,
int8; 2.80-3.04 / 2.88 on the float32 blob), af3-any-model's own `run_alphafold.py --model bioemu2` 2.88-3.20 A
(mean 3.06). Bond geometry against the CCD's ideals on the same targets as colab's own samples: ligand + SEP 0.084
against 0.077 A, RNA + DNA 0.037 against 0.036, mainchain 0.034 and side chains 0.052 on both.

## Not done

- **Steering** (`edm_gd`, `edm_fk`, `edm_fkc`) and the physicality report. `sampler.cuh` steps all particles
  together and carries a `StepHook` (`guide` on the denoised estimates, `resample` over states and companions) at
  exactly the seams the reference's steered samplers need; `edm_fkc` also needs the denoiser's VJP, which this port
  does not have.
- **The native featuriser and the worker**: `export_input.mjs` is the input path. The C++ twin
  (`cuda/featurise/bioemu2_features.h`) produces the same named arrays, so the native path is a serialisation of
  them into these entries (the list at the top of `export_input.mjs`, plus `pdb.template` and
  `confidence.template.json`).
- The token attention is cuBLAS (per sample, a softmax kernel between): ~10 ms of a 58 ms step at 494 tokens.
  BioEmu2's token heads are 48 wide, which none of cuda/af3's flash kernels take.

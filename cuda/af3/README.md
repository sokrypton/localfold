# AlphaFold 3 in CUDA

A native CUDA/cuBLAS AlphaFold 3, transcribed stage by stage from this repository's CPU
references under `cpu/af3/` (the specification) and checked against af3-any-model's own
oracle dumps in `oracle-dumps/`, for all seven AF3-lineage families (AlphaFold 3, Protenix-2,
Boltz-2, IntelliFold-2, RoseTTAFold3, OpenBind-0, OpenDDE). It takes what AlphaFold 3 takes - an
AF3 job JSON in either dialect (all fourteen of AF3's example jobs fold), or a sequence with
alignments - proteins, DNA, RNA, CCD and SMILES ligands, glycans, ions, modified residues and
bases, covalent bonds, templates, several seeds and samples, its own `userCCD` - and, with
`--search`, fetches the alignments and templates itself. It writes what AF3 writes: the structure
(PDB or mmCIF), pLDDT, PAE, PDE, contact probabilities, pTM/ipTM and their per-chain forms,
`has_clash`, `fraction_disordered` and the ranking score (exact against AF3's own functions), the
ranking CSV, and on request the embeddings and the distogram. 6MRR folds in 0.45 s warm, a
1044-token complex in 8.4 s; `--af3-defaults` runs AF3's own 10 recycles and 5 samples.

## Guide

### Quick start

`--help` prints every option; with no arguments the binary prints a short usage.

```
bash cuda/build.sh af3                          # once: the binary for this GPU (and the featuriser it links)
cuda/af3/localfold-af3 --job=tools/fixtures/af3-jobs/kras_g12c_sotorasib.json --out=kras.pdb
cuda/af3/localfold-af3 --sequence=GWSTELEKHREEL... --model=boltz2 --search --out=6mrr.pdb
```

One command runs the whole protocol in one process: the model's weights are downloaded the first time, the
input is read and featurised (the page's own featuriser, natively), the MSA search runs if asked, and the fold
writes the structure and its confidences. Nothing else is installed or started - no Python, no Node. Folding
the KRAS job above takes 1.6 s from command to PDB on an A100 once the weights are on disk.

Needs an NVIDIA GPU (T4 and newer: sm_75, sm_80, sm_86, sm_89, ...), the CUDA toolkit to build (`nvcc`,
cuBLAS), `g++`, and `curl` and `gzip` at run time (the weight download and the MMseqs2 search).

### Models

`--model=` picks the checkpoint. All eight run through this one binary on af3-any-model's published int8
weights (huggingface.co/sokrypton/af3-any-model), each with its own conventions (shared/af3/dialect.js).

| `--model=` | | download |
|---|---|---:|
| `af3` (default) | AlphaFold 3 - Google DeepMind's parameters, **academic non-commercial use only**: the first download asks you to accept the [terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md), or set `LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3` | 350 MB |
| `boltz2` | Boltz-2 | 487 MB |
| `chai1` | Chai-1 - also downloads ESM2 3B (2.5 GB), whose embeddings it reads | 312 MB |
| `protenix2` | Protenix-2 | 292 MB |
| `intellifold2` | IntelliFold-2 | 800 MB |
| `rosettafold3` | RoseTTAFold3 (diffusion sampler only) | 352 MB |
| `opendde` | OpenDDE | 541 MB |
| `openbind0` | OpenBind-0 | 351 MB |

Every set of weights carries its authors' licence; check it before use beyond research.

### Input

**An AlphaFold 3 job file** - `--job=<job.json>`, either dialect (AlphaFold Server's or the open one). It is the
fullest form: proteins, DNA, RNA, ligands by CCD code or SMILES, ions, glycans (a ligand chain of several
components), modified residues and bases, declared covalent bonds (`bondedAtomPairs`), alignments and
templates carried inline, `userCCD`, and `modelSeeds` (every seed is folded). All fourteen of AlphaFold 3's own
example jobs fold (`tools/fixtures/af3-jobs/`).

**Or flags**, for the common cases:

| flag | |
|---|---|
| `--sequence=<A>:<B>:...` | the chains, joined by `:` (a comma inside a sequence is one chain with an unknown residue) |
| `--kinds=protein,dna,rna` | each chain's kind, in order (protein when absent) |
| `--ligands=GOL,ATP` | ligands by CCD code, one copy each |
| `--smiles='CCO\|c1ccccc1'` | ligands by SMILES, `\|`-separated: the conformer is built natively |
| `--modify=SEP@3[@<chain>]` | modified residues or bases, `CODE@position` (1-based), comma-separated |

**The alignment**: none (single sequence) by default;
- `--search` - the protein chains' alignments from the ColabFold MMseqs2 server, the paired block included for a
  complex. **It sends your sequences to api.colabfold.com**, which is why it is a flag. The alignment is kept
  beside the structure as `<out>.a3m`, so a second fold can reuse it with `--a3m=`.
- `--a3m=<a.a3m>[,<b.a3m>]` - one per protein chain; `--paired-a3m=` for a complex's paired rows.
- `--max-msa=1024` - rows kept.

**Templates** (up to four slots):
- `--template=<file>:<chain>[@<query chain>]` - a PDB or mmCIF structure aligned to a query chain (0-based, 0 when
  absent); `+` joins parts into one slot spanning several chains, `,` separates slots. Example:
  `--template=1brs.pdb:A@0+1brs.pdb:D@1` gives one slot over both chains of a complex.
- `--search-templates` - each protein chain's best four hits from the same MMseqs2 search, fetched and aligned.

### Fold options

| flag | default | |
|---|---|---|
| `--out=<path>` | `fold.pdb` | `.cif` writes mmCIF |
| `--samples=N` | 1 | diffusion samples a seed, ranked by AlphaFold 3's ranking score |
| `--seed=N`, `--seeds=a,b,c` | the job's `modelSeeds`, else 42 | |
| `--recycles=N` | 3 | trunk recycles (N+1 passes) |
| `--recycle-tolerance=<A>` | off | stop recycling once the prediction moves less than this |
| `--steps=N` | 200 | diffusion steps |
| `--flow` | | the page's Flow sampler instead of diffusion (not `rosettafold3`, `chai1`) |
| `--af3-defaults` | | AlphaFold 3's own run settings: 10 recycles, 5 samples |
| `--frames=<dir>` | | every intermediate result as it lands (below) |
| `--save-embeddings`, `--save-distogram` | | what AlphaFold 3's flags of the same names write |

### Output

For `--out=fold.pdb`:

| file | |
|---|---|
| `fold.pdb` (or `.cif`) | the structure (the best sample when there are several), B-factors are pLDDT |
| `fold_confidences.json` | per-atom pLDDT, the PAE matrix, contact probabilities, token chain/residue ids - AlphaFold 3's layout |
| `fold_summary_confidences.json` | pTM, ipTM, per-chain and chain-pair pTM/ipTM, min PAE, `has_clash`, `fraction_disordered`, `ranking_score` |
| `fold_sample<k>.pdb` + its two JSONs | each sample, when `--samples` or several seeds |
| `fold_ranking_scores.csv` | seed, sample, ranking score |
| `fold.a3m` | the searched alignment (`--search`) |

### Watching a fold

`--frames=<dir>` writes each intermediate result while the fold runs - what the website draws live:
`contacts-PP-of-NN.u8` after every trunk pass (an n x n byte map of contact probability x 255) and
`frame-SSSS.pdb` for every diffusion step's prediction. Files land complete (written then renamed), so a viewer
can poll the directory.

### Weights

Downloaded once into `af3am-<model>/` in the directory `--weights-dir=<dir>` names, else in the checkout the binary was built in (`~/.cache/localfold` for a binary copied out of its checkout). The download is locked, so two folds
asking at once share one, and an interrupted one leaves nothing that reads as finished. `cuda/featurise/fetch-weights <model>`
fetches without folding (it takes `--weights-dir` too).

### Ligand and residue definitions (the CCD)

A ligand, ion or modified residue named by its CCD code is read from wwPDB's chemical component dictionary. By
default each component is fetched from the RCSB the first time it is used and cached in `~/.cache/localfold/ccd` -
as the website does - so a new component needs the network once. To fold with none:
- `cuda/featurise/fetch-weights ccd` (`localfold-fetch ccd` from the wheel) downloads the whole dictionary once into
  the weights directory as `ccd/components.cif` (519 MB uncompressed, ~6 s), and every fold then reads it - no
  `--ccd` needed;
- or `--ccd=<components.cif>` names a dictionary you already have (AlphaFold 3 installs carry one; `.gz` works,
  plain is ~30x faster to read).

A run that reads a dictionary reads that alone: a code it lacks is refused (`Error: XYZ is not in the CCD ...`),
never fetched from the network behind it, and its components never enter the per-component cache. Folds are
byte-identical either way: the native featurisers reading the dictionary match the website's, which fetches from
the RCSB, on every case of `npm run test:featurise` (`--ccd=ccd/components.cif`).

### Limits and errors

- Up to ~6,000 tokens on a 40 GB A100 and ~3,200 on a 16 GB T4; past what the card holds it refuses up front and
  says the longest it takes (see "How large a fold fits" below).
- A refusal is one sentence, `Error: ...`, and a nonzero exit: an unknown model, a job the reader cannot take,
  Flow on a model without it, a flag the input cannot use.

### On Colab and the website

The website's Colab backend (`notebooks/localfold.ipynb`) folds with this binary: `cuda/worker.py` keeps each model
resident (`localfold-af3 - --serve=<dir>`) and streams every trunk pass and diffusion frame to the page. See docs/WEB.md.


## The `fold` wrapper, search and templates, in detail

```
cuda/af3/fold kras.pdb --job=tools/fixtures/af3-jobs/kras_g12c_sotorasib.json -- --samples=5
cuda/af3/fold 5caj.pdb --sequence=<SEQ> --a3m=oracle-dumps/5caj-a.a3m
```


`--search` gets the protein chains' alignments from the ColabFold MMseqs2 server - the page's own
client and merge, the paired block included for a complex - instead of an A3M: barnase-barstar
folds to 0.647 A (ipTM 0.93) from its two sequences, against 16.7 A without, the search 4.0 s.
`--search-templates` adds the templates from the same search: each protein chain's best four hits,
fetched as mmCIF and aligned by the page's `buildTemplate`, every chain's k-th in slot k as AF3
does (5CAJ: its own chain B found, 1.864 A with the alignment - a crystal template beside a deep
alignment moves this target little: chain A by hand gives 1.836, and 0.289 without the alignment).
Both send the sequences to api.colabfold.com, so they are flags, never defaults.

`fold` builds `localfold-af3` if it is missing, reads the model's weights as published - **af3-any-model's
own int8 blob** for all eight (`af3am-<model>/`, fetched once by `cuda/featurise/fetch-weights
<model>` from huggingface.co/sokrypton/af3-any-model, the files its JAX backend reads), with
its codes decoded on the device and a weight walk naming each tensor's slice of it (below).
**AlphaFold 3's are Google DeepMind's parameters, hosted for academic, non-commercial use under its
[AF3 terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)**: the
first fetch asks you to accept them, or `LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3` says you have (the
CUDA worker sets it, the page having asked). Then `fold` featurises the input with the **native featuriser**
(`cuda/featurise/af3-featurise`, below) into a temporary directory (10-200 ms, while `af3` starts) and folds it
(`--fold --fast`); everything after `--` goes to `af3`. **No JavaScript runs anywhere on that path.** An alignment keeps a seeded 1024 of its rows, AF3's own `num_msa`
(`--max-msa=N` to change it: 512 is 3.6% less trunk on 5CAJ's 7907-row search, pLDDT 95.08
against 95.20). A job JSON to a PDB is 1.5 s of wall clock (KRAS with sotorasib), 6MRR from
its sequence 1.0 s, 5CAJ with its alignment 1.7 s. By hand:

```
bash cuda/build.sh af3                       # localfold-af3, the featuriser linked in (cuda/featurise/standalone.o)
cd cuda/af3
../featurise/af3-featurise in --no-weights --family=af3 --job=<job.json>    # or --sequence=<SEQ> [--a3m=...]
./localfold-af3 in --bundle=../../af3am-af3 --family=af3 --fold --fast --out=fold.pdb
python3 score.py fold.pdb ../../tools/fixtures/5caj-crystal.pdb A
node --js-float16array --max-old-space-size=24000 export-model.mjs data   # + every oracle
./localfold-af3 data                         # f32 path, every stage against AF3
```

The oracles want the float32 weights (`export-model.mjs weights --weights-only
--bundle=<f32 manifest>`, then `--weights=weights`), which are not published.

🔴 **NO MAP FILE: THE WEIGHTS ARE WALKED FROM THE BUNDLE** (cuda/featurise/af3_weights.h, the port of
shared/af3/weights/weights.js and diffusion-weights.js, with the dialect table generated from
shared/af3/dialect.js by tools/gen-native-featuriser-tables.mjs). `--family=<m>` names the dialect; the
walk reads the bundle's tensor names and shapes and says which slice of which tensor each native weight
is - zeros, ones, or a per-block LayerNorm scale folded into its projection as the page's loader folds
it - so a re-exported bundle needs nothing regenerated. It replaced `cuda/af3/maps/*.map`, which a
value search over a float32 export had written: the walk is line-for-line those maps but where the
search took a coincidence (a zero or duplicate tensor matched under the wrong name, values identical or
within 1e-21) and where dialect flags had gone stale (none read by the binary), and every family's 6MRR
fold through it is byte-identical to the map's but OpenDDE's, 0.010 A rms away at the same pLDDT and pTM.

🔴 **A BLOB IS READ AS IT IS PUBLISHED** (common.cuh's blob reader): one zstd stream of haiku records, its
tensor names the ones LocalFold's bundles were exported under - so the same walk reads either, checked on
all eight: every bundle tensor is in its blob at its shape but chai1's structure-pair three, which
`tools/add_chai1_structure_to_blob.py` added to af3-any-model's chai1 blobs (commit 28141c7, every other
record byte-identical), and AlphaFold 3's two Fourier tensors - a constant of its source, frozen from a
fixed seed, which DeepMind's file does not carry - compiled in as their values (`cuda/featurise/af3_fourier.inc`,
a `c` line the walk emits for stock AF3): one walk reads the int5 bundle and the blob alike,
the bundle's fold byte-identical through it. Decompressed once, through the system's libzstd, into record-aligned shards
beside it (`<blob>.raw/`, 256 MB each, an int8 tensor and its scales in one) so the upload streams them
as it streams a bundle's: ESM2's 2.8 GB loads in 2.0 s warm at 1 GB of host memory. int8 is a float32
scale per output channel and per block of rows; the decode is af3-any-model's `dequantise_int8` exactly
(0.0 over boltz2's 120 int8 tensors). Measured against the int5 bundles on 6MRR: af3 0.620 / 0.620 A, boltz2 0.573 / 0.564,
protenix2 1.536 / 1.514, intellifold2 1.564 / 1.551, openbind0 1.716 / 1.104, opendde 1.519 / 1.533,
rosettafold3 1.644 / 1.817 - one sample's seed band - and chai1's ligands better (GOL 0.021 / 0.053 A,
SEP 0.074 / 0.108).

**Chai-1** (`--family=chai1`) folds through the same binary on af3-any-model's conventions (the
`chai*` flags in shared/af3/dialect.js), and takes two things nobody else does: ESM2 3B's embeddings,
computed in the fold from af3-any-model's `lm/esm2.bin.zst` (`--esm-bundle=<its directory>`, within
1.0e-2 of the float32 model, its int8 codes resident), and chai-lab's structure token-pair weights,
which af3-any-model's converter drops and its chai1 blobs now carry (tools/export_chai1_structure_pair.py
reads them out of chai-lab, tools/add_chai1_structure_to_blob.py adds them). Against af3-any-model at f32, every trunk seam
is 2e-7 to 6e-7 on 6MRR and **5.4e-7 on 5CAJ with its crystal as template**; denoise 4.4e-7,
confidence 1e-6. 6MRR from its sequence folds at 0.92-1.93 A over five samples (chai-lab
0.95-1.78); 5CAJ at 2.32 A, 2.28 with its template - the N-terminal tag and the doubly-modelled
165-169 loop, 0.46 / 0.39 A over the best 90%. The CUDA worker folds it (`test:cuda` has four
cases); the page does not offer it, having no WebGPU forward for it yet.

Where this port leaves af3-any-model for chai-lab, measured each time:
- **One denoiser call a step** (sampler.cuh; `--chai-second-order` is chai-lab's two). Same seed and steps,
  the two land 0.02-0.07 A apart at 200 steps with the same bonds on protein, glycerol, ATP, a
  phosphoserine and DNA, for half the diffusion (5CAJ, 5 x 200: 2.82 -> 1.93 s a fold).
- **Atom attention within one reference space, not one token** - chai-lab ANDs its block mask with the
  same-ref-space mask in place before attending. The two agree wherever a token is a residue (6MRR, a
  nucleic acid) and the token rule blinds a ligand's atoms to each other: glycerol 0.29 -> 0.053 A bond
  rms, ATP 0.27 -> 0.058.
- **An atomised residue's tokens are unknown** (gemmi's fasta code, X for SEP): a phosphoserine
  0.21 -> 0.074 A from the int8 blob, inside chai-lab's 0.054-0.089.
- **No token-bond term**: chai-lab's carries declared covalent bonds alone, so an ordinary ligand's is
  zero, and a declared one (`bondedAtomPairs`, a glycan) is refused until its trunk weights are exported.
- The diffusion's pair input (chai's structure token-pair features) and the confidence head's single. The
  first is fixed in af3-any-model as well (its c38fec3); the attention mask and the unknown restype in its
  4ec4850.

`--out=x.pdb`, or `--out=x.cif` for mmCIF as AlphaFold 3 writes it (entities, polymer sequences
and chains declared, so AF3's own reader and gemmi both load it; the pLDDT in B_iso_or_equiv).
`--save-embeddings` and `--save-distogram` write what AF3's `--save_embeddings` / `--save_distogram`
do, as NumPy files beside the structure: `<stem>_single_embeddings.npy` (tokens x 384) and
`<stem>_pair_embeddings.npy` (tokens x tokens x 128) from the trunk's last pass, and
`<stem>_distogram.npy` (tokens x tokens x bins, the head's probabilities).
`--af3-defaults` runs AlphaFold 3's own settings - 10 recycles (11 trunk passes) and 5 samples, as
`run_alphafold.py` does - where the command sets neither; the plain defaults are the page's (3
recycles, 1 sample). Warm, with them: 6MRR 0.73 s (trunk 197 ms, five samples' diffusion 489),
261 tokens 2.69 s (1.20 + 1.20).
`--fold` options: `--steps=200 --recycles=3 --seed=42 --folds=N` (N warm repeats), `--seeds=a,b,c`
(every seed, as AF3 runs a job's modelSeeds - a job's list is used when no seed is given; one trunk
serves them all, its features not depending on the seed, and every (seed, sample) goes through the
denoiser in one batch of up to ten: three seeds of two samples 1.69 -> 1.02 s of diffusion on
barnase-barstar; the files are `<out>_seed<s>_sample<k>.pdb`, the scores AF3's
`<out>_ranking_scores.csv`), `--samples=N`
(N diffusion samples off one trunk, AF3 runs five: each scored by the confidence head and ranked by
AF3's ranking score, 0.8 ipTM + 0.2 pTM (pTM for one chain) + 0.5 fraction_disordered -
100 has_clash - the best written to `--out`, all to `<out>_sample<k>.pdb`). The two structure terms
are AF3's own definitions (src/scores.cuh): has_clash, a polymer chain with more than 100 atoms or
half its atoms within 1.1 A of a non-neighbouring polymer atom; fraction_disordered, the protein
residues whose DSSP accessibility (each chain alone, mkdssp's dot surface reproduced) averaged
over 25 residues exceeds 0.581 of their maximum. Both are exact against AF3's functions on the
structures this port writes (`scores_oracle.py`, run with af3-any-model's environment: every
residue's DSSP accessibility identical over 1,500 residues, a modified residue included, and
has_clash on both sides of its threshold); `af3 - --score-pdb=FILE` scores any PDB. Beside every structure, what the page's archive
writes: `<stem>_confidences.json` (atom_chain_ids and atom_plddts in the PDB's atom order, the
distogram's contact_probs, pae, token_chain_ids, token_res_ids) and
`<stem>_summary_confidences.json` (chain_ids, chain_pair_iptm, chain_pair_max_contact,
chain_plddt, chain_ptm, chain_iptm, chain_pair_pae_min, iptm, ptm, ranking_score,
fraction_disordered, has_clash, mean_plddt). The samples run as ONE batch through the
denoiser - the transformer's GEMMs at 5x the rows, the samples as the flash kernel's batch, what
they share (conditioning, masks, biases) read once - and sample k of seed s draws exactly what a
one-sample run seeded s + k 2^32 draws (sample 0 is the seed's own run; no two samples of distinct
seeds share a stream), so its structure is the same: 5CAJ with its template, sample 1 of seed 42 and
a one-sample run with `--seed=4294967338` are 0.002 A apart, 0.201 A from the crystal each. Five samples' diffusion: **1.33 s against 0.54 s** for
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
| 6MRR, 68 tokens | 5.1 s / 2.4 s | 0.40 s / **0.34 s** | 0.40 s |
| 5CAJ, 261 tokens, 512 MSA rows | 11.4 s / 7.4 s | 0.71 s / **0.61 s** | 0.93 s |
| 5CAJ x 2, 522 tokens, 512 rows | | 1.24 s / 1.18 s | 2.40 s |
| 5CAJ x 4, 1044 tokens, no MSA | | 3.47 s / 3.42 s | 9.24 s |

WebGPU with the developer flags (`fold.js --folds=2`); a stock-Chrome NVIDIA visitor gets
about half its speed. A whole `fold` command - the export, the CUDA context, the weights, the
fold - is 1.0 s for 6MRR and 1.7 s for 5CAJ. No 2 GiB binding ceiling. At 1044 tokens a trunk
pass is 1.97 s - grid attention's flash kernel 38% of it - so recycles dominate there; up to ~300
tokens the 200 denoiser steps do. 2088 tokens: 11.3 s a trunk pass, 2.9 s of diffusion.

Memory: from 512 tokens (a pair over 128 MB) every phase - trunk, diffusion, confidence, and the
next fold - starts from the card it had, its predecessor's scratch given back: 1044 tokens peaked
at 21.3 GB with the trunk's 8 GB of scratch held to the end, and peaks at 14.8 GB (the trunk's
phase) now, byte-identical and no slower. `LOCALFOLD_MEM=1` prints what is in use at each phase,
and the largest scratch buffers.

Against AlphaFold 3 itself - af3-any-model's JAX (bf16, Triton flash attention) with DeepMind's
weights, on this A100, `tools/oracle/bench_af3_native.py` at matched settings (one sample unless
said, no token bucketing, the same MSA rows), steady-state calls:

| 200 steps | JAX AF3 | native `--fast` | |
|---|---|---|---|
| 6MRR, 68 tokens, 1 pass | 1.67 s | **0.34 s** | 4.9x |
| 5CAJ, 261 tokens, 512 rows, 1 pass | 2.76 s | **0.61 s** | 4.5x |
| 5CAJ, 4 passes (3 recycles) | 3.47 s | **0.93 s** | 3.7x |
| 5CAJ, 1 pass, 5 samples | 5.00 s | **1.40 s** | 3.6x |
| 5CAJ x 4, 1044 tokens, no MSA, 1 pass | 11.1 s | **3.42 s** | 3.2x |

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
log2-domain scores) - ~85 TFLOP/s at 1044 tokens, see below. The MSA stack in f16 too. The small attention GEMMs a
token count that is not a multiple of 8 would put on cuBLAS's align-1 kernels - the MSA attention's weights x
values, the single track's scores and P V - take rows padded to 8 (and K with them, the padding zero): AF3 at 261
tokens with a 1024-row alignment, trunk 371.1 -> 367.0 ms; level at 68 and 510.

Diffusion: everything derived from the conditioning computed once per fold (atom pair
conditioning and pair logits, every block's adaptive-LayerNorm scales/shifts and zero-init gates,
the transformer's 24 pair-logit sets as f16 flash biases); the transformer's 144 conditioning
projections as two GEMMs per step with each block's LayerNorm scale folded into its weights
(folded on the device); the whole step replayed as a CUDA graph; the sampler on the device, its
Gaussians a counter-based hash computed where they are used; the atom blocks' keys and values
projected once per atom and gathered; a split-over-keys flash kernel for the transformer's few
blocks at small n, and 48-key tiles for its 48-wide heads (four blocks an SM rather than three: 261
tokens x 5 samples' diffusion 1202 -> 1154 ms); `--samples` batched through the whole denoiser.

Start-up: model.bin mapped and copied to the device in one transfer; every fused weight
concatenated on the device. A first fold is within 5-15% of a warm one.

## The input, natively (cuda/featurise)

`af3-featurise` is `export-model.mjs --no-weights` in C++ - the page's own featuriser, every convention of every
family, written to the same `model.idx`/`model.bin`/`template.pdb` **byte for byte**: the job reader (both
dialects, every refusal in the page's words), CCD components (the RCSB through `~/.cache/localfold/ccd`, or the
job's `userCCD`), SMILES ligands (shared/chem's parser and distance-geometry conformer, with V8's own `Math` so
the conformer is the page's to the last bit), glycans, modified residues and bases, declared bonds, inline
alignments and A3Ms, the MMseqs2 search and its templates (`--search`, `--search-templates`: curl and gzip, no
library), templates (PDB or mmCIF, the page's alignment or the job's mapping) through every family's embedder,
OpenDDE's structural tokens, rf3's stereocentres and chai-1's ESM2 inputs. `tools/check-native-featuriser.py`
holds it to the JavaScript (171 cases over every family, AF3's own example jobs among them; `--network` for the
searches). `export-model.mjs` stays as that reference, and as the tool for the weights and the oracles.

```
../featurise/af3-featurise data-gol --no-weights --family=af3 --sequence=<SEQ> --ligands=GOL     # CCD codes
../featurise/af3-featurise data-sep --no-weights --family=af3 --sequence=<SEQ> --modify=SEP@3
../featurise/af3-featurise data-dna --no-weights --family=af3 --sequence=<SEQ>:GCGATCGC:GCGATCGC --kinds=protein,dna,dna
../featurise/af3-featurise data-smi --no-weights --family=af3 --sequence=<SEQ> --smiles='OCC(O)CO'
../featurise/af3-featurise data-ab --no-weights --family=af3 --sequence=<A>:<B> --a3m=a.a3m,b.a3m [--paired-a3m=pa.a3m,pb.a3m]
../featurise/af3-featurise data-job --no-weights --family=af3 --job=../../tools/fixtures/af3-jobs/kras_g12c_sotorasib.json
node bonds.mjs fold.pdb GOL        # bond lengths by class, ideals from the CCD
```

## Ligands, modified residues, nucleic acids

`--job` reads an AlphaFold 3 job JSON (either dialect) as the page's own reader does
(`web/job-json.js`, `web/entities.js`): chains and their kinds, CCD and SMILES ligands, modified
residues AND modified bases (a base's parent through its chain's alphabet and the component's CCD
parent), a ligand of several components as one chain (a glycan: `ccdCodes` [NAG, NAG, BMA, MAN,
MAN], a residue each), `bondedAtomPairs` (one direction, the job's, covalent - as AF3 lists and
codes them), every model seed (`--seeds`), the job's own templates - up to four a chain as AF3's
data pipeline writes them, chain i's k-th in slot k, with their `queryIndices`/`templateIndices`
mapping when given (a job asking for a template search wants `--search-templates`) - its `userCCD`
components (resolved before the RCSB) - and, unlike the page, the `unpairedMsa` / `pairedMsa`
AF3's data pipeline writes into the job, which become the alignment, an RNA chain's in RNA's
alphabet (5CAJ with 200 inline rows: 2.02 A, pLDDT 94.2). Each of these is exact against AF3's own
featurised batch (`tools/check-batch-fields.js`, targets dna-5cm, rna-mods, glycan, rna-msa,
prot-rna-msa). All fourteen of AF3's example jobs fold, its kitchen-sink `alphafold_input.json`
(every field the format has at once) included: ubiquitin, calmodulin + 4 Ca, KRAS G12C + covalent
sotorasib (SG-C25 1.53 A - bonded), ERK2 with two phosphorylations, streptavidin + SMILES biotin,
the TetR dimer on DNA (476 tokens), U1A on an RNA hairpin, methylated DNA (pLDDT 94.2), a modified
tRNA fragment, glycosylated RNase B (Asn34 ND2-C1 1.42 A).

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

`cuda/af3/fold --batch=<file>` folds every line of `<file>` (`<out.pdb> <input flags>`) in one
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

`cuda/af3/fold --serve` starts one `af3` (`--serve=DIR`) that keeps the weights on the device and
every kernel and cuBLAS plan warm; until `cuda/af3/fold --stop`, every ordinary `fold` command
for that model featurises its input (the native featuriser starts in milliseconds, where the Node exporter it
replaced needed a resident process of its own) and hands it to the server as a job (per-job
`--samples`, `--steps`, `--recycles`, `--seed`) - 6MRR 0.53 s a command, against 1.00 s starting
both each time. The outputs are byte-identical either way.

What a warm job costs around the fold, measured through cuda/worker.py (AF3, 5CAJ, its crystal as the
template, A100): 732 ms job to result, of which the trunk and the sampler were ~410. Two host costs grew as the
square of the length and are gone, byte-identical: the template's distogram crosses as each pair's one-hot BIN
(`template.k.distogramBin`) instead of 39 floats a pair - the export 1.15 -> 0.76 s and 181 -> 30 MiB at 1,020
tokens, its conversion, copy and write having been 0.63 s where the geometry is 0.16 - and the worker passes the
PAE and contact matrices through as the binary wrote them (`flat_matrix`, two-decimal text spliced into the
result line) instead of parsing, rounding and re-serialising 136,000 Python floats (95 of the 167 ms between the
last sampler step and the result). Now 580 ms.

...and the fused template embedder's columns (protenix2, boltz2, rf3) cross sparse - each row's nonzero (column,
value) pairs, `template.k.featuresIdx/Val/K` - scattered back on the device into the dense matrix the projection
reads, byte-identical (boltz2's 5CAJ fold compared against the dense path, file for file). Dense, every pass wrote
108-109 floats a pair, an empty slot's included, whose rows are all one row: protenix2's input at 1,020 tokens was
**864 MB with no template at all** (2.65 s of export) and 1,292 MB with one (4.93 s). Sparse, and an empty slot's
row taken from a two-token input and tiled: 38 MB / 0.50 s and 102 MB / 1.78 s; 54 -> 3 MB at 255 tokens.

## The gate

`python3 cuda/af3/gate.py` folds 6MRR from its sequence through all seven models, plus 5CAJ and
barnase-barstar with their crystals as templates, scores each against the deposited structure and
holds RMSD and mean pLDDT to `gate-baseline.json` (0.05 A, 0.5 points; `--write` re-records,
`--only=` takes models or case names). Run it after any kernel change; it takes about three
minutes. The baseline is this A100's, on the published int5 bundles (the float32 ones gave
af3-6mrr 0.515 A / 87.57 where int5 gives 0.550 / 87.39; the kitchen sink 79.04 against 73.47).

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

With `LOCALFOLD_UNREAD=1`, after a first fold `af3` lists every weight family it never read; for these models the list is
only what should be there (heads not computed, alternative per-block forms, absent bonds and
template geometry) - rf3's atom-block q/k norms and chirality term were found by it.

## On a T4 (Turing, sm_75)

Profiled on a Colab T4 (2026-10-03), a 262-token fold at the page's settings is 4.0 s against 0.57 on the
A100, 91% trunk; 524 tokens, 18 s. The T4 has no `cp.async` and 64 KB of shared memory a block, and the
grid attention's flash kernel, written for Ampere, ran at **3.5-6 TFLOP/s** there: its "async" copies
become synchronous loads that stall each warp before every tile, and its two stages (39 KB at D 32) let
one block an SM. `flashGridHalf<..., REG = true>` - taken on any device before Ampere
(`LOCALFOLD_FLASH_REG=0|1` to measure either form anywhere) - stages the next tile in registers while
this one computes and stores it into a single buffer, three blocks an SM: **2.64 -> 1.23 ms at 262
tokens, 11.9 -> 8.2 at 524**, every output identical (`--bench-grid` compares the two forms, masked and
not: 0 of 8,786,432 differ), whole folds 3.99 -> 3.73 s and 18.1 -> 17.2. On the A100 it would be 0.245
against 0.184 ms, so Ampere keeps `cp.async`.

AF2's strided copy of the kernel (cuda/af2/src/flash2.cuh) takes the same form there. Its fold with 512
alignment rows at 262 residues moved only 3% (12.25 -> 11.84 s, identical), and the profile says why: the
two flash kernels fell 2.91 -> 1.99 and 2.39 -> 1.69 s while cuBLAS's GEMMs ROSE 2.21 -> 2.94 - **the T4 is
power-capped (70 W)**, so a kernel that keeps the card busier lowers the clock for everything else. On a
T4 what pays is less work, not better-overlapped work.

Tried on the T4 and not taken: more blocks an SM for that kernel (`__launch_bounds__` minimums of 4-8,
32-key tiles, 2-warp blocks: every one slower - it is not short of warps), its prefetch loads pinned in
place with volatile asm (no change), and the grid attention unfused (`LOCALFOLD_UNFUSED=grid`; also
`triangle`, `transition`): level at 262 tokens, 4% slower at 524. The T4 drifts up to 7% between repeats
of one arm (it throttles to ~1 GHz under load), so interleave arms there.

## On an L4 (Ada, sm_89, 99 KB of shared memory a block)

Profiled on a Colab L4 (2026-10-03): a 262-token AF3 fold was 2.03 s, and the triangle multiplications
were 53% of its trunk - the fused triangle's output kernel needs 104-139 KB at 8 warps, so the L4 ran the
whole 128-channel lineage on the UNFUSED triangle (its gate, add and norm passes 31% of the fold). The
output kernel now takes 4 warps where 8 do not fit (71 KB): **1.98 -> 1.54 s a fold on the L4**, the A100
byte-identical (8 warps fit there). And the flash kernel's 64-key tiles were the wrong size on both cards:
48-key tiles (smaller stages, more blocks an SM) are 0.566 against 0.747 ms on the L4 at 262 tokens, and
on the A100 1.036 against 1.211 at 524, 7.23 against 7.80 at 1044, level at 262 - AF3's trunk at 1048
tokens 8.11 -> 7.87 s. The T4's register-staged form keeps 64 (48 is slower there).

## Kernels the three ports share

cuda/af2 and cuda/ef2 include this directory's headers, and where two ports had written the same kernel
it now lives here once:

| kernel | was | now |
|---|---|---|
| the fused triangle multiplication (`triInK`, `triOutPK`) | AF3 only; AF2 had its own unfused path | AF2's 128-channel triangle too (a `BIAS` option, its weights re-laid once): 5% of an AF2 fold with an alignment |
| the fused transition (`fusedTransitionK`) | AF3 only | AF2's pair transition too (a `RELU` form with biases) |
| the 256-channel fused kernels (`fused256.cuh`) | ESMFold2 only | protenix2 too: 6.4% (A100), 20% (L4) |
| the flash grid attention (`flashGridHalf`) | AF2 had a 200-line copy with strides (`flashStrided`) | one kernel, a `STRIDED` form - AF3's dense form unchanged |
| the outer product mean's permute and residual | both ports | shared; AF3 takes AF2's 16-byte permute (9.9 -> 3.3 ms) |
| eight element-wise kernels (`elementwise.cuh`) | a copy in each port | once |

On a Colab L4, this round against the commit before it (684a04f), two interleaved rounds at 262 tokens: AF2
with 512 alignment rows 6.76 / 7.11 -> 6.44 / 6.68 s (5-6%: the fused triangle and transition reach it
there), ESMFold2's trunk 1067 / 1089 -> 1052 / 1074 ms and 4271 / 4351 -> 4206 / 4299 at 524, AF3 and
protenix2 level within the card's ~3% drift.

In every case the side that already ran the shared kernel is byte-identical. Measured and left as they are:
AF3's LayerNorm against the vectorised `layerNormVK` (both ~1.2 TB/s, already bandwidth-bound, and AF3's
uses AF3's own variance formula), and ESMFold2's float32 token attention against the f16 flash kernel (its
sampler is 76 ms at 262 tokens and 101 at 524, so the T^2 attention is not what it spends). And the grid flash
attention reading its pair bias straight from global into the score registers instead of staging it (the
staged bias tile is as large as K and V together, and a head's bias is one n x n shared by every row, so it
is hot in L2): bit-identical, half the shared memory a stage, and **about twice as slow** - 0.182 -> 0.374 ms
at 262 tokens, 1.032 -> 2.062 at 524, 7.23 -> 17.1 at 1044 (`--bench-grid`, 4 warps, 48-key tiles; 64-key
tiles and 8 warps recover some and still lose). The loads land in the scores' dependency chain, where the
staged copy is hidden behind cp.async; occupancy was not what this kernel lacked.

## The wider pair tracks

protenix2 (256 channels), OpenDDE (384) and IntelliFold-2 (512) are 2.4x, 4x and 5.2x AlphaFold 3's trunk
at 262 tokens - no more than their widths predict (attention heads 8, 12 and 8 x 64 against 4; projections
by the width squared) - but their unfused path spends about a third of it in LayerNorm, gate, SwiGLU and
add passes. ESMFold2's port already had fused kernels for a 256-channel track that stream their weights
in narrow steps (two blocks an SM where cuda/af3's 128-channel ones would need 255 registers); they now
live in src/fused256.cuh and protenix2 takes them from 80 tokens - triangle in, the f16 contraction,
triangle out, and the transition's widening ahead of cuBLAS's second GEMM: **trunk 1259 -> 1180 ms (6.4%)
at 262 tokens**, templated 5CAJ 0.191 A either way (`test:cuda` holds it). At 384 and 512 channels the
same kernels fit one block an SM and LOSE (2160 against 1925 ms, 3360 against 2821), so those stay unfused.
On an L4 they pay more than on the A100, at one block an SM: protenix2's trunk **5.13 -> 4.12 s** at 262
tokens, ESMFold2's (whose they were) 1.60 -> 1.07.

The triangle-out kernel of the set was latency-bound (Nsight Compute: 12.5% occupancy, tensor pipes 8% busy,
the warps waiting on global loads) - its float tile of the product, 66.5 KB, fitted two 4-warp blocks an
SM. It holds the product in bf16 now (AF3's own activation precision) and takes the output weights 16
columns a stage: 34 KB, four blocks (the registers' limit), and its load loop keeps 32 loads in flight
rather than 8. ESMFold2's trunk 326.8 -> 313.1 ms at 262 tokens and 1114 -> 1036 at 524; 5CAJ 2.100 ->
2.101 A (ESMFold2) and 0.190 -> 0.191 (protenix2 templated); `LOCALFOLD_TRIOUT_F32=1` is the float tile.

The triangle-out kernel's centre norm read its scale and offset from global memory a float at a time -
128 scalar loads a thread a tile, four times the loads of the residual and gate it exists to stream, and
Nsight Compute's top stall (lg_throttle) on a persistent kernel holding one block an SM. Read once into
shared memory: `--bench-tri` 0.1185 -> 0.1089 ms at 261 tokens and 1.416 -> 1.326 at 1044, byte-identical,
and at 1044 the kernel now moves ~1.27 TB/s of its 1.55 - memory-bound, so what is left there is bytes.
The input kernel's LayerNorm keeps its lane's scale and offset in registers for all its rows (2% of
`triInK` at 261, flat at 1044; at 1044 it runs ~117 TFLOP/s beside ~0.9 TB/s). A warm templated 5CAJ
fold's trunk 320.4 -> 314.5 ms, the PDB byte-identical.

The unfused grid attention (every pair width but 128: Chai-1, protenix2, OpenDDE, IntelliFold-2) wrote its
column direction's output projection to a float temporary and added it into the pair TRANSPOSED in a
pass of its own (`addGridK`, 150 us a call at 255 tokens). A strided-batched GEMM adds it there itself
(attention row r's token j is pair (j, r): ldc = n*C, a block stride of C, beta = 1), and Chai-1's
parallel block, whose ending-node residual is untransposed, takes an ordinary beta = 1 GEMM. Warm folds
at 255 tokens, byte-identical: Chai-1's trunk 573.7 -> 557.0 ms, protenix2's 798 -> 778, OpenDDE's 1654
-> 1623. `LOCALFOLD_GRID_STRIDED=0` is the gathered-and-scattered arm. The same trick on the INPUT side
(reading the normed pair transposed in place, no gather) loses: 41.6 against 35 ms for the gather and
one GEMM.

The same path's LayerNorm and pair-bias projection are one kernel at 256, 384 and 512 channels
(`lnNormHeadsK`): the pair normed with `layerNormK`'s own arithmetic (so the normed pair is byte-identical)
into global memory and shared memory, then projected on the tensor cores to 16 padded heads. cuBLAS had
taken the few-column projection as a 16x16 WMMA kernel re-reading the whole normed pair (11.7 ms of a
Chai-1 fold). Trunks at 255 tokens: Chai-1 556.9 -> 547.9 ms, protenix2 777.7 -> 766.0, OpenDDE 1622 ->
1604, every PDB byte-identical; `LOCALFOLD_LN_NORM_HEADS=0` is the old pair of kernels. It is slower than
the plain norm it absorbs (88 against 83 us a call - four 4-warp blocks an SM where the norm runs full),
so it pays only the GEMM's difference: eight rows in flight a warp took it from 102 us, sixteen and eight
warps no further.

In the column direction the same kernel writes the normed pair TRANSPOSED (row (i, j) at (j, i)), which is the
layout the gather pass then produced - so the gather and one full read and write of the normed plane are gone
(2026-10-07). Warm trunks at 261 tokens, byte-identical: OpenDDE 1413.9 -> 1403.9 ms, IntelliFold-2 2241.5/2249.0
-> 2232.2/2239.4; `LOCALFOLD_GRID_NORM_T=0` is the gathered arm. The rest of item "a fused 384/512 grid input"
is NOT worth a kernel: the q/k/v/gate GEMM must read the normed plane either way, and cuBLAS already takes it
at full rate - an LN-prologue GEMM (triingemm.cuh's main loop) would re-implement the same GEMM.

At 384 channels (OpenDDE) the unfused triangle's projection and the unfused transition run every row in one
pass where the card has the room (`roomFor`, 0.2 and 0.6 GB at 255 tokens), and the triangle's gate reads
each channel's two halves in one load: `triGateK` 156.6 -> 140.8 ms a fold, the trunk 1600 -> ~1560 ms,
atoms identical. Chunks small enough for L2 to hold the transition's widening (1-8k rows) LOSE, 397-615
against 370 ms - the GEMMs shrink faster than the elementwise passes speed up.

Both triangle input kernels (`triInK` at 128 channels, `triIn256K` for ESMFold2 and protenix2) take the
gating linear TWO tiles a step, one in each weight stage - the second stage idled in those steps, and
Nsight Compute had `triIn256K` waiting on its MMA chains (stall_wait 25%, the HMMAs accumulating into one
register back to back): twice the independent chains and half the steps. Byte-identical:
`--bench-tri` `triInK` 0.1205 -> 0.1134 ms at 261 tokens, 1.514 -> 1.403 at 1044; `triIn256K` 68.6 ->
64.2 ms over an ESMFold2 5CAJ fold. And `triangleOutK` stages its centre norm's scale and offset in shared
memory (2 KB; four blocks an SM still fit): 0.2414 -> 0.2353 ms at 261, 3.204 -> 3.095 at 1044, exact.

A cold start (the first fold after a model is chosen) was mostly PINNING, not reading: `bundleUp` read each
shard into one of two shard-sized pinned buffers, and on this A100 host 2 x 256 MB of `cudaHostAlloc` is
0.4 s to allocate and 0.16 to free, per bundle - where reading AF3's 367 MB from the page cache is 0.04. A
ring of four 4 MB pinned pieces, kept for the process, now feeds each shard's device buffer (resident codes
copied from it device to device). Byte-identical; whole cold runs at 255 tokens: AF3 1.72 -> 1.20 s,
Chai-1 (two bundles, ESM2 3B's among them) 2.84 -> 1.67, ESMFold2 (16 MB shards, so it pinned little)
1.10 -> 1.05. A first try at four 16 MB pieces made ESMFold2 0.1 s SLOWER: it pinned more than before.

`triangleOutK` (the 256-channel triangle output: ESMFold2, protenix2) loads each chunk's residual and gate
before the chunk's MMAs rather than after them - Nsight Compute had long scoreboard at 39% of its stalls,
on those loads: `--bench-tri` 0.2357 -> 0.2138 ms at 261 tokens, 3.095 -> 2.906 at 1044, byte-identical.
(Issuing them at the very top of the iteration, ahead of the stage wait, is level.)

On a T4 (64 KB of shared memory a block) the 128-channel triangle ran unfused - a LayerNorm, the
projection GEMM, the gate pass, the contraction, a centre norm, two GEMMs and the gated add - because the
fused output kernel holds the whole 128 x 128 weight (71 KB). It takes `fused256.cuh`'s two kernels now,
instantiated at 128 channels, which stream that weight 16 columns a stage (~35 KB), with the contraction in
f16 into f32 where the device has no bf16 MMA (`bf16Tensor`). Measured on a Colab T4, boltz2 at 255 tokens,
arms interleaved: the trunk 3.38-3.51 -> 3.02-3.08 s, a fold ~3.95 -> ~3.55 s (pLDDT 32.04 -> 32.05: other
kernels, not other arithmetic in kind). Rehearsed here under `LOCALFOLD_SMEM_LIMIT=65536 --no-tri-bf16`,
where the trunk is 425.7 -> 331.6 ms. The A100 and an L4, where the 128-channel kernels fit, are unchanged.
The 256-channel triangle (Chai-1, protenix2) takes the same kernels on a T4 too, its input kernel at 4 warps
where 8 do not fit (byte-identical per row; the stages then outgrow the rows, so the launch sizes the larger).
Rehearsed here it looked like 15% (protenix2's trunk 977.7 -> 832.2 ms, Chai-1's 682.7 -> 580.8); ON A REAL
T4 it is 1.5-2.5% (protenix2 at 255 tokens, interleaved: 6960 -> 6857 and 7538 -> 7352 ms, the card drifting
8% between rounds). The rehearsal caps shared memory, not a T4's arithmetic or its f16-into-f32 contraction:
read a T4 number off a T4.
On a T4 at 256 channels the input kernel ran 4 warps holding a whole SM (its rows and stages, 34 KB, of 64);
it now runs 16 warps that LayerNorm their rows 32 at a time into a buffer of their own (`XROUNDS`), each
round's warps taking their fragments before the next - one round's rows beside the stages, 57 KB, every row
normed by the same arithmetic. Byte-identical; on a Colab T4 protenix2's input kernel 1768 -> 1477 ms over a
fold. NOT occupancy-bound after all: the whole fold moves less than the card's thermal drift, and at 128
channels against the 8-warp form it is 428 -> 420 - so it is taken only where 8 warps do not fit.
ESMFold2 on a T4 ran its whole 256-channel pair track UNFUSED: `fused256Fits` asked all three of the
8-warp kernels to fit, and none does in 64 KB. It now takes their T4 forms - the rounded input kernel, the
output kernel's bf16 tile, and `transitionUpK` in rounds with 16-column stages (`transitionUpK<256, 16, 16,
8>`, ~49 KB) - with its contraction in f16 where there is no bf16 MMA. Colab T4, 5CAJ: the trunk 2367/2385 ->
1622/1647 ms (-31%), and under `LOCALFOLD_SMEM_LIMIT=65536` here it is byte-identical to the A100's fold. The
same transition form for the AF3 lineage measured level there (protenix2's 1083 ms against ~1100 for its
LN, GEMM and SwiGLU passes) and is not taken.
AlphaFold 2 on a T4 ran its triangle unfused for the same reason. `triIn256K` and `triangleOutK` take AF2's
biases now (`BIAS`), and at 128 channels with AF2's f32 product the output kernel keeps a float tile (~34 KB):
the bf16 tile read the update 2e-3 off AF2's own kernels, the float tile is within its f16 operands' rounding.
Colab T4, 5CAJ with 512 + 1024 alignment rows, interleaved: 11027/11284/11770 -> 10640/11021/11475 ms
(-3%), pLDDT and pTM unchanged.




A T4 has no `cp.async`, so `cpAsync16` there is a load and a store - and every streaming kernel's "issue the
next stage, then compute this one" blocked on the issue: a memory round trip a step, exposed. The four that
run on a T4 (`triIn256K`, `triangleOutK`, `fusedTransitionK`, `gridInK`) now hold the next stage in registers
across the step (`RegStage`, `LF_REG_STAGES`: loaded before the MMAs, stored into the idle stage after them,
the barrier that opens the next step publishing it). Byte-identical on the T4 and, through a compute_75 PTX
build, here. Colab T4, 255 tokens, interleaved: boltz2's trunk 2889/2937 -> 2689/2727 ms (-7%), protenix2's
6349/6536 -> 6149/6489 (-1 to -3%: its 256-channel input kernel at 4 warps is bound elsewhere). The same
sm_75 code JIT-compiled on the A100 is 16% faster for boltz2 - the A100's own sm_80 path is unchanged.

A T4's unmasked 32-wide grid attention - AF2's MSA row and column attention, the AF3 lineage's pair track -
runs `flashGrid2R` in a ONE-stage form: its two query tiles a warp (each K/V fragment feeding both) at the
footprint of the register-staged kernel it replaces (~19.5 KB: three 4-warp blocks an SM), the next key tile
held in registers across the compute and stored between two barriers. The two-stage form was slower there
(one block an SM; below). Nsight Compute on the T4 had the old kernel latency-bound - 0.27 instructions a
scheduler a cycle, 37.5% occupancy. Colab T4: AF2 5CAJ 10806/11156 -> 10336/10754 ms (its flash kernels
3682 -> 3266), boltz2's trunk 2815/2867 -> 2775/2813; 64-key tiles no better. Byte-identical to the A100's
two-stage kernel (it is the same arithmetic), and AF2's column attention no longer builds a zero bias for it.
`LOCALFOLD_FLASH_2R1=0` is the old kernel.

The pairformer's pair in bf16 (AlphaFold 3's own activation precision), phase one: where every update a stack
runs has a bf16 form (`pairBf16Ok`: the 128-channel fused triangle - or the T4's streaming one - the fused grid
attention and the fused transition), the trunk converts its pair to bf16 before the 48 blocks and back after, and
those kernels take a pair element type (`PT`, `PAIR16`, `WITH_PAIR_T`); every path without one calls `needF32Pair`
and refuses rather than misread. The row-direction grid output's GEMM writes f16 and one pass adds it (cuBLAS has
no f16-in, bf16-out GEMM; `gridOutK` there was slower on a T4). AF3 at 255 tokens on the A100: trunk 312.4 ->
302.5 ms (-3.2%); boltz2 on a Colab T4 2566/2597/2628 -> 2498/2523/2560 (-2.7%). The gate's AF3 RMSDs move
in the third decimal (0.552 -> 0.553, 0.168 -> 0.170, 0.541 -> 0.536). `LOCALFOLD_PAIR_F32=1` keeps the f32
pair. Phase two holds it in bf16 for the whole trunk (`usePair16`, `pair16Eligible`): `t.pair` and the recycled
pair are allocated at half the bytes, the embedder, the template stack's boundary, the MSA stack (outer product,
attention's pair LayerNorm, its pair updates), boltz2's pre-MSA add and the distogram read and write it in bf16
(`layerNormPairRows`, `linearIntoPair`, `WITH_PT`), and it converts to f32 once when the trunk is done, for the
heads, the sampler and the confidence head. The big-input paths take it too now (the blocked triangle's two pair
kernels, in-place recycling, the chunked grid and single-track offsets, the pair parked at its own size). Measured
for the ceiling by holding the card from a second process (`LOCALFOLD_NO_FOLD_FITS=1`), 1530 tokens: bf16 and f32
both fold with 7.75 GB free and both fail at 7.5 - 🔴 THE MEMORY CEILING DID NOT MOVE, because what binds there is
the template stack, which runs with the trunk's pair parked in host memory (its own 64-channel f32 activation and
unfused triangle buffers); raising it means shrinking that stack. (Its bf16 chunk held into the next pass's
template stack first made bf16 WORSE - 7.75 failed - until it was given back.) What it does do under pressure: the
trunk 29.9 -> 24.3 s at 7.75 GB free (-18%); 3% where memory is plentiful. Then the template stack itself was
shrunk: near the card's limit (`TIGHT_STACK`) its triangles take the blocked form, `pairUpdates(..., releaseBetween)`
hands each update's scratch back before the next (`tri.`/`trib.`, `grid.`, `tr.`), and the embedder's
`emb.prevln`/`emb.prevproj` and the bf16 chunks go before the stack starts. 1530 tokens now folds with **7.0 GB free in
f32 and 6.5 in bf16** (was 7.75 both); f32 fails at 6.5 in `trib.t2`, and at 6.0 both fail in the blocked triangle
(`trib.prod`/`trib.t2`), the next thing to shrink. On a simulated T4 (14.6 GiB, `LOCALFOLD_SMEM_LIMIT`) the template
stack is not what binds: 3570 tokens folds in bf16 (trunk 262 s, before and after) and f32 runs out in `trib.a`;
4080 fails in `trib.prod` either way. RMSDs on all three gates unchanged. Then three more releases on a card short of
room: the target_feat atom encoder's buffers (`targetFeat.encoder.`, `enc.`, `apl.`, ~0.5 GB at 1530 tokens - nothing
reads them once target_feat is built) before the trunk, each pair update's scratch between updates in the MSA stack and
the pairformer too (`releaseBetween`), and the conditioning's chunk buffers before the diffusion's encoder and decoder
prepare (they ran out beside `enc.tp` and `dt.pn16`). 1530 tokens now folds with **6.0 GB free in both** (bf16 trunk
29.8 s, f32 35.5 s); at 5.5 the sampler's own step buffers (`dt.flat`) are what is short. 🔴 THE PROTOTYPE PROMISED 5% AND 14% - it moved the right bytes through garbage values and kept the row
output on cuBLAS, which only garbage allows; the kernels are partly latency-bound, so halving their bytes is
~15% of each, not half.

...and the wider tracks too - protenix2's 256 channels, OpenDDE's 384, IntelliFold-2's 512 (`pairBf16Ok`), on Ampere
and later and not on a card short of room: the streaming triangle and the unfused one (`lnPairRows`, a typed
`gatedAddK`), the 256-channel transition (`transitionUpK` on a bf16 pair, writing its gated rows in bf16 so the
second GEMM runs bf16 throughout and accumulates straight into the pair - `Wbf`, a weight's bf16 copy) and the
unfused one, the unfused grid attention (`lnNormHeadsK`, the row output through an f16 product and
`addHalfToBf16K`, the column output through a typed `addGridK`), and the single track's unfused pair logits. A100 at
261 tokens: protenix2's trunk **749.5 -> 712.2 ms (-5.0%)**, OpenDDE's **1566 -> 1510 (-3.6%)**; the gate's rows move
in the third decimal and OpenDDE's 6MRR 1.08 -> 0.949 A, inside its seed band. What is left of the bytes it saves
goes to the add passes where cuBLAS cannot write a bf16 pair from f16 operands (41 ms of protenix2's 805). 🔴 ON A
COLAB T4 IT IS LEVEL OR SLOWER (protenix2's trunk 5654/6045/6589 -> 5725/6102/6602 ms, OpenDDE's 14811/14693 ->
14603/14694), so it is off below sm_80; the 128-channel track keeps it there (boltz2 -2.7%).

...and the streaming triangle (`triIn256K`, `triangleOutK`) at 384 and 512 channels too, where OpenDDE and
IntelliFold-2 ran the unfused one - the LayerNorm, a [C, 4C] GEMM, the gate, the centre norm, two more GEMMs and a
gated add. The kernels were written for any width and taken only at 256; 384 spills nothing that matters (255
registers, 60 bytes, the 8-warp input form, ~100 KB) and 512 spills more and still wins. A100 at 261 tokens with a
1024-row alignment: OpenDDE's trunk **1688 -> 1544 ms (-8.5%)** (1507 -> 1339 single-sequence), IntelliFold-2's
**2649 -> 2453 (-7.4%)**; 5CAJ's CA RMSD the same to the third decimal on three seeds either way (1.775 / 1.896 /
1.836 A), IntelliFold-2's 2.057 against 2.055. On a Colab T4, which takes OpenDDE's in its 4-warp form (512's does
not fit 64 KB there): trunk 10787/11288/11676 -> 10404/10874/11414 ms. `LOCALFOLD_NO_WIDER=1` keeps the unfused
triangle. 🔴 THE GATE CANNOT SEE IT: OpenDDE's case is 6MRR, 68 tokens, under the streaming triangle's 80.
And the 256-channel fused transition (`transitionUpK`) at those widths too, which an older measurement had losing
(262 tokens, the triangle and transition together: 2160 against 1925 ms, 3360 against 2821 - the kernels have
changed since): OpenDDE's trunk **1529 -> 1477 ms**, IntelliFold-2's **2455 -> 2327**, on top of the triangle, RMSDs
within 0.005 A on two seeds each. ~100 KB a block at 8 warps, so a T4 keeps the unfused one; `LOCALFOLD_NO_WIDER=1`
restores both. The triangle's input kernel takes 4 warps past 256 channels, where 8 warps' rows are one block an
SM (byte-identical; OpenDDE 1476 -> 1449 ms, IntelliFold-2 2323 -> 2301, level at 256). OpenDDE's trunk from 1688 to
~1450 ms and IntelliFold-2's from 2649 to ~2300 in all (-14%, -13%).

...and Chai-1, the last 256-channel model on an f32 pair: its parallel pair block keeps the block's input at the
pair's own width (`par.base`, which the single track must ask for at that same size or it is reallocated and
lost), its untransposed column residual goes through the f16 product and add, and its first pass's z_init
recycle and relative encoding take the bf16 pair. Trunk **535.3 -> 496.4 ms (-7.3%)** at 255 tokens; the gate's
three Chai-1 rows move in the third decimal.

**A new kernel for the wide triangle's input side** (`triingemm.cuh`, 384 and 512 channels, Ampere on): the LN'd rows
written once as f16 (`lnPlaneK`), then a tiled GEMM against the [projection | gate | gating linear] weights packed
into 128-column tiles whose warps hold 32 projection columns and their gates, so the gating and the mask happen in
registers and a and b leave through a per-warp transpose. cuBLAS's own choice for the bare GEMM is the same shape
(128 x 128 blocks of 4 warps, 2 an SM, 32-deep k), and three things took this one from slower than triIn256K to
faster: the fragments double-buffered in registers (the helpers are `asm volatile`, so the order written is the
order issued), every swizzled address reduced to a per-thread base plus an XOR or add with a constant over a fully
unrolled k loop (the generic form spent a third of its issue slots on integer arithmetic), and the masks read before
the main loop rather than after it. Nsight: tensor pipe 68% active against triIn256K's ~50% and cuBLAS's 76% on the
bare GEMM, 65M instructions against cuBLAS's 83M. **Byte-identical to triIn256K** (the same f16 rounding of the LN'd
rows, the same k order in every f32 accumulation), on three seeds of OpenDDE and IntelliFold-2. Trunk at 261 tokens:
OpenDDE **1453 -> 1422 ms**, IntelliFold-2 **2309 -> 2261**; at 256 channels its LN pass costs more than the GEMM
saves (protenix2 713 -> 719), so it starts at 384. `LOCALFOLD_TRIIN_GEMM=0` keeps triIn256K. A resident-rows form
(a block's 64 LN'd rows held in shared memory, warps of 32 x 64) was written first and lost: 50 KB of rows a block
holds the SM at 8 warps, and it ran 4-18% slower.

### A T4's own profile (2026-10-07)

Measured on a Colab T4 rather than simulated, AlphaFold 3 at 261 tokens (alignment, template) folds in 3.49 s warm and
at 510 in 11.9 s; Boltz-2 at 510 in 12.6. The time is the pair track's fused kernels - the grid attention's flash
kernel 26%, the fused transition 16%, the grid attention's input 13%, the triangle's input 12% and output 8% - and the
unfused forms lose on every family there too (Boltz-2 at 510: 16.3-17.3 s with `LOCALFOLD_UNFUSED=transition`,
`grid` or `triangle` against 12.3-14.5 fused, the T4 throttling between runs). **One kernel was a defect, not a
cost**: the confidence head's `reembedPairK` took each pair's distance bin per ELEMENT - a double-precision sqrt and a
63-step double comparison loop, 128 times a pair - on a part whose FP64 is 1/32 of its f32: **252 ms of one launch at
510 tokens**. The bin is taken once a pair now (`reembedBinK`, the same double arithmetic): byte-identical, Boltz-2's
confidence head on the T4 628 -> 379 ms.

**The fused transition's form follows the blocks an SM holds.** Nsight Compute on the T4: the two-tile form (4 warps x
two 16-row tiles, 59 KB) runs at 255 registers and one block an SM, 12.5% occupancy. Neither alternative helps there -
8 warps x one tile is 0.7-1.5% slower and 4 x one tile 4-7% slower, interleaved over three throttling rounds - because
two tiles feed each weight fragment to two MMAs, which a part with a single resident block needs. On an A100, which
holds three of the one-tile form's 42 KB blocks, the one-tile form is 1.6% of the trunk faster at 261 tokens and ~0.4%
at 1,000, bit-identical; it is taken where an SM holds three (an L4 holds two and is unmeasured).
`LOCALFOLD_FT_FORM=1/2` forces one or the other.

## Tried and not taken

- **Rewriting the T4's pair-track kernels, the measurements that came first** (2026-10-07, a Colab T4, AlphaFold 3 at
  510 tokens). Each fused kernel against the unfused passes with the bf16 pair kept (`LOCALFOLD_SKIP_FUSED`; the
  older `LOCALFOLD_UNFUSED` also takes the pair to f32, which is what made its arms look so slow): the transition
  1882 ms fused against 2400, the triangle 3226 against 4603, and the grid attention **a tie** (5459 against 5464) -
  half the T4's trunk either way, most of it the flash kernel at ~9 TFLOP/s. Nsight Compute there: tensor pipe 37%
  active, the transcendental pipe 36%, stalls spread over fixed-latency waits, the shared-memory queue and global
  loads, the issue slots 12% used - latency-bound at two warps a scheduler (195 registers, two blocks an SM). Two
  ideas measured dead: the emulated k16's two dependent k8 halves are NOT issued back to back (`asm volatile` or
  not, ptxas interleaves them - identical schedules, 108 HMMAs a tile, none waiting on its neighbour); and the
  register-staged form's shape, swept with `LOCALFOLD_BENCH_ONE=1 --bench-grid=N`, where 64-key tiles win the bench
  by 8% at 510 tokens and lose or tie in the fold (9.43/10.45/10.85 s against 9.76/11.10/10.79, and slower at 261).
  The shipped forms are the best of their family there; what would move the T4 is a different algorithm, not a knob.
- **Research: what the T4's grid-attention flash kernel spends its time on** (2026-10-07, five Colab rounds; the
  `ABL`, `S32`, `PREF` and `MINB` forms named here were taken back out of the kernel afterwards - commit 0947029 has
  them, with their `--bench-grid` arms). Bench-only ablations at 510 tokens: the
  whole kernel 6.33 ms, without P V 5.20, without Q K^T 4.88, without the max and exponentials 5.60, the MMAs alone
  (no softmax, no tile loads) 3.77 - which is itself ~3.6x off the tensor peak, so the units are fed, not starved of
  arithmetic: ~6 bytes of shared memory a score (bias 2, K 2, V 2) on a part with half an A100's shared bandwidth an
  SM. Three designs built on that and measured in folds, none taken: **f32 scores** (`S32`: the max and exponentials
  in f32, no f16<->f32 round trips - __hmax2 and ex2.f16x2 are both emulated through f32 on sm_75), level everywhere;
  **four query tiles a warp** (K and V shared over 64 rows, ~4 bytes a score; 255 registers, no spills at 32-key
  tiles), 11-15% in the bench at 510 and level or slower in folds (shipped fastest in eight of nine size/round
  pairs); and the register-reduced forms below. 🔴 **AND ONE ABLATION LIED**: "no max" read 6.8x faster - past the
  T4's peak - because Colab's CUDA 13 mis-parsed an `else` followed by `#pragma unroll` (CUDA 12.2 here compiled it
  fine); braced, the same arm is as slow as the whole kernel. A number past the hardware's peak is a broken
  measurement, not a discovery. Also found and fixed on the way: the register-staged form stages its output in its
  one stage's memory, which four tiles overflow (an illegal access) - it is sized to the larger of the two now.
  The T4 drifts 30% across rounds (9.0 -> 11.9 s for one fold), so a kernel win under ~10% cannot be seen in its folds.
- **A register-reduced flash kernel for the T4** (2026-10-07; `PREF`/`MINB` on flashGrid2R, since removed - commit
  0947029). Without the register-held prefetch the shipped shape compiles to 159 registers
  (from 195: three blocks an SM instead of two, no spills, bit-identical), and 4 warps of one tile with 32-key tiles to
  80 (four blocks, every warp slot). **Occupancy was not the limit**: in AlphaFold 3 folds on the T4, three rounds
  interleaved, all within ±1% (510 tokens 10.92/11.06/11.16 s and 10.98/11.33/11.39 against 11.75/10.98/11.05; 261
  tokens 3.107/3.098/3.084 against 3.117/3.104/3.100), while the bench put the leaner forms 12-25% behind at 1000
  tokens - the exposed loads and the extra warps' traffic cost what the hidden latency gives.
- **OpenDDE's refiner and confidence stacks on a bf16 pair** (2026-10-07; the structural pair converted around their
  eight pairformer blocks, the refiner's f32 copy given back while they ran): **no gain at any size** - 15.59 against
  15.69 s at 765 residues, 74.38 against 74.30 at 1450, the peak 30.75 GB both ways (it is the diffusion's, holding
  the structural pair for the head after it). At 384 channels the bf16 pair is worth ~2% of a pairformer block and
  these eight are a fifth of the fold; PAE moved 0.03 A on average for nothing.
- **Four of the 2026-10-07 list, measured first and declined** (each against the number that decides it):
  the **denoiser transformer at small sizes** - its block is 67 us at 261 tokens, 46 of it four GEMMs cuBLAS runs at
  ~85 TFLOP/s on 272 rows with its first heuristic choice already the best, and every attention and adaLN variant
  `--bench-ops` lists (block, warp, split, 32/48/64-key tiles, 1-8 warps) already measured with the shipped one the
  fastest; SwiGLU and the two adaLNs are 9 us of it. The **outer product mean's output with a fused epilogue** - the
  whole OPM is 22 of a ~500 ms stage-synced trunk (4.4%), and on the bf16 pair cuBLAS has no f16-in, bf16-out GEMM
  to take the residual, so the permute and add it would remove are ~1%. **bf16 on the big-input paths** - with the
  free-memory line a wide model takes them only past ~900-1100 tokens on 40 GB, and where both arms were measured
  the bf16 pair is 2% of OpenDDE's trunk (765 tokens: 12.77 s against 13.0 with `LOCALFOLD_PAIR_F32=1`).
  **Template pair features on the device** - the export is 0.50-0.56 s at 1,020 tokens since the sparse fused
  features, and moving it would put a second featuriser beside the page's, which this port exists not to do.
- **A row-resident flash attention for the grid attention** (2026-10-07; one block a (row, head) loading all of its
  keys and values once - 37 KB at 261 tokens - and its warps walking the query tiles with no barrier; the arithmetic
  flashGrid2R's to the instruction, and **byte-identical**): slower every way it was fed its bias. From L2 a fragment
  at a time, AF3's trunk 367 -> 398 ms at 261 tokens and 1121 -> 1290 at 510 (long scoreboard the stall); prefetched a
  tile ahead, the same; copied per warp into its own double buffer, 420 and 1436 (the resident keys and the per-warp
  buffers hold an SM at 8 warps, and a two-deep per-warp pipeline hides nothing). flashGrid2R's trade - one bias
  tile shared by two rows through the block's memory, keys and values re-read per query block - is the better one.
- **The wide triangle's output side as a tiled GEMM** (a centre-norm pass writing the normed rows row-major, then
  `tgMain` with the gate and the residual staged through shared memory): byte-identical to triangleOutK, and slower -
  OpenDDE's trunk 1417 -> 1535 ms. Two passes move half again the bytes of the fused kernel, and the centre-norm
  pass's transpose of a channel-major tile was shared-memory bound (mio throttle, 1.4M bank conflicts a call); an
  `ldmatrix.trans` form might rescue the pass, not the extra bytes. (Removed; the GEMM's main loop, `tgMain`, is
  the input kernel's.)

- **The 384/512-channel fused transition at 4 warps and 16-column stages** (~50 KB, two blocks an SM, where 8 warps
  and 32 columns are ~100 KB and one): byte-identical and **10% slower** a trunk (OpenDDE 1446 -> 1587 ms,
  IntelliFold-2 2298 -> 2448) - the opposite of the triangle's input kernel, which gained at 4 warps.

- **More blocks an SM for the grid attention's `flashGrid2R`** (2026-10-07: `__launch_bounds__(..., 4)`, from
  162 registers and three blocks to 128 and four - Nsight has it L2-bound at 18.75% occupancy): the shipped form
  (2 warps x 2 rows, 48-key tiles) went **0.671 -> 0.737 ms** at 510 tokens on its 20 bytes of spills; only arms
  that do not ship gained (the 8-warp one 0.864 -> 0.790).

- **`gridOutK` for the row direction too** (the bf16 pair's f16 GEMM + add pass, `rowOut16`): the trunk level at
  261 and 510 tokens on an A100 (370.3 against 371.0 ms, 1123 against 1121). AlphaFold 2's port takes it, where it
  was 6-10 ms of a 494-residue fold faster than its own GEMM + add.

- **The fused grid-attention kernels at every pair width** (2026-10-03: `gridInK`/`gridOutK` launched at
  C = 256, 384, 512 for protenix2, OpenDDE and IntelliFold-2, whose unfused path spends ~a third of a fold
  in LayerNorm, gate, SwiGLU and add passes): no gain - protenix2's trunk 1269 against 1259 ms, OpenDDE's
  **2267 against 1928** (its 384-wide kernel fits two warps a block and took 704 ms). Every block re-streams
  the 4C x C projection weights for its 64-128 rows (512 KB a block at C 256) where cuBLAS reads them once:
  at these widths the extra memory passes are the cheaper side. A wide fusion needs the weights read once,
  not a wider template.

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

- **The trunk pass as a CUDA graph, everywhere** (a pass is ~1000 launches): capturing and
  instantiating costs ~15 ms against ~2 ms saved per replayed pass at 68 tokens, so a single fold
  at the page's 3 recycles came out slower there (105.8 -> 114.8 ms). TAKEN where it pays: the
  recycle passes replay one graph captured from the second pass when there are at least 7 recycles
  (AlphaFold 3's 10: 68-token trunk 231 -> 226 ms cold, 197 -> 176 warm) or 200 tokens (261: 494
  -> 489 cold); the output is byte-identical.
- **Larger blocks for the fused pair transition** (16 warps, halving its weight reads from L2):
  219 against 199 ms at 1044 tokens. Stripping it, neither the residual (-23 ms) nor the second
  GEMM (-27) dominates.

- **An L2 prefetch of the fused transition's residual** (`prefetch.global.L2` over the block's rows right
  after its norm, so the epilogue's read at the end hits L2): 0.199 -> 0.201 ms at 261 tokens, 2.797 ->
  2.816 at 1044 - level to slightly worse.
- **int8 tensor cores through cuBLASLt** (IMMA, int32 accumulate: the weights already ship int8, the activations
  quantised per row), for AF2's MSA transition shapes. cuBLASLt's int8 writes int32 only (no f16 output), which
  doubles the bytes of GEMMs already bound by their output: on a T4 133632x256x1024 is 3.08 ms against f16's 2.80,
  65536x128x512 0.66 against 0.39; only the narrow down-projection pays (133632x1024x256: 1.41 against 1.97) and
  its input would need a quantising pass (~1.3 ms over its 273 MB on a T4) costing more than it saves. An A100:
  0.418 against 0.356, 0.244 against 0.295. Worth it only as hand-written int8 kernels with fused epilogues.
- **`flashGrid2R` on a T4** (its two query tiles a warp, with the next key tile held in registers where there
  is no cp.async): slower than the register-staged `flashGridHalf` there - 698 ms (48-key tiles) and 645 (64)
  against 624 over a boltz2 fold. Its two stages at 39-51 KB leave a T4 SM one block of 4 warps.
- **The trunk's pair as a persisting L2 window** (`cudaAccessPolicyWindow`: at 255 tokens the f32 pair is
  33 MB and an A100 sets aside up to 26 MB): byte-identical and 14% SLOWER - trunk 311.6 -> 354.5 ms, and
  the diffusion, which never reads the pair, 60.0 -> 72.1. The set-aside costs every other stream more
  than the pair's hits return.
- **The unfused centre norm's statistics over all eight rows of threads** (one row of 32 summed every
  channel serially): 3 ms of OpenDDE's 1600 - its time is the strided load, not the sum. **And the
  unfused gated residual four elements a thread**: level (105.6 against 105.9 ms) - it moves ~1.36 TB/s.
- **The triangle-out norm two rows a thread** (one 32-bit shared read of a bf16 pair for two adjacent
  rows, half the threads, each row's sums in the same order): byte-identical and level at 261, 524 and
  1044 tokens - the norm is not what bounds the kernel. **And fused256's `triangleOutK` with its scale
  and offset as float2 loads** (half the loads, exact): 11% SLOWER at every size (0.241 -> 0.269 ms at
  261) - the registers to hold them cost a kernel that is register-limited at four blocks an SM (shared
  memory is what paid, above).
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
- **A fourth skinny GEMM: split over K across the WARPS of one block**, each warp double-buffering
  its own K slice, the partials summed in shared memory (no second kernel, no cross-block
  reduction): slower than cuBLAS at every tiling swept - 11-25 us against 7.9 for 68 x 768 x 3072,
  7.2 against 6.7 for N = 768 - and slower the more blocks it has: every block reads all of X.
- **The next block's weights prefetched into L2** while the current block runs: the 68-row GEMMs
  are not bandwidth-bound at all - the same GEMM with its weights hot in L2 is 8.9 against 9.4 us
  cold, and 8.1 us for N = 768 either way.
- **The pair residual in bf16** (AF3's own activation precision; the f32 residual is half of every
  pair kernel's read-modify-write). Measured before building it, by pointing two kernels' residual
  at bf16 with the traffic right and the numbers wrong, at 1044 tokens: the triangle's output
  kernel 170 -> 138 ms, the fused transition 201 -> 201 (its arithmetic binds). Extended to the
  other pair kernels that is ~4% of a trunk pass - not worth templating every pair consumer on the
  storage type and moving every fold's numerics.
- **The grid attention skipping its padding**: at 261 tokens a row's last 64-query block holds 5
  real queries and its last 64-key tile 5 real keys. Warps whose queries are all past the end
  computing nothing (they still load and wait): 0.190 against 0.191 ms at 261, 6.97 against 6.89
  at 1044. Adding the key tail (groups of keys wholly past the end skipping their products, inside
  the unrolled loops): 0.206 and 8.2 ms - the branches cost the kernel more than the work saved.
- **The denoiser's token attention at mid sizes, three ways** (261 tokens: 80 blocks of 64
  queries, 11.4 us for 0.21 GFLOP; the kernel costs ~3.6 us plus ~1.6 us a 64-key tile): the keys
  split over blocks with a merge kernel (flash-decoding) - 15.7 / 21.9 / 27.6 us against 11.4 /
  18.6 / 24.7 at 261 / 400 / 522; three to six cp.async stages - flat, then worse past 400; and
  two warps a 16-query group, each over half of every tile, merged in shared memory - within
  noise of the plain kernel, better at some sizes and worse at others.
- **The outer product mean's product straight into the output GEMM's layout** (a strided-batched
  GEMM over query tokens, the output weight's rows permuted to match): its m = 32 tiling ran
  0.84 ms a call against 0.5 for the GEMM and permute it replaced.
- **Grid attention without bounds checks** (inputs padded so loads past n read finite values):
  7.22 against 6.89 ms - at 125 registers the kernel is one register from losing a block an SM.

- **Five more forms of the grid attention's flash kernel** (`flashGrid2K`-`6K`, 2026-10-03, against the
  shipped `flashGridHalf<32, 4>` on random grids of 4 heads of 32, arms interleaved): QT query subtiles
  a warp over a ring of K/V/bias stages with one barrier a tile; rows of warps sharing one bias tile;
  the bias accumulated by the tensor cores (S = I . B + Q K^T, read by `ldmatrix.trans`) with each copy's
  pointer advanced rather than recomputed; the same with one barrier; and each warp pipelined across
  tiles (tile t+1's scores before tile t's softmax). None beat it: 7.02 ms at 1044 tokens against 7.08
  for the best (v5) and 7.95-9.62 for the pipelined form; 0.191 against 0.191 at 261; 0.849 against
  0.854 at 512. Nsight Compute had the old kernel L2-bound and latency-bound together (L2 81% busy, one
  eligible warp a scheduler); the forms that cut its instructions (2.17 -> 1.86 G at 1044) moved the
  stall rather than removing it. The one place a form won is a MASKED grid - 0.209 against 0.231 ms at
  261 (v4) - which a fold with every token real never runs.

## What the grid attention's time was

Without a profiler on this box, by counting SASS: the kernel issued ~800 integer instructions a
tile (the cp.async addresses, recomputed every tile in a strided loop the compiler could not
unroll) against 36 tensor-core MMAs. Compile-time trip counts and 32-bit offsets, the bias as S's
starting value, f16x2 exponentials straight into the P fragments, P's row sums as one more MMA,
and a tree for the row maxima: 8.07 -> 7.47 ms at 1044 tokens (`--bench-grid=1044`).

The triangle contraction runs in a padded np x np space (np a multiple of 8): cuBLAS's GEMM is
twice as fast on an aligned size (1044: 5.4 against 2.7 ms the pair of them). A multiple of 32 was
the first choice and padded the fused kernels' rows for nothing (68 tokens: 96^2 rows against 72^2).


## Device memory: what a fold holds

`LOCALFOLD_MEM=1` prints the device in use at each phase and the largest scratch buffers.

- **The transformer's conditioning GEMMs are no longer precomputed over every step.** They were held
  for the schedule up to a 4 GB budget - rows x 24 blocks x 6C halves, **3.0 GB at 68 tokens x 200
  steps and 2.3 at 525 x 25** - for 2% of the diffusion at 68 x 200 (308.8 against 315.6 ms), 1% at
  68 x 25 (52.0 against 52.8), and a LOSS at 262 and 525 (71.5 against 69.9, 101.3 against 98.1),
  interleaved on the A100. The diffusion phase now peaks at 5.31 GB at 68 x 200 (was 8.32) and 7.63
  at 525 (was 10.54). The single conditioning's precompute stays: it is 42 MB at 68 x 200.
- **A large input gives scratch back as it goes**, where it used to hold every stage's buffers to
  the end. At 1048 tokens the trunk peaked at 12.88 GB with the template stack's five pair tensors
  (1.4 GB), the MSA attention's and the recycled-pair LayerNorm still held through the pairformer, and
  the diffusion at 11.31 GB with the preparation's pair LayerNorms and per-super-block logits (1.4 GB)
  held through every step. Now **10.47 and 9.89 GB**. The diffusion's releases cost nothing measurable
  and run whenever the pair is over 128 MB; the trunk's cost the recycles their graph and a
  reallocation a pass - 2.5% of the trunk at 525 tokens, 0.3% at 1048 - so they run only where the
  pair is short of room (`shortPair`; see "The big-input line" below). Releasing the
  conditioning's chunk buffers too (`dc.f2*`, `pt.*`) was measured and not taken: they are CHUNK-sized
  at any length, and reallocating them cost 16 ms of a 100 ms diffusion at 525 tokens.
- **The grid attention takes every row in one pass only while its q/k/v/gate are a 32nd of the
  card** (it was a fixed 4 GB): whole is 3.3% of the trunk faster at 262 tokens and 4.8% at 1048, and
  at 1048 it is the 1.13 GB the row chunks do not hold - nothing on 40 GB, and the difference between
  fitting and not near a T4's 15.
- **A concatenated weight's f32 copy is given back once its f16 one exists** (`CONCAT`, `Wh`):
  the q/k/v/gate and paired projections are built by concatenating the file's tensors, and the fast
  path reads only the f16 result, so the f32 copy - 585 MB of AF3's - is queued and freed at the next
  phase boundary (one drain, not one a weight: freeing each at once was 27 ms of a cold 464 ms
  trunk), and rebuilt exactly from the file by a later W() if anything asks. AF3's peak at 262 tokens
  7.06 -> 6.58 GB, every fold byte-identical across the AF3 lineage, a cold fold ~7 ms slower.
  🔴 **IT FOUND A BUG BY MOVING THE ALLOCATOR**: rosettafold3 alone came out 0.12 A different, and
  the cause was its chirality gradient launched over half its atoms (a 128-thread launch counted at
  256 a block), the rest read from whatever memory the buffer landed in - a fold that depended on
  the memory layout, invisible until the layout changed.
- **What is left at the floor is the weights, twice**: the file's f32 device copy and, on `--fast`,
  its f16 mirror (3.8 GB in use at "trunk built" with 0.04 of scratch). cuda/ef2 drops the ESM-C
  tower's f32 copy after mirroring (`compactWeights`, 2.2 GB) because its tower reads only the mirror.
  cuda/af3 cannot do that blind: layer-norm scales, biases and the f32 conditioning GEMMs read the f32
  copy, and a path a first fold did not take (templates, an alignment, a ligand) may read one later -
  so it wants the set of f32 readers named, as ef2's `towerHalf` names its own.

## How large a fold fits

Measured on the A100 (40 GB, `peak` sampled from the device every 0.1 s, one fold, 25 steps, single
sequence). On a card short of room (`shortPair`, below) every stage gives
back what it alone used as soon as it is done, and what is read once is computed in row chunks rather
than held whole: the recycled pair (handed to the next pass rather than copied, freed once the
embedder has read it), the embedder's and the template query's normalised pairs, the template stack's
output norm (in place) and its 64-channel triangle buffers, the pair track's scratch before the next
pass's template stack, the conditioning pair after the diffusion is prepared, the encoder's and the
transformer's normalised pairs (the transformer's a super block and a chunk of rows at a time), and
the confidence stack's scratch before its heads. Every fold measured byte-identical to the build
before - the chunking changes no sum.

| tokens | peak, start of the pass | peak now | whole fold |
|---:|---:|---:|---:|
| 1572 | 15.9 GB | **10.2 GB** | 23 s |
| 2096 | 24.7 GB | **13.9 GB** | 49 s |
| 2620 | 36.0 GB | **19.2 GB** | 91 s |
| 3144 | 37.8 GB (after the first round) | **25.6 GB** | 149 s |
| 3668 | - | **33.6 GB** | 242 s |

The peak is now about **4.4 GB + 2.17 GB per million token pairs** (it was 3.36), so the ceiling is
**~2200 tokens on a T4** (15 GB) - measured: with a second process holding all but 14.6 GiB of this
card, 2200 folds and 2250 runs out, where 1725 was the limit before - **~2850 on an L4** (22.5 GB),
**~4000 on a 40 GB A100** (3668 measured), and more on an 80 GB card, where time is the constraint
(the trunk grows as the cube: ~4 minutes at 3668). 🔴 Past ~4096 tokens a pair tensor at 128
channels is more than 2^31 elements, so any kernel indexing it with an `int` overflows; not audited.
`LOCALFOLD_MEM=1` prints the device at every stage of the trunk; the peaks hunted here were INSIDE a
phase (the diffusion's preparation held the conditioning pair, two normalised copies of it, the
per-super-block logits and the 24 cached biases at once - 12.4 GB of scratch at 2096 tokens), which
the per-phase lines alone could not show. The WebGPU page stops earlier and for a different reason:
every pair-sized dispatch binds `tokens^2 x channels` floats against a 2 GiB binding limit on NVIDIA,
so 2047 tokens for AF3 and 1023 for IntelliFold-2.

### 🔴 The big-input line is the card's FREE memory, not a 64th of it

`shortPair` - the switch every big-input path above hangs off, and the one that also takes the recycles' CUDA
graph away - was "the pair over a 64th of the card": 1118 tokens for AlphaFold 3 on 40 GB, **646 for OpenDDE and
560 for IntelliFold-2**, whose pairs are 3-4x wider, with the card two-thirds empty either way. It is now **18x the
f32 pair against the room this process had at its first ask** (free memory, held scratch counted, a 20th of the
card spare; fixed then, so a warm fold decides as a cold one does, and a second process holding memory moves it
where a card fraction could not). The ordinary paths peak at 14-15x the f32 pair beyond what is resident by then
(AlphaFold 3 32.7 GB at 2000 tokens on a 2.05 GB pair, OpenDDE 16.3 GB at 765 on 0.90). On this A100 the room
reads 32-35 GB, so the line is ~1940 tokens for AlphaFold 3, ~1090 for OpenDDE, ~930 for IntelliFold-2 and ~1360
at 256 channels. `LOCALFOLD_SHORT_PAIR_TIMES=0` is the old rule; cuda/af2 and cuda/ef2 keep it until measured.
One fold, 25 steps, single sequence, the two rules (2026-10-07):

| | tokens | old rule | free-memory rule | peak |
|---|---:|---:|---:|---:|
| AlphaFold 3 | 1905 | 50.5 s | **28.3 s** | 11.3 -> 30.3 GB |
| OpenDDE | 765 | trunk 17.75 s | **12.77 s** | 14.2 -> 16.3 GB |
| OpenDDE | 1080 | 43.9 s | **37.8 s** | 20.7 -> 23.3 GB |
| IntelliFold-2 | 920 | 38.3 s | **32.3 s** | 11.6 -> 22.1 GB |
| protenix2 | 1345 | 34.6 s | **29.4 s** | 10.1 -> 23.2 GB |

🔴 **AND PAST THE LINE THE PENALTY WAS THE ALLOCATOR, NOT THE BIG-INPUT KERNELS.** At 2000 tokens the big-input mode's
kernels took 8.09 s of GPU time a pass against the ordinary mode's 8.25 - the GPU sat idle 45% of the pass. Each block
gave the triangle's whole-form planes back and the grid attention allocated its own afresh, a synchronous `cudaFree`
and a `cudaMalloc` of a 4.1 GB `grid.qkvg` - and a `cudaMalloc` past 2 GB is not cheap here (1 GB 0.9 ms, 2 GB 37 ms,
4 GB 145 ms). **The named scratch now comes from one stream-ordered pool that keeps what it is given back**
(`scratchPool`, common.cuh): the release hands the triangle's pages to the grid attention's buffers in 0.01 ms, and
the pool reserves the most any stage asks for, not the sum. Its idle pages count as free to every question asked of
the card (`deviceMemInfo`), and a plain allocation refused while it holds some trims it and asks again
(`devMalloc`). Byte-identical:

| | before | the pool | peak |
|---|---:|---:|---:|
| AlphaFold 3, 2000 tokens | 55.2 s | **31.2 s** (the ordinary paths: 31.7, at 32.7 GB) | 11.8 GB |
| IntelliFold-2, 960 | 40.4 s | **36.4 s** | 11.8 GB |
| AlphaFold 3, 1100 on a simulated T4 (big-input paths) | 17.3 s | **8.6 s** (the ordinary paths: 9.0) | |

Keeping the three updates' scratch instead (no release when the card holds it) was the first fix and measured the same
31.3 s at 19.0 GB, so the release stays and the pool makes it free. 🔴 **A POOL MAY NOT BE ASKED ITS SIZE WHILE A
STREAM CAPTURES** ("operation not permitted when stream is capturing" out of `roomFor` inside a recycle pass's
capture), so there `poolIdle` gives the last answer, which the eager pass the capture repeats was given.

**A wide model's big-input paths take its bf16 pair too** (the grid attention's streamed LayerNorm and the single
track's chunked pair logits were the two that read it as f32, so past the line its pair went back to f32):
IntelliFold-2 at 960 tokens 36.4 -> 35.1 s, against 34.1 on the ordinary paths. The gates' big lane moves by the bf16
pair and no further (OpenDDE 6MRR 1.083 -> 0.936 A, where its ordinary lanes read 1.081 and 0.939).

🔴 **OpenDDE's CEILING ON 40 GB WAS ~1100 RESIDUES AND IS PAST 1450 NOW, AND NONE OF IT WAS THE TRUNK.** Its second
token space (~2 structural tokens a residue) holds f32 [pairs, 384] tensors of 7.7 GB at 1150 residues and 12.2 at
1450, and three stages held several at once: the confidence head's working copy, symmetrised and LayerNorm'd pairs
beside the refined one (four whole copies - the PAE and PDE heads are per pair, so they run in row chunks now, and the
fold's last head works on the refined pair in place instead of copying it); the expander's two work buffers, held
through the refiner after it (given back now); and those same two buffers beside the 12.2 GB pair they build (in
chunks of the role-sorted order where they do not fit - only there, since a GEMM over other rows may round
differently). Every fold that fitted before is byte-identical, four samples across two seeds included. 1150 residues
41.9 s at 22.3 GB, 1300 57.3 s at 26.3, **1450 73.9 s at 30.8**.

Most of it is not the bf16 pair: OpenDDE at 765 with `LOCALFOLD_PAIR_F32=1` on the ordinary paths is 13.0 s.
🔴 **AND THE FIRST RUN PAST THE OLD LINE RAN OUT OF MEMORY, IN A STAGE THE PAIR DOES NOT SIZE**: OpenDDE's
structural expander works in its second token space (~2 tokens a residue) and takes two f32 [pairs, C] buffers
there - 6.8 GB each at 1080 residues - which ran beside the trunk's ordinary-path scratch because only the big-input
mode gave that back between stages. The trunk's scratch now goes back BEFORE the expansion whenever the pair is over
128 MB (`tightPair`), as it already did after it.

🔴 **KNOWN: `foldFits` DOES NOT SIZE OpenDDE's SECOND TOKEN SPACE** (measured before its ceiling moved, below - the
numbers here are a simulated T4's). It asks by residues, and OpenDDE's sampler and
confidence head work over ~2 structural tokens a residue. Under a simulated T4 (all but 14.6 GiB held, its
shared-memory limit, register-staged flash) OpenDDE folds 500 residues (971 structural tokens, ~11 GB) and runs out
at 600 (1167: `dc.sym` wants 2.09 GB) - in both big-input rules, so not the line above - while `foldFits` would admit
~2000. A constant does not fix it: the structural stages hold ~5x their pair on the T4's unfused kernels and ~2.5x
on the A100's (1080 residues, 23.3 GB), so one number either admits the T4's failure or refuses folds the A100
runs. The reader gets an error either way, a late one instead of an early one.

### `--recycle-tolerance`: the page's AF3 early stop, as an option (off)

`--recycle-tolerance=<A>` stops recycling once two consecutive passes moved the distogram's predicted distances
(the expectation over its bins, every pair) by less than that RMS - the criterion docs/AF3.md built for the page and
ships at zero, ported as it stands (shared/af3/feature-convergence.js; two passes and not one because GB1's trunk dips
under 0.5 A and then moves 1.09). Each pass computes the distogram once more and reads one number back. **Off by
default and not wired into the worker**: dropping recycles is a default for the page to decide, and the threshold is
not calibrated (2026-10-07). At 0.5 A, 25 steps, a warm fold:

| input | recycles | trunk, off | at 0.5 A | passes | CA RMSD to the full run |
|---|---:|---:|---:|---:|---:|
| 5CAJ, alignment (pLDDT 95) | 3 | 367 ms | 292 | 3 of 4 | 0.037 A |
| 5CAJ | 10 | 1005 | 295 | 3 of 11 | 0.072 |
| 1TIM, alignment (pLDDT 95) | 3 | 1176 | 901 | 3 of 4 | 0.053 |
| 1TIM | 10 | 3251 | 905 | 3 of 11 | 0.208 (3 against 10 recycles with no stop: 0.190) |
| OpenDDE 765, single sequence (pLDDT 44) | 10 | 34.9 s | 9.6 s | 3 of 11 | 14.4 (3 against 10 with no stop: 6.8 - the sampler, docs/AF3.md) |

Every input converged at the earliest pass the rule allows (0.02 A on the confident two, 0.23 on the low one).

### 🔴 5,000 tokens on a 40 GB A100

`cuda/af3/fold` folds a **5,000-token** chain on this A100 at a **35.0 GB** peak in 12.2 minutes (trunk
11.0, diffusion at 100 steps 1.0, confidence 0.2), every consecutive CA-CA in band (median 3.913 A).
4,192 tokens: 34.4 GB, 6.5 minutes. What it took, beyond the stage releases above - each only on a card
short of room, so a fold that fits runs exactly as before (68, 262 and 1572 tokens byte-identical):

- **the recycled pair is re-embedded IN PLACE**: each pair row's new value reads only the same row of the
  last pass, so its LayerNorm and projection are taken a chunk of rows at a time and the rows overwritten
  - no second pair (9 GB at 4,192 tokens), and the first pass starts from a zeroed pair.
- **the diffusion's conditioning pair is STREAMED**: each chunk of its rows, once transitioned, goes
  straight into the encoder's pair projection and the transformer's f16 LayerNorm'd pair; the f32 pair
  (9 GB at 4,192, beside the trunk's) never exists. It was the peak hidden inside the preparation.
- **the triangle multiplication in output blocks** and **the diffusion biases per super block** where
  the whole forms do not fit with an eighth of the card to spare (`roomFor`) - both cost time, so the
  card's free memory decides, and whichever form runs gives back the other's buffers first.
- **the template stack's sum is its one pass's activation** when only one pass counts (no templates, or
  one) - exact, and at every size; the MSA row attention's and the template query's LayerNorm'd pairs
  in row chunks; the f32 logits of the diffusion transformer only on the f32 path that reads them.

🔴 **A 25-STEP SAMPLER IS NOT CONVERGED PAST ~2,000 TOKENS, AND THAT IS NOT THIS PASS.** A 2,096-token
fold at 25 steps put 294 of 2,102 consecutive CA-CA distances out of band on the path from before any
of this; at 100 steps, 0. The large folds here are checked at 100 steps, which at these sizes is cheap.
🔴 **AND A COMMA IS NOT A CHAIN BREAK HERE**: `--sequence=A,B` exports one chain with an unknown residue
between, which is why these "repeats" are one chain.

`LOCALFOLD_BIG=1` runs every big-input path at any size; 6MRR through all seven AF3-lineage models agrees
with the ordinary paths within rounding (0.738 A for AlphaFold 3 either way).

### 🔴 And on a T4: 2,900 tokens

Simulated on this A100 as a T4 sees it - `LOCALFOLD_SMEM_LIMIT=65536` (its 64 KB of shared memory, so the
fused triangle and grid kernels step aside for the unfused ones, as they do there), `LOCALFOLD_FLASH_REG=1`
(its register-staged flash kernel) and a second process holding all but 14.6 GiB - a T4 folds **2,900
tokens** (3 minutes of A100 arithmetic; clean geometry at 100 steps) where it folded 2,000 before this
pass. What the T4's own paths needed beyond the A100's:

- **the unfused grid attention's LayerNorm'd pair is not kept** - per pair position, so the bias pass and
  each chunk take it again from the pair (a column chunk gathered transposed first): safe in place, since
  a row chunk writes only its rows and a column chunk only its columns.
- **the trunk's pair PARKED in pinned host memory** while nothing reads it: during the template stack
  (one pass: the pair is read for the query before and for the output after) and through the sampler
  (back for the confidence head). Only where the card is short (`parkWorthIt`), since it crosses PCIe.
- **the sampler's per-step conditioning is not precomputed** where it does not fit: steps x tokens rows,
  1.8 GB at 100 steps and 2,900 tokens.
- the 64-key branch of the grid flash kernel checks its own shared memory - it asked for 66 KB at 4 warps.

🔴 **AND NOW 3,200** (4 minutes of A100 arithmetic, clean at 100 steps): the blocked triangle gives its
fixed operand back at the end of every call rather than at the end of a stage, since the MSA attention
of the next block peaks beside the pair too. At 3,500 the operand itself (3.1 GB) no longer fits beside
the pair (6.3 GB) and ~1.8 GB of weights. Before this:

At 3,200 the MSA stack was what stopped it: the pair (5.2 GB), the blocked triangle's fixed operand (2.6 GB)
and ~1.9 GB of weights. The next levers are 2-D tiles for the triangle (the fixed operand a quarter the
size, recomputed four times - the projection is a tenth of the contraction's arithmetic) and dropping the
file's f32 copy of tensors only read through their f16 mirror.

### 🔴 6,000 tokens on a 40 GB A100

**6,000 tokens folds** at a 37.7 GB peak in 19.9 minutes (trunk 18.1, diffusion at 100 steps 1.4,
confidence 0.4), every consecutive CA-CA in band (median 3.885 A). Each attempt failed at the next holder
and each holder had the same shape - a pair-sized intermediate kept whole for a reader that wants rows:

| attempt failed at | what held | now |
|---|---|---|
| the pairformer's single attention | `[heads, n, n]` pair logits, scores and probabilities (6.9 GB) | blocks of query rows, each block's pair logits from its own pair rows |
| the MSA stack, pass 2 | the MSA attention's pair-sized buffers through the block's pair track | given back as the attention ends |
| the grid attention's bias | its 16-column projection of the whole pair (2.3 GB) | in chunks of pairs, each laid into the bias |
| the distogram | `[pairs, bins]` logits and their half (9.2 GB each) | blocks of rows: a row's logit is its own half plus the transposed pair's |
| the diffusion's preparation | the trunk's pair (18.4 GB) beside the f16 LayerNorm'd pair | the pair parked BEFORE the preparation, which reads it from the host a chunk at a time |

Past 2^32 elements in one pair tensor (4.6e9 here) nothing overflowed: the indices are 64-bit and the
per-channel ones stay under 2^32 to 65,536 tokens. The fixed cost at this size is the trunk's pair
itself (18.4 GB in f32) and the blocked triangle's fixed operand (9.2 GB), which must be whole: a
2-D tiling would read pair entries earlier tiles had written.

### 🔴 7,900 tokens on a 40 GB A100: the pair stays bf16 past the trunk (2026-10-09)

**7,904 tokens folds on this 40 GB card** (one pass and 4 steps for the measurement: 41.2 GB peak in the trunk, 36.1 in
the diffusion, 23.3 in the confidence head; a trunk pass 9.0 minutes), where 6,000 was the ceiling. The trunk's pair
was already bf16; past the trunk it was widened to f32 (`pairToF32`) for the distogram, the diffusion and the
confidence head - 18.4 GB at 6,000 tokens, 51 GB at 10,000, and the widening held both copies at once. On a card
short of room (`pairStays16`) it stays bf16 and every reader takes bf16 rows: the distogram's contacts (a gathered
bf16 transpose), the streamed diffusion preparation (each chunk widened as it is copied, from the device or from the
parked pair), and the confidence head (its pair bf16, its blocks under `PAIR16`, its heads widening a chunk of rows).
**The structure is byte-identical** (the diffusion reads exactly the values the widening made); the confidence head
storing its pair in bf16 moves PAE by at most 0.12 A and pLDDT by 0.06 on 6MRR under `LOCALFOLD_BIG=1`. Every fold
off the big-input path is byte-identical. Not yet for boltz2 (its head re-embeds the pair), rf3 (its global norm) or
OpenDDE (its expander) - those widen as before. What it took beside that, each found by folding past the old line:

- **the confidence heads' logits a chunk at a time**: `[pairs, 64]` f32 is TWICE a 128-channel f32 pair (25.6 GB at
  10,000), held whole only to be reduced to the PAE, the PDE and the pTM term - now each chunk's logits are reduced
  as they are made. Exact for the PAE and pTM and the pre-symmetrised PDE; AF3's PDE symmetrises after the projection
  (l_ij + l_ji), taken chunked through linearity as W (LN(z_ij) + LN(z_ji)) - 0.0008 A of rounding, and the PDE
  reaches no output file.
- **the trunk's pair allocated bf16 from the start** (`makeTrunk`): it was allocated f32, zeroed, then swapped -
  a 24.5 GB transient at 6,916 tokens, and 51 GB at 10,000, which does not allocate at all.
- **the pair parked before the diffusion when the PREPARATION does not fit beside it** (`diffusionPrepBytes`: the
  f16 LayerNorm'd pair, the encoder's projection, a super block of biases), not when the pair alone does not: asked
  for the pair alone, 6,916 tokens kept its 12.2 GB pair on the device and ran out in the preparation.
- **the bias cache counting only what its need includes**: a streamed preparation's handed-in f16 pair (12.2 GB) was
  counted as room toward the biases and admitted ones that did not fit.
- **the summary file in one pass over the pairs**: every chain's and chain pair's pTM/ipTM rescanned all n^2 pairs
  through a `std::function` - minutes at 6,916 tokens in 28 chains. Now each anchor's row is summed per chain once;
  every summary checked byte-identical (calmodulin's five chains, TetR+DNA's four, RNase B's two, 1BRS, 1TIM).

`foldFits` sizes a fold on this path at **1.15x the f32 pair** (measured 1.16 at the edge, the trunk's bf16 pair and
the blocked triangle's f16 operand being its floor), so it admits ~7,700 tokens on 40 GB and, by the same rule,
**~10,900 on an 80 GB card and ~12,000 on Colab's 96 GB RTX PRO 6000** - not measured there. The trunk is the time:
a pass is 9 minutes at 7,904 tokens, ~18 at 10,000 (the grid attention's flash kernel half of it, at ~40% of the
tensor cores' peak with heads 32 wide: its exponentials cost as much as its matrix work).

### 🔴 Past the card: refused up front (folding there is on a branch)

A fold that would not fit the card is refused **before anything is allocated**, with the longest this card
takes (`foldFits` in trunk.cuh: 1.8x the f32 pair plus a 20th of the card, against what is free - which
admits 6,000 tokens on 40 GB, and 3,200 but not 3,500 on a simulated T4, each as measured). So the limits are
the card's: ~6,000 tokens on a 40 GB A100, ~3,200 on a T4, scaling with the square root of card memory.

Folding PAST the card works and was measured - the pair in pinned host memory, through the card a window of
rows or columns at a time: **10,761 tokens on this 40 GB A100 in 2.7 hours** (trunk 150 min, diffusion 5.7,
confidence 3.1), clean geometry - but it is ~800 lines (windowed passes, double-buffered copies, a host-memory
bias cache), costs ~770 bytes of host RAM a token pair (89 GB at 10,761; a Colab A100 has 83.5, a T4 12.7),
and buys folds that take hours of a model trained on a few hundred tokens. It lives on branch
**`tier2-host-pair`** (d0b061b) to revisit. What it found that every big fold keeps: the MSA attention in
blocks of alignment rows, the blocked triangle's blocks as wide as what is free allows (each reads the whole
fixed operand), and the sampler's per-step conditioning and the encoder's pair projection given back when
their last reader is done.

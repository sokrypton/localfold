# BioEmu2

Microsoft Research's BioEmu2 (`microsoft/bioemu2`, checkpoint `bioemu2-preview`, MIT), ported from
af3-any-model's colab branch, which transcribes the vendor at `7846769` and holds itself to it at 0.000 A.
Everything here is held to THAT port - `/tmp/.../af3c/repo` is a clone of `sokrypton/alphafold3@colab` - through
the oracles in `tools/oracle/dump_bioemu2*.py`.

It is Boltz-derived and shares neither of AlphaFold 3's two big stages:

| | |
|---|---|
| trunk | a PairMixer: Boltz's MSA module without triangle attention (4 layers), then 48 blocks of triangle multiplication and a transition, on a 256-channel pair; 4 passes |
| denoiser | 16 dual-stream layers - a windowed atom transformer (32 queries, 128 keys) and a full 768-wide token transformer side by side, DiT adaLN-Zero, coupled through each token's centre atom. **415M of its 512M parameters** |
| sampler | EDM, Boltz's loop: gamma 0.8 above sigma 1, noise 1.0, step 1.5 (structure) / 1.0 (equilibrium), rho 7, sigma 4e-4 to 160 (x sigma_data 16), 200 steps, a 0 appended, the noisy coordinates rigid-aligned onto the denoised ones before every step; each sample centred on its real atoms |
| heads | a distogram (contact probabilities as P(d < 8 A) over Boltz's 64 bins on 2-22 A). **No confidence head**: no pLDDT, no PAE |
| modes | structure (method token 7) or equilibrium (4); contact conditioning; propagation from a structure (coords_cond, dt in ps, a companion frame augmented with the state); three steered samplers (edm_gd, edm_fk, edm_fkc) with the vendor's physicality potentials or a `--potentials` file; the physicality report on every sample |

## What is where, and what holds it

| piece | JavaScript | native | held to | gate |
|---|---|---|---|---|
| features | `shared/bioemu2/featurise.js` | `cuda/featurise/bioemu2_features.h` | colab's `bioemu2_features.featurise`, field by field | `tools/check-bioemu2-features.js`; `npm run test:bioemu2-native` (C++ = JS, byte for byte) |
| bonded graph | `shared/bioemu2/graph.js` | `cuda/featurise/bioemu2_steering.h` | colab's `pipeline/bioemu2_steering.Graph` | `tools/check-bioemu2-topology.js` |
| physicality report | `shared/bioemu2/physicality.js` | (the page runs it for every backend) | colab's `bioemu2_physicality.check`, all four presets | `tools/check-bioemu2-topology.js` |
| steering tables, polymers | `shared/bioemu2/steering-tables.js` | `bioemu2_steering.h` | colab's `tables()` | `tools/check-bioemu2-topology.js` |
| steering tables, ligands | `shared/bioemu2/ligand-tables.js` | `bioemu2_steering.h` | **an approximation**, reported | `tools/check-bioemu2-ligand-tables.js` (a report) |
| potentials + gradients | `shared/bioemu2/potentials.js` | - | colab's `energy()` and `jax.grad` | `tools/check-bioemu2-potentials.js` |
| `--potentials` files | `shared/bioemu2/potentials-yaml.js` | - | colab's `load_potentials` | `test/bioemu2-potentials-yaml.test.js` |
| steered samplers | `shared/bioemu2/sampler.js` | - | colab's `steered_sample` / `fkc_sample` on a toy denoiser | `tools/check-bioemu2-steering.js` |
| denoiser + its VJP (CPU reference) | `cpu/bioemu2/denoiser.js` on `cpu/bioemu2/tape.js` | - | colab's `denoise` and `jax.grad` | `tools/check-bioemu2-denoiser.js` |

The fixture jobs are in `tools/fixtures/bioemu2/`: 6MRR alone; with a phosphoserine and a glycerol; RNA + DNA; a
SMILES glycerol; with NAG and retinal (the only target that reaches the chiral, stereo-bond and planar terms); a
heteromer with paired alignments, two copies of one chain and an ATP; and `contacts.json`.

## Measured

- **Features**: every field and the atom keys exact on six targets, plus equilibrium mode with contacts.
- **Graph, polymer tables, physicality**: exact on five targets, every count, verdict and first-failed reason
  under all four presets, on a sample and three noised copies of it.
- **Potentials**: all ten default potentials, energies to 1e-7..1e-5 and gradients to 1e-7..5e-6 relRMS against
  `jax.grad`. The intra-chain VDW energy reads ~1e-4: the reference sums the dense overlap in float32 and SUBTRACTS
  the bonded pairs, which cancels; its gradient agrees to 1e-7.
- **Samplers**: at 2 and 4 steps all three agree with the reference to 5e-7..3e-6 relRMS, inside the reference's
  own float32 floor, with every resampling index the same. Over 20 steps `edm_fkc` agrees to 2.9e-7.
- **Ligand tables**: the same pair sets, flags and tuple counts as the vendor's on glycerol and NAG + retinal;
  stereo-bond and planar energies identical, PoseBusters within a few percent, chirality different in value (the
  vendor orders neighbours by CIP rank) and not in the hand it enforces.

- **CPU denoiser and its VJP** (`cpu/bioemu2/denoiser.js`): D(x, sigma) to 3.2e-7 relRMS at sigma 1; d/dx's network
  part to 1.4e-4 at sigma 16 (total 1.2e-3 - see the trap below) and 1.5e-5 at sigma 1; d/dsigma to 1.6e-3 / 7e-4. The
  tape in float32 against itself in float64: 4e-6, so the residual is the reference's float32 backward.

## The ports

| backend | where | float32 oracle, worst stage | fold | speed (A100) |
|---|---|---|---|---|
| CUDA (Colab, `localfold-bioemu2`) | `cuda/bioemu2`, `cuda/bioemu2/gate.py` | 1.6e-6 .. 3.6e-6 relRMS on six targets; `--fast` (f16 GEMMs, ef2's fused trunk) 1.7e-3 .. 3.2e-3; int8 blob 5-7e-3 trunk, which is the quantisation (decoded tensors match colab's dequantiser to four digits) | 6MRR single sequence 2.78-3.04 A over 5 samples, mean 2.88 - colab's own 2.88-3.20, mean 3.06; bonds against the CCD ideals equal to colab's (mainchain 0.034, side chains 0.052, nucleic 0.037 vs 0.036, ligand + SEP 0.084 vs 0.077) | 5 samples x 200 steps: 71 tokens 3.3 s, 255 tokens 7.0 s, 494 tokens 13.2 s; warm `--serve` trunk 100 ms at 6MRR |
| WebGPU (the page) | `webgpu/bioemu2`, `tools/gpu/check-bioemu2-network.js`, `tools/gpu/fold-bioemu2.js`; bundle `bioemu2-int5` (372 MB) | trunk single ~1e-7, pair 2.4-3.1e-4 (the AF3 pair track's float32 band over 52 blocks), denoise 2.6e-7 .. 4.2e-6, propagation 6e-7 .. 1.1e-6; int5 pair 2.6e-2, denoise 6.2e-2 at sigma 160 falling to 1.9e-3 | 6MRR (68) 0.90 / 1.58 / 1.65 A float32, 0.89 / 1.59 / 1.64 int5 on the same seed | 68 residues 1 sample 13.7 s on the page (32 ms a denoiser call), 255 residues 27.7 s |
| Metal | - | not ported | | |

🔴 **THE 6MRR FIXTURE IS 71 RESIDUES, NOT 68**: it was read off every CA line of the crystal, which repeats K14, E43
and D64 at their alternate locations. The oracle dumps share it, so parity is unaffected; RMSD to the crystal uses
`tools/fixtures/bioemu2/6mrr-68.json`.

## Traps

- 🔴 **JAX ON AN A100 RUNS A float32 einsum IN TF32** unless `JAX_DEFAULT_MATMUL_PRECISION=highest`. The steering
  oracle's rigid augmentation read 2.7e-4 off on its FIRST step - large enough to look like a logic bug - until
  the dumper set it. Every BioEmu2 dumper sets it now.
- 🔴 **edm_gd AND edm_fk ARE CHAOTIC IN THE REFERENCE ITSELF.** Nudge the initial draw by 1e-6 and over 20 steps
  the reference moves 2.4e-2 (`edm_gd`) and 1.0 (`edm_fk`, whose resampling flips a lineage): the flat-bottom
  potentials' gradients are discontinuous, so float32 alone flips a term. A fixed bound would fail a correct
  port; `dump_bioemu2_steering.py` measures that floor and the checker judges against it.
- 🔴 **A CONTACT NAMES A TOKEN, NOT AN ATOM**: a residue's token is centred on CA, so `atom_name: CB` on a protein
  residue matches no token and is refused - by the reference too.
- 🔴 **COLAB'S OUTPUT DROPS AN ATOM BIOEMU2 PREDICTS**: it writes the AF3 layout, which has no glycan O1, so a
  NAG's predicted O1 never reaches the mmCIF. Write the BioEmu2 atom set, not the AF3 one.
- 🔴 **A DENOISER'S GRADIENT CANCELS ITS OWN SKIP TERM**: D = c_skip x + c_out F(x), and at sigma 16 the network's
  d/dx (|.| 21) nearly cancels c_skip w (21) to a total of 2.6, so the network's 1.4e-4 float32 difference reads 1.2e-3
  in the total and looked like a missing term. Compare the network's part; and run the tape in float64 before
  blaming the tape.
- 🔴 **Python's `json` writes `NaN`**, which `JSON.parse` refuses; the checkers read a dump with it replaced.
- 🔴 **The steering reference data is other people's**: `stereo_chemistry.yaml` is CCP4 MON_LIB (LGPL), fetched at
  run time from BioEmu2's repository at `7846769`, never shipped; the vendor's ligand tier reads Boltz-1's CCD
  pickle of RDKit molecules (346 MB of Python objects), which is why that tier is built from the CCD here.

## Not done yet

- Steering on either backend: the CUDA sampler has a `StepHook` (guide, resample) seam and the page none yet;
  edm_gd/edm_fk need only the shared potentials.
- The native worker path: `bioemu2_features.h`'s arrays serialised into `cuda/bioemu2/export_input.mjs`'s entries,
  `pdb.template`, `confidence.template.json`, and a worker branch running `localfold-bioemu2 --serve`.
- `edm_fkc` on a backend: it needs the denoiser's reverse pass (d U(D(x, sigma)) / dx and / dsigma) on the device;
  `cpu/bioemu2/denoiser.js` is the reference it will be held to.
- The ligand steering tier is an approximation (above), not the vendor's RDKit tables.
- A SMILES ligand's conformer and atom names are this port's (the vendor's ETKDG draw is unseeded and its names
  come from RDKit's canonical ranking).

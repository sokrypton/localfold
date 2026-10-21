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
| potentials + gradients | `shared/bioemu2/potentials.js` | `cuda/bioemu2/src/steer.h` (= JS to the bit) | colab's `energy()` and `jax.grad` | `tools/check-bioemu2-potentials.js` |
| `--potentials` files | `shared/bioemu2/potentials-yaml.js` | `cuda/bioemu2/src/potentials_yaml.h` (= JS) | colab's `load_potentials` | `test/bioemu2-potentials-yaml.test.js` |
| steered samplers | `shared/bioemu2/sampler.js` | `steer.h` (= JS to the bit; `npm run test:bioemu2-steering-native`) | colab's `steered_sample` / `fkc_sample` on a toy denoiser | `tools/check-bioemu2-steering.js` |
| denoiser + its VJP (CPU reference) | `cpu/bioemu2/denoiser.js` on `cpu/bioemu2/tape.js` | `cuda/bioemu2/src/vjp.cuh` (`gate.py --only=oracles`) | colab's `denoise` and `jax.grad` | `tools/check-bioemu2-denoiser.js` |

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

- **Native steering** (`steer.h`): every energy and gradient element on three topology dumps bit for bit with
  potentials.js (680,040 of 680,040); edm_gd, edm_fk and edm_fkc on the toy denoiser identical to sampler.js, every
  resampling index; 9 `--potentials` files read and refused alike. The VDW overlap goes through a cell grid that
  visits the dense loop's pairs in its order (a pair further apart than the reach scores exactly 0).
- **Native VJP** (`vjp.cuh`, one particle a tape, cuBLAS held to float32): against jax.grad, d/dx's network part
  1.6e-4 at sigma 16 and 2.3e-6 at sigma 1, d/dsigma 1.7e-3 / 8.9e-4 - beside the CPU tape's 1.4e-4 / 2.3e-6 and
  1.6e-3 / 7.1e-4; the tape's D against the port's own denoise 2-5e-7. 55 ms and ~2 GB a call at 640 atoms.
- **Steered folds, native** (A100, 5 samples x 2 seeds, 200 steps; the potentials' violations at t = 0 counted over
  the 10 samples, a NAG's O1 left out because colab's mmCIF drops it): 6MRR + NAG + retinal - edm leaves StereoBond
  and intra-chain VDW violations in 9 of 10 samples, edm_gd and edm_fk in **none** (colab's edm the same 9-10, its
  edm_fk 1); edm_fkc is no cleaner than edm on either port (native StereoBond 10 / VDW 8, colab 10 / 8). 6MRR alone:
  no violations from any sampler but edm_fkc (VDW 3 natively, 0 colab). Time 4.4 s edm, 5.8 s edm_gd / edm_fk
  (the host's guidance 1.3 s), ~55 s edm_fkc (1,000 reverse passes).

## The ports

| backend | where | float32 oracle, worst stage | fold | speed (A100) |
|---|---|---|---|---|
| CUDA (Colab, `localfold-bioemu2`) | `cuda/bioemu2`, `cuda/bioemu2/gate.py` | 1.6e-6 .. 3.6e-6 relRMS on six targets; `--fast` (f16 GEMMs, ef2's fused trunk) 1.7e-3 .. 3.2e-3; int8 blob 5-7e-3 trunk, which is the quantisation (decoded tensors match colab's dequantiser to four digits) | 6MRR single sequence 2.78-3.04 A over 5 samples, mean 2.88 - colab's own 2.88-3.20, mean 3.06; bonds against the CCD ideals equal to colab's (mainchain 0.034, side chains 0.052, nucleic 0.037 vs 0.036, ligand + SEP 0.084 vs 0.077) | 5 samples x 200 steps: 71 tokens 3.3 s, 255 tokens 7.0 s, 494 tokens 13.2 s; warm `--serve` trunk 100 ms at 6MRR |
| WebGPU (the page) | `webgpu/bioemu2`, `tools/gpu/check-bioemu2-network.js`, `tools/gpu/fold-bioemu2.js`; bundle `bioemu2-int5` (372 MB) | trunk single ~1e-7, pair 2.4-3.1e-4 (the AF3 pair track's float32 band over 52 blocks), denoise 2.6e-7 .. 4.2e-6, propagation 6e-7 .. 1.1e-6; int5 pair 2.6e-2, denoise 6.2e-2 at sigma 160 falling to 1.9e-3 | 6MRR (68) 0.90 / 1.58 / 1.65 A float32, 0.89 / 1.59 / 1.64 int5 on the same seed | 68 residues 1 sample 13.7 s on the page (32 ms a denoiser call), 255 residues 27.7 s |
| Metal (this Mac, `localfold-bioemu2 --job=`, `localfold serve`) | `metal/bioemu2`, `tools/dump-bioemu2-webgpu.mjs` | held to **webgpu/bioemu2 on the same int5 bundle** (below): trunk 1.4e-8 .. 2.6e-7, context 4.6e-8 .. 9.0e-8, denoise 1.3e-7 .. 3.9e-5 against its float32 arm | 6MRR (68) through `--job` on the int8 blob 1.58-1.73 A over 5 samples, mean 1.65, every potential 0; on the int5 bundle 1.57-1.71, mean 1.61 (WebGPU on the same job and box 0.95-1.62); 5/5 pass the physicality report | M5, 5 samples x 200 steps at 68 tokens: the fast arm (default) trunk 0.53 s, sampler 14.5 s; the float32 arm (`--f32`) 2.3 s / 55 s |

🔴 **THE 6MRR FIXTURE IS 71 RESIDUES, NOT 68**: it was read off every CA line of the crystal, which repeats K14, E43
and D64 at their alternate locations. The oracle dumps share it, so parity is unaffected; RMSD to the crystal uses
`tools/fixtures/bioemu2/6mrr-68.json`.

## The Metal port, and what it is held to

`metal/bioemu2` is `cuda/bioemu2` on metal/core. `localfold-bioemu2 --job=<job.json>` is every port's command line
(cuda/featurise/standalone.h: the weights fetched, the input featurised in this process, the fold) and reads what
fetch.h fetches for `bioemu2`, af3-any-model's int8 blob - **AS the page's bundle**: `metal/bioemu2/weights.mm` is
tools/export_bioemu2_model.py done on the device as a weight walk (the stacks unstacked, the pair track transposed into
AF3's layout, the triangle's halves interleaved and the incoming one's swapped), so the port has one set of names
whichever weights it reads. `--bundle=model-bioemu2-int5` reads the page's int5 bundle instead, which is what the WebGPU
oracle is made from. 🔴 **THE WALK IS HELD TENSOR BY TENSOR** (`--weights=<blob> --compare-weights=<int5 bundle>`): all
1908 against the bundle's, every one within 0.058 - the two quantisations' gap - where a wrong layout reads ~1.4, and the
run carries its own NEGATIVE CONTROL (the outgoing triangle's projection against the incoming one's: 1.445, i.e. sqrt 2,
uncorrelated) so that it could not pass a transpose it cannot see.

🔴 **ITS ORACLE IS WEBGPU, NOT COLAB, AND THAT IS A REAL REFERENCE.** The colab dumps need af3-any-model's environment and
the vendor checkpoint, which no Apple box has. `webgpu/bioemu2` is a SEPARATE implementation (WGSL, its own kernels, its
own featuriser) reading THE SAME bundle, so the quantisation cancels and a residual is a port difference, not a weight
difference; and it is itself held to the colab dumps, so the absolute claim transfers. `tools/dump-bioemu2-webgpu.mjs`
runs it in Chrome and writes its stages as a featuriser directory; `--oracle=<dir>` then feeds EACH stage the oracle's
own input (as check-bioemu2-network.js does), so a residual is that stage's alone:

    node tools/dump-bioemu2-webgpu.mjs --out=/tmp/or --half=off --trunk=f32      # the float32 arm (default: shipped)
    cuda/featurise/bioemu2-featurise /tmp/in --job=tools/fixtures/bioemu2/6mrr.json
    metal/bioemu2/localfold-bioemu2 /tmp/in --bundle=model-bioemu2-int5 --oracle=/tmp/or

Against the float32 arm the first build read 1e-7 on the trunk and context and 1e-5 on the denoiser - on 6MRR and on the
`heteromer` fixture (156 tokens, 41 alignment rows, two copies of a chain and an ATP: 2.4e-7 / 1.3e-5). 🔴 **6MRR ALONE
CANNOT HOLD THE MSA MODULE**: it has ONE alignment row, so the outer product's GEMM over the rows, the pair-weighted
averaging and the valid-row mean all run at S = 1, where a wrong stride or a wrong row count is invisible. Hold a change
to the trunk to the heteromer too. 🔴 **TWO STAGES READ
EXACTLY 0, AND A CHECK THAT CANNOT FAIL LOOKS LIKE THAT** - so it was falsified before it was believed: against the
SHIPPED (float16) arm the same check reads 7e-4 .. 2.8e-3, and `pass1.pair` reads 7.371e-04 to every digit, the gap
WebGPU's own two arms show between themselves. The zeros are bitwise agreement (the same float operations on the same
inputs), not a blind comparison.

🔴 **WEBGPU'S TWO ARMS ARE 1e-3 APART, WHICH IS WHAT A BUG IS NOT.** Float16 weights and the device's pair track move
the pair 7e-4 .. 1.2e-3, the distogram 4e-3 and the denoiser 1.8e-3 .. 6.1e-3 (2e-5 on the single, which never meets
float16). A port defect reads O(1); measure that band before deciding a residual is one.

🔴 **THE FAST ARM'S SwiGLU PRODUCTS ARE STORED AS g / 64, BECAUSE FLOAT16 CANNOT HOLD THEM.** cuda/bioemu2 met the
atom transition's (87,750, past 65,504) and took bf16; metal/core has no bf16. Measured here over three jobs, the largest
product per module: the coupling's `s_to_q_mixer` **268,272** - four times float16's range, and not the one CUDA named -
the atom transition 84,467, the MSA transition 25,943, the pair transitions 9,845, the rest under 2,000; nearly identical
on every job, so properties of the weights. A power of two folded into fc2's rows of the derived weight and undone by
fc3's alpha is exact; /64 leaves 15x headroom. Varied over S = 8, 64, 512 the denoiser's residual moves without trend
(7.4e-3 .. 8.7e-3 at sigma 160): float16 noise, not the scale. Every GEMM whose input is NOT normalised (the distogram
reads the raw pair, RMS ~4.6e3; the readout the raw streams) multiplies in float instead - rounded to half they were inf,
and an inf in a PADDING atom's row meets an attention weight of exactly 0 as NaN (cuda/bioemu2's staging saturates,
metal/core's does not). The fast arm against the float32 oracle: trunk pair 2.5e-4, denoiser 3.2e-3 .. 6.9e-3 - WebGPU's
own float16 arm is 1.8e-3 .. 6.1e-3 from its float32 one - and 6MRR (68) folds to the same 1.61 A.

🔴 **THE DUMPER RUNS IN CHROME, NOT NODE, THOUGH webgpu/node.js EXISTS**: Dawn's node binding here has no `device.tick`
and its `mapAsync` never settles, so a node run exits 0 having written nothing - which is why nothing else in the tree
imports createNodeDevice. The page hands the dump back through gpu-chrome's result channel as base64, in chunks of a
multiple of 3 bytes (32768 is not: each chunk then ends in `=` padding and a decoder stops at the first - 38 MB written,
32 KB read back).

🔴 **SCORE 6MRR ON THE 68-RESIDUE JOB, BY ORDER.** The same Metal samples read 5.4 A against the crystal on `6mrr.json`
matched by residue number and 1.6 A on `6mrr-68.json` superposed in order (as fold-bioemu2.js does): the 71-residue
fixture repeats three residues, so its sequence is not the crystal's and every residue number after K14 is shifted.

🔴 **ON THE WORKER, EVERY CASE CUDA'S GATE HAS** (`python3 tools/check-native-worker.py --only=bioemu2 --offline`):
python/localfold/worker.py drives `metal/bioemu2` unchanged - the same `- --weights=<blob> --fast --serve=<dir>` it gives
the CUDA binary - and the eleven cases pass: a fold, propagation, equilibrium with contacts, the warm resident repeat
(the first fold's bytes), a ligand, edm_fk on NAG + retinal, edm_fkc REFUSED by name, and four refusals. The live preview
is the worker's `--frames` protocol: the trunk's contact map, then each step's estimate of the first sample superposed
onto the first step's. Steering is cuda/bioemu2's own steer.h over this port's denoiser: on 6MRR + NAG + retinal, plain
edm leaves mean ChiralAtom 0.90, StereoBond 1.17 and intra-chain VDW 0.71, and edm_gd and edm_fk take every potential to
0.000 - the pattern the native and colab runs above record - for ~0.45 s of host time.

🔴 **A RESIDENT SERVER'S SECOND INPUT FOLDED WITH THE FIRST INPUT'S ATOM-TO-TOKEN MAP**, caught before it shipped, by the
CUDA trap below: the denoiser cached its row maps keyed on the sample count alone, so the page's next fold of a DIFFERENT
protein read the first one's (past its end, for a larger one). Forgotten every fold now. 🔴 **AND THE TEST'S FIRST FORM
COULD NOT SEE IT**: it served 6MRR, the heteromer, then 6MRR again and compared the repeat with the first - which the stale
maps get RIGHT, since they are 6MRR's own. Falsified with the fix removed, it passed. Holding EVERY served fold to a fresh
process's bytes is what catches it: with the fix removed the heteromer served second differs; with it all four match.

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
- 🔴 **THE RESIDENT SERVER'S SECOND FOLD OF A SAME-LENGTH INPUT WAS GARBAGE** (median CA-CA 7-9 A, any mode): the
  fast trunk's all-ones triangle mask was filled once and remembered by its ADDRESS, and releaseScratch() hands that
  memory to the denoiser before the next job's scratch comes back at the same address. Filled every call now;
  `npm run test:native` folds a warm repeat and holds it to the first fold's bytes.
- 🔴 **edm_fkc's VJP READS A float32 STATIC CONTEXT UNDER --fast**: built on the f16 tensor cores it moved the
  network part of d/dx 2% at sigma 16 (2.0e-2 against jax.grad, 1.6e-4 in float32), the forward only 1e-3.
- 🔴 **Python's `json` writes `NaN`**, which `JSON.parse` refuses; the checkers read a dump with it replaced.
- 🔴 **The steering reference data is other people's**: `stereo_chemistry.yaml` is CCP4 MON_LIB (LGPL), fetched at
  run time from BioEmu2's repository at `7846769`, never shipped; the vendor's ligand tier reads Boltz-1's CCD
  pickle of RDKit molecules (346 MB of Python objects), which is why that tier is built from the CCD here.

## Not done yet

- Metal: **edm_fkc**, which takes the potentials' gradient back THROUGH the denoiser (cuda/bioemu2's vjp.cuh, the
  denoiser's whole backward) - refused by name, and tools/check-native-worker.py asserts that refusal on a Mac
  (METAL_REFUSALS) rather than skipping the case. edm, edm_gd and edm_fk all fold there.
- **The coupling split - on both, Metal's with Z in its GEMM's epilogue.** The coupling's atom-side mixer (fc1 | fc2,
  a third of the sampler) reads the adaLN'd row [a_token (768) | q_atom (128)], so W y splits exactly into a per-TOKEN
  part plus the 128-wide atom part, the row's LayerNorm statistics combined from the two halves' (the parallel-variance
  form, not E[x^2] - mu^2, which cancels on a stream with a large mean): 9.25 -> 2.7 G multiply-adds. WebGPU
  (`couplingSplit`, `false` is the control) took it first: a denoiser call 398 -> 288 ms on an M5, a 68-residue
  5-sample fold 84.9 -> 62.3 s, the RMSDs identical. Metal's first build stored Z and lost: in half its precision, in
  float 25% of its speed. 🔴 **WHAT MADE IT PAY THERE** (metal/bioemu2's couplingSplit, `BE2_COUPLING_SPLIT=0` the
  control): Z never leaves the GEMM's float tile - metal/core's `gemmCouplingCombine`, epilogue 32768 on the matrix
  units, the combine and the SwiGLU in it - and **Xq as TWO halves, hi and the remainder**, against W_q stacked twice
  (K 128 -> 256, free). Against webgpu/bioemu2's float32 arm (sigma 160 / 16 / 1 / 0.05): unsplit 7.6e-3 / 5.6e-3 /
  3.3e-3 / 4.0e-5, split with Xq in one half 7.8e-3 / 7.9e-3 / 9.7e-3 / 2.7e-4, with hi | lo **6.9e-3 / 6.1e-3 /
  3.1e-3 / 3.5e-5** - better than unsplit; Xa in one half costs nothing (hi | lo there moved nothing). The sampler
  14.5 -> 12.9 s at 68 tokens x 5 samples x 200 steps, the worker gate's BioEmu2 cases at the same RMSDs. Off the
  matrix units (an M2) Metal keeps the unsplit mixer. 🔴 **AND IT IS CHOSEN PER CALL**: Qa | Qq | B is a cost a layer
  that the samples do not share out, so at ONE sample the split LOSES (2.41 -> 2.78 s at 100 steps), ties at two and
  wins from three (4.94 -> 4.68; five 8.0 -> 7.0). `splitPays` compares multiply-adds with a margin of 1.7 fitted to
  those; `BE2_COUPLING_SPLIT=0/1` forces either arm. Qa | Qq | B themselves run on the matrix units as two half GEMMs
  (hi, then the remainder at beta 1) and the conditioning linears take their bias in the epilogue: 12.86 -> 12.75 s.
  🔴 ONE UNREPRODUCED FAILURE: the worker gate's 6MRR + GOL case once came back at RMSD 99 (an exploded sample) on the
  first run after that build, and did not recur in 38 runs since - 20 back to back, 4 with the kernel-specialisation
  cache deleted, 10 with the split on and off. Its log was overwritten before it was read. If it recurs, keep
  /tmp/localfold-cuda-check.log.
- **WebGPU's GEMM retiled: `gemm2Source` in webgpu/bioemu2/ops.js**, a 64 x 128 block of 4 x 8 lanes with vec4 panel
  reads (`gemmVersion = 1` is the original, the control; `--gemm=1` on profile-bioemu2.js and dump-bioemu2-stages.js):
  a call 286 -> 266 ms, its stage dumps against metal/bioemu2's `--f32` IDENTICAL to the old kernel's to four digits
  (1.397e-3 / 2.759e-3 / 9.961e-4 / 1.305e-5 at sigma 160 / 16 / 1 / 0.05). Two traps: a panel write must be a whole
  vec4 from one lane (component writes from four lanes raced), and split-K must count a tile's WORK, since a 128-wide
  tile halves the tile count and the old threshold split everything. 🔴 **f16 STAGING WAS BUILT AND DECLINED**: the
  panels in half and a half running sum promoted to float every N products - every 32, 266 -> 237 ms at FIVE times the
  error (sigma 16: 1.4e-2); every 8, 267 ms at 2-4x; every 4, 278 ms. A sum in half rounds against itself, and by the
  time it does not, the speed is gone.
- **Where that GEMM stands, and four arms that lost** (M5, ABAB, a call 266-268 ms on the shipped tile): the big
  shapes run at ~1.6 TFLOP/s against `probe-alu.js`'s 7.3 float32 (14.6 f16), so it is bound by none of the ALU, the
  workgroup reads (~75 G/s of 320) or the cached global reads. Hand-unrolled named accumulators 280 ms, BK 32 270,
  register prefetch of the next panel 270, every accumulator index a constant (epilogue included) 282 - the compiler's
  own unrolling of the shipped loop is better than any of them. And the same tile on 128 lanes of 8 x 8 (64 multiply-adds a
  lane for 4 workgroup reads, where 4 x 8 is 32 for 3): bit-identical and 3.5x SLOWER (862 against 243 ms a call) - 64
  accumulators a lane is past the registers; 4 x 8 is the ceiling that way too. The step past it is the matrix units, and a STOCK
  Chrome on the M5 still has no `chromium-experimental-subgroup-matrix` (only behind `--enable-unsafe-webgpu`), so no
  visitor would get it.
- **Both fusions, done.** `cs-combine` runs in the Z GEMM's epilogue (`fuseCombine`, `false` the control; `--fuse=off`):
  the GEMM's columns are reordered (`bRow`, gemm2Source's pair mode) so a lane's two vec4 halves are four gates and
  their four values, and the 41 MB Z is never stored - BIT-IDENTICAL stage dumps, a call 267 -> 259 ms. And split-K
  splits far less: its target 1024 -> 128 work units (`--split-target=`), a 5-sample call 258 -> 248 ms, a 1-sample
  one 107 -> 101 (no split at all is 106 there - it still pays where rows are few). The partial planes' traffic was
  most of what the deeper split bought. A 68-residue 5-sample fold 62.3 -> 56.5 s, RMSD 0.95-1.62 A.
- **Every SwiGLU in its GEMM's epilogue** (`ops.swigluLinear` on the same pair mode; `--fuse-swiglu=off` is the
  control): the transitions and q_to_s, the [rows][2 hidden] pre-activations never stored - bit-identical, a call
  248 -> 242 ms. A GEMM that would split over K keeps the two-pass path (a pair GEMM is not split). 🔴 The per-dispatch
  profile reads the fused arm 30 ms SLOWER and the wall clock 6 ms faster: the profile times each dispatch in a pass
  of its own, so read it for proportions only, as its header says.
- **The conditioning batched over the layers** (`conditionGroup`; `batchConditioning: false` / `--batch-cond=off` is the
  control): every layer's four adaLN MLPs read the SAME rows - the call's conditioning - so each is one GEMM with the
  layers as the batch over a stack of their weights (`stacked`), LN's statistics once a call and only its affine a
  layer's; the coupling split's gamma and Qa | Qq | B the same way, each layer handed VIEWS of the group's buffers
  (`ops.view`, a binding at an offset). 1302 -> 979 dispatches, a call 244 -> 239 ms (1 sample 101.8 -> 96.6), against
  metal/bioemu2 --f32 unchanged, the per-layer path 4e-6..1e-5 away (the trunk identical). 🔴 IT WAS WORTH 5 ms OF A
  33 ms UPPER BOUND (the conditioning recorded for one layer only): the rest is the GEMMs' own arithmetic, ~30 GFLOP a
  call, not their dispatch. Groups sized to 64 MiB. A 68-residue 5-sample fold 56.5 -> 52.2 s, the same RMSDs.
- **The GEMM is bound by WORKGROUP MEMORY, and the weight panels are staged in f16** (`b16Panels`, `--b16=off` the
  control; where the device has `shader-f16`). A lane reads 12 floats of panel per 32 multiply-adds; at the measured
  0.8 T multiply-adds/s that is ~300 G floats/s against `probe-alu.js`'s ~320 G workgroup reads/s - the ceiling, which
  is why f16 ARITHMETIC (twice the ALU rate on an M5) never paid. Two thirds of the reads are the weight's panel, and
  the weight is half already: staged as f16 and widened as it is read, it is EXACT - stage dumps bit-identical - and
  a third less traffic. A call 244 -> 231 ms; the fold 52.2 -> 50.4 s, the same RMSDs. (The activations' panel stays
  float32: rounding it is metal/bioemu2's precision, 2-5x this port's against the float32 reference.)
- **The atom stream's windowed attention in one kernel** (`ops.windowAttention`; `fuseWindow: false` / `--fuse-win=off`
  the control): the scores GEMM, the softmax, the P.V GEMM and the gating pass - three dispatches a sample and one a
  layer - as one dispatch a (sample, window, head), 128 lanes, four a query holding its query's row, the 128 keys in
  chunks of 32 with an online softmax, 12 KiB of workgroup memory. Worth 14.3 ms of a call as an upper bound (the
  attention skipped); taken: 230.4 -> 218.5 ms (1 sample 92.5 -> 89.9), 991 -> 751 dispatches. The trunk's atom
  encoder takes it too, so its stage dumps move by ~2e-8 at s_inputs and the 48-block pair track carries that to its
  float32 band (2e-4..8e-4); against metal/bioemu2 --f32 unchanged (1.392e-3 / 2.772e-3 / 9.96e-4 / 1.305e-5).
  6MRR-68, 5 samples: 50.4 -> 48.1 s, the same RMSDs.
- Steered propagation (refused on both backends, as BioEmu2's own propagation does not resample across frames).
- The VJP keeps every activation (~2 GB at 640 atoms, so a ~5,000-atom input would want recomputation) and is a
  plain f32 tape: edm_fkc is ~10x an unsteered fold.
- The ligand steering tier is an approximation (above), not the vendor's RDKit tables.
- A SMILES ligand's conformer and atom names are this port's (the vendor's ETKDG draw is unseeded and its names
  come from RDKit's canonical ranking).

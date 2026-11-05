# cuda/grad: backpropagation for the native ports, for training

Every CUDA port in this tree is forward-only, and fast because of that: fused kernels, int5 weights folded at load,
LayerNorm scales folded into the next GEMM, inputs of a Linear dropped the moment its GEMM retires. Training needs
the opposite of each of those choices, so the training graph is written **beside** each port, not inside it:

- **The inference forward is untouched.** Nothing under `cuda/af2`, `cuda/af3`, `cuda/ef2`, ... includes
  `cuda/grad`. A fold compiles and runs exactly as before; every `cuda/*/gate.py` baseline still holds because no
  byte of what it builds changed.
- **Weight gradients as well as input gradients.** bindcraft2's native AF2 (sokrypton/bindcraft2, `native/`) has
  the forward and backward for design, where the weights are frozen: every backward GEMM there is `dX = dY W^T`
  alone, and nothing a `dW` would need is kept. Training needs `dW = X^T dY` too, so the saves are different.
- **Float32 master weights**, read from the original checkpoints (AF2's `params_*.npz`), never the int5 bundles.
- **Memory by rematerialisation**: a stack keeps each block's input and replays one block's forward, with its saves,
  immediately before that block's backward (AF2's `use_remat`). Inside a block every module's backward runs in an
  arena `Scratch` scope, so only the saves and the running gradients outlive it.

## Status

Every port's main network has a training forward and backward. Two gates each (`run_gates.sh` runs all of them):
the float32 forward held to the PORT's own float32 path on its real weights, and the float64 build's backward held
to central differences of that same forward (`gradcheck.cuh`, below) - so the training graph computes what the
port computes, and its gradient is the gradient of that.

| port | what trains | forward vs the port | backward |
|---|---|---|---|
| **AF2** (`af2.cuh`) | Evoformer block, main and extra-MSA stacks, OPM first/last; the rematerialised 48-block stack | JAX `jax.vjp` in float64 (`oracle_af2_block.py`) | <= 3.3e-5, and 0.00-0.03 of the JAX bar |
| **AF2 structure module** (`af2sm.cuh`) | the eight shared-weight fold iterations (IPA with its point and pair terms, the transition, the quaternion backbone update) and the last iteration's side chain (angles -> the eight group frames -> atom14/atom37); cotangents on the whole trajectory, the last act, both angle forms and both atom layouts; `stopRotGradient` for AlphaFold 2's own training (off: af3-any-model's folding.py, which the port follows, keeps the rotation's gradient) | `structureModule()` on model_1_ptm and the trunk's real single and pair (data-test59's oracle): act 3.5e-7, rigid 1.8e-7, atom14 1.2e-7 - the port's own sensitivity to a 1e-7 nudge is 3.4e-7 | 51 tensors <= 3.5e-7 |
| **AF2's whole model** (`test_af2model.cu`: af2io + af2 + af2sm) | one pass end to end: the embedder, the extra-MSA stack, the main stack, the single, the structure module, every head - the composition a loss trains through | **af3-any-model's JAX** (data-test59, pass 0, 4 + 48 blocks, 512 + 1,024 rows): the pairs <= 4.3e-7, the msa 1.7e-6, trajectory 4.0e-7, atom14 3.5e-7, every head's logits <= 8.7e-7; a step 593 ms forward, 2,037 ms backward | 255 tensors <= 9.0e-5 through the whole composition (each stack a block), 1,024 extra rows |
| **AF2 embedder and heads** (`af2io.cuh`) | the input embedder (msa_feat and target projections, left/right singles, the recycled position's distogram, the recycled pair and first-row norms, both relative encodings, the extra MSA) and every head (single, pLDDT, PAE, distogram, masked MSA, experimentally resolved) | **af3-any-model's JAX itself** (cuda/af2/oracle.py's dump): embedded msa 1.5e-9, pair 2.9e-8; the heads on JAX's own representations 0 (PAE, distogram: the same bits) to 2.8e-7 | 37 tensors <= 1.1e-6, both relative encodings |
| **AF2's template embedders** (`af2tmpl.cuh`) | the multimer's (nine pair-input projections and the query's, the pair stack in its order, the ReLU mean, the projection) and the monomer's (embedding2d, the stack with both attentions first, the pointwise attention over templates); the template torsion rows into the MSA | `templateEmbedding` on 1BRS with a template: 1.2e-7; `templateEmbeddingMonomer` on 5CAJ: 2.3e-7; `templateRows` 0 | 129 and 114 tensors <= 4.1e-5, two templates (the second moved 0.7 A) |
| **AF3 lineage**, nine families (`af3.cuh`, `af3diff.cuh`) | pairformer block (sequential, chai-1's parallel, Mol-1's PairMixer), MSA block (AF3, update-first, chai-1 grouped), the 24-block diffusion transformer (AdaLN, chai-1's raw AdaLN, Boltz-2's up-gate, rf3's q/k norm and noResidual) | `pairformerBlockAt`, `msaBlock`, `diffusionTransformer` <float>: 2e-7..3.3e-5 on af3, boltz2, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1, chai1 | <= 8.5e-6 under every dialect switch |
| **AF3 lineage: the whole denoiser** (`af3denoise.cuh`) | D(x, sigma) from the trunk's single and pair and target_feat: the diffusion conditioning (relpos, Fourier noise, the split/relpe pair forms), the atom encoder (reference embedding, atom pair and its MLP, the atom stack, aggregation), the transformer, the decoder, EDM's skip/out - into every parameter, the trunk's outputs and x | `denoise()`: af3 8.9e-6, protenix2 1.5e-6, intellifold2 1.7e-6, mol1 1.6e-6, rf3 9.9e-8 (its chirality term as the port computes it, a constant here), chai1 4.2e-6 (its structure-pair conditioning, raw AdaLN, decoder conditioning), opendde 4.9e-7 (on its 130 structural tokens), openbind0 2.0e-4, boltz2 4.1e-4 - the last two the transformer's float32 arithmetic (below) | every tensor of 688 <= 1.1e-5 (af3), at h = 1e-8; chai1 433 <= 6.3e-5; opendde 699 <= 1.2e-5 at h = 1e-7 |
| **AF3 lineage: the input embedder and distogram** (`af3embed.cuh`) | target_feat (its atom encoder and the concatenation; boltz2's atom-only sum of six token tables; chai-1's token embedding with its ESM2 rows, both projections and its encoder - LayerNormed conditioning, the 14-feature biased atom pair, the two-layer MLP, the reference-space mask), z_init (the left/right outer sum, the recycled pair, the relative encoding, the bond terms; chai-1's own relative encoding, chain-entity and declared-bond terms, its first pass recycling z_init and s_init through themselves), s_init and the recycled single, the MSA embedding (chai-1's 41 columns), the distogram head (chai-1's MLP form) | `buildTargetFeat()` and `embed<float>()`'s z_init seam, single and msa on pass 0: <= 3.1e-7 on af3, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1, boltz2 (7.3e-8), chai1 (1.6e-7, its target_feat with ESM2 3B) | 109 tensors <= 2.4e-5 (af3), boltz2 121, chai1 99 <= 8.8e-5 at h = 1e-8 |
| **OpenDDE's structural tokens** (`opendde.cuh`) | the expander (s_inputs and the single gathered from each subtoken's residue with their role embeddings, the split MLP, each structural pair through one of 49 role-pair matrices plus five feature tables, the attention bias), the refiner (four pairformer blocks, the bias on every single-attention head), OpenDDE's own confidence head (the clamped single's norm, s1/s2, the distance bins and raw distance, its blocks, PDE LN(z + z^T), PAE, the slot-major pLDDT) | `expandStructural` + `ddeConfidence` on 6MRR's 130 subtokens: s_inputs and bias exact, the refined pair 2.0e-6, single 4.9e-6, PAE/PDE/pLDDT <= 1.1e-6 | 137 tensors <= 6.6e-6 (one block each), the trunk's outputs included |
| **AF3 lineage: the template embedder** (`af3tmpl.cuh`) | the query's LayerNorm and projection, each slot's features (the nine projections with the distogram and geometry, or the fused feature columns; chai-1's bias), the pair-only stack with grid attention (chai-1's parallel), boltz2's outer residual, the output norm (chai-1's per-template coverage), the weighted sum, its ReLU and projection | the template term against `embed()`'s z_init_generic -> z_after_template on 6MRR with its own crystal in two slots: 1.4e-7..2.5e-7 on af3, boltz2, protenix2, intellifold2, rosettafold3, opendde, openbind0, mol1; chai-1 3.3e-7 against `templateEmbedding` alone (its target_feat reads ESM2 3B) | 80-88 tensors <= 9.6e-6 (af3, boltz2, chai1, rosettafold3, protenix2) |
| **AF3 lineage: the confidence head** (`af3conf.cuh`) | target_feat's projections and the pseudo-beta distogram into the pair, its four pairformer blocks, the PDE (AF3's symmetrised-after, protenix2's LN(z + z^T)), PAE and per-atom pLDDT logits; boltz2's re-embedded pair and single (its norms, the relative encoding, the s_inputs projections and their outer product, the 64-bin distance and bond terms) and split intra/inter-chain heads; rf3's parameter-free global norms (target_feat's over 449 columns, its two absent columns through the statistics) and CA distogram; chai-1's 16-bin unmasked distances, its transposed output twins (W + W_T, both trained) and ATOM37 pLDDT | `confidenceHead()`'s expectations: PAE/PDE <= 4.5e-6 on af3, protenix2, intellifold2, openbind0, mol1, boltz2 (4.8e-7), rf3 (1.6e-7), chai1 (1.5e-6); pLDDT held to 4x the port's own sensitivity (below) | 219 tensors <= 5.5e-6 (af3), boltz2 433 <= 4.1e-6, rf3 237 <= 1.9e-5, chai1 227 <= 5.5e-6 |
| **ESMFold2** (`af3.cuh`, `msap.cuh`) | pair trunk blocks of the trunk, coda, LM and MSA encoders | `trunkBlock`: 1.9e-7..3.8e-7 | the pairformer's |
| **ESMFold2's atom stack** (`ef2atom.cuh`) | c0 and the sliding-window adaLN-Zero blocks (q/k rms-normed per head and rotated by the 3D RoPE table, the window by valid rank, the live gate, the gated SwiGLU) the inputs embedder and the diffusion module's encoder and decoder share | `prepareAtoms` + `swaBlock` in their float32 arm: <= 9.0e-8; **the vendor's own float32-attention dump**: <= 1.3e-7 | 21 tensors <= 5.5e-7 |
| **ESMFold2's diffusion module** (`ef2diff.cuh`) | one denoiser call: the conditioning pair (its relative encoding, the two transitions) and every token block's bias, the noise-conditioned single, the atom encoder and decoder (the SWA stacks), the 12 adaLN token blocks with pair-biased gated attention and gated SwiGLU, the pooling and broadcast, EDM's combine | `makeDenoiser` + `denoise` in their float32 arm: 1.1e-6 | 349 tensors <= 2.6e-5 |
| **ESMFold2's trunk glue** (`ef2trunk.cuh`) | the language-model shim (the softmax mix of all 37 hidden states - their cotangents go back into the tower through `towerBackward`'s `dstate` - the projection, the zero row of a token the tower never saw, the pair MLP over `[a*b \| a-b]`, rebuilt a block of rows at a time in the backward rather than kept), the inputs embedder (c0, the windowed atom stack, toToken, the masked pool, the tail), z_init (the outer sum, the relative one-hot, the bonds, lm_z), both recycles (the experimental tier's `Linear(LN(z)) + z_init`; the parcae recycle's lmEncoder over the dropped-out pair, the per-channel decay, the readout and the coda - with the port's own dropout draw, so a seed drops what it drops), the full ESMFold2's MSA encoder (the embedding and s_inputs' projection, the count-clamped outer product, the unmasked pair-weighted averaging, the transition, the pair block), the distogram | **the vendor's own tensors** on 6MRR (its float32-attention dump): lm_z 1.8e-7, s_inputs 1.0e-7, z_init 5.7e-8, all four passes in <= 1.2e-7 and out <= 9.5e-7, the distogram 5.2e-7; `foldingTrunk` 7.4e-7. The parcae recycle against `foldingTrunk` (EF2_DETERMINISTIC): ESMFold2 4.1e-5, ESMFold2-Fast 1.2e-4 - the port moves 4.1e-5 and 5.5e-5 by itself for a 1e-7 nudge of these random inputs, and the bar is 4x that. The MSA encoder against `msaEncode` on a real 59-residue alignment's first 1,024 rows: 1.9e-7 | 67 tensors <= 1.3e-5 (experimental), 114 <= 1.9e-5 (parcae, dropout on), the inputs embedder's 22 <= 3.0e-6, the MSA encoder's 140 <= 8.3e-6 (a column mask on) |
| **ESMFold2's confidence head** (`ef2conf.cuh`) | Synthyra's head: the four s_inputs projections and their outer product, the normalised trunk pair with z_init's relative and bond tables (shared parameters, both gradients), the distance embedding, four pair-only blocks around a residual, the row-attention pooling, the per-slot pLDDT tables, the PAE | **the vendor's own tensors** on 6MRR: per-atom pLDDT 7.2e-8, PAE logits 4.6e-7; `confidenceHead` 9.4e-8 | 102 tensors <= 4.5e-6 |
| **ESM-C / ESM2** (`plm.cuh`) | the towers, ESM-C's sequence head, a masked-LM cross entropy | `esmcTower`, 36 layers: 5.8e-7 | <= 5.3e-7 |
| **MSA Pairformer** (`msap.cuh`) | its layer: PWA, transition, query-biased OPM (differential sequence weights), pair block | the port's float32 path: <= 4.2e-7 | 2.6e-6 |
| **MSA Pairformer's whole model** (`msapmodel.cuh`) | the MSA embedding, the initial pair (relative position over the break-shifted index, the bond), the layers, both contact heads (Cb, ConFind) at their layers and the LM head on the query row | `msaEmbedK`, `pairInitK`, `contacts()`, `lmHead()`: 0 to 1.5e-7 (the layers: the row above) | 115 tensors <= 2.8e-6 |
| **BioEmu2** (`bioemu2.cuh`) | MSA layer and PairMixer layer (Boltz modules, stacked (out, in) weights) | `pairWeightedAveraging`/`transition`/`outerProductMean`/`trimul`: <= 2.1e-6 | <= 2.1e-6 |
| **BioEmu2's denoiser** (`bioemu2den.cuh` on `tape.cuh`) | the static context (the pairwise conditioner's pair and relative encoding, the atom features, the windowed atom pair's geometry, token term and MLP tail, every layer's window and token biases, the single conditioner's static half, a's trunk half) and one D(x, sigma): the single conditioning, 16 dual-stream layers (windowed and full conditioned attention, the coupling mixers), the readout | `staticContext` + `denoise()` float32: the context <= 1.1e-7, D 5.7e-5 at sigma 5 and 5.3e-6 at 40; d/dx against the port's own vjp tape (jax.grad's) at float32's floor (below) | 709 tensors <= 7.6e-5, the context's weights and the trunk's three inputs included |
| **BioEmu2's trunk** (`bioemu2trunk.cuh`) | the input embedder (the per-atom features, the windowed pair and its tail, three windowed conditioned layers - on the tape - the mean over each token's atoms, the tail columns), s_init and z_init (the method rows, the bonds, the contact conditioning's class vectors and its threshold encoder through its Fourier features, the relative encoding), the recycle, the MSA embedding, the MSA and PairMixer layers, the distogram | `inputEmbedder` + `trunkInit` + two `trunkPass` + `distogram`, float32, on 6MRR: s_inputs 1.0e-7, z_init 6.5e-8, the pair 2.4e-7, the distogram 1.2e-6 | 161 tensors <= 5.0e-5 (one MSA layer, two PairMixer layers, contact classes on some pairs) |
| **RFD4-Proteina** (`rfd4p.cuh`) | token layer (RMS AdaLN, pair-biased attention, SwiGLU, output scales) and pair update | `tokenLayer`/`pairReprUpdate` <float>: 5e-8 / 1.6e-7 | 1.9e-6 |
| **RFD4-Proteina's whole trunk** (`rfd4trunk.cuh`) | one cycle: the conditioning and its two SwiGLU transitions, s_init, z_init (the pair factory's RMS-normed projection adaLN-conditioned on the pair condition), the 14 token layers and 7 pair updates, the CA, local-latent, sequence and insertion heads | `forward()` FP32, one recycle, on a real 54-token input: every stage and head <= 4.4e-7 | 93 tensors <= 1.9e-6 |
| **RFD4-Proteina's autoencoder** (`rfd4ae.cuh`) | the layer both halves share (LayerNorm throughout, no pair bias, the zero conditioning's one-row gammas, betas and gates - whose weights train through that row), the decoder's features (two embedding tables) and three heads, the encoder's per-atom features (three tables) and mean / log-scale head | `decode()`: every layer <= 1.5e-5 (the port's own floor for this near-uniform softmax), heads <= 8.5e-6; `encode()`: <= 4.7e-7 | 69 + 35 tensors <= 4.2e-6 |

Found by the whole-model gate: the global column attention's backward kept H S reals in shared memory, past a launch's
48 KB at AF2's 5,120 extra rows in float32 (1,000 in float64) - the launch failed unchecked and the extra stack's
gradients were whatever the buffers held, while every single-block gate (small S) passed. ESMFold2's confidence pooling
had the same shape at 6,144 tokens; both are global buffers now.

Every CUDA model's forward now has a backward here; what is not is said in each header (e.g. rf3's chirality term is a
constant input to the denoiser, a template's geometry and a sampler's coordinates take no gradient).

Also: AdamW (`adam.cuh`); `train-af2-smoke` - real weights perturbed 10% and trained back, loss 918 -> 14.7 in 60
steps. `train-of3` - **OpenFold3's training, in CUDA** (OpenBind-0 is OpenFold3 v0.5.0, so the recipe is its own, read from
aqlaboratory/openfold-3 at `~/beta/of3-src`): the family's whole training network composed from its bundle (target_feat,
the trunk embedder, the template term - which weighs even with no template, its padded slots' restype zero being a
one-hot one - the MSA stack, the pairformer, the distogram, the denoiser, the confidence head) and trained as
`projects/of3_all_atom` trains it:

- **data** - `cuda/grad/make_of3_example.py <entry.cif> <dir>` turns a deposited structure into an example: the AF3 job of its
  chains (each polymer its entity's FULL sequence, modified residues as modifications, every other chain a ligand of its
  components; polymers first, the order the featuriser gives asymId), the truth keyed by chain, residue and atom name,
  the resolution, the RDKit permutations, and the featurised input - and OpenFold3's datapoints: a line a chain and a
  line an interface (chains with heavy atoms within 5 A), each weighted as AF3 SI 2.5.1's Algorithm 1 weighs it,
  (w_chain 0.5 or w_interface 1) / N_clust x (3 n_prot + 3 n_nuc + 1 n_ligand), and naming it `preferred` for the crop.
  `--examples=<file>` lists them (`input truth [resolution=X] [weight=W] [preferred=i[,j]] [perms=P] [set=S]`), drawn by
  weight; lines naming one input share one load. `set=` is OpenFold3's dataset (`dataset_configs.py`): `pdb` (the
  weighted PDB set), `monomer`/`rna` (the distillation sets: no confidence losses, no chain crop, crops 0.25 contiguous
  0.75 spatial) and `disordered` (no confidence losses, the mse and smooth LDDT over protein tokens alone, the
  alignment still over every resolved atom).
- **cropping** - first OpenFold3's chain crop (AF3 SI 2.5.4: past 20 polymer chains, the 20 nearest a random preferred
  centre, the preferred chains and every ligand within 5 A of a kept chain - the truth becomes that subset, as there),
  then to `--crop` tokens (384) by the three strategies (contiguous 0.2, spatial 0.4, spatial-interface 0.4; whole when
  it fits), over the resolved token centre atoms, the preferred chain or interface honoured. GroEL-GroES (1AON: 8,533
  tokens, 35 chains, 98 datapoints) trains at ~8.6 s a 384-token step with 4 samples, 0.7 s of it the host's crops. The crop
  cuts the host arrays and REBUILDS the atom windows with the featuriser's own `atomGathers` - held, on every example
  at load, to the windows the featuriser wrote.
- **the set** - `cuda/grad/make_of3_dataset.py --ids=... --val=... --out=<dir> --search` builds a training set from PDB
  entries as AF3 SI 2.5 does: fetched from RCSB, resolution at most 9 A and released before `--released-before`
  (AF3's 2021-09-30), AF3's structure filters (crystallisation aids, polymer chains of under 4 resolved residues,
  all-unknown chains, clashing chains), the featuriser's MMseqs2 alignments, and templates from RCSB's sequence search
  released at least 60 days before the entry and never the entry itself (SI 2.4: the featuriser's own template pick has
  no date, so it would hand an entry its own structure). Chains are clustered (proteins at 40% identity - an affine-gap
  alignment at 80% coverage, as MMseqs2 would; peptides and nucleic chains at 100%; ligands by their components) and
  every datapoint's weight divided by its cluster's size, N_clust. `--mmseqs=<binary>` clusters the proteins with
  OpenFold3's own command (`clustering.py`: `easy-cluster --min-seq-id 0.4 -c 0.8 --cov-mode 0 -s 8 --max-seqs 1000
  --cluster-mode 0`); on the set below it gives the same 16 clusters and the same 54 weights as the aligner. 9 entries -> 54 datapoints in 16 clusters (the six
  haemoglobin chains one), 7BNY left out by its date, four entries with templates.
- **conformers** - a fresh conformer every draw, as OpenFold3 embeds one per featurisation: the example builder runs
  RDKit's ETKDGv3 (hydrogens on, random seeds, then off) 8 times on each component's CCD graph and the trainer draws
  one per component instance (`--check-conformers`: on 1STP 593 of 1,181 atoms placed differently, every bonded pair
  within 0.02 A rms); then each reference space centred, randomly rotated and moved (`centre_random_augmentation`).
- **templates** - drawn per chain each step as OpenFold3's training draws them (`sample_templates`, AF3 SI 2.4: k =
  min(U{0..l}, n) of a chain's l templates, at random, where inference takes the top k), over the slots the input
  carries: each new slot assembled a chain at a time from the slot that chain drew, its cross-chain pairs masked, the
  rest padding (the gap restype), the all-padding slots folded into one pass. A self-templated 6MRR draws 0, 1 or 2.
- **dropout** - OpenFold3 trains with it and the AF3 blocks had none: `pair_dropout` 0.25 on the triangle
  multiplications' and the starting attention's updates rowwise and the ending attention's columnwise (the transposed
  pair's rowwise, as there), in every pairformer-style stack - trunk, MSA module, template, confidence head - and
  `msa_dropout` 0.15 rowwise on the MSA's pair-weighted averaging. The masks are af2.cuh's counter hash of (seed,
  module, position), so a recomputed block and the backward regenerate them; `test-af3-f64` gradchecks both blocks with
  it on (1.1e-6, 3.9e-7) and `--check-ckpt` holds the recomputation to the kept saves with it on. Off for measurements
  and validation, as OpenFold3's eval mode; `--no-dropout` turns it off for training.
- **the trunk** - 1 + U{0..3} cycles (drawn from a generator every rank shares, `synced_generator`), the gradient
  through the last; each cycle's MSA a fresh 1024 rows, the valid ones
  in a random order then invalid ones (`_subsample_all_msa`). Every MSA and pairformer block's saves are dropped as it
  finishes and recomputed before its backward (OpenFold3's `blocks_per_ckpt` 1): `--check-ckpt` holds every gradient so
  computed to the one with the saves kept - 2.6e-5, against 2.3e-5 between two identical runs (the backward's atomics).
- **losses** - `of3loss.h`, OpenFold3's nine (mse with the per-type weights and Algorithm 28's weighted alignment, bond,
  smooth LDDT, the diffusion weighting; distogram; pLDDT, PDE, PAE, experimentally resolved) with their gradients, held
  by `test-of3loss` + `of3_loss_reference.py` - **OpenFold3's own loss functions**, imported from a checkout and
  called, the case laid out in OpenFold3's flat atom order (its canonical atoms a token, a dropped leaving atom a masked
  padding atom) - to **1.4e-14** on every value and gradient, on the kitchen sink, where every branch is reached
  (`of3_loss_oracle.py`, a line-by-line transcription, is the check that needs no checkout). 🔴 It found one thing the
  transcription had copied wrongly: OpenFold3 sums five of its counts in float32, where the eps vanishes against the
  count - the bond loss over the kitchen sink's one bonded pair read 1.0e-8 off until the C++ did the same. The confidence head gained its
  experimentally-resolved output (LayerNorm and a [Cs, dense x 2] projection; `test-af3conf-f64` gradchecks it).
- **the rollout** - AF3's sampler, 20 steps, one sample, no gradient; the confidence head on it with every input
  detached, its losses 1e-4 each and zero (its head out of the step) outside 0.1-4.0 A.
- **the truth's permutation** - `of3perm.h`, OpenFold3's `permutation_alignment.py` (AF3 SI 4.2) on the rollout, before
  every loss: the least ambiguous molecule as anchor, a candidate alignment a predicted copy of its entity, greedy
  centroid matching of every predicted molecule to an unused truth copy, then each component's naming chosen among its
  RDKit automorphisms (`GetSubstructMatches(mol, mol, uniquify=False, useChirality=False)` on the CCD graph, written by
  the example builder as `perms.txt`: Val, Leu, Phe, Tyr and a sulfate swap; Asp, Glu and Arg do not, their CCD bond
  orders telling the oxygens apart). The prediction is the crop and the truth the whole entry. `--check-perm` disguises
  an entry - every repeated molecule's copies relabelled, every symmetric component renamed, moved and perturbed 0.1 A -
  and it must come back: 1BRS (three barnase/barstar pairs) 21.7 A as labelled, **0.175 A** permuted; 7BNY (four
  chains, nine sulfates) 37.3 A to 0.174.
- **diffusion** - `--samples` (48) noised copies of the truth, centred over its resolved atoms, rotated and moved,
  sigma = 16 exp(-1.2 + 1.5 N(0, 1)), batched as OpenFold3 batches them: `--sample-batch` (8) samples a denoiser pass
  (`denoiseBatchForward`/`Backward`: one pair conditioning - its gradient summed over the samples and backed once -
  each sample's own noise level in its single conditioning, its atom encoder and decoder, and ONE transformer pass over
  the samples' tokens, the attention batched with the pair logits shared). `--check-batch` holds it to the samples one
  at a time: **1.4e-14** in float64 (`train-of3-f64`), the same function; 2e-5 in float32, the batched sums' order. At
  384 tokens the 48 samples go 3.52 s -> 1.82 s (8 a pass), 1.68 s (16).
- **the update** - `of3train.cuh`: each example's gradient clipped to norm 10 and accumulated (`grad_manager.py`), each
  parameter averaged over the examples it took part in, Adam (0.9, 0.95, eps 1e-8, no decay) after `--batch` examples
  with `AlphaFoldLRScheduler` (1.8e-3, a 1000-step warmup from 0, x0.95 every 50k past 50k), the EMA (0.999);
  `--save`/`--save-every`/`--resume` checkpoint all of it by parameter name.

Before training it holds its trunk to cuda/af3's runTrunk on the first example, pass 0 and three recycled passes (1STP
with its biotin: pair 6.5e-7 / 1.3e-6; 6MRR 3.5e-7 / 3.4e-7). The gate's smoke test (`--smoke`: before and after, on the
first example whole with every draw seeded) asks the loss above its floor - smooth LDDT cannot reach zero, a perfect
pair's eps being 0.804 - to halve: 6MRR, 20 steps at 1e-4, **0.491 -> 0.206** (mse 0.555 -> 0.095, distogram 1.70 ->
0.75, pLDDT CE 2.56 -> 0.82), ~1 s a step at 68 tokens and 4 samples. 🔴 **THE CONFIDENCE TARGETS ARE THE ONES THE HEAD
WAS TRAINED ON, CHECKED RATHER THAN ASSUMED**: before any step the pretrained head's expectations read pLDDT 0.854
against a measured 0.837, PAE 3.18 against 2.92 A, PDE 0.71 against 0.51 A. `--init=of3` starts from OpenFold3's initialisation
instead of the bundle (`linear_init_config.py` and `Linear`'s schemes: LeCun and He fan-in truncated normals, gating
and final zeros, a gating bias 1 and an adaptive-zero gate's -2, LayerNorms 1 and 0 - each weight's kind from the module
its name is, its fan-in recorded where the module builds the linear). The heads' final projections being zero, every
cross entropy starts at log(bins) exactly - distogram, PAE and PDE 4.1589, pLDDT 3.912, resolved 0.693, as read - and
from there 6MRR's mse goes 33.3 -> 12.8 in 40 steps. The Fourier noise embedding is a BUFFER in OpenFold3 (N(0, 1), U(0,
1), never trained): it is frozen here, under either start. - **precision** - float32 by default; `--tf32` puts the GEMMs on the tensor cores (the losses agree to three digits)
  and `--bf16` runs the trunk's 4 MSA blocks and 48 pairformer blocks on their bf16 kernels - OpenFold3 trains under
  bf16 autocast over both - the pair, MSA and single between blocks staying float32. `--check-bf16` holds every gradient
  so computed to float32's: 1.3e-2 to 1.8e-2 (bf16's 8 bits; 8.9e-3 with the pairformer alone), against 1e-5 between two
  float32 runs; the composition check always runs in float32. `--bf16-confidence` puts the confidence head's four
  blocks on them too, which OpenFold3 does not (it turns autocast off around the head in training): 2.3 GB less at 640
  tokens, the losses unchanged to four digits. A 384-token, 48-sample step on 7BNY: 13.8 s float32, 8.2 s `--tf32`,
  6.9 s `--tf32 --bf16`, ~5.1 s with the samples batched.
- **memory** - `--mem` prints, after each phase of a step, the arena's high-water mark over the phase, what the arena
  and the stream pool hold, and the device's use. What a 40 GB A100 holds, in order of what each change bought (1AON's
  8,533 tokens cropped, 1024 MSA rows, 48 samples, `--tf32 --bf16`):
  - **the arena is one growable range** (`Arena::growable`, core.cuh: CUDA's virtual memory, physical pages mapped onto
    its end as the high-water mark rises). 256 MB chunks fragmented - a request larger than a chunk's tail skips it -
    so it held 13 GB for a 9.1 GB peak; now it holds its peak. 384 tokens: 31.9 -> 26.3 GiB.
  - **the MSA blocks in bf16** (above): 1.9 GB at 384 tokens; one MSA block's float32 saves were 5.6 GB.
  - **`--offload-pairs`**: the 53 block inputs the backward recomputes from - n^2 x 128 floats each, 16 GB at 768
    tokens - in pinned host memory, copied out on a second stream while the next block runs and back one block
    ahead of the backward; three on the device in a ring. Identical losses, the same step time (the first step pins
    the host buffer). 384 tokens: 26.3 -> 22.8 GiB.
  - **the transitions in row chunks** (`--transition-chunk=64`, millions of hidden elements a chunk; OpenFold3's
    `transition_ckpt_chunk_size`): the pair transition's hidden layer is 8 pairs wide; chunked, nothing is saved but
    its input and the backward recomputes each chunk's. `--check-chunk` holds every gradient to the whole form's:
    2.1e-14 in float64. 640 tokens: 37.5 -> 34.9 GiB, and faster (16.7 -> 13.6 s).
  - **one allocation a kind**: the parameters and gradients from a growable range, Adam's moments, the accumulator and
    the EMA four slabs (the data-parallel sum reads the accumulator in place, without the flat copy it had), and the
    bundle's device copy given back once read: 14.5 -> 9.8 GB before the first step.
  - `--sample-batch=4` instead of 8: 3 GB less at 640 tokens for 0.6 s a step.

  The longest crop that trains, 48 samples, OpenFold3's PyTorch (`bench_of3_torch.py`, bf16-mixed) beside it (each
  row measured as its changes landed: the A100's 384 and 512 before the chunked transitions and the slabs):

  | | 384 | 512 | 640 | 768 | 896 | 1024 |
  |---|---|---|---|---|---|---|
  | A100 40 GB, before | 5.7 s, 31.9 GiB | out of memory | | | | |
  | A100 40 GB, now (`--offload-pairs --bf16-confidence --sample-batch=4`) | 6.0 s, 22.8 GiB | 9.0 s | 13.3 s, 32.6 GiB | 23.3 s, 32.2 GiB | 30.3 s, 38.3 GiB | |
  | A100, OpenFold3 PyTorch | 12.0 s, ~29 GiB | out of memory | | | | |
  | GH200 96 GB, before | 3.1 s, 31.7 GB | 4.6 s | 7.0 s | 9.9 s, 76.4 GB | | out of memory |
  | GH200 96 GB, `--offload-pairs` | | | | 9.5 s, 47.3 GB | | 16.7 s, 72.4 GB |
  | GH200, OpenFold3 PyTorch | 6.2 s, 27.3 GiB | 10.0 s, 45.7 GiB | 16.2 s, 70.1 GiB | out of memory | | |

  And on the GH200 with every change (`--offload-pairs --bf16-confidence --sample-batch=4`): 1024 tokens 16.7 s and
  49.2 GB, 1280 tokens 27.1 s and 71.0 GB, 1536 tokens 43.4 s and 94.0 GB - twice the crop it took before, and 2.4x
  OpenFold3's 640.

  What is left at 640 tokens on the A100 (28.9 GB): 9.8 GB the parameters, Adam and the EMA; 2.3 GB the step's
  activations; 0.9 GB the stream pool; the arena's 14.3 GB - the MSA block's backward, the confidence head, a
  diffusion chunk, and 3.6 GB the embedder keeps for the whole step.
- **data parallel** - `--world=N --rank=r --rendezvous=<dir> [--dp=nccl|files]` (`of3dist.cuh`): each rank clips and
  accumulates its own examples, then every rank's accumulators and participation counts are summed before the one Adam
  step (`_sync_and_average_grads`), so the weights stay one set without being exchanged. NCCL for a GPU a rank; a file
  transport (through a tmpfs) for ranks sharing a GPU, which is how it is tested here on one A100: two ranks stay in
  sync (checksums equal to the last digit), and two ranks on the same draws (`--same-draws`) land 1.2e-7 from one
  process's step - the backward's atomics. Rank 0 writes the log, the checkpoint and the validation. 🔴 The NCCL
  transport has run only at one rank here (its id exchange, init and both all-reduces): this box has one GPU, and NCCL
  refuses two ranks on one device.
- **end to end** - the whole recipe at OpenFold3's own settings (lr 1.8e-3 and its 1000-step warmup, 48 samples, 384-token
  crops, `--tf32 --bf16`) on the set above, from the bundle: 150 steps over the 54 datapoints, the EMA validated and
  checkpointed every 50 - 6MRR and 1CRN at lDDT 0.8862, 0.8869, 0.8874 (the EMA moving as slowly as a 0.999 decay
  under a warmup at 2.7e-4 should) - and resumed from the step-150 checkpoint at step 150, the schedule continuing
  (lr 2.70e-4 -> 2.86e-4 by step 160). A checkpoint is 5.9 GB (weights, two moments, the EMA): put it on a volume
  with room, such as a tmpfs.
- **validation** - `--val=<file>` (`--val-every`): OpenFold3's validation on the EMA weights, 1 + 3 cycles, a full
  rollout of 5 x 200 steps, each sample's truth permuted, all-atom lDDT (mean, best, the confidence head's top-ranked).
  The pretrained bundle reads 0.831 on 6MRR and 0.846 on 5K9P from their sequences alone.



🔴 **THE DENOISER'S 4e-4 IS THE PORT'S ROUNDING, AND IT WAS SHOWN RATHER THAN ASSUMED.** boltz2 reads 4.1e-4 against
`denoise()` and openbind0 2.0e-4, where every seam up to the encoder's skip is 1e-6 or under. Three measurements put it
in the transformer's float32 and not in a convention: this denoiser in **float64** is as far from the port as in
float32 (4.0e-4, 1.9e-4); its float32 is only 8.5e-5 / 4.4e-5 from its own float64; and `test-af3diff` - the
transformer alone, on random inputs at 68 tokens - reads 7.4e-5 (af3) and 1.45e-4 (openbind0) against the port.
Its bar is 1e-3, and `GRAD_FALSIFY=1` (pre-trunk queries and the padded-key mask flipped) reads **0.74 and 0.88**.
`GRAD_STAGES=1` prints every seam; `GRAD_SENS=1` the port against itself with its inputs moved 1e-7.

🔴 **THE STRUCTURE MODULE ON RANDOM INPUTS IS CHAOTIC, SO ITS GATE READS THE TRUNK'S REAL ONES.** With unit-variance
random single and pair it read 9.3e-4 against the port - and the port moves 1.2e-3 against ITSELF when those inputs are
moved 1e-7: eight iterations of sharp IPA softmaxes amplify a last place by 1e4. On the evoformer's real outputs the
same comparison is 3.5e-7, the port's own sensitivity 3.4e-7.

🔴 **AND THE CONFIDENCE HEAD'S pLDDT IS HELD TO THE PORT'S OWN CONDITIONING, MEASURED EACH RUN.** PAE and PDE agree to
4.5e-6, while intellifold2's pLDDT read 6.6e-4 and openbind0's 3.6e-4 - and the single after the blocks already 3.4e-4,
the pair 3e-6. The single track reads the pair through the single attention's bias: moving the port's PAIR by 1e-7 moves
its own pLDDT logits 2.0e-4 (intellifold2) and 3.9e-3 (openbind0), where moving its single moves them 2.3e-5. So the
gate takes the port against itself with both inputs moved 1e-7 and holds pLDDT to max(1e-4, 4x that).

🔴 **A LAUNCH BOUND BY `blocks(n)` IS A 256-THREAD BOUND.** `blocks(n, t = 256)` - five kernels here launched 128
threads a block over `blocks(n)` blocks, so the second half of their work never ran: ESMFold2's q/k preparation (the atom
stack read 0.12 against the port at its first block) and the AF2 structure module's rigid update and identity, which no
gate saw because both its inputs were under 128 residues. The structure module's gate takes 1BRS (199) now.

`tape.cuh` is a reverse-mode tape over `real` whose weights are Params (cuda/bioemu2/src/vjp.cuh's tape, op for op,
with the weights' gradients): BioEmu2's denoiser is written on it, the static context included, because a graph of
~2,000 ops with a handful of custom ones ported more faithfully op by op than as hand-written backwards.

🔴 **BioEmu2's d/dx IS FLOAT32-LIMITED, AND THE PORT'S TWO FLOAT32 PATHS AGREEING DOES NOT SAY OTHERWISE.** At sigma 5 on
random noise the network's part of d/dx read 2.1e-2 against the port's vjp tape, where the port against itself (x moved
1e-7) reads 3.1e-4 - so it looked like a defect. It is not: both sides cut to the same depth agree to 1.3e-7 with no
layers, the context agrees to 1e-7, the float64 gradcheck holds every tensor, and this denoiser's float32 is 1.0e-2 from
its own float64 - as far as the port's float32 is. The port's vjp and denoise() agree to 8e-6 because they make the
same cuBLAS calls in the same order. A nudge of the INPUT measures conditioning; it does not measure rounding.

## The reference-free gate (`gradcheck.cuh`)

The whole library is written on `real`, float or (`-DGRAD_F64`) double. In double, each tensor's analytic gradient
dotted with a random direction is held to a Richardson-extrapolated central difference of the same forward, the
outputs differenced elementwise (never two large losses subtracted), at h = 1e-7 x the tensor's RMS: a correct
gradient reads ~1e-6, and deleting one term of the LayerNorm backward reads 1.5 (74 tensors fail). Up to three
directions a tensor, because a ReLU input within h of its kink spoils one direction and a wrong gradient all of them.
When all three fail, three more are tried with the direction scaled elementwise by |w| / rms(w): a weight with
near-dead columns (protenix2's `projectAtomFeaturesForAggr` has 130 output channels whose largest entry is 3e-3 of the
tensor's rms) holds ReLU inputs that a step sized to the tensor crosses in EVERY random direction - 14% of that
projection's outputs sit within 1e-6 rms of zero - and stepped in proportion to itself each entry stays on its side.
🔴 A loss over padded positions is not defined: a query masked in every key has logits q.k - 1e9, and even float64
rounds q.k there to 1e-7, so every gate's cotangent is zero on padding - as a loss over real residues is.

## The gate

`oracle_af2_block.py` runs af3-any-model's JAX `EvoformerIteration` in **float64** on random inputs with a padded
tail and a partly masked MSA row, the block's parameters sliced out of the real checkpoint, and takes `jax.vjp` of
`sum(msa_out * ct_msa) + sum(pair_out * ct_pair)` to both inputs and to all 85 parameter tensors. It runs the same
vjp in **float32** too and records, per tensor, how far the reference's own float32 lands from its float64.

`test-af2-block` holds every tensor to `max(1e-4, 3 x JAX float32's error)`: a gradient ill-conditioned in JAX is
ill-conditioned here, and one that is analytically zero (the row attention's `feat_2d_norm/offset` - that norm feeds
only the pair bias, and a softmax gradient sums to zero over the keys) has only rounding on either side. Every
parameter gradient starts as NaN, so one the backward never writes fails.

```
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --out cuda/grad/data/af2-block --seed 1
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --extra --out cuda/grad/data/af2-extra-block
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --opm-last --out cuda/grad/data/af2-block-opmlast
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --pad 0 --L 48 --S 7 --block 20 --out cuda/grad/data/af2-block-nopad
~/.venv-lfjax/bin/python cuda/grad/oracle_af2_block.py --L 48 --S 7 --block 20 --out cuda/grad/data/af2-block-b20
nvcc -O2 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2-block cuda/grad/test_af2_block.cu -lcublas
for d in cuda/grad/data/*/; do cuda/grad/test-af2-block $d | tail -1; done
```

`--nblocks 3 --block 10 --seed 3 --out cuda/grad/data/af2-stack3` dumps a three-block stack, which the same
binary runs through `af2::Stack` - inputs kept, every block replayed before its backward.

All five single-block dumps pass, the worst tensor at 0.26-0.45 of its bar (2026-11-03, A100). The forward agrees with float64 at
1e-7 to 3e-6, inside JAX float32's own error on every case.

🔴 **A FAILING SEED CAN BE A ReLU TIE, NOT A DEFECT.** Two seeds failed first, both the same way: ONE element of
the pair transition's hidden layer within float32 rounding of zero, so its ReLU gate went the other way from
float64 - one channel of `transition1/bias` off by 0.75 (or 2.3) with every other channel within 1e-6, and every
tensor upstream inheriting it. A block has ~900k ReLU inputs, so a tie is a few-percent event, and JAX's float32
happened to land on float64's side. `GRAD_SHOW=<param>` prints a tensor's worst elements; a single outlier
channel is the signature. The gate's seeds are chosen tie-free.

🔴 **AND A FULLY MASKED SOFTMAX ROW HAS NO DEFINED GRADIENT.** A padded residue's query row is masked in every key,
so its logits are `q.k - 1e9`: float64 keeps `q.k`, float32 keeps nothing. The oracle zeroes the cotangent of every
padded output, which is what a loss over real residues does; with it nonzero the comparison was 5e8 on one tensor
and 1e-3 everywhere, measuring nothing.

## Cost

One block trained at the training crops, A100, random weights; the PyTorch bars are the same blocks in PyTorch
2.7 (`bench_torch_af2_block.py`, `bench_torch_af3_block.py`), backward in ms:

| block | float32 | TF32 | **bf16** (`--bf16`) | PyTorch eager, bf16 autocast | PyTorch `torch.compile`, pure bf16 |
|---|---:|---:|---:|---:|---:|
| AF2, L 256, S 128 | 38.1 | 17.6 | **7.22** | 32.6 (4.5x) | 12.7 (1.76x) |
| AF2, L 384, S 128 | | | **14.78** | | 26.3 (1.78x) |
| AF3 pairformer, n 384 | 46.3 | 26.1 | **10.9** | 48.0 (4.4x) | 19.6 (1.80x) |
| AF3 pairformer, n 768 | | | **51.6** | | 128.9 (2.50x) |

(AF3's float32 and TF32 columns are from before the bf16 work; eager PyTorch is 59.6 / 35.7 there on AF2 and
81.2 / 51.3 on AF3.) A stack's AF2 backward keeps its running gradients bf16 from block to block, so the bf16 column
is `blockBackwardRun`'s; `blockBackward`, converting a float cotangent in and float gradients out, is 0.1-0.2 ms
more. The AF2 forward in bf16 is 6.0 ms against compile's 7.2 (12.8 against 16.4 at L 384, before the fused projection); one block's activations are 1.6 GiB at L 256 (compile: 1.9) and 12 GiB for AF3 at n 768 (compile: 16.6).

**A whole stack trained** (`bench-af2-block --bf16 --stack --graph`: AF2's 48 blocks at L 256, rematerialised,
forward then backward; bindcraft2's stack-level tricks, `af2::Stack`):

| | forward | backward | step |
|---|---:|---:|---:|
| every block replayed before its backward | 317 | 676 | 993 ms |
| ...each block's replay and backward one CUDA graph (`graphs`, on a capturable stream) | 318 | 657 | 975 ms |
| ...and **keep tiers** (17 of 48 blocks kept, their backward with no replay) | 318 | 549 | 867 ms |
| ...out-of-place residual links, bf16 LayerNorm-input saves, the trimul's smaller save (20 kept) | 306 | 510 | 816 ms |
| ...and the **graded keep** (19 whole blocks kept, the other 28 keep their attentions' outputs) | 306 | 495 | 801 ms |
| ...the triangle multiplication's projection fused with its prep; the LayerNorms read their transposes | 289 | 485 | **774 ms** |
| ...and the opt-in **bf16 residual states** (`Block::bf16Res`) | 280 | 479 | 759 ms |

- **Keep tiers**: the forward keeps the last blocks' whole saves - the first the backward reaches - in the stack's own
  arena, as many as the free memory holds past one block's working set (measured on block 0; `keepBytes` sets it).
  Exact: a kept block's backward reads what its replay would have recomputed. 1.7 GiB of saves a block at L 256,
  after the forward-only float LayerNorm outputs moved to the stream-ordered pool (15 blocks kept to 17).
- **Graphs**: each block's backward captured and the exec updated in place from block to block (same kernels, other
  pointers), the first kept and first replayed block uncaptured so one-time work happens outside a capture: 2.6%.
  An exec is updated only from a block of its own kind (kept, graded, replayed): a graded block's replay is the plain
  replay less its flash forwards, and cudaGraphExecUpdate from one to the other SUCCEEDED and computed the wrong
  gradient (1.4e-1 in `test-bf16`) rather than refuse.
- **The graded keep** (bindcraft2's per-tensor tiers, `gradedKeep`): the replayed blocks keep their four attentions'
  bf16 output and lse (~70 MB a block at L 256) and their replay skips the flash forwards - ~0.8 ms for 70 MB, three
  times what a whole kept block buys a GB, so it is priced first and the whole blocks take what is left: 815 -> 801 ms.
- **Fewer bytes in the forward** (bindcraft2's `BLK_ONEPF`, its bf16 save-set, `projGateK`'s save): every residual link
  `y = a + b` out of place (a copy then an add was five passes for three) and no copy of the block's input: the
  forward 6.63 -> 6.22 ms. Every LayerNorm on a residual state saves x - mean in bf16 (centred, so the rounding is
  relative to the row's spread) in the pass that reads x, and the float states are temporaries: 17 -> 19 kept blocks.
  The triangle multiplication keeps its two sigmoids and ab = pj g (dgate_pre = dab ab (1 - g), no division) rather
  than the 640-wide pre-activation, and its centre only through its LayerNorm: 20 kept. Not taken: bindcraft2's
  uint8 sigmoids - with weight gradients a small gate's rounding reaches dW of the projection.
- **The triangle multiplication's projection fused with its prep** (bindcraft2's `projGateK`; `triProjK`): xn W on
  mma.sync, its epilogue writing ab channel-major (staged through shared memory) and the two sigmoids from the
  accumulators, so the 640-wide pre-activation never reaches memory - bit-identical to cuBLASLt + the two prep passes,
  377 -> 252 us at L 256 (a 128-row tile, halving the weight's re-reads, was slower: 287). And the column and
  ending-node LayerNorms read their rows transposed (`lnForward`'s `inner`) instead of a transposed copy being made for
  them: the AF2 forward 6.35 -> 6.02 ms, AF3's pairformer 8.5 -> 7.9.
- **The bf16 residual states** (`Block::bf16Res`, off by default): inside a block the states between the links in bf16
  (each added in float and rounded once; the block's input and output stay float), as a pure-bf16 PyTorch block
  holds them. The forward 6.36 -> 6.11 ms; against float32 the outputs move (pair out 1.3e-4 -> 2.9e-3) and the
  gradients hardly (every parameter 4.9e-3 -> 4.4e-3) - `test-bf16` holds it to twice the bars.
- **The bf16 carry** (`bf16Carry`, off by default): the blocks' inputs kept in bf16 and every block reading its input
  widened from that copy - the residual stream rounded to bf16 between blocks, held to float32 at the blocks' bars in
  `test-bf16`. It frees 1.4 of the 3.1 GiB of saved inputs at L 256 and costs 1% (the conversions); it is for the
  lengths where those inputs, quadratic in L, are what limits.
- Already so here: one arena serves the extra-MSA and the main stack (bindcraft2 shares its scratch between them).
  Not taken: bindcraft2's fused residual + gate + narrow (it belongs to a bf16 chain inside the block, not at its
  boundary) and one call for a whole design step (a design loop's, not a kernel's).

**What BF16 mode is.** `--bf16` / `setBf16(true)`: parameters and their gradients float; the big modules - AF2's
attention, transitions, triangle multiplications and OPM, AF3's grid attention, gated transitions and triangle
multiplications - keep their saved activations in bf16 and run every GEMM on bf16 operands with float
accumulation (`linearFwdBf`, `mmATBf`; a bias and a ReLU in the cuBLASLt epilogue); the pair's (and AF2's MSA's)
running gradient is bf16 (bindcraft2's dp), every module reading it in place and adding into it. `test-bf16` holds
whole blocks to float32 by relative rms at PyTorch autocast's own error on the same blocks.

**Where the time went**, in the order it paid (each commit message has its numbers):
- the flash attention in bf16 with FlashAttention-2's structure (`flash.cuh`: cp.async double buffering, m16n8k16
  from registers, ldmatrix.trans, ex2 softmax, q/k/v read in place from the packed projection);
- bf16 storage for the large activations, then bindcraft2's fusions: packed projection GEMMs (the triangle
  multiplication's proj | gate | output gate, AF3's Q | K | V | G), the gates' backward forming the softmax row
  term, the bias gradients summed in the kernels that write the gradients they sum, in-place gradient
  accumulation, bf16 running gradients;
- keeping what the backward used to recompute (the OPM's pair-layout product, AF3's swish product, the triangle
  multiplication's pre-gate output, the gated attention output);
- every bf16 GEMM through cuBLASLt with its algorithm TIMED (`gemmEx`: the first call of a shape runs the 16
  best-ranked candidates and keeps the fastest - cuBLAS's own pick for a bf16 output was often a one-stage s1688
  kernel, 186 us against 95 on the OPM's), which also retired the hand split of the weight gradients: 9.6 -> 8.9 ms;
- the pair bias projections' backward inside the LayerNorm backward that precedes them (`lnBackwardPairBias`:
  dy += dnb W^T and dW from the normalised input it recomputes - no transpose, no rank-H float GEMM, no saved output
  re-read), the running gradients bf16 between blocks, the OPM's four elementwise passes in one, the flash
  backward with chain ids and fixed-point sums compiled out where unused (5%), the attentions' input gradient bf16
  where only a LayerNorm reads it, and the elementwise passes eight bf16 a thread (the gate's bias summed in the
  kernel that writes it): 8.9 -> 8.22 ms;
- the attention backward's atomics priced by the 32-byte sector, not the lane (L2 charges an atomic a sector it
  touches: dQ's fragment pairs put four lanes in each of eight sectors, and trading one value between rows g and
  g ^ 1 puts eight in each of four - 0.68 -> 0.59 ms at the triangle shape alone); the bias gradients added straight
  from the registers that hold dS, unconditionally (a masked element's dS is exactly 0), in place of a pass reading dS
  back out of shared memory; the bias tile read by ldmatrix.trans, dS stored transposed with 32-bit stores into its
  own swizzled tile (read back for dQ by ldmatrix.trans), the clip as two per-key bounds, a query's four terms one
  128-bit load: the attention backward 0.83 -> 0.58 ms at the triangle shape, 17.6 -> 11.6 at AF3's n 768;
  8.22 -> 7.50 ms;
- dQ narrowed into the projections' bf16 copy by the last of a (b, h)'s key blocks to finish, while it is in L2 (a
  separate pass re-read the whole float dQ): 7.50 -> 7.46 ms, AF3 n 768 52.2 -> 51.6. (Counted per query tile, the
  fences after every tile's atomics cost more than the pass: 7.57.) The bias atomics past Q or K are predicated off,
  not pointed at one address: redirected, they serialised there whenever a length is not a multiple of 64 - the
  triangle shape at L 300 took 12.3 ms for 1.0 (`bench-flash`'s tri300; found porting this kernel to bindcraft2);
- the outer product mean's backward projection written straight in the [(i c), (j e)] layout its two GEMMs read
  (`opmDop`: a 64 x 64 mma.sync tile over K = 128 whole, its epilogue placing each column) where cuBLAS wrote
  [(i j), (c e)] and a 134 MB pass turned it round - 391 -> 210 us, bit-identical, and that buffer gone: 7.46 ->
  7.30 ms, 15.17 -> 14.78 at L 384.

**Measured and not taken**: a CUDA graph of the whole backward (2.4%: 11.40 against 11.68 ms on AF2, bindcraft2's
3%); a 32-row query tile for the attention backward (slower capped at 4 CTAs an SM and uncapped); unrolling its two
query halves; keeping a bf16 copy of each LayerNorm's input (as much forward as backward); cuBLASLt's BGRADA bias
gradient (a separate 104 us kernel of its own, even on the timed unsplit GEMM); cuBLASLt's RELU_AUX / DRELU epilogues
for the transitions' mask (both GEMMs slower: the forward 6.6 -> 7.5 ms, the backward 8.7 -> 9.3); the triangle
multiplication's gate backward over 64-row tiles or several 32-row tiles a block (152 -> 158-220 us); bindcraft2's OPM
reassociation (it pays where the alignment is shallower than O = 32); the attention backward's bias gradient summed
over two batch rows a CTA before its atomics (at a fixed register budget a CTA's keys and batch rows trade one for one
between dQ's atomics and the bias's, and two rows' Q and dO tiles do not fit three CTAs' shared memory), and its
Q and dO B fragments by ldmatrix.x4 (no change); the triangle-end and column attentions' transpose back folded into
their LayerNorm backward, each row written to its transposed place (7.46 -> 7.57: the scattered rows lose the tiled
transpose's locality); the pair-bias LayerNorm backward's register cap and grid (no change). `bench-flash --bf16-in` measured nothing of the bias atomics until
its packed q, k, v were random: zero v made every dS 0. `Sdpa::deterministic` sums the attention's bias gradients in fixed point (bitwise
reproducible; 5% of AF3's backward); AF2 training dropout is `Block::dropout`, its masks regenerated, never stored.

**What is left**: the flash backward is 1.5 of AF2's 7.3 ms at L 256 - latency-bound at 12 warps an SM, about twice
the forward without its atomics, and its bias gradient's atomics a quarter of it (the pair bias is shared by every
batch row, so they grow with it; one sector-op an eight elements, the least an f32 atomic does on an A100); the rest
is GEMMs and memory-bound passes around the float residual stream and the LayerNorms, which run near the bytes they
move.

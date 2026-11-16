# ProteinEBM: Langevin dynamics on WebGPU

ProteinEBM (Roney et al., bioRxiv 2025.12.09.693073; jproney/ProteinEBM, **MIT**) is an energy model: given a
sequence and a CA trace `x`, it returns an energy `E(x, t)`. Its score is the gradient of that energy, so running
it means a **forward and a backward pass at every step**. LocalFold runs its Langevin dynamics at a fixed time level,
starting from a structure, the way the vendor's `ProteinEBM_Langevin_Dynamics.ipynb` and `run_dynamics.py` do. Started
2026-11-15 on branch `proteinebm`.

| piece | file |
|---|---|
| the network: forward and input-only backward | `webgpu/proteinebm/network.js` (+ `kernels.js`: LayerNorm with saved stats and its backward, softmax and its backward) |
| dimensions, relative-position classes, the Fourier row, CA reader and writer | `shared/proteinebm/model.js` |
| the Langevin update and the denoised prediction | `shared/proteinebm/langevin.js` |
| the page: weights, the network kept on the device, starting-structure check | `web/proteinebm-model.js`, `runProteinEbm` in `web/app.js` |
| exporter (torch checkpoint to f32 bundle) | `tools/export_proteinebm_model.py` |
| oracle (vendor torch: energy, score, seams, 3 Langevin steps with their noise) | `tools/oracle/dump_proteinebm.py` |
| vendor dynamics for distribution comparisons | `tools/oracle/proteinebm_dynamics.py` |
| the gate | `tools/gpu/check-proteinebm.js` |
| CLI dynamics: trajectory, energy, RMSD, CA–CA, Rg | `tools/gpu/dynamics-proteinebm.js` |
| page driver | `tools/fold-in-page.py --model proteinebm --proteinebm-from <pdb> [--proteinebm-steps N]` |

## The network, and the backward pass

The network is CA-only and small: 84 M parameters, a 512-wide Boltz-1 diffusion transformer of 16 layers, 8 heads
and a 128-wide pair.

- **Inputs.** `[seq_emb | W_n 0.1x | contact_emb | W_sc 0.1 sc]` (1024 wide) goes through the single conditioner:
  LayerNorm, a linear layer plus a Fourier time embedding, then 2 transitions.
- **Pair.** `[s_i | s_j | relpos]` goes through a LayerNorm and a linear layer to make the pair, then 2 transitions.
- **Energy.** `E = Σ|W_r LN(a)|²`.

The coordinates enter in one place only, the coordinate embedding. But `s` depends on `x`, and `z` depends on `s`, so
the gradient flows back through every AdaLN, every gate, the attention biases of all 16 layers, and the pair track.

The backward follows colabfold-cuda's `grad/` library. Each module's forward returns its saved activations; its
backward takes those and the output gradient. It differs in two ways: it computes **input gradients only** (no weight
gradients), and it is recorded once per length and replayed every step. Three folds keep it cheap:

1. **AdaLN.** Its `s_norm` has a weight and no bias. So all 32 AdaLNs share one normalised `shat`, with the weight
   folded into `s_scale`/`s_bias` on the host. The gradient to `shat` is summed over every layer, and a **single**
   LayerNorm backward takes it to `s`.
2. **The attention bias.** It works the same way. Every layer's `proj_z` LayerNorm normalises the same `zhat` with
   its own affine, so `W_z ⊙ w` is folded. `proj_z`'s bias term is constant along a softmax row and drops out.
3. **The pair init LayerNorm** over 1152 features is never materialised.
   - Its statistics come from per-residue sums of `s` and a relative-position table of 132 classes.
   - Its projection is `A_i + B_j + T[class]`.
   - Its backward reduces to per-residue sums of rstd-weighted pair gradients plus three scalars per pair.
     `mean(dxhat · xhat)` is `g · (z0 − c)/1152`, because `Wp' xhat = z0 − c`.
   - So both directions cost O(N²·128), not O(N²·1152).

## Verified

**Against the vendor's torch** (`dump_proteinebm.py` on 6MRR, 68 residues, t = 0.1; relative RMS):

| seam | self-cond off | self-cond current |
|---|---:|---:|
| s (single conditioner) | 3.6e-7 | 3.7e-7 |
| z (pair conditioner) | 6.4e-7 | 6.7e-7 |
| a (a_norm) | 5.9e-7 | 6.2e-7 |
| energy | 6.0e-7 (1243.4956 / 1243.4963) | 5.8e-7 |
| dE/dz | 2.6e-6 | 2.8e-6 |
| dE/ds | 1.7e-6 | 2.0e-6 |
| **score** | **3.6e-6** | **3.6e-6** |
| 3 Langevin steps, positions (the dump's own noise and translations) | ≤ 1.0e-7 | ≤ 1.0e-7 |

The bound is 1e-4. I watched the gate fail once:

- **The break.** I dropped the pair init's per-residue scalars from the backward.
- **What failed.** dE/ds read 3.8e-3 and the score 4.1e-3, against the 1e-4 bound.
- **What still passed.** Every forward row, so a failure points at its stage.

**Dynamics in distribution.** I ran 300 steps from the crystal, two seeds per side. WebGPU used shared/proteinebm's
update; torch used the vendor's model with torch's noise.

| | WebGPU | torch |
|---|---|---|
| energy plateau | 1730–2010 | 1708–1900 |
| noisy state, RMSD to start | 4.6–5.6 Å | 4.2–5.1 Å |
| **denoised**, RMSD to start | 1.5–2.5 Å | 1.4–2.5 Å |
| denoised median CA–CA | 3.58–3.77 Å | 3.61–3.74 Å |
| denoised Rg (start 11.27) | 11.2–11.5 Å | 11.2–11.4 Å |

**Speed.** On an Apple M2 at 68 residues, a step (forward and backward) takes **120 ms**; the forward alone is 61 ms.
The page runs 60 steps in 9.1 s, including recording the length.

## Things to know

- 🔴 **The walker is not the structure.** At t = 0.1 the sampler walks a sample of the *noised* marginal. That is
  about 3.2 Å of noise per coordinate, with consecutive CA atoms ~6.5 Å apart. The structure is `compute_score`'s
  `pred_coords`, the denoised prediction, and that is what `run_dynamics.py` logs. The page and the CLI show it
  (`--show=noisy` gives the walker).
- 🔴 **Self-conditioning: the scripts' was off.**
  - `run_dynamics.py` and `proteinebm_reverse_diffusion.py` pass the coordinates as `sc_ca_t`, and the contacts as
    `contacts`. `compute_energy` reads neither key: it reads `selfcond_coords` and `external_contacts`.
  - So the vendor's scripts actually run with a **zero** self-conditioning channel. Only the Langevin notebook passes
    `selfcond_coords = pos_t`.
  - Both are offered. **Off is the default**, because it is what the scripts ran; the two arms' energies differ
    (1243.5 vs 1347.1 on the crystal).
- 🔴 **The notebook's custom-sequence cell uses the wrong alphabet.** It maps `ACDEFGHIKLMNPQRSTVWY` to 0–19. The
  model's order is AlphaFold's `ARNDCQEGHILKMFPSTWYV`, which `protein_utils.restype_order` is. `aatypeOf` here uses the
  right one.
- **Translation stays.** `center_random_augmentation(rotate=False)` adds an N(0, 1) translation after each recentring.
  The network embeds raw coordinates and is not translation invariant, so the translation is part of the sampler as
  run, and it is kept (`translate: false` drops it).
- **Residue numbering is the PDB's own**, as `residues_to_features` reads it. Only differences matter (relative
  position, clipped at ±32).
- A non-standard residue (MSE included) is unknown (20), as the vendor's `restype_3to1` makes it.

## Open

- **The weights are not published.** The bundle is float32 (321 MiB) and the registry's `remote` is `null`, so the
  site build drops the option and a checkout with the export offers it. Before a remote can be pinned:
  - A quantised bundle (int8, ~85 MiB) needs measuring against this gate's score bound.
  - Uploading it to Hugging Face needs your go-ahead.
- **One protein chain.** The model has chain encoding and relative-position chain classes, and was trained on
  complexes too. The page path refuses more than one chain.
- **Only checkpoint model_1** (`model_1_frozen_1m_md`, the dynamics/ranking one) is exported.
  - The paper's structure-prediction protocol (reverse diffusion from noise) uses model_4 for the initial sampling
    and model_3 for refinement.
  - That protocol is not built.
  - Nor is the Metropolis-adjusted variant (`--metropolis`).
- **Speed is untouched.** Every activation is saved, in float32 everywhere, with no fused kernels; attention is
  materialised (H·N² per layer). The obvious levers:
  - half weights on the forward GEMMs;
  - recomputing rather than saving the transition activations;
  - a fused attention backward.
- **Not run yet:** Firefox, the WebGL2 path (the page refuses WebGPU-only families by name), a native port, and a
  standing npm gate.

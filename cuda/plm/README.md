# Protein language models in CUDA

The language-model towers the folding networks read, kept apart from those networks so that any model can use
one. A tower turns a protein chain's residues into per-residue states; what a model does with them is the
model's own and lives with it.

| tower | file | read by | weights |
|---|---|---|---|
| **ESM-C** 600M, 300M, 6B | `esmc.cuh` | ESMFold2 (`cuda/ef2`), through its shim `cuda/ef2/src/shim.cuh`, which mixes all 37 hidden states | the website's int3 bundles (600M, 300M); 6B as resident int8 codes |
| **ESM2** 3B | `esm2.cuh` | Chai-1 (`cuda/af3 --model=chai1`), its last state onto each protein token | af3-any-model's `lm/esm2.bin.zst`, resident int8 codes (2.7 GB, not 11) |

- `esmcTower(e, ids, seq, onState)` runs ESM-C over the rows and hands every hidden state - the embedding, each
  block's, the last after the final LayerNorm - to `onState`; attention stays within a chain (`seq`). Weights
  under `c/`; its float32 building blocks are cuda/ef2's `ops.cuh`.
- `esm2::` runs ESM2 3B one chain at a time as `[BOS, residues, EOS]` and keeps the last state, as af3-any-model's
  `alphafold3/model/esm.py` does. Weights under `e/`; its kernels are cuda/af3's.

Both are held to their references through the models that read them: ESM-C stage by stage in ESMFold2's oracle
checks (cuda/ef2/README.md, "Exactness"), ESM2 within 1.0e-2 of the float32 model (cuda/af3/README.md,
Chai-1). Moving them here changed no fold: Chai-1 and ESMFold2 are byte-identical across the move.

Chai-1 stays in the AlphaFold 3 family: apart from these embeddings its network is AlphaFold 3's - a pairformer
with an MSA module and templates, the atom transformer, the diffusion sampler, the confidence head - and it runs
on cuda/af3's kernels. A language model is a part a model plugs in, not a family.

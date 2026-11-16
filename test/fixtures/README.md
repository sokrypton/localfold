# Test fixtures, and which script made each

The fixtures are generated once and committed; the scripts that made them are kept in `tools/` as their
provenance. None runs in a gate, and each needs an environment this repository does not install (OpenFold and
PyTorch, or AlphaFold's own JAX/Haiku package and its `params_model_*.npz`). Each script's header has its arguments.

| fixture | made by |
|---|---|
| `openfold-triangle-small/` | `tools/export_openfold_triangle.py` - OpenFold's `TriangleMultiplicationOutgoing` at its deterministic test weights (the manifest's `source`) |
| `alphafold-a3m/model1-reference-59/`, `alphafold-single-sequence/model1-reference-59/`, `alphafold-single-sequence/short-9/` | a captured triangle-multiplication bundle (the capture script is not in the repository), whose `expected.f32.bin` and `referenceExecution` were written by `tools/replay_alphafold_triangle_reference.py` - the bundle replayed through AlphaFold's own Haiku operator on the CPU |
| `evoformer/model1-query-59-stack/` | a captured stack, extended section by section: `structureModule` by `tools/export_alphafold_structure_reference.py`, `templateEmbedding` by `tools/export_alphafold_template_parameters.py`, `confidenceHeads` by `tools/export_confidence_reference.py`, and `residueGeometry` by `tools/export_residue_geometry.py`. `tools/replay_alphafold_evoformer_block.py` replays one main or extra-MSA block from it and stage-captures the block (the per-block fixtures it wrote are not in the repository). Only the input features and geometry tables are on disk - see CLAUDE.md |

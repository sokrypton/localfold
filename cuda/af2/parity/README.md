# AlphaFold 2 parity against ColabFold

The network is held to af3-any-model's JAX AlphaFold 2 by `cuda/af2/oracle.py`, on the features LocalFold's own
exporter wrote. That leaves the half before the network unchecked: how an alignment is merged, cropped, sampled,
masked and clustered into each recycle's features. This directory holds that half to **ColabFold** - the
front end people compare against - running its own code (`colabfold.batch`, alphafold-colabfold) in its own
environment, the way af3-any-model's `vendor_native_parity.py` holds each model to its vendor.

| file | runs in | what it does |
|---|---|---|
| `vendor_features.py` | `~/venv_colabfold` | ColabFold's merged, pre-sampling input for a case (`generate_input_feature`), and optionally that alignment as an A3M |
| `ours_features.mjs` | node | LocalFold's merged, deduplicated input for the same case (`mergeSearchedChains`, `planA3mRecycles`), and optionally the merged A3M |
| `compare_features.py` | python3 | the two side by side: rows as a multiset, row 0, chain ids, residue numbering |
| `ours_draws.mjs` | node | LocalFold's own random draws, per recycle: centres, extra rows, BERT-masked centre codes |
| `sampling_parity.py` | `~/venv_colabfold` | stock AF2's feature functions (the multimer's `nearest_neighbor_clusters` / `create_msa_feat`, or the monomer's TF-pipeline port) applied to LocalFold's draws and compared channel by channel with what `af2-featurise` wrote; `--write` makes a foldable copy with the stock features |
| `vendor_fold.py` | `~/venv_colabfold` | ColabFold's own fold of a case (`colabfold.batch.run`) |
| `webgpu_fold.py` | python3 + node + Chrome | the page's AlphaFold 2 in headless Chrome (`tools/gpu/fold-af2.js --pdb`) |

A case is a JSON file: `sequences` (fold order), `unpaired` and `paired` (one A3M path a chain, the paired ones
row-aligned, as ColabFold's MMseqs2 server returns them), `model_type` (`alphafold2_multimer_v3` or
`alphafold2_ptm`), and optionally `merged` (an alignment already merged, used as it stands).

## Setting up

```
python3 -m venv ~/venv_colabfold && ~/venv_colabfold/bin/pip install -e "<ColabFold checkout>[alphafold]" "jax[cuda12]<0.12"
python3 -m venv ~/venv_dockq && ~/venv_dockq/bin/pip install DockQ
```

DeepMind's float32 parameters go in `~/params/params_model_*.npz` (`vendor_fold.py --data-dir=~`).

🔴 **A LINUX/NVIDIA BOX NEEDS THE DRIVER'S VULKAN HALF FOR THE WEBGPU ARM.** A compute-only driver install has
no `nvidia_icd.json`, so Chrome offers only Mesa and SwiftShader adapters and `gpu-chrome.mjs` hangs rather than
failing. `libnvidia-gl-<branch>` at the running driver's exact version supplies it; Chrome ships its own Vulkan
loader, so the system loader's age (docs/A100.md) does not matter with Chrome 156. Then `Xvfb :99` and
`DISPLAY=:99`, since headless Chrome steers the adapter to software. `webgpu_fold.py` refuses a run that did
not report an `nvidia` or `apple` adapter.

## The three gates

```
# 1. the merge: rows, order, ids
~/venv_colabfold/bin/python cuda/af2/parity/vendor_features.py case.json vendor.npz
node cuda/af2/parity/ours_features.mjs case.json ours.json
python3 cuda/af2/parity/compare_features.py vendor.npz ours.json

# 2. the per-recycle features, from LocalFold's own draws
node cuda/af2/parity/ours_draws.mjs case.json draws.json --recycles=3 --max-msa=508 --max-extra=2048 --seed=0
cuda/featurise/af2-featurise data --bundle=model-multimer --sequence=A:B --a3m=... --paired-a3m=... \
    --recycles=3 --max-msa=508 --max-extra=2048 --seed=0
JAX_PLATFORMS=cpu ~/venv_colabfold/bin/python cuda/af2/parity/sampling_parity.py draws.json data --pool=all

# 3. the network, every recycled pass, against the float32 reference (cuda/af2/oracle.py)
cuda/featurise/af2-featurise ora --weights=cuda/af2/weights-model_1_multimer_v3 ...
python cuda/af2/oracle.py ora --weights ... --model model_1_multimer_v3 --params ~/params --reference <af3-any-model> \
    --out ora/oracle --passes 4
cuda/af2/localfold-af2 ora --weights=... --oracle=ora/oracle --recycles=3
```

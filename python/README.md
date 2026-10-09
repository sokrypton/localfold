# localfold

AlphaFold 3 and its lineage, AlphaFold 2 and ESMFold2, as native GPU programs: an AlphaFold 3 job file (or a
sequence) in, a structure and its confidences out. Prebuilt for every current NVIDIA GPU (CUDA, on Linux) and for
Apple silicon (Metal, on macOS) - one `pip install`, which picks the build for the machine; no compiler, no Python
deep-learning stack, no Node.

```
pip install localfold

localfold-af3 --job=kras.json --out=kras.pdb                  # AlphaFold 3 (or --model=boltz2, chai1, ...)
localfold-af2 --sequence=MKV...:GSH... --search --out=dimer.pdb
localfold-ef2 --sequence=MKV... --out=fold.pdb                # ESMFold2: no alignment needed
```

Or the whole website, on this machine, folding with its GPU:

```
localfold serve                      # opens http://127.0.0.1:8710 - the LocalFold page, every fold native
```

`localfold serve` serves the [LocalFold website](https://localfold.org)'s own page on 127.0.0.1 and sends every fold
it asks for to these native ports instead of the browser's WebGPU: about twice as fast on an Apple M2, every model
(chai-1 and OpenDDE included), folds past the browser's buffer limits, and no weights loaded into a browser tab. The
link it opens carries a token that every request must present, so no other page can use the GPU; Ctrl-C or
Disconnect on the page stops it. `--port`, `--no-open` and `--weights-dir` are its options.

Each command fetches its model's weights the first time (into `~/.cache/localfold`, or `--weights-dir=<dir>`),
featurises the input exactly as the [LocalFold website](https://localfold.org) does, and folds. `--help` lists every
option.

`localfold-fetch <model>` downloads a model's weights ahead of time, and `localfold-fetch ccd` wwPDB's whole chemical
component dictionary (519 MB): ligands and modified residues are then read from it instead of being fetched from the
RCSB one at a time - so once the weights and the dictionary are here, a fold needs no network. `--ccd=<components.cif>`
names a dictionary you already have.

| command | models (`--model=`) | takes |
|---|---|---|
| `localfold-af3` | `af3` (AlphaFold 3, default), `boltz2`, `chai1`, `protenix2`, `intellifold2`, `rosettafold3`, `opendde`, `openbind0` | proteins, DNA, RNA, ligands (CCD or SMILES), ions, glycans, modified residues, covalent bonds, alignments, templates |
| `localfold-af2` | `model_1_ptm` ... `model_5_ptm`, `model_1_multimer_v3` ... `model_5_multimer_v3` | protein chains, alignments, templates |
| `localfold-ef2` | `ef2-fast-600m` (default), `ef2-fast-300m` | proteins, DNA, RNA, ligands, modified residues - from the sequence alone |

`--search` fetches alignments from the ColabFold MMseqs2 server, which **sends your sequences to
api.colabfold.com**; without it a fold runs from the sequence (or the alignments you pass) and nothing leaves the
machine but the first weight download.

## Requirements

On Linux (the CUDA build):

- Linux x86_64 with glibc 2.28 or newer (RHEL/Alma/Rocky 8, Ubuntu 20.04, Debian 11 and later)
- an NVIDIA GPU from the T4 on (Turing, Ampere, Ada, Hopper, Blackwell) and its driver; Blackwell GPUs need a
  driver of the CUDA 12.8 era (570 or newer)
- `curl` and `gzip` (the weight download and the MMseqs2 search)

cuBLAS comes from NVIDIA's `nvidia-cublas-cu12` wheel, installed as a dependency.

On a Mac (the Metal build):

- Apple silicon (M1 or later) and macOS 13 or newer
- nothing else: the binaries use only the system's frameworks. Their GPU kernels are compiled for your Mac the first
  time each is used (cached in `~/.cache/localfold/metal`), so a first fold takes a few seconds longer than the next.

The Metal build is the CUDA programs translated (the repository's `metal/`) and folds the same structures: run on the
NVIDIA build's regression cases, AlphaFold 2 and ESMFold2 agree with an A100's figures to within a few hundredths of
an angstrom, and the AlphaFold 3 lineage to within a seed's spread. `chai1` has not been run on it.

## Weights and licences

AlphaFold 3's parameters are Google DeepMind's, for academic non-commercial use only: the first download asks you to
accept the [terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) (or set
`LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3`). Every other model's weights carry their authors' licences; check them
before use beyond research.

## More

The guides for each family - inputs, options, outputs, intermediate results with `--frames=<dir>` - are in the
repository: [cuda/af3](https://github.com/sokrypton/localfold/tree/main/cuda/af3),
[cuda/af2](https://github.com/sokrypton/localfold/tree/main/cuda/af2),
[cuda/ef2](https://github.com/sokrypton/localfold/tree/main/cuda/ef2).

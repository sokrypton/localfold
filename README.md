# LocalFold

Protein structure prediction that runs **in your browser**, on your own
machine. No account, no queue, no upload of your sequence to a fold server.

**[localfold.org](https://localfold.org)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sokrypton/localfold/blob/main/notebooks/localfold.ipynb)

## What it does

| Page | What it is for |
|---|---|
| **[localfold.org](https://localfold.org)** | Fold a protein, a complex, a ligand or a nucleic acid with AlphaFold 2 or AlphaFold 3 |

Paste a sequence and press Fold. The structure appears as it is being built,
and you can play the trajectory back.

- **Complexes.** Add an entity per chain, with copies for a homo-oligomer.
- **Ligands, DNA and RNA**, and modified residues such as phosphoserine.
- **Templates.** Give a chain a known structure to work from - a PDB entry, an
  AlphaFold DB accession, a file from your machine, or whatever the alignment
  search turns up.
- **Alignments.** Fold from the sequence alone, paste an A3M, upload one, or
  let it search.

## What leaves your machine

The model runs entirely in the browser and your sequence is not sent anywhere
to be folded. Two optional things do use the network, and only when you ask
for them:

- **MSA search** sends your sequence to the public ColabFold server
  (`api.colabfold.com`) to find related sequences. Set the alignment to
  *Single sequence* and nothing is sent.
- **Templates** fetch a structure from the RCSB or AlphaFold DB by the
  identifier you type.

The model weights themselves are downloaded once and cached by the browser.

## What you get out

**Download All** writes a `.zip` in the same layout the AlphaFold 3 server
uses: the structure, the confidence scores, the predicted aligned error and
contact probabilities, the request that produced the fold, and one alignment
per chain.

Dropping that same zip back on the alignment upload box re-folds with exactly
the alignments it recorded.

## Requirements

A browser with WebGPU - Chrome or Edge 113+, or Safari 18+. A discrete or
Apple-silicon GPU makes a large difference; a 60-residue protein folds in a few
seconds on an M2, and a few hundred residues takes minutes.

Nothing is installed, and there is nothing to set up: open the page and fold.

Running it from a checkout, the two Chrome flags worth ~2x on Linux/NVIDIA, and
using the kernels as a library are all in **[docs/RUNNING.md](docs/RUNNING.md)**.

## Acknowledgments and citation

LocalFold began as a fork of **[AlphaFold2 WebGPU](https://github.com/martin-steinegger/alphafold2-webgpu)**
by Martin Steinegger, which is where running AlphaFold 2 in a browser on WebGPU
started. The entire codebase this one grew from is his: the first eight commits
in this repository's history are his commits. The two projects diverged on
2026-08-30 and are developed independently — AlphaFold2 WebGPU is still going,
and parts of LocalFold (the MMseqs2 search client, the row-chunked transition)
are ports of work done there since.

The viewer, the sequence strip, the contact-map panel and the session format
are **[py2Dmol](https://github.com/sokrypton/py2Dmol)**, vendored into
`web/vendor/`: everything on this page that draws a structure is its code
running its own way.

Neither project originates the protein-structure prediction method or the model
parameters. We thank the AlphaFold team at Google DeepMind for developing
AlphaFold and releasing its source code and parameters. The scientific method
should be credited to the original publication:

> Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

Please also consult and cite the [official AlphaFold repository](https://github.com/google-deepmind/alphafold) as appropriate. The bundled AlphaFold parameters remain subject to DeepMind's [CC BY 4.0 parameters license](https://github.com/google-deepmind/alphafold/blob/main/WEIGHTS_LICENSE); this repository does not alter their ownership or license.

The alignment search runs against the public ColabFold MMseqs2 server. If you
use it, please also cite:

> Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679–682 (2022). [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

The AlphaFold 3 parameters carry DeepMind's Prohibited Use Policy and are not
redistributed here.

## For developers

[docs/RUNNING.md](docs/RUNNING.md) is how to run it from a checkout, the
Chrome flags, and the library API. `docs/DEVELOPING.md` has the reference
numbers, benchmarks and deployment notes. `AGENTS.md` is the engineering invariants, `CLAUDE.md` is how
to run things in this checkout, and `docs/AF3.md` is the state of the
AlphaFold 3 port.

## License

[Beerware](LICENSE) (Revision 42) for the code. The model parameters are not
covered by it: the AlphaFold weights remain under DeepMind's
[CC BY 4.0 parameters license](https://github.com/google-deepmind/alphafold/blob/main/WEIGHTS_LICENSE),
and the ESM-C, ESMFold2, OpenBind-0, OpenDDE, Boltz-2 and Protenix-v2
checkpoints under their own. Each is published by its own authors and this
repository redistributes none of them: a bundle names the Hugging Face revision
it fetches from, and the weights travel from there rather than with the site.

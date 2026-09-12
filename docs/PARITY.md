# Parity: what LocalFold checks, against what, and what cannot run

The reference for every model here except AF2 is
[`sokrypton/alphafold3` on `af3-any-model`](https://github.com/sokrypton/alphafold3/tree/af3-any-model),
which has moved a long way since this port was taken from it. This file is the
inventory: what LocalFold's own differential suite covers, what it cannot
currently execute, and what the reference now offers that this repository does
not.

## 🔴 NINETEEN OF TWENTY-ONE AF3 CHECKERS DO NOT RUN ON THIS BOX

Run at `a8a6e70`, each through `tools/gpu-chrome.mjs`, with no arguments:

| | count | why |
|---|---:|---|
| run and pass | **2** | `check-af3-target-feat-gpu.js`, `check-grid-attend-matrix.js` |
| blocked on a model manifest | 15 | `failed to load model manifest: 404` - they open a bundle this box does not have |
| blocked on an oracle dump | 4 | `/oracle-dumps/af3-oracle-atom-f32.json: 404` - needs torch, which does not fit here |

So **"is the port at parity" currently has no answer on this machine**, and the
failure is silent in the worst way: each one exits reporting a 404 rather than a
mismatch, so a suite run reads as twenty-one things that did not object. This is
the repository's own rule - a gate that cannot fail is not a gate - applied to
the whole differential suite at once.

🔴 **AND CLAUDE.md ALREADY RECORDS HALF OF IT.** Two of the checkers were fixed
to take `--model=` precisely because they were pinned to `/model-af3-full-f32/`,
"so on a machine with the published int5 bundle and not the float32 one they 404
rather than skip". The other fifteen were not, and the note stopped at the two.
Five of the twenty-two take `--model=` today.

## What the reference has that this does not

`PARITY.md` there measures level by level, and the structure is worth copying
before the content is:

| level | what it compares |
|---|---|
| L0 | conversion coverage - every checkpoint tensor accounted for, both directions |
| L1 | the trunk: z-init, pairformer, MSA module, template embedder, distogram |
| L1x | the same on a COMPLEX, where the cross-chain terms stop being constant |
| L2 | diffusion conditioning, token transformer, atom encoder and decoder |
| L3 | one full denoise step |
| L4 | the confidence head: PAE, PDE, pLDDT, resolved |
| L5 | an end-to-end fold against an experimental structure |
| L6 | modality: RNA, DNA, ligands, complexes, modified residues |

Its current state is **294 comparisons, 276 OK, 0 FAIL** across 18 model types.
Three pieces of method are worth taking on their own:

- **`gate_applies.py` decides N/A from the registry**, so an empty cell can
  never quietly mean "not run". That is exactly the failure the table above
  documents here.
- **FLOOR is not a pass.** A cell is graded FLOOR when perturbing the gate's own
  input by 1e-6 moves its output further than the port differs - the comparison
  has no resolution left and the number cannot be read. Seventeen of theirs are
  in that state. LocalFold's checkers have no such grade and would report those
  as passes.
- **Graded on correlation AND `max|d|/rms`**, not on one bound. A constant head
  reads r ~ 0 however plausible its mean is.

## Models: four families here, eighteen there

LocalFold ports AF2 (monomer and multimer), AF3, ESMFold2 and OpenDDE, plus the
`openbind0` dialect. The reference's own accuracy table - best A, and it warns
to read it as a band rather than a ranking, since several models are
nondeterministic run to run and best-of-5 is a tail statistic:

| model | best A | here? | weights |
|---|---:|---|---|
| **`boltz2`** | **0.459** | **no** | MIT |
| `alphafold3` | 0.628 | yes | request from DeepMind |
| `opendde` | 0.729 | yes | see upstream |
| **`rosettafold3`** | 0.967 | **no** | see upstream |
| **`protenix2`** | 1.016 | **no** | Apache 2.0 |
| `esmfold2_fast` | 1.182 | yes | MIT |
| `esmfold2` | 1.330 | yes | MIT |
| **`intellifold2`** | 1.512 | **no** | see upstream |
| `openfold3` | 1.547 | no | Apache 2.0 |
| `openbind0` | 1.578 | dialect only | Apache 2.0 |
| `chai1` | 1.713 | no | Apache 2.0 |

And by modality, where the ordering is different:

| model | protein+ligand | complex | RNA | DNA | confidence |
|---|---|---|---|---|---|
| `intellifold2` | **0.301** / 0.881 | **0.92** | 1.621 | 1.95 | 90.4 / r .776 |
| `boltz2` | 0.385 / 0.907 | **0.92** | 1.419 | **1.59** | **96.9 / r .830** |
| `rosettafold3` | 0.464 / 0.889 | 1.12 | **1.143** | 2.67 | 85.2 / r .708 |
| `opendde` | 1.269 / 0.870 | 1.49 | 1.372 | 1.99 | 89.9 / r .635 |

**`boltz2` is the strongest candidate and it is MIT**: best overall, best
confidence correlation, joint-best on complexes, best on DNA. `intellifold2` is
the ligand and complex specialist. Both are the same architecture family this
port already runs - a trunk of pairformer blocks feeding a diffusion head - so
the work is dialect-shaped, which docs/AF3.md's openbind0 section is the
precedent for.

## The order this should be done in

1. **Make the suite runnable before adding to it.** Nineteen dead checkers is a
   larger hole than any single model, and two of the five that take `--model=`
   got that way by someone hitting the 404 in person.
2. **Adopt the applies-matrix idea.** Whatever the level names, the property
   that matters is that a cell which did not run cannot look like one that
   passed.
3. **Then a model.** `boltz2` on the numbers and the licence.

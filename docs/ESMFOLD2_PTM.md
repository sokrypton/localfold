# A phosphoserine, three implementations of ESMFold2

🔴 **THIS IS A BRIEF FOR THE af3-any-model SIDE, NOT A FINDING OF OURS.** The
vendor's own `esm` package places a modified residue correctly; two independent
ports of the same checkpoint do not. Same shape as docs/BOLTZ2_PTM.md, and
found the same way — by running the vendor implementation after the port
disagreed with chemistry.

## The measurement

One job: `GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK`, 58
residues, `SEP@3`, single sequence, no template. Scored as a **mean bond ratio**
against the CCD ideals — 1.000 is perfect — over the phosphoserine's nine bonds.
The control in every row is the same **171 backbone bonds of the 57 unmodified
residues in that same structure**, which is what says the fold is worth reading
at all.

| implementation | checkpoint | control | the SEP |
|---|---|---:|---:|
| `esm` 3.4.1, the vendor | ESMFold2-Experimental-Fast-base600M-step1500k + ESM-C 600M | 1.000 | **0.997** |
| af3-any-model, int8 | the same, via `--model=esmfold2_lm600m` | 0.996 | 1.548 |
| af3-any-model, fp32 | the same | 0.996 | **1.542** |
| LocalFold (this port) | the same, WebGPU | 0.999 | **2.349** |

The vendor's worst single bond is 2.6% out. af3-any-model's `CB-OG` is 3.26 Å
against a 1.417 ideal and its `OG-P` 3.75 against 1.610. This port's `OG-P`
reaches 4.25.

**Not the quantisation**: int8 1.548 against fp32 1.542. **Not one unlucky
sample**: all five of af3-any-model's samples are 1.474 / 1.542 / 1.715 / 1.771
/ 1.833 with controls 0.996-1.001. **Not the sampler's budget**: on this port,
raising the steps 11 → 45 → 138 moves the SEP 1.399 → 1.883 → 2.279 while the
control holds at 1.00 — a converging sampler moving away from the chemistry.

## Reproducing it

```bash
# the vendor
~/venv_ef2/bin/python tools/esmc/probe-esmfold2-modified.py

# af3-any-model, from a clone of sokrypton/alphafold3
python run_alphafold.py --model=esmfold2_lm600m --use_esm_embeddings \
  --norun_data_pipeline --weights_precision=fp32 \
  --json_path=sep_job.json --output_dir=out/
```

🔴 The job JSON needs `"unpairedMsa": ""`, `"pairedMsa": ""` and
`"templates": []` explicitly, or `validate_fold_input` refuses it with "Protein
chain 1 is missing unpaired MSA" — an empty string is "fold with no alignment"
and an absent field is "go and search", which is the same distinction
`web/job-json.js` documents on the reading side.

## What is already excluded, on this port's side

Checked against the batch `featuriseForEsmfold2` actually produces:

- the SEP's **reference conformer is exact** — N-CA 1.469, CA-CB 1.529, CB-OG
  1.428, OG-P 1.609, P-O1P 1.480
- its ten atoms **share one `refSpaceUid`**, as a ligand's six do
- its **bonds are in the matrix and are read** — `modifiedSpans` is in
  `bondedGroups`, and `tokenBonds` is uploaded and multiplied by
  `featuriser/tokenBonds` in the trunk
- **`molType` is PROTEIN and that is correct.** It is built from `ligandSpans`
  alone, so a modification's atom tokens come back PROTEIN — which looks like
  the bug and is not: the vendor reports the same **67 tokens, ten tokens of one
  atom each, all `mol_type` 0**, and its tokeniser's docstring says "Modified
  residues (from modifications) are atom-tokenized (1 token per atom)"
- **a plain CCD ligand in the same fold is placed correctly** — a glycerol at
  0.958 beside the SEP at 2.349, so it is not an atom decoder that cannot place
  a rigid group

## The discriminator that would halve the search

**Does af3-any-model's esmfold2 misplace a plain CCD LIGAND too, or only an
atomised residue inside a polymer?** On this port the ligand is fine and the
modification is not, which is what localises it to the atomised-residue path
rather than to atom handling in general. If af3-any-model shows the same split,
the two ports share a cause and it is upstream of both; if its ligand is also
out, they are two different bugs that happen to land on the same residue.

## Two traps this cost

🔴 **THE ATOM NAMES.** The vendor's first number was **1.101**, which reads as
"the vendor is imperfect here too" and would have closed the question the wrong
way. The names came from slicing the CCD list to the first ten, which keeps
**OXT** — a leaving atom a mid-chain residue drops — so every name after it
shifted by one and `OG-P` was measured against the phosphorus's neighbour.
Through the vendor's own `get_ccd_leaving_atoms` the order is
`N,CA,CB,OG,C,O,P,O1P,O2P,O3P` and the mean is 0.997. **A near-miss number is
the dangerous kind.**

🔴 **AND A SORTED-DISTANCE MATCH IS NOT A SHAPE MATCH.** Comparing the 45
intra-residue distances with labels ignored gives rms 0.592 Å against the
labelled 1.892, which reads as "right shape, wrong names — a permutation bug".
It is not: any compact blob of ten atoms matches another to about that, and a
backbone superposition refutes it outright (N 0.57, CA 0.80, C 0.74 Å, side
chain 1.4-4.1). Match the labels before believing a shape.

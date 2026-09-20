# A phosphoserine, three implementations of ESMFold2

🔴 **THIS IS A BRIEF FOR THE `sokrypton/alphafold3` SIDE - af3-any-model,
now merged to `main` - AND NOT A FINDING OF OURS.** The
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

| implementation | control | the SEP | a glycerol |
|---|---:|---:|---:|
| `esm` 3.4.1, the vendor | 1.000 | **1.002** | **0.986** |
| sokrypton/alphafold3, before | 0.999 | 1.446 | 1.346 |
| **sokrypton/alphafold3, after `7df8d97`** | 1.002 | **0.991** | **0.991** |
| LocalFold, before | 0.999 | 2.349 | 0.958 |
| **LocalFold, after** | 1.005 | **0.994** | 0.974 |

🔴 **BOTH ARE FIXED, AND THE CAUSES WERE NOT THE SAME ONE.** sokrypton's side
took three (`7df8d97`): a ref-pos table that rewrote the six atoms a SER has and
left the phosphate on its CCD frame, an unsymmetrised token-bond matrix that
left the glycerol half-bonded, and the restype below. This port had only the
last of them - its conformer was already exact and it already symmetrised - and
its glycerol was correct throughout, which is why its ligand column never
moved.

All on `ESMFold2-Experimental-Fast-base600M-step1500k` + ESM-C 600M, af3-any-model
through `--model=esmfold2_lm600m`. The glycerol column is a plain CCD ligand in
the SAME job.

### The cause, on this port: an atomised residue is UNKNOWN, not its parent

AF3 gives every atom token of a modified residue the **parent's** restype, so a
phosphoserine's ten tokens all said SER and `featuriseForEsmfold2` passed that
through: the model was handed ten single atoms labelled as ten serines. The
vendor's own atomised branch is explicit -
`TokenInfo(res_type=PROTEIN_UNK_RES_TYPE, input_id=DNA_RNA_LIGAND_INPUT_ID)` -
so **22 and 24**, where this port wrote 17 (SER) and its ESM id.

🔴 **AND 24 IS NOT WHAT THE RESTYPE TABLE GIVES, WHICH IS THE SUBTLE HALF.**
`AATYPE_TO_ESM_ID` maps the unknown restype to `<unk>` (3) *deliberately*: an
`X` written in a sequence is a residue nobody has identified and the tower was
trained to see `<unk>` there. An atomised residue is not that - it is a row of
single ATOMS, and the tower sees what it sees for a ligand. Two unknowns, two
tokens; the fix cannot be one lookup, and the note on that table stays true of
its own case.

🔴 **AND THE SECOND HALF OF SOKRYPTON'S BOND FIX APPLIED AFTER ALL.** This port
already symmetrised, so the block counts matched - but the peptide bonds tying
the atomised residue to its chain NEIGHBOURS were missing here too. Counted
against `esm` 3.4.1's own featuriser on a SEP + glycerol job:

| | vendor | this port, before | after |
|---|---:|---:|---:|
| inside the SEP block | 18 | 18 | 18 |
| inside the glycerol | 10 | 10 | 10 |
| SEP to its neighbours | 2 | **0** | **2** |

`atomizedBackboneBonds` existed and was rf3's alone. The fold barely moves on it
- SEP 0.994 → 0.989, the junction CA-CA 3.73/3.67 → 3.67/3.80 against a 3.8
ideal, one sample each, which is noise - and it is taken because the **feature
now matches**, not because the number did. AF3 extracts inter-residue bonds only
where one side is a LIGAND chain, so a residue atomised inside a polymer loses
its backbone bond to each neighbour; `featurise.js` already called that "a
ligand floating beside the chain as far as the pair track is concerned".

Gated in `test/esmfold2-lm-mask.test.js` - both fields, and the whole row
asserted rather than "not 22", since a row that went wrong some other way would
pass that. Watched failing with the bug restored.

🔴 **AND THE LIGAND COLUMN IS WHY THE TWO PORTS NEEDED DIFFERENT FIXES.** A ligand is
atomised exactly as a modified residue is - one token per atom, one shared
reference frame - but it sits in its own chain. af3-any-model misplaces **both**
(1.446 and 1.346), so its problem is atom placement in general. This port
misplaces **only the modification** (2.349 against a correct 0.958), so its
problem is an atomised residue *inside a polymer chain*. **Fixing one will not
fix the other**, and the earlier version of this file guessed the opposite.

The vendor's worst single bond is 2.6% out. af3-any-model's `CB-OG` is 3.26 Å
against a 1.417 ideal and its `OG-P` 3.75 against 1.610. This port's `OG-P`
reaches 4.25.

🔴 **AND NOT THE STEP COUNT, ASKED PROPERLY.** The checkpoint's own default is
`inference_num_steps: 15`, so all three rows above were already at ~15 - but
that is an argument, not a measurement. Swept on the vendor, same job:

| vendor steps | control | SEP | GOL |
|---:|---:|---:|---:|
| 11 | **2.155** | 1.290 | 3.653 |
| 15 | 1.001 | **1.003** | 0.988 |
| 64 | 1.001 | **1.002** | 0.981 |
| 138 | 1.001 | **1.000** | 0.984 |

Below 15 the vendor breaks *everything* - its CONTROL goes to 2.155 - and from
15 up it is flat and correct. So the matched comparison is at 138, where both
sides are converged:

| at 138 steps | control | the SEP | a glycerol |
|---|---:|---:|---:|
| the vendor | 1.001 | **1.000** | 0.984 |
| LocalFold | 0.999 | **2.349** | 0.958 |

**The controls agree to 0.002 and the phosphoserine differs by 2.35x.** On this
port more steps make the SEP worse (1.399 → 1.883 → 2.279 at 11 → 45 → 138)
while the vendor's is flat, which is the opposite of what a step shortage looks
like.

🔴 **AND OUR 11 STEPS IS NOT THE VENDOR'S 11.** At 11 the vendor's control is
2.155 and this port's is 0.994 - a fold that is fine where the vendor's has come
apart - so the two schedules are not the same walk at the same count, and
`actualSteps` mapping the preset 15 to 11 is not a like-for-like number. It does
not affect the conclusion, which rests on the matched 138 row, but it is worth
knowing before anyone compares step counts across the two.

**Not the quantisation**: int8 1.548 against fp32 1.542. **Not one unlucky
sample**: all five of af3-any-model's samples are 1.474 / 1.542 / 1.715 / 1.771
/ 1.833 with controls 0.996-1.001. **Not the sampler's budget**: on this port,
raising the steps 11 → 45 → 138 moves the SEP 1.399 → 1.883 → 2.279 while the
control holds at 1.00 — a converging sampler moving away from the chemistry.

## Reproducing it

```bash
# the vendor, with the ligand arm that separates the two bugs
~/venv_ef2/bin/python tools/esmc/probe-esmfold2-modified.py --ligand=GOL

# af3-any-model - which is sokrypton/alphafold3 MAIN now, not a branch
git clone --depth 1 https://github.com/sokrypton/alphafold3
python run_alphafold.py --model=esmfold2_lm600m --use_esm_embeddings \
  --norun_data_pipeline --weights_precision=fp32 \
  --json_path=sep_job.json --output_dir=out/
```

🔴 **CLONE IT; DO NOT USE THE COPIES ON THIS BOX.** `~/af3fork` and
`~/af3src/alphafold3-af3-any-model` are non-git SNAPSHOTS and are behind main -
`run_alphafold.py` differs. Measured here against main at `301cc16`.

🔴 **AND ITS LAUNCHER WANTS ITS OWN PACKAGE VERSION.** A fresh clone run against
the `alphafold3` installed in a venv dies on `model_registry.AF2_SPECS`; with
`PYTHONPATH=<clone>/src` it then wants the compiled `cpp` extension and the
generated `*.pickle` data, which are build products and live only in the
installed copy. Symlinking those two across the clone is enough for an
inference run, and is what these numbers were taken through.

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

## The discriminator, run

**Does af3-any-model's esmfold2 misplace a plain CCD ligand too, or only an
atomised residue inside a polymer?** Asked because on this port the ligand is
fine and the modification is not. **It misplaces both** - so the two ports do
not share a cause, and the table above is two findings rather than one:

- **af3-any-model**: any atomised entity, ligand or residue. The vendor places
  both, so this is the port.
- **LocalFold**: atomised residues only. Its ligand path is correct at 0.958,
  which is the control that localises it.

Each needs its own fix. The af3-any-model side is the wider one and, because
this port checks itself against af3-any-model for the AF3 lineage, the one whose
correctness the rest of that comparison rests on.

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

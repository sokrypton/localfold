# af3-any-model's `boltz2` tears apart a modified residue

Genuine Boltz-2 does not. The fault is in the port. This is a brief with the
measurements, the reproduction, what is already excluded, and where to look.

Written from the LocalFold side (WebGPU port of the same lineage), which
inherits the defect. **I only measured — I have not read the port's code.**

## The defect

Fold `ACSEFGHIKLWY` with a phosphoserine (`SEP`) at position 3, no MSA, no
template. In `boltz2` the modified residue's own bonds come out ~2.7x too long.
It is not subtle, and it is not a sampler artefact: it gets **worse** with more
sampling steps, which is what an unconstrained residue drifting looks like.

## Three references, one target, the same nine bonds

| | mean bond ratio | bond rms |
|---|---:|---:|
| genuine Boltz-2 2.2.1, 3 samples | **0.986 / 0.996 / 1.011** | 0.043–0.076 Å |
| af3-any-model `boltz2` | **2.705** | 2.99 Å |
| af3-any-model `alphafold3` | 0.988 | 0.143 Å |
| LocalFold `boltz2` (inherits it) | 1.813 | — |
| LocalFold, every other model | 0.73–1.16 | — |

Worst bonds in `boltz2`: **N-CA 4.928** against an ideal 1.425, **CA-CB 6.191**
against 1.528.

So Boltz-2 can place a PTM perfectly well, and the harness is not the cause —
af3-any-model's own AlphaFold 3 is correct on the identical target through the
identical code.

## Reproduce both sides

Everything below is already installed on this box.

```bash
# af3-any-model: the broken arm and its control
cd ~/alphafold3
PYTHONPATH=src:. MODEL=boltz2     ~/venv/bin/python dev/oracles/probe_af3_ptm_bonds.py
PYTHONPATH=src:. MODEL=alphafold3 ~/venv/bin/python dev/oracles/probe_af3_ptm_bonds.py

# genuine Boltz-2: the ground truth
~/venv_boltz/bin/boltz predict /tmp/sep_boltz.yaml --out_dir /tmp/bz \
  --override --diffusion_samples 3 --no_kernels
~/venv_boltz/bin/python /tmp/score_modified_cif.py \
  /tmp/bz/boltz_results_sep_boltz/predictions/sep_boltz/sep_boltz_model_0.cif 3 SEP
```

Two practical notes:

- `--no_kernels` is **required**. Boltz's fused triangle path imports
  `cuequivariance_torch`, which is not a default dependency; without the flag it
  dies in `ModuleNotFoundError` partway through the first batch.
- The venv is `~/venv_boltz`, deliberately separate from `~/venv`, so the JAX
  environment is untouched.

## Where to look first

`boltz2` is the only family with **`tokenBondsTypeEmbed`** — its z-init reads a
*second* plane carrying the bond order/type, where every other family reads only
the contact flag. It is therefore the only family that would notice that plane
being empty or wrong for an atomised residue, and it is the only family that
breaks.

That exact channel has bitten once before in a downstream port of the same
architecture: parsed, forwarded, and never filled — five consumers and no
producer — and the symptom was a glycerol coming apart at bond rms 3.6 Å while
pLDDT read 92.4.

### The discriminator worth running first

**Does `boltz2` break a plain ligand (e.g. `GOL`) too, or only an atomised
residue inside a polymer?** A ligand is its own entity; a modified residue is
atomised tokens *within* a chain. If the ligand is fine and only the PTM breaks,
the fault is in the atomised-residue-in-a-chain path rather than in bond
handling generally. That halves the search and is one run.

## A trap in the measurement

**Do not select the residue's atoms by name across tokens.** An atomised residue
is one token per atom, and `ref_pos` is each residue's *own local frame*, so a
cross-residue distance is not a distance — it lands inside any plausible bond
window by accident.

My first probe did exactly that and reported `CB-OG = 16.321 Å` on a fold whose
N-CA, CA-CB, CA-C and C-O were all within 3%. "The reference does it too, mean
ratio 2.894" was one step from being written down and would have been wrong.

The tell was in the output all along: the **ideal** read 1.571 where the
dictionary says 1.428. *When the ideal is wrong, the pair is wrong.* The probe
now selects by `residue_index`, which is what actually identifies an atomised
residue's atoms.

The scorers are also three separate implementations on purpose
(`probe_af3_ptm_bonds.py`, `/tmp/score_modified_cif.py`, and LocalFold's
`bond-geometry.js`); they agree on the models they share. One scorer across
three references would let a scorer bug masquerade as a model difference.

## What is already excluded

On the LocalFold side, which reproduces the defect:

| | |
|---|---|
| the bond matrix | 9 bonded pairs and 9 bond-order entries, byte-identical across af3, boltz2 and intellifold2; both planes present |
| `ref_pos` | 10 live slots, ONE `ref_space_uid`, 6.924 Å across the residue — identical for all four models, so conformer centring is not collapsing it |
| the weights | the four duplicated `_1` diffusion tensors are byte-identical in boltz2's bundle (only AlphaFold 3's differ), so a recent weight-name change there is a proven no-op |

If af3-any-model's batch is likewise correct, the divergence is in the forward.

## Verifying a fix

- `MODEL=boltz2` should read mean ratio ≈ 1.0 at bond rms < 0.15 Å.
- `MODEL=alphafold3` **must stay at 0.988** — a change that moves the control
  has moved something shared, and is not this fix.
- Worth checking a ligand and a plain protein have not moved either.

## Downstream

LocalFold pins this rather than papering over it: `npm run test:modified`
carries `boltz2` as an expected failure with these numbers as its evidence, and
reports `FIXED` — telling the reader to delete the entry rather than widen it —
the moment it comes good. So the fix will be noticed on the next run.

## 2026-09-29: fixed upstream, as measured through the JAX backend

Re-measured through LocalFold's JAX worker (sokrypton/alphafold3 `colab`
branch over the `alphafold3-colabfold` 3.1.11 wheel, on the A100), SEP at
position 3 of 6MRR, single sequence, one sample: af3-any-model's **boltz2 reads
a mean bond ratio of 0.994** (worst 1.010) - in line with genuine Boltz-2's
0.986-1.011 - where this brief was written against 2.705. The other six
AF3-lineage families read 0.958-1.023 on the same job. So the reference's
boltz2 no longer inflates a modified residue; this brief is history.

## 2026-10-02: a second residue, 3-hydroxyproline - and the reference has it too

Found by folding AlphaFold 3's kitchen-sink example (`alphafold_input.json`) through all seven
families natively: boltz2 alone tore its modified residues apart - ligand-class bond rms **0.500 Å**
against 0.04-0.10 for the other six - and the culprits were HY3 (1.12), P1L (1.07), 6OG and 2MG
(0.82), where 5MC, the glycan, HEM and the SMILES ATP were all clean.

Isolated on 6MRR with a proline introduced (`P..P..`), single sequence, bond rms of the modified
residue (`native/af3/bonds.mjs`):

| | AlphaFold 3 | boltz2 |
|---|---:|---:|
| SEP@3 | 0.048 | 0.078 |
| HYP@1 (4-hydroxyproline) | 0.082 | 0.160 |
| HY3@1 (3-hydroxyproline) | 0.087 | **0.463** |
| HY3@4 | 0.060 | **0.897** |

- **The featurisation is not it.** boltz2's batch for HY3@4 is exact against af3-any-model's own
  (`tools/check-batch-fields.js --model=boltz2 --target=hy3`, 47 fields, the conformer floor aside):
  one token, the component's slots, its restype and bonds.
- **The reference does it too.** af3-any-model's own boltz2 (`run_alphafold.py --model=boltz2`, the
  int8 weights in `~/lfjax/weights`, HY3 given as `userCCD` because the pip CCD lacks it) reads
  **0.994 / 0.907 / 0.703 Å** over three samples - the same tearing. So LocalFold's boltz2 is
  faithful to its reference here, and the question is the one the SEP case asked: does GENUINE
  Boltz-2 place a 3-hydroxyproline? 🔴 **NO - MEASURED 2026-10-02, AND IT IS THE MODEL.** Boltz 2.2.1
  installed on the A100 (`~/venv_boltz`, torch 2.7.1+cu126 - the default wheel is built for a newer
  CUDA than this driver's 12.8 and dies in `_cuda_init`), the same 68-mer with HY3 at 4, single
  sequence, three samples, `--no_kernels`: HY3 bond rms **1.312 / 1.169 / 1.010 Å** over its 8 bonds,
  the ring collapsed (C3, C4 and C5 within 0.4 Å of each other), while the same run's SEP@3 control is
  **0.121 / 0.113 / 0.046 Å** and every protein class is 0.008-0.051. So af3-any-model's boltz2 (0.70-0.99)
  and this port's (0.897) inherit it from Boltz-2 itself; there is no port defect to find here, and
  boltz2 should not be expected to hold a hydroxyproline together. Scored with `native/af3/bonds.mjs`
  on gemmi's PDB of each mmCIF.

**And the modified BASES the same way, with the batch exact.** After boltz2's batch was made exact
on modified bases too (`--target=dna-5cm|rna-mods`, four conventions fixed - the profile at the
parent nucleotide, atoms past the 24 dense slots dropped as the reference drops them, the
representative falling back to the first held atom), the kitchen-sink job still reads, for boltz2:
6OG **0.834**, 2MG **0.823**, 6MA 0.229, 5MC 0.067 Å bond rms - where AlphaFold 3 and the other five
families are 0.04-0.10 on every one. 6OG and 2MG are the 25-atom guanosines whose one-token form
loses a ring atom by construction, so they cannot be whole; 6MA is the open question beside HY3. 🔴 **6MA IS NOT A DEFECT EITHER, MEASURED 2026-10-02**: a 14-bp duplex with 6MA at 2 and 9 of one
strand, single sequence, three samples - genuine Boltz-2 **0.064 / 0.132 / 0.139 Å** bond rms over the two
residues' 48 bonds, this port's native boltz2 **0.104 / 0.122 / 0.135 Å** on the identical input. The
kitchen sink's 0.229 is one sample of a harder job; on a like-for-like target the two are one band.

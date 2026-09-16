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

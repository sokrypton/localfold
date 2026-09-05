# ESMFold2 and ESM-C in LocalFold

Nothing here ships yet. This is the answer to one question - **can a protein
language model replace the MSA search in a browser** - and the measurements that
answer it.

`docs/AF3.md` holds the AF3 port's state; this holds the ESMFold2 investigation's.
The tools are `tools/esmc/`.

## Why ask

LocalFold's alignment comes from an MMseqs2 server that queues, and a target
with no homologues gets nothing useful back from it at any wait. ESMFold2-Fast
folds from ESM-C's hidden states instead of from an alignment: no search, no
round trip, no database. The whole question is whether the language model fits.

## The models, and which one this is about

Biohub publish ESMFold2 (MSA-capable) and **ESMFold2-Fast** (single sequence,
no MSA encoder at all), each in a released and an *experimental* line, and the
experimental line additionally as a per-LM-size ablation series that exists to
reproduce the paper.

| | LM | folding model | tower |
|---|---|---|---|
| `ESMFold2-Experimental-Fast` | ESM-C 6B | 179.3 M | 6352 M |
| `...-Fast-base600M-step1500k` | ESM-C 600M | 171.2 M | **573.3 M** |
| `...-Fast-base300M-step1500k` | ESM-C 300M | 171.1 M | 333 M |

The folding model is the same size in all three; only the tower changes. All are
ungated and MIT - there is no terms dialog to answer and nothing
`build_site.py` has to refuse, which is a real difference from AF3.

🔴 **THE TOWER IS PAIRED WITH ONE CHECKPOINT OF ITSELF, NOT WITH A WIDTH.** The
folding model's layer mix is a `(n_layers + 1,)` vector trained against exactly
the tower named in its `esmc_id` - `biohub/ESMC-600M-1500000`, which is NOT the
general-release `esmc-600m-2024-12`. `tools/esmc/fetch.py` reads the pairing out
of the folding model's own config rather than writing it down twice.

🔴 **AND THE `-step` CHECKPOINTS HAVE NO CONFIDENCE HEAD.**
`confidence_head.enabled: false`, zero confidence tensors in the file - so no
pLDDT, no PAE, no pTM. The fold succeeds and `output_to_pdb` raises `KeyError:
'plddt'` on the way out. LocalFold colours by pLDDT and its scores card is built
on it, so a shipped 600M path needs a checkpoint that has one; the ablation
series is for measuring, not for shipping.

## The first 2x is not compression

🔴 **THE CHECKPOINT IS bfloat16 STORED AS float32.** Not one of the 95.6M
weights sampled has any of its low sixteen mantissa bits set. The tower was
trained in bfloat16 and widened on the way out, and ESMFold2's own forward
casts it straight back (`torch.autocast(device_type=..., dtype=torch.bfloat16)`
around the ESM-C call; the experimental loader's `finalize_esmc` is
`self.esmc.bfloat16()`). So half of the 2.30 GB download is zeros.

Consequences, both measured rather than argued:

* the bfloat16 arm reads relRMS **exactly 0.00e+00** on the weights and on every
  downstream column, which is what a correct harness must say and is also how
  this was found - a scheme that costs nothing is either free or not running.
* float16 reads **3.2e-9**, which is not rounding. bfloat16 holds 7 mantissa
  bits and float16 holds 10, so every value in range is exact; the residual is
  the tail below float16's subnormal floor, and the tower's smallest weights go
  down to 3.9e-10.

**So quote a scheme against the 16 bits that are really there.** A table
starting at "32.00 bits" flatters every row in it by a factor of two.

## What compression costs, on the tensor ESMFold2 actually reads

`python3 tools/esmc/probe-esmc-compression.py`. Three targets - ubiquitin and
the two crystal structures the AF3 side already scores against - and the error
is reported at four depths, because they do not agree and only the last one is
the question:

* `weights` - what a weight-space study would report.
* `states` - the flat mean over ESM-C's 37 hidden states.
* `mixed` - the same, weighted by ESMFold2's learned softmax over them.
* `pair` - the pair representation the folding trunk receives. **This is the
  column that prices a scheme.**

| scheme | bits/w | MiB | weights | states | mixed | pair |
|---|---:|---:|---:|---:|---:|---:|
| float32 (control) | 32.00 | 2187 | 0 | 0 | 0 | 0 |
| **bfloat16** | 16.00 | 1094 | **0** | **0** | **0** | **0** |
| float16 | 16.00 | 1094 | 3.5e-9 | 6.7e-7 | 1.3e-6 | 1.2e-6 |
| int8 g64 sym | 8.25 | 564 | 6.1e-3 | 5.4e-3 | 9.6e-3 | 7.4e-3 |
| int8 g128 asym | 8.25 | 564 | 6.0e-3 | 5.3e-3 | 9.2e-3 | 7.3e-3 |
| int6 g32 asym | 7.00 | 478 | 1.9e-2 | 1.6e-2 | 2.8e-2 | 2.2e-2 |
| int6 g64 asym | 6.50 | 444 | 2.2e-2 | 1.8e-2 | 3.2e-2 | 2.5e-2 |
| **int5 g32 asym** | 6.00 | 410 | 3.8e-2 | 3.3e-2 | 5.7e-2 | **4.4e-2** |
| int5 g32 asym+search | 6.00 | 410 | 3.7e-2 | 3.5e-2 | 5.8e-2 | 4.4e-2 |
| int5 g64 asym+2 outliers | 5.69 | 389 | 3.5e-2 | 3.1e-2 | 5.4e-2 | 4.4e-2 |
| int5 g64 asym | 5.50 | 376 | 4.4e-2 | 3.7e-2 | 6.4e-2 | 4.9e-2 |
| int4 g64 asym+2 outliers | 5.19 | 355 | 7.2e-2 | 6.1e-2 | 1.1e-1 | 8.0e-2 |
| int4 g32 asym | 5.00 | 342 | 7.9e-2 | 6.9e-2 | 1.2e-1 | 9.4e-2 |
| int4 g32 asym+search | 5.00 | 342 | 7.6e-2 | 6.9e-2 | 1.2e-1 | 8.8e-2 |
| int4 g64 asym | 4.50 | 308 | 9.1e-2 | 7.7e-2 | 1.3e-1 | 1.0e-1 |
| int3 g32 asym | 4.00 | 273 | 1.7e-1 | 1.5e-1 | 2.5e-1 | 1.9e-1 |
| int3 g32 asym+search | 4.00 | 273 | 1.6e-1 | 1.6e-1 | 2.5e-1 | 1.8e-1 |
| int2 g32 asym | 3.00 | 205 | 4.0e-1 | 4.1e-1 | 7.4e-1 | 5.2e-1 |

Three things worth taking from it.

**ESM-C quantises like AF3 does.** int5 group 32 asymmetric costs relRMS 3.8e-2
in weight space here against 4.3e-2 on AF3's six biggest tensors
(`tools/analyse_quantisation.py`) - the same scheme, the same neighbourhood, a
different model. The AF3 frontier's shape transfers: the zero point is the big
win at low precision, searching the range is worth a few percent, and pulling
outliers out beats clipping them.

**The tower does not amplify.** weights 3.8e-2 -> mixed 5.7e-2 -> pair 4.4e-2 at
int5. Thirty-six residual blocks pass the error through at roughly unit gain,
and the two LayerNorms in the shim take some of it back off. A stack that
amplified would show it here and would settle the question by itself.

**Searching the range stops paying above four bits.** 7% off the pair error at
int4 (9.4e-2 -> 8.8e-2), nothing at all at int5, and it costs a minute a tensor
sweep. Same for outliers: 5.69 bits with two outliers buys exactly what 6.00
bits buys without them, so it is 0.31 bits of saving for a packing format
nothing here can read.

## And what that costs the STRUCTURE

    python3 tools/esmc/fetch_targets.py --out targets --count 16
    .venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py \
        --pdb-dir targets --seed-spread 2 --csv rows.csv

Sixteen single protein chains, 67-205 residues, X-ray, better than 2.0 A,
**released after the checkpoint's September 2021 cutoff** and filtered so no two
are near-duplicates - the RCSB returns deposition groups together and a plain
"newest 20" came back as nine consecutive entries of the same protein, which
would read as nine measurements and be one.

The folding model is loaded once and `forward` takes `lm_hidden_states`
directly, so an arm replaces exactly the tensor the previous table prices and
nothing else - not the folding weights, not the diffusion noise, not the seed.

🔴 **AND THE SAMPLER'S OWN SPREAD IS MEASURED FIRST, BECAUSE NOTHING ELSE HERE
MEANS ANYTHING WITHOUT IT.** The same float32 weights at three seeds, one
diffusion sample each: **mean 0.99 A, worst 7.06 A** over 32 folds. A scheme
that moves a structure by less than that has not moved it.

| scheme | bits/w | bundle | moved: median | mean | worst | targets past 1 A | median vs crystal |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 32.00 | 2840 MiB | - | - | - | - | **2.52 A** |
| **no language model** | - | - | 16.04 A | 15.52 | 23.77 | **16/16** | 17.33 A |
| int8 g64 sym | 8.25 | 732 MiB | 0.04 A | 0.10 | 0.84 | **0/16** | 2.52 A |
| **int5 g32 asym** | 6.00 | **533 MiB** | 0.08 A | 0.43 | 3.32 | 2/16 | **2.53 A** |
| int4 g32 asym | 5.00 | 444 MiB | 0.12 A | 1.27 | 13.17 | 3/16 | 2.53 A |
| int3 g32 asym | 4.00 | 355 MiB | 0.26 A | 1.71 | 15.22 | 4/16 | 2.57 A |

🔴 **THE DAMAGE IS A TAIL, NOT A DEGRADATION.** The median target moves 0.26 A
even at three bits and the median crystal RMSD is FLAT from float32 to int3 -
2.52, 2.52, 2.53, 2.53, 2.57. What actually changes is how many targets get
knocked into a different basin entirely: none at int8, two at int5, three at
int4, four at int3, with worst cases of 13 and 15 A. So a mean is the wrong
summary here and a median alone is too kind; the honest column is **how many
targets moved further than the sampler moves them by itself**.

🔴 **int5 IS FREE ON THAT READING, AND THE SAME SCHEME IS FREE ON AF3.** Its
mean displacement (0.43 A) and its worst (3.32 A) are both inside the sampler's
own 0.99 / 7.06, and its median crystal RMSD is 2.53 against float32's 2.52.
`tools/quantize_af3.py` records int5 group-32 asymmetric costing AF3 nothing
either (0.66 A against float32's 0.69, inside the spread between diffusion
seeds). Two models, two graphs, one packer, the same verdict - and LocalFold
already has the GPU decoder for it (`src/runtime/quantised-upload.js`).

🔴 **int4 IS THE EDGE AND int3 IS OVER IT**, which is again where AF3 lands.
int4's worst case (13.17 A) is nearly twice the worst the sampler produces on
its own, so it breaks targets seeding does not. The floor is about **444 MiB**
for the 600M pair if a rare flipped fold is acceptable and **533 MiB** if it is
not.

🔴 **AND A pair RELATIVE ERROR IS A POOR PREDICTOR OF ANGSTROMS.** int5's 4.4e-2
and int4's 9.4e-2 are a factor of 2.1 apart in the pair representation and a
factor of 3 apart in mean displacement - but 4x apart in the worst case, which
is the number that decides. Use the pair column to choose which schemes are
worth folding; price them on the structure.

## Halving the tower again costs nothing measurable

The other compression axis is the tower itself, and Biohub publish the ablation:
`...-Fast-base300M-step1500k` is the same folding model against ESM-C 300M (30
layers x 960, 333 M parameters). Same sixteen targets, same protocol:

| | median vs crystal | mean | seed-to-seed |
|---|---:|---:|---:|
| ESM-C 600M, float32 | **2.52 A** | 4.88 | 0.99 A |
| ESM-C 300M, float32 | **2.55 A** | 5.31 | 1.10 A |
| ESM-C 300M, int8 | 2.55 A | 5.32 | |
| ESM-C 300M, int5 g32 | 2.57 A | 5.27 | |
| ESM-C 300M, int4 g32 | 2.54 A | 5.26 | |

**0.03 A of median between a tower of 573 M parameters and one of 333 M**, which
is a third of the seed-to-seed spread. On the tails they differ - 600M's mean is
0.4 A better - but on this set the small tower is not the limiting factor.

That puts the **300M pair at int5 at 361 MiB**, against LocalFold's own
`model-af3-int5` at 264.6 MiB and AF2 monomer at 227 MB. It is the smallest
configuration measured that still folds, and it is in the same class as what the
page already ships.

| bundle | float32 | float16 | int8 | **int5** | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M + folding model | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M + folding model | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B + folding model | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB.

## The catch, which is not about compression at all

🔴 **THE COMPRESSIBLE MODEL IS THE WEAK ONE AND THE STRONG ONE IS NOT
COMPRESSIBLE ENOUGH.** The `base300M` and `base600M` checkpoints are paper
ablations - Biohub's own README says "please use ESMFold2 for research work" -
and the RELEASED `ESMFold2-Fast` folds from ESM-C **6B**, which is 4672 MiB at
int5 and out of reach at any precision this study reached. So a browser build
would be shipping an ablation checkpoint, and its accuracy is what the tables
above measure: **median 2.52 A on held-out targets, mean 4.88 A, and two of
sixteen outright failures** (7ILM at 11.9 A, 9TLM at 26.0 A). Compression is not
the obstacle. Whether that accuracy is worth having when no alignment exists is
the actual question, and it is a product question rather than a measurement one.

🔴 **AND THOSE CHECKPOINTS HAVE NO CONFIDENCE HEAD**, so no pLDDT, no PAE, no
pTM - see the model table above. LocalFold colours by pLDDT, its scores card is
built on it, and its archive writer emits it. That is not a small gap to paper
over.

## Chasing 200 MiB, and where the floor actually is

A 200 MiB bundle is an arithmetic statement before it is an experiment:

| bundle | parameters | bits/weight it allows |
|---|---:|---:|
| ESM-C 600M + folding model | 744.5 M | **2.25** |
| ESM-C 300M + folding model | 504.1 M | **3.33** |

🔴 **SO 200 MiB IS OUT OF REACH FOR THE 600M TOWER BEFORE ANYTHING IS
MEASURED.** 2.25 bits a weight is int2 at group 128, the far corner of the
format, and int2 at group 32 - a strictly more generous setting - already reads
pair relRMS 5.2e-1. There is no training run behind that.

### The half of the bundle that had never been priced

🔴 **THE FOLDING MODEL IS 171 M PARAMETERS AND EVERY TABLE ABOVE IGNORED IT.**
It is 41% of the 300M bundle at equal bits, so a size target measured on the
tower alone is a size target for three fifths of the download.
`--fold-bits` quantises it by the same rule `tools/quantize_af3.py` uses - 379
tensors touched, 439 kept float32, 99.2% of the parameters reached. Sixteen
held-out targets, tower held at int5:

| folding model | MiB | median vs crystal |
|---|---:|---:|
| float32 | 653 | 2.55 A |
| int8 g32 | 168 | 2.57 A |
| int5 g32 | 127 | 2.53 A |
| int4 g32 | 107 | 2.54 A |
| **int3 g32** | **86** | **2.58 A** |
| int3 g128 | 71 | **4.59 A** |

🔴 **IT GIVES UP BITS ALMOST FOR FREE AND WILL NOT GIVE UP ITS GROUP.** Three
bits at group 32 costs 0.03 A of median; three bits at group 128 costs **two
angstroms**, and it does so underneath everything else, so a tower ladder run
on top of it measures the folding model and reports the tower. That is how the
first version of this table was nearly written. The reason is the one
`docs/AF3.md` already records for AF3's atom decoder: this model's output is a
POSITION, in angstroms, and nothing downstream renormalises a relative error
in it.

### The tower, with a folding model that is not the problem

Folding model pinned at int3 group 32, ESM-C 300M, same sixteen targets:

| tower | bits/w | tower MiB | moved: mean | TM | median vs crystal |
|---|---:|---:|---:|---:|---:|
| int5 g32 | 6.00 | 238 | 0.53 A | 0.976 | 2.58 A |
| **int3 g32** | 4.00 | 159 | 1.15 A | 0.937 | **2.46 A** |
| **int3 g64** | 3.50 | 139 | 1.44 A | 0.921 | **2.43 A** |
| int2 g32 | 3.00 | 119 | **8.81 A** | **0.484** | **12.15 A** |
| int2 g64 | 2.50 | 99 | **10.04 A** | **0.395** | **12.24 A** |

with `no language model` at 11.99 A / TM 0.299 / median 13.30 A.

🔴 **THE THREE-TO-TWO BIT STEP IS A CLIFF, NOT A SLOPE.** int3 group 64 still
folds - median 2.43 A against the float32 tower's 2.55, inside the sampler's
own spread - and int2 group 32, half a bit later, is **within a whisker of
having no language model at all**: 12.15 A median against 13.30 A for feeding
the trunk nothing. Two bits does not degrade this tower, it deletes it.

### So the floor, by scalar group quantisation, is about 225 MiB

| | tower | folding model | total |
|---|---:|---:|---:|
| comfortable | int3 g32, 159 | int3 g32, 86 | **245 MiB** |
| **tightest that folds** | int3 g64, 139 | int3 g32, 86 | **225 MiB** |
| under 200, and broken | int2 g64, 99 | int3 g32, 86 | 186 MiB |

🔴 **NOTHING IN THE FORMAT REACHES 200 MiB WHILE STILL FOLDING.** The only
scalar combinations that fit put two bits on the tower, and two bits is the
cliff. Getting under 200 needs a different representation - QuIP#-style
incoherence processing with vector codebooks is the class that makes 2-bit
language models work - and that is a new WebGPU decoder, not a new packer.

## What the AF3 port learned about the language model, and what it means here

`../alphafold3` now folds all six ESMFold2 releases against native, and getting
there turned up four things this directory has to know.

🔴 **THE SHIM IS PER MODEL, NOT PER FAMILY.** Every release trains its own
`language_model.*` - the module that turns ESM-C's 37 hidden states into a pair
representation. They share the TOWER and nothing else. The AF3 port shipped one
`esmfold2.lm.npz` for the whole family and fed every variant the BASE model's,
which against native's own `lm_z` for that variant reads **corr 0.026** where the
variant's own shim reads **0.999998**. It cost `esmfold2_exp_fast` **8.798 A
against 0.812** on 6MRR.

🔴 **THIS DIRECTORY AVOIDS IT BY CONSTRUCTION, AND THAT IS WORTH KNOWING BEFORE
SOMEBODY OPTIMISES IT.** `Shim.__init__` reads `language_model.*` out of the
folding checkpoint it is handed, so the tower and the shim cannot come from
different models without the caller naming two directories. Verified against
the `esm` package's own module on the 600M experimental-fast checkpoint:

| | |
|---|---|
| `Shim.pair` against native `model.language_model` | **relRMS 3.20e-7** |
| correlation | 1.000000 |
| standard deviation | 3.354 against 3.354 |

The failure mode returns the moment anyone caches a precomputed `lm_pair` or an
`lm.npz` beside the tower - which is exactly the shape of optimisation this
repository likes, and exactly what made the mistake easy upstream.

🔴 **AND BETWEEN 600M AND 300M IT WOULD RAISE, WHICH IS NOT REASSURANCE.** Their
shims differ in SHAPE - 37 mix entries against 31, a (256, 1152) projection
against (256, 960) - so crossing those two is a loud error. Upstream's case was
two releases of the SAME width, where it is silent and folds anyway. Any check
here has to discriminate on the weights, not on the shapes.

🔴 **AND THE EXPERIMENTAL LINE IS A DIFFERENT IMPLEMENTATION, NOT A DIFFERENT
CONFIG.** `ESMFold2ExperimentalModel` is its own class. Three divergences the
AF3 port found by reading it rather than inferring from `config.json`, all of
which a LocalFold port would have to carry:

* **no `lm_encoder`**: the shim's output is added straight to `z_init`, ONCE,
  outside the recycle loop. Reading the released line's early-return here meant
  the language model never reached the trunk at all - and it still folded, one
  variant at 1.694 A, with ESM-C changing the answer by nothing to three
  decimals.
* **`lm_dropout` is 0.0**, not the released line's 0.25, which lives in
  `lm_encoder.per_loop_lm_dropout` and therefore does not exist here.
* the MSA encoder runs AFTER the recycle and is ADDED - moot for "fast", which
  has no MSA encoder at all.

🔴 **AND ALL SIX VARIANTS THE AF3 PORT WIRES IN USE ESM-C 6B.** Its registry
says so in as many words ("all six share ESM-C 6B") and there is no reference to
a 600M tower anywhere in that tree. The checkpoints this document compresses -
`...-base600M-step1500k` and `...-base300M-step1500k` - are the paper's LM-size
ablation series and are not in it. So the model that is validated end to end is
not the model that fits a browser, and the model that fits a browser is not
validated end to end. See the note on the confidence head above.

## The 600M bundle, all the way down

Everything above measures the tower with a float32 folding model, which prices
about two thirds of a download. This is the whole bundle: ESM-C 600M plus the
folding model at int3 group 32, sixteen held-out targets, against a seed
yardstick of **2.43 A mean and 11.00 A worst** over 32 folds of identical
weights.

| bundle | tower scheme | bits/w | median vs crystal | mean | past the seed noise |
|---:|---|---:|---:|---:|---:|
| - | float32 | 32.00 | **2.13 A** | 5.17 | - |
| 496 MiB | int5 g32 | 6.00 | 2.12 A | 5.22 | 2/16 |
| 428 MiB | int4 g32 | 5.00 | 2.19 A | 5.12 | 2/16 |
| 360 MiB | int3 g32 | 4.00 | 2.39 A | 5.21 | 2/16 |
| 325 MiB | int3 g64 | 3.50 | **2.13 A** | 5.19 | 2/16 |
| 308 MiB | int3 g128 | 3.25 | **2.13 A** | 5.19 | 2/16 |
| **308 MiB** | **codebook 4096 x 4d** | 3.25 | 2.32 A | 5.33 | **1/16** |
| **274 MiB** | **codebook 1024 x 4d** | 2.75 | 2.37 A | 5.26 | **1/16** |
| 291 MiB | int2 g32 | 3.00 | 6.16 A | 8.52 | 10/16 |
| 257 MiB | int2 g64 | 2.50 | 11.37 A | - | - |
| 240 MiB | codebook 256 x 4d | 2.25 | 7.13 A | 11.71 | 9/16 |
| - | *no language model* | - | 14.03 A | 17.12 | 16/16 |

🔴 **THE FLOOR IS 274 MiB, AND IT WAS 533 WHEN THIS DOCUMENT STARTED.** Three
separate things moved it, none of them a better packer: quantising the FOLDING
MODEL, which nobody had priced and which gives up bits almost for free; opening
the tower's GROUP from 32 to 128, which is a quarter of a bit at no measurable
cost; and replacing the uniform grid with a codebook.

🔴 **AND THE TWO-BIT CLIFF IS A PROPERTY OF THE GRID, NOT OF THE BUDGET.**
Scalar int2 group 32 spends **3.00** bits a weight and folds at 6.16 A median;
a 1024-entry codebook spends **2.75** and folds at 2.37 A. A uniform grid puts
its levels where there are no weights, and at two bits there are too few levels
to waste any. That is the whole of the difference - same information budget,
one representation folds and the other does not.

🔴 **AND AT MATCHED RATE THE CODEBOOK IS THE MORE ROBUST ONE, WHICH THE MEDIAN
HIDES.** At 3.25 bits, scalar int3 g128 and the 4096-entry codebook read 2.13 A
and 2.32 A of median crystal RMSD - the scalar arm looks better. On the tail it
is the other way round: the codebook moves **1 of 16** targets past the
sampler's own spread against the scalar arm's 2, and **none at all** past 5 A
against the scalar arm's 1, with a worst case of 3.43 A against 9.06 A. The
median is where these schemes agree; the tail is where they differ, and the
tail is what a user notices.

### What the codebook costs to decode, which is less than what it replaces

🔴 **A TABLE LOOKUP IS CHEAPER THAN UNPACKING FIVE BITS.** `fit-codebook.py`
writes ONE shared table for the whole tower - 1024 entries of 4 float16 is
**8 KB** - so decoding a weight is an index into it and a multiply by the
group's scale, against int5's shift-mask-across-a-byte-boundary. LocalFold
already expands quantised weights into a dense float16 buffer in one dispatch
(`src/runtime/quantised-upload.js`); this is that same dispatch with a simpler
body. It is not the same shader, but it is not a harder one.

🔴 **AND THE ROTATION - QuIP#'s OTHER HALF - IS NOT WORTH IT HERE.** Multiplying
by a random orthogonal matrix before quantising is what makes 2-bit language
models work in the literature, and it is the expensive half: the kernel has to
un-rotate at run time. On ESM-C's weights it buys **1%** (relRMS 0.2090 against
0.2106 at 2.75 bits) and 5.6% on the scalar path. Measured before building it,
which is the only reason it was not built.

🔴 **AND AN EMPTY CENTRE IS A WASTED CODE.** Lloyd's algorithm strands centres
that nothing selects, so a 1024-entry table silently becomes a 900-entry table
at the same price. Restarting a dead centre on the vectors currently worst
served is four lines and it is why all three tables report every entry used.

🔴 **AND THE GROUP SCALE IS AN RMS, NOT A MAXIMUM.** A shared table only works
if every group hands it the same distribution, and dividing by the group's
largest weight standardises the OUTLIER rather than the bulk. There is also no
zero point: a scalar scheme needs one because its grid is symmetric and its
weights are not, and a codebook's entries are already wherever k-means put
them. That is half a bit per group not spent.

### And what sparsity is worth, which is nothing here

Zeroing weights reaches rates dense quantisation cannot - 1:8 with int4
survivors is 1.00 bits a weight - but it is dominated everywhere the model
still folds:

| scheme | bits/w | reconstruction relRMS |
|---|---:|---:|
| 2:8 int4 | 1.85 | 0.5442 |
| scalar int2 g32 | 3.00 | 0.3947 |
| 2:4 int4 | 3.15 | 0.3468 |
| **scalar int3 g64** | **3.50** | **0.2093** |
| 2:4 int8 | 5.15 | 0.3383 |
| **scalar int4 g32** | **5.00** | **0.0845** |

🔴 **BECAUSE THE DISCARDED ENERGY IS A FLOOR NO PRECISION RECOVERS.** The top 4
of every 8 weights carry 96.5% of the energy, so 2:4 throws away 3.5% and
cannot beat relRMS **0.187** at any survivor precision - and scalar int3 g64
*achieves* 0.209 at 3.50 bits. The same table shows it directly: 2:4 with int4
survivors reads 0.3468 and with int8 survivors 0.3383, so **two extra bits a
weight buy 2%**. The error is what was zeroed, not how the rest is stored.

🔴 **AND THE MASK IS NOT FREE.** "90% of the weights are zero" says nothing
until the reader is told WHICH. An arbitrary mask costs the binary entropy of
the density - 0.54 bits a weight at one in eight, more than the surviving
values themselves. n:m fixes the count per block so the mask is log2(C(m,n))/m
and needs no search, which is the only version worth quoting.

## Calibrated quantisation, which is what the LLM world does instead

Everything above rounds each weight to the nearest code and looks at nothing
else. The methods that made 3- and 4-bit language models usable all ask a
different question - *what codes keep this LAYER'S OUTPUT the same on real
data* - and they are cheap enough to be worth trying here. `tools/esmc/gptq.py`
implements two, `tools/esmc/calibrate-esmc.py` drives them, and the calibration
set is 288 UniRef50 sequences (three length buckets, 96 each).

| | what it does | cost |
|---|---|---|
| **RTN** | nearest code, per-group affine range | free |
| **imatrix** | keeps RTN's codes, picks each group's RANGE to minimise error weighted by that channel's activation energy. llama.cpp's importance matrix | one forward pass |
| **GPTQ** | quantise column by column, pushing each column's rounding error into the columns not yet done, along the inverse Hessian of `2 XᵀX` | one forward pass + a Cholesky per matmul |
| AWQ | search a per-input-channel scale, fold it into the neighbouring op | comparable |
| SpQR / SqueezeLLM | keep the few extreme weights per group in float16 | measured above: 0.31 bits |
| QuIP# / AQLM | random rotations to kill outliers, then vector codebooks | **needs a different decoder** |
| SmoothQuant, LLM.int8() | move activation outliers into the weights | **irrelevant here** - LocalFold quantises STORAGE and computes in f16/f32, so there are no activation outliers to migrate |

🔴 **THE BLOCKS ARE CALIBRATED IN ORDER AND SO ARE THE FOUR MATMULS INSIDE ONE**,
so every layer corrects for the error its predecessors actually made rather
than for an error nothing will make. Five passes per block instead of one, and
the whole 573M tower takes **70 seconds on an A100**.

🔴 **AND NEITHER METHOD CHANGES THE STORAGE FORMAT, WHICH IS WHY THESE TWO AND
NOT THE OTHERS.** Both emit exactly what `src/runtime/quantised-upload.js`
already decodes: asymmetric codes, one float16 scale and one float16 zero per
group of 32. GPTQ's group axis lines up for free - LocalFold groups 32
CONSECUTIVE elements of a row-major `(out, in)` tensor, which is 32 consecutive
input channels of one output channel, which is what GPTQ calls `group_size`.
`act-order` is deliberately not implemented: it is worth a few tenths of a bit
and it permutes the input axis, so the groups stop being consecutive and the
shader would need the permutation.

### What they buy

Same sixteen held-out targets, same protocol, same seed yardstick (1.00 A mean,
7.16 A worst between two seeds of identical weights):

| arm | bits/w | tower | moved: mean | TM | median | worst | past 1 A |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GPTQ int5** | 6.00 | 410 MiB | **0.39 A** | **0.979** | 0.06 | 3.07 | 2/16 |
| RTN int5 | 6.00 | 410 MiB | 0.44 A | 0.977 | 0.08 | 3.29 | 2/16 |
| **GPTQ int4** | 5.00 | 342 MiB | **1.14 A** | **0.947** | 0.10 | 13.33 | 3/16 |
| imatrix int4 | 5.00 | 342 MiB | 1.16 A | 0.940 | 0.08 | **9.25** | 3/16 |
| RTN int4 | 5.00 | 342 MiB | 1.29 A | 0.933 | 0.12 | 13.26 | 3/16 |
| **imatrix int3** | 4.00 | 273 MiB | **1.29 A** | **0.929** | 0.20 | 11.20 | 3/16 |
| RTN int3 | 4.00 | 273 MiB | 1.72 A | 0.920 | 0.27 | 15.09 | 4/16 |
| GPTQ int3 | 4.00 | 273 MiB | 1.87 A | 0.904 | 0.60 | 14.03 | **7/16** |

🔴 **GPTQ MAKES THE WEIGHTS WORSE AND THE OUTPUT BETTER, WHICH IS THE WHOLE
IDEA.** At int4 it moves the weights 9.78e-2 from float32 where plain rounding
moves them 7.91e-2 - and the pair representation it produces is 7.87e-2 against
rounding's 9.42e-2. A weight-space study would have reported GPTQ as the worse
method. `tools/analyse_quantisation.py` is a weight-space study.

🔴 **CALIBRATION IS WORTH ABOUT ONE BIT, AND ONLY AT THE BOTTOM.** `imatrix`
int3 at **4.00 bits** matches plain int4 at **5.00 bits** on every column - 1.29
A mean, TM 0.929 against 0.933 - which is a whole bit for one forward pass. At
int5 there is nothing left to win: GPTQ's 0.39 A against rounding's 0.44 A is
inside the sampler's own 1.00 A, so both are the same answer.

🔴 **AND IT DOES NOT CLOSE THE int4 -> int5 GAP.** GPTQ int4 is 1.14 A where
plain int5 is 0.44 A. Calibration recovers about 15% of a bit-step, not a whole
one, so **int5 remains the smallest scheme that is free** and 533 MiB (600M) /
361 MiB (300M) stands as the answer.

🔴 **AND GPTQ IS THE WORST ARM AT THREE BITS, WHICH THE PAIR METRIC DOES NOT
SAY.** Its pair error (1.77e-1) beats plain rounding's (1.92e-1) and its folds
are worse on every structural column - mean 1.87 A against 1.72, TM 0.904
against 0.920, and **seven of sixteen targets past the seed noise against four**.
Pushing a large rounding error down the Hessian spreads it over columns that
were fine; at three bits there is more error than there is room to put it. The
two metrics disagree in SIGN here, which is the strongest argument in this file
for folding the structure rather than trusting a tensor norm.

### Would quantisation-aware TRAINING go further

Probably, by about another bit, and it is not obviously worth it.

The literature's ordering is RTN < imatrix/AWQ < GPTQ < QAT, and the step from
GPTQ to QAT is worth roughly what the step from RTN to GPTQ is - which here was
one bit at the bottom and nothing at the top. The right objective would not be
the masked-LM loss ESM-C was trained on: ESMFold2 never reads the logits. It
would be **self-distillation against the float32 tower's own hidden states** -
no labels, just UniRef50 sequences and the teacher already on disk - because
those are literally the tensors the shim mixes. Straight-through estimator on
the codes, the scales and zeros left as they are so the format does not move.

Cost on this A100: teacher forward plus student forward and backward, about 3x
a plain pass, so ~1-2 days for a few thousand steps at 573M parameters. The
prize is the 600M bundle at 355 MiB instead of 533, or the **300M bundle at
about 240 MiB - which is AF2 monomer's 227 MB**.

🔴 **BUT THE BINDING CONSTRAINT IS NOT SIZE.** Both towers already fit in the
same class as what the page ships, the compute is a tenth of an AF3 trunk pass,
and the checkpoints that fit are paper ablations with **no confidence head at
all** and a median 2.52 A on held-out targets with two failures in sixteen.
Spending two days of GPU to move 533 MiB to 355 MiB does not change any of
that. The experiment to run before any training run is whether ESMFold2-Fast is
accurate enough to be worth shipping at ALL, and that is a question about the
model rather than about its bytes.

## Spending bits where the layer mix is heavy does NOT work

ESMFold2 takes **58.8%** of its softmax from the last three of the 37 states and
**0.017%** from states 12-17 between them. That looks like an obvious place to
allocate precision, and it is not, at any budget tried:

| arm | bits/w | pair | the uniform scheme at that budget |
|---|---:|---:|---|
| int4 everywhere, int6 on the last 6 blocks | 5.33 | 7.4e-2 | ~6.4e-2 interpolated |
| int4 everywhere, int8 on the last 3 blocks | 5.33 | 8.5e-2 | ~6.4e-2 |
| int3 everywhere, int6 on the last 6 blocks | 4.50 | 1.4e-1 | 1.0e-1 (int4 g64) |

🔴 **BECAUSE THE MIX SAYS WHERE A STATE IS *READ*, NOT WHERE PRECISION
MATTERS.** Every block feeds every later one, so a cheap early block damages the
expensive late states too - and it damages them through 30-odd more blocks of
residual stream. Uniform allocation beats every graded one measured, by 15-40%,
which is a large enough margin not to be a tuning question.

## The forward pass is not ours alone

`tools/esmc/esmc_forward.py` is written from the block layout
`../alphafold3/converters/esmc.py` documents, deliberately not by importing the
`esm` package, so that a quantisation arm and its reference do not share a
forward. Against `transformers`' own `EsmcForMaskedLM`, on ubiquitin:

| | relRMS |
|---|---|
| hidden states 0-3 | 0.0, 1.7e-7, 2.2e-7, 2.5e-7 |
| hidden state 18 | 5.2e-7 |
| hidden states 34, 35, 36 | 2.1e-6, 2.2e-6, 2.1e-6 |
| masked-LM logits | 1.0e-6 |

Float32 accumulation order, and it pins every convention that could have been
wrong invisibly: RoPE's split-halves rotation, the QK-LayerNorm being over the
FULL `d_model` rather than per head, the fused projection's `[q|k|v]` order, the
SwiGLU's `[gate|up]` order, the ESM3 residual scale, and the final LayerNorm
being applied to the last state only.

🔴 **`transformers` CANNOT LOAD THESE CHECKPOINTS DIRECTLY.** Its ESMC expects
its own converted layout (`esmc.layers.N.self_attn.q_proj.weight`) while Biohub
publish the fused TransformerEngine one
(`esmc.transformer.blocks.N.attn.layernorm_qkv.weight`), so
`from_pretrained` loads **nothing**, reports every parameter MISSING, silently
builds an 80-layer model from a default config and answers with random weights.
The check above builds the state dict by hand. A first attempt that did not
would have been comparing against noise.

🔴 **AND THE FIRST SANITY CHECK WAS A MASKED-LM ONE, BEFORE ANY ORACLE.** Mask
every fifth residue of ubiquitin and see what the head puts back: **86.7%**
against a chance rate of about 5%. That is cheap, needs no reference, and no
wrong RoPE convention survives it.

## The shim, and why the 37 states never have to exist at once

ESMFold2's `LanguageModelEncoder` is: LayerNorm the states, project 1152 -> 256,
mix them by `softmax(base_z_combine)`, downproject, outer product carrying BOTH
a product and a difference, two-layer MLP, LayerNorm.

🔴 **THE MIX IS A CONSTANT AND THE NORM AND PROJECTION ARE SHARED ACROSS
LAYERS**, so `sum_k combine[k] * LN(h_k) @ W` is a running accumulator of
`(tokens, 256)` and the `(37, tokens, 1152)` tensor never needs to be
materialised - 4.5x smaller at any length, and the tower can be streamed one
block at a time. `Shim.accumulate` is that form and it agrees with
`Shim.single` to **1.1e-7**, which is what says it is the same arithmetic.
The downprojection stays outside the sum because it is affine and its bias must
not be added thirty-seven times.

That is the fact that decides device memory. A block is 15.9M parameters, so at
float16 the tower's resident footprint on the GPU is **32 MB one block at a
time**, not 1.15 GB - the same shape of trade `releaseResidentWeights` makes for
AF3's stages.

## Would it run

Arithmetic, not measurement, but the terms are not close:

| | |
|---|---|
| ESM-C 600M, matmuls | **1.15 GFLOP per token** |
| at 300 residues | 359 GFLOP, ~0.3 s at this M2's measured 1.0-1.5 TFLOP/s |
| weight traffic, one pass | 1147 MB at float16, **358 MB read as packed int5** |
| resident, streamed | one block, 32 MB at float16 |

For scale, `bench-trunk.js` measures AF3's own trunk at **3372 ms at 200
tokens**. The language model is a fraction of that, and ESMFold2's trunk is 24
PAIR-ONLY blocks - no triangle attention, no single track - against AF3's 48
pairformer blocks. The compute is not the problem. **The download is the whole
question**, which is why this document is mostly a quantisation table.

| bundle, tower + folding model | float32 | float16 | int8 | int5 | int4 |
|---|---:|---:|---:|---:|---:|
| ESM-C 300M | 1923 | 961 | 496 | **361** | 300 |
| ESM-C 600M | 2840 | 1420 | 732 | **533** | 444 |
| ESM-C 6B | 24915 | 12457 | 6423 | 4672 | 3893 |

MiB. For comparison LocalFold ships `model-af3-int5` at **264.6 MiB** and AF2
monomer at 227 MB. So the 600M pair at int5 is about twice the AF3 bundle and
the 6B is out of reach by an order of magnitude at any precision - which is what
makes the 600M line the interesting one and the reason the question was asked
about it.

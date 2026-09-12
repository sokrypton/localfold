# AlphaFold 3 in LocalFold

Where the AF3 port stands, what it costs, and the things that have already been
got wrong once. Written to be read before touching any of it.

`AGENTS.md` holds the invariants; this holds the state.

## What works

A protein chain typed into `index.html` folds with AlphaFold 3 entirely in the
browser: featurisation, trunk, diffusion and confidence, no server. Pick **AF3**
in the Model dropdown.

- **From a sequence, not a dump.** `src/af3/featurise.js` builds AF3's whole
  batch in JavaScript. Checked array-by-array against AF3's own batch for 6MRR
  and for a three-chain complex: `node tools/oracle/check_af3_featurise.js`.
- **Complexes**, chains separated by `:`. Chain identity comes from
  `src/input/chains.js` - the same `chainIdentity()` AlphaFold-multimer uses.
- **Two samplers.** *Flow* (default) draws once at the top of the schedule and
  walks it down deterministically, ~8 calls. *Diffusion* is AF3's own stochastic
  sampler, 20+ steps. Both are seeded.
- **DNA and RNA chains**, as their own entity types. A standard nucleotide is
  ONE TOKEN PER RESIDUE, so this needed no tokeniser change - only teaching the
  featuriser that a chain has a KIND, because `ACGT` is a valid protein as well
  as a valid DNA chain and nothing about the letters says which. Checked
  array-by-array against AF3 for protein+DNA, protein+RNA and a three-chain
  complex; folded geometry checked by `tools/gpu/probe-nucleic.js` (bond ratio
  1.013 DNA, 1.009 RNA, against 1.017 for the protein control). Their reference
  conformers are `src/af3/reference-conformers-nucleic.js`, generated from the
  oracle rather than typed. No MSA: AF3 searches an RNA database this page has
  no server for, and DNA gets none in AF3 either.
- **Modified residues** on protein chains; modified BASES are refused, since the
  modified-residue path resolves parents through the amino-acid table.
- **Recycles** for AF3 as well as AF2.
- **MSAs**, through the page's own alignment controls - search, paste or upload,
  shared with both AlphaFold 2 models. `src/af3/msa-features.js` is the whole of
  the adapter. On the 59-residue demo sequence a 512-row alignment moves pLDDT
  55.8 -> 65.7 and costs about 2 s (the MSA stack goes from nothing to 239 ms at
  512 rows).
- The trajectory animates as it computes, and the finished structure gets
  py2Dmol's PAE panel and prediction-quality card.

### Accuracy, against crystal structures

| | RMSD | TM | notes |
|---|---|---|---|
| 6MRR, flow 8 | 0.69-0.75 | 0.949 | four seeds, sigma0 160 |
| 6MRR, flow 8, sigma0 2560 | 0.65-0.77 | 0.950 | four seeds, AF3's schedule |
| 6MRR, diffusion 200 | 0.66 | 0.953 | |
| 1QYS (Top7), flow 8 | 0.89-0.92 | 0.944 | four seeds, sigma0 160 |
| 1QYS, flow 8, sigma0 2560 | 0.85-0.88 | 0.948 | four seeds, AF3's schedule |
| 1QYS, diffusion 200 | 0.93-1.12 | 0.92-0.94 | across four seeds |

🔴 **1QYS DOES NOT REPRODUCE ITS ROW AND HAS NOT FOR SOME TIME.** Measured
2026-09-02 at `--mode=flow --steps=8 --recycles=1`, seeds 1, 2 and 3: **1.24,
1.27 and 1.25 A**, TM 0.899-0.903, against the 0.89-0.92 above. Three recycles
gives 1.23 A and pLDDT 81.2 rather than 69.8, so it is not a recycle count. It
is not a regression from the kernel work either - the same input on the tree as
of `ea063a0`, before any of it, gives 1.24 A and pLDDT 69.8 to the digit. 6MRR
still measures inside its row (0.64 A, TM 0.960). The likeliest explanation is
that this table predates the side-chain fix, which changed what the denoiser
produces; it has not been re-run.

🔴 THE FLOW STARTS AT 160 A NOW, WHICH COSTS 1QYS 0.04 A. Most of AF3's
schedule sits above the level where the denoiser begins trusting the
coordinates it is handed, so a walk from 2560 spends its first calls on a
regime a flow does not need - and a ligand pays for it, HEM's bond error at
eight steps being 0.218 A from 2560 against 0.129 A from 160. On the proteins
6MRR is unchanged and 1QYS loses 0.04 A with non-overlapping seed ranges. That
trade was made deliberately; `schedule: {sigmaMax: 160}` restores AF3's own.
AF3's DIFFUSION sampler is untouched. See tools/gpu/probe-sigma0.js.

Flow matches or beats the 200-step sampler with ~25x fewer denoiser calls, on
both proteins measured. It is two proteins, both small designed alpha/beta
folds, both single-sequence - an observation, not a result.

🔴 AND BOTH ARE 68 AND 92 RESIDUES, WHICH IS WHY THEY KEPT PASSING. Every AF3
number in this file was a small single-sequence protein until 3RPF; the fold
that dropped a 512-row alignment on the floor scored the same on these two,
because they never had an alignment to drop. With an MSA, 3RPF's 146-residue
chain reaches 1.10 A and its complex 0.32 A - see the section on the AlphaFold
Server below. A regression suite of two proteins under 100 residues, both
folded single-sequence, is not one.

 ### Speed, 68 tokens

🔴 **THE FIRST FOLD OF A SESSION IS NOT THE FOLD, AND THIS FILE ONLY EVER
QUOTED THE FIRST.** Pipelines and the resident f16 weights are cached for the
life of the DEVICE, so the page pays for them once and every fold after is
cheaper. `tools/gpu/fold.js --folds=3` runs three in one process:

| | cold | warm | device |
|---|---|---|---|
| every f16 path off | 3.23 s | 2.18, 2.17 | 1463 MiB |
| as shipped | 2.48 s | 1.97, 1.90 | 855 MiB |

So a flow-8 fold is **1.9 s** once a session is going, and 2.5 the first time.
The f16 work is worth 23% cold and 11% warm - bigger cold because the f32 arm's
larger weight UPLOAD costs more than the f16 arm's one-off conversion.

It was 3.2-3.4 s and 1406 MiB before the f16 work of 2026-09-04, ~150 s when the first end-to-end fold ran, and 7 s before
the kernel work of 2026-09-02. A diffusion-200 fold was 25.9 s at the 3.0 s
era.

Where it goes: two trunk passes, then eight denoiser calls of which the FIRST is
~550 ms and the rest ~85. It is the largest single item left, and it is not
compilation, which measures 6 ms.

🔴 **AND IT IS NOT THE f16 CONVERSION EITHER, WHICH IS WHAT THIS FILE USED TO
SAY.** `tools/gpu/probe-pack.js` splits it over all 24 blocks of the int5
manifest, which pack to 378.2 MiB:

| cold, f16 (decode + convert + concatenate) | 494 ms |
| warm, f16 (convert + concatenate) | 244 |
| warm, f32 (concatenate alone) | 66 |

So the int5 DECODE is **250 ms**, the f16 conversion 178, and the concatenation
66. The decode is the store binding a block lazily - the first read of each of
its ~40 tensors decodes that tensor out of the shard - and it is paid whatever
precision the buffer ends up in. Dropping to f32 weights would save 178 ms of
494 and cost 378 MiB more on the device.

🔴 **AND IT IS A COMPUTE PASS NOW, NOT HOST WORK AT ALL.** Hiding it behind the
trunk was considered and would have cost 378 MiB; decoding it on the GPU costs
nothing and is 3.7x faster. `src/runtime/quantised-upload.js` uploads the int5
CODES - about an eighth of the bytes - and decodes them straight into the
resident buffer. Over the 24 blocks, both arms cold: **437 ms on the host
against 119 on the device**, and 0 of 198 million elements differ.

The same helper (`src/af3/device-weights.js`) took the trunk's two f16 labels,
which were the next largest:

| packer | MiB | host, cold |
|---|---|---|
| the transformer's 24 blocks | 378.2 | 440 ms |
| the trunk's single transition | 162.1 | 200 |
| its single attention | 67.6 | 77 |
| its pair transition (f32) | 36.0 | 16 |
| its outgoing triangle (f32) | 18.2 | 15 |
| its first grid attention (f32) | 15.1 | 19 |

What a real page fold does, through the dev panel's own timeline:

| | before | + the transformer | + the trunk |
|---|---|---|---|
| Trunk 1/2 | 673 ms | 647 | **355** |
| Folding | 920 | 311 | **257** |
| the whole fold | 3.31 s | 2.62 | **2.30** |

🔴 **THE f32 PACKERS ARE NOT WORTH CONVERTING, AND NEITHER WAS THE LAST f16
ONE, ON TIME ALONE.** Moving the single ATTENTION changed the fold by 2.31 s
against 2.30 - nothing - because once the single TRANSITION was off the main
thread the rest of the trunk's packing fits inside the pairformer's own GPU
time. It is kept for the 77 ms of main-thread work, which matters on a slower
CPU, not for the clock here. The f32 ones total about 50 ms, already hidden,
and would need an f32 variant of the kernel.

🔴 **AND THE RELEASE OF THE STAGING BUFFERS MUST NOT BE AWAITED.** The first
version awaited `onSubmittedWorkDone` per block, which put 48 host-device
synchronisations inside the pairformer's block loop - a loop written to run
ahead of the device on purpose. Trunk 1/2 went 647 ms to **778**. It rides an
unawaited promise now, which is the idiom that loop already uses for its
progress callback. Inside a trunk pass the order is now pair-transition
86 ms, grid.project 58, tri.project 48, grid.attend 47, tri.project-out 39, at
118 tokens over two passes.

🔴 **THE 12 ms THIS FILE ATTRIBUTED TO THE SAMPLER'S PER-STEP WORK WAS NOT
THERE.** It said a denoiser call cost 123 ms inside the sampler against 111 on
tools/gpu/bench-head.js, blamed the gap on the random augmentation, the noise
injection, the Euler step and the two trajectory copies, and called it 2.4 s of
a 200-step fold that nobody had looked at. Someone has now:
`tools/gpu/probe-sampler-overhead.js` times each phase inside the loop.

    59 tokens    step 112.8 ms   head.run 112.2   everything else 0.6
    150 tokens   step 250.5 ms   head.run 249.0   everything else 1.5

So the loop costs **0.6%**, not 10%, and the host arithmetic is 0.4 ms of it -
nearly all the noise injection's 4,248 gaussian draws. The augmentation, the
Euler step and both copies measure zero. They were always going to: at 59
tokens the whole of that work is about fifteen thousand float operations, which
is microseconds, and the count was checkable without running anything.

The 12 ms was two numbers from two processes - the drift this file warns about
three times over. `head.run` itself is within 0.2 ms of the GPU time it
reports, so there is no round trip to win back either. **Do not build a GPU
sampler step; there is nothing under it.**

## Modified residues, and why they need sixteen steps

Supported as of 2026-09-02, verified against AF3 array by array
(`check_af3_featurise.js` with a `--modification` dump) and structurally
(`tools/gpu/probe-modified.js`).

A modified residue is **one token per heavy atom**, inside the chain: SEP at
position 3 of a twelve-residue chain is 21 tokens, the ten belonging to it each
carrying one atom, all holding the PARENT residue's aatype (serine), all
sharing that residue's index, and all keeping the chain's asym, entity and sym.
Its own bonds go through the ligand-bond machinery; its peptide bonds to its
neighbours stay implicit in `residue_index`, as a standard chain's do. The
dictionary describes a FREE amino acid, so `polymerResidue` drops the OXT it
loses on forming a peptide bond and puts it back at a C-terminus.

🔴 **MSE IS NOT ONE OF THESE.** AF3 folds selenomethionine into methionine's
alphabet slot and leaves it one token with its own chemistry - SD becomes SE.
Every other modification tried is atom-tokenised. The page refuses MSE with
that reason rather than offering it and doing something else.

🔴 **AN ATOM-TOKENISED RESIDUE IS PLACED LESS PRECISELY THAN A STANDARD ONE,
AND THAT IS AF3'S BEHAVIOUR.** Its atoms are each their own token rather than
coming from a shared residue conformer, so the sampler places them
individually. Folding `ACSEFGHIKLWY` with SEP at 3, as the median
predicted-to-ideal bond ratio:

| | control | modified |
|---|---|---|
| AF3 itself, 32 diffusion steps | 1.003 | **0.956** |
| this port, 32 diffusion steps | 1.005 | **0.953** |

We match AF3 to three thousandths on both, so the gap between 0.95 and 1.00 is
the architecture's price and not a porting bug. What IS ours to get right is the
step count: at eight flow steps the residue comes out visibly compressed while
its neighbours are fine.

    flow-8    0.835   (control 1.003)
    flow-16   0.974   (control 1.007)
    flow-32   0.996   (control 1.010)

So sixteen is the lowest the dial offers, in both modes. It looked exactly like
the side-chain bug below - everything short, worse with distance from the
backbone, P-O3P at 0.483 - and the two are told apart by the fact that this one
improves with more steps and that one did not.

## Running it

    python3 -m http.server 8080          # then open /index.html

    # The GPU lane. Dawn (`npm run test:gpu`) cannot load on this macOS; Chrome
    # can. Every checker and bench is a module for this harness.
    node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]

    node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
      --mode=flow --steps=8 --recycles=1 --model=/model-af3-int5/manifest.json
    python3 tools/score_fold.py <the log> --reference tools/fixtures/6mrr-crystal.pdb

    # The featuriser, including the MSA path, against AF3's own batch.
    python3 tools/oracle/dump_af3_trunk.py --blocks 0 --a3m rows.a3m --out d.json
    node tools/oracle/check_af3_featurise.js d.json rows.a3m

    node tools/gpu-chrome.mjs tools/gpu/bench-blocks.js   # AF2 vs AF3 per block
    node tools/gpu-chrome.mjs tools/gpu/bench-ab.js --skip=single

Checkers, all differential against the CPU reference:
`check-af3-{triangle,transition,grid-attention,single-attention,opm,msa-attention,embedder,template,block,msa-block,trunk,confidence,diffusion-*,atom-*,sampler-gpu,target-feat-gpu}.js`

## Traps

Each of these cost real time. They are in the code as `🔴` comments too.

**Benchmarks drift up to 3.2x between processes.** Two numbers from two
invocations of the same bench cannot be compared. `tools/gpu/bench-ab.js`
alternates A and B inside one process and reports medians; within a process the
spread is about +-10 ms. A whole round of per-pass profiling was thrown away
after this was ignored - the tell was "everything skipped" measuring *slower*
than "full".

**The parts do not sum to the whole.** Skipping one pass measures what it costs
on top of everything else pipelining, not its share. Individually the pair
track's passes account for 79 ms; together they cost 348.

**pLDDT is not the check.** It comes off the trunk and can look healthy over
coordinates that are not a molecule. A batch with one broken gather folded a
17 A spaghetti at pLDDT 55 with 15 A between consecutive CA. Backbone CA-CA is
what a wrong sampler cannot fake, which is why the fold prints it.

**A gather's `count` is not decoration.** `convert()` sizes its output from it,
so a gather without one silently yields a zero-length tensor and the model runs
anyway.

**`sigma` is a claim about the input, not a dial.** It reaches the network
through the Fourier noise embedding. Feeding a black hole at sigma 4 - "this
structure is nearly right" - diverges; at sigma 2560 - "this is noise, ignore
it" - the same input gives 1.39 A.

**The diffusion head has its own five reference embeddings**, distinct from the
conditioning module's. Same shapes, different weights. Reusing one for both
type-checks.

**Four conditioning weights exist twice in the checkpoint**, `..._1` and
unsuffixed, identical shapes. Dropping the suffix loads clean and gives the
wrong `target_feat`.

**AF3's MSA gap is 21, not 31.** The alphabet is 21 protein codes, then the
gap, then the nucleotides - the gap is in the MIDDLE of the 32-wide one-hot. A
gap at 31 type-checks, folds, and tells the model every gap is an unknown
nucleotide. Related and just as quiet: the deletion counts stay RAW, because
AF3's embedder does the `atan(n/3)` squashing itself, and AF2's featuriser does
it on the way in.

**AF3's unpaired chain merge is NOT block-diagonal.** `merge_msa_features` pads
each chain's alignment to the deepest and concatenates along the TOKEN axis, so
merged row r is chain A's row r beside chain B's row r, for every chain, with no
notion of entity - there is no `block_diag` anywhere in AF3. AlphaFold-Multimer
is the one that distinguishes: `_merge_homomers_dense_msa` merges copies of one
sequence densely and block-diagonalises only distinct entities, so it agrees
with AF3 on a homo-oligomer and differs on a heteromer.
`mergeUnpairedChainA3ms`, which block-diagonalises copies too, is neither: it
belongs to the AF2-MONOMER hack, where the +200 residue offset stands in for
chain awareness. AF3 has `mergeRowAlignedChainA3ms`. Two consequences, both
silent: the block-diagonal
form halves the information in every row and doubles the depth to carry it; and
for a HOMO-oligomer the row-aligned merge already IS the paired construction, so
supplying a paired block as well duplicates every row. That combination made a
homodimer fold worse with an MSA than without one, and a monomer shows neither.

**AF3's `msa` is two blocks and its `profile` is over one of them.** The array
the model reads is the paired block followed by the unpaired one; the profile
and deletion_mean are computed upstream, per chain, over the unpaired block
ALONE. So a 32-row A3M gives a 33-row `msa` - the query appears twice, because
an absent paired block becomes the query alone - and a profile over 32. Deriving
the profile from the array instead double-counts the query in every column, and
looks completely reasonable. `unpairedFrom` is threaded through featurise.js for
exactly this.

**AF3 resamples the reference conformer per residue instance** - fixed bond
lengths and angles, random torsions - so a baked table cannot reproduce a dump.
Measured cost of baking one: 0.01 A of structure. `check_af3_featurise.js`
therefore holds the chemistry (bonded pairs, from the bond graph) and lets the
torsions go.

**A ligand's bonds were read correctly and then dropped TWICE, on two paths
that could not see each other.** `ccd-component.js` parses the CCD bond table
and `featurise.js` turns it into the contact matrix AF3's `_embed_bonds` wants -
one direction per bond, `[0,0]` cleared, symmetrised only for the OF3 dialect.
After that:

- **The model never saw it.** `fold.js` assembles the trunk's input as an object
  literal and did not name `bondMatrix`, so the embedder got `undefined` - which
  is indistinguishable from a fold with no ligand. And `embedder-webgpu.js`, the
  one a browser fold runs, had neither the `bondEmbedding` weight nor the term,
  while `embedder-reference.js` had both. `diffuser/evoformer/bond_embedding/
  weights` was in the shipped bundle, downloaded on every fold, multiplied by
  nothing. `tools/gpu/check-af3-embedder.js` passed throughout because its
  fixture carried no bond matrix either: **a feature absent from both sides of a
  differential test is not tested by it.**
- **The viewer never saw it either.** `toPdb` wrote no CONECT records, so
  py2Dmol derived the ligand's bonds from the DISTANCE between atoms - and
  re-derived them from every trajectory frame, whose coordinates are noise until
  the last few diffusion steps. Measured on a six-atom cofactor whose truth is
  five bonds: **4 sticks at convergence, then 4/4/3, 3/4/3 and 2/3/1 as the
  noise grows** - a different molecule every frame. With CONECT it is 5 at every
  noise level, because the bonds stop being a function of the geometry.
- **And `ELEMENT_SYMBOL` had four entries** - C, N, O, S - with everything else
  falling through to carbon. Across a corpus of 51 distinct hetero components,
  **28 carry an element it dropped**: every phosphate-bearing ligand, every
  heme, every metal ion. That is a wrong colour and a wrong radius, and it
  breaks the distance fallback a second way, because that rule is per ELEMENT
  PAIR: a disulfide at 2.05 A read as C-C (ceiling 1.8) vanishes.

`test/af3-ligand-bonds.test.js` covers all of it on the CPU lane; seven
mutations, each caught.

🔴 **AND THE PAE WAS CROPPED TO THE POLYMER, ON A PREMISE THAT WAS WRONG.**
`paeSize` took the top-left `residues x residues` block of AF3's token matrix,
because a ligand is one token per heavy atom and "the matrix is wider than the
residues the viewer draws". The second half of that is not true: **py2Dmol
carries one POSITION per ligand heavy atom too**, and reads them in file order,
which is the order `toPdb` writes, which is token order. Measured across the two
repos on a 20-residue chain plus an 8-atom ligand: AF3 says **28 tokens**,
py2Dmol says **28 positions**, ligand starting at index **20 on both sides**, and
every cell of the matrix lands where its coordinates say - including the
protein-to-ligand corner, which is the whole reason to look at a mixed fold's
PAE. Reported as the PAE missing the ligand part.

Driven on py2Dmol's own page with that 28x28 matrix: the panel sizes itself from
what it is handed (`n = 28`), `pae_n` equal to that width makes its
cell-to-residue crossings the identity, 112k pixels of plot are drawn, and a
drag over the ligand block selects positions **20-27** - the ligand's own. A
ligand-only fold now falls out of the same rule instead of needing the special
case it used to have.

🔴 **AND THE FIRST TEST FOR IT DID NOT CATCH THE BUG.** It called `paeMatrix`
with the full stride and asserted the rows survived - proving the function keeps
what it is given, when the fault was in what the CALLER asked for. Restoring the
crop left it green. `paeSize` is a closure and cannot be called from a test, so
it is READ, the same way the trunk input's `bondMatrix` key is.

**py2Dmol read CONECT partners one column late**, which `trim()` hid up to 9,999
atoms - a right-justified four-digit serial survives a one-column slip, a
five-digit one does not. Serial 10000's partner came back as **1**: not a
dropped bond but a stick drawn to a real atom somewhere else. Fixed in py2Dmol's
`src/io/parse.js`, with the case in its `tests/interaction.js`.

**This build's py2Dmol renderer has no `setColor` or `setColorScheme`** - those
belong to the embed build. Drive the app's own colour `<select>` instead.
Writing `renderer.colors` directly is overwritten on the next recompute.

### The two traps nucleic acids set

🔴 **AN A3M COLUMN IS NOT A TOKEN INDEX.** `msa.set(row, ...)` copies an
alignment row in flat, which is right for exactly as long as one token means
one residue. A modified residue is ten tokens, so from the first one onward
every column of the alignment sat over the wrong residue - silently, with every
array still the right shape. It survived because the two features were tested
apart: the modified-residue work was checked without an alignment and every
alignment dump was a plain protein. ONE DUMP WITH BOTH FOUND IT. The batch now
carries a residue->column map and the msa, the deletion matrix and the profile
all read through it.

🔴 **THE PSEUDO-BETA IS C4 FOR A PURINE AND C2 FOR A PYRIMIDINE**, read out of
AF3's own gather rather than reasoned about. C1' is the plausible wrong answer -
the sugar carbon the base hangs off - and it disagreed. It has to be decided by
BASE rather than by name, because a pyrimidine has a C4 as well, so matching the
name alone takes the wrong atom in three components of five. And a nucleotide's
extra terminal atom is at the OTHER END from a protein's: OP3 at the 5-prime,
where a protein takes OXT at its last residue.

## Performance, and what has already been tried

The pairformer went 3468 ms -> 621 ms over 48 blocks at 59-68 tokens, and then
to **261 ms** with the f16 work of 2026-09-04.

🔴 **AND IT IS NO LONGER SLOWER THAN AF2'S BLOCK, WHICH THIS FILE SAID TWICE.**
"AF3's block is 1.09x AF2's evoformer block - for a block with no MSA row
attention, no column attention and no outer product mean in it" was a standing
complaint, and it is now the other way round: `tools/gpu/bench-blocks.js` at 59
tokens reports **5.44 ms a pairformer block against AF2's 8.93, or 0.61x.**

The reason is not that AF3 got better at the same thing. Both models took the
same class of change, but at 5 MSA rows an AF2 block is almost all PAIR track -
its triangle and its transition, none of which was touched - while the
pairformer is nothing but pair track and every one of its kernels was. At 512
MSA rows, where AF2's block is nine tenths MSA track, AF2 is the one that
moved: 109.25 -> 87.1 ms.

🔴 **AND THE TRUNK IS COMPUTE BOUND, WHICH THIS FILE USED TO SAY WAS THE NEXT
LEAD.** It said ~5 ms a block in the encoder, submit and validation path was
unexplained. It is not there. The labelled compute passes do sum to well under
the wall clock - it was 352 ms against 1201 when this was written - and three
separate measurements say that gap is the instrument and not the machine. The
absolutes below have all moved since (the trunk is much faster now); the SHAPE
is the argument and it has not:

- `Af3PairformerStackGpu` now returns its own `split`. A 16-block pass is
  encoded in **1.3-2 ms** and spends **82, 308 and 1345** inside
  `onSubmittedWorkDone` at 59, 118 and 236 tokens.
- Doubling the tokens quadruples the time, three times over: those same three
  shapes give **87, 318 and 1379 ms**. A pass paying a fixed cost per block does
  not scale like that.
- `tools/gpu/probe-dispatch.js` prices a dispatch before it computes anything:
  **3.5 us** in a shared compute pass, 4.9 in its own, 5.1 when it changes
  pipeline and builds a bind group, 29 when it gets its own encoder and
  submission, and **294 us** for a full round trip (submit, drain, map back).
  At ~500 dispatches a pass that is 2.5 ms.

Merging every dispatch of a block into ONE compute pass was tried on the
strength of the first number and measured 1278 ms against 1269 - nothing. The
gap is `tools/gpu/profile.js`: it adds 30% to the wall clock it is measured
against, and its timestamps are quantised to ~100 us across 1,521 short passes.

What worked, in order of size:

1. **Weight layout.** q, k and the gate were stored `(out, channels)`, so
   consecutive threads read 128 floats apart and nothing coalesced. Transposed
   at pack time into `v`'s `(channels, out)`. This was nearly all of the win.
2. **Row-blocked projection.** One pair row per workgroup meant one multiply per
   weight loaded - arithmetic intensity 1. Eight rows per workgroup, measured
   against 4 and 16.
3. **Submission window 16.** Each `onSubmittedWorkDone` is a full pipeline
   drain: 1 gives 881 ms, 8 gives 622, 16 gives 609, 48 gives 607.
4. **Flash attention** for the grid attention - online softmax, one thread per
   query, vec4 accumulators. Correct and the better kernel, but worth only 2.7%.
5. **`target_feat`'s atom encoder onto the GPU**: 5267 ms -> 160 ms, 33x. It
   reuses `Af3AtomEncoderGpu` by zeroing the three inputs this encoder does not
   have; they enter through bias-free linears of layer-normed values, so zeros
   contribute exactly zero. Checked by `check-af3-target-feat-gpu.js`.

7. **f16 accumulators in the two triangle projections**, 2026-09-04.
   `tri.project` and `tri.project-out` were 23% of the trunk's GPU time between
   them, and are 19% after this and hold their accumulators in WGSL ARRAYS - eight vec4 and eight vec2 -
   which is the shape a driver spills first. In f16: **1.688 -> 1.087 ms and
   1.250 -> 0.875** at 118 tokens, 1.55x and 1.43x, at the tile they already
   had. As the pairformer's wall time: **118 tokens 180 -> 162 ms, 236
   772 -> 702, 384 2200 -> 2034.** `accumulatePrecision`.

6. **f16 for the staged workgroup blocks**, 2026-09-04. Grid attention's key and
   value tile and the pair transition's two blocks are read once per output by
   every lane, and narrowing them halves the workgroup memory that bounds the
   occupancy. As the pairformer's wall time: **59 tokens 55 -> 53 ms, 118
   387 -> 361, 236 848 -> 777, 384 2461 -> 2226** - it grows with the problem.
   The arithmetic is untouched; only the staged copy narrows. `stagedPrecision`.

What did **not** work, measured, so it is not retried:

- **Uploading all 48 blocks' weights once** instead of per block: 30% *slower*
  (640 -> 830 ms), reproducibly. Recycling a few buffers beats holding 384, and
  the up-front burst serialises ahead of all compute.
- **Caching bind groups**: exactly zero, 636-639 either way, though ~1,680 are
  created per stack.
- **f16 weights for the triangle, for SPEED.** `src/triangle/` has had a
  precision option since before this port and `bench-triangle.js` reports 1.40x
  for it at L=128, which reads exactly like an unclaimed win. Wired through to
  the pairformer it measured **377 ms against 378**. The bench's 1.40x is its
  per-call weight UPLOAD shrinking; in the trunk the weights are resident and
  never uploaded, and halving their bytes does not halve the read INSTRUCTIONS -
  these kernels read weights one scalar at a time and this machine is
  instruction-bound. It is still worth doing for MEMORY; see below.
- **The same f16 staging in the MSA STACK, which is the same kernels one stage
  earlier.** Its pair track and transition give 580 -> 516 ms at 236 tokens -
  11%, a bigger relative win than the pairformer's - and the trunk's contact
  probabilities go **1.86e-4 to 5.21e-3**, 28x, with the pair at 6.64e-5
  against 4e-5. Staging only its pair track and leaving the MSA transition
  alone still costs 3.72e-3 for 44 ms.

  🔴 **THE DIFFERENCE IS POSITION, NOT ARITHMETIC.** The MSA stack writes the
  pair representation that all 48 pairformer blocks then read, so its rounding
  is amplified by everything downstream; the pairformer's own is not. That is
  the rule for the next one of these: the same trade is worth taking near the
  output and not near the input.
- **Preparing the diffusion head's weights during the trunk.** A pairformer
  pass encodes in ~5 ms and waits ~340 for the device, and none of the
  transformer's 24 blocks depends on the trunk - so packing them into that gap
  should be free, and the first denoiser call would stop being several times a
  steady one. Measured, the work MOVED and the fold did not: **trunk 1.1 -> 1.4
  s, diffusion 1.4 -> 1.0, total 2.5 either way.** The host time is idle but
  the BUS is not - filling 378 MiB of resident buffers competes with the
  trunk's own traffic for exactly what it saves. It is the same finding as the
  up-front weight burst above, in a new place.
- **Anything that adds registers to the flash attention kernel.** See
  src/evoformer/attention.js: a vec4 q.k accumulator is worth exactly zero
  (the compiler already does it), grouping the keys to amortise the softmax
  rescale is 2.3x SLOWER, and two queries a lane is 4.7x slower. The query and
  the accumulators are already 64 registers a lane and that is the ceiling.
- **Binding AF2's attention kernel directly** rather than rewriting on its
  principles: it takes a uniform for its shape, folds `1/sqrt(d)` into the query
  projection and applies the gate itself. The adapter was wrong at relRMS
  2.96e-1 and cost more to find than the rewrite took.

~~Where the remaining time goes, at 59 tokens: 348 ms of dispatch work and
284 ms of per-block overhead that is not uploads (40 ms) and not bind groups
(0 ms). About 5 ms a block in the encoder, submit and validation path is
unexplained. That is the next lead and it is a small one.~~

🔴 **STRUCK OUT 2026-09-04: THERE IS NO SUCH OVERHEAD.** The paragraph above
survived because the numbers it quotes are real - the labelled compute passes
genuinely do sum to less than the wall clock - but the gap is the profiler and
not the machine. See the compute-bound note in the section above, which prices
a dispatch, splits the encode from the wait, and shows the pass scaling as a
clean square. It was a lead for a long time and it was never there.

### The f16 budget, spent and accounted for

Half precision is used in four places now, and it is worth seeing what they cost
together rather than one at a time. The trunk against the all-f32 tree, 48
blocks and real weights:

| | pair | single | contact | logits |
|---|---|---|---|---|
| all f32 | 6.18e-7 | 6.21e-7 | 9.76e-5 | 4.46e-7 |
| + staged tiles | 1.04e-5 | 2.91e-6 | 1.86e-4 | 6.83e-6 |
| + resident weights | 1.04e-5 | 8.06e-5 | 1.86e-4 | - |
| + triangle accumulators | 1.99e-5 | 8.06e-5 | 1.33e-3 | 1.26e-5 |

So the contacts - the most sensitive thing the trunk emits - are 13.6x their
f32 value. What that bought, measured by forcing every f16 path off with
`--staged=f32 --weights=f32 --accumulate=f32`:

| | all f32 | shipped | |
|---|---|---|---|
| pairformer, 236 tokens | 854 ms | 727 | 15% |
| whole trunk pass, 236 tokens | 1677 ms | 1558 | 7% |
| a fold, cold / warm | 3.23 / 2.18 s | 2.48 / 1.93 | 23% / 11% |
| a fold's device memory | 1406 MiB | 798 | 43% |

🔴 **THE FOUR ROWS ARE NOT THE SAME NUMBER SEEN FOUR WAYS, and quoting the
biggest one is the mistake.** A trunk pass is 7% because the MSA stack and the
template are most of what is left in it and neither was touched; a fold is 23%
cold because the diffusion head's weights are most of its memory traffic and
the f32 arm uploads 608 MiB more of them. The pairformer's 15% is the one that
matches the kernel work.

🔴 **WHAT DECIDES IS WHETHER A NUMBER A USER SEES MOVES, AND NONE DOES.** On
6MRR and 1QYS at flow-8: CA RMSD 0.032 A and 0.005 against the f32 tree, where
AF3's own accuracy on these is 0.7-0.9 A and the sampler's seed-to-seed spread
is ~0.1. Mean pLDDT within 0.01 and worst per-residue 0.04 on 6MRR and 0.22 on
1QYS, on a number the page shows to one decimal. pTM within 2e-4, shown to two. A contact probability
moves by ~1e-3 in [0, 1].

🔴 **AND RMSD IS NOT MONOTONE IN THE ERROR, so do not read one structure as a
verdict.** Adding the triangle accumulators took 6MRR from 0.0086 to 0.0322 A
and 1QYS from 0.0119 to 0.0053 - the sampler is chaotic and both are noise
around a small perturbation. The deterministic trunk numbers above are the
signal; the folds say the scale.

## Memory, which is a separate question from speed

🔴 **THE TOTALS COULD NOT SAY WHICH TENSOR TO ATTACK, AND NOW THEY CAN.** The
allocator was already given a label for every buffer and threw it away.
`memorySnapshot(device)` returns `byLabel` as well as the totals, and
`bench-trunk.js`, `fold.js` and `fold-af2.js` all print it. The answer for AF3
was three rows out of a 1406 MiB fold:

    difftx.block.resident   756.4 MiB   24 diffusion transformer blocks
    w.single-transition     324.1       48 pairformer blocks, 384 channels x4
    w.single                135.2       48 single-track attentions

1216 of 1406 MiB, all of it weights. In f16 they are 608, and **a fold now
holds 798 MiB against 1406; the trunk alone 337 against 567.**

🔴 **IT BUYS NO TIME AND IS NOT SUPPOSED TO - IT COSTS ABOUT 2%.** Halving the
bytes does not halve the read instructions, and the `f32()` at each read is not
free. Two separate measurements, both on the pairformer:

- the PAIR track's weights (triangle and pair transition): 377 ms against 378,
  which is nothing;
- the SINGLE track's, which is where the memory is: 163, 163, 166 ms in f32
  against 166, 167, 168 in f16 across three interleaved pairs.

What it buys is a device small enough to hold the model at all - a 4 GiB
phone's whole budget is 1.3 GiB, which a fold was exceeding on its own.

🔴 **AND THE DIFFUSION TRANSFORMER IS THE EXCEPTION, so do not generalise the
paragraph above to it.** That stack streams its whole weight set once per token
tile instead of keeping it in cache, so there the bytes ARE the cost: 48 -> 41
ms at 59 tokens and 103 -> 89 at 150. See its shader factory.

Two folds say what THIS CHANGE costs, against the same seeds on the all-f32
tree at flow-8 with one recycle: **6MRR 0.0077 A CA RMSD, 1QYS 0.0093 A**, worst
per-residue pLDDT 0.02 and 0.24, bond geometry identical to five decimals. The
triangle accumulators landed after it and moved those to 0.032 and 0.005; the
budget table above is the combined figure and this one is the weights alone.

🔴 **TWO PLACES KEEP f32 ON PURPOSE, AND BOTH ARE ABOUT THE RATIO.**

- **The confidence head's four pairformer blocks.** pLDDT and PAE are what the
  page shows and they are a softmax over 50 and 64 bins - the most amplifying
  thing either model emits. In f16, pLDDT's relRMS goes 1.16e-4 to **2.32e-2**
  and PAE's 5.75e-6 to 2.51e-3. Four blocks of 52 is ~14 MiB, so f32 here costs
  almost nothing and keeps both numbers checked at the tolerance their own
  arithmetic reaches.
- **The pair track's own weights**, offered as `pairWeightPrecision` and off by
  default. They save 38 MiB - 6% of the 608 - and cost the worst amplification
  measured anywhere here: `check-af3-block`'s pair goes from 17x its rounding
  envelope to 51x. A caller that would otherwise not fold can still ask, which
  is what `--budget` already exists for.

🔴 **AND `createTriangleShaders` CONFLATED TWO FORMATS.** Its `precision` named
the weights AND the activations: at "f16" the normalize shader declared
`source` - the pair representation itself - as an f16 array, which is right for
the standalone runner (it converts `z` on the way in) and wrong for anything
sharing that buffer with a track. Wired into AF3 it read every f32 pair value
as two halves of one float and the trunk produced NaN.
`shape.weightPrecision` now narrows the weight buffer alone.

🔴 **EVERY CHECKER THE CHANGE REACHES GREW A PRECISION AXIS RATHER THAN A RAISED
BOUND**, and both arms run: `check-af3-trunk` (two axes, four combinations),
`check-af3-block`, `check-af3-diffusion-head`,
`check-af3-diffusion-transformer`, `check-af3-grid-attention`,
`check-af3-confidence`. Raising one bound would have stopped the f32 path being
checked at all - which is the whole reason the f32 arms still measure what they
did before. The bounds are derived from the arithmetic and the table of
measurements is in each file.

## Open

- **ipTM**, which is what a complex is actually judged by, is still not
  implemented for AF3 - the confidence head emits PAE and PDE, and pTM/ipTM are
  absent rather than approximated.
- **The A3M parser is narrower than AF3's alphabet.** `src/input/a3m.js` rejects
  B, Z, J, O and U, which AF3 maps to D, E, X, X and C. The codes are in
  `AF3_MSA_CODES` and unreachable through that parser - for AlphaFold 2 too, so
  widening it is a change to all three models rather than to AF3's path.
- **No ipTM**, which is what a complex is actually judged by. The confidence
  head emits PAE and PDE; pTM and ipTM are not implemented.
- **Per-atom conditioning is still on the CPU**, but it is no longer a few
  hundred milliseconds and most of it never needed a kernel. Two of its five
  embeddings multiplied by a materialised ONE-HOT - the element, 128 columns
  indexed by atomic number, and the atom name, four 64-way one-hots flattened
  to 256 - which is 384 of its 389 input columns, so 99% of its arithmetic was
  multiplying by zero. As gathers: **72.5 -> 3.7 ms at 59 tokens, 219.8 -> 9.2
  at 150, 343 -> 14.6 at 300**, bitwise identical, and
  test/af3-atom-conditioning.test.js holds it to the matmul form's exact
  floats. Both the summation ORDER and the float64 accumulation are load-
  bearing there; getting either wrong moved 1e-7 through a reference the GPU
  checkers compare against at 1e-6, which is a tolerance nobody chose. What is
  left is ~4 ms of genuine dense work and is no longer worth a kernel.
- ~~**Templates raise** rather than compute.~~ Closed on the CPU 2026-09-04:
  `src/af3/template-features.js` computes all six geometry features and
  `template-reference.js` loops over real slots. Against AF3 on a 16-residue
  query with Top7 in slot 0 of four:

  | | relRMS |
  |---|---|
  | slot 0, real | 5.03e-7 |
  | slots 1-3, empty | 2.35e-7 |
  | the module's 128-channel output | 1.74e-7 |

  The GPU path computes them too, checked against the CPU reference at 32
  tokens with 0, 1 and 4 of 4 slots occupied: relRMS 2.5e-7, 2.2e-7, 1.6e-7.

  🔴 **AND A REUSED BUFFER LOOKED EXACTLY LIKE A WRONG KERNEL.** The first
  version wrote each slot's aatype and geometry into one buffer inside the slot
  loop. `queue.writeBuffer` is ordered against SUBMITS, not against the
  recording of a command encoder - so all four writes landed before the single
  submit and every slot ran against the LAST slot's data. With one occupied
  slot of four the module computed the all-empty answer, which differs by only
  that slot's quarter share: relRMS 2.1e-2, small enough to read as precision
  and wrong enough to lose the template entirely. One buffer per slot.

  🔴 **AND IDENTICAL SLOTS RUN ONCE BETWEEN THEM.** Four empty slots produce
  the same embedding by construction, so a naive per-slot loop is 4x the work
  for an identical answer on every de novo fold - the trunk's template stage
  went 83 ms -> 150 ms before this was put back. Each pass carries how many
  slots it stands for and the accumulate shader multiplies by it, so a fold
  with no templates runs one pass, which is what it always did.

  🔴 **AND THE CROSS-CHAIN MASK WAS DEFAULTING PERMISSIVELY.** AF3 masks the
  geometry features ACROSS chains, because its `Template` is one protein chain
  and a complex's chains are templated by separate searches - so a cross-chain
  distance is computed from two structures that were never in one frame. The
  embedders defaulted to "every pair is intra-chain", which is right for a
  one-chain query and is what every check had. Measured on a two-chain query
  with a template on EACH chain:

  | mask | slot 0 |
  |---|---|
  | per chain, as AF3 | 5.5e-7 |
  | all-ones | **1.09** |

  So it is most of the module's answer, not a correction. The mask is derived
  per slot from `asymId` now and there is no permissive default: a template
  with neither `asymId` nor an explicit mask raises.

  ### Inter-chain templates, which AF3 does not do

  The masking is about PROVENANCE, not modelling. When two chains come from
  ONE file - a real co-crystal, or a complex this page predicted - they ARE in
  one frame, and the cross-chain distances are exactly the interface geometry a
  binder method wants. So `spanChains` is a flag on the SLOT, not a setting on
  the model: one slot built from one structure may span while another built
  from a separate search may not, in the same fold. Spanning opens only the
  pairs the slot covers at BOTH ends, because a pair with one end outside the
  template is still two frames apart.

  Measured at 32 tokens over two chains, against the same slots masked per
  chain: spanning moves the module's output by relRMS 1.5e-2 with one occupied
  slot and 4.9e-2 with four. There is no oracle for this - AF3 does not do it -
  so it is checked by construction: the GPU and CPU paths agree to 2e-7 on
  every arm, and an arm that failed to move would fail the check.

  The page is unchanged: nothing yet builds `slots` from a user's structure.

  🔴 **ONE THING WAS WRONG AND ONLY THE ORACLE COULD HAVE SAID SO.** The unit
  vector is `R_i^-1 (t_j - t_i)` - the FRAME is the row index and the POINT is
  the column - because AF3 writes
  `rigid[:, None].inverse().apply_to_point(points)` and the broadcast puts
  frames on axis 0. Written the other way it is the exact transpose: still unit
  length, still smooth, still masked correctly. Measured both ways against
  AF3's own `make_backbone_rigid`:

  | | unit vector | distogram | both masks |
  |---|---|---|---|
  | frame on the row | **1.29e-7** | 0 | 0 |
  | frame on the column | 1.51e+0 | 0 | 0 |

  Everything else was bit-exact either way, so nothing but a real template
  could have found it - which is precisely the argument the module's own
  header used for not writing these features at all.

  The rest of the entry, kept because it is what the checkers read: It was "the geometry features are unverifiable without a
  reference", which was true: with no template all six are identically zero, so
  nothing here could tell a correct implementation from a wrong one.
  `tools/oracle/dump_af3_trunk.py --template <pdb>` now produces one. Measured
  on a 16-residue query with Top7 as its template, four slots:

  | slot | template | module output |
  |---|---|---|
  | 0 | real, 374 atoms | mean -0.0781 std 2.0651 |
  | 1, 2, 3 | empty | mean +0.2583 std 0.7561, all three IDENTICAL |

  which is the module's documented behaviour seen from outside: an empty slot
  is a learned transform of the QUERY and three of them produce the same
  answer. The dump carries `template_aatype [4, 16]`,
  `template_atom_mask [4, 16, 24]` and `template_atom_positions [4, 16, 24, 3]`
  as inputs, and the per-slot 64-channel output beside the module's 128-channel
  contribution - so the six features can be checked in isolation from the two
  pairformer blocks that follow them.

  The two constant tables they need are small and, for protein, trivial:
  `RESTYPE_PSEUDOBETA_INDEX` is dense slot 4 (CB) for every amino acid except
  glycine, which takes slot 1 (CA); the backbone frame is
  `RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[:, 0]` = (2, 1, 0) = (C, CA, N) for all
  twenty. Nucleotides differ and AF3's own `Template` is documented as one
  protein chain.
- ~~**AF3's block is still 1.09x AF2's** for strictly less work.~~ Closed
  2026-09-04: it is 0.61x. See the performance section.
- **No RNA alignment.** AF3's pipeline searches an RNA database; single-sequence
  is what an RNA chain gets here, and the status line says so. The seam is
  ready for one - featuriseProtein reads the MSA by ALIGNMENT COLUMN now, and
  a nucleic chain's columns are simply absent from a protein A3M.
- **No modified bases**, and no nucleic ligand bonds: a DNA chain and a ligand
  in the same job are two separate molecules to the featuriser.

## Pairing, as the server actually returns it

`generateMmseqs2PairedMsa` posts every distinct sequence of a complex to
`ticket/pair` as `>101, >102, ...` with `mode=pairgreedy`, and `pair.a3m` comes
back holding one NUL-separated block per query. Confirmed live on 3RPF's two
chains (146 and 74 residues): 9,904 rows for each, equal depth, each block
carrying its own query and its own width, and no all-gap padding rows.

The pairing is real and not merely aligned. Row 2 of both chains is
`UniRef100_UPI00129C3066`, one 214-residue protein matching chain A over 83-209
and chain B over 1-75 - a single partner supplying both halves, which is the
signal the paired block exists to carry.

It took 88 s for that pair, against about the same for the unpaired searches
that run alongside it. A homomer skips this entirely.

End to end through the page, AlphaFold-Multimer on those two chains at three
recycles: **pLDDT 96.5, pTM 0.907, ipTM 0.897** in 684 s. ipTM is the number
that matters and the one pairing is for - a confident INTERFACE needs
cross-chain coevolution, which is precisely what the paired block carries and
what a heteromer folded without until now.

AF3 takes the same alignment and the budget split is AF3's own: of 9,904 paired
and 7,283 unpaired rows, it reads 255 paired + 256 unpaired + the query = 512,
with `unpairedFrom` at 256. That is `max_paired_sequences = msa_size // 2` with
the remainder to the unpaired block, which is what featurise.js needs to compute
the profile over the right half.

🔴 THE FOLD PASSED ONE MSA ROW TO THE TRUNK, and that was the whole of it.
`src/af3/fold.js` called the trunk with `sequences: 1` and sliced every MSA
array down to the query, so the MSA stack never saw an alignment. Fixed; the
numbers below are after.

It is worth knowing how it hid, because the next bug of this kind will hide the
same way. An alignment still reached the model: `profile` and `deletion_mean`
are computed over all of it and ride into `target_feat`, so supplying one DID
improve the fold (44.5 -> 62.6 pLDDT on chain A) and the status line honestly
reported the depth that had been FEATURISED. Nothing reported the depth the
trunk was handed. `foldBatch` now emits it and `tools/gpu/fold.js` prints a
marker when the two disagree.

It also explains evidence that fitted none of the theories being tested: more
recycles made the structure worse while raising pLDDT, the sampler and the
precision barely mattered, and every component measured exact against AF3 while
the assembled fold was poor. The model was right the whole way down.

### Against the AlphaFold Server, on 3RPF

Its own run of the same two chains is the reference (`useStructureTemplate:
false`, ptm 0.91 / iptm 0.91 across five seeds), and its per-chain MSAs are in
the zip, which is what makes this comparison clean - no MMseqs2 in the loop.

| | before | after |
|---|---|---|
| chain A, 146 res | 9.96 A, TM 0.409, pLDDT 58.1 | **1.10 A, TM 0.962, pLDDT 87.9** |
| both chains, 220 res | 17.21 A, TM 0.196 | **0.32 A, TM 0.997, pLDDT 93.9** |

217 of 220 CA within one angstrom. 6MRR is unchanged at 0.76 A, because a
single-sequence fold always had one row.

Through the page, with ColabFold's own MSAs rather than the server's: the same
complex reaches **pLDDT 94.4**, against 62.6 before.

🔴 AND IT NOW COSTS WHAT AN MSA COSTS: 227 s against 84 s, because the MSA stack
has 512 rows to work on instead of one. The old number was cheap because it was
not doing the work. `--max-msa` trades this back if a fold has to be quick.

### Templates are not the difference

The server scores the same with them off, so do not implement templates to
chase a complex that folds badly. That was the prime suspect for an hour and it
was wrong.

### Fixed: a fold crawled in a background tab

The per-block yield was `setTimeout(resolve, 0)`, and Chrome clamps setTimeout
to >=1 s in a hidden tab - so a 48-block pass that takes under a second took
the better part of a minute and the trunk appeared to hang. Measured at the
time: pass 1 reached block 11 in five minutes hidden, then jumped to block 28
the moment the tab was touched.

`src/runtime/yield.js` is the fix (commit 882e3f2) and the whole of it: a
MessageChannel message is a task, so it still lets the page paint and still
lets Stop respond, but it is not a timer and is not clamped.

## The weights

🔴 **`model-af3-int5` is DeepMind's AlphaFold 3, not OpenFold3.** Every manifest
in the lineage says so - `model.name` is `alphafold3`, `source` is
`af3.bin.zst`. They carry a Prohibited Use Policy. `tools/build_site.py` reads
the manifest and refuses to publish them without
`LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3`, and the deploy workflow demands the
same repository variable before it untars the release - that check is on the
only path that actually publishes.

An Apache-2.0 bundle needs an OpenFold3 export. The blob is at
`~/af3_converted_cd2/of3_ported_weights.bin.zst` and
`tools/export_af3_model.py --model openfold3` exists, but the OF3 dialect turns
on four branches the stock graph does not have - a column-attention pair-bias
swap, a symmetrised bond matrix, an element index shift, and Fourier weights
read from the checkpoint - and none of those are implemented or verified here.

int5 costs nothing measurable: 1405 MiB -> 265 MiB, and 6MRR folds to 0.66 A
against float32's 0.69, which is the spread between diffusion seeds.

## What a denoiser call costs, and what made it cost less

A sampler calls the diffusion head up to 200 times and everything else once, so
the head is the whole optimisation target. On a 59-residue chain, steady state:

| stage           | as found | now |
|-----------------|---------|-----|
| conditioning    |    48   |  11 |
| atom encoder    |   100   |  18 |
| single-proj     |    -    |   2 |
| transformer     |   549   |  72 |
| atom decoder    |    48   |  24 |
| **one call**    | **760** | **134** |

🔴 **AND AT 150 TOKENS IT IS A DIFFERENT KERNEL LIST, WHICH IS WHERE THE 2026-09-03
WORK CAME FROM.** The table above is a 59-residue chain, where the atom blocks
are small. At 150 tokens a call was 267 ms - transformer 143, atom decoder 59,
atom encoder 46 - and **three kernels were reading the conditioning from global
memory once per token per channel, on every lane**: `ffw-out` and
`attention-output` in the transformer, and the atom blocks' `output`. The value
is indexed by (token, d) and never by the channel a lane owns, so all 256 lanes
of a workgroup wanted the same tile-by-C_COND floats. `conditionedProject` did
the same with two tensors.

| kernel | before | after | |
|---|---:|---:|---|
| ffw-out | 33.4 | **14.1** | the token tile lifted from two, then the conditioning staged |
| attention-output | 19.1 | **9.0** | the zero-gate loop was 60% of it |
| output (x3) | 11.6 | **10.0** | five global reads of the conditioning became one stage |
| project / project-keys | 3.3, 2.1, 2.1 | **1.8** | act AND queries_cond, four loops each |

A denoiser call **267 -> 221 ms**, its transformer 143 -> 111. A 200-step fold
at that length is about nine seconds less.

🔴 **AND WHERE THE STAGE GOES IS DECIDED BY RESIDENCY, NOT BY STYLE.** Three of
these reuse an array that is already dead - `wt` in ffw-out, `gated` in
attention-output, the raw tensors in conditionedProject - because a second
array would have taken those kernels from 6 to 12 KiB, or 8 to 16, and halved
how many workgroups a core can hold. The atom `output` kernel is the exception
and gets a fifth array: it already holds 24 of this device's 32 KiB, so it is
one workgroup a core either way and the array is free. It also CANNOT reuse
`cond_norm`, because its second adaptive-zero projection reads the RAW
conditioning after the normalised form has been written - which would have been
silent.

So 200 steps is about 27 seconds on a 59-residue chain, where it was 152, and a
whole diffusion-200 fold measures 26.3 s end to end against a flow-8 fold's 2.6.

🔴 **AND AT 150 TOKENS THE HEAD IS A DIFFERENT SHAPE, WHICH IS WHERE THE LAST
WIN CAME FROM.** A call there is 261 ms - transformer 133, atom decoder 61, atom
encoder 45, conditioning 15 - and `ffw-out` alone was 33.4 of it. Its token tile
was pinned at two by `Math.min(2, tile, fits(intermediate))`, a sizing term that
assumed a workgroup staged `outTile * INTERMEDIATE` floats. Chunking had removed
that long before: it stages `outTile * outChunk`, 6 KiB at four. The cap was
never lifted, so the measurement that set it - outTile 4 at 85 ms against 2's 74
- stood against a kernel that no longer existed.

The tile is what amortises the weight read, one per step of the intermediate
multiplied into every token of the tile. Re-measured with the chunking in place,
as medians of repeated runs of bench-diffusion-transformer.js: **at 150 tokens
2 -> 138 ms, 4 -> 128, 8 -> 132; at 59 tokens they tie.** `ffw-out` 33.4 -> 25.0,
the transformer 143 -> 133, a call 267 -> 261. Arithmetically neutral to every
digit - each token's sum runs over the same intermediate in the same order
whatever the tile - and check-af3-diffusion-transformer.js reports the identical
relRMS either side.

The lesson is not the tile. It is that a cap and the measurement justifying it
outlive the kernel they were about, and nothing fails when they do.

THE REST OF THE FOLD, for scale, all on the same 59-mer:

| trunk pass, 32 MSA rows   | 756 -> 570 ms  |
| trunk pass, 1024 MSA rows | 1093 -> 804 ms |
| AF3 checkpoint load       | 5470 -> 1364 ms |
| AF2 monomer / multimer load | 1012 / 874 -> 417 / 400 ms |

🔴 **AND BOTH HOT PATHS ARE NOW FLAT, WHICH IS WHERE THE CHEAP WORK ENDS.** The
head's largest kernel is ffw-out at 21 ms of 134; the trunk's top six were
pair-transition 84, tri.project 63, tri.project-out 43, grid.project 41,
grid.attend 40, grid.project-out 39, with no outlier. Everything left is a
kernel rewrite - tiling both operands in shared memory - rather than a shape
fix, and the failures listed below are what that has to beat.

## The pairformer's kernels, rewritten for the shape rather than the arithmetic

A trunk pass went **540 -> 408 ms** at 59 tokens, **3.38 -> 2.25 s** at 150, and
**670 -> 544 ms** at 59 tokens with a 1024-row alignment; its pairformer 435 ->
311 and 2879 -> 1900, and its MSA stack at that depth 334 -> 196. A denoiser
call went 134 -> 111. 6MRR folds to 0.64 A, TM 0.960. AF2's evoformer shares five of these
kernels and its triangle projection went 0.581 -> 0.422 ms a block and its
output projection 0.405 -> 0.327, with the contraction dropping off the
profiler's list entirely.

🔴 **AND ITS BLOCK TOTAL IS NOT MEASURABLE TO THAT PRECISION HERE.**
`profile-af2-block.js` reports a block at 11.0 to 12.7 ms for the SAME build
across processes, so a 4% change in it says nothing; the per-dispatch numbers
above are stable to about 1% and are what a claim about AF2 should rest on. An
earlier "12.6 -> 10.4" in this file was two numbers from two processes and has
been withdrawn.
Nothing computes anything different; every checker is unmoved and the denoiser's
worst error against AF3's own moved 1.19e-5 -> 5.86e-6.

🔴 **THOSE ARE UNPROFILED NUMBERS AND THE TABLE BELOW IS NOT.** `--profile`
writes a timestamp pair per compute pass, and at these shapes that is about a
fifth of the trunk: the same build measures 439 ms without it and 528 with. Use
the per-pass numbers to rank kernels against each other, never to quote a total,
and take before/after totals from two runs that are both unprofiled.

| kernel | before | after | what changed |
|---|---|---|---|
| pair-transition   | 83.3 | 73.0 | chunked, then its rows made vec4 lanes |
| tri.project       | 61.3 | 45.4 | 2x2 -> 4x2, one vec4 a cell, rows staged as one |
| tri.project-out   | 42.8 | 32.8 | the same |
| grid.project      | 41.1 | 38.4 | q/k/v/gate interleaved, read as one vec4 |
| grid.attend       | 39.0 | 21.7 | a chunk of keys staged in workgroup memory |
| add               | 11.2 |  2.4 | four of the five folded into their producer |
| grid.project-out  | 39.1 | 15.6 | a tile of rows, where it was one |
| single-transition | 28.1 | 29.6 | untouched |
| tri.contract      | 23.3 |  8.6 | 1 output a thread -> 4x4, both tiles vectors |
| opm.contract      | 15.7 |  9.0 | a block of (i, j) pairs, and a bigger chunk |
| pair-logits       | 11.8 |  4.9 | heads as vec4, normalised once |
| grid.bias         |  5.6 |  3.0 | the same |
| single.project    | 20.4 | 11.1 | the width split over workgroups |
| single.project    | 20.4 | 11.1 | the width split over workgroups, outputs blocked |
| tri.normalize     | 13.7 |  8.3 | the LayerNorm staged, to coalesce |
| grid.normalize    | 11.8 |  7.4 | the same |

Every one of those is a ratio of reads to multiply-adds or a count of
workgroups, and every one is measured by a bench that runs in about a second an
arm - `bench-triangle-project.js`, `bench-grid-project.js`,
`bench-transition.js`, `bench-single-project.js` - against `bench-trunk.js`'s
forty seconds and 48-block average. Each checks its arms against the first,
because a tile the dispatch does not match leaves rows unprocessed and reads as
a speedup.

🔴 **AND THE NEXT PROTEIN IS NOT THIS ONE. `grid.attend` IS CUBIC IN N.**
Everything else in the pairformer is quadratic. Before it was staged, the
attention was 39 ms of 400 at 59 tokens and **564 of 2429** at 150 - the largest
kernel in the trunk by half again. Staged it is 21.7 and 318, second at both,
but the exponent has not changed and it will lead again on a longer chain.
Anything further should be measured at 150, not at 59.

🔴 **AND ON A DEVICE WITH MATRIX UNITS THIS KERNEL IS NOW A DIFFERENT ONE.**
`src/af3/grid-attention-matrix.js` is the same online-softmax flash attention
AF2 runs, and it is 1.40x to 1.53x over the staged scalar kernel below across
200 to 640 tokens on an A100 - the whole measurement, and the AF2 tile rule that
does NOT transfer to it, are in docs/A100.md. It is off unless
`gridAttendMatrix` is set, and everything in this section is still what runs
everywhere else.

**Staging is what fixed it, and the reason generalises.** The dispatch gives a
workgroup one (pair row, head) and sixty-four queries, and the key loop runs over
the same axis for all of them - so each of the `2 * dimension/4` vectors a key
needs was fetched by sixty-four lanes issuing sixty-four IDENTICAL global loads.
A chunk of keys in workgroup memory makes that one load and sixty-four workgroup
reads: 1.9x at every length measured (0.425 -> 0.237 ms at 59 tokens, 5.75 ->
3.05 at 150, 36.9 -> 19.5 at 300). Chunks of 16 and 32 tie and 64 loses, so the
bound is 8 KiB. Two shape notes: the kernel reads `workgroup_id` rather than
`global_invocation_id`, because WGSL's uniformity analysis has to SEE that row
and head are workgroup-uniform or it rejects a barrier under the branch on them;
and a lane past the last query no longer returns, because it has to reach every
barrier the staging loop makes.

### What was tried on these and lost

- **Accumulating the query-key score into vec4s** reduced once, instead of a
  chain of `dot()`s - which looked like the problem, since `dot()` is four
  multiplies and four DEPENDENT adds and eight of them are a chain about
  thirty-two deep. Interleaved in one process at 150 tokens: the dot form
  5.10 ms, one accumulator 5.80, two 6.00, four 5.05.
- **More than one query per attention invocation.** The attention reads
  `dimension/4` vectors of k and as many of v per key and does the same number
  of vector operations with them - one load per multiply-add - and those loads
  do not depend on the query, so two queries an invocation should halve them.
  At 150 tokens it was **1.85x slower**, and four queries 3.9x: a query costs
  `dimension/4` vectors of q plus as many accumulators, so two is already 128
  floats of register and it spills.
- **Widening the transition's workgroup to 256 lanes** where the single track
  has only 59 rows to hand out: 0.728 ms against 128 lanes' 0.591. The
  LayerNorm's reduction grows a level and 384 channels split unevenly.
- **Raising the transition's row tile to 8 without chunking the intermediate**:
  1.77x slower. It fits in the 32 KiB this device grants and leaves one
  workgroup resident per core.
- **Blocking the transition's first matmul over i**, on its own: nothing.
- **Barriers.** Priced by removing them from the projection's k loop: exactly
  zero. The step stays at 8.
- **Batching a pairformer block's dispatches into ONE compute pass**, the way
  AF2's stack does. A trunk pass opens 1,332 of them, and the profiler's timed
  GPU work summed to 335 ms of a 451 ms wall - which looked like 116 ms of pass
  boundaries. It is not: batching measured 312 ms against 311. The gap was the
  profiler's own timestamp writes (451 profiled against 403 not) plus the
  labels the report does not list. Pass boundaries cost nothing here, and
  splitting them per dispatch is what lets profile.js see in, so they stay.
- **Splitting the SINGLE track's transition into two dispatches.** Its rows are
  its tokens, so the fused kernel gets 59 workgroups on a 59-residue chain and
  cannot tile out of it - 27 ms of a 307 ms pairformer for a fiftieth of its
  arithmetic. Splitting the two matmuls apart, with the widened intermediate
  travelling through a 363 KB buffer, gives twelve times the workgroups and the
  same weight traffic. It measured **65 ms** - a widening pass of 53.5 and a
  contraction of 11.9 - against the fused 27. The widening repeats the row's
  LayerNorm once per slice of the intermediate, twelve tree reductions a row
  where there was one, and that is more than the occupancy was worth. A third
  dispatch to normalise once would remove it; the fused form is 27 ms and this
  would have to beat it from 53.5, so it was not pursued.
- **Reading the transition's widening weights as vec4.** Its two weight reads a
  channel were half its instructions, and consecutive lanes read consecutive
  slots - so four consecutive slots to a lane makes those two reads two vec4
  reads and the multiply-adds eight: eleven instructions to buy thirty-two where
  it was five to buy eight. It needs a chunk of four workgroup widths, and at
  the tile that then fits it measured 1.63 ms against the current shape's 1.39.
  A third measurement saying this kernel is not waiting on its weight reads.

### What the MSA stack cost, once anyone measured it at depth

At 59 tokens and 1024 rows the stack was 334 ms against a 310 ms pairformer -
the untouched half of the trunk. Two kernels were most of it and both had the
same shape of fault:

| kernel, 1024 rows | before | after |
|---|---|---|
| opm.contract | 113 | 60 |
| msa.project | 62 | out of the top twelve |

`msa.project` gave a ROW TO A THREAD, walking WIDTH outputs by C_M channels and
re-deriving the normalised activation inside both loops - so a row's 64 values
were recomputed 64 times each, and two thirds of the kernel was that. A
workgroup a row: 64 lanes share the reduction, stage the normalised row once,
and take an output each.

`opm.contract` needed a two-dimensional block, and the reason generalises. A
cell's product is `left[i][c] * right[j][e]`, so an i-by-j block of pairs reads
BLOCK_I values of left and BLOCK_J of right to make BLOCK_I * BLOCK_J products;
a run of consecutive SLOTS shares an i only by accident, and nothing the
compiler can see says so.

### The ceiling these are measured against

`tools/gpu/probe-alu.js` asks the device directly, with no memory in the way:

| | GFLOP/s |
|---|---|
| scalar f32 multiply-add | 1287 |
| vec2 | 2526 |
| vec4 | 5034 |

and 396 billion workgroup-memory reads a second. Every one of those is about
**640 billion instructions a second**, which is the number that actually
governs: a vec4 multiply-add and a scalar one and a workgroup read all cost one
instruction, so this is an instruction-count machine and vectorising pays
exactly when it reduces the count.

That reframes everything above. The trunk's kernels ran at 900 GFLOP/s to
1.1 TFLOP/s, which is not 25% of a 3.6 TFLOP/s paper peak - it is **70-85% of
the scalar ceiling**, and about a third of the instruction rate once their loads
are counted. The remaining factor is in the loads, not the arithmetic.

🔴 **AND HALF PRECISION MOVED THE CEILING ITSELF, which is the one way past
that argument.** An f16 multiply-add issues at 1.7x an f32 one for the same
instruction, so a kernel at 85% of the f32 scalar ceiling is not finished - it
is finished IN f32. The kernels that took f16 accumulators now run past that
line; see the f16 budget section above and tools/gpu/profile-af2-block.js.

🔴 **WHICH IS WHY VECTORISING THE TRANSITION BOUGHT NOTHING BY ITSELF.** Its
tile's rows became vec4 lanes - four multiply-adds into one, and a quarter of
the workgroup reads - and the pair shape measured 1.488 ms against the scalar
1.475. The kernel waits on its two weight reads per channel, not on its
arithmetic. What the vectorisation DID buy is room for the tile: as scalar code
tile 8 lost to tile 4 (1.556 against 1.525), and vectorised it wins (1.394
against 1.494), because the rows now cost a quarter of the workgroup memory they
did.

🔴 **AND RANKING THE TRUNK'S KERNELS BY GFLOP/s MISLEADS, WHICH COST A DAY'S
LEAD TO FIND OUT.** `tri.project-out` measures 672 GFLOP/s against
`tri.project`'s 977 on the same shape, and it had the fault to match: it
contracts TWO matrices at each (k, output channel) cell and staged their
weights as two SCALAR arrays, where projectAB packs its four into one vec4.
Scalar, its inner loop was two source reads, four weight reads and sixteen
multiply-adds a step of k - 0.73 useful operations an instruction against 2.9.
Packing the pair into a vec2, with one vec2 accumulator a cell, should have
been worth about 1.5x.

It was worth **3%** (210.5 -> 205.2 ms), because the kernel is not instruction
bound. Counted properly, per dispatch at 150 tokens:

| | source | weights | store | total | achieved |
|---|---:|---:|---:|---:|---:|
| tri.project | 92 MB | 184 | 23 | 300 MB in 2.675 ms | **112 GB/s** |
| tri.project-out | 184 MB | 92 | 12 | 288 MB in 1.950 ms | **148 GB/s** |

Both are AT or ABOVE the 114 GB/s `probe-alu.js` measures for streamed global
reads. They differ in GFLOP/s because they differ in multiply-adds per BYTE -
project-out reads two source tensors and one set of weights, project reads one
and four - and not because one is better written than the other. The change is
kept because the code is simpler and the 3% is real, not because the reasoning
that motivated it was right.

The lever on a memory-bound kernel is traffic, and traffic is what the tile
divides - which is why every tile here has been swept and why every sweep ends
at the same place: a bigger tile cuts traffic and loses more to occupancy. 32x16
beats 32x32, 32x8, 16x16 and 16x32 at 150 tokens.

🔴 **AND THE KERNELS ARE NOW AT THE PRACTICAL CEILING, WHICH IS NOT THE PAPER
ONE.** At 150 tokens with a 512-row alignment - a real fold - a trunk pass is
2.60 s, and every one of its top kernels runs at about **270 billion
instructions a second** against the 640 billion probe-alu.js measures with no
memory in the way. Their instruction counts are within about 1.3x of what the
arithmetic needs. The remaining factor is latency the machine is not hiding, and
it does not yield to another tile: the list above is eight attempts at cutting
instructions further, and every one measured worse or level.

🔴 **AND PRICING A READ BY SUBSTITUTING A CONSTANT OVERSTATES IT.** Replacing
the projection's weight-tile reads with a constant took it from 0.525 to 0.375
ms, suggesting 29% to win; packing those four reads into one vec4 - which is as
far as that goes - was worth 5%. A constant lets the compiler hoist the
multiply-add too, so the arm measures the read AND the arithmetic that depended
on it. Useful for ranking, useless as a target.

🔴 **AND ONE ALGEBRAIC IDENTITY IS NOT ONE HERE.** `grid.attend` subtracts 1e9
from each masked logit inside an `if`. Computing that penalty once per key and
ADDING it - with `select(-1.0e9, 0.0, masked > 0.0)`, or with a plain `var` set
in an `if` - is the same expression and measures **relRMS 2.24e-1** against the
CPU reference where the `if` measures 9.63e-7. Deterministic, and identical to
the last digit whichever of the two rewrites is used, so it is a real difference
and not noise. It was not run down. Do not rewrite it.

The matrix kernel routes around it rather than resolving it: it hoists the
mask's READ out of the per-query-row loop into a staged per-key array, which is
what the redundancy was, and leaves the conditional itself written exactly as it
is here. Whatever this is, it is still unexplained.

What paid, in order of size:

1. **Two kernels had no grid at all.** `aggregate` and `single-initial` both
   dispatched ceil(TOKENS/64) workgroups with one thread to a token - ONE
   workgroup for any protein under 64 residues - and each lane then walked a
   whole matmul, 2.4M multiply-adds in `aggregate`'s case. It was 43 ms of the
   atom encoder's 82: more than its three cross-attention blocks put together,
   in the pass that only pools their output.
2. **The transformer was weight-bandwidth bound by 25x.** One workgroup per
   token meant every workgroup read the block's entire weight set: 5.9M floats
   for 2.4M MACs, a quarter of a MAC per byte where the device needs about
   twelve. A call read 33 GB of weights, which at ~350 GB/s is the 107 ms it
   took. Tiling over tokens - with the output range split so tiling does not
   cost occupancy - took the stack to 74.
3. **The block loop awaited `popErrorScope()` AND `onSubmittedWorkDone()` per
   block**, two host-device round trips a block, 48 a call. `DeferredValidation`
   exists for exactly this and the pairformer had used it for a year.
4. **Per-token workgroups were 64 lanes wide**, so the token count was the
   occupancy. 256 is the ceiling and the optimum.
5. **Weights packed and uploaded per call.** Now packed once per weight object
   and resident on the device - src/runtime/resident.js.
6. **The key rows are a gather of the query rows**, four slots to an atom, and
   the atom transformer projected each slot separately. Projecting per atom and
   expanding is a quarter of the work and numerically identical.
7. **The pair conditioning does not depend on sigma** and was rebuilt 200 times.

🔴 **AND SIX THINGS THAT LOOKED OBVIOUS AND WERE WORTH NOTHING**, which is most
of what this section is for. Batching the 24 per-block submits into one encoder
(199 vs 205 ms). Skipping ~14 MB of per-call readback that is immediately
re-uploaded (4 ms - unified memory makes a copy back nearly free). Widening the
ATOM kernels the way the transformer's were widened (they already launch 1440
workgroups). Replacing the atom kernels' redundant serial LayerNorms - all 64
lanes walking all 128 channels, four times - with workgroup reductions, which is
strictly less work and landed inside the noise. Raising
maxComputeWorkgroupStorageSize to lift the token tile from four to eight, which
lifts a ceiling that was never binding. And tiling further in general: 4 and 2
beat every larger pair measured.

🔴 **TWO MEASUREMENTS LIED, BOTH BECAUSE THEY WERE TOO CHEAP.** A bisect of the
atom encoder on a bench that averaged two calls, with a ten millisecond spread,
reported a REMOVED pass as costing negative time and named the attention blocks;
`aggregate`, four times bigger than anything guessed, only appeared once
bench-head.js reported a median over nine calls with its range. And a 30%
"speedup" from tile 8 was the shader factory defaulting the tile to 4 while the
dispatch divided the token count by 8 - half the tokens were never projected. It
was caught by two numbers disagreeing that should have been identical, not by a
checker; the factory now throws rather than defaulting.

🔴 **THE TRANSFORMER IS AT A LOCAL OPTIMUM AND FOUR THINGS SAY SO.** Its tile
and lane counts are both measured maxima; giving each workgroup ONE of q/k/v/gate
instead of four - on the theory that 4 x TILE accumulators were spilling - was
worse at 59 tokens and at 200 (the token tile is then staged four times);
reading the token tile from global instead of workgroup memory, to let the tile
grow past what that memory caps, was worse again; and `splits` makes no
difference at all, so the lane imbalance it creates at SLICE 384 against 256
lanes is not costing anything. f16 is settled too, in README.md: 13% SLOWER at
1.89e-4 error, because Apple GPUs run f32 and f16 ALU at the same rate and these
kernels are ALU-bound.

🔴 **AND THERE IS A BETTER TOOL THAN THE ONE USED HERE.** Every bisect above
disabled a pass and re-measured, which is noisy and cost two wrong conclusions.
`timestamp-query` is in the features src/runtime/device.js already requests, and
README.md records per-kernel timestamp profiling being used on the triangle
stack. Use that first next time.

The remaining floor is the transformer's arithmetic intensity, which tiling by
four improves and does not fix. Measure with
tools/gpu/bench-diffusion-transformer.js, which takes about three seconds
because it synthesises its weights, and gate any change on
tools/gpu/probe-head-vs-af3-steps.js.

## The denoiser, and the law that governs it

A call is **125 -> 122 ms** at 59 tokens, its transformer 73 -> 67, from chunking
and vectorising `ffw-out` (20.0 -> 16.7 ms of the 24 blocks). That is small, and
the reason it is small is the useful part.

🔴 **AND THAT MODEL WAS WRONG, WHICH TOOK A PROFILE AT 240 TOKENS TO SEE.**
Halving the traffic by doubling the tile bought nothing once the structure
allowed it (323 ms against 324), so the stack is not bandwidth-bound; the
numbers below were a coincidence of scale. What it IS bound by is the
instruction count of four kernels that each read one or two weights per
multiply-add, and restructuring those took a 240-token stack **324 -> 234 ms**
and a 59-token denoiser call 121 -> 117:

| kernel, 240 tokens, 8 blocks | before | after | what changed |
|---|---|---|---|
| pair-logits | 27.2 | 7.4 | heads as four vec4, channels outside |
| adaln | 8.6 | 3.1 | a tile of tokens |
| ffw-adaln | 7.9 | 2.8 | the same |
| attention-output | 12.5 | ~10 | a tile, and one output an invocation |

`pair-logits` is the one that mattered: it is quadratic in tokens where the
token projections are linear, so it leads on any real protein. It looped heads
OUTSIDE channels, re-reading the normalised pair row for each of the sixteen and
reading one weight per multiply-add - 48 instructions to buy 16. The heads are
contiguous in the projection, so they are the vector: channels outside, four
vec4 accumulators, nine instructions.

🔴 **AND ONE OUTPUT AN INVOCATION IS WHAT MADE THE REST OF IT WORK.** These
kernels' accumulators are (matrices x tile groups x outputs a lane) vectors and
every one is live across the whole channel loop, so a second output a lane
doubles the registers - enough to spill at eight tokens, where tile 8 measured
542 ms against tile 4's 332. Each kernel now splits its own output range to
exactly `lanes` wide rather than sharing one `splits`; their ranges differ
(heads*dimension, the doubled intermediate, the channels) so one number cannot
make all three exact.

The superseded reasoning, kept because the arithmetic is still worth seeing:

🔴 **THE TOKEN TRANSFORMER READS ALL 566 MB OF ITS WEIGHTS ONCE PER TILE OF FOUR
TOKENS.** Twenty-four blocks of 5.9M floats
is 566 MB; the tile is 4, so a 59-token call makes fifteen passes over it, 8.5
GB. At the 114 GB/s `tools/gpu/probe-alu.js` measures for STREAMED global reads
- and 23.6 MB a block is far past any cache - that is 74 ms. The transformer
measured 73. The model holds at every length:

| tokens | tiles | measured | per tile |
|---|---|---|---|
| 59 | 15 | 67 ms | 4.5 |
| 120 | 30 | 139 | 4.6 |
| 240 | 60 | 326 | 5.4 |
| 480 | 120 | 912 | 7.6 |

The drift upward is the attention, which is quadratic; the linear term is 4.5 ms
a tile, which is 566 MB at 126 GB/s.

**The tile is capped by workgroup memory, and lifting the cap changed nothing.**
`xt` holds TILE x 768 activations - 12 KB at four tokens, 24 at eight. Chunking
the channels unties that, and it is implemented; with it, tile 8 at 240 tokens
measures 253 against tile 4's 235, and tile 4 wins at 59 tokens too (65 against
82). So the traffic the tile divides was not what the stack was waiting on.

🔴 **AND f16 WEIGHTS ARE NOT THE LEVER EITHER, WHICH IS THE MEASUREMENT THAT
SETTLES THE MODEL.** README records f16 COMPUTE being rejected (13% slower;
Apple runs f32 and f16 ALU at the same rate), and that says nothing about f16
STORAGE with f32 accumulation - a bandwidth change rather than an arithmetic
one, and the obvious move if 630 MB of resident weights were the problem. Built
(the device feature, half-width packing, and one rewrite of the finished WGSL
turning every `weights[...]` into `f32(weights[...])`) it measured **85 ms
against 65 at 59 tokens** and 232 against 234 at 240. Slower, or level. The
bytes were never the constraint; the conversions are instructions and
instructions are.

It also costs accuracy that is not free: relRMS against the f32 reference goes
1.88e-2, against a 3.02e-6 rounding envelope. That is still inside AF3's own
bfloat16 noise - eleven mantissa bits against eight - but there is no reason to
spend it for nothing.

### What else was tried on the head, and lost

- **Tiling `attention-output` over tokens, on its own.** 69 ms against 65 for
  the stack, and 70 with a split of the output range added. It only became a
  small win once every kernel took one output an invocation.
- **Lifting the token tile past four**, which is what the traffic model above
  said to do. 253 ms against 235 at 240 tokens with the channels chunked, and
  worse without.
- **Vectorising `qkvg` and `ffw-wide` over the token tile**, which takes qkvg
  from 24 instructions a channel to nine: 68 ms against 67. The same lesson the
  trunk's transition taught - these kernels wait on their weight reads, not on
  their arithmetic. It is kept because the code is simpler, not because it is
  faster.

## Fixed: the side chains were compressed, and the loader was reading four wrong tensors

Reported from the page: side chains badly placed and rings wrong, at any number
of steps. `tools/gpu/probe-sidechains.js` measured it against the reference
conformers' own rigid tables, and against AF3's own 200-step sample of the same
59-mer:

|                    | bond ratio | 1-3 ratio | PHE ring bonds |
|--------------------|-----------|-----------|----------------|
| AF3 itself         | 1.017     | 1.015     | 1.407 1.404 1.404 1.405 1.409 1.408 |
| this port, before  | 0.927     | 0.908     | 1.122 1.099 1.287 1.198 1.164 1.303 |
| this port, after   | 1.015     | 1.017     | 1.407 1.403 1.407 1.402 1.407 1.401 |

**The bug.** `diffusionWeights` loaded the diffusion atom encoder's four pair
tensors under their UNSUFFIXED names. The checkpoint has each of them twice, at
identical shapes: the unsuffixed set belongs to the pair conditioning computed
over a token's own 24 dense atom slots (AF3 captures it as `[tokens, 24, 24,
16]`), and the `_1` set to the queries-keys layout the atom transformer actually
works in (`[subsets, 32, 128, 16]`). Loading the wrong four threw nothing,
changed no shape, and folded a plausible protein - with every side chain about
8% short. `targetFeatureWeights`, ten lines above in the same file, carries a
comment warning about exactly this trap.

**Why nothing caught it.** The only checker that reaches the whole head,
`tools/oracle/check_af3_denoiser.js`, builds its weight dict BY HAND rather than
through the loader - so it scored 6.8e-6 against AF3 the whole time the shipped
pipeline was wrong. Everything downstream compared the GPU against our own CPU
reference, which was fed the same hand-built weights. A checker that does not go
through the loader does not check the loader.

**How it was found**, in the order the possibilities died:

1. `tools/gpu/probe-af3-trunk-sample.js` substituted AF3's OWN trunk into
   `foldBatch`'s `reuse` path. The side chains stayed at 0.921, which
   exonerated the trunk and its 3.7e-2 pair disagreement.
2. `tools/gpu/probe-head-cpu-vs-gpu.js` ran one denoising step both ways on
   AF3's own 59-token batch: 5e-7. Not the shaders either - the CPU reference
   and the GPU shared the error.
3. A 20-step oracle dump WITH the head's arguments captured
   (`--capture-args 'diffusion_head/__call__$'`) gave AF3's own answer at every
   rung of the schedule. `tools/gpu/probe-head-vs-af3-steps.js` asked ours the
   identical twenty questions and got 2e-2 to 6e-2 at EVERY level - which said
   the divergence was the molecule, not the noise level, and killed the
   hypothesis that the EDM preconditioning was wrong at low sigma.
4. `tools/gpu/probe-head-stages-vs-af3.js` then ran the same comparison on the
   TWELVE-mer, where the head is supposed to be exact, and got 0.102. That is
   the moment it stopped being about the molecule: the same dump, the same
   reference code, two weight dicts.

After the fix the head reproduces AF3 to about 1e-6 at all twenty noise levels
from 4608 A down to 0.03, and the bond ratios agree to three decimals.

🔴 **AND EVERY SAMPLER MEASUREMENT PREDATED THE FIX.** The sigma0 sweep, the
ligand-flow knee at sigma_data, the flow-versus-diffusion step counts and the
160 A default were all measured against a denoiser that was 3-6% wrong at every
noise level. Two have been re-run and their docstrings now carry the new
tables; what changed is worth reading, because it is a lesson about what a
sampler sweep can and cannot tell you:

- **The rankings did not move.** sigma0 still trades a ligand's bond lengths
  against a protein's backbone in the same direction, with the knee still at
  sigma_data. A sweep that ranks settings is nearly blind to whether the model
  underneath is the right one.
- **The magnitudes did.** On 1QYS the 160 A default cost 0.043 A before and
  0.025 A after, and the seed ranges that used to be disjoint now overlap. On
  HEM at sixteen steps AF3's own top of schedule got WORSE, 0.065 to 0.168,
  while 160 A improved to 0.047 - so the gap the default exists to close is
  wider against the correct weights, not narrower.
- **pLDDT moved most of all**, 6MRR to 85.8 and 1QYS to 79.8 from the high
  fifties, which is the clearest single sign that the weights were wrong: the
  confidence head was reading a structure the trunk did not predict.

Still to re-run: the flow-versus-diffusion step counts, and the ligand-only
HEM-forms-by-8-steps observation the default was chosen from.

## Templates, and where the slots are placed

🔴 **AND foldAf3 PLACES THE SLOTS, BECAUSE ONLY THE FEATURISER KNOWS THE TOKEN
LAYOUT.** A slot is indexed by TOKEN; a modified residue is one token PER ATOM
and a ligand is a chain of its own, so a chain's first token is not the sum of
the preceding chains' residue counts. Callers hand over TEXT and a chain index,
and `batch.chainOfResidue` / `batch.residueOfToken` do the placing. The earlier
offset version had a matching bug:

🔴 **A SLOT BUILT BEFORE THE BINDER'S LENGTH IS KNOWN IS BUILT AT THE WRONG
OFFSET.** Protein Hunter draws its designed chain inside the loop, so
`chains[0].length` is 0 when the templates are fetched - which made the complex
53 tokens instead of 69 and put the target's template across the binder. It
folded, and it moved ipTM from 0.324 to **0.533**, which looks like a template
working well. The binder's length is passed in now.

🔴 **AND THE TEMPLATE EMBEDDER IS NO LONGER UNCHECKABLE.**
`tools/oracle/dump_af3_trunk.py --template <pdb>[:CHAIN]` folds a query with a
real structure as its template and captures the module's inputs and its
per-slot outputs. That answers the objection at the top of
`src/af3/template-reference.js` - "with no template the six geometry features
are identically zero, so nothing here can tell a correct implementation of them
from a wrong one" - which was true and is the reason only the empty-slot path
exists. See docs/AF3.md's template entry for the numbers.

It writes the template's mmCIF from the PDB's OWN ATOM NAMES rather than
through atom37. ColabDesign2's `_mmcif_for` does the same job but reaches AF2's
`residue_constants`, which imports `dm-tree` - not installed here, and not
worth installing to copy 37 strings that every PDB line already spells out.

## A second set of weights: the dialect

🔴 **THE MODEL IS OpenBind-0, AND THE NUMBER IS PART OF THE NAME.** Upstream's
announcement (openbind.uk, 2026-08-21) calls it that and points at
`aqlaboratory/openfold-3` releases/tag/v0.5.0, which is the release ported
here. Their registry's bare `openbind` is a name a LATER release would answer
to as well - which is exactly how `openfold3` came to mean two models with
different forward conventions and sent this port reading notes about the wrong
one. So the family, the dialect, the bundle and the manifest's `model.name` all
say `openbind0`, and `dialectFor("openbind1")` RAISES rather than resolving.
`openbind` stays as an alias in two places only - `DIALECT_ALIASES`, because
upstream publishes the blob under that name, and `MODEL_ALIASES`, for `?model=`.

🔴 **OPENBIND IS NOT OPENFOLD3, AND READING THE OF3 PORTING NOTES WHOLESALE GETS
TWO THINGS BACKWARDS.** `../alphafold3/OF3_AF3_PORTING_NOTES.md` describes
OpenFold3 **preview-2** (`of3-p2-155k.pt`). OpenBind is OpenFold3 **v0.5.0**, a
separate model in that checkout's `model_config.MODELS`, and it moved TOWARD
AlphaFold 3 in exactly the two places that would have cost the most here:

- **the column attention's pair bias.** Preview-2 computes `Linear(z[k, q])`;
  AF3's Algorithm 15 says `Linear(z[q, k])`. Upstream's list is
  `TRANSPOSED_COLUMN_PAIR_BIAS` and **openbind is deliberately not in it**, so
  `swapTransposedBias` stays **false** - the same value stock AF3 uses. The flag
  already in `fold.js` reads "stock AF3 is false, the openfold3 lineage true",
  which is about the OTHER release; taking it as "the non-AF3 dialect" would
  transpose the bias in the pairformer, the MSA stack, the template embedder and
  the confidence head against weights that do not want it.
- **the diffusion transformer's pair LayerNorm**, which preview-2 runs per block
  and v0.5.0 runs once for the whole stack, as AF3 does. Their release note:
  "Moved the pair layer norm in the diffusion transformer out of attention pair
  bias. The pair layer norm is run once to match the AlphaFold3 SI."

🔴 **SO THE RUNTIME PORT IS THREE BRANCHES, NOT THIRTY.** Everything else in
those notes is absorbed by the weight converter, because it is a row permutation
of a weight matrix: the residue-alphabet permutation, the i/j crossing between
AF3's two pair-embedding sites, the SwiGLU gate/value concatenation, and the
element index shift - `one_hot(e - 1) @ W` is exactly
`one_hot(e) @ W[max(0, arange - 1)]`, which `converters/common.py` proves to
max|d| = 0. `src/af3/dialect.js` is the table:

| flag | what changes | where |
|---|---|---|
| `symmetriseBonds` | the token bond matrix sets `[j][i]` too | `featurise.js` |
| `maskPaddedKeys` | `offsets_valid &= keys_mask` in the atom cross-attention | `atom-encoder-{reference,webgpu}.js` |
| `padSingleCondUnknownDna` | the diffusion single conditioning is 833 channels, not 831 | `diffusion-{reference,conditioning-webgpu}.js` |

🔴 **AND THE BUNDLE NAMES ITS OWN GRAPH, so a caller cannot pair them wrongly.**
`af3Dialect(store)` reads `manifest.model.name` - which `export_af3_model.py`
has always written - and `trunkWeights`, `confidenceWeights`, `diffusionWeights`
and `targetFeatureWeights` each stamp it onto what they return. An unnamed
bundle RAISES rather than defaulting to stock: a ported checkpoint folded
through AF3's branches returns a structure, which is the failure this exists to
prevent.

🔴 **A ZERO COLUMN IS FREE BEFORE A LINEAR AND IS NOT FREE BEFORE A LAYERNORM.**
That is the whole of the third flag. OpenFold3's restype and profile blocks
carry 32 classes to AF3's 31, and everywhere else the extra class is dropped
from the converted weights because a column that is always zero contributes
nothing to a matrix multiply. The diffusion single conditioning LayerNorms its
concatenation, so a zero input maps to `-mean/std` AND the width becomes 833:
upstream measures dropping the two columns at 2.2e-3 against 3.4e-7.

🔴 **AND THE GPU FIX IS A SENTINEL, NOT A NINTH BINDING.** `maskPaddedKeys` is a
boolean AND in the CPU reference. On the GPU the keys' reference space is
already in the gathers buffer, real space uids are counters and never negative,
so writing **-1** into a padded key makes the equality test fail on its own -
which is `(q == k) && keys_mask` exactly, with no shader change and no extra
storage buffer. The QUERIES stay at zero: upstream gates on `keys_mask` alone,
and masking both would be a third model. The comment beside it that says a
sentinel "is the tidier choice and a different model; it cost 3.1e-2" is about
the STOCK dialect and is still true there.

🔴 **BOTH ARE CHECKED BY SWEEPING THE DIALECT AS AN AXIS, WITH A DISCRIMINATING
CONTROL.** Two arms agreeing with their reference says the GPU matches the CPU;
it does not say the flag reached either. `check-af3-atom-encoder.js` and
`check-af3-diffusion-conditioning.js` both fail if the openbind arm does not
DIFFER from the stock one. Measured:

| | openbind vs alphafold3 | each arm vs its reference |
|---|---|---|
| atom encoder `pairCond` | **7.77e-2** | 2.48e-7 |
| ...`tokenAct` | 2.96e-6 | 9.48e-6 |
| ...`skipConnection` | 3.72e-6 | 2.05e-5 |
| single conditioning | **1.54e-3** (2370x) | 6.48e-7 |

🔴 **AND THE ENCODER'S OUTPUT BARELY MOVES, WHICH IS NOT A BUG.** The atom pair
conditioning moves by 7.8e-2 because that is where a padded key's offset term
lived; almost none of it reaches `tokenAct`, because the attention masks those
same keys anyway. What leaks is the mask bias being a large FINITE negative
rather than -infinity, so a padded key keeps a softmax weight of about 1e-6. So
the control is "some output moved past its envelope", not "every one did" -
demanding all three fails on a correct implementation.

🔴 **AND THE STOCK PATH IS BIT-IDENTICAL, WHICH IS THE OTHER HALF OF THE GATE.**
`tools/gpu/fold.js` before and after the dialect work: PDB SHA
`aef231158a174daf` both ways, mean pLDDT 86.13324126798517 and pTM
0.7378317753181738 to every digit. Per-stage: target_feat 7.98e-8, denoiser
1.28e-4, atom encoder 9.48e-6 / 2.05e-5 - every one of them the figure already
recorded in this file.

🔴 **AND OpenBind FOLDS, WITH ITS OWN NUMBERS.** `openbind.bin.zst` is
published already converted, `read_blob` in `tools/export_af3_model.py` now
reads that format directly - no torch, no 2.3 GB checkpoint, no second checkout
- and the two existing tools do the rest:

```
python3 tools/export_af3_model.py --model openbind --include diffuser \
  --out model-openbind-full-f32           # 406 tensors, 368.4 M, 1405 MiB
python3 tools/quantize_af3.py --source model-openbind-full-f32 \
  --out model-openbind-int5               # 264.6 MiB, 5.31x
node tools/gpu-chrome.mjs tools/gpu/fold.js --model=/model-openbind-int5/manifest.json
```

Scored against `tools/fixtures/6mrr-crystal.pdb`, both bundles at int5:

| | RMSD | TM | pLDDT | CA-CA |
|---|---|---|---|---|
| alphafold3 | **0.67 A** | **0.952** | 85.8 | 3.82 |
| openbind | 1.99 | 0.833 | 79.4 | 3.78 |
| openbind, `swapTransposedBias` forced TRUE | 2.03 | 0.820 | 76.7 | 3.78 |

The AF3 row reproduces this file's own quantisation table (0.66 / 0.953), which
is what says the harness is sound. **The third row is the experiment that
matters**: it turns on the convention OpenFold3 preview-2 was trained with, and
it is WORSE on every measure - so `swapTransposedBias: false` is confirmed
against the weights themselves rather than against a reading of somebody else's
table. One target, so read it as a direction and not a margin.

🔴 **AND THE SHAPES CONFIRM THE WHOLE ANALYSIS INDEPENDENTLY.** `openbind.shapes.json`
against the AF3 bundle: **406 arrays each, identical names, exactly two shape
differences** - `single_cond_initial_norm/scale` 833 against 831 and
`single_cond_initial_projection/weights` [833, 384] against [831, 384]. Nothing
else. Preview-2's per-block diffusion pair LayerNorm would have added
twenty-four tensors and does not appear, which is the tree agreeing with the
release note.

🔴 **`--ablate` AND `--enable` PRICE A DIALECT BRANCH, because a branch that is
silent when wrong cannot be trusted on a reading.** `tools/gpu/fold.js
--ablate=maskPaddedKeys` turns one off and `--enable=swapTransposedBias` turns
one on. `padSingleCondUnknownDna` cannot be ablated: it changes the LayerNorm's
WIDTH, so the bundle's 833-long scale stops matching and the conditioning
throws - the structural gate doing its job, and the reason that branch needs no
measurement.

🔴 **A PADDED KEY ONLY EXISTS BELOW 128 ATOMS, AND EVEN THERE IT REACHES
NOTHING.** `featurise.js` clamps the 128-wide key window against the REAL atom
count, so at 128 atoms or more every key lands on a real atom and the key mask
is identically one. `tools/gpu/probe-ablate.js` sweeps the boundary:

| residues | atoms | padded keys | `pairCond` | `tokenAct` | control |
|---|---|---|---|---|---|
| 4 | 32 | 288 of 384 | **4.13e-1** | **0.00e+0** | 1.04e-6 |
| 8 | 67 | 366 of 768 | 2.72e-1 | 0.00e+0 | |
| 12 | 106 | 198 of 1152 | 1.37e-1 | 0.00e+0 | 1.03e-6 |
| 16 | 143 | **0** of 1536 | 0.00e+0 | 0.00e+0 | |
| 68 | 574 | **0** of 6528 | 0.00e+0 | 0.00e+0 | 1.00e-6 |

So the branch moves the atom PAIR conditioning by up to 41% and the encoder's
OUTPUT by exactly nothing. The control column is a 1e-6 nudge to the
conditioning, and it is there because "relRMS 0.00e+0" is also what a broken
comparison says.

🔴 **WHICH IS WHY AN ABLATION ON 6MRR CAME BACK BIT-IDENTICAL.** Same PDB, same
pLDDT to every digit, with `maskPaddedKeys` off - because 6MRR has 574 atoms and
therefore no padded keys at all. The dumps differ on this and it decides what a
checker can see: `af3-6mrr.json` has 6528 of 6528 keys live, while
`af3-oracle-atom-f32.json` has 873 of 1152 - a 97-atom molecule, under the
window - which is why `check-af3-atom-encoder.js` measures a 7.77e-2 separation
and a whole fold measures none.

🔴 **AND THE PADDING IS GONE, BECAUSE NOTHING HERE NEEDED IT.** AF3 pads because
JAX wants static shapes; these kernels are generated per shape and size their
workgroup storage from `keys`, so the window is `min(128, atomCount)` now and
**no batch this featuriser produces has a padded key at any size**. The
`ref_space_uid = 0` collision that OpenFold3 trained around cannot occur here.

Measured before and after, `tools/gpu/fold.js --sequence=`:

| | 68 residues, 574 atoms | 12 residues, 106 atoms |
|---|---|---|
| diffusion 50 | **bit-identical** (`1f3a312051898379`) | 84.3979 -> 84.4696 pLDDT |
| flow 16 | **bit-identical** (`8cd298eb6bd61f82`) | 84.7656 -> 84.8080 |

Exactly the scope the table above predicts: at or above 128 atoms the window was
already inside the molecule and nothing moves, and below it the padded keys stop
contributing. Geometry is unchanged either way (N-CA 1.458, CA-CA 3.807 on the
12-mer).

🔴 **AND `maskPaddedKeys` IS NOT DEAD, WHICH IS WHY IT STAYS.** The differential
checkers do not use this featuriser - they feed AF3's OWN gathers out of an
oracle dump, and `af3-oracle-atom-f32.json` is a 97-atom molecule with 279
padded keys of 1152. So `check-af3-atom-encoder.js` still separates the two
dialects by 7.77e-2 on `pairCond`, and the flag still decides what a bundle
converted for OpenBind computes on somebody else's batch. What changed is that
the SHIPPING path can no longer reach the case.

🔴 **A SECOND MODEL IS MISTAKEN FOR THE FIRST IN A CACHE, NOT IN A LOADER.**
Two bundles now build AF3's graph, and every memo keyed on something that does
not distinguish them is a silent wrong answer. Two were found, one of them the
hard way:

* `loadAf3Weights` memoised ONE promise, so the second family's fold got the
  first family's weights. Caught while writing it - `weightsPromises` is a Map
  keyed by family.
* **`trunkKey` did not include the family.** The cached trunk is a pair and a
  single representation, and those have the same shapes whichever parameters
  produced them - so folding with OpenBind and then AlphaFold 3 on the same
  sequence matched every other field and handed AF3's diffusion head OpenBind's
  trunk. Reproduced in the page at 32 residues: **pLDDT 41.5 with the status
  line reading "trunk reused", against 83.3 once `family` was in the key.**
  Nothing errors, nothing warns; the chain comes apart, which is how it was
  reported ("atoms are no longer attached").

Neither is findable by folding one model. The reproduction is
openbind -> af3 -> openbind in one page session, which
`test/model-family.test.js` pins by asserting the key names the family.

🔴 **AND "IS THIS AF3" IS NOT `family === "af3"` ANY MORE.** That comparison sat
in five places and every one meant "is this the AF3 pipeline", not "is this
DeepMind's checkpoint" - so a second AF3-graph family took the AlphaFold 2
branch at each. Three of them were CAPABILITY guards, and they refused ligands,
modified residues and nucleic chains under OpenBind with a message naming a
capability the model has: *"Ligands need AlphaFold 3; the model is set to
openbind"*. `AF3_FAMILIES` in `src/reference/manifests/index.js` is the list;
`isAf3Family` is the test.

🔴 **AND THE DIALOG SAYS "NOT AVAILABLE FOR COMMERCIAL USE" AND NOT "ACADEMIC
USE ONLY".** The second is the phrase that comes to hand and it is wrong twice:
DeepMind's terms cover non-profits, research institutes, journalism and
government bodies as well as universities, and they exclude a researcher
employed by a commercial organisation. The short form has to stay TRUE while
being short; the linked terms carry the detail. "Not open source" was the
earlier version of the same mistake - the CODE is openly licensed and it is the
PARAMETERS that are restricted.

🔴 **THE LICENCE DIALOG ASKS THE PERSON FOLDING, NOT THE DEPLOYER.**
`build_site.py` already refuses to publish DeepMind's parameters without
`LOCALFOLD_ACCEPT_MODEL_TERMS`; the page's `#model-terms` dialog gates the first
AF3 fold and remembers the answer in `localStorage`. It offers OpenBind as the
alternative rather than only an "I agree", because a dialog with one button
teaches people to click it. `tools/model-terms.py` is the check - it drives the
real page and asserts the dialog opens, remembers, switches the model row, and
**still opens for `?model=af3`**, since a URL must not be able to accept
somebody else's terms.

🔴 **AND A `<dialog>` IS NEVER LAID OUT UNTIL IT IS OPENED**, which is the load
dial's blind spot again: `tools/mobile-layout.py` cannot see it. `model-terms.py`
forces it open under a 320px device override and asserts it is neither clipped
nor sideways-scrolling and that its two buttons stack (measured 298px wide,
buttons 260px, left 11px).

🔴 **AND `?model=` HAD TWO READERS, WHICH IS WORSE THAN BEING IGNORED.**
`applyModelFromUrl` takes it as the model row's value; `web/model.js` took it as
a manifest URL whenever the family was monomer, which is how a page is pointed
at weights somewhere else. `?model=monomer` is the one spelling that reaches
both - it selected AlphaFold 2 and then fetched `<origin>/monomer`, so the live
page said **"failed to load model manifest: 404"** for a model that folds
perfectly well from the dropdown, at pLDDT 96.5. The override now requires a
PATH - a slash, or a `.json` ending - because a path is what it was for; a bare
family name belongs to the other reader.

🔴 **AND IT WAS FOUND BY MISTAKING IT FOR SOMETHING ELSE.** The 404 appeared
minutes after 129.8 MiB of `af2-monomer/*.js` were deleted from Hugging Face, on
the one bundle those files belonged to, which is as convincing a coincidence as
this repository has produced. The deletion was innocent - `ScriptTensorStore`
reads `manifest.js` only under `file://`, and from the bundle's own directory,
never from a remote - and folding monomer from the dropdown proved it. **A
regression that appears next to a change is not evidence it came from it.**

🔴 **AND A `?model=` THAT IS IGNORED LOOKS EXACTLY LIKE ONE THAT WORKED.** The
complaint about an unknown name was written twice before it was visible: once
before the vendored viewer's own "Ready." line overwrote it, and once before the
parameter had even been read, because it was called beside the Fold button's
enabling - which runs EARLIER in the file. It waits for that specific string
now. `of3` is deliberately not an alias for `openbind`.

## The trunk at 512 tokens, which nothing had profiled

The AF3 priors were fitted at 200 tokens. AF2's were fitted at 400 and one of
them moved 2% when re-swept at 825 (docs/AF2.md), so the same question is worth
asking here. `bench-trunk.js --model=/model-af3-int5/manifest.json --tokens=512
--msa=128 --passes=2 --profile`, steady pass:

| stage | ms |
|---|---:|
| pairformer | 1896 |
| msa-stack | 628 |
| embedder | 514 |
| template | 409 |
| distogram | 213 |
| **whole** | **3724** |

🔴 **AND THE PAIRFORMER IS GPU-BOUND HERE, WHICH IS WORTH KNOWING BEFORE
OPTIMISING ANYTHING ELSE.** `pairformerSplit` reports encode **22.8 ms**, wait
**1644.7**, release 0 - the host finishes encoding in a fortieth of the time the
GPU takes, so nothing on the host side of this stage is worth moving.

The GPU passes, 1374 ms over 2048 dispatches:

| pass | ms | share | groups a pass |
|---|---:|---:|---:|
| `grid.attend` | 397.1 | 30.6% | 16384 |
| `tri.contract` | 167.3 | 12.9% | 5837 |
| `tri.project` | 107.9 | 8.3% | 11878 |
| `pair-transition.wide` | 94.3 | 7.3% | **2048** |
| `grid.project` | 94.0 | 7.2% | 15974 |
| `pair-transition.down` | 77.5 | 6.0% | **256** |
| `tri.project-out` | 66.6 | 5.1% | 6656 |

🔴 **`pair-transition.down` LAUNCHES 256 WORKGROUPS ON A CARD THAT FITS 4542.**
That is 6% of the trunk's GPU time in a kernel using about a twentieth of the
device, and it is the clearest starved dispatch anywhere in this port -
`probe-occupancy.js` is what makes it legible as one rather than as a number.
`pair-transition.wide` at 2048 is short of the same mark by half.

It is NOT a knob. `pairTransitionChunkBytes` at 64, 128 and 256 MiB leaves the
group count at exactly 256 and the pass at 77.5 / 77.48 / 77.87 ms - the
transition is not chunked at this shape, so the chunk target never binds and the
count comes from the split kernel's own tiling. Fixing it means changing that
tiling, which is kernel work and wants its own differential.

The ceiling is worth stating before anyone starts: eliminating the pass entirely
is 2% of the trunk, because `grid.attend` is 30% of the GPU time and is already
on the matrix units with 16,384 workgroups a pass - well fed, and the reason the
trunk looks the way it does.

### 🔴 And `pairTransitionChunkBytes` never reached this track at all

Chasing that starved dispatch found the reason a knob could not move it.
`pairformer-block-webgpu.js` and `msa-stack-webgpu.js` both put
`pairTransitionChunkBytes` into the options they hand `encodePairTrack`, and
`pair-track-gpu.js` - the only caller of `transitionSplitChunkRows` - never read
it, so the rule fell back to its own 64 MiB default. **Every AF3 and OpenDDE arm
ever measured with that knob was measured at 64 MiB**, which is why the first
sweep of 64, 128 and 256 moved the group count not at all and the pass by 0.4 ms
of 77.5. The same shape as `matrixLinear: false` falling through into the matrix
path: a knob with no off position, and a knob with no effect, are both worse than
no knob.

Wired through, it does what it says and still does not pay:

| chunk | down | groups | wide | groups | whole trunk | peak |
|---|---:|---:|---:|---:|---:|---:|
| 64 MiB | 77.12 ms | 256 | 94.08 | 2048 | 3717 ms | 1284 MiB |
| 256 | 66.72 | 1024 | 121.95 | 8192 | 3671 | 1500 |
| 512 | **63.04** | 2048 | **120.52** | 16384 | 3741 | 1788 |

The starved pass gets its workgroups - 256 to 2048, and 1.22x - and the `wide`
pass loses 27 ms, more than `down` gains. `wide` was never starved at 2048
groups, so a bigger chunk buys it nothing and costs it locality: the widened
buffer it streams grows with the chunk. The trunk is 3717 / 3671 / 3741, a wash,
for up to 504 MiB of extra peak.

So the default stays at 64 MiB and the fix is the plumbing, not the value. It
matches what docs/PERF.md already records for the ESMFold2 trunk - "a 6% GPU win
and a 1% WALL loss for 144 MiB" - reached there by a different route.

## The single projection was running an M2's constant, and it is 4x

`single.project` is one workgroup per token and per split. At 68 tokens that is
136 workgroups on a card that holds 4542, and it was **15.0 ms of a 104.3 ms
trunk** - more than any kernel in the model except `grid.attend`, for a track
that is O(n) where the pair track is O(n²).

Two knobs fix it and **neither works alone**, which is why two earlier attempts
found nothing. `bench-trunk.js --profile`, reading `gpuTotalMs` (every pass, not
the listed rows), three rounds an arm, arms interleaved:

| n=68 | trunk GPU | `single.project` | groups |
|---|---:|---:|---:|
| shipped (target 110, 64 lanes) | 104.3 / 104.3 / 104.3 | 15.00 | 136 |
| target 2048 alone | 96.9 / 96.8 / 96.8 | 7.48 | 204 |
| **target 2048 + 128 lanes** | **93.0 / 92.8 / 93.1** | **3.69** | 204 |

| n=300 | trunk GPU | `single.project` | groups |
|---|---:|---:|---:|
| shipped | 607.8 / 609.3 / 608.4 | 15.67 | 300 |
| target 2048 alone | 601.1 / 601.3 / 601.3 | 7.86 | 900 |
| **target 2048 + 128 lanes** | **599.0 / 597.7 / 596.9** | **4.99** | 900 |

Across four sizes, whole-trunk GPU: **68 tokens −10.8%, 150 −6.0%, 300 −1.7%,
512 −0.5%**, and the kernel itself 4.05x, 3.97x, 3.13x, 2.22x. The win is
largest at the SHORT chains, which is what a page mostly folds, because the
starvation is: fewer tokens, fewer workgroups, same fixed cost.

🔴 **WHY NEITHER KNOB PAYS ALONE, AND BOTH TOGETHER DO.** 128 lanes over the
unsplit 384-wide output leaves each lane three outputs deep and buys nothing -
measured at 15.48 -> **16.24**, worse, which reproduces docs' earlier
"`project` at 128 lanes is WORSE (15.60 -> 16.09)" exactly. Split three ways the
output is 128 wide, one lane an output, and the two compose. The earlier split
sweep used `maxSplits` 6, whose `perSplit` is 64 and which therefore cannot use
a wider lane at all. **Sweeping one axis of a pair says the pair does not pay.**

🔴 **AND A LANE WIDTH IS A REQUEST, NOT A VALUE.** `perSplit` is `width /
splits` and the split count is derived from the token count, so 128 divides it
at one length and not at another: at AF3's 384 the rule picks 3 splits for every
n below 1024 - `perSplit` 128 - and **2 from 1024 to 2047, where `perSplit` is
192**. A prior naming 128 outright would have folded every chain up to a
thousand tokens and thrown on the next one. `createSingleAttentionShaders`
halves the request until it divides, landing on the 64 every caller had before
the knob existed; and the pipeline key names the RESOLVED width, not the
request, because that is what the shader contains.

🔴 **IT IS A REORDERING.** Three workgroups normalise the row where one did and
128 lanes reduce it in a different tree, so `check-af3-block-any`'s single
residual moves in the eighth figure (2.0899e-4 either way) and its pair not at
all. On a fold: **max |dx| 0.001 A, 531 of 574 atoms identical** - the same
class as `attnSplits` 1 -> 4, which the ampere prior already ships at 541/574.
pLDDT 84.26493204096884 against 84.26493426067073.

**ampere only.** `singleProjectSplits`' own table shows 6 splits LOSING on an M2
at every n it was measured at, which is why these were left as parameters in the
first place - see DEFAULTS_ARE_MEASUREMENTS.

🔴 **AND EVERY NUMBER ABOVE IS A TRUNK NUMBER, NOT A FOLD NUMBER.** The trunk's
GPU time is about 104 ms of a 1.18 s warm AF3 fold at 68 tokens - 9% - so 11 ms
off it is **1%**, and that is what a whole fold shows:

| | first | warm |
|---|---:|---:|
| AF3 68 tokens, new | 2.159 / 2.153 / 2.177 s | **1.166 / 1.159 / 1.219** |
| AF3 68 tokens, old | 2.142 / 2.156 s | 1.184 / 1.169 |
| OpenDDE 6mrr, new | | **2423 / 2420 ms** |
| OpenDDE 6mrr, old | | 2432 / 2493 ms |

Consistently in the right direction and consistently about 1%, with the first
fold unmoved because it is compilation and weights. Take the change - it is
free, and it is a kernel running at four times its old rate - but do not quote
the trunk figure as a fold figure. This is the distinction docs/PERF.md keeps
making about `--profile`: a share of a stage is not a share of a wall.

## The pair-logits cache was fitted at 200 tokens, and covers a quarter at 400

`PAIR_LOGITS_CACHE_BYTES` is 64 MiB and its own note says the cache is
`64 x tokens^2` bytes a block - so at the 200 tokens it was measured at it
covers **all twenty-four blocks**, and at 400 it covers **six**. That is the
same shape as AF2's `TRANSITION_CHUNK_TARGET_BYTES` (docs/AF2.md): a byte cap
fitted where it happened to cover the whole workload, found by parsing every
`export const` in `src/` for a comment that cites only a short length.

`pairLogitsCacheBytes` is the knob now, null taking the constant.
`bench-head.js --tokens=400 --calls=9 --model=/model-af3-int5/manifest.json`,
median of the eight steady calls, two rounds:

| | round 1 | round 2 |
|---|---:|---:|
| 64 MiB (the constant) | 51.5 ms | 51.5 |
| 128 MiB | 50.0 | 50.5 |
| **256 MiB** | **47.0** | **47.5** |
| 512 MiB | 47.0 | 47.0 |

**8.7% of a denoiser call**, and 256 is the knee because 24 blocks of 10.24 MiB
is 246 MiB - 512 buys nothing because it caches the same twenty-four.

🔴 **AND AT THE PAGE'S DEFAULT IT IS WORTH ABOUT 1.5%, WHICH IS THE HONEST
NUMBER.** `AF3_COUNTS.flow` prefers **16** steps, so the sampler is about 0.8 s
of a 4.5 s fold and a whole fold reads **4.5 s on both arms** - the saving is
under the resolution of that clock. It is the diffusion mode, whose dial reaches
**200** steps because that is what AF3 was trained with, where 8.7% of a call
becomes about 0.9 s. Peak goes **1457.7 -> 1623.8 MiB** at 400 tokens.

Set in the **ampere prior only**, beside `opmPairBlockBytes` and
`transitionChunkBytes`, which is now three 256 MiB caps on a 40 GB card.

**Bit-exact, and checked with the right instrument.** A cache of a recomputed
value cannot change an answer unless it goes stale, and
`tools/diff-fold-coords.py --b='--tune-json={"pairLogitsCacheBytes":67108864}'`
says so: **574/574 atoms identical, max |dx| 0.000000 A**. 🔴 The first attempt
to check it compared two PDB captures that were both EMPTY - `fold.js --folds=1`
prints JSON and the grep matched nothing - and reported "STRUCTURE IDENTICAL",
which is this repository's "a gate that cannot fail is not a gate" in one line.

## The outer product mean's block was fitted at 150 tokens and inverts at 300

`OPM_BLOCK_I` is 2 and its note reasons from "the workgroup count still wins at
these sizes", measured at 59 and 150 tokens. At 400 tokens `opm.contract` is the
**second-largest kernel in the trunk** - 245.9 ms against `grid.attend`'s 259.4 -
and the block that wins there is the one that note measured as worst.
`bench-trunk.js --profile --model=/model-af3-int5/manifest.json --msa=512`:

| tokens | 59 | 150 | 256 | 300 | 350 | 400 |
|---|---:|---:|---:|---:|---:|---:|
| blockI 1 | **3.67** | 19.21 | 53.51 | **75.67** | **108.61** | **147.76** |
| blockI 2 | 5.53 | **14.41** | **38.81** | 111.52 | 178.05 | 245.69 |

Two wins only in a band, and **the cliff between 256 and 300 is sharp and
reproducible** - 256 reads 38.81/39.63 for two against 53.51/53.98 for one, and
300 reads 111.52/112.33 against 75.67/75.73. It is not a depth effect: at 1024
rows the ordering is identical (59: 6.92 against 11.06; 150: 39.01 against
33.15; 256: 121.60 against 84.26; 400: 325.20 against 499.72). `blockI 2` goes
superlinear past 256 where `blockI 1` stays smooth in the pair count; the
mechanism is not established and the L2 working set does not explain it, since
256 tokens at 1024 rows already exceeds L2 and two still wins there.

`opmCellChunk` swept beside it stays at 256: 128 is 247.8, 256 is 147.9, 512 is
150.7, 1024 is 173.2. Only the block moves.

**A PRIOR, one-sided and conservative** - `opmBlockITokens`, above which the
stack takes 1. 🔴 It shipped for one commit as a DERIVATION applying to every
device, which was wrong: the five derivations this repository has all read
something the device reports - its measured width, its memory budget - and 256
tokens is not that, it is where this A100's memory system turns over. Another
part keeps the measured block of two until someone measures it there. The band where two wins is left exactly as measured, and so is
the 59-token case, where one wins by 1.9 ms. A trunk pass at 400 tokens is
**1244.4 -> 1143.6 ms, 8.1%**, and a fold at 200 tokens is untouched
(`opm.contract` 39.08 ms, the block-of-two path). Bit-exact where it fires:
`diff-fold-coords.py --sequence=<400> --b='--tune-json={"opmBlockI":2}'` gives
**3046/3046 atoms identical, max |dx| 0.000000 A**.

## A recycle criterion for a trunk that returns no structure

AF2 stops recycling on ColabFold's `compute_tol` - the RMS change of every
C-alpha pair distance - and `src/model/recycle-convergence.js` implements it.
**AF3 has no early stop at all**, and the reason is structural rather than an
oversight: `src/af3/fold.js`'s recycle loop is a bare `for (let pass = firstPass;
pass <= recycles; pass += 1)` because AF2 recycles a STRUCTURE and AF3 recycles
the single and pair representations alone, running the sampler once at the end.
There are no coordinates to compare until every recycle is already paid for.
OpenDDE runs the same fold; ESMFold2 has its own recycle path and does not.

So the only signal is the representation. `src/model/feature-convergence.js` is
the metric - relative RMS change, `||b - a|| / ||b||`, dimensionless so it can
be compared across models and token counts - and it costs nothing: the loop
already holds `previousPair` and `previousSingle` as HOST arrays, because the
pairformer reads them back each pass. No kernel, no readback, no device memory.

`recycleDeltas` reports it per pass and `--recycle-tolerance` acts on it, **0 and
off by default**. On the 68-token default input:

| pass | pair | single |
|---:|---:|---:|
| 0 | 1 (the zero seed) | 1 |
| 1 | 0.0725 | 0.0554 |
| 2 | 0.0159 | 0.0070 |
| 3 | 0.0045 | 0.0023 |

Monotone, and decaying about 4x a pass. `--recycles=3 --recycle-tolerance=0.02`
runs three passes instead of four, 2.243 s -> 2.067 s.

🔴 **AND THE OBVIOUS WAY TO CALIBRATE IT DOES NOT WORK, WHICH IS THE FINDING.**
The natural experiment is to fold at 0, 1, 2 and 3 recycles and see where the
STRUCTURE stops moving. Done that way the 68-token input reads 0.551, 0.120,
0.076 and 0 A against the deepest fold, which looks like convergence by recycle
1. It is not a measurement of the trunk. **The sampler is stochastic, so the
control is to vary the seed and hold the recycles fixed** - and superposed, at
three recycles:

| | sampler noise, same trunk | recycle 2 vs 3 |
|---|---|---|
| 68 tokens, pLDDT 87.5 | **0.133 - 0.294 A** | **0.023 A** |
| 250 tokens, pLDDT 42 | **4.580 - 12.702 A** | 3.265 A |

**At both confidence levels the recycle-to-recycle difference is at or below the
sampler's own noise**, by an order of magnitude at high confidence. A structure
RMSD cannot resolve one recycle count from another on a diffusion model, so it
cannot calibrate a criterion either, and the reading that suggested it could was
noise.

🔴 **AND THE FIRST VERSION OF THAT CONTROL WAS WRONG TOO, IN THE OTHER
DIRECTION.** Unsuperposed, the seed-to-seed numbers are 7.4 to 15.1 A at
pLDDT 87.5, which reads as chaos. Two folds of the same molecule are in
arbitrary rigid-body poses: different seeds start from different noise and there
is no canonical frame. Same-SEED comparisons share one, which is why the recycle
sweep gave small numbers without alignment and why mixing the two designs is a
trap. Kabsch first, always, unless the seed is held.

pLDDT is better behaved but not clean either - its own seed spread is 0.520 at
68 tokens and 1.146 at 250, against recycle spreads of 1.707 and 7.747. Usable
at 3-7x the noise, and not a fine instrument.

**What IS deterministic is the trunk.** The pair delta reads 0.0479 on all three
seeds of the 250-token input, to four figures, because only the sampler is
stochastic. That is the whole argument for measuring convergence on the
representation rather than on what comes out of it.

🔴 **AND THE FIRST PASS'S SEED IS NOT THE RIGHT SHAPE, WHICH THE GATE CAUGHT.**
The loop seeds `previousPair` with `tokens^2 * 128` zeros - AF3's pair width -
and OpenDDE's pair is 384 wide, so the two differ by exactly 3x and a strict
comparison raised `feature convergence over 591872 and 1775616 elements` on a
model that had folded a moment earlier. The strictness is worth keeping for the
passes that DO compare, so pass 0 is reported as 1 rather than measured: it has
no previous by construction, which the zero seed was only ever standing in for.

🔴 **SO THE CRITERION READS THE DISTOGRAM, WHICH IS THE ONE TRUNK QUANTITY IN
ANGSTROMS.** Every pass computes one already, to draw the contact map while the
trunk is still recycling, so the expectation over its 64 bins is a predicted
distance matrix for free - and the RMS change of that matrix is exactly what
ColabFold's `compute_tol` takes over a structure. AF2's criterion and this one
are the same measurement in the same unit, one from coordinates and one from
the trunk. `expectedDistances` and `distanceChange` in
src/model/feature-convergence.js; the tolerance is angstroms and ColabFold's
default for the analogous number is 0.5.

| pass | pair | single | **distogram** |
|---:|---:|---:|---:|
| 68 tokens, 1 | 0.0725 | 0.0554 | **0.1244 A** |
| 2 | 0.0159 | 0.0070 | **0.0393** |
| 3 | 0.0045 | 0.0023 | **0.0184** |
| 250 tokens, 1 | 0.0970 | 0.0373 | **0.4195** |
| 2 | 0.0705 | 0.0115 | **0.2602** |
| 3 | 0.0479 | 0.0084 | **0.2289** |

🔴 **AND IT IS BIT-IDENTICAL ACROSS SAMPLER SEEDS** - 0.4195, 0.2602 and 0.2289
on every seed of the 250-token input, to four figures, while the structures
those same folds produced differ by 4.6 to 12.7 A. That is the whole argument
for the instrument: the trunk is deterministic and only the sampler is not, so
this is the one number a stochastic structure cannot contradict.

**And it resolves the 250-token puzzle.** That input looked unconverged because
its structures moved 3.3 A between recycle counts. Its TRUNK was converging the
whole time - 0.42 to 0.26 to 0.23 A, under ColabFold's tolerance from the first
pass - and every angstrom of that structural variation was the sampler on a
pLDDT-42 input. Low confidence does not mean the trunk is still moving; it means
the sampler cannot commit, which is a different thing and the structure cannot
tell them apart.

At 0.5 A the default input runs **two passes instead of four**, 2.301 s ->
1.944 s, for a pLDDT change of 0.126 - smaller than that fold's own seed spread
of 0.520.

🔴 **AND SIX REAL SEQUENCES KILLED THE ONE-CROSSING RULE.** The two inputs above
both settle monotonically, which is exactly what made a first-crossing threshold
look sound. Folded at three recycles, the per-pass change in angstroms:

| | tokens | pLDDT | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|---:|---:|
| 1qys (Top7, de novo) | 91 | 81.64 | 1.329 | 0.715 | 1.240 |
| 6mrr | 71 | 70.20 | 0.386 | 0.122 | 0.140 |
| **GB1** | 56 | 74.13 | **0.488** | **1.092** | 0.394 |
| lysozyme C | 129 | 36.23 | 0.329 | 0.132 | 0.095 |
| ubiquitin | 76 | 90.43 | 1.684 | 2.418 | 0.336 |
| villin HP36 | 36 | 90.55 | 2.842 | 0.611 | 0.134 |

**The trunk does not settle monotonically.** GB1 dips under 0.5 A at pass 1 and
then moves **1.092 A** at pass 2; ubiquitin rises from 1.684 to 2.418 before
falling; 1qys never settles at all. A rule that stops at the first crossing
throws GB1's second pass away.

**Two consecutive passes under the tolerance is safe on all six** and still
stops 6mrr and lysozyme a pass early. It is not fitted to these numbers - one
step under a threshold is the textbook thing not to trust - but GB1 is why it is
there, and `test/feature-convergence.test.js` pins that sequence by name so the
rule cannot be simplified back. At 0.5 A the default input then runs three
passes instead of four, 2.255 -> 2.131 s, for a pLDDT change of 0.084 against
its own seed spread of 0.520.

🔴 **IT STILL SHIPS AT ZERO.** Six sequences are a corpus in the sense that they
falsified a rule, and not in the sense that they calibrate one: four of the six
never reach two consecutive passes under 0.5 A within three recycles, so what
the tolerance buys on a real workload is two folds' worth of evidence. The
saving where it does fire is one pass of four. What is established is the
instrument, its unit and the shape of the rule; the threshold is not, and a
default that silently drops a recycle should be worth more than that before it
is one. `recycleDeltas` reports all three numbers on every fold, so the corpus
can keep growing from runs people were doing anyway.

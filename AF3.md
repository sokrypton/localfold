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

A flow-8 fold is about 7 s, of which the trunk is ~1 s. It was ~150 s when the
first end-to-end fold ran.

## Running it

    python3 -m http.server 8080          # then open /index.html

    # The GPU lane. Dawn (`npm run test:gpu`) cannot load on this macOS; Chrome
    # can. Every checker and bench is a module for this harness.
    node tools/gpu-chrome.mjs tools/gpu/<module>.js [--flags]

    node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
      --mode=flow --steps=8 --recycles=1 --model=/model-af3-int5/manifest.json
    python3 tools/score_fold.py <the log> --reference 6mrr-crystal.pdb

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

## Performance, and what has already been tried

The pairformer went 3468 ms -> 621 ms over 48 blocks at 59-68 tokens, and AF3's
block is now 1.09x AF2's evoformer block - for a block with no MSA row
attention, no column attention and no outer product mean in it.

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

What did **not** work, measured, so it is not retried:

- **Uploading all 48 blocks' weights once** instead of per block: 30% *slower*
  (640 -> 830 ms), reproducibly. Recycling a few buffers beats holding 384, and
  the up-front burst serialises ahead of all compute.
- **Caching bind groups**: exactly zero, 636-639 either way, though ~1,680 are
  created per stack.
- **Binding AF2's attention kernel directly** rather than rewriting on its
  principles: it takes a uniform for its shape, folds `1/sqrt(d)` into the query
  projection and applies the gate itself. The adapter was wrong at relRMS
  2.96e-1 and cost more to find than the rewrite took.

Where the remaining time goes, at 59 tokens: 348 ms of dispatch work and 284 ms
of per-block overhead that is **not** uploads (40 ms) and **not** bind groups
(0 ms). About 5 ms a block in the encoder, submit and validation path is
unexplained. That is the next lead and it is a small one.

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
- **Per-atom conditioning is still on the CPU** - a real gap against AGENTS.md,
  now a few hundred ms rather than five seconds. It has no kernel.
- **Templates raise** rather than compute: the geometry features are
  unverifiable without a reference.
- **AF3's block is still 1.09x AF2's** for strictly less work.

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

### A separate bug, found while measuring and not yet fixed

**A fold crawls in a background tab.** The per-block yield is
`await new Promise((resolve) => setTimeout(resolve, 0))`, and Chrome clamps
setTimeout to >=1 s in a hidden tab - so a 48-block pass that takes under a
second takes the better part of a minute and the trunk appears to hang.
Measured: pass 1 reached block 11 in five minutes hidden, then jumped to block
28 the moment the tab was touched. A MessageChannel yield is not throttled.

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

## Open: the side chains are compressed, and it is ours

Reported from the page: side chains badly placed and rings wrong, at any
number of steps. Measured by `tools/gpu/probe-sidechains.js` against the
reference conformers' own rigid tables, and then against AF3's own 200-step
sample of the same 59-mer:

|              | bond ratio | 1-3 ratio | PHE50 ring bonds |
|--------------|-----------|-----------|------------------|
| AF3 itself   | 1.017     | 1.015     | 1.407 1.404 1.404 1.405 1.409 1.408 |
| this port    | 0.927     | 0.908     | 1.122 1.099 1.287 1.198 1.164 1.303 |

AF3 gives a textbook benzene ring; this port gives one about 8% compressed
and 18% irregular *within a single ring*. Not a scale factor, and not
under-convergence - 160 diffusion steps score 0.226 mean error against
flow-8's 0.206. Glycine (backbone alone) is nearly right at 0.078 and the
error grows with distance from the backbone.

**Already ruled out.** Not atom labelling: the probe matches pairs by NAME
out of the batch's own `ref_atom_name_chars`, and check_af3_featurise.js
proves those names and every gather exact against AF3. Not the conformer
input: the table reproduces its own rigid distances to 1.5e-4 A. Not the
featuriser: folding AF3's OWN batch from the dump gives the same pLDDT and
the same backbone.

**Why nothing caught it.** `check-af3-diffusion-head.js` and its neighbours
are pinned to `af3-oracle-atom-f32.json`, which is `GSMKQIEDKIEE` - twelve
residues with no F, Y, W, H or P, longest side chain a lysine. Every
ring-bearing residue is untested. The featurise checkers had the same fault
against the same class of toy dump.

**Where to resume.** `tools/gpu/fold.js --dump=` now runs on a real dump
(it was missing `sequences` and `asymId`, and the trunk comparison did not
know that a recycled dump names its captures `pair#0..#3`). On AF3's own
batch with matching recycles:

    node tools/gpu-chrome.mjs tools/gpu/fold.js --dump=/af3-sample200.json \
      --steps=200 --recycles=3
    pair   vs AF3  relRMS 3.69e-2      single vs AF3  relRMS 1.71e-2

which is near the 2.7e-2 the conformer difference is worth, so the trunk is
roughly right and the diffusion head is the suspect. The next step is a
dump with the atom-level captures for a sequence that HAS rings - the
existing atom dump has none - and then check-af3-atom-decoder.js and
check-af3-diffusion-head.js against it. Regenerate with:

    python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 3 \
      --diffusion 200 --sequence <a sequence with F Y W H P> --out af3-rings.json

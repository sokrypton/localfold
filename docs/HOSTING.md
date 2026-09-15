# Hosting the weights somewhere other than Pages


GitHub Pages publishes at most a gigabyte, and the weights are most of it: AF2
monomer 227 MB, AF3 150 MB, before a third model exists. A page meaning to offer
five keeps its parameters elsewhere.

Everything a bundle needs is one field. In `src/bundles/manifests/index.js`:

```js
af3: {
  directory: "./model-af3-int5/",                       // the fallback
  remote: "https://huggingface.co/USER/REPO/resolve/<sha>/",
  ...
}
```

and that is the whole change. Shard URLs are resolved against the bundle's base,
so the store never learns the difference; `build_site.py` and the Pages workflow
both ask `build_site.py --is-remote <family>` and stop publishing a copy.

🔴 **THE ESM-C BUNDLE IS 54 SHARDS WHERE AF3's IS 8, AND MEASURED, THE 8 IS THE
ONE THAT COSTS.** `export_esmc_model.py`'s `SHARD_LIMIT` is 48 MiB applied to the
FLOAT32 export - 2190 MiB, about 46 pieces - and `quantize_af3.py` preserves that
layout rather than re-sharding, so int3 shrinks each eightfold to a 4.1 MiB
median while the count stays. That looked like an oversight and is not.

Fetched from Hugging Face at eight connections, longest first, two interleaved
passes:

| bundle | shards | MiB/s | with a connection idle |
|---|---|---|---|
| `af3-int5` | 8 | 27.9 / 30.1 | **5.30 / 4.52 s** |
| `esmc-600m-int3` | 54 | 24.0 / 28.1 | **1.70 / 0.62 s** |

🔴 **AND THE LOCAL NUMBER IS NOT THE USER'S NUMBER, BY 5-12x.** Every timing
taken through `tools/serve.py` reads shards over loopback, where
`probe-shard-read.js` measures 371 MB/s and the cap is six HTTP/1.1
connections. To the real remote from the A100 box: one OpenDDE shard at
**56 MB/s**, and all twelve in parallel at **78 MB/s aggregate** - Hugging Face
serves HTTP/2, so the connection cap does not apply at all and the constraint
is bandwidth, which twelve connections improve by 1.4x rather than by twelve.
The table above, taken from a different machine, says 27.9-30.1 MiB/s.

So `fold-opendde.js`'s `weightSeconds` of 1.73 is about **6 s** for a user, and
the conclusion below - that a bundle wants more shards than connections - is
about the IDLE TAIL and not about throughput. Where the wire is the limit,
shard count buys nothing and only the byte count does.

### So the next thing worth doing to a bundle is making it SMALLER

Not re-sharding it. At 56-78 MB/s, OpenDDE's 472 MiB is about **6 s of a ~7 s
first experience** - the fold behind it is 2.5 s and the shader compilation now
hides under the download entirely. Nothing else on the page is within an order
of magnitude of that, and every millisecond of kernel work is now competing for
the small half of a user's wait.

It was not attempted here and the reason is mechanical: **the float32 exports
are not on this box**, and `quantize_af3.py` needs them - the bundles here are
already int5. It also ends in a re-publish to the Hugging Face repository the
manifests pin, which is a different kind of decision from a kernel change.

What a serious attempt would have to hold, all of which already exist:
`fold-opendde.js --target=6mrr` (RMSD 1.68 A, TM 0.865),
`probe-nucleic.js` for bond lengths, which RMSD cannot see and where OpenDDE is
already 15% short, `check-opendde-confidence.js`, and
`check-bundle-device-decode.js` for whatever codec comes out. 🔴 And the group
size is not a free parameter: OpenDDE at `--group 128` folds 6MRR into a 3283 A
explosion at pLDDT 46.69, so the knob to move is the BIT DEPTH at group 32, one
tensor class at a time, against those gates.

🔴 **EIGHT SHARDS ON EIGHT CONNECTIONS IS NO PACKING AT ALL.** Every connection
takes one shard, the first to finish has nothing else to do, and the load ends
when the single SLOWEST shard does - which is half the download running
under-parallel. Fifty-four costs 1.83 s of request overhead against eight's 0.27
(the fixed cost of a shard request is a measured **271 ms**, the 307 to
`cdn.hf.co` included) and recovers more than that in packing. Throughput is the
same within this machine's noise either way.

**So the "eight" recorded elsewhere in this file is eight parallel CONNECTIONS
and a longest-first order, not a shard count** - and a bundle wants comfortably
more shards than connections, not the same number.

🔴 **AND AF3 IS THE EXCEPTION, BECAUSE ITS FLOOR IS A TENSOR AND NOT A LAYOUT.**
Two of its float32 tensors are **216 MiB** each - the stacked `transition1`
weights of the diffusion transformer and of the trunk pairformer's single
transition - which are 40.5 MiB apiece at int5. A tensor is contiguous within
one file, so 40.5 MiB on one connection is the makespan floor in ANY sharding,
and the shipped layout already isolates them: tensors per shard reads
`[1, 1, 65, 66, 66, 68, 69, 70]`. Re-sharding to sixteen produces the same two
40.5 MiB shards and gains nothing, which is why it was not done. **Its 4.5-5.3 s
idle tail is those two tensors.**

### 🔴 STREAMING THE FOLD AS THE WEIGHTS ARRIVE: DECLINED, AND THE ARITHMETIC IS THE REASON

"Run the model module by module as it downloads" is the obvious answer to a
first visit being almost all bytes. Three measurements say it cannot pay, and
the third is the one that settles it.

**1. The bundle is not in execution order, so today the answer is 0%.**
AlphaFold 3's shards 2-7 each carry diffusion, confidence, MSA, trunk, embedder
and template tensors mixed together - the packer packs longest-first into even
shards, which destroys stage order by construction. The page says the same from
the other side: loading IntelliFold-2 from Hugging Face, `weightPhases` is
**trunk 8991 ms and diffusion 49 ms**. The "trunk" phase is pulling essentially
the whole bundle, and by the time the diffusion head wants its weights they have
already arrived. Nothing can start early because nothing arrives early.

**2. Even with perfect stage ordering, first compute starts most of the way in.**
The trunk-side weights - embedder, MSA, template, pairformer - are:

| | trunk-side | diffusion | confidence | earliest compute |
|---|---:|---:|---:|---:|
| AlphaFold 3 | 41% | 55% | 4% | **41% of the download** |
| RoseTTAFold3 | 41% | 55% | 3% | 41% |
| OpenDDE | 64% | 31% | 5% | 64% |
| IntelliFold-2 | 70% | 24% | 6% | **70%** |

So the overlappable window is the LAST 59% to 30% of the transfer, and it is
smallest for the biggest bundle.

**3. The saving is `min(download, compute)`, and compute is a second.** Warm
folds at 25 steps are **af3 0.64 s** and **if2 1.39 s**; the trunk's share is
less. Against that:

| | download | trunk compute that could fill the window | saving |
|---|---:|---:|---:|
| if2, from HF at ~68 MB/s | 8991 ms | ~1.3 s | **~10%** |
| af3, throttled to the 8 MB/s a visitor sees | 35.5 s | ~0.5 s | **~1.3%** |

🔴 **AND THAT IS THE WHOLE ARGUMENT: THE SAVING IS CAPPED BY COMPUTE, SO IT IS A
FIXED HALF-SECOND HOWEVER SLOW THE NETWORK IS.** Streaming helps the visitor on
a fast connection, who does not need it, and vanishes for the visitor on a slow
one, who does. That is the opposite of the shape you want from a latency fix.

**And the structure fights it twice more.** The page runs the trunk for TWO
recycles, so pass 2 needs all 48 blocks and only pass 1 could ever overlap; the
diffusion head runs 25 steps over the same 24 transformer blocks, so they must
all be resident by the end of step 1. Incremental residency is also where
CLAUDE.md already records the eviction path being wrong under a budget.

**A weaker version of this was already tried and measured at nothing**: the
biggest-shard-first prefetch order, whose own comment says it cannot be measured
from here, measures as zero.

🔴 **THE ONE VARIANT THAT IS NOT DOMINATED IS PERCEIVED LATENCY, AND IT IS
UNMEASURED.** The contact map is a TRUNK output. With stage ordering AlphaFold 3
could draw one at 41% of the download rather than at 100% - not a second saved,
but a first picture at fourteen seconds instead of thirty-five on a throttled
link. That is a different claim from throughput, nobody here has measured
whether it changes how the wait feels, and it would still want the export
reordered. Recorded so it is not confused with the throughput case, which is
declined.

**What the same arithmetic says IS worth it: fewer bytes.** Which is the section
above, and the cheap lever there does not pay either.


### 🔴 CAN THE BUNDLES BE SMALLER? MEASURED, AND THE CHEAP LEVER DOES NOT PAY

`int5` is **6.02 bits a parameter**, not 5: the scheme is asymmetric per-group at
group 32 with a **float16 scale AND a float16 zero** per group, so 5 + 32/32, and
those four bytes per group of thirty-two are 20% of every bundle. The obvious
move is to widen the group. Measured end to end rather than argued.

**The weight error first**, from `tools/analyse_quantisation.py`, which now takes
`--model` because it was pinned to AF3 and the answer is not a property of the
scheme. Six biggest tensors, four checkpoints, relative RMS:

| scheme | bits/w | saving | af3 | opendde | boltz2 | protenix2 |
|---|---:|---:|---:|---:|---:|---:|
| **int5 asym g32** (shipped) | 6.00 | - | 0.0435 | 0.0411 | 0.0420 | 0.0425 |
| int5 asym g64 | 5.50 | 8.3% | 0.0526 | 0.0517 | 0.0506 | 0.0556 |
| int5 asym g128 | 5.25 | 12.5% | 0.0630 | 0.0639 | 0.0594 | 0.0723 |
| int4 asym g32 | 5.00 | 16.7% | 0.0903 | 0.0848 | 0.0869 | 0.0877 |

g128 is already known to fold OpenDDE into a 3283 A explosion at pLDDT 46.69, and
int4 g32 doubles the error, so **int5 g64 is the only candidate worth a fold.**

**Then the fold, which is the only thing that settles it.** Exported both at
group 64 - OpenDDE 2501.7 -> **433.4 MiB** against the shipped 472, AlphaFold 3
1405.3 -> **242.6** against 265, both 8.2% as predicted:

| 6MRR, 4 seeds, 25 steps | g32 RMSD | g64 RMSD | g32 pLDDT | g64 pLDDT |
|---|---:|---:|---:|---:|
| seed 1 | 0.650 | 0.646 | 83.050 | 81.699 |
| seed 7 | 0.662 | **0.814** | 83.739 | 82.408 |
| seed 21 | 0.620 | **0.755** | 83.634 | 81.876 |
| seed 20260831 | 0.706 | **0.881** | 83.524 | 83.506 |
| **mean (sd)** | **0.660 (0.037)** | **0.774 (0.099)** | **83.49 (0.30)** | **82.37 (0.86)** |

🔴 **SO IT IS A REAL LOSS AND NOT A SEED, WHICH IS THE ONLY REASON FOUR SEEDS
WERE RUN.** One seed would have shown 0.706 -> 0.881 and been inside the 1 A band
this repository measures elsewhere. Four show AlphaFold 3's RMSD **17% worse on
the mean with 2.7x the spread**, worse on three seeds of four, and pLDDT lower on
**four of four**. OpenDDE barely moves at one seed (1.518 -> 1.527, TM 0.9304 ->
0.9278) - so the cost is per checkpoint, and it lands on the default model.

**8.3% of a bundle is about 0.65 s of a first visit at the 78 MB/s this machine
measures to Hugging Face. It is not worth a fifth of AlphaFold 3's accuracy.**

🔴 **AND THE ONE OPTION THAT LOOKS BETTER CANNOT BE BUILT TODAY.** `int5
SYMMETRIC g32` is also 5.50 bits - the same 8.3% - and on two of the four
checkpoints it beats asymmetric g64 outright and comes close to the shipped
scheme: **opendde 0.0434 and protenix2 0.0452**, against asym g64's 0.0517 and
0.0556 and shipped asym g32's 0.0411 and 0.0425. On af3 and boltz2 the ranking
INVERTS and asym g64 wins. Neither can be tried: `quantize_af3.py` emits
asymmetric only, and `codecOf` in src/weights/quantised-upload.js ties symmetry
to width - `signed: bits === 8` - so a symmetric five-bit code has no
representation. Adding one is a codec, a packer flag and a gate arm, for 8.3% on
two models of seven. Written down so the next person costs it rather than
rediscovers it.

**What did change:** `readTensorRange`'s int5 branch refused any group but 32,
and the general path beside it had always been able to read them - the refusal
dated from when the unrolled 20-byte loop was the only int5 reader. int5 at any
group now takes the general path, held to the DEVICE decoder at **zero differing
elements for 5:32, 5:64, 5:128 and 4:32** by `check-quantised-upload.js`. Every
shipped bundle is group 32 and still takes the unrolled fast path, so nothing
moves: af3 folds 83.16921495311351, unchanged to the last digit.


### Every bundle's shard count, and why none of them wants repacking

Counted on 2026-09-15 from the manifest module the PAGE loads, against the
bundle on disk. The two counts agree for all ten present; three bundles are
hosted but not on this box.

| family | shards | on disk | MiB | shard sizes | the largest shard is |
|---|---:|---:|---:|---|---|
| monomer | 8 | 8 | 97 | 12.1-12.4 | |
| multimer | 8 | 8 | 97 | 12.1-12.4 | |
| af3 | 8 | 8 | 265 | 30.6-40.5 | **one tensor** |
| boltz2 | 8 | 8 | 364 | 44.3-54.0 | **one tensor** |
| protenix2 | 8 | 8 | 334 | 41.7-41.7 | **one tensor** |
| ef2-fast-600m | 8 | 8 | 122 | 15.3-15.3 | packing |
| opendde | 12 | 12 | 472 | 39.0-40.5 | **one tensor** |
| esmc (600m) | 16 | 16 | 224 | 13.9-14.1 | packing |
| **rosettafold3** | 16 | 16 | 265 | 12.0-**40.5** | **one tensor** |
| **intellifold2** | 40 | 40 | 611 | 10.3-**72.0** | **one tensor** |
| openbind0 | 8 | - | - | | not on this box |
| ef2-fast-300m | 8 | - | - | | not on this box |
| esmc-300m | 8 | - | - | | not on this box |

🔴 **AND EVERY RAGGED BUNDLE IS AT ITS FLOOR, WHICH IS WORTH KNOWING BEFORE
SOMEBODY "FIXES" ONE.** IntelliFold-2's spread is **86%** - 10.3 MiB against 72.0
- and RoseTTAFold3's is 70%, which reads as a packing failure and is not. The
section above says af3's makespan floor is a TENSOR and not a layout, because a
tensor is contiguous within one file; measured across the panel, that is true of
**every AF3-lineage bundle**, and the biggest shard is exactly one tensor in each:

| | largest shard | and it is |
|---|---:|---|
| af3 | 40.5 MiB | `trunk_pairformer/single_transition/transition1/weights` |
| boltz2 | 54.0 | the same tensor, 64 blocks deep instead of 48 |
| rosettafold3 | 40.5 | the same tensor |
| **intellifold2** | **72.0** | `trunk_pairformer/**pair**_transition/transition1/weights` |

IntelliFold-2's is the PAIR transition rather than the single one, which is the
512-channel pair track again - the same width that makes it the heaviest model in
the panel at 2229 MiB of device memory. **72 MiB on one connection is its floor
in any sharding**, so `repack_shards.py` would move bytes and change nothing that
matters, exactly as re-sharding af3 to sixteen was measured to gain nothing.

The only two bundles whose largest shard is NOT a single tensor are
`ef2-fast-600m` and `esmc`, and those are the two the quantiser's own packer has
already laid out - 15.3 MiB x8 and 13.9-14.1 x16, spreads of 0% and 1%.

**So the two unpublished bundles need no work before they go up.** They are
ragged because their biggest tensors are big.

🔴 **AND af2-monomer's `.js` SHARDS ARE GONE, 129.8 MiB OF THEM.**
`tools/export-js-weights.py` writes a base64-in-JavaScript copy of every shard
for `file://` pages, which is why that bundle had eighteen files where the
others had nine - and `build_site.py` excludes `weights-*.js` and `manifest.js`
from the site, so nothing ever fetched them over HTTP. They are still generated
locally for the offline page; they are simply not hosted.

🔴 **PIN A COMMIT SHA, NOT `main`.** A shard fetched from a moving branch can
change under a manifest that did not, which is the failure the shard-cache token
exists to prevent - and three separate hours have already gone into "<file> has
an invalid byte length", a message that names neither half.

🔴 **AND A TRAILING SLASH, OR THE LAST SEGMENT IS LOST.** `new URL(file, base)`
against ".../resolve/abc123" puts the shard beside `abc123` rather than inside
it. `bundleBaseUrl` adds one; `test/model-bundles.test.js` holds it to that.

Verified against Hugging Face from the browser: CORS passes, the 302 to
`cdn.hf.co` is followed, `?v=` cache tokens survive, ranges answer 206, and the
responses come back `type: "cors"` so the shard cache can store them.

🔴 **AND EVERY BUNDLE IS HOSTED NOW, SO THE PAGES BUILD CARRIES NO WEIGHTS AT
ALL.** `sokrypton/localfold` holds all eight, one directory each. The four
EF2-fast bundles were the last to go up and were 598.6 MiB of a 1 GB allowance
until they did; the build now publishes **0.0 MiB** of parameters.

```
hf upload sokrypton/localfold model-esmfold2-int5 ef2-fast-600m-int5 --repo-type=model
```

🔴 **AND `remote_families()` MATCHED ONLY UNQUOTED KEYS, WHICH IS THE OPPOSITE
OF SAFE.** Three of the four EF2 families are quoted in `index.js` - a key is,
when it is not a bare identifier - so their `remote:` lines were read as
belonging to no family and the build counted hosted bundles as LOCAL. It would
have published 375 MiB a second time, on top of the copies the browser fetches
from Hugging Face: the allowance spent twice for nothing. The same pattern was
wrong in `registry_mismatches` and in two JavaScript tests, all fixed the same
way.

DeepMind's AF3 parameters carry a Prohibited Use Policy - `build_site.py`
already refuses to publish them without `LOCALFOLD_ACCEPT_MODEL_TERMS`. On
Hugging Face the equivalent is a **gated repository**, which is a better fit
than a CI variable because it asks each downloader rather than the deployer.


## A first visit, measured on a throttled link at last

Every weight number in this repository was taken over `tools/serve.py` on
loopback at 371 MB/s. `tools/fold-in-page.py --throttle=<MB/s>` shapes the whole
page through CDP's `Network.emulateNetworkConditions`, so a user's link can be
asked directly rather than argued about. Hugging Face measures 8 MB/s from this
machine; `--latency` sets the round trip and defaults to 30 ms.

An AF3 first visit at 8 MB/s, 58 residues, 4 sampler steps:

| | ms |
|---|---:|
| whole page, click to structure | **39320** |
| of which the trunk WEIGHT phase | 35491 |
| diffusion, confidence, atomReference, targetFeat weights | 22 / 5 / 1 / 1 |
| the fold itself, as the status line reports it | ~2000 |

265 MiB at 8 MB/s is 33 s, so **a first visit is bytes and almost nothing else**.
The four loaders after the trunk cost 29 ms between them because their shards
have already arrived by the time they are asked.

### 🔴 The biggest-shard-first prefetch order is worth nothing, and now we know

`HttpTensorStore.prefetch` sorts shards longest-first, on the makespan argument
that the last shard to start decides when the load ends - and its own comment
says the case it protects "cannot be measured from here". It can now. Against
manifest order, two rounds each at 8 MB/s:

| | run 1 | run 2 |
|---|---:|---:|
| biggest first | 39457 | 39442 |
| manifest order | 39378 | 39398 |

**No difference**, and manifest order is nominally the faster of the two. With
eight connections sharing one 8 MB/s pipe every shard finishes at about the same
time whatever order they start in; the tail can only bind when a single
connection is fast enough that one 40 MiB shard's serial time exceeds what the
others have left. The ordering is three lines and harmless, so it stays - but it
is not a lever, and nobody should reach for it again.

### 🔴 And overlapping the download with the fold cannot beat the bytes either

The tempting next move is to start folding once the TRUNK's weights are in and
stream the diffusion head's behind it. Two measurements say how little that
buys.

The bundles are not laid out by stage. Shards were packed in tensor order, so
trunk tensors are spread through nearly all of them - **AF3 needs 224 of its 265
MiB before its trunk can start, and OpenDDE 432 of 472**. Only one shard per
model is purely diffusion (AF3's `weights-01`, 40.5 MiB; OpenDDE's `weights-02`,
40.5).

| | trunk | diffusion head | confidence |
|---|---:|---:|---:|
| AF3 | 41.0% | 55.2% | 3.5% |
| OpenDDE | 58.8% | 31.0% | 4.7% |

And even with a stage-grouped re-export, the win is bounded by the trunk's own
COMPUTE time, because the link is slower than the arithmetic: at 58 residues the
trunk is about 2 s against 33 s of download, so overlapping perfectly would hide
2 s of 39. It grows with the protein - a 300-token trunk is worth more - but it
is never the download.

**Fewer bytes is the only lever, and quantisation is already at its frontier**:
`tools/quantize_af3.py`'s own header prices int4 g32 with a per-group range
search at 219.6 MiB against int5's 263.5, for RMSD 0.76 against 0.66 and TM
0.942 against 0.953. That is a 17% smaller bundle for a real loss, and it is a
product decision rather than a free win. The search is implemented in
`tools/analyse_quantisation.py` and NOT in the exporter, which is fine for int5
- the same file measures the search at 3-4% of the error - and is what int4
would need first.

## The quantised shard geometry, as it stands

| bundle | shards | total | a shard |
|---|---:|---:|---:|
| af3-int5 | 8 | 264.6 MiB | 33.1 |
| **boltz2-int5** | 8 | 364.4 | 45.5 |
| **protenix2-int5** | 8 | 333.9 | 41.7 |
| opendde-int5 | 12 | 472.5 | 39.4 |
| ef2-fast-600m (esmfold2-int5) | 8 | 122.5 | 15.3 |
| esmc-600m-int3 | 16 | 223.6 | 14.0 |
| af2-multimer | 8 | 97.4 | 12.2 |

Eight is the shape of the shipped set; OpenDDE's twelve and ESM-C's sixteen are
the two exceptions and both are the bigger bundle held at ~40 and ~14 MiB a
shard respectively.

🔴 **boltz2 AND protenix2 HAD NO QUANTISED BUNDLE AT ALL.** Nine families are
published and neither is among them: both existed only as a local float32
export of 1.9 and 1.8 GiB, which is not a thing a browser can be handed. They
are int5 group 32 now, 5.31x, and they fold:

| | pLDDT | RMSD to 6MRR | TM | against the f32 bundle |
|---|---:|---:|---:|---|
| boltz2 int5 | 96.37 | **0.542 A** | 0.972 | 0.537 A |
| protenix2 int5 | 84.76 | 1.723 | 0.917 | 1.564 A |

🔴 **AND "SO QUANTISATION COSTS 0.16 A" WAS ONE SEED OF NOISE.** That is what
this table said, and three seeds say otherwise:

| | seed 20260831 | 7 | 99 | spread |
|---|---:|---:|---:|---:|
| protenix2 f32 | 1.564 | 1.642 | 0.607 | **1.03 A** |
| protenix2 int5 | 1.723 | 1.173 | 0.676 | 1.05 |
| boltz2 int5 | 0.542 | 0.556 | 0.486 | **0.07** |
| af3 int5 | 0.657 | 0.844 | 0.664 | 0.19 |

int5 is indistinguishable from float32 for every model - AlphaFold 3's int5 fold
is BETTER than its float32 one at the shared seed (0.657 against 0.698), which
is the same statement. What the table does show is that **protenix2's sampler
has a 1 A seed spread on this target where boltz2's has 0.07**, so a
single-seed comparison of protenix2 means nothing and this file made one.

Group 32, not 128: docs/EF2FAST.md records OpenDDE at group 128 folding 6MRR
into a 3283 A explosion at pLDDT 46.69, and 128 is ESM-C's alone.

🔴 **NEITHER IS IN THE REGISTRY AND NEITHER IS PUBLISHED.**
`src/bundles/manifests/` has no `boltz2.js` or `protenix2.js`, so the page
cannot load either however good the bundle is - and a bundle the CLI likes can
still be one the page cannot, because the page reads the manifest baked into the
module and pinned to a commit. Publishing them is a `tools/build_site.py` and a
Hugging Face upload away; the bundles exist and are gated.

🔴 **AND THREE PUBLISHED BUNDLES ARE NOW STALE.** `opendde-int5` and
`opendde-full-f32` were re-exported here (the 833-channel single conditioning
and eight zeroed encoder tensors), and `boltz2-f32`'s four negated
`embed_pair_offsets` were corrected in place. The loader REFUSES the old OpenDDE
rather than folding at the wrong width, so the published one is not merely
worse, it no longer loads. See docs/AF3.md.

## THE THREE int5 BUNDLES ARE PUBLISHED, AND opendde WAS THE ONE THAT HAD TO BE

Uploaded to `sokrypton/localfold` at commit
**`068c905dfb0f8cf9b9432eef80d2220ef3ff697f`**: `opendde-int5/` (re-export),
`boltz2-int5/` and `protenix2-int5/` (new). Eleven directories there now.
`check_remote_bundle.py` reads all twelve OpenDDE shards and all eight of
AlphaFold 3's back from the remote.

`src/bundles/manifests/opendde.js` is regenerated from the uploaded bundle and
its `remote:` re-pinned to that commit;
`boltz2.js` and `protenix2.js` are new modules.
`test/registry-manifest-widths.test.js` is GREEN, which is the signal the deploy
blocker below is cleared - it covers all five AF3-lineage families now.

🔴 **AND THE TWO NEW MODELS ARE NOT IN `MODEL_BUNDLES` YET.** Their manifests
exist and their weights are hosted, so nothing about them can 404 - and nothing
LOADS them either, because the registry has no entry. That is deliberate: the
upload is reversible in effect (an unreferenced directory costs storage and
nothing else) and adding a family to the page is a product decision, not a
porting one.

### What the state WAS, before the upload

**No weights had been uploaded and nothing pushed to git.** There was no HF
token on this box, and the branch was 47 commits ahead of `origin/a100` and 38
ahead of `origin/main`.

| bundle | local state | published state | consequence |
|---|---|---|---|
| **opendde-int5** | re-exported (833 scale, eight tensors filled) | 831 scale, eight ZEROS | 🔴 **the loader RAISES on the published one** |
| opendde-full-f32 | re-exported | not published | none |
| boltz2-f32 | four tensors corrected in place | not published | none |
| **boltz2-int5** | new | **not published, not in the registry** | the page cannot load it |
| **protenix2-int5** | new | **not published, not in the registry** | the page cannot load it |
| af3-int5 | unchanged | unchanged, and 404 of 404 tensors agree with the reference | none |
| openbind0-int5 | unchanged | unchanged, 406 of 406 agree | none |

🔴 **THE FIRST ROW IS A DEPLOY BLOCKER AND NOT A DEGRADATION.** OpenDDE joined
`PADDED_SINGLE_COND` upstream, so its diffusion single conditioning is 833
channels; the published bundle carries an 831-wide
`single_cond_initial_norm/scale` and the loader RAISES rather than folding at
the wrong width. That is the right behaviour - the alternative is a silent
target_feat of 445 - and it means **pushing this branch to `main` takes OpenDDE
off the live site until the bundle is re-uploaded.** Pushing to `main` IS the
deploy; see CLAUDE.md.

`test/registry-manifest-widths.test.js` is the gate, and it is RED on purpose:
it reads the committed manifest module and derives the width the same code
derives, so it names the bundle and the numbers without a GPU or the network.
It should stay red until the upload happens.

🔴 **AND DO NOT "FIX" IT BY REGENERATING THE MANIFEST MODULE.** The module
carries byteOffsets and shardDigests, and its `remote:` still pins the OLD
Hugging Face commit - so a locally regenerated manifest against unchanged remote
shards is strictly worse than the raise: the page would fetch the old bytes at
the new offsets. Upload first, then re-pin.

### What publishing takes

    python3 tools/quantize_af3.py --source model-opendde-full-f32 \
      --out model-opendde-int5 --bits 5 --group 32 --shards 12   # done, local
    # upload model-opendde-int5/, model-boltz2-int5/, model-protenix2-int5/
    #   to huggingface.co/sokrypton/localfold
    python3 tools/write_manifest_module.py opendde                # re-pin the sha
    #   ...and NEW modules for boltz2 and protenix2, plus registry entries
    python3 tools/deploy.py                                        # push and verify

The shard counts are already the registry's: twelve for OpenDDE, eight for the
two new ones.

### 🔴 int5 IS THE ONLY THING PUBLISHED, AND THE ORACLES WERE RUN ON float32

Every parity number in docs/AF3.md is from an f32 bundle. The float32 exports
are the QUANTISER'S SOURCE and are never published, so the artefact a visitor
gets had not been held to an oracle at all. Run on the int5 bundles:

| one denoise step against af3-any-model | f32 bundle | int5 bundle |
|---|---:|---:|
| alphafold3 | 1.62e-5 | **3.77e-1** |
| protenix2 | 1.91e-6 | 2.39e-1 |
| boltz2 | 3.31e-3 | 8.22e-1 |

🔴 **AND THE FOLDS ARE UNAFFECTED**, which is the finding rather than a
reassurance: a single denoise step is enormously sensitive to weight
quantisation and the 200-step sampler averages it away. **relRMS on one step is
not a proxy for fold quality once the weights are quantised**, in either
direction - so `check-af3-denoise.js` must be pointed at the float32 bundle, or
it measures the quantiser instead of the port. Its 2e-2 bound is an f32 bound.

The chain that does cover the shipped artefact is:
`check-bundle-vs-params.py` on the f32 source (481 of 481 for opendde, 442 of
442 for boltz2, 404 of 404 for af3), then the quantiser, which has its own gates
in `check-quantised-upload.js` and `check-bundle-device-decode.js`, then the
fold.

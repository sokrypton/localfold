# Hosting the weights somewhere other than Pages


GitHub Pages publishes at most a gigabyte, and the weights are most of it: AF2
monomer 227 MB, AF3 150 MB, before a third model exists. A page meaning to offer
five keeps its parameters elsewhere.

Everything a bundle needs is one field. In `src/reference/manifests/index.js`:

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

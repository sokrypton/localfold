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


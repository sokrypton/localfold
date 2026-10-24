# RFD4-Proteina's weights live in the port they were exported for

🔴 **THE ~420 MB OF INT8 SHARDS ARE NOT IN THIS REPOSITORY AND SHOULD NOT BE
ADDED.** They are published by `sokrypton/rfd4-proteina-webgpu` and fetched
from its Pages URL at run time, so this repository's build keeps carrying
0.0 MiB of parameters. `.gitignore` holds `model-rfd4p-int8/*.bin`; the loader's
chain is local export → the published port → Hugging Face
(`shared/rfd4p/safetensors.js`).

**What IS tracked here, and why:** the architecture config and the tensor
manifest. They are small, they are text, and the tests read them without a
network — `test/rfd4p-core.test.js` pins the config's dimensions and checks the
manifest indexes all 1,257 tensors.

- upstream: `sokrypton/rfd4-proteina-webgpu`
- commit: `7e93c47674eb04febff48c24e34388209f8718c3`
- `config.json` — the model architecture (upstream keeps a second, identical
  copy beside its shards; one is enough here and `tools/check-rfd4p-manifest.py`
  maps our paths onto theirs)
- `manifest.json` — which shard and offset every tensor lives at

🔴 **THE DIRECTORY IS `model-rfd4p-int8/`, NOT `weights/`.** It was renamed once
the hosting decision was settled: the shards are served by that port's Pages URL
and are never kept here, so a top-level directory called `weights/` advertised
something this repository does not have. It follows the nine fold bundles'
`model-*` convention now.

🔴 **AND THIS IS A COPY, SO IT CAN DRIFT.** The manifest records byte offsets
into shards it does not contain. If that port re-exports its weights, these
offsets describe the old ones and the shards fetched from it are the new ones:
every tensor would be read from the wrong place, and nothing here would say so.
It is the same failure `model-boltz2-int5/` has today against its own manifest
module, which `tools/build_site.py` refuses to build over. **`tools/check-rfd4p-manifest.py` is that check** — it compares our copies
against the pinned commit and reports upstream moving, and it reaches the
network by necessity, because the whole question is whether a file HERE agrees
with files THERE.

Regenerate a local export with `python3 tools/export_rfd4p_model.py`; the
loader prefers a local `model-rfd4p-int8` over the network when one exists.

# Running LocalFold yourself

Everything here is optional. [localfold.org](https://localfold.org) needs none
of it - this is for running the site from a checkout, getting the fast path on
Linux/NVIDIA, and using the kernels as a library.

## Running the site from a checkout

The site is static files with no build step, so a plain file server is enough:

```bash
git clone https://github.com/sokrypton/localfold && cd localfold
python3 -m http.server 4173     # then open http://127.0.0.1:4173/
```

The weights are fetched from the network on first use, or read from a local
`model/` directory if you have exported one.

🔴 **While DEVELOPING, use `python3 tools/serve.py` instead.** A plain
`http.server` sends no cache headers, so Chrome caches the ES modules
heuristically and a reload serves the old ones - an edit lands, nothing
changes, and the code looks wrong. `tools/serve.py` sends `no-store` on
everything a developer edits and a year on the weight shards, which are the one
thing that must still cache.

### On Linux with an NVIDIA GPU, two Chrome flags are worth about 2x

Chrome does not give a page WebGPU's `shader-f16` on **any** NVIDIA GPU unless
asked: Dawn gates it vendor-wide pending a conformance investigation
([crbug.com/42251215](https://crbug.com/42251215)). The subgroup matrix units
this port uses are experimental and behind a second flag. Measured here on an
A100 with Chrome 152, the same 825-residue fold takes **21.0 s** with a stock
browser and **10.8 s** with both flags — and folds correctly either way.

Chrome on this machine also needed to be pointed at Vulkan before it would
offer a WebGPU adapter at all, which is common on headless and server installs
and unnecessary on many desktops. The whole set, in a **separate profile** so
your everyday browser keeps its defaults:

```bash
google-chrome \
  --user-data-dir=/tmp/localfold-chrome \
  --enable-unsafe-webgpu \
  --enable-dawn-features=vulkan_enable_f16_on_nvidia \
  --use-angle=vulkan --use-vulkan=native --enable-features=Vulkan \
  http://127.0.0.1:4173/
```

`chrome://gpu` in that window should list WebGPU as hardware accelerated. Drop
the two Vulkan lines first if your desktop already provides an adapter; keep
the two flags above them, which are the ones worth the time.

These are developer flags and the f16 one turns off a gate Chrome set for a
reason, so use the separate profile rather than making it your default browser.
On Apple silicon `shader-f16` is available as shipped, so only
`--enable-unsafe-webgpu` changes anything there — and measured on an M2 it
changes nothing, because this port's matrix kernel wants 16x16 units and Apple
offers 8x8x8, so the register kernel is chosen either way.

`--enable-unsafe-webgpu` is what gates the subgroup matrix units, on **both**
platforms. No visitor to a page has them without it. On Apple that costs
nothing; on NVIDIA it is part of the ~2x above.

### As a library

```bash
npm install localfold webgpu
```

```js
import { createNodeDevice } from "localfold/node";
const { device } = await createNodeDevice();   // Dawn, with the toggles below
```

🔴 **AND NODE IS THE ONLY PLACE THE FAST PATH IS REACHABLE WITHOUT ASKING THE
USER FOR ANYTHING.** The two capabilities above are browser flags a visitor does
not have; Dawn's node binding takes them as arguments, so `createNodeDevice`
turns them on for itself and reports `shader-f16` and the subgroup matrix units
on a machine that has them. `webgpu` is an optional peer dependency - if its
prebuilt binary wants a newer GLIBC than you have, `npm i webgpu@0.4.0`.

The API is the kernels (`EvoformerStackGpu` and 48 others, all taking a
`GPUDevice`), and the `0.0.x` version is meant literally: treat it as unstable
until a single sequence-in-structure-out entry point exists.

## Folding without a GPU

**Settings, Fold on, CPU** folds ESMFold2 in a worker, using half the logical cores so the
machine stays usable. Choosing it reloads the page once, keeping what you typed: threads need
`SharedArrayBuffer`, which needs the COOP and COEP headers, and GitHub Pages sends neither, so
`coi-worker.js` adds them. That service worker is registered only then, and removed when you
choose the GPU again.

One sample, a second fold in the same process, node on an eight-CPU cloud container
(`tools/fold-esmfold2-cpu.mjs`), with the shipped int5 bundles and int3 towers:

| target | model | tokens | threads | fold (s) | trunk (s) | sampler (s) | pLDDT | pTM | ipTM | CA RMSD to the crystal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ubiquitin (1UBQ) | 300M | 76 | 1 | 135 | 104 | 22 | 87.0 | 0.86 | | 0.58 A |
| ubiquitin (1UBQ) | 300M | 76 | 4 | 68 | 47 | 15 | 87.0 | 0.86 | | 0.58 A |
| ubiquitin (1UBQ) | 300M | 76 | 8 | 54 | 36 | 14 | 87.0 | 0.86 | | 0.58 A |
| ubiquitin (1UBQ) | 600M | 76 | 8 | 59 | 38 | 16 | 88.2 | 0.88 | | 0.69 A |
| barnase-barstar (1BRS) | 300M | 199 | 8 | 301 | 220 | 59 | 77.2 | 0.73 | 0.61 | 1.59 A |
| barnase-barstar (1BRS) | 600M | 199 | 8 | 318 | 233 | 61 | 91.9 | 0.94 | 0.92 | 0.71 A |
| ubiquitin, repeated | 300M | 150 | 8 | 207 | 148 | 43 | | | | |
| ubiquitin, repeated | 300M | 300 | 8 | 670 | 509 | 110 | | | | |

The 600M tower costs about 6% more time here, since the trunk dominates, and about a gibibyte
more memory. Choosing the CPU picks it where the browser reports 16 GB or more, and 300M
otherwise; the variant row still says which. One target each, so read the 1BRS gap with care.

A first fold adds about 10 s to load the weights and about 20 s to decode the tower's blocks,
which a later fold keeps. The time grows close to the square of the tokens over this range, as
most of it is not the triangle's cube yet: at 76 tokens the helper threads sat idle about 77%
of the fold, waiting on single-threaded JavaScript, which is why one thread is only 2.5 times slower
than eight.

Without isolation (a browser with no service worker) the fold runs on one thread. The kernels
here are the references in `cpu/`, with their matrix multiplies, triangle contractions and logistics in generated
WebAssembly SIMD (`cpu/runtime/`). `npm run test:cpu` drives the whole path through the page.

## More

`docs/DEVELOPING.md` has the reference numbers, benchmarks and deployment
notes; `CLAUDE.md` is how to run the gates in this checkout.

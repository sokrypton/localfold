# ESM-C, compressed

ESMFold2-Fast folds from a protein language model instead of an alignment, which
is the whole reason to care about it here: LocalFold's MSA search is a round trip
to a queueing server, and a target with no homologues gets nothing back. The
question this directory answers is whether the language model fits in a browser.

    python3 tools/esmc/fetch.py                        # 3.0 GB, ungated, MIT
    python3 tools/esmc/probe-esmc-compression.py       # the table
    python3 tools/esmc/fetch_targets.py --out targets --count 16
    .venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py \
        --pdb-dir targets --seed-spread 2

| file | what it is |
|---|---|
| `fetch.py` | pulls a tower and the folding model that was trained against it |
| `safetensors_read.py` | the format, in 60 lines, because this machine's Python refuses `pip install safetensors` |
| `esmc_forward.py` | ESM-C's 36 blocks and ESMFold2's shim, in torch |
| `quantisation.py` | the schemes, including the exact int5 packer LocalFold ships AF3 under |
| `probe-esmc-compression.py` | what each scheme costs the folding trunk's input |
| `probe-esmfold2-structure.py` | ...and what that costs the structure. Needs `esm`, so run it with `.venv-esm/bin/python` |
| `fetch_targets.py` | held-out targets: single chains released after the checkpoint's cutoff |

`esmc_forward.py` is written from the block layout
`../alphafold3/converters/esmc.py` documents rather than by importing the `esm`
package, so that an arm and its reference do not share a forward pass. It agrees
with `transformers`' own ESMC to **2.1e-6** on every one of the 37 hidden states
and to 1.0e-6 on the logits, which is float32 accumulation-order noise - see
`docs/ESMFOLD2.md` for what that check pins down.

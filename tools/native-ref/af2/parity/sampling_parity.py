"""Stock AlphaFold-Multimer's feature functions on LocalFold's OWN draws, against LocalFold's features.

    JAX_PLATFORMS=cpu ~/venv_colabfold/bin/python cuda/af2/parity/sampling_parity.py \
        <draws.json> <data dir> [--pool=kept|all] [--write=<new data dir>]

draws.json is ours_draws.mjs's (per recycle: centres, extras, BERT-masked centre codes over the merged,
deduplicated alignment); the data dir is af2-featurise's with the same options. For each recycle the
centres' masked codes and the extra rows go through alphafold-colabfold's nearest_neighbor_clusters,
create_msa_feat and create_extra_msa_feature - the multimer's in-model sampling with its draws replaced by
ours - and the result is compared, channel group by channel group, with the f<k>/ features LocalFold wrote.

--pool=kept clusters over the extra rows LocalFold keeps (isolates the cluster FORMULA); --pool=all over
every non-centre row, ours first, which is what stock does (sample_msa puts the whole remainder in
extra_msa and only create_extra_msa_feature crops it). --write copies the data dir with the stock
features in place, for `localfold-af2 <dir>`: the network is gated separately (oracle.py), so folding
both directories prices the feature difference in structure.
"""
import argparse
import base64
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
from alphafold.model import modules_multimer as mm

jax.config.update("jax_default_matmul_precision", "highest")

GROUPS = {"onehot 0-22": slice(0, 23), "has_del 23": slice(23, 24), "del_val 24": slice(24, 25),
          "profile 25-47": slice(25, 48), "del_mean 48": slice(48, 49)}


def read_native(path):
    entries, raw = {}, np.fromfile(os.path.join(path, "model.bin"), dtype=np.uint8)
    for line in open(os.path.join(path, "model.idx")):
        p = line.split()
        if not p or p[0] == "m":
            continue
        entries[p[1]] = (p[0], int(p[2]), int(p[3]))
    return raw, entries


def tensor(raw, entries, name):
    kind, off, n = entries[name]
    return raw[off * 4:(off + n) * 4].view(np.int32 if kind == "i" else np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draws"); ap.add_argument("data")
    ap.add_argument("--pool", default="kept", choices=("kept", "all"))
    ap.add_argument("--write", default="")
    # the MONOMER's stock clustering is the TF pipeline's (ColabFold's numpy port of it): argmax assignment and
    # (sum + centre) / (1e-6 + 1 + count); the multimer's is modules_multimer's, in-model
    ap.add_argument("--model", default="multimer", choices=("multimer", "monomer"))
    a = ap.parse_args()
    import json
    d = json.load(open(a.draws))
    L = d["L"]
    msa = np.frombuffer(base64.b64decode(d["msa"]), np.uint8).reshape(-1, L).astype(np.int32)
    dele = np.frombuffer(base64.b64decode(d["deletions"]), np.float32).reshape(-1, L)
    raw, entries = read_native(a.data)
    out = raw.copy() if a.write else None
    for k, plan in enumerate(d["plans"]):
        centers, extras = np.asarray(plan["centers"]), np.asarray(plan["extras"])
        codes = np.frombuffer(base64.b64decode(plan["centerCodes"]), np.uint8).reshape(-1, L).astype(np.int32)
        if a.pool == "all":
            rest = np.setdiff1d(np.arange(len(msa)), np.concatenate([centers, extras]), assume_unique=True)
            pool = np.concatenate([extras, rest])
        else:
            pool = extras
        batch = {"msa": jnp.asarray(codes), "deletion_matrix": jnp.asarray(dele[centers]),
                 "msa_mask": jnp.ones(codes.shape, jnp.float32),
                 "extra_msa": jnp.asarray(msa[pool]), "extra_deletion_matrix": jnp.asarray(dele[pool]),
                 "extra_msa_mask": jnp.ones((len(pool), L), jnp.float32)}
        if a.model == "multimer":
            prof, delmean = mm.nearest_neighbor_clusters(batch)
        else:
            from alphafold.model.tf import data_transforms as dt
            protein = {key: np.asarray(value) for key, value in batch.items()}
            protein = dt.summarize_clusters()(dt.nearest_neighbor_clusters()(protein))   # (both curry1)
            prof, delmean = protein["cluster_profile"], protein["cluster_deletion_mean"]
        batch["cluster_profile"], batch["cluster_deletion_mean"] = jnp.asarray(prof), jnp.asarray(delmean)
        stock = np.asarray(mm.create_msa_feat(batch))
        xf, xm = mm.create_extra_msa_feature(batch, len(extras))
        ours = tensor(raw, entries, f"f{k}/msa_feat").reshape(stock.shape)
        line = [f"pass {k}: {len(centers)} centres, {len(extras)} extra, pool {len(pool)}"]
        for name, sl in GROUPS.items():
            diff = np.abs(stock[..., sl] - ours[..., sl])
            line.append(f"{name} max {diff.max():.2e} mean {diff.mean():.2e}")
        ex = tensor(raw, entries, f"f{k}/extra_msa").reshape(-1, L)[:len(extras)]
        xf = np.asarray(xf)
        line.append("extra rows " + ("SAME" if np.array_equal(xf[..., :23].argmax(-1), ex) else "DIFF"))
        ev = tensor(raw, entries, f"f{k}/extra_deletion_value").reshape(-1, L)[:len(extras)]
        line.append(f"extra del max {np.abs(xf[..., 24] - ev).max():.2e}")
        print("\n  ".join(line))
        if out is not None:
            kind, off, n = entries[f"f{k}/msa_feat"]
            out[off * 4:(off + n) * 4] = np.ascontiguousarray(stock, np.float32).view(np.uint8).ravel()
    if out is not None:
        os.makedirs(a.write, exist_ok=True)
        shutil.copy(os.path.join(a.data, "model.idx"), a.write)
        out.tofile(os.path.join(a.write, "model.bin"))
        print("wrote", a.write)


if __name__ == "__main__":
    main()

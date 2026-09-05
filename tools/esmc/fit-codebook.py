"""One shared codebook for the whole tower: fit it, assign it, write the shards.

    python3 tools/esmc/fit-codebook.py --esmc esmc-600m --clusters 1024 \
        --dimension 4 --group 64 --out codebook-1024x4.npz --device cuda

Scalar quantisation puts its levels on a uniform grid and spends most of them
where there are no weights. A codebook puts them where the weights ARE, and
`tools/esmc/vector-quantise.py` measures that as 0.75 bits a weight on ESM-C:
1024 entries over 4-dimensional vectors reaches the reconstruction error of
scalar int3 at group 64, at 2.75 bits against 3.50.

🔴 ONE CODEBOOK FOR EVERY TENSOR, NOT ONE PER TENSOR. A table per tensor is 144
tables to ship, 144 k-means to fit, and a decoder that has to find the right one
before it can read a weight. A single table is 8 KB, is fitted once on a sample
pooled across the tower, and makes the decode a lookup with no bookkeeping -
which is what makes this shippable rather than merely small.

🔴 AND THE GROUP SCALE IS AN RMS, NOT A MAXIMUM. A shared codebook only works if
every group presents it the same distribution, and dividing by the group's
largest weight standardises the OUTLIER rather than the bulk - one heavy tail
and the other 63 weights arrive squashed into the middle of a table fitted on
something else. The root mean square standardises the bulk, which is what the
table was fitted on.

🔴 AND THERE IS NO ZERO POINT, DELIBERATELY. An asymmetric scalar scheme needs
one because its grid is symmetric and its weights are not; a codebook's entries
are wherever k-means put them, so the offset is already inside the table. That
is the half a bit per group this does not have to spend.

What it writes is what a shard would hold: per tensor a code array and its
float16 scales, plus the one shared table. `quantisation.codebook()` replays it,
so the same probes that price a scalar scheme price this one.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import esmc_forward as E          # noqa: E402


def normalise(values, group, dimension, device):
    """-> (vectors, scales, pad). Unit-RMS groups, split into short vectors."""
    flat = torch.as_tensor(np.asarray(values, np.float32).reshape(-1),
                           device=device)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    grouped = flat.reshape(-1, group)
    scales = torch.sqrt((grouped ** 2).mean(1, keepdim=True)).half().float()
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    return (grouped / safe).reshape(-1, dimension), scales.reshape(-1), pad


def assign(vectors, codebook, budget=2 ** 27):
    """Nearest centre per vector, chunked by the size of the DISTANCE BLOCK.

    🔴 CHUNKING BY VECTOR COUNT IS CHUNKING BY THE WRONG AXIS. The block this
    materialises is (chunk x clusters), so a fixed chunk of two million vectors
    is 8 GB at 1024 entries and 32 GB at 4096 - the same code runs and then
    stops running when the table gets bigger. The budget is on the product.
    """
    clusters = len(codebook)
    chunk = max(1, budget // clusters)
    norms = (codebook ** 2).sum(1)
    out = torch.empty(len(vectors), dtype=torch.long, device=vectors.device)
    for start in range(0, len(vectors), chunk):
        block = vectors[start:start + chunk]
        out[start:start + chunk] = torch.argmin(
            norms[None, :] - 2.0 * (block @ codebook.T), dim=1)
    return out


def fit(vectors, clusters, iterations, seed=0):
    """Lloyd's algorithm, with empty centres restarted on the worst-fit vectors.

    🔴 AN EMPTY CENTRE IS A WASTED CODE AND k-means MAKES THEM. Left alone it
    keeps its initial position and no vector ever selects it, so a 1024-entry
    table can be a 900-entry table at the same price. Restarting it on the
    vectors currently worst served is what keeps the rate honest.
    """
    generator = torch.Generator(device=vectors.device).manual_seed(seed)
    pick = torch.randperm(len(vectors), generator=generator,
                          device=vectors.device)[:clusters]
    codebook = vectors[pick].clone()
    for _ in range(iterations):
        index = assign(vectors, codebook)
        counts = torch.bincount(index, minlength=clusters).float()
        total = torch.zeros_like(codebook)
        total.index_add_(0, index, vectors)
        alive = counts > 0
        codebook[alive] = total[alive] / counts[alive][:, None]
        dead = (~alive).nonzero().flatten()
        if len(dead):
            error = ((vectors - codebook[index]) ** 2).sum(1)
            worst = torch.topk(error, len(dead)).indices
            codebook[dead] = vectors[worst]
    return codebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--clusters', type=int, default=1024)
    parser.add_argument('--dimension', type=int, default=4)
    parser.add_argument('--group', type=int, default=64)
    parser.add_argument('--sample', type=int, default=4_000_000,
                        help='vectors pooled across the tower to fit on')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()

    if arguments.group % arguments.dimension:
        raise SystemExit('group %d must be a multiple of the vector dimension %d'
                         % (arguments.group, arguments.dimension))
    torch.set_grad_enabled(False)
    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    keys = [key for key, _ in checkpoint.matrices()]
    rate = (np.log2(arguments.clusters) / arguments.dimension
            + 16.0 / arguments.group)
    print('%d tensors, %d entries x %dd, group %d -> %.2f bits/weight'
          % (len(keys), arguments.clusters, arguments.dimension,
             arguments.group, rate), flush=True)

    # Pool a sample from every tensor, so the table is not fitted on the first
    # block and applied to the last.
    per_tensor = max(1, arguments.sample // len(keys))
    generator = np.random.default_rng(0)
    pooled = []
    for key in keys:
        vectors, _, _ = normalise(checkpoint[key], arguments.group,
                                  arguments.dimension, arguments.device)
        take = min(per_tensor, len(vectors))
        pick = torch.as_tensor(
            generator.choice(len(vectors), take, replace=False),
            device=arguments.device)
        pooled.append(vectors[pick])
    pooled = torch.cat(pooled)
    print('fitting on %d pooled vectors' % len(pooled), flush=True)

    started = time.time()
    codebook = fit(pooled, arguments.clusters, arguments.iterations)
    used = len(torch.unique(assign(pooled, codebook)))
    print('fitted in %.0fs, %d of %d entries used'
          % (time.time() - started, used, arguments.clusters), flush=True)

    dtype = np.uint8 if arguments.clusters <= 256 else np.uint16
    output = {'__codebook__': codebook.cpu().numpy().astype(np.float16),
              '__meta__': np.array(['codebook', str(arguments.clusters),
                                    str(arguments.dimension),
                                    str(arguments.group), '%.4f' % rate])}
    errors = []
    for number, key in enumerate(keys):
        original = checkpoint[key].reshape(-1)
        vectors, scales, pad = normalise(original, arguments.group,
                                         arguments.dimension, arguments.device)
        index = assign(vectors, codebook)
        rebuilt = (codebook[index].reshape(-1, arguments.group)
                   * scales[:, None]).reshape(-1)
        if pad:
            rebuilt = rebuilt[:rebuilt.numel() - pad]
        reference = torch.as_tensor(original, device=arguments.device)
        errors.append(float(torch.sqrt(((rebuilt - reference) ** 2).sum()
                                       / (reference ** 2).sum())))
        output[key + '.codes'] = index.cpu().numpy().astype(dtype)
        output[key + '.scales'] = scales.cpu().numpy().astype(np.float16)
        output[key + '.pad'] = np.array([pad], np.int32)
        if (number + 1) % 36 == 0:
            print('  %d/%d tensors, relRMS so far %.4f'
                  % (number + 1, len(keys), float(np.mean(errors))), flush=True)

    np.savez(arguments.out, **output)
    print('wrote %s   reconstruction relRMS %.4f at %.2f bits/weight'
          % (arguments.out, float(np.mean(errors)), rate))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

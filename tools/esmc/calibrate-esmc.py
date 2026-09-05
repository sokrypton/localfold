"""Quantise ESM-C against real sequences: GPTQ, or an importance-weighted search.

    python3 tools/esmc/calibrate-esmc.py --fasta uniref50.head.fasta \\
        --bits 4 --method gptq --out calibrated-int4.npz --device cuda

Round-to-nearest asks what the closest code to a weight is. This asks what
codes keep the LAYER'S OUTPUT the same on protein sequences, which is the
question that matters, and it is why the calibration set is UniRef50 rather
than anything synthetic.

🔴 THE BLOCKS ARE QUANTISED IN ORDER, AND SO ARE THE FOUR MATMULS INSIDE ONE.
Every layer is calibrated on activations produced by weights that are ALREADY
quantised, so each one corrects for the error its predecessors introduced
rather than for an error nothing will make. That costs five passes over the
calibration set per block instead of one, which on this model is seconds. The
cheap variant - capture all four inputs from the float32 block and quantise
them together - is what most implementations do and it leaves the compensation
half-applied.

🔴 AND THE CALIBRATION SET IS BUCKETED BY LENGTH, NOT PADDED. A padded batch
needs an attention mask everywhere, and a mask that is wrong in one place is a
Hessian that is wrong everywhere and silently. Sequences are cropped to a few
fixed lengths and batched with their own kind.
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
import gptq as G                  # noqa: E402

LENGTHS = (96, 160, 256)


def read_fasta(path, minimum, limit):
    """-> sequences, in file order, stopping once `limit` are collected."""
    out, current = [], []
    with open(path, 'r', errors='ignore') as handle:
        for line in handle:
            if line.startswith('>'):
                if current:
                    sequence = ''.join(current)
                    if len(sequence) >= minimum:
                        out.append(sequence)
                        if len(out) >= limit:
                            return out
                    current = []
            else:
                current.append(line.strip())
    if current and len(''.join(current)) >= minimum:
        out.append(''.join(current))
    return out


def batches(sequences, lengths, per_length, batch_size, device, seed=0):
    """-> [(ids tensor of one uniform length)], cropped rather than padded."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    out, cursor = [], 0
    for length in lengths:
        chosen = []
        while len(chosen) < per_length and cursor < len(order):
            sequence = sequences[order[cursor]]
            cursor += 1
            if len(sequence) < length:
                continue
            start = rng.integers(0, len(sequence) - length + 1)
            chosen.append(E.sequence_ids(sequence[start:start + length]))
        for i in range(0, len(chosen), batch_size):
            block = chosen[i:i + batch_size]
            if block:
                out.append(torch.as_tensor(np.stack(block), dtype=torch.long,
                                           device=device))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--method', default='gptq',
                        choices=('gptq', 'imatrix', 'rtn'))
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group', type=int, default=32)
    parser.add_argument('--damping', type=float, default=0.01)
    parser.add_argument('--sequences', type=int, default=96,
                        help='per length bucket, so three times this in all')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    tower = E.Tower(checkpoint, device=arguments.device)

    pool = read_fasta(arguments.fasta, min(LENGTHS),
                      arguments.sequences * len(LENGTHS) * 12)
    calibration = batches(pool, LENGTHS, arguments.sequences, arguments.batch,
                          arguments.device, arguments.seed)
    tokens = sum(int(b.shape[0]) * int(b.shape[1]) for b in calibration)
    print('%d sequences from %s -> %d batches, %d tokens'
          % (len(pool), arguments.fasta, len(calibration), tokens), flush=True)

    # The residual stream for every calibration batch, carried forward as the
    # blocks are quantised one at a time.
    states = [tower.embed(ids) for ids in calibration]
    positions = [tower.positions(int(ids.shape[1])) for ids in calibration]

    output = {}
    started = time.time()
    for layer in range(checkpoint.n_layers):
        for leaf in E.BLOCK_MATRICES:
            key = checkpoint.block_key(layer, leaf)
            weight = tower._t(key)
            inner = int(weight.shape[1])
            hessian = torch.zeros(inner, inner, dtype=torch.float32,
                                  device=arguments.device)
            energy = torch.zeros(inner, dtype=torch.float32,
                                 device=arguments.device)
            if arguments.method != 'rtn':
                count = 0
                for x, position in zip(states, positions):
                    record = {}
                    tower.block(x, layer, position, record=record)
                    flat = record[leaf].reshape(-1, inner).float()
                    hessian += 2.0 * (flat.T @ flat)
                    energy += (flat ** 2).sum(0)
                    count += int(flat.shape[0])
                energy /= max(count, 1)

            if arguments.method == 'gptq':
                values, packed = G.gptq_quantise(
                    weight, hessian, arguments.bits, arguments.group,
                    damping=arguments.damping, return_codes=True)
            elif arguments.method == 'imatrix':
                values, packed = G.imatrix_quantise(
                    weight, energy, arguments.bits, arguments.group,
                    return_codes=True)
            else:
                values, packed = G.round_to_nearest(
                    weight, arguments.bits, arguments.group, return_codes=True)

            tower._cache[key] = values          # every later layer sees this
            codes, scales, zeros = packed
            output[key + '.codes'] = codes.cpu().numpy()
            output[key + '.scales'] = scales.cpu().numpy()
            output[key + '.zeros'] = zeros.cpu().numpy()
            del hessian

        states = [tower.block(x, layer, position)
                  for x, position in zip(states, positions)]
        print('  block %2d/%d  %.0fs' % (layer + 1, checkpoint.n_layers,
                                         time.time() - started), flush=True)

    output['__meta__'] = np.array(
        [arguments.method, str(arguments.bits), str(arguments.group),
         str(tokens), str(arguments.damping)])
    np.savez(arguments.out, **output)
    print('wrote %s  (%d tensors, %s at %d bits group %d, %d calibration tokens)'
          % (arguments.out, len(output) // 3, arguments.method, arguments.bits,
             arguments.group, tokens))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

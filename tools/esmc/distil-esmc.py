"""Quantisation-aware distillation: train the tower to survive its own codes.

    python3 tools/esmc/distil-esmc.py --esmc esmc-300m --esmfold2 esmfold2-fast-300m \
        --bits 3 --group 128 --fasta uniref50-50k.fasta --steps 4000 \
        --out distilled-int3-g128.npz --device cuda

GPTQ and the importance matrix choose better codes for weights that are fixed.
This moves the WEIGHTS so that their codes are better, which is the one thing
post-training quantisation cannot do, and it is where the remaining bit lives.

🔴 THE OBJECTIVE IS NOT THE MASKED-LM LOSS. ESMFold2 never reads ESM-C's
logits; it reads the 37 hidden states and mixes them with a learned softmax. So
the teacher is the float32 tower itself and the loss is the error in those
states WEIGHTED BY THAT MIX - self-distillation, no labels, and the quantity
being minimised is the quantity `probe-esmc-compression.py` reports. Training
against perplexity would optimise a head that gets thrown away.

🔴 AND THE FAKE QUANTISER IS THE SHIPPING PACKER, float16 METADATA INCLUDED.
Training against a scale the reader will not see trains the weights to sit on a
grid that does not exist. `group_codes` is the same function
`tools/esmc/gptq.py` and `tools/quantize_af3.py` agree on.

🔴 AND THE GROUPS ARE OVER THE FLATTENED TENSOR, NOT PER ROW. That is what
`tools/quantize_af3.py` does - it flattens and takes 32 at a time, so a group
may straddle two output channels - and it is the only version that works at
group 128 on the 300M tower, whose 960 columns are not a multiple of it. GPTQ
needs the per-row alignment and therefore cannot go past group 64 there; this
does not.
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


def flat_groups(values, group):
    """-> (groups, group) over the FLATTENED tensor, tail padded with zeros."""
    flat = values.reshape(-1)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.reshape(-1, group), pad


def fake_quantise(values, bits, group):
    """Straight-through: the forward sees codes, the gradient sees the weights."""
    grouped, pad = flat_groups(values, group)
    codes, scales, zeros = G.group_codes(grouped, bits, group)
    rebuilt = G.group_values(codes, scales, zeros).reshape(-1)
    if pad:
        rebuilt = rebuilt[:-pad]
    rebuilt = rebuilt.reshape(values.shape)
    return values + (rebuilt - values).detach()


def export(values, bits, group):
    """The three arrays a shard holds, in the flattened layout it holds them."""
    grouped, pad = flat_groups(values.detach(), group)
    codes, scales, zeros = G.group_codes(grouped, bits, group)
    return (codes.to(torch.uint8).cpu().numpy(),
            scales.reshape(-1).half().cpu().numpy(),
            zeros.reshape(-1).half().cpu().numpy(), int(pad))


def read_fasta(path, minimum, limit):
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
    return out


class Sampler:
    """Uniform-length batches by cropping. Padding would need a mask, and a
    mask that is wrong in one place is a gradient that is wrong everywhere."""

    def __init__(self, sequences, batch, device, seed=0):
        buckets = {n: [s for s in sequences if len(s) >= n] for n in LENGTHS}
        self.by_length = {n: v for n, v in buckets.items() if v}
        self.lengths = sorted(self.by_length)
        if not self.lengths:
            raise ValueError('no sequence reaches %d residues' % min(LENGTHS))
        self.batch, self.device = batch, device
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        length = self.lengths[int(self.rng.integers(len(self.lengths)))]
        pool = self.by_length[length]
        rows = []
        for index in self.rng.integers(0, len(pool), self.batch):
            sequence = pool[int(index)]
            start = int(self.rng.integers(0, len(sequence) - length + 1))
            rows.append(E.sequence_ids(sequence[start:start + length]))
        return torch.as_tensor(np.stack(rows), dtype=torch.long,
                               device=self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-300m')
    parser.add_argument('--esmfold2', default='esmfold2-fast-300m')
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--bits', type=int, default=3)
    parser.add_argument('--group', type=int, default=128)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--evaluate-every', type=int, default=250)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    shim = E.Shim(ROOT / arguments.esmfold2, device=arguments.device)
    mix = shim.combine.to(arguments.device)

    teacher = E.Tower(checkpoint, device=arguments.device)
    student = E.Tower(checkpoint, device=arguments.device)
    keys = [key for key, _ in checkpoint.matrices()]
    with torch.no_grad():
        for key in keys:
            teacher._t(key)
    parameters = {key: torch.nn.Parameter(teacher._t(key).clone())
                  for key in keys}
    trainable = list(parameters.values())
    total = sum(p.numel() for p in trainable)
    print('%d matrices, %.1fM trainable, int%d group %d = %.2f bits/weight'
          % (len(keys), total / 1e6, arguments.bits, arguments.group,
             arguments.bits + 32.0 / arguments.group), flush=True)

    sequences = read_fasta(arguments.fasta, min(LENGTHS), 200000)
    split = max(64, len(sequences) // 50)
    held_out = Sampler(sequences[:split], arguments.batch, arguments.device, 777)
    sampler = Sampler(sequences[split:], arguments.batch, arguments.device,
                      arguments.seed)
    print('%d sequences, %d held out' % (len(sequences), split), flush=True)

    optimiser = torch.optim.AdamW(trainable, lr=arguments.lr, weight_decay=0.0,
                                  betas=(0.9, 0.95))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: min(1.0, (step + 1) / max(1, arguments.warmup))
        * (0.5 * (1 + np.cos(np.pi * min(1.0, step / arguments.steps)))))

    def bind(quantised):
        for key in keys:
            student._cache[key] = (
                fake_quantise(parameters[key], arguments.bits, arguments.group)
                if quantised else parameters[key])

    def per_state_error(got, want):
        axes = tuple(range(1, want.dim()))
        scale = (want ** 2).mean(dim=axes).clamp_min(1e-12)
        return ((got - want) ** 2).mean(dim=axes) / scale

    def loss_for(ids):
        bind(True)
        got = student.hidden_states(ids)
        with torch.no_grad():
            want = teacher.hidden_states(ids)
        # 🔴 RELATIVE PER STATE, BECAUSE THE RESIDUAL STREAM GROWS WITH DEPTH.
        # A plain MSE would put almost all of the gradient on the last blocks
        # for no reason but their magnitude, which is not what the mix says.
        return (mix * per_state_error(got, want)).sum()

    def evaluate():
        """The number the compression probe reports, on held-out sequences."""
        with torch.no_grad():
            ids = held_out()
            bind(True)
            got = student.hidden_states(ids)
            bind(False)
            want = student.hidden_states(ids)
            errors = [E.relative_rms(shim.single(got[:, row]),
                                     shim.single(want[:, row]))
                      for row in range(got.shape[1])]
            return float(np.mean(errors))

    print('mixed-single relRMS before training: %.4e' % evaluate(), flush=True)
    started, running = time.time(), None
    for step in range(arguments.steps):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_for(sampler())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        schedule.step()
        value = float(loss)
        running = value if running is None else 0.98 * running + 0.02 * value
        if (step + 1) % arguments.evaluate_every == 0 or step == 0:
            print('  step %5d/%d  loss %.4e  mixed-single relRMS %.4e  %.0fs'
                  % (step + 1, arguments.steps, running, evaluate(),
                     time.time() - started), flush=True)

    output = {}
    for key in keys:
        codes, scales, zeros, pad = export(parameters[key], arguments.bits,
                                           arguments.group)
        output[key + '.codes'] = codes
        output[key + '.scales'] = scales
        output[key + '.zeros'] = zeros
        output[key + '.pad'] = np.array([pad], np.int32)
        output[key + '.shape'] = np.array(list(parameters[key].shape), np.int32)
    output['__meta__'] = np.array(['distilled', str(arguments.bits),
                                   str(arguments.group), str(arguments.steps),
                                   'flat'])
    np.savez(arguments.out, **output)
    print('wrote %s' % arguments.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

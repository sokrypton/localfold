"""Is a codebook worth a new decoder? Scalar codes against vector ones, at 2 bits.

    python3 tools/esmc/vector-quantise.py --esmc esmc-600m --tensors 8

Everything else in this directory keeps LocalFold's existing shader: asymmetric
scalar codes, one float16 scale and zero per group. That format has a floor -
`docs/ESMFOLD2.md` measures two bits per weight destroying the tower - and the
only way under it is the representation the 2-bit language-model work uses:
quantise a short VECTOR of weights to an entry in a learned codebook, so the
codebook can put its levels where the weights actually are instead of on a
uniform grid.

This is the cheap version of that question. It reports RECONSTRUCTION error on
real ESM-C tensors at matched bits per weight, which is the number that says
whether the decoder work could possibly pay - not whether it does. A weight
error is an upper bound on what folds (this directory has the scars), so a
scheme that loses here loses, and a scheme that wins here still has to be
folded.

🔴 RATE IS log2(K)/d PLUS THE SCALE, AND COMPARING WITHOUT IT COMPARES NOTHING.
A 256-entry codebook over 4-dimensional vectors is 8 bits per 4 weights, so
2.00 bits a weight, plus one float16 scale per group of 64 for 0.25 more -
exactly the 2.25 that a 200 MiB bundle of the 600M pair allows.

🔴 AND THE ROTATION IS THE HALF THAT COSTS A SHADER. QuIP#'s advantage is not
the codebook alone: multiplying by a random orthogonal matrix first makes the
weights near-Gaussian and deletes the outliers that force a wide range. A
codebook is a table lookup and is CHEAPER to decode than unpacking five bits;
a rotation has to be undone on the activations at run time, which is a real
change to how a kernel reads its operand. So the two are reported separately -
if the codebook alone is enough, the expensive half is not needed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import esmc_forward as E          # noqa: E402
import quantisation as Q          # noqa: E402


def relative(candidate, original):
    return float(np.sqrt(((candidate - original) ** 2).sum()
                         / (original ** 2).sum()))


def hadamard(n):
    """The Walsh-Hadamard matrix of order n, normalised. n must be a power of 2."""
    if n & (n - 1):
        raise ValueError('%d is not a power of two' % n)
    matrix = np.ones((1, 1), np.float32)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(n)


def rotation(size, rng):
    """A signed Hadamard operator and its inverse, drawn ONCE.

    🔴 THE FORWARD AND THE INVERSE MUST SHARE THEIR SIGNS. The first version
    took an rng and drew a fresh sign vector inside each call, so the arm
    un-rotated with a different orthogonal matrix than it rotated with - and
    reported relRMS 1.377, worse than reconstructing the tensor as zero. That
    reads as "rotation does not help" and is a bug in the harness. A relRMS
    above 1 is never a property of a quantiser; it means the two halves of the
    comparison are not the same transform.
    """
    signs = rng.choice([-1.0, 1.0], size=size).astype(np.float32)
    operator = hadamard(size) * signs[None, :]
    return operator, operator.T


def apply_rotation(values, operator):
    size = operator.shape[0]
    pad = (-values.size) % size
    flat = np.concatenate([values.reshape(-1),
                           np.zeros(pad, np.float32)]) if pad else values.reshape(-1)
    out = (flat.reshape(-1, size) @ operator).reshape(-1)
    return out[:out.size - pad] if pad else out


def kmeans(vectors, clusters, iterations=25, seed=0):
    """Lloyd's algorithm on a sample. Plain, because the question is the RATE."""
    rng = np.random.default_rng(seed)
    centres = vectors[rng.choice(len(vectors), clusters, replace=False)].copy()
    for _ in range(iterations):
        # (n, k) distances without materialising the difference tensor
        norms = (centres ** 2).sum(1)
        assignment = np.argmin(
            norms[None, :] - 2.0 * (vectors @ centres.T), axis=1)
        for index in range(clusters):
            members = vectors[assignment == index]
            if len(members):
                centres[index] = members.mean(0)
    return centres


def vector_quantise(values, dimension, clusters, group, rng, sample=200000):
    """Per-group scale, then a codebook over `dimension`-long vectors.

    The scale is float16 and per group exactly as the scalar path's is, so the
    only thing that differs between the two arms is what the codes mean.
    """
    flat = np.asarray(values, np.float32).reshape(-1)
    pad = (-flat.size) % max(group, dimension)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    grouped = flat.reshape(-1, group)
    scales = (np.abs(grouped).max(1, keepdims=True) / 3.0).astype(np.float16)
    safe = scales.astype(np.float32)
    safe[safe == 0] = 1.0
    normalised = (grouped / safe).reshape(-1, dimension)

    chosen = normalised[rng.choice(len(normalised),
                                   min(sample, len(normalised)),
                                   replace=False)]
    codebook = kmeans(chosen, clusters)
    norms = (codebook ** 2).sum(1)
    assignment = np.argmin(norms[None, :] - 2.0 * (normalised @ codebook.T),
                           axis=1)
    rebuilt = (codebook[assignment].reshape(-1, group) * safe).reshape(-1)
    return rebuilt[:rebuilt.size - pad] if pad else rebuilt


def rotated_codebook(values, dimension, clusters, group, size, rng):
    forward, inverse = rotation(size, rng)
    turned = apply_rotation(np.asarray(values, np.float32), forward)
    coded = vector_quantise(turned, dimension, clusters, group, rng)
    return apply_rotation(coded[:turned.size], inverse)[:values.size]


def rotated_scalar(values, bits, group, size, rng):
    """The same rotation on the SCALAR path, so the rotation and the codebook
    are separable. Without this arm a win could be either one."""
    forward, inverse = rotation(size, rng)
    turned = apply_rotation(np.asarray(values, np.float32), forward)
    coded = Q.asymmetric(turned, bits, group)
    return apply_rotation(coded, inverse)[:values.size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--tensors', type=int, default=8)
    parser.add_argument('--group', type=int, default=64)
    parser.add_argument('--rotate', type=int, default=64,
                        help='Hadamard block size for the rotated arms')
    arguments = parser.parse_args()

    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    matrices = checkpoint.matrices()
    biggest = sorted(matrices, key=lambda item: -int(np.prod(item[1])))
    # Spread over the tower rather than taking the first block's four, so one
    # unusual layer cannot carry the answer.
    picked = [biggest[i] for i in
              np.linspace(0, len(biggest) - 1, arguments.tensors).astype(int)]
    print('%d tensors, %.1fM weights, group %d'
          % (len(picked), sum(np.prod(s) for _, s in picked) / 1e6,
             arguments.group))

    scalar_rate = lambda bits: bits + 32.0 / arguments.group
    vector_rate = lambda k, d: np.log2(k) / d + 16.0 / arguments.group
    arms = [
        ('scalar int3 g%d' % arguments.group, scalar_rate(3),
         lambda v, r: Q.asymmetric(v, 3, arguments.group)),
        ('scalar int2 g%d' % arguments.group, scalar_rate(2),
         lambda v, r: Q.asymmetric(v, 2, arguments.group)),
        ('codebook 256 x 4d', vector_rate(256, 4),
         lambda v, r: vector_quantise(v, 4, 256, arguments.group, r)),
        ('codebook 16 x 2d', vector_rate(16, 2),
         lambda v, r: vector_quantise(v, 2, 16, arguments.group, r)),
        ('codebook 256 x 4d, rotated', vector_rate(256, 4),
         lambda v, r: rotated_codebook(v, 4, 256, arguments.group,
                                       arguments.rotate, r)),
        ('codebook 1024 x 4d, rotated', vector_rate(1024, 4),
         lambda v, r: rotated_codebook(v, 4, 1024, arguments.group,
                                       arguments.rotate, r)),
        ('scalar int2 g64, rotated', scalar_rate(2),
         lambda v, r: rotated_scalar(v, 2, arguments.group, arguments.rotate, r)),
        ('codebook 1024 x 4d', vector_rate(1024, 4),
         lambda v, r: vector_quantise(v, 4, 1024, arguments.group, r)),
    ]

    header = '%-30s %8s %12s' % ('scheme', 'bits/w', 'relRMS')
    print('\n' + header)
    print('-' * len(header))
    for name, rate, apply in arms:
        errors = []
        for key, _ in picked:
            original = checkpoint[key].reshape(-1).astype(np.float32)
            rng = np.random.default_rng(0)
            errors.append(relative(apply(original, rng).reshape(-1), original))
        print('%-30s %8.2f %12.4f' % (name, rate, float(np.mean(errors))))

    print('\nA 200 MiB bundle of the 600M pair allows 2.25 bits a weight, and '
          'of the\n300M pair 3.33. Read the rows against those, not against '
          'each other.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

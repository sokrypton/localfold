"""Calibrated weight quantisation: GPTQ and an importance-weighted range search.

Round-to-nearest asks "what is the closest code to this weight". Both methods
here ask the better question - "what codes make this LAYER produce the same
OUTPUT on real protein sequences" - and they differ in how far they take it:

* `imatrix_quantise` keeps RTN's codes but chooses each group's RANGE to
  minimise the error the layer's own activations actually see. One number per
  input channel (its mean square over the calibration set), which is
  llama.cpp's importance matrix. Cheap, and it isolates "does knowing which
  channels matter help".
* `gptq_quantise` is GPTQ (Frantar et al., 2022): quantise one column at a
  time and push each column's rounding error into the columns not yet done,
  along the inverse Hessian of the layer's least-squares problem. It needs the
  full `H = 2 XᵀX`, and it is the method that made 3- and 4-bit LLMs usable.

🔴 NEITHER CHANGES THE STORAGE FORMAT, WHICH IS THE POINT. Both emit exactly
what `tools/quantize_af3.py` emits and `src/runtime/quantised-upload.js`
decodes - asymmetric codes, one float16 scale and one float16 zero per group of
32 - because a scheme that needs a new shader is a scheme that needs a new
shader. GPTQ's group axis lines up for free: LocalFold groups 32 CONSECUTIVE
elements of a row-major (out, in) tensor, which is 32 consecutive input
channels of one output channel, which is what GPTQ calls `group_size`.

🔴 AND `act-order` IS DELIBERATELY NOT IMPLEMENTED. Processing columns in
descending Hessian order is worth a few tenths of a bit, and it permutes the
input axis - so the groups are no longer 32 consecutive elements and the
decoder would need the permutation. That is the trade this file refuses.
"""
from __future__ import annotations

import numpy as np
import torch


def group_codes(values, bits, group, *, clip=None):
    """(rows, group) -> codes, scales, zeros. tools/quantize_af3.py's rule.

    🔴 THE SCALE AND ZERO ARE ROUNDED TO float16 BEFORE THE CODES ARE CHOSEN.
    Quantising against a scale the reader will not see puts a second, silent
    error on top of the first; it is the reader's value that decides what the
    weight becomes, so it is the reader's value the codes are fitted to.
    """
    levels = 2 ** bits - 1
    low = values.min(-1, keepdim=True).values
    high = values.max(-1, keepdim=True).values
    if clip is not None:
        middle = (high + low) / 2
        half = (high - low) / 2 * clip
        low, high = middle - half, middle + half
    zeros = low.half().float()
    scales = ((high - low) / levels).half().float()
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.clamp(torch.round((values - zeros) / safe), 0, levels)
    return codes, scales, zeros


def group_values(codes, scales, zeros):
    return codes * scales + zeros


def _as_groups(weight, group):
    """(out, in) -> (out * in/group, group), the order LocalFold packs in."""
    out, inner = weight.shape
    if inner % group:
        raise ValueError('input dimension %d is not a multiple of the group %d'
                         % (inner, group))
    return weight.reshape(out * inner // group, group)


def _packed(weight, codes, scales, zeros, group):
    """The three arrays a shard actually holds, in the shapes it holds them.

    🔴 A CALIBRATED SCHEME CANNOT BE RE-DERIVED FROM ITS OUTPUT. GPTQ takes a
    group's range from the ERROR-COMPENSATED columns, so running
    `tools/quantize_af3.py`'s min/max packer over the reconstructed weights
    picks a different range and gives different codes. The codes, scales and
    zeros have to be written out here and carried; that is the one thing
    shipping a calibrated bundle needs that shipping an RTN one does not.
    """
    out, inner = weight.shape
    return (codes.reshape(out, inner).to(torch.uint8),
            scales.reshape(out, inner // group).half(),
            zeros.reshape(out, inner // group).half())


def round_to_nearest(weight, bits, group, *, return_codes=False):
    grouped = _as_groups(weight, group)
    codes, scales, zeros = group_codes(grouped, bits, group)
    values = group_values(codes, scales, zeros).reshape(weight.shape)
    if not return_codes:
        return values
    return values, _packed(weight, codes, scales, zeros, group)


SEARCH_GRID = torch.arange(0.60, 1.0001, 0.02)


def imatrix_quantise(weight, importance, bits, group, grid=SEARCH_GRID,
                     *, return_codes=False):
    """Range search per group, scored by the layer's own activation energy.

    `importance` is one non-negative number per INPUT channel - `mean(x**2)`
    over the calibration set - so a group's error is weighted by how much
    current actually flows through each of its 32 columns.
    """
    out, inner = weight.shape
    grouped = _as_groups(weight, group)
    weights = importance.reshape(1, inner // group, group).expand(
        out, inner // group, group).reshape(-1, group)
    best_error, best = None, torch.empty_like(grouped)
    kept = None
    for clip in grid:
        codes, scales, zeros = group_codes(grouped, bits, group, clip=float(clip))
        candidate = group_values(codes, scales, zeros)
        error = (weights * (candidate - grouped) ** 2).sum(-1)
        if best_error is None:
            best_error, best, kept = error, candidate, [codes, scales, zeros]
        else:
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best[better] = candidate[better]
            for slot, new in zip(kept, (codes, scales, zeros)):
                slot[better] = new[better]
    values = best.reshape(weight.shape)
    if not return_codes:
        return values
    return values, _packed(weight, kept[0], kept[1], kept[2], group)


def gptq_quantise(weight, hessian, bits, group, *, damping=0.01, block=128,
                  return_codes=False):
    """GPTQ, in the group layout LocalFold already decodes.

    `weight` is (out, in) and `hessian` is (in, in) = 2 XᵀX over the
    calibration activations. Returns the reconstructed weights.
    """
    weight = weight.to(torch.float32).clone()
    out, inner = weight.shape
    hessian = hessian.to(torch.float32).clone()

    # A channel that never fires carries no information about its weights and
    # makes the Hessian singular; pin it and zero its column, as GPTQ does.
    dead = torch.diagonal(hessian) == 0
    hessian[dead, dead] = 1.0
    weight[:, dead] = 0.0

    average = torch.mean(torch.diagonal(hessian))
    index = torch.arange(inner, device=weight.device)
    hessian[index, index] += damping * average

    upper = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(upper)
    inverse = torch.linalg.cholesky(inverse, upper=True)

    result = torch.zeros_like(weight)
    all_codes = torch.zeros(out, inner, device=weight.device)
    all_scales = torch.zeros(out, inner // group, device=weight.device)
    all_zeros = torch.zeros(out, inner // group, device=weight.device)
    scales = torch.ones(out, 1, device=weight.device)
    zeros = torch.zeros(out, 1, device=weight.device)
    levels = 2 ** bits - 1

    for start in range(0, inner, block):
        stop = min(start + block, inner)
        columns = stop - start
        window = weight[:, start:stop].clone()
        quantised = torch.zeros_like(window)
        errors = torch.zeros_like(window)
        local = inverse[start:stop, start:stop]

        for i in range(columns):
            column = window[:, i]
            diagonal = local[i, i]
            if (start + i) % group == 0:
                # 🔴 THE GROUP'S RANGE IS TAKEN FROM THE ERROR-COMPENSATED
                # COLUMNS, NOT THE ORIGINAL ONES. By the time a group is
                # reached its weights have already absorbed the corrections of
                # every column before it, and fitting the range to the
                # untouched values would fit a tensor that is no longer being
                # quantised.
                span = window[:, i:i + group]
                if span.shape[1] < group:
                    span = torch.cat(
                        [span, weight[:, stop:stop + group - span.shape[1]]], 1)
                low = span.min(-1, keepdim=True).values
                high = span.max(-1, keepdim=True).values
                zeros = low.half().float()
                scales = ((high - low) / levels).half().float()
                scales = torch.where(scales == 0, torch.ones_like(scales), scales)
                all_scales[:, (start + i) // group] = scales[:, 0]
                all_zeros[:, (start + i) // group] = zeros[:, 0]

            code = torch.clamp(torch.round((column[:, None] - zeros) / scales),
                               0, levels)
            value = (code * scales + zeros)[:, 0]
            quantised[:, i] = value
            all_codes[:, start + i] = code[:, 0]
            error = (column - value) / diagonal
            window[:, i:] -= error[:, None] * local[i, i:][None, :]
            errors[:, i] = error

        result[:, start:stop] = quantised
        weight[:, stop:] -= errors @ inverse[start:stop, stop:]

    if not return_codes:
        return result
    return result, (all_codes.to(torch.uint8), all_scales.half(),
                    all_zeros.half())


def _self_test():
    """GPTQ must beat RTN where it claims to: correlated inputs, few bits.

    A differential check, not an oracle one - it says the implementation does
    what the method is for, on a problem whose answer is known by construction.
    """
    torch.manual_seed(0)
    inner, out, samples = 256, 128, 4096
    mix = torch.randn(inner, inner) / inner ** 0.5
    activations = torch.randn(samples, inner) @ mix        # correlated columns
    weight = torch.randn(out, inner) * 0.05
    hessian = 2.0 * (activations.T @ activations)
    importance = (activations ** 2).mean(0)
    reference = activations @ weight.T

    def output_error(candidate):
        got = activations @ candidate.T
        return float(((got - reference) ** 2).sum() / (reference ** 2).sum()) ** 0.5

    for bits in (3, 4):
        # The codes a shard would hold must reconstruct what the method
        # returned, or the number measured is not the number that would ship.
        values, (codes, scales, zeros) = gptq_quantise(
            weight, hessian, bits, 32, return_codes=True)
        rebuilt = (codes.float().reshape(out, -1, 32)
                   * scales.float()[:, :, None]
                   + zeros.float()[:, :, None]).reshape(out, inner)
        assert torch.equal(rebuilt, values), 'codes do not rebuild the weights'
        rtn = output_error(round_to_nearest(weight, bits, 32))
        mat = output_error(imatrix_quantise(weight, importance, bits, 32))
        gpt = output_error(gptq_quantise(weight, hessian, bits, 32))
        print('  %d bits: RTN %.4f  imatrix %.4f (%.2fx)  GPTQ %.4f (%.2fx)'
              % (bits, rtn, mat, rtn / mat, gpt, rtn / gpt))
        assert gpt < rtn, 'GPTQ did not beat round-to-nearest at %d bits' % bits


if __name__ == '__main__':
    print('gptq.py self-test, on a synthetic layer with correlated inputs:')
    _self_test()

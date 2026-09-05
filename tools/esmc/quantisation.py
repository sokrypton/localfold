"""The quantisation schemes, applied to a flat array of weights.

`asymmetric` is tools/quantize_af3.py's own packer, generalised over bits and
group size and given its inverse - the same one LocalFold ships AF3 and OpenBind
under, down to rounding the scale and the zero point to float16 BEFORE the codes
are chosen. That detail is not cosmetic: quantising against a scale the reader
will not see puts a second, silent error on top of the first.

Cost is quoted in BITS PER WEIGHT, never as "int4 against int5", because a
scheme costs its codes plus its metadata and at these group sizes the metadata
is most of the difference - an asymmetric group carries two float16, so int4 at
group 32 is 5.00 bits a weight and int5 at group 64 is 5.50.
"""
import numpy as np


def _grouped(values, group):
    padded = (-len(values)) % group
    grid = np.concatenate([values, np.zeros(padded, np.float32)]).reshape(-1, group)
    return grid, padded


def _ungroup(grid, padded, length):
    flat = grid.reshape(-1)
    return flat[:length] if padded else flat


def asymmetric(values, bits, group, search=None):
    """Per-group codes against a float16 scale and zero point, then back.

    `search` is a grid of range multipliers; each group keeps whichever
    minimises its own error. tools/analyse_quantisation.py measures that at
    3-4% on AF3's weights, which is why it is off by default.
    """
    levels = 2 ** bits - 1
    grid, padded = _grouped(values, group)
    low = grid.min(1, keepdims=True).astype(np.float32)
    high = grid.max(1, keepdims=True).astype(np.float32)
    best = None
    output = np.empty_like(grid)
    for clip in (search if search is not None else (1.0,)):
        middle = (high + low) / 2
        half = (high - low) / 2 * clip
        zeros = (middle - half).astype(np.float16).astype(np.float32)
        scales = ((2 * half) / levels).astype(np.float16).astype(np.float32)
        safe = np.where(scales == 0, np.float32(1), scales)
        codes = np.clip(np.rint((grid - zeros) / safe), 0, levels)
        candidate = codes * scales + zeros
        error = ((candidate - grid) ** 2).sum(1)
        if best is None:
            best, output = error, candidate
        else:
            better = error < best
            best = np.where(better, error, best)
            output[better] = candidate[better]
    return _ungroup(output, padded, len(values))


def symmetric(values, bits, group):
    """One float16 scale a group, no zero point - half the metadata."""
    levels = 2 ** (bits - 1) - 1
    grid, padded = _grouped(values, group)
    scales = (np.abs(grid).max(1, keepdims=True) / levels).astype(np.float16)
    safe = scales.astype(np.float32)
    safe[safe == 0] = 1.0
    codes = np.clip(np.rint(grid / safe), -levels - 1, levels)
    return _ungroup(codes * safe, padded, len(values))


def with_outliers(values, bits, group, count):
    """The `count` most extreme weights of a group kept in float16.

    They are replaced by the group's median before the range is taken, so one
    outlier cannot stretch the remaining levels across a span nothing visits.
    """
    grid, padded = _grouped(values, group)
    order = np.argsort(-np.abs(grid - grid.mean(1, keepdims=True)), axis=1)[:, :count]
    index = np.arange(len(grid))[:, None]
    kept = grid[index, order].astype(np.float16).astype(np.float32)
    body = grid.copy()
    body[index, order] = np.median(grid, axis=1, keepdims=True)
    output = asymmetric(body.reshape(-1), bits, group).reshape(grid.shape)
    output[index, order] = kept
    return _ungroup(output, padded, len(values))


def nm_sparse(values, n, m, bits, group, importance=None):
    """Keep the n largest of every m weights, quantise those, zero the rest.

    🔴 THE MASK IS NOT FREE AND IT IS MOST OF THE BUDGET AT HIGH SPARSITY.
    Saying "90% of the weights are zero" says nothing until you say how the
    reader learns WHICH. An arbitrary mask costs the binary entropy of the
    density - 0.54 bits a weight at one in eight - which at that density is
    more than the surviving values. n:m fixes the count per block instead, so
    the mask is log2(C(m, n))/m and is decodable without a search: 0.375 bits
    a weight at 1:8 against the entropy bound's 0.544.

    🔴 AND IT COSTS NO KERNEL. LocalFold already expands quantised weights into
    a dense float16 buffer in one dispatch (src/runtime/quantised-upload.js);
    a mask is the same shape of operation, so nothing downstream has to know
    the weights were ever sparse. That is the difference between this and the
    rotation a codebook scheme wants.

    `importance` is Wanda's criterion - the score is |w| times the root of the
    channel's activation energy, so a weight is kept for what it CARRIES, not
    for how large it is. Magnitude alone is the special case of importance
    being flat.
    """
    flat = np.asarray(values, np.float32).reshape(-1)
    pad = (-flat.size) % m
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    blocks = flat.reshape(-1, m)
    score = np.abs(blocks)
    if importance is not None:
        weight = np.asarray(importance, np.float32).reshape(-1)
        if weight.size != flat.size:
            weight = np.resize(weight, flat.size)
        score = score * np.sqrt(weight.reshape(-1, m))
    keep = np.argsort(-score, axis=1)[:, :n]
    rows = np.arange(len(blocks))[:, None]

    survivors = blocks[rows, keep].reshape(-1)
    approximate = asymmetric(survivors, bits, group).reshape(len(blocks), n)

    out = np.zeros_like(blocks)
    out[rows, keep] = approximate
    result = out.reshape(-1)
    return result[:result.size - pad] if pad else result


def nm_rate(n, m, bits, group):
    """bits per weight: the mask, plus the survivors and their metadata."""
    from math import comb, log2
    return log2(comb(m, n)) / m + (n / m) * (bits + 32.0 / group)


def sparse_scheme(n, m, bits, group=32):
    return Scheme('%d:%d int%d g%d' % (n, m, bits, group),
                  nm_rate(n, m, bits, group),
                  lambda v: nm_sparse(v, n, m, bits, group))


def to_bfloat16(values):
    u = np.ascontiguousarray(values, np.float32).view(np.uint32)
    rounded = (((u >> 16) + ((u >> 15) & 1)) & 0xFFFF).astype(np.uint32)
    return (rounded << 16).view(np.float32)


def to_float16(values):
    return values.astype(np.float16).astype(np.float32)


class Scheme:
    """A named arm: what it costs per weight, and what it does to a tensor.

    `keyed` schemes are handed the tensor's name as well, which is how a
    per-layer bit allocation is expressed without every other arm growing an
    argument it ignores.
    """

    def __init__(self, name, bits_per_weight, apply, keyed=False):
        self.name = name
        self.bits_per_weight = bits_per_weight
        self.apply = apply
        self.keyed = keyed

    def __call__(self, values, key=None):
        flat = np.asarray(values, np.float32).reshape(-1)
        if self.keyed and key is None:
            raise ValueError('%s needs the tensor name' % self.name)
        out = self.apply(flat, key) if self.keyed else self.apply(flat)
        return out.reshape(np.asarray(values).shape).astype(np.float32)


def layer_of(key):
    """The block index in an ESM-C tensor name, or None for the rest."""
    parts = key.split('.')
    if len(parts) > 3 and parts[1] == 'transformer' and parts[2] == 'blocks':
        return int(parts[3])
    return None


def mixed(spec, n_layers, group=32):
    """`mixed:4/6@6` - int4 everywhere but the last 6 blocks, which get int6.

    🔴 THE LAST BLOCKS ARE NOT WHERE THE MIX'S MASS IS, THEY ARE WHERE ITS MASS
    IS *READ*. ESMFold2 takes 59% of its softmax from the last three states, so
    spending bits there looks obvious - but every block feeds every later one,
    so a cheap early block damages the expensive late states too. That is the
    hypothesis this arm exists to test, not one it assumes.
    """
    low, high, count = spec
    def apply(values, key):
        layer = layer_of(key)
        bits = high if layer is not None and layer >= n_layers - count else low
        return asymmetric(values, bits, group)
    share = count / n_layers
    cost = (1 - share) * (low + 32 / group) + share * (high + 32 / group)
    return Scheme('int%d/%d last %d' % (low, high, count), cost, apply, keyed=True)


SEARCH_GRID = np.round(np.arange(0.6, 1.0001, 0.02), 3)


def calibrated(path, name=None):
    """An arm whose codes were chosen against real activations, not in a vacuum.

    `tools/esmc/calibrate-esmc.py` writes one `.npz` of codes, float16 scales
    and float16 zeros per tensor - the three arrays a shard holds - and this
    replays them. It is a lookup rather than a computation, which is the point:
    what is measured is what a bundle would contain.

    🔴 THE ARM MUST FAIL LOUDLY ON A TENSOR IT DOES NOT COVER. A calibrated
    file that silently fell back to round-to-nearest for the tensors it missed
    would measure a mixture and report it as a method.
    """
    store = np.load(path)
    meta = store['__meta__'] if '__meta__' in store else None
    method, bits, group = (meta[0], int(meta[1]), int(meta[2])) if meta is not None \
        else ('calibrated', 0, 32)

    def apply(values, key):
        codes = store[key + '.codes'].astype(np.float32)
        scales = store[key + '.scales'].astype(np.float32)
        zeros = store[key + '.zeros'].astype(np.float32)
        if key + '.pad' in store:
            # 🔴 THE FLAT LAYOUT, which is what tools/quantize_af3.py writes
            # and what a group that does not divide the row length forces.
            # Distinguished by the presence of a pad, not by a name: a file
            # read under the wrong layout is a permutation of the right
            # weights and reports as a broken model rather than a broken read.
            pad = int(store[key + '.pad'][0])
            flat = (codes.reshape(-1, group) * scales[:, None]
                    + zeros[:, None]).reshape(-1)
            return flat[:flat.size - pad] if pad else flat
        rows, inner = codes.shape
        grouped = codes.reshape(rows, inner // group, group)
        return (grouped * scales[:, :, None] + zeros[:, :, None]).reshape(-1)

    label = name or '%s int%d g%d' % (method, bits, group)
    return Scheme(label, bits + 32.0 / group, apply, keyed=True)


def codebook(path, name=None):
    """An arm whose codes index a shared table, written by fit-codebook.py.

    🔴 THE TABLE IS THE WHOLE DIFFERENCE AND IT IS 8 KB. Everything else here
    reconstructs a weight arithmetically - code times scale plus zero - and
    this looks it up. That is CHEAPER to decode than unpacking five bits from a
    twenty-byte group, so the shader this would need is simpler than the one
    LocalFold already runs; what it is not is the same shader.
    """
    store = np.load(path)
    meta = store['__meta__']
    clusters, dimension, group = int(meta[1]), int(meta[2]), int(meta[3])
    table = store['__codebook__'].astype(np.float32)

    def apply(values, key):
        codes = store[key + '.codes']
        scales = store[key + '.scales'].astype(np.float32)
        pad = int(store[key + '.pad'][0])
        rebuilt = (table[codes].reshape(-1, group)
                   * scales[:, None]).reshape(-1)
        return rebuilt[:rebuilt.size - pad] if pad else rebuilt

    rate = np.log2(clusters) / dimension + 16.0 / group
    label = name or 'codebook %dx%dd g%d' % (clusters, dimension, group)
    return Scheme(label, float(rate), apply, keyed=True)


def catalogue():
    """The arms, cheapest storage last, so a table reads down to the frontier."""
    def asym(bits, group, search=None):
        return lambda v: asymmetric(v, bits, group, search)

    out = [
        Scheme('float32', 32.00, lambda v: v),
        Scheme('bfloat16', 16.00, to_bfloat16),
        Scheme('float16', 16.00, to_float16),
        Scheme('int8 g64 sym', 8.25, lambda v: symmetric(v, 8, 64)),
        Scheme('int8 g128 asym', 8.25, asym(8, 128)),
        Scheme('int6 g64 asym', 6.50, asym(6, 64)),
        Scheme('int6 g32 asym', 7.00, asym(6, 32)),
        Scheme('int5 g32 asym', 6.00, asym(5, 32)),
        Scheme('int5 g64 asym', 5.50, asym(5, 64)),
        Scheme('int5 g32 asym+search', 6.00, asym(5, 32, SEARCH_GRID)),
        Scheme('int5 g64 asym+2 outliers', 5.69,
               lambda v: with_outliers(v, 5, 64, 2)),
        Scheme('int4 g32 asym', 5.00, asym(4, 32)),
        Scheme('int4 g32 asym+search', 5.00, asym(4, 32, SEARCH_GRID)),
        Scheme('int4 g64 asym+2 outliers', 5.19,
               lambda v: with_outliers(v, 4, 64, 2)),
        Scheme('int4 g64 asym', 4.50, asym(4, 64)),
        Scheme('int3 g64 asym', 3.50, asym(3, 64)),
        Scheme('int3 g128 asym', 3.25, asym(3, 128)),
        Scheme('int2 g64 asym', 2.50, asym(2, 64)),
        Scheme('int2 g128 asym', 2.25, asym(2, 128)),
        Scheme('int3 g32 asym', 4.00, asym(3, 32)),
        Scheme('int3 g32 asym+search', 4.00, asym(3, 32, SEARCH_GRID)),
        Scheme('int2 g32 asym', 3.00, asym(2, 32)),
        # Sparse arms, at rates the dense frontier cannot reach. Read them
        # against a DENSE arm of the same bits/weight, never against each other.
        sparse_scheme(2, 4, 8), sparse_scheme(2, 4, 6), sparse_scheme(2, 4, 4),
        sparse_scheme(1, 4, 8), sparse_scheme(1, 4, 6), sparse_scheme(1, 4, 4),
        sparse_scheme(2, 8, 6), sparse_scheme(2, 8, 4),
        sparse_scheme(1, 8, 8), sparse_scheme(1, 8, 6),
    ]
    return {s.name: s for s in out}

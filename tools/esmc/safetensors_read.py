"""Read a .safetensors file into numpy arrays, without the safetensors package.

This machine's Python is PEP 668 managed and refuses `pip install safetensors`,
and the format does not need a dependency: eight bytes of little-endian header
length, that many bytes of JSON naming each tensor's dtype, shape and byte
range, then the payload. Reading it here also hands back plain numpy rather
than torch tensors, which is what the quantisation work wants anyway.

🔴 THE OFFSETS ARE RELATIVE TO THE END OF THE HEADER, not to the file. An
absolute reading gives every tensor the right shape and the wrong contents,
which looks like a broken model rather than a broken reader.
"""
import json
import mmap
import numpy as np

# safetensors names its dtypes itself; only what these checkpoints use is here,
# so an unexpected one raises rather than being guessed at.
DTYPES = {
    "F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"), "U8": np.dtype("<u1"), "BOOL": np.dtype("?"),
}


class SafeTensors:
    """Lazy: the header is read at open, a tensor's bytes only when asked for."""

    def __init__(self, path):
        self.path = str(path)
        self._file = open(self.path, "rb")
        length = int.from_bytes(self._file.read(8), "little")
        self.header = json.loads(self._file.read(length))
        self._base = 8 + length
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.metadata = self.header.pop("__metadata__", {})

    def keys(self):
        return self.header.keys()

    def __contains__(self, name):
        return name in self.header

    def __len__(self):
        return len(self.header)

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def __getitem__(self, name):
        entry = self.header[name]
        dtype = DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError("unhandled safetensors dtype %r for %s"
                             % (entry["dtype"], name))
        start, end = entry["data_offsets"]
        raw = self._map[self._base + start:self._base + end]
        return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])

    def close(self):
        self._map.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load(path, prefix=""):
    """-> {name: array} for every tensor, materialised. Convenience for scripts."""
    with SafeTensors(path) as f:
        return {k: np.array(f[k]) for k in f.keys() if k.startswith(prefix)}

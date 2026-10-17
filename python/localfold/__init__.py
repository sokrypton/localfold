"""LocalFold's native ports - CUDA on Linux, Metal on Apple silicon: localfold-af3, localfold-af2 and localfold-ef2
(see `localfold-af3 --help`)."""
import os

__version__ = "0.1.0"
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def binary(name):
    """The path of a bundled binary: "af3", "af2", "ef2" or "fetch"."""
    return os.path.join(BIN, f"localfold-{name}")

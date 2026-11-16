"""LocalFold's native ports - CUDA on Linux, Metal on Apple silicon: cf-af3, cf-af2 and cf-ef2
(see `cf-af3 --help`)."""
import os

__version__ = "0.1.0"
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def binary(name):
    """The path of a bundled binary: "af3", "af2", "ef2" or "fetch"."""
    return os.path.join(BIN, f"cf-{name}")

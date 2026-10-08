"""The console commands: each replaces this process with its native binary, arguments untouched."""
import os
import sys

from . import binary


def _run(name):
    path = binary(name)
    if not os.access(path, os.X_OK):
        sys.exit(f"localfold: {path} is missing or not executable - reinstall the localfold wheel")
    os.execv(path, [path, *sys.argv[1:]])


def af3():
    _run("af3")


def af2():
    _run("af2")


def ef2():
    _run("ef2")

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


def fetch():
    _run("fetch")


def serve():
    from .serve import main
    main()


def main():
    """`localfold <command>`: serve, or one of the ports by name (af3, af2, ef2, fetch)."""
    commands = {"af3": af3, "af2": af2, "ef2": ef2, "fetch": fetch}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: localfold serve [--port N] [--no-open]   the website on this machine, folding natively\n"
                 "       localfold af3|af2|ef2|fetch ...          a port's command (as localfold-af3 ...)")
    name = sys.argv.pop(1)
    if name == "serve":
        from .serve import main as serve_main
        return serve_main()
    if name not in commands:
        sys.exit(f"localfold: no command {name!r} (serve, af3, af2, ef2, fetch)")
    commands[name]()

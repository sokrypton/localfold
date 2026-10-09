"""`localfold serve`: the LocalFold website on this machine, folding with this machine's GPU.

    localfold serve                 # http://127.0.0.1:8710, the browser opened on it
    localfold serve --port 9000 --no-open

The page is the website's own (built into the wheel as localfold/site), served on 127.0.0.1 by
localfold/server.py, and every fold it asks for goes to localfold/worker.py - the native port this wheel carries,
Metal on Apple silicon and CUDA on Linux - instead of the browser's WebGPU. The weights are the binaries' own,
fetched once into ~/.cache/localfold (or --weights-dir), and the page holds none.

A token generated at start is in the link it opens; every request to the server carries it, so another page in the
same browser cannot spend this machine's GPU. Ctrl-C, or Disconnect on the page, stops it.
"""
import argparse
import os
import socket
import sys

from . import BIN

HERE = os.path.dirname(os.path.abspath(__file__))
# the featuriser binary's tools, by the names it answers to (cuda/featurise/featurise.cpp: argv[0])
FEATURISERS = ("af3-featurise", "af2-featurise", "ef2-featurise", "resolve-templates", "fetch-weights")


def site_dir():
    """The built page: the wheel's, or - run from a checkout - the checkout itself."""
    packaged = os.path.join(HERE, "site")
    if os.path.exists(os.path.join(packaged, "index.html")):
        return packaged
    checkout = os.path.normpath(os.path.join(HERE, "..", ".."))
    if os.path.exists(os.path.join(checkout, "index.html")):
        return checkout
    sys.exit("localfold serve: this install has no built page (localfold/site) - reinstall the localfold wheel")


def featurisers(cache):
    """A directory naming the bundled featuriser as each of its tools: links to bin/localfold-fetch, which is the one
    featuriser binary (a wheel cannot carry the links themselves). Made once a version."""
    from . import __version__
    target = os.path.join(BIN, "localfold-fetch")
    if not os.access(target, os.X_OK):
        sys.exit(f"localfold serve: {target} is missing - reinstall the localfold wheel")
    where = os.path.join(cache, f"featurise-{__version__}")
    os.makedirs(where, exist_ok=True)
    for name in FEATURISERS:
        link = os.path.join(where, name)
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    return where


def free_port(host, wanted):
    """`wanted` if nothing holds it, else the next free one (a second `localfold serve`, an old one still up)."""
    for port in range(wanted, wanted + 50):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    sys.exit(f"localfold serve: no free port from {wanted}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="localfold serve", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--host", default="127.0.0.1",
                        help="what to bind (default loopback: only this machine reaches it)")
    parser.add_argument("--weights-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "localfold"),
                        help="where the models' weights are kept (default ~/.cache/localfold)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--token", default=None, help="the shared secret (default: a new one each start)")
    arguments = parser.parse_args(argv)

    cache = os.path.abspath(os.path.expanduser(arguments.weights_dir))
    os.makedirs(cache, exist_ok=True)
    os.environ.update({
        "LOCALFOLD_SITE_DIR": site_dir(),
        "LOCALFOLD_BIN_DIR": BIN,
        "LOCALFOLD_FEATURISE_DIR": featurisers(cache),
        "LOCALFOLD_WEIGHTS_DIR": cache,
        "LOCALFOLD_REPO": cache,               # (the worker's working directory: nothing of a checkout is read)
    })
    port = free_port(arguments.host, arguments.port)
    # (imported now: the server reads where it serves from the environment above)
    from . import server
    sys.argv = ["localfold serve", "--native", "--host", arguments.host, "--port", str(port),
                *(["--token", arguments.token] if arguments.token else []),
                *([] if arguments.no_open else ["--open"])]
    server.main()


if __name__ == "__main__":
    main()

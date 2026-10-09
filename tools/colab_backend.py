"""Moved: python/localfold/server.py (it serves a reader's own machine too, not only Colab). Kept for notebooks saved
before the move, which run this path; it runs the server with the same arguments."""
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "localfold", "server.py")
runpy.run_path(sys.argv[0], run_name="__main__")

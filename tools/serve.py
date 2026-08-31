#!/usr/bin/env python3
"""Serve the checkout for development, with caching turned off.

    python3 tools/serve.py            # http://127.0.0.1:8000
    python3 tools/serve.py 8402       # ...on another port

🔴 WHY NOT `python3 -m http.server`. Its responses are cacheable, and a browser
holding an old copy of web/app.js is indistinguishable from a change that did
not work. That has now cost three wrong conclusions in one sitting: a flag that
looked unwired, a dropdown that looked inert, and a fold reported 8 pLDDT lower
than the code actually produces. Each time the code was already correct and the
page was not running it.

The workaround was to restart on a fresh port, which changes every module URL.
That works and is easy to forget, so this sends no-store instead: every reload
re-fetches, and a reload is enough to see an edit.

Static files only, bound to the loopback address. Not a production server.
"""
import functools
import http.server
import socketserver
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # ...errors only. A fold pulls hundreds of weight shards and the log
        # buries anything worth reading.
        if not args or str(args[1]).startswith("2"):
            return
        super().log_message(fmt, *args)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(NoCacheHandler, directory=".")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        print(f"serving . at http://127.0.0.1:{port}/  (no-store; reload picks up edits)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

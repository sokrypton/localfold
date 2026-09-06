#!/usr/bin/env python3
"""Serve the checkout with caching turned off.

    python3 tools/serve.py [port]

🔴 `python3 -m http.server` SENDS NO CACHE HEADERS, AND THAT LOOKS EXACTLY LIKE
A BROKEN FEATURE. Chrome caches `web/app.js` and every other ES module
heuristically, and `location.reload()` does not refetch them - so a change lands,
the page is reloaded, nothing happens, and the code looks wrong. Three separate
sessions have lost time to it, and `tools/fold-in-page.py` never reproduces it
because it launches a fresh profile every run: the tool passes while the browser
in front of you does not.

This sends `Cache-Control: no-store` on everything, which is the one header that
makes a reload a refetch.

🔴 EXCEPT THE WEIGHTS, WHICH MUST STILL CACHE. A model bundle is 346 MiB and
re-downloading it on every reload would make the page unusable to develop
against - which is the opposite of the problem this solves. Shards are served
with a long max-age instead; they are content-addressed by the manifest and a
re-export changes their names.
"""
import functools
import http.server
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Anything a fold streams rather than something a developer edits.
CACHEABLE = (".bin", ".safetensors", ".zst", ".gz")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.split("?")[0].endswith(CACHEABLE):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # ...only failures, so a fold's thousands of shard reads do not bury them.
        if not args or not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
            super().log_message(fmt, *args)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    handler = functools.partial(Handler, directory=str(ROOT))
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {ROOT} at http://127.0.0.1:{port}/index.html")
        print("  modules: no-store, so a plain reload picks up an edit")
        print(f"  weights: cached for a year ({', '.join(CACHEABLE)})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

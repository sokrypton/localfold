"""
Serve the repo with caching switched off.

🔴 CHROME CACHES app.js AND A RELOAD DOES NOT ALWAYS CLEAR IT. python's
http.server sends Last-Modified and no Cache-Control, so Chrome is free to
reuse a module heuristically - `performance.getEntriesByType('resource')`
reports transferSize 0 for app.js while the CSS beside it revalidates. The
symptom is not an error: the page runs the PREVIOUS build, so a change appears
to have had no effect and the next hour goes on debugging code that is correct.

    python3 tools/serve-nocache.py [port]
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"serving {port} with no-store", flush=True)
    ThreadingHTTPServer(("", port), NoCache).serve_forever()

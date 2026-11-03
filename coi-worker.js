/**
 * Adds the two headers GitHub Pages cannot send, so a CPU fold can share memory
 * between threads. Registered only when a reader chooses the CPU (web/settings-panel.js)
 * and removed when they choose otherwise.
 *
 * `credentialless` rather than `require-corp`, because the page takes a stylesheet and
 * scripts from a CDN that does not send a resource policy. Safari has no credentialless
 * and stays unisolated, where the fold runs on one thread.
 */
self.addEventListener("install", () => { self.skipWaiting(); });
self.addEventListener("activate", (event) => { event.waitUntil(self.clients.claim()); });
self.addEventListener("fetch", (event) => {
  const { request } = event;
  // A cross-origin fetch keeps its own CORS answer; only this origin's responses change.
  if (new URL(request.url).origin !== self.location.origin) return;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith(fetch(request).then((response) => {
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Embedder-Policy", "credentialless");
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    return new Response(response.body, {
      status: response.status, statusText: response.statusText, headers,
    });
  }));
});

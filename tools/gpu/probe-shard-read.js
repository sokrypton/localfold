/**
 * How fast can this browser get a weight shard into an ArrayBuffer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-shard-read.js --bundle=/model
 *
 * 🔴 BECAUSE `HttpTensorStore` READS EVERY SHARD THROUGH A STREAM, and the
 * reason is the progress dial: a page downloading 472 MiB over a network wants
 * to move it as the bytes arrive, so `#readStream` pulls chunks and reports
 * each one. That costs something, and nothing had measured what. A raw parallel
 * fetch of AF2's eight shards from this harness's own server is 102 MB in 61 ms
 * outside the browser; inside it, the store's stream path measures 227 MB/s.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model");
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();
  const files = [...new Set(Object.values(manifest.tensors).map((r) => r.file))].sort();

  const arms = {};
  // 🔴 EVERY ARM FETCHES FRESH, and the browser's HTTP cache would make the
  // second one look free - so each carries its own query string.
  const stamp = Date.now();
  const time = async (label, read) => {
    const at = performance.now();
    const sizes = await Promise.all(files.map((file, index) =>
      read(`${bundle}/${file}?${label}=${stamp}-${index}`)));
    const bytes = sizes.reduce((total, size) => total + size, 0);
    const ms = performance.now() - at;
    arms[label] = { ms: Number(ms.toFixed(1)), megabytesPerSecond: Number((bytes / 1e6 / (ms / 1e3)).toFixed(0)) };
    return bytes;
  };

  const bytes = await time("arraybuffer", async (url) => {
    const response = await fetch(url);
    return (await response.arrayBuffer()).byteLength;
  });
  // ...and the shape the store actually uses: pull chunks into a pre-sized
  // buffer, which is what lets it report progress as the bytes arrive.
  await time("stream", async (url) => {
    const response = await fetch(url);
    const reader = response.body.getReader();
    const total = Number(response.headers.get("content-length") ?? 0);
    const output = new Uint8Array(total);
    let offset = 0;
    let chunks = 0;
    while (true) {
      // eslint-disable-next-line no-await-in-loop
      const { done, value } = await reader.read();
      if (done) break;
      output.set(value, offset);
      offset += value.byteLength;
      chunks += 1;
    }
    arms.chunks = (arms.chunks ?? 0) + chunks;
    return offset;
  });

  return { bundle, shards: files.length, megabytes: Number((bytes / 1e6).toFixed(1)), arms };
}

import { deflateRawSync, gzipSync } from "node:zlib";
import { describe, expect, it } from "./harness.js";
import { gunzip, inflateRaw } from "../shared/input/inflate.js";
import { extractMmseqs2A3m, generateMmseqs2Msa } from "../shared/input/mmseqs2-api.js";
import { readZip, writeZip } from "../web/zip.js";

/**
 * Every path that decompresses on the page works without `DecompressionStream`.
 *
 * 🔴 BECAUSE A READER ON AN OLDER MAC REPORTED ONE OF THEM. Safari gained
 * `DecompressionStream` in 16.4; before that an MSA search ended in "This
 * browser cannot decompress the MMseqs2 result" - and the other two doors, a
 * dropped fold archive and a saved session, failed the same way for the same
 * reason. The decoder was already in the tree for the design page's `.cif.gz`.
 *
 * 🔴 AND THE ARM IS FORCED, BECAUSE IT RUNS NOWHERE ELSE. Node has the API and
 * so does every browser this repository is measured in, so the fallback is
 * reached only where nothing runs - the same reason web/zip.js's stored arm is
 * forced in test/zip.test.js. The bar is the PLATFORM's own answer: not that
 * the fallback returns something plausible, but that it returns the same bytes
 * the browser with the API would have returned.
 */
async function withoutDecompressionStream(body) {
  const present = globalThis.DecompressionStream;
  // ...a guard rather than an assumption: on a runtime that never had it, the
  // two arms are the same arm and this file would be comparing nothing.
  expect(typeof present).toBe("function");
  delete globalThis.DecompressionStream;
  // ...and the delete took, or every arm below runs the native path twice and
  // compares it with itself.
  expect(typeof globalThis.DecompressionStream).toBe("undefined");
  try {
    return await body();
  } finally {
    globalThis.DecompressionStream = present;
  }
}

function tar(files) {
  const chunks = [];
  for (const [name, value] of Object.entries(files)) {
    const header = new Uint8Array(512);
    header.set(new TextEncoder().encode(name), 0);
    header.set(new TextEncoder().encode(`${value.length.toString(8).padStart(11, "0")}\0`), 124);
    header[156] = 48;
    const data = new TextEncoder().encode(value);
    const padded = new Uint8Array(Math.ceil(data.length / 512) * 512);
    padded.set(data);
    chunks.push(header, padded);
  }
  chunks.push(new Uint8Array(1024));
  const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

// An alignment shaped like a search result: long runs of gaps, so the archive
// deflates rather than falling through to a stored member, and long enough to
// need back references past the first window of literals.
const QUERY = "ACDEFGHIKLMNPQRSTVWY".repeat(6);
const alignment = `>101\n${QUERY}\n`
  + Array.from({ length: 80 }, (_, row) => {
    const gaps = row % 40;
    return `>hit_${row} OS=Something OX=${row}\n`
      + `${"-".repeat(gaps)}${QUERY.slice(gaps)}\n`;
  }).join("");

const resultTar = tar({
  "uniref.a3m": alignment,
  "bfd.mgnify30.metaeuk30.smag30.a3m": alignment.replace(/hit_/g, "env_"),
});

async function platformInflate(bytes, format) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

describe("decompression without DecompressionStream", () => {
  it("gunzips what the platform gunzips, byte for byte", async () => {
    for (const level of [0, 1, 6, 9]) {
      const packed = new Uint8Array(gzipSync(resultTar, { level }));
      expect([...gunzip(packed)]).toEqual([...await platformInflate(packed, "gzip")]);
    }
  });

  it("inflates a raw DEFLATE member as the platform does", async () => {
    for (const payload of [resultTar, new Uint8Array(0), Uint8Array.from([7]),
                           new TextEncoder().encode(alignment)]) {
      const packed = new Uint8Array(deflateRawSync(payload));
      expect([...inflateRaw(packed)]).toEqual([...await platformInflate(packed, "deflate-raw")]);
    }
  });

  it("searches, downloads and reads an MMseqs2 result", async () => {
    const archive = gzipSync(resultTar);
    const search = async () => {
      const responses = [
        new Response(JSON.stringify({ status: "COMPLETE", id: "ticket-1" })),
        new Response(new Uint8Array(archive)),
      ];
      // No `decompress` option: the page's own default is what is under test,
      // and every other test in this suite replaces it with a stub.
      return generateMmseqs2Msa(QUERY, {
        fetchImplementation: async () => responses.shift(), wait: async () => {},
      });
    };
    const withApi = await search();
    const withoutApi = await withoutDecompressionStream(search);
    expect(withoutApi.a3m).toBe(withApi.a3m);
    expect(withoutApi.a3m).toBe(extractMmseqs2A3m(resultTar));
    expect(withoutApi.depth).toBe(withApi.depth);
    expect(withoutApi.depth > 1).toBe(true);
  });

  it("reads back a deflated fold archive", async () => {
    const members = [["job.json", '{"name":"a fold"}'],
                     ["msas/unpaired_chains_a.a3m", alignment]];
    // Written WITH the API and read without it, which is the case that matters:
    // such a browser writes stored archives, so its own round trip was never
    // the broken half - what it could not read was everybody else's archive.
    const deflated = await writeZip(members);
    const stored = await writeZip(members, { store: true });
    expect(deflated.length < stored.length).toBe(true);
    const files = await withoutDecompressionStream(() => readZip(deflated));
    for (const [name, content] of members) expect(files.get(name)).toBe(content);
  });
});

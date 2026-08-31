import { parseA3m } from "./a3m.js";
import { mergeChainA3ms, mergeUnpairedChainA3ms } from "./chains.js";

const DEFAULT_API_URL = "https://api.colabfold.com";
const QUERY_ID = 101;
const TRANSIENT_SUBMISSION = new Set(["UNKNOWN", "RATELIMIT"]);
const TRANSIENT_JOB = new Set(["UNKNOWN", "PENDING", "RUNNING", "RATELIMIT"]);
































function normalizedSequence(sequence) {
  const value = sequence.replace(/\s+/g, "").toUpperCase();
  if (!/^[ARNDCQEGHILKMFPSTWYVX]+$/.test(value)) {
    throw new Error("Sequence must contain only standard amino-acid letters or X");
  }
  return value;
}

function abortError() {
  return new DOMException("MMseqs2 search was cancelled", "AbortError");
}

async function waitWithAbort(milliseconds, signal) {
  if (signal?.aborted === true) throw abortError();
  await new Promise ((resolve, reject) => {
    const onAbort = () => { clearTimeout(timer); reject(abortError()); };
    const timer = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(); }, milliseconds);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function parseTicket(value) {
  const status = typeof value.status === "string" ? value.status.toUpperCase() : "ERROR";
  return { status, ...(typeof value.id === "string" && value.id !== "" ? { id: value.id } : {}) };
}

async function request(fetchImplementation, url, init, label) {
  let lastError;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const response = await fetchImplementation(url, init);
      if (response.ok) return response;
      if (![429, 500, 502, 503, 504].includes(response.status)) {
        lastError = new Error(`${label} failed: HTTP ${response.status}`);
        break;
      }
      lastError = new Error(`${label} failed: HTTP ${response.status}`);
    } catch (error) {
      if (init.signal?.aborted === true) throw abortError();
      lastError = error;
    }
    await waitWithAbort(500 * 2 ** attempt, init.signal ?? undefined);
  }
  throw lastError instanceof Error ? lastError : new Error(`${label} failed`);
}

function tarString(bytes, start, length) {
  const end = bytes.indexOf(0, start);
  return new TextDecoder().decode(bytes.subarray(start, end < 0 || end > start + length ? start + length : end));
}

function tarOctal(bytes, start, length) {
  const value = tarString(bytes, start, length).trim().replace(/\0/g, "");
  const parsed = Number.parseInt(value || "0", 8);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error("MMseqs2 result contains an invalid tar entry size");
  return parsed;
}

/** Reads regular files from the small POSIX tar archive returned by the ColabFold API. */
export function readTarFiles(bytes) {
  const files = new Map ();
  for (let offset = 0; offset + 512 <= bytes.byteLength;) {
    const header = bytes.subarray(offset, offset + 512);
    if (header.every((value) => value === 0)) break;
    const name = tarString(header, 0, 100);
    const prefix = tarString(header, 345, 155);
    const path = prefix === "" ? name : `${prefix}/${name}`;
    const size = tarOctal(header, 124, 12);
    const dataStart = offset + 512; const dataEnd = dataStart + size;
    if (path === "" || dataEnd > bytes.byteLength) throw new Error("MMseqs2 result contains a truncated tar entry");
    const type = header[156];
    if (type === 0 || type === 48) files.set(path.replace(/^\.\//, ""), bytes.slice(dataStart, dataEnd));
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  return files;
}

function queryBlock(contents, queryId) {
  for (const block of contents.replace(/\r/g, "").split(/\0+/)) {
    const trimmed = block.trim();
    if (new RegExp(`^>${queryId}(?:\\s|$)`).test(trimmed)) return trimmed;
  }
  throw new Error(`MMseqs2 result does not contain query ${queryId}`);
}

/** Extracts and combines the UniRef and environmental A3Ms exactly as ColabFold does. */
export function extractMmseqs2A3m(tarBytes, useEnvironmental = true) {
  const files = readTarFiles(tarBytes);
  const find = (suffix) =>
    [...files].find(([path]) => path === suffix || path.endsWith(`/${suffix}`))?.[1];
  const uniref = find("uniref.a3m");
  if (uniref === undefined) throw new Error("MMseqs2 result is missing uniref.a3m");
  const blocks = [queryBlock(new TextDecoder().decode(uniref), QUERY_ID)];
  if (useEnvironmental) {
    const environmental = find("bfd.mgnify30.metaeuk30.smag30.a3m");
    if (environmental === undefined) throw new Error("MMseqs2 result is missing the environmental A3M");
    blocks.push(queryBlock(new TextDecoder().decode(environmental), QUERY_ID));
  }
  return `${blocks.join("\n")}\n`;
}

async function decompressGzip(archive) {
  if (typeof DecompressionStream === "undefined") throw new Error("This browser cannot decompress the MMseqs2 result");
  const stream = new Blob([archive]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Generates a monomer MSA through the public ColabFold MMseqs2 API. */
export async function generateMmseqs2Msa(sequenceValue,
  options= {}) {
  const sequence = normalizedSequence(sequenceValue);
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const wait = options.wait ?? waitWithAbort;
  const decompress = options.decompress ?? decompressGzip;
  const useEnvironmental = options.useEnvironmental ?? true;
  const apiUrl = new URL(options.apiUrl ?? DEFAULT_API_URL); if (!apiUrl.pathname.endsWith("/")) apiUrl.pathname += "/";
  const start = performance.now();
  const report = (phase, status, ticket) => options.onProgress?.({
    phase, status, ...(ticket === undefined ? {} : { ticket }), elapsedMilliseconds: performance.now() - start,
  });
  const signal = options.signal;
  const body = new URLSearchParams({ q: `>${QUERY_ID}\n${sequence}\n`, mode: useEnvironmental ? "env" : "all" });
  let ticket;
  let status = "UNKNOWN";
  while (ticket === undefined) {
    report("submitting", "SUBMIT");
    const response = await request(fetchImplementation, new URL("ticket/msa", apiUrl), {
      method: "POST", body, ...(signal === undefined ? {} : { signal }),
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
    }, "MMseqs2 submission");
    const submitted = parseTicket(await response.json()); status = submitted.status; ticket = submitted.id;
    if (TRANSIENT_SUBMISSION.has(status)) {
      ticket = undefined; report("retrying", status); await wait(5_000 + Math.floor(Math.random() * 5_000), signal); continue;
    }
    if (status === "ERROR") throw new Error("MMseqs2 rejected the sequence or is temporarily unavailable");
    if (status === "MAINTENANCE") throw new Error("The MMseqs2 API is undergoing maintenance; try again later");
    if (ticket === undefined) throw new Error(`MMseqs2 returned ${status} without a ticket`);
  }
  while (TRANSIENT_JOB.has(status)) {
    report(status === "RUNNING" ? "running" : "queued", status, ticket);
    await wait(5_000 + Math.floor(Math.random() * 5_000), signal);
    const response = await request(fetchImplementation, new URL(`ticket/${encodeURIComponent(ticket)}`, apiUrl),
      signal === undefined ? {} : { signal }, "MMseqs2 status");
    status = parseTicket(await response.json()).status;
  }
  if (status !== "COMPLETE") throw new Error(`MMseqs2 search ended with status ${status}`);
  report("downloading", status, ticket);
  const response = await request(fetchImplementation,
    new URL(`result/download/${encodeURIComponent(ticket)}`, apiUrl),
    signal === undefined ? {} : { signal }, "MMseqs2 result download");
  const a3m = extractMmseqs2A3m(await decompress(await response.arrayBuffer()), useEnvironmental);
  const alignment = parseA3m(a3m);
  if (alignment.query !== sequence) throw new Error("MMseqs2 returned an A3M for a different query sequence");
  const elapsedMilliseconds = performance.now() - start;
  report("complete", status, ticket);
  return { a3m, ticket, depth: alignment.depth, elapsedMilliseconds };
}

/**
 * Search each unique physical chain and assemble a complex A3M.
 * Identical chains share one network search and are PAIRED into single rows,
 * which is both cheaper in the cluster budget and the only form that carries
 * coevolution between the copies. Distinct chains stay block-diagonal, the
 * monomer-model heterooligomer construction ColabFold uses.
 *
 * @param {readonly string[]} sequenceValues one sequence per physical chain
 * @param {any} [options] the options accepted by generateMmseqs2Msa, plus
 *   `pairRepeatedChains` (default true) to pair copies of one protein
 * @returns {Promise<{a3m: string, tickets: string[], depth: number, elapsedMilliseconds: number}>}
 */
export async function generateMmseqs2ComplexMsa(sequenceValues, options = {}) {
  if (!Array.isArray(sequenceValues) || sequenceValues.length < 2) {
    throw new RangeError("a complex MSA requires at least two chain sequences");
  }
  const sequences = sequenceValues.map(normalizedSequence);
  const unique = [...new Set(sequences)];
  const started = performance.now();
  const entries = await Promise.all(unique.map(async(sequence, uniqueIndex) => {
    const searched = await generateMmseqs2Msa(sequence, {
      ...options,
      onProgress: options.onProgress === undefined ? undefined : (progress) => options.onProgress({
        ...progress, chain: sequences.indexOf(sequence), search: uniqueIndex, searches: unique.length,
      }),
    });
    return [sequence, searched];
  }));
  const bySequence = new Map();
  for (const [sequence, searched] of entries) bySequence.set(sequence, searched);
  // ...`pairRepeatedChains: false` restores the block-diagonal form, so the two
  // constructions can be folded against each other without a rebuild.
  const merge = options.pairRepeatedChains === false ? mergeUnpairedChainA3ms : mergeChainA3ms;
  const a3m = merge(sequences.map((sequence) => bySequence.get(sequence).a3m));
  const alignment = parseA3m(a3m);
  return {
    a3m,
    tickets: unique.map((sequence) => bySequence.get(sequence).ticket),
    depth: alignment.depth,
    elapsedMilliseconds: performance.now() - started,
  };
}

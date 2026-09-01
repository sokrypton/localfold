import { parseA3m } from "./a3m.js";
import { concatenateA3mBlocks, mergeChainA3ms, mergeRowAlignedChainA3ms, mergeUnpairedChainA3ms }
  from "./chains.js";

/** How each model wants a complex's per-chain alignments merged. */
const CHAIN_MERGES = {
  monomer: mergeUnpairedChainA3ms,
  multimer: mergeChainA3ms,
  af3: mergeRowAlignedChainA3ms,
};

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

/**
 * The per-query paired A3Ms from a `ticket/pair` archive.
 *
 * 🔴 ONE FILE, ONE BLOCK PER QUERY, AND THE ROW ORDER IS THE PAIRING. pair.a3m
 * holds every query's alignment concatenated with NUL separators, and row r of
 * one query is the SAME SPECIES as row r of every other. That correspondence is
 * the entire product of the pair endpoint; reordering, deduplicating or
 * independently cropping these destroys it while leaving valid A3Ms behind.
 *
 * @param {Uint8Array} tarBytes
 * @param {number} queryCount
 * @returns {string[]} one A3M per query, in query order, all the same depth
 */
export function extractMmseqs2PairedA3ms(tarBytes, queryCount) {
  const files = readTarFiles(tarBytes);
  const pair = [...files].find(([path]) => path === "pair.a3m" || path.endsWith("/pair.a3m"))?.[1];
  if (pair === undefined) throw new Error("MMseqs2 pairing result is missing pair.a3m");
  const contents = new TextDecoder().decode(pair);
  return Array.from({ length: queryCount }, (_, index) => `${queryBlock(contents, QUERY_ID + index)}\n`);
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

/**
 * Submit one job and return its archive: submit, poll, download.
 *
 * The two endpoints differ only in where they are posted and what mode they are
 * given, so they share this. `ticket/msa` takes one query and searches it;
 * `ticket/pair` takes ALL of a complex's distinct sequences in a single query
 * and returns rows already aligned across them, which is why pairing cannot be
 * assembled from separate per-chain searches however they are merged.
 *
 * @param {readonly string[]} sequences one or more, written as >101, >102, ...
 * @param {"msa"|"pair"} endpoint
 * @param {string} mode `env`/`all` for msa, `pairgreedy`/`paircomplete` for pair
 */
async function runMmseqs2Job(sequences, endpoint, mode, options = {}) {
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const wait = options.wait ?? waitWithAbort;
  const decompress = options.decompress ?? decompressGzip;
  const apiUrl = new URL(options.apiUrl ?? DEFAULT_API_URL);
  if (!apiUrl.pathname.endsWith("/")) apiUrl.pathname += "/";
  const start = performance.now();
  const report = (phase, status, ticket) => options.onProgress?.({
    phase, status, search: endpoint === "pair" ? "paired" : "unpaired",
    ...(ticket === undefined ? {} : { ticket }), elapsedMilliseconds: performance.now() - start,
  });
  const signal = options.signal;
  const body = new URLSearchParams({
    q: sequences.map((sequence, index) => `>${QUERY_ID + index}\n${sequence}\n`).join(""),
    mode,
  });
  let ticket;
  let status = "UNKNOWN";
  while (ticket === undefined) {
    report("submitting", "SUBMIT");
    const response = await request(fetchImplementation, new URL(`ticket/${endpoint}`, apiUrl), {
      method: "POST", body, ...(signal === undefined ? {} : { signal }),
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
    }, "MMseqs2 submission");
    const submitted = parseTicket(await response.json()); status = submitted.status; ticket = submitted.id;
    if (TRANSIENT_SUBMISSION.has(status)) {
      ticket = undefined; report("retrying", status);
      await wait(5_000 + Math.floor(Math.random() * 5_000), signal); continue;
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
  const archive = await decompress(await response.arrayBuffer());
  report("complete", status, ticket);
  return { archive, ticket, elapsedMilliseconds: performance.now() - start };
}

/**
 * The species-paired alignments for a complex's distinct sequences.
 *
 * @param {readonly string[]} uniqueSequences at least two
 * @param {any} [options] plus `pairingStrategy`: "greedy" (default) or "complete"
 * @returns {Promise<{a3ms: string[], ticket: string, depth: number}>}
 */
export async function generateMmseqs2PairedMsa(uniqueSequences, options = {}) {
  if (!Array.isArray(uniqueSequences) || uniqueSequences.length < 2) {
    throw new RangeError("pairing requires at least two distinct sequences");
  }
  const sequences = uniqueSequences.map(normalizedSequence);
  const strategy = options.pairingStrategy ?? "greedy";
  const job = await runMmseqs2Job(sequences, "pair", `pair${strategy}`, options);
  const a3ms = extractMmseqs2PairedA3ms(job.archive, sequences.length);
  const alignments = a3ms.map((text) => parseA3m(text));
  alignments.forEach((alignment, index) => {
    if (alignment.query !== sequences[index]) {
      throw new Error("MMseqs2 returned a paired A3M for a different query sequence");
    }
  });
  // 🔴 EQUAL DEPTH IS WHAT MAKES THEM PAIRED. Row r must exist in every entity
  // for "row r is one species" to mean anything; a ragged set would still merge
  // into a valid alignment and would silently pair different organisms.
  if (!alignments.every((alignment) => alignment.depth === alignments[0].depth)) {
    throw new Error("MMseqs2 paired A3Ms do not have aligned row counts");
  }
  return { a3ms, ticket: job.ticket, depth: alignments[0].depth };
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
 *   `model` (default "monomer") naming which model will read the alignment,
 *   which is what selects the chain merge - see CHAIN_MERGES
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
  // 🔴 THE MERGE IS CHOSEN BY THE MODEL THAT WILL READ IT, never by a flag
  // describing the shape. Three models, three constructions, and each is wrong
  // for the other two in a way nothing downstream can detect - so the caller
  // names the model and this table is the only place the mapping lives.
  //
  // monomer  block-diagonal for every chain. The AF2 monomer has no chain input
  //          at all; the +200 residue offset stands in for one, so a row
  //          carrying two chains would claim their residues coevolved.
  // multimer dense WITHIN an entity, block-diagonal BETWEEN entities. This is
  //          AlphaFold-Multimer's own construction: merge_chain_features runs
  //          _merge_homomers_dense_msa first, which groups by entity_id and
  //          concatenates each group along num_res, and only then block
  //          diagonalises what is left. Copies of one sequence are never block
  //          diagonalised; distinct entities always are.
  // af3      dense for every chain. AF3 has no block_diag anywhere in it -
  //          merge_msa_features pads to the deepest alignment and concatenates
  //          along the token axis, entity or not.
  //
  // multimer and af3 therefore agree exactly on a homomer, where there is one
  // entity, and differ on a heteromer. That is the models differing, not us.
  const model = options.model ?? "monomer";
  const merge = CHAIN_MERGES[model];
  if (merge === undefined) {
    throw new RangeError(`unknown model ${model}:`
      + ` expected ${Object.keys(CHAIN_MERGES).join(", ")}`);
  }
  const chainA3ms = sequences.map((sequence) => bySequence.get(sequence).a3m);
  const unpaired = merge(chainA3ms);

  // 🔴 PAIRING IS FOR DISTINCT SEQUENCES, AND ONLY THEM. AlphaFold decides this
  // the same way - feature_processing sets pair_msa_sequences from
  // `not _is_homomer_or_monomer` - and the reason is that copies of one protein
  // are already paired: one search speaks for every copy, so row r IS one
  // organism across all of them, which the unpaired merges above already
  // express. Asking the pair endpoint for a homomer costs a second search to
  // learn nothing.
  //
  // The AF2 MONOMER never gets paired rows whatever the input: it has no chain
  // input at all, so a row spanning two chains would claim their residues
  // coevolved. See CHAIN_MERGES.
  const wantsPairing = model !== "monomer" && unique.length > 1;
  let pairedResult;
  if (wantsPairing) {
    pairedResult = await generateMmseqs2PairedMsa(unique, {
      ...options,
      onProgress: options.onProgress === undefined ? undefined : (progress) =>
        options.onProgress({ ...progress, search: "paired" }),
    });
  }
  const pairedBySequence = new Map();
  if (pairedResult !== undefined) {
    unique.forEach((sequence, index) => pairedBySequence.set(sequence, pairedResult.a3ms[index]));
  }
  // 🔴 THE PAIRED BLOCK IS A ROW-ALIGNED MERGE FOR EVERY MODEL. Row r of each
  // entity's paired alignment is the same species, so placing them side by side
  // is exactly what "paired" means - there is no per-model choice here, unlike
  // the unpaired block. mergeRowAlignedChainA3ms already does precisely this.
  const paired = pairedResult === undefined ? undefined
    : mergeRowAlignedChainA3ms(sequences.map((sequence) => pairedBySequence.get(sequence)));

  // What a single-alignment consumer reads: paired rows above unpaired ones,
  // which is the order AlphaFold-Multimer's own merge produces and what
  // ColabFold writes out.
  const a3m = paired === undefined ? unpaired : concatenateA3mBlocks(paired, unpaired);
  const alignment = parseA3m(a3m);
  return {
    a3m,
    // ...and the two halves apart, for AF3, whose `msa` is the paired block
    // followed by the unpaired one with a profile computed over the second.
    blocks: { paired: paired ?? null, unpaired },
    pairedTicket: pairedResult?.ticket,
    pairedDepth: pairedResult?.depth ?? 0,
    // ...the merged text is what the viewer shows; the model takes these, so
    // clustering and masking can run separately for each copy.
    chainA3ms,
    tickets: unique.map((sequence) => bySequence.get(sequence).ticket),
    depth: alignment.depth,
    elapsedMilliseconds: performance.now() - started,
  };
}

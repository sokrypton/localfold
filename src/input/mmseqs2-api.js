import { parseA3m } from "./a3m.js";
import { concatenateA3mBlocks, mergeChainA3ms, deduplicateUnpairedAgainstPaired, mergeRowAlignedChainA3ms, mergeUnpairedChainA3ms }
  from "./chains.js";

/** How each model wants a complex's per-chain alignments merged. */
// 🔴 KEYED BY FAMILY, AND EVERY AF3-GRAPH FAMILY TAKES THE SAME MERGE. The
// alignment is an INPUT to the graph, not a property of whose weights are in
// it, so OpenBind reads its MSA exactly as AlphaFold 3 does. Absent here, a
// fold with it raises rather than quietly picking somebody's merge - which is
// why this throws on an unknown name instead of defaulting.
const CHAIN_MERGES = {
  monomer: mergeUnpairedChainA3ms,
  multimer: mergeChainA3ms,
  af3: mergeRowAlignedChainA3ms,
  openbind0: mergeRowAlignedChainA3ms,
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
/**
 * The template hits the search already returned.
 *
 * 🔴 THEY COME FREE WITH THE ALIGNMENT. `pdb70.m8` is in the MSA job's own tar
 * beside `uniref.a3m` - no second search, no extra request - and nothing here
 * had ever read it.
 *
 * 🔴 THE CIGAR IS KEPT AND DELIBERATELY NOT USED FOR THE MAPPING. The last
 * column is a real alignment - `34M1D57M` with the query and target starts -
 * but its target coordinates index pdb70's SEQUENCE, and what a template
 * actually offers is its RESOLVED residues. A structure missing a loop has
 * fewer of the second than the first, so reading the cigar against the parsed
 * chain would shift every residue after the first gap. Reconciling the two
 * needs the entry's SEQRES, which is a third parse of the mmCIF for a gain
 * that only shows up on remote homologs - and pdb70's top hits are not that.
 * web/template-source.js aligns the query to the resolved sequence instead,
 * and the coverage line says what it got.
 *
 * The fields are carried anyway, because choosing WHICH hits to use is what
 * the identity and the e-value are for.
 *
 * The row is BLAST tabular plus that column:
 * query, target, identity, alnlen, mismatch, gapopen, qstart, qend, tstart,
 * tend, evalue, bits, cigar.
 *
 * @param {Uint8Array} tarBytes the decompressed MSA result
 * @returns {Map<number, object[]>} query index -> hits, best first
 */
export function extractMmseqs2TemplateHits(tarBytes) {
  const files = readTarFiles(tarBytes);
  const entry = [...files].find(([path]) => path === "pdb70.m8"
    || path.endsWith("/pdb70.m8"));
  const hits = new Map();
  if (entry === undefined) return hits;
  for (const line of new TextDecoder().decode(entry[1]).split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 12) continue;
    const [query, target, identity, , , , qstart, , tstart, , evalue, bits] = parts;
    // 🔴 THE QUERY IS NUMBERED FROM 101, which is how the server labels the
    // chains of a job: the first is 101, the second 102. Subtracting gives the
    // chain index the rest of this page counts in.
    const chainIndex = Number(query) - 101;
    if (!Number.isInteger(chainIndex) || chainIndex < 0) continue;
    // `1qys_A` is a PDB entry and a chain; the structure arrives as `1qys.cif`.
    const [id, chain] = target.split("_");
    if (id === undefined) continue;
    if (!hits.has(chainIndex)) hits.set(chainIndex, []);
    hits.get(chainIndex).push({
      id: id.toLowerCase(),
      chain: chain ?? "A",
      target,
      identity: Number(identity),
      evalue: Number(evalue),
      bits: Number(bits),
      queryStart: Number(qstart),
      templateStart: Number(tstart),
      cigar: parts[12] ?? "",
    });
  }
  return hits;
}

/**
 * The structures for a set of hits, as mmCIF.
 *
 * 🔴 ONE HOST FOR EVERYTHING, AND NOT THE RCSB. ColabFold asks its own server
 * for these - `{api}/template/{comma-separated ids}` - and gets a gzipped tar
 * of `<id>.cif` back, so a page that already talks to the MSA API needs no
 * second origin and no second CORS story.
 *
 * 🔴 ASK FOR THE HIT NAMES, `1qys_A`, NOT THE BARE ENTRY. The endpoint takes
 * the pdb70 target names the m8 gave - chain suffix included - and returns
 * ONE mmCIF PER ENTRY, named `1qys.cif`. Asked for `1qys` alone it answers 200
 * with a tar holding only the hhsearch index and no structure at all, which is
 * a success with nothing in it rather than an error.
 *
 * @param {string[]} targets pdb70 target names, `1qys_A`
 * @returns {Promise<Map<string, string>>} entry id -> mmCIF text
 */
export async function fetchMmseqs2Templates(targets, options = {}) {
  const wanted = [...new Set(targets)].filter((id) => /^[A-Za-z0-9]{4}_[A-Za-z0-9]+$/.test(id));
  if (wanted.length === 0) return new Map();
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const decompress = options.decompress ?? decompressGzip;
  const apiUrl = new URL(options.apiUrl ?? DEFAULT_API_URL);
  if (!apiUrl.pathname.endsWith("/")) apiUrl.pathname += "/";
  const response = await request(
    fetchImplementation,
    new URL(`template/${wanted.join(",")}`, apiUrl),
    options.signal === undefined ? {} : { signal: options.signal },
    "MMseqs2 template download");
  const files = readTarFiles(new Uint8Array(await decompress(await response.arrayBuffer())));
  const structures = new Map();
  for (const [path, bytes] of files) {
    const name = path.split("/").pop() ?? path;
    if (!name.endsWith(".cif")) continue;
    structures.set(name.slice(0, -4).toLowerCase(), new TextDecoder().decode(bytes));
  }
  return structures;
}

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
  return { a3ms, ticket: job.ticket, depth: alignments[0].depth,
           templateHits: extractMmseqs2TemplateHits(job.archive) };
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
  // ...the archive is kept, not consumed inline: the template hits come out of
  // the same tar and a second decompression would be a second copy of it.
  const archive = await decompress(await response.arrayBuffer());
  const a3m = extractMmseqs2A3m(archive, useEnvironmental);
  const alignment = parseA3m(a3m);
  if (alignment.query !== sequence) throw new Error("MMseqs2 returned an A3M for a different query sequence");
  const elapsedMilliseconds = performance.now() - start;
  report("complete", status, ticket);
  // 🔴 THE TEMPLATE HITS COME OUT OF THE SAME TAR, so a page that wants them
  // pays for no second request. Returned rather than fetched later because the
  // archive is discarded here and nothing else ever sees it.
  return { a3m, ticket, depth: alignment.depth, elapsedMilliseconds,
           templateHits: extractMmseqs2TemplateHits(archive) };
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
  // 🔴 THE MERGE IS THE ONLY MODEL-SPECIFIC STEP, and it is separate for that
  // reason. The searches above are the expensive part and none of them depends
  // on the model; only this does. Keeping them apart lets a caller hold the raw
  // per-chain results and re-merge them when the model changes, instead of
  // asking the server the same question again.
  const { a3m, blocks } = mergeSearchedChains({
    sequences, chainA3ms, pairedA3ms: pairedBySequence, model,
  });
  const alignment = parseA3m(a3m);
  return {
    a3m,
    // ...and the two halves apart, for AF3, whose `msa` is the paired block
    // followed by the unpaired one with a profile computed over the second.
    // `unpairedProfile` is the unpaired block BEFORE deduplication, because
    // AF3 computes the profile first (features.py:543) and deduplicates after
    // (:559) - the removed rows still counted towards it.
    blocks,
    pairedTicket: pairedResult?.ticket,
    pairedDepth: pairedResult?.depth ?? 0,
    // ...the merged text is what the viewer shows; the model takes these, so
    // clustering and masking can run separately for each copy.
    chainA3ms,
    // ...and the paired ones beside them, keyed by the UNIQUE sequence they
    // were searched for, so a caller can re-merge for another model without a
    // second round trip. Absent when the search did not pair.
    pairedA3ms: pairedResult === undefined ? undefined : pairedBySequence,
    tickets: unique.map((sequence) => bySequence.get(sequence).ticket),
    depth: alignment.depth,
    elapsedMilliseconds: performance.now() - started,
  };
}

/**
 * Merge searched per-chain alignments the way one model wants them.
 *
 * Split out of generateMmseqs2ComplexMsa because it is the only part of that
 * function the model changes: the searches are the expensive half and they are
 * the same whatever will read them, so a caller holding these raw pieces can
 * switch models without asking the server again.
 *
 * @param {{sequences: string[], chainA3ms: string[],
 *          pairedA3ms?: Map<string, string>, model: string}} input
 *   `chainA3ms` is one unpaired A3M per chain, in chain order; `pairedA3ms`
 *   maps a UNIQUE sequence to its paired A3M, and is absent when the search
 *   did not pair.
 * @returns {{a3m: string, blocks: {paired: string|null, unpaired: string,
 *            unpairedProfile: string}}}
 */
/**
 * Whether a cached MSA search can answer this fold, and how.
 *
 * 🔴 THE SEARCH DOES NOT DEPEND ON THE MODEL AND THE MERGE DOES, so changing
 * the model re-merges what is already here rather than asking api.colabfold.com
 * the same question again. It is the one request this page makes off the
 * machine and the slow part of a fold; repeating it to answer a question
 * already answered is the worst thing the path can do.
 *
 * 🔴 EXCEPT WHEN THE NEW MODEL NEEDS PAIRING THE OLD SEARCH DID NOT ASK FOR.
 * Pairing is a second request that only multimer and AF3 make, so a monomer
 * search has no paired block to re-merge from. Reusing it anyway would silently
 * fold a complex with no paired rows - which looks like a worse prediction
 * rather than like a bug.
 *
 * 🔴 THIS LIVES HERE, NOT IN web/app.js, BECAUSE IT IS WHERE A CRASH CAME FROM.
 * The reuse branch calls mergeSearchedChains, app.js did not import it, and the
 * path needs a filled cache AND more than one chain - so no first fold reaches
 * it and nothing in a DOM-free test could. As a pure function of the cache, the
 * chains and the model, the decision is testable on its own.
 *
 * @param {{cache: {key: string, raw: object}|undefined, chains: string[],
 *          family: "monomer"|"multimer"|"af3"}} input
 * @returns {{reuse: "single"|"merge"|false, key: string, needsPairing: boolean}}
 */
export function planSearchReuse({ cache, chains, family }) {
  const key = JSON.stringify(chains);
  const needsPairing = family !== "monomer" && new Set(chains).size > 1;
  const usable = cache?.key === key
    && (!needsPairing || cache.raw?.pairedA3ms !== undefined);
  if (!usable) return { reuse: false, key, needsPairing };
  return { reuse: chains.length === 1 ? "single" : "merge", key, needsPairing };
}

/**
 * What to remember from a completed search, in the shape planSearchReuse reads.
 *
 * 🔴 A ONE-CHAIN SEARCH KEEPS ITS RESULT WHOLE and a complex keeps the PARTS,
 * because only a complex is ever re-merged - and the parts are what the merge
 * needs. Keeping the merged text for a complex would cache an answer that is
 * only right for the model that asked.
 */
export function searchCacheEntry({ chains, searched }) {
  // 🔴 THE TEMPLATE HITS ARE PART OF WHAT WAS SEARCHED, so they are cached with
  // it. A complex kept only its per-chain alignments here, so the SECOND fold -
  // the one that reuses this - had no hits, and a chain asking for a template
  // "from the MSA search" was told the MSA was not a search. It was; the hits
  // had simply been dropped on the way into the cache. They come out of the
  // same tar and cost nothing to keep.
  return chains.length === 1
    ? { single: searched, depth: searched.depth, templateHits: searched.templateHits }
    : { chainA3ms: searched.chainA3ms, pairedA3ms: searched.pairedA3ms,
      depth: searched.depth, templateHits: searched.templateHits };
}

export function mergeSearchedChains({ sequences, chainA3ms, pairedA3ms, model }) {
  const merge = CHAIN_MERGES[model];
  if (merge === undefined) {
    throw new RangeError(`unknown model ${model}:`
      + ` expected ${Object.keys(CHAIN_MERGES).join(", ")}`);
  }
  const hasPairing = pairedA3ms !== undefined && pairedA3ms.size > 0
    // The AF2 monomer never takes paired rows, whatever was searched: it has no
    // chain input, so a row spanning two chains would claim their residues
    // coevolved. See CHAIN_MERGES.
    && model !== "monomer";
  const paired = !hasPairing ? undefined
    : mergeRowAlignedChainA3ms(sequences.map((sequence) => pairedA3ms.get(sequence)));

  // 🔴 AND THE UNPAIRED BLOCK IS DEDUPLICATED AGAINST THE PAIRED ONE FIRST,
  // per chain, before the merge - which is why the merge happens here and not
  // above beside the search. AlphaFold's own pipeline does exactly this in
  // msa_pairing.deduplicate_unpaired_sequences, and it matters because the two
  // blocks are drawn from the same databases: on the 59-mer homodimer AF3
  // drops every one of the 32 unpaired rows as a duplicate. The blocks share a
  // fixed row budget, so a duplicate evicts a sequence that carried something.
  const unpairedProfile = merge(chainA3ms);
  const deduplicated = !hasPairing ? chainA3ms
    : sequences.map((sequence, chain) => deduplicateUnpairedAgainstPaired(
      chainA3ms[chain], pairedA3ms.get(sequence)));
  const unpaired = merge(deduplicated);

  // What a single-alignment consumer reads: paired rows above unpaired ones,
  // which is the order AlphaFold-Multimer's own merge produces and what
  // ColabFold writes out.
  return {
    a3m: paired === undefined ? unpaired : concatenateA3mBlocks(paired, unpaired),
    blocks: { paired: paired ?? null, unpaired, unpairedProfile },
  };
}


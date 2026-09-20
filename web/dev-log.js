import { memorySnapshot, memoryTotals } from "../src/runtime/device-memory.js";

/**
 * What a fold spent, phase by phase, behind a footer button.
 *
 * 🔴 IT LISTENS TO THE STATUS LINE RATHER THAN TO THE FOLD. Every path this
 * page can take - AF2, AF2-multimer, AF3, a cached trunk, a search or an
 * upload - already names what it is doing there, once, as it starts doing it.
 * Threading a callback through all of them instead would be five call sites to
 * keep in step, and would still miss whatever was added next. One hook in
 * `status()` gets them all and cannot go stale.
 *
 * 🔴 AND IT COSTS A STRING COMPARE PER STATUS WRITE, which is the reason it can
 * be on for everybody rather than behind a flag. A phase is recorded when the
 * status line's LEADING SEGMENT changes, so "Trunk · 41%" and "Trunk · 42%" are
 * one row rather than two hundred. A sampler step DOES get its own row, because
 * the page names it - "Folding 7/16" - and a row a step is worth having: it is
 * where a stall would show. MAX_ROWS bounds the rest. Nothing here runs per
 * dispatch and nothing awaits the GPU: `memorySnapshot` reads counters the
 * allocator already keeps.
 *
 * 🔴 THE MEMORY IS THE DEVICE'S OWN ACCOUNTING, NOT A GUESS. See
 * src/runtime/device-memory.js: `residentBytes` is what is on the device now,
 * pooled buffers included, and `peakBytes` is the high-water mark. Both are
 * exact - the allocator counts every buffer before it creates it - and the
 * per-label breakdown is what says which tensor to blame.
 */

const MAX_ROWS = 400;

let device;
let rows = [];
/**
 * 🔴 WHOSE MACHINE THIS IS ABOUT, WHICH IS NOT ALWAYS THIS ONE. On a page
 * folding through a Colab runtime the phases, the memory and the user agent
 * that matter are the RUNTIME's - this browser draws and nothing else - and a
 * report headed with the reader's user agent and "device memory: not
 * measured" describes a machine that did no work. `devOnEntry` is how the
 * runtime's own log leaves that page and `devAdopt` is how it arrives here;
 * the page in between is told to stop recording its own.
 */
let listener;
let adopting = false;
let source;
let runStartedAt = 0;
let currentPhase;
let currentStartedAt = 0;
let currentStartPeak = 0;

/** The part of a status message that names the phase. */
function phaseOf(text) {
  // "Trunk · 41%" and "Trunk · 42%" are one phase; "MSA search · queued
  // (PENDING) · 41s" is another. The separator is the page's own.
  const head = String(text ?? "").split("·")[0].trim();
  return head === "" ? "(idle)" : head;
}

function megabytes(bytes) {
  return Math.round((bytes / (1024 * 1024)) * 10) / 10;
}

function snapshot() {
  if (device === undefined) return undefined;
  try {
    // ...the totals only. The breakdowns are built when a reader opens the
    // panel, not on the fold's path; see memoryTotals.
    const gpu = memoryTotals(device);
    return { resident: megabytes(gpu.residentBytes), peak: megabytes(gpu.peakBytes),
             peakBytes: gpu.peakBytes };
  } catch {
    // A report must never be able to break a fold.
    return undefined;
  }
}

const emit = (entry) => {
  if (listener === undefined || adopting) return;
  try { listener(entry); } catch { /* a report must never break a fold */ }
};

/** Hear every entry as it is recorded - the door out of this page's log. */
export function devOnEntry(fn) {
  listener = fn;
}

/** ...and the door in. An entry recorded somewhere else, kept verbatim. */
export function devAdopt(entry) {
  if (entry === null || typeof entry !== "object") return;
  adopting = true;
  try {
    if (entry.reset !== undefined) {
      rows = [];
      runStartedAt = performance.now();
      currentPhase = undefined;
      currentStartedAt = runStartedAt;
      devNote(entry.reset);
      return;
    }
    rows.push(entry);
    if (rows.length > MAX_ROWS) rows.shift();
  } finally {
    adopting = false;
  }
}

/**
 * Name the machine these rows came from, or undefined for this one.
 *
 * What it changes is the HEADER: the timings and the memory below it are
 * whatever was recorded, and saying they are this browser's when they are not
 * is the whole complaint this answers.
 */
export function devSourceIs(label) {
  source = label;
}

/** Let the log read this device's memory counters. */
export function devUseDevice(value) {
  device = value;
}

/** Start a fresh timeline. Called when a fold begins. */
export function devBeginRun(label) {
  rows = [];
  runStartedAt = performance.now();
  currentPhase = undefined;
  currentStartedAt = runStartedAt;
  currentStartPeak = snapshot()?.peakBytes ?? 0;
  emit({ reset: label });
  devNote(label);
}

/** Close the phase that is running, if any. */
function closePhase(at) {
  if (currentPhase === undefined) return;
  const memory = snapshot();
  // 🔴 THE RISE, NOT ONLY THE LEVEL. `held` is read when the phase CLOSES, so
  // the last phase of a fold reads zero - everything has been released by then
  // - and that looks like the fold used no memory. How much this phase pushed
  // the high-water mark up is the number that says where the memory went, and
  // it survives the cleanup that follows.
  const row = {
    phase: currentPhase,
    ms: Math.round(at - currentStartedAt),
    atMs: Math.round(currentStartedAt - runStartedAt),
    ...(memory === undefined ? {} : {
      resident: memory.resident,
      peak: memory.peak,
      rise: megabytes(Math.max(0, memory.peakBytes - currentStartPeak)),
    }),
  };
  rows.push(row);
  if (rows.length > MAX_ROWS) rows.shift();
  emit(row);
}

/**
 * Record what the status line now says. Cheap enough to call on every write.
 */
export function devStatus(text) {
  const phase = phaseOf(text);
  if (phase === currentPhase) return;
  const at = performance.now();
  closePhase(at);
  currentPhase = phase;
  currentStartedAt = at;
  currentStartPeak = snapshot()?.peakBytes ?? 0;
}

/** A one-off line that is not a phase - a size, a score, a setting. */
export function devNote(text) {
  if (text === undefined) return;
  const row = { note: String(text), atMs: Math.round(performance.now() - runStartedAt) };
  rows.push(row);
  if (rows.length > MAX_ROWS) rows.shift();
  emit(row);
}

/** Close the last phase and note the total. Called when a fold ends. */
export function devEndRun(note) {
  closePhase(performance.now());
  currentPhase = undefined;
  devNote(note ?? "done");
}

/** The timeline as plain text, for the copy button. */
export function devReport() {
  const memory = snapshot();
  const agent = typeof navigator === "object" ? navigator.userAgent : "unknown";
  const lines = [
    `LocalFold timing · ${new Date().toISOString()}`,
    // 🔴 THE MACHINE THAT FOLDED FIRST, AND THIS ONE SECOND. With a Colab
    // runtime the rows below were recorded there, on its card, against its
    // clock; heading them with this browser's user agent is a report about a
    // machine that did nothing but draw.
    ...(source === undefined
      ? [`user agent: ${agent}`]
      : [`folded on: ${source}`, `shown in: ${agent}`]),
    source !== undefined
      ? "device memory: the runtime's, in the rows below"
      : (memory === undefined ? "device memory: not measured"
        : `device memory: ${memory.resident} MiB held, ${memory.peak} MiB peak`),
    "",
    "     at        ms   held  +peak    peak   phase",
  ];
  for (const row of rows) {
    if (row.note !== undefined) {
      lines.push(`${String(row.atMs).padStart(7)}                                  - ${row.note}`);
      continue;
    }
    lines.push(`${String(row.atMs).padStart(7)} ${String(row.ms).padStart(9)}`
      + `${(row.resident === undefined ? "" : `${row.resident}`).padStart(7)}`
      + `${(row.rise === undefined ? "" : `+${row.rise}`).padStart(7)}`
      + `${(row.peak === undefined ? "" : `${row.peak}`).padStart(8)}   ${row.phase}`);
  }
  const total = rows.reduce((sum, row) => sum + (row.ms ?? 0), 0);
  lines.push("", `total in phases: ${(total / 1000).toFixed(2)} s`);
  // ...and the breakdown below is read off THIS device, so it is left out
  // where the fold happened on another one rather than shown as a row of
  // zeros belonging to nothing.
  if (device !== undefined && source === undefined) {
    try {
      const gpu = memorySnapshot(device);
      lines.push("", "largest tensors on the device now:");
      if (gpu.currentByLabel.length === 0) lines.push("  (nothing - the fold released everything)");
      for (const entry of gpu.currentByLabel.slice(0, 12)) {
        lines.push(`  ${String(megabytes(entry.bytes)).padStart(8)} MiB  x${entry.count} ${entry.label}`);
      }
      if (gpu.peakByLabel.length > 0) {
        lines.push("", "what the peak was made of:");
        for (const entry of gpu.peakByLabel.slice(0, 12)) {
          lines.push(`  ${String(megabytes(entry.bytes)).padStart(8)} MiB  x${entry.count} ${entry.label}`);
        }
      }
    } catch { /* the report is best-effort */ }
  }
  return lines.join("\n");
}

/** Whether anything has been recorded yet. */
export function devHasRows() {
  return rows.length > 0;
}

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
  rows.push({
    phase: currentPhase,
    ms: Math.round(at - currentStartedAt),
    atMs: Math.round(currentStartedAt - runStartedAt),
    ...(memory === undefined ? {} : {
      resident: memory.resident,
      peak: memory.peak,
      rise: megabytes(Math.max(0, memory.peakBytes - currentStartPeak)),
    }),
  });
  if (rows.length > MAX_ROWS) rows.shift();
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
  rows.push({ note: String(text), atMs: Math.round(performance.now() - runStartedAt) });
  if (rows.length > MAX_ROWS) rows.shift();
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
  const lines = [
    `LocalFold timing · ${new Date().toISOString()}`,
    `user agent: ${typeof navigator === "object" ? navigator.userAgent : "unknown"}`,
    memory === undefined ? "device memory: not measured"
      : `device memory: ${memory.resident} MiB held, ${memory.peak} MiB peak`,
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
  if (device !== undefined) {
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

import { memorySnapshot, memoryTotals } from "../webgpu/runtime/device-memory.js";

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
 * webgpu/runtime/device-memory.js: `residentBytes` is what is on the device now,
 * pooled buffers included, and `peakBytes` is the high-water mark. Both are
 * exact - the allocator counts every buffer before it creates it - and the
 * per-label breakdown is what says which tensor to blame.
 */

const MAX_ROWS = 400;

let device;
let rows = [];
/**
 * 🔴 WHOSE MACHINE THIS IS ABOUT, WHICH IS NOT ALWAYS THIS ONE. On a page
 * folding through a Colab runtime this browser draws and nothing else, so a
 * report headed with the reader's user agent and "device memory: not measured"
 * describes a machine that did no work - `devSourceIs` names the runtime
 * instead, and the page stops recording its own phases (see status in app.js).
 */
let source;
let runStartedAt = 0;
let runEndedAt = 0;
let currentPhase;
let currentStartedAt = 0;
let currentStartPeak = 0;
let runLabel;
// the model's name, where its status lines lead with it ("EF2-fast · Trunk 1/4 · 41%"): stripped, or every stage of
// that fold is one phase called "EF2-fast"
let modelPrefix;

/** The part of a status message that names the phase. */
function phaseOf(text) {
  // "Trunk · 41%" and "Trunk · 42%" are one phase; "MSA search · queued
  // (PENDING) · 41s" is another. The separator is the page's own.
  let line = String(text ?? "");
  if (modelPrefix && line.startsWith(`${modelPrefix} ·`)) line = line.slice(modelPrefix.length + 2);
  const head = line.split("·")[0].trim();
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

/** Start a fresh timeline. Called when a fold begins; `model` is the name its status lines may lead with. */
export function devBeginRun(label, model = undefined) {
  rows = [];
  runLabel = label;
  modelPrefix = model;
  runStartedAt = performance.now();
  runEndedAt = 0;
  currentPhase = undefined;
  currentStartedAt = runStartedAt;
  currentStartPeak = snapshot()?.peakBytes ?? 0;
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
    local: true,
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

/**
 * A phase timed somewhere else - by the native worker, where the fold ran - with its own duration: `sub` breaks
 * the phase before it down (its stages, as the binary printed them) and is shown indented, outside the total.
 */
export function devPhase(phase, ms, sub = false, atMs = undefined) {
  if (phase === undefined || !Number.isFinite(ms)) return;
  // (`atMs`: when the step began, by the worker's clock from the job's start - not when its row arrived here)
  rows.push({ phase: String(phase), ms: Math.round(ms), sub,
              atMs: Number.isFinite(atMs) ? Math.round(atMs) : Math.round(performance.now() - runStartedAt) });
  if (rows.length > MAX_ROWS) rows.shift();
}

/** A one-off line that is not a phase - a size, a score, a setting. */
export function devNote(text) {
  if (text === undefined) return;
  const row = { note: String(text), atMs: Math.round(performance.now() - runStartedAt) };
  rows.push(row);
  if (rows.length > MAX_ROWS) rows.shift();
}

/** Close the last phase and note the total. Called when a fold ends. */
export function devEndRun(note) {
  runEndedAt = performance.now();
  closePhase(runEndedAt);
  currentPhase = undefined;
  if (note !== undefined) devNote(note);
}

// a step that is getting the model ready rather than folding: the weights, the kernels, the wait for them
const LOADING = /^(loading|starting webgpu|waiting for the model|the weights on disk)/i;

/** "Trunk" -> "trunk", but "MSA search" stays as it is. */
function sentence(text) {
  return /^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text;
}

/**
 * The rows as a reader wants them, the same shape for a fold here (WebGPU) and on the native backend: the time before
 * this page's first status line named as the model loading, a run of numbered steps one row ("Trunk 1/4" ...
 * "Trunk 4/4" -> "trunk (4 passes)", "Folding 1/20" ... -> "folding (20 steps)"), and the zero-length rows that only
 * mark the end gone.
 */
function steps() {
  const out = [];
  const timed = rows.filter((row) => row.note === undefined);
  const first = timed.find((row) => row.local);
  if (first !== undefined && first.atMs > 50 && !/^(loading|starting webgpu)$/i.test(first.phase)) {
    out.push({ phase: "loading the model (weights, kernels)", ms: first.atMs, atMs: 0 });
  }
  for (const row of rows) {
    if (row.note !== undefined) { out.push(row); continue; }
    if (row.local && row.ms === 0) continue;
    // (the model getting ready has one name, whichever words its status used: "loading", "Starting WebGPU")
    const phase = !row.local ? row.phase : /^(loading|starting webgpu)$/i.test(row.phase)
      ? "loading the model (weights, kernels)" : row.phase;
    const numbered = /^(.*?)\s+(\d+)\/(\d+)$/.exec(phase);
    // ...and one step said two ways is one row: "Folding 68 residues" then "Folding"
    const base = numbered ? numbered[1] : row.local ? phase.split(" ")[0] : phase;
    const last = out[out.length - 1];
    if (!row.sub && last !== undefined && last.base === base && (numbered || last.count === undefined)) {
      last.ms += row.ms;
      if (numbered) last.count = Number(numbered[3]);
      else last.phase = base;
      continue;
    }
    out.push({ ...row, phase: row.local && !numbered ? (last?.base === base ? base : phase) : phase, base,
               count: numbered ? Number(numbered[3]) : undefined });
  }
  for (const row of out) {
    if (row.phase === undefined) continue;
    const unit = /trunk|pass/i.test(row.base ?? "") ? "passes" : "steps";
    row.label = sentence(row.count !== undefined ? `${row.base} (${row.count} ${unit})` : row.phase);
  }
  return out;
}

const seconds = (ms) => `${(ms / 1000).toFixed(2)} s`;

/** The timeline as plain text, for the copy button. */
export function devReport() {
  const memory = snapshot();
  const agent = typeof navigator === "object" ? navigator.userAgent : "unknown";
  const list = steps();
  const top = list.filter((row) => row.phase !== undefined && !row.sub);
  const inSteps = top.reduce((sum, row) => sum + row.ms, 0);
  const total = runEndedAt > runStartedAt ? runEndedAt - runStartedAt : inSteps;
  const loading = top.filter((row) => LOADING.test(row.label)).reduce((sum, row) => sum + row.ms, 0);
  const lines = [
    `LocalFold timing · ${new Date().toISOString()}`,
    // 🔴 THE MACHINE THAT FOLDED FIRST, AND THIS ONE SECOND. With a native
    // backend the rows below were recorded there, on its GPU, against its
    // clock; heading them with this browser's user agent is a report about a
    // machine that did nothing but draw.
    ...(source === undefined
      ? [`folded in: this browser (WebGPU) · ${agent}`]
      : [`folded on: ${source}`, `shown in: ${agent}`]),
    ...(runLabel === undefined ? [] : [runLabel]),
    // ...the seconds a reader waited, and how much of it was the model getting ready rather than folding
    `total ${seconds(total)}: loading the model ${seconds(loading)}, folding ${seconds(Math.max(0, total - loading))}`,
    ...(source === undefined && memory !== undefined ? [`device memory: ${memory.peak} MiB at the peak`] : []),
    "",
    "     at        ms   step",
  ];
  for (const row of list) {
    if (row.note !== undefined) {
      lines.push(`${String(row.atMs).padStart(7)}             - ${row.note}`);
      continue;
    }
    lines.push(`${String(row.atMs).padStart(7)} ${String(row.ms).padStart(9)}   ${row.sub ? "    " : ""}${row.label}`);
  }
  // ...and what is not in a step: the page's own work and, for a native fold, the trip between it and the worker
  if (total - inSteps > 100) lines.push(`${" ".repeat(7)} ${String(Math.round(total - inSteps)).padStart(9)}   (between the steps above)`);
  // what the peak was made of, read off THIS device - so left out where the fold happened on another one
  if (device !== undefined && source === undefined) {
    try {
      const gpu = memorySnapshot(device);
      if (gpu.peakByLabel.length > 0) {
        lines.push("", "what the device memory's peak was made of:");
        for (const entry of gpu.peakByLabel.slice(0, 8)) {
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

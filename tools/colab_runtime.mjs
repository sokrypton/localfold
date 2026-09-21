#!/usr/bin/env node
/**
 * The Colab runtime, folding in THIS PROCESS over Dawn instead of in a browser.
 *
 *     node tools/colab_runtime.mjs --port 8710 --token XXX
 *
 * 🔴 WHAT THIS REPLACES IS A BROWSER THAT WAS NEVER WANTED FOR ITS BROWSING.
 * The runtime used to be a headless Chrome opening `index.html?role=runtime`
 * and DRIVING ITS OWN CONTROLS - `window.__entityList.set(...)`, four selects
 * written, `#predict` clicked - because the fold lived in a page. It does not:
 * `foldAf3` in web/af3-model.js takes a device and three callbacks and touches
 * no DOM, which is why this file is short. What the browser cost was 261 MB of
 * Chrome for Testing, four X11 libraries the image lacks, and 15.4-16.7 s of a
 * cold runtime's setup; Dawn is 6.7-6.9 s, measured counterbalanced on two
 * fresh T4s. See docs/WEB.md.
 *
 * 🔴 AND IT IS NOT A SECOND FOLD PATH. Every line of model work here is the
 * same module the page calls, down to the prediction object: `foldAf3` and
 * `predictionFromAf3`, both from web/af3-model.js. The one thing this file
 * owns is the TRANSPORT - the same protocol web/colab-bridge.js speaks, with
 * `fetch` where that has the page's own.
 */
import process from "node:process";
import { createNodeDevice } from "../src/node.js";
import { foldAf3, loadAf3Weights, af3SequenceProblem, samplerModeFor, AF3_COUNTS }
  from "../web/af3-model.js";
import { predictionFromAf3 } from "../web/af3-model.js";
// 🔴 WITHOUT THIS THE STREAMED TRAJECTORY TUMBLES. AF3's sampler
// re-augments the whole system every step, so consecutive frames differ
// by a rigid motion far larger than the denoiser's - `fittedPdb` fits
// them, and it asks a viewer library for the superposition. There is no
// viewer here; `superposeApi` is the same Kabsch this repository already
// folds AF2 with, in the shape that function expects.
import { superposeApi } from "./gpu/superpose.js";
import { MODEL_LABELS, AF3_FAMILIES, SINGLE_SEQUENCE_FAMILIES }
  from "../src/bundles/manifests/index.js";
import { foldEsmfold2Job, predictionFromEsmfold2 } from "../web/esmfold2-model.js";

const option = (name, fallback) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (hit !== undefined) return hit.slice(name.length + 3);
  const at = process.argv.indexOf(`--${name}`);
  return at === -1 || at + 1 >= process.argv.length ? fallback : process.argv[at + 1];
};

const PORT = Number(option("port", "8710"));
const TOKEN = option("token", "");
const BASE = `http://127.0.0.1:${PORT}`;
const door = (route, extra = "") =>
  `${BASE}${route}?t=${encodeURIComponent(TOKEN)}${extra}`;

// ── the transport ────────────────────────────────────────────────────────────

let seqOut = 0;
const pending = [];
let flushing = false;

/**
 * 🔴 PUSHED IN THE TASK THAT MADE IT, NOT POLLED FOR. The whole reason this
 * bridge exists is that the reader's feed was pulled over CDP and arrived in
 * batches: a fold looked frozen and then jumped. An event posted where it
 * happens is p50 1 ms. The flush is serialised so a burst does not open
 * twenty sockets, and never awaited by its caller - a status line must not be
 * able to stall a fold.
 */
function tapOut(kind, payload) {
  pending.push({ kind, payload, at: Date.now(), seq: seqOut });
  seqOut += 1;
  void flush();
}

async function flush() {
  if (flushing || pending.length === 0) return;
  flushing = true;
  try {
    while (pending.length > 0) {
      const events = pending.splice(0, pending.length);
      try {
        await fetch(door("/up"), {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ events }),
        });
      } catch (cause) {
        // A broker that is not answering is not a reason to lose the fold.
        // The events are gone; the fold continues and its `result` will say
        // what happened. Reported rather than thrown for the same reason the
        // worker fallback in py2Dmol's aligner does not reject.
        console.error("could not push events:", String(cause?.message ?? cause));
      }
    }
  } finally {
    flushing = false;
  }
}

const idle = (ms) => new Promise((done) => setTimeout(done, ms));

/** How long one fold may take before it is abandoned. */
const FOLD_TIMEOUT_MS = Number(option("fold-timeout-ms", "600000"));

// ── the fold ─────────────────────────────────────────────────────────────────

let device = null;
let adapterInfo = null;
let folding = false;
let abort = null;
let lastStatus = "Ready.";

const status = (text) => { lastStatus = text; tapOut("status", text); };

/**
 * 🔴 THE WEIGHTS ARE LOADED ONCE AND KEPT. `loadAf3Weights` fetches the
 * bundle - hundreds of megabytes - and a runtime that dropped them between
 * folds would re-download on every press. The page keeps them in `trunkCache`
 * for exactly this reason.
 */
// 🔴 AND ITS ARGUMENTS ARE POSITIONAL: `loadAf3Weights(onProgress, family)`,
// not an options object. Handed one, `onProgress` is a non-callable object and
// the store's `this.#onProgress?.(...)` THROWS - optional call only forgives
// null and undefined - so the fold died with "this[#onProgress] is not a
// function" three lines into loading, which reads like a broken store rather
// than a wrong call. It also takes no device: a tensor store is bytes.
const weightsByFamily = new Map();
async function weightsFor(family, onProgress) {
  if (!weightsByFamily.has(family)) {
    weightsByFamily.set(family, await loadAf3Weights(onProgress, family));
  }
  return weightsByFamily.get(family);
}

/**
 * What travels back to the reader, for any graph.
 *
 * 🔴 A TYPED ARRAY HAS TO ARRIVE AS ONE. JSON has none, so a plain stringify
 * flattens them - and they LOOK right everywhere until `matrixRows` slices a
 * PAE with `values.subarray(...)` and a download dies while the picture beside
 * it is perfect. The kind travels with the numbers; `revivePrediction` on the
 * reader's side puts it back.
 */
function wireResult(prediction, pdb, chains) {
  const predJson = JSON.stringify(prediction, (key, value) =>
    (ArrayBuffer.isView(value) && !(value instanceof DataView))
      ? { __typed: value.constructor.name, v: Array.from(value) } : value);
  return {
    predJson,
    a3m: prediction.a3m ?? null,
    confidence: prediction.confidence ?? null,
    scores: prediction.scores ?? null,
    chains: prediction.chains ?? null,
    length: chains.join("").length,
    status: lastStatus,
    pdb,
    atoms: (pdb.match(/^ATOM|^HETATM/gm) || []).length,
  };
}

async function runFold(request) {
  if (folding) return { error: "the runtime is already folding", status: lastStatus };

  // The entity list is the page's input; here a request carries either that
  // or a bare sequence, and copies are expanded the way expandEntities does.
  const entities = request.entities ?? [{
    type: "protein", value: request.sequence ?? "", copies: 1,
  }];
  const chains = [];
  for (const entity of entities) {
    const value = String(entity.value ?? "").trim().toUpperCase();
    if (value === "") continue;
    for (let copy = 0; copy < Number(entity.copies ?? 1); copy += 1) chains.push(value);
  }
  if (chains.length === 0) {
    return { error: "nothing to fold: the request carried no sequence",
             status: lastStatus };
  }
  const sequence = chains.join(":");
  const problem = af3SequenceProblem?.(sequence);
  if (problem) return { error: problem, status: lastStatus };

  const family = request.model ?? "af3";
  // 🔴 REFUSED BY NAME, NOT BY LEAKING AN EXCEPTION. This runtime folds the
  // AlphaFold 3 GRAPH - `foldAf3` and nothing else - and asked for AlphaFold 2
  // it used to fail deep inside `loadAf3Weights` with "monomer is not an
  // AlphaFold 3-graph family", which reads to a reader as a broken runtime
  // rather than as a runtime that does not do that yet. The other two drivers
  // exist but neither is reachable from here: ESMFold2's orchestration lives
  // in web/app.js as a page function, and tools/gpu/fold-af2.js returns a
  // BENCH REPORT - timings and checksums - rather than a prediction.
  // 🔴 EF2-fast IS WIRED AND DOES NOT WORK HERE, SO IT IS OPT-IN. Measured on
  // a Colab T4, twice: through the runtime the shard fetch dies with
  // "weights-00.int5.bin ... 0 bytes arrived ... after three attempts", and
  // the same job with a 300 ms poll beside it aborts natively -
  // `std::system_error: Invalid argument`, a core dump, no JavaScript error.
  // What is NOT the cause, each eliminated by measurement: the loader (it
  // succeeds standalone, with and without the 224 MiB tower), the
  // `timestamp_quantization` flag (both arms load), `navigator.gpu` being
  // unset by createNodeDevice (both arms load), and createNodeDevice itself
  // (one core dump that did not reproduce). The job alone did not finish in
  // 500 s either, on a two-CPU box with 11 GiB free.
  //
  // A family the runtime ACCEPTS and then fails on is worse than one it
  // refuses with a reason - the rule this project applies to controls. So the
  // default is the graph that is proven on a T4, and EF2-fast needs
  // `--allow-esmfold2` from somebody who is debugging it.
  const esmfold2Allowed = process.argv.includes("--allow-esmfold2");
  const known = [...AF3_FAMILIES, ...(esmfold2Allowed ? SINGLE_SEQUENCE_FAMILIES : [])];
  if (!known.includes(family)) {
    return {
      error: `this runtime folds ${known.join(", ")};`
        + ` ${family} needs a browser - start the service with --runtime chrome`
        + (SINGLE_SEQUENCE_FAMILIES.includes(family)
          ? " (EF2-fast is wired here but aborts on a Colab T4;"
            + " --allow-esmfold2 to work on it)" : ""),
      status: lastStatus,
    };
  }
  const modelName = MODEL_LABELS?.[family] ?? family;
  const mode = samplerModeFor?.(family) ?? "diffusion";
  folding = true;
  abort = new AbortController();
  // 🔴 A FOLD THAT RUNS AWAY MUST END BY ITSELF. The reader's only sign of
  // trouble is silence, and the broker cannot tell a long fold from a wedged
  // one - so the bound lives here, where the AbortController is. A 58-residue
  // AF3 fold is seconds on a T4; the ceiling is generous enough that a real
  // complex is never cut short and small enough that nobody watches a dead
  // page for an hour. `stop` cannot do this job: the command loop does not
  // poll while the fold holds the thread, which is measured - `runtimeSeen`
  // reached 431 s during one.
  const ceiling = Number(request.timeoutMs ?? FOLD_TIMEOUT_MS);
  const alarm = setTimeout(() => {
    status(`Fold gave up after ${Math.round(ceiling / 1000)}s`);
    abort?.abort();
  }, ceiling);
  tapOut("fold-begin", { at: Date.now(), family, residues: chains.join("").length });
  try {
    // 🔴 THE SECOND GRAPH, THROUGH THE SAME DOOR IT USES IN THE PAGE. EF2-fast
    // folds from the sequence alone - no alignment, no templates - and has no
    // confidence head, so its prediction is a different SHAPE and is built by
    // its own assembly. What is the same is the arrangement: a job that takes
    // callbacks, and an assembly that takes `(result, about)`.
    if (SINGLE_SEQUENCE_FAMILIES.includes(family)) {
      const out = await foldEsmfold2Job({
        chains, chainKinds: chains.map(() => "protein"),
        signal: abort.signal, device, family, modelName,
        // Every one of these reads a control in the page. Here they are the
        // request's, with the checkpoint's own preferences as defaults.
        languageModel: request.languageModel !== false,
        plm: request.plm ?? "esmc-600m",
        loops: Number(request.recycles ?? 0) + 1,
        sampler: request.sampler ?? "balanced",
        seed: Number(request.seed ?? 0),
        onStatus: status,
        onProgress: (fraction) => tapOut("progress", fraction),
        onFrame: (pdb) => tapOut("frame", pdb),
      });
      const prediction = predictionFromEsmfold2(out.result, {
        chains, stem: request.stem ?? family, pdb: out.pdb, modelName,
        certainty: out.certainty,
        context: { seed: Number(request.seed ?? 0) },
        settings: { "trunk passes": Number(request.recycles ?? 0) + 1,
                    "language model": request.plm ?? "esmc-600m",
                    sampler: request.sampler ?? "balanced" },
      });
      status(`${modelName} · ${chains.join("").length} residues`);
      return wireResult(prediction, out.pdb, chains);
    }

    status(`Loading ${modelName}`);
    // The download is most of a cold fold, so its progress is the reader's
    // only sign of life until the trunk starts.
    const weights = await weightsFor(family, (fraction) => {
      tapOut("progress", typeof fraction === "number" ? fraction
        : (fraction?.fraction ?? fraction?.loaded ?? 0));
    });
    const out = await foldAf3({
      sequence,
      device,
      weights,
      family,
      mode,
      calls: Number(request.steps ?? AF3_COUNTS?.[mode]?.default ?? 25),
      recycles: Number(request.recycles ?? 3),
      seed: Number(request.seed ?? 0),
      signal: abort.signal,
      alignment: request.alignment ?? null,
      onStatus: status,
      onProgress: (fraction) => tapOut("progress", fraction),
      // The live feed. `foldAf3` hands a PDB per sampler call, which is what
      // the reader's play bar fills from.
      onFrame: (pdb) => tapOut("frame", pdb),
      superpose: superposeApi,
    });
    const scored = out.confidence !== undefined;
    const prediction = predictionFromAf3(out, {
      chains, alignment: request.alignment ?? null, modelName,
      stem: request.stem ?? family, scored,
      context: { seed: Number(request.seed ?? 0), recycles: Number(request.recycles ?? 3) },
    });
    status(`${modelName} · ${chains.join("").length} residues`
      + (out.meanPlddt === undefined ? "" : ` · pLDDT ${out.meanPlddt.toFixed(1)}`));
    // 🔴 THE SAME WIRE SHAPE THE PAGE SENDS, TYPED ARRAYS AND ALL. JSON has no
    // typed arrays, so a plain stringify flattens them - and `matrixRows`
    // slices the PAE with `values.subarray(...)`, which a reader then dies on
    // while the picture beside it is perfect. `revivePrediction` on the other
    // side puts the kind back; this is the encoder that matches it.
    return wireResult(prediction, out.pdb, chains);
  } catch (cause) {
    const message = String(cause?.message ?? cause);
    status(`Fold failed: ${message}`);
    return { error: message, status: lastStatus };
  } finally {
    clearTimeout(alarm);
    folding = false;
    abort = null;
  }
}

// ── the command loop ─────────────────────────────────────────────────────────

async function obey(command) {
  const op = command?.op;
  const payload = command?.payload ?? command;
  if (op === "ping") {
    tapOut("pong", { at: Date.now(), folding, status: lastStatus });
    return;
  }
  if (op === "stop") {
    abort?.abort();
    tapOut("stopped", { at: Date.now() });
    return;
  }
  if (op === "fold") {
    // 🔴 NOT AWAITED, OR `stop` CANNOT BE HEARD. The loop below is the only
    // thing reading commands, so awaiting a fold here means the abort arrives
    // after the thing it would have aborted.
    void (async () => { tapOut("result", await runFold(payload ?? {})); })();
    return;
  }
  tapOut("status", `the runtime does not know the command "${op}"`);
}

async function serveCommands() {
  // FROM WHERE THE QUEUE STANDS NOW, NOT FROM ZERO - the same rule the page
  // has, and for the same reason: a restarted runtime that replays the
  // session re-runs a fold from ten minutes ago, weights and all.
  let since = 0;
  try {
    const answer = await fetch(door("/out", "&head=1"));
    if (answer.ok) since = (await answer.json()).n ?? 0;
  } catch { /* the first poll starts at zero, which is the old behaviour */ }
  for (;;) {
    try {
      const answer = await fetch(door("/out", `&since=${since}`));
      if (answer.ok) {
        const said = await answer.json();
        since = said.n ?? since;
        for (const command of said.commands ?? []) await obey(command);
      }
    } catch { /* the broker will be back, or the process will be stopped */ }
    await idle(300);
  }
}

// ── start ────────────────────────────────────────────────────────────────────

const made = await createNodeDevice();
device = made.device;
const info = made.adapter.info ?? {};
adapterInfo = {
  webgpu: true,
  vendor: info.vendor ?? null,
  architecture: info.architecture ?? null,
  device: info.device ?? null,
  shaderF16: made.device.features.has("shader-f16"),
  subgroupMatrix: made.device.features.has("chromium-experimental-subgroup-matrix"),
  maxBufferSize: made.device.limits.maxBufferSize,
  maxStorageBufferBindingSize: made.device.limits.maxStorageBufferBindingSize,
};
// One line the broker reads, the way the page's `runtime-ready` is read: it is
// how `/health` learns what the card is without a browser to ask.
console.log("ADAPTER " + JSON.stringify(adapterInfo));
tapOut("runtime-ready", { at: Date.now(), runtime: "node", gpu: adapterInfo });
await serveCommands();

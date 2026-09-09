/**
 * What AF2's FIRST fold is made of, beyond the fold.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af2-warmup.js --length=59
 *
 * 🔴 A FIRST FOLD IS 1.13 s AND A REPEAT IS 0.21, AND NOTHING SAID WHERE THE
 * OTHER 0.92 GOES. The weights stopped being decoded on the host two commits
 * ago, so what is left is pipeline compilation, the shader modules those are
 * built from, and the traffic that fills the resident buffers - three
 * different fixes, and the block profiler can see none of them because they
 * happen once and outside a compute pass.
 *
 * This wraps the device: every `createShaderModule`,
 * `createComputePipeline{,Async}` and `writeBuffer` is counted and timed
 * around, then one fold runs and the totals are reported against its wall.
 * `createComputePipelineAsync` returns a promise, so the number that matters
 * is not the sum of its awaits - several can be in flight - but the span from
 * the first request to the last settle, which is what `pipelineSpanMs` is.
 */
import { blockUploadStats } from "../../src/runtime/quantised-upload.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { AlphaFoldMonomerGpu } from "../../src/model/monomer.js";
import { AlphaFoldUnifiedGpu } from "../../src/multimer/model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

function instrument(device) {
  const totals = {
    shaderModules: 0, shaderModuleMs: 0,
    pipelinesAsync: 0, pipelinesSync: 0, pipelineSyncMs: 0,
    pipelineFirstRequest: undefined, pipelineLastSettle: undefined,
    pipelineIntervals: [],
    writeBuffers: 0, writeBufferBytes: 0, writeBufferMs: 0,
    buffers: 0, bufferBytes: 0, bufferMs: 0,
  };
  const makeModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor) => {
    const at = performance.now();
    const built = makeModule(descriptor);
    totals.shaderModules += 1;
    totals.shaderModuleMs += performance.now() - at;
    return built;
  };
  const makeSync = device.createComputePipeline.bind(device);
  device.createComputePipeline = (descriptor) => {
    const at = performance.now();
    const built = makeSync(descriptor);
    totals.pipelinesSync += 1;
    totals.pipelineSyncMs += performance.now() - at;
    return built;
  };
  const makeAsync = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = (descriptor) => {
    const at = performance.now();
    if (totals.pipelineFirstRequest === undefined) totals.pipelineFirstRequest = at;
    totals.pipelinesAsync += 1;
    return makeAsync(descriptor).then((pipeline) => {
      const settled = performance.now();
      totals.pipelineLastSettle = settled;
      totals.pipelineIntervals.push([at, settled, descriptor.label ?? "?"]);
      return pipeline;
    });
  };
  const makeBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const at = performance.now();
    const built = makeBuffer(descriptor);
    totals.buffers += 1;
    totals.bufferBytes += descriptor.size ?? 0;
    totals.bufferMs += performance.now() - at;
    return built;
  };
  const write = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (...rest) => {
    const at = performance.now();
    const done = write(...rest);
    totals.writeBuffers += 1;
    const source = rest[2];
    totals.writeBufferBytes += rest[4] ?? source?.byteLength ?? 0;
    totals.writeBufferMs += performance.now() - at;
    return done;
  };
  return totals;
}

export async function main(device, args) {
  const length = Number(option(args, "length", "59"));
  const rows = Number(option(args, "rows", "128"));
  const extraRows = Number(option(args, "extra", "128"));
  const family = option(args, "family", "monomer");
  const multimer = family === "multimer";
  const sequence = Array.from({ length },
    (_, i) => ALPHABET[i % ALPHABET.length]).join("");

  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const store = await HttpTensorStore.fromManifest(
    MODEL_BUNDLES[family].directory, await loadManifest(family));
  const fixture = AlphaFoldFixture.fromStore(store);
  const loadStart = performance.now();
  const [embedding, template, templateEmbedding, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks] = await Promise.all([
    fixture.embeddingWeights(),
    multimer ? Promise.resolve(undefined) : fixture.templateWeights(),
    multimer ? fixture.templateEmbeddingWeights() : Promise.resolve(undefined),
    fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ]);
  const weights = {
    embedding, template, templateEmbedding, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  };
  const loadMs = Math.round(performance.now() - loadStart);

  const lines = [">query", sequence];
  for (let row = 1; row < rows + extraRows; row += 1) {
    lines.push(`>synthetic${row}`);
    lines.push([...sequence].map((code, column) =>
      (column % (row % 11 + 3) === 0 ? "-" : code)).join(""));
  }
  const a3m = `${lines.join("\n")}\n`;
  const regime = multimer
    ? { outerProductMeanFirst: true, positionScale: 20, chainAware: true,
        chainSequences: [length] }
    : {};
  const options = { recycles: 0, randomSeed: 0, maxMsaSequences: rows,
                    maxExtraSequences: extraRows, chainLengths: [length], ...regime };

  const totals = instrument(device);
  const uploadsBefore = { ...blockUploadStats };
  const Model = multimer ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu;
  const first = performance.now();
  await new Model(device).predictA3m(a3m, weights, featureTables, options, paeBreaks);
  const firstMs = performance.now() - first;
  const cold = { ...totals };

  const again = performance.now();
  await new Model(device).predictA3m(a3m, weights, featureTables, options, paeBreaks);
  const repeatMs = performance.now() - again;

  const round = (value) => Math.round(value * 10) / 10;
  return {
    family, length, rows, extraRows, weightLoadMs: loadMs,
    firstFoldMs: Math.round(firstMs), repeatMs: Math.round(repeatMs),
    warmupMs: Math.round(firstMs - repeatMs),
    shaderModules: cold.shaderModules, shaderModuleMs: round(cold.shaderModuleMs),
    pipelinesAsync: cold.pipelinesAsync, pipelinesSync: cold.pipelinesSync,
    pipelineSyncMs: round(cold.pipelineSyncMs),
    // 🔴 A SPAN, NOT A SUM. Several async compiles can be in flight, so adding
    // their awaits double-counts; this is when the last one settled minus when
    // the first was asked for, which is what a warm could overlap.
    pipelineSpanMs: cold.pipelineLastSettle === undefined ? 0
      : round(cold.pipelineLastSettle - cold.pipelineFirstRequest),
    buffers: cold.buffers, bufferMiB: round(cold.bufferBytes / 1048576),
    bufferMs: round(cold.bufferMs),
    writeBuffers: cold.writeBuffers, writeBufferMiB: round(cold.writeBufferBytes / 1048576),
    writeBufferMs: round(cold.writeBufferMs),
    // 🔴 THE SUM AGAINST THE SPAN IS THE CONCURRENCY. If the two are equal the
    // compiles ran one at a time, which is what a fold that asks for a pipeline
    // at the moment it needs it gets; a warm that asks for all of them at once
    // would show a sum much larger than the span.
    pipelineAwaitSumMs: round(cold.pipelineIntervals
      .reduce((sum, [a, b]) => sum + (b - a), 0)),
    pipelineMaxInFlight: (() => {
      const marks = [];
      for (const [a, b] of cold.pipelineIntervals) { marks.push([a, 1]); marks.push([b, -1]); }
      marks.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      let live = 0; let peak = 0;
      for (const [, delta] of marks) { live += delta; peak = Math.max(peak, live); }
      return peak;
    })(),
    slowestPipelines: cold.pipelineIntervals
      .map(([a, b, label]) => [label, round(b - a)])
      .sort((x, y) => y[1] - x[1]).slice(0, 12),
    weightDecode: Object.fromEntries(Object.entries(blockUploadStats)
      .map(([key, value]) => [key, round(value - uploadsBefore[key])])),
    repeatShaderModules: totals.shaderModules - cold.shaderModules,
    repeatPipelines: (totals.pipelinesAsync + totals.pipelinesSync)
      - (cold.pipelinesAsync + cold.pipelinesSync),
  };
}

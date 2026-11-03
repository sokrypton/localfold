/**
 * Does the AF3 lineage's sequential track compute the whole-tensor track's answer, to the bit?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-sequential.js --family=protenix2 --lengths=130,257 --blocks=2
 *
 * The same bar as check-esmfold2-sequential.js, over the part that one cannot reach: the two grid attentions, whose
 * normalised input is computed a chunk at a time from a pair that an earlier chunk of the same pass has already written
 * into, and whose column direction reads that pair across its rows. It adds no arithmetic, so any differing element at
 * all is a mapping: a row taken after it was written, a transposed row, a tail tile, a bias swapped twice.
 *
 * Each stack runs over its own bundle's weights and one random input in one process: the pairformer with a mask with
 * holes and with none, because the attention's mask-free form is a different kernel and a fold's mask has no holes; the
 * MSA stack, whose outer product and row update live in the buffers the sequential track gives up; and the template stack.
 */
import { Af3MsaStackGpu } from "../../webgpu/af3/trunk/msa-stack.js";
import { Af3PairformerStackGpu } from "../../webgpu/af3/trunk/pairformer-block.js";
import { Af3TemplateEmbedderGpu } from "../../webgpu/af3/trunk/template.js";
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { bundleBaseUrl, loadManifest } from "../../shared/bundles/manifests/index.js";
import { af3Dialect, trunkWeights } from "../../shared/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function random(length, seed) {
  const output = new Float32Array(length);
  let state = seed;
  for (let i = 0; i < length; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    output[i] = (state / 0x7fffffff) * 2 - 1;
  }
  return output;
}

const ARMS = [
  ["compact", { forceCompact: true, sequentialTriangle: false }],
  ["sequential", { forceCompact: true, sequentialTriangle: true, sequentialChunkBytes: 1048576 }],
  ["sequential g16", { forceCompact: true, sequentialTriangle: true, sequentialGroups: 16, sequentialChunkBytes: 262144 }],
];

function assertSameBits(where, arm, name, expected, actual) {
  if (expected.length !== actual.length) throw new Error(`${where}, ${arm}: ${name} has ${actual.length} elements`);
  if (!expected.every(Number.isFinite)) throw new Error(`${where}: ${name} has non-finite elements`);
  let differing = 0;
  for (let i = 0; i < expected.length; i += 1) if (!Object.is(expected[i], actual[i])) differing += 1;
  if (differing !== 0) throw new Error(`${where}, ${arm}: ${name} differs from the whole-tensor track in ${differing} elements`);
}

export async function main(device, args) {
  const lengths = option(args, "lengths", "130,257").split(",").map(Number);
  const blockCount = Number(option(args, "blocks", "2"));
  const report = [];
  for (const family of option(args, "family", "protenix2").split(",")) {
    report.push(...await checkFamily(device, family, lengths, blockCount));
  }
  return report;
}

async function checkFamily(device, family, lengths, blockCount) {
  const store = await HttpTensorStore.fromManifest(bundleBaseUrl(family), await loadManifest(family));
  const dialect = af3Dialect(store);
  const report = [];
  const { pairformerBlocks: pairformer, msaBlocks, template } = await trunkWeights(store, blockCount, blockCount, { allowPrefix: true });

  for (const n of lengths) {
    const pairs = n * n;
    const holes = new Float32Array(pairs).fill(1);
    for (let i = 0; i < pairs; i += 97) holes[i] = 0;
    for (const [maskName, pairMask] of [["holes", holes], ["no holes", new Float32Array(pairs).fill(1)]]) {
      const channels = pairformer[0].pairChannels;
      const pair = random(pairs * channels, 12345);
      const single = random(n * pairformer[0].singleChannels, 54321);
      const seqMask = new Float32Array(n).fill(1);
      const run = async (options) => {
        const started = performance.now();
        const out = await new Af3PairformerStackGpu(device, { residentWeights: false })
          .run({ pair: pair.slice(), single: single.slice(), pairMask, seqMask, tokens: n }, pairformer, dialect, options);
        return { ...out, ms: performance.now() - started };
      };
      const whole = await run({ forceCompact: false, sequentialTriangle: false });
      for (const [arm, options] of ARMS) {
        const out = await run(options);
        const where = `the pairformer at ${n} tokens, mask with ${maskName}`;
        assertSameBits(where, arm, "the pair", whole.pair, out.pair);
        assertSameBits(where, arm, "the single", whole.single, out.single);
        report.push({ family, stack: "pairformer", n, mask: maskName, arm, differing: 0, ms: Math.round(out.ms), wholeMs: Math.round(whole.ms),
                      peakMiB: Math.round((out.memory?.peakBytes ?? 0) / 1048576), wholePeakMiB: Math.round((whole.memory?.peakBytes ?? 0) / 1048576) });
      }
    }
    const channels = msaBlocks[0].pairChannels;
    const sequences = 8;
    const msaChannels = msaBlocks[0].msaChannels;
    const pair = random(pairs * channels, 777);
    const msa = random(sequences * n * msaChannels, 888);
    const msaMask = new Float32Array(sequences * n).fill(1);
    const pairMask = new Float32Array(pairs).fill(1);
    const run = async (options) => {
      const started = performance.now();
      const out = await new Af3MsaStackGpu(device, { residentWeights: false })
        .run({ pair: pair.slice(), msa: msa.slice(), pairMask, msaMask, sequences, tokens: n }, msaBlocks, dialect, options);
      return { ...out, ms: performance.now() - started };
    };
    const whole = await run({ forceCompact: false, sequentialTriangle: false });
    for (const [arm, options] of ARMS) {
      const out = await run(options);
      const where = `the MSA stack at ${n} tokens`;
      assertSameBits(where, arm, "the pair", whole.pair, out.pair);
      assertSameBits(where, arm, "the MSA", whole.msa, out.msa);
      report.push({ family, stack: "msa", n, arm, differing: 0, ms: Math.round(out.ms), wholeMs: Math.round(whole.ms) });
    }
    // The template stack: a different width from the track's in IntelliFold-2, attention wider than its channels in
    // Boltz-2, and two heads in OpenDDE, which is the bias kernel's other arm.
    const asymId = new Int32Array(n);
    for (let token = 0; token < n; token += 1) asymId[token] = token < n / 2 ? 1 : 2;
    const input = { pair: random(pairs * template.queryChannels, 999), pairMask, tokens: n, templates: 2, asymId };
    const templated = (options) => new Af3TemplateEmbedderGpu(device, { residentWeights: false }).run(input, template, dialect, options);
    const wholeTemplate = await templated({ forceCompact: false, sequentialTriangle: false });
    for (const [arm, options] of ARMS) {
      const out = await templated(options);
      assertSameBits(`the template stack at ${n} tokens`, arm, "the output", wholeTemplate.output, out.output);
      report.push({ family, stack: "template", n, arm, differing: 0 });
    }
  }
  return report;
}

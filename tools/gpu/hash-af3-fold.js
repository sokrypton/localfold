/**
 * A whole fold of the AlphaFold 3 lineage, arm by arm in one browser: does it come out the same, and what did it take?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/hash-af3-fold.js --family=openbind0 --tokens=600 --arms=auto,compact,g8,g4,g2 --rounds=3
 *
 * `auto` is what the page runs, `compact` the channel-chunked track and `gN` the sequential one with N channel groups.
 * `pieces` is `auto` with the conditioning's pair kernels and the distogram cut into many dispatches.
 * A row is one fold: the PDB's hash, the contact map's hash at each pass, the mean pLDDT printed whole because a mean over
 * every atom moves when any of them does, the median pass after the first, the seconds of the whole fold, and what it left
 * on the device, which must not grow from fold to fold.
 *
 * 🔴 THIS IS THE COMPARISON TO BELIEVE, NOT A STACK'S TIME ALONE. Here the queue is paced as a page paces it and the arms
 * alternate in one browser, because two containers differ by more than the track does. A stack's time with each block
 * drained counts the device waiting on the host, and read OpenBind-0's channel groups at 600 tokens as 154 against 157 ms
 * a block where a fold's are 107 against 124.
 *
 * 🔴 AND THE FIRST FOLD IN A BROWSER IS NOT A LATER ONE, SO COMPARE FROM THE SECOND ROUND. `runtimeLoopBounds: "tiered"` runs
 * each kernel's opaque-bound form first and swaps in the unrolled one when it has compiled behind the fold, and the two
 * round differently, so a first fold's digits depend on when each swap landed: at 600 tokens on an A10 it read two
 * different PDBs on two runs of `main`, and every later fold read a third, on `main` and here in every arm.
 */
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";
import { memoryTotals } from "../../webgpu/runtime/device-memory.js";
import { setDeviceTuning } from "../../webgpu/runtime/device-profile.js";

const UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function fnv1a(bytes) {
  let hash = 2166136261;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}
const hashText = (text) => fnv1a(new TextEncoder().encode(text));
const hashFloats = (floats) => fnv1a(new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength));

function armTuning(arm) {
  if (arm === "auto") return { sequentialTriangle: null, sequentialGroups: null, pairHeadStepWork: null };
  if (arm === "pieces") return { sequentialTriangle: null, sequentialGroups: null, pairHeadStepWork: 1e9 };
  if (arm === "compact") return { sequentialTriangle: false, sequentialGroups: null, pairHeadStepWork: null };
  return { sequentialTriangle: null, sequentialGroups: Number(/^g(\d+)$/.exec(arm)[1]), pairHeadStepWork: null };
}

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export async function main(device, args) {
  const family = option(args, "family", "protenix2");
  const tokens = Number(option(args, "tokens", "400"));
  const recycles = Number(option(args, "recycles", "3"));
  const seed = Number(option(args, "seed", "0"));
  const arms = option(args, "arms", "auto").split(",");
  const rounds = Number(option(args, "rounds", "1"));
  const sequence = UBIQUITIN.repeat(Math.ceil(tokens / UBIQUITIN.length)).slice(0, tokens);
  const weights = await loadAf3Weights(() => {}, family);
  const report = [];
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) {
      setDeviceTuning(device, armTuning(arm));
      const stamps = [];
      const contacts = [];
      const started = performance.now();
      // eslint-disable-next-line no-await-in-loop
      const out = await foldAf3({
        sequence, calls: 25, recycles, seed, device, weights, alignment: null, maxMsaSequences: 1,
        onStatus: () => {}, onProgress: () => {},
        onContacts: (map, pass) => {
          if (pass === undefined) return;
          stamps.push(performance.now());
          contacts.push(hashFloats(map));
        },
      });
      const passes = stamps.map((at, index) => at - (index === 0 ? started : stamps[index - 1]));
      const row = { family, tokens, arm, round, pdbHash: hashText(out.pdb), contacts, meanPlddt: out.meanPlddt,
                    passMs: Math.round(median(passes.slice(1))), seconds: Math.round(out.seconds * 10) / 10,
                    leftMiB: Math.round(memoryTotals(device).residentBytes / 1048576), buffers: memoryTotals(device).bufferCount };
      console.log(JSON.stringify(row));
      report.push(row);
    }
  }
  return report;
}

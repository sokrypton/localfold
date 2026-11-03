/**
 * Does a whole fold of the AlphaFold 3 lineage come out the same as another checkout's, to the last digit it writes?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/hash-af3-fold.js --family=protenix2 --tokens=400 --recycles=3
 *
 * A fold at one seed on one device is one PDB, so a change that should not touch the arithmetic (how work reaches the
 * queue, what holds a tensor) is checked by running this on both sides and comparing the two numbers. `meanPlddt` is
 * printed whole, because a mean over every atom moves when any of them does.
 */
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function fnv1a(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}

export async function main(device, args) {
  const family = option(args, "family", "protenix2");
  const tokens = Number(option(args, "tokens", "400"));
  const recycles = Number(option(args, "recycles", "3"));
  const seed = Number(option(args, "seed", "0"));
  const sequence = UBIQUITIN.repeat(Math.ceil(tokens / UBIQUITIN.length)).slice(0, tokens);
  const weights = await loadAf3Weights(() => {}, family);
  const out = await foldAf3({
    sequence, calls: 25, recycles, seed, device, weights, alignment: null, maxMsaSequences: 1,
    onStatus: () => {}, onProgress: () => {},
  });
  return { family, tokens, recycles, seed, pdbHash: fnv1a(out.pdb), meanPlddt: out.meanPlddt, seconds: Math.round(out.seconds * 10) / 10 };
}

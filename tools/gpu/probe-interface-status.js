/**
 * Does the status line carry p(inter), and only when there is an interface?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-interface-status.js
 *
 * WHY IT EXISTS. p(inter) comes from the trunk's cross-chain contacts and is
 * meant to appear WHILE the fold runs - before the confidence head has produced
 * a real ipTM - so it cannot be checked from a finished result. This captures
 * every status line the page emits and asks two things of them: a complex
 * prints a p(inter), and a single chain prints none at all, because a monomer
 * has no interface and a number claiming otherwise would be worse than
 * silence.
 */
import { foldAf3, loadAf3Weights } from "../../web/af3-model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const GCN4 = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER";
const MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";

export async function main(device, args) {
  const calls = Number(option(args, "steps", "4"));
  const weights = await loadAf3Weights(() => {});
  const run = async (sequence) => {
    const lines = [];
    await foldAf3({
      sequence, mode: "flow", calls, recycles: 0, seed: 3, device, weights,
      onStatus: (text) => { lines.push(text); }, onProgress: () => {},
    });
    const mentions = lines.filter((line) => line.includes("p(inter)"));
    return {
      statusLines: lines.length,
      mentioning: mentions.length,
      // ...and WHEN: it must arrive before the sampler is done, or it is not
      // buying the earliness it exists for.
      firstAt: mentions.length === 0 ? null : lines.indexOf(mentions[0]),
      example: mentions[0] ?? null,
    };
  };
  return {
    // A coiled-coil homodimer: the trunk should be confident these touch.
    dimer: await run(`${GCN4}:${GCN4}`),
    // ...and a pair the trunk should not be confident about.
    weakPair: await run(`${GCN4}:${MRR}`),
    // A single chain has no interface; the line must stay silent.
    monomer: await run(MRR),
  };
}

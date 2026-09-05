/**
 * Does the download dial's fraction actually reach 1?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-load-dial.js
 *
 * The dial is loadedBytes / totalBytes straight from the store, so a dial that
 * stops halfway is one of those two numbers being wrong. This reports the last
 * sample and the total the store computed, beside the bytes the shards
 * actually are.
 */
import { loadAf3Weights } from "../../web/af3-model.js";

export async function main(device, args) {
  void device; void args;
  let last = null;
  let samples = 0;
  await loadAf3Weights((progress) => { last = { ...progress }; samples += 1; });
  return {
    samples,
    loadedMiB: Number((last.loadedBytes / 1048576).toFixed(1)),
    totalMiB: Number((last.totalBytes / 1048576).toFixed(1)),
    fraction: Number((last.loadedBytes / last.totalBytes).toFixed(3)),
    loadedTensors: last.loadedTensors, totalTensors: last.totalTensors,
  };
}

/**
 * Every buffer a fold actually creates, against what the memory budget counts.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-live-buffers.js --tool=fold \
 *       --model=/model-intellifold2-int5/manifest.json --sequence=<...> --budget=0
 *
 * 🔴 THE BUDGET IS ONLY AS GOOD AS WHAT IT COUNTS. `noteAllocation`
 * (src/runtime/device-memory.js) is called by the allocators this port owns,
 * and the budget refuses a fold against that sum - but on an A100 the driver
 * reported 1.4-2.1x the tracked peak (IntelliFold-2 at 512 residues: 8.4 GB
 * against 4.0), and a T4 budgeted from its own size ran past the card and lost
 * the device. This wraps the tool the way probe-compiles.js does and sees every
 * `createBuffer` and `destroy` on the device, so the gap is either buffers the
 * accounting never hears of (named here, by label, at the moment of the live
 * peak) or memory held below WebGPU after a release, which no buffer explains.
 */
import { memorySnapshot } from "../../src/runtime/device-memory.js";

const option = (args, name) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);

export async function main(device, args) {
  const tool = option(args, "tool");
  if (tool === undefined) throw new Error("probe-live-buffers needs --tool=<module under tools/gpu>");
  const rest = args.filter((a) => !a.startsWith("--tool="));
  const live = new Map();
  const creates = [];
  let liveBytes = 0;
  let peak = { bytes: 0 };
  const realCreate = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = realCreate(descriptor);
    const entry = { label: descriptor.label ?? "(unlabelled)", bytes: descriptor.size };
    if (descriptor.size >= 32 * 1048576) {
      creates.push([Date.now(), entry.label, Math.round(descriptor.size / 1048576),
                    descriptor.mappedAtCreation === true ? "mapped" : ""]);
    }
    live.set(buffer, entry);
    liveBytes += entry.bytes;
    const realDestroy = buffer.destroy.bind(buffer);
    buffer.destroy = () => {
      if (live.delete(buffer)) liveBytes -= entry.bytes;
      return realDestroy();
    };
    if (liveBytes > peak.bytes) {
      const byLabel = new Map();
      for (const { label, bytes } of live.values()) {
        const family = label.replace(/[-.:]\d+$/, "").replace(/-\d+(?=[-.])/g, "");
        byLabel.set(family, (byLabel.get(family) ?? 0) + bytes);
      }
      peak = { bytes: liveBytes, tracked: memorySnapshot(device).residentBytes,
               byLabel: [...byLabel].sort((a, b) => b[1] - a[1]).slice(0, 15) };
    }
    return buffer;
  };
  // ...and the live total over time, wall-clock stamped, so it can be laid
  // beside the driver's own counter (nvidia-smi -lms 100) after the run.
  const timeline = [];
  const sampler = setInterval(() => timeline.push([Date.now(), Math.round(liveBytes / 1048576)]), 200);
  const module = await import(`./${tool}.js`);
  const result = await module.main(device, rest);
  clearInterval(sampler);
  device.createBuffer = realCreate;
  const mib = (bytes) => Math.round(bytes / 1048576);
  return {
    tool,
    livePeakMiB: mib(peak.bytes),
    trackedAtThatMomentMiB: mib(peak.tracked ?? 0),
    untrackedAtPeakMiB: mib(peak.bytes - (peak.tracked ?? 0)),
    liveAtEndMiB: mib(liveBytes),
    peakByLabel: (peak.byLabel ?? []).map(([label, bytes]) => [label, mib(bytes)]),
    meanPlddt: result?.meanPlddt,
    timeline,
    creates,
  };
}

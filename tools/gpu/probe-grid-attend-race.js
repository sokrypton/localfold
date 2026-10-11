/**
 * What grid.attend's tiled-form race decides on this device, and what it saw.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-grid-attend-race.js
 *
 * webgpu/af3/trunk/grid-attend-race.js runs once per device before a pair track compiles, where no prior answers
 * `gridAttendTiled` (an Apple GPU: the M2 and the M5 report the same adapter and want opposite answers). This runs it
 * on its own, with the time it costs, so the decision and its margin can be read off any machine.
 */
import { forgetGridAttendRace, gridAttendRaceDetail, settleGridAttendTiling }
  from "../../webgpu/af3/trunk/grid-attend-race.js";
import { deviceTuning } from "../../webgpu/runtime/device-profile.js";

export async function main(device, args = []) {
  // (the race itself, not this browser's remembered answer - unless --remembered asks what a repeat visit gets)
  if (!args.includes("--remembered")) forgetGridAttendRace();
  const started = performance.now();
  await settleGridAttendTiling(device);
  const tuning = deviceTuning(device);
  return {
    raceMs: Number((performance.now() - started).toFixed(1)),
    detail: gridAttendRaceDetail(device),
    gridAttendTiled: tuning.gridAttendTiled,
    gridAttendTiledHalf: tuning.gridAttendTiledHalf,
  };
}

import { deviceProfile, deviceTuning } from "../../src/runtime/device-profile.js";
export async function main(device) {
  const p = deviceProfile(device);
  return { vendor: p.vendor, architecture: p.architecture, software: p.software,
    diffusionSplitK: deviceTuning(device).diffusionSplitK,
    diffusionTokenTile: deviceTuning(device).diffusionTokenTile,
    matrixConfigs: p.matrixConfigs.length };
}

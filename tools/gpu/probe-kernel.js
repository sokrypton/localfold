/**
 * Which attention kernel this device gets, and why.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-kernel.js
 *
 * 🔴 THE SUBGROUP KERNELS PIN `@subgroup_size(32)`, WHICH IS A CORRECTNESS
 * REQUIREMENT AND NOT A PREFERENCE. A device can carry both subgroup features
 * and still refuse that size: reported from a Pixel as "The subgroup_size
 * attribute (32) is not in the allowed range ([16, 16])" while building
 * block:attention:flash-subgroup-key32 - which fails PIPELINE CREATION, so the
 * fold stops rather than falling back. This prints the range the device
 * reports, whether 32 is inside it, and the kernel that gets chosen, which is
 * the first thing to ask when a fold dies on an unfamiliar GPU.
 *
 * A device that reports no range at all reads as "allowed" here, deliberately -
 * see allowsAttentionSubgroupSize - and buildAttentionFlashKernel catches the
 * refusal at build time instead.
 */
import { selectAttentionFlashKernel, supportsAttentionSubgroups, allowsAttentionSubgroupSize }
  from "../../src/evoformer/attention.js";

export async function main(device) {
  const info = device.adapterInfo ?? device.info ?? {};
  return {
    features: [...device.features].sort(),
    subgroupMinSize: info.subgroupMinSize ?? null,
    subgroupMaxSize: info.subgroupMaxSize ?? null,
    allowsSize32: allowsAttentionSubgroupSize(device),
    supportsSubgroups: supportsAttentionSubgroups(device, 32),
    chosen: selectAttentionFlashKernel(device, 32, "auto").cacheKey,
  };
}

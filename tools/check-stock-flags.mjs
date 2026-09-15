/**
 * Does every model fold on the browser a VISITOR has?
 *
 *     node tools/check-stock-flags.mjs
 *
 * 🔴 WHY THIS IS A GATE AND NOT A CURIOSITY. Every other gate in this
 * repository runs through `tools/gpu-chrome.mjs`, which passes
 * `--enable-dawn-features=vulkan_enable_f16_on_nvidia` and
 * `--enable-unsafe-webgpu`. A stock Chrome has NEITHER on any NVIDIA GPU -
 * Dawn gates f16 vendor-wide pending crbug.com/42251215, and the subgroup
 * matrix units are `chromium-experimental-` on every platform. So the
 * configuration every gate checks is not the configuration the site ships.
 *
 * That is not a hypothetical: measured this way, **OpenDDE and ESMFold2 did not
 * fold at all**, both on `extension 'f16' is not allowed in the current
 * environment`, for as long as those paths had existed. See docs/A100.md.
 *
 * It runs the four folds and nothing else - this gate is about a capability
 * being absent, which a fold either survives or does not. The differential
 * checkers are the flagged run's job.
 */
import { runFolds } from "./gate-folds.mjs";

// The fold list and the verdict are shared with check-portable-limits.mjs -
// see gate-folds.mjs for why there is one copy.
await runFolds({
  env: { LOCALFOLD_STOCK_FLAGS: "1" },
  question: "on a stock Chrome",
  broke: /enables (f16|subgroups|subgroup_size_control|chromium_experimental)|"error"|Error:|uncaptured/,
  arm: "stock",
});

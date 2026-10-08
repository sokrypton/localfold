// An alignment's features with the nearest-centre search on the WebGPU device - the host featuriser's plan and
// finishing (src/input/a3m-features.js) around src/input/nearest-centres-webgpu.js's search.
import { featureStats, finishA3mRecycle, makeA3mFeatures, paddedCodeWords, planA3mRecycles } from "./a3m-features.js";
import { assignNearestCentres } from "./nearest-centres-webgpu.js";
import { deviceTuning } from "../runtime/device-profile.js";

/**
 * The same features, with the nearest-centre search on the device.
 *
 * 🔴 THE ONLY DIFFERENCE IS WHERE THE ARGMAX RAN, and it is held to zero
 * differing assignments by tools/gpu/check-nearest-centres.js. Everything else
 * - the shuffling, the masking, the profile, the 49 channels - is the same
 * code as the host path, because they are literally the same two functions.
 */
export async function makeA3mFeaturesOnDevice(device, a3mText, tables, options = {}) {
  const { plans, context } = planA3mRecycles(a3mText, tables, options);
  const mark = performance.now();
  const searches = plans.map((plan) => ({
    ...paddedCodeWords(plan.centerCodes, context.encoded, plan.extras,
      plan.centers.length, context.length),
    centres: plan.centers.length,
  }));
  const assignments = await assignNearestCentres(device, searches);
  featureStats.nearestMs += performance.now() - mark;
  return plans.map((plan, index) => finishA3mRecycle(plan, assignments[index], context));
}

/**
 * The features, by whichever route is cheaper for THIS alignment.
 *
 * 🔴 ONE PLACE, BECAUSE THE SEAM HAS TWO CALLERS. src/af2/model/monomer.js and
 * src/af2/multimer/model.js both chose between the two paths with the same
 * expression, and this file's own history is what says not to leave a rule in
 * two homes - see the allow-list that went stale at exactly this seam and took
 * the contact overlay off the shipped page with it.
 *
 * The device path is flat in alignment depth and the host path is linear, so
 * below `deviceFeaturisationMinBytes` the dispatch costs more than the search;
 * see that knob for the table. Both return the same features.
 *
 * 🔴 `options.hostFeaturisation` STILL FORCES THE HOST, and still means what it
 * meant: the control arm for a differential, not a fallback. A device that
 * cannot run the kernel raises - this routes on SIZE, and never on failure.
 */
export async function makeA3mFeaturesFor(device, a3mText, tables, options = {}) {
  const minBytes = deviceTuning(device).deviceFeaturisationMinBytes;
  const small = typeof minBytes === "number"
    && typeof a3mText === "string" && a3mText.length < minBytes;
  return options.hostFeaturisation === true || small
    ? makeA3mFeatures(a3mText, tables, options)
    : await makeA3mFeaturesOnDevice(device, a3mText, tables, options);
}

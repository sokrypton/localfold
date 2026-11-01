// One recycle's A3M features in a Node worker thread, from a plan the parent made once: that recycle's
// nearest-centre search and finishing (searchA3mRecycle, finishA3mRecycle), its typed arrays transferred back
// rather than copied (cuda/af2/export_input.mjs runs one a recycle). Node only - the page builds features on the device.
import { parentPort, workerData } from "node:worker_threads";
import { searchA3mRecycle, finishA3mRecycle } from "./a3m-features.js";

const { plan, context } = workerData;
const features = finishA3mRecycle(plan, searchA3mRecycle(plan, context), context);
const buffers = Object.values(features).filter(ArrayBuffer.isView).map((view) => view.buffer);
parentPort.postMessage(features, [...new Set(buffers)]);

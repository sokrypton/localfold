// One recycle's A3M features in a Node worker thread: makeA3mFeatureRecycle on what the parent hands
// over, its typed arrays transferred back rather than copied (native/af2/export_input.mjs runs one a
// recycle). Node only - the page builds features on the device instead.
import { parentPort, workerData } from "node:worker_threads";
import { makeA3mFeatureRecycle } from "./a3m-features.js";

const { a3m, tables, options, index } = workerData;
const features = makeA3mFeatureRecycle(a3m, tables, options, index);
const buffers = Object.values(features).filter(ArrayBuffer.isView).map((view) => view.buffer);
parentPort.postMessage(features, [...new Set(buffers)]);

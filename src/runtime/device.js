/** Requests optional WebGPU features used by LocalFold's exact fast paths. */
export async function requestAlphaFoldDevice(adapter) {
  // subgroup-size-control is shipping ahead of the current @webgpu/types union.
  const optional = ["subgroups", "subgroup-size-control", "timestamp-query"];
  const requiredFeatures = optional.filter(
    (feature) => adapter.features.has(feature),
  );
  return adapter.requestDevice({ requiredFeatures });
}

import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { AtomGeometryGpu } from "../src/structure/geometry.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
describe.skipIf(!enabled)("all-atom geometry WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("matches official atom14 and atom37 positions", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const length = store.shape("feature_aatype_recycle3")[0];
    const affineAll = await store.tensor("structureStage_affine_output");
    const anglesAll = await store.tensor("structureStage_angles");
    const tables = {
      defaultFrames: await store.tensor("geometryDefaultFrames"),
      atom14ToGroup: await store.tensor("geometryAtom14ToGroup"),
      atom14Positions: await store.tensor("geometryAtom14Positions"),
      atom14Mask: await store.tensor("geometryAtom14Mask"),
    };
    const result = await new AtomGeometryGpu(device).run({
      affine: affineAll.subarray(7 * length * 7, 8 * length * 7),
      angles: anglesAll.subarray(7 * length * 14, 8 * length * 14),
      aatype: await store.tensor("feature_aatype_recycle3"),
      atom37ToAtom14: await store.tensor("feature_residx_atom37_to_atom14_recycle3"),
      atom37Mask: await store.tensor("feature_atom37_atom_exists_recycle3"),
      length, tables,
    });
    const atom14All = await store.tensor("structureStage_atom14");
    const atom14Metrics = errorMetrics(result.atom14, atom14All.subarray(7 * length * 14 * 3));
    const atom37Metrics = errorMetrics(result.atom37, await store.tensor("structureFinalAtomPositions"));
    expect(atom14Metrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(atom14Metrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(atom37Metrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(atom37Metrics.maxAbsoluteError).toBeLessThan(3e-3);
  });
});

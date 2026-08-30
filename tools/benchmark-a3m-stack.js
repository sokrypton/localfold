import { create, globals } from "webgpu";
import { EvoformerStackGpu } from "../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";

Object.assign(globalThis, globals);
const adapterName = process.env.LOCALFOLD_ADAPTER;
const gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const device = await requestAlphaFoldDevice(adapter);

try {
  const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
  const weights = await fixture.mainStackWeights();
  const sequences = fixture.shape("feature_msa_feat_recycle0")[0];
  const length = 59;
  const cM = 256;
  const inputMsa = await fixture.tensor("stackRecycle0InputMsa");
  const profileBlockText = process.env.LOCALFOLD_PROFILE_BLOCK;
  const profileBlock = profileBlockText === undefined ? undefined : Number(profileBlockText);
  const submissionWindowText = process.env.LOCALFOLD_SUBMISSION_WINDOW;
  const submissionWindow = submissionWindowText === undefined ? undefined : Number(submissionWindowText);
  const result = await new EvoformerStackGpu(device).run({
    msa: inputMsa.subarray(0, sequences * length * cM),
    pair: await fixture.tensor("stackRecycle0InputPair"),
    msaMask: await fixture.tensor("feature_msa_mask_recycle0"),
    pairMask: await fixture.tensor("stackPairMask"),
    sequences,
    length,
    cM,
    cZ: 128,
    cOuter: 32,
    triangleHidden: 128,
    blockWeights: weights,
    ...(profileBlock === undefined ? {} : { profileBlock }),
    ...(submissionWindow === undefined ? {} : { submissionWindow }),
  });
  const expectedMsa = (await fixture.tensor("stackRecycle0ExpectedMsa"))
    .subarray(0, sequences * length * cM);
  const report = {
    adapter: adapter.info,
    shape: { sequences, length, cM, cZ: 128, blocks: result.blocks },
    elapsedMilliseconds: result.elapsedMilliseconds,
    memory: result.memory,
    error: {
      msa: errorMetrics(result.msa, expectedMsa),
      pair: errorMetrics(result.pair, await fixture.tensor("stackRecycle0ExpectedPair")),
    },
    ...(result.timestampProfile === undefined ? {} : {
      timestampProfile: result.timestampProfile,
      timestampTotalNanoseconds: result.timestampProfile.reduce((sum, entry) => sum + entry.nanoseconds, 0),
    }),
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  device.destroy();
}

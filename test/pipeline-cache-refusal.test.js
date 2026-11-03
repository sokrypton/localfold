import { afterEach, describe, expect, it, vi } from "./harness.js";
import { pipelineCacheForDevice } from "../webgpu/runtime/pipeline-cache.js";

const SOURCE = `@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1) fn main() { data[0] = 1u; }`;
/** A kernel that bakes a length in, so that a second length shares one compile through a uniform. */
const OF_LENGTH = (length) => `const LENGTH: u32 = ${length}u;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1) fn main() { data[0] = LENGTH + LENGTH; }`;

/** Refuses the first `refusals` compiles, and counts them all. */
function fakeDevice(refusals) {
  const device = {
    features: new Set(),
    limits: {},
    asked: 0,
    queue: { writeBuffer() {} },
    createBuffer: ({ size }) => ({ size }),
    createBindGroup: () => ({}),
    createShaderModule: ({ code }) => ({ code }),
    createComputePipelineAsync() {
      device.asked += 1;
      return device.asked <= refusals ? Promise.reject(new Error("refused")) : Promise.resolve({ id: device.asked, getBindGroupLayout: () => ({}) });
    },
  };
  return device;
}

const settled = (promise) => promise.then(() => "compiled", (error) => error.message);
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("a pipeline a driver refuses", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("is not kept: the next request asks again, and gets it", async () => {
    const device = fakeDevice(1);
    const cache = pipelineCacheForDevice(device);
    expect(await settled(cache.get("k", SOURCE))).toBe("refused");
    await tick();
    expect(await settled(cache.get("k", SOURCE))).toBe("compiled");
    expect(device.asked).toBe(2);
  });

  it("takes what shared its source with it", async () => {
    const device = fakeDevice(1);
    const cache = pipelineCacheForDevice(device);
    const first = cache.get("one", SOURCE);
    const second = cache.get("two", SOURCE);
    expect(await settled(first)).toBe("refused");
    expect(await settled(second)).toBe("refused");
    await tick();
    expect(await settled(cache.get("two", SOURCE))).toBe("compiled");
    expect(device.asked).toBe(2);
  });

  it("is not kept as the kernel other lengths share", async () => {
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 64, COPY_DST: 8 });
    const device = fakeDevice(1);
    const cache = pipelineCacheForDevice(device);
    expect(await settled(cache.get("at-10", OF_LENGTH(10)))).toBe("refused");
    await tick();
    expect(await settled(cache.get("at-20", OF_LENGTH(20)))).toBe("compiled");
    expect(device.asked).toBe(2);
  });

  it("keeps what compiled", async () => {
    const device = fakeDevice(0);
    const cache = pipelineCacheForDevice(device);
    expect(await settled(cache.get("k", SOURCE))).toBe("compiled");
    await tick();
    expect(await settled(cache.get("k", SOURCE))).toBe("compiled");
    expect(device.asked).toBe(1);
  });
});

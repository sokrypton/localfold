/**
 * How fast can host bytes reach a STORAGE buffer on this device, and by which
 * route?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-upload-path.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-upload-path.js --mib=64 --chunk=1024
 *
 * 🔴 WHY IT EXISTS. The on-device weight decode moves 776 MiB of shard through
 * `writeBuffer` on an OpenDDE fold and 383 on an ESMFold2 one, and that call is
 * 250-310 ms of the first fold - more than the staging memcpy feeding it, more
 * than every compile, and about a fifth of the whole wall. Nothing had asked
 * whether `writeBuffer` is the fastest route or merely the obvious one.
 *
 * The arms, all landing in a buffer a compute shader could read:
 *
 *  - `writeBuffer`: today's path. Host array -> queue staging -> buffer.
 *  - `mappedAtCreation`: the bytes are written straight into the new buffer's
 *    own mapped range, so there is ONE copy instead of two - but the buffer is
 *    fresh every time, which is the allocation pattern the staging pool exists
 *    to avoid.
 *  - `mapWritePooled`: a pooled MAP_WRITE|COPY_SRC buffer, mapped, written,
 *    unmapped and copied on the DEVICE into the storage buffer. One host copy
 *    and one device copy, and the device copy is nearly free - but `mapAsync`
 *    is an await, and an await inside a weight loop is what the pairformer's
 *    own notes warn about.
 *  - `writeBufferChunked`: the same bytes as `writeBuffer` in many calls, which
 *    prices the CALL against the COPY. Trading one for the other is what a
 *    scatter-gather upload does, and on this box it was an exact wash.
 *
 * 🔴 ARMS INTERLEAVED AND MINIMUM TAKEN, because this box drifts. And every arm
 * ends with `onSubmittedWorkDone`, because a route that only queues the work
 * has not moved the bytes yet and would win every time.
 */

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

export async function main(device, args) {
  const totalMiB = Number(option(args, "mib", "64"));
  const chunkKiB = Number(option(args, "chunk", "1024"));
  const rounds = Number(option(args, "rounds", "5"));
  const bytes = totalMiB * 1024 * 1024;
  const chunkBytes = chunkKiB * 1024;
  const chunks = Math.max(1, Math.round(bytes / chunkBytes));

  // One source array, filled once: every arm moves the same bytes.
  const source = new Uint8Array(bytes);
  for (let at = 0; at < bytes; at += 4093) source[at] = at & 0xff;

  const target = device.createBuffer({ size: bytes, usage: STORAGE });
  const pooledMap = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
  });

  // 🔴 AND THE REAL WORKLOAD IS MANY SMALL UPLOADS, NOT ONE BIG ONE. OpenDDE's
  // fold does 445 of them averaging 1.2 MiB, each into its own buffer, and a
  // route that wins on 64 MiB in one call can lose on that: `mapAsync` is a
  // round trip through the queue, and one per upload is 445 of them. The
  // `Each` arms below are the same bytes as their namesakes, split up and
  // landing in separate buffers, with a POOL of mapped buffers rotating so that
  // an acquire finds one already mapped - which is the only shape in which the
  // map route is usable inside a weight loop.
  const perChunk = [];
  for (let index = 0; index < chunks; index += 1) {
    perChunk.push(device.createBuffer({ size: chunkBytes, usage: STORAGE }));
  }
  const POOL = Number(option(args, "pool", "4"));
  const mapPool = [];
  for (let index = 0; index < POOL; index += 1) {
    mapPool.push({
      buffer: device.createBuffer({
        size: chunkBytes, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      }),
      mapped: true,
    });
  }
  let mapWaits = 0;
  let waiters = [];
  const acquireMapped = async () => {
    for (;;) {
      const ready = mapPool.find((entry) => entry.mapped);
      if (ready !== undefined) { ready.mapped = false; return ready; }
      // 🔴 THE POOL IS EMPTY, WHICH IS THE FAILURE MODE THIS ARM IS FOR. Every
      // wait here is the map route back on the critical path; `mapWaits` says
      // how often, and a pool one buffer too small turns a 5x into a loss.
      mapWaits += 1;
      await new Promise((resolve) => waiters.push(resolve));
    }
  };
  const releaseMapped = (entry) => {
    // Re-mapped in the BACKGROUND, so the next acquire is synchronous. This is
    // what makes the route usable inside a weight loop: the await moves off the
    // critical path, exactly as the staging pool's release already does.
    void device.queue.onSubmittedWorkDone()
      .then(() => entry.buffer.mapAsync(GPUMapMode.WRITE))
      .then(() => {
        entry.mapped = true;
        const woken = waiters;
        waiters = [];
        for (const resolve of woken) resolve();
      });
  };

  const arms = {
    async writeBuffer() {
      device.queue.writeBuffer(target, 0, source, 0, bytes);
      await device.queue.onSubmittedWorkDone();
    },
    async writeBufferChunked() {
      for (let index = 0; index < chunks; index += 1) {
        const at = index * chunkBytes;
        device.queue.writeBuffer(target, at, source, at,
                                 Math.min(chunkBytes, bytes - at));
      }
      await device.queue.onSubmittedWorkDone();
    },
    async mappedAtCreation() {
      const staging = device.createBuffer({
        size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
      });
      new Uint8Array(staging.getMappedRange()).set(source);
      staging.unmap();
      const encoder = device.createCommandEncoder({ label: "upload.mapped" });
      encoder.copyBufferToBuffer(staging, 0, target, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      staging.destroy();
    },
    async mapWritePooled() {
      await pooledMap.mapAsync(GPUMapMode.WRITE);
      new Uint8Array(pooledMap.getMappedRange()).set(source);
      pooledMap.unmap();
      const encoder = device.createCommandEncoder({ label: "upload.pooled" });
      encoder.copyBufferToBuffer(pooledMap, 0, target, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    },
    async writeBufferEach() {
      // One submit per upload, because that is what runBlockUpload does: the
      // decode pass for a pack rides in the same command buffer as its bytes.
      for (let index = 0; index < chunks; index += 1) {
        device.queue.writeBuffer(perChunk[index], 0, source, index * chunkBytes, chunkBytes);
        device.queue.submit([device.createCommandEncoder().finish()]);
      }
      await device.queue.onSubmittedWorkDone();
    },
    async mapWriteEach() {
      for (let index = 0; index < chunks; index += 1) {
        const entry = await acquireMapped();
        new Uint8Array(entry.buffer.getMappedRange()).set(
          source.subarray(index * chunkBytes, (index + 1) * chunkBytes));
        entry.buffer.unmap();
        const encoder = device.createCommandEncoder({ label: "upload.each" });
        encoder.copyBufferToBuffer(entry.buffer, 0, perChunk[index], 0, chunkBytes);
        device.queue.submit([encoder.finish()]);
        releaseMapped(entry);
      }
      await device.queue.onSubmittedWorkDone();
    },
  };

  const names = Object.keys(arms);
  // A warm-up of every arm before any of them is timed: the first call of each
  // allocates, and a pool that is cold measures its allocator.
  for (const name of names) await arms[name]();

  const best = Object.fromEntries(names.map((name) => [name, Infinity]));
  for (let round = 0; round < rounds; round += 1) {
    for (const name of names) {
      const at = performance.now();
      await arms[name]();
      best[name] = Math.min(best[name], performance.now() - at);
    }
  }

  target.destroy();
  pooledMap.destroy();
  for (const buffer of perChunk) buffer.destroy();
  const round1 = (value) => Math.round(value * 10) / 10;
  return {
    totalMiB, chunkKiB, chunks, rounds,
    // How often the pooled route had to WAIT for a buffer rather than finding
    // one already mapped. A large number means the pool is too small and the
    // await is back on the critical path.
    mapWaits,
    arms: names.map((name) => ({
      arm: name,
      ms: round1(best[name]),
      gigabytesPerSecond: round1(bytes / (best[name] / 1000) / 1e9 * 10) / 10,
      relativeToWriteBuffer: Math.round(best.writeBuffer / best[name] * 100) / 100,
    })).sort((a, b) => a.ms - b.ms),
  };
}

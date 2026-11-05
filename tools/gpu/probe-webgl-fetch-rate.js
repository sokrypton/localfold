/**
 * What `fetch4` RATE does this device give a fragment of a given shape?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-fetch-rate.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-fetch-rate.js --k=68 --fragments=36992
 *
 * 🔴 THIS EXISTS TO SETTLE THE ONE QUESTION SEVEN NULL HYPOTHESES DID NOT.
 * webgl/README.md records the dense projection at about **6.4 G fetch4 a
 * second** and the triangle contraction at **1.0**, on the same device, with the
 * same eight-render-target structure and the same 32 output floats a fragment -
 * and the contraction's arithmetic intensity is only 2x lower, which cannot
 * explain 6x. Fragment count, memory-level parallelism, row alignment,
 * bandwidth, operand cache conflict, loop fusion and partial output writes have
 * all been measured and are all null.
 *
 * So this measures the RATE ITSELF as a function of the fragment's shape, with
 * everything else nailed down: the same two textures, the same k axis, the same
 * fragment count, the same 32 output floats. An arm is `R x C x B` - `B`
 * sequential BLOCKS, each reading `R` rows of a and `C` rows of b per k-step
 * and accumulating `R * C` products:
 *
 *   - `8x4x1` is the dense projection's shape: 12 fetches a k-step, 32 outputs.
 *   - `2x4x4` is the contraction's: 24 fetches a k-step, 32 outputs, and the
 *     four blocks read four DIFFERENT planes, as four channels do.
 *   - `4x4x2` and `2x4x1` interpolate.
 *
 * 🔴 AND IT NEEDS ROUNDS, BECAUSE ONE ROUND OF IT IS NOT A MEASUREMENT. The
 * first version ran each arm once after two warm-up dispatches and reported
 * 7.54, 16.88 and 17.04 G fetch4/s for the SAME arm on three consecutive runs -
 * a 2.3x spread, which is an Apple GPU's clock ramping under a load that lasts
 * fifty milliseconds, on a machine with another process on it. A bound founded
 * on the low sample understates the hardware and a bound founded on the high one
 * overstates it; `--rounds` interleaves the arms several times over and keeps
 * each arm's MINIMUM, and `spreadWithinArm` reports how far apart the rounds
 * were so a reader can see whether the number settled.
 *
 * 🔴 READ THE RATE, NOT THE TIME. The arms do different amounts of arithmetic
 * on purpose - what is held constant is the fragment count and the data - so
 * `gigaFetchesPerSecond` is the comparable column and `milliseconds` is not.
 * If the contraction's shape gives the projection's rate here, the structure is
 * innocent and the fault is in the kernel; if it does not, the rate is a
 * property of the SHAPE and the contraction is already at its ceiling.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const fill = (count, seed) => {
  const values = new Float32Array(count);
  let state = seed | 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 0.5;
  }
  return values;
};

/**
 * `blocks` sequential register tiles of `rows` by `columns`, into eight targets.
 *
 * Its own prelude so nothing but the shape varies between arms.
 */
const source = (rows, columns, blocks, k, plane) => {
  const targets = 8;
  const outputs = rows * columns * blocks;
  if (outputs > targets * 4) throw new Error(`${outputs} outputs is past the budget`);
  const component = ["x", "y", "z", "w"];
  const slot = (n) => `result${Math.floor(n / 4)}.${component[n % 4]}`;
  const block = (b) => `  {
${Array.from({ length: rows * columns }, (_, n) => `    float s${n} = 0.0;`).join("\n")}
    int base = ${b} * PLANE + row;
    for (int k4 = 0; k4 < K / 4; ++k4) {
${Array.from({ length: rows }, (_, r) =>
    `      vec4 a${r} = fetch4(aTex, (base + ${r} * K) / 4 + k4);`).join("\n")}
${Array.from({ length: columns }, (_, c) =>
    `      vec4 b${c} = fetch4(bTex, (base + ${c} * K) / 4 + k4);`).join("\n")}
${Array.from({ length: rows }, (_, r) => Array.from({ length: columns }, (_, c) =>
    `      s${r * columns + c} += dot(a${r}, b${c});`).join(" ")).join("\n")}
    }
${Array.from({ length: rows * columns }, (_, n) =>
    `    ${slot(b * rows * columns + n)} = s${n};`).join("\n")}
  }`;
  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = 2048;
const int K = ${k};
const int PLANE = ${plane};
${Array.from({ length: targets }, (_, t) =>
    `layout(location = ${t}) out vec4 result${t};`).join("\n")}
uniform sampler2D aTex;
uniform sampler2D bTex;

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
${Array.from({ length: targets }, (_, t) => `  result${t} = vec4(0.0);`).join("\n")}
  if (texel >= FRAGMENTS) return;
  // A row offset that moves with the fragment, so neighbours read neighbouring
  // rows exactly as both real kernels do, and every read is in range.
  int row = (texel % ROWBLOCKS) * ${rows + columns} * K;
${Array.from({ length: blocks }, (_, b) => block(b)).join("\n")}
}`;
};

/**
 * The ISSUE CEILING: `count` fetches a fragment from one small texture, so every
 * one after the first is an L1 hit, and nothing else is in the loop.
 *
 * 🔴 THIS IS WHAT webgl/README.md's FLOOR ARGUMENT RESTS ON AND IT WAS NEVER
 * MEASURED. The floor says a fragment kernel caps at 21.33 flops a `fetch4`
 * because 32 output floats is `MAX_DRAW_BUFFERS * 4`; multiply that by the
 * fetch RATE and you have the backend's ceiling in GFLOP/s. The rate had only
 * ever been inferred from kernels that also miss, stream and compute.
 *
 * The four reads are independent and their results are combined so nothing is
 * dead, and the texel index varies with the fragment so the compiler cannot
 * hoist the whole loop.
 */
const ceilingSource = (count, window, targets = 1) => `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = 2048;
const int COUNT = ${count};
const int WINDOW = ${window};
${Array.from({ length: targets }, (_, t) =>
    `layout(location = ${t}) out vec4 result${t};`).join("\n")}
uniform sampler2D aTex;

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  // 🔴 THE INDEX MUST NOT BE HOISTABLE. A first version walked i modulo 16 inside
  // a window of 64 texels, and a compiler that unrolls sixteen iterations can
  // hoist every one of those fetches out of the loop - which would report an
  // issue rate that is really a register read. The stride below has period
  // WINDOW, so hoisting it would mean unrolling the whole loop. (No backticks in
  // here: it is inside a JS template literal and one would end it.)
  int base = texel % 1024;
  vec4 total = vec4(0.0);
  for (int i = 0; i < COUNT / 4; ++i) {
    total += fetch4(aTex, base + (i * 53) % WINDOW);
    total += fetch4(aTex, base + (i * 53 + 1013) % WINDOW);
    total += fetch4(aTex, base + (i * 53 + 2027) % WINDOW);
    total += fetch4(aTex, base + (i * 53 + 3041) % WINDOW);
  }
${Array.from({ length: targets }, (_, t) =>
    `  result${t} = total * float(${t + 1});`).join("\n")}
}`;

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "24"));
  const rounds = Number(option(args, "rounds", "4"));
  const k = Number(option(args, "k", "68"));
  const fragments = Number(option(args, "fragments", "36992"));
  const arms = option(args, "arms", "8x4x1,4x4x2,2x4x4,2x4x1")
    .split(",").map((a) => a.split("x").map(Number));
  const gl = createWebGL2Context();
  const execution = new Execution(gl, { checkErrors: false });

  // One plane per block, each holding enough rows for every fragment's window.
  const rowBlocks = 64;
  const widest = Math.max(...arms.map(([r, c]) => r + c));
  const plane = rowBlocks * widest * k;
  const operand = plane * 8;
  const a = execution.upload(fill(operand, 7), "probe.a");
  const b = execution.upload(fill(operand, 11), "probe.b");

  const rows = [];

  // The issue ceiling first, so every shaped arm below can be read against it.
  const ceilingCount = Number(option(args, "ceiling-fetches", "1024"));
  // The texel window the ceiling arm walks, in TEXELS: 4096 is 64 KiB, which is
  // resident but far too big to hoist or to hold in registers.
  const ceilingWindow = Number(option(args, "ceiling-window", "4096"));
  const ceilingTargets = option(args, "ceiling-targets", "1,2,4,8")
    .split(",").map(Number);
  const ceilingFragments = 65536;
  {
    // 🔴 AND THE SAME LOOP AT ONE, TWO, FOUR AND EIGHT RENDER TARGETS. Eight
    // RGBA32F targets is 128 bytes a pixel of TILE MEMORY, and tile memory is
    // what bounds how many fragments a core keeps in flight - so the register
    // tile that buys arithmetic intensity may be spending the occupancy that
    // hides the fetch latency. Nothing else in the loop changes.
    for (const targets of ceilingTargets) {
      /** @type {number[]} */
      const samples = [];
      const ring = Array.from({ length: iterations }, () =>
        Array.from({ length: targets },
          () => execution.allocate(ceilingFragments * 4, "probe.ceiling")));
      const text = ceilingSource(ceilingCount, ceilingWindow, targets);
      const run = (outs) => execution.dispatch({
        source: text, outputs: outs, inputs: { aTex: a }, label: `probe.ceiling:${targets}`,
      });
      for (let i = 0; i < 2; i += 1) {
        run(ring[i % ring.length]);
        execution.read(ring[i % ring.length][0]);
      }
      for (let round = 0; round < rounds; round += 1) {
        const start = performance.now();
        for (let i = 0; i < iterations; i += 1) run(ring[i % ring.length]);
        execution.read(ring[0][0]);
        samples.push((performance.now() - start) / iterations);
      }
      const ms = Math.min(...samples);
      for (const set of ring) for (const out of set) execution.release(out);
      rows.push({
        arm: `ceiling:${targets}`,
        outputs: targets * 4,
        fetchesPerStep: 4,
        milliseconds: Number(ms.toFixed(2)),
        gigaFetchesPerSecond:
          Number((ceilingFragments * ceilingCount / ms / 1e6).toFixed(2)),
        gigaflops: 0,
        flopsPerFetch: 0,
        spreadWithinArm: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
      });
    }
  }

  for (const [r, c, blocks] of arms) {
    const outputs = r * c * blocks;
    const ring = Array.from({ length: iterations }, () =>
      Array.from({ length: 8 }, () => execution.allocate(fragments * 4, "probe.out")));
    const text = source(r, c, blocks, k, plane)
      .replace(/FRAGMENTS/g, String(fragments))
      .replace(/ROWBLOCKS/g, String(rowBlocks));
    const dispatch = (outs) => execution.dispatch({
      source: text, outputs: outs, inputs: { aTex: a, bTex: b },
      label: `probe.tile:${r}x${c}x${blocks}`,
    });
    for (let i = 0; i < 2; i += 1) {
      dispatch(ring[i % ring.length]);
      execution.read(ring[i % ring.length][0]);
    }
    /** @type {number[]} */
    const samples = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) dispatch(ring[i % ring.length]);
      execution.read(ring[0][0]);
      samples.push((performance.now() - start) / iterations);
    }
    const ms = Math.min(...samples);
    for (const set of ring) for (const out of set) execution.release(out);
    const fetches = fragments * blocks * (r + c) * (k / 4);
    const flops = fragments * blocks * r * c * k * 2;
    rows.push({
      arm: `${r}x${c}x${blocks}`,
      outputs,
      fetchesPerStep: blocks * (r + c),
      milliseconds: Number(ms.toFixed(2)),
      gigaFetchesPerSecond: Number((fetches / ms / 1e6).toFixed(2)),
      gigaflops: Number((flops / ms / 1e6).toFixed(1)),
      flopsPerFetch: Number((flops / fetches).toFixed(1)),
      spreadWithinArm: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
    });
  }
  execution.release(a);
  execution.release(b);
  const shaped = rows.filter((row) => !row.arm.startsWith("ceiling"));
  const ceiling = rows.filter((row) => row.arm.startsWith("ceiling"))
    .reduce((x, y) => (y.gigaFetchesPerSecond > x.gigaFetchesPerSecond ? y : x));
  const best = shaped.reduce((x, y) =>
    (y.gigaFetchesPerSecond > x.gigaFetchesPerSecond ? y : x));
  const worst = shaped.reduce((x, y) =>
    (y.gigaFetchesPerSecond < x.gigaFetchesPerSecond ? y : x));
  const renderer = gl.getParameter(gl.RENDERER);
  execution.destroy();
  return {
    renderer,
    k,
    fragments,
    iterations,
    rows,
    spread: Number((best.gigaFetchesPerSecond / worst.gigaFetchesPerSecond).toFixed(2)),
    // 🔴 THE BACKEND'S CEILING IN GFLOP/s, measured rather than inferred: the
    // issue rate times the 21.33 flops a `fetch4` that 32 output floats allow.
    issueCeilingGigaFetches: ceiling?.gigaFetchesPerSecond ?? null,
    backendCeilingGflops: ceiling
      ? Number((ceiling.gigaFetchesPerSecond * 21.333).toFixed(0)) : null,
    verdict: `the issue ceiling is ${ceiling?.gigaFetchesPerSecond} G fetch4/s, so a fragment `
      + `kernel at the 21.33 flops a fetch that 32 output floats allow caps at about `
      + `${ceiling ? Math.round(ceiling.gigaFetchesPerSecond * 21.333) : "?"} GFLOP/s; `
      + `${best.arm} reaches ${best.gigaFetchesPerSecond} G fetch4/s and ${worst.arm} `
      + `${worst.gigaFetchesPerSecond} - a ${(best.gigaFetchesPerSecond / worst.gigaFetchesPerSecond).toFixed(2)}x `
      + `spread at one fragment count on one pair of textures, so the rate is a property of the `
      + `fragment's SHAPE`,
  };
}

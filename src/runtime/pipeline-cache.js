import { deviceTuning } from "./device-profile.js";
/**
 * What the pipeline cache was handed and threw away.
 *
 * 🔴 A CACHE HIT STILL PAYS FOR ITS SOURCE. `get` takes the finished WGSL, and
 * a block asks for the same pipelines on every block of every recycle, so every
 * generated source after the first is built and discarded. Nothing measured it:
 * `shaderSourceMiB` in probe-compiles.js counts what reaches
 * `createShaderModule`, which is exactly the sources that were NOT wasted.
 */
/**
 * The WGSL extensions this repository emits, and the feature each one needs.
 *
 * 🔴 A SHADER THAT ENABLES ONE WITHOUT THE FEATURE FAILS AS A WGSL PARSE ERROR
 * NAMING NO KEY, which cost a whole search: OpenDDE and ESMFold2 both did it and
 * neither folded on a stock Chrome. See docs/A100.md.
 */
export const EXTENSION_FEATURES = [
  ["f16", "shader-f16"],
  ["subgroups", "subgroups"],
  ["subgroup_size_control", "subgroup-size-control"],
  ["chromium_experimental_subgroup_matrix", "chromium-experimental-subgroup-matrix"],
];

export const pipelineCacheStats = {
  hits: 0, misses: 0, shared: 0, hitSourceBytes: 0, byKey: new Map(),
};

export class ComputePipelineCache {
  device;
  #pipelines = new Map

  ();

  /**
   * The pipelines already built, by the WGSL and entry point they were built
   * from.
   *
   * 🔴 TWO KEYS CAN NAME THE SAME SHADER, AND 78 OF OpenDDE's 269 DO. Measured
   * with `distinctSources` in probe-compiles.js: 269 modules made from 191
   * distinct texts, so 29% of the compiles reproduce a pipeline that already
   * exists. The keys differ for good reasons - they carry a token count, a
   * direction, a geometry - and two different shapes can still generate
   * character-for-character the same kernel.
   *
   * The source itself is the key, not a hash of it: a hash collision here would
   * hand a caller somebody else's kernel, and this cache exists to make that
   * impossible rather than unlikely. The texts are held by the source memo in
   * src/runtime/shader-source-cache.js anyway, so this retains nothing new.
   *
   * 🔴 SAFE ONLY BECAUSE THE LAYOUT IS `auto` AND DERIVED FROM THE SOURCE. Two
   * pipelines built from identical WGSL with the same entry point have the same
   * bind group layouts by construction. The label differs and is cosmetic -
   * profile.js times labelled compute PASSES, not pipelines.
   */
  #byContent = new Map();

  constructor(device) {
    this.device = device;
  }

  /** Whether sources compile with opaque loop bounds; see withRuntimeLoopBounds. */
  get runtimeLoopBounds() {
    const value = deviceTuning(this.device).runtimeLoopBounds;
    return value === true || value === "tiered" ? value : false;
  }

  get(key, code, entryPoint = "main") {
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) {
      pipelineCacheStats.hits += 1;
      const wasted = typeof code === "string" ? code.length : 0;
      pipelineCacheStats.hitSourceBytes += wasted;
      const row = pipelineCacheStats.byKey.get(key)
        ?? pipelineCacheStats.byKey.set(key, { hits: 0, bytes: 0 }).get(key);
      row.hits += 1;
      row.bytes += wasted;
      if (cached.code !== code || cached.entryPoint !== entryPoint) {
        // 🔴 A COLLISION IS A KEY THAT DOES NOT NAME ITS SHADER, AND THE NEXT
        // QUESTION IS ALWAYS "WHICH PART". Saying only the key leaves that to a
        // bisect; the first differing line usually names the constant that
        // moved - a weight offset, a tile, a channel count - and is what turns
        // this from an afternoon into a minute.
        const was = cached.code.split("\n");
        const now = String(code).split("\n");
        let at = 0;
        while (at < Math.max(was.length, now.length) && was[at] === now[at]) at += 1;
        const detail = cached.entryPoint !== entryPoint
          ? `entry point ${cached.entryPoint} against ${entryPoint}`
          : `line ${at + 1} of ${now.length}: ${JSON.stringify((was[at] ?? "").trim().slice(0, 90))}`
            + ` against ${JSON.stringify((now[at] ?? "").trim().slice(0, 90))}`;
        throw new Error(`WebGPU pipeline cache key collision for ${key} - ${detail}`);
      }
      return cached.target.upgraded ?? cached.target.pipeline;
    }
    // 🔴 COMPILED WITHOUT ITS UNUSED CONSTANTS, SO TWO KERNELS THAT DIFFER
    // ONLY IN ONE THEY DO NOT READ ARE ONE PIPELINE. Most factories emit a
    // shared preamble - token count, widths, weight offsets - and a kernel
    // reads a few of them; 27 of an AF3 fold's 151 pipelines were copies of
    // another but for such a line. On a fresh Colab T4 a pipeline is ~85 ms of
    // driver compile on a user's first fold. The collision check above still
    // compares what the caller passed.
    const stripped = stripUnusedConstants(code);
    const mode = this.runtimeLoopBounds;
    const opaque = mode === false ? stripped : withRuntimeLoopBounds(stripped);
    const compiled = opaque;
    const content = `${entryPoint}\u0000${stripped}`;
    const shared = this.#byContent.get(content);
    if (shared !== undefined) {
      pipelineCacheStats.shared += 1;
      this.#pipelines.set(key, { code, entryPoint, target: shared });
      return shared.upgraded ?? shared.pipeline;
    }
    pipelineCacheStats.misses += 1;
    // 🔴 A SHADER THAT ASKS FOR f16 ON A DEVICE WITHOUT IT FAILS AS A WGSL PARSE
    // ERROR WITH NO KEY, WHICH IS HOURS. Dawn reports "extension 'f16' is not
    // allowed in the current environment" against an uncaptured device error,
    // and nothing in it says which of the twelve shader factories emitted the
    // `enable f16`. It is a real shipped state and not a hypothetical: a stock
    // Chrome has no `shader-f16` on ANY NVIDIA GPU (see docs/A100.md), and two
    // of the four models were reaching here with it. Name the key instead.
    // 🔴 EVERY EXTENSION, NOT JUST f16, AND NOT ONLY THE FIRST LINE. The first
    // version of this read `code.startsWith("enable f16;")`, which is true of
    // the sources that happened to put f16 first and silently false of any that
    // do not - a guard with a false negative built in. WGSL requires every
    // `enable` before the first declaration, so scanning the leading directives
    // is exact, and the three extensions this repository emits each map to a
    // feature a device may not have.
    for (const [extension, feature] of EXTENSION_FEATURES) {
      if (!new RegExp(`(^|\\n)\\s*enable\\s+${extension}\\s*;`).test(code)) continue;
      if (this.device.features.has(feature)) continue;
      throw new Error(`${key} enables ${extension} on a device without `
        + `${feature}. The caller must gate on the feature - see `
        + "halfPrecisionAvailable and deviceProfile in src/runtime/device-profile.js. "
        + "A stock Chrome has neither shader-f16 nor the subgroup matrix units on "
        + "NVIDIA; see docs/A100.md.");
    }
    const pipeline = this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code: compiled }),
          entryPoint,
        },
      });
    const target = { pipeline };
    this.#pipelines.set(key, { code, entryPoint, target });
    this.#byContent.set(content, target);
    // 🔴 TIERED: THE UNROLLED KERNEL FOLLOWS, ONE AT A TIME, BEHIND THE FOLD.
    // The opaque-bound variant compiles ~12x faster and runs up to 1.5x slower
    // (a T4's warm AF3 fold at 68 residues, 1.2 -> 1.9 s), so a first fold takes
    // it and the constant-bound one compiles in the background, serially so it
    // holds one of a Colab VM's two cores; a later `get` returns it once ready.
    if (mode === "tiered" && opaque !== stripped) {
      this.#upgrades = this.#upgrades
        .then(() => pipeline)
        .then(() => this.device.createComputePipelineAsync({
          label: `${key}.unrolled`, layout: "auto",
          compute: { module: this.device.createShaderModule({ label: `${key}.unrolled.wgsl`,
            code: stripped }), entryPoint },
        }))
        .then((unrolled) => { target.upgraded = Promise.resolve(unrolled); }, () => {});
    }
    return pipeline;
  }

  #upgrades = Promise.resolve();

  get size() {
    return this.#pipelines.size;
  }
}

/**
 * WGSL with every `const NAME ... = ...;` line whose name appears nowhere else
 * removed, repeated until none is left (a constant read only by another unused
 * one goes in the second round). Only whole single-line declarations are
 * touched, and a name mentioned anywhere - a comment included - is kept, so the
 * rule errs towards keeping a line. Exported for its test.
 */
/**
 * 🔴 CONSTANT-BOUND LOOPS, GIVEN A BOUND THE DRIVER CANNOT FOLD. A loop like
 * `for (var c = 0u; c < C; c += 1u)` over a `const C` is unrolled by NVIDIA's
 * compiler, and a kernel with a dozen of them - the atom encoder's `output` -
 * took 800-1029 ms to compile on a Colab T4 against 66-93 ms with the bound
 * made opaque. This adds `arrayLength(&<first runtime-sized storage array>)
 * >> 31u` to each such bound: zero for any buffer under 8 GiB, uniform, and not
 * known when the shader is compiled. The arithmetic is untouched. Opt-in per
 * device (`runtimeLoopBounds`), because an unrolled hot loop can also be the
 * faster one at run time. Exported for its test.
 */
export function withRuntimeLoopBounds(code) {
  if (typeof code !== "string") return code;
  const binding = /var<storage,\s*[a-z_]+>\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*array<[^,>]+>\s*;/.exec(code);
  if (binding === null) return code;
  const zero = `(arrayLength(&${binding[1]}) >> 31u)`;
  // ...every loop whose bound is a constant, lane-strided ones included: that
  // is 4.2 s of an AF3 fold's compile on the T4 where only the `= 0u; += 1u`
  // form was 6.9 (and all of them constant, ~13).
  // Only a bound declared u32: `i32 + u32` would not compile, and a shader
  // that fails to build fails the fold.
  const unsigned = new Set([...code.matchAll(
    /\bconst\s+([A-Z][A-Z0-9_]*)\s*(?::\s*u32\s*=|=\s*\d+u\s*;)/g)].map((m) => m[1]));
  return code.replace(/for \(([^;]*); (\w+) < ([A-Z][A-Z0-9_]*);/g,
    (whole, init, name, bound) => (unsigned.has(bound)
      ? `for (${init}; ${name} < ${bound} + ${zero};` : whole));
}

export function stripUnusedConstants(code) {
  if (typeof code !== "string" || !code.includes("const ")) return code;
  let lines = code.split("\n");
  for (;;) {
    const counts = new Map();
    for (const name of lines.join("\n").match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) ?? []) {
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    const kept = lines.filter((line) => {
      const match = /^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(:[^=;]*)?=[^;]*;\s*$/.exec(line);
      return match === null || counts.get(match[1]) !== 1;
    });
    if (kept.length === lines.length) return kept.join("\n");
    lines = kept;
  }
}

const DEVICE_PIPELINE_CACHES = new WeakMap();

/**
 * Returns the pipeline cache owned by a device.
 *
 * AlphaFold executes the same kernels in every block and recycle. Keeping this
 * cache at device lifetime avoids asking the browser to recreate identical
 * compute pipelines whenever a short-lived operator/execution object is made.
 */
export function pipelineCacheForDevice(device) {
  let cache = DEVICE_PIPELINE_CACHES.get(device);
  if (cache === undefined) {
    cache = new ComputePipelineCache(device);
    DEVICE_PIPELINE_CACHES.set(device, cache);
  }
  return cache;
}

/**
 * An object of pipeline promises, awaited together: `{ a: cache.get(...), b:
 * cache.get(...) }` -> `{ a, b }`. Asking for every pipeline before awaiting any
 * is what lets the browser compile them in parallel; awaiting each as it is
 * asked for puts every compile of a stage on a cold fold's path, one after
 * another.
 */
export async function settleAll(promises) {
  return Object.fromEntries(await Promise.all(Object.entries(promises)
    .map(async ([name, promise]) => [name, await promise])));
}

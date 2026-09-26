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
  hits: 0, misses: 0, shared: 0, generic: 0, hitSourceBytes: 0, byKey: new Map(),
};

/** How long the cache must go without a new pipeline before upgrading one. */
const UPGRADE_QUIET_MS = 2000;

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
      cached.target.requests += 1;
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
      shared.requests += 1;
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
    const generic = mode === "tiered" ? this.#lengthGeneric(key, stripped, entryPoint) : null;
    const pipeline = generic ?? this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code: compiled }),
          entryPoint,
        },
      });
    const target = { pipeline, requests: 1 };
    this.#pipelines.set(key, { code, entryPoint, target });
    this.#byContent.set(content, target);
    // 🔴 TIERED: THE UNROLLED KERNEL FOLLOWS, ONE AT A TIME, BEHIND THE FOLD.
    // The opaque-bound variant compiles ~12x faster and runs up to 1.5x slower
    // (a T4's warm AF3 fold at 68 residues, 1.2 -> 1.9 s), so a first fold takes
    // it and the constant-bound one compiles in the background, serially so it
    // holds one of a Colab VM's two cores; a later `get` returns it once ready.
    // 🔴 AND ONLY ONCE THE FOLD HAS STOPPED ASKING FOR PIPELINES. Started
    // beside the first fold, the unrolled compiles took one of a Colab VM's two
    // cores from it - ESMFold2's first fold went 4.5 -> 6.7 s - so they wait
    // for a quiet spell with no new request (see #drainUpgrades).
    this.#lastMiss = performance.now();
    if (mode === "tiered" && (opaque !== stripped || generic !== null)) {
      this.#pendingUpgrades.push({ key, target, code: stripped, entryPoint, pipeline });
      this.#drainUpgrades();
    }
    return pipeline;
  }

  /**
   * 🔴 A NEW PROTEIN LENGTH RECOMPILED EVERY KERNEL, BECAUSE EVERY KERNEL BAKES
   * THE LENGTH IN. A Colab T4 spends ~5 s of a fold compiling them, and a
   * second protein of another length paid it again. Most of those kernels
   * differ between two lengths only in `const NAME: u32 = <n>u;` lines (77 of
   * AF3's 104 at 68 and 70 residues), so the cache groups kernels by their
   * text with those values blanked, and the second time it meets one at new
   * values it compiles ONE generic kernel that reads the differing constants
   * from a uniform in bind group 1. Every later length reuses it with its own
   * values - no compile - while the length-specific kernel compiles behind the
   * fold as a tiered upgrade, exactly as the unrolled one does.
   *
   * It is the same arithmetic in the same order: only integers that were
   * compile-time constants become uniforms, and a kernel that uses one where
   * WGSL needs a constant (an array size, a workgroup size, another constant)
   * is refused and compiled length-specific as before.
   */
  #bySkeleton = new Map();

  #lengthGeneric(key, stripped, entryPoint) {
    const plan = lengthPlan(stripped);
    if (plan.generic.length === 0) return null;
    const { skeleton, values } = lengthSkeleton(stripped, plan);
    const groupKey = `${entryPoint}\u0000${skeleton}`;
    let generic = this.#bySkeleton.get(groupKey);
    if (generic === undefined) {
      const source = genericLengthSource(stripped, plan);
      generic = source === null ? null : this.device.createComputePipelineAsync({
        label: `${key}.generic`, layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.generic.wgsl`,
            code: this.runtimeLoopBounds === false ? source : withRuntimeLoopBounds(source) }),
          entryPoint,
        },
      });
      this.#bySkeleton.set(groupKey, generic);
    } else {
      pipelineCacheStats.generic += 1;
    }
    if (generic === null) return null;
    return generic.then((pipeline) => withLengthValues(this.device, pipeline,
      plan.generic.map((name) => Number(values.get(name)))));
  }

  #lastMiss = 0;
  #pendingUpgrades = [];
  #draining = false;

  /** Compile the queued unrolled kernels one at a time, each after a quiet spell. */
  async #drainUpgrades() {
    if (this.#draining) return;
    this.#draining = true;
    try {
      while (this.#pendingUpgrades.length > 0) {
        const quiet = UPGRADE_QUIET_MS - (performance.now() - this.#lastMiss);
        if (quiet > 0) { await new Promise((resolve) => setTimeout(resolve, quiet)); continue; }
        // The most-requested kernel first: a stack asks again per dispatch, so
        // the count is how hot it is, and the hot ones are the warm fold's time.
        let pick = 0;
        this.#pendingUpgrades.forEach((entry, at) => {
          if (entry.target.requests > this.#pendingUpgrades[pick].target.requests) pick = at;
        });
        const { key, target, code, entryPoint, pipeline } = this.#pendingUpgrades.splice(pick, 1)[0];
        try {
          await pipeline;
          const unrolled = await this.device.createComputePipelineAsync({
            label: `${key}.unrolled`, layout: "auto",
            compute: { module: this.device.createShaderModule({ label: `${key}.unrolled.wgsl`,
              code }), entryPoint },
          });
          target.upgraded = Promise.resolve(unrolled);
        } catch { /* the opaque kernel stays; it computes the same thing */ }
      }
    } finally {
      this.#draining = false;
    }
  }

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

/** A module-scope `const NAME[: type] = <expression>;` on one line. */
const MODULE_CONSTANT = /^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([a-z0-9<>]+)\s*)?=\s*([^;]+?)\s*;\s*$/;
const U32_LITERAL = /^(\d+)u$/;

/**
 * Which of a kernel's module constants can be read at run time instead: every
 * `const NAME[: u32] = <n>u;` that code reads, unless it (or a constant derived
 * from it) sits where WGSL needs a constant - an array size, a workgroup size,
 * an override, or a constant this does not inline. `derived` are the module
 * constants computed from those, which are inlined as their expressions.
 * Exported for its test.
 */
export function lengthPlan(code) {
  const lines = code.split("\n");
  const uncommented = lines.map((line) => line.replace(/\/\/.*$/, ""));
  const text = uncommented.join("\n");
  const uses = (name) => (text.match(new RegExp(`\\b${name}\\b`, "g")) ?? []).length;
  const declared = new Map();
  for (const [index, line] of uncommented.entries()) {
    const match = MODULE_CONSTANT.exec(line);
    if (match !== null) declared.set(match[1], { index, type: match[2], rhs: match[3] });
  }
  const refersTo = (rhs, names) => [...names].some((name) => new RegExp(`\\b${name}\\b`).test(rhs));
  let generic = new Set([...declared].filter(([name, d]) => U32_LITERAL.test(d.rhs)
    && (d.type === undefined || d.type === "u32") && uses(name) > 1).map(([name]) => name));
  for (;;) {
    // ...the module constants computed from them, transitively.
    const derived = new Set();
    for (let grew = true; grew;) {
      grew = false;
      for (const [name, d] of declared) {
        if (generic.has(name) || derived.has(name)) continue;
        if (refersTo(d.rhs, generic) || refersTo(d.rhs, derived)) { derived.add(name); grew = true; }
      }
    }
    const moving = new Set([...generic, ...derived]);
    // ...integers only: a float the compiler folds is not bit-for-bit the one
    // computed at run time, and this must be the same arithmetic.
    const floating = [...derived].filter((name) => {
      const { type, rhs } = declared.get(name);
      return (type !== undefined && type !== "u32" && type !== "i32") || /\d\.\d|\bf32\b|\bf16\b|sqrt|exp|log/.test(rhs);
    });
    const bad = [...floating, ...[...moving].filter((name) => uncommented.some((line, index) => {
      if (declared.get(name)?.index === index || !new RegExp(`\\b${name}\\b`).test(line)) return false;
      if (MODULE_CONSTANT.test(line)) return false;
      return /^\s*(const|override)\b/.test(line)
        || new RegExp(`array<[^;]*\\b${name}\\b`).test(line)
        || new RegExp(`workgroup_size\\([^)]*\\b${name}\\b`).test(line)
        // ...and not into a float: `x / f32(C)` folds to an exact constant,
        // and a GPU's run-time division is not exact.
        || new RegExp(`\\bf(32|16)\\([^;]*\\b${name}\\b`).test(line);
    }))];
    if (bad.length === 0) return { generic: [...generic], derived: [...derived] };
    // A refused name takes back every literal it is computed from.
    const back = new Set(bad);
    for (let grew = true; grew;) {
      grew = false;
      for (const name of back) {
        const rhs = declared.get(name)?.rhs ?? "";
        for (const other of generic) {
          if (!back.has(other) && new RegExp(`\\b${other}\\b`).test(rhs)) { back.add(other); grew = true; }
        }
      }
    }
    const next = new Set([...generic].filter((name) => !back.has(name)));
    if (next.size === generic.size) return { generic: [], derived: [] };
    generic = next;
  }
}

/**
 * The kernel's text with the plan's constants blanked, and their values - the
 * key two kernels share a generic pipeline under. Exported for its test.
 */
export function lengthSkeleton(code, plan = lengthPlan(code)) {
  const values = new Map();
  const moving = new Set(plan.generic);
  const skeleton = code.split("\n").map((line) => {
    const match = MODULE_CONSTANT.exec(line);
    if (match === null || !moving.has(match[1])) return line;
    values.set(match[1], U32_LITERAL.exec(match[3])[1]);
    return `const ${match[1]}: u32 = ?;`;
  }).join("\n");
  return { skeleton, values };
}

/**
 * The kernel with the plan's constants read from `localfold_lengths` in bind
 * group 1, and the ones derived from them inlined. Exported for its test.
 */
export function genericLengthSource(code, plan = lengthPlan(code)) {
  if (plan.generic.length === 0 || /@group\(\s*1\s*\)/.test(code)) return null;
  const at = new Map(plan.generic.map((name, index) => [name, index]));
  const derived = new Map();
  let lines = code.split("\n").filter((line) => {
    const match = MODULE_CONSTANT.exec(line);
    if (match === null) return true;
    if (plan.derived.includes(match[1])) {
      derived.set(match[1], match[2] === undefined ? `(${match[3]})` : `${match[2]}(${match[3]})`);
      return false;
    }
    return !at.has(match[1]);
  });
  let body = lines.join("\n");
  // ...derived constants as their expressions, until none is left.
  for (let round = 0; round < 16 && derived.size > 0; round += 1) {
    const before = body;
    body = body.replace(new RegExp(`\\b(${[...derived.keys()].join("|")})\\b`, "g"),
      (name) => derived.get(name));
    if (body === before) break;
  }
  body = body.replace(new RegExp(`\\b(${plan.generic.join("|")})\\b`, "g"),
    (name) => `localfold_lengths[${at.get(name) >> 2}].${"xyzw"[at.get(name) & 3]}`);
  return `${body}\n@group(1) @binding(0) var<uniform> localfold_lengths: `
    + `array<vec4<u32>, ${Math.ceil(plan.generic.length / 4)}>;\n`;
}

/** Which bind group 1 a generic pipeline handed out under one key needs. */
const LENGTH_VALUES = new WeakMap();

/**
 * A stand-in for `pipeline` that carries these values: `setPipeline` (patched
 * below) sets the real pipeline and its bind group 1. It answers
 * `getBindGroupLayout` and `label`, which is all a caller asks of a pipeline.
 */
function withLengthValues(device, pipeline, numbers) {
  const buffer = device.createBuffer({ label: `${pipeline.label}.lengths`,
    size: 16 * Math.ceil(numbers.length / 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const words = new Uint32Array(buffer.size / 4);
  words.set(numbers);
  device.queue.writeBuffer(buffer, 0, words);
  const group = device.createBindGroup({ label: `${pipeline.label}.lengths`,
    layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer } }] });
  const standIn = { label: pipeline.label, getBindGroupLayout: (index) => pipeline.getBindGroupLayout(index) };
  LENGTH_VALUES.set(standIn, { pipeline, group });
  return standIn;
}

if (typeof GPUComputePassEncoder !== "undefined") {
  const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function setLengthPipeline(pipeline) {
    const lengths = LENGTH_VALUES.get(pipeline);
    if (lengths === undefined) return setPipeline.call(this, pipeline);
    setPipeline.call(this, lengths.pipeline);
    this.setBindGroup(1, lengths.group);
  };
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

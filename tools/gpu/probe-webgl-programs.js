/**
 * Which WebGL2 programs a fold compiles and what each one costs, for ANY of the fold tools -
 * by wrapping one rather than by being one, as probe-compiles.js does for WGSL.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-programs.js --tool=fold-af2-webgl --cold
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-programs.js --tool=fold-bioemu2-webgl \
 *       --job=/tools/fixtures/bioemu2/6mrr-68.json --crystal=/tools/fixtures/6mrr-crystal.pdb
 *
 * 🔴 PASS `--cold`, OR THE DRIVER ANSWERS OUT OF A CACHE THIS REPOSITORY DOES NOT OWN. An AF2
 * WebGL fold compiled its 53 programs in **1507 ms** the first time those shaders had ever been
 * built on this machine and in **70 ms** on every run after, through a FRESH Chrome profile each
 * time: Metal keeps its own compiled-shader cache per user, outside the profile `gpu-chrome.mjs`
 * throws away. So a warm run measures a returning visitor and says a first visit is free. `--cold`
 * prepends a per-run salt to every shader text (through `shaderSource`, so the runtime's own
 * caches are untouched and only the DRIVER is defeated), which is the first-visit number and the
 * one Safari's compile freezes are about.
 *
 * 🔴 `probe-webgl-compile.js` ANSWERS THIS FOR AF3 ALONE, because it dry-runs `foldPrograms` -
 * which only the AF3 driver has. This one needs nothing of the kind: it times `Execution.program`
 * over whatever the tool draws, so AF2, ESMFold2 and bioemu2 are measurable too. The cost of that
 * is that a program is timed WHERE THE FOLD FIRST NEEDS IT, so a tool that calls `prepare` (the
 * AF3 fold does) compiles ahead of the first draw and those compiles are not in `slowest`.
 *
 * Read `kernelsPerFamily` before `slowest`: a family with one program is done, whatever it cost,
 * and a family with eleven is where the uniforms and the parts (`part`, `Execution.combine`) are
 * still worth taking - a label forks a program for every SHAPE until its sizes are uniforms.
 *
 * Every flag after `--tool` is passed to the tool untouched, so a gate does not stop being a gate
 * when it is measured.
 */
import { Execution, ONE_TARGET, PRELUDE } from "../../webgl/runtime/execution.js";
import { createWebGL2Context, linkProgram } from "../../webgl/runtime/context.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * Record every program this process compiles from here on, with the dispatch label that first
 * asked for it. Returns the rows; patching is permanent for the process, which is a probe's
 * privilege and not something a fold should do.
 *
 * @param {{cold?: boolean}} options `cold` salts every shader text so no driver cache can hit
 */
export function instrumentPrograms({ cold = false } = {}) {
  const rows = new Map();
  const state = { label: "prepare" };
  if (cold) {
    // 🔴 AND THE SALT MUST BE SEMANTIC, NOT A COMMENT. A comment defeats a cache keyed on the
    // GLSL text and nothing else: ANGLE translates to MSL and the comment is gone, so Metal's own
    // cache hits and `--cold` measured exactly the warm number (69 ms either way). A discard
    // guarded by a unique constant against `gl_FragCoord.x`, which is never negative and which no
    // compiler can prove, reaches the driver as different code and never runs.
    state.salt = (1e6 + Math.random() * 1e6).toFixed(3);
    const shaderSource = WebGL2RenderingContext.prototype.shaderSource;
    WebGL2RenderingContext.prototype.shaderSource = function saltedShaderSource(shader, source) {
      const { salt } = state;
      // The vertex stage is shared by every program and has no `gl_FragCoord`, so it is left be.
      const end = source.includes("gl_Position") ? -1 : source.lastIndexOf("}");
      const salted = end < 0 ? source
        : `${source.slice(0, end)}  if (gl_FragCoord.x < -${salt}) discard;\n${source.slice(end)}`;
      return shaderSource.call(this, shader, salted);
    };
  }
  // A recorded null, kept because it was the first hypothesis: `Execution.programs` is keyed by
  // the kernel's whole SOURCE TEXT, and a fold asks it once a dispatch. Hashing kilobytes of GLSL
  // 6154 times costs **1 ms** - V8 caches a string's hash, and the generated source is memoised
  // per shape - so the lookup is not where a first fold's host time is.
  const lookups = { hits: 0, hitMs: 0 };
  rows.lookups = lookups;
  rows.prepared = new Set();
  rows.state = state;
  const dispatch = Execution.prototype.dispatch;
  Execution.prototype.dispatch = function instrumentedDispatch(call) {
    state.label = call.label ?? "kernel";
    try {
      return dispatch.call(this, call);
    } finally {
      state.label = "prepare";
    }
  };
  // What `prepare` packed into each combined program, which is the question a regression in the
  // compile asks: a bin of twenty kernels is one link, and whether that is cheaper than twenty
  // depends on how much GLSL it holds (and, on Chrome, on the links it stops happening at once).
  const labels = new Map();
  const bins = [];
  rows.bins = bins;
  const collectPrograms = Execution.prototype.collectPrograms;
  Execution.prototype.collectPrograms = async function instrumentedCollect(run) {
    const collected = await collectPrograms.call(this, run);
    for (const [source, label] of collected) labels.set(source, label);
    return collected;
  };
  const prepare = Execution.prototype.prepare;
  Execution.prototype.prepare = async function instrumentedPrepare(sources) {
    // Materialised first: `prepare` is given an iterator by some callers and consumes it.
    const asked = [...sources];
    const result = await prepare.call(this, asked);
    // The programs it linked, which `program` never sees: what a browser without
    // KHR_parallel_shader_compile has to link one at a time on its own thread.
    for (const source of asked) rows.prepared.add(this.packaged(source));
    const held = new Map();
    for (const [source, { code }] of this.combined) {
      const bin = held.get(code) ?? { bytes: code.length, kernels: [] };
      bin.kernels.push(labels.get(source) ?? "?");
      held.set(code, bin);
    }
    bins.length = 0;
    bins.push(...held.values());
    return result;
  };
  const program = Execution.prototype.program;
  Execution.prototype.program = function instrumentedProgram(source) {
    const at = performance.now();
    const fresh = !this.programs.has(source);
    const entry = program.call(this, source);
    if (!fresh) {
      lookups.hits += 1;
      lookups.hitMs += performance.now() - at;
      return entry;
    }
    // A combined program (`Execution.combine`) carries several kernels' bodies; the count is read
    // off its own text, since `combined` is keyed by each PART's source rather than by this one.
    const kernels = (source.match(/^void kernel\d+\(\) \{$/gm) ?? []).length;
    rows.set(source, {
      label: state.label, ms: Math.round((performance.now() - at) * 10) / 10,
      bytes: source.length, source, ...(kernels > 1 ? { kernels } : {}),
    });
    return entry;
  };
  return rows;
}

/**
 * Could this finished source be a `part` without being rewritten by hand - that is, is everything
 * between the shared PRELUDE and its `void main` only uniform declarations and the one `result`
 * target? A kernel that also declares a helper, a module-scope `const` or several targets cannot
 * join a combined program as it stands, and the reason says which.
 *
 * This is the measurement that decides between converting ~150 kernels by hand and teaching the
 * runtime to convert the simple ones itself.
 */
export function partable(source) {
  const at = source.indexOf("void main()");
  if (at < 0) return "no main";
  const head = source.slice(0, at);
  const targets = head.match(/^\s*(layout\(location[^)]*\)\s*)?out\s+/gm) ?? [];
  if (targets.length !== 1 || !head.includes(ONE_TARGET.trim())) return "not one `result` target";
  // The prelude, the size uniforms `sized` injects into it, and the target are all expected.
  const rest = head
    .replace(PRELUDE, "")
    .replace(/#version 300 es\n/, "")
    .replace(ONE_TARGET.trim(), "")
    .replace(/^\s*uniform\s+[^;]+;$/gm, "")
    .replace(/^\s*\/\/.*$/gm, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .trim();
  if (rest === "") return "partable";
  if (/\bconst\b/.test(rest)) return "module-scope const";
  if (/\w+\s*\([^)]*\)\s*\{/.test(rest)) return "declares a helper";
  return "other declarations";
}

export async function main(device, args) {
  const tool = option(args, "tool");
  if (tool === undefined) throw new Error("probe-webgl-programs needs --tool=<module under tools/gpu>");
  const cold = args.includes("--cold");
  const rows = instrumentPrograms({ cold });
  const module = await import(`./${tool}.js`);
  const started = performance.now();
  const result = await module.main(device, args.filter((a) => !a.startsWith("--tool=") && a !== "--cold"));
  const wallMs = Math.round(performance.now() - started);
  // 🔴 AND THE SUM OF SERIAL LINKS, WHICH IS THE NUMBER SAFARI PAYS. Chrome has
  // KHR_parallel_shader_compile and `prepare` starts every link at once, so its wall is the
  // longest link and not the total - 53 AF2 programs come back in 519 ms, under 10 ms each. The
  // browsers this backend exists for link inside `linkProgram`, on the page's thread, ~19 ms each
  // on Safari 16.3. So every program the fold needs is linked again here, one at a time, in a
  // fresh context and under a fresh salt, and the total is what combining has to move.
  const linkedSerially = [...new Set([...rows.prepared, ...rows.keys()])];
  if (rows.state.salt !== undefined) rows.state.salt = (2e6 + Math.random() * 1e6).toFixed(3);
  const serialFrom = performance.now();
  if (!args.includes("--no-serial")) {
    const fresh = createWebGL2Context();
    for (const code of linkedSerially) linkProgram(fresh, code);
  }
  const serialMs = Math.round(performance.now() - serialFrom);
  const table = [...rows.values()];
  const family = (label) => String(label).split(":")[0];
  const families = new Map();
  for (const row of table) {
    const held = families.get(family(row.label)) ?? { programs: 0, ms: 0 };
    families.set(family(row.label),
      { programs: held.programs + 1, ms: Math.round((held.ms + row.ms) * 10) / 10 });
  }
  return {
    tool, cold, wallMs,
    programs: table.length,
    programMs: Math.round(table.reduce((sum, row) => sum + row.ms, 0)),
    sourceKiB: Math.round(table.reduce((sum, row) => sum + row.bytes, 0) / 1024),
    // How many of them are one combined program, and how many kernels those hold.
    combinedPrograms: table.filter((row) => row.kernels !== undefined).length,
    combinedKernels: table.reduce((sum, row) => sum + (row.kernels ?? 0), 0),
    lookupHits: rows.lookups.hits,
    lookupMs: Math.round(rows.lookups.hitMs),
    // What a runtime-side conversion could reach, and why the rest cannot (`partable`).
    partable: table.reduce((held, row) => {
      const why = row.kernels !== undefined ? "already combined" : partable(row.source);
      return { ...held, [why]: (held[why] ?? 0) + 1 };
    }, {}),
    // Each combined program: how much GLSL it holds and which kernels chose to share it.
    bins: rows.bins.map(({ bytes, kernels }) => ({ bytes, kernels: kernels.length,
      labels: kernels.map((label) => String(label).split(":")[0]) })),
    kernelsPerFamily: [...families.entries()]
      .map(([name, held]) => ({ family: name, ...held }))
      .sort((a, b) => b.ms - a.ms),
    slowest: table.sort((a, b) => b.ms - a.ms).slice(0, Number(option(args, "top", "20")))
      .map(({ source, ...row }) => ({ ...row, partable: row.kernels === undefined ? partable(source) : "already combined" })),
    // 🔴 AND THE FOLD'S OWN COUNT, BECAUSE `prepare` DOES NOT GO THROUGH `program`: a tool that
    // compiles ahead of its first draw links through `startProgram`, so `programs` above is 0 and
    // `programs_` is the real number. The wall is then what says what it cost.
    // Every program the fold needed, linked one at a time: the cost a thread-linking browser pays.
    serialPrograms: linkedSerially.length,
    serialMs,
    programs_: result?.memory?.programs ?? null,
    combined_: result?.memory === undefined ? null
      : [result.memory.combinedKernels, result.memory.combinedPrograms],
    // The wrapped tool's own clocks, which is where a fold's draw time is if combining cost any.
    ms_: result?.milliseconds ?? result?.ms ?? null,
    // The fold's own host split, which says what share of it the compiles are.
    host_: result?.memory?.host ?? null,
    tool_: Object.fromEntries(Object.entries(result ?? {})
      .filter(([, value]) => typeof value === "number" || typeof value === "boolean"
        || typeof value === "string")),
  };
}

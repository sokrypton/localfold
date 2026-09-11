/**
 * How much of the pairformer's compile cost is specialisation on the TOKEN
 * COUNT, and how deep the specialisation goes.
 *
 * 🔴 OpenDDE COMPILES ITS PAIRFORMER TWICE, at 68 residues and again at the 130
 * structural tokens its expander produces, and `af3-block` is 65 of the 191
 * pipelines a fold builds with **88% of the compile work** - 35.1 s of 39.6 s
 * of `sumMs`, against 41 ms a kernel for the atom stack. Sharing a pipeline
 * whose WGSL happens to be identical is already done
 * (ComputePipelineCache indexes by source); what is left is the kernels whose
 * text really does change with the count.
 *
 * This says which, and how much of the text moves. A kernel where the count
 * appears only in a bound or a guard could take it in a uniform and compile
 * once; a kernel where it is a LOOP TRIP COUNT could not - docs/AF2.md prices a
 * runtime loop bound in a hot WGSL loop at 4.3x, which is the whole reason the
 * chunk loops here are generated with their counts typed in.
 *
 * So this is a survey and not a proposal: it reports the size of the prize and
 * where the constants sit, and every candidate still has to be read.
 */
import { warmTrunkPipelines } from "../../src/af3/fold.js";
import { openAf3Store, STRUCTURAL_REFINER } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Capture every shader source built while `body` runs, by label. */
async function capture(device, body) {
  const seen = new Map();
  const make = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor) => {
    seen.set(descriptor.label ?? "?", descriptor.code ?? "");
    return make(descriptor);
  };
  try { await body(); } finally { device.createShaderModule = make; }
  return seen;
}

/** A label with its token-count field replaced, so two counts line up. */
const normalise = (label, tokens) =>
  String(label).split(":").map((part) => (part === String(tokens) ? "<tokens>" : part)).join(":");

/** Where two texts differ, and whether every difference is only a number. */
function compare(a, b) {
  const left = a.split("\n");
  const right = b.split("\n");
  let differing = 0;
  let numericOnly = true;
  const examples = [];
  for (let at = 0; at < Math.max(left.length, right.length); at += 1) {
    const one = left[at] ?? "";
    const two = right[at] ?? "";
    if (one === two) continue;
    differing += 1;
    // 🔴 THE TEST THAT MATTERS: strip every integer and see if the lines match.
    // If they do, the count is a constant in the text and nothing structural
    // moved; if they do not, the shape of the code itself changed - a different
    // unroll, a different number of chunks - and no uniform can express that.
    if (one.replace(/\d+/g, "#") !== two.replace(/\d+/g, "#")) numericOnly = false;
    if (examples.length < 2) examples.push(one.trim().slice(0, 72));
  }
  return { differing, numericOnly, lines: Math.max(left.length, right.length), examples };
}

export async function main(device, args = []) {
  const model = option(args, "model", "/model-opendde-int5/manifest.json");
  const low = Number(option(args, "low", "68"));
  const high = Number(option(args, "high", "130"));
  const store = await openAf3Store(model);

  const roots = [{ name: "trunk", options: {} },
                 { name: "refiner", options: { root: STRUCTURAL_REFINER,
                   stack: { pairWeightPrecision: undefined },
                   run: { extraPairBias: new Float32Array(0), keepPair: true } } }];
  const rows = [];
  for (const root of roots) {
    const at = new Map();
    for (const tokens of [low, high]) {
      at.set(tokens, await capture(device, () =>
        warmTrunkPipelines(device, store, tokens, root.options).catch(() => {})));
    }
    const byNormalised = new Map();
    for (const [tokens, sources] of at) {
      for (const [label, code] of sources) {
        const key = normalise(label, tokens);
        const entry = byNormalised.get(key) ?? {};
        entry[tokens] = code;
        byNormalised.set(key, entry);
      }
    }
    for (const [key, entry] of byNormalised) {
      const a = entry[low];
      const b = entry[high];
      if (a === undefined || b === undefined) {
        rows.push({ root: root.name, kernel: key.slice(0, 70), state: "one count only" });
        continue;
      }
      if (a === b) { rows.push({ root: root.name, kernel: key.slice(0, 70), state: "identical" }); continue; }
      const { differing, numericOnly, lines, examples } = compare(a, b);
      rows.push({ root: root.name, kernel: key.slice(0, 70),
                  state: numericOnly ? "numbers only" : "structural",
                  differing, lines, bytes: a.length, examples });
    }
  }

  const count = (state) => rows.filter((r) => r.state === state).length;
  const summary = {
    kernels: rows.length,
    identical: count("identical"),
    numbersOnly: count("numbers only"),
    structural: count("structural"),
    oneCountOnly: count("one count only"),
    // The bytes a shareable kernel would stop compiling twice.
    numbersOnlyBytes: rows.filter((r) => r.state === "numbers only")
      .reduce((total, r) => total + (r.bytes ?? 0), 0),
    structuralBytes: rows.filter((r) => r.state === "structural")
      .reduce((total, r) => total + (r.bytes ?? 0), 0),
  };
  console.log(JSON.stringify(summary));
  // Biggest first: the prize is bytes compiled twice, not kernels.
  for (const row of rows.filter((r) => r.state === "numbers only" || r.state === "structural")
    .sort((a, b) => (b.bytes ?? 0) - (a.bytes ?? 0))) {
    console.log(`${row.state.padEnd(12)} ${String(row.differing).padStart(4)}/`
      + `${String(row.lines).padEnd(5)} ${String(Math.round((row.bytes ?? 0) / 1024)).padStart(3)}K `
      + `${row.kernel.split(":").slice(-3).join(":")}`);
    for (const line of row.examples ?? []) console.log(`                  | ${line}`);
  }
  return { summary, rows };
}

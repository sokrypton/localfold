/**
 * AF3's transition block in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-transition.js [--falsify]
 *
 * `cpu/af3/trunk/pairformer.js`'s `transition` is the f64 reference - the one
 * AGENTS.md means by independent, since it shares no code with any shader - and
 * `Af3TransitionGpu` is the WGSL one, run in the same page. Two references,
 * because a single one cannot say which side is wrong: that rule has already
 * paid for itself twice in this port, on the triangle contraction's direction
 * and on an unpinned precision axis.
 *
 * 🔴 THREE CONVENTIONS, AND A WRONG ONE COSTS NO ERROR AND NO NaN:
 *
 *   - the split is BLOCKED: the gate half is the first `intermediate` outputs
 *     and the value half the second. Its twin in the triangle multiplication,
 *     forty lines away in AF3's own modules.py, is INTERLEAVED. The existing
 *     WGSL gate says what a mistake here reads as: "a wrong split convention or
 *     a swapped gate/value half lands near 1".
 *   - the activation is swish on the GATE half only, times the value half - not
 *     a ReLU, and not applied to the product.
 *   - the LayerNorm takes AF3's FAST variance, `E[x^2] - E[x]^2`. A port that
 *     centred first would be wrong on the single track, which reaches 1.7e5.
 *
 * 🔴 AND BOTH TRACKS RUN, because they exercise different corners: the pair
 * track is 128 channels widened by 4 and the single track 384 by 4, and the
 * widening factor is NOT always four - the template stack's is 2. This port
 * derives it from the weight's own length rather than taking it as an argument,
 * so the `--factor=2` arm checks that derivation against a reference told the
 * factor explicitly.
 */
import { transition as cpuTransition } from "../../cpu/af3/trunk/pairformer.js";
import { Af3TransitionGpu } from "../../webgpu/af3/trunk/transition.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  releaseTransitionWeights, transition, uploadTransitionWeights,
} from "../../webgl/af3/trunk/transition.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "3e-6"));
  const rows = Number(option(args, "rows", "17"));
  const factorArg = option(args, "factor", null);

  const random = generator(314159);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rowsList = [];
  const failures = [];

  /** @type {{name: string, channels: number, factor: number}[]} */
  const CASES = factorArg !== null
    ? [{ name: `factor ${factorArg}`, channels: 64, factor: Number(factorArg) }]
    : [
      { name: "pair track, 128 x 4", channels: 128, factor: 4 },
      { name: "single track, 384 x 4", channels: 384, factor: 4 },
      // The template stack's widening, which is where a derived factor earns
      // its keep: the same two weight names at half the width.
      { name: "template stack, 64 x 2", channels: 64, factor: 2 },
    ];

  for (const { name, channels, factor } of CASES) {
    const intermediate = channels * factor;
    const weights = {
      inputLayerNormScale: noise(channels, 0.2, 1),
      inputLayerNormOffset: noise(channels, 0.1),
      transition1: noise(channels * intermediate * 2, 1 / Math.sqrt(channels)),
      transition2: noise(intermediate * channels, 1 / Math.sqrt(intermediate)),
    };
    const input = noise(rows * channels, 1.5);
    // The control skews half of each row's channels: the input's first stop is
    // a LayerNorm, and a whole-row affine change is invisible to one.
    const skewed = falsify
      ? Float32Array.from(input,
        (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
      : input;

    const uploaded = uploadTransitionWeights(execution, weights);
    const x = execution.upload(skewed, `af3.transition.x.${channels}`);
    const update = transition(execution, { x, weights: uploaded, rows, channels });
    const ours = Float32Array.from(execution.read(update));
    execution.release(update);
    // 🔴 THE WINDOWED PATH, against the whole one: a transition whose `gated` is past the texture capacity goes a
    // window of rows at a time (protenix2's and mol1's every-row MSA stack). The capacity is shrunk so 17 rows take
    // four windows, the last ragged, with the residual inside as the MSA stack passes it.
    {
      const whole = transition(execution, { x, weights: uploaded, rows, channels, residualOf: x });
      const wholeValues = Float32Array.from(execution.read(whole));
      execution.release(whole);
      const capacity = execution.storage.capacity;
      execution.storage.capacity = 5 * 2 * intermediate;
      let windowedValues;
      try {
        const windowed = transition(execution, { x, weights: uploaded, rows, channels, residualOf: x });
        windowedValues = Float32Array.from(execution.read(windowed));
        execution.release(windowed);
      } finally {
        execution.storage.capacity = capacity;
      }
      const relative = relativeRms(windowedValues, wholeValues, `${name}:windowed`);
      rowsList.push({ arm: `${name}: four row windows vs whole`, relativeRms: Number(relative.toPrecision(4)),
        bar: 1e-6, ok: relative <= 1e-6, falsifiable: false, reason: "both arms read the same input" });
      if (!(relative <= 1e-6)) failures.push(`${name}: windowed differs from whole by ${relative.toExponential(3)}`);
    }
    execution.release(x);
    releaseTransitionWeights(execution, uploaded);

    const expected = cpuTransition(input, rows, channels, weights, factor);
    const theirs = await new Af3TransitionGpu(device).run(
      input, { rows, channels, factor }, weights, { variance: "fast" });

    for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs.output]]) {
      const relative = relativeRms(ours, want, `${name}:${label}`);
      const ok = relative <= bar;
      rowsList.push({
        arm: `${name} vs the ${label}`,
        relativeRms: Number(relative.toPrecision(4)), bar, ok,
      });
      if (!ok) {
        failures.push(`${name} vs ${label}: relRMS ${relative.toExponential(3)} over ${bar}`);
      }
    }
    // 🔴 AND THE TWO REFERENCES ARE HELD TO EACH OTHER TOO. If they disagree,
    // the instrument is wrong and neither number above means anything - the
    // tell this port has already had to read twice.
    const between = relativeRms(expected, theirs.output, `${name}:between`);
    rowsList.push({
      arm: `${name}: the two references agree`,
      relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
      falsifiable: false,
      reason: "both references read the unperturbed input",
    });
    if (!(between <= bar)) {
      failures.push(`${name}: the two references disagree by ${between.toExponential(3)}`);
    }
  }

  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = rowsList.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    rows,
    cases: CASES,
    bar,
    falsify,
    arms: rowsList,
    memory: execution.stats(),
    liveTensorsAfter: execution.stats().live,
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0 && execution.stats().live === 0
        ? `ok: ${rowsList.length} arms over ${CASES.length} shapes, both references agreeing`
        : `FAILED: ${failures.join("; ")}`),
  };
}

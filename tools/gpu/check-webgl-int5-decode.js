/**
 * The WebGL2 in-shader int5 decoder against `readTensor`, element by element.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-int5-decode.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-int5-decode.js --falsify
 *
 * 🔴 THE BAR IS EQUALITY, NOT A TOLERANCE, AND THAT IS NOT PEDANTRY. The host
 * decode is `code * scale + zero` in float32 and the shader's is the same
 * expression on the same three numbers, so every element must match exactly -
 * the scale and zero reach the GPU as the float32 values a `Float16Array` read
 * produces, not as a second conversion. The one difference the compiler is
 * allowed is fusing the multiply and the add, which moves a result by at most
 * one unit in the last place, so the check reports the count of elements that
 * differ at all beside the worst absolute gap and fails on anything past an ulp.
 *
 * 🔴 AND IT IS A DIFFERENTIAL BECAUSE A QUANTISED DECODER THAT IS WRONG STAYS
 * PLAUSIBLE. `tools/gpu/check-quantised-upload.js` found the WGSL device
 * decoder and the host one disagreeing on **104,170 of 131,072 elements** at
 * int5 group 64, and it was the HOST that was wrong; reading a code as a
 * multiple of the scale rather than an offset from the zero "stays finite,
 * stays smooth, and is a different model" (shared/weights/dtype.js). Neither
 * fault moves a fold enough for a geometry gate to see it.
 *
 * It runs on REAL bundle tensors - the largest int5 ones bioemu2 ships, which
 * are the ones the blocked projection decodes - because a synthetic group would
 * not exercise the bit offsets a real tensor's row boundaries produce.
 */
import { ONE_TARGET, PRELUDE } from "../../webgl/runtime/execution.js";
import { INT5_DECODE, INT5_GROUP } from "../../webgl/runtime/int5.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { readTensor } from "../../shared/weights/dtype.js";
import { webglExecution } from "./webgl-execution-options.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Decode every element into an ordinary float tensor, so the whole thing is comparable. */
const decodeAllSource = (falsify) => `${PRELUDE}${ONE_TARGET}
uniform int ELEMENTS;
uniform highp usampler2D codeTex;
uniform sampler2D tableTex;
${INT5_DECODE}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int element = texel * 4;
  int group = element >> 5;
  vec2 sz = int5ScaleZero(tableTex, group);
  ${falsify
    // 🔴 THE MUTATION IS A CODE READ AS A MULTIPLE OF THE SCALE, which is the
    // real fault this gate exists for and not an arbitrary perturbation.
    ? "result = decodeInt5x4(codeTex, element, sz.x, 0.0);"
    : "result = decodeInt5x4(codeTex, element, sz.x, sz.y);"}
}`;

export async function main(device, args) {
  const bundle = option(args, "model", "/model-bioemu2-int5").replace(/\/$/, "");
  const falsify = flag(args, "falsify");
  const wanted = Number(option(args, "tensors", "6"));
  const read = bundleReader(bundle);
  const manifest = await read.manifest?.() ?? null;
  const gl = createWebGL2Context();
  const execution = webglExecution(gl, [...(args ?? []), "--int5-weights"]);

  // The largest int5 tensors of a whole number of groups: the ones a fold's
  // blocked projection actually decodes.
  const table = manifest?.tensors ?? {};
  const candidates = Object.entries(table)
    .filter(([, record]) => record.dtype === "int5" && record.block === INT5_GROUP)
    .map(([name, record]) => ({ name, record, elements: record.shape.reduce((a, b) => a * b, 1) }))
    .filter((entry) => entry.elements % INT5_GROUP === 0)
    .sort((a, b) => b.elements - a.elements)
    .slice(0, wanted);
  if (candidates.length === 0) {
    throw new Error(`${bundle} has no int5 tensor at group ${INT5_GROUP} to check`);
  }

  const rows = [];
  for (const { name, elements } of candidates) {
    const source = await read.source(name);
    const uploaded = execution.uploadInt5Weight(source, `check.${name}`);
    if (uploaded === null) {
      rows.push({ tensor: name, elements, skipped: "not int5 at group 32, or too small" });
      continue;
    }
    const output = execution.allocate(elements, "check.decoded");
    execution.dispatch({
      source: decodeAllSource(falsify),
      output,
      inputs: { codeTex: uploaded.codes, tableTex: uploaded.table },
      ints: { ELEMENTS: elements },
      label: `check.int5:${name}`,
    });
    const got = execution.read(output);
    const want = readTensor(source.record, source.buffer, source.byteOffset, true);
    let differing = 0;
    let worst = 0;
    let worstUlps = 0;
    for (let i = 0; i < elements; i += 1) {
      const a = got[i];
      const b = want[i];
      if (a === b) continue;
      differing += 1;
      const gap = Math.abs(a - b);
      if (gap > worst) worst = gap;
      // An ulp at this magnitude: float32 has 24 bits of mantissa.
      const ulp = Math.max(Math.abs(b), Number.MIN_VALUE) * 2 ** -23;
      worstUlps = Math.max(worstUlps, gap / ulp);
    }
    execution.release(output);
    uploaded.free();
    rows.push({
      tensor: name,
      elements,
      differing,
      fractionDiffering: Number((differing / elements).toExponential(2)),
      worstAbsolute: Number(worst.toExponential(2)),
      worstUlps: Number(worstUlps.toFixed(2)),
      ok: worstUlps <= 1,
    });
  }

  const checked = rows.filter((row) => row.skipped === undefined);
  const bad = checked.filter((row) => !row.ok);
  const worstUlps = Math.max(0, ...checked.map((row) => row.worstUlps));
  const totalElements = checked.reduce((sum, row) => sum + row.elements, 0);
  const totalDiffering = checked.reduce((sum, row) => sum + row.differing, 0);
  execution.destroy();
  return {
    bundle,
    falsify,
    tensorsChecked: checked.length,
    elementsChecked: totalElements,
    elementsDiffering: totalDiffering,
    worstUlps,
    rows,
    verdict: checked.length === 0
      ? "FAILED: no int5 tensor was checked, so this gate measured nothing"
      : bad.length > 0
        ? `FAILED: ${bad.length} of ${checked.length} tensors decode differently - worst `
          + `${worstUlps.toFixed(2)} ulps over ${totalElements} elements, which is not a `
          + "fused multiply-add"
        : `ok: ${checked.length} int5 tensors, ${totalElements} elements, `
          + `${totalDiffering} differing from readTensor at all and the worst `
          + `${worstUlps.toFixed(2)} ulps`,
  };
}

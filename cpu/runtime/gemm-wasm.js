/**
 * A matrix multiply in WebAssembly SIMD, assembled here rather than compiled.
 *
 *     out[row, o] = sum_i input[row, i] * weights[i, o]
 *
 * Nearly all of a fold's arithmetic is this one loop: at 100 residues the trunk's
 * projections are about 97% of its multiply-adds and the triangles, which are the
 * same loop over a plane, are the rest. Plain JavaScript runs it at about 1 GFLOP/s
 * a core because V8 does not vectorise; four lanes and four input values per pass
 * give it the order of magnitude that decides whether a CPU fold is usable.
 *
 * Built from this file at load, so there is no compiler and no blob: the bytes
 * below are the source, which keeps the page's rule of no build step between the
 * source and the browser. The reference kernels in cpu/ stay the specification;
 * test/cpu-backend.test.js holds this to them.
 */

const OP = Object.freeze({
  block: 0x02, loop: 0x03, br: 0x0c, br_if: 0x0d, end: 0x0b, void: 0x40,
  local_get: 0x20, local_set: 0x21, local_tee: 0x22,
  f32_load: 0x2a, f32_store: 0x38,
  i32_const: 0x41,
  i32_ge_u: 0x4f, i32_gt_u: 0x4b,
  i32_add: 0x6a, i32_sub: 0x6b, i32_mul: 0x6c, i32_and: 0x71, i32_shl: 0x74,
  f32_add: 0x92, f32_mul: 0x94,
  simd: 0xfd, misc: 0xfc,
});

/** The SIMD opcodes this uses, after the 0xfd prefix, as their LEB128 bytes. */
const SIMD = Object.freeze({
  v128_load: [0x00], load32_splat: [0x09], v128_store: [0x0b],
  f32x4_add: [0xe4, 0x01], f32x4_mul: [0xe6, 0x01],
  // a * b + c, fused where the CPU can. "Relaxed" means a CPU may round it either way,
  // which is inside what the differential test allows and no less than the GPU does.
  relaxed_madd: [0x85, 0x02],
});

function unsigned(value) {
  const bytes = [];
  let rest = value >>> 0;
  do {
    const byte = rest & 0x7f;
    rest >>>= 7;
    bytes.push(rest === 0 ? byte : byte | 0x80);
  } while (rest !== 0);
  return bytes;
}

function signed(value) {
  const bytes = [];
  let rest = value | 0;
  for (;;) {
    const byte = rest & 0x7f;
    rest >>= 7;
    const done = (rest === 0 && (byte & 0x40) === 0) || (rest === -1 && (byte & 0x40) !== 0);
    bytes.push(done ? byte : byte | 0x80);
    if (done) return bytes;
  }
}

const section = (id, body) => [id, ...unsigned(body.length), ...body];
const vector = (items) => [...unsigned(items.length), ...items.flat()];
const name = (text) => [...unsigned(text.length), ...new TextEncoder().encode(text)];

/** The function's locals: six i32 parameters, then what the body declares. */
const L = Object.freeze({
  out: 0, input: 1, weights: 2, rows: 3, inner: 4, outer: 5,
  row: 6, i: 7, o: 8, outerBytes: 9, vectorBytes: 10,
  w0: 11, w1: 12, w2: 13, w3: 14,
  v0: 15, v1: 16, v2: 17, v3: 18,
  s0: 19, s1: 20, s2: 21, s3: 22,
});
const LOCAL_DECLARATIONS = [
  [...unsigned(9), 0x7f], // row, i, o, outerBytes, vectorBytes, w0..w3 : i32
  [...unsigned(4), 0x7b], // v0..v3 : v128
  [...unsigned(4), 0x7d], // s0..s3 : f32
];

const get = (local) => [OP.local_get, ...unsigned(local)];
const set = (local) => [OP.local_set, ...unsigned(local)];
const memarg = (offset = 0) => [0x00, ...unsigned(offset)];
const i32 = (value) => [OP.i32_const, ...signed(value)];

/** `while (!stop) { body }`, with `stop` an expression leaving an i32. */
const whileNot = (stop, body) => [
  OP.block, OP.void, OP.loop, OP.void,
  ...stop, OP.br_if, 1,
  ...body,
  OP.br, 0, OP.end, OP.end,
];

/** v128 at `base + o`. */
const loadVector = (base) => [...get(base), ...get(L.o), OP.i32_add, OP.simd, ...SIMD.v128_load, ...memarg()];
const loadScalar = (base) => [...get(base), ...get(L.o), OP.i32_add, OP.f32_load, ...memarg()];

/** `out[o..] += v * w[o..]` for each of `lanes` (v, w) pairs, then the scalar tail. */
function accumulate(lanes, relaxed) {
  const vectors = [L.v0, L.v1, L.v2, L.v3].slice(0, lanes);
  const scalars = [L.s0, L.s1, L.s2, L.s3].slice(0, lanes);
  const weights = [L.w0, L.w1, L.w2, L.w3].slice(0, lanes);
  let vectorSum;
  if (relaxed) {
    // madd(v3, w3, madd(v2, w2, ... load)), which on a stack is every pair and then the sum.
    vectorSum = [...get(L.out), ...get(L.o), OP.i32_add];
    for (let at = lanes - 1; at >= 0; at -= 1) vectorSum.push(...get(vectors[at]), ...loadVector(weights[at]));
    vectorSum.push(...loadVector(L.out));
    for (let at = 0; at < lanes; at += 1) vectorSum.push(OP.simd, ...SIMD.relaxed_madd);
  } else {
    vectorSum = [...get(L.out), ...get(L.o), OP.i32_add, ...loadVector(L.out)];
    vectors.forEach((v, at) => {
      vectorSum.push(...get(v), ...loadVector(weights[at]), OP.simd, ...SIMD.f32x4_mul,
                     OP.simd, ...SIMD.f32x4_add);
    });
  }
  const scalarSum = [...get(L.out), ...get(L.o), OP.i32_add, ...loadScalar(L.out)];
  vectors.forEach((v, at) => {
    scalarSum.push(...get(scalars[at]), ...loadScalar(weights[at]), OP.f32_mul, OP.f32_add);
  });
  return [
    ...i32(0), ...set(L.o),
    ...whileNot([...get(L.o), ...get(L.vectorBytes), OP.i32_ge_u], [
      ...vectorSum, OP.simd, ...SIMD.v128_store, ...memarg(),
      ...get(L.o), ...i32(16), OP.i32_add, ...set(L.o),
    ]),
    ...whileNot([...get(L.o), ...get(L.outerBytes), OP.i32_ge_u], [
      ...scalarSum, OP.f32_store, ...memarg(),
      ...get(L.o), ...i32(4), OP.i32_add, ...set(L.o),
    ]),
  ];
}

/** Load `lanes` consecutive input values, as splats and as scalars. */
function loadInputs(lanes) {
  const code = [];
  for (let lane = 0; lane < lanes; lane += 1) {
    const at = lane * 4;
    code.push(...get(L.input), OP.simd, ...SIMD.load32_splat, ...memarg(at), ...set(L.v0 + lane));
    code.push(...get(L.input), OP.f32_load, ...memarg(at), ...set(L.s0 + lane));
  }
  return code;
}

function gemmBody(relaxed) {
  const fourLanes = [
    // while (i + 4 <= inner)
    ...whileNot([...get(L.i), ...i32(4), OP.i32_add, ...get(L.inner), OP.i32_gt_u], [
      ...loadInputs(4),
      ...get(L.w0), ...get(L.outerBytes), OP.i32_add, OP.local_tee, ...unsigned(L.w1),
      ...get(L.outerBytes), OP.i32_add, OP.local_tee, ...unsigned(L.w2),
      ...get(L.outerBytes), OP.i32_add, ...set(L.w3),
      ...accumulate(4, relaxed),
      ...get(L.w3), ...get(L.outerBytes), OP.i32_add, ...set(L.w0),
      ...get(L.input), ...i32(16), OP.i32_add, ...set(L.input),
      ...get(L.i), ...i32(4), OP.i32_add, ...set(L.i),
    ]),
  ];
  const oneLane = [
    ...whileNot([...get(L.i), ...get(L.inner), OP.i32_ge_u], [
      ...loadInputs(1),
      ...accumulate(1, relaxed),
      ...get(L.w0), ...get(L.outerBytes), OP.i32_add, ...set(L.w0),
      ...get(L.input), ...i32(4), OP.i32_add, ...set(L.input),
      ...get(L.i), ...i32(1), OP.i32_add, ...set(L.i),
    ]),
  ];
  return [
    ...get(L.outer), ...i32(2), OP.i32_shl, ...set(L.outerBytes),
    ...get(L.outerBytes), ...i32(-16), OP.i32_and, ...set(L.vectorBytes),
    ...whileNot([...get(L.row), ...get(L.rows), OP.i32_ge_u], [
      // memory.fill(out, 0, outerBytes)
      ...get(L.out), ...i32(0), ...get(L.outerBytes), OP.misc, 0x0b, 0x00,
      ...i32(0), ...set(L.i),
      ...get(L.weights), ...set(L.w0),
      ...fourLanes,
      ...oneLane,
      ...get(L.out), ...get(L.outerBytes), OP.i32_add, ...set(L.out),
      ...get(L.row), ...i32(1), OP.i32_add, ...set(L.row),
    ]),
    OP.end,
  ];
}

const SIMD_MATH = Object.freeze({
  v128_const: [0x0c], splat: [0x13], floor: [0x68],
  add: [0xe4, 0x01], sub: [0xe5, 0x01], mul: [0xe6, 0x01], div: [0xe7, 0x01],
  min: [0xe8, 0x01], max: [0xe9, 0x01],
  trunc_sat: [0xf8, 0x01], i32x4_add: [0xae, 0x01], i32x4_shl: [0xab, 0x01],
});

/** Four copies of a float32, as a v128.const. */
function constant(value) {
  const bytes = new Uint8Array(new Float32Array([value, value, value, value]).buffer);
  return [OP.simd, ...SIMD_MATH.v128_const, ...bytes];
}
const m = (name) => [OP.simd, ...SIMD_MATH[name]];

/**
 * out[i] = 1 / (1 + exp(-in[i])), four at a time, over `count & ~3` elements.
 *
 * Every gate in the trunk is this: at 76 residues about 1.4 billion of them a fold, at
 * 20-30 ns each through Math.exp, which was more of the trunk than its multiplies.
 * Cephes' single-precision exp, good to about 1 ulp, with the exponent clamped to
 * where a float32 logistic has saturated anyway. Params: out, in, count.
 */
function logisticBody() {
  const P = { out: 0, input: 1, count: 2, at: 3, bytes: 4, x: 5, whole: 6, y: 7 };
  return [
    ...get(P.count), ...i32(-4), OP.i32_and, ...i32(2), OP.i32_shl, ...set(P.bytes),
    ...whileNot([...get(P.at), ...get(P.bytes), OP.i32_ge_u], [
      // x = clamp(-input)
      ...constant(0), ...get(P.input), ...get(P.at), OP.i32_add, OP.simd, ...SIMD.v128_load, ...memarg(),
      ...m("sub"), ...constant(88), ...m("min"), ...constant(-88), ...m("max"), ...set(P.x),
      // whole = floor(x * log2(e) + 0.5)
      ...get(P.x), ...constant(1.44269504), ...m("mul"), ...constant(0.5), ...m("add"), ...m("floor"),
      ...set(P.whole),
      // x -= whole * ln2, in two parts
      ...get(P.x),
      ...get(P.whole), ...constant(0.693359375), ...m("mul"), ...m("sub"),
      ...get(P.whole), ...constant(-2.12194440e-4), ...m("mul"), ...m("sub"), ...set(P.x),
      // y = polynomial(x)
      ...constant(1.9875691500e-4), ...get(P.x), ...m("mul"), ...constant(1.3981999507e-3), ...m("add"),
      ...get(P.x), ...m("mul"), ...constant(8.3334519073e-3), ...m("add"),
      ...get(P.x), ...m("mul"), ...constant(4.1665795894e-2), ...m("add"),
      ...get(P.x), ...m("mul"), ...constant(1.6666665459e-1), ...m("add"),
      ...get(P.x), ...m("mul"), ...constant(5.0000001201e-1), ...m("add"),
      ...get(P.x), ...get(P.x), ...m("mul"), ...m("mul"), ...get(P.x), ...m("add"), ...constant(1), ...m("add"),
      ...set(P.y),
      // out = 1 / (1 + y * 2^whole); the exponent goes straight into the float's bits.
      ...get(P.out), ...get(P.at), OP.i32_add,
      ...constant(1), ...constant(1),
      ...get(P.y),
      ...get(P.whole), ...m("trunc_sat"),
      OP.simd, ...SIMD_MATH.v128_const, ...new Uint8Array(new Int32Array([127, 127, 127, 127]).buffer),
      ...m("i32x4_add"), ...i32(23), ...m("i32x4_shl"),
      ...m("mul"), ...m("add"), ...m("div"),
      OP.simd, ...SIMD.v128_store, ...memarg(),
      ...get(P.at), ...i32(16), OP.i32_add, ...set(P.at),
    ]),
    OP.end,
  ];
}

/** The module's bytes, importing its memory from `env.memory`. */
export function gemmModuleBytes({ shared, pages, relaxed = false }) {
  const i32Type = 0x7f;
  const gemm = [...vector(LOCAL_DECLARATIONS), ...gemmBody(relaxed)];
  // logistic's locals: at, bytes : i32; x, whole, y : v128
  const logistic = [...vector([[...unsigned(2), i32Type], [...unsigned(3), 0x7b]]), ...logisticBody()];
  const memoryLimits = [shared ? 0x03 : 0x01, ...unsigned(pages), ...unsigned(pages)];
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, vector([
      [0x60, ...vector(Array(6).fill([i32Type])), 0x00],
      [0x60, ...vector(Array(3).fill([i32Type])), 0x00],
    ])),
    ...section(2, vector([[...name("env"), ...name("memory"), 0x02, ...memoryLimits]])),
    ...section(3, vector([[0x00], [0x01]])),
    ...section(7, vector([[...name("gemm"), 0x00, 0x00], [...name("logistic"), 0x00, 0x01]])),
    ...section(10, vector([[...unsigned(gemm.length), ...gemm],
                           [...unsigned(logistic.length), ...logistic]])),
  ]);
}

/** Whether this engine has fused multiply-add, which is about half again on a desktop CPU. */
export function relaxedSimdSupported() {
  try {
    return WebAssembly.validate(gemmModuleBytes({ shared: false, pages: 1, relaxed: true }));
  } catch {
    return false;
  }
}

/** Whether this runtime can run the module at all: SIMD and bulk memory. */
export function gemmSupported() {
  try {
    return WebAssembly.validate(gemmModuleBytes({ shared: false, pages: 1 }));
  } catch {
    return false;
  }
}

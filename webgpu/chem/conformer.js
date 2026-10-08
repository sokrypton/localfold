/**
 * The conformer refinement, on the device: one dispatch for the whole solve.
 *
 * 🔴 THE OBVIOUS PORT OF THIS IS SLOWER THAN THE CPU AND THE MEASUREMENT IS
 * WHAT SAYS SO. The refinement is 400 steepest-descent steps, each with a
 * backtracking line search of up to 20 trial evaluations, and each evaluation
 * is a reduction over every pair. Written the natural way - one dispatch per
 * evaluation - that is up to 8,000 dispatches for one ligand, and at this
 * device's ~0.1 ms of launch overhead the overhead ALONE is 800 ms against the
 * whole CPU refinement's 27.6 ms at 100 atoms. The arithmetic is not the
 * problem; the problem is that the algorithm is iteration-serial and the
 * iterations are tiny.
 *
 * So the whole solve lives inside ONE dispatch. A workgroup holds one
 * conformer's coordinates in workgroup memory and runs every step, every line
 * search and every reduction against `workgroupBarrier`, and the coordinates
 * never leave the workgroup until it is done. The parallelism that is left is
 * across ATTEMPTS and across LIGANDS - distance geometry needs several random
 * starts anyway, and a screen has many molecules - so the batch dimension is
 * workgroups, and one dispatch does all of them at once.
 *
 * 🔴 AND THIS IS A SEAM, NOT A BRANCH. `smilesComponent` takes an `embed`
 * option and the CPU path is the default; nothing in the chemistry knows
 * whether a device exists. That is also what makes the differential gate
 * possible - `tools/gpu/check-smiles-conformer-gpu.js` runs both over the same
 * bounds and the same seed and holds them to each other.
 *
 * 🔴 AND THE ANSWER IS NOT BIT-IDENTICAL TO THE CPU's, BY CONSTRUCTION. The
 * host reduces a pair sum in index order in f64 and the device reduces it as a
 * tree in f32, so the two descend slightly different paths from the same start
 * and land in slightly different places in the same basin. The gate therefore
 * compares what a conformer IS - its bond lengths, its angles, its chirality -
 * and not its coordinates, which is the same rule
 * `tools/check-smiles-conformer.mjs` applies to RDKit and for the same reason.
 */

/** Lanes per workgroup; one workgroup is one conformer. */
const LANES = 64;

/**
 * 🔴 THE ATOM CAP IS A WORKGROUP-STORAGE CAP AND IT IS DECLARED, NOT DISCOVERED.
 * Coordinates, the gradient and a trial position are all held in workgroup
 * memory at 3 floats each, plus the reduction scratch: at 256 atoms that is
 * 256 * 3 * 4 * 3 + 64 * 4 = 9472 bytes, inside the 16384 the WebGPU spec
 * guarantees. A device at the portable floor must still run this, which is
 * what `npm run test:spec-floor` exists to check - see CLAUDE.md, where two of
 * four models once could not create a pipeline on a conforming minimum device.
 */
export const MAX_ATOMS = 256;

const SHADER = (atoms, steps, planarCount, chiralCount, projections, linearCount) => `
const N: u32 = ${atoms}u;
const LANES: u32 = ${LANES}u;
const STEPS: u32 = ${steps}u;
const PLANAR: u32 = ${planarCount}u;
const CHIRAL: u32 = ${chiralCount}u;
const PLANARITY_WEIGHT: f32 = 0.5;
const PROJECTIONS: u32 = ${projections}u;
const LINEAR: u32 = ${linearCount}u;
const LINEARITY_WEIGHT: f32 = 0.5;

@group(0) @binding(0) var<storage, read> lower: array<f32>;
@group(0) @binding(1) var<storage, read> upper: array<f32>;
@group(0) @binding(2) var<storage, read> starts: array<f32>;
// Each planar group is four atom indices; each chiral centre is four and a sign.
@group(0) @binding(3) var<storage, read> planar: array<u32>;
@group(0) @binding(4) var<storage, read> chiral: array<i32>;
// Triples that must be collinear: three atom indices each. See linearTriples.
@group(0) @binding(7) var<storage, read> linear: array<u32>;
@group(0) @binding(5) var<storage, read_write> result: array<f32>;
@group(0) @binding(6) var<storage, read_write> errors: array<f32>;

var<workgroup> point: array<vec3<f32>, ${atoms}u>;
var<workgroup> trial: array<vec3<f32>, ${atoms}u>;
var<workgroup> grad: array<vec3<f32>, ${atoms}u>;
var<workgroup> partial: array<f32, ${LANES}u>;
// (no scalar scratch needed: the reductions land in partial[0])

// 🔴 THE SIGNED VOLUME IS SIX TIMES THE TETRAHEDRON'S AND THE SIX IS NOT
// DIVIDED OUT, on either side. The chiral target and the planarity weight are
// both expressed in the same units, so dividing here would silently rescale
// two terms the CPU path does not rescale.
fn volume(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
  let u = a - d; let v = b - d; let w = c - d;
  return dot(u, cross(v, w));
}

/** The error of whichever buffer useTrial names, reduced across the group. */
fn totalError(local: u32, useTrial: bool) -> f32 {
  var sum: f32 = 0.0;
  // Every pair, striped across the lanes. The upper triangle is walked as a
  // flat index so the stripe is even - walking i and nesting j gives lane
  // 0 almost all the work.
  let pairs = N * (N - 1u) / 2u;
  var at = local;
  loop {
    if (at >= pairs) { break; }
    // Recover (i, j) from the flat upper-triangular index.
    let i = triangleRow(at);
    let j = at - i * (i - 1u) / 2u;
    let pi = select(point[i], trial[i], useTrial);
    let pj = select(point[j], trial[j], useTrial);
    let delta = pi - pj;
    let squared = dot(delta, delta);
    let high = upper[i * N + j];
    let low = lower[i * N + j];
    if (squared > high * high) {
      let over = squared / (high * high) - 1.0;
      sum = sum + over * over;
    } else if (squared < low * low) {
      let ratio = (2.0 * low * low) / (low * low + squared);
      let over = ratio - 1.0;
      sum = sum + over * over;
    }
    at = at + LANES;
  }
  var g = local;
  loop {
    if (g >= PLANAR) { break; }
    let base = g * 4u;
    let v = volume(select(point[planar[base]], trial[planar[base]], useTrial),
                   select(point[planar[base + 1u]], trial[planar[base + 1u]], useTrial),
                   select(point[planar[base + 2u]], trial[planar[base + 2u]], useTrial),
                   select(point[planar[base + 3u]], trial[planar[base + 3u]], useTrial));
    sum = sum + v * v * PLANARITY_WEIGHT;
    g = g + LANES;
  }
  // 🔴 THE CHIRAL TERM, WHICH IS WHAT STOPS THE DEVICE RETURNING A MIRROR
  // IMAGE. Every pairwise distance is identical in both enantiomers, so
  // without this the GPU path would disagree with the CPU one half the time
  // and the disagreement would look like a precision problem.
  var t = local;
  loop {
    if (t >= LINEAR) { break; }
    let base = t * 3u;
    let pa = select(point[linear[base]], trial[linear[base]], useTrial);
    let pc = select(point[linear[base + 1u]], trial[linear[base + 1u]], useTrial);
    let pb = select(point[linear[base + 2u]], trial[linear[base + 2u]], useTrial);
    let w = cross(pa - pc, pb - pc);
    sum = sum + dot(w, w) * LINEARITY_WEIGHT;
    t = t + LANES;
  }
  var c = local;
  loop {
    if (c >= CHIRAL) { break; }
    let base = c * 5u;
    let v = volume(select(point[u32(chiral[base])], trial[u32(chiral[base])], useTrial),
                   select(point[u32(chiral[base + 1u])], trial[u32(chiral[base + 1u])], useTrial),
                   select(point[u32(chiral[base + 2u])], trial[u32(chiral[base + 2u])], useTrial),
                   select(point[u32(chiral[base + 3u])], trial[u32(chiral[base + 3u])], useTrial));
    let wanted = f32(chiral[base + 4u]);
    let short = 0.4 - v * wanted;
    if (short > 0.0) { sum = sum + short * short * 4.0; }
    c = c + LANES;
  }
  partial[local] = sum;
  workgroupBarrier();
  // A tree reduction, which is where the f32 sum order differs from the host's.
  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (local < stride) { partial[local] = partial[local] + partial[local + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  return partial[0];
}

fn triangleRow(flat: u32) -> u32 {
  // The largest i with i*(i-1)/2 <= flat, by search - N is small and this is
  // cheaper than the floating-point inverse, which is off by one near a
  // boundary and silently pairs an atom with itself.
  var i = 1u;
  loop {
    if (i >= N) { break; }
    if ((i + 1u) * i / 2u > flat) { break; }
    i = i + 1u;
  }
  return i;
}

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) lane: vec3<u32>) {
  let local = lane.x;
  let which = group.x;
  var i = local;
  loop {
    if (i >= N) { break; }
    let base = which * N * 3u + i * 3u;
    point[i] = vec3<f32>(starts[base], starts[base + 1u], starts[base + 2u]);
    i = i + LANES;
  }
  workgroupBarrier();

  // 🔴 THE PROJECTION FIRST, THE SAME AS THE HOST - AND JACOBI WHERE THE HOST
  // IS GAUSS-SEIDEL. The host repairs one violated pair at a time and lets the
  // next pair see the result; that is inherently serial. Here every lane owns
  // a row of atoms, sums the corrections that row's pairs want, and applies
  // the AVERAGE - so no two lanes write the same atom and no atomic is needed,
  // which WGSL has no float version of anyway. Averaging converges more slowly
  // per sweep than the host's in-place walk, so it gets more sweeps; both end
  // at a satisfied set of bounds, which is what the differential compares.
  for (var sweep: u32 = 0u; sweep < PROJECTIONS; sweep = sweep + 1u) {
    projectSweep(local);
  }
  var step: f32 = 0.05;
  var error = totalError(local, false);
  var restarts: u32 = 0u;
  var finished = false;

  // 🔴 EVERY BARRIER HERE IS UNCONDITIONAL, AND THAT IS WHY THE LOOPS LOOK
  // ODD. WGSL requires workgroupBarrier() to be reached by every lane
  // together, and its uniformity analysis does NOT consider a value read back
  // out of workgroup memory to be uniform - even when it provably is, as a
  // reduction result is. So none of the natural shapes compile: not
  // "if (next < error) { accept; barrier; }", not "if (converged) { break; }"
  // out of a loop whose body barriers. Dawn refuses the module outright with
  // "must only be called from uniform control flow", which is the same rule
  // test/uniform-barrier.test.js gates on the AF2 kernels.
  //
  // So the step count is FIXED, a finished flag makes later iterations
  // no-ops, and every acceptance is a select rather than a branch. The cost
  // is running the full budget on a molecule that converged early; the benefit
  // is a kernel that exists.
  for (var iteration: u32 = 0u; iteration < STEPS; iteration = iteration + 1u) {
    gradientInto(local);
    var mine: f32 = 0.0;
    var g = local;
    loop {
      if (g >= N) { break; }
      mine = mine + dot(grad[g], grad[g]);
      g = g + LANES;
    }
    partial[local] = mine;
    workgroupBarrier();
    var stride = LANES / 2u;
    loop {
      if (stride == 0u) { break; }
      if (local < stride) { partial[local] = partial[local] + partial[local + stride]; }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let size = sqrt(max(partial[0], 1e-30));
    workgroupBarrier();

    var improved = false;
    for (var attempt: u32 = 0u; attempt < 20u; attempt = attempt + 1u) {
      let live = !finished && !improved;
      var t = local;
      loop {
        if (t >= N) { break; }
        trial[t] = point[t] - (step / size) * grad[t];
        t = t + LANES;
      }
      workgroupBarrier();
      let next = totalError(local, true);
      workgroupBarrier();
      let take = live && (next < error);
      var c = local;
      loop {
        if (c >= N) { break; }
        point[c] = select(point[c], trial[c], take);
        c = c + LANES;
      }
      workgroupBarrier();
      error = select(error, next, take);
      step = select(select(step, step * 0.4, live), step * 1.3, take);
      improved = improved || take;
    }

    let stalled = !improved && !finished;
    restarts = select(restarts, restarts + 1u, stalled);
    step = select(step, 0.05, stalled);
    finished = finished || (stalled && restarts > 4u) || (error < 1e-8);
  }

  var w = local;
  loop {
    if (w >= N) { break; }
    let base = which * N * 3u + w * 3u;
    result[base] = point[w].x;
    result[base + 1u] = point[w].y;
    result[base + 2u] = point[w].z;
    w = w + LANES;
  }
  if (local == 0u) { errors[which] = error; }
}

fn projectSweep(local: u32) {
  var i = local;
  loop {
    if (i >= N) { break; }
    var shift = vec3<f32>(0.0, 0.0, 0.0);
    var hits: f32 = 0.0;
    for (var j: u32 = 0u; j < N; j = j + 1u) {
      if (j == i) { continue; }
      let a = min(i, j); let b = max(i, j);
      let delta = point[i] - point[j];
      let distance = length(delta);
      if (distance < 1e-9) { continue; }
      let low = lower[a * N + b];
      let high = upper[a * N + b];
      var want = distance;
      if (distance < low) { want = low; } else if (distance > high) { want = high; }
      else { continue; }
      shift = shift + (0.5 * (want - distance) / distance) * delta;
      hits = hits + 1.0;
    }
    trial[i] = select(point[i], point[i] + shift / max(hits, 1.0), hits > 0.0);
    i = i + LANES;
  }
  // 🔴 WRITTEN TO trial AND COPIED BACK, NOT IN PLACE. A lane reads every
  // other atom's position while computing its own correction, so updating
  // point[i] inside the loop would let a lane see a neighbour that has already
  // moved this sweep and a neighbour that has not, depending only on which
  // lane got there first - the answer would differ run to run.
  workgroupBarrier();
  var c = local;
  loop {
    if (c >= N) { break; }
    point[c] = trial[c];
    c = c + LANES;
  }
  workgroupBarrier();
}

fn gradientInto(local: u32) {
  var z = local;
  loop {
    if (z >= N) { break; }
    grad[z] = vec3<f32>(0.0, 0.0, 0.0);
    z = z + LANES;
  }
  workgroupBarrier();
  // 🔴 ONE LANE PER ATOM ROW, NOT ONE PER PAIR. A pair contributes to BOTH its
  // atoms, so striping over pairs needs an atomic add per component or the
  // writes race - and WGSL has no atomic float. Walking whole rows means every
  // write is to an atom this lane owns, which is the same answer with no
  // atomics and no barrier inside the loop.
  var i = local;
  loop {
    if (i >= N) { break; }
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j: u32 = 0u; j < N; j = j + 1u) {
      if (j == i) { continue; }
      let a = min(i, j); let b = max(i, j);
      let delta = point[i] - point[j];
      let squared = dot(delta, delta);
      let high = upper[a * N + b];
      let low = lower[a * N + b];
      if (squared > high * high) {
        let over = squared / (high * high) - 1.0;
        sum = sum + (4.0 * over / (high * high)) * delta;
      } else if (squared < low * low) {
        let ratio = (2.0 * low * low) / (low * low + squared);
        let over = ratio - 1.0;
        sum = sum + (-2.0 * over * ratio * ratio / (low * low)) * delta;
      }
    }
    grad[i] = sum;
    i = i + LANES;
  }
  workgroupBarrier();
  // 🔴 THE VOLUME TERMS ARE ADDED BY ONE LANE, BECAUSE THEY TOUCH FOUR ATOMS
  // THAT NO LANE OWNS. Striping them the way the pair loop is striped would
  // have two lanes writing the same atom's gradient, which WGSL cannot do for
  // floats without an atomic. There are a handful of groups against N^2 pairs,
  // so serialising them costs nothing measurable.
  if (local == 0u) {
    for (var g: u32 = 0u; g < PLANAR; g = g + 1u) {
      let base = g * 4u;
      let a = planar[base]; let b = planar[base + 1u];
      let c = planar[base + 2u]; let d = planar[base + 3u];
      let v = volume(point[a], point[b], point[c], point[d]);
      addVolumeGradient(a, b, c, d, 2.0 * v * PLANARITY_WEIGHT);
    }
    for (var m: u32 = 0u; m < LINEAR; m = m + 1u) {
      let base = m * 3u;
      let a = linear[base]; let centre = linear[base + 1u]; let b = linear[base + 2u];
      let u = point[a] - point[centre];
      let v = point[b] - point[centre];
      let w = cross(u, v);
      let du = 2.0 * LINEARITY_WEIGHT * cross(v, w);
      let dv = 2.0 * LINEARITY_WEIGHT * cross(w, u);
      grad[a] = grad[a] + du;
      grad[b] = grad[b] + dv;
      grad[centre] = grad[centre] - du - dv;
    }
    for (var k: u32 = 0u; k < CHIRAL; k = k + 1u) {
      let base = k * 5u;
      let a = u32(chiral[base]); let b = u32(chiral[base + 1u]);
      let c = u32(chiral[base + 2u]); let d = u32(chiral[base + 3u]);
      let wanted = f32(chiral[base + 4u]);
      let v = volume(point[a], point[b], point[c], point[d]);
      let short = 0.4 - v * wanted;
      if (short > 0.0) { addVolumeGradient(a, b, c, d, -8.0 * short * wanted); }
    }
  }
  workgroupBarrier();
}

fn addVolumeGradient(a: u32, b: u32, c: u32, d: u32, push: f32) {
  let u = point[a] - point[d];
  let v = point[b] - point[d];
  let w = point[c] - point[d];
  let da = cross(v, w);
  let db = cross(w, u);
  let dc = cross(u, v);
  grad[a] = grad[a] + push * da;
  grad[b] = grad[b] + push * db;
  grad[c] = grad[c] + push * dc;
  grad[d] = grad[d] - push * (da + db + dc);
}
`;

/**
 * Refine several starting points at once, one workgroup each.
 *
 * @param {GPUDevice} device
 * @param {{lower: Float64Array, upper: Float64Array, n: number}} bounds
 * @param {Float64Array[]} startingPoints
 * @param {{planar?: number[][], chiral?: object[], steps?: number}} [options]
 * @returns {Promise<{coordinates: Float64Array, error: number}[]>}
 */
export async function refineOnDevice(device, bounds, startingPoints, options = {}) {
  const { lower, upper, n } = bounds;
  if (n > MAX_ATOMS) {
    throw new Error(`${n} atoms is past this kernel's ${MAX_ATOMS}; `
      + "the coordinates live in workgroup storage and a device at the "
      + "portable floor guarantees 16 KiB");
  }
  const planar = options.planar ?? [];
  const steps = options.steps ?? 400;
  const batch = startingPoints.length;

  const asF32 = (source) => {
    const out = new Float32Array(source.length);
    for (let index = 0; index < source.length; index += 1) out[index] = source[index];
    return out;
  };
  const starts = new Float32Array(batch * n * 3);
  startingPoints.forEach((point, index) => starts.set(asF32(point), index * n * 3));

  const buffers = {
    lower: upload(device, asF32(lower), GPUBufferUsage.STORAGE),
    upper: upload(device, asF32(upper), GPUBufferUsage.STORAGE),
    starts: upload(device, starts, GPUBufferUsage.STORAGE),
    planar: upload(device, Uint32Array.from(planar.flat()), GPUBufferUsage.STORAGE),
    chiral: upload(device, Int32Array.from(
      (options.chiral ?? []).flatMap((centre) => [...centre.neighbours, centre.sign])),
      GPUBufferUsage.STORAGE),
    result: device.createBuffer({
      size: Math.max(16, batch * n * 3 * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    linear: upload(device, Uint32Array.from((options.linear ?? []).flat()),
                   GPUBufferUsage.STORAGE),
    errors: device.createBuffer({
      size: Math.max(16, batch * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
  };

  const module = device.createShaderModule({
    code: SHADER(n, steps, planar.length, (options.chiral ?? []).length,
      options.projections ?? 1000, (options.linear ?? []).length),
    label: "chem.conformer-refine",
  });
  const pipeline = device.createComputePipeline({
    layout: "auto", compute: { module, entryPoint: "main" },
  });
  // 🔴 THE BINDING NUMBERS ARE WRITTEN OUT, NOT TAKEN FROM KEY ORDER. The
  // first version mapped `Object.values(buffers)` to 0, 1, 2... which is
  // correct exactly until somebody inserts a buffer in the middle - adding
  // `linear` before `errors` silently moved `errors` from 6 to 7 and handed
  // the shader its results buffer where it expected its error buffer. Nothing
  // would have thrown; the numbers would simply have been wrong. This is
  // CLAUDE.md's `createAddShader` note, where two copies of one kernel bind
  // the accumulator and the delta in opposite orders and a pipeline from one
  // with a bind group from the other writes into the wrong one.
  const BINDINGS = {
    lower: 0, upper: 1, starts: 2, planar: 3, chiral: 4,
    result: 5, errors: 6, linear: 7,
  };
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: Object.entries(buffers).map(([name, buffer]) => {
      const binding = BINDINGS[name];
      if (binding === undefined) throw new Error(`no binding declared for ${name}`);
      return { binding, resource: { buffer } };
    }),
  });

  const encoder = device.createCommandEncoder({ label: "chem.conformer" });
  const pass = encoder.beginComputePass({ label: "chem.conformer-refine" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(batch);
  pass.end();

  const readResult = device.createBuffer({
    size: buffers.result.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readErrors = device.createBuffer({
    size: buffers.errors.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(buffers.result, 0, readResult, 0, buffers.result.size);
  encoder.copyBufferToBuffer(buffers.errors, 0, readErrors, 0, buffers.errors.size);
  device.queue.submit([encoder.finish()]);

  await readResult.mapAsync(GPUMapMode.READ);
  await readErrors.mapAsync(GPUMapMode.READ);
  const coordinates = new Float32Array(readResult.getMappedRange().slice(0));
  const errors = new Float32Array(readErrors.getMappedRange().slice(0));
  readResult.unmap();
  readErrors.unmap();
  for (const buffer of Object.values(buffers)) buffer.destroy();
  readResult.destroy();
  readErrors.destroy();

  return startingPoints.map((_, index) => ({
    coordinates: Float64Array.from(
      coordinates.subarray(index * n * 3, (index + 1) * n * 3)),
    error: errors[index],
  }));
}

function upload(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/*
 * ATOM GEOMETRY FOR THE MULTIMER GRAPH. See src/multimer/input-embedder.js for
 * why these files are a copy rather than an edit in place.
 *
 * WHAT DIFFERS HERE: position_scale. The structure module works in units the
 * frames are scaled by on the way out - 10 angstroms for monomer, 20 for
 * multimer - and the weights were trained against their own value, so it is a
 * per-model fact rather than a constant. It was written into the shader as a
 * bare 10.0; here the shader is generated per scale, which keeps the binding
 * list alone. Two pipelines, cached separately.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

/** AF2's position_scale: 10 angstroms for the monomer models, 20 for multimer. */
export const MONOMER_POSITION_SCALE = 10;
export const MULTIMER_POSITION_SCALE = 20;

const createAtom14Shader = (positionScale) => `
struct Rotation { row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32> };
struct Frame { rotation: Rotation, translation: vec3<f32> };
@group(0) @binding(0) var<storage, read> affine: array<f32>;
@group(0) @binding(1) var<storage, read> angles: array<f32>;
@group(0) @binding(2) var<storage, read> aatype: array<f32>;
@group(0) @binding(3) var<storage, read> default_frames: array<f32>;
@group(0) @binding(4) var<storage, read> atom_groups: array<f32>;
@group(0) @binding(5) var<storage, read> literature_positions: array<f32>;
@group(0) @binding(6) var<storage, read> atom_mask: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

fn apply_rotation(r: Rotation, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(r.row0, v), dot(r.row1, v), dot(r.row2, v));
}
fn multiply_rotation(a: Rotation, b: Rotation) -> Rotation {
  let column0 = vec3<f32>(b.row0.x, b.row1.x, b.row2.x);
  let column1 = vec3<f32>(b.row0.y, b.row1.y, b.row2.y);
  let column2 = vec3<f32>(b.row0.z, b.row1.z, b.row2.z);
  return Rotation(
    vec3<f32>(dot(a.row0, column0), dot(a.row0, column1), dot(a.row0, column2)),
    vec3<f32>(dot(a.row1, column0), dot(a.row1, column1), dot(a.row1, column2)),
    vec3<f32>(dot(a.row2, column0), dot(a.row2, column1), dot(a.row2, column2))
  );
}
fn compose(a: Frame, b: Frame) -> Frame {
  return Frame(multiply_rotation(a.rotation, b.rotation), apply_rotation(a.rotation, b.translation) + a.translation);
}
fn quaternion_rotation(q: vec4<f32>) -> Rotation {
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return Rotation(
    vec3<f32>(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)),
    vec3<f32>(2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)),
    vec3<f32>(2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y))
  );
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let residue = id.x;
  if (residue >= arrayLength(&aatype)) { return; }
  let aa = min(u32(aatype[residue]), 20u);
  var frames: array<Frame, 8>;
  for (var group = 0u; group < 8u; group += 1u) {
    let base = (aa * 8u + group) * 16u;
    var r = Rotation(
      vec3<f32>(default_frames[base], default_frames[base+1u], default_frames[base+2u]),
      vec3<f32>(default_frames[base+4u], default_frames[base+5u], default_frames[base+6u]),
      vec3<f32>(default_frames[base+8u], default_frames[base+9u], default_frames[base+10u])
    );
    let t = vec3<f32>(default_frames[base+3u], default_frames[base+7u], default_frames[base+11u]);
    var sine = 0.0;
    var cosine = 1.0;
    if (group > 0u) {
      let angle_base = residue * 14u + (group - 1u) * 2u;
      sine = angles[angle_base]; cosine = angles[angle_base + 1u];
    }
    let row0 = vec3<f32>(r.row0.x, r.row0.y * cosine + r.row0.z * sine,
      -r.row0.y * sine + r.row0.z * cosine);
    let row1 = vec3<f32>(r.row1.x, r.row1.y * cosine + r.row1.z * sine,
      -r.row1.y * sine + r.row1.z * cosine);
    let row2 = vec3<f32>(r.row2.x, r.row2.y * cosine + r.row2.z * sine,
      -r.row2.y * sine + r.row2.z * cosine);
    r = Rotation(row0, row1, row2);
    frames[group] = Frame(r, t);
  }
  frames[5] = compose(frames[4], frames[5]);
  frames[6] = compose(frames[5], frames[6]);
  frames[7] = compose(frames[6], frames[7]);
  let affine_base = residue * 7u;
  let backbone = Frame(
    quaternion_rotation(vec4<f32>(affine[affine_base], affine[affine_base+1u],
      affine[affine_base+2u], affine[affine_base+3u])),
    ${positionScale.toFixed(1)} * vec3<f32>(affine[affine_base+4u], affine[affine_base+5u], affine[affine_base+6u])
  );
  for (var atom = 0u; atom < 14u; atom += 1u) {
    let table_index = aa * 14u + atom;
    let group = u32(atom_groups[table_index]);
    let position_base = table_index * 3u;
    let local = vec3<f32>(literature_positions[position_base], literature_positions[position_base+1u],
      literature_positions[position_base+2u]);
    let group_position = apply_rotation(frames[group].rotation, local) + frames[group].translation;
    let global = (apply_rotation(backbone.rotation, group_position) + backbone.translation) * atom_mask[table_index];
    let output_base = (residue * 14u + atom) * 3u;
    output[output_base] = global.x; output[output_base+1u] = global.y; output[output_base+2u] = global.z;
  }
}`;

const ATOM37_SHADER = `
@group(0) @binding(0) var<storage, read> atom14: array<f32>;
@group(0) @binding(1) var<storage, read> atom37_to_atom14: array<f32>;
@group(0) @binding(2) var<storage, read> atom37_mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= arrayLength(&output)) { return; }
  let coordinate = index % 3u;
  let atom37_index = index / 3u;
  let residue = atom37_index / 37u;
  let atom14_index = u32(atom37_to_atom14[atom37_index]);
  output[index] = atom14[(residue * 14u + atom14_index) * 3u + coordinate] * atom37_mask[atom37_index];
}`;

export class AtomGeometryGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device; this.allocator = new GpuBufferAllocator(device); this.pipelines = pipelineCacheForDevice(device);
  }
  async run(input) {
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value) => keep(this.allocator.upload(label, value, GPUBufferUsage.STORAGE));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    // ...defaults to the monomer value, so a caller that says nothing gets the
    // behaviour the bare 10.0 in the original gave.
    const positionScale = input.positionScale ?? MONOMER_POSITION_SCALE;
    if (!Number.isFinite(positionScale) || positionScale <= 0) {
      throw new RangeError("position scale must be a positive finite number");
    }
    try {
      const [atom14Pipeline, atom37Pipeline] = await Promise.all([
        this.pipelines.get(`geometry:atom14:scale-${positionScale}`, createAtom14Shader(positionScale)), this.pipelines.get("geometry:atom37", ATOM37_SHADER),
      ]);
      const affine = upload("geometry.affine", input.affine); const angles = upload("geometry.angles", input.angles);
      const aatype = upload("geometry.aatype", input.aatype);
      const frames = upload("geometry.frames", input.tables.defaultFrames);
      const groups = upload("geometry.groups", input.tables.atom14ToGroup);
      const positions = upload("geometry.positions", input.tables.atom14Positions);
      const atom14Mask = upload("geometry.atom14-mask", input.tables.atom14Mask);
      const mapping = upload("geometry.atom37-mapping", input.atom37ToAtom14);
      const atom37Mask = upload("geometry.atom37-mask", input.atom37Mask);
      const atom14 = allocate("geometry.atom14", input.length * 14 * 3, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const atom37 = allocate("geometry.atom37", input.length * 37 * 3, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({ label: "atom-geometry" });
      const pass = (pipeline, buffers, x) => {
        const compute = encoder.beginComputePass(); compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })) }));
        compute.dispatchWorkgroups(x); compute.end();
      };
      pass(atom14Pipeline, [affine, angles, aatype, frames, groups, positions, atom14Mask, atom14], input.length);
      pass(atom37Pipeline, [atom14, mapping, atom37Mask, atom37], Math.ceil(input.length * 37 * 3 / 64));
      const readbacks = [atom14, atom37].map((value, index) => {
        const result = allocate(`geometry.readback-${index}`, value.byteLength / 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(value.buffer, 0, result.buffer, 0, value.byteLength); return result;
      });
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(readbacks.map((value) => value.buffer.mapAsync(GPUMapMode.READ)));
      const result = readbacks.map((value) => {
        const array = new Float32Array(value.buffer.getMappedRange().slice(0)); value.buffer.unmap(); return array;
      });
      return { atom14: result[0], atom37: result[1] };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}

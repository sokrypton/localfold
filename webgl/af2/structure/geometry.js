/**
 * Atom coordinates from backbone frames and sidechain torsions - the last stage
 * of the structure module, and the first place this port produces something a
 * person can look at.
 *
 * Three draws:
 *
 *   1. the eight rigid groups of every residue, composed:  [length][8][16]
 *   2. atom14 - fourteen dense slots a residue             [length][14][3]
 *   3. atom37 - the sparse all-atom layout                 [length][37][3]
 *
 * 🔴 THE FRAMES ARE SPLIT OUT INTO THEIR OWN DRAW BECAUSE atom14 IS THREE FLOATS
 * AN ATOM. A texel holds four, so a fragment's four output elements straddle
 * atoms and sometimes residues - and rebuilding a residue's eight-frame chain
 * for each of them would be about twenty times the work. With the frames
 * materialised, the second draw is one 3x3 multiply-add per coordinate.
 *
 * 🔴 FIVE CONVENTIONS, EVERY ONE READ OFF THE WGSL SHADER:
 *
 *   - a default frame is a 4x4 ROW-MAJOR matrix, so its texels are
 *     `(row0, t.x)`, `(row1, t.y)`, `(row2, t.z)` - the translation is the
 *     fourth COLUMN, interleaved with the rows, not three floats at the end
 *   - the quaternion is `(w, x, y, z)` and is used UNNORMALISED, as the
 *     reference does; normalising here would be a different rotation
 *   - the backbone translation is scaled by **10**, which is AF2's position
 *     scale, and nothing in the tensor says so
 *   - only groups 5, 6 and 7 chain, each off the one before, and groups 0 to 4
 *     stand alone; chaining all eight would place every sidechain atom wrongly
 *   - the torsion rotates about x by mixing COLUMNS 1 and 2 of each row, with
 *     `(sine, cosine)` taken in that order from the angle pair - the opposite
 *     order is a plausible structure with every sidechain mirrored
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** The composed rigid groups, four vec4s a frame: three rotation rows then the translation. */
export const composedFramesSource = (length) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
// No affine here: the backbone frame is applied in the atom draw, not this one -
// and declaring a sampler a kernel does not read is refused by name, because the
// compiler removes it and the binding would silently go to unit 0.
uniform sampler2D anglesTex;
uniform sampler2D aatypeTex;
uniform sampler2D defaultFramesTex;

struct Frame { vec3 row0; vec3 row1; vec3 row2; vec3 t; };

// A default frame is a 4x4 row-major matrix: texel k is (row k, translation k).
Frame defaultFrame(int aa, int group) {
  int base = (aa * 8 + group) * 4;
  vec4 d0 = fetch4(defaultFramesTex, base);
  vec4 d1 = fetch4(defaultFramesTex, base + 1);
  vec4 d2 = fetch4(defaultFramesTex, base + 2);
  return Frame(d0.xyz, d1.xyz, d2.xyz, vec3(d0.w, d1.w, d2.w));
}

// The torsion, about x, mixing columns 1 and 2 of every row.
Frame twisted(Frame f, float sine, float cosine) {
  return Frame(
    vec3(f.row0.x, f.row0.y * cosine + f.row0.z * sine, -f.row0.y * sine + f.row0.z * cosine),
    vec3(f.row1.x, f.row1.y * cosine + f.row1.z * sine, -f.row1.y * sine + f.row1.z * cosine),
    vec3(f.row2.x, f.row2.y * cosine + f.row2.z * sine, -f.row2.y * sine + f.row2.z * cosine),
    f.t);
}

Frame groupFrame(int residue, int aa, int group) {
  Frame f = defaultFrame(aa, group);
  if (group == 0) return f;
  int anglePair = residue * 14 + (group - 1) * 2;
  return twisted(f, fetch1(anglesTex, anglePair), fetch1(anglesTex, anglePair + 1));
}

vec3 applyRotation(Frame f, vec3 v) {
  return vec3(dot(f.row0, v), dot(f.row1, v), dot(f.row2, v));
}

Frame compose(Frame a, Frame b) {
  vec3 c0 = vec3(b.row0.x, b.row1.x, b.row2.x);
  vec3 c1 = vec3(b.row0.y, b.row1.y, b.row2.y);
  vec3 c2 = vec3(b.row0.z, b.row1.z, b.row2.z);
  return Frame(
    vec3(dot(a.row0, c0), dot(a.row0, c1), dot(a.row0, c2)),
    vec3(dot(a.row1, c0), dot(a.row1, c1), dot(a.row1, c2)),
    vec3(dot(a.row2, c0), dot(a.row2, c1), dot(a.row2, c2)),
    applyRotation(a, b.t) + a.t);
}

// Only 5, 6 and 7 chain, each off the one before it.
Frame chained(int residue, int aa, int group) {
  if (group < 5) return groupFrame(residue, aa, group);
  Frame f = compose(groupFrame(residue, aa, 4), groupFrame(residue, aa, 5));
  if (group == 5) return f;
  f = compose(f, groupFrame(residue, aa, 6));
  if (group == 6) return f;
  return compose(f, groupFrame(residue, aa, 7));
}

void main() {
  int texel = linearTexel();
  if (texel >= L * 8 * 4) { result = vec4(0.0); return; }
  int part = texel % 4;
  int rest = texel / 4;
  int group = rest % 8;
  int residue = rest / 8;
  int aa = min(int(fetch1(aatypeTex, residue)), 20);
  Frame f = chained(residue, aa, group);
  vec3 vector = part == 0 ? f.row0 : part == 1 ? f.row1 : part == 2 ? f.row2 : f.t;
  result = vec4(vector, 0.0);
}`;

export const atom14Source = (length) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int ELEMENTS = ${length * 14 * 3};
uniform sampler2D affineTex;
uniform sampler2D aatypeTex;
uniform sampler2D framesTex;
uniform sampler2D atomGroupsTex;
uniform sampler2D literatureTex;
uniform sampler2D atomMaskTex;

// The quaternion is (w, x, y, z) and is used as given - unnormalised, as the
// reference does.
void quaternionRows(vec4 q, out vec3 row0, out vec3 row1, out vec3 row2) {
  float w = q.x; float x = q.y; float y = q.z; float z = q.w;
  row0 = vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w));
  row1 = vec3(2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w));
  row2 = vec3(2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

float coordinateOf(int element) {
  int axis = element % 3;
  int atomIndex = element / 3;
  int atom = atomIndex % 14;
  int residue = atomIndex / 14;
  int aa = min(int(fetch1(aatypeTex, residue)), 20);
  int table = aa * 14 + atom;
  int group = int(fetch1(atomGroupsTex, table));

  int frameBase = (residue * 8 + group) * 4;
  vec3 r0 = fetch4(framesTex, frameBase).xyz;
  vec3 r1 = fetch4(framesTex, frameBase + 1).xyz;
  vec3 r2 = fetch4(framesTex, frameBase + 2).xyz;
  vec3 groupT = fetch4(framesTex, frameBase + 3).xyz;

  vec3 local = vec3(
    fetch1(literatureTex, table * 3),
    fetch1(literatureTex, table * 3 + 1),
    fetch1(literatureTex, table * 3 + 2));
  vec3 inGroup = vec3(dot(r0, local), dot(r1, local), dot(r2, local)) + groupT;

  int affineBase = residue * 7;
  vec4 q = vec4(
    fetch1(affineTex, affineBase), fetch1(affineTex, affineBase + 1),
    fetch1(affineTex, affineBase + 2), fetch1(affineTex, affineBase + 3));
  vec3 b0; vec3 b1; vec3 b2;
  quaternionRows(q, b0, b1, b2);
  // The backbone translation carries AF2's factor of ten.
  vec3 backboneT = 10.0 * vec3(
    fetch1(affineTex, affineBase + 4), fetch1(affineTex, affineBase + 5),
    fetch1(affineTex, affineBase + 6));
  vec3 global = vec3(dot(b0, inGroup), dot(b1, inGroup), dot(b2, inGroup)) + backboneT;
  float masked = fetch1(atomMaskTex, table);
  return (axis == 0 ? global.x : axis == 1 ? global.y : global.z) * masked;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    coordinateOf(base),
    base + 1 < ELEMENTS ? coordinateOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? coordinateOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? coordinateOf(base + 3) : 0.0);
}`;

export const atom37Source = (length) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${length * 37 * 3};
uniform sampler2D atom14Tex;
uniform sampler2D mapTex;
uniform sampler2D maskTex;

float slotOf(int element) {
  int axis = element % 3;
  int slot = element / 3;
  int residue = slot / 37;
  int atom14 = int(fetch1(mapTex, slot));
  return fetch1(atom14Tex, (residue * 14 + atom14) * 3 + axis) * fetch1(maskTex, slot);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    slotOf(base),
    base + 1 < ELEMENTS ? slotOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? slotOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? slotOf(base + 3) : 0.0);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{affine: Tensor, angles: Tensor, aatype: Tensor, atom37ToAtom14: Tensor,
 *   atom37Mask: Tensor, length: number,
 *   tables: {defaultFrames: Tensor, atom14ToGroup: Tensor, atom14Positions: Tensor,
 *     atom14Mask: Tensor}}} call
 * @returns {{atom14: Tensor, atom37: Tensor, frames: Tensor}} the caller owns all three
 */
export function atomGeometry(execution, call) {
  const { affine, angles, aatype, atom37ToAtom14, atom37Mask, length, tables } = call;
  if (affine.elements !== length * 7) {
    throw new RangeError(`atomGeometry: affine holds ${affine.elements} floats, not ${length} x 7`);
  }
  if (angles.elements !== length * 14) {
    throw new RangeError(`atomGeometry: angles hold ${angles.elements} floats, not ${length} x 14`);
  }

  const frames = execution.allocate(length * 8 * 4 * 4, "geometry.frames");
  execution.dispatch({
    source: composedFramesSource(length),
    output: frames,
    inputs: {
      anglesTex: angles, aatypeTex: aatype, defaultFramesTex: tables.defaultFrames,
    },
    label: `geometry.frames:${length}`,
  });

  const atom14 = execution.allocate(length * 14 * 3, "geometry.atom14");
  execution.dispatch({
    source: atom14Source(length),
    output: atom14,
    inputs: {
      affineTex: affine, aatypeTex: aatype, framesTex: frames,
      atomGroupsTex: tables.atom14ToGroup, literatureTex: tables.atom14Positions,
      atomMaskTex: tables.atom14Mask,
    },
    label: `geometry.atom14:${length}`,
  });

  const atom37 = execution.allocate(length * 37 * 3, "geometry.atom37");
  execution.dispatch({
    source: atom37Source(length),
    output: atom37,
    inputs: { atom14Tex: atom14, mapTex: atom37ToAtom14, maskTex: atom37Mask },
    label: `geometry.atom37:${length}`,
  });
  return { atom14, atom37, frames };
}

/**
 * Length-generic WebGL2 programs: one program for a kernel at every sequence length.
 *
 * 🔴 A NEW LENGTH RECOMPILED NEARLY EVERY KERNEL, BECAUSE NEARLY EVERY KERNEL BAKES IT IN. An AF3
 * fold draws with 156 programs and 140 of them differ between 60 and 101 tokens
 * (tools/gpu/probe-webgl-af3-programs.js), only in `const int NAME = <n>;` lines - ROWS, ELEMENTS,
 * PAIRS. On Safari 16.3 every one of those is a link of 20-60 ms on the page's thread, ~3 s a new
 * length. WebGPU met the same thing and answered it in webgpu/runtime/pipeline-cache.js
 * (`lengthPlan`, `genericLengthSource`); this is that answer for GLSL.
 *
 * Kernels are grouped by their SKELETON: the source with every module-scope integer constant's
 * value blanked. The constants whose values follow the length are found by comparing the same
 * fold's kernels at two lengths (`learnLengthNames`); `genericSource` turns those into `uniform int` and
 * any module constant computed from them into a `#define` of its expression, so one program
 * serves the group and the dispatch sets the values. Constants that never differ - channel
 * widths - stay constants, so the loops over them keep their compile-time bounds.
 *
 * It is the same integer arithmetic, so the same results: a FLOAT computed from a moving constant
 * is refused (`null`, compiled specialised as before), because one the compiler folds is not
 * bit-for-bit one computed at run time. A program that does not compile generic (a moving
 * constant used as an array size) is the caller's to fall back from.
 */

const LITERAL = /^const int (\w+) = (-?\d+);$/;
const DECLARATION = /^const (\w+) (\w+) = (.+);$/;

/**
 * @param {string} source a fragment shader
 * @returns {{skeleton: string, values: Map<string, number>}} the source with its module-scope
 *   integer literals blanked, and those literals by name
 */
export function lengthSkeleton(source) {
  const values = new Map();
  const skeleton = source.split("\n").map((line) => {
    const match = LITERAL.exec(line);
    if (match === null) return line;
    values.set(match[1], Number(match[2]));
    return `const int ${match[1]} = ?;`;
  }).join("\n");
  return { skeleton, values };
}

/**
 * Add to `known` (skeleton -> the names that move) the constants that follow the LENGTH: the same
 * kernels collected at two lengths (a dry run of the same fold at each), and within each skeleton
 * the constants whose SET of values differs between the two.
 *
 * 🔴 THE SETS, NOT ANY DIFFERENCE BETWEEN MEMBERS. One kernel runs at several channel widths
 * within a fold (a layer norm at 64, 128 and 384), so its CHANNELS differs between members of one
 * length - but it takes the same values at every length, and a width made a uniform loses the
 * compile-time bound of the loop over it. A length moves the set; a width does not.
 *
 * @param {Iterable<string>} atOne sources at one length
 * @param {Iterable<string>} atOther the same fold's sources at another
 * @param {Map<string, Set<string>>} known
 */
export function learnLengthNames(atOne, atOther, known) {
  const valueSets = (sources) => {
    const bySkeleton = new Map();
    for (const source of sources) {
      const { skeleton, values } = lengthSkeleton(source);
      if (!bySkeleton.has(skeleton)) bySkeleton.set(skeleton, new Map());
      const names = bySkeleton.get(skeleton);
      for (const [name, value] of values) {
        if (!names.has(name)) names.set(name, new Set());
        names.get(name).add(value);
      }
    }
    return bySkeleton;
  };
  const one = valueSets(atOne);
  const other = valueSets(atOther);
  for (const [skeleton, names] of one) {
    const theirs = other.get(skeleton);
    if (theirs === undefined) continue;
    const moving = known.get(skeleton) ?? new Set();
    for (const [name, values] of names) {
      const those = theirs.get(name) ?? new Set();
      if (values.size !== those.size || [...values].some((value) => !those.has(value))) moving.add(name);
    }
    if (moving.size > 0) known.set(skeleton, moving);
  }
  return known;
}

/**
 * The generic program for `source`, if `names` move in its group.
 *
 * @param {string} source
 * @param {Set<string>} names the constants that become uniforms
 * @returns {{source: string, values: [string, number][]} | null} the generic source and the values
 *   this member sets; null when nothing moves or a float depends on what does
 */
export function genericSource(source, names) {
  const lines = source.split("\n");
  const values = [];
  const moving = new Set();
  for (const line of lines) {
    const match = LITERAL.exec(line);
    if (match !== null && names.has(match[1])) {
      values.push([match[1], Number(match[2])]);
      moving.add(match[1]);
    }
  }
  if (values.length === 0) return null;
  // ...the module constants computed from them, transitively.
  const derived = new Set();
  const refers = (rhs) => [...moving, ...derived].some((name) => new RegExp(`\\b${name}\\b`).test(rhs));
  for (let grew = true; grew;) {
    grew = false;
    for (const line of lines) {
      const match = DECLARATION.exec(line);
      if (match === null || moving.has(match[2]) || derived.has(match[2])) continue;
      if (LITERAL.test(line) && !names.has(match[2])) continue;
      if (!refers(match[3])) continue;
      if (match[1] !== "int") return null;
      derived.add(match[2]);
      grew = true;
    }
  }
  const out = lines.map((line) => {
    const literal = LITERAL.exec(line);
    if (literal !== null && moving.has(literal[1])) return `uniform int ${literal[1]};`;
    const declared = DECLARATION.exec(line);
    if (declared !== null && derived.has(declared[2])) return `#define ${declared[2]} (${declared[3]})`;
    return line;
  });
  return { source: out.join("\n"), values };
}

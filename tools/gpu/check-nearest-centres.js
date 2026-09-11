/**
 * The GPU nearest-centre search against the host loop it replaces.
 *
 * 🔴 THE BAR IS ZERO DIFFERING ELEMENTS, not a tolerance. The assignment picks
 * which cluster an extra row joins, which decides the cluster profile and so
 * the prediction; an assignment that is "nearly right" is a different fold.
 *
 * 🔴 AND TIES ARE THE WHOLE RISK, so this makes them certain: one arm
 * DUPLICATES centres so that several score identically, and the host keeps the
 * FIRST at an equal score. A join that broke the other way would agree on
 * every random alignment and disagree on every real one, where near-identical
 * sequences are the normal case.
 *
 * The host loop is carried here as a copy rather than imported, so the two
 * cannot drift into agreement through a shared edit - the same reason
 * check-esmfold2-trunk-pack.js keeps its own arm.
 */
import { paddedCodeWords } from "../../src/input/a3m-features.js";
import { assignNearestCentres } from "../../src/input/nearest-centres-webgpu.js";

/** The shipped host loop, copied. Keep this in step deliberately, never by import. */
function hostAssign(centreWords, extraWords, words, centreCount, rows) {
  const assignments = new Uint16Array(rows);
  for (let index = 0; index < rows; index += 1) {
    const extraBase = index * words;
    let best = 0;
    let bestScore = -1;
    for (let centre = 0; centre < centreCount; centre += 1) {
      const centreBase = centre * words;
      let score = 0;
      for (let word = 0; word < words; word += 1) {
        const difference = centreWords[centreBase + word] ^ extraWords[extraBase + word];
        const zeros = ~(((difference & 0x7f7f7f7f) + 0x7f7f7f7f) | difference) & 0x80808080;
        score += ((zeros >>> 7) & 1) + ((zeros >>> 15) & 1)
          + ((zeros >>> 23) & 1) + ((zeros >>> 31) & 1);
      }
      if (score > bestScore) { bestScore = score; best = centre; }
    }
    assignments[index] = best;
  }
  return assignments;
}

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function makeCase({ length, centres, rows, duplicate, seed }) {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  // Codes 0..21 as an alignment holds them, plus the occasional 22 - which is
  // above 20 and so can never agree, the rule the padding encodes.
  const depth = centres + rows;
  const encoded = new Uint8Array(depth * length);
  for (let index = 0; index < encoded.length; index += 1) {
    encoded[index] = Math.floor(next() * 23);
  }
  const centerCodes = new Uint8Array(centres * length);
  for (let centre = 0; centre < centres; centre += 1) {
    // 🔴 THE DUPLICATE ARM. Every centre past the first `duplicate` is a copy
    // of centre `centre % duplicate`, so whole groups score identically and the
    // tie rule is exercised on nearly every row.
    const from = duplicate > 0 ? (centre % duplicate) : centre;
    centerCodes.set(encoded.subarray(from * length, (from + 1) * length), centre * length);
  }
  const extras = Array.from({ length: rows }, (_, index) => centres + index);
  return { encoded, centerCodes, extras, centres, length };
}

export async function main(device, args = []) {
  const cases = [
    { name: "af2 monomer, 59 residues", length: 59, centres: 508, rows: 1024, duplicate: 0, seed: 7 },
    { name: "825 residues", length: 825, centres: 508, rows: 1024, duplicate: 0, seed: 11 },
    { name: "a length that is not a multiple of four", length: 61, centres: 64, rows: 96, duplicate: 0, seed: 13 },
    { name: "more centres than lanes, all tied in groups of 8", length: 59, centres: 508, rows: 512, duplicate: 8, seed: 17 },
    // 🔴 THESE TWO ARE DEGENERATE ON PURPOSE and the right answer is centre 0
    // for every row - every centre identical, so the tie rule decides them all,
    // and a single centre, so the join has one candidate and 63 empty lanes.
    // The distinctness control below would call both of them trivial, which is
    // exactly what they are, so they carry their own assertion instead.
    { name: "every centre identical", length: 40, centres: 130, rows: 64, duplicate: 1, seed: 19, allZero: true },
    { name: "one centre", length: 37, centres: 1, rows: 32, duplicate: 0, seed: 23, allZero: true },
    { name: "fewer centres than lanes", length: 53, centres: 9, rows: 40, duplicate: 3, seed: 29 },
  ];
  const only = option(args, "case", "");

  const searches = [];
  const expected = [];
  const used = [];
  for (const spec of cases) {
    if (only !== "" && !spec.name.includes(only)) continue;
    const built = makeCase(spec);
    const padded = paddedCodeWords(built.centerCodes, built.encoded, built.extras,
      built.centres, built.length);
    searches.push({ ...padded, centres: built.centres });
    expected.push(hostAssign(padded.centreWords, padded.extraWords, padded.words,
      built.centres, padded.rows));
    used.push(spec);
  }

  // Every case in ONE submit, which is also the batching a fold's recycles use.
  const actual = await assignNearestCentres(device, searches);

  const results = [];
  let failed = 0;
  for (const [index, spec] of used.entries()) {
    let differing = 0;
    let firstAt = -1;
    for (let row = 0; row < expected[index].length; row += 1) {
      if (expected[index][row] !== actual[index][row]) {
        differing += 1;
        if (firstAt < 0) firstAt = row;
      }
    }
    // 🔴 AND A CONTROL THAT THE CASE IS NOT TRIVIAL. An arm where every row
    // lands on centre 0 would agree perfectly and check nothing - unless zero
    // is the answer the case exists to demand, which `allZero` states.
    const distinct = new Set(expected[index]).size;
    const zeros = actual[index].every((centre) => centre === 0);
    const ok = differing === 0
      && (spec.allZero === true ? zeros && distinct === 1 : distinct > 1);
    if (!ok) failed += 1;
    results.push({ case: spec.name, rows: expected[index].length, centres: spec.centres,
                   differing, firstAt, distinctCentresChosen: distinct, ok });
    console.log(`${ok ? "ok  " : "FAIL"} ${spec.name}: ${differing} differing of `
      + `${expected[index].length}, ${distinct} distinct centres chosen`
      + (firstAt < 0 ? "" : `, first at row ${firstAt}`));
  }
  if (failed > 0) throw new Error(`${failed} case(s) disagree with the host loop`);
  return { cases: results };
}

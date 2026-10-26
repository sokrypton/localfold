/**
 * Upstream's selection language: `atomworks.io.utils.query.QueryExpression` and its `sel('...')`
 * path DSL (`AtomSelectionStack.from_query`), as RFD4-Proteina's `resolve_mask` evaluates them -
 * the query parsed as a PYTHON expression and evaluated with numpy's semantics over the atoms.
 *
 * 🔴 IT IS A PYTHON EXPRESSION, SO PYTHON'S PRECEDENCE DECIDES WHAT A QUERY MEANS. `&` binds
 * tighter than a comparison, so `res_id >= 2 & chain_id == 'A'` is `res_id >= (2 & chain_id) ==
 * 'A'` - a chained comparison that is all-false - and not what it reads like. The hand-written
 * evaluator this replaced parsed it the other way, ignored trailing text (`res_name == 'ALA'
 * garbage` matched every alanine), dropped chained comparisons (`2 <= res_id <= 3` matched
 * nothing), turned a bare chain id or residue name into a mask where upstream raises NameError,
 * and knew five of the forty annotations upstream derives on demand.
 *
 * 🔴 AND IT IS HELD TO UPSTREAM ITSELF, NOT TO A READING OF IT: tools/check-rfd4p-query.py
 * evaluates a corpus of queries with atomworks and requires this module, and the native
 * featuriser's port of it, to give the same mask on every atom.
 */
import { getTokenStarts } from './bonds.js';
import {
  CARBOHYDRATE_NAMES, CHAIN_TYPE, DNA_BACKBONE_ATOM_NAMES, MASKED,
  NON_POLYMER_CHAIN_TYPES, NUCLEIC_ACID_BACKBONE_ATOM_NAMES, NUCLEIC_CHAIN_TYPES,
  POLYMER_CHAIN_TYPES, PROTEIN_BACKBONE_ATOM_NAMES, PROTEIN_CHAIN_TYPES, PURINE_RESIDUES,
  PYRIMIDINE_RESIDUES, RNA_BACKBONE_ATOM_NAMES, STANDARD_AA, STANDARD_DNA, STANDARD_RNA,
  TIP_ATOM_NAMES, UNKNOWN_AA, UNKNOWN_DNA, UNKNOWN_RNA,
} from './query-tables.js';

// ---------------------------------------------------------------- the tokenizer

const KEYWORDS = new Set(['and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'if', 'else',
  'lambda', 'for', 'await', 'yield']);
const OPERATORS = ['**', '//', '<<', '>>', '<=', '>=', '==', '!=', '(', ')', '[', ']', ',', '<', '>',
  '&', '|', '^', '~', '+', '-', '*', '/', '%', '@', '.', ':'];

class QuerySyntaxError extends Error {}

function decodeEscapes(body) {
  let out = '';
  for (let i = 0; i < body.length; i += 1) {
    const c = body[i];
    if (c !== '\\' || i + 1 >= body.length) { out += c; continue; }
    const n = body[(i += 1)];
    const simple = { n: '\n', t: '\t', r: '\r', '\\': '\\', "'": "'", '"': '"', a: '\x07', b: '\b', f: '\f', v: '\v', 0: '\0' };
    if (n in simple) out += simple[n];
    else if (n === '\n') out += '';
    else out += `\\${n}`;
  }
  return out;
}

function tokenize(src) {
  const toks = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i += 1; continue; }
    // A string, with an optional prefix: plain and r/R are literals; b/f/u and the rest are not a
    // `str` upstream either (bytes, f-strings), so they are refused as unsupported.
    const pre = /^([rRuUbBfF]{1,2})?(['"])/.exec(src.slice(i));
    if (pre && (pre[1] === undefined || /^[a-zA-Z]+$/.test(pre[1]))) {
      const prefix = (pre[1] ?? '').toLowerCase();
      const quote = pre[2];
      let j = i + pre[0].length;
      const triple = src.startsWith(quote.repeat(3), j - 1);
      if (triple) j += 2;
      let body = '';
      let closed = false;
      while (j < src.length) {
        if (!prefix.includes('r') && src[j] === '\\' && j + 1 < src.length) { body += src[j] + src[j + 1]; j += 2; continue; }
        if (prefix.includes('r') && src[j] === '\\' && j + 1 < src.length) { body += src[j] + src[j + 1]; j += 2; continue; }
        if (triple ? src.startsWith(quote.repeat(3), j) : src[j] === quote) { closed = true; j += triple ? 3 : 1; break; }
        if (!triple && src[j] === '\n') break;
        body += src[j];
        j += 1;
      }
      if (!closed) throw new QuerySyntaxError('unterminated string literal');
      if (prefix.includes('b') || prefix.includes('f')) {
        throw new Error(`Unsupported node type: ${prefix.includes('b') ? 'bytes' : 'JoinedStr'}`);
      }
      toks.push({ t: 'str', v: prefix.includes('r') ? body : decodeEscapes(body) });
      i = j;
      continue;
    }
    const num = /^(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?)([jJ])?/.exec(src.slice(i));
    if (num && num[0] && /[\d]/.test(num[0][0] === '.' ? num[0][1] : num[0][0])) {
      if (num[1]) throw new Error('Unsupported node type: complex');
      const text = num[0].replace(/_/g, '');
      const isInt = /^(0[xX]|0[oO]|0[bB])/.test(text) || /^\d+$/.test(text);
      const v = /^0[oO]/.test(text) ? parseInt(text.slice(2), 8)
        : /^0[bB]/.test(text) ? parseInt(text.slice(2), 2)
          : Number(text);
      toks.push({ t: 'num', v, isInt });
      i += num[0].length;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (name) {
      toks.push({ t: KEYWORDS.has(name[0]) ? 'kw' : 'name', v: name[0] });
      i += name[0].length;
      continue;
    }
    const op = OPERATORS.find((o) => src.startsWith(o, i));
    if (!op) throw new QuerySyntaxError(`invalid character '${c}'`);
    toks.push({ t: 'op', v: op });
    i += op.length;
  }
  toks.push({ t: 'eof' });
  return toks;
}

// ---------------------------------------------------------------- the parser (Python's grammar)

function parse(src) {
  const toks = tokenize(src);
  let p = 0;
  const peek = (k = 0) => toks[p + k];
  const isOp = (v, k = 0) => peek(k).t === 'op' && peek(k).v === v;
  const isKw = (v, k = 0) => peek(k).t === 'kw' && peek(k).v === v;
  const expectOp = (v) => {
    if (!isOp(v)) throw new QuerySyntaxError(`expected '${v}'`);
    p += 1;
  };

  function expression() {
    const e = orTest();
    if (isKw('if')) throw new Error('Unsupported node type: IfExp');
    return e;
  }
  function orTest() {
    const values = [andTest()];
    while (isKw('or')) { p += 1; values.push(andTest()); }
    return values.length > 1 ? { k: 'bool', op: 'or', values } : values[0];
  }
  function andTest() {
    const values = [notTest()];
    while (isKw('and')) { p += 1; values.push(notTest()); }
    return values.length > 1 ? { k: 'bool', op: 'and', values } : values[0];
  }
  function notTest() {
    if (isKw('not')) { p += 1; return { k: 'unary', op: 'not', operand: notTest() }; }
    return comparison();
  }
  function compOp() {
    const t = peek();
    if (t.t === 'op' && ['<', '>', '==', '>=', '<=', '!='].includes(t.v)) { p += 1; return t.v; }
    if (isKw('in')) { p += 1; return 'in'; }
    if (isKw('not') && isKw('in', 1)) { p += 2; return 'not in'; }
    if (isKw('is')) {
      p += 1;
      if (isKw('not')) { p += 1; return 'is not'; }
      return 'is';
    }
    return null;
  }
  function comparison() {
    const left = bitOr();
    const ops = [];
    const comparators = [];
    for (let op = compOp(); op !== null; op = compOp()) {
      ops.push(op);
      comparators.push(bitOr());
    }
    return ops.length ? { k: 'compare', left, ops, comparators } : left;
  }
  const binary = (next, ops) => () => {
    let left = next();
    while (peek().t === 'op' && ops.includes(peek().v)) {
      const op = peek().v;
      p += 1;
      left = { k: 'binop', op, left, right: next() };
    }
    return left;
  };
  function factor() {
    if (peek().t === 'op' && ['+', '-', '~'].includes(peek().v)) {
      const op = peek().v;
      p += 1;
      return { k: 'unary', op, operand: factor() };
    }
    return power();
  }
  function power() {
    const base = primary();
    if (isOp('**')) { p += 1; return { k: 'binop', op: '**', left: base, right: factor() }; }
    return base;
  }
  const term = binary(factor, ['*', '/', '//', '%', '@']);
  const arith = binary(term, ['+', '-']);
  const shift = binary(arith, ['<<', '>>']);
  const bitAnd = binary(shift, ['&']);
  const bitXor = binary(bitAnd, ['^']);
  const bitOr = binary(bitXor, ['|']);

  function primary() {
    let node = atom();
    for (;;) {
      if (isOp('(')) {
        p += 1;
        const args = [];
        let keywords = false;
        while (!isOp(')')) {
          if (peek().t === 'name' && isOp('=', 1)) keywords = true;
          if (isOp('*') || isOp('**')) throw new Error('Unsupported node type: Starred');
          args.push(expression());
          if (isOp(',')) p += 1;
          else break;
        }
        expectOp(')');
        node = { k: 'call', func: node, args, keywords };
      } else if (isOp('[')) {
        throw new Error('Unsupported node type: Subscript');
      } else if (isOp('.')) {
        throw new Error('Unsupported node type: Attribute');
      } else {
        return node;
      }
    }
  }
  function atom() {
    const t = peek();
    if (t.t === 'num') { p += 1; return { k: 'const', v: t.v }; }
    if (t.t === 'str') {
      let v = '';
      while (peek().t === 'str') { v += peek().v; p += 1; }
      return { k: 'const', v };
    }
    if (t.t === 'kw' && (t.v === 'True' || t.v === 'False' || t.v === 'None')) {
      p += 1;
      return { k: 'const', v: t.v === 'True' ? true : t.v === 'False' ? false : null };
    }
    if (t.t === 'name') { p += 1; return { k: 'name', id: t.v }; }
    if (isOp('(')) {
      p += 1;
      if (isOp(')')) { p += 1; return { k: 'tuple', elts: [] }; }
      const first = expression();
      if (isOp(',')) {
        const elts = [first];
        while (isOp(',')) {
          p += 1;
          if (isOp(')')) break;
          elts.push(expression());
        }
        expectOp(')');
        return { k: 'tuple', elts };
      }
      expectOp(')');
      return first;
    }
    if (isOp('[')) {
      p += 1;
      const elts = [];
      while (!isOp(']')) {
        elts.push(expression());
        if (isOp(',')) p += 1;
        else break;
      }
      expectOp(']');
      return { k: 'list', elts };
    }
    if (t.t === 'kw' && t.v === 'lambda') throw new Error('Unsupported node type: Lambda');
    throw new QuerySyntaxError(`invalid syntax at ${t.t === 'eof' ? 'end of query' : `'${t.v}'`}`);
  }

  const tree = expression();
  // 🔴 THE WHOLE STRING IS ONE EXPRESSION. `ast.parse(..., mode="eval")` refuses anything left
  // over, where the evaluator this replaced stopped at the first token it did not want.
  if (peek().t !== 'eof') throw new QuerySyntaxError(`invalid syntax at '${peek().v}'`);
  return tree;
}

// ---------------------------------------------------------------- numpy's semantics

const arr = (v) => ({ k: 'arr', v });
const sc = (v) => ({ k: 'sc', v });

function truthy(x) {
  if (x === null || x === undefined) return false;
  if (typeof x === 'number') return x !== 0;   // NaN is truthy, as `bool(nan)` is
  if (typeof x === 'string') return x.length > 0;
  return Boolean(x);
}

function ensureBool(val, n) {
  if (val.k === 'sc') return new Array(n).fill(truthy(val.v));
  const out = val.k === 'arr' ? val.v.map(truthy) : val.v.map((e) => truthy(e.k === 'sc' ? e.v : e));
  if (out.length !== n) {
    throw new Error(`Query resulted in mask of length ${out.length}, but AtomArray has length ${n}`);
  }
  return out;
}

const numeric = (x) => (typeof x === 'boolean' ? Number(x) : x);

function cmp(op, a, b) {
  const x = numeric(a);
  const y = numeric(b);
  if (op === '==' || op === '!=') {
    let eq;
    if (x === null || y === null) eq = x === y;
    else if (typeof x !== typeof y) eq = false;
    else eq = x === y;
    return op === '==' ? eq : !eq;
  }
  if (x === null || y === null || typeof x !== typeof y) {
    throw new TypeError(`'${op}' not supported between instances of '${typeof a}' and '${typeof b}'`);
  }
  switch (op) {
    case '<': return x < y;
    case '<=': return x <= y;
    case '>': return x > y;
    case '>=': return x >= y;
    default: throw new Error(`Unsupported comparison '${op}'`);
  }
}

const memberOf = (x, list) => list.some((y) => cmp('==', x, y));

function scalars(val) {
  if (val.k === 'arr') return val.v;
  if (val.k === 'list') return val.v.map((e) => (e.k === 'sc' ? e.v : e));
  throw new TypeError(`Argument of type '${typeof val.v}' is not iterable`);
}

function elementwise(a, b, f) {
  if (a.k === 'sc' && b.k === 'sc') return sc(f(a.v, b.v));
  if (a.k === 'arr' && b.k === 'arr') {
    if (a.v.length !== b.v.length) throw new Error('operands could not be broadcast together');
    return arr(a.v.map((x, i) => f(x, b.v[i])));
  }
  if (a.k === 'arr' && b.k === 'sc') return arr(a.v.map((x) => f(x, b.v)));
  if (a.k === 'sc' && b.k === 'arr') return arr(b.v.map((y) => f(a.v, y)));
  // A Python list on one side: numpy makes it an array, so it broadcasts by position.
  const av = a.k === 'list' ? scalars(a) : null;
  const bv = b.k === 'list' ? scalars(b) : null;
  return elementwise(av ? arr(av) : a, bv ? arr(bv) : b, f);
}

// ---------------------------------------------------------------- the atoms

function residueStarts(atoms) {
  const n = atoms.nAtoms;
  const starts = [];
  for (let i = 0; i < n; i += 1) {
    if (i === 0 || atoms.chain_id[i] !== atoms.chain_id[i - 1] || atoms.res_id[i] !== atoms.res_id[i - 1]
      || (atoms.ins_code?.[i] ?? '') !== (atoms.ins_code?.[i - 1] ?? '')) starts.push(i);
  }
  return starts;
}

function chainStarts(atoms) {
  const n = atoms.nAtoms;
  const starts = [];
  for (let i = 0; i < n; i += 1) {
    if (i === 0 || atoms.chain_id[i] !== atoms.chain_id[i - 1] || atoms.res_id[i] < atoms.res_id[i - 1]) starts.push(i);
  }
  return starts;
}

function spreadSegments(starts, n, values, reduce) {
  const out = new Array(n);
  for (let s = 0; s < starts.length; s += 1) {
    const a = starts[s];
    const b = s + 1 < starts.length ? starts[s + 1] : n;
    const r = reduce(values.slice(a, b));
    for (let i = a; i < b; i += 1) out[i] = r;
  }
  return out;
}

const REDUCTIONS = {
  all: (xs) => xs.every(truthy),
  any: (xs) => xs.some(truthy),
  max: (xs) => xs.reduce((m, x) => (numeric(x) > m ? numeric(x) : m), -Infinity),
  min: (xs) => xs.reduce((m, x) => (numeric(x) < m ? numeric(x) : m), Infinity),
  sum: (xs) => xs.reduce((s, x) => s + numeric(x), 0),
  mean: (xs) => xs.reduce((s, x) => s + numeric(x), 0) / xs.length,
};

const inSet = (values, set) => values.map((v) => set.has(v));

/**
 * The annotations upstream derives on demand (`atomworks.io.utils.annotator.ANNOTATOR_REGISTRY`),
 * each as atomworks computes it.
 */
function derived(atoms, name) {
  const n = atoms.nAtoms;
  const ct = atoms.chain_type;
  const rn = atoms.res_name;
  const an = atoms.atom_name;
  const typeIn = (types) => ct.map((t) => types.includes(t));
  const named = (list) => inSet(rn, new Set(list));
  const masked = (type) => rn.map((r, i) => r === MASKED && ct[i] === type);
  switch (name) {
    case 'is_protein': return typeIn(PROTEIN_CHAIN_TYPES);
    case 'is_nucleic_acid': return typeIn(NUCLEIC_CHAIN_TYPES);
    case 'is_dna': return ct.map((t) => t === CHAIN_TYPE.DNA);
    case 'is_rna': return ct.map((t) => t === CHAIN_TYPE.RNA);
    case 'is_polymer': return typeIn(POLYMER_CHAIN_TYPES);
    case 'is_ligand': return typeIn(NON_POLYMER_CHAIN_TYPES);
    case 'is_standard_aa': return named(STANDARD_AA);
    case 'is_standard_or_unknown_aa': return named([...STANDARD_AA, UNKNOWN_AA]);
    case 'is_standard_unknown_or_masked_aa': {
      const m = masked(CHAIN_TYPE.POLYPEPTIDE_L);
      return named([...STANDARD_AA, UNKNOWN_AA]).map((v, i) => v || m[i]);
    }
    case 'is_standard_rna': return named(STANDARD_RNA);
    case 'is_standard_dna': return named(STANDARD_DNA);
    case 'is_standard_or_unknown_rna': return named([...STANDARD_RNA, UNKNOWN_RNA]);
    case 'is_standard_or_unknown_dna': return named([...STANDARD_DNA, UNKNOWN_DNA]);
    case 'is_standard_unknown_or_masked_rna': {
      const m = masked(CHAIN_TYPE.RNA);
      return named([...STANDARD_RNA, UNKNOWN_RNA]).map((v, i) => v || m[i]);
    }
    case 'is_standard_unknown_or_masked_dna': {
      const m = masked(CHAIN_TYPE.DNA);
      return named([...STANDARD_DNA, UNKNOWN_DNA]).map((v, i) => v || m[i]);
    }
    case 'is_protein_backbone': {
      const p = derived(atoms, 'is_protein');
      const bb = new Set(PROTEIN_BACKBONE_ATOM_NAMES);
      return an.map((a, i) => p[i] && bb.has(a));
    }
    case 'is_protein_sidechain': {
      const p = derived(atoms, 'is_protein');
      const bb = new Set(PROTEIN_BACKBONE_ATOM_NAMES);
      return an.map((a, i) => p[i] && !bb.has(a));
    }
    case 'is_rna_backbone': {
      const p = derived(atoms, 'is_rna');
      const bb = new Set(RNA_BACKBONE_ATOM_NAMES);
      return an.map((a, i) => p[i] && bb.has(a));
    }
    case 'is_dna_backbone': {
      const p = derived(atoms, 'is_dna');
      const bb = new Set(DNA_BACKBONE_ATOM_NAMES);
      return an.map((a, i) => p[i] && bb.has(a));
    }
    case 'is_nucleic_acid_backbone': {
      const p = derived(atoms, 'is_nucleic_acid');
      const bb = new Set(NUCLEIC_ACID_BACKBONE_ATOM_NAMES);
      return an.map((a, i) => p[i] && bb.has(a));
    }
    case 'is_pyrimidine': return named(PYRIMIDINE_RESIDUES);
    case 'is_purine': return named(PURINE_RESIDUES);
    case 'is_tip_atom': return an.map((a, i) => (TIP_ATOM_NAMES[rn[i]] ?? []).includes(a));
    // 🔴 NEVER TRUE, BECAUSE UPSTREAM'S IS NEVER TRUE: atomworks computes it as
    // `np.isin(res_name, METAL_ELEMENTS)` and METAL_ELEMENTS is a frozenset, which numpy takes as
    // ONE object rather than a collection - so a calcium named `CA` is not a metal there, and a
    // selection `... or is_metal` conditions nothing it would add. Measured on 1PRW's four Ca ions.
    case 'is_metal': return new Array(n).fill(false);
    case 'is_carbohydrate': return named(CARBOHYDRATE_NAMES);
    case 'is_res_start': {
      const out = new Array(n).fill(false);
      for (const s of residueStarts(atoms)) out[s] = true;
      return out;
    }
    case 'is_chain_start': {
      const out = new Array(n).fill(false);
      for (const s of chainStarts(atoms)) out[s] = true;
      return out;
    }
    case 'is_token_start': {
      const out = new Array(n).fill(false);
      for (const s of getTokenStarts(atoms, false)) out[s] = true;
      return out;
    }
    case 'token_id': {
      const starts = getTokenStarts(atoms, false);
      const out = new Array(n);
      starts.forEach((a, t) => {
        const b = t + 1 < starts.length ? starts[t + 1] : n;
        for (let i = a; i < b; i += 1) out[i] = t;
      });
      return out;
    }
    case 'res_min_occupancy': return spreadSegments(residueStarts(atoms), n, atoms.occupancy, REDUCTIONS.min);
    case 'token_min_occupancy': return spreadSegments(getTokenStarts(atoms, false), n, atoms.occupancy, REDUCTIONS.min);
    case 'res_has_tip_atom': return spreadSegments(residueStarts(atoms), n, derived(atoms, 'is_tip_atom'), REDUCTIONS.any);
    case 'within_chain_res_idx': {
      // `get_within_group_res_idx`: a residue's ordinal within its chain (by `chain_iid` where
      // there is one), counted over residue starts.
      const group = atoms.chain_iid ?? atoms.chain_id;
      const out = new Array(n);
      const seen = new Map();
      const starts = residueStarts(atoms);
      for (let s = 0; s < starts.length; s += 1) {
        const a = starts[s];
        const b = s + 1 < starts.length ? starts[s + 1] : n;
        const g = group[a];
        const k = seen.get(g) ?? 0;
        seen.set(g, k + 1);
        for (let i = a; i < b; i += 1) out[i] = k;
      }
      return out;
    }
    default: return undefined;
  }
}

const HIDDEN = new Set(['is_aromatic', 'is_leaving_atom', 'bonds', 'coord', 'annotations', 'nAtoms']);

function attribute(atoms, name) {
  if (name === 'x' || name === 'y' || name === 'z') {
    const k = { x: 0, y: 1, z: 2 }[name];
    return atoms.coord.map((c) => c[k]);
  }
  // The namespace is upstream's: atomworks parses no aromatic or leaving-atom flag onto an atom,
  // names the alternate location `altloc_id` as well, and has no `b_factor` where the file had
  // no B column (cif.js records that as `noBFactorColumn`).
  if (HIDDEN.has(name)) return undefined;
  if (name === 'altloc_id') return atoms.label_alt_id;
  if (name === 'b_factor' && atoms.noBFactorColumn) return undefined;
  if (name !== 'coord' && name !== 'annotations' && Array.isArray(atoms[name])) return atoms[name];
  if (atoms.annotations && Array.isArray(atoms.annotations[name])) return atoms.annotations[name];
  return derived(atoms, name);
}

// ---------------------------------------------------------------- `sa_mask(...)`

/** Each condition's presence mask, as its own `mask_from_annotation` defines it upstream. */
const SA_MASK = {
  condition_coordinate_1_atom: { def: null, mask: (v) => Array.isArray(v) && v.some(Number.isFinite) },
  condition_InternalCoordinate_1_atom: { def: null, mask: (v) => Array.isArray(v) && v.some(Number.isFinite) },
  condition_ChainCentroid_1_atom: { def: null, mask: (v) => Array.isArray(v) && v.some(Number.isFinite) },
  condition_CoordinateGroup_1_atom: { def: -1, mask: () => true },
  'condition_index-group_1_token': { def: 0, mask: () => true },
  condition_sequence_1_residue: { def: MASKED, mask: (v) => v !== MASKED },
  'condition_cath-label_1_residue': { def: 'x.x.x.x', mask: (v) => v !== 'x.x.x.x' },
  'annotation_src-component_1_atom': { def: '.', mask: (v) => v !== '.' },
  annotation_expsegmin_1_residue: { def: -1, mask: (v) => v !== -1 },
  annotation_expsegmax_1_residue: { def: -1, mask: (v) => v !== -1 },
};

function saMask(atoms, name, resolve) {
  const spec = resolve(name);
  if (spec.dtype === 'twobody') {
    throw new Error(`sa_mask('${name}') requires a one-body standard annotation, but \`${spec.fullName}\` is 2-body and has no per-atom mask.`);
  }
  const n = atoms.nAtoms;
  const values = atoms.annotations?.[spec.fullName];
  const rule = SA_MASK[spec.fullName]
    ?? (spec.fullName.startsWith('condition_sequence-probability-')
      ? { def: NaN, mask: (v) => !Number.isNaN(v) }
      : spec.dtype === 'bool' ? { def: spec.default, mask: (v) => Boolean(v) }
        : { def: spec.default, mask: (v) => v !== spec.default });
  return Array.from({ length: n }, (_, i) => rule.mask(values ? values[i] : rule.def));
}

// ---------------------------------------------------------------- `sel('...')`

function parseField(raw, isResId) {
  const v = raw.trim();
  if (v === '*' || v === '') return ['*'];
  if (v.includes('[') || v.includes(']')) {
    if (!(v.startsWith('[') && v.endsWith(']'))) throw new Error(`Malformed bracket list in selection field: '${v}'`);
    const items = v.slice(1, -1).split(',').map((s) => s.trim()).filter(Boolean);
    if (!items.length) throw new Error(`Empty bracket list in selection field: '${v}'`);
    const out = [];
    for (const item of items) {
      if (item === '*') { out.push('*'); continue; }
      if (isResId) {
        const m = /^(-?\d+)-(-?\d+)$/.exec(item);
        if (m) {
          const s = parseInt(m[1], 10);
          const e = parseInt(m[2], 10);
          const step = s <= e ? 1 : -1;
          for (let k = s; step > 0 ? k <= e : k >= e; k += step) out.push(k);
          continue;
        }
      }
      out.push(scalarField(item, isResId));
    }
    return out;
  }
  return [scalarField(v, isResId)];
}

function scalarField(value, isResId) {
  if (!isResId) return value;
  // Python's `int()`: optional sign, digits (with underscores between them), surrounding space.
  if (!/^\s*[+-]?\d+(_\d+)*\s*$/.test(value)) {
    throw new Error(`Invalid res_id '${value}': ranges must be wrapped in brackets, e.g. '[5-10]'.`);
  }
  return parseInt(value.replace(/_/g, ''), 10);
}

/** `AtomSelectionStack.from_query(...).get_mask(raise_on_empty=False)`. */
export function pathMask(atoms, query) {
  const n = atoms.nAtoms;
  const out = new Array(n).fill(false);
  const tokens = [];
  let buf = '';
  let depth = 0;
  for (const ch of query) {
    if (ch === '[') depth += 1;
    else if (ch === ']') depth -= 1;
    if (ch === ',' && depth === 0) { tokens.push(buf); buf = ''; } else buf += ch;
  }
  tokens.push(buf);
  for (const token of tokens.map((t) => t.trim()).filter(Boolean)) {
    const parts = token.split('/');
    while (parts.length < 5) parts.push('*');
    const fields = [parseField(parts[0], false), parseField(parts[1], false), parseField(parts[2], true),
      parseField(parts[3], false), parseField(parts[4], false)];
    for (const c of fields[0]) for (const r of fields[1]) for (const id of fields[2]) {
      for (const a of fields[3]) for (const t of fields[4]) {
        for (let i = 0; i < n; i += 1) {
          if (out[i]) continue;
          if (c && c !== '*' && atoms.chain_id[i] !== c) continue;
          if (r && r !== '*' && atoms.res_name[i] !== r) continue;
          if (id !== '*' && atoms.res_id[i] !== id) continue;
          if (a && a !== '*' && atoms.atom_name[i] !== a) continue;
          // `transformation_id` is "1" where nothing set one - the asymmetric unit.
          if (t && t !== '*' && String(atoms.transformation_id?.[i] ?? '1') !== String(t)) continue;
          out[i] = true;
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- the evaluator

function evaluate(node, atoms, resolve) {
  const n = atoms.nAtoms;
  const ev = (x) => evaluate(x, atoms, resolve);
  switch (node.k) {
    case 'const': return sc(node.v);
    case 'list': return { k: 'list', v: node.elts.map(ev) };
    case 'tuple': return { k: 'list', v: node.elts.map(ev), tuple: true };
    case 'name': {
      const v = attribute(atoms, node.id);
      if (v === undefined) throw new Error(`Name '${node.id}' is not defined`);
      return arr(v);
    }
    case 'compare': {
      let left = ev(node.left);
      let result = null;
      node.ops.forEach((op, j) => {
        const right = ev(node.comparators[j]);
        let r;
        if (op === 'in' || op === 'not in') {
          if (right.k === 'sc') throw new TypeError(`Argument of type '${typeof right.v}' is not iterable`);
          const pool = scalars(right);
          // `np.isin` makes the pool ONE array first, and numpy types a list holding any string as
          // strings - so `res_id in [5, 6, '7']` matches nothing. A scalar on the left is Python's
          // own `in`, which does not coerce.
          const asArray = pool.some((y) => typeof y === 'string') ? pool.map(String) : pool;
          r = left.k === 'arr' ? arr(left.v.map((x) => memberOf(x, asArray) !== (op === 'not in')))
            : sc(memberOf(left.k === 'sc' ? left.v : left, pool) !== (op === 'not in'));
        } else if (op === 'is' || op === 'is not') {
          throw new Error(`Unsupported comparison operator: ${op}`);
        } else {
          r = elementwise(left, right, (a, b) => cmp(op, a, b));
        }
        result = result === null ? r : elementwise(result, r, (a, b) => truthy(a) && truthy(b));
        left = right;
      });
      return result;
    }
    case 'bool': {
      const values = node.values.map((v) => ensureBool(ev(v), n));
      return arr(values.reduce((acc, v) => acc.map((x, i) => (node.op === 'and' ? x && v[i] : x || v[i]))));
    }
    case 'binop': {
      if (node.op !== '&' && node.op !== '|') throw new Error(`Unsupported binary operation: ${node.op}`);
      const a = ensureBool(ev(node.left), n);
      const b = ensureBool(ev(node.right), n);
      return arr(a.map((x, i) => (node.op === '&' ? x && b[i] : x || b[i])));
    }
    case 'unary': {
      const v = ev(node.operand);
      if (node.op === 'not' || node.op === '~') return arr(ensureBool(v, n).map((x) => !x));
      const f = node.op === '-' ? (x) => -numeric(x) : (x) => numeric(x);
      const check = (x) => {
        if (typeof x === 'string') throw new TypeError(`bad operand type for unary ${node.op}: 'str'`);
        return f(x);
      };
      return v.k === 'sc' ? sc(check(v.v)) : arr(scalars(v).map(check));
    }
    case 'call': {
      if (node.func.k !== 'name') throw new Error('Complex function calls not supported');
      const fn = node.func.id;
      const known = ['apply_and_spread', 'has_nan_coord', 'has_bonds', 'sa_mask', 'sel'];
      if (!known.includes(fn)) throw new Error(`Function '${fn}' is not defined`);
      if (fn === 'apply_and_spread') {
        if (node.args.length !== 3 || node.keywords) {
          throw new Error("Function 'apply_and_spread' expects level, data, and reduction arguments");
        }
        const level = ev(node.args[0]);
        const data = ev(node.args[1]);
        const reduction = ev(node.args[2]);
        if (level.k !== 'sc' || typeof level.v !== 'string') throw new TypeError('apply_and_spread() level must be a string');
        if (reduction.k !== 'sc' || typeof reduction.v !== 'string') throw new TypeError('apply_and_spread() reduction must be a string');
        if (!(reduction.v in REDUCTIONS)) {
          throw new Error(`Unknown apply_and_spread() reduction '${reduction.v}'. Expected one of ${JSON.stringify(Object.keys(REDUCTIONS).sort())}.`);
        }
        if (data.k === 'sc' || scalars(data).length !== n) {
          throw new Error(`apply_and_spread() data must have one value per atom; got ${data.k === 'sc' ? 'a scalar' : scalars(data).length} for ${n} atoms.`);
        }
        const values = scalars(data);
        const red = REDUCTIONS[reduction.v];
        let spread;
        if (level.v === 'atom') throw new Error('Apply and spread does not make sense for atom level.');
        else if (level.v === 'residue') spread = spreadSegments(residueStarts(atoms), n, values, red);
        else if (level.v === 'token') spread = spreadSegments(getTokenStarts(atoms, false), n, values, red);
        else if (level.v === 'chain' || level.v === 'molecule' || level.v === 'system') {
          const key = level.v === 'chain' ? (atoms.chain_iid ?? atoms.chain_id)
            : level.v === 'molecule' ? atoms.molecule_id : new Array(n).fill(1);
          const groups = new Map();
          key.forEach((g, i) => { if (!groups.has(g)) groups.set(g, []); groups.get(g).push(i); });
          spread = new Array(n);
          for (const idx of groups.values()) {
            const r = red(idx.map((i) => values[i]));
            for (const i of idx) spread[i] = r;
          }
        } else {
          throw new Error(`'${level.v}' is not a valid Level`);
        }
        return arr(spread);
      }
      if (fn === 'sel' || fn === 'sa_mask') {
        if (node.args.length !== 1 || node.keywords) throw new Error(`Function '${fn}' expects a single argument`);
        const a = ev(node.args[0]);
        if (a.k !== 'sc' || typeof a.v !== 'string') throw new TypeError(`Function '${fn}' argument must evaluate to a string`);
        return arr(fn === 'sel' ? pathMask(atoms, a.v) : saMask(atoms, a.v, resolve));
      }
      if (node.args.length || node.keywords) throw new Error(`Function '${fn}' does not accept arguments`);
      if (fn === 'has_nan_coord') return arr(atoms.coord.map((c) => c.some((x) => Number.isNaN(x))));
      const bonded = new Array(n).fill(false);
      for (const [u, v] of atoms.bonds ?? []) { bonded[u] = true; bonded[v] = true; }
      return arr(bonded);
    }
    default:
      throw new Error(`Unsupported node type: ${node.k}`);
  }
}

/**
 * `resolve_mask(atom_array, query)`: the query string's boolean mask over `atoms`.
 *
 * @param {object} atoms
 * @param {string} query
 * @param {(name: string) => {fullName: string, dtype: string, default: any}} resolve the condition
 *   registry, for `sa_mask(...)`
 * @returns {boolean[]}
 */
export function evalQuery(atoms, query, resolve) {
  let tree;
  try {
    tree = parse(query);
  } catch (e) {
    if (e instanceof QuerySyntaxError) throw new Error(`invalid selection '${query}': ${e.message}`);
    throw e;
  }
  return ensureBool(evaluate(tree, atoms, resolve), atoms.nAtoms);
}

/**
 * The source order of an object's keys, beside it. JavaScript enumerates an integer-like key
 * ("1", "2") before every other one and in numeric order whatever order it was written in, where
 * upstream's dict keeps the file's; a condition's arms are applied in that order and the last one
 * wins an atom two of them select, so `C_CRDGRP: {"2": ..., "1": ...}` over overlapping
 * selections came out the other way round here. Not enumerable, so it never reaches a JSON.
 */
export const KEY_ORDER = Symbol.for('rfd4p.keyOrder');

/** An object's entries in the order its source wrote them, where that is known. */
export function orderedEntries(obj) {
  if (obj === null || typeof obj !== 'object') return [];
  const order = obj[KEY_ORDER];
  return order ? order.filter((k) => Object.hasOwn(obj, k)).map((k) => [k, obj[k]]) : Object.entries(obj);
}

/** Record `keys` as `obj`'s source order. */
export function withKeyOrder(obj, keys) {
  Object.defineProperty(obj, KEY_ORDER, { value: [...keys], enumerable: false, configurable: true, writable: true });
  return obj;
}

/** `JSON.parse`, every object carrying its keys' source order (`orderedEntries`). */
export function parseJsonOrdered(text) {
  let at = 0;
  const ws = () => { while (at < text.length && ' \t\n\r'.includes(text[at])) at += 1; };
  const fail = () => { throw new SyntaxError(`Unexpected token in JSON at position ${at}`); };
  const string = () => {
    const start = at;
    at += 1;
    while (at < text.length && text[at] !== '"') at += text[at] === '\\' ? 2 : 1;
    if (text[at] !== '"') fail();
    at += 1;
    return JSON.parse(text.slice(start, at));
  };
  const value = () => {
    ws();
    const c = text[at];
    if (c === '{') {
      at += 1;
      const obj = {};
      const keys = [];
      ws();
      if (text[at] === '}') { at += 1; return withKeyOrder(obj, keys); }
      for (;;) {
        ws();
        if (text[at] !== '"') fail();
        const key = string();
        ws();
        if (text[at] !== ':') fail();
        at += 1;
        const v = value();
        if (!Object.hasOwn(obj, key)) keys.push(key);
        obj[key] = v;
        ws();
        if (text[at] === ',') { at += 1; continue; }
        if (text[at] === '}') { at += 1; return withKeyOrder(obj, keys); }
        fail();
      }
    }
    if (c === '[') {
      at += 1;
      const arr = [];
      ws();
      if (text[at] === ']') { at += 1; return arr; }
      for (;;) {
        arr.push(value());
        ws();
        if (text[at] === ',') { at += 1; continue; }
        if (text[at] === ']') { at += 1; return arr; }
        fail();
      }
    }
    if (c === '"') return string();
    const m = /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/.exec(text.slice(at));
    if (!m) fail();
    at += m[0].length;
    return JSON.parse(m[0]);
  };
  const out = value();
  ws();
  if (at !== text.length) fail();
  return out;
}

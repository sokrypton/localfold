/**
 * A BioEmu2 `--potentials` file: the YAML subset it is written in, read without a YAML library.
 *
 *     potentials:
 *       - _target_: bioemu2.denoising.steering.potentials.BondLengthPotential
 *         sigma_tolerance: 10.0
 *         parameters:
 *           guidance_weight: 0.05
 *           buffer:
 *             _target_: bioemu2.denoising.steering.potentials.PiecewiseStepFunction
 *             thresholds: [0.4]
 *             values: [0.125, 0.0]
 *
 * Block mappings, block sequences of mappings (`- key: value`), flow lists of
 * scalars (`[0.4, 0.6]`), flat flow mappings (`{guidance_weight: 0.3}`), scalars (numbers, true/false, null/~, plain and
 * quoted strings) and `#` comments - every construct the vendor's own preset
 * files and its README use. Anything else (anchors, multi-line strings, nested
 * flow collections, tabs) is REFUSED by name rather than half-read, because a
 * potential whose parameter silently became a string is a steering run that
 * did not do what its file says. potentialSpecs() in potentials.js then reads
 * the result exactly as af3-any-model's load_potentials does.
 */

function scalar(text, where) {
  const t = text.trim();
  if (t === "" || t === "~" || t === "null" || t === "Null" || t === "NULL") return null;
  if (t === "true" || t === "True" || t === "TRUE") return true;
  if (t === "false" || t === "False" || t === "FALSE") return false;
  if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t)) return Number(t);
  if (/^[-+]?\.(inf|Inf|INF)$/.test(t)) return t.startsWith("-") ? -Infinity : Infinity;
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) return t.slice(1, -1);
  if (t.startsWith("[")) {
    if (!t.endsWith("]")) throw new Error(`${where}: an unclosed flow list`);
    const inner = t.slice(1, -1).trim();
    return inner === "" ? [] : inner.split(",").map((v) => scalar(v, where));
  }
  if (t.startsWith("{")) {
    // a flat flow mapping of scalars, `{guidance_weight: 0.3, buffer: 0.2}`
    if (!t.endsWith("}")) throw new Error(`${where}: an unclosed flow mapping`);
    const inner = t.slice(1, -1).trim();
    if (/[{[]/.test(inner)) throw new Error(`${where}: a nested flow collection; write it as a block`);
    const out = {};
    if (inner === "") return out;
    for (const part of inner.split(",")) {
      const colon = part.indexOf(":");
      if (colon < 0) throw new Error(`${where}: "${part.trim()}" in a flow mapping is not "key: value"`);
      out[part.slice(0, colon).trim().replace(/^['"]|['"]$/g, "")] = scalar(part.slice(colon + 1), where);
    }
    return out;
  }
  if (/^[&*!|>]/.test(t)) throw new Error(`${where}: "${t}" is a YAML construct a potentials file does not use`);
  return t;
}

/** Strip a trailing comment that is not inside quotes. */
function uncomment(line) {
  let quote = null;
  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];
    if (quote) { if (c === quote) quote = null; continue; }
    if (c === '"' || c === "'") quote = c;
    else if (c === "#" && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
  }
  return line;
}

/** Parse the YAML subset above into plain objects and arrays. */
export function parseSimpleYaml(text) {
  const lines = [];
  text.split(/\r?\n/).forEach((raw, number) => {
    if (raw.includes("\t")) throw new Error(`line ${number + 1}: a tab; YAML indents with spaces`);
    const line = uncomment(raw).replace(/\s+$/, "");
    if (line.trim() === "" || line.trim() === "---") return;
    lines.push({ indent: line.length - line.trimStart().length, text: line.trim(), number: number + 1 });
  });
  let at = 0;
  const block = (indent) => {
    if (at >= lines.length) return null;
    const first = lines[at];
    if (first.indent < indent) return null;
    if (first.text.startsWith("- ") || first.text === "-") return sequence(first.indent);
    return mapping(first.indent);
  };
  const keyValue = (line, text) => {
    const colon = text.search(/:(\s|$)/);
    if (colon < 0) throw new Error(`line ${line.number}: expected "key: value", got "${text}"`);
    let key = text.slice(0, colon).trim();
    if ((key.startsWith('"') && key.endsWith('"')) || (key.startsWith("'") && key.endsWith("'"))) key = key.slice(1, -1);
    return [key, text.slice(colon + 1).trim()];
  };
  function mapping(indent) {
    const out = {};
    while (at < lines.length && lines[at].indent === indent && !lines[at].text.startsWith("- ")) {
      const line = lines[at];
      const [key, rest] = keyValue(line, line.text);
      at += 1;
      if (key in out) throw new Error(`line ${line.number}: ${key} given twice`);
      if (rest === "") {
        // a block below it - or a sequence at the KEY'S OWN indent (`potentials:` then `- _target_: ...`), which YAML
        // allows for a list and the vendor's files use
        const next = lines[at];
        const child = next && next.indent > indent ? block(next.indent)
          : next && next.indent === indent && (next.text.startsWith("- ") || next.text === "-") ? sequence(indent) : null;
        out[key] = child;
      } else {
        out[key] = scalar(rest, `line ${line.number}`);
      }
    }
    if (at < lines.length && lines[at].indent > indent) throw new Error(`line ${lines[at].number}: unexpected indentation`);
    return out;
  }
  function sequence(indent) {
    const out = [];
    while (at < lines.length && lines[at].indent === indent && (lines[at].text.startsWith("- ") || lines[at].text === "-")) {
      const line = lines[at];
      const rest = line.text === "-" ? "" : line.text.slice(2).trim();
      if (rest === "") {
        at += 1;
        out.push(at < lines.length && lines[at].indent > indent ? block(lines[at].indent) : null);
      } else if (/:(\s|$)/.test(rest) && !rest.startsWith("[") && !/^['"]/.test(rest)) {
        // `- key: value` opens a mapping whose further keys sit two columns in
        const childIndent = indent + 2 + (rest.length - rest.trimStart().length);
        lines[at] = { ...line, indent: childIndent, text: rest };
        out.push(mapping(childIndent));
      } else {
        at += 1;
        out.push(scalar(rest, `line ${line.number}`));
      }
    }
    return out;
  }
  const result = block(0);
  if (at < lines.length) throw new Error(`line ${lines[at].number}: unexpected "${lines[at].text}"`);
  return result ?? {};
}

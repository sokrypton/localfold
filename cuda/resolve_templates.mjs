// The page's template rows, resolved as the page resolves them, for a native fold:
//
//   node cuda/resolve_templates.mjs <request.json> <out dir>
//
// <request.json> is what the page sends a remote fold ({entities, ...}). Its rows go through the page's own
// expandEntities (web/entities.js) - one template per chain COPY, numbered over every polymer - and each
// structure is fetched by the page's own fetchStructure (web/template-source.js: the RCSB's PDB file
// first, the AlphaFold DB by accession) or taken as the uploaded text. Each is written to <out dir> and
// one JSON line printed: [{chain, kind, file, chainId, source}] in the page's order. A "from the MSA
// search" row has no file - its structure is the search's best hit, which the exporters fetch
// (--template-search-chains).
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { expandEntities, templateKind } from "../web/entities.js";
import { fetchStructure } from "../web/template-source.js";

const [requestPath, out] = process.argv.slice(2);
if (!out) { console.error("usage: resolve_templates.mjs <request.json> <out dir>"); process.exit(1); }
const request = JSON.parse(readFileSync(requestPath, "utf8"));
const { templates } = expandEntities(request.entities ?? []);
mkdirSync(out, { recursive: true });
const resolved = [];
for (const [k, template] of (templates ?? []).entries()) {
  const kind = templateKind(template);
  const source = (template.source ?? "").trim();
  if (kind === "search") { resolved.push({ chain: template.chain, kind }); continue; }
  let text, chainId, named;
  if (kind === "upload") {
    text = template.text ?? "";
    chainId = source === "" ? undefined : source;
    named = template.filename ?? "the uploaded structure";
  } else {
    if (kind === "none" || source === "") continue;
    const structure = await fetchStructure(source, { kind });
    text = structure.text; chainId = structure.chain; named = source;
  }
  const cif = /^\s*(data_|#|loop_|_)/m.test(text.slice(0, 4096)) && text.includes("_atom_site.");
  const file = `${out}/template-${k}.${cif ? "cif" : "pdb"}`;
  writeFileSync(file, text);
  resolved.push({ chain: template.chain, kind, file, chainId: chainId ?? null, source: named });
}
console.log(JSON.stringify(resolved));

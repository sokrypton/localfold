import { test } from "node:test";
import assert from "node:assert/strict";
import {
  chainTmInfo, interfaceTmTerm, tmPerBinFor, tmScoreD0, tmTermFromLogits, tmTermsFromLogits,
} from "../shared/heads/tm-score.js";

// `tmTermsFromLogits` takes each pair's softmax once for both terms; it must be bit-identical to the two separately,
// including masked tokens (interface 0) and more than one chain (a per-pair table).
test("tmTermsFromLogits matches tmTermFromLogits and interfaceTmTerm bit for bit", () => {
  const tokens = 13;
  const bins = 64;
  let seed = 5;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const logits = Float32Array.from({ length: tokens * tokens * bins }, () => (random() - 0.5) * 12);
  const centres = Array.from({ length: bins }, (_, k) => 0.25 + 0.5 * k);
  const asymId = Int32Array.from({ length: tokens }, (_, i) => (i < 6 ? 1 : i < 11 ? 2 : 3));
  const seqMask = Float32Array.from({ length: tokens }, (_, i) => (i === 4 || i === 12 ? 0 : 1));
  const perBin = tmPerBinFor(centres, tmScoreD0(tokens));
  const info = chainTmInfo(asymId, seqMask, centres);
  const fused = tmTermsFromLogits(logits, tokens, perBin, info);
  const adjusted = tmTermFromLogits(logits, tokens, perBin);
  const chained = interfaceTmTerm(logits, tokens, info);
  assert.equal(fused.adjusted.length, adjusted.length);
  for (let i = 0; i < adjusted.length; i += 1) assert.equal(fused.adjusted[i], adjusted[i], `adjusted ${i}`);
  for (let i = 0; i < chained.length; i += 1) assert.equal(fused.interface[i], chained[i], `interface ${i}`);
});

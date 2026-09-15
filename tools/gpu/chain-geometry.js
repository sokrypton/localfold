/**
 * The chain-geometry gate for the COMMAND-LINE tools.
 *
 * 🔴 THE RULE ITSELF MOVED TO src/af3/chain-geometry.js SO THE PAGE COULD USE
 * IT. It lived here, under `tools/`, which meant the SITE ran no geometry check
 * at all - every CLI fold gated on this and the one path a visitor takes did
 * not. See the note there; this file is now the terminal's half: the throw, and
 * the "--allow-broken-geometry" advice that only makes sense in a shell.
 */
export { CHAIN_GEOMETRY_BANDS, chainGeometryVerdict, chainGeometryOf }
  from "../../src/af3/chain-geometry.js";
import { chainGeometryVerdict } from "../../src/af3/chain-geometry.js";

/** The same verdict, thrown. `allow` is the tool's --allow-broken-geometry. */
export function assertChainGeometry(geometry, options = {}) {
  const verdict = chainGeometryVerdict(geometry,
    { hint: "Pass --allow-broken-geometry to report anyway.", ...options });
  if (!verdict.ok && options.allow !== true) throw new Error(verdict.reason);
  return verdict;
}

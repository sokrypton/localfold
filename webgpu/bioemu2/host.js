/**
 * BioEmu2's host arithmetic, which now lives under `cpu/`.
 *
 * 🔴 A RE-EXPORT RATHER THAN A COPY, so this backend's behaviour cannot change:
 * every caller here gets the same function objects it had before the move. The
 * reason for the move is in `cpu/bioemu2/host.js`'s own header - the arithmetic
 * is per atom and per atom-window pair, has no GPU in it, and a second backend
 * must be able to read it without importing this one.
 */
export {
  addTokenPairToWindows, atomFeatures, atomPairTail, dampedHarmonic, linearRows,
  lnRows, normalizeWindowPair, sInputs, windowBias,
} from "../../cpu/bioemu2/host.js";

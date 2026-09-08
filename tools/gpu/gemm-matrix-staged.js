/**
 * Moved to src/runtime/matrix-linear.js, because src/ now uses it and src/ may
 * not import from tools/. This re-export keeps the benches' import path.
 */
export {
  createStagedMatrixShader, stagedMatrixFits, stagedMatrixStorage,
} from "../../src/runtime/matrix-linear.js";

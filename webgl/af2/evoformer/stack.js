/**
 * The Evoformer stacks: a loop over blocks, each reading what the last wrote.
 *
 * `webgpu/af2/evoformer/stack.js` is the shape, and there is almost nothing to
 * it beyond the loop - which is the point. A stack is where a port stops being
 * a set of kernels and starts being a model, and the one thing it can get wrong
 * is feeding a block the WRONG representation: the previous block's input
 * instead of its output, or the main stack's pair into the extra stack's. That
 * is invisible to any block-level gate, because a block handed the right input
 * returns the right answer however it got there.
 *
 * 🔴 SO THE STACK GATE RUNS TWO BLOCKS, NOT ONE. With one block a stack and a
 * block are the same thing and the loop is untested; with two, the second
 * block's output depends on the first's, and a stack that re-ran the first
 * block's input would agree with a one-block reference and differ here.
 *
 * 🔴 AND THE EXTRA STACK RETURNS BOTH REPRESENTATIONS BUT AF2 KEEPS ONLY THE
 * PAIR. Its MSA is thousands of alignment rows that the main stack never sees -
 * the extra stack exists to get their information into the pair track and is
 * then discarded. Both are returned here because a gate needs to compare both;
 * which one a fold keeps is the caller's business.
 */
import { evoformerBlock, extraMsaBlock } from "./block.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {Omit<Parameters<typeof evoformerBlock>[1], "weights">
 *   & {blockWeights: Parameters<typeof evoformerBlock>[1]["weights"][],
 *      onBlock?: (block: number, blocks: number) => void}} call
 * @returns {{msa: Tensor, pair: Tensor, blocks: number}}
 */
export function evoformerStack(execution, call) {
  return runStack(execution, call, evoformerBlock, "stack");
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {Omit<Parameters<typeof extraMsaBlock>[1], "weights">
 *   & {blockWeights: Parameters<typeof extraMsaBlock>[1]["weights"][],
 *      onBlock?: (block: number, blocks: number) => void}} call
 * @returns {{msa: Tensor, pair: Tensor, blocks: number}}
 */
export function extraMsaStack(execution, call) {
  return runStack(execution, call, extraMsaBlock, "extra-stack");
}

function runStack(execution, call, block, label) {
  const { blockWeights } = call;
  if (!Array.isArray(blockWeights) || blockWeights.length === 0) {
    throw new RangeError(`${label}: needs at least one block's weights`);
  }
  let msa = call.msa;
  let pair = call.pair;
  let own = false;
  for (const [index, weights] of blockWeights.entries()) {
    const next = block(execution, { ...call, msa, pair, weights });
    // 🔴 AFTER THE BLOCK, NOT BEFORE IT, AND SYNCHRONOUS. A caller counting
    // progress wants to hear that a block is DONE; the WebGPU stack's own note
    // says the same thing about its `onBlockDone` ("when the device finishes a
    // block, not when one is queued"). Here a draw is issued as it is encoded,
    // so returning from `block` is as close to done as this API gets - the GPU
    // may still be behind, and a count of blocks ISSUED is what this reports.
    call.onBlock?.(index, blockWeights.length);
    // The caller's two tensors are never released; everything this loop
    // produced is, as soon as the next block has replaced it.
    if (own) {
      execution.release(msa);
      execution.release(pair);
    }
    msa = next.msa;
    pair = next.pair;
    own = true;
  }
  msa.label = `${label}.msa`;
  pair.label = `${label}.pair`;
  return { msa, pair, blocks: blockWeights.length };
}

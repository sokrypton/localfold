/**
 * ESMFold2's folding trunk in WebGL2: 24 blocks of the pair track alone.
 *
 *     pair += triangleMultiplication(pair, "outgoing")
 *     pair += triangleMultiplication(pair, "incoming")
 *     pair += transition(pair)
 *
 * 🔴 IT IS AF3's PAIRFORMER BLOCK WITH THE TWO GRID ATTENTIONS AND THE SINGLE
 * TRACK REMOVED, and that is composed here rather than borrowed from
 * `af3/trunk/pairformer-block.js` for the reason the CPU reference gives:
 * `pairformerBlock` "always runs singleAttention and a single transition;
 * feeding it a synthesised zero single would check arithmetic ESMFold2 never
 * does, and zeroed grid attention weights would make it compute 24 blocks of
 * zeros to add". The three kernels it DOES use are the AF3 ones unchanged, which
 * is the whole saving.
 *
 * 🔴 AND THE LOOP AROUND IT RUNS `num_loops + 1` TIMES, which lives in
 * `pair-features.js` with z_init - see the note there. This file is one pass of
 * the trunk.
 */
import { add } from "../runtime/elementwise.js";
import { triangleMultiplication } from "../af3/trunk/triangle.js";
import { transition } from "../af3/trunk/transition.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, mask: Tensor, tokens: number, channels: number,
 *   blocks: any[], divideByLength?: boolean, epsilon?: number,
 *   onBlock?: (index: number, tensor: Tensor) => void}} call
 * @returns {Tensor} `tokens^2 * channels`, a new tensor; the caller's is untouched
 */
export function foldingTrunk(execution, call) {
  const {
    pair: input, mask, tokens, channels, blocks,
    divideByLength = false, epsilon = 1e-5,
  } = call;
  const pairs = tokens * tokens;
  let pair = input;
  let own = false;
  /** Thread an update into the running pair and release what it replaces. */
  const residual = (update) => {
    const next = add(execution, pair, update);
    execution.release(update);
    if (own) execution.release(pair);
    pair = next;
    own = true;
  };
  for (const [index, block] of blocks.entries()) {
    for (const direction of /** @type {const} */ (["outgoing", "incoming"])) {
      residual(triangleMultiplication(execution, {
        pair, mask, length: tokens, channels, direction,
        weights: direction === "outgoing"
          ? block.triangleMultiplicationOutgoing : block.triangleMultiplicationIncoming,
        divideByLength, epsilon,
      }));
    }
    residual(transition(execution, {
      x: pair, weights: block.pairTransition, rows: pairs, channels, epsilon,
    }));
    call.onBlock?.(index, pair);
  }
  pair.label = "ef2.trunk.pair";
  return pair;
}

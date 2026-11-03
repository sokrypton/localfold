/**
 * How much of a stack may wait in the queue, as a time rather than a number of blocks.
 *
 * Sixteen blocks in the queue is a third of a second at 150 tokens and forty seconds at 1400, all of it committed: Stop
 * waits for it, and so does closing the tab. So the window is a time, about a second of queued work, up to the sixteen a
 * small block was measured at. See the note in webgpu/af3/trunk/pairformer-block.js.
 *
 * It slides. Waiting for the queue to empty leaves the card idle while the next block is encoded, so each new block
 * waits only for the oldest. The card goes idle only where a block alone is worth the second and the window is one, and
 * encoding is then a small part of it.
 *
 * And a block too long for one submission goes in pieces, as many to a submission as fill SUBMISSION_MS, where one piece
 * each is forty submissions a block. A piece is a mean, and the kernels that dominate a block are far past it: at 1000
 * tokens of IntelliFold-2 a run of grid chunks and transition chunks counted as cheap filled 900 ms against a target of 50.
 * So a piece known to be heavy ends its submission whatever the count.
 */
const QUEUED_WORK_MS = 1000;
const MOST_BLOCKS = 16;
const SUBMISSION_MS = 50;

export class QueuePacer {
  /**
   * @param {GPUDevice} device
   * @param {{fixed?: number, start?: number}} [options] `fixed`: a window to keep, for a bench or a device's tuning.
   *   `start`: the blocks to queue before the first measurement.
   */
  constructor(device, { fixed, start } = {}) {
    this.device = device;
    this.fixed = fixed;
    this.window = fixed ?? start ?? 1;
    this.piecesEach = 1;
    this.pieces = 0;
    this.unsent = 0;
    this.queued = [];
    this.finishedAt = performance.now();
  }

  /** A piece of a block is encoded: whether to send what has built up. One at a time until a block says what a piece takes. */
  pieceDue(heavy = false) {
    this.pieces += 1;
    this.unsent += 1;
    if (!heavy && this.unsent < this.piecesEach) return false;
    this.unsent = 0;
    return true;
  }

  /** One block is in the queue. Resolves when there is room for the next; `drain` empties the queue whatever its length. */
  async blockSubmitted(drain = false) {
    const finished = this.device.queue.onSubmittedWorkDone().then(() => performance.now());
    // ...waited on by a later block, if there is one: a device lost before then is said by the fold, not by this.
    finished.catch(() => {});
    this.queued.push({ pieces: this.pieces, finished });
    this.pieces = 0;
    this.unsent = 0;
    while (this.queued.length > 0 && (drain || this.queued.length >= this.window)) {
      const { pieces, finished } = this.queued.shift();
      const at = await finished;
      const perBlock = Math.max(at - this.finishedAt, 1);
      this.finishedAt = at;
      if (this.fixed === undefined) this.window = Math.min(MOST_BLOCKS, Math.ceil(QUEUED_WORK_MS / perBlock));
      if (pieces > 0) this.piecesEach = Math.max(1, Math.floor(SUBMISSION_MS / (perBlock / pieces)));
    }
  }
}

/** Where a stack's first measurement can wait: a block too small to matter queues as many as it ever would. */
export const firstWindow = (large) => (large ? 1 : MOST_BLOCKS);

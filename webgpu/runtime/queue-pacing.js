/**
 * How much of a stack may wait in the queue, as a time rather than a number of blocks, and how long one submission may be.
 *
 * Sixteen blocks in the queue is a third of a second at 150 tokens and forty seconds at 1400, all of it committed: Stop
 * waits for it, and so does closing the tab. So the window is a time, about a second of queued work, up to the sixteen a
 * small block was measured at. See the note in webgpu/af3/trunk/pairformer-block.js.
 *
 * It slides. Waiting for the queue to empty leaves the card idle while the next block is encoded, so each new block
 * waits only for the oldest. The card goes idle only where a block alone is worth the second and the window is one, and
 * encoding is then a small part of it.
 *
 * 🔴 AND A SUBMISSION IS SHORTER THAN WINDOWS WILL WAIT ON. On an RTX 3070 under Windows, another program bringing up a
 * device while a submission of 35 ms or more was running reset the graphics driver, every time, and at 26 ms and under it
 * never did, however many dispatches the time was cut into and however deep the queue. See "What reaches the queue" in
 * docs/PERF.md. So a block too long for one submission goes in pieces, as many to a submission as fill SUBMISSION_MS.
 *
 * Pieces differ by a hundred times, so a mean over them says little about any one submission: at 1000 tokens of
 * IntelliFold-2 a run of pieces counted as cheap filled 900 ms against a target of 50. So the first block sends every piece
 * alone and times it, on this card, and later blocks, which are the same pieces in the same order, add them up. A block's
 * time is the longest of the last few, because the page hears a fence late while it encodes the next block, and the block
 * after that reads short by as much. And kernels tier up during a first fold, each by its own factor, so a piece that did
 * not looks cheap beside a block that got quicker: a stack well under its steady time sends its pieces alone once more.
 */
const QUEUED_WORK_MS = 1000;
const MOST_BLOCKS = 16;
const SUBMISSION_MS = 15;
const RECENT_BLOCKS = 4;
/** What a piece can be under-read by before it is timed again, as the time a block has fallen to. */
const STALE_BLOCK = 0.8;
/** The page hears fences a message at a time once it has finished encoding, microseconds apart, whenever they finished. */
const HEARD_TOGETHER_MS = 0.25;
/** A block this short can be cut by count however its pieces differ: nothing of it is held past the longest that never reset. */
const WHOLE_BLOCK_MS = 25;
/** A pair tensor this size or larger offers its block in pieces, for the card to decide: a block is about 4 ms there on an A10. */
export const FLUSH_MIN_BYTES = 8 * 1024 * 1024;

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
    /** Each piece's share of a block, the last being what follows the last boundary. Unset until a block has timed them. */
    this.shares = undefined;
    this.blockMs = 0;
    /** What the blocks since then took, and the first full count of them: a timed block is no measure of one that is not. */
    this.recent = [];
    this.steadyMs = undefined;
    this.piece = 0;
    this.filled = 0;
    this.timed = [];
    this.timedFrom = 0;
    this.queued = [];
    this.finishedAt = performance.now();
  }

  /** What a piece of this block is expected to take. A block with more pieces than the timed one reads them at the mean. */
  pieceMs(at) {
    return (this.shares[at] ?? 1 / this.shares.length) * this.blockMs;
  }

  /** A piece of a block is encoded. `send` submits what has built up, which happens if the next piece would not fit. */
  pieceEnded(send) {
    const at = this.piece;
    this.piece += 1;
    if (this.shares === undefined) {
      // What was queued before this block is not part of its first piece.
      if (at === 0) {
        this.timedFrom = performance.now();
        this.timed.push(this.fence());
      }
      send();
      this.timed.push(this.fence());
      return;
    }
    this.filled += this.pieceMs(at);
    if (this.filled + this.pieceMs(at + 1) <= SUBMISSION_MS) return;
    send();
    this.filled = 0;
  }

  /** When everything submitted so far has finished. Waited on by a later block, if there is one: a device lost before then is said by the fold, not by this. */
  fence() {
    const finished = this.device.queue.onSubmittedWorkDone().then(() => performance.now());
    finished.catch(() => {});
    return finished;
  }

  /**
   * One block is in the queue. Resolves when there is room for the next; `drain` empties the queue whatever its length.
   * A block whose pieces were timed is always waited for, because the next one cannot be cut up without what it found.
   */
  async blockSubmitted(drain = false) {
    const finished = this.fence();
    const timed = this.timed;
    this.timed = [];
    this.queued.push(finished);
    this.piece = 0;
    this.filled = 0;
    if (timed.length > 0) this.learnPieces(await Promise.all([...timed, finished]), this.timedFrom);
    while (this.queued.length > 0 && (drain || timed.length > 0 || this.queued.length >= this.window)) {
      const at = await this.queued.shift();
      const perBlock = Math.max(at - this.finishedAt, 1);
      this.finishedAt = at;
      if (this.fixed === undefined) this.window = Math.min(MOST_BLOCKS, Math.ceil(QUEUED_WORK_MS / perBlock));
      if (timed.length === 0) this.noteBlock(perBlock);
    }
  }

  noteBlock(perBlock) {
    this.recent.push(perBlock);
    if (this.recent.length > RECENT_BLOCKS) this.recent.shift();
    this.blockMs = Math.max(...this.recent);
    if (this.recent.length < RECENT_BLOCKS) return;
    this.steadyMs ??= this.blockMs;
    if (this.blockMs < STALE_BLOCK * this.steadyMs) this.shares = undefined;
  }

  /**
   * Each piece's share of a block, from when the page heard each finish: what was queued before it, then every piece, then
   * the rest. The page hears nothing while it encodes, so pieces finish together, and pieces counted as free would next
   * time share a submission. Those heard together split the time since the last heard, and those heard with the first
   * fence, with nothing to split, read as the mean of the others.
   *
   * 🔴 THE TIME SINCE THE BLOCK'S FIRST SUBMISSION IS NOT THEIRS. Much of it is the page encoding what comes after them, more
   * of it the smaller a block is, and every share they take is taken from a piece the card did spend time on: read that
   * way, submissions at 200 and 300 tokens that had been 18 ms went to 29 ms on an A10 and 33 ms on a T4.
   *
   * 🔴 FENCES HEARD LONG AFTER THE CARD RAN THE BLOCK SAY NOTHING ABOUT ITS PIECES, and they arrive microseconds apart, so they are
   * not even equal. Read as if they said something, AF2's extra stack at 1000 tokens on an A10 sent the next three blocks of its
   * second pass whole, 257 ms each, and ESMFold2's trunk at 2046 sent 175 dispatches at once, 276 ms. So if most of a block is
   * heard with the first fence, or its fences span under half the time since its first submission, the next one is timed
   * instead, unless that time is short enough to cut the block by count, which holds nothing past WHOLE_BLOCK_MS.
   */
  learnPieces(finishes, startedAt) {
    const count = finishes.length - 1;
    let heardFirst = 0;
    while (heardFirst < count && finishes[heardFirst + 1] - finishes[heardFirst] < HEARD_TOGETHER_MS) heardFirst += 1;
    const whole = finishes[count] - startedAt;
    this.recent = [];
    this.steadyMs = undefined;
    if (2 * heardFirst > count || 2 * (finishes[count] - finishes[0]) < whole) {
      this.blockMs = whole;
      this.shares = whole > 0 && whole <= WHOLE_BLOCK_MS ? new Array(count).fill(1 / count) : undefined;
      return;
    }
    const mean = (finishes[count] - finishes[heardFirst]) / (count - heardFirst);
    const pieces = new Array(count).fill(mean);
    for (let start = heardFirst, end = start + 1; end <= count; start = end, end += 1) {
      while (end < count && finishes[end + 1] - finishes[end] < HEARD_TOGETHER_MS) end += 1;
      pieces.fill(Math.max(finishes[end] - finishes[start], 0) / (end - start), start, end);
    }
    this.blockMs = pieces.reduce((sum, ms) => sum + ms, 0);
    this.shares = this.blockMs > 0 ? pieces.map((ms) => ms / this.blockMs) : undefined;
  }
}

/**
 * A command encoder a block can be cut out of, for code that was handed one encoder and keeps it: WebGpuExecution offers a
 * cut after each dispatch, and what the code holds goes on working. Without a pacer it is the one encoder it wraps.
 */
export class PacedEncoder {
  constructor(device, pacer, label) {
    this.device = device;
    this.pacer = pacer;
    this.label = label;
    this.encoder = device.createCommandEncoder({ label });
  }

  beginComputePass(descriptor) { return this.encoder.beginComputePass(descriptor); }

  clearBuffer(...given) { this.encoder.clearBuffer(...given); }

  copyBufferToBuffer(...given) { this.encoder.copyBufferToBuffer(...given); }

  finish(descriptor) { return this.encoder.finish(descriptor); }

  /** `endPass` closes what the encoder has open, which only the caller knows about. */
  pieceEnded(endPass) {
    this.pacer?.pieceEnded(() => {
      endPass();
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = this.device.createCommandEncoder({ label: this.label });
    });
  }
}

/** Where a stack's first measurement can wait: a block too small to matter queues as many as it ever would. */
export const firstWindow = (large) => (large ? 1 : MOST_BLOCKS);

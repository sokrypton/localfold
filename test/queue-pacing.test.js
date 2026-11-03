import assert from "node:assert/strict";
import test from "node:test";

import { firstWindow, QueuePacer } from "../webgpu/runtime/queue-pacing.js";

/** A device whose queue reports each block done the moment it is asked, on a clock the test moves. */
function clocked(t) {
  let now = 0;
  const fences = [];
  t.mock.method(performance, "now", () => now);
  return {
    device: { queue: { onSubmittedWorkDone: async () => { fences.push(now); } } },
    fences,
    tick: (ms) => { now += ms; },
  };
}

/** A block of `pieces`, each taking `pieceMs`, and how many submissions it asked for. */
async function block(pacer, clock, pieces, pieceMs) {
  let sent = 0;
  for (let piece = 0; piece < pieces; piece += 1) {
    clock.tick(pieceMs);
    if (pacer.pieceDue()) sent += 1;
  }
  await pacer.blockSubmitted();
  return sent;
}

test("every piece goes alone until a block has measured one", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device);
  assert.equal(await block(pacer, clock, 40, 8), 40);
});

test("a fast card then sends as many as fill a submission", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device);
  await block(pacer, clock, 40, 8);
  assert.equal(pacer.piecesEach, 6);
  assert.equal(await block(pacer, clock, 40, 8), 6);
});

test("a slow card keeps one piece to a submission", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device);
  await block(pacer, clock, 40, 100);
  assert.equal(pacer.piecesEach, 1);
  assert.equal(await block(pacer, clock, 40, 100), 40);
});

test("what a block leaves unsent goes with its own submission, not into the next block's count", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device);
  await block(pacer, clock, 40, 8);
  await block(pacer, clock, 7, 8);
  assert.equal(pacer.unsent, 0);
});

test("the window is about a second of queued work, up to sixteen blocks", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device);
  await block(pacer, clock, 10, 25);
  assert.equal(pacer.window, 4);

  const slow = new QueuePacer(clock.device);
  await block(slow, clock, 10, 500);
  assert.equal(slow.window, 1);

  const fast = new QueuePacer(clock.device);
  await block(fast, clock, 10, 1);
  assert.equal(fast.window, 16);
});

test("a window past one waits for the oldest block and leaves the newest on the card", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { fixed: 3 });
  for (let at = 0; at < 2; at += 1) await block(pacer, clock, 1, 1);
  assert.equal(pacer.queued.length, 2);
  for (let at = 0; at < 5; at += 1) {
    await block(pacer, clock, 1, 1);
    assert.equal(pacer.queued.length, 2);
  }
});

test("a block's time is between finishes, so a block finished long before anyone asked is not read as instant", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { fixed: 2 });
  clock.tick(300);
  await pacer.blockSubmitted();
  clock.tick(300);
  // The first fence was taken at 300 and resolves then, though it is only waited on at 600.
  await pacer.blockSubmitted();
  assert.equal(pacer.finishedAt, 300);
});

test("a shrinking window waits for as many as it takes", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { start: 16 });
  for (let at = 0; at < 15; at += 1) await block(pacer, clock, 1, 1);
  assert.equal(pacer.queued.length, 15);
  clock.tick(5000);
  await pacer.blockSubmitted();
  assert.ok(pacer.queued.length < pacer.window);
});

test("a fixed window stays, and its pieces are still measured", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { fixed: 1 });
  for (let at = 0; at < 3; at += 1) await block(pacer, clock, 40, 2);
  assert.equal(pacer.window, 1);
  assert.equal(pacer.piecesEach, 25);
});

test("a heavy piece ends its submission whatever the count", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { fixed: 1 });
  await block(pacer, clock, 40, 2);
  assert.equal(pacer.piecesEach, 25);
  assert.equal(pacer.pieceDue(), false);
  assert.equal(pacer.pieceDue(true), true);
  assert.equal(pacer.unsent, 0);
});

test("a drain empties the queue whatever the window", async (t) => {
  const clock = clocked(t);
  const pacer = new QueuePacer(clock.device, { fixed: 8 });
  await pacer.blockSubmitted();
  assert.equal(pacer.queued.length, 1);
  await pacer.blockSubmitted(true);
  assert.equal(pacer.queued.length, 0);
});

test("a block too small to matter starts at the most it ever queues", () => {
  assert.equal(firstWindow(false), 16);
  assert.equal(firstWindow(true), 1);
});

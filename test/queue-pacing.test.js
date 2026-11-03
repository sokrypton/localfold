import assert from "node:assert/strict";
import test from "node:test";

import { firstWindow, PacedEncoder, QueuePacer } from "../webgpu/runtime/queue-pacing.js";

/** A device with a clock the test owns: work runs one thing at a time, and a fence resolves when all that came before it has finished. */
function simulated(t) {
  let now = 0;
  let busyUntil = 0;
  let fences = Promise.resolve();
  const submissions = [];
  t.mock.method(performance, "now", () => now);
  return {
    device: {
      queue: {
        onSubmittedWorkDone: () => {
          const at = busyUntil;
          fences = fences.then(() => { now = Math.max(now, at); });
          return fences;
        },
      },
    },
    /** What the card does when this much work is submitted. */
    submit(ms) {
      busyUntil = Math.max(busyUntil, now) + ms;
      submissions.push(ms);
    },
    tick(ms) { now += ms; },
    submissions,
    get now() { return now; },
  };
}

/** A block of pieces of these lengths, the last going with the block's own submission. Returns the length of each submission. */
async function block(pacer, card, pieces, { drain = false } = {}) {
  const before = card.submissions.length;
  let unsent = 0;
  for (const ms of pieces.slice(0, -1)) {
    unsent += ms;
    pacer.pieceEnded(() => {
      card.submit(unsent);
      unsent = 0;
    });
  }
  card.submit(unsent + pieces[pieces.length - 1]);
  await pacer.blockSubmitted(drain);
  return card.submissions.slice(before);
}

const same = (count, ms) => Array.from({ length: count }, () => ms);

test("every piece goes alone until a block has timed them", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  assert.deepEqual(await block(pacer, card, same(40, 8)), same(40, 8));
});

test("a later block fills each submission with what the first found", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(40, 2));
  const sent = await block(pacer, card, same(40, 2));
  assert.equal(sent.length, 6);
  assert.ok(sent.every((ms) => ms <= 15), String(sent));
});

test("a piece longer than a submission goes alone, and takes nothing with it", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  const pieces = [3, 3, 3, 90, 3, 3, 3];
  await block(pacer, card, pieces);
  assert.deepEqual(await block(pacer, card, pieces), [9, 90, 9]);
});

test("a mean over pieces that differ a hundred times does not fill a submission with the long ones", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  // Mean 5 ms, which a count of three to a submission would read as 15 ms, and three of the long ones are 39.
  const pieces = [...same(30, 1), ...same(10, 13), ...same(30, 1)];
  await block(pacer, card, pieces);
  const sent = await block(pacer, card, pieces);
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("what a piece is expected to take follows a slower block's time, with no need to time it again", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  await block(pacer, card, same(30, 2));
  // Something else has the card: the same pieces in twice the time.
  await block(pacer, card, same(30, 4));
  const sent = await block(pacer, card, same(30, 4));
  assert.ok(sent.length < 15, String(sent));
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("a stack much quicker than it was steady at is timed again, because kernels tier up by different amounts", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  const generic = [...same(10, 20), ...same(10, 2)];
  // Only the long kernels tier up, which leaves the short ones looking a tenth of what they cost.
  const tiered = [...same(10, 1), ...same(10, 2)];
  await block(pacer, card, generic);
  for (let at = 0; at < 4; at += 1) await block(pacer, card, generic);
  for (let at = 0; at < 4; at += 1) await block(pacer, card, tiered);
  assert.deepEqual(await block(pacer, card, tiered), tiered);
  await block(pacer, card, tiered);
  const sent = await block(pacer, card, tiered);
  assert.ok(sent.length < tiered.length, String(sent));
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("a timed block takes longer than one that is not, and that alone starts no more timing", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  await block(pacer, card, same(30, 4));
  // Each piece alone cost a millisecond more, which the same pieces together do not.
  for (let at = 0; at < 12; at += 1) {
    const sent = await block(pacer, card, same(30, 3));
    assert.ok(sent.length < 30, String(sent));
  }
});

test("a stack a little quicker than it was steady at is not worth timing again", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  await block(pacer, card, same(30, 2));
  for (let at = 0; at < 4; at += 1) await block(pacer, card, same(30, 2));
  for (let at = 0; at < 8; at += 1) assert.ok((await block(pacer, card, same(30, 1.8))).length < 30);
});

test("a block read as instant does not send the next one whole", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  await block(pacer, card, same(30, 4));
  await block(pacer, card, same(30, 4));
  // A busy page hears a fence late, and the block after it reads as nothing.
  pacer.finishedAt = 1e9;
  await block(pacer, card, same(30, 4));
  pacer.finishedAt = card.now;
  const sent = await block(pacer, card, same(30, 4));
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("a longer block raises what a piece is expected to take at once, and a shorter one only after a few", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  await block(pacer, card, same(10, 10));
  await block(pacer, card, same(10, 10));
  await block(pacer, card, same(10, 20));
  assert.equal(pacer.blockMs, 200);
  await block(pacer, card, same(10, 10));
  assert.equal(pacer.blockMs, 200);
});

test("pieces heard finishing together split the time since the last one heard", () => {
  const pacer = new QueuePacer({ queue: {} });
  pacer.learnPieces([0, 10, 40, 40, 40, 50], 0);
  assert.equal(pacer.blockMs, 50);
  assert.deepEqual(pacer.shares, [0.2, 0.2, 0.2, 0.2, 0.2]);
});

test("pieces heard with the first fence read at the mean of the others and do not go together as free", async (t) => {
  const pacer = new QueuePacer({ queue: {} });
  pacer.learnPieces([100, 100, 100, 110, 120, 130], 90);
  assert.equal(pacer.blockMs, 50);
  assert.deepEqual(pacer.shares, [0.2, 0.2, 0.2, 0.2, 0.2]);
  const card = simulated(t);
  const paced = new QueuePacer(card.device);
  paced.learnPieces([100, ...same(30, 100), ...Array.from({ length: 30 }, (_, at) => 110 + 10 * at)], 0);
  const sent = await block(paced, card, same(61, 10));
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("fences microseconds apart are heard together, and pieces heard so split the time since the last heard", () => {
  const pacer = new QueuePacer({ queue: {} });
  pacer.learnPieces([0, 10, 40, 40.01, 40.02, 50], 0);
  assert.ok(pacer.shares.every((share) => Math.abs(share - 0.2) < 1e-3), String(pacer.shares));
  // The first piece's own microseconds are no reading of it either.
  pacer.learnPieces([100, 100.01, 110, 120, 130, 140], 80);
  assert.ok(Math.abs(pacer.shares[0] - 0.2) < 1e-3, String(pacer.shares));
});

test("the time the page took to encode a block is not taken from the pieces the card spent time on", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  // 20 ms of encoding, heard with the first fence and a cheap first piece, then ten pieces of 2.5 ms.
  pacer.learnPieces([120, 120, ...Array.from({ length: 10 }, (_, at) => 122.5 + 2.5 * at)], 100);
  const pieces = same(11, 2.5);
  await block(pacer, card, pieces);
  await block(pacer, card, pieces);
  const sent = await block(pacer, card, same(11, 2.5));
  assert.ok(Math.max(...sent) <= 15, String(sent));
  assert.ok(Math.abs(pacer.shares[5] - 1 / 11) < 1e-9, String(pacer.shares));
});

test("a block mostly heard with the first fence says nothing about its pieces, and the next is timed instead", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  // The page took 300 ms to encode what the card had finished by then, and the two pieces it heard last were short.
  pacer.learnPieces([400, ...same(28, 400), 400.1, 400.2], 100);
  assert.equal(pacer.shares, undefined);
  assert.deepEqual(await block(pacer, card, same(5, 3)), same(5, 3));
});

test("fences that span little of the time since the first submission say nothing about the pieces, however far apart", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  // A block of about 300 ms, read as 9 ms: this is what sent AF2's next three blocks whole.
  pacer.learnPieces(Array.from({ length: 31 }, (_, at) => 400 + 0.3 * at), 100);
  assert.equal(pacer.shares, undefined);
  assert.deepEqual(await block(pacer, card, same(5, 3)), same(5, 3));
});

test("a block all heard at once, in less time than a submission, fits one", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  pacer.learnPieces(same(11, 8), 3);
  assert.equal(pacer.blockMs, 5);
  assert.deepEqual(await block(pacer, card, same(10, 0.5)), [5]);
});

test("a block too short to hold anything past what never reset goes by count, and is not timed for ever", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  pacer.learnPieces([112, ...same(10, 112.01)], 100);
  assert.ok(Math.abs(pacer.blockMs - 12.01) < 1e-9, String(pacer.blockMs));
  const longer = new QueuePacer(card.device, { fixed: 1 });
  longer.learnPieces([120, ...same(10, 120.01)], 100);
  const sent = await block(longer, card, same(10, 2));
  assert.deepEqual(sent, [14, 6]);
});

test("pieces whose fences all arrive with no time gone have told nothing, and a block goes on sending them alone", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(5, 0));
  assert.equal(pacer.shares, undefined);
  assert.deepEqual(await block(pacer, card, same(5, 3)), same(5, 3));
});

test("a fast card sends a block that fits whole", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(10, 1));
  assert.deepEqual(await block(pacer, card, same(10, 1)), [10]);
});

test("a slow card keeps one piece to a submission", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(10, 100));
  assert.deepEqual(await block(pacer, card, same(10, 100)), same(10, 100));
});

test("what came before the block is not read as its first piece, and the block after it reads the card alone", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  card.submit(500);
  await block(pacer, card, same(10, 5));
  assert.equal(pacer.shares, undefined);
  await block(pacer, card, same(10, 5));
  assert.ok(Math.abs(pacer.shares[0] - 0.1) < 1e-9, String(pacer.shares));
  assert.equal(pacer.blockMs, 50);
});

test("a block with more pieces than the timed one reads the rest at the mean", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(4, 5));
  const sent = await block(pacer, card, same(12, 5));
  assert.ok(Math.max(...sent) <= 15, String(sent));
});

test("a block with no pieces times nothing and is not waited for", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { start: 16 });
  await block(pacer, card, [100]);
  assert.equal(pacer.shares, undefined);
  assert.equal(pacer.queued.length, 1);
});

test("the block whose pieces are timed is waited for whatever the window", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 16 });
  await block(pacer, card, same(4, 5));
  assert.equal(pacer.queued.length, 0);
  assert.equal(card.now, 20);
});

test("the window is about a second of queued work, up to sixteen blocks", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device);
  await block(pacer, card, same(10, 25));
  assert.equal(pacer.window, 4);

  const slow = new QueuePacer(card.device);
  await block(slow, card, same(10, 500));
  assert.equal(slow.window, 1);

  const fast = new QueuePacer(card.device);
  await block(fast, card, same(10, 1));
  assert.equal(fast.window, 16);
});

test("a window past one waits for the oldest block and leaves the newest on the card", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 3 });
  for (let at = 0; at < 2; at += 1) await block(pacer, card, [1]);
  assert.equal(pacer.queued.length, 2);
  for (let at = 0; at < 5; at += 1) {
    await block(pacer, card, [1]);
    assert.equal(pacer.queued.length, 2);
  }
});

test("a block's time is between finishes, so a block finished long before anyone asked is not read as instant", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 2 });
  card.tick(300);
  await pacer.blockSubmitted();
  await new Promise(setImmediate);
  card.tick(300);
  // The first fence was taken at 300 and resolves then, though it is only waited on at 600.
  await pacer.blockSubmitted();
  assert.equal(pacer.finishedAt, 300);
});

test("a shrinking window waits for as many as it takes", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { start: 16 });
  for (let at = 0; at < 15; at += 1) await block(pacer, card, [1]);
  assert.equal(pacer.queued.length, 15);
  card.tick(5000);
  await pacer.blockSubmitted();
  assert.ok(pacer.queued.length < pacer.window);
});

test("a fixed window stays, and its pieces are still timed", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 1 });
  for (let at = 0; at < 3; at += 1) await block(pacer, card, same(40, 2));
  assert.equal(pacer.window, 1);
  assert.equal(pacer.shares.length, 40);
});

test("a drain empties the queue whatever the window", async (t) => {
  const card = simulated(t);
  const pacer = new QueuePacer(card.device, { fixed: 8 });
  await pacer.blockSubmitted();
  assert.equal(pacer.queued.length, 1);
  await pacer.blockSubmitted(true);
  assert.equal(pacer.queued.length, 0);
});

test("a device lost while pieces are timed is said by the fold's own wait", async () => {
  const lost = new Error("device lost");
  const pacer = new QueuePacer({ queue: { onSubmittedWorkDone: () => Promise.reject(lost) } });
  pacer.pieceEnded(() => {});
  await assert.rejects(pacer.blockSubmitted(), lost);
});

test("a block too small to matter starts at the most it ever queues", () => {
  assert.equal(firstWindow(false), 16);
  assert.equal(firstWindow(true), 1);
});

/** An encoder that records what it was asked to do, so a test can read where the cuts fell. */
function recordingDevice() {
  const made = [];
  const submitted = [];
  const device = {
    createCommandEncoder: ({ label }) => {
      const encoder = {
        label,
        log: [],
        beginComputePass: () => { encoder.log.push("pass"); return {}; },
        clearBuffer: (...given) => { encoder.log.push(["clear", ...given]); },
        copyBufferToBuffer: (...given) => { encoder.log.push(["copy", ...given]); },
        finish: () => encoder,
      };
      made.push(encoder);
      return encoder;
    },
    queue: { submit: ([buffer]) => { submitted.push(buffer); } },
  };
  return { device, made, submitted };
}

test("a paced encoder without a pacer is the one encoder it wraps", () => {
  const { device, made, submitted } = recordingDevice();
  const encoder = new PacedEncoder(device, undefined, "block");
  encoder.beginComputePass({});
  encoder.pieceEnded(() => assert.fail("nothing to end"));
  encoder.clearBuffer("a");
  encoder.copyBufferToBuffer("a", 0, "b", 0, 4);
  assert.equal(made.length, 1);
  assert.equal(submitted.length, 0);
  assert.equal(encoder.finish(), made[0]);
  assert.deepEqual(made[0].log, ["pass", ["clear", "a"], ["copy", "a", 0, "b", 0, 4]]);
});

test("a paced encoder's cut closes what is open and carries on in a new one, under the same label", () => {
  const { device, made, submitted } = recordingDevice();
  const ended = [];
  const encoder = new PacedEncoder(device, { pieceEnded: (send) => send() }, "block");
  encoder.beginComputePass({});
  encoder.pieceEnded(() => ended.push("pass"));
  encoder.clearBuffer("a");
  assert.deepEqual(ended, ["pass"]);
  assert.deepEqual(submitted, [made[0]]);
  assert.equal(made[1].label, "block");
  assert.deepEqual(made[1].log, [["clear", "a"]]);
  assert.equal(encoder.finish(), made[1]);
});

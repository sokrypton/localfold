/**
 * Chrome's own benchmark of the GPU, moved out of a fold's way.
 *
 * Chrome benchmarks the card once per version of Chrome, in a process of its own, three minutes after the browser starts,
 * to decide whether its built-in AI can run here. A fold started within three minutes of opening the browser is on the
 * card when that process brings up its device, and on an RTX 3070 under Windows that reset the graphics driver. See "What
 * reaches the queue" in docs/PERF.md.
 *
 * 🔴 ASKING RUNS IT. `Summarizer.availability()` runs the benchmark there and then if it is still owed, answers a second or
 * two later and downloads nothing, and three minutes then finds the result and starts no process. So the page asks as it
 * loads and the first device waits for the answer, which has the benchmark run on a card holding nothing of ours.
 *
 * A browser without the API has nothing to wait for, and one that does not answer must not hold up Fold.
 */
const ANSWER_WAIT_MS = 15 * 1000;

let asked;

async function askForAvailability() {
  try {
    await globalThis.Summarizer?.availability?.();
  } catch (cause) {
    console.warn("Summarizer.availability", cause);
  }
}

/** Resolves when Chrome has no benchmark left to run, or has had long enough. Asks once a page. */
export function settleBrowserBenchmark(waitMs = ANSWER_WAIT_MS) {
  asked ??= new Promise((done) => {
    const timer = setTimeout(done, waitMs);
    void askForAvailability().then(() => {
      clearTimeout(timer);
      done();
    });
  });
  return asked;
}

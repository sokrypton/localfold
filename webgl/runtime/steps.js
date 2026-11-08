/**
 * Running a fold that yields, synchronously or not.
 *
 * 🔴 A FOLD ON THE PAGE'S OWN THREAD MUST LET GO OF IT. WebGL2 has no compute
 * queue of its own: every draw is issued from the thread that owns the context,
 * and every read blocks it until the GPU has caught up. A whole AF3 fold run as
 * one call held a tab for minutes - no paint, no Stop - and on Safari 16, the
 * browsers this backend exists for, there is no OffscreenCanvas WebGL2 to move
 * it into a Worker. So the long stages are GENERATORS that yield at their
 * natural seams (a block, a denoiser call), and the caller chooses how to run
 * them: `drain` straight through, as every tool and gate does, or `drainAsync`,
 * which waits for the GPU at each yield without blocking the thread.
 */

/** Run a generator to completion, synchronously; its return value. */
export function drain(steps) {
  let next = steps.next();
  while (!next.done) next = steps.next();
  return next.value;
}

/**
 * Run a generator to completion, awaiting `pause(yielded)` at each yield; its
 * return value. `signal` is checked at every yield, so an abort lands at the next
 * block or denoiser call rather than at the end of the fold.
 */
export async function drainAsync(steps, pause, signal) {
  let next = steps.next();
  while (!next.done) {
    if (signal?.aborted) {
      steps.return?.();
      throw signal.reason ?? new DOMException("The fold was stopped", "AbortError");
    }
    await pause(next.value);
    next = steps.next();
  }
  return next.value;
}

/**
 * Resolves once the GPU has run everything issued on `gl` so far: a fence,
 * polled from timers, so the thread is free while the queue drains. A read
 * after this returns at once instead of blocking.
 */
export async function gpuIdle(gl, work) {
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  gl.flush();
  try {
    await fenceSignalled(gl, sync, work);
  } finally {
    gl.deleteSync(sync);
  }
}

/** Resolves once `sync` has signalled, polled from timers; `work` as `gpuIdle`'s. */
async function fenceSignalled(gl, sync, work) {
  for (;;) {
    const status = gl.clientWaitSync(sync, 0, 0);
    if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED
      || status === gl.WAIT_FAILED) return;
    // ...and the time the GPU takes is the caller's to use: one piece of `work` a task (a
    // shader compile the fold will need later), else a short sleep.
    const busy = work?.() === true;
    await new Promise((resolve) => { setTimeout(resolve, busy ? 0 : 4); });
  }
}

/**
 * A pause that lets go of the thread at every yield WITHOUT letting the GPU run dry: each call fences what has been
 * issued so far and waits for the fence of the call BEFORE it, so one slice is always queued while the next is being
 * issued. `finish()` releases the last fence.
 *
 * 🔴 `gpuIdle` AT EVERY SEAM DRAINED THE GPU EVERY TIME. A page fold yields ~300 times (a trunk block, a super-block
 * of the denoiser) and each yield waited for the queue to EMPTY, polled on 4 ms timers, before the next slice was
 * even issued - so the GPU sat idle for the poll's overshoot and the next slice's whole issue time, every seam. The
 * same AF3 fold was ~1.6 s slower on the page than in the tool that runs it straight through (`drain`).
 */
export function laggedPause(gl, work) {
  let previous = null;
  const pause = async () => {
    const current = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    if (previous !== null) {
      try {
        await fenceSignalled(gl, previous, work);
      } finally {
        gl.deleteSync(previous);
      }
    }
    previous = current;
  };
  pause.finish = () => {
    if (previous !== null) gl.deleteSync(previous);
    previous = null;
  };
  return pause;
}

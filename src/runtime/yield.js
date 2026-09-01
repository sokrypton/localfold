/**
 * Hand control back to the browser without being throttled for it.
 *
 * 🔴 setTimeout IS CLAMPED TO >=1 SECOND IN A BACKGROUND TAB, and a fold yields
 * after every pairformer block. Forty-eight blocks a pass, four passes, and a
 * prediction that takes two minutes in a visible tab takes over three hundred
 * in a hidden one - measured: pass 1 reached block 11 in five minutes hidden,
 * then jumped to block 28 the moment the tab was touched. The work is on the
 * GPU and does not care whether anyone is looking; only the yield does.
 *
 * A MessageChannel message is a task, so it still lets the browser paint and
 * still lets a click land, but it is not a timer and is not clamped. This is
 * the standard workaround and it is the whole of the file.
 *
 * 🔴 IT IS STILL A REAL YIELD, WHICH IS THE POINT. Resolving a promise directly
 * would be a microtask: it would return control to the event loop's microtask
 * queue and never to the browser, so the page would not paint and the Stop
 * button would not respond. That is the failure this replaces, not one it
 * introduces.
 */

/** One channel for the process; ports are cheap to reuse and not to create. */
let channel;

/** @returns {Promise<void>} resolved on the next task, throttled or not. */
export function yieldToBrowser() {
  if (typeof MessageChannel === "undefined") {
    // Node, and any browser old enough to lack it: a timer is all there is, and
    // in Node there is no tab to background so the clamp does not apply.
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (channel === undefined) channel = new MessageChannel();
  return new Promise((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(undefined);
  });
}

/**
 * What a long fold asks of the browser while it runs.
 *
 * Two tabs folding at once each get their own device and each believe they have the card's whole budget, and that is
 * how a fold that fits loses its device. A screen left to sleep takes the machine with it on most laptops. And somebody
 * who has switched away wants to see from the tab strip how far it has got.
 *
 * Every API here is optional, and a browser without one folds exactly as it did.
 */

const LOCK = "localfold-fold";
/** A fold past this is worth a question before the tab closes on it. */
const CONFIRM_CLOSE_AFTER_MS = 30 * 1000;
/** And past this the screen stays on. */
const KEEP_AWAKE_AFTER_MS = 60 * 1000;

const PAGE_TITLE = document.title;

function keepAwake() {
  let lock;
  let released = false;
  const take = async () => {
    if (released || document.visibilityState !== "visible") return;
    try { lock = await navigator.wakeLock?.request("screen"); } catch { /* refused: the screen does as it likes */ }
  };
  // The browser lets go of it whenever the tab is hidden.
  const onVisibility = () => { void take(); };
  document.addEventListener("visibilitychange", onVisibility);
  void take();
  return () => {
    released = true;
    document.removeEventListener("visibilitychange", onVisibility);
    void lock?.release().catch(() => {});
  };
}

function confirmClose() {
  const started = performance.now();
  const onLeave = (event) => {
    if (performance.now() - started < CONFIRM_CLOSE_AFTER_MS) return;
    event.preventDefault();
    event.returnValue = "";
  };
  window.addEventListener("beforeunload", onLeave);
  return () => window.removeEventListener("beforeunload", onLeave);
}

/**
 * One fold at a time across this browser's tabs: this waits its turn, and Stop ends the wait.
 *
 * Held for the whole fold, which also keeps a hidden tab from being frozen under it.
 *
 * @param {AbortSignal} signal
 * @param {() => void} onWaiting called if another tab is folding
 * @returns {Promise<() => void>} call to let the next one go
 */
async function foldTurn(signal, onWaiting) {
  if (navigator.locks === undefined) return () => {};
  let finish;
  const held = new Promise((resolve) => { finish = resolve; });
  const take = (options) => new Promise((resolve, reject) => {
    navigator.locks.request(LOCK, options, (lock) => {
      resolve(lock !== null);
      return lock === null ? undefined : held;
    }).catch(reject);
  });
  if (!await take({ ifAvailable: true })) {
    onWaiting();
    await take({ signal });
  }
  return finish;
}

/**
 * Everything above, for one fold.
 *
 * @param {{signal: AbortSignal, onWaiting: () => void}} options
 * @returns {Promise<() => void>} ends it, whatever became of the fold
 */
export async function beginFold({ signal, onWaiting }) {
  const endTurn = await foldTurn(signal, onWaiting);
  const allowClose = confirmClose();
  let allowSleep = () => {};
  const staying = setTimeout(() => { allowSleep = keepAwake(); }, KEEP_AWAKE_AFTER_MS);
  return () => {
    clearTimeout(staying);
    allowSleep();
    allowClose();
    endTurn();
  };
}

/** The tab's title carries the fold's progress, for a tab somebody has left. `null` puts it back. */
export function showProgressInTitle(fraction) {
  document.title = fraction === null || !Number.isFinite(fraction) ? PAGE_TITLE
    : `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}% · ${PAGE_TITLE}`;
}

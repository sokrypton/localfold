/**
 * THE READER'S PAGE, FOLDING ON A COLAB RUNTIME'S GPU.
 *
 * `notebooks/localfold.ipynb` runs tools/colab_backend.py on the runtime, which
 * serves this checkout and keeps one worker process - cuda/worker.py, LocalFold's
 * native CUDA ports on the page's own weights and inputs. The notebook's link
 * opens `index.html?backend=colab&t=…` on that server; this page then sends each
 * fold there and draws what the worker says as it says it: every status line,
 * bar fraction, sampler frame and contact map, then the finished fold.
 *
 * 🔴 THERE WAS A SECOND HALF, AND IT IS GONE. The runtime used to run this same
 * page headlessly (`?role=runtime`) and fold with its WebGPU code on commands
 * relayed through the broker; once every fold went to CUDA nothing reached it,
 * and it was removed (docs/WEB.md, 2026-10-08).
 *
 * Absent `?backend=colab`, every function here is inert and index.html is what
 * it was: the website folds in the reader's own browser.
 */

import { devSourceIs } from "./dev-log.js";

/**
 * The broker is the server this page was served BY, so every route is relative.
 *
 * 🔴 THE TOKEN IS TAKEN ONCE, AT LOAD, NOT READ PER REQUEST. Disconnect drops
 * the query string with `replaceState` - the page is no longer a reader - and
 * a `door()` that re-read `location.search` then asked with `t=` empty:
 * reported from a real runtime as a console full of
 * `GET /down?t=&head=1 403 (Forbidden)`. The URL is where the token ARRIVES;
 * it is not where it lives.
 */
const TOKEN = new URLSearchParams(location.search).get("t") ?? "";
const door = (route, extra = "") =>
  `${route}?t=${encodeURIComponent(TOKEN)}${extra}`;

const ask = async (route, body) => {
  const answer = await fetch(door(route), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  // 🔴 A REFUSAL CARRIES ITS REASON. "one GPU, one fold: try again" is the
  // whole of what a reader needs to know, and `the broker answered 429` is
  // the same event with the answer taken out of it.
  const said = await answer.json().catch(() => ({}));
  if (!answer.ok) {
    throw new Error(said.error ?? `the broker answered ${answer.status}`);
  }
  return said;
};

export const colabRole = () =>
  new URLSearchParams(location.search).get("backend") === "colab" ? "reader" : null;

/**
 * Where a reader's fold runs: LocalFold's native CUDA ports on the runtime
 * (cuda/worker.py), always. There is no choice on the page - a runtime the
 * notebook could not build CUDA on refuses the fold by saying so (see
 * tools/colab_backend.py), rather than folding somewhere slower in silence.
 */
export const remoteBackendChoice = () => "cuda";
// ...and whether a CUDA fold streams its intermediate results to this page while it runs (each trunk pass's
// contact map, the sampler's frames, AF2's passes with their scores) - the reader's choice, on the badge,
// because it is this page that draws them. Measured at under 1% of a fold (cuda/worker.py).
let liveChoice = true;
export const remoteLiveChoice = () => liveChoice;

/** Ask the runtime for something. Returns the command's sequence number. */
export const remoteCommand = (op, payload) => ask("/in", { op, payload });

/**
 * What the runtime has said since `since` - held by the broker until there is
 * something (or `waitMs` passes), so a fold's events arrive as they happen and
 * not on the next poll. The answer's `waits` says the broker did hold it: an
 * older one answers at once, and its reader must still pause between asks.
 */
export async function remoteEvents(since, signal, waitMs = 8000) {
  const answer = await fetch(door("/down", `&since=${since}&wait=${waitMs}`), { signal });
  if (!answer.ok) throw new Error(`the runtime answered ${answer.status} while folding`);
  return answer.json();
}

/** ...and where the stream stands, without being handed it. */
export async function remoteHead(signal) {
  const answer = await fetch(door("/down", "&head=1"), { signal });
  if (!answer.ok) throw new Error(`the runtime answered ${answer.status}`);
  return answer.json();
}

/* ------------------------------------------- ...and whose GPU this page is using */

/**
 * SAY THAT THIS PAGE IS FOLDING SOMEWHERE ELSE, AND OFFER THE WAY BACK.
 *
 * 🔴 NOTHING ON THE PAGE SAID SO. `?backend=colab` is in the URL and the fold
 * happens on a machine the reader cannot see - so a tab left open after the
 * notebook was closed looks exactly like a tab that folds here, and the first
 * news of the difference is a fold that goes nowhere. The badge is the one
 * place that answers "where does Fold run" and "is that thing still there".
 *
 * 🔴 BUILT, NOT MARKED UP, so index.html carries nothing for a mode it is
 * usually not in - and so a page served from anywhere gets it. Same rule as
 * py2Dmol's own tab strip.
 *
 * 🔴 AND THE HEARTBEAT IS THE BROKER ANSWERING `/down?head=1`, which costs it
 * nothing: a runtime that has gone (a closed notebook, a recycled VM) takes the
 * server with it, and three unanswered asks in a row say so. `/health` is
 * asked until it answers once, for the card's name and what it offers.
 */
function installColabStatus() {
  devSourceIs("the Colab runtime");
  const head = document.querySelector(".page-head-fold") ?? document.body;
  const badge = document.createElement("div");
  badge.id = "colab-status";
  badge.className = "colab-status";
  const dot = document.createElement("span");
  dot.className = "colab-dot";
  const said = document.createElement("span");
  said.className = "colab-said";
  said.textContent = "Colab runtime";
  const leave = document.createElement("button");
  leave.type = "button";
  leave.className = "btn btn-grey btn-small";
  leave.textContent = "Disconnect";
  // 🔴 IT STOPS THE SERVICE, WHICH IS WHAT FREES THE CARD. Walking away from
  // the runtime and leaving it folding for nobody is not disconnecting - the
  // worker on that machine holds the GPU for as long as it lives. What this
  // CANNOT do is end the Colab runtime itself: that machine belongs to the
  // notebook, and only its own Runtime menu releases it. The title says so,
  // because a button that half-does what its name says is worse than one that
  // says what it does.
  leave.title = "Stop the fold service on the runtime and free its GPU. This"
    + " page then folds in your browser. The notebook itself stays open -"
    + " Runtime > Disconnect and delete runtime releases the machine.";
  badge.append(dot, said, leave);
  // 🔴 IN THE MIDDLE, IN ITS OWN SLOT, rather than appended to the head. The
  // head is `space-between` with a title and the actions, so a third child
  // pushed Fold and Add entity out of the place a reader had already learnt.
  // A stretching slot between them takes the leftover width and centres the
  // badge in it; the buttons do not move.
  const slot = document.createElement("div");
  slot.className = "colab-status-slot";
  slot.append(badge);
  const actions = head.querySelector(".fold-actions");
  if (actions === null) head.append(slot);
  else head.insertBefore(slot, actions);

  // 🔴 ASKED UNTIL IT ANSWERS, AND THEN NOT AGAIN: the card does not change while
  // the service lives, and a single attempt at load lost the race often enough
  // that the badge read "Colab runtime" with no card.
  let card = "";
  let releases = false;
  let cudaOffered = null;
  const nameTheCard = async () => {
    if (card !== "") return;
    try {
      const health = await (await fetch(door("/health"))).json();
      const gpu = health.gpu ?? {};
      card = gpu.name ?? "";
      // 🔴 AND WHETHER DISCONNECT RELEASES THE MACHINE OR ONLY STOPS THE
      // SERVICE ON IT. On Colab it is both; on a runtime somebody is hosting
      // by hand there is no machine to hand back, and a button that promises
      // one either way is wrong half the time.
      releases = health.colabRuntime === true;
      // 🔴 NO CUDA, NO FOLD - AND THE BADGE SAYS SO BEFORE THE READER TRIES. The
      // notebook builds the CUDA ports on every runtime it starts; one with no
      // NVIDIA card (a TPU, a CPU runtime) offers none, and the broker refuses
      // the fold with the same sentence.
      cudaOffered = (health.backends ?? []).includes("cuda");
      if (cudaOffered && !badge.querySelector(".colab-live")) {
        // 🔴 LIVE PREVIEW, ON THE PAGE AND NOT IN THE NOTEBOOK: it decides what this page draws, so it
        // is set here, per fold, by whoever is watching
        const live = document.createElement("label");
        live.className = "colab-live";
        live.title = "Live preview: stream a CUDA fold's intermediate results as it runs - each trunk"
          + " pass's contact map, the sampler's frames, AlphaFold 2's passes with their scores. Off, the"
          + " page shows the finished fold only. Measured at under 1% of a fold.";
        const box = document.createElement("input");
        box.type = "checkbox";
        try { box.checked = localStorage.getItem("localfold.colabLive") !== "off"; }
        catch (cause) { box.checked = true; }
        liveChoice = box.checked;
        box.addEventListener("change", () => {
          liveChoice = box.checked;
          try { localStorage.setItem("localfold.colabLive", box.checked ? "on" : "off"); }
          catch (cause) { /* remembered for this page only */ }
        });
        live.append(box, document.createTextNode(" Live"));
        badge.insertBefore(live, leave);
      }
      leave.title = releases
        ? "Stop folding here and release this Colab machine: the service"
          + " stops, its GPU is freed and the runtime is unassigned. This page"
          + " keeps what it has already folded."
        : "Stop the fold service on the runtime and free its GPU. This page"
          + " keeps what it has already folded.";
      // ...and the timing report is headed with it, because the rows in it
      // were recorded on that card and not on this one.
      if (card !== "") devSourceIs(`the Colab runtime · ${card}`);
    } catch (cause) { /* the pulse below is what matters; this is its name */ }
  };

  let folding = false;
  let beating = 0;
  // 🔴 AND THE PULSE STOPS WHEN THERE IS NOTHING LEFT TO ASK. The badge polled
  // every three seconds for the life of the tab, so after Disconnect it went
  // on knocking at a service that was shutting down - 403, then 500 after
  // 500, in a console a reader was reading to find out whether Disconnect had
  // worked. Two ways to stop: being told (below), and a handful of failures
  // in a row, which is the runtime having gone without being told.
  let misses = 0;
  const beat = async () => {
    // 🔴 COUNTED WHERE IT HAPPENS. A wrapper on `window.fetch` was the
    // obvious way to watch this from outside and it measured ZERO while the
    // badge was visibly updating - so what a gate (or a reader in the
    // console) can read is the pulse's own count, which cannot be wrong
    // about whether the pulse is running.
    window.__colabBeats = (window.__colabBeats ?? 0) + 1;
    try {
      await nameTheCard();
      const head2 = await remoteHead();
      folding = !!head2.folding;
      misses = 0;
      badge.dataset.state = "live";
      said.textContent = `Colab runtime${card ? ` · ${card}` : ""}${cudaOffered === false ? " · no CUDA backend" : ""}`;
      leave.textContent = folding ? "Stop & disconnect" : "Disconnect";
    } catch (cause) {
      badge.dataset.state = "gone";
      said.textContent = "Colab runtime · unreachable";
      misses += 1;
      if (misses >= 3) stopBeating();
    }
  };

  const stopBeating = () => {
    if (beating !== 0) clearInterval(beating);
    beating = 0;
  };
  void beat();
  beating = setInterval(() => void beat(), 3000);

  leave.addEventListener("click", async () => {
    leave.disabled = true;
    // 🔴 THE FOLD FIRST, THEN THE SERVICE, so nothing is left finishing a fold
    // nobody will read.
    if (folding) await remoteCommand("stop", null).catch(() => {});
    await remoteCommand("shutdown", null).catch(() => {});
    // 🔴 AND THIS PAGE DOES NOT RELOAD, because the server it was served BY is
    // the thing that just stopped. Everything it needs is already here: the
    // bundle, the viewer, and weights that come from huggingface rather than
    // from the runtime - so dropping the parameters with `replaceState` is the
    // whole of coming home. A navigate would have asked a dead server for the
    // page and got nothing.
    // 🔴 THE PULSE FIRST, THEN THE URL. Both orders leave the same page, and
    // only this one leaves a quiet console: the service is on its way down
    // and there is nothing left to ask it.
    stopBeating();
    history.replaceState({}, "", location.pathname);
    // 🔴 THE PAGE DOES NOT START FOLDING HERE INSTEAD. The reader ended the
    // service; a page that quietly took the work over would be answering a
    // question nobody asked, on a laptop that may be nothing like the card
    // they were using. What is left is what they still want: the structure,
    // the plots and the downloads of what was already folded.
    document.dispatchEvent(new CustomEvent("localfold-runtime-stopped", {
      detail: { why: "the Colab runtime was stopped from this page - open the"
        + " notebook's link again to fold" },
    }));
    badge.dataset.state = "gone";
    badge.textContent = "";
    const dot2 = document.createElement("span");
    dot2.className = "colab-dot";
    const gone = document.createElement("span");
    gone.className = "colab-said";
    gone.textContent = releases ? "Colab runtime · released"
    : "Colab runtime · stopped";
    badge.append(dot2, gone);
    const line = document.getElementById("status-message");
    if (line !== null) {
      line.textContent = (releases
        ? "The Colab runtime has been released - the machine is handed back"
          + " and the notebook is disconnected."
        // 🔴 AND WHERE IT CANNOT BE HANDED BACK, SAY WHAT IS LEFT TO DO.
        // Reported as "hitting disconnect did not disconnect the runtime" -
        // which is true of a runtime started before this existed, and of any
        // host that is not Colab. A line that stops at "the service is
        // stopped" leaves a reader thinking the machine went with it.
        : "The fold service has been stopped and its GPU freed. This runtime"
          + " cannot hand its machine back - if it is a Colab one, run the"
          + " notebook's cell again to pick this up, or use Runtime >"
          + " Disconnect and delete runtime.")
        + " This page is now showing what it already has; the notebook's link"
        + " starts a new one.";
    }
  });
}

/** The badge, on a reader's page. Called once, by web/app.js. */
export function installColabBridge() {
  if (colabRole() === "reader") installColabStatus();
}

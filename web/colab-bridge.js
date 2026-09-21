/**
 * TWO WAY, BETWEEN THE PAGE A READER LOOKS AT AND THE ONE THAT FOLDS.
 *
 * In a Colab runtime the GPU is on the machine that has no screen. WebGPU only
 * runs in a browser, so the runtime runs one headlessly on `index.html` - the
 * same page and the same fold, which is the whole reason this arrangement
 * exists rather than a port of the model into Python. What has to travel is
 * everything a person would have seen: the status line, the bar, every sampler
 * frame, the structure and the prediction behind it - and, in the other
 * direction, what they asked for.
 *
 * 🔴 IT PUSHES. THE FIRST VERSION WAS PULLED OVER CDP AND THAT IS WHY IT WAS
 * NOT LIVE. `tools/colab_backend.py` used to collect `window.__remoteTap` by
 * evaluating a splice every 250 ms, so a reader saw the fold only as often as
 * a busy page answered the debugger: reported as the bar sitting at "embedder
 * · 1%" for a whole fold and the finished structure appearing at the end. A
 * push leaves at the MOMENT of the event, in the same task that made it - the
 * page has to be running to produce an event at all, so nothing is gained by
 * asking it again later.
 *
 * 🔴 AND THE COMMANDS COME BACK THE SAME WAY. The runtime page asks the broker
 * what has been requested, rather than the backend pretending to be a mouse
 * over CDP: the entity list, the model row and the Fold button are set from
 * INSIDE the page, by the same writes a person's click would make. CDP is left
 * with the two jobs only it can do - start the browser and say what card it
 * got.
 *
 * ROLES, BOTH FROM THE URL, so neither page needs a build of its own:
 *   ?role=runtime&t=…    the headless page on the runtime. Pushes, obeys.
 *   ?backend=colab&t=…   the reader's page. Asks, receives. (web/app.js)
 * Absent both, every function here is inert and index.html is what it was.
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

export const colabRole = () => {
  const asked = new URLSearchParams(location.search);
  if (asked.get("role") === "runtime") return "runtime";
  if (asked.get("backend") === "colab") return "reader";
  return null;
};

/* ------------------------------------------------------------------ the page
   that folds: what it says, sent as it says it. */

let pending = [];
let seqOut = 0;

/**
 * 🔴 NOT ONE REQUEST AT A TIME, WHICH IS WHERE THE FIRST VERSION OF THIS PUT
 * THE FAULT BACK. Holding the next batch until the last one RESOLVED needs the
 * main thread to run the response, and a page in the middle of a fold does not
 * give it up - so twenty events pushed across six seconds of 300 ms tasks
 * reached the broker at **p50 3.0 s, worst 5.7 s**, which is the pulled feed's
 * behaviour wearing a push's clothes. `tools/check-colab-bridge.py` measures
 * exactly that and holds it under a second.
 *
 * A send is STARTED in the task that made the event and nothing waits for its
 * answer. What that costs is ordering - several requests in flight can arrive
 * in any order - so every event carries a `seq` and the reader applies each
 * batch in it.
 */
function flush() {
  if (pending.length === 0) return;
  const batch = pending;
  pending = [];
  // 🔴 A LOST BATCH IS A LOST PICTURE, NOT A LOST FOLD. The fold is running on
  // this page and its result is read back at the end from what the page HAS;
  // throwing here would take the fold down to save the commentary.
  ask("/up", { events: batch }).catch((cause) => {
    console.warn("colab bridge: an event batch did not send:", cause.message);
  });
}

/**
 * One thing the page said, on its way out.
 *
 * Called by web/app.js's `remoteTap` for every status write, bar fraction and
 * sampler frame - the SAME calls a local fold makes, which is what keeps this
 * from being a second reporting path that can drift from the real one.
 *
 * 🔴 `at` IS THE PAGE'S OWN CLOCK and the broker stamps its arrival beside it,
 * so "produced late" and "delivered late" are two numbers rather than one
 * argument. That distinction is what the pulled version could not make.
 */
export function tapOut(kind, payload) {
  if (colabRole() !== "runtime") return;
  pending.push({ kind, payload, at: Date.now(), seq: seqOut });
  seqOut += 1;
  flush();
}

/* ------------------------------------------------- ...and what it is told to do */

const idle = (ms) => new Promise((done) => setTimeout(done, ms));

const statusText = () =>
  document.getElementById("status-message")?.textContent ?? "";

/**
 * 🔴 THE PAGE SAYS WHEN IT HAS FAILED, AND IT SAYS IT IN A CLASS.
 * `status(text, true)` marks the line `.error`, which is the same signal a
 * reader gets - where the word list this replaces ("stopped", "failed",
 * "refus") was a guess at the page's vocabulary, kept in another file, in
 * another language.
 */
const failed = () =>
  !!document.getElementById("status-message")?.classList.contains("error");

/** Is a fold running here? The page states it; everything else is a proxy. */
const folding = () => !!(window.__foldState && window.__foldState.running);

/**
 * The readback, which is the page's own download button and its own prediction.
 *
 * 🔴 THE WHOLE PREDICTION, AS A STRING, NOT FIELD BY FIELD. The archive the
 * download button writes reads `stem`, `model`, `settings`, `entities`,
 * `msas`, `msaOrigin`, `confidence`, `chainLengths` and more; naming them here
 * is the field-by-field rebuild this repository has been bitten by six times,
 * and it always fails silently - a zip with a piece missing.
 */
async function readBack() {
  const blobs = [];
  const made = URL.createObjectURL;
  URL.createObjectURL = (b) => { blobs.push(b); return made.call(URL, b); };
  document.getElementById("download-pdb")?.click();
  for (let tick = 0; tick < 40 && blobs.length === 0; tick += 1) await idle(100);
  URL.createObjectURL = made;
  const pdb = blobs[0] ? await blobs[0].text() : "";
  const pred = (window.__lastPrediction && window.__lastPrediction()) || {};
  // 🔴 A TYPED ARRAY HAS TO ARRIVE AS ONE. JSON has no typed arrays, so this
  // used to flatten them to plain arrays - which LOOK right everywhere and
  // then are not: `download-all` reached `matrixRows`, which slices the PAE
  // with `values.subarray(...)`, and a remote fold's download died on
  // "values.subarray is not a function" while the picture beside it was
  // perfect. The kind travels with the numbers and `revivePrediction` puts it
  // back, so what the reader holds is what a local fold would have held.
  const predJson = JSON.stringify(pred, (key, value) =>
    (ArrayBuffer.isView(value) && !(value instanceof DataView))
      ? { __typed: value.constructor.name, v: Array.from(value) } : value);
  return {
    predJson,
    a3m: pred.a3m ?? null,
    confidence: pred.confidence ?? null,
    scores: pred.scores ?? null,
    chains: pred.chains ?? null,
    length: pred.length ?? null,
    status: statusText(),
    pdb,
    atoms: (pdb.match(/^ATOM|^HETATM/gm) || []).length,
  };
}

/**
 * A fold, asked for from the other machine and pressed here.
 *
 * Every line of this was a `cdp.evaluate` string in tools/colab_backend.py.
 * It is the same code in the place it belongs: the page driving its own
 * controls, where a `#predict` that has not been wired yet is something to
 * wait for rather than a race nobody can see from outside.
 */
async function runFold(request) {
  if (folding()) {
    return { error: "the runtime is already folding", status: statusText() };
  }
  const entities = request.entities ?? [{
    type: "protein", value: request.sequence ?? "", copies: 1,
    modifications: request.modifications ?? [],
  }];
  // 🔴 AN EMPTY REQUEST IS REFUSED HERE, NOT BY WAITING. With nothing to fold
  // the page keeps `#predict` disabled - correctly - and the wait below would
  // sit out its whole bound before saying so, with the reader watching a bar
  // that means nothing. The page's own words are what a person would read.
  if (entities.length === 0 || entities.every((e) => !String(e.value ?? "").trim())) {
    return { error: "nothing to fold: the request carried no sequence",
             status: statusText() };
  }
  window.__entityList?.set(entities);
  const controls = {
    "model-family": request.model ?? "af3",
    recycles: String(request.recycles ?? 3),
    "af3-count": String(request.steps ?? 25),
    // 🔴 THE CONTROL'S OWN VALUE, NOT THE RESOLVED MODE. A select silently
    // refuses a value it has no option for, and "single" is not one of them -
    // it left the control empty and the fold died with "unknown alignment
    // mode". What travels is what the reader's select said.
    "msa-mode": request.msa ?? "none",
  };
  for (const [id, value] of Object.entries(controls)) {
    const control = document.getElementById(id);
    if (control === null) continue;
    control.value = value;
    control.dispatchEvent(new Event("change", { bubbles: true }));
  }
  // The button comes up when the page has wired it and the entity list reads
  // as foldable; a minute is longer than either has ever taken and short
  // enough that a request the page will never accept says so.
  const button = document.getElementById("predict");
  for (let tick = 0; tick < 120 && (button === null || button.disabled); tick += 1) {
    await idle(500);
  }
  if (button === null || button.disabled) {
    return { error: "the fold button never came up", status: statusText() };
  }
  // 🔴 THE PAGE'S OWN CLOCK ON BOTH SIDES OF THE PRESS. What says the fold
  // ended is `__foldState.since` moving PAST the click, and two clocks
  // agreeing is not something to rest a completion test on.
  const pressed = Date.now();
  button.click();
  const deadline = Date.now() + Math.min((request.timeout ?? 300) * 1000, 1800_000);
  for (;;) {
    await idle(250);
    const state = window.__foldState ?? null;
    if (state !== null && state.running === false && state.since > pressed) break;
    if (Date.now() > deadline) return { error: "timed out", status: statusText() };
  }
  // 🔴 "READY" IS "IT CAN HAND ONE OVER", NOT "THE FOLD ENDED".
  // `loadIntoViewer` clears the object's frames and re-adds them, so the
  // moment after a fold is a settled status line over an EMPTY object and a
  // download button that writes nothing. Ask for the artefact until it exists.
  // 🔴 AND A FOLD THAT FAILED IS NOT WAITED FOR. The wait below exists for the
  // window where `loadIntoViewer` has cleared the object's frames and not yet
  // re-added them, which only happens on the way to a structure; a fold that
  // died - no weights, no network, a refused allocation - has nothing coming,
  // and waiting two minutes to say so is two minutes of a reader watching a
  // bar that has already lost.
  if (failed()) return { error: statusText(), status: statusText() };
  const until = Date.now() + 120_000;
  for (;;) {
    const out = await readBack();
    if (out.atoms > 0) return out;
    if (failed()) return { ...out, error: statusText() };
    if (Date.now() > until) return { ...out, error: "the page never produced a structure" };
    await idle(500);
  }
}

async function obey(command) {
  const { op, payload } = command;
  if (op === "ping") {
    // The transport's own check, and the only op that needs no GPU: it is what
    // tools/check-colab-bridge.py proves the two directions with.
    tapOut("pong", { at: Date.now(), folding: folding(), status: statusText() });
    return;
  }
  if (op === "stop") {
    // 🔴 STOPPING IS THE SAME BUTTON. `predict` is a toggle - it is how a
    // reader stops a fold - so there is no second control to keep in step.
    document.getElementById("predict")?.click();
    tapOut("stopped", { at: Date.now() });
    return;
  }
  if (op === "fold") {
    // 🔴 NOT AWAITED, OR NOTHING ELSE IS HEARD UNTIL THE FOLD ENDS - AND
    // `stop` IS THE COMMAND THAT ONLY MATTERS DURING ONE. The loop below
    // obeys in order and a fold is minutes long, so awaiting it here left the
    // reader's Stop sitting in the mailbox until the fold it was meant to
    // interrupt had finished on its own. A fold is a JOB; the loop goes on
    // listening while it runs, and the broker refuses a second one.
    tapOut("fold-begin", { at: Date.now() });
    void (async () => {
      let out;
      try {
        out = await runFold(payload ?? {});
      } catch (cause) {
        out = { error: String(cause && cause.message ? cause.message : cause) };
      }
      tapOut("result", out);
    })();
    return;
  }
  tapOut("status", `the runtime does not know the command "${op}"`);
}

/**
 * The runtime page's own loop: what has been asked of me since I last looked.
 *
 * A POLL AND NOT A SOCKET, because the broker is `http.server` and the thing
 * that would justify a socket - latency - is a tenth of a second against a
 * fold measured in minutes. What matters is that the EVENTS are pushed; a
 * command arriving 300 ms late is a button pressed 300 ms late.
 */
async function serveCommands() {
  // 🔴 FROM WHERE THE QUEUE STANDS NOW, NOT FROM ZERO. This page can be
  // reloaded - by the backend, by a crash, by anything - and a watermark that
  // restarts at zero obeys the whole session again: measured, a reloaded
  // runtime page re-ran a fold from ten minutes earlier, downloaded the
  // weights for it and reported it as the current one. What was asked before
  // this page existed was asked of a page that has already answered.
  let since = 0;
  try {
    const answer = await fetch(door("/out", "&head=1"));
    if (answer.ok) since = (await answer.json()).n ?? 0;
  } catch (cause) {
    /* the first poll below will simply start at zero, which is the old
       behaviour and is only wrong for a page that has been reloaded */
  }
  for (;;) {
    try {
      const answer = await fetch(door("/out", `&since=${since}`));
      if (answer.ok) {
        const said = await answer.json();
        since = said.n ?? since;
        for (const command of said.commands ?? []) await obey(command);
      }
    } catch (cause) {
      // The broker restarting is not this page's problem to solve; it is one
      // missed poll and the next one carries whatever was queued.
    }
    await idle(300);
  }
}

/* ----------------------------------------------------- the reader's two doors */

/**
 * The runtime's prediction, with its typed arrays back.
 *
 * 🔴 THE KINDS ARE NAMED RATHER THAN GUESSED, because guessing is what the
 * flattened form already did: "an array of numbers" is a Float32Array, a
 * Uint8Array or nothing in particular depending on which field it is, and
 * every reader downstream has its own opinion. The writer knows; it says.
 */
const TYPED = {
  Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
  Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array,
};

export function revivePrediction(json) {
  return JSON.parse(json, (key, value) => {
    if (value === null || typeof value !== "object") return value;
    const kind = TYPED[value.__typed];
    return (kind !== undefined && Array.isArray(value.v)) ? kind.from(value.v) : value;
  });
}

/** Ask the runtime for something. Returns the command's sequence number. */
export const remoteCommand = (op, payload) => ask("/in", { op, payload });

/** What the runtime has said since `since`. */
export async function remoteEvents(since, signal) {
  const answer = await fetch(door("/down", `&since=${since}`), { signal });
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
 * 🔴 AND THE HEARTBEAT IS THE ONE THE FOLD LOOP ALREADY USES: `runtimeSeen`
 * off `/down?head=1`, which is the runtime page's own command poll and costs
 * the broker nothing. `/health` is asked ONCE, for the card's name, because it
 * reaches over CDP to the browser on the other side.
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
  // browser on that machine holds the GPU for as long as it lives. What this
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

  // 🔴 ASKED UNTIL IT ANSWERS, AND THEN NOT AGAIN. `/health` reaches over CDP
  // to the browser on the other side, so it is not a thing to poll - and a
  // single attempt at load lost the race often enough to matter: the badge
  // read "Colab runtime" with no card, which is the one word that separates a
  // T4 from SwiftShader wearing its clothes.
  let card = "";
  let releases = false;
  const nameTheCard = async () => {
    if (card !== "") return;
    try {
      const health = await (await fetch(door("/health"))).json();
      const gpu = health.gpu ?? {};
      card = [gpu.vendor, gpu.architecture].filter(Boolean).join(" ");
      // 🔴 AND WHETHER DISCONNECT RELEASES THE MACHINE OR ONLY STOPS THE
      // SERVICE ON IT. On Colab it is both; on a runtime somebody is hosting
      // by hand there is no machine to hand back, and a button that promises
      // one either way is wrong half the time.
      releases = health.colabRuntime === true;
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
      // 🔴 TWENTY SECONDS IS THE FOLD LOOP'S OWN BOUND, and the two must agree:
      // a badge that still says connected while `followRemoteFold` is giving
      // up is the page telling a reader two things at once.
      // The badge asks the same question the fold loop does: silence alone is
      // a page that is busy, and the browser being gone is a runtime that is.
      const gone = (head2.runtimeSeen ?? 0) > 20000 && head2.browserAlive === false;
      badge.dataset.state = gone ? "gone" : "live";
      said.textContent = gone
        ? "Colab runtime · not answering"
        : `Colab runtime${card ? ` · ${card}` : ""}`;
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
    // 🔴 THE FOLD FIRST, THEN THE SERVICE. Stopping the page mid-fold leaves
    // the runtime's browser finishing a fold nobody will read.
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

/** Start whichever half of this page is. Called once, by web/app.js. */
export function installColabBridge() {
  if (colabRole() === "reader") {
    installColabStatus();
    return;
  }
  if (colabRole() !== "runtime") return;
  // 🔴 ANNOUNCED, SO THE BACKEND KNOWS THE PAGE IS UP WITHOUT ASKING IT.
  // `tools/colab_backend.py` waits for this rather than polling an internal
  // over CDP, which is the one thing that used to tie startup to the debugger.
  tapOut("runtime-ready", { at: Date.now(), href: location.href });
  void serveCommands();
}

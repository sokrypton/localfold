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

/** The broker is the server this page was served BY, so every route is relative. */
const door = (route, extra = "") => {
  const token = new URLSearchParams(location.search).get("t") ?? "";
  return `${route}?t=${encodeURIComponent(token)}${extra}`;
};

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
  const predJson = JSON.stringify(pred, (key, value) =>
    (ArrayBuffer.isView(value) && !(value instanceof DataView))
      ? Array.from(value) : value);
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

/** Start whichever half of this page is. Called once, by web/app.js. */
export function installColabBridge() {
  if (colabRole() !== "runtime") return;
  // 🔴 ANNOUNCED, SO THE BACKEND KNOWS THE PAGE IS UP WITHOUT ASKING IT.
  // `tools/colab_backend.py` waits for this rather than polling an internal
  // over CDP, which is the one thing that used to tie startup to the debugger.
  tapOut("runtime-ready", { at: Date.now(), href: location.href });
  void serveCommands();
}

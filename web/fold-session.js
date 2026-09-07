/**
 * The last session, kept so closing the tab does not lose it.
 *
 * 🔴 IT IS py2Dmol'S OWN SESSION, NOT A FORMAT OF OURS. `window.buildViewerState`
 * returns exactly what the Save button writes to a `.py2dmol.json`, and
 * `window.loadViewerState` is the reader that has always been there for a
 * dropped one - so the whole trajectory comes back, not just the answer, along
 * with the camera, the colour mode, the style, the side chains, the PAE, every
 * heatmap and the MSA. Writing our own would have been a second description of
 * the same viewer to keep in step, and it would have restored one frame where
 * this restores sixteen.
 *
 * Both were reachable only through a file: the builder downloaded its result
 * and the loader was not exported. Splitting the two upstream is the whole
 * change on that side - see `../py2Dmol/src/app/session.js`.
 *
 * 🔴 WHAT py2Dmol DOES NOT CARRY IS THE JOB, which is the half that is ours.
 * Its frames know coordinates and maps; nothing in them says which model ran,
 * against which sequence, with which alignment, or what the confidence head
 * said. That goes in under `localfold`, a key the loader ignores and a
 * round trip preserves.
 *
 * 🔴 ONE RECORD, DELIBERATELY. Not a history and not a second object picker:
 * py2Dmol already does multi-object - `#objectSelect`, prev/next and a Multi
 * overlay are all in the markup - and a saved-fold LIST is a second selector
 * for the same set, which is where this went wrong the first time. What the
 * page genuinely cannot do is survive a reload.
 */

const DB_NAME = "localfold-session";
const DB_VERSION = 1;
const STORE = "session";
const KEY = "current";

/**
 * 🔴 EVERY CALL IS GUARDED, AS localStorage IS IN app.js. IndexedDB is absent
 * in some configurations and THROWS rather than returning null in others -
 * permanent private browsing, site data blocked, a page opened as `file:`. A
 * page that cannot save one still folds, so every entry point here resolves to
 * a harmless empty value rather than rejecting.
 */
function open() {
  return new Promise((resolve) => {
    let request;
    try {
      request = globalThis.indexedDB?.open(DB_NAME, DB_VERSION);
    } catch {
      resolve(undefined);
      return;
    }
    if (request === undefined || request === null) {
      resolve(undefined);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(undefined);
    request.onblocked = () => resolve(undefined);
  });
}

/**
 * The job behind a session: what py2Dmol's state does not know.
 *
 * 🔴 pLDDT IS ABSENT, NOT ZERO, FOR A MODEL WITH NO CONFIDENCE HEAD. EF2-fast
 * stores no `confidence` object at all - on purpose, since an object of zeros
 * reads as the model's opinion - so this is `undefined` there and the line
 * renders without a score. Writing 0 is the mistake the archive was taught not
 * to make in its B-factor column.
 *
 * 🔴 AND THE CONFIDENCE MATRICES ARE NOT COPIED HERE. The PAE and the contact
 * map are already on the frames py2Dmol saves, which is where the panels read
 * them; a second copy in this object would double the file to restate what is
 * beside it. What this holds is the SUMMARY - the numbers the score card shows
 * and the frames do not carry.
 */
export function jobMeta({ stem, model, prediction, sequence, settings, entities,
                          msaOrigin, savedAt = Date.now() }) {
  const chainLengths = prediction?.chainLengths ?? [];
  const confidence = prediction?.confidence;
  return {
    stem,
    model,
    savedAt,
    sequence,
    settings,
    entities,
    msaOrigin,
    chainLengths,
    residues: chainLengths.reduce((total, length) => total + length, 0),
    // 🔴 THE STRUCTURE ITSELF, WHICH py2Dmol'S SESSION DOES NOT CARRY. Its
    // frames hold coordinates, element symbols and residue numbers - enough to
    // DRAW the fold and not the text the fold produced. Both download buttons
    // read `prediction.pdb`, so a restored session without it put the structure
    // on screen and greyed nothing out: "PDB" wrote a file with `undefined` in
    // it and "All" threw. Rebuilding the text from the frames is a second PDB
    // writer to keep in step with the first; keeping the one the fold wrote is
    // exact and, gzipped, costs almost nothing.
    pdb: prediction?.pdb,
    tokens: prediction?.tokens,
    // ...and everything `buildFoldArchive` reads, so "Download all" on a
    // restored session writes the same archive a live fold does.
    // 🔴 THE SUMMARY ONLY - THE MATRICES ARE ALREADY IN THE FRAMES. Storing
    // our own float copies of the PAE and the contact map wrote every pair
    // TWICE: once as py2Dmol's frame data, once here. Both are recoverable
    // from what the frames already hold, and the loss is bounded and small:
    //
    //   pae      2D float rows rounded to 1 decimal    -> 0.05 A
    //   contact  bytes, round(p * 255), vmin 0 vmax 1  -> 0.004
    //   plddt    per-residue, rounded to integers      -> 1
    //
    // The archive rounds everything to TWO decimals when it writes it, so the
    // contact grid is finer than what is written either way and costs nothing;
    // the PAE gives up one decimal place. pLDDT feeds only
    // `fraction_disordered`, whose threshold is 50, and the per-ATOM pLDDTs the
    // archive writes come off the stored PDB's B-factor column rather than
    // from here. n^2 per matrix is also the term that grows fastest with chain
    // length, so this is the copy worth not making.
    //
    // What cannot be recovered is what is not in a frame: the scalars.
    confidence: confidence === undefined ? undefined : {
      meanPlddt: confidence.meanPlddt,
      ptm: confidence.ptm,
      iptm: confidence.iptm,
      multimerScore: confidence.multimerScore,
      chainPairIptm: confidence.chainPairIptm,
      chainPtm: confidence.chainPtm,
      chainIptm: confidence.chainIptm,
      maxPredictedAlignedError: confidence.maxPredictedAlignedError,
    },
  };
}

/**
 * Replace the saved session.
 *
 * @returns {"quota"|object|undefined} the meta on success, "quota" when the
 *   origin is full, undefined when there is no store to write to.
 */
export async function saveSession(state) {
  const packed = await pack(state);
  const db = await open();
  if (db === undefined) return undefined;
  return new Promise((resolve) => {
    let transaction;
    try {
      transaction = db.transaction(STORE, "readwrite");
    } catch {
      db.close();
      resolve(undefined);
      return;
    }
    transaction.oncomplete = () => { db.close(); resolve(state); };
    transaction.onerror = () => { db.close(); resolve(undefined); };
    // 🔴 A FULL QUOTA IS REPORTED, NOT SWALLOWED. A page that says "saved" and
    // remembers nothing is worse than one that never offered to.
    transaction.onabort = () => {
      db.close();
      resolve(transaction.error?.name === "QuotaExceededError" ? "quota" : undefined);
    };
    transaction.objectStore(STORE).put(packed, KEY);
  });
}

/**
 * gzip, through the browser's own CompressionStream.
 *
 * 🔴 MEASURED: 188,767 bytes of session JSON become 42,137, a ratio of 4.48.
 * The payload is rounded decimal coordinates repeated over every frame of a
 * trajectory, which is about as compressible as text gets, and it grows with
 * BOTH the length of the chain and the number of sampler steps - py2Dmol's own
 * note records a 212 MB session for a 305,004-position structure. A stored
 * Uint8Array also skips IndexedDB's structured clone of a deep object graph.
 *
 * Falls back to the object itself where CompressionStream is missing, and the
 * reader tells the two apart by type rather than by a flag that could be wrong.
 */
async function pack(state) {
  if (typeof CompressionStream !== "function") return state;
  try {
    const raw = new TextEncoder().encode(JSON.stringify(state));
    const stream = new Blob([raw]).stream()
      .pipeThrough(new CompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    return state;
  }
}

async function unpack(stored) {
  // ...an object is a session written before this, or by a browser without
  // CompressionStream. Both are still readable, which is the point of
  // deciding by type.
  if (stored === undefined || stored === null) return undefined;
  if (!(stored instanceof Uint8Array)) return stored;
  try {
    const stream = new Blob([stored]).stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  } catch {
    return undefined;
  }
}

/** The saved session, or undefined: py2Dmol's state plus our `localfold` key. */
export async function readSession() {
  const db = await open();
  if (db === undefined) return undefined;
  return new Promise((resolve) => {
    let transaction;
    try {
      transaction = db.transaction(STORE, "readonly");
    } catch {
      db.close();
      resolve(undefined);
      return;
    }
    const request = transaction.objectStore(STORE).get(KEY);
    transaction.oncomplete = () => { db.close(); resolve(unpack(request.result)); };
    transaction.onerror = () => { db.close(); resolve(undefined); };
    transaction.onabort = () => { db.close(); resolve(undefined); };
  });
}

/** Forget it. */
export async function clearSession() {
  const db = await open();
  if (db === undefined) return false;
  return new Promise((resolve) => {
    const transaction = db.transaction(STORE, "readwrite");
    transaction.oncomplete = () => { db.close(); resolve(true); };
    transaction.onerror = () => { db.close(); resolve(false); };
    transaction.onabort = () => { db.close(); resolve(false); };
    transaction.objectStore(STORE).delete(KEY);
  });
}

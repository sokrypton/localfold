/**
 * What one trajectory frame costs the sampler, per step, on the critical path.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-frame.js --tokens=59
 *
 * 🔴 THE SAMPLER AWAITS onStep, SO EVERYTHING IN IT IS THE FOLD'S OWN TIME.
 * That was invisible when a denoiser call cost 760 ms and a frame cost a few;
 * a call is 176 ms now, so the same few are a fifth of it. This times the parts
 * that are plain JavaScript - the copies, the PDB text, the yield - which is
 * everything the page does per step except the superposition (py2Dmol's) and
 * the viewer's own re-render.
 *
 * WHAT IT FOUND, on a 59-residue chain: 0.73 ms a step, against a denoiser call
 * of 176. The PDB text and py2Dmol's own reparse of it are half of that and
 * neither is worth touching.
 *
 * 🔴 ONE PIECE IS STILL NOT MEASURED HERE, AND IT MUST NOT BE READ AS IF IT
 * WERE. The page's viewer comes from py2dmolLoadFiles and needs index.html's
 * DOM; py2Dmol.show attaches to a bare div but gives back a viewer with no
 * object to append frames to, so addFrame/setFrame cannot be timed. What CAN be
 * said is that `render` returns in about zero and the viewer's prototype
 * carries _scheduleSettle, ensureAnimationLoop and animate - so the redraw
 * looks scheduled rather than synchronous, and the sampler is not waiting on a
 * paint. "Looks" is doing real work in that sentence: to settle it, instrument
 * onFrame in web/app.js and read the numbers off a real fold.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { toPdb, normalFrom } from "../../src/af3/fold.js";
import { yieldToBrowser } from "../../src/runtime/yield.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const toPoints = (positions, count) => Array.from(
  { length: count }, (_, i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  const steps = Number(option(args, "steps", "40"));
  const sequence = Array.from({ length: tokens },
    (_, index) => ALPHABET[index % ALPHABET.length]).join("");
  const batch = featuriseProtein(sequence, {});
  const count = batch.tokens * batch.dense;

  const noise = normalFrom(3);
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < positions.length; i += 1) positions[i] = noise() * 12;

  const time = async (label, work) => {
    await work();                                   // warm
    const started = performance.now();
    for (let i = 0; i < steps; i += 1) await work();
    return [label, Number(((performance.now() - started) / steps).toFixed(3))];
  };

  // 🔴 THE VENDOR BUNDLE IS A CLASSIC SCRIPT, not a module, so it arrives
  // through a tag rather than an import. Only the parts that need no DOM can be
  // timed here: frameFromText and superpose are pure, the viewer's re-render is
  // not.
  await new Promise((resolve) => {
    const tag = document.createElement("script");
    tag.src = "/web/vendor/py2Dmol.full.min.js";
    tag.onload = resolve;
    tag.onerror = () => resolve();
    document.head.append(tag);
  });
  const api = window.py2Dmol;
  const pdb = toPdb(batch, positions, null);
  const points = toPoints(positions, count);
  const slots = [];
  for (let token = 0; token < batch.tokens; token += 1) slots.push(token * batch.dense + 1);

  // 🔴 SEQUENTIALLY, AND THE FIRST ATTEMPT WAS NOT. Running the timers inside a
  // Promise.all interleaves them, so every one measures the wall time of all of
  // them - six different pieces of work reported within 3% of each other, which
  // is the shape of that mistake rather than a finding.
  const rows = {};
  const measure = async (label, work) => {
    const [name, ms] = await time(label, work);
    rows[name] = ms;
  };
  await measure("3 array copies", () => {
    Float32Array.from(positions); Float32Array.from(positions);
    return Float32Array.from(positions);
  });
  await measure("toPoints (an array of 3-arrays)", () => toPoints(positions, count));
  await measure("toPdb", () => toPdb(batch, positions, null));
  await measure("yieldToBrowser", () => yieldToBrowser());
  if (api?.frameFromText !== undefined) {
    await measure("py2Dmol frameFromText", () => api.frameFromText(pdb));
  }
  if (api?.superpose !== undefined) {
    await measure("py2Dmol superpose",
      () => api.superpose(points, points, { from: slots, to: slots }));
  }

  // 🔴 AND THE RE-RENDER, WHICH IS THE ONE THAT COULD ACTUALLY MATTER.
  // py2Dmol.show attaches a viewer to any element, so the page's DOM is not
  // needed: a detached-but-attached div is enough to time what onFrame does per
  // step - parse the text, append the frame, select it, redraw.
  let viewerNote = "no viewer: py2Dmol.show unavailable";
  if (typeof api?.show === "function") {
    const mount = document.createElement("div");
    mount.style.width = "640px";
    mount.style.height = "480px";
    document.body.append(mount);
    try {
      const viewer = api.show(mount, pdb);
      const objectName = viewer?.objects?.[0]?.name;
      if (viewer?.addFrame !== undefined && objectName !== undefined) {
        await measure("py2Dmol addFrame + setFrame + render", () => {
          const frame = api.frameFromText(pdb);
          frame.name = frame.label = frame.title = `bench_${Math.random()}`;
          viewer.addFrame(frame, objectName);
          const object = viewer.objects?.find((entry) => entry.name === objectName);
          if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
          viewer.render("bench");
        });
        viewerNote = "viewer measured";
      } else {
        const proto = Object.getOwnPropertyNames(Object.getPrototypeOf(viewer ?? {}) ?? {});
        // Named rather than dumped: the prototype has ~400 methods and printing
        // them buries the one fact that matters.
        viewerNote = "show() gives no object to append frames to, so addFrame is"
          + " untimed here; render is scheduled: "
          + ["render", "_scheduleSettle", "ensureAnimationLoop", "animate"]
            .filter((name) => proto.includes(name)).join(", ");
        if (typeof viewer?.render === "function") {
          await measure("py2Dmol viewer.render", () => viewer.render("bench"));
          viewerNote += " | render measured";
        }
      }
    } catch (error) {
      viewerNote = `viewer failed: ${error.message}`;
    }
  }

  const total = Object.values(rows).reduce((a, b) => a + b, 0);
  return { tokens, atoms: count, msPerStep: rows,
           totalMsPerStep: Number(total.toFixed(2)), viewerNote };
}

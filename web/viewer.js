/**
 * The structure viewer, as both pages need it.
 *
 * WHAT THIS OWNS: building py2Dmol on the first frame of a run, appending the
 * rest, keeping the camera across rebuilds, and turning side chains on. What it
 * deliberately does NOT own is anything about the prediction - superposition,
 * morphing, mutation clicks, which residue is selected. Those differ between
 * the single-sequence page and the MSA page, and they live with the page.
 *
 * 🔴 EVERY RUN REBUILDS. The atom count changes with a mutated residue, so
 * frames cannot be shared between folds and the viewer has to be made again.
 * Everything below that looks like bookkeeping - the kept camera, the
 * generation counter - exists because of that one fact.
 */

/**
 * A CONSTANT, not the container's height. The container has to be taller than
 * the canvas - the play bar goes below it - so measuring the box and handing
 * that back as the canvas size guarantees the bar is pushed out of whatever
 * room was left. It was invisible for two sittings that way.
 */
export const VIEWER_CANVAS_HEIGHT = 420;

/**
 * @param {object} options
 * @param {HTMLElement} options.container emptied and rebuilt on each first frame
 * @param {number} [options.canvasHeight]
 * @param {(message: string) => void} [options.onFail] shown in place of the
 *   structure; defaults to writing the message into the container.
 */
export function createStructureViewer({ container, canvasHeight = VIEWER_CANVAS_HEIGHT, onFail }) {
  let renderer;
  let objectName;
  let frames = 0;
  let keptCamera;
  // ...WHAT A RUNNING ANIMATION CHECKS ITSELF AGAINST. A morph walks frames
  // over ~900 ms and a new fold can start inside that window; if it kept
  // writing it would be writing into a viewer that has been replaced. Every
  // reset and every appended frame bumps this, and the animation stops the
  // moment the number it captured is stale.
  let generation = 0;

  const fail = (message) => {
    if (onFail !== undefined) { onFail(message); return; }
    container.replaceChildren(Object.assign(document.createElement("p"), {
      textContent: `${message}. The PDB download still works.`,
    }));
  };

  return {
    /** The py2Dmol renderer, or undefined before the first frame lands. */
    get renderer() { return renderer; },
    get object() { return objectName; },
    get frames() { return frames; },
    get generation() { return generation; },
    get built() { return renderer !== undefined; },

    /** Stop any running animation without touching what is drawn. */
    cancelAnimations() { generation += 1; },

    /** Start a new run: the next pushed frame will build the viewer afresh. */
    reset() {
      // ...HANDED TO THE NEXT VIEWER, not thrown away. Without this the reader
      // is returned to a default view on every press of Fold.
      keptCamera = renderer ? renderer.viewerState : keptCamera;
      renderer = undefined;
      objectName = undefined;
      frames = 0;
      generation += 1;
    },

    /** Forget the camera too, for a run that should frame itself anew. */
    forgetCamera() { keptCamera = undefined; },

    paint() {
      if (renderer === undefined) return;
      renderer.setColor("plddt");
      renderer.setSidechainColor("hydrophobicity");
    },

    fail,

    /**
     * Append one frame, building the viewer on the first.
     *
     * @param {string} pdb
     * @param {(container: HTMLElement) => void} [onBuild] run once, after the
     *   viewer exists, for wiring the page hangs off it.
     * @returns {{built: boolean, frames: number}} `built` is true only on the
     *   call that created the viewer.
     */
    push(pdb, onBuild) {
      const api = window.py2Dmol;
      // ...any running morph is cancelled first: the next thing this does is
      // append a frame, and a morph replaces the last one.
      generation += 1;
      if (api === undefined) {
        if (frames === 0) fail("The bundled py2Dmol viewer did not load");
        return { built: false, frames };
      }
      let built = false;
      try {
        if (renderer === undefined) {
          // ...THE CAMERA THE READER LEFT, if there is one. A fresh viewer
          // flies to its own best view, and after a point mutation that view is
          // ALMOST the old one - near enough to read as a glitch, far enough to
          // lose someone who had turned to look at the site they just mutated.
          const camera = keptCamera;
          container.replaceChildren();
          renderer = api.show(container, pdb, {
            width: container.clientWidth || 640,
            height: canvasHeight,
            box: false,
            // `richardson` is what `cartoon` already resolved to - mol.js maps
            // both to the same preset - but naming it says which drawing this is.
            style: "richardson",
            // SHEETS WITH THEIR PLEAT. The preset's own default flattens them,
            // and an explicit 0 is honoured because _pick takes any finite
            // value >= 0 ahead of the preset. A beta sheet that twists reads as
            // a beta sheet; flattened it reads as a wide ribbon.
            rendering: { sheet_flat: 0 },
            // THE PLAY BAR, NOT PLAYBACK. `play` builds the transport controls;
            // `autoplay` would start them running, and is deliberately left off
            // - a structure that spins away on its own is harder to look at
            // than one that waits. The bar hides itself while there is a single
            // frame and reveals itself the moment a second is appended.
            play: true,
            // ...and do not orient when we are about to put a camera back
            orient: camera === undefined,
          });
          objectName = renderer.currentObjectName;
          if (camera) Object.assign(renderer.viewerState, camera);
          built = true;
        } else {
          renderer.addFrame(api.frameFromText(pdb), objectName);
        }
        // AFTER EVERY FRAME, not just the first. py2Dmol builds side-chain
        // atoms as real coordinates at frame-load time - its own note says
        // "only a frame load builds them" - so a frame appended later than the
        // call would come in bare. _setSidechains reports whether anything
        // changed and does nothing when it has not, so repeating it costs a
        // check per frame. It throws on a structure carrying no side-chain
        // atoms at all, which a poly-glycine legitimately is, and that is not a
        // reason to lose the fold.
        try { renderer.showSidechains(); } catch { /* backbone-only structure */ }
        frames += 1;
        if (built) { this.paint(); onBuild?.(container); }
        return { built, frames };
      } catch (error) {
        if (frames === 0) fail(error instanceof Error ? error.message : String(error));
        return { built: false, frames };
      }
    },

    /** Show a frame and draw it. Indices are zero-based. */
    show(index, reason) {
      if (renderer === undefined) return;
      renderer.setFrame(index);
      renderer.render(reason);
    },
  };
}

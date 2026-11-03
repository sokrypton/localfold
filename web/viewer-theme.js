/**
 * Keep the 3D viewer on the page's theme.
 *
 * py2Dmol reads one string to decide its look: "#000000" clears to black and picks its
 * dark selection outline. Its style presets write that field back to white, so the dark
 * theme answers the read rather than racing the writes, and hands the value the presets
 * wrote back when the theme goes light. Nothing here touches the vendored file.
 */
const DARK_BACKGROUND = "#000000";
const dressed = new WeakSet();

const isDark = () => document.documentElement.dataset.theme === "dark";

function dress(renderer) {
  if (dressed.has(renderer)) return;
  dressed.add(renderer);
  let written = renderer.backgroundColor;
  Object.defineProperty(renderer, "backgroundColor", {
    configurable: true,
    get: () => (isDark() ? DARK_BACKGROUND : written),
    set: (value) => { written = value; },
  });
}

function sync() {
  for (const { renderer } of Object.values(window.py2dmol_viewers ?? {})) {
    if (renderer === undefined) continue;
    dress(renderer);
    if (renderer.canvas) renderer.canvas.style.background = renderer.backgroundColor;
    renderer.render?.("theme");
  }
}

/** The viewer is built when the first frames land, which is the first this can see of it. */
export function installViewerTheme() {
  document.addEventListener("localfold-theme", sync);
  document.addEventListener("py2dmol-frame-change", () => {
    const missing = Object.values(window.py2dmol_viewers ?? {})
      .some(({ renderer }) => renderer !== undefined && !dressed.has(renderer));
    if (missing) sync();
  });
  sync();
}

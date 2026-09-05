import { devHasRows, devReport } from "./dev-log.js";

/**
 * The footer's "dev" button and the panel behind it.
 *
 * 🔴 IT IS BUILT WHEN IT IS FIRST OPENED, not when the page loads. A reader who
 * never presses it pays for one event listener; the panel's markup, and the
 * report that fills it, do not exist until asked for. tools/mobile-layout.py
 * measures a page whose rows are mostly `display: none` for exactly this
 * reason - something hidden that is still laid out is still a cost.
 */

let panel;
let body;

function build() {
  panel = document.createElement("div");
  panel.id = "dev-panel";
  panel.hidden = true;
  panel.style.cssText = "position: fixed; inset: auto 12px 12px 12px; max-width: 900px;"
    + " margin: 0 auto; max-height: 60vh; background: #111827; color: #e5e7eb;"
    + " border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); z-index: 60;"
    + " display: flex; flex-direction: column; font-size: 11px;";

  const head = document.createElement("div");
  head.style.cssText = "display: flex; align-items: center; gap: 8px; padding: 8px 12px;"
    + " border-bottom: 1px solid #374151; flex: 0 0 auto;";
  const title = document.createElement("strong");
  title.textContent = "Timing and memory";
  title.style.cssText = "flex: 1 1 auto; font-weight: 600;";
  head.append(title);

  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy";
  copy.className = "btn btn-grey btn-small";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(devReport());
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1200);
    } catch {
      // ...a clipboard a browser refuses is not an error worth a dialog; the
      // text is on screen and selectable either way.
      copy.textContent = "Select it";
      setTimeout(() => { copy.textContent = "Copy"; }, 1200);
    }
  });
  head.append(copy);

  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.className = "btn btn-grey btn-small";
  close.addEventListener("click", () => { panel.hidden = true; });
  head.append(close);

  body = document.createElement("pre");
  body.style.cssText = "margin: 0; padding: 10px 12px; overflow: auto; flex: 1 1 auto;"
    + " font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;"
    + " line-height: 1.45; white-space: pre; tab-size: 4;";
  panel.append(head, body);
  document.body.append(panel);
}

function refresh() {
  body.textContent = devHasRows()
    ? devReport()
    : "Nothing recorded yet - fold something and open this again.";
}

/** Put the button in the footer. Safe to call before or after the page loads. */
export function installDevPanel() {
  const links = document.getElementById("footer-links");
  if (links === null) return;
  const separator = document.createElement("span");
  separator.style.color = "#d1d5db";
  separator.textContent = "·";
  const button = document.createElement("button");
  button.type = "button";
  button.id = "dev-toggle";
  button.textContent = "dev";
  button.title = "Time and device memory, phase by phase";
  // ...a link's look rather than a button's: it sits in a row of links and a
  // grey pill there reads as something a reader is meant to press.
  button.style.cssText = "background: none; border: 0; padding: 0; margin: 0;"
    + " color: #3b82f6; font: inherit; cursor: pointer;";
  button.addEventListener("click", () => {
    if (panel === undefined) build();
    panel.hidden = !panel.hidden;
    if (!panel.hidden) refresh();
  });
  links.append(separator, button);
}

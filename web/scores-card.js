/**
 * The confidence card: pLDDT, pTM, ipTM and the ranking score, by element id.
 *
 * 🔴 ONE IMPLEMENTATION, TWO PAGES. This was inside web/app.js, which is
 * index.html's driver, and proteinhunter.html wanted the same card following
 * its own play bar. The choice was to copy forty lines of `document
 * .getElementById` into a second driver or to move them somewhere both can
 * reach; a copy would have been two places to remember when a metric is added,
 * and the metric that is easy to forget is the one that hides itself.
 *
 * 🔴 A MISSING ELEMENT IS NOT AN ERROR. The card is optional furniture - a
 * page that has no #metricIptm simply does not show ipTM - so every lookup is
 * guarded rather than asserted. That is what lets the same function serve a
 * page with four cells and a page with two.
 *
 * @param {{meanPlddt?: number, ptm?: number, iptm?: number,
 *          multimerScore?: number}} [confidence] absent hides the card
 */
export function updateScoresCard(confidence) {
  const box = document.getElementById("predictionScoresBox");
  if (!box) return;
  if (!confidence) {
    box.style.display = "none";
    return;
  }
  box.style.display = "flex";

  const set = (id, value, digits) => {
    const node = document.getElementById(id);
    if (node) node.textContent = value === undefined ? "-" : Number(value).toFixed(digits);
  };
  // A cell that has no number hides rather than showing a dash: a monomer has
  // no interface, and "ipTM -" reads as a measurement that failed.
  const cell = (id, shown) => {
    const node = document.getElementById(id);
    if (node) node.style.display = shown ? "flex" : "none";
  };
  const real = (value) => value !== undefined && !Number.isNaN(Number(value));

  set("metricMeanPlddt", confidence.meanPlddt, 1);
  set("metricPtm", confidence.ptm, 2);

  cell("metricIptmCell", real(confidence.iptm));
  if (real(confidence.iptm)) set("metricIptm", confidence.iptm, 2);

  // AlphaFold-Multimer's ranking score, which is what orders a set of
  // predictions rather than describing one.
  const ranking = confidence.multimerScore
    ?? (real(confidence.iptm) && confidence.ptm !== undefined
      ? 0.8 * Number(confidence.iptm) + 0.2 * Number(confidence.ptm)
      : undefined);
  cell("metricMultimerCell", real(ranking));
  if (real(ranking)) set("metricMultimer", ranking, 2);
}

/**
 * Keep the card on whatever frame the play bar is showing.
 *
 * 🔴 py2Dmol HAS NO FRAME-CHANGE CALLBACK, so this polls - which is what
 * web/app.js already does for the same reason and at the same interval. The
 * play button runs its own loop and the slider moves under the mouse; neither
 * announces itself, and a card that only updated when the page changed the
 * frame would freeze the moment a reader touched the bar.
 *
 * @param {() => {renderer: object|undefined, object: string|undefined}} source
 *   read fresh each tick: the viewer is rebuilt between runs.
 * @param {number} [interval]
 * @returns {() => void} stops the poll
 */
export function followActiveFrame(source, interval = 50) {
  let last;
  const timer = setInterval(() => {
    try {
      const { renderer, object } = source();
      const data = renderer?.objectsData?.[object];
      const index = data?.currentFrame ?? renderer?.currentFrame;
      if (index === undefined || index === last) return;
      last = index;
      updateScoresCard(data?.frames?.[index]?.confidence);
    } catch { /* a viewer mid-rebuild is not worth a message every 50 ms */ }
  }, interval);
  return () => clearInterval(timer);
}

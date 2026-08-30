/**
 * Drawing confidence: the four bands, the per-residue plot, and the legends.
 *
 * BOTH PAGES REPORT pLDDT THE SAME WAY, and they had better keep doing so - a
 * reader who learns the colours on one page and finds them meaning something
 * else on the other has been taught a falsehood by the interface. The bands
 * here are the same four AlphaFold itself uses, and the viewer paints the
 * cartoon from py2Dmol's own `plddt` colouring, so this file's job is to make
 * the plot and the legend agree with what the structure already shows.
 */
import { HYDROPHOBICITY_BANDS } from "./hydrophobicity.js";

/** The four AlphaFold confidence bands, most confident first. */
export const PLDDT_BANDS = [
  { color: "#187bd1", min: 90, label: "≥90 very high" },
  { color: "#56b9dc", min: 70, label: "70–90 confident" },
  { color: "#f2c94c", min: 50, label: "50–70 low" },
  { color: "#ef6a62", min: 0, label: "<50 very low" },
];

/** @param {number} value pLDDT, 0-100 */
export function plddtColor(value) {
  for (const band of PLDDT_BANDS) if (value >= band.min) return band.color;
  return PLDDT_BANDS[PLDDT_BANDS.length - 1] .color;
}

/**
 * A row of labelled colour chips.
 *
 * @param {HTMLElement} host emptied and refilled
 * @param {Array<{color?: string, hex?: string, label: string}>} bands py2Dmol's
 *   hydrophobicity bands say `hex` where these say `color`; both are accepted so
 *   the two legends can be drawn by one function.
 */
export function swatches(host, bands) {
  host.replaceChildren(...bands.map((band) => {
    const item = document.createElement("span");
    item.className = "swatch";
    item.style.setProperty("--swatch", band.color ?? band.hex);
    item.textContent = band.label;
    return item;
  }));
}

/**
 * Both legends, for a page that has both hosts.
 * Missing hosts are skipped rather than thrown over: the MSA page and the
 * single-sequence page do not necessarily show the same legends.
 */
export function drawLegends({ plddt, hydropathy }) {
  if (plddt != null) swatches(plddt, PLDDT_BANDS);
  if (hydropathy != null) swatches(hydropathy, HYDROPHOBICITY_BANDS);
}

/**
 * The per-residue confidence bar chart.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {ArrayLike<number>} values one pLDDT per residue
 */
export function drawPlddt(canvas, values) {
  const context = canvas.getContext("2d");
  if (context === null) return;
  const { width, height } = canvas;
  const left = 40; const bottom = 26; const top = 14; const right = 10;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#07110f"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "#24413a"; context.font = "11px DM Mono";
  for (const tick of [0, 50, 70, 90, 100]) {
    const y = top + (100 - tick) / 100 * (height - top - bottom);
    context.beginPath(); context.moveTo(left, y); context.lineTo(width - right, y); context.stroke();
    context.fillStyle = "#94aaa1"; context.fillText(String(tick), 8, y + 4);
  }
  const barWidth = (width - left - right) / values.length;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.max(0, Math.min(100, values[index]));
    const barHeight = value / 100 * (height - top - bottom);
    context.fillStyle = plddtColor(value);
    context.fillRect(left + index * barWidth, height - bottom - barHeight, Math.max(1, barWidth), barHeight);
  }
  context.fillStyle = "#94aaa1";
  context.fillText("pLDDT per residue", left, height - 8);
}

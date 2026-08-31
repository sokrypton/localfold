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

/** The py2Dmol rainbow pLDDT confidence bands. */
export const PLDDT_BANDS = [
  { color: "#4040ff", min: 90, label: "≥90 very high" },
  { color: "#40ffff", min: 80, label: "80 confident" },
  { color: "#40ff40", min: 70, label: "70 medium" },
  { color: "#ffff40", min: 60, label: "60 low" },
  { color: "#ff4040", min: 0, label: "<50 very low" },
];

/** @param {number} value pLDDT, 0-100 */
export function plddtColor(value) {
  const t = Math.max(0, Math.min(1, (value - 50) / 40));
  if (t <= 0.25) {
    const s = t / 0.25;
    return `rgb(255, ${Math.round(64 + s * 191)}, 64)`;
  } else if (t <= 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgb(${Math.round(255 - s * 191)}, 255, 64)`;
  } else if (t <= 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgb(64, 255, ${Math.round(64 + s * 191)})`;
  } else {
    const s = (t - 0.75) / 0.25;
    return `rgb(64, ${Math.round(255 - s * 191)}, 255)`;
  }
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
  const left = 32; const bottom = 20; const top = 10; const right = 8;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "#e2e8f0"; context.lineWidth = 1; context.font = "10px ui-monospace, SFMono-Regular, monospace";
  for (const tick of [0, 50, 70, 90, 100]) {
    const y = top + (100 - tick) / 100 * (height - top - bottom);
    context.beginPath(); context.moveTo(left, y); context.lineTo(width - right, y); context.stroke();
    context.fillStyle = "#94a3b8"; context.fillText(String(tick), 4, y + 3);
  }
  const barWidth = (width - left - right) / values.length;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.max(0, Math.min(100, values[index]));
    const barHeight = value / 100 * (height - top - bottom);
    context.fillStyle = plddtColor(value);
    context.fillRect(left + index * barWidth, height - bottom - barHeight, Math.max(1, barWidth), barHeight);
  }
  context.fillStyle = "#64748b";
  context.fillText("pLDDT per residue", left, height - 5);
}

/**
 * The Predicted Aligned Error (PAE) heatmap with interactive crosshair support.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {ArrayLike<number> | Array<Array<number>>} paeData values in angstroms (0 to ~31.75)
 * @param {number} length sequence length
 * @param {{row: number, col: number} | null} [crosshair] highlighted pair
 */
export function drawPae(canvas, paeData, length, crosshair = null) {
  const context = canvas.getContext("2d");
  if (context === null || !paeData || length === 0) return;
  const { width, height } = canvas;
  const imgData = context.createImageData(length, length);
  const data = imgData.data;
  const isNested = Array.isArray(paeData) && Array.isArray(paeData[0]);

  for (let r = 0; r < length; r += 1) {
    for (let c = 0; c < length; c += 1) {
      const val = isNested ? paeData[r][c] : paeData[r * length + c];
      const t = Math.max(0, Math.min(1, val / 30));
      // AlphaFold PAE colormap: dark teal/green (13, 92, 58) -> white (255, 255, 255)
      const idx = (r * length + c) * 4;
      data[idx] = Math.round(13 + (255 - 13) * t);
      data[idx + 1] = Math.round(92 + (255 - 92) * t);
      data[idx + 2] = Math.round(58 + (255 - 58) * t);
      data[idx + 3] = 255;
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = length;
  offscreen.height = length;
  const offCtx = offscreen.getContext("2d");
  if (offCtx !== null) {
    offCtx.putImageData(imgData, 0, 0);
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, width, height);
    context.drawImage(offscreen, 0, 0, width, height);
  }

  if (crosshair !== null && crosshair.row >= 0 && crosshair.col >= 0) {
    const x = Math.floor(((crosshair.col + 0.5) / length) * width);
    const y = Math.floor(((crosshair.row + 0.5) / length) * height);
    context.save();
    context.strokeStyle = "rgba(15, 23, 42, 0.7)";
    context.lineWidth = 1;
    context.setLineDash([2, 2]);

    // Horizontal line
    context.beginPath();
    context.moveTo(0, y + 0.5);
    context.lineTo(width, y + 0.5);
    context.stroke();

    // Vertical line
    context.beginPath();
    context.moveTo(x + 0.5, 0);
    context.lineTo(x + 0.5, height);
    context.stroke();

    // Target intersection indicator
    context.setLineDash([]);
    context.fillStyle = "#ef4444";
    context.beginPath();
    context.arc(x + 0.5, y + 0.5, 3.5, 0, 2 * Math.PI);
    context.fill();
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1;
    context.stroke();
    context.restore();
  }
}

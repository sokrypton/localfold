/**
 * How many channel groups the sequential triangle takes.
 *
 * Each group reads the whole pair to project its channels, so fewer groups are faster, and a and b for one group are
 * 2/groups of a pair tensor, so fewer groups hold more. Neither end is right for everyone, and the answer is the device's:
 * as few as fit what it has left, which is two on a card with room and eight where a fold only just fits.
 */
const MIB = 1024 * 1024;

/** The choices, fewest first. */
export const SEQUENTIAL_GROUP_CHOICES = [2, 4, 8];
/** The most: what a fold's memory is planned around, because no fold holds more than this needs and still has to fit. */
export const MOST_SEQUENTIAL_GROUPS = SEQUENTIAL_GROUP_CHOICES[SEQUENTIAL_GROUP_CHOICES.length - 1];
/** Bytes of each of the track's two row chunks, which also sets how long a dispatch over a chunk's rows holds the card. */
export const SEQUENTIAL_CHUNK_BYTES = 32 * MIB;
/** What the choice leaves free: a stack holds more than the track (the grid's bias, a split transition's buffers). */
export const GROUPS_MARGIN_BYTES = 512 * MIB;

/**
 * The pieces to dispatch a kernel that treats each row alone in, so that no one dispatch holds the device for long. A row's
 * answer does not depend on which piece it fell in. `rowWork`: the kernel's multiply-adds a row. `tile`: what a piece's rows
 * must be a multiple of, so that no workgroup straddles two.
 */
export function rowPieces(rows, rowWork, stepWork, tile = 1) {
  const step = Math.max(tile, Math.floor(stepWork / rowWork / tile) * tile);
  const pieces = [];
  for (let first = 0; first < rows; first += step) pieces.push({ first, count: Math.min(step, rows - first) });
  return pieces;
}

/**
 * The fewest groups that fit. `residentBytes` is what the device holds already, the pair among it; beside that the track
 * adds the contraction's output, a and b, and the row chunks. a and b share one buffer, so they also have to fit in one.
 *
 * 🔴 TWO LIMITS, AND ONLY THE BUFFER IS REFUSED HERE. `maxBufferBytes` is the device's: a and b past it cannot be created,
 * so a pair whose share at the most groups does not fit one buffer is refused by name rather than by a validation error
 * naming a buffer. A fold over the BUDGET takes the most groups, the least this track can hold, and goes on: the
 * allocator holds the budget itself (device-memory.js, a GpuMemoryBudgetError naming the tensor, which the page offers to
 * lift), and this estimate carries a 512 MiB margin, so refusing here would refuse folds the allocator admits.
 */
export function fewestSequentialGroups({ pairBytes, residentBytes = 0, budgetBytes, maxBufferBytes = Infinity,
                                         chunkBytes = SEQUENTIAL_CHUNK_BYTES }) {
  const fits = (groups) => {
    const ab = 2 * pairBytes / groups;
    return ab <= maxBufferBytes && (budgetBytes === undefined || budgetBytes === null
      || residentBytes + pairBytes + ab + 2 * chunkBytes + GROUPS_MARGIN_BYTES <= budgetBytes);
  };
  if (2 * pairBytes / MOST_SEQUENTIAL_GROUPS > maxBufferBytes) {
    throw new RangeError(`a pair of ${pairBytes} bytes: a and b for one of ${MOST_SEQUENTIAL_GROUPS} channel groups are `
      + `${2 * pairBytes / MOST_SEQUENTIAL_GROUPS} bytes, past this device's largest buffer (${maxBufferBytes})`);
  }
  return SEQUENTIAL_GROUP_CHOICES.find(fits) ?? MOST_SEQUENTIAL_GROUPS;
}

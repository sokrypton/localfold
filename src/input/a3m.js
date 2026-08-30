const ALIGNED_RESIDUES = /^[ACDEFGHIKLMNPQRSTVWYX-]+$/;

export function parseA3m(text) {
  const descriptions = [];
  const rawSequences = [];
  let current = -1;
  for (const sourceLine of text.split(/\r?\n/)) {
    const line = sourceLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    if (line.startsWith(">")) {
      const description = line.slice(1).trim();
      if (description === "") throw new Error("A3M contains an empty FASTA header");
      descriptions.push(description);
      rawSequences.push("");
      current += 1;
      continue;
    }
    if (current < 0) throw new Error("A3M sequence data appears before the first FASTA header");
    if (/\s/.test(line)) throw new Error(`A3M sequence ${descriptions[current]} contains whitespace`);
    rawSequences[current] += line;
  }
  if (rawSequences.length === 0) throw new Error("A3M contains no sequences");

  const sequences = [];
  const deletionMatrix = [];
  for (let row = 0; row < rawSequences.length; row += 1) {
    const raw = rawSequences[row];
    if (raw === "") throw new Error(`A3M sequence ${descriptions[row]} is empty`);
    let aligned = "";
    let insertionCount = 0;
    const deletions = [];
    for (const residue of raw) {
      if (residue >= "a" && residue <= "z") {
        insertionCount += 1;
      } else {
        const upper = residue.toUpperCase();
        if (!ALIGNED_RESIDUES.test(upper)) {
          throw new Error(`A3M sequence ${descriptions[row]} contains invalid residue ${JSON.stringify(residue)}`);
        }
        aligned += upper;
        deletions.push(insertionCount);
        insertionCount = 0;
      }
    }
    sequences.push(aligned);
    deletionMatrix.push(deletions);
  }

  const length = sequences[0] .length;
  if (length === 0 || sequences[0] .includes("-")) {
    throw new Error("the first A3M sequence must be a non-empty, ungapped query");
  }
  for (let row = 0; row < sequences.length; row += 1) {
    if (sequences[row] .length !== length) {
      throw new Error(
        `A3M row ${descriptions[row]} has aligned length ${sequences[row] .length}; expected ${length}`,
      );
    }
  }
  return {
    query: sequences[0],
    descriptions,
    sequences,
    deletionMatrix,
    depth: sequences.length,
    length,
  };
}

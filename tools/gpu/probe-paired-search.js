/**
 * Does the PAIRED MMseqs2 search work for a two-entity complex?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-paired-search.js
 *
 * 🔴 IT GOES TO THE NETWORK AND DOES NO GPU WORK. It exists because
 * probe-designed-binder-sampler.js died with Chrome exiting cleanly and not one
 * progress line printed - so the failure is before or inside the search, and a
 * probe that also folds cannot tell which. A homodimer never reaches this path:
 * `wantsPairing` needs more than one UNIQUE sequence, so every aligned probe
 * here until now searched unpaired only.
 *
 * It reports shapes rather than sequences: what came back, how deep, and
 * whether the two halves line up.
 */
import { generateMmseqs2ComplexMsa } from "../../src/input/mmseqs2-api.js";
import { af3MsaFromA3m } from "../../src/af3/msa-features.js";

const VHH = "EVQLVESGGGLVQPGGSLRLSCAASGDTSFIIAMAWYRQAPGKGRELVAGLNRLTSSISYADSVKG"
  + "RFTISRDNAKNTLYLQMNSLRPEDTAVYYCAAARVLGGTTERAWGQGTLVTVSS";
const S100A4 = "SMACPLEKALDVMVSTFHKYSGKEGDKFKLNKSELKELLTRELPSFLGKRTDEAAFQKLMSNLDSN"
  + "RDNEVDFQEYCVFLSCIAMMCNEFFEGFPDKQPRKK";

const rows = (text) => (typeof text === "string"
  ? text.split("\n").filter((line) => line.startsWith(">")).length : null);

export async function main(device, args) {
  const steps = [];
  try {
    const search = await generateMmseqs2ComplexMsa([VHH, S100A4], {
      model: "af3",
      onProgress: (progress) => {
        steps.push(progress);
        console.log("[msa]", JSON.stringify(progress));
      },
    });
    // 🔴 THE CROP IS THE SUSPECT, NOT THE SEARCH. `maxSequences` crops, but
    // af3MsaFromA3m parses the WHOLE alignment first and computes the profile
    // over the full unpaired block before deduplication - so an 11712-row
    // result is materialised in full whatever the cap says.
    const alignment = search.blocks ?? { unpaired: search.a3m };
    const built = {};
    for (const maxSequences of [128, 512]) {
      const before = performance.now();
      const rowsBuilt = af3MsaFromA3m(alignment, { maxSequences });
      built[`max ${maxSequences}`] = {
        depth: rowsBuilt.depth,
        unpairedFrom: rowsBuilt.unpairedFrom,
        profileRows: rowsBuilt.profileMsa?.length ?? null,
        ms: Math.round(performance.now() - before),
      };
    }
    return {
      ok: true,
      progressSeen: steps.length,
      built,
      pairedDepth: search.pairedDepth,
      mergedRows: rows(search.a3m),
      blockRows: {
        paired: rows(search.blocks?.paired),
        unpaired: rows(search.blocks?.unpaired),
      },
      chainRows: (search.chainA3ms ?? []).map(rows),
    };
  } catch (cause) {
    // 🔴 REPORTED, NOT THROWN. A rejection out of main gives the harness
    // "Chrome exited before reporting", which names nothing.
    return {
      ok: false,
      progressSeen: steps.length,
      lastProgress: steps[steps.length - 1] ?? null,
      error: String(cause?.message ?? cause),
      stack: String(cause?.stack ?? "").split("\n").slice(0, 4),
    };
  }
}

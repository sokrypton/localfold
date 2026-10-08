// The seeded generators a fold and its featurisation share - out of src/af3/fold.js (which re-exports them).

/**
 * A reproducible uniform in (0, 1), so a seed names a choice.
 *
 * Split out of normalFrom rather than written twice: the MSA subsample needs
 * uniforms and the sampler needs deviates, and two generators would mean two
 * things a seed could mean.
 */
export function uniformFrom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
}

/** A reproducible normal deviate, so a seed names a structure. */
export function normalFrom(seed) {
  const uniform = uniformFrom(seed);
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

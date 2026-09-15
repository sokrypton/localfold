/**
 * The gate that says a fold is a chain.
 *
 * 🔴 THE RULE IS TESTED HERE BECAUSE THREE OF THE FOUR TOOLS THAT USE IT
 * CANNOT BE RUN ON EVERY BOX. `fold-af2.js` needs the monomer bundle,
 * `fold.js` an AF3 one, `fold-opendde.js` OpenDDE's and `fold-esmfold2.js`
 * three gigabytes of ESM-C - so the wiring is checked on device wherever the
 * weights are and the BAND is checked here, always. The numbers below are the
 * measured ones: docs/AF2.md's healthy folds and the 825-residue collapse that
 * walked through an ungated tool for a whole campaign.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
  CHAIN_GEOMETRY_BANDS, assertChainGeometry, chainGeometryOf, chainGeometryVerdict,
} from "../tools/gpu/chain-geometry.js";

describe("the chain geometry gate", () => {
  it("passes every healthy fold measured", () => {
    // docs/AF2.md: median 3.485 to 3.972, worst 1.69 to 4.55.
    for (const [caca, worstCaca] of [[3.485, 1.69], [3.972, 4.55], [3.80, 3.80],
                                     [3.82, 4.10], [3.485, 4.55]]) {
      assert.equal(chainGeometryVerdict({ caca, worstCaca }).ok, true,
        `${caca} / ${worstCaca} should pass`);
    }
  });

  it("fails the 825-residue collapse, which pLDDT called 69.31", () => {
    const verdict = chainGeometryVerdict({ caca: 1.44, worstCaca: 0.06 }, { plddt: 69.31 });
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /not a chain/);
    // ...and it says pLDDT is not the gate, in the message, where it is read.
    assert.match(verdict.reason, /69\.31/);
    assert.match(verdict.reason, /not a correctness gate/);
  });

  it("fails every other broken fold measured", () => {
    // docs/AF2.md: median 1.44 to 3.41, worst 0.06 or 7.73 to 70.45.
    for (const [caca, worstCaca] of [[1.44, 0.06], [3.41, 7.73], [3.60, 70.45],
                                     [2.00, 3.80]]) {
      assert.equal(chainGeometryVerdict({ caca, worstCaca }).ok, false,
        `${caca} / ${worstCaca} should fail`);
    }
  });

  it("catches one link thrown across the box, which the median cannot see", () => {
    // A chain that is right everywhere but one bond. The median is perfect.
    const spacing = new Array(200).fill(3.80);
    spacing[100] = 41.7;
    const geometry = chainGeometryOf(spacing);
    assert.equal(geometry.caca, 3.80);
    assert.equal(geometry.worstCaca, 41.7);
    assert.equal(chainGeometryVerdict(geometry).ok, false);
  });

  it("skips a fold with no alpha carbons rather than passing on a NaN", () => {
    // 🔴 NaN > x IS FALSE IN BOTH DIRECTIONS, so a gate written as a bare
    // comparison would pass a ligand-only fold silently. It has to say so.
    const verdict = chainGeometryVerdict({ caca: NaN, worstCaca: NaN });
    assert.equal(verdict.ok, true);
    assert.equal(verdict.skipped, true);
    assert.equal(chainGeometryOf([]).caca !== chainGeometryOf([]).caca, true);
  });

  it("throws unless the caller passed --allow-broken-geometry", () => {
    const broken = { caca: 1.44, worstCaca: 0.06 };
    assert.throws(() => assertChainGeometry(broken), /not a chain/);
    assert.doesNotThrow(() => assertChainGeometry(broken, { allow: true }));
  });

  it("takes the median of the spacings and the value furthest from 3.8", () => {
    const geometry = chainGeometryOf([3.6, 3.8, 3.9, 4.9, 2.1]);
    assert.equal(geometry.caca, 3.8);
    // 2.1 is 1.7 away and 4.9 is 1.1; the worst is the further one, either side.
    assert.equal(geometry.worstCaca, 2.1);
  });

  it("keeps the bands where the measurements put them", () => {
    assert.deepEqual(CHAIN_GEOMETRY_BANDS,
      { medianLow: 3.4, medianHigh: 4.2, worstFrom38: 2.8 });
  });
});

// 🔴 THE RULE LIVES IN src/ NOW, BECAUSE THE PAGE COULD NOT REACH tools/.
// Every command-line fold gated on this and the SITE ran no geometry check at
// all - the same shape as LOCALFOLD_STOCK_FLAGS, where the configuration every
// gate checks was not the one that ships. Measured: intellifold2 in Flow
// returns a fold this rule REFUSES on 1 seed in 6 (CA median 4.255 A against
// 3.80) with pLDDT 83.30 beside it.
describe("one rule, reachable from both the tools and the page", () => {
  it("the tools' wrapper and src/ agree, because there is one implementation",
    async () => {
      const shared = await import("../src/af3/chain-geometry.js");
      const tool = await import("../tools/gpu/chain-geometry.js");
      assert.equal(tool.chainGeometryVerdict, shared.chainGeometryVerdict,
        "the tools re-export the rule rather than carrying a second copy");
      assert.deepEqual(tool.CHAIN_GEOMETRY_BANDS, shared.CHAIN_GEOMETRY_BANDS);
    });

  it("refuses the intellifold2 flow fold that the page used to draw", async () => {
    const { chainGeometryVerdict } = await import("../src/af3/chain-geometry.js");
    const verdict = chainGeometryVerdict({ caca: 4.255, worstCaca: 4.70 },
      { plddt: 83.30 });
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /not a chain/);
    // ...and it names the pLDDT, because that is the number that said otherwise.
    assert.match(verdict.reason, /83\.30/);
  });

  it("...and rosettafold3's flow fold, which is the collapsed kind", async () => {
    const { chainGeometryVerdict } = await import("../src/af3/chain-geometry.js");
    assert.equal(chainGeometryVerdict({ caca: 3.071, worstCaca: 0.23 }).ok, false);
  });

  it("passes the diffusion folds those two models actually ship", async () => {
    const { chainGeometryVerdict } = await import("../src/af3/chain-geometry.js");
    for (const good of [{ caca: 3.81, worstCaca: 3.85 },
                        { caca: 3.76, worstCaca: 3.69 },
                        { caca: 3.88, worstCaca: 4.08 }]) {
      assert.equal(chainGeometryVerdict(good).ok, true, JSON.stringify(good));
    }
  });

  // 🔴 ONLY THE TOOLS' WRAPPER MENTIONS A FLAG. "Pass --allow-broken-geometry"
  // is advice for a terminal; the page shows the same verdict without it.
  it("the CLI hint is the wrapper's, not the rule's", async () => {
    const { chainGeometryVerdict } = await import("../src/af3/chain-geometry.js");
    const { assertChainGeometry } = await import("../tools/gpu/chain-geometry.js");
    const bare = chainGeometryVerdict({ caca: 1.0, worstCaca: 1.0 });
    assert.ok(!bare.reason.includes("--allow-broken-geometry"));
    const thrown = assertChainGeometry({ caca: 1.0, worstCaca: 1.0 }, { allow: true });
    assert.match(thrown.reason, /--allow-broken-geometry/);
  });
});

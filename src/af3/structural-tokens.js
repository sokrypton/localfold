/**
 * OpenDDE's structural tokens: a REGROUPING of AlphaFold 3's atoms.
 *
 * 🔴 OpenDDE DOES NOT FOLD THE TOKENS THE TRUNK RAN ON. Between the trunk and
 * the diffusion it re-tokenises each standard protein residue into TWO tokens -
 * a backbone one (N, CA, C, O, OXT; representative atom CA) and a sidechain one
 * (everything else; representative atom CB) - and runs the diffusion and its
 * own confidence head on that set. Glycine, and any residue whose split would
 * leave one side empty, stays a single backbone token. Nucleotides split
 * backbone/base the same way. Everything else - a ligand, a modified residue,
 * a lone atom - is one role-0 token per atom, which is what AlphaFold 3's
 * featuriser already produces for them.
 *
 * 🔴 AND IT IS A REGROUPING, WHICH IS THE WHOLE REASON THIS IS AN ADAPTER AND
 * NOT A SECOND FEATURISER. The ATOMS do not change: their reference positions,
 * elements, charges and names are the same values, and only the (token, slot)
 * they sit in moves. So everything here is a gather through one mapping, and
 * the properties worth testing are conservation laws - every real atom appears
 * exactly once, and no atom changes its identity.
 *
 * 🔴 AND THE REFERENCE SPACE IS THE RESIDUE'S, NOT THE TOKEN'S. AlphaFold 3
 * keys `ref_space_uid` on `(chain_id, res_id)` - upstream's own
 * `features.py` - so a residue's backbone and sidechain tokens SHARE it. The
 * natural-looking "one space per token" would give the two halves of one
 * residue different spaces, and the atom encoder uses the uid to decide whether
 * a PAIR of atoms may compare reference coordinates at all: every side chain
 * would stop seeing its own backbone. Nothing about that is visible in a shape.
 */

import { atomGathers } from "./featurise.js";

/** opendde/data/tokenizer.py's STRUCTURAL_TOKEN_ROLES. */
export const ROLE_ATOM = 0;
export const ROLE_PROTEIN_BB = 1;
export const ROLE_PROTEIN_SC = 2;
export const ROLE_DNA_BB = 3;
export const ROLE_DNA_BASE = 4;
export const ROLE_RNA_BB = 5;
export const ROLE_RNA_BASE = 6;
export const ROLES = 7;
/** A token with no twin; NO_TWIN upstream. */
export const NO_TWIN = -1;

const PROTEIN_BACKBONE = new Set(["N", "CA", "C", "O", "OXT"]);
const NUCLEIC_BACKBONE = new Set([
  "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "C5'", "C4'", "O4'",
  "C3'", "O3'", "C2'", "O2'", "C1'", "O5*", "C5*", "C4*", "O4*", "C3*", "O3*",
  "C2*", "O2*", "C1*", "O5T", "O3T",
]);

/**
 * The representative atom, by preference. A structural token's own pseudo-beta.
 *
 * 🔴 A NUCLEIC BASE'S CENTRE DEPENDS ON THE RING SYSTEM, which one list cannot
 * express: a purine takes N9 and a pyrimidine N1, and purines carry an N1 as
 * well - in the six-membered ring - so a plain ["N1", "N9", ...] preference
 * silently picks N1 for every A, G, DA and DG. Upstream records that costing a
 * tRNA fold 5.6 A against 1.1, invisible to every protein test.
 */
const CENTRE = {
  proteinBackbone: ["CA", "N", "C"],
  proteinSidechain: ["CB"],
  nucleicBackbone: ["C4'", "C4*", "C1'", "C1*"],
  purineBase: ["N9", "C4", "C8", "N7", "C5"],
  pyrimidineBase: ["N1", "C2", "C6", "C5", "C4"],
  otherBase: ["C1'", "C1*", "N9", "N1"],
};

/** AF3 restypes: 0-19 amino acids, 20 UNK, 21 gap, 22-25 RNA, 26-29 DNA, 30 N. */
const PURINE_RESTYPES = new Set([22, 23, 26, 27]);   // A, G, DA, DG
const PYRIMIDINE_RESTYPES = new Set([24, 25, 28, 29]); // C, U, DC, DT

/** One atom's name, decoded from the four packed characters. */
export function atomNameAt(batch, token, slot) {
  const base = (token * batch.dense + slot) * 4;
  let name = "";
  for (let character = 0; character < 4; character += 1) {
    const value = batch.refAtomNameChars[base + character];
    if (value > 0) name += String.fromCharCode(value + 32);
  }
  return name;
}

/** The molecule kind a token belongs to: "protein", "dna", "rna" or "ligand". */
function kindOfToken(batch, token) {
  const residue = batch.residueOfToken[token];
  const chain = batch.chainOfResidue?.[residue];
  return batch.chainKinds?.[chain] ?? "protein";
}

/**
 * OpenDDE's structural-token layout over an AlphaFold 3 batch.
 *
 * Returns the bookkeeping the expander needs and the (residue token, slot)
 * source of every structural slot, which is what every gather here runs on.
 */
export function structuralLayout(batch) {
  const dense = batch.dense;
  const residueTokens = batch.tokens;
  const parent = [];
  const role = [];
  const sources = [];        // per structural token: the residue slots it draws
  const pseudoBetaSlot = [];

  for (let token = 0; token < residueTokens; token += 1) {
    const live = [];
    for (let slot = 0; slot < dense; slot += 1) {
      if (batch.refMask[token * dense + slot]) live.push(slot);
    }
    if (live.length === 0) continue;
    const names = live.map((slot) => atomNameAt(batch, token, slot));
    const kind = kindOfToken(batch, token);
    const restype = batch.aatype[token];
    // 🔴 A SPLIT NEEDS A STANDARD RESIDUE WITH MORE THAN ONE ATOM, and the
    // second half of that is what makes ligands and modified residues fall out
    // for free: AlphaFold 3 already atomises them, one token per atom, so they
    // arrive here with a single slot and take the role-0 branch without this
    // code having to know what they are.
    const polymer = kind === "protein" ? restype <= 20
      : (kind === "dna" || kind === "rna") ? (restype >= 22 && restype <= 30)
      : false;
    const groups = [];
    if (polymer && live.length > 1) {
      const backboneSet = kind === "protein" ? PROTEIN_BACKBONE : NUCLEIC_BACKBONE;
      const backbone = [];
      const child = [];
      live.forEach((slot, index) => {
        (backboneSet.has(names[index]) ? backbone : child).push(index);
      });
      const [backboneRole, childRole] = kind === "protein"
        ? [ROLE_PROTEIN_BB, ROLE_PROTEIN_SC]
        : kind === "dna" ? [ROLE_DNA_BB, ROLE_DNA_BASE] : [ROLE_RNA_BB, ROLE_RNA_BASE];
      const backboneCentre = kind === "protein"
        ? CENTRE.proteinBackbone : CENTRE.nucleicBackbone;
      if (backbone.length === 0 || child.length === 0) {
        // Glycine, or any residue whose split leaves a side empty: one token.
        groups.push([backboneRole, live.map((_, index) => index), backboneCentre]);
      } else {
        const childCentre = kind === "protein" ? CENTRE.proteinSidechain
          : PURINE_RESTYPES.has(restype) ? CENTRE.purineBase
          : PYRIMIDINE_RESTYPES.has(restype) ? CENTRE.pyrimidineBase
          : CENTRE.otherBase;
        groups.push([backboneRole, backbone, backboneCentre]);
        groups.push([childRole, child, childCentre]);
      }
    } else {
      live.forEach((_, index) => groups.push([ROLE_ATOM, [index], [names[index]]]));
    }

    for (const [roleId, indices, preference] of groups) {
      parent.push(token);
      role.push(roleId);
      sources.push(indices.map((index) => live[index]));
      // The representative atom, by preference, else the first.
      let chosen = 0;
      outer: for (const wanted of preference) {
        for (let at = 0; at < indices.length; at += 1) {
          if (names[indices[at]] === wanted) { chosen = at; break outer; }
        }
      }
      pseudoBetaSlot.push(chosen);
    }
  }

  const tokens = parent.length;
  // 🔴 THE TWIN IS THE OTHER HALF OF THE SAME RESIDUE, and only where the
  // residue really split: upstream sets it when a parent has EXACTLY two
  // members, so a ligand's many role-0 tokens are not twins of one another.
  const twin = new Int32Array(tokens).fill(NO_TWIN);
  const members = new Map();
  for (let index = 0; index < tokens; index += 1) {
    if (!members.has(parent[index])) members.set(parent[index], []);
    members.get(parent[index]).push(index);
  }
  for (const group of members.values()) {
    if (group.length === 2) { twin[group[0]] = group[1]; twin[group[1]] = group[0]; }
  }

  // The chain-adjacent parents, which the expander turns into a pair feature.
  const prevParent = new Int32Array(tokens).fill(-1);
  const nextParent = new Int32Array(tokens).fill(-1);
  const chainOf = (token) => batch.chainOfResidue?.[batch.residueOfToken[token]] ?? 0;
  for (let index = 0; index < tokens; index += 1) {
    const token = parent[index];
    if (token - 1 >= 0 && chainOf(token - 1) === chainOf(token)) prevParent[index] = token - 1;
    if (token + 1 < residueTokens && chainOf(token + 1) === chainOf(token)) {
      nextParent[index] = token + 1;
    }
  }

  // residueAtomGather[r * dense + j] = the flat structural slot holding residue
  // token r's atom j, so the diffusion's coordinates can be scattered back into
  // the layout every writer here already understands.
  const residueAtomGather = new Int32Array(residueTokens * dense).fill(-1);
  for (let index = 0; index < tokens; index += 1) {
    sources[index].forEach((slot, at) => {
      residueAtomGather[parent[index] * dense + slot] = index * dense + at;
    });
  }

  // The structural token standing for a residue when a per-token quantity has
  // to come back: its FIRST, which is the one carrying the backbone role.
  const residueRepToken = new Int32Array(residueTokens);
  const seen = new Uint8Array(residueTokens);
  for (let index = 0; index < tokens; index += 1) {
    if (!seen[parent[index]]) { residueRepToken[parent[index]] = index; seen[parent[index]] = 1; }
  }

  return {
    tokens, dense,
    parent: Int32Array.from(parent),
    role: Int32Array.from(role),
    twin, prevParent, nextParent,
    sources, pseudoBetaSlot: Int32Array.from(pseudoBetaSlot),
    residueAtomGather, residueRepToken,
  };
}

/**
 * The diffusion's coordinates, scattered back onto the residue layout.
 *
 * 🔴 THE STRUCTURE EVERY CONSUMER READS IS THE RESIDUE ONE. The PDB writer, the
 * geometry checks and anything comparing against a deposition all index
 * (residue token, atom slot); handing them 91 structural tokens where they
 * expect 47 residues is where the writer stops. `residueAtomGather` is the
 * inverse of the regrouping and was built with it.
 */
export function structuralToResidue(positions, layout, residueTokens, dense) {
  const out = new Float32Array(residueTokens * dense * 3);
  for (let index = 0; index < residueTokens * dense; index += 1) {
    const from = layout.residueAtomGather[index];
    if (from < 0) continue;
    out[index * 3] = positions[from * 3];
    out[index * 3 + 1] = positions[from * 3 + 1];
    out[index * 3 + 2] = positions[from * 3 + 2];
  }
  return out;
}

/**
 * A batch over OpenDDE's structural tokens, from AlphaFold 3's residue batch.
 *
 * Shaped exactly like `featuriseProtein`'s output, because the diffusion head,
 * the atom encoder and the sampler all read a batch and none of them should
 * learn that a second token space exists. What changes is which token an atom
 * belongs to; the atoms themselves are gathered through `sources` unchanged.
 *
 * 🔴 THE MSA FIELDS ARE THE RESIDUE BATCH'S AND ARE NOT REGROUPED. Nothing
 * downstream of the trunk reads them - the diffusion takes the trunk's single
 * and pair, not an alignment - and a per-structural-token MSA is not a thing
 * OpenDDE has. They are carried so the object is a batch rather than a subset
 * of one, and a caller that reached for them would be asking the wrong question.
 */
export function structuralBatch(batch, layout = structuralLayout(batch)) {
  const dense = batch.dense;
  const tokens = layout.tokens;
  const size = tokens * dense;

  const refPos = new Float32Array(size * 3);
  const refMask = new Float32Array(size);
  const refElement = new Int32Array(size);
  const refCharge = new Float32Array(size);
  const refAtomNameChars = new Int32Array(size * 4);
  const refSpaceUid = new Int32Array(size);

  for (let token = 0; token < tokens; token += 1) {
    const from = layout.parent[token];
    layout.sources[token].forEach((slot, at) => {
      const source = from * dense + slot;
      const target = token * dense + at;
      refPos[target * 3] = batch.refPos[source * 3];
      refPos[target * 3 + 1] = batch.refPos[source * 3 + 1];
      refPos[target * 3 + 2] = batch.refPos[source * 3 + 2];
      refMask[target] = batch.refMask[source];
      refElement[target] = batch.refElement[source];
      refCharge[target] = batch.refCharge[source];
      for (let c = 0; c < 4; c += 1) {
        refAtomNameChars[target * 4 + c] = batch.refAtomNameChars[source * 4 + c];
      }
      // 🔴 THE RESIDUE'S SPACE, CARRIED THROUGH THE GATHER. It comes from the
      // source slot, so a backbone token and its sidechain twin keep the one
      // uid their residue had - which is what AlphaFold 3 keys on
      // `(chain_id, res_id)` and what lets a side chain see its own backbone.
      refSpaceUid[target] = batch.refSpaceUid[source];
    });
  }

  // Per-token features: the parent's, since a structural token is part of one
  // residue and inherits its position in the chain.
  const take = (source) => Int32Array.from(layout.parent, (from) => source[from]);
  const residueIndex = take(batch.residueIndex);
  const tokenIndex = Int32Array.from({ length: tokens }, (_, index) => index);
  const asymId = take(batch.asymId);
  const entityId = take(batch.entityId);
  const symId = take(batch.symId);
  const aatype = take(batch.aatype);
  const residueOfToken = take(batch.residueOfToken);
  const seqMask = Float32Array.from(layout.parent, (from) => batch.seqMask[from]);

  const realAtoms = [];
  for (let index = 0; index < size; index += 1) if (refMask[index]) realAtoms.push(index);

  const gathers = atomGathers({
    tokens, dense, realAtoms, pseudoBetaSlot: layout.pseudoBetaSlot,
  });

  // 🔴 THE BOND MATRIX IS REBUILT ON THE NEW TOKENS, NOT GATHERED. It is
  // tokens x tokens, so a gather of the residue one would be the wrong SHAPE as
  // well as the wrong content: a bond between two residue tokens becomes a bond
  // between whichever structural tokens now hold those atoms. The backbone
  // halves carry the polymer bond, which is what `prevParent`/`nextParent`
  // already say, so what remains here is the residue batch's own bonds mapped
  // through the parent - a ligand's, and a modified residue's.
  const bondMatrix = new Float32Array(tokens * tokens);
  if (batch.bondMatrix !== undefined) {
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        if (batch.bondMatrix[layout.parent[i] * batch.tokens + layout.parent[j]]) {
          bondMatrix[i * tokens + j] = 1;
        }
      }
    }
    bondMatrix[0] = 0;
  }

  return {
    ...batch,
    structural: layout,
    tokens, dense, subsets: gathers.subsets, atomCount: gathers.atomCount,
    shape: { tokens, dense, subsets: gathers.subsets, queries: 32, keys: gathers.keys },
    aatype, residueIndex, tokenIndex, asymId, entityId, symId, seqMask,
    refPos, refMask, refElement, refCharge, refAtomNameChars, refSpaceUid,
    predDenseAtomMask: refMask,
    bondMatrix, residueOfToken,
    tokenAtomsToQueries: gathers.tokenAtomsToQueries,
    queriesToTokenAtoms: gathers.queriesToTokenAtoms,
    queriesToKeys: gathers.queriesToKeys,
    tokensToQueries: gathers.tokensToQueries,
    tokensToKeys: gathers.tokensToKeys,
    tokenAtomsToPseudoBeta: gathers.tokenAtomsToPseudoBeta,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
  };
}

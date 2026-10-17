// Every weight the port reads through a derived form - a triangle's interleaved projection and gate, an attention's
// q | k | v | gate, a transition's SwiGLU interleaving, the token transformer's folded conditioning - built at load,
// and its sources retired: the bundle's allocations are then rebuilt without them. Held side by side they were
// ~0.6 GB of AlphaFold 3's 1.4 GB.
#include "af3.h"
#include <cstring>

bool derivedSource(const std::string& name) {
  auto ends = [&](const char* tail) { size_t n = strlen(tail); return name.size() >= n && !name.compare(name.size() - n, n, tail); };
  for (const char* tail : {".qProjection", ".kProjection", ".vProjection", ".gatingQuery", "ransition1", ".projection", ".gate",
                           "SingleCondScaleWeights", "SingleCondBias", "AdaptiveZeroCondWeights"})
    if (ends(tail)) return true;
  return false;
}
static void retire(std::initializer_list<std::string> names) {
  for (auto& n : names) M.retire(n);
}
static void transitionWeight(const std::string& pre) {      // a LayerNorm'd SwiGLU transition: its first weight
  if (!hasW(pre + ".transition1")) return;
  int C = (int)lenW(pre + ".inputLayerNormScale"), I = (int)(lenW(pre + ".transition1") / (2 * (size_t)C));
  swigluPairs(pre + ".transition1", C, I);
  retire({pre + ".transition1"});
}
static void attentionWeight(const std::string& A, int C, bool transposedQkg) {
  int Wd = (int)(lenW(A + ".vProjection") / C);
  qkvgWeight(A, C, Wd, transposedQkg);
  retire({A + ".qProjection", A + ".kProjection", A + ".vProjection", A + ".gatingQuery"});
}
// a pairformer / MSA / template block's pair track, and its single track where it has one
static void blockWeights(const std::string& pre) {
  int C = (int)lenW(pre + ".triangleMultiplicationOutgoing.leftNormInputScale");
  for (const char* dir : {".triangleMultiplicationOutgoing", ".triangleMultiplicationIncoming"}) {
    triGateWeight(pre + dir, C);
    retire({pre + dir + ".projection", pre + dir + ".gate"});
  }
  for (const char* a : {".pairAttention1", ".pairAttention2"}) attentionWeight(pre + a, C, true);
  transitionWeight(pre + ".pairTransition");
  transitionWeight(pre + ".msaTransition");
  if (hasW(pre + ".singleAttention.qProjection")) {
    attentionWeight(pre + ".singleAttention", (int)lenW(pre + ".singleAttention.layerNormScale"), false);
    transitionWeight(pre + ".singleTransition");
  }
}
static void atomBlocks(const std::string& E) {
  if (!M.has(E + ".channels")) return;
  int C = metaI(E + ".channels");
  for (int b = 0; M.has(E + ".blocks." + num(b) + ".qProjection"); ++b) {
    std::string B = E + ".blocks." + num(b);
    int Wd = (int)(lenW(B + ".qProjection") / C);
    concatColumns(B + ".qg", C, {{B + ".qProjection", Wd, false}, {B + ".gatingQuery", Wd, false}});
    concatColumns(B + ".kv", C, {{B + ".kProjection", Wd, false}, {B + ".vProjection", Wd, false}});
    swigluPairs(B + ".ffwTransition1", C, 2 * C);
    retire({B + ".qProjection", B + ".gatingQuery", B + ".kProjection", B + ".vProjection", B + ".ffwTransition1"});
  }
}

void prepareWeights() {
  for (const char* stack : {"trunk.pairformerBlocks.", "trunk.msaBlocks.", "trunk.template.blocks.", "confidence.blocks.",
                            "refiner.blocks.", "ddeConfidence.blocks."})
    for (int k = 0; M.has(stack + num(k) + ".triangleMultiplicationOutgoing.projection"); ++k) blockWeights(stack + num(k));
  for (const char* E : {"targetFeat.encoder", "diffusion.encoder", "diffusion.decoder"}) atomBlocks(E);
  // the diffusion conditioning's plain transitions
  const std::string P = "diffusion.conditioning";
  if (M.has(P + ".pairChannels")) {
    int Cz = metaI(P + ".pairChannels"), Cs = metaI(P + ".seqChannels");
    for (int k = 0; k < 2; ++k) {
      for (auto [track, C] : {std::pair<const char*, int>{".pairTransitions.", Cz}, {".singleTransitions.", Cs}}) {
        std::string T = P + track + num(k);
        if (!hasW(T + ".ffwTransition1")) continue;
        swigluPairs(T + ".ffwTransition1", C, (int)(lenW(T + ".ffwTransition1") / (2 * (size_t)C)));
        retire({T + ".ffwTransition1"});
      }
    }
  }
  // the token transformer: its folded conditioning, and each block's projections and transition
  if (M.has("diffusion.transformer.channels")) {
    const std::string Tn = "diffusion.transformer";
    transformerWeights();
    int C = metaI(Tn + ".channels"), I = C * metaI(Tn + ".transitionFactor"), perSuper = metaI(Tn + ".blocksPerSuperBlock");
    for (int sb = 0; hasW(Tn + ".superBlocks." + num(sb) + ".pairLogitsProjection"); ++sb)
      for (int k = 0; k < perSuper; ++k) {
        std::string B = Tn + ".superBlocks." + num(sb) + ".blocks." + num(k);
        int Wd = (int)(lenW(B + ".qProjection") / C);
        qkvgWeight(B, C, Wd, false);
        swigluPairs(B + ".ffwTransition1", C, I);
        retire({B + ".qProjection", B + ".kProjection", B + ".vProjection", B + ".gatingQuery", B + ".ffwTransition1"});
        for (const char* slot : {".", ".ffw"})
          retire({B + slot + "SingleCondScaleWeights", B + slot + "SingleCondBias", B + slot + "AdaptiveZeroCondWeights"});
      }
  }
  M.compact();
}

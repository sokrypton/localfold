// Every weight the port reads through a derived form - a triangle's interleaved projection and gate, an attention's
// q | k | v | gate, a transition's SwiGLU interleaving, the token transformer's folded conditioning - built at load,
// and its sources retired: the bundle's allocations are then rebuilt without them. Held side by side they were
// ~0.6 GB of AlphaFold 3's 1.4 GB.
#include "af3.h"
#include <cstring>

bool derivedSource(const std::string& name) {
  if (name.find(".msaAttention1.") != std::string::npos) return false;
  auto ends = [&](const char* tail) { size_t n = strlen(tail); return name.size() >= n && !name.compare(name.size() - n, n, tail); };
  for (const char* tail : {".qProjection", ".kProjection", ".vProjection", ".gatingQuery", "ransition1", ".projection", ".gate",
                           "SingleCondScaleWeights", "SingleCondBias", "AdaptiveZeroCondWeights"})
    if (ends(tail)) return true;
  return false;
}
bool halfWeight(const std::string& name, size_t elements) {
  if (elements >= 16384) return true;
  if (derivedSource(name)) return false;
  auto ends = [&](const char* tail) { size_t n = strlen(tail); return name.size() >= n && !name.compare(name.size() - n, n, tail); };
  for (const char* tail : {".pairBiasProjection", ".singlePairLogitsProjection", ".gatingLinear", ".transition2",
                           ".Transition2", ".ffwTransition2", ".pairLogits", ".pairLogitsProjection",
                           ".leftProjection", ".rightProjection", ".leftSingle", ".rightSingle", ".extraMsaTargetFeat",
                           ".templatePairEmbedding2", ".templatePairEmbedding3", ".templatePairEmbedding8",
                           ".outputLinear", ".zProjection", ".aProjection", ".vProjection", ".gatingQuery",
                           ".paeLogits", ".leftHalfDistanceLogits", ".interHalfDistanceLogits", ".paeInterLogits",
                           ".pde", ".pae"})
    if (ends(tail)) return true;
  if (!name.rfind("trunk.", 0) && ends(".outputProjection")) return true;
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
static void blockWeights(const std::string& pre, bool foldParallel) {
  int C = (int)lenW(pre + ".triangleMultiplicationOutgoing.leftNormInputScale");
  for (const char* dir : {".triangleMultiplicationOutgoing", ".triangleMultiplicationIncoming"}) {
    std::string d = pre + dir;
    const half* tw = triGateWeight(d, C);
    retire({d + ".projection", d + ".gate"});
    if (foldParallel) {
      foldNorm("tg:" + d, tw, C, 4 * C, d + ".leftNormInput", nullptr, true);
      foldNorm("gl:" + d, Wh(d + ".gatingLinear"), C, C, d + ".leftNormInput", nullptr);
    }
  }
  for (const char* a : {".pairAttention1", ".pairAttention2"}) {
    std::string A = pre + a;
    if (!hasW(A + ".vProjection")) continue;      // (Mol-1's PairMixer blocks carry no triangle attention)
    int Wd = (int)(lenW(A + ".vProjection") / C);
    const half* qw = qkvgWeight(A, C, Wd, true);
    retire({A + ".qProjection", A + ".kProjection", A + ".vProjection", A + ".gatingQuery"});
    if (foldParallel) {
      int heads = metaI(A + ".heads");
      foldNorm("gb:" + A, Wh(A + ".pairBiasProjection"), C, heads, A + ".actNorm", nullptr);
      foldNorm("gq:" + A, qw, C, 4 * Wd, A + ".actNorm", qkvgBias(A, Wd), true);
    }
    if (!pre.compare(0, 11, "confidence.") && hasW(A + ".outputProjectionTransposed")) {
      M.derived<half>("dual:" + A, (size_t)Wd * C, [&](half* out) {
        float* sum = allocT<float>((size_t)Wd * C);
        copy(sum, W(A + ".outputProjection"), (size_t)Wd * C * 4);
        add(sum, W(A + ".outputProjectionTransposed"), (size_t)Wd * C);
        toHalf(sum, out, (size_t)Wd * C);
        release(sum);
      });
    }
  }
  if (hasW(pre + ".pairTransition.transition1")) {
    std::string T = pre + ".pairTransition";
    int I = (int)(lenW(T + ".transition1") / (2 * (size_t)C));
    const half* sw = swigluPairs(T + ".transition1", C, I);
    retire({T + ".transition1"});
    static const bool ft = !getenv("LOCALFOLD_FOLD_TRANSITION") || atoi(getenv("LOCALFOLD_FOLD_TRANSITION")) != 0;
    if (foldParallel && ft) foldNorm("tr:" + T, sw, C, 2 * I, T + ".inputLayerNorm", nullptr, true);
  }
  transitionWeight(pre + ".msaTransition");
  if (hasW(pre + ".singleAttention.qProjection")) {
    attentionWeight(pre + ".singleAttention", (int)lenW(pre + ".singleAttention.layerNormScale"), false);
    transitionWeight(pre + ".singleTransition");
    if (foldParallel && hasW(pre + ".singlePairLogitsProjection")) {
      int heads = metaI(pre + ".singleAttention.heads");
      foldNorm("sp:" + pre, Wh(pre + ".singlePairLogitsProjection"), C, heads, pre + ".singlePairLogitsNorm", nullptr);
    }
  }
}
static void atomBlocks(const std::string& E) {
  if (!M.has(E + ".channels")) return;
  int C = metaI(E + ".channels");
  for (int b = 0; M.has(E + ".blocks." + num(b) + ".qProjection"); ++b) {
    std::string B = E + ".blocks." + num(b);
    int Wd = (int)(lenW(B + ".qProjection") / C);
    qgkvWeight(B, C, Wd);
    swigluPairs(B + ".ffwTransition1", C, 2 * C);
    retire({B + ".qProjection", B + ".gatingQuery", B + ".kProjection", B + ".vProjection", B + ".ffwTransition1"});
  }
}

void prepareWeights() {
  const bool parallel = flag("trunk.dialect.parallelPairformer") &&
                        (!getenv("LOCALFOLD_FOLD_NORM") || atoi(getenv("LOCALFOLD_FOLD_NORM")) != 0);
  for (const char* stack : {"trunk.pairformerBlocks.", "trunk.pairMixerBlocks.", "trunk.msaBlocks.", "trunk.template.blocks.", "confidence.blocks.",
                            "refiner.blocks.", "ddeConfidence.blocks."}) {
    const bool foldStack = parallel && strcmp(stack, "trunk.template.blocks.") != 0;
    for (int k = 0; M.has(stack + num(k) + ".triangleMultiplicationOutgoing.projection"); ++k)
      blockWeights(stack + num(k), foldStack);
  }
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

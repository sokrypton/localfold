// RFD4-Proteina's published INT8 bundle, read natively.
//
// `model-rfd4p-int8/manifest.json` plus 16 shards, as `sokrypton/rfd4-proteina-webgpu` publishes
// them and as the page reads them - 469 MB against the float32 export's 1,226, which is the whole
// point: the weights are what a cold run waits for.
//
// The format (`shared/rfd4p/safetensors.js`, `readShardedTensor`) is two cases:
//   float32   `elements` floats at `byteOffset`                       - 720 of the 1,257 tensors
//   int8      `elements` codes at `byteOffset`, and a float16 SCALE a
//             block of 64 at `scaleOffset`; value = code * scale      - the other 537
//
// 🔴 AND IT USES af3/src/common.cuh's OWN `Json`, not cuda/featurise/json.h's. Both are called
// `Json` at a scope this file can see, and including the second one makes every use ambiguous.
//
// 🔴 THE SMALL TENSORS ARE NOT QUANTISED AND THAT IS NOT AN OVERSIGHT. Every bias, every norm
// weight and every embedding table stays float32; only the matrices are coded. Quantising a
// LayerNorm's scale saves a few kilobytes and moves every number downstream of it.
#pragma once
#include "trunk.cuh"
#include <fstream>
#include <map>

namespace rfd4p {

// 🔴 ONE `read`, NOT AN ITERATOR. `std::istreambuf_iterator` walks a file a CHARACTER at a time;
// over 331 MB of shards that alone was most of a cold run, and it looked exactly like the int8
// format being slow to load.
inline std::string readBlob(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  const std::streamsize size = in.tellg();
  std::string out((size_t)size, '\0');
  in.seekg(0);
  in.read(out.data(), size);
  return out;
}

// `out[i] = codes[i] * scales[i / block]`, the scales in float16.
__global__ void dequantInt8K(const int8_t* __restrict__ codes, const __half* __restrict__ scales,
                             float* __restrict__ out, size_t n, int block) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (float)codes[i] * (float)scales[i / block];
}

/**
 * Load the int8 bundle from `dir`, registering every tensor under `w/<name>` as float32 - the same
 * names and the same layout the float32 export produces, so nothing downstream can tell which it
 * got. `shardDir` holds the `.bin` files; the manifest and config sit in `dir`.
 */
inline void loadInt8Bundle(const std::string& dir, const std::string& shardDir) {
  // 🔴 THE ARCHITECTURE SCALARS COME FROM config.json HERE, where the float32 export writes them as
  // `m meta/...` lines. Same keys, same values, so the port sizes itself identically whichever
  // bundle it was given - and no dimension is typed into the CUDA source either way.
  {
    const Json cfgFile = Json::parse(readBlob(dir + "/config.json"));
    const Json* cfg = cfgFile.get("config");
    const Json* nn = cfg ? cfg->get("nn") : nullptr;
    const Json* ae = cfg ? cfg->get("autoencoder_config") : nullptr;
    const Json* nnAe = ae ? ae->get("nn_ae") : nullptr;
    const Json* dec = nnAe ? nnAe->get("decoder") : nullptr;
    if (!nn || !dec) { fprintf(stderr, "%s/config.json has no nn/decoder\n", dir.c_str()); exit(1); }
    auto num = [](const Json* o, const char* k, double fallback) {
      const Json* v = o ? o->get(k) : nullptr;
      return v && v->t == Json::NUM ? v->num : fallback;
    };
    auto flag = [](const Json* o, const char* k, double fallback) {
      const Json* v = o ? o->get(k) : nullptr;
      return v ? (v->t == Json::BOOL ? (v->num != 0 ? 1.0 : 0.0) : v->num) : fallback;
    };
    const Json* readout = nn->get("sequence_readout");
    const std::pair<const char*, double> meta[] = {
        {"meta/nlayers", num(nn, "nlayers", 14)},
        {"meta/tokenDim", num(nn, "token_dim", 768)},
        {"meta/nheads", num(nn, "nheads", 12)},
        {"meta/pairDim", num(nn, "pair_repr_dim", 384)},
        {"meta/condDim", num(nn, "dim_cond", 256)},
        {"meta/pairCondDim", num(nn, "pair_cond_factory_dim", 256)},
        {"meta/triMultC", num(nn, "tri_mult_c", 256)},
        {"meta/updatePairRepr", flag(nn, "update_pair_repr", 1)},
        {"meta/updatePairEveryN", num(nn, "update_pair_repr_every_n", 2)},
        {"meta/useQkln", flag(nn, "use_qkln", 1)},
        {"meta/useTriMult", flag(nn, "use_tri_mult", 1)},
        {"meta/seqReadoutClasses", num(readout, "num_classes", 147)},
        {"meta/latentDim", num(nnAe, "latent_z_dim", 32)},
        {"meta/decoderLayers", num(dec, "nlayers", 12)},
        {"meta/decoderTokenDim", num(dec, "token_dim", 768)},
        {"meta/decoderHeads", num(dec, "nheads", 12)},
        {"meta/decoderPairDim", num(dec, "pair_repr_dim", 256)},
        {"meta/decoderCondDim", num(dec, "dim_cond", 128)},
        {"meta/nResidueTypes", num(nnAe, "n_residue_types", 147)},
    };
    for (const auto& [key, value] : meta) M.index[key] = Entry{'m', 0, 0, value, 0};
  }

  const Json manifest = Json::parse(readBlob(dir + "/manifest.json"));
  const Json* tensors = manifest.get("tensors");
  if (!tensors || tensors->t != Json::OBJ) {
    fprintf(stderr, "%s/manifest.json has no tensors\n", dir.c_str());
    exit(1);
  }

  // One shard at a time, so the host never holds more than the largest of them.
  //
  // 🔴 THE ENCODER'S FIVE SHARDS ARE SKIPPED WHEN THEY ARE NOT THERE, and that is the published
  // bundle's own split: 331 MB of `core` and 138 MB of `encoder`, the latter fetched only for
  // partial diffusion and inverse folding. This port does neither, so a bundle without them is
  // complete rather than broken - but a tensor the trunk or the decoder actually wants must still
  // be present, which `W()` enforces by name at first use.
  std::map<std::string, std::vector<std::pair<std::string, const Json*>>> byShard;
  for (const auto& [name, rec] : tensors->obj) {
    const Json* file = rec.get("file");
    if (!file || file->t != Json::STR) continue;
    byShard[file->str].push_back({name, &rec});
  }
  size_t skipped = 0;
  for (auto it = byShard.begin(); it != byShard.end();) {
    if (std::ifstream(shardDir + "/" + it->first, std::ios::binary)) { ++it; continue; }
    skipped += it->second.size();
    it = byShard.erase(it);
  }

  // 🔴 ONE ARENA AND ONE UPLOAD A SHARD, NOT ONE OF EACH A TENSOR. The first version of this
  // allocated, copied, dequantised, synchronised and freed per tensor - 910 `cudaMalloc`s, 405
  // `cudaFree`s and 405 syncs - and made a COLD run 1.95 -> 3.96 s while reading a THIRD of the
  // bytes. An allocator call is not free and there is no reason to make one a tensor.
  size_t total = 0;
  for (const auto& [shard, entries] : byShard) {
    for (const auto& [name, rec] : entries) {
      const Json* shapeJ = rec->get("shape");
      size_t elements = 1;
      if (shapeJ && shapeJ->t == Json::ARR) for (const Json& d : shapeJ->arr) elements *= (size_t)d.num;
      total += (elements + 15) / 16 * 16;         // see the alignment note below
    }
  }
  float* arena = dalloc(total);
  size_t at = 0, coded = 0, plain = 0, bytes = 0, devShardBytes = 0;
  for (const auto& [shard, entries] : byShard) {
    (void)entries;
    std::ifstream probe(shardDir + "/" + shard, std::ios::binary | std::ios::ate);
    devShardBytes = std::max(devShardBytes, (size_t)probe.tellg());
  }
  uint8_t* devShard = dallocT<uint8_t>(devShardBytes);

  for (const auto& [shard, entries] : byShard) {
    const std::string blob = readBlob(shardDir + "/" + shard);
    bytes += blob.size();
    CK(cudaMemcpyAsync(devShard, blob.data(), blob.size(), cudaMemcpyHostToDevice, STREAM));
    for (const auto& [name, rec] : entries) {
      const Json* shapeJ = rec->get("shape");
      size_t elements = 1;
      if (shapeJ && shapeJ->t == Json::ARR) for (const Json& d : shapeJ->arr) elements *= (size_t)d.num;
      const Json* dtypeJ = rec->get("dtype");
      const Json* offJ = rec->get("byteOffset");
      const size_t off = offJ ? (size_t)offJ->num : 0;
      const std::string dtype = dtypeJ && dtypeJ->t == Json::STR ? dtypeJ->str : "float32";
      // 🔴 EVERY TENSOR STARTS 64-BYTE ALIGNED, which a per-tensor `cudaMalloc` gave for free and an
      // arena does not. cuBLAS's tensor-op paths refuse an operand that is merely 4-byte aligned -
      // `CUBLAS_STATUS_NOT_SUPPORTED`, raised from `linear` and nowhere near this file.
      float* dev = arena + at;
      at += (elements + 15) / 16 * 16;

      if (dtype == "float32") {
        if (off + elements * 4 > blob.size()) {
          fprintf(stderr, "%s: %s runs past %s\n", dir.c_str(), name.c_str(), shard.c_str());
          exit(1);
        }
        CK(cudaMemcpyAsync(dev, devShard + off, elements * 4, cudaMemcpyDeviceToDevice, STREAM));
        ++plain;
      } else if (dtype == "int8") {
        const Json* blockJ = rec->get("block");
        const Json* scaleJ = rec->get("scaleOffset");
        const int block = blockJ ? (int)blockJ->num : 64;
        const size_t scaleAt = scaleJ ? (size_t)scaleJ->num : 0;
        const size_t nBlocks = (elements + block - 1) / block;
        if (off + elements > blob.size() || scaleAt + nBlocks * 2 > blob.size()) {
          fprintf(stderr, "%s: %s runs past %s\n", dir.c_str(), name.c_str(), shard.c_str());
          exit(1);
        }
        dequantInt8K<<<blocks(elements), 256, 0, STREAM>>>(
            (const int8_t*)(devShard + off), (const __half*)(devShard + scaleAt), dev, elements,
            block);
        ++coded;
      } else {
        fprintf(stderr, "%s: %s has dtype %s, which this reader does not know\n", dir.c_str(),
                name.c_str(), dtype.c_str());
        exit(1);
      }
      deviceWeight("w/" + name, dev, elements);
    }
    CK(cudaStreamSynchronize(STREAM));     // the shard's host bytes go out of scope here
  }
  CK(cudaFree(devShard));
  printf("int8 bundle: %zu tensors (%zu coded, %zu float32), %.1f MB read", coded + plain, coded,
         plain, bytes / 1e6);
  if (skipped) printf("; %zu absent (the encoder's shards, which nothing here reads)", skipped);
  printf("\n");
}

}  // namespace rfd4p

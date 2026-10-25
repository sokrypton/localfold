// RFD4-Proteina's weights for the native CUDA port: every tensor of the released float32 checkpoint
// written verbatim under `w/<name>`, in cuda/af3's model.idx/model.bin format.
//
//   node cuda/rfd4p/export_weights.mjs cuda/rfd4p/weights-f32 \
//        [--checkpoint=<model.safetensors>] [--config=<config.json>] [--include=trunk|decoder|encoder|all]
//
// Entries:  t w/<name> ...   a checkpoint tensor (nn.transformer_layers.0.mhba.mha.to_qkv.weight, ...)
//           m w/<name>#r, #k  its rank and k-th dimension
//           m meta/<key>     the architecture scalars the port reads (nlayers, tokenDim, ...)
//
// 🔴 THE NAMES AND THE LAYOUT ARE THE CHECKPOINT'S, UNTRANSPOSED. A torch `nn.Linear.weight` is
// `[out, in]` and cuda's `gemm` wants `[in, out]`, so every read on the CUDA side passes
// `transposed = true` (one `CUBLAS_OP_T`, no materialised copy) rather than this file turning them
// round. That is deliberate: `cpu/rfd4p/reference.js` indexes `weight[o * inDim + k]` on these same
// bytes, so a stage that disagrees with the reference disagrees about arithmetic and never about
// which number went where - and the exporter stays a copy, with nothing in it to get wrong.
//
// 🔴 AND THE FLOAT CHECKPOINT IS NVIDIA'S, UNDER NVIDIA'S OWN TERMS. This writes the oracle arm's
// weights from a checkpoint the user has already accepted; the port's shipping path is the INT8
// bundle (`model-rfd4p-int8/`, published by sokrypton/rfd4-proteina-webgpu) through
// `Model::loadBundle`, which needs no float download. Nothing here fetches anything.
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { parseSafetensors } from "../../shared/rfd4p/safetensors.js";

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--"));
if (!out) {
  console.error("usage: export_weights.mjs <out dir> [--checkpoint=<path>] [--config=<path>] [--include=trunk|all]");
  process.exit(1);
}
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repo = new URL("../..", import.meta.url).pathname;
const hub = `${process.env.HOME}/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots`;
const checkpoint = option("checkpoint", `${hub}/a06e2509c7033b010fc43614e9fd840303588502/model.safetensors`);
const configPath = option("config", `${repo}model-rfd4p-int8/config.json`);
const include = option("include", "trunk");

// The trunk is `nn.*`; the autoencoder's two halves are a third of the checkpoint each and only the
// decoder is needed to turn local latents into atoms (the encoder is partial diffusion and inverse
// folding alone).
const WANTED = {
  trunk: (name) => name.startsWith("nn."),
  decoder: (name) => name.startsWith("nn.") || name.startsWith("autoencoder.decoder."),
  encoder: (name) => name.startsWith("nn.") || name.startsWith("autoencoder.encoder."),
  all: () => true,
};
const wanted = WANTED[include];
if (!wanted) { console.error(`--include must be one of ${Object.keys(WANTED).join(", ")}`); process.exit(1); }

const cfg = JSON.parse(readFileSync(configPath, "utf8")).config;
const nn = cfg.nn;
const ae = cfg.autoencoder_config.nn_ae;

const raw = readFileSync(checkpoint);
const parsed = parseSafetensors(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const tensors = parsed.tensors ?? parsed;

mkdirSync(out, { recursive: true });
const fd = openSync(`${out}/model.bin`, "w");
const lines = [];
let offset = 0;

// The architecture scalars the port sizes itself from, so no dimension is typed into the CUDA source.
const meta = {
  nlayers: nn.nlayers,
  tokenDim: nn.token_dim,
  nheads: nn.nheads,
  pairDim: nn.pair_repr_dim,
  condDim: nn.dim_cond,
  pairCondDim: nn.pair_cond_factory_dim ?? 256,
  triMultC: nn.tri_mult_c,
  updatePairRepr: nn.update_pair_repr ? 1 : 0,
  updatePairEveryN: nn.update_pair_repr_every_n ?? 2,
  useQkln: nn.use_qkln ? 1 : 0,
  useTriMult: nn.use_tri_mult ? 1 : 0,
  seqReadoutClasses: nn.sequence_readout?.num_classes ?? 147,
  latentDim: ae.latent_z_dim,
  decoderLayers: ae.decoder.nlayers,
  decoderTokenDim: ae.decoder.token_dim,
  decoderHeads: ae.decoder.nheads,
  decoderPairDim: ae.decoder.pair_repr_dim,
  decoderCondDim: ae.decoder.dim_cond,
  encoderLayers: ae.encoder.nlayers,
  nResidueTypes: ae.n_residue_types,
};
for (const [key, value] of Object.entries(meta)) lines.push(`m meta/${key} ${value}`);

let kept = 0;
for (const [name, record] of Object.entries(tensors)) {
  if (!wanted(name)) continue;
  const values = record.data instanceof Float32Array ? record.data : Float32Array.from(record.data);
  const key = `w/${name}`;
  lines.push(`t ${key} ${offset} ${values.length}`, `m ${key}#r ${record.shape.length}`);
  record.shape.forEach((d, k) => lines.push(`m ${key}#${k} ${d}`));
  writeSync(fd, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
  offset += values.length;
  kept += 1;
}
closeSync(fd);

// model.idx last and by rename, which is what the port's --wait-input polls for.
writeFileSync(`${out}/model.idx.tmp`, `${lines.join("\n")}\n`);
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${out}: ${kept} tensors (${include}), ${(offset * 4 / 1e6).toFixed(1)} MB`);

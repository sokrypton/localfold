#!/bin/bash
# Every cuda/grad gate, built and run. Float32: the training forward against each port; float64 (-DGRAD_F64): the
# backward against central differences of the same forward. Needs the bundles under the repository root and the AF2
# oracle dumps (cuda/grad/README.md); a missing bundle is reported, not skipped silently.
#   bash cuda/grad/run_gates.sh            (from the repository root)
set -u
cd "$(dirname "$0")/../.."
NV="nvcc -std=c++17 -arch=sm_${LOCALFOLD_SM:-80}"
fail=0
run() { echo "== $*"; "$@" 2>&1 | grep -E "PASS|FAIL|ABOVE|against|gradcheck:|worst" || true; "$@" > /dev/null 2>&1 || fail=1; }
# the builds run LOCALFOLD_JOBS at a time (16 by default): each is one nvcc of the whole header set, ~2 min, and
# one after another the 55 of them were most of the suite's two hours with the box's other cores idle
J=${LOCALFOLD_JOBS:-16}
BL=$(mktemp -d)
build() {
  local out=$1; shift
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
  rm -f "$out"   # (a failed build must not leave a stale binary for the runs)
  ( $NV "$@" -o "$out" -lcublas -lcublasLt -ldl -lcurand > "$BL/$(basename "$out").log" 2>&1 ) &
}
built() {   # every build finished, its errors printed
  wait
  for f in "$BL"/*.log; do grep -i -B1 -A3 " error" "$f" && { echo "  (in $(basename "$f" .log))"; fail=1; }; done
  rm -rf "$BL"
}
build cuda/grad/test-flash -O1 cuda/grad/test_flash.cu
build cuda/grad/test-bf16 -O1 cuda/grad/test_bf16.cu
build cuda/grad/test-dropout-f64 -O2 -DGRAD_F64 cuda/grad/test_dropout.cu
build cuda/grad/test-af2-block -O2 cuda/grad/test_af2_block.cu
build cuda/grad/test-af2-block-f64 -O2 -DGRAD_F64 cuda/grad/test_af2_block.cu
build cuda/grad/test-af2sm -O1 cuda/grad/test_af2sm.cu
build cuda/grad/test-af2io -O1 cuda/grad/test_af2io.cu
build cuda/grad/test-af2io-f64 -O2 -DGRAD_F64 cuda/grad/test_af2io.cu
build cuda/grad/test-af2sm-f64 -O2 -DGRAD_F64 cuda/grad/test_af2sm.cu
build cuda/grad/test-af2tmpl -O1 cuda/grad/test_af2tmpl.cu
build cuda/grad/test-af2model -O1 cuda/grad/test_af2model.cu
build cuda/grad/test-af2model-f64 -O2 -DGRAD_F64 cuda/grad/test_af2model.cu
build cuda/grad/train-af2-smoke -O2 cuda/grad/train_af2_smoke.cu
build cuda/grad/train-of3 -O2 --default-stream per-thread cuda/grad/train_of3.cu -lnccl
build cuda/grad/test-of3loss -O2 cuda/grad/test_of3loss.cu
build cuda/grad/test-af2tmpl-f64 -O2 -DGRAD_F64 cuda/grad/test_af2tmpl.cu
build cuda/grad/test-plm -O2 cuda/grad/test_plm.cu
build cuda/grad/test-plm-f64 -O2 -DGRAD_F64 cuda/grad/test_plm.cu
build cuda/grad/test-af3 -O1 --default-stream per-thread cuda/grad/test_af3.cu
build cuda/grad/test-af3-f64 -O2 -DGRAD_F64 cuda/grad/test_af3.cu
build cuda/grad/test-af3diff -O1 --default-stream per-thread cuda/grad/test_af3diff.cu
build cuda/grad/test-af3diff-f64 -O2 -DGRAD_F64 cuda/grad/test_af3diff.cu
build cuda/grad/test-ef2 -O1 cuda/grad/test_ef2.cu
build cuda/grad/test-ef2atom -O1 cuda/grad/test_ef2atom.cu
build cuda/grad/test-ef2atom-f64 -O2 -DGRAD_F64 cuda/grad/test_ef2atom.cu
build cuda/grad/test-ef2diff -O1 cuda/grad/test_ef2diff.cu
build cuda/grad/test-ef2diff-f64 -O2 -DGRAD_F64 cuda/grad/test_ef2diff.cu
build cuda/grad/test-ef2trunk -O1 cuda/grad/test_ef2trunk.cu
build cuda/grad/test-ef2trunk-f64 -O2 -DGRAD_F64 cuda/grad/test_ef2trunk.cu
build cuda/grad/test-msap -O1 cuda/grad/test_msap.cu
build cuda/grad/test-msap-f64 -O2 -DGRAD_F64 cuda/grad/test_msap.cu
build cuda/grad/test-msapmodel -O1 cuda/grad/test_msapmodel.cu
build cuda/grad/test-msapmodel-f64 -O2 -DGRAD_F64 cuda/grad/test_msapmodel.cu
build cuda/grad/test-bioemu2 -O1 cuda/grad/test_bioemu2.cu
build cuda/grad/test-bioemu2-f64 -O2 -DGRAD_F64 cuda/grad/test_bioemu2.cu
build cuda/grad/test-bioemu2den -O1 cuda/grad/test_bioemu2den.cu
build cuda/grad/test-bioemu2den-f64 -O2 -DGRAD_F64 cuda/grad/test_bioemu2den.cu
build cuda/grad/test-bioemu2trunk -O1 cuda/grad/test_bioemu2trunk.cu
build cuda/grad/test-bioemu2trunk-f64 -O2 -DGRAD_F64 cuda/grad/test_bioemu2trunk.cu
build cuda/grad/test-atom -O1 --default-stream per-thread cuda/grad/test_atom.cu
build cuda/grad/test-atom-f64 -O2 -DGRAD_F64 cuda/grad/test_atom.cu
build cuda/grad/test-af3denoise -O1 --default-stream per-thread cuda/grad/test_af3denoise.cu
build cuda/grad/test-af3denoise-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3denoise.cu
build cuda/grad/test-af3embed -O1 --default-stream per-thread cuda/grad/test_af3embed.cu
build cuda/grad/test-af3embed-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3embed.cu
build cuda/grad/test-af3conf -O1 --default-stream per-thread cuda/grad/test_af3conf.cu
build cuda/grad/test-af3tmpl -O1 --default-stream per-thread cuda/grad/test_af3tmpl.cu
build cuda/grad/test-af3tmpl-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3tmpl.cu
build cuda/grad/test-af3conf-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3conf.cu
build cuda/grad/test-opendde -O1 --default-stream per-thread cuda/grad/test_opendde.cu
build cuda/grad/test-opendde-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_opendde.cu
build cuda/grad/test-rfd4p -O1 cuda/grad/test_rfd4p.cu
build cuda/grad/test-rfd4p-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4p.cu
build cuda/grad/test-rfd4trunk -O1 cuda/grad/test_rfd4trunk.cu
build cuda/grad/test-rfd4trunk-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4trunk.cu
build cuda/grad/test-rfd4ae -O1 cuda/grad/test_rfd4ae.cu
build cuda/grad/test-rfd4ae-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4ae.cu
built
# AF2 against af3-any-model's jax.vjp: the float64 build on every dump (both sides float64 - no ReLU input can sit
# within rounding of its kink on one side only), with the reference-free check beside it; the float32 build on the
# single blocks. (The three-block stack in float32 holds ~2.7M ReLU inputs, and a tie among them is a coin toss on
# the arithmetic's last bit, not a defect: it moved with an unrelated change of rounding, see README.)
run cuda/grad/test-flash                                     # the flash attention against the bmm path it replaces
run cuda/grad/test-bf16                                      # BF16 mode (bf16 storage, bf16 tensor cores) against float32, whole blocks
run cuda/grad/test-dropout-f64                               # AF2 training dropout: the backward regenerates the forward's masks
for d in cuda/grad/data/af2-*/; do run cuda/grad/test-af2-block-f64 "$d" --gradcheck; done
for d in cuda/grad/data/af2-block*/ cuda/grad/data/af2-extra-block/; do run cuda/grad/test-af2-block "$d"; done
run cuda/grad/test-af2sm
run cuda/grad/test-af2sm --input=cuda/af2/data-1brs --weights=cuda/af2/weights-model_1_multimer_v3   # (199 residues: past a 128-thread block)
run cuda/grad/test-af2sm-f64
run cuda/grad/test-af2io
run cuda/grad/test-af2io-f64
run cuda/grad/test-af2model                                  # the whole model, pass 0, against JAX
run cuda/grad/test-af2model-f64                              # ...and its backward, each stack a block, 1,024 extra rows
run cuda/grad/train-af2-smoke cuda/grad/data/af2-stack3      # (a training loop: the loss must come down)
run cuda/grad/test-af2tmpl                                   # the multimer's, 1BRS with a template
run cuda/grad/test-af2tmpl-f64
run cuda/grad/test-af2tmpl --weights=cuda/af2/weights-model_1_ptm --input=cuda/af2/data-5caj-t     # the monomer's
run cuda/grad/test-af2tmpl-f64 --weights=cuda/af2/weights-model_1_ptm --input=cuda/af2/data-5caj-t
run cuda/grad/test-plm
run cuda/grad/test-plm-f64
run cuda/grad/test-af3-f64
run cuda/grad/test-af3diff-f64
for fb in "model-af3-full-f32 af3" "model-boltz2-f32 boltz2" "model-protenix2-f32 protenix2" "model-intellifold2-int5 intellifold2" \
          "model-rosettafold3-int5 rosettafold3" "model-opendde-full-f32 opendde" "model-openbind0-f32 openbind0" \
          "model-mol1-f32 mol1" "model-chai1-f32 chai1"; do
  set -- $fb
  if [ ! -d "$1" ]; then echo "== $2: no $1 here"; continue; fi
  run cuda/grad/test-af3 --bundle=$1 --family=$2
  run cuda/grad/test-af3 --msa --bundle=$1 --family=$2
  run cuda/grad/test-af3diff --bundle=$1 --family=$2
  in=cuda/grad/data/af3-in-$2
  [ -d $in ] || cuda/featurise/af3-featurise $in --no-weights --family=$2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE > /dev/null
  run cuda/grad/test-af3denoise --bundle=$1 --family=$2 --input=$in          # (OpenDDE's on its structural tokens)
  run cuda/grad/test-af3denoise-f64 --bundle=$1 --family=$2 --input=$in
  # the template embedder: 6MRR with its own crystal in two slots (every family; chai-1 through templateEmbedding alone)
  tin=cuda/grad/data/af3-tmpl-$2
  [ -d $tin ] || cuda/featurise/af3-featurise $tin --no-weights --family=$2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE \
    --template=tools/fixtures/6mrr-crystal.pdb:A,tools/fixtures/6mrr-crystal.pdb:A > /dev/null
  run cuda/grad/test-af3tmpl --bundle=$1 --family=$2 --input=$tin
  case $2 in af3|boltz2|chai1|rosettafold3|protenix2) run cuda/grad/test-af3tmpl-f64 --bundle=$1 --family=$2 --input=$tin;; esac
  # the input embedder (every family's form; chai-1's reads ESM2 3B from af3am-esm2); the confidence head (every form
  # but OpenDDE's own head, which test-opendde holds)
  run cuda/grad/test-af3embed --bundle=$1 --family=$2 --input=$in
  run cuda/grad/test-af3embed-f64 --bundle=$1 --family=$2 --input=$in
  case $2 in opendde) run cuda/grad/test-opendde --bundle=$1 --input=$in        # (its structural tokens and own head)
                      run cuda/grad/test-opendde-f64 --bundle=$1 --input=$in ;;
             *) run cuda/grad/test-af3conf --bundle=$1 --family=$2 --input=$in
                run cuda/grad/test-af3conf-f64 --bundle=$1 --family=$2 --input=$in ;; esac
  for E in diffusion.encoder diffusion.decoder targetFeat.encoder; do run cuda/grad/test-atom --bundle=$1 --family=$2 --stack=$E; done
  # OpenFold3's training (OpenBind-0 is OpenFold3 v0.5.0): its losses held to a transcription of openfold3/core/loss on
  # the kitchen sink (every branch: modified residues, DNA, RNA, ligands, a ligand-polymer bond), then train-of3 - the
  # trunk composed against runTrunk, activation checkpointing against kept saves, 6MRR against its crystal (the loss
  # above its floor must halve)
  if [ $2 = openbind0 ]; then
    ks=cuda/grad/data/of3-kitchen-sink
    [ -d $ks/input ] || cuda/featurise/af3-featurise $ks/input --no-weights --family=$2 --job=tools/fixtures/kitchen-sink-relabelled.json > /dev/null
    mkdir -p $ks/case && run cuda/grad/test-of3loss --input=$ks/input --out=$ks/case
    if [ -x "$HOME/venv_ef2/bin/python" ]; then run "$HOME/venv_ef2/bin/python" cuda/grad/of3_loss_oracle.py $ks/case
    else echo "== of3_loss_oracle.py: no torch at ~/venv_ef2 here"; fi
    ex=cuda/grad/data/of3-6mrr
    mkdir -p $ex
    awk '/^ATOM/ && substr($0,22,1)=="A" && (substr($0,17,1)==" " || substr($0,17,1)=="A") {
           n=substr($0,13,4); gsub(/ /,"",n); printf "1 %d %s %s %s %s\n", substr($0,23,4), n, substr($0,31,8), substr($0,39,8), substr($0,47,8) }' \
      tools/fixtures/6mrr-crystal.pdb > $ex/truth.txt
    echo "$in $ex/truth.txt resolution=1.18" > $ex/example.txt
    run cuda/grad/train-of3 --bundle=$1 --family=$2 --examples=$ex/example.txt --steps=20 --samples=4 --lr=1e-4 --warmup=1 --check-ckpt=0 --smoke
    # the truth's permutation (AF3 SI 4.2) on 1BRS - three barnase/barstar pairs - disguised and recovered; the example
    # builder wants gemmi and RDKit (~/venv)
    if [ -x "$HOME/venv/bin/python" ]; then
      b=cuda/grad/data/of3-1brs
      [ -f $b/example.txt ] || { mkdir -p $b && zstd -dqf tools/fixtures/rcsb/1brs.cif.zst -o $b/1brs.cif &&
                                 "$HOME/venv/bin/python" tools/make_of3_example.py $b/1brs.cif $b --family=$2 > /dev/null; }
      run cuda/grad/train-of3 --bundle=$1 --family=$2 --examples=$b/example.txt --steps=0 --check-perm=0
    else echo "== the truth's permutation: no ~/venv (gemmi, RDKit) here"; fi
    # data parallel (of3dist.cuh), two ranks sharing this GPU through the file transport: their weights must stay one set
    rdv=$(mktemp -d /dev/shm/of3-rdv-XXXX)
    for r in 0 1; do cuda/grad/train-of3 --bundle=$1 --family=$2 --examples=$ex/example.txt --steps=2 --samples=2 --lr=1e-4 --warmup=1 \
      --world=2 --rank=$r --dp=files --rendezvous=$rdv > /dev/null 2> $rdv.r$r & done
    wait; grep -h "checksum" $rdv.r0 $rdv.r1; grep -q "OUT OF SYNC" $rdv.r0 $rdv.r1 && fail=1; grep -q "in sync" $rdv.r0 || fail=1
    rm -rf $rdv $rdv.r0 $rdv.r1
  fi
done
run cuda/grad/test-atom-f64
run cuda/grad/test-af3 --bundle=model-mol1-f32 --family=mol1 --prefix=trunk.pairMixerBlocks.
run cuda/grad/test-ef2
run cuda/grad/test-ef2atom
run cuda/grad/test-ef2atom-f64
run cuda/grad/test-ef2diff
run cuda/grad/test-ef2diff-f64
run cuda/grad/test-ef2trunk
run cuda/grad/test-ef2trunk-f64
run cuda/grad/test-ef2trunk-f64 --inputs
run cuda/grad/test-ef2trunk --confidence            # (Synthyra's head, on the vendor's 6MRR tensors)
run cuda/grad/test-ef2trunk-f64 --confidence
for b in model-esmfold2-fast-f32 model-esmfold2-f32; do
  if [ ! -d $b ]; then echo "== ef2trunk --parcae: no $b here"; continue; fi
  run cuda/grad/test-ef2trunk --parcae=$b
  run cuda/grad/test-ef2trunk-f64 --parcae=$b
done
# the full ESMFold2's MSA encoder, on a real alignment (tools/fixtures/test.a3m, 59 residues, 8,076 rows)
if [ -d model-esmfold2-f32 ]; then
  in=cuda/grad/data/ef2-msa59
  [ -d $in ] || cuda/featurise/ef2-featurise $in --sequence=$(sed -n 3p tools/fixtures/test.a3m) --a3m=tools/fixtures/test.a3m > /dev/null
  run cuda/grad/test-ef2trunk --parcae=model-esmfold2-f32 --msa --input=$in
  run cuda/grad/test-ef2trunk-f64 --parcae=model-esmfold2-f32 --msa --input=$in
fi
run cuda/grad/test-msap --layer=0
run cuda/grad/test-msap-f64
run cuda/grad/test-msapmodel
run cuda/grad/test-msapmodel-f64
run cuda/grad/test-bioemu2
run cuda/grad/test-bioemu2 --layer=3
run cuda/grad/test-bioemu2-f64
[ -d cuda/grad/data/be2-6mrr ] || node cuda/bioemu2/export_input.mjs cuda/grad/data/be2-6mrr --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE > /dev/null
run cuda/grad/test-bioemu2den
run cuda/grad/test-bioemu2den --sigma=40
run cuda/grad/test-bioemu2den-f64
run cuda/grad/test-bioemu2trunk
run cuda/grad/test-bioemu2trunk-f64
run cuda/grad/test-rfd4p
run cuda/grad/test-rfd4p --layer=12      # (pair updates on even layers only)
run cuda/grad/test-rfd4p-f64
run cuda/grad/test-rfd4trunk
run cuda/grad/test-rfd4trunk-f64
run cuda/grad/test-rfd4ae
run cuda/grad/test-rfd4ae --encoder
run cuda/grad/test-rfd4ae-f64
[ $fail = 0 ] && echo "ALL GATES PASS" || echo "SOME GATE FAILED"
exit $fail

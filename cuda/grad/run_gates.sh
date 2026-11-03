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
build() { local out=$1; shift; $NV "$@" -o "$out" -lcublas -lcublasLt -ldl -lcurand 2>&1 | grep -i -B1 -A3 " error" && fail=1; }
build cuda/grad/test-af2-block -O2 cuda/grad/test_af2_block.cu
build cuda/grad/test-af2-block-f64 -O2 -DGRAD_F64 cuda/grad/test_af2_block.cu
build cuda/grad/test-af2sm -O1 cuda/grad/test_af2sm.cu
build cuda/grad/test-af2io -O1 cuda/grad/test_af2io.cu
build cuda/grad/test-af2io-f64 -O2 -DGRAD_F64 cuda/grad/test_af2io.cu
build cuda/grad/test-af2sm-f64 -O2 -DGRAD_F64 cuda/grad/test_af2sm.cu
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
build cuda/grad/test-atom -O1 --default-stream per-thread cuda/grad/test_atom.cu
build cuda/grad/test-atom-f64 -O2 -DGRAD_F64 cuda/grad/test_atom.cu
build cuda/grad/test-af3denoise -O1 --default-stream per-thread cuda/grad/test_af3denoise.cu
build cuda/grad/test-af3denoise-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3denoise.cu
build cuda/grad/test-af3embed -O1 --default-stream per-thread cuda/grad/test_af3embed.cu
build cuda/grad/test-af3embed-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3embed.cu
build cuda/grad/test-af3conf -O1 --default-stream per-thread cuda/grad/test_af3conf.cu
build cuda/grad/test-af3conf-f64 -O2 -DGRAD_F64 --default-stream per-thread cuda/grad/test_af3conf.cu
build cuda/grad/test-rfd4p -O1 cuda/grad/test_rfd4p.cu
build cuda/grad/test-rfd4p-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4p.cu
build cuda/grad/test-rfd4trunk -O1 cuda/grad/test_rfd4trunk.cu
build cuda/grad/test-rfd4trunk-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4trunk.cu
build cuda/grad/test-rfd4ae -O1 cuda/grad/test_rfd4ae.cu
build cuda/grad/test-rfd4ae-f64 -O2 -DGRAD_F64 cuda/grad/test_rfd4ae.cu
# AF2 against af3-any-model's jax.vjp: the float64 build on every dump (both sides float64 - no ReLU input can sit
# within rounding of its kink on one side only), with the reference-free check beside it; the float32 build on the
# single blocks. (The three-block stack in float32 holds ~2.7M ReLU inputs, and a tie among them is a coin toss on
# the arithmetic's last bit, not a defect: it moved with an unrelated change of rounding, see README.)
for d in cuda/grad/data/af2-*/; do run cuda/grad/test-af2-block-f64 "$d" --gradcheck; done
for d in cuda/grad/data/af2-block*/ cuda/grad/data/af2-extra-block/; do run cuda/grad/test-af2-block "$d"; done
run cuda/grad/test-af2sm
run cuda/grad/test-af2sm --input=cuda/af2/data-1brs --weights=cuda/af2/weights-model_1_multimer_v3   # (199 residues: past a 128-thread block)
run cuda/grad/test-af2sm-f64
run cuda/grad/test-af2io
run cuda/grad/test-af2io-f64
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
  if [ "$2" != chai1 ] && [ "$2" != opendde ]; then
    in=cuda/grad/data/af3-in-$2
    [ -d $in ] || cuda/featurise/af3-featurise $in --no-weights --family=$2 --sequence=GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE > /dev/null
    run cuda/grad/test-af3denoise --bundle=$1 --family=$2 --input=$in
    run cuda/grad/test-af3denoise-f64 --bundle=$1 --family=$2 --input=$in
  fi
  # the input embedder's generic path (not boltz2's atom-only target_feat sum, nor chai-1's token embedding); the
  # confidence head's (not boltz2's re-embedding, rf3's global norms, chai-1's, nor OpenDDE's own head)
  case $2 in boltz2|chai1) ;; *) run cuda/grad/test-af3embed --bundle=$1 --family=$2 --input=cuda/grad/data/af3-in-$2
                                 run cuda/grad/test-af3embed-f64 --bundle=$1 --family=$2 --input=cuda/grad/data/af3-in-$2 ;; esac
  case $2 in boltz2|chai1|rosettafold3|opendde) ;; *) run cuda/grad/test-af3conf --bundle=$1 --family=$2 --input=cuda/grad/data/af3-in-$2
                                                      run cuda/grad/test-af3conf-f64 --bundle=$1 --family=$2 --input=cuda/grad/data/af3-in-$2 ;; esac
  for E in diffusion.encoder diffusion.decoder targetFeat.encoder; do run cuda/grad/test-atom --bundle=$1 --family=$2 --stack=$E; done
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
for b in model-esmfold2-fast-f32 model-esmfold2-f32; do
  if [ ! -d $b ]; then echo "== ef2trunk --parcae: no $b here"; continue; fi
  run cuda/grad/test-ef2trunk --parcae=$b
  run cuda/grad/test-ef2trunk-f64 --parcae=$b
done
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

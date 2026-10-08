#!/bin/sh
# The native featurisers (host C++, no CUDA): af3-featurise, af2-featurise, esmfold2-featurise and chem-probe.
set -e
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O2 -ffp-contract=off -pthread"
for p in af3 af2 esmfold2; do [ -f ${p}_featurise.cpp ] && $CXX $FLAGS -o ${p}-featurise ${p}_featurise.cpp & done
$CXX $FLAGS -o chem-probe chem_probe.cpp &
wait

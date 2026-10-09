#!/usr/bin/env bash
# Build the localfold wheel for Apple silicon: the three ports native on Metal (metal/build.sh: metal/core and each port),
# in one py3-none-macosx_13_0_arm64 wheel. `pip install localfold` takes this one on a Mac and the manylinux CUDA wheel
# (build_wheel.sh) on Linux, by its platform tag: the same package, the same commands, a different backend.
#
#   bash python/build_wheel_macos.sh [<out dir>]        (default python/dist)
#
# Needs the Xcode command-line tools and a Python with pip - no GPU and no Metal compiler: the binaries carry their
# kernels as Metal source and compile them on first use (cached in ~/.cache/localfold/metal). macOS 13 is the floor:
# a buffer's GPU address (MTLBuffer.gpuAddress), which the ports' raw device pointers become, is Metal 3.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/.." && pwd)"
out="$(mkdir -p "${1:-$here/dist}" && cd "${1:-$here/dist}" && pwd)"
py="${PYTHON:-python3}"
[ "$(uname -s)" = Darwin ] || { echo "build_wheel_macos.sh builds on macOS" >&2; exit 1; }
[ "$(uname -m)" = arm64 ] || { echo "build_wheel_macos.sh builds the arm64 wheel on Apple silicon" >&2; exit 1; }
export MACOSX_DEPLOYMENT_TARGET=13.0
# a fresh build: the deployment target is a compile flag, so nothing built without it is kept
rm -f "$repo"/metal/{af3,af2,ef2}/localfold-* "$repo/cuda/featurise/featurise"
bash "$repo/cuda/build.sh" featurise
bash "$repo/metal/build.sh" af3 af2 ef2
bin="$here/localfold/bin"
rm -f "$bin"/localfold-*
for port in af3 af2 ef2; do
  install -m 755 "$repo/metal/$port/localfold-$port" "$bin/localfold-$port"
  strip -x "$bin/localfold-$port"
done
# each port's list of the kernel specialisations its folds use, where this machine's folds have recorded one for these
# very kernels (~/.cache/localfold/metal, named by the kernels' hash): shipped, a user's first fold compiles them all up
# front and in parallel instead of one at a time as each is first reached (run metal/<port>/gate.py first to fill it)
mkdir -p "$bin/metal-specs"; rm -f "$bin"/metal-specs/*.specs
for port in af3 af2 ef2; do
  name="$(LOCALFOLD_METAL_SPECS_NAME=1 "$bin/localfold-$port")"
  if [ -f "$HOME/.cache/localfold/metal/$name" ]; then
    cp "$HOME/.cache/localfold/metal/$name" "$bin/metal-specs/$name"
    echo "$port: $(wc -l < "$bin/metal-specs/$name" | tr -d ' ') kernel specialisations shipped"
  else echo "$port: no recorded specialisations for $name - its first fold compiles each kernel as it is reached"; fi
done
# localfold-fetch: the featuriser binary under the name that runs its weight and CCD fetcher
install -m 755 "$repo/cuda/featurise/featurise" "$bin/localfold-fetch"; strip -x "$bin/localfold-fetch"
# what the wheel's tag promises: arm64, nothing newer than macOS 13, and nothing outside the system
for f in "$bin"/localfold-*; do   # (the binaries; metal-specs/ is data)
  arch="$(lipo -archs "$f")"; [ "$arch" = arm64 ] || { echo "$f is $arch, not arm64" >&2; exit 1; }
  minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/ { found = 1 } found && /minos/ { print $2; exit }')"
  [ "$(printf '%s\n13.0\n' "$minos" | sort -V | tail -1)" = 13.0 ] || { echo "$f needs macOS $minos, past 13.0" >&2; exit 1; }
  if otool -L "$f" | tail -n +2 | grep -vE '^\s*/(System/Library|usr/lib)/'; then echo "$f links outside the system" >&2; exit 1; fi
  echo "$(basename "$f"): macOS $minos, $(du -m "$f" | cut -f1) MB"
done
# the website `localfold serve` serves: tools/build_site.py's dist/, the page as Pages publishes it
python3 "$repo/tools/build_site.py" > /dev/null
rm -rf "$here/localfold/site"; cp -R "$repo/dist" "$here/localfold/site"
echo "site: $(find "$here/localfold/site" -type f | wc -l | tr -d ' ') files, $(du -sm "$here/localfold/site" | cut -f1) MB"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
# (in a venv of its own: a Homebrew or system Python refuses a pip install outside one, PEP 668)
"$py" -m venv "$work/venv"; py="$work/venv/bin/python"
"$py" -m pip install -q "wheel>=0.40" "setuptools>=68"
"$py" -m pip wheel --no-deps --no-build-isolation -w "$work" "$here"
"$py" -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag macosx_13_0_arm64 "$work"/localfold-*.whl
mv "$work"/localfold-*-py3-none-macosx_13_0_arm64.whl "$out/"
ls -la "$out"/localfold-*.whl

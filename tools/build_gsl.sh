#!/usr/bin/env bash
# Build the pinned, unmodified GSL source that accompanies the release.
set -euo pipefail
project_root=$(cd "$(dirname "$0")/.." && pwd)
gsl_prefix=${GSL_DIR:-/tmp/gaiamock-gsl-2.8}
archive="$project_root/vendor/gsl-2.8.tar.gz"
expected=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190
if command -v sha256sum >/dev/null; then
    actual=$(sha256sum "$archive")
else
    actual=$(shasum -a 256 "$archive")
fi
if [[ ${actual%% *} != "$expected" ]]; then
    echo "GSL source checksum mismatch" >&2
    exit 1
fi
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/gaiamock-gsl.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT
tar -xzf "$archive" -C "$build_dir"
cd "$build_dir/gsl-2.8"
configure_args=(--prefix="$gsl_prefix" --disable-static --enable-shared)
# GSL's bundled config.sub uses the GNU name aarch64 for Apple Silicon.
if [[ $(uname -s) == Darwin && $(uname -m) == arm64 ]]; then
    configure_args+=(--build=aarch64-apple-darwin)
fi
./configure "${configure_args[@]}"
make -j "${GAIAMOCK_BUILD_JOBS:-4}"
make install
"$gsl_prefix/bin/gsl-config" --version

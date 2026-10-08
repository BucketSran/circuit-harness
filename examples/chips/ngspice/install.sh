#!/usr/bin/env bash
# Offline, user-owned Linux installation. Download the official tarball separately.
set -euo pipefail
umask 077
# EDA login environments can add network-mounted compiler/library search paths.
# Keep this build reproducible without changing the caller or system environment.
if [ "${CHIPS_BUILD_CLEAN_ENV:-}" != 1 ]; then
  exec env -i HOME="$HOME" PATH=/usr/bin:/bin LC_ALL=C CHIPS_BUILD_CLEAN_ENV=1 \
    bash "$0" "$@"
fi
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: bash install.sh /absolute/private-root /path/to/ngspice-47.tar.gz [private-build-parent]" >&2
  exit 2
fi
root=$1
archive=$(realpath "$2")
case "$root" in /*) ;; *) echo "private-root must be absolute" >&2; exit 2 ;; esac
test -d "$root" && test ! -L "$root"
test "$(stat -c %u "$root")" = "$(id -u)"
test "$(stat -c %a "$root")" = 700
root=$(realpath "$root")
checksum=894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f
echo "$checksum  $archive" | sha256sum --check --status
prefix="$root/envs/ngspice/47"
build_parent=${3:-"$root/build"}
mkdir -p "$build_parent"
test ! -L "$build_parent" && test "$(stat -c %u "$build_parent")" = "$(id -u)"
test "$(stat -c %a "$build_parent")" = 700
build="$(realpath "$build_parent")/ngspice-47"
# Never overwrite an existing build or installation, including an incomplete one.
test ! -e "$prefix" && test ! -L "$prefix"
test ! -e "$build" && test ! -L "$build"
mkdir -p "$root/envs/ngspice" "$root/build"
mkdir "$build"
tar -xzf "$archive" -C "$build"
mkdir "$build/release"
cd "$build/release"
printf '%s\n' "$checksum" > ../source.sha256
gcc --version > ../compiler.txt
# Headless, single-thread build: no X11/readline/FFTW development packages needed.
../ngspice-47/configure --prefix="$prefix" --without-x --with-readline=no \
  --without-fftw3 --disable-openmp CFLAGS='-O2' > ../configure.log 2>&1
/usr/bin/time -v -o ../build-resources.txt timeout 1800 make -j2 > ../make.log 2>&1
make install > ../install.log 2>&1
"$prefix/bin/ngspice" -v > ../version.txt
sha256sum "$prefix/bin/ngspice" > ../binary.sha256
du -sk "$prefix" "$build" > ../disk-kib.txt
cat ../version.txt ../disk-kib.txt

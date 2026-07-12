#!/bin/bash
# scripts/setup/setup_lmdb.sh
# Install liblmdb for the plawk multi-pass cache LMDB backend.
#
# The plawk LLVM/WAM target has two persistent-cache backends
# (PLAWK_MULTIPASS_CACHE.md): a portable file backend (the default, pure
# LLVM IR, no dependency) and an LMDB backend (`BEGIN cache("x.lmdb")
# backend "lmdb")`). Only the LMDB backend needs liblmdb, and only at build
# and run time for programs that select it. This script installs the dev
# package (header + shared lib) so `clang ... -llmdb` links.

set -e

echo "=========================================="
echo "liblmdb setup for UnifyWeaver (plawk LMDB cache backend)"
echo "=========================================="
echo ""

# Already present?
if [ -f /usr/include/lmdb.h ] || [ -f /usr/local/include/lmdb.h ]; then
    if ldconfig -p 2>/dev/null | grep -qi 'liblmdb'; then
        echo "✅ liblmdb already installed (header + shared library present)."
        exit 0
    fi
fi

install_apt()    { sudo apt-get update -qq && sudo apt-get install -y liblmdb-dev; }
install_dnf()    { sudo dnf install -y lmdb-devel; }
install_yum()    { sudo yum install -y lmdb-devel; }
install_pacman() { sudo pacman -S --noconfirm lmdb; }
install_apk()    { sudo apk add lmdb-dev; }
install_brew()   { brew install lmdb; }

if   command -v apt-get >/dev/null 2>&1; then echo "Using apt-get...";  install_apt
elif command -v dnf     >/dev/null 2>&1; then echo "Using dnf...";      install_dnf
elif command -v yum     >/dev/null 2>&1; then echo "Using yum...";      install_yum
elif command -v pacman  >/dev/null 2>&1; then echo "Using pacman...";   install_pacman
elif command -v apk     >/dev/null 2>&1; then echo "Using apk...";      install_apk
elif command -v brew    >/dev/null 2>&1; then echo "Using brew...";     install_brew
else
    echo "❌ No supported package manager found."
    echo "   Install liblmdb (dev package) manually: it must provide lmdb.h"
    echo "   and a linkable liblmdb so 'clang ... -llmdb' works."
    exit 1
fi

echo ""
if [ -f /usr/include/lmdb.h ] || [ -f /usr/local/include/lmdb.h ]; then
    echo "✅ liblmdb installed. plawk programs using backend \"lmdb\" can now build."
else
    echo "⚠️  Install completed but lmdb.h was not found in a standard include path."
    echo "   You may need to point clang at its location."
fi

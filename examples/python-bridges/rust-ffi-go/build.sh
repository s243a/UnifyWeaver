#!/bin/bash
# Build script for Go + Rust FFI + RPyC example
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Rust FFI Bridge ==="
cargo build --release

echo ""
echo "=== Copying library ==="
cp target/release/librpyc_bridge.so .
echo "  Copied librpyc_bridge.so"

echo ""
echo "=== Building Go example ==="
CGO_ENABLED=1 go build -o rpyc_example rpyc_client.go
echo "  Built rpyc_example"

echo ""
echo "=== Build complete ==="
echo ""
echo "To run:"
echo "  1. Start RPyC server: python ../../../examples/rpyc-integration/rpyc_server.py"
echo "  2. Run: LD_LIBRARY_PATH=. ./rpyc_example"

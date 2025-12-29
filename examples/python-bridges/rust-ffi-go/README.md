# Go + Rust FFI Bridge + RPyC Example

Access Python's ML/AI ecosystem from Go via a Rust FFI bridge.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│ Go Application                                         │
│ ┌────────────────────────────────────────────────────┐ │
│ │ CGO (C FFI)                                        │ │
│ │ ┌────────────────────────────────────────────────┐ │ │
│ │ │ Rust cdylib (librpyc_bridge.so)                │ │ │
│ │ │ ┌────────────────────────────────────────────┐ │ │ │
│ │ │ │ PyO3 (CPython embedding)                   │ │ │ │
│ │ │ │ ┌────────────────────────────────────────┐ │ │ │ │
│ │ │ │ │ RPyC Client                            │ │ │ │ │
│ │ │ │ └────────────────────────────────────────┘ │ │ │ │
│ │ │ └────────────────────────────────────────────┘ │ │ │
│ │ └────────────────────────────────────────────────┘ │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
                              │
                              │ TCP (live proxies)
                              ▼
┌────────────────────────────────────────────────────────┐
│ RPyC Server (Python)                                   │
│ - NumPy, SciPy, pandas, scikit-learn, PyTorch         │
│ - Custom ML services                                   │
└────────────────────────────────────────────────────────┘
```

## Why This Approach?

Go lacks mature CPython embedding libraries:
- **go-python3** (DataDog): Archived since 2021, complex GIL management
- **go-embed-python** (kluctl): Uses subprocess (loses live proxy benefit)

Rust's PyO3 is mature, actively maintained, and handles GIL properly. By using
Rust as a bridge, Go gets stable access to Python via C FFI (CGO).

## Prerequisites

1. **Rust toolchain**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Go 1.18+**
   ```bash
   go version  # Should be 1.18+
   ```

3. **Python 3.8+ with rpyc**
   ```bash
   pip install rpyc numpy
   ```

4. **Python dev headers** (for PyO3)
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-dev

   # Fedora/RHEL
   sudo dnf install python3-devel
   ```

## Build

```bash
# 1. Build the Rust FFI bridge
cargo build --release

# 2. Copy the library and header to Go's directory
cp target/release/librpyc_bridge.so .
# (rpyc_bridge.h is already in place)

# 3. Build the Go example (requires CGO)
CGO_ENABLED=1 go build -o rpyc_example rpyc_client.go
```

## Run

```bash
# 1. Start RPyC server (in another terminal)
python ../../../examples/rpyc-integration/rpyc_server.py

# 2. Run the Go example
LD_LIBRARY_PATH=. ./rpyc_example
```

## Expected Output

```
Go + Rust FFI + RPyC Integration
================================

Connecting to RPyC server...
  Connected!

Test 1: math.sqrt(16)
  Result: 4.0

Test 2: numpy.mean([1,2,3,4,5])
  Result: 3.0

Test 3: math.pi
  Result: 3.141592653589793

================================
All tests passed!
```

## API Reference

The Rust FFI bridge provides these C functions:

| Function | Description |
|----------|-------------|
| `rpyc_init()` | Initialize Python runtime |
| `rpyc_connect(host, port)` | Connect to RPyC server |
| `rpyc_disconnect()` | Disconnect from server |
| `rpyc_call(module, func, args_json)` | Call remote function |
| `rpyc_getattr(module, attr)` | Get remote attribute |
| `rpyc_free_string(s)` | Free returned string |
| `rpyc_is_connected()` | Check connection status |

## Go Wrapper

The `rpyc_client.go` provides a Go-idiomatic wrapper:

```go
client, err := NewRPyCClient("localhost", 18812)
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// Call Python function
result, err := client.Call("numpy", "mean", []float64{1, 2, 3, 4, 5})

// Get Python attribute
pi, err := client.GetAttr("math", "pi")
```

## Troubleshooting

### "error while loading shared libraries: librpyc_bridge.so"
Set `LD_LIBRARY_PATH`:
```bash
LD_LIBRARY_PATH=. ./rpyc_example
```

### "failed to connect to RPyC server"
Ensure the RPyC server is running:
```bash
python ../../../examples/rpyc-integration/rpyc_server.py
```

### CGO build errors
Ensure Python dev headers are installed and CGO is enabled:
```bash
CGO_ENABLED=1 go build ...
```

## Files

| File | Description |
|------|-------------|
| `src/lib.rs` | Rust FFI bridge (generated) |
| `Cargo.toml` | Rust project config (generated) |
| `rpyc_bridge.h` | C header for FFI (generated) |
| `rpyc_client.go` | Go client wrapper (generated) |

All files were generated from Prolog:
```prolog
?- generate_rust_ffi_bridge([host("localhost"), port(18812)], RustCode).
?- generate_go_ffi_client([lib_name(rpyc_bridge)], GoCode).
```

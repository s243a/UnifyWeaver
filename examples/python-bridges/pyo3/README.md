# PyO3 + RPyC Example

Use PyO3 to embed CPython in Rust and access RPyC.

## Status

| Feature | Status |
|---------|--------|
| PyO3 binding | ✅ Mature, well-supported |
| RPyC access | ✅ Tested and working |

## Overview

[PyO3](https://pyo3.rs/) is the de-facto standard for Rust-Python interop.
It embeds real CPython, allowing full access to RPyC's live object proxies.

## Prerequisites

### 1. Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Python 3.8+ with rpyc

```bash
pip install rpyc
```

### 3. RPyC Server Running

```bash
python examples/rpyc-integration/rpyc_server.py
```

## Project Setup

### Cargo.toml

```toml
[package]
name = "rpyc-client"
version = "0.1.0"
edition = "2021"

[dependencies]
pyo3 = { version = "0.22", features = ["auto-initialize"] }
```

### src/main.rs

```rust
use pyo3::prelude::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Import rpyc (use import_bound for PyO3 0.21+)
        let rpyc = py.import_bound("rpyc")?;
        let classic = rpyc.getattr("classic")?;

        // Connect to server
        let conn = classic.call_method1("connect", ("localhost", 18812))?;

        // Access remote math module
        let modules = conn.getattr("modules")?;
        let math = modules.getattr("math")?;
        let result: f64 = math.call_method1("sqrt", (16,))?.extract()?;

        println!("math.sqrt(16) = {}", result);  // 4.0

        // Access remote numpy
        let np = modules.getattr("numpy")?;
        let arr = np.call_method1("array", (vec![1, 2, 3, 4, 5],))?;
        let mean: f64 = np.call_method1("mean", (arr,))?.extract()?;

        println!("numpy.mean([1,2,3,4,5]) = {}", mean);  // 3.0

        // Close connection
        conn.call_method0("close")?;

        Ok(())
    })
}
```

## Running

```bash
cd examples/python-bridges/pyo3
cargo run
```

## Expected Output

```
math.sqrt(16) = 4.0
numpy.mean([1,2,3,4,5]) = 3.0
```

## Key Concepts

### Automatic GIL Management

PyO3's `Python::with_gil()` automatically acquires and releases the Python
Global Interpreter Lock (GIL).

### Type Extraction

PyO3 can extract Python values to Rust types:

```rust
let result: f64 = math.call_method1("sqrt", (16,))?.extract()?;
```

### Error Handling

PyO3 uses Rust's `Result` type for Python exceptions:

```rust
fn connect_rpyc() -> PyResult<()> {
    // Errors automatically convert to PyResult
    let conn = classic.call_method1("connect", ("localhost", 18812))?;
    Ok(())
}
```

## Advanced: Wrapping in a Rust Struct

```rust
use pyo3::prelude::*;

struct RPyCClient {
    conn: Py<PyAny>,
}

impl RPyCClient {
    fn connect(host: &str, port: u16) -> PyResult<Self> {
        Python::with_gil(|py| {
            let rpyc = py.import_bound("rpyc")?;
            let conn = rpyc.getattr("classic")?.call_method1("connect", (host, port))?;
            Ok(Self { conn: conn.into() })
        })
    }

    fn sqrt(&self, value: f64) -> PyResult<f64> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let math = conn.getattr("modules")?.getattr("math")?;
            math.call_method1("sqrt", (value,))?.extract()
        })
    }

    fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.conn.bind(py).call_method0("close")?;
            Ok(())
        })
    }
}

impl Drop for RPyCClient {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
```

## PyO3 vs Other Bridges

| Aspect | PyO3 (Rust) | JPype (Java) | Python.NET (C#) |
|--------|-------------|--------------|-----------------|
| Type Safety | Strong | Dynamic | Dynamic |
| Memory Safety | Guaranteed | JVM managed | CLR managed |
| Performance | Excellent | Good | Good |
| Ecosystem | Growing | Mature | Mature |

## Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [PyO3 GitHub](https://github.com/pyo3/pyo3)
- [PyO3 User Guide](https://pyo3.rs/main/python-from-rust)

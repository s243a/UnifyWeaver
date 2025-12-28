//! PyO3 + RPyC Example
//!
//! Demonstrates using PyO3 to embed Python in Rust and access RPyC
//! for remote Python computation.
//!
//! Usage:
//!   1. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
//!   2. Run: cargo run

use pyo3::prelude::*;

/// Connect to RPyC server and run tests
fn main() -> PyResult<()> {
    println!("PyO3 + RPyC Integration");
    println!("=======================\n");

    Python::with_gil(|py| {
        // Import rpyc module
        let rpyc = py.import("rpyc")?;
        let classic = rpyc.getattr("classic")?;

        println!("Connecting to RPyC server...");

        // Connect to server
        let conn = match classic.call_method1("connect", ("localhost", 18812_u16)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to connect: {}", e);
                eprintln!("\nMake sure RPyC server is running:");
                eprintln!("  python examples/rpyc-integration/rpyc_server.py");
                return Err(e);
            }
        };

        // Get remote modules
        let modules = conn.getattr("modules")?;

        // Test 1: math.sqrt
        println!("\nTest 1: Remote math.sqrt");
        let math = modules.getattr("math")?;
        let result: f64 = math.call_method1("sqrt", (16.0_f64,))?.extract()?;
        println!("  math.sqrt(16) = {}", result);
        assert!((result - 4.0).abs() < 0.001, "Expected 4.0");
        println!("  ✓ Passed");

        // Test 2: numpy.mean
        println!("\nTest 2: Remote numpy.mean");
        let np = modules.getattr("numpy")?;
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = np.call_method1("array", (values,))?;
        let mean: f64 = np.call_method1("mean", (arr,))?.extract()?;
        println!("  numpy.mean([1,2,3,4,5]) = {}", mean);
        assert!((mean - 3.0).abs() < 0.001, "Expected 3.0");
        println!("  ✓ Passed");

        // Test 3: Get server Python version
        println!("\nTest 3: Server Python version");
        let sys = modules.getattr("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        let version_short = version.split_whitespace().next().unwrap_or(&version);
        println!("  Server Python: {}", version_short);
        println!("  ✓ Passed");

        // Close connection
        conn.call_method0("close")?;

        println!("\n=======================");
        println!("All tests passed!");

        Ok(())
    })
}

/// Wrapper struct for RPyC connection (for more complex use cases)
#[allow(dead_code)]
struct RPyCClient {
    conn: Py<PyAny>,
}

#[allow(dead_code)]
impl RPyCClient {
    /// Connect to an RPyC server
    fn connect(host: &str, port: u16) -> PyResult<Self> {
        Python::with_gil(|py| {
            let rpyc = py.import("rpyc")?;
            let conn = rpyc
                .getattr("classic")?
                .call_method1("connect", (host, port))?;
            Ok(Self { conn: conn.into() })
        })
    }

    /// Call remote math.sqrt
    fn sqrt(&self, value: f64) -> PyResult<f64> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let math = conn.getattr("modules")?.getattr("math")?;
            math.call_method1("sqrt", (value,))?.extract()
        })
    }

    /// Call remote numpy.mean
    fn numpy_mean(&self, values: Vec<f64>) -> PyResult<f64> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let np = conn.getattr("modules")?.getattr("numpy")?;
            let arr = np.call_method1("array", (values,))?;
            np.call_method1("mean", (arr,))?.extract()
        })
    }

    /// Get remote Python version
    fn python_version(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let sys = conn.getattr("modules")?.getattr("sys")?;
            let version: String = sys.getattr("version")?.extract()?;
            Ok(version.split_whitespace().next().unwrap_or(&version).to_string())
        })
    }

    /// Close the connection
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

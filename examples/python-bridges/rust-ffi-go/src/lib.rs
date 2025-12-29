//! Rust FFI Bridge for RPyC
//!
//! This library provides C-compatible FFI functions that any language
//! can use to access Python via RPyC. Uses PyO3 for CPython embedding.
//!
//! Compile: cargo build --release
//! Output: target/release/librpyc_bridge.so (or .dylib/.dll)

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

// Global connection state (thread-safe)
static CONNECTION: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

/// Initialize Python runtime. Call once at startup.
#[no_mangle]
pub extern "C" fn rpyc_init() -> i32 {
    match pyo3::prepare_freethreaded_python() {
        () => 0,
    }
}

/// Connect to RPyC server. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn rpyc_connect(host: *const c_char, port: i32) -> i32 {
    let host_str = if host.is_null() {
        "localhost"
    } else {
        match unsafe { CStr::from_ptr(host) }.to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    let port_num = if port <= 0 { 18812 } else { port };

    Python::with_gil(|py| {
        match py.import_bound("rpyc") {
            Ok(rpyc) => {
                match rpyc.getattr("classic") {
                    Ok(classic) => {
                        match classic.call_method1("connect", (host_str, port_num)) {
                            Ok(conn) => {
                                let mut guard = CONNECTION.lock().unwrap();
                                *guard = Some(conn.unbind());
                                0
                            }
                            Err(_) => -1,
                        }
                    }
                    Err(_) => -1,
                }
            }
            Err(_) => -1,
        }
    })
}

/// Disconnect from RPyC server.
#[no_mangle]
pub extern "C" fn rpyc_disconnect() {
    Python::with_gil(|py| {
        let mut guard = CONNECTION.lock().unwrap();
        if let Some(ref conn) = *guard {
            let _ = conn.bind(py).call_method0("close");
        }
        *guard = None;
    });
}

/// Call a function on a remote module.
/// Returns JSON string (caller must free with rpyc_free_string).
///
/// Example: rpyc_call("math", "sqrt", "[16]") -> "4.0"
#[no_mangle]
pub extern "C" fn rpyc_call(
    module: *const c_char,
    func: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    if module.is_null() || func.is_null() {
        return ptr::null_mut();
    }

    let module_str = match unsafe { CStr::from_ptr(module) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let func_str = match unsafe { CStr::from_ptr(func) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let args_str = if args_json.is_null() {
        "[]"
    } else {
        match unsafe { CStr::from_ptr(args_json) }.to_str() {
            Ok(s) => s,
            Err(_) => "[]",
        }
    };

    Python::with_gil(|py| {
        let guard = CONNECTION.lock().unwrap();
        let conn = match guard.as_ref() {
            Some(c) => c.bind(py),
            None => return ptr::null_mut(),
        };

        // Get remote module
        let modules = match conn.getattr("modules") {
            Ok(m) => m,
            Err(_) => return ptr::null_mut(),
        };

        let remote_module = match modules.getattr(module_str) {
            Ok(m) => m,
            Err(_) => return ptr::null_mut(),
        };

        // Parse args from JSON
        let json_mod = match py.import_bound("json") {
            Ok(j) => j,
            Err(_) => return ptr::null_mut(),
        };

        let args_list = match json_mod.call_method1("loads", (args_str,)) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };

        // Call remote function
        let result = match remote_module.call_method1(func_str, (args_list,)) {
            Ok(r) => r,
            Err(_) => {
                // Try calling without unpacking (single arg or no args)
                match remote_module.getattr(func_str) {
                    Ok(f) => {
                        let args_tuple: &PyAny = args_list.downcast().unwrap_or(&args_list);
                        match f.call1((args_tuple,)) {
                            Ok(r) => r,
                            Err(_) => match f.call0() {
                                Ok(r) => r,
                                Err(_) => return ptr::null_mut(),
                            }
                        }
                    }
                    Err(_) => return ptr::null_mut(),
                }
            }
        };

        // Serialize result to JSON
        let result_json = match json_mod.call_method1("dumps", (&result,)) {
            Ok(j) => j,
            Err(_) => {
                // Fallback: convert to string
                match result.str() {
                    Ok(s) => s.into_any(),
                    Err(_) => return ptr::null_mut(),
                }
            }
        };

        match result_json.extract::<String>() {
            Ok(s) => {
                match CString::new(s) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => ptr::null_mut(),
                }
            }
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Get attribute from remote module (e.g., numpy.pi).
/// Returns JSON string (caller must free with rpyc_free_string).
#[no_mangle]
pub extern "C" fn rpyc_getattr(
    module: *const c_char,
    attr: *const c_char,
) -> *mut c_char {
    if module.is_null() || attr.is_null() {
        return ptr::null_mut();
    }

    let module_str = match unsafe { CStr::from_ptr(module) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let attr_str = match unsafe { CStr::from_ptr(attr) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    Python::with_gil(|py| {
        let guard = CONNECTION.lock().unwrap();
        let conn = match guard.as_ref() {
            Some(c) => c.bind(py),
            None => return ptr::null_mut(),
        };

        let modules = match conn.getattr("modules") {
            Ok(m) => m,
            Err(_) => return ptr::null_mut(),
        };

        let remote_module = match modules.getattr(module_str) {
            Ok(m) => m,
            Err(_) => return ptr::null_mut(),
        };

        let value = match remote_module.getattr(attr_str) {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };

        // Serialize to JSON
        let json_mod = match py.import_bound("json") {
            Ok(j) => j,
            Err(_) => return ptr::null_mut(),
        };

        let result_json = match json_mod.call_method1("dumps", (&value,)) {
            Ok(j) => j,
            Err(_) => {
                match value.str() {
                    Ok(s) => s.into_any(),
                    Err(_) => return ptr::null_mut(),
                }
            }
        };

        match result_json.extract::<String>() {
            Ok(s) => {
                match CString::new(s) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => ptr::null_mut(),
                }
            }
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Free a string returned by rpyc_call or rpyc_getattr.
#[no_mangle]
pub extern "C" fn rpyc_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

/// Check if connected to RPyC server.
#[no_mangle]
pub extern "C" fn rpyc_is_connected() -> i32 {
    let guard = CONNECTION.lock().unwrap();
    if guard.is_some() { 1 } else { 0 }
}

/// Get last error message (placeholder - always returns null for now).
#[no_mangle]
pub extern "C" fn rpyc_last_error() -> *const c_char {
    ptr::null()
}

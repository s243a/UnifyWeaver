//! LLVM FFI Integration Test for Rust
//! Tests calling LLVM-compiled Prolog predicates via FFI

mod ffi {
    extern "C" {
        pub fn sum(n: i64) -> i64;
        pub fn factorial(n: i64) -> i64;
    }
}

/// Calls the LLVM-compiled sum function (tail recursion with musttail)
pub fn sum(n: i64) -> i64 {
    unsafe { ffi::sum(n) }
}

/// Calls the LLVM-compiled factorial function
pub fn factorial(n: i64) -> i64 {
    unsafe { ffi::factorial(n) }
}

fn main() {
    println!("===========================================");
    println!("LLVM FFI Integration Test - Rust");
    println!("===========================================");
    println!();

    let tests: Vec<(&str, fn(i64) -> i64, i64, i64)> = vec![
        ("sum(10)", sum, 10, 55),
        ("sum(100)", sum, 100, 5050),
        ("factorial(5)", factorial, 5, 120),
        ("factorial(10)", factorial, 10, 3628800),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, func, input, expected) in tests {
        let result = func(input);
        if result == expected {
            println!("[PASS] {} = {}", name, result);
            passed += 1;
        } else {
            println!("[FAIL] {} = {} (expected {})", name, result, expected);
            failed += 1;
        }
    }

    println!();
    println!("Results: {} passed, {} failed", passed, failed);
    println!("===========================================");

    if failed > 0 {
        println!("INTEGRATION TEST FAILED");
        std::process::exit(1);
    } else {
        println!("INTEGRATION TEST PASSED");
    }
}

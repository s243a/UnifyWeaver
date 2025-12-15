fn main() {
    // Tell cargo to look for the prolog_math library in the parent directory
    println!("cargo:rustc-link-search=native=../");
    println!("cargo:rustc-link-lib=dylib=prolog_math");
    
    // Add rpath for runtime library search
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../");
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// Rust Display length oracle for WAM-C atom_length/2 float cases.
// rustc atom_length_float_oracle.rs && ./atom_length_float_oracle

fn emit(id: &str, f: f64) {
    println!("CASE {}", id);
    println!("STATUS ok");
    println!("TERM {}", format!("{}", f).chars().count());
}

fn grid() {
    const MANTS: [u64; 6] = [0, 1, 2, 3, 1 << 51, (1 << 52) - 1];
    for exp in 0u64..2048 {
        for &mant0 in &MANTS {
            let mant = if exp == 0x7ff && mant0 > 1 { 1 } else { mant0 };
            let bits = (exp << 52) | mant;
            let f = f64::from_bits(bits);
            emit(&format!("oracle_{:016x}", bits), f);
            if exp != 0x7ff || mant == 0 {
                let nbits = bits | (1u64 << 63);
                emit(&format!("oracle_{:016x}", nbits), f64::from_bits(nbits));
            }
        }
    }
    for center in [1e-6_f64, 1.0, 1e6, 1e23, 1e24, 1e28, 1e100] {
        let bits = center.to_bits();
        for delta in -1_i64..=1 {
            let candidate = (bits as i64 + delta) as u64;
            emit(&format!("power_{:016x}", candidate), f64::from_bits(candidate));
            let negative = candidate | (1_u64 << 63);
            emit(&format!("power_{:016x}", negative), f64::from_bits(negative));
        }
    }
}

fn named() {
    emit("float_input", 3.5);
    emit("float_1e6", 1_000_000.0);
    emit("float_1e_minus_6", 0.000001);
    emit("float_1_23456789", 1.23456789);
    emit("float_1e100", 1e100);
    emit("float_min_subnormal", f64::from_bits(1));
    emit("float_neg", -3.5);
    emit("float_zero", 0.0);
    emit("float_neg_zero", -0.0);
    emit("float_inf", f64::INFINITY);
    emit("float_neg_inf", f64::NEG_INFINITY);
    emit("float_nan", f64::NAN);
    emit("float_tenth", 0.1);
    emit("float_one", 1.0);
    emit("float_pow2", 2.0);
    emit("float_three_tenths", 0.3);
}

fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    match args.next().as_deref() {
        Some("grid") => grid(),
        _ => named(),
    }
}

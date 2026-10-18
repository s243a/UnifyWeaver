use std::io::{self, BufRead};

fn main() {
let s: f64 = io::stdin().lock().lines().filter_map(|l| l.ok()?.trim().parse().ok()).sum(); println!("{}", s);}

use std::io::{self, BufRead};
use std::collections::HashSet;

fn main() {
    let stdin = io::stdin();
    let mut seen = HashSet::new();

    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split(":").collect();
            if parts.len() == 2 {




                let result = format!("{}:{}", parts[1], parts[0]);
            if seen.insert(result.clone()) {
                println!("{}", result);
            }
            }
        }
    }
}

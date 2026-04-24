// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// mysql_stream CLI — reads a MySQL dump (optionally gzipped) and writes
// every INSERT row as TAB-separated TSV on stdout.
//
// Usage: mysql_stream <path-to-dump.sql[.gz]>

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use mysql_stream::{iter_mysql_rows, Field};

fn tsv_escape_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    // TSV convention:
    //   \\ \t \n \r as two-byte escapes
    //   non-printable ASCII and non-UTF-8 bytes as \xNN (hex)
    //   valid UTF-8 passes through verbatim
    for &b in bytes {
        match b {
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\t' => out.extend_from_slice(b"\\t"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            0x20..=0x7E => out.push(b),  // printable ASCII
            _ => {
                // Non-ASCII or non-printable — emit \xNN.
                // Note: this over-escapes valid multi-byte UTF-8. The
                // alternative (validate UTF-8 chunks, pass through valid
                // ones) is possible but trades speed for readability.
                // For benchmark consumers that only read ASCII-safe
                // columns (cl_type, integer IDs), this is fine.
                static HEX: &[u8; 16] = b"0123456789abcdef";
                out.extend_from_slice(b"\\x");
                out.push(HEX[(b >> 4) as usize]);
                out.push(HEX[(b & 0xf) as usize]);
            }
        }
    }
}

fn write_field<W: Write>(out: &mut W, f: &Field) -> io::Result<()> {
    match f {
        Field::Int(n) => write!(out, "{}", n),
        Field::Str(bytes) => {
            let mut buf = Vec::with_capacity(bytes.len());
            tsv_escape_bytes(bytes, &mut buf);
            out.write_all(&buf)
        }
        Field::Null => out.write_all(b"\\N"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: mysql_stream <path-to-dump.sql[.gz]>");
        return ExitCode::from(2);
    }
    let path = &args[1];

    let iter = match iter_mysql_rows(path) {
        Ok(it) => it,
        Err(e) => {
            eprintln!("mysql_stream: cannot open {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 16, stdout.lock());

    let handle_err = |e: &io::Error| -> Option<ExitCode> {
        if matches!(e.kind(), io::ErrorKind::BrokenPipe) {
            return Some(ExitCode::SUCCESS);
        }
        eprintln!("mysql_stream: write error: {}", e);
        Some(ExitCode::from(1))
    };

    for row in iter {
        for (i, field) in row.iter().enumerate() {
            if i > 0 {
                if let Err(e) = out.write_all(b"\t") {
                    if let Some(code) = handle_err(&e) { return code; }
                }
            }
            if let Err(e) = write_field(&mut out, field) {
                if let Some(code) = handle_err(&e) { return code; }
            }
        }
        if let Err(e) = out.write_all(b"\n") {
            if let Some(code) = handle_err(&e) { return code; }
        }
    }

    ExitCode::SUCCESS
}

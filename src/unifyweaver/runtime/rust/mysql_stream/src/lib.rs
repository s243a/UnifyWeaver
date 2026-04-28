// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// mysql_stream — streaming parser for MySQL INSERT dumps.
//
// This is a leaf primitive for the UnifyWeaver cross-target streaming
// glue: byte-level tokenization only, no filtering, no column selection.
// Consumers (Prolog / Python / etc.) apply logic at their layer.
//
// See docs/design/cross-target-glue/streaming-pipelines/ for the design.

use std::io::{self, BufRead, BufReader, Read};
use std::fs::File;

use flate2::read::GzDecoder;

/// A single column value from a MySQL INSERT row.
///
/// `Str` holds raw bytes rather than a String because MySQL VARBINARY
/// columns (e.g. MediaWiki's `cl_sortkey`) contain arbitrary bytes that
/// aren't necessarily valid UTF-8. Consumers decode as needed.
#[derive(Debug, Clone, PartialEq)]
pub enum Field {
    /// Integer literal (unquoted digits, optional minus sign).
    Int(i64),
    /// Quoted string/bytes literal, with MySQL escape sequences already
    /// decoded. May or may not be valid UTF-8.
    Str(Vec<u8>),
    /// The literal `NULL`.
    Null,
}

impl Field {
    /// Borrow the field as `&[u8]` if it's a `Str`; otherwise `None`.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Field::Str(v) => Some(v),
            _ => None,
        }
    }

    /// Try to interpret the field as valid UTF-8. Returns `None` for
    /// non-`Str` fields or invalid UTF-8.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Field::Str(v) => std::str::from_utf8(v).ok(),
            _ => None,
        }
    }
}

/// Iterator over INSERT VALUES tuples in a MySQL dump.
///
/// Emits every row as `Vec<Field>` — no filtering, no column projection.
/// Skips the CREATE TABLE preamble and any non-INSERT statements.
pub struct MysqlInsertIter<R: BufRead> {
    reader: R,
    /// Buffer for the current INSERT statement being parsed.
    /// We accumulate into this and walk it tuple-by-tuple.
    buf: Vec<u8>,
    /// Current position within `buf` — advances as we yield tuples.
    pos: usize,
    /// `true` once we've seen the first INSERT keyword and are actively
    /// emitting tuples. Before that, we skip lines (preamble).
    in_insert: bool,
    /// `true` once the underlying reader has returned EOF.
    eof: bool,
}

impl<R: BufRead> MysqlInsertIter<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Vec::with_capacity(1 << 16),
            pos: 0,
            in_insert: false,
            eof: false,
        }
    }

    /// Refill the buffer by reading the next INSERT statement.
    /// A MySQL dump INSERT ends with `;` followed by newline. We read
    /// line-by-line until we hit one ending in `);` (the canonical shape).
    ///
    /// Returns `true` if we loaded a new INSERT, `false` at EOF.
    fn load_next_insert(&mut self) -> io::Result<bool> {
        self.buf.clear();
        self.pos = 0;
        self.in_insert = false;

        // Use raw bytes (not String) because MySQL dumps can contain
        // non-UTF-8 bytes in VARBINARY fields (e.g. MediaWiki cl_sortkey).
        let mut line: Vec<u8> = Vec::with_capacity(1 << 20);
        loop {
            line.clear();
            let n = self.reader.read_until(b'\n', &mut line)?;
            if n == 0 {
                self.eof = true;
                return Ok(self.in_insert && !self.buf.is_empty());
            }

            if !self.in_insert {
                if line.starts_with(b"INSERT INTO") {
                    self.in_insert = true;
                    if let Some(idx) = find_subsequence(&line, b" VALUES ") {
                        self.buf.extend_from_slice(
                            &line[idx + b" VALUES ".len()..]
                        );
                    } else {
                        self.buf.extend_from_slice(&line);
                    }
                }
                // else: preamble line, ignore
            } else {
                self.buf.extend_from_slice(&line);
            }

            if self.in_insert && line_trimmed_ends_with(&line, b';') {
                return Ok(true);
            }
        }
    }

    /// Parse a single `(v1, v2, ...)` tuple starting at `self.pos`.
    /// Advances `self.pos` past the closing `)` and any trailing `,` or `;`.
    /// Returns `Some(row)` on success, `None` if no more tuples in buffer.
    fn parse_tuple(&mut self) -> Option<Vec<Field>> {
        // Skip whitespace, commas, and trailing semicolon.
        while self.pos < self.buf.len() {
            match self.buf[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' | b',' | b';' => self.pos += 1,
                b'(' => break,
                _ => {
                    // Unknown filler — defensively advance.
                    self.pos += 1;
                }
            }
        }

        if self.pos >= self.buf.len() {
            return None;
        }

        if self.buf[self.pos] != b'(' {
            return None;
        }
        self.pos += 1;  // consume '('

        let mut row = Vec::with_capacity(8);
        loop {
            // Skip whitespace before a field.
            while self.pos < self.buf.len()
                && matches!(self.buf[self.pos], b' ' | b'\t')
            {
                self.pos += 1;
            }

            if self.pos >= self.buf.len() {
                // Truncated tuple — shouldn't happen on well-formed input.
                return None;
            }

            let field = self.parse_field()?;
            row.push(field);

            // Skip whitespace before separator.
            while self.pos < self.buf.len()
                && matches!(self.buf[self.pos], b' ' | b'\t')
            {
                self.pos += 1;
            }

            match self.buf.get(self.pos)? {
                b',' => {
                    self.pos += 1;
                    continue;
                }
                b')' => {
                    self.pos += 1;
                    return Some(row);
                }
                _ => return None,
            }
        }
    }

    /// Parse one field at `self.pos`. Advances past it.
    fn parse_field(&mut self) -> Option<Field> {
        let c = *self.buf.get(self.pos)?;
        match c {
            b'\'' => self.parse_string(),
            b'N' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_int(),
            _ => None,
        }
    }

    /// Parse `NULL` literal (case-sensitive in MySQL dumps).
    fn parse_null(&mut self) -> Option<Field> {
        if self.buf.get(self.pos..self.pos + 4)? == b"NULL" {
            self.pos += 4;
            Some(Field::Null)
        } else {
            None
        }
    }

    /// Parse an integer literal: optional `-`, then decimal digits.
    fn parse_int(&mut self) -> Option<Field> {
        let start = self.pos;
        if self.buf.get(self.pos)? == &b'-' {
            self.pos += 1;
        }
        let digit_start = self.pos;
        while let Some(&c) = self.buf.get(self.pos) {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == digit_start {
            return None;
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos]).ok()?;
        Some(Field::Int(s.parse().ok()?))
    }

    /// Parse a quoted string. Assumes `self.pos` points at the opening `'`.
    /// Handles MySQL escape sequences: `\'`, `\\`, `\n`, `\r`, `\t`, `\0`,
    /// `\"`, `\b`, `\Z`. Advances past the closing `'`.
    ///
    /// Returns `Field::Str(Vec<u8>)` — raw bytes, not UTF-8-validated.
    fn parse_string(&mut self) -> Option<Field> {
        self.pos += 1;  // consume opening '
        let mut out = Vec::with_capacity(32);
        loop {
            let c = *self.buf.get(self.pos)?;
            self.pos += 1;
            match c {
                b'\'' => {
                    // Closing quote — or escaped '' sequence (ANSI SQL style).
                    // MySQL dumps use \' normally, but defensively handle '' too.
                    if self.buf.get(self.pos) == Some(&b'\'') {
                        out.push(b'\'');
                        self.pos += 1;
                    } else {
                        return Some(Field::Str(out));
                    }
                }
                b'\\' => {
                    let esc = *self.buf.get(self.pos)?;
                    self.pos += 1;
                    let decoded = match esc {
                        b'\'' => b'\'',
                        b'"' => b'"',
                        b'\\' => b'\\',
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'0' => 0,
                        b'b' => 0x08,
                        b'Z' => 0x1A,
                        // Unknown escape — keep the byte literally (MySQL
                        // behavior for undefined escapes).
                        other => other,
                    };
                    out.push(decoded);
                }
                _ => out.push(c),
            }
        }
    }
}

/// Find the byte offset of `needle` within `haystack`, or `None`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// True if `line`, after trimming trailing whitespace/newline, ends
/// with byte `c`.
fn line_trimmed_ends_with(line: &[u8], c: u8) -> bool {
    let end = line.iter().rposition(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'));
    match end {
        Some(i) => line[i] == c,
        None => false,
    }
}

impl<R: BufRead> Iterator for MysqlInsertIter<R> {
    type Item = Vec<Field>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(row) = self.parse_tuple() {
                return Some(row);
            }
            // Current buffer exhausted — load the next INSERT.
            if self.eof {
                return None;
            }
            match self.load_next_insert() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(_) => return None,
            }
        }
    }
}

/// Open a file (optionally gzipped based on extension) and return an
/// iterator over INSERT rows. Path ending in `.gz` is auto-decompressed.
pub fn iter_mysql_rows(
    path: &str,
) -> io::Result<MysqlInsertIter<BufReader<Box<dyn Read + Send>>>> {
    let file = File::open(path)?;
    let reader: Box<dyn Read + Send> = if path.ends_with(".gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let buf = BufReader::with_capacity(1 << 20, reader);
    Ok(MysqlInsertIter::new(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(input: &str) -> Vec<Vec<Field>> {
        let reader = BufReader::new(Cursor::new(input.as_bytes()));
        MysqlInsertIter::new(reader).collect()
    }

    #[test]
    fn simple_int_row() {
        let input = "INSERT INTO `t` VALUES (1,2,3);\n";
        assert_eq!(
            parse(input),
            vec![vec![Field::Int(1), Field::Int(2), Field::Int(3)]]
        );
    }

    fn s(b: &[u8]) -> Field {
        Field::Str(b.to_vec())
    }

    #[test]
    fn simple_string_row() {
        let input = "INSERT INTO `t` VALUES ('foo','bar');\n";
        assert_eq!(
            parse(input),
            vec![vec![s(b"foo"), s(b"bar")]]
        );
    }

    #[test]
    fn mixed_types_with_null() {
        let input = "INSERT INTO `t` VALUES (42,'hello',NULL,-7);\n";
        assert_eq!(
            parse(input),
            vec![vec![
                Field::Int(42),
                s(b"hello"),
                Field::Null,
                Field::Int(-7)
            ]]
        );
    }

    #[test]
    fn escaped_quote_in_string() {
        let input = "INSERT INTO `t` VALUES ('it\\'s','a \\\"test\\\"');\n";
        assert_eq!(
            parse(input),
            vec![vec![s(b"it's"), s(b"a \"test\"")]]
        );
    }

    #[test]
    fn backslash_and_null_byte() {
        let input = "INSERT INTO `t` VALUES ('a\\\\b','c\\0d');\n";
        assert_eq!(
            parse(input),
            vec![vec![s(b"a\\b"), s(b"c\0d")]]
        );
    }

    #[test]
    fn multi_row_batch() {
        let input = "INSERT INTO `t` VALUES (1,'a'),(2,'b'),(3,'c');\n";
        assert_eq!(
            parse(input),
            vec![
                vec![Field::Int(1), s(b"a")],
                vec![Field::Int(2), s(b"b")],
                vec![Field::Int(3), s(b"c")],
            ]
        );
    }

    #[test]
    fn binary_bytes_in_string() {
        // MySQL VARBINARY columns (like MediaWiki cl_sortkey) can contain
        // arbitrary bytes including invalid UTF-8. Must not cause parse failure.
        let input = b"INSERT INTO `t` VALUES ('\xff\xfe\x00\x01bytes');\n";
        let reader = BufReader::new(Cursor::new(input.as_ref()));
        let rows: Vec<Vec<Field>> = MysqlInsertIter::new(reader).collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 1);
        match &rows[0][0] {
            Field::Str(b) => assert_eq!(b, b"\xff\xfe\x00\x01bytes"),
            _ => panic!("expected Str"),
        }
    }

    #[test]
    fn preamble_skipped() {
        let input = "\
-- MySQL dump
CREATE TABLE `t` (id INT);
LOCK TABLES `t` WRITE;
INSERT INTO `t` VALUES (1),(2);
UNLOCK TABLES;
";
        assert_eq!(parse(input), vec![vec![Field::Int(1)], vec![Field::Int(2)]]);
    }

    #[test]
    fn multiple_insert_statements() {
        let input = "\
INSERT INTO `t` VALUES (1,'a');
INSERT INTO `t` VALUES (2,'b');
";
        assert_eq!(
            parse(input),
            vec![
                vec![Field::Int(1), Field::Str("a".into())],
                vec![Field::Int(2), Field::Str("b".into())],
            ]
        );
    }

    #[test]
    fn empty_string() {
        let input = "INSERT INTO `t` VALUES ('');\n";
        assert_eq!(parse(input), vec![vec![s(b"")]]);
    }

    #[test]
    fn embedded_newline_in_string() {
        let input = "INSERT INTO `t` VALUES ('a\\nb');\n";
        assert_eq!(parse(input), vec![vec![s(b"a\nb")]]);
    }
}

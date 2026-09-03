// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// json.rs -- dependency-free JSON reader/writer for the uw-resolve Rust shim.
// Deliberately minimal: enough to round-trip the pkg_resolver corpus and
// differential JSONL. No resolver logic lives here.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> &[J] {
        match self {
            J::Arr(items) => items,
            _ => &[],
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => "",
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            J::Int(n) => *n,
            J::Float(f) => *f as i64,
            _ => 0,
        }
    }

    pub fn obj(fields: Vec<(&str, J)>) -> J {
        J::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn s(text: &str) -> J {
        J::Str(text.to_string())
    }
}

// --- writer ---------------------------------------------------------------

pub fn write_json(v: &J, out: &mut String) {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(true) => out.push_str("true"),
        J::Bool(false) => out.push_str("false"),
        J::Int(n) => {
            let _ = write!(out, "{}", n);
        }
        J::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                let _ = write!(out, "{}", *f as i64);
            } else {
                let _ = write!(out, "{}", f);
            }
        }
        J::Str(s) => write_string(s, out),
        J::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        J::Obj(fields) => {
            out.push('{');
            for (i, (k, val)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_json(val, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn to_string(v: &J) -> String {
    let mut s = String::new();
    write_json(v, &mut s);
    s
}

// --- parser ---------------------------------------------------------------

pub struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    _src: &'a str,
}

pub fn parse(src: &str) -> Result<J, String> {
    let mut p = Parser {
        chars: src.chars().collect(),
        pos: 0,
        _src: src,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format!("trailing input at {}", p.pos));
    }
    Ok(v)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n') | Some('\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at {}", c, self.pos))
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        for c in word.chars() {
            if self.bump() != Some(c) {
                return Err(format!("bad literal {} at {}", word, self.pos));
            }
        }
        Ok(())
    }

    fn value(&mut self) -> Result<J, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(J::Str(self.string()?)),
            Some('t') => {
                self.literal("true")?;
                Ok(J::Bool(true))
            }
            Some('f') => {
                self.literal("false")?;
                Ok(J::Bool(false))
            }
            Some('n') => {
                self.literal("null")?;
                Ok(J::Null)
            }
            Some(_) => self.number(),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<J, String> {
        self.expect('{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(J::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(':')?;
            let val = self.value()?;
            fields.push((key, val));
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                other => return Err(format!("bad object separator {:?} at {}", other, self.pos)),
            }
        }
        Ok(J::Obj(fields))
    }

    fn array(&mut self) -> Result<J, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(J::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                other => return Err(format!("bad array separator {:?} at {}", other, self.pos)),
            }
        }
        Ok(J::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.bump() {
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let cp = self.hex4()?;
                        if (0xD800..0xDC00).contains(&cp) {
                            // surrogate pair
                            self.expect('\\')?;
                            self.expect('u')?;
                            let lo = self.hex4()?;
                            let combined =
                                0x10000 + ((cp - 0xD800) << 10) + (lo.saturating_sub(0xDC00));
                            match char::from_u32(combined) {
                                Some(c) => out.push(c),
                                None => return Err("bad surrogate pair".to_string()),
                            }
                        } else {
                            match char::from_u32(cp) {
                                Some(c) => out.push(c),
                                None => return Err("bad \\u escape".to_string()),
                            }
                        }
                    }
                    other => return Err(format!("bad escape {:?}", other)),
                },
                Some(c) => out.push(c),
                None => return Err("unterminated string".to_string()),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.bump().ok_or("short \\u escape")?;
            let d = c.to_digit(16).ok_or("bad hex digit")?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<J, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => self.pos += 1,
                '.' | 'e' | 'E' | '+' | '-' => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if text.is_empty() {
            return Err(format!("bad number at {}", start));
        }
        if is_float {
            text.parse::<f64>().map(J::Float).map_err(|e| e.to_string())
        } else {
            text.parse::<i64>().map(J::Int).map_err(|e| e.to_string())
        }
    }
}

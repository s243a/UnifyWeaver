# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# parse_inserts.awk — AWK implementation of the mysql_stream leaf.
#
# Reads an un-gzipped MySQL dump on stdin. Emits every INSERT row as
# TAB-separated TSV on stdout. Matches the contract of the Rust
# implementation at src/unifyweaver/runtime/rust/mysql_stream/.
#
# Usage:
#   zcat dump.sql.gz | gawk -f parse_inserts.awk
#
# Notes:
#   * gawk required (not mawk) for array parameters and function
#     definitions used below.
#   * Assumes INSERT INTO statements are entirely on one line, which
#     matches MediaWiki's mysqldump output. Multi-line INSERT (rare)
#     is not handled.
#   * Binary bytes in VARBINARY fields pass through verbatim; they
#     will show up as whatever locale interpretation gawk makes of
#     them. For ASCII-safe columns (cl_type, integer IDs) this is
#     fine. Use -b flag for byte mode on binary-heavy dumps:
#       gawk -b -f parse_inserts.awk

/^INSERT INTO/ {
    # Extract the VALUES ... ; portion.
    v_idx = index($0, "VALUES ")
    if (v_idx == 0) next
    rest = substr($0, v_idx + 7)
    process_values(rest)
}

function process_values(s,    i, n, c, j, depth, in_quote, escape, tuple) {
    n = length(s)
    i = 1
    while (i <= n) {
        c = substr(s, i, 1)
        if (c != "(") { i++; continue }

        # Walk until matching ')' respecting quotes and backslash-escapes.
        j = i + 1
        in_quote = 0
        escape = 0
        while (j <= n) {
            if (escape) { escape = 0; j++; continue }
            ch = substr(s, j, 1)
            if (ch == "\\" && in_quote) { escape = 1; j++; continue }
            if (ch == "'") { in_quote = !in_quote; j++; continue }
            if (ch == ")" && !in_quote) break
            j++
        }

        tuple = substr(s, i + 1, j - i - 1)
        emit_tuple(tuple)
        i = j + 1
    }
}

# emit_tuple: walks a (...) tuple body and writes each field separated
# by TABs to stdout, terminated with NEWLINE.
function emit_tuple(s,    i, n, c, in_quote, escape, field, first) {
    n = length(s)
    i = 1
    first = 1
    field = ""
    in_quote = 0
    escape = 0
    while (i <= n) {
        c = substr(s, i, 1)
        if (escape) {
            escape = 0
            # MySQL escape → TSV escape.
            # Everything stays \-escaped so the Python consumer's
            # tsv_unescape() produces identical bytes as from the Rust
            # producer.
            if      (c == "n")  field = field "\\n"
            else if (c == "t")  field = field "\\t"
            else if (c == "r")  field = field "\\r"
            else if (c == "\\") field = field "\\\\"
            else if (c == "'")  field = field "\x27"
            else if (c == "\"") field = field "\x22"
            else if (c == "0")  field = field "\x00"
            else if (c == "Z")  field = field "\x1a"
            else if (c == "b")  field = field "\x08"
            else                field = field c
            i++
            continue
        }
        if (c == "\\" && in_quote) { escape = 1; i++; continue }
        if (c == "'") { in_quote = !in_quote; i++; continue }
        if (c == "," && !in_quote) {
            emit_field(field, first); first = 0; field = ""; i++; continue
        }
        # Outside a quote, skip whitespace (VALUES (1, 2) vs (1,2)).
        if (!in_quote && (c == " " || c == "\t")) { i++; continue }
        field = field c
        i++
    }
    emit_field(field, first)
    printf "\n"
}

function emit_field(f, first,    out, i, n, c, b) {
    # Match the Rust binary's TSV escaping: printable ASCII passes
    # through; \\ \t \n \r become two-byte escapes; non-printable
    # bytes become \xNN. NULL literals are \N.
    if (f == "NULL") { out = "\\N" }
    else {
        out = ""
        n = length(f)
        for (i = 1; i <= n; i++) {
            c = substr(f, i, 1)
            b = byte_of(c)
            if      (c == "\\") out = out "\\\\"
            else if (c == "\t") out = out "\\t"
            else if (c == "\n") out = out "\\n"
            else if (c == "\r") out = out "\\r"
            else if (b >= 0x20 && b <= 0x7E) out = out c
            else                 out = out sprintf("\\x%02x", b)
        }
    }
    if (!first) printf "\t"
    printf "%s", out
}

# byte_of: gawk portably returns the character code via ord-style trick.
# Uses a cached hex table.
function byte_of(c,    v) {
    if (!(c in BYTE_CACHE)) {
        v = index(BYTE_TABLE, c)
        BYTE_CACHE[c] = (v > 0 ? v - 1 : 0)
    }
    return BYTE_CACHE[c]
}

BEGIN {
    # BYTE_TABLE[i+1] = char with code i.  Used by byte_of().
    BYTE_TABLE = ""
    for (i = 0; i < 256; i++) BYTE_TABLE = BYTE_TABLE sprintf("%c", i)
}

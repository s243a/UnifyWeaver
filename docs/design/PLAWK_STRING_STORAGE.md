# plawk string storage: builders, ownership, and memory

Status: section 2 landed; sections 3-5 are the plan.

Related: `PLAWK_SCALAR_VALUE_MODEL.md` (string / strnum slots hold an atom id, `0` =
unset), `PLAWK_ARRAY_VALUE_MODEL.md` (str-valued tables, plan item 4),
`PLAWK_FIELD_BUFFER.md` (the field buffer's zero-intern edits).

## 1. The problem

Every runtime string value in plawk (a string or strnum scalar, a str-valued
array element) is an **interned atom id**. Interning copies the bytes into the
process-wide atom table, and the atom table never frees. So memory grows with the
**total bytes ever written**, not with the live data. gawk's grows with the live data.
Measured on gawk 5.1.0 (`LC_ALL=C`), maximum resident set:

| program | input | plawk | gawk |
|---|---|---|---|
| `{ last = $0 } END { print last }` | 3M records | 263 MB | 4 MB |
| `{ x = $2 } END { print x }` | 1M records | 73 MB | 4 MB |
| `{ a[$1] = $2 } END { ... }` (10 keys) | 1M records | 73 MB | 4 MB |
| `{ s = s $2 } END { print s }` (8-byte `$2`) | 8000 records | 373 MB | 4 MB |
| the same | 30000 records | 5.2 GB, 77 s | 4 MB, 0.02 s |

The overwrite cases are linear in the input (every distinct value written stays).
The self-append `s = s X` is quadratic: append i interns a new i-piece value.
`{ n++ }` and `{ c[$1]++ }` are flat (about 2 MB), and so is reading the record.
The line reader's transient atom (id 2^62, see §3) never interns.

## 2. String builders use a measured heap buffer (landed)

A concatenation or `sprintf` assigned to a scalar (`s = $1 "-" $2`,
`s = sprintf(...)`) and a row constructor (`a[$k] = row($a, $b)`) formatted into a
4096-byte `alloca` in the rule body. That had two consequences:

- **Truncation.** A value past 4095 bytes was cut short. `{ s = s $2 }` printed
  4096 bytes, not 8001, on 1000 records.
- **A crash.** An `alloca` outside the entry block is released only when the
  function returns. The record loop never returns, so the stack grew by 4 KiB per
  record and overflowed 8 MiB after about 2000 records (signal 11 on 4000).

The builders now share `plawk_snprintf_heap_intern_lines/4`:

1. a measuring `snprintf(NULL, 0, ...)`;
2. `malloc(need + 1)`;
3. the real `snprintf`;
4. `@wam_intern_atom`, which copies;
5. `free`.

The interned length is still `strlen`'s, so a value stops at an embedded NUL as
before. Element writes (arrays PR 5a) already used this pattern.

This fixes correctness only: the builders still intern, so the memory rows above
are unchanged.

## 3. Owned string handles (planned)

Keep the i64 representation and the single read path. Add a second kind of
id: an **owned string handle**.

- **Encoding.** A handle is `2^61 | ptr`, where `ptr` points to a heap header
  `{ i64 len, i64 cap, bytes[cap + 1] }`. User-space pointers are below 2^47, so
  the tag never collides with an atom id (interned ids are small) or the transient
  line id (2^62).
- **Reads are unchanged.** `@wam_atom_to_string` gains one tag test and returns
  `ptr + 16` for a handle. That is the precedent the transient line id set.
  Every existing reader goes through that call: prints, `@wam_strnum_cmp`,
  `length`, field splitting of a string, `sub`/`gsub` sources and the for-in value
  guards (about 43 emitter sites). They need no change. A NUL-terminated buffer
  keeps `%s` correct.
- **Ownership.** A handle is owned by exactly one slot or element; it is never
  shared. That rules out reference counts:
  - **overwrite** reuses the old handle's buffer when it is large enough, and
    otherwise `realloc`s it. A steady-state `x = $2` allocates nothing, which is
    cheaper than today's per-record hash and intern;
  - **copy** (`m = s`, `a[k] = s`) duplicates a handle and passes an atom id
    through unchanged;
  - **delete** (`delete a[k]`, `delete a`, a `split` target's clear, a str
    table's teardown) frees it.
- **Appends.** `s = s X` and `a[k] = a[k] X` are recognised (the target read as the
  first part) and lowered to `@wam_sv_append`. It reallocs with capacity
  doubling, so n appends cost O(total bytes), as in gawk. Any other form that
  reads the target (`s = X s`) builds into a fresh buffer, then replaces the old
  one: a source must never alias the buffer being written.
- **Constants stay atoms.** A string literal, an `ENVIRON` / `ARGV` value and a
  for-in key are already interned and bounded. Assigning one stores its atom id,
  and a later overwrite of that slot frees nothing.

Runtime (in `strnum.ll.mustache`, beside the other value helpers):

| function | contract |
|---|---|
| `@wam_sv_set(i64 old, i8* p, i64 len) -> i64` | the slot's new value: `old`'s buffer reused when it is a handle (grown if needed), a fresh handle otherwise |
| `@wam_sv_reserve(i64 old, i64 len) -> i64` | the same, without the copy: a builder `snprintf`s straight into the buffer, then sets `len` |
| `@wam_sv_append(i64 old, i8* p, i64 len) -> i64` | append in place; an atom or `0` `old` starts a handle with its text |
| `@wam_sv_dup(i64 v) -> i64` | copy a handle; an atom id is returned as is |
| `@wam_sv_free(i64 v)` | free a handle; an atom id or `0` is a no-op |
| `@wam_sv_key(i64 v) -> i64` | an interned atom id for use as an array KEY (a handle's text interned; an atom unchanged) |

### 3.1 What the change must audit

Every consumer of a string id that does not go through `@wam_atom_to_string`:

- **id equality.** `icmp eq` on two string ids is equality only for interned
  atoms. No current emitter compares string values by id (comparisons go through
  `@wam_strnum_cmp` on the text), and none may start.
- **keys.** A string value used as an array key (`b[s]`, `b[a[k]]`, `k in b`) must
  go through `@wam_sv_key`. Table keys stay interned, so `in` and lookups remain id
  compares.
- **crossing into the WAM.** A string passed to a Prolog goal, a foreign call or a
  cache write needs an atom: `@wam_sv_key` again.
- **ownership moves at control-flow merges.** A slot's value is carried by phi
  nodes across branches and the record loop. A phi is a move, not a copy. The
  emitter must free the value being replaced at the write, never at a merge.

## 4. Plan

1. **S1 -- heap builders (§2): landed.**
2. **S2 -- str-valued array elements.** Element writes (`a[k] = $N`,
   `= "lit"`, `= PARTS`, row capture `= $0`, `row(...)`) store handles through
   `@wam_sv_set`. Deletes free. The self-append `a[k] = a[k] X` lands on
   `@wam_sv_append`: this is arrays plan item 4(b). The measured patch is kept
   out of tree until then.
3. **S3 -- string and strnum scalars.** `x = $N` uses `@wam_sv_set`; a builder
   uses `@wam_sv_reserve` plus a direct `snprintf`; `s = s X` uses
   `@wam_sv_append`. Copies use `@wam_sv_dup`. Also `sub` / `gsub` targets.
4. **S4 -- split pieces.** `split` interns each piece today, so `split($0, p)` per
   record grows with the distinct pieces. The split target becomes a handle table.

Each step is verified against gawk, on the IR corpus (the changed programs are
exactly the touched class), and by a peak-RSS check in its tests: a bounded
program must stay under a fixed ceiling on a large input.

## 5. Known limits that remain

- **Table keys are interned and never freed.** A sliding window
  (`a[NR] = $0; delete a[NR - 10]`) still grows by one key atom per record, and so
  does any program that probes ever-new absent keys. Fixing that needs a
  reclaimable key space, such as integer keys that skip interning; it is out of
  scope here.
- **`dyncall_at` with a field path** copies the path into a 4096-byte `alloca` in
  the rule body. That is the same stack-growth shape as §2, but on a foreign-call
  path that is rarely per-record. Not yet fixed.

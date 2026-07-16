<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk field buffer: representation and scaling

**Status**: the flat-array representation described in §1 is implemented and
shipped (field assignment, `$N = expr`). §3 is a **forward-looking design
note** — the alternative representations are *not* implemented; this captures
when they would be worth building and how they would slot in behind the same
API.

## 1. Current representation — a flat slice array

awk's `$1..$NF` are modelled at runtime by `%WamFieldBuf`
(`templates/targets/llvm_wam/types.ll.mustache`):

```llvm
%WamFieldBuf = type { i64, i64, %WamSlice* }  ; count, cap, slices
```

Field *i* (1-based) lives in slot *i−1* as a `%WamSlice` = `{ i8* ptr, i64 len }`.
The primitives (`src/unifyweaver/targets/wam_llvm_target.pl`):

- `@wam_fields_new(rec, sep)` — split the record on the single-byte separator
  into one slice per field. Slices **point into the record text**; nothing is
  interned or copied. Initial capacity is `max(NF, 4)`.
- `@wam_fields_get(idx)` — bounds-checked read; out of range → `{null, 0}`
  (an empty field).
- `@wam_fields_set(idx, ptr, len)` — set a slot in place, growing the array
  (realloc) and padding intervening slots with empties when `idx` is past the
  current count.
- `@wam_fields_join(ofs)` — join fields `1..count` with OFS into one fresh
  NUL-terminated buffer (the rebuilt `$0`); the caller prints and frees it.
- `@wam_fields_free`.

### Cost

| operation | cost |
|---|---|
| split (`new`) | O(record length) |
| `get` / `set` on a dense field | O(1) amortized |
| `set` at index `k` past the end | O(k) — reallocation + padding the gap |
| `join` | O(Σ field lengths + count) |
| memory | O(highest index touched) — one `%WamSlice` per slot, gaps included |

This is the right default. Typical awk input is FS-delimited lines with a
**small, dense column count** (a handful to a few dozen fields), and for that a
flat array is strictly best: O(1) indexing, contiguous and cache-friendly,
no per-field allocation, and a trivial in-order `join`. The refactor that
introduced it (split once / mutate in place / join once) already removed the
per-assignment re-intern cost, so editing a record is O(record) with zero
interning.

## 2. Where the flat array degrades

The flat array is indexed *by position*, so its footprint is the **highest
index touched**, not the number of fields actually present. Two shapes stress
it:

- **Sparse high-index assignment.** `$1000000 = "x"` on a 3-field record grows
  the slice array to a million slots, almost all empty. This is
  awk-faithful (assigning a high field number does extend `NF` and create the
  intervening empty fields), but the memory is proportional to the index, not
  the data.
- **Very wide records.** Columnar / wide-table input (thousands of columns per
  row) makes every `new`/`join` walk a large dense array each record. Still
  linear, but the constant and the working set grow with width.

Neither shape appears in ordinary text-processing awk, which is why the flat
array is the shipped default. They *do* appear if plawk is pointed at
wide/columnar data structures — the case worth planning for, not building yet.

## 3. Alternative representations (not implemented)

Keep `%WamFieldBuf` an **abstract handle**: `new` / `get` / `set` / `join` /
`free` are the whole contract, and the field-assignment codegen only ever calls
those. That lets the backing representation change without touching codegen or
the driver. Candidates, by scale:

| field-space size / density | structure | rationale |
|---|---|---|
| small, dense (the norm) | **flat slice array** (current) | best constants, contiguous, O(1) index, trivial ordered join |
| medium / sparse | **ordered map** (sorted `(index, slice)` — balanced tree or sorted vector) | stores only present fields; iterates in index order for `join`; a sparse high index costs O(present), not O(index). O(log n) access |
| very large / highly sparse | **hashtable** (open-addressing `i64 → slice`, à la `%WamAssocI64Table`) | O(1) average access at scale; needs a max-index tracker and an ordered walk (sorted keys, or min/max range scan) so `join` stays deterministic |

Notes:

- **Gap semantics carry over.** With a map or hashtable, `join` must still
  synthesize empty fields for indices between 1 and the max present/assigned
  index (awk semantics), so the max-index must be tracked explicitly rather
  than read off `count`.
- **Reuse.** The open-addressing `%WamAssocI64Table` that already backs `split`
  and associative arrays is a ready hashtable; a field map could store slice
  indices or interned ids in it. (Interning would reintroduce per-field intern
  cost, so a slice-valued variant is preferable — the win of the flat array was
  *not* interning.)
- **Adaptive promotion.** Rather than a compile-time choice, the buffer could
  start flat and promote to a map/hashtable when the count or the
  sparsity ratio (highest-index / present-fields) crosses a threshold. The
  thresholds above are illustrative, not tuned — they should be measured before
  anyone commits to them.

## 4. Why this is a note, not a task

YAGNI. The flat array is optimal for the field counts real awk programs see,
and adding a second representation is only justified once wide/sparse field
workloads are a real target. This note exists so that, when that day comes, the
API boundary is already the right one and the trade-offs are written down.

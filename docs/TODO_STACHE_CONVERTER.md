# `.mustache` → `.stache` Converter TODO

## Overview

A one-shot tool that converts a legacy `.mustache` template into the proposed `pattern_stache`
dialect: adds the dialect header and quotes any `{{case}}` value that would not read as the
intended atom.

**It has not been built, and that is the current recommendation rather than an oversight.** This
document records why, and the condition under which the answer changes.

Design context: [`docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`](design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md).

## Why it has not been built

**Starting fresh is the better default.** Write a new `.stache` file and leave the `.mustache`
one in place. The old file keeps working for anyone who wants the old behaviour, the new patterns
are isolated, and the extension alone tells a reader which files use them. Conversion is only
attractive when an existing file has enough cases that retyping them is tedious and error-prone.

**No file in this repository is that large.** Counting `{{case}}` blocks per file:

| cases | file |
|---:|---|
| 32 | `src/unifyweaver/core/template_system.pl` — its own self-tests and documentation, not a template |
| **5** | `templates/targets/fsharp_wam/program.fs.mustache` — the largest real library |
| 2 | each of the other twelve `{{match}}`-using templates |

Five cases is not a conversion burden. Hand-writing the new file is faster than running,
reviewing, and trusting a tool.

**The alternative shape is worse.** A load-time shim that auto-quotes `.mustache` files on read
was considered and rejected: its only benefit was collapsing to a single parser, but of 307
`.mustache` files across 44 target directories, only 13 use `{{match}}` at all. The other 294
dispatch on nothing and will never migrate, so the string parser survives regardless. A shim
would be permanent maintenance buying nothing, and it would apply a semantic edit invisibly at
every load rather than as a reviewable diff.

## When to build it

Check the condition, do not wait for a date:

> A template used as a **library** grows enough cases that hand-copying them into a new file is
> error-prone.

Practically: a `{{match}}`-using file in the low tens of cases. Re-run the count above to decide.

Secondary trigger: if many `.mustache` files ever need converting at once — a bulk migration
rather than a file at a time.

## What it would have to do

Quote any case value that would not read as the atom the author meant. Verified against SWI's
reader:

| shape | example | action | how it fails unquoted |
|---|---|---|---|
| lowercase atom | `helpers` | none | — reads as the same atom |
| hyphenated | `wam-fsharp` | quote | **silently** — reads as compound `-/2` |
| other operator chars | `3-way` | quote | **silently** — reads as compound `-/2` |
| uppercase-initial | `Helpers` | quote | **silently and totally** — becomes a variable matching every input |
| contains a space | `a b` | quote | loudly — the read raises |

Plus inserting the dialect header, `{{! dialect(pattern_stache, 1) }}`, as the first non-empty
line.

## The limit that makes it a migration aid, not a compatibility layer

The rule "quote anything that is not already a lowercase atom" is unambiguous **only for a file
in which every case value is a literal** — which is exactly what a legacy `.mustache` file is.

It cannot be applied inside a `.stache` file. Faced with `{{case substrate(C)}}` the tool has no
way to distinguish a pattern it must leave alone from a literal tag it must quote; both are
merely "not a lowercase atom." So the converter is strictly one-directional and one-shot.

## Related hazard, if a converted copy is kept alongside the original

Two live files sharing most of their cases will drift. The clean fix is delegation — the new
file's `{{default}}` falling through to the legacy template — but that is **not expressible
today**: the supported mustache subset has no partials (`{{> name}}`), and `compose_templates/3`
concatenates rendered output rather than delegating dispatch. Delegation would be new machinery
and belongs in the same deferred bucket.

## Status

| | |
|---|---|
| Priority | Low |
| Effort | Low once needed |
| Blocked by | Nothing — it is unnecessary, not blocked |
| Prerequisite | The `pattern_stache` dialect existing at all, which is itself only proposed |

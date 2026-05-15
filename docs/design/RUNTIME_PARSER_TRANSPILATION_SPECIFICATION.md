# Runtime Parser Transpilation Specification

This document defines the target-runtime contract for parsing Prolog terms from
text. It is separate from the build-time WAM items parser used by target
generators.

## Canonical Source

The portable parser source is:

```text
src/unifyweaver/core/prolog_term_parser.pl
```

Its public predicates are:

```prolog
parse_term_from_codes(+Codes, +OpTable, -Term).
parse_term_from_atom(+Atom, +OpTable, -Term).
canonical_op_table(-OpTable).
```

`OpTable` is a list of `op(Name, Prec, Type)` terms, where `Type` is one of
`xfx`, `xfy`, `yfx`, `fx`, `fy`, `xf`, or `yf`.

## Runtime Contract

A target that enables runtime term parsing must provide a parser entry point
that accepts text or character codes, an operator table snapshot, and returns a
target-native Prolog term. Invalid input fails unless the calling predicate
explicitly maps it to an error.

The returned term must use the target runtime's ordinary term representation:
atoms, integers, floats, compounds, lists, and logic variables must be values
the rest of the WAM runtime can unify with. Variable names repeated in one parse
must share one logic variable. `_` must produce a fresh anonymous variable for
each occurrence.

The parser must not own mutable operator state. Runtime predicates such as
`op/3` and `current_op/3` belong to the target runtime. Before a parse, the
runtime passes the parser a snapshot derived from its current table.

## Required Consumers

Targets that expose the corresponding predicates should route these operations
through the runtime parser:

- `read_term_from_atom/2`
- `read_term_from_atom/3`
- `term_to_atom/2` in reverse mode
- `read/2`

Targets may also use the same parser for CLI argument decoding, REPL input, or
runtime data-file ingestion.

## Stream Read Semantics

`read/2` must read from a target stream until it can parse one complete dotted
term. The R target is the current reference behavior:

- accumulate source text across lines;
- only attempt completion when the trimmed buffer ends with `.`;
- parse the buffer without the trailing `.`;
- if parsing fails, continue reading because the dot may be inside a quoted
  atom, string, or incomplete compound;
- EOF on an empty buffer binds `end_of_file`;
- EOF on a non-empty buffer makes one final parse attempt.

The parser itself does not own stream state. It only answers whether the current
source buffer is a valid term.

## Operator Semantics

The initial table should come from `canonical_op_table/1` or an equivalent
target seed table. Runtime `op/3` declarations mutate the target's table, and
later parses must observe the new table.

Compile-time target options may seed additional operators before user input is
parsed. In WAM-R this role is served by `r_op_decls(...)`, which emits runtime
operator table initialization before CLI parsing.

The current parser intentionally matches the R Pratt-style parser. It supports
standard prefix, infix, and postfix operators, but treats `fx`/`fy` and
`xf`/`yf` the same at chained prefix/postfix sites. This compatibility note
should remain visible until stricter ISO distinctions are implemented.

## Target Status

R is the reference target because it already has runtime consumers:
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI parsing.
The R target currently has an inline parser in `runtime.R.mustache`, while
`prolog_term_parser.pl` compiles structurally to WAM-R.

The structural guard is:

```text
tests/test_prolog_term_parser_wam_r_compile.pl
```

Full replacement of the inline R parser is blocked by WAM-R cut-barrier
semantics. The parser source relies on cuts in multi-clause helper predicates;
the WAM-R runtime currently drops only the most recent choice point in cases
where a proper cut barrier is needed. Once that runtime issue is fixed, the
compiled parser should be compared against the inline parser and then used as
the canonical R runtime parser.

## Test Requirements

Runtime parser changes should be covered at three levels:

- SWI tests for `prolog_term_parser.pl` itself;
- target compile tests proving the parser source lowers into the target;
- target runtime tests through real predicates such as `read/2` and
  `read_term_from_atom/2,3`.

For R, existing runtime coverage lives in `tests/test_wam_r_generator.pl`, and
parser performance/equivalence work is tracked by
`tests/benchmarks/wam_r_parser_bench.pl`.

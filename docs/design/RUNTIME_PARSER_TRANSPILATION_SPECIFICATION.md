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

## Target Capability Hook

Each WAM target exposes its runtime parser posture through the shared
generator-side hook in `src/unifyweaver/targets/wam_runtime_parser_capability.pl`:

```prolog
%% wam_target_runtime_parser(+Target, +Options, -Mode)
%
%  Mode is one of:
%    none
%      The generated runtime has no source-term parser.
%    native(Entry)
%      The target has a hand-written parser entry point named by Entry.
%    compiled(SourceModule)
%      The target should include the portable Prolog parser module and call
%      the generated wrapper predicates.
%
%  Options may override the target default:
%    runtime_parser(off)       -> none
%    runtime_parser(native)    -> native(...) if the target supports one
%    runtime_parser(compiled)  -> compiled(prolog_term_parser)
%    runtime_parser(auto)      -> target default
```

The hook is a generator contract, not a runtime predicate. It lets project
writers decide whether to include parser library predicates, whether parser
builtins should dispatch to a native parser, and whether a target should reject
parser-dependent builtins at generation time.

The same module also defines `parser_dependent_builtin/1`, the catalogue of
builtins that require runtime source-term parsing.

Target defaults should be conservative:

- targets with a fast existing runtime parser, such as R, default to
  `native(...)`;
- targets without a parser default to `none`;
- targets may opt into `compiled(prolog_term_parser)` only after they can run
  the portable parser's compile and runtime tests.

The generated runtime should expose only one builtin-facing parser entry point
per target. A target may keep both native and compiled parsers for testing, but
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI parsing
should route through the selected mode consistently.

## Required Consumers And Inclusion

Targets that expose the corresponding predicates should route these operations
through the runtime parser:

- `read_term_from_atom/2`
- `read_term_from_atom/3`
- `term_to_atom/2` in reverse mode
- `read/2`

Targets may also use the same parser for CLI argument decoding, REPL input, or
runtime data-file ingestion.

When `wam_target_runtime_parser/3` returns `compiled(prolog_term_parser)`, the
project writer must add `src/unifyweaver/core/prolog_term_parser.pl` predicates
to the generated project before compiling parser-dependent drivers. Inclusion
should be demand-driven by default: if no emitted builtin or target feature
needs runtime source-term parsing, the parser library should not be included.

When the selected mode is `none`, parser-dependent builtins must either be
omitted from the target's advertised capability set or fail with a clear
generation-time diagnostic. Silent runtime stubs are not acceptable for new
targets because parser failure is otherwise hard to distinguish from ordinary
predicate failure.

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
`prolog_term_parser.pl` compiles to WAM-R and runs representative parser
drivers end to end.

The compile and runtime guard is:

```text
tests/test_prolog_term_parser_wam_r_compile.pl
```

Full replacement of the inline R parser is not the current R target direction.
The compiled parser is semantically useful, but R benchmark data shows it is
far slower than the inline parser for ordinary atom, compound, list, and
operator-expression parsing. R should keep the inline parser as its hot path
unless future runtime changes close that gap. The portable parser remains the
canonical cross-target specification and the preferred starting point for
targets that do not already have a native runtime parser.

## Test Requirements

Runtime parser changes should be covered at three levels:

- SWI tests for `prolog_term_parser.pl` itself;
- target compile tests proving the parser source lowers into the target;
- target runtime tests through real predicates such as `read/2` and
  `read_term_from_atom/2,3`.

For R, existing runtime coverage lives in `tests/test_wam_r_generator.pl`, and
parser performance/equivalence work is tracked by
`tests/benchmarks/wam_r_parser_bench.pl`.

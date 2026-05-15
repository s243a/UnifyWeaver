# Runtime Parser Transpilation Philosophy

UnifyWeaver has two parser problems that should stay separate:

- build-time WAM item parsing, where the generator currently may emit WAM
  text and parse it back into structured instruction items;
- runtime Prolog term parsing, where a generated target program must turn
  source text back into Prolog terms while it is running.

PR #2086 addresses the first problem. Runtime parser transpilation addresses
the second one.

The R target already demonstrates why the runtime parser matters. Its
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI argument
parsing all need an operator-aware Prolog term parser in the generated runtime.
`read/2` is the clearest example: a stream reader cannot just read one line or
split on a period. It must keep accumulating text until the buffer ends with a
dotted term and the parser accepts the term before the trailing dot. That is a
runtime behavior, not a generator-time convenience.

## Direction

The canonical runtime parser should live in portable Prolog and be compiled
into target runtimes when a target needs source-term input. The current source
for that role is:

```text
src/unifyweaver/core/prolog_term_parser.pl
```

Keeping the parser in portable Prolog gives us one parser specification, one
test surface, and one place to harden operator semantics. Targets may keep a
temporary native parser while the transpiled parser matures, but target-local
parsers should be treated as bridges rather than as independent semantic
sources.

## Why Transpile The Parser

Runtime parsing needs target-local term values, variables, lists, compounds,
and operator tables. Reimplementing that logic per target creates drift:
variable sharing, anonymous variables, prefix and infix precedence, runtime
`op/3`, and stream-read completion rules all become subtly target-specific.

A transpiled parser keeps these semantics under the same compiler and runtime
contract as ordinary user code. It also tests an important ability: generated
targets should be able to host non-trivial Prolog library code, not just user
predicates selected by a benchmark.

## Design Constraints

The parser should remain pure with respect to operator state. Targets own their
runtime operator table, including mutations from `op/3`; the parser receives a
snapshot of that table as data.

The parser must preserve source variable identity within one parse. Repeated
variable names share one logic variable; `_` remains anonymous.

The parser should fail cleanly on invalid input. Higher-level predicates decide
whether failure becomes predicate failure, EOF handling, or a target-specific
error.

The parser should stay in the cross-target-compilable subset. If the parser
requires stronger WAM behavior, such as correct cut-barrier semantics, that is
a runtime correctness requirement rather than a reason to fork the parser.

## Non-Goals

This work is not the shared WAM items API. It does not replace the Prolog-side
WAM text parser used by target generators.

This work is not a full source-program parser. The first goal is term parsing
for runtime predicates such as `read/2` and `read_term_from_atom/2,3`, not full
module loading, comments, source layout preservation, or complete
`read_term/3` option handling.

This work does not require every target to adopt the transpiled parser at once.
R is the reference consumer because it already needed the runtime behavior.

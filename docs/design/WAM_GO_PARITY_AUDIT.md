# WAM Go Parity Audit

This note compares the Go hybrid WAM target against the Rust and Haskell WAM
targets, using the Lua and Python parity audits as secondary references for the
current cross-target builtin/runtime baseline.

## Verified Runtime Surface

| Area | Go support | Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Choice points and backtracking | `try_me_else`, `retry_me_else`, `trust_me`, indexed alternatives, foreign stream retries, indexed atom fact streams, `member/2` builtin retries | Choice point stack with normal, builtin, and fact-stream resume states | Present for current resume-state baseline |
| Direct fact dispatch | `call_indexed_atom_fact2`, `AtomFact2Source` registry, TSV-backed and LMDB-artifact atom fact loading, foreign/native kernel registration, indexed atom/weighted fact tables | `call_indexed_atom_fact2`, inline/external fact stream paths | Present for current external atom fact-source baseline |
| Aggregates | `begin_aggregate`, `end_aggregate`; `collect`, `bag`, `bagof`, `count`, `sum`, `min`, `max`, `set`, `setof` | `findall/3`, `aggregate_all/3` count/sum/min/max/set families | Present for current aggregate baseline |
| Structural builtins | `member/2`, `memberchk/2`, `select/3`, `delete/3`, `permutation/2`, `length/2`, `append/3`, `reverse/2`, `last/2`, `nth0/3`, `nth1/3`, `numlist/3`, `sum_list/2`, `min_list/2`, `max_list/2`, `sort/2`, `msort/2` | `member/2`, `memberchk/2`, `length/2`; Rust `append/3` is explicitly unimplemented. Clojure/R/C++ also cover richer list builtins including `select/3`, `delete/3`, `permutation/2`, `reverse/2`, `last/2`, `nth0/3`, `nth1/3`, `numlist/3`, numeric list reducers, `sort/2`, and `msort/2` | Present for current baseline structural checks, with expanded cross-target list builtins |
| Type builtins | `var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `atomic/1`, `is_list/1`, `ground/1` | Includes `is_list/1` and Clojure direct `ground/1` in the current baseline | Present for current baseline type and groundness checks |
| Arithmetic builtins | `is/2`, `succ/2`, `between/3` | Arithmetic evaluation plus sibling-target `succ/2` coverage in Clojure and Elixir; R/C++ also cover `between/3` | Present for current baseline arithmetic checks, with expanded successor and integer-range coverage |
| Atom/text conversion | `atom_number/2`, `atom_codes/2`, `atom_chars/2`, `string_codes/2`, `string_chars/2`, `number_codes/2`, `number_chars/2`, `atom_string/2`, `string_to_atom/2`, `upcase_atom/2`, `downcase_atom/2`, `atom_concat/3`, `atom_length/2`, `string_length/2`, `char_code/2`, `sub_atom/5` forward mode, `char_type/2` forward mode, `string_code/3` forward mode, `split_string/4` forward mode | Clojure direct builtin surface includes bidirectional `atom_number/2`, atom/string/number code-list and char-list conversion, atom/string conversion, atom case conversion, deterministic atom concatenation, text length checks, and char-code conversion | Present for current atom-number, atom/string/number code-list, atom/string/number char-list, atom/string, atom-case, atom-concat, text-length, char-code, forward sub-atom, forward char-type, forward string-code, and forward split-string checks |
| Comparison builtins | `==/2`, `\==/2`, `\=/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2`, `@</2`, `@=</2`, `@>/2`, `@>=/2`, `compare/3` | Includes `=</2`; Haskell mode analysis and Clojure lowering also cover term-order comparisons | Present for current baseline comparisons, with expanded term-order coverage |
| Unification builtin | `=/2`, `\=/2` | `=/2`, `\=/2` | Present |
| Term inspection | `functor/3`, `arg/3` | `functor/3`, `arg/3` | Present |
| Univ | `=../2` compose/decompose | `=../2` compose/decompose | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | `copy_term/2` with fresh variables and preserved sharing | Present |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Same baseline, with broader isolated-goal NAF in Haskell/Python | Present for current baseline, including isolated user-goal NAF and race-to-true over multi-clause WAM targets |
| IO | `write/1`, `display/1`, `nl/0`, `tab/1` | `write/1`, `display/1`, `nl/0`; R/C++ also cover `tab/1` | Present for current baseline plus tab output |

## Immediate Findings

- `tests/test_wam_go_generator.pl` had stale expectations for atom literals.
  The Go target now emits `internAtom("...")` instead of raw
  `&Atom{Name: "..."}` literals, so the assertions needed to follow the
  current intern-table design.
- The Go WAM runtime has a substantial execution core; `set` aggregate results
  are now deduplicated like Haskell's `nub` behavior.
- `member/2` now pushes builtin choice points for later list members, so
  `findall/3` can collect every unifiable element.
- Deterministic `memberchk/2` is now covered by the generated Go WAM builtin
  E2E test, committing to the first unifiable list element without adding
  builtin choice points.
- Nondeterministic `select/3` is now covered by the generated Go WAM builtin
  E2E test for first-match selection, missing/empty/malformed-list failure,
  and `findall/3` enumeration of every selected element/rest pair, matching
  the Clojure/C++ structural-list surface.
- Deterministic `delete/3` is now covered by the generated Go WAM builtin E2E
  test for removing one, none, or all matching atom elements and malformed-list
  failure, matching the Clojure/C++ structural-list surface for proper lists.
- `permutation/2` is now covered by the generated Go WAM builtin E2E test for
  deterministic `(+,+)` permutation checks and the R-compatible `(+,-)`
  identity mode. Full nondeterministic permutation enumeration remains outside
  this scaffold.
- Deterministic `reverse/2` is now covered by the generated Go WAM builtin E2E
  test for both forward and reverse-list binding modes, closing one richer
  Clojure/R/C++ list-builtin parity gap.
- Deterministic `last/2` is now covered by the generated Go WAM builtin E2E
  test for non-empty list success and empty-list failure, closing another
  richer Clojure/R/C++ list-builtin parity gap.
- Deterministic `nth0/3` and `nth1/3` are now covered by the generated Go WAM
  builtin E2E test for in-range element lookup and out-of-range failure,
  matching the Clojure/R deterministic indexed-list surface.
- Deterministic `numlist/3` is now covered by the generated Go WAM builtin E2E
  test for closed integer range construction and `Low > High` failure,
  matching the Clojure/R range-list surface.
- `between/3` is now covered by the generated Go WAM builtin E2E test for
  deterministic `(+,+,+)` range checks and enumerable `(+,+,-)` integer
  generation through `findall/3`, matching the bounded R/C++ range surface.
- Numeric list reducers `sum_list/2`, `min_list/2`, and `max_list/2` are now
  covered by the generated Go WAM builtin E2E test for integer and mixed-float
  lists, empty-list behavior, and malformed/non-numeric list failure, matching
  the bounded C++/LLVM reducer surface.
- Deterministic `sort/2` and `msort/2` are now covered by the generated Go WAM
  builtin E2E test for standard ordered sorting, duplicate removal in
  `sort/2`, duplicate preservation in `msort/2`, mixed atom/integer ordering,
  and malformed-list failure, matching the Clojure/R ordered-list surface.
- Term-order comparisons `@</2`, `@=</2`, `@>/2`, `@>=/2`, and `compare/3`
  are now covered by the generated Go WAM builtin E2E test over the same
  atom/integer ordering used by Go WAM sort and indexed-switch ordering,
  matching the Clojure term-order lowering surface and the Haskell mode
  analysis baseline.
- `ground/1` is now covered by the generated Go WAM builtin E2E test for
  ground atoms, numbers, compounds, lists, empty lists, fresh variables,
  nested unbound compound arguments, and list elements with unbound variables,
  matching the current Clojure groundness surface.
- Deterministic forward-mode `sub_atom/5` is now covered by the generated
  Go WAM builtin E2E test for bound source, before, and length arguments,
  after/sub-atom unification, empty substring extraction, numeric source
  conversion, out-of-range failure, unbound source failure, unsupported
  enumerable modes, and mismatch failure, matching the R WAM forward-mode
  baseline while leaving full nondeterministic enumeration for a later slice.
- Forward-mode `char_type/2` is now covered by the generated Go WAM
  builtin E2E test for alpha, alnum, digit, space, white, upper, lower,
  punct, ascii, csym, csymf, newline, multi-character failure, unbound
  argument failure, and unknown category failure, matching the R/Clojure
  atom-category surface.
- Forward-mode `string_code/3` is now covered by the generated Go WAM
  builtin E2E test for 1-based code lookup, code unification, invalid index
  failure, unbound argument failure, non-text source failure, and bound-code
  mismatch failure, matching the R/C++ deterministic text-code lookup surface.
- Forward-mode `split_string/4` is now covered by the generated Go WAM
  builtin E2E test for separator splitting, empty input, adjacent separators,
  no-separator padding, separator plus pad trimming, multiple separators,
  numeric source conversion, unbound input failure, and bound-output mismatch
  failure, matching the R/C++ deterministic split-string surface.
- `tab/1` is now covered by the generated Go WAM builtin E2E test for
  nonnegative space output, zero-width success, negative integer failure,
  unbound argument failure, and non-integer failure, matching the R/C++
  basic I/O polish surface.
- Bidirectional `succ/2` is now covered by the generated Go WAM builtin E2E
  test for forward binding, reverse binding, matching integer pairs, mismatch
  failure, negative predecessor failure, non-positive successor failure, and
  both-unbound failure, matching the sibling Clojure/Elixir successor surface.
- Bidirectional `atom_number/2` is now covered by the generated Go WAM builtin
  E2E test for atom-to-integer, integer-to-atom, atom-to-float, float-to-atom,
  numeric first-argument compatibility, bad atom failure, and both-unbound
  failure, matching the Clojure atom-number surface.
- Deterministic `upcase_atom/2` and `downcase_atom/2` are now covered by the
  generated Go WAM builtin E2E test for converted output binding, bound-output
  success, mismatch failure, unbound-source failure, and non-atom source
  failure, matching the Clojure atom-case surface.
- Deterministic `atom_concat/3` is now covered by the generated Go WAM builtin
  E2E test for output binding, bound-output success, mismatch failure, unbound
  input failure, and non-atom input failure, matching the current Clojure
  atom-concat surface.
- Deterministic `atom_length/2` and `string_length/2` are now covered by the
  generated Go WAM builtin E2E test for length binding, bound-length success,
  mismatch failure, unbound-source failure, and non-atom source failure,
  matching the current Clojure text-length surface.
- Bidirectional `char_code/2` is now covered by the generated Go WAM builtin
  E2E test for char-to-code binding, bound-code success, mismatch failure,
  code-to-char binding, multi-character atom failure, both-unbound failure,
  and invalid-code failure, matching the current Clojure char-code surface.
- Bidirectional `atom_codes/2` is now covered by the generated Go WAM builtin
  E2E test for atom-to-code-list binding, bound-list success, mismatch failure,
  code-list-to-atom binding, empty-atom conversion, both-unbound failure, and
  invalid-code failure, matching the current Clojure atom/code-list conversion
  surface.
- Bidirectional `atom_chars/2` is now covered by the generated Go WAM builtin
  E2E test for atom-to-char-list binding, bound-list success, mismatch failure,
  char-list-to-atom binding, empty-atom conversion, both-unbound failure, and
  invalid-char failure, matching the current Clojure atom/char-list conversion
  surface.
- Bidirectional `string_codes/2` and `string_chars/2` are now covered by the
  generated Go WAM builtin E2E test for string-to-list binding, bound-list
  success, mismatch failure, list-to-string binding, empty-string conversion,
  both-unbound failure, and invalid-list failure, matching the current Clojure
  string list-conversion aliases.
- Bidirectional `number_codes/2` and `number_chars/2` are now covered by the
  generated Go WAM builtin E2E test for number-to-list binding, bound-list
  success, mismatch failure, list-to-number binding for integer, negative, and
  float text, both-unbound failure, and invalid-list failure, matching the
  current Clojure number list-conversion surface.
- Bidirectional `atom_string/2` and `string_to_atom/2` are now covered by the
  generated Go WAM builtin E2E test for atom-to-text binding, bound-text
  success, mismatch failure, text-to-atom binding, and both-unbound failure,
  matching the current Clojure atom/string conversion surface.
- Go WAM choice points now have explicit generated-runtime coverage for normal
  clause retries, indexed alternatives, foreign stream retries, indexed atom
  fact streams, and `member/2` builtin retries.
- `=</2`, `is_list/1`, and `display/1` are now covered by the generated Go
  WAM builtin E2E test.
- `=/2` and `\=/2` are now covered by the generated Go WAM builtin E2E test.
- `functor/3`, `arg/3`, `=../2`, and `copy_term/2` are now covered by the
  generated Go WAM builtin E2E test.
- `aggregate_all(set(X), Goal, Set)` is now covered by the generated Go WAM
  builtin E2E test.
- `\+/1` over user predicates is now covered by the generated Go WAM builtin
  E2E test for both failing and succeeding inner goals.
- `\+/1` now dispatches multi-clause WAM targets through a `runNegationParallel`
  helper that races isolated clause bodies and fails fast when any branch
  succeeds, matching the Haskell race-to-true control shape.
- `call_indexed_atom_fact2` is now parsed and executed by the Go WAM runtime,
  including backtracking over multiple indexed atom facts.
- Headered two-column TSV files can now be loaded into Go WAM indexed atom
  fact tables and queried through `call_indexed_atom_fact2`.
- Inline and TSV atom fact pairs now register through a Go `AtomFact2Source`
  interface with `Scan` and `LookupArg1`, matching the Haskell FactSource shape
  while preserving the existing indexed pair table used by native kernels.
- LMDB relation artifacts can now be registered as Go `AtomFact2Source`
  adapters via the existing `lmdb_relation_artifact` helper command. The helper
  path defaults to `lmdb_relation_artifact` and can be overridden with
  `UW_LMDB_RELATION_ARTIFACT_BIN`, so tests and deployments can provide the
  concrete LMDB reader without adding a generated-project Go dependency.

## Recommended Follow-Up Order

1. Continue broadening generated Go WAM E2E coverage for any remaining
   cross-target builtin edge cases.
2. If Go should avoid a helper process for LMDB, add an optional native Go
   LMDB build-tag path once a dependency policy is settled.

## Verification Commands

Use these checks after touching Go WAM parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```

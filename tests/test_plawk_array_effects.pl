:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The one enumeration of array effects -- docs/design/PLAWK_ARRAY_VALUE_MODEL.md
% §6.3. Value-kind inference, the escape rule and `readonly` all need EVERY read,
% write and import of an array, so they share plawk_array_effects/2 instead of each
% walking the AST. Before it, `n = split($0, a, ":")` hid its write from the older
% helpers (plawk_action_table_name/2, plawk_scalar_nested_action/2): the parser
% wraps the split in split_count/2. Completeness is CHECKED: every occurrence of an
% array's var(Name) outside a recognised table position is reported by
% plawk_array_unknown_uses/2 (empty for every program in the suites).

:- use_module(library(plunit)).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

effects(Src, Effects) :-
    plawk_parse_string(Src, Program),
    plawk_array_effects(Program, Effects).

:- begin_tests(plawk_array_effects).

% the write a wrapper node used to hide
test(split_count_write_is_found) :-
    effects("{ n = split($0, a, \":\"); print a[1] }\n", E),
    assertion(E == [read(a), write(a, split)]),
    !.

test(writes_by_kind) :-
    effects("{ c[$1]++; s[$1] += $2; delete d[$1]; t[$1] = $0 }\n", E),
    assertion(E == [write(c, inc), write(d, delete), write(s, add(field(2))),
                    write(t, row)]),
    !.

% BEGIN, rule patterns, rule bodies, nested branches and END are all covered
test(every_program_part_is_covered) :-
    effects("BEGIN { c[\"x\"]++ } { if ($1 in c) delete c[$1] } END { for (k in c) print k, c[k] }\n", E),
    assertion(E == [read(c), write(c, delete), write(c, inc)]),
    !.

% the `!seen[$1]++` pattern both reads and writes
test(postinc_pattern_reads_and_writes) :-
    effects("!seen[$1]++\n", E),
    assertion(E == [read(seen), write(seen, inc)]),
    !.

% multi-pass: a row capture in one pass, a `rows of` reader in another
test(pass_containers_are_covered) :-
    effects("pass { t[$1] = $0 }\npass rows of t { print $1 }\n", E),
    assertion(E == [read(t), write(t, row)]),
    !.

% `rows of t as r`: r is a view of t's current row, not an array; `r["col"]` is a
% read of t (found by the completeness check on the suites' own programs)
test(row_variable_reads_are_table_reads) :-
    Program = program_passes([], [pass([rule(always, [set_row(var(t), field(1))])]),
        pass_rows(var(r), var(t), [rule(always, [print([assoc(var(r), string("x"))])])])], []),
    plawk_array_effects(Program, E),
    assertion(E == [read(t), row_binding(r, t), write(t, row)]),
    plawk_array_names(Program, Names),
    assertion(Names == [t]),
    plawk_array_unknown_uses(Program, U),
    assertion(U == []),
    !.

% a Prolog / dynamic bind is an IMPORT, and the array escapes through it
test(bind_is_an_import_and_an_escape) :-
    plawk_parse_string("{ arr = dyncall($1) as assoc(str) }\n", P),
    plawk_array_effects(P, E),
    assertion(E == [import(arr, bind)]),
    assertion(plawk_array_escapes(P, arr, bind)),
    !.

% a local split is NOT an escape (the surface posarray producer lists it too, so
% it cannot serve as the escape enumerator unfiltered)
test(local_split_does_not_escape) :-
    plawk_parse_string("{ n = split($0, a, \":\"); print a[1] }\n", P),
    assertion(\+ plawk_array_escapes(P, a, _)),
    !.

% completeness: a var(Name) of an array in a node the enumeration does not know is
% reported, so a consumer can decline rather than miss an effect
test(unknown_use_is_reported) :-
    Program = program([], [rule(always, [inc_assoc(var(c), field(1)),
                                         mystery_node(var(c))])], []),
    plawk_array_unknown_uses(Program, U),
    assertion(U == [c-mystery_node/1]),
    !.

test(known_programs_have_no_unknown_uses) :-
    forall(member(Src,
            [ "{ c[$1]++ } END { for (k in c) print k, c[k] }\n",
              "{ n = split($0, a, \":\"); for (i = 1; i <= n; i++) print a[i] }\n",
              "{ if (($1, $2) in pairs) print; pairs[$1, $2]++ }\n",
              "{ k = $1; c[k]++ } END { print c[\"a\"] }\n"
            ]),
        ( plawk_parse_string(Src, P),
          plawk_array_unknown_uses(P, U),
          assertion(U == []) )),
    !.

:- end_tests(plawk_array_effects).

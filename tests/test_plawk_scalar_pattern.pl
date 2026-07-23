:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Scalar-variable patterns: a user scalar as a rule guard, both a bare
% scalar-vs-integer (`n > 2 { … }`, reversed `2 < n`, parsed to
% scalar_cmp(Name, Op, Value)) and a field-vs-scalar (`$1 > n { … }`, reversed
% `n > $1`, parsed to field_scalar_cmp(Index, Op, Name)). The codegen resolves
% Name to the rule's current slot SSA value (rule 0: the loop-head phi; rule N>0:
% the per-rule input phi) via plawk_resolve_scalar_cmp/4 and emits either an i64
% icmp (bare scalar) or the same numeric field-comparison runtime as `$I OP int`
% (field-vs-scalar), so both compose with `!`/`&&`/`||` and the other base
% patterns. Field-vs-scalar shares the field-vs-int-literal semantics exactly (a
% non-numeric/missing field makes the comparison false).
%
% Scope (v1): the scalar must be an i64 (counter) slot -- i.e. assigned somewhere
% in the program (a rule body), so it has a slot. A scalar never assigned (no
% slot) or a double/string scalar is cleanly declined at compile time rather than
% mis-lowered (an intentional conservative boundary; auto-creating a zero-valued
% read-only slot, and double/string scalar compares, are follow-ons).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_scalar_pattern).

% --- parsing ----------------------------------------------------------------

% `n > 2` parses to scalar_cmp(n, gt, 2).
test(scalar_gt_parses) :-
    plawk_parse_string("n > 2 { print $0 }\n",
        program([], [rule(scalar_cmp(n, gt, 2), [print([field(0)])])], [])),
    !.

% reversed `2 < n` swaps the operator to gt.
test(reversed_scalar_parses) :-
    plawk_parse_string("2 < n { print $0 }\n",
        program([], [rule(scalar_cmp(n, gt, 2), [print([field(0)])])], [])),
    !.

% composes with a field-equality guard via `&&`.
test(scalar_combinator_parses) :-
    plawk_parse_string("n > 2 && $1 == \"x\" { print $0 }\n",
        program([], [rule(and_pat(scalar_cmp(n, gt, 2), field_eq(1, "x")),
            [print([field(0)])])], [])),
    !.

% the numeric specials keep the special path (not captured as scalars).
test(nr_stays_special) :-
    plawk_parse_string("NR > 2 { print $0 }\n",
        program([], [rule(special_cmp('NR', gt, 2), [print([field(0)])])], [])),
    !.
test(nf_stays_special) :-
    plawk_parse_string("NF > 2 { print $0 }\n",
        program([], [rule(special_cmp('NF', gt, 2), [print([field(0)])])], [])),
    !.

% a call `foo(1)` is a foreign guard, not a scalar comparison.
test(call_stays_prolog_guard) :-
    plawk_parse_string("foo(1) { print $0 }\n",
        program([], [rule(prolog_guard(foo, [int(1)]), [print([field(0)])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `n > 2` on a per-record counter selects records after the second.
test(scalar_gt_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sg', "{ n++ } n > 2 { print $0 }\n", "a\nb\nc\nd\ne\n", Out),
    assertion(Out == "c\nd\ne\n"), !.

% reversed `2 < n` matches the same records.
test(reversed_scalar_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rs', "{ n++ } 2 < n { print $0 }\n", "a\nb\nc\nd\n", Out),
    assertion(Out == "c\nd\n"), !.

% a window via `&&`: records 2 through 3.
test(scalar_window, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sw', "{ n++ } n >= 2 && n <= 3 { print $0 }\n",
        "a\nb\nc\nd\n", Out),
    assertion(Out == "b\nc\n"), !.

% negated scalar pattern via the combinator path.
test(scalar_negated, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sn', "{ n++ } !(n > 2) { print $0 }\n", "a\nb\nc\nd\n", Out),
    assertion(Out == "a\nb\n"), !.

% scalar pattern combined with a field-equality guard.
test(scalar_with_field, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sf', "{ n++ } n > 1 && $1 == \"x\" { print $2 }\n",
        "x a\nx b\ny c\n", Out),
    assertion(Out == "b\n"), !.

% scalar pattern feeding an accumulator read from END (multi-rule path).
test(scalar_scalar_end, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'se', "{ n++ } n > 2 { c++ } END { print c }\n",
        "a\nb\nc\nd\ne\n", Out),
    assertion(Out == "3\n"), !.

% --- field-vs-scalar ($I OP NAME) ------------------------------------------

% `$1 > n` parses to field_scalar_cmp(1, gt, n).
test(field_scalar_parses) :-
    plawk_parse_string("$1 > n { print $0 }\n",
        program([], [rule(field_scalar_cmp(1, gt, n), [print([field(0)])])], [])),
    !.

% reversed `n > $1` swaps to the field-relative op (lt).
test(reversed_field_scalar_parses) :-
    plawk_parse_string("n > $1 { print $0 }\n",
        program([], [rule(field_scalar_cmp(1, lt, n), [print([field(0)])])], [])),
    !.

% `$1 > NR` is NOT a field-vs-scalar (NR is a special); field-vs-special is not
% this feature, so it does not parse.
test(field_vs_special_rejected) :-
    \+ plawk_parse_string("$1 > NR { print $0 }\n", _).

% `$1 > t` compares the field to the scalar's current value (a rising threshold).
test(field_scalar_gt, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fg', "{ t++ } $1 > t { print $0 }\n",
        "5 1\n1 2\n9 3\n2 4\n", Out),
    assertion(Out == "5 1\n9 3\n"), !.

% reversed `t < $1` selects the same records.
test(field_scalar_reversed, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fv', "{ t++ } t < $1 { print $0 }\n",
        "5 1\n1 2\n9 3\n2 4\n", Out),
    assertion(Out == "5 1\n9 3\n"), !.

% `$2 <= lim` compares a different field.
test(field_scalar_le_other_field, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fl', "{ lim++ } $2 <= lim { print $1 }\n",
        "a 1\nb 5\nc 3\n", Out),
    assertion(Out == "a\nc\n"), !.

% composes with a field-int guard via `&&`.
test(field_scalar_combined, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fc', "{ t++ } $1 > t && $1 < 100 { print $0 }\n",
        "5 1\n200 2\n9 3\n", Out),
    assertion(Out == "5 1\n9 3\n"), !.

% a non-numeric field makes the comparison false (same as `$1 > int`).
test(field_scalar_nonnumeric, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fn', "{ t++ } $1 > t { print $0 }\n",
        "abc 1\n9 2\n", Out),
    assertion(Out == "9 2\n"), !.

:- end_tests(plawk_scalar_pattern).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_scalar_pattern', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

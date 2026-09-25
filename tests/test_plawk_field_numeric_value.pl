:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The NUMERIC value of a field, awk-style -- phase A of the numeric-fields fix.
%
% ---------------------------------------------------------------------------
% A field in a numeric context was read with a strict decimal-integer parse whose
% failure defaults to 0, so any non-integer text -- "30.25", "3abc" -- was worth 0.
% Silent wrong output, exit 0. awk reads a field numerically with strtod semantics
% (leading number, trailing text ignored, non-numeric -> 0).
%
% Phase A (this suite) fixes the contexts that do not change a slot's TYPE:
%   - a field in a DOUBLE expression (`$2 * 1.5`, `$2 / 2`) reads via strtod
%     (plawk_f64_expr_ir(field(N)) -> the `float($N)` row);
%   - `int($N)` is the numeric value truncated toward zero
%     (@wam_awk_field_int_value: the exact strict parse first, else strtod +
%     range-checked truncation), in print, arithmetic, `+=` and assignment; and
%     `x = int($N)` is a NUMBER, no longer a strnum copy of the field text.
%
% STILL WRONG, pinned below as known divergences for phases B/C: a field in pure
% INTEGER arithmetic (`$2 + 1`, `s += $2`) and `$N CMP int` comparisons.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("alice 30.25 NY 7\nbob 5 LA -2.9\ncarol 3abc SF x\n").

:- begin_tests(plawk_field_numeric_value).

% --- double expressions -------------------------------------------------------

test(field_in_double_expression, [condition(clang_available)]) :-
    run("{ print $2 * 1.5 }\n", "45.375\n7.5\n4.5\n"),
    !,
    run("{ print $2 * 0.5 + $4 }\n", "22.125\n-0.4\n1.5\n"),
    !,
    run("{ print $2 / 2 }\n", "15.125\n2.5\n1.5\n"),
    !.

test(double_accumulator_over_fields, [condition(clang_available)]) :-
    run("{ s += $2 * 1.0 } END { print s }\n", "38.25\n"),
    !,
    run("{ x = $2 * 1.0; print x }\n", "30.25\n5\n3\n"),
    !.

% --- int() -------------------------------------------------------------------

test(int_truncates_the_numeric_value, [condition(clang_available)]) :-
    run("{ print int($2) }\n", "30\n5\n3\n"),
    !,
    run("{ print int($4) }\n", "7\n-2\n0\n"),
    !,
    run("{ print int($2) + 1 }\n", "31\n6\n4\n"),
    !.

test(int_in_accumulators_and_assignment, [condition(clang_available)]) :-
    run("{ s += int($2) } END { print s }\n", "38\n"),
    !,
    run("{ x = int($2); y = x + 1; print y }\n", "31\n6\n4\n"),
    !.

% `x = int($N)` is a number, not a copy of the field's text.
test(int_assignment_is_a_number_not_text, [condition(clang_available)]) :-
    run("{ x = int($2); print x }\n", "30\n5\n3\n"),
    !,
    % while a bare copy still keeps the text, as in awk
    run("{ x = $2; print x }\n", "30.25\n5\n3abc\n"),
    !.

% Integer text keeps its EXACT value (the strict parse is the fast path). gawk rounds
% through a double above 2^53 and prints 9007199254740992; plawk's integers are
% exact i64 -- a deliberate, pre-existing difference, pinned so it is not mistaken
% for a regression.
test(int_of_large_integer_text_is_exact, [condition(clang_available)]) :-
    run_with("a 9007199254740993\n", "{ print int($2) }\n", "9007199254740993\n", 0),
    !.

% Outside the i64 range the truncation is fatal (exit 2), never a wrong number.
test(int_out_of_range_is_fatal, [condition(clang_available)]) :-
    run_with("a 5\nb 1e30\n", "{ print int($2) }\n", "5\n", 2),
    !.

% --- known divergences, left for phases B and C -------------------------------
% These pin CURRENT behaviour so the phases that fix them flip a visible test.

test(known_integer_arithmetic_divergence, [condition(clang_available)]) :-
    % gawk: 31.25 6 4
    run("{ print $2 + 1 }\n", "1\n6\n1\n"),
    !,
    % gawk: 38.25
    run("{ s += $2 } END { print s }\n", "5\n"),
    !.

% Phase B fixed `$N OP int` (awk strnum semantics): this pin, a known divergence
% until then (plawk printed nothing), now matches gawk. Full coverage in
% tests/test_plawk_strnum_field_cmp.pl.
test(integer_comparison_now_matches_gawk, [condition(clang_available)]) :-
    % "3abc" is not numeric-looking: a STRING comparison, "3abc" > "10"
    run("$2 > 10 { print $1 }\n", "alice\ncarol\n"),
    !.

:- end_tests(plawk_field_numeric_value).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_field_numeric_value', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected, 0).

run_with(Input, Src, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'fnv_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'fnv', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(BO)), stderr(null), process(BPid)]),
    read_string(BO, _, _), close(BO),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(null), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q (exit ~w)~n  expected ~q (exit ~w)~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

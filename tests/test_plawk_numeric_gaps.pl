:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The two numeric gaps left by the numeric-semantics phases, both SILENT WRONG
% OUTPUT before this change:
%
% 1. A strnum scalar in arithmetic (x = $2 copies the field as a strnum):
%        { x = $2; y = x + 1; print y }      "30.25" -> gawk 31.25, plawk 1
%        { x = $2; s += x } END { print s }  decimals summed as 0
%    A strnum's numeric value is its strtod reading, so arithmetic over one is
%    double, exactly as for a field: the double fixpoint is told the strnum names
%    (a strnum operand of a binary tree, and `s += x` accumulators of one), and a
%    substituted strnum read reads through @wam_awk_strtod.
%
% 2. A DOUBLE-valued assoc array:
%        { c[$1] += $2 } END { for (k in c) print k, c[k] }  30.25 + 3abc -> 0
%    An array with a field-valued `+=` stores double BIT PATTERNS in the ordinary
%    i64 table (an absent key is bit pattern 0 = +0.0) through @wam_assoc_f64_add,
%    @wam_assoc_f64_value_at and @wam_assoc_f64_print, taught to the `+=` emitter,
%    the iterated END for-in value print and the literal-key print.
%
%    The guarantee is checked on the OUTPUT at the driver entry: every IR line that
%    touches such a table must be an allowed call (iteration, key, existence,
%    delete, new/free, an f64 entry), and every f64 entry must touch a table the
%    `+=` emitter marked. Anything else -- `++` on the same array, a cross-table
%    lookup, a concatenated read -- DECLINES rather than taking the bits for an
%    integer. Teeth: with the check disabled, `print c["a"]` printed the raw bit
%    pattern 4629876338797314048.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 30.25\nb 5\na 3abc\nc 1250000\n").

:- begin_tests(plawk_numeric_gaps).

% --- 1. strnum scalars in arithmetic ------------------------------------------

test(strnum_operand_of_arithmetic, [condition(clang_available)]) :-
    run("{ x = $2; y = x + 1; print y }\n", "31.25\n6\n4\n1250001\n"),
    !,
    run("{ x = $2; z = (x + 1) * 2; print z }\n", "62.5\n12\n8\n2500002\n"),
    !.

test(strnum_accumulator, [condition(clang_available)]) :-
    % 1250038.25 is not integral: gawk prints it through %.6g
    run("{ x = $2; s += x } END { print s }\n", "1.25004e+06\n"),
    !,
    run("{ x = $2; s = s + x; n++ } END { print n, s }\n", "4 1.25004e+06\n"),
    !.

% The copy itself stays a strnum: text in print, strnum semantics in comparisons.
test(strnum_copy_is_unchanged, [condition(clang_available)]) :-
    run("{ x = $2; print x }\n", "30.25\n5\n3abc\n1250000\n"),
    !,
    run("{ x = $2; y = x; print y }\n", "30.25\n5\n3abc\n1250000\n"),
    !,
    run("{ x = $2; if (x > 10) print $1 }\n", "a\na\nc\n"),
    !.

% --- 2. double-valued assoc arrays --------------------------------------------

test(assoc_sum_of_decimals, [condition(clang_available)]) :-
    run_sorted("{ c[$1] += $2 } END { for (k in c) print k, c[k] }\n",
        ["a 33.25", "b 5", "c 1250000"]),
    !,
    run_sorted("{ c[$1] += $2 } END { for (k in c) print c[k], k }\n",
        ["1250000 c", "33.25 a", "5 b"]),
    !.

test(assoc_literal_key_print, [condition(clang_available)]) :-
    run("{ c[$1] += $2 } END { print c[\"a\"] }\n", "33.25\n"),
    !.

test(assoc_double_next_to_an_integer_array, [condition(clang_available)]) :-
    run("{ c[$1] += $2; d[$1]++ } END { print c[\"a\"], d[\"a\"] }\n", "33.25 2\n"),
    !,
    run_sorted("{ c[$1] += $2; d[$1]++ } END { for (k in c) print k, c[k] }\n",
        ["a 33.25", "b 5", "c 1250000"]),
    !.

test(assoc_under_a_pattern, [condition(clang_available)]) :-
    run_sorted("$2 > 0 { c[$1] += $2 } END { for (k in c) print k, c[k] }\n",
        ["a 33.25", "b 5", "c 1250000"]),
    !.

% Integer-only deltas and counters keep the i64 table (no f64 entry at all).
test(integer_arrays_unchanged) :-
    ir_of("{ c[$1] += 2; d[$1]++ } END { for (k in c) print k, c[k] }\n", IR),
    \+ sub_atom(IR, _, _, _, '@wam_assoc_f64_'),
    \+ sub_atom(IR, _, _, _, 'plawk-f64-table'),
    !.

test(ir_double_table_uses_only_f64_value_entries) :-
    ir_of("{ c[$1] += $2 } END { for (k in c) print k, c[k] }\n", IR),
    sub_atom(IR, _, _, _, 'call double @wam_assoc_f64_add(%WamAssocI64Table* %plawk_assoc_table_0'),
    sub_atom(IR, _, _, _, 'call double @wam_assoc_f64_value_at(%WamAssocI64Table* %plawk_assoc_table_0'),
    \+ sub_atom(IR, _, _, _, '@wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_0'),
    !.

% Uses of a double table no emitter is taught decline -- never read the bits.
test(untaught_double_table_uses_decline) :-
    forall(member(Src,
            [ "{ c[$1] += $2; c[$1]++ } END { for (k in c) print k, c[k] }\n",
              "{ c[$1] += $2; d[$1]++ } END { for (k in d) print k, d[k], c[k] }\n",
              "{ c[$1] += $2 } END { print c[\"a\"], c[\"zz\"] \"|\" }\n"
            ]),
        build_status(Src, 3)),
    !.

:- end_tests(plawk_numeric_gaps).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_numeric_gaps', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    build_and_run(Src, Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

% for-in order is unspecified in awk: compare the lines as a set.
run_sorted(Src, ExpectedLines) :-
    build_and_run(Src, Out),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines1),
    msort(Lines1, Lines),
    maplist([A, S]>>atom_string(A, S), ExpectedLines, Expected0),
    msort(Expected0, Expected),
    ( Lines == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Lines, Expected]), fail
    ).

build_and_run(Src, Out) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'ng_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ng', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ng_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( Status == ExpectedStatus
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, ExpectedStatus]), fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).

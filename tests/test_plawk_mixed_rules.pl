:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Rules that mix array (assoc) actions with scalar work -- the mixed driver.
%
% ---------------------------------------------------------------------------
% PR 1 of the mixed-rules plan (design reviewed before implementation). The rule
% walker already had table rows (field-key `c[$1]++`, var-key updates, `delete`,
% `arr[k]` reads), shared by the scalar and mixed chains. What declined was
% ADMISSION:
%
%   - plawk_mixed_assoc_count_plan/3 required the END to print an array element or
%     some rule to test `in`, so a program whose tables are only WRITTEN declined:
%         { c[$1]++; n++ } END { print n }        (exit 3; gawk: the record count)
%     The mixed plan is program-wide, so splitting the work across two rules did
%     not help either. The guard is gone; a table-establishing action and a scalar
%     slot (or a conditional) are still required, so pure-assoc and pure-scalar
%     programs keep their own drivers -- IR of every program that compiled before
%     is byte-identical.
%   - there was no mixed driver for a program with NO END; one now exists (the
%     single-print clause minus the END print: end_print frees the tables).
%
% PR 2: field-key reads in a body print (`print n, c[$1]`); an absent element prints
% "" (not 0). PR 3: `split($N, arr, "sep")` with scalar work -- the split table is
% POSITIONAL and str-valued, a different kind from a counted table, so it is read
% only by a numeric position and never shares an array with counted uses (both
% checked on the generated IR). PR 4: `n = split(...)` (the count is a counter
% slot, marked assigned so an unset count still prints "") and loops in a mixed
% body -- the walker's loop rows are shared, so `for (i = 1; i <= n; i++) print
% a[i]` walks the pieces. PR 5: `for (k in arr) print ...` in a mixed body --
% print-only bodies through the walker's own print emitter; the loop key prints as
% a number on a split table and as text on a counted one (the split-table IR check
% requires the positional key read to be tagged).
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

input("a 1\nb 2\na 3\n").

:- begin_tests(plawk_mixed_rules).

test(written_tables_with_scalar_end, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n }\n", "3\n"),
    !,
    run("{ c[$1]++; s += $2 } END { print s }\n", "6\n"),
    !.

% The mixed plan is program-wide: the same work split across two rules.
test(across_two_rules, [condition(clang_available)]) :-
    run("{ c[$1]++ } { n++ } END { print n }\n", "3\n"),
    !.

% No END at all.
test(no_end, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ }\n", ""),
    !.

% The shapes that compiled before still do (and their IR is unchanged).
test(end_reading_the_table_still_works, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n, c[\"a\"] }\n", "3 2\n"),
    !,
    run("{ c[$1]++; n++ } END { print n; print c[\"a\"] }\n", "3\n2\n"),
    !.

% --- PR 2: element reads in a rule-body print -------------------------------

% `print c[$1]` -- a FIELD-keyed read, through the table print helper.
test(field_key_read_in_a_body_print, [condition(clang_available)]) :-
    run("{ c[$1]++; n++; print n, c[$1] }\n", "1 1\n2 1\n3 2\n"),
    !,
    run("{ c[$1]++; n++; print $1 \"=\" c[$1] }\n", "a=1\nb=1\na=2\n"),
    !.

% An ABSENT element prints as awk's empty string, not 0 -- for a field key and for a
% variable key. The variable-key read printed 0 before this change (wrong output,
% also on the base: `{ k = $1; print c[k] "|"; c[k]++ }` printed "0|").
test(absent_element_prints_empty, [condition(clang_available)]) :-
    run("{ n++; print c[$1] \"|\"; c[$1]++ }\n", "|\n|\n1|\n"),
    !,
    run("{ k = $1; print c[k] \"|\"; c[k]++ }\n", "|\n|\n1|\n"),
    !,
    run("{ k = $1; c[k]++; j = \"zz\"; print c[j] \"|\" }\n", "|\n|\n|\n"),
    !,
    run("{ n++; c[n]++; m = n + 5; print c[m] \"|\" }\n", "|\n|\n|\n"),
    !,
    % present keys unchanged
    run("{ k = $1; c[k]++; print k, c[k] }\n", "a 1\nb 1\na 2\n"),
    !.

% --- PR 3: split with scalar work -------------------------------------------

% Pieces read by an integer position or a counter-valued scalar. The empty line's
% missing $1 splits as "" (the array is emptied, a[2] prints ""); a regex separator
% and a split of $0 take the same path. gawk 5.1.0.
test(split_with_scalar_work, [condition(clang_available)]) :-
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ split($1, a, \",\"); n++; print a[2], n }\n", "b 1\n 2\n 3\nr 4\n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ split($1, a, \",\"); n++; print n, a[1] \"|\" a[3] \"|\" }\n",
        "1 a|c|\n2 x||\n3 ||\n4 q||\n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ split($1, a, \",\"); i = 2; n++; print a[i] }\n", "b\n\n\nr\n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ split($1, a, \"[,r]\"); n++; print a[1], a[2] }\n", "a b\nx \n \nq \n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n++; split($0, a, \" \"); print a[2] }\n", "1\n2\n\n3\n"),
    !.

% A split under a condition: a record that does not split keeps the previous pieces.
test(conditional_split_keeps_previous_pieces, [condition(clang_available)]) :-
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ if (NF > 1) { split($1, a, \",\") } n++; print a[1] }\n", "a\nx\nx\nq\n"),
    !.

% The generated IR marks the split table, and every use of that table is the split
% fill, the positional str print, or its new/free.
test(split_table_is_marked_and_used_only_positionally) :-
    build_ll("{ split($1, a, \",\"); n++; print a[2], n }\n", LL),
    assertion(sub_string(LL, _, _, _, "; plawk-mixed-split(a)=0")),
    assertion(sub_string(LL, _, _, _, "@wam_assoc_str_print(%WamAssocI64Table* %plawk_assoc_table_0, i64 2)")),
    assertion(\+ sub_string(LL, _, _, _, "@wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_0")),
    !.

% Kind boundaries: a string key (a field, or a string-valued scalar) would need awk's
% string-to-position conversion; a counted use of a split array (an update, an `in`
% test, an END element read or for-in) mixes the two key spaces. All decline cleanly.
test(split_kind_boundaries_decline) :-
    forall(member(Src,
            [ "{ split($1, a, \",\"); n++; print a[$2] }\n",
              "{ split($1, a, \",\"); k = $2; print a[k] }\n",
              "{ split($1, a, \",\"); a[$2]++; n++ }\n",
              "{ split($1, a, \",\"); n++; if ((1 in a)) print \"y\" }\n",
              "{ split($1, a, \",\"); n++ } END { print n, a[1] }\n",
              "{ split($1, a, \",\"); n++ } END { for (k in a) print k, a[k] }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% --- PR 4: n = split(...), loops ---------------------------------------------

test(split_count_and_loops_over_the_pieces, [condition(clang_available)]) :-
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n = split($1, a, \",\"); print n }\n", "3\n1\n0\n2\n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n = split($1, a, \",\"); for (i = 1; i <= n; i++) print i, a[i] }\n",
        "1 a\n2 b\n3 c\n1 x\n1 q\n2 r\n"),
    !,
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n = split($1, a, \",\"); i = n; while (i > 0) { print a[i]; i-- } }\n",
        "c\nb\na\nx\nr\nq\n"),
    !,
    % do-while runs once on the empty record: a[0] is absent, prints ""
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n = split($1, a, \",\"); do { print a[n]; n-- } while (n > 0) }\n",
        "c\nb\na\nx\n\nr\nq\n"),
    !.

% The count is an assignment: on EMPTY input it was never assigned, and awk prints
% "" -- the slot carries the assigned mark like any update.
test(split_count_unset_prints_empty, [condition(clang_available)]) :-
    run_in("a,b,c 1\nx 2\n\nq,r 3\n",
        "{ n = split($1, a, \",\") } END { print n }\n", "2\n"),
    !,
    run_in("", "{ n = split($1, a, \",\") } END { print n }\n", "\n"),
    !.

% Loops beside counted tables (shared walker rows; the table plan reaches the body).
test(loops_beside_counted_tables, [condition(clang_available)]) :-
    run_in("a 1\nb 2\na 3\nc 4\n",
        "{ c[$1]++; i = 0; while (i < 2) { n++; i++ } } END { print n, c[\"a\"] }\n",
        "8 2\n"),
    !,
    run_in("a 1\nb 2\na 3\nc 4\n",
        "{ c[$1]++; for (i = 0; i < 2; i++) c[$1]++ } END { print c[\"a\"], c[\"b\"] }\n",
        "6 3\n"),
    !,
    run_in("a 1\nb 2\na 3\nc 4\n",
        "{ c[$1]++; i = 0; while (i < 3) { if (i == 1) break; i++ } ; n += i } END { print n, c[\"a\"] }\n",
        "4 2\n"),
    !.

% --- PR 5: for-in in a mixed body ------------------------------------------
% for-in order is unspecified: outputs compare as sorted lines.

test(forin_over_split_pieces, [condition(clang_available)]) :-
    run_sorted_in("a,b,c 1\nx 2\n\nq,r 1\n",
        "{ n = split($1, a, \",\"); for (k in a) print k, a[k] }\n",
        "1 a\n1 q\n1 x\n2 b\n2 r\n3 c\n"),
    !,
    run_sorted_in("a,b,c 1\nx 2\n\nq,r 1\n",
        "{ n = split($1, a, \",\"); for (k in a) print \"[\" k \"]\" a[k] }\n",
        "[1]a\n[1]q\n[1]x\n[2]b\n[2]r\n[3]c\n"),
    !,
    run_sorted_in("a,b,c 1\nx 2\n\nq,r 1\n",
        "{ n = split($1, a, \",\"); if (n > 1) for (k in a) print k, a[k] }\n",
        "1 a\n1 q\n2 b\n2 r\n3 c\n"),
    !.

test(forin_over_a_counted_table, [condition(clang_available)]) :-
    run_sorted_in("a 1\nb 2\na 3\nc 4\n",
        "{ c[$1]++; n++; for (k in c) print k, c[k], n }\n",
        "a 1 1\na 1 2\na 2 3\na 2 4\nb 1 2\nb 1 3\nb 1 4\nc 1 4\n"),
    !,
    run_sorted_in("a 1\nb 2\na 3\nc 4\n",
        "{ c[$1]++; n++; for (k in c) { print k; print c[k] } }\n",
        "1\n1\n1\n1\n1\n1\n2\n2\na\na\na\na\nb\nb\nb\nc\n"),
    !.

% NR/FNR in a for-in body, a cross-table read by the loop key, and a loop key that
% is also a scalar decline cleanly (never exit 4 -- NR was an undefined counter).
test(forin_boundaries_decline) :-
    forall(member(Src,
            [ "{ c[$1]++; n++; for (k in c) print k, NR }\n",
              "{ c[$1]++; n++; for (k in c) print k, FNR }\n",
              "{ n = split($1, a, \",\"); for (k in a) print k, c[k] }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% Forms the later PRs teach decline or fail to parse -- never exit 4.
test(later_forms_do_not_miscompile) :-
    forall(member(Src,
            [ "{ c[$1] += $2; n++; print c[$1] }\n",
              "{ c[$1]++; if (c[$1] > 1) print \"dup\", $1 }\n",
              "{ split($1, a, \",\"); n++; printf \"%s\\n\", a[1] }\n",
              "{ n = split($1, a, \",\"); for (i = 1; i <= n; i++) if (a[i] == \"b\") print \"found\" }\n",
              "{ for (i = 0; i < 2; i++) c[$1]++; n++ } END { print n, c[\"a\"] }\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_mixed_rules).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_rules', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_in(Input, Src, Expected).

run_in(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'mr_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mr', Prog0),
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
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status_not_4(Src) :-
    odir(Dir),
    directory_file_path(Dir, 'mr_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( memberchk(Status, [0, 2, 3])
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w (4 = clang miscompile)~n",
           [Src, Status]), fail
    ).

run_sorted_in(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'mr_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mr', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out0),
    close(PS),
    process_wait(Pid, exit(0)),
    split_string(Out0, "\n", "", Lines0),
    append(Lines, [""], Lines0),
    msort(Lines, Sorted),
    atomic_list_concat(Sorted, '\n', Joined),
    atom_string(Joined, JoinedS),
    string_concat(JoinedS, "\n", Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'mr_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( Status == Expected
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, Expected]), fail
    ).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).

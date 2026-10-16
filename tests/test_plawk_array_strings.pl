:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% String and strnum element writes (arrays PR 5a) --
% docs/design/PLAWK_ARRAY_VALUE_MODEL.md sections 5.1 / 5.2, plan item 4.
%
% `a[KEY] = $N` (strnum), `a[KEY] = "lit"` and a concatenation of fields and
% literals (string) store the joined text as an interned atom id -- a str-valued
% table, like a row capture's -- at a field, SUBSEP or literal key. The buffer is
% sized by a measuring snprintf, so a long value is never truncated. A strnum
% array compares numerically when the text looks numeric, a STRING array always
% lexically. Mixing kinds in one array declines (section 5.2), and so does a
% numeric fold over a str-valued table.
%
% PR 5c: the same writes beside scalar work (the mixed walker), and at an NR key
% -- `a[NR] = $0` with a numeric END loop is tac. A counter key reads by its
% decimal text (awk keys `a[1]` and `a["1"]` alike); a str-valued element prints
% its text, nothing when absent. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 5\na 2\nc 1\nb 4\na 7\n").
% strnum vs string matters here: 10 vs "10", 010, a non-numeric word
strnum_input("x 10\ny 9\nz abc\nw 010\n").

:- begin_tests(plawk_array_strings).

test(strnum_and_string_writes, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = $2 } END { for (k in a) print k, a[k] }\n", "a 7\nb 4\nc 1\n"),
    run_sorted("{ a[$1] = \"x\" } END { for (k in a) print k, a[k] }\n", "a x\nb x\nc x\n"),
    run_sorted("{ a[$1] = $1 \"-\" $2 } END { for (k in a) print a[k] }\n", "a-7\nb-4\nc-1\n"),
    strnum_input(S),
    run_sorted_on(S, "{ a[$1] = $2 } END { for (k in a) print k, a[k] }\n",
        "w 010\nx 10\ny 9\nz abc\n"),
    !.

% A literal part is snprintf FORMAT text: `%` must be doubled, never interpreted.
test(literal_percent_and_escapes, [condition(clang_available)]) :-
    strnum_input(S),
    run_sorted_on(S, "{ a[$1] = \"100% \" $2 \"\\t%s%d\" } END { for (k in a) print k, a[k] }\n",
        "w 100% 010\t%s%d\nx 100% 10\t%s%d\ny 100% 9\t%s%d\nz 100% abc\t%s%d\n"),
    !.

test(missing_field_is_empty, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = $5 } END { for (k in a) print k, a[k], \"|\" }\n",
        "a  |\nb  |\nc  |\n"),
    !.

test(key_forms, [condition(clang_available)]) :-
    run_sorted("{ a[$1,$2] = $2 } END { for (k in a) print a[k] }\n", "1\n2\n3\n4\n5\n7\n"),
    run_sorted("{ a[\"last\"] = $1 } END { print a[\"last\"] }\n", "a\n"),
    run_sorted("{ a[5] = $1 } END { print a[5], a[\"5\"] }\n", "a a\n"),
    run_sorted("{ last[$1] = $2 } END { print last[\"a\"], last[\"zz\"] }\n", "7 \n"),
    run_sorted("{ a[$1] = $2 } $2 > 4 { print $1, a[$1] }\n", "a 7\nb 5\n"),
    !.

% The value is heap-sized: a 6000-byte field survives whole.
test(long_value_is_not_truncated, [condition(clang_available)]) :-
    length(Cs, 6000), maplist(=(0'x), Cs), string_codes(Long, Cs),
    format(string(In), "k ~s~n", [Long]),
    format(string(Want), "~s~n", [Long]),
    run_sorted_on(In, "{ a[$1] = $2 } END { for (k in a) print a[k] }\n", Want),
    !.

% strnum (`= $2`) is numeric when the text looks numeric; string (`= $2 ""`) is
% always lexical -- "10" < "4" -- including against a float.
test(strnum_vs_string_comparison, [condition(clang_available)]) :-
    strnum_input(S),
    run_sorted_on(S, "{ a[$1] = $2 } END { for (k in a) if (a[k] > 9) print k }\n",
        "w\nx\nz\n"),
    run_sorted_on(S, "{ a[$1] = $2 \"\" } END { for (k in a) if (a[k] > 4) print k }\n",
        "y\nz\n"),
    run_sorted_on(S, "{ a[$1] = $2 \"\" } END { for (k in a) if (a[k] < 3.5) print k }\n",
        "w\nx\n"),
    run_sorted_on(S, "{ a[$1] = $2 } END { for (k in a) if (a[k] == \"10\") print k }\n",
        "x\n"),
    !.

test(mixed_kinds_and_numeric_folds_decline) :-
    forall(member(Src,
            [ % strnum + string, strnum + counter, split (positional) + strnum
              "{ a[$1] = $2; a[$1] = \"x\" } END { for (k in a) print k, a[k] }\n",
              "{ a[$1] = $2; a[$1]++ } END { for (k in a) print k, a[k] }\n",
              "{ split($0, p, \" \"); p[$1] = $2 } END { for (k in p) print k, p[k] }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% An END fold over a str-valued table sums awk's strtod of each element (array
% plan item 5(d)); an i64 fold summed atom ids -- row capture did, before PR 5a.
test(numeric_folds, [condition(clang_available)]) :-
    run_exact("{ a[$1] = $2 } END { for (k in a) t += a[k]; print t }\n", "12\n"),
    run_exact("{ a[$1] = $0 } END { for (k in a) t += a[k]; print t }\n", "0\n"),
    !.

test(string_kind_comparison_declines) :-
    forall(member(Src,
            [ "{ a[$1] = $2 \"\"; n++; if (a[$1] > 9) print $1 }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% tac and its relatives: output ORDER is the point, so these compare exactly.
test(tac_and_nr_keys, [condition(clang_available)]) :-
    run_exact("{ a[NR] = $0 } END { for (i = NR; i >= 1; i--) print a[i] }\n",
        "a 7\nb 4\nc 1\na 2\nb 5\na 3\n"),
    run_exact("{ a[NR] = $0 } END { i = NR; while (i > 0) { print a[i]; i-- } }\n",
        "a 7\nb 4\nc 1\na 2\nb 5\na 3\n"),
    run_exact("{ n++; a[NR] = $1 \"=\" $2 } END { for (i = 1; i <= n; i++) print a[i] }\n",
        "a=3\nb=5\na=2\nc=1\nb=4\na=7\n"),
    run_exact("{ a[NR] = $2 } END { for (i = 1; i <= NR; i++) print i, a[i] }\n",
        "1 3\n2 5\n3 2\n4 1\n5 4\n6 7\n"),
    run_exact("{ a[NR] = $0 } END { print a[1], a[3] }\n", "a 3 a 2\n"),
    run_sorted("{ a[NR] = $0 } END { for (k in a) print k, a[k] }\n",
        "1 a 3\n2 b 5\n3 a 2\n4 c 1\n5 b 4\n6 a 7\n"),
    !.

% Beside scalar work (the mixed walker): END for-in and literal-key prints, a rule
% body's field-keyed print, and a counter-keyed read of an absent element.
test(string_writes_beside_scalars, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = $2; n++ } END { for (k in a) print k, a[k], n }\n",
        "a 7 6\nb 4 6\nc 1 6\n"),
    run_exact("{ a[$1] = $2; n++ } END { print a[\"a\"], a[\"zz\"], n }\n", "7  6\n"),
    run_exact("{ a[$1] = $2; n++; print $1, a[$1] \"|\", a[\"zz\"] \"|\" }\n",
        "a 3| |\nb 5| |\na 2| |\nc 1| |\nb 4| |\na 7| |\n"),
    run_exact("{ a[$1] = $0; n++ } END { for (i = 1; i <= 3; i++) print i, a[i] \"|\" }\n",
        "1 |\n2 |\n3 |\n"),
    !.

% Numeric reads of a strnum table in the mixed walker (array plan item 5(d)): the
% element's text read as awk's strtod; they declined before, rather than read an
% atom id as a number.
test(numeric_reads_of_strnum_tables, [condition(clang_available)]) :-
    run_exact("{ a[$1] = $2; n++ } END { for (k in a) { t += a[k] }; print t }\n", "12\n"),
    run_exact("{ a[$1] = $2; n++; t += a[$1] } END { print t }\n", "22\n"),
    run_exact("{ a[$1] = $2; n++; if (a[$1] > 3) print $1 }\n", "b\nb\na\n"),
    run_exact("{ a[NR] = $2 } END { for (i = 1; i <= NR; i++) t += a[i]; print t }\n", "22\n"),
    run_sorted("{ a[$1] = $2; n++ } END { for (k in a) print k, a[k] * 2 }\n",
        "a 14\nb 8\nc 2\n"),
    !.

:- end_tests(plawk_array_strings).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_array_strings', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Src, Expected) :-
    input(Input),
    run_sorted_on(Input, Src, Expected).

run_exact(Src, Expected) :-
    input(Input),
    run_out(Input, Src, Out),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

run_out(Input, Src, Out) :-
    odir(Dir),
    directory_file_path(Dir, 'ase_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ase', Prog0),
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

run_sorted_on(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'as_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'as', Prog0),
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
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    atomic_list_concat(Sorted, '\n', Joined0),
    atom_string(Joined0, Joined),
    string_concat(Joined, "\n", Out),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'as_st_bin', Bin),
    directory_file_path(Dir, 'as_st', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

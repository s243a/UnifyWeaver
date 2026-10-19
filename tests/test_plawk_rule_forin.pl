:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `for (k in arr) BODY` in a RULE body with a general body -- scalar updates, an
% `if`, element deletes, break / continue -- and so `delete arr` (which parses to
% that loop) in a rule. A rule-body for-in was print-only; the general for-in row
% the END block uses now serves rule bodies too, and the rule-body planner types
% the loop key a string (the END planning flatten, at any depth).
%
% for-in order is unspecified, so outputs are compared as sorted lines.
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(fields, "a 3\nb 10\na 7\nc 1\nb 2\nd 4\n").

:- begin_tests(plawk_rule_forin).

test(scalar_work_in_the_loop, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++; for (k in c) n++ } END { print n }\n", "15\n"),
    run(fields, "{ c[$1]++; t = 0; for (k in c) t += c[k]; print t }\n",
        "1\n2\n3\n4\n5\n6\n"),
    run(fields, "{ c[$1] += $2; s = 0; for (k in c) s += c[k]; print NR, s }\n",
        "1 3\n2 13\n3 20\n4 21\n5 23\n6 27\n"),
    run(fields, "{ c[$1]++; for (k in c) if (c[k] > 1) m++ } END { print m }\n", "6\n"),
    run(fields, "{ c[$1]++; for (k in c) { if (k == \"b\") continue; n++ } } END { print n }\n",
        "10\n"),
    !.

test(delete_whole_array_in_a_rule, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++; if (NR == 3) delete c } END { for (k in c) print k, c[k] }\n",
        "b 1\nc 1\nd 1\n"),
    run(fields, "{ c[$1]++ } NR == 3 { delete c } END { for (k in c) print k, c[k] }\n",
        "b 1\nc 1\nd 1\n"),
    run(fields,
        "{ c[$1]++; if ($2 > 5) { delete c; n++ } } END { print n; for (k in c) print k, c[k] }\n",
        "2\nb 1\nc 1\nd 1\n"),
    run(fields,
        "{ c[$1]++; if (NR == 3) for (j in c) delete c[j] } END { for (k in c) print k, c[k] }\n",
        "b 1\nc 1\nd 1\n"),
    !.

:- end_tests(plawk_rule_forin).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rule_forin', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

write_prog(Src, Prog0, Prog) :-
    odir(Dir),
    directory_file_path(Dir, Prog0, P0),
    atom_concat(P0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)).

build_status(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'rf_st_bin', Bin),
    write_prog(Src, 'rf_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'rf_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'rf', Prog),
    directory_file_path(Dir, 'rf_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)),
    (   sorted_lines(Out, Got), sorted_lines(Expected, Want), Got == Want
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

sorted_lines(Text, Sorted) :-
    split_string(Text, "\n", "", Lines),
    msort(Lines, Sorted).

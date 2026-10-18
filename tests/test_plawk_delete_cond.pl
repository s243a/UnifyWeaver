:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `delete arr[k]` under an `if` in a rule body, beside other work:
% `{ c[$1]++; if ($2 > 5) delete c[$1] }`. A field, literal or SUBSEP-keyed
% delete was kept out of the mixed chain (only a scalar-variable key was
% admitted), and the pure-assoc chain has no `if`, so these declined. The mixed
% walker interns the key as the writes do (plawk_cond_elem_key/9).
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

:- begin_tests(plawk_delete_cond).

test(delete_under_if, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++; if ($2 > 5) delete c[$1] } END { for (k in c) print k, c[k] }\n",
        "b 1\nc 1\nd 1\n"),
    run(fields, "{ c[$1]++; if ($2 > 5) { delete c[$1] } } END { for (k in c) print k, c[k] }\n",
        "b 1\nc 1\nd 1\n"),
    run(fields,
        "{ c[$1]++; if ($2 > 5) { delete c[$1]; m++ } } END { print m; for (k in c) print k, c[k] }\n",
        "2\nb 1\nc 1\nd 1\n"),
    run(fields,
        "{ c[$1]++; if ($2 > 5) delete c[$1]; else c[$1] += 10 } END { for (k in c) print k, c[k] }\n",
        "b 11\nc 11\nd 11\n"),
    !.

% an absent element prints "" after the delete
test(delete_then_read, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++; if ($2 > 5) delete c[$1]; print $1, c[$1] }\n",
        "a 1\nb \na \nc 1\nb 1\nd 1\n"),
    run(fields, "{ c[$1]++; if (c[$1] > 1) delete c[$1] } END { for (k in c) print k, c[k] }\n",
        "c 1\nd 1\n"),
    !.

test(other_keys_and_kinds, [condition(clang_available)]) :-
    run(fields, "{ s[$1] = $2 \"\"; if ($2 < 3) delete s[$1] } END { for (k in s) print k, s[k] }\n",
        "a 7\nd 4\n"),
    run(fields, "{ c[$1]++; if ($1 in c) n++ ; delete c[\"zz\"] } END { print n }\n", "6\n"),
    run(fields, "{ c[$1]++; if ($2 > 5) delete c[5] } END { for (k in c) print k, c[k] }\n",
        "a 2\nb 2\nc 1\nd 1\n"),
    run(fields, "{ c[$1] += $2; if ($2 > 8) delete c[$1] } END { print c[\"a\"], c[\"b\"] }\n",
        "10 2\n"),
    !.

:- end_tests(plawk_delete_cond).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_delete_cond', Dir),
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
    directory_file_path(Dir, 'dc_st_bin', Bin),
    write_prog(Src, 'dc_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'dc_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'dc', Prog),
    directory_file_path(Dir, 'dc_in.txt', In),
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

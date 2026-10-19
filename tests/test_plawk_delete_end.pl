:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `delete` in END, and `delete arr` of a whole array. These were parse errors:
%
%   * `END { delete c["a"] }` -- END had no delete statement; a literal key is
%     interned as the writes intern it (a variable key already lowered);
%   * `for (k in c) delete c[k]` -- a braceless delete was not a for-in body;
%   * `delete c` -- parses to that loop, so it lowers where the loop does (END)
%     and declines elsewhere.
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

:- begin_tests(plawk_delete_end).

test(end_delete_literal_key, [condition(clang_available)]) :-
    run(fields, "{ c[$1] = $2 } END { delete c[\"a\"]; for (k in c) print k, c[k] }\n",
        "b 2\nc 1\nd 4\n"),
    run(fields, "{ c[$1]++ } END { delete c[\"b\"]; for (k in c) print k, c[k] }\n",
        "a 2\nc 1\nd 1\n"),
    run(fields, "{ c[$1]++ } END { delete c[\"zz\"]; n = 0; for (k in c) n++; print n }\n",
        "4\n"),
    !.

test(forin_braceless_delete, [condition(clang_available)]) :-
    run(fields,
        "{ c[$1]++ } END { for (k in c) delete c[k]; for (k in c) print k, c[k]; print \"x\" }\n",
        "x\n"),
    !.

test(delete_whole_array, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++ } END { delete c; for (k in c) print k; print \"e\" }\n", "e\n"),
    run(fields, "{ c[$1]++ } END { n = 0; delete c; for (k in c) n++; print n }\n", "0\n"),
    run(fields, "{ s[$1] = $2 \"\" } END { delete s; for (k in s) print k; print \"e\" }\n",
        "e\n"),
    !.

% a for-in delete in a rule body does not lower yet: a clean decline, not a parse error
test(delete_whole_array_in_rule_declines) :-
    build_status("{ c[$1]++ } NR == 3 { delete c } END { for (k in c) print k, c[k] }\n", 3),
    build_status("{ c[$1]++; delete c } END { for (k in c) print k }\n", 3),
    !.

:- end_tests(plawk_delete_end).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_delete_end', Dir),
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
    directory_file_path(Dir, 'de_st_bin', Bin),
    write_prog(Src, 'de_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'de_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'de', Prog),
    directory_file_path(Dir, 'de_in.txt', In),
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

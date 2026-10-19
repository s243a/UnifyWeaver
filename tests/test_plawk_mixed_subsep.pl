:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Element writes at a LITERAL or SUBSEP key in the mixed chain -- beside scalar
% work or an `if`: `{ c[$1,$2]++; if ($2 > 5) n++ }`, `{ n++; c["x"]++ }`,
% `c[$1,"t"] += $2 * 0.5`, `c[$1,$2] = $2 * 2`. The mixed walker's element-write
% row took a field key only, so all of these declined; it now interns a literal /
% SUBSEP key through the same key builder the pure-assoc writes use.
%
% Keys print with SUBSEP ("\034"). for-in order is unspecified, so outputs are
% compared as sorted lines. gawk 5.1.0 (LC_ALL=C) gives every expected output.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(fields, "a 3\nb 10\na 7\nc 1\nb 2\nd 4\n").

:- begin_tests(plawk_mixed_subsep).

test(subsep_counter_beside_if, [condition(clang_available)]) :-
    run(fields, "{ c[$1,$2]++; if ($2 > 5) n++ } END { print n; for (k in c) print k, c[k] }\n",
        "2\na\x1c\3 1\na\x1c\7 1\nb\x1c\10 1\nb\x1c\2 1\nc\x1c\1 1\nd\x1c\4 1\n"),
    run(fields, "{ n++; if (($1,$2) in c) m++; c[$1,$2]++ } END { print n, m }\n", "6 \n"),
    run(fields, "{ c[$1,$2]++; if ($2 > 5) delete c[$1,$2] } END { for (k in c) print c[k] }\n",
        "1\n1\n1\n1\n"),
    !.

test(subsep_add_and_set, [condition(clang_available)]) :-
    run(fields, "{ c[$1,\"x\"] += $2; if ($2 > 5) n++ } END { print n; for (k in c) print c[k] }\n",
        "2\n10\n12\n1\n4\n"),
    run(fields, "{ c[$1,\"t\"] += $2 * 0.5; n++ } END { print n; for (k in c) print c[k] }\n",
        "6\n5\n6\n0.5\n2\n"),
    run(fields, "{ c[$1,$2] = $2 * 2; n++ } END { for (k in c) print c[k] }\n",
        "6\n14\n20\n4\n2\n8\n"),
    !.

test(literal_key_beside_scalar_work, [condition(clang_available)]) :-
    run(fields, "{ c[$1]++; if ($2 > 5) c[\"big\"]++ } END { print c[\"big\"], c[\"a\"] }\n",
        "2 2\n"),
    !.

% a literal key on a positional (split) table is unrepresentable: still declines
test(literal_key_on_split_table_declines) :-
    build_status("{ split($0, p, \" \"); p[\"x\"]++; n++; print p[\"x\"] }\n", 3),
    !.

:- end_tests(plawk_mixed_subsep).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_subsep', Dir),
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
    directory_file_path(Dir, 'ms_st_bin', Bin),
    write_prog(Src, 'ms_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'ms_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'ms', Prog),
    directory_file_path(Dir, 'ms_in.txt', In),
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

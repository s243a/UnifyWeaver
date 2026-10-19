:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% awk's uninitialised value, and END assignments -- docs/design/PLAWK_SCALAR_VALUE_MODEL.md.
%
%   * `m == ""` of an UNSET string / strnum scalar was false (@wam_sv_eq_id
%     compared id 0 with the interned "" atom), so the min idiom
%     `x = $2; if (m == "" || x < m) m = x` printed nothing: a wrong output.
%   * A straight-line END assignment declined (`END { avg = t / NR; print avg }`):
%     only an END block with a loop reached the shared sequence emitter. Now any
%     END block with a top-level assignment does, in a program without arrays, and
%     the END block joins the scalar planning so its targets get their kind.
%   * A copy `m = n` of a possibly-unset numeric n marked m assigned (printed 0
%     where awk prints ""); it now carries n's mark. And a name only READ by an
%     assignment (`m = n * 2`) lost its unset rendering: the trackability check
%     counted the read as an unmarked write.
%
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(fields, "a 3\nb 10\nc x\nd 9x\ne 5.5\na 7\n").
input(empty, "").

:- begin_tests(plawk_unset_end_assign).

test(unset_equals_empty_string, [condition(clang_available)]) :-
    run(fields, "{ x = $2; if (m == \"\" || x < m) m = x } END { print m }\n", "3\n"),
    run(fields, "{ x = $2; if (m == \"\") m = x; else if (x < m) m = x } END { print m }\n",
        "3\n"),
    run(fields, "{ x = $2; if (m == \"\") print \"empty\"; m = x }\n", "empty\n"),
    run(fields, "{ x = $2; if (m != \"\") print \"set\", m; m = x }\n",
        "set 3\nset 10\nset x\nset 9x\nset 5.5\n"),
    run(fields, "{ if (s == \"\") print \"e\"; s = $1 \"x\" }\n", "e\n"),
    !.

test(straight_line_end_assignments, [condition(clang_available)]) :-
    run(fields, "{ t += $2; n++ } END { avg = t / n; printf \"%.2f\\n\", avg }\n", "5.75\n"),
    run(fields, "{ t += $2 } END { t = t * 2; print t }\n", "69\n"),
    run(fields, "{ n++ } END { m = n - 10; if (m < 0) print \"neg\"; else print \"pos\" }\n",
        "neg\n"),
    run(fields, "{ s = $2 } END { n = length(s); print n, s }\n", "1 7\n"),
    run(fields, "{ s = $2 } END { m = s; print m }\n", "7\n"),
    !.

test(copies_and_reads_keep_unset, [condition(clang_available)]) :-
    run(fields, "{ u = v; print u \"|\" }\n", "|\n|\n|\n|\n|\n|\n"),
    run(empty, "{ n++ } END { m = n; print m \"|\" }\n", "|\n"),
    run(empty, "{ t += $2 } END { m = t; print m }\n", "\n"),
    run(empty, "{ n++; m = n * 2 } END { print n \"|\" }\n", "|\n"),
    run(empty, "{ t += $2; avg = t / NR } END { print t \"|\" }\n", "|\n"),
    !.

:- end_tests(plawk_unset_end_assign).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_unset_end_assign', Dir),
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
    directory_file_path(Dir, 'ue_st_bin', Bin),
    write_prog(Src, 'ue_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'ue_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'ue', Prog),
    directory_file_path(Dir, 'ue_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

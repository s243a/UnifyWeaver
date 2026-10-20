:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Building a string from numbers, and assignments in an END-only program.
%
% 1. A number as a part of a string concatenation -- `s = "a" n`, `s = "x" NR`,
%    `msg = "avg=" t / NR` -- declined in a rule body and in END alike: the concat
%    builder took text parts only (literals, fields, string scalars). A number part
%    is now its awk text: an integer through `%ld`, a double through
%    @wam_awk_num_text (integral as an integer, else %.6g), and a tracked number
%    that may still be unset as "" (`{ s = "<" n ">"; n = NR }`), as print renders it.
% 2. A strnum scalar (`last = $1`) as a concat part kept the name a plain counter
%    (the strnum read gate counted any read in a string RHS as unsupported), so
%    `s = last "!"` declined. A bare part reads it as its text now.
% 3. An END-only program with an assignment or a loop (`END { x = NR; print x }`)
%    declined: awk still reads the input, so it compiles as the program with a main
%    rule that does nothing.
%
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(rows, "a 3\nb 10\nc 2.5\nd 1e6\n").
input(empty, "").

:- begin_tests(plawk_concat_numbers).

test(number_parts_in_a_rule_body, [condition(clang_available)]) :-
    run(rows, "{ n++; s = \"a\" n; print s }\n", "a1\na2\na3\na4\n"),
    run(rows, "{ s = \"x\" NR; print s }\n", "x1\nx2\nx3\nx4\n"),
    run(rows, "{ s = $1 \":\" NR * 2; print s }\n", "a:2\nb:4\nc:6\nd:8\n"),
    run(rows, "{ t += $2; s = \"t=\" t; print s }\n",
        "t=3\nt=13\nt=15.5\nt=1.00002e+06\n"),
    run(rows, "{ x = $2 / 4; s = \"[\" x \"]\"; print s }\n",
        "[0.75]\n[2.5]\n[0.625]\n[250000]\n"),
    run(rows, "{ s = \"a\" (NR % 2); print s }\n", "a1\na0\na1\na0\n"),
    !.

% unset reads as "", then the number
test(unset_number_part_is_empty, [condition(clang_available)]) :-
    run(rows, "{ s = \"<\" n \">\"; print s; n = NR }\n", "<>\n<1>\n<2>\n<3>\n"),
    !.

test(number_parts_accumulate, [condition(clang_available)]) :-
    run(rows, "{ n++; s = s \",\" n } END { print s }\n", ",1,2,3,4\n"),
    !.

test(number_parts_in_end, [condition(clang_available)]) :-
    run(rows, "{ n++ } END { s = \"n=\" n; print s }\n", "n=4\n"),
    run(rows, "{ t += $2 } END { msg = \"avg=\" t / NR; print msg }\n", "avg=250004\n"),
    run(rows, "{ n++ } END { s = \"x\" NR; print s }\n", "x4\n"),
    !.

test(strnum_scalar_part, [condition(clang_available)]) :-
    run(rows, "{ last = $1; s = last \"!\"; print s }\n", "a!\nb!\nc!\nd!\n"),
    run(rows, "{ last = $1 } END { s = last \"!\"; print s }\n", "d!\n"),
    run(rows, "{ k = $2; s = k k; print s; if (k == \"10\") print \"ten\" }\n",
        "33\n1010\nten\n2.52.5\n1e61e6\n"),
    !.

test(end_only_assignments, [condition(clang_available)]) :-
    run(rows, "END { s = \"x\" NR; print s }\n", "x4\n"),
    run(rows, "END { x = NR; print x }\n", "4\n"),
    run(rows, "END { n = 2; n += 3; print n }\n", "5\n"),
    run(rows, "END { i = 0; while (i < 3) i++; print i }\n", "3\n"),
    run(rows, "BEGIN { x = 1 } END { x = x * 10; print x }\n", "10\n"),
    run(empty, "END { x = NR; print x }\n", "0\n"),
    !.

:- end_tests(plawk_concat_numbers).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_concat_numbers', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'cn_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'cn.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'cn_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, BuildStatus),
    assertion(BuildStatus == exit(0)),
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

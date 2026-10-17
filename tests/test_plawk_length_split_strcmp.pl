:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Three gaps that were parse errors or declines:
%
%   * `length(s)` of a variable -- print, arithmetic, assignment, END and a
%     comparison; a string or strnum slot is measured by its text (a numeric
%     slot still declines: its text is a number formatting);
%   * `split(s, a, /re/)` -- a regex literal as the separator; a one-character
%     regex that means more than itself (`/./`, `/ /`) is kept a regex;
%   * a string comparison inside `&&` / `||` (`if (s == "a" || s == "b")`, the
%     min idiom `if (m == "" || x < m)`).
%
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(fields, "a 3\nb 10\nc x\nd 9x\ne 5.5\nf 5.6\na 7\nh -1.5\n").
input(seps, "a.b c..d\nx y  z\nq \n").

:- begin_tests(plawk_length_split_strcmp).

test(length_of_variable, [condition(clang_available)]) :-
    run(fields, "{ s = $1 \"xy\"; print length(s) }\n", "3\n3\n3\n3\n3\n3\n3\n3\n"),
    run(fields, "{ s = $1 \"xy\"; n = length(s); print n }\n", "3\n3\n3\n3\n3\n3\n3\n3\n"),
    run(fields, "{ s = $0; print length(s) + 1 }\n", "4\n5\n4\n5\n6\n6\n4\n7\n"),
    run(fields, "{ s = $2 } END { print length(s) }\n", "4\n"),
    run(fields, "{ s = $1 $2; if (length(s) > 2) print s }\n",
        "b10\nd9x\ne5.5\nf5.6\nh-1.5\n"),
    !.

test(length_of_number_declines) :-
    build_status("{ n++; print length(n) }\n", 3),
    !.

test(split_regex_literal, [condition(clang_available)]) :-
    run(seps, "{ n = split($0, p, /[ ]+/); print n, p[1], p[2] }\n",
        "2 a.b c..d\n3 x y\n2 q \n"),
    run(seps, "{ n = split($0, p, /\\./); print n, p[1] }\n", "4 a\n1 x y  z\n1 q \n"),
    run(seps, "{ n = split($0, p, /./); print n }\n", "9\n7\n3\n"),
    run(seps, "{ n = split($0, p, / /); print n, p[2] }\n", "2 c..d\n4 y\n2 \n"),
    !.

test(string_comparison_in_boolean, [condition(clang_available)]) :-
    run(fields, "{ s = $1; if (s == \"a\" || s == \"b\") print }\n", "a 3\nb 10\na 7\n"),
    run(fields, "{ s = $1; if (s != \"a\" && NR > 2) print $1 }\n", "c\nd\ne\nf\nh\n"),
    run(fields, "{ x = $2; if (x == \"x\" || x > 5) print $1 }\n", "b\nc\nd\ne\nf\na\n"),
    run(fields, "{ x = $2; if (m == \"\" || x < m) m = x } END { print m }\n", "-1.5\n"),
    run(fields,
        "{ a[$1] = $2 } END { for (k in a) if (m == \"\" || a[k] < m) { m = a[k] }; print m }\n",
        "-1.5\n"),
    !.

:- end_tests(plawk_length_split_strcmp).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_length_split_strcmp', Dir),
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
    directory_file_path(Dir, 'ls_st_bin', Bin),
    write_prog(Src, 'ls_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'ls_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'ls', Prog),
    directory_file_path(Dir, 'ls_in.txt', In),
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

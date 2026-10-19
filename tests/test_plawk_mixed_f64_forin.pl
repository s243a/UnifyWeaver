:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A DOUBLE-valued table (`c[$1] += $2`) printed by key in the mixed chain -- a
% program whose rule also has an `if` (or a delete): `{ c[$1] += $2; if ($2 > 8)
% n++ } END { for (k in c) print k, c[k] }`. The keyed print read the cell with
% @wam_assoc_i64_get, which would print double bits as an integer, so the
% double-table IR check declined the program. The print now goes through the
% table's print helper (the double printer, nothing for an absent key).
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

input(fields, "a 3\nb 10.5\na 7\nc 1\nb 2\nd 4\n").

:- begin_tests(plawk_mixed_f64_forin).

test(end_forin_print, [condition(clang_available)]) :-
    run(fields, "{ c[$1] += $2; if ($2 > 8) n++ } END { for (k in c) print k, c[k] }\n",
        "a 10\nb 12.5\nc 1\nd 4\n"),
    run(fields, "{ c[$1] += $2; if ($2 > 8) delete c[$1] } END { for (k in c) print k, c[k] }\n",
        "a 10\nb 2\nc 1\nd 4\n"),
    run(fields,
        "{ c[$1] += $2; if ($2 > 8) n++ } END { print n; for (k in c) print k, c[k], n }\n",
        "1\na 10 1\nb 12.5 1\nc 1 1\nd 4 1\n"),
    run(fields, "{ c[$1] += $2 * 0.5; if ($2 > 8) n++ } END { for (k in c) print k, c[k] }\n",
        "a 5\nb 6.25\nc 0.5\nd 2\n"),
    !.

test(rule_body_keyed_print, [condition(clang_available)]) :-
    run(fields, "{ c[$1] += $2; k = $1; if ($2 > 8) n++; print k, c[k] }\n",
        "a 3\nb 10.5\na 10\nc 1\nb 12.5\nd 4\n"),
    !.

% a printf of the element still declines (no printf form of the double print)
test(printf_declines) :-
    build_status("{ c[$1] += $2; if ($2 > 8) n++ } END { for (k in c) printf \"%s %d\\n\", k, c[k] }\n", 3),
    !.

:- end_tests(plawk_mixed_f64_forin).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_f64_forin', Dir),
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
    directory_file_path(Dir, 'mf_st_bin', Bin),
    write_prog(Src, 'mf_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'mf_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'mf', Prog),
    directory_file_path(Dir, 'mf_in.txt', In),
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

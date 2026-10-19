:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Two text-element gaps:
%
%   * `x = a[k]` copied from a STRING-kind element (`a[k] = $2 ""`) declined: x
%     is now a string slot and the element's text is copied into it (a strnum
%     element's copy already worked);
%   * `length(a[k])` was a parse error: it measures a text element's text in
%     print, assignment, `+=`, arithmetic and an END for-in (a counter element
%     declines -- its text is a number formatting).
%
% Outputs are compared as sorted lines (for-in order is unspecified).
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(fields, "a 3\nb 10\nc x\nd 9x\ne 5.5\na 7\n").

:- begin_tests(plawk_elem_text_copy_length).

test(copy_string_element, [condition(clang_available)]) :-
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1]; print x \"|\" }\n",
        "3|\n10|\nx|\n9x|\n5.5|\n7|\n"),
    run(fields, "{ a[$1] = $2 \"\"; x = a[\"zz\"]; print x \"|\" }\n", "|\n|\n|\n|\n|\n|\n"),
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1] } END { print x }\n", "7\n"),
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1]; if (x == \"10\") print $1 }\n", "b\n"),
    % the copy owns its text: overwriting the element leaves x alone
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1]; a[$1] = \"q\"; print x, a[$1] }\n",
        "3 q\n10 q\nx q\n9x q\n5.5 q\n7 q\n"),
    !.

test(length_of_element, [condition(clang_available)]) :-
    run(fields, "{ a[$1] = $2 \"\"; print length(a[$1]) }\n", "1\n2\n1\n2\n3\n1\n"),
    run(fields, "{ a[$1] = $2; print length(a[$1]) }\n", "1\n2\n1\n2\n3\n1\n"),
    run(fields, "{ n++; a[$1] = $2 \"\"; print length(a[$1]) }\n", "1\n2\n1\n2\n3\n1\n"),
    run(fields, "{ a[$1] = $2 \"\"; n = length(a[$1]); print n }\n", "1\n2\n1\n2\n3\n1\n"),
    run(fields, "{ a[$1] = $2 \"\"; t += length(a[$1]) } END { print t }\n", "10\n"),
    run(fields, "{ a[$1] = $2; n = length(a[$1]) * 2; print n }\n", "2\n4\n2\n4\n6\n2\n"),
    run(fields, "{ a[$1] = $2 \"\" } END { for (k in a) print k, length(a[k]) }\n",
        "a 1\nb 2\nc 1\nd 2\ne 3\n"),
    !.

test(length_of_counter_element_declines) :-
    build_status("{ c[$1]++; print length(c[$1]) }\n", 3),
    build_status("{ c[$1]++; n = length(c[$1]); print n }\n", 3),
    !.

% beside scalar work, printf of string-element arithmetic reads it as a number
test(printf_of_string_element_arithmetic, [condition(clang_available)]) :-
    run(fields, "{ n++; a[$1] = $2 \"\"; printf \"%d\\n\", a[$1] * 2 }\n",
        "6\n20\n0\n18\n11\n14\n"),
    !.

:- end_tests(plawk_elem_text_copy_length).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_elem_text_copy_length', Dir),
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
    directory_file_path(Dir, 'el_st_bin', Bin),
    write_prog(Src, 'el_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'el_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'el', Prog),
    directory_file_path(Dir, 'el_in.txt', In),
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

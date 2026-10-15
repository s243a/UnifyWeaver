:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Text values read in the array-rule (assoc) driver's prints.
%
% A string element (arrays PR 5a, `a[k] = $2`) and a scalar set from a field
% (`x = $2`, the key-scalar idiom) hold an interned atom id. The driver's print
% arithmetic read an element as an i64 count, and printed or computed with such
% a scalar as the id itself: `{ a[$1] = $2; print a[$1] * 2 }` printed 42 for 3,
% and `{ x = $2; c[$1]++; print x }` printed 21. A text value now prints as its
% text and reads, in arithmetic, as the number its text starts with (awk's
% strtod), as in gawk. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 5\na 2\nc 10\nb 4\nd 1.5\ne x7\n").

:- begin_tests(plawk_assoc_text_reads).

test(string_element_in_arithmetic, [condition(clang_available)]) :-
    run("{ a[$1] = $2; print a[$1] * 2 }\n", "6\n10\n4\n20\n8\n3\n0\n"),
    run("{ a[$1] = $2; print $1, a[$1] - 1 }\n",
        "a 2\nb 4\na 1\nc 9\nb 3\nd 0.5\ne -1\n"),
    run("{ a[$1] = $1 \"x\"; print a[$1] + 1 }\n", "1\n1\n1\n1\n1\n1\n1\n"),
    !.

% A count table is unchanged.
test(count_element_in_arithmetic, [condition(clang_available)]) :-
    run("{ c[$1]++; print c[$1] * 2 }\n", "2\n2\n4\n2\n4\n2\n2\n"),
    !.

test(field_set_scalar_prints_its_text, [condition(clang_available)]) :-
    run("{ x = $2; c[$1]++; print x, c[$1] }\n",
        "3 1\n5 1\n2 2\n10 1\n4 2\n1.5 1\nx7 1\n"),
    run("{ k = $1; c[k]++; print k, c[k] }\n",
        "a 1\nb 1\na 2\nc 1\nb 2\nd 1\ne 1\n"),
    run("$1 == \"a\" { x = $2; c[$1]++; print x }\n", "3\n2\n"),
    !.

test(field_set_scalar_in_arithmetic, [condition(clang_available)]) :-
    run("{ x = $2; c[$1]++; print x * 2, c[$1] }\n",
        "6 1\n10 1\n4 2\n20 1\n8 2\n3 1\n0 1\n"),
    !.

% One scalar both set from a field and accumulated: the array-rule driver
% steps aside (its global would hold an id and a number), and the program is
% compiled elsewhere, correctly.
test(set_and_accumulated_scalar, [condition(clang_available)]) :-
    run("{ x = $1; x += 1; c[$1]++; print x }\n", "1\n1\n1\n1\n1\n1\n1\n"),
    !.

:- end_tests(plawk_assoc_text_reads).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_text_reads', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tr_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tr', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(BuildStatus)),
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

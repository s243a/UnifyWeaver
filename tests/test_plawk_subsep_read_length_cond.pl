:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Two element reads that declined or did not parse:
%
% 1. `print c[$1,$2]` beside scalar work (the mixed chain), and an all-literal
%    SUBSEP key in END (`END { print c["a","x"] }`). The pure-array chain printed
%    the first; the mixed chain's print admission had no SUBSEP row, and the END
%    print emitters take a string key only. A mixed print now interns the key as
%    the writes do (plawk_subsep_key_n_ir/7); an END literal SUBSEP key is the
%    string key it denotes, joined with the program's SUBSEP.
% 2. `length(...)` of an element or a field in an `if` condition
%    (`if (length(a[$1]) > 2)`, `if (length($2) > 2)`) was a parse error, though
%    `n = length(a[$1])` parsed. The condition pre-pass computes the length first:
%    the element's text (an absent element is "", length 0) or the field's bytes.
%
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(pairs, "a x 3\nb y 10\na x 2\nc z 5\n").
input(words, "a xyz\nb q\nc hello\n\n").

:- begin_tests(plawk_subsep_read_length_cond).

test(mixed_chain_subsep_print, [condition(clang_available)]) :-
    run(pairs, "{ c[$1,$2]++; n++; print n, c[$1,$2] }\n", "1 1\n2 1\n3 2\n4 1\n"),
    run(pairs, "{ c[$1,$2] += $3 / 2; n++; print n, c[$1,$2] }\n",
        "1 1.5\n2 5\n3 2.5\n4 2.5\n"),
    run(pairs, "{ c[$1,\"k\"]++; if (NR > 1) print c[$1,\"k\"], c[\"a\",\"k\"] }\n",
        "1 1\n2 2\n1 2\n"),
    !.

test(end_literal_subsep_print, [condition(clang_available)]) :-
    % an absent element prints "" (the comma form: an element inside a concat in an
    % END print still declines, with any key)
    run(pairs, "{ c[$1,$2]++ } END { print c[\"a\",\"x\"], c[\"b\",\"y\"], c[\"q\",\"q\"], \"|\" }\n",
        "2 1  |\n"),
    run(pairs, "BEGIN { SUBSEP = \":\" } { c[$1,$2]++; n++ } END { print n, c[\"a\",\"x\"] }\n",
        "4 2\n"),
    run(pairs, "{ c[$1,1]++ } END { print c[\"a\",1] }\n", "2\n"),
    !.

test(length_of_field_in_condition, [condition(clang_available)]) :-
    run(words, "{ if (length($2) > 2) print $2; else print \"short\" }\n",
        "xyz\nshort\nhello\nshort\n"),
    run(words, "{ n++; if (length($2) >= 3) print n }\n", "1\n3\n"),
    !.

test(length_of_element_in_condition, [condition(clang_available)]) :-
    run(words, "{ a[$1] = $2; if (length(a[$1]) > 1) print $1 }\n", "a\nc\n"),
    run(words, "{ a[$1] = $2 \"!\"; n++; if (length(a[$1]) == 4) print n, $1 }\n",
        "1 a\n"),
    run(words, "{ a[$1] = $2 } END { for (k in a) if (length(a[k]) > 2) n++; print n }\n",
        "2\n"),
    !.

:- end_tests(plawk_subsep_read_length_cond).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_subsep_read_length_cond', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'sl_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'sl.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'sl_in.txt', In),
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

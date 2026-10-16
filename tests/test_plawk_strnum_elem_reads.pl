:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Numeric reads of strnum array elements -- docs/design/PLAWK_ARRAY_VALUE_MODEL.md,
% plan item 5(d).
%
% A strnum element (`a[$1] = $2`, a row capture, a split piece) is text that
% reads, in arithmetic, as the number it starts with (awk's strtod), and compares
% numerically only when it looks numeric -- otherwise as a string. These reads
% declined: `t += a[k]`, `a[$1] > 5`, `print p[2] * 2` and an END fold over the
% table all went to the WAM fallback. gawk 5.1.0 (LC_ALL=C) gives every expected
% output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% "x" and "9x" do not look numeric; "1e2" does.
input("a 3\nb 10\na 2.5\nc x\nb 7\nd 1e2\nh 9x\n").

:- begin_tests(plawk_strnum_elem_reads).

test(rule_body_arithmetic, [condition(clang_available)]) :-
    run("{ a[$1] = $2; t += a[$1] } END { print t }\n", "131.5\n"),
    run("{ a[$1] = $2; t = t + a[$1] * 2 } END { print t }\n", "263\n"),
    !.

% The END fold `for (k in a) t += a[k]` reads each element's text; it summed the
% ids before the fold declined str tables. A string-kind table sums the same way.
test(end_folds, [condition(clang_available)]) :-
    run("{ a[$1] = $2 } END { for (k in a) t += a[k]; print t }\n", "118.5\n"),
    run("{ a[$1] = $2 } END { for (k in a) { s = s + a[k] }; print s }\n", "118.5\n"),
    run("{ a[$1] = $0 } END { for (k in a) t += a[k]; print t }\n", "0\n"),
    run("{ a[$1] = $2 \"\" } END { for (k in a) t += a[k]; print t }\n", "118.5\n"),
    !.

% An absent element reads 0 (its id is 0, which is no text).
test(end_literal_lookups, [condition(clang_available)]) :-
    run("{ a[$1] = $2 } END { print a[\"zz\"] + 1, a[\"c\"] * 3, a[\"d\"] / 4 }\n",
        "1 0 25\n"),
    run("{ c[$1]++; print c[\"a\"] * 10 }\n", "10\n10\n20\n20\n20\n20\n20\n"),
    !.

% "x" > 5 and "9x" > 5 compare as strings (true); "2.5" > 5 numerically (false).
test(rule_body_conditions, [condition(clang_available)]) :-
    run("{ a[$1] = $2; if (a[$1] > 5) print $1, \"big\"; else print $1, \"small\" }\n",
        "a small\nb big\na small\nc big\nb big\nd big\nh big\n"),
    run("{ a[$1] = $2; if (a[$1] == \"x\") print \"X\", $1 }\n", "X c\n"),
    run("{ a[$1] = $2; if (a[$1] < \"5\") print $1 }\n", "a\nb\na\nd\n"),
    run("{ a[$1] = $2; if (a[$1] > a[\"b\"]) print $1 }\n", "a\nc\nd\nh\n"),
    run("{ a[$1] = $2; n++; if (a[$1] > n) print $1, n }\n",
        "a 1\nb 2\nc 4\nb 5\nd 6\nh 7\n"),
    !.

test(end_forin_condition, [condition(clang_available)]) :-
    run("{ a[$1] = $2 } END { for (k in a) if (a[k] > 9) n++; print n + 0 }\n", "3\n"),
    !.

% A split piece is keyed by its raw position: a literal or a counter.
test(split_pieces, [condition(clang_available)]) :-
    run("{ n = split($0, p, \" \"); print p[2] * 2 }\n", "6\n20\n5\n0\n14\n200\n18\n"),
    run("{ split($0, p, \" \"); i = 2; print p[i] * 2 }\n", "6\n20\n5\n0\n14\n200\n18\n"),
    run("{ split($0, p, \" \"); if (p[2] > 5) print p[1] }\n", "b\nc\nb\nd\nh\n"),
    run("{ split($0, p, \" \"); t += p[2] } END { print t }\n", "131.5\n"),
    run("{ split($0, p, \" \"); print p[2] + 1, p[\"x\"] + 1 }\n",
        "4 1\n11 1\n3.5 1\n1 1\n8 1\n101 1\n10 1\n"),
    !.

% A string-kind element compares as a string even against a number, which the
% strnum read does not model: still declined.
test(string_kind_condition_declines) :-
    build_status_is("{ a[$1] = $2 \"\"; if (a[$1] > 5) print $1 }\n", 3),
    !.

:- end_tests(plawk_strnum_elem_reads).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strnum_elem_reads', Dir),
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

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'se_st_bin', Bin),
    write_prog(Src, 'se_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'se_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'se', Prog),
    directory_file_path(Dir, 'se_in.txt', In),
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

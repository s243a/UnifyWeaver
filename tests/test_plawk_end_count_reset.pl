:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The END cardinality idiom keeps its reset, and an unassigned accumulator
% prints nothing -- docs/design/PLAWK_SCALAR_VALUE_MODEL.md section 3.
%
% The parser lowered `END { n = 0; for (k in a) n++; print n }` to the for-in
% accumulation AST and DROPPED `n = 0`. The array-only driver starts its
% accumulator at 0, so it never mattered there; beside scalar rules it did: the
% rule-time value of n leaked in (`n = 7` in a rule printed 10, gawk 3), and an
% empty array printed nothing (gawk 0). The reset now stays in the AST. The
% array-only driver also told the two forms apart wrongly the other way: with
% NO reset, `for (k in c) t += c[k]; print t` on an empty array printed 0 where
% t is unset and gawk prints an empty line. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 5\na 2\nc 1\nb 4\na 7\n").

:- begin_tests(plawk_end_count_reset).

% beside scalar rules (the mixed walker): the reset runs
test(reset_beside_scalar_rules, [condition(clang_available)]) :-
    run("{ n = 7; c[$1]++ } END { n = 0; for (q in c) n++; print n }\n", "3\n"),
    run("{ k = $1; c[k]++ } END { n = 0; for (q in c) n++; print n }\n", "3\n"),
    run("{ k = $1; c[k]++; delete c[k] } END { n = 0; for (q in c) n++; print n }\n",
        "0\n"),
    !.

% array-only: a reset accumulator is assigned even when the array is empty
test(reset_array_only, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { n = 0; for (k in c) n++; print n }\n", "3\n"),
    run("{ c[$1]++; delete c[$1] } END { n = 0; for (k in c) n++; print n }\n", "0\n"),
    !.

% no reset: unset until the first iteration, so an empty array prints nothing
test(no_reset_empty_prints_nothing, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { for (k in c) t += c[k]; print t }\n", "6\n"),
    run("{ c[$1]++; delete c[$1] } END { for (k in c) t += c[k]; print t }\n", "\n"),
    run("{ c[$1]++; delete c[$1] } END { for (k in c) t += c[k]; print \"total\", t, \"x\" }\n",
        "total  x\n"),
    run("{ c[$1] += $2 * 0.5 } END { for (k in c) t += c[k]; print t }\n", "11\n"),
    run("{ c[$1] += $2 * 0.5; delete c[$1] } END { for (k in c) t += c[k]; print t }\n",
        "\n"),
    !.

:- end_tests(plawk_end_count_reset).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_count_reset', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'ec_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ec', Prog0),
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
    process_wait(Pid, exit(0)),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

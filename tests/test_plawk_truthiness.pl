:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Truth tests: a bare operand as a condition -- `if (x)`, `if (!seen)`,
% `while (i)`, `if (c[$1])`, `if ($1)`, and the patterns `NF { }`, `!NF { }`,
% `$2 { }`. All were parse errors. awk's truth value: a number is true when
% nonzero, a string when non-empty, and a strnum (a field, input-derived text)
% numerically when its text looks numeric ("0", "0.0" false) and by emptiness
% otherwise ("x", "0x" true) -- @wam_strnum_truth. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 0\nb 1\n\nc x\nd 0.0\ne 0x\nf -2\ng\n").

:- begin_tests(plawk_truthiness).

test(parses) :-
    plawk_parse_string("NF { print }\n",
        program([], [rule(truth_pat(special('NF')), _)], [])),
    plawk_parse_string("!$2 { print }\n",
        program([], [rule(not_pat(truth_pat(field(2))), _)], [])),
    plawk_parse_string("{ if (!seen) print }\n",
        program([], [rule(always, [if(scalar_if(not(truth(var(seen)))), _, [])])], [])),
    % a comparison still parses as one
    plawk_parse_string("$2 > 1 { print }\n",
        program([], [rule(field_cmp(2, gt, 1), _)], [])),
    !.

test(patterns, [condition(clang_available)]) :-
    run("NF { print }\n", "a 0\nb 1\nc x\nd 0.0\ne 0x\nf -2\ng\n"),
    run("!NF { print \"blank\" }\n", "blank\n"),
    run("$2 { print $1 }\n", "b\nc\ne\nf\n"),
    run("$1 && $2 { print $1 }\n", "b\nc\ne\nf\n"),
    !.

test(field_and_strnum_conditions, [condition(clang_available)]) :-
    run("{ if ($2) print $1; else print \"no\", $1 }\n",
        "no a\nb\nno \nc\nno d\ne\nf\nno g\n"),
    run("{ x = $2; if (x) print $1 }\n", "b\nc\ne\nf\n"),
    run("{ a[$1] = $2; if (a[$1]) print $1 }\n", "b\nc\ne\nf\n"),
    !.

test(number_and_string_conditions, [condition(clang_available)]) :-
    run("{ n++; if (n) print n }\n", "1\n2\n3\n4\n5\n6\n7\n8\n"),
    run("{ if (!seen) print \"first\"; seen = 1 }\n", "first\n"),
    run("{ s = $2 \"y\"; if (s) print $1 }\n", "a\nb\n\nc\nd\ne\nf\ng\n"),
    run("{ t += $2; if (t) print $1, t }\n", "b 1\n 1\nc 1\nd 1\ne 1\nf -1\ng -1\n"),
    run("{ c[$1]++ } END { if (c[\"a\"]) print \"has a\"; if (!c[\"zz\"]) print \"no zz\" }\n",
        "has a\nno zz\n"),
    !.

test(loop_conditions, [condition(clang_available)]) :-
    run("{ i = 3; while (i) { i--; c++ } } END { print c }\n", "24\n"),
    !.

:- end_tests(plawk_truthiness).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_truthiness', Dir),
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

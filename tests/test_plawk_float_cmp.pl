:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Comparisons against a FLOAT literal: a field (`$2 > 5.5`, a pattern or an `if`
% guard, either operand order) and an array element (`if (a[$1] > 5.5)`). Both
% were parse errors. A field is a strnum, so it compares numerically when its
% text looks numeric and as a string otherwise (@wam_atom_field_strnum_cmp_double:
% "x" > 5.5 and "9x" > 5.5 are true, as in gawk); an element compares by its
% table's kind (a counter or double table numerically, a strnum table like a
% field). gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 10\nc x\nd 9x\ne 5.5\nf 5.6\na 7\nh -1.5\n").

:- begin_tests(plawk_float_cmp).

test(parses) :-
    plawk_parse_string("$2 > 5.5 { print }\n",
        program([], [rule(field_fcmp(2, gt, float_const(55, 10)), _)], [])),
    plawk_parse_string("5.5 < $2 { print }\n",
        program([], [rule(field_fcmp(2, gt, float_const(55, 10)), _)], [])),
    plawk_parse_string("$2 > 5 { print }\n",
        program([], [rule(field_cmp(2, gt, 5), _)], [])),
    plawk_parse_string("{ if (a[$1] > -1.25) print }\n",
        program([], [rule(always, [if(scalar_if(cmp(assoc(var(a), field(1)), gt,
            float_const(-125, 100))), _, [])])], [])),
    !.

test(field_patterns, [condition(clang_available)]) :-
    run("$2 == 5.5 { print $1 }\n", "e\n"),
    run("$2 > 5.5 { print $1 }\n", "b\nc\nd\nf\na\n"),
    run("5.5 < $2 { print $1 }\n", "b\nc\nd\nf\na\n"),
    run("$2 <= -1.25 { print $1 }\n", "h\n"),
    run("$2 < 5.5 { n++ } $2 >= 5.5 { m++ } END { print n, m }\n", "2 6\n"),
    !.

test(field_if_guard, [condition(clang_available)]) :-
    run("{ if ($2 > 5.5) print $1; else print \"no\", $1 }\n",
        "no a\nb\nc\nd\nno e\nf\na\nno h\n"),
    !.

test(element_guards, [condition(clang_available)]) :-
    run("{ a[$1] = $2; if (a[$1] > 5.5) print $1 }\n", "b\nc\nd\nf\na\n"),
    run("{ c[$1]++; if (c[$1] > 1.5) print $1 }\n", "a\n"),
    run("{ s[$1] += $2; if (s[$1] >= 5.5) print $1, s[$1] }\n",
        "b 10\nd 9\ne 5.5\nf 5.6\na 10\n"),
    !.

:- end_tests(plawk_float_cmp).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_float_cmp', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'fc_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'fc', Prog0),
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

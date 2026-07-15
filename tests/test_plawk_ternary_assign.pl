:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk ternary in a scalar assignment RHS: `x = COND ? A : B` (the assignment
% mirror of the print/printf ternary). COND is a numeric comparison and both
% branches are numeric (fields, NR/NF, int literals, i64 arithmetic); the
% operation lowers through the shared i64 ternary select. Scalar-var operands
% and string branches remain follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_ternary_assign).

% --- parsing ----------------------------------------------------------------

test(assign_ternary_parses) :-
    plawk_parse_string("{ x = $1 > $2 ? $1 : $2 }\n",
        program([], [rule(always,
            [set(var(x), ternary(cmp(field(1), gt, field(2)), field(1), field(2)))])], [])),
    !.

% a plain arithmetic assignment is unchanged (no cmp-then-?).
test(assign_arith_unchanged) :-
    plawk_parse_string("{ x = $1 + $2 }\n",
        program([], [rule(always, [set(var(x), add_i64(field(1), field(2)))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% assign the max to a scalar; END reads the last one.
test(assign_max, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mx', "{ big = $1 > $2 ? $1 : $2 }\nEND { print big }\n",
        "3 7\n9 2\n", Out, St),
    assertion(St == 0), assertion(Out == "9\n"), !.

% clamp-to-zero, printed each record.
test(assign_clamp_body, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'cl', "{ c = $1 > 0 ? $1 : 0; print c }\n",
        "5\n-3\n0\n", Out, St),
    assertion(St == 0), assertion(Out == "5\n0\n0\n"), !.

% NR in the condition and a branch, read at END.
test(assign_nr, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nr', "{ last = NR > 1 ? NR : 0 }\nEND { print last }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n"), !.

% assigned scalar printed directly per record.
test(assign_body_print, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'bp', "{ m = $1 > $2 ? $1 : $2; print m }\n",
        "3 7\n8 2\n", Out, St),
    assertion(St == 0), assertion(Out == "7\n8\n"), !.

% equality condition selecting between two literals.
test(assign_eq, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'eq', "{ f = $1 == 0 ? 100 : 200; print f }\n",
        "0\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "100\n200\n"), !.

:- end_tests(plawk_ternary_assign).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_assign', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

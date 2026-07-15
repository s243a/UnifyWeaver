:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk ternary `COND ? A : B` (AWK `?:`) in print / printf argument position.
% COND is a numeric comparison `L <op> R`; both branches are numeric (fields,
% NR/NF, integer literals, i64 arithmetic). Lowered to an LLVM `select` (both
% branches evaluated -- no side effects in an i64 expression -- so it composes
% straight-line anywhere an i64 value is used). Assignment-context ternary
% (scalar-var operands) is a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_ternary).

% --- parsing ----------------------------------------------------------------

test(ternary_parses) :-
    plawk_parse_string("{ print $1 > $2 ? $1 : $2 }\n",
        program([], [rule(always,
            [print([ternary(cmp(field(1), gt, field(2)), field(1), field(2))])])], [])),
    !.

test(ternary_int_literal_operands_parse) :-
    plawk_parse_string("{ print $1 > 0 ? $1 : 0 }\n",
        program([], [rule(always,
            [print([ternary(cmp(field(1), gt, int(0)), field(1), int(0))])])], [])),
    !.

% --- runtime (print) --------------------------------------------------------

% max($1, $2)
test(ternary_max, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'max', "{ print $1 > $2 ? $1 : $2 }\n", "3 7\n9 2\n", Out, St),
    assertion(St == 0), assertion(Out == "7\n9\n"), !.

% clamp negatives to 0 (int-literal branch)
test(ternary_clamp, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'clamp', "{ print $1 > 0 ? $1 : 0 }\n", "5\n-3\n0\n", Out, St),
    assertion(St == 0), assertion(Out == "5\n0\n0\n"), !.

% NR in the condition and a branch
test(ternary_nr, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nr', "{ print NR > 1 ? NR : 0 }\n", "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "0\n2\n3\n"), !.

% arithmetic in a branch
test(ternary_arith_branch, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ar', "{ print $1 > $2 ? $1 + 100 : $2 + 100 }\n", "3 7\n", Out, St),
    assertion(St == 0), assertion(Out == "107\n"), !.

% equality condition
test(ternary_eq, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'eq', "{ print $1 == 0 ? 111 : 222 }\n", "0\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "111\n222\n"), !.

% under a rule guard
test(ternary_guarded, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gd', "$1 > 0 { print $1 > 5 ? 1 : 0 }\n", "3\n9\n-1\n", Out, St),
    assertion(St == 0), assertion(Out == "0\n1\n"), !.

% --- runtime (printf) -------------------------------------------------------

test(ternary_printf_arg, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'pf', "{ printf \"max=%d\\n\", $1 > $2 ? $1 : $2 }\n", "3 7\n", Out, St),
    assertion(St == 0), assertion(Out == "max=7\n"), !.

:- end_tests(plawk_ternary).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary', Dir),
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

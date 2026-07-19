:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% AWK numeric builtins that map to libm and return a double: sqrt, sin, cos,
% exp, log (unary) and atan2 (binary). Each parses to math_call(Fn, Args) and is
% emitted through the double expression path (plawk_f64_expr_ir): the argument
% is evaluated in f64 (integer leaves promote by sitofp, so sqrt($1) works), the
% libm function is called, and the result prints via the %g path -- an integer-
% valued result like sqrt(16) prints "4", an irrational one "1.41421".
%
% `log` is the natural logarithm (matching awk). A name only becomes a builtin
% when directly followed by `(`, so a variable like `sine` stays an identifier.
%
% Supported surface mirrors exponentiation: math builtins as a value in print /
% printf argument positions, including inside larger arithmetic. Guards and
% double-scalar-reread contexts are pre-existing limitations (see
% test_plawk_pow.pl), not specific to these builtins.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_math).

test(sqrt_integer, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'sq', "{ print sqrt(16) }\n", "x\n", Out),
    assertion(Out == "4\n"), !.

test(sqrt_irrational, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'sq2', "{ print sqrt(2) }\n", "x\n", Out),
    assertion(Out == "1.41421\n"), !.

test(exp_zero, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ex', "{ print exp(0) }\n", "x\n", Out),
    assertion(Out == "1\n"), !.

% log is the natural logarithm: log(exp(1)) == 1.
test(log_natural, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'lg', "{ print log(exp(1)) }\n", "x\n", Out),
    assertion(Out == "1\n"), !.

test(sin_zero, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'si', "{ print sin(0) }\n", "x\n", Out),
    assertion(Out == "0\n"), !.

test(cos_zero, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'co', "{ print cos(0) }\n", "x\n", Out),
    assertion(Out == "1\n"), !.

test(atan2_binary, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'at', "{ print atan2(0, 1) }\n", "x\n", Out),
    assertion(Out == "0\n"), !.

% A field argument is coerced to f64: sqrt of $1=9 is 3.
test(field_argument, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'fa', "{ print sqrt($1) }\n", "9\n", Out),
    assertion(Out == "3\n"), !.

% An arithmetic-expression argument: sqrt($1 * 4) with $1=9 is 6.
test(expression_argument, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ea', "{ print sqrt($1 * 4) }\n", "9\n", Out),
    assertion(Out == "6\n"), !.

% A math call inside a larger arithmetic expression: 2 * sqrt(9) == 6.
test(in_arithmetic, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ia', "{ print 2 * sqrt(9) }\n", "x\n", Out),
    assertion(Out == "6\n"), !.

% A pow expression as the argument: sqrt(2^4) == sqrt(16) == 4.
test(argument_is_pow, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ap', "{ print sqrt(2^4) }\n", "x\n", Out),
    assertion(Out == "4\n"), !.

% printf with a precision on a math result.
test(printf_precision, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'pf', "{ printf \"%.3f\\n\", sqrt(2) }\n", "x\n", Out),
    assertion(Out == "1.414\n"), !.

% Name boundary: `sine` is an ordinary identifier, not a `sin` call, because it
% is not directly followed by `(`.
test(name_boundary, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'nb', "{ sine = 5; print sine }\n", "x\n", Out),
    assertion(Out == "5\n"), !.

:- end_tests(plawk_math).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_math', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

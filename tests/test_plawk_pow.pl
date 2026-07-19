:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Exponentiation `^` / `**` (awk). Both spellings are the same operator; it
% binds tighter than `* / %` and is right-associative (2^3^2 == 2^(3^2)). As in
% awk it is always computed in floating point (via libm pow), so the result
% keeps its fraction and prints through the double %g path -- an integer-valued
% result like 2^10 prints as "1024", a fractional one as "1.41421".
%
% Supported surface: exponentiation as a value in print / printf(%g/%e/%f)
% argument positions, including inside larger arithmetic and juxtaposition
% concat. Operands may be integers, fields, or floats.
%
% Follow-ons (each a pre-existing limitation the operator inherits, not a
% regression it introduces): pow in an if/while guard (guards take no
% arithmetic operand at all -- `$1*2 > 5` is rejected the same way); pow fed to
% a %d/%i conversion (no double->i64 truncation); pow assigned to a scalar then
% reread (same as `x = 2.5; print x`); a unary-minus exponent (`2^-1`).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_pow).

% 2^10 = 1024, printed as an integer-valued double ("1024", not "1024.0").
test(pow_basic, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pb', "{ print 2^10 }\n", "x\n", Out),
    assertion(Out == "1024\n"), !.

% `**` is a synonym for `^`.
test(pow_starstar, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ps', "{ print 2**10 }\n", "x\n", Out),
    assertion(Out == "1024\n"), !.

% Right-associative: 2^3^2 == 2^(3^2) == 2^9 == 512.
test(pow_right_assoc, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pr', "{ print 2^3^2 }\n", "x\n", Out),
    assertion(Out == "512\n"), !.

% Binds tighter than `*`: 2*3^2 == 2*(3^2) == 2*9 == 18.
test(pow_precedence, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pp', "{ print 2*3^2 }\n", "x\n", Out),
    assertion(Out == "18\n"), !.

% A float base: 2.5^2 == 6.25.
test(pow_float_base, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pf', "{ print 2.5^2 }\n", "x\n", Out),
    assertion(Out == "6.25\n"), !.

% A fractional exponent is a root: 4^0.5 == 2.
test(pow_fractional_exponent, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pe', "{ print 4^0.5 }\n", "x\n", Out),
    assertion(Out == "2\n"), !.

% A field base: $1^2 with $1=3 is 9.
test(pow_field_base, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pd', "{ print $1^2 }\n", "3\n", Out),
    assertion(Out == "9\n"), !.

% Inside a larger arithmetic expression: $1^2 + 1 with $1=3 is 10.
test(pow_in_arithmetic, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pa', "{ print $1^2 + 1 }\n", "3\n", Out),
    assertion(Out == "10\n"), !.

% Subtraction with a pow term: 10 - 2^3 == 2.
test(pow_mixed_subtract, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pm', "{ print 10 - 2^3 }\n", "x\n", Out),
    assertion(Out == "2\n"), !.

% In a juxtaposition concat: "r=" 2^3 -> "r=8".
test(pow_concat, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pc', "{ print \"r=\" 2^3 }\n", "x\n", Out),
    assertion(Out == "r=8\n"), !.

% printf %g of a pow value.
test(pow_printf_g, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pg', "{ printf \"%g\\n\", 2^10 }\n", "x\n", Out),
    assertion(Out == "1024\n"), !.

:- end_tests(plawk_pow).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_pow', Dir),
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

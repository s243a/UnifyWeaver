:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% AWK PRNG builtins rand() and srand(). rand() returns a double in [0, 1) drawn
% from libm drand48 -- a 0-argument math builtin lowered through the double
% expression path (math_call(rand, []) -> call double @drand48()). srand(N)
% seeds the generator with integer N (reproducible), srand() seeds from the wall
% clock (time); both are rule-body statements that emit call void @srand48(...).
%
% Because drand48 is POSIX-specified, srand(N) makes rand() bit-reproducible:
% srand(1) then rand() is 0.0416303, srand(2) is 0.912433 -- the assertions below
% check those to prove the seed takes effect. Without srand a program uses the
% fixed default seed, so it is still reproducible (just not chosen).
%
% Follow-ons: srand in a BEGIN block (BEGIN hosts no non-print action yet); a
% variable / expression seed (srand($1)); srand's awk return value (the previous
% seed). rand() composes in arithmetic (2 * rand(), sqrt(rand()*0)) like any
% double expression.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_rand).

% rand() is a double in [0, 1).
test(rand_in_range, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'rr', "{ print rand() }\n", "x\n", Out),
    parse_num(Out, N),
    assertion((N >= 0.0, N < 1.0)), !.

% srand(N) seeds deterministically: srand(1) then rand() is the POSIX drand48
% first value 0.0416303.
test(srand_seed_reproducible, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'sr1', "{ srand(1); print rand() }\n", "x\n", Out),
    parse_num(Out, N),
    assertion(abs(N - 0.0416303) < 1.0e-5), !.

% A different seed yields a different sequence: srand(2) -> 0.912433.
test(srand_seed_differs, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'sr2', "{ srand(2); print rand() }\n", "x\n", Out),
    parse_num(Out, N),
    assertion(abs(N - 0.912433) < 1.0e-5), !.

% srand() (no arg) seeds from time; the draw is still a valid [0, 1) double.
test(srand_time_in_range, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'st', "{ srand(); print rand() }\n", "x\n", Out),
    parse_num(Out, N),
    assertion((N >= 0.0, N < 1.0)), !.

% rand() composes in arithmetic: rand() * 0 is exactly 0.
test(rand_in_arithmetic, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ra', "{ print rand() * 0 }\n", "x\n", Out),
    assertion(Out == "0\n"), !.

% rand() nested inside a math builtin: sqrt(rand() * 0) == 0.
test(rand_nested, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'rn', "{ print sqrt(rand() * 0) }\n", "x\n", Out),
    assertion(Out == "0\n"), !.

% Identifier boundary: `srander` and `randy` are ordinary identifiers, not the
% builtins, because they are not directly followed by `(`.
test(srand_name_boundary, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'sb', "{ srander = 5; print srander }\n", "x\n", Out),
    assertion(Out == "5\n"), !.

test(rand_name_boundary, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'rb', "{ randy = 3; print randy }\n", "x\n", Out),
    assertion(Out == "3\n"), !.

:- end_tests(plawk_rand).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rand', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Parse the single numeric line of output into a float.
parse_num(Out, N) :-
    split_string(Out, "\n", "", Parts),
    exclude(==(""), Parts, [NumStr | _]),
    number_string(N0, NumStr),
    N is float(N0).

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

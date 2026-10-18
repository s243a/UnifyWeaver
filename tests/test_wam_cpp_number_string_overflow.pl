:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_cpp_number_string_overflow.pl
%
% P3 regression (Astra, PR #4259): number_string/2 rejected valid floats.
%
% The parser tried std::stoll BEFORE std::stod, and stoll throws
% std::out_of_range on an int64 overflow — swallowed by catch(...), which made
% number_string(N, "9223372036854775808.0") FAIL even though the text is a
% perfectly good float SWI accepts. The fix parses the float syntax
% INDEPENDENTLY of integer overflow: take the integer parse only when it both
% succeeds AND consumes the whole string, otherwise (partial / invalid /
% overflow) fall back to a whole-string float parse.
%
% Coverage (Astra's checklist): the overflow-float case, a plain float, a plain
% int (integer behaviour preserved), and a non-number that must still fail.
% Each is a 0-arity driver that writes the bound number, or "fail" when
% number_string/2 fails; the compiled binary is run once per driver.
%
% Skipped automatically when no C++ compiler is available.
%
%   swipl -q -g run_tests -t halt tests/test_wam_cpp_number_string_overflow.pl

:- module(test_wam_cpp_number_string_overflow,
          [test_wam_cpp_number_string_overflow/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(filesex), [directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ).
cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% The 0-arity driver predicates. Each writes the parsed number, or "fail".
install_program :-
    retractall(user:t_ovf),
    retractall(user:t_flt),
    retractall(user:t_int),
    retractall(user:t_bad),
    % '9223372036854775808' = 2^63, one past the signed-64-bit max: the ".0"
    % makes it a valid float, but the old int-first parse overflowed and bailed.
    assertz(user:(t_ovf :-
        ( number_string(N, '9223372036854775808.0') -> write(N) ; write(fail) ), nl)),
    assertz(user:(t_flt :-
        ( number_string(N, '3.5') -> write(N) ; write(fail) ), nl)),
    assertz(user:(t_int :-
        ( number_string(N, '42') -> write(N) ; write(fail) ), nl)),
    assertz(user:(t_bad :-
        ( number_string(_, 'abc') -> write(parsed) ; write(fail) ), nl)).

compile_project(Bin) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_number_string_overflow',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    install_program,
    write_wam_cpp_project(
        [user:t_ovf/0, user:t_flt/0, user:t_int/0, user:t_bad/0],
        [ module_name(numstr), emit_mode(interpreter), emit_main(true),
          on_compile_error(throw) ],
        Dir),
    !,
    directory_file_path(Dir, cpp, CppDir),
    format(atom(BuildCmd),
        '~w -std=c++17 -O0 ~w/main.cpp ~w/generated_program.cpp \c
         ~w/wam_runtime.cpp -o ~w/numstrbin 2>&1',
        [CC, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, BuildOut), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, '~n[number_string build failed]~n~w~n', [BuildOut]),
        throw(cpp_number_string_build_failed(Status))
    ),
    directory_file_path(CppDir, numstrbin, Bin).

driver_line(Bin, Key, Line) :-
    process_create(Bin, [Key],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    % First non-empty line is the driver's own write/1 output (the shim's
    % trailing true/false comes after).
    split_string(S1, "\n", "\r \t", Lines0),
    exclude(==(""), Lines0, Lines),
    ( Lines = [L0|_] -> Line = L0 ; Line = S2 ).

test_wam_cpp_number_string_overflow :-
    run_tests(wam_cpp_number_string_overflow).

:- begin_tests(wam_cpp_number_string_overflow, [condition(cpp_compiler(_))]).

test(number_string_overflow_and_variants, [setup(compile_project(Bin))]) :-
    driver_line(Bin, 't_ovf/0', Ovf),
    driver_line(Bin, 't_flt/0', Flt),
    driver_line(Bin, 't_int/0', Int),
    driver_line(Bin, 't_bad/0', Bad),
    % Overflow-float: MUST parse (pre-fix this printed "fail"). We don't pin the
    % exact float rendering (it differs from SWI's), only that a float came back.
    assertion(Ovf \== "fail"),
    assertion(( sub_string(Ovf, _, _, _, "e") ; sub_string(Ovf, _, _, _, ".") )),
    % Plain float and plain int: integer behaviour preserved.
    assertion(Flt == "3.5"),
    assertion(Int == "42"),
    % Non-number: must still fail.
    assertion(Bad == "fail"),
    ( exists_directory('output/test_wam_cpp_number_string_overflow')
    -> delete_directory_and_contents('output/test_wam_cpp_number_string_overflow')
    ; true ).

:- end_tests(wam_cpp_number_string_overflow).

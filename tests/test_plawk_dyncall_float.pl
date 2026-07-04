:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT): float-returning runtime-object calls. float(dyncall(...))
% and float(dyncall_at(...)) read the loaded grammar's numeric output as a
% double via @wam_object_call_f64 (@value_to_double), keeping fractions
% that the i64 form would lose -- or, for a Float-typed output, that the
% i64 form cannot read at all (it demands an Integer).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammars whose output is a Float (division yields float in eval)
half(X, R) :- R is X / 2.       % 7 -> 3.5   (dyncall, binary i64 arg)
threehalf(R) :- R is 7 / 2.     % -> 3.5     (dyncall_at, nullary)

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_float).

% float(dyncall(...)) / float(dyncall_at(...)) parse to their own nodes.
test(float_dyncall_forms_parse) :-
    plawk_parse_string("{ s += float(dyncall($1)) }\nEND { print s }\n",
        program(_, [rule(_, A1)], _)),
    memberchk(add(var(s), float_dyncall([field(1)])), A1),
    plawk_parse_string("{ s += float(dyncall_at($1)) }\nEND { print s }\n",
        program(_, [rule(_, A2)], _)),
    memberchk(add(var(s), float_dyncall_at(field(1), [])), A2),
    !.

test(float_arities_collected) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"g.wamo\" }\n\c
         { s += float(dyncall($1)) }\nEND { print s }\n",
        Program),
    plawk_program_dyncall_float_arities(Program, FArities),
    assertion(FArities == [1]),
    !.

% float(dyncall($1)) reads a Float grammar output and keeps the fraction.
test(float_dyncall_keeps_fraction, [condition(clang_available)]) :-
    f_dir(Dir),
    directory_file_path(Dir, 'half.wamo', Wamo),
    write_wam_object([user:half/2], [wamo_entry(half/2)], Wamo),
    binfmt_dynload_src(Wamo, "s += float(dyncall($1))", Src),
    build_prog(Dir, 'pf.plawk', 'pf', Src, Bin),
    write_i64(Dir, 'seven.bin', 7),
    directory_file_path(Dir, 'seven.bin', Input),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "3.5\n"),
    !.

% The i64 form of the SAME grammar returns 0: a Float output fails the
% integer read. This is why the float form is needed.
test(i64_dyncall_of_float_output_is_zero, [condition(clang_available)]) :-
    f_dir(Dir),
    directory_file_path(Dir, 'half.wamo', Wamo),
    ( exists_file(Wamo) -> true ; write_wam_object([user:half/2], [wamo_entry(half/2)], Wamo) ),
    binfmt_dynload_src(Wamo, "s += dyncall($1)", Src),
    build_prog(Dir, 'pi.plawk', 'pi', Src, Bin),
    ( exists_file_in(Dir, 'seven.bin') -> true ; write_i64(Dir, 'seven.bin', 7) ),
    directory_file_path(Dir, 'seven.bin', Input),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "0\n"),
    !.

% float(dyncall_at(...)) over a dynamic source, text mode: a nullary Float
% grammar chosen by a filename column.
test(float_dyncall_at_keeps_fraction, [condition(clang_available)]) :-
    f_dir(Dir),
    directory_file_path(Dir, 'threehalf.wamo', Wamo),
    write_wam_object([user:threehalf/1], [wamo_entry(threehalf/1)], Wamo),
    build_prog(Dir, 'pat.plawk', 'pat',
        "{ w += float(dyncall_at($1)) }\nEND { print w }\n", Bin),
    directory_file_path(Dir, 'in.txt', Input),
    setup_call_cleanup(
        open(Input, write, S, [encoding(utf8)]),
        format(S, "~w~n", [Wamo]),
        close(S)),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "3.5\n"),
    !.

:- end_tests(plawk_dyncall_float).

% --- helpers ---------------------------------------------------------------

f_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_float', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

exists_file_in(Dir, Name) :-
    directory_file_path(Dir, Name, Path), exists_file(Path).

binfmt_dynload_src(Wamo, Body, Src) :-
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n{ ~w }\nEND { print s }\n",
        [Wamo, Body]).

build_prog(Dir, ProgName, BinName, Src, Bin) :-
    directory_file_path(Dir, ProgName, Prog),
    setup_call_cleanup(
        open(Prog, write, S, [encoding(utf8)]),
        write(S, Src),
        close(S)),
    directory_file_path(Dir, BinName, Bin),
    cli([build, Prog, '-o', Bin], _, 0).

write_i64(Dir, Name, V) :-
    directory_file_path(Dir, Name, Path),
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(between(0, 7, I),
            ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )),
        close(Out)).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(null), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

% Compiled into the probe binary alongside the plawk driver.
plawk_ff_wf(I, F, R) :- R is I * F.
plawk_ff_bigf(F) :- F > 1.
plawk_ff_posval(F, R) :- F > 0.0, R is F + F.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_f64_foreign, [condition(clang_available)]).

test(f64_field_arg_marshals_as_wam_float) :-
    build_ff_ir("BEGIN { BINFMT = \"i64 f64\" } plawk_ff_bigf($2) { hits++ } END { print hits }\n", DriverIR),
    % a typed double load, then the Float constructor -- no text, no
    % interning anywhere near the argument
    assertion(once(sub_atom(DriverIR, _, _, _, '@value_float(double %'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    !.

test(float_call_uses_double_wrapper) :-
    build_ff_ir("BEGIN { BINFMT = \"i64 f64\" } { wsum += float(plawk_ff_wf($1, $2)) } END { print wsum }\n", DriverIR),
    % the double-returning wrapper: numeric-output check, promote via
    % @value_to_double, {double, i1} contract, 0.0 on failure
    assertion(once(sub_atom(DriverIR, _, _, _, 'define { double, i1 } @plawk_foreign_fcall_plawk_ff_wf_2'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@value_is_number'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@value_to_double'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'double %'))),
    % and the site selects 0.0 when the call fails
    assertion(once(sub_atom(DriverIR, _, _, _, ', double 0.0'))),
    !.

test(f64_foreign_rejections) :-
    Rejects = [
        % $3 does not exist in the layout
        "BEGIN { BINFMT = \"i64 f64\" } { wsum += float(plawk_ff_wf($1, $3)) } END { print wsum }\n",
        % string fields still do not marshal in binary mode
        "BEGIN { BINFMT = \"s8 f64\" } { wsum += float(plawk_ff_wf($1, $2)) } END { print wsum }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ build_ff_ir_for(Program, _))
        ;  true
        )).

test(surface_float_call_and_float_guard) :-
    % wf multiplies the i64 by the f64 in WAM arithmetic (Float
    % result); bigf compares the f64 against an integer.
    % (2,1.5) (3,0.25) (1,2.5): wsum = 3.0+0.75+2.5, hits = 2.
    run_ff_smoke("BEGIN { BINFMT = \"i64 f64\" } { wsum += float(plawk_ff_wf($1, $2)) } plawk_ff_bigf($2) { hits++ } END { print hits, wsum }\n",
        [rec(2, 1.5), rec(3, 0.25), rec(1, 2.5)],
        "2 6.25\n").

test(surface_failed_float_call_contributes_zero) :-
    % posval fails on non-positive inputs: those records add 0.0.
    % (1,1.5) (2,-2.5) (3,0.25): wsum = 3.0 + 0.0 + 0.5.
    run_ff_smoke("BEGIN { BINFMT = \"i64 f64\" } { wsum += float(plawk_ff_posval($2)) } END { print wsum }\n",
        [rec(1, 1.5), rec(2, -2.5), rec(3, 0.25)],
        "3.5\n").

:- end_tests(plawk_f64_foreign).

% --- helpers ---------------------------------------------------------------

ff_preds([ user:plawk_ff_wf/3,
           user:plawk_ff_bigf/1,
           user:plawk_ff_posval/2
         ]).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5,   0x3FF8000000000000).
double_bits(0.25,  0x3FD0000000000000).
double_bits(2.5,   0x4004000000000000).
double_bits(-2.5,  0xC004000000000000).

% rec(I64, F64)
write_ff_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(I, F), Recs),
            ( write_i64_le(Out, I),
              double_bits(F, Bits),
              write_i64_le(Out, Bits) )),
        close(Out)).

build_ff_ir(Source, DriverIR) :-
    plawk_parse_string(Source, Program),
    build_ff_ir_for(Program, DriverIR).

build_ff_ir_for(Program, DriverIR) :-
    once(( tmp_root(Root),
           directory_file_path(Root, 'uw_plawk_f64_foreign_ir', Dir),
           clean_dir(Dir),
           make_directory_path(Dir),
           directory_file_path(Dir, 'ir_probe.ll', LLPath),
           ff_preds(Preds),
           write_wam_llvm_project(Preds, [module_name('plawk_ff_ir')], LLPath),
           wam_llvm_last_compile_counts(InstrCount, LabelCount) )),
    plawk_program_native_driver_ir(Program, 'input.bin',
        [wam_vm(InstrCount, LabelCount)], DriverIR).

build_ff_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_f64_foreign', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    directory_file_path(Dir, 'probe.ll', LLPath),
    ff_preds(Preds),
    write_wam_llvm_project(Preds, [module_name('plawk_f64_foreign')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_program_native_driver_ir(Program, InputPath,
        [wam_vm(InstrCount, LabelCount)], DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk f64 foreign build output]~n~w~n", [BuildOut]),
       throw(plawk_f64_foreign_build_failed(Status))
    ).

run_ff_smoke(Source, Recs, ExpectedOutput) :-
    build_ff_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_ff_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.

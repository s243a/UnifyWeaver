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

% A DCG over payload bytes, written as its difference-list expansion so
% the WAM compiler sees plain clauses: parses "12,7,100"-style
% comma-separated decimal numbers and sums them. Backtracking between
% plawk_digits/4's greedy and stop clauses exercises real WAM choice
% points over the payload.
:- dynamic user:plawk_payload_sum/2.
:- dynamic user:plawk_payload_ok/1.
:- dynamic user:plawk_nums/3.
:- dynamic user:plawk_nums_rest/4.
:- dynamic user:plawk_num/3.
:- dynamic user:plawk_digits/4.
:- dynamic user:plawk_digit/3.

user:plawk_payload_sum(Blob, Sum) :-
    atom_codes(Blob, Codes),
    user:plawk_nums(Sum, Codes, []).

% guard flavor: payload parses to a sum above 50
user:plawk_payload_ok(Blob) :-
    user:plawk_payload_sum(Blob, Sum),
    Sum > 50.

user:plawk_nums(Sum, S0, S) :-
    user:plawk_num(N, S0, S1),
    user:plawk_nums_rest(N, Sum, S1, S).

% NOTE: the separator is matched with a variable head plus an ==/2
% guard rather than the natural [0', | S0] head: the hybrid WAM
% compiler currently miscompiles clause heads with a constant inside a
% list cell (the get_list/unify_constant path never matches at
% runtime). Known upstream bug -- see the PR notes; this formulation is
% semantically identical.
user:plawk_nums_rest(Acc, Sum, [C | S0], S) :-
    C == 0',,
    user:plawk_num(N, S0, S1),
    Acc2 is Acc + N,
    user:plawk_nums_rest(Acc2, Sum, S1, S).
user:plawk_nums_rest(Sum, Sum, S, S).

user:plawk_num(N, S0, S) :-
    user:plawk_digit(D, S0, S1),
    user:plawk_digits(D, N, S1, S).

user:plawk_digits(Acc, N, S0, S) :-
    user:plawk_digit(D, S0, S1),
    Acc2 is Acc * 10 + D,
    user:plawk_digits(Acc2, N, S1, S).
user:plawk_digits(N, N, S, S).

user:plawk_digit(D, [C | S], S) :-
    C >= 48,
    C =< 57,
    D is C - 48.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_tier2_blob, [condition(clang_available)]).

test(blob_ir_marshals_transient_atom_and_int_field) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 blob32\" } { total += plawk_payload_sum($2) ; wsum += plawk_payload_sum($2) * $1 } END { print total }\n", Program),
    build_tier2_ir(Program, DriverIR),
    % blob payload: length load + transient copy, marshaled as an atom Value
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_transient_atom_from_bytes('))),
    % the payload region sits after the 8-byte length at offset 8+8
    assertion(once(sub_atom(DriverIR, _, _, _, 'getelementptr i8, i8* %rec, i64 16'))),
    % no interning anywhere in the record loop
    assertion(\+ sub_atom(DriverIR, _, _, _, 'wam_intern_atom(i8* %rec')),
    !.

test(binfmt_i64_foreign_arg_is_value_integer) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } { total += plawk_payload_sum($1) } END { print total }\n", Program),
    % (arg typing only: an i64 field marshals as a WAM integer)
    build_tier2_ir(Program, DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@value_integer(i64 %'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    !.

test(tier2_rejections) :-
    Rejects = [
        % f64 fields are not marshalable in this slice
        "BEGIN { BINFMT = \"f64 blob32\" } { t += plawk_payload_sum($1) } END { print t }\n",
        % blob fields have no print/guard/arithmetic role
        "BEGIN { BINFMT = \"i64 blob32\" } { print $2 }\n",
        "BEGIN { BINFMT = \"i64 blob32\" } $2 == \"x\" { c++ } END { print c }\n",
        "BEGIN { BINFMT = \"i64 blob32\" } { t += $2 } END { print t }\n",
        % at most one blob argument per call (shared transient buffer)
        "BEGIN { BINFMT = \"blob8 blob8\" } { t += plawk_payload_sum($1, $2) } END { print t }\n",
        % blob is input-only
        "BEGIN { BINFMT = \"i64 blob32\" ; OUTFMT = \"blob32\" } { writebin $2 }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ build_tier2_ir(Program, _))
        ;  true
        )).

test(surface_blob_payload_dcg_sum) :-
    % The record loop frames natively; the payload is parsed by the
    % compiled Prolog DCG through the bridge. Guarded records only.
    run_tier2_smoke("BEGIN { BINFMT = \"i64 blob32\" } $1 > 0 { total += plawk_payload_sum($2) } END { print total }\n",
        [rec(1, "12,7"), rec(2, "100"), rec(-5, "9")],
        "119\n").

test(surface_blob_prolog_guard) :-
    run_tier2_smoke("BEGIN { BINFMT = \"i64 blob32\" } plawk_payload_ok($2) { hits++ } END { print hits }\n",
        [rec(1, "12,7"), rec(2, "100"), rec(3, "40,41")],
        "2\n").

test(surface_blob_with_i64_arg_mix) :-
    % i64 field marshaled as a WAM integer alongside the payload sum.
    run_tier2_smoke("BEGIN { BINFMT = \"i64 blob32\" } { total += plawk_payload_sum($2) * $1 } END { print total }\n",
        [rec(2, "10"), rec(3, "5")],
        "35\n").

test(surface_blob_error_paths) :-
    build_tier2_probe("BEGIN { BINFMT = \"i64 blob32\" } { total += plawk_payload_sum($2) } END { print total }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % oversized payload length
    write_raw(InputPath, [i64(1), i64(33), pad(33)]),
    run_status(BinPath, exit(11)),
    % truncated payload
    write_raw(InputPath, [i64(1), i64(5), bytes("12")]),
    run_status(BinPath, exit(11)),
    % clean EOF runs END
    write_raw(InputPath, []),
    run_expect(BinPath, "0\n"),
    !.

% --- helpers ---------------------------------------------------------------

tier2_preds([ user:plawk_payload_sum/2,
              user:plawk_payload_ok/1,
              user:plawk_nums/3,
              user:plawk_nums_rest/4,
              user:plawk_num/3,
              user:plawk_digits/4,
              user:plawk_digit/3
            ]).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% rec(Id, PayloadString): i64 id, i64 payload length, payload bytes
write_blob_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(Id, Payload), Recs),
            ( write_i64_le(Out, Id),
              string_codes(Payload, Codes),
              length(Codes, Len),
              write_i64_le(Out, Len),
              forall(member(C, Codes), put_byte(Out, C)) )),
        close(Out)).

write_raw(Path, Items) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Item, Items),
            ( Item = i64(V) -> write_i64_le(Out, V)
            ; Item = bytes(S) -> ( string_codes(S, Cs),
                                   forall(member(C, Cs), put_byte(Out, C)) )
            ; Item = pad(N) -> forall(between(1, N, _), put_byte(Out, 0))
            )),
        close(Out)).

% IR-only build (throwaway module compile for the vm counts). The
% compile is fenced with once/1: rejection tests negate this goal, and
% without the fence \+ would backtrack into write_wam_llvm_project's
% internal choice points and re-run the whole compiler search.
build_tier2_ir(Program, DriverIR) :-
    once(( tmp_root(Root),
           directory_file_path(Root, 'uw_plawk_tier2_blob_ir', Dir),
           clean_dir(Dir),
           make_directory_path(Dir),
           directory_file_path(Dir, 'ir_probe.ll', LLPath),
           tier2_preds(Preds),
           write_wam_llvm_project(Preds, [module_name('plawk_tier2_ir')], LLPath),
           wam_llvm_last_compile_counts(InstrCount, LabelCount) )),
    plawk_program_native_driver_ir(Program, 'input.bin',
        [wam_vm(InstrCount, LabelCount)], DriverIR).

build_tier2_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_tier2_blob', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    directory_file_path(Dir, 'probe.ll', LLPath),
    tier2_preds(Preds),
    write_wam_llvm_project(Preds, [module_name('plawk_tier2_blob')], LLPath),
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
    ;  format(user_error, "~n[plawk tier2 blob build output]~n~w~n", [BuildOut]),
       throw(plawk_tier2_blob_build_failed(Status))
    ).

run_status(BinPath, ExpectedStatus) :-
    process_create(BinPath, [],
                   [stdout(null), stderr(std), process(Pid)]),
    process_wait(Pid, Status),
    assertion(Status == ExpectedStatus).

run_expect(BinPath, ExpectedOutput) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == ExpectedOutput).

run_tier2_smoke(Source, Recs, ExpectedOutput) :-
    build_tier2_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_blob_records(InputPath, Recs),
    run_expect(BinPath, ExpectedOutput),
    !.

:- end_tests(plawk_tier2_blob).

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 4, surface: structured-return destructuring.
% `(v1, ..., vn) = dyncall[@name](args) as (T1 ... Tn)` calls a grammar
% whose entry returns a Compound and deserializes each arg into a typed
% scalar variable (i64 or f64), via @wam_object_call_record. Chosen surface
% is "destructure to variables" (record-view $1.. reindex is a later phase).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar predicates compiled into one multi-entry .wamo. Each returns a
% Compound record (arity 2): an integer field and a float field.
rec(X, R)  :- H is X + 0.5, R = pair(X, H).       % rec(10) -> pair(10, 10.5)
rec2(X, R) :- D is X * 2, H is X / 4.0, R = pair(D, H). % rec2(10)-> pair(20,2.5)

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_struct).

% The destructure statement parses to its own node.
test(struct_parse) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { (n, half) = dyncall@rec($1) as (i64 f64) }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynrec_bind([n, half], dyncall_named(rec, [field(1)]),
        [i64, f64]), Actions),
    !.

% A malformed shape (arity mismatch) is not a valid destructure.
test(struct_arity_mismatch_rejected) :-
    \+ plawk_native_codegen:plawk_dynrec_bind_ok(
        dynrec_bind([a, b], dyncall(_), [i64])),
    !.

% Record shim + primitive call are emitted (named entry here).
test(struct_ir_emitted) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { (n, half) = dyncall@rec($1) as (i64 f64) ; total += n }\n\c
         END { print total }\n",
        Program),
    plawk_program_dyncall_named_rec_entries(Program, Entries),
    assertion(Entries == [rec-1]),
    plawk_program_native_driver_ir(Program, stdin_or_argv,
        [wam_vm(10, 10)], IR),
    sub_atom(IR, _, _, _, '@plawk_dyncall_named_rec_rec_1'),
    sub_atom(IR, _, _, _, '@wam_object_call_record'),
    !.

% Full round trip: rec(X) returns pair(X, X+0.5). Destructure into an i64
% (n) and an f64 (half); accumulate both. For inputs 10, 20:
%   n:    10 + 20            = 30
%   half: 10.5 + 20.5        = 31.0
test(struct_destructure_runs, [condition(clang_available)]) :-
    st_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:rec/2, user:rec2/2],
        [wamo_entries([rec/2, rec2/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { (n, half) = dyncall@rec($1) as (i64 f64) ; total += n ; sum += half }\n\c
         END { print total }\n", [Wamo]),
    write_prog(Dir, 'st.plawk', Src),
    directory_file_path(Dir, 'st.plawk', Prog),
    directory_file_path(Dir, 'st_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, [10, 20]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "30\n"),
    !.

% The float half of the same destructure: print the accumulated f64 sum
% (10.5 + 20.5 = 31), proving the second field deserialized as a double.
test(struct_destructure_float_field, [condition(clang_available)]) :-
    st_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:rec/2, user:rec2/2],
          [wamo_entries([rec/2, rec2/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { (n, half) = dyncall@rec($1) as (i64 f64) ; sum += half }\n\c
         END { print sum }\n", [Wamo]),
    write_prog(Dir, 'stf.plawk', Src),
    directory_file_path(Dir, 'stf.plawk', Prog),
    directory_file_path(Dir, 'stf_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, [10, 20]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "31\n"),
    !.

:- end_tests(plawk_dyncall_struct).

% --- helpers ---------------------------------------------------------------

st_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_struct', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_i64_records(Path, Values) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(V, Values), write_i64_le(Out, V)),
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

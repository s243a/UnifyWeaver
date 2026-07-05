:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 4, assoc-return surface: a grammar returning a
% list of integer key-value pairs populates a plawk associative array.
%   arr = dyncall@tally($1) as assoc
% Per record the returned [K-V, ...] pairs are inserted (accumulated) into
% arr's i64 table via @wam_object_call_assoc; END arr[key] lookups see the
% result. Named entry, integer keys/values (the mechanism's shape).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% tally(X) returns two pairs: bucket 1 gets X, bucket 2 gets a flat 100.
tally(X, R) :- R = [1-X, 2-100].

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_assoc).

% `arr = dyncall@name(...) as assoc` parses to its own node.
test(assoc_bind_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall@tally($1) as assoc }\nEND { print arr[1] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynassoc_bind(var(arr), dyncall_named(tally, [field(1)])),
        Actions),
    !.

% The named-assoc entry is collected, and the shim + primitive are emitted.
test(assoc_ir_emitted) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall@tally($1) as assoc }\nEND { print arr[1] }\n",
        Program),
    plawk_program_dyncall_named_assoc_entries(Program, Entries),
    assertion(Entries == [tally-1]),
    plawk_program_native_driver_ir(Program, stdin_or_argv,
        [wam_vm(10, 10)], IR),
    sub_atom(IR, _, _, _, '@plawk_dyncall_assoc_tally_1'),
    sub_atom(IR, _, _, _, '@wam_object_call_assoc'),
    !.

% Full round trip: per record, tally($1) -> [1-$1, 2-100] accumulates into
% arr. Over inputs 5, 7: arr[1] = 5+7 = 12, arr[2] = 100+100 = 200.
test(assoc_populates_and_reads, [condition(clang_available)]) :-
    da_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:tally/2], [wamo_entries([tally/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@tally($1) as assoc }\n\c
         END { print arr[1], arr[2] }\n", [Wamo]),
    build_run(Dir, 'da', Src, [5, 7], Out),
    assertion(Out == "12 200\n"),
    !.

:- end_tests(plawk_dyncall_assoc).

% --- helpers ---------------------------------------------------------------

da_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_assoc', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Values, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, Values),
    run_bin(Bin, [Input], Out, 0).

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

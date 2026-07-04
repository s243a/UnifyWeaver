:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 3, surface A follow-on: float(dyncall@name(...))
% and blob(dyncall@name(...)). The named-entry form gains double and
% byte-slice returns, mirroring float(dyncall(...)) / blob(dyncall(...)).
% All three return kinds (i64/double/bytes) for one entry share a single
% per-entry PC resolver, so an entry used in more than one position
% resolves once.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar predicates compiled into one multi-entry .wamo
halve(X, R)  :- R is X / 2.          % Float output -> read via float(...)
greet(A, A).                         % echoes its input atom -> read via blob(...)

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_named_fb).

% float(dyncall@name(...)) / blob(dyncall@name(...)) parse to their own nodes.
test(named_fb_parse) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n{ print float(dyncall@halve($1)) }\n",
        program(_, [rule(_, A1)], _)),
    memberchk(print([float_dyncall_named(halve, [field(1)])]), A1),
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n{ print blob(dyncall@greet($1)) }\n",
        program(_, [rule(_, A2)], _)),
    memberchk(print([blob_dyncall_named(greet, [field(1)])]), A2),
    !.

% The named float/blob entries are collected and their shims emitted.
test(named_fb_shims_emitted) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n{ print float(dyncall@halve($1)) }\n",
        FP),
    plawk_program_dyncall_named_float_entries(FP, FE),
    assertion(FE == [halve-1]),
    plawk_program_native_driver_ir(FP, stdin_or_argv, [wam_vm(10, 10)], FIR),
    sub_atom(FIR, _, _, _, '@plawk_dyncall_named_f_halve_1'),
    sub_atom(FIR, _, _, _, '@plawk_dyncall_resolve_halve_1'),
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n{ print blob(dyncall@greet($1)) }\n",
        BP),
    plawk_program_dyncall_named_blob_entries(BP, BE),
    assertion(BE == [greet-1]),
    plawk_program_native_driver_ir(BP, stdin_or_argv, [wam_vm(10, 10)], BIR),
    sub_atom(BIR, _, _, _, '@plawk_dyncall_named_b_greet_1'),
    !.

% An entry used as both i64 and float shares ONE resolver + PC global (no
% duplicate LLVM symbols).
test(shared_entry_single_resolver) :-
    plawk_native_codegen:plawk_dyncall_support_ir('lib.wamo',
        [], [], [], [square-1], [square-1], [square-1], IR),
    findall(P, sub_atom(IR, P, _, _,
        '@plawk_dyncall_pc_square_1 = internal'), PCs),
    assertion(PCs = [_]),
    findall(R, sub_atom(IR, R, _, _,
        'define i32 @plawk_dyncall_resolve_square_1('), Rs),
    assertion(Rs = [_]),
    !.

% float(dyncall@halve($1)) reads a named entry's Float output as a double:
% halve(N) = N/2, so summing over 3 and 5 gives 1.5 + 2.5 = 4.0.
test(named_float_runs, [condition(clang_available)]) :-
    fb_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:halve/2, user:greet/2],
        [wamo_entries([halve/2, greet/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { total += float(dyncall@halve($1)) }\nEND { print total }\n",
        [Wamo]),
    write_prog(Dir, 'nf.plawk', Src),
    directory_file_path(Dir, 'nf.plawk', Prog),
    directory_file_path(Dir, 'nf_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, [3, 5]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "4\n"),           % 1.5 + 2.5, printed as a whole double
    !.

% blob(dyncall@greet($1)) reads a named entry's Atom output as bytes: greet
% echoes its input field, printed via %.*s.
test(named_blob_runs, [condition(clang_available)]) :-
    fb_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:halve/2, user:greet/2],
          [wamo_entries([halve/2, greet/2])], Wamo) ),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n{ print blob(dyncall@greet($1)) }\n",
        [Wamo]),
    write_prog(Dir, 'nb.plawk', Src),
    directory_file_path(Dir, 'nb.plawk', Prog),
    directory_file_path(Dir, 'nb_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.txt', Input),
    write_text(Input, "alpha\nbeta\n"),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "alpha\nbeta\n"),
    !.

:- end_tests(plawk_dyncall_named_fb).

% --- helpers ---------------------------------------------------------------

fb_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_named_fb', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

write_text(Path, Text) :-
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

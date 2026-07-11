:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The FULLY-JIT binary reader: a reader grammar COMPILED FROM SOURCE TEXT
% at runtime parses a binary payload and returns a structured record.
%
%   { (s, c) = dyncall_at@gr_parse(compile("[...]"), $2) as (i64 i64) }
%
% This is the runtime-compiled counterpart of the grammar-driven reader
% capstone (test_plawk_grammar_reader.pl, which used a SHIPPED grammar):
% here the reader itself is compiled inside the running binary via the
% bootstrap-compiler object, loaded into a fresh VM, and its named entry
% resolved per record against that VM's materialized entry table
% (@wam_object_vm_entry_pc). The BINFMT blobN payload flows in as a
% transient atom; the returned pair(Sum, Count) compound is deserialized
% into typed plawk scalars by @wam_object_call_record. Bytes-in +
% record-out + reader-compiled-at-runtime.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% The reader grammar as SOURCE TEXT (compiled at runtime). Parses a
% decimal number from the payload bytes and returns pair(Value, DigitCount)
% -- a structured record. A difference-list DCG with a real choice point
% (gr_digits' greedy vs. stop clauses) over the payload, staying inside
% the bootstrap compiler's subset (variable-only list heads; a constant
% char literal in a list head, e.g. a comma separator, is a documented
% bootstrap-subset gap, so the reader consumes one number per record).
reader_source(
  "[(gr_parse(B, pair(N, C)) :- atom_codes(B, Cs), gr_num(N, C, Cs, [])), \c
    (gr_num(N, C, S0, S) :- gr_digit(D, S0, S1), gr_digits(D, 1, N, C, S1, S)), \c
    (gr_digits(A, AC, N, C, S0, S) :- gr_digit(D, S0, S1), A2 is A * 10 + D, AC2 is AC + 1, gr_digits(A2, AC2, N, C, S1, S)), \c
    (gr_digits(N, C, N, C, S, S)), \c
    (gr_digit(D, [Ch | S], S) :- Ch >= 48, Ch =< 57, D is Ch - 48)]").

:- begin_tests(plawk_jit_reader).

% A record destructure over a runtime source parses (dyncall_at@name with
% a compile() source) and is collected as both an at-named-rec entry and
% a compile site (so the CLI ships the compiler object).
test(jit_reader_parses_and_collects) :-
    reader_source(RSrc),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" }\n\c
         { (s, c) = dyncall_at@gr_parse(compile(\"~w\"), $2) as (i64 i64) ; total += s }\n\c
         END { print total }\n", [RSrc]),
    plawk_parse_string(Src, Program),
    Program = program(_, [rule(_, Actions)], _),
    memberchk(dynrec_bind([s, c],
        dyncall_at_named(gr_parse, compile_src(string(_)), [field(2)]),
        [i64, i64]), Actions),
    plawk_program_dyncall_at_named_rec_entries(Program, Entries),
    assertion(Entries == [gr_parse-1]),
    plawk_program_compile_sites(Program, Sites),
    assertion(Sites \== []),
    !.

% The emitted driver composes the at-record shim, the record primitive,
% the per-call entry resolver, and the blob transient-atom marshal.
test(jit_reader_ir_composes, [condition(clang_available)]) :-
    reader_source(RSrc),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" }\n\c
         { (s, c) = dyncall_at@gr_parse(compile(\"~w\"), $2) as (i64 i64) ; total += s }\n\c
         END { print total }\n", [RSrc]),
    plawk_parse_string(Src, Program),
    % evalc_path option stands in for the CLI-shipped compiler object
    plawk_program_native_driver_ir(Program, 'in.bin',
        [wam_vm(10, 10), evalc_path('cgfull.wamo')], IR),
    assertion(once(sub_atom(IR, _, _, _, '@plawk_dyncall_at_named_rec_gr_parse_1'))),
    assertion(once(sub_atom(IR, _, _, _, '@wam_object_call_record'))),
    assertion(once(sub_atom(IR, _, _, _, '@wam_object_vm_entry_pc'))),
    !.

% THE FULLY-JIT READER, end to end: the reader grammar is compiled from
% source text at runtime and used to parse each record's binary payload
% into pair(Value, DigitCount). Payloads "12" (12,2), "7" (7,1),
% "345" (345,3): total value = 364, total digits = 6.
test(jit_reader_runtime_compiled, [condition(clang_available)]) :-
    jr_dir(Dir),
    reader_source(RSrc),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" }\n\c
         { (s, c) = dyncall_at@gr_parse(compile(\"~w\"), $2) as (i64 i64) ; total += s ; recs += c }\n\c
         END { print total, recs }\n", [RSrc]),
    write_prog(Dir, 'jit.plawk', Src),
    directory_file_path(Dir, 'jit.plawk', Prog),
    directory_file_path(Dir, 'jit_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    % the bootstrap-compiler object ships next to the binary
    atom_concat(Bin, '.evalc.wamo', EvalcWamo),
    assertion(exists_file(EvalcWamo)),
    directory_file_path(Dir, 'in.bin', Input),
    write_blob_records(Input, [rec(1, "12"), rec(2, "7"), rec(3, "345")]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "364 6\n"),
    !.

% A SECOND runtime-compiled reader coexists (per-source dedup): the same
% binary compiles a different grammar that returns pair(Value, Value*2).
% Payloads "5","20": values 5,20 -> sum 25; doubles 10,40 -> 50.
test(jit_reader_two_grammars, [condition(clang_available)]) :-
    jr_dir(Dir),
    reader_source(RSrc),
    format(string(DblSrc),
        "[(gr_dbl(B, pair(N, D)) :- atom_codes(B, Cs), gr_num(N, _C, Cs, []), D is N * 2), \c
          (gr_num(N, C, S0, S) :- gr_digit(G, S0, S1), gr_digits(G, 1, N, C, S1, S)), \c
          (gr_digits(A, AC, N, C, S0, S) :- gr_digit(G, S0, S1), A2 is A * 10 + G, AC2 is AC + 1, gr_digits(A2, AC2, N, C, S1, S)), \c
          (gr_digits(N, C, N, C, S, S)), \c
          (gr_digit(G, [Ch | S], S) :- Ch >= 48, Ch =< 57, G is Ch - 48)]", []),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" }\n\c
         { (n, d) = dyncall_at@gr_dbl(compile(\"~w\"), $2) as (i64 i64) ; sv += n ; dv += d }\n\c
         END { print sv, dv }\n", [DblSrc]),
    write_prog(Dir, 'jit2.plawk', Src),
    directory_file_path(Dir, 'jit2.plawk', Prog),
    directory_file_path(Dir, 'jit2_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in2.bin', Input),
    write_blob_records(Input, [rec(1, "5"), rec(2, "20")]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "25 50\n"),
    !.

:- end_tests(plawk_jit_reader).

% --- helpers ---------------------------------------------------------------

jr_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_jit_reader', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text), close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

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

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
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

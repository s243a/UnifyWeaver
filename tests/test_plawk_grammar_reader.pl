:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Grammar-driven binary reader -- the endgame JIT roadmap item 4 pointed
% at, and the DCG-binary-readers Tier-2/3 story (PLAWK_DCG_BINARY_READERS.md):
% BYTES IN + RECORD OUT through a LOADED WAM grammar.
%
% The record loop frames a binary payload natively (BINFMT `blobN`: an
% 8-byte length + payload bytes). The payload is handed -- as a transient
% atom, constant-memory, no interning -- to a grammar shipped as a .wamo
% object, which PARSES the bytes with a real WAM DCG (choice points and
% all) and RETURNS A STRUCTURED RECORD (a Compound). @wam_object_call_record
% deserializes that compound's args into typed plawk scalars, so a format
% too irregular for the native Tier-1 reader is read by a grammar without
% leaving the compiled loop.
%
% This composes two arcs the campaign built separately: the blob bridge
% (bytes into a compiled predicate, item 2 / Tier-2) and structured-return
% destructuring (a Compound out, deserialized to typed fields, item 4).
% The novelty here is routing the two THROUGH A LOADED OBJECT
% (dyncall@name), not a compiled-in foreign predicate -- so the reader
% grammar is a shippable, swappable artifact.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% The reader grammar, shipped as a .wamo. It takes the payload atom,
% parses comma-separated decimal numbers with a difference-list DCG (the
% same shape the Tier-2 blob bridge exercises), and returns a two-field
% record pair(Sum, Count) -- a STRUCTURED result, not a scalar.
:- dynamic user:gr_parse/2.
:- dynamic user:gr_nums/4.
:- dynamic user:gr_nums_rest/6.
:- dynamic user:gr_num/3.
:- dynamic user:gr_digits/4.
:- dynamic user:gr_digit/3.

user:gr_parse(Blob, pair(Sum, Count)) :-
    atom_codes(Blob, Codes),
    user:gr_nums(Sum, Count, Codes, []).

% Sum and Count threaded together: one number seen, then the rest.
user:gr_nums(Sum, Count, S0, S) :-
    user:gr_num(N, S0, S1),
    user:gr_nums_rest(N, 1, Sum, Count, S1, S).

user:gr_nums_rest(AccSum, AccCount, Sum, Count, [0', | S0], S) :-
    user:gr_num(N, S0, S1),
    AccSum2 is AccSum + N,
    AccCount2 is AccCount + 1,
    user:gr_nums_rest(AccSum2, AccCount2, Sum, Count, S1, S).
user:gr_nums_rest(Sum, Count, Sum, Count, S, S).

user:gr_num(N, S0, S) :-
    user:gr_digit(D, S0, S1),
    user:gr_digits(D, N, S1, S).

user:gr_digits(Acc, N, S0, S) :-
    user:gr_digit(D, S0, S1),
    Acc2 is Acc * 10 + D,
    user:gr_digits(Acc2, N, S1, S).
user:gr_digits(N, N, S, S).

user:gr_digit(D, [C | S], S) :-
    C >= 48,
    C =< 57,
    D is C - 48.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_grammar_reader).

% A blob field flowing into a record-returning dyncall parses and passes
% the binary-mode surface check (a blob field is a legal dyncall arg; the
% record bind is a legal binary-mode action).
test(reader_parses_and_collects) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64 blob32\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { (s, c) = dyncall@gr_parse($2) as (i64 i64) ; total += s ; recs += c }\n\c
         END { print total }\n",
        Program),
    Program = program(_, [rule(_, Actions)], _),
    memberchk(dynrec_bind([s, c], dyncall_named(gr_parse, [field(2)]),
        [i64, i64]), Actions),
    plawk_program_dyncall_named_rec_entries(Program, Entries),
    assertion(Entries == [gr_parse-1]),
    !.

% The record shim + the record primitive + the blob transient-atom
% marshal all appear: bytes-in (transient atom) meets record-out
% (@wam_object_call_record) in one emitted driver.
test(reader_ir_composes_blob_and_record) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64 blob32\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { (s, c) = dyncall@gr_parse($2) as (i64 i64) ; total += s }\n\c
         END { print total }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'in.bin',
        [wam_vm(10, 10)], IR),
    assertion(once(sub_atom(IR, _, _, _, '@plawk_dyncall_named_rec_gr_parse_1'))),
    assertion(once(sub_atom(IR, _, _, _, '@wam_object_call_record'))),
    assertion(once(sub_atom(IR, _, _, _, '@wam_transient_atom_from_bytes('))),
    !.

% THE CAPSTONE, end to end: each record carries an i64 id and a
% comma-separated-number payload. The shipped grammar parses the payload
% and returns pair(Sum, Count); the destructure lands Sum in s and Count
% in c. Over the payloads "12,7" (sum 19, count 2), "100" (100, 1),
% "40,41,9" (90, 3): total sum = 209, total count = 6.
test(reader_bytes_in_record_out, [condition(clang_available)]) :-
    gr_dir(Dir),
    directory_file_path(Dir, 'reader.wamo', Wamo),
    reader_preds(Preds),
    write_wam_object(Preds, [wamo_entries([gr_parse/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" ; DYNLOAD = \"~w\" }\n\c
         { (s, c) = dyncall@gr_parse($2) as (i64 i64) ; total += s ; recs += c }\n\c
         END { print total, recs }\n", [Wamo]),
    write_prog(Dir, 'reader.plawk', Src),
    directory_file_path(Dir, 'reader.plawk', Prog),
    directory_file_path(Dir, 'reader_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_blob_records(Input, [rec(1, "12,7"), rec(2, "100"),
                               rec(3, "40,41,9")]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "209 6\n"),
    !.

% A guard can gate on the id field while the payload is read by the
% grammar: only id > 1 records contribute. "100" (sum 100) + "40,41,9"
% (sum 90) = 190; the "12,7" record (id 1) is skipped.
test(reader_guarded_by_scalar_field, [condition(clang_available)]) :-
    gr_dir(Dir),
    directory_file_path(Dir, 'reader.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; reader_preds(Preds),
      write_wam_object(Preds, [wamo_entries([gr_parse/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 blob32\" ; DYNLOAD = \"~w\" }\n\c
         $1 > 1 { (s, c) = dyncall@gr_parse($2) as (i64 i64) ; total += s }\n\c
         END { print total }\n", [Wamo]),
    write_prog(Dir, 'readerg.plawk', Src),
    directory_file_path(Dir, 'readerg.plawk', Prog),
    directory_file_path(Dir, 'readerg_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_blob_records(Input, [rec(1, "12,7"), rec(2, "100"),
                               rec(3, "40,41,9")]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "190\n"),
    !.

:- end_tests(plawk_grammar_reader).

% --- helpers ---------------------------------------------------------------

reader_preds([ user:gr_parse/2,
               user:gr_nums/4,
               user:gr_nums_rest/6,
               user:gr_num/3,
               user:gr_digits/4,
               user:gr_digit/3
             ]).

gr_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_grammar_reader', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text), close(Out)).

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

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% JIT roadmap item 2 follow-on: the blob (ptr,len) slice beyond print
% position. A runtime grammar's byte output now feeds three more
% consumers:
%   - writebin sN / lpsN slots (clamped copy; failed call = empty),
%   - rule equality guards (blob(dyncall...) == "literal"),
%   - assoc keys (counts[blob(dyncall...)]++ interns the bytes like a
%     field slice; failed call skips the increment).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammars: echo returns its input atom; classify buckets a numeric atom
echo2(A, A).
classify(X, R) :-
    atom_number(X, N),
    ( N > 100 -> R = big ; R = small ).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_blob_consumers).

% The new positions parse to their nodes, and the generic blob walk
% collects arities from ALL of them (patterns, assoc keys, writebin
% slots) -- the shims must exist wherever a blob evaluates.
test(blob_consumer_positions_parse_and_collect) :-
    plawk_parse_source(
        "BEGIN { DYNLOAD = \"g.wamo\" ; OUTFMT = \"lps8\" }\nblob(dyncall($1)) == \"big\" { counts[blob(dyncall($2))]++ }\n{ writebin blob(dyncall($3)) }\nEND { for (k in counts) print k, counts[k] }\n",
        Program, _),
    Program = program(_, Rules, _),
    memberchk(rule(blob_eq(blob_dyncall([field(1)]), "big"), _), Rules),
    memberchk(rule(_, [inc_assoc(var(counts), blob_dyncall([field(2)]))]), Rules),
    memberchk(rule(_, [writebin([blob_dyncall([field(3)])])]), Rules),
    plawk_program_dyncall_blob_arities(Program, Arities),
    assertion(Arities == [1]),
    !.

% blob(dyncall($1)) == "big" as a rule guard: classify/2 buckets each
% number and only the "big" bucket increments. 50, 200, 300, 7 -> 2.
test(blob_eq_guard_filters_rules, [condition(clang_available)]) :-
    bc_dir(Dir),
    directory_file_path(Dir, 'classify.wamo', Wamo),
    write_wam_object([user:classify/2], [wamo_entry(classify/2)], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         blob(dyncall($1)) == \"big\" { hits += 1 }\n\c
         END { print hits }\n", [Wamo]),
    build_prog(Dir, 'guard.plawk', 'guard_bin', Src, Bin),
    directory_file_path(Dir, 'nums.txt', Input),
    write_text(Input, "50\n200\n300\n7\n"),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "2\n"),
    !.

% counts[blob(dyncall($1))]++: the grammar's byte output keys the
% table. classify/2 buckets the numbers; the END for-in walks the
% table in slot order (deterministic for a fixed key set).
% 50, 200, 300, 7 -> big 2, small 2.
test(blob_assoc_key_counts, [condition(clang_available)]) :-
    bc_dir(Dir),
    directory_file_path(Dir, 'classify.wamo', Wamo),
    ( exists_file(Wamo)
    -> true
    ;  write_wam_object([user:classify/2], [wamo_entry(classify/2)], Wamo)
    ),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { counts[blob(dyncall($1))]++ }\n\c
         END { for (k in counts) print k, counts[k] }\n", [Wamo]),
    build_prog(Dir, 'akey.plawk', 'akey_bin', Src, Bin),
    directory_file_path(Dir, 'nums.txt', Input),
    ( exists_file(Input) -> true ; write_text(Input, "50\n200\n300\n7\n") ),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "big 2\nsmall 2\n"),
    !.

% writebin an lps8 slot from a blob: echo2/2 passes each field through,
% so the output stream is per record an 8-byte LE length + the bytes.
test(blob_writebin_lps_slot, [condition(clang_available)]) :-
    bc_dir(Dir),
    directory_file_path(Dir, 'echo2.wamo', Wamo),
    write_wam_object([user:echo2/2], [wamo_entry(echo2/2)], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" ; OUTFMT = \"lps8\" }\n\c
         { writebin blob(dyncall($1)) }\n", [Wamo]),
    build_prog(Dir, 'wblps.plawk', 'wblps_bin', Src, Bin),
    directory_file_path(Dir, 'words.txt', Input),
    write_text(Input, "ab\nxyz\n"),
    run_bin_bytes(Bin, [Input], Bytes, 0),
    lps_record("ab", R1), lps_record("xyz", R2),
    append(R1, R2, Expected),
    assertion(Bytes == Expected),
    !.

% writebin an s4 slot from a blob next to a numeric slot: the blob bytes
% land zero-padded in the fixed-width slot ("ab\0\0", "xyz\0").
test(blob_writebin_s_slot, [condition(clang_available)]) :-
    bc_dir(Dir),
    directory_file_path(Dir, 'echo2.wamo', Wamo),
    ( exists_file(Wamo)
    -> true
    ;  write_wam_object([user:echo2/2], [wamo_entry(echo2/2)], Wamo)
    ),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" ; OUTFMT = \"s4 i64\" }\n\c
         { writebin blob(dyncall($1)), NR }\n", [Wamo]),
    build_prog(Dir, 'wbs.plawk', 'wbs_bin', Src, Bin),
    directory_file_path(Dir, 'words.txt', Input),
    ( exists_file(Input) -> true ; write_text(Input, "ab\nxyz\n") ),
    run_bin_bytes(Bin, [Input], Bytes, 0),
    i64_le(1, N1), i64_le(2, N2),
    append([[0'a, 0'b, 0, 0], N1, [0'x, 0'y, 0'z, 0], N2], Expected),
    assertion(Bytes == Expected),
    !.

:- end_tests(plawk_blob_consumers).

% --- helpers ---------------------------------------------------------------

bc_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_blobc', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_text(Path, Text) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

build_prog(Dir, Name, BinName, Src, Bin) :-
    directory_file_path(Dir, Name, Prog),
    write_text(Prog, Src),
    directory_file_path(Dir, BinName, Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(std), process(Pid)]),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin_bytes(Bin, Args, Bytes, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    set_stream(S, type(binary)),
    read_stream_to_codes(S, Bytes),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

lps_record(Text, Bytes) :-
    string_codes(Text, Payload),
    length(Payload, Len),
    i64_le(Len, Prefix),
    append(Prefix, Payload, Bytes).

i64_le(V, Bytes) :-
    findall(B, ( between(0, 7, I), B is (V >> (8 * I)) /\ 0xFF ), Bytes).

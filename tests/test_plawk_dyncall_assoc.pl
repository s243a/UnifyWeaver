:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 4, assoc-return surface: a grammar returning a
% list of integer key-value pairs populates a plawk associative array.
%   arr = dyncall@tally($1) as assoc
% Per record the returned [K-V, ...] pairs are inserted (accumulated) into
% arr's i64 table via @wam_object_call_assoc; END arr[key] lookups see the
% result. Named entry; keys are integers OR atoms (an atom key's id is a
% global-registry id, the same keyspace text-mode field slices intern
% into, so for-in reports and "literal" lookups resolve it); values are
% integers. One key kind per table.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% tally(X) returns two pairs: bucket 1 gets X, bucket 2 gets a flat 100.
tally(X, R) :- R = [1-X, 2-100].

% tallyc(X) returns ATOM-keyed pairs: every record tallies under seen,
% and the big/small bucket gets weight 2/1 (text fields arrive as atoms).
tallyc(X, R) :-
    atom_number(X, N),
    ( N > 100 -> R = [big-2, seen-1] ; R = [small-1, seen-1] ).

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

% ATOM keys (JIT roadmap item 4, the string story's remaining container):
% a grammar returning [Atom-V, ...] pairs populates the table under the
% atom's global-registry id -- the same keyspace text-mode field slices
% and blob keys intern into -- so the END for-in report resolves the key
% names like any text-mode table. 50, 200, 300, 7 -> big 4, small 2,
% seen 4 (line order is table slot order, so compare sorted).
test(assoc_atom_keys_populate_and_report, [condition(clang_available)]) :-
    da_dir(Dir),
    directory_file_path(Dir, 'libc.wamo', Wamo),
    write_wam_object([user:tallyc/2], [wamo_entries([tallyc/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@tallyc($1) as assoc }\n\c
         END { for (k in arr) print k, arr[k] }\n", [Wamo]),
    build_run_text(Dir, 'dac', Src, "50\n200\n300\n7\n", Out),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    assertion(Sorted == ["big 4", "seen 4", "small 2"]),
    !.

% An atom-keyed table also answers END string-literal lookups: the
% literal interns to the same registry id the grammar's atom carries.
test(assoc_atom_keys_string_lookup, [condition(clang_available)]) :-
    da_dir(Dir),
    directory_file_path(Dir, 'libc.wamo', Wamo),
    ( exists_file(Wamo)
    -> true
    ;  write_wam_object([user:tallyc/2], [wamo_entries([tallyc/2])], Wamo)
    ),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@tallyc($1) as assoc }\n\c
         END { print arr[\"big\"], arr[\"seen\"] }\n", [Wamo]),
    build_run_text(Dir, 'dacl', Src, "50\n200\n300\n7\n", Out),
    assertion(Out == "4 4\n"),
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

% text-mode variant: the input is text lines rather than binary records
build_run_text(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
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

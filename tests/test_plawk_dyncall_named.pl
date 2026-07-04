:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 3, surface A: dyncall@name(...). One
% DYNLOAD object exposes several named entries (a multi-entry .wamo via
% wamo_entries([...])); dyncall@square($1) / dyncall@cube($1) select an
% entry by name at the call site. The @name is a compile-time token, so
% the shim resolves the entry's label index once at startup (via
% @wam_object_entry_index) and caches the PC -- no per-call dispatch.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar predicates compiled into one multi-entry .wamo (arity = inputs + 1)
square(X, R) :- R is X * X.
cube(X, R)   :- R is X * X * X.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_named).

% dyncall@name(...) parses to its own node, distinct from bare dyncall.
test(named_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n{ print dyncall@square($1) }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(print([dyncall_named(square, [field(1)])]), Actions),
    !.

% Each dyncall@name site is collected as a Name-NArgs entry (deduped,
% sorted); the entry-predicate arity the loader matches is NArgs+1.
test(named_entries_collected) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { s += dyncall@square($1) ; c += dyncall@cube($1) }\n\c
         END { print c - s }\n",
        Program),
    plawk_program_dyncall_named_entries(Program, Entries),
    assertion(Entries == [cube-1, square-1]),
    !.

% The named shim + entry-name string + cached-PC global are emitted, and
% only when a named site exists (performance invariant: bare programs get
% none of this).
test(named_shim_ir_emitted) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { s += dyncall@square($1) }\nEND { print s }\n",
        Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv,
        [wam_vm(10, 10)], IR),
    sub_atom(IR, _, _, _, '@plawk_dyncall_named_square_1'),
    sub_atom(IR, _, _, _, '@plawk_dyncall_pc_square_1'),
    sub_atom(IR, _, _, _, 'square/2'),
    sub_atom(IR, _, _, _, '@wam_object_entry_index'),
    !.

test(bare_program_has_no_named_ir) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"lib.wamo\" }\n\c
         { total += dyncall($1) }\nEND { print total }\n",
        Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv,
        [wam_vm(10, 10)], IR),
    \+ sub_atom(IR, _, _, _, 'plawk_dyncall_named'),
    !.

% Full round trip: one .wamo exposes square/2 and cube/2; the program
% sums each by name over the same input. Distinct names -> distinct sums
% from the SAME loaded object.
test(named_entries_run_from_one_object, [condition(clang_available)]) :-
    dyn_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:square/2, user:cube/2],
        [wamo_entries([square/2, cube/2])], Wamo),
    % squares 4+9=13 into s, cubes 8+27=35 into c; print c-s = 22, a value
    % reachable only if BOTH named entries resolved (a missing entry
    % contributes 0, giving 35 or -13 instead).
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { s += dyncall@square($1) ; c += dyncall@cube($1) }\n\c
         END { print c - s }\n", [Wamo]),
    write_prog(Dir, 'named.plawk', Src),
    directory_file_path(Dir, 'named.plawk', Prog),
    directory_file_path(Dir, 'named_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, [2, 3]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "22\n"),
    !.

% A name no entry exposes yields 0 (the shim's resolve-fail path), not a
% crash or a link error.
test(unknown_named_entry_yields_zero, [condition(clang_available)]) :-
    dyn_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:square/2, user:cube/2],
          [wamo_entries([square/2, cube/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { t += dyncall@nosuch($1) }\nEND { print t }\n", [Wamo]),
    write_prog(Dir, 'miss.plawk', Src),
    directory_file_path(Dir, 'miss.plawk', Prog),
    directory_file_path(Dir, 'miss_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, [2, 3]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "0\n"),
    !.

:- end_tests(plawk_dyncall_named).

% --- helpers ---------------------------------------------------------------

dyn_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_named', Dir),
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

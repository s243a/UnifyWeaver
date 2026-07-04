:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) slice: the plawk DYNLOAD / dyncall surface. A program
% declares BEGIN { DYNLOAD = "file.wamo" } and calls dyncall(args), which
% routes to the runtime-loaded object's entry. The object is loaded once
% at the first dyncall and reused; swapping the .wamo file changes program
% behaviour with no rebuild.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar predicates compiled into .wamo objects (arity = inputs + 1 out)
square(X, R) :- R is X * X.
twice(X, R)  :- R is X * 2.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall).

% dyncall parses to its own node (never the compiled-foreign path) and
% DYNLOAD is a recognized BEGIN directive.
test(dyncall_and_dynload_parse) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"g.wamo\" }\n{ t += dyncall($1) }\nEND { print t }\n",
        program(Begin, Rules, _End)),
    memberchk(begin(BActions), Begin),
    memberchk(set(var('DYNLOAD'), string("g.wamo")), BActions),
    Rules = [rule(_, Actions)],
    memberchk(add(var(t), dyncall([field(1)])), Actions),
    !.

% dyncall is recognized as a scalar-accumulator update over a binary i64
% program (drives emit_wamo_loader in the CLI).
test(dyncall_arity_collected) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"g.wamo\" }\n\c
         { t += dyncall($1) }\nEND { print t }\n",
        Program),
    plawk_program_dyncall_arities(Program, Arities),
    assertion(Arities == [1]),
    plawk_program_dynload_path(Program, Path),
    assertion(Path == "g.wamo"),
    !.

% Full round trip: build a plawk binary that dyncalls a squaring grammar
% over binary i64 records, and check the sum of squares.
test(dyncall_runs_loaded_grammar, [condition(clang_available)]) :-
    dyn_dir(Dir),
    directory_file_path(Dir, 'square.wamo', Wamo),
    write_wam_object([user:square/2], [wamo_entry(square/2)], Wamo),
    build_dyncall_prog(Dir, Wamo, Bin),
    directory_file_path(Dir, 'nums.bin', Input),
    write_i64_records(Input, [3, 4, 5, 10]),        % 9+16+25+100 = 150
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "150\n"),
    !.

% Swap the object file for a different grammar and rerun the SAME binary:
% behaviour changes with no rebuild. This is the point of dyncall.
test(swapped_grammar_changes_result_no_rebuild, [condition(clang_available)]) :-
    dyn_dir(Dir),
    directory_file_path(Dir, 'square.wamo', Wamo),
    directory_file_path(Dir, 'prog_bin', Bin),
    ( exists_file(Bin) -> true ; build_dyncall_prog(Dir, Wamo, Bin) ),
    % overwrite the object with a doubling grammar (different entry name,
    % same input+output arity)
    write_wam_object([user:twice/2], [wamo_entry(twice/2)], Wamo),
    directory_file_path(Dir, 'nums.bin', Input),
    ( exists_file(Input) -> true ; write_i64_records(Input, [3, 4, 5, 10]) ),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "44\n"),                        % 6+8+10+20 = 44
    !.

% dyncall without a DYNLOAD declaration is a compile error (exit 3), not a
% silent miscompile.
test(dyncall_without_dynload_fails_build, [condition(clang_available)]) :-
    dyn_dir(Dir),
    write_prog(Dir, 'no_dynload.plawk',
        "BEGIN { BINFMT = \"i64\" }\n{ t += dyncall($1) }\nEND { print t }\n"),
    directory_file_path(Dir, 'no_dynload.plawk', Prog),
    cli([build, Prog, '-o', '/dev/null'], _, 3),
    !.

:- end_tests(plawk_dyncall).

% --- helpers ---------------------------------------------------------------

dyn_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

% write a plawk program that sums dyncall($1) over binary i64 records
% loaded from the given .wamo path, then build it with the CLI.
build_dyncall_prog(Dir, Wamo, Bin) :-
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { total += dyncall($1) }\nEND { print total }\n", [Wamo]),
    write_prog(Dir, 'prog.plawk', Src),
    directory_file_path(Dir, 'prog.plawk', Prog),
    directory_file_path(Dir, 'prog_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0).

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

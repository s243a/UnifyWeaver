:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 2: binary-data returns, opaque bytes. A
% grammar whose entry binds its output to an Atom (a byte string) is read
% as a byte slice by blob(dyncall(...)) / blob(dyncall_at(...)) via
% @wam_object_call_bytes -- no deserialization, the bytes pass through to
% a print (%.*s). NUL-free by the blob convention.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% atom-returning grammars
echo(A, A).                                  % returns its input atom
greeting(R) :- R = hello.                    % nullary, returns atom hello
kind(X, R) :- ( X > 100 -> R = big ; R = small ).  % if-then-else -> jump

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_blob).

% blob(dyncall(...)) / blob(dyncall_at(...)) parse to their own nodes.
test(blob_forms_parse) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"g.wamo\" }\n{ print blob(dyncall($1)) }\n",
        program(_, [rule(_, A1)], _)),
    memberchk(print([blob_dyncall([field(1)])]), A1),
    plawk_parse_string("{ print blob(dyncall_at($1)) }\n",
        program(_, [rule(_, A2)], _)),
    memberchk(print([blob_dyncall_at(field(1), [])]), A2),
    !.

test(blob_arities_collected) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"g.wamo\" }\n{ print blob(dyncall($1)) }\n",
        Program),
    plawk_program_dyncall_blob_arities(Program, BArities),
    assertion(BArities == [1]),
    !.

% blob(dyncall($1)) reads the grammar's Atom output and prints its bytes:
% echo/2 returns its input atom, so each line's field 1 is echoed.
test(blob_dyncall_prints_bytes, [condition(clang_available)]) :-
    blob_dir(Dir),
    directory_file_path(Dir, 'echo.wamo', Wamo),
    write_wam_object([user:echo/2], [wamo_entry(echo/2)], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n{ print blob(dyncall($1)) }\n", [Wamo]),
    build_prog(Dir, 'pe.plawk', 'pe', Src, Bin),
    directory_file_path(Dir, 'in.txt', Input),
    write_text(Input, "alpha\nbeta\ngamma\n"),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "alpha\nbeta\ngamma\n"),
    !.

% blob(dyncall_at($1)) over a dynamic source: a nullary grammar chosen by a
% filename column returns the atom `hello`.
test(blob_dyncall_at_prints_bytes, [condition(clang_available)]) :-
    blob_dir(Dir),
    directory_file_path(Dir, 'greeting.wamo', Wamo),
    write_wam_object([user:greeting/1], [wamo_entry(greeting/1)], Wamo),
    build_prog(Dir, 'pat.plawk', 'pat',
        "{ print blob(dyncall_at($1)) }\n", Bin),
    directory_file_path(Dir, 'at.txt', Input),
    setup_call_cleanup(
        open(Input, write, S, [encoding(utf8)]),
        ( format(S, "~w~n", [Wamo]), format(S, "~w~n", [Wamo]) ),
        close(S)),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "hello\nhello\n"),
    !.

% An if-then-else grammar (which lowers through `jump`, now in the loadable
% subset) returning a per-branch atom compiles into a .wamo.
test(if_then_else_atom_grammar_compiles) :-
    wam_object_encode([user:kind/2], [wamo_entry(kind/2)], Codes),
    string_codes(Text, Codes),
    sub_string(Text, 0, 4, _, "WAMO"),
    sub_string(Text, _, _, _, "big"),
    sub_string(Text, _, _, _, "small"),
    !.

:- end_tests(plawk_dyncall_blob).

% --- helpers ---------------------------------------------------------------

blob_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_blob', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_text(Path, Text) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

build_prog(Dir, ProgName, BinName, Src, Bin) :-
    directory_file_path(Dir, ProgName, Prog),
    setup_call_cleanup(
        open(Prog, write, S, [encoding(utf8)]),
        write(S, Src),
        close(S)),
    directory_file_path(Dir, BinName, Bin),
    cli([build, Prog, '-o', Bin], _, 0).

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

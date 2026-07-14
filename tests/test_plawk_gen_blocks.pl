:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk generator blocks (PLAWK_GENERATOR_BLOCKS.md), PR 1: the SURFACE.
% `gen { emit E ... } as name` parses to gen_block(name(Name), Body) -- a
% producer whose `emit E` statements will define a non-deterministic relation
% `name/1` callable from a Prolog goal (the producer dual of the query reader).
% The runtime (materialise-then-iterate) is not wired yet, so building a program
% that defines a generator block is a clean, specific compile error rather than
% the generic "outside the multi-pass surface".

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_gen_blocks).

% `gen { emit N ... } as name` parses to gen_block(name(Name), Body); each
% `emit E` is an emit(Expr) action. Integer literals fall back to int(N).
test(gen_block_int_literals_parses) :-
    plawk_parse_string(
        "gen { emit 1; emit 2; emit 3 } as small\n\c
         pass over query(small(X)) { print $1 }\n",
        program_passes([],
            [gen_block(name(small),
                 [emit(int(1)), emit(int(2)), emit(int(3))]),
             pass_query(query(small, ['X']), [print([field(1)])])],
            [])),
    !.

% `emit` accepts the print field-expression grammar: a field ($1) or a string
% literal (carried as string(_), an atom/string to emit).
test(gen_block_field_and_string_emit_parses) :-
    plawk_parse_string(
        "gen { emit $1; emit \"red\" } as g\n",
        program_passes([],
            [gen_block(name(g), Body)],
            [])),
    Body = [emit(field(1)), emit(StrEmit)],
    StrEmit = string(_),
    !.

% The `gen` keyword is distinct from `pass`: an ordinary pass is untouched.
test(pass_unchanged) :-
    plawk_parse_string(
        "pass { print $1 }\n",
        program_passes([],
            [pass([rule(always, [print([field(1)])])])],
            [])),
    !.

% A pure generator with constant integer emits compiles to facts and feeds a
% query pass: `gen { emit 1; emit 2; emit 3 } as small` -> the query prints the
% three solutions (the producer dual, closing the plawk<->Prolog loop).
test(gen_block_int_run, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "gen { emit 1; emit 2; emit 3 } as small\n\c
           pass over query(small(X)) { print $1 }\n",
    build_run(Dir, 'genint', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "1\n2\n3\n"),
    !.

% String emits become atoms (`emit "red"` -> `color(red)`); the query reader
% resolves them to text via its tagged column materialisation.
test(gen_block_string_run, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "gen { emit \"red\"; emit \"green\"; emit \"blue\" } as color\n\c
           pass over query(color(C)) { print $1 }\n",
    build_run(Dir, 'genstr', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "red\ngreen\nblue\n"),
    !.

% A reader guard on the consuming query pass composes with a generator source.
test(gen_block_guarded_consume_run, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "gen { emit 10; emit 20; emit 30 } as weight\n\c
           pass over query(weight(W)) { if ($1 > 10) print $1 }\n",
    build_run(Dir, 'genguard', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "20\n30\n"),
    !.

% A computed emit (a field, not a constant) needs runtime collection -- not yet
% implemented -- so it is a clean not-yet compile error (exit 2).
test(gen_block_computed_emit_not_yet) :-
    gdir(Dir),
    Src = "gen { emit $1 } as g\npass over query(g(X)) { print $1 }\n",
    build_status(Dir, 'gencomp', Src, St),
    assertion(St == 2),
    !.

% A program with no generator block is unaffected (no false trigger).
test(non_gen_program_builds, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "@prolog\nedge(1).\nedge(2).\n@end\npass over query(edge(X)) { print $1 }\n",
    build_status(Dir, 'ok', Src, St),
    assertion(St == 0),
    !.

:- end_tests(plawk_gen_blocks).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_gen_blocks', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_status(Dir, Name, Src, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

% Build a program, then run the resulting binary with Args, capturing stdout.
build_run(Dir, Name, Src, Args, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, Args,
        [stdout(pipe(RS)), stderr(std), process(RPid)]),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

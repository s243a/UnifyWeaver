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

% Building a generator-block program is a clean compile error (exit 2) until
% the runtime lands; the message names the generated relation.
test(gen_block_is_not_yet_error) :-
    gdir(Dir),
    Src = "gen { emit 1; emit 2 } as small\npass over query(small(X)) { print $1 }\n",
    build_status(Dir, 'g', Src, St),
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

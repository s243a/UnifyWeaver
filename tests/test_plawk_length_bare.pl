:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Bare `length` (no argument) is awk shorthand for `length($0)` -- the length of
% the current record. It parses to the same `length(field(0))` AST the
% parenthesised call produces, so it reaches every context `length($0)` already
% did (print, arithmetic, scalar assignment). A non-consuming lookahead keeps
% `length(...)` / `length (...)` function calls and `lengthy` / `length_x`
% ordinary identifiers.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_length_bare).

% --- parsing ----------------------------------------------------------------

% Bare `length` parses to length($0).
test(bare_length_parses) :-
    plawk_parse_string("{ print length }\n",
        program([], [rule(always, [print([length(field(0))])])], [])),
    !.

% `length($0)` still parses to the same AST (unchanged).
test(paren_length_unchanged) :-
    plawk_parse_string("{ print length($0) }\n",
        program([], [rule(always, [print([length(field(0))])])], [])),
    !.

% `lengthy` is an ordinary identifier, not `length` + `y`.
test(lengthy_is_identifier) :-
    plawk_parse_string("{ lengthy = 3; print lengthy }\n",
        program([], [rule(always, [set(var(lengthy), int(3)), print([var(lengthy)])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% Bare `length` prints the current record's byte length.
test(bare_length_prints_record_length, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'bl', "{ print length }\n", "hello\nhi\n", Out),
    assertion(Out == "5\n2\n"), !.

% Bare `length` composes in arithmetic (length + 1).
test(bare_length_in_arithmetic, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'la', "{ print length + 1 }\n", "abc\n", Out),
    assertion(Out == "4\n"), !.

% Bare `length` in a scalar assignment.
test(bare_length_scalar_assign, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ls', "{ n = length; print n }\n", "abcd\n", Out),
    assertion(Out == "4\n"), !.

% `length($0)` still works at runtime (regression).
test(paren_length_runtime, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'lp', "{ print length($0) }\n", "abcde\n", Out),
    assertion(Out == "5\n"), !.

% `length ($1)` (space before the paren) stays a call on $1 (regression).
test(spaced_paren_length_is_call, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'lsp', "{ x = length ($1); print x }\n", "ab cd\n", Out),
    assertion(Out == "2\n"), !.

:- end_tests(plawk_length_bare).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_length_bare', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

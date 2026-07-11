:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Handle-in-scalar: compile(src) / compile_file(path) as EXPRESSIONS
% whose value is the grammar HANDLE (an i64 registry index), stored in
% an ordinary scalar and used as a dyncall_at source:
%
%     { h = compile("[...]") ; total += dyncall_at@sq(h, $1) }
%
% Names the compiled source once instead of repeating it per call site.
% Content dedup makes the per-record re-assignment a registry hit, and
% the handle travels as (null path, handle id) -- the discriminator the
% cache getter already speaks. No runtime changes; parser + codegen only.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_handle_scalar).

% compile(...) in a set position parses to the handle expression;
% a bare identifier in dyncall_at source position parses as a handle
% read. compile_file takes the same expression form.
test(handle_expressions_parse) :-
    plawk_parse_string(
        "{ h = compile(\"[(sq(X, R) :- R is X)]\") ; g = compile_file($2) ; t += dyncall_at@sq(h, $1) + dyncall_at(g, $1) }\nEND { print t }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(set(var(h), compile_handle(string(_))), Actions),
    memberchk(set(var(g), compile_file_handle(field(2))), Actions),
    memberchk(add(var(t), add_i64(
        dyncall_at_named(sq, handle_src(var(h)), [field(1)]),
        dyncall_at(handle_src(var(g)), [field(1)]))), Actions),
    !.

% THE POINT: one compile() of a two-grammar source, named ONCE, called
% by name at two sites through the scalar. Squares 150 + doubles 44
% over 3,4,5,10 -> 194 -- the family payoff without repeating the
% source text.
test(handle_names_family_once, [condition(clang_available)]) :-
    hs_dir(Dir),
    format(string(Src),
        "{ h = compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2), (dbl(X3, R3) :- atom_number(X3, N3), R3 is N3 * 2)]\") ; total += dyncall_at@sq(h, $1) + dyncall_at@dbl(h, $1) }\n\c
         END { print total }\n", []),
    build_run(Dir, 'fam', Src, "3\n4\n5\n10\n", Out),
    assertion(Out == "194\n"),
    !.

% The bare (default-entry) form works through a handle too: squares
% over 3,4,5,10 -> 150.
test(handle_default_entry, [condition(clang_available)]) :-
    hs_dir(Dir),
    format(string(Src),
        "{ h = compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\") ; total += dyncall_at(h, $1) }\n\c
         END { print total }\n", []),
    build_run(Dir, 'def', Src, "3\n4\n5\n10\n", Out),
    assertion(Out == "150\n"),
    !.

% compile_file through a scalar keeps the edit-no-rebuild property: the
% per-record re-assignment re-reads the file, and content dedup makes
% an edited file a fresh compile (150 squares -> 44 doubles, same
% binary).
test(handle_compile_file_edit_no_rebuild, [condition(clang_available)]) :-
    hs_dir(Dir),
    directory_file_path(Dir, 'gram.pl', GramPath),
    write_text(GramPath,
        '[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]'),
    format(string(Src),
        "{ h = compile_file(\"~w\") ; total += dyncall_at(h, $1) }\n\c
         END { print total }\n", [GramPath]),
    directory_file_path(Dir, 'cf.plawk', Prog),
    write_text(Prog, Src),
    directory_file_path(Dir, 'cf_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'nums.txt', Input),
    write_text(Input, "3\n4\n5\n10\n"),
    run_bin(Bin, [Input], Out1, 0),
    assertion(Out1 == "150\n"),
    write_text(GramPath,
        '[(dbl(X3, R3) :- atom_number(X3, N3), R3 is N3 * 2)]'),
    run_bin(Bin, [Input], Out2, 0),
    assertion(Out2 == "44\n"),
    !.

% A handle expression with the cache off is a build error (exit 3) --
% the handle lives in the cache registry.
test(handle_with_cache_off_fails_build, [condition(clang_available)]) :-
    hs_dir(Dir),
    directory_file_path(Dir, 'off.plawk', Prog),
    write_text(Prog,
        "BEGIN { DYNCACHE = \"off\" }\n{ h = compile(\"[(sq(X, R) :- R is X)]\") ; t += dyncall_at(h, $1) }\nEND { print t }\n"),
    cli([build, Prog, '-o', '/dev/null'], _, 3),
    !.

:- end_tests(plawk_handle_scalar).

% --- helpers ---------------------------------------------------------------

hs_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_handle_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_text(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
        write(S, Text), close(S)).

build_run(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    write_text(Prog, Src),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    write_text(Input, InputText),
    run_bin(Bin, [Input], Out, 0).

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

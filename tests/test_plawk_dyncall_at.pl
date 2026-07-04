:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT): the dynamic-source dyncall_at surface. dyncall_at(Source,
% args...) picks the .wamo object by a runtime value (a field or string),
% so a program chooses its grammar per record. Object management is set by
% BEGIN { DYNCACHE = "on" | "mtime" | "off" } (default "on").

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar objects: nullary entry returning a constant (entry(out=A0))
seven(R) :- R is 7.
nine(R)  :- R is 9.
eleven(R) :- R is 11.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_at).

% dyncall_at parses to its own node with the source split from the args,
% and DYNCACHE is a recognized directive.
test(dyncall_at_parses) :-
    plawk_parse_string(
        "BEGIN { DYNCACHE = \"mtime\" }\n{ t += dyncall_at($1) }\nEND { print t }\n",
        program(Begin, Rules, _)),
    memberchk(begin(B), Begin),
    memberchk(set(var('DYNCACHE'), string("mtime")), B),
    Rules = [rule(_, Actions)],
    memberchk(add(var(t), dyncall_at(field(1), [])), Actions),
    !.

test(dyncall_at_arities_and_mode_collected) :-
    plawk_parse_string(
        "BEGIN { DYNCACHE = \"off\" }\n{ t += dyncall_at($1, $2) }\nEND { print t }\n",
        Program),
    plawk_program_dyncall_at_arities(Program, Arities),
    assertion(Arities == [1]),               % one call arg (source excluded)
    plawk_program_dyncache_mode(Program, Mode),
    assertion(Mode == "off"),
    !.

% Default (on) mode: pick the grammar by a filename column; a repeated file
% loads once and is reused.
test(on_mode_selects_grammar_by_field, [condition(clang_available)]) :-
    at_dir(Dir),
    make_grammar(Dir, 'seven.wamo', seven/1),
    make_grammar(Dir, 'nine.wamo', nine/1),
    build_at_prog(Dir, "on", Bin),
    directory_file_path(Dir, 'in.txt', Input),
    write_lines(Input, ['seven.wamo', 'nine.wamo', 'seven.wamo', 'seven.wamo'], Dir),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "30\n"),                % 7+9+7+7
    !.

% off mode: same answer, but each call reloads+frees (no cache).
test(off_mode_reloads_each_call, [condition(clang_available)]) :-
    at_dir(Dir),
    ( exists_file_in(Dir, 'seven.wamo') -> true ; make_grammar(Dir, 'seven.wamo', seven/1) ),
    ( exists_file_in(Dir, 'nine.wamo') -> true ; make_grammar(Dir, 'nine.wamo', nine/1) ),
    build_at_prog(Dir, "off", Bin),
    directory_file_path(Dir, 'in.txt', Input),
    ( exists_file(Input) -> true ; write_lines(Input, ['seven.wamo', 'nine.wamo'], Dir) ),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "30\n"),
    !.

% mtime mode: a cached grammar is reused until the .wamo file changes, then
% the entry is busted and reloaded -- redefinition takes effect with no
% rebuild of the plawk binary.
test(mtime_mode_picks_up_redefinition, [condition(clang_available)]) :-
    at_dir(Dir),
    make_grammar(Dir, 'g.wamo', seven/1),        % g returns 7
    build_at_prog(Dir, "mtime", Bin),
    directory_file_path(Dir, 'one.txt', Input),
    write_lines(Input, ['g.wamo'], Dir),
    run_bin(Bin, [Input], Out1, 0),
    assertion(Out1 == "7\n"),
    sleep(1.1),                                  % ensure a new mtime
    make_grammar(Dir, 'g.wamo', eleven/1),       % redefine g -> 11
    run_bin(Bin, [Input], Out2, 0),
    assertion(Out2 == "11\n"),
    !.

:- end_tests(plawk_dyncall_at).

% --- helpers ---------------------------------------------------------------

at_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_at', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

exists_file_in(Dir, Name) :-
    directory_file_path(Dir, Name, Path),
    exists_file(Path).

make_grammar(Dir, Name, Pred/Arity) :-
    directory_file_path(Dir, Name, Path),
    write_wam_object([user:Pred/Arity], [wamo_entry(Pred/Arity)], Path).

% one input line per grammar-file name (absolute path under Dir)
write_lines(Path, Names, Dir) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        forall(member(Name, Names),
            ( directory_file_path(Dir, Name, Full), format(Out, "~w~n", [Full]) )),
        close(Out)).

% a text-mode program: sum dyncall_at($1) over lines naming grammar files
build_at_prog(Dir, CacheMode, Bin) :-
    format(string(Src),
        "BEGIN { DYNCACHE = \"~w\" }\n{ total += dyncall_at($1) }\nEND { print total }\n",
        [CacheMode]),
    directory_file_path(Dir, 'prog.plawk', Prog),
    setup_call_cleanup(
        open(Prog, write, S, [encoding(utf8)]),
        write(S, Src),
        close(S)),
    directory_file_path(Dir, 'prog_bin', Bin),
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

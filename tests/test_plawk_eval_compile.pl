:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The eval surface (JIT roadmap item 5, the payoff): compile a grammar
% from SOURCE TEXT at runtime, inside the running binary, and call it.
%
%   { total += dyncall_at(compile("[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]"), $1) }
%
% compile(Src) runs the shipped bootstrap-compiler object (the
% self-hosted cgfull, emitted as <bin>.evalc.wamo at build time) on the
% Prolog source text via @wam_object_eval, loads the .wamo bytes it
% emits into a fresh VM, and yields a HANDLE into the dyncall_at cache
% registry -- deduplicated by source text, so a per-record compile of
% the same grammar compiles once and reuses the loaded VM thereafter.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_eval_compile).

% compile(...) in the dyncall_at source position parses to its own node
% and is collected as a compile site (which is what makes the CLI ship
% the compiler object and the codegen emit @plawk_compile).
test(compile_source_parses_and_collects) :-
    plawk_parse_source(
        "{ t += dyncall_at(compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\"), $1) }\nEND { print t }\n",
        Program, _Clauses),
    Program = program(_, Rules, _),
    memberchk(rule(_, Actions), Rules),
    memberchk(add(var(t), dyncall_at(compile_src(string(Src)), [field(1)])),
              Actions),
    sub_string(Src, _, _, _, "sq(X2, R2)"),
    plawk_program_compile_sites(Program, Sites),
    assertion(Sites == [string(Src)]),
    !.

% A plain path source still parses as before (no compile site).
test(plain_path_source_unchanged) :-
    plawk_parse_source(
        "{ t += dyncall_at($2, $1) }\nEND { print t }\n",
        Program, _),
    Program = program(_, Rules, _),
    memberchk(rule(_, Actions), Rules),
    memberchk(add(var(t), dyncall_at(field(2), [field(1)])), Actions),
    plawk_program_compile_sites(Program, Sites),
    assertion(Sites == []),
    !.

% THE PAYOFF: a grammar compiled from source text at runtime, inside
% the running binary. The .plawk program carries only Prolog SOURCE for
% the grammar; the binary compiles it on first use (through the shipped
% bootstrap-compiler .wamo) and sums sq(x) over the input records.
test(compile_runs_runtime_compiled_grammar, [condition(clang_available)]) :-
    ev_dir(Dir),
    build_eval_prog(Dir, Bin),
    % the bootstrap-compiler object ships next to the binary
    atom_concat(Bin, '.evalc.wamo', EvalcWamo),
    assertion(exists_file(EvalcWamo)),
    directory_file_path(Dir, 'nums.txt', Input),
    write_num_lines(Input, [3, 4, 5, 10]),        % 9+16+25+100 = 150
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "150\n"),
    !.

% Two DIFFERENT grammars compiled at runtime coexist in the registry
% (per-source dedup, distinct handles): x*x + 2*x summed over the same
% records. 150 + 44 = 194.
test(two_runtime_grammars_coexist, [condition(clang_available)]) :-
    ev_dir(Dir),
    format(string(Src),
        "{ total += dyncall_at(compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\"), $1) + dyncall_at(compile(\"[(dbl(X3, R3) :- atom_number(X3, N3), R3 is N3 * 2)]\"), $1) }\n\c
         END { print total }\n", []),
    write_prog(Dir, 'two.plawk', Src),
    directory_file_path(Dir, 'two.plawk', Prog),
    directory_file_path(Dir, 'two_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'nums.txt', Input),
    ( exists_file(Input) -> true ; write_num_lines(Input, [3, 4, 5, 10]) ),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "194\n"),
    !.

% compile_file(path) parses to its own node and counts as a compile
% site (needs the shipped compiler object like compile(...)).
test(compile_file_parses_and_collects) :-
    plawk_parse_source(
        "{ t += dyncall_at(compile_file($2), $1) }\nEND { print t }\n",
        Program, _),
    Program = program(_, Rules, _),
    memberchk(rule(_, Actions), Rules),
    memberchk(add(var(t), dyncall_at(compile_file_src(field(2)), [field(1)])),
              Actions),
    plawk_program_compile_sites(Program, Sites),
    assertion(Sites == [field(2)]),
    !.

% compile_file(path): the grammar SOURCE lives in a file the binary
% reads at runtime. Editing the file changes behaviour on the NEXT RUN
% of the SAME binary -- no rebuild -- because content dedup treats the
% edited text as a fresh source (the query/userspace redefinition
% story, by content rather than mtime).
test(compile_file_edit_changes_behavior_no_rebuild,
     [condition(clang_available)]) :-
    ev_dir(Dir),
    directory_file_path(Dir, 'gram.pl', GramPath),
    write_prog(Dir, 'gram.pl',
        '[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]'),
    format(string(Src),
        "{ total += dyncall_at(compile_file(\"~w\"), $1) }\n\c
         END { print total }\n", [GramPath]),
    write_prog(Dir, 'evfile.plawk', Src),
    directory_file_path(Dir, 'evfile.plawk', Prog),
    directory_file_path(Dir, 'evfile_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'nums.txt', Input),
    write_num_lines(Input, [3, 4, 5, 10]),
    run_bin(Bin, [Input], Out1, 0),
    assertion(Out1 == "150\n"),                   % squares: 9+16+25+100
    % swap the grammar source; same binary, no rebuild
    write_prog(Dir, 'gram.pl',
        '[(dbl(X3, R3) :- atom_number(X3, N3), R3 is N3 * 2)]'),
    run_bin(Bin, [Input], Out2, 0),
    assertion(Out2 == "44\n"),                    % doubles: 6+8+10+20
    !.

% Demand-driven subset growth: real Prolog idioms -- committed choice
% (cut), bare backtrackable disjunction -- now compile AT RUNTIME.
% cls2(3): guard fails, second clause, first branch (7); cls2(4):
% first branch fails, BACKTRACKS to the second (1); cls2(5)/cls2(10):
% guard passes, cut commits (100 each). 7+1+100+100 = 208.
test(compile_runs_grammar_with_cut_and_disjunction,
     [condition(clang_available)]) :-
    ev_dir(Dir),
    format(string(Src),
        "{ total += dyncall_at(compile(\"[(cls(X2, R2) :- atom_number(X2, N2), cls2(N2, R2)), (cls2(N3, R3) :- N3 > 4, !, R3 = 100), (cls2(N4, R4) :- (N4 == 3, R4 = 7 ; R4 = 1))]\"), $1) }\n\c
         END { print total }\n", []),
    write_prog(Dir, 'cutdisj.plawk', Src),
    directory_file_path(Dir, 'cutdisj.plawk', Prog),
    directory_file_path(Dir, 'cutdisj_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'nums.txt', Input),
    write_num_lines(Input, [3, 4, 5, 10]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "208\n"),
    !.

% CROSS-RECORD STATE in a runtime-compiled grammar: each record asserts
% its value into the process-global clause store, then findall over
% call/1 (the store consult) reads the whole history back -- a running
% sum computed inside the grammar. 3, 4, 5 -> 3 + 7 + 12 = 22.
test(compile_runs_stateful_grammar, [condition(clang_available)]) :-
    ev_dir(Dir),
    format(string(Src),
        "{ total += dyncall_at(compile(\"[(cnt(X2, R2) :- atom_number(X2, N2), assertz(v(N2)), G = v(Q), findall(Q, call(G), L), sum_list(L, R2))]\"), $1) }\n\c
         END { print total }\n", []),
    write_prog(Dir, 'stateful.plawk', Src),
    directory_file_path(Dir, 'stateful.plawk', Prog),
    directory_file_path(Dir, 'stateful_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'nums3.txt', Input),
    write_num_lines(Input, [3, 4, 5]),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == "22\n"),
    !.

% compile(...) with the cache disabled is a build error (exit 3), not a
% silent miscompile: the grammar handle lives in the cache registry.
test(compile_with_cache_off_fails_build, [condition(clang_available)]) :-
    ev_dir(Dir),
    write_prog(Dir, 'nocache.plawk',
        "BEGIN { DYNCACHE = \"off\" }\n{ t += dyncall_at(compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\"), $1) }\nEND { print t }\n"),
    directory_file_path(Dir, 'nocache.plawk', Prog),
    cli([build, Prog, '-o', '/dev/null'], _, 3),
    !.

:- end_tests(plawk_eval_compile).

% --- helpers ---------------------------------------------------------------

ev_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_eval', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

% the roadmap goal, minus the intermediate variable (the handle dedups
% by source, so the nested form compiles once and reuses thereafter).
% Text-mode fields marshal as ATOMS (awk strings), so the grammar
% converts with atom_number/2 before the arithmetic -- same convention
% as any dyncall grammar fed text fields.
build_eval_prog(Dir, Bin) :-
    format(string(Src),
        "{ total += dyncall_at(compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\"), $1) }\n\c
         END { print total }\n", []),
    write_prog(Dir, 'eval.plawk', Src),
    directory_file_path(Dir, 'eval.plawk', Prog),
    directory_file_path(Dir, 'eval_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0).

write_num_lines(Path, Values) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        forall(member(V, Values), format(Out, "~w~n", [V])),
        close(Out)).

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

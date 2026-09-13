:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_cpp_foreign_nested_cut.pl
%
% P1 regression (Astra, PR #4259): nested cuts corrupt foreign iteration.
%
% foreign_try_next resumes foreign_iters.back(), but before the fix a cut
% (choice_points.resize) pruned a nested foreign choice point WITHOUT pruning
% the parallel foreign_iters side stack. The inner iterator leaked, and a later
% foreign_try_next (resuming the OUTER foreign call) read the leaked INNER
% iterator instead — a wrong answer.
%
% Astra's minimal reproducer (store rows: outer -> a,b ; inner -> c,d):
%
%     pick(Y) :- store(inner,Y), !.
%     q(L)    :- findall(X, (store(outer,X), pick(_)), L).
%
% SWI gives q([a,b]); the compiled C++ gave q([a,_]) before the fix. The fix
% (prune_iters_to_choice_points, called at every choice_points-shrink cut/drain
% site) drops the cut-away iterator in lockstep, so the outer continuation
% resumes the OUTER iterator and the compiled answer is [a,b] again.
%
% store/2 is a D43 store-backed foreign fact source (indexed seek store built
% here with scripts/js_wam/uw_fact_index.js), so this exercises the
% foreign_iters path specifically. run/0 drives q/1 and writes the list; the
% compiled output must equal the SWI oracle ([a,b]).
%
% Skipped automatically when no C++ compiler or node is available.
%
%   swipl -q -g run_tests -t halt tests/test_wam_cpp_foreign_nested_cut.pl

:- module(test_wam_cpp_foreign_nested_cut,
          [test_wam_cpp_foreign_nested_cut/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(filesex), [directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).

% ---------------------------------------------------------------------
% Tooling probes
% ---------------------------------------------------------------------

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ).
cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).
node_ok :-
    catch(( process_create(path(node), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

tooling_ready :- cpp_compiler(_), node_ok.

% ---------------------------------------------------------------------
% The reproducer program (store/2 is a FOREIGN source, not a clause set).
% ---------------------------------------------------------------------

install_program :-
    retractall(user:pick(_)),
    retractall(user:q(_)),
    retractall(user:run),
    assertz(user:(pick(Y) :- store(inner, Y), !)),
    assertz(user:(q(L) :- findall(X, (store(outer, X), pick(_)), L))),
    assertz(user:(run :- q(L), write(L), nl)).

% SWI oracle needs store/2 as ordinary facts (the C++ build gets them from the
% seek store instead). Kept in a private module so they never leak into the set
% of predicates compiled to C++.
oracle_store(outer, a).
oracle_store(outer, b).
oracle_store(inner, c).
oracle_store(inner, d).

% ---------------------------------------------------------------------
% Build the indexed seek store for store/2 from a 4-row JSONL.
% ---------------------------------------------------------------------

build_store(Prefix) :-
    file_directory_name(Prefix, Dir),
    make_directory_path_safe(Dir),
    atom_concat(Prefix, '.jsonl', Jsonl),
    setup_call_cleanup(
        open(Jsonl, write, S),
        ( write(S, '["outer","a"]\n'),
          write(S, '["outer","b"]\n'),
          write(S, '["inner","c"]\n'),
          write(S, '["inner","d"]\n') ),
        close(S)),
    absolute_file_name('../scripts/js_wam/uw_fact_index.js', Index,
                       [relative_to('src/'), access(read)]),
    process_create(path(node), [Index, build, Jsonl, Prefix],
                   [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

make_directory_path_safe(Dir) :-
    ( exists_directory(Dir) -> true ; make_directory(Dir) ).

% ---------------------------------------------------------------------
% Compile + run the C++ project.
% ---------------------------------------------------------------------

compile_project(Prefix, Bin) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_foreign_nested_cut',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    install_program,
    write_wam_cpp_project(
        [user:pick/1, user:q/1, user:run/0, user:store/2],
        [ module_name(foreigncut), emit_mode(interpreter), emit_main(true),
          on_compile_error(throw),
          cpp_wam_fact_sources([source(store/2, indexed(Prefix))]) ],
        Dir),
    !,
    directory_file_path(Dir, cpp, CppDir),
    format(atom(BuildCmd),
        '~w -std=c++17 -O0 ~w/main.cpp ~w/generated_program.cpp \c
         ~w/wam_runtime.cpp -o ~w/foreigncutbin 2>&1',
        [CC, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, BuildOut), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, '~n[foreign nested-cut build failed]~n~w~n',
               [BuildOut]),
        throw(cpp_foreign_cut_build_failed(Status))
    ),
    directory_file_path(CppDir, foreigncutbin, Bin).

cpp_output(Bin, Out) :-
    process_create(Bin, ['run/0'],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    atomic_list_concat([S1, S2], Raw),
    normalize(Raw, Out).

swi_output(Out) :-
    install_program,
    setup_call_cleanup(
        ( assertz((user:store(A, B) :- oracle_store(A, B)), Ref) ),
        with_output_to(string(Raw), ( catch(user:run, _, true) -> true ; true )),
        erase(Ref)),
    normalize(Raw, Out).

% Strip the shim's trailing true/false and all whitespace so only the written
% list survives — same normalisation the cut-semantics suite uses.
normalize(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(noise_line, Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

noise_line("").
noise_line("true").
noise_line("false").

% ---------------------------------------------------------------------
% Test
% ---------------------------------------------------------------------

test_wam_cpp_foreign_nested_cut :-
    run_tests(wam_cpp_foreign_nested_cut).

:- begin_tests(wam_cpp_foreign_nested_cut, [condition(tooling_ready)]).

test(nested_cut_foreign_iteration_matches_swi) :-
    tmp_file(wam_cpp_foreign_cut_store, Base),
    atom_concat(Base, '_store', Prefix),
    build_store(Prefix),
    compile_project(Prefix, Bin),
    cpp_output(Bin, Cpp),
    swi_output(Swi),
    % The oracle must genuinely be [a,b] (guards a broken oracle), and the
    % compiled answer must match it (pre-fix it was [a,_V1] — a wrong answer).
    % normalize/2 yields an atom (atomic_list_concat), so compare against '[a,b]'.
    assertion(Swi == '[a,b]'),
    assertion(Cpp == Swi),
    ( exists_directory('output/test_wam_cpp_foreign_nested_cut')
    -> delete_directory_and_contents('output/test_wam_cpp_foreign_nested_cut')
    ; true ).

:- end_tests(wam_cpp_foreign_nested_cut).

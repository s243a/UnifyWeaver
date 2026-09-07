:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build.pl -- compile examples/pkg_resolver/resolver.pl into a C++ WAM
% project (interpreter emit mode -- instruction array only, no per-predicate
% lowered C++ functions, so generated_program.cpp stays compile-cheap).
% Clauses are re-asserted into `user` (the WAM compiler reads user:clause/2).
% Directives are skipped.
%
%   swipl -q -g main -t halt build.pl -- <resolver.pl> <OutDir>

:- use_module('../../../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).

% load_into_user(+File, +Acc0, -Preds)
%  Assert every clause of File into `user`, skipping directives.  Stops
%  early at a `:- begin_tests(_)` directive so a plunit test block at the
%  tail of a file (test_resolver.pl) is not pulled in.  Returns the running
%  set of Pred/Arity indicators (accumulated across files).
load_into_user(File, Acc0, Preds) :-
    setup_call_cleanup(open(File, read, S), load_terms(S, Acc0, Acc1), close(S)),
    sort(Acc1, Preds).

load_terms(S, Acc, Preds) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  Preds = Acc
    ;   T = (:- begin_tests(_))
    ->  Preds = Acc                       % stop before the plunit block
    ;   T = (:- _)
    ->  load_terms(S, Acc, Preds)
    ;   pred_of_term(T, PA),
        assertz(user:T),
        load_terms(S, [PA|Acc], Preds)
    ).

pred_of_term((Head :- _), P/A) :- !, functor(Head, P, A).
pred_of_term(Head, P/A) :- functor(Head, P, A).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutDir|_]
    ->  true
    ;   OutDir = '.'
    ),
    HERE = '.',
    atom_concat(HERE, '/../resolver.pl',      Resolver),
    atom_concat(HERE, '/../test_resolver.pl', TestResolver),
    atom_concat(HERE, '/driver.pl',           Driver),
    load_into_user(Resolver,     [],     Preds1),
    load_into_user(TestResolver, Preds1, Preds2),
    load_into_user(Driver,       Preds2, Preds0),
    maplist(qualify_user, Preds0, Preds),
    length(Preds, N),
    format("build.pl: compiling ~w predicates (resolver + corpus data + driver)~n", [N]),
    write_wam_cpp_project(Preds,
        [emit_mode(interpreter),
         emit_main(true),
         module_name('uw-pkg-resolver')],
        OutDir),
    format("build.pl: wrote C++ WAM project under ~w/cpp/~n", [OutDir]).

qualify_user(P/A, user:P/A).

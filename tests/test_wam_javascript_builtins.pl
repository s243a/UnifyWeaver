:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_builtins.pl
%
% Compiles JS WAM probes (findall, functor, arg, =.., copy_term, \+,
% call, bagof, setof, aggregate_all, op/3) plus the shared classic-program
% fixtures, runs them under Node, and checks SWI-Prolog answers.
%
%   swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl

:- module(test_wam_javascript_builtins, [test_wam_javascript_builtins/0]).

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3,
               javascript_wam_resolve_emit_mode/2]).
:- use_module('wam_conformance_fixtures',
              [conformance_program/2, conformance_query/4]).

:- dynamic user:probe_findall/0.
:- dynamic user:probe_functor/0.
:- dynamic user:probe_arg/0.
:- dynamic user:probe_univ/0.
:- dynamic user:probe_copy_term/0.
:- dynamic user:probe_naf/0.
:- dynamic user:probe_call/0.
:- dynamic user:probe_bagof/0.
:- dynamic user:probe_setof/0.
:- dynamic user:probe_bagof_group/0.
:- dynamic user:probe_bagof_anon/0.
:- dynamic user:probe_bagof_exist/0.
:- dynamic user:probe_setof_exist/0.
:- dynamic user:probe_bagof_empty/0.
:- dynamic user:probe_setof_mixed/0.
:- dynamic user:probe_setof_group/0.
:- dynamic user:probe_bagof_first/0.
:- dynamic user:probe_bagof_second/0.
:- dynamic user:age/2.
:- dynamic user:probe_agg_count/0.
:- dynamic user:probe_agg_sum/0.
:- dynamic user:probe_agg_bag/0.
:- dynamic user:probe_agg_set/0.
:- dynamic user:color/2.
:- dynamic user:probe_color_det/0.
:- dynamic user:probe_color_enum/0.
:- dynamic user:probe_color_miss/0.
:- dynamic user:probe_sort/0.
:- dynamic user:probe_lists/0.
:- dynamic user:probe_atoms/0.
:- dynamic user:probe_format/0.
:- dynamic user:probe_assoc/0.
:- dynamic user:sum3/2.
:- dynamic user:unwrap/2.
:- dynamic user:gt_float/1.
:- dynamic user:qatom/1.
:- dynamic user:probe_parse_atom/0.
:- dynamic user:probe_term_meta/0.
:- dynamic user:probe_op3/0.
:- dynamic user:probe_parse_likes/0.
:- dynamic user:probe_string_tag/0.
:- dynamic user:probe_string_polish/0.

install_probes :-
    retractall(user:probe_findall),
    retractall(user:probe_functor),
    retractall(user:probe_arg),
    retractall(user:probe_univ),
    retractall(user:probe_copy_term),
    retractall(user:probe_naf),
    retractall(user:probe_call),
    retractall(user:probe_bagof),
    retractall(user:probe_setof),
    retractall(user:probe_bagof_group),
    retractall(user:probe_bagof_anon),
    retractall(user:probe_bagof_exist),
    retractall(user:probe_setof_exist),
    retractall(user:probe_bagof_empty),
    retractall(user:probe_setof_mixed),
    retractall(user:probe_setof_group),
    retractall(user:probe_bagof_first),
    retractall(user:probe_bagof_second),
    retractall(user:age/2),
    retractall(user:probe_agg_count),
    retractall(user:probe_agg_sum),
    retractall(user:probe_agg_bag),
    retractall(user:probe_agg_set),
    retractall(user:color/2),
    retractall(user:probe_color_det),
    retractall(user:probe_color_enum),
    retractall(user:probe_color_miss),
    retractall(user:probe_sort),
    retractall(user:probe_lists),
    retractall(user:probe_atoms),
    retractall(user:probe_format),
    retractall(user:probe_assoc),
    retractall(user:sum3/2),
    retractall(user:unwrap/2),
    retractall(user:gt_float/1),
    retractall(user:qatom/1),
    retractall(user:probe_parse_atom),
    retractall(user:probe_term_meta),
    retractall(user:probe_op3),
    retractall(user:probe_parse_likes),
    retractall(user:probe_string_tag),
    retractall(user:probe_string_polish),
    assertz((user:probe_findall :-
        findall(X, member(X, [1,2,3]), L), write(L), nl, L == [1,2,3])),
    assertz((user:probe_functor :-
        functor(foo(a,b), N, A), write(N), write(' '), write(A), nl,
        N == foo, A == 2)),
    assertz((user:probe_arg :-
        arg(2, foo(a,b), X), write(X), nl, X == b)),
    assertz((user:probe_univ :-
        foo(a,b) =.. L, write(L), nl, L == [foo,a,b])),
    assertz((user:probe_copy_term :-
        copy_term(f(X,X), C), write(C), nl,
        C = f(Y,Y), X \== Y)),
    assertz((user:probe_naf :-
        \+ member(9, [1,2,3]))),
    assertz((user:probe_call :-
        call(member(2, [1,2,3])))),
    assertz(user:age(alice, 30)),
    assertz(user:age(bob, 25)),
    assertz(user:age(carol, 30)),
    assertz((user:probe_bagof :-
        bagof(X, member(X, [1,2,1]), L), write(L), nl, L == [1,2,1])),
    assertz((user:probe_setof :-
        setof(X, member(X, [1,2,1]), L), write(L), nl, L == [1,2])),
    assertz((user:probe_bagof_group :-
        findall(N-Cs, bagof(C, age(N, C), Cs), All),
        write(All), nl,
        All == [alice-[30], bob-[25], carol-[30]])),
    assertz((user:probe_bagof_anon :-
        findall(Cs, bagof(C, age(_, C), Cs), All),
        write(All), nl,
        All == [[30], [25], [30]])),
    assertz((user:probe_bagof_exist :-
        bagof(C, N^age(N, C), Cs), write(Cs), nl, Cs == [30, 25, 30])),
    assertz((user:probe_setof_exist :-
        setof(C, N^age(N, C), Cs), write(Cs), nl, Cs == [25, 30])),
    assertz((user:probe_bagof_empty :-
        \+ bagof(x, fail, _))),
    assertz((user:probe_setof_mixed :-
        setof(X, member(X, [foo, 1, bar, 1, z(a), 3.5]), L),
        write(L), nl,
        L == [1, 3.5, bar, foo, z(a)])),
    assertz((user:probe_setof_group :-
        findall(N-Cs, setof(C, age(N, C), Cs), All),
        write(All), nl,
        All == [alice-[30], bob-[25], carol-[30]])),
    assertz((user:probe_bagof_first :-
        bagof(C, age(N, C), Cs),
        write(N-Cs), nl,
        N-Cs == alice-[30])),
    assertz((user:probe_bagof_second :-
        bagof(C, age(N, C), Cs),
        N \== alice,
        write(N-Cs), nl,
        N-Cs == bob-[25])),
    assertz((user:probe_agg_count :-
        aggregate_all(count, member(_, [1,2,3]), N), write(N), nl, N == 3)),
    assertz((user:probe_agg_sum :-
        aggregate_all(sum(X), member(X, [1,2,3]), N), write(N), nl, N == 6)),
    assertz((user:probe_agg_bag :-
        aggregate_all(bag(X), member(X, [1,2,1]), L), write(L), nl, L == [1,2,1])),
    assertz((user:probe_agg_set :-
        aggregate_all(set(X), member(X, [1,2,1]), L), write(L), nl,
        (L == [1,2] ; L == [2,1]))),
    assertz(user:color(red, 1)),
    assertz(user:color(green, 2)),
    assertz(user:color(blue, 3)),
    assertz((user:probe_color_det :-
        color(green, X), write(X), nl, X == 2, deterministic)),
    assertz((user:probe_color_enum :-
        findall(C-N, color(C, N), L), write(L), nl,
        L == [red-1, green-2, blue-3])),
    assertz((user:probe_color_miss :-
        \+ color(yellow, _))),
    assertz((user:probe_sort :-
        sort([c, a, c, b], S), S == [a, b, c],
        msort([c, a, c, b], M), M == [a, b, c, c],
        keysort([c-1, a-2, c-0], K), K == [a-2, c-1, c-0],
        sort(0, @<, [c, a, c, b], S4), S4 == [a, b, c],
        predsort(compare, [c, a, b], P), P == [a, b, c],
        write(ok), nl)),
    assertz((user:probe_lists :-
        append([1, 2], [3], L), L == [1, 2, 3],
        reverse([1, 2, 3], R), R == [3, 2, 1],
        nth0(1, [a, b, c], X), X == b,
        nth1(1, [a, b, c], Y), Y == a,
        last([1, 2, 3], Z), Z == 3,
        sum_list([1, 2, 3], Sum), Sum == 6,
        max_list([1, 8, 3], Mx), Mx == 8,
        min_list([1, 8, 3], Mn), Mn == 1,
        list_to_set([c, a, c, b], Set), Set == [c, a, b],
        select(b, [a, b, c], Rest), Rest == [a, c],
        include(atom, [a, 1, b], In), In == [a, b],
        exclude(atom, [a, 1, b], Ex), Ex == [1],
        write(ok), nl)),
    assertz((user:probe_atoms :-
        atom_concat(foo, bar, C), C == foobar,
        atom_length(hello, N), N == 5,
        atom_chars(ab, Ch), Ch == [a, b],
        atom_codes(ab, Co), Co == [97, 98],
        char_code(a, Cc), Cc == 97,
        sub_atom(hello, 1, 3, After, Sub), After == 1, Sub == ell,
        atom_string(foo, Str), atom_length(Str, 3),
        number_codes(12, Nc), Nc == [49, 50],
        number_string(12, Ns), atom_concat('', Ns, '12'),
        split_string('a,b,c', ',', '', Parts),
        Parts = [P1, P2, P3],
        atom_string(a, P1), atom_string(b, P2), atom_string(c, P3),
        string_concat(foo, bar, SC), atom_concat('', SC, foobar),
        upcase_atom(hi, U), U == 'HI',
        downcase_atom('HI', D), D == hi,
        write(ok), nl)),
    assertz((user:probe_format :-
        format(atom(A), '~w-~d', [hi, 2]),
        A == 'hi-2',
        format('~w ~d~n', [ok, 7]),
        tab(2), writeln(done))),
    assertz((user:probe_assoc :-
        empty_assoc(A0),
        put_assoc(b, A0, 2, A1),
        put_assoc(a, A1, 1, A2),
        get_assoc(a, A2, V), V == 1,
        assoc_to_keys(A2, Ks), Ks == [a, b],
        assoc_to_list(A2, AL), AL == [a-1, b-2],
        list_to_assoc([a-1, b-2], A3),
        get_assoc(b, A3, W), W == 2,
        write(ok), nl)),
    assertz((user:sum3(L, S) :- sum_list(L, S))),
    assertz(user:unwrap(foo(X, bar(b), 3), X)),
    assertz((user:gt_float(X) :- X > 3.0)),
    assertz(user:qatom('hello world')),
    assertz((user:probe_parse_atom :-
        read_term_from_atom('[1,2,3]', L), L == [1,2,3],
        read_term_from_atom('foo(a, bar(b), 3)', T), T == foo(a, bar(b), 3),
        read_term_from_atom('3.14', F), F > 3.0,
        read_term_from_atom('-2', N), N == -2,
        read_term_from_atom('[a,b|c]', PL), PL == [a, b|c],
        read_term_from_atom('1+2', SumT), SumT == +(1, 2),
        atom_to_term('bar(X)', U, B), U = bar(hello), B == ['X'=hello],
        write(ok), nl)),
    assertz((user:probe_term_meta :-
        term_variables(f(X, Y, X), L), L == [X, Y],
        term_variables(foo(a), Empty), Empty == [],
        numbervars(g(A, B), 0, E), E == 2,
        A == '$VAR'(0), B == '$VAR'(1),
        f(P, Q) =@= f(R, S),
        f(T, T) \=@= f(U, V),
        \+ (f(T, T) =@= f(U, V)),
        foo(a) =@= foo(a),
        write(ok), nl)),
    assertz((user:probe_op3 :-
        op(700, xfx, likes),
        read_term_from_atom('alice likes bob', T),
        T == likes(alice, bob),
        op(900, fy, please),
        read_term_from_atom('please hello', P),
        P == please(hello),
        op(400, xf, km),
        read_term_from_atom('3 km', U),
        U == km(3),
        op(700, xfx, [loves, adores]),
        read_term_from_atom('alice loves bob', T2),
        T2 == loves(alice, bob),
        write(ok), nl)),
    assertz((user:probe_parse_likes :-
        read_term_from_atom('alice likes bob', T),
        T == likes(alice, bob),
        write(ok), nl)),
    assertz((user:probe_string_tag :-
        atom_string(a, S), string(S), \+ atom(S),
        string_to_atom(S, A0), A0 == a,
        split_string('a,b,c', ',', '', Parts),
        Parts = [P1, P2, P3],
        string(P1), string(P2), string(P3),
        atom_string(a, EA), atom_string(b, EB), atom_string(c, EC),
        P1 == EA, P2 == EB, P3 == EC,
        string_concat(x, y, Z), string(Z),
        atom_string(xy, EZ), Z == EZ,
        atom_string(foo, SFoo),
        sort([foo, SFoo, 1, bar], Ord),
        Ord == [1, SFoo, bar, foo],
        write(ok), nl)),
    assertz((user:probe_string_polish :-
        atom_string(abc, Sabc), string_length(Sabc, N), N == 3,
        string_length(abc, N2), N2 == 3,
        string_length(123, N3), N3 == 3,
        atom_string(ab, SAB),
        writeq([SAB, foo]), nl,
        writeq('hello world'), nl,
        writeq(foo), nl,
        writeq([]), nl,
        atom_string(ab, S2), write(S2), nl,
        atom_string(x, SX),
        format('~q', [SX]), nl,
        format('~q', [[SX, y]]), nl,
        write(ok), nl)).

probe_preds([
    user:probe_findall/0,
    user:probe_functor/0,
    user:probe_arg/0,
    user:probe_univ/0,
    user:probe_copy_term/0,
    user:probe_naf/0,
    user:probe_call/0,
    user:probe_bagof/0,
    user:probe_setof/0,
    user:probe_bagof_group/0,
    user:probe_bagof_anon/0,
    user:probe_bagof_exist/0,
    user:probe_setof_exist/0,
    user:probe_bagof_empty/0,
    user:probe_setof_mixed/0,
    user:probe_setof_group/0,
    user:probe_bagof_first/0,
    user:probe_bagof_second/0,
    user:age/2,
    user:probe_agg_count/0,
    user:probe_agg_sum/0,
    user:probe_agg_bag/0,
    user:probe_agg_set/0,
    user:color/2,
    user:probe_color_det/0,
    user:probe_color_enum/0,
    user:probe_color_miss/0,
    user:probe_sort/0,
    user:probe_lists/0,
    user:probe_atoms/0,
    user:probe_format/0,
    user:probe_assoc/0,
    user:probe_parse_atom/0,
    user:probe_term_meta/0,
    user:probe_op3/0,
    user:probe_string_tag/0,
    user:probe_string_polish/0
]).

:- dynamic user:ctw_js/0.

install_conformance_wrappers(Map) :-
    retractall(user:ctw_js),
    findall(q(K,A,E), conformance_query(_, K, A, E), Queries),
    synth(Queries, 1, [], Map).

synth([], _, Acc, Map) :- reverse(Acc, Map).
synth([q(K,A,E)|Rest], I, Acc, Map) :-
    atomic_list_concat([ctw_js_, I], WName),
    atomic_list_concat([PredName|_], '/', K),
    atom_string(Pred, PredName),
    Goal =.. [Pred|A],
    assertz(user:(WName :- Goal)),
    I1 is I + 1,
    synth(Rest, I1, [ctw(K,A,E,WName)|Acc], Map).

conformance_preds(Preds) :-
    findall(P, (conformance_program(_, Ps), member(P, Ps)), Preds).

run_node(Dir, Key, Exit, Out) :-
    run_node_args(Dir, [Key], Exit, Out).

run_node_args(Dir, Args, Exit, Out) :-
    directory_file_path(Dir, 'js', JsDir),
    process_create(path(node), ['generated_program.js'|Args],
        [cwd(JsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(Exit)),
    atomic_list_concat([OS, ES], Out).

node_succeeded(Out) :-
    split_string(Out, "\n", " \t\r", Lines0),
    exclude([L]>>(L == ""), Lines0, Lines),
    last(Lines, Last),
    (Last == "true" ; Last == "true\n").

compile_js_project(Preds, Dir) :-
    Dir = 'output/js_wam_builtin_probes',
    make_directory_path(Dir),
    write_wam_javascript_project(Preds, [emit_mode(interpreter)], Dir).

test_wam_javascript_builtins :-
    run_tests(js_wam_builtins).

:- begin_tests(js_wam_builtins).

test(compile_and_probes, [setup(install_probes)]) :-
    probe_preds(Probes),
    conformance_preds(CPreds),
    install_conformance_wrappers(Map),
    findall(W/0, member(ctw(_,_,_,W), Map), Wrappers),
    append([Probes, CPreds, Wrappers], All),
    compile_js_project(All, Dir),
    forall(member(user:Name/0, Probes),
           (   format(atom(Key), '~w/0', [Name]),
               atom_string(Key, KeyStr),
               run_node(Dir, KeyStr, Exit, Out),
               format('PROBE ~w exit=~w~n~w~n', [Name, Exit, Out]),
               assertion(Exit =:= 0),
               assertion(node_succeeded(Out))
           )),
    forall(member(ctw(K,A,E,W), Map),
           (   format(atom(Key), '~w/0', [W]),
               atom_string(Key, KeyStr),
               run_node(Dir, KeyStr, _Exit, Out),
               (   E == true
               ->  (   node_succeeded(Out)
                   ->  true
                   ;   format('CONFORMANCE FAIL ~w ~w expected true got ~w~n',
                              [K, A, Out]),
                       fail
                   )
               ;   (   node_succeeded(Out)
                   ->  format('CONFORMANCE FAIL ~w ~w expected false got ~w~n',
                              [K, A, Out]),
                       fail
                   ;   true
                   )
               )
           )).

test(cli_structured_args, [setup(install_probes)]) :-
    % SWI oracle for the same shapes the CLI parser must intern.
    assertion((sum_list([1, 2, 3], S0), S0 == 6)),
    assertion((foo(a, bar(b), 3) = foo(X0, bar(b), 3), X0 == a)),
    assertion(3.14 > 3.0),
    Dir = 'output/js_wam_parser_cli',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:sum3/2, user:unwrap/2, user:gt_float/1, user:qatom/1],
        [emit_mode(interpreter)], Dir),
    run_node_args(Dir, ['sum3/2', '[1,2,3]'], ListExit, ListOut),
    assertion(ListExit =:= 0),
    assertion(node_succeeded(ListOut)),
    assertion(sub_string(ListOut, _, _, _, "A2 = 6")),
    run_node_args(Dir, ['unwrap/2', 'foo(a,bar(b),3)'], CompExit, CompOut),
    assertion(CompExit =:= 0),
    assertion(node_succeeded(CompOut)),
    assertion(sub_string(CompOut, _, _, _, "A2 = a")),
    run_node_args(Dir, ['gt_float/1', '3.14'], FloatExit, FloatOut),
    assertion(FloatExit =:= 0),
    assertion(node_succeeded(FloatOut)),
    run_node_args(Dir, ['qatom/1', '\'hello world\''], QExit, QOut),
    assertion(QExit =:= 0),
    assertion(node_succeeded(QOut)).

test(emitted_op_decls, [setup(install_probes)]) :-
    Dir = 'output/js_wam_op_decls',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:probe_parse_likes/0],
        [emit_mode(interpreter),
         javascript_wam_ops([op(700, xfx, likes)])],
        Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "install_declared_ops")),
    assertion(sub_string(Code, _, _, _, "likes")),
    run_node(Dir, 'probe_parse_likes/0', Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)).

test(string_polish_output, [setup(install_probes)]) :-
    Dir = 'output/js_wam_string_polish',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:probe_string_polish/0],
        [emit_mode(interpreter)], Dir),
    run_node(Dir, 'probe_string_polish/0', Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, '["ab",foo]')),
    assertion(sub_string(Out, _, _, _, "'hello world'")),
    assertion(sub_string(Out, _, _, _, "foo")),
    assertion(sub_string(Out, _, _, _, '["x",y]')),
    split_string(Out, "\n", "", Lines),
    assertion(member("ab", Lines)),
    assertion(member("\"x\"", Lines)).

:- end_tests(js_wam_builtins).

% ---------------------------------------------------------------------------
% Tier-2 lowered / mixed emit modes (interpreter suite above is unchanged)
% ---------------------------------------------------------------------------

:- dynamic user:hello/1.
:- dynamic user:probe_lowered_hello/0.

install_lowered_probes :-
    retractall(user:hello/1),
    retractall(user:probe_lowered_hello),
    retractall(user:color/2),
    retractall(user:probe_color_det),
    retractall(user:age/2),
    assertz(user:hello(world)),
    assertz((user:probe_lowered_hello :-
        hello(world), deterministic)),
    assertz(user:color(red, 1)),
    assertz(user:color(green, 2)),
    assertz(user:color(blue, 3)),
    assertz((user:probe_color_det :-
        color(green, X), write(X), nl, X == 2, deterministic)),
    assertz(user:age(alice, 30)),
    assertz(user:age(bob, 25)).

read_generated_js(Dir, Text) :-
    directory_file_path(Dir, 'js', JsDir),
    directory_file_path(JsDir, 'generated_program.js', Path),
    read_file_to_string(Path, Text, []).

:- begin_tests(js_wam_lowered).

test(emit_mode_resolver) :-
    javascript_wam_resolve_emit_mode([], interpreter),
    javascript_wam_resolve_emit_mode([emit_mode(interpreter)], interpreter),
    javascript_wam_resolve_emit_mode([emit_mode(functions)], functions),
    javascript_wam_resolve_emit_mode([emit_mode(mixed([color/2, hello/1]))],
                                    mixed([color/2, hello/1])),
    catch(javascript_wam_resolve_emit_mode([emit_mode(bogus)], _),
          error(domain_error(wam_javascript_emit_mode, bogus), _),
          true).

test(functions_mode_direct_function, [setup(install_lowered_probes)]) :-
    Dir = 'output/js_wam_lowered_functions',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:probe_lowered_hello/0,
         user:color/2, user:probe_color_det/0],
        [emit_mode(functions)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_color_2")),
    assertion(sub_string(Code, _, _, _, "return lowered_hello_1(shared_program, state) === true")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_hello_1()")),
    run_node_args(Dir, ['hello/1', 'world'], HelloExit, HelloOut),
    assertion(HelloExit =:= 0),
    assertion(node_succeeded(HelloOut)),
    run_node(Dir, 'probe_lowered_hello/0', DetExit, DetOut),
    assertion(DetExit =:= 0),
    assertion(node_succeeded(DetOut)),
    run_node(Dir, 'probe_color_det/0', ColorExit, ColorOut),
    assertion(ColorExit =:= 0),
    assertion(node_succeeded(ColorOut)),
    assertion(sub_string(ColorOut, _, _, _, "2")).

test(mixed_lowers_only_named, [setup(install_lowered_probes)]) :-
    Dir = 'output/js_wam_lowered_mixed',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:color/2, user:age/2],
        [emit_mode(mixed([color/2]))], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_color_2")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_age_2")),
    assertion(sub_string(Code, _, _, _, "Runtime.run_predicate(shared_program")),
    run_node_args(Dir, ['color/2', 'green'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "2")).

test(interpreter_mode_has_no_lowered, [setup(install_lowered_probes)]) :-
    Dir = 'output/js_wam_lowered_interpreter',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:color/2],
        [emit_mode(interpreter)], Dir),
    read_generated_js(Dir, Code),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_")),
    assertion(sub_string(Code, _, _, _, "Runtime.run_predicate(shared_program")).

:- end_tests(js_wam_lowered).

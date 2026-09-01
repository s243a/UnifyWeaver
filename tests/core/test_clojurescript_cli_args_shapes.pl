:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_clojurescript_cli_args_shapes.pl
%
% Compile-shape suite for the Prolog constructs that
% `examples/cli_args/cli_args.pl` actually uses, pushed through the CLOJURE /
% CLOJURESCRIPT pattern targets -- the port of the machinery
% tests/core/test_typescript_cli_args_shapes.pl pins for the TypeScript lane.
%
% It has three halves:
%
%   1. COMPILE tests, one per shape, asserting the lowering is the one the A3
%      design calls for (the tuple, the sentinel, the tagged map, the fact-table
%      inline, the guard placement, the clause chain).
%
%   2. EXECUTION tests, `condition(nbb_available)`-gated, that RUN the emitted
%      ClojureScript under nbb and compare against SWI running the same clauses.
%      A substring assertion cannot tell correct output from output the runtime
%      refuses to load -- that was A3's headline lesson in the TS lane, where
%      `return const arg2 = ...;;` passed every has/2 check for as long as it
%      existed. Here the analogue of `node --check` is `nbb` itself: it reads,
%      analyses and evaluates the whole namespace at load time, so an unbalanced
%      form, an unresolved symbol (the failure mode a missing `(declare ...)`
%      produces) or a misplaced `recur` all surface before any argv is seen.
%
%   3. REFUSAL tests for what deliberately stays unsupported, so a silent wrong
%      answer cannot replace a loud refusal without a test noticing.
%
% See docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md for the full catalogue and
% for the four representation decisions this suite pins (tuple / compound /
% equality / sentinel).
%
% Run: swipl -q -g test_clojurescript_cli_args_shapes -t halt
%            tests/core/test_clojurescript_cli_args_shapes.pl

:- module(test_clojurescript_cli_args_shapes, [test_clojurescript_cli_args_shapes/0]).
:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module('../../src/unifyweaver/targets/clojurescript_target').
:- use_module('../../src/unifyweaver/targets/clojure_target', []).

test_clojurescript_cli_args_shapes :-
    run_tests([clojurescript_cli_args_shapes]).

:- begin_tests(clojurescript_cli_args_shapes).

has(Code, Substr)   :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

nbb_available :-
    catch(( process_create(path(nbb), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

% ============================================================================
% Helpers
% ============================================================================

%% a3_clauses(+Pred/Arity, -Clauses)
a3_clauses(Pred/Arity, Clauses) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses).

%% a3_whole(+Pred/Arity, -Code) — isolate the A3 whole-program lowering.
a3_whole(Pred/Arity, Code) :-
    a3_clauses(Pred/Arity, Clauses),
    Clauses \= [],
    once(clojure_target:native_clj_whole(Pred/Arity, Clauses, Code)).

%% a3_structural(+Pred/Arity, -Code) / a3_general(+Pred/Arity, -Code)
a3_structural(Pred/Arity, Code) :-
    a3_clauses(Pred/Arity, Clauses), Clauses \= [],
    once(clojure_target:native_clj_structural(Pred/Arity, Clauses, Code)).

a3_general(Pred/Arity, Code) :-
    a3_clauses(Pred/Arity, Clauses), Clauses \= [],
    once(clojure_target:native_clj_general(Pred/Arity, Clauses, Code)).

%% a3_module(+Roots, -Code) — the whole dependency closure, as one CLJS namespace.
a3_module(Roots, Code) :-
    findall(pred(N, A, facts), member(N/A, Roots), Preds),
    clojurescript_target:compile_module(
        Preds,
        [namespace('generated.shapes'), include_dependencies(true), runtime(nbb)],
        Code).

%% a3_nbb_run(+Code, +Driver, -Out)
%  Write Code (a whole namespace) plus Driver (expressions appended after it) to
%  a temp .cljs and run it under nbb, returning trimmed stdout. nbb LOADS the
%  namespace before evaluating the driver, so this doubles as the load check.
a3_nbb_run(Code, Driver, Out) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.cljs', File),
    setup_call_cleanup(
        ( open(File, write, W), format(W, '~w~n~w~n', [Code, Driver]), close(W) ),
        a3_nbb_exec(File, Out),
        catch(delete_file(File), _, true)).

a3_nbb_exec(File, Out) :-
    process_create(path(nbb), [File],
                   [stdout(pipe(O)), stderr(null), process(P)]),
    read_string(O, _, Str), close(O), process_wait(P, Status),
    Status == exit(0),
    normalize_space(atom(Out), Str).

%% a3_nbb_loads(+Code)
%  The `node --check` analogue: nbb reads, analyses and evaluates the namespace.
a3_nbb_loads(Code) :-
    a3_nbb_run(Code, '(println "load-ok")', Out),
    Out == 'load-ok'.

%% a3_serializer(-Src)
%  A canonical printer, defined in the DRIVER rather than assumed of pr-str, so
%  the comparison against SWI is exact and does not depend on whether a value
%  came back as a vector or a seq (Clojure prints those differently and compares
%  them equal). Mirrored by a3_ser/2 on the Prolog side.
a3_serializer(
'(declare uw-ser)
(defn uw-ser [x]
  (cond
    (true? x) "true"
    (false? x) "false"
    (string? x) (str "\'" x "\'")
    (map? x) (str "f(" (:$ x) "," (clojure.string/join "," (mapv uw-ser (:args x))) ")")
    (sequential? x) (str "[" (clojure.string/join "," (mapv uw-ser x)) "]")
    :else (str x)))').

%% a3_ser(+Term, -String) — the Prolog side of the same canonical printer.
a3_ser(V, "_") :- var(V), !.
a3_ser(true,  "true")  :- !.
a3_ser(false, "false") :- !.
a3_ser([], "[]") :- !.
a3_ser(L, S) :- is_list(L), !,
    maplist(a3_ser, L, Es), atomic_list_concat(Es, ',', I),
    format(string(S), "[~w]", [I]).
a3_ser(N, S) :- number(N), !, format(string(S), "~w", [N]).
a3_ser(A, S) :- atom(A), !, format(string(S), "'~w'", [A]).
a3_ser(Str, S) :- string(Str), !, format(string(S), "'~w'", [Str]).
a3_ser(T, S) :- compound(T), !,
    T =.. [F|Args], maplist(a3_ser, Args, Es), atomic_list_concat(Es, ',', I),
    format(string(S), "f(~w,~w)", [F, I]).

%% a3_call_expr(+Fn, +Args, -Expr) — a Clojure call with a3_lit/2 arguments.
a3_call_expr(Fn, Args, Expr) :-
    maplist(a3_lit, Args, Es),
    atomic_list_concat(Es, ' ', I),
    format(string(Expr), "(~w ~w)", [Fn, I]).

%% a3_lit(+Term, -ClojureLiteral) — a driver-side input value.
a3_lit(true,  "true")  :- !.
a3_lit(false, "false") :- !.
a3_lit([], "[]") :- !.
a3_lit(L, S) :- is_list(L), !,
    maplist(a3_lit, L, Es), atomic_list_concat(Es, ' ', I), format(string(S), "[~w]", [I]).
a3_lit(N, S) :- number(N), !, format(string(S), "~w", [N]).
a3_lit(T, S) :- compound(T), !,
    T =.. [F|Args], maplist(a3_lit, Args, Es), atomic_list_concat(Es, ' ', I),
    format(string(S), '{:$ "~w" :args [~w]}', [F, I]).
a3_lit(A, S) :- atom(A), !, format(string(S), '"~w"', [A]).
a3_lit(Str, S) :- string(Str), !, format(string(S), '"~w"', [Str]).

%% a3_fn(+Pred/Arity, -Name)
%  The name the compiler ACTUALLY emits for this predicate, asked of the
%  compiler rather than hardcoded -- a module can mix the two lowerings and they
%  spell a function differently, so a test that guesses the name tests nothing.
a3_fn(Pred/Arity, Name) :-
    clojure_target:clj_emitted_name(Pred, Arity, Name).

%% a3_oracle(+Roots, +Pred/Arity, +Goal, +InArgs, +OutTerm)
%  Compile the dependency closure of Roots, run the emitted function on InArgs
%  under nbb, and assert the answer equals what SWI's own `Goal` produces.
a3_oracle(Roots, Spec, Goal, InArgs, OutTerm) :-
    a3_module(Roots, Code),
    a3_serializer(Ser),
    a3_fn(Spec, Fn),
    a3_call_expr(Fn, InArgs, Call),
    format(string(Driver), '~w~n(println (uw-ser ~w))', [Ser, Call]),
    a3_nbb_run(Code, Driver, Got),
    ( call(user:Goal) -> a3_ser(OutTerm, Want) ; Want = "<swi-failed>" ),
    atom_string(Got, GotS),
    ( GotS == Want -> true
    ; format(user_error, "~n  nbb: ~w~n  swi: ~w~n", [GotS, Want]), fail ).

%% a3_setup — every shape this suite uses, asserted into `user`.
%  A single setup for the whole suite: the analyses are call-graph fixpoints, so
%  keeping every shape visible at once is also a test that they do not interfere.
a3_setup :-
    a3_teardown,
    % --- G-A3-1: deterministic text builtins -------------------------------
    assertz(user:(s_len(S, L) :- string_length(S, L))),
    assertz(user:(s_cat(A, B, C) :- string_concat(A, B, C))),
    assertz(user:(s_from(S, Start, Sub) :-
        string_length(S, L), Len is L - Start, sub_string(S, Start, Len, 0, Sub))),
    assertz(user:(s_chars(S, Cs) :- string_chars(S, Cs))),
    assertz(user:(s_build(Cs, S) :- string_chars(S, Cs))),
    %  The COMPOSING direction, in the shape the direction rule is written for:
    %  the known side (Kept) comes from a PRECEDING GOAL, not from a head
    %  argument. This is `strip_brackets/2`'s shape.
    assertz(user:(s_round(In, Out) :-
        string_chars(In, Cs), drop_br(Cs, Kept), string_chars(Out, Kept))),
    assertz(user:(c_code(C, X) :- char_code(C, X))),
    % --- G-A3-2/3: guards, control flow in a guard --------------------------
    assertz(user:(alpha(C) :-
        char_code(C, X), ( X >= 0'a, X =< 0'z -> true ; X >= 0'A, X =< 0'Z ))),
    % --- G-A3-5/6: guard placement, semidet with arity > 1 ------------------
    assertz(user:(starts_w(S, P) :-
        string_length(S, L), string_length(P, N), L >= N,
        sub_string(S, 0, N, _, Sub), Sub == P)),
    % --- G-A3-6: cross-predicate calls, by callee output count --------------
    assertz(user:(uses_test(T) :- starts_w(T, "--"))),
    assertz(user:(uses_neg(T) :- \+ starts_w(T, "--"))),
    assertz(user:(uses_one(S, Out) :- s_from(S, 2, Out))),
    assertz(user:(uses_two(T, K, V) :- split_kv(T, K, V))),
    % --- G-A3-9: two outputs ------------------------------------------------
    assertz(user:(split_kv(T, K, V) :-
        string_length(T, L), Half is L // 2,
        sub_string(T, 0, Half, _, K), Rest is L - Half,
        sub_string(T, Half, Rest, 0, V))),
    assertz(user:sum_len([], A, B, A, B)),
    assertz(user:(sum_len([X|Xs], A0, B0, A, B) :-
        A1 is A0 + X, B1 is B0 + 1, sum_len(Xs, A1, B1, A, B))),
    % --- G-A3-10: if-then-else in a recursive body --------------------------
    assertz(user:acc_pos([], A, A)),
    assertz(user:(acc_pos([X|Xs], A0, A) :-
        ( X > 0 -> A1 is A0 + X ; A1 = A0 ), acc_pos(Xs, A1, A))),
    assertz(user:idx_of([], _T, _I, -1)),
    assertz(user:(idx_of([C|Cs], T, I, Ix) :-
        ( C == T -> Ix = I ; I1 is I + 1, idx_of(Cs, T, I1, Ix) ))),
    % --- G-A3-12/13: compounds and the boolean atoms ------------------------
    assertz(user:(mk_some(V, some(V)))),
    assertz(user:(un_some(some(V), V))),
    assertz(user:(mk_bool(V, B) :- ( V == "off" -> B = false ; B = true ))),
    assertz(user:(eq_terms(A, B, R) :- ( A == B -> R = true ; R = false ))),
    % --- G-A3-16: a pair walk ----------------------------------------------
    assertz(user:(pair_get([K-V|Rest], Key, Val) :-
        ( K == Key -> Val = V ; pair_get(Rest, Key, Val) ))),
    assertz(user:put_kv([], K, V, [K-V])),
    assertz(user:(put_kv([K0-V0|R], K, V, Out) :-
        ( K0 == K -> Out = [K0-V|R] ; Out = [K0-V0|R1], put_kv(R, K, V, R1) ))),
    % --- G-A3-18: the sentinel, in condition and body position --------------
    assertz(user:(kind_of(Opts, K, Kind) :-
        ( pair_get(Opts, K, Kd) -> Kind = Kd ; Kind = other ))),
    assertz(user:(body_lookup(Opts, K, R) :- pair_get(Opts, K, V), R = some(V))),
    % --- G-A3-19: a ground-fact CONSTANT TABLE ------------------------------
    assertz(user:globals(["state"-string, "name"-string])),
    assertz(user:(is_global(K) :- globals(G), pair_get(G, K, _))),
    assertz(user:protos(["toString"])),
    assertz(user:protos(["valueOf"])),
    % --- G-A3-20: list-building recursion, deferred binding -----------------
    assertz(user:drop_br([], [])),
    assertz(user:(drop_br([C|Cs], Kept) :-
        ( ( C == '[' ; C == ']' ) -> Kept = Kept1 ; Kept = [C|Kept1] ),
        drop_br(Cs, Kept1))),
    % --- arity overloading (name mangling) ----------------------------------
    assertz(user:(ovl(X, Y) :- ovl(X, 1, Y))),
    assertz(user:(ovl(X, N, Y) :- Y is X + N)),
    % --- refusal shapes ------------------------------------------------------
    assertz(user:(calls_unknown(X, Y) :- no_such_pred(X, Y))),
    assertz(user:(enumerates(K) :- protos(K))).

a3_teardown :-
    forall(member(P/A, [s_len/2, s_cat/3, s_from/3, s_chars/2, s_build/2, c_code/2,
                        alpha/1, starts_w/2, uses_test/1, uses_neg/1, uses_one/2,
                        uses_two/3, split_kv/3, sum_len/5, acc_pos/3, idx_of/4,
                        mk_some/2, un_some/2, mk_bool/2, eq_terms/3, pair_get/3,
                        put_kv/4, kind_of/3, body_lookup/3, globals/1, is_global/1,
                        protos/1, drop_br/2, ovl/2, ovl/3, calls_unknown/2,
                        enumerates/1]),
           ( functor(H, P, A), retractall(user:H) )),
    % The output/fallibility analyses are cached on a signature of the reachable
    % clause set, so a stale cache cannot survive a retract -- but clearing it
    % keeps each run independent of test order.
    retractall(clojure_target:clj_out_cache(_, _, _)),
    retractall(clojure_target:clj_fail_cache(_, _, _)).

% ============================================================================
% G-A3-1 -- deterministic string / char builtins
% ============================================================================

test(g1_string_length_is_count, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(s_len/2, C),
    has(C, "(count a1)").

test(g1_string_concat_is_str, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(s_cat/3, C),
    has(C, "(str a1 a2)").

test(g1_sub_string_is_subs, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(s_from/3, C),
    has(C, "(subs a1 a2 (+ a2 "),
    has(C, "(count a1)").

%  A Prolog char is a ONE-CHARACTER STRING in this target on both hosts, so the
%  decomposition is `(mapv str (seq s))` and NOT `(vec s)` -- `vec` answers JVM
%  characters on the JVM and one-character strings under ClojureScript, and only
%  one of those compares equal to the atom `'='` this program tests against.
test(g1_string_chars_decomposes_to_one_char_strings,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(s_chars/2, C),
    has(C, "(mapv str (seq a1))"),
    hasnt(C, "(vec a1)").

%  The COMPOSING direction, where the direction rule applies: the known side
%  comes from a preceding goal. This is `strip_brackets/2`'s shape, and getting
%  it right is what makes that predicate have an output at all.
test(g1_string_chars_composes_when_the_known_side_is_produced_earlier,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_pred_outputs(s_round, 2, [2]),
    a3_whole(s_round/2, C),
    has(C, "(apply str "),
    has(C, "(mapv str (seq a1))").

%  PORTED LIMIT, pinned rather than papered over -- the Clojure twin of the TS
%  lane's `gap_g_a3_15_reversible_builtin_picks_decompose_when_ambiguous`.
%
%      s_build(Cs, S) :- string_chars(S, Cs).      % should BUILD S from Cs
%
%  Both arguments are head arguments, so neither side comes from a preceding
%  goal, and the reversible-builtin direction rule declines by design (its
%  `Earlier` set carries only variables a preceding GOAL produced). The output
%  analysis then sees no output and the predicate compiles as a semidet test that
%  computes a value and answers `true`. That is the TypeScript lane's behaviour
%  for the same clause, reproduced exactly; closing it there and not here would
%  be the divergence worth catching, so this probe FAILS the day either lane
%  changes.
test(gap_reversible_builtin_direction_is_ambiguous_between_head_arguments,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_pred_outputs(s_build, 2, []),
    a3_whole(s_build/2, C),
    has(C, "(defn s-build [a1 a2]").

test(g1_char_code_uses_the_runtime_helper,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(c_code/2, C),
    has(C, "(uw-char-code a1)").

test(g1_char_code_helper_is_js_host_after_the_interop_rewrite,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([c_code/2], C),
    has(C, "(.charCodeAt (str c) 0)"),
    hasnt(C, "(.charAt (str c) 0)").

test(g1_string_builtins_run_and_match_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([s_from/3], s_from/3,
              s_from("state=alpha", 6, R), ["state=alpha", 6], R).

% ============================================================================
% G-A3-2 / G-A3-3 -- a clause body is ONE expression; control flow in a guard
% ============================================================================

%  The structural difference from the TypeScript lane, asserted rather than
%  described: a Clojure clause body is an expression, so there is no `return`
%  statement anywhere and no statement/expression distinction to get wrong.
test(g2_clause_body_is_an_expression_not_a_statement_block,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(s_len/2, C),
    has(C, "(defn s-len [a1]"),
    hasnt(C, "return").

test(g3_control_flow_inside_a_guard_renders,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(alpha/1, C),
    has(C, "(>= _s0 97)"), has(C, "(<= _s0 122)"),
    has(C, "(>= _s0 65)"), has(C, "(<= _s0 90)").

test(g3_alpha_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([alpha/1], Code),
    a3_serializer(Ser),
    a3_fn(alpha/1, Fn),
    forall(member(Ch, ['a', 'Z', '0', '-']),
           ( format(string(Driver), '~w~n(println (uw-ser (~w "~w")))', [Ser, Fn, Ch]),
             a3_nbb_run(Code, Driver, Got),
             ( user:alpha(Ch) -> Want = 'true' ; Want = 'false' ),
             Got == Want )).

% ============================================================================
% G-A3-5 / G-A3-6 -- guard placement, and cross-predicate calls
% ============================================================================

%  A guard that reads a value a preceding binding produced cannot be hoisted into
%  the clause condition; it becomes an in-place `if` whose else is the clause's
%  fall-through expression. In TypeScript hoisting it is a temporal-dead-zone
%  ReferenceError; in Clojure it is an unresolved symbol. Either way it must not
%  happen.
test(g6_guard_follows_the_bindings_it_reads,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(starts_w/2, C),
    once(sub_string(C, LetAt, _, _, "(let [_s0 (count a1)")),
    once(sub_string(C, IfAt, _, _, "(if (>= _s0 _s1)")),
    LetAt < IfAt.

test(g6_arity_two_semidet_answers_a_boolean,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(starts_w/2, C),
    has(C, "(defn starts-w [a1 a2]"),
    has(C, "true"), has(C, "false"),
    hasnt(C, "uw-fail").

test(g6_semidet_callee_is_a_boolean_condition,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(uses_test/1, C),
    has(C, "(starts-w a1 \"--\")").

test(g6_negated_semidet_callee_composes,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(uses_neg/1, C),
    has(C, "(not (starts-w a1 \"--\"))").

test(g6_single_output_callee_is_a_let_binding,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(uses_one/2, C),
    has(C, "(s-from a1 2)"),
    has(C, "(let [").

test(g6_multi_output_callee_is_destructured,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(uses_two/3, C),
    has(C, "(let [[_s0 _s1] (split-kv a1)]").

test(g6_module_pulls_in_the_dependency_closure,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([uses_test/1], C),
    has(C, "(defn uses-test"),
    has(C, "(defn starts-w").

test(g6_cross_predicate_calls_run_and_match_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([uses_one/2], uses_one/2, uses_one("--state", R), ["--state"], R).

% ============================================================================
% G-A3-9 -- multi-output predicates return a positional VECTOR
% ============================================================================

test(g9_two_outputs_return_a_vector, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(split_kv/3, C),
    has(C, "(defn split-kv [a1]"),
    has(C, "[_s"),
    hasnt(C, "a2"), hasnt(C, "a3").

%  The historical reading -- "the last argument, full stop" -- would have made
%  argument 4 a required INPUT that the base clause compares against the
%  accumulator, so the caller would have to know half the answer.
test(g9_accumulator_loop_keeps_both_outputs,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_structural(sum_len/5, C),
    has(C, "(defn sum-len [a1 a2 a3]"),
    has(C, "[a2 a3]").

test(g9_multi_output_tail_call_flows_through,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_structural(sum_len/5, C),
    has(C, "(sum-len (rest a1)").

test(g9_two_output_loop_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([sum_len/5], sum_len/5, sum_len([3,4,5], 0, 0, S, N), [[3,4,5], 0, 0],
              [S, N]).

test(g9_two_output_helper_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([split_kv/3], split_kv/3, split_kv("abcd", K, V), ["abcd"], [K, V]).

% ============================================================================
% G-A3-10 -- if-then-else in a recursive body
% ============================================================================

%  In TypeScript the VALUE form needs `let _s0;` before the block and an
%  assignment at the end of each branch, because a JS `if` is a statement. In
%  Clojure the `if` IS the value, so the whole mechanism collapses to one `let`.
test(g10_value_ite_is_a_let_bound_if, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_structural(acc_pos/3, C),
    has(C, "(let [_s1 (if (> (first a1) 0)"),
    %  No mutable slot declared ahead of the branch -- that is the TypeScript
    %  mechanism this port does not need.
    hasnt(C, "let _s1;").

test(g10_tail_ite_is_the_clause_value, [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(idx_of/4, C),
    has(C, "(if (= (first a1) a2)").

test(g10_ite_loop_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([acc_pos/3], acc_pos/3, acc_pos([3,-1,4,-9,2], 0, R), [[3,-1,4,-9,2], 0], R).

test(g10_index_walk_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([idx_of/4], idx_of/4, idx_of(['a','=','b'], '=', 0, R),
              [["a","=","b"], "=", 0], R).

% ============================================================================
% G-A3-12 / G-A3-13 -- compounds as tagged MAPS; the boolean atoms
% ============================================================================

%  Not a tagged VECTOR, for the reason the TS lane rejected a tagged array: a
%  Prolog list is already a vector here, so `["some" v]` could not be told from
%  the list `[some, v]` -- and telling them apart IS the gap.
test(g12_compound_is_constructed_as_a_tagged_map,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(mk_some/2, C),
    has(C, "{:$ \"some\" :args [a1]}").

test(g12_compound_head_pattern_is_a_tag_test_and_destructure,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(un_some/2, C),
    has(C, "(map? a1)"),
    has(C, "(= (:$ a1) \"some\")"),
    has(C, "(= (count (:args a1)) 1)"),
    has(C, "(nth (:args a1) 0)").

%  Clojure's `=` is ALREADY structural, so the TS lane's emitted `_uwEq` helper
%  has no analogue here and must not appear.
test(g12_equality_needs_no_emitted_helper,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(eq_terms/3, C),
    has(C, "(= a1 a2)"),
    hasnt(C, "uwEq"), hasnt(C, "uw-eq").

test(g13_boolean_atoms_are_real_booleans,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(mk_bool/2, C),
    has(C, "false"), has(C, "true"),
    hasnt(C, "\"true\""), hasnt(C, "\"false\"").

%  The four representations must be pairwise distinguishable at run time, and no
%  test may throw. This is the assertion the whole compound design rests on.
test(g12_compound_atom_list_boolean_are_distinguishable,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([mk_some/2], Code),
    format(string(Driver),
'(println
  (str
    (mapv (fn [x] [(boolean (string? x)) (boolean (boolean? x))
                   (boolean (sequential? x)) (boolean (map? x))])
          ["s" true false 7 ["a" "b"] {:$ "f" :args ["a"]}])))', []),
    a3_nbb_run(Code, Driver, Got),
    Got == '[[true false false false] [false true false false] [false true false false] [false false false false] [false false true false] [false false false true]]'.

test(g13_boolean_stays_distinct_from_its_own_name,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([mk_bool/2], Code),
    a3_serializer(Ser),
    a3_fn(mk_bool/2, Fn),
    format(string(Driver),
           '~w~n(println (str (uw-ser (~w "on")) "|" (= (~w "on") "true")))', [Ser, Fn, Fn]),
    a3_nbb_run(Code, Driver, Got),
    Got == 'true|false'.

test(g12_compound_round_trips_under_nbb,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([mk_some/2], mk_some/2, mk_some("v", R), ["v"], R).

% ============================================================================
% G-A3-16 -- a K-V pair walk
% ============================================================================

test(g16_pair_head_pattern_destructures,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(pair_get/3, C),
    has(C, "(= (:$ (first a1)) \"-\")"),
    has(C, "(nth (:args (first a1)) 0)"),
    has(C, "(nth (:args (first a1)) 1)").

test(g16_pair_walk_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([pair_get/3], pair_get/3,
              pair_get(["a"-1, "b"-2], "b", R), [["a"-1, "b"-2], "b"], R).

test(g16_list_building_put_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([put_kv/4], put_kv/4,
              put_kv(["a"-1, "b"-2], "b", 9, R), [["a"-1, "b"-2], "b", 9], R).

% ============================================================================
% G-A3-18 -- the failure sentinel
% ============================================================================

%  WHICH predicates get it: a LEAST fixpoint over the call graph. `pair_get/3`
%  has no clause for `[]`, so its head coverage has a gap and it can fail;
%  `put_kv/4` covers both list shapes and cannot.
test(g18_head_coverage_gap_makes_a_predicate_semidet,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_pred_can_fail(pair_get, 3),
    \+ clojure_target:clj_pred_can_fail(put_kv, 4).

test(g18_semidet_with_outputs_exits_with_the_sentinel,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(pair_get/3, C),
    has(C, "uw-fail"),
    hasnt(C, "no matching clause").

%  A det predicate keeps the throwing exit, so a match that should not have
%  failed is a crash naming the predicate, never a wrong answer.
test(g18_det_predicate_keeps_the_throwing_exit,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(put_kv/4, C),
    has(C, "no matching clause for put-kv/4"),
    hasnt(C, "uw-fail").

%  In CONDITION position the call is bound by an ordinary `let` and the condition
%  reads the name -- where TypeScript needs a mutable `let _t0;` slot assigned
%  INSIDE the condition to smuggle the value out.
test(g18_semidet_callee_in_condition_position_binds_its_output,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(kind_of/3, C),
    has(C, "(let [_t0 (pair-get a1 a2)]"),
    has(C, "(not (identical? _t0 uw-fail))").

%  In BODY-GOAL position a failing call must fall through to the caller's next
%  clause, which is Prolog's reading of a failing goal.
test(g18_semidet_callee_in_body_position_falls_through,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(body_lookup/3, C),
    has(C, "(not (identical? _s0 uw-fail))").

test(g18_sentinel_is_a_fresh_host_object,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([pair_get/3], C),
    has(C, "(def ^:private uw-fail (js/Object.))").

test(g18_sentinel_condition_runs_and_matches_swi_on_both_branches,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([kind_of/3], kind_of/3,
              kind_of(["a"-string], "a", RHit), [["a"-string], "a"], RHit),
    a3_oracle([kind_of/3], kind_of/3,
              kind_of(["a"-string], "zz", RMiss), [["a"-string], "zz"], RMiss).

%  The sentinel must not be forgeable by any value a term lowers to.
test(g18_sentinel_is_not_equal_to_any_term_representation,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([pair_get/3], Code),
    format(string(Driver),
'(println (str (mapv (fn [x] (identical? x uw-fail))
                    ["s" true false 7 [] ["a"] {:$ "f" :args []} (js/Object.)])))', []),
    a3_nbb_run(Code, Driver, Got),
    Got == '[false false false false false false false false]'.

% ============================================================================
% G-A3-19 -- a ground-fact predicate used as a CONSTANT TABLE
% ============================================================================

test(g19_single_row_table_is_inlined_not_called,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(is_global/1, C),
    hasnt(C, "(globals"),
    has(C, "{:$ \"-\" :args [\"state\" \"string\"]}").

test(g19_constant_table_is_not_a_module_member,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([is_global/1], C),
    hasnt(C, "(defn globals").

test(g19_multi_row_table_is_a_membership_test,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_clauses(enumerates/1, Cs),
    clojure_target:clj_fact_pred(protos, 1),
    Cs = [_|_].

test(g19_constant_table_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([is_global/1], Code),
    a3_serializer(Ser),
    a3_fn(is_global/1, Fn),
    format(string(Driver),
           '~w~n(println (str (uw-ser (~w "state")) "|" (uw-ser (~w "zz"))))',
           [Ser, Fn, Fn]),
    a3_nbb_run(Code, Driver, Got),
    ( user:is_global("state") -> A = "true" ; A = "false" ),
    ( user:is_global("zz")    -> B = "true" ; B = "false" ),
    format(atom(Want), '~w|~w', [A, B]),
    Got == Want.

% ============================================================================
% G-A3-20 -- list-BUILDING recursion (a binding the NEXT goal produces)
% ============================================================================

test(g20_deferred_binding_renders_after_the_rest,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_whole(drop_br/2, C),
    has(C, "(drop-br (rest a1))").

test(g20_list_building_recursion_runs_and_matches_swi,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_oracle([drop_br/2], drop_br/2,
              drop_br(['[','a',']'], R), [["[","a","]"]], R).

% ============================================================================
% Naming: hyphenation, arity mangling, forward declarations
% ============================================================================

test(names_are_hyphenated, [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_fn_name(pair_get, 3, N), N == "pair-get".

%  Clojure has no arity overloading across separate defns, so `ovl/2` and
%  `ovl/3` cannot both be `(defn ovl ...)`; an overloaded name gets its arity
%  appended, an un-overloaded one is untouched.
test(names_overloaded_arity_is_mangled, [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_fn_name(ovl, 2, N2), N2 == "ovl-2",
    clojure_target:clj_fn_name(ovl, 3, N3), N3 == "ovl-3",
    clojure_target:clj_fn_name(pair_get, 3, NP), NP == "pair-get".

%  A JS `function` declaration is hoisted; a Clojure var is not. A
%  multi-predicate module must therefore open with `(declare ...)` or a caller
%  emitted before its callee cannot resolve.
test(module_declares_every_function_up_front,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([uses_test/1], C),
    has(C, "(declare "),
    once(sub_string(C, DeclAt, _, _, "(declare ")),
    once(sub_string(C, DefnAt, _, _, "(defn ")),
    DeclAt < DefnAt.

test(overloaded_module_loads_under_nbb,
     [condition(nbb_available), setup(a3_setup), cleanup(a3_teardown)]) :-
    a3_module([ovl/2], C),
    has(C, "(defn ovl-2"), has(C, "(defn ovl-3"),
    a3_nbb_loads(C).

% ============================================================================
% REFUSALS -- what deliberately stays unsupported
% ============================================================================

%  An UNKNOWN callee has no visible clauses, so clj_pred_outputs/3 fails, so the
%  cross-call lowering declines and the whole path refuses. That failure is
%  load-bearing: without it the caller would emit a call to a function nothing
%  defines, which reads as perfectly good Clojure and dies at run time.
test(refusal_unknown_callee_is_declined,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    \+ clojure_target:clj_pred_outputs(no_such_pred, 2, _),
    \+ a3_whole(calls_unknown/2, _).

%  Calling a MULTI-ROW ground-fact table with an unbound argument is an
%  ENUMERATION -- nondeterminism this target has no form for -- so it is refused
%  rather than lowered to something that answers one row.
test(refusal_multi_row_table_with_an_unbound_argument,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    clojure_target:clj_fact_pred(protos, 1),
    \+ clojure_target:clj_fact_call('$no_self', protos(_Unbound), [], _, _).

%  A predicate the A3 paths cannot lower must not silently reach the historical
%  fact fallback and get compiled by RUNNING it.
test(refusal_a3_path_declines_rather_than_guessing,
     [setup(a3_setup), cleanup(a3_teardown)]) :-
    \+ a3_general(calls_unknown/2, _).

% ============================================================================
% ENDGAME -- the real examples/cli_args/cli_args.pl, compiled whole
% ============================================================================

%% a3_load_cli_args
%  Read the FROZEN reference and assert its clauses into `user`, the way
%  examples/cli_args/cljs/build.pl does. Nothing is executed.
a3_load_cli_args :-
    a3_cli_args_file(File),
    exists_file(File),
    setup_call_cleanup(open(File, read, S), a3_read_terms(S), close(S)).

a3_cli_args_file(File) :-
    source_file(a3_cli_args_file(_), Here),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/../../examples/cli_args/cli_args.pl'], File0),
    absolute_file_name(File0, File).

a3_read_terms(S) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  true
    ;   ( T = (:- _) -> true ; assertz(user:T) ),
        a3_read_terms(S)
    ).

a3_unload_cli_args :-
    forall(member(P/A, [global_options/1, default_registry/1, js_object_prototype_keys/1,
                        js_object_prototype_key/1, is_long_flag/1, long_flag_tail/1,
                        looks_like_legacy_flag/1, legacy_flag_tail/1, js_alpha/1,
                        js_flag_char/1, starts_with/2, substring_from/3,
                        substring_range/4, first_equals_index/2, first_char_index/4,
                        split_flag_token/3, string_member/2, pair_lookup/3,
                        nth0_default/4, last_element/2, flags_set/4, flags_put/4,
                        merge_flags/3, merge_flags_/3, schema_for/5, registry_entry/3,
                        action_entry/3, parse_lenient/3, lenient_loop/5,
                        parse_strict/4, strict_loop/8, strict_option/11,
                        next_value/2, option_kind/3, check_arity/3, count_required/3,
                        strip_brackets/2, drop_brackets/2, parse_args/2, parse_args/3,
                        lenient_result/2, scan_leading_globals/4, is_global_key/1]),
           ( functor(H, P, A), retractall(user:H) )),
    retractall(clojure_target:clj_out_cache(_, _, _)),
    retractall(clojure_target:clj_fail_cache(_, _, _)).

%  THE CENSUS. 43 predicates: 40 rule predicates become functions, and the 3
%  ground-fact CONSTANT TABLES are inlined at their call sites and are therefore
%  not module members.
test(endgame_whole_cli_args_program_compiles_into_one_namespace,
     [setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    a3_module([parse_args/2], Code),
    hasnt(Code, "TODO: Implement"),
    findall(L, ( split_string(Code, "\n", "", Ls), member(L, Ls),
                 sub_string(L, 0, _, _, "(defn "),
                 \+ sub_string(L, 0, _, _, "(defn ^:private ") ), Defns),
    length(Defns, N),
    ( N =:= 40 -> true
    ; format(user_error, "~n  expected 40 defns, got ~w~n", [N]), fail ),
    has(Code, "(defn parse-args-2"),
    has(Code, "(defn parse-args-3"),
    hasnt(Code, "(defn global-options"),
    hasnt(Code, "(defn default-registry"),
    hasnt(Code, "(defn js-object-prototype-keys").

test(endgame_namespace_loads_under_nbb,
     [condition(nbb_available), setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    a3_module([parse_args/2], Code),
    a3_nbb_loads(Code).

a3_endgame_argv([ ["block", "--include-key", "bob"],
                  ["block", "bob", "--include-key"],
                  ["block", "bob", "--include-key=false"],
                  ["block", "bob", "--include-keey"],
                  ["--state", "P", "block", "bob"],
                  ["commands", "add", "deploy", "--", "./run.sh", "--env", "prod"],
                  ["add", "bob", "--key", "-----BEGIN-PUBLIC-KEY-----"],
                  ["daemon", "--debug"],
                  ["daemon", "--debug", "2"],
                  ["profiles", "pin", "trusted", "--force"],
                  ["add", "bob", "--profile"],
                  ["block"],
                  ["tunnels", "--a", "--b=c"],
                  ["unblock", "--key", "ABCDEF12"],
                  ["block", "bob", "--__proto__", "v"],
                  [] ]).

%  The compiled parse_args/2 run under nbb, against SWI running the same clauses,
%  on every argv line the corpus and the quirk sweep care about.
test(endgame_compiled_parse_args_matches_swi_under_nbb,
     [condition(nbb_available), setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    a3_module([parse_args/2], Code),
    a3_serializer(Ser),
    a3_endgame_argv(Cases),
    forall(member(Argv, Cases),
           ( a3_lit(Argv, Lit),
             format(string(Driver), '~w~n(println (uw-ser (parse-args-2 ~w)))',
                    [Ser, Lit]),
             a3_nbb_run(Code, Driver, Got),
             user:parse_args(Argv, Result),
             a3_ser(Result, Want),
             atom_string(Got, GotS),
             ( GotS == Want
             -> true
             ;  format(user_error, "~n  argv: ~w~n  nbb: ~w~n  swi: ~w~n",
                       [Argv, GotS, Want]), fail )
           )).

%  The four mechanisms are all reachable from parse_args/2, so the closure must
%  contain every predicate each of them needs.
test(endgame_all_four_mechanisms_are_in_the_closure,
     [setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    a3_module([parse_args/2], Code),
    forall(member(F, ["(defn parse-lenient", "(defn lenient-loop",
                      "(defn parse-strict", "(defn strict-loop", "(defn strict-option",
                      "(defn scan-leading-globals", "(defn is-global-key",
                      "(defn schema-for", "(defn registry-entry", "(defn action-entry"]),
           has(Code, F)).

%% a3_gets_sentinel(+Pred, +Arity)
%  The G-A3-18 convention applies to a predicate that can fail AND has outputs.
%  A predicate that can fail with NO outputs is an ordinary semidet TEST and
%  carries its failure as `false`, which is why `starts_with/2` is not in the
%  sentinel set even though it plainly can fail.
a3_gets_sentinel(P, A) :-
    clojure_target:clj_pred_can_fail(P, A),
    clojure_target:clj_pred_outputs(P, A, Outs),
    Outs \== [].

%  G-A3-18's determinacy analysis, on the real program. These are exactly the
%  NINE predicates the TypeScript lane's fixpoint names, which is the check that
%  the port reproduces the analysis rather than approximating it.
test(endgame_sentinel_set_is_the_same_nine_as_the_ts_lane,
     [setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    Nine = [pair_lookup/3, last_element/2, option_kind/3, registry_entry/3,
            action_entry/3, schema_for/5, parse_strict/4, parse_args/3, parse_args/2],
    forall(member(P/A, Nine),
           ( a3_gets_sentinel(P, A)
           -> true
           ;  format(user_error, "~n  expected ~w/~w to carry the sentinel~n", [P, A]),
              fail )),
    forall(member(Q/B, [substring_from/3, substring_range/4, flags_put/4, flags_set/4,
                        lenient_loop/5, parse_lenient/3, merge_flags/3, split_flag_token/3,
                        first_char_index/4, count_required/3, check_arity/3,
                        scan_leading_globals/4, strict_loop/8, strict_option/11]),
           ( \+ a3_gets_sentinel(Q, B)
           -> true
           ;  format(user_error, "~n  expected ~w/~w NOT to carry the sentinel~n", [Q, B]),
              fail )),
    %  A semidet TEST carries failure as `false`, not as the sentinel.
    clojure_target:clj_pred_outputs(starts_with, 2, []),
    \+ a3_gets_sentinel(starts_with, 2).

%  G-A3-9's output analysis, on the real program's three multi-output loops.
test(endgame_output_analysis_finds_every_multi_output_loop,
     [setup(a3_load_cli_args), cleanup(a3_unload_cli_args)]) :-
    clojure_target:clj_pred_outputs(lenient_loop, 5, [4, 5]),
    clojure_target:clj_pred_outputs(strict_loop, 8, [6, 7, 8]),
    clojure_target:clj_pred_outputs(scan_leading_globals, 4, [3, 4]),
    clojure_target:clj_pred_outputs(split_flag_token, 3, [2, 3]),
    clojure_target:clj_pred_outputs(starts_with, 2, []).

:- end_tests(clojurescript_cli_args_shapes).

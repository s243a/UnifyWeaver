:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% deterministic_recursion.pl — general deterministic-recursion classifier
%
% Target-AGNOSTIC front-end for the WAM lowered tiers: the compositional
% `deterministic_recursion_class/2` pipeline recommended in
% docs/proposals/DETERMINISTIC_RECURSION_TAXONOMY.md §11. It classifies a
% predicate (read as clauses from a module) into one of the deterministic
% recursion CLASSES of the taxonomy — or `decline(Reason)` — so a single
% recognizer replaces the family of hand-built, `=@=`-frozen, one-per-predicate
% region recognizers (rust_region1..5 in wam_rust_target.pl). Only step 7
% (emission) is per-target; every stage here operates on the Prolog-clause /
% WAM-level shape and is reusable by every hybrid WAM backend, exactly the
% "shared front-end, per-target back-end" split of
% WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% PIPELINE STAGES (taxonomy §11):
%   1. Call-graph + SCC decomposition (mutual recursion analysed as a whole SCC).
%   2. Meta-call resolution preconditions (closed-world / closure / mode).
%   3. Per-clause self/SCC-call arity + position (tail vs non-tail).
%   4. Commit / aggregate-boundary detection.
%   5. Whole-predicate/SCC at-most-one-solution proof (the det lattice).
%   6. Termination classification (structural decrease vs finite-domain Seen).
%   (Step 7, mechanical emission selection, lives in each target — for Rust see
%    wam_rust_target.pl's genrec driver.)
%
% The class vocabulary (Class in deterministic_recursion_class/2):
%   tail_loop(M:Name/Arity, Shape)      — pattern 1 (native loop)
%   committed(M:Name/Arity, Shape)      — pattern 6 (committed-choice, native loop)
%   mutual(SCC, MemberClasses)          — pattern 5 (SCC of size >= 2)
%   decline(Reason)                     — not lowered this round (safety net)
%
% Shape (the emitter's input, precise enough for byte-identical lowering):
%   list_filter(Guard, In, Pass, Out)   — walk a list, per-element committed
%                                          keep/drop guard, difference-list build
%   list_map_index(EF/EA, Row, In, Idx, Out)
%                                       — walk a list, destructure element,
%                                          emit a row with a running index
%   bst_descent(TF/TA, KPos, VPos, In, Key, Out)
%                                       — compare/3 trichotomy, one of two tail
%                                          self-calls, semidet lookup
%
% The recognizers below are STRUCTURAL (term inspection + variable-sharing
% checks), never `=@=` against a frozen literal, so the same classifier fires
% for any predicate of the shape — the sibling-gap family
% (filter_satisfies/key_pkg_rows/tree_lookup) AND the already-lowered siblings
% (matching_versions/key_dep_rows/build_tree), which is what lets a later
% cleanup subsume the hand-built regions.
%
% Attribution: the whole-predicate "optimistic then swept" determinism
% classification and the tail-loop (F11) shape are ideas adopted from Kenichi
% Sasagawa's M-Prolog / N-Prolog (Modified BSD), never code — see
% docs/proposals/MPROLOG_MINING_NOTES.md.

:- module(deterministic_recursion, [
    deterministic_recursion_class/2,   % +M:Name/Arity, -Class
    deterministic_recursion_class/3,   % +Module, +Name/Arity, -Class
    dr_scc/3,                          % +Module, +Name/Arity, -SCC (sorted PIs)
    dr_self_recursive/2,               % +Module, +Name/Arity  (semidet)
    dr_callees/3,                      % +Module, +Name/Arity, -Callees
    dr_lowerable_guard/1               % +Guard  (semidet: known det-lowerable)
   ]).

:- use_module(library(lists)).

% =====================================================================
% Clause access
% =====================================================================

%% dr_clauses(+Module, +Name/Arity, -Clauses) is det.
%  Clauses = list of Head-Body copies (fresh variables per clause). Fails to
%  bind nothing if the predicate is undefined (Clauses = []).
dr_clauses(M, Name/Arity, Clauses) :-
    ( \+ current_predicate(Name/Arity)
    -> Clauses = []
    ;  findall(H-B, ( functor(H, Name, Arity), clause(M:H, B) ), Clauses)
    ).

%% dr_defined_user(+Module, +Name/Arity) is semidet.
%  The predicate is defined in Module (not a library builtin) with clauses.
dr_defined_user(M, Name/Arity) :-
    current_predicate(Name/Arity),
    functor(H, Name, Arity),
    \+ predicate_property(M:H, built_in),
    \+ predicate_property(M:H, imported_from(_)),
    once(clause(M:H, _)).

% =====================================================================
% Stage 1 — call graph + SCC (reachability-based, exact for tiny graphs)
% =====================================================================

%% dr_body_goals(+Body, -Goals) is det.
%  Flatten a clause body into its atomic goals, descending through control
%  constructs (,/;/->/*->/\+/call/once/findall/bagof/setof/forall/aggregate_all).
dr_body_goals(B, Goals) :-
    findall(G, dr_body_goal(B, G), Goals).

dr_body_goal(V, _) :- var(V), !, fail.
dr_body_goal((A,B), G)   :- !, ( dr_body_goal(A,G) ; dr_body_goal(B,G) ).
dr_body_goal((A;B), G)   :- !, ( dr_body_goal(A,G) ; dr_body_goal(B,G) ).
dr_body_goal((A->B), G)  :- !, ( dr_body_goal(A,G) ; dr_body_goal(B,G) ).
dr_body_goal((A*->B), G) :- !, ( dr_body_goal(A,G) ; dr_body_goal(B,G) ).
dr_body_goal(\+ A, G)    :- !, dr_body_goal(A, G).
dr_body_goal(once(A), G) :- !, dr_body_goal(A, G).
dr_body_goal(ignore(A), G) :- !, dr_body_goal(A, G).
dr_body_goal(findall(_,A,_), G) :- !, dr_body_goal(A, G).
dr_body_goal(bagof(_,A,_), G)   :- !, dr_body_goal(A, G).
dr_body_goal(setof(_,A,_), G)   :- !, dr_body_goal(A, G).
dr_body_goal(forall(A,B), G)    :- !, ( dr_body_goal(A,G) ; dr_body_goal(B,G) ).
dr_body_goal(aggregate_all(_,A,_), G) :- !, dr_body_goal(A, G).
% call/N: record the meta-call itself (stage 2 inspects it); do not descend.
dr_body_goal(G, G).

%% dr_callees(+Module, +Name/Arity, -Callees) is det.
%  User predicates (with clauses) called directly from any clause body of P.
dr_callees(M, Name/Arity, Callees) :-
    dr_clauses(M, Name/Arity, Clauses),
    findall(CN/CA,
            ( member(_-B, Clauses),
              dr_body_goal(B, G),
              nonvar(G),
              \+ dr_is_metacall(G),
              functor(G, CN, CA),
              dr_defined_user(M, CN/CA) ),
            Cs0),
    sort(Cs0, Callees).

dr_is_metacall(G) :- functor(G, call, N), N >= 1.

%% dr_reaches(+Module, +From, +To) is semidet.
%  From can reach To in the module-local call graph (one or more hops).
dr_reaches(M, From, To) :-
    once(dr_reaches_(M, [From], [], To)).

dr_reaches_(M, [P|_], _, To) :-
    dr_callees(M, P, Cs),
    memberchk(To, Cs), !.
dr_reaches_(M, [P|Ps], Seen, To) :-
    dr_callees(M, P, Cs),
    subtract(Cs, [P|Seen], New0),
    subtract(New0, Ps, New),
    append(Ps, New, Agenda),
    dr_reaches_(M, Agenda, [P|Seen], To).

%% dr_self_recursive(+Module, +Name/Arity) is semidet.
dr_self_recursive(M, PI) :- dr_reaches(M, PI, PI).

%% dr_scc(+Module, +Name/Arity, -SCC) is det.
%  The strongly-connected component containing PI: PI plus every module-local
%  predicate mutually reachable with it. Sorted; always contains PI.
dr_scc(M, PI, SCC) :-
    dr_module_pis(M, PI, All),
    include(dr_mutual(M, PI), All, Others),
    sort([PI|Others], SCC),
    !.

dr_mutual(M, PI, Q) :-
    PI \== Q,
    dr_reaches(M, PI, Q),
    dr_reaches(M, Q, PI).

%% dr_module_pis(+Module, +Seed, -PIs) is det.
%  All user predicates reachable from Seed (closure of the call graph).
dr_module_pis(M, Seed, PIs) :-
    dr_closure(M, [Seed], [], Closure),
    sort(Closure, PIs).

dr_closure(_, [], Acc, Acc).
dr_closure(M, [P|Ps], Acc, Out) :-
    ( memberchk(P, Acc)
    -> dr_closure(M, Ps, Acc, Out)
    ;  dr_callees(M, P, Cs),
       append(Cs, Ps, Agenda),
       dr_closure(M, Agenda, [P|Acc], Out)
    ).

% =====================================================================
% Stage 2 — meta-call resolution preconditions
% =====================================================================
%
% A clause reaching call/N is admitted to a deterministic class ONLY under the
% three taxonomy §6 preconditions (closed-world target set, closure/partial-app
% resolution, mode-sensitive per-target determinism). None of the sibling-gap
% family, close_moving, or the tail-recursive mutual SCCs use a meta-call, so
% for this round any predicate whose analysed clauses reach a call/N is
% DECLINED (the safe, correct answer until the process_all/4 path is built).

%% dr_scc_has_metacall(+Module, +SCC) is semidet.
dr_scc_has_metacall(M, SCC) :-
    member(PI, SCC),
    dr_clauses(M, PI, Clauses),
    member(_-B, Clauses),
    dr_body_goal(B, G),
    nonvar(G),
    dr_is_metacall(G), !.

% =====================================================================
% Main entry
% =====================================================================

%% deterministic_recursion_class(+M:Name/Arity, -Class) is det.
deterministic_recursion_class(M:PI, Class) :- !,
    deterministic_recursion_class(M, PI, Class).
deterministic_recursion_class(PI, Class) :-
    deterministic_recursion_class(user, PI, Class).

%% deterministic_recursion_class(+Module, +Name/Arity, -Class) is det.
%  Never fails: always binds Class (a recognised class or decline/1).
deterministic_recursion_class(M, Name/Arity, Class) :-
    ( catch(dr_classify(M, Name/Arity, C), _E, C = decline(classifier_error))
    -> Class = C
    ;  Class = decline(unclassified)
    ).

dr_classify(M, PI, Class) :-
    ( \+ dr_defined_user(M, PI)
    -> Class = decline(not_user_predicate)
    ;  dr_scc(M, PI, SCC),
       ( dr_scc_has_metacall(M, SCC)
       -> Class = decline(meta_call_unresolved)
       ;  SCC = [PI]
       -> dr_classify_single(M, PI, Class)     % stages 3-6, single predicate
       ;  dr_classify_scc(M, PI, SCC, Class)    % stage 1 wrapper: mutual
       )
    ).

% =====================================================================
% Single-predicate classification (SCC size 1) — stages 3-6
% =====================================================================

dr_classify_single(M, PI, Class) :-
    dr_clauses(M, PI, Clauses),
    (   dr_list_filter_shape(PI, Clauses, Shape)
    ->  Class = tail_loop(M:PI, Shape)
    ;   dr_list_map_index_shape(PI, Clauses, Shape)
    ->  Class = tail_loop(M:PI, Shape)
    ;   dr_bst_descent_shape(PI, Clauses, Shape)
    ->  Class = tail_loop(M:PI, Shape)
    ;   dr_committed_tail_shape(M, PI, Clauses, Shape)
    ->  Class = committed(M:PI, Shape)
    ;   \+ dr_self_recursive(M, PI)
    ->  Class = decline(not_recursive)
    ;   Class = decline(unrecognized_single_shape)
    ).

% =====================================================================
% Shape 1: list_filter (pattern 1) — filter_satisfies / matching_versions kin
% =====================================================================
%
%   p([], Pass.., []).
%   p([H|T], Pass.., Out) :- ( Guard -> Out=[H|Os] ; Out=Os ), p(T, Pass.., Os).
%
% where Guard is a committed, determinism-lowerable goal over {H} ∪ Pass and the
% ONLY self-call is the tail. In = input list arg position, Out = output arg
% position, Pass = the other (threaded, unchanged) arg positions.

dr_list_filter_shape(Name/Arity, Clauses, list_filter(GuardPI, In, ConPos, Out)) :-
    Clauses = [_,_],                      % exactly base + recursive
    select(BaseH-BaseB, Clauses, [RecH-RecB]),
    BaseB == true,
    % locate the input-list position (empty list in base head) and output
    % position (empty list in base head) — both are [] in the base clause.
    BaseH =.. [Name|BaseArgs],
    nth1(In, BaseArgs, BaseIn), dr_is_nil(BaseIn),
    nth1(Out, BaseArgs, BaseOut), dr_is_nil(BaseOut),
    In \== Out,
    % recursive head: arg[In] = [H|T], arg[Out] = OutV (var), pass args are vars.
    RecH =.. [Name|RecArgs],
    nth1(In, RecArgs, [H|T]), var(H), var(T),
    nth1(Out, RecArgs, OutV), var(OutV),
    dr_pass_positions(Arity, [In,Out], Pass),
    % recursive body: ( Guard -> OutV=[H|Os] ; OutV=Os ), Self(T, Pass.., Os)
    RecB = (Ite, Self),
    Ite = (Guard -> Then ; Else),
    Then = (OutV = [Hk|Os1]), Hk == H,
    Else = (OutV = Os2), Os2 == Os1,
    % the sole self-call is the tail: same functor, arg[In]=T, arg[Out]=Os1,
    % pass args identical to the head's pass args.
    Self =.. [Name|SelfArgs],
    nth1(In, SelfArgs, ST), ST == T,
    nth1(Out, SelfArgs, SOut), SOut == Os1,
    dr_pass_identical(Pass, RecArgs, SelfArgs),
    % Guard must be a known determinism-lowerable goal: satisfies(H, Con) with
    % the element H as the FIRST argument and the constraint a single pass arg.
    dr_lowerable_guard(Guard),
    Guard = satisfies(GA, GB), GuardPI = satisfies/2,
    GA == H,
    nth1(ConPos, RecArgs, ConVar), ConVar == GB,
    memberchk(ConPos, Pass),
    % no other guard variable escapes {H} ∪ pass args.
    dr_guard_vars_ok(Guard, H, Pass, RecArgs).

% Guard's variables are drawn only from {H} ∪ pass-arg vars. (No findall here:
% it would COPY the vars and break the `==` identity checks.)
dr_guard_vars_ok(Guard, H, Pass, RecArgs) :-
    term_variables(Guard, GVars),
    forall(member(GV, GVars),
           ( GV == H ; dr_pass_has_var(Pass, RecArgs, GV) )).

dr_pass_has_var([P|_], RecArgs, V) :- nth1(P, RecArgs, PV), PV == V, !.
dr_pass_has_var([_|Ps], RecArgs, V) :- dr_pass_has_var(Ps, RecArgs, V).

% =====================================================================
% Shape 2: list_map_index (pattern 1) — key_pkg_rows / key_dep_rows kin
% =====================================================================
%
%   p([], _I, []).
%   p([Elem|Rest], I, [Row|Ks]) :- <maps>, I1 is I+1, p(Rest, I1, Ks).
%
% Row/element built structurally in the head; body is (optional maps then)
% `I1 is I+1` and the tail self-call. Recognised precisely for the two frozen
% row shapes actually emittable this round (see the emitter): the pkg-row
% `N-I-V` from `package(N,V)`, and the dep-row `(N-V)-I-Req` from
% `depends(N,V,D,C)` via dep_to_req/3.

dr_list_map_index_shape(Name/Arity, Clauses, Shape) :-
    Arity =:= 3,
    Clauses = [_,_],
    select(BaseH-BaseB, Clauses, [RecH-RecB]),
    BaseB == true,
    BaseH =.. [Name, BIn, _, BOut],
    dr_is_nil(BIn), dr_is_nil(BOut),
    RecH =.. [Name, [Elem|Rest], I, [Row|Ks]],
    nonvar(Elem),
    ( dr_map_pkg_row(Elem, I, Row, Rest, Ks, Name, RecB, Shape)
    ; dr_map_dep_row(Elem, I, Row, Rest, Ks, Name, RecB, Shape)
    ).

% pkg row: Elem = package(N,V), Row = N-I-V, body = I1 is I+1, p(Rest,I1,Ks).
dr_map_pkg_row(Elem, I, Row, Rest, Ks, Name, Body,
               list_map_index(package/2, pkg_row, 1, 2, 3)) :-
    Elem = package(N, V), var(N), var(V),
    Row = ((RN - RI) - RV), RN == N, RI == I, RV == V,
    Body = (I1 is I + 1, Self),
    Self =.. [Name, SRest, SI1, SKs],
    SRest == Rest, SI1 == I1, SKs == Ks.

% dep row: Elem = depends(N,V,D,C), Row = (N-V)-I-Req, body = dep_to_req(D,C,Req),
% I1 is I+1, p(Rest,I1,Ks).
dr_map_dep_row(Elem, I, Row, Rest, Ks, Name, Body,
               list_map_index(depends/4, dep_row, 1, 2, 3)) :-
    Elem = depends(N, V, D, C), var(N), var(V), var(D), var(C),
    Row = (((RN - RV) - RI) - Req), RN == N, RV == V, RI == I,
    Body = (dep_to_req(D2, C2, Req2), I1 is I + 1, Self),
    D2 == D, C2 == C, Req2 == Req,
    Self =.. [Name, SRest, SI1, SKs],
    SRest == Rest, SI1 == I1, SKs == Ks.

% =====================================================================
% Shape 3: bst_descent (pattern 1, tree) — tree_lookup
% =====================================================================
%
%   p(t(L,K,V,R), Key, Val) :-
%       compare(Ord, Key, K),
%       ( Ord = (=) -> Val = V
%       ; Ord = (<) -> p(L, Key, Val)
%       ;             p(R, Key, Val) ).
%
% Single clause; the empty-tree atom `t` has no matching clause head, so the
% predicate is semidet (fails on empty tree / missing key) — reproduced by the
% native method returning Some(false).

dr_bst_descent_shape(Name/3, [H-B], bst_descent('t'/4, 2, 3, 1, 2, 3)) :-
    H =.. [Name, Tree, Key, Val],
    nonvar(Tree),
    Tree = t(L, K, V, R), var(L), var(K), var(V), var(R),
    var(Key), var(Val),
    B = (compare(Ord, K1, K2), Ite),
    K1 == Key, K2 == K,
    Ite = (E1 -> T1 ; E2 -> T2 ; T3),
    E1 = (Ord1 = (=)), Ord1 == Ord,
    T1 = (Val1 = Vv), Val1 == Val, Vv == V,
    E2 = (Ord2 = (<)), Ord2 == Ord,
    T2 =.. [Name, TL, TKey, TVal], TL == L, TKey == Key, TVal == Val,
    T3 =.. [Name, TR, TKey2, TVal2], TR == R, TKey2 == Key, TVal2 == Val.

% =====================================================================
% Shape 4: committed-choice tail (pattern 6) — close_moving kin
% =====================================================================
%
% Detected but NOT emitted this round (declined at the emitter). The marker:
% single recursive clause whose sole self-call is a tail call confined to run
% after a committing boundary (`->`/once/cut), with a base clause. This
% classifies close_moving/first_broken-style fixpoint chains; the emitter's
% step-7 decides whether to lower or decline.

dr_committed_tail_shape(M, Name/Arity, Clauses, committed_tail(SelfCount)) :-
    dr_self_recursive(M, Name/Arity),
    % every clause makes at most one self-call, always in tail position after a
    % commit; there is a commit boundary somewhere before it.
    findall(N,
            ( member(_-B, Clauses),
              dr_count_self_calls(B, Name/Arity, N) ),
            Counts),
    max_list(Counts, MaxSelf),
    MaxSelf =< 1,
    SelfCount = MaxSelf,
    once(( member(_-B2, Clauses), dr_has_commit(B2) )).

dr_count_self_calls(B, Name/Arity, N) :-
    dr_body_goals(B, Gs),
    include(dr_is_self(Name/Arity), Gs, Selfs),
    length(Selfs, N).

dr_is_self(Name/Arity, G) :- nonvar(G), functor(G, Name, Arity).

dr_has_commit((_ -> _ ; _)) :- !.
dr_has_commit((_ -> _)) :- !.
dr_has_commit((A, B)) :- !, ( dr_has_commit(A) ; dr_has_commit(B) ).
dr_has_commit((A ; B)) :- !, ( dr_has_commit(A) ; dr_has_commit(B) ).
dr_has_commit(once(_)) :- !.
dr_has_commit(!).

% =====================================================================
% SCC (mutual recursion, size >= 2) — stage 1 wrapper
% =====================================================================
%
% DETECTION only this round. Emission of a mutual-recursion call-cycle without
% stack blowup is deferred (see the taxonomy §5): we classify the SCC and
% whether it is tail-shaped, but the emitter declines to the interpreter unless
% a target has a proven-safe fused emission. This scaffolds detection without
% half-building a stack-unsafe emission.

dr_classify_scc(M, _PI, SCC, mutual(SCC, tail_shaped(Tail), Members)) :-
    findall(Mi-Ci,
            ( member(Mi, SCC),
              dr_scc_member_shape(M, Mi, SCC, Ci) ),
            Members),
    ( \+ member(_-nontail, Members)
    -> Tail = true
    ;  Tail = false
    ).

% Classify one SCC member's self/SCC-call position: tail or nontail.
dr_scc_member_shape(M, Mi, SCC, Shape) :-
    dr_clauses(M, Mi, Clauses),
    ( forall(member(_-B, Clauses), dr_scc_calls_tail(B, SCC))
    -> Shape = tail
    ;  Shape = nontail
    ).

% A clause is tail w.r.t. the SCC iff every SCC-member call it makes is in tail
% position — i.e. there is at most ONE SCC call in the body and, when present,
% it is the syntactically last goal. A second SCC call (like topo_one's inner
% walk before its post-order prepend, or a helper SCC-call before the tail
% self-call) makes the clause non-tail.
dr_scc_calls_tail(B, SCC) :-
    findall(G, ( dr_body_goal(B, G), nonvar(G), dr_goal_in_scc(G, SCC) ), SccCalls),
    ( SccCalls == []
    -> true
    ;  SccCalls = [_],                     % exactly one SCC call in the body
       dr_conj_last(B, Last),
       dr_goal_in_scc(Last, SCC)           % and it is the tail goal
    ).

dr_conj_last((_, B), Last) :- !, dr_conj_last(B, Last).
dr_conj_last((_ -> T ; E), Last) :- !, ( dr_conj_last(T, Last) ; dr_conj_last(E, Last) ).
dr_conj_last((_ -> T), Last) :- !, dr_conj_last(T, Last).
dr_conj_last((_ ; E), Last) :- !, dr_conj_last(E, Last).
dr_conj_last(G, G).

dr_goal_in_scc(G, SCC) :-
    nonvar(G), functor(G, N, A), memberchk(N/A, SCC).

% =====================================================================
% Small utilities
% =====================================================================

dr_is_nil([]) :- !.
dr_is_nil(A) :- A == [].

%% dr_pass_positions(+Arity, +Exclude, -Pass) is det.
dr_pass_positions(Arity, Exclude, Pass) :-
    numlist(1, Arity, All),
    subtract(All, Exclude, Pass).

%% dr_pass_identical(+Pass, +RecArgs, +SelfArgs) is semidet.
%  Each pass-position argument is the SAME variable in the head and the self-call.
dr_pass_identical([], _, _).
dr_pass_identical([P|Ps], RecArgs, SelfArgs) :-
    nth1(P, RecArgs, RA),
    nth1(P, SelfArgs, SA),
    RA == SA,
    dr_pass_identical(Ps, RecArgs, SelfArgs).

% =====================================================================
% dr_lowerable_guard/1 — known determinism-lowerable per-element guards
% =====================================================================
%
% The set of goals the Rust emitter has a proven native lowering for as a
% committed per-element filter guard. Extend this as more fused callees are
% lowered. For this round: satisfies/2 (region 2's private native copy).
dr_lowerable_guard(satisfies(_, _)) :- !.

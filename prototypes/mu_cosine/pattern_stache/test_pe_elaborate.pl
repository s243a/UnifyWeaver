:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pe_elaborate.pl - verification of the elaborator prototype
% against the note's own oracle inventory (DESIGN_prolog_elaborator.md
% §4, rulings decided in PR #4093).
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pe_elaborate.pl
%
% Contents:
%   - GROUND PATH against sealed bytes: all 25 golden rows through
%     elaborate/3 -> pe_semantic/pe_full, byte-equal to the bundle;
%   - GROUND PATH against sealed STRUCTURE: the note's find — the
%     bundle's resolved_ast field as a structural oracle; the
%     elaborated ground term is converted and compared against the
%     sealed JSON for all 25 rows (new coverage, required);
%   - the fixpoint loop: binding chains needing multiple passes, and
%     the note's own observation (discharging a binding makes a
%     residual checkable) as a test;
%   - RESIDUAL PATH, behavioral (no sealed oracle exists — the note
%     says so honestly): discharge order-independence, termination,
%     the sort key's measured stability properties re-asserted, and
%     pattern-state shape;
%   - the generated mirror: load-time hash check, drift refusal
%     (tampered copy refuses to load, in a subprocess), and
%     cross-checks against process_expression_vnext/testdata/
%     frontend_registry_fixture.json AS DATA ONLY (never importing,
%     wrapping, or porting the Python machinery);
%   - fail-closed battery with named errors and origins.
%
% Pattern-state fixtures use file-local names under the mandatory
% no_identity/1 marker (ruling 7): no digests, no canonical bytes, no
% persistent naming — peid-v1's fence.

:- use_module(pe_elaborate).
:- use_module(pe_emit, [pe_semantic/2, pe_full/2]).
:- use_module(pe_registry_mirror,
              [pe_output/2, pe_required/2, pe_registry_version/1]).
:- use_module(library(plunit)).
:- use_module(library(http/json)).
:- use_module(library(pairs)).
:- use_module(library(apply)).
:- use_module(library(lists)).
:- use_module(library(process)).

%% ============================================
%% Golden bundle access
%% ============================================

:- dynamic golden_row/4.   % golden_row(Name, Identity, Full, ResolvedAstDict)

here(Dir) :-
    module_property(pe_elaborate, file(Here)),
    file_directory_name(Here, Dir).

load_golden :-
    retractall(golden_row(_, _, _, _)),
    here(Dir),
    atomic_list_concat([Dir, '/../PROCESS_EXPRESSION_GOLDEN_v3.json'], Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        json_read_dict(S, Bundle),
        close(S)),
    Bundle.registry_version == "v0.4",
    forall(member(Row, Bundle.rows),
           ( atom_string(Name, Row.name),
             assertz(golden_row(Name,
                                Row.canonical_identity_string,
                                Row.canonical_full_string,
                                Row.resolved_ast)) )).

:- initialization(load_golden).

%% The 25 goal terms (as in test_pe_emit.pl; the elaborator receives
%% them with an EMPTY goal store — every one must come back ground).

golden_goal('atom-bare',        mod(graph, discrim)).
golden_goal('atom-dual-bare',   e5).
golden_goal(blend,              blend(mod(luna, 'D'), mod(luna, 'S'))).
golden_goal('blend-variadic',   blend(mod(luna, 'D'), mod(luna, 'S'), graph)).
golden_goal('dir-blend',        blend(mod(graph, discrim), mod(llm, element), mod(llm, subcat))).
golden_goal('distill-3tier',
    distill(e5(routing(e5, mod(sonnet, lineage), t([0.02, 0.03]), menus([10, 20]))))).
golden_goal('e5-auto',          e5(margin(t(0.03)))).
golden_goal('escaped-string',
    routing(e5, haiku, t([0.02]), menus([10]), manifest("a\"b\\c"))).
golden_goal('estimand-impl',
    lca_frac(simplemind, estimand(path), impl(structural))).
golden_goal('graph-judge',
    max(0.02,
        product(hop_decay(simplemind, gamma(0.6)), lca_frac(simplemind)),
        estimand(path))).
golden_goal('haiku-n10',        e5(routing(e5, haiku, t([0.02]), menus([10])))).
golden_goal('int-spelled-number',      margin(t(1))).
golden_goal('int-spelled-number-list', routing(e5, haiku, t([1]), menus([10]))).
golden_goal('kalman-fused',     kalman(mod(luna, 'D'), mod(luna, 'S'))).
golden_goal('lineage-haiku',
    lineage(pearltrees, mu(haiku), estimand(ancestry))).
golden_goal('margin-number',    margin(t(0.03))).
golden_goal('menu-required-int', menu(graph, n(10))).
golden_goal('mu-judge-kwarg',   lineage(simplewiki, mu(mod(sonnet, lineage)))).
golden_goal('neg-number',       lineage(fs, decay(-0.5))).
golden_goal('numeric-positional-literal', max(0.02, e5(margin(t(0.03))))).
golden_goal(pinned,
    pin(lineage(pearltrees, decay(0.85)), 'run/2026-07-25')).
golden_goal('sonnet-lin-n10',
    e5(routing(e5, mod(sonnet, lineage), t([0.02]), menus([10])))).
golden_goal('sonnet-lin-n20',
    e5(routing(e5, mod(sonnet, lineage), t([0.02, 0.03]), menus([10, 20])))).
golden_goal('substrate-atom',   fs).
golden_goal('utf8-string',
    routing(e5, haiku, t([0.02]), menus([10]), manifest("héllo·wörld"))).

%% Where-style spellings: the five repeated-atom rows, entering the
%% elaborator as term + BINDING STORE (the §12 form, now through the
%% fixpoint loop rather than pe_where).

where_case(blend,
    blend(mod(L, 'D'), mod(L, 'S')), [L = luna]).
where_case('blend-variadic',
    blend(mod(L, 'D'), mod(L, 'S'), graph), [L = luna]).
where_case('dir-blend',
    blend(mod(graph, discrim), mod(L, element), mod(L, subcat)), [L = llm]).
where_case('graph-judge',
    max(0.02, product(hop_decay(C, gamma(0.6)), lca_frac(C)), estimand(path)),
    [C = simplemind]).
where_case('kalman-fused',
    kalman(mod(L, 'D'), mod(L, 'S')), [L = luna]).

%% ============================================
%% Ground path: sealed golden bytes
%% ============================================

:- begin_tests(elab_golden_bytes).

test(empty_store_grounds_all_rows, [forall(golden_goal(Name, Goal))]) :-
    golden_row(Name, Identity, Full, _),
    elaborate(Goal, [], State),
    State = ground(Ground),
    pe_semantic(Ground, Sem),
    pe_full(Ground, F),
    Sem == Identity,
    F == Full.

test(binding_store_grounds_where_rows, [forall(where_case(Name, Term, Store))]) :-
    golden_row(Name, Identity, _, _),
    elaborate(Term, Store, State),
    State = ground(Ground),
    pe_semantic(Ground, Sem),
    Sem == Identity.

:- end_tests(elab_golden_bytes).

%% ============================================
%% Ground path: sealed STRUCTURE (resolved_ast oracle)
%% ============================================
%
% The elaborated ground term is converted to the bundle's resolved_ast
% shape and compared against the sealed JSON — structural verification
% with no Python involved.  Both sides are normalized to plain terms
% (dicts with anonymous tags are never ==) before comparison.

% normalize_json(+Value, -Term): dicts -> obj(SortedPairs), lists
% mapped, scalars as-is.
normalize_json(V, obj(Norm)) :-
    is_dict(V),
    !,
    dict_pairs(V, _, Pairs),
    msort(Pairs, Sorted),
    maplist([K-V0, K-N]>>normalize_json(V0, N), Sorted, Norm).
normalize_json(V, list(Norm)) :-
    is_list(V),
    !,
    maplist(normalize_json, V, Norm).
normalize_json(V, V).

% goal_ast_dict(+GroundGoal, -Dict): the resolved_ast shape, built by
% the same introspection discipline as pe_emit (registry mirror as
% Config), producing dicts that normalize equal to json_read_dict's.
goal_ast_dict(pin(E, P), D) :-
    !,
    goal_ast_dict(E, D0),
    atom_string(P, PS),
    Pins0 = D0.pins,
    append(Pins0, [PS], Pins),
    D = D0.put(pins, Pins).
goal_ast_dict(mod(B, M), D) :-
    !,
    goal_ast_dict(B, D0),
    atom_string(M, MS),
    Mods0 = D0.mods,
    append(Mods0, [MS], Mods),
    D = D0.put(mods, Mods).
goal_ast_dict(N, D) :-
    number(N),
    !,
    lex_plain(N, Lex),
    D = _{lexical: Lex, literal: true, value_type: "number"}.
goal_ast_dict(A, D) :-
    atom(A),
    pe_registry_mirror:pe_atom(A),
    !,
    atom_string(A, NameS),
    pe_output(A, Out),
    atom_string(Out, OutS),
    D = _{args: [], kind: "atom", kwargs: [], mods: [], name: NameS,
          output: OutS, pins: []}.
goal_ast_dict(Goal, D) :-
    compound(Goal),
    compound_name_arguments(Goal, Name, RawArgs),
    pe_registry_mirror:pe_operator(Name),
    !,
    split_kwargs(RawArgs, Name, Pos, Kw0),
    findall(K-V, ( pe_registry_mirror:pe_kwspec(Name, K, _, default(V)),
                   \+ memberchk(K-_, Kw0) ),
            Defaults),
    append(Kw0, Defaults, Kw1),
    msort(Kw1, Kw),
    maplist(goal_ast_dict, Pos, Args),
    maplist(kw_entry(Name), Kw, Kwargs),
    atom_string(Name, NameS),
    pe_output(Name, Out),
    atom_string(Out, OutS),
    D = _{args: Args, kind: "apply", kwargs: Kwargs, mods: [],
          name: NameS, output: OutS, pins: []}.

split_kwargs([], _, [], []).
split_kwargs([A|Rest], Op, Pos, [K-V|Kw]) :-
    compound(A),
    compound_name_arguments(A, K, [V]),
    pe_registry_mirror:pe_kwspec(Op, K, _, _),
    !,
    split_kwargs(Rest, Op, Pos, Kw).
split_kwargs([A|Rest], Op, [A|Pos], Kw) :-
    split_kwargs(Rest, Op, Pos, Kw).

kw_entry(Op, K-V, Entry) :-
    once(pe_registry_mirror:pe_kwspec(Op, K, Kind, _)),
    atom_string(K, KS),
    atom_string(Kind, KindS),
    (   value_kind_lex(Kind, V, Lex)
    ->  Entry = _{key: KS, lexical: Lex, value_type: KindS}
    ;   % node-valued kwarg (declared kind is an output type)
        goal_ast_dict(V, Node),
        Entry = _{key: KS, node: Node, value_type: KindS}
    ).

value_kind_lex(number, V, Lex)   :- lex_plain(V, Lex).
value_kind_lex(int, V, Lex)      :- lex_plain(V, Lex).
value_kind_lex(string, V, Lex)   :- lex_quoted(V, Lex).
value_kind_lex(estimand, V, Lex) :- lex_quoted(V, Lex).
value_kind_lex(impl, V, Lex)     :- lex_quoted(V, Lex).
value_kind_lex(number_list, V, Lex) :- lex_list(V, Lex).
value_kind_lex(int_list, V, Lex)    :- lex_list(V, Lex).

lex_plain(V, Lex) :- format(string(Lex), "~w", [V]).
lex_list(Vs, Lex) :-
    maplist([V, S]>>format(string(S), "~w", [V]), Vs, Ss),
    atomics_to_string(Ss, ",", Body),
    format(string(Lex), "[~w]", [Body]).
% the quoted lexical is the JSON-escaped, double-quoted spelling; the
% escaper is pe_emit's (test-only qualified read, byte-anchored here)
lex_quoted(V, Lex) :-
    pe_emit:json_escape(V, Esc),
    format(string(Lex), "\"~w\"", [Esc]).

atomics_to_string(Ss, Sep, Out) :-
    atomic_list_concat(Ss, Sep, A),
    atom_string(A, Out).

:- begin_tests(elab_resolved_ast_oracle).

test(structural_oracle_all_rows, [forall(golden_goal(Name, Goal))]) :-
    golden_row(Name, _, _, SealedAst),
    elaborate(Goal, [], ground(Ground)),
    goal_ast_dict(Ground, BuiltAst),
    normalize_json(SealedAst, N1),
    normalize_json(BuiltAst, N2),
    (   N1 == N2
    ->  true
    ;   format(user_error, "~w: structural mismatch~n  sealed ~q~n  built  ~q~n",
               [Name, N1, N2]),
        fail
    ).

:- end_tests(elab_resolved_ast_oracle).

%% ============================================
%% The fixpoint loop
%% ============================================

:- begin_tests(elab_fixpoint).

% The note's own observation as a test: discharging a binding makes a
% residual checkable.  Pass 1: has_type is nonground (residuates),
% C=simplemind discharges.  Pass 2: has_type is now ground, checks,
% discharges.  Result: fully ground.
test(binding_discharge_unlocks_residual) :-
    elaborate(lca_frac(C),
              [has_type(t1, substrate(C)), C = simplemind],
              State),
    State = ground(Ground),
    Ground == lca_frac(simplemind).

% A chain of bindings across passes: C = D needs D ground first.
test(binding_chain_terminates_and_grounds) :-
    elaborate(lca_frac(C),
              [C = D, D = simplemind],
              State),
    State = ground(lca_frac(simplemind)).

% Three-pass chain, order deliberately worst-case (most-dependent
% first) — termination is the point.
test(three_pass_chain) :-
    elaborate(hop_decay(A, gamma(G)),
              [A = B, B = simplemind, G = 0.6],
              State),
    State = ground(hop_decay(simplemind, gamma(0.6))).

% The ground result feeds pe_emit unchanged: chain then render.
test(chained_ground_renders_golden_bytes) :-
    golden_row('graph-judge', Identity, _, _),
    elaborate(max(0.02, product(hop_decay(C, gamma(0.6)), lca_frac(C)),
                  estimand(path)),
              [C = S, S = simplemind],
              ground(Ground)),
    pe_semantic(Ground, Sem),
    Sem == Identity.

:- end_tests(elab_fixpoint).

%% ============================================
%% Residual path — behavioral (no sealed oracle exists)
%% ============================================
%
% Fixtures carry the mandatory no_identity/1 marker (ruling 7).

% pattern_fixture(no_identity(LocalName), Term, Goals)
pattern_fixture(no_identity(px_two_residuals),
    product(hop_decay(C, gamma(0.6)), lca_frac(C)),
    [has_type(_X, substrate(C)), has_type(_Y, judge(_J))]).
pattern_fixture(no_identity(px_ground_term_residual_store),
    fs,
    [has_type(_X, substrate(_C))]).
pattern_fixture(no_identity(px_no_goals_open_term),
    lca_frac(_C),
    []).

:- begin_tests(elab_residual_behavior).

% Every fixture in this file carries the marker — the fence stays
% visible (ruling 7).
test(fixtures_carry_no_identity_marker) :-
    forall(clause(pattern_fixture(Marker, _, _), _),
           Marker = no_identity(_)).

% Residuals residuate: nonground goals come back in the store, term
% state is pattern/2, and the store holds PURE goals (origin never
% inside — ruling 3).
test(residuals_residuate_as_pattern) :-
    pattern_fixture(no_identity(px_two_residuals), Term, Goals),
    elaborate(Term, Goals, State),
    State = pattern(T, Store),
    T = product(_, _),
    length(Store, 2),
    forall(member(G, Store), G = has_type(_, _)).

% Discharge order-independence.  elaborate/3 COPIES its input (the
% copy-don't-mutate discipline reused from pe_where per the note), so
% two calls can never share live variables and cross-call == of live
% stores is impossible BY DESIGN — an implementation-contact finding
% reported in the elaborator report, not silently repaired.  The
% ==-identical claim holds at the level the canonical order is defined
% on: the numbered projection.  Live states are variants (=@=).
numbered_state(pattern(T, S), Numbered) :-
    copy_term(T-S, Numbered),
    numbervars(Numbered, 0, _).

test(order_independence_same_instances) :-
    pattern_fixture(no_identity(px_two_residuals), Term, [G1, G2]),
    elaborate(Term, [G1, G2], P1),
    elaborate(Term, [G2, G1], P2),
    P1 =@= P2,
    numbered_state(P1, N1),
    numbered_state(P2, N2),
    N1 == N2.

% ...including with a discharging binding mixed in at different
% positions.
test(order_independence_with_binding) :-
    Term = product(hop_decay(C, gamma(0.6)), lca_frac(C)),
    G1 = has_type(_X, substrate(C)),
    G2 = has_type(_Y, judge(_J)),
    B = (C = simplemind),
    elaborate(Term, [G1, B, G2], P1),
    elaborate(Term, [G2, G1, B], P2),
    P1 =@= P2,
    numbered_state(P1, N1),
    numbered_state(P2, N2),
    N1 == N2,
    % the binding grounded the substrate has_type's TYPE argument, but
    % _X is still free, so per §3's mode table the goal correctly
    % RESIDUATES rather than discharging — both residuals remain, one
    % of them now partially instantiated by the discharged binding
    P1 = pattern(_, Store),
    length(Store, 2),
    memberchk(has_type(_, substrate(simplemind)), Store),
    memberchk(has_type(_, judge(_)), Store).

% Within ONE call the ==-identity of the canonical store over live
% goals holds directly — canonical_store/3 returns the caller's own
% goal instances reordered (tested in elab_canonical_store below);
% this is where the ruling's ==-dedup is observable on live terms.

% A ground term with a residual store is a PATTERN state — residuals
% are part of what the term is (the conflation table's second row).
test(ground_term_residual_store_is_pattern) :-
    pattern_fixture(no_identity(px_ground_term_residual_store), Term, Goals),
    elaborate(Term, Goals, State),
    State = pattern(fs, [has_type(_, substrate(_))]).

% An open term with no goals is a pattern state with an empty store.
test(open_term_empty_store_is_pattern) :-
    pattern_fixture(no_identity(px_no_goals_open_term), Term, Goals),
    elaborate(Term, Goals, State),
    State = pattern(lca_frac(_), []).

% Origins ride alongside and align with the canonical store order;
% the store itself contains no origin wrappers (ruling 3).
test(origins_alongside_never_inside) :-
    Term = lca_frac(C),
    elaborate(Term,
              [has_type(_X, substrate(C))-'surface:X::substrate[C]'],
              State, ResidualOrigins),
    State = pattern(_, [Goal]),
    Goal = has_type(_, substrate(_)),
    ResidualOrigins = [Goal2-'surface:X::substrate[C]'],
    Goal == Goal2.

:- end_tests(elab_residual_behavior).

%% ============================================
%% v1.1: eager validation without goal rewriting (ruled on PR #4095)
%% ============================================

:- begin_tests(elab_eager_validation).

% THE STORE-INVARIANCE TEST, explicit per the ruling: a legal
% residuating goal elaborated with the eager check produces a store
% whose numbered projection is ==-identical to v1's output (captured
% from the v1 build before this refinement landed).  No goal
% splitting, no narrowed residuals — the check changes nothing on the
% legal path.
test(store_bytes_identical_to_v1_on_legal_path) :-
    elaborate(product(hop_decay(C, gamma(0.6)), lca_frac(C)),
              [has_type(_X, substrate(C))],
              pattern(T, S)),
    copy_term(T-S, Numbered),
    numbervars(Numbered, 0, _),
    % frozen v1 baseline, captured before the v1.1 edit:
    Numbered == product(hop_decay('$VAR'(0), gamma(0.6)), lca_frac('$VAR'(0)))
                - [has_type('$VAR'(1), substrate('$VAR'(0)))].

% The ruling's own example: ground type side, free subject —
% under-instantiated used to residuate silently; ground-and-false now
% throws at elaboration, named distinctly from a ground discharge
% failure.
test(ground_and_false_type_throws_eagerly,
     error(pe_elaborate(constraint_unsatisfiable(has_type(_, substrate(frobnicate)), origin(none))))) :-
    elaborate(fs, [has_type(_X, substrate(frobnicate))], _).

% A kind that no registered name inhabits can never ground: dormant
% forever under v1, an error now.
test(uninhabited_kind_throws_eagerly,
     error(pe_elaborate(constraint_unsatisfiable(has_type(_, frobnicate(_)), origin(none))))) :-
    elaborate(fs, [has_type(_X, frobnicate(_C))], _).

% Partially-ground mod spine: a ground base that is not a judge, or a
% base with no registered modifiers, can never satisfy the check.
test(unregistered_mod_base_throws_eagerly,
     error(pe_elaborate(constraint_unsatisfiable(_, _)))) :-
    elaborate(fs, [has_type(_X, judge(mod(frobnicate, _M)))], _).

test(modifierless_base_with_free_mod_throws_eagerly,
     error(pe_elaborate(constraint_unsatisfiable(_, _)))) :-
    % human is a registered judge but declares no modifiers, so
    % mod(human, _M) can never ground
    elaborate(fs, [has_type(_X, judge(mod(human, _M)))], _).

% The legal partial spines residuate exactly as before: inhabited kind
% with free subject; registered base with free modifier slot where a
% modifier exists; wholly unbound type side.
test(legal_partial_spines_still_residuate) :-
    elaborate(fs, [has_type(_A, substrate(_C))], pattern(fs, [has_type(_, substrate(_))])),
    elaborate(fs, [has_type(_B, judge(mod(sonnet, _M)))], pattern(fs, [has_type(_, judge(mod(sonnet, _)))])),
    elaborate(fs, [has_type(_D, _T)], pattern(fs, [has_type(_, _)])).

% The error carries the origin when one was supplied.
test(eager_error_names_origin,
     error(pe_elaborate(constraint_unsatisfiable(_, origin('surface:X::substrate[frobnicate]'))))) :-
    elaborate(fs,
              [has_type(_X, substrate(frobnicate))-'surface:X::substrate[frobnicate]'],
              _, _).

:- end_tests(elab_eager_validation).

%% ============================================
%% Canonical store: measured stability, re-asserted
%% ============================================

:- begin_tests(elab_canonical_store).

% Same-shape goals over distinct store-only variables: both kept
% (==-dedup only merges identical goals), and the NUMBERED projection
% is permutation-stable even though which live variable lands first is
% symmetric (the note's known boundary case, fenced behind peid-v1).
test(same_shape_goals_both_kept) :-
    G1 = has_type(_A, substrate(pearltrees)),
    G2 = has_type(_B, substrate(pearltrees)),
    canonical_store(fs, [G1, G2], Canon),
    length(Canon, 2).

% ==-identical duplicate goals collapse to one.
test(identical_goals_dedup) :-
    G = has_type(_A, substrate(pearltrees)),
    canonical_store(fs, [G, G], Canon),
    Canon == [G].

% Store-only variables: canonical order is input-order independent for
% distinguishable goals (the exact case raw msort/2 gets wrong when
% goals share a functor — re-asserting the probe's measurement at the
% API level).
test(store_only_vars_order_independent) :-
    G1 = has_type(_A, substrate(pearltrees)),
    G2 = has_type(_B, judge(haiku)),
    canonical_store(fs, [G1, G2], C1),
    canonical_store(fs, [G2, G1], C2),
    C1 == C2.

% Term-occurring variables are numbered by the TERM's traversal, so
% goal input order cannot reorder them.
test(term_vars_numbered_by_term_traversal) :-
    Term = product(hop_decay(C, gamma(0.6)), lca_frac(D)),
    G1 = has_type(t1, substrate(C)),
    G2 = has_type(t2, substrate(D)),
    canonical_store(Term, [G1, G2], A),
    canonical_store(Term, [G2, G1], B),
    A == B.

:- end_tests(elab_canonical_store).

%% ============================================
%% The generated mirror: hash check and cross-checks
%% ============================================

:- begin_tests(elab_mirror).

% The mirror loaded (or nothing in this file would run); assert the
% verify predicate also passes when called directly.
test(mirror_verify_passes) :-
    pe_registry_mirror:pe_mirror_verify.

test(mirror_version_is_v04) :-
    pe_registry_version('v0.4').

% Drift refusal: a tampered copy (zeroed hash) placed beside a copy of
% the sealed source REFUSES TO LOAD, in a separate process (separate
% because the module name is already loaded here).
test(tampered_mirror_refuses_to_load) :-
    here(Dir),
    tmp_file(pe_mirror_drift, Base),
    atomic_list_concat([Base, '_dir'], TmpRoot),
    atomic_list_concat([TmpRoot, '/sub'], SubDir),
    make_directory_path(SubDir),
    % sealed source copy at TmpRoot/process_cards.py
    atomic_list_concat([Dir, '/../process_cards.py'], Src),
    atomic_list_concat([TmpRoot, '/process_cards.py'], SrcCopy),
    copy_file(Src, SrcCopy),
    % tampered mirror at TmpRoot/sub/, expecting ../process_cards.py
    atomic_list_concat([Dir, '/pe_registry_mirror.pl'], Mirror),
    read_file_to_string(Mirror, MText, []),
    re_replace("pe_mirror_source_sha256\\('[0-9a-f]+'\\)"/a,
               "pe_mirror_source_sha256('0000000000000000000000000000000000000000000000000000000000000000')",
               MText, Tampered),
    atomic_list_concat([SubDir, '/pe_registry_mirror.pl'], TamperedPath),
    setup_call_cleanup(open(TamperedPath, write, WS),
                       write(WS, Tampered),
                       close(WS)),
    % the check goal must run as -g: a load-time initialization error
    % prints but does not set the exit status, so the verify predicate
    % is invoked explicitly and its throw drives the exit code
    process_create(path(swipl),
                   ['-g', 'pe_registry_mirror:pe_mirror_verify', '-t', 'halt',
                    TamperedPath],
                   [process(PID), stderr(null), stdout(null)]),
    process_wait(PID, Exit),
    Exit \== exit(0).

% Cross-check against the vNext registry FIXTURE — as data only.  The
% two implementations use different type vocabularies; the documented
% overlap mapping is corpus->substrate and judge->judge, checked for
% every fixture reference entry whose name the v0.4 mirror also
% registers.  Names outside the overlap are skipped, not guessed at.
test(vnext_fixture_type_crosscheck) :-
    here(Dir),
    atomic_list_concat(
        [Dir, '/../process_expression_vnext/testdata/frontend_registry_fixture.json'],
        Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        json_read_dict(S, Fixture),
        close(S)),
    dict_pairs(Fixture.entries, _, Entries),
    forall(( member(Name-Entry, Entries),
             Entry.get(kind) == "reference",
             vnext_type_maps(Entry.get(type), V04Output),
             atom_string(NameA, Name),   % dict keys are atoms already; normalize
             pe_output(NameA, _)
           ),
           pe_output(NameA, V04Output)).

vnext_type_maps("corpus", substrate).
vnext_type_maps("judge", judge).

% The fixture's margin entry marks t required; so does the mirror.
test(vnext_fixture_margin_required_crosscheck) :-
    pe_required(margin, t).

:- end_tests(elab_mirror).

%% ============================================
%% Fail-closed battery
%% ============================================

:- begin_tests(elab_fail_closed).

% Outside ruling 4(a)'s scope: unknown constraint forms error at first
% sight — their functor can never become known.
test(unknown_constraint_is_error,
     error(pe_elaborate(unknown_constraint(frobnicate(x), origin(none))))) :-
    elaborate(fs, [frobnicate(x)], _).

% ...and the error names the origin when one was supplied.
test(unknown_constraint_names_origin,
     error(pe_elaborate(unknown_constraint(_, origin('surface:frob'))))) :-
    elaborate(fs, [frobnicate(x)-'surface:frob'], _, _).

% A ground has_type whose check fails is an ERROR (typed diagnostic),
% never a silent residual: haiku is a judge, not a substrate.
test(failed_constraint_is_error_with_origin,
     error(pe_elaborate(constraint_failed(has_type(t1, substrate(haiku)),
                                          origin('surface:t1::substrate'))))) :-
    elaborate(fs, [has_type(t1, substrate(haiku))-'surface:t1::substrate'], _, _).

% Modifier-aware has_type: registered modifier discharges, unknown
% modifier fails the check.
test(modified_judge_discharges) :-
    elaborate(fs, [has_type(j, judge(mod(sonnet, lineage)))], ground(fs)).

test(unknown_modifier_fails,
     error(pe_elaborate(constraint_failed(_, _)))) :-
    elaborate(fs, [has_type(j, judge(mod(sonnet, bogus)))], _).

% Binding validation is pe_where's, reused: duplicates and malformed
% bindings are refused with pe_where's own error terms.
test(duplicate_bindings_refused,
     error(pe_where(duplicate_binding_for_one_variable))) :-
    elaborate(lca_frac(C), [C = simplemind, C = fs], _).

test(malformed_binding_refused,
     error(pe_where(bad_binding(foo = bar)))) :-
    elaborate(fs, [foo = bar], _).

% Dead bindings hide typos; a binding is live if its variable occurs
% in the term or another goal.
test(dead_binding_refused,
     error(pe_elaborate(dead_binding(_)))) :-
    elaborate(fs, [_C = simplemind], _).

test(binding_live_via_other_goal_not_dead) :-
    elaborate(fs, [has_type(t1, substrate(C)), C = simplemind], ground(fs)).

% The ratified pin refusals, now through the elaborator: a binding
% variable in a pin position...
test(binding_in_pin_position_refused,
     error(pe_elaborate(binding_rejected(binding_reaches_pin_channel(pin_name), _)))) :-
    elaborate(pin(lineage(pearltrees, decay(0.85)), P), [P = 'run/1'], _).

% ...and a pin smuggled in as a binding value.
test(pin_as_binding_value_refused,
     error(pe_elaborate(binding_rejected(illegal_binding_value(_, _), _)))) :-
    elaborate(lca_frac(C), [C = pin(fs, 'run/1')], _).

% Structural legality with the position named, through the loop.
test(illegal_binding_value_names_position,
     error(pe_elaborate(binding_rejected(illegal_binding_value(foo, at(kwarg(margin, t))), _)))) :-
    elaborate(e5(margin(t(T))), [T = foo], _).

% Malformed pair list on the /4 entry.
test(bad_goal_pairs_refused,
     error(pe_elaborate(bad_goal_pairs(_)))) :-
    elaborate(fs, [not_a_pair], _, _).

:- end_tests(elab_fail_closed).

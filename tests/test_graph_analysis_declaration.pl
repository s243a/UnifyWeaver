% test_graph_analysis_declaration.pl
%
% INCREMENT 3 of WAM_RUST_BRIDGE_CLUSTERING.md: the graph_analysis/4 pipeline
% declaration collapses the verbose per-predicate foreign_lowering blocks of
% increments 1+2 into ONE statement. This test proves the bundling is exact:
% graph_analysis_expand/5 produces foreign_lowering specs that are TERM-IDENTICAL
% to the hand-written blocks, and that generate BYTE-IDENTICAL Rust setup code
% (rust_foreign_setup_code) per predicate.
%
% No cargo needed — pure declaration-expansion identity. The behavioural e2e
% (dispatch) is covered by test_wam_rust_{bridge,cluster}_foreign_dispatch.pl,
% which are unchanged by this increment (graph_analysis adds Prolog only).

:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [ graph_analysis_expand/5, graph_analysis/4, graph_analysis_options/2 ]).

% The author's mu facts (read by graph_analysis into register_ffi_mu).
:- dynamic category_mu/2.
category_mu('Matter', 1.0).
category_mu('Energy', 0.5).

% The shared declaration inputs (declared ONCE).
edge_pred(category_parent).
inputs([mu(category_mu), threshold(0.3)]).
stages([sketches(k=32), bridges, clusters]).
queries([category_bridge_score/2, bridge/3, category_cluster/2, cluster_members/2]).

% ---- BEFORE: the hand-written verbose blocks (increments 1+2 style, k threaded) ----
% This is exactly what an author wrote per predicate before graph_analysis/4.
handwritten(category_bridge_score/2, category_bridge_score, deterministic, tuple(1)).
handwritten(bridge/3,                bridge,                stream,        tuple(3)).
handwritten(category_cluster/2,      category_cluster,      deterministic, tuple(1)).
handwritten(cluster_members/2,       cluster_members,       stream,        tuple(1)).

handwritten_spec(Pred/Arity, MuData,
        foreign_lowering(foreign_predicate(Pred/Arity,
          [ register_foreign_native_kind(Pred/Arity, NativeKind),
            register_foreign_result_mode(Pred/Arity, Mode),
            register_foreign_result_layout(Pred/Arity, Layout),
            register_foreign_string_config(Pred/Arity, edge_pred, category_parent),
            register_foreign_string_config(Pred/Arity, mu_pred, category_mu),
            register_foreign_f64_config(Pred/Arity, threshold, 0.3),
            register_foreign_usize_config(Pred/Arity, k, 32),
            register_ffi_mu(Pred/Arity, category_mu, MuData) ],
          []))) :-
    handwritten(Pred/Arity, NativeKind, Mode, Layout).

:- begin_tests(graph_analysis_declaration).

% (1) graph_analysis_expand reproduces the hand-written foreign_lowering terms EXACTLY.
test(expansion_is_term_identical_to_handwritten) :-
    edge_pred(E), inputs(I), stages(S), queries(Qs),
    graph_analysis_expand(E, I, S, Qs, Options),
    MuData = ['Matter'-1.0, 'Energy'-0.5],
    findall(Spec, ( member(Q, Qs), handwritten_spec(Q, MuData, Spec) ), Expected),
    ( Options == Expected
    -> true
    ;  forall(nth0(N, Options, O),
              ( nth0(N, Expected, X),
                ( O == X -> true
                ; format(user_error, "~nMISMATCH at ~w:~n got: ~q~n exp: ~q~n", [N, O, X]) ) )),
       fail ).

% (2) Each expanded spec generates Rust setup code identical to the hand-written spec.
test(generated_rust_setup_is_identical) :-
    edge_pred(E), inputs(I), stages(S), queries(Qs),
    graph_analysis_expand(E, I, S, Qs, Options),
    MuData = ['Matter'-1.0, 'Energy'-0.5],
    forall(member(Q, Qs),
       ( handwritten_spec(Q, MuData, foreign_lowering(foreign_predicate(Q, HOps, []))),
         memberchk(foreign_lowering(foreign_predicate(Q, GOps, [])), Options),
         wam_rust_target:rust_foreign_setup_code(HOps, HRust),
         wam_rust_target:rust_foreign_setup_code(GOps, GRust),
         ( HRust == GRust -> true
         ; format(user_error, "~nRust setup differs for ~w~n hand: ~w~n gen : ~w~n", [Q, HRust, GRust]),
           fail ) )).

% (3) The generated Rust setup carries the expected register_foreign_* lines (sanity).
test(generated_rust_has_native_kind_and_configs) :-
    edge_pred(E), inputs(I), stages(S), queries(Qs),
    graph_analysis_expand(E, I, S, Qs, Options),
    memberchk(foreign_lowering(foreign_predicate(category_cluster/2, Ops, [])), Options),
    wam_rust_target:rust_foreign_setup_code(Ops, Rust),
    sub_atom(Rust, _, _, _, 'register_foreign_native_kind("category_cluster/2", "category_cluster")'),
    sub_atom(Rust, _, _, _, 'register_foreign_result_mode("category_cluster/2", "deterministic")'),
    sub_atom(Rust, _, _, _, 'register_foreign_string_config("category_cluster/2", "edge_pred", "category_parent")'),
    sub_atom(Rust, _, _, _, 'register_foreign_string_config("category_cluster/2", "mu_pred", "category_mu")'),
    sub_atom(Rust, _, _, _, 'register_foreign_f64_config("category_cluster/2", "threshold", 0.3)'),
    sub_atom(Rust, _, _, _, 'register_ffi_mu("category_mu", &[("Matter", 1.0), ("Energy", 0.5)])').

% (4) The directive form bundles into one options list + the flat query-pred list.
test(directive_bundles_all_queries) :-
    retractall(wam_rust_target:graph_analysis_decl(_,_,_,_)),
    graph_analysis(category_parent, [mu(category_mu), threshold(0.3)],
                   [sketches(k=32), bridges, clusters],
                   [category_bridge_score/2, bridge/3, category_cluster/2, cluster_members/2]),
    graph_analysis_options(Options, QueryPreds),
    QueryPreds == [category_bridge_score/2, bridge/3, category_cluster/2, cluster_members/2],
    length(Options, 4),
    forall(member(O, Options), O = foreign_lowering(foreign_predicate(_,_,_))),
    retractall(wam_rust_target:graph_analysis_decl(_,_,_,_)).

:- end_tests(graph_analysis_declaration).

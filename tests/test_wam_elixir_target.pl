:- encoding(utf8).
% Test suite for WAM-to-Elixir transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_target.pl

:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_lowered_emitter',
              [lower_predicate_to_elixir/4, classify_predicate/4,
               extract_facts/3, extract_arg1_index/3,
               tier2_purity_eligible/3, par_wrap_segment/4]).
% For Tier-2 purity-gate tests — user-annotation producer reads
% clause_body_analysis:order_independent/1 dynamic facts.
:- use_module('../src/unifyweaver/core/clause_body_analysis').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Core generation tests

test_step_generation :-
    Test = 'WAM-Elixir: step/2 case expression generation',
    (   compile_step_wam_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'case instr do'),
        sub_string(S, _, _, _, ':get_constant')
    ->  pass(Test)
    ;   fail_test(Test, 'step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-Elixir: helper functions generation',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def run(state)'),
        sub_string(S, _, _, _, 'def backtrack(state)'),
        sub_string(S, _, _, _, 'def unwind_trail(state, mark)')
    ->  pass(Test)
    ;   fail_test(Test, 'helper generation missing expected content')
    ).

test_runtime_assembly :-
    Test = 'WAM-Elixir: full runtime assembly',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'defmodule WamRuntime'),
        sub_string(S, _, _, _, 'defmodule WamState'),
        sub_string(S, _, _, _, 'defstruct'),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'def run(state)')
    ->  pass(Test)
    ;   fail_test(Test, 'runtime assembly missing expected content')
    ).

test_instruction_count :-
    Test = 'WAM-Elixir: instruction arm count',
    (   findall(N, wam_elixir_case(N, _), Cases),
        length(Cases, Count),
        Count >= 26
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 26 instruction arms')
    ).

%% Instruction category tests

test_head_unification_instructions :-
    Test = 'WAM-Elixir: head unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(get_constant, Cases),
        member(get_variable, Cases),
        member(get_value, Cases),
        member(get_structure, Cases),
        member(get_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing head unification instruction arms')
    ).

test_body_construction_instructions :-
    Test = 'WAM-Elixir: body construction instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(put_constant, Cases),
        member(put_variable, Cases),
        member(put_value, Cases),
        member(put_structure, Cases),
        member(put_list, Cases),
        member(set_variable, Cases),
        member(set_value, Cases),
        member(set_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing body construction instruction arms')
    ).

test_unification_instructions :-
    Test = 'WAM-Elixir: unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_control_flow_instructions :-
    Test = 'WAM-Elixir: control flow instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(call, Cases),
        member(execute, Cases),
        member(proceed, Cases),
        member(allocate, Cases),
        member(deallocate, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing control flow instruction arms')
    ).

test_choice_point_instructions :-
    Test = 'WAM-Elixir: choice point instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_bytecode :-
    Test = 'WAM-Elixir: choice point code saves/restores state',
    (   wam_elixir_case(try_me_else, TryCode),
        wam_elixir_case(retry_me_else, RetryCode),
        wam_elixir_case(trust_me, TrustCode),
        sub_string(TryCode, _, _, _, 'choice_points'),
        sub_string(RetryCode, _, _, _, 'choice_points'),
        sub_string(TrustCode, _, _, _, 'choice_points')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point code missing state save/restore')
    ).

test_builtin_call_delegates :-
    Test = 'WAM-Elixir: builtin_call delegates to execute_builtin',
    (   wam_elixir_case(builtin_call, Code),
        sub_string(Code, _, _, _, 'execute_builtin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call does not delegate to execute_builtin')
    ).

%% Code style tests

test_elixir_idioms :-
    Test = 'WAM-Elixir: uses Elixir idioms (pipe, pattern match)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '|>'),
        sub_string(S, _, _, _, 'Map.put'),
        sub_string(S, _, _, _, 'match?')
    ->  pass(Test)
    ;   fail_test(Test, 'missing Elixir idioms')
    ).

test_immutable_state_updates :-
    Test = 'WAM-Elixir: uses immutable state updates (%{state | ...})',
    (   wam_elixir_case(put_constant, Code),
        sub_string(Code, _, _, _, '%{state |')
    ->  pass(Test)
    ;   fail_test(Test, 'not using immutable struct update syntax')
    ).

test_functional_run_loop :-
    Test = 'WAM-Elixir: run loop is recursive (not imperative)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'run(new_state)')
    ->  pass(Test)
    ;   fail_test(Test, 'run loop not recursive')
    ).

%% Phase A fact-shape classification tests

:- dynamic
    phase_a_test:small_fact/2,
    phase_a_test:big_fact/2,
    phase_a_test:rule/2,
    phase_a_test:variable_head/1.

phase_a_fixture_setup :-
    % Small fact-only predicate (4 clauses).
    retractall(phase_a_test:small_fact(_, _)),
    forall(member(X-Y, [a-1, b-2, c-3, d-4]),
           assertz(phase_a_test:small_fact(X, Y))),
    % Big fact-only predicate (150 clauses — above default threshold of 100).
    retractall(phase_a_test:big_fact(_, _)),
    forall(between(1, 150, I), (
        atom_number(K, I),
        assertz(phase_a_test:big_fact(K, I))
    )),
    % Rule-bearing predicate (non-fact-only).
    retractall(phase_a_test:rule(_, _)),
    assertz((phase_a_test:rule(X, Y) :- phase_a_test:small_fact(X, Y))),
    % Variable-head fact (for first_arg_groundness=all_variable).
    retractall(phase_a_test:variable_head(_)),
    assertz((phase_a_test:variable_head(_X))).

compile_and_segments(PredArity, Segments) :-
    wam_target:compile_predicate_to_wam(phase_a_test:PredArity, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments).

test_classify_small_fact_only :-
    Test = 'Phase A: small fact-only predicate → compiled (below threshold)',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs, [],
                       fact_shape_info(NCl, FactOnly, G, Layout)),
    (   NCl == 4, FactOnly == true, G == all_ground, Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, classified(NCl, FactOnly, G, Layout))
    ).

test_classify_big_fact_only :-
    Test = 'Phase A: big fact-only predicate → inline_data',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs, [],
                       fact_shape_info(NCl, FactOnly, _G, Layout)),
    (   NCl == 150, FactOnly == true, Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, classified(NCl, FactOnly, Layout))
    ).

test_classify_rule :-
    Test = 'Phase A: rule-bearing predicate → compiled (not fact-only)',
    phase_a_fixture_setup,
    compile_and_segments(rule/2, Segs),
    classify_predicate(rule/2, Segs, [],
                       fact_shape_info(_NCl, FactOnly, _G, Layout)),
    (   FactOnly == false, Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, classified(FactOnly, Layout))
    ).

test_classify_variable_head :-
    Test = 'Phase A: all-variable first arg → first_arg=all_variable',
    phase_a_fixture_setup,
    compile_and_segments(variable_head/1, Segs),
    classify_predicate(variable_head/1, Segs, [],
                       fact_shape_info(_, _, G, _)),
    (   G == all_variable
    ->  pass(Test)
    ;   fail_test(Test, groundness(G))
    ).

test_classify_user_override_layout :-
    Test = 'Phase A: user fact_layout/2 override wins over default',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    Opts = [fact_layout(small_fact/2, external_source(tsv("/tmp/x.tsv")))],
    classify_predicate(small_fact/2, Segs, Opts,
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = external_source(_)
    ->  pass(Test)
    ;   fail_test(Test, layout(Layout))
    ).

test_classify_user_override_threshold :-
    Test = 'Phase A: fact_count_threshold(1) promotes small fact set to inline_data',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs, [fact_count_threshold(1)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, layout(Layout))
    ).

test_shape_comment_in_generated_module :-
    Test = 'Phase A: generated module carries fact-shape comment',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, 'Fact-shape classification'),
        sub_string(Code, _, _, _, 'clauses=150'),
        sub_string(Code, _, _, _, 'fact_only=true'),
        sub_string(Code, _, _, _, 'layout=inline_data')
    ->  pass(Test)
    ;   fail_test(Test, 'comment missing or wrong content')
    ).

test_phase_a_preserves_compiled_output :-
    Test = 'Phase A: small predicate with default threshold uses compiled defps',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    lower_predicate_to_elixir(small_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, 'defp clause_'),
        sub_string(Code, _, _, _, 'def run(%WamRuntime.WamState{}')
    ->  pass(Test)
    ;   fail_test(Test, 'expected compiled-shape output not present')
    ).

%% Phase B inline_data emission tests

test_extract_facts_simple :-
    Test = 'Phase B: extract_facts yields tuple list for fact-only predicate',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_facts(Segments, 2, Literal),
    (   sub_string(Literal, _, _, _, '{"a", "1"}'),
        sub_string(Literal, _, _, _, '{"d", "4"}')
    ->  pass(Test)
    ;   fail_test(Test, Literal)
    ).

test_phase_b_emits_inline_data_when_chosen :-
    Test = 'Phase B: inline_data layout emits @facts and no defp clauses',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    % Phase C (PR #1525) added first-arg indexing: the emitted `run/1`
    % picks between `@facts` (unbound arg1) and the `@facts_by_arg1`
    % indexed subset, binds the result to a local `facts` variable,
    % then streams that. Assert the indexed shape, not the pre-Phase-C
    % direct `stream_facts(state, @facts, 2)` call.
    (   sub_string(Code, _, _, _, '@facts ['),
        sub_string(Code, _, _, _, 'WamRuntime.stream_facts(state, facts, 2)'),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline_data shape missing or leaked defp')
    ).

test_phase_b_variable_head_becomes_sentinel :-
    Test = 'Phase B: variable head arg emits :_var sentinel',
    phase_a_fixture_setup,
    % Force variable_head/1 to inline_data via low threshold.
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    lower_predicate_to_elixir(variable_head/1, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    (   sub_string(Code, _, _, _, ':_var'),
        sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'variable_head did not lower to :_var sentinel')
    ).

test_phase_b_fallback_on_unextractable :-
    Test = 'Phase B: unextractable fact falls back to compiled (safe default)',
    phase_a_fixture_setup,
    % Inject a compound-head fact; extraction should fail and fall back.
    retractall(phase_a_test:compound_head(_)),
    assertz((phase_a_test:compound_head(foo(1, 2)))),
    wam_target:compile_predicate_to_wam(phase_a_test:compound_head/1, [], WamCode),
    % Force inline_data via override, but extraction should fail and
    % lower_predicate_to_elixir/4 falls through to compiled.
    lower_predicate_to_elixir(compound_head/1, WamCode,
                              [module_name('TestMod'),
                               fact_layout(compound_head/1, inline_data([]))], Code),
    (   sub_string(Code, _, _, _, 'defp clause_'),
        \+ sub_string(Code, _, _, _, '@facts [')
    ->  pass(Test)
    ;   fail_test(Test, 'expected fallback to compiled did not occur')
    ).

%% WAM constant quoting tests

:- dynamic phase_a_test:comma_atom/2.

test_quote_wam_constant_plain :-
    Test = 'Quoting: identifier-like atom passes through unquoted',
    wam_target:quote_wam_constant('foo_bar_123', Str),
    (   Str == "foo_bar_123"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_quote_wam_constant_comma :-
    Test = 'Quoting: atom containing comma is single-quoted',
    wam_target:quote_wam_constant('Has,comma', Str),
    (   Str == "'Has,comma'"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_quote_wam_constant_escape :-
    Test = 'Quoting: atom containing single-quote is escaped',
    wam_target:quote_wam_constant('it\'s', Str),
    (   Str == "'it\\'s'"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_tokenize_unquoted :-
    Test = 'Tokenizer: plain line splits on spaces and commas',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant foo, A1", Tokens),
    (   Tokens == ["get_constant", "foo", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_tokenize_quoted_atom_with_comma :-
    Test = 'Tokenizer: quoted atom containing comma stays one token',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant 'Has,comma', A1", Tokens),
    (   Tokens == ["get_constant", "Has,comma", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_tokenize_quoted_atom_with_escape :-
    Test = 'Tokenizer: quoted atom with \\\' escape unquotes to literal',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant 'it\\'s', A1", Tokens),
    (   Tokens == ["get_constant", "it's", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_round_trip_comma_atom :-
    Test = 'Round trip: atom with comma survives WAM-text -> tokenizer',
    retractall(phase_a_test:comma_atom(_, _)),
    assertz((phase_a_test:comma_atom('Has,comma', value))),
    wam_target:compile_predicate_to_wam(phase_a_test:comma_atom/2, [], WamCode),
    lower_predicate_to_elixir(comma_atom/2, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    % Should reach inline_data and emit the atom as an Elixir string.
    (   sub_string(Code, _, _, _, '{"Has,comma", "value"}'),
        \+ sub_string(Code, _, _, _, 'raw:')
    ->  pass(Test)
    ;   fail_test(Test, 'comma-atom round-trip failed')
    ).

%% Phase C first-argument indexing tests

test_extract_arg1_index_ground :-
    Test = 'Phase C: all-ground first arg → indexed map literal',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_arg1_index(Segments, 2, IndexResult),
    (   IndexResult = indexed(Lit),
        sub_string(Lit, _, _, _, '"a" => [{"a", "1"}]'),
        sub_string(Lit, _, _, _, '"d" => [{"d", "4"}]')
    ->  pass(Test)
    ;   fail_test(Test, IndexResult)
    ).

test_extract_arg1_index_variable :-
    Test = 'Phase C: variable first arg → no_index',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_arg1_index(Segments, 1, IndexResult),
    (   IndexResult == no_index
    ->  pass(Test)
    ;   fail_test(Test, IndexResult)
    ).

test_phase_c_indexed_module_emission :-
    Test = 'Phase C: indexed predicate emits @facts_by_arg1 and dispatching run/1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, '@facts_by_arg1 %{'),
        sub_string(Code, _, _, _, 'case arg1 do'),
        sub_string(Code, _, _, _, 'Map.get(@facts_by_arg1, key, [])')
    ->  pass(Test)
    ;   fail_test(Test, 'indexed shape missing')
    ).

test_phase_c_index_policy_none :-
    Test = 'Phase C: fact_index_policy(none) suppresses @facts_by_arg1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode,
                              [module_name('TestMod'), fact_index_policy(none)], Code),
    (   sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, '@facts_by_arg1')
    ->  pass(Test)
    ;   fail_test(Test, 'index was not suppressed by policy(none)')
    ).

test_phase_c_variable_head_no_index :-
    Test = 'Phase C: variable head uses flat-only inline_data (no index block)',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    lower_predicate_to_elixir(variable_head/1, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    (   sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, '@facts_by_arg1')
    ->  pass(Test)
    ;   fail_test(Test, 'variable-head emitted unexpected index')
    ).

%% Phase D external_source tests

test_phase_d_emits_external_source_shape :-
    Test = 'Phase D: fact_layout external_source emits FactSource-facade run/1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    Opts = [module_name('TestMod'),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    lower_predicate_to_elixir(big_fact/2, WamCode, Opts, Code),
    (   sub_string(Code, _, _, _, '@pred_indicator "big_fact/2"'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSourceRegistry.lookup!'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSource.stream_all'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSource.lookup_by_arg1'),
        \+ sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'external_source shape missing or bled through to other layouts')
    ).

test_phase_d_external_source_preserves_shared_preprocess_metadata :-
    Test = 'Phase D: external_source preserves shared preprocess metadata in generated module',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    setup_call_cleanup(
        assertz(user:preprocess(big_fact/2, exact_hash_index([key([1]), values([2])]))),
        (   Opts = [module_name('TestMod'),
                    fact_layout(big_fact/2, external_source(tsv_marker))],
            once(lower_predicate_to_elixir(big_fact/2, WamCode, Opts, Code)),
            contains_string(Code, '@external_source_spec "tsv_marker"'),
            contains_string(Code, '@external_source_metadata %{source_spec: @external_source_spec, preprocess: %{'),
            contains_string(Code, 'source: "shared_preprocess"'),
            contains_string(Code, 'mode: "artifact"'),
            contains_string(Code, 'kind: "exact_hash_index"'),
            contains_string(Code, 'format: "exact_hash_index"'),
            contains_string(Code, 'access_contracts: ["arg_position_lookup(1)", "exact_key_lookup", "grouped_values_lookup([2])", "scan"]'),
            contains_string(Code, 'options: ["key([1])", "values([2])"]'),
            contains_string(Code, 'def external_source_metadata, do: @external_source_metadata')
        ->  pass(Test)
        ;   fail_test(Test, 'shared preprocess metadata missing from external_source module')
        ),
        maybe_abolish_test_predicate(preprocess/2)
    ).

test_phase_d_runtime_emits_fact_source :-
    Test = 'Phase D: runtime assembly emits FactSource behaviour + Tsv adaptor + Registry',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource do'),
        sub_string(RuntimeCode, _, _, _, '@callback open'),
        sub_string(RuntimeCode, _, _, _, '@callback stream_all'),
        sub_string(RuntimeCode, _, _, _, '@callback lookup_by_arg1'),
        sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.Tsv do'),
        sub_string(RuntimeCode, _, _, _, '@behaviour WamRuntime.FactSource'),
        sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSourceRegistry do'),
        sub_string(RuntimeCode, _, _, _, ':persistent_term')
    ->  pass(Test)
    ;   fail_test(Test, 'FactSource runtime pieces missing')
    ).

test_phase_d_external_beats_inline_override :-
    Test = 'Phase D: external_source user override preempts default inline_data',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    % big_fact defaults to inline_data (150 clauses > threshold 100).
    % Explicit external_source must win.
    Opts = [module_name('TestMod'),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    lower_predicate_to_elixir(big_fact/2, WamCode, Opts, Code),
    (   sub_string(Code, _, _, _, '(external_source)'),
        \+ sub_string(Code, _, _, _, '(inline_data)')
    ->  pass(Test)
    ;   fail_test(Test, 'external_source did not win over default inline_data')
    ).

%% Phase E pluggable layout policy tests

test_phase_e_auto_matches_pre_phase_e :-
    Test = 'Phase E: auto policy is equivalent to pre-Phase-E default',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs, [fact_layout_policy(auto)],
                       fact_shape_info(_, _, _, Layout1)),
    classify_predicate(big_fact/2, Segs, [],
                       fact_shape_info(_, _, _, Layout2)),
    (   Layout1 == Layout2, Layout1 = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, mismatch(Layout1, Layout2))
    ).

test_phase_e_compiled_only_forces_compiled :-
    Test = 'Phase E: compiled_only policy forces big fact set to compiled',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs,
                       [fact_layout_policy(compiled_only)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_inline_eager_ignores_threshold :-
    Test = 'Phase E: inline_eager picks inline_data even below threshold',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs,
                       [fact_layout_policy(inline_eager)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_inline_eager_respects_fact_only :-
    Test = 'Phase E: inline_eager still falls to compiled for rule-bearing',
    phase_a_fixture_setup,
    compile_and_segments(rule/2, Segs),
    classify_predicate(rule/2, Segs,
                       [fact_layout_policy(inline_eager)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_user_override_preempts_policy :-
    Test = 'Phase E: user fact_layout/2 preempts any policy',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    Opts = [fact_layout_policy(compiled_only),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    classify_predicate(big_fact/2, Segs, Opts,
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = external_source(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

%% cost_aware policy tests

test_cost_aware_promotes_big_fact_set :-
    Test = 'cost_aware: 150-clause arity-2 (score 300) > default threshold 200 → inline_data',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs, [fact_layout_policy(cost_aware)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_cost_aware_keeps_small_preds_compiled :-
    Test = 'cost_aware: 4-clause arity-2 (score 8) < default threshold → compiled',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs, [fact_layout_policy(cost_aware)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_cost_aware_threshold_override :-
    Test = 'cost_aware: fact_cost_threshold(5) promotes 4-clause predicate (score 8 > 5)',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs,
                       [fact_layout_policy(cost_aware), fact_cost_threshold(5)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_cost_aware_respects_fact_only :-
    Test = 'cost_aware: rule-bearing predicate stays compiled regardless of score',
    phase_a_fixture_setup,
    compile_and_segments(rule/2, Segs),
    classify_predicate(rule/2, Segs,
                       [fact_layout_policy(cost_aware), fact_cost_threshold(1)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_ets_adaptor_emitted_in_runtime :-
    Test = 'ETS adaptor: runtime assembly emits FactSource.Ets',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.Ets do'),
        sub_string(RuntimeCode, _, _, _, ':ets.tab2list'),
        sub_string(RuntimeCode, _, _, _, ':ets.lookup')
    ->  pass(Test)
    ;   fail_test(Test, 'ETS adaptor missing from emitted runtime')
    ).

test_sqlite_adaptor_emitted_in_runtime :-
    Test = 'SQLite adaptor: runtime assembly emits FactSource.Sqlite',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.Sqlite do'),
        sub_string(RuntimeCode, _, _, _, '@behaviour WamRuntime.FactSource'),
        sub_string(RuntimeCode, _, _, _, 'defstruct [:db, :query_all, :query_by_arg1, :arity]'),
        sub_string(RuntimeCode, _, _, _, 'query_by_arg1: q_by1')
    ->  pass(Test)
    ;   fail_test(Test, 'SQLite adaptor missing expected structure')
    ).

test_sqlite_adaptor_uses_indirect_module_resolution :-
    Test = 'SQLite adaptor: uses Module.concat so runtime compiles without :exqlite',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Critical: the adaptor must NOT reference Exqlite.Sqlite3 as a
    % literal module call — that would generate compile-time warnings
    % in drivers without :exqlite. Module.concat/1 defers resolution
    % to call time.
    (   sub_string(RuntimeCode, _, _, _, 'Module.concat([Exqlite, Sqlite3])'),
        sub_string(RuntimeCode, _, _, _, 'apply(mod,'),
        % Sanity: no literal `Exqlite.Sqlite3.` call anywhere.
        \+ sub_string(RuntimeCode, _, _, _, 'Exqlite.Sqlite3.open')
    ->  pass(Test)
    ;   fail_test(Test, 'SQLite adaptor has literal Exqlite references — will warn without dep')
    ).

%% Tier-2 infrastructure tests (see docs/design/WAM_TIERED_LOWERING.md)
%% These exercise precondition scaffolding only — PR2 wires
%% par_wrap_segment/3 on top of them.

test_tier2_wamstate_has_parallel_depth :-
    Test = 'Tier-2 infra: WamState defstruct carries parallel_depth: 0',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'parallel_depth: 0')
    ->  pass(Test)
    ;   fail_test(Test, 'parallel_depth field missing from emitted WamState')
    ).

test_tier2_aggregate_helpers_emitted :-
    Test = 'Tier-2 infra: WamRuntime emits in_forkable_aggregate_frame?/1 + merge_into_aggregate/2',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def in_forkable_aggregate_frame?(state)'),
        sub_string(S, _, _, _, 'def merge_into_aggregate(state, branch_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'aggregate-frame helpers missing from emitted runtime')
    ).

test_tier2_aggregate_forkable_types :-
    Test = 'Tier-2 infra: forkable-frame check covers findall + aggregate_all',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, ':findall -> true'),
        sub_string(S, _, _, _, ':aggregate_all -> true')
    ->  pass(Test)
    ;   fail_test(Test, 'emitted forkable predicate does not cover both findall and aggregate_all')
    ).

test_tier2_purity_gate_rejects_unknown :-
    Test = 'Tier-2 infra: purity gate fails for unknown predicate',
    (   \+ tier2_purity_eligible(no_such_pred, 2, _)
    ->  pass(Test)
    ;   fail_test(Test, 'gate should have rejected predicate with no purity certificate')
    ).

test_tier2_purity_gate_accepts_declared :-
    Test = 'Tier-2 infra: purity gate accepts user-declared parallel predicate (confidence 1.0)',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_test_pred/2)),
        (   tier2_purity_eligible(tier2_test_pred, 2, Cert),
            Cert = purity_cert(pure, declared, Conf, _Reasons),
            Conf >= 0.85
        ->  pass(Test)
        ;   fail_test(Test, 'gate did not accept declared parallel predicate')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_test_pred/2))
    ).

%% par_wrap_segment/4 tests — exercise the three static gates (purity,
%% clause count, kill-switch) plus the shape of the emitted super-wrapper.

three_segment_fixture([
    'clause_start'-[1-try_me_else(l_b), 2-proceed],
    'l_b'-[3-retry_me_else(l_c), 4-proceed],
    'l_c'-[5-trust_me, 6-proceed]
]).

two_segment_fixture([
    'clause_start'-[1-try_me_else(l_b), 2-proceed],
    'l_b'-[3-trust_me, 4-proceed]
]).

test_par_wrap_segment_emits_super_wrapper :-
    Test = 'Tier-2 wrapper: 3-clause declared-pure predicate emits cond-based super-wrapper',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [], Code),
            Code \= "",
            % Entry func name derives from the first segment; fixture
            % uses 'clause_start' atom so segment_func_name emits
            % 'clause_ClauseStart'. Assertion checks the surface shape,
            % not a specific hard-coded name.
            sub_string(Code, _, _, _, 'defp clause_ClauseStart(state) do'),
            sub_string(Code, _, _, _, 'not WamRuntime.in_forkable_aggregate_frame?(state)'),
            sub_string(Code, _, _, _, 'Map.get(state, :parallel_depth, 0) > 0'),
            sub_string(Code, _, _, _, 'clause_ClauseStart_impl(state)'),
            sub_string(Code, _, _, _, 'cut_point: state.choice_points'),
            sub_string(Code, _, _, _, 'Task.async_stream'),
            sub_string(Code, _, _, _, 'try do'),
            sub_string(Code, _, _, _, '{:fail, _state} -> []'),
            sub_string(Code, _, _, _, 'WamRuntime.merge_into_aggregate(state, branch_results)')
        ->  pass(Test)
        ;   fail_test(Test, 'super-wrapper shape missing')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_references_all_branches :-
    Test = 'Tier-2 wrapper: branch list references every clause _impl function',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [], Code),
            sub_string(Code, _, _, _, '&clause_ClauseStart_impl/1'),
            sub_string(Code, _, _, _, '&clause_LB_impl/1'),
            sub_string(Code, _, _, _, '&clause_LC_impl/1')
        ->  pass(Test)
        ;   fail_test(Test, 'branch list does not reference all three clause _impl functions')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_rejects_two_clauses :-
    Test = 'Tier-2 wrapper: 2-clause predicate falls through (clause-count gate)',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure2/2)),
        (   two_segment_fixture(Segs),
            par_wrap_segment(tier2_pure2/2, Segs, [], Code),
            Code == ""
        ->  pass(Test)
        ;   fail_test(Test, 'clause-count gate did not reject 2-clause predicate')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure2/2))
    ).

test_par_wrap_segment_rejects_impure :-
    Test = 'Tier-2 wrapper: impure predicate falls through (no certificate)',
    three_segment_fixture(Segs),
    par_wrap_segment(no_such_pred/2, Segs, [], Code),
    (   Code == ""
    ->  pass(Test)
    ;   fail_test(Test, 'purity gate did not reject unknown predicate')
    ).

%% switch_arm_targets(+Instrs, -Targets)
%  Extract the non-default arm targets from the first segment's
%  switch_on_constant instruction. Used by the switch-arm coverage
%  test to confirm every switch target corresponds to a segment in
%  the Segments list — the invariant par_wrap_segment/4 relies on
%  to fan out all clause alternatives when first-arg indexing is
%  present.
%
%  Each arm is represented as a string "Key:Target" (not a compound
%  term), so splitting on ":" is required to extract the target.
switch_arm_targets(Instrs, Targets) :-
    member(_PC-switch_on_constant(Arms), Instrs),
    !,
    findall(TargetStr,
            (  member(Arm, Arms),
               split_string(Arm, ":", "", [_KeyStr, TargetStr]),
               TargetStr \= "default"
            ),
            Targets).
switch_arm_targets(_, []).

test_switch_arm_targets_are_segments :-
    Test = 'Tier-2 wrapper: switch_on_constant arm targets are all named segments (prereq 1)',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    Segments = [_-FirstInstrs | _],
    switch_arm_targets(FirstInstrs, ArmTargets),
    % Turn each segment name into an atom for comparison.
    findall(Name, member(Name-_, Segments), SegNames),
    % Every non-default arm target must appear as a segment name.
    (   ArmTargets \= [],
        forall(member(T, ArmTargets), memberchk(T, SegNames))
    ->  pass(Test)
    ;   format(atom(Reason),
               'arm targets ~w not all in segment names ~w',
               [ArmTargets, SegNames]),
        fail_test(Test, Reason)
    ).

test_par_wrap_segment_covers_switch_targets :-
    Test = 'Tier-2 wrapper: emitted branch list references every switch_on_constant target',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:small_fact/2)),
        (   phase_a_fixture_setup,
            wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
            atom_string(WamCode, WamStr),
            split_string(WamStr, "\n", "", Lines),
            wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
            Segments = [_-FirstInstrs | _],
            switch_arm_targets(FirstInstrs, ArmTargets),
            par_wrap_segment(small_fact/2, Segments, [], Code),
            Code \= "",
            % For each switch arm target, confirm the emitted super-wrapper
            % references &clause_<camelCase(target)>_impl/1.
            forall(member(Target, ArmTargets),
                   (  wam_elixir_lowered_emitter:segment_func_name(Target, BaseFunc),
                      format(string(ImplRef), '&~w_impl/1', [BaseFunc]),
                      sub_string(Code, _, _, _, ImplRef)
                   ))
        ->  pass(Test)
        ;   fail_test(Test, 'emitted branch list does not cover every switch_on_constant target')
        ),
        retract(clause_body_analysis:order_independent(user:small_fact/2))
    ).

test_par_wrap_segment_kill_switch :-
    Test = 'Tier-2 wrapper: intra_query_parallel(false) option forces fall-through',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [intra_query_parallel(false)], Code),
            Code == ""
        ->  pass(Test)
        ;   fail_test(Test, 'kill-switch did not force fall-through')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

maybe_abolish_test_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).

contains_string(Haystack, Needle) :-
    once(sub_string(Haystack, _, _, _, Needle)).

%% Wiring tests — exercise the full lower_predicate_to_elixir/4 entry
%% point (not par_wrap_segment/4 in isolation). small_fact/2 has 4
%% ground clauses, classified `compiled` under the default threshold,
%% so it routes through render_compiled_module/8 where the wiring lives.

test_wiring_emits_tier2_eligible_attr :-
    Test = 'Wiring: gate-pass emits @tier2_eligible true module attribute',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:small_fact/2)),
        (   phase_a_fixture_setup,
            wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
            lower_predicate_to_elixir(small_fact/2, WamCode,
                                      [module_name('TestMod')], Code),
            atom_string(Code, S),
            sub_string(S, _, _, _, '@tier2_eligible true')
        ->  pass(Test)
        ;   fail_test(Test, '@tier2_eligible attribute not emitted on gate-pass')
        ),
        retract(clause_body_analysis:order_independent(user:small_fact/2))
    ).

test_wiring_clause_main_is_super_wrapper_on_gate_pass :-
    Test = 'Wiring: gate-pass — surface entry is the super-wrapper, not the first clause body',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:small_fact/2)),
        (   phase_a_fixture_setup,
            wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
            lower_predicate_to_elixir(small_fact/2, WamCode,
                                      [module_name('TestMod')], Code),
            atom_string(Code, S),
            % Structural shape: a `cond do` super-wrapper signature is
            % present (uniquely identifies the Tier-2 wrapper), and at
            % least one `defp clause_X_impl(state) do` body is emitted
            % — confirming the rename happened. Name is not hardcoded
            % because the entry func name derives from the first
            % segment label, which varies per compiled predicate.
            sub_string(S, _, _, _, 'cond do'),
            sub_string(S, _, _, _, 'not WamRuntime.in_forkable_aggregate_frame?(state)'),
            sub_string(S, _, _, _, 'Map.get(state, :parallel_depth, 0) > 0'),
            sub_string(S, _, _, _, '_impl(state) do')
        ->  pass(Test)
        ;   fail_test(Test, 'super-wrapper signature or _impl body absent on gate-pass')
        ),
        retract(clause_body_analysis:order_independent(user:small_fact/2))
    ).

test_wiring_gate_reject_preserves_naming :-
    Test = 'Wiring: gate-reject leaves clause names unchanged (no _impl, no @tier2_eligible)',
    % No order_independent declaration → purity gate rejects.
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    lower_predicate_to_elixir(small_fact/2, WamCode,
                              [module_name('TestMod')], Code),
    atom_string(Code, S),
    (   \+ sub_string(S, _, _, _, '_impl'),
        \+ sub_string(S, _, _, _, '@tier2_eligible'),
        \+ sub_string(S, _, _, _, 'in_forkable_aggregate_frame?')
    ->  pass(Test)
    ;   fail_test(Test, 'gate-reject path leaked _impl, @tier2_eligible, or super-wrapper signature into output')
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-Elixir Target Tests ===~n~n'),
    test_step_generation,
    test_helpers_generation,
    test_runtime_assembly,
    test_instruction_count,
    test_head_unification_instructions,
    test_body_construction_instructions,
    test_unification_instructions,
    test_control_flow_instructions,
    test_choice_point_instructions,
    test_choice_point_bytecode,
    test_builtin_call_delegates,
    test_elixir_idioms,
    test_immutable_state_updates,
    test_functional_run_loop,
    test_classify_small_fact_only,
    test_classify_big_fact_only,
    test_classify_rule,
    test_classify_variable_head,
    test_classify_user_override_layout,
    test_classify_user_override_threshold,
    test_shape_comment_in_generated_module,
    test_phase_a_preserves_compiled_output,
    test_extract_facts_simple,
    test_phase_b_emits_inline_data_when_chosen,
    test_phase_b_variable_head_becomes_sentinel,
    test_phase_b_fallback_on_unextractable,
    test_quote_wam_constant_plain,
    test_quote_wam_constant_comma,
    test_quote_wam_constant_escape,
    test_tokenize_unquoted,
    test_tokenize_quoted_atom_with_comma,
    test_tokenize_quoted_atom_with_escape,
    test_round_trip_comma_atom,
    test_extract_arg1_index_ground,
    test_extract_arg1_index_variable,
    test_phase_c_indexed_module_emission,
    test_phase_c_index_policy_none,
    test_phase_c_variable_head_no_index,
    test_phase_d_emits_external_source_shape,
    test_phase_d_external_source_preserves_shared_preprocess_metadata,
    test_phase_d_runtime_emits_fact_source,
    test_phase_d_external_beats_inline_override,
    test_phase_e_auto_matches_pre_phase_e,
    test_phase_e_compiled_only_forces_compiled,
    test_phase_e_inline_eager_ignores_threshold,
    test_phase_e_inline_eager_respects_fact_only,
    test_phase_e_user_override_preempts_policy,
    test_cost_aware_promotes_big_fact_set,
    test_cost_aware_keeps_small_preds_compiled,
    test_cost_aware_threshold_override,
    test_cost_aware_respects_fact_only,
    test_ets_adaptor_emitted_in_runtime,
    test_sqlite_adaptor_emitted_in_runtime,
    test_sqlite_adaptor_uses_indirect_module_resolution,
    test_tier2_wamstate_has_parallel_depth,
    test_tier2_aggregate_helpers_emitted,
    test_tier2_aggregate_forkable_types,
    test_tier2_purity_gate_rejects_unknown,
    test_tier2_purity_gate_accepts_declared,
    test_par_wrap_segment_emits_super_wrapper,
    test_par_wrap_segment_references_all_branches,
    test_par_wrap_segment_rejects_two_clauses,
    test_par_wrap_segment_rejects_impure,
    test_switch_arm_targets_are_segments,
    test_par_wrap_segment_covers_switch_targets,
    test_par_wrap_segment_kill_switch,
    test_wiring_emits_tier2_eligible_attr,
    test_wiring_clause_main_is_super_wrapper_on_gate_pass,
    test_wiring_gate_reject_preserves_naming,
    format('~n=== WAM-Elixir Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

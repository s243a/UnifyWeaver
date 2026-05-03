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

%% LMDB adaptor (memory-mapped fact source — addresses the
%  materialisation-cost bottleneck documented in
%  docs/WAM_TARGET_ROADMAP.md). Mirrors the SQLite tests pattern
%  emit-and-grep + indirect-module-resolution check.
test_lmdb_adaptor_emitted_in_runtime :-
    Test = 'LMDB adaptor: runtime assembly emits FactSource.Lmdb',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.Lmdb do'),
        sub_string(RuntimeCode, _, _, _, '@behaviour WamRuntime.FactSource'),
        sub_string(RuntimeCode, _, _, _, 'defstruct [:env, :dbi, :arity, :dupsort]'),
        % Three FactSource callbacks present.
        sub_string(RuntimeCode, _, _, _, 'def stream_all('),
        sub_string(RuntimeCode, _, _, _, 'def lookup_by_arg1('),
        sub_string(RuntimeCode, _, _, _, 'def close(')
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB adaptor missing expected structure')
    ).

test_lmdb_adaptor_uses_indirect_module_resolution :-
    Test = 'LMDB adaptor: uses Module.concat so runtime compiles without an LMDB binding dep',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Same constraint as SQLite: the adaptor must NOT call Elmdb.X
    % as a literal — that warns in drivers without the LMDB binding.
    (   sub_string(RuntimeCode, _, _, _, 'Module.concat([Elmdb])'),
        sub_string(RuntimeCode, _, _, _, 'apply(mod,'),
        \+ sub_string(RuntimeCode, _, _, _, 'Elmdb.ro_txn_begin'),
        \+ sub_string(RuntimeCode, _, _, _, 'Elmdb.txn_get(')
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB adaptor has literal Elmdb references — will warn without dep')
    ).

test_lmdb_adaptor_targets_safe_keyvalue_api :-
    Test = 'LMDB adaptor: uses safe key/value + cursor API (no raw-pointer ops)',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Contract per WAM_TARGET_ROADMAP.md: avoid the raw-pointer
    % interface that crashed Haskell. txn_get + cursor get/next are
    % the safe layer; reject any pointer-deref shape.
    (   sub_string(RuntimeCode, _, _, _, 'txn_get'),
        sub_string(RuntimeCode, _, _, _, 'ro_txn_cursor_get'),
        % Dupsort path uses MDB_SET / MDB_NEXT_DUP cursor ops.
        sub_string(RuntimeCode, _, _, _, ':next_dup'),
        sub_string(RuntimeCode, _, _, _, ':set'),
        % No raw pointer / mdb_val references (those are the unsafe layer).
        \+ sub_string(RuntimeCode, _, _, _, 'mdb_val'),
        \+ sub_string(RuntimeCode, _, _, _, 'raw_pointer')
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB adaptor missing safe-API surface or includes raw-pointer ops')
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

%% Findall substrate (Phase 1) — exercises the runtime helpers added
%% per docs/proposals/WAM_ELIXIR_TIER2_FINDALL.md §4. End-to-end findall
%% execution waits on Phase 2 (begin_aggregate/end_aggregate instruction
%% lowering) and Phase 3 (integration tests). These are emit-and-grep
%% checks on the emitted Elixir source.

test_findall_substrate_emits_push_aggregate_frame :-
    Test = 'Findall substrate: WamRuntime emits push_aggregate_frame/4 with correct CP shape',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def push_aggregate_frame(state, agg_type, value_reg, result_reg)'),
        % CP must capture aggregate-specific fields.
        sub_string(S, _, _, _, 'agg_type: agg_type'),
        sub_string(S, _, _, _, 'agg_value_reg: value_reg'),
        sub_string(S, _, _, _, 'agg_result_reg: result_reg'),
        sub_string(S, _, _, _, 'agg_accum: []'),
        % Plus the standard CP snapshot fields finalise restores.
        % `cp:` here doubles as the post-finalise continuation — the
        % proposal §4.1 listed a separate :agg_return_cp field but it
        % always equalled state.cp at push time, so it was dropped
        % during implementation. The docstring on the emitted helper
        % explains the deviation; finalise_aggregate's tail-call uses
        % `restored.cp.(restored)` (asserted in the finalise test).
        sub_string(S, _, _, _, 'cp: state.cp'),
        sub_string(S, _, _, _, 'trail_len: state.trail_len'),
        sub_string(S, _, _, _, 'heap_len: state.heap_len')
    ->  pass(Test)
    ;   fail_test(Test, 'push_aggregate_frame absent or missing required CP fields')
    ).

test_findall_substrate_emits_aggregate_collect :-
    Test = 'Findall substrate: WamRuntime emits aggregate_collect/2 (deep-copy + prepend to nearest agg frame)',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def aggregate_collect(state, value_reg)'),
        % Must deep-copy the value register before capturing — atomic
        % values pass through unchanged, compound heap structures
        % become self-contained {:struct, ...} tuples that survive
        % backtrack's heap-rewind.
        sub_string(S, _, _, _, 'raw = Map.get(state.regs, value_reg)'),
        sub_string(S, _, _, _, 'val = deep_copy_value(state, raw)'),
        % Must prepend (O(1)) to the nearest aggregate frame's accum.
        sub_string(S, _, _, _, '[val | prior]'),
        % And the deep_copy_value helper itself exists.
        sub_string(S, _, _, _, 'def deep_copy_value(state, val)'),
        sub_string(S, _, _, _, '{:str, functor}')
    ->  pass(Test)
    ;   fail_test(Test, 'aggregate_collect absent or missing deep-copy/prepend logic')
    ).

test_findall_substrate_emits_finalise_aggregate :-
    Test = 'Findall substrate: WamRuntime emits finalise_aggregate/4 covering all aggregator types',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def finalise_aggregate(state, agg_cp, rest_cps, agg_type)'),
        % Reverses accumulator (we prepend in collect for O(1)).
        sub_string(S, _, _, _, 'Enum.reverse(agg_cp.agg_accum)'),
        % All aggregator atoms emitted by compile_aggregate_all/5 must
        % be handled — see wam_target.pl:712 for the alphabet.
        sub_string(S, _, _, _, ':collect, :findall, :aggregate_all, :bag'),
        sub_string(S, _, _, _, ':set -> Enum.uniq'),
        sub_string(S, _, _, _, ':sum -> Enum.sum'),
        sub_string(S, _, _, _, ':count -> length'),
        % :max / :min throw {:fail, state} on empty accumulator
        % (canonical Prolog semantics — no identity exists).
        sub_string(S, _, _, _, 'if accum_rev == [], do: throw({:fail, state}), else: Enum.max(accum_rev)'),
        sub_string(S, _, _, _, 'if accum_rev == [], do: throw({:fail, state}), else: Enum.min(accum_rev)'),
        % Must bind the result into agg_result_reg, restore from
        % snapshot, and tail-call the restored state.cp (no
        % separate :agg_return_cp — see push_aggregate_frame test).
        sub_string(S, _, _, _, 'Map.put(agg_cp.regs, agg_cp.agg_result_reg, result)'),
        sub_string(S, _, _, _, 'restored.cp.(restored)')
    ->  pass(Test)
    ;   fail_test(Test, 'finalise_aggregate absent or missing required aggregator handling')
    ).

%% Note: aggregate_collect/2 with no aggregate frame on the stack is
%% a documented safety contract (returns state unchanged). Coverage
%% deferred to Phase 3 runtime tests, where a fixture state with an
%% empty CP stack can exercise the contract end-to-end.

%% Findall instructions (Phase 2) — exercises the parser entries and
%% wam_elixir_lower_instr/6 clauses for begin_aggregate / end_aggregate
%% per docs/proposals/WAM_ELIXIR_TIER2_FINDALL.md §4.2 + §4.3.

test_findall_instr_parser_begin_aggregate :-
    Test = 'Findall instr: instr_from_parts parses `begin_aggregate sum, A1, A3`',
    (   wam_elixir_lowered_emitter:instr_from_parts(
            ["begin_aggregate", "sum", "A1", "A3"],
            begin_aggregate("sum", "A1", "A3"))
    ->  pass(Test)
    ;   fail_test(Test, 'parser did not match begin_aggregate 4-element form')
    ).

test_findall_instr_parser_end_aggregate :-
    Test = 'Findall instr: instr_from_parts parses `end_aggregate A1`',
    (   wam_elixir_lowered_emitter:instr_from_parts(
            ["end_aggregate", "A1"],
            end_aggregate("A1"))
    ->  pass(Test)
    ;   fail_test(Test, 'parser did not match end_aggregate 2-element form')
    ).

test_findall_instr_lowers_begin_aggregate_sum :-
    Test = 'Findall instr: begin_aggregate(sum, A1, A3) lowers to push_aggregate_frame call with :sum',
    (   wam_elixir_lowered_emitter:wam_elixir_lower_instr(
            begin_aggregate("sum", "A1", "A3"),
            1, [], "clause_main", "", Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'WamRuntime.push_aggregate_frame(state, :sum, 1, 3)')
    ->  pass(Test)
    ;   fail_test(Test, 'sum aggregator did not lower to push_aggregate_frame call')
    ).

test_findall_instr_translates_collect_to_findall :-
    Test = 'Findall instr: begin_aggregate(collect, ...) translates to :findall (proposal §6.4)',
    (   wam_elixir_lowered_emitter:wam_elixir_lower_instr(
            begin_aggregate("collect", "A1", "A3"),
            1, [], "clause_main", "", Code),
        atom_string(Code, S),
        % Translation is at the emission site so the substrate's
        % in_forkable_aggregate_frame?/1 (which only recognises
        % :findall and :aggregate_all) sees a forkable frame.
        sub_string(S, _, _, _, ':findall'),
        \+ sub_string(S, _, _, _, ':collect')
    ->  pass(Test)
    ;   fail_test(Test, 'collect aggregator was not translated to :findall at emission')
    ).

test_findall_instr_lowers_end_aggregate :-
    Test = 'Findall instr: end_aggregate(A1) lowers to aggregate_collect + throw({:fail, state})',
    (   wam_elixir_lowered_emitter:wam_elixir_lower_instr(
            end_aggregate("A1"),
            1, [], "clause_main", "", Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'WamRuntime.aggregate_collect(state, 1)'),
        % Throw drives backtrack into finalise_aggregate per the
        % fail-driven enumeration model (proposal §3.3 / §4.4).
        sub_string(S, _, _, _, 'throw({:fail, state})')
    ->  pass(Test)
    ;   fail_test(Test, 'end_aggregate did not emit collect + throw shape')
    ).

%% End-to-end bridge: a predicate whose body uses findall/3 should now
%% lower through the full pipeline (WAM compile → parse → lower) and
%% produce a module containing both runtime calls. This is the first
%% test that exercises the full Phase 2 chain. Phase 3 will add actual
%% Elixir-execution tests.

:- dynamic phase_a_test:has_findall/1.

phase_a_findall_fixture_setup :-
    phase_a_fixture_setup,
    retractall(phase_a_test:has_findall(_)),
    assertz((phase_a_test:has_findall(L) :-
        findall(X, phase_a_test:small_fact(X, _), L))).

test_findall_instr_end_to_end_lowering :-
    Test = 'Findall instr: predicate body using findall/3 lowers to push_aggregate_frame + aggregate_collect',
    phase_a_findall_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:has_findall/1, [], WamCode),
    lower_predicate_to_elixir(has_findall/1, WamCode,
                              [module_name('TestMod')], Code),
    atom_string(Code, S),
    (   % :findall (translated from WAM-emitted :collect) must appear
        % in the push call — confirms the collect→findall translation
        % survives the full WAM compile → parse → lower pipeline, not
        % just the unit-level lowering test above.
        sub_string(S, _, _, _, 'WamRuntime.push_aggregate_frame(state, :findall'),
        sub_string(S, _, _, _, 'WamRuntime.aggregate_collect(state'),
        sub_string(S, _, _, _, 'throw({:fail, state})')
    ->  pass(Test)
    ;   fail_test(Test, 'end-to-end findall lowering missing push/collect/throw shape')
    ).

%% Module-qualified findall regression test for the wam_target.pl fix
%% (Finding 1 from #1647). Before the fix, `findall(X, M:p(X), L)`
%% emitted `begin_aggregate collect, A1, X_n` — value_reg=A1 — which
%% got clobbered by the `:/2` builtin's put_constant of the module
%% name string. After the fix, compile_aggregate_all/5 unwraps
%% compile_findall's `collect-Template` wrapper to expose the
%% Template variable, so the var(ValueVar) branch fires and a Y-reg
%% is allocated. End_aggregate then reads from a slot that survives
%% any inner-call register churn.
%%
%% Target-agnostic at the WAM byte-shape level: this test asserts
%% the emitted WAM uses Y... not A1 for findall's value_reg, which
%% is the contract every target's lowering depends on.

:- dynamic phase_a_test:findall_qualified_target/1.

test_findall_module_qualified_unwrap :-
    Test = 'WAM compiler: findall/3 with static module-qualified inner goal unwraps to direct call (fix for #1647 Finding 1)',
    setup_call_cleanup(
        (   retractall(phase_a_test:findall_qualified_target(_)),
            assertz(phase_a_test:findall_qualified_target('a')),
            assertz(phase_a_test:findall_qualified_target('b')),
            assertz((phase_a_test:findall_qualified_caller(L) :-
                findall(X, phase_a_test:findall_qualified_target(X), L)))
        ),
        (   wam_target:compile_predicate_to_wam(
                phase_a_test:findall_qualified_caller/1, [], WamCode),
            atom_string(WamCode, S),
            % Y-reg-allocation portion of #1650 still applies — findall's
            % value_reg should be a Y-register regardless of inner-goal shape.
            sub_string(S, _, _, _, 'begin_aggregate collect, Y'),
            sub_string(S, _, _, _, 'end_aggregate Y'),
            \+ sub_string(S, _, _, _, 'begin_aggregate collect, A1'),
            % Static-module-qualifier unwrap (this PR): the inner goal
            % `phase_a_test:findall_qualified_target(X)` should compile
            % to a regular `call findall_qualified_target/1, 1` instead
            % of routing through the `:/2` builtin path. Cross-target
            % win — every target avoids the meta-call overhead on the
            % common static-qualifier case.
            sub_string(S, _, _, _, 'call findall_qualified_target/1'),
            \+ sub_string(S, _, _, _, 'builtin_call :/2'),
            % And the put_constant for the module name is gone too —
            % the unwrapped call doesn't construct a `:/2` structure
            % on the heap.
            \+ sub_string(S, _, _, _, 'put_constant phase_a_test')
        ->  pass(Test)
        ;   fail_test(Test, 'module-qualified findall did not unwrap to direct call (Option B regressed?)')
        ),
        (   retractall(phase_a_test:findall_qualified_target(_)),
            retractall(phase_a_test:findall_qualified_caller(_))
        )
    ).

test_findall_substrate_backtrack_routes_aggregate_frames :-
    Test = 'Findall substrate: backtrack/1 dispatches aggregate frames to finalise_aggregate',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        % The dispatcher arm: presence of :agg_type on the popped CP
        % routes to finalise instead of the ordinary unwind path.
        sub_string(S, _, _, _, 'case Map.get(cp, :agg_type) do'),
        sub_string(S, _, _, _, 'nil ->'),
        sub_string(S, _, _, _, 'backtrack_ordinary(state, cp, rest)'),
        sub_string(S, _, _, _, 'agg_type ->'),
        sub_string(S, _, _, _, 'finalise_aggregate(state, cp, rest, agg_type)'),
        % And the existing ordinary path must still exist as a defp.
        sub_string(S, _, _, _, 'defp backtrack_ordinary(state, cp, rest)')
    ->  pass(Test)
    ;   fail_test(Test, 'backtrack/1 does not dispatch on :agg_type or backtrack_ordinary missing')
    ).

%% Phase 4a substrate — branch_backtrack/1 helper for parallel-branch
%% context. See WAM_ELIXIR_TIER2_FINDALL_PHASE4.md sections 4.2/4.3.
%% Phase 4a only adds the helper; Phase 4b wires it into the super-
%% wrappers branch wrapper. Tests are emit-and-grep; runtime
%% verification (via the Phase 3-style harness) belongs to Phase 4c.

test_findall_phase4a_emits_branch_backtrack :-
    Test = 'Phase 4a: WamRuntime emits branch_backtrack/1 with required dispatch arms',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def branch_backtrack(state)'),
        % Empty CP stack arm — branch produced nothing.
        sub_string(S, _, _, _, '{:branch_exhausted, []}'),
        % Agg frame arm — return reversed local accum (aggregate_collect
        % prepends for O(1), branch_backtrack reverses on return so the
        % parent sees enumeration-order values).
        sub_string(S, _, _, _, '{:branch_exhausted, Enum.reverse'),
        sub_string(S, _, _, _, 'Map.get(cp, :agg_accum, [])'),
        % Fall-through arm — non-agg CP routes to backtrack_ordinary
        % (the existing path), letting the resumed clause continue
        % the branchs local enumeration.
        sub_string(S, _, _, _, 'backtrack_ordinary(state, cp, rest)')
    ->  pass(Test)
    ;   fail_test(Test, 'branch_backtrack/1 missing or has wrong dispatch shape')
    ).

test_findall_phase4a_branch_backtrack_distinct_from_backtrack :-
    Test = 'Phase 4a: branch_backtrack/1 is a distinct def from backtrack/1',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        % Both must exist as separate top-level defs — branch_backtrack
        % is additive, not a replacement for backtrack. Phase 4b will
        % decide how the super-wrapper routes between them (proposal §9
        % Q1 — return shape vs. throw shape, branch-mode state field
        % vs. catch-arm dispatch).
        sub_string(S, _, _, _, 'def backtrack(state) do'),
        sub_string(S, _, _, _, 'def branch_backtrack(state) do')
    ->  pass(Test)
    ;   fail_test(Test, 'branch_backtrack/1 should coexist with backtrack/1')
    ).

%% Phase 4b — super-wrapper rework. Branches run with branch_mode:
%% true; backtrack/1 dispatches on it for agg-frame CPs to return
%% {:branch_exhausted, accum} instead of finalising. Super-wrapper
%% merges branch results and throws fail to drive the parents
%% standard finalise.

test_findall_phase4b_backtrack_dispatches_on_branch_mode :-
    Test = 'Phase 4b/4d: backtrack/1 dispatches agg-frame CPs on branch_mode + branch_sentinel',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        % Phase 4b introduced branch_mode dispatch; Phase 4d added the
        % :branch_sentinel guard so only the branchs PARENT agg CP
        % returns :branch_exhausted — nested agg CPs (from inner
        % findalls inside a branch body) finalise normally.
        sub_string(S, _, _, _, 'Map.get(state, :branch_mode, false) and'),
        sub_string(S, _, _, _, 'Map.get(cp, :branch_sentinel, false) ->'),
        sub_string(S, _, _, _, '{:branch_exhausted, Enum.reverse(Map.get(cp, :agg_accum, []))}'),
        % And the existing finalise path is preserved as the else branch.
        sub_string(S, _, _, _, 'finalise_aggregate(state, cp, rest, agg_type)')
    ->  pass(Test)
    ;   fail_test(Test, 'backtrack/1 missing branch_mode + branch_sentinel dispatch')
    ).

test_findall_phase4b_wamstate_has_branch_mode_field :-
    Test = 'Phase 4b: WamState defstruct includes :branch_mode field',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        % The WamState struct must declare branch_mode so the super-
        % wrapper can set it on branch_state via %{state | branch_mode:
        % true}. Default false so all existing call sites that dont
        % explicitly set it remain in non-branch-mode (sequential).
        sub_string(S, _, _, _, 'branch_mode: false')
    ->  pass(Test)
    ;   fail_test(Test, ':branch_mode field missing from WamState defstruct')
    ).

%% Phase 4b.5 — _branch variants emitted alongside _impl for Tier-2-
%% eligible predicates. Closes the chain-CP duplicate-enumeration
%% finding from Phase 4b's runtime probe.

test_findall_phase4b5_emits_branch_variants :-
    Test = 'Phase 4b.5: Tier-2-eligible predicate emits both _impl and _branch clause variants',
    setup_call_cleanup(
        (   phase_a_fixture_setup,
            assertz(clause_body_analysis:order_independent(user:small_fact/2))
        ),
        (   wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
            lower_predicate_to_elixir(small_fact/2, WamCode,
                                      [module_name('TestMod')], Code),
            atom_string(Code, S),
            % Both `_impl` and `_branch` variants must be present.
            % `_impl` is the existing sequential chain (try_me_else
            % CP push); `_branch` is the no-push variant for parallel
            % super-wrapper dispatch. Suffix-based check rather than
            % hardcoded names because real WAM-compiled predicates
            % produce `clause_<PredCamel><Arity>` entry names, not
            % `clause_main` which only fires for hand-built fixtures
            % whose first segment label is literally "clause_start".
            sub_string(S, _, _, _, '_impl(state) do'),
            sub_string(S, _, _, _, '_branch(state) do')
        ->  pass(Test)
        ;   fail_test(Test, '_impl and/or _branch variants missing for Tier-2 predicate')
        ),
        retract(clause_body_analysis:order_independent(user:small_fact/2))
    ).

test_findall_phase4b5_branch_skips_cp_push :-
    Test = 'Phase 4b.5: _branch variant skips try_me_else CP push',
    setup_call_cleanup(
        (   phase_a_fixture_setup,
            assertz(clause_body_analysis:order_independent(user:small_fact/2))
        ),
        (   wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
            lower_predicate_to_elixir(small_fact/2, WamCode,
                                      [module_name('TestMod')], Code),
            atom_string(Code, S),
            % Find a `_branch(state) do` defp opener; verify the
            % immediate body section contains `try do` without a
            % preceding `cp = %{pc:` push or `choice_points: [cp |`
            % update. The `_branch` body should jump straight into
            % `try do` (no CP-push preamble between the defp opener
            % and the try).
            sub_string(S, BranchOff, _, _, '_branch(state) do'),
            string_length(S, TotalLen),
            ScanLen is min(200, TotalLen - BranchOff),
            sub_string(S, BranchOff, ScanLen, _, BranchHead),
            sub_string(BranchHead, _, _, _, 'try do'),
            \+ sub_string(BranchHead, _, _, _, 'cp = %{pc:'),
            \+ sub_string(BranchHead, _, _, _, 'choice_points: [cp |')
        ->  pass(Test)
        ;   fail_test(Test, '_branch variant unexpectedly contains a CP push')
        ),
        retract(clause_body_analysis:order_independent(user:small_fact/2))
    ).

:- dynamic integer_match_test:int_p/1.
:- dynamic arith_cmp_test:gt_p/1, arith_cmp_test:neq_p/1.
:- dynamic inline_list_test:len_p/1.

%% Arithmetic-comparison regression: the runtime must implement the
%  full comparison family (`<`, `>`, `>=`, `=<`, `=:=`, `=\=`) — pre-
%  fix only `</2` was implemented, so any predicate body using `>`
%  silently fell through to execute_builtin's default :fail arm.
%  Surfaced when iterate(N) :- N > 0, ... returned :fail for any N>0.
test_runtime_emits_full_comparison_family :-
    Test = 'Runtime: execute_builtin covers full arithmetic-comparison family',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, '{"</2", 2}'),
        sub_string(S, _, _, _, '{">/2", 2}'),
        sub_string(S, _, _, _, '{">=/2", 2}'),
        sub_string(S, _, _, _, '{"=</2", 2}'),
        sub_string(S, _, _, _, '{"=:=/2", 2}'),
        sub_string(S, _, _, _, '{"=\\\\=/2", 2}')
    ->  pass(Test)
    ;   fail_test(Test, 'one or more comparison ops missing from execute_builtin')
    ).

%% Audit follow-up (benchmarks/wam_elixir_builtin_coverage.md): the
%  runtime now also implements =/2, \\=/2, fail/0, append/3, write/1,
%  nl/0, format/1, format/2, and hardens the default arm to throw
%  {:unknown_builtin, op, arity} instead of silently :fail-ing.
test_runtime_emits_extended_builtin_set :-
    Test = 'Runtime: execute_builtin covers =/2, \\=/2, fail/0, append/3, write/nl/format',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, '{"=/2", 2}'),
        sub_string(S, _, _, _, '{"\\\\=/2", 2}'),
        sub_string(S, _, _, _, '{"fail/0", 0}'),
        sub_string(S, _, _, _, '{"append/3", 3}'),
        sub_string(S, _, _, _, '{"write/1", 1}'),
        sub_string(S, _, _, _, '{"nl/0", 0}'),
        sub_string(S, _, _, _, '{"format/1", 1}'),
        sub_string(S, _, _, _, '{"format/2", 2}')
    ->  pass(Test)
    ;   fail_test(Test, 'one or more extended builtins missing from execute_builtin')
    ).

%% Audit follow-up (benchmarks/wam_elixir_builtin_coverage.md
%  medium-priority): the runtime now also implements the term meta-
%  programming primitives functor/3, arg/3, =../2, copy_term/2.
test_runtime_emits_meta_builtin_set :-
    Test = 'Runtime: execute_builtin covers functor/3, arg/3, =../2, copy_term/2',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, '{"functor/3", 3}'),
        sub_string(S, _, _, _, '{"arg/3", 3}'),
        sub_string(S, _, _, _, '{"=../2", 2}'),
        sub_string(S, _, _, _, '{"copy_term/2", 2}')
    ->  pass(Test)
    ;   fail_test(Test, 'one or more meta-builtins missing from execute_builtin')
    ).

%% Build-mode follow-up to PR #1782: functor/3 and =../2 now also
%  support build mode (Term unbound, Name+Arity OR list bound),
%  and copy_term/2 walks heap structures with fresh-var renaming
%  (sharing preserved). Emit-and-grep on the runtime confirms the
%  new helper defps are present.
test_runtime_emits_meta_builtin_build_mode :-
    Test = 'Runtime: meta-builtins emit build_functor_term/build_compound_term/deep_copy_with_fresh_vars helpers',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, 'defp build_functor_term('),
        sub_string(S, _, _, _, 'defp build_compound_term('),
        sub_string(S, _, _, _, 'defp deep_copy_with_fresh_vars(')
    ->  pass(Test)
    ;   fail_test(Test, 'one or more meta-builtin helpers missing from runtime')
    ).

test_runtime_default_arm_throws_unknown_builtin :-
    Test = 'Runtime: execute_builtin default arm throws {:unknown_builtin, ...} (hardening)',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, 'throw({:unknown_builtin, op, arity})')
    ->  pass(Test)
    ;   fail_test(Test, 'default arm not hardened — throw not emitted')
    ).

%% Inline-list-build regression: put_list / put_structure must bind
%  the target reg's previous unbound heap_ref to the new structure
%  address, so cons-cell tails link to the next cons cell. Without
%  this link, tail-of-first-cons stays as a self-pointing unbound
%  and list-walking builtins terminate at the broken link.
%  Surfaced by the append/3 work in PR #1780 (length/2 had the
%  same latent bug — verified at the time but out of scope there).
test_lowered_put_structure_links_unbound_heap_ref :-
    Test = 'Lowering: put_structure binds previous unbound heap_ref (inline-list link)',
    setup_call_cleanup(
        (   retractall(inline_list_test:len_p(_)),
            assertz((inline_list_test:len_p(N) :- length([1,2,3,4], N)))
        ),
        (   wam_target:compile_predicate_to_wam(inline_list_test:len_p/1, [], WamCode),
            lower_predicate_to_elixir(len_p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            % The fix emits a "link_addr" branch in put_structure /
            % put_list that detects {:unbound, {:heap_ref, ...}} and
            % writes the new ref into that heap cell.
            sub_string(S, _, _, _, 'link_addr'),
            sub_string(S, _, _, _, 'trail_binding({:heap_ref, link_addr})')
        ->  pass(Test)
        ;   fail_test(Test, 'put_structure / put_list does not link unbound heap_ref')
        ),
        retractall(inline_list_test:len_p(_))
    ).

%% Companion runtime check: wam_list_length / wam_list_member? must
%  walk both `./2` (early put_list) and `[|]/2` (later put_structure)
%  cons-cell functor tags. Inline lists mix both; pre-fix only `./2`
%  was accepted, so even with the heap-link fix, length still
%  terminated when it hit the [|]/2 second-cons.
test_runtime_list_walkers_accept_both_cons_tags :-
    Test = 'Runtime: wam_list_length / wam_list_member? walk both ./2 and [|]/2 cons tags',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, 'cons in ["./2", "[|]/2"]')
    ->  pass(Test)
    ;   fail_test(Test, 'list walkers do not accept both cons tags')
    ).

%% Backslash-escape regression: `=\=/2` op name must be emitted as
%  `"=\\=/2"` in the lowered call site so Elixir parses it as the
%  runtime string `=\=/2` (5 chars). Without escaping, Elixir 1.14+
%  drops the unrecognised `\=` escape and the call becomes the
%  runtime string `==/2` (4 chars), which never matches the
%  execute_builtin pattern.
test_lowered_escapes_backslash_in_builtin_call :-
    Test = 'Lowering: builtin_call escapes backslash so =\\= reaches the runtime intact',
    setup_call_cleanup(
        (   retractall(arith_cmp_test:neq_p(_)),
            assertz((arith_cmp_test:neq_p(X) :- X =\= 0))
        ),
        (   wam_target:compile_predicate_to_wam(arith_cmp_test:neq_p/1, [], WamCode),
            lower_predicate_to_elixir(neq_p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            % Positive: properly escaped form.
            sub_string(S, _, _, _, 'execute_builtin(state, "=\\\\=/2"'),
            % Negative: the buggy unescaped form would parse as ==/2
            % at runtime; ensure it isn't emitted.
            \+ sub_string(S, _, _, _, 'execute_builtin(state, "=\\=/2"')
        ->  pass(Test)
        ;   fail_test(Test, 'backslash not escaped — =\\= would be misparsed by Elixir')
        ),
        retractall(arith_cmp_test:neq_p(_))
    ).

%% Integer-quoting regression: the lowered emitter must emit numeric
%  constants as bare Elixir integer literals, not as quoted strings.
%  Pre-fix, `get_constant 0, A1` lowered to `val == "0"` (string),
%  silently breaking numeric head-match for any predicate matching
%  on integer literals (e.g. `iterate(0).` or fact base `r(1).`).
%  See benchmarks/wam_elixir_tier2_findall.md "Side finding" for the
%  original surface — and the broader implication that arithmetic-
%  heavy Prolog code was silently no-op on this target.
test_lowered_emits_integer_literals :-
    Test = 'Lowering: integer constants emit as bare Elixir literals (not stringified)',
    setup_call_cleanup(
        (   retractall(integer_match_test:int_p(_)),
            assertz(integer_match_test:int_p(1)),
            assertz(integer_match_test:int_p(2)),
            assertz(integer_match_test:int_p(3))
        ),
        (   wam_target:compile_predicate_to_wam(integer_match_test:int_p/1, [], WamCode),
            lower_predicate_to_elixir(int_p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            % Positive: bare integer literal in head-match comparison.
            sub_string(S, _, _, _, 'val == 1'),
            sub_string(S, _, _, _, 'val == 2'),
            sub_string(S, _, _, _, 'val == 3'),
            % Negative: stringified form must NOT appear.
            \+ sub_string(S, _, _, _, 'val == "1"'),
            \+ sub_string(S, _, _, _, 'val == "2"'),
            \+ sub_string(S, _, _, _, 'val == "3"')
        ->  pass(Test)
        ;   fail_test(Test, 'integer constants stringified — head-match would silently fail')
        ),
        retractall(integer_match_test:int_p(_))
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
            % Phase 4b: branches run with branch_mode: true; backtrack
            % returns {:branch_exhausted, accum} which the task wrapper
            % case-of unpacks. The fail-throw catch arm is now `{:fail, _s}`.
            sub_string(Code, _, _, _, 'branch_mode: true'),
            sub_string(Code, _, _, _, '{:branch_exhausted, accum} when is_list(accum) -> accum'),
            sub_string(Code, _, _, _, '{:fail, _s} -> []'),
            % Phase 4d: branchs PARENT agg CP is stamped with
            % :branch_sentinel before fan-out so backtrack/1 only
            % returns :branch_exhausted on THAT CP — nested agg CPs
            % from inner findalls inside the branch body finalise
            % normally. Without the stamp, an inner findalls accum
            % would leak up as the branchs return value.
            sub_string(Code, _, _, _, '[parent_agg_cp | rest_cps] = state.choice_points'),
            sub_string(Code, _, _, _, 'stamped_parent = Map.put(parent_agg_cp, :branch_sentinel, true)'),
            sub_string(Code, _, _, _, 'choice_points: [stamped_parent | rest_cps]'),
            % Parent merges branch contributions, then throws fail to
            % drive the parents standard finalise flow on the merged
            % accum (parent is NOT in branch_mode so finalise fires).
            sub_string(Code, _, _, _, 'merged = WamRuntime.merge_into_aggregate(state, branch_results)'),
            sub_string(Code, _, _, _, 'throw({:fail, merged})')
        ->  pass(Test)
        ;   fail_test(Test, 'super-wrapper shape missing')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_references_all_branches :-
    Test = 'Tier-2 wrapper: branch list references every clause _branch function (Phase 4b.5)',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [], Code),
            % Phase 4b.5: super-wrappers BranchFuncs reference `_branch`
            % variants (no next-clause CP push) instead of `_impl` —
            % closes the chain-CP duplicate-enumeration finding from
            % Phase 4bs runtime probe.
            sub_string(Code, _, _, _, '&clause_ClauseStart_branch/1'),
            sub_string(Code, _, _, _, '&clause_LB_branch/1'),
            sub_string(Code, _, _, _, '&clause_LC_branch/1')
        ->  pass(Test)
        ;   fail_test(Test, 'branch list does not reference all three clause _branch functions')
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
            % Phase 4b.5: super-wrapper references `_branch` variants
            % (skip next-clause CP push) instead of `_impl`. Closes
            % chain-CP duplicate-enumeration finding from #1706.
            forall(member(Target, ArmTargets),
                   (  wam_elixir_lowered_emitter:segment_func_name(Target, BaseFunc),
                      format(string(BranchRef), '&~w_branch/1', [BaseFunc]),
                      sub_string(Code, _, _, _, BranchRef)
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

%% Cost-aware gate (audit-driven follow-up to PRs #1774 / #1778).
%  par_wrap_segment/4 now consults a `forkMinCost(N)` Option: if the
%  predicates worst-case clause body scores below N on the static
%  instruction-weighted cost model, the super-wrapper is suppressed
%  (Code = "") and the clauses run sequentially. Default MinCost=0
%  preserves prior behaviour.

%% A high-cost fixture: each clause body has begin_aggregate (10) +
%  call (5) + put_constant (1) + end_aggregate (5) + proceed (1) +
%  control (try_me_else / retry_me_else / trust_me, 1 each) → ~22-23
%  per clause. Above any reasonable forkMinCost threshold.
heavy_segment_fixture([
    'clause_start'-[1-try_me_else(l_b), 2-put_constant("x", "A1"),
                    3-begin_aggregate(":collect", "X2", "X1"),
                    4-call("inner/1", 1), 5-end_aggregate("X2"), 6-proceed],
    'l_b'-[7-retry_me_else(l_c), 8-put_constant("y", "A1"),
           9-begin_aggregate(":collect", "X2", "X1"),
           10-call("inner/1", 1), 11-end_aggregate("X2"), 12-proceed],
    'l_c'-[13-trust_me, 14-put_constant("z", "A1"),
           15-begin_aggregate(":collect", "X2", "X1"),
           16-call("inner/1", 1), 17-end_aggregate("X2"), 18-proceed]
]).

test_par_wrap_segment_cost_gate_rejects_low_cost :-
    Test = 'Tier-2 cost gate: forkMinCost above clause cost forces fall-through',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),  % cost ~3-4 per clause
            par_wrap_segment(tier2_pure3/2, Segs, [forkMinCost(20)], Code),
            Code == ""
        ->  pass(Test)
        ;   fail_test(Test, 'cost gate did not suppress wrapper for low-cost predicate')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_cost_gate_passes_high_cost :-
    Test = 'Tier-2 cost gate: forkMinCost below clause cost still emits wrapper',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   heavy_segment_fixture(Segs),  % cost ~22-23 per clause
            par_wrap_segment(tier2_pure3/2, Segs, [forkMinCost(20)], Code),
            Code \== "",
            % Sanity: the emitted wrapper still has the cond/Task.async_stream shape.
            sub_string(Code, _, _, _, "Task.async_stream")
        ->  pass(Test)
        ;   fail_test(Test, 'cost gate wrongly suppressed wrapper for high-cost predicate')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_cost_gate_default_zero :-
    Test = 'Tier-2 cost gate: default MinCost=0 preserves prior behaviour (low-cost still wraps)',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [], Code),
            Code \== ""
        ->  pass(Test)
        ;   fail_test(Test, 'default MinCost=0 incorrectly suppressed wrapper')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

%% Runtime cost probe (companion to forkMinCost static gate).
%  par_wrap_segment/4 reads `runtime_cost_probe(ThresholdUs)` from
%  Options. When present, the super-wrappers `true ->` arm is wrapped
%  in a case statement that dispatches via WamRuntime.tier2_probe_*
%  helpers — first call goes sequential and measures, subsequent
%  calls follow the recorded decision.
test_par_wrap_segment_runtime_probe_emits_dispatch :-
    Test = 'Tier-2 runtime probe: super-wrapper emits ETS-backed dispatch when option set',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs,
                             [runtime_cost_probe(1000)], Code),
            sub_string(Code, _, _, _, "tier2_probe_decision"),
            sub_string(Code, _, _, _, "tier2_probe_update"),
            sub_string(Code, _, _, _, ":timer.tc"),
            sub_string(Code, _, _, _, ":probe ->"),
            sub_string(Code, _, _, _, ":go_sequential ->"),
            sub_string(Code, _, _, _, ":go_parallel ->")
        ->  pass(Test)
        ;   fail_test(Test, 'probe-dispatch arm not emitted when runtime_cost_probe set')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_par_wrap_segment_runtime_probe_default_unwired :-
    Test = 'Tier-2 runtime probe: default (no option) keeps the unconditional-parallel template',
    setup_call_cleanup(
        assertz(clause_body_analysis:order_independent(user:tier2_pure3/2)),
        (   three_segment_fixture(Segs),
            par_wrap_segment(tier2_pure3/2, Segs, [], Code),
            % Probe markers must NOT appear when option absent.
            \+ sub_string(Code, _, _, _, "tier2_probe_decision"),
            \+ sub_string(Code, _, _, _, ":probe ->")
        ->  pass(Test)
        ;   fail_test(Test, 'probe-dispatch arm leaked into default code path')
        ),
        retract(clause_body_analysis:order_independent(user:tier2_pure3/2))
    ).

test_runtime_emits_tier2_probe_helpers :-
    Test = 'Runtime: tier2_probe_decision/1 + tier2_probe_update/3 + ensure_tier2_probe_table available',
    wam_elixir_target:compile_wam_runtime_to_elixir([], Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, 'def tier2_probe_decision('),
        sub_string(S, _, _, _, 'def tier2_probe_update('),
        sub_string(S, _, _, _, 'defp ensure_tier2_probe_table'),
        sub_string(S, _, _, _, ':tier2_cost_probe')
    ->  pass(Test)
    ;   fail_test(Test, 'one or more tier2 probe helpers missing from runtime')
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
    test_lmdb_adaptor_emitted_in_runtime,
    test_lmdb_adaptor_uses_indirect_module_resolution,
    test_lmdb_adaptor_targets_safe_keyvalue_api,
    test_tier2_wamstate_has_parallel_depth,
    test_tier2_aggregate_helpers_emitted,
    test_tier2_aggregate_forkable_types,
    test_findall_substrate_emits_push_aggregate_frame,
    test_findall_substrate_emits_aggregate_collect,
    test_findall_substrate_emits_finalise_aggregate,
    test_findall_substrate_backtrack_routes_aggregate_frames,
    test_findall_phase4a_emits_branch_backtrack,
    test_findall_phase4a_branch_backtrack_distinct_from_backtrack,
    test_findall_phase4b_backtrack_dispatches_on_branch_mode,
    test_findall_phase4b_wamstate_has_branch_mode_field,
    test_findall_phase4b5_emits_branch_variants,
    test_findall_phase4b5_branch_skips_cp_push,
    test_findall_instr_parser_begin_aggregate,
    test_findall_instr_parser_end_aggregate,
    test_findall_instr_lowers_begin_aggregate_sum,
    test_findall_instr_translates_collect_to_findall,
    test_findall_instr_lowers_end_aggregate,
    test_findall_instr_end_to_end_lowering,
    test_findall_module_qualified_unwrap,
    test_lowered_emits_integer_literals,
    test_runtime_emits_full_comparison_family,
    test_runtime_emits_extended_builtin_set,
    test_runtime_emits_meta_builtin_set,
    test_runtime_emits_meta_builtin_build_mode,
    test_runtime_default_arm_throws_unknown_builtin,
    test_lowered_put_structure_links_unbound_heap_ref,
    test_runtime_list_walkers_accept_both_cons_tags,
    test_lowered_escapes_backslash_in_builtin_call,
    test_tier2_purity_gate_rejects_unknown,
    test_tier2_purity_gate_accepts_declared,
    test_par_wrap_segment_emits_super_wrapper,
    test_par_wrap_segment_references_all_branches,
    test_par_wrap_segment_rejects_two_clauses,
    test_par_wrap_segment_rejects_impure,
    test_switch_arm_targets_are_segments,
    test_par_wrap_segment_covers_switch_targets,
    test_par_wrap_segment_kill_switch,
    test_par_wrap_segment_cost_gate_rejects_low_cost,
    test_par_wrap_segment_cost_gate_passes_high_cost,
    test_par_wrap_segment_cost_gate_default_zero,
    test_par_wrap_segment_runtime_probe_emits_dispatch,
    test_par_wrap_segment_runtime_probe_default_unwired,
    test_runtime_emits_tier2_probe_helpers,
    test_wiring_emits_tier2_eligible_attr,
    test_wiring_clause_main_is_super_wrapper_on_gate_pass,
    test_wiring_gate_reject_preserves_naming,
    format('~n=== WAM-Elixir Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

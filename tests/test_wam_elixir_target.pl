:- encoding(utf8).
% Test suite for WAM-to-Elixir transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_target.pl

:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_lowered_emitter',
              [lower_predicate_to_elixir/4, classify_predicate/4,
               extract_facts/3, extract_arg1_index/3,
               tier2_purity_eligible/3, par_wrap_segment/4]).
:- use_module('../src/unifyweaver/core/recursive_kernel_detection',
              [detect_recursive_kernel/4]).
% For Tier-2 purity-gate tests — user-annotation producer reads
% clause_body_analysis:order_independent/1 dynamic facts.
:- use_module('../src/unifyweaver/core/clause_body_analysis').
% For test_lmdb_int_ids_mock_e2e: subprocess-invokes elixir against
% the runtime + the mock-Elmdb test fixture in tests/elixir_e2e/.
% Skips gracefully if elixir is not installed.
:- use_module(library(process)).
:- use_module(library(filesex)).

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

%% call/N meta-call generator gates (PR #1)

test_call_n_dispatch_meta_helper :-
    Test = 'WAM-Elixir: dispatch_call_meta helper present in runtime',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _,
                   'defp dispatch_call_meta(state, total_arity, after_pc)')
    ->  pass(Test)
    ;   fail_test(Test, 'dispatch_call_meta missing from runtime helpers')
    ).

test_call_n_step_arms :-
    Test = 'WAM-Elixir: :call and :execute step arms catch "call/N" \
                       before label lookup',
    (   wam_elixir_case(call, CallCode),
        sub_string(CallCode, _, _, _, '"call/" <> _'),
        sub_string(CallCode, _, _, _,
                   'dispatch_call_meta(state, total_arity, state.pc + 1)'),
        wam_elixir_case(execute, ExecCode),
        sub_string(ExecCode, _, _, _, '"call/" <> arity_str'),
        sub_string(ExecCode, _, _, _, 'Integer.parse(arity_str)'),
        sub_string(ExecCode, _, _, _,
                   'dispatch_call_meta(state, total_arity, state.cp)')
    ->  pass(Test)
    ;   fail_test(Test, 'call/N dispatch missing from step arms')
    ).

test_build_call_target_compound_clause :-
    Test = 'WAM-Elixir: build_call_target has {:ref, addr} compound clause',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _,
                   'defp build_call_target(state, {:ref, addr}, extras)'),
        % Heap deref + functor parse should be inline in the clause body.
        sub_string(S, _, _, _, 'Map.get(state.heap, addr)'),
        sub_string(S, _, _, _, 'parse_functor_arity(base_pred_arity)')
    ->  pass(Test)
    ;   fail_test(Test, '{:ref, addr} compound dispatch clause missing')
    ).

test_true_zero_builtin :-
    Test = 'WAM-Elixir: true/0 builtin arm exists',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '{"true/0", 0}')
    ->  pass(Test)
    ;   fail_test(Test, 'true/0 builtin missing')
    ).

test_build_call_target_helpers :-
    Test = 'WAM-Elixir: build_call_target + load_args_into_regs helpers \
                       present',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'defp build_call_target('),
        sub_string(S, _, _, _, 'defp load_args_into_regs(')
    ->  pass(Test)
    ;   fail_test(Test,
                  'build_call_target / load_args_into_regs missing')
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

test_lmdb_int_ids_adaptor_emitted_in_runtime :-
    % LmdbIntIds is the int-id-keyed sibling of FactSource.Lmdb,
    % designed for production-scale workloads where atom-interning
    % isn't viable (>50k unique nodes). Same emit-and-grep posture
    % as the original Lmdb adaptor — runtime validation requires
    % :elmdb which the sandbox can't install. See the design proposal
    % at docs/proposals/wam_elixir_lmdb_int_id_factsource.md.
    Test = 'LmdbIntIds adaptor: runtime emits FactSource.LmdbIntIds with three-sub-DB shape',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.LmdbIntIds do'),
        sub_string(RuntimeCode, _, _, _, '@behaviour WamRuntime.FactSource'),
        % Three sub-DB handles in the struct: facts, key→id, id→key.
        sub_string(RuntimeCode, _, _, _, ':facts_dbi'),
        sub_string(RuntimeCode, _, _, _, ':key_to_id_dbi'),
        sub_string(RuntimeCode, _, _, _, ':id_to_key_dbi'),
        % Bounded overwrite cache fields for Haskell-style LMDB reuse.
        sub_string(RuntimeCode, _, _, _, ':cache_table'),
        sub_string(RuntimeCode, _, _, _, ':cache_capacity'),
        % Fast-path int-id API (the whole point of this adaptor).
        sub_string(RuntimeCode, _, _, _, 'def lookup_by_arg1_id('),
        sub_string(RuntimeCode, _, _, _, 'defp cache_get('),
        sub_string(RuntimeCode, _, _, _, 'def preload_arg1_cache('),
        sub_string(RuntimeCode, _, _, _, 'Enum.chunk_by(fn {key_id, _value_id} -> key_id end)'),
        sub_string(RuntimeCode, _, _, _, ':erlang.phash2(key_id, cap)'),
        % Boundary translators for input/output id↔string conversion.
        sub_string(RuntimeCode, _, _, _, 'def lookup_id('),
        sub_string(RuntimeCode, _, _, _, 'def lookup_key('),
        % Backwards-compat binary-key entry (delegates to int-id path).
        sub_string(RuntimeCode, _, _, _, 'def lookup_by_arg1(%__MODULE__'),
        % 8-byte BE u64 id encoding (the on-disk wire format).
        sub_string(RuntimeCode, _, _, _, '<<id::64-big-unsigned>>')
    ->  pass(Test)
    ;   fail_test(Test, 'LmdbIntIds adaptor missing expected structure')
    ).

test_lmdb_int_ids_adaptor_uses_indirect_module_resolution :-
    Test = 'LmdbIntIds adaptor: uses Module.concat so runtime compiles without an LMDB binding dep',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Verify the LmdbIntIds-specific block (not the original Lmdb)
    % uses the same indirect-resolution pattern. Pull out the
    % LmdbIntIds module body and check it doesn't contain literal
    % Elmdb.X calls.
    Pattern = "defmodule WamRuntime.FactSource.LmdbIntIds do",
    EndPattern = "defmodule WamRuntime.GraphKernel.TransitiveClosure do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        sub_string(Body, _, _, _, "Module.concat([Elmdb])"),
        sub_string(Body, _, _, _, "apply(mod,"),
        \+ sub_string(Body, _, _, _, "Elmdb.ro_txn_begin"),
        \+ sub_string(Body, _, _, _, "Elmdb.txn_get(")
    ->  pass(Test)
    ;   fail_test(Test, 'LmdbIntIds adaptor has literal Elmdb references — will warn without dep')
    ).

test_lmdb_int_ids_design_proposal_referenced :-
    % Doc-fix invariant: the kernel docstring no longer claims
    % "Haskell uses LMDB IDs as interning". Instead it points at
    % the proposal that describes the actual LMDB-native design.
    Test = 'LmdbIntIds: kernel docstring points at the design proposal',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'wam_elixir_lmdb_int_id_factsource.md'),
        % Negative invariant: the old (false) claim is gone.
        \+ sub_string(RuntimeCode, _, _, _, 'no separate intern step')
    ->  pass(Test)
    ;   fail_test(Test, 'kernel docstring still has the old LMDB-IDs claim or doesnt reference the proposal')
    ).

test_lmdb_int_ids_ingest_pairs_emitted :-
    % Driver-side ingestion helper. Without this, the proposal
    % documents a contract no driver can fulfill (insert-time
    % ID assignment requires a coordinated write to all three
    % sub-DBs; a one-off helper is the natural shape).
    Test = 'LmdbIntIds: ingest_pairs/3 driver helper emitted with three-DB write surface',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'def ingest_pairs(%__MODULE__{}'),
        % Idempotent string-interning: txn_get on key_to_id_dbi
        % before allocating a new ID.
        sub_string(RuntimeCode, _, _, _, 'defp intern_one('),
        % Sequential ID allocation: increment next_id when
        % :not_found in key_to_id_dbi.
        sub_string(RuntimeCode, _, _, _, ':not_found ->'),
        % Writes all three sub-DBs: key_to_id_dbi, id_to_key_dbi,
        % facts_dbi (each via txn_put through Module.concat).
        sub_string(RuntimeCode, _, _, _, 'handle.key_to_id_dbi'),
        sub_string(RuntimeCode, _, _, _, 'handle.id_to_key_dbi'),
        sub_string(RuntimeCode, _, _, _, 'handle.facts_dbi'),
        % Returns a status map with the next_id sentinel for
        % batched calls.
        sub_string(RuntimeCode, _, _, _, ':next_id')
    ->  pass(Test)
    ;   fail_test(Test, 'ingest_pairs/3 missing or has wrong API surface')
    ).

test_lmdb_int_ids_mock_e2e :-
    % End-to-end exercise of WamRuntime.FactSource.LmdbIntIds against
    % a fake `Elmdb` module backed by an in-memory Agent. Validates:
    % - encode_id/decode_id round-trips on integer IDs.
    % - ingest_pairs idempotent re-ingest reuses existing IDs.
    % - lookup_by_arg1_id with dupsort returns all values for a key.
    % - lookup_by_arg1 (binary entry) round-trips through ID translation.
    % - lookup_id/lookup_key on missing keys return nil cleanly.
    % - stream_all returns ordered (int_key_id, int_value_id) pairs.
    % - migrate_from_string_keyed transfers all source pairs.
    %
    % Skips if `elixir` is not on PATH. Fails if elixir runs but a test
    % fails or stderr contains a syntax/compile error.
    Test = 'LmdbIntIds: end-to-end against MockElmdb (encode/ingest/lookup/migrate)',
    (   catch(process_create(path(elixir), ['-v'],
                              [stdout(null), stderr(null), process(P)]),
              _, fail),
        process_wait(P, _)
    ->  run_lmdb_int_ids_mock_e2e(Test)
    ;   format('[SKIP] ~w: elixir not installed~n', [Test])
    ).

run_lmdb_int_ids_mock_e2e(Test) :-
    % Set up a fresh tempdir, emit the runtime, copy the mock + test.
    TmpDir = '/tmp/test_lmdb_int_ids_mock_e2e',
    (   exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true ),
    make_directory(TmpDir),
    directory_file_path(TmpDir, 'lib', LibDir),
    make_directory(LibDir),
    compile_wam_runtime_to_elixir([], RuntimeCode),
    directory_file_path(LibDir, 'wam_runtime.ex', RuntimePath),
    open(RuntimePath, write, RS),
    write(RS, RuntimeCode),
    close(RS),
    % Copy the mock + test fixture into the project.
    source_file(test_lmdb_int_ids_mock_e2e, SrcFile),
    file_directory_name(SrcFile, TestsDir),
    directory_file_path(TestsDir, 'elixir_e2e/mock_elmdb.exs', MockSrc),
    directory_file_path(TestsDir, 'elixir_e2e/lmdb_int_ids_mock_test.exs', TestSrc),
    directory_file_path(TmpDir, 'mock_elmdb.exs', MockDst),
    directory_file_path(TmpDir, 'lmdb_int_ids_mock_test.exs', TestDst),
    copy_file(MockSrc, MockDst),
    copy_file(TestSrc, TestDst),
    % Run elixir.
    process_create(path(elixir),
                   ['-r', 'mock_elmdb.exs', 'lmdb_int_ids_mock_test.exs'],
                   [cwd(TmpDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, StdOut),
    read_string(Err, _, StdErr),
    process_wait(Pid, Status),
    close(Out),
    close(Err),
    % Parse PASS/FAIL markers.
    split_string(StdOut, "\n", "", Lines),
    findall(L, (member(L, Lines), string_concat("[PASS] ", _, L)), PassLines),
    findall(L, (member(L, Lines), string_concat("[FAIL] ", _, L)), FailLines),
    length(PassLines, NPass),
    length(FailLines, NFail),
    % Expected: 7 PASS, 0 FAIL.
    (   Status = exit(0), NPass >= 7, NFail = 0
    ->  format('[PASS] ~w (~w sub-tests via elixir subprocess)~n', [Test, NPass])
    ;   format('[FAIL] ~w: status=~w pass=~w fail=~w~n', [Test, Status, NPass, NFail]),
        (   FailLines = [_|_]
        ->  forall(member(F, FailLines), format('  ~s~n', [F]))
        ;   true
        ),
        (   StdErr \= ""
        ->  format('  stderr:~n~s~n', [StdErr])
        ;   true
        ),
        assertz(test_failed)
    ).

test_lmdb_int_ids_real_lmdb_e2e :-
    % End-to-end exercise against an actual LMDB env through the Hex
    % :elmdb package. The Elixir script supplies the real-driver bridge
    % from the runtime's dependency-free `Elmdb` shape to the Erlang
    % `:elmdb` module. It skips cleanly if the local NIF toolchain cannot
    % compile :elmdb.
    Test = 'LmdbIntIds: real LMDB e2e through :elmdb bridge',
    (   catch(process_create(path(elixir), ['-v'],
                              [stdout(null), stderr(null), process(P)]),
              _, fail),
        process_wait(P, _)
    ->  run_lmdb_int_ids_real_lmdb_e2e(Test)
    ;   format('[SKIP] ~w: elixir not installed~n', [Test])
    ).

run_lmdb_int_ids_real_lmdb_e2e(Test) :-
    TmpDir = '/tmp/test_lmdb_int_ids_real_lmdb_e2e',
    (   exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true ),
    make_directory(TmpDir),
    directory_file_path(TmpDir, 'lib', LibDir),
    make_directory(LibDir),
    compile_wam_runtime_to_elixir([], RuntimeCode),
    directory_file_path(LibDir, 'wam_runtime.ex', RuntimePath),
    open(RuntimePath, write, RS),
    write(RS, RuntimeCode),
    close(RS),
    source_file(test_lmdb_int_ids_real_lmdb_e2e, SrcFile),
    file_directory_name(SrcFile, TestsDir),
    directory_file_path(TestsDir, 'elixir_e2e/lmdb_int_ids_real_test.exs', TestSrc),
    directory_file_path(TmpDir, 'lmdb_int_ids_real_test.exs', TestDst),
    copy_file(TestSrc, TestDst),
    process_create(path(elixir),
                   ['lmdb_int_ids_real_test.exs'],
                   [cwd(TmpDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, StdOut),
    read_string(Err, _, StdErr),
    process_wait(Pid, Status),
    close(Out),
    close(Err),
    split_string(StdOut, "\n", "", Lines),
    (   member(SkipLine, Lines),
        string_concat("[SKIP] ", _, SkipLine)
    ->  format('[SKIP] ~w: ~s~n', [Test, SkipLine])
    ;   findall(L, (member(L, Lines), string_concat("[PASS] ", _, L)), PassLines),
        findall(L, (member(L, Lines), string_concat("[FAIL] ", _, L)), FailLines),
        length(PassLines, NPass),
        length(FailLines, NFail),
        (   Status = exit(0), NPass >= 3, NFail = 0
        ->  format('[PASS] ~w (~w sub-tests via real LMDB)~n', [Test, NPass])
        ;   format('[FAIL] ~w: status=~w pass=~w fail=~w~n', [Test, Status, NPass, NFail]),
            (   FailLines = [_|_]
            ->  forall(member(F, FailLines), format('  ~s~n', [F]))
            ;   true
            ),
            (   StdErr \= ""
            ->  format('  stderr:~n~s~n', [StdErr])
            ;   true
            ),
            assertz(test_failed)
        )
    ).

test_lmdb_int_ids_migrate_from_string_keyed_emitted :-
    % Migration helper: existing PR #1792 string-keyed Lmdb env
    % -> int-id-keyed LmdbIntIds env. Without this, deployments
    % that already use the original Lmdb adaptor have no path to
    % the int-id win.
    Test = 'LmdbIntIds: migrate_from_string_keyed/3 helper emitted, batches via ingest_pairs/3',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'def migrate_from_string_keyed('),
        % Reads from the existing PR #1792 Lmdb adaptor.
        sub_string(RuntimeCode, _, _, _, 'WamRuntime.FactSource.Lmdb.stream_all'),
        % Batches via Enum.chunk_every for memory safety at scale.
        sub_string(RuntimeCode, _, _, _, 'Enum.chunk_every(batch_size)'),
        % Delegates to ingest_pairs/3 (so insert-time ID assignment
        % logic lives in one place).
        sub_string(RuntimeCode, _, _, _, 'ingest_pairs(dest_handle, batch')
    ->  pass(Test)
    ;   fail_test(Test, 'migrate_from_string_keyed/3 missing or has wrong shape')
    ).

%% Atom-interning experiment (opt-in via intern_atoms(true) Option):
%  identifier-shape constants emit as Elixir atom literals (`:foo`)
%  instead of binaries (`"foo"`). BEAM atoms compare via pointer
%  equality and avoid binary copies on hot paths. Non-identifier
%  constants (whitespace, leading uppercase, etc.) still emit as
%  quoted strings — `:Foo` would mean a module reference in Elixir.
%% First graph kernel — transitive closure. Per
%  docs/WAM_TARGET_ROADMAP.md, kernel-based lowering is the largest
%  unrealised perf lever for graph workloads. This kernel emits as
%  WamRuntime.GraphKernel.TransitiveClosure and is callable directly
%  from driver code; future work adds compile-time pattern recognition
%  to route source-Prolog tc/2 calls through the kernel automatically.
%% Kernel dispatch — uses the shared target-neutral detector
%  (src/unifyweaver/core/recursive_kernel_detection.pl). The detector
%  is the same one Rust uses; structural pattern matching against
%  registered kernel kinds (transitive_closure2, category_ancestor,
%  transitive_distance3, ...). When a kernel matches AND the project
%  Options include kernel_dispatch(true), Elixir emits a kernel-
%  dispatch module that bypasses the WAM lower chain.
:- dynamic user:kdtc/2, user:kdedge/2.
:- dynamic user:kdca/4, user:kdcparent/2, user:max_depth/1.

setup_kernel_fixtures :-
    retractall(user:kdtc(_, _)),
    retractall(user:kdedge(_, _)),
    retractall(user:kdca(_, _, _, _)),
    retractall(user:kdcparent(_, _)),
    retractall(user:kdtd(_, _, _)),
    retractall(user:max_depth(_)),
    assertz((user:kdtc(X, Z) :- user:kdedge(X, Z))),
    assertz((user:kdtc(X, Z) :- user:kdedge(X, Y), user:kdtc(Y, Z))),
    assertz(user:max_depth(7)),
    assertz((user:kdca(C, P, 1, V) :- user:kdcparent(C, P), \+ member(P, V))),
    assertz((user:kdca(C, A, H, V) :-
                user:max_depth(MaxD), length(V, D), D < MaxD, !,
                user:kdcparent(C, M), \+ member(M, V),
                user:kdca(M, A, H1, [M|V]),
                H is H1 + 1)),
    % transitive_distance3 fixture: matches the canonical shape
    %   td(X, Y, 1) :- edge(X, Y).
    %   td(X, Y, N) :- edge(X, Mid), td(Mid, Y, N1), N is N1 + 1.
    assertz((user:kdtd(X, Y, 1) :- user:kdedge(X, Y))),
    assertz((user:kdtd(X, Y, N) :-
                user:kdedge(X, Mid),
                user:kdtd(Mid, Y, N1),
                N is N1 + 1)),
    % transitive_parent_distance4 fixture: matches the canonical shape
    %   pd(X, Y, X, 1) :- edge(X, Y).                  % parent == start, dist == 1
    %   pd(X, Y, P, N) :- edge(X, Mid), pd(Mid, Y, P, N1), N is N1 + 1.
    retractall(user:kdpd(_, _, _, _)),
    assertz((user:kdpd(X, Y, X, 1) :- user:kdedge(X, Y))),
    assertz((user:kdpd(X, Y, P, N) :-
                user:kdedge(X, Mid),
                user:kdpd(Mid, Y, P, N1),
                N is N1 + 1)),
    % transitive_step_parent_distance5 fixture: matches the canonical shape
    %   sp(X, Y, Y, X, 1) :- edge(X, Y).                       % step==target==Y, parent==start, dist==1
    %   sp(X, Y, Mid, P, N) :- edge(X, Mid), sp(Mid, Y, _, P, N1), N is N1 + 1.
    retractall(user:kdsp(_, _, _, _, _)),
    assertz((user:kdsp(X, Y, Y, X, 1) :- user:kdedge(X, Y))),
    assertz((user:kdsp(X, Y, Mid, P, N) :-
                user:kdedge(X, Mid),
                user:kdsp(Mid, Y, _IgnoredStep, P, N1),
                N is N1 + 1)),
    % weighted_shortest_path3 fixture: matches the canonical shape over
    % a 3-arity weighted_edge predicate (kdweighted_edge in this fixture).
    %   wsp(X, Y, W) :- weighted_edge(X, Y, W).
    %   wsp(X, Y, TotalW) :- weighted_edge(X, Mid, W1), wsp(Mid, Y, RestW), TotalW is RestW + W1.
    retractall(user:kdweighted_edge(_, _, _)),
    retractall(user:kdwsp(_, _, _)),
    assertz((user:kdwsp(X, Y, W) :- user:kdweighted_edge(X, Y, W))),
    assertz((user:kdwsp(X, Y, TotalW) :-
                user:kdweighted_edge(X, Mid, W1),
                user:kdwsp(Mid, Y, RestW),
                TotalW is RestW + W1)),
    % astar_shortest_path4 fixture: 4-ary canonical shape with Dim
    % passthrough. Reuses kdweighted_edge/3 as the forward edge source;
    % a separate kddirect/3 is asserted as the heuristic source via
    % the optional direct_dist_pred/1 user fact (detector falls back
    % to the same edge predicate if no direct_dist_pred is asserted).
    %   astar(X, Y, _Dim, W) :- weighted_edge(X, Y, W).
    %   astar(X, Y, Dim, TotalW) :- weighted_edge(X, Mid, W1),
    %                              astar(Mid, Y, Dim, RestW),
    %                              TotalW is RestW + W1.
    retractall(user:kdastar(_, _, _, _)),
    retractall(user:direct_dist_pred(_)),
    retractall(user:dimensionality(_)),
    assertz((user:kdastar(X, Y, _Dim, W) :- user:kdweighted_edge(X, Y, W))),
    assertz((user:kdastar(X, Y, Dim, TotalW) :-
                user:kdweighted_edge(X, Mid, W1),
                user:kdastar(Mid, Y, Dim, RestW),
                TotalW is RestW + W1)),
    % Optional config the detector reads: tells it to use kdweighted_edge/3
    % as the heuristic source (same as forward edges — admissible because
    % a direct edges weight is a lower bound on shortest path through it).
    assertz(user:direct_dist_pred(kdweighted_edge/3)),
    assertz(user:dimensionality(5)).

test_shared_detector_finds_tc :-
    Test = 'Shared detector: kdtc/2 with canonical TC shape detected as transitive_closure2',
    setup_kernel_fixtures,
    functor(Head, kdtc, 2),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdtc, 2, Clauses,
            recursive_kernel(transitive_closure2, kdtc/2, ConfigOps)),
        member(edge_pred(kdedge/2), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'detector did not find transitive_closure2 kernel')
    ).

test_shared_detector_finds_category_ancestor :-
    Test = 'Shared detector: kdca/4 with canonical category_ancestor shape detected',
    setup_kernel_fixtures,
    functor(Head, kdca, 4),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdca, 4, Clauses,
            recursive_kernel(category_ancestor, kdca/4, ConfigOps)),
        member(max_depth(7), ConfigOps),
        member(edge_pred(kdcparent/2), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'detector did not find category_ancestor kernel')
    ).

test_kernel_dispatch_emits_tc_module :-
    Test = 'Kernel dispatch: transitive_closure2 kernel emits Probe.Tc dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdtc/2, [], TcWam),
    write_wam_elixir_project([kdtc/2-TcWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_tc'),
    read_file_to_string('/tmp/test_kernel_disp_tc/lib/kdtc.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveClosure"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?")
    ->  pass(Test)
    ;   fail_test(Test, 'TC kernel dispatch module not emitted as expected')
    ).

test_kernel_dispatch_emits_category_ancestor_module :-
    Test = 'Kernel dispatch: category_ancestor kernel emits Probe.Kdca dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdca/4, [], CaWam),
    write_wam_elixir_project([kdca/4-CaWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_ca'),
    read_file_to_string('/tmp/test_kernel_disp_ca/lib/kdca.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.CategoryAncestor"),
        sub_string(S, _, _, _, "@max_depth 7"),
        sub_string(S, _, _, _, "collect_hops")
    ->  pass(Test)
    ;   fail_test(Test, 'category_ancestor kernel dispatch module not emitted as expected')
    ).

test_kernel_dispatch_uses_fold_form_in_aggregate_frame :-
    Test = 'Kernel dispatch: category_ancestor wrapper uses fold_hops + split_at_aggregate_cp when in aggregate frame',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdca/4, [], CaWam),
    write_wam_elixir_project([kdca/4-CaWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_ca_fold'),
    read_file_to_string('/tmp/test_kernel_disp_ca_fold/lib/kdca.ex', S, []),
    (   sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        sub_string(S, _, _, _, "fold_hops("),
        % Fix for PR #1813's per-hit cp-walk regression: extract the
        % aggregate cp once and thread agg_accum directly through the fold.
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)"),
        % Fall-through (backtracking) path still uses collect_hops.
        sub_string(S, _, _, _, "collect_hops(")
    ->  pass(Test)
    ;   fail_test(Test, 'dispatch wrapper does not branch into fold-form for aggregate frames')
    ).

test_runtime_emits_fold_hops :-
    Test = 'GraphKernel CategoryAncestor: runtime emits fold_hops + fold_hops_with_dests',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'def fold_hops(neighbors_fn'),
        sub_string(RuntimeCode, _, _, _, 'def fold_hops_with_dests(dests_fn'),
        sub_string(RuntimeCode, _, _, _, 'def fold_hops_with_dests_seeded(dests_fn'),
        sub_string(RuntimeCode, _, _, _, 'defp fold_n_recurse('),
        sub_string(RuntimeCode, _, _, _, 'defp fold_d_recurse(')
    ->  pass(Test)
    ;   fail_test(Test, 'fold_hops/6 (tuple variant) or its walkers missing from runtime')
    ).

test_runtime_emits_aggregate_push_one :-
    Test = 'WamRuntime: runtime emits aggregate_push_one/2 helper',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'def aggregate_push_one(state, value)')
    ->  pass(Test)
    ;   fail_test(Test, 'aggregate_push_one/2 helper missing from runtime')
    ).

test_runtime_emits_split_at_aggregate_cp :-
    Test = 'WamRuntime: runtime emits split_at_aggregate_cp/1 helper for one-pass cp-stack walk',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'def split_at_aggregate_cp(state)')
    ->  pass(Test)
    ;   fail_test(Test, 'split_at_aggregate_cp/1 helper missing from runtime')
    ).

test_kernel_docstring_documents_integer_id_path :-
    % Production-scale workloads (e.g. full Wikipedia category data
    % ~1M unique categories) exceed the BEAM atom table cap. The kernel
    % itself is term-agnostic; the docstring must surface integer-id
    % usage so a future maintainer doesn't add an atom-only assumption.
    Test = 'GraphKernel CategoryAncestor: docstring documents integer-id scale-up path (atoms < 50k, int-tuple > 50k)',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'kernel is term-agnostic'),
        sub_string(RuntimeCode, _, _, _, 'Integers with tuple-as-array'),
        % After the int-id LMDB FactSource doc fix, the old "no separate
        % intern step" claim was removed. The replacement points readers
        % at the proposal doc that describes the LMDB-native design.
        sub_string(RuntimeCode, _, _, _, 'wam_elixir_lmdb_int_id_factsource.md')
    ->  pass(Test)
    ;   fail_test(Test, 'CategoryAncestor moduledoc missing the integer-id scale-up note')
    ).

test_graph_kernel_tc_emitted_in_runtime :-
    Test = 'GraphKernel TC: runtime assembly emits WamRuntime.GraphKernel.TransitiveClosure',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.TransitiveClosure do'),
        sub_string(RuntimeCode, _, _, _, 'def reachable_from('),
        sub_string(RuntimeCode, _, _, _, 'def reachable_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.TransitiveClosure module missing expected API')
    ).

test_graph_kernel_transitive_distance_emitted_in_runtime :-
    % First of the five kernel kinds Rust+Haskell have but Elixir was missing
    % (per the audit in benchmarks/wam_effective_distance_cross_target.md
    % footnote about kernel coverage). This fills the simplest of the five —
    % shape ports cleanly from the existing CategoryAncestor walker.
    Test = 'GraphKernel TransitiveDistance: runtime assembly emits WamRuntime.GraphKernel.TransitiveDistance',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.TransitiveDistance do'),
        sub_string(RuntimeCode, _, _, _, 'def collect_pairs('),
        sub_string(RuntimeCode, _, _, _, 'def collect_pairs_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.TransitiveDistance module missing expected API')
    ).

test_graph_kernel_transitive_distance_uses_per_path_visited :-
    % Per-path visited list (matches Rust collect_native_transitive_distance_results).
    % Without it the recursion would loop on cyclic graphs. The walker should
    % use :lists.member for the visited check (consistent with PR #1817 where
    % the explicit :lists.member call beats `Enum.member?` on the hot path).
    Test = 'GraphKernel TransitiveDistance: kernel uses per-path visited list with :lists.member',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Locate the TransitiveDistance module body and check its walker
    % uses :lists.member — anchored to the module to avoid matching
    % CategoryAncestors visited check.
    Pattern = "defmodule WamRuntime.GraphKernel.TransitiveDistance do",
    EndPattern = "defmodule WamRuntime.GraphKernel.CategoryAncestor do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        sub_string(Body, _, _, _, ":lists.member(target, visited)"),
        sub_string(Body, _, _, _, "[target | visited]")
    ->  pass(Test)
    ;   fail_test(Test, 'TransitiveDistance walker missing :lists.member visited check')
    ).

test_kernel_dispatch_emits_transitive_distance_module :-
    % End-to-end: detector recognises kdtd/3 as transitive_distance3,
    % the dispatch wrapper emits with the correct shape (calls
    % collect_pairs, branches on agg_value_reg for findall slicing,
    % binds two regs in driver-direct mode).
    Test = 'Kernel dispatch: transitive_distance3 kernel emits Probe.Kdtd dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdtd/3, [], TdWam),
    write_wam_elixir_project([kdtd/3-TdWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_td'),
    read_file_to_string('/tmp/test_kernel_disp_td/lib/kdtd.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveDistance"),
        sub_string(S, _, _, _, "collect_pairs("),
        % Aggregate-frame slicing logic: pick targets/distances/pairs
        % based on which register the active aggregate is capturing.
        sub_string(S, _, _, _, "agg_cp.agg_value_reg"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        % Driver-direct binding of both target (reg 2) and distance (reg 3).
        sub_string(S, _, _, _, "bind_two_regs(state, "),
        % Falls through to collect_pairs (no separate fold variant yet —
        % future work; for now the dispatch matches the TC shape
        % structurally).
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)")
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_distance3 dispatch module missing expected shape')
    ).

test_shared_detector_finds_transitive_distance :-
    % Sanity check that the shared detector recognises the canonical
    % td/3 shape. Same form as test_shared_detector_finds_tc.
    Test = 'Shared detector: kdtd/3 with canonical transitive_distance shape detected',
    setup_kernel_fixtures,
    functor(Head, kdtd, 3),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdtd, 3, Clauses,
            recursive_kernel(transitive_distance3, kdtd/3, ConfigOps)),
        member(edge_pred(kdedge/2), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'kdtd/3 not detected as transitive_distance3 by shared detector')
    ).

test_graph_kernel_transitive_parent_distance_emitted_in_runtime :-
    % Second of the five missing kernels (after TransitiveDistance in PR
    % #1822). Stack-based DFS with no cycle detection — matches Rust's
    % collect_native_transitive_parent_distance_results reference exactly.
    Test = 'GraphKernel TransitiveParentDistance: runtime emits WamRuntime.GraphKernel.TransitiveParentDistance',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.TransitiveParentDistance do'),
        sub_string(RuntimeCode, _, _, _, 'def collect_triples('),
        sub_string(RuntimeCode, _, _, _, 'def collect_triples_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.TransitiveParentDistance module missing expected API')
    ).

test_graph_kernel_transitive_parent_distance_no_visited_set :-
    % Documented contract: NO cycle detection, matches Rust's
    % stack-based DFS exactly. The walker must NOT carry a visited
    % list — that would change the per-path enumeration semantics.
    % Locate the TransitiveParentDistance module body and check it.
    Test = 'GraphKernel TransitiveParentDistance: walker has no visited list (matches Rust DFS)',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.TransitiveParentDistance do",
    EndPattern = "defmodule WamRuntime.GraphKernel.CategoryAncestor do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        % Stack-based DFS shape: explicit `[{node, depth} | rest]`
        % stack pattern in the walker, with the records pushed as
        % `(target, predecessor, next_depth)` triples.
        sub_string(Body, _, _, _, "[{node, depth} | rest]"),
        sub_string(Body, _, _, _, "{[{target, node, next_depth}"),
        % Negative invariant: no `:lists.member` visited check (would
        % silently change the semantics from "every edge in DFS" to
        % "every edge with simple-path constraint").
        \+ sub_string(Body, _, _, _, ":lists.member"),
        % Documents the cycle caveat in the moduledoc.
        sub_string(Body, _, _, _, "NO cycle detection")
    ->  pass(Test)
    ;   fail_test(Test, 'TransitiveParentDistance walker has wrong shape or missing cycle caveat')
    ).

test_kernel_dispatch_emits_transitive_parent_distance_module :-
    % End-to-end: detector recognises kdpd/4 as
    % transitive_parent_distance4, dispatch wrapper emits with the
    % correct shape (calls collect_triples, branches on agg_value_reg
    % across reg 2/3/4 for findall slicing, binds three regs in
    % driver-direct mode).
    Test = 'Kernel dispatch: transitive_parent_distance4 kernel emits Probe.Kdpd dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdpd/4, [], PdWam),
    write_wam_elixir_project([kdpd/4-PdWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_pd'),
    read_file_to_string('/tmp/test_kernel_disp_pd/lib/kdpd.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveParentDistance"),
        sub_string(S, _, _, _, "collect_triples("),
        % Aggregate-frame slicing for three registers (2/3/4).
        sub_string(S, _, _, _, "agg_cp.agg_value_reg"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        % Driver-direct binding of all three (target, parent, distance).
        sub_string(S, _, _, _, "bind_three_regs(state, "),
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)")
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_parent_distance4 dispatch module missing expected shape')
    ).

test_shared_detector_finds_transitive_parent_distance :-
    Test = 'Shared detector: kdpd/4 with canonical transitive_parent_distance shape detected',
    setup_kernel_fixtures,
    functor(Head, kdpd, 4),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdpd, 4, Clauses,
            recursive_kernel(transitive_parent_distance4, kdpd/4, ConfigOps)),
        member(edge_pred(kdedge/2), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'kdpd/4 not detected as transitive_parent_distance4 by shared detector')
    ).

test_graph_kernel_transitive_step_parent_distance_emitted_in_runtime :-
    % Third missing kernel after PRs #1822/#1823. Reuses
    % TransitiveParentDistance for the inner walk; tags every result
    % with the FIRST hop taken from start. Same no-cycle-detection
    % contract as transitive_parent_distance4.
    Test = 'GraphKernel TransitiveStepParentDistance: runtime emits WamRuntime.GraphKernel.TransitiveStepParentDistance',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.TransitiveStepParentDistance do'),
        sub_string(RuntimeCode, _, _, _, 'def collect_quads('),
        sub_string(RuntimeCode, _, _, _, 'def collect_quads_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.TransitiveStepParentDistance module missing expected API')
    ).

test_graph_kernel_tspd_reuses_parent_distance_walker :-
    % Documents the implementation strategy: instead of writing a
    % third walker, this kernel reuses TransitiveParentDistance's
    % collect_triples and decorates results with the first-hop step.
    % Negative invariant: there should NOT be a separate `walk` defp
    % inside TSPD module body — it shouldnt have its own DFS walker.
    Test = 'GraphKernel TransitiveStepParentDistance: reuses TransitiveParentDistance.collect_triples',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.TransitiveStepParentDistance do",
    EndPattern = "defmodule WamRuntime.GraphKernel.CategoryAncestor do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        % Calls TransitiveParentDistance.collect_triples for the inner walk.
        sub_string(Body, _, _, _, "WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples"),
        % Emits the depth-1 base case: {next, next, start, 1}.
        sub_string(Body, _, _, _, "{next, next, start, 1}"),
        % Bumps inner triples by 1 (dist + 1).
        sub_string(Body, _, _, _, "dist + 1"),
        % Documents the cycle caveat (inherited from
        % TransitiveParentDistance).
        sub_string(Body, _, _, _, "NO cycle detection")
    ->  pass(Test)
    ;   fail_test(Test, 'TransitiveStepParentDistance does not reuse the parent_distance walker correctly')
    ).

test_kernel_dispatch_emits_transitive_step_parent_distance_module :-
    % End-to-end: detector recognises kdsp/5 as
    % transitive_step_parent_distance5, dispatch wrapper emits
    % FOUR-register binding (target, step, parent, distance).
    Test = 'Kernel dispatch: transitive_step_parent_distance5 kernel emits Probe.Kdsp dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdsp/5, [], SpWam),
    write_wam_elixir_project([kdsp/5-SpWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_sp'),
    read_file_to_string('/tmp/test_kernel_disp_sp/lib/kdsp.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveStepParentDistance"),
        sub_string(S, _, _, _, "collect_quads("),
        % Aggregate-frame slicing for FOUR registers (2/3/4/5).
        sub_string(S, _, _, _, "agg_cp.agg_value_reg"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        % Driver-direct binding of all four (target, step, parent, distance).
        sub_string(S, _, _, _, "bind_four_regs(state, "),
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)")
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_step_parent_distance5 dispatch module missing expected shape')
    ).

test_shared_detector_finds_transitive_step_parent_distance :-
    Test = 'Shared detector: kdsp/5 with canonical transitive_step_parent_distance shape detected',
    setup_kernel_fixtures,
    functor(Head, kdsp, 5),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdsp, 5, Clauses,
            recursive_kernel(transitive_step_parent_distance5, kdsp/5, ConfigOps)),
        member(edge_pred(kdedge/2), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'kdsp/5 not detected as transitive_step_parent_distance5 by shared detector')
    ).

test_graph_kernel_weighted_shortest_path_emitted_in_runtime :-
    % First weighted-graph kernel (after the four unweighted DFS
    % kernels in PRs #1799/#1803/#1822/#1823/#1824). Dijkstra via
    % :gb_sets priority queue. Same testing posture as the
    % unweighted kernels: emit-and-grep on the runtime + dispatch
    % wrapper, then end-to-end smoke test against the runtime.
    Test = 'GraphKernel WeightedShortestPath: runtime emits WamRuntime.GraphKernel.WeightedShortestPath',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.WeightedShortestPath do'),
        sub_string(RuntimeCode, _, _, _, 'def collect_path_costs('),
        sub_string(RuntimeCode, _, _, _, 'def collect_path_costs_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.WeightedShortestPath module missing expected API')
    ).

test_graph_kernel_wsp_uses_gb_sets_priority_queue :-
    % BEAMs natural min-heap is :gb_sets ordered by Erlang term
    % comparison ({cost, node} tuples sort by cost first, then node).
    % :gb_sets.take_smallest/1 is the pop primitive. Without a real
    % priority queue Dijkstra is O(V^2); with it we get O((V+E) log V).
    Test = 'GraphKernel WeightedShortestPath: uses :gb_sets priority queue for Dijkstra',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.WeightedShortestPath do",
    EndPattern = "defmodule WamRuntime.GraphKernel.CategoryAncestor do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        % Min-heap construction + pop.
        sub_string(Body, _, _, _, ":gb_sets.singleton({0, start})"),
        sub_string(Body, _, _, _, ":gb_sets.take_smallest(heap)"),
        % Stale-entry skip (the canonical Dijkstra optimisation).
        sub_string(Body, _, _, _, "if cost > best do"),
        % Documents the semantic narrowing (all-paths -> shortest).
        sub_string(Body, _, _, _, "semantic narrowing"),
        sub_string(Body, _, _, _, "computes only the SHORTEST")
    ->  pass(Test)
    ;   fail_test(Test, 'WeightedShortestPath kernel missing :gb_sets primitives or semantic-narrowing note')
    ).

test_kernel_dispatch_emits_weighted_shortest_path_module :-
    % End-to-end: detector recognises kdwsp/3 as
    % weighted_shortest_path3, dispatch wrapper emits with two-register
    % binding (target, cost) and 3-arity edge predicate ("kdweighted_edge/3").
    Test = 'Kernel dispatch: weighted_shortest_path3 kernel emits Probe.Kdwsp dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdwsp/3, [], WspWam),
    write_wam_elixir_project([kdwsp/3-WspWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_wsp'),
    read_file_to_string('/tmp/test_kernel_disp_wsp/lib/kdwsp.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.WeightedShortestPath"),
        sub_string(S, _, _, _, "collect_path_costs("),
        % 3-arity edge predicate indicator: weighted_edge/3
        sub_string(S, _, _, _, "kdweighted_edge/3"),
        % Aggregate-frame slicing for two regs (target=2, cost=3).
        sub_string(S, _, _, _, "agg_cp.agg_value_reg"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        % Driver-direct binding of both regs.
        sub_string(S, _, _, _, "bind_two_regs(state, "),
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)")
    ->  pass(Test)
    ;   fail_test(Test, 'weighted_shortest_path3 dispatch module missing expected shape')
    ).

test_shared_detector_finds_weighted_shortest_path :-
    Test = 'Shared detector: kdwsp/3 with canonical weighted_shortest_path shape detected',
    setup_kernel_fixtures,
    functor(Head, kdwsp, 3),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdwsp, 3, Clauses,
            recursive_kernel(weighted_shortest_path3, kdwsp/3, ConfigOps)),
        member(edge_pred(kdweighted_edge/3), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'kdwsp/3 not detected as weighted_shortest_path3 by shared detector')
    ).

test_graph_kernel_astar_shortest_path_emitted_in_runtime :-
    % Final kernel — closes the 7-of-7 coverage table. Builds on
    % WSPs Dijkstra primitives (:gb_sets priority queue + dist map)
    % adding a heuristic estimate and early termination.
    Test = 'GraphKernel AstarShortestPath: runtime emits WamRuntime.GraphKernel.AstarShortestPath',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.GraphKernel.AstarShortestPath do'),
        sub_string(RuntimeCode, _, _, _, 'def collect_path_costs('),
        sub_string(RuntimeCode, _, _, _, 'def collect_path_costs_from_source(')
    ->  pass(Test)
    ;   fail_test(Test, 'GraphKernel.AstarShortestPath module missing expected API')
    ).

test_graph_kernel_astar_uses_minkowski_f_cost :-
    % UnifyWeavers A* uses Minkowski-style f(n) = g^D + h^D, NOT the
    % standard f = g + h. Per Rusts reference impl which documents
    % "By Minkowski inequality this is admissible and tighter than L1
    % A*". The kernel must use :math.pow for both g and h to preserve
    % this contract.
    Test = 'GraphKernel AstarShortestPath: f-cost uses Minkowski g^D + h^D (NOT g + h)',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.AstarShortestPath do",
    EndPattern = "defmodule WamRuntime.GraphKernel.CategoryAncestor do",
    (   sub_string(RuntimeCode, Start, _, _, Pattern),
        sub_string(RuntimeCode, End, _, _, EndPattern),
        End > Start,
        BodyLen is End - Start,
        sub_string(RuntimeCode, Start, BodyLen, _, Body),
        % Minkowski f-cost helper.
        sub_string(Body, _, _, _, ":math.pow(g, dim) + :math.pow(h, dim)"),
        % Stale-entry skip + early termination on target.
        sub_string(Body, _, _, _, "g_cost > best"),
        sub_string(Body, _, _, _, "target != nil and node == target"),
        % Documents the Minkowski narrowing.
        sub_string(Body, _, _, _, "Minkowski-style f-cost"),
        % :gb_sets primitives reused from WSP.
        sub_string(Body, _, _, _, ":gb_sets.take_smallest(heap)")
    ->  pass(Test)
    ;   fail_test(Test, 'AstarShortestPath kernel missing Minkowski f-cost or A* primitives')
    ).

test_kernel_dispatch_emits_astar_shortest_path_module :-
    % End-to-end: detector recognises kdastar/4 as
    % astar_shortest_path4 (with the user-asserted direct_dist_pred
    % and dimensionality facts), dispatch wrapper emits with TWO
    % FactSource lookups + two-register binding (target=2, cost=4),
    % skipping target rebinding when bound on entry.
    Test = 'Kernel dispatch: astar_shortest_path4 kernel emits Probe.Kdastar dispatch module',
    setup_kernel_fixtures,
    wam_target:compile_predicate_to_wam(user:kdastar/4, [], AstarWam),
    write_wam_elixir_project([kdastar/4-AstarWam],
        [module_name(probe), emit_mode(lowered),
         intra_query_parallel(false),
         kernel_dispatch(true), source_module(user)],
        '/tmp/test_kernel_disp_astar'),
    read_file_to_string('/tmp/test_kernel_disp_astar/lib/kdastar.ex', S, []),
    (   sub_string(S, _, _, _, "WamRuntime.GraphKernel.AstarShortestPath"),
        sub_string(S, _, _, _, "collect_path_costs("),
        % TWO FactSource lookups (forward + heuristic).
        sub_string(S, _, _, _, "weighted_handle = WamRuntime.FactSourceRegistry.lookup!"),
        sub_string(S, _, _, _, "direct_handle = WamRuntime.FactSourceRegistry.lookup!"),
        % Edge indicator is /3.
        sub_string(S, _, _, _, "kdweighted_edge/3"),
        % Compile-time dim default attribute.
        sub_string(S, _, _, _, "@default_dim 5"),
        % Aggregate-frame slicing for two regs (target=2, cost=4).
        sub_string(S, _, _, _, "agg_cp.agg_value_reg"),
        sub_string(S, _, _, _, "in_forkable_aggregate_frame?"),
        % Driver-direct binding: target may be already bound on entry.
        sub_string(S, _, _, _, "bind_target_and_cost(state, "),
        sub_string(S, _, _, _, "split_at_aggregate_cp(state)")
    ->  pass(Test)
    ;   fail_test(Test, 'astar_shortest_path4 dispatch module missing expected shape')
    ).

test_shared_detector_finds_astar_shortest_path :-
    Test = 'Shared detector: kdastar/4 with canonical astar_shortest_path shape detected',
    setup_kernel_fixtures,
    functor(Head, kdastar, 4),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   detect_recursive_kernel(kdastar, 4, Clauses,
            recursive_kernel(astar_shortest_path4, kdastar/4, ConfigOps)),
        member(edge_pred(kdweighted_edge/3), ConfigOps),
        member(direct_dist_pred(kdweighted_edge/3), ConfigOps),
        member(dimensionality(5), ConfigOps)
    ->  pass(Test)
    ;   fail_test(Test, 'kdastar/4 not detected as astar_shortest_path4 by shared detector')
    ).

test_graph_kernel_tc_uses_visited_tracking :-
    Test = 'GraphKernel TC: kernel uses MapSet for visited tracking (avoids O(N^2) revisits)',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % Without visited tracking, transitive closure on a cyclic graph
    % loops forever. MapSet is the BEAM-native O(log N) set; the
    % kernel must use it to be correct AND to be faster than WAMs
    % naive recursive tc/2.
    (   sub_string(RuntimeCode, _, _, _, 'MapSet.member?'),
        sub_string(RuntimeCode, _, _, _, 'MapSet.put'),
        sub_string(RuntimeCode, _, _, _, 'MapSet.new')
    ->  pass(Test)
    ;   fail_test(Test, 'TC kernel missing MapSet visited-tracking primitives')
    ).

test_graph_kernel_tc_factsource_bridge :-
    Test = 'GraphKernel TC: reachable_from_source bridges to FactSource lookup_by_arg1',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    % The bridge function lets callers pass a FactSource
    % (LMDB / SQLite / ETS / TSV) instead of building the
    % neighbors_fn closure themselves.
    (   sub_string(RuntimeCode, _, _, _, 'reachable_from_source(source_module, source_handle, start, state'),
        sub_string(RuntimeCode, _, _, _, 'source_module.lookup_by_arg1(source_handle, node, state)')
    ->  pass(Test)
    ;   fail_test(Test, 'reachable_from_source bridge missing or wrong shape')
    ).

test_intern_atoms_default_off :-
    Test = 'Atom interning: default (no intern_atoms option) emits constants as binary literals',
    setup_call_cleanup(
        (   retractall(intern_test:p(_)),
            assertz(intern_test:p('hello'))
        ),
        (   wam_target:compile_predicate_to_wam(intern_test:p/1, [], WamCode),
            lower_predicate_to_elixir(p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            sub_string(S, _, _, _, 'val == "hello"'),
            \+ sub_string(S, _, _, _, 'val == :hello')
        ->  pass(Test)
        ;   fail_test(Test, 'default mode wrongly emitted atom literal')
        ),
        retractall(intern_test:p(_))
    ).

test_intern_atoms_on_emits_atom_literals :-
    Test = 'Atom interning: intern_atoms(true) emits identifier constants as :atom literals',
    setup_call_cleanup(
        (   retractall(intern_test:p(_)),
            assertz(intern_test:p('hello')),
            assertz(wam_elixir_lowered_emitter:intern_atoms_enabled)
        ),
        (   wam_target:compile_predicate_to_wam(intern_test:p/1, [], WamCode),
            lower_predicate_to_elixir(p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            sub_string(S, _, _, _, 'val == :hello'),
            \+ sub_string(S, _, _, _, 'val == "hello"')
        ->  pass(Test)
        ;   fail_test(Test, 'intern_atoms mode failed to emit atom literal')
        ),
        (   retractall(intern_test:p(_)),
            retractall(wam_elixir_lowered_emitter:intern_atoms_enabled)
        )
    ).

test_intern_atoms_keeps_non_identifiers_as_strings :-
    Test = 'Atom interning: leading-uppercase / non-identifier constants stay as binary literals',
    setup_call_cleanup(
        (   retractall(intern_test:p(_)),
            assertz(intern_test:p('Foo')),         % uppercase-leading: would mean module
            assertz(intern_test:p('hello world')), % whitespace: not an identifier
            assertz(wam_elixir_lowered_emitter:intern_atoms_enabled)
        ),
        (   wam_target:compile_predicate_to_wam(intern_test:p/1, [], WamCode),
            lower_predicate_to_elixir(p/1, WamCode, [module_name('TestMod')], Code),
            atom_string(Code, S),
            sub_string(S, _, _, _, 'val == "Foo"'),
            sub_string(S, _, _, _, 'val == "hello world"'),
            % Sanity: did NOT emit `:Foo` (would parse as a module name).
            \+ sub_string(S, _, _, _, 'val == :Foo')
        ->  pass(Test)
        ;   fail_test(Test, 'non-identifier constants wrongly emitted as atoms')
        ),
        (   retractall(intern_test:p(_)),
            retractall(wam_elixir_lowered_emitter:intern_atoms_enabled)
        )
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
:- dynamic intern_test:p/1.

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
    test_call_n_dispatch_meta_helper,
    test_call_n_step_arms,
    test_true_zero_builtin,
    test_build_call_target_helpers,
    test_build_call_target_compound_clause,
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
    test_lmdb_int_ids_adaptor_emitted_in_runtime,
    test_lmdb_int_ids_adaptor_uses_indirect_module_resolution,
    test_lmdb_int_ids_design_proposal_referenced,
    test_lmdb_int_ids_ingest_pairs_emitted,
    test_lmdb_int_ids_migrate_from_string_keyed_emitted,
    test_lmdb_int_ids_mock_e2e,
    test_lmdb_int_ids_real_lmdb_e2e,
    test_shared_detector_finds_tc,
    test_shared_detector_finds_category_ancestor,
    test_kernel_dispatch_emits_tc_module,
    test_kernel_dispatch_emits_category_ancestor_module,
    test_kernel_dispatch_uses_fold_form_in_aggregate_frame,
    test_runtime_emits_fold_hops,
    test_runtime_emits_aggregate_push_one,
    test_runtime_emits_split_at_aggregate_cp,
    test_kernel_docstring_documents_integer_id_path,
    test_graph_kernel_tc_emitted_in_runtime,
    test_graph_kernel_transitive_distance_emitted_in_runtime,
    test_graph_kernel_transitive_distance_uses_per_path_visited,
    test_kernel_dispatch_emits_transitive_distance_module,
    test_shared_detector_finds_transitive_distance,
    test_graph_kernel_transitive_parent_distance_emitted_in_runtime,
    test_graph_kernel_transitive_parent_distance_no_visited_set,
    test_kernel_dispatch_emits_transitive_parent_distance_module,
    test_shared_detector_finds_transitive_parent_distance,
    test_graph_kernel_transitive_step_parent_distance_emitted_in_runtime,
    test_graph_kernel_tspd_reuses_parent_distance_walker,
    test_kernel_dispatch_emits_transitive_step_parent_distance_module,
    test_shared_detector_finds_transitive_step_parent_distance,
    test_graph_kernel_weighted_shortest_path_emitted_in_runtime,
    test_graph_kernel_wsp_uses_gb_sets_priority_queue,
    test_kernel_dispatch_emits_weighted_shortest_path_module,
    test_shared_detector_finds_weighted_shortest_path,
    test_graph_kernel_astar_shortest_path_emitted_in_runtime,
    test_graph_kernel_astar_uses_minkowski_f_cost,
    test_kernel_dispatch_emits_astar_shortest_path_module,
    test_shared_detector_finds_astar_shortest_path,
    test_graph_kernel_tc_uses_visited_tracking,
    test_graph_kernel_tc_factsource_bridge,
    test_intern_atoms_default_off,
    test_intern_atoms_on_emits_atom_literals,
    test_intern_atoms_keeps_non_identifiers_as_strings,
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

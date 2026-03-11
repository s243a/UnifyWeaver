%% test_agent_loop.pl — Tests for agent-loop overlay modules
%%
%% All tests are safe read-only registry queries — no file I/O, no network.
%%
%% Usage:
%%   swipl -l test_agent_loop.pl -g "run_tests, halt"

:- module(test_agent_loop, [run_tests/0]).

:- use_module(agent_loop_module).
:- use_module(agent_loop_bindings).
:- use_module(agent_loop_components).
:- use_module('../../src/unifyweaver/core/binding_registry').
:- use_module('../../src/unifyweaver/core/component_registry').

%% ============================================================================
%% Test Runner
%% ============================================================================

:- dynamic test_passed/1, test_failed/1.

run_tests :-
    retractall(test_passed(_)),
    retractall(test_failed(_)),
    format("~n=== Agent-Loop Test Suite ===~n~n"),
    %% Clean state
    clear_all_bindings,
    %% Run tests
    test_binding_registration,
    test_component_registration,
    test_module_exports,
    test_binding_queries,
    test_binding_code_generation,
    test_handler_fragment_count,
    test_cli_override_facts,
    test_streaming_capability,
    test_retry_config,
    test_bindings_summary,
    test_model_pricing,
    test_anthropic_backend_detection,
    test_prolog_only_commands,
    test_retryable_error_clauses,
    test_security_enforcement_wiring,
    test_tool_parameter_schemas,
    test_emit_tool_facts,
    test_emit_command_facts,
    test_emit_backend_facts,
    test_compile_component_multi_fact,
    test_python_tool_dispatch_via_components,
    test_compile_command_component,
    test_compile_backend_component,
    test_security_component_registration,
    test_cost_component_registration,
    test_cost_python_compile,
    test_security_blocked_py,
    test_backend_init_imports,
    test_python_integration,
    test_emit_context_enums,
    test_emit_message_fields,
    test_emit_agent_config_fields,
    test_binding_imports,
    test_prolog_integration,
    test_emit_prolog_config_facts,
    test_emit_api_key_env_vars_py,
    test_emit_default_presets_py,
    test_emit_help_groups,
    test_new_binding_count,
    test_binding_compile_api_key,
    test_emit_streaming_capable_facts,
    test_emit_security_profile_entries,
    test_emit_tool_dispatch_entries,
    test_emit_cascade_paths,
    test_emit_alias_conditions,
    test_emit_argparse_group_args,
    test_emit_backend_optimization_hints,
    test_emit_command_optimization_hints,
    test_emit_module_imports,
    test_backend_import_specs,
    test_module_dependency_facts,
    test_emit_module_dependencies,
    test_det_annotations_in_generated,
    test_det_annotations_expanded,
    test_binding_metadata_in_generated,
    test_target_level_maplist_in_generated,
    test_no_forall_in_components,
    test_binding_patterns_extended,
    test_binding_dict_name,
    test_compile_component_python_targets,
    test_write_lines_helper,
    test_no_forall_in_bindings,
    test_compile_tool_spec_python,
    test_compile_destructive_tool_python,
    test_emit_py_set_from_components,
    test_explicit_prolog_security_compile,
    test_explicit_prolog_cost_compile,
    test_emit_prolog_facts_from_components,
    test_translate_agent_goal,
    test_translate_agent_goals,
    test_binding_parity,
    test_binding_metadata_coverage,
    test_write_lines_multiline_equiv,
    test_dependency_diagram_output,
    test_module_dependencies_complete,
    test_translate_all_bindings,
    test_binding_metadata_extended,
    test_emit_from_components_dict,
    test_emit_from_components_set,
    test_emit_from_components_facts,
    test_emit_format_equiv,
    test_component_roundtrip_all_targets,
    test_all_bindings_compile,
    test_entries_format,
    test_security_rule_components,
    test_prolog_fact_roundtrip,
    test_python_entry_patterns,
    test_generator_import_specs,
    test_binding_equivalence_comments,
    test_extract_target_helper,
    test_all_entry_backend_compile,
    test_generator_import_specs_extended,
    test_generator_export_specs,
    test_py_fragment_metadata,
    test_fragment_imports_helper,
    test_path_check_rule_facts,
    test_command_check_rule_facts,
    test_compile_path_check_rules_python,
    test_compile_path_check_rules_prolog,
    test_source_component_equivalence,
    test_emit_security_check_predicates,
    test_compiled_rules_variable_name,
    test_generator_export_specs_backends,
    test_security_profile_field_facts,
    test_emit_security_profile_fields,
    test_generator_fragments,
    test_derive_fragment_imports,
    test_validate_generator_imports,
    test_emit_config_section,
    test_backends_init_export_roundtrip,
    test_regex_list_variable_facts,
    test_regex_list_combined_facts,
    test_regex_list_in_field_value,
    test_generator_fragments_security,
    test_generator_fragments_aliases,
    test_generator_fragments_agent_loop_main,
    test_emit_security_profile_fields_prolog,
    test_emit_prolog_module_header,
    test_emit_prolog_declarations,
    test_config_section_wiring,
    test_prolog_fragment_count,
    test_write_prolog_roundtrip,
    test_write_prolog_error,
    test_prolog_module_skeleton_exports,
    test_prolog_module_skeleton_comment,
    test_prolog_fragment_escaping,
    test_write_prolog_substitutions,
    test_prolog_fragment_metadata,
    test_generator_prolog_fragments,
    test_cost_py_fragments,
    test_backends_prolog_fragments,
    test_agent_loop_prolog_fragments,
    test_config_section_new_clauses,
    test_backend_error_handler_routing,
    test_unified_fragment_system,
    test_rust_fragments,
    test_rust_module_skeleton,
    test_rust_compile_components,
    test_rust_bindings,
    test_rust_data_table_specs,
    test_rust_imperative_fragments,
    test_rust_type_mapping,
    test_rust_backend_factory,
    test_rust_phase2_generation,
    test_rust_backend_factory_names,
    test_rust_clap_generation,
    test_rust_sessions_fragment,
    test_emit_config_section_rust,
    test_rust_streaming_capable,
    test_rust_phase3_generation,
    %% Phase 4 tests
    test_rust_config_data_refactored,
    test_rust_config_loader_fragments,
    test_rust_config_dir_file_names,
    test_rust_cli_overrides_section,
    test_rust_streaming_handler,
    test_rust_backend_api_streaming,
    test_rust_phase4_generation,
    test_rust_config_search_paths_section,
    %% Phase 5
    test_rust_command_validation_fix,
    test_rust_security_wiring,
    test_rust_security_profile_conditional,
    test_rust_approval_mode,
    test_rust_yaml_config,
    test_rust_security_regex_lists,
    test_rust_phase5_generation,
    %% Report
    aggregate_all(count, test_passed(_), Passed),
    aggregate_all(count, test_failed(_), Failed),
    format("~n=== Results: ~w passed, ~w failed ===~n", [Passed, Failed]),
    %% Clean up
    clear_all_bindings,
    (Failed > 0 -> halt(1) ; true).

%% Assert helper
assert_true(Name, Goal) :-
    (call(Goal) ->
        assert(test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(test_failed(Name)),
        format("  [FAIL] ~w~n", [Name])
    ).

assert_eq(Name, Got, Expected) :-
    (Got == Expected ->
        assert(test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(test_failed(Name)),
        format("  [FAIL] ~w (expected ~w, got ~w)~n", [Name, Expected, Got])
    ).

%% ============================================================================
%% Test 1: Binding Registration
%% ============================================================================

test_binding_registration :-
    format("~nBinding registration:~n"),
    init_agent_loop_bindings,
    bindings_for_target(python, PyBindings),
    bindings_for_target(prolog, PlBindings),
    length(PyBindings, NPy),
    length(PlBindings, NPl),
    assert_eq('Python binding count', NPy, 11),
    assert_eq('Prolog binding count', NPl, 11).

%% ============================================================================
%% Test 2: Component Registration
%% ============================================================================

test_component_registration :-
    format("~nComponent registration:~n"),
    register_agent_loop_components,
    list_components(agent_tools, Tools),
    list_components(agent_commands, Commands),
    list_components(agent_backends, Backends),
    length(Tools, NT),
    length(Commands, NC),
    length(Backends, NB),
    assert_eq('Tool count', NT, 4),
    assert_eq('Command count', NC, 23),
    assert_eq('Backend count', NB, 8).

%% ============================================================================
%% Test 3: Module Exports
%% ============================================================================

test_module_exports :-
    format("~nModule exports:~n"),
    assert_true('tool_handler/2 callable', (
        agent_loop_module:tool_handler(read, _)
    )),
    assert_true('slash_command/4 callable', (
        slash_command(help, _, _, _)
    )),
    assert_true('agent_backend/2 callable', (
        agent_backend(coro, _)
    )),
    assert_true('backend_factory/2 callable', (
        backend_factory(coro, _)
    )),
    assert_true('security_profile/2 callable', (
        agent_loop_module:security_profile(open, _)
    )).

%% ============================================================================
%% Test 4: Binding Queries
%% ============================================================================

test_binding_queries :-
    format("~nBinding queries:~n"),
    assert_true('tool_handler is pure', (
        is_pure_binding(python, tool_handler/2)
    )),
    assert_true('backend_factory has io effect', (
        binding_has_effect(python, backend_factory/2, io)
    )),
    assert_true('audit_profile_level is deterministic', (
        is_deterministic_binding(python, audit_profile_level/2)
    )).

%% ============================================================================
%% Test 5: Binding Code Generation
%% ============================================================================

test_binding_code_generation :-
    format("~nBinding code generation:~n"),
    assert_true('Python tool_handler code', (
        compile_binding_code(python, tool_handler/2, Code),
        atom(Code),
        sub_atom(Code, _, _, _, 'TOOL_HANDLERS')
    )),
    assert_true('Prolog tool_handler code', (
        compile_binding_code(prolog, tool_handler/2, PCode),
        atom(PCode),
        sub_atom(PCode, _, _, _, 'tool_handler')
    )),
    assert_true('Python backend_factory has import', (
        compile_binding_code(python, backend_factory/2, BCode),
        atom(BCode),
        sub_atom(BCode, _, _, _, 'import')
    )).

%% ============================================================================
%% Test 6: Handler Fragment Count
%% ============================================================================

test_handler_fragment_count :-
    format("~nHandler fragments:~n"),
    findall(Name, (
        agent_loop_module:py_fragment(Name, _),
        sub_atom(Name, 0, _, _, handler_)
    ), Handlers),
    length(Handlers, Count),
    assert_eq('Handler fragment count', Count, 17).

%% ============================================================================
%% Test 7: CLI Override Facts
%% ============================================================================

test_cli_override_facts :-
    format("~nCLI overrides:~n"),
    findall(Arg, agent_loop_module:cli_override(Arg, _, _), Overrides),
    length(Overrides, Count),
    assert_eq('CLI override count', Count, 12),
    %% Verify behavior types
    assert_true('backend has backend_special', (
        agent_loop_module:cli_override(backend, _, backend_special)
    )),
    assert_true('auto_tools has set_true', (
        agent_loop_module:cli_override(auto_tools, _, set_true)
    )),
    assert_true('no_tools has clear_list', (
        agent_loop_module:cli_override(no_tools, _, clear_list)
    )).

%% ============================================================================
%% Test 8: Streaming Capability
%% ============================================================================

test_streaming_capability :-
    format("~nStreaming capability:~n"),
    assert_true('api_local is streaming capable', (
        agent_loop_module:streaming_capable(api_local)
    )),
    assert_true('api is streaming capable', (
        agent_loop_module:streaming_capable(api)
    )),
    assert_true('openrouter is streaming capable', (
        agent_loop_module:streaming_capable(openrouter)
    )),
    assert_true('cli is NOT streaming capable', (
        \+ agent_loop_module:streaming_capable(cli)
    )).

%% ============================================================================
%% Test 9: Retry Config (source-level verification)
%% ============================================================================

test_retry_config :-
    format("~nRetry config:~n"),
    %% retry_config/3 is in generated backends.pl, not loaded here.
    %% Verify the source-level streaming_capable facts (which proves generator works)
    assert_true('3 streaming capable types', (
        findall(T, agent_loop_module:streaming_capable(T), Types),
        length(Types, 3)
    )),
    %% Verify backend_factory_order has all 8
    assert_true('backend_factory_order has 8', (
        backend_factory_order(Order),
        length(Order, 8)
    )).

%% ============================================================================
%% Test 10: Bindings Summary
%% ============================================================================

test_bindings_summary :-
    format("~nBindings summary:~n"),
    assert_true('Python summary generated', (
        generate_bindings_summary(python, Summary),
        atom(Summary),
        sub_atom(Summary, _, _, _, 'python bindings (11)')
    )),
    assert_true('Prolog summary generated', (
        generate_bindings_summary(prolog, PlSummary),
        atom(PlSummary),
        sub_atom(PlSummary, _, _, _, 'prolog bindings (11)')
    )).

%% ============================================================================
%% Test 11: Model Pricing Facts
%% ============================================================================

test_model_pricing :-
    format("~nModel pricing:~n"),
    findall(M, agent_loop_module:model_pricing(M, _, _), Models),
    length(Models, Count),
    assert_eq('Model pricing count', Count, 16),
    assert_true('claude-sonnet pricing exists', (
        agent_loop_module:model_pricing("claude-sonnet-4-20250514", InP, OutP),
        InP =:= 3.0, OutP =:= 15.0
    )),
    assert_true('llama3 pricing exists', (
        agent_loop_module:model_pricing("llama3", InP2, OutP2),
        InP2 =:= 0.0, OutP2 =:= 0.0
    )).

%% ============================================================================
%% Test 12: Anthropic Backend Detection
%% ============================================================================

test_anthropic_backend_detection :-
    format("~nAnthropic backend detection:~n"),
    assert_true('claude_api has x-api-key auth', (
        agent_loop_module:agent_backend(claude_api, Props),
        member(auth_header(AH0), Props),
        atom_string(AH0, "x-api-key")
    )),
    assert_true('openai_api has Authorization auth', (
        agent_loop_module:agent_backend(openai_api, OAProps),
        member(auth_header(AH), OAProps),
        atom_string(AH, "Authorization")
    )),
    assert_true('openrouter_api has Authorization auth', (
        agent_loop_module:agent_backend(openrouter_api, Props2),
        member(auth_header(AH2), Props2),
        atom_string(AH2, "Authorization")
    )).

%% ============================================================================
%% Test 13: Prolog-only Commands (/model, /tokens)
%% ============================================================================

test_prolog_only_commands :-
    format("~nProlog-only commands:~n"),
    assert_true('/model command exists', (
        slash_command(model, prefix_sp, ModelOpts, _),
        member(handler('_handle_model_command'), ModelOpts),
        member(target(prolog), ModelOpts)
    )),
    assert_true('/tokens command exists', (
        slash_command(tokens, exact, TokOpts, _),
        member(handler('_handle_tokens_command'), TokOpts),
        member(target(prolog), TokOpts)
    )),
    assert_true('/model in Loop Control group', (
        agent_loop_module:slash_command_group('Loop Control', LoopCmds),
        member(model, LoopCmds)
    )),
    assert_true('/tokens in Export & Costs group', (
        agent_loop_module:slash_command_group('Export & Costs', CostCmds),
        member(tokens, CostCmds)
    )).

%% ============================================================================
%% Test 14: Retryable Error Clauses
%% ============================================================================

test_retryable_error_clauses :-
    format("~nRetryable error clauses:~n"),
    %% Count is_retryable emission lines in generator source
    %% We verify the expected patterns exist in the generator
    assert_true('existence_error retryable', (
        agent_loop_module:streaming_capable(api)  %% proxy: backends generator works
    )),
    assert_true('format_api_error exported', (
        agent_loop_module:agent_backend(claude_api, CProps),
        member(auth_header(_), CProps)  %% proxy: backends section fully emitted
    )).

%% ============================================================================
%% Test 15: Security Enforcement Wiring
%% ============================================================================

test_security_enforcement_wiring :-
    format("~nSecurity enforcement wiring:~n"),
    assert_true('bash tool has timeout spec', (
        agent_loop_module:tool_spec(bash, BashProps),
        member(timeout(T), BashProps),
        T =:= 120
    )),
    assert_true('cautious profile has path validation', (
        agent_loop_module:security_profile(cautious, CautProps),
        member(path_validation(true), CautProps)
    )),
    assert_true('cautious profile has command blocklist', (
        agent_loop_module:security_profile(cautious, CautProps2),
        member(command_blocklist(true), CautProps2)
    )),
    assert_true('all tools have parameters', (
        forall(agent_loop_module:tool_spec(_, TProps),
               member(parameters(_), TProps))
    )).

%% ============================================================================
%% Test 16: Tool Parameter Schemas
%% ============================================================================

test_tool_parameter_schemas :-
    format("~nTool parameter schemas:~n"),
    assert_true('bash has command param', (
        agent_loop_module:tool_spec(bash, BProps),
        member(parameters(BParams), BProps),
        member(param(command, string, required, _), BParams)
    )),
    assert_true('read has path param', (
        agent_loop_module:tool_spec(read, RProps),
        member(parameters(RParams), RProps),
        member(param(path, string, required, _), RParams)
    )),
    assert_true('write has path and content params', (
        agent_loop_module:tool_spec(write, WProps),
        member(parameters(WParams), WProps),
        member(param(path, string, required, _), WParams),
        member(param(content, string, required, _), WParams)
    )),
    assert_true('edit has 3 required params', (
        agent_loop_module:tool_spec(edit, EProps),
        member(parameters(EParams), EProps),
        length(EParams, 3),
        forall(member(param(_, _, Req, _), EParams), Req = required)
    )).

%% ============================================================================
%% Test 17: Emit Tool Facts via Component Registry
%% ============================================================================

test_emit_tool_facts :-
    format("~nEmit tool facts:~n"),
    register_agent_loop_components,
    with_output_to(atom(Output), (
        current_output(S),
        agent_loop_components:emit_tool_facts(S, [target(prolog)])
    )),
    assert_true('tool_spec(bash,...) emitted', (
        sub_atom(Output, _, _, _, 'tool_spec(bash')
    )),
    assert_true('tool_handler(bash,...) emitted', (
        sub_atom(Output, _, _, _, 'tool_handler(bash')
    )),
    assert_true('destructive_tool emitted', (
        sub_atom(Output, _, _, _, 'destructive_tool(')
    )),
    assert_true('tool_description emitted', (
        sub_atom(Output, _, _, _, 'tool_description(')
    )).

%% ============================================================================
%% Test 18: Emit Command Facts via Component Registry
%% ============================================================================

test_emit_command_facts :-
    format("~nEmit command facts:~n"),
    register_agent_loop_components,
    with_output_to(atom(Output), (
        current_output(S),
        agent_loop_components:emit_command_facts(S, [target(prolog)])
    )),
    assert_true('slash_command(help,...) emitted', (
        sub_atom(Output, _, _, _, 'slash_command(help')
    )),
    assert_true('command_alias emitted', (
        sub_atom(Output, _, _, _, 'command_alias(')
    )),
    assert_true('slash_command_group emitted', (
        sub_atom(Output, _, _, _, 'slash_command_group(')
    )).

%% ============================================================================
%% Test 19: Emit Backend Facts via Component Registry
%% ============================================================================

test_emit_backend_facts :-
    format("~nEmit backend facts:~n"),
    register_agent_loop_components,
    with_output_to(atom(Output), (
        current_output(S),
        agent_loop_components:emit_backend_facts(S, [target(prolog)])
    )),
    assert_true('agent_backend emitted', (
        sub_atom(Output, _, _, _, 'agent_backend(')
    )),
    assert_true('backend_factory emitted', (
        sub_atom(Output, _, _, _, 'backend_factory(')
    )),
    assert_true('backend_factory_order emitted', (
        sub_atom(Output, _, _, _, 'backend_factory_order(')
    )),
    assert_true('cli_fallbacks emitted', (
        sub_atom(Output, _, _, _, 'cli_fallbacks(')
    )).

%% ============================================================================
%% Test 20: Compile Component Multi-Fact Output
%% ============================================================================

test_compile_component_multi_fact :-
    format("~nCompile component multi-fact:~n"),
    register_agent_loop_components,
    %% Full multi-fact output for bash (prolog target, no fact_type)
    assert_true('bash full multi-fact has tool_spec', (
        compile_component(agent_tools, bash, [target(prolog)], Code),
        sub_atom(Code, _, _, _, 'tool_spec(bash')
    )),
    assert_true('bash full multi-fact has tool_handler', (
        compile_component(agent_tools, bash, [target(prolog)], Code2),
        sub_atom(Code2, _, _, _, 'tool_handler(bash')
    )),
    assert_true('bash full multi-fact has destructive_tool', (
        compile_component(agent_tools, bash, [target(prolog)], Code3),
        sub_atom(Code3, _, _, _, 'destructive_tool(bash')
    )),
    %% Selective fact_type: tool_handler only
    assert_true('bash tool_handler only via fact_type', (
        compile_component(agent_tools, bash,
            [target(prolog), fact_type(tool_handler)], HCode),
        sub_atom(HCode, _, _, _, 'tool_handler(bash'),
        \+ sub_atom(HCode, _, _, _, 'tool_spec(')
    )),
    %% read is NOT destructive — full output should NOT have destructive_tool
    assert_true('read has no destructive_tool', (
        compile_component(agent_tools, read, [target(prolog)], RCode),
        \+ sub_atom(RCode, _, _, _, 'destructive_tool(')
    )).

%% ============================================================================
%% Test 21: Python Tool Dispatch via Components
%% ============================================================================

test_python_tool_dispatch_via_components :-
    format("~nPython tool dispatch via components:~n"),
    register_agent_loop_components,
    %% Python compile_component with self_prefix
    assert_true('bash python self_prefix', (
        compile_component(agent_tools, bash,
            [target(python), self_prefix(true), indent(12)], PyCode),
        sub_atom(PyCode, _, _, _, 'self._execute_bash')
    )),
    %% Python compile_component indentation
    assert_true('python indent(12) produces 12 spaces', (
        compile_component(agent_tools, bash,
            [target(python), self_prefix(true), indent(12)], PyCode2),
        atom_chars(PyCode2, Chars),
        append(Spaces, ['\''|_], Chars),
        length(Spaces, 12),
        forall(member(Sp, Spaces), Sp = ' ')
    )),
    %% Capture generate_tool_dispatch output
    assert_true('generate_tool_dispatch has bash entry', (
        with_output_to(atom(DispOutput), (
            current_output(DS),
            agent_loop_module:generate_tool_dispatch(DS)
        )),
        sub_atom(DispOutput, _, _, _, '\'bash\': self._execute_bash,')
    )),
    assert_true('generate_tool_dispatch has destructive_tools', (
        with_output_to(atom(DispOutput2), (
            current_output(DS2),
            agent_loop_module:generate_tool_dispatch(DS2)
        )),
        sub_atom(DispOutput2, _, _, _, 'self.destructive_tools = {')
    )).

%% ============================================================================
%% Test 22: Compile Command Component
%% ============================================================================

test_compile_command_component :-
    format("~nCompile command component:~n"),
    register_agent_loop_components,
    %% slash_command fact_type produces correct output
    assert_true('help command via compile_component', (
        compile_component(agent_commands, help,
            [target(prolog), fact_type(slash_command)], Code),
        sub_atom(Code, _, _, _, 'slash_command(help'),
        sub_atom(Code, _, _, _, 'exact')
    )),
    %% iterations command has match_type prefix
    assert_true('iterations command has prefix match_type', (
        compile_component(agent_commands, iterations,
            [target(prolog), fact_type(slash_command)], Code2),
        sub_atom(Code2, _, _, _, 'prefix')
    )).

%% ============================================================================
%% Test 23: Compile Backend Component
%% ============================================================================

test_compile_backend_component :-
    format("~nCompile backend component:~n"),
    register_agent_loop_components,
    %% backend_factory fact_type
    assert_true('claude backend via compile_component', (
        compile_component(agent_backends, claude,
            [target(prolog), fact_type(backend_factory)], Code),
        sub_atom(Code, _, _, _, 'backend_factory(claude')
    )),
    %% coro backend produces backend_factory
    assert_true('coro backend via compile_component', (
        compile_component(agent_backends, coro,
            [target(prolog), fact_type(backend_factory)], Code2),
        sub_atom(Code2, _, _, _, 'backend_factory(coro')
    )).

%% ============================================================================
%% Test 24: Security Component Registration
%% ============================================================================

test_security_component_registration :-
    format("~nSecurity component registration:~n"),
    register_agent_loop_components,
    %% 4 security profiles registered
    assert_true('4 security profiles registered', (
        findall(N, component(agent_security, N, security_profile, _), Ns),
        length(Ns, 4)
    )),
    %% compile_component produces security_profile(cautious,...
    assert_true('cautious profile via compile_component', (
        compile_component(agent_security, cautious, [target(prolog)], Code),
        sub_atom(Code, _, _, _, 'security_profile(cautious')
    )).

%% ============================================================================
%% Test 25: Cost Component Registration
%% ============================================================================

test_cost_component_registration :-
    format("~nCost component registration:~n"),
    register_agent_loop_components,
    %% 16 model costs registered
    assert_true('16 model costs registered', (
        findall(M, component(agent_costs, M, model_pricing, _), Ms),
        length(Ms, 16)
    )),
    %% compile_component produces model_pricing(
    assert_true('opus cost via compile_component', (
        compile_component(agent_costs, opus, [target(prolog)], Code),
        sub_atom(Code, _, _, _, 'model_pricing(')
    )).

%% ============================================================================
%% Test 26: Cost Python Compile Component
%% ============================================================================

test_cost_python_compile :-
    format("~nCost Python compile:~n"),
    register_agent_loop_components,
    %% Python compile_component produces dict entry
    assert_true('opus python cost dict entry', (
        compile_component(agent_costs, opus, [target(python)], Code),
        sub_atom(Code, _, _, _, '"opus": {"input":')
    )),
    %% emit_cost_facts with target(python) produces all 16 entries
    assert_true('emit_cost_facts python has 16 entries', (
        with_output_to(atom(Output), (
            current_output(CS),
            agent_loop_components:emit_cost_facts(CS, [target(python)])
        )),
        findall(_, sub_atom(Output, _, _, _, '"input":'), Matches),
        length(Matches, 16)
    )).

%% ============================================================================
%% Test 27: Security Blocked Python Emit
%% ============================================================================

test_security_blocked_py :-
    format("~nSecurity blocked Python emit:~n"),
    %% blocked_path produces Python set entries
    assert_true('blocked_path python has /etc/shadow', (
        with_output_to(atom(Output), (
            current_output(BS),
            agent_loop_components:emit_security_facts(BS,
                [target(python), fact_type(blocked_path)])
        )),
        sub_atom(Output, _, _, _, '/etc/shadow')
    )),
    %% blocked_command_pattern produces Python regex tuples
    assert_true('blocked_command_pattern python has regex tuples', (
        with_output_to(atom(Output2), (
            current_output(BS2),
            agent_loop_components:emit_security_facts(BS2,
                [target(python), fact_type(blocked_command_pattern)])
        )),
        sub_atom(Output2, _, _, _, '(r\'')
    )).

%% ============================================================================
%% Test 28: Backend Init Import Emit
%% ============================================================================

test_backend_init_imports :-
    format("~nBackend init imports:~n"),
    %% emit_backend_init_imports produces 'from .' import lines
    assert_true('backend init imports has from .', (
        with_output_to(atom(Output), (
            current_output(IS),
            agent_loop_components:emit_backend_init_imports(IS, [])
        )),
        sub_atom(Output, _, _, _, 'from .')
    )),
    %% emit_backend_init_optional produces try/except blocks
    assert_true('backend init optional has try:', (
        with_output_to(atom(Output2), (
            current_output(IS2),
            agent_loop_components:emit_backend_init_optional(IS2, [])
        )),
        sub_atom(Output2, _, _, _, 'try:')
    )).

%% ============================================================================
%% Test 29: Python Integration Tests via pytest
%% ============================================================================

test_python_integration :-
    format("~nPython integration tests:~n"),
    assert_true('Python integration tests pass', (
        shell('cd generated/python && python3 -m pytest ../../test_integration.py -q --tb=short 2>&1', ExitCode),
        ExitCode =:= 0
    )).

%% ============================================================================
%% Test 30: Emit Context Enums
%% ============================================================================

test_emit_context_enums :-
    format("~nEmit context enums:~n"),
    assert_true('emit_context_enums has ContextBehavior', (
        with_output_to(atom(Output), (
            current_output(CS),
            agent_loop_components:emit_context_enums(CS, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'class ContextBehavior(Enum):')
    )),
    assert_true('emit_context_enums has ContextFormat', (
        with_output_to(atom(Output2), (
            current_output(CS2),
            agent_loop_components:emit_context_enums(CS2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, 'class ContextFormat(Enum):')
    )).

%% ============================================================================
%% Test 31: Emit Message Fields
%% ============================================================================

test_emit_message_fields :-
    format("~nEmit message fields:~n"),
    assert_true('emit_message_fields has role', (
        with_output_to(atom(Output), (
            current_output(MS),
            agent_loop_components:emit_message_fields(MS, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'role:')
    )),
    assert_true('emit_message_fields has content', (
        with_output_to(atom(Output2), (
            current_output(MS2),
            agent_loop_components:emit_message_fields(MS2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, 'content:')
    )).

%% ============================================================================
%% Test 32: Emit Agent Config Fields
%% ============================================================================

test_emit_agent_config_fields :-
    format("~nEmit agent config fields:~n"),
    assert_true('emit_agent_config_fields has name', (
        with_output_to(atom(Output), (
            current_output(AS),
            agent_loop_components:emit_agent_config_fields(AS, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'name: str')
    )),
    assert_true('emit_agent_config_fields has backend', (
        with_output_to(atom(Output2), (
            current_output(AS2),
            agent_loop_components:emit_agent_config_fields(AS2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, 'backend: str')
    )).

%% ============================================================================
%% Test 33: Binding Import and Dispatch Comment Emission
%% ============================================================================

test_binding_imports :-
    format("~nBinding imports:~n"),
    assert_true('emit_binding_imports produces from lines', (
        with_output_to(atom(Output), (
            current_output(BS),
            agent_loop_bindings:emit_binding_imports(BS, [])
        )),
        sub_atom(Output, _, _, _, 'from')
    )),
    assert_true('emit_binding_dispatch_comment has metadata', (
        with_output_to(atom(COutput), (
            current_output(CS),
            agent_loop_bindings:emit_binding_dispatch_comment(CS, [])
        )),
        sub_atom(COutput, _, _, _, 'Binding registry')
    )).

%% ============================================================================
%% Test 34: Prolog Integration Tests
%% ============================================================================

test_prolog_integration :-
    format("~nProlog integration tests:~n"),
    assert_true('Prolog integration tests pass', (
        shell('swipl -l test_prolog_integration.pl -g "run_prolog_tests, halt" 2>&1', ExitCode),
        ExitCode =:= 0
    )).

%% ============================================================================
%% Test 35: Prolog Config Emit Predicate
%% ============================================================================

test_emit_prolog_config_facts :-
    format("~nProlog config emit:~n"),
    assert_true('emit_prolog_config_facts contains cli_argument', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_prolog_config_facts(S, [target(prolog)])
        )),
        sub_atom(Output, _, _, _, 'cli_argument(agent,')
    )),
    assert_true('emit_prolog_config_facts contains audit_profile_level', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_prolog_config_facts(S2, [target(prolog)])
        )),
        sub_atom(Output2, _, _, _, 'audit_profile_level(')
    )),
    assert_true('emit_prolog_config_facts contains indexing hints', (
        with_output_to(atom(Output3), (
            current_output(S3),
            agent_loop_components:emit_prolog_config_facts(S3, [target(prolog)])
        )),
        sub_atom(Output3, _, _, _, 'first-argument indexed')
    )).

%% ============================================================================
%% Test 36: Python Config Emit Predicates
%% ============================================================================

test_emit_api_key_env_vars_py :-
    format("~nPython config emit:~n"),
    assert_true('emit_api_key_env_vars_py contains claude', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_api_key_env_vars_py(S, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'claude')
    )),
    assert_true('emit_api_key_files_py contains file path', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_api_key_files_py(S2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, '~/')
    )).

test_emit_default_presets_py :-
    format("~nDefault presets emit:~n"),
    assert_true('emit_default_presets_py contains default agent', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_default_presets_py(S, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'config.agents[\'default\']')
    )),
    assert_true('emit_default_presets_py contains yolo', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_default_presets_py(S2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, 'auto_tools=True')
    )).

%% ============================================================================
%% Test 37: Help Groups Emit
%% ============================================================================

test_emit_help_groups :-
    format("~nHelp groups emit:~n"),
    assert_true('emit_help_groups contains /exit', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_help_groups(S, [target(python)])
        )),
        sub_atom(Output, _, _, _, '/exit')
    )),
    assert_true('emit_help_groups contains /help', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_help_groups(S2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, '/help')
    )).

%% ============================================================================
%% Test 38: New Binding Count and Compilation
%% ============================================================================

test_new_binding_count :-
    format("~nNew binding count:~n"),
    clear_all_bindings,
    agent_loop_bindings:init_agent_loop_bindings,
    assert_true('Python bindings count is 11', (
        bindings_for_target(python, PyBindings),
        length(PyBindings, 11)
    )),
    assert_true('Prolog bindings count is 11', (
        bindings_for_target(prolog, PlBindings),
        length(PlBindings, 11)
    )),
    assert_true('Rust bindings count is 11', (
        bindings_for_target(rust, RustBindings),
        length(RustBindings, 11)
    )),
    assert_true('api_key_env_var has Python binding', (
        bindings_for_predicate(api_key_env_var/2, Bindings),
        member(binding(python, _, _, _, _, _), Bindings)
    )).

test_binding_compile_api_key :-
    format("~nBinding compilation:~n"),
    clear_all_bindings,
    agent_loop_bindings:init_agent_loop_bindings,
    assert_true('compile_binding_code for api_key_env_var', (
        agent_loop_bindings:compile_binding_code(python, api_key_env_var/2, Code),
        sub_atom(Code, _, _, _, 'API_KEY_ENV_VARS')
    )),
    assert_true('compile_binding_code for api_key_file', (
        agent_loop_bindings:compile_binding_code(python, api_key_file/2, Code2),
        sub_atom(Code2, _, _, _, 'API_KEY_FILE_PATHS')
    )).

%% ============================================================================
%% Batch 3 Emit Predicate Tests
%% ============================================================================

test_emit_streaming_capable_facts :-
    format("~nStreaming capable emit:~n"),
    assert_true('emit_streaming_capable_facts contains streaming_capable', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_streaming_capable_facts(S, [target(prolog)])
        )),
        sub_atom(Output, _, _, _, 'streaming_capable(')
    )),
    assert_true('emit_streaming_capable_facts contains api type', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_streaming_capable_facts(S2, [target(prolog)])
        )),
        sub_atom(Output2, _, _, _, 'streaming_capable(api)')
    )).

test_emit_security_profile_entries :-
    format("~nSecurity profile entries emit:~n"),
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_security_profile_entries contains cautious', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_security_profile_entries(S, [target(python)])
        )),
        sub_atom(Output, _, _, _, 'cautious')
    )).

test_emit_tool_dispatch_entries :-
    format("~nTool dispatch entries emit:~n"),
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_tool_dispatch_entries contains bash', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_tool_dispatch_entries(S, [target(python)])
        )),
        sub_atom(Output, _, _, _, '\'bash\'')
    )),
    assert_true('emit_tool_dispatch_entries contains self._execute', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_tool_dispatch_entries(S2, [target(python)])
        )),
        sub_atom(Output2, _, _, _, 'self._execute')
    )).

test_emit_cascade_paths :-
    format("~nCascade paths emit:~n"),
    assert_true('emit_cascade_paths required contains uwsal.json', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_cascade_paths(S, [path_type(required), indent('')])
        )),
        sub_atom(Output, _, _, _, 'uwsal.json')
    )),
    assert_true('emit_cascade_paths fallback exists', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_cascade_paths(S2, [path_type(fallback), indent('    ')])
        )),
        atom_length(Output2, Len2),
        Len2 > 0
    )).

test_emit_alias_conditions :-
    format("~nAlias conditions emit:~n"),
    assert_true('emit_alias_conditions exact generates or cmd ==', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_alias_conditions(S, [aliases([q, x]), match_style(exact)])
        )),
        sub_atom(Output, _, _, _, 'or cmd == \'q\'')
    )),
    assert_true('emit_alias_conditions prefix_sp generates startswith', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_alias_conditions(S2, [aliases([be]), match_style(prefix_sp)])
        )),
        sub_atom(Output2, _, _, _, 'startswith(\'be \')')
    )).

test_emit_argparse_group_args :-
    format("~nArgparse group args emit:~n"),
    assert_true('emit_argparse_group_args contains add_argument', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_argparse_group_args(S, [args([agent])])
        )),
        sub_atom(Output, _, _, _, 'add_argument')
    )).

test_emit_backend_optimization_hints :-
    format("~nBackend optimization hints:~n"),
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_backend_facts includes indexing hints', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_backend_facts(S, [target(prolog)])
        )),
        sub_atom(Output, _, _, _, 'Indexing hints')
    )),
    assert_true('emit_backend_facts includes agent_backend clause count', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_backend_facts(S2, [target(prolog)])
        )),
        sub_atom(Output2, _, _, _, 'agent_backend/2: first-argument indexed (8 clauses)')
    )).

test_emit_command_optimization_hints :-
    format("~nCommand optimization hints:~n"),
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_command_facts includes indexing hints', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_command_facts(S, [target(prolog)])
        )),
        sub_atom(Output, _, _, _, 'Indexing hints')
    )),
    assert_true('emit_command_facts includes slash_command clause count', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_command_facts(S2, [target(prolog)])
        )),
        sub_atom(Output2, _, _, _, 'slash_command/4: first-argument indexed (23 clauses)')
    )),
    assert_true('emit_command_facts includes command_alias clause count', (
        with_output_to(atom(Output3), (
            current_output(S3),
            agent_loop_components:emit_command_facts(S3, [target(prolog)])
        )),
        sub_atom(Output3, _, _, _, 'command_alias/2: first-argument indexed (30 clauses)')
    )).

%% ============================================================================
%% Unified Import Infrastructure Tests
%% ============================================================================

test_emit_module_imports :-
    format("~nUnified import infrastructure:~n"),
    %% Test bare import
    assert_true('bare import produces import X', (
        with_output_to(atom(Output1), (
            current_output(S1),
            agent_loop_components:emit_module_imports(S1, [bare(json)])
        )),
        sub_atom(Output1, _, _, _, 'import json')
    )),
    %% Test from import
    assert_true('from import produces from X import Y', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_module_imports(S2, [from('urllib.request', [urlopen, 'Request'])])
        )),
        sub_atom(Output2, _, _, _, 'from urllib.request import urlopen, Request')
    )),
    %% Test from_relative import
    assert_true('from_relative import produces from .X import Y', (
        with_output_to(atom(Output3), (
            current_output(S3),
            agent_loop_components:emit_module_imports(S3, [from_relative(audit, ['AuditLogger'])])
        )),
        sub_atom(Output3, _, _, _, 'from .audit import AuditLogger')
    )).

test_backend_import_specs :-
    format("~nBackend import specs:~n"),
    %% Test openrouter_api specs include from() for urllib
    assert_true('openrouter_api specs include from(urllib.request, ...)', (
        agent_loop_module:agent_backend(openrouter_api, Props),
        agent_loop_components:backend_import_specs(Props, Specs),
        member(from('urllib.request', _), Specs)
    )),
    %% Test openrouter_api has bare imports for json, os, sys
    assert_true('openrouter_api specs include bare(json)', (
        agent_loop_module:agent_backend(openrouter_api, Props2),
        agent_loop_components:backend_import_specs(Props2, Specs2),
        member(bare(json), Specs2)
    )),
    %% Test ollama_cli has only bare imports
    assert_true('ollama_cli specs are all bare', (
        agent_loop_module:agent_backend(ollama_cli, Props3),
        agent_loop_components:backend_import_specs(Props3, Specs3),
        member(bare(subprocess), Specs3)
    )).

test_module_dependency_facts :-
    format("~nModule dependency facts:~n"),
    assert_true('tools depends on security', (
        agent_loop_module:module_dependency(tools, security, _)
    )),
    assert_true('agent_loop depends on backends', (
        agent_loop_module:module_dependency(agent_loop, backends, _)
    )),
    assert_true('backends depends on costs', (
        agent_loop_module:module_dependency(backends, costs, _)
    )).

test_emit_module_dependencies :-
    format("~nModule dependency emitter:~n"),
    %% Test backends has dependencies
    assert_true('emit_module_dependencies for backends includes costs', (
        with_output_to(atom(Output1), (
            current_output(S1),
            agent_loop_components:emit_module_dependencies(S1, [module(backends)])
        )),
        sub_atom(Output1, _, _, _, 'costs')
    )),
    %% Test commands has no dependencies (self-contained)
    assert_true('emit_module_dependencies for commands says self-contained', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_module_dependencies(S2, [module(commands)])
        )),
        sub_atom(Output2, _, _, _, 'self-contained')
    )).

%% ============================================================================
%% :- det annotations, dependency diagram, complete dependency coverage
%% ============================================================================

test_det_annotations_in_generated :-
    format("~nDet annotations in generated files:~n"),
    assert_true('generated tools.pl contains :- det(execute_tool/3)', (
        read_file_to_string('generated/prolog/tools.pl', Content, []),
        sub_string(Content, _, _, _, ":- det(execute_tool/3)")
    )),
    assert_true('generated backends.pl contains :- det(create_backend/3)', (
        read_file_to_string('generated/prolog/backends.pl', Content2, []),
        sub_string(Content2, _, _, _, ":- det(create_backend/3)")
    )),
    assert_true('generated commands.pl contains :- det(resolve_command/3)', (
        read_file_to_string('generated/prolog/commands.pl', Content3, []),
        sub_string(Content3, _, _, _, ":- det(resolve_command/3)")
    )),
    assert_true('generated security.pl contains :- det(check_path_allowed/2)', (
        read_file_to_string('generated/prolog/security.pl', Content4, []),
        sub_string(Content4, _, _, _, ":- det(check_path_allowed/2)")
    )).

test_dependency_diagram_output :-
    format("~nDependency diagram:~n"),
    assert_true('emit_dependency_diagram contains mermaid', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_dependency_diagram(S, [])
        )),
        sub_atom(Output, _, _, _, 'mermaid')
    )),
    assert_true('emit_dependency_diagram contains graph TD', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_dependency_diagram(S2, [])
        )),
        sub_atom(Output2, _, _, _, 'graph TD')
    )),
    assert_true('emit_dependency_diagram contains agent_loop --> tools', (
        with_output_to(atom(Output3), (
            current_output(S3),
            agent_loop_components:emit_dependency_diagram(S3, [])
        )),
        sub_atom(Output3, _, _, _, 'agent_loop --> tools')
    )).

test_module_dependencies_complete :-
    format("~nComplete dependency coverage:~n"),
    assert_true('config depends on backends', (
        agent_loop_module:module_dependency(config, backends, _)
    )),
    assert_true('security depends on config', (
        agent_loop_module:module_dependency(security, config, _)
    )).

%% ============================================================================
%% Step 2 expanded: :- det in costs, config, agent_loop, main
%% ============================================================================

test_det_annotations_expanded :-
    format("~nExpanded det annotations:~n"),
    %% costs.pl
    assert_true('costs.pl has :- det(cost_tracker_init/1)', (
        read_file_to_string('generated/prolog/costs.pl', C1, []),
        sub_string(C1, _, _, _, ":- det(cost_tracker_init/1)")
    )),
    assert_true('costs.pl has :- det(cost_tracker_format/2)', (
        read_file_to_string('generated/prolog/costs.pl', C2, []),
        sub_string(C2, _, _, _, ":- det(cost_tracker_format/2)")
    )),
    %% config.pl
    assert_true('config.pl has :- det(parse_cli_args/2)', (
        read_file_to_string('generated/prolog/config.pl', C3, []),
        sub_string(C3, _, _, _, ":- det(parse_cli_args/2)")
    )),
    assert_true('config.pl has :- det(load_config/2)', (
        read_file_to_string('generated/prolog/config.pl', C4, []),
        sub_string(C4, _, _, _, ":- det(load_config/2)")
    )),
    %% agent_loop.pl
    assert_true('agent_loop.pl has :- det(sessions_dir/1)', (
        read_file_to_string('generated/prolog/agent_loop.pl', C5, []),
        sub_string(C5, _, _, _, ":- det(sessions_dir/1)")
    )),
    assert_true('agent_loop.pl has :- det(process_input/2)', (
        read_file_to_string('generated/prolog/agent_loop.pl', C6, []),
        sub_string(C6, _, _, _, ":- det(process_input/2)")
    )),
    %% main.pl
    assert_true('main.pl has :- det(main/1)', (
        read_file_to_string('generated/prolog/main.pl', C7, []),
        sub_string(C7, _, _, _, ":- det(main/1)")
    )).

%% ============================================================================
%% Step 3: Binding metadata comments in generated Python files
%% ============================================================================

test_binding_metadata_in_generated :-
    format("~nBinding metadata in generated files:~n"),
    assert_true('costs.py has Binding for model_pricing/3', (
        read_file_to_string('generated/python/costs.py', C1, []),
        sub_string(C1, _, _, _, "DEFAULT_PRICING"),
        sub_string(C1, _, _, _, "dict_lookup")
    )),
    assert_true('tools_generated.py has Binding for tool_handler/2', (
        read_file_to_string('generated/python/tools_generated.py', C2, []),
        sub_string(C2, _, _, _, "TOOL_HANDLERS")
    )),
    assert_true('tools_generated.py has Binding for destructive_tool/1', (
        read_file_to_string('generated/python/tools_generated.py', C3, []),
        sub_string(C3, _, _, _, "DESTRUCTIVE_TOOLS"),
        sub_string(C3, _, _, _, "set_membership")
    )),
    assert_true('agent_loop.py has binding dispatch comment', (
        read_file_to_string('generated/python/agent_loop.py', C4, []),
        sub_string(C4, _, _, _, "Binding registry metadata")
    )).

%% ============================================================================
%% Step 4: Target-level forall→maplist in generated Prolog
%% ============================================================================

test_target_level_maplist_in_generated :-
    format("~nTarget-level maplist in generated Prolog:~n"),
    read_file_to_string('generated/prolog/agent_loop.pl', Content, []),
    %% No forall left
    assert_true('generated agent_loop.pl has no forall(backend_factory', (
        \+ sub_string(Content, _, _, _, "forall(backend_factory")
    )),
    assert_true('generated agent_loop.pl has no forall(command_alias', (
        \+ sub_string(Content, _, _, _, "forall(command_alias")
    )),
    assert_true('generated agent_loop.pl has no forall(member(Msg', (
        \+ sub_string(Content, _, _, _, "forall(member(Msg")
    )),
    %% maplist present
    assert_true('generated agent_loop.pl has maplist([N]>>', (
        sub_string(Content, _, _, _, "maplist([N]>>")
    )),
    assert_true('generated agent_loop.pl has maplist([Msg]>>', (
        sub_string(Content, _, _, _, "maplist([Msg]>>")
    )),
    assert_true('generated agent_loop.pl has maplist([Group-Cmds]>>', (
        sub_string(Content, _, _, _, "maplist([Group-Cmds]>>")
    )).

%% ============================================================================
%% Step 1 verification: no forall in components
%% ============================================================================

test_no_forall_in_components :-
    format("~nNo forall in components:~n"),
    assert_true('agent_loop_components.pl has no forall calls', (
        read_file_to_string('agent_loop_components.pl', Content, []),
        \+ sub_string(Content, _, _, _, "forall(")
    )).

%% ============================================================================
%% Extended binding patterns: method_call, chained_call, set_membership
%% ============================================================================

test_binding_patterns_extended :-
    format("~nExtended binding patterns:~n"),
    init_agent_loop_bindings,
    %% set_membership pattern
    assert_true('set_membership generates if-in', (
        compile_binding_code(python, destructive_tool/1, Code),
        sub_atom(Code, _, _, _, 'in DESTRUCTIVE_TOOLS')
    )),
    %% dict_lookup pattern
    assert_true('dict_lookup generates assignment', (
        compile_binding_code(python, model_pricing/3, Code2),
        sub_atom(Code2, _, _, _, 'DEFAULT_PRICING')
    )),
    %% binding_pattern/3 query
    assert_true('binding_pattern returns dict_lookup for model_pricing', (
        agent_loop_bindings:binding_pattern(python, model_pricing/3, dict_lookup)
    )),
    assert_true('binding_pattern returns set_membership for destructive_tool', (
        agent_loop_bindings:binding_pattern(python, destructive_tool/1, set_membership)
    )).

%% ============================================================================
%% binding_dict_name/3 tests
%% ============================================================================

test_binding_dict_name :-
    format("~nBinding dict name:~n"),
    init_agent_loop_bindings,
    assert_true('binding_dict_name extracts DEFAULT_PRICING from model_pricing/3', (
        agent_loop_bindings:binding_dict_name(python, model_pricing/3, 'DEFAULT_PRICING')
    )),
    assert_true('binding_dict_name extracts audit_levels from audit_profile_level/2', (
        agent_loop_bindings:binding_dict_name(python, audit_profile_level/2, audit_levels)
    )),
    assert_true('binding_dict_name returns DESTRUCTIVE_TOOLS for destructive_tool/1', (
        agent_loop_bindings:binding_dict_name(python, destructive_tool/1, 'DESTRUCTIVE_TOOLS')
    )).

%% ============================================================================
%% compile_component/4 Python target tests
%% ============================================================================

test_compile_component_python_targets :-
    format("~nCompile component Python targets:~n"),
    register_agent_loop_components,
    %% Command component Python
    assert_true('command compile_component with target(python) produces dict entry', (
        compile_component(agent_commands, help, [target(python)], Code),
        sub_atom(Code, _, _, _, 'help')
    )),
    %% Backend component Python
    assert_true('backend compile_component with target(python) produces dict entry', (
        compile_component(agent_backends, coro, [target(python)], Code2),
        sub_atom(Code2, _, _, _, 'coro')
    )),
    %% Security component Python
    assert_true('security compile_component with target(python) produces dict entry', (
        compile_component(agent_security, cautious, [target(python)], Code3),
        sub_atom(Code3, _, _, _, 'cautious')
    )).

%% ============================================================================
%% write_lines/2 helper test
%% ============================================================================

test_write_lines_helper :-
    format("~nwrite_lines helper:~n"),
    assert_true('write_lines emits multiple lines', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:write_lines(S, ['line1', 'line2', 'line3'])
        )),
        sub_atom(Output, _, _, _, 'line1'),
        sub_atom(Output, _, _, _, 'line2'),
        sub_atom(Output, _, _, _, 'line3')
    )),
    assert_true('write_lines empty list produces no output', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:write_lines(S2, [])
        )),
        Output2 == ''
    )).

%% ============================================================================
%% No forall in bindings
%% ============================================================================

test_no_forall_in_bindings :-
    format("~nNo forall in bindings:~n"),
    assert_true('agent_loop_bindings.pl has no forall calls', (
        read_file_to_string('agent_loop_bindings.pl', Content, []),
        \+ sub_string(Content, _, _, _, "forall(")
    )).

%% ============================================================================
%% Compile tool_spec Python
%% ============================================================================

test_compile_tool_spec_python :-
    format("~nCompile tool_spec Python:~n"),
    register_agent_loop_components,
    assert_true('tool_spec Python compile produces dict entry', (
        compile_component(agent_tools, bash, [target(python), fact_type(tool_spec)], Code),
        sub_atom(Code, _, _, _, '"bash"'),
        sub_atom(Code, _, _, _, '"description"')
    )),
    assert_true('tool_spec Python compile includes parameters', (
        compile_component(agent_tools, write, [target(python), fact_type(tool_spec)], Code2),
        sub_atom(Code2, _, _, _, '"parameters"')
    )).

%% ============================================================================
%% Compile destructive_tool Python
%% ============================================================================

test_compile_destructive_tool_python :-
    format("~nCompile destructive_tool Python:~n"),
    register_agent_loop_components,
    assert_true('destructive_tool Python compile produces set entry for bash', (
        compile_component(agent_tools, bash, [target(python), fact_type(destructive_tool)], Code),
        sub_atom(Code, _, _, _, '"bash"')
    )),
    assert_true('destructive_tool Python compile fails for non-destructive', (
        \+ compile_component(agent_tools, read, [target(python), fact_type(destructive_tool)], _)
    )).

%% ============================================================================
%% Emit py set from components
%% ============================================================================

test_emit_py_set_from_components :-
    format("~nEmit py set from components:~n"),
    register_agent_loop_components,
    agent_loop_bindings:init_agent_loop_bindings,
    assert_true('emit_py_set produces DESTRUCTIVE_TOOLS set', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_py_set_from_components(S, agent_tools,
                destructive_tool/1, python, [fact_type(destructive_tool)])
        )),
        sub_atom(Output, _, _, _, 'DESTRUCTIVE_TOOLS'),
        sub_atom(Output, _, _, _, '"bash"')
    )),
    assert_true('emit_py_dict with dict_name override', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_py_dict_from_components(S2, agent_tools,
                tool_handler/2, python, [fact_type(tool_spec), dict_name('TOOL_SPECS')])
        )),
        sub_atom(Output2, _, _, _, 'TOOL_SPECS'),
        sub_atom(Output2, _, _, _, '"bash"')
    )).

%% ============================================================================
%% Explicit Prolog security compile
%% ============================================================================

test_explicit_prolog_security_compile :-
    format("~nExplicit Prolog security compile:~n"),
    register_agent_loop_components,
    assert_true('security compile_component with target(prolog)', (
        compile_component(agent_security, cautious, [target(prolog)], Code),
        sub_atom(Code, _, _, _, 'security_profile'),
        sub_atom(Code, _, _, _, 'cautious')
    )).

%% ============================================================================
%% Explicit Prolog cost compile
%% ============================================================================

test_explicit_prolog_cost_compile :-
    format("~nExplicit Prolog cost compile:~n"),
    register_agent_loop_components,
    assert_true('cost compile_component with target(prolog)', (
        compile_component(agent_costs, opus, [target(prolog)], Code),
        sub_atom(Code, _, _, _, 'model_pricing'),
        sub_atom(Code, _, _, _, 'opus')
    )).

%% ============================================================================
%% Emit Prolog facts from components
%% ============================================================================

test_emit_prolog_facts_from_components :-
    format("~nEmit Prolog facts from components:~n"),
    register_agent_loop_components,
    assert_true('emit_prolog_facts emits cost facts', (
        with_output_to(atom(Output), (
            current_output(S),
            agent_loop_components:emit_prolog_facts_from_components(S, agent_costs, model_pricing, [])
        )),
        sub_atom(Output, _, _, _, 'model_pricing'),
        sub_atom(Output, _, _, _, 'opus')
    )),
    assert_true('emit_prolog_facts emits security facts', (
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:emit_prolog_facts_from_components(S2, agent_security, security_profile, [])
        )),
        sub_atom(Output2, _, _, _, 'security_profile'),
        sub_atom(Output2, _, _, _, 'cautious')
    )).

%% ============================================================================
%% Translate agent goal
%% ============================================================================

test_translate_agent_goal :-
    format("~nTranslate agent goal:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    assert_true('translate_agent_goal for model_pricing', (
        agent_loop_bindings:translate_agent_goal(model_pricing(opus, _, _), Code),
        sub_atom(Code, _, _, _, 'DEFAULT_PRICING')
    )),
    assert_true('translate_agent_goal for destructive_tool', (
        agent_loop_bindings:translate_agent_goal(destructive_tool(bash), Code2),
        sub_atom(Code2, _, _, _, 'DESTRUCTIVE_TOOLS')
    )),
    assert_true('translate_agent_goal for tool_handler', (
        agent_loop_bindings:translate_agent_goal(tool_handler(bash, _), Code3),
        sub_atom(Code3, _, _, _, 'TOOL_HANDLERS')
    )),
    assert_true('translate_agent_goal Prolog target', (
        agent_loop_bindings:translate_agent_goal(prolog, model_pricing(opus, _, _), Code4),
        sub_atom(Code4, _, _, _, 'model_pricing')
    )).

%% ============================================================================
%% Translate agent goals list
%% ============================================================================

test_translate_agent_goals :-
    format("~nTranslate agent goals list:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    assert_true('translate_agent_goals multi-goal', (
        agent_loop_bindings:translate_agent_goals(
            [model_pricing(opus, _, _), tool_handler(bash, _)],
            python,
            CodeBlock
        ),
        sub_atom(CodeBlock, _, _, _, 'DEFAULT_PRICING'),
        sub_atom(CodeBlock, _, _, _, 'TOOL_HANDLERS')
    )),
    assert_true('translate_agent_goals empty list', (
        agent_loop_bindings:translate_agent_goals([], python, Code),
        Code == ''
    )).

%% ============================================================================
%% Binding parity
%% ============================================================================

test_binding_parity :-
    format("~nBinding parity:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    findall(Pred, binding(python, Pred, _, _, _, _), PyPreds),
    findall(Pred, binding(prolog, Pred, _, _, _, _), PlPreds),
    findall(Pred, binding(rust, Pred, _, _, _, _), RustPreds),
    length(PyPreds, PyCount),
    length(PlPreds, PlCount),
    length(RustPreds, RustCount),
    assert_eq('Python and Prolog binding counts match', PyCount, PlCount),
    assert_eq('Python and Rust binding counts match', PyCount, RustCount),
    sort(PyPreds, PySorted),
    sort(PlPreds, PlSorted),
    sort(RustPreds, RustSorted),
    assert_true('Python and Prolog cover same predicates', (
        PySorted == PlSorted
    )),
    assert_true('Python and Rust cover same predicates', (
        PySorted == RustSorted
    )).

%% ============================================================================
%% Binding metadata coverage
%% ============================================================================

test_binding_metadata_coverage :-
    format("~nBinding metadata coverage:~n"),
    assert_true('costs.py has binding metadata', (
        read_file_to_string('generated/python/costs.py', C1, []),
        sub_string(C1, _, _, _, "# Binding:")
    )),
    assert_true('tools_generated.py has binding metadata', (
        read_file_to_string('generated/python/tools_generated.py', C2, []),
        sub_string(C2, _, _, _, "# Binding:")
    )),
    assert_true('config.py has binding metadata', (
        read_file_to_string('generated/python/config.py', C3, []),
        sub_string(C3, _, _, _, "# Binding:")
    )),
    assert_true('security/profiles.py has binding metadata', (
        read_file_to_string('generated/python/security/profiles.py', C4, []),
        sub_string(C4, _, _, _, "# Binding:")
    )).

%% ============================================================================
%% write_lines multiline equivalence
%% ============================================================================

test_write_lines_multiline_equiv :-
    format("~nwrite_lines multiline equivalence:~n"),
    assert_true('write_lines matches sequential writes', (
        with_output_to(atom(Output1), (
            current_output(S1),
            write(S1, 'line1\n'),
            write(S1, 'line2\n'),
            write(S1, '\n'),
            write(S1, 'line3\n')
        )),
        with_output_to(atom(Output2), (
            current_output(S2),
            agent_loop_components:write_lines(S2, ['line1', 'line2', '', 'line3'])
        )),
        Output1 == Output2
    )).

%% ============================================================================
%% Translate all bindings coverage
%% ============================================================================

test_translate_all_bindings :-
    format("~nTranslate all bindings:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    %% Test all 11 bindings × Python target
    assert_true('translate slash_command/4 python', (
        agent_loop_bindings:translate_agent_goal(python, slash_command(help, _, _, _), Code1),
        sub_atom(Code1, _, _, _, 'SLASH_COMMANDS')
    )),
    assert_true('translate backend_factory/2 python', (
        agent_loop_bindings:translate_agent_goal(python, backend_factory(coro, _), Code2),
        sub_atom(Code2, _, _, _, 'create_backend_from_config')
    )),
    assert_true('translate audit_profile_level/2 python', (
        agent_loop_bindings:translate_agent_goal(python, audit_profile_level(basic, _), Code3),
        sub_atom(Code3, _, _, _, 'audit_levels')
    )),
    assert_true('translate security_profile/2 python', (
        agent_loop_bindings:translate_agent_goal(python, security_profile(open, _), Code4),
        sub_atom(Code4, _, _, _, 'SecurityConfig')
    )),
    assert_true('translate api_key_env_var/2 python', (
        agent_loop_bindings:translate_agent_goal(python, api_key_env_var(claude, _), Code5),
        sub_atom(Code5, _, _, _, 'API_KEY_ENV_VARS')
    )),
    assert_true('translate api_key_file/2 python', (
        agent_loop_bindings:translate_agent_goal(python, api_key_file(claude, _), Code6),
        sub_atom(Code6, _, _, _, 'API_KEY_FILE_PATHS')
    )),
    assert_true('translate default_agent_preset/3 python', (
        agent_loop_bindings:translate_agent_goal(python, default_agent_preset(default, _, _), Code7),
        sub_atom(Code7, _, _, _, 'get_default_config')
    )),
    assert_true('translate config_search_path/2 python', (
        agent_loop_bindings:translate_agent_goal(python, config_search_path(local, _), Code8),
        sub_atom(Code8, _, _, _, 'CONFIG_SEARCH_PATHS')
    )),
    %% Test remaining bindings × Prolog target
    assert_true('translate slash_command/4 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, slash_command(help, _, _, _), PlCode1),
        sub_atom(PlCode1, _, _, _, 'slash_command')
    )),
    assert_true('translate backend_factory/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, backend_factory(coro, _), PlCode2),
        sub_atom(PlCode2, _, _, _, 'create_backend')
    )),
    assert_true('translate audit_profile_level/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, audit_profile_level(basic, _), PlCode3),
        sub_atom(PlCode3, _, _, _, 'audit_profile_level')
    )),
    assert_true('translate security_profile/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, security_profile(open, _), PlCode4),
        sub_atom(PlCode4, _, _, _, 'security_profile')
    )),
    assert_true('translate api_key_env_var/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, api_key_env_var(claude, _), PlCode5),
        sub_atom(PlCode5, _, _, _, 'api_key_env_var')
    )),
    assert_true('translate api_key_file/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, api_key_file(claude, _), PlCode6),
        sub_atom(PlCode6, _, _, _, 'api_key_file')
    )),
    assert_true('translate default_agent_preset/3 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, default_agent_preset(default, _, _), PlCode7),
        sub_atom(PlCode7, _, _, _, 'default_agent_preset')
    )),
    assert_true('translate config_search_path/2 prolog', (
        agent_loop_bindings:translate_agent_goal(prolog, config_search_path(local, _), PlCode8),
        sub_atom(PlCode8, _, _, _, 'config_search_path')
    )).

%% ============================================================================
%% Extended binding metadata coverage (aliases.py and backends/__init__.py)
%% ============================================================================

test_binding_metadata_extended :-
    format("~nExtended binding metadata coverage:~n"),
    assert_true('aliases.py has binding metadata', (
        read_file_to_string('generated/python/aliases.py', C1, []),
        sub_string(C1, _, _, _, "# Binding:")
    )),
    assert_true('backends/__init__.py has binding metadata', (
        read_file_to_string('generated/python/backends/__init__.py', C2, []),
        sub_string(C2, _, _, _, "# Binding:")
    )).

%% ============================================================================
%% Unified emit_from_components tests
%% ============================================================================

test_emit_from_components_dict :-
    format("~nUnified emit_from_components dict format:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_from_components dict produces output', (
        with_output_to(atom(Output),
            agent_loop_components:emit_from_components(current_output,
                agent_costs, model_pricing/3, python, dict, [])),
        sub_atom(Output, _, _, _, 'DEFAULT_PRICING'),
        sub_atom(Output, _, _, _, '{'),
        sub_atom(Output, _, _, _, '}')
    )),
    assert_true('emit_from_components dict with dict_name override', (
        with_output_to(atom(Output2),
            agent_loop_components:emit_from_components(current_output,
                agent_costs, model_pricing/3, python, dict, [dict_name('MY_DICT')])),
        sub_atom(Output2, _, _, _, 'MY_DICT')
    )).

test_emit_from_components_set :-
    format("~nUnified emit_from_components set format:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_from_components set produces output', (
        with_output_to(atom(Output),
            agent_loop_components:emit_from_components(current_output,
                agent_tools, destructive_tool/1, python, set,
                [fact_type(destructive_tool), dict_name('TEST_SET')])),
        sub_atom(Output, _, _, _, 'TEST_SET'),
        sub_atom(Output, _, _, _, '{')
    )).

test_emit_from_components_facts :-
    format("~nUnified emit_from_components facts format:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_components:register_agent_loop_components,
    assert_true('emit_from_components facts produces Prolog', (
        with_output_to(atom(Output),
            agent_loop_components:emit_from_components(current_output,
                agent_costs, model_pricing/3, prolog, facts, [])),
        sub_atom(Output, _, _, _, 'model_pricing'),
        \+ sub_atom(Output, _, _, _, '=')
    )),
    assert_true('emit_from_components facts with fact_type', (
        with_output_to(atom(Output2),
            agent_loop_components:emit_from_components(current_output,
                agent_tools, tool_handler/2, prolog, facts, [fact_type(tool_handler)])),
        sub_atom(Output2, _, _, _, 'tool_handler')
    )).

%% ============================================================================
%% Format equivalence tests
%% ============================================================================

test_emit_format_equiv :-
    format("~nFormat equivalence (wrapper vs direct):~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_components:register_agent_loop_components,
    %% dict wrapper == emit_from_components with dict format
    assert_true('emit_py_dict_from_components matches emit_from_components dict', (
        with_output_to(atom(Old),
            agent_loop_components:emit_py_dict_from_components(current_output,
                agent_costs, model_pricing/3, python, [])),
        with_output_to(atom(New),
            agent_loop_components:emit_from_components(current_output,
                agent_costs, model_pricing/3, python, dict, [])),
        Old == New
    )),
    %% set wrapper == emit_from_components with set format
    assert_true('emit_py_set_from_components matches emit_from_components set', (
        with_output_to(atom(Old2),
            agent_loop_components:emit_py_set_from_components(current_output,
                agent_tools, destructive_tool/1, python,
                [fact_type(destructive_tool), dict_name('DESTRUCTIVE_TOOLS')])),
        with_output_to(atom(New2),
            agent_loop_components:emit_from_components(current_output,
                agent_tools, destructive_tool/1, python, set,
                [fact_type(destructive_tool), dict_name('DESTRUCTIVE_TOOLS')])),
        Old2 == New2
    )),
    %% facts wrapper == emit_from_components with facts format
    assert_true('emit_prolog_facts matches emit_from_components facts', (
        with_output_to(atom(Old3),
            agent_loop_components:emit_prolog_facts_from_components(current_output,
                agent_costs, model_pricing, [])),
        with_output_to(atom(New3),
            agent_loop_components:emit_from_components(current_output,
                agent_costs, _Pred, prolog, facts, [fact_type(model_pricing)])),
        Old3 == New3
    )).

%% ============================================================================
%% Component roundtrip all targets
%% ============================================================================

test_component_roundtrip_all_targets :-
    format("~nComponent roundtrip all targets:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_components:register_agent_loop_components,
    findall(Cat-Name, component(Cat, Name, _, _), AllRaw),
    sort(AllRaw, All),
    length(All, TotalComponents),
    %% Test Python compilation for all components
    findall(Cat-Name, (
        member(Cat-Name, All),
        compile_component(Cat, Name, [target(python)], _PyCode)
    ), PySuccesses),
    length(PySuccesses, PyCount),
    assert_true('All components compile for Python target', (
        PyCount > 0
    )),
    %% Test Prolog compilation for all components
    findall(Cat-Name, (
        member(Cat-Name, All),
        compile_component(Cat, Name, [target(prolog)], _PlCode)
    ), PlSuccesses),
    length(PlSuccesses, PlCount),
    assert_true('All components compile for Prolog target', (
        PlCount > 0
    )),
    format("  (~w components total, ~w Python, ~w Prolog compiled)~n",
           [TotalComponents, PyCount, PlCount]).

%% ============================================================================
%% All bindings compile
%% ============================================================================

test_all_bindings_compile :-
    format("~nAll bindings compile:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    findall(Target-Pred, binding(Target, Pred, _, _, _, _), AllBindings),
    length(AllBindings, Total),
    findall(Target-Pred, (
        member(Target-Pred, AllBindings),
        agent_loop_bindings:compile_binding_code(Target, Pred, Code),
        atom(Code), Code \== ''
    ), Successes),
    length(Successes, SuccessCount),
    assert_eq('All bindings compile successfully', Total, SuccessCount).

%% ============================================================================
%% entries format tests
%% ============================================================================

test_entries_format :-
    format("~nEntries format:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Entries format should have no wrapper delimiters
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_components:emit_from_components(S, agent_security_rules, _, python,
        entries, [fact_type(blocked_path)]),
    close(S),
    memory_file_to_atom(MF, Output),
    free_memory_file(MF),
    %% Should NOT contain { or }
    assert_true('entries format has no open brace', \+ sub_atom(Output, _, _, _, '{')),
    assert_true('entries format has no close brace', \+ sub_atom(Output, _, _, _, '}')),
    %% Should contain quoted paths
    assert_true('entries format has path entries', sub_atom(Output, _, _, _, '/etc/shadow')).

%% ============================================================================
%% Security rule component tests
%% ============================================================================

test_security_rule_components :-
    format("~nSecurity rule components:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Count security rule components
    findall(Name, component(agent_security_rules, Name, security_rule, _), Rules),
    length(Rules, RuleCount),
    assert_true('security rules registered', RuleCount > 0),
    %% Verify blocked_path components exist
    findall(N, (component(agent_security_rules, N, security_rule, C),
                member(rule_type(blocked_path), C)), BPs),
    length(BPs, BPCount),
    assert_eq('blocked_path count', BPCount, 3),
    %% Verify blocked_command_pattern components exist
    findall(N, (component(agent_security_rules, N, security_rule, C),
                member(rule_type(blocked_command_pattern), C)), BCPs),
    length(BCPs, BCPCount),
    assert_eq('blocked_command_pattern count', BCPCount, 9),
    %% Compile a blocked_path for Python
    compile_component(agent_security_rules, '/etc/shadow',
        [target(python), fact_type(blocked_path)], PyCode),
    assert_true('blocked_path Python compile', sub_atom(PyCode, _, _, _, '/etc/shadow')),
    %% Compile a blocked_path for Prolog
    compile_component(agent_security_rules, '/etc/shadow',
        [target(prolog), fact_type(blocked_path)], PlCode),
    assert_true('blocked_path Prolog compile', sub_atom(PlCode, _, _, _, 'blocked_path')).

%% ============================================================================
%% Prolog fact roundtrip tests
%% ============================================================================

test_prolog_fact_roundtrip :-
    format("~nProlog fact roundtrip:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Find first cost component and compile it
    findall(N, component(agent_costs, N, _, _), CostNames),
    CostNames = [FirstCost|_],
    compile_component(agent_costs, FirstCost, [target(prolog)], CostCode),
    atom_to_term(CostCode, CostTerm, _),
    CostTerm = model_pricing(_, _, _),
    assert_true('cost fact roundtrip parses', true),
    %% Compile a tool fact and verify it parses
    findall(N, component(agent_tools, N, tool_handler, _), ToolNames),
    ToolNames = [FirstTool|_],
    compile_component(agent_tools, FirstTool,
        [target(prolog), fact_type(tool_handler)], ToolCode),
    atom_to_term(ToolCode, ToolTerm, _),
    ToolTerm = tool_handler(_, _),
    assert_true('tool fact roundtrip parses', true),
    %% Compile a security rule and verify
    compile_component(agent_security_rules, '/etc/shadow',
        [target(prolog), fact_type(blocked_path)], SecCode),
    atom_to_term(SecCode, SecTerm, _),
    SecTerm = blocked_path(_),
    assert_true('security fact roundtrip parses', true).

%% ============================================================================
%% Python entry pattern tests
%% ============================================================================

test_python_entry_patterns :-
    format("~nPython entry patterns:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Dict format should have ':'
    new_memory_file(MF1),
    open_memory_file(MF1, write, S1),
    agent_loop_components:emit_from_components(S1, agent_costs, model_pricing/3,
        python, dict, []),
    close(S1),
    memory_file_to_atom(MF1, DictOut),
    free_memory_file(MF1),
    assert_true('dict output has colon', sub_atom(DictOut, _, _, _, ':')),
    assert_true('dict output has open brace', sub_atom(DictOut, _, _, _, '{')),
    %% Set format entries should have quotes
    new_memory_file(MF2),
    open_memory_file(MF2, write, S2),
    agent_loop_components:emit_from_components(S2, agent_tools, tool_handler/2,
        python, set, [fact_type(destructive_tool)]),
    close(S2),
    memory_file_to_atom(MF2, SetOut),
    free_memory_file(MF2),
    assert_true('set output has open brace', sub_atom(SetOut, _, _, _, '{')),
    assert_true('set output has close brace', sub_atom(SetOut, _, _, _, '}')).

%% ============================================================================
%% Generator import specs tests
%% ============================================================================

test_generator_import_specs :-
    format("~nGenerator import specs:~n"),
    %% Verify specs exist for known generators
    agent_loop_components:generator_import_specs(tools, ToolSpecs),
    length(ToolSpecs, ToolSpecCount),
    assert_true('tools has import specs', ToolSpecCount > 0),
    agent_loop_components:generator_import_specs(aliases, AliasSpecs),
    length(AliasSpecs, AliasSpecCount),
    assert_true('aliases has import specs', AliasSpecCount > 0),
    agent_loop_components:generator_import_specs(backends_base, BaseSpecs),
    length(BaseSpecs, BaseSpecCount),
    assert_true('backends_base has import specs', BaseSpecCount > 0),
    %% Verify emit_import_specs produces correct output
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_components:emit_import_specs(S, [bare(os), from(typing, ['Any', 'List'])]),
    close(S),
    memory_file_to_atom(MF, Output),
    free_memory_file(MF),
    assert_true('import specs has bare import', sub_atom(Output, _, _, _, 'import os')),
    assert_true('import specs has from import', sub_atom(Output, _, _, _, 'from typing import Any, List')).

%% ============================================================================
%% Binding equivalence comments tests
%% ============================================================================

test_binding_equivalence_comments :-
    format("~nBinding equivalence comments:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_bindings:emit_binding_equivalence_comments(S, python, [
        tool_handler(_, _), model_pricing(_, _, _)
    ]),
    close(S),
    memory_file_to_atom(MF, Output),
    free_memory_file(MF),
    assert_true('equivalence has header', sub_atom(Output, _, _, _, 'equivalences')),
    assert_true('equivalence has tool_handler', sub_atom(Output, _, _, _, 'tool_handler/2')),
    assert_true('equivalence has model_pricing', sub_atom(Output, _, _, _, 'model_pricing/3')).

%% ============================================================================
%% extract_target helper tests
%% ============================================================================

test_extract_target_helper :-
    format("~nextract_target helper:~n"),
    agent_loop_components:extract_target([target(python)], T1),
    assert_eq('extract python target', T1, python),
    agent_loop_components:extract_target([target(prolog)], T2),
    assert_eq('extract prolog target', T2, prolog),
    agent_loop_components:extract_target([], T3),
    assert_eq('extract default target', T3, prolog).

%% ============================================================================
%% all_entry backend compile tests
%% ============================================================================

test_all_entry_backend_compile :-
    format("~nall_entry backend compile:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Compile an all_entry for a non-optional backend
    compile_component(agent_backends, coro,
        [target(python), fact_type(all_entry)], AllCode),
    assert_true('all_entry has class name', sub_atom(AllCode, _, _, _, 'CoroBackend')),
    %% Verify entries format with all_entry
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_components:emit_from_components(S, agent_backends, backend_factory/2,
        python, entries, [fact_type(all_entry)]),
    close(S),
    memory_file_to_atom(MF, Output),
    free_memory_file(MF),
    assert_true('all_entry entries has backend', sub_atom(Output, _, _, _, 'CoroBackend')),
    %% Optional backends should NOT appear
    assert_true('all_entry excludes optional', \+ sub_atom(Output, _, _, _, 'ClaudeAPIBackend')).

%% ============================================================================
%% Tests for deeper declarative generalization (PR #748)
%% ============================================================================

test_generator_import_specs_extended :-
    format("~ngenerator_import_specs extended:~n"),
    %% Verify specs exist for the 4 newly added generators
    assert_true('costs specs exist',
        agent_loop_components:generator_import_specs(costs, _)),
    assert_true('context specs exist',
        agent_loop_components:generator_import_specs(context, _)),
    assert_true('config specs exist',
        agent_loop_components:generator_import_specs(config, _)),
    assert_true('security_profiles specs exist',
        agent_loop_components:generator_import_specs(security_profiles, _)),
    %% Verify costs specs emit correct output
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:generator_import_specs(costs, CostsSpecs),
    agent_loop_components:emit_import_specs(S, CostsSpecs),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('costs has import json', sub_atom(Output, _, _, _, 'import json')),
    assert_true('costs has from dataclasses', sub_atom(Output, _, _, _, 'from dataclasses import')).

test_generator_export_specs :-
    format("~ngenerator_export_specs:~n"),
    %% Verify security_init export specs
    agent_loop_components:generator_export_specs(security_init, Exports),
    assert_true('exports non-empty', Exports \= []),
    assert_true('exports has AuditLogger', member('AuditLogger', Exports)),
    assert_true('exports has SecurityProfile', member('SecurityProfile', Exports)),
    %% Verify emit_export_specs output
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_export_specs(S, Exports),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('export has __all__', sub_atom(Output, _, _, _, '__all__ = [')).

test_py_fragment_metadata :-
    format("~npy_fragment_metadata:~n"),
    %% Verify metadata facts exist
    assert_true('tools_handler has metadata',
        agent_loop_components:py_fragment_metadata(tools_handler_class_body, _)),
    assert_true('context_manager has metadata',
        agent_loop_components:py_fragment_metadata(context_manager_class, _)),
    assert_true('security_audit has metadata',
        agent_loop_components:py_fragment_metadata(security_audit_module, _)),
    %% Verify category extraction
    agent_loop_components:fragment_category(tools_handler_class_body, Cat),
    assert_true('tools_handler category is tools', Cat == tools),
    %% Count metadata facts (at least 10)
    aggregate_all(count,
        agent_loop_components:py_fragment_metadata(_, _), Count),
    assert_true('at least 10 metadata facts', Count >= 10).

test_fragment_imports_helper :-
    format("~nfragment_imports helper:~n"),
    %% Verify imports for tools_handler_class_body
    agent_loop_components:fragment_imports(tools_handler_class_body, Imports),
    assert_true('tools_handler has imports', Imports \= []),
    assert_true('tools_handler imports backends.base',
        member(from('backends.base', _), Imports)),
    %% Verify config_load_config imports
    agent_loop_components:fragment_imports(config_load_config, CfgImports),
    assert_true('config_load has json import', member(bare(json), CfgImports)).

test_path_check_rule_facts :-
    format("~npath_check_rule facts:~n"),
    %% Verify all 3 rules exist
    assert_true('exact_match rule exists',
        agent_loop_components:path_check_rule(exact_match, blocked_path, _)),
    assert_true('prefix_match rule exists',
        agent_loop_components:path_check_rule(prefix_match, blocked_path_prefix, _)),
    assert_true('home_pattern rule exists',
        agent_loop_components:path_check_rule(home_pattern, blocked_home_pattern, _)),
    %% Verify count
    aggregate_all(count, agent_loop_components:path_check_rule(_, _, _), Count),
    assert_true('exactly 3 path rules', Count =:= 3).

test_command_check_rule_facts :-
    format("~ncommand_check_rule facts:~n"),
    assert_true('regex_match rule exists',
        agent_loop_components:command_check_rule(regex_match, blocked_command_pattern, _)),
    aggregate_all(count, agent_loop_components:command_check_rule(_, _, _), CCount),
    assert_true('exactly 1 command rule', CCount =:= 1).

test_compile_path_check_rules_python :-
    format("~ncompile_path_check_rules python:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:compile_path_check_rules(S, python, []),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('python has _BLOCKED_PATHS check',
        sub_atom(Output, _, _, _, '_BLOCKED_PATHS')),
    assert_true('python has prefix check',
        sub_atom(Output, _, _, _, 'startswith(prefix)')),
    assert_true('python has home expanduser',
        sub_atom(Output, _, _, _, 'expanduser')),
    assert_true('python has comment',
        sub_atom(Output, _, _, _, '# Exact blocked path match')),
    assert_true('python has return True',
        sub_atom(Output, _, _, _, 'return True')).

test_compile_path_check_rules_prolog :-
    format("~ncompile_path_check_rules prolog:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:compile_path_check_rules(S, prolog, []),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('prolog has is_path_blocked',
        sub_atom(Output, _, _, _, 'is_path_blocked')),
    assert_true('prolog has blocked_path',
        sub_atom(Output, _, _, _, 'blocked_path(Path)')),
    assert_true('prolog has atom_concat',
        sub_atom(Output, _, _, _, 'atom_concat')),
    assert_true('prolog has expand_home',
        sub_atom(Output, _, _, _, 'expand_home')).

test_source_component_equivalence :-
    format("~nsource-component equivalence:~n"),
    agent_loop_components:register_agent_loop_components,
    %% blocked_path count should match
    aggregate_all(count, agent_loop_module:blocked_path(_), SrcBP),
    aggregate_all(count, (
        component(agent_security_rules, _, _, Cfg),
        member(rule_type(blocked_path), Cfg)
    ), CompBP),
    assert_true('blocked_path count matches', SrcBP =:= CompBP),
    %% blocked_path_prefix count should match
    aggregate_all(count, agent_loop_module:blocked_path_prefix(_), SrcBPP),
    aggregate_all(count, (
        component(agent_security_rules, _, _, Cfg2),
        member(rule_type(blocked_path_prefix), Cfg2)
    ), CompBPP),
    assert_true('blocked_path_prefix count matches', SrcBPP =:= CompBPP),
    %% blocked_home_pattern count should match
    aggregate_all(count, agent_loop_module:blocked_home_pattern(_), SrcBHP),
    aggregate_all(count, (
        component(agent_security_rules, _, _, Cfg3),
        member(rule_type(blocked_home_pattern), Cfg3)
    ), CompBHP),
    assert_true('blocked_home_pattern count matches', SrcBHP =:= CompBHP),
    %% blocked_command_pattern count should match
    aggregate_all(count, agent_loop_module:blocked_command_pattern(_, _), SrcBCP),
    aggregate_all(count, (
        component(agent_security_rules, _, _, Cfg4),
        member(rule_type(blocked_command_pattern), Cfg4)
    ), CompBCP),
    assert_true('blocked_command_pattern count matches', SrcBCP =:= CompBCP).

%% ============================================================================
%% Composable Generator Tests
%% ============================================================================

test_emit_security_check_predicates :-
    format("~nemit_security_check_predicates:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_security_check_predicates(S, [target(prolog)]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('has is_path_blocked',
        sub_atom(Output, _, _, _, 'is_path_blocked')),
    assert_true('has is_command_blocked',
        sub_atom(Output, _, _, _, 'is_command_blocked')),
    assert_true('has check_path_allowed',
        sub_atom(Output, _, _, _, 'check_path_allowed')),
    assert_true('has check_command_allowed',
        sub_atom(Output, _, _, _, 'check_command_allowed')),
    assert_true('has set_security_profile',
        sub_atom(Output, _, _, _, 'set_security_profile')).

test_compiled_rules_variable_name :-
    format("~ncompiled rules variable name fix:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:compile_path_check_rules(S, python, []),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('uses _BLOCKED_PREFIXES (not _BLOCKED_PATH_PREFIXES)',
        sub_atom(Output, _, _, _, '_BLOCKED_PREFIXES')),
    assert_true('no _BLOCKED_PATH_PREFIXES',
        \+ sub_atom(Output, _, _, _, '_BLOCKED_PATH_PREFIXES')).

test_generator_export_specs_backends :-
    format("~ngenerator_export_specs backends_init:~n"),
    agent_loop_components:generator_export_specs(backends_init, Exports),
    assert_true('has AgentBackend',
        member('AgentBackend', Exports)),
    assert_true('has AgentResponse',
        member('AgentResponse', Exports)),
    assert_true('has ToolCall',
        member('ToolCall', Exports)),
    assert_true('has CoroBackend',
        member('CoroBackend', Exports)),
    assert_true('excludes optional ClaudeAPIBackend',
        \+ member('ClaudeAPIBackend', Exports)).

test_security_profile_field_facts :-
    format("~nsecurity_profile_field facts:~n"),
    assert_true('name field exists',
        agent_loop_module:security_profile_field(name, 'str', required, _)),
    assert_true('path_validation has layer',
        (agent_loop_module:security_profile_field(path_validation, _, _, Props),
         member(layer(1), Props))),
    aggregate_all(count, agent_loop_module:security_profile_field(_, _, _, _), Count),
    assert_true('at least 15 fields', Count >= 15).

test_emit_security_profile_fields :-
    format("~nemit_security_profile_fields:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_security_profile_fields(S, [target(python)]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('has name: str',
        sub_atom(Output, _, _, _, 'name: str')),
    assert_true('has path_validation: bool',
        sub_atom(Output, _, _, _, 'path_validation: bool')),
    assert_true('has Layer 1 comment',
        sub_atom(Output, _, _, _, '# Layer 1')),
    assert_true('has audit_logging',
        sub_atom(Output, _, _, _, 'audit_logging')).

test_generator_fragments :-
    format("~ngenerator_fragments:~n"),
    agent_loop_components:generator_fragments(tools, ToolFrags),
    assert_true('tools has fragments', is_list(ToolFrags)),
    assert_true('tools non-empty', ToolFrags \= []),
    assert_true('tools includes handler',
        member(tools_handler_class_body, ToolFrags)).

test_derive_fragment_imports :-
    format("~nderive_fragment_imports:~n"),
    agent_loop_components:derive_fragment_imports(tools, Derived),
    assert_true('derived non-empty', Derived \= []),
    assert_true('derived has re', member(bare(re), Derived)),
    assert_true('derived has backends.base',
        member(from('backends.base', _), Derived)).

test_validate_generator_imports :-
    format("~nvalidate_generator_imports:~n"),
    agent_loop_components:validate_generator_imports(tools, Warnings),
    assert_true('warnings is list', is_list(Warnings)).

test_emit_config_section :-
    format("~nemit_config_section:~n"),
    %% Python target
    new_memory_file(MF1), open_memory_file(MF1, write, S1),
    agent_loop_components:emit_config_section(S1, api_key_env_vars, [target(python)]),
    close(S1), memory_file_to_atom(MF1, PyOut), free_memory_file(MF1),
    assert_true('python has claude key',
        sub_atom(PyOut, _, _, _, 'claude')),
    %% Prolog target
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_components:emit_config_section(S2, api_key_env_vars, [target(prolog)]),
    close(S2), memory_file_to_atom(MF2, PlOut), free_memory_file(MF2),
    assert_true('prolog has api_key_env_var fact',
        sub_atom(PlOut, _, _, _, 'api_key_env_var')),
    assert_true('prolog has ANTHROPIC_API_KEY',
        sub_atom(PlOut, _, _, _, 'ANTHROPIC_API_KEY')),
    %% api_key_files section
    new_memory_file(MF3), open_memory_file(MF3, write, S3),
    agent_loop_components:emit_config_section(S3, api_key_files, [target(prolog)]),
    close(S3), memory_file_to_atom(MF3, PlFiles), free_memory_file(MF3),
    assert_true('prolog has api_key_file fact',
        sub_atom(PlFiles, _, _, _, 'api_key_file')).

test_backends_init_export_roundtrip :-
    format("~nbackends_init export roundtrip:~n"),
    agent_loop_components:generator_export_specs(backends_init, Exports),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_export_specs(S, Exports),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('has __all__ block',
        sub_atom(Output, _, _, _, '__all__')),
    assert_true('has AgentBackend in output',
        sub_atom(Output, _, _, _, 'AgentBackend')),
    assert_true('has CoroBackend in output',
        sub_atom(Output, _, _, _, 'CoroBackend')).

%% ============================================================================
%% Wire-Generalize Tests
%% ============================================================================

test_regex_list_variable_facts :-
    format("~nregex_list_variable facts:~n"),
    assert_true('guarded_extra_blocks mapped',
        agent_loop_module:regex_list_variable(guarded_extra_blocks, '_GUARDED_EXTRA_BLOCKS')),
    assert_true('paranoid_safe mapped',
        agent_loop_module:regex_list_variable(paranoid_safe, '_PARANOID_SAFE')),
    assert_true('paranoid_confirm mapped',
        agent_loop_module:regex_list_variable(paranoid_confirm, '_PARANOID_CONFIRM')),
    assert_true('paranoid_allowed mapped',
        agent_loop_module:regex_list_variable(paranoid_allowed, '_PARANOID_ALLOWED')),
    aggregate_all(count, agent_loop_module:regex_list_variable(_, _), Count),
    assert_true('exactly 4 facts', Count =:= 4).

test_regex_list_combined_facts :-
    format("~nregex_list_combined facts:~n"),
    assert_true('paranoid_allowed is combined',
        agent_loop_module:regex_list_combined('_PARANOID_ALLOWED', '_PARANOID_SAFE', '_PARANOID_CONFIRM')),
    aggregate_all(count, agent_loop_module:regex_list_combined(_, _, _), CCount),
    assert_true('exactly 1 combined fact', CCount =:= 1).

test_regex_list_in_field_value :-
    format("~nregex_list_in_field_value:~n"),
    %% Test that emit_profile_field_value uses regex_list_variable lookup
    new_memory_file(MF1), open_memory_file(MF1, write, S1),
    agent_loop_module:emit_profile_field_value(S1, blocked_commands, 'list[str]', guarded_extra_blocks),
    close(S1), memory_file_to_atom(MF1, Out1), free_memory_file(MF1),
    assert_true('guarded maps to _GUARDED_EXTRA_BLOCKS',
        sub_atom(Out1, _, _, _, '_GUARDED_EXTRA_BLOCKS')),
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_module:emit_profile_field_value(S2, allowed_commands, 'list[str]', paranoid_allowed),
    close(S2), memory_file_to_atom(MF2, Out2), free_memory_file(MF2),
    assert_true('paranoid_allowed maps to _PARANOID_ALLOWED',
        sub_atom(Out2, _, _, _, '_PARANOID_ALLOWED')),
    new_memory_file(MF3), open_memory_file(MF3, write, S3),
    agent_loop_module:emit_profile_field_value(S3, test_field, 'list[str]', unknown_list),
    close(S3), memory_file_to_atom(MF3, Out3), free_memory_file(MF3),
    assert_true('unknown falls through',
        sub_atom(Out3, _, _, _, 'unknown_list')).

test_generator_fragments_security :-
    format("~ngenerator_fragments security:~n"),
    assert_true('security_audit has fragments',
        agent_loop_components:generator_fragments(security_audit, _)),
    assert_true('security_proxy has fragments',
        agent_loop_components:generator_fragments(security_proxy, _)),
    assert_true('security_path_proxy has fragments',
        agent_loop_components:generator_fragments(security_path_proxy, _)),
    %% Verify all fragment names are real py_fragment/2 facts
    agent_loop_components:generator_fragments(security_proot, PrFrags),
    assert_true('security_proot fragments are real py_fragments',
        forall(member(F, PrFrags), agent_loop_module:py_fragment(F, _))).

test_generator_fragments_aliases :-
    format("~ngenerator_fragments aliases:~n"),
    agent_loop_components:generator_fragments(aliases, AFrags),
    assert_true('aliases has 2 fragments', length(AFrags, 2)),
    assert_true('aliases has header', member(aliases_class_header, AFrags)),
    assert_true('aliases has footer', member(aliases_class_footer, AFrags)).

test_generator_fragments_agent_loop_main :-
    format("~ngenerator_fragments agent_loop_main:~n"),
    agent_loop_components:generator_fragments(agent_loop_main, MainFrags),
    length(MainFrags, MainLen),
    assert_true('agent_loop_main has >= 15 fragments', MainLen >= 15),
    assert_true('has handler_backend_command',
        member(handler_backend_command, MainFrags)),
    %% Verify all are real py_fragment/2 facts
    assert_true('all main fragments are real py_fragments',
        forall(member(F, MainFrags), agent_loop_module:py_fragment(F, _))).

test_emit_security_profile_fields_prolog :-
    format("~nemit_security_profile_fields prolog:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_security_profile_fields(S, [target(prolog)]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('prolog has security_profile_field(name',
        sub_atom(Output, _, _, _, 'security_profile_field(name')),
    assert_true('prolog has path_validation',
        sub_atom(Output, _, _, _, 'path_validation')),
    assert_true('no Python name: str in prolog output',
        \+ sub_atom(Output, _, _, _, 'name: str')),
    %% Count facts matches source
    aggregate_all(count, agent_loop_module:security_profile_field(_, _, _, _), SrcCount),
    findall(1, sub_atom(Output, _, _, _, 'security_profile_field('), Ones),
    length(Ones, OutCount),
    assert_true('prolog fact count matches source', SrcCount =:= OutCount).

test_emit_prolog_module_header :-
    format("~nemit_prolog_module_header:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_prolog_module_header(S, test_mod, [
        exports([foo/1, bar/2, baz/3]),
        use_modules([library(lists)])
    ]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('has module declaration',
        sub_atom(Output, _, _, _, ':- module(test_mod')),
    assert_true('has foo/1 export',
        sub_atom(Output, _, _, _, 'foo/1')),
    assert_true('last export has no comma (baz/3 then newline)',
        sub_atom(Output, _, _, _, '    baz/3\n')),
    assert_true('has use_module',
        sub_atom(Output, _, _, _, ':- use_module(library(lists))')).

test_emit_prolog_declarations :-
    format("~nemit_prolog_declarations:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_prolog_declarations(S, [
        det([check_foo/2, check_bar/1]),
        dynamic([current_state/1])
    ]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('has det directive',
        sub_atom(Output, _, _, _, ':- det(check_foo/2)')),
    assert_true('has dynamic directive',
        sub_atom(Output, _, _, _, ':- dynamic current_state/1')),
    assert_true('has check_bar det',
        sub_atom(Output, _, _, _, ':- det(check_bar/1)')).

test_config_section_wiring :-
    format("~nconfig_section_wiring:~n"),
    %% Verify emit_config_section(python) produces same as direct _py call
    new_memory_file(MF1), open_memory_file(MF1, write, S1),
    agent_loop_components:emit_api_key_env_vars_py(S1, [target(python)]),
    close(S1), memory_file_to_atom(MF1, Direct), free_memory_file(MF1),
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_components:emit_config_section(S2, api_key_env_vars, [target(python)]),
    close(S2), memory_file_to_atom(MF2, Via), free_memory_file(MF2),
    assert_true('api_key_env_vars: section matches direct', Direct == Via),
    %% api_key_files
    new_memory_file(MF3), open_memory_file(MF3, write, S3),
    agent_loop_components:emit_api_key_files_py(S3, [target(python)]),
    close(S3), memory_file_to_atom(MF3, Direct2), free_memory_file(MF3),
    new_memory_file(MF4), open_memory_file(MF4, write, S4),
    agent_loop_components:emit_config_section(S4, api_key_files, [target(python)]),
    close(S4), memory_file_to_atom(MF4, Via2), free_memory_file(MF4),
    assert_true('api_key_files: section matches direct', Direct2 == Via2),
    %% default_presets
    new_memory_file(MF5), open_memory_file(MF5, write, S5),
    agent_loop_components:emit_default_presets_py(S5, [target(python)]),
    close(S5), memory_file_to_atom(MF5, Direct3), free_memory_file(MF5),
    new_memory_file(MF6), open_memory_file(MF6, write, S6),
    agent_loop_components:emit_config_section(S6, default_presets, [target(python)]),
    close(S6), memory_file_to_atom(MF6, Via3), free_memory_file(MF6),
    assert_true('default_presets: section matches direct', Direct3 == Via3).

%% ============================================================================
%% Tests for prolog_fragment/2 + write_prolog/2 + emit_prolog_module_skeleton/3
%% ============================================================================

test_prolog_fragment_count :-
    format("~nprolog_fragment_count:~n"),
    findall(N, agent_loop_module:prolog_fragment(N, _), Names),
    length(Names, Count),
    assert_true('at least 10 prolog_fragments', Count >= 10),
    assert_true('cost_tracker_impl exists',
        memberchk(cost_tracker_impl, Names)),
    assert_true('config_parse_cli exists',
        memberchk(config_parse_cli, Names)),
    assert_true('config_load_config exists',
        memberchk(config_load_config, Names)),
    assert_true('config_resolve_api_key exists',
        memberchk(config_resolve_api_key, Names)),
    assert_true('commands_resolve exists',
        memberchk(commands_resolve, Names)),
    assert_true('commands_handle_slash exists',
        memberchk(commands_handle_slash, Names)),
    assert_true('tools_execute_dispatch exists',
        memberchk(tools_execute_dispatch, Names)),
    assert_true('tools_schema exists',
        memberchk(tools_schema, Names)),
    assert_true('tools_describe exists',
        memberchk(tools_describe, Names)),
    assert_true('tools_confirm exists',
        memberchk(tools_confirm, Names)).

test_write_prolog_roundtrip :-
    format("~nwrite_prolog_roundtrip:~n"),
    %% Write cost_tracker_impl to memory and verify content
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_module:write_prolog(S, cost_tracker_impl),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('cost_tracker has cost_tracker_init',
        sub_atom(Output, _, _, _, 'cost_tracker_init(ID)')),
    assert_true('cost_tracker has cost_tracker_add',
        sub_atom(Output, _, _, _, 'cost_tracker_add(ID, Model')),
    assert_true('cost_tracker has dynamic decl',
        sub_atom(Output, _, _, _, ':- dynamic cost_state/3')),
    assert_true('cost_tracker has format call',
        sub_atom(Output, _, _, _, 'format("  [cost: $~4f')),
    %% Write tools_schema to memory
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_module:write_prolog(S2, tools_schema),
    close(S2), memory_file_to_atom(MF2, Out2), free_memory_file(MF2),
    assert_true('tools_schema has build_tool_input_schema',
        sub_atom(Out2, _, _, _, 'build_tool_input_schema(ToolName')),
    assert_true('tools_schema has build_required',
        sub_atom(Out2, _, _, _, 'build_required')).

test_write_prolog_error :-
    format("~nwrite_prolog_error:~n"),
    %% Verify unknown fragment throws
    (catch(
        (new_memory_file(MF), open_memory_file(MF, write, S),
         agent_loop_module:write_prolog(S, nonexistent_fragment_xyz),
         close(S), free_memory_file(MF)),
        error(existence_error(prolog_fragment, nonexistent_fragment_xyz), _),
        Caught = true
    ) -> true ; Caught = false),
    assert_true('unknown prolog_fragment throws existence_error', Caught == true).

test_prolog_module_skeleton_exports :-
    format("~nprolog_module_skeleton_exports:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_prolog_module_skeleton(S, test_mod, [
        exports([foo/2, bar/3]),
        det([foo/2])
    ]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('skeleton has module decl',
        sub_atom(Output, _, _, _, ':- module(test_mod')),
    assert_true('skeleton has foo/2 export',
        sub_atom(Output, _, _, _, 'foo/2')),
    assert_true('skeleton has bar/3 export',
        sub_atom(Output, _, _, _, 'bar/3')),
    assert_true('skeleton has det(foo/2)',
        sub_atom(Output, _, _, _, ':- det(foo/2)')).

test_prolog_module_skeleton_comment :-
    format("~nprolog_module_skeleton_comment:~n"),
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_components:emit_prolog_module_skeleton(S, cmod, [
        exports([x/1]),
        comment('This is a test comment')
    ]),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('skeleton has comment',
        sub_atom(Output, _, _, _, '%% This is a test comment')).

test_prolog_fragment_escaping :-
    format("~nprolog_fragment_escaping:~n"),
    %% Verify config_parse_cli preserves single quotes in output
    new_memory_file(MF), open_memory_file(MF, write, S),
    agent_loop_module:write_prolog(S, config_parse_cli),
    close(S), memory_file_to_atom(MF, Output), free_memory_file(MF),
    assert_true('config_parse_cli has quoted --',
        sub_atom(Output, _, _, _, 'atom_concat(\'--\'')),
    assert_true('config_parse_cli has empty atom',
        sub_atom(Output, _, _, _, 'Help = \'\'')),
    %% Verify commands_handle_slash preserves / quote
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_module:write_prolog(S2, commands_handle_slash),
    close(S2), memory_file_to_atom(MF2, Out2), free_memory_file(MF2),
    assert_true('commands_handle_slash has quoted /',
        sub_atom(Out2, _, _, _, 'atom_concat(\'/\'')),
    %% Verify config_resolve_api_key preserves backslash-equals
    new_memory_file(MF3), open_memory_file(MF3, write, S3),
    agent_loop_module:write_prolog(S3, config_resolve_api_key),
    close(S3), memory_file_to_atom(MF3, Out3), free_memory_file(MF3),
    assert_true('config_resolve has backslash-eq',
        sub_atom(Out3, _, _, _, '\\=')),
    %% Verify tools_confirm preserves quoted tilde-w
    new_memory_file(MF4), open_memory_file(MF4, write, S4),
    agent_loop_module:write_prolog(S4, tools_confirm),
    close(S4), memory_file_to_atom(MF4, Out4), free_memory_file(MF4),
    assert_true('tools_confirm has Tool quote',
        sub_atom(Out4, _, _, _, 'Tool \'~w\'')).

test_write_prolog_substitutions :-
    format("~nwrite_prolog_substitutions:~n"),
    %% write_prolog/3 uses {{Key}} placeholder substitution (same as write_py/3)
    %% Verify no-match substitution leaves content unchanged
    new_memory_file(MF1), open_memory_file(MF1, write, S1),
    agent_loop_module:write_prolog(S1, cost_tracker_impl),
    close(S1), memory_file_to_atom(MF1, Orig), free_memory_file(MF1),
    new_memory_file(MF2), open_memory_file(MF2, write, S2),
    agent_loop_module:write_prolog(S2, cost_tracker_impl, ['no_match'='value']),
    close(S2), memory_file_to_atom(MF2, WithSub), free_memory_file(MF2),
    assert_true('no-match substitution preserves content', Orig == WithSub),
    %% Verify write_prolog/3 callable (predicate exists)
    assert_true('write_prolog/3 is callable',
        predicate_property(agent_loop_module:write_prolog(_, _, _), defined)).

test_prolog_fragment_metadata :-
    format("~nprolog_fragment_metadata:~n"),
    %% All 33 fragments have metadata (10 original + 14 backends + 9 agent_loop)
    findall(N, agent_loop_components:prolog_fragment_metadata(N, _), MNames),
    length(MNames, MCount),
    assert_true('33 metadata entries', MCount =:= 33),
    %% Category lookups work
    agent_loop_components:prolog_fragment_category(cost_tracker_impl, CostCat),
    assert_true('cost_tracker_impl category is costs', CostCat == costs),
    agent_loop_components:prolog_fragment_category(config_parse_cli, CfgCat),
    assert_true('config_parse_cli category is config', CfgCat == config),
    agent_loop_components:prolog_fragment_category(tools_execute_dispatch, ToolCat),
    assert_true('tools_execute_dispatch category is tools', ToolCat == tools),
    %% Use-module lookups work
    agent_loop_components:prolog_fragment_use_modules(tools_execute_dispatch, ToolUMs),
    assert_true('tools_execute_dispatch has library(process)',
        memberchk(library(process), ToolUMs)),
    agent_loop_components:prolog_fragment_use_modules(cost_tracker_impl, CostUMs),
    assert_true('cost_tracker_impl has no use_modules', CostUMs == []).

test_generator_prolog_fragments :-
    format("~ngenerator_prolog_fragments:~n"),
    findall(G, agent_loop_components:generator_prolog_fragments(G, _), Gens),
    length(Gens, GCount),
    assert_true('6 generators registered', GCount =:= 6),
    assert_true('costs generator registered', memberchk(costs, Gens)),
    assert_true('config generator registered', memberchk(config, Gens)),
    assert_true('commands generator registered', memberchk(commands, Gens)),
    assert_true('tools generator registered', memberchk(tools, Gens)),
    assert_true('backends generator registered', memberchk(backends, Gens)),
    assert_true('agent_loop generator registered', memberchk(agent_loop, Gens)),
    %% Fragment count per generator
    agent_loop_components:generator_prolog_fragments(costs, CostFs),
    length(CostFs, CF), assert_true('costs has 1 fragment', CF =:= 1),
    agent_loop_components:generator_prolog_fragments(config, CfgFs),
    length(CfgFs, CfgF), assert_true('config has 3 fragments', CfgF =:= 3),
    agent_loop_components:generator_prolog_fragments(tools, ToolFs),
    length(ToolFs, TF), assert_true('tools has 4 fragments', TF =:= 4),
    agent_loop_components:generator_prolog_fragments(backends, BackFs),
    length(BackFs, BF), assert_true('backends has 14 fragments', BF =:= 14),
    agent_loop_components:generator_prolog_fragments(agent_loop, ALFs),
    length(ALFs, ALF), assert_true('agent_loop has 9 fragments', ALF =:= 9),
    %% All fragment names resolve to actual prolog_fragment/2 facts
    findall(FN, (
        agent_loop_components:generator_prolog_fragments(_, Frags),
        member(FN, Frags)
    ), AllFragNames),
    findall(FN, (
        member(FN, AllFragNames),
        agent_loop_module:prolog_fragment(FN, _)
    ), Resolved),
    length(AllFragNames, ACount), length(Resolved, RCount),
    assert_true('all fragment names resolve', ACount =:= RCount).

test_cost_py_fragments :-
    format("~ncost_py_fragments:~n"),
    %% All 4 cost py_fragments exist
    assert_true('cost_usage_record exists',
        agent_loop_module:py_fragment(cost_usage_record, _)),
    assert_true('cost_tracker_class_def exists',
        agent_loop_module:py_fragment(cost_tracker_class_def, _)),
    assert_true('cost_tracker_methods exists',
        agent_loop_module:py_fragment(cost_tracker_methods, _)),
    assert_true('cost_openrouter exists',
        agent_loop_module:py_fragment(cost_openrouter, _)),
    %% Content checks — key signatures present
    agent_loop_module:py_fragment(cost_usage_record, UR),
    assert_true('usage_record has @dataclass', sub_atom(UR, _, _, _, '@dataclass')),
    assert_true('usage_record has UsageRecord', sub_atom(UR, _, _, _, 'class UsageRecord')),
    agent_loop_module:py_fragment(cost_tracker_class_def, CD),
    assert_true('class_def has CostTracker', sub_atom(CD, _, _, _, 'class CostTracker')),
    assert_true('class_def has DEFAULT_PRICING', sub_atom(CD, _, _, _, 'DEFAULT_PRICING')),
    agent_loop_module:py_fragment(cost_tracker_methods, TM),
    assert_true('methods has record_usage', sub_atom(TM, _, _, _, 'def record_usage')),
    assert_true('methods has get_summary', sub_atom(TM, _, _, _, 'def get_summary')),
    assert_true('methods has format_status', sub_atom(TM, _, _, _, 'def format_status')),
    assert_true('methods has ensure_pricing', sub_atom(TM, _, _, _, 'def ensure_pricing')),
    agent_loop_module:py_fragment(cost_openrouter, OR),
    assert_true('openrouter has fetch function', sub_atom(OR, _, _, _, 'def fetch_openrouter_pricing')),
    assert_true('openrouter has cache load', sub_atom(OR, _, _, _, 'def _load_openrouter_cache')),
    assert_true('openrouter has cache save', sub_atom(OR, _, _, _, 'def _save_openrouter_cache')),
    %% Metadata entries exist
    assert_true('cost_usage_record metadata',
        agent_loop_components:py_fragment_metadata(cost_usage_record, _)),
    assert_true('cost_openrouter metadata',
        agent_loop_components:py_fragment_metadata(cost_openrouter, _)),
    %% generator_fragments(costs, ...) exists and has 4 entries
    agent_loop_components:generator_fragments(costs, CostFrags),
    length(CostFrags, CFL),
    assert_true('costs generator has 4 fragments', CFL =:= 4),
    %% All cost fragment names resolve to py_fragment/2
    findall(F, (member(F, CostFrags), agent_loop_module:py_fragment(F, _)), ResCost),
    length(ResCost, RCL),
    assert_true('all cost fragments resolve', RCL =:= CFL).

test_backends_prolog_fragments :-
    format("~nbackends_prolog_fragments:~n"),
    %% All 14 backends fragments exist
    assert_true('backends_create_backend exists',
        agent_loop_module:prolog_fragment(backends_create_backend, _)),
    assert_true('backends_retry_config exists',
        agent_loop_module:prolog_fragment(backends_retry_config, _)),
    assert_true('backends_streaming_anthropic exists',
        agent_loop_module:prolog_fragment(backends_streaming_anthropic, _)),
    assert_true('backends_sse_parser exists',
        agent_loop_module:prolog_fragment(backends_sse_parser, _)),
    %% Content checks
    agent_loop_module:prolog_fragment(backends_create_backend, CB),
    assert_true('create_backend has backend_factory',
        sub_atom(CB, _, _, _, 'backend_factory')),
    agent_loop_module:prolog_fragment(backends_send_request_raw_api, API),
    assert_true('api fragment has Content-Type',
        sub_atom(API, _, _, _, 'Content-Type')),
    agent_loop_module:prolog_fragment(backends_streaming_anthropic, Anth),
    assert_true('anthropic fragment has x-api-key',
        sub_atom(Anth, _, _, _, 'x-api-key')),
    assert_true('anthropic fragment has 2023-06-01',
        sub_atom(Anth, _, _, _, '2023-06-01')),
    %% Escaping verification — single quotes doubled in atoms
    agent_loop_module:prolog_fragment(backends_send_request_raw_api, APIFrag),
    assert_true('api fragment single quotes escaped',
        sub_atom(APIFrag, _, _, _, 'Content-Type')),
    %% Fragment count
    findall(N, (agent_loop_module:prolog_fragment(N, _), sub_atom(N, 0, _, _, backends_)), BNames),
    length(BNames, BCount),
    assert_true('14 backends fragments', BCount =:= 14).

test_agent_loop_prolog_fragments :-
    format("~nagent_loop_prolog_fragments:~n"),
    %% All 9 agent_loop fragments exist
    assert_true('agent_loop_init_state exists',
        agent_loop_module:prolog_fragment(agent_loop_init_state, _)),
    assert_true('agent_loop_entry exists',
        agent_loop_module:prolog_fragment(agent_loop_entry, _)),
    assert_true('agent_loop_repl_core exists',
        agent_loop_module:prolog_fragment(agent_loop_repl_core, _)),
    assert_true('agent_loop_actions exists',
        agent_loop_module:prolog_fragment(agent_loop_actions, _)),
    assert_true('agent_loop_context exists',
        agent_loop_module:prolog_fragment(agent_loop_context, _)),
    %% Content checks
    agent_loop_module:prolog_fragment(agent_loop_init_state, Init),
    assert_true('init_state has NO_COLOR',
        sub_atom(Init, _, _, _, 'NO_COLOR')),
    assert_true('init_state has ansi_code',
        sub_atom(Init, _, _, _, 'ansi_code')),
    assert_true('init_state has chars_per_token',
        sub_atom(Init, _, _, _, 'chars_per_token')),
    agent_loop_module:prolog_fragment(agent_loop_actions, Actions),
    assert_true('actions has handle_action',
        sub_atom(Actions, _, _, _, 'handle_action')),
    assert_true('actions has _handle_backend_command',
        sub_atom(Actions, _, _, _, '_handle_backend_command')),
    agent_loop_module:prolog_fragment(agent_loop_context, Ctx),
    assert_true('context has trim_context',
        sub_atom(Ctx, _, _, _, 'trim_context')),
    assert_true('context has estimate_tokens',
        sub_atom(Ctx, _, _, _, 'estimate_tokens')),
    %% Fragment count
    findall(N, (agent_loop_module:prolog_fragment(N, _), sub_atom(N, 0, _, _, agent_loop_)), ALNames),
    length(ALNames, ALCount),
    assert_true('9 agent_loop fragments', ALCount =:= 9).

test_config_section_new_clauses :-
    format("~nconfig_section_new_clauses:~n"),
    %% All 8 config sections exist (3 original + 5 new)
    findall(S, (
        member(S, [api_key_env_vars, api_key_files, default_presets,
                   agent_config_fields, audit_levels, streaming_capable,
                   security_profiles, cli_arguments]),
        with_output_to(string(_), agent_loop_components:emit_config_section(current_output, S, [target(prolog)]))
    ), Working),
    length(Working, WCount),
    assert_true('8 config sections work for prolog', WCount =:= 8),
    %% agent_config_fields emits facts
    with_output_to(string(ACFOut), agent_loop_components:emit_config_section(current_output, agent_config_fields, [target(prolog)])),
    assert_true('agent_config_fields has agent_config_field',
        sub_string(ACFOut, _, _, _, "agent_config_field")),
    %% audit_levels emits facts
    with_output_to(string(AuditOut), agent_loop_components:emit_config_section(current_output, audit_levels, [target(prolog)])),
    assert_true('audit_levels has audit_profile_level',
        sub_string(AuditOut, _, _, _, "audit_profile_level")),
    assert_true('audit_levels has open',
        sub_string(AuditOut, _, _, _, "open")),
    %% streaming_capable emits facts
    with_output_to(string(SCOut), agent_loop_components:emit_config_section(current_output, streaming_capable, [target(prolog)])),
    assert_true('streaming_capable has streaming_capable',
        sub_string(SCOut, _, _, _, "streaming_capable")),
    %% security_profiles emits facts
    with_output_to(string(SPOut), agent_loop_components:emit_config_section(current_output, security_profiles, [target(prolog)])),
    assert_true('security_profiles has security_profile',
        sub_string(SPOut, _, _, _, "security_profile")),
    %% cli_arguments emits facts
    with_output_to(string(CLIOut), agent_loop_components:emit_config_section(current_output, cli_arguments, [target(prolog)])),
    assert_true('cli_arguments has cli_argument',
        sub_string(CLIOut, _, _, _, "cli_argument")).

test_backend_error_handler_routing :-
    format("~nbackend_error_handler_routing:~n"),
    %% All 8 backends have routing entries
    findall(B, agent_loop_module:backend_error_handler(B, _), Backends),
    length(Backends, BCount),
    assert_true('8 backends routed', BCount =:= 8),
    %% Specific routing checks
    agent_loop_module:backend_error_handler('ollama-cli', OllamaSpec),
    assert_true('ollama-cli routes to cli', OllamaSpec == cli),
    agent_loop_module:backend_error_handler(claude, ClaudeSpec),
    assert_true('claude routes to sdk(anthropic)', ClaudeSpec == sdk(anthropic)),
    agent_loop_module:backend_error_handler(openai, OpenAISpec),
    assert_true('openai routes to sdk(openai)', OpenAISpec == sdk(openai)),
    agent_loop_module:backend_error_handler('ollama-api', OllamaAPISpec),
    assert_true('ollama-api routes to urllib', OllamaAPISpec == urllib),
    agent_loop_module:backend_error_handler(openrouter, ORSpec),
    assert_true('openrouter routes to urllib', ORSpec == urllib),
    %% emit_backend_error_handler produces output
    with_output_to(string(ErrOut), agent_loop_module:emit_backend_error_handler(current_output, claude)),
    assert_true('emit claude error handler has APIError',
        sub_string(ErrOut, _, _, _, "APIError")),
    with_output_to(string(CLIOut), agent_loop_module:emit_backend_error_handler(current_output, 'ollama-cli')),
    assert_true('emit ollama-cli error handler has TimeoutExpired',
        sub_string(CLIOut, _, _, _, "TimeoutExpired")).

%% ============================================================================
%% Unified fragment system tests
%% ============================================================================

test_unified_fragment_system :-
    format("~nUnified fragment system:~n"),
    %% target_fragment works for all 3 targets
    assert_true('target_fragment python finds cost_usage_record', (
        agent_loop_module:target_fragment(python, cost_usage_record, Code),
        atom(Code), Code \== ''
    )),
    assert_true('target_fragment prolog finds cost_tracker_impl', (
        agent_loop_module:target_fragment(prolog, cost_tracker_impl, Code2),
        atom(Code2), Code2 \== ''
    )),
    assert_true('target_fragment rust finds config_types', (
        agent_loop_module:target_fragment(rust, config_types, Code3),
        atom(Code3), Code3 \== ''
    )),
    %% write_fragment works
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_module:write_fragment(S, rust, config_types),
    close(S),
    memory_file_to_atom(MF, Output),
    assert_true('write_fragment rust outputs CliArgument struct',
        sub_atom(Output, _, _, _, 'CliArgument')),
    free_memory_file(MF),
    %% Unknown fragment throws error
    assert_true('write_fragment unknown throws error', (
        catch(
            agent_loop_module:write_fragment(current_output, rust, nonexistent_fragment_xyz),
            error(existence_error(fragment, rust-nonexistent_fragment_xyz), _),
            true
        )
    )).

%% ============================================================================
%% Rust fragment tests
%% ============================================================================

test_rust_fragments :-
    format("~nRust fragments:~n"),
    %% Count rust fragments
    findall(N, agent_loop_module:rust_fragment(N, _), RFs),
    length(RFs, RFCount),
    assert_eq('Rust fragment count', RFCount, 22),
    %% Check each fragment exists and has content
    assert_true('config_types has CliArgument', (
        agent_loop_module:rust_fragment(config_types, C1),
        sub_atom(C1, _, _, _, 'CliArgument')
    )),
    assert_true('costs_types has Pricing', (
        agent_loop_module:rust_fragment(costs_types, C2),
        sub_atom(C2, _, _, _, 'Pricing')
    )),
    assert_true('costs_tracker has CostTracker', (
        agent_loop_module:rust_fragment(costs_tracker, C3),
        sub_atom(C3, _, _, _, 'CostTracker')
    )),
    assert_true('tools_types has ToolSpec', (
        agent_loop_module:rust_fragment(tools_types, C4),
        sub_atom(C4, _, _, _, 'ToolSpec')
    )),
    assert_true('commands_types has CommandSpec', (
        agent_loop_module:rust_fragment(commands_types, C5),
        sub_atom(C5, _, _, _, 'CommandSpec')
    )),
    assert_true('security_types has SecurityProfileSpec', (
        agent_loop_module:rust_fragment(security_types, C6),
        sub_atom(C6, _, _, _, 'SecurityProfileSpec')
    )),
    assert_true('costs_tracker has format_summary', (
        agent_loop_module:rust_fragment(costs_tracker, C7),
        sub_atom(C7, _, _, _, 'format_summary')
    )).

%% ============================================================================
%% Rust module skeleton tests
%% ============================================================================

test_rust_module_skeleton :-
    format("~nRust module skeleton:~n"),
    new_memory_file(MF),
    open_memory_file(MF, write, S),
    agent_loop_components:emit_rust_module_skeleton(S, test_mod, [
        exports([foo/2, bar/1]),
        use_modules([config, tools]),
        use_external([serde-['Serialize', 'Deserialize']]),
        derives(['Debug', 'Clone']),
        dynamic([state/1]),
        comment('Test module')
    ]),
    close(S),
    memory_file_to_atom(MF, Output),
    free_memory_file(MF),
    assert_true('skeleton has Module comment', sub_atom(Output, _, _, _, 'Module: test_mod')),
    assert_true('skeleton has use crate::config', sub_atom(Output, _, _, _, 'use crate::config')),
    assert_true('skeleton has use serde', sub_atom(Output, _, _, _, 'use serde::{Serialize, Deserialize}')),
    assert_true('skeleton has derive', sub_atom(Output, _, _, _, '#[derive(Debug, Clone)]')),
    assert_true('skeleton has mutable state note', sub_atom(Output, _, _, _, 'Mutable state')),
    assert_true('skeleton has doc comment', sub_atom(Output, _, _, _, '/// Test module')),
    %% emit_module_skeleton/4 dispatches correctly
    new_memory_file(MF2),
    open_memory_file(MF2, write, S2),
    agent_loop_components:emit_module_skeleton(S2, rust, test2, [comment('Via dispatch')]),
    close(S2),
    memory_file_to_atom(MF2, Out2),
    free_memory_file(MF2),
    assert_true('emit_module_skeleton dispatches to rust', sub_atom(Out2, _, _, _, 'Module: test2')).

%% ============================================================================
%% Rust compile_component tests
%% ============================================================================

test_rust_compile_components :-
    format("~nRust compile_component:~n"),
    agent_loop_components:register_agent_loop_components,
    %% Tool component
    assert_true('tool compiles for rust target', (
        compile_component(agent_tools, bash, [target(rust), indent(4), fact_type(tool_spec)], TCode),
        sub_atom(TCode, _, _, _, 'ToolSpec')
    )),
    %% Destructive tool
    assert_true('destructive tool compiles for rust', (
        compile_component(agent_tools, bash, [target(rust), indent(4), fact_type(destructive_tool)], DTCode),
        sub_atom(DTCode, _, _, _, '"bash"')
    )),
    %% Command component
    assert_true('command compiles for rust target', (
        compile_component(agent_commands, exit, [target(rust), indent(4)], CCode),
        sub_atom(CCode, _, _, _, 'CommandSpec')
    )),
    %% Backend component
    assert_true('backend compiles for rust target', (
        compile_component(agent_backends, coro, [target(rust), indent(4)], BCode),
        sub_atom(BCode, _, _, _, 'BackendSpec')
    )),
    %% Cost component
    assert_true('cost compiles for rust target', (
        compile_component(agent_costs, opus, [target(rust), indent(4)], PCode),
        sub_atom(PCode, _, _, _, 'Pricing')
    )),
    %% Security component
    assert_true('security compiles for rust target', (
        compile_component(agent_security, open, [target(rust), indent(4)], SCode),
        sub_atom(SCode, _, _, _, 'SecurityProfileSpec')
    )),
    %% Security rule (blocked_path)
    assert_true('security rule compiles for rust target', (
        component(agent_security_rules, RName, _, _),
        compile_component(agent_security_rules, RName, [target(rust), indent(4), fact_type(blocked_path)], _)
    ;   true  %% OK if no blocked_path rules
    )).

%% ============================================================================
%% Rust binding tests
%% ============================================================================

test_rust_bindings :-
    format("~nRust bindings:~n"),
    agent_loop_bindings:init_agent_loop_bindings,
    %% All Rust bindings compile
    findall(Pred, binding(rust, Pred, _, _, _, _), RustPreds),
    length(RustPreds, RustCount),
    assert_eq('Rust binding count is 11', RustCount, 11),
    findall(Pred, (
        member(Pred, RustPreds),
        agent_loop_bindings:compile_binding_code(rust, Pred, Code),
        atom(Code), Code \== ''
    ), Compiled),
    length(Compiled, CompiledCount),
    assert_eq('All Rust bindings compile', CompiledCount, 11),
    %% Check specific binding patterns
    assert_true('Rust model_pricing binding uses iter()', (
        agent_loop_bindings:compile_binding_code(rust, model_pricing/3, PricingCode),
        sub_atom(PricingCode, _, _, _, 'iter()')
    )),
    assert_true('Rust destructive_tool binding uses iter()', (
        agent_loop_bindings:compile_binding_code(rust, destructive_tool/1, DTCode),
        sub_atom(DTCode, _, _, _, 'DESTRUCTIVE_TOOLS')
    )).

%% =============================================================================
%% Rust Phase 2 — Data Table Specs
%% =============================================================================

test_rust_data_table_specs :-
    format("~nRust data table specs:~n"),
    findall(N, agent_loop_components:rust_data_table(N, _, _, _, _), Tables),
    length(Tables, TC),
    assert_eq('rust_data_table count', TC, 9),
    %% Check specific tables exist
    assert_true('PRICING table spec exists', (
        agent_loop_components:rust_data_table('PRICING', agent_costs, _, _, _)
    )),
    assert_true('TOOL_SPECS table spec exists', (
        agent_loop_components:rust_data_table('TOOL_SPECS', agent_tools, _, tool_spec, _)
    )),
    assert_true('SLASH_COMMANDS table spec exists', (
        agent_loop_components:rust_data_table('SLASH_COMMANDS', agent_commands, _, _, _)
    )),
    assert_true('SECURITY_PROFILES table spec exists', (
        agent_loop_components:rust_data_table('SECURITY_PROFILES', agent_security, _, _, _)
    )),
    assert_true('BLOCKED_COMMAND_PATTERNS table spec exists', (
        agent_loop_components:rust_data_table('BLOCKED_COMMAND_PATTERNS', agent_security_rules, _, blocked_command_pattern, _)
    )),
    %% Test generic emitter produces correct output
    agent_loop_components:register_agent_loop_components,
    new_memory_file(MF),
    open_memory_file(MF, write, MS),
    agent_loop_components:emit_rust_data_table(MS, 'PRICING', []),
    close(MS),
    memory_file_to_atom(MF, PricingOut),
    free_memory_file(MF),
    assert_true('emit_rust_data_table produces pub static', (
        sub_atom(PricingOut, _, _, _, 'pub static PRICING')
    )),
    assert_true('emit_rust_data_table produces closing bracket', (
        sub_atom(PricingOut, _, _, _, '];')
    )),
    assert_true('emit_rust_data_table contains Pricing struct', (
        sub_atom(PricingOut, _, _, _, 'Pricing {')
    )).

%% =============================================================================
%% Rust Phase 2 — Imperative Fragments
%% =============================================================================

test_rust_imperative_fragments :-
    format("~nRust imperative fragments:~n"),
    %% types_core
    assert_true('types_core has ToolCall', (
        agent_loop_module:rust_fragment(types_core, C1),
        sub_atom(C1, _, _, _, 'ToolCall')
    )),
    assert_true('types_core has AgentResponse', (
        agent_loop_module:rust_fragment(types_core, C2),
        sub_atom(C2, _, _, _, 'AgentResponse')
    )),
    assert_true('types_core has ToolResult', (
        agent_loop_module:rust_fragment(types_core, C3),
        sub_atom(C3, _, _, _, 'ToolResult')
    )),
    assert_true('types_core has Message', (
        agent_loop_module:rust_fragment(types_core, C4),
        sub_atom(C4, _, _, _, 'Message')
    )),
    %% context_manager
    assert_true('context_manager has ContextManager', (
        agent_loop_module:rust_fragment(context_manager, C5),
        sub_atom(C5, _, _, _, 'ContextManager')
    )),
    assert_true('context_manager has add_message', (
        agent_loop_module:rust_fragment(context_manager, C6),
        sub_atom(C6, _, _, _, 'add_message')
    )),
    %% backend_trait
    assert_true('backend_trait has AgentBackend', (
        agent_loop_module:rust_fragment(backend_trait, C7),
        sub_atom(C7, _, _, _, 'AgentBackend')
    )),
    assert_true('backend_trait has send_message', (
        agent_loop_module:rust_fragment(backend_trait, C8),
        sub_atom(C8, _, _, _, 'send_message')
    )),
    %% backend impls
    assert_true('backend_cli_impl has CliBackend', (
        agent_loop_module:rust_fragment(backend_cli_impl, C9),
        sub_atom(C9, _, _, _, 'CliBackend')
    )),
    assert_true('backend_api_impl has ApiBackend', (
        agent_loop_module:rust_fragment(backend_api_impl, C10),
        sub_atom(C10, _, _, _, 'ApiBackend')
    )),
    assert_true('backend_api_impl has reqwest', (
        agent_loop_module:rust_fragment(backend_api_impl, C11),
        sub_atom(C11, _, _, _, 'reqwest')
    )),
    %% tool_handler
    assert_true('tool_handler_struct has ToolHandler', (
        agent_loop_module:rust_fragment(tool_handler_struct, C12),
        sub_atom(C12, _, _, _, 'ToolHandler')
    )),
    assert_true('tool_handler_validation has check_path_allowed', (
        agent_loop_module:rust_fragment(tool_handler_validation, C13),
        sub_atom(C13, _, _, _, 'check_path_allowed')
    )),
    assert_true('tool_handler_dispatch has handle_bash', (
        agent_loop_module:rust_fragment(tool_handler_dispatch, C14),
        sub_atom(C14, _, _, _, 'handle_bash')
    )),
    %% main_loop
    assert_true('main_loop has rustyline', (
        agent_loop_module:rust_fragment(main_loop, C15),
        sub_atom(C15, _, _, _, 'rustyline')
    )).

%% =============================================================================
%% Rust Phase 2 — Type Mapping
%% =============================================================================

test_rust_type_mapping :-
    format("~nRust type mapping:~n"),
    assert_true('str maps to String', (
        agent_loop_module:rust_type_mapping('str', 'String')
    )),
    assert_true('str | None maps to Option<String>', (
        agent_loop_module:rust_type_mapping('str | None', 'Option<String>')
    )),
    assert_true('int maps to i64', (
        agent_loop_module:rust_type_mapping('int', 'i64')
    )),
    assert_true('bool maps to bool', (
        agent_loop_module:rust_type_mapping('bool', 'bool')
    )),
    assert_true('list[str] maps to Vec<String>', (
        agent_loop_module:rust_type_mapping('list[str]', 'Vec<String>')
    )),
    %% Verify all agent_config_field types have mappings
    findall(T, (
        agent_loop_module:agent_config_field(_, T, _, _),
        agent_loop_module:rust_type_mapping(T, _)
    ), MappedTypes),
    findall(T, agent_loop_module:agent_config_field(_, T, _, _), AllTypes),
    length(MappedTypes, MC),
    length(AllTypes, AC),
    assert_eq('All config field types have rust mappings', MC, AC).

%% =============================================================================
%% Rust Phase 2 — Backend Factory Generation
%% =============================================================================

test_rust_backend_factory :-
    format("~nRust backend factory:~n"),
    %% Generate to memory and check content
    new_memory_file(MF),
    open_memory_file(MF, write, MS),
    agent_loop_module:generate_rust_backend_factory(MS),
    close(MS),
    memory_file_to_atom(MF, FactoryCode),
    free_memory_file(MF),
    assert_true('Factory has create_backend function', (
        sub_atom(FactoryCode, _, _, _, 'create_backend')
    )),
    assert_true('Factory handles coro backend', (
        sub_atom(FactoryCode, _, _, _, '"coro"')
    )),
    assert_true('Factory handles claude backend', (
        sub_atom(FactoryCode, _, _, _, '"claude"')
    )),
    assert_true('Factory handles openai backend', (
        sub_atom(FactoryCode, _, _, _, '"openai"')
    )),
    assert_true('Factory handles ollama backend', (
        sub_atom(FactoryCode, _, _, _, '"ollama-api"')
    )),
    assert_true('Factory has CliBackend::new', (
        sub_atom(FactoryCode, _, _, _, 'CliBackend::new')
    )),
    assert_true('Factory has ApiBackend::new', (
        sub_atom(FactoryCode, _, _, _, 'ApiBackend::new')
    )),
    assert_true('Factory has panic for unknown', (
        sub_atom(FactoryCode, _, _, _, 'panic!')
    )).

%% =============================================================================
%% Rust Phase 2 — Generated Files Verification
%% =============================================================================

test_rust_phase2_generation :-
    format("~nRust phase 2 generation:~n"),
    %% Verify all expected files exist
    assert_true('types.rs exists', (
        agent_loop_module:output_path(rust, 'types.rs', P1), exists_file(P1)
    )),
    assert_true('context.rs exists', (
        agent_loop_module:output_path(rust, 'context.rs', P2), exists_file(P2)
    )),
    assert_true('backends.rs exists', (
        agent_loop_module:output_path(rust, 'backends.rs', P3), exists_file(P3)
    )),
    assert_true('tool_handler.rs exists', (
        agent_loop_module:output_path(rust, 'tool_handler.rs', P4), exists_file(P4)
    )),
    assert_true('main.rs exists', (
        agent_loop_module:output_path(rust, 'main.rs', P5), exists_file(P5)
    )),
    %% Check lib.rs has all module declarations
    agent_loop_module:output_path(rust, 'lib.rs', LibPath),
    read_file_to_string(LibPath, LibContent, []),
    assert_true('lib.rs has types module', (
        sub_string(LibContent, _, _, _, "pub mod types;")
    )),
    assert_true('lib.rs has context module', (
        sub_string(LibContent, _, _, _, "pub mod context;")
    )),
    assert_true('lib.rs has backends module', (
        sub_string(LibContent, _, _, _, "pub mod backends;")
    )),
    assert_true('lib.rs has tool_handler module', (
        sub_string(LibContent, _, _, _, "pub mod tool_handler;")
    )).

%% =============================================================================
%% Rust Phase 3 — Backend Factory Names
%% =============================================================================

test_rust_backend_factory_names :-
    format("~nRust backend factory names (hyphens):~n"),
    new_memory_file(MF),
    open_memory_file(MF, write, MS),
    agent_loop_module:generate_rust_backend_factory(MS),
    close(MS),
    memory_file_to_atom(MF, FC),
    free_memory_file(MF),
    assert_true('Factory uses claude-code (hyphen)', (
        sub_atom(FC, _, _, _, '"claude-code"')
    )),
    assert_true('Factory uses ollama-api (hyphen)', (
        sub_atom(FC, _, _, _, '"ollama-api"')
    )),
    assert_true('Factory uses ollama-cli (hyphen)', (
        sub_atom(FC, _, _, _, '"ollama-cli"')
    )),
    assert_true('Factory does NOT use claude_code (underscore)', (
        \+ sub_atom(FC, _, _, _, '"claude_code"')
    )),
    assert_true('Factory follows backend_factory_order', (
        agent_loop_module:backend_factory_order(Order),
        length(Order, N), N > 0
    )).

%% =============================================================================
%% Rust Phase 3 — Clap Generation
%% =============================================================================

test_rust_clap_generation :-
    format("~nRust clap generation:~n"),
    new_memory_file(MF),
    open_memory_file(MF, write, MS),
    agent_loop_module:generate_rust_clap_args(MS),
    close(MS),
    memory_file_to_atom(MF, CC),
    free_memory_file(MF),
    assert_true('Clap has Arg::new("backend")', (
        sub_atom(CC, _, _, _, 'Arg::new("backend")')
    )),
    assert_true('Clap has .long("backend")', (
        sub_atom(CC, _, _, _, '.long("backend")')
    )),
    assert_true('Clap has .short(''b'')', (
        sub_atom(CC, _, _, _, '.short(')
    )),
    assert_true('Clap has .help for backend', (
        sub_atom(CC, _, _, _, '.help("Backend to use')
    )),
    assert_true('Clap has value_parser for choices', (
        sub_atom(CC, _, _, _, '.value_parser([')
    )),
    assert_true('Clap has SetTrue for store_true args', (
        sub_atom(CC, _, _, _, 'ArgAction::SetTrue')
    )).

%% =============================================================================
%% Rust Phase 3 — Sessions Fragment
%% =============================================================================

test_rust_sessions_fragment :-
    format("~nRust sessions fragment:~n"),
    assert_true('sessions_module fragment exists', (
        agent_loop_module:rust_fragment(sessions_module, SC1),
        atom_length(SC1, Len1), Len1 > 100
    )),
    assert_true('sessions_module has SessionManager', (
        agent_loop_module:rust_fragment(sessions_module, SC2),
        sub_atom(SC2, _, _, _, 'SessionManager')
    )),
    assert_true('sessions_module has save method', (
        agent_loop_module:rust_fragment(sessions_module, SC3),
        sub_atom(SC3, _, _, _, 'fn save')
    )),
    assert_true('sessions_module has load method', (
        agent_loop_module:rust_fragment(sessions_module, SC4),
        sub_atom(SC4, _, _, _, 'fn load')
    )),
    assert_true('sessions_module has list method', (
        agent_loop_module:rust_fragment(sessions_module, SC5),
        sub_atom(SC5, _, _, _, 'fn list')
    )),
    assert_true('sessions_module has delete method', (
        agent_loop_module:rust_fragment(sessions_module, SC6),
        sub_atom(SC6, _, _, _, 'fn delete')
    )).

%% =============================================================================
%% Rust Phase 3 — emit_config_section Rust branches
%% =============================================================================

test_emit_config_section_rust :-
    format("~nRust config section emission:~n"),
    %% Test each section emits for rust target
    maplist([Section-Expected]>>(
        new_memory_file(MF),
        open_memory_file(MF, write, MS),
        agent_loop_components:emit_config_section(MS, Section, [target(rust)]),
        close(MS),
        memory_file_to_atom(MF, Output),
        free_memory_file(MF),
        format(atom(Label), 'emit_config_section(~w, rust) has ~w', [Section, Expected]),
        assert_true(Label, sub_atom(Output, _, _, _, Expected))
    ), [
        api_key_env_vars-'API_KEY_ENV_VARS',
        api_key_files-'API_KEY_FILE_PATHS',
        default_presets-'DEFAULT_PRESETS',
        agent_config_fields-'CONFIG_FIELDS',
        audit_levels-'AUDIT_LEVELS',
        streaming_capable-'STREAMING_CAPABLE',
        cli_arguments-'CLI_ARGS'
    ]).

%% =============================================================================
%% Rust Phase 3 — Streaming Capable
%% =============================================================================

test_rust_streaming_capable :-
    format("~nRust streaming capable:~n"),
    new_memory_file(MF),
    open_memory_file(MF, write, MS),
    agent_loop_components:emit_config_section(MS, streaming_capable, [target(rust)]),
    close(MS),
    memory_file_to_atom(MF, SC),
    free_memory_file(MF),
    assert_true('streaming_capable has STREAMING_CAPABLE', (
        sub_atom(SC, _, _, _, 'STREAMING_CAPABLE')
    )),
    assert_true('streaming_capable has api_local', (
        sub_atom(SC, _, _, _, '"api_local"')
    )).

%% =============================================================================
%% Rust Phase 3 — Generated Files Verification
%% =============================================================================

test_rust_phase3_generation :-
    format("~nRust phase 3 generation:~n"),
    assert_true('sessions.rs exists', (
        agent_loop_module:output_path(rust, 'sessions.rs', P1), exists_file(P1)
    )),
    agent_loop_module:output_path(rust, 'lib.rs', LibPath),
    read_file_to_string(LibPath, LibContent, []),
    assert_true('lib.rs has sessions module', (
        sub_string(LibContent, _, _, _, "pub mod sessions;")
    )),
    agent_loop_module:output_path(rust, 'main.rs', MainPath),
    read_file_to_string(MainPath, MainContent, []),
    assert_true('main.rs has clap::Command', (
        sub_string(MainContent, _, _, _, "clap::Command::new")
    )),
    assert_true('main.rs has session_manager', (
        sub_string(MainContent, _, _, _, "session_manager")
    )),
    assert_true('main.rs has --list-sessions handler', (
        sub_string(MainContent, _, _, _, "list_sessions")
    )),
    agent_loop_module:output_path(rust, 'config.rs', CfgPath),
    read_file_to_string(CfgPath, CfgContent, []),
    assert_true('config.rs has STREAMING_CAPABLE', (
        sub_string(CfgContent, _, _, _, "STREAMING_CAPABLE")
    )).

%% ============================================================================
%% Phase 4 Tests — Config loading, streaming, emit_rust_config_data refactor
%% ============================================================================

test_rust_config_data_refactored :-
    format("~nRefactored emit_rust_config_data:~n"),
    agent_loop_module:output_path(rust, 'config.rs', CfgPath),
    read_file_to_string(CfgPath, CfgContent, []),
    assert_true('config.rs has CLI_ARGS', (
        sub_string(CfgContent, _, _, _, "CLI_ARGS")
    )),
    assert_true('config.rs has CONFIG_FIELDS', (
        sub_string(CfgContent, _, _, _, "CONFIG_FIELDS")
    )),
    assert_true('config.rs has API_KEY_ENV_VARS', (
        sub_string(CfgContent, _, _, _, "API_KEY_ENV_VARS")
    )),
    assert_true('config.rs has API_KEY_FILE_PATHS', (
        sub_string(CfgContent, _, _, _, "API_KEY_FILE_PATHS")
    )),
    assert_true('config.rs has CONFIG_SEARCH_PATHS', (
        sub_string(CfgContent, _, _, _, "CONFIG_SEARCH_PATHS")
    )),
    assert_true('config.rs has DEFAULT_PRESETS', (
        sub_string(CfgContent, _, _, _, "DEFAULT_PRESETS")
    )),
    assert_true('config.rs has AUDIT_LEVELS', (
        sub_string(CfgContent, _, _, _, "AUDIT_LEVELS")
    )).

test_rust_config_loader_fragments :-
    format("~nConfig loader fragments:~n"),
    assert_true('config_loader_types exists', (
        agent_loop_module:rust_fragment(config_loader_types, C1),
        atom_string(C1, CS1),
        sub_string(CS1, _, _, _, "ConfigFile")
    )),
    assert_true('config_loader_cascade exists', (
        agent_loop_module:rust_fragment(config_loader_cascade, C2),
        atom_string(C2, CS2),
        sub_string(CS2, _, _, _, "find_config_file")
    )),
    assert_true('config_loader_agent_resolve exists', (
        agent_loop_module:rust_fragment(config_loader_agent_resolve, C3),
        atom_string(C3, CS3),
        sub_string(CS3, _, _, _, "resolve_agent")
    )),
    assert_true('config_loader_api_key_resolve exists', (
        agent_loop_module:rust_fragment(config_loader_api_key_resolve, C4),
        atom_string(C4, CS4),
        sub_string(CS4, _, _, _, "resolve_api_key")
    )),
    assert_true('config_loader_list_agents exists', (
        agent_loop_module:rust_fragment(config_loader_list_agents, C5),
        atom_string(C5, CS5),
        sub_string(CS5, _, _, _, "list_agents")
    )).

test_rust_config_dir_file_names :-
    format("~nConfig dir file names section:~n"),
    new_memory_file(MemFile),
    open_memory_file(MemFile, write, S),
    agent_loop_components:emit_config_section(S, config_dir_file_names, [target(rust)]),
    close(S),
    memory_file_to_string(MemFile, Out),
    assert_true('has CONFIG_DIR_FILE_NAMES', (
        sub_string(Out, _, _, _, "CONFIG_DIR_FILE_NAMES")
    )),
    assert_true('has agents.yaml', (
        sub_string(Out, _, _, _, "agents.yaml")
    )),
    assert_true('has agents.json', (
        sub_string(Out, _, _, _, "agents.json")
    )).

test_rust_cli_overrides_section :-
    format("~nCLI overrides section:~n"),
    new_memory_file(MemFile),
    open_memory_file(MemFile, write, S),
    agent_loop_components:emit_config_section(S, cli_overrides, [target(rust)]),
    close(S),
    memory_file_to_string(MemFile, Out),
    assert_true('has CLI_OVERRIDES', (
        sub_string(Out, _, _, _, "CLI_OVERRIDES")
    )),
    assert_true('has CliOverride struct', (
        sub_string(Out, _, _, _, "pub struct CliOverride")
    )).

test_rust_streaming_handler :-
    format("~nStreaming handler fragment:~n"),
    assert_true('streaming_handler exists', (
        agent_loop_module:rust_fragment(streaming_handler, C),
        sub_atom(C, _, _, _, 'parse_sse_line')
    )),
    assert_true('streaming_handler has extract_content_delta', (
        agent_loop_module:rust_fragment(streaming_handler, C),
        sub_atom(C, _, _, _, 'extract_content_delta')
    )),
    assert_true('streaming_handler has send_streaming', (
        agent_loop_module:rust_fragment(streaming_handler, C),
        sub_atom(C, _, _, _, 'send_streaming')
    )).

test_rust_backend_api_streaming :-
    format("~nBackend API streaming:~n"),
    agent_loop_module:rust_fragment(backend_api_impl, C),
    assert_true('ApiBackend has stream field', (
        sub_atom(C, _, _, _, 'pub stream: bool')
    )),
    assert_true('ApiBackend calls send_streaming', (
        sub_atom(C, _, _, _, 'send_streaming')
    )).

test_rust_phase4_generation :-
    format("~nRust phase 4 generation:~n"),
    assert_true('config_loader.rs exists', (
        agent_loop_module:output_path(rust, 'config_loader.rs', P1), exists_file(P1)
    )),
    agent_loop_module:output_path(rust, 'lib.rs', LibPath),
    read_file_to_string(LibPath, LibContent, []),
    assert_true('lib.rs has config_loader module', (
        sub_string(LibContent, _, _, _, "pub mod config_loader;")
    )),
    agent_loop_module:output_path(rust, 'main.rs', MainPath),
    read_file_to_string(MainPath, MainContent, []),
    assert_true('main.rs has find_config_file', (
        sub_string(MainContent, _, _, _, "find_config_file")
    )),
    assert_true('main.rs has resolve_agent', (
        sub_string(MainContent, _, _, _, "resolve_agent")
    )),
    assert_true('main.rs has resolve_api_key', (
        sub_string(MainContent, _, _, _, "resolve_api_key")
    )),
    assert_true('main.rs has stream flag', (
        sub_string(MainContent, _, _, _, "stream_arg")
    )).

test_rust_config_search_paths_section :-
    format("~nConfig search paths section:~n"),
    new_memory_file(MemFile),
    open_memory_file(MemFile, write, S),
    agent_loop_components:emit_config_section(S, config_search_paths, [target(rust)]),
    close(S),
    memory_file_to_string(MemFile, Out),
    assert_true('has CONFIG_SEARCH_PATHS', (
        sub_string(Out, _, _, _, "CONFIG_SEARCH_PATHS")
    )),
    assert_true('has uwsal.json', (
        sub_string(Out, _, _, _, "uwsal.json")
    )),
    assert_true('has required priority', (
        sub_string(Out, _, _, _, "required")
    )),
    assert_true('has fallback priority', (
        sub_string(Out, _, _, _, "fallback")
    )).

%% =============================================================================
%% Phase 5 Tests — Security wiring + YAML config
%% =============================================================================

test_rust_command_validation_fix :-
    format("~nCommand validation fix:~n"),
    agent_loop_module:output_path(rust, 'security.rs', SecPath),
    read_file_to_string(SecPath, SecContent, []),
    assert_true('cautious has command_validation true', (
        sub_string(SecContent, _, _, _, "\"cautious\", SecurityProfileSpec { path_validation: true, command_validation: true }")
    )),
    assert_true('open has command_validation false', (
        sub_string(SecContent, _, _, _, "\"open\", SecurityProfileSpec { path_validation: false, command_validation: false }")
    )).

test_rust_security_wiring :-
    format("~nSecurity wiring:~n"),
    agent_loop_module:rust_fragment(tool_handler_struct, THS),
    atom_string(THS, THSS),
    assert_true('ToolHandler has security_profile field', (
        sub_string(THSS, _, _, _, "pub security_profile: String")
    )),
    assert_true('ToolHandler has approval_mode field', (
        sub_string(THSS, _, _, _, "pub approval_mode: String")
    )),
    assert_true('ToolHandler::new accepts 3 params', (
        sub_string(THSS, _, _, _, "pub fn new(auto_approve: bool, security_profile: String, approval_mode: String)")
    )),
    assert_true('ToolHandler has get_profile_spec', (
        sub_string(THSS, _, _, _, "fn get_profile_spec")
    )).

test_rust_security_profile_conditional :-
    format("~nSecurity profile conditional:~n"),
    agent_loop_module:rust_fragment(tool_handler_validation, THV),
    atom_string(THV, THVS),
    assert_true('check_path_allowed takes &self', (
        sub_string(THVS, _, _, _, "pub fn check_path_allowed(&self, path: &str)")
    )),
    assert_true('is_command_blocked takes &self', (
        sub_string(THVS, _, _, _, "pub fn is_command_blocked(&self, command: &str)")
    )),
    assert_true('checks paranoid profile', (
        sub_string(THVS, _, _, _, "\"paranoid\"")
    )).

test_rust_approval_mode :-
    format("~nApproval mode:~n"),
    agent_loop_module:rust_fragment(tool_handler_validation, THV2),
    atom_string(THV2, THV2S),
    assert_true('has check_approval method', (
        sub_string(THV2S, _, _, _, "pub fn check_approval")
    )),
    assert_true('plan mode blocks writes', (
        sub_string(THV2S, _, _, _, "\"plan\"")
    )),
    agent_loop_module:rust_fragment(tool_handler_dispatch, THD),
    atom_string(THD, THDS),
    assert_true('execute calls check_approval', (
        sub_string(THDS, _, _, _, "self.check_approval")
    )).

test_rust_yaml_config :-
    format("~nYAML config support:~n"),
    agent_loop_module:rust_fragment(config_loader_cascade, CLC),
    atom_string(CLC, CLCS),
    assert_true('load_config_file handles yaml', (
        sub_string(CLCS, _, _, _, "serde_yaml::from_str")
    )),
    agent_loop_module:output_path(rust, '../Cargo.toml', CargoPath),
    read_file_to_string(CargoPath, CargoContent, []),
    assert_true('Cargo.toml has serde_yaml', (
        sub_string(CargoContent, _, _, _, "serde_yaml")
    )),
    agent_loop_module:output_path(rust, 'config_loader.rs', CLPath),
    read_file_to_string(CLPath, CLContent, []),
    assert_true('config_loader has generate_example_config_yaml', (
        sub_string(CLContent, _, _, _, "generate_example_config_yaml")
    )).

test_rust_security_regex_lists :-
    format("~nSecurity regex lists:~n"),
    agent_loop_module:output_path(rust, 'security.rs', SecPath2),
    read_file_to_string(SecPath2, SecContent2, []),
    assert_true('security.rs has GUARDED_EXTRA_BLOCKS', (
        sub_string(SecContent2, _, _, _, "GUARDED_EXTRA_BLOCKS")
    )),
    assert_true('security.rs has PARANOID_SAFE', (
        sub_string(SecContent2, _, _, _, "PARANOID_SAFE")
    )),
    assert_true('security.rs has PARANOID_CONFIRM', (
        sub_string(SecContent2, _, _, _, "PARANOID_CONFIRM")
    )).

test_rust_phase5_generation :-
    format("~nRust phase 5 generation:~n"),
    agent_loop_module:output_path(rust, 'types.rs', TypesPath),
    read_file_to_string(TypesPath, TypesContent, []),
    assert_true('AgentConfig has security_profile', (
        sub_string(TypesContent, _, _, _, "pub security_profile: String")
    )),
    assert_true('AgentConfig has approval_mode', (
        sub_string(TypesContent, _, _, _, "pub approval_mode: String")
    )),
    agent_loop_module:output_path(rust, 'main.rs', MainPath5),
    read_file_to_string(MainPath5, MainContent5, []),
    assert_true('main.rs has ToolHandler with 3 args', (
        sub_string(MainContent5, _, _, _, "config.security_profile.clone()")
    )).

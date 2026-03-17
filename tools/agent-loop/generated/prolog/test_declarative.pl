%% Auto-generated declarative tests — do not edit manually.
%% Regenerate with: swipl -g "generate_all(tests), halt" agent_loop_module.pl

:- use_module(library(lists)).

:- dynamic decl_test_passed/1, decl_test_failed/1.

decl_assert_true(Name, Goal) :-
    (call(Goal) ->
        assert(decl_test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(decl_test_failed(Name)),
        format("  [FAIL] ~w~n", [Name])
    ).

decl_assert_eq(Name, Got, Expected) :-
    (Got == Expected ->
        assert(decl_test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(decl_test_failed(Name)),
        format("  [FAIL] ~w (expected ~w, got ~w)~n", [Name, Expected, Got])
    ).

%% =============================================================================
%% Existence Tests
%% =============================================================================

test_decl_existence :-
    format("~nDeclarative existence tests:~n"),
    decl_assert_true('tool_spec(bash) exists', agent_loop_module:tool_spec(bash, _)),
    decl_assert_true('tool_spec(read) exists', agent_loop_module:tool_spec(read, _)),
    decl_assert_true('tool_spec(write) exists', agent_loop_module:tool_spec(write, _)),
    decl_assert_true('tool_spec(edit) exists', agent_loop_module:tool_spec(edit, _)),
    decl_assert_true('agent_backend(coro) exists', agent_loop_module:agent_backend(coro, _)),
    decl_assert_true('agent_backend(claude_code) exists', agent_loop_module:agent_backend(claude_code, _)),
    decl_assert_true('agent_backend(gemini) exists', agent_loop_module:agent_backend(gemini, _)),
    decl_assert_true('agent_backend(claude_api) exists', agent_loop_module:agent_backend(claude_api, _)),
    decl_assert_true('agent_backend(openai_api) exists', agent_loop_module:agent_backend(openai_api, _)),
    decl_assert_true('agent_backend(ollama_api) exists', agent_loop_module:agent_backend(ollama_api, _)),
    decl_assert_true('agent_backend(ollama_cli) exists', agent_loop_module:agent_backend(ollama_cli, _)),
    decl_assert_true('agent_backend(openrouter_api) exists', agent_loop_module:agent_backend(openrouter_api, _)),
    decl_assert_true('security_profile(open) exists', agent_loop_module:security_profile(open, _)),
    decl_assert_true('security_profile(cautious) exists', agent_loop_module:security_profile(cautious, _)),
    decl_assert_true('security_profile(guarded) exists', agent_loop_module:security_profile(guarded, _)),
    decl_assert_true('security_profile(paranoid) exists', agent_loop_module:security_profile(paranoid, _)),
    decl_assert_true('agent_config_field(name) exists', agent_loop_module:agent_config_field(name, _, _, _)),
    decl_assert_true('agent_config_field(backend) exists', agent_loop_module:agent_config_field(backend, _, _, _)),
    decl_assert_true('agent_config_field(model) exists', agent_loop_module:agent_config_field(model, _, _, _)),
    decl_assert_true('agent_config_field(host) exists', agent_loop_module:agent_config_field(host, _, _, _)),
    decl_assert_true('agent_config_field(port) exists', agent_loop_module:agent_config_field(port, _, _, _)),
    decl_assert_true('agent_config_field(api_key) exists', agent_loop_module:agent_config_field(api_key, _, _, _)),
    decl_assert_true('agent_config_field(command) exists', agent_loop_module:agent_config_field(command, _, _, _)),
    decl_assert_true('agent_config_field(system_prompt) exists', agent_loop_module:agent_config_field(system_prompt, _, _, _)),
    decl_assert_true('agent_config_field(agent_md) exists', agent_loop_module:agent_config_field(agent_md, _, _, _)),
    decl_assert_true('agent_config_field(tools) exists', agent_loop_module:agent_config_field(tools, _, _, _)),
    decl_assert_true('agent_config_field(auto_tools) exists', agent_loop_module:agent_config_field(auto_tools, _, _, _)),
    decl_assert_true('agent_config_field(context_mode) exists', agent_loop_module:agent_config_field(context_mode, _, _, _)),
    decl_assert_true('agent_config_field(max_context_tokens) exists', agent_loop_module:agent_config_field(max_context_tokens, _, _, _)),
    decl_assert_true('agent_config_field(max_messages) exists', agent_loop_module:agent_config_field(max_messages, _, _, _)),
    decl_assert_true('agent_config_field(max_chars) exists', agent_loop_module:agent_config_field(max_chars, _, _, _)),
    decl_assert_true('agent_config_field(max_words) exists', agent_loop_module:agent_config_field(max_words, _, _, _)),
    decl_assert_true('agent_config_field(skills) exists', agent_loop_module:agent_config_field(skills, _, _, _)),
    decl_assert_true('agent_config_field(max_iterations) exists', agent_loop_module:agent_config_field(max_iterations, _, _, _)),
    decl_assert_true('agent_config_field(timeout) exists', agent_loop_module:agent_config_field(timeout, _, _, _)),
    decl_assert_true('agent_config_field(show_tokens) exists', agent_loop_module:agent_config_field(show_tokens, _, _, _)),
    decl_assert_true('agent_config_field(stream) exists', agent_loop_module:agent_config_field(stream, _, _, _)),
    decl_assert_true('agent_config_field(security_profile) exists', agent_loop_module:agent_config_field(security_profile, _, _, _)),
    decl_assert_true('agent_config_field(approval_mode) exists', agent_loop_module:agent_config_field(approval_mode, _, _, _)),
    decl_assert_true('agent_config_field(paste_mode) exists', agent_loop_module:agent_config_field(paste_mode, _, _, _)),
    decl_assert_true('agent_config_field(tool_cache_ttl) exists', agent_loop_module:agent_config_field(tool_cache_ttl, _, _, _)),
    decl_assert_true('agent_config_field(mcp_servers) exists', agent_loop_module:agent_config_field(mcp_servers, _, _, _)),
    decl_assert_true('agent_config_field(token_budget) exists', agent_loop_module:agent_config_field(token_budget, _, _, _)),
    decl_assert_true('agent_config_field(extra) exists', agent_loop_module:agent_config_field(extra, _, _, _)),
    decl_assert_true('slash_command(exit) exists', agent_loop_module:slash_command(exit, _, _, _)),
    decl_assert_true('slash_command(clear) exists', agent_loop_module:slash_command(clear, _, _, _)),
    decl_assert_true('slash_command(help) exists', agent_loop_module:slash_command(help, _, _, _)),
    decl_assert_true('slash_command(status) exists', agent_loop_module:slash_command(status, _, _, _)),
    decl_assert_true('slash_command(iterations) exists', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('slash_command(backend) exists', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('slash_command(save) exists', agent_loop_module:slash_command(save, _, _, _)),
    decl_assert_true('slash_command(load) exists', agent_loop_module:slash_command(load, _, _, _)),
    decl_assert_true('slash_command(sessions) exists', agent_loop_module:slash_command(sessions, _, _, _)),
    decl_assert_true('slash_command(format) exists', agent_loop_module:slash_command(format, _, _, _)),
    decl_assert_true('slash_command(export) exists', agent_loop_module:slash_command(export, _, _, _)),
    decl_assert_true('slash_command(cost) exists', agent_loop_module:slash_command(cost, _, _, _)),
    decl_assert_true('slash_command(search) exists', agent_loop_module:slash_command(search, _, _, _)),
    decl_assert_true('slash_command(stream) exists', agent_loop_module:slash_command(stream, _, _, _)),
    decl_assert_true('slash_command(model) exists', agent_loop_module:slash_command(model, _, _, _)),
    decl_assert_true('slash_command(tokens) exists', agent_loop_module:slash_command(tokens, _, _, _)),
    decl_assert_true('slash_command(multiline) exists', agent_loop_module:slash_command(multiline, _, _, _)),
    decl_assert_true('slash_command(aliases) exists', agent_loop_module:slash_command(aliases, _, _, _)),
    decl_assert_true('slash_command(templates) exists', agent_loop_module:slash_command(templates, _, _, _)),
    decl_assert_true('slash_command(history) exists', agent_loop_module:slash_command(history, _, _, _)),
    decl_assert_true('slash_command(undo) exists', agent_loop_module:slash_command(undo, _, _, _)),
    decl_assert_true('slash_command(delete) exists', agent_loop_module:slash_command(delete, _, _, _)),
    decl_assert_true('slash_command(edit) exists', agent_loop_module:slash_command(edit, _, _, _)),
    decl_assert_true('slash_command(replay) exists', agent_loop_module:slash_command(replay, _, _, _)),
    decl_assert_true('slash_command(init) exists', agent_loop_module:slash_command(init, _, _, _)),
    decl_assert_true('slash_command(reload) exists', agent_loop_module:slash_command(reload, _, _, _)),
    decl_assert_true('slash_command(clear-cache) exists', agent_loop_module:slash_command('clear-cache', _, _, _)),
    true.

%% =============================================================================
%% Property Tests
%% =============================================================================

test_prop_tool_bash :-
    decl_assert_true('tool_spec(bash) has description', (agent_loop_module:tool_spec(bash, P), member(description(_), P))),
    decl_assert_true('tool_spec(bash) has parameters', (agent_loop_module:tool_spec(bash, P), member(parameters(_), P))).

test_prop_tool_read :-
    decl_assert_true('tool_spec(read) has description', (agent_loop_module:tool_spec(read, P), member(description(_), P))),
    decl_assert_true('tool_spec(read) has parameters', (agent_loop_module:tool_spec(read, P), member(parameters(_), P))).

test_prop_tool_write :-
    decl_assert_true('tool_spec(write) has description', (agent_loop_module:tool_spec(write, P), member(description(_), P))),
    decl_assert_true('tool_spec(write) has parameters', (agent_loop_module:tool_spec(write, P), member(parameters(_), P))).

test_prop_tool_edit :-
    decl_assert_true('tool_spec(edit) has description', (agent_loop_module:tool_spec(edit, P), member(description(_), P))),
    decl_assert_true('tool_spec(edit) has parameters', (agent_loop_module:tool_spec(edit, P), member(parameters(_), P))).

test_prop_backend_coro :-
    decl_assert_true('backend(coro) has type', (agent_loop_module:agent_backend(coro, P), member(type(_), P))),
    decl_assert_true('backend(coro) has class_name', (agent_loop_module:agent_backend(coro, P), member(class_name(_), P))).

test_prop_backend_claude_code :-
    decl_assert_true('backend(claude_code) has type', (agent_loop_module:agent_backend(claude_code, P), member(type(_), P))),
    decl_assert_true('backend(claude_code) has class_name', (agent_loop_module:agent_backend(claude_code, P), member(class_name(_), P))).

test_prop_backend_gemini :-
    decl_assert_true('backend(gemini) has type', (agent_loop_module:agent_backend(gemini, P), member(type(_), P))),
    decl_assert_true('backend(gemini) has class_name', (agent_loop_module:agent_backend(gemini, P), member(class_name(_), P))).

test_prop_backend_claude_api :-
    decl_assert_true('backend(claude_api) has type', (agent_loop_module:agent_backend(claude_api, P), member(type(_), P))),
    decl_assert_true('backend(claude_api) has class_name', (agent_loop_module:agent_backend(claude_api, P), member(class_name(_), P))).

test_prop_backend_openai_api :-
    decl_assert_true('backend(openai_api) has type', (agent_loop_module:agent_backend(openai_api, P), member(type(_), P))),
    decl_assert_true('backend(openai_api) has class_name', (agent_loop_module:agent_backend(openai_api, P), member(class_name(_), P))).

test_prop_backend_ollama_api :-
    decl_assert_true('backend(ollama_api) has type', (agent_loop_module:agent_backend(ollama_api, P), member(type(_), P))),
    decl_assert_true('backend(ollama_api) has class_name', (agent_loop_module:agent_backend(ollama_api, P), member(class_name(_), P))).

test_prop_backend_ollama_cli :-
    decl_assert_true('backend(ollama_cli) has type', (agent_loop_module:agent_backend(ollama_cli, P), member(type(_), P))),
    decl_assert_true('backend(ollama_cli) has class_name', (agent_loop_module:agent_backend(ollama_cli, P), member(class_name(_), P))).

test_prop_backend_openrouter_api :-
    decl_assert_true('backend(openrouter_api) has type', (agent_loop_module:agent_backend(openrouter_api, P), member(type(_), P))),
    decl_assert_true('backend(openrouter_api) has class_name', (agent_loop_module:agent_backend(openrouter_api, P), member(class_name(_), P))).

test_prop_security_open :-
    decl_assert_true('security(open) has path_validation', (agent_loop_module:security_profile(open, P), member(path_validation(_), P))),
    decl_assert_true('security(open) has audit_logging', (agent_loop_module:security_profile(open, P), member(audit_logging(_), P))).

test_prop_security_cautious :-
    decl_assert_true('security(cautious) has path_validation', (agent_loop_module:security_profile(cautious, P), member(path_validation(_), P))),
    decl_assert_true('security(cautious) has audit_logging', (agent_loop_module:security_profile(cautious, P), member(audit_logging(_), P))).

test_prop_security_guarded :-
    decl_assert_true('security(guarded) has path_validation', (agent_loop_module:security_profile(guarded, P), member(path_validation(_), P))),
    decl_assert_true('security(guarded) has audit_logging', (agent_loop_module:security_profile(guarded, P), member(audit_logging(_), P))).

test_prop_security_paranoid :-
    decl_assert_true('security(paranoid) has path_validation', (agent_loop_module:security_profile(paranoid, P), member(path_validation(_), P))),
    decl_assert_true('security(paranoid) has audit_logging', (agent_loop_module:security_profile(paranoid, P), member(audit_logging(_), P))).

test_decl_properties :-
    format("~nDeclarative property tests:~n"),
    test_prop_tool_bash,
    test_prop_tool_read,
    test_prop_tool_write,
    test_prop_tool_edit,
    test_prop_backend_coro,
    test_prop_backend_claude_code,
    test_prop_backend_gemini,
    test_prop_backend_claude_api,
    test_prop_backend_openai_api,
    test_prop_backend_ollama_api,
    test_prop_backend_ollama_cli,
    test_prop_backend_openrouter_api,
    test_prop_security_open,
    test_prop_security_cautious,
    test_prop_security_guarded,
    test_prop_security_paranoid,
    decl_assert_true('command(exit) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(clear) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(help) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(status) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(iterations) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(backend) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(save) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(load) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(sessions) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(format) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(export) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(cost) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(search) has valid match type', member(prefix, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(stream) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(model) has valid match type', member(prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(tokens) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(multiline) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(aliases) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(templates) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(history) has valid match type', member(exact_or_prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(undo) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(delete) has valid match type', member(prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(edit) has valid match type', member(prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(replay) has valid match type', member(prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(init) has valid match type', member(exact_or_prefix_sp, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(reload) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    decl_assert_true('command(clear-cache) has valid match type', member(exact, [exact, prefix, prefix_sp, exact_or_prefix_sp])),
    test_prop_config_name,
    test_prop_config_backend,
    test_prop_config_model,
    test_prop_config_host,
    test_prop_config_port,
    test_prop_config_api_key,
    test_prop_config_command,
    test_prop_config_system_prompt,
    test_prop_config_agent_md,
    test_prop_config_tools,
    test_prop_config_auto_tools,
    test_prop_config_context_mode,
    test_prop_config_max_context_tokens,
    test_prop_config_max_messages,
    test_prop_config_max_chars,
    test_prop_config_max_words,
    test_prop_config_skills,
    test_prop_config_max_iterations,
    test_prop_config_timeout,
    test_prop_config_show_tokens,
    test_prop_config_stream,
    test_prop_config_security_profile,
    test_prop_config_approval_mode,
    test_prop_config_paste_mode,
    test_prop_config_tool_cache_ttl,
    test_prop_config_mcp_servers,
    test_prop_config_token_budget,
    test_prop_config_extra,
    true.

test_prop_config_name :- decl_assert_true('config(name) has type', (agent_loop_module:agent_config_field(name, T, _, _), T \= '')).
test_prop_config_backend :- decl_assert_true('config(backend) has type', (agent_loop_module:agent_config_field(backend, T, _, _), T \= '')).
test_prop_config_model :- decl_assert_true('config(model) has type', (agent_loop_module:agent_config_field(model, T, _, _), T \= '')).
test_prop_config_host :- decl_assert_true('config(host) has type', (agent_loop_module:agent_config_field(host, T, _, _), T \= '')).
test_prop_config_port :- decl_assert_true('config(port) has type', (agent_loop_module:agent_config_field(port, T, _, _), T \= '')).
test_prop_config_api_key :- decl_assert_true('config(api_key) has type', (agent_loop_module:agent_config_field(api_key, T, _, _), T \= '')).
test_prop_config_command :- decl_assert_true('config(command) has type', (agent_loop_module:agent_config_field(command, T, _, _), T \= '')).
test_prop_config_system_prompt :- decl_assert_true('config(system_prompt) has type', (agent_loop_module:agent_config_field(system_prompt, T, _, _), T \= '')).
test_prop_config_agent_md :- decl_assert_true('config(agent_md) has type', (agent_loop_module:agent_config_field(agent_md, T, _, _), T \= '')).
test_prop_config_tools :- decl_assert_true('config(tools) has type', (agent_loop_module:agent_config_field(tools, T, _, _), T \= '')).
test_prop_config_auto_tools :- decl_assert_true('config(auto_tools) has type', (agent_loop_module:agent_config_field(auto_tools, T, _, _), T \= '')).
test_prop_config_context_mode :- decl_assert_true('config(context_mode) has type', (agent_loop_module:agent_config_field(context_mode, T, _, _), T \= '')).
test_prop_config_max_context_tokens :- decl_assert_true('config(max_context_tokens) has type', (agent_loop_module:agent_config_field(max_context_tokens, T, _, _), T \= '')).
test_prop_config_max_messages :- decl_assert_true('config(max_messages) has type', (agent_loop_module:agent_config_field(max_messages, T, _, _), T \= '')).
test_prop_config_max_chars :- decl_assert_true('config(max_chars) has type', (agent_loop_module:agent_config_field(max_chars, T, _, _), T \= '')).
test_prop_config_max_words :- decl_assert_true('config(max_words) has type', (agent_loop_module:agent_config_field(max_words, T, _, _), T \= '')).
test_prop_config_skills :- decl_assert_true('config(skills) has type', (agent_loop_module:agent_config_field(skills, T, _, _), T \= '')).
test_prop_config_max_iterations :- decl_assert_true('config(max_iterations) has type', (agent_loop_module:agent_config_field(max_iterations, T, _, _), T \= '')).
test_prop_config_timeout :- decl_assert_true('config(timeout) has type', (agent_loop_module:agent_config_field(timeout, T, _, _), T \= '')).
test_prop_config_show_tokens :- decl_assert_true('config(show_tokens) has type', (agent_loop_module:agent_config_field(show_tokens, T, _, _), T \= '')).
test_prop_config_stream :- decl_assert_true('config(stream) has type', (agent_loop_module:agent_config_field(stream, T, _, _), T \= '')).
test_prop_config_security_profile :- decl_assert_true('config(security_profile) has type', (agent_loop_module:agent_config_field(security_profile, T, _, _), T \= '')).
test_prop_config_approval_mode :- decl_assert_true('config(approval_mode) has type', (agent_loop_module:agent_config_field(approval_mode, T, _, _), T \= '')).
test_prop_config_paste_mode :- decl_assert_true('config(paste_mode) has type', (agent_loop_module:agent_config_field(paste_mode, T, _, _), T \= '')).
test_prop_config_tool_cache_ttl :- decl_assert_true('config(tool_cache_ttl) has type', (agent_loop_module:agent_config_field(tool_cache_ttl, T, _, _), T \= '')).
test_prop_config_mcp_servers :- decl_assert_true('config(mcp_servers) has type', (agent_loop_module:agent_config_field(mcp_servers, T, _, _), T \= '')).
test_prop_config_token_budget :- decl_assert_true('config(token_budget) has type', (agent_loop_module:agent_config_field(token_budget, T, _, _), T \= '')).
test_prop_config_extra :- decl_assert_true('config(extra) has type', (agent_loop_module:agent_config_field(extra, T, _, _), T \= '')).

%% =============================================================================
%% Cross-Reference Tests
%% =============================================================================

test_decl_crossrefs :-
    format("~nDeclarative cross-reference tests:~n"),
    decl_assert_true('alias(q) -> command(exit)', agent_loop_module:slash_command(exit, _, _, _)),
    decl_assert_true('alias(x) -> command(exit)', agent_loop_module:slash_command(exit, _, _, _)),
    decl_assert_true('alias(h) -> command(help)', agent_loop_module:slash_command(help, _, _, _)),
    decl_assert_true('alias(?) -> command(help)', agent_loop_module:slash_command(help, _, _, _)),
    decl_assert_true('alias(c) -> command(clear)', agent_loop_module:slash_command(clear, _, _, _)),
    decl_assert_true('alias(s) -> command(status)', agent_loop_module:slash_command(status, _, _, _)),
    decl_assert_true('alias(sv) -> command(save)', agent_loop_module:slash_command(save, _, _, _)),
    decl_assert_true('alias(ld) -> command(load)', agent_loop_module:slash_command(load, _, _, _)),
    decl_assert_true('alias(ls) -> command(sessions)', agent_loop_module:slash_command(sessions, _, _, _)),
    decl_assert_true('alias(exp) -> command(export)', agent_loop_module:slash_command(export, _, _, _)),
    decl_assert_true('alias(md) -> command(export)', agent_loop_module:slash_command(export, _, _, _)),
    decl_assert_true('alias(html) -> command(export)', agent_loop_module:slash_command(export, _, _, _)),
    decl_assert_true('alias(be) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(sw) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(yolo) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(opus) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(sonnet) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(haiku) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(gpt) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(local) -> command(backend)', agent_loop_module:slash_command(backend, _, _, _)),
    decl_assert_true('alias(cc) -> command(clear-cache)', agent_loop_module:slash_command('clear-cache', _, _, _)),
    decl_assert_true('alias(fmt) -> command(format)', agent_loop_module:slash_command(format, _, _, _)),
    decl_assert_true('alias(iter) -> command(iterations)', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('alias(i0) -> command(iterations)', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('alias(i1) -> command(iterations)', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('alias(i3) -> command(iterations)', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('alias(i5) -> command(iterations)', agent_loop_module:slash_command(iterations, _, _, _)),
    decl_assert_true('alias(str) -> command(stream)', agent_loop_module:slash_command(stream, _, _, _)),
    decl_assert_true('alias($) -> command(cost)', agent_loop_module:slash_command(cost, _, _, _)),
    decl_assert_true('alias(find) -> command(search)', agent_loop_module:slash_command(search, _, _, _)),
    decl_assert_true('alias(grep) -> command(search)', agent_loop_module:slash_command(search, _, _, _)),
    decl_assert_true('backend(claude_code) helper fragment stream_json_return exists', agent_loop_module:py_fragment(stream_json_return, _)),
    decl_assert_true('backend(claude_code) helper fragment format_prompt exists', agent_loop_module:py_fragment(format_prompt, _)),
    decl_assert_true('backend(gemini) helper fragment stream_json_return exists', agent_loop_module:py_fragment(stream_json_return, _)),
    decl_assert_true('backend(gemini) helper fragment format_prompt exists', agent_loop_module:py_fragment(format_prompt, _)),
    decl_assert_true('backend(claude_api) helper fragment extract_tool_calls_anthropic exists', agent_loop_module:py_fragment(extract_tool_calls_anthropic, _)),
    decl_assert_true('backend(openai_api) helper fragment extract_tool_calls_openai exists', agent_loop_module:py_fragment(extract_tool_calls_openai, _)),
    decl_assert_true('backend(ollama_api) helper fragment list_models_api exists', agent_loop_module:py_fragment(list_models_api, _)),
    decl_assert_true('backend(ollama_cli) helper fragment format_prompt exists', agent_loop_module:py_fragment(format_prompt, _)),
    decl_assert_true('backend(ollama_cli) helper fragment clean_output_simple exists', agent_loop_module:py_fragment(clean_output_simple, _)),
    decl_assert_true('backend(ollama_cli) helper fragment list_models_cli exists', agent_loop_module:py_fragment(list_models_cli, _)),
    decl_assert_true('backend(openrouter_api) helper fragment supports_streaming_true exists', agent_loop_module:py_fragment(supports_streaming_true, _)),
    decl_assert_true('backend(openrouter_api) helper fragment sse_streaming_openrouter exists', agent_loop_module:py_fragment(sse_streaming_openrouter, _)),
    true.

%% =============================================================================
%% Count Consistency Tests
%% =============================================================================

test_decl_counts :-
    format("~nDeclarative count tests:~n"),
    findall(N, agent_loop_module:tool_spec(N, _), Ts), length(Ts, TC),
    decl_assert_eq('tool_spec count', TC, 4),
    findall(N, agent_loop_module:agent_backend(N, _), Bs), length(Bs, BC),
    decl_assert_eq('backend count', BC, 8),
    findall(N, agent_loop_module:security_profile(N, _), Ps), length(Ps, PC),
    decl_assert_eq('security_profile count', PC, 4),
    findall(N, agent_loop_module:agent_config_field(N, _, _, _), Cs), length(Cs, CC),
    decl_assert_eq('config_field count', CC, 28),
    findall(N, agent_loop_module:slash_command(N, _, _, _), Ss), length(Ss, SC),
    decl_assert_eq('slash_command count', SC, 27),
    findall(N, agent_loop_module:py_fragment(N, _), PFs), length(PFs, PFC),
    decl_assert_eq('py_fragment count', PFC, 95),
    findall(N, agent_loop_module:rust_fragment(N, _), RFs), length(RFs, RFC),
    decl_assert_eq('rust_fragment count', RFC, 38),
    true.

%% =============================================================================
%% Shared Logic Tests
%% =============================================================================

test_decl_shared_logic :-
    format("~nDeclarative shared logic tests:~n"),
    decl_assert_true('shared_logic(is_over_budget) exists', agent_loop_module:shared_logic(_, is_over_budget, _)),
    decl_assert_true('shared_logic(is_over_budget) compiles for python', agent_loop_module:compile_logic(python, is_over_budget, _)),
    decl_assert_true('shared_logic(is_over_budget) compiles for rust', agent_loop_module:compile_logic(rust, is_over_budget, _)),
    decl_assert_true('shared_logic(budget_remaining) exists', agent_loop_module:shared_logic(_, budget_remaining, _)),
    decl_assert_true('shared_logic(budget_remaining) compiles for python', agent_loop_module:compile_logic(python, budget_remaining, _)),
    decl_assert_true('shared_logic(budget_remaining) compiles for rust', agent_loop_module:compile_logic(rust, budget_remaining, _)),
    decl_assert_true('shared_logic(cache_clear) exists', agent_loop_module:shared_logic(_, cache_clear, _)),
    decl_assert_true('shared_logic(cache_clear) compiles for python', agent_loop_module:compile_logic(python, cache_clear, _)),
    decl_assert_true('shared_logic(cache_clear) compiles for rust', agent_loop_module:compile_logic(rust, cache_clear, _)),
    decl_assert_true('shared_logic(cache_len) exists', agent_loop_module:shared_logic(_, cache_len, _)),
    decl_assert_true('shared_logic(cache_len) compiles for python', agent_loop_module:compile_logic(python, cache_len, _)),
    decl_assert_true('shared_logic(cache_len) compiles for rust', agent_loop_module:compile_logic(rust, cache_len, _)),
    true.

%% =============================================================================
%% Test Runner
%% =============================================================================

run_declarative_tests :-
    retractall(decl_test_passed(_)),
    retractall(decl_test_failed(_)),
    format("~n=== Declarative Tests (auto-generated) ===~n"),
    test_decl_existence,
    test_decl_properties,
    test_decl_crossrefs,
    test_decl_counts,
    test_decl_shared_logic,
    aggregate_all(count, decl_test_passed(_), Passed),
    aggregate_all(count, decl_test_failed(_), Failed),
    format("~n=== Declarative: ~w passed, ~w failed ===~n", [Passed, Failed]),
    true.  %% Main runner handles halt

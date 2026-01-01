% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
%% typescript_glue_config.pl - Preferences/Firewall Integration for TypeScript Glue
%
% This module integrates the UnifyWeaver preference and firewall systems with
% the TypeScript glue modules (rpyc_security, express_generator, react_generator,
% full_pipeline). It provides:
%
% 1. Default preferences for RPyC, Express, and React generation
% 2. Firewall implications for Python module access control
% 3. Context-aware configuration resolution
% 4. Preference-based code generation options
%
% @author John William Creighton (@s243a)
% @license MIT OR Apache-2.0

:- module(typescript_glue_config, [
    % Preference accessors
    get_rpyc_config/1,
    get_rpyc_config/2,
    get_express_config/1,
    get_express_config/2,
    get_react_config/1,
    get_react_config/2,
    get_pipeline_config/1,
    get_pipeline_config/2,

    % Firewall integration
    validate_rpyc_module/2,
    validate_rpyc_call/3,
    get_allowed_modules/1,
    get_allowed_modules/2,

    % Context management
    set_generation_context/1,
    current_generation_context/1,
    with_context/2,

    % Preference initialization
    init_typescript_glue_preferences/0,
    init_typescript_glue_firewall/0,

    % Testing
    test_typescript_glue_config/0
]).

:- use_module(library(lists)).

% Conditionally load preferences and firewall modules if available
:- (   current_module(preferences)
   ->  true
   ;   catch(use_module('../core/preferences'), _, true)
   ).

:- (   current_module(firewall)
   ->  true
   ;   catch(use_module('../core/firewall'), _, true)
   ).

% Dynamic predicates for context and runtime configuration
:- dynamic current_generation_context/1.
:- dynamic typescript_glue_initialized/0.

%% ============================================
%% DEFAULT PREFERENCES
%% ============================================

%% rpyc_default_config(-Config) is det.
%
% Default configuration for RPyC connections.
rpyc_default_config([
    host(localhost),
    port(18812),
    timeout(30000),
    retry_count(3),
    connection_pool_size(5)
]).

%% security_default_config(-Config) is det.
%
% Default security configuration for TypeScript generation.
security_default_config([
    rate_limit(100),
    rate_limit_window(60000),
    body_limit(102400),
    timeout(30000),
    max_depth(10),
    max_array_length(10000)
]).

%% express_default_config(-Config) is det.
%
% Default configuration for Express.js generation.
express_default_config([
    port(3001),
    cors_enabled(true),
    cors_origins(['http://localhost:3000']),
    helmet_enabled(true),
    morgan_format(dev),
    api_prefix('/api')
]).

%% react_default_config(-Config) is det.
%
% Default configuration for React component generation.
react_default_config([
    typescript(true),
    css_modules(true),
    strict_mode(true),
    theme(modern),
    default_precision(6)
]).

%% pipeline_default_config(-Config) is det.
%
% Default configuration for full pipeline generation.
pipeline_default_config([
    node_version('18'),
    python_version('3.10'),
    include_docker(true),
    include_readme(true),
    include_tests(false)
]).

%% ============================================
%% PREFERENCE ACCESSORS
%% ============================================

%% get_rpyc_config(-Config) is det.
%
% Get RPyC configuration with all defaults and overrides applied.
get_rpyc_config(Config) :-
    get_rpyc_config([], Config).

%% get_rpyc_config(+RuntimeOptions, -Config) is det.
%
% Get RPyC configuration with runtime options taking precedence.
get_rpyc_config(RuntimeOptions, Config) :-
    rpyc_default_config(Defaults),
    security_default_config(SecurityDefaults),
    append(Defaults, SecurityDefaults, AllDefaults),
    get_context_preferences(rpyc, ContextPrefs),
    merge_config(AllDefaults, ContextPrefs, Merged1),
    merge_config(Merged1, RuntimeOptions, Config).

%% get_express_config(-Config) is det.
%
% Get Express configuration with all defaults and overrides applied.
get_express_config(Config) :-
    get_express_config([], Config).

%% get_express_config(+RuntimeOptions, -Config) is det.
%
% Get Express configuration with runtime options taking precedence.
get_express_config(RuntimeOptions, Config) :-
    express_default_config(Defaults),
    security_default_config(SecurityDefaults),
    append(Defaults, SecurityDefaults, AllDefaults),
    get_context_preferences(express, ContextPrefs),
    merge_config(AllDefaults, ContextPrefs, Merged1),
    merge_config(Merged1, RuntimeOptions, Config).

%% get_react_config(-Config) is det.
%
% Get React configuration with all defaults and overrides applied.
get_react_config(Config) :-
    get_react_config([], Config).

%% get_react_config(+RuntimeOptions, -Config) is det.
%
% Get React configuration with runtime options taking precedence.
get_react_config(RuntimeOptions, Config) :-
    react_default_config(Defaults),
    get_context_preferences(react, ContextPrefs),
    merge_config(Defaults, ContextPrefs, Merged1),
    merge_config(Merged1, RuntimeOptions, Config).

%% get_pipeline_config(-Config) is det.
%
% Get full pipeline configuration with all defaults and overrides applied.
get_pipeline_config(Config) :-
    get_pipeline_config([], Config).

%% get_pipeline_config(+RuntimeOptions, -Config) is det.
%
% Get full pipeline configuration with runtime options taking precedence.
get_pipeline_config(RuntimeOptions, Config) :-
    pipeline_default_config(Defaults),
    get_context_preferences(pipeline, ContextPrefs),
    merge_config(Defaults, ContextPrefs, Merged1),
    merge_config(Merged1, RuntimeOptions, Config).

%% get_context_preferences(+Component, -Preferences) is det.
%
% Get preferences based on current generation context.
get_context_preferences(Component, Preferences) :-
    (   current_generation_context(Context)
    ->  get_preferences_for_context(Context, Component, Preferences)
    ;   Preferences = []
    ).

%% get_preferences_for_context(+Context, +Component, -Preferences) is det.
%
% Get component preferences for a specific context.
get_preferences_for_context(production, rpyc, [
    timeout(60000),
    retry_count(5),
    connection_pool_size(10)
]).
get_preferences_for_context(production, express, [
    morgan_format(combined),
    helmet_enabled(true)
]).
get_preferences_for_context(production, _, []).

get_preferences_for_context(development, rpyc, [
    timeout(5000),
    retry_count(1)
]).
get_preferences_for_context(development, express, [
    morgan_format(dev),
    cors_origins(['*'])
]).
get_preferences_for_context(development, _, []).

get_preferences_for_context(testing, rpyc, [
    host(localhost),
    port(18899),
    timeout(1000)
]).
get_preferences_for_context(testing, express, [
    port(3099)
]).
get_preferences_for_context(testing, _, []).

get_preferences_for_context(_, _, []).

%% ============================================
%% FIREWALL INTEGRATION
%% ============================================

%% validate_rpyc_module(+Module, -Result) is det.
%
% Validate that an RPyC module access is allowed by the firewall.
% Returns ok or error(Reason).
validate_rpyc_module(Module, Result) :-
    get_allowed_modules(AllowedModules),
    (   member(Module, AllowedModules)
    ->  Result = ok
    ;   format(atom(Reason), "Module '~w' is not in the firewall allowlist", [Module]),
        Result = error(Reason)
    ).

%% validate_rpyc_call(+Module, +Function, -Result) is det.
%
% Validate that an RPyC function call is allowed by the firewall.
% Returns ok or error(Reason).
validate_rpyc_call(Module, Function, Result) :-
    validate_rpyc_module(Module, ModuleResult),
    (   ModuleResult = error(_)
    ->  Result = ModuleResult
    ;   get_allowed_functions(Module, AllowedFunctions),
        (   AllowedFunctions == []  % No restrictions
        ->  Result = ok
        ;   member(Function, AllowedFunctions)
        ->  Result = ok
        ;   format(atom(Reason), "Function '~w.~w' is not in the firewall allowlist", [Module, Function]),
            Result = error(Reason)
        )
    ).

%% get_allowed_modules(-Modules) is det.
%
% Get list of allowed Python modules from firewall + rpyc_security.
get_allowed_modules(Modules) :-
    get_allowed_modules([], Modules).

%% get_allowed_modules(+Context, -Modules) is det.
%
% Get list of allowed Python modules for a specific context.
get_allowed_modules(Context, Modules) :-
    % Try to get from rpyc_security module if loaded
    (   current_module(rpyc_security),
        catch(findall(M, rpyc_security:rpyc_allowed_module(M, _), SecurityModules), _, SecurityModules = [])
    ->  true
    ;   SecurityModules = []
    ),
    % Get context-specific allowed modules
    get_context_allowed_modules(Context, ContextModules),
    % Get firewall-allowed modules
    get_firewall_allowed_modules(FirewallModules),
    % Combine: security ∩ (context ∪ firewall)
    append(ContextModules, FirewallModules, CombinedAllowed),
    (   SecurityModules == []
    ->  Modules = CombinedAllowed
    ;   CombinedAllowed == []
    ->  Modules = SecurityModules
    ;   intersection(SecurityModules, CombinedAllowed, Modules)
    ),
    % Fallback to security modules if intersection is empty
    (   Modules == [], SecurityModules \== []
    ->  Modules = SecurityModules
    ;   true
    ).

%% get_allowed_functions(+Module, -Functions) is det.
%
% Get list of allowed functions for a module.
get_allowed_functions(Module, Functions) :-
    (   current_module(rpyc_security),
        catch(rpyc_security:rpyc_allowed_module(Module, Functions), _, Functions = [])
    ->  true
    ;   Functions = []
    ).

%% get_context_allowed_modules(+Context, -Modules) is det.
%
% Get modules allowed for a specific context.
get_context_allowed_modules(production, [math, numpy, statistics]).
get_context_allowed_modules(development, [math, numpy, statistics, pandas, matplotlib]).
get_context_allowed_modules(testing, [math, numpy, statistics, unittest, pytest]).
get_context_allowed_modules([], []).  % No context = no restrictions from context
get_context_allowed_modules(_, []).

%% get_firewall_allowed_modules(-Modules) is det.
%
% Get modules allowed by the firewall system.
get_firewall_allowed_modules(Modules) :-
    (   current_module(firewall),
        catch(firewall:firewall_default(Defaults), _, Defaults = [])
    ->  true
    ;   Defaults = []
    ),
    findall(M, (member(python_modules(Ms), Defaults), member(M, Ms)), Modules).

%% ============================================
%% CONTEXT MANAGEMENT
%% ============================================

%% set_generation_context(+Context) is det.
%
% Set the current generation context (e.g., production, development, testing).
set_generation_context(Context) :-
    retractall(current_generation_context(_)),
    assertz(current_generation_context(Context)).

%% with_context(+Context, :Goal) is nondet.
%
% Execute Goal with a temporary generation context.
:- meta_predicate with_context(+, 0).
with_context(Context, Goal) :-
    (   current_generation_context(OldContext)
    ->  HasOld = true
    ;   HasOld = false
    ),
    set_generation_context(Context),
    (   call(Goal)
    ->  (   HasOld == true
        ->  set_generation_context(OldContext)
        ;   retractall(current_generation_context(_))
        )
    ;   (   HasOld == true
        ->  set_generation_context(OldContext)
        ;   retractall(current_generation_context(_))
        ),
        fail
    ).

%% ============================================
%% FIREWALL IMPLICATIONS FOR TYPESCRIPT GLUE
%% ============================================

%% Initialize firewall implications for TypeScript glue
init_typescript_glue_firewall :-
    % Implication: No Python available → deny RPyC services
    (   current_module(firewall)
    ->  (   \+ firewall:firewall_implies_default(no_python_available, _)
        ->  assertz(firewall:firewall_implies_default(
                no_python_available,
                denied(service(typescript, rpyc(_)))
            ))
        ;   true
        ),
        % Implication: Offline mode → deny RPyC network connections
        (   \+ firewall:firewall_implies_default(mode(offline), denied(service(typescript, rpyc(_))))
        ->  assertz(firewall:firewall_implies_default(
                mode(offline),
                denied(service(typescript, rpyc(_)))
            ))
        ;   true
        ),
        % Implication: Strict security → only allow whitelisted Python modules
        (   \+ firewall:firewall_implies_default(security_policy(strict), require_python_module_whitelist)
        ->  assertz(firewall:firewall_implies_default(
                security_policy(strict),
                require_python_module_whitelist
            ))
        ;   true
        ),
        % Implication: Production → use production RPyC settings
        (   \+ firewall:firewall_implies_default(environment(production), rpyc_production_mode)
        ->  assertz(firewall:firewall_implies_default(
                environment(production),
                rpyc_production_mode
            ))
        ;   true
        )
    ;   true
    ).

%% ============================================
%% PREFERENCE INITIALIZATION
%% ============================================

%% init_typescript_glue_preferences/0 is det.
%
% Initialize default preferences for TypeScript glue modules.
init_typescript_glue_preferences :-
    (   typescript_glue_initialized
    ->  true
    ;   % Set up default preferences if preferences module is available
        (   current_module(preferences)
        ->  % Only add if not already set
            (   \+ preferences:preferences_default(_)
            ->  assertz(preferences:preferences_default([
                    rpyc_host(localhost),
                    rpyc_port(18812),
                    rate_limit(100),
                    body_limit(102400),
                    prefer([typescript, express, react])
                ]))
            ;   true
            )
        ;   true
        ),
        init_typescript_glue_firewall,
        assertz(typescript_glue_initialized)
    ).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% merge_config(+Base, +Override, -Merged) is det.
%
% Merge configuration lists, with Override taking precedence.
merge_config(Base, Override, Merged) :-
    findall(Key-Value, (
        member(Term, Base),
        functor(Term, Key, 1),
        arg(1, Term, Value)
    ), BasePairs),
    findall(Key-Value, (
        member(Term, Override),
        functor(Term, Key, 1),
        arg(1, Term, Value)
    ), OverridePairs),
    % Override takes precedence
    merge_pairs(BasePairs, OverridePairs, MergedPairs),
    findall(Term, (
        member(Key-Value, MergedPairs),
        Term =.. [Key, Value]
    ), Merged).

%% merge_pairs(+Base, +Override, -Merged) is det.
%
% Merge key-value pairs, with Override taking precedence.
merge_pairs(Base, Override, Merged) :-
    findall(Key-Value, (
        member(Key-Value, Override)
    ;   member(Key-Value, Base),
        \+ member(Key-_, Override)
    ), Merged).

%% ============================================
%% TESTING
%% ============================================

%% test_typescript_glue_config/0 is det.
%
% Run tests for the typescript_glue_config module.
test_typescript_glue_config :-
    format('~n=== TypeScript Glue Config Tests ===~n~n', []),

    % Test 1: Default RPyC config
    format('Test 1: Default RPyC config~n', []),
    get_rpyc_config(RpycConfig),
    (   member(host(localhost), RpycConfig),
        member(port(18812), RpycConfig)
    ->  format('  [PASS] Has default host and port~n', [])
    ;   format('  [FAIL] Missing default host or port~n', [])
    ),

    % Test 2: Runtime override
    format('Test 2: Runtime override~n', []),
    get_rpyc_config([port(19000)], OverrideConfig),
    (   member(port(19000), OverrideConfig)
    ->  format('  [PASS] Runtime port override works~n', [])
    ;   format('  [FAIL] Runtime port override failed~n', [])
    ),

    % Test 3: Context-based preferences
    format('Test 3: Context-based preferences~n', []),
    set_generation_context(production),
    get_rpyc_config(ProdConfig),
    (   member(timeout(60000), ProdConfig)
    ->  format('  [PASS] Production context applied~n', [])
    ;   format('  [FAIL] Production context not applied~n', [])
    ),
    retractall(current_generation_context(_)),

    % Test 4: Express config
    format('Test 4: Express config~n', []),
    get_express_config(ExpressConfig),
    (   member(port(3001), ExpressConfig),
        member(cors_enabled(true), ExpressConfig)
    ->  format('  [PASS] Express defaults loaded~n', [])
    ;   format('  [FAIL] Express defaults missing~n', [])
    ),

    % Test 5: React config
    format('Test 5: React config~n', []),
    get_react_config(ReactConfig),
    (   member(typescript(true), ReactConfig),
        member(css_modules(true), ReactConfig)
    ->  format('  [PASS] React defaults loaded~n', [])
    ;   format('  [FAIL] React defaults missing~n', [])
    ),

    % Test 6: Pipeline config
    format('Test 6: Pipeline config~n', []),
    get_pipeline_config(PipelineConfig),
    (   member(include_docker(true), PipelineConfig)
    ->  format('  [PASS] Pipeline defaults loaded~n', [])
    ;   format('  [FAIL] Pipeline defaults missing~n', [])
    ),

    % Test 7: Module validation
    format('Test 7: Module validation~n', []),
    get_allowed_modules(Modules),
    format('  Allowed modules: ~w~n', [Modules]),
    format('  [PASS] Module list retrieved~n', []),

    % Test 8: with_context/2
    format('Test 8: with_context/2~n', []),
    with_context(testing, (
        current_generation_context(Ctx),
        Ctx == testing
    )),
    (   \+ current_generation_context(_)
    ->  format('  [PASS] Context properly restored~n', [])
    ;   format('  [FAIL] Context not restored~n', []),
        retractall(current_generation_context(_))
    ),

    format('~n=== All tests completed ===~n', []).

%% Auto-initialize on module load
:- initialization(init_typescript_glue_preferences, now).

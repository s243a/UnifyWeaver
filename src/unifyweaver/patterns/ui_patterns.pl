% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% ui_patterns.pl - Generalizable UI Patterns for Code Generation
%
% Provides abstract UI patterns that can be compiled to multiple targets:
%   - React Native
%   - Vue
%   - Flutter (future)
%   - SwiftUI (future)
%
% Patterns are composed using the component registry and binding system.
% Each pattern specifies:
%   - Abstract structure (target-agnostic)
%   - Required capabilities
%   - Composition rules
%
% Categories:
%   - Navigation patterns (stack, tab, drawer)
%   - State patterns (local, global, derived)
%   - Data patterns (fetch, cache, paginate)
%   - Persistence patterns (local, secure, sync)

:- module(ui_patterns, [
    % Pattern definition
    define_pattern/3,               % +PatternName, +Spec, +Options
    pattern/3,                      % ?PatternName, ?Spec, ?Options
    pattern_requires/2,             % +PatternName, -Capabilities

    % Pattern composition
    compose_patterns/3,             % +Patterns, +Options, -ComposedSpec
    pattern_compatible/2,           % +Pattern1, +Pattern2

    % Pattern compilation
    compile_pattern/4,              % +PatternName, +Target, +Options, -Code

    % Navigation patterns
    navigation_pattern/3,           % +Type, +Screens, -Pattern
    screen_spec/3,                  % +Name, +Component, +Options

    % State patterns
    state_pattern/3,                % +Type, +Shape, -Pattern
    local_state/2,                  % +Name, +InitialValue
    global_state/3,                 % +StoreName, +Slices, -Pattern
    derived_state/3,                % +Name, +Dependencies, +Derivation

    % Data patterns
    data_pattern/3,                 % +Type, +Config, -Pattern
    query_pattern/4,                % +Name, +Endpoint, +Options, -Pattern
    mutation_pattern/4,             % +Name, +Endpoint, +Options, -Pattern
    paginated_pattern/4,            % +Name, +Endpoint, +Options, -Pattern

    % Persistence patterns
    persistence_pattern/3,          % +Type, +Config, -Pattern
    local_storage/3,                % +Key, +Schema, -Pattern
    secure_storage/3,               % +Key, +Schema, -Pattern

    % Testing
    test_ui_patterns/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_pattern/3.        % stored_pattern(Name, Spec, Options)

% ============================================================================
% PATTERN DEFINITION
% ============================================================================

%% define_pattern(+PatternName, +Spec, +Options)
%
%  Define a new UI pattern.
%
%  Spec structure:
%    pattern(Type, Config, Children)
%    - Type: navigation | state | data | persistence | composite
%    - Config: pattern-specific configuration
%    - Children: nested patterns (for composition)
%
%  Options:
%    - requires([Capability, ...]) - required target capabilities
%    - composable_with([Pattern, ...]) - compatible patterns
%    - singleton(bool) - only one instance allowed
%
define_pattern(Name, Spec, Options) :-
    atom(Name),
    (   stored_pattern(Name, _, _)
    ->  retract(stored_pattern(Name, _, _))
    ;   true
    ),
    assertz(stored_pattern(Name, Spec, Options)).

%% pattern(?Name, ?Spec, ?Options)
%
%  Query defined patterns.
%
pattern(Name, Spec, Options) :-
    stored_pattern(Name, Spec, Options).

%% pattern_requires(+PatternName, -Capabilities)
%
%  Get required capabilities for a pattern.
%
pattern_requires(Name, Capabilities) :-
    pattern(Name, _, Options),
    (   member(requires(Capabilities), Options)
    ->  true
    ;   Capabilities = []
    ).

% ============================================================================
% PATTERN COMPOSITION
% ============================================================================

%% compose_patterns(+Patterns, +Options, -ComposedSpec)
%
%  Compose multiple patterns into a single spec.
%
compose_patterns(Patterns, Options, ComposedSpec) :-
    maplist(get_pattern_spec, Patterns, Specs),
    merge_specs(Specs, MergedSpec),
    ComposedSpec = composite(MergedSpec, Options).

get_pattern_spec(Name, Spec) :-
    atom(Name),
    pattern(Name, Spec, _).
get_pattern_spec(Spec, Spec) :-
    \+ atom(Spec).

merge_specs(Specs, merged(Specs)).

%% pattern_compatible(+Pattern1, +Pattern2)
%
%  Check if two patterns can be composed together.
%
pattern_compatible(P1, P2) :-
    pattern(P1, _, Opts1),
    pattern(P2, _, Opts2),
    (   member(composable_with(Compatible), Opts1)
    ->  member(P2, Compatible)
    ;   true
    ),
    (   member(composable_with(Compatible2), Opts2)
    ->  member(P1, Compatible2)
    ;   true
    ).

% ============================================================================
% PATTERN COMPILATION
% ============================================================================

%% compile_pattern(+PatternName, +Target, +Options, -Code)
%
%  Compile a pattern to target-specific code.
%
compile_pattern(Name, Target, Options, Code) :-
    pattern(Name, Spec, PatternOpts),
    check_target_capabilities(Target, PatternOpts),
    compile_pattern_spec(Spec, Target, Options, Code).

check_target_capabilities(Target, PatternOpts) :-
    (   member(requires(Required), PatternOpts)
    ->  forall(member(Cap, Required), target_has_capability(Target, Cap))
    ;   true
    ).

% Placeholder - would check actual target capabilities
target_has_capability(react_native, navigation).
target_has_capability(react_native, state_hooks).
target_has_capability(react_native, async_storage).
target_has_capability(react_native, react_query).
target_has_capability(vue, state_composition).
target_has_capability(vue, vue_router).
target_has_capability(_, _) :- !.  % Default allow for now

compile_pattern_spec(navigation(Type, Screens, Config), Target, Options, Code) :-
    compile_navigation_pattern(Type, Screens, Config, Target, Options, Code).
compile_pattern_spec(state(Type, Shape, Config), Target, Options, Code) :-
    compile_state_pattern(Type, Shape, Config, Target, Options, Code).
compile_pattern_spec(data(Type, Config), Target, Options, Code) :-
    compile_data_pattern(Type, Config, Target, Options, Code).
compile_pattern_spec(persistence(Type, Config), Target, Options, Code) :-
    compile_persistence_pattern(Type, Config, Target, Options, Code).
compile_pattern_spec(composite(Specs, _), Target, Options, Code) :-
    compile_composite_pattern(Specs, Target, Options, Code).

% ============================================================================
% NAVIGATION PATTERNS
% ============================================================================

%% navigation_pattern(+Type, +Screens, -Pattern)
%
%  Create a navigation pattern specification.
%
%  Types: stack, tab, drawer, native_stack
%
navigation_pattern(Type, Screens, Pattern) :-
    member(Type, [stack, tab, drawer, native_stack]),
    maplist(validate_screen_spec, Screens),
    Pattern = navigation(Type, Screens, []).

%% screen_spec(+Name, +Component, +Options)
%
%  Define a screen specification.
%
screen_spec(Name, Component, Options) :-
    atom(Name),
    atom(Component),
    is_list(Options).

validate_screen_spec(screen(Name, Component, Options)) :-
    atom(Name),
    atom(Component),
    is_list(Options).

%% compile_navigation_pattern(+Type, +Screens, +Config, +Target, +Options, -Code)
compile_navigation_pattern(stack, Screens, _Config, react_native, Options, Code) :-
    option_value(Options, component_name, 'AppNavigator', Name),
    generate_rn_stack_navigator(Screens, Name, Code).
compile_navigation_pattern(tab, Screens, _Config, react_native, Options, Code) :-
    option_value(Options, component_name, 'TabNavigator', Name),
    generate_rn_tab_navigator(Screens, Name, Code).
compile_navigation_pattern(drawer, Screens, _Config, react_native, Options, Code) :-
    option_value(Options, component_name, 'DrawerNavigator', Name),
    generate_rn_drawer_navigator(Screens, Name, Code).

generate_rn_stack_navigator(Screens, Name, Code) :-
    generate_screen_imports(Screens, Imports),
    generate_screen_definitions(Screens, stack, ScreenDefs),
    format(string(Code),
"import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
~w

const Stack = createStackNavigator();

export const ~w: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
~w
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default ~w;
", [Imports, Name, ScreenDefs, Name]).

generate_rn_tab_navigator(Screens, Name, Code) :-
    generate_screen_imports(Screens, Imports),
    generate_screen_definitions(Screens, tab, ScreenDefs),
    format(string(Code),
"import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer } from '@react-navigation/native';
~w

const Tab = createBottomTabNavigator();

export const ~w: React.FC = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator>
~w
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default ~w;
", [Imports, Name, ScreenDefs, Name]).

generate_rn_drawer_navigator(Screens, Name, Code) :-
    generate_screen_imports(Screens, Imports),
    generate_screen_definitions(Screens, drawer, ScreenDefs),
    format(string(Code),
"import React from 'react';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { NavigationContainer } from '@react-navigation/native';
~w

const Drawer = createDrawerNavigator();

export const ~w: React.FC = () => {
  return (
    <NavigationContainer>
      <Drawer.Navigator>
~w
      </Drawer.Navigator>
    </NavigationContainer>
  );
};

export default ~w;
", [Imports, Name, ScreenDefs, Name]).

generate_screen_imports(Screens, Imports) :-
    findall(ImportLine, (
        member(screen(_, Component, _), Screens),
        format(string(ImportLine), "import { ~w } from './screens/~w';", [Component, Component])
    ), ImportLines),
    atomic_list_concat(ImportLines, '\n', Imports).

generate_screen_definitions(Screens, Type, Defs) :-
    navigator_prefix(Type, Prefix),
    findall(ScreenDef, (
        member(screen(Name, Component, Opts), Screens),
        generate_single_screen(Prefix, Name, Component, Opts, ScreenDef)
    ), ScreenDefs),
    atomic_list_concat(ScreenDefs, '\n', Defs).

navigator_prefix(stack, 'Stack').
navigator_prefix(tab, 'Tab').
navigator_prefix(drawer, 'Drawer').

generate_single_screen(Prefix, Name, Component, Opts, Def) :-
    (   member(title(Title), Opts)
    ->  format(string(Def), "        <~w.Screen name=\"~w\" component={~w} options={{ title: '~w' }} />",
               [Prefix, Name, Component, Title])
    ;   format(string(Def), "        <~w.Screen name=\"~w\" component={~w} />",
               [Prefix, Name, Component])
    ).

% ============================================================================
% STATE PATTERNS
% ============================================================================

%% state_pattern(+Type, +Shape, -Pattern)
%
%  Create a state pattern specification.
%
%  Types: local, global, derived
%
state_pattern(local, Shape, Pattern) :-
    Pattern = state(local, Shape, []).
state_pattern(global, Shape, Pattern) :-
    Pattern = state(global, Shape, []).
state_pattern(derived, Shape, Pattern) :-
    Pattern = state(derived, Shape, []).

%% local_state(+Name, +InitialValue)
local_state(Name, InitialValue) :-
    atom(Name),
    define_pattern(Name,
        state(local, [field(Name, InitialValue)], []),
        [requires([state_hooks])]).

%% global_state(+StoreName, +Slices, -Pattern)
global_state(StoreName, Slices, Pattern) :-
    atom(StoreName),
    is_list(Slices),
    Pattern = state(global, [store(StoreName), slices(Slices)], []),
    define_pattern(StoreName, Pattern, [requires([zustand]), singleton(true)]).

%% derived_state(+Name, +Dependencies, +Derivation)
derived_state(Name, Dependencies, Derivation) :-
    atom(Name),
    is_list(Dependencies),
    define_pattern(Name,
        state(derived, [deps(Dependencies), derive(Derivation)], []),
        [requires([state_hooks])]).

%% compile_state_pattern(+Type, +Shape, +Config, +Target, +Options, -Code)
compile_state_pattern(local, Shape, _Config, react_native, _Options, Code) :-
    generate_rn_local_state(Shape, Code).
compile_state_pattern(global, Shape, _Config, react_native, _Options, Code) :-
    generate_rn_zustand_store(Shape, Code).
compile_state_pattern(derived, Shape, _Config, react_native, _Options, Code) :-
    generate_rn_derived_state(Shape, Code).

generate_rn_local_state(Shape, Code) :-
    findall(Hook, (
        member(field(Name, Initial), Shape),
        format(string(Hook), "const [~w, set~w] = useState(~w);",
               [Name, Name, Initial])
    ), Hooks),
    atomic_list_concat(Hooks, '\n  ', HooksCode),
    format(string(Code),
"// Local state hooks
import { useState } from 'react';

// Usage in component:
  ~w
", [HooksCode]).

generate_rn_zustand_store(Shape, Code) :-
    member(store(StoreName), Shape),
    member(slices(Slices), Shape),
    generate_slice_types(Slices, TypeDefs),
    generate_slice_state(Slices, StateDefs),
    generate_slice_actions(Slices, ActionDefs),
    format(string(Code),
"import { create } from 'zustand';

~w

interface ~wState {
~w
~w
}

export const use~wStore = create<~wState>((set, get) => ({
~w
~w
}));
", [TypeDefs, StoreName, StateDefs, ActionDefs, StoreName, StoreName, StateDefs, ActionDefs]).

generate_slice_types(Slices, TypeDefs) :-
    findall(TypeDef, (
        member(slice(Name, Fields, _), Slices),
        generate_slice_interface(Name, Fields, TypeDef)
    ), TypeDefList),
    atomic_list_concat(TypeDefList, '\n', TypeDefs).

generate_slice_interface(Name, Fields, TypeDef) :-
    findall(FieldDef, (
        member(field(FName, FType), Fields),
        format(string(FieldDef), "  ~w: ~w;", [FName, FType])
    ), FieldDefs),
    atomic_list_concat(FieldDefs, '\n', FieldsStr),
    format(string(TypeDef), "interface ~wSlice {\n~w\n}", [Name, FieldsStr]).

generate_slice_state(Slices, StateDefs) :-
    findall(StateDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _FType), Fields),
        format(string(StateDef), "  ~w: null,", [FName])
    ), StateDefList),
    atomic_list_concat(StateDefList, '\n', StateDefs).

generate_slice_actions(Slices, ActionDefs) :-
    findall(ActionDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, ABody), Actions),
        format(string(ActionDef), "  ~w: ~w,", [AName, ABody])
    ), ActionDefList),
    atomic_list_concat(ActionDefList, '\n', ActionDefs).

generate_rn_derived_state(Shape, Code) :-
    member(deps(Deps), Shape),
    member(derive(Derivation), Shape),
    atomic_list_concat(Deps, ', ', DepsStr),
    format(string(Code),
"// Derived state using useMemo
import { useMemo } from 'react';

// Usage in component:
  const derivedValue = useMemo(() => {
    return ~w;
  }, [~w]);
", [Derivation, DepsStr]).

% ============================================================================
% DATA PATTERNS
% ============================================================================

%% data_pattern(+Type, +Config, -Pattern)
%
%  Create a data fetching pattern specification.
%
%  Types: query, mutation, infinite
%
data_pattern(Type, Config, Pattern) :-
    member(Type, [query, mutation, infinite]),
    Pattern = data(Type, Config).

%% query_pattern(+Name, +Endpoint, +Options, -Pattern)
query_pattern(Name, Endpoint, Options, Pattern) :-
    atom(Name),
    Pattern = data(query, [name(Name), endpoint(Endpoint) | Options]),
    define_pattern(Name, Pattern, [requires([react_query])]).

%% mutation_pattern(+Name, +Endpoint, +Options, -Pattern)
mutation_pattern(Name, Endpoint, Options, Pattern) :-
    atom(Name),
    Pattern = data(mutation, [name(Name), endpoint(Endpoint) | Options]),
    define_pattern(Name, Pattern, [requires([react_query])]).

%% paginated_pattern(+Name, +Endpoint, +Options, -Pattern)
paginated_pattern(Name, Endpoint, Options, Pattern) :-
    atom(Name),
    Pattern = data(infinite, [name(Name), endpoint(Endpoint) | Options]),
    define_pattern(Name, Pattern, [requires([react_query])]).

%% compile_data_pattern(+Type, +Config, +Target, +Options, -Code)
compile_data_pattern(query, Config, react_native, _Options, Code) :-
    generate_rn_query_hook(Config, Code).
compile_data_pattern(mutation, Config, react_native, _Options, Code) :-
    generate_rn_mutation_hook(Config, Code).
compile_data_pattern(infinite, Config, react_native, _Options, Code) :-
    generate_rn_infinite_query(Config, Code).

generate_rn_query_hook(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(stale_time(StaleTime), Config) -> true ; StaleTime = 300000 ),
    (   member(cache_time(CacheTime), Config) -> true ; CacheTime = 600000 ),
    capitalize_first(Name, HookName),
    format(string(Code),
"import { useQuery, UseQueryResult } from '@tanstack/react-query';

interface ~wData {
  // TODO: Define response type
  [key: string]: unknown;
}

interface ~wVariables {
  // TODO: Define variables type
  [key: string]: unknown;
}

export const use~w = (variables?: ~wVariables): UseQueryResult<~wData> => {
  return useQuery({
    queryKey: ['~w', variables],
    queryFn: async () => {
      const response = await fetch('~w', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    },
    staleTime: ~w,
    gcTime: ~w,
  });
};
", [Name, Name, HookName, Name, Name, Name, Endpoint, StaleTime, CacheTime]).

generate_rn_mutation_hook(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    capitalize_first(Name, HookName),
    format(string(Code),
"import { useMutation, UseMutationResult, useQueryClient } from '@tanstack/react-query';

interface ~wVariables {
  // TODO: Define input type
  [key: string]: unknown;
}

interface ~wResponse {
  // TODO: Define response type
  [key: string]: unknown;
}

export const use~w = (): UseMutationResult<~wResponse, Error, ~wVariables> => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (variables: ~wVariables) => {
      const response = await fetch('~w', {
        method: '~w',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(variables),
      });
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    },
    onSuccess: () => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['~w'] });
    },
  });
};
", [Name, Name, HookName, Name, Name, Name, Endpoint, Method, Name]).

generate_rn_infinite_query(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'page' ),
    capitalize_first(Name, HookName),
    format(string(Code),
"import { useInfiniteQuery, UseInfiniteQueryResult } from '@tanstack/react-query';

interface ~wPage {
  data: unknown[];
  nextPage?: number;
  hasMore: boolean;
}

export const use~w = (): UseInfiniteQueryResult<~wPage> => {
  return useInfiniteQuery({
    queryKey: ['~w'],
    queryFn: async ({ pageParam = 1 }) => {
      const response = await fetch(`~w?~w=${pageParam}`);
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.nextPage : undefined,
  });
};
", [Name, HookName, Name, Name, Endpoint, PageParam]).

% ============================================================================
% PERSISTENCE PATTERNS
% ============================================================================

%% persistence_pattern(+Type, +Config, -Pattern)
%
%  Create a persistence pattern specification.
%
%  Types: local, secure, mmkv
%
persistence_pattern(Type, Config, Pattern) :-
    member(Type, [local, secure, mmkv]),
    Pattern = persistence(Type, Config).

%% local_storage(+Key, +Schema, -Pattern)
local_storage(Key, Schema, Pattern) :-
    atom(Key),
    Pattern = persistence(local, [key(Key), schema(Schema)]),
    define_pattern(Key, Pattern, [requires([async_storage])]).

%% secure_storage(+Key, +Schema, -Pattern)
secure_storage(Key, Schema, Pattern) :-
    atom(Key),
    Pattern = persistence(secure, [key(Key), schema(Schema)]),
    define_pattern(Key, Pattern, [requires([secure_store])]).

%% compile_persistence_pattern(+Type, +Config, +Target, +Options, -Code)
compile_persistence_pattern(local, Config, react_native, _Options, Code) :-
    generate_rn_async_storage_hook(Config, Code).
compile_persistence_pattern(mmkv, Config, react_native, _Options, Code) :-
    generate_rn_mmkv_hook(Config, Code).

generate_rn_async_storage_hook(Config, Code) :-
    member(key(Key), Config),
    (   member(schema(Schema), Config) -> true ; Schema = 'unknown' ),
    capitalize_first(Key, HookName),
    format(string(Code),
"import { useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '~w';

type ~wData = ~w;

interface Use~wResult {
  data: ~wData | null;
  loading: boolean;
  error: Error | null;
  save: (value: ~wData) => Promise<void>;
  remove: () => Promise<void>;
}

export const use~w = (): Use~wResult => {
  const [data, setData] = useState<~wData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const stored = await AsyncStorage.getItem(STORAGE_KEY);
        if (stored) setData(JSON.parse(stored));
      } catch (e) {
        setError(e as Error);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const save = useCallback(async (value: ~wData) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(value));
      setData(value);
    } catch (e) {
      setError(e as Error);
      throw e;
    }
  }, []);

  const remove = useCallback(async () => {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
      setData(null);
    } catch (e) {
      setError(e as Error);
      throw e;
    }
  }, []);

  return { data, loading, error, save, remove };
};
", [Key, Key, Schema, Key, Key, Key, HookName, Key, Key, Key]).

generate_rn_mmkv_hook(Config, Code) :-
    member(key(Key), Config),
    (   member(schema(Schema), Config) -> true ; Schema = 'unknown' ),
    capitalize_first(Key, HookName),
    format(string(Code),
"import { useState, useCallback } from 'react';
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV();
const STORAGE_KEY = '~w';

type ~wData = ~w;

interface Use~wResult {
  data: ~wData | null;
  save: (value: ~wData) => void;
  remove: () => void;
}

export const use~w = (): Use~wResult => {
  const [data, setData] = useState<~wData | null>(() => {
    const stored = storage.getString(STORAGE_KEY);
    return stored ? JSON.parse(stored) : null;
  });

  const save = useCallback((value: ~wData) => {
    storage.set(STORAGE_KEY, JSON.stringify(value));
    setData(value);
  }, []);

  const remove = useCallback(() => {
    storage.delete(STORAGE_KEY);
    setData(null);
  }, []);

  return { data, save, remove };
};
", [Key, Key, Schema, Key, Key, Key, HookName, Key, Key, Key]).

% ============================================================================
% COMPOSITE PATTERN COMPILATION
% ============================================================================

compile_composite_pattern(merged(Specs), Target, Options, Code) :-
    maplist(compile_single_merged_spec(Target, Options), Specs, Codes),
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

compile_single_merged_spec(Target, Options, Spec, Code) :-
    compile_pattern_spec(Spec, Target, Options, Code).

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Atom, Capitalized) :-
    atom_string(Atom, Str),
    string_chars(Str, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]),
    atom_string(Capitalized, Cap).

% ============================================================================
% TESTING
% ============================================================================

test_ui_patterns :-
    format('~n=== UI Patterns Tests ===~n~n'),

    % Test 1: Navigation pattern
    format('Test 1: Navigation pattern creation...~n'),
    navigation_pattern(stack, [
        screen(home, 'HomeScreen', [title('Home')]),
        screen(detail, 'DetailScreen', [])
    ], NavPattern),
    (   NavPattern = navigation(stack, _, _)
    ->  format('  PASS: Stack navigator pattern created~n')
    ;   format('  FAIL: Navigation pattern incorrect~n')
    ),

    % Test 2: Compile navigation to React Native
    format('~nTest 2: Navigation compilation...~n'),
    define_pattern(test_nav, NavPattern, [requires([navigation])]),
    compile_pattern(test_nav, react_native, [], NavCode),
    (   sub_string(NavCode, _, _, _, "createStackNavigator")
    ->  format('  PASS: Generated React Navigation code~n')
    ;   format('  FAIL: Missing createStackNavigator~n')
    ),

    % Test 3: Query pattern
    format('~nTest 3: Query pattern creation...~n'),
    query_pattern(fetch_users, '/api/users', [stale_time(60000)], QueryPattern),
    (   QueryPattern = data(query, _)
    ->  format('  PASS: Query pattern created~n')
    ;   format('  FAIL: Query pattern incorrect~n')
    ),

    % Test 4: Compile query to React Native
    format('~nTest 4: Query compilation...~n'),
    compile_pattern(fetch_users, react_native, [], QueryCode),
    (   sub_string(QueryCode, _, _, _, "useQuery")
    ->  format('  PASS: Generated useQuery hook~n')
    ;   format('  FAIL: Missing useQuery~n')
    ),

    % Test 5: Persistence pattern
    format('~nTest 5: Persistence pattern creation...~n'),
    local_storage(user_prefs, '{ theme: string }', PersistPattern),
    (   PersistPattern = persistence(local, _)
    ->  format('  PASS: Persistence pattern created~n')
    ;   format('  FAIL: Persistence pattern incorrect~n')
    ),

    % Test 6: Compile persistence to React Native
    format('~nTest 6: Persistence compilation...~n'),
    compile_pattern(user_prefs, react_native, [], PersistCode),
    (   sub_string(PersistCode, _, _, _, "AsyncStorage")
    ->  format('  PASS: Generated AsyncStorage hook~n')
    ;   format('  FAIL: Missing AsyncStorage~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('UI Patterns module loaded~n', [])
), now).

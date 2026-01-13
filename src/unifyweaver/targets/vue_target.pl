% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% vue_target.pl - Vue 3 SFC (Single File Component) Code Generation
%
% Generates Vue 3 components from Prolog predicates. Supports:
% - Composition API with <script setup>
% - TypeScript support
% - Props and emits definitions
% - Lifecycle hooks
% - Reactive state management
%
% Note: Vue is a frontend framework, so this target is primarily for
% generating UI components that consume data from backend services.
% For targets without native Vue support (Go, Rust, C#), use glue
% to generate data endpoints that Vue components can consume.
%
% Usage:
%   ?- compile_predicate_to_vue(my_module:my_predicate/2, [], VueCode).

:- module(vue_target, [
    compile_predicate_to_vue/3,
    compile_component_to_vue/4,
    compile_mindmap_to_vue/4,
    init_vue_target/0,
    test_vue_target/0,
    % Target capabilities
    vue_capabilities/1,
    target_capabilities/1,          % Alias for consistency
    requires_glue/2,

    % Pattern compilation (for ui_patterns integration)
    compile_navigation_pattern/6,
    compile_state_pattern/6,
    compile_data_pattern/5,
    compile_persistence_pattern/5
]).

:- use_module(library(lists)).

% ============================================================================
% TARGET CAPABILITIES
% ============================================================================

%% vue_capabilities(-Capabilities)
%
%  Lists what Vue target can do directly vs needs glue for.
%
vue_capabilities([
    % Direct capabilities
    supports(ui_components),
    supports(mindmap_visualization),
    supports(interactive_graphs),
    supports(viewport_controls),
    supports(form_generation),
    supports(data_display),
    % Requires glue (backend services)
    glue_required(database_queries),
    glue_required(file_io),
    glue_required(heavy_computation),
    glue_required(embedding_generation),
    glue_required(clustering_algorithms)
]).

%% requires_glue(+PredicateSpec, -Reason)
%
%  Check if a predicate requires glue to work with Vue.
%
requires_glue(Pred/Arity, database_access) :-
    member(Pred/Arity, [
        pearl_trees/5, pearl_children/6, pearl_search/5
    ]).
requires_glue(Pred/Arity, computation) :-
    member(Pred/Arity, [
        compute_embedding/2, cluster_trees/3, build_semantic_hierarchy/3
    ]).
requires_glue(Pred/Arity, file_system) :-
    member(Pred/Arity, [
        read_file/2, write_file/2, list_directory/2
    ]).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_vue_target :-
    format('Vue target initialized~n', []).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%% compile_predicate_to_vue(:Pred, +Options, -VueCode)
%
%  Compile a Prolog predicate to a Vue component.
%  This wraps the predicate logic in a Vue component structure.
%
compile_predicate_to_vue(Module:Pred/Arity, Options, VueCode) :-
    (   requires_glue(Pred/Arity, Reason)
    ->  compile_glue_component(Module:Pred/Arity, Reason, Options, VueCode)
    ;   compile_direct_component(Module:Pred/Arity, Options, VueCode)
    ).

%% compile_component_to_vue(+Name, +Props, +Template, -VueCode)
%
%  Generate a Vue SFC from component specification.
%
compile_component_to_vue(_Name, Props, Template, VueCode) :-
    generate_props_interface(Props, PropsInterface),
    generate_props_defaults(Props, PropsDefaults),
    format(string(VueCode),
"<template>
~w
</template>

<script setup lang=\"ts\">
import { ref, computed, onMounted } from 'vue';

~w

const props = withDefaults(defineProps<Props>(), ~w);

const emit = defineEmits<{
  (e: 'update', value: any): void;
}>();

// Component logic here
</script>

<style scoped>
/* Component styles */
</style>
", [Template, PropsInterface, PropsDefaults]).

%% compile_mindmap_to_vue(+Nodes, +Edges, +Options, -VueCode)
%
%  Generate a Vue mindmap visualization component.
%  Delegates to the D3 renderer with Vue output format.
%
compile_mindmap_to_vue(Nodes, Edges, Options, VueCode) :-
    % Use the D3 renderer's Vue generation
    use_module('../mindmap/render/d3_renderer'),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], Options, VueCode).

% ============================================================================
% DIRECT COMPONENT GENERATION
% ============================================================================

compile_direct_component(Module:Pred/Arity, Options, VueCode) :-
    atom_string(Pred, PredStr),
    atom_string(Module, ModStr),
    option_value(Options, component_name, Name, PredStr),
    capitalize_first(Name, ComponentName),

    generate_arg_types(Arity, ArgTypes),

    format(string(VueCode),
"<template>
  <div class=\"~w-component\">
    <h3>~w</h3>
    <div class=\"content\">
      <slot />
    </div>
  </div>
</template>

<script setup lang=\"ts\">
import { ref, computed, onMounted } from 'vue';

// Generated from: ~w:~w/~w

interface Props {
  ~w
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'result', value: any): void;
  (e: 'error', error: Error): void;
}>();

const loading = ref(false);
const result = ref<any>(null);
const error = ref<Error | null>(null);

// Computed properties from predicate logic
const computedResult = computed(() => {
  // TODO: Implement predicate logic
  return result.value;
});

onMounted(() => {
  // Initialize component
});
</script>

<style scoped>
.~w-component {
  padding: 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
}

.~w-component h3 {
  margin: 0 0 1rem 0;
  font-size: 1.25rem;
  color: #1a202c;
}

.~w-component .content {
  color: #4a5568;
}
</style>
", [Name, ComponentName, ModStr, PredStr, Arity, ArgTypes, Name, Name, Name]).

% ============================================================================
% GLUE COMPONENT GENERATION
% ============================================================================

compile_glue_component(Module:Pred/Arity, Reason, Options, VueCode) :-
    atom_string(Pred, PredStr),
    atom_string(Module, ModStr),
    option_value(Options, component_name, Name, PredStr),
    option_value(Options, api_endpoint, Endpoint, "/api/prolog"),
    capitalize_first(Name, _ComponentName),

    % Generate endpoint path from predicate
    format(string(EndpointPath), "~w/~w/~w", [Endpoint, ModStr, PredStr]),

    format(string(VueCode),
"<template>
  <div class=\"~w-component\">
    <div v-if=\"loading\" class=\"loading\">Loading...</div>
    <div v-else-if=\"error\" class=\"error\">{{ error.message }}</div>
    <div v-else class=\"content\">
      <slot :data=\"data\" />
    </div>
  </div>
</template>

<script setup lang=\"ts\">
import { ref, onMounted, watch } from 'vue';

// Generated from: ~w:~w/~w
// Requires glue because: ~w
// This component fetches data from a backend service

interface Props {
  args?: Record<string, any>;
  autoFetch?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  args: () => ({}),
  autoFetch: true
});

const emit = defineEmits<{
  (e: 'data', value: any): void;
  (e: 'error', error: Error): void;
  (e: 'loading', loading: boolean): void;
}>();

const loading = ref(false);
const data = ref<any>(null);
const error = ref<Error | null>(null);

const fetchData = async () => {
  loading.value = true;
  error.value = null;
  emit('loading', true);

  try {
    const response = await fetch('~w', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(props.args)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    data.value = await response.json();
    emit('data', data.value);
  } catch (e) {
    error.value = e as Error;
    emit('error', error.value);
  } finally {
    loading.value = false;
    emit('loading', false);
  }
};

// Watch for arg changes
watch(() => props.args, () => {
  if (props.autoFetch) fetchData();
}, { deep: true });

onMounted(() => {
  if (props.autoFetch) fetchData();
});

// Expose fetch method for manual triggering
defineExpose({ fetchData });
</script>

<style scoped>
.~w-component {
  min-height: 100px;
}

.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #718096;
}

.error {
  color: #e53e3e;
  padding: 1rem;
  background: #fff5f5;
  border-radius: 0.25rem;
}

.content {
  width: 100%;
}
</style>
", [Name, ModStr, PredStr, Arity, Reason, EndpointPath, Name]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

option_value(Options, Key, Value, Default) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HUC]),
    string_chars(Cap, [HUC|T]).

generate_props_interface(Props, Interface) :-
    findall(PropLine, (
        member(prop(Name, Type, _Default), Props),
        format(string(PropLine), "  ~w: ~w;", [Name, Type])
    ), Lines),
    (   Lines = []
    ->  Interface = "interface Props {}"
    ;   atomic_list_concat(Lines, '\n', PropsBody),
        format(string(Interface), "interface Props {~n~w~n}", [PropsBody])
    ).

generate_props_defaults(Props, Defaults) :-
    findall(DefaultLine, (
        member(prop(Name, _Type, Default), Props),
        format(string(DefaultLine), "  ~w: ~w", [Name, Default])
    ), Lines),
    (   Lines = []
    ->  Defaults = "{}"
    ;   atomic_list_concat(Lines, ',\n', DefaultsBody),
        format(string(Defaults), "{~n~w~n}", [DefaultsBody])
    ).

generate_arg_types(0, "// No arguments") :- !.
generate_arg_types(Arity, Types) :-
    Arity > 0,
    findall(ArgLine, (
        between(1, Arity, N),
        format(string(ArgLine), "arg~w?: any;", [N])
    ), Lines),
    atomic_list_concat(Lines, '\n  ', Types).

% Alias for target_capabilities
target_capabilities(Caps) :- vue_capabilities(Caps).

% ============================================================================
% PATTERN COMPILATION - Navigation
% ============================================================================
%
% Vue uses vue-router for navigation. Unlike mobile frameworks, Vue apps
% typically use URL-based routing rather than screen stacks.

%% compile_navigation_pattern(+Type, +Screens, +Config, +Target, +Options, -Code)
compile_navigation_pattern(stack, Screens, _Config, vue, Options, Code) :-
    option_value(Options, component_name, Name, 'AppRouter'),
    generate_vue_router(Screens, stack, Name, Code).
compile_navigation_pattern(tab, Screens, _Config, vue, Options, Code) :-
    option_value(Options, component_name, Name, 'TabRouter'),
    generate_vue_tab_navigation(Screens, Name, Code).
compile_navigation_pattern(drawer, Screens, _Config, vue, Options, Code) :-
    option_value(Options, component_name, Name, 'DrawerLayout'),
    generate_vue_drawer_navigation(Screens, Name, Code).

generate_vue_router(Screens, _Type, Name, Code) :-
    generate_vue_route_definitions(Screens, RouteDefs),
    generate_vue_route_imports(Screens, Imports),
    format(string(Code),
"// ~w - Vue Router Configuration
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
~w

const routes: RouteRecordRaw[] = [
~w
];

export const ~w = createRouter({
  history: createWebHistory(),
  routes,
});

export default ~w;
", [Name, Imports, RouteDefs, Name, Name]).

generate_vue_route_definitions(Screens, Defs) :-
    findall(RouteDef, (
        member(screen(ScreenName, Component, Opts), Screens),
        (   member(title(Title), Opts) -> true ; Title = ScreenName ),
        atom_string(ScreenName, NameStr),
        format(string(RouteDef),
"  {
    path: '/~w',
    name: '~w',
    component: ~w,
    meta: { title: '~w' },
  }", [NameStr, NameStr, Component, Title])
    ), RouteDefList),
    atomic_list_concat(RouteDefList, ',\n', Defs).

generate_vue_route_imports(Screens, Imports) :-
    findall(ImportLine, (
        member(screen(_, Component, _), Screens),
        format(string(ImportLine), "import ~w from '@/views/~w.vue';", [Component, Component])
    ), ImportLines),
    atomic_list_concat(ImportLines, '\n', Imports).

generate_vue_tab_navigation(Screens, Name, Code) :-
    generate_vue_tab_items(Screens, TabItems),
    format(string(Code),
"<template>
  <div class=\"~w\">
    <nav class=\"tab-bar\">
~w
    </nav>
    <main class=\"tab-content\">
      <router-view />
    </main>
  </div>
</template>

<script setup lang=\"ts\">
import { useRoute } from 'vue-router';

const route = useRoute();
</script>

<style scoped>
.~w {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.tab-bar {
  display: flex;
  border-bottom: 1px solid #e2e8f0;
  background: #fff;
}

.tab-bar a {
  flex: 1;
  padding: 1rem;
  text-align: center;
  text-decoration: none;
  color: #4a5568;
  border-bottom: 2px solid transparent;
}

.tab-bar a.router-link-active {
  color: #3182ce;
  border-bottom-color: #3182ce;
}

.tab-content {
  flex: 1;
  overflow: auto;
}
</style>
", [Name, TabItems, Name]).

generate_vue_tab_items(Screens, Items) :-
    findall(TabItem, (
        member(screen(ScreenName, _, Opts), Screens),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        atom_string(ScreenName, NameStr),
        format(string(TabItem),
"      <router-link to=\"/~w\">~w</router-link>", [NameStr, Title])
    ), TabItemList),
    atomic_list_concat(TabItemList, '\n', Items).

generate_vue_drawer_navigation(Screens, Name, Code) :-
    generate_vue_drawer_items(Screens, DrawerItems),
    format(string(Code),
"<template>
  <div class=\"~w\">
    <aside class=\"drawer\" :class=\"{ open: isOpen }\">
      <nav class=\"drawer-nav\">
~w
      </nav>
    </aside>
    <div class=\"drawer-overlay\" v-if=\"isOpen\" @click=\"isOpen = false\" />
    <main class=\"drawer-content\">
      <button class=\"drawer-toggle\" @click=\"isOpen = !isOpen\">
        <span class=\"hamburger\"></span>
      </button>
      <router-view />
    </main>
  </div>
</template>

<script setup lang=\"ts\">
import { ref } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();
const isOpen = ref(false);
</script>

<style scoped>
.~w {
  display: flex;
  height: 100vh;
}

.drawer {
  position: fixed;
  left: -280px;
  width: 280px;
  height: 100vh;
  background: #fff;
  box-shadow: 2px 0 8px rgba(0,0,0,0.1);
  transition: left 0.3s ease;
  z-index: 100;
}

.drawer.open {
  left: 0;
}

.drawer-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.3);
  z-index: 99;
}

.drawer-nav a {
  display: block;
  padding: 1rem 1.5rem;
  color: #4a5568;
  text-decoration: none;
  border-bottom: 1px solid #e2e8f0;
}

.drawer-nav a.router-link-active {
  background: #ebf8ff;
  color: #3182ce;
}

.drawer-content {
  flex: 1;
  overflow: auto;
}

.drawer-toggle {
  position: fixed;
  top: 1rem;
  left: 1rem;
  z-index: 50;
  padding: 0.5rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 0.25rem;
  cursor: pointer;
}
</style>
", [Name, DrawerItems, Name]).

generate_vue_drawer_items(Screens, Items) :-
    findall(DrawerItem, (
        member(screen(ScreenName, _, Opts), Screens),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        atom_string(ScreenName, NameStr),
        format(string(DrawerItem),
"        <router-link to=\"/~w\" @click=\"isOpen = false\">~w</router-link>", [NameStr, Title])
    ), DrawerItemList),
    atomic_list_concat(DrawerItemList, '\n', Items).

% ============================================================================
% PATTERN COMPILATION - State
% ============================================================================
%
% Vue uses the Composition API (ref, reactive) for local state and
% Pinia for global state management.

%% compile_state_pattern(+Type, +Shape, +Config, +Target, +Options, -Code)
compile_state_pattern(local, Shape, _Config, vue, _Options, Code) :-
    generate_vue_local_state(Shape, Code).
compile_state_pattern(global, Shape, _Config, vue, _Options, Code) :-
    generate_vue_pinia_store(Shape, Code).
compile_state_pattern(derived, Shape, _Config, vue, _Options, Code) :-
    generate_vue_computed_state(Shape, Code).

generate_vue_local_state(Shape, Code) :-
    findall(RefDef, (
        member(field(Name, Initial), Shape),
        format(string(RefDef), "const ~w = ref(~w);", [Name, Initial])
    ), RefDefs),
    atomic_list_concat(RefDefs, '\n', RefsCode),
    format(string(Code),
"// Vue Composition API - Local State
import { ref } from 'vue';

// Usage in <script setup>:
~w
", [RefsCode]).

generate_vue_pinia_store(Shape, Code) :-
    member(store(StoreName), Shape),
    member(slices(Slices), Shape),
    generate_pinia_state(Slices, StateDefs),
    generate_pinia_getters(Slices, GetterDefs),
    generate_pinia_actions(Slices, ActionDefs),
    format(string(Code),
"// Pinia Store - ~w
import { defineStore } from 'pinia';

export const use~wStore = defineStore('~w', {
  state: () => ({
~w
  }),

  getters: {
~w
  },

  actions: {
~w
  },
});
", [StoreName, StoreName, StoreName, StateDefs, GetterDefs, ActionDefs]).

generate_pinia_state(Slices, StateDefs) :-
    findall(StateDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType), Fields),
        pinia_default_value(FType, Default),
        format(string(StateDef), "    ~w: ~w,", [FName, Default])
    ), StateDefList),
    atomic_list_concat(StateDefList, '\n', StateDefs).

pinia_default_value("'light' | 'dark'", "'light'").
pinia_default_value("boolean", "false").
pinia_default_value("string | null", "null").
pinia_default_value("number", "0").
pinia_default_value(_, "null").

generate_pinia_getters(_Slices, Getters) :-
    Getters = "    // Add computed getters here".

generate_pinia_actions(Slices, ActionDefs) :-
    findall(ActionDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, _ABody), Actions),
        generate_pinia_action(AName, ActionDef)
    ), ActionDefList),
    (   ActionDefList = []
    ->  ActionDefs = "    // Add actions here"
    ;   atomic_list_concat(ActionDefList, '\n', ActionDefs)
    ).

generate_pinia_action(toggleTheme, Code) :-
    format(string(Code),
"    toggleTheme() {
      this.theme = this.theme === 'light' ? 'dark' : 'light';
    },", []).
generate_pinia_action(toggleSidebar, Code) :-
    format(string(Code),
"    toggleSidebar() {
      this.sidebarOpen = !this.sidebarOpen;
    },", []).
generate_pinia_action(Name, Code) :-
    \+ member(Name, [toggleTheme, toggleSidebar]),
    format(string(Code),
"    ~w(value: unknown) {
      // TODO: Implement ~w action
    },", [Name, Name]).

generate_vue_computed_state(Shape, Code) :-
    member(deps(Deps), Shape),
    member(derive(Derivation), Shape),
    atomic_list_concat(Deps, ', ', DepsStr),
    format(string(Code),
"// Vue Composition API - Computed State
import { computed } from 'vue';

// Usage in <script setup>:
// Assumes ~w are defined as refs
const derivedValue = computed(() => {
  return ~w;
});
", [DepsStr, Derivation]).

% ============================================================================
% PATTERN COMPILATION - Data Fetching
% ============================================================================
%
% Vue uses @tanstack/vue-query for data fetching, similar to React Query.

%% compile_data_pattern(+Type, +Config, +Target, +Options, -Code)
compile_data_pattern(query, Config, vue, _Options, Code) :-
    generate_vue_query_composable(Config, Code).
compile_data_pattern(mutation, Config, vue, _Options, Code) :-
    generate_vue_mutation_composable(Config, Code).
compile_data_pattern(infinite, Config, vue, _Options, Code) :-
    generate_vue_infinite_query(Config, Code).

generate_vue_query_composable(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(stale_time(StaleTime), Config) -> true ; StaleTime = 300000 ),
    capitalize_first(Name, HookName),
    format(string(Code),
"// Vue Query Composable - ~w
import { useQuery } from '@tanstack/vue-query';
import type { Ref } from 'vue';

interface ~wData {
  // TODO: Define response type
  [key: string]: unknown;
}

interface ~wVariables {
  // TODO: Define variables type
  [key: string]: unknown;
}

export function use~w(variables?: Ref<~wVariables> | ~wVariables) {
  return useQuery({
    queryKey: ['~w', variables],
    queryFn: async () => {
      const response = await fetch('~w', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json() as Promise<~wData>;
    },
    staleTime: ~w,
  });
}
", [Name, Name, Name, HookName, Name, Name, Name, Endpoint, Name, StaleTime]).

generate_vue_mutation_composable(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    capitalize_first(Name, HookName),
    format(string(Code),
"// Vue Mutation Composable - ~w
import { useMutation, useQueryClient } from '@tanstack/vue-query';

interface ~wVariables {
  // TODO: Define input type
  [key: string]: unknown;
}

interface ~wResponse {
  // TODO: Define response type
  [key: string]: unknown;
}

export function use~w() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (variables: ~wVariables) => {
      const response = await fetch('~w', {
        method: '~w',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(variables),
      });
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json() as Promise<~wResponse>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['~w'] });
    },
  });
}
", [Name, Name, Name, HookName, Name, Endpoint, Method, Name, Name]).

generate_vue_infinite_query(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'page' ),
    capitalize_first(Name, HookName),
    format(string(Code),
"// Vue Infinite Query Composable - ~w
import { useInfiniteQuery } from '@tanstack/vue-query';

interface ~wPage {
  data: unknown[];
  nextPage?: number;
  hasMore: boolean;
}

export function use~w() {
  return useInfiniteQuery({
    queryKey: ['~w'],
    queryFn: async ({ pageParam = 1 }) => {
      const response = await fetch(`~w?~w=${pageParam}`);
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json() as Promise<~wPage>;
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.nextPage : undefined,
  });
}
", [Name, Name, HookName, Name, Endpoint, PageParam, Name]).

% ============================================================================
% PATTERN COMPILATION - Persistence
% ============================================================================
%
% Vue uses localStorage/sessionStorage or plugins like pinia-plugin-persistedstate.

%% compile_persistence_pattern(+Type, +Config, +Target, +Options, -Code)
compile_persistence_pattern(local, Config, vue, _Options, Code) :-
    generate_vue_storage_composable(Config, Code).
compile_persistence_pattern(secure, Config, vue, _Options, Code) :-
    % For Vue web, "secure" storage uses the same localStorage but with encryption note
    generate_vue_secure_storage_composable(Config, Code).

generate_vue_storage_composable(Config, Code) :-
    member(key(Key), Config),
    (   member(schema(Schema), Config) -> true ; Schema = 'unknown' ),
    capitalize_first(Key, HookName),
    format(string(Code),
"// Vue Storage Composable - ~w
import { ref, watch, onMounted } from 'vue';

const STORAGE_KEY = '~w';

type ~wData = ~w;

export function use~w() {
  const data = ref<~wData | null>(null);
  const loading = ref(true);
  const error = ref<Error | null>(null);

  const load = () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        data.value = JSON.parse(stored);
      }
    } catch (e) {
      error.value = e as Error;
    } finally {
      loading.value = false;
    }
  };

  const save = (value: ~wData) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
      data.value = value;
    } catch (e) {
      error.value = e as Error;
      throw e;
    }
  };

  const remove = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      data.value = null;
    } catch (e) {
      error.value = e as Error;
      throw e;
    }
  };

  onMounted(load);

  return { data, loading, error, save, remove };
}
", [Key, Key, Key, Schema, HookName, Key, Key]).

generate_vue_secure_storage_composable(Config, Code) :-
    member(key(Key), Config),
    (   member(schema(Schema), Config) -> true ; Schema = 'unknown' ),
    capitalize_first(Key, HookName),
    format(string(Code),
"// Vue Secure Storage Composable - ~w
// Note: For true secure storage in web apps, consider using
// encrypted localStorage or a secure backend API.
import { ref, onMounted } from 'vue';

const STORAGE_KEY = '~w';

type ~wData = ~w;

export function use~w() {
  const data = ref<~wData | null>(null);
  const loading = ref(true);
  const error = ref<Error | null>(null);

  const load = () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        // TODO: Add decryption here for secure storage
        data.value = JSON.parse(stored);
      }
    } catch (e) {
      error.value = e as Error;
    } finally {
      loading.value = false;
    }
  };

  const save = (value: ~wData) => {
    try {
      // TODO: Add encryption here for secure storage
      localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
      data.value = value;
    } catch (e) {
      error.value = e as Error;
      throw e;
    }
  };

  const remove = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      data.value = null;
    } catch (e) {
      error.value = e as Error;
      throw e;
    }
  };

  onMounted(load);

  return { data, loading, error, save, remove };
}
", [Key, Key, Key, Schema, HookName, Key]).

% ============================================================================
% TESTING
% ============================================================================

test_vue_target :-
    format('~n=== Vue Target Tests ===~n~n'),

    % Test 1: Direct component generation
    format('Test 1: Direct component generation...~n'),
    compile_predicate_to_vue(test_module:test_pred/2, [], Code1),
    (   sub_string(Code1, _, _, _, "<template>"),
        sub_string(Code1, _, _, _, "<script setup")
    ->  format('  PASS: Direct component generated~n')
    ;   format('  FAIL: Direct component incorrect~n')
    ),

    % Test 2: Glue component generation
    format('~nTest 2: Glue component generation...~n'),
    compile_predicate_to_vue(pearltrees:pearl_trees/5, [], Code2),
    (   sub_string(Code2, _, _, _, "Requires glue"),
        sub_string(Code2, _, _, _, "fetch(")
    ->  format('  PASS: Glue component generated~n')
    ;   format('  FAIL: Glue component incorrect~n')
    ),

    % Test 3: Capabilities check
    format('~nTest 3: Capabilities check...~n'),
    vue_capabilities(Caps),
    (   member(supports(mindmap_visualization), Caps),
        member(glue_required(database_queries), Caps)
    ->  format('  PASS: Capabilities correctly defined~n')
    ;   format('  FAIL: Capabilities incorrect~n')
    ),

    % Test 4: Requires glue check
    format('~nTest 4: Requires glue check...~n'),
    (   requires_glue(pearl_trees/5, database_access),
        requires_glue(compute_embedding/2, computation),
        \+ requires_glue(unknown_pred/1, _)
    ->  format('  PASS: Glue requirements correct~n')
    ;   format('  FAIL: Glue requirements incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Vue target module loaded~n', [])
), now).

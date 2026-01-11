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
    requires_glue/2
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
compile_component_to_vue(Name, Props, Template, VueCode) :-
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
    capitalize_first(Name, ComponentName),

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

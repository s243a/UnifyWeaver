% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% cpp_bindings.pl - C++-specific bindings for UnifyWeaver
%
% C++ bindings for STL, iostream, algorithms, and nlohmann/json.

:- encoding(utf8).

:- module(cpp_bindings, [
    init_cpp_bindings/0,
    cpp_binding/5,
    cpp_binding_include/2,      % cpp_binding_include(Pred, Header)
    test_cpp_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_cpp_bindings
init_cpp_bindings :-
    register_stl_bindings,
    register_iostream_bindings,
    register_algorithm_bindings,
    register_json_bindings.

%% cpp_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
cpp_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(cpp, Pred, TargetName, Inputs, Outputs, Options).

%% cpp_binding_include(?Pred, ?Header)
%  Get the #include header required for a C++ binding.
cpp_binding_include(Pred, Header) :-
    cpp_binding(Pred, _, _, _, Options),
    member(include(Header), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- cpp_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined C++ bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- cpp_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(cpp, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% STL CONTAINER BINDINGS
% ============================================================================

register_stl_bindings :-
    % Vector
    declare_binding(cpp, vector_push_back/2, 'push_back',
        [any], [],
        [effect(mutate), deterministic, total,
         include('<vector>')]),
    
    declare_binding(cpp, vector_pop_back/1, 'pop_back',
        [], [],
        [effect(mutate), deterministic, partial,
         include('<vector>')]),
    
    declare_binding(cpp, vector_size/2, 'size',
        [], [size_t],
        [pure, deterministic, total,
         include('<vector>')]),
    
    declare_binding(cpp, vector_empty/1, 'empty',
        [], [bool],
        [pure, deterministic, total,
         include('<vector>')]),
    
    declare_binding(cpp, vector_at/3, 'at',
        [size_t], [any],
        [pure, deterministic, partial,
         include('<vector>')]),
    
    declare_binding(cpp, vector_front/2, 'front',
        [], [any],
        [pure, deterministic, partial,
         include('<vector>')]),
    
    declare_binding(cpp, vector_back/2, 'back',
        [], [any],
        [pure, deterministic, partial,
         include('<vector>')]),
    
    % Map
    declare_binding(cpp, map_insert/3, 'insert',
        [key, value], [],
        [effect(mutate), deterministic, total,
         include('<map>')]),
    
    declare_binding(cpp, map_find/3, 'find',
        [key], [iterator],
        [pure, deterministic, total,
         include('<map>')]),
    
    declare_binding(cpp, map_erase/2, 'erase',
        [key], [size_t],
        [effect(mutate), deterministic, total,
         include('<map>')]),
    
    declare_binding(cpp, map_count/3, 'count',
        [key], [size_t],
        [pure, deterministic, total,
         include('<map>')]),
    
    % String
    declare_binding(cpp, string_length/2, 'length',
        [], [size_t],
        [pure, deterministic, total,
         include('<string>')]),
    
    declare_binding(cpp, string_substr/4, 'substr',
        [size_t, size_t], [string],
        [pure, deterministic, partial,
         include('<string>')]),
    
    declare_binding(cpp, string_find/3, 'find',
        [string], [size_t],
        [pure, deterministic, total,
         include('<string>')]),
    
    declare_binding(cpp, string_append/2, 'append',
        [string], [],
        [effect(mutate), deterministic, total,
         include('<string>')]),
    
    % Optional
    declare_binding(cpp, optional_value/2, 'value',
        [], [any],
        [pure, deterministic, partial,
         include('<optional>')]),
    
    declare_binding(cpp, optional_value_or/3, 'value_or',
        [any], [any],
        [pure, deterministic, total,
         include('<optional>')]),
    
    declare_binding(cpp, optional_has_value/1, 'has_value',
        [], [bool],
        [pure, deterministic, total,
         include('<optional>')]).

% ============================================================================
% IOSTREAM BINDINGS
% ============================================================================

register_iostream_bindings :-
    declare_binding(cpp, cout/1, 'std::cout <<',
        [any], [],
        [effect(io), deterministic, total,
         include('<iostream>')]),
    
    declare_binding(cpp, cerr/1, 'std::cerr <<',
        [any], [],
        [effect(io), deterministic, total,
         include('<iostream>')]),
    
    declare_binding(cpp, cin/1, 'std::cin >>',
        [], [any],
        [effect(io), deterministic, partial,
         include('<iostream>')]),
    
    declare_binding(cpp, getline/3, 'std::getline',
        [istream, string], [istream],
        [effect(io), deterministic, partial,
         include('<iostream>')]),
    
    declare_binding(cpp, endl/0, 'std::endl',
        [], [],
        [effect(io), deterministic, total,
         include('<iostream>')]),
    
    % File streams
    declare_binding(cpp, ifstream_open/2, 'std::ifstream',
        [string], [ifstream],
        [effect(io), deterministic, partial,
         include('<fstream>')]),
    
    declare_binding(cpp, ofstream_open/2, 'std::ofstream',
        [string], [ofstream],
        [effect(io), deterministic, partial,
         include('<fstream>')]).

% ============================================================================
% ALGORITHM BINDINGS
% ============================================================================

register_algorithm_bindings :-
    declare_binding(cpp, std_find/4, 'std::find',
        [iterator, iterator, value], [iterator],
        [pure, deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_find_if/4, 'std::find_if',
        [iterator, iterator, predicate], [iterator],
        [pure, deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_sort/3, 'std::sort',
        [iterator, iterator], [],
        [effect(mutate), deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_transform/5, 'std::transform',
        [iterator, iterator, iterator, function], [iterator],
        [effect(write), deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_accumulate/5, 'std::accumulate',
        [iterator, iterator, init, function], [any],
        [pure, deterministic, total,
         include('<numeric>')]),
    
    declare_binding(cpp, std_for_each/4, 'std::for_each',
        [iterator, iterator, function], [],
        [effect(apply), deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_count_if/4, 'std::count_if',
        [iterator, iterator, predicate], [size_t],
        [pure, deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_any_of/4, 'std::any_of',
        [iterator, iterator, predicate], [bool],
        [pure, deterministic, total,
         include('<algorithm>')]),
    
    declare_binding(cpp, std_all_of/4, 'std::all_of',
        [iterator, iterator, predicate], [bool],
        [pure, deterministic, total,
         include('<algorithm>')]).

% ============================================================================
% NLOHMANN/JSON BINDINGS
% ============================================================================

register_json_bindings :-
    declare_binding(cpp, json_parse/2, 'nlohmann::json::parse',
        [string], [json],
        [pure, deterministic, partial,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_dump/2, 'dump',
        [], [string],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_dump_pretty/3, 'dump',
        [int], [string],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_at/3, 'at',
        [string], [json],
        [pure, deterministic, partial,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_value/4, 'value',
        [string, default], [any],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_contains/3, 'contains',
        [string], [bool],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_is_null/1, 'is_null',
        [], [bool],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_is_array/1, 'is_array',
        [], [bool],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_is_object/1, 'is_object',
        [], [bool],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_size/2, 'size',
        [], [size_t],
        [pure, deterministic, total,
         include('<nlohmann/json.hpp>')]),
    
    declare_binding(cpp, json_push_back/2, 'push_back',
        [json], [],
        [effect(mutate), deterministic, total,
         include('<nlohmann/json.hpp>')]).

% ============================================================================
% TESTS
% ============================================================================

test_cpp_bindings :-
    format('~n=== C++ Bindings Tests ===~n~n'),

    format('[Test 1] Initializing C++ bindings~n'),
    init_cpp_bindings,
    format('  [PASS] C++ bindings initialized~n'),

    format('~n[Test 2] Checking STL bindings~n'),
    (   cpp_binding(vector_push_back/2, 'push_back', _, _, _)
    ->  format('  [PASS] vector::push_back binding exists~n')
    ;   format('  [FAIL] vector::push_back binding missing~n')
    ),

    format('~n[Test 3] Checking nlohmann/json bindings~n'),
    (   cpp_binding(json_parse/2, 'nlohmann::json::parse', _, _, _)
    ->  format('  [PASS] json::parse binding exists~n')
    ;   format('  [FAIL] json::parse binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, cpp_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total C++ bindings: ~w~n', [Count]),

    format('~n=== C++ Bindings Tests Complete ===~n').

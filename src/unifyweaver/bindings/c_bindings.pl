% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% c_bindings.pl - C-specific bindings for UnifyWeaver
%
% C bindings for stdlib, I/O, strings, and cJSON.

:- encoding(utf8).

:- module(c_bindings, [
    init_c_bindings/0,
    c_binding/5,
    c_binding_include/2,        % c_binding_include(Pred, Header)
    test_c_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_c_bindings
init_c_bindings :-
    register_stdlib_bindings,
    register_io_bindings,
    register_string_bindings,
    register_cjson_bindings.

%% c_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
c_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(c, Pred, TargetName, Inputs, Outputs, Options).

%% c_binding_include(?Pred, ?Header)
%  Get the #include header required for a C binding.
c_binding_include(Pred, Header) :-
    c_binding(Pred, _, _, _, Options),
    member(include(Header), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- c_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined C bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- c_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(c, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% STDLIB BINDINGS
% ============================================================================

register_stdlib_bindings :-
    declare_binding(c, malloc/2, 'malloc',
        [size_t], [ptr],
        [effect(alloc), deterministic, partial,
         include('<stdlib.h>')]),
    
    declare_binding(c, calloc/3, 'calloc',
        [size_t, size_t], [ptr],
        [effect(alloc), deterministic, partial,
         include('<stdlib.h>')]),
    
    declare_binding(c, realloc/3, 'realloc',
        [ptr, size_t], [ptr],
        [effect(alloc), deterministic, partial,
         include('<stdlib.h>')]),
    
    declare_binding(c, free/1, 'free',
        [ptr], [],
        [effect(dealloc), deterministic, total,
         include('<stdlib.h>')]),
    
    declare_binding(c, exit/1, 'exit',
        [int], [],
        [effect(termination), deterministic, total,
         include('<stdlib.h>')]),
    
    declare_binding(c, atoi/2, 'atoi',
        [string], [int],
        [pure, deterministic, total,
         include('<stdlib.h>')]),
    
    declare_binding(c, atof/2, 'atof',
        [string], [double],
        [pure, deterministic, total,
         include('<stdlib.h>')]).

% ============================================================================
% I/O BINDINGS
% ============================================================================

register_io_bindings :-
    declare_binding(c, printf/2, 'printf',
        [string, vararg], [int],
        [effect(io), deterministic, total,
         include('<stdio.h>')]),
    
    declare_binding(c, fprintf/3, 'fprintf',
        [file, string, vararg], [int],
        [effect(io), deterministic, total,
         include('<stdio.h>')]),
    
    declare_binding(c, sprintf/3, 'sprintf',
        [buffer, string, vararg], [int],
        [effect(write), deterministic, total,
         include('<stdio.h>')]),
    
    declare_binding(c, snprintf/4, 'snprintf',
        [buffer, size_t, string, vararg], [int],
        [effect(write), deterministic, total,
         include('<stdio.h>')]),
    
    declare_binding(c, fgets/4, 'fgets',
        [buffer, int, file], [string],
        [effect(io), deterministic, partial,
         include('<stdio.h>')]),
    
    declare_binding(c, fopen/3, 'fopen',
        [string, string], [file],
        [effect(io), deterministic, partial,
         include('<stdio.h>')]),
    
    declare_binding(c, fclose/2, 'fclose',
        [file], [int],
        [effect(io), deterministic, total,
         include('<stdio.h>')]),
    
    declare_binding(c, fread/5, 'fread',
        [buffer, size_t, size_t, file], [size_t],
        [effect(io), deterministic, partial,
         include('<stdio.h>')]),
    
    declare_binding(c, fwrite/5, 'fwrite',
        [buffer, size_t, size_t, file], [size_t],
        [effect(io), deterministic, partial,
         include('<stdio.h>')]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(c, strlen/2, 'strlen',
        [string], [size_t],
        [pure, deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strcmp/3, 'strcmp',
        [string, string], [int],
        [pure, deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strncmp/4, 'strncmp',
        [string, string, size_t], [int],
        [pure, deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strcpy/3, 'strcpy',
        [dest, src], [string],
        [effect(write), deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strncpy/4, 'strncpy',
        [dest, src, size_t], [string],
        [effect(write), deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strcat/3, 'strcat',
        [dest, src], [string],
        [effect(write), deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, strdup/2, 'strdup',
        [string], [string],
        [effect(alloc), deterministic, partial,
         include('<string.h>')]),
    
    declare_binding(c, strtok/3, 'strtok',
        [string, delim], [string],
        [effect(state), deterministic, partial,
         include('<string.h>')]),
    
    declare_binding(c, strstr/3, 'strstr',
        [haystack, needle], [string],
        [pure, deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, memcpy/4, 'memcpy',
        [dest, src, size_t], [ptr],
        [effect(write), deterministic, total,
         include('<string.h>')]),
    
    declare_binding(c, memset/4, 'memset',
        [ptr, int, size_t], [ptr],
        [effect(write), deterministic, total,
         include('<string.h>')]).

% ============================================================================
% cJSON BINDINGS
% ============================================================================

register_cjson_bindings :-
    declare_binding(c, cjson_parse/2, 'cJSON_Parse',
        [string], [cjson],
        [effect(alloc), deterministic, partial,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_delete/1, 'cJSON_Delete',
        [cjson], [],
        [effect(dealloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_print/2, 'cJSON_Print',
        [cjson], [string],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_print_unformatted/2, 'cJSON_PrintUnformatted',
        [cjson], [string],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_get_object_item/3, 'cJSON_GetObjectItem',
        [cjson, string], [cjson],
        [pure, deterministic, partial,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_get_array_item/3, 'cJSON_GetArrayItem',
        [cjson, int], [cjson],
        [pure, deterministic, partial,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_get_array_size/2, 'cJSON_GetArraySize',
        [cjson], [int],
        [pure, deterministic, total,
         include('"cJSON.h"')]),
        
    declare_binding(c, cjson_create_object/1, 'cJSON_CreateObject',
        [], [cjson],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_create_array/1, 'cJSON_CreateArray',
        [], [cjson],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_create_string/2, 'cJSON_CreateString',
        [string], [cjson],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_create_number/2, 'cJSON_CreateNumber',
        [double], [cjson],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_add_item_to_object/3, 'cJSON_AddItemToObject',
        [cjson, string, cjson], [],
        [effect(mutate), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_add_item_to_array/2, 'cJSON_AddItemToArray',
        [cjson, cjson], [],
        [effect(mutate), deterministic, total,
         include('"cJSON.h"')]),
    
    declare_binding(c, cjson_duplicate/3, 'cJSON_Duplicate',
        [cjson, int], [cjson],
        [effect(alloc), deterministic, total,
         include('"cJSON.h"')]).

% ============================================================================
% TESTS
% ============================================================================

test_c_bindings :-
    format('~n=== C Bindings Tests ===~n~n'),

    format('[Test 1] Initializing C bindings~n'),
    init_c_bindings,
    format('  [PASS] C bindings initialized~n'),

    format('~n[Test 2] Checking stdlib bindings~n'),
    (   c_binding(malloc/2, 'malloc', _, _, _)
    ->  format('  [PASS] malloc binding exists~n')
    ;   format('  [FAIL] malloc binding missing~n')
    ),

    format('~n[Test 3] Checking cJSON bindings~n'),
    (   c_binding(cjson_parse/2, 'cJSON_Parse', _, _, _)
    ->  format('  [PASS] cJSON_Parse binding exists~n')
    ;   format('  [FAIL] cJSON_Parse binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, c_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total C bindings: ~w~n', [Count]),

    format('~n=== C Bindings Tests Complete ===~n').

:- module(ruby_bindings, [
    init_ruby_bindings/0,
    rb_binding/5,
    rb_binding_import/2,
    test_ruby_bindings/0
]).

:- use_module('../core/binding_registry').

%% ============================================================================
%% INITIALIZATION
%% ============================================================================

init_ruby_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_array_bindings,
    register_hash_bindings,
    register_io_bindings,
    register_regex_bindings,
    register_math_bindings,
    register_conversion_bindings.

%% ============================================================================
%% CONVENIENCE PREDICATES
%% ============================================================================

%% rb_binding(+Pred, -TargetName, -Inputs, -Outputs, -Options)
%% Query Ruby bindings
rb_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(ruby, Pred, TargetName, Inputs, Outputs, Options).

%% rb_binding_import(+Pred, -Import)
%% Get required import for a predicate
rb_binding_import(Pred, Import) :-
    rb_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

%% ============================================================================
%% DIRECTIVE SUPPORT (User-defined bindings)
%% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- rb_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(ruby, Pred, TargetName, Inputs, Outputs, Options)))
).

%% ============================================================================
%% CORE BUILT-IN BINDINGS
%% ============================================================================

register_builtin_bindings :-
    % Type inspection
    declare_binding(ruby, length/2, '.length', [sequence], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, size/2, '.size', [sequence], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, empty/1, '.empty?', [sequence], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, nil/1, '.nil?', [any], [],
                   [pure, deterministic, total, pattern(method_call)]),

    % Object identity
    declare_binding(ruby, class/2, '.class', [any], [class],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, is_a/2, '.is_a?', [any, class], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, kind_of/2, '.kind_of?', [any, class], [],
                   [pure, deterministic, total, pattern(method_call)]),

    % Cloning
    declare_binding(ruby, dup/2, '.dup', [any], [any],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, clone/2, '.clone', [any], [any],
                   [pure, deterministic, total, pattern(method_call)]),

    % Freezing
    declare_binding(ruby, freeze/1, '.freeze', [any], [],
                   [effect(state), deterministic, total, pattern(method_call)]),
    declare_binding(ruby, frozen/1, '.frozen?', [any], [],
                   [pure, deterministic, total, pattern(method_call)]).

%% ============================================================================
%% STRING OPERATION BINDINGS
%% ============================================================================

register_string_bindings :-
    % Basic string operations
    declare_binding(ruby, string_length/2, '.length', [string], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_upcase/2, '.upcase', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_downcase/2, '.downcase', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_capitalize/2, '.capitalize', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_swapcase/2, '.swapcase', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % Trimming
    declare_binding(ruby, string_strip/2, '.strip', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_lstrip/2, '.lstrip', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_rstrip/2, '.rstrip', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_chomp/2, '.chomp', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_chop/2, '.chop', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % Splitting and joining
    declare_binding(ruby, string_split/2, '.split', [string], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_split/3, '.split', [string, string], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_lines/2, '.lines', [string], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_chars/2, '.chars', [string], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_bytes/2, '.bytes', [string], [array],
                   [pure, deterministic, total, pattern(method_call)]),

    % Searching
    declare_binding(ruby, string_include/2, '.include?', [string, string], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_start_with/2, '.start_with?', [string, string], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_end_with/2, '.end_with?', [string, string], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_index/3, '.index', [string, string], [int],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, string_rindex/3, '.rindex', [string, string], [int],
                   [pure, deterministic, partial, pattern(method_call)]),

    % Replacement
    declare_binding(ruby, string_sub/4, '.sub', [string, string, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_gsub/4, '.gsub', [string, string, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_tr/4, '.tr', [string, string, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, string_delete/3, '.delete', [string, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % Substring
    declare_binding(ruby, string_slice/4, '.slice', [string, int, int], [string],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, string_reverse/2, '.reverse', [string], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % Concatenation
    declare_binding(ruby, string_concat/3, '+', [string, string], [string],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, string_repeat/3, '*', [string, int], [string],
                   [pure, deterministic, total, pattern(operator)]).

%% ============================================================================
%% ARRAY OPERATION BINDINGS
%% ============================================================================

register_array_bindings :-
    % Basic array operations
    declare_binding(ruby, array_length/2, '.length', [array], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_first/2, '.first', [array], [any],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, array_last/2, '.last', [array], [any],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, array_at/3, '.at', [array, int], [any],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, array_fetch/3, '.fetch', [array, int], [any],
                   [pure, deterministic, partial, effect(throws), pattern(method_call)]),

    % Adding elements
    declare_binding(ruby, array_push/2, '.push', [array, any], [],
                   [effect(state), deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_append/2, '<<', [array, any], [],
                   [effect(state), deterministic, total, pattern(operator)]),
    declare_binding(ruby, array_unshift/2, '.unshift', [array, any], [],
                   [effect(state), deterministic, total, pattern(method_call)]),

    % Removing elements
    declare_binding(ruby, array_pop/2, '.pop', [array], [any],
                   [effect(state), deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, array_shift/2, '.shift', [array], [any],
                   [effect(state), deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, array_delete/2, '.delete', [array, any], [],
                   [effect(state), deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_delete_at/3, '.delete_at', [array, int], [any],
                   [effect(state), deterministic, partial, pattern(method_call)]),

    % Transformations (pure - return new array)
    declare_binding(ruby, array_reverse/2, '.reverse', [array], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_sort/2, '.sort', [array], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_uniq/2, '.uniq', [array], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_compact/2, '.compact', [array], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_flatten/2, '.flatten', [array], [array],
                   [pure, deterministic, total, pattern(method_call)]),

    % Block operations
    declare_binding(ruby, array_map/3, '.map', [array, block], [array],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, array_select/3, '.select', [array, block], [array],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, array_reject/3, '.reject', [array, block], [array],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, array_find/3, '.find', [array, block], [any],
                   [pure, nondeterministic, partial, pattern(block_call)]),
    declare_binding(ruby, array_reduce/4, '.reduce', [array, any, block], [any],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, array_each/2, '.each', [array, block], [],
                   [effect(io), nondeterministic, pattern(block_call)]),

    % Set operations
    declare_binding(ruby, array_include/2, '.include?', [array, any], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_union/3, '|', [array, array], [array],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, array_intersection/3, '&', [array, array], [array],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, array_difference/3, '-', [array, array], [array],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, array_concat/3, '+', [array, array], [array],
                   [pure, deterministic, total, pattern(operator)]),

    % Joining
    declare_binding(ruby, array_join/2, '.join', [array], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, array_join/3, '.join', [array, string], [string],
                   [pure, deterministic, total, pattern(method_call)]).

%% ============================================================================
%% HASH OPERATION BINDINGS
%% ============================================================================

register_hash_bindings :-
    % Basic hash operations
    declare_binding(ruby, hash_length/2, '.length', [hash], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_empty/1, '.empty?', [hash], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_keys/2, '.keys', [hash], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_values/2, '.values', [hash], [array],
                   [pure, deterministic, total, pattern(method_call)]),

    % Access
    declare_binding(ruby, hash_get/3, '[]', [hash, any], [any],
                   [pure, deterministic, partial, pattern(bracket)]),
    declare_binding(ruby, hash_fetch/3, '.fetch', [hash, any], [any],
                   [pure, deterministic, partial, effect(throws), pattern(method_call)]),
    declare_binding(ruby, hash_fetch/4, '.fetch', [hash, any, any], [any],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_dig/3, '.dig', [hash, list], [any],
                   [pure, deterministic, partial, pattern(method_call)]),

    % Modification
    declare_binding(ruby, hash_set/3, '[]=', [hash, any, any], [],
                   [effect(state), deterministic, total, pattern(bracket_assign)]),
    declare_binding(ruby, hash_delete/3, '.delete', [hash, any], [any],
                   [effect(state), deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, hash_merge/3, '.merge', [hash, hash], [hash],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_merge_bang/2, '.merge!', [hash, hash], [],
                   [effect(state), deterministic, total, pattern(method_call)]),

    % Predicates
    declare_binding(ruby, hash_has_key/2, '.key?', [hash, any], [],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, hash_has_value/2, '.value?', [hash, any], [],
                   [pure, deterministic, total, pattern(method_call)]),

    % Block operations
    declare_binding(ruby, hash_each/2, '.each', [hash, block], [],
                   [effect(io), nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_each_key/2, '.each_key', [hash, block], [],
                   [effect(io), nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_each_value/2, '.each_value', [hash, block], [],
                   [effect(io), nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_select/3, '.select', [hash, block], [hash],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_reject/3, '.reject', [hash, block], [hash],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_transform_keys/3, '.transform_keys', [hash, block], [hash],
                   [pure, nondeterministic, pattern(block_call)]),
    declare_binding(ruby, hash_transform_values/3, '.transform_values', [hash, block], [hash],
                   [pure, nondeterministic, pattern(block_call)]).

%% ============================================================================
%% I/O OPERATION BINDINGS
%% ============================================================================

register_io_bindings :-
    % Console output
    declare_binding(ruby, puts/1, 'puts', [any], [],
                   [effect(io), deterministic, total, pattern(function_call)]),
    declare_binding(ruby, print/1, 'print', [any], [],
                   [effect(io), deterministic, total, pattern(function_call)]),
    declare_binding(ruby, p/1, 'p', [any], [],
                   [effect(io), deterministic, total, pattern(function_call)]),
    declare_binding(ruby, pp/1, 'pp', [any], [],
                   [effect(io), deterministic, total, pattern(function_call), import('pp')]),

    % Console input
    declare_binding(ruby, gets/1, 'gets', [], [string],
                   [effect(io), deterministic, partial, pattern(function_call)]),
    declare_binding(ruby, gets_chomp/1, 'gets.chomp', [], [string],
                   [effect(io), deterministic, partial, pattern(method_chain)]),

    % File operations
    declare_binding(ruby, file_read/2, 'File.read', [string], [string],
                   [effect(io), deterministic, partial, effect(throws), pattern(static_call)]),
    declare_binding(ruby, file_write/3, 'File.write', [string, string], [int],
                   [effect(io), deterministic, partial, effect(throws), pattern(static_call)]),
    declare_binding(ruby, file_readlines/2, 'File.readlines', [string], [array],
                   [effect(io), deterministic, partial, effect(throws), pattern(static_call)]),
    declare_binding(ruby, file_exist/1, 'File.exist?', [string], [],
                   [effect(io), deterministic, total, pattern(static_call)]),
    declare_binding(ruby, file_directory/1, 'File.directory?', [string], [],
                   [effect(io), deterministic, total, pattern(static_call)]),
    declare_binding(ruby, file_size/2, 'File.size', [string], [int],
                   [effect(io), deterministic, partial, effect(throws), pattern(static_call)]),

    % File path operations
    declare_binding(ruby, file_basename/2, 'File.basename', [string], [string],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, file_dirname/2, 'File.dirname', [string], [string],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, file_extname/2, 'File.extname', [string], [string],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, file_join/3, 'File.join', [string, string], [string],
                   [pure, deterministic, total, pattern(static_call)]).

%% ============================================================================
%% REGEX OPERATION BINDINGS
%% ============================================================================

register_regex_bindings :-
    % Matching
    declare_binding(ruby, regex_match/3, '.match', [string, regex], [match],
                   [pure, deterministic, partial, pattern(method_call)]),
    declare_binding(ruby, regex_match_all/3, '.scan', [string, regex], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, regex_test/2, '=~', [string, regex], [],
                   [pure, deterministic, total, pattern(operator)]),

    % Replacement
    declare_binding(ruby, regex_sub/4, '.sub', [string, regex, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, regex_gsub/4, '.gsub', [string, regex, string], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % Splitting
    declare_binding(ruby, regex_split/3, '.split', [string, regex], [array],
                   [pure, deterministic, total, pattern(method_call)]).

%% ============================================================================
%% MATH OPERATION BINDINGS
%% ============================================================================

register_math_bindings :-
    % Basic arithmetic
    declare_binding(ruby, plus/3, '+', [number, number], [number],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, minus/3, '-', [number, number], [number],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, mult/3, '*', [number, number], [number],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, div/3, '/', [number, number], [number],
                   [pure, deterministic, partial, pattern(operator)]),
    declare_binding(ruby, mod/3, '%', [int, int], [int],
                   [pure, deterministic, partial, pattern(operator)]),
    declare_binding(ruby, power/3, '**', [number, number], [number],
                   [pure, deterministic, total, pattern(operator)]),

    % Comparison
    declare_binding(ruby, eq/2, '==', [any, any], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, neq/2, '!=', [any, any], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, lt/2, '<', [number, number], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, gt/2, '>', [number, number], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, lte/2, '<=', [number, number], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, gte/2, '>=', [number, number], [],
                   [pure, deterministic, total, pattern(operator)]),
    declare_binding(ruby, spaceship/3, '<=>', [any, any], [int],
                   [pure, deterministic, total, pattern(operator)]),

    % Math module functions
    declare_binding(ruby, sqrt/2, 'Math.sqrt', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, cbrt/2, 'Math.cbrt', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, log/2, 'Math.log', [number], [float],
                   [pure, deterministic, partial, pattern(static_call)]),
    declare_binding(ruby, log10/2, 'Math.log10', [number], [float],
                   [pure, deterministic, partial, pattern(static_call)]),
    declare_binding(ruby, log2/2, 'Math.log2', [number], [float],
                   [pure, deterministic, partial, pattern(static_call)]),
    declare_binding(ruby, exp/2, 'Math.exp', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, sin/2, 'Math.sin', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, cos/2, 'Math.cos', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, tan/2, 'Math.tan', [number], [float],
                   [pure, deterministic, total, pattern(static_call)]),
    declare_binding(ruby, floor/2, '.floor', [float], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, ceil/2, '.ceil', [float], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, round/2, '.round', [float], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, abs/2, '.abs', [number], [number],
                   [pure, deterministic, total, pattern(method_call)]),

    % Random
    declare_binding(ruby, random/1, 'rand', [], [float],
                   [effect(random), deterministic, total, pattern(function_call)]),
    declare_binding(ruby, random/2, 'rand', [int], [int],
                   [effect(random), deterministic, total, pattern(function_call)]),
    declare_binding(ruby, random_range/3, 'rand', [range], [number],
                   [effect(random), deterministic, total, pattern(function_call)]).

%% ============================================================================
%% TYPE CONVERSION BINDINGS
%% ============================================================================

register_conversion_bindings :-
    % To string
    declare_binding(ruby, to_s/2, '.to_s', [any], [string],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, inspect/2, '.inspect', [any], [string],
                   [pure, deterministic, total, pattern(method_call)]),

    % To number
    declare_binding(ruby, to_i/2, '.to_i', [any], [int],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, to_f/2, '.to_f', [any], [float],
                   [pure, deterministic, total, pattern(method_call)]),

    % To array/hash
    declare_binding(ruby, to_a/2, '.to_a', [any], [array],
                   [pure, deterministic, total, pattern(method_call)]),
    declare_binding(ruby, to_h/2, '.to_h', [any], [hash],
                   [pure, deterministic, partial, pattern(method_call)]),

    % To symbol
    declare_binding(ruby, to_sym/2, '.to_sym', [string], [symbol],
                   [pure, deterministic, total, pattern(method_call)]),

    % JSON
    declare_binding(ruby, json_parse/2, 'JSON.parse', [string], [any],
                   [pure, deterministic, partial, effect(throws), import('json'), pattern(static_call)]),
    declare_binding(ruby, json_generate/2, 'JSON.generate', [any], [string],
                   [pure, deterministic, total, import('json'), pattern(static_call)]),
    declare_binding(ruby, json_pretty/2, 'JSON.pretty_generate', [any], [string],
                   [pure, deterministic, total, import('json'), pattern(static_call)]).

%% ============================================================================
%% TESTING
%% ============================================================================

test_ruby_bindings :-
    format('~n========================================~n', []),
    format('  Ruby Bindings Tests~n', []),
    format('========================================~n~n', []),

    % Test binding queries
    format('Testing binding queries...~n', []),

    (   rb_binding(length/2, Target, _, _, _)
    ->  format('  length/2 -> ~w [PASS]~n', [Target])
    ;   format('  length/2 [FAIL]~n', [])
    ),

    (   rb_binding(string_split/3, Target2, _, _, _)
    ->  format('  string_split/3 -> ~w [PASS]~n', [Target2])
    ;   format('  string_split/3 [FAIL]~n', [])
    ),

    (   rb_binding(json_parse/2, _, _, _, Options),
        member(import('json'), Options)
    ->  format('  json_parse/2 requires json import [PASS]~n', [])
    ;   format('  json_parse/2 import check [FAIL]~n', [])
    ),

    % Count bindings
    findall(P, rb_binding(P, _, _, _, _), Bindings),
    length(Bindings, Count),
    format('~nTotal Ruby bindings: ~d~n', [Count]),
    format('~n========================================~n', []).

%% ============================================================================
%% AUTO-INITIALIZATION
%% ============================================================================

:- initialization(init_ruby_bindings, now).

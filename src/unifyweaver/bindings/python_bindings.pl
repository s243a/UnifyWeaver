% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_bindings.pl - Python-specific bindings
%
% This module defines bindings for Python target language features.
% Mirrors the structure of powershell_bindings.pl for consistency.
%
% Categories:
%   - Core Built-ins (len, type, isinstance, etc.)
%   - String Operations (split, join, replace, etc.)
%   - List Operations (append, extend, sort, etc.)
%   - Dict Operations (get, keys, items, update, etc.)
%   - I/O Operations (open, read, write, json)
%   - Pattern Matching (re.search, re.match, etc.)
%   - Math Operations (sqrt, floor, ceil, etc.)
%   - Collections (deque, Counter, defaultdict, etc.)
%   - Itertools (map, filter, zip, etc.)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(python_bindings, [
    init_python_bindings/0,
    py_binding/5,               % Convenience: py_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_python_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_python_bindings
%
%  Initialize all Python bindings. Call this before using the compiler.
%
init_python_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_list_bindings,
    register_dict_bindings,
    register_io_bindings,
    register_pattern_bindings,
    register_math_bindings,
    register_collections_bindings,
    register_itertools_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% py_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query Python bindings with reduced arity (Target=python implied).
%
py_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(python, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Type Inspection
    % -------------------------------------------

    % len - get length of sequence
    declare_binding(python, length/2, 'len',
        [sequence], [int],
        [pure, deterministic, total]),

    % type - get type of object
    declare_binding(python, type_of/2, 'type',
        [any], [type],
        [pure, deterministic, total]),

    % isinstance - type checking
    declare_binding(python, is_instance/2, 'isinstance',
        [any, type], [bool],
        [pure, deterministic, total]),

    % id - get unique identifier
    declare_binding(python, object_id/2, 'id',
        [any], [int],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Type Conversions
    % -------------------------------------------

    % str - convert to string
    declare_binding(python, to_string/2, 'str',
        [any], [string],
        [pure, deterministic, total]),

    % int - convert to integer
    declare_binding(python, to_int/2, 'int',
        [any], [int],
        [pure, deterministic, partial, effect(throws)]),

    % float - convert to float
    declare_binding(python, to_float/2, 'float',
        [any], [float],
        [pure, deterministic, partial, effect(throws)]),

    % bool - convert to boolean
    declare_binding(python, to_bool/2, 'bool',
        [any], [bool],
        [pure, deterministic, total]),

    % list - convert to list
    declare_binding(python, to_list/2, 'list',
        [iterable], [list],
        [pure, deterministic, total]),

    % tuple - convert to tuple
    declare_binding(python, to_tuple/2, 'tuple',
        [iterable], [tuple],
        [pure, deterministic, total]),

    % set - convert to set
    declare_binding(python, to_set/2, 'set',
        [iterable], [set],
        [pure, deterministic, total]),

    % dict - convert to dict
    declare_binding(python, to_dict/2, 'dict',
        [iterable], [dict],
        [pure, deterministic, partial, effect(throws)]),

    % -------------------------------------------
    % Comparison and Equality
    % -------------------------------------------

    % abs - absolute value
    declare_binding(python, abs/2, 'abs',
        [number], [number],
        [pure, deterministic, total]),

    % min - minimum value
    declare_binding(python, min/2, 'min',
        [iterable], [any],
        [pure, deterministic, partial]),

    % max - maximum value
    declare_binding(python, max/2, 'max',
        [iterable], [any],
        [pure, deterministic, partial]),

    % sum - sum of sequence
    declare_binding(python, sum/2, 'sum',
        [iterable], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Sequence Operations
    % -------------------------------------------

    % range - generate range
    declare_binding(python, range/2, 'range',
        [int], [range],
        [pure, deterministic, total]),

    declare_binding(python, range/3, 'range',
        [int, int], [range],
        [pure, deterministic, total]),

    declare_binding(python, range/4, 'range',
        [int, int, int], [range],
        [pure, deterministic, total]),

    % enumerate - add index to iterable
    declare_binding(python, enumerate/2, 'enumerate',
        [iterable], [enumerate],
        [pure, deterministic, total]),

    % zip - pair iterables
    declare_binding(python, zip/3, 'zip',
        [iterable, iterable], [zip],
        [pure, deterministic, total]),

    % reversed - reverse sequence
    declare_binding(python, reversed/2, 'reversed',
        [sequence], [reversed],
        [pure, deterministic, total]),

    % sorted - sort sequence
    declare_binding(python, sorted/2, 'sorted',
        [iterable], [list],
        [pure, deterministic, total]),

    % -------------------------------------------
    % I/O Built-ins
    % -------------------------------------------

    % print - output to stdout
    declare_binding(python, print/1, 'print',
        [any], [],
        [effect(io), deterministic]),

    % input - read from stdin
    declare_binding(python, input/1, 'input',
        [], [string],
        [effect(io), deterministic]),

    declare_binding(python, input/2, 'input',
        [string], [string],
        [effect(io), deterministic]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Methods
    % -------------------------------------------

    % str.split - split string
    declare_binding(python, string_split/2, '.split()',
        [string], [list],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(python, string_split/3, '.split',
        [string, string], [list],
        [pure, deterministic, total, pattern(method_call)]),

    % str.join - join strings
    declare_binding(python, string_join/3, '.join',
        [string, list], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % str.replace - replace substring
    declare_binding(python, string_replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % str.strip - strip whitespace
    declare_binding(python, string_strip/2, '.strip()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % str.lower - convert to lowercase
    declare_binding(python, string_lower/2, '.lower()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % str.upper - convert to uppercase
    declare_binding(python, string_upper/2, '.upper()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % str.startswith - check prefix
    declare_binding(python, string_starts_with/2, '.startswith',
        [string, string], [bool],
        [pure, deterministic, total, pattern(method_call)]),

    % str.endswith - check suffix
    declare_binding(python, string_ends_with/2, '.endswith',
        [string, string], [bool],
        [pure, deterministic, total, pattern(method_call)]),

    % str.find - find substring
    declare_binding(python, string_find/3, '.find',
        [string, string], [int],
        [pure, deterministic, total, pattern(method_call)]),

    % str.format - format string
    declare_binding(python, string_format/3, '.format',
        [string, list], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % f-string (special handling)
    declare_binding(python, fstring/2, 'f"..."',
        [string], [string],
        [pure, deterministic, total, pattern(fstring)]).

% ============================================================================
% LIST OPERATION BINDINGS
% ============================================================================

register_list_bindings :-
    % -------------------------------------------
    % List Methods
    % -------------------------------------------

    % list.append - add element (mutates)
    declare_binding(python, list_append/2, '.append',
        [list, any], [],
        [effect(state), deterministic, pattern(method_call)]),

    % list.extend - extend with iterable (mutates)
    declare_binding(python, list_extend/2, '.extend',
        [list, iterable], [],
        [effect(state), deterministic, pattern(method_call)]),

    % list.insert - insert at index (mutates)
    declare_binding(python, list_insert/3, '.insert',
        [list, int, any], [],
        [effect(state), deterministic, pattern(method_call)]),

    % list.remove - remove first occurrence (mutates)
    declare_binding(python, list_remove/2, '.remove',
        [list, any], [],
        [effect(state), deterministic, partial, effect(throws), pattern(method_call)]),

    % list.pop - remove and return element (mutates)
    declare_binding(python, list_pop/2, '.pop()',
        [list], [any],
        [effect(state), deterministic, partial, effect(throws), pattern(method_call)]),

    declare_binding(python, list_pop/3, '.pop',
        [list, int], [any],
        [effect(state), deterministic, partial, effect(throws), pattern(method_call)]),

    % list.index - find index of element
    declare_binding(python, list_index/3, '.index',
        [list, any], [int],
        [pure, deterministic, partial, effect(throws), pattern(method_call)]),

    % list.count - count occurrences
    declare_binding(python, list_count/3, '.count',
        [list, any], [int],
        [pure, deterministic, total, pattern(method_call)]),

    % list.sort - sort in place (mutates)
    declare_binding(python, list_sort/1, '.sort()',
        [list], [],
        [effect(state), deterministic, pattern(method_call)]),

    % list.reverse - reverse in place (mutates)
    declare_binding(python, list_reverse/1, '.reverse()',
        [list], [],
        [effect(state), deterministic, pattern(method_call)]),

    % list.copy - shallow copy
    declare_binding(python, list_copy/2, '.copy()',
        [list], [list],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% DICT OPERATION BINDINGS
% ============================================================================

register_dict_bindings :-
    % -------------------------------------------
    % Dict Methods
    % -------------------------------------------

    % dict.get - get value with default
    declare_binding(python, dict_get/3, '.get',
        [dict, any], [any],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(python, dict_get/4, '.get',
        [dict, any, any], [any],
        [pure, deterministic, total, pattern(method_call)]),

    % dict.keys - get keys
    declare_binding(python, dict_keys/2, '.keys()',
        [dict], [dict_keys],
        [pure, deterministic, total, pattern(method_call)]),

    % dict.values - get values
    declare_binding(python, dict_values/2, '.values()',
        [dict], [dict_values],
        [pure, deterministic, total, pattern(method_call)]),

    % dict.items - get key-value pairs
    declare_binding(python, dict_items/2, '.items()',
        [dict], [dict_items],
        [pure, deterministic, total, pattern(method_call)]),

    % dict.update - merge dicts (mutates)
    declare_binding(python, dict_update/2, '.update',
        [dict, dict], [],
        [effect(state), deterministic, pattern(method_call)]),

    % dict.pop - remove and return value (mutates)
    declare_binding(python, dict_pop/3, '.pop',
        [dict, any], [any],
        [effect(state), deterministic, partial, effect(throws), pattern(method_call)]),

    % dict.setdefault - get or set default
    declare_binding(python, dict_setdefault/4, '.setdefault',
        [dict, any, any], [any],
        [effect(state), deterministic, pattern(method_call)]),

    % dict.copy - shallow copy
    declare_binding(python, dict_copy/2, '.copy()',
        [dict], [dict],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% I/O OPERATION BINDINGS
% ============================================================================

register_io_bindings :-
    % -------------------------------------------
    % File Operations
    % -------------------------------------------

    % open - open file
    declare_binding(python, file_open/3, 'open',
        [string, string], [file],
        [effect(io), deterministic, partial, effect(throws), import('builtins')]),

    % with open(...) as f: - context manager
    declare_binding(python, with_open/3, 'with open(...) as',
        [string, string], [file],
        [effect(io), deterministic, partial, effect(throws), pattern(context_manager)]),

    % file.read - read all content
    declare_binding(python, file_read/2, '.read()',
        [file], [string],
        [effect(io), deterministic, pattern(method_call)]),

    % file.readlines - read lines
    declare_binding(python, file_readlines/2, '.readlines()',
        [file], [list],
        [effect(io), deterministic, pattern(method_call)]),

    % file.write - write content
    declare_binding(python, file_write/2, '.write',
        [file, string], [int],
        [effect(io), deterministic, pattern(method_call)]),

    % file.close - close file
    declare_binding(python, file_close/1, '.close()',
        [file], [],
        [effect(io), deterministic, pattern(method_call)]),

    % -------------------------------------------
    % JSON Operations
    % -------------------------------------------

    % json.load - load JSON from file
    declare_binding(python, json_load/2, 'json.load',
        [file], [any],
        [effect(io), deterministic, partial, effect(throws), import('json')]),

    % json.loads - parse JSON string
    declare_binding(python, json_loads/2, 'json.loads',
        [string], [any],
        [pure, deterministic, partial, effect(throws), import('json')]),

    % json.dump - write JSON to file
    declare_binding(python, json_dump/2, 'json.dump',
        [any, file], [],
        [effect(io), deterministic, import('json')]),

    % json.dumps - serialize to JSON string
    declare_binding(python, json_dumps/2, 'json.dumps',
        [any], [string],
        [pure, deterministic, total, import('json')]),

    % -------------------------------------------
    % Path Operations
    % -------------------------------------------

    % os.path.exists - check path exists
    declare_binding(python, path_exists/1, 'os.path.exists',
        [string], [bool],
        [effect(io), deterministic, import('os')]),

    % os.path.join - join paths
    declare_binding(python, path_join/3, 'os.path.join',
        [string, string], [string],
        [pure, deterministic, total, import('os')]),

    % os.path.dirname - get directory name
    declare_binding(python, path_dirname/2, 'os.path.dirname',
        [string], [string],
        [pure, deterministic, total, import('os')]),

    % os.path.basename - get base name
    declare_binding(python, path_basename/2, 'os.path.basename',
        [string], [string],
        [pure, deterministic, total, import('os')]).

% ============================================================================
% PATTERN MATCHING BINDINGS (re module)
% ============================================================================

register_pattern_bindings :-
    % -------------------------------------------
    % Regular Expression Operations
    % -------------------------------------------

    % re.search - search for pattern
    declare_binding(python, regex_search/3, 're.search',
        [string, string], [match],
        [pure, deterministic, total, import('re')]),

    % re.match - match at beginning
    declare_binding(python, regex_match/3, 're.match',
        [string, string], [match],
        [pure, deterministic, total, import('re')]),

    % re.fullmatch - full string match
    declare_binding(python, regex_fullmatch/3, 're.fullmatch',
        [string, string], [match],
        [pure, deterministic, total, import('re')]),

    % re.findall - find all matches
    declare_binding(python, regex_findall/3, 're.findall',
        [string, string], [list],
        [pure, deterministic, total, import('re')]),

    % re.finditer - find all matches as iterator
    declare_binding(python, regex_finditer/3, 're.finditer',
        [string, string], [iterator],
        [pure, deterministic, total, import('re')]),

    % re.sub - substitute pattern
    declare_binding(python, regex_sub/4, 're.sub',
        [string, string, string], [string],
        [pure, deterministic, total, import('re')]),

    % re.split - split by pattern
    declare_binding(python, regex_split/3, 're.split',
        [string, string], [list],
        [pure, deterministic, total, import('re')]),

    % re.compile - compile pattern
    declare_binding(python, regex_compile/2, 're.compile',
        [string], [pattern],
        [pure, deterministic, total, import('re')]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % Math Module Functions
    % -------------------------------------------

    % math.sqrt - square root
    declare_binding(python, sqrt/2, 'math.sqrt',
        [number], [float],
        [pure, deterministic, partial, effect(throws), import('math')]),

    % math.floor - floor
    declare_binding(python, floor/2, 'math.floor',
        [number], [int],
        [pure, deterministic, total, import('math')]),

    % math.ceil - ceiling
    declare_binding(python, ceil/2, 'math.ceil',
        [number], [int],
        [pure, deterministic, total, import('math')]),

    % math.pow - power
    declare_binding(python, pow/3, 'math.pow',
        [number, number], [float],
        [pure, deterministic, total, import('math')]),

    % math.log - natural logarithm
    declare_binding(python, log/2, 'math.log',
        [number], [float],
        [pure, deterministic, partial, effect(throws), import('math')]),

    % math.log10 - base 10 logarithm
    declare_binding(python, log10/2, 'math.log10',
        [number], [float],
        [pure, deterministic, partial, effect(throws), import('math')]),

    % math.sin - sine
    declare_binding(python, sin/2, 'math.sin',
        [number], [float],
        [pure, deterministic, total, import('math')]),

    % math.cos - cosine
    declare_binding(python, cos/2, 'math.cos',
        [number], [float],
        [pure, deterministic, total, import('math')]),

    % math.tan - tangent
    declare_binding(python, tan/2, 'math.tan',
        [number], [float],
        [pure, deterministic, total, import('math')]),

    % math.pi - constant
    declare_binding(python, pi/1, 'math.pi',
        [], [float],
        [pure, deterministic, total, import('math')]),

    % math.e - constant
    declare_binding(python, e/1, 'math.e',
        [], [float],
        [pure, deterministic, total, import('math')]).

% ============================================================================
% COLLECTIONS MODULE BINDINGS
% ============================================================================

register_collections_bindings :-
    % -------------------------------------------
    % Collections Module
    % -------------------------------------------

    % collections.deque - double-ended queue
    declare_binding(python, deque/2, 'collections.deque',
        [iterable], [deque],
        [pure, deterministic, total, import('collections')]),

    % collections.Counter - counter
    declare_binding(python, counter/2, 'collections.Counter',
        [iterable], [counter],
        [pure, deterministic, total, import('collections')]),

    % collections.defaultdict - default dict
    declare_binding(python, defaultdict/2, 'collections.defaultdict',
        [callable], [defaultdict],
        [pure, deterministic, total, import('collections')]),

    % collections.namedtuple - named tuple factory
    declare_binding(python, namedtuple/3, 'collections.namedtuple',
        [string, list], [type],
        [pure, deterministic, total, import('collections')]),

    % collections.OrderedDict - ordered dict
    declare_binding(python, ordereddict/2, 'collections.OrderedDict',
        [iterable], [ordereddict],
        [pure, deterministic, total, import('collections')]).

% ============================================================================
% ITERTOOLS MODULE BINDINGS
% ============================================================================

register_itertools_bindings :-
    % -------------------------------------------
    % Itertools Functions
    % -------------------------------------------

    % itertools.chain - chain iterables
    declare_binding(python, chain/2, 'itertools.chain',
        [list(iterable)], [iterator],
        [pure, deterministic, total, import('itertools'), variadic]),

    % itertools.combinations - combinations
    declare_binding(python, combinations/3, 'itertools.combinations',
        [iterable, int], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.permutations - permutations
    declare_binding(python, permutations/3, 'itertools.permutations',
        [iterable, int], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.product - cartesian product
    declare_binding(python, product/2, 'itertools.product',
        [list(iterable)], [iterator],
        [pure, deterministic, total, import('itertools'), variadic]),

    % itertools.groupby - group consecutive elements
    declare_binding(python, groupby/3, 'itertools.groupby',
        [iterable, callable], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.islice - slice iterator
    declare_binding(python, islice/4, 'itertools.islice',
        [iterable, int, int], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.takewhile - take while condition
    declare_binding(python, takewhile/3, 'itertools.takewhile',
        [callable, iterable], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.dropwhile - drop while condition
    declare_binding(python, dropwhile/3, 'itertools.dropwhile',
        [callable, iterable], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.cycle - cycle through iterable
    declare_binding(python, cycle/2, 'itertools.cycle',
        [iterable], [iterator],
        [pure, deterministic, total, import('itertools')]),

    % itertools.repeat - repeat value
    declare_binding(python, repeat/3, 'itertools.repeat',
        [any, int], [iterator],
        [pure, deterministic, total, import('itertools')]).

% ============================================================================
% TESTS
% ============================================================================

test_python_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Python Bindings Tests                ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing Python bindings~n', []),
    init_python_bindings,
    format('[✓] Python bindings initialized~n~n', []),

    % Test builtin bindings exist
    format('[Test 2] Checking built-in bindings~n', []),
    (   py_binding(length/2, 'len', _, _, _)
    ->  format('[✓] length/2 -> len binding exists~n', [])
    ;   format('[✗] length/2 binding missing~n', []), fail
    ),
    (   py_binding(to_string/2, 'str', _, _, _)
    ->  format('[✓] to_string/2 -> str binding exists~n', [])
    ;   format('[✗] to_string/2 binding missing~n', []), fail
    ),

    % Test string bindings exist
    format('~n[Test 3] Checking string bindings~n', []),
    (   py_binding(string_split/2, _, _, _, _)
    ->  format('[✓] string_split/2 binding exists~n', [])
    ;   format('[✗] string_split/2 binding missing~n', []), fail
    ),

    % Test math bindings with import
    format('~n[Test 4] Checking math bindings with imports~n', []),
    (   py_binding(sqrt/2, 'math.sqrt', _, _, Options),
        member(import('math'), Options)
    ->  format('[✓] sqrt/2 has import(math)~n', [])
    ;   format('[✗] sqrt/2 missing import~n', []), fail
    ),

    % Test JSON bindings
    format('~n[Test 5] Checking JSON bindings~n', []),
    (   py_binding(json_loads/2, 'json.loads', _, _, Options2),
        member(import('json'), Options2)
    ->  format('[✓] json_loads/2 has import(json)~n', [])
    ;   format('[✗] json_loads/2 missing~n', []), fail
    ),

    % Test regex bindings
    format('~n[Test 6] Checking regex bindings~n', []),
    (   py_binding(regex_search/3, 're.search', _, _, Options3),
        member(import('re'), Options3)
    ->  format('[✓] regex_search/3 has import(re)~n', [])
    ;   format('[✗] regex_search/3 missing~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 7] Counting total bindings~n', []),
    findall(P, py_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total Python bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All Python Bindings Tests Passed     ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).

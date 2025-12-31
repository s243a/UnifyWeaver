% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% go_bindings.pl - Go-specific bindings
%
% This module defines bindings for Go target language features.
% Maps Prolog predicates to Go stdlib functions and packages.
%
% Categories:
%   - Core Built-ins (len, make, new, etc.)
%   - String Operations (strings package)
%   - Math Operations (math package)
%   - I/O Operations (fmt, os, io)
%   - JSON Operations (encoding/json)
%   - Regex Operations (regexp)
%   - Time Operations (time)
%   - Path Operations (path/filepath)
%   - Collections (slices, maps, sort)
%   - Conversion Operations (strconv)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(go_bindings, [
    init_go_bindings/0,
    go_binding/5,               % Convenience: go_binding(Pred, TargetName, Inputs, Outputs, Options)
    go_binding_import/2,        % go_binding_import(Pred, Import) - get required import
    test_go_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_go_bindings
%
%  Initialize all Go bindings. Call this before using the compiler.
%
init_go_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_io_bindings,
    register_json_bindings,
    register_regex_bindings,
    register_time_bindings,
    register_path_bindings,
    register_collection_bindings,
    register_conversion_bindings.

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

%% go_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query Go bindings with reduced arity (Target=go implied).
%
go_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(go, Pred, TargetName, Inputs, Outputs, Options).

%% go_binding_import(?Pred, ?Import)
%
%  Get the import required for a Go binding.
%
go_binding_import(Pred, Import) :-
    go_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- go_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(go, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Length and Capacity
    % -------------------------------------------

    % len - get length of array, slice, string, map, channel
    declare_binding(go, length/2, 'len',
        [any], [int],
        [pure, deterministic, total]),

    % cap - get capacity of slice, array, channel
    declare_binding(go, capacity/2, 'cap',
        [any], [int],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Memory Allocation
    % -------------------------------------------

    % make - create slice, map, or channel
    declare_binding(go, make_slice/3, 'make',
        [type, int], [slice],
        [pure, deterministic, total]),

    declare_binding(go, make_map/2, 'make',
        [type], [map],
        [pure, deterministic, total]),

    declare_binding(go, make_chan/2, 'make',
        [type], [chan],
        [pure, deterministic, total]),

    % new - allocate memory and return pointer
    declare_binding(go, new/2, 'new',
        [type], [pointer],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Slice Operations
    % -------------------------------------------

    % append - append to slice
    declare_binding(go, append/3, 'append',
        [slice, any], [slice],
        [pure, deterministic, total]),

    % copy - copy slice elements
    declare_binding(go, copy/3, 'copy',
        [slice, slice], [int],
        [effect(state), deterministic, total]),

    % -------------------------------------------
    % Type Assertions and Conversions
    % -------------------------------------------

    % Type assertion: x.(Type)
    declare_binding(go, type_assert/3, '.()',
        [any, type], [any],
        [pure, deterministic, partial, effect(panic), pattern(type_assert)]),

    % Type conversion: Type(x)
    declare_binding(go, convert/3, 'Type()',
        [any, type], [any],
        [pure, deterministic, total, pattern(type_convert)]),

    % -------------------------------------------
    % Panic and Recover
    % -------------------------------------------

    % panic - trigger panic
    declare_binding(go, panic/1, 'panic',
        [any], [],
        [effect(panic), deterministic]),

    % recover - recover from panic
    declare_binding(go, recover/1, 'recover',
        [], [any],
        [effect(recover), deterministic]).

% ============================================================================
% STRING OPERATION BINDINGS (strings package)
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Searching
    % -------------------------------------------

    % strings.Contains - check if substring exists
    declare_binding(go, string_contains/3, 'strings.Contains',
        [string, string], [bool],
        [pure, deterministic, total, import('strings')]),

    % strings.HasPrefix - check prefix
    declare_binding(go, string_has_prefix/3, 'strings.HasPrefix',
        [string, string], [bool],
        [pure, deterministic, total, import('strings')]),

    % strings.HasSuffix - check suffix
    declare_binding(go, string_has_suffix/3, 'strings.HasSuffix',
        [string, string], [bool],
        [pure, deterministic, total, import('strings')]),

    % strings.Index - find substring index
    declare_binding(go, string_index/3, 'strings.Index',
        [string, string], [int],
        [pure, deterministic, total, import('strings')]),

    % strings.LastIndex - find last substring index
    declare_binding(go, string_last_index/3, 'strings.LastIndex',
        [string, string], [int],
        [pure, deterministic, total, import('strings')]),

    % strings.Count - count non-overlapping occurrences
    declare_binding(go, string_count/3, 'strings.Count',
        [string, string], [int],
        [pure, deterministic, total, import('strings')]),

    % -------------------------------------------
    % String Transformation
    % -------------------------------------------

    % strings.ToLower - convert to lowercase
    declare_binding(go, string_lower/2, 'strings.ToLower',
        [string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.ToUpper - convert to uppercase
    declare_binding(go, string_upper/2, 'strings.ToUpper',
        [string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.ToTitle - convert to title case
    declare_binding(go, string_title/2, 'strings.ToTitle',
        [string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.TrimSpace - trim whitespace
    declare_binding(go, string_trim_space/2, 'strings.TrimSpace',
        [string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.Trim - trim characters
    declare_binding(go, string_trim/3, 'strings.Trim',
        [string, string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.TrimPrefix - remove prefix
    declare_binding(go, string_trim_prefix/3, 'strings.TrimPrefix',
        [string, string], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.TrimSuffix - remove suffix
    declare_binding(go, string_trim_suffix/3, 'strings.TrimSuffix',
        [string, string], [string],
        [pure, deterministic, total, import('strings')]),

    % -------------------------------------------
    % String Splitting and Joining
    % -------------------------------------------

    % strings.Split - split string
    declare_binding(go, string_split/3, 'strings.Split',
        [string, string], [slice_string],
        [pure, deterministic, total, import('strings')]),

    % strings.SplitN - split with limit
    declare_binding(go, string_split_n/4, 'strings.SplitN',
        [string, string, int], [slice_string],
        [pure, deterministic, total, import('strings')]),

    % strings.Fields - split by whitespace
    declare_binding(go, string_fields/2, 'strings.Fields',
        [string], [slice_string],
        [pure, deterministic, total, import('strings')]),

    % strings.Join - join strings with separator
    declare_binding(go, string_join/3, 'strings.Join',
        [slice_string, string], [string],
        [pure, deterministic, total, import('strings')]),

    % -------------------------------------------
    % String Replacement
    % -------------------------------------------

    % strings.Replace - replace substring
    declare_binding(go, string_replace/5, 'strings.Replace',
        [string, string, string, int], [string],
        [pure, deterministic, total, import('strings')]),

    % strings.ReplaceAll - replace all occurrences
    declare_binding(go, string_replace_all/4, 'strings.ReplaceAll',
        [string, string, string], [string],
        [pure, deterministic, total, import('strings')]),

    % -------------------------------------------
    % String Comparison
    % -------------------------------------------

    % strings.EqualFold - case-insensitive equality
    declare_binding(go, string_equal_fold/3, 'strings.EqualFold',
        [string, string], [bool],
        [pure, deterministic, total, import('strings')]),

    % strings.Compare - compare strings
    declare_binding(go, string_compare/3, 'strings.Compare',
        [string, string], [int],
        [pure, deterministic, total, import('strings')]),

    % -------------------------------------------
    % String Building
    % -------------------------------------------

    % strings.Repeat - repeat string
    declare_binding(go, string_repeat/3, 'strings.Repeat',
        [string, int], [string],
        [pure, deterministic, total, import('strings')]).

% ============================================================================
% MATH OPERATION BINDINGS (math package)
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % Basic Math Functions
    % -------------------------------------------

    % math.Abs - absolute value
    declare_binding(go, abs/2, 'math.Abs',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Max - maximum of two floats
    declare_binding(go, max/3, 'math.Max',
        [float64, float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Min - minimum of two floats
    declare_binding(go, min/3, 'math.Min',
        [float64, float64], [float64],
        [pure, deterministic, total, import('math')]),

    % -------------------------------------------
    % Power and Roots
    % -------------------------------------------

    % math.Sqrt - square root
    declare_binding(go, sqrt/2, 'math.Sqrt',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % math.Pow - power
    declare_binding(go, pow/3, 'math.Pow',
        [float64, float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Cbrt - cube root
    declare_binding(go, cbrt/2, 'math.Cbrt',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Exp - e^x
    declare_binding(go, exp/2, 'math.Exp',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % -------------------------------------------
    % Logarithms
    % -------------------------------------------

    % math.Log - natural logarithm
    declare_binding(go, log/2, 'math.Log',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % math.Log10 - base 10 logarithm
    declare_binding(go, log10/2, 'math.Log10',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % math.Log2 - base 2 logarithm
    declare_binding(go, log2/2, 'math.Log2',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % -------------------------------------------
    % Trigonometric Functions
    % -------------------------------------------

    % math.Sin - sine
    declare_binding(go, sin/2, 'math.Sin',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Cos - cosine
    declare_binding(go, cos/2, 'math.Cos',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Tan - tangent
    declare_binding(go, tan/2, 'math.Tan',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Asin - arc sine
    declare_binding(go, asin/2, 'math.Asin',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % math.Acos - arc cosine
    declare_binding(go, acos/2, 'math.Acos',
        [float64], [float64],
        [pure, deterministic, partial, import('math')]),

    % math.Atan - arc tangent
    declare_binding(go, atan/2, 'math.Atan',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Atan2 - arc tangent of y/x
    declare_binding(go, atan2/3, 'math.Atan2',
        [float64, float64], [float64],
        [pure, deterministic, total, import('math')]),

    % -------------------------------------------
    % Rounding Functions
    % -------------------------------------------

    % math.Floor - floor
    declare_binding(go, floor/2, 'math.Floor',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Ceil - ceiling
    declare_binding(go, ceil/2, 'math.Ceil',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Round - round to nearest
    declare_binding(go, round/2, 'math.Round',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Trunc - truncate
    declare_binding(go, trunc/2, 'math.Trunc',
        [float64], [float64],
        [pure, deterministic, total, import('math')]),

    % -------------------------------------------
    % Constants
    % -------------------------------------------

    % math.Pi - π
    declare_binding(go, pi/1, 'math.Pi',
        [], [float64],
        [pure, deterministic, total, import('math')]),

    % math.E - e
    declare_binding(go, e/1, 'math.E',
        [], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Phi - φ (golden ratio)
    declare_binding(go, phi/1, 'math.Phi',
        [], [float64],
        [pure, deterministic, total, import('math')]),

    % -------------------------------------------
    % Special Values
    % -------------------------------------------

    % math.IsNaN - check NaN
    declare_binding(go, is_nan/2, 'math.IsNaN',
        [float64], [bool],
        [pure, deterministic, total, import('math')]),

    % math.IsInf - check infinity
    declare_binding(go, is_inf/3, 'math.IsInf',
        [float64, int], [bool],
        [pure, deterministic, total, import('math')]),

    % math.NaN - return NaN
    declare_binding(go, nan/1, 'math.NaN',
        [], [float64],
        [pure, deterministic, total, import('math')]),

    % math.Inf - return infinity
    declare_binding(go, inf/2, 'math.Inf',
        [int], [float64],
        [pure, deterministic, total, import('math')]).

% ============================================================================
% I/O OPERATION BINDINGS (fmt, os, io packages)
% ============================================================================

register_io_bindings :-
    % -------------------------------------------
    % fmt package - Formatting and Printing
    % -------------------------------------------

    % fmt.Println - print with newline
    declare_binding(go, println/1, 'fmt.Println',
        [any], [],
        [effect(io), deterministic, import('fmt')]),

    % fmt.Printf - formatted print
    declare_binding(go, printf/2, 'fmt.Printf',
        [string, list], [],
        [effect(io), deterministic, import('fmt'), variadic]),

    % fmt.Sprintf - format to string
    declare_binding(go, sprintf/3, 'fmt.Sprintf',
        [string, list], [string],
        [pure, deterministic, total, import('fmt'), variadic]),

    % fmt.Errorf - format error
    declare_binding(go, errorf/3, 'fmt.Errorf',
        [string, list], [error],
        [pure, deterministic, total, import('fmt'), variadic]),

    % fmt.Scan - scan input
    declare_binding(go, scan/2, 'fmt.Scan',
        [pointer], [int, error],
        [effect(io), deterministic, import('fmt')]),

    % fmt.Sscanf - scan from string
    declare_binding(go, sscanf/4, 'fmt.Sscanf',
        [string, string, list], [int, error],
        [pure, deterministic, total, import('fmt'), variadic]),

    % -------------------------------------------
    % os package - File Operations
    % -------------------------------------------

    % os.Open - open file for reading
    declare_binding(go, file_open/2, 'os.Open',
        [string], [file, error],
        [effect(io), deterministic, import('os')]),

    % os.Create - create file
    declare_binding(go, file_create/2, 'os.Create',
        [string], [file, error],
        [effect(io), deterministic, import('os')]),

    % os.OpenFile - open with options
    declare_binding(go, file_open_flags/4, 'os.OpenFile',
        [string, int, int], [file, error],
        [effect(io), deterministic, import('os')]),

    % os.Remove - remove file
    declare_binding(go, file_remove/2, 'os.Remove',
        [string], [error],
        [effect(io), deterministic, import('os')]),

    % os.RemoveAll - remove recursively
    declare_binding(go, file_remove_all/2, 'os.RemoveAll',
        [string], [error],
        [effect(io), deterministic, import('os')]),

    % os.Rename - rename file
    declare_binding(go, file_rename/3, 'os.Rename',
        [string, string], [error],
        [effect(io), deterministic, import('os')]),

    % os.Mkdir - create directory
    declare_binding(go, mkdir/3, 'os.Mkdir',
        [string, int], [error],
        [effect(io), deterministic, import('os')]),

    % os.MkdirAll - create directory tree
    declare_binding(go, mkdir_all/3, 'os.MkdirAll',
        [string, int], [error],
        [effect(io), deterministic, import('os')]),

    % os.ReadFile - read entire file
    declare_binding(go, read_file/2, 'os.ReadFile',
        [string], [bytes, error],
        [effect(io), deterministic, import('os')]),

    % os.WriteFile - write entire file
    declare_binding(go, write_file/4, 'os.WriteFile',
        [string, bytes, int], [error],
        [effect(io), deterministic, import('os')]),

    % os.Stat - get file info
    declare_binding(go, file_stat/2, 'os.Stat',
        [string], [file_info, error],
        [effect(io), deterministic, import('os')]),

    % -------------------------------------------
    % os package - Environment
    % -------------------------------------------

    % os.Getenv - get environment variable
    declare_binding(go, getenv/2, 'os.Getenv',
        [string], [string],
        [effect(env), deterministic, total, import('os')]),

    % os.Setenv - set environment variable
    declare_binding(go, setenv/3, 'os.Setenv',
        [string, string], [error],
        [effect(env), deterministic, import('os')]),

    % os.Args - command line arguments
    declare_binding(go, args/1, 'os.Args',
        [], [slice_string],
        [effect(env), deterministic, total, import('os')]),

    % os.Exit - exit with code
    declare_binding(go, exit/1, 'os.Exit',
        [int], [],
        [effect(terminate), deterministic, import('os')]),

    % -------------------------------------------
    % io package - Reader/Writer interfaces
    % -------------------------------------------

    % io.ReadAll - read all from reader
    declare_binding(go, io_read_all/2, 'io.ReadAll',
        [reader], [bytes, error],
        [effect(io), deterministic, import('io')]),

    % io.Copy - copy from reader to writer
    declare_binding(go, io_copy/3, 'io.Copy',
        [writer, reader], [int64, error],
        [effect(io), deterministic, import('io')]),

    % io.WriteString - write string to writer
    declare_binding(go, io_write_string/3, 'io.WriteString',
        [writer, string], [int, error],
        [effect(io), deterministic, import('io')]).

% ============================================================================
% JSON OPERATION BINDINGS (encoding/json package)
% ============================================================================

register_json_bindings :-
    % json.Marshal - encode to JSON
    declare_binding(go, json_marshal/2, 'json.Marshal',
        [any], [bytes, error],
        [pure, deterministic, total, import('encoding/json')]),

    % json.MarshalIndent - encode to indented JSON
    declare_binding(go, json_marshal_indent/4, 'json.MarshalIndent',
        [any, string, string], [bytes, error],
        [pure, deterministic, total, import('encoding/json')]),

    % json.Unmarshal - decode from JSON
    declare_binding(go, json_unmarshal/3, 'json.Unmarshal',
        [bytes, pointer], [error],
        [pure, deterministic, partial, import('encoding/json')]),

    % json.NewEncoder - create encoder
    declare_binding(go, json_new_encoder/2, 'json.NewEncoder',
        [writer], [encoder],
        [pure, deterministic, total, import('encoding/json')]),

    % json.NewDecoder - create decoder
    declare_binding(go, json_new_decoder/2, 'json.NewDecoder',
        [reader], [decoder],
        [pure, deterministic, total, import('encoding/json')]),

    % encoder.Encode - encode to writer
    declare_binding(go, json_encode/2, '.Encode',
        [encoder, any], [error],
        [effect(io), deterministic, pattern(method_call), import('encoding/json')]),

    % decoder.Decode - decode from reader
    declare_binding(go, json_decode/2, '.Decode',
        [decoder, pointer], [error],
        [effect(io), deterministic, pattern(method_call), import('encoding/json')]).

% ============================================================================
% REGEX OPERATION BINDINGS (regexp package)
% ============================================================================

register_regex_bindings :-
    % regexp.Compile - compile pattern
    declare_binding(go, regex_compile/2, 'regexp.Compile',
        [string], [regexp, error],
        [pure, deterministic, partial, import('regexp')]),

    % regexp.MustCompile - compile pattern (panics on error)
    declare_binding(go, regex_must_compile/2, 'regexp.MustCompile',
        [string], [regexp],
        [pure, deterministic, partial, effect(panic), import('regexp')]),

    % regexp.MatchString - check if string matches pattern
    declare_binding(go, regex_match_string/3, 'regexp.MatchString',
        [string, string], [bool, error],
        [pure, deterministic, total, import('regexp')]),

    % (*Regexp).MatchString - check if string matches compiled pattern
    declare_binding(go, regex_matches/3, '.MatchString',
        [regexp, string], [bool],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).FindString - find first match
    declare_binding(go, regex_find_string/3, '.FindString',
        [regexp, string], [string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).FindAllString - find all matches
    declare_binding(go, regex_find_all_string/4, '.FindAllString',
        [regexp, string, int], [slice_string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).FindStringSubmatch - find match with groups
    declare_binding(go, regex_find_submatch/3, '.FindStringSubmatch',
        [regexp, string], [slice_string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).FindAllStringSubmatch - find all matches with groups
    declare_binding(go, regex_find_all_submatch/4, '.FindAllStringSubmatch',
        [regexp, string, int], [slice_slice_string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).ReplaceAllString - replace all matches
    declare_binding(go, regex_replace_all/4, '.ReplaceAllString',
        [regexp, string, string], [string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]),

    % (*Regexp).Split - split by pattern
    declare_binding(go, regex_split/4, '.Split',
        [regexp, string, int], [slice_string],
        [pure, deterministic, total, pattern(method_call), import('regexp')]).

% ============================================================================
% TIME OPERATION BINDINGS (time package)
% ============================================================================

register_time_bindings :-
    % time.Now - current time
    declare_binding(go, time_now/1, 'time.Now',
        [], [time],
        [effect(time), deterministic, total, import('time')]),

    % time.Parse - parse time string
    declare_binding(go, time_parse/3, 'time.Parse',
        [string, string], [time, error],
        [pure, deterministic, partial, import('time')]),

    % time.Unix - create time from unix timestamp
    declare_binding(go, time_unix/3, 'time.Unix',
        [int64, int64], [time],
        [pure, deterministic, total, import('time')]),

    % time.Since - time elapsed since
    declare_binding(go, time_since/2, 'time.Since',
        [time], [duration],
        [effect(time), deterministic, total, import('time')]),

    % time.Until - time until
    declare_binding(go, time_until/2, 'time.Until',
        [time], [duration],
        [effect(time), deterministic, total, import('time')]),

    % time.Sleep - sleep for duration
    declare_binding(go, time_sleep/1, 'time.Sleep',
        [duration], [],
        [effect(time), deterministic, import('time')]),

    % (*Time).Format - format time
    declare_binding(go, time_format/3, '.Format',
        [time, string], [string],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % (*Time).Unix - get unix timestamp
    declare_binding(go, time_to_unix/2, '.Unix',
        [time], [int64],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % (*Time).Add - add duration to time
    declare_binding(go, time_add/3, '.Add',
        [time, duration], [time],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % (*Time).Sub - subtract times
    declare_binding(go, time_sub/3, '.Sub',
        [time, time], [duration],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % (*Time).Before - check if before
    declare_binding(go, time_before/3, '.Before',
        [time, time], [bool],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % (*Time).After - check if after
    declare_binding(go, time_after/3, '.After',
        [time, time], [bool],
        [pure, deterministic, total, pattern(method_call), import('time')]),

    % Duration constants
    declare_binding(go, nanosecond/1, 'time.Nanosecond',
        [], [duration],
        [pure, deterministic, total, import('time')]),

    declare_binding(go, microsecond/1, 'time.Microsecond',
        [], [duration],
        [pure, deterministic, total, import('time')]),

    declare_binding(go, millisecond/1, 'time.Millisecond',
        [], [duration],
        [pure, deterministic, total, import('time')]),

    declare_binding(go, second/1, 'time.Second',
        [], [duration],
        [pure, deterministic, total, import('time')]),

    declare_binding(go, minute/1, 'time.Minute',
        [], [duration],
        [pure, deterministic, total, import('time')]),

    declare_binding(go, hour/1, 'time.Hour',
        [], [duration],
        [pure, deterministic, total, import('time')]).

% ============================================================================
% PATH OPERATION BINDINGS (path/filepath package)
% ============================================================================

register_path_bindings :-
    % filepath.Join - join path components
    declare_binding(go, path_join/2, 'filepath.Join',
        [list], [string],
        [pure, deterministic, total, import('path/filepath'), variadic]),

    % filepath.Dir - get directory
    declare_binding(go, path_dir/2, 'filepath.Dir',
        [string], [string],
        [pure, deterministic, total, import('path/filepath')]),

    % filepath.Base - get base name
    declare_binding(go, path_base/2, 'filepath.Base',
        [string], [string],
        [pure, deterministic, total, import('path/filepath')]),

    % filepath.Ext - get extension
    declare_binding(go, path_ext/2, 'filepath.Ext',
        [string], [string],
        [pure, deterministic, total, import('path/filepath')]),

    % filepath.Abs - get absolute path
    declare_binding(go, path_abs/2, 'filepath.Abs',
        [string], [string, error],
        [effect(io), deterministic, import('path/filepath')]),

    % filepath.Rel - get relative path
    declare_binding(go, path_rel/3, 'filepath.Rel',
        [string, string], [string, error],
        [pure, deterministic, partial, import('path/filepath')]),

    % filepath.Clean - clean path
    declare_binding(go, path_clean/2, 'filepath.Clean',
        [string], [string],
        [pure, deterministic, total, import('path/filepath')]),

    % filepath.Split - split path
    declare_binding(go, path_split/3, 'filepath.Split',
        [string], [string, string],
        [pure, deterministic, total, import('path/filepath')]),

    % filepath.Match - match pattern
    declare_binding(go, path_match/3, 'filepath.Match',
        [string, string], [bool, error],
        [pure, deterministic, partial, import('path/filepath')]),

    % filepath.Glob - glob pattern
    declare_binding(go, path_glob/2, 'filepath.Glob',
        [string], [slice_string, error],
        [effect(io), deterministic, import('path/filepath')]),

    % filepath.Walk - walk directory tree
    declare_binding(go, path_walk/3, 'filepath.Walk',
        [string, func], [error],
        [effect(io), deterministic, import('path/filepath')]).

% ============================================================================
% COLLECTION OPERATION BINDINGS (slices, maps, sort packages)
% ============================================================================

register_collection_bindings :-
    % -------------------------------------------
    % slices package (Go 1.21+)
    % -------------------------------------------

    % slices.Sort - sort slice
    declare_binding(go, slices_sort/1, 'slices.Sort',
        [slice], [],
        [effect(state), deterministic, import('slices')]),

    % slices.SortFunc - sort with custom function
    declare_binding(go, slices_sort_func/2, 'slices.SortFunc',
        [slice, func], [],
        [effect(state), deterministic, import('slices')]),

    % slices.Contains - check if contains element
    declare_binding(go, slices_contains/3, 'slices.Contains',
        [slice, any], [bool],
        [pure, deterministic, total, import('slices')]),

    % slices.Index - find index of element
    declare_binding(go, slices_index/3, 'slices.Index',
        [slice, any], [int],
        [pure, deterministic, total, import('slices')]),

    % slices.Reverse - reverse slice in place
    declare_binding(go, slices_reverse/1, 'slices.Reverse',
        [slice], [],
        [effect(state), deterministic, import('slices')]),

    % slices.Equal - check equality
    declare_binding(go, slices_equal/3, 'slices.Equal',
        [slice, slice], [bool],
        [pure, deterministic, total, import('slices')]),

    % slices.Clone - clone slice
    declare_binding(go, slices_clone/2, 'slices.Clone',
        [slice], [slice],
        [pure, deterministic, total, import('slices')]),

    % slices.Compact - remove consecutive duplicates
    declare_binding(go, slices_compact/2, 'slices.Compact',
        [slice], [slice],
        [pure, deterministic, total, import('slices')]),

    % -------------------------------------------
    % maps package (Go 1.21+)
    % -------------------------------------------

    % maps.Clone - clone map
    declare_binding(go, maps_clone/2, 'maps.Clone',
        [map], [map],
        [pure, deterministic, total, import('maps')]),

    % maps.Equal - check equality
    declare_binding(go, maps_equal/3, 'maps.Equal',
        [map, map], [bool],
        [pure, deterministic, total, import('maps')]),

    % maps.Keys - get keys
    declare_binding(go, maps_keys/2, 'maps.Keys',
        [map], [slice],
        [pure, deterministic, total, import('maps')]),

    % maps.Values - get values
    declare_binding(go, maps_values/2, 'maps.Values',
        [map], [slice],
        [pure, deterministic, total, import('maps')]),

    % maps.DeleteFunc - delete by predicate
    declare_binding(go, maps_delete_func/2, 'maps.DeleteFunc',
        [map, func], [],
        [effect(state), deterministic, import('maps')]),

    % -------------------------------------------
    % sort package
    % -------------------------------------------

    % sort.Strings - sort string slice
    declare_binding(go, sort_strings/1, 'sort.Strings',
        [slice_string], [],
        [effect(state), deterministic, import('sort')]),

    % sort.Ints - sort int slice
    declare_binding(go, sort_ints/1, 'sort.Ints',
        [slice_int], [],
        [effect(state), deterministic, import('sort')]),

    % sort.Float64s - sort float64 slice
    declare_binding(go, sort_float64s/1, 'sort.Float64s',
        [slice_float64], [],
        [effect(state), deterministic, import('sort')]),

    % sort.StringsAreSorted - check if sorted
    declare_binding(go, sort_strings_sorted/2, 'sort.StringsAreSorted',
        [slice_string], [bool],
        [pure, deterministic, total, import('sort')]),

    % sort.IntsAreSorted - check if sorted
    declare_binding(go, sort_ints_sorted/2, 'sort.IntsAreSorted',
        [slice_int], [bool],
        [pure, deterministic, total, import('sort')]),

    % sort.Search - binary search
    declare_binding(go, sort_search/3, 'sort.Search',
        [int, func], [int],
        [pure, deterministic, total, import('sort')]).

% ============================================================================
% CONVERSION OPERATION BINDINGS (strconv package)
% ============================================================================

register_conversion_bindings :-
    % -------------------------------------------
    % String to Number
    % -------------------------------------------

    % strconv.Atoi - string to int
    declare_binding(go, atoi/2, 'strconv.Atoi',
        [string], [int, error],
        [pure, deterministic, partial, import('strconv')]),

    % strconv.ParseInt - string to int with base
    declare_binding(go, parse_int/4, 'strconv.ParseInt',
        [string, int, int], [int64, error],
        [pure, deterministic, partial, import('strconv')]),

    % strconv.ParseFloat - string to float
    declare_binding(go, parse_float/3, 'strconv.ParseFloat',
        [string, int], [float64, error],
        [pure, deterministic, partial, import('strconv')]),

    % strconv.ParseBool - string to bool
    declare_binding(go, parse_bool/2, 'strconv.ParseBool',
        [string], [bool, error],
        [pure, deterministic, partial, import('strconv')]),

    % -------------------------------------------
    % Number to String
    % -------------------------------------------

    % strconv.Itoa - int to string
    declare_binding(go, itoa/2, 'strconv.Itoa',
        [int], [string],
        [pure, deterministic, total, import('strconv')]),

    % strconv.FormatInt - int64 to string with base
    declare_binding(go, format_int/3, 'strconv.FormatInt',
        [int64, int], [string],
        [pure, deterministic, total, import('strconv')]),

    % strconv.FormatFloat - float to string
    declare_binding(go, format_float/5, 'strconv.FormatFloat',
        [float64, byte, int, int], [string],
        [pure, deterministic, total, import('strconv')]),

    % strconv.FormatBool - bool to string
    declare_binding(go, format_bool/2, 'strconv.FormatBool',
        [bool], [string],
        [pure, deterministic, total, import('strconv')]),

    % -------------------------------------------
    % Quoting
    % -------------------------------------------

    % strconv.Quote - quote string
    declare_binding(go, quote/2, 'strconv.Quote',
        [string], [string],
        [pure, deterministic, total, import('strconv')]),

    % strconv.Unquote - unquote string
    declare_binding(go, unquote/2, 'strconv.Unquote',
        [string], [string, error],
        [pure, deterministic, partial, import('strconv')]).

% ============================================================================
% TESTS
% ============================================================================

test_go_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Go Bindings Tests                     ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing Go bindings~n', []),
    init_go_bindings,
    format('[✓] Go bindings initialized~n~n', []),

    % Test builtin bindings exist
    format('[Test 2] Checking built-in bindings~n', []),
    (   go_binding(length/2, 'len', _, _, _)
    ->  format('[✓] length/2 -> len binding exists~n', [])
    ;   format('[✗] length/2 binding missing~n', []), fail
    ),
    (   go_binding(append/3, 'append', _, _, _)
    ->  format('[✓] append/3 -> append binding exists~n', [])
    ;   format('[✗] append/3 binding missing~n', []), fail
    ),

    % Test string bindings exist
    format('~n[Test 3] Checking string bindings~n', []),
    (   go_binding(string_split/3, 'strings.Split', _, _, Opts1),
        member(import('strings'), Opts1)
    ->  format('[✓] string_split/3 has import(strings)~n', [])
    ;   format('[✗] string_split/3 binding missing~n', []), fail
    ),

    % Test math bindings with import
    format('~n[Test 4] Checking math bindings with imports~n', []),
    (   go_binding(sqrt/2, 'math.Sqrt', _, _, Opts2),
        member(import('math'), Opts2)
    ->  format('[✓] sqrt/2 has import(math)~n', [])
    ;   format('[✗] sqrt/2 missing import~n', []), fail
    ),

    % Test JSON bindings
    format('~n[Test 5] Checking JSON bindings~n', []),
    (   go_binding(json_marshal/2, 'json.Marshal', _, _, Opts3),
        member(import('encoding/json'), Opts3)
    ->  format('[✓] json_marshal/2 has import(encoding/json)~n', [])
    ;   format('[✗] json_marshal/2 missing~n', []), fail
    ),

    % Test regex bindings
    format('~n[Test 6] Checking regex bindings~n', []),
    (   go_binding(regex_compile/2, 'regexp.Compile', _, _, Opts4),
        member(import('regexp'), Opts4)
    ->  format('[✓] regex_compile/2 has import(regexp)~n', [])
    ;   format('[✗] regex_compile/2 missing~n', []), fail
    ),

    % Test time bindings
    format('~n[Test 7] Checking time bindings~n', []),
    (   go_binding(time_now/1, 'time.Now', _, _, Opts5),
        member(import('time'), Opts5)
    ->  format('[✓] time_now/1 has import(time)~n', [])
    ;   format('[✗] time_now/1 missing~n', []), fail
    ),

    % Test path bindings
    format('~n[Test 8] Checking path bindings~n', []),
    (   go_binding(path_join/2, 'filepath.Join', _, _, Opts6),
        member(import('path/filepath'), Opts6)
    ->  format('[✓] path_join/2 has import(path/filepath)~n', [])
    ;   format('[✗] path_join/2 missing~n', []), fail
    ),

    % Test strconv bindings
    format('~n[Test 9] Checking strconv bindings~n', []),
    (   go_binding(atoi/2, 'strconv.Atoi', _, _, Opts7),
        member(import('strconv'), Opts7)
    ->  format('[✓] atoi/2 has import(strconv)~n', [])
    ;   format('[✗] atoi/2 missing~n', []), fail
    ),

    % Test go_binding_import/2
    format('~n[Test 10] Testing go_binding_import/2~n', []),
    (   go_binding_import(sqrt/2, Import),
        Import == 'math'
    ->  format('[✓] go_binding_import(sqrt/2, math) works~n', [])
    ;   format('[✗] go_binding_import failed~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 11] Counting total bindings~n', []),
    findall(P, go_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total Go bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All Go Bindings Tests Passed          ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).

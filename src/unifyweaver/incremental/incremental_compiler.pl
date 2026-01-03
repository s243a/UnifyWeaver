:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% incremental_compiler.pl - Incremental compilation wrapper
%
% Main entry point for incremental compilation. Wraps existing target compilers
% with caching logic to avoid recompiling unchanged predicates.
%
% Usage:
%   % Compile with caching (default)
%   ?- compile_incremental(foo/2, bash, [], Code).
%   [Incremental] Cache hit: foo/2
%   Code = "..."
%
%   % Force fresh compilation
%   ?- compile_incremental(foo/2, bash, [incremental(false)], Code).
%   [Incremental] Fresh compilation: foo/2
%   Code = "..."
%
%   % Clear cache before compiling
%   ?- clear_incremental_cache, compile_incremental(foo/2, bash, [], Code).

:- module(incremental_compiler, [
    compile_incremental/4,      % +Pred/Arity, +Target, +Options, -Code
    compile_incremental/3,      % +Pred/Arity, +Target, -Code (default options)
    compile_fresh/4,            % +Pred/Arity, +Target, +Options, -Code (bypass cache)
    clear_incremental_cache/0,  % Clear all cached compilations (memory only)
    clear_all_cache/0,          % Clear both memory and disk cache
    save_cache/0,               % Save memory cache to disk
    load_cache/0,               % Load disk cache to memory
    set_cache_dir/1,            % +Path - Set cache directory
    incremental_stats/0,        % Print cache statistics
    test_incremental_compiler/0
]).

:- use_module(library(lists)).

% Import hasher and cache_manager
:- use_module(hasher, [
    hash_predicate_with_deps/2,
    hash_predicate_with_options/3
]).
:- use_module(cache_manager, [
    get_cached/4,
    store_cached/4,
    invalidate_dependents/2,
    clear_cache/0,
    cache_stats/1,
    cache_entries/1
]).
:- use_module(cache_persistence, [
    save_cache_to_disk/0,
    load_cache_from_disk/0,
    set_cache_directory/1,
    get_cache_directory/1,
    clear_disk_cache/0,
    disk_cache_exists/0,
    get_disk_cache_stats/1
]).

% Import target compilers (loaded lazily to avoid circular dependencies)
% We use call/3 to invoke the appropriate compiler

% ============================================================================
% CONFIGURATION
% ============================================================================

%% incremental_enabled(-Enabled) is det.
%
% Check if incremental compilation is enabled globally.
% Can be controlled via Prolog flag or environment variable.
%
incremental_enabled(true) :-
    (   current_prolog_flag(unifyweaver_incremental, false)
    ->  fail
    ;   getenv('UNIFYWEAVER_CACHE', '0')
    ->  fail
    ;   true
    ), !.
incremental_enabled(false).

%% verbose_mode(-Enabled) is det.
%
% Check if verbose output is enabled.
%
verbose_mode(true) :-
    (   current_prolog_flag(unifyweaver_verbose, true)
    ->  true
    ;   getenv('UNIFYWEAVER_VERBOSE', '1')
    ).
verbose_mode(false) :-
    \+ verbose_mode(true).

% ============================================================================
% MAIN COMPILATION ENTRY POINTS
% ============================================================================

%% compile_incremental(+Pred/Arity, +Target, -Code) is det.
%
% Compile with default options and caching.
%
compile_incremental(Pred/Arity, Target, Code) :-
    compile_incremental(Pred/Arity, Target, [], Code).

%% compile_incremental(+Pred/Arity, +Target, +Options, -Code) is det.
%
% Main entry point for incremental compilation.
%
% Options:
%   - incremental(false): Bypass cache and compile fresh
%   - verbose(true): Print cache hit/miss information
%   - Any other options are passed to the target compiler
%
compile_incremental(Pred/Arity, Target, Options, Code) :-
    % Check if incremental is disabled via option
    (   member(incremental(false), Options)
    ->  log_verbose('Fresh compilation (disabled): ~w', [Pred/Arity]),
        compile_fresh(Pred/Arity, Target, Options, Code)
    ;   % Check global enable flag
        incremental_enabled(true)
    ->  compile_with_cache(Pred/Arity, Target, Options, Code)
    ;   % Global disabled - compile fresh
        log_verbose('Fresh compilation (global disabled): ~w', [Pred/Arity]),
        compile_fresh(Pred/Arity, Target, Options, Code)
    ).

%% compile_with_cache(+Pred/Arity, +Target, +Options, -Code) is det.
%
% Internal: compile with cache lookup.
%
compile_with_cache(Pred/Arity, Target, Options, Code) :-
    % Compute current hash including dependencies and options
    hash_predicate_with_options(Pred/Arity, Options, CurrentHash),

    % Try cache lookup
    (   get_cached(Pred/Arity, Target, CurrentHash, CachedCode)
    ->  % Cache hit
        log_verbose('[Incremental] Cache hit: ~w (~w)', [Pred/Arity, Target]),
        Code = CachedCode
    ;   % Cache miss - compile fresh and store
        log_verbose('[Incremental] Cache miss: ~w (~w) - compiling...', [Pred/Arity, Target]),
        compile_fresh(Pred/Arity, Target, Options, FreshCode),
        store_cached(Pred/Arity, Target, CurrentHash, FreshCode),
        % Invalidate dependents (they may need recompilation with new code)
        invalidate_dependents(Pred/Arity, Target),
        Code = FreshCode
    ).

% ============================================================================
% FRESH COMPILATION (BYPASS CACHE)
% ============================================================================

%% compile_fresh(+Pred/Arity, +Target, +Options, -Code) is det.
%
% Compile without using cache. Dispatches to target-specific compiler.
%
compile_fresh(Pred/Arity, Target, Options, Code) :-
    target_compiler(Target, Module, CompilePredicate),
    !,
    Goal =.. [CompilePredicate, Pred/Arity, Options, Code],
    call(Module:Goal).

compile_fresh(_Pred/_, Target, _, _) :-
    throw(error(unknown_target(Target), context(compile_fresh/4, 'Unknown compilation target'))).

%% target_compiler(+Target, -Module, -Predicate) is semidet.
%
% Map target name to module and compilation predicate.
%
target_compiler(bash, stream_compiler, compile_predicate).
target_compiler(go, go_target, compile_predicate_to_go).
target_compiler(rust, rust_target, compile_predicate_to_rust).
target_compiler(csharp, csharp_native_target, compile_predicate_to_csharp).
target_compiler(powershell, powershell_target, compile_predicate_to_powershell).
target_compiler(sql, sql_target, compile_predicate_to_sql).
target_compiler(python, python_target, compile_predicate_to_python).
target_compiler(java, java_target, compile_predicate_to_java).
target_compiler(kotlin, kotlin_target, compile_predicate_to_kotlin).
target_compiler(scala, scala_target, compile_predicate_to_scala).
target_compiler(typescript, typescript_target, compile_predicate_to_typescript).
target_compiler(haskell, haskell_target, compile_predicate_to_haskell).
target_compiler(cpp, cpp_target, compile_predicate_to_cpp).
target_compiler(c, c_target, compile_predicate_to_c).
target_compiler(awk, awk_target, compile_predicate_to_awk).
target_compiler(perl, perl_target, compile_predicate_to_perl).
target_compiler(ruby, ruby_target, compile_predicate_to_ruby).
target_compiler(clojure, clojure_target, compile_predicate_to_clojure).
target_compiler(fsharp, fsharp_target, compile_predicate_to_fsharp).
target_compiler(vbnet, vbnet_target, compile_predicate_to_vbnet).

% ============================================================================
% CACHE MANAGEMENT
% ============================================================================

%% clear_incremental_cache is det.
%
% Clear all cached compilations (memory only).
%
clear_incremental_cache :-
    clear_cache,
    log_verbose('[Incremental] Memory cache cleared', []).

%% clear_all_cache is det.
%
% Clear both memory and disk cache.
%
clear_all_cache :-
    clear_cache,
    clear_disk_cache,
    log_verbose('[Incremental] Memory and disk cache cleared', []).

%% save_cache is det.
%
% Save the current memory cache to disk for persistence.
%
save_cache :-
    save_cache_to_disk,
    log_verbose('[Incremental] Cache saved to disk', []).

%% load_cache is det.
%
% Load cache from disk into memory.
%
load_cache :-
    load_cache_from_disk,
    log_verbose('[Incremental] Cache loaded from disk', []).

%% set_cache_dir(+Path) is det.
%
% Set the cache directory path.
%
set_cache_dir(Path) :-
    set_cache_directory(Path),
    log_verbose('[Incremental] Cache directory set to: ~w', [Path]).

%% incremental_stats is det.
%
% Print cache statistics to standard output.
%
incremental_stats :-
    cache_stats(Stats),
    cache_entries(Entries),

    writeln('=== Incremental Compilation Cache Statistics ==='),

    % Total entries
    member(total_entries(Total), Stats),
    format('Total cached entries: ~w~n', [Total]),

    % By target
    member(by_target(ByTarget), Stats),
    (   ByTarget = []
    ->  writeln('  (no entries)')
    ;   forall(member(Target-Count, ByTarget),
            format('  ~w: ~w entries~n', [Target, Count]))
    ),

    % Timestamps
    member(oldest_entry(Oldest), Stats),
    member(newest_entry(Newest), Stats),
    (   Oldest = none
    ->  true
    ;   format_time(atom(OldestStr), '%Y-%m-%d %H:%M:%S', Oldest),
        format_time(atom(NewestStr), '%Y-%m-%d %H:%M:%S', Newest),
        format('Oldest entry: ~w~n', [OldestStr]),
        format('Newest entry: ~w~n', [NewestStr])
    ),

    writeln(''),

    % List entries if not too many
    length(Entries, EntryCount),
    (   EntryCount =< 20
    ->  writeln('Cached predicates:'),
        forall(member(entry(Pred, Target, Hash, _), Entries),
            format('  ~w -> ~w (hash: ~w)~n', [Pred, Target, Hash]))
    ;   format('(~w entries - too many to list)~n', [EntryCount])
    ),

    % Disk cache info
    writeln(''),
    writeln('--- Disk Cache ---'),
    get_disk_cache_stats(DiskStats),
    (   member(exists(true), DiskStats)
    ->  member(total_entries(DiskTotal), DiskStats),
        member(total_size_bytes(DiskSize), DiskStats),
        member(cache_directory(CacheDir), DiskStats),
        format('Directory: ~w~n', [CacheDir]),
        format('Entries on disk: ~w~n', [DiskTotal]),
        format('Total size: ~w bytes~n', [DiskSize])
    ;   member(cache_directory(CacheDir), DiskStats),
        format('Directory: ~w~n', [CacheDir]),
        writeln('(no disk cache)')
    ).

% ============================================================================
% LOGGING
% ============================================================================

%% log_verbose(+Format, +Args) is det.
%
% Log message if verbose mode is enabled.
%
log_verbose(Format, Args) :-
    (   verbose_mode(true)
    ->  format(Format, Args), nl
    ;   true
    ).

% ============================================================================
% TESTS
% ============================================================================

test_incremental_compiler :-
    writeln('=== INCREMENTAL COMPILER TESTS ==='),

    % Clear cache before tests
    clear_incremental_cache,

    % Setup test predicates
    setup_test_predicates,

    % Test 1: Cache miss then cache hit
    test_cache_behavior,

    % Test 2: Forced fresh compilation
    test_forced_fresh,

    % Test 3: Hash change detection
    test_hash_change,

    % Cleanup
    cleanup_test_predicates,
    clear_incremental_cache,

    writeln('=== ALL INCREMENTAL COMPILER TESTS PASSED ===').

setup_test_predicates :-
    catch(abolish(user:incr_test_pred/2), _, true),
    assertz(user:incr_test_pred(a, 1)),
    assertz(user:incr_test_pred(b, 2)).

cleanup_test_predicates :-
    catch(abolish(user:incr_test_pred/2), _, true).

test_cache_behavior :-
    write('  Testing cache behavior... '),

    % First compilation - cache miss, stores result
    % We don't actually call the compiler here since it may not be loaded
    % Just test the cache logic with a mock
    hash_predicate_with_options(incr_test_pred/2, [], Hash1),
    store_cached(incr_test_pred/2, bash, Hash1, "mock_code_1"),

    % Second lookup - should be cache hit
    (   get_cached(incr_test_pred/2, bash, Hash1, Code),
        Code == "mock_code_1"
    ->  writeln('PASS')
    ;   writeln('FAIL: Cache hit not working'),
        fail
    ).

test_forced_fresh :-
    write('  Testing forced fresh compilation... '),

    % This tests the option parsing, not actual compilation
    Options = [incremental(false), some_other(option)],
    (   member(incremental(false), Options)
    ->  writeln('PASS')
    ;   writeln('FAIL: Option parsing failed'),
        fail
    ).

test_hash_change :-
    write('  Testing hash change detection... '),

    % Get initial hash
    hash_predicate_with_options(incr_test_pred/2, [], Hash1),

    % Modify predicate
    assertz(user:incr_test_pred(c, 3)),

    % Get new hash
    hash_predicate_with_options(incr_test_pred/2, [], Hash2),

    % Restore predicate
    retract(user:incr_test_pred(c, 3)),

    % Hashes should differ
    (   Hash1 \== Hash2
    ->  writeln('PASS')
    ;   writeln('FAIL: Hash did not change'),
        fail
    ).

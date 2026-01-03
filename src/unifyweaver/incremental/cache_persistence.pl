:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% cache_persistence.pl - File-based cache persistence for incremental compilation
%
% Saves and loads compilation cache to/from disk for persistence across sessions.
% Uses a simple directory structure with a JSON manifest.
%
% Directory structure:
%   .unifyweaver_cache/
%   ├── manifest.json           # Cache metadata and index
%   ├── bash/
%   │   ├── foo_2_a1b2c3d4.code # Compiled code files
%   │   └── bar_3_e5f6g7h8.code
%   ├── go/
%   │   └── ...
%   └── ...
%
% Usage:
%   % Save current cache to disk
%   ?- save_cache_to_disk.
%   true.
%
%   % Load cache from disk
%   ?- load_cache_from_disk.
%   true.
%
%   % Configure cache directory
%   ?- set_cache_directory('/path/to/.cache').

:- module(cache_persistence, [
    save_cache_to_disk/0,       % Save current in-memory cache to disk
    load_cache_from_disk/0,     % Load cache from disk into memory
    set_cache_directory/1,      % +Path - Set cache directory path
    get_cache_directory/1,      % -Path - Get current cache directory
    clear_disk_cache/0,         % Remove all disk cache files
    clear_disk_cache_target/1,  % +Target - Remove cache for specific target
    disk_cache_exists/0,        % Check if disk cache exists
    get_disk_cache_stats/1,     % -Stats - Get disk cache statistics
    test_cache_persistence/0
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).     % For directory_file_path, make_directory_path
:- use_module(library(readutil)).    % For read_file_to_string
:- use_module(library(http/json)).   % For JSON parsing

% Import cache_manager for accessing in-memory cache
:- use_module(cache_manager, [
    compilation_cache/5,
    store_cached/4,
    clear_cache/0,
    cache_entries/1
]).

% ============================================================================
% CONFIGURATION
% ============================================================================

%% cache_directory(-Path) is det.
%
% Dynamic predicate storing the cache directory path.
% Default: .unifyweaver_cache in current working directory.
%
:- dynamic cache_directory/1.
cache_directory('.unifyweaver_cache').

%% set_cache_directory(+Path) is det.
%
% Set the cache directory path.
%
set_cache_directory(Path) :-
    retractall(cache_directory(_)),
    assertz(cache_directory(Path)).

%% get_cache_directory(-Path) is det.
%
% Get the current cache directory path.
%
get_cache_directory(Path) :-
    cache_directory(Path).

% ============================================================================
% SAVING CACHE TO DISK
% ============================================================================

%% save_cache_to_disk is det.
%
% Save the current in-memory cache to disk.
% Creates the cache directory if it doesn't exist.
%
save_cache_to_disk :-
    get_cache_directory(CacheDir),
    ensure_cache_directory(CacheDir),
    cache_entries(Entries),
    save_all_entries(CacheDir, Entries),
    save_manifest(CacheDir, Entries).

%% ensure_cache_directory(+CacheDir) is det.
%
% Create cache directory and target subdirectories if they don't exist.
%
ensure_cache_directory(CacheDir) :-
    (   exists_directory(CacheDir)
    ->  true
    ;   make_directory_path(CacheDir)
    ).

%% save_all_entries(+CacheDir, +Entries) is det.
%
% Save all cache entries to disk.
%
save_all_entries(CacheDir, Entries) :-
    forall(
        member(entry(Pred/Arity, Target, Hash, _Timestamp), Entries),
        save_entry(CacheDir, Pred/Arity, Target, Hash)
    ).

%% save_entry(+CacheDir, +Pred/Arity, +Target, +Hash) is det.
%
% Save a single cache entry to disk.
%
save_entry(CacheDir, Pred/Arity, Target, Hash) :-
    % Get the cached code
    compilation_cache(Pred/Arity, Target, Hash, Code, _),
    !,
    % Ensure target directory exists
    atom_string(Target, TargetStr),
    directory_file_path(CacheDir, TargetStr, TargetDir),
    ensure_cache_directory(TargetDir),
    % Create filename: predname_arity_hash.code
    format(atom(Filename), '~w_~w_~w.code', [Pred, Arity, Hash]),
    directory_file_path(TargetDir, Filename, FilePath),
    % Write code to file
    (   is_list(Code)
    ->  atomic_list_concat(Code, '\n', CodeStr)
    ;   atom_string(Code, CodeStr)
    ),
    open(FilePath, write, Stream, [encoding(utf8)]),
    write(Stream, CodeStr),
    close(Stream).
save_entry(_, _, _, _).  % Entry not found, skip

%% save_manifest(+CacheDir, +Entries) is det.
%
% Save the manifest.json file with cache metadata.
%
save_manifest(CacheDir, Entries) :-
    directory_file_path(CacheDir, 'manifest.json', ManifestPath),
    get_time(Now),
    format_time(atom(Timestamp), '%Y-%m-%dT%H:%M:%SZ', Now),

    % Build entries by target
    findall(Target-EntryList,
        (   member(Target, [bash, go, rust, csharp, powershell, sql, python,
                           java, kotlin, scala, typescript, haskell, cpp, c,
                           awk, perl, ruby, clojure, fsharp, vbnet]),
            findall(EntryJson,
                (   member(entry(Pred/Arity, Target, Hash, EntryTimestamp), Entries),
                    format(atom(PredStr), '~w/~w', [Pred, Arity]),
                    format(atom(FileStr), '~w_~w_~w.code', [Pred, Arity, Hash]),
                    EntryJson = json([
                        predicate=PredStr,
                        hash=Hash,
                        file=FileStr,
                        timestamp=EntryTimestamp
                    ])
                ),
                EntryList),
            EntryList \= []
        ),
        TargetEntries),

    % Build manifest JSON
    findall(TargetAtom=json(TargetData),
        (   member(Target-EntryList, TargetEntries),
            atom_string(Target, TargetAtom),
            findall(PredAtom=EntryJson,
                (   member(EntryJson, EntryList),
                    EntryJson = json(Fields),
                    member(predicate=PredStr, Fields),
                    atom_string(PredAtom, PredStr)
                ),
                TargetData)
        ),
        EntriesJson),

    Manifest = json([
        version='1.0',
        created=Timestamp,
        entries=json(EntriesJson)
    ]),

    % Write manifest
    open(ManifestPath, write, Stream, [encoding(utf8)]),
    json_write(Stream, Manifest, [width(80)]),
    close(Stream).

% ============================================================================
% LOADING CACHE FROM DISK
% ============================================================================

%% load_cache_from_disk is det.
%
% Load cache from disk into memory.
% Existing in-memory cache entries are preserved (disk entries merged in).
%
load_cache_from_disk :-
    get_cache_directory(CacheDir),
    directory_file_path(CacheDir, 'manifest.json', ManifestPath),
    (   exists_file(ManifestPath)
    ->  read_manifest(ManifestPath, Entries),
        load_all_entries(CacheDir, Entries)
    ;   true  % No manifest, nothing to load
    ).

%% read_manifest(+ManifestPath, -Entries) is det.
%
% Read the manifest.json and extract entry metadata.
%
read_manifest(ManifestPath, Entries) :-
    open(ManifestPath, read, Stream, [encoding(utf8)]),
    json_read(Stream, Json),
    close(Stream),
    extract_entries_from_manifest(Json, Entries).

%% extract_entries_from_manifest(+Json, -Entries) is det.
%
% Extract cache entries from manifest JSON.
%
extract_entries_from_manifest(json(Fields), Entries) :-
    (   member(entries=json(TargetEntries), Fields)
    ->  findall(entry(Pred/Arity, Target, Hash, File, Timestamp),
            (   member(TargetAtom=json(PredEntries), TargetEntries),
                atom_string(TargetAtom, TargetStr),
                atom_string(Target, TargetStr),
                member(_PredAtom=json(EntryFields), PredEntries),
                member(predicate=PredStr, EntryFields),
                parse_pred_arity(PredStr, Pred, Arity),
                member(hash=Hash, EntryFields),
                member(file=File, EntryFields),
                (member(timestamp=Timestamp, EntryFields) -> true ; Timestamp = 0)
            ),
            Entries)
    ;   Entries = []
    ).

%% parse_pred_arity(+PredStr, -Pred, -Arity) is det.
%
% Parse "foo/2" into Pred=foo, Arity=2.
%
parse_pred_arity(PredStr, Pred, Arity) :-
    (   atom(PredStr) -> atom_string(PredStr, PredString) ; PredString = PredStr ),
    split_string(PredString, "/", "", [PredPart, ArityPart]),
    atom_string(Pred, PredPart),
    number_string(Arity, ArityPart).

%% load_all_entries(+CacheDir, +Entries) is det.
%
% Load all entries from disk into memory cache.
%
load_all_entries(CacheDir, Entries) :-
    forall(
        member(entry(Pred/Arity, Target, Hash, File, Timestamp), Entries),
        load_entry(CacheDir, Pred/Arity, Target, Hash, File, Timestamp)
    ).

%% load_entry(+CacheDir, +Pred/Arity, +Target, +Hash, +File, +Timestamp) is det.
%
% Load a single entry from disk into memory cache.
%
load_entry(CacheDir, Pred/Arity, Target, Hash, File, Timestamp) :-
    atom_string(Target, TargetStr),
    directory_file_path(CacheDir, TargetStr, TargetDir),
    (   atom(File) -> FileAtom = File ; atom_string(FileAtom, File) ),
    directory_file_path(TargetDir, FileAtom, FilePath),
    (   exists_file(FilePath)
    ->  read_file_to_string(FilePath, Code, [encoding(utf8)]),
        % Store in memory cache (using internal predicate)
        retractall(compilation_cache(Pred/Arity, Target, _, _, _)),
        assertz(compilation_cache(Pred/Arity, Target, Hash, Code, Timestamp))
    ;   true  % File missing, skip
    ).

% ============================================================================
% DISK CACHE MANAGEMENT
% ============================================================================

%% disk_cache_exists is semidet.
%
% Check if disk cache directory exists and has a manifest.
%
disk_cache_exists :-
    get_cache_directory(CacheDir),
    directory_file_path(CacheDir, 'manifest.json', ManifestPath),
    exists_file(ManifestPath).

%% clear_disk_cache is det.
%
% Remove all disk cache files and directories.
%
clear_disk_cache :-
    get_cache_directory(CacheDir),
    (   exists_directory(CacheDir)
    ->  remove_directory_recursive(CacheDir)
    ;   true
    ).

%% clear_disk_cache_target(+Target) is det.
%
% Remove disk cache for a specific target.
%
clear_disk_cache_target(Target) :-
    get_cache_directory(CacheDir),
    atom_string(Target, TargetStr),
    directory_file_path(CacheDir, TargetStr, TargetDir),
    (   exists_directory(TargetDir)
    ->  remove_directory_recursive(TargetDir)
    ;   true
    ).

%% remove_directory_recursive(+Dir) is det.
%
% Recursively delete a directory and all its contents.
%
remove_directory_recursive(Dir) :-
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files),
        forall(
            (member(File, Files), File \= '.', File \= '..'),
            (   directory_file_path(Dir, File, Path),
                (   exists_directory(Path)
                ->  remove_directory_recursive(Path)
                ;   delete_file(Path)
                )
            )
        ),
        delete_directory(Dir)
    ;   true
    ).

%% get_disk_cache_stats(-Stats) is det.
%
% Get statistics about the disk cache.
%
get_disk_cache_stats(Stats) :-
    get_cache_directory(CacheDir),
    (   disk_cache_exists
    ->  directory_file_path(CacheDir, 'manifest.json', ManifestPath),
        read_manifest(ManifestPath, Entries),
        length(Entries, TotalEntries),
        % Count by target
        findall(Target-Count,
            (   member(Target, [bash, go, rust, python, typescript]),
                findall(E, (member(E, Entries), E = entry(_, Target, _, _, _)), Es),
                length(Es, Count),
                Count > 0
            ),
            ByTarget),
        % Calculate total size
        calculate_cache_size(CacheDir, TotalSize),
        Stats = [
            exists(true),
            total_entries(TotalEntries),
            by_target(ByTarget),
            total_size_bytes(TotalSize),
            cache_directory(CacheDir)
        ]
    ;   Stats = [
            exists(false),
            total_entries(0),
            by_target([]),
            total_size_bytes(0),
            cache_directory(CacheDir)
        ]
    ).

%% calculate_cache_size(+Dir, -Size) is det.
%
% Calculate total size of files in cache directory.
%
calculate_cache_size(Dir, Size) :-
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files),
        findall(FileSize,
            (   member(File, Files),
                File \= '.', File \= '..',
                directory_file_path(Dir, File, Path),
                (   exists_directory(Path)
                ->  calculate_cache_size(Path, FileSize)
                ;   size_file(Path, FileSize)
                )
            ),
            Sizes),
        sum_list(Sizes, Size)
    ;   Size = 0
    ).

% ============================================================================
% TESTS
% ============================================================================

test_cache_persistence :-
    writeln('=== CACHE PERSISTENCE TESTS ==='),

    % Use a temporary test directory
    set_cache_directory('/tmp/unifyweaver_cache_test'),

    % Clear any existing test cache
    clear_disk_cache,

    % Test 1: Save and load round-trip
    test_save_load_roundtrip,

    % Test 2: Disk cache stats
    test_disk_cache_stats,

    % Test 3: Clear disk cache
    test_clear_disk_cache,

    % Cleanup
    clear_disk_cache,
    set_cache_directory('.unifyweaver_cache'),

    writeln('=== ALL CACHE PERSISTENCE TESTS PASSED ===').

test_save_load_roundtrip :-
    write('  Testing save/load round-trip... '),

    % Clear memory cache
    clear_cache,

    % Add some test entries to memory cache
    store_cached(test_persist/2, bash, 12345, "echo test"),
    store_cached(test_persist/2, go, 67890, "fmt.Println(\"test\")"),

    % Save to disk
    save_cache_to_disk,

    % Clear memory cache
    clear_cache,

    % Load from disk
    load_cache_from_disk,

    % Verify entries are back
    (   compilation_cache(test_persist/2, bash, 12345, BashCode, _),
        compilation_cache(test_persist/2, go, 67890, GoCode, _),
        BashCode = "echo test",
        GoCode = "fmt.Println(\"test\")"
    ->  writeln('PASS')
    ;   writeln('FAIL: Round-trip failed'),
        fail
    ).

test_disk_cache_stats :-
    write('  Testing disk cache stats... '),
    get_disk_cache_stats(Stats),
    (   member(exists(true), Stats),
        member(total_entries(N), Stats),
        N >= 2
    ->  writeln('PASS')
    ;   format('FAIL: Stats = ~w~n', [Stats]),
        fail
    ).

test_clear_disk_cache :-
    write('  Testing clear disk cache... '),
    clear_disk_cache,
    (   \+ disk_cache_exists
    ->  writeln('PASS')
    ;   writeln('FAIL: Cache should not exist'),
        fail
    ).

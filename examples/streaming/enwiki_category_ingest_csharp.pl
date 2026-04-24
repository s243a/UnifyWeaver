% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% examples/streaming/enwiki_category_ingest_csharp.pl
%
% Identical in intent to enwiki_category_ingest.pl, but uses the C#
% LMDB consumer instead of the Python one.  Demonstrates consumer-
% side target interchangeability — the `declare_target` for
% ingest_to_lmdb/3 is the only line that differs from the Python
% example.

:- use_module('../../src/unifyweaver/core/target_mapping').
:- use_module('../../src/unifyweaver/glue/streaming_glue').

:- declare_target(parse_mysql_rows/2, rust,
                  [leaf(true),
                   native_crate(mysql_stream)]).

%% ONE-LINE SWAP from the Python example:
%%   declare_target(ingest_to_lmdb/3, python, [leaf(true), script_path(...), ...]).
%% becomes:
%%   declare_target(ingest_to_lmdb/3, csharp, [leaf(true), project_dir(...)]).
%%
%% The C# project reads the same env-var contract as the Python
%% consumer, so the pipeline env block below is unchanged.

:- declare_target(ingest_to_lmdb/3, csharp,
                  [leaf(true),
                   project_dir('src/unifyweaver/runtime/csharp/lmdb_ingest')]).

process_dump(DumpPath, LmdbPath) :-
    make_directory_if_missing(LmdbPath),
    streaming_glue:run_streaming_pipeline(
        parse_mysql_rows/2,
        ingest_to_lmdb/3,
        [producer_args([DumpPath]),
         env([
            'UW_LMDB_PATH'    = LmdbPath,
            'UW_FILTER_COL'   = 4,
            'UW_FILTER_VAL'   = subcat,
            'UW_KEY_COL'      = 0,
            'UW_VAL_COL'      = 6,
            'UW_KEY_ENCODING' = int32_le,
            'UW_VAL_ENCODING' = int32_le,
            'UW_LMDB_DUPSORT' = 1,
            'UW_BATCH_SIZE'   = 50000
         ])],
        ExitCode
    ),
    (   ExitCode =:= 0
    ->  format(user_error, '[enwiki-ingest/csharp] pipeline completed OK~n', [])
    ;   format(user_error, '[enwiki-ingest/csharp] pipeline exited ~w~n', [ExitCode]),
        halt(ExitCode)
    ).

make_directory_if_missing(Path) :-
    (   exists_directory(Path)
    ->  true
    ;   make_directory_path(Path)
    ).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [DumpPath, LmdbPath]
    ->  process_dump(DumpPath, LmdbPath)
    ;   format(user_error,
               'usage: ... enwiki_category_ingest_csharp.pl -- <dump.sql.gz> <lmdb-path>~n',
               []),
        halt(1)
    ).

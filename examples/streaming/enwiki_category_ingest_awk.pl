% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% examples/streaming/enwiki_category_ingest_awk.pl
%
% Identical in intent to enwiki_category_ingest.pl, but uses the AWK
% parser variant instead of the Rust one.  The *only* difference from
% the Rust example is the declare_target block for parse_mysql_rows/2
% — everything downstream (process_dump, consumer declaration, CLI)
% is unchanged.  This is the multi-target demo: one line of Prolog
% swaps the implementation.

:- use_module('../../src/unifyweaver/core/target_mapping').
:- use_module('../../src/unifyweaver/glue/streaming_glue').

% ---------------------------------------------------------------------------
% ONE-LINE SWAP from the Rust example:
%   declare_target(parse_mysql_rows/2, rust, [leaf(true), native_crate(mysql_stream)]).
% becomes
%   declare_target(parse_mysql_rows/2, awk,  [leaf(true), script_path(...), input_filter(zcat)]).
% ---------------------------------------------------------------------------

:- declare_target(parse_mysql_rows/2, awk,
                  [leaf(true),
                   script_path('src/unifyweaver/runtime/awk/mysql_stream/parse_inserts.awk'),
                   input_filter(zcat),
                   awk_exec(gawk)]).

:- declare_target(ingest_to_lmdb/3, python,
                  [leaf(true),
                   script_path('src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py'),
                   python_min_version('3.9'),
                   pip_packages([lmdb])]).

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
    ->  format(user_error, '[enwiki-ingest/awk] pipeline completed OK~n', [])
    ;   format(user_error, '[enwiki-ingest/awk] pipeline exited ~w~n', [ExitCode]),
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
               'usage: ... enwiki_category_ingest_awk.pl -- <dump.sql.gz> <lmdb-path>~n',
               []),
        halt(1)
    ).

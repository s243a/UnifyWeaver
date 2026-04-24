% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% examples/streaming/enwiki_category_ingest.pl
%
% End-to-end streaming ingest example for MediaWiki categorylinks dumps.
% Parses a gzipped SQL dump (simplewiki or enwiki) and writes subcat
% edges into an LMDB database, suitable for the WAM Haskell scaling
% benchmarks.
%
% Usage:
%   swipl -q -g "process_dump('.../categorylinks.sql.gz', './cats.lmdb')" \
%         -t halt examples/streaming/enwiki_category_ingest.pl
%
% Or, for a Prolog top-level:
%   ?- [examples/streaming/enwiki_category_ingest].
%   ?- process_dump('path/to/dump.sql.gz', 'path/to/output.lmdb').

:- use_module('../../src/unifyweaver/core/target_mapping').
:- use_module('../../src/unifyweaver/glue/streaming_glue').

% ---------------------------------------------------------------------------
% Target declarations — the whole pipeline lives here.
% Parser is Rust (leaf primitive, general-purpose MySQL tokenizer).
% Consumer is Python (schema-specific filter/project/write to LMDB).
% Switching parser to AWK / Haskell is a one-line change below.
% ---------------------------------------------------------------------------

:- declare_target(parse_mysql_rows/2, rust,
                  [leaf(true),
                   native_crate(mysql_stream)]).

:- declare_target(ingest_to_lmdb/3, python,
                  [leaf(true),
                   script_path('src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py'),
                   python_min_version('3.9'),
                   pip_packages([lmdb])]).

% ---------------------------------------------------------------------------
% Pipeline composition.
% The declarative forall-over-side-effects shape: produce raw rows in
% Rust, filter/project/write in Python.  The filter/project logic lives
% in env vars for now; transpiling it from Prolog into the consumer is
% a future milestone (see 01-philosophy.md §Declarative specialization).
% ---------------------------------------------------------------------------

%% process_dump(+DumpPath, +LmdbPath)
%  Parse a categorylinks.sql.gz dump and write subcat edges (cl_from →
%  cl_target_id, both int32 LE) to an LMDB database at LmdbPath.
process_dump(DumpPath, LmdbPath) :-
    make_directory_if_missing(LmdbPath),
    streaming_glue:run_streaming_pipeline(
        parse_mysql_rows/2,
        ingest_to_lmdb/3,
        [producer_args([DumpPath]),
         env([
            'UW_LMDB_PATH'    = LmdbPath,
            'UW_FILTER_COL'   = 4,        % cl_type column (0-based)
            'UW_FILTER_VAL'   = subcat,
            'UW_KEY_COL'      = 0,        % cl_from
            'UW_VAL_COL'      = 6,        % cl_target_id
            'UW_KEY_ENCODING' = int32_le,
            'UW_VAL_ENCODING' = int32_le,
            'UW_LMDB_DUPSORT' = 1,        % a category can have many parents
            'UW_BATCH_SIZE'   = 50000
         ])],
        ExitCode
    ),
    (   ExitCode =:= 0
    ->  format(user_error, '[enwiki-ingest] pipeline completed OK~n', [])
    ;   format(user_error, '[enwiki-ingest] pipeline exited ~w~n', [ExitCode]),
        halt(ExitCode)
    ).

make_directory_if_missing(Path) :-
    (   exists_directory(Path)
    ->  true
    ;   make_directory_path(Path)
    ).

% ---------------------------------------------------------------------------
% CLI entry point.
% ---------------------------------------------------------------------------

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [DumpPath, LmdbPath]
    ->  process_dump(DumpPath, LmdbPath)
    ;   format(user_error,
               'usage: ... enwiki_category_ingest.pl -- <dump.sql.gz> <lmdb-path>~n',
               []),
        halt(1)
    ).

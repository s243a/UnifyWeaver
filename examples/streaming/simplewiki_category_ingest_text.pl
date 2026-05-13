% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% examples/streaming/simplewiki_category_ingest_text.pl
%
% Text-keyed ingest example.  Identical in producer/consumer shape to
% enwiki_category_ingest.pl, but pulls strings (rather than MediaWiki
% page_ids) into the LMDB and interns them on the fly.  Designed for
% the SimpleWiki-derived `100k_cats` / `50k_cats` benchmark fixtures
% where the canonical identifier is the category name itself.
%
% After ingest the LMDB contains four sub-databases:
%
%   category_parent (DUPSORT)  child_id  -> parent_id   (int32_le)
%   s2i                        atom_str  -> int32_le    (forward intern)
%   i2s                        int32_le  -> atom_str    (reverse intern)
%   meta                       schema_version, next_id, ...
%
% The s2i / i2s / meta sub-dbs let the Haskell runtime open the LMDB
% and skip TSV parsing + atom-table rebuild on warm runs.  See
% docs/design/WAM_LMDB_RESIDENT_INTERNING_*.md for the design.
%
% Usage:
%   swipl -q -g "process_dump('.../categorylinks.sql.gz', './cats.lmdb')" \
%         -t halt examples/streaming/simplewiki_category_ingest_text.pl
%
% Or, for a Prolog top-level:
%   ?- [examples/streaming/simplewiki_category_ingest_text].
%   ?- process_dump('path/to/dump.sql.gz', 'path/to/output.lmdb').

:- use_module('../../src/unifyweaver/core/target_mapping').
:- use_module('../../src/unifyweaver/glue/streaming_glue').

% ---------------------------------------------------------------------------
% Producer (Rust) — same as enwiki_category_ingest.pl.  Benchmarks pin
% the parser to Rust so preprocessing cost is constant across query
% targets.  Swap to AWK or a future Haskell variant by changing only
% this declare_target.
% ---------------------------------------------------------------------------

:- declare_target(parse_mysql_rows/2, rust,
                  [leaf(true),
                   native_crate(mysql_stream)]).

% ---------------------------------------------------------------------------
% Consumer (Python) — extended ingest_to_lmdb with text-keyed intern
% mode.  When/if a Rust-native LMDB sink is wanted, the swap is a
% one-line declare_target change to {rust, native_crate(lmdb_sink)} —
% the env-var contract is the same.
% ---------------------------------------------------------------------------

:- declare_target(ingest_to_lmdb/3, python,
                  [leaf(true),
                   script_path('src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py'),
                   python_min_version('3.9'),
                   pip_packages([lmdb])]).

%% process_dump(+DumpPath, +LmdbPath)
%  Parse a categorylinks.sql.gz dump and write subcat edges (cl_from →
%  cl_to, both interned to int32_le) plus the s2i/i2s intern table to
%  an LMDB at LmdbPath.
process_dump(DumpPath, LmdbPath) :-
    process_dump(DumpPath, LmdbPath, []).

%% process_dump(+DumpPath, +LmdbPath, +Options)
%  Options:
%    compile_time_atoms(Path) — sidecar file aligning low IDs with the
%                               codegen's compileTimeAtomTable.
%    force_reingest(Bool)     — overwrite an existing populated LMDB.
%    source_sha(Sha)          — record meta.source_dump_sha256 verbatim.
process_dump(DumpPath, LmdbPath, Options) :-
    make_directory_if_missing(LmdbPath),
    % cl_from is column 0 (numeric in MediaWiki, but we read its TSV
    % representation as a string and intern it — see SPECIFICATION §7.1
    % regime "Text-keyed").  cl_to (column 2) is the parent category
    % name as a string.  cl_type (column 4) filters to subcat rows.
    BaseEnv = [
        'UW_LMDB_PATH'      = LmdbPath,
        'UW_LMDB_DBNAME'    = 'category_parent',
        'UW_LMDB_DUPSORT'   = 1,
        'UW_FILTER_COL'     = 4,
        'UW_FILTER_VAL'     = subcat,
        'UW_KEY_COL'        = 0,
        'UW_VAL_COL'        = 2,
        'UW_INTERN_KEY'     = 1,
        'UW_INTERN_VAL'     = 1,
        'UW_LMDB_S2I_DB'    = s2i,
        'UW_LMDB_I2S_DB'    = i2s,
        'UW_LMDB_META_DB'   = meta,
        'UW_LMDB_APPEND'    = 0,
        'UW_SCHEMA_VERSION' = 1,
        'UW_BATCH_SIZE'     = 50000
    ],
    options_to_env(Options, ExtraEnv),
    append(BaseEnv, ExtraEnv, Env),
    streaming_glue:run_streaming_pipeline(
        parse_mysql_rows/2,
        ingest_to_lmdb/3,
        [producer_args([DumpPath]),
         env(Env)],
        ExitCode
    ),
    (   ExitCode =:= 0
    ->  format(user_error, '[simplewiki-text-ingest] pipeline completed OK~n', [])
    ;   format(user_error, '[simplewiki-text-ingest] pipeline exited ~w~n', [ExitCode]),
        halt(ExitCode)
    ).

%% options_to_env(+Options, -ExtraEnv)
%  Translate Prolog options into the consumer's env-var surface.
options_to_env([], []).
options_to_env([compile_time_atoms(Path) | Rest], [
    'UW_COMPILE_TIME_ATOMS' = Path | More
]) :- !,
    options_to_env(Rest, More).
options_to_env([force_reingest(true) | Rest], [
    'UW_FORCE_REINGEST' = 1 | More
]) :- !,
    options_to_env(Rest, More).
options_to_env([source_sha(Sha) | Rest], [
    'UW_SOURCE_SHA' = Sha | More
]) :- !,
    options_to_env(Rest, More).
options_to_env([_ | Rest], More) :-
    options_to_env(Rest, More).

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
    (   Argv = [DumpPath, LmdbPath | _]
    ->  process_dump(DumpPath, LmdbPath)
    ;   format(user_error,
               'usage: ... simplewiki_category_ingest_text.pl -- <dump.sql.gz> <lmdb-path>~n',
               []),
        halt(1)
    ).

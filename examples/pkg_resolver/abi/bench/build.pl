:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build.pl -- generate a minimal C++ WAM project whose ONLY purpose is to
% materialise the store-backed seek runtime (wam_runtime.h / wam_runtime.cpp)
% for the ABI symprov/2 store, so bench_main.cpp can drive the C++
% SeekFactSource read path directly.
%
% The trivial program is:
%     sym_lookup(K, V) :- symprov(K, V).
% with symprov/2 declared as a D43 store fact source. sym_lookup/2 is not used
% by the benchmark (bench_main.cpp calls wam_cpp::SeekFactSource::rows()
% directly) but it makes the generated project a well-formed lookup program and
% pins symprov/2 as a fact-source predicate so the seek runtime is emitted.
%
% Declaration-level backend switch (4th argv, mirrors cpp_store/build.pl):
%   indexed  -- source(symprov/2, indexed(Prefix))  Prefix.data + Prefix.idx
%   lmdb     -- source(symprov/2, lmdb(Dir))         data.mdb + lock.mdb
% The lmdb build auto-#defines WAM_CPP_ENABLE_LMDB in the generated header
% (compiled with -llmdb); the L1 direct-mapped + L2 FIFO caches live in that
% code path. The indexed read path (positioned seeks, OS page cache only) is
% always compiled.

:- use_module('../../../../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).

backend_kind(Kind) :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_, _, Raw|_]
    ->  downcase_atom(Raw, Kind0)
    ;   Kind0 = indexed
    ),
    (   memberchk(Kind0, [indexed, lmdb])
    ->  Kind = Kind0
    ;   format(user_error, "bench/build.pl: unknown backend ~w (indexed|lmdb)~n",
               [Kind0]),
        halt(2)
    ).

store_source(indexed, Path, source(symprov/2, indexed(Path))).
store_source(lmdb, Path, source(symprov/2, lmdb(Path))).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutDir, StorePath|_]
    ->  true
    ;   format(user_error,
               "usage: swipl -g main -t halt build.pl -- OUTDIR STOREPATH [indexed|lmdb]~n", []),
        halt(2)
    ),
    backend_kind(Kind),
    % sym_lookup/2 as a real (thin) clause; symprov/2 as a fact source.
    assertz((user:sym_lookup(K, V) :- user:symprov(K, V))),
    store_source(Kind, StorePath, Source),
    Preds = [user:sym_lookup/2, user:symprov/2],
    format("bench/build.pl: backend=~w store=~w out=~w~n", [Kind, StorePath, OutDir]),
    write_wam_cpp_project(Preds,
        [ module_name('uw-abi-bench'),
          emit_mode(interpreter),
          emit_main(false),
          cpp_wam_fact_sources([Source]) ],
        OutDir),
    format("bench/build.pl: wrote C++ WAM project under ~w/cpp/~n", [OutDir]).

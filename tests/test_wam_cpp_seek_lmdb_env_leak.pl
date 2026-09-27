:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_cpp_seek_lmdb_env_leak.pl
%
% P2 regression (Astra, PR #4259): SeekFactSource leaked its MDB_env.
%
% SeekFactSource::ensure_open_lmdb() opens a raw MDB_env* (lmdb_env_) that owns
% OS file descriptors (data.mdb + lock.mdb), but the class had NO destructor —
% opening+destroying sources leaked ~2 FDs each (Astra measured 6 -> 46 over 20
% sources). The fix adds a RAII destructor (mdb_env_close) plus env cleanup on
% the partial-open failure paths, and spells out the deleted copy/move ops.
%
% This test compiles a tiny C++ harness against the generated LMDB-enabled
% runtime header, builds a real directory-format LMDB env with liblmdb, then
% opens+destroys 30 SeekFactSource("lmdb", dir) instances and asserts the
% process FD count (entries in /proc/self/fd) does NOT grow. Pre-fix the count
% grew by ~2*30; post-fix it is flat.
%
% Skipped automatically when no C++ compiler or linkable liblmdb is available.
%
%   swipl -q -g run_tests -t halt tests/test_wam_cpp_seek_lmdb_env_leak.pl

:- module(test_wam_cpp_seek_lmdb_env_leak,
          [test_wam_cpp_seek_lmdb_env_leak/0]).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [compile_wam_runtime_header_to_cpp/2]).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ).
cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% liblmdb is usable iff a trivial program that pulls in <lmdb.h> and calls
% mdb_version links with -llmdb.
lmdb_linkable :-
    cpp_compiler(CC),
    tmp_file(lmdb_probe, Base),
    atom_concat(Base, '.cpp', Src),
    atom_concat(Base, '.bin', Bin),
    setup_call_cleanup(
        open(Src, write, S),
        ( write(S, '#include <lmdb.h>\n'),
          write(S, 'int main(){ int a,b,c; mdb_version(&a,&b,&c); return 0; }\n') ),
        close(S)),
    format(atom(Cmd), '~w -std=c++17 ~w -llmdb -o ~w 2>/dev/null', [CC, Src, Bin]),
    catch(( process_create(path(sh), ['-c', Cmd],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

tooling_ready :- cpp_compiler(_), lmdb_linkable.

% ---------------------------------------------------------------------
% Harness sources
% ---------------------------------------------------------------------

harness_source(Src) :-
    Src = "#include \"wam_runtime.h\"\n\c
#include <lmdb.h>\n\c
#include <cstdio>\n\c
#include <string>\n\c
#include <optional>\n\c
#include <dirent.h>\n\c
#include <sys/stat.h>\n\c
\n\c
static int count_fds() {\n\c
    int n = 0;\n\c
    DIR* d = opendir(\"/proc/self/fd\");\n\c
    if (!d) return -1;\n\c
    struct dirent* e;\n\c
    while ((e = readdir(d)) != nullptr) { if (e->d_name[0] != '.') ++n; }\n\c
    closedir(d);\n\c
    return n;\n\c
}\n\c
\n\c
// A valid directory-format LMDB env with one record so the unnamed DB exists.\n\c
static bool make_env(const char* dir) {\n\c
    mkdir(dir, 0775);\n\c
    MDB_env* env = nullptr;\n\c
    if (mdb_env_create(&env) != MDB_SUCCESS) return false;\n\c
    mdb_env_set_mapsize(env, (size_t)1 << 20);\n\c
    if (mdb_env_open(env, dir, 0, 0664) != MDB_SUCCESS) { mdb_env_close(env); return false; }\n\c
    MDB_txn* txn = nullptr;\n\c
    if (mdb_txn_begin(env, nullptr, 0, &txn) != MDB_SUCCESS) { mdb_env_close(env); return false; }\n\c
    MDB_dbi dbi;\n\c
    if (mdb_dbi_open(txn, nullptr, 0, &dbi) != MDB_SUCCESS) { mdb_txn_abort(txn); mdb_env_close(env); return false; }\n\c
    MDB_val k, v;\n\c
    char kb[2] = {0, 0}; char vb[2] = {0, 0};\n\c
    k.mv_size = 2; k.mv_data = kb; v.mv_size = 2; v.mv_data = vb;\n\c
    mdb_put(txn, dbi, &k, &v, 0);\n\c
    mdb_txn_commit(txn);\n\c
    mdb_env_close(env);\n\c
    return true;\n\c
}\n\c
\n\c
int main(int argc, char** argv) {\n\c
    const char* dir = argc > 1 ? argv[1] : \"/tmp/p2_lmdb_env\";\n\c
    if (!make_env(dir)) { std::fprintf(stderr, \"make_env failed\\n\"); return 2; }\n\c
    const int N = 30;\n\c
    int before = count_fds();\n\c
    for (int i = 0; i < N; ++i) {\n\c
        wam_cpp::SeekFactSource src(\"lmdb\", dir);\n\c
        try { src.rows(std::optional<std::string>(std::string(\"k\"))); }\n\c
        catch (const std::exception& e) { std::fprintf(stderr, \"rows threw: %s\\n\", e.what()); }\n\c
    }\n\c
    int after = count_fds();\n\c
    std::fprintf(stderr, \"fd before=%d after=%d (opened+destroyed %d lmdb sources)\\n\", before, after, N);\n\c
    // RAII destructor closes each env at scope exit -> flat count. A small slack\n\c
    // absorbs a one-off handle; a leak would add ~2*N descriptors.\n\c
    return (after <= before + 3) ? 0 : 1;\n\c
}\n".

% ---------------------------------------------------------------------
% Build + run
% ---------------------------------------------------------------------

build_harness(Dir, Bin) :-
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory(Dir),
    % LMDB-enabled runtime header (auto-#defines WAM_CPP_ENABLE_LMDB + pulls in
    % <lmdb.h>); the declared source is a dummy, only the gate matters here.
    compile_wam_runtime_header_to_cpp(
        [cpp_fact_sources([source(edge/2, lmdb('/tmp/wam_cpp_p2_seed'))])], Header),
    directory_file_path(Dir, 'wam_runtime.h', HdrPath),
    setup_call_cleanup(open(HdrPath, write, HS), write(HS, Header), close(HS)),
    harness_source(Src),
    directory_file_path(Dir, 'harness.cpp', SrcPath),
    setup_call_cleanup(open(SrcPath, write, SS), write(SS, Src), close(SS)),
    cpp_compiler(CC),
    directory_file_path(Dir, 'leakbin', Bin),
    format(atom(Cmd),
        '~w -std=c++17 -O0 -I~w ~w -llmdb -o ~w 2>&1',
        [CC, Dir, SrcPath, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, BuildOut), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, '~n[seek lmdb leak build failed]~n~w~n', [BuildOut]),
        throw(cpp_seek_lmdb_build_failed(Status))
    ).

test_wam_cpp_seek_lmdb_env_leak :-
    run_tests(wam_cpp_seek_lmdb_env_leak).

:- begin_tests(wam_cpp_seek_lmdb_env_leak, [condition(tooling_ready)]).

test(seek_fact_source_does_not_leak_fds) :-
    Dir = 'output/test_wam_cpp_seek_lmdb_env_leak',
    build_harness(Dir, Bin),
    directory_file_path(Dir, 'env', EnvDir),
    process_create(Bin, [EnvDir],
                   [stdout(null), stderr(pipe(E)), process(Pid)]),
    read_string(E, _, ErrOut), close(E),
    process_wait(Pid, Status),
    format(user_error, '~w', [ErrOut]),
    assertion(Status == exit(0)),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_cpp_seek_lmdb_env_leak).

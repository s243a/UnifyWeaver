:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_lmdb_runtime_smoke.pl
%
% Runtime smoke test for the Scala LmdbFactSource adaptor introduced
% in PR #1804. Exercises the adaptor end-to-end: generates a Scala
% project with an `lmdb(...)` fact source, compiles with lmdbjava on
% the classpath, populates a temp LMDB env, runs a query, and
% verifies the result.
%
% The emit-and-grep tests in test_wam_scala_generator.pl cover
% codegen regressions; this test catches protocol-level regressions
% (method signatures, ByteBuffer encoding, dupsort cursor walking).
%
% Triple gating - the test is skipped unless ALL of:
%   1. SCALA_LMDB_TESTS=1                         (opt in)
%   2. scalac and scala on PATH                   (existing smoke gate)
%   3. LMDBJAVA_CLASSPATH points to a colon-     (driver responsibility:
%      separated list of JARs containing          provide lmdbjava +
%      lmdbjava + its native deps                 transitive deps)
%
% Setup recipe (a one-time setup the developer runs locally):
%   1. Download lmdbjava + jnr-ffi + jffi + asm transitive deps
%      (e.g. via `coursier fetch org.lmdbjava:lmdbjava:0.9.0`).
%   2. export LMDBJAVA_CLASSPATH=/path/to/jar1.jar:/path/to/jar2.jar:...
%   3. export SCALA_LMDB_TESTS=1
%   4. swipl -g 'use_module(library(plunit)),consult("tests/test_wam_scala_lmdb_runtime_smoke.pl"),run_tests,halt' -t 'halt(1)'
%
% Expected to be skipped in default CI; the developer running it
% locally validates the protocol contract before benchmarking.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

:- dynamic user:wam_pair_lmdb/2.
:- dynamic user:wam_triple_lmdb/3.

% Body is irrelevant - the predicate is replaced by a CallForeign stub
% that delegates to the LmdbFactSource. Listed here so write_wam_scala_project
% finds it.
user:wam_pair_lmdb(_, _).
user:wam_triple_lmdb(_, _, _).

% LMDB-backed graph kernel: transitive closure whose edge relation is an
% LMDB fact source. With kernel_dispatch(true) the kernel enumerates the
% edges through collectBinarySolutions -> the LMDB streamAll cursor scan
% (the path the key-lookup tests above never exercise).
:- dynamic user:wam_edge_lmdb/2.
:- dynamic user:wam_tc_lmdb/2.
user:wam_edge_lmdb(_, _).
user:wam_tc_lmdb(X, Y) :- wam_edge_lmdb(X, Y).
user:wam_tc_lmdb(X, Y) :- wam_edge_lmdb(X, Z), wam_tc_lmdb(Z, Y).

% ============================================================
% Gating
% ============================================================

%% scala_lmdb_available/0 - true if all preconditions hold.
%  Opt-in via SCALA_LMDB_TESTS being set (to any value); requires scalac on
%  PATH and LMDBJAVA_CLASSPATH pointing at the lmdbjava JARs. Note getenv/2
%  unifies the value as an *atom*, so we must not compare it to the string
%  "1" (that never matches) — presence of the variable is the opt-in signal.
scala_lmdb_available :-
    getenv('SCALA_LMDB_TESTS', _),
    scalac_on_path,
    getenv('LMDBJAVA_CLASSPATH', _).

scalac_on_path :-
    catch(
        process_create(path(scalac), ['--version'],
                       [stdout(null), stderr(null), process(Pid)]),
        _,
        fail
    ),
    process_wait(Pid, exit(0)).

% ============================================================
% Tests
% ============================================================

:- begin_tests(wam_scala_lmdb_runtime_smoke,
               [ condition(scala_lmdb_available) ]).

% Single end-to-end smoke: seed an LMDB env with three (key, value)
% pairs, compile a Scala project that reads via LmdbFactSource,
% query it, verify the output matches the seeded data.
%
% The seeder is a small Scala main compiled alongside the WAM
% project. It uses lmdbjava DIRECTLY (no FactSource indirection) so
% the seeder's own bytes-on-disk shape matches what the FactSource
% will read back.
test(lmdb_fact_source_end_to_end) :-
    once((
        unique_lmdb_tmp_dir('tmp_scala_lmdb_smoke', TmpDir),
        directory_file_path(TmpDir, 'lmdb_env', LmdbEnv),
        make_directory_path(LmdbEnv),
        absolute_file_name(LmdbEnv, AbsLmdbEnv),
        % Generate the WAM project with an LMDB FactSource pointing at
        % AbsLmdbEnv. dupsort=false (one value per key in this test).
        write_wam_scala_project(
            [user:wam_pair_lmdb/2],
            [package('generated.wam_scala_lmdb_smoke.core'),
             runtime_package('generated.wam_scala_lmdb_smoke.runtime'),
             scala_fact_sources([
                source(wam_pair_lmdb/2,
                       lmdb([env_path(AbsLmdbEnv),
                             dbi(''),
                             dupsort(false)]))
             ])],
            TmpDir),
        % Drop the seeder source alongside the generated sources so
        % scalac sees them all as one compilation unit.
        write_lmdb_seeder_source(TmpDir,
            "    put(\"alpha\",   \"bravo\")\n    put(\"charlie\", \"delta\")\n    put(\"echo\",    \"foxtrot\")\n"),
        compile_lmdb_project(TmpDir),
        seed_lmdb_env(TmpDir, AbsLmdbEnv),
        % Query: wam_pair_lmdb(alpha, X) should return true (alpha is
        % a seeded key whose value is bravo). wam_pair_lmdb(missing, _)
        % should return false.
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['alpha', 'bravo'], "true"),
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['alpha', 'wrong'], "false"),
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['missing', 'bravo'], "false"),
        delete_directory_and_contents(TmpDir)
    )).

% Arity-3 end-to-end: the LMDB value holds args 2..N tab-joined, which
% LmdbFactSource splits back out into registers 2..N. Seed two rows
% (k1 -> a<TAB>b, k2 -> c<TAB>d) and verify triples bind correctly.
test(lmdb_arity3_fact_source) :-
    once((
        unique_lmdb_tmp_dir('tmp_scala_lmdb3', TmpDir),
        directory_file_path(TmpDir, 'lmdb_env', LmdbEnv),
        make_directory_path(LmdbEnv),
        absolute_file_name(LmdbEnv, AbsLmdbEnv),
        write_wam_scala_project(
            [user:wam_triple_lmdb/3],
            [package('generated.wam_scala_lmdb_smoke.core'),
             runtime_package('generated.wam_scala_lmdb_smoke.runtime'),
             scala_fact_sources([
                source(wam_triple_lmdb/3,
                       lmdb([env_path(AbsLmdbEnv),
                             dbi(''),
                             dupsort(false)]))
             ])],
            TmpDir),
        % Tab (\t) separates arg2 and arg3 in the stored value.
        write_lmdb_seeder_source(TmpDir,
            "    put(\"k1\", \"a\\tb\")\n    put(\"k2\", \"c\\td\")\n"),
        compile_lmdb_project(TmpDir),
        seed_lmdb_env(TmpDir, AbsLmdbEnv),
        verify_lmdb(TmpDir, 'wam_triple_lmdb/3', ['k1', 'a', 'b'], "true"),
        verify_lmdb(TmpDir, 'wam_triple_lmdb/3', ['k1', 'a', 'x'], "false"),
        verify_lmdb(TmpDir, 'wam_triple_lmdb/3', ['k2', 'c', 'd'], "true"),
        verify_lmdb(TmpDir, 'wam_triple_lmdb/3', ['nope', 'a', 'b'], "false"),
        delete_directory_and_contents(TmpDir)
    )).

% LMDB-backed graph kernel: tc/2 (transitive_closure2) over an edge
% relation stored in LMDB. The native BFS kernel reads its adjacency by
% enumerating the edges via streamAll (cursor scan) — exercising the
% full-scan path that the ground-key tests above do not. Chain
% a -> b -> c -> d -> e seeded into LMDB; verify reachability.
test(lmdb_backed_kernel) :-
    once((
        unique_lmdb_tmp_dir('tmp_scala_lmdb_kn', TmpDir),
        directory_file_path(TmpDir, 'lmdb_env', LmdbEnv),
        make_directory_path(LmdbEnv),
        absolute_file_name(LmdbEnv, AbsLmdbEnv),
        write_wam_scala_project(
            [user:wam_tc_lmdb/2, user:wam_edge_lmdb/2],
            [package('generated.wam_scala_lmdb_smoke.core'),
             runtime_package('generated.wam_scala_lmdb_smoke.runtime'),
             kernel_dispatch(true),
             scala_fact_sources([
                source(wam_edge_lmdb/2,
                       lmdb([env_path(AbsLmdbEnv),
                             dbi(''),
                             dupsort(false)]))
             ])],
            TmpDir),
        write_lmdb_seeder_source(TmpDir,
            "    put(\"a\", \"b\")\n    put(\"b\", \"c\")\n    put(\"c\", \"d\")\n    put(\"d\", \"e\")\n"),
        compile_lmdb_project(TmpDir),
        seed_lmdb_env(TmpDir, AbsLmdbEnv),
        verify_lmdb(TmpDir, 'wam_tc_lmdb/2', ['a', 'b'], "true"),
        verify_lmdb(TmpDir, 'wam_tc_lmdb/2', ['a', 'e'], "true"),
        verify_lmdb(TmpDir, 'wam_tc_lmdb/2', ['c', 'e'], "true"),
        verify_lmdb(TmpDir, 'wam_tc_lmdb/2', ['e', 'a'], "false"),
        verify_lmdb(TmpDir, 'wam_tc_lmdb/2', ['a', 'x'], "false"),
        delete_directory_and_contents(TmpDir)
    )).

:- end_tests(wam_scala_lmdb_runtime_smoke).

% ============================================================
% Helpers
% ============================================================

unique_lmdb_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]),
    make_directory_path(TmpDir).

%% write_lmdb_seeder_source(+ProjectDir, +Puts)
%  Drops a small Scala main that uses lmdbjava DIRECTLY to populate a
%  test LMDB env. Compiled alongside the generated WAM runtime so the
%  same JVM classpath is used for both seed and read. `Puts` is a Scala
%  source fragment of `put("k", "v")` statements (the value may embed a
%  tab `\t` to encode args 2..N for arity > 2).
write_lmdb_seeder_source(ProjectDir, Puts) :-
    directory_file_path(ProjectDir,
        'src/main/scala/generated/wam_scala_lmdb_smoke/core/LmdbSeeder.scala',
        Path),
    file_directory_name(Path, Dir),
    make_directory_path(Dir),
    SeederHead = "package generated.wam_scala_lmdb_smoke.core\n\nimport java.io.File\nimport java.nio.ByteBuffer\nimport java.nio.charset.StandardCharsets\n\n// Standalone seeder - uses lmdbjava directly (no FactSource\n// indirection) to populate the default (unnamed) LMDB database via\n// auto-txn Dbi.put(key, value). For arity > 2 the value embeds tab\n// separators. Map size matches LmdbFactSource's default.\nobject LmdbSeeder {\n  def main(args: Array[String]): Unit = {\n    val envPath = args(0)\n    val envDir = new File(envPath)\n    envDir.mkdirs()\n    val envClass = Class.forName(\"org.lmdbjava.Env\")\n    val builder = envClass.getMethod(\"create\").invoke(null)\n    builder.getClass.getMethod(\"setMapSize\", java.lang.Long.TYPE)\n           .invoke(builder, java.lang.Long.valueOf(1L << 30))\n    val envFlagsClass = Class.forName(\"org.lmdbjava.EnvFlags\")\n    val emptyEnvFlags = java.lang.reflect.Array.newInstance(envFlagsClass, 0)\n    val openMethod = builder.getClass.getMethod(\"open\", classOf[File], emptyEnvFlags.getClass)\n    val env = openMethod.invoke(builder, envDir, emptyEnvFlags)\n    val dbiFlagsClass = Class.forName(\"org.lmdbjava.DbiFlags\")\n    val emptyDbiFlags = java.lang.reflect.Array.newInstance(dbiFlagsClass, 0)\n    val openDbi = envClass.getMethod(\"openDbi\", classOf[String], emptyDbiFlags.getClass)\n    val dbi = openDbi.invoke(env, null, emptyDbiFlags)\n    val putMethod = dbi.getClass.getMethod(\"put\", classOf[Object], classOf[Object])\n    def put(k: String, v: String): Unit = { putMethod.invoke(dbi, utf8(k), utf8(v)); () }\n",
    SeederTail = "    val envClose = envClass.getMethod(\"close\")\n    envClose.invoke(env)\n  }\n  private def utf8(s: String): ByteBuffer = {\n    val bytes = s.getBytes(StandardCharsets.UTF_8)\n    val buf = ByteBuffer.allocateDirect(bytes.length)\n    buf.put(bytes)\n    buf.flip()\n    buf\n  }\n}\n",
    atomic_list_concat([SeederHead, Puts, SeederTail], SeederSrc),
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, SeederSrc),
        close(Stream)).

%% compile_lmdb_project(+ProjectDir)
%  Same shape as the regular smoke harness's compile_scala_project,
%  but with lmdbjava JARs prepended to scalac's classpath.
compile_lmdb_project(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources_lmdb(AbsProjectDir, Sources),
    Sources \= [],
    getenv('LMDBJAVA_CLASSPATH', LmdbCp),
    process_create(path(scalac),
                   ['-classpath', LmdbCp, '-d', ClassDir | Sources],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, _OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_compile_failed(ExitCode, ErrStr), _))
    ).

find_scala_sources_lmdb(AbsProjectDir, Sources) :-
    directory_file_path(AbsProjectDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF,
              [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F)
        ),
        Sources).

%% seed_lmdb_env(+ProjectDir, +EnvPath)
%  Invokes the LmdbSeeder main against the temp env path. Must run
%  AFTER compile but BEFORE the WAM main is queried, so the data is
%  in place when LmdbFactSource opens the env.
seed_lmdb_env(ProjectDir, EnvPath) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    getenv('LMDBJAVA_CLASSPATH', LmdbCp),
    format(atom(FullCp), '~w:~w', [ClassDir, LmdbCp]),
    process_create(path(scala),
                   ['-J--add-opens=java.base/java.nio=ALL-UNNAMED',
                    '-J--add-exports=java.base/sun.nio.ch=ALL-UNNAMED',
                    '-classpath', FullCp,
                    'generated.wam_scala_lmdb_smoke.core.LmdbSeeder',
                    EnvPath],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, _OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(lmdb_seed_failed(ExitCode, ErrStr), _))
    ).

%% verify_lmdb(+ProjectDir, +PredKey, +Args, +Expected)
%  Same as verify_scala_args from the regular smoke harness, but
%  with lmdbjava JARs on the runtime classpath so LmdbFactSource can
%  resolve its lmdbjava classes via Class.forName.
verify_lmdb(ProjectDir, PredKey, Args, Expected) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    getenv('LMDBJAVA_CLASSPATH', LmdbCp),
    format(atom(FullCp), '~w:~w', [ClassDir, LmdbCp]),
    % lmdbjava's ByteBufferProxy reflects on java.nio.Buffer.address, which
    % JDK 16+ strong-encapsulates; --add-opens grants the access.
    append(['-J--add-opens=java.base/java.nio=ALL-UNNAMED',
                    '-J--add-exports=java.base/sun.nio.ch=ALL-UNNAMED',
            '-classpath', FullCp,
            'generated.wam_scala_lmdb_smoke.core.GeneratedProgram',
            PredStr], ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    normalize_space(string(Actual), OutStr0),
    (   ExitCode =:= 0,
        Actual == Expected
    ->  true
    ;   throw(error(lmdb_run_failed(ExitCode, PredKey, Args,
                                     Expected, Actual, ErrStr), _))
    ).

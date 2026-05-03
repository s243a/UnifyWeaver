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

% Body is irrelevant - the predicate is replaced by a CallForeign stub
% that delegates to the LmdbFactSource. Listed here so write_wam_scala_project
% finds it.
user:wam_pair_lmdb(_, _).

% ============================================================
% Gating
% ============================================================

%% scala_lmdb_available/0 - true if all preconditions hold.
scala_lmdb_available :-
    getenv('SCALA_LMDB_TESTS', '1'),
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
             runtime_package('generated.wam_scala_lmdb_smoke.core'),
             scala_fact_sources([
                source(wam_pair_lmdb/2,
                       lmdb([env_path(AbsLmdbEnv),
                             dbi(''),
                             dupsort(true)]))
             ])],
            TmpDir),
        % Drop the seeder source alongside the generated sources so
        % scalac sees them all as one compilation unit.
        write_lmdb_seeder_source(TmpDir),
        compile_lmdb_project(TmpDir),
        seed_lmdb_env(TmpDir, AbsLmdbEnv),
        % Query: wam_pair_lmdb(alpha, X) should return true (alpha is
        % a seeded key with duplicate values). wam_pair_lmdb(missing, _)
        % should return false.
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['alpha', 'bravo'], "true"),
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['alpha', 'charlie'], "true"),
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['alpha', 'wrong'], "false"),
        verify_lmdb(TmpDir, 'wam_pair_lmdb/2', ['missing', 'bravo'], "false"),
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

%% write_lmdb_seeder_source(+ProjectDir)
%  Drops a small Scala main that uses lmdbjava DIRECTLY to populate a
%  test LMDB env. Compiled alongside the generated WAM runtime so the
%  same JVM classpath is used for both seed and read.
write_lmdb_seeder_source(ProjectDir) :-
    directory_file_path(ProjectDir,
        'src/main/scala/generated/wam_scala_lmdb_smoke/core/LmdbSeeder.scala',
        Path),
    file_directory_name(Path, Dir),
    make_directory_path(Dir),
    SeederSrc = "package generated.wam_scala_lmdb_smoke.core\n\nimport java.io.File\nimport java.nio.ByteBuffer\nimport java.nio.charset.StandardCharsets\n\n// Standalone seeder - uses lmdbjava directly (no FactSource\n// indirection) to populate a small LMDB env with known duplicate\n// (key, value) pairs. Run once before the WAM project queries.\nobject LmdbSeeder {\n  def main(args: Array[String]): Unit = {\n    val envPath = args(0)\n    val envDir = new File(envPath)\n    envDir.mkdirs()\n    val envClass = Class.forName(\"org.lmdbjava.Env\")\n    val builder = envClass.getMethod(\"create\").invoke(null)\n    // setMapSize(1L << 20) - 1 MiB, ample for three rows.\n    val setMapSize = builder.getClass.getMethod(\"setMapSize\", java.lang.Long.TYPE)\n    setMapSize.invoke(builder, java.lang.Long.valueOf(1L << 20))\n    val envFlagsClass = Class.forName(\"org.lmdbjava.EnvFlags\")\n    val envFlagArr = java.lang.reflect.Array.newInstance(envFlagsClass, 0)\n    val openMethod = builder.getClass.getMethod(\"open\", classOf[File], envFlagArr.getClass)\n    val env = openMethod.invoke(builder, envDir, envFlagArr.asInstanceOf[Object])\n    val flagsClass = Class.forName(\"org.lmdbjava.DbiFlags\")\n    val mdbCreate = flagsClass.getField(\"MDB_CREATE\").get(null)\n    val mdbDupsort = flagsClass.getField(\"MDB_DUPSORT\").get(null)\n    val flagArr = java.lang.reflect.Array.newInstance(flagsClass, 2)\n    java.lang.reflect.Array.set(flagArr, 0, mdbCreate)\n    java.lang.reflect.Array.set(flagArr, 1, mdbDupsort)\n    val openDbi = envClass.getMethod(\"openDbi\", classOf[String], flagArr.getClass)\n    val dbi = openDbi.invoke(env, null, flagArr.asInstanceOf[Object])\n    val txnClass = Class.forName(\"org.lmdbjava.Txn\")\n    val putFlagsClass = Class.forName(\"org.lmdbjava.PutFlags\")\n    val putFlagArr = java.lang.reflect.Array.newInstance(putFlagsClass, 0)\n    val txnWrite = envClass.getMethod(\"txnWrite\")\n    val txn = txnWrite.invoke(env)\n    try {\n      val putMethod = dbi.getClass.getMethod(\"put\",\n        txnClass, classOf[Object], classOf[Object], putFlagArr.getClass)\n      def put(k: String, v: String): Unit = {\n        val kb = utf8(k); val vb = utf8(v)\n        putMethod.invoke(dbi, txn, kb, vb, putFlagArr.asInstanceOf[Object])\n      }\n      put(\"alpha\", \"bravo\")\n      put(\"alpha\", \"charlie\")\n      put(\"echo\",  \"foxtrot\")\n      val commit = txn.getClass.getMethod(\"commit\")\n      commit.invoke(txn)\n    } finally {\n      val close = txn.getClass.getMethod(\"close\")\n      close.invoke(txn)\n    }\n    val envClose = envClass.getMethod(\"close\")\n    envClose.invoke(env)\n  }\n  private def utf8(s: String): ByteBuffer = {\n    val bytes = s.getBytes(StandardCharsets.UTF_8)\n    val buf = ByteBuffer.allocateDirect(bytes.length)\n    buf.put(bytes)\n    buf.flip()\n    buf\n  }\n}\n",
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

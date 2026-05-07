:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

:- begin_tests(wam_scala_generator).

:- dynamic user:wam_fact/1.
:- dynamic user:wam_execute_caller/1.
:- dynamic user:wam_call_caller/1.
:- dynamic user:wam_choice_fact/1.
:- dynamic user:wam_choice_caller/1.
:- dynamic user:wam_struct_fact/1.
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_pair_lmdb/2.

user:wam_fact(a).
user:wam_execute_caller(X) :- user:wam_fact(X).
user:wam_call_caller(X)    :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X)  :- user:wam_choice_fact(X).
user:wam_struct_fact(f(a)).
user:wam_use_struct(X)     :- user:wam_struct_fact(X).

% ------------------------------------------------------------------
% Test 1: project layout — all required files are created
% ------------------------------------------------------------------
test(project_layout) :-
    once((
        unique_scala_tmp_dir('tmp_scala_layout', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1, user:wam_execute_caller/1],
            [module_name('wam-scala-layout-test'),
             package('generated.wam_scala_layout.core'),
             runtime_package('generated.wam_scala_layout.runtime')],
            TmpDir),
        directory_file_path(TmpDir, 'build.sbt', BuildSbt),
        directory_file_path(TmpDir, 'project/build.properties', BuildProps),
        assertion(exists_file(BuildSbt)),
        assertion(exists_file(BuildProps)),
        % Scala sources
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_layout/core/GeneratedProgram.scala',
            ProgramPath),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_layout/core/WamRuntime.scala',
            RuntimePath),
        assertion(exists_file(ProgramPath)),
        assertion(exists_file(RuntimePath)),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 2: shared instruction table is emitted
% ------------------------------------------------------------------
test(shared_instruction_table) :-
    once((
        unique_scala_tmp_dir('tmp_scala_instrs', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_instrs.core'),
             runtime_package('generated.wam_scala_instrs.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_instrs/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'val sharedInstructionsRaw: Array[Instruction]')),
        assertion(sub_string(Code, _, _, _, 'GetConstant(Atom(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 3: label map is emitted
% ------------------------------------------------------------------
test(label_map) :-
    once((
        unique_scala_tmp_dir('tmp_scala_labels', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_labels.core'),
             runtime_package('generated.wam_scala_labels.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_labels/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'val sharedLabels: Map[String, Int]')),
        assertion(sub_string(Code, _, _, _, '"wam_fact/1" ->')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 4: predicate wrapper methods are emitted
% ------------------------------------------------------------------
test(predicate_wrappers) :-
    once((
        unique_scala_tmp_dir('tmp_scala_wrappers', TmpDir),
        write_wam_scala_project(
            [user:wam_execute_caller/1, user:wam_fact/1],
            [package('generated.wam_scala_wrappers.core'),
             runtime_package('generated.wam_scala_wrappers.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_wrappers/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'def wamExecuteCaller(')),
        assertion(sub_string(Code, _, _, _, 'def wamFact(')),
        assertion(sub_string(Code, _, _, _, 'WamRuntime.runPredicate(sharedProgram,')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 5: multiple predicates share one instruction array
% ------------------------------------------------------------------
test(multi_predicate_shared_table) :-
    once((
        unique_scala_tmp_dir('tmp_scala_multi', TmpDir),
        write_wam_scala_project(
            [user:wam_execute_caller/1, user:wam_call_caller/1, user:wam_fact/1],
            [package('generated.wam_scala_multi.core'),
             runtime_package('generated.wam_scala_multi.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_multi/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        % Only one instruction array declaration
        aggregate_all(count, sub_string(Code, _, _, _, 'val sharedInstructionsRaw'), N),
        assertion(N =:= 1),
        % All three predicates referenced in dispatch/label map
        assertion(sub_string(Code, _, _, _, '"wam_execute_caller/1" ->')),
        assertion(sub_string(Code, _, _, _, '"wam_call_caller/1" ->')),
        assertion(sub_string(Code, _, _, _, '"wam_fact/1" ->')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 6: choice point instructions appear for multi-clause predicates
% ------------------------------------------------------------------
test(choice_point_instructions) :-
    once((
        unique_scala_tmp_dir('tmp_scala_choice', TmpDir),
        write_wam_scala_project(
            [user:wam_choice_fact/1],
            [package('generated.wam_scala_choice.core'),
             runtime_package('generated.wam_scala_choice.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_choice/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'TryMeElse(')),
        assertion(sub_string(Code, _, _, _, 'TrustMe')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 7: foreign predicate stub emits CallForeign
% ------------------------------------------------------------------
test(foreign_predicate_stub) :-
    once((
        unique_scala_tmp_dir('tmp_scala_foreign', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1, user:wam_execute_caller/1],
            [package('generated.wam_scala_foreign.core'),
             runtime_package('generated.wam_scala_foreign.runtime'),
             foreign_predicates([wam_fact/1])],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_foreign/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'CallForeign("wam_fact", 1)')),
        assertion(\+ sub_string(Code, _, _, _, 'Call("wam_fact", 1)')),
        % The fully WAM-compiled predicate should NOT have a CallForeign
        % for itself — wam_execute_caller should have a Call or CallPc
        assertion(\+ sub_string(Code, _, _, _, 'CallForeign("wam_execute_caller"')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 8: atom intern table is emitted in generated program
% ------------------------------------------------------------------
test(atom_intern_table) :-
    once((
        unique_scala_tmp_dir('tmp_scala_intern', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_intern.core'),
             runtime_package('generated.wam_scala_intern.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_intern/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'val internTable: InternTable')),
        assertion(sub_string(Code, _, _, _, 'InternTable(Array(')),
        % Well-known atoms must appear at fixed positions in the seed array.
        % InternTable's apply de-duplicates, so order = id; codegen emits
        % true=0, fail=1, []=2 first.
        assertion(sub_string(Code, _, _, _, '"true"')),
        assertion(sub_string(Code, _, _, _, '"fail"')),
        assertion(sub_string(Code, _, _, _, '"[]"')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 9: runtime source contains key runtime components
% ------------------------------------------------------------------
test(runtime_source_components) :-
    once((
        unique_scala_tmp_dir('tmp_scala_runtime', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_runtime.core'),
             runtime_package('generated.wam_scala_runtime.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_runtime/core/WamRuntime.scala',
            RuntimePath),
        read_file_to_string(RuntimePath, Code, []),
        assertion(sub_string(Code, _, _, _, 'sealed trait WamTerm')),
        assertion(sub_string(Code, _, _, _, 'final case class Atom(id: Int)')),
        assertion(sub_string(Code, _, _, _, 'sealed trait Instruction')),
        assertion(sub_string(Code, _, _, _, 'final case class WamProgram(')),
        assertion(sub_string(Code, _, _, _, 'final class WamState(')),
        assertion(sub_string(Code, _, _, _, 'object WamRuntime')),
        assertion(sub_string(Code, _, _, _, 'def step(')),
        assertion(sub_string(Code, _, _, _, 'def run(')),
        assertion(sub_string(Code, _, _, _, 'def runPredicate(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Phase S8: LMDB FactSource (memory-mapped fact source)
% Mirrors the Elixir test pattern (test_wam_elixir_target.pl):
% emit-and-grep on the runtime adaptor, indirect class resolution,
% safe key/value API surface.
% ------------------------------------------------------------------
test(lmdb_adaptor_emitted_in_runtime) :-
    once((
        unique_scala_tmp_dir('tmp_scala_lmdb_adaptor', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_lmdb.core'),
             runtime_package('generated.wam_scala_lmdb.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_lmdb/core/WamRuntime.scala',
            RuntimePath),
        read_file_to_string(RuntimePath, Code, []),
        assertion(sub_string(Code, _, _, _, 'final class LmdbFactSource(')),
        % Spec parameters every driver needs.
        assertion(sub_string(Code, _, _, _, 'envPath: String')),
        assertion(sub_string(Code, _, _, _, 'dbName: String')),
        assertion(sub_string(Code, _, _, _, 'dupsort: Boolean')),
        % Three FactSource callbacks (the Elixir behaviour callbacks).
        assertion(sub_string(Code, _, _, _, 'def streamAll(')),
        assertion(sub_string(Code, _, _, _, 'def lookupByArg1(')),
        assertion(sub_string(Code, _, _, _, 'def close(')),
        delete_directory_and_contents(TmpDir)
    )).

test(lmdb_adaptor_uses_indirect_class_resolution) :-
    once((
        unique_scala_tmp_dir('tmp_scala_lmdb_indirect', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_lmdb.core'),
             runtime_package('generated.wam_scala_lmdb.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_lmdb/core/WamRuntime.scala',
            RuntimePath),
        read_file_to_string(RuntimePath, Code, []),
        % Same constraint as the Elixir SQLite/LMDB adaptors: the
        % runtime must NOT have a literal `import org.lmdbjava._` —
        % drivers without lmdbjava on the classpath would fail to
        % compile. Class.forName resolves at runtime instead.
        assertion(sub_string(Code, _, _, _, 'Class.forName("org.lmdbjava.')),
        assertion(\+ sub_string(Code, _, _, _, 'import org.lmdbjava')),
        delete_directory_and_contents(TmpDir)
    )).

test(lmdb_adaptor_targets_safe_keyvalue_api) :-
    once((
        unique_scala_tmp_dir('tmp_scala_lmdb_safeapi', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [package('generated.wam_scala_lmdb.core'),
             runtime_package('generated.wam_scala_lmdb.runtime')],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_lmdb/core/WamRuntime.scala',
            RuntimePath),
        read_file_to_string(RuntimePath, Code, []),
        % Contract per docs/WAM_TARGET_ROADMAP.md: avoid the raw-
        % pointer interface that crashed Haskell. Txn.get + cursor
        % seek/first/next are the safe layer. Reject any pointer-deref
        % shape (mdb_val, getDirectBufferAddress, nativeAddress).
        assertion(sub_string(Code, _, _, _, 'txnRead')),
        assertion(sub_string(Code, _, _, _, 'openCursor')),
        assertion(sub_string(Code, _, _, _, 'MDB_SET')),
        assertion(sub_string(Code, _, _, _, 'MDB_NEXT_DUP')),
        assertion(\+ sub_string(Code, _, _, _, 'MDB_val')),
        assertion(\+ sub_string(Code, _, _, _, 'getDirectBufferAddress')),
        assertion(\+ sub_string(Code, _, _, _, 'nativeAddress')),
        delete_directory_and_contents(TmpDir)
    )).

test(lmdb_fact_source_emits_handler) :-
    once((
        unique_scala_tmp_dir('tmp_scala_lmdb_handler', TmpDir),
        write_wam_scala_project(
            [user:wam_pair_lmdb/2],
            [package('generated.wam_scala_lmdb_h.core'),
             runtime_package('generated.wam_scala_lmdb_h.runtime'),
             scala_fact_sources([
                source(wam_pair_lmdb/2,
                       lmdb([env_path('/tmp/test_lmdb_env'),
                             dbi('test_db'),
                             dupsort(true)]))
             ])],
            TmpDir),
        directory_file_path(TmpDir,
            'src/main/scala/generated/wam_scala_lmdb_h/core/GeneratedProgram.scala',
            ProgramPath),
        read_file_to_string(ProgramPath, Code, []),
        % The spec should produce a thin handler that delegates to
        % LmdbFactSource. Driver-supplied env_path / dbi / dupsort
        % are baked into the handler-instance constructor.
        assertion(sub_string(Code, _, _, _, 'new LmdbFactSource(')),
        assertion(sub_string(Code, _, _, _, 'envPath = "/tmp/test_lmdb_env"')),
        assertion(sub_string(Code, _, _, _, 'dbName  = "test_db"')),
        assertion(sub_string(Code, _, _, _, 'dupsort = true')),
        % Dispatch on arg1 ground / unbound — same shape as the
        % Elixir lookup_by_arg1 vs stream_all callback split.
        assertion(sub_string(Code, _, _, _, 'source.lookupByArg1(')),
        assertion(sub_string(Code, _, _, _, 'source.streamAll(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 10: build.sbt contains module name
% ------------------------------------------------------------------
test(build_sbt_module_name) :-
    once((
        unique_scala_tmp_dir('tmp_scala_build', TmpDir),
        write_wam_scala_project(
            [user:wam_fact/1],
            [module_name('my-wam-test'),
             package('generated.wam_scala_build.core'),
             runtime_package('generated.wam_scala_build.runtime')],
            TmpDir),
        directory_file_path(TmpDir, 'build.sbt', BuildPath),
        read_file_to_string(BuildPath, Code, []),
        assertion(sub_string(Code, _, _, _, 'name := "my-wam-test"')),
        assertion(sub_string(Code, _, _, _, 'scalaVersion := "3.3.1"')),
        delete_directory_and_contents(TmpDir)
    )).

test(escaped_quoted_atom_tokenizes_as_constant) :-
    wam_scala_target:wam_line_to_scala_literal(
        "get_constant 'People\\'s_Republic_of_China' A1",
        Literal),
    assertion(sub_string(Literal, _, _, _, 'GetConstant(Atom(')),
    assertion(\+ sub_string(Literal, _, _, _, 'Raw(')).

test(raw_fallback_escapes_scala_string) :-
    wam_scala_target:wam_parts_to_scala(["unsupported", "People\\", "s"], Literal),
    assertion(sub_string(Literal, _, _, _, 'Raw("unsupported People\\\\ s")')).

test(switch_on_constant_preserves_comma_atoms) :-
    wam_scala_target:wam_line_to_scala_literal(
        "switch_on_constant 'Washington,_D.C. ':L_one plain:L_two",
        Literal),
    assertion(sub_string(Literal, _, _, _, 'SwitchOnConstant(Array(')),
    assertion(\+ sub_string(Literal, _, _, _, 'Raw(')).

test(switch_on_constant_rejoins_space_before_colon) :-
    wam_scala_target:wam_line_to_scala_literal(
        "switch_on_constant 'Academics_from_Washington,_D.C. ' :L_one plain:L_two",
        Literal),
    assertion(sub_string(Literal, _, _, _, 'SwitchOnConstant(Array(')),
    assertion(\+ sub_string(Literal, _, _, _, 'Raw(')).

% ------------------------------------------------------------------
% Helpers
% ------------------------------------------------------------------

unique_scala_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

:- end_tests(wam_scala_generator).

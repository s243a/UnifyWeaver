:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_go_templates.pl - byte-freeze for the Go WAM template refactor.
%
% See docs/proposals/wam_go_template_refactor_design.md (§7.4, §8).
%
% Phase 0 (freeze): these digests are pinned against the CURRENT generator
% predicates BEFORE any refactor. They MUST pass before the atom-embedded Go
% is moved into template files; the whole point of the refactor is to keep
% every one of these bytes identical afterwards.
%
% Phase 2 (split): the helper blob is now eight ordered runtime/*.go.mustache
% partials spliced by the shell. The digests above are unchanged (the eight
% sections re-concatenate byte-for-byte). The wam_go_template_faults block adds
% missing/empty/EOL/tag-contaminated fault fixtures for each new section plus
% shell-marker faults, mirroring tests/test_wam_cpp_templates.pl.
%
% Recorded environment: SWI-Prolog 9.0.4 (x86_64-linux); baselines captured
% from wam_go_target.pl at repo HEAD e1d709070 (code identical to main).
% SHA-256 covers UTF-8 bytes; the string_length values count Prolog chars.

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3, crypto_file_hash/3]).
:- use_module(library(filesex),
              [directory_file_path/3, copy_file/2,
               delete_directory_and_contents/1]).
:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module('../src/unifyweaver/targets/wam_go_target',
              [compile_wam_helpers_to_go/2,
               compile_step_wam_to_go/2,
               compile_wam_runtime_to_go/2,
               write_wam_go_project/3]).
:- use_module('../src/unifyweaver/targets/wam_go_templates',
              [go_render_helper_methods/1,
               go_render_helper_methods_at_root/2,
               go_render_template_at_root/4]).

% -- Frozen digests (design §2.3, §2.4, §2.5) -------------------------------

% Body (A): the helper blob. Ends with exactly one LF.
helpers_digest(61313,
    'dab3b8c24103f54c863b0c89ecc4b1b7cb27cd8ebbb9c29b83dbe9000cd06552').

% Body (B): the assembled Step() method. Ends with '}', no trailing LF.
step_digest(27593,
    '433eea3c43c09a55633885b9978c534a9a0b1b8930a33f4edcf3ba79a031416e').

% Full-project runtime.go / state.go for a package_name(wam) project. Both
% files depend only on package_name (the WAM runtime and VM are predicate
% independent), so a one-fact fixture pins the same bytes as the pkg_resolver
% `go` lane (design §2.5).
project_file_digest('runtime.go', 89244,
    '6b0e707ca7a76f6daad8ce28b96e72a25193344e14eceae3c9c1e25980fdabbd').
project_file_digest('state.go', 146916,
    'f5a9b98f07415b7e6c1a0bd5434fad1aa8336cae180a5734bb364d7e610a97f0').

assert_bytes(Text, Length, Digest) :-
    string_length(Text, Length),
    crypto_data_hash(Text, Actual, [algorithm(sha256), encoding(utf8)]),
    assertion(Actual == Digest).

:- dynamic user:go_tmpl_freeze_fact/1.

with_minimal_go_project(Goal) :-
    tmp_file(wam_go_templates_freeze, Project),
    setup_call_cleanup(
        assertz(user:go_tmpl_freeze_fact(a), Ref),
        (   write_wam_go_project([user:go_tmpl_freeze_fact/1],
                [prefer_wam(true), module_name('uw-go-tmpl-freeze'),
                 package_name(wam)],
                Project),
            call(Goal, Project)),
        (   erase(Ref),
            (   exists_directory(Project)
            ->  delete_directory_and_contents(Project)
            ;   true))).

% Project-file digests are byte counts (design §2.5): size_file gives bytes,
% while string_length would count characters (runtime.go carries multi-byte
% UTF-8 em dashes and ×).
check_project_file(Name, ByteLength, Digest, Project) :-
    directory_file_path(Project, Name, Path),
    size_file(Path, ActualBytes),
    assertion(ActualBytes == ByteLength),
    crypto_file_hash(Path, Actual, [algorithm(sha256), encoding(octet)]),
    assertion(Actual == Digest).

:- begin_tests(wam_go_templates).

test(helpers_exact_bytes) :-
    compile_wam_helpers_to_go([], Helpers),
    helpers_digest(Length, Digest),
    assert_bytes(Helpers, Length, Digest).

test(step_exact_bytes) :-
    compile_step_wam_to_go([], Step),
    step_digest(Length, Digest),
    assert_bytes(Step, Length, Digest).

% compile_wam_runtime_to_go/2 is Step ++ "\n\n" ++ Helpers; three external
% suites grep its content, so pin its shape here too.
test(runtime_to_go_shape) :-
    compile_step_wam_to_go([], Step),
    compile_wam_helpers_to_go([], Helpers),
    compile_wam_runtime_to_go([], Runtime),
    format(atom(Expected), "~w\n\n~w", [Step, Helpers]),
    assertion(Runtime == Expected).

test(helpers_repeated_render) :-
    compile_wam_helpers_to_go([], First),
    compile_wam_helpers_to_go([], Second),
    assertion(First == Second).

test(step_repeated_render) :-
    compile_step_wam_to_go([], First),
    compile_step_wam_to_go([], Second),
    assertion(First == Second).

% Template reads must not depend on the working directory.
test(helpers_foreign_cwd) :-
    setup_call_cleanup(
        working_directory(Here, '/tmp'),
        (   compile_wam_helpers_to_go([], Helpers),
            helpers_digest(Length, Digest),
            assert_bytes(Helpers, Length, Digest)),
        working_directory(_, Here)).

test(project_files_exact_bytes,
     [forall(project_file_digest(Name, Length, Digest))]) :-
    once(with_minimal_go_project(check_project_file(Name, Length, Digest))).

:- end_tests(wam_go_templates).

% -- Phase 2 fault fixtures (design §8 Phase 2 gate) ------------------------
%
% For each of the eight new runtime sections, build a fixture template root,
% copy every real asset, corrupt exactly one, and assert the adapter throws a
% contextual error (never emitting a diagnostic comment as success). Mirrors
% tests/test_wam_cpp_templates.pl's missing/malformed-section fixtures against
% the Go adapter's *_at_root/2,4 entry points and thread-local-free root
% override.

% Adapter section Id -> file name under runtime/. The Id is the first argument
% of every contextual error the adapter throws for that section.
go_runtime_section(runtime_run_loop,           'run_loop.go.mustache').
go_runtime_section(runtime_aggregate,          'aggregate.go.mustache').
go_runtime_section(runtime_foreign_registry,   'foreign_registry.go.mustache').
go_runtime_section(runtime_atom_fact2_sources, 'atom_fact2_sources.go.mustache').
go_runtime_section(runtime_foreign_results,    'foreign_results.go.mustache').
go_runtime_section(runtime_native_kernels,     'native_kernels.go.mustache').
go_runtime_section(runtime_execute_foreign,    'execute_foreign.go.mustache').
go_runtime_section(runtime_seek_fact_source,   'seek_fact_source.go.mustache').

go_real_template_root(RealRoot) :-
    source_file(wam_go_templates:go_render_template(_, _, _), Source),
    file_directory_name(Source, ModuleDir),
    directory_file_path(ModuleDir,
        '../../../templates/targets/go_wam', RealRoot).

write_fixture_file(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
                       write(S, Text),
                       close(S)).

% Build a complete fixture template root (shell + eight sections), applying
% exactly one Fault. Faults:
%   none                       -- untouched copy (control)
%   missing_section(Id)        -- omit that section file
%   corrupt_section(Id, Text)  -- write Text in place of that section
%   shell(Text)                -- write Text in place of runtime.go.mustache
with_go_template_fixture(Fault, Goal) :-
    tmp_file(wam_go_tmpl_fixture, Root),
    setup_call_cleanup(
        build_go_fixture_root(Fault, Root),
        call(Goal, Root),
        (   exists_directory(Root)
        ->  delete_directory_and_contents(Root)
        ;   true)).

build_go_fixture_root(Fault, Root) :-
    make_directory(Root),
    directory_file_path(Root, runtime, RuntimeDir),
    make_directory(RuntimeDir),
    go_real_template_root(RealRoot),
    directory_file_path(RealRoot, 'runtime.go.mustache', RealShell),
    directory_file_path(Root, 'runtime.go.mustache', ShellCopy),
    (   Fault = shell(ShellText)
    ->  write_fixture_file(ShellCopy, ShellText)
    ;   copy_file(RealShell, ShellCopy)
    ),
    directory_file_path(RealRoot, runtime, RealRuntimeDir),
    forall(go_runtime_section(Id, Name),
        (   directory_file_path(RealRuntimeDir, Name, RealSection),
            directory_file_path(RuntimeDir, Name, SectionCopy),
            (   Fault = missing_section(Id)
            ->  true
            ;   Fault = corrupt_section(Id, Text)
            ->  write_fixture_file(SectionCopy, Text)
            ;   copy_file(RealSection, SectionCopy)
            ))).

% Corrupt-a-marker mutation of the real shell (split/rejoin, no regex).
shell_without_marker(Marker, Shell) :-
    go_real_template_root(RealRoot),
    directory_file_path(RealRoot, 'runtime.go.mustache', RealShell),
    read_file_to_string(RealShell, Real, [encoding(utf8)]),
    atomic_list_concat(Parts, Marker, Real),
    atomic_list_concat(Parts, "", Shell).

shell_with_duplicate_marker(Marker, Shell) :-
    go_real_template_root(RealRoot),
    directory_file_path(RealRoot, 'runtime.go.mustache', RealShell),
    read_file_to_string(RealShell, Real, [encoding(utf8)]),
    atomic_list_concat([Real, Marker, "\n"], Shell).

:- begin_tests(wam_go_template_faults).

% Control: an untouched fixture root reproduces the real helper bytes exactly,
% so the fault assertions below are proving the corruption, not the harness.
test(fixture_baseline_matches_real) :-
    go_render_helper_methods(Real),
    with_go_template_fixture(none, check_fixture_helpers(Real)).

test(missing_section_throws_load,
     [forall(go_runtime_section(Id, _))]) :-
    with_go_template_fixture(missing_section(Id),
        expect_helper_error(go_wam_template_load(Id, _, _))).

test(empty_section_throws_empty,
     [forall(go_runtime_section(Id, _))]) :-
    with_go_template_fixture(corrupt_section(Id, ""),
        expect_helper_error(go_wam_template_empty(Id, _))).

test(no_final_lf_section_throws_eol,
     [forall(go_runtime_section(Id, _))]) :-
    with_go_template_fixture(corrupt_section(Id, "func broken() {}"),
        expect_helper_error(go_wam_template_eol(Id, _))).

test(reserved_marker_in_section_throws_tags,
     [forall(go_runtime_section(Id, _))]) :-
    with_go_template_fixture(
        corrupt_section(Id, "func x() {}\n{{step_method}}\n"),
        expect_helper_error(go_wam_template_tags(Id, _))).

test(structural_tag_in_section_throws_tags,
     [forall(go_runtime_section(Id, _))]) :-
    with_go_template_fixture(
        corrupt_section(Id, "func x() {}\n{{match instr}}\n"),
        expect_helper_error(go_wam_template_tags(Id, _))).

% Shell-marker faults: the ordered exactly-once split must reject a shell that
% drops or duplicates a section marker (design §6.3 / §6.4).
test(shell_missing_marker_throws_tags) :-
    shell_without_marker("{{seek_fact_source}}", Shell),
    with_go_template_fixture(shell(Shell),
        expect_shell_error(go_wam_template_tags(runtime_shell, _))).

test(shell_duplicate_marker_throws_tags) :-
    shell_with_duplicate_marker("{{run_loop}}", Shell),
    with_go_template_fixture(shell(Shell),
        expect_shell_error(go_wam_template_tags(runtime_shell, _))).

:- end_tests(wam_go_template_faults).

check_fixture_helpers(Expected, Root) :-
    go_render_helper_methods_at_root(Root, Got),
    assertion(Got == Expected).

expect_helper_error(Expected, Root) :-
    catch(go_render_helper_methods_at_root(Root, _),
          error(Formal, _), Caught = Formal),
    assertion(nonvar(Caught)),
    assertion(subsumes_term(Expected, Caught)).

expect_shell_error(Expected, Root) :-
    catch(go_render_template_at_root(Root, runtime_shell,
              [package_name="wam", step_method="// step"], _),
          error(Formal, _), Caught = Formal),
    assertion(nonvar(Caught)),
    assertion(subsumes_term(Expected, Caught)).

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
               write_wam_go_project/3,
               init_atom_intern_table_go/0]).
% wam_go_target (for the non-exported go_compile_predicate_to_wam/3) and
% wam_go_lowered_emitter (for the non-exported emit_one/2 and the exported
% lower_predicate_to_go/4) are loaded so the Phase 4 slice-0 lowered digests
% below can call them module-qualified.
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter',
              [lower_predicate_to_go/4]).
:- use_module('../src/unifyweaver/targets/wam_go_templates',
              [go_render_helper_methods/1,
               go_render_helper_methods_at_root/2,
               go_render_template_at_root/4,
               go_step_case_order/1,
               go_render_step_method/1,
               go_render_step_method_at_root/2]).

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

% The switch order list has 50 distinct names, and the union of the five
% library files' case names equals it exactly (no missing/duplicate/unknown).
test(case_order_and_set) :-
    go_step_case_order(Order),
    length(Order, 50),
    sort(Order, Sorted),
    assertion(length(Sorted, 50)),
    findall(N, library_case_name(N), FileNames),
    sort(FileNames, FileSorted),
    assertion(FileSorted == Sorted).

:- end_tests(wam_go_templates).

% -- Phase 4 slice 0: freeze the lowered emitter bytes (design §8) ------------
%
% The Go lowered suites (test_wam_go_lowered_phase{1,2,3}.pl, _t{4,5,6}.pl,
% _ite_exec.pl) assert with sub_string/sub_atom only; none pins a digest, so a
% one-byte drift in a lowered fragment would slip through. These tables, modelled
% on the C++ old_head_constant_digest/4, old_head_integer_nil_digest/3 and
% old_lowered_function_digest/3 tables, freeze the exact bytes BEFORE any
% Phase-4 template extraction. Slices 1-3 (function/ITE shells, the head-match
% fragment) must keep every one of these digests identical.
%
% Baselines captured from wam_go_lowered_emitter.pl at the unchanged (pre-slice-1)
% emitter. SHA-256 covers UTF-8 bytes; string_length counts Prolog chars.

% -- Per-fragment digests: with_output_to(string(T), emit_one(Instr, "    ")) --
%
% Each is captured right after init_atom_intern_table_go/0 so the interned var
% names (wamAtom_<sanitized>_<seq>) are stable. The '{{name}}'/'{{Ai}}' atoms are
% the placeholder-shaped rescan traps slice 2 must survive; the escaped atoms
% ('a"b', 'a\b') and the raw get_integer spelling (0007) pin the escape/spelling
% handling the emitter keeps in Prolog.
old_lowered_fragment_digest(get_constant("foo", "A1"), 322,
    '37a3d0cbe1b2d8548e1fa8d34201b126f5ac177574851fe7944c58a84151faaa').
old_lowered_fragment_digest(get_constant("42", "X2"), 331,
    '32c3627f2a5f34fd079dc4b877cde7b6be7878e672c574e7850cbf3822105436').
old_lowered_fragment_digest(get_constant("3.5", "Y3"), 330,
    '322e8128726a99f97c1a47a9aca2d13e79b6f98c94a0197986e601d043cfa06d').
old_lowered_fragment_digest(get_constant("'a\"b'", "A1"), 324,
    '16cea83ae959df720473c7b39be26042d0cb7ed9ef0fc7dfa4cc20198f7d557e').
old_lowered_fragment_digest(get_constant("'a\\b'", "A1"), 324,
    '6a743da38ea10a1ea9f62c6ea4b9336301fa5d02a29d1886d3605cf74abe8899').
old_lowered_fragment_digest(get_constant("'{{name}}'", "A1"), 339,
    '2d205a388772ae2df60c68902c6b515e11bf894753fdc16a06a98ddd947763a2').
old_lowered_fragment_digest(get_constant("'{{Ai}}'", "A1"), 333,
    'cdd761d4e948a4f285c218a1f8932fdf93d0403057c15c7856f149a0dd0c3496').
old_lowered_fragment_digest(get_integer("42", "A1"), 328,
    '1039bc8863dd40313b925c6fe051c520cadbc0ea3ab24dbc731b080dde7105df').
old_lowered_fragment_digest(get_integer("-7", "X2"), 330,
    'b459ec45c2eb89152a00b11b006221430723a345d4bf4fbe49224ecc72d347aa').
old_lowered_fragment_digest(get_integer("0007", "Y3"), 336,
    'bbfb72c04f11e8be33f8cc9608f9ae7aef21af073e0e327b841ccd0ef21b4360').
old_lowered_fragment_digest(get_nil("A1"), 310,
    '8baad2c619989c0cb0af914854acd787e5f9227e0f866197d71fae4be3b6793f').
old_lowered_fragment_digest(get_nil("Y3"), 312,
    '58f1fe600e4dd84927ee7fe83e78f928c478c804e110a688236814329bccb55e').

% -- Whole-function digests: lower_predicate_to_go/4 over fixed shapes ---------
%
% shell_plain is a literal WAM list (as the C++ freeze does); the other six are
% compiled from the fixture predicates below via the SAME entry the lowered path
% uses (go_compile_predicate_to_wam/3, which adds inline_bagof_setof(true) and no
% ite_use_y_level — matching compile_lowered_predicates/3). t6 uses the emitter
% option t6_min_clauses(3) exactly as test_wam_go_lowered_t6.pl does.
%   Spec = wam(List) | compile(user:PI, LowerOptions)
old_lowered_function_digest(shell_plain,
    wam([get_constant("foo", "A1"), proceed]), 435,
    'cb2bf19657b95d227cac00311b241fe32afa6a1782d9c8a5c7d980b5a685b7e8').
old_lowered_function_digest(shell_t4,
    compile(user:grade/2, []), 2698,
    '677b15e94a65e4df8a319ed907957adb01d5959339c1e17f158a17673a3ba539').
old_lowered_function_digest(shell_t5,
    compile(user:color/1, []), 487,
    '8e7f7f8501b856ee46579b32d55d5af7bcc30b5e3ac00e8160cd750d1888059c').
old_lowered_function_digest(shell_t6,
    compile(user:few/1, [t6_min_clauses(3)]), 531,
    'f0aa76906e8dd0ba5b69411e14a1f794f573c2db1bf0891b1a642cdcf25d3509').
old_lowered_function_digest(shell_ite,
    compile(user:gite/2, []), 1627,
    '0e5c5664372503ffb99673d22ec9f38eb5b045dcac511e72bb568ebbdd1ac905').
old_lowered_function_digest(shell_seqite,
    compile(user:gseqite/3, []), 2911,
    'cb8306a68c562cc85037aeaf1ce9c28e6fe62c135b553bd3fa897fe8d2dcd34b').
old_lowered_function_digest(shell_nestedite,
    compile(user:gnestite/2, []), 2780,
    '92c915cf8a4d994c077217ab6169ca6c1a57092b13ef25596fe02d6c8be92f71').

% Fixture predicates for the whole-function digests (mirror the shapes the Go
% lowered t4/t5/t6/ite_exec suites use). Kept here so the freeze is self-contained.
:- dynamic user:grade/2, user:color/1, user:few/1,
           user:gite/2, user:gseqite/3, user:gnestite/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:color(red).
user:color(green).
user:color(blue).

user:few(a). user:few(b). user:few(c).

user:gite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:gseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( Z > 0 -> Y = a ; Y = b ).
user:gnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

lowered_function_code(wam(List), Code) :-
    !,
    init_atom_intern_table_go,
    once(lower_predicate_to_go(shell_plain/1, List, [], Lines)),
    atomic_list_concat(Lines, '\n', Code).
lowered_function_code(compile(Module:PI, LowerOptions), Code) :-
    once(wam_go_target:go_compile_predicate_to_wam(Module:PI, [], Wam)),
    init_atom_intern_table_go,
    once(lower_predicate_to_go(PI, Wam, LowerOptions, Lines)),
    atomic_list_concat(Lines, '\n', Code).

:- begin_tests(wam_go_lowered_freeze).

% Per-fragment: emit_one(Instr, "    ") captured after a fresh intern table.
test(lowered_fragment_bytes,
     [forall(old_lowered_fragment_digest(Instr, Length, Digest))]) :-
    init_atom_intern_table_go,
    with_output_to(string(Text),
        wam_go_lowered_emitter:emit_one(Instr, "    ")),
    assert_bytes(Text, Length, Digest).

% A fragment whose atom is placeholder-shaped ('{{Ai}}') must render without the
% surrounding template machinery ever rescanning it (the slice-2 rescan trap).
test(lowered_fragment_placeholder_atom_literal) :-
    init_atom_intern_table_go,
    with_output_to(string(Text),
        wam_go_lowered_emitter:emit_one(get_constant("'{{Ai}}'", "A1"), "    ")),
    assertion(sub_string(Text, _, _, _, "// get_constant '{{Ai}}', A1")).

% Whole-function: lower_predicate_to_go over each fixed shape.
test(lowered_function_bytes,
     [forall(old_lowered_function_digest(_Name, Spec, Length, Digest))]) :-
    lowered_function_code(Spec, Code),
    assert_bytes(Code, Length, Digest).

% Determinism: the same fixture re-renders byte-identically.
test(lowered_function_repeated_render,
     [forall(old_lowered_function_digest(_Name, Spec, _Length, _Digest))]) :-
    lowered_function_code(Spec, First),
    lowered_function_code(Spec, Second),
    assertion(First == Second).

:- end_tests(wam_go_lowered_freeze).

% Every {{case NAME}} tag across the five real step libraries.
step_library_rel('step/head_unification.go.mustache').
step_library_rel('step/body_construction.go.mustache').
step_library_rel('step/control.go.mustache').
step_library_rel('step/choice_point.go.mustache').
step_library_rel('step/indexing.go.mustache').

library_case_name(Name) :-
    go_real_template_root(Root),
    step_library_rel(Rel),
    directory_file_path(Root, Rel, Path),
    read_file_to_string(Path, Text, [encoding(utf8)]),
    sub_string(Text, Begin, _, _, "{{case "),
    Start is Begin + 7,
    sub_string(Text, Start, _, 0, Tail),
    once(sub_string(Tail, EndRel, 2, _, "}}")),
    sub_string(Tail, 0, EndRel, _, ValStr),
    atom_string(Name, ValStr).

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

% -- Phase 3 step-library fault fixtures (design §8 Phase 3 gate) -----------
%
% Build a fixture root holding the step shell + five step libraries, apply
% exactly one fault, and assert go_render_step_method_at_root/2 throws the
% right contextual error (missing/duplicate/unknown case, missing/empty/
% no-final-LF file, missing shell marker). Mirrors the runtime-section faults.

go_step_library_fixture(head_unification,  'head_unification.go.mustache').
go_step_library_fixture(body_construction, 'body_construction.go.mustache').
go_step_library_fixture(control,           'control.go.mustache').
go_step_library_fixture(choice_point,      'choice_point.go.mustache').
go_step_library_fixture(indexing,          'indexing.go.mustache').

% Faults:
%   none                    -- untouched copy (control)
%   missing_lib(Id)         -- omit that library file
%   corrupt_lib(Id, Text)   -- write Text in place of that library
%   missing_shell           -- omit step_shell.go.mustache
%   shell(Text)             -- write Text in place of step_shell.go.mustache
with_go_step_fixture(Fault, Goal) :-
    tmp_file(wam_go_step_fixture, Root),
    setup_call_cleanup(
        build_go_step_fixture_root(Fault, Root),
        call(Goal, Root),
        (   exists_directory(Root)
        ->  delete_directory_and_contents(Root)
        ;   true)).

build_go_step_fixture_root(Fault, Root) :-
    make_directory(Root),
    directory_file_path(Root, step, StepDir),
    make_directory(StepDir),
    go_real_template_root(RealRoot),
    directory_file_path(RealRoot, step, RealStepDir),
    % step shell
    directory_file_path(RealStepDir, 'step_shell.go.mustache', RealShell),
    directory_file_path(StepDir, 'step_shell.go.mustache', ShellCopy),
    (   Fault = missing_shell
    ->  true
    ;   Fault = shell(ShellText)
    ->  write_fixture_file(ShellCopy, ShellText)
    ;   copy_file(RealShell, ShellCopy)
    ),
    % five libraries
    forall(go_step_library_fixture(Id, Name),
        (   directory_file_path(RealStepDir, Name, RealLib),
            directory_file_path(StepDir, Name, LibCopy),
            (   Fault = missing_lib(Id)
            ->  true
            ;   Fault = corrupt_lib(Id, Text)
            ->  write_fixture_file(LibCopy, Text)
            ;   copy_file(RealLib, LibCopy)
            ))).

% A minimal but well-formed {{match instr}} library over the given case names.
minimal_step_lib(Names, Text) :-
    maplist([N, B]>>format(string(B), "{{case ~w}}\n        return false\n", [N]),
            Names, Blocks),
    atomic_list_concat(Blocks, '', Body),
    format(string(Text), "{{match instr}}\npreamble discarded\n~w{{/match}}\n",
           [Body]).

expect_step_error(Expected, Root) :-
    catch(go_render_step_method_at_root(Root, _),
          error(Formal, _), Caught = Formal),
    assertion(nonvar(Caught)),
    assertion(subsumes_term(Expected, Caught)).

:- begin_tests(wam_go_step_faults).

% Control: an untouched step fixture reproduces the real Step() bytes exactly.
test(step_fixture_baseline_matches_real) :-
    go_render_step_method(Real),
    with_go_step_fixture(none,
        [Root]>>( go_render_step_method_at_root(Root, Got),
                  assertion(Got == Real) )).

test(missing_library_throws_load,
     [forall(go_step_library_fixture(Id, _))]) :-
    with_go_step_fixture(missing_lib(Id),
        expect_step_error(go_wam_template_load(Id, _, _))).

test(empty_library_throws_empty,
     [forall(go_step_library_fixture(Id, _))]) :-
    with_go_step_fixture(corrupt_lib(Id, ""),
        expect_step_error(go_wam_template_empty(Id, _))).

test(no_final_lf_library_throws_eol,
     [forall(go_step_library_fixture(Id, _))]) :-
    with_go_step_fixture(corrupt_lib(Id, "{{match instr}}\n{{case X}}\n y\n{{/match}}"),
        expect_step_error(go_wam_template_eol(Id, _))).

% Drop a case from the indexing library -> the case-set check reports it missing.
test(missing_case_throws_missing) :-
    minimal_step_lib(['SwitchOnConstant','SwitchOnConstantPc','SwitchOnStructure',
                      'SwitchOnStructurePc','SwitchOnConstantA2'], LibText),
    with_go_step_fixture(corrupt_lib(indexing, LibText),
        expect_step_error(go_wam_step_case_missing('SwitchOnConstantA2Pc'))).

% Duplicate a case within a library -> reported as a duplicate.
test(duplicate_case_throws_duplicate) :-
    minimal_step_lib(['SwitchOnConstant','SwitchOnConstant','SwitchOnConstantPc',
                      'SwitchOnStructure','SwitchOnStructurePc',
                      'SwitchOnConstantA2','SwitchOnConstantA2Pc'], LibText),
    with_go_step_fixture(corrupt_lib(indexing, LibText),
        expect_step_error(go_wam_step_case_duplicate('SwitchOnConstant'))).

% All real names present plus one not in the order list -> reported as unknown.
test(unknown_case_throws_unknown) :-
    minimal_step_lib(['SwitchOnConstant','SwitchOnConstantPc','SwitchOnStructure',
                      'SwitchOnStructurePc','SwitchOnConstantA2',
                      'SwitchOnConstantA2Pc','Bogus'], LibText),
    with_go_step_fixture(corrupt_lib(indexing, LibText),
        expect_step_error(go_wam_step_case_unknown('Bogus'))).

test(missing_shell_throws_load) :-
    with_go_step_fixture(missing_shell,
        expect_step_error(go_wam_template_load(step_shell, _, _))).

test(shell_missing_cases_marker_throws_tags) :-
    with_go_step_fixture(
        shell("func (vm *WamState) Step(instr Instruction) bool {\n}\n"),
        expect_step_error(go_wam_template_tags(step_shell, _))).

:- end_tests(wam_go_step_faults).

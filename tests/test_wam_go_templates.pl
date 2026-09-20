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
% Recorded environment: SWI-Prolog 9.0.4 (x86_64-linux); baselines captured
% from wam_go_target.pl at repo HEAD e1d709070 (code identical to main).
% SHA-256 covers UTF-8 bytes; the string_length values count Prolog chars.

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3, crypto_file_hash/3]).
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module('../src/unifyweaver/targets/wam_go_target',
              [compile_wam_helpers_to_go/2,
               compile_step_wam_to_go/2,
               compile_wam_runtime_to_go/2,
               write_wam_go_project/3]).

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

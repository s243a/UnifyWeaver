:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [compile_wam_runtime_header_to_cpp/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_templates',
              [cpp_render_template_at_root/4, cpp_render_stache_at_root/4,
               cpp_lowered_function_lines_at_root/5,
               cpp_render_ite_shell_at_root/7,
               cpp_with_template_root_for_test/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter', []).
:- use_module('../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).

:- dynamic user:shell_t4/2, user:shell_t5/1, user:shell_t6/1, user:shell_ite/2,
           user:shell_seqite/3, user:shell_nestedite/2.
user:shell_t4(a,x). user:shell_t4(b,y). user:shell_t4(a,z).
user:shell_t5(red). user:shell_t5(green). user:shell_t5(blue).
user:shell_t6(s01). user:shell_t6(s02). user:shell_t6(s03). user:shell_t6(s04).
user:shell_t6(s05). user:shell_t6(s06). user:shell_t6(s07). user:shell_t6(s08).
user:shell_ite(X,Y) :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:shell_seqite(X,Y,Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                             ( X > 5 -> Z = big ; Z = small ).
user:shell_nestedite(X,Y) :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

% SHA-256 covers UTF-8 bytes; lengths below count Prolog characters.
% Re-baselined after the D43 store-backed seek FactSource + lazy+cached LMDB
% seek reader were re-applied to the runtime header (pkg_resolver cpp_store
% lane). The pre-extraction baseline was compile_wam_runtime_header_to_cpp/2 at
% local main 1c5ce3682c7047ef9470ee0d5563e3313eccfa24 (plain 57611 / lmdb
% 57852); the SeekFactSource class and its guarded LMDB two-cache backend add
% the byte delta.
%
% Re-baselined AGAIN (PR #4259, Astra's three fixes) from (plain 86911 / lmdb
% 87152): P1 added the prune_iters_to_choice_points() declaration to the header,
% and P2 added SeekFactSource's RAII destructor + close_lmdb() helper + deleted
% copy/move ops (env-leak fix). Both variants grew by the SAME +3108 characters
% (the additions are gate-independent text), which is the whole header delta.
old_header_digest(plain, 90019,
    'd65645f69ff27e69319a67b43c42399b5d7c1067a804f40faa44247fe430c19d').
old_header_digest(lmdb, 90260,
    '1fcfa56c3eca176cd75b7878b2a30710f6c062a0cbcfaf90fbc3fb27422994e2').

assert_old_header_bytes(Mode, Header) :-
    old_header_digest(Mode, Length, Digest),
    string_length(Header, Length),
    crypto_data_hash(Header, Actual, [algorithm(sha256), encoding(utf8)]),
    assertion(Actual == Digest).

:- begin_tests(wam_cpp_templates).

test(plain_header_exact_bytes) :-
    compile_wam_runtime_header_to_cpp([], New),
    assert_old_header_bytes(plain, New).

test(lmdb_header_exact_bytes) :-
    compile_wam_runtime_header_to_cpp(
        [cpp_fact_sources([source(edge/2, lmdb('/tmp/wam_cpp_test_lmdb'))])], New),
    assert_old_header_bytes(lmdb, New).

test(foreign_cwd) :-
    setup_call_cleanup(
        working_directory(Here, '/tmp'),
        (compile_wam_runtime_header_to_cpp([], New),
         assert_old_header_bytes(plain, New)),
        working_directory(_, Here)).

test(repeated_render) :-
    compile_wam_runtime_header_to_cpp([], First),
    compile_wam_runtime_header_to_cpp([], Second),
    assertion(First == Second).

test(main_shim_exact_bytes) :-
    wam_cpp_target:emit_main_shim(Main),
    string_length(Main, 3252),
    crypto_data_hash(Main, Digest, [algorithm(sha256), encoding(utf8)]),
    assertion(Digest == '0c0f30bcf02ebd2adf39e546e02cbdf53970bcd07eb0fd35d7f778be2f95c327').

test(runtime_source_exact_bytes, [forall(member(Options,
     [[], [cpp_fact_sources([source(edge/2, lmdb('/tmp/phase3b_lmdb'))])]]))]) :-
    wam_cpp_target:compile_wam_runtime_to_cpp(Options, Runtime),
    string_length(Runtime, 323458),
    crypto_data_hash(Runtime, Digest, [algorithm(sha256), encoding(utf8)]),
    % Re-baselined after the D43 CallForeign seek dispatch (dispatch_foreign_call
    % / foreign_try_next + the ForeignNextClause case) was re-applied to the
    % runtime source. Still option-invariant (both variants share this digest).
    %
    % Re-baselined AGAIN (PR #4259, Astra's three fixes) from 320342: P1 added
    % prune_iters_to_choice_points() + its calls at every choice_points-shrink
    % cut/drain site + the query() side-stack resets; P3 rewrote number_string/2's
    % overflow-tolerant number parse. The +3116 characters are exactly those two
    % source-side fixes (P2 is header-only). Still option-invariant.
    assertion(Digest == '02d4163853a695550fc7c69d729a1ef33210dd5170b293054c04d8c52e21c880').

test(program_shell_does_not_rescan_fragments) :-
    cpp_render_template_at_root('templates/targets/cpp_wam', generated_program,
        [predicates_code="P {{setup_code}}", setup_code="S {{predicates_code}}"],
        Text),
    assertion(sub_string(Text, _, _, _, "P {{setup_code}}")),
    assertion(sub_string(Text, _, _, _, "S {{predicates_code}}")),
    assertion(\+ sub_string(Text, _, _, _, "P S {{predicates_code}}")),
    assertion(\+ sub_string(Text, _, _, _, "S P {{setup_code}}")).

% Captured before the four lowered function headers/footers shared a shell.
old_lowered_function_digest(shell_plain/1, 382,
    'a0606ef0e168e0822e306a29e0e14973e16d8350adfedf696be279f2b1bfe0ee').
old_lowered_function_digest(shell_t4/2, 2270,
    '2ac4fc8a17f27245cdd071e7d2ef68db107974d819f26c4a8bd6995a0aae80a7').
old_lowered_function_digest(shell_t5/1, 365,
    'fd54ee8777448912e2632849a2d76036a1b069c5e2540bdea3fb8514be3f852f').
old_lowered_function_digest(shell_t6/1, 1151,
    '9326ae27c9eb3df2010e3d83bfe768b2d2641544005ef0b349468940dff9ae92').
old_lowered_function_digest(shell_ite/2, 1393,
    '2c7fc9718f9f2f37fed6b07a6b22449e80e210445e672ceb7a23411fa62aae51').
old_lowered_function_digest(shell_seqite/3, 2498,
    '6c65043391d2892c9fa6c0e5e80a1f8f7dc2fd973a68083ac0be5e2a179d147c').
old_lowered_function_digest(shell_nestedite/2, 2283,
    'ee19e17bc3b261feddbdf66053619ea0484a57974157d19bb7d35f445e21f078').

test(lowered_function_old_bytes,
     [forall(old_lowered_function_digest(PI, Length, Digest))]) :-
    ( PI == shell_plain/1
    -> Wam = [get_constant("foo", "A1"), proceed]
    ;  once(compile_predicate_to_wam(PI,
           [inline_bagof_setof(true), ite_use_y_level(true)], Wam)) ),
    once(wam_cpp_lowered_emitter:lower_predicate_to_cpp(PI, Wam, [], Lines)),
    assertion(Lines = [_, _, _]),
    atomic_list_concat(Lines, '\n', Code),
    string_length(Code, Length),
    crypto_data_hash(Code, Actual, [algorithm(sha256), encoding(utf8)]),
    assertion(Actual == Digest).

test(lowered_function_body_and_metadata_are_literal) :-
    cpp_lowered_function_lines_at_root('templates/targets/cpp_wam',
        "lowered_literal_1", "// {{name}}", "return \"{{comment}} {{name}} {{body}}\";",
        Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_string(Code, _, _, _, "// {{name}}")),
    assertion(sub_string(Code, _, _, _, "return \"{{comment}} {{name}} {{body}}\";")),
    assertion(sub_string(Code, _, _, _, "bool lowered_literal_1(WamState* vm)")).

test(lowered_function_rejects_nonground_name,
     [throws(error(domain_error(cpp_wam_lowered_function_variables, _), _))]) :-
    cpp_lowered_function_lines_at_root('templates/targets/cpp_wam',
        _, "// comment", "return true;", _).

test(lowered_function_asset_errors_survive_warn_policy,
     [forall(member(Problem,
         [cpp_wam_template_load(lowered_function, fixture, missing),
          cpp_wam_template_empty(lowered_function, fixture),
          cpp_wam_template_tags(lowered_function, fixture),
          domain_error(cpp_wam_lowered_function_variables, bad)]))]) :-
    Error = error(Problem, context(lowered_function, fixture)),
    catch(wam_cpp_target:handle_compile_error(warn, shell_t5/1, Error),
          Actual, true),
    assertion(Actual == Error).

test(ite_shell_source_only_splice) :-
    cpp_render_ite_shell_at_root('templates/targets/cpp_wam', "  ", 7,
        "COND {{then}} {{counter}}\n", "THEN {{else}}\n",
        "ELSE {{condition}}\n", Text),
    assertion(sub_string(Text, _, _, _, "COND {{then}} {{counter}}\n")),
    assertion(sub_string(Text, _, _, _, "THEN {{else}}\n")),
    assertion(sub_string(Text, _, _, _, "ELSE {{condition}}\n")),
    assertion(sub_string(Text, _, _, _, "_ite_mark7")),
    assertion(sub_string(Text, _, _, _, "_ite_cond7")).

test(ite_shell_rejects_bad_counter,
     [throws(error(domain_error(cpp_wam_ite_variables, _), _))]) :-
    cpp_render_ite_shell_at_root('templates/targets/cpp_wam', "", 0,
        "", "", "", _).

test(ite_shell_asset_errors_survive_warn_policy,
     [forall(member(Problem,
         [cpp_wam_template_load(lowered_ite, fixture, missing),
          cpp_wam_template_empty(lowered_ite, fixture),
          cpp_wam_template_tags(lowered_ite, fixture),
          domain_error(cpp_wam_ite_variables, bad)]))]) :-
    Error = error(Problem, context(lowered_ite, fixture)),
    catch(wam_cpp_target:handle_compile_error(warn, shell_ite/2, Error),
          Actual, true),
    assertion(Actual == Error).

% Re-baselined after the store-lane re-application: emit_setup_function now also
% emits the foreign_next_clause trailing instruction + vm.foreign_next_clause_pc
% and the (here empty) seek-source registration block, shifting the setup
% section of generated_program.cpp for every variant.
project_variant(functions, [emit_mode(functions), emit_main(true)],
    'ec00ecb77568ab7d0e1105327f4093dadbc57bc2fac1f642afc54c66a9338330').
project_variant(interpreter, [emit_mode(interpreter)],
    '45e030dfd9314ea0dd8d0fd0c938b424db4f0aeaa264e0e4c838eff016c2bc1f').
project_variant(lmdb,
    [emit_mode(functions), emit_main(true),
     cpp_fact_sources([source(phase3_pair/2, lmdb('/tmp/phase3b_lmdb'))])],
    'b12075dec8d51e3aa1dd6a11fe66c6e7915ebbd509c9f788df32737259076ddd').

test(generated_program_old_bytes,
     [forall(project_variant(_, Options, Expected))]) :-
    tmp_file(cpp_wam_program_bytes, Project),
    setup_call_cleanup(
        (assertz(user:phase3_pair(a, '{{setup_code}}'), Ref1),
         assertz(user:phase3_pair(b, '{{predicates_code}}'), Ref2)),
        (wam_cpp_target:write_wam_cpp_project([user:phase3_pair/2], Options, Project),
         directory_file_path(Project, 'cpp/generated_program.cpp', Program),
         read_file_to_string(Program, Text, []),
         crypto_data_hash(Text, Actual, [algorithm(sha256), encoding(utf8)]),
         assertion(Actual == Expected)),
        (erase(Ref1), erase(Ref2),
         (exists_directory(Project) -> delete_directory_and_contents(Project) ; true))).

test(project_preflight_missing_stache_no_files) :-
    with_project_template_fixture(missing_stache, check_project_preflight_failure).

test(project_preflight_malformed_stache_no_files) :-
    with_project_template_fixture(malformed_stache, check_project_preflight_failure).

test(project_preflight_missing_lowered_shell_no_files) :-
    with_project_template_fixture(missing_lowered_shell, check_project_preflight_failure).

test(project_preflight_malformed_lowered_shell_no_files) :-
    with_project_template_fixture(malformed_lowered_shell, check_project_preflight_failure).

test(project_preflight_missing_lowered_slot_no_files) :-
    with_project_template_fixture(missing_lowered_slot, check_project_preflight_failure).

test(project_preflight_duplicate_lowered_slot_no_files) :-
    with_project_template_fixture(duplicate_lowered_slot, check_project_preflight_failure).

test(project_preflight_reordered_lowered_slots_no_files) :-
    with_project_template_fixture(reordered_lowered_slots, check_project_preflight_failure).

test(project_preflight_missing_ite_shell_no_files) :-
    with_project_template_fixture(missing_ite_shell, check_project_preflight_failure).

test(project_preflight_empty_ite_shell_no_files) :-
    with_project_template_fixture(empty_ite_shell, check_project_preflight_failure).

test(project_preflight_unknown_ite_slot_no_files) :-
    with_project_template_fixture(unknown_ite_slot, check_project_preflight_failure).

test(project_preflight_missing_ite_slot_no_files) :-
    with_project_template_fixture(missing_ite_slot, check_project_preflight_failure).

test(project_preflight_duplicate_ite_slot_no_files) :-
    with_project_template_fixture(duplicate_ite_slot, check_project_preflight_failure).

test(project_preflight_reordered_ite_slots_no_files) :-
    with_project_template_fixture(reordered_ite_slots, check_project_preflight_failure).

test(project_preflight_malformed_main_no_files) :-
    with_project_template_fixture(malformed_main, check_project_preflight_failure).

test(project_preflight_missing_main_no_files) :-
    with_project_template_fixture(missing_main, check_project_preflight_failure).

test(project_preflight_malformed_header_no_files) :-
    with_project_template_fixture(malformed_header, check_project_preflight_failure).

test(project_preflight_missing_program_no_files) :-
    with_project_template_fixture(missing_program, check_project_preflight_failure).

test(project_preflight_malformed_program_no_files) :-
    with_project_template_fixture(malformed_program, check_project_preflight_failure).

test(project_preflight_missing_runtime_no_files) :-
    with_project_template_fixture(missing_runtime, check_project_preflight_failure).

test(project_preflight_malformed_runtime_no_files) :-
    with_project_template_fixture(malformed_runtime, check_project_preflight_failure).

test(project_preflight_duplicate_runtime_marker_no_files) :-
    with_project_template_fixture(duplicate_runtime_marker,
                                  check_project_preflight_failure).

test(project_preflight_duplicate_builtin_marker_no_files) :-
    with_project_template_fixture(duplicate_builtin_marker,
                                  check_project_preflight_failure).

test(project_preflight_duplicate_step_marker_no_files) :-
    with_project_template_fixture(duplicate_step_marker,
                                  check_project_preflight_failure).

test(project_preflight_reordered_runtime_markers_no_files) :-
    with_project_template_fixture(reordered_runtime_markers,
                                  check_project_preflight_failure).

runtime_section(cell_helpers, 'cell_helpers.cpp.mustache', runtime_cell_helpers).
runtime_section(arithmetic_eval, 'arithmetic_eval.cpp.mustache', runtime_arithmetic_eval).
runtime_section(builtin_dispatch, 'builtin_dispatch.cpp.mustache', runtime_builtin_dispatch).
runtime_section(step_execution, 'step_execution.cpp.mustache', runtime_step_execution).
runtime_section(lmdb_fact_source, 'lmdb_fact_source.cpp.mustache', runtime_lmdb_fact_source).

test(project_preflight_missing_runtime_section_no_files,
     [forall(runtime_section(Section, _, _))]) :-
    with_project_template_fixture(missing_runtime_section(Section),
                                  check_project_preflight_failure).

test(project_preflight_malformed_runtime_section_no_files,
     [forall(runtime_section(Section, _, _))]) :-
    with_project_template_fixture(malformed_runtime_section(Section),
                                  check_project_preflight_failure).

with_project_template_fixture(Fault, Goal) :-
    tmp_file(cpp_wam_preflight, Root),
    setup_call_cleanup(
        (make_directory(Root),
         directory_file_path(Root, templates, Templates),
         make_directory(Templates),
         directory_file_path(Templates, lowered, Lowered),
         make_directory(Lowered),
         directory_file_path(Templates, runtime, RuntimeSections),
         make_directory(RuntimeSections),
         source_file(wam_cpp_target:write_wam_cpp_project(_, _, _), Source),
         file_directory_name(Source, ModuleDir),
         directory_file_path(ModuleDir, '../../../templates/targets/cpp_wam', RealRoot),
         directory_file_path(RealRoot, 'runtime.h.mustache', Header),
         directory_file_path(Templates, 'runtime.h.mustache', HeaderCopy),
         copy_file(Header, HeaderCopy),
         (Fault == malformed_header -> write_fixture(HeaderCopy, "{{bad") ; true),
         directory_file_path(RealRoot, 'runtime.cpp.mustache', Runtime),
         directory_file_path(Templates, 'runtime.cpp.mustache', RuntimeCopy),
         (Fault == missing_runtime -> true ; copy_file(Runtime, RuntimeCopy)),
         (Fault == malformed_runtime -> write_fixture(RuntimeCopy, "{{bad") ; true),
         (Fault == duplicate_runtime_marker
         -> read_file_to_string(RuntimeCopy, RuntimeShell, []),
            string_concat(RuntimeShell, "{{cell_helpers}}", DuplicateShell),
            write_fixture(RuntimeCopy, DuplicateShell)
         ;  true),
         (Fault == duplicate_builtin_marker
         -> read_file_to_string(RuntimeCopy, RuntimeShell2, []),
            string_concat(RuntimeShell2, "{{builtin_dispatch}}", DuplicateBuiltin),
            write_fixture(RuntimeCopy, DuplicateBuiltin)
         ;  true),
         (Fault == duplicate_step_marker
         -> read_file_to_string(RuntimeCopy, RuntimeShell3, []),
            string_concat(RuntimeShell3, "{{step_execution}}", DuplicateStep),
            write_fixture(RuntimeCopy, DuplicateStep)
         ;  true),
         (Fault == reordered_runtime_markers
         -> write_fixture(RuntimeCopy,
                "{{cell_helpers}}{{builtin_dispatch}}{{arithmetic_eval}}{{step_execution}}{{lmdb_fact_source}}")
         ;  true),
         forall(runtime_section(Section, Name, _),
                (directory_file_path(RealRoot, 'runtime', RealRuntimeSections),
                 directory_file_path(RealRuntimeSections, Name, RealSection),
                 directory_file_path(RuntimeSections, Name, SectionCopy),
                 (Fault == missing_runtime_section(Section)
                 -> true ; copy_file(RealSection, SectionCopy)),
                 (Fault == malformed_runtime_section(Section)
                 -> write_fixture(SectionCopy, "{{bad") ; true))),
         directory_file_path(RealRoot, 'main.cpp.mustache', Main),
         directory_file_path(Templates, 'main.cpp.mustache', MainCopy),
         (Fault == missing_main -> true ; copy_file(Main, MainCopy)),
         directory_file_path(RealRoot, 'generated_program.cpp.mustache', ProgramShell),
         directory_file_path(Templates, 'generated_program.cpp.mustache', ProgramCopy),
         (Fault == missing_program -> true ; copy_file(ProgramShell, ProgramCopy)),
         (Fault == malformed_program -> write_fixture(ProgramCopy, "{{setup_code}}") ; true),
         directory_file_path(RealRoot, 'lowered/head_constant.cpp.stache', Stache),
         directory_file_path(Lowered, 'head_constant.cpp.stache', StacheCopy),
         (Fault == missing_stache -> true ; copy_file(Stache, StacheCopy)),
         (Fault == malformed_stache -> write_fixture(StacheCopy, "{{bad") ; true),
         directory_file_path(RealRoot, 'lowered/function.cpp.mustache', Function),
         directory_file_path(Lowered, 'function.cpp.mustache', FunctionCopy),
         (Fault == missing_lowered_shell -> true ; copy_file(Function, FunctionCopy)),
         (Fault == malformed_lowered_shell
         -> write_fixture(FunctionCopy, "{{comment}}\nbool {{name}}(WamState* vm) {{{body}}}{{unknown}}")
         ;  true),
         (Fault == missing_lowered_slot
         -> write_fixture(FunctionCopy, "{{comment}}\nbool {{name}}(WamState* vm) {}")
         ;  true),
         (Fault == duplicate_lowered_slot
         -> write_fixture(FunctionCopy, "{{comment}}{{name}}{{body}}{{body}}")
         ;  true),
         (Fault == reordered_lowered_slots
         -> write_fixture(FunctionCopy, "{{name}}{{comment}}{{body}}")
         ;  true),
         directory_file_path(RealRoot, 'lowered/ite.cpp.mustache', Ite),
         directory_file_path(Lowered, 'ite.cpp.mustache', IteCopy),
         (Fault == missing_ite_shell -> true ; copy_file(Ite, IteCopy)),
         (Fault == empty_ite_shell -> write_fixture(IteCopy, "") ; true),
         (Fault == unknown_ite_slot
         -> read_file_to_string(IteCopy, IteShell, []),
            string_concat(IteShell, "{{unknown}}", UnknownIte),
            write_fixture(IteCopy, UnknownIte)
         ;  true),
         (Fault == missing_ite_slot
         -> read_file_to_string(IteCopy, IteShell2, []),
            replace_first_fixture_marker(IteShell2, "{{condition}}", "", MissingIte),
            write_fixture(IteCopy, MissingIte)
         ;  true),
         (Fault == duplicate_ite_slot
         -> read_file_to_string(IteCopy, IteShell3, []),
            string_concat(IteShell3, "{{else}}", DuplicateIte),
            write_fixture(IteCopy, DuplicateIte)
         ;  true),
         (Fault == reordered_ite_slots
         -> read_file_to_string(IteCopy, IteShell4, []),
            replace_first_fixture_marker(IteShell4, "{{condition}}", "{{TEMP}}", TempIte),
            replace_first_fixture_marker(TempIte, "{{then}}", "{{condition}}", TempIte2),
            replace_first_fixture_marker(TempIte2, "{{TEMP}}", "{{then}}", ReorderedIte),
            write_fixture(IteCopy, ReorderedIte)
         ;  true),
         (Fault == malformed_main -> write_fixture(MainCopy, "{{bad") ; true)),
        call(Goal, Fault, Root, Templates),
        delete_directory_and_contents(Root)).

write_fixture(Path, Text) :-
    setup_call_cleanup(open(Path, write, Out), write(Out, Text), close(Out)).

replace_first_fixture_marker(Text, Marker, Replacement, Result) :-
    string_length(Marker, Length),
    sub_string(Text, At, Length, _, Marker),
    sub_string(Text, 0, At, _, Before),
    End is At + Length,
    sub_string(Text, End, _, 0, After),
    atomics_to_string([Before, Replacement, After], "", Result).

check_project_preflight_failure(Fault, Root, Templates) :-
    directory_file_path(Root, project, Project),
    catch(cpp_with_template_root_for_test(Templates,
          wam_cpp_target:write_wam_cpp_project([], [emit_main(true)], Project)),
          Error, true),
    (   Fault == malformed_main
    ->  assertion(Error = error(cpp_wam_template_tags(main_shim, _), _))
    ;   Fault == missing_main
    ->  assertion(Error = error(cpp_wam_template_load(main_shim, _, _), _))
    ;   Fault == malformed_header
    ->  assertion(Error = error(cpp_wam_template_tags(runtime_header, _), _))
    ;   Fault == missing_runtime
    ->  assertion(Error = error(cpp_wam_template_load(runtime_source, _, _), _))
    ;   memberchk(Fault, [malformed_runtime, duplicate_runtime_marker,
                          duplicate_builtin_marker, duplicate_step_marker,
                          reordered_runtime_markers])
    ->  assertion(Error = error(cpp_wam_template_tags(runtime_source, _), _))
    ;   Fault = missing_runtime_section(Section)
    ->  runtime_section(Section, _, TemplateId),
        assertion(Error = error(cpp_wam_template_load(TemplateId, _, _), _))
    ;   Fault = malformed_runtime_section(Section)
    ->  runtime_section(Section, _, TemplateId),
        assertion(Error = error(cpp_wam_template_tags(TemplateId, _), _))
    ;   Fault == missing_program
    ->  assertion(Error = error(cpp_wam_template_load(generated_program, _, _), _))
    ;   Fault == malformed_program
    ->  assertion(Error = error(cpp_wam_template_tags(generated_program, _), _))
    ;   Fault == missing_lowered_shell
    ->  assertion(Error = error(cpp_wam_template_load(lowered_function, _, _), _))
    ;   memberchk(Fault, [malformed_lowered_shell, missing_lowered_slot,
                          duplicate_lowered_slot,
                          reordered_lowered_slots])
    ->  assertion(Error = error(cpp_wam_template_tags(lowered_function, _), _))
    ;   Fault == missing_ite_shell
    ->  assertion(Error = error(cpp_wam_template_load(lowered_ite, _, _), _))
    ;   Fault == empty_ite_shell
    ->  assertion(Error = error(cpp_wam_template_empty(lowered_ite, _), _))
    ;   memberchk(Fault, [unknown_ite_slot, missing_ite_slot,
                          duplicate_ite_slot, reordered_ite_slots])
    ->  assertion(Error = error(cpp_wam_template_tags(lowered_ite, _), _))
    ;   assertion(Error = error(cpp_wam_stache_failure(head_constant, _, _), _))
    ),
    assertion(\+ exists_directory(Project)).

test(missing_file, [throws(error(cpp_wam_template_load(runtime_header, _, _), _))]) :-
    cpp_render_template_at_root('/tmp/absent_wam_cpp_template_fixture',
                                runtime_header, [], _).

test(unexpected_variables,
     [throws(error(domain_error(cpp_wam_template_variables, _), _))]) :-
    cpp_render_template_at_root('/tmp', runtime_header, [name=x], _).

% Frozen from the pre-Phase-2 emit_one/2 on e1bf1d2. Lengths count characters;
% SHA-256 covers the exact UTF-8 bytes, including indentation and final LF.
old_head_constant_digest("foo", "A1", 246,
    'c8410e0ae0fbd5fbe0102dadf95a8bc9066fef43384eeadb841aaeadec154c35').
old_head_constant_digest("42", "X2", 262,
    'c79a1be4022216fb1117c2697986b610c481b5c45a0e6cfe006019c850d2c97c').
old_head_constant_digest("3.5", "Y3", 261,
    'e23ca93f7c255678dc3a27e2f50ff5dc462348a43c570bc06ddb3fff61396a6e').
old_head_constant_digest("'a\\\"b'", "A1", 255,
    '7f854e3ced5690ecf72f6fa97a6b95d81d615dc3a8e7cfe084a9000c819b68e2').
old_head_constant_digest("'a\\\\b'", "A1", 255,
    'f34ee254e9903c8c04f693e9d803359052a2f7b84c5a13d79f75f8474357fa06').
old_head_constant_digest("'{{name}}'", "A1", 263,
    '4c9757bd40782eb98ba2bead2f60d54ada9aaa3b583efe7e189fe47aaf36b916').
old_head_constant_digest("'{{Ai}}'", "A1", 257,
    '9c9e8d89c15580f873544aa6a249c6b1c0ec45fd5eb5506899a43137c97c9024').

test(head_constant_old_bytes, [forall(old_head_constant_digest(C, Reg, Length, Digest))]) :-
    with_output_to(string(Text),
        wam_cpp_lowered_emitter:emit_one(get_constant(C, Reg), "  ")),
    string_length(Text, Length),
    crypto_data_hash(Text, Actual, [algorithm(sha256), encoding(utf8)]),
    assertion(Actual == Digest).

% Frozen before get_integer/get_nil joined the head-match template family.
old_head_integer_nil_digest(get_integer("42", "A1"), 261,
    '7988437cdb2b295c42c01a14b2a9c6a9d0e88c294cc33ed56043ed19ff5ec8f0').
old_head_integer_nil_digest(get_integer("-7", "X2"), 261,
    '166ebf499cf5288151ad8beca50a10a723c373777fa18e7aade4035b45e21f67').
old_head_integer_nil_digest(get_integer("0007", "Y3"), 267,
    '00bea1e5930550854be3b020ef1d5e537d82a9f9b55833b6de53f97f783f61d2').
old_head_integer_nil_digest(get_nil("A1"), 234,
    'e33550a193983e50ec0dcb23c4268c6cc93ce29ce6e603ec2f50ea2c1a600af8').
old_head_integer_nil_digest(get_nil("Y3"), 234,
    'e437b88586617da1ca7fe991ea9dcb646a745534d6e7a6e479e70f08cea24e49').

test(head_integer_nil_old_bytes,
     [forall(old_head_integer_nil_digest(Instruction, Length, Digest))]) :-
    with_output_to(string(Text), wam_cpp_lowered_emitter:emit_one(Instruction, "  ")),
    string_length(Text, Length),
    crypto_data_hash(Text, Actual, [algorithm(sha256), encoding(utf8)]),
    assertion(Actual == Digest).

head_atom_vars([op=head_constant(atom("foo")), 'I'="",
                'Comment'="get_constant foo, A1", 'Ai'="A1"]).

test(head_constant_unknown_shape,
     [throws(error(domain_error(cpp_wam_stache_variables, _), _))]) :-
    head_atom_vars([_|Tail]),
    cpp_render_stache_at_root('/tmp', head_constant, [op=head_constant(other)|Tail], _).

test(head_constant_nonground,
     [throws(error(domain_error(cpp_wam_stache_variables, _), _))]) :-
    cpp_render_stache_at_root('/tmp', head_constant, [op=head_constant(atom(_))|_], _).

test(head_constant_missing_key,
     [throws(error(domain_error(cpp_wam_stache_variables, _), _))]) :-
    cpp_render_stache_at_root('/tmp', head_constant,
        [op=head_constant(atom("foo")), 'I'="", 'Ai'="A1"], _).

test(head_constant_foreign_cwd) :-
    setup_call_cleanup(
        working_directory(Here, '/tmp'),
        (with_output_to(string(Text),
            wam_cpp_lowered_emitter:emit_one(get_constant("foo", "A1"), "  ")),
         old_head_constant_digest("foo", "A1", Length, Digest),
         string_length(Text, Length),
         crypto_data_hash(Text, Actual, [algorithm(sha256), encoding(utf8)]),
         assertion(Actual == Digest)),
        working_directory(_, Here)).

test(head_constant_missing_file,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    cpp_render_stache_at_root('/tmp/absent_wam_cpp_stache_fixture', head_constant, Vars, _).

with_head_fixture(Body, Goal) :-
    tmp_file(cpp_wam_stache, Root),
    setup_call_cleanup(
        (make_directory(Root), directory_file_path(Root, lowered, Dir),
         make_directory(Dir), directory_file_path(Dir, 'head_constant.cpp.stache', Path),
         setup_call_cleanup(open(Path, write, Stream), format(Stream, '~s', [Body]), close(Stream))),
        call(Goal, Root),
        delete_directory_and_contents(Root)).

test(head_constant_malformed_block,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 1) }}\n{{match op}}{{case head_constant(atom(Esc))}}x",
        render_head_fixture(Vars)).

test(head_constant_invalid_dialect,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 2) }}\n{{match op}}{{case head_constant(atom(Esc))}}x{{case head_constant(value(CppVal))}}y{{/match}}",
        render_head_fixture(Vars)).

test(head_constant_nonlinear_case,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 1) }}\n{{match op}}{{case head_constant(atom(Esc-Esc))}}x{{case head_constant(value(CppVal))}}y{{/match}}",
        render_head_fixture(Vars)).

test(head_constant_unknown_placeholder,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 1) }}\n{{match op}}{{case head_constant(atom(Esc))}}{{missing}}{{case head_constant(value(CppVal))}}value{{/match}}",
        render_head_fixture(Vars)).

test(head_constant_wrong_case_placeholder,
     [throws(error(cpp_wam_stache_failure(head_constant, _, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 1) }}\n{{match op}}{{case head_constant(atom(Esc))}}{{CppVal}}{{case head_constant(value(CppVal))}}value{{/match}}",
        render_head_fixture(Vars)).

test(head_constant_empty_required,
     [throws(error(cpp_wam_stache_empty(head_constant, _), _))]) :-
    head_atom_vars(Vars),
    with_head_fixture("{{! dialect(pattern_stache, 1) }}\n{{match op}}{{case head_constant(atom(Esc))}}{{case head_constant(value(CppVal))}}value{{/match}}",
        render_head_fixture(Vars)).

render_head_fixture(Vars, Root) :-
    cpp_render_stache_at_root(Root, head_constant, Vars, _).

test(head_constant_cache_cold_warm_and_mutation) :-
    head_cache_body("A", Initial),
    with_head_fixture(Initial, check_head_cache_mutation).

head_cache_body(AtomBody, Text) :-
    format(string(Text),
        '{{! dialect(pattern_stache, 1) }}~n{{match op}}{{case head_constant(atom(Esc))}}~s{{case head_constant(value(CppVal))}}V{{/match}}',
        [AtomBody]).

write_head_cache_body(Path, AtomBody) :-
    head_cache_body(AtomBody, Text),
    setup_call_cleanup(open(Path, write, Stream),
        format(Stream, '~s', [Text]), close(Stream)).

check_head_cache_mutation(Root) :-
    head_atom_vars(Vars),
    directory_file_path(Root, 'lowered/head_constant.cpp.stache', Path),
    cpp_render_stache_at_root(Root, head_constant, Vars, Cold),
    cpp_render_stache_at_root(Root, head_constant, Vars, Warm),
    assertion(Cold == "A"), assertion(Warm == Cold),
    write_head_cache_body(Path, "B"),
    cpp_render_stache_at_root(Root, head_constant, Vars, Fresh),
    assertion(Fresh == "B"),
    write_head_cache_body(Path, "{{missing}}"),
    catch(cpp_render_stache_at_root(Root, head_constant, Vars, _), Error, true),
    assertion(Error = error(cpp_wam_stache_failure(head_constant, _, _), _)).

:- end_tests(wam_cpp_templates).

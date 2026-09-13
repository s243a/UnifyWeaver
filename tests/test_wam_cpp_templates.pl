:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [compile_wam_runtime_header_to_cpp/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_templates',
              [cpp_render_template_at_root/4, cpp_render_stache_at_root/4,
               cpp_with_template_root_for_test/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter', []).

% SHA-256 covers UTF-8 bytes; lengths below count Prolog characters.
% Captured from the old compile_wam_runtime_header_to_cpp/2 at local main
% 1c5ce3682c7047ef9470ee0d5563e3313eccfa24 before extraction.
old_header_digest(plain, 57611,
    '268fd386b593d5826b54376a455d58a1ad9148439996530833d34ab9e4e06352').
old_header_digest(lmdb, 57852,
    'cc9ffdacc92b1234212340e7fbb2ae27786180a36b5818a43924708bc0c8f471').

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
    string_length(Runtime, 310726),
    crypto_data_hash(Runtime, Digest, [algorithm(sha256), encoding(utf8)]),
    assertion(Digest == 'fa0317a11da5069d0fa18a36bc1937419bcf187e7f6ec4c5d278fd0270cfd300').

test(program_shell_does_not_rescan_fragments) :-
    cpp_render_template_at_root('templates/targets/cpp_wam', generated_program,
        [predicates_code="P {{setup_code}}", setup_code="S {{predicates_code}}"],
        Text),
    assertion(sub_string(Text, _, _, _, "P {{setup_code}}")),
    assertion(sub_string(Text, _, _, _, "S {{predicates_code}}")),
    assertion(\+ sub_string(Text, _, _, _, "P S {{predicates_code}}")),
    assertion(\+ sub_string(Text, _, _, _, "S P {{setup_code}}")).

project_variant(functions, [emit_mode(functions), emit_main(true)],
    'fadbf0020e10e86fbf206a5f1f6b0b912891676c8dc13a3168a34cece76aafa7').
project_variant(interpreter, [emit_mode(interpreter)],
    '172c9e2924ade91fc01513916c2bab0dd96cdf700be0d95df5c9f939ec143ba3').
project_variant(lmdb,
    [emit_mode(functions), emit_main(true),
     cpp_fact_sources([source(phase3_pair/2, lmdb('/tmp/phase3b_lmdb'))])],
    'a9bf8ebfdf493820f05a9323ffc86c365b12844d83a91de927ae8b045edb0e3c').

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

test(project_preflight_reordered_runtime_markers_no_files) :-
    with_project_template_fixture(reordered_runtime_markers,
                                  check_project_preflight_failure).

runtime_section(cell_helpers, 'cell_helpers.cpp.mustache', runtime_cell_helpers).
runtime_section(arithmetic_eval, 'arithmetic_eval.cpp.mustache', runtime_arithmetic_eval).
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
         (Fault == reordered_runtime_markers
         -> write_fixture(RuntimeCopy,
                "{{arithmetic_eval}}{{cell_helpers}}{{lmdb_fact_source}}")
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
         (Fault == malformed_main -> write_fixture(MainCopy, "{{bad") ; true)),
        call(Goal, Fault, Root, Templates),
        delete_directory_and_contents(Root)).

write_fixture(Path, Text) :-
    setup_call_cleanup(open(Path, write, Out), write(Out, Text), close(Out)).

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

head_atom_vars([op=head_constant(atom("foo")), 'I'="", 'CStr'="foo", 'AiStr'="A1",
                'Ai'="A1"]).

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
        [op=head_constant(atom("foo")), 'I'="", 'CStr'="foo", 'AiStr'="A1"], _).

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

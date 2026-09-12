:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [compile_wam_runtime_header_to_cpp/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_templates',
              [cpp_render_template_at_root/4, cpp_render_stache_at_root/4]).
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

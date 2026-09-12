:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- use_module(library(plunit)).
:- use_module(library(crypto), [crypto_data_hash/3]).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [compile_wam_runtime_header_to_cpp/2]).
:- use_module('../src/unifyweaver/targets/wam_cpp_templates',
              [cpp_render_template_at_root/4]).

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

:- end_tests(wam_cpp_templates).

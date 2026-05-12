:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_cpp_generator.pl — plunit tests for the hybrid C++ WAM target.
%
% Mirrors tests/test_wam_lua_generator.pl in structure, but scoped to the
% subset of behaviour the initial wam_cpp_target / wam_cpp_lowered_emitter
% pair guarantees: exports, registry wiring, project layout, lowerability
% checks, and lowered-function emission. End-to-end compile-and-run tests
% are gated on the presence of a host C++17 compiler (g++ / clang++).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- use_module('../src/unifyweaver/core/target_registry').

:- begin_tests(wam_cpp_generator).

:- dynamic user:wam_cpp_fact/1.
:- dynamic user:wam_cpp_choice/1.
:- dynamic user:wam_cpp_caller/1.

user:wam_cpp_fact(a).
user:wam_cpp_choice(a).
user:wam_cpp_choice(b).
user:wam_cpp_caller(X) :- user:wam_cpp_fact(X).

% --------------------------------------------------------------------
% Module-level exports
% --------------------------------------------------------------------
test(exports) :-
    assertion(current_predicate(wam_cpp_target:write_wam_cpp_project/3)),
    assertion(current_predicate(wam_cpp_target:compile_wam_predicate_to_cpp/4)),
    assertion(current_predicate(wam_cpp_target:compile_wam_runtime_to_cpp/2)),
    assertion(current_predicate(wam_cpp_target:compile_wam_runtime_header_to_cpp/2)),
    assertion(current_predicate(wam_cpp_target:cpp_wam_resolve_emit_mode/2)),
    assertion(current_predicate(wam_cpp_target:escape_cpp_string/2)),
    assertion(current_predicate(wam_cpp_lowered_emitter:wam_cpp_lowerable/3)),
    assertion(current_predicate(wam_cpp_lowered_emitter:lower_predicate_to_cpp/4)),
    assertion(current_predicate(wam_cpp_lowered_emitter:cpp_lowered_func_name/2)).

% --------------------------------------------------------------------
% Registry wiring
% --------------------------------------------------------------------
test(registry) :-
    assertion(target_exists(wam_cpp)),
    assertion(target_family(wam_cpp, native)),
    assertion(target_module(wam_cpp, wam_cpp_target)).

% --------------------------------------------------------------------
% Emit-mode resolution
% --------------------------------------------------------------------
test(emit_mode_default) :-
    cpp_wam_resolve_emit_mode([], Mode),
    assertion(Mode == interpreter).

test(emit_mode_functions) :-
    cpp_wam_resolve_emit_mode([emit_mode(functions)], Mode),
    assertion(Mode == functions).

test(emit_mode_mixed) :-
    cpp_wam_resolve_emit_mode([emit_mode(mixed([foo/2,bar/3]))], Mode),
    assertion(Mode == mixed([foo/2, bar/3])).

test(emit_mode_invalid, [throws(error(domain_error(wam_cpp_emit_mode, garbage), _))]) :-
    cpp_wam_resolve_emit_mode([emit_mode(garbage)], _).

% --------------------------------------------------------------------
% Lowered function naming
% --------------------------------------------------------------------
test(lowered_func_name_simple) :-
    cpp_lowered_func_name(foo/2, Name),
    assertion(Name == 'lowered_foo_2').

test(lowered_func_name_sanitised) :-
    cpp_lowered_func_name('my-pred'/3, Name),
    assertion(Name == 'lowered_my_pred_3').

% --------------------------------------------------------------------
% Lowerability classification (operates on instruction lists directly)
% --------------------------------------------------------------------
test(lowerability_deterministic) :-
    Instrs = [get_constant("a", "A1"), proceed],
    wam_cpp_lowerable(wam_cpp_fact/1, Instrs, Reason),
    assertion(Reason == deterministic).

test(lowerability_multi_clause_1) :-
    Instrs = [try_me_else("L2"),
              get_constant("a", "A1"),
              proceed,
              trust_me,
              get_constant("b", "A1"),
              proceed],
    wam_cpp_lowerable(wam_cpp_choice/1, Instrs, Reason),
    assertion(Reason == multi_clause_1).

test(is_deterministic_helper) :-
    assertion(is_deterministic_pred_cpp([proceed])),
    assertion(\+ is_deterministic_pred_cpp([try_me_else("L"), proceed])).

% --------------------------------------------------------------------
% Lowered function emission
% --------------------------------------------------------------------
test(lower_predicate_emits_signature_and_proceed) :-
    Instrs = [get_constant("a", "A1"), proceed],
    lower_predicate_to_cpp(wam_cpp_fact/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'bool lowered_wam_cpp_fact_1(WamState* vm)')),
    assertion(sub_atom(Code, _, _, _, 'return true;')),
    assertion(sub_atom(Code, _, _, _, 'get_constant a, A1')).

test(lower_predicate_emits_unify_for_constants) :-
    Instrs = [get_constant("hello", "A1"), proceed],
    lower_predicate_to_cpp(test_const/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'Value::Atom("hello")')),
    assertion(sub_atom(Code, _, _, _, 'vm->trail_binding')),
    assertion(sub_atom(Code, _, _, _, 'return false;')).

test(lower_predicate_emits_call_dispatch) :-
    Instrs = [put_constant("a", "A1"), call("wam_cpp_fact/1", "1"), proceed],
    lower_predicate_to_cpp(wam_cpp_caller/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'vm->labels.find("wam_cpp_fact/1")')),
    assertion(sub_atom(Code, _, _, _, 'vm->run()')).

test(lower_predicate_routes_foreign_calls) :-
    Instrs = [put_constant("a", "A1"), call("edge/2", "2"), proceed],
    lower_predicate_to_cpp(uses_foreign/1, Instrs,
                           [foreign_pred_keys(["edge/2"])], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'Instruction::CallForeign("edge/2", 2)')).

% --------------------------------------------------------------------
% Instruction literal emission (for the interpreter array)
% --------------------------------------------------------------------
test(instruction_literal_get_constant) :-
    wam_instruction_to_cpp_literal(get_constant("a", "A1"), Code),
    assertion(Code == 'Instruction::GetConstant(Value::Atom("a"), "A1")').

test(instruction_literal_proceed) :-
    wam_instruction_to_cpp_literal(proceed, Code),
    assertion(Code == 'Instruction::Proceed()').

test(instruction_literal_call) :-
    wam_instruction_to_cpp_literal(call("foo/2", "2"), Code),
    assertion(Code == 'Instruction::Call("foo/2", 2)').

% --------------------------------------------------------------------
% String escaping
% --------------------------------------------------------------------
test(escape_cpp_string_backslash) :-
    escape_cpp_string("a\\b", Out),
    assertion(Out == "a\\\\b").

test(escape_cpp_string_quote) :-
    escape_cpp_string("a\"b", Out),
    assertion(Out == "a\\\"b").

% --------------------------------------------------------------------
% Project layout
% --------------------------------------------------------------------
test(project_layout) :-
    unique_cpp_tmp_dir('tmp_cpp_layout', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/wam_runtime.h', Header),
          directory_file_path(TmpDir, 'cpp/wam_runtime.cpp', Runtime),
          directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          assertion(exists_file(Header)),
          assertion(exists_file(Runtime)),
          assertion(exists_file(Program))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(project_runtime_header_content) :-
    unique_cpp_tmp_dir('tmp_cpp_header', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/wam_runtime.h', Header),
          read_file_to_string(Header, Code, []),
          assertion(sub_string(Code, _, _, _, 'namespace wam_cpp')),
          assertion(sub_string(Code, _, _, _, 'struct Value')),
          assertion(sub_string(Code, _, _, _, 'struct Instruction')),
          assertion(sub_string(Code, _, _, _, 'struct WamState')),
          assertion(sub_string(Code, _, _, _, 'unify(const Value&'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(project_program_includes_runtime) :-
    unique_cpp_tmp_dir('tmp_cpp_prog', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, '#include "wam_runtime.h"'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lowered_functions_mode) :-
    unique_cpp_tmp_dir('tmp_cpp_lowered', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_fact/1],
            [emit_mode(functions)],
            TmpDir),
        ( directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'bool lowered_wam_cpp_fact_1(WamState* vm)'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

% --------------------------------------------------------------------
% Optional: header compiles cleanly with a C++17 compiler if one
% is on PATH. Skipped silently otherwise — we don't want to gate
% Prolog-side CI on host toolchains.
% --------------------------------------------------------------------
test(cpp_compiler_smoke, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_smoke', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1],
                              [emit_mode(functions)], TmpDir),
        ( directory_file_path(TmpDir, 'cpp', CppDir),
          % Compile each .cpp separately (g++ disallows -o with -c and
          % multiple inputs). The generated program has no main(); -c
          % produces .o files and that is all we need to verify the
          % runtime + lowered code is syntactically valid.
          compile_one(CppDir, 'wam_runtime.cpp', 'wam_runtime.o', R1),
          assertion(R1 == exit(0)),
          compile_one(CppDir, 'generated_program.cpp', 'generated_program.o', R2),
          assertion(R2 == exit(0))
        ),
        delete_directory_and_contents(TmpDir)
    ).

compile_one(CppDir, Src, Obj, Status) :-
    directory_file_path(CppDir, Src, SrcPath),
    directory_file_path(CppDir, Obj, ObjPath),
    process_create(path('g++'),
                   ['-std=c++17', '-c', '-o', ObjPath, SrcPath],
                   [stderr(null), process(PID)]),
    process_wait(PID, Status).

:- end_tests(wam_cpp_generator).

% --------------------------------------------------------------------
% Helpers
% --------------------------------------------------------------------

unique_cpp_tmp_dir(Prefix, Dir) :-
    get_time(T), N is round(T * 1000),
    format(atom(Dir), 'tests/~w_~w', [Prefix, N]).

cpp_compiler_available :-
    catch(
        ( process_create(path('g++'), ['--version'],
                         [stdout(null), stderr(null), process(PID)]),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

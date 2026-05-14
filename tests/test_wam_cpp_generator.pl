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
:- dynamic user:wam_cpp_rect/1.
:- dynamic user:wam_cpp_has_rect/0.
:- dynamic user:wam_cpp_has_rect_wrong/0.
:- dynamic user:wam_cpp_first/2.
:- dynamic user:wam_cpp_lst/1.
:- dynamic user:wam_cpp_add1/2.
:- dynamic user:wam_cpp_gt/2.
:- dynamic user:wam_cpp_test_arith/0.
:- dynamic user:wam_cpp_test_eq/0.
:- dynamic user:wam_cpp_test_neq/0.
:- dynamic user:wam_cpp_is_atom/1.
:- dynamic user:wam_cpp_is_int/1.
:- dynamic user:wam_cpp_is_num/1.
:- dynamic user:wam_cpp_is_var/1.
:- dynamic user:wam_cpp_is_compound/1.
:- dynamic user:wam_cpp_test_nonvar/0.
:- dynamic user:wam_cpp_test_functor/0.
:- dynamic user:wam_cpp_test_arg1/0.
:- dynamic user:wam_cpp_test_arg_bad/0.
:- dynamic user:wam_cpp_test_univ_decompose/0.
:- dynamic user:wam_cpp_test_univ_compose/0.
:- dynamic user:wam_cpp_test_unify/0.
:- dynamic user:wam_cpp_test_unify_fail/0.
:- dynamic user:wam_cpp_test_write/0.
:- dynamic user:wam_cpp_item/1.
:- dynamic user:wam_cpp_num/1.
:- dynamic user:wam_cpp_test_findall/0.
:- dynamic user:wam_cpp_test_findall_empty/0.
:- dynamic user:wam_cpp_test_findall_doubled/0.
:- dynamic user:wam_cpp_test_bagof/0.
:- dynamic user:wam_cpp_test_bagof_empty/0.
:- dynamic user:wam_cpp_test_setof/0.
:- dynamic user:wam_cpp_test_setof_empty/0.
:- dynamic user:wam_cpp_test_count/0.
:- dynamic user:wam_cpp_test_sum/0.
:- dynamic user:wam_cpp_test_min/0.
:- dynamic user:wam_cpp_test_max/0.
:- dynamic user:wam_cpp_test_set/0.
:- dynamic user:wam_cpp_h1/1.
:- dynamic user:wam_cpp_h2/1.
:- dynamic user:wam_cpp_two_helpers/0.
:- dynamic user:wam_cpp_two_helpers_swap/0.
:- dynamic user:wam_cpp_length_acc/3.
:- dynamic user:wam_cpp_list_length/2.
:- dynamic user:wam_cpp_test_len_empty/0.
:- dynamic user:wam_cpp_test_len_one/0.
:- dynamic user:wam_cpp_test_len_three/0.
:- dynamic user:wam_cpp_test_len_five/0.
% List & term builtins (member/2, length/2, copy_term/2):
:- dynamic user:wam_cpp_test_member_yes/0.
:- dynamic user:wam_cpp_test_member_no/0.
:- dynamic user:wam_cpp_test_member_first/0.
:- dynamic user:wam_cpp_test_length_three/0.
:- dynamic user:wam_cpp_test_length_zero/0.
:- dynamic user:wam_cpp_test_length_bad/0.
:- dynamic user:wam_cpp_test_copy_basic/0.
:- dynamic user:wam_cpp_test_copy_atom/0.
:- dynamic user:wam_cpp_test_enum_member/0.

user:wam_cpp_test_member_yes   :- member(b, [a, b, c]).
user:wam_cpp_test_member_no    :- member(z, [a, b, c]).
user:wam_cpp_test_member_first :- member(a, [a, b, c]).
user:wam_cpp_test_length_three :- length([a, b, c], 3).
user:wam_cpp_test_length_zero  :- length([], 0).
user:wam_cpp_test_length_bad   :- length([a, b, c], 5).
user:wam_cpp_test_copy_basic   :- copy_term(foo(X, X, _Y), T), T = foo(A, A, _B).
user:wam_cpp_test_copy_atom    :- copy_term(hello, T), T = hello.
user:wam_cpp_test_enum_member  :- findall(X, member(X, [a, b, c]), L),
                                  L = [a, b, c].

% Indexing-instruction fixtures (switch_on_constant / switch_on_term):
:- dynamic user:wam_cpp_color/1.
:- dynamic user:wam_cpp_shape/2.
:- dynamic user:wam_cpp_mixed/1.
:- dynamic user:wam_cpp_listy/1.

user:wam_cpp_color(red).
user:wam_cpp_color(green).
user:wam_cpp_color(blue).
user:wam_cpp_shape(circle,   round).
user:wam_cpp_shape(square,   angular).
user:wam_cpp_shape(triangle, angular).
user:wam_cpp_mixed(a).
user:wam_cpp_mixed(1).
user:wam_cpp_mixed(foo(x)).
user:wam_cpp_listy([]).
user:wam_cpp_listy([_|_]).

user:wam_cpp_test_write :- write(hello), nl.
% Y-reg isolation: both helpers use Y1/Y2 internally. Caller relies on
% preserved Y1 across the two calls.
user:wam_cpp_h1(X) :- user:wam_cpp_num(_), X = a.
user:wam_cpp_h2(Y) :- user:wam_cpp_num(_), Y = b.
user:wam_cpp_two_helpers      :- user:wam_cpp_h1(A), user:wam_cpp_h2(B), A = a, B = b.
user:wam_cpp_two_helpers_swap :- user:wam_cpp_h1(A), user:wam_cpp_h2(B), A = b, B = a.
% Tail-recursive list length — exercises cp threading + Y-reg framing
% across recursive calls.
user:wam_cpp_length_acc([], Acc, Acc).
user:wam_cpp_length_acc([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    user:wam_cpp_length_acc(T, Acc1, N).
user:wam_cpp_list_length(L, N) :- user:wam_cpp_length_acc(L, 0, N).
user:wam_cpp_test_len_empty :- user:wam_cpp_list_length([], 0).
user:wam_cpp_test_len_one   :- user:wam_cpp_list_length([a], 1).
user:wam_cpp_test_len_three :- user:wam_cpp_list_length([a, b, c], 3).
user:wam_cpp_test_len_five  :- user:wam_cpp_list_length([a, b, c, d, e], 5).
user:wam_cpp_item(a). user:wam_cpp_item(b). user:wam_cpp_item(c).
user:wam_cpp_num(1).  user:wam_cpp_num(2).  user:wam_cpp_num(3). user:wam_cpp_num(2).
user:wam_cpp_test_findall         :- findall(X, user:wam_cpp_item(X), L), L = [a, b, c].
user:wam_cpp_test_findall_empty   :- findall(_, fail, L), L = [].
user:wam_cpp_test_findall_doubled :- findall(p(X, X), user:wam_cpp_item(X), L),
                                     L = [p(a, a), p(b, b), p(c, c)].
user:wam_cpp_test_bagof           :- bagof(X, user:wam_cpp_item(X), L), L = [a, b, c].
user:wam_cpp_test_bagof_empty     :- bagof(_, fail, _).
user:wam_cpp_test_setof           :- setof(X, user:wam_cpp_num(X), L), L = [1, 2, 3].
user:wam_cpp_test_setof_empty     :- setof(_, fail, _).
user:wam_cpp_test_count :- aggregate_all(count, user:wam_cpp_item(_), N), N = 3.
user:wam_cpp_test_sum   :- aggregate_all(sum(X),  user:wam_cpp_num(X), S), S = 8.
user:wam_cpp_test_min   :- aggregate_all(min(X),  user:wam_cpp_num(X), M), M = 1.
user:wam_cpp_test_max   :- aggregate_all(max(X),  user:wam_cpp_num(X), M), M = 3.
user:wam_cpp_test_set   :- aggregate_all(set(X),  user:wam_cpp_num(X), S), S = [1, 2, 3].

user:wam_cpp_fact(a).
user:wam_cpp_choice(a).
user:wam_cpp_choice(b).
user:wam_cpp_caller(X) :- user:wam_cpp_fact(X).
user:wam_cpp_rect(box(1, 2)).
user:wam_cpp_has_rect          :- user:wam_cpp_rect(box(1, 2)).
user:wam_cpp_has_rect_wrong    :- user:wam_cpp_rect(box(1, 3)).
user:wam_cpp_first(box(X, _), X).
user:wam_cpp_lst([a, b, c]).
% Arithmetic & comparison
user:wam_cpp_add1(X, Y)        :- Y is X + 1.
user:wam_cpp_gt(X, Y)          :- X > Y.
user:wam_cpp_test_arith        :- 6 is 2 + 4, 12 is 3 * 4, 5 is 10 / 2.
user:wam_cpp_test_eq           :- 5 =:= 2 + 3.
user:wam_cpp_test_neq          :- 5 =\= 6.
% Type checks
user:wam_cpp_is_atom(X)        :- atom(X).
user:wam_cpp_is_int(X)         :- integer(X).
user:wam_cpp_is_num(X)         :- number(X).
user:wam_cpp_is_var(X)         :- var(X).
user:wam_cpp_is_compound(X)    :- compound(X).
user:wam_cpp_test_nonvar       :- X = foo, nonvar(X).
% Term inspection
user:wam_cpp_test_functor      :- functor(box(1, 2), box, 2).
user:wam_cpp_test_arg1         :- arg(1, box(a, b), a).
user:wam_cpp_test_arg_bad      :- arg(1, box(a, b), z).
user:wam_cpp_test_univ_decompose :- box(1, 2) =.. [box, 1, 2].
user:wam_cpp_test_univ_compose   :- T =.. [foo, a, b], T = foo(a, b).
% =/2 / \\=/2
user:wam_cpp_test_unify        :- X = foo, X = foo.
user:wam_cpp_test_unify_fail   :- foo \= foo.

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

% ------------------------------------------------------------------
% End-to-end: build a binary with main.cpp, run queries, check exit.
% ------------------------------------------------------------------

test(cpp_e2e_fact, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_fact', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_fact/1', [a], true),
          run_query(BinPath, 'wam_cpp_fact/1', [b], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_choice_backtracking, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_choice', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_choice/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_choice/1', [a], true),
          % Clause 2 only reachable via backtracking — exercises the
          % choice point / trail / TrustMe path.
          run_query(BinPath, 'wam_cpp_choice/1', [b], true),
          run_query(BinPath, 'wam_cpp_choice/1', [c], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_caller, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_caller', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_caller/1, user:wam_cpp_fact/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % caller(X) :- fact(X). Exercises Call dispatch + Proceed
          % through the labels table.
          run_query(BinPath, 'wam_cpp_caller/1', [a], true),
          run_query(BinPath, 'wam_cpp_caller/1', [b], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Compound terms + lists: heap-resident structures via shared_ptr cells.
% Exercises Get/PutStructure + Get/PutList + Unify*/Set* + the CLI parser
% for compound and list syntax.
% ------------------------------------------------------------------

test(cpp_e2e_structure_head_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_struct', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_rect/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(1,2)'], true),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(1,3)'], false),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(2,2)'], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_structure_build_and_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_build', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_has_rect/0, user:wam_cpp_has_rect_wrong/0,
             user:wam_cpp_rect/1],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % has_rect builds box(1,2) and calls rect/1 — exercises
          % PutStructure + SetConstant + Execute.
          run_query(BinPath, 'wam_cpp_has_rect/0',       [], true),
          run_query(BinPath, 'wam_cpp_has_rect_wrong/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_structure_destructure, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_destr', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_first/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % first(box(X, _), X). Pulls X out of the compound and unifies
          % with A2 — exercises UnifyVariable + GetValue across compounds.
          run_query(BinPath, 'wam_cpp_first/2', ['box(1,2)', '1'], true),
          run_query(BinPath, 'wam_cpp_first/2', ['box(7,8)', '7'], true),
          run_query(BinPath, 'wam_cpp_first/2', ['box(1,2)', '9'], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Builtins: arithmetic, comparison, type checks, term inspection, =/2.
% ------------------------------------------------------------------

test(cpp_e2e_builtin_arithmetic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_arith', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_add1/2, user:wam_cpp_test_arith/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_add1/2',     [5, 6], true),
          run_query(BinPath, 'wam_cpp_add1/2',     [5, 7], false),
          run_query(BinPath, 'wam_cpp_test_arith/0', [],  true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_comparison, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_cmp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt/2,
                               user:wam_cpp_test_eq/0,
                               user:wam_cpp_test_neq/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_gt/2',     [5, 3], true),
          run_query(BinPath, 'wam_cpp_gt/2',     [3, 5], false),
          run_query(BinPath, 'wam_cpp_test_eq/0',  [],  true),
          run_query(BinPath, 'wam_cpp_test_neq/0', [],  true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_type_checks, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_types', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_is_atom/1, user:wam_cpp_is_int/1,
                               user:wam_cpp_is_num/1, user:wam_cpp_is_compound/1,
                               user:wam_cpp_test_nonvar/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_is_atom/1',  [foo],         true),
          run_query(BinPath, 'wam_cpp_is_atom/1',  [5],           false),
          run_query(BinPath, 'wam_cpp_is_int/1',   [5],           true),
          run_query(BinPath, 'wam_cpp_is_int/1',   [foo],         false),
          run_query(BinPath, 'wam_cpp_is_num/1',   [5],           true),
          run_query(BinPath, 'wam_cpp_is_num/1',   [foo],         false),
          run_query(BinPath, 'wam_cpp_is_compound/1', ['box(1,2)'], true),
          run_query(BinPath, 'wam_cpp_is_compound/1', [foo],        false),
          run_query(BinPath, 'wam_cpp_test_nonvar/0', [],            true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_term_inspection, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_term', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_functor/0,
                               user:wam_cpp_test_arg1/0,
                               user:wam_cpp_test_arg_bad/0,
                               user:wam_cpp_test_univ_decompose/0,
                               user:wam_cpp_test_univ_compose/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_functor/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_arg1/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_arg_bad/0',         [], false),
          run_query(BinPath, 'wam_cpp_test_univ_decompose/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_univ_compose/0',    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_io, [condition(cpp_compiler_available)]) :-
    % write/1 + nl/0 should print "hello\n" before the driver prints
    % "true". Captures full stdout (not just the last line).
    unique_cpp_tmp_dir('tmp_cpp_e2e_io', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_write/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          process_create(BinPath, ['wam_cpp_test_write/0'],
                         [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Output),
          close(Out),
          process_wait(PID, _),
          normalize_space(string(Trimmed), Output),
          assertion(Trimmed == "hello true")
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% findall/3 + aggregate_all/3 — exercises BeginAggregate / EndAggregate
% with all standard aggregate kinds (collect / count / sum / min / max / set).
% ------------------------------------------------------------------

test(cpp_e2e_findall, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_findall', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1,
                               user:wam_cpp_test_findall/0,
                               user:wam_cpp_test_findall_empty/0,
                               user:wam_cpp_test_findall_doubled/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_findall/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_findall_empty/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_findall_doubled/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_setof, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_setof', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1, user:wam_cpp_num/1,
                               user:wam_cpp_test_bagof/0,
                               user:wam_cpp_test_bagof_empty/0,
                               user:wam_cpp_test_setof/0,
                               user:wam_cpp_test_setof_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bagof/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_bagof_empty/0', [], false),
          run_query(BinPath, 'wam_cpp_test_setof/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_setof_empty/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_aggregate_all, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_agg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1, user:wam_cpp_num/1,
                               user:wam_cpp_test_count/0, user:wam_cpp_test_sum/0,
                               user:wam_cpp_test_min/0,   user:wam_cpp_test_max/0,
                               user:wam_cpp_test_set/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_count/0', [], true),
          run_query(BinPath, 'wam_cpp_test_sum/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_min/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_max/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_set/0',   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Environment frames: Y-reg isolation across nested calls + cp threading
% through tail-recursive arithmetic. Both are correctness bugs that
% existed in #2036 and are fixed by this PR''s env-frame implementation
% (Allocate pushes a frame saving cp; Deallocate pops + restores;
% Y-reg lookup is scoped to the top frame).
% ------------------------------------------------------------------

test(cpp_e2e_yreg_isolation, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_yreg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_num/1,
                               user:wam_cpp_h1/1, user:wam_cpp_h2/1,
                               user:wam_cpp_two_helpers/0,
                               user:wam_cpp_two_helpers_swap/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Both helpers use Y1/Y2 internally. The caller calls h1 then h2
          % and must NOT see h2''s Y1 stomp on h1''s result.
          run_query(BinPath, 'wam_cpp_two_helpers/0',      [], true),
          run_query(BinPath, 'wam_cpp_two_helpers_swap/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_recursive_arithmetic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_recur', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_length_acc/3,
                               user:wam_cpp_list_length/2,
                               user:wam_cpp_test_len_empty/0,
                               user:wam_cpp_test_len_one/0,
                               user:wam_cpp_test_len_three/0,
                               user:wam_cpp_test_len_five/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Tail-recursive length with accumulator. Exercises:
          %   - cp threading through nested Call/Execute
          %   - Y-reg isolation across recursive frames
          %   - PutConstant allocating fresh cells (not mutating
          %     X-reg-aliased cells)
          run_query(BinPath, 'wam_cpp_test_len_empty/0', [], true),
          run_query(BinPath, 'wam_cpp_test_len_one/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_len_three/0', [], true),
          run_query(BinPath, 'wam_cpp_test_len_five/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_unification, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_unif', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_unify/0,
                               user:wam_cpp_test_unify_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_unify/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_unify_fail/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_list_head_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_list', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_lst/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % lst([a, b, c]). Exercises GetList + UnifyConstant +
          % UnifyVariable + GetStructure([|]/2) cell-by-cell.
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b,c]'], true),
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b]'],   false),
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b,d]'], false),
          run_query(BinPath, 'wam_cpp_lst/1', ['[]'],      false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Indexing instructions: switch_on_constant (atoms / integers) +
% switch_on_term (typed dispatch with structure / list handling).
% Exercises constant-bound A1 dispatch (color, shape) and the
% combined type dispatch (mixed atom/int/struct/list).
% ------------------------------------------------------------------

% ------------------------------------------------------------------
% List & term builtins: member/2, length/2, copy_term/2. member and
% length are auto-injected as helper predicates (so they can backtrack
% naturally through their two clauses); copy_term is a direct builtin
% with structural deep-copy and shared-variable renaming.
% ------------------------------------------------------------------

test(cpp_e2e_member, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_member', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_member_yes/0,
                               user:wam_cpp_test_member_no/0,
                               user:wam_cpp_test_member_first/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_member_yes/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_member_no/0',    [], false),
          run_query(BinPath, 'wam_cpp_test_member_first/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_length, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_length', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_length_three/0,
                               user:wam_cpp_test_length_zero/0,
                               user:wam_cpp_test_length_bad/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_length_three/0', [], true),
          run_query(BinPath, 'wam_cpp_test_length_zero/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_length_bad/0',   [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_copy_term, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_copy', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_copy_basic/0,
                               user:wam_cpp_test_copy_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % copy_term(foo(X,X,Y), T) → T = foo(A,A,B) with A and B fresh.
          % The two X-positions in source must share a single fresh var
          % in the copy; Y becomes a different fresh var.
          run_query(BinPath, 'wam_cpp_test_copy_basic/0', [], true),
          run_query(BinPath, 'wam_cpp_test_copy_atom/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_member_enumeration, [condition(cpp_compiler_available)]) :-
    % findall enumerating through member is the full nondet test:
    % member must push a choice point on each match so the driver can
    % backtrack into it for the next solution.
    unique_cpp_tmp_dir('tmp_cpp_e2e_enum', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_enum_member/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_enum_member/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_switch_on_constant, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_swc', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_color/1, user:wam_cpp_shape/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % First clause via "default" fall-through.
          run_query(BinPath, 'wam_cpp_color/1', [red],    true),
          % Later clauses reached via direct switch jump (bypassing
          % try_me_else; verifies the retry_me_else no-op fix).
          run_query(BinPath, 'wam_cpp_color/1', [green],  true),
          run_query(BinPath, 'wam_cpp_color/1', [blue],   true),
          % Bound non-key: switch returns false directly.
          run_query(BinPath, 'wam_cpp_color/1', [orange], false),
          run_query(BinPath, 'wam_cpp_shape/2', [circle,   round],   true),
          run_query(BinPath, 'wam_cpp_shape/2', [square,   angular], true),
          run_query(BinPath, 'wam_cpp_shape/2', [triangle, angular], true),
          run_query(BinPath, 'wam_cpp_shape/2', [circle,   angular], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_switch_on_term, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_swt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_mixed/1, user:wam_cpp_listy/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Mixed clauses: atom, integer, structure — switch_on_term
          % dispatches by type.
          run_query(BinPath, 'wam_cpp_mixed/1', [a],          true),
          run_query(BinPath, 'wam_cpp_mixed/1', ['1'],        true),
          run_query(BinPath, 'wam_cpp_mixed/1', ['foo(x)'],   true),
          run_query(BinPath, 'wam_cpp_mixed/1', [b],          false),
          run_query(BinPath, 'wam_cpp_mixed/1', ['bar(x)'],   false),
          % List dispatch: [] takes the constant table, [_|_] takes
          % the list-pc path.
          run_query(BinPath, 'wam_cpp_listy/1', ['[]'],       true),
          run_query(BinPath, 'wam_cpp_listy/1', ['[a,b]'],    true),
          run_query(BinPath, 'wam_cpp_listy/1', [foo],        false)
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

build_e2e_binary(TmpDir, BinPath) :-
    directory_file_path(TmpDir, 'cpp', CppDir),
    directory_file_path(CppDir, 'wam_runtime.cpp', Rt),
    directory_file_path(CppDir, 'generated_program.cpp', Prog),
    directory_file_path(CppDir, 'main.cpp', Main),
    directory_file_path(CppDir, 'cpp_test', BinPath),
    process_create(path('g++'),
                   ['-std=c++17', '-O0', '-o', BinPath, Rt, Prog, Main],
                   [stderr(null), process(PID)]),
    process_wait(PID, Status),
    assertion(Status == exit(0)).

run_query(BinPath, PredKey, Args, Expected) :-
    maplist(atom_string, Args, ArgStrs),
    process_create(BinPath, [PredKey|ArgStrs],
                   [stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, _),
    normalize_space(string(Trimmed), Output),
    expected_str(Expected, ExpStr),
    assertion(Trimmed == ExpStr).

expected_str(true,  "true").
expected_str(false, "false").

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

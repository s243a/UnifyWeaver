:- encoding(utf8).
% Codegen tests for WAM-to-Python transpilation.
%
% These tests assert that the generated Python source contains the
% expected instruction literals, label mappings, and dispatch cases.
% Runtime correctness is validated separately by running the generated
% Python project.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_python_target.pl

:- use_module(library(plunit)).
:- use_module(library(filesex), [delete_directory_and_contents/1, make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_python_target').
:- use_module('../src/unifyweaver/core/target_registry').

% ============================================================================
% Registry smoke tests
% ============================================================================

:- begin_tests(wam_python_registry).

test(wam_python_registered) :-
	target_registry:registered_target(wam_python, python, _Capabilities).

test(wam_python_module) :-
	target_registry:target_module(wam_python, wam_python_target).

test(wam_python_has_wam_capability) :-
	target_registry:target_has_capability(wam_python, wam).

test(wam_python_has_trail_backtracking) :-
	target_registry:target_has_capability(wam_python, trail_backtracking).

:- end_tests(wam_python_registry).

% ============================================================================
% wam_instruction_to_python_literal/2 unit tests
% ============================================================================

:- begin_tests(wam_python_literals).

% --- New instruction clauses (get_nil, get_integer, get_float,
%     put_integer, put_float, is) ---

test(get_nil_literal) :-
	wam_python_target:wam_instruction_to_python_literal(get_nil(1), Lit),
	assertion(Lit == '("get_nil", 1)').

test(get_integer_literal) :-
	wam_python_target:wam_instruction_to_python_literal(get_integer(42, 1), Lit),
	assertion(Lit == '("get_integer", 42, 1)').

test(get_float_literal) :-
	wam_python_target:wam_instruction_to_python_literal(get_float(3.14, 2), Lit),
	assertion(Lit == '("get_float", 3.14, 2)').

test(put_integer_literal) :-
	wam_python_target:wam_instruction_to_python_literal(put_integer(7, 2), Lit),
	assertion(Lit == '("put_integer", 7, 2)').

test(put_float_literal) :-
	wam_python_target:wam_instruction_to_python_literal(put_float(2.71, 3), Lit),
	assertion(Lit == '("put_float", 2.71, 3)').

test(is_literal) :-
	wam_python_target:wam_instruction_to_python_literal(is(1, 2), Lit),
	assertion(Lit == '("is", 1, 2)').

% --- Pre-existing instructions (regression tests) ---

test(get_constant_literal) :-
	wam_python_target:wam_instruction_to_python_literal(get_constant(a, 1), Lit),
	assertion(Lit == '("get_constant", Atom("a"), 1)').

test(get_variable_literal) :-
	wam_python_target:wam_instruction_to_python_literal(get_variable(101, 1), Lit),
	assertion(Lit == '("get_variable", 101, 1)').

test(put_variable_literal) :-
	wam_python_target:wam_instruction_to_python_literal(put_variable(101, 1), Lit),
	assertion(Lit == '("put_variable", 101, 1)').

test(proceed_literal) :-
	wam_python_target:wam_instruction_to_python_literal(proceed, Lit),
	assertion(Lit == '("proceed",)').

test(call_literal) :-
	wam_python_target:wam_instruction_to_python_literal(call(foo, 2), Lit),
	assertion(Lit == '("call", "foo", 2)').

test(allocate_literal) :-
	wam_python_target:wam_instruction_to_python_literal(allocate, Lit),
	assertion(Lit == '("allocate",)').

test(deallocate_literal) :-
	wam_python_target:wam_instruction_to_python_literal(deallocate, Lit),
	assertion(Lit == '("deallocate",)').

test(try_me_else_literal) :-
	wam_python_target:wam_instruction_to_python_literal(try_me_else(clause2), Lit),
	assertion(Lit == '("try_me_else", "clause2")').

test(trust_me_literal) :-
	wam_python_target:wam_instruction_to_python_literal(trust_me, Lit),
	assertion(Lit == '("trust_me",)').

test(builtin_call_literal) :-
	wam_python_target:wam_instruction_to_python_literal(builtin_call(write, 1), Lit),
	assertion(Lit == '("builtin_call", "write", 1)').

test(call_foreign_literal) :-
	wam_python_target:wam_instruction_to_python_literal(call_foreign(numpy_dot, 3), Lit),
	assertion(Lit == '("call_foreign", "numpy_dot", 3)').

:- end_tests(wam_python_literals).

% ============================================================================
% wam_line_to_python_literal/2 — text-line parsing tests
% ============================================================================

:- begin_tests(wam_python_line_literals).

test(line_get_nil) :-
	wam_python_target:wam_line_to_python_literal(["get_nil", "1"], Lit),
	assertion(Lit == '("get_nil", 1)').

test(line_get_integer) :-
	wam_python_target:wam_line_to_python_literal(["get_integer", "42", "1"], Lit),
	assertion(Lit == '("get_integer", 42, 1)').

test(line_put_integer) :-
	wam_python_target:wam_line_to_python_literal(["put_integer", "7", "2"], Lit),
	assertion(Lit == '("put_integer", 7, 2)').

test(line_proceed) :-
	wam_python_target:wam_line_to_python_literal(["proceed"], Lit),
	assertion(Lit == '("proceed",)').

:- end_tests(wam_python_line_literals).

% ============================================================================
% Predicate compilation round-trip tests
% ============================================================================

:- begin_tests(wam_python_compile).

test(compile_fact) :-
	% Compile a trivial predicate: foo(a).
	% WAM instructions: get_constant(a, 1), proceed
	WamCode = 'foo/1:\n  get_constant a, 1\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(foo/1, WamCode, [], PythonCode),
	% Should be a non-empty string
	atom_string(PythonCode, S),
	assertion(S \== ""),
	% Should contain a def line
	assertion(sub_string(S, _, _, _, "def pred_foo_1")),
	% Should contain the get_constant instruction
	assertion(sub_string(S, _, _, _, "get_constant")).

test(compile_binary_predicate) :-
	% Compile: bar(X, Y) :- X = Y.
	WamCode = 'bar/2:\n  get_variable 101, 1\n  get_value 101, 2\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(bar/2, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "def pred_bar_2")),
	assertion(sub_string(S, _, _, _, "get_variable")),
	assertion(sub_string(S, _, _, _, "proceed")).

test(compile_generates_label_mapping) :-
	% A label line should produce a __label__ marker consumed by load_program/1.
	WamCode = 'baz/1:\n  get_constant hello, 1\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(baz/1, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, '"__label__", "baz/1"')).

test(compile_arg_setup) :-
	% Arity-2 predicate should be registered under its pred/arity key.
	WamCode = 'qux/2:\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(qux/2, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, 'raw_program["qux/2"]')),
	assertion(sub_string(S, _, _, _, '("proceed",)')).

test(compile_lowered_mode_direct_predicate) :-
	WamCode = 'foo/1:\n  get_constant a, 1\n  proceed',
	wam_python_target:compile_one_predicate([emit_mode(lowered)], foo/1-WamCode, PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "def pred_foo_1(state)")),
	assertion(sub_string(S, _, _, _, "def register_pred_foo_1(raw_program)")),
	assertion(sub_string(S, _, _, _, '("call_lowered", pred_foo_1, 1)')).

test(compile_lowered_mode_falls_back_for_nondet) :-
	WamCode = 'choice/1:\n  try_me_else choice_2\n  get_constant a, 1\n  proceed\nchoice_2:\n  trust_me\n  get_constant b, 1\n  proceed',
	wam_python_target:compile_one_predicate([emit_mode(lowered)], choice/1-WamCode, PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "def register_pred_choice_1(raw_program)")),
	assertion(sub_string(S, _, _, _, '("try_me_else", "choice_2")')),
	assertion(\+ sub_string(S, _, _, _, "def pred_choice_1(state)")).

test(compile_all_lowered_mode_build_program_uses_registrars, [nondet]) :-
	WamCode = 'foo/1:\n  get_constant a, 1\n  proceed',
	wam_python_target:compile_all_predicates([foo/1-WamCode], [emit_mode(lowered)], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "register_pred_foo_1(raw_program)")),
	assertion(sub_string(S, _, _, _, '("call_lowered", pred_foo_1, 1)')).

test(compile_all_lowered_mode_ffi_uses_registrar_prefix) :-
	wam_python_target:compile_all_predicates(
		[category_parent/2],
		[emit_mode(lowered), foreign_predicates([category_parent/2])],
		PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "def register_pred_category_parent_2(raw_program)")),
	assertion(sub_string(S, _, _, _, "register_pred_category_parent_2(raw_program)")).

test(compile_all_lowered_mode_keeps_direct_call_graph_consistent) :-
	OuterWam = 'outer/1:\n  call inner/1, 1\n  proceed',
	InnerWam = 'inner/1:\n  try_me_else inner_2\n  get_constant a, 1\n  proceed\ninner_2:\n  trust_me\n  get_constant b, 1\n  proceed',
	wam_python_target:compile_all_predicates(
		[outer/1-OuterWam, inner/1-InnerWam],
		[emit_mode(lowered)],
		PythonCode),
	atom_string(PythonCode, S),
	assertion(\+ sub_string(S, _, _, _, "def pred_outer_1(state)")),
	assertion(sub_string(S, _, _, _, 'raw_program["outer/1"] = (')),
	assertion(sub_string(S, _, _, _, '("call", "inner/1", 1)')).

:- end_tests(wam_python_compile).

% ============================================================================
% Step dispatch codegen tests
% ============================================================================

:- begin_tests(wam_python_step_dispatch).

test(step_contains_get_nil_branch) :-
	wam_python_target:compile_step_wam_to_python([], Code),
	atom_string(Code, S),
	assertion(sub_string(S, _, _, _, "get_nil")).

test(step_contains_get_integer_branch) :-
	wam_python_target:compile_step_wam_to_python([], Code),
	atom_string(Code, S),
	assertion(sub_string(S, _, _, _, "get_integer")).

test(step_contains_get_float_branch) :-
	wam_python_target:compile_step_wam_to_python([], Code),
	atom_string(Code, S),
	assertion(sub_string(S, _, _, _, "get_float")).

test(step_contains_put_integer_branch) :-
	wam_python_target:compile_step_wam_to_python([], Code),
	atom_string(Code, S),
	assertion(sub_string(S, _, _, _, "put_integer")).

test(step_contains_is_branch) :-
	wam_python_target:compile_step_wam_to_python([], Code),
	atom_string(Code, S),
	assertion(sub_string(S, _, _, _, "\"is\"")).

:- end_tests(wam_python_step_dispatch).

% ============================================================================
% Phase A: WamState layout — heap as dict, heap_len, trail_len
% ============================================================================

:- begin_tests(wam_python_phase_a).

test(wamstate_has_heap_len, [nondet]) :-
	% WamState bindings emit heap_len field
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "heap_len").

test(wamstate_has_trail_len, [nondet]) :-
	% WamState bindings emit trail_len field
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "trail_len").

test(wamstate_heap_is_dict, [nondet]) :-
	% Heap is initialised as dict, not list
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "self.heap = {}").

test(heap_put_uses_heap_len, [nondet]) :-
	% heap_put helper uses heap_len as the address counter
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "heap_len").

test(heap_trim_emitted, [nondet]) :-
	% heap_trim helper is emitted (dict-based trimming)
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "heap_trim").

:- end_tests(wam_python_phase_a).

% ============================================================================
% Phase B: Label pre-resolution — load_program, call_pc, run_wam(code, labels, ...)
% ============================================================================

:- begin_tests(wam_python_phase_b).

test(load_program_in_main, [nondet]) :-
	% generate_main_py emits a load_program call in main.py
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "load_program").

test(build_program_in_main, [nondet]) :-
	% generate_main_py populates raw_program through predicates.build_program().
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "raw_program = build_program()").

test(init_query_args_in_main, [nondet]) :-
	% main.py initialises A-register query variables from the predicate arity.
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "def _init_query_args"),
	sub_string(Code, _, _, _, "_init_query_args(query, state)").

test(run_wam_four_arg_in_main, [nondet]) :-
	% main.py calls run_wam(code, labels, query, state)
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "run_wam(code, labels").

test(main_prints_results_with_repr, [nondet]) :-
	% Static runtime does not export underscored helpers through import *.
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "repr(r)").

test(static_runtime_has_load_program, [nondet]) :-
	% The static WamRuntime.py defines the load_program function
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "def load_program").

test(static_runtime_has_resolve_instr, [nondet]) :-
	% The static WamRuntime.py defines the _resolve_instr helper
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "_resolve_instr").

test(static_runtime_has_call_pc, [nondet]) :-
	% The static WamRuntime.py handles call_pc opcode
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "call_pc").

test(static_runtime_has_call_lowered, [nondet]) :-
	% The static WamRuntime.py handles lowered predicate stubs.
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "'call_lowered'").

test(static_runtime_run_wam_four_args, [nondet]) :-
	% run_wam takes (code, labels, entry, state)
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "def run_wam(code").

:- end_tests(wam_python_phase_b).

% ============================================================================
% Phase C: Lowered emitter — deterministic detection, func naming, emit
% ============================================================================

:- use_module('../src/unifyweaver/targets/wam_python_lowered_emitter').

:- begin_tests(wam_python_lowered_emitter).

test(is_deterministic_fact) :-
	% A simple fact with get_constant + proceed is deterministic
	Instrs = [get_constant(a, "1"), proceed],
	wam_python_lowered_emitter:is_deterministic_pred_py(Instrs).

test(is_not_deterministic_with_try) :-
	% Predicates with try_me_else are NOT deterministic
	Instrs = [try_me_else(lbl), get_constant(a, "1"), proceed],
	\+ wam_python_lowered_emitter:is_deterministic_pred_py(Instrs).

test(is_not_deterministic_with_retry) :-
	% Predicates with retry_me_else are NOT deterministic
	Instrs = [retry_me_else(lbl), proceed],
	\+ wam_python_lowered_emitter:is_deterministic_pred_py(Instrs).

test(is_not_deterministic_with_trust_me) :-
	% Predicates with trust_me are NOT deterministic
	Instrs = [trust_me, proceed],
	\+ wam_python_lowered_emitter:is_deterministic_pred_py(Instrs).

test(is_not_deterministic_with_indexed_fact_call) :-
	% Indexed fact lookup may push choice points for additional values.
	Instrs = [call_indexed_atom_fact2("category_parent/2"), proceed],
	\+ wam_python_lowered_emitter:is_deterministic_pred_py(Instrs).

test(python_func_name_basic) :-
	wam_python_lowered_emitter:python_func_name(foo/2, Name),
	Name = 'pred_foo_2'.

test(python_func_name_special_chars, [nondet]) :-
	% Functors with - get sanitised to underscores
	wam_python_lowered_emitter:python_func_name('foo-bar'/1, Name),
	atom_string(Name, NameStr),
	sub_string(NameStr, _, _, _, "pred_foo_bar_1").

test(python_func_name_zero_arity) :-
	wam_python_lowered_emitter:python_func_name(hello/0, Name),
	Name = 'pred_hello_0'.

test(emit_lowered_proceed, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python('simple'/0, [proceed], [], Lines),
	Lines \= "",
	sub_string(Lines, _, _, _, "return True").

test(emit_lowered_get_constant, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python('foo'/1, [get_constant(a, "1"), proceed], [], Lines),
	Lines \= "",
	sub_string(Lines, _, _, _, "def pred_foo_1"),
	sub_string(Lines, _, _, _, "Atom").

test(emit_lowered_get_numeric_constant, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python('foo'/1, [get_constant("10", "1"), proceed], [], Lines),
	Lines \= "",
	sub_string(Lines, _, _, _, "Int(10)"),
	assertion(\+ sub_string(Lines, _, _, _, "Atom(\"10\")")).

test(emit_lowered_def_line, [nondet]) :-
	% The emitted function starts with a proper def line
	wam_python_lowered_emitter:emit_lowered_python('bar'/2, [proceed], [], Lines),
	sub_string(Lines, _, _, _, "def pred_bar_2(state)").

test(emit_lowered_fail, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python('nope'/0, [fail], [], Lines),
	sub_string(Lines, _, _, _, "return False").

test(emit_lowered_call_indexed_atom_fact2, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python(
		category_parent/2,
		[call_indexed_atom_fact2("category_parent/2"), proceed],
		[],
		Lines),
	sub_string(Lines, _, _, _, "indexed_atom_fact2"),
	sub_string(Lines, _, _, _, "category_parent/2").

test(emit_lowered_base_category_ancestor_bind, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python(
		category_ancestor/4,
		[base_category_ancestor_bind("A1", "A2", "A3", "A4")],
		[],
		Lines),
	sub_string(Lines, _, _, _, "_atom_in_cons_list"),
	sub_string(Lines, _, _, _, "Int(1)").

test(emit_lowered_recurse_category_ancestor, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python(
		category_ancestor/4,
		[recurse_category_ancestor("X1", "A2", "X2", "A4", "category_ancestor/4", "3")],
		[],
		Lines),
	sub_string(Lines, _, _, _, "pred_category_ancestor_4(state)"),
	sub_string(Lines, _, _, _, 'Compound(".", [_mid, _visited])').

test(emit_lowered_return_add1, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python(
		category_ancestor/4,
		[return_add1("A3", "X3")],
		[],
		Lines),
	sub_string(Lines, _, _, _, "_result = Int"),
	sub_string(Lines, _, _, _, "pop_environment(state)").

:- end_tests(wam_python_lowered_emitter).

% ============================================================================
% Phase D: FFI skip — is_ffi_predicate/3
% ============================================================================

:- begin_tests(wam_python_phase_d).

test(is_ffi_predicate_detected) :-
	Options = [foreign_predicates([my_pred/2])],
	wam_python_target:is_ffi_predicate(my_pred, 2, Options).

test(is_not_ffi_predicate) :-
	Options = [foreign_predicates([my_pred/2])],
	\+ wam_python_target:is_ffi_predicate(other_pred, 1, Options).

test(is_ffi_predicate_empty_list) :-
	Options = [foreign_predicates([])],
	\+ wam_python_target:is_ffi_predicate(anything, 1, Options).

test(is_ffi_predicate_no_option) :-
	% When no foreign_predicates option is given, nothing is FFI
	Options = [],
	\+ wam_python_target:is_ffi_predicate(foo, 1, Options).

test(is_ffi_predicate_multiple, [nondet]) :-
	% Multiple predicates in the foreign list
	Options = [foreign_predicates([pred_a/1, pred_b/2, pred_c/3])],
	wam_python_target:is_ffi_predicate(pred_b, 2, Options),
	\+ wam_python_target:is_ffi_predicate(pred_b, 1, Options).

:- end_tests(wam_python_phase_d).

% ============================================================================
% ITE (if/then/else) block detection and emission tests
% ============================================================================

:- begin_tests(wam_python_ite).

test(is_match_get_constant) :-
	wam_python_lowered_emitter:is_match_instr_py(get_constant(a, 1)).

test(is_match_get_nil) :-
	wam_python_lowered_emitter:is_match_instr_py(get_nil(1)).

test(is_match_get_structure) :-
	wam_python_lowered_emitter:is_match_instr_py(get_structure(f, 2)).

test(is_match_get_integer) :-
	wam_python_lowered_emitter:is_match_instr_py(get_integer(42, 1)).

test(is_match_get_float) :-
	wam_python_lowered_emitter:is_match_instr_py(get_float(3.14, 1)).

test(is_match_get_list) :-
	wam_python_lowered_emitter:is_match_instr_py(get_list(1)).

test(is_match_get_value) :-
	wam_python_lowered_emitter:is_match_instr_py(get_value(1, 2)).

test(is_not_match_put_constant) :-
	\+ wam_python_lowered_emitter:is_match_instr_py(put_constant(a, 1)).

test(is_not_match_proceed) :-
	\+ wam_python_lowered_emitter:is_match_instr_py(proceed).

test(is_not_match_call) :-
	\+ wam_python_lowered_emitter:is_match_instr_py(call(foo, 2)).

test(ite_block_detected) :-
	% Two-clause ITE: foo(a) and foo(b)
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	length(Blocks, 2).

test(ite_block_three_clauses) :-
	% Three-clause ITE: foo(a), foo(b), foo(c)
	Instrs = [get_constant(a, "1"), proceed,
	          get_constant(b, "1"), proceed,
	          get_constant(c, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	length(Blocks, 3).

test(ite_block_not_detected_single_clause) :-
	% Single clause with no branching is NOT an ITE block
	Instrs = [get_constant(a, "1"), proceed],
	\+ wam_python_lowered_emitter:is_ite_block_py(Instrs, _).

test(ite_block_with_fail_fallthrough) :-
	% Two clauses plus fail fallthrough
	Instrs = [get_constant(a, "1"), proceed,
	          get_constant(b, "1"), proceed,
	          fail],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	length(Blocks, 3).

test(emit_ite_generates_if, [nondet]) :-
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	wam_python_lowered_emitter:emit_ite_block_py('pred_foo_1', Blocks, 4, [], Lines),
	Lines \= [],
	atomic_list_concat(Lines, '\n', AllText),
	sub_string(AllText, _, _, _, "if ").

test(emit_ite_generates_elif, [nondet]) :-
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	wam_python_lowered_emitter:emit_ite_block_py('pred_foo_1', Blocks, 4, [], Lines),
	atomic_list_concat(Lines, '\n', AllText),
	sub_string(AllText, _, _, _, "elif ").

test(emit_ite_generates_else, [nondet]) :-
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	wam_python_lowered_emitter:emit_ite_block_py('pred_foo_1', Blocks, 4, [], Lines),
	atomic_list_concat(Lines, '\n', AllText),
	sub_string(AllText, _, _, _, "else:").

test(emit_ite_generates_return_true, [nondet]) :-
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	wam_python_lowered_emitter:emit_ite_block_py('pred_foo_1', Blocks, 4, [], Lines),
	atomic_list_concat(Lines, '\n', AllText),
	sub_string(AllText, _, _, _, "return True").

test(emit_lowered_ite_full, [nondet]) :-
	% Full round-trip: emit_lowered_python detects ITE and emits if/elif/else
	Instrs = [get_constant(a, "1"), proceed, get_constant(b, "1"), proceed],
	wam_python_lowered_emitter:emit_lowered_python(foo/1, Instrs, [], Code),
	sub_string(Code, _, _, _, "def pred_foo_1"),
	sub_string(Code, _, _, _, "if "),
	sub_string(Code, _, _, _, "elif "),
	sub_string(Code, _, _, _, "else:").

test(emit_ite_integer_condition, [nondet]) :-
	Instrs = [get_integer("1", "1"), proceed, get_integer("2", "1"), proceed],
	wam_python_lowered_emitter:is_ite_block_py(Instrs, Blocks),
	wam_python_lowered_emitter:emit_ite_block_py('pred_num_1', Blocks, 4, [], Lines),
	atomic_list_concat(Lines, '\n', AllText),
	sub_string(AllText, _, _, _, "Int").

:- end_tests(wam_python_ite).

% ============================================================================
% Rust-parity: indexed-fact + category-ancestor kernel instructions
% ============================================================================

:- begin_tests(wam_python_kernel_parity).

% --- Line-literal parsing for new ops ---

test(line_call_indexed_atom_fact2) :-
	wam_python_target:wam_line_to_python_literal(
		["call_indexed_atom_fact2", "category_parent/2"], Lit),
	assertion(Lit == '("call_indexed_atom_fact2", "category_parent/2")').

test(line_base_category_ancestor) :-
	wam_python_target:wam_line_to_python_literal(
		["base_category_ancestor", "A1", "A2", "A4"], Lit),
	assertion(Lit == '("base_category_ancestor", A1, A2, A4)').

test(line_base_category_ancestor_bind) :-
	wam_python_target:wam_line_to_python_literal(
		["base_category_ancestor_bind", "A1", "A2", "A3", "A4"], Lit),
	assertion(Lit == '("base_category_ancestor_bind", A1, A2, A3, A4)').

test(line_recurse_category_ancestor) :-
	wam_python_target:wam_line_to_python_literal(
		["recurse_category_ancestor", "X1", "A2", "X2", "A4", "category_ancestor/4", "3"], Lit),
	assertion(Lit == '("recurse_category_ancestor", X1, A2, X2, A4, "category_ancestor/4", 3)').

test(line_return_add1) :-
	wam_python_target:wam_line_to_python_literal(
		["return_add1", "A3", "X3"], Lit),
	assertion(Lit == '("return_add1", A3, X3)').

% --- Runtime contains the new opcodes and registration helpers ---

% Helper: get path to the static WamRuntime.py
runtime_py_path(SrcPath) :-
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath).

test(runtime_has_indexed_atom_fact2_field, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "self.indexed_atom_fact2").

test(runtime_has_indexed_weighted_edge_field, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "self.indexed_weighted_edge").

test(runtime_has_register_indexed_atom_fact2_pairs, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "def register_indexed_atom_fact2_pairs").

test(runtime_has_register_indexed_weighted_edge_triples, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "def register_indexed_weighted_edge_triples").

test(runtime_choicepoint_saves_one_indexed_argument_registers, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "state.regs[:n_args + 1]"),
	sub_string(Content, _, _, _, "state.regs[:cp.n_args + 1]").

test(runtime_choicepoint_discards_younger_frames, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "del state.stack[cp_index + 1:]"),
	sub_string(Content, _, _, _, "del state.stack[cp_index:]").

test(runtime_has_call_indexed_atom_fact2_handler, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "'call_indexed_atom_fact2'").

test(runtime_has_base_category_ancestor_handler, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "'base_category_ancestor'").

test(runtime_has_base_category_ancestor_bind_handler, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "'base_category_ancestor_bind'").

test(runtime_has_recurse_category_ancestor_pc_handler, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "'recurse_category_ancestor_pc'").

test(runtime_has_return_add1_handler, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "'return_add1'").

test(runtime_resolves_recurse_category_ancestor_to_pc, [nondet]) :-
	% _resolve_instr should rewrite recurse_category_ancestor → ..._pc
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "recurse_category_ancestor_pc").

:- end_tests(wam_python_kernel_parity).

% ============================================================================
% Lua/Rust/Haskell builtin parity guard for packaged static runtime
% ============================================================================

:- begin_tests(wam_python_builtin_parity_guard).

runtime_py_path(SrcPath) :-
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath).

test(static_runtime_has_lua_baseline_builtins, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	forall(member(Needle, [
		"member/2",
		"length/2",
		"atom/1",
		"integer/1",
		"float/1",
		"number/1",
		"compound/1",
		"var/1",
		"nonvar/1",
		"is_list/1",
		"==/2",
		"=/2",
		"\\\\=/2",
		"=:=/2",
		"=\\\\=/2",
		">/2",
		"</2",
		">=/2",
		"=</2",
		"functor/3",
		"arg/3",
		"=../2",
		"copy_term/2",
		"true/0",
		"fail/0",
		"\\\\+/1",
		"write/1",
		"display/1",
		"nl/0"
	]), sub_string(Content, _, _, _, Needle)).

test(static_runtime_naf_uses_isolated_goal_execution, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "def _goal_succeeds_once"),
	sub_string(Content, _, _, _, "return not _goal_succeeds_once(goal, state)"),
	\+ sub_string(Content, _, _, _, "default: treat unknown \\+ as success").

test(static_runtime_io_emits_output, [nondet]) :-
	runtime_py_path(P), read_file_to_string(P, Content, []),
	sub_string(Content, _, _, _, "print(_format_value(get_reg(state, 1), state), end='')"),
	sub_string(Content, _, _, _, "print()").

:- end_tests(wam_python_builtin_parity_guard).

% ============================================================================
% Generated-project E2E coverage for packaged static runtime builtins
% ============================================================================

:- begin_tests(wam_python_builtin_e2e).

test(generated_project_runs_term_builtins) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			run_generated_query(ProjectDir, 'term_demo/0', Output),
			once(sub_string(Output, _, _, _, "A1 = g")),
			once(sub_string(Output, _, _, _, "A3 = 7")),
			once(sub_string(Output, _, _, _, "A4 = f/2")),
			once(sub_string(Output, _, _, _, "A5 = 2"))
		),
		cleanup_tmp_dir(ProjectDir)).

test(generated_project_runs_copy_naf_and_io) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			run_generated_query(ProjectDir, 'copy_naf_io_demo/0', Output),
			once(sub_string(Output, _, _, _, "ok")),
			once(sub_string(Output, _, _, _, "A1 = ok")),
			once(sub_string(Output, _, _, _, "A2 = pair/2("))
		),
		cleanup_tmp_dir(ProjectDir)).

test(generated_project_runs_type_and_comparison_builtins) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			run_generated_query(ProjectDir, 'type_compare_demo/0', Output),
			once(sub_string(Output, _, _, _, "A1 = ok"))
		),
		cleanup_tmp_dir(ProjectDir)).

write_builtin_project(ProjectDir) :-
	term_builtin_wam(TermWam),
	copy_naf_io_wam(CopyNafIoWam),
	type_compare_wam(TypeCompareWam),
	wam_python_target:write_wam_python_project(
		[term_demo/0-TermWam,
		 copy_naf_io_demo/0-CopyNafIoWam,
		 type_compare_demo/0-TypeCompareWam],
		[],
		ProjectDir).

term_builtin_wam(
'term_demo/0:
  put_structure f/2, 1
  set_constant a
  set_integer 7
  put_variable 4, 2
  put_variable 5, 3
  builtin_call functor/3 3
  put_integer 2, 1
  put_structure f/2, 2
  set_constant a
  set_integer 7
  put_variable 6, 3
  builtin_call arg/3 3
  put_variable 7, 1
  put_list 2
  set_constant g
  set_nil
  builtin_call =../2 2
  proceed').

copy_naf_io_wam(
'copy_naf_io_demo/0:
  put_structure pair/2, 1
  set_variable 4
  set_value 4
  put_variable 5, 2
  builtin_call copy_term/2 2
  put_structure member/2, 1
  set_constant z
  put_list 3
  set_constant a
  set_nil
  builtin_call \\+/1 1
  put_constant ok, 1
  builtin_call write/1 1
  builtin_call nl/0 0
  proceed').

type_compare_wam(
'type_compare_demo/0:
  put_float 3.5, 1
  builtin_call float/1 1
  builtin_call number/1 1
  put_structure f/1, 1
  set_constant a
  builtin_call compound/1 1
  put_list 1
  set_constant a
  set_nil
  builtin_call is_list/1 1
  put_constant same, 1
  put_constant same, 2
  builtin_call ==/2 2
  put_integer 2, 1
  put_integer 2, 2
  builtin_call =:=/2 2
  put_integer 2, 1
  put_integer 3, 2
  builtin_call =\\=/2 2
  put_constant ok, 1
  proceed').

run_generated_query(ProjectDir, Query, Output) :-
	process_create(path(python), ['main.py', Query],
		[cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
	read_string(Out, _, Output),
	read_string(Err, _, ErrText),
	close(Out),
	close(Err),
	process_wait(Pid, Status),
	(   Status = exit(0)
	->  true
	;   format(user_error, 'generated WAM Python query ~w failed: ~w~n', [Query, ErrText]),
		fail
	).

unique_tmp_dir(Prefix, TmpDir) :-
	tmp_file(Prefix, TmpDir),
	catch(delete_directory_and_contents(TmpDir), _, true),
	make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
	catch(delete_directory_and_contents(TmpDir), _, true).

:- end_tests(wam_python_builtin_e2e).

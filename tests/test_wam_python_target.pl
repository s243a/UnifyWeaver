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
	assertion(sub_string(S, _, _, _, "def wam_foo")),
	% Should contain the get_constant instruction
	assertion(sub_string(S, _, _, _, "get_constant")).

test(compile_binary_predicate) :-
	% Compile: bar(X, Y) :- X = Y.
	WamCode = 'bar/2:\n  get_variable 101, 1\n  get_value 101, 2\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(bar/2, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "def wam_bar")),
	assertion(sub_string(S, _, _, _, "get_variable")),
	assertion(sub_string(S, _, _, _, "proceed")).

test(compile_generates_label_mapping) :-
	% A label line should produce a labels[] assignment
	WamCode = 'baz/1:\n  get_constant hello, 1\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(baz/1, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "state.labels")).

test(compile_arg_setup) :-
	% Arity-2 predicate should set up a1, a2 arguments
	WamCode = 'qux/2:\n  proceed',
	wam_python_target:compile_wam_predicate_to_python(qux/2, WamCode, [], PythonCode),
	atom_string(PythonCode, S),
	assertion(sub_string(S, _, _, _, "a1")),
	assertion(sub_string(S, _, _, _, "a2")).

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

test(run_wam_four_arg_in_main, [nondet]) :-
	% main.py calls run_wam(code, labels, query, state)
	wam_python_target:generate_main_py([], test_mod, Code),
	sub_string(Code, _, _, _, "run_wam(code, labels").

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

test(emit_lowered_def_line, [nondet]) :-
	% The emitted function starts with a proper def line
	wam_python_lowered_emitter:emit_lowered_python('bar'/2, [proceed], [], Lines),
	sub_string(Lines, _, _, _, "def pred_bar_2(state)").

test(emit_lowered_fail, [nondet]) :-
	wam_python_lowered_emitter:emit_lowered_python('nope'/0, [fail], [], Lines),
	sub_string(Lines, _, _, _, "return False").

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

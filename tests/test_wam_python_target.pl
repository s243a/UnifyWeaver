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
:- use_module('../src/unifyweaver/core/prolog_term_parser').

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
	sub_string(Code, _, _, _, "self.heap: Dict[int, Term] = {}").

test(heap_put_uses_heap_len, [nondet]) :-
	% heap_put helper uses heap_len as the address counter
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "heap_len").

test(heap_trim_emitted, [nondet]) :-
	% heap_trim helper is emitted (dict-based trimming)
	wam_python_target:compile_wam_runtime_to_python([], Code),
	sub_string(Code, _, _, _, "heap_trim").

test(compile_runtime_reads_static_runtime_source) :-
	wam_python_target:compile_wam_runtime_to_python([], Code),
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, StaticCode, []),
	assertion(Code == StaticCode).

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

test(static_runtime_has_iso_error_helpers, [nondet]) :-
	source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
	file_directory_name(ThisFile, ThisDir),
	directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
	read_file_to_string(SrcPath, Content, []),
	sub_string(Content, _, _, _, "def make_instantiation_error(state: WamState)"),
	sub_string(Content, _, _, _, "def make_type_error(state: WamState, expected: str, culprit: Term)"),
	sub_string(Content, _, _, _, "def make_domain_error(state: WamState, domain: str, culprit: Term)"),
	sub_string(Content, _, _, _, "def make_evaluation_error(state: WamState, kind: str)"),
	sub_string(Content, _, _, _, "def throw_iso_error(state: WamState, err_term: Term)"),
	sub_string(Content, _, _, _, "def _execute_is_lax(state: WamState)"),
	sub_string(Content, _, _, _, "def _execute_is_iso(state: WamState)").


:- end_tests(wam_python_phase_b).

% ============================================================================
% ISO error config/rewrite plumbing
% ============================================================================

:- begin_tests(wam_python_iso_errors_config).

test(iso_errors_config_loader_basic) :-
	iso_errors_temp_config_file(Path, [
		'iso_errors_default(true).',
		'iso_errors_override(legacy_lookup/3, false).',
		'iso_errors_override(experimental:my_pred/2, true).',
		'ignored_fact(ok).'
	]),
	setup_call_cleanup(
		true,
		(   wam_python_target:iso_errors_load_config(Path, Config),
			assertion(Config == iso_config(true,
				[legacy_lookup/3-false,
				 (experimental:my_pred/2)-true])),
			wam_python_target:iso_errors_mode_for(Config, user:legacy_lookup/3, M1),
			assertion(M1 == false),
			wam_python_target:iso_errors_mode_for(Config, user:never_listed/2, M2),
			assertion(M2 == true),
			wam_python_target:iso_errors_mode_for(Config, experimental:my_pred/2, M3),
			assertion(M3 == true)
		),
		delete_file(Path)).

test(iso_errors_inline_wins_over_file) :-
	iso_errors_temp_config_file(Path, [
		'iso_errors_default(false).',
		'iso_errors_override(legacy_lookup/3, false).'
	]),
	setup_call_cleanup(
		true,
		(   wam_python_target:iso_errors_resolve_options(
				[iso_errors_config(Path), iso_errors(true), iso_errors(legacy_lookup/3, true)],
				Config),
			wam_python_target:iso_errors_mode_for(Config, user:legacy_lookup/3, M1),
			assertion(M1 == true),
			wam_python_target:iso_errors_mode_for(Config, user:never_listed/2, M2),
			assertion(M2 == true)
		),
		delete_file(Path)).

test(iso_errors_text_rewrite_uses_is_key_tables) :-
	Wam0 = 'demo/0:\n  put_structure is/2, 1\n  builtin_call is/2 2\n  call is/2, 2\n  execute is/2',
	wam_python_target:iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
	assertion(sub_string(IsoWam, _, _, _, "put_structure is_iso/2, 1")),
	assertion(sub_string(IsoWam, _, _, _, "builtin_call is_iso/2 2")),
	assertion(sub_string(IsoWam, _, _, _, "call is_iso/2, 2")),
	assertion(sub_string(IsoWam, _, _, _, "execute is_iso/2")),
	wam_python_target:iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
	assertion(sub_string(LaxWam, _, _, _, "builtin_call is_lax/2 2")),
	Cmp0 = 'cmp/0:\n  builtin_call >/2 2\n  builtin_call =:=/2 2',
	wam_python_target:iso_errors_rewrite_text(iso_config(true, []), cmp/0, Cmp0, CmpIso),
	assertion(sub_string(CmpIso, _, _, _, "builtin_call >_iso/2 2")),
	assertion(sub_string(CmpIso, _, _, _, "builtin_call =:=_iso/2 2")),
	wam_python_target:iso_errors_rewrite_text(iso_config(false, []), cmp/0, Cmp0, CmpLax),
	assertion(sub_string(CmpLax, _, _, _, "builtin_call >_lax/2 2")),
	assertion(sub_string(CmpLax, _, _, _, "builtin_call =:=_lax/2 2")),
	Succ0 = 'succ_demo/0:\n  builtin_call succ/2 2',
	wam_python_target:iso_errors_rewrite_text(iso_config(true, []), succ_demo/0, Succ0, SuccIso),
	assertion(sub_string(SuccIso, _, _, _, "builtin_call succ_iso/2 2")),
	wam_python_target:iso_errors_rewrite_text(iso_config(false, []), succ_demo/0, Succ0, SuccLax),
	assertion(sub_string(SuccLax, _, _, _, "builtin_call succ_lax/2 2")).

test(iso_errors_project_generation_rewrites_wam_text) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_iso_rewrite', ProjectDir),
		(   Wam = 'iso_rewrite_demo/0:\n  put_variable 4, 1\n  put_integer 1, 2\n  builtin_call is/2 2\n  proceed',
			wam_python_target:write_wam_python_project([iso_rewrite_demo/0-Wam], [iso_errors(true)], ProjectDir),
			directory_file_path(ProjectDir, 'predicates.py', PredicatesPath),
			read_file_to_string(PredicatesPath, Code, []),
			assertion(sub_string(Code, _, _, _, '"is_iso/2", 2')),
			\+ sub_string(Code, _, _, _, '"is/2", 2')
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(iso_errors_audit_structure) :-
	wam_python_target:wam_python_iso_audit(
		[user:py_iso_audit_pred/0],
		[iso_errors(py_iso_audit_pred/0, true)],
		Audit),
	assertion(Audit = [audit(user:py_iso_audit_pred/0, true, _Sites)]).

:- dynamic user:py_iso_audit_pred/0.
user:py_iso_audit_pred :- X is 1 + 2, X = 3.

iso_errors_temp_config_file(Path, Lines) :-
	tmp_file('tmp_wam_python_iso_cfg', Path),
	setup_call_cleanup(
		open(Path, write, Out),
		forall(member(L, Lines), format(Out, '~w~n', [L])),
		close(Out)).

:- end_tests(wam_python_iso_errors_config).

% ============================================================================
% Runtime parser capability metadata and validation
% ============================================================================

python_parser_tmp_dir(Prefix, TmpDir) :-
	tmp_file(Prefix, TmpDir),
	catch(delete_directory_and_contents(TmpDir), _, true),
	make_directory_path(TmpDir).

python_parser_cleanup_tmp_dir(TmpDir) :-
	catch(delete_directory_and_contents(TmpDir), _, true).

python_parser_predicates(Predicates) :-
	findall(prolog_term_parser:Name/Arity,
		(   current_predicate(prolog_term_parser:Name/Arity),
			functor(Head, Name, Arity),
			once(clause(prolog_term_parser:Head, _)),
			\+ predicate_property(prolog_term_parser:Head, imported_from(_))
		),
		Raw),
	sort(Raw, Predicates).

python_parser_run_snippet(ProjectDir, Script, Output) :-
	process_create(path(python), ['-c', Script],
		[cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
	read_string(Out, _, Output),
	read_string(Err, _, ErrText),
	close(Out),
	close(Err),
	process_wait(Pid, Status),
	(   Status = exit(0)
	->  true
	;   format(user_error, 'generated WAM Python parser snippet failed: ~w~n', [ErrText]),
		fail
	).

:- begin_tests(wam_python_runtime_parser_mode).

test(runtime_parser_mode_metadata) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_mode', ProjectDir),
		(   wam_python_target:write_wam_python_project([user:py_parser_fact/1], [], ProjectDir),
			directory_file_path(ProjectDir, 'predicates.py', PredicatesPath),
			read_file_to_string(PredicatesPath, Code, []),
			assertion(sub_string(Code, _, _, _,
				'RUNTIME_PARSER = {"kind": "none", "entry": None, "source": None}'))
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(runtime_parser_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_python), native))]) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_native_err', ProjectDir),
		wam_python_target:write_wam_python_project([user:py_parser_fact/1],
			[runtime_parser(native)],
			ProjectDir),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(runtime_parser_compiled_includes_portable_parser) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_compiled_meta', ProjectDir),
		(   wam_python_target:write_wam_python_project([user:py_parser_fact/1],
				[runtime_parser(compiled)],
				ProjectDir),
			directory_file_path(ProjectDir, 'predicates.py', PredicatesPath),
			read_file_to_string(PredicatesPath, Code, []),
			assertion(sub_string(Code, _, _, _,
				'RUNTIME_PARSER = {"kind": "compiled", "entry": None, "source": "prolog_term_parser"}')),
			assertion(sub_string(Code, _, _, _, 'parse_term_from_atom/3'))
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(runtime_parser_compiled_runs_read_term_from_atom) :-
	setup_call_cleanup(
		(   retractall(user:py_read_term_demo),
			assertz((user:py_read_term_demo :-
				read_term_from_atom('p(a)', T),
				T = p(a))),
			user:python_parser_tmp_dir('tmp_wam_python_parser_read_term', ProjectDir)
		),
		(   wam_python_target:write_wam_python_project([user:py_read_term_demo/0],
				[runtime_parser(compiled)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"state = wr.WamState()",
				"print(wr.run_wam(code, labels, 'py_read_term_demo/0', state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "True"))
		),
		(   retractall(user:py_read_term_demo),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_compiled_read_term_variable_names) :-
	setup_call_cleanup(
		(   retractall(user:py_read_term_vars_demo),
			assertz((user:py_read_term_vars_demo :-
				read_term_from_atom('p(A,B,A)', T, [variable_names(Vs)]),
				T = p(X, Y, X),
				Vs = ['A'=X, 'B'=Y])),
			user:python_parser_tmp_dir('tmp_wam_python_parser_read_vars', ProjectDir)
		),
		(   wam_python_target:write_wam_python_project([user:py_read_term_vars_demo/0],
				[runtime_parser(compiled)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"state = wr.WamState()",
				"print(wr.run_wam(code, labels, 'py_read_term_vars_demo/0', state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "True"))
		),
		(   retractall(user:py_read_term_vars_demo),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_compiled_read_term_variables) :-
	setup_call_cleanup(
		(   retractall(user:py_read_term_variables_demo),
			assertz((user:py_read_term_variables_demo :-
				read_term_from_atom('p(A,_,B,A)', T,
					[variables(Vars), variable_names(Vs)]),
				T = p(X, Anon, Y, X),
				Vars = [X, Anon, Y],
				Vs = ['A'=X, 'B'=Y])),
			user:python_parser_tmp_dir('tmp_wam_python_parser_read_variables', ProjectDir)
		),
		(   wam_python_target:write_wam_python_project([user:py_read_term_variables_demo/0],
				[runtime_parser(compiled)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"state = wr.WamState()",
				"print(wr.run_wam(code, labels, 'py_read_term_variables_demo/0', state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "True"))
		),
		(   retractall(user:py_read_term_variables_demo),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_compiled_read_term_singletons) :-
	setup_call_cleanup(
		(   retractall(user:py_read_term_singletons_demo),
			assertz((user:py_read_term_singletons_demo :-
				read_term_from_atom('p(A,B,A,_,_C,_C,D)', T,
					[variables(Vars), variable_names(Vs), singletons(Ss)]),
				T = p(X, Y, X, Anon, Z, Z, W),
				Vars = [X, Y, Anon, Z, W],
				Vs = ['A'=X, 'B'=Y, '_C'=Z, 'D'=W],
				Ss = ['B'=Y, '_'=Anon, 'D'=W])),
			user:python_parser_tmp_dir('tmp_wam_python_parser_read_singletons', ProjectDir)
		),
		(   wam_python_target:write_wam_python_project([user:py_read_term_singletons_demo/0],
				[runtime_parser(compiled)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"state = wr.WamState()",
				"print(wr.run_wam(code, labels, 'py_read_term_singletons_demo/0', state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "True"))
		),
		(   retractall(user:py_read_term_singletons_demo),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_compiled_runs_reverse_term_to_atom) :-
	setup_call_cleanup(
		(   retractall(user:py_term_to_atom_demo),
			assertz((user:py_term_to_atom_demo :-
				term_to_atom(T, 'p(a)'),
				T = p(a))),
			user:python_parser_tmp_dir('tmp_wam_python_parser_term_to_atom', ProjectDir)
		),
		(   wam_python_target:write_wam_python_project([user:py_term_to_atom_demo/0],
				[runtime_parser(compiled)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"state = wr.WamState()",
				"print(wr.run_wam(code, labels, 'py_term_to_atom_demo/0', state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "True"))
		),
		(   retractall(user:py_term_to_atom_demo),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_none_rejects_parser_dependent_builtin,
     [error(permission_error(use, runtime_parser, read_term_from_atom/2))]) :-
	setup_call_cleanup(
		(   retractall(user:py_parser_dep),
			assertz((user:py_parser_dep :-
				read_term_from_atom('f(a)', _))),
			user:python_parser_tmp_dir('tmp_wam_python_parser_reject', ProjectDir)
		),
		wam_python_target:write_wam_python_project([user:py_parser_dep/0], [], ProjectDir),
		(   retractall(user:py_parser_dep),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(runtime_parser_none_allows_term_to_atom_forward) :-
	setup_call_cleanup(
		(   retractall(user:py_t2a_forward),
			assertz((user:py_t2a_forward :-
				term_to_atom(f(a), _))),
			user:python_parser_tmp_dir('tmp_wam_python_parser_t2a_fwd', ProjectDir)
		),
		wam_python_target:write_wam_python_project([user:py_t2a_forward/0], [], ProjectDir),
		(   retractall(user:py_t2a_forward),
			user:python_parser_cleanup_tmp_dir(ProjectDir)
		)).

test(portable_parser_predicates_compile_to_python_project) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_compile', ProjectDir),
		(   user:python_parser_predicates(Predicates),
			wam_python_target:write_wam_python_project(Predicates, [runtime_parser(off)], ProjectDir),
			user:python_parser_run_snippet(ProjectDir,
				"import py_compile; py_compile.compile('wam_runtime.py', doraise=True); py_compile.compile('predicates.py', doraise=True); print('ok')",
				Output),
			once(sub_string(Output, _, _, _, "ok"))
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(portable_parser_tokenize_runs_under_python_wam) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_tokenize', ProjectDir),
		(   user:python_parser_predicates(Predicates),
			wam_python_target:write_wam_python_project(Predicates, [runtime_parser(off)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"for text in ('foo', 'A = 1'):",
				"    state = wr.WamState()",
				"    out = wr.Var([None], 1)",
				"    wr.set_reg(state, 1, wr._list_from_codes([ord(ch) for ch in text]))",
				"    wr.set_reg(state, 2, out)",
				"    ok = wr.run_wam(code, labels, 'tokenize/2', state)",
				"    print(text, ok, wr._format_value(wr.deref(out, state), state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "foo True [tk_atom(foo)]")),
			once(sub_string(Output, _, _, _, "A = 1 True [tk_var(A), tk_sym(=), tk_num(1)]"))
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

test(portable_parser_parse_atom_and_compound_under_python_wam) :-
	setup_call_cleanup(
		user:python_parser_tmp_dir('tmp_wam_python_parser_parse', ProjectDir),
		(   user:python_parser_predicates(Predicates),
			wam_python_target:write_wam_python_project(Predicates, [runtime_parser(off)], ProjectDir),
			atomic_list_concat([
				"import predicates as p, wam_runtime as wr",
				"code, labels = wr.load_program(p.build_program())",
				"for text in ('foo', 'p(a, 1)', '[a,1]', 'A = 1', '1+2*3'):",
				"    state = wr.WamState()",
				"    ops = wr.Var([None], 1)",
				"    wr.set_reg(state, 1, ops)",
				"    assert wr.run_wam(code, labels, 'canonical_op_table/1', state)",
				"    ops_value = wr.deref(ops, state)",
				"    out = wr.Var([None], 3)",
				"    wr.set_reg(state, 1, wr.make_atom(text))",
				"    wr.set_reg(state, 2, ops_value)",
				"    wr.set_reg(state, 3, out)",
				"    ok = wr.run_wam(code, labels, 'parse_term_from_atom/3', state)",
				"    print(text, ok, wr._format_value(wr.deref(out, state), state))"
			], '\n', Script),
			user:python_parser_run_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "foo True foo")),
			once(sub_string(Output, _, _, _, "p(a, 1) True p(a, 1)")),
			once(sub_string(Output, _, _, _, "[a,1] True [a, 1]")),
			once(sub_string(Output, _, _, _, "A = 1 True =(")),
			once(sub_string(Output, _, _, _, "1+2*3 True +(1, *(2, 3))"))
		),
		user:python_parser_cleanup_tmp_dir(ProjectDir)).

:- end_tests(wam_python_runtime_parser_mode).

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

test(generated_project_enumerates_member_builtin) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			run_generated_query(ProjectDir, 'member_collect_demo/0', Output),
			once(sub_string(Output, _, _, _, "A3 = .(a, .(b, []))"))
		),
		cleanup_tmp_dir(ProjectDir)).

test(generated_project_runs_catch_throw_builtins) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			run_generated_query(ProjectDir, 'catch_match_demo/0', MatchOutput),
			once(sub_string(MatchOutput, _, _, _, "caught")),
			run_generated_query(ProjectDir, 'catch_compound_demo/0', CompoundOutput),
			once(sub_string(CompoundOutput, _, _, _, "type_error")),
			run_generated_query(ProjectDir, 'catch_no_throw_demo/0', NoThrowOutput),
			once(sub_string(NoThrowOutput, _, _, _, "ok")),
			\+ sub_string(NoThrowOutput, _, _, _, "bad")
		),
		cleanup_tmp_dir(ProjectDir)).

test(generated_runtime_catches_iso_helper_error_terms) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			iso_helper_catch_script(Script),
			run_python_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "foo")),
			once(sub_string(Output, _, _, _, "zero_divisor"))
		),
		cleanup_tmp_dir(ProjectDir)).

test(generated_runtime_runs_is_iso_and_lax_variants) :-
	setup_call_cleanup(
		unique_tmp_dir('tmp_wam_python_builtin_e2e', ProjectDir),
		(   write_builtin_project(ProjectDir),
			is_iso_lax_script(Script),
			run_python_snippet(ProjectDir, Script, Output),
			once(sub_string(Output, _, _, _, "type")),
			once(sub_string(Output, _, _, _, "inst")),
			once(sub_string(Output, _, _, _, "zero")),
			once(sub_string(Output, _, _, _, "cmp_inst")),
			once(sub_string(Output, _, _, _, "cmp_type")),
			once(sub_string(Output, _, _, _, "cmp_zero")),
			once(sub_string(Output, _, _, _, "succ_inst")),
			once(sub_string(Output, _, _, _, "succ_type")),
			once(sub_string(Output, _, _, _, "succ_domain")),
			once(sub_string(Output, _, _, _, "float_zero_iso")),
			once(sub_string(Output, _, _, _, "lax_inf")),
			once(sub_string(Output, _, _, _, "lax_nan")),
			once(sub_string(Output, _, _, _, "lax_neg_inf")),
			once(sub_string(Output, _, _, _, "lax_int_zero_false")),
			once(sub_string(Output, _, _, _, "succ_forward")),
			once(sub_string(Output, _, _, _, "succ_backward")),
			once(sub_string(Output, _, _, _, "lax_false")),
			once(sub_string(Output, _, _, _, "cmp_lax_false")),
			once(sub_string(Output, _, _, _, "succ_lax_false"))
		),
		cleanup_tmp_dir(ProjectDir)).

write_builtin_project(ProjectDir) :-
	term_builtin_wam(TermWam),
	copy_naf_io_wam(CopyNafIoWam),
	type_compare_wam(TypeCompareWam),
	member_collect_wam(MemberCollectWam),
	catch_match_wam(CatchMatchWam),
	catch_compound_wam(CatchCompoundWam),
	catch_no_throw_wam(CatchNoThrowWam),
	wam_python_target:write_wam_python_project(
		[term_demo/0-TermWam,
		 copy_naf_io_demo/0-CopyNafIoWam,
		 type_compare_demo/0-TypeCompareWam,
		 member_collect_demo/0-MemberCollectWam,
		 catch_match_demo/0-CatchMatchWam,
		 catch_compound_demo/0-CatchCompoundWam,
		 catch_no_throw_demo/0-CatchNoThrowWam],
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

member_collect_wam(
'member_collect_demo/0:
  put_variable 8, 3
  begin_aggregate collect 1 3
  put_variable 6, 1
  put_list 4
  set_constant b
  set_nil
  put_list 2
  set_constant a
  set_value 4
  builtin_call member/2 2
  end_aggregate 1
  proceed').

catch_match_wam(
'catch_match_demo/0:
  allocate
  put_structure throw/1, 1
  set_constant foo
  put_constant foo, 2
  put_structure write/1, 3
  set_constant caught
  deallocate
  execute catch/3').

catch_compound_wam(
'catch_compound_demo/0:
  allocate
  put_structure error/2, 4
  set_constant type_error
  set_constant ctx
  put_structure throw/1, 1
  set_value 4
  put_structure error/2, 2
  set_variable 104
  set_variable 105
  put_structure write/1, 3
  set_value 104
  deallocate
  execute catch/3').

catch_no_throw_wam(
'catch_no_throw_demo/0:
  allocate
  put_structure write/1, 1
  set_constant ok
  put_constant foo, 2
  put_structure write/1, 3
  set_constant bad
  deallocate
  execute catch/3').

iso_helper_catch_script(Script) :-
	atomic_list_concat([
		"import wam_runtime as wr",
		"",
		"def run_type_error():",
		"    s = wr.WamState()",
		"    wr.set_reg(s, 1, wr.Compound('throw_iso_type', []))",
		"    wr.set_reg(s, 2, wr.Compound('error/2', [wr.Compound('type_error/2', [wr.make_atom('evaluable'), s.fresh_var()]), s.fresh_var()]))",
		"    wr.set_reg(s, 3, wr.Compound('write/1', [wr.get_reg(s, 2).args[0].args[1]]))",
		"    def builtin(name, arity, state, resume_ip=-1):",
		"        if name in ('throw_iso_type', 'throw_iso_type/0'):",
		"            return wr.throw_iso_error(state, wr.make_type_error(state, 'evaluable', wr.make_atom('foo')))",
		"        return old_builtin(name, arity, state, resume_ip)",
		"    wr._execute_builtin = builtin",
		"    wr._execute_catch(s)",
		"",
		"def run_eval_error():",
		"    s = wr.WamState()",
		"    wr.set_reg(s, 1, wr.Compound('throw_iso_eval', []))",
		"    wr.set_reg(s, 2, wr.Compound('error/2', [wr.Compound('evaluation_error/1', [s.fresh_var()]), s.fresh_var()]))",
		"    wr.set_reg(s, 3, wr.Compound('write/1', [wr.get_reg(s, 2).args[0].args[0]]))",
		"    def builtin(name, arity, state, resume_ip=-1):",
		"        if name in ('throw_iso_eval', 'throw_iso_eval/0'):",
		"            return wr.throw_iso_error(state, wr.make_evaluation_error(state, 'zero_divisor'))",
		"        return old_builtin(name, arity, state, resume_ip)",
		"    wr._execute_builtin = builtin",
		"    wr._execute_catch(s)",
		"",
		"old_builtin = wr._execute_builtin",
		"run_type_error()",
		"print()",
		"run_eval_error()",
		""
	], '\n', Script).

is_iso_lax_script(Script) :-
	atomic_list_concat([
		"import wam_runtime as wr",
		"",
		"def catch_is(goal, catcher, recovery):",
		"    s = wr.WamState()",
		"    wr.set_reg(s, 1, goal)",
		"    wr.set_reg(s, 2, catcher)",
		"    wr.set_reg(s, 3, recovery)",
		"    return wr._execute_catch(s)",
		"",
		"v = wr.Var([None], 100)",
		"catch_is(wr.Compound('is_iso/2', [v, wr.make_atom('foo')]), wr.Compound('error/2', [wr.Compound('type_error/2', [wr.make_atom('evaluable'), wr.Var([None], 101)]), wr.Var([None], 102)]), wr.Compound('write/1', [wr.make_atom('type')]))",
		"print()",
		"v = wr.Var([None], 200)",
		"expr = wr.Compound('+/2', [wr.Var([None], 201), wr.Int(1)])",
		"catch_is(wr.Compound('is_iso/2', [v, expr]), wr.Compound('error/2', [wr.make_atom('instantiation_error'), wr.Var([None], 202)]), wr.Compound('write/1', [wr.make_atom('inst')]))",
		"print()",
		"v = wr.Var([None], 300)",
		"expr = wr.Compound('//2', [wr.Int(1), wr.Int(0)])",
		"catch_is(wr.Compound('is_iso/2', [v, expr]), wr.Compound('error/2', [wr.Compound('evaluation_error/1', [wr.make_atom('zero_divisor')]), wr.Var([None], 301)]), wr.Compound('write/1', [wr.make_atom('zero')]))",
		"print()",
		"catch_is(wr.Compound('>_iso/2', [wr.Var([None], 500), wr.Int(5)]), wr.Compound('error/2', [wr.make_atom('instantiation_error'), wr.Var([None], 501)]), wr.Compound('write/1', [wr.make_atom('cmp_inst')]))",
		"print()",
		"catch_is(wr.Compound('<_iso/2', [wr.make_atom('foo'), wr.Int(5)]), wr.Compound('error/2', [wr.Compound('type_error/2', [wr.make_atom('evaluable'), wr.Var([None], 502)]), wr.Var([None], 503)]), wr.Compound('write/1', [wr.make_atom('cmp_type')]))",
		"print()",
		"expr = wr.Compound('//2', [wr.Int(1), wr.Int(0)])",
		"catch_is(wr.Compound('=:=_iso/2', [expr, wr.Int(0)]), wr.Compound('error/2', [wr.Compound('evaluation_error/1', [wr.make_atom('zero_divisor')]), wr.Var([None], 504)]), wr.Compound('write/1', [wr.make_atom('cmp_zero')]))",
		"print()",
		"catch_is(wr.Compound('succ_iso/2', [wr.Var([None], 700), wr.Var([None], 701)]), wr.Compound('error/2', [wr.make_atom('instantiation_error'), wr.Var([None], 702)]), wr.Compound('write/1', [wr.make_atom('succ_inst')]))",
		"print()",
		"catch_is(wr.Compound('succ_iso/2', [wr.make_atom('foo'), wr.Var([None], 703)]), wr.Compound('error/2', [wr.Compound('type_error/2', [wr.make_atom('integer'), wr.Var([None], 704)]), wr.Var([None], 705)]), wr.Compound('write/1', [wr.make_atom('succ_type')]))",
		"print()",
		"catch_is(wr.Compound('succ_iso/2', [wr.Var([None], 706), wr.Int(0)]), wr.Compound('error/2', [wr.Compound('domain_error/2', [wr.make_atom('not_less_than_zero'), wr.Var([None], 707)]), wr.Var([None], 708)]), wr.Compound('write/1', [wr.make_atom('succ_domain')]))",
		"print()",
		"expr = wr.Compound('//2', [wr.Float(1.0), wr.Float(0.0)])",
		"catch_is(wr.Compound('is_iso/2', [wr.Var([None], 709), expr]), wr.Compound('error/2', [wr.Compound('evaluation_error/1', [wr.make_atom('zero_divisor')]), wr.Var([None], 712)]), wr.Compound('write/1', [wr.make_atom('float_zero_iso')]))",
		"print()",
		"s = wr.WamState()",
		"v = wr.Var([None], 713)",
		"wr.set_reg(s, 1, v)",
		"wr.set_reg(s, 2, wr.Compound('//2', [wr.Float(1.0), wr.Float(0.0)]))",
		"ok = wr._execute_builtin('is_lax/2', 2, s) and isinstance(wr.deref(v, s), wr.Float) and wr.deref(v, s).f == float('inf')",
		"print('lax_inf' if ok else 'bad')",
		"s = wr.WamState()",
		"v = wr.Var([None], 714)",
		"wr.set_reg(s, 1, v)",
		"wr.set_reg(s, 2, wr.Compound('//2', [wr.Float(0.0), wr.Float(0.0)]))",
		"ok = wr._execute_builtin('is_lax/2', 2, s) and isinstance(wr.deref(v, s), wr.Float) and wr.deref(v, s).f != wr.deref(v, s).f",
		"print('lax_nan' if ok else 'bad')",
		"s = wr.WamState()",
		"v = wr.Var([None], 715)",
		"wr.set_reg(s, 1, v)",
		"wr.set_reg(s, 2, wr.Compound('//2', [wr.Float(-1.0), wr.Float(0.0)]))",
		"ok = wr._execute_builtin('is_lax/2', 2, s) and isinstance(wr.deref(v, s), wr.Float) and wr.deref(v, s).f == float('-inf')",
		"print('lax_neg_inf' if ok else 'bad')",
		"s = wr.WamState()",
		"v = wr.Var([None], 716)",
		"wr.set_reg(s, 1, v)",
		"wr.set_reg(s, 2, wr.Compound('//2', [wr.Int(1), wr.Int(0)]))",
		"print('lax_int_zero_false' if not wr._execute_builtin('is_lax/2', 2, s) else 'bad')",
		"s = wr.WamState()",
		"v = wr.Var([None], 710)",
		"wr.set_reg(s, 1, wr.Int(2))",
		"wr.set_reg(s, 2, v)",
		"ok = wr._execute_builtin('succ/2', 2, s) and isinstance(wr.deref(v, s), wr.Int) and wr.deref(v, s).n == 3",
		"print('succ_forward' if ok else 'bad')",
		"s = wr.WamState()",
		"v = wr.Var([None], 711)",
		"wr.set_reg(s, 1, v)",
		"wr.set_reg(s, 2, wr.Int(3))",
		"ok = wr._execute_builtin('succ/2', 2, s) and isinstance(wr.deref(v, s), wr.Int) and wr.deref(v, s).n == 2",
		"print('succ_backward' if ok else 'bad')",
		"s = wr.WamState()",
		"wr.set_reg(s, 1, wr.Var([None], 400))",
		"wr.set_reg(s, 2, wr.make_atom('foo'))",
		"print('lax_false' if not wr._execute_builtin('is_lax/2', 2, s) else 'bad')",
		"s = wr.WamState()",
		"wr.set_reg(s, 1, wr.Var([None], 600))",
		"wr.set_reg(s, 2, wr.Int(5))",
		"print('cmp_lax_false' if not wr._execute_builtin('>_lax/2', 2, s) else 'bad')",
		"s = wr.WamState()",
		"wr.set_reg(s, 1, wr.Int(-1))",
		"wr.set_reg(s, 2, wr.Var([None], 800))",
		"print('succ_lax_false' if not wr._execute_builtin('succ_lax/2', 2, s) else 'bad')",
		""
	], '\n', Script).

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

run_python_snippet(ProjectDir, Script, Output) :-
	process_create(path(python), ['-c', Script],
		[cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
	read_string(Out, _, Output),
	read_string(Err, _, ErrText),
	close(Out),
	close(Err),
	process_wait(Pid, Status),
	(   Status = exit(0)
	->  true
	;   format(user_error, 'generated WAM Python snippet failed: ~w~n', [ErrText]),
		fail
	).

unique_tmp_dir(Prefix, TmpDir) :-
	tmp_file(Prefix, TmpDir),
	catch(delete_directory_and_contents(TmpDir), _, true),
	make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
	catch(delete_directory_and_contents(TmpDir), _, true).

:- end_tests(wam_python_builtin_e2e).

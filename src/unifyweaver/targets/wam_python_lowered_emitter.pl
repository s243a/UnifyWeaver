:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_python_lowered_emitter.pl - WAM-to-Python Lowered Emitter
%
% Compiles deterministic (single-clause, no choice points) WAM predicates
% into direct Python functions instead of routing through the run_wam
% dispatch loop. This is the highest-impact optimisation — the Haskell
% equivalent reduced query time from 2518ms to ~193ms.
%
% Design modelled on wam_fsharp_lowered_emitter.pl (646 lines) and
% wam_elixir_lowered_emitter.pl (492 lines).
%
% Key design points:
%   - `call` instructions become direct Python function calls
%   - `execute` (tail position) becomes `return pred_foo_2(state)`
%   - `proceed` becomes `return True`
%   - `fail` becomes `return False`
%   - Predicates with choice points stay in interpreter mode
%   - Function names: pred_{functor}_{arity} (sanitised for Python)

:- module(wam_python_lowered_emitter, [
	emit_lowered_python/4,           % +FunctorArity, +Instrs, +Options, -Lines
	is_deterministic_pred_py/1,      % +Instrs
	python_func_name/2,              % +Functor/Arity, -PythonName
	parse_wam_text_py/2,             % +WamText, -Instrs
	is_match_instr_py/1,             % +Instr
	is_ite_block_py/2,               % +Instrs, -Blocks
	emit_ite_block_py/5              % +FuncName, +Blocks, +Indent, +Opts, -Lines
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% ============================================================================
% Deterministic predicate detection
% ============================================================================

%% is_deterministic_pred_py(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_py(Instrs) :-
	\+ has_choice_point_instr(Instrs).

has_choice_point_instr(Instrs) :-
	member(I, Instrs),
	choice_point_instr(I),
	!.

choice_point_instr(try_me_else(_)).
choice_point_instr(retry_me_else(_)).
choice_point_instr(trust_me).
choice_point_instr(try(_)).
choice_point_instr(retry(_)).
choice_point_instr(trust(_)).
choice_point_instr(call_indexed_atom_fact2(_)).

% ============================================================================
% Function name generation
% ============================================================================

%% python_func_name(+Functor/Arity, -PythonName)
%  Generate a valid Python function name from a Prolog functor/arity.
%  foo/2 -> 'pred_foo_2'
%  Sanitise: replace non-alphanumeric chars in functor with '_'
python_func_name(Functor/Arity, Name) :-
	atom_string(Functor, FStr),
	sanitize_python_ident(FStr, SanStr),
	format(atom(Name), 'pred_~w_~w', [SanStr, Arity]).

sanitize_python_ident(In, Out) :-
	string_codes(In, Codes),
	maplist(sanitize_code, Codes, OutCodes),
	string_codes(OutStr, OutCodes),
	atom_string(Out, OutStr).

sanitize_code(C, C) :-
	(   C >= 0'a, C =< 0'z -> true
	;   C >= 0'A, C =< 0'Z -> true
	;   C >= 0'0, C =< 0'9 -> true
	;   C =:= 0'_ -> true
	),
	!.
sanitize_code(_, 0'_).

% ============================================================================
% WAM text parsing (shared pattern with other emitters)
% ============================================================================

parse_wam_text_py(WamText, Instrs) :-
	atom_string(WamText, S),
	split_string(S, "\n", "", Lines),
	parse_lines_py(Lines, Instrs).

parse_lines_py([], []).
parse_lines_py([Line|Rest], Instrs) :-
	split_string(Line, " \t,", " \t,", Parts),
	delete(Parts, "", CleanParts),
	(   CleanParts == []
	->  parse_lines_py(Rest, Instrs)
	;   CleanParts = [First|_],
		(   sub_string(First, _, 1, 0, ":")
		->  % Label line — skip
			parse_lines_py(Rest, Instrs)
		;   instr_from_parts_py(CleanParts, Instr)
		->  Instrs = [Instr|RestInstrs],
			parse_lines_py(Rest, RestInstrs)
		;   % Unknown — skip
			parse_lines_py(Rest, Instrs)
		)
	).

% ============================================================================
% Instruction parsing (WAM text line → Prolog term)
% ============================================================================

instr_from_parts_py(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts_py(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
instr_from_parts_py(["get_value", Xn, Ai], get_value(Xn, Ai)).
instr_from_parts_py(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts_py(["get_list", Ai], get_list(Ai)).
instr_from_parts_py(["get_nil", Ai], get_nil(Ai)).
instr_from_parts_py(["get_integer", N, Ai], get_integer(N, Ai)).
instr_from_parts_py(["get_float", F, Ai], get_float(F, Ai)).
instr_from_parts_py(["unify_variable", Xn], unify_variable(Xn)).
instr_from_parts_py(["unify_value", Xn], unify_value(Xn)).
instr_from_parts_py(["unify_constant", C], unify_constant(C)).
instr_from_parts_py(["unify_nil"], unify_nil).
instr_from_parts_py(["unify_void", N], unify_void(N)).
instr_from_parts_py(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
instr_from_parts_py(["put_value", Xn, Ai], put_value(Xn, Ai)).
instr_from_parts_py(["put_unsafe_value", Yn, Ai], put_unsafe_value(Yn, Ai)).
instr_from_parts_py(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts_py(["put_nil", Ai], put_nil(Ai)).
instr_from_parts_py(["put_integer", N, Ai], put_integer(N, Ai)).
instr_from_parts_py(["put_float", F, Ai], put_float(F, Ai)).
instr_from_parts_py(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts_py(["put_list", Ai], put_list(Ai)).
instr_from_parts_py(["call", P, N], call(P, N)).
instr_from_parts_py(["execute", P], execute(P)).
instr_from_parts_py(["proceed"], proceed).
instr_from_parts_py(["fail"], fail).
instr_from_parts_py(["halt"], halt).
instr_from_parts_py(["allocate"], allocate).
instr_from_parts_py(["deallocate"], deallocate).
instr_from_parts_py(["try_me_else", Label], try_me_else(Label)).
instr_from_parts_py(["retry_me_else", Label], retry_me_else(Label)).
instr_from_parts_py(["trust_me"], trust_me).
instr_from_parts_py(["try", Label], try(Label)).
instr_from_parts_py(["retry", Label], retry(Label)).
instr_from_parts_py(["trust", Label], trust(Label)).
instr_from_parts_py(["is", Target, Expr], is(Target, Expr)).
instr_from_parts_py(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
instr_from_parts_py(["call_foreign", Pred, Ar], call_foreign(Pred, Ar)).
instr_from_parts_py(["call_indexed_atom_fact2", Pred], call_indexed_atom_fact2(Pred)).
instr_from_parts_py(["base_category_ancestor", CatReg, TargetReg, VisitedReg],
	base_category_ancestor(CatReg, TargetReg, VisitedReg)).
instr_from_parts_py(["base_category_ancestor_bind", CatReg, TargetReg, HopsReg, VisitedReg],
	base_category_ancestor_bind(CatReg, TargetReg, HopsReg, VisitedReg)).
instr_from_parts_py(["recurse_category_ancestor", MidReg, RootReg, ChildHopsReg, VisitedReg, Pred, Skip],
	recurse_category_ancestor(MidReg, RootReg, ChildHopsReg, VisitedReg, Pred, Skip)).
instr_from_parts_py(["return_add1", OutReg, InReg], return_add1(OutReg, InReg)).
instr_from_parts_py(["neck_cut"], neck_cut).
instr_from_parts_py(["get_level", Yn], get_level(Yn)).
instr_from_parts_py(["cut", Yn], cut(Yn)).

% ============================================================================
% ITE (if/then/else) block detection
% ============================================================================

%% is_match_instr_py(+Instr)
%  True if the instruction is a match/test instruction that can serve as
%  the head of an ITE branch (analogous to is_match_instr_fs/1 in the
%  F# lowered emitter).
is_match_instr_py(get_constant(_, _)).
is_match_instr_py(get_integer(_, _)).
is_match_instr_py(get_float(_, _)).
is_match_instr_py(get_nil(_)).
is_match_instr_py(get_structure(_, _)).
is_match_instr_py(get_list(_)).
is_match_instr_py(get_value(_, _)).
is_match_instr_py(switch_on_term(_, _, _, _)).
is_match_instr_py(switch_on_constant(_, _)).
is_match_instr_py(switch_on_structure(_, _)).

%% is_ite_block_py(+Instrs, -Blocks)
%  Detects if Instrs contains a multi-clause ITE pattern: a sequence of
%  [match_instrs..., proceed/execute] pairs that can be rendered as
%  if/elif/else chains.
%
%  Blocks = list of clause_block(MatchInstrs, BodyInstrs).
%  Requires at least 2 blocks to qualify as an ITE pattern.
is_ite_block_py(Instrs, Blocks) :-
	split_ite_clauses_py(Instrs, Blocks),
	length(Blocks, N),
	N >= 2.

%% split_ite_clauses_py(+Instrs, -Blocks)
%  Split a flat instruction list into clause_block(Match, Body) entries.
%  Each clause is: zero or more match instructions followed by a terminator
%  (proceed, execute(_), or fail).
split_ite_clauses_py([], []).
split_ite_clauses_py(Instrs, [clause_block(MatchInstrs, BodyInstrs)|Rest]) :-
	Instrs \= [],
	take_clause_py(Instrs, MatchInstrs, BodyInstrs, Remaining),
	split_ite_clauses_py(Remaining, Rest).

%% take_clause_py(+Instrs, -MatchInstrs, -BodyInstrs, -Remaining)
%  Extract one clause: match instructions up to (and including) the
%  first terminator instruction (proceed, execute/1, fail).
take_clause_py([proceed|Rest], [], [proceed], Rest) :- !.
take_clause_py([execute(P)|Rest], [], [execute(P)], Rest) :- !.
take_clause_py([fail|Rest], [], [fail], Rest) :- !.
take_clause_py([Instr|Rest], [Instr|MatchRest], BodyInstrs, Remaining) :-
	is_match_instr_py(Instr),
	take_clause_py(Rest, MatchRest, BodyInstrs, Remaining).
% Non-match, non-terminator instructions go into body
take_clause_py([Instr|Rest], [], [Instr|BodyRest], Remaining) :-
	\+ is_match_instr_py(Instr),
	Instr \= proceed,
	Instr \= fail,
	\+ (Instr = execute(_)),
	take_body_to_terminator_py(Rest, BodyRest, Remaining).

%% take_body_to_terminator_py(+Instrs, -BodyInstrs, -Remaining)
take_body_to_terminator_py([proceed|Rest], [proceed], Rest) :- !.
take_body_to_terminator_py([execute(P)|Rest], [execute(P)], Rest) :- !.
take_body_to_terminator_py([fail|Rest], [fail], Rest) :- !.
take_body_to_terminator_py([I|Rest], [I|BodyRest], Remaining) :-
	take_body_to_terminator_py(Rest, BodyRest, Remaining).
take_body_to_terminator_py([], [], []).

% ============================================================================
% ITE emission — if/elif/else Python code generation
% ============================================================================

%% emit_ite_block_py(+FuncName, +Blocks, +Indent, +Opts, -Lines)
%  Emits an if/elif/else chain for the detected ITE blocks.
%  Indent is an integer (number of spaces).
emit_ite_block_py(FuncName, Blocks, Indent, _Opts, Lines) :-
	emit_ite_branches_py(Blocks, FuncName, Indent, first, BranchLines),
	% Add else fallthrough
	indent_str(Indent, Pad),
	format(string(ElseLine), "~welse:~n~w    return False", [Pad, Pad]),
	append(BranchLines, [ElseLine], Lines).

%% emit_ite_branches_py(+Blocks, +FuncName, +Indent, +Position, -Lines)
%  Position is 'first' for the first block (emits 'if'),
%  or 'rest' for subsequent blocks (emits 'elif').
emit_ite_branches_py([], _, _, _, []).
emit_ite_branches_py([clause_block(MatchInstrs, BodyInstrs)|Rest], FuncName, Indent, Pos, Lines) :-
	indent_str(Indent, Pad),
	(   Pos = first -> Keyword = "if"
	;   Keyword = "elif"
	),
	% Generate the condition from match instructions
	match_instrs_to_condition_py(MatchInstrs, Condition),
	% Generate the body
	match_instrs_to_binding_py(MatchInstrs, Indent, BindingLines),
	body_instrs_to_code_py(BodyInstrs, Indent, BodyCodeLines),
	append(BindingLines, BodyCodeLines, InnerLines),
	format(string(BranchHeader), "~w~w ~w:", [Pad, Keyword, Condition]),
	(   InnerLines = []
	->  format(string(BranchBody), "~w    pass", [Pad]),
	    BranchAll = [BranchHeader, BranchBody]
	;   BranchAll = [BranchHeader|InnerLines]
	),
	emit_ite_branches_py(Rest, FuncName, Indent, rest, RestLines),
	append(BranchAll, RestLines, Lines).

%% match_instrs_to_condition_py(+MatchInstrs, -Condition)
%  Generate a Python condition expression from match instructions.
match_instrs_to_condition_py([], "True").
match_instrs_to_condition_py([Instr], Cond) :-
	single_match_condition_py(Instr, Cond).
match_instrs_to_condition_py([Instr|Rest], Cond) :-
	Rest \= [],
	single_match_condition_py(Instr, C1),
	match_instrs_to_condition_py(Rest, C2),
	format(string(Cond), "(~w) and (~w)", [C1, C2]).

%% single_match_condition_py(+Instr, -Condition)
%  Generate a Python condition for a single match instruction.
single_match_condition_py(get_constant(C, AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	escape_py(C, EC),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Atom) and _a~w.name == "~w")',
		[Ai, Ai, Ai, EC]).
single_match_condition_py(get_integer(NStr, AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Int) and _a~w.n == ~w)',
		[Ai, Ai, Ai, NStr]).
single_match_condition_py(get_float(FStr, AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Float) and _a~w.f == ~w)',
		[Ai, Ai, Ai, FStr]).
single_match_condition_py(get_nil(AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Atom) and _a~w.name == "[]")',
		[Ai, Ai, Ai]).
single_match_condition_py(get_structure(FStr, AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	parse_functor_arity_py(FStr, FuncName, Arity),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Compound) and _a~w.functor == "~w" and len(_a~w.args) == ~w)',
		[Ai, Ai, Ai, FuncName, Ai, Arity]).
single_match_condition_py(get_list(AiStr), Cond) :-
	reg_int_py(AiStr, Ai),
	format(string(Cond),
		'isinstance(_a~w, Var) or (isinstance(_a~w, Compound) and _a~w.functor == ".")',
		[Ai, Ai, Ai]).
single_match_condition_py(get_value(XnStr, AiStr), Cond) :-
	reg_int_py(XnStr, Xn), reg_int_py(AiStr, Ai),
	format(string(Cond),
		'unify(deref(state.regs[~w], state), deref(state.regs[~w], state), state)',
		[Ai, Xn]).
% Default fallback for unknown match instructions
single_match_condition_py(_, "True").

%% match_instrs_to_binding_py(+MatchInstrs, +Indent, -Lines)
%  Generate Python binding code for variables encountered during matching.
match_instrs_to_binding_py([], _, []).
match_instrs_to_binding_py([Instr|Rest], Indent, Lines) :-
	single_match_binding_py(Instr, Indent, BindLines),
	match_instrs_to_binding_py(Rest, Indent, RestLines),
	append(BindLines, RestLines, Lines).

%% single_match_binding_py(+Instr, +Indent, -Lines)
single_match_binding_py(get_constant(C, AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	escape_py(C, EC),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Atom(\"~w\"), state)",
		[Pad, Ai, Ai, EC]),
	Lines = [DerefLine, BindLine].
single_match_binding_py(get_integer(NStr, AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Int(~w), state)",
		[Pad, Ai, Ai, NStr]),
	Lines = [DerefLine, BindLine].
single_match_binding_py(get_float(FStr, AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Float(~w), state)",
		[Pad, Ai, Ai, FStr]),
	Lines = [DerefLine, BindLine].
single_match_binding_py(get_nil(AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Atom(\"[]\"), state)",
		[Pad, Ai, Ai]),
	Lines = [DerefLine, BindLine].
single_match_binding_py(get_structure(FStr, AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	parse_functor_arity_py(FStr, FuncName, Arity),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Compound(\"~w\", [None]*~w), state)",
		[Pad, Ai, Ai, FuncName, Arity]),
	Lines = [DerefLine, BindLine].
single_match_binding_py(get_list(AiStr), Indent, Lines) :-
	reg_int_py(AiStr, Ai),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(DerefLine), "~w_a~w = deref(state.regs[~w], state)", [Pad, Ai, Ai]),
	format(string(BindLine), "~wif isinstance(_a~w, Var): bind(_a~w, Compound(\".\", [None, None]), state)",
		[Pad, Ai, Ai]),
	Lines = [DerefLine, BindLine].
% get_value — no binding needed beyond what's in the condition
single_match_binding_py(get_value(_, _), _, []).
% Default fallback
single_match_binding_py(_, _, []).

%% body_instrs_to_code_py(+BodyInstrs, +Indent, -Lines)
%  Emit the body instructions as Python code inside an ITE branch.
body_instrs_to_code_py([], _, []).
body_instrs_to_code_py([proceed], Indent, [Line]) :-
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(Line), "~wreturn True", [Pad]).
body_instrs_to_code_py([fail], Indent, [Line]) :-
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(Line), "~wreturn False", [Pad]).
body_instrs_to_code_py([execute(PStr)], Indent, [Line]) :-
	pred_to_func_name_py(PStr, FN),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	format(string(Line), "~wreturn ~w(state)", [Pad, FN]).
body_instrs_to_code_py([Instr|Rest], Indent, [Line|Lines]) :-
	Instr \= proceed,
	Instr \= fail,
	\+ (Instr = execute(_)),
	emit_instr_py(Instr, Code),
	Indent1 is Indent + 4,
	indent_str(Indent1, Pad),
	% Re-indent the emitted code to the ITE body level
	split_string(Code, "\n", "", CodeLines),
	reindent_lines(CodeLines, Pad, ReindentedStr),
	Line = ReindentedStr,
	body_instrs_to_code_py(Rest, Indent, Lines).

%% reindent_lines(+Lines, +Pad, -Result)
%  Strip leading whitespace from each line and prepend Pad.
reindent_lines([], _, "").
reindent_lines([L], Pad, Result) :-
	split_string(L, "", " \t", [Trimmed]),
	format(string(Result), "~w~w", [Pad, Trimmed]).
reindent_lines([L|Rest], Pad, Result) :-
	Rest \= [],
	split_string(L, "", " \t", [Trimmed]),
	reindent_lines(Rest, Pad, RestResult),
	format(string(Result), "~w~w\n~w", [Pad, Trimmed, RestResult]).

%% indent_str(+N, -Str)
%  Generate a string of N spaces.
indent_str(0, "") :- !.
indent_str(N, Str) :-
	N > 0,
	length(Codes, N),
	maplist(=(0' ), Codes),
	string_codes(Str, Codes).

% ============================================================================
% Main entry point
% ============================================================================

%% emit_lowered_python(+FunctorArity, +Instrs, +Options, -Lines)
%  Emits a single deterministic predicate as a Python function.
%  FunctorArity: Functor/Arity
%  Instrs: list of WAM instruction terms (parsed)
%  Options: option list
%  Lines: string containing the Python function definition
%
%  If the instructions form an ITE block (multi-clause pattern),
%  emit if/elif/else chains instead of instruction-by-instruction code.
emit_lowered_python(FunctorArity, Instrs, Options, Lines) :-
	is_list(Instrs),
	is_ite_block_py(Instrs, Blocks),
	!,
	python_func_name(FunctorArity, FuncName),
	FunctorArity = Functor/Arity,
	atom_string(Functor, FStr),
	emit_ite_block_py(FuncName, Blocks, 4, Options, BodyLines),
	atomic_list_concat(BodyLines, '\n', Body),
	format(string(Lines),
'def ~w(state):
    """Lowered predicate: ~w/~w (ITE)"""
~w', [FuncName, FStr, Arity, Body]).

emit_lowered_python(FunctorArity, Instrs, _Options, Lines) :-
	is_list(Instrs),
	python_func_name(FunctorArity, FuncName),
	FunctorArity = Functor/Arity,
	atom_string(Functor, FStr),
	maplist(emit_instr_py, Instrs, InstrLines),
	atomic_list_concat(InstrLines, '\n', Body),
	format(string(Lines),
'def ~w(state):
    """Lowered predicate: ~w/~w"""
~w', [FuncName, FStr, Arity, Body]).

%% emit_lowered_python(+FunctorArity, +WamCode, +Options, -Lines)
%  Alternate entry: takes raw WAM text, parses, then emits.
emit_lowered_python(FunctorArity, WamCode, Options, Lines) :-
	atom(WamCode),
	parse_wam_text_py(WamCode, Instrs),
	emit_lowered_python(FunctorArity, Instrs, Options, Lines).

% ============================================================================
% Instruction lowering — one clause per WAM opcode
% ============================================================================

%% emit_instr_py(+Instr, -PythonLine)
%  Emit a single lowered instruction as Python code.

% --- Head unification (get_*) ---

emit_instr_py(get_constant(C, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	escape_py(C, EC),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var): bind(_a~w, Atom("~w"), state)
    elif not (isinstance(_a~w, Atom) and _a~w.name == "~w"): return False',
		[Ai, Ai, Ai, Ai, EC, Ai, Ai, EC]).

emit_instr_py(get_variable(XnStr, AiStr), Code) :-
	reg_int_py(XnStr, Xn), reg_int_py(AiStr, Ai),
	format(string(Code),
'    state.regs[~w] = state.regs[~w]', [Xn, Ai]).

emit_instr_py(get_value(XnStr, AiStr), Code) :-
	reg_int_py(XnStr, Xn), reg_int_py(AiStr, Ai),
	format(string(Code),
'    if not unify(deref(state.regs[~w], state), deref(state.regs[~w], state), state): return False',
		[Ai, Xn]).

emit_instr_py(get_nil(AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var): bind(_a~w, Atom("[]"), state)
    elif not (isinstance(_a~w, Atom) and _a~w.name == "[]"): return False',
		[Ai, Ai, Ai, Ai, Ai, Ai]).

emit_instr_py(get_integer(NStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var): bind(_a~w, Int(~w), state)
    elif not (isinstance(_a~w, Int) and _a~w.n == ~w): return False',
		[Ai, Ai, Ai, Ai, NStr, Ai, Ai, NStr]).

emit_instr_py(get_float(FStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var): bind(_a~w, Float(~w), state)
    elif not (isinstance(_a~w, Float) and _a~w.f == ~w): return False',
		[Ai, Ai, Ai, Ai, FStr, Ai, Ai, FStr]).

emit_instr_py(get_structure(FStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	parse_functor_arity_py(FStr, FuncName, Arity),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var):
        _addr = heap_put(state, Compound("~w", [None]*~w))
        bind(_a~w, Ref(_addr), state)
        state.mode = "write"; state.s = _addr
    elif isinstance(_a~w, Ref):
        _h = state.heap[_a~w.addr]
        if isinstance(_h, Compound) and _h.functor == "~w" and len(_h.args) == ~w:
            state.mode = "read"; state.s = _a~w.addr
        else: return False
    elif isinstance(_a~w, Compound) and _a~w.functor == "~w" and len(_a~w.args) == ~w:
        state.mode = "read"
    else: return False',
		[Ai, Ai, Ai, FuncName, Arity, Ai,
		 Ai, Ai, FuncName, Arity, Ai,
		 Ai, Ai, FuncName, Ai, Arity]).

emit_instr_py(get_list(AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    _a~w = deref(state.regs[~w], state)
    if isinstance(_a~w, Var):
        _addr = heap_put(state, Compound(".", [None, None]))
        bind(_a~w, Ref(_addr), state)
        state.mode = "write"; state.s = _addr
    elif isinstance(_a~w, Ref):
        _h = state.heap[_a~w.addr]
        if isinstance(_h, Compound) and _h.functor == "." and len(_h.args) == 2:
            state.mode = "read"; state.s = _a~w.addr
        else: return False
    else: return False',
		[Ai, Ai, Ai, Ai, Ai, Ai, Ai]).

% --- Body construction (put_*) ---

emit_instr_py(put_variable(XnStr, AiStr), Code) :-
	reg_int_py(XnStr, Xn), reg_int_py(AiStr, Ai),
	format(string(Code),
'    _v = state.fresh_var()
    heap_put(state, _v)
    state.regs[~w] = _v; state.regs[~w] = _v', [Xn, Ai]).

emit_instr_py(put_value(XnStr, AiStr), Code) :-
	reg_int_py(XnStr, Xn), reg_int_py(AiStr, Ai),
	format(string(Code),
'    state.regs[~w] = state.regs[~w]', [Ai, Xn]).

emit_instr_py(put_unsafe_value(YnStr, AiStr), Code) :-
	reg_int_py(YnStr, Yn), reg_int_py(AiStr, Ai),
	format(string(Code),
'    _d = deref(state.regs[~w], state)
    if isinstance(_d, Var):
        _nv = state.fresh_var()
        heap_put(state, _nv)
        bind(_d, _nv, state)
        state.regs[~w] = _nv
    else:
        state.regs[~w] = _d', [Yn, Ai, Ai]).

emit_instr_py(put_constant(C, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	escape_py(C, EC),
	format(string(Code),
'    state.regs[~w] = Atom("~w")', [Ai, EC]).

emit_instr_py(put_nil(AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    state.regs[~w] = Atom("[]")', [Ai]).

emit_instr_py(put_integer(NStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    state.regs[~w] = Int(~w)', [Ai, NStr]).

emit_instr_py(put_float(FStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    state.regs[~w] = Float(~w)', [Ai, FStr]).

emit_instr_py(put_structure(FStr, AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	parse_functor_arity_py(FStr, FuncName, Arity),
	format(string(Code),
'    _addr = heap_put(state, Compound("~w", [None]*~w))
    state.regs[~w] = Ref(_addr)
    state.mode = "write"; state.s = _addr', [FuncName, Arity, Ai]).

emit_instr_py(put_list(AiStr), Code) :-
	reg_int_py(AiStr, Ai),
	format(string(Code),
'    _addr = heap_put(state, Compound(".", [None, None]))
    state.regs[~w] = Ref(_addr)
    state.mode = "write"; state.s = _addr', [Ai]).

% --- Unify instructions ---

emit_instr_py(unify_variable(XnStr), Code) :-
	reg_int_py(XnStr, Xn),
	format(string(Code),
'    if state.mode == "read":
        _h = state.heap[state.s]
        if isinstance(_h, Compound):
            state.regs[~w] = _h.args[0]; state.s += 1
        else:
            state.regs[~w] = _h
    else:
        _v = state.fresh_var()
        heap_put(state, _v)
        state.regs[~w] = _v', [Xn, Xn, Xn]).

emit_instr_py(unify_value(XnStr), Code) :-
	reg_int_py(XnStr, Xn),
	format(string(Code),
'    if state.mode == "read":
        _h = state.heap[state.s]
        if not unify(state.regs[~w], _h, state): return False
    else:
        heap_put(state, state.regs[~w])', [Xn, Xn]).

emit_instr_py(unify_constant(C), Code) :-
	escape_py(C, EC),
	format(string(Code),
'    if state.mode == "read":
        _h = deref(state.heap[state.s], state)
        if isinstance(_h, Var): bind(_h, Atom("~w"), state)
        elif not (isinstance(_h, Atom) and _h.name == "~w"): return False
    else:
        heap_put(state, Atom("~w"))', [EC, EC, EC]).

emit_instr_py(unify_nil, Code) :-
	Code = '    if state.mode == "read":
        _h = deref(state.heap[state.s], state)
        if isinstance(_h, Var): bind(_h, Atom("[]"), state)
        elif not (isinstance(_h, Atom) and _h.name == "[]"): return False
    else:
        heap_put(state, Atom("[]"))'.

emit_instr_py(unify_void(NStr), Code) :-
	format(string(Code),
'    if state.mode == "write":
        for _ in range(~w):
            _v = state.fresh_var()
            heap_put(state, _v)', [NStr]).

% --- Control instructions ---

emit_instr_py(call(PStr, _NStr), Code) :-
	pred_to_func_name_py(PStr, FN),
	format(string(Code),
'    _saved_cp = state.cp
    if not ~w(state): return False
    state.cp = _saved_cp', [FN]).

emit_instr_py(execute(PStr), Code) :-
	pred_to_func_name_py(PStr, FN),
	format(string(Code),
'    return ~w(state)', [FN]).

emit_instr_py(proceed, Code) :-
	Code = '    return True'.

emit_instr_py(fail, Code) :-
	Code = '    return False'.

emit_instr_py(halt, Code) :-
	Code = '    return True'.

% --- Environment instructions ---

emit_instr_py(allocate, Code) :-
	Code = '    push_environment(state, 0)'.

emit_instr_py(deallocate, Code) :-
	Code = '    pop_environment(state)'.

% --- Arithmetic ---

emit_instr_py(is(TargetStr, ExprStr), Code) :-
	reg_int_py(TargetStr, Target),
	reg_int_py(ExprStr, Expr),
	format(string(Code),
'    _expr = deref(state.regs[~w], state)
    _result = eval_arith(_expr, state)
    _rv = Int(_result) if isinstance(_result, int) else Float(_result)
    if not unify(state.regs[~w], _rv, state): return False', [Expr, Target]).

% --- Built-in calls ---

emit_instr_py(builtin_call(OpStr, ArStr), Code) :-
	escape_py(OpStr, EOp),
	format(string(Code),
'    _args = [deref(state.regs[i+1], state) for i in range(~w)]
    if not execute_foreign("~w", ~w, _args, state): return False',
		[ArStr, EOp, ArStr]).

emit_instr_py(call_foreign(PredStr, ArStr), Code) :-
	escape_py(PredStr, EP),
	format(string(Code),
'    _args = [deref(state.regs[i+1], state) for i in range(~w)]
    if not execute_foreign("~w", ~w, _args, state): return False',
		[ArStr, EP, ArStr]).

emit_instr_py(call_indexed_atom_fact2(PredStr), Code) :-
	escape_py(PredStr, EP),
	format(string(Code),
'    _key = deref(state.regs[1], state)
    if not isinstance(_key, Atom): return False
    _values = state.indexed_atom_fact2.get("~w", {}).get(_key.name, [])
    if not _values: return False
    if not unify(state.regs[2], Atom(_values[0]), state): return False', [EP]).

emit_instr_py(base_category_ancestor(CatRegStr, TargetRegStr, VisitedRegStr), Code) :-
	reg_int_py(CatRegStr, CatReg),
	reg_int_py(TargetRegStr, TargetReg),
	reg_int_py(VisitedRegStr, VisitedReg),
	format(string(Code),
'    _cat = deref(state.regs[~w], state)
    _target = deref(state.regs[~w], state)
    _visited = deref(state.regs[~w], state)
    if not isinstance(_cat, Atom) or not isinstance(_target, Atom): return False
    if _atom_in_cons_list(_target, _visited, state): return False
    if _target.name not in state.indexed_atom_fact2.get("category_parent/2", {}).get(_cat.name, []): return False
    pop_environment(state)
    return True', [CatReg, TargetReg, VisitedReg]).

emit_instr_py(base_category_ancestor_bind(CatRegStr, TargetRegStr, HopsRegStr, VisitedRegStr), Code) :-
	reg_int_py(CatRegStr, CatReg),
	reg_int_py(TargetRegStr, TargetReg),
	reg_int_py(HopsRegStr, HopsReg),
	reg_int_py(VisitedRegStr, VisitedReg),
	format(string(Code),
'    _cat = deref(state.regs[~w], state)
    _target = deref(state.regs[~w], state)
    _visited = deref(state.regs[~w], state)
    if not isinstance(_cat, Atom) or not isinstance(_target, Atom): return False
    if _atom_in_cons_list(_target, _visited, state): return False
    if _target.name not in state.indexed_atom_fact2.get("category_parent/2", {}).get(_cat.name, []): return False
    if not unify(state.regs[~w], Int(1), state): return False
    pop_environment(state)
    return True', [CatReg, TargetReg, VisitedReg, HopsReg]).

emit_instr_py(recurse_category_ancestor(MidRegStr, RootRegStr, ChildHopsRegStr, VisitedRegStr, PredStr, _SkipStr), Code) :-
	reg_int_py(MidRegStr, MidReg),
	reg_int_py(RootRegStr, RootReg),
	reg_int_py(ChildHopsRegStr, ChildHopsReg),
	reg_int_py(VisitedRegStr, VisitedReg),
	pred_to_func_name_py(PredStr, FN),
	format(string(Code),
'    _mid = deref(state.regs[~w], state)
    _root = deref(state.regs[~w], state)
    _visited = deref(state.regs[~w], state)
    _child_hops = state.fresh_var()
    state.regs[~w] = _child_hops
    state.regs[1] = _mid
    state.regs[2] = _root
    state.regs[3] = _child_hops
    state.regs[4] = Compound(".", [_mid, _visited])
    if not ~w(state): return False', [MidReg, RootReg, VisitedReg, ChildHopsReg, FN]).

emit_instr_py(return_add1(OutRegStr, InRegStr), Code) :-
	reg_int_py(OutRegStr, OutReg),
	reg_int_py(InRegStr, InReg),
	format(string(Code),
'    _in = deref(state.regs[~w], state)
    if isinstance(_in, Int):
        _result = Int(_in.n + 1)
    elif isinstance(_in, Float):
        _next = _in.f + 1.0
        _result = Int(int(round(_next))) if abs(round(_next) - _next) < 1e-12 else Float(_next)
    elif isinstance(_in, Atom):
        try:
            _next = float(_in.name) + 1.0
            _result = Int(int(round(_next))) if abs(round(_next) - _next) < 1e-12 else Float(_next)
        except (TypeError, ValueError):
            return False
    else:
        return False
    if not unify(state.regs[~w], _result, state): return False
    pop_environment(state)
    return True', [InReg, OutReg]).

% --- Cut instructions ---

emit_instr_py(neck_cut, Code) :-
	Code = '    state.b = state.cut_b'.

emit_instr_py(get_level(YnStr), Code) :-
	reg_int_py(YnStr, Yn),
	format(string(Code),
'    state.regs[~w] = state.b', [Yn]).

emit_instr_py(cut(YnStr), Code) :-
	reg_int_py(YnStr, Yn),
	format(string(Code),
'    state.b = state.regs[~w]', [Yn]).

% ============================================================================
% Helpers
% ============================================================================

%% reg_int_py(+RegStr, -Int)
%  Parse register string to integer. Handles both numeric strings
%  and symbolic register names (A1, X1, Y1).
reg_int_py(RegStr, Int) :-
	(   number_string(Int, RegStr)
	->  true
	;   atom_string(RegA, RegStr),
		sub_atom(RegA, 0, 1, _, Prefix),
		sub_atom(RegA, 1, _, 0, NumA),
		atom_number(NumA, Num),
		(   Prefix == 'A' -> Int = Num
		;   Prefix == 'X' -> Int is Num + 100
		;   Prefix == 'Y' -> Int is Num + 200
		;   Int = 0
		)
	).

%% parse_functor_arity_py(+FStr, -FuncName, -Arity)
parse_functor_arity_py(FStr, FuncName, Arity) :-
	atom_string(FA, FStr),
	(   sub_atom(FA, B, 1, _, '/')
	->  sub_atom(FA, 0, B, _, FuncName),
		B1 is B + 1,
		sub_atom(FA, B1, _, 0, AS),
		atom_number(AS, Arity)
	;   FuncName = FA, Arity = 0
	).

%% pred_to_func_name_py(+PredStr, -FuncName)
%  Convert a "functor/arity" or "functor" string to a Python function name.
pred_to_func_name_py(PredStr, FuncName) :-
	atom_string(PA, PredStr),
	(   sub_atom(PA, B, 1, _, '/')
	->  sub_atom(PA, 0, B, _, Functor),
		B1 is B + 1,
		sub_atom(PA, B1, _, 0, AS),
		atom_number(AS, Arity)
	;   Functor = PA, Arity = 0
	),
	python_func_name(Functor/Arity, FuncName).

%% escape_py(+Str, -Escaped)
%  Escape a string for Python string literal (backslashes and double quotes).
escape_py(Str, Esc) :-
	atom_string(Str, S),
	split_string(S, "\\", "", Parts),
	atomic_list_concat(Parts, "\\\\", S1),
	split_string(S1, "\"", "", Parts2),
	atomic_list_concat(Parts2, "\\\"", Esc).

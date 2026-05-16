:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_cpp_target.pl - WAM-to-C++ Hybrid Transpilation Target
%
% Generates a C++ project with an instruction-array WAM interpreter plus
% optional lowered per-predicate C++ functions. The architecture mirrors
% the Rust/Haskell/Lua hybrid WAM targets while exploiting C++ ergonomics
% (std::variant, std::unordered_map, std::string).
%
% Entry points:
%   compile_wam_predicate_to_cpp/4 - Translate a single predicate.
%   write_wam_cpp_project/3        - Materialise a full C++ project.
%   compile_wam_runtime_to_cpp/2   - Emit the wam_runtime.cpp source.
%   wam_instruction_to_cpp_literal/2,3 - Emit C++ struct-initializer
%       literals for raw WAM instructions (mirrors wam_c_target.pl's
%       designated-initializer style for use in the instruction array).
%
% Emit modes (resolved by cpp_wam_resolve_emit_mode/2):
%   interpreter  - All predicates dispatched via the instruction array.
%   functions    - Lowerable predicates emitted as direct C++ functions
%                  (lowered_<pred>_<arity>) on top of the array.
%   mixed(List)  - Only the listed Pred/Arity entries are lowered.

:- module(wam_cpp_target, [
    compile_wam_predicate_to_cpp/4,        % +Pred/Arity, +WamCode, +Options, -CppCode
    write_wam_cpp_project/3,               % +Predicates, +Options, +ProjectDir
    compile_wam_runtime_to_cpp/2,          % +Options, -RuntimeCppCode
    compile_wam_runtime_header_to_cpp/2,   % +Options, -RuntimeHeaderCode
    cpp_wam_resolve_emit_mode/2,           % +Options, -Mode
    wam_instruction_to_cpp_literal/2,      % +Instr, -CppCode
    wam_instruction_to_cpp_literal/3,      % +Instr, +LabelMap, -CppCode
    cpp_safe_function_name/2,              % +Pred, -CppFuncName
    escape_cpp_string/2,                   % +InStr, -OutStr
    cpp_value_literal/2,                   % +Constant, -CppLiteral
    iso_errors_resolve_options/2,          % +Options, -Config
    iso_errors_load_config/2,              % +File, -Config
    iso_errors_mode_for/3,                 % +Config, +PI, -Mode
    wam_cpp_iso_audit/3,                   % +Predicates, +Options, -Audit
    wam_cpp_iso_audit_report/1             % +Audit
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module(wam_cpp_lowered_emitter, [
    wam_cpp_lowerable/3,
    lower_predicate_to_cpp/4,
    cpp_lowered_func_name/2,
    parse_wam_text/2
]).
% Shared WAM-text tokenizer + recogniser (PR following #2086). Local
% tokenizer + parser predicates were lifted into this module; the
% wrapper parse_pred_blocks/2 below is kept as a thin alias since
% existing call sites still use that name.
:- use_module(wam_text_parser, [
    wam_text_to_items/2
]).

:- multifile user:wam_cpp_emit_mode/1.

% ============================================================================
% Emit-mode resolution
% ============================================================================

%% cpp_wam_resolve_emit_mode(+Options, -Mode)
%  Options key:    emit_mode(interpreter | functions | mixed(List))
%  User hook:      user:wam_cpp_emit_mode/1
%  Default:        interpreter
cpp_wam_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_cpp_emit_mode(M0, Mode)
    ;   catch(user:wam_cpp_emit_mode(M1), _, fail)
    ->  validate_cpp_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_cpp_emit_mode(interpreter, interpreter) :- !.
validate_cpp_emit_mode(functions, functions) :- !.
validate_cpp_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
validate_cpp_emit_mode(Other, _) :-
    throw(error(domain_error(wam_cpp_emit_mode, Other),
                cpp_wam_resolve_emit_mode/2)).

should_lower(functions, _, _) :- !.
should_lower(mixed(HotPreds), P, A) :-
    member(P/A, HotPreds), !.
should_lower(_, _, _) :- fail.

% ============================================================================
% Identifier sanitisation & string helpers
% ============================================================================

cpp_safe_function_name(Pred, FuncName) :-
    atom_string(Pred, PredStr),
    string_codes(PredStr, Codes),
    maplist(cpp_safe_identifier_code, Codes, SafeCodes),
    string_codes(FuncStr, SafeCodes),
    atom_string(FuncName, FuncStr).

cpp_safe_identifier_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ),
    !.
cpp_safe_identifier_code(_, 0'_).

%% escape_cpp_string(+In, -Out)
%  Escape backslashes and double quotes for embedding inside a C++
%  string literal. Matches the wam_rust_target.pl helper of the same
%  shape so wam_cpp_lowered_emitter.pl can use it interchangeably.
escape_cpp_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    cpp_atomics_to_string(Parts, "\\\\", Escaped1),
    split_string(Escaped1, "\"", "", Parts2),
    cpp_atomics_to_string(Parts2, "\\\"", Out),
    !.

cpp_atomics_to_string([], _, "") :- !.
cpp_atomics_to_string([X], _, X) :- !.
cpp_atomics_to_string([X, Y|Rest], Sep, Result) :-
    cpp_atomics_to_string([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result),
    !.

%% cpp_value_literal(+ConstantToken, -CppLiteral)
%  Convert a WAM constant (atom, integer, float, []) into a C++ Value
%  literal usable inside the generated source.
cpp_value_literal(C, Val) :-
    to_string(C, Str),
    (   atom_marker_prefix(Str, AtomContent)
    ->  % Atom-marker (\\x01 prefix) — emit as Atom regardless of
        % whether the content re-parses as a number. The marker
        % itself is stripped; only AtomContent reaches the output.
        escape_cpp_string(AtomContent, EscStr),
        format(atom(Val), 'Value::Atom("~w")', [EscStr])
    ;   number_string(N, Str), integer(N)
    ->  format(atom(Val), 'Value::Integer(~w)', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(Val), 'Value::Float(~w)', [F])
    ;   Str == "[]"
    ->  Val = 'Value::Atom("[]")'
    ;   escape_cpp_string(Str, EscStr),
        format(atom(Val), 'Value::Atom("~w")', [EscStr])
    ).

%% atom_marker_prefix(+Str, -Stripped) is semidet.
%  True iff Str starts with the atom-marker (\\x01); Stripped is the
%  remainder. Matches the marker convention emitted by
%  wam_target:quote_wam_constant/2 for atoms-that-look-like-numbers.
atom_marker_prefix(Str, Stripped) :-
    string_codes(Str, [1|RestCodes]),
    string_codes(Stripped, RestCodes).

to_string(X, S) :- string(X), !, S = X.
to_string(X, S) :- atom(X), !, atom_string(X, S).
to_string(X, S) :- number(X), !, number_string(X, S).
to_string(X, S) :- format(string(S), "~w", [X]).

% execute_arity_from_key(+Key, -Arity)
%   For a predicate key like "foo/3", returns the integer arity (3).
%   For anything that doesn''t parse, returns 0. Used by the Execute
%   instruction emitter so Instruction::Execute carries arity in
%   instr.n — matches Call''s factory and lets meta-builtins
%   (call/N) read arity uniformly from instr.n regardless of dispatch
%   path.
execute_arity_from_key(Key, Arity) :-
    to_string(Key, KS),
    (   split_string(KS, "/", "", Parts),
        last(Parts, ArS),
        number_string(N, ArS),
        integer(N)
    ->  Arity = N
    ;   Arity = 0
    ).

% ============================================================================
% Phase 2: WAM instructions -> C++ struct-initializer literals
% ============================================================================
% These produce designated-initialiser style strings suitable for embedding
% in the interpreter's instruction array. Mirrors wam_c_target.pl's
% wam_instruction_to_c_literal/2 pattern but emits C++ aggregate init
% braces (no `.tag =` designators so the output also compiles under
% C++17 without the GCC extension).

wam_instruction_to_cpp_literal(Instr, Code) :-
    wam_instruction_to_cpp_literal(Instr, [], Code),
    !.

wam_instruction_to_cpp_literal(Instr, LabelMap, Code) :-
    wam_instruction_to_cpp_literal_det(Instr, LabelMap, Code),
    !.

wam_instruction_to_cpp_literal_det(get_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal_det(get_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(get_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(get_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::GetStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal_det(get_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetList("~w")', [R]).
wam_instruction_to_cpp_literal_det(get_nil(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetNil("~w")', [R]).
wam_instruction_to_cpp_literal_det(get_integer(N, Ai), _, Code) :-
    to_string(N, NS), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetInteger(~w, "~w")', [NS, R]).
wam_instruction_to_cpp_literal_det(put_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal_det(put_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(put_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(put_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::PutStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal_det(put_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::PutList("~w")', [R]).
wam_instruction_to_cpp_literal_det(unify_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyVariable("~w")', [X]).
wam_instruction_to_cpp_literal_det(unify_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyValue("~w")', [X]).
wam_instruction_to_cpp_literal_det(unify_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::UnifyConstant(~w)', [V]).
wam_instruction_to_cpp_literal_det(set_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetVariable("~w")', [X]).
wam_instruction_to_cpp_literal_det(set_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetValue("~w")', [X]).
wam_instruction_to_cpp_literal_det(set_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::SetConstant(~w)', [V]).
wam_instruction_to_cpp_literal_det(call(P, N), _, Code) :-
    to_string(P, PS), to_string(N, NS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::Call("~w", ~w)', [EP, NS]).
wam_instruction_to_cpp_literal_det(execute(P), _, Code) :-
    to_string(P, PS),
    escape_cpp_string(PS, EP),
    % Derive arity from the "Name/Arity" predicate-key suffix so the
    % runtime can read instr.n directly (matches Call''s factory).
    % Anything that doesn''t parse falls back to arity 0 — Execute
    % targeting a label-only predicate (e.g. the auto-injected
    % helpers) doesn''t care about arity downstream.
    execute_arity_from_key(PS, NArity),
    format(atom(Code), 'Instruction::Execute("~w", ~w)', [EP, NArity]).
wam_instruction_to_cpp_literal_det(proceed, _, 'Instruction::Proceed()').
wam_instruction_to_cpp_literal_det(fail, _, 'Instruction::Fail()').
wam_instruction_to_cpp_literal_det(allocate, _, 'Instruction::Allocate()').
wam_instruction_to_cpp_literal_det(deallocate, _, 'Instruction::Deallocate()').
wam_instruction_to_cpp_literal_det(builtin_call(Op, Ar), _, Code) :-
    to_string(Op, OpS), to_string(Ar, ArS),
    escape_cpp_string(OpS, EOp),
    format(atom(Code), 'Instruction::BuiltinCall("~w", ~w)', [EOp, ArS]).
wam_instruction_to_cpp_literal_det(call_foreign(P, Ar), _, Code) :-
    to_string(P, PS), to_string(Ar, ArS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::CallForeign("~w", ~w)', [EP, ArS]).
wam_instruction_to_cpp_literal_det(try_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::TryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(retry_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::RetryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(trust_me, _, 'Instruction::TrustMe()').
wam_instruction_to_cpp_literal_det(jump(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Jump(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(cut_ite, _, 'Instruction::CutIte()').
wam_instruction_to_cpp_literal_det(begin_aggregate(K, V, R), _, Code) :-
    to_string(K, KS), to_string(V, VS), to_string(R, RS),
    escape_cpp_string(KS, EK), escape_cpp_string(VS, EV), escape_cpp_string(RS, ER),
    format(atom(Code),
           'Instruction::BeginAggregate("~w", "~w", "~w")', [EK, EV, ER]).
wam_instruction_to_cpp_literal_det(begin_aggregate(K, V, R, W), _, Code) :-
    to_string(K, KS), to_string(V, VS), to_string(R, RS), to_string(W, WS),
    escape_cpp_string(KS, EK), escape_cpp_string(VS, EV),
    escape_cpp_string(RS, ER), escape_cpp_string(WS, EW),
    format(atom(Code),
           'Instruction::BeginAggregate("~w", "~w", "~w", "~w")',
           [EK, EV, ER, EW]).
wam_instruction_to_cpp_literal_det(end_aggregate(R), _, Code) :-
    to_string(R, RS),
    escape_cpp_string(RS, ER),
    format(atom(Code), 'Instruction::EndAggregate("~w")', [ER]).

label_index(L, LabelMap, Idx) :-
    to_string(L, LS),
    (   member(LS-I, LabelMap)
    ->  Idx = I
    ;   format(atom(Idx), '"~w"', [LS])
    ).

% ============================================================================
% Per-predicate compilation
% ============================================================================

%% compile_wam_predicate_to_cpp(+Pred/Arity, +WamCode, +Options, -CppCode)
%  Top-level per-predicate compile. Output is one of:
%    - lowered C++ function definition (emit_mode functions / mixed match)
%    - interpreter-array wrapper (default)
compile_wam_predicate_to_cpp(PI, WamCode, Options, CppCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    cpp_wam_resolve_emit_mode(Options, Mode),
    (   should_lower(Mode, Pred, Arity),
        wam_cpp_lowerable(Pred/Arity, WamCode, _Reason)
    ->  lower_predicate_to_cpp(Pred/Arity, WamCode, Options, Lines),
        atomic_list_concat(Lines, '\n', CppCode)
    ;   instrs_for(WamCode, Instrs),
        compile_predicate_wrapper(Pred, Arity, Instrs, Options, CppCode)
    ).

% Accept either an instruction list (from in-memory pipelines) or raw WAM
% text (compile_predicate_to_wam/3 returns a string). The text parser is
% shared with the lowered emitter to keep a single source of truth.
instrs_for(WamCode, Instrs) :-
    is_list(WamCode), !, Instrs = WamCode.
instrs_for(WamCode, Instrs) :-
    catch(parse_wam_text(WamCode, Instrs), _, fail), !.
instrs_for(_, []).

% For per-predicate emission (used by tests inspecting individual
% predicates), produce a single-line interpreter-mode marker. Most of
% the runtime semantics live in the program-wide wam_cpp_setup function
% emitted by emit_setup_function/3 below.
compile_predicate_wrapper(Pred, Arity, _Instrs, _Options, CppCode) :-
    cpp_safe_function_name(Pred, SafeName),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    format(string(CppCode),
'// Interpreter-mode predicate ~w — instructions are emitted into the
// shared wam_cpp_setup() function below; this declaration only exists
// so per-predicate emission tests have something to assert against.
extern void wam_cpp_setup_~w(WamState&);
', [Key, SafeName]).

% ============================================================================
% Program-wide assembly: parse WAM text for every predicate, collect a
% mixed (label / instruction) stream, walk it once to compute absolute
% PCs, then emit a single wam_cpp_setup(WamState&) that pushes each
% instruction with label targets resolved to those PCs.
% ============================================================================

%% parse_pred_blocks(+WamText, -Items)
%  Thin alias for wam_text_parser:wam_text_to_items/2. Kept under the
%  pre-existing name since several call sites in this file still use
%  it; new code should call wam_text_to_items/2 directly.
parse_pred_blocks(WamText, Items) :-
    wam_text_to_items(WamText, Items).

%% walk_blocks(+AllItems, -Labels, -FlatInstrs)
%  Single pass: every label() records its PC; every instruction goes into
%  the flat list. Labels is a list of NameStr-PC pairs.
walk_blocks(Items, Labels, FlatInstrs) :-
    walk_blocks_(Items, 0, [], LabelsRev, [], FlatRev),
    reverse(LabelsRev, Labels),
    reverse(FlatRev, FlatInstrs).

walk_blocks_([], _, LAcc, LAcc, FAcc, FAcc).
walk_blocks_([label(N)|Rest], PC, LAcc, LOut, FAcc, FOut) :- !,
    walk_blocks_(Rest, PC, [N-PC|LAcc], LOut, FAcc, FOut).
walk_blocks_([Instr|Rest], PC, LAcc, LOut, FAcc, FOut) :-
    PC1 is PC + 1,
    walk_blocks_(Rest, PC1, LAcc, LOut, [Instr|FAcc], FOut).

%% lookup_label(+NameStr, +Labels, -PC)
lookup_label(NameStr, Labels, PC) :-
    ( member(NameStr-PC, Labels) -> true ; PC = 0 ).

%% emit_setup_function(+Predicates, +Options, -SetupCpp)
%  Top-level project assembly. Compiles each predicate to WAM text,
%  parses into blocks, concatenates, resolves PCs, and emits a single
%  wam_cpp_setup() function that populates WamState::instrs / labels.
emit_setup_function(Predicates, Options, SetupCpp) :-
    foreign_pred_keys_from_options(Options, _ForeignKeys),
    iso_errors_resolve_options(Options, IsoConfig),
    iso_errors_warn_multi_module(IsoConfig, Predicates),
    findall(Items, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamText),
              parse_pred_blocks(WamText, Items0),
              iso_errors_rewrite(IsoConfig, PI, Items0, Items)
            ),
            _, fail)
    ), PerPredItems),
    % Auto-inject builtin helper predicates (member/2, length/2). They
    % go BEFORE user predicates so user definitions of member/2 or
    % length/2 (rare) shadow the helpers via labels-map overwrite.
    helper_predicate_items(HelperItems),
    flatten_blocks([HelperItems|PerPredItems], AllItems),
    walk_blocks(AllItems, Labels, FlatInstrs0),
    % Append twelve trailing synthetic-return instructions for the
    % meta-call control-flow surface — see WamState for each pc''s
    % role.
    append(FlatInstrs0,
           [catch_return, negation_return, findall_collect,
            conj_return, disj_alt, if_then_commit, if_then_else,
            aggregate_next_group, dynamic_next_clause, sub_atom_next,
            body_next, retract_next],
           FlatInstrs),
    length(FlatInstrs0, CatchReturnPC),
    NegationReturnPC is CatchReturnPC + 1,
    FindallCollectPC is CatchReturnPC + 2,
    ConjReturnPC is CatchReturnPC + 3,
    DisjAltPC is CatchReturnPC + 4,
    IfThenCommitPC is CatchReturnPC + 5,
    IfThenElsePC is CatchReturnPC + 6,
    AggregateNextGroupPC is CatchReturnPC + 7,
    DynamicNextClausePC is CatchReturnPC + 8,
    SubAtomNextPC is CatchReturnPC + 9,
    BodyNextPC is CatchReturnPC + 10,
    RetractNextPC is CatchReturnPC + 11,
    findall(LabelLine, (
        member(NameStr-PC, Labels),
        format(atom(LabelLine),
               '    vm.labels["~w"] = ~w;', [NameStr, PC])
    ), LabelLines),
    atomic_list_concat(LabelLines, '\n', LabelBody),
    findall(InstrLine, (
        member(I, FlatInstrs),
        instr_to_setup_line(I, Labels, InstrLine)
    ), InstrLines),
    atomic_list_concat(InstrLines, '\n', InstrBody),
    length(FlatInstrs, Reserve),
    format(string(SetupCpp),
'void wam_cpp_setup(WamState& vm) {
    vm.instrs.clear();
    vm.labels.clear();
    vm.instrs.reserve(~w);
    vm.catch_return_pc = ~w;
    vm.negation_return_pc = ~w;
    vm.findall_collect_pc = ~w;
    vm.conj_return_pc = ~w;
    vm.disj_alt_pc = ~w;
    vm.if_then_commit_pc = ~w;
    vm.if_then_else_pc = ~w;
    vm.aggregate_next_group_pc = ~w;
    vm.dynamic_next_clause_pc = ~w;
    vm.sub_atom_next_pc = ~w;
    vm.body_next_pc = ~w;
    vm.retract_next_pc = ~w;
~w
~w
}
static const int _wam_cpp_setup_register = []() {
    Program::register_setup(&wam_cpp_setup);
    return 0;
}();
', [Reserve, CatchReturnPC, NegationReturnPC, FindallCollectPC,
    ConjReturnPC, DisjAltPC, IfThenCommitPC, IfThenElsePC,
    AggregateNextGroupPC, DynamicNextClausePC, SubAtomNextPC,
    BodyNextPC, RetractNextPC,
    LabelBody, InstrBody]).

% ============================================================================
% ISO error configuration
% ============================================================================
%
% Per-predicate ISO-error dispatch. See:
%   docs/design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md
%   docs/design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md
%
% The key tables iso_errors_default_to_iso/2 and iso_errors_default_to_lax/2
% are intentionally empty in this plumbing PR. iso_errors_rewrite/4 is a
% no-op when both tables are empty (every default key falls through
% unchanged). The first ISO-aware builtin lands in the follow-up PR.

%% iso_errors_default_to_iso(+DefaultKey, -IsoKey)
%  Maps a default builtin key (e.g. "is/2") to its ISO flavour
%  ("is_iso/2"). Empty in this PR. Declared `dynamic` so calls
%  with no clauses simply fail (rather than throwing
%  existence_error) and so follow-up PRs can populate it either as
%  asserted facts or as ordinary static clauses below this line.
%
%% iso_errors_default_to_lax(+DefaultKey, -LaxKey)
%  Maps a default key to its explicit-lax flavour. Same shape as
%  the iso table — empty for now since default and lax will share
%  an implementation in the runtime, so the rewrite stays a no-op
%  even when populated.
:- dynamic iso_errors_default_to_iso/2.
:- dynamic iso_errors_default_to_lax/2.
% Paired clauses (iso/lax for the same default key) interleave so
% each builtin reads as a logical unit. Tell the consult-time
% checker not to warn about that.
:- discontiguous iso_errors_default_to_iso/2.
:- discontiguous iso_errors_default_to_lax/2.

% is/2 — first ISO-aware builtin. ISO-mode predicates get their
% default is/2 calls rewritten to is_iso/2 (which throws on bad
% RHS); lax-mode predicates rewrite to is_lax/2 (a no-op rename
% since is/2 and is_lax/2 share the runtime body). Explicit
% is_iso/2 or is_lax/2 in user source survives the rewrite — see
% iso_errors_rewrite_item/3.
% is/2 — landed in PR #2084.
iso_errors_default_to_iso("is/2", "is_iso/2").
iso_errors_default_to_lax("is/2", "is_lax/2").
% Arithmetic comparisons. ISO body classifies args + throws via the
% same machinery is_iso/2 uses; lax body shares with default.
iso_errors_default_to_iso(">/2",   ">_iso/2").
iso_errors_default_to_lax(">/2",   ">_lax/2").
iso_errors_default_to_iso("</2",   "<_iso/2").
iso_errors_default_to_lax("</2",   "<_lax/2").
iso_errors_default_to_iso(">=/2",  ">=_iso/2").
iso_errors_default_to_lax(">=/2",  ">=_lax/2").
iso_errors_default_to_iso("=</2",  "=<_iso/2").
iso_errors_default_to_lax("=</2",  "=<_lax/2").
iso_errors_default_to_iso("=:=/2", "=:=_iso/2").
iso_errors_default_to_lax("=:=/2", "=:=_lax/2").
iso_errors_default_to_iso("=\\=/2","=\\=_iso/2").
iso_errors_default_to_lax("=\\=/2","=\\=_lax/2").
% succ/2 — bidirectional with proper ISO error throws.
iso_errors_default_to_iso("succ/2", "succ_iso/2").
iso_errors_default_to_lax("succ/2", "succ_lax/2").

%% iso_errors_resolve_options(+Options, -Config)
%  Merges the (optional) file config with inline options into one
%  iso_config(Default, Overrides) struct. Inline options override
%  file entries for the same PI.
iso_errors_resolve_options(Options, iso_config(Default, Overrides)) :-
    (   option(iso_errors_config(File), Options)
    ->  iso_errors_load_config(File, iso_config(FileDefault, FileOv))
    ;   FileDefault = false, FileOv = []
    ),
    iso_errors_inline_default(Options, FileDefault, Default),
    iso_errors_inline_overrides(Options, InlineOv),
    iso_errors_merge_overrides(FileOv, InlineOv, Overrides).

iso_errors_inline_default(Options, FileDefault, Default) :-
    (   member(iso_errors(M), Options),
        (M == true ; M == false)
    ->  Default = M
    ;   Default = FileDefault
    ).

iso_errors_inline_overrides(Options, InlineOv) :-
    findall(PI-Mode,
            ( member(iso_errors(PI, Mode), Options),
              (Mode == true ; Mode == false),
              iso_errors_valid_pi(PI)
            ),
            InlineOv).

% Accepts both bare Name/Arity and Module:Name/Arity.
iso_errors_valid_pi(Name/Arity) :- atom(Name), integer(Arity), Arity >= 0, !.
iso_errors_valid_pi(Module:Name/Arity) :-
    atom(Module), atom(Name), integer(Arity), Arity >= 0.

iso_errors_merge_overrides(FileOv, InlineOv, Merged) :-
    exclude(iso_errors_shadowed(InlineOv), FileOv, Kept),
    append(Kept, InlineOv, Merged).

iso_errors_shadowed(InlineOv, PI-_) :-
    member(InlinePI-_, InlineOv),
    iso_errors_pi_matches(InlinePI, PI), !.

% PI matching: bare Name/Arity matches Module:Name/Arity in any
% module; Module:Name/Arity matches only its own module.
iso_errors_pi_matches(PI, PI) :- !.
iso_errors_pi_matches(Name/Arity, _:Name/Arity) :-
    atom(Name), integer(Arity), !.
iso_errors_pi_matches(_:Name/Arity, Name/Arity) :-
    atom(Name), integer(Arity), !.

%% iso_errors_load_config(+File, -Config)
%  Reads a Prolog facts file and extracts iso_errors_default/1 +
%  iso_errors_override/2 terms. Unrecognised terms are silently
%  ignored. Returns iso_config(false, []) on I/O error.
iso_errors_load_config(File, iso_config(Default, Overrides)) :-
    catch(
        setup_call_cleanup(
            open(File, read, Stream),
            iso_errors_read_terms(Stream, RawTerms),
            close(Stream)),
        _,
        RawTerms = []),
    iso_errors_extract_terms(RawTerms, false, [], Default, RevOv),
    reverse(RevOv, Overrides).

iso_errors_read_terms(Stream, Terms) :-
    read_term(Stream, T, []),
    ( T == end_of_file
    -> Terms = []
    ; Terms = [T|Rest],
      iso_errors_read_terms(Stream, Rest)
    ).

iso_errors_extract_terms([], D, Ov, D, Ov).
iso_errors_extract_terms([T|Rest], D0, Ov0, D, Ov) :-
    (   T = iso_errors_default(NewD), (NewD == true ; NewD == false)
    ->  iso_errors_extract_terms(Rest, NewD, Ov0, D, Ov)
    ;   T = iso_errors_override(PI, Mode),
        (Mode == true ; Mode == false),
        iso_errors_valid_pi(PI)
    ->  iso_errors_extract_terms(Rest, D0, [PI-Mode|Ov0], D, Ov)
    ;   iso_errors_extract_terms(Rest, D0, Ov0, D, Ov)
    ).

%% iso_errors_mode_for(+Config, +PI, -Mode)
%  Resolves a predicate''s ISO mode. Bare Name/Arity in overrides
%  matches Module:Name/Arity in any module and vice versa. Falls
%  back to the config''s default when no override matches.
iso_errors_mode_for(iso_config(Default, Overrides), PI, Mode) :-
    (   member(OvPI-OvMode, Overrides),
        iso_errors_pi_matches(OvPI, PI)
    ->  Mode = OvMode
    ;   Mode = Default
    ).

%% iso_errors_warn_multi_module(+Config, +Predicates)
%  Emits a user_error warning for each bare override that matches
%  predicates from more than one module in the input list. Catches
%  the safe_div/2-in-two-modules footgun (SPECIFICATION §1).
iso_errors_warn_multi_module(iso_config(_, Overrides), Predicates) :-
    forall(member(OvPI-_, Overrides),
           iso_errors_check_override_scope(OvPI, Predicates)).

iso_errors_check_override_scope(Name/Arity, Predicates) :-
    atom(Name), integer(Arity), !,
    findall(M, ( member(P, Predicates),
                 iso_errors_pi_module(P, Name, Arity, M)
               ), Modules),
    list_to_set(Modules, Unique),
    (   Unique = [_, _ | _]
    ->  length(Unique, N),
        format(user_error,
               'Warning: iso_errors_override(~w/~w, _) matches ~w predicates~n         in different modules (~w).~n         Qualify with `mod:~w/~w` for module-scoped overrides.~n',
               [Name, Arity, N, Unique, Name, Arity])
    ;   true
    ).
iso_errors_check_override_scope(_, _).

iso_errors_pi_module(Module:Name/Arity, Name, Arity, Module) :- !.
iso_errors_pi_module(Name/Arity,         Name, Arity, user).

%% iso_errors_rewrite(+Config, +PI, +Items0, -Items)
%  Rewrites the default-form builtin_call keys for one predicate.
%  Explicit *_iso / *_lax keys pass through unchanged. With empty
%  key tables (this PR), the rewrite is a no-op.
iso_errors_rewrite(Config, PI, Items0, Items) :-
    iso_errors_mode_for(Config, PI, Mode),
    maplist(iso_errors_rewrite_item(Mode), Items0, Items).

% Two surfaces to rewrite for an ISO-relevant builtin:
%
% - builtin_call("is/2", _) — the direct call form (`X is Expr` in a
%   regular conjunction).
% - put_structure("is/2", _) — the data form when `X is Expr` appears
%   inside a meta-goal that takes the goal as a term (notably
%   catch(Goal, ...) — the WAM compiler builds `is(X, Expr)` into A1
%   and passes it through). Without this rewrite,
%   invoke_goal_as_call would dispatch the un-rewritten functor and
%   miss the surrounding predicate''s ISO mode.
%
% Also `call("is/2", _)` and `execute("is/2")` for completeness, even
% though the SWI WAM compiler emits these less commonly for is/2.
iso_errors_rewrite_item(true, builtin_call(Key, N), builtin_call(IsoKey, N)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, put_structure(Key, Reg), put_structure(IsoKey, Reg)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, call(Key, N), call(IsoKey, N)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, execute(Key), execute(IsoKey)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(false, builtin_call(Key, N), builtin_call(LaxKey, N)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, put_structure(Key, Reg), put_structure(LaxKey, Reg)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, call(Key, N), call(LaxKey, N)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, execute(Key), execute(LaxKey)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(_, Item, Item).

%% wam_cpp_iso_audit(+Predicates, +Options, -Audit)
%  Read-only inspection of how each predicate''s builtin_call sites
%  would resolve under the given options. Returns a list of
%  audit(PI, Mode, Sites). Each Site is:
%    site(PC, OrigKey, ResolvedKey, Source, WouldChangeOnFlip)
%  Source is one of: default, explicit_iso, explicit_lax.
wam_cpp_iso_audit(Predicates, Options, Audit) :-
    iso_errors_resolve_options(Options, Config),
    findall(audit(PI, Mode, Sites), (
        member(PI, Predicates),
        iso_errors_mode_for(Config, PI, Mode),
        iso_errors_audit_predicate(PI, Mode, Sites)
    ), Audit).

iso_errors_audit_predicate(PI, Mode, Sites) :-
    (   catch(
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamText),
              parse_pred_blocks(WamText, Items)
            ),
            _, fail)
    ->  iso_errors_audit_walk(Items, 0, Mode, [], SitesRev),
        reverse(SitesRev, Sites)
    ;   Sites = []
    ).

iso_errors_audit_walk([], _, _, Acc, Acc).
iso_errors_audit_walk([label(_)|Rest], PC, Mode, Acc, Out) :- !,
    iso_errors_audit_walk(Rest, PC, Mode, Acc, Out).
iso_errors_audit_walk([builtin_call(Key, _)|Rest], PC, Mode, Acc, Out) :- !,
    iso_errors_audit_classify(Key, Mode, Source, Resolved, Flip),
    Site = site(PC, Key, Resolved, Source, Flip),
    PC1 is PC + 1,
    iso_errors_audit_walk(Rest, PC1, Mode, [Site|Acc], Out).
iso_errors_audit_walk([_|Rest], PC, Mode, Acc, Out) :-
    PC1 is PC + 1,
    iso_errors_audit_walk(Rest, PC1, Mode, Acc, Out).

% Classify a builtin_call key. Explicit *_iso/N or *_lax/N suffix
% always wins; otherwise it's a default site whose resolution
% depends on Mode and the (currently empty) swap tables.
iso_errors_audit_classify(Key, _Mode, explicit_iso, Key, false) :-
    iso_errors_key_has_suffix(Key, '_iso'), !.
iso_errors_audit_classify(Key, _Mode, explicit_lax, Key, false) :-
    iso_errors_key_has_suffix(Key, '_lax'), !.
iso_errors_audit_classify(Key, Mode, default, Resolved, Flip) :-
    iso_errors_resolve_default(Key, Mode, Resolved),
    iso_errors_other_mode(Mode, OtherMode),
    iso_errors_resolve_default(Key, OtherMode, OtherResolved),
    ( Resolved == OtherResolved -> Flip = false ; Flip = true ).

iso_errors_resolve_default(Key, true, IsoKey) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_resolve_default(Key, false, LaxKey) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_resolve_default(Key, _, Key).

iso_errors_other_mode(true,  false).
iso_errors_other_mode(false, true).

% Match keys like "foo_iso/N" or "foo_lax/N" — Suffix is e.g.
% "_iso". Splits on the / first, then checks the name component.
% Keys come through parse_pred_blocks as strings; convert defensively
% so atom-shaped table entries also work.
iso_errors_key_has_suffix(Key, Suffix) :-
    ( atom(Key) -> atom_string(Key, KS) ; KS = Key ),
    split_string(KS, "/", "", Parts),
    Parts = [Name | _],
    string_concat(_, Suffix, Name).

%% wam_cpp_iso_audit_report(+Audit)
%  Human-readable pretty-print of an audit result.
wam_cpp_iso_audit_report([]).
wam_cpp_iso_audit_report([audit(PI, Mode, Sites)|Rest]) :-
    format('~w [~w]~n', [PI, Mode]),
    (   Sites == []
    ->  format('  (no builtin_call sites)~n', [])
    ;   forall(member(site(PC, Orig, Res, Src, Flip), Sites),
               format('  pc=~w  ~w -> ~w  (~w)  flip-changes=~w~n',
                      [PC, Orig, Res, Src, Flip]))
    ),
    wam_cpp_iso_audit_report(Rest).

%% helper_predicate_items(-Items)
%  Canonical WAM for builtin "library" predicates the user can call via
%  builtin_call but that aren''t implemented directly by builtin().
%  These instructions match what compile_predicate_to_wam emits for the
%  standard Prolog definitions of member/2 and length/2 (verified via
%  probe during this PR''s development). When a user predicate of the
%  same name is also emitted, the user''s label-map entry overwrites
%  ours and the helper instructions become harmless dead code.
helper_predicate_items(Items) :-
    Items = [
        % --- member/2 ----------------------------------------------------
        % member(X, [X|_]).
        % member(X, [_|T]) :- member(X, T).
        label("member/2"),
        try_me_else("L_cpp_member_2_2"),
        get_variable("X1", "A1"),
        get_list("A2"),
        unify_value("X1"),
        unify_variable("X2"),
        proceed,
        label("L_cpp_member_2_2"),
        trust_me,
        allocate,
        get_variable("X1", "A1"),
        get_list("A2"),
        unify_variable("X2"),
        unify_variable("X3"),
        put_value("X1", "A1"),
        put_value("X3", "A2"),
        deallocate,
        execute("member/2"),
        % --- select/3 ----------------------------------------------------
        % select(X, [X|T], T).
        % select(X, [Y|T], [Y|R]) :- select(X, T, R).
        label("select/3"),
        try_me_else("L_cpp_select_3_2"),
        get_variable("X1", "A1"),
        get_list("A2"),
        unify_value("X1"),
        unify_variable("X2"),
        get_value("X2", "A3"),
        proceed,
        label("L_cpp_select_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_value("X3"),
        unify_variable("Y3"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("select/3"),
        % --- maplist/2 ---------------------------------------------------
        % maplist(_, []).
        % maplist(Goal, [X|Xs]) :- call(Goal, X), maplist(Goal, Xs).
        label("maplist/2"),
        try_me_else("L_cpp_maplist_2_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        proceed,
        label("L_cpp_maplist_2_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        put_value("Y1", "A1"),
        put_value("X3", "A2"),
        call("call/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        deallocate,
        execute("maplist/2"),
        % --- maplist/3 ---------------------------------------------------
        % maplist(_, [], []).
        % maplist(Goal, [X|Xs], [Y|Ys]) :-
        %     call(Goal, X, Y), maplist(Goal, Xs, Ys).
        label("maplist/3"),
        try_me_else("L_cpp_maplist_3_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        proceed,
        label("L_cpp_maplist_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_variable("X5"),
        unify_variable("Y3"),
        put_value("Y1", "A1"),
        put_value("X4", "A2"),
        put_value("X5", "A3"),
        call("call/3", "3"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("maplist/3"),
        % --- maplist/4 ---------------------------------------------------
        % maplist(_, [], [], []).
        % maplist(Goal, [X|Xs], [Y|Ys], [Z|Zs]) :-
        %     call(Goal, X, Y, Z), maplist(Goal, Xs, Ys, Zs).
        label("maplist/4"),
        try_me_else("L_cpp_maplist_4_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        get_constant("[]", "A4"),
        proceed,
        label("L_cpp_maplist_4_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X5"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_variable("X6"),
        unify_variable("Y3"),
        get_list("A4"),
        unify_variable("X7"),
        unify_variable("Y4"),
        put_value("Y1", "A1"),
        put_value("X5", "A2"),
        put_value("X6", "A3"),
        put_value("X7", "A4"),
        call("call/4", "4"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        put_value("Y4", "A4"),
        deallocate,
        execute("maplist/4"),
        % --- maplist/5 ---------------------------------------------------
        % maplist(_, [], [], [], []).
        % maplist(Goal, [X|Xs], [Y|Ys], [Z|Zs], [W|Ws]) :-
        %     call(Goal, X, Y, Z, W), maplist(Goal, Xs, Ys, Zs, Ws).
        label("maplist/5"),
        try_me_else("L_cpp_maplist_5_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        get_constant("[]", "A4"),
        get_constant("[]", "A5"),
        proceed,
        label("L_cpp_maplist_5_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X6"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_variable("X7"),
        unify_variable("Y3"),
        get_list("A4"),
        unify_variable("X8"),
        unify_variable("Y4"),
        get_list("A5"),
        unify_variable("X9"),
        unify_variable("Y5"),
        put_value("Y1", "A1"),
        put_value("X6", "A2"),
        put_value("X7", "A3"),
        put_value("X8", "A4"),
        put_value("X9", "A5"),
        call("call/5", "5"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        put_value("Y4", "A4"),
        put_value("Y5", "A5"),
        deallocate,
        execute("maplist/5"),
        % --- include/3 ---------------------------------------------------
        % include(_, [], []).
        % include(G, [X|Xs], [X|R]) :- call(G, X), include(G, Xs, R).
        % include(G, [X|Xs], R) :- \+ call(G, X), include(G, Xs, R).
        % Three-clause non-cut form: clauses 2 and 3 are mutually
        % exclusive (call succeeds OR fails), so the predicate is
        % effectively deterministic — the extra CP from try_me_else is
        % harmless. Skips ITE/cut complexity in the hand-crafted WAM.
        label("include/3"),
        try_me_else("L_cpp_include_3_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        proceed,
        label("L_cpp_include_3_2"),
        retry_me_else("L_cpp_include_3_3"),
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_value("X3"),
        unify_variable("Y3"),
        put_value("Y1", "A1"),
        put_value("X3", "A2"),
        call("call/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("include/3"),
        label("L_cpp_include_3_3"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_variable("Y3", "A3"),
        put_structure("call/2", "A1"),
        set_value("Y1"),
        set_value("X3"),
        builtin_call("\\+/1", "1"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("include/3"),
        % --- exclude/3 ---------------------------------------------------
        % Mirror of include: clause 2 INCLUDES X when goal FAILS;
        % clause 3 keeps the recursion going when goal SUCCEEDS.
        label("exclude/3"),
        try_me_else("L_cpp_exclude_3_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        proceed,
        label("L_cpp_exclude_3_2"),
        retry_me_else("L_cpp_exclude_3_3"),
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_value("X3"),
        unify_variable("Y3"),
        put_structure("call/2", "A1"),
        set_value("Y1"),
        set_value("X3"),
        builtin_call("\\+/1", "1"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("exclude/3"),
        label("L_cpp_exclude_3_3"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_variable("Y3", "A3"),
        put_value("Y1", "A1"),
        put_value("X3", "A2"),
        call("call/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("exclude/3"),
        % --- partition/4 -------------------------------------------------
        % partition(_, [], [], []).
        % partition(G, [X|Xs], [X|In], Ex) :-
        %     call(G, X), partition(G, Xs, In, Ex).
        % partition(G, [X|Xs], In, [X|Ex]) :-
        %     \+ call(G, X), partition(G, Xs, In, Ex).
        label("partition/4"),
        try_me_else("L_cpp_partition_4_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        get_constant("[]", "A4"),
        proceed,
        label("L_cpp_partition_4_2"),
        retry_me_else("L_cpp_partition_4_3"),
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_value("X4"),
        unify_variable("Y3"),
        get_variable("Y4", "A4"),
        put_value("Y1", "A1"),
        put_value("X4", "A2"),
        call("call/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        put_value("Y4", "A4"),
        deallocate,
        execute("partition/4"),
        label("L_cpp_partition_4_3"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_variable("Y3", "A3"),
        get_list("A4"),
        unify_value("X4"),
        unify_variable("Y4"),
        put_structure("call/2", "A1"),
        set_value("Y1"),
        set_value("X4"),
        builtin_call("\\+/1", "1"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        put_value("Y4", "A4"),
        deallocate,
        execute("partition/4"),
        % --- foldl/4 -----------------------------------------------------
        % foldl(_, [], V, V).
        % foldl(G, [X|Xs], V0, V) :-
        %     call(G, X, V0, V1), foldl(G, Xs, V1, V).
        label("foldl/4"),
        try_me_else("L_cpp_foldl_4_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_variable("X2", "A3"),
        get_value("X2", "A4"),
        proceed,
        label("L_cpp_foldl_4_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_variable("Y3", "A3"),
        get_variable("Y4", "A4"),
        put_value("Y1", "A1"),
        put_value("X3", "A2"),
        put_value("Y3", "A3"),
        put_variable("Y5", "A4"),
        call("call/4", "4"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y5", "A3"),
        put_value("Y4", "A4"),
        deallocate,
        execute("foldl/4"),
        % --- foldl/5 -----------------------------------------------------
        % foldl(_, [], [], V, V).
        % foldl(G, [X|Xs], [Y|Ys], V0, V) :-
        %     call(G, X, Y, V0, V1), foldl(G, Xs, Ys, V1, V).
        label("foldl/5"),
        try_me_else("L_cpp_foldl_5_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        get_variable("X2", "A4"),
        get_value("X2", "A5"),
        proceed,
        label("L_cpp_foldl_5_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_variable("X5"),
        unify_variable("Y3"),
        get_variable("Y4", "A4"),
        get_variable("Y5", "A5"),
        put_value("Y1", "A1"),
        put_value("X4", "A2"),
        put_value("X5", "A3"),
        put_value("Y4", "A4"),
        put_variable("Y6", "A5"),
        call("call/5", "5"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        put_value("Y6", "A4"),
        put_value("Y5", "A5"),
        deallocate,
        execute("foldl/5"),
        % --- pairs_keys/2 ------------------------------------------------
        % pairs_keys([], []).
        % pairs_keys([K-_|T], [K|KT]) :- pairs_keys(T, KT).
        label("pairs_keys/2"),
        try_me_else("L_cpp_pairs_keys_2_2"),
        get_constant("[]", "A1"),
        get_constant("[]", "A2"),
        proceed,
        label("L_cpp_pairs_keys_2_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_structure("-/2", "X3"),
        unify_variable("X4"),
        unify_variable("X5"),
        get_list("A2"),
        unify_value("X4"),
        unify_variable("Y3"),
        put_value("Y2", "A1"),
        put_value("Y3", "A2"),
        deallocate,
        execute("pairs_keys/2"),
        % --- pairs_values/2 ----------------------------------------------
        % pairs_values([], []).
        % pairs_values([_-V|T], [V|VT]) :- pairs_values(T, VT).
        label("pairs_values/2"),
        try_me_else("L_cpp_pairs_values_2_2"),
        get_constant("[]", "A1"),
        get_constant("[]", "A2"),
        proceed,
        label("L_cpp_pairs_values_2_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X3"),
        unify_variable("Y2"),
        get_structure("-/2", "X3"),
        unify_variable("X4"),
        unify_variable("X5"),
        get_list("A2"),
        unify_value("X5"),
        unify_variable("Y3"),
        put_value("Y2", "A1"),
        put_value("Y3", "A2"),
        deallocate,
        execute("pairs_values/2"),
        % --- pairs_keys_values/3 -----------------------------------------
        % pairs_keys_values([], [], []).
        % pairs_keys_values([K-V|T], [K|KT], [V|VT]) :-
        %     pairs_keys_values(T, KT, VT).
        label("pairs_keys_values/3"),
        try_me_else("L_cpp_pairs_kv_3_2"),
        get_constant("[]", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        proceed,
        label("L_cpp_pairs_kv_3_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_structure("-/2", "X4"),
        unify_variable("X5"),
        unify_variable("X6"),
        get_list("A2"),
        unify_value("X5"),
        unify_variable("Y3"),
        get_list("A3"),
        unify_value("X6"),
        unify_variable("Y4"),
        put_value("Y2", "A1"),
        put_value("Y3", "A2"),
        put_value("Y4", "A3"),
        deallocate,
        execute("pairs_keys_values/3"),
        % --- length/2 ----------------------------------------------------
        % length([], 0).
        % length([_|T], N) :- length(T, M), N is M + 1.
        label("length/2"),
        try_me_else("L_cpp_length_2_2"),
        get_constant("[]", "A1"),
        get_constant("0", "A2"),
        proceed,
        label("L_cpp_length_2_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X3"),
        unify_variable("X4"),
        get_variable("Y1", "A2"),
        put_value("X4", "A1"),
        put_variable("Y2", "A2"),
        call("length/2", "2"),
        put_value("Y1", "A1"),
        put_structure("+/2", "A2"),
        set_value("Y2"),
        set_constant("1"),
        builtin_call("is/2", "2"),
        deallocate,
        proceed,
        % --- append/3 ----------------------------------------------------
        % append([], L, L).
        % append([H|T], L, [H|R]) :- append(T, L, R).
        label("append/3"),
        try_me_else("L_cpp_append_3_2"),
        get_constant("[]", "A1"),
        get_variable("X1", "A2"),
        get_value("X1", "A3"),
        proceed,
        label("L_cpp_append_3_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X1"),
        unify_variable("X2"),
        get_variable("X3", "A2"),
        get_list("A3"),
        unify_value("X1"),
        unify_variable("X4"),
        put_value("X2", "A1"),
        put_value("X3", "A2"),
        put_value("X4", "A3"),
        deallocate,
        execute("append/3"),
        % --- reverse/2 + reverse_acc/3 -----------------------------------
        % reverse(L, R) :- reverse_acc(L, [], R).
        % reverse_acc([], Acc, Acc).
        % reverse_acc([H|T], Acc, R) :- reverse_acc(T, [H|Acc], R).
        label("reverse/2"),
        allocate,
        get_variable("X1", "A1"),
        get_variable("X2", "A2"),
        put_value("X1", "A1"),
        put_constant("[]", "A2"),
        put_value("X2", "A3"),
        deallocate,
        execute("reverse_acc/3"),
        label("reverse_acc/3"),
        try_me_else("L_cpp_revacc_3_2"),
        get_constant("[]", "A1"),
        get_variable("X1", "A2"),
        get_value("X1", "A3"),
        proceed,
        label("L_cpp_revacc_3_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X1"),
        unify_variable("X2"),
        get_variable("X3", "A2"),
        get_variable("X4", "A3"),
        put_value("X2", "A1"),
        put_list("A2"),
        set_value("X1"),
        set_value("X3"),
        put_value("X4", "A3"),
        deallocate,
        execute("reverse_acc/3"),
        % --- last/2 ------------------------------------------------------
        % last([X], X).
        % last([_|T], X) :- last(T, X).
        label("last/2"),
        try_me_else("L_cpp_last_2_2"),
        get_list("A1"),
        unify_variable("X1"),
        unify_constant("[]"),
        get_value("X1", "A2"),
        proceed,
        label("L_cpp_last_2_2"),
        trust_me,
        allocate,
        get_list("A1"),
        unify_variable("X1"),
        unify_variable("X2"),
        get_variable("X3", "A2"),
        put_value("X2", "A1"),
        put_value("X3", "A2"),
        deallocate,
        execute("last/2"),
        % --- nth0/3 ------------------------------------------------------
        % nth0(0, [X|_], X).
        % nth0(N, [_|T], X) :- N > 0, M is N - 1, nth0(M, T, X).
        label("nth0/3"),
        try_me_else("L_cpp_nth0_3_2"),
        get_constant("0", "A1"),
        get_list("A2"),
        unify_variable("X1"),
        unify_variable("X2"),
        get_value("X1", "A3"),
        proceed,
        label("L_cpp_nth0_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X5"),
        unify_variable("Y3"),
        get_variable("Y4", "A3"),
        put_value("Y1", "A1"),
        put_constant("0", "A2"),
        builtin_call(">/2", "2"),
        put_variable("Y2", "A1"),
        put_structure("+/2", "A2"),
        set_value("Y1"),
        set_constant("-1"),
        builtin_call("is/2", "2"),
        put_value("Y2", "A1"),
        put_value("Y3", "A2"),
        put_value("Y4", "A3"),
        deallocate,
        execute("nth0/3"),
        % --- between/3 ---------------------------------------------------
        % between(L, H, L) :- L =< H.
        % between(L, H, X) :- L < H, L1 is L + 1, between(L1, H, X).
        label("between/3"),
        try_me_else("L_cpp_between_3_2"),
        get_variable("X1", "A1"),
        get_variable("X2", "A2"),
        get_value("X1", "A3"),
        put_value("X1", "A1"),
        put_value("X2", "A2"),
        builtin_call("=</2", "2"),
        proceed,
        label("L_cpp_between_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_variable("Y3", "A2"),
        get_variable("Y4", "A3"),
        put_value("Y1", "A1"),
        put_value("Y3", "A2"),
        builtin_call("</2", "2"),
        put_variable("Y2", "A1"),
        put_structure("+/2", "A2"),
        set_value("Y1"),
        set_constant("1"),
        builtin_call("is/2", "2"),
        put_value("Y2", "A1"),
        put_value("Y3", "A2"),
        put_value("Y4", "A3"),
        deallocate,
        execute("between/3"),
        % --- maplist/2 ---------------------------------------------------
        % maplist(_, []).
        % maplist(P, [X|Xs]) :- call(P, X), maplist(P, Xs).
        % Higher-order: dispatches user predicate P over each list
        % element via call/2 (added in PR #2094). Verifies the
        % helper-injection mechanism scales to higher-order use.
        label("maplist/2"),
        try_me_else("L_cpp_maplist_2_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        proceed,
        label("L_cpp_maplist_2_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X3"),
        unify_variable("Y2"),
        put_value("Y1", "A1"),
        put_value("X3", "A2"),
        call("call/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        deallocate,
        execute("maplist/2"),
        % --- maplist/3 ---------------------------------------------------
        % maplist(_, [], []).
        % maplist(P, [X|Xs], [Y|Ys]) :- call(P, X, Y), maplist(P, Xs, Ys).
        % Paired list mapping — the canonical "transform each X to Y
        % via P" pattern. With both lists ground, succeeds iff P
        % holds for every paired (X, Y); with the second list
        % unbound, builds Ys by calling P(X, Y) per element.
        label("maplist/3"),
        try_me_else("L_cpp_maplist_3_2"),
        get_variable("X1", "A1"),
        get_constant("[]", "A2"),
        get_constant("[]", "A3"),
        proceed,
        label("L_cpp_maplist_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X4"),
        unify_variable("Y2"),
        get_list("A3"),
        unify_variable("X5"),
        unify_variable("Y3"),
        put_value("Y1", "A1"),
        put_value("X4", "A2"),
        put_value("X5", "A3"),
        call("call/3", "3"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        put_value("Y3", "A3"),
        deallocate,
        execute("maplist/3")
    ].

flatten_blocks([], []).
flatten_blocks([Items|Rest], All) :-
    flatten_blocks(Rest, RestAll),
    append(Items, RestAll, All).

%% instr_to_setup_line(+Instr, +Labels, -Line)
%  Emit a single `vm.instrs.push_back(Instruction::...);` line. Label
%  references inside try_me_else / retry_me_else / jump are looked up.
instr_to_setup_line(try_me_else(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::TryMeElse(~w));', [PC]).
instr_to_setup_line(retry_me_else(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::RetryMeElse(~w));', [PC]).
instr_to_setup_line(jump(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Jump(~w));', [PC]).
instr_to_setup_line(switch_on_constant(Entries), Labels, Line) :- !,
    parse_switch_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnConstant({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_constant_a2(Entries), Labels, Line) :- !,
    % Treated as a no-op for now (interpreter falls through to the
    % try_me_else chain on A2 dispatch). Emit as a comment.
    parse_switch_entries(Entries, Labels, _EntriesCpp),
    format(atom(Line),
           '    // switch_on_constant_a2 ~w (no-op; falls through)', [Entries]).
instr_to_setup_line(switch_on_structure(Entries), Labels, Line) :- !,
    parse_switch_struct_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnStructure({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_term(Tokens), Labels, Line) :- !,
    parse_switch_term(Tokens, Labels, ConstsCpp, StructsCpp, ListPC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnTerm({~w}, {~w}, ~w));',
           [ConstsCpp, StructsCpp, ListPC]).
instr_to_setup_line(catch_return, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::CatchReturn());'.
instr_to_setup_line(negation_return, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::NegationReturn());'.
instr_to_setup_line(findall_collect, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::FindallCollect());'.
instr_to_setup_line(conj_return, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::ConjReturn());'.
instr_to_setup_line(disj_alt, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::DisjAlt());'.
instr_to_setup_line(if_then_commit, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::IfThenCommit());'.
instr_to_setup_line(if_then_else, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::IfThenElse());'.
instr_to_setup_line(aggregate_next_group, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::AggregateNextGroup());'.
instr_to_setup_line(dynamic_next_clause, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::DynamicNextClause());'.
instr_to_setup_line(sub_atom_next, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::SubAtomNext());'.
instr_to_setup_line(body_next, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::BodyNext());'.
instr_to_setup_line(retract_next, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::RetractNext());'.
instr_to_setup_line(Instr, _Labels, Line) :-
    wam_instruction_to_cpp_literal(Instr, Lit),
    format(atom(Line), '    vm.instrs.push_back(~w);', [Lit]).

label_resolve(L, Labels, PC) :-
    to_string(L, LS),
    lookup_label(LS, Labels, PC).

% =====================================================================
% Switch-table parsing
% =====================================================================
% Switch entries arrive as tokens like "red:default" or "foo/1:L_xxx".
% We split each on the LAST `:`, distinguishing the special sentinels
% "default" (= fall through, SWITCH_DEFAULT) and "none" (= fail,
% SWITCH_NONE) from real label names (resolved to PCs).

%% switch_pc_for(+LabelStr, +Labels, -CppPc)
%  CppPc is either a numeric PC or one of the sentinel C++ identifiers
%  Instruction::SWITCH_DEFAULT / Instruction::SWITCH_NONE.
switch_pc_for("default", _, 'Instruction::SWITCH_DEFAULT') :- !.
switch_pc_for("none",    _, 'Instruction::SWITCH_NONE')    :- !.
switch_pc_for(L, Labels, PC) :- lookup_label(L, Labels, PC).

%% split_entry(+Entry, -Key, -LabelStr)
%  Split "key:label" on the LAST `:` so functor keys like "foo/1" stay
%  intact. Returns Key (string) and LabelStr (string).
split_entry(Entry, Key, Label) :-
    string_codes(Entry, Codes),
    last_colon_index(Codes, 0, -1, Idx),
    Idx >= 0,
    length(Pre, Idx),
    append(Pre, [_|Post], Codes),
    string_codes(KeyStr, Pre),
    string_codes(Label, Post),
    Key = KeyStr.

last_colon_index([], _, Acc, Acc).
last_colon_index([0':|T], I, _, Out) :- !,
    I1 is I + 1, last_colon_index(T, I1, I, Out).
last_colon_index([_|T], I, Acc, Out) :-
    I1 is I + 1, last_colon_index(T, I1, Acc, Out).

%% key_to_cpp_value(+KeyStr, -CppLiteral)
%  Atom / Integer / Float literal for a switch key.
key_to_cpp_value(KeyStr, Lit) :-
    (   number_string(N, KeyStr), integer(N)
    ->  format(atom(Lit), 'Value::Integer(~w)', [N])
    ;   number_string(F, KeyStr), float(F)
    ->  format(atom(Lit), 'Value::Float(~w)', [F])
    ;   escape_cpp_string(KeyStr, Esc),
        format(atom(Lit), 'Value::Atom("~w")', [Esc])
    ).

%% parse_switch_entries(+Entries, +Labels, -CppPairs)
%  CppPairs is a comma-joined list of "{ValueLiteral, PC}".
parse_switch_entries(Entries, Labels, CppPairs) :-
    findall(Pair, (
        member(E, Entries),
        split_entry(E, KeyStr, LabelStr),
        key_to_cpp_value(KeyStr, KCpp),
        switch_pc_for(LabelStr, Labels, PC),
        format(atom(Pair), '{~w, ~w}', [KCpp, PC])
    ), Pairs),
    atomic_list_concat(Pairs, ', ', CppPairs).

%% parse_switch_struct_entries(+Entries, +Labels, -CppPairs)
%  Like parse_switch_entries but keys are functor strings (e.g. "foo/1").
parse_switch_struct_entries(Entries, Labels, CppPairs) :-
    findall(Pair, (
        member(E, Entries),
        split_entry(E, KeyStr, LabelStr),
        escape_cpp_string(KeyStr, EscKey),
        switch_pc_for(LabelStr, Labels, PC),
        format(atom(Pair), '{"~w", ~w}', [EscKey, PC])
    ), Pairs),
    atomic_list_concat(Pairs, ', ', CppPairs).

%% parse_switch_term(+Tokens, +Labels, -ConstsCpp, -StructsCpp, -ListPC)
%  Token format: <NConsts> <consts...> <NStructs> <structs...> <listLabel>
parse_switch_term(Tokens, Labels, ConstsCpp, StructsCpp, ListPC) :-
    Tokens = [NCStr | Rest1],
    number_string(NC, NCStr),
    length(ConstEntries, NC),
    append(ConstEntries, Rest2, Rest1),
    Rest2 = [NSStr | Rest3],
    number_string(NS, NSStr),
    length(StructEntries, NS),
    append(StructEntries, [ListLabel], Rest3),
    parse_switch_entries(ConstEntries, Labels, ConstsCpp),
    parse_switch_struct_entries(StructEntries, Labels, StructsCpp),
    switch_pc_for(ListLabel, Labels, ListPC).

% ============================================================================
% Project-level assembly
% ============================================================================

%% write_wam_cpp_project(+Predicates, +Options, +ProjectDir)
%  Materialise a self-contained C++ project at ProjectDir/cpp/.
%  Layout:
%    cpp/wam_runtime.h
%    cpp/wam_runtime.cpp
%    cpp/generated_program.cpp
write_wam_cpp_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'cpp', CppDir),
    make_directory_path(CppDir),
    compile_wam_runtime_header_to_cpp(Options, HeaderCode),
    directory_file_path(CppDir, 'wam_runtime.h', HeaderPath),
    write_text_file(HeaderPath, HeaderCode),
    compile_wam_runtime_to_cpp(Options, RuntimeCode),
    directory_file_path(CppDir, 'wam_runtime.cpp', RuntimePath),
    write_text_file(RuntimePath, RuntimeCode),
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    emit_setup_function(Predicates, Options, SetupCpp),
    format(string(ProgramCode),
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
#include "wam_runtime.h"

~w

// ----------------------------------------------------------------------
// Program assembly: wam_cpp_setup() populates the WamState with all
// predicates concatenated into one instruction vector, with absolute PCs
// resolved for try_me_else / retry_me_else / jump targets.
// ----------------------------------------------------------------------
~w
', [PredicatesCode, SetupCpp]),
    directory_file_path(CppDir, 'generated_program.cpp', ProgramPath),
    write_text_file(ProgramPath, ProgramCode),
    (   option(emit_main(true), Options, false)
    ->  emit_main_shim(MainCode),
        directory_file_path(CppDir, 'main.cpp', MainPath),
        write_text_file(MainPath, MainCode)
    ;   true
    ).

%% emit_main_shim(-MainCpp)
%  A tiny CLI driver: argv[1] is the predicate key, remaining args are
%  parsed as Prolog-style terms (atom, integer, compound foo(a,b),
%  list [a,b,c]). Prints "true" / "false" and exits 0 / 1.
emit_main_shim('// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
#include "wam_runtime.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Parser {
    const char* p;

    void skip_ws() { while (*p == \' \' || *p == \'\\t\') ++p; }

    bool eof() const { return *p == \'\\0\'; }

    bool peek(char c) { skip_ws(); return *p == c; }

    bool eat(char c) {
        skip_ws();
        if (*p == c) { ++p; return true; }
        return false;
    }

    std::string read_ident() {
        std::string r;
        while (std::isalnum(static_cast<unsigned char>(*p))
               || *p == \'_\' || *p == \'.\') {
            r.push_back(*p++);
        }
        return r;
    }

    Value parse_atom_or_int(const std::string& name) {
        // name is non-empty and starts with non-digit / non-bracket / non-paren.
        if (!name.empty()
            && (std::isdigit(static_cast<unsigned char>(name[0])) || name[0] == \'-\')) {
            try { return Value::Integer(std::stoll(name)); }
            catch (...) { return Value::Atom(name); }
        }
        return Value::Atom(name);
    }

    Value parse_term() {
        skip_ws();
        if (*p == \'[\') {
            ++p;
            return parse_list_tail();
        }
        // Read an ident / number.
        std::string name = read_ident();
        if (eat(\'(\')) {
            std::vector<wam_cpp::CellPtr> args;
            if (!eat(\')\')) {
                args.push_back(std::make_shared<Value>(parse_term()));
                while (eat(\',\')) {
                    args.push_back(std::make_shared<Value>(parse_term()));
                }
                eat(\')\');
            }
            std::string functor = name + "/" + std::to_string(args.size());
            return Value::Compound(functor, std::move(args));
        }
        return parse_atom_or_int(name);
    }

    // After "[" has been consumed: parse "]" or "elem, elem, ... ]".
    Value parse_list_tail() {
        skip_ws();
        if (eat(\']\')) return Value::Atom("[]");
        std::vector<Value> elems;
        elems.push_back(parse_term());
        while (eat(\',\')) elems.push_back(parse_term());
        eat(\']\');
        // Build right-associative cons chain: [|]/2(h, [|]/2(...,[]))
        Value tail = Value::Atom("[]");
        for (auto it = elems.rbegin(); it != elems.rend(); ++it) {
            std::vector<wam_cpp::CellPtr> args;
            args.push_back(std::make_shared<Value>(*it));
            args.push_back(std::make_shared<Value>(std::move(tail)));
            tail = Value::Compound("[|]/2", std::move(args));
        }
        return tail;
    }
};

Value parse_arg(const char* s) {
    Parser pr{s};
    return pr.parse_term();
}

} // anonymous

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s pred/arity [arg ...]\\n", argv[0]);
        return 2;
    }
    WamState vm;
    Program::apply_setup(vm);
    std::vector<Value> args;
    for (int i = 2; i < argc; ++i) args.push_back(parse_arg(argv[i]));
    bool ok = vm.query(argv[1], args);
    std::printf("%s\\n", ok ? "true" : "false");
    return ok ? 0 : 1;
}
').

compile_predicates_for_project(Predicates, Options, PredicatesCode) :-
    foreign_pred_keys_from_options(Options, ForeignKeys),
    (   member(foreign_pred_keys(_), Options)
    ->  Options1 = Options
    ;   Options1 = [foreign_pred_keys(ForeignKeys)|Options]
    ),
    findall(Code, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamCode),
              compile_wam_predicate_to_cpp(PI, WamCode, Options1, Code)
            ),
            _Err,
            fail)
    ), Codes),
    atomic_list_concat(Codes, '\n\n', PredicatesCode).

foreign_pred_keys_from_options(Options, Keys) :-
    (   member(foreign_pred_keys(Keys0), Options)
    ->  Keys = Keys0
    ;   Keys = []
    ).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        write(Stream, Content),
        close(Stream)
    ).

% ============================================================================
% C++ runtime header & source — bundled inline so the project is
% self-contained without a template engine. The runtime is intentionally
% compact: just enough Instruction / Value / WamState surface to satisfy
% the lowered emitter and a minimal step() interpreter loop.
% ============================================================================

compile_wam_runtime_header_to_cpp(_Options,
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
//
// WAM runtime header. Provides the Value, Instruction, ChoicePoint and
// WamState types referenced by code emitted from wam_cpp_lowered_emitter.pl
// and wam_cpp_target.pl.
//
// Heap model: every "register" or compound argument is a std::shared_ptr<Value>
// (alias Cell). Binding a variable mutates *cell so all aliases see the
// update — equivalent to classic WAM heap refs but without explicit indices.
// The trail records (cell, prev_value) pairs and undoes them on backtrack.

#ifndef UNIFYWEAVER_WAM_CPP_RUNTIME_H
#define UNIFYWEAVER_WAM_CPP_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace wam_cpp {

struct Value;
using Cell = Value;
using CellPtr = std::shared_ptr<Cell>;

struct Value {
    enum class Tag { Uninit, Atom, Integer, Float, Unbound, Compound };
    Tag tag = Tag::Uninit;
    std::string s;     // Atom name / Unbound name / Compound functor ("f/n")
    std::int64_t i = 0;
    double f = 0.0;
    std::vector<CellPtr> args; // Compound only

    Value() = default;
    static Value Atom(std::string a)    { Value v; v.tag = Tag::Atom;    v.s = std::move(a); return v; }
    static Value Integer(std::int64_t n){ Value v; v.tag = Tag::Integer; v.i = n;             return v; }
    static Value Float(double d)        { Value v; v.tag = Tag::Float;   v.f = d;             return v; }
    static Value Unbound(std::string n) { Value v; v.tag = Tag::Unbound; v.s = std::move(n);  return v; }
    static Value Compound(std::string fn, std::vector<CellPtr> a)
        { Value v; v.tag = Tag::Compound; v.s = std::move(fn); v.args = std::move(a); return v; }

    bool is_unbound() const { return tag == Tag::Uninit || tag == Tag::Unbound; }

    bool operator==(const Value& o) const {
        if (tag != o.tag) return false;
        switch (tag) {
            case Tag::Atom:    return s == o.s;
            case Tag::Integer: return i == o.i;
            case Tag::Float:   return f == o.f;
            case Tag::Unbound: return s == o.s;
            case Tag::Compound: {
                if (s != o.s) return false;
                if (args.size() != o.args.size()) return false;
                for (std::size_t k = 0; k < args.size(); ++k) {
                    if (!(*args[k] == *o.args[k])) return false;
                }
                return true;
            }
            default: return true;
        }
    }
};

struct Instruction {
    enum class Op {
        GetConstant, GetVariable, GetValue, GetStructure, GetList, GetNil, GetInteger,
        PutConstant, PutVariable, PutValue, PutStructure, PutList,
        UnifyVariable, UnifyValue, UnifyConstant,
        SetVariable, SetValue, SetConstant,
        Call, Execute, Proceed, Fail, Allocate, Deallocate,
        BuiltinCall, CallForeign,
        TryMeElse, RetryMeElse, TrustMe, Jump, CutIte,
        BeginAggregate, EndAggregate,
        SwitchOnConstant, SwitchOnStructure, SwitchOnTerm,
        CatchReturn, NegationReturn, FindallCollect, ConjReturn, DisjAlt,
        IfThenCommit, IfThenElse, AggregateNextGroup, DynamicNextClause,
        SubAtomNext, BodyNext, RetractNext
    };
    // Sentinel pc values for switch-table entries that should not jump.
    static constexpr std::size_t SWITCH_DEFAULT = static_cast<std::size_t>(-1);
    static constexpr std::size_t SWITCH_NONE    = static_cast<std::size_t>(-2);

    Op op = Op::Proceed;
    Value val;
    std::string a;
    std::string b;
    // Reused by BeginAggregate for free-witness register names —
    // a semicolon-delimited list ("Y1;Y2") matching the 4th arg of
    // the 4-token wam-text shape. Empty for non-grouping aggregates.
    std::string c;
    std::int64_t n = 0;
    std::size_t target = 0;
    // Indexing dispatch tables. const_table holds Value→pc for atom/int
    // dispatch; struct_table holds "functor/arity"→pc for compound
    // dispatch; target doubles as the list-pc for SwitchOnTerm.
    std::vector<std::pair<Value, std::size_t>> const_table;
    std::vector<std::pair<std::string, std::size_t>> struct_table;

    static Instruction GetConstant(Value v, std::string ai)
        { Instruction i; i.op = Op::GetConstant; i.val = std::move(v); i.a = std::move(ai); return i; }
    static Instruction GetVariable(std::string xn, std::string ai)
        { Instruction i; i.op = Op::GetVariable; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction GetValue(std::string xn, std::string ai)
        { Instruction i; i.op = Op::GetValue; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction GetStructure(std::string f, std::string ai)
        { Instruction i; i.op = Op::GetStructure; i.a = std::move(f); i.b = std::move(ai); return i; }
    static Instruction GetList(std::string ai)
        { Instruction i; i.op = Op::GetList; i.a = std::move(ai); return i; }
    static Instruction GetNil(std::string ai)
        { Instruction i; i.op = Op::GetNil; i.a = std::move(ai); return i; }
    static Instruction GetInteger(std::int64_t n, std::string ai)
        { Instruction i; i.op = Op::GetInteger; i.n = n; i.a = std::move(ai); return i; }
    static Instruction PutConstant(Value v, std::string ai)
        { Instruction i; i.op = Op::PutConstant; i.val = std::move(v); i.a = std::move(ai); return i; }
    static Instruction PutVariable(std::string xn, std::string ai)
        { Instruction i; i.op = Op::PutVariable; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction PutValue(std::string xn, std::string ai)
        { Instruction i; i.op = Op::PutValue; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction PutStructure(std::string f, std::string ai)
        { Instruction i; i.op = Op::PutStructure; i.a = std::move(f); i.b = std::move(ai); return i; }
    static Instruction PutList(std::string ai)
        { Instruction i; i.op = Op::PutList; i.a = std::move(ai); return i; }
    static Instruction UnifyVariable(std::string xn)
        { Instruction i; i.op = Op::UnifyVariable; i.a = std::move(xn); return i; }
    static Instruction UnifyValue(std::string xn)
        { Instruction i; i.op = Op::UnifyValue; i.a = std::move(xn); return i; }
    static Instruction UnifyConstant(Value v)
        { Instruction i; i.op = Op::UnifyConstant; i.val = std::move(v); return i; }
    static Instruction SetVariable(std::string xn)
        { Instruction i; i.op = Op::SetVariable; i.a = std::move(xn); return i; }
    static Instruction SetValue(std::string xn)
        { Instruction i; i.op = Op::SetValue; i.a = std::move(xn); return i; }
    static Instruction SetConstant(Value v)
        { Instruction i; i.op = Op::SetConstant; i.val = std::move(v); return i; }
    static Instruction Call(std::string p, std::int64_t n)
        { Instruction i; i.op = Op::Call; i.a = std::move(p); i.n = n; return i; }
    static Instruction Execute(std::string p, std::int64_t n = 0)
        { Instruction i; i.op = Op::Execute; i.a = std::move(p); i.n = n; return i; }
    static Instruction Proceed()    { Instruction i; i.op = Op::Proceed;    return i; }
    static Instruction Fail()       { Instruction i; i.op = Op::Fail;       return i; }
    static Instruction Allocate()   { Instruction i; i.op = Op::Allocate;   return i; }
    static Instruction Deallocate() { Instruction i; i.op = Op::Deallocate; return i; }
    static Instruction BuiltinCall(std::string op, std::int64_t n)
        { Instruction i; i.op = Op::BuiltinCall; i.a = std::move(op); i.n = n; return i; }
    static Instruction CallForeign(std::string p, std::int64_t n)
        { Instruction i; i.op = Op::CallForeign; i.a = std::move(p); i.n = n; return i; }
    static Instruction TryMeElse(std::size_t target)
        { Instruction i; i.op = Op::TryMeElse; i.target = target; return i; }
    static Instruction RetryMeElse(std::size_t target)
        { Instruction i; i.op = Op::RetryMeElse; i.target = target; return i; }
    static Instruction TrustMe()    { Instruction i; i.op = Op::TrustMe;    return i; }
    static Instruction Jump(std::size_t target)
        { Instruction i; i.op = Op::Jump; i.target = target; return i; }
    static Instruction CutIte()     { Instruction i; i.op = Op::CutIte;     return i; }
    static Instruction BeginAggregate(std::string kind, std::string vreg, std::string rreg)
        { Instruction i; i.op = Op::BeginAggregate; i.a = std::move(kind);
          i.b = std::move(vreg); i.val = Value::Atom(std::move(rreg)); return i; }
    static Instruction BeginAggregate(std::string kind, std::string vreg, std::string rreg,
                                      std::string wregs)
        { Instruction i; i.op = Op::BeginAggregate; i.a = std::move(kind);
          i.b = std::move(vreg); i.val = Value::Atom(std::move(rreg));
          i.c = std::move(wregs); return i; }
    static Instruction EndAggregate(std::string vreg)
        { Instruction i; i.op = Op::EndAggregate; i.a = std::move(vreg); return i; }
    static Instruction SwitchOnConstant(std::vector<std::pair<Value, std::size_t>> table)
        { Instruction i; i.op = Op::SwitchOnConstant; i.const_table = std::move(table); return i; }
    static Instruction SwitchOnStructure(std::vector<std::pair<std::string, std::size_t>> table)
        { Instruction i; i.op = Op::SwitchOnStructure; i.struct_table = std::move(table); return i; }
    static Instruction SwitchOnTerm(std::vector<std::pair<Value, std::size_t>> consts,
                                    std::vector<std::pair<std::string, std::size_t>> structs,
                                    std::size_t list_pc)
        { Instruction i; i.op = Op::SwitchOnTerm;
          i.const_table = std::move(consts);
          i.struct_table = std::move(structs);
          i.target = list_pc;
          return i; }
    static Instruction CatchReturn()
        { Instruction i; i.op = Op::CatchReturn; return i; }
    static Instruction NegationReturn()
        { Instruction i; i.op = Op::NegationReturn; return i; }
    static Instruction FindallCollect()
        { Instruction i; i.op = Op::FindallCollect; return i; }
    static Instruction ConjReturn()
        { Instruction i; i.op = Op::ConjReturn; return i; }
    static Instruction DisjAlt()
        { Instruction i; i.op = Op::DisjAlt; return i; }
    static Instruction IfThenCommit()
        { Instruction i; i.op = Op::IfThenCommit; return i; }
    static Instruction IfThenElse()
        { Instruction i; i.op = Op::IfThenElse; return i; }
    static Instruction AggregateNextGroup()
        { Instruction i; i.op = Op::AggregateNextGroup; return i; }
    static Instruction DynamicNextClause()
        { Instruction i; i.op = Op::DynamicNextClause; return i; }
    static Instruction SubAtomNext()
        { Instruction i; i.op = Op::SubAtomNext; return i; }
    static Instruction BodyNext()
        { Instruction i; i.op = Op::BodyNext; return i; }
    static Instruction RetractNext()
        { Instruction i; i.op = Op::RetractNext; return i; }
};

struct TrailEntry {
    CellPtr cell;
    Value prev;
};

// Environment frame for permanent variables (Y-regs) and continuation
// preservation. Allocate pushes one; Deallocate pops. Y-reg lookup goes
// through env_stack.back().y_regs rather than the flat regs map, so two
// nested calls that both use Y1 don''t clobber each other.
struct EnvFrame {
    std::size_t saved_cp = 0;
    std::unordered_map<std::string, CellPtr> y_regs;
};

// Read/write mode for the most recent Get-/Put-Structure / Get-/Put-List.
// Only one is "active" at a time — nested compounds restart the mode
// when their own Get-/Put-Structure fires.
struct ModeFrame {
    enum class Kind { None, Read, Write };
    Kind kind = Kind::None;
    CellPtr target;                // Write: the compound cell being filled
    std::vector<CellPtr> args;     // Read: the compound''s arg cells
    std::size_t idx = 0;           // Read: next arg index
    std::size_t expected_arity = 0; // Write: stop when args.size() reaches this
};

// Rule-body sequencing frame for the dynamic-database dispatcher.
// Declared at namespace scope so ChoicePoint can snapshot a stack of
// these for backtrack-into-body correctness — without that snapshot,
// popping the frame forward (when the last goal succeeds) would
// leave subsequent backtrack-into-goal-CPs unable to find their
// frame, so the next-goal dispatch chain would unravel.
struct BodyFrame {
    std::vector<CellPtr> goals;
    std::size_t next_idx = 0;
    std::size_t after_pc = 0;
    std::size_t base_cp_count = 0;
};

struct ChoicePoint {
    std::size_t alt_pc;
    std::size_t saved_cp;
    std::size_t trail_mark;
    std::size_t cut_barrier;
    std::unordered_map<std::string, CellPtr> saved_regs;
    std::vector<ModeFrame> saved_mode_stack;
    std::vector<EnvFrame> saved_env_stack;
    std::vector<BodyFrame> saved_body_frames;
};

// Aggregate scope opened by BeginAggregate. Backtrack() finalises the
// frame when choice_points has been drained back to base_cp_count and
// no normal CP is available.
struct AggregateFrame {
    std::string agg_kind;       // "collect" / "count" / "sum" / "min" / "max" / "set" / "bag"
    std::string value_reg;
    std::string result_reg;
    // Direct CellPtrs for the meta-call findall/3 path — when set,
    // EndAggregate / finalise use these instead of looking up by
    // register name. Needed because the meta-call dispatches the
    // goal as a tail call that overwrites the original A-registers
    // before the value can be collected. Set by dispatch_findall_call;
    // left null by the inlined BeginAggregate path.
    CellPtr     value_cell;
    CellPtr     result_cell;
    std::size_t begin_pc = 0;   // pc of the BeginAggregate instruction
    std::size_t return_pc = 0;  // pc after end_aggregate (0 if never fired)
    bool return_pc_set = false;
    std::size_t base_cp_count = 0;
    std::size_t trail_mark = 0;
    std::size_t saved_cp = 0;
    std::size_t saved_cut_barrier = 0;
    std::unordered_map<std::string, CellPtr> saved_regs;
    std::vector<ModeFrame> saved_mode_stack;
    std::vector<EnvFrame> saved_env_stack;
    std::vector<Value> acc;
    // Witness vars for bagof/setof grouping. Populated by
    // dispatch_aggregate_call from a walk of the goal-term (vars
    // present in Goal but NOT in Template and NOT under ^/2). When
    // empty, the aggregate behaves as findall (single flat group).
    // When non-empty, acc_witnesses[i] is the witness snapshot
    // parallel to acc[i]; finalise groups acc by witness equality
    // and binds the result for the first group, with the witness
    // vars also bound. Backtracking through additional groups is
    // not yet supported (deferred follow-up).
    std::vector<CellPtr> witness_cells;
    std::vector<std::vector<Value>> acc_witnesses;
    // Inlined-path witness registers: stored as names because Y-regs
    // for free witnesses get allocated by put_variable INSIDE the
    // aggregate body, AFTER BeginAggregate fires. EndAggregate resolves
    // them once into witness_cells on the first iteration, then reuses
    // the cached cells for later iterations (the env frame keeps the
    // same Y-reg shared_ptr across body retries).
    std::vector<std::string> witness_regs;
};

// Catcher frame opened by catch/3. Sits on a side stack (not the
// regular choice-point stack) so it doesn''t participate in normal
// backtracking — throw/1 walks this stack explicitly to find a
// catcher whose pattern unifies with the thrown term; backtrack()
// pops frames whose protected goal has exhausted its solutions.
struct CatcherFrame {
    CellPtr catcher_term;       // A2 from catch/3 (pattern to match)
    CellPtr recovery_term;      // A3 from catch/3 (goal to invoke on match)
    std::size_t saved_cp = 0;   // proceed target after recovery returns
    std::size_t trail_mark = 0;
    std::size_t base_cp_count = 0;
    std::size_t base_agg_count = 0;
    std::size_t saved_cut_barrier = 0;
    std::unordered_map<std::string, CellPtr> saved_regs;
    std::vector<ModeFrame> saved_mode_stack;
    std::vector<EnvFrame> saved_env_stack;
};

// Frame for \\+/1 and not/1 (negation as failure). Same shape as
// CatcherFrame minus the pattern/recovery slots — symmetric inverse
// of catch: NegationReturn fires when the protected goal SUCCEEDS
// (negation then fails); backtrack() pops the frame when the goal
// FAILS (negation then succeeds at saved_cp).
struct NegationFrame {
    std::size_t saved_cp = 0;
    std::size_t trail_mark = 0;
    std::size_t base_cp_count = 0;
    std::size_t base_agg_count = 0;
    std::size_t base_catcher_count = 0;
    std::size_t saved_cut_barrier = 0;
    std::unordered_map<std::string, CellPtr> saved_regs;
    std::vector<ModeFrame> saved_mode_stack;
    std::vector<EnvFrame> saved_env_stack;
};

// Frame for a conjunction goal (,(G1, G2)) dispatched as a goal-term.
// When invoke_goal_as_call sees a ,/2 goal it pushes a ConjFrame
// remembering G2 and the original after_pc, then dispatches G1 with
// cp = conj_return_pc. On G1 success, the ConjReturn step arm pops
// the frame and dispatches G2 with the original after_pc. On G1
// failure, backtrack() pops the frame at CP-drain time and
// propagates failure to the caller.
struct ConjFrame {
    CellPtr     second_goal;
    std::size_t after_pc = 0;
    std::size_t base_cp_count = 0;
};

// Frame for a disjunction goal (;(G1, G2)) dispatched as a goal-term.
// invoke_goal_as_call pushes a DisjFrame AND a regular ChoicePoint
// whose alt_pc = disj_alt_pc, then dispatches G1. If G1 succeeds the
// CP stays around for outer backtracking to retry G2 later; if G1
// fails (or backtrack reaches the CP), DisjAlt pops both the CP and
// the DisjFrame and dispatches G2 with the original after_pc.
struct DisjFrame {
    CellPtr     second_goal;
    std::size_t after_pc = 0;
};

// Frame for an if-then-else goal (;(->( Cond, Then), Else)) or a
// bare if-then (->(Cond, Then) with no enclosing ;) dispatched as a
// goal-term. invoke_goal_as_call pushes an IfThenFrame and a
// ChoicePoint with alt_pc = if_then_else_pc, then dispatches Cond
// with cp = if_then_commit_pc.
//
//   - Cond succeeds  → IfThenCommit fires: cut CPs back to
//                       base_cp_count (so Cond can''t backtrack-
//                       retry), pop the frame, dispatch Then with
//                       the original after_pc.
//   - Cond fails     → backtrack drains to our CP, lands on
//                       if_then_else_pc: pop the frame, dispatch
//                       Else with after_pc. For bare if-then (no
//                       Else clause), else_goal is null and the
//                       op propagates failure.
struct IfThenFrame {
    CellPtr     then_goal;
    CellPtr     else_goal;            // null for bare (Cond -> Then)
    std::size_t after_pc = 0;
    std::size_t base_cp_count = 0;   // CP-stack depth BEFORE we
                                     // pushed the if-then-else CP
};

// Iterator pushed by aggregate-finalise for bagof/setof with
// witnesses. Holds the remaining-groups list along with the cell
// pointers needed to bind each group at backtrack time. The first
// group is bound at finalise; if more remain, a ChoicePoint is
// pushed whose alt_pc = aggregate_next_group_pc. The AggregateNextGroup
// step arm reads the top iterator, pops one group, and binds it —
// pushing another CP if any groups still remain.
struct AggregateGroupIterator {
    std::string agg_kind;                 // "bagof" or "setof"
    std::vector<std::pair<std::vector<Value>, std::vector<Value>>>
                remaining_groups;         // (witness_values, templates)
    std::vector<CellPtr> witness_cells;
    CellPtr     result_cell;
    std::size_t return_pc = 0;
};

struct WamState {
    std::unordered_map<std::string, CellPtr> regs;
    std::unordered_map<std::string, std::size_t> labels;
    std::vector<Instruction> instrs;
    std::vector<TrailEntry> trail;
    std::vector<ChoicePoint> choice_points;
    std::vector<AggregateFrame> aggregate_frames;
    std::vector<CatcherFrame> catcher_frames;
    // \\+/1 + not/1 — symmetric inverse of catcher_frames.
    std::vector<NegationFrame> negation_frames;
    // Conjunction-goal-term dispatch (,(G1, G2) passed as a meta-call
    // argument, e.g. inside catch(_, _, _) or findall/3).
    std::vector<ConjFrame> conj_frames;
    // Disjunction-goal-term dispatch (;(G1, G2)). Paired with a
    // regular ChoicePoint — see DisjFrame struct doc.
    std::vector<DisjFrame> disj_frames;
    // If-then-else-goal-term dispatch (;(->(Cond, Then), Else)).
    // Paired with a regular ChoicePoint — see IfThenFrame doc.
    std::vector<IfThenFrame> if_then_frames;
    // Iterators for bagof/setof group backtracking — see
    // AggregateGroupIterator struct doc.
    std::vector<AggregateGroupIterator> aggregate_group_iters;
    // pc of the auto-injected single CatchReturn instruction. catch/3
    // sets cp to this value before dispatching to the protected goal;
    // when the goal proceeds, control lands here and the catcher frame
    // is popped.
    std::size_t catch_return_pc = 0;
    // pc of the auto-injected single NegationReturn instruction. \\+/1
    // sets cp to this value before dispatching the protected goal;
    // when the goal succeeds via proceed, control lands here and the
    // negation frame is popped + the negation FAILS.
    std::size_t negation_return_pc = 0;
    // pc of the auto-injected single FindallCollect instruction.
    // Meta-call findall/3 sets cp to this value before dispatching the
    // goal; on goal success, lands here, collects the template''s
    // current value into the top aggregate frame''s acc, then forces
    // backtrack to find the next solution. Standard aggregate-frame
    // finalise (in backtrack()) wraps things up when CPs drain.
    std::size_t findall_collect_pc = 0;
    // pc of the auto-injected single ConjReturn instruction. ,/2
    // dispatched as a goal-term lands here when G1 succeeds; pops
    // the top ConjFrame and dispatches its G2 with the original
    // after_pc.
    std::size_t conj_return_pc = 0;
    // pc of the auto-injected single DisjAlt instruction. ;/2
    // dispatched as a goal-term arrives here via a ChoicePoint''s
    // alt_pc when G1 fails; pops the matching DisjFrame and CP,
    // then dispatches G2 with the original after_pc.
    std::size_t disj_alt_pc = 0;
    // pcs for the two if-then-else synthetic ops. Cond reaches
    // if_then_commit_pc on success (cut + dispatch Then); the
    // paired CP''s alt_pc is if_then_else_pc, reached on Cond
    // failure (dispatch Else).
    std::size_t if_then_commit_pc = 0;
    std::size_t if_then_else_pc = 0;
    // pc of the auto-injected AggregateNextGroup op. ChoicePoints
    // pushed by aggregate-finalise (when bagof/setof has more than
    // one witness group) point alt_pc here; on backtrack the op
    // pops the next group from the top AggregateGroupIterator and
    // binds it.
    std::size_t aggregate_next_group_pc = 0;
    // Per-predicate dynamic clause store. Maps "name/arity" → list
    // of fact terms (Compound or 0-arity Atom cells). Populated by
    // assertz/asserta builtins; mutated by retract/retractall.
    // dispatch_dynamic_call iterates these, pushing a CP whose
    // alt_pc = dynamic_next_clause_pc when more clauses remain.
    // Rules (Head :- Body) are NOT supported in this PR — only facts.
    std::unordered_map<std::string, std::vector<CellPtr>> dynamic_db;
    // Iteration state for dispatch_dynamic_call. Mirrors the
    // AggregateGroupIterator pattern: one entry per active dynamic
    // call, popped when clauses are exhausted or the call commits
    // past its last clause.
    struct DynamicIterator {
        std::string key;
        std::size_t next_idx = 0;
        std::vector<CellPtr> call_args;
        std::size_t after_pc = 0;
    };
    std::vector<DynamicIterator> dynamic_iters;
    std::size_t dynamic_next_clause_pc = 0;
    // Mutable globals — nb_setval/2, nb_getval/2, b_setval/2, b_getval/2.
    // Each key maps to a CellPtr holding the current value. nb_setval
    // replaces the pointer (non-backtrackable: prior bindings to the
    // old cell stay trail-tracked but no longer reachable through
    // the global). b_setval mutates the existing cell via bind_cell,
    // so a trail entry restores the previous value on backtrack.
    // Both setvals deep-copy the input to give the stored term
    // independent vars.
    std::unordered_map<std::string, CellPtr> nb_globals;
    // Iteration state for retract/1''s nondeterministic behaviour.
    // Each successful retract removes the matched clause; on
    // backtrack, retract_try_next continues from where it left off
    // (in the now-modified clause list) looking for the next match.
    // Per ISO, retract is destructive — backtracking doesn''t undo
    // the removal.
    struct RetractIterator {
        std::string key;
        CellPtr pattern;          // the call''s A1 (full match pattern)
        std::size_t next_idx = 0; // next clause index to test
        std::size_t after_pc = 0;
    };
    std::vector<RetractIterator> retract_iters;
    std::size_t retract_next_pc = 0;
    // Rule-body sequencing — see BodyFrame doc at namespace scope.
    // Each rule body is flattened into a sequential goal list;
    // body_next dispatches them one at a time with cp=body_next_pc.
    // Frames are popped FORWARD when goals are exhausted; backtrack
    // restores body_frames from the CP that fires (since CPs
    // snapshot body_frames at push time).
    std::vector<BodyFrame> body_frames;
    std::size_t body_next_pc = 0;
    // Iteration state for sub_atom/5. Each entry holds the source
    // atom, the bound cells (Before/Length/After/Sub), the call''s
    // continuation, and the list of (before, length) pairs still
    // to try. The pairs are pre-filtered at dispatch time to respect
    // whichever args were bound, so iteration is just "pop next,
    // unify, succeed".
    struct SubAtomIterator {
        std::string atom_str;
        CellPtr before_cell;
        CellPtr length_cell;
        CellPtr after_cell;
        CellPtr sub_cell;
        std::size_t after_pc = 0;
        // remaining_pairs[k] = (before_k, length_k). We pop from the
        // front so order matches typical "left-to-right first match"
        // expectations (e.g. sub_atom(abcabc, B, _, _, b) yields
        // B=1 first, then B=4).
        std::vector<std::pair<std::size_t, std::size_t>> remaining_pairs;
    };
    std::vector<SubAtomIterator> sub_atom_iters;
    std::size_t sub_atom_next_pc = 0;
    // Mode stack: Get-/Put-Structure / Get-/Put-List PUSH; Unify*/Set*
    // operate on top(); each frame auto-pops once its arity is filled.
    std::vector<ModeFrame> mode_stack;
    // Environment stack: Allocate pushes (saving cp + giving fresh Y-regs),
    // Deallocate pops (restoring cp). Y-reg lookup is scoped to top().
    std::vector<EnvFrame> env_stack;
    std::size_t pc = 0;
    std::size_t cp = 0;
    std::size_t cut_barrier = 0;
    std::uint64_t var_counter = 0;
    bool halt = false;

    // Cell-aware accessors. get_reg/put_reg keep their Value-shaped API
    // so existing lowered code keeps compiling; get_cell exposes the cell
    // for instructions that need sharing semantics.
    Value   get_reg(const std::string& name) const;
    void    put_reg(const std::string& name, Value v);
    CellPtr get_cell(const std::string& name);
    void    set_cell(const std::string& name, CellPtr c);

    // Deref through Unbound chains until a concrete value (or a terminal
    // unbound cell) is reached. Returns by value (snapshot).
    Value   deref(const Value& v) const;

    // bind_cell mutates *cell, recording the previous content on the trail.
    void    bind_cell(CellPtr cell, Value v);
    void    trail_binding(const std::string& name); // legacy reg-name trail

    // Unification: takes Cell pointers so binding works correctly.
    bool    unify_cells(CellPtr a, CellPtr b);
    // Convenience for the lowered emitter (no real binding — equality
    // / unbound-as-success check only).
    bool    unify(const Value& a, const Value& b);

    bool    step(const Instruction& instr);
    bool    run();
    bool    backtrack();
    bool    builtin(const std::string& op, std::int64_t arity);
    bool    query(const std::string& pred_key, const std::vector<Value>& args);

    // ---- catch/3 + throw/1 helpers ----------------------------------
    // Treats arg-register A1..AN setup + label dispatch like a Call.
    // after_pc becomes the new cp (where the goal proceeds to).
    // Returns false if the goal''s functor has no registered label.
    bool    invoke_goal_as_call(CellPtr goal_cell, std::size_t after_pc);
    // call/N meta-call dispatch. Total arity comes from the op
    // suffix (call/3 → 3). For total_arity == 1 the goal in A1 is
    // invoked as-is; otherwise A2..AN are appended to the goal''s
    // existing args and the combined goal is dispatched. after_pc
    // is supplied by the caller — pc+1 for non-tail (Call), cp for
    // tail (Execute). Returns invoke_goal_as_call''s result.
    bool    dispatch_call_meta(const std::string& op,
                               std::int64_t total_arity,
                               std::size_t after_pc);
    // findall(Template, Goal, List) meta-call. Pushes an
    // AggregateFrame with value_cell/result_cell set to A1/A3''s
    // current cells, then dispatches A2 as a goal with cp =
    // findall_collect_pc. Goal-success triggers FindallCollect
    // (collect template, force backtrack); goal-exhaustion triggers
    // backtrack()''s aggregate-finalise (build list, bind to result).
    bool    dispatch_findall_call(std::size_t after_pc);
    // bagof/setof meta-call. Same shape as dispatch_findall_call
    // but the AggregateFrame''s kind is "bagof" or "setof", so
    // finalize_aggregate fails on empty acc (bagof) or
    // sorts + dedup''s + fails on empty (setof).
    bool    dispatch_aggregate_call(const std::string& kind,
                                    std::size_t after_pc);
    // Walk a goal term collecting unbound-variable cells, ignoring
    // any cells reachable through the LHS of a ^/2 binder (those
    // are existentially quantified). Used by dispatch_aggregate_call
    // to determine grouping witnesses for bagof/setof — see the
    // AggregateFrame::witness_cells doc.
    void    collect_goal_witnesses(CellPtr goal,
                                   const std::set<Cell*>& exclude,
                                   std::vector<CellPtr>& out,
                                   std::set<Cell*>& seen) const;
    // Pop the next group from the top AggregateGroupIterator, bind
    // its witness values + result list, and (if more groups remain)
    // push a ChoicePoint for the next iteration. Sets pc =
    // iterator.return_pc on success. Returns false if no iterator,
    // no remaining groups, or a binding mismatch.
    bool    aggregate_bind_next_group();
    // Dispatch a Call/Execute to a dynamic predicate. Snapshots
    // A-registers as call_args, pushes a DynamicIterator, then
    // delegates to dynamic_try_next which unifies the first
    // clause (and pushes a CP for the rest if any).
    bool    dispatch_dynamic_call(const std::string& key,
                                  std::size_t after_pc);
    // Try the next clause in the top DynamicIterator. On match,
    // unify the call_args with the clause''s args and proceed to
    // after_pc; on no-match, undo trail and recurse (CP-style).
    // On exhausted iterator, pop and return false.
    bool    dynamic_try_next();
    // Nondet retract/1 — finds the next clause in dynamic_db[key]
    // (from iter.next_idx onward) that unifies with iter.pattern,
    // removes it, leaves the unification bindings in place, pushes
    // a CP if more candidates remain, and proceeds to after_pc.
    bool    dispatch_retract(std::size_t after_pc);
    bool    retract_try_next();
    // body_next — dispatch the top BodyFrame''s next goal, or pop
    // and proceed to outer after_pc when goals are exhausted.
    bool    body_next();
    // sub_atom/5 — enumerate (Before, Length) tuples matching the
    // bound constraints. Pre-filters the candidate list at entry,
    // then iterates via the SubAtomIterator + sub_atom_next_pc CP
    // pattern. Returns true on first successful match (with a CP
    // pushed for subsequent matches when any remain).
    bool    dispatch_sub_atom(std::size_t after_pc);
    bool    sub_atom_try_next();
    bool    execute_catch();
    bool    execute_throw();

    // ---- ISO error term helpers --------------------------------------
    // Construct error(ErrTerm, _) with a fresh unbound Context slot,
    // bind it into A1, and dispatch via execute_throw(). Callers
    // typically build ErrTerm with one of the make_*_error helpers
    // below. Returns whatever execute_throw() returns (always false
    // unless a matching catcher exists).
    bool    throw_iso_error(Value err_term);

    // Concrete error-term constructors. Each returns a Value that
    // sits inside the outer error/2 wrapper built by
    // throw_iso_error. Shapes match WAM_CPP_ISO_ERRORS_SPECIFICATION
    // §6 (the table of thrown terms per builtin per trigger).
    Value   make_type_error(const std::string& expected, Value culprit);
    Value   make_instantiation_error();
    Value   make_domain_error(const std::string& domain, Value culprit);
    Value   make_evaluation_error(const std::string& kind);

    // ISO-arith helpers. term_contains_unbound walks an arith term
    // tree to detect an instantiation_error condition; arith_culprit
    // shapes a non-evaluable into the Name/Arity compound ISO mandates
    // for the type_error culprit slot.
    bool    term_contains_unbound(CellPtr c) const;
    Value   arith_culprit(const Value& v) const;
    // Walk an arith term tree looking for a "/" or "//" or "mod"
    // node whose RHS dereferences to a zero integer or zero float.
    // Used by ISO arith builtins to throw
    // evaluation_error(zero_divisor) before eval_arith silently
    // succeeds with IEEE 754 inf / NaN (lax behavior) — see
    // SPECIFICATION §6.1.
    bool    term_has_zero_divide(CellPtr c) const;
    // Deep-copy a value tree, allocating fresh cells for any Unbound
    // leaves (multiple references to the same source name share the
    // same fresh cell). Used by throw/1 to snapshot the thrown term
    // before unwinding state.
    CellPtr deep_copy_term(CellPtr src);
    // Same as above but with a pre-seeded var-name → cell rename map.
    // Pre-seeded entries are NOT overwritten; deep_copy_term reuses
    // them whenever it encounters an unbound var with that name. Used
    // by dynamic_try_next to make head args share the caller''s cells
    // (essential so unbound-unbound unification works correctly —
    // our cell model doesn''t alias-via-ref).
    CellPtr deep_copy_term_seeded(
        CellPtr src,
        std::unordered_map<std::string, CellPtr> seed);

    // ---- Builtin helpers ---------------------------------------------
    // Arithmetic: returns Integer or Float; sets ok=false on failure
    // (unbound argument, unknown operator, division by zero, ...).
    Value   eval_arith(CellPtr c, bool& ok) const;
    bool    arith_compare(const std::string& op, const Value& a, const Value& b) const;

    // Term rendering for write/1 — recursive, prints Prolog-like syntax.
    std::string render(const Value& v) const;
};

struct Program {
    // Setup callback: each generated_program.cpp defines a unique
    // wam_cpp_setup() function that populates instrs + labels at startup.
    using Setup = void (*)(WamState&);
    static Setup& setup_hook();
    static void register_setup(Setup s);
    static void apply_setup(WamState& vm);
};

} // namespace wam_cpp

using wam_cpp::Value;
using wam_cpp::Instruction;
using wam_cpp::WamState;
using wam_cpp::Program;

#endif // UNIFYWEAVER_WAM_CPP_RUNTIME_H
').

compile_wam_runtime_to_cpp(_Options,
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
//
// WAM runtime implementation. Handles atom/integer-level unification,
// compound terms and lists via shared_ptr cells, choice points with full
// trail-based undo, and the standard control-flow opcode set.

#include "wam_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace wam_cpp {

// Forward declarations for free functions defined later in this TU.
static int standard_order_cmp(const Value& a, const Value& b);

// ----------------------------------------------------------------------
// Cell helpers
// ----------------------------------------------------------------------

static CellPtr make_cell(Value v = Value{}) {
    return std::make_shared<Cell>(std::move(v));
}

// Y-regs are scoped to the top env frame; A/X-regs use the flat map.
static bool is_y_reg(const std::string& name) {
    return !name.empty() && name[0] == \'Y\';
}

CellPtr WamState::get_cell(const std::string& name) {
    if (is_y_reg(name)) {
        if (env_stack.empty()) env_stack.emplace_back();
        auto& y = env_stack.back().y_regs;
        auto it = y.find(name);
        if (it != y.end()) return it->second;
        CellPtr c = make_cell(Value::Unbound("_U_" + name));
        y[name] = c;
        return c;
    }
    auto it = regs.find(name);
    if (it != regs.end()) return it->second;
    CellPtr c = make_cell(Value::Unbound("_U_" + name));
    regs[name] = c;
    return c;
}

void WamState::set_cell(const std::string& name, CellPtr c) {
    if (is_y_reg(name)) {
        if (env_stack.empty()) env_stack.emplace_back();
        env_stack.back().y_regs[name] = std::move(c);
        return;
    }
    regs[name] = std::move(c);
}

Value WamState::get_reg(const std::string& name) const {
    if (is_y_reg(name)) {
        if (env_stack.empty()) return Value{};
        auto& y = env_stack.back().y_regs;
        auto it = y.find(name);
        if (it == y.end()) return Value{};
        return *it->second;
    }
    auto it = regs.find(name);
    if (it == regs.end()) return Value{};
    return *it->second;
}

void WamState::put_reg(const std::string& name, Value v) {
    if (is_y_reg(name)) {
        if (env_stack.empty()) env_stack.emplace_back();
        auto& y = env_stack.back().y_regs;
        auto it = y.find(name);
        if (it == y.end()) y[name] = make_cell(std::move(v));
        else               *it->second = std::move(v);
        return;
    }
    auto it = regs.find(name);
    if (it == regs.end()) regs[name] = make_cell(std::move(v));
    else                  *it->second = std::move(v);
}

void WamState::trail_binding(const std::string& name) {
    auto it = regs.find(name);
    if (it == regs.end()) return;
    TrailEntry e;
    e.cell = it->second;
    e.prev = *it->second;
    trail.push_back(std::move(e));
}

void WamState::bind_cell(CellPtr cell, Value v) {
    TrailEntry e;
    e.cell = cell;
    e.prev = *cell;
    trail.push_back(std::move(e));
    *cell = std::move(v);
}

Value WamState::deref(const Value& v) const {
    // Followers are explicitly bound Unbound cells. With our shared_ptr
    // model the cell content is mutated in place, so deref through Value
    // alone is a no-op — we just hand back v.
    return v;
}

// Cell-aware unification. Binds variables via bind_cell so the trail
// records every change.
bool WamState::unify_cells(CellPtr a, CellPtr b) {
    if (a.get() == b.get()) return true;
    Value& va = *a;
    Value& vb = *b;
    if (va.is_unbound()) { bind_cell(a, vb); return true; }
    if (vb.is_unbound()) { bind_cell(b, va); return true; }
    if (va.tag != vb.tag) return false;
    switch (va.tag) {
        case Value::Tag::Atom:    return va.s == vb.s;
        case Value::Tag::Integer: return va.i == vb.i;
        case Value::Tag::Float:   return va.f == vb.f;
        case Value::Tag::Compound: {
            if (va.s != vb.s) return false;
            if (va.args.size() != vb.args.size()) return false;
            for (std::size_t k = 0; k < va.args.size(); ++k) {
                if (!unify_cells(va.args[k], vb.args[k])) return false;
            }
            return true;
        }
        default: return true;
    }
}

bool WamState::unify(const Value& a, const Value& b) {
    if (a.is_unbound() || b.is_unbound()) return true;
    return a == b;
}

// ----------------------------------------------------------------------
// Arithmetic eval
// ----------------------------------------------------------------------

Value WamState::eval_arith(CellPtr c, bool& ok) const {
    if (!c) { ok = false; return Value{}; }
    const Value& v = *c;
    switch (v.tag) {
        case Value::Tag::Integer:
        case Value::Tag::Float:
            return v;
        case Value::Tag::Atom:
            // Numeric atoms ("5", "-3.14") are tolerated for robustness.
            try {
                std::size_t pos;
                long long n = std::stoll(v.s, &pos);
                if (pos == v.s.size()) return Value::Integer(n);
            } catch (...) {}
            try {
                std::size_t pos;
                double d = std::stod(v.s, &pos);
                if (pos == v.s.size()) return Value::Float(d);
            } catch (...) {}
            ok = false; return Value{};
        case Value::Tag::Compound: {
            // Unary minus
            if (v.s == "-/1" && v.args.size() == 1) {
                Value a = eval_arith(v.args[0], ok);
                if (!ok) return Value{};
                if (a.tag == Value::Tag::Integer) return Value::Integer(-a.i);
                return Value::Float(-a.f);
            }
            if (v.args.size() != 2) { ok = false; return Value{}; }
            Value a = eval_arith(v.args[0], ok);
            if (!ok) return Value{};
            Value b = eval_arith(v.args[1], ok);
            if (!ok) return Value{};
            bool either_float = (a.tag == Value::Tag::Float || b.tag == Value::Tag::Float);
            auto as_d = [](const Value& w){ return w.tag == Value::Tag::Float ? w.f : (double)w.i; };
            if (v.s == "+/2") {
                if (either_float) return Value::Float(as_d(a) + as_d(b));
                return Value::Integer(a.i + b.i);
            }
            if (v.s == "-/2") {
                if (either_float) return Value::Float(as_d(a) - as_d(b));
                return Value::Integer(a.i - b.i);
            }
            if (v.s == "*/2") {
                if (either_float) return Value::Float(as_d(a) * as_d(b));
                return Value::Integer(a.i * b.i);
            }
            if (v.s == "//2") {
                // Prolog ''/'' is float division. Lax / default mode
                // follows IEEE 754 for float operands (inf / -inf /
                // NaN on divide-by-zero) per
                // WAM_CPP_ISO_ERRORS_SPECIFICATION §6.1; integer
                // divide-by-zero remains uniform-fail. ISO mode
                // detects divide-by-zero before calling eval_arith
                // (via term_has_zero_divide) and throws
                // evaluation_error(zero_divisor) regardless of
                // operand type.
                if (as_d(b) == 0.0) {
                    if (either_float) {
                        double num = as_d(a);
                        if (num == 0.0) return Value::Float(std::nan(""));
                        return Value::Float(num > 0
                            ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity());
                    }
                    ok = false; return Value{};
                }
                if (either_float) return Value::Float(as_d(a) / as_d(b));
                if (a.i % b.i == 0) return Value::Integer(a.i / b.i);
                return Value::Float((double)a.i / (double)b.i);
            }
            if (v.s == "///2") {
                if (b.i == 0) { ok = false; return Value{}; }
                return Value::Integer(a.i / b.i);
            }
            if (v.s == "mod/2") {
                if (b.i == 0) { ok = false; return Value{}; }
                return Value::Integer(a.i % b.i);
            }
            ok = false; return Value{};
        }
        default:
            ok = false; return Value{};
    }
}

bool WamState::arith_compare(const std::string& op, const Value& a, const Value& b) const {
    auto as_d = [](const Value& v){ return v.tag == Value::Tag::Float ? v.f : (double)v.i; };
    double x = as_d(a), y = as_d(b);
    if (op == ">/2")  return x >  y;
    if (op == "</2")  return x <  y;
    if (op == ">=/2") return x >= y;
    if (op == "=</2") return x <= y;
    if (op == "=:=/2") return x == y;
    if (op == "=\\\\=/2") return x != y;
    return false;
}

// ----------------------------------------------------------------------
// Term rendering for write/1
// ----------------------------------------------------------------------

std::string WamState::render(const Value& v) const {
    switch (v.tag) {
        case Value::Tag::Atom:    return v.s;
        case Value::Tag::Integer: return std::to_string(v.i);
        case Value::Tag::Float: {
            std::ostringstream os; os << v.f; return os.str();
        }
        case Value::Tag::Unbound: return v.s.empty() ? std::string("_") : v.s;
        case Value::Tag::Compound: {
            // Pretty-print lists as [a, b, c] when the spine is well-formed.
            if (v.s == "[|]/2" && v.args.size() == 2) {
                std::string out = "[";
                const Value* cur = &v;
                bool first = true;
                while (cur && cur->tag == Value::Tag::Compound
                       && cur->s == "[|]/2" && cur->args.size() == 2) {
                    if (!first) out += ", ";
                    first = false;
                    out += render(*cur->args[0]);
                    cur = cur->args[1].get();
                }
                if (cur && !(cur->tag == Value::Tag::Atom && cur->s == "[]")) {
                    out += " | "; out += render(*cur);
                }
                out += "]";
                return out;
            }
            // functor(arg1, arg2, ...). Strip "/N" suffix from functor.
            std::string name = v.s;
            auto slash = name.find_last_of(\'/\');
            if (slash != std::string::npos) name.resize(slash);
            std::string out = name + "(";
            for (std::size_t k = 0; k < v.args.size(); ++k) {
                if (k) out += ", ";
                out += render(*v.args[k]);
            }
            out += ")";
            return out;
        }
        default: return "_";
    }
}

// ----------------------------------------------------------------------
// Builtin dispatch
// ----------------------------------------------------------------------

bool WamState::builtin(const std::string& op, std::int64_t /*arity*/) {
    // ---- Control ----------------------------------------------------
    if (op == "true/0") { pc += 1; return true; }
    if (op == "fail/0") { return false; }
    if (op == "!/0")    {
        if (choice_points.size() > cut_barrier) choice_points.resize(cut_barrier);
        pc += 1; return true;
    }

    // ---- =/2 (unify) -----------------------------------------------
    if (op == "=/2") {
        if (!unify_cells(get_cell("A1"), get_cell("A2"))) return false;
        pc += 1; return true;
    }

    // ---- copy_term/2 -----------------------------------------------
    // copy_term(T1, T2) makes T2 a structural copy of T1 where each
    // unbound variable in T1 maps to a single fresh cell in T2 (so
    // copy_term(foo(X, X), C) yields C = foo(Y, Y) with Y fresh).
    if (op == "copy_term/2") {
        std::unordered_map<std::string, CellPtr> rename;
        std::function<CellPtr(CellPtr)> rec = [&](CellPtr src) -> CellPtr {
            Value& v = *src;
            if (v.tag == Value::Tag::Unbound || v.tag == Value::Tag::Uninit) {
                const std::string& name = v.s;
                auto it = rename.find(name);
                if (it != rename.end()) return it->second;
                CellPtr fresh = std::make_shared<Cell>(
                    Value::Unbound("_C" + std::to_string(var_counter++)));
                rename[name] = fresh;
                return fresh;
            }
            if (v.tag == Value::Tag::Compound) {
                std::vector<CellPtr> args;
                for (auto& c : v.args) args.push_back(rec(c));
                return std::make_shared<Cell>(
                    Value::Compound(v.s, std::move(args)));
            }
            // Atom / Integer / Float: cheap shallow copy.
            return std::make_shared<Cell>(v);
        };
        CellPtr copy = rec(get_cell("A1"));
        if (!unify_cells(get_cell("A2"), copy)) return false;
        pc += 1; return true;
    }

    // ---- succ/2 ----------------------------------------------------
    // ---- \\+/1, not/1 (negation as failure) -------------------------
    // Snapshot VM state into a NegationFrame, push it, dispatch A1
    // as a goal with cp = negation_return_pc. Two outcomes:
    //   - Goal succeeds via proceed → lands on NegationReturn op →
    //     pops frame + restores state + returns false (negation
    //     fails).
    //   - Goal fails → backtrack() drains CPs to the frame''s base →
    //     pops frame + restores state + succeeds at saved_cp
    //     (negation succeeds).
    // Symmetric inverse of catch/3 (without the pattern-match step).
    if (op == "\\\\+/1" || op == "not/1") {
        NegationFrame f;
        f.saved_cp = pc + 1;
        f.trail_mark = trail.size();
        f.base_cp_count = choice_points.size();
        f.base_agg_count = aggregate_frames.size();
        f.base_catcher_count = catcher_frames.size();
        f.saved_cut_barrier = cut_barrier;
        f.saved_regs = regs;
        f.saved_mode_stack = mode_stack;
        f.saved_env_stack = env_stack;
        std::size_t my_depth = negation_frames.size();
        // Snapshot a backup of saved_cp before moving f into the
        // stack, so the no-dispatch path can still proceed.
        std::size_t restore_cp = f.saved_cp;
        negation_frames.push_back(std::move(f));
        CellPtr goal = get_cell("A1");
        if (!invoke_goal_as_call(goal, negation_return_pc)) {
            // Goal failed to dispatch (unknown functor, plain
            // failure atom, etc.) → treat as G failed → negation
            // succeeds. Pop our frame if still ours and proceed to
            // saved_cp manually.
            if (negation_frames.size() > my_depth) {
                NegationFrame nf = std::move(negation_frames.back());
                negation_frames.pop_back();
                while (trail.size() > nf.trail_mark) {
                    TrailEntry t = std::move(trail.back());
                    trail.pop_back();
                    *t.cell = std::move(t.prev);
                }
                while (aggregate_frames.size() > nf.base_agg_count)
                    aggregate_frames.pop_back();
                while (catcher_frames.size() > nf.base_catcher_count)
                    catcher_frames.pop_back();
                regs        = std::move(nf.saved_regs);
                mode_stack  = std::move(nf.saved_mode_stack);
                env_stack   = std::move(nf.saved_env_stack);
                cut_barrier = nf.saved_cut_barrier;
            }
            if (restore_cp == 0) { halt = true; return true; }
            pc = restore_cp; cp = 0;
            return true;
        }
        return true;
    }

    // succ(X, Y): Y is X + 1. Bidirectional — if X is a non-negative
    // integer, derive Y; if Y is a positive integer, derive X. Fails
    // if X < 0 or Y <= 0 or both args unbound (ISO would throw
    // instantiation_error in the both-unbound case; v1 just fails).
    // succ/2 and succ_lax/2 share the lax body. succ_lax exists as
    // a distinct dispatch key for the three-forms guarantee.
    if (op == "succ/2" || op == "succ_lax/2") {
        Value a = deref(*get_cell("A1"));
        Value b = deref(*get_cell("A2"));
        if (a.tag == Value::Tag::Integer) {
            if (a.i < 0) return false;
            long long next = a.i + 1;
            if (b.tag == Value::Tag::Integer) {
                if (b.i != next) return false;
                pc += 1; return true;
            }
            if (!unify_cells(get_cell("A2"),
                             std::make_shared<Cell>(Value::Integer(next))))
                return false;
            pc += 1; return true;
        }
        if (b.tag == Value::Tag::Integer) {
            if (b.i <= 0) return false;
            long long prev = b.i - 1;
            if (!unify_cells(get_cell("A1"),
                             std::make_shared<Cell>(Value::Integer(prev))))
                return false;
            pc += 1; return true;
        }
        return false;
    }
    // ---- succ_iso/2 -------------------------------------------------
    // ISO semantics:
    //   - both args unbound → instantiation_error.
    //   - either arg non-integer (and bound) → type_error(integer, X).
    //   - X negative → type_error(not_less_than_zero, X).
    //   - Y zero or negative → domain_error(not_less_than_zero, Y).
    // (SPECIFICATION §6.)
    if (op == "succ_iso/2") {
        Value a = deref(*get_cell("A1"));
        Value b = deref(*get_cell("A2"));
        bool a_unbound = (a.tag == Value::Tag::Unbound
                          || a.tag == Value::Tag::Uninit);
        bool b_unbound = (b.tag == Value::Tag::Unbound
                          || b.tag == Value::Tag::Uninit);
        if (a_unbound && b_unbound) {
            return throw_iso_error(make_instantiation_error());
        }
        if (!a_unbound && a.tag != Value::Tag::Integer) {
            return throw_iso_error(make_type_error("integer", a));
        }
        if (!b_unbound && b.tag != Value::Tag::Integer) {
            return throw_iso_error(make_type_error("integer", b));
        }
        if (a.tag == Value::Tag::Integer) {
            if (a.i < 0) {
                return throw_iso_error(
                    make_type_error("not_less_than_zero", a));
            }
            long long next = a.i + 1;
            if (b.tag == Value::Tag::Integer) {
                if (b.i != next) return false;
                pc += 1; return true;
            }
            if (!unify_cells(get_cell("A2"),
                             std::make_shared<Cell>(Value::Integer(next))))
                return false;
            pc += 1; return true;
        }
        // a unbound, b is integer (covered by guards above).
        if (b.i <= 0) {
            return throw_iso_error(
                make_domain_error("not_less_than_zero", b));
        }
        long long prev = b.i - 1;
        if (!unify_cells(get_cell("A1"),
                         std::make_shared<Cell>(Value::Integer(prev))))
            return false;
        pc += 1; return true;
    }
    // ---- \\=/2 (cannot unify) --------------------------------------
    if (op == "\\\\=/2") {
        // Try to unify; if it succeeds we have to undo and fail.
        std::size_t mark = trail.size();
        bool ok = unify_cells(get_cell("A1"), get_cell("A2"));
        // Roll back any bindings we just made.
        while (trail.size() > mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        if (ok) return false;
        pc += 1; return true;
    }

    // ---- is/2, is_lax/2 ---------------------------------------------
    // Lax arithmetic evaluation. is_lax/2 shares the body so user
    // code can write is_lax(X, Expr) directly to opt out of an
    // enclosing ISO-mode predicate''s rewrite (the three-forms
    // guarantee from WAM_CPP_ISO_ERRORS_PHILOSOPHY §3.3).
    if (op == "is/2" || op == "is_lax/2") {
        bool ok = true;
        Value rhs = eval_arith(get_cell("A2"), ok);
        if (!ok) return false;
        CellPtr lhs = get_cell("A1");
        if (lhs->is_unbound()) { bind_cell(lhs, std::move(rhs)); pc += 1; return true; }
        if (!(*lhs == rhs)) return false;
        pc += 1; return true;
    }
    // ---- is_iso/2 ---------------------------------------------------
    // ISO-strict arithmetic. Throws instantiation_error for unbound
    // sub-terms, evaluation_error(zero_divisor) for /0 (regardless
    // of operand type — see SPECIFICATION §6 + §6.1), and
    // type_error(evaluable, Culprit) for non-evaluable atoms or
    // unknown functors. Classification happens BEFORE eval so the
    // diagnostic captures the actual culprit shape.
    if (op == "is_iso/2") {
        CellPtr rhs_cell = get_cell("A2");
        if (term_contains_unbound(rhs_cell)) {
            return throw_iso_error(make_instantiation_error());
        }
        if (term_has_zero_divide(rhs_cell)) {
            return throw_iso_error(make_evaluation_error("zero_divisor"));
        }
        bool ok = true;
        Value rhs = eval_arith(rhs_cell, ok);
        if (!ok) {
            Value culprit = arith_culprit(deref(*rhs_cell));
            return throw_iso_error(make_type_error("evaluable", culprit));
        }
        CellPtr lhs = get_cell("A1");
        if (lhs->is_unbound()) { bind_cell(lhs, std::move(rhs)); pc += 1; return true; }
        if (!(*lhs == rhs)) return false;
        pc += 1; return true;
    }

    // ---- Arithmetic comparisons (lax / default) --------------------
    // The _lax/2 keys share this body so user code can write e.g.
    // `>_lax(X, Y)` to opt out of an enclosing ISO-mode predicate''s
    // rewrite (three-forms guarantee from PHILOSOPHY §3.3). The
    // dispatch op-name is also passed to arith_compare for the
    // operator semantics, so we strip the "_lax" suffix first.
    if (op == ">/2" || op == "</2" || op == ">=/2" || op == "=</2"
        || op == "=:=/2" || op == "=\\\\=/2"
        || op == ">_lax/2" || op == "<_lax/2" || op == ">=_lax/2"
        || op == "=<_lax/2" || op == "=:=_lax/2" || op == "=\\\\=_lax/2") {
        bool ok = true;
        Value a = eval_arith(get_cell("A1"), ok);
        if (!ok) return false;
        Value b = eval_arith(get_cell("A2"), ok);
        if (!ok) return false;
        std::string base_op = op;
        // Strip "_lax" before "/" so arith_compare sees ">/2" etc.
        auto pos = base_op.find("_lax/");
        if (pos != std::string::npos) base_op.replace(pos, 4, "");
        if (!arith_compare(base_op, a, b)) return false;
        pc += 1; return true;
    }
    // ---- Arithmetic comparisons (ISO) ------------------------------
    // ISO arith compares throw the same errors is_iso/2 throws:
    // instantiation_error for unbound args, evaluation_error for
    // zero_divisor in either operand subterm,
    // type_error(evaluable, ...) for non-evaluable atoms / unknown
    // functors. arith_compare gets the un-suffixed operator key.
    if (op == ">_iso/2" || op == "<_iso/2" || op == ">=_iso/2"
        || op == "=<_iso/2" || op == "=:=_iso/2" || op == "=\\\\=_iso/2") {
        CellPtr a_cell = get_cell("A1");
        CellPtr b_cell = get_cell("A2");
        if (term_contains_unbound(a_cell) || term_contains_unbound(b_cell)) {
            return throw_iso_error(make_instantiation_error());
        }
        if (term_has_zero_divide(a_cell) || term_has_zero_divide(b_cell)) {
            return throw_iso_error(make_evaluation_error("zero_divisor"));
        }
        bool ok = true;
        Value a = eval_arith(a_cell, ok);
        if (!ok) {
            Value culprit = arith_culprit(deref(*a_cell));
            return throw_iso_error(make_type_error("evaluable", culprit));
        }
        Value b = eval_arith(b_cell, ok);
        if (!ok) {
            Value culprit = arith_culprit(deref(*b_cell));
            return throw_iso_error(make_type_error("evaluable", culprit));
        }
        std::string base_op = op;
        auto pos = base_op.find("_iso/");
        if (pos != std::string::npos) base_op.replace(pos, 4, "");
        if (!arith_compare(base_op, a, b)) return false;
        pc += 1; return true;
    }

    // ---- ==/2 and \\==/2 (structural equality) ----------------------
    if (op == "==/2" || op == "\\\\==/2") {
        Value a = *get_cell("A1");
        Value b = *get_cell("A2");
        bool eq = (a == b);
        if (op == "==/2"  && !eq) return false;
        if (op == "\\\\==/2" && eq) return false;
        pc += 1; return true;
    }

    // ---- Type checks -----------------------------------------------
    if (op == "atom/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Atom) return false;
        pc += 1; return true;
    }
    if (op == "integer/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Integer) return false;
        pc += 1; return true;
    }
    if (op == "float/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Float) return false;
        pc += 1; return true;
    }
    if (op == "number/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Integer && a.tag != Value::Tag::Float) return false;
        pc += 1; return true;
    }
    if (op == "atomic/1") {
        Value a = *get_cell("A1");
        if (a.tag == Value::Tag::Compound || a.is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "compound/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Compound) return false;
        pc += 1; return true;
    }
    if (op == "var/1") {
        if (!get_cell("A1")->is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "nonvar/1") {
        if (get_cell("A1")->is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "ground/1") {
        // Recursive groundness check.
        std::function<bool(const Value&)> g = [&](const Value& v) -> bool {
            if (v.is_unbound()) return false;
            if (v.tag == Value::Tag::Compound) {
                for (auto& c : v.args) if (!g(*c)) return false;
            }
            return true;
        };
        if (!g(*get_cell("A1"))) return false;
        pc += 1; return true;
    }

    // ---- atom_codes/2, atom_chars/2, number_codes/2 ----------------
    // Bidirectional decomposition. Forward (A1 bound): split A1 into
    // a list of integer codes (codes/) or single-char atoms (chars/),
    // unify with A2. Reverse (A2 bound): walk A2 list, reassemble
    // the string, unify with A1. number_codes/2 additionally parses
    // the reassembled string as an integer or float.
    if (op == "atom_codes/2" || op == "atom_chars/2"
        || op == "number_codes/2") {
        bool is_codes = (op != "atom_chars/2");   // codes vs single-char atoms
        bool is_number = (op == "number_codes/2");
        Value a1v = deref(*get_cell("A1"));
        // Forward path: A1 bound → render to string, build list.
        if (!a1v.is_unbound()) {
            std::string buf;
            if (a1v.tag == Value::Tag::Atom) {
                if (is_number) return false; // number_codes needs a number
                buf = a1v.s;
            } else if (a1v.tag == Value::Tag::Integer) {
                buf = std::to_string(a1v.i);
            } else if (a1v.tag == Value::Tag::Float) {
                buf = render(a1v);
            } else {
                return false;
            }
            CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
            for (auto it = buf.rbegin(); it != buf.rend(); ++it) {
                CellPtr head;
                if (is_codes) {
                    head = std::make_shared<Cell>(Value::Integer(
                        static_cast<std::int64_t>(
                            static_cast<unsigned char>(*it))));
                } else {
                    head = std::make_shared<Cell>(
                        Value::Atom(std::string(1, *it)));
                }
                std::vector<CellPtr> cons_args;
                cons_args.push_back(head);
                cons_args.push_back(list);
                list = std::make_shared<Cell>(
                    Value::Compound("[|]/2", std::move(cons_args)));
            }
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, *list); pc += 1; return true; }
            if (!unify_cells(tgt, list)) return false;
            pc += 1; return true;
        }
        // Reverse path: A2 must be a ground list. Read each cell,
        // accumulate, then build the atom / number and unify with A1.
        std::string buf;
        CellPtr lc = get_cell("A2");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            Value hv = deref(*lv.args[0]);
            if (is_codes) {
                if (hv.tag != Value::Tag::Integer) return false;
                buf.push_back(static_cast<char>(
                    static_cast<unsigned char>(hv.i)));
            } else {
                if (hv.tag != Value::Tag::Atom || hv.s.size() != 1)
                    return false;
                buf.push_back(hv.s[0]);
            }
            lc = lv.args[1];
        }
        Value result;
        if (is_number) {
            // Try integer first, then float. Empty / malformed → fail.
            if (buf.empty()) return false;
            try {
                std::size_t pos = 0;
                std::int64_t i = std::stoll(buf, &pos);
                if (pos == buf.size()) {
                    result = Value::Integer(i);
                } else {
                    double d = std::stod(buf, &pos);
                    if (pos != buf.size()) return false;
                    result = Value::Float(d);
                }
            } catch (...) { return false; }
        } else {
            result = Value::Atom(buf);
        }
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
        pc += 1; return true;
    }

    // ---- atom_concat/3 ----------------------------------------------
    // atom_concat(+A1, +A2, ?A3) — concatenate the renderings of A1
    // and A2 into A3. The (-, -, +) split mode is nondeterministic
    // and not supported here; for that, use sub_atom (planned).
    if (op == "atom_concat/3") {
        Value a1 = deref(*get_cell("A1"));
        Value a2 = deref(*get_cell("A2"));
        if (a1.is_unbound() || a2.is_unbound()) return false;
        std::string s1 = (a1.tag == Value::Tag::Atom) ? a1.s : render(a1);
        std::string s2 = (a2.tag == Value::Tag::Atom) ? a2.s : render(a2);
        Value result = Value::Atom(s1 + s2);
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
        pc += 1; return true;
    }

    // ---- atom_length/2 ----------------------------------------------
    if (op == "atom_length/2") {
        Value a = deref(*get_cell("A1"));
        if (a.is_unbound()) return false;
        std::string s;
        if (a.tag == Value::Tag::Atom) s = a.s;
        else if (a.tag == Value::Tag::Integer) s = std::to_string(a.i);
        else return false;
        Value result = Value::Integer(static_cast<std::int64_t>(s.size()));
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
        pc += 1; return true;
    }

    // ---- char_code/2 ------------------------------------------------
    // Bidirectional. (+Char, ?Code) → unify Code with the integer code
    // of single-char atom Char. (?Char, +Code) → build the single-char
    // atom for Code and unify with Char.
    if (op == "char_code/2") {
        Value cv = deref(*get_cell("A1"));
        Value iv = deref(*get_cell("A2"));
        if (cv.tag == Value::Tag::Atom && cv.s.size() == 1) {
            Value code = Value::Integer(
                static_cast<std::int64_t>(
                    static_cast<unsigned char>(cv.s[0])));
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, code); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(code))) return false;
            pc += 1; return true;
        }
        if (iv.tag == Value::Tag::Integer && iv.i >= 0 && iv.i <= 255) {
            Value ch = Value::Atom(std::string(
                1, static_cast<char>(static_cast<unsigned char>(iv.i))));
            CellPtr tgt = get_cell("A1");
            if (tgt->is_unbound()) { bind_cell(tgt, ch); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(ch))) return false;
            pc += 1; return true;
        }
        return false;
    }

    // ---- char_type/2 ------------------------------------------------
    // ISO-style character classification + bidirectional case
    // conversion. Supports a useful subset of SWI''s char_type/2:
    //   alpha, alnum, digit, whitespace, space, punct, ascii,
    //   upper, lower (simple classifications; deterministic check),
    //   upper(Lower), lower(Upper) — Char is upper/lower; the arg
    //                  unifies with the opposite case.
    //   to_upper(U), to_lower(L) — bidirectional case conversion
    //                  (Char or U/L may be unbound).
    //   digit(Weight)  — digit char ↔ integer 0-9.
    //   code(Code)     — Char''s code ↔ integer (bidirectional;
    //                    overlaps with char_code/2).
    if (op == "char_type/2") {
        Value cv = deref(*get_cell("A1"));
        Value tv = deref(*get_cell("A2"));
        // Helper: extract char if A1 is a single-char atom; -1 if not.
        auto char_or_minus = [&]() -> int {
            if (cv.tag == Value::Tag::Atom && cv.s.size() == 1)
                return static_cast<unsigned char>(cv.s[0]);
            return -1;
        };
        // Forward classifications (Char bound, Type bound to an atom).
        if (tv.tag == Value::Tag::Atom) {
            int c = char_or_minus();
            if (c < 0) return false;
            const std::string& t = tv.s;
            bool ok = false;
            if (t == "alpha")      ok = std::isalpha(c) != 0;
            else if (t == "alnum")  ok = std::isalnum(c) != 0;
            else if (t == "digit")  ok = std::isdigit(c) != 0;
            else if (t == "whitespace") ok = std::isspace(c) != 0;
            else if (t == "space")  ok = std::isspace(c) != 0;
            else if (t == "punct")  ok = std::ispunct(c) != 0;
            else if (t == "ascii")  ok = (c >= 0 && c <= 127);
            else if (t == "upper")  ok = std::isupper(c) != 0;
            else if (t == "lower")  ok = std::islower(c) != 0;
            else return false;
            if (!ok) return false;
            pc += 1; return true;
        }
        // Compound Type — case conversion or digit(W) / code(N).
        if (tv.tag == Value::Tag::Compound && tv.args.size() == 1) {
            const std::string& tname = tv.s;
            CellPtr arg = tv.args[0];
            Value av = deref(*arg);
            // upper(Lower): Char must be uppercase; bind Lower to lowercase form.
            // lower(Upper): Char must be lowercase; bind Upper to uppercase form.
            if (tname == "upper/1") {
                int c = char_or_minus();
                if (c < 0 || !std::isupper(c)) return false;
                Value lo = Value::Atom(std::string(
                    1, static_cast<char>(std::tolower(c))));
                if (av.is_unbound()) { bind_cell(arg, lo); pc += 1; return true; }
                if (!unify_cells(arg, std::make_shared<Cell>(lo))) return false;
                pc += 1; return true;
            }
            if (tname == "lower/1") {
                int c = char_or_minus();
                if (c < 0 || !std::islower(c)) return false;
                Value up = Value::Atom(std::string(
                    1, static_cast<char>(std::toupper(c))));
                if (av.is_unbound()) { bind_cell(arg, up); pc += 1; return true; }
                if (!unify_cells(arg, std::make_shared<Cell>(up))) return false;
                pc += 1; return true;
            }
            // to_lower(L) / to_upper(U): bidirectional case conversion.
            // Either A1 or the inner arg may drive.
            if (tname == "to_lower/1" || tname == "to_upper/1") {
                bool to_lower = (tname == "to_lower/1");
                int c = char_or_minus();
                if (c >= 0) {
                    char r = to_lower ? static_cast<char>(std::tolower(c))
                                       : static_cast<char>(std::toupper(c));
                    Value cv2 = Value::Atom(std::string(1, r));
                    if (av.is_unbound()) { bind_cell(arg, cv2); pc += 1; return true; }
                    if (!unify_cells(arg, std::make_shared<Cell>(cv2))) return false;
                    pc += 1; return true;
                }
                // Char is unbound — derive from arg.
                if (av.tag == Value::Tag::Atom && av.s.size() == 1) {
                    int ac = static_cast<unsigned char>(av.s[0]);
                    char r = to_lower ? static_cast<char>(std::toupper(ac))
                                       : static_cast<char>(std::tolower(ac));
                    Value out = Value::Atom(std::string(1, r));
                    CellPtr tgt = get_cell("A1");
                    if (tgt->is_unbound()) { bind_cell(tgt, out); pc += 1; return true; }
                    if (!unify_cells(tgt, std::make_shared<Cell>(out))) return false;
                    pc += 1; return true;
                }
                return false;
            }
            // digit(Weight): forward (char→0-9), reverse (0-9→char).
            if (tname == "digit/1") {
                int c = char_or_minus();
                if (c >= 0) {
                    if (!std::isdigit(c)) return false;
                    Value w = Value::Integer(c - ''0'');
                    if (av.is_unbound()) { bind_cell(arg, w); pc += 1; return true; }
                    if (!unify_cells(arg, std::make_shared<Cell>(w))) return false;
                    pc += 1; return true;
                }
                if (av.tag == Value::Tag::Integer && av.i >= 0 && av.i <= 9) {
                    Value ch = Value::Atom(std::string(
                        1, static_cast<char>(''0'' + av.i)));
                    CellPtr tgt = get_cell("A1");
                    if (tgt->is_unbound()) { bind_cell(tgt, ch); pc += 1; return true; }
                    if (!unify_cells(tgt, std::make_shared<Cell>(ch))) return false;
                    pc += 1; return true;
                }
                return false;
            }
            // code(Code): same shape as char_code/2.
            if (tname == "code/1") {
                int c = char_or_minus();
                if (c >= 0) {
                    Value w = Value::Integer(c);
                    if (av.is_unbound()) { bind_cell(arg, w); pc += 1; return true; }
                    if (!unify_cells(arg, std::make_shared<Cell>(w))) return false;
                    pc += 1; return true;
                }
                if (av.tag == Value::Tag::Integer && av.i >= 0 && av.i <= 255) {
                    Value ch = Value::Atom(std::string(
                        1, static_cast<char>(av.i)));
                    CellPtr tgt = get_cell("A1");
                    if (tgt->is_unbound()) { bind_cell(tgt, ch); pc += 1; return true; }
                    if (!unify_cells(tgt, std::make_shared<Cell>(ch))) return false;
                    pc += 1; return true;
                }
                return false;
            }
        }
        return false;
    }

    // ---- upcase_atom/2, downcase_atom/2 -----------------------------
    // Whole-atom case conversion. (+Atom, ?Result).
    if (op == "upcase_atom/2" || op == "downcase_atom/2") {
        Value av = deref(*get_cell("A1"));
        if (av.tag != Value::Tag::Atom) return false;
        std::string out = av.s;
        if (op == "upcase_atom/2") {
            for (auto& ch : out) ch = static_cast<char>(std::toupper(
                static_cast<unsigned char>(ch)));
        } else {
            for (auto& ch : out) ch = static_cast<char>(std::tolower(
                static_cast<unsigned char>(ch)));
        }
        Value rv = Value::Atom(out);
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, rv); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(rv))) return false;
        pc += 1; return true;
    }

    // ---- assertz/1, asserta/1, retract/1, retractall/1 -------------
    // Dynamic database manipulation. Both FACTS and RULES are
    // supported — a rule is stored as a ":-/2"(Head, Body) compound,
    // and dynamic_try_next decomposes it on dispatch (binding head
    // args, then invoking body as a goal-term).
    //
    // Storage: dynamic_db["name/arity"] is a std::vector<CellPtr> of
    // deep-copied clause terms. The key is derived from the head''s
    // functor (or, for rules, from the head inside ":-/2").
    auto dyn_key_of = [&](const Value& v, std::string& out) -> bool {
        // Rule form ":-/2"(Head, Body): index by Head''s functor.
        if (v.tag == Value::Tag::Compound && v.s == ":-/2"
            && v.args.size() == 2) {
            Value head = deref(*v.args[0]);
            if (head.tag == Value::Tag::Atom) { out = head.s + "/0"; return true; }
            if (head.tag == Value::Tag::Compound) { out = head.s; return true; }
            return false;
        }
        // Fact form: the term itself is the head.
        if (v.tag == Value::Tag::Atom) {
            out = v.s + "/0";
            return true;
        }
        if (v.tag == Value::Tag::Compound) {
            out = v.s; // already "name/arity"
            return true;
        }
        return false;
    };
    if (op == "assertz/1" || op == "asserta/1") {
        Value t = deref(*get_cell("A1"));
        if (t.is_unbound()) return false;
        std::string key;
        if (!dyn_key_of(t, key)) return false;
        // Deep-copy so the stored term has FRESH unbound vars,
        // independent of the caller''s vars (which will be unwound
        // on backtrack).
        CellPtr stored = deep_copy_term(get_cell("A1"));
        auto& vec = dynamic_db[key];
        if (op == "assertz/1") vec.push_back(stored);
        else                   vec.insert(vec.begin(), stored);
        pc += 1; return true;
    }
    // Note: retract/1 is nondeterministic — dispatched via
    // dispatch_retract from the Call/Execute step arms, not here.
    // See is_builtin_pred for the corresponding compile-side change.
    if (op == "retractall/1") {
        // Remove every clause whose head unifies with the pattern.
        // Always succeeds (even when there are no matches, per ISO).
        // Pattern bindings are undone after each match attempt so
        // they don''t leak; only the database changes persist.
        Value t = deref(*get_cell("A1"));
        if (t.is_unbound()) return false;
        std::string key;
        if (!dyn_key_of(t, key)) return false;
        auto db_it = dynamic_db.find(key);
        if (db_it != dynamic_db.end()) {
            CellPtr pattern = get_cell("A1");
            auto& vec = db_it->second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const CellPtr& clause) {
                    std::size_t mark = trail.size();
                    CellPtr fresh = deep_copy_term(clause);
                    bool ok = unify_cells(pattern, fresh);
                    // Undo bindings either way.
                    while (trail.size() > mark) {
                        TrailEntry te = std::move(trail.back());
                        trail.pop_back();
                        *te.cell = std::move(te.prev);
                    }
                    return ok;
                }), vec.end());
        }
        pc += 1; return true;
    }

    // ---- nb_setval/2, nb_getval/2, b_setval/2, b_getval/2 -----------
    // Mutable globals. Key must be a ground atom. nb_setval REPLACES
    // the stored value (no undo on backtrack); b_setval MUTATES the
    // existing cell via bind_cell so the trail records the prior
    // content, and backtrack restores it. Both deep-copy the value
    // so the stored term has fresh variables (independent of the
    // caller''s bindings). getvals deep-copy on retrieval so the
    // returned term''s vars are fresh too — repeated reads see the
    // same shape but disjoint vars.
    auto read_global_key = [&](std::string& out) -> bool {
        Value k = deref(*get_cell("A1"));
        if (k.tag != Value::Tag::Atom) return false;
        out = k.s;
        return true;
    };
    if (op == "nb_setval/2") {
        std::string key;
        if (!read_global_key(key)) return false;
        Value v = deref(*get_cell("A2"));
        if (v.is_unbound()) return false; // need a concrete value
        CellPtr fresh = deep_copy_term(get_cell("A2"));
        nb_globals[key] = std::move(fresh);
        pc += 1; return true;
    }
    if (op == "b_setval/2") {
        std::string key;
        if (!read_global_key(key)) return false;
        Value v = deref(*get_cell("A2"));
        if (v.is_unbound()) return false;
        CellPtr fresh = deep_copy_term(get_cell("A2"));
        auto it = nb_globals.find(key);
        if (it == nb_globals.end()) {
            // First binding for this key — no prior cell to mutate,
            // so just install fresh. Subsequent b_setvals will mutate
            // this cell in place so the trail can restore them.
            nb_globals[key] = std::move(fresh);
        } else {
            // Mutate the existing cell so the trail entry can undo
            // it on backtrack.
            bind_cell(it->second, *fresh);
        }
        pc += 1; return true;
    }
    if (op == "nb_getval/2" || op == "b_getval/2") {
        std::string key;
        if (!read_global_key(key)) return false;
        auto it = nb_globals.find(key);
        if (it == nb_globals.end()) return false;
        // Deep-copy on retrieval so each get returns a fresh-var
        // copy; repeated reads share structure but not bindings.
        CellPtr fresh = deep_copy_term(it->second);
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, *fresh); pc += 1; return true; }
        if (!unify_cells(tgt, fresh)) return false;
        pc += 1; return true;
    }

    // ---- @</2, @=</2, @>/2, @>=/2, compare/3 -----------------------
    // ISO §7.2 standard order of terms (delegated to
    // standard_order_cmp). These don''t unify their arguments; they
    // just compare the current term values.
    if (op == "@</2" || op == "@=</2" || op == "@>/2" || op == "@>=/2") {
        int c = standard_order_cmp(deref(*get_cell("A1")),
                                   deref(*get_cell("A2")));
        bool ok = false;
        if (op == "@</2")  ok = (c < 0);
        else if (op == "@=</2") ok = (c <= 0);
        else if (op == "@>/2")  ok = (c > 0);
        else                    ok = (c >= 0); // "@>="
        if (!ok) return false;
        pc += 1; return true;
    }
    if (op == "compare/3") {
        // compare(?Order, @A, @B) — Order unifies with one of <, =, >.
        // If Order is bound to a different atom, fail.
        int c = standard_order_cmp(deref(*get_cell("A2")),
                                   deref(*get_cell("A3")));
        const char* sym = (c < 0) ? "<" : (c > 0 ? ">" : "=");
        Value ov = Value::Atom(sym);
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, ov); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(ov))) return false;
        pc += 1; return true;
    }
    // ---- numlist/3 --------------------------------------------------
    // numlist(+Low, +High, -List) — generate [Low, Low+1, ..., High].
    // Empty list if Low > High. Both bounds must be integers.
    if (op == "numlist/3") {
        Value lv = deref(*get_cell("A1"));
        Value hv = deref(*get_cell("A2"));
        if (lv.tag != Value::Tag::Integer
            || hv.tag != Value::Tag::Integer) return false;
        std::int64_t lo = lv.i, hi = hv.i;
        // Build the list bottom-up (tail = [], cons each value).
        CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
        for (std::int64_t i = hi; i >= lo; --i) {
            CellPtr head = std::make_shared<Cell>(Value::Integer(i));
            std::vector<CellPtr> cons_args;
            cons_args.push_back(head);
            cons_args.push_back(list);
            list = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(cons_args)));
        }
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, *list); pc += 1; return true; }
        if (!unify_cells(tgt, list)) return false;
        pc += 1; return true;
    }

    // ---- sort/2, msort/2 --------------------------------------------
    // sort(+List, -Sorted) — standard order, dedup.
    // msort(+List, -Sorted) — standard order, NO dedup (stable).
    if (op == "sort/2" || op == "msort/2") {
        // Walk A1 as a list, deep-copying each element so subsequent
        // unifications can''t mutate the sort key.
        std::vector<CellPtr> items;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            // Take the cell directly — sort/msort don''t recurse into
            // user code, so binding mutations can''t race with the
            // sort''s element comparisons.
            items.push_back(lv.args[0]);
            lc = lv.args[1];
        }
        std::stable_sort(items.begin(), items.end(),
            [](const CellPtr& a, const CellPtr& b) {
                return standard_order_cmp(*a, *b) < 0;
            });
        if (op == "sort/2") {
            // Dedup adjacent equals.
            items.erase(std::unique(items.begin(), items.end(),
                [](const CellPtr& a, const CellPtr& b) {
                    return standard_order_cmp(*a, *b) == 0;
                }), items.end());
        }
        // Build result list.
        CellPtr result = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            std::vector<CellPtr> cons_args;
            cons_args.push_back(*it);
            cons_args.push_back(result);
            result = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(cons_args)));
        }
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, *result); pc += 1; return true; }
        if (!unify_cells(tgt, result)) return false;
        pc += 1; return true;
    }

    // ---- keysort/2 --------------------------------------------------
    // keysort(+Pairs, -Sorted) — stable sort of Key-Value pairs by
    // Key (standard order). Pairs must be ground at the head; the
    // Value side is opaque (sorted only by Key). Items that aren''t
    // -/2 pairs cause failure.
    if (op == "keysort/2") {
        std::vector<CellPtr> items;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            // Validate that the head is a -/2 pair.
            Value hv = deref(*lv.args[0]);
            if (hv.tag != Value::Tag::Compound || hv.s != "-/2"
                || hv.args.size() != 2) return false;
            items.push_back(lv.args[0]);
            lc = lv.args[1];
        }
        // Stable sort by the pair''s Key (args[0]) — Values stay in
        // their original relative order on key ties.
        std::stable_sort(items.begin(), items.end(),
            [](const CellPtr& a, const CellPtr& b) {
                return standard_order_cmp(*(a->args[0]),
                                          *(b->args[0])) < 0;
            });
        CellPtr result = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            std::vector<CellPtr> cons_args;
            cons_args.push_back(*it);
            cons_args.push_back(result);
            result = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(cons_args)));
        }
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, *result); pc += 1; return true; }
        if (!unify_cells(tgt, result)) return false;
        pc += 1; return true;
    }

    // ---- functor/3 -------------------------------------------------
    if (op == "functor/3") {
        CellPtr t = get_cell("A1");
        if (!t->is_unbound()) {
            // Decompose: t -> functor name + arity
            const Value& v = *t;
            std::string name;
            std::int64_t arity = 0;
            if (v.tag == Value::Tag::Compound) {
                name = v.s;
                auto sl = name.find_last_of(\'/\');
                if (sl != std::string::npos) name.resize(sl);
                arity = (std::int64_t)v.args.size();
            } else if (v.tag == Value::Tag::Atom) {
                name = v.s; arity = 0;
            } else if (v.tag == Value::Tag::Integer) {
                // Numbers are their own functor with arity 0.
                CellPtr fc = get_cell("A2");
                CellPtr ac = get_cell("A3");
                if (fc->is_unbound()) bind_cell(fc, v);
                else if (!(*fc == v)) return false;
                Value zero = Value::Integer(0);
                if (ac->is_unbound()) bind_cell(ac, zero);
                else if (!(*ac == zero)) return false;
                pc += 1; return true;
            } else {
                return false;
            }
            CellPtr fc = get_cell("A2");
            CellPtr ac = get_cell("A3");
            Value fv = Value::Atom(name);
            Value av = Value::Integer(arity);
            if (fc->is_unbound()) bind_cell(fc, fv); else if (!(*fc == fv)) return false;
            if (ac->is_unbound()) bind_cell(ac, av); else if (!(*ac == av)) return false;
            pc += 1; return true;
        }
        // Build mode: A2 = functor name, A3 = arity.
        Value name_v = *get_cell("A2");
        Value ar_v   = *get_cell("A3");
        if (name_v.tag != Value::Tag::Atom || ar_v.tag != Value::Tag::Integer) return false;
        if (ar_v.i == 0) { bind_cell(t, name_v); pc += 1; return true; }
        std::vector<CellPtr> args;
        for (std::int64_t k = 0; k < ar_v.i; ++k) {
            args.push_back(std::make_shared<Value>(Value::Unbound("_FA" + std::to_string(k))));
        }
        std::string functor = name_v.s + "/" + std::to_string(ar_v.i);
        bind_cell(t, Value::Compound(functor, std::move(args)));
        pc += 1; return true;
    }

    // ---- arg/3 -----------------------------------------------------
    if (op == "arg/3") {
        Value n_v = *get_cell("A1");
        Value t_v = *get_cell("A2");
        if (n_v.tag != Value::Tag::Integer) return false;
        if (t_v.tag != Value::Tag::Compound) return false;
        std::int64_t idx = n_v.i;
        if (idx < 1 || (std::size_t)idx > t_v.args.size()) return false;
        CellPtr want = t_v.args[idx - 1];
        if (!unify_cells(get_cell("A3"), want)) return false;
        pc += 1; return true;
    }

    // ---- =../2 (univ) ----------------------------------------------
    if (op == "=../2") {
        CellPtr t = get_cell("A1");
        CellPtr l = get_cell("A2");
        if (!t->is_unbound()) {
            // Decompose: t -> [Functor | Args]
            const Value& v = *t;
            std::vector<CellPtr> items;
            if (v.tag == Value::Tag::Compound) {
                std::string name = v.s;
                auto sl = name.find_last_of(\'/\');
                if (sl != std::string::npos) name.resize(sl);
                items.push_back(std::make_shared<Value>(Value::Atom(name)));
                for (auto& a : v.args) items.push_back(a);
            } else {
                // Atomic term: univ -> [v]
                items.push_back(std::make_shared<Value>(v));
            }
            // Build [|]/2 chain
            Value tail = Value::Atom("[]");
            for (auto it = items.rbegin(); it != items.rend(); ++it) {
                std::vector<CellPtr> args;
                args.push_back(*it);
                args.push_back(std::make_shared<Value>(std::move(tail)));
                tail = Value::Compound("[|]/2", std::move(args));
            }
            CellPtr built = std::make_shared<Value>(std::move(tail));
            if (!unify_cells(l, built)) return false;
            pc += 1; return true;
        }
        // Build mode: collect list elements from A2.
        std::vector<Value> items;
        const Value* cur = l.get();
        while (cur && cur->tag == Value::Tag::Compound
               && cur->s == "[|]/2" && cur->args.size() == 2) {
            items.push_back(*cur->args[0]);
            cur = cur->args[1].get();
        }
        if (!(cur && cur->tag == Value::Tag::Atom && cur->s == "[]")) return false;
        if (items.empty()) return false;
        if (items.size() == 1) {
            bind_cell(t, items[0]);
            pc += 1; return true;
        }
        if (items[0].tag != Value::Tag::Atom) return false;
        std::vector<CellPtr> args;
        for (std::size_t k = 1; k < items.size(); ++k) {
            args.push_back(std::make_shared<Value>(items[k]));
        }
        std::string functor = items[0].s + "/" + std::to_string(items.size() - 1);
        bind_cell(t, Value::Compound(functor, std::move(args)));
        pc += 1; return true;
    }

    // ---- I/O -------------------------------------------------------
    if (op == "write/1") {
        std::printf("%s", render(*get_cell("A1")).c_str());
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "nl/0") {
        std::printf("\\n");
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "write_atom/1" || op == "writeln/1") {
        std::printf("%s\\n", render(*get_cell("A1")).c_str());
        std::fflush(stdout);
        pc += 1; return true;
    }
    // ---- format/1, format/2, format/3 -------------------------------
    // Walks the format string, expanding ~-directives.
    // Supported: ~w (write), ~p (print, same as ~w), ~a (atom),
    // ~d (integer), ~s (codes/atom), ~n (newline), ~~ (literal ~).
    // Unsupported directives are echoed verbatim. Args list is walked
    // left-to-right; running off the end of args for a directive that
    // needs one fails.
    //
    // Shape variants:
    //   format/1   — just a format string, no args (implicit []).
    //   format/2   — Format, Args.
    //   format/3   — Dest, Format, Args. Dest selects output:
    //     atom user_output / user_error → stdout / stderr (printed).
    //     atom(V) / string(V)           → V is unified with the rendered
    //                                     string as an Atom value.
    //     codes(V)                      → V is unified with a list of
    //                                     character codes.
    if (op == "format/1" || op == "format/2" || op == "format/3") {
        bool is_f3 = (op == "format/3");
        // Resolve format-string + args-list positions.
        CellPtr fmt_cell  = get_cell(is_f3 ? "A2" : "A1");
        CellPtr args_cell = (op == "format/2")
            ? get_cell("A2")
            : (is_f3 ? get_cell("A3") : CellPtr{});
        Value fv = deref(*fmt_cell);
        std::string fmt;
        if (fv.tag == Value::Tag::Atom) fmt = fv.s;
        else if (fv.tag == Value::Tag::Integer) fmt = std::to_string(fv.i);
        else return false;
        // Build a vector of arg cells from the args list (if any).
        std::vector<CellPtr> args;
        if (args_cell) {
            CellPtr lc = args_cell;
            for (;;) {
                Value lv = deref(*lc);
                if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
                if (lv.tag == Value::Tag::Compound && lv.s == "[|]/2"
                    && lv.args.size() == 2) {
                    args.push_back(lv.args[0]);
                    lc = lv.args[1];
                    continue;
                }
                // Malformed args list (partial / unbound tail).
                return false;
            }
        }
        // Render into a buffer first (used by format/3 string-destinations
        // and by stdout/stderr printing alike — single rendering path).
        std::string buf;
        std::size_t ai = 0;
        for (std::size_t i = 0; i < fmt.size(); ++i) {
            char c = fmt[i];
            if (c != ''~'' || i + 1 >= fmt.size()) {
                buf.push_back(c);
                continue;
            }
            char d = fmt[++i];
            switch (d) {
                case ''n'': buf.push_back(''\\n''); break;
                case ''t'': buf.push_back(''\\t''); break;
                case ''~'': buf.push_back(''~''); break;
                case ''w'':
                case ''p'': {
                    if (ai >= args.size()) return false;
                    buf += render(deref(*args[ai++]));
                    break;
                }
                case ''a'': {
                    if (ai >= args.size()) return false;
                    Value v = deref(*args[ai++]);
                    if (v.tag == Value::Tag::Atom) buf += v.s;
                    else buf += render(v);
                    break;
                }
                case ''d'': {
                    if (ai >= args.size()) return false;
                    Value v = deref(*args[ai++]);
                    if (v.tag == Value::Tag::Integer)
                        buf += std::to_string(v.i);
                    else buf += render(v);
                    break;
                }
                case ''s'': {
                    // String form: accept an atom (use as-is) or a list
                    // of integer character codes.
                    if (ai >= args.size()) return false;
                    Value v = deref(*args[ai++]);
                    if (v.tag == Value::Tag::Atom) {
                        buf += v.s;
                    } else if (v.tag == Value::Tag::Compound
                               && v.s == "[|]/2") {
                        CellPtr lc = args[ai - 1];
                        for (;;) {
                            Value lv = deref(*lc);
                            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
                            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                                || lv.args.size() != 2) return false;
                            Value hv = deref(*lv.args[0]);
                            if (hv.tag != Value::Tag::Integer) return false;
                            buf.push_back(static_cast<char>(hv.i));
                            lc = lv.args[1];
                        }
                    } else {
                        buf += render(v);
                    }
                    break;
                }
                default:
                    // Unknown directive: echo literally.
                    buf.push_back(''~'');
                    buf.push_back(d);
                    break;
            }
        }
        // Dispatch on destination (format/3) or just print (format/1, /2).
        if (!is_f3) {
            std::fwrite(buf.data(), 1, buf.size(), stdout);
            std::fflush(stdout);
            pc += 1; return true;
        }
        // format/3: A1 selects destination.
        Value dv = deref(*get_cell("A1"));
        if (dv.tag == Value::Tag::Atom) {
            if (dv.s == "user_output") {
                std::fwrite(buf.data(), 1, buf.size(), stdout);
                std::fflush(stdout);
                pc += 1; return true;
            }
            if (dv.s == "user_error") {
                std::fwrite(buf.data(), 1, buf.size(), stderr);
                std::fflush(stderr);
                pc += 1; return true;
            }
        }
        if (dv.tag == Value::Tag::Compound && dv.args.size() == 1) {
            // atom(V) / string(V) → V unifies with the buffer as Atom.
            if (dv.s == "atom/1" || dv.s == "string/1") {
                Value out = Value::Atom(buf);
                CellPtr tgt = dv.args[0];
                if (tgt->is_unbound()) { bind_cell(tgt, out); pc += 1; return true; }
                if (!unify_cells(tgt, std::make_shared<Cell>(out))) return false;
                pc += 1; return true;
            }
            // codes(V) → V unifies with a [|]/2 list of integer codes,
            // built bottom-up: end with [], then prepend each code.
            if (dv.s == "codes/1") {
                CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
                for (auto it = buf.rbegin(); it != buf.rend(); ++it) {
                    CellPtr head = std::make_shared<Cell>(
                        Value::Integer(static_cast<std::int64_t>(
                            static_cast<unsigned char>(*it))));
                    std::vector<CellPtr> cons_args;
                    cons_args.push_back(head);
                    cons_args.push_back(list);
                    list = std::make_shared<Cell>(
                        Value::Compound("[|]/2", std::move(cons_args)));
                }
                CellPtr tgt = dv.args[0];
                if (tgt->is_unbound()) { bind_cell(tgt, *list); pc += 1; return true; }
                if (!unify_cells(tgt, list)) return false;
                pc += 1; return true;
            }
        }
        // Unrecognised destination — fail.
        return false;
    }

    return false;
}

// ----------------------------------------------------------------------
// Mode helpers for read/write of Get-/Put-Structure / Get-/Put-List.
// ----------------------------------------------------------------------

namespace {

void push_read_mode(std::vector<ModeFrame>& stack, const std::vector<CellPtr>& args) {
    ModeFrame m;
    m.kind = ModeFrame::Kind::Read;
    m.args = args;
    m.idx = 0;
    m.expected_arity = args.size();
    stack.push_back(std::move(m));
}

void push_write_mode(std::vector<ModeFrame>& stack, CellPtr target, std::size_t arity) {
    if (target) target->args.clear();
    ModeFrame m;
    m.kind = ModeFrame::Kind::Write;
    m.target = std::move(target);
    m.idx = 0;
    m.expected_arity = arity;
    stack.push_back(std::move(m));
}

// Auto-pop frames whose arity has been satisfied. Cascades upward so a
// fully-filled inner struct doesn''t leave subsequent Set*/Unify* writing
// into it.
void auto_pop_modes(std::vector<ModeFrame>& stack) {
    while (!stack.empty()) {
        ModeFrame& top = stack.back();
        bool full = false;
        if (top.kind == ModeFrame::Kind::Write && top.target
            && top.target->args.size() >= top.expected_arity) full = true;
        if (top.kind == ModeFrame::Kind::Read
            && top.idx >= top.expected_arity) full = true;
        if (!full) break;
        stack.pop_back();
    }
}

// Recursive deep-copy of a Value. Atom / Integer / Float / Unbound are
// independent — Compound rebuilds its args vector with fresh cells whose
// contents are themselves deep-copied, so later mutations to the source
// tree don''t leak into a collected snapshot (used by EndAggregate).
Value deep_copy(const Value& v) {
    Value out = v;
    if (v.tag == Value::Tag::Compound) {
        out.args.clear();
        for (auto& c : v.args) {
            out.args.push_back(std::make_shared<Cell>(deep_copy(*c)));
        }
    }
    return out;
}

} // anonymous

// Begin a write-mode structure / list. When the register''s existing
// cell is Unbound, we MUTATE it via bind_cell — so chained patterns
// like set_variable + put_structure that share a cell with a parent
// struct see the new compound. When the cell already holds a concrete
// value (atom/integer/compound), we instead allocate a FRESH cell and
// rebind only the named register (classic WAM "put_structure overwrites
// Ai" semantics) — other aliases keep their old value.
static void begin_write(WamState& vm, const std::string& reg_name,
                        const std::string& functor, std::size_t arity,
                        CellPtr& chosen) {
    CellPtr existing = vm.get_cell(reg_name);
    if (existing->is_unbound()) {
        vm.bind_cell(existing, Value::Compound(functor, {}));
        chosen = existing;
    } else {
        chosen = std::make_shared<Cell>(Value::Compound(functor, {}));
        vm.set_cell(reg_name, chosen);
    }
    push_write_mode(vm.mode_stack, chosen, arity);
}

// ----------------------------------------------------------------------
// Step
// ----------------------------------------------------------------------

bool WamState::step(const Instruction& instr) {
    switch (instr.op) {
        // ---- Head unification --------------------------------------
        case Instruction::Op::GetConstant: {
            CellPtr a = get_cell(instr.a);
            if (a->is_unbound())          { bind_cell(a, instr.val); }
            else if (!(*a == instr.val))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetInteger: {
            CellPtr a = get_cell(instr.a);
            Value want = Value::Integer(instr.n);
            if (a->is_unbound())     { bind_cell(a, want); }
            else if (!(*a == want))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetNil: {
            CellPtr a = get_cell(instr.a);
            Value want = Value::Atom("[]");
            if (a->is_unbound())     { bind_cell(a, want); }
            else if (!(*a == want))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetVariable: {
            // X-reg shares the same cell as A-reg.
            set_cell(instr.a, get_cell(instr.b));
            pc += 1; return true;
        }
        case Instruction::Op::GetValue: {
            if (!unify_cells(get_cell(instr.a), get_cell(instr.b))) return false;
            pc += 1; return true;
        }
        case Instruction::Op::GetStructure: {
            // instr.a = functor ("box/2"), instr.b = register ("A1" / "X1" ...).
            CellPtr a = get_cell(instr.b);
            const std::string& functor = instr.a;
            std::size_t arity = 0;
            auto p = functor.find_last_of(\'/\');
            if (p != std::string::npos) arity = std::stoull(functor.substr(p + 1));
            if (a->is_unbound()) {
                CellPtr target;
                begin_write(*this, instr.b, functor, arity, target);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == arity) {
                push_read_mode(mode_stack, a->args);
                pc += 1; return true;
            }
            return false;
        }
        case Instruction::Op::GetList: {
            CellPtr a = get_cell(instr.a);
            const std::string functor = "[|]/2";
            if (a->is_unbound()) {
                CellPtr target;
                begin_write(*this, instr.a, functor, 2, target);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == 2) {
                push_read_mode(mode_stack, a->args);
                pc += 1; return true;
            }
            return false;
        }

        // ---- Body construction -------------------------------------
        case Instruction::Op::PutConstant: {
            // Must allocate a fresh cell, not mutate the existing one:
            // any X-reg aliasing this register (e.g. via prior
            // get_variable) must NOT see the new value.
            set_cell(instr.a, make_cell(instr.val));
            pc += 1; return true;
        }
        case Instruction::Op::PutVariable: {
            // Fresh shared cell visible from both registers.
            CellPtr v = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
            set_cell(instr.a, v);
            set_cell(instr.b, v);
            pc += 1; return true;
        }
        case Instruction::Op::PutValue: {
            // Target reg shares the source''s cell.
            set_cell(instr.b, get_cell(instr.a));
            pc += 1; return true;
        }
        case Instruction::Op::PutStructure: {
            // X-regs and Y-regs are local — when set_variable
            // attached the cell to a parent structure''s args slot,
            // the cell is shared between the X/Y reg AND the parent.
            // Subsequent put_structure on that reg MUST bind in place
            // so the parent''s args slot sees the new structure too.
            //
            // A-regs are different — they''re argument-passing slots
            // that may be aliased with OTHER A-regs via PutValue
            // (e.g. the maplist/3 + call/3 case where the callee''s
            // A1 and A2 ended up sharing the same caller''s var).
            // Always allocate fresh for A-regs so we don''t corrupt
            // the aliased value.
            const std::string& functor = instr.a;
            std::size_t arity = 0;
            auto p = functor.find_last_of(\'/\');
            if (p != std::string::npos) arity = std::stoull(functor.substr(p + 1));
            CellPtr existing = get_cell(instr.b);
            bool is_a_reg = !instr.b.empty() && instr.b[0] == \'A\';
            CellPtr target;
            if (existing->is_unbound() && !is_a_reg) {
                bind_cell(existing, Value::Compound(functor, {}));
                target = existing;
            } else {
                target = std::make_shared<Cell>(
                    Value::Compound(functor, {}));
                set_cell(instr.b, target);
            }
            push_write_mode(mode_stack, target, arity);
            pc += 1; return true;
        }
        case Instruction::Op::PutList: {
            // See PutStructure — same A-reg-vs-X/Y-reg rule.
            CellPtr existing = get_cell(instr.a);
            bool is_a_reg = !instr.a.empty() && instr.a[0] == \'A\';
            CellPtr target;
            if (existing->is_unbound() && !is_a_reg) {
                bind_cell(existing, Value::Compound("[|]/2", {}));
                target = existing;
            } else {
                target = std::make_shared<Cell>(
                    Value::Compound("[|]/2", {}));
                set_cell(instr.a, target);
            }
            push_write_mode(mode_stack, target, 2);
            pc += 1; return true;
        }

        // ---- Unify (post Get-/Put-Structure / List) -----------------
        case Instruction::Op::UnifyVariable: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                set_cell(instr.a, m.args[m.idx]);
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
                set_cell(instr.a, fresh);
                m.target->args.push_back(fresh);
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::UnifyValue: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                if (!unify_cells(get_cell(instr.a), m.args[m.idx])) return false;
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                m.target->args.push_back(get_cell(instr.a));
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::UnifyConstant: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                CellPtr c = m.args[m.idx];
                if (c->is_unbound())         { bind_cell(c, instr.val); }
                else if (!(*c == instr.val)) { return false; }
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                m.target->args.push_back(make_cell(instr.val));
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }

        // ---- Set (always write mode) -------------------------------
        case Instruction::Op::SetVariable: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
            set_cell(instr.a, fresh);
            mode_stack.back().target->args.push_back(fresh);
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::SetValue: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            mode_stack.back().target->args.push_back(get_cell(instr.a));
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::SetConstant: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            mode_stack.back().target->args.push_back(make_cell(instr.val));
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }

        // ---- Environment frames -----------------------------------
        case Instruction::Op::Allocate: {
            EnvFrame f;
            f.saved_cp = cp;
            env_stack.push_back(std::move(f));
            pc += 1; return true;
        }
        case Instruction::Op::Deallocate: {
            if (!env_stack.empty()) {
                cp = env_stack.back().saved_cp;
                env_stack.pop_back();
            }
            pc += 1; return true;
        }

        // ---- Control flow ------------------------------------------
        case Instruction::Op::Call: {
            if (instr.a == "catch/3") { cp = pc + 1; return execute_catch(); }
            if (instr.a == "throw/1") { cp = pc + 1; return execute_throw(); }
            // findall/3, bagof/3, setof/3 meta-call — handles the
            // non-inlined cases (nested aggregates, where only the
            // OUTER one gets BeginAggregate-inlined; the inner is
            // emitted as a plain Call to findall/bagof/setof which
            // we dispatch here).
            if (instr.a == "findall/3") return dispatch_findall_call(pc + 1);
            if (instr.a == "bagof/3")   return dispatch_aggregate_call("bagof", pc + 1);
            if (instr.a == "setof/3")   return dispatch_aggregate_call("setof", pc + 1);
            // sub_atom/5 — nondeterministic substring enumeration.
            // Needs its own dispatch arm (not via builtin()) so the
            // CP machinery sees the correct continuation pc.
            if (instr.a == "sub_atom/5") return dispatch_sub_atom(pc + 1);
            // retract/1 — nondeterministic clause removal.
            if (instr.a == "retract/1") return dispatch_retract(pc + 1);
            // ^/2 — existential quantification. Transparent for our
            // find-style aggregation: invoke A2 (the goal) with the
            // standard non-tail after-pc.
            if (instr.a == "^/2") {
                return invoke_goal_as_call(get_cell("A2"), pc + 1);
            }
            // call/N meta — needs its own arm so the after_pc is
            // correctly pc + 1 (non-tail) rather than going through
            // the Execute fallback''s post-builtin pc=cp override.
            if (instr.a.size() > 5
                && instr.a.compare(0, 5, "call/") == 0) {
                return dispatch_call_meta(instr.a, instr.n, pc + 1);
            }
            auto it = labels.find(instr.a);
            if (it != labels.end()) {
                cp = pc + 1;
                pc = it->second;
                return true;
            }
            // Dynamic predicate path: assertz/retract have put facts
            // into dynamic_db. Iterate through them via the iterator
            // pattern. Tried before the builtin fallback so a user
            // can assertz over a builtin name (unusual, but allowed).
            if (dynamic_db.count(instr.a)) {
                return dispatch_dynamic_call(instr.a, pc + 1);
            }
            // No user predicate matches: fall back to builtin dispatch.
            // builtin() advances pc itself on success, mirroring the
            // BuiltinCall path.
            return builtin(instr.a, instr.n);
        }
        case Instruction::Op::Execute: {
            if (instr.a == "catch/3") return execute_catch();
            if (instr.a == "throw/1") return execute_throw();
            if (instr.a == "findall/3" || instr.a == "bagof/3"
                || instr.a == "setof/3") {
                std::size_t tail_after = cp;
                std::string kind =
                    (instr.a == "findall/3") ? "collect" :
                    (instr.a == "bagof/3")   ? "bagof"   : "setof";
                return dispatch_aggregate_call(kind, tail_after);
            }
            if (instr.a == "sub_atom/5") return dispatch_sub_atom(cp);
            if (instr.a == "retract/1") return dispatch_retract(cp);
            if (instr.a == "^/2") {
                return invoke_goal_as_call(get_cell("A2"), cp);
            }
            // call/N meta in tail position — after_pc is the
            // caller''s saved cp (or halt if cp == 0).
            if (instr.a.size() > 5
                && instr.a.compare(0, 5, "call/") == 0) {
                std::size_t tail_after = cp;
                bool ok = dispatch_call_meta(instr.a, instr.n, tail_after);
                if (!ok) return false;
                return true;
            }
            auto it = labels.find(instr.a);
            if (it != labels.end()) {
                pc = it->second;
                return true;
            }
            // Dynamic predicate path — tail-call form. after_pc is cp
            // (the caller''s saved continuation) for TCO.
            if (dynamic_db.count(instr.a)) {
                return dispatch_dynamic_call(instr.a, cp);
            }
            // Fall back to a deterministic builtin, then proceed to cp
            // since execute is a tail call (no in-body continuation).
            if (!builtin(instr.a, instr.n)) return false;
            if (cp == 0) { halt = true; return true; }
            pc = cp; cp = 0;
            return true;
        }
        case Instruction::Op::CatchReturn: {
            // Reached when a catch-protected goal proceeds normally.
            // Pop the topmost catcher frame and proceed to its
            // saved continuation.
            if (catcher_frames.empty()) {
                // Should not happen — defensive: halt as if normal proceed.
                if (cp == 0) { halt = true; return true; }
                pc = cp; cp = 0; return true;
            }
            CatcherFrame f = std::move(catcher_frames.back());
            catcher_frames.pop_back();
            cp = f.saved_cp;
            if (cp == 0) { halt = true; return true; }
            pc = cp;
            cp = 0;
            return true;
        }
        case Instruction::Op::NegationReturn: {
            // Reached when a \\+ or not protected goal proceeds normally
            // (i.e. the goal succeeded). Pop the frame, fully restore
            // pre-call VM state so any bindings / CPs the goal made
            // are undone, and return false — the negation FAILS.
            // backtrack() then resumes the outer flow.
            if (negation_frames.empty()) return false;
            NegationFrame f = std::move(negation_frames.back());
            negation_frames.pop_back();
            while (trail.size() > f.trail_mark) {
                TrailEntry t = std::move(trail.back());
                trail.pop_back();
                *t.cell = std::move(t.prev);
            }
            while (choice_points.size() > f.base_cp_count)
                choice_points.pop_back();
            while (aggregate_frames.size() > f.base_agg_count)
                aggregate_frames.pop_back();
            while (catcher_frames.size() > f.base_catcher_count)
                catcher_frames.pop_back();
            regs = std::move(f.saved_regs);
            mode_stack = std::move(f.saved_mode_stack);
            env_stack = std::move(f.saved_env_stack);
            cut_barrier = f.saved_cut_barrier;
            cp = f.saved_cp;
            // Negation fails; let backtrack do its thing.
            return false;
        }
        case Instruction::Op::FindallCollect: {
            // Reached when a findall/3 protected goal succeeds via
            // proceed back to findall_collect_pc. Snapshot the
            // template''s current value (deep copy so subsequent
            // trail unwinds don''t corrupt what we collected),
            // append to the top aggregate''s acc, then return false
            // to force backtrack for the next solution. The
            // existing aggregate-finalise in backtrack() takes
            // over when CPs drain to base_cp_count.
            if (aggregate_frames.empty()) return false;
            AggregateFrame& f = aggregate_frames.back();
            CellPtr value_cell = f.value_cell
                ? f.value_cell
                : get_cell(f.value_reg);
            if (!value_cell) return false;
            f.acc.push_back(deep_copy(*value_cell));
            // For bagof/setof: also snapshot the current witness
            // values (parallel to acc) so finalise can group by
            // witness binding.
            if (!f.witness_cells.empty()) {
                std::vector<Value> witness_snap;
                witness_snap.reserve(f.witness_cells.size());
                for (auto& wc : f.witness_cells) {
                    witness_snap.push_back(deep_copy(*wc));
                }
                f.acc_witnesses.push_back(std::move(witness_snap));
            }
            return false;
        }
        case Instruction::Op::ConjReturn: {
            // Reached when a conjunction goal-term''s G1 succeeded.
            // Dispatch G2 with the original after_pc. Do NOT pop the
            // frame — G1 may have left choice points that get retried
            // when G2 fails or when findall/3 forces backtrack for
            // more solutions; each G1 retry must re-dispatch THE SAME
            // G2. backtrack() pops the frame when G1''s CPs are
            // fully drained.
            if (conj_frames.empty()) return false;
            const ConjFrame& f = conj_frames.back();
            return invoke_goal_as_call(f.second_goal, f.after_pc);
        }
        case Instruction::Op::DisjAlt: {
            // Reached when the disjunction''s CP fired (G1 has
            // exhausted its solutions). Pop the CP (trust_me-style —
            // it''s a one-shot, no more alternatives after G2) and
            // the matching DisjFrame, then dispatch G2 with the
            // original after_pc.
            if (!choice_points.empty()) choice_points.pop_back();
            if (disj_frames.empty()) return false;
            DisjFrame f = std::move(disj_frames.back());
            disj_frames.pop_back();
            return invoke_goal_as_call(f.second_goal, f.after_pc);
        }
        case Instruction::Op::IfThenCommit: {
            // Reached when Cond succeeded. Cut: drop all CPs the
            // dispatch pushed (including our own paired CP). Pop
            // the IfThenFrame. Dispatch Then with the original
            // after_pc.
            if (if_then_frames.empty()) return false;
            IfThenFrame f = std::move(if_then_frames.back());
            if_then_frames.pop_back();
            while (choice_points.size() > f.base_cp_count)
                choice_points.pop_back();
            return invoke_goal_as_call(f.then_goal, f.after_pc);
        }
        case Instruction::Op::IfThenElse: {
            // Reached when Cond failed and backtrack landed at our
            // CP''s alt_pc. Pop the CP (trust_me-style — once-only,
            // same as DisjAlt) and the matching IfThenFrame, then
            // dispatch Else with the original after_pc. For bare
            // (Cond -> Then) goal-terms (no Else clause), else_goal
            // is null — propagate failure instead.
            if (!choice_points.empty()) choice_points.pop_back();
            if (if_then_frames.empty()) return false;
            IfThenFrame f = std::move(if_then_frames.back());
            if_then_frames.pop_back();
            if (!f.else_goal) return false;
            return invoke_goal_as_call(f.else_goal, f.after_pc);
        }
        case Instruction::Op::AggregateNextGroup: {
            // Reached via the CP pushed by aggregate-finalise (or
            // by aggregate_bind_next_group itself) when more
            // witness groups remain. Pop the CP and bind the next
            // group via the shared helper, which itself pushes a
            // CP if even MORE groups remain.
            if (!choice_points.empty()) choice_points.pop_back();
            return aggregate_bind_next_group();
        }
        case Instruction::Op::DynamicNextClause: {
            // Reached when a dynamic-predicate call backtracks into
            // its remaining clauses. Pop the CP and delegate to
            // dynamic_try_next, which unifies the next clause and
            // pushes another CP if more remain.
            if (!choice_points.empty()) choice_points.pop_back();
            return dynamic_try_next();
        }
        case Instruction::Op::SubAtomNext: {
            // Reached when sub_atom/5 backtracks for more matches.
            // Pop the CP and try the next (Before, Length) pair.
            if (!choice_points.empty()) choice_points.pop_back();
            return sub_atom_try_next();
        }
        case Instruction::Op::BodyNext: {
            // Reached after a dynamic-rule body goal succeeded
            // (either forward via cp=body_next_pc, or via backtrack
            // into a CP that landed at one of the body''s goals).
            // body_next dispatches the next goal in the BodyFrame''s
            // sequence, or pops the frame + proceeds to outer
            // after_pc when exhausted.
            return body_next();
        }
        case Instruction::Op::RetractNext: {
            // Reached when retract/1 backtracks for the next match.
            // Pop the CP and continue scanning from iter.next_idx
            // (which the previous retract_try_next advanced past
            // the just-removed clause).
            if (!choice_points.empty()) choice_points.pop_back();
            return retract_try_next();
        }
        case Instruction::Op::Proceed: {
            if (cp == 0) { halt = true; return true; }
            pc = cp;
            cp = 0;
            return true;
        }
        case Instruction::Op::Fail:
            return false;
        case Instruction::Op::Jump:
            pc = instr.target;
            return true;

        // ---- Choice points -----------------------------------------
        case Instruction::Op::TryMeElse: {
            ChoicePoint cp_;
            cp_.alt_pc = instr.target;
            cp_.saved_cp = cp;
            cp_.trail_mark = trail.size();
            cp_.cut_barrier = cut_barrier;
            cp_.saved_regs = regs;
            cp_.saved_mode_stack = mode_stack;
            cp_.saved_env_stack = env_stack;
            cp_.saved_body_frames = body_frames;
            choice_points.push_back(std::move(cp_));
            pc += 1; return true;
        }
        case Instruction::Op::RetryMeElse: {
            // No-op if no CP exists (e.g. when an indexing instruction
            // jumped past the originating try_me_else). Classic WAM:
            // retry_me_else updates the top CP''s alt; if there is no
            // top CP we simply continue with no alt to manage.
            if (!choice_points.empty()) {
                choice_points.back().alt_pc = instr.target;
            }
            pc += 1; return true;
        }
        case Instruction::Op::TrustMe: {
            if (!choice_points.empty()) choice_points.pop_back();
            pc += 1; return true;
        }
        case Instruction::Op::CutIte: {
            // Soft cut for inlined if-then-else. The topmost CP at this
            // point is the matching try_me_else (Cond just succeeded;
            // anything Cond pushed sits above it and also wants to die
            // on cut). Its saved cut_barrier is the OUTER barrier from
            // before try_me_else fired — restoring it means subsequent
            // !/0 in the Then branch still respects the enclosing scope.
            // Crucially, CPs BELOW the try_me_else (e.g. an aggregate
            // body''s generator CP) are preserved, so `findall(X, (gen,
            // (cond -> then ; else)), L)` keeps iterating gen on
            // backtrack — the previous behaviour of `resize(cut_barrier)`
            // would have dropped gen''s CP too.
            if (!choice_points.empty()) {
                std::size_t top_idx = choice_points.size() - 1;
                std::size_t saved_barrier =
                    choice_points[top_idx].cut_barrier;
                choice_points.resize(top_idx);
                cut_barrier = saved_barrier;
            }
            pc += 1; return true;
        }

        // ---- Builtins ----------------------------------------------
        case Instruction::Op::BuiltinCall: {
            // If the op has a registered label (e.g. an auto-injected
            // helper like member/2 or length/2), dispatch as a Call so
            // backtracking through clauses works naturally.
            auto it = labels.find(instr.a);
            if (it != labels.end()) {
                cp = pc + 1;
                pc = it->second;
                return true;
            }
            return builtin(instr.a, instr.n);
        }
        case Instruction::Op::CallForeign:
            pc += 1; return true;

        // ---- Aggregate / findall driver ----------------------------
        case Instruction::Op::BeginAggregate: {
            AggregateFrame frame;
            frame.agg_kind          = instr.a;
            frame.value_reg         = instr.b;
            frame.result_reg        = instr.val.s; // stuffed into val.s by factory
            frame.begin_pc          = pc;
            frame.return_pc         = 0;
            frame.return_pc_set     = false;
            frame.base_cp_count     = choice_points.size();
            frame.trail_mark        = trail.size();
            frame.saved_cp          = cp;
            frame.saved_cut_barrier = cut_barrier;
            frame.saved_regs        = regs;
            frame.saved_mode_stack  = mode_stack;
            frame.saved_env_stack   = env_stack;
            // For bagof/setof with witnesses: instr.c is a semicolon-
            // delimited list of register names ("Y2" / "Y1;Y2"). Store
            // the names here — actual cell resolution happens lazily at
            // EndAggregate, since Y-regs for free witnesses get
            // allocated by put_variable INSIDE the aggregate body.
            if (!instr.c.empty()) {
                std::string buf;
                for (char ch : instr.c) {
                    if (ch == '';'') {
                        if (!buf.empty()) frame.witness_regs.push_back(buf);
                        buf.clear();
                    } else {
                        buf.push_back(ch);
                    }
                }
                if (!buf.empty()) frame.witness_regs.push_back(buf);
            }
            aggregate_frames.push_back(std::move(frame));
            pc += 1;
            return true;
        }
        case Instruction::Op::EndAggregate: {
            if (aggregate_frames.empty()) return false;
            AggregateFrame& f = aggregate_frames.back();
            // Snapshot the current value of value_reg (deep copy so later
            // trail-driven mutations don''t alter what we collected).
            CellPtr v = get_cell(instr.a);
            f.acc.push_back(deep_copy(*v));
            // For bagof/setof on the inlined path: resolve witness regs
            // lazily on the first iteration (they get allocated inside
            // the aggregate body, after BeginAggregate fired). The env
            // frame keeps the same Y-reg shared_ptrs across body retries,
            // so a single resolution is reusable for all later snapshots.
            if (f.witness_cells.empty() && !f.witness_regs.empty()) {
                for (auto& rname : f.witness_regs) {
                    CellPtr wc = get_cell(rname);
                    if (wc) f.witness_cells.push_back(wc);
                }
            }
            // Snapshot witness values parallel to acc so finalise can
            // partition by witness equality. Mirrors the FindallCollect
            // arm used by the meta-call path.
            if (!f.witness_cells.empty()) {
                std::vector<Value> witness_snap;
                witness_snap.reserve(f.witness_cells.size());
                for (auto& wc : f.witness_cells) {
                    witness_snap.push_back(deep_copy(*wc));
                }
                f.acc_witnesses.push_back(std::move(witness_snap));
            }
            f.return_pc     = pc + 1;
            f.return_pc_set = true;
            // Force backtrack to find the next solution of the body.
            return false;
        }

        // ---- Indexing ----------------------------------------------
        case Instruction::Op::SwitchOnConstant: {
            // Dispatch on A1''s atom/integer value.
            CellPtr ac = get_cell("A1");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag != Value::Tag::Atom && a.tag != Value::Tag::Integer
                && a.tag != Value::Tag::Float) {
                pc += 1; return true; // not a constant we index on
            }
            for (auto& kv : instr.const_table) {
                if (kv.first == a) {
                    if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (kv.second == Instruction::SWITCH_NONE)    return false;
                    pc = kv.second; return true;
                }
            }
            return false; // bound constant with no matching clause
        }
        case Instruction::Op::SwitchOnStructure: {
            CellPtr ac = get_cell("A1");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag != Value::Tag::Compound) { pc += 1; return true; }
            for (auto& kv : instr.struct_table) {
                if (kv.first == a.s) {
                    if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (kv.second == Instruction::SWITCH_NONE)    return false;
                    pc = kv.second; return true;
                }
            }
            return false;
        }
        case Instruction::Op::SwitchOnTerm: {
            CellPtr ac = get_cell("A1");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            // Atom / integer / float → const_table.
            if (a.tag == Value::Tag::Atom || a.tag == Value::Tag::Integer
                || a.tag == Value::Tag::Float) {
                for (auto& kv : instr.const_table) {
                    if (kv.first == a) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        pc = kv.second; return true;
                    }
                }
                return false;
            }
            // Compound → list_pc for cons cells, otherwise struct_table.
            if (a.tag == Value::Tag::Compound) {
                if (a.s == "[|]/2") {
                    if (instr.target == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (instr.target == Instruction::SWITCH_NONE)    return false;
                    pc = instr.target; return true;
                }
                for (auto& kv : instr.struct_table) {
                    if (kv.first == a.s) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        pc = kv.second; return true;
                    }
                }
                return false;
            }
            pc += 1; return true;
        }
    }
    return false;
}

// (deep_copy is defined above, before step().)

// Build the finalisation result for an aggregate based on its kind.
// Falls back to the raw list when the kind isn''t recognised.
// Standard-order-of-terms comparison for setof''s sort + dedup.
// Sorts: Var < Number < Atom < Compound, with compounds compared by
// (functor-string, arity, args lexicographically). The functor
// string already encodes "Name/Arity" so comparing it covers both
// name and arity at once when they differ. Args are deref''d so
// indirect bindings come through correctly.
// ISO §7.2 standard order of terms — returns -1 / 0 / +1.
//   Variable @< Number @< Atom @< Compound
//   Variables: by internal cell-name lex order (stable but
//              implementation-defined per ISO).
//   Numbers: by value. Equal-value integer @< float (ISO tie-break).
//   Atoms: lex (codepoint).
//   Compounds: arity first, then functor name lex, then args lex.
//
// Used by @</2, @=</2, @>/2, @>=/2, compare/3. Distinct from
// term_less below, which compares the "Name/Arity" string as a
// whole (good enough for setof''s internal sort, but not strictly
// ISO since "foo/10" comes BEFORE "foo/2" lexicographically).
static int standard_order_cmp(const Value& a, const Value& b) {
    auto category = [](const Value& v) -> int {
        if (v.is_unbound()) return 0;       // variable
        switch (v.tag) {
            case Value::Tag::Integer:
            case Value::Tag::Float:    return 1; // number
            case Value::Tag::Atom:     return 2; // atom
            case Value::Tag::Compound: return 3; // compound
            default:                   return 4;
        }
    };
    int ca = category(a), cb = category(b);
    if (ca != cb) return ca < cb ? -1 : 1;
    if (ca == 0) {
        // Both variables.
        if (a.s < b.s) return -1;
        if (a.s > b.s) return 1;
        return 0;
    }
    if (ca == 1) {
        double av = (a.tag == Value::Tag::Integer)
                    ? static_cast<double>(a.i) : a.f;
        double bv = (b.tag == Value::Tag::Integer)
                    ? static_cast<double>(b.i) : b.f;
        if (av < bv) return -1;
        if (av > bv) return 1;
        // Equal value — int @< float per ISO tie-break.
        if (a.tag == Value::Tag::Integer && b.tag == Value::Tag::Float)
            return -1;
        if (a.tag == Value::Tag::Float && b.tag == Value::Tag::Integer)
            return 1;
        return 0;
    }
    if (ca == 2) {
        if (a.s < b.s) return -1;
        if (a.s > b.s) return 1;
        return 0;
    }
    // Compound: arity, then name, then args.
    if (a.args.size() != b.args.size())
        return a.args.size() < b.args.size() ? -1 : 1;
    auto name_of = [](const std::string& s) -> std::string {
        auto slash = s.rfind(''/'');
        return slash == std::string::npos ? s : s.substr(0, slash);
    };
    std::string an = name_of(a.s);
    std::string bn = name_of(b.s);
    if (an < bn) return -1;
    if (an > bn) return 1;
    for (std::size_t i = 0; i < a.args.size(); ++i) {
        if (!a.args[i] || !b.args[i]) continue;
        int c = standard_order_cmp(*a.args[i], *b.args[i]);
        if (c != 0) return c;
    }
    return 0;
}

static bool term_less(const Value& a, const Value& b) {
    if (a.tag != b.tag)
        return static_cast<int>(a.tag) < static_cast<int>(b.tag);
    switch (a.tag) {
        case Value::Tag::Atom:
        case Value::Tag::Unbound:
            return a.s < b.s;
        case Value::Tag::Integer:
            return a.i < b.i;
        case Value::Tag::Float:
            return a.f < b.f;
        case Value::Tag::Compound: {
            if (a.s != b.s) return a.s < b.s;
            // Same functor/arity — compare args lexicographically.
            for (std::size_t i = 0;
                 i < a.args.size() && i < b.args.size(); ++i) {
                if (!a.args[i] || !b.args[i]) continue;
                if (term_less(*a.args[i], *b.args[i])) return true;
                if (term_less(*b.args[i], *a.args[i])) return false;
            }
            return a.args.size() < b.args.size();
        }
        default:
            return false;
    }
}

static Value finalize_aggregate(const std::string& kind, const std::vector<Value>& acc) {
    auto as_d = [](const Value& v) { return v.tag == Value::Tag::Float ? v.f : (double)v.i; };
    auto is_num = [](const Value& v) {
        return v.tag == Value::Tag::Integer || v.tag == Value::Tag::Float;
    };
    if (kind == "count") {
        return Value::Integer((std::int64_t)acc.size());
    }
    if (kind == "sum") {
        bool has_float = false;
        double sum_d = 0.0; std::int64_t sum_i = 0;
        for (auto& v : acc) {
            if (!is_num(v)) return Value{};
            if (v.tag == Value::Tag::Float) { has_float = true; sum_d += v.f; }
            else                            { sum_i += v.i; sum_d += (double)v.i; }
        }
        if (has_float) return Value::Float(sum_d);
        return Value::Integer(sum_i);
    }
    if (kind == "min" || kind == "max") {
        if (acc.empty()) return Value::Atom("[]");
        const Value* best = &acc[0];
        for (std::size_t k = 1; k < acc.size(); ++k) {
            if (!is_num(acc[k])) return Value{};
            if (kind == "min" ? (as_d(acc[k]) < as_d(*best))
                              : (as_d(acc[k]) > as_d(*best))) {
                best = &acc[k];
            }
        }
        return *best;
    }
    // "collect" (findall), bagof/setof, and aggregate_all bag/set all
    // build a list. bagof/setof fail on empty; findall and
    // aggregate_all(bag/set) keep the historical empty-list behaviour.
    if ((kind == "bagof" || kind == "setof") && acc.empty()) {
        return Value{};
    }
    std::vector<Value> items = acc;
    if (kind == "set" || kind == "setof") {
        std::vector<Value> uniq;
        for (auto& v : items) {
            bool dup = false;
            for (auto& w : uniq) { if (v == w) { dup = true; break; } }
            if (!dup) uniq.push_back(v);
        }
        items = std::move(uniq);
        std::sort(items.begin(), items.end(), term_less);
    }
    Value tail = Value::Atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        std::vector<CellPtr> args;
        args.push_back(std::make_shared<Cell>(*it));
        args.push_back(std::make_shared<Cell>(std::move(tail)));
        tail = Value::Compound("[|]/2", std::move(args));
    }
    return tail;
}

// Walk forward from begin_pc to find the matching EndAggregate, counting
// nested Begin/End pairs. Used to compute pc continuation when the body
// of an aggregate fails before any EndAggregate fires.
static std::size_t find_matching_end_aggregate(
        const std::vector<Instruction>& instrs, std::size_t begin_pc) {
    int depth = 1;
    for (std::size_t k = begin_pc + 1; k < instrs.size(); ++k) {
        if (instrs[k].op == Instruction::Op::BeginAggregate) ++depth;
        else if (instrs[k].op == Instruction::Op::EndAggregate) {
            if (--depth == 0) return k;
        }
    }
    return instrs.size();
}

// Dispatch a Call/Execute to a dynamic predicate (one populated by
// assertz/asserta and not present in labels). Snapshots the A-args
// from regs, pushes a DynamicIterator, then delegates to
// dynamic_try_next which unifies the first clause and pushes a CP
// for the rest when more than one exists.
bool WamState::dispatch_dynamic_call(const std::string& key,
                                     std::size_t after_pc) {
    auto db_it = dynamic_db.find(key);
    if (db_it == dynamic_db.end() || db_it->second.empty()) return false;
    auto sl = key.rfind(''/'');
    if (sl == std::string::npos) return false;
    std::size_t arity = 0;
    try { arity = std::stoul(key.substr(sl + 1)); }
    catch (...) { return false; }
    // Snapshot A-registers BEFORE pushing the iterator so backtrack
    // (which restores regs from the CP) gets the same call_args.
    DynamicIterator dit;
    dit.key = key;
    dit.next_idx = 0;
    dit.after_pc = after_pc;
    dit.call_args.reserve(arity);
    for (std::size_t i = 1; i <= arity; ++i) {
        dit.call_args.push_back(get_cell("A" + std::to_string(i)));
    }
    dynamic_iters.push_back(std::move(dit));
    return dynamic_try_next();
}

// Try the next clause in the top DynamicIterator. Unify the call
// args with a fresh-renamed copy of the clause; if more clauses
// remain, push a CP whose alt_pc = dynamic_next_clause_pc. Pop the
// iterator and fail if exhausted.
bool WamState::dynamic_try_next() {
    if (dynamic_iters.empty()) return false;
    DynamicIterator& it = dynamic_iters.back();
    auto db_it = dynamic_db.find(it.key);
    // The predicate may have been retract-all''d mid-iteration. Treat
    // a missing or empty clause list as exhaustion.
    if (db_it == dynamic_db.end() || it.next_idx >= db_it->second.size()) {
        dynamic_iters.pop_back();
        return false;
    }
    std::size_t idx = it.next_idx;
    bool has_more = (idx + 1 < db_it->second.size());
    std::string saved_key = it.key;
    std::vector<CellPtr> call_args = it.call_args;
    std::size_t saved_after_pc = it.after_pc;
    // Advance the index BEFORE we push the CP — the CP snapshots
    // dynamic_iters state, so the captured `next_idx` should already
    // point at the SUBSEQUENT clause to try on backtrack.
    it.next_idx = idx + 1;
    // Push the CP (only if more clauses remain after this one).
    if (has_more) {
        ChoicePoint cp_;
        cp_.alt_pc            = dynamic_next_clause_pc;
        cp_.saved_cp          = cp;
        cp_.trail_mark        = trail.size();
        cp_.cut_barrier       = cut_barrier;
        cp_.saved_regs        = regs;
        cp_.saved_mode_stack  = mode_stack;
        cp_.saved_env_stack   = env_stack;
        cp_.saved_body_frames = body_frames;
        choice_points.push_back(std::move(cp_));
    } else {
        // Last clause for this call — drop the iterator. It''s no
        // longer needed; clause exhaustion will fall through to
        // ordinary backtrack at the OUTER scope.
        dynamic_iters.pop_back();
    }
    // Build a rename seed for fresh-copy: when a top-level head var is
    // unbound AND the matching call arg is also unbound, map the
    // stored var''s name to the caller''s cell. This makes the head
    // arg (and any body occurrences of the same var) SHARE the
    // caller''s cell — essential because our cell model lacks a ref-
    // chain for unbound-unbound unification, so a plain unify_cells
    // would leave the two unbound cells unaliased.
    CellPtr stored = db_it->second[idx];
    Value stored_v = deref(*stored);
    CellPtr stored_head = stored;
    if (stored_v.tag == Value::Tag::Compound && stored_v.s == ":-/2"
        && stored_v.args.size() == 2) {
        stored_head = stored_v.args[0];
    }
    std::unordered_map<std::string, CellPtr> seed;
    Value sh = deref(*stored_head);
    if (sh.tag == Value::Tag::Compound
        && sh.args.size() == call_args.size())
    {
        for (std::size_t i = 0; i < call_args.size(); ++i) {
            Value harg = deref(*sh.args[i]);
            Value carg = deref(*call_args[i]);
            if (harg.is_unbound() && carg.is_unbound()) {
                seed.emplace(harg.s, call_args[i]);
            }
        }
    }
    CellPtr fresh = deep_copy_term_seeded(stored, std::move(seed));
    Value fv = deref(*fresh);
    // Decompose: clauses are either bare heads (facts) or
    // ":-/2"(Head, Body) compounds (rules). Body defaults to a "true"
    // atom for facts so the common dispatch path is the same.
    CellPtr head_cell;
    CellPtr body_cell;
    if (fv.tag == Value::Tag::Compound && fv.s == ":-/2"
        && fv.args.size() == 2) {
        head_cell = fv.args[0];
        body_cell = fv.args[1];
    } else {
        head_cell = fresh;
        body_cell = std::make_shared<Cell>(Value::Atom("true"));
    }
    Value hv = deref(*head_cell);
    // Unify the call args with the head''s args.
    if (hv.tag == Value::Tag::Atom) {
        // 0-arity head — no args to bind. call_args must be empty;
        // the key-match already covered name + arity.
        if (!call_args.empty()) return false;
    } else if (hv.tag == Value::Tag::Compound) {
        if (hv.args.size() != call_args.size()) return false;
        for (std::size_t i = 0; i < call_args.size(); ++i) {
            if (!unify_cells(call_args[i], hv.args[i])) return false;
        }
    } else {
        return false;
    }
    // Body == "true" → fact match completes; jump to continuation.
    Value bv = deref(*body_cell);
    if (bv.tag == Value::Tag::Atom && bv.s == "true") {
        if (saved_after_pc == 0) { halt = true; return true; }
        pc = saved_after_pc;
        cp = 0;
        return true;
    }
    // Otherwise build a BodyFrame from the flattened conjunction body
    // and dispatch sequentially. The flattening avoids ConjFrame
    // nesting on the shared conj_return_pc — see BodyFrame doc.
    std::vector<CellPtr> goals;
    std::function<void(CellPtr)> flatten = [&](CellPtr g) {
        Value gv = deref(*g);
        if (gv.tag == Value::Tag::Compound && gv.s == ",/2"
            && gv.args.size() == 2)
        {
            flatten(gv.args[0]);
            flatten(gv.args[1]);
        } else {
            goals.push_back(g);
        }
    };
    flatten(body_cell);
    BodyFrame bf;
    bf.goals = std::move(goals);
    bf.next_idx = 0;
    bf.after_pc = saved_after_pc;
    bf.base_cp_count = choice_points.size();
    body_frames.push_back(std::move(bf));
    return body_next();
}

// retract/1 dispatcher. Reads A1 (the pattern), derives the
// predicate''s "name/arity" key, pushes a RetractIterator, and
// delegates to retract_try_next to find the first matching clause.
bool WamState::dispatch_retract(std::size_t after_pc) {
    CellPtr pat = get_cell("A1");
    Value pv = deref(*pat);
    if (pv.is_unbound()) return false;
    std::string key;
    if (pv.tag == Value::Tag::Compound && pv.s == ":-/2"
        && pv.args.size() == 2)
    {
        Value head = deref(*pv.args[0]);
        if (head.tag == Value::Tag::Atom) key = head.s + "/0";
        else if (head.tag == Value::Tag::Compound) key = head.s;
        else return false;
    } else if (pv.tag == Value::Tag::Atom) {
        key = pv.s + "/0";
    } else if (pv.tag == Value::Tag::Compound) {
        key = pv.s;
    } else {
        return false;
    }
    RetractIterator it;
    it.key = std::move(key);
    it.pattern = pat;
    it.next_idx = 0;
    it.after_pc = after_pc;
    retract_iters.push_back(std::move(it));
    return retract_try_next();
}

// Scan dynamic_db[key] from iter.next_idx onward for a clause that
// unifies with iter.pattern. On match: remove that clause, leave
// bindings in place (per ISO), push a CP if any clauses remain
// after the removal, and proceed to after_pc. On exhaustion: pop
// the iterator and fail.
bool WamState::retract_try_next() {
    if (retract_iters.empty()) return false;
    RetractIterator& it = retract_iters.back();
    auto db_it = dynamic_db.find(it.key);
    if (db_it == dynamic_db.end()) {
        retract_iters.pop_back();
        return false;
    }
    auto& vec = db_it->second;
    std::size_t i = it.next_idx;
    while (i < vec.size()) {
        std::size_t mark = trail.size();
        CellPtr fresh = deep_copy_term(vec[i]);
        if (unify_cells(it.pattern, fresh)) {
            // Match. We need to:
            // 1) push a CP (if more candidates exist) whose trail
            //    mark is BEFORE the unification, so backtrack undoes
            //    the pattern''s bindings and the pattern is reusable;
            // 2) remove the matched clause from vec;
            // 3) leave the unification bindings in place (per ISO
            //    retract) and proceed to after_pc.
            // The CP''s saved regs/etc are the current values — at
            // entry to retract_try_next, before any binding — which
            // match what the caller of dispatch_retract set up.
            vec.erase(vec.begin() + i);
            it.next_idx = i;
            bool has_more = (i < vec.size());
            std::size_t saved_after_pc = it.after_pc;
            if (has_more) {
                ChoicePoint cp_;
                cp_.alt_pc            = retract_next_pc;
                cp_.saved_cp          = cp;
                cp_.trail_mark        = mark; // PRE-unify trail mark
                cp_.cut_barrier       = cut_barrier;
                cp_.saved_regs        = regs;
                cp_.saved_mode_stack  = mode_stack;
                cp_.saved_env_stack   = env_stack;
                cp_.saved_body_frames = body_frames;
                choice_points.push_back(std::move(cp_));
            } else {
                retract_iters.pop_back();
            }
            if (saved_after_pc == 0) { halt = true; return true; }
            pc = saved_after_pc;
            cp = 0;
            return true;
        }
        // No match at i — undo any partial bindings and try i+1.
        while (trail.size() > mark) {
            TrailEntry te = std::move(trail.back());
            trail.pop_back();
            *te.cell = std::move(te.prev);
        }
        ++i;
    }
    // Exhausted — drop the iterator and fail.
    retract_iters.pop_back();
    return false;
}

// Dispatch the top BodyFrame''s next goal (with cp=body_next_pc so
// subsequent BodyNext fires advance through the sequence), or pop
// the frame and proceed to outer after_pc when exhausted. Forward
// pop is safe because ChoicePoints snapshot body_frames at push
// time; if a CP from inside this body fires later, it restores
// body_frames including this frame, and dispatch resumes correctly.
bool WamState::body_next() {
    if (body_frames.empty()) return false;
    BodyFrame& f = body_frames.back();
    if (f.next_idx >= f.goals.size()) {
        std::size_t after = f.after_pc;
        body_frames.pop_back();
        if (after == 0) { halt = true; return true; }
        pc = after;
        cp = 0;
        return true;
    }
    CellPtr g = f.goals[f.next_idx++];
    return invoke_goal_as_call(g, body_next_pc);
}

// sub_atom/5 dispatcher. Reads A1..A5, decides which dimensions are
// bound, enumerates valid (Before, Length) tuples that satisfy the
// constraints, then iterates them via the SubAtomIterator + CP
// pattern. Atom (A1) MUST be bound. Other args may be unbound — the
// enumerator restricts the candidate set accordingly.
bool WamState::dispatch_sub_atom(std::size_t after_pc) {
    CellPtr a1 = get_cell("A1");
    CellPtr a2 = get_cell("A2");
    CellPtr a3 = get_cell("A3");
    CellPtr a4 = get_cell("A4");
    CellPtr a5 = get_cell("A5");
    Value av = deref(*a1);
    // Atom must be bound. Accept atoms; integers/floats are rendered
    // so sub_atom(42, B, L, A, S) treats "42" as the source.
    std::string s;
    if (av.tag == Value::Tag::Atom) s = av.s;
    else if (av.tag == Value::Tag::Integer) s = std::to_string(av.i);
    else if (av.tag == Value::Tag::Float) s = render(av);
    else return false;
    const std::size_t N = s.size();
    // Read bound integer constraints (Before/Length/After).
    auto read_int = [&](CellPtr c, bool& bound, std::size_t& out) -> bool {
        Value v = deref(*c);
        if (v.is_unbound()) { bound = false; return true; }
        if (v.tag != Value::Tag::Integer || v.i < 0) return false;
        bound = true;
        out = static_cast<std::size_t>(v.i);
        return true;
    };
    bool b_bound = false, l_bound = false, a_bound = false;
    std::size_t b_val = 0, l_val = 0, a_val = 0;
    if (!read_int(a2, b_bound, b_val)) return false;
    if (!read_int(a3, l_bound, l_val)) return false;
    if (!read_int(a4, a_bound, a_val)) return false;
    // Read Sub if bound. Accept atom; integer rendered like A1.
    bool sub_bound = false;
    std::string sub_str;
    Value sv = deref(*a5);
    if (!sv.is_unbound()) {
        if (sv.tag == Value::Tag::Atom) sub_str = sv.s;
        else if (sv.tag == Value::Tag::Integer) sub_str = std::to_string(sv.i);
        else if (sv.tag == Value::Tag::Float) sub_str = render(sv);
        else return false;
        sub_bound = true;
    }
    // Sanity: bound constraints must satisfy B + L + A == N.
    if (b_bound && l_bound && a_bound) {
        if (b_val + l_val + a_val != N) return false;
    }
    // Sub-bound determines length immediately.
    if (sub_bound) {
        if (l_bound && l_val != sub_str.size()) return false;
        l_bound = true;
        l_val = sub_str.size();
    }
    // Enumerate candidate (B, L) pairs. After is computed; we just
    // need to satisfy the bound subset of (B, L, A) constraints.
    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    auto b_range = [&]() {
        return b_bound ? std::pair<std::size_t, std::size_t>{b_val, b_val}
                       : std::pair<std::size_t, std::size_t>{0, N};
    };
    auto l_range = [&]() {
        return l_bound ? std::pair<std::size_t, std::size_t>{l_val, l_val}
                       : std::pair<std::size_t, std::size_t>{0, N};
    };
    auto [b_lo, b_hi] = b_range();
    auto [l_lo, l_hi] = l_range();
    for (std::size_t b = b_lo; b <= b_hi; ++b) {
        if (b > N) break;
        for (std::size_t l = l_lo; l <= l_hi; ++l) {
            if (b + l > N) break;
            std::size_t a = N - b - l;
            if (a_bound && a != a_val) continue;
            if (sub_bound) {
                // Compare s[b..b+l] with sub_str.
                if (s.compare(b, l, sub_str) != 0) continue;
            }
            candidates.emplace_back(b, l);
        }
    }
    if (candidates.empty()) return false;
    SubAtomIterator it;
    it.atom_str = std::move(s);
    it.before_cell = a2;
    it.length_cell = a3;
    it.after_cell = a4;
    it.sub_cell = a5;
    it.after_pc = after_pc;
    it.remaining_pairs = std::move(candidates);
    sub_atom_iters.push_back(std::move(it));
    return sub_atom_try_next();
}

// Pop one (Before, Length) candidate from the top SubAtomIterator,
// unify all four output cells (Before/Length/After/Sub), and push
// a CP for further matches when any remain.
bool WamState::sub_atom_try_next() {
    if (sub_atom_iters.empty()) return false;
    SubAtomIterator& it = sub_atom_iters.back();
    if (it.remaining_pairs.empty()) {
        sub_atom_iters.pop_back();
        return false;
    }
    auto [b, l] = it.remaining_pairs.front();
    it.remaining_pairs.erase(it.remaining_pairs.begin());
    bool has_more = !it.remaining_pairs.empty();
    // Snapshot before pushing CP, since the CP captures regs and
    // we want the saved regs to reflect pre-binding state.
    std::size_t saved_after_pc = it.after_pc;
    CellPtr bc = it.before_cell;
    CellPtr lc = it.length_cell;
    CellPtr ac = it.after_cell;
    CellPtr sc = it.sub_cell;
    std::size_t N = it.atom_str.size();
    std::string sub = it.atom_str.substr(b, l);
    // Pop the iterator before pushing the CP / binding when this is
    // the last candidate — otherwise the CP would still see a stale
    // entry on top.
    if (!has_more) {
        sub_atom_iters.pop_back();
    } else {
        ChoicePoint cp_;
        cp_.alt_pc            = sub_atom_next_pc;
        cp_.saved_cp          = cp;
        cp_.trail_mark        = trail.size();
        cp_.cut_barrier       = cut_barrier;
        cp_.saved_regs        = regs;
        cp_.saved_mode_stack  = mode_stack;
        cp_.saved_env_stack   = env_stack;
        cp_.saved_body_frames = body_frames;
        choice_points.push_back(std::move(cp_));
    }
    // Unify each output cell with its computed value.
    auto bind_or_unify_int = [&](CellPtr c, std::size_t v) -> bool {
        Value iv = Value::Integer(static_cast<std::int64_t>(v));
        if (c->is_unbound()) { bind_cell(c, iv); return true; }
        return unify_cells(c, std::make_shared<Cell>(iv));
    };
    if (!bind_or_unify_int(bc, b)) return false;
    if (!bind_or_unify_int(lc, l)) return false;
    if (!bind_or_unify_int(ac, N - b - l)) return false;
    Value sv = Value::Atom(sub);
    if (sc->is_unbound()) bind_cell(sc, sv);
    else if (!unify_cells(sc, std::make_shared<Cell>(sv))) return false;
    if (saved_after_pc == 0) { halt = true; return true; }
    pc = saved_after_pc;
    cp = 0;
    return true;
}

// Deep-copy a value tree. Unbound leaves are renamed via a name→cell
// map so multiple occurrences in the source share a single fresh cell
// in the copy. Used by throw/1 to snapshot the thrown term before
// state-unwind tears down the goal''s bindings.
CellPtr WamState::deep_copy_term(CellPtr src) {
    return deep_copy_term_seeded(src, {});
}

CellPtr WamState::deep_copy_term_seeded(
    CellPtr src,
    std::unordered_map<std::string, CellPtr> seed)
{
    std::unordered_map<std::string, CellPtr> rename = std::move(seed);
    std::function<CellPtr(CellPtr)> rec = [&](CellPtr c) -> CellPtr {
        Value v = deref(*c);
        if (v.tag == Value::Tag::Unbound || v.tag == Value::Tag::Uninit) {
            const std::string& name = v.s;
            auto it = rename.find(name);
            if (it != rename.end()) return it->second;
            CellPtr fresh = std::make_shared<Cell>(
                Value::Unbound("_T" + std::to_string(var_counter++)));
            rename[name] = fresh;
            return fresh;
        }
        if (v.tag == Value::Tag::Compound) {
            std::vector<CellPtr> args;
            for (auto& a : v.args) args.push_back(rec(a));
            return std::make_shared<Cell>(
                Value::Compound(v.s, std::move(args)));
        }
        return std::make_shared<Cell>(v);
    };
    return rec(src);
}

// Treat goal_cell as a Prolog goal-term and dispatch to it as a Call:
// sets A-registers from the goal''s args, looks up "<name>/<arity>" in
// labels, and arranges for the goal to proceed to after_pc when done.
// Atoms are treated as 0-arity calls. Returns false if the goal''s
// functor isn''t a known label.
bool WamState::invoke_goal_as_call(CellPtr goal_cell, std::size_t after_pc) {
    Value g = deref(*goal_cell);
    if (g.tag == Value::Tag::Atom) {
        // Special-case the trivial control atoms so a recovery goal of
        // "true" or "fail" works without needing them registered as
        // 0-arity predicates.
        if (g.s == "true") {
            cp = after_pc;
            if (cp == 0) { halt = true; return true; }
            pc = cp; cp = 0;
            return true;
        }
        if (g.s == "fail" || g.s == "false") {
            return false;
        }
        std::string key = g.s + "/0";
        auto it = labels.find(key);
        if (it != labels.end()) {
            cp = after_pc;
            pc = it->second;
            return true;
        }
        // 0-arity dynamic predicate fallback.
        if (dynamic_db.count(key)) {
            return dispatch_dynamic_call(key, after_pc);
        }
        return false;
    }
    if (g.tag == Value::Tag::Compound) {
        // s is "<name>/<arity>".
        const std::string& key = g.s;
        // Conjunction goal-term (G1, G2): dispatched as a meta-call
        // argument (e.g. inside catch(_,_,_), findall/3, \\+/1).
        // Push a ConjFrame remembering G2 and the original after_pc;
        // dispatch G1 with cp = conj_return_pc. When G1 succeeds,
        // ConjReturn pops the frame and dispatches G2 with after_pc.
        if (key == ",/2" && g.args.size() == 2) {
            ConjFrame f;
            f.second_goal   = g.args[1];
            f.after_pc      = after_pc;
            f.base_cp_count = choice_points.size();
            conj_frames.push_back(std::move(f));
            return invoke_goal_as_call(g.args[0], conj_return_pc);
        }
        // Bare if-then `(Cond -> Then)` as a goal-term (no ;
        // wrapper, so there''s no Else branch). Same machinery as
        // if-then-else but with else_goal = nullptr — Cond failure
        // propagates as failure of the whole construct.
        if (key == "->/2" && g.args.size() == 2) {
            IfThenFrame itf;
            itf.then_goal     = g.args[1];
            // else_goal stays null — IfThenElse propagates failure
            // when it sees a null else slot.
            itf.after_pc      = after_pc;
            itf.base_cp_count = choice_points.size();
            if_then_frames.push_back(std::move(itf));
            ChoicePoint cp_;
            cp_.alt_pc            = if_then_else_pc;
            cp_.saved_cp          = cp;
            cp_.trail_mark        = trail.size();
            cp_.cut_barrier       = cut_barrier;
            cp_.saved_regs        = regs;
            cp_.saved_mode_stack  = mode_stack;
            cp_.saved_env_stack   = env_stack;
            cp_.saved_body_frames = body_frames;
            choice_points.push_back(std::move(cp_));
            return invoke_goal_as_call(g.args[0], if_then_commit_pc);
        }
        // Disjunction goal-term — two flavours, both shaped ;/2.
        // First check for the if-then-else special case
        // `;(->( Cond, Then), Else)`: built by the WAM compiler
        // when the user writes `(Cond -> Then ; Else)` as data
        // (passed to catch/3, call/1, etc.). Cut semantics differ
        // from plain disjunction, so we route to a separate frame.
        if (key == ";/2" && g.args.size() == 2) {
            // Peek the first arg — is it ->/2 ?
            Value first = deref(*g.args[0]);
            if (first.tag == Value::Tag::Compound
                && first.s == "->/2"
                && first.args.size() == 2)
            {
                IfThenFrame itf;
                itf.then_goal     = first.args[1];
                itf.else_goal     = g.args[1];
                itf.after_pc      = after_pc;
                itf.base_cp_count = choice_points.size();
                if_then_frames.push_back(std::move(itf));
                // CP whose alt_pc dispatches Else on Cond failure.
                ChoicePoint cp_;
                cp_.alt_pc            = if_then_else_pc;
                cp_.saved_cp          = cp;
                cp_.trail_mark        = trail.size();
                cp_.cut_barrier       = cut_barrier;
                cp_.saved_regs        = regs;
                cp_.saved_mode_stack  = mode_stack;
                cp_.saved_env_stack   = env_stack;
                cp_.saved_body_frames = body_frames;
                choice_points.push_back(std::move(cp_));
                // Dispatch Cond with cp = if_then_commit_pc so
                // success lands at the commit op.
                return invoke_goal_as_call(first.args[0],
                                            if_then_commit_pc);
            }
            // Plain disjunction: push a CP whose alt_pc is
            // disj_alt_pc and a paired DisjFrame carrying G2 + the
            // original after_pc. Dispatch G1 normally.
            ChoicePoint cp_;
            cp_.alt_pc            = disj_alt_pc;
            cp_.saved_cp          = cp;
            cp_.trail_mark        = trail.size();
            cp_.cut_barrier       = cut_barrier;
            cp_.saved_regs        = regs;
            cp_.saved_mode_stack  = mode_stack;
            cp_.saved_env_stack   = env_stack;
            cp_.saved_body_frames = body_frames;
            choice_points.push_back(std::move(cp_));
            DisjFrame df;
            df.second_goal = g.args[1];
            df.after_pc    = after_pc;
            disj_frames.push_back(std::move(df));
            return invoke_goal_as_call(g.args[0], after_pc);
        }
        auto slash = key.rfind(''/'');
        if (slash == std::string::npos) return false;
        std::size_t arity = 0;
        try { arity = std::stoul(key.substr(slash + 1)); }
        catch (...) { return false; }
        if (arity != g.args.size()) return false;
        // Always set up A-registers from the goal''s args before any
        // dispatch path — meta-builtins, user predicates, and direct
        // builtins all read them.
        for (std::size_t i = 0; i < arity; ++i) {
            std::string an = "A" + std::to_string(i + 1);
            regs[an] = g.args[i];
        }
        // Meta-builtins handled directly by step() arms — go through
        // the same code path so nested catch / re-throw works.
        if (key == "throw/1")   { cp = after_pc; return execute_throw(); }
        if (key == "catch/3")   { cp = after_pc; return execute_catch(); }
        if (key == "findall/3") { cp = after_pc; return dispatch_findall_call(after_pc); }
        if (key == "bagof/3")   { cp = after_pc; return dispatch_aggregate_call("bagof", after_pc); }
        if (key == "setof/3")   { cp = after_pc; return dispatch_aggregate_call("setof", after_pc); }
        if (key == "sub_atom/5") return dispatch_sub_atom(after_pc);
        if (key == "retract/1") return dispatch_retract(after_pc);
        // ^/2 — existential quantification. For find-style
        // aggregation (which is all this runtime currently
        // supports for bagof/setof) ^/2 is transparent: invoke A2
        // and ignore the binder. (Full bagof/setof grouping
        // semantics would treat ^/2 specially during the bagof
        // body; here we just dispatch the goal.)
        if (key == "^/2") {
            return invoke_goal_as_call(g.args[1], after_pc);
        }
        // once(G) — succeed once with G''s first solution, fail if
        // G has no solutions. Equivalent to `(G -> true)`. Build an
        // IfThenFrame with then_goal=true_atom and else_goal=null;
        // the existing if-then machinery handles the rest (Cond
        // success → IfThenCommit cuts and runs then; Cond failure →
        // IfThenElse with null else propagates failure).
        if (key == "once/1" && g.args.size() == 1) {
            IfThenFrame itf;
            itf.then_goal     = std::make_shared<Cell>(Value::Atom("true"));
            // else_goal stays null — IfThenElse propagates failure.
            itf.after_pc      = after_pc;
            itf.base_cp_count = choice_points.size();
            if_then_frames.push_back(std::move(itf));
            ChoicePoint cp_;
            cp_.alt_pc            = if_then_else_pc;
            cp_.saved_cp          = cp;
            cp_.trail_mark        = trail.size();
            cp_.cut_barrier       = cut_barrier;
            cp_.saved_regs        = regs;
            cp_.saved_mode_stack  = mode_stack;
            cp_.saved_env_stack   = env_stack;
            cp_.saved_body_frames = body_frames;
            choice_points.push_back(std::move(cp_));
            return invoke_goal_as_call(g.args[0], if_then_commit_pc);
        }
        // forall(G, T) — succeed iff for every solution of G, T
        // succeeds. Desugars to `\\+ (G, \\+ T)`: build the inner
        // (G, \\+ T) conjunction on the heap and dispatch through
        // \\+/1''s builtin handler, which handles double-negation
        // correctly via paired NegationFrames.
        if (key == "forall/2" && g.args.size() == 2) {
            std::vector<CellPtr> neg_t_args;
            neg_t_args.push_back(g.args[1]);
            CellPtr neg_t = std::make_shared<Cell>(
                Value::Compound("\\\\+/1", std::move(neg_t_args)));
            std::vector<CellPtr> conj_args;
            conj_args.push_back(g.args[0]);
            conj_args.push_back(neg_t);
            CellPtr conj = std::make_shared<Cell>(
                Value::Compound(",/2", std::move(conj_args)));
            std::vector<CellPtr> outer_neg_args;
            outer_neg_args.push_back(conj);
            CellPtr outer = std::make_shared<Cell>(
                Value::Compound("\\\\+/1", std::move(outer_neg_args)));
            return invoke_goal_as_call(outer, after_pc);
        }
        // User predicate path.
        auto it = labels.find(key);
        if (it != labels.end()) {
            cp = after_pc;
            pc = it->second;
            return true;
        }
        // Dynamic predicate path: when a rule''s body invokes a
        // dynamic predicate (or a goal-term arg to catch/findall/etc.
        // names one), we route through dispatch_dynamic_call so the
        // clause-iteration CP machinery handles backtracking.
        if (dynamic_db.count(key)) {
            return dispatch_dynamic_call(key, after_pc);
        }
        // Builtin path: run the builtin inline. On success, jump to
        // the catch continuation (overriding the pc advance builtin()
        // performs on its own — that increment refers to a non-existent
        // surrounding instruction).
        if (!builtin(key, static_cast<std::int64_t>(arity))) return false;
        if (after_pc == 0) { halt = true; return true; }
        pc = after_pc;
        cp = 0;
        return true;
    }
    return false;
}

bool WamState::dispatch_call_meta(const std::string& op,
                                  std::int64_t instr_n,
                                  std::size_t after_pc) {
    // Prefer the arity the caller passed (instr.n from both Call AND
    // Execute now that Execute carries arity via its factory). Fall
    // back to parsing the op-name suffix for robustness in case any
    // direct caller forgets to pass it.
    std::int64_t total_arity = instr_n;
    if (total_arity < 1) {
        auto slash = op.rfind(''/'');
        if (slash != std::string::npos) {
            try { total_arity = std::stoll(op.substr(slash + 1)); }
            catch (...) { return false; }
        }
    }
    if (total_arity < 1) return false;
    CellPtr goal_cell = get_cell("A1");
    if (total_arity == 1) {
        // No extras — dispatch A1 as-is.
        return invoke_goal_as_call(goal_cell, after_pc);
    }
    // Snapshot extras BEFORE we mutate A-registers further. Extras
    // come from A2..AN as cell pointers (so any aliasing into the
    // future combined goal stays sharing-correct).
    std::vector<CellPtr> extras;
    extras.reserve(total_arity - 1);
    for (std::int64_t i = 2; i <= total_arity; ++i) {
        extras.push_back(get_cell("A" + std::to_string(i)));
    }
    // Resolve the goal''s base name and existing args.
    Value goal = deref(*goal_cell);
    std::string base_name;
    std::size_t base_arity = 0;
    std::vector<CellPtr> all_args;
    if (goal.tag == Value::Tag::Atom) {
        base_name = goal.s;
    } else if (goal.tag == Value::Tag::Compound) {
        auto slash = goal.s.rfind(''/'');
        if (slash == std::string::npos) return false;
        base_name = goal.s.substr(0, slash);
        try { base_arity = std::stoul(goal.s.substr(slash + 1)); }
        catch (...) { return false; }
        if (base_arity != goal.args.size()) return false;
        all_args = goal.args;
    } else {
        // Unbound goal — ISO call/N throws instantiation_error;
        // v1 lax just fails.
        return false;
    }
    for (auto& e : extras) all_args.push_back(e);
    std::size_t new_arity = base_arity + extras.size();
    std::string new_functor = base_name + "/" + std::to_string(new_arity);
    CellPtr combined = std::make_shared<Cell>(
        Value::Compound(new_functor, std::move(all_args)));
    return invoke_goal_as_call(combined, after_pc);
}

bool WamState::dispatch_findall_call(std::size_t after_pc) {
    return dispatch_aggregate_call("collect", after_pc);
}

bool WamState::dispatch_aggregate_call(const std::string& kind,
                                       std::size_t after_pc) {
    // findall/bagof/setof(Template, Goal, List). Reached via the
    // Call/Execute special-case for non-inlined occurrences (the
    // WAM compiler only inlines BeginAggregate/EndAggregate for the
    // OUTERMOST findall/bagof/setof in a body; nested ones are
    // emitted as plain calls). kind selects finalize_aggregate''s
    // behaviour:
    //   "collect" — findall semantics (list, empty on no solutions).
    //   "bagof"   — list, FAILS on no solutions, GROUPS by witness.
    //   "setof"   — like bagof but sort+dedup within each group.
    //
    // For bagof/setof: walk Goal to find free witnesses (unbound
    // vars that aren''t in Template and aren''t under ^/2). Witness
    // values are snapshotted alongside each acc[i] for group
    // partitioning at finalise time.
    AggregateFrame frame;
    frame.agg_kind          = kind;
    frame.value_cell        = get_cell("A1");
    frame.result_cell       = get_cell("A3");
    frame.begin_pc          = pc;
    frame.return_pc         = after_pc;
    frame.return_pc_set     = true;
    frame.base_cp_count     = choice_points.size();
    frame.trail_mark        = trail.size();
    frame.saved_cp          = cp;
    frame.saved_cut_barrier = cut_barrier;
    frame.saved_regs        = regs;
    frame.saved_mode_stack  = mode_stack;
    frame.saved_env_stack   = env_stack;
    if (kind == "bagof" || kind == "setof") {
        // Build the "vars in Template" exclude set, then walk Goal
        // to find witness cells (skipping anything in the exclude
        // set or reachable only via the LHS of ^/2). The result
        // order is deterministic (DFS over the goal tree).
        std::set<Cell*> exclude;
        std::set<Cell*> seen_in_template;
        std::vector<CellPtr> tmpl_vars_unused;
        collect_goal_witnesses(get_cell("A1"), exclude,
                               tmpl_vars_unused, seen_in_template);
        for (auto& c : tmpl_vars_unused) exclude.insert(c.get());
        std::set<Cell*> seen;
        collect_goal_witnesses(get_cell("A2"), exclude,
                               frame.witness_cells, seen);
    }
    aggregate_frames.push_back(std::move(frame));
    CellPtr goal = get_cell("A2");
    return invoke_goal_as_call(goal, findall_collect_pc);
}

void WamState::collect_goal_witnesses(CellPtr goal,
                                      const std::set<Cell*>& exclude,
                                      std::vector<CellPtr>& out,
                                      std::set<Cell*>& seen) const {
    if (!goal) return;
    // Use the DEREF''d underlying cell for identity (so var-chains
    // collapse to one address).
    Value v = deref(*goal);
    Cell* key = goal.get();
    // Re-deref via the trail: if goal is bound, follow the chain.
    // For our purposes the cell-identity of goal''s direct pointer
    // suffices, since unbound vars always point to themselves in
    // this runtime''s cell model.
    if (v.tag == Value::Tag::Unbound || v.tag == Value::Tag::Uninit) {
        if (seen.count(key)) return;
        seen.insert(key);
        if (exclude.count(key)) return;
        out.push_back(goal);
        return;
    }
    if (v.tag == Value::Tag::Compound) {
        // ^/2 binder: skip the LHS (existentially quantified vars),
        // recurse into the RHS only.
        if (v.s == "^/2" && v.args.size() == 2) {
            // Collect LHS vars into a local exclude set merged with
            // the caller''s, then walk only the RHS.
            std::set<Cell*> exclude2 = exclude;
            std::set<Cell*> lhs_seen;
            std::vector<CellPtr> lhs_vars;
            collect_goal_witnesses(v.args[0], exclude, lhs_vars, lhs_seen);
            for (auto& c : lhs_vars) exclude2.insert(c.get());
            collect_goal_witnesses(v.args[1], exclude2, out, seen);
            return;
        }
        for (auto& arg : v.args) {
            collect_goal_witnesses(arg, exclude, out, seen);
        }
    }
}

bool WamState::aggregate_bind_next_group() {
    if (aggregate_group_iters.empty()) return false;
    AggregateGroupIterator& it = aggregate_group_iters.back();
    if (it.remaining_groups.empty()) {
        // No more groups — drop the iterator and propagate failure.
        aggregate_group_iters.pop_back();
        return false;
    }
    // Snapshot before popping the next group: if we have more
    // groups after this one, we''ll push a CP that on retry brings
    // us back here (with one fewer remaining).
    std::pair<std::vector<Value>, std::vector<Value>> group =
        std::move(it.remaining_groups.front());
    it.remaining_groups.erase(it.remaining_groups.begin());
    std::size_t saved_return_pc = it.return_pc;
    std::string saved_kind = it.agg_kind;
    std::vector<CellPtr> saved_witness_cells = it.witness_cells;
    CellPtr saved_result_cell = it.result_cell;
    bool more_groups = !it.remaining_groups.empty();
    // Snapshot the FULL iterator state in case we need to push a
    // CP — the CP''s saved_regs etc come from "now," and the iter
    // remains on the stack for AggregateNextGroup to peek at.
    // Push the CP BEFORE binding so trail-undo on backtrack
    // restores pre-binding state.
    if (more_groups) {
        ChoicePoint cp_;
        cp_.alt_pc            = aggregate_next_group_pc;
        cp_.saved_cp          = cp;
        cp_.trail_mark        = trail.size();
        cp_.cut_barrier       = cut_barrier;
        cp_.saved_regs        = regs;
        cp_.saved_mode_stack  = mode_stack;
        cp_.saved_env_stack   = env_stack;
        cp_.saved_body_frames = body_frames;
        choice_points.push_back(std::move(cp_));
    }
    // Bind witness cells to this group''s witness values.
    for (std::size_t k = 0; k < saved_witness_cells.size()
             && k < group.first.size(); ++k) {
        CellPtr wc = saved_witness_cells[k];
        if (!wc) continue;
        if (wc->is_unbound()) bind_cell(wc, group.first[k]);
        else if (!(*wc == group.first[k])) return false;
    }
    Value result = finalize_aggregate(saved_kind, group.second);
    if (result.tag == Value::Tag::Uninit) return false;
    if (!saved_result_cell) return false;
    if (saved_result_cell->is_unbound()) {
        bind_cell(saved_result_cell, result);
    } else if (!unify_cells(saved_result_cell,
                            std::make_shared<Cell>(result))) {
        return false;
    }
    if (saved_return_pc == 0) { halt = true; return true; }
    pc = saved_return_pc;
    cp = 0;
    return true;
}

// catch(Goal, Catcher, Recovery): push a CatcherFrame snapshotting
// current VM state, then dispatch to Goal as a tail-call whose
// continuation is the auto-injected CatchReturn instruction.
bool WamState::execute_catch() {
    CatcherFrame f;
    f.catcher_term = get_cell("A2");
    f.recovery_term = get_cell("A3");
    f.saved_cp = cp;
    f.trail_mark = trail.size();
    f.base_cp_count = choice_points.size();
    f.base_agg_count = aggregate_frames.size();
    f.saved_cut_barrier = cut_barrier;
    f.saved_regs = regs;
    f.saved_mode_stack = mode_stack;
    f.saved_env_stack = env_stack;
    std::size_t my_depth = catcher_frames.size();
    catcher_frames.push_back(std::move(f));
    // Snapshot the goal cell BEFORE we touch any A-registers (since
    // invoke_goal_as_call sets A1..AN from the goal''s args, which
    // would clobber A1 mid-read if goal_cell aliases A1).
    CellPtr goal = get_cell("A1");
    if (!invoke_goal_as_call(goal, catch_return_pc)) {
        // Goal failed to dispatch. If the failure is a plain failure
        // (frame still on stack), pop it ourselves and propagate. If
        // the failure was an uncaught throw walking past us, the
        // frame is already gone and we just propagate.
        if (catcher_frames.size() > my_depth) catcher_frames.pop_back();
        return false;
    }
    return true;
}

// throw(Term): snapshot Term, then walk catcher_frames top→bottom,
// unwinding state at each frame and trying to unify the snapshot with
// the frame''s catcher pattern. The first matching frame invokes its
// recovery goal as a tail-call. No match → uncaught exception.
bool WamState::execute_throw() {
    CellPtr thrown = deep_copy_term(get_cell("A1"));
    while (!catcher_frames.empty()) {
        CatcherFrame f = std::move(catcher_frames.back());
        catcher_frames.pop_back();
        // Restore VM state to the frame''s snapshot.
        while (trail.size() > f.trail_mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        while (choice_points.size() > f.base_cp_count) choice_points.pop_back();
        while (aggregate_frames.size() > f.base_agg_count) aggregate_frames.pop_back();
        regs = f.saved_regs;
        mode_stack = f.saved_mode_stack;
        env_stack = f.saved_env_stack;
        cut_barrier = f.saved_cut_barrier;
        // Try to unify the thrown term with this frame''s catcher.
        std::size_t mark = trail.size();
        if (unify_cells(thrown, f.catcher_term)) {
            // Matched. Invoke recovery as a tail call back to the
            // saved continuation.
            return invoke_goal_as_call(f.recovery_term, f.saved_cp);
        }
        // No match: undo the failed unify attempt and try the next
        // outer frame.
        while (trail.size() > mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
    }
    // Uncaught exception. Print it and signal a hard failure.
    std::fprintf(stderr, "uncaught exception: %s\\n",
                 render(deref(*thrown)).c_str());
    halt = true;
    return false;
}

// ----------------------------------------------------------------------
// ISO error term constructors + thrower. Used by the *_iso/N builtin
// flavours added in follow-up PRs. The shapes match
// WAM_CPP_ISO_ERRORS_SPECIFICATION §6.
// ----------------------------------------------------------------------

bool WamState::throw_iso_error(Value err_term) {
    // Wrap in error(ErrTerm, Context) with Context = fresh unbound.
    CellPtr err_cell = std::make_shared<Cell>(std::move(err_term));
    CellPtr ctx_cell = std::make_shared<Cell>(
        Value::Unbound("_E" + std::to_string(var_counter++)));
    std::vector<CellPtr> args = { err_cell, ctx_cell };
    Value wrapper = Value::Compound("error/2", std::move(args));
    // Set A1 = the wrapped error term, then dispatch through the
    // existing throw machinery so catch/3 frames see a normal throw.
    regs["A1"] = std::make_shared<Cell>(std::move(wrapper));
    return execute_throw();
}

Value WamState::make_type_error(const std::string& expected, Value culprit) {
    std::vector<CellPtr> args = {
        std::make_shared<Cell>(Value::Atom(expected)),
        std::make_shared<Cell>(std::move(culprit))
    };
    return Value::Compound("type_error/2", std::move(args));
}

Value WamState::make_instantiation_error() {
    // ISO uses the atom "instantiation_error" (no args).
    return Value::Atom("instantiation_error");
}

Value WamState::make_domain_error(const std::string& domain, Value culprit) {
    std::vector<CellPtr> args = {
        std::make_shared<Cell>(Value::Atom(domain)),
        std::make_shared<Cell>(std::move(culprit))
    };
    return Value::Compound("domain_error/2", std::move(args));
}

Value WamState::make_evaluation_error(const std::string& kind) {
    std::vector<CellPtr> args = {
        std::make_shared<Cell>(Value::Atom(kind))
    };
    return Value::Compound("evaluation_error/1", std::move(args));
}

bool WamState::term_contains_unbound(CellPtr c) const {
    Value v = deref(*c);
    if (v.tag == Value::Tag::Unbound || v.tag == Value::Tag::Uninit) return true;
    if (v.tag == Value::Tag::Compound) {
        for (auto& arg : v.args) {
            if (term_contains_unbound(arg)) return true;
        }
    }
    return false;
}

bool WamState::term_has_zero_divide(CellPtr c) const {
    Value v = deref(*c);
    if (v.tag != Value::Tag::Compound) return false;
    if (v.args.size() == 2 && (v.s == "//2" || v.s == "///2"
                            || v.s == "mod/2" || v.s == "rem/2")) {
        Value rhs = deref(*v.args[1]);
        if (rhs.tag == Value::Tag::Integer && rhs.i == 0) return true;
        if (rhs.tag == Value::Tag::Float && rhs.f == 0.0) return true;
    }
    for (auto& arg : v.args) {
        if (term_has_zero_divide(arg)) return true;
    }
    return false;
}

Value WamState::arith_culprit(const Value& v) const {
    // ISO type_error(evaluable, Culprit) expects Culprit to be a
    // Name/Arity compound (`foo/0` for an atom; `unknown/3` for an
    // unknown compound). Fall back to the raw value if the shape is
    // unrecognised — diagnostic is still useful, just not standards-
    // perfect.
    if (v.tag == Value::Tag::Atom) {
        std::vector<CellPtr> args = {
            std::make_shared<Cell>(Value::Atom(v.s)),
            std::make_shared<Cell>(Value::Integer(0))
        };
        return Value::Compound("//2", std::move(args));
    }
    if (v.tag == Value::Tag::Compound) {
        // Compound s is "name/arity". Split and rebuild as the
        // ISO-shaped name/arity compound.
        auto slash = v.s.rfind(''/'');
        if (slash != std::string::npos) {
            std::string name  = v.s.substr(0, slash);
            std::string arity = v.s.substr(slash + 1);
            try {
                long long n = std::stoll(arity);
                std::vector<CellPtr> args = {
                    std::make_shared<Cell>(Value::Atom(name)),
                    std::make_shared<Cell>(Value::Integer(n))
                };
                return Value::Compound("//2", std::move(args));
            } catch (...) {}
        }
    }
    return v;
}

bool WamState::backtrack() {
    // Pop normal choice points until we either find one to retry or run
    // into an open aggregate frame''s base — at which point the frame is
    // finalised and execution continues past its EndAggregate.
    for (;;) {
        // Conjunction frame: G1 failed (CPs drained to G1''s base).
        // Pop the ConjFrame so the failure propagates to whatever
        // pushed it (e.g. an outer findall/3''s goal-dispatch path).
        // The frame just remembers G2 + after_pc; popping it discards
        // those and lets normal backtracking continue.
        if (!conj_frames.empty()
            && choice_points.size() <= conj_frames.back().base_cp_count)
        {
            conj_frames.pop_back();
            continue;
        }
        // BodyFrame: a dynamic rule''s body sequence ran out of CPs
        // for one of its goals. Drop the frame so failure propagates
        // out of the rule and the outer dynamic_try_next (if any)
        // can try the next clause.
        if (!body_frames.empty()
            && choice_points.size() <= body_frames.back().base_cp_count)
        {
            body_frames.pop_back();
            continue;
        }
        // Negation frame: if the \\+ / not protected goal exhausted
        // all its CPs (i.e. the goal FAILED), pop the frame, unwind
        // state, and SUCCEED at saved_cp. Symmetric inverse of the
        // catcher-frame pop below.
        if (!negation_frames.empty()
            && choice_points.size() == negation_frames.back().base_cp_count
            && (aggregate_frames.empty()
                || aggregate_frames.back().base_cp_count
                       <= negation_frames.back().base_cp_count)
            && (catcher_frames.empty()
                || catcher_frames.back().base_cp_count
                       <= negation_frames.back().base_cp_count))
        {
            NegationFrame f = std::move(negation_frames.back());
            negation_frames.pop_back();
            while (trail.size() > f.trail_mark) {
                TrailEntry t = std::move(trail.back());
                trail.pop_back();
                *t.cell = std::move(t.prev);
            }
            while (aggregate_frames.size() > f.base_agg_count)
                aggregate_frames.pop_back();
            while (catcher_frames.size() > f.base_catcher_count)
                catcher_frames.pop_back();
            regs        = std::move(f.saved_regs);
            mode_stack  = std::move(f.saved_mode_stack);
            env_stack   = std::move(f.saved_env_stack);
            cut_barrier = f.saved_cut_barrier;
            cp          = f.saved_cp;
            // Negation succeeds — proceed to the caller''s continuation.
            if (cp == 0) { halt = true; return true; }
            pc = cp;
            cp = 0;
            return true;
        }
        // If the catch-protected goal has exhausted all its CPs (and
        // didn''t throw), pop the catcher frame and propagate failure
        // outside the catch. We do this only when the catcher''s
        // base_cp_count is strictly above any open aggregate''s base
        // (so aggregate finalisation still takes precedence).
        if (!catcher_frames.empty()
            && choice_points.size() == catcher_frames.back().base_cp_count
            && (aggregate_frames.empty()
                || aggregate_frames.back().base_cp_count
                       <= catcher_frames.back().base_cp_count))
        {
            CatcherFrame f = std::move(catcher_frames.back());
            catcher_frames.pop_back();
            // Unwind trail to the catcher''s mark — failure of the
            // protected goal undoes all bindings it made.
            while (trail.size() > f.trail_mark) {
                TrailEntry t = std::move(trail.back());
                trail.pop_back();
                *t.cell = std::move(t.prev);
            }
            // Restore regs/env/mode/cp/cut so the world looks exactly
            // as it did at catch entry, then continue backtracking
            // outside the catch.
            regs        = std::move(f.saved_regs);
            mode_stack  = std::move(f.saved_mode_stack);
            env_stack   = std::move(f.saved_env_stack);
            cp          = f.saved_cp;
            cut_barrier = f.saved_cut_barrier;
            continue;
        }
        std::size_t agg_base =
            aggregate_frames.empty() ? 0 : aggregate_frames.back().base_cp_count;
        if (!aggregate_frames.empty() && choice_points.size() == agg_base) {
            // Aggregate exhausted. Finalise.
            AggregateFrame f = std::move(aggregate_frames.back());
            aggregate_frames.pop_back();
            // Unwind any trail entries added inside the aggregate scope.
            while (trail.size() > f.trail_mark) {
                TrailEntry t = std::move(trail.back());
                trail.pop_back();
                *t.cell = std::move(t.prev);
            }
            regs        = std::move(f.saved_regs);
            mode_stack  = std::move(f.saved_mode_stack);
            env_stack   = std::move(f.saved_env_stack);
            cp          = f.saved_cp;
            cut_barrier = f.saved_cut_barrier;
            // Resolve the result cell up-front so both the
            // grouped-bagof/setof and the simple find-style paths
            // can use it.
            CellPtr rcell = f.result_cell
                ? f.result_cell
                : get_cell(f.result_reg);
            if (!rcell) return false;
            // Bagof/setof with free witnesses: partition acc by
            // acc_witnesses, push an AggregateGroupIterator, and
            // bind the first group via the shared helper. The
            // helper also pushes a CP if more groups remain — that
            // CP''s alt_pc = aggregate_next_group_pc takes over on
            // backtrack to bind the next group.
            if (!f.witness_cells.empty() && !f.acc_witnesses.empty()
                && f.acc.size() == f.acc_witnesses.size())
            {
                // Build groups in discovery order, with each
                // group''s template list filled in for matching
                // witness rows.
                std::vector<std::pair<std::vector<Value>, std::vector<Value>>> groups;
                std::vector<bool> assigned(f.acc.size(), false);
                for (std::size_t i = 0; i < f.acc.size(); ++i) {
                    if (assigned[i]) continue;
                    std::vector<Value> templates;
                    templates.push_back(f.acc[i]);
                    assigned[i] = true;
                    for (std::size_t j = i + 1; j < f.acc.size(); ++j) {
                        if (assigned[j]) continue;
                        bool same = (f.acc_witnesses[j].size()
                                     == f.acc_witnesses[i].size());
                        for (std::size_t k = 0;
                             same && k < f.acc_witnesses[i].size(); ++k)
                        {
                            if (!(f.acc_witnesses[j][k]
                                  == f.acc_witnesses[i][k]))
                                same = false;
                        }
                        if (same) {
                            templates.push_back(f.acc[j]);
                            assigned[j] = true;
                        }
                    }
                    groups.emplace_back(f.acc_witnesses[i],
                                        std::move(templates));
                }
                if (groups.empty()) {
                    // No solutions — bagof/setof fail per ISO.
                    continue;
                }
                AggregateGroupIterator iter;
                iter.agg_kind        = f.agg_kind;
                iter.remaining_groups = std::move(groups);
                iter.witness_cells   = std::move(f.witness_cells);
                iter.result_cell     = rcell;
                iter.return_pc       = f.return_pc_set
                    ? f.return_pc
                    : find_matching_end_aggregate(instrs, f.begin_pc) + 1;
                aggregate_group_iters.push_back(std::move(iter));
                if (!aggregate_bind_next_group()) {
                    // Mismatch on bind (or empty groups race) —
                    // continue backtracking.
                    continue;
                }
                return true;
            }
            // Build and bind the result. The meta-call findall/3
            // path stores the result cell directly (set by
            // dispatch_findall_call); the inlined BeginAggregate
            // path uses the result_reg name.
            Value result = finalize_aggregate(f.agg_kind, f.acc);
            if (result.tag == Value::Tag::Uninit) {
                // bagof/setof with empty acc → the aggregate
                // FAILS (per ISO). Continue backtracking so any
                // enclosing if-then-else / catch-frame / outer
                // choice point gets to react. Returning false here
                // would short-circuit the WAM''s normal failure
                // propagation.
                continue;
            }
            if (rcell->is_unbound()) bind_cell(rcell, result);
            else if (!unify_cells(rcell, std::make_shared<Cell>(result))) return false;
            // Jump past the matching EndAggregate.
            if (f.return_pc_set) {
                pc = f.return_pc;
            } else {
                pc = find_matching_end_aggregate(instrs, f.begin_pc) + 1;
            }
            return true;
        }
        if (choice_points.empty()) return false;
        // Classic WAM: do NOT pop the CP here. retry_me_else updates the
        // CP''s alt_pc to the next clause as it runs; trust_me pops it.
        // We restore state via COPY so the snapshot remains intact for
        // any subsequent backtracks into this same CP.
        const ChoicePoint& cp_ = choice_points.back();
        while (trail.size() > cp_.trail_mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        regs        = cp_.saved_regs;
        mode_stack  = cp_.saved_mode_stack;
        env_stack   = cp_.saved_env_stack;
        body_frames = std::move(cp_.saved_body_frames);
        cp          = cp_.saved_cp;
        cut_barrier = cp_.cut_barrier;
        pc          = cp_.alt_pc;
        return true;
    }
}

bool WamState::run() {
    halt = false;
    while (!halt) {
        if (pc >= instrs.size()) return false;
        if (!step(instrs[pc])) {
            if (!backtrack()) return false;
        }
    }
    return true;
}

bool WamState::query(const std::string& pred_key, const std::vector<Value>& args) {
    auto it = labels.find(pred_key);
    if (it == labels.end()) return false;
    regs.clear();
    for (std::size_t k = 0; k < args.size(); ++k) {
        regs["A" + std::to_string(k + 1)] = make_cell(args[k]);
    }
    trail.clear();
    choice_points.clear();
    aggregate_frames.clear();
    mode_stack.clear();
    env_stack.clear();
    pc = it->second;
    cp = 0;
    cut_barrier = 0;
    halt = false;
    return run();
}

Program::Setup& Program::setup_hook() {
    static Setup s = nullptr;
    return s;
}

void Program::register_setup(Setup s) { setup_hook() = s; }
void Program::apply_setup(WamState& vm) { if (setup_hook()) setup_hook()(vm); }

} // namespace wam_cpp
').

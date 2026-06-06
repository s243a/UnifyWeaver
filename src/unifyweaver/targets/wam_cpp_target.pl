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
:- use_module('../core/relation_policy', [
       get_effective_policy/4
   ]).
:- use_module(wam_runtime_parser_capability, [
       wam_target_runtime_parser/3,
       parser_dependent_body_goal/2
   ]).
% Load the portable Prolog term parser so its predicates are
% visible to current_predicate/1 when runtime_parser(compiled) is
% requested. The module is small (~445 lines, ~40 predicates) and
% load-once; non-compiled-mode callers pay the load cost but no
% runtime cost (the predicates only enter the generated output
% when the expansion explicitly references them).
:- use_module('../core/prolog_term_parser', []).
:- use_module('../core/cpp_runtime_parser_wrappers', []).
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
    wam_text_to_items/2,
    wam_classify_constant_token/2
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
%
%  Atom-vs-number disambiguation: tokens that arrived with outer
%  single quotes preserved (per wam_text_parser:wam_tokenize_line/2)
%  are atoms regardless of whether the inner text reparses as a
%  number. wam_classify_constant_token/2 implements the convention;
%  this emitter just maps the classification to a C++ Value literal.
cpp_value_literal(C, Val) :-
    to_string(C, Str),
    wam_classify_constant_token(Str, Class),
    (   Class = integer(N)
    ->  format(atom(Val), 'Value::Integer(~w)', [N])
    ;   Class = float(F)
    ->  format(atom(Val), 'Value::Float(~w)', [F])
    ;   Class = atom(Name),
        escape_cpp_string(Name, EscStr),
        format(atom(Val), 'Value::Atom("~w")', [EscStr])
    ).

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
%% Indexed-dispatch chain ops (issue #2400).
wam_instruction_to_cpp_literal_det(try(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Try(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(retry(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Retry(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(trust(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Trust(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(jump(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Jump(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(cut_ite, _, 'Instruction::CutIte()').
wam_instruction_to_cpp_literal_det(get_level(Yn), _, Code) :-
    to_string(Yn, YS),
    format(atom(Code), 'Instruction::GetLevel("~w")', [YS]).
wam_instruction_to_cpp_literal_det(cut(Yn), _, Code) :-
    to_string(Yn, YS),
    format(atom(Code), 'Instruction::Cut("~w")', [YS]).
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
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true), ite_use_y_level(true)], WamText),
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
    % Append thirteen trailing synthetic-return instructions for the
    % meta-call control-flow surface — see WamState for each pc''s
    % role.
    append(FlatInstrs0,
           [catch_return, negation_return, findall_collect,
            conj_return, disj_alt, if_then_commit, if_then_else,
            aggregate_next_group, dynamic_next_clause, sub_atom_next,
            body_next, retract_next, output_capture_return,
            current_pred_next],
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
    OutputCaptureReturnPC is CatchReturnPC + 12,
    CurrentPredNextPC is CatchReturnPC + 13,
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
    % cpp_fact_sources LMDB load calls -- one per source, appended
    % at the end of wam_cpp_setup so dynamic_db is populated before
    % any query runs. Idempotency is enforced runtime-side.
    lmdb_sources_from_options(Options, LmdbSources),
    warn_runtime_sorts(LmdbSources),
    findall(LoadLine, (
        member(lmdb_source(Key, Path, DbName, Unique, OnDup, Order, _),
               LmdbSources),
        emit_lmdb_load_call(Key, Path, DbName, Unique, OnDup,
                            Order, LoadLine)
    ), LoadLines),
    atomic_list_concat(LoadLines, '\n', LmdbLoadBody),
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
    vm.output_capture_return_pc = ~w;
    vm.current_pred_next_pc = ~w;
~w
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
    BodyNextPC, RetractNextPC, OutputCaptureReturnPC,
    CurrentPredNextPC,
    LabelBody, InstrBody, LmdbLoadBody]).

% Render one cpp_load_lmdb_fact_source call. DbName is either an
% atom (named sub-DB) or [] (default unnamed DB -> nullptr).
% Unique is a boolean controlling value-uniqueness enforcement at
% load time; OnDup is one of {throw,warn,overwrite,first_wins,
% keep_all,fallback(P)} from the relation_policy directive; Order
% is the declared order spec (translated to a SortKey list, empty
% when LMDB's natural iteration order satisfies it).
emit_lmdb_load_call(Key, Path, DbName, Unique, OnDup, Order, Line) :-
    escape_cpp_string(Key, EKey),
    escape_cpp_string(Path, EPath),
    db_name_arg(DbName, DbArg),
    on_dup_cpp_enum(OnDup, DupEnum),
    cpp_bool(Unique, UniqueLit),
    order_cpp_sort_keys(Order, SortKeysLit),
    format(atom(Line),
        '    cpp_load_lmdb_fact_source(vm, "~w", "~w", ~w, LmdbLoadOptions{~w, LmdbLoadOptions::OnDup::~w, ~w});',
        [EKey, EPath, DbArg, UniqueLit, DupEnum, SortKeysLit]).

db_name_arg([], 'nullptr') :- !.
db_name_arg(DbName, Arg) :-
    escape_cpp_string(DbName, EDb),
    format(atom(Arg), '"~w"', [EDb]).

cpp_bool(true, 'true') :- !.
cpp_bool(false, 'false').

% Map the Prolog on_duplicate policy atom to the C++ enum tag.
% fallback(Policy) collapses to the inner policy for v1 (Phase 2
% does not yet chain).
on_dup_cpp_enum(throw,      'throw_').
on_dup_cpp_enum(warn,       'warn').
on_dup_cpp_enum(overwrite,  'overwrite').
on_dup_cpp_enum(first_wins, 'first_wins').
on_dup_cpp_enum(keep_all,   'keep_all').
on_dup_cpp_enum(fallback(P), Tag) :- on_dup_cpp_enum(P, Tag).

%% order_cpp_sort_keys(+OrderSpec, -SortKeysLiteral) is det.
%
% Translate the relation_policy order(...) spec into a C++
% initializer-list literal for LmdbLoadOptions::sort_keys.
% Emits "{}" (empty) when the declared order is trivially
% satisfied by LMDB's natural key-ascending iteration -- this is
% the cheap path. Emits "{ {N, asc}, ... }" otherwise.
%
% LMDB key uniqueness means a leading `arg(1)` / `asc(arg(1))`
% guarantees a total order on its own, so any trailing sort
% keys after such a leader are redundant and we drop them too.
%
% v1: arity 2; column indices are 1 or 2.
order_cpp_sort_keys(OrderSpec, '{}') :-
    trivial_order(OrderSpec), !.
order_cpp_sort_keys(OrderSpec, Literal) :-
    order_to_sort_keys(OrderSpec, Keys),
    findall(KeyLit, (
        member(key(Col, Asc), Keys),
        cpp_bool(Asc, AscLit),
        format(atom(KeyLit),
               'LmdbLoadOptions::SortKey{~w, ~w}', [Col, AscLit])
    ), KeyLits),
    atomic_list_concat(KeyLits, ', ', Inner),
    format(atom(Literal), '{ ~w }', [Inner]).

% True for order specs that LMDB satisfies without a sort pass.
trivial_order(natural)   :- !.
trivial_order(insertion) :- !.
trivial_order([])        :- !.
trivial_order([First|_]) :-
    % A leading asc(arg(1)) (or bare arg(1)) totally orders by
    % unique key, so the rest of the chain is redundant.
    sort_term_to_key(First, key(1, true)).

% Convert an order term to a (Col, Asc) pair.
sort_term_to_key(arg(N),       key(N, true)) :- integer(N).
sort_term_to_key(asc(arg(N)),  key(N, true)) :- integer(N).
sort_term_to_key(desc(arg(N)), key(N, false)) :- integer(N).

% Translate a non-trivial order spec into a list of key(Col, Asc).
order_to_sort_keys([], []).
order_to_sort_keys([H|T], [K|KT]) :-
    sort_term_to_key(H, K),
    order_to_sort_keys(T, KT).

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

% predsort/3 stdlib + assertion/1: defined as ordinary user-module
% Prolog clauses asserted at module load. The compile path picks
% them up via clause/2 like any other user predicate. Users opt in
% either by including the specific helpers in their predicate list,
% or via the include_stdlib(true|List) option to
% write_wam_cpp_project, which auto-prepends the registered helpers.
:- (   current_predicate(user:wam_cpp_predsort_/5)
   ->  true
   ;   assertz((user:predsort(P, L, S) :-
           length(L, N), wam_cpp_predsort_(N, P, L, _, S))),
       assertz((user:wam_cpp_predsort_(2, P, [X1, X2|T], T, R) :- !,
           call(P, O, X1, X2), wam_cpp_sort2(O, X1, X2, R))),
       assertz((user:wam_cpp_predsort_(1, _, [X|T], T, [X]) :- !)),
       assertz((user:wam_cpp_predsort_(0, _, T, T, []) :- !)),
       assertz((user:wam_cpp_predsort_(N, P, L1, L3, R) :-
           N1 is N // 2, N2 is N - N1,
           wam_cpp_predsort_(N1, P, L1, L2, R1),
           wam_cpp_predsort_(N2, P, L2, L3, R2),
           wam_cpp_predmerge(P, R1, R2, R))),
       assertz((user:wam_cpp_sort2(<, A, B, [A, B]))),
       assertz((user:wam_cpp_sort2(=, A, _, [A]))),
       assertz((user:wam_cpp_sort2(>, A, B, [B, A]))),
       assertz((user:wam_cpp_predmerge(_, [], R, R) :- !)),
       assertz((user:wam_cpp_predmerge(_, R, [], R) :- !)),
       assertz((user:wam_cpp_predmerge(P, [H1|T1], [H2|T2], R) :-
           call(P, O, H1, H2),
           wam_cpp_predmerge_(O, P, H1, H2, T1, T2, R))),
       assertz((user:wam_cpp_predmerge_(<, P, H1, H2, T1, T2, [H1|R]) :-
           wam_cpp_predmerge(P, T1, [H2|T2], R))),
       assertz((user:wam_cpp_predmerge_(=, P, H1, _, T1, T2, [H1|R]) :-
           wam_cpp_predmerge(P, T1, T2, R))),
       assertz((user:wam_cpp_predmerge_(>, P, H1, H2, T1, T2, [H2|R]) :-
           wam_cpp_predmerge(P, [H1|T1], T2, R)))
   ).
:- (   current_predicate(user:assertion/1)
   ->  true
   ;   assertz((user:assertion(G) :-
           ( call(G)
           -> true
           ;  throw(error(assertion_failed, G))
           )))
   ).
% assoc/2 stdlib: AVL tree keyed by standard order. The empty
% tree is the atom `t`; non-empty nodes are
% t(Key, Value, Balance, Left, Right) where Balance is one of:
%   <   left subtree is one taller
%   =   subtrees are equal height
%   >   right subtree is one taller
% This is the standard SWI library(assoc) AVL representation. Insert
% tracks a height-change flag (same / grew) up the spine; once a
% node would tip to imbalance, a single or double rotation restores
% the invariant. Re-uses compare/3 (already in the runtime) for
% standard-order key comparison.
:- (   current_predicate(user:put_assoc/4)
   ->  true
   ;   assertz((user:empty_assoc(t))),
       assertz((user:get_assoc(Key, t(K, V, _, L, R), Value) :-
           compare(Order, Key, K),
           wam_cpp_get_assoc_(Order, Key, V, L, R, Value))),
       assertz((user:wam_cpp_get_assoc_(=, _, Value, _, _, Value))),
       assertz((user:wam_cpp_get_assoc_(<, Key, _, L, _, Value) :-
           get_assoc(Key, L, Value))),
       assertz((user:wam_cpp_get_assoc_(>, Key, _, _, R, Value) :-
           get_assoc(Key, R, Value))),
       % put_assoc/4: drop the height-change flag from the worker.
       assertz((user:put_assoc(Key, Tree, Val, NewTree) :-
           wam_cpp_put_assoc_recur(Key, Tree, Val, NewTree, _))),
       % Insert into empty subtree — the new node grew the tree.
       assertz((user:wam_cpp_put_assoc_recur(Key, t, Val,
                                            t(Key, Val, =, t, t), grew))),
       % Insert into a non-empty node — compare + dispatch.
       assertz((user:wam_cpp_put_assoc_recur(Key, t(K, V, B, L, R), Val,
                                            NewTree, Change) :-
           compare(Order, Key, K),
           wam_cpp_put_assoc_dispatch(Order, Key, Val, K, V, B,
                                      L, R, NewTree, Change))),
       % =: replace value, balance preserved.
       assertz((user:wam_cpp_put_assoc_dispatch(=, Key, Val, _, _, B,
                                               L, R,
                                               t(Key, Val, B, L, R),
                                               same))),
       % <: recurse into L, then ask wam_cpp_rebalance_left to decide
       % whether the subtree got taller and whether to rotate.
       assertz((user:wam_cpp_put_assoc_dispatch(<, Key, Val, K, V, B,
                                               L, R, NewTree, Change) :-
           wam_cpp_put_assoc_recur(Key, L, Val, L1, LChange),
           wam_cpp_rebalance_left(LChange, B, K, V, L1, R,
                                  NewTree, Change))),
       % >: symmetric — recurse into R, then rebalance from the right.
       assertz((user:wam_cpp_put_assoc_dispatch(>, Key, Val, K, V, B,
                                               L, R, NewTree, Change) :-
           wam_cpp_put_assoc_recur(Key, R, Val, R1, RChange),
           wam_cpp_rebalance_right(RChange, B, K, V, L, R1,
                                   NewTree, Change))),
       % wam_cpp_rebalance_left(LChange, B, K, V, L, R, NewTree,
       %                       Change)
       % LChange=same — copy through.
       assertz((user:wam_cpp_rebalance_left(same, B, K, V, L, R,
                                           t(K, V, B, L, R), same))),
       % LChange=grew, was right-heavy — now balanced, height same.
       assertz((user:wam_cpp_rebalance_left(grew, >, K, V, L, R,
                                           t(K, V, =, L, R), same))),
       % LChange=grew, was balanced — now left-heavy, height grew.
       assertz((user:wam_cpp_rebalance_left(grew, =, K, V, L, R,
                                           t(K, V, <, L, R), grew))),
       % LChange=grew, was left-heavy — rotate. Look at L''s balance.
       assertz((user:wam_cpp_rebalance_left(grew, <, K, V, L, R,
                                           NewTree, same) :-
           wam_cpp_rotate_left_heavy(L, K, V, R, NewTree))),
       % LL case — single right rotation. L''s left grew.
       assertz((user:wam_cpp_rotate_left_heavy(t(LK, LV, <, LL, LR),
                                              K, V, R,
                                              t(LK, LV, =, LL,
                                                t(K, V, =, LR, R))))),
       % LR case — double rotation. L''s right grew. New balances
       % depend on the inner node''s balance.
       assertz((user:wam_cpp_rotate_left_heavy(
                   t(LK, LV, >, LL, t(LRK, LRV, LRB, LRL, LRR)),
                   K, V, R,
                   t(LRK, LRV, =,
                     t(LK, LV, NLB, LL, LRL),
                     t(K,  V,  NRB, LRR, R))) :-
           wam_cpp_lr_balance(LRB, NLB, NRB))),
       % wam_cpp_rebalance_right(RChange, B, K, V, L, R, NewTree,
       %                        Change) — mirror of left.
       assertz((user:wam_cpp_rebalance_right(same, B, K, V, L, R,
                                            t(K, V, B, L, R), same))),
       assertz((user:wam_cpp_rebalance_right(grew, <, K, V, L, R,
                                            t(K, V, =, L, R), same))),
       assertz((user:wam_cpp_rebalance_right(grew, =, K, V, L, R,
                                            t(K, V, >, L, R), grew))),
       assertz((user:wam_cpp_rebalance_right(grew, >, K, V, L, R,
                                            NewTree, same) :-
           wam_cpp_rotate_right_heavy(L, K, V, R, NewTree))),
       % RR case — single left rotation.
       assertz((user:wam_cpp_rotate_right_heavy(L, K, V,
                                               t(RK, RV, >, RL, RR),
                                               t(RK, RV, =,
                                                 t(K, V, =, L, RL),
                                                 RR)))),
       % RL case — double rotation. New balances depend on inner.
       assertz((user:wam_cpp_rotate_right_heavy(
                   L, K, V,
                   t(RK, RV, <, t(RLK, RLV, RLB, RLL, RLR), RR),
                   t(RLK, RLV, =,
                     t(K,  V,  NLB, L,   RLL),
                     t(RK, RV, NRB, RLR, RR))) :-
           wam_cpp_rl_balance(RLB, NLB, NRB))),
       % Balance-factor tables for the double-rotation cases:
       %   LR rotation: the inner node (LR) had balance LRB; afterwards
       %   the new left child gets NLB and the new right child gets NRB.
       assertz((user:wam_cpp_lr_balance(<, =, >))),
       assertz((user:wam_cpp_lr_balance(=, =, =))),
       assertz((user:wam_cpp_lr_balance(>, <, =))),
       %   RL rotation (mirror image):
       assertz((user:wam_cpp_rl_balance(<, =, >))),
       assertz((user:wam_cpp_rl_balance(=, =, =))),
       assertz((user:wam_cpp_rl_balance(>, <, =))),
       % min_assoc/3 — leftmost (smallest-key) node.
       assertz((user:min_assoc(t(K, V, _, t, _), K, V) :- !)),
       assertz((user:min_assoc(t(_, _, _, L, _), K, V) :- min_assoc(L, K, V))),
       % max_assoc/3 — rightmost (largest-key) node.
       assertz((user:max_assoc(t(K, V, _, _, t), K, V) :- !)),
       assertz((user:max_assoc(t(_, _, _, _, R), K, V) :- max_assoc(R, K, V))),
       % del_assoc/4 — fails if Key not in Tree0. Returns the removed
       % value and a rebalanced tree. Cross-checked against
       % library(assoc) on ascending/descending bulk deletes.
       assertz((user:del_assoc(K, T0, V, T) :-
           wam_cpp_del_assoc_recur(K, T0, V, T, _))),
       % wam_cpp_del_assoc_recur(+Key, +Tree, -Val, -NewTree, -Change)
       %   Change ∈ {same, shrunk}.
       assertz((user:wam_cpp_del_assoc_recur(K, t(K0, V0, B, L, R),
                                            V, T, Change) :-
           compare(Order, K, K0),
           wam_cpp_del_assoc_dispatch(Order, K, K0, V0, B, L, R,
                                      V, T, Change))),
       % Found it — replace this node.
       assertz((user:wam_cpp_del_assoc_dispatch(=, _, _, V, B, L, R,
                                               V, T, Change) :-
           wam_cpp_del_assoc_replace(B, L, R, T, Change))),
       % Recurse left, then rebalance from the left.
       assertz((user:wam_cpp_del_assoc_dispatch(<, K, K0, V0, B, L, R,
                                               V, T, Change) :-
           wam_cpp_del_assoc_recur(K, L, V, L1, LC),
           wam_cpp_del_rebalance_left(LC, B, K0, V0, L1, R, T, Change))),
       % Recurse right, then rebalance from the right.
       assertz((user:wam_cpp_del_assoc_dispatch(>, K, K0, V0, B, L, R,
                                               V, T, Change) :-
           wam_cpp_del_assoc_recur(K, R, V, R1, RC),
           wam_cpp_del_rebalance_right(RC, B, K0, V0, L, R1, T, Change))),
       % wam_cpp_del_assoc_replace(B, L, R, NewTree, Change)
       %   Build the replacement tree when the target node is found.
       % Empty left — collapse to right.
       assertz((user:wam_cpp_del_assoc_replace(_, t, R, R, shrunk))),
       % Empty right (and non-empty left) — collapse to left.
       assertz((user:wam_cpp_del_assoc_replace(_, L, t, L, shrunk) :-
           L \= t)),
       % Both sides non-empty — pull the in-order successor from the
       % right subtree and use its (K, V) as the new root.
       assertz((user:wam_cpp_del_assoc_replace(B, L, R, T, Change) :-
           L \= t, R \= t,
           wam_cpp_del_min_extract(R, SK, SV, R1, RC),
           wam_cpp_del_rebalance_right(RC, B, SK, SV, L, R1, T, Change))),
       % wam_cpp_del_min_extract(+Tree, -K, -V, -NewTree, -Change)
       %   Extract the minimum key/value, returning the rest of the
       %   tree and a height-change flag.
       assertz((user:wam_cpp_del_min_extract(t(K, V, _, t, R), K, V, R,
                                            shrunk))),
       assertz((user:wam_cpp_del_min_extract(t(K0, V0, B, L, R),
                                            K, V, T, Change) :-
           L \= t,
           wam_cpp_del_min_extract(L, K, V, L1, LC),
           wam_cpp_del_rebalance_left(LC, B, K0, V0, L1, R, T, Change))),
       % wam_cpp_del_rebalance_left(LChange, B, K, V, L, R, NewTree,
       %                           Change)
       %   Rebalance after the LEFT subtree shrunk by 0 or 1.
       assertz((user:wam_cpp_del_rebalance_left(same, B, K, V, L, R,
                                               t(K, V, B, L, R), same))),
       % Was left-heavy, lost the extra → balanced, height shrunk.
       assertz((user:wam_cpp_del_rebalance_left(shrunk, <, K, V, L, R,
                                               t(K, V, =, L, R), shrunk))),
       % Was balanced → right-heavy, height stayed.
       assertz((user:wam_cpp_del_rebalance_left(shrunk, =, K, V, L, R,
                                               t(K, V, >, L, R), same))),
       % Was right-heavy → now over-tipped right; rotate based on R''s
       % balance.
       assertz((user:wam_cpp_del_rebalance_left(shrunk, >, K, V, L, R,
                                               T, Change) :-
           wam_cpp_del_rotate_right(L, K, V, R, T, Change))),
       % Single left rotation (R right-heavy or balanced) + double
       % rotation via R''s left subtree (R left-heavy). Three RB cases;
       % only the RB=> case matches the structure of put''s
       % rotate_right_heavy, the other two are delete-specific.
       % RB=>: single left rotation, height shrunk.
       assertz((user:wam_cpp_del_rotate_right(L, K, V,
                   t(RK, RV, >, RL, RR),
                   t(RK, RV, =, t(K, V, =, L, RL), RR),
                   shrunk))),
       % RB==: single left rotation, height stays (rotation tilts the
       % outer node to <, inner to >).
       assertz((user:wam_cpp_del_rotate_right(L, K, V,
                   t(RK, RV, =, RL, RR),
                   t(RK, RV, <, t(K, V, >, L, RL), RR),
                   same))),
       % RB=<: double rotation via RL — same shape as put''s RL case,
       % height shrunk.
       assertz((user:wam_cpp_del_rotate_right(L, K, V,
                   t(RK, RV, <, t(RLK, RLV, RLB, RLL, RLR), RR),
                   t(RLK, RLV, =,
                     t(K,  V,  NLB, L,   RLL),
                     t(RK, RV, NRB, RLR, RR)),
                   shrunk) :-
           wam_cpp_rl_balance(RLB, NLB, NRB))),
       % wam_cpp_del_rebalance_right(RChange, B, K, V, L, R, NewTree,
       %                            Change) — mirror of left.
       assertz((user:wam_cpp_del_rebalance_right(same, B, K, V, L, R,
                                                t(K, V, B, L, R), same))),
       assertz((user:wam_cpp_del_rebalance_right(shrunk, >, K, V, L, R,
                                                t(K, V, =, L, R), shrunk))),
       assertz((user:wam_cpp_del_rebalance_right(shrunk, =, K, V, L, R,
                                                t(K, V, <, L, R), same))),
       assertz((user:wam_cpp_del_rebalance_right(shrunk, <, K, V, L, R,
                                                T, Change) :-
           wam_cpp_del_rotate_left(L, K, V, R, T, Change))),
       % LB=<: single right rotation, height shrunk.
       assertz((user:wam_cpp_del_rotate_left(
                   t(LK, LV, <, LL, LR), K, V, R,
                   t(LK, LV, =, LL, t(K, V, =, LR, R)),
                   shrunk))),
       % LB==: single right rotation, height stays.
       assertz((user:wam_cpp_del_rotate_left(
                   t(LK, LV, =, LL, LR), K, V, R,
                   t(LK, LV, >, LL, t(K, V, <, LR, R)),
                   same))),
       % LB=>: double rotation via LR — same shape as put''s LR case.
       assertz((user:wam_cpp_del_rotate_left(
                   t(LK, LV, >, LL, t(LRK, LRV, LRB, LRL, LRR)), K, V, R,
                   t(LRK, LRV, =,
                     t(LK, LV, NLB, LL,  LRL),
                     t(K,  V,  NRB, LRR, R)),
                   shrunk) :-
           wam_cpp_lr_balance(LRB, NLB, NRB))),
       % del_min_assoc/4 — delete the smallest key, return its
       % (K, V) and the rebalanced tree. Fails on empty.
       assertz((user:del_min_assoc(Assoc, K, V, NewAssoc) :-
           Assoc \= t,
           wam_cpp_del_min_extract(Assoc, K, V, NewAssoc, _))),
       % del_max_assoc/4 — mirror, plus a max-extract helper that
       % wam_cpp_del_min_extract didn''t need.
       assertz((user:del_max_assoc(Assoc, K, V, NewAssoc) :-
           Assoc \= t,
           wam_cpp_del_max_extract(Assoc, K, V, NewAssoc, _))),
       assertz((user:wam_cpp_del_max_extract(t(K, V, _, L, t), K, V, L,
                                            shrunk))),
       assertz((user:wam_cpp_del_max_extract(t(K0, V0, B, L, R),
                                            K, V, T, Change) :-
           R \= t,
           wam_cpp_del_max_extract(R, K, V, R1, RC),
           wam_cpp_del_rebalance_right(RC, B, K0, V0, L, R1, T, Change))),
       % get_assoc/5 — atomic test-and-set. Returns the current value
       % (fails if absent) AND builds a tree with that slot replaced
       % by NewVal. Tree structure / balance is preserved (the value
       % swap can''t change heights), so no rebalance is needed.
       assertz((user:get_assoc(Key, Assoc0, OldVal, Assoc, NewVal) :-
           wam_cpp_get_replace_assoc(Key, Assoc0, OldVal, NewVal, Assoc))),
       assertz((user:wam_cpp_get_replace_assoc(Key, t(K0, V0, B, L, R),
                                              OldVal, NewVal, NewTree) :-
           compare(Order, Key, K0),
           wam_cpp_get_replace_dispatch(Order, Key, K0, V0, B, L, R,
                                        OldVal, NewVal, NewTree))),
       assertz((user:wam_cpp_get_replace_dispatch(=, _, K0, V0, B, L, R,
                                                 V0, NewVal,
                                                 t(K0, NewVal, B, L, R)))),
       assertz((user:wam_cpp_get_replace_dispatch(<, Key, K0, V0, B, L, R,
                                                 OldVal, NewVal,
                                                 t(K0, V0, B, L1, R)) :-
           wam_cpp_get_replace_assoc(Key, L, OldVal, NewVal, L1))),
       assertz((user:wam_cpp_get_replace_dispatch(>, Key, K0, V0, B, L, R,
                                                 OldVal, NewVal,
                                                 t(K0, V0, B, L, R1)) :-
           wam_cpp_get_replace_assoc(Key, R, OldVal, NewVal, R1))),
       % map_assoc/3 — call(Goal, OldVal, NewVal) on every value;
       % build a tree with the same structure / balance but the
       % transformed values.
       assertz((user:map_assoc(_, t, t))),
       assertz((user:map_assoc(Goal, t(K, V, B, L, R),
                                     t(K, V1, B, L1, R1)) :-
           map_assoc(Goal, L, L1),
           call(Goal, V, V1),
           map_assoc(Goal, R, R1))),
       assertz((user:list_to_assoc(List, Assoc) :-
           wam_cpp_list_to_assoc_(List, t, Assoc))),
       assertz((user:wam_cpp_list_to_assoc_([], A, A))),
       assertz((user:wam_cpp_list_to_assoc_([K-V|T], A0, A) :-
           put_assoc(K, A0, V, A1),
           wam_cpp_list_to_assoc_(T, A1, A))),
       assertz((user:assoc_to_list(t, []))),
       assertz((user:assoc_to_list(t(K, V, _, L, R), Pairs) :-
           assoc_to_list(L, LP),
           assoc_to_list(R, RP),
           append(LP, [K-V|RP], Pairs))),
       assertz((user:assoc_to_keys(Assoc, Keys) :-
           assoc_to_list(Assoc, Pairs),
           wam_cpp_pairs_keys(Pairs, Keys))),
       assertz((user:assoc_to_values(Assoc, Values) :-
           assoc_to_list(Assoc, Pairs),
           wam_cpp_pairs_values(Pairs, Values))),
       assertz((user:wam_cpp_pairs_keys([], []))),
       assertz((user:wam_cpp_pairs_keys([K-_|T], [K|KT]) :-
           wam_cpp_pairs_keys(T, KT))),
       assertz((user:wam_cpp_pairs_values([], []))),
       assertz((user:wam_cpp_pairs_values([_-V|T], [V|VT]) :-
           wam_cpp_pairs_values(T, VT)))
   ).
% lists_extra stdlib: small grab-bag of common helpers not yet
% covered as runtime builtins. Asserted in the user module same way
% as predsort/assoc; pulled in by include_stdlib(lists_extra) or
% include_stdlib(true).
:- (   current_predicate(user:pairs_keys/2)
   ->  true
   ;   assertz((user:pairs_keys([], []))),
       assertz((user:pairs_keys([K-_|T], [K|KT]) :- pairs_keys(T, KT))),
       assertz((user:pairs_values([], []))),
       assertz((user:pairs_values([_-V|T], [V|VT]) :- pairs_values(T, VT))),
       assertz((user:pairs_keys_values([], [], []))),
       assertz((user:pairs_keys_values([K-V|T], [K|KT], [V|VT]) :-
           pairs_keys_values(T, KT, VT))),
       % transpose_pairs/2 — swap K-V in each pair, then keysort on
       % the new keys. Matches SWI library(pairs) semantics.
       assertz((user:transpose_pairs(Pairs, Transposed) :-
           wam_cpp_flip_pairs(Pairs, Flipped),
           keysort(Flipped, Transposed))),
       assertz((user:wam_cpp_flip_pairs([], []))),
       assertz((user:wam_cpp_flip_pairs([K-V|T0], [V-K|T]) :-
           wam_cpp_flip_pairs(T0, T))),
       % map_list_to_pairs(:Function, +List, -Pairs) — for each E in
       % List, build Function(E)-E. Used to attach a sort key to
       % each element before keysort/2 (Schwartzian-style).
       assertz((user:map_list_to_pairs(_, [], []))),
       assertz((user:map_list_to_pairs(F, [V|T0], [K-V|T]) :-
           call(F, V, K),
           map_list_to_pairs(F, T0, T))),
       assertz((user:take(0, _, []) :- !)),
       assertz((user:take(_, [], []) :- !)),
       assertz((user:take(N, [H|T], [H|R]) :-
           N > 0, N1 is N - 1, take(N1, T, R))),
       assertz((user:drop(0, L, L) :- !)),
       assertz((user:drop(_, [], []) :- !)),
       assertz((user:drop(N, [_|T], R) :-
           N > 0, N1 is N - 1, drop(N1, T, R))),
       % Standard three-clause intersection/union. Earlier the
       % runtime couldn''t synthesize a CP at an indexed sub-chain
       % entry, so 3+-clause predicates with overlapping compound
       % heads lost choices. With the SwitchOnTerm + RetryMeElse
       % CP-synthesis path in place (this PR''s runtime change),
       % three clauses backtrack normally.
       assertz((user:intersection([], _, []))),
       assertz((user:intersection([H|T], L, [H|R]) :-
           member(H, L), !, intersection(T, L, R))),
       assertz((user:intersection([_|T], L, R) :- intersection(T, L, R))),
       assertz((user:union([], L, L))),
       assertz((user:union([H|T], L, R) :-
           member(H, L), !, union(T, L, R))),
       assertz((user:union([H|T], L, [H|R]) :- union(T, L, R))),
       assertz((user:permutation([], []))),
       assertz((user:permutation(L, [H|T]) :-
           select(H, L, Rest), permutation(Rest, T))),
       % memberchk/2 — deterministic member: commits on the first
       % match. Same semantics as SWI library(lists).
       assertz((user:memberchk(X, [X|_]) :- !)),
       assertz((user:memberchk(X, [_|T]) :- memberchk(X, T))),
       % sum_list/2 — sum of numeric elements via tail-recursive
       % accumulator. Empty list sums to 0 (matches SWI).
       assertz((user:sum_list(L, S) :- wam_cpp_sum_list_acc(L, 0, S))),
       assertz((user:wam_cpp_sum_list_acc([], A, A))),
       assertz((user:wam_cpp_sum_list_acc([H|T], A, S) :-
           A1 is A + H, wam_cpp_sum_list_acc(T, A1, S))),
       % max_list/2, min_list/2 — numeric max/min. Fail on empty
       % list (matches SWI: needs at least one element to compare).
       assertz((user:max_list([H|T], M) :- wam_cpp_max_list_acc(T, H, M))),
       assertz((user:wam_cpp_max_list_acc([], M, M))),
       assertz((user:wam_cpp_max_list_acc([H|T], A, M) :-
           H > A, !, wam_cpp_max_list_acc(T, H, M))),
       assertz((user:wam_cpp_max_list_acc([_|T], A, M) :-
           wam_cpp_max_list_acc(T, A, M))),
       assertz((user:min_list([H|T], M) :- wam_cpp_min_list_acc(T, H, M))),
       assertz((user:wam_cpp_min_list_acc([], M, M))),
       assertz((user:wam_cpp_min_list_acc([H|T], A, M) :-
           H < A, !, wam_cpp_min_list_acc(T, H, M))),
       assertz((user:wam_cpp_min_list_acc([_|T], A, M) :-
           wam_cpp_min_list_acc(T, A, M))),
       % max_member/2, min_member/2 — standard order of terms (not
       % just numbers). Note arg order: (-Max, +List), matching SWI.
       assertz((user:max_member(M, [H|T]) :- wam_cpp_max_member_acc(T, H, M))),
       assertz((user:wam_cpp_max_member_acc([], M, M))),
       assertz((user:wam_cpp_max_member_acc([H|T], A, M) :-
           H @> A, !, wam_cpp_max_member_acc(T, H, M))),
       assertz((user:wam_cpp_max_member_acc([_|T], A, M) :-
           wam_cpp_max_member_acc(T, A, M))),
       assertz((user:min_member(M, [H|T]) :- wam_cpp_min_member_acc(T, H, M))),
       assertz((user:wam_cpp_min_member_acc([], M, M))),
       assertz((user:wam_cpp_min_member_acc([H|T], A, M) :-
           H @< A, !, wam_cpp_min_member_acc(T, H, M))),
       assertz((user:wam_cpp_min_member_acc([_|T], A, M) :-
           wam_cpp_min_member_acc(T, A, M))),
       % same_length(?L1, ?L2) — true if L1 and L2 have the same
       % length. Bidirectional: works in any mode pattern.
       assertz((user:same_length([], []))),
       assertz((user:same_length([_|T1], [_|T2]) :-
           same_length(T1, T2))),
       % proper_length(@List, ?Length) — length of a proper list.
       % Fails on partial lists (unbound tail) and non-lists. The
       % explicit var/1 check is what distinguishes this from
       % length/2 on a partially-instantiated list.
       assertz((user:proper_length(L, N) :-
           wam_cpp_proper_length_acc(L, 0, N))),
       assertz((user:wam_cpp_proper_length_acc(L, _, _) :-
           var(L), !, fail)),
       assertz((user:wam_cpp_proper_length_acc([], N, N) :- !)),
       assertz((user:wam_cpp_proper_length_acc([_|T], A, N) :-
           A1 is A + 1, wam_cpp_proper_length_acc(T, A1, N))),
       % list_to_set(+List, -Set) — dedup, preserving first
       % occurrence. Uses memberchk against an accumulator of
       % already-seen elements (O(n^2) worst case, like SWI's
       % naive path; SWI also has a sort-based shortcut we don't
       % bother with).
       assertz((user:list_to_set(L, S) :- wam_cpp_l2s(L, [], S))),
       assertz((user:wam_cpp_l2s([], _, []))),
       assertz((user:wam_cpp_l2s([H|T], Seen, R) :-
           memberchk(H, Seen), !, wam_cpp_l2s(T, Seen, R))),
       assertz((user:wam_cpp_l2s([H|T], Seen, [H|R]) :-
           wam_cpp_l2s(T, [H|Seen], R))),
       % flatten(+NestedList, -FlatList) — flatten nested list
       % structure into a single proper list. Unbound terms and
       % non-list atoms are treated as leaves (matches SWI).
       % Clause order matters: var-check first, then [] for the
       % empty-list fast path, then [_|_] for descent, finally a
       % catch-all that wraps non-list values as singletons.
       assertz((user:flatten(X, [X]) :- var(X), !)),
       assertz((user:flatten([], []) :- !)),
       assertz((user:flatten([H|T], Flat) :- !,
           flatten(H, FH), flatten(T, FT), append(FH, FT, Flat))),
       assertz((user:flatten(X, [X])))
   ).

%% stdlib_feature_predicates(+Feature, -Predicates)
%  Registry mapping a stdlib feature name to its predicate
%  indicators (in compile order — main first, helpers after). Used
%  by write_wam_cpp_project's include_stdlib option to auto-prepend
%  the right helpers without callers having to spell them out.
stdlib_feature_predicates(predsort, [
    user:predsort/3,
    user:wam_cpp_predsort_/5,
    user:wam_cpp_sort2/4,
    user:wam_cpp_predmerge/4,
    user:wam_cpp_predmerge_/7
]).
stdlib_feature_predicates(assertion, [
    user:assertion/1
]).
stdlib_feature_predicates(assoc, [
    user:empty_assoc/1,
    user:get_assoc/3,
    user:wam_cpp_get_assoc_/6,
    user:put_assoc/4,
    user:wam_cpp_put_assoc_recur/5,
    user:wam_cpp_put_assoc_dispatch/10,
    user:wam_cpp_rebalance_left/8,
    user:wam_cpp_rebalance_right/8,
    user:wam_cpp_rotate_left_heavy/5,
    user:wam_cpp_rotate_right_heavy/5,
    user:wam_cpp_lr_balance/3,
    user:wam_cpp_rl_balance/3,
    user:min_assoc/3,
    user:max_assoc/3,
    user:del_assoc/4,
    user:wam_cpp_del_assoc_recur/5,
    user:wam_cpp_del_assoc_dispatch/10,
    user:wam_cpp_del_assoc_replace/5,
    user:wam_cpp_del_min_extract/5,
    user:wam_cpp_del_rebalance_left/8,
    user:wam_cpp_del_rebalance_right/8,
    user:wam_cpp_del_rotate_right/6,
    user:wam_cpp_del_rotate_left/6,
    user:del_min_assoc/4,
    user:del_max_assoc/4,
    user:wam_cpp_del_max_extract/5,
    user:get_assoc/5,
    user:wam_cpp_get_replace_assoc/5,
    user:wam_cpp_get_replace_dispatch/10,
    user:map_assoc/3,
    user:list_to_assoc/2,
    user:wam_cpp_list_to_assoc_/3,
    user:assoc_to_list/2,
    user:assoc_to_keys/2,
    user:assoc_to_values/2,
    user:wam_cpp_pairs_keys/2,
    user:wam_cpp_pairs_values/2
]).
stdlib_feature_predicates(lists_extra, [
    user:pairs_keys/2,
    user:pairs_values/2,
    user:pairs_keys_values/3,
    user:transpose_pairs/2,
    user:wam_cpp_flip_pairs/2,
    user:map_list_to_pairs/3,
    user:take/3,
    user:drop/3,
    user:intersection/3,
    user:union/3,
    user:permutation/2,
    user:memberchk/2,
    user:sum_list/2,
    user:wam_cpp_sum_list_acc/3,
    user:max_list/2,
    user:wam_cpp_max_list_acc/3,
    user:min_list/2,
    user:wam_cpp_min_list_acc/3,
    user:max_member/2,
    user:wam_cpp_max_member_acc/3,
    user:min_member/2,
    user:wam_cpp_min_member_acc/3,
    user:same_length/2,
    user:proper_length/2,
    user:wam_cpp_proper_length_acc/3,
    user:list_to_set/2,
    user:wam_cpp_l2s/3,
    user:flatten/2
]).

%% all_stdlib_features(-Features)
%  List of every known stdlib feature, in a stable order. Used when
%  the caller passes include_stdlib(true).
all_stdlib_features([predsort, assertion, assoc, lists_extra]).

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
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true), ite_use_y_level(true)], WamText),
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
        % --- nth1/3 ------------------------------------------------------
        % nth1(1, [X|_], X).
        % nth1(N, [_|T], X) :- N > 1, M is N - 1, nth1(M, T, X).
        label("nth1/3"),
        try_me_else("L_cpp_nth1_3_2"),
        get_constant("1", "A1"),
        get_list("A2"),
        unify_variable("X1"),
        unify_variable("X2"),
        get_value("X1", "A3"),
        proceed,
        label("L_cpp_nth1_3_2"),
        trust_me,
        allocate,
        get_variable("Y1", "A1"),
        get_list("A2"),
        unify_variable("X5"),
        unify_variable("Y3"),
        get_variable("Y4", "A3"),
        put_value("Y1", "A1"),
        put_constant("1", "A2"),
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
        execute("nth1/3"),
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
%% Indexed-dispatch chain ops (issue #2400).  See wam_target.pl''s
%% build_term_index_with_chains.  Target is the body label of one of
%% the matching clauses; the chain''s next-instruction PC is captured
%% in the CP at runtime so backtrack resumes through the chain
%% rather than re-entering the linear retry chain.
instr_to_setup_line(try(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Try(~w));', [PC]).
instr_to_setup_line(retry(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Retry(~w));', [PC]).
instr_to_setup_line(trust(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Trust(~w));', [PC]).
instr_to_setup_line(jump(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Jump(~w));', [PC]).
instr_to_setup_line(switch_on_constant(Entries), Labels, Line) :- !,
    parse_switch_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnConstant({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_constant_fallthrough(Entries), Labels, Line) :- !,
    % Mixed-mode A1 indexing: bound A1 with no entry in the table
    % must NOT fail — fall through to the try_me_else chain so the
    % variable-A1 clauses get a chance to match. Same opcode as
    % SwitchOnConstant, just with the no_match_fallthrough flag set.
    parse_switch_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnConstant({~w}, true));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_constant_a2(Entries), Labels, Line) :- !,
    % Real A2 dispatch — emitted when the WAM compiler decides A1 is
    % too variable to index on but A2 is all constants. Same dispatch
    % shape as SwitchOnConstant (jump to first matching entry, set
    % indexed_entry so RetryMeElse synthesizes a CP), just reads A2
    % instead of A1.
    parse_switch_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnConstantA2({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_constant_a2_fallthrough(Entries), Labels, Line) :- !,
    % Mixed-mode A2 indexing — A1 is variable in every clause, AND
    % the predicate has a variable-A2 clause somewhere. The indexed
    % prefix (clauses before the first variable A2) gets switch
    % entries; bound A2 with no match falls through to the chain so
    % the variable-A2 clauses still match. Mirror of A1''s
    % switch_on_constant_fallthrough.
    parse_switch_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnConstantA2({~w}, true));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_structure(Entries), Labels, Line) :- !,
    parse_switch_struct_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnStructure({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_structure_a2(Entries), Labels, Line) :- !,
    parse_switch_struct_entries(Entries, Labels, EntriesCpp),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnStructureA2({~w}));',
           [EntriesCpp]).
instr_to_setup_line(switch_on_term(Tokens), Labels, Line) :- !,
    parse_switch_term(Tokens, Labels, ConstsCpp, StructsCpp, ListPC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnTerm({~w}, {~w}, ~w));',
           [ConstsCpp, StructsCpp, ListPC]).
instr_to_setup_line(switch_on_term_a2(Tokens), Labels, Line) :- !,
    parse_switch_term(Tokens, Labels, ConstsCpp, StructsCpp, ListPC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::SwitchOnTermA2({~w}, {~w}, ~w));',
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
instr_to_setup_line(output_capture_return, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::OutputCaptureReturn());'.
instr_to_setup_line(current_pred_next, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::CurrentPredNext());'.
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
%  Atom / Integer / Float literal for a switch key. KeyStr comes
%  from wam_target:quote_wam_constant/2 (via the switch_on_constant
%  entry serialiser), so atom-vs-number disambiguation goes through
%  wam_classify_constant_token/2 — same convention as
%  cpp_value_literal/2.
key_to_cpp_value(KeyStr, Lit) :-
    wam_classify_constant_token(KeyStr, Class),
    (   Class = integer(N)
    ->  format(atom(Lit), 'Value::Integer(~w)', [N])
    ;   Class = float(F)
    ->  format(atom(Lit), 'Value::Float(~w)', [F])
    ;   Class = atom(Name),
        escape_cpp_string(Name, Esc),
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
write_wam_cpp_project(Predicates0, Options, ProjectDir) :-
    expand_stdlib_predicates(Predicates0, Options, Predicates1),
    % Resolve and act on the runtime-parser capability mode. The
    % hook (PR #2329) registered C++ as native(parse_term) by
    % default; this consults that registration plus the caller''s
    % runtime_parser(...) option, then:
    %   - none: rejects predicates whose statically visible body
    %           uses a parser-dependent builtin (read/2, etc).
    %   - native(_): no expansion (C++''s hand-written canonical
    %           parser is in the runtime already).
    %   - compiled(prolog_term_parser): auto-includes the portable
    %           parser predicates so 1+2-style operator notation
    %           works at runtime.
    wam_target_runtime_parser(cpp, Options, RuntimeParserMode),
    validate_cpp_runtime_parser_mode(Predicates1, RuntimeParserMode),
    expand_cpp_runtime_parser_predicates(Predicates1,
                                         RuntimeParserMode,
                                         Options,
                                         Predicates),
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
    % on_compile_error policy: warn (default) | throw | skip.
    %   warn  -- print "WAM C++: ... <Err>" to stderr and drop pred
    %   throw -- re-throw, surfacing the error at the caller
    %   skip  -- old silent-skip behavior (pre-#2293)
    (   member(on_compile_error(Policy0), Options)
    ->  Policy = Policy0
    ;   Policy = warn
    ),
    findall(Code, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true), ite_use_y_level(true)], WamCode),
              compile_wam_predicate_to_cpp(PI, WamCode, Options1, Code)
            ),
            Err,
            handle_compile_error(Policy, PI, Err)
        )
    ), Codes),
    atomic_list_concat(Codes, '\n\n', PredicatesCode).

%% handle_compile_error(+Policy, +PI, +Err)
%  Per-predicate compile failure handler. Logs / re-throws / drops
%  based on Policy. Always fails after warn/skip so findall just
%  omits the predicate''s Code entry; throw policy propagates the
%  original exception so the caller can react.
handle_compile_error(throw, PI, Err) :- !,
    format(user_error,
           "WAM C++: re-throwing compile error for ~w: ~w~n",
           [PI, Err]),
    throw(Err).
handle_compile_error(skip, _PI, _Err) :- !, fail.
handle_compile_error(_, PI, Err) :-
    % default + explicit "warn"
    format(user_error,
           "WAM C++: failed to compile ~w: ~w~n",
           [PI, Err]),
    fail.

foreign_pred_keys_from_options(Options, Keys) :-
    (   member(foreign_pred_keys(Keys0), Options)
    ->  Keys = Keys0
    ;   Keys = []
    ).

%% lmdb_sources_from_options(+Options, -Sources)
%  Extract the LMDB fact-source entries from the cpp_fact_sources
%  option. Returns a list of
%    lmdb_source(Key, Path, DbName, Unique, OnDup)
%  terms where Key is "Pred/Arity" string, Path is the env path
%  atom, DbName is an atom or [] (default DB), Unique is a boolean,
%  and OnDup is the on_duplicate policy (atom).
%
%  Unique + OnDup come from relation_policy/2 declarations and
%  per-source spec overrides via get_effective_policy/4. Phase 2
%  enforcement happens at load time in the runtime (PR follow-up
%  to PR #2325).
%
%  Per WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md v1: only arity-2 lmdb()
%  sources are accepted; other shapes will be added in follow-ups.
lmdb_sources_from_options(Options, Sources) :-
    (   member(cpp_fact_sources(Specs), Options)
    ->  findall(lmdb_source(Key, Path, DbName, Unique, OnDup, Order, SrcOpts), (
            member(source(Functor/Arity, SourceSpec), Specs),
            validate_lmdb_v1_arity(Functor, Arity),
            lmdb_source_spec(SourceSpec, Path, DbName, SrcOpts),
            format(atom(Key), '~w/~w', [Functor, Arity]),
            get_effective_policy(Functor/Arity, SrcOpts,
                                 unique, Unique),
            get_effective_policy(Functor/Arity, SrcOpts,
                                 on_duplicate, OnDup),
            get_effective_policy(Functor/Arity, SrcOpts,
                                 order, Order)
        ), Sources)
    ;   Sources = []
    ).

%% warn_runtime_sorts(+LmdbSources) is det.
%
% Walk the resolved sources and emit one user_error warning per
% source whose declared order requires a runtime sort. Called
% from emit_setup_function only -- not from the header-generation
% call site of lmdb_sources_from_options, otherwise the warning
% would fire twice per project.
warn_runtime_sorts(Sources) :-
    forall(member(lmdb_source(_, _, _, _, _, Order, SrcOpts), Sources),
           maybe_warn_runtime_sort_one(Order, SrcOpts, Sources)).

maybe_warn_runtime_sort_one(_, SrcOpts, _) :-
    member(quiet_sort(true), SrcOpts), !.
maybe_warn_runtime_sort_one(Order, _, Sources) :-
    order_cpp_sort_keys(Order, SortKeysLit),
    SortKeysLit \== '{}', !,
    % Recover the PredArity for the message via a reverse-lookup
    % in Sources; cheap since list is small.
    member(lmdb_source(Key, _, _, _, _, Order, _), Sources),
    format(user_error,
        "[wam-cpp] note: ~w order(~w) requires a runtime sort. LMDB iterates ascending by key; consider a compound-key schema (e.g. <sort_column>-<id>) so the data is already in the desired order at load time. Suppress with quiet_sort(true) in the source options.~n",
        [Key, Order]).
maybe_warn_runtime_sort_one(_, _, _).

% v1 supports arity 2 only; loudly reject anything else so the
% user sees a real codegen-time error rather than a confusing
% runtime mismatch.
validate_lmdb_v1_arity(_, 2) :- !.
validate_lmdb_v1_arity(Functor, Arity) :-
    throw(error(domain_error(lmdb_v1_arity_2, Functor/Arity), _)).

% Recognise the spec shapes lmdb(Path) and lmdb(Path, Opts).
% Returns the raw SrcOpts list (or []) so the caller can also
% consult per-source policy overrides (unique, on_duplicate, ...).
lmdb_source_spec(lmdb(Path), Path, [], []).
lmdb_source_spec(lmdb(Path, Opts), Path, DbName, SrcOpts) :-
    ( member(db_name(DbName0), Opts) -> DbName = DbName0 ; DbName = [] ),
    SrcOpts = Opts.

%% expand_stdlib_predicates(+Predicates0, +Options, -Predicates)
%  Honours the include_stdlib option: prepends helper predicate
%  indicators for each requested feature. Duplicates (when the user
%  already listed a helper explicitly) are removed.
%
%  Option shapes:
%    include_stdlib(true)            -- include every known feature
%    include_stdlib(false)           -- no expansion (the default)
%    include_stdlib([F1, F2, ...])   -- include only listed features
%    include_stdlib(F)               -- single-feature shorthand
%% validate_cpp_runtime_parser_mode(+Predicates, +Mode) is det.
%
% When the resolved mode is `none`, reject any input predicate
% whose statically visible body uses a parser-dependent builtin
% (read/2, read_term_from_atom/2,3, term_to_atom/2 in reverse
% mode). Mirrors R''s validate_r_runtime_parser_mode/2. Other
% modes are accepted without inspection.
validate_cpp_runtime_parser_mode(Predicates, none) :-
    !,
    (   cpp_predicates_parser_dependency(Predicates, Pred, Builtin)
    ->  throw(error(permission_error(use, runtime_parser, Builtin),
                    context(write_wam_cpp_project/3,
                            parser_disabled_for_predicate(Pred))))
    ;   true
    ).
validate_cpp_runtime_parser_mode(_Predicates, _Mode).

cpp_predicates_parser_dependency(Predicates, Pred, Builtin) :-
    member(Pred, Predicates),
    cpp_predicate_clause(Pred, _Head, Body),
    parser_dependent_body_goal(Body, Builtin),
    !.

cpp_predicate_clause(Module:Name/Arity, Head, Body) :-
    !,
    functor(Head, Name, Arity),
    clause(Module:Head, Body).
cpp_predicate_clause(Name/Arity, Head, Body) :-
    functor(Head, Name, Arity),
    clause(user:Head, Body).

%% expand_cpp_runtime_parser_predicates(+P0, +Mode, +Options, -P)
%
% For compiled(prolog_term_parser): prepend every predicate of
% src/unifyweaver/core/prolog_term_parser.pl + the wrappers to
% the input list, skipping imported helpers like member/2 which
% the WAM-cpp runtime supplies separately. This makes operator
% notation (1+2 etc.) parsable at runtime via
% parse_term_from_atom/3 -- the C++ native parser only handles
% canonical form (+(1, 2)).
%
% When the caller passes runtime_parser_subset([PI, ...]) in
% Options, only the transitive closure of those entry points is
% pulled in (reachable-from analysis over the parser source).
% Without subset, the full ~40-predicate parser is included.
% See docs/design/RUNTIME_PARSER_TRANSPILATION_IMPLEMENTATION_PLAN.md
% "Subset generation".
%
% Other modes leave the list unchanged.
expand_cpp_runtime_parser_predicates(P0, compiled(prolog_term_parser),
                                     Options, P) :-
    !,
    cpp_runtime_parser_module_predicates(ParserPreds),
    cpp_runtime_parser_wrapper_predicates(WrapperPreds),
    append(WrapperPreds, ParserPreds, AllParserPreds),
    cpp_runtime_parser_apply_subset(AllParserPreds, Options, Extras),
    % Wrappers / entry points already go before the rest in
    % AllParserPreds because we appended WrapperPreds first;
    % the subset preserves first-occurrence order via
    % dedupe_keep_first below.
    append(Extras, P0, Combined),
    dedupe_keep_first(Combined, P).
expand_cpp_runtime_parser_predicates(P, _Mode, _Options, P).

% Apply runtime_parser_subset(EntryPIs) from Options. Without the
% option, returns the full predicate universe unchanged. With it,
% computes the transitive closure of called parser predicates
% starting from each EntryPI and returns only the reachable set.
cpp_runtime_parser_apply_subset(AllParserPreds, Options, Subset) :-
    (   member(runtime_parser_subset(EntryPIs), Options)
    ->  must_be(list, EntryPIs),
        cpp_runtime_parser_resolve_entries(EntryPIs,
                                           AllParserPreds,
                                           ResolvedEntries),
        cpp_runtime_parser_closure(ResolvedEntries,
                                   AllParserPreds,
                                   Closure),
        % Preserve order of AllParserPreds in the output (so the
        % wrappers / entry points still come first).
        include({Closure}/[PI]>>memberchk(PI, Closure),
                AllParserPreds, Subset)
    ;   Subset = AllParserPreds
    ).

% Resolve user-supplied entry-point indicators (which may be bare
% Name/Arity) against the universe so we end up with the
% canonical Module:Name/Arity form used in AllParserPreds.
cpp_runtime_parser_resolve_entries([], _, []).
cpp_runtime_parser_resolve_entries([PI|Rest], Universe, [Canonical|Out]) :-
    cpp_runtime_parser_canonicalize_entry(PI, Universe, Canonical),
    cpp_runtime_parser_resolve_entries(Rest, Universe, Out).

cpp_runtime_parser_canonicalize_entry(Mod:N/A, Universe, Mod:N/A) :-
    memberchk(Mod:N/A, Universe), !.
cpp_runtime_parser_canonicalize_entry(N/A, Universe, Mod:N/A) :-
    member(Mod:N/A, Universe), !.
cpp_runtime_parser_canonicalize_entry(PI, _Universe, _) :-
    throw(error(domain_error(parser_subset_entry_point, PI), _)).

% Worklist closure: start with Entries, walk each clause body to
% find called predicates that are in the parser/wrapper universe,
% add them to the worklist if not already visited.
cpp_runtime_parser_closure(Entries, Universe, Closure) :-
    cpp_runtime_parser_closure_loop(Entries, Universe, [], Closure).

cpp_runtime_parser_closure_loop([], _, Acc, Acc).
cpp_runtime_parser_closure_loop([PI|Rest], Universe, Acc, Final) :-
    (   memberchk(PI, Acc)
    ->  cpp_runtime_parser_closure_loop(Rest, Universe, Acc, Final)
    ;   cpp_runtime_parser_called_preds(PI, Universe, Called),
        append(Rest, Called, NewQueue),
        cpp_runtime_parser_closure_loop(NewQueue, Universe,
                                        [PI|Acc], Final)
    ).

% Enumerate the universe-internal predicates called from PI's
% clause bodies. Body goals not in the universe (member/2,
% append/3, character-code arithmetic, etc.) are silently
% ignored -- they are either WAM-cpp builtins or stdlib helpers
% that come in via include_stdlib(lists_extra).
cpp_runtime_parser_called_preds(Mod:Name/Arity, Universe, Called) :-
    functor(Head, Name, Arity),
    findall(InUniverse,
            (   clause(Mod:Head, Body),
                body_goal(Body, Goal),
                callable(Goal),
                \+ control_construct(Goal),
                functor(Goal, GN, GA),
                ( memberchk(Mod:GN/GA, Universe)
                -> InUniverse = Mod:GN/GA
                ; member(M:GN/GA, Universe), InUniverse = M:GN/GA
                )
            ),
            CalledRaw),
    sort(CalledRaw, Called).

% Walk control constructs to extract leaf goals.
body_goal((A, _), G) :- body_goal(A, G).
body_goal((_, B), G) :- body_goal(B, G).
body_goal((A ; _), G) :- body_goal(A, G).
body_goal((_ ; B), G) :- body_goal(B, G).
body_goal((A -> _), G) :- body_goal(A, G).
body_goal((_ -> B), G) :- body_goal(B, G).
body_goal(\+ A, G)    :- body_goal(A, G).
body_goal(once(A), G) :- body_goal(A, G).
body_goal(G, G)       :- \+ control_construct(G).

control_construct((_,_)).
control_construct((_;_)).
control_construct((_->_)).
control_construct(\+_).
control_construct(once(_)).
control_construct(!).
control_construct(true).
control_construct(fail).

cpp_runtime_parser_module_predicates(Preds) :-
    findall(prolog_term_parser:N/A,
            (   current_predicate(prolog_term_parser:N/A),
                functor(H, N, A),
                once(clause(prolog_term_parser:H, _)),
                % Skip predicates imported from library(lists) etc.
                % -- the WAM-cpp runtime already provides member/2,
                % append/3, reverse/2 via include_stdlib(lists_extra).
                \+ predicate_property(prolog_term_parser:H,
                                       imported_from(_))
            ),
            Raw),
    sort(Raw, Preds).

% Enumerate the target-agnostic wrapper predicates that surface
% SWI-style parser builtin names on top of the portable parser.
% Currently: read_term_from_atom/2,3. See
% src/unifyweaver/core/cpp_runtime_parser_wrappers.pl.
cpp_runtime_parser_wrapper_predicates(Preds) :-
    findall(cpp_runtime_parser_wrappers:N/A,
            (   current_predicate(cpp_runtime_parser_wrappers:N/A),
                functor(H, N, A),
                once(clause(cpp_runtime_parser_wrappers:H, _)),
                \+ predicate_property(cpp_runtime_parser_wrappers:H,
                                       imported_from(_))
            ),
            Raw),
    sort(Raw, Preds).

expand_stdlib_predicates(Predicates0, Options, Predicates) :-
    (   member(include_stdlib(Spec), Options)
    ->  resolve_stdlib_features(Spec, Features),
        findall(Extras,
                ( member(F, Features),
                  stdlib_feature_predicates(F, Extras)
                ),
                NestedExtras),
        flatten_once(NestedExtras, AllExtras),
        % Append user-supplied entries last so we don't shadow their
        % explicit ordering; then dedupe keeping first occurrence.
        append(AllExtras, Predicates0, Combined),
        dedupe_keep_first(Combined, Predicates)
    ;   Predicates = Predicates0
    ).

resolve_stdlib_features(true, Features) :- !, all_stdlib_features(Features).
resolve_stdlib_features(false, []) :- !.
resolve_stdlib_features(List, List) :- is_list(List), !.
resolve_stdlib_features(F, [F]).

flatten_once([], []).
flatten_once([L|Ls], Out) :- append(L, Rest, Out), flatten_once(Ls, Rest).

dedupe_keep_first([], []).
dedupe_keep_first([H|T], [H|Out]) :-
    exclude(==(H), T, T1),
    dedupe_keep_first(T1, Out).

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

compile_wam_runtime_header_to_cpp(Options, Code) :-
    compile_wam_runtime_header_body_to_cpp(Options, Body),
    lmdb_sources_from_options(Options, LmdbSources),
    (   LmdbSources == []
    ->  Code = Body
    ;   % Auto-enable LMDB when the codegen has fact sources. The
        % flag has to live in the header so every translation unit
        % (wam_runtime.cpp, generated_program.cpp, main.cpp) sees
        % it -- otherwise the LmdbFactSource class is invisible in
        % one TU and we get link errors. User still needs -llmdb.
        format(string(Code),
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
//
// LMDB FactSource enabled by cpp_fact_sources codegen option.
// Link with -llmdb at compile time.
#define WAM_CPP_ENABLE_LMDB 1

~w', [Body])
    ).

compile_wam_runtime_header_body_to_cpp(_Options,
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
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

} // namespace wam_cpp

namespace std {
// Hash specialization so Instruction::const_map can use Value as a
// key. Only Atom / Integer / Float ever appear in indexing tables
// (the dispatch handler filters out unbound and compounds before
// looking up); other tags hash to zero, which is fine since they
// never reach a switch table. Must be visible at the point where
// Instruction declares its unordered_map<Value, ...> field.
template <> struct hash<wam_cpp::Value> {
    std::size_t operator()(const wam_cpp::Value& v) const noexcept {
        using T = wam_cpp::Value::Tag;
        std::size_t h = static_cast<std::size_t>(v.tag);
        std::size_t k = 0;
        switch (v.tag) {
            case T::Atom:    k = std::hash<std::string>()(v.s); break;
            case T::Integer: k = std::hash<std::int64_t>()(v.i); break;
            case T::Float:   k = std::hash<double>()(v.f); break;
            default:         k = 0; break;
        }
        // Mix tag in so atoms and integers with same numeric/string
        // form hash differently.
        return h ^ (k + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
};
} // namespace std

namespace wam_cpp {

struct Instruction {
    enum class Op {
        GetConstant, GetVariable, GetValue, GetStructure, GetList, GetNil, GetInteger,
        PutConstant, PutVariable, PutValue, PutStructure, PutList,
        UnifyVariable, UnifyValue, UnifyConstant,
        SetVariable, SetValue, SetConstant,
        Call, Execute, Proceed, Fail, Allocate, Deallocate,
        BuiltinCall, CallForeign,
        TryMeElse, RetryMeElse, TrustMe,
        Try, Retry, Trust,                                   // chain ops (#2400)
        Jump, CutIte,
        GetLevel, Cut,                                       // M17 soft cut
        BeginAggregate, EndAggregate,
        SwitchOnConstant, SwitchOnConstantA2,
        SwitchOnStructure, SwitchOnStructureA2,
        SwitchOnTerm, SwitchOnTermA2,
        CatchReturn, NegationReturn, FindallCollect, ConjReturn, DisjAlt,
        IfThenCommit, IfThenElse, AggregateNextGroup, DynamicNextClause,
        SubAtomNext, BodyNext, RetractNext, OutputCaptureReturn,
        CurrentPredNext,
        Nop
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
    //
    // const_map is built once in the factory from const_table and used
    // by SwitchOnConstant{,A2} at runtime — O(1) hash lookup instead
    // of O(N) linear scan. The vector is retained for emission /
    // debugging; iteration order does not matter once we have the map
    // since the WAM compiler emits each constant exactly once.
    std::vector<std::pair<Value, std::size_t>> const_table;
    std::unordered_map<Value, std::size_t> const_map;
    std::vector<std::pair<std::string, std::size_t>> struct_table;
    // Mixed-mode indexing flag: when a bound A1/A2 misses every
    // entry in the switch table, fall through (pc += 1) instead of
    // failing. Set when the predicate has variable-headed clauses
    // that act as a default after the indexed prefix.
    bool no_match_fallthrough = false;

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
    // Indexed-dispatch chain ops (issue #2400).
    static Instruction Try(std::size_t target)
        { Instruction i; i.op = Op::Try; i.target = target; return i; }
    static Instruction Retry(std::size_t target)
        { Instruction i; i.op = Op::Retry; i.target = target; return i; }
    static Instruction Trust(std::size_t target)
        { Instruction i; i.op = Op::Trust; i.target = target; return i; }
    static Instruction Jump(std::size_t target)
        { Instruction i; i.op = Op::Jump; i.target = target; return i; }
    static Instruction CutIte()     { Instruction i; i.op = Op::CutIte;     return i; }
    static Instruction GetLevel(std::string yn)
        { Instruction i; i.op = Op::GetLevel; i.a = std::move(yn); return i; }
    static Instruction Cut(std::string yn)
        { Instruction i; i.op = Op::Cut; i.a = std::move(yn); return i; }
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
    static Instruction SwitchOnConstant(std::vector<std::pair<Value, std::size_t>> table,
                                        bool fall_on_miss = false)
        { Instruction i; i.op = Op::SwitchOnConstant;
          i.const_table = std::move(table);
          i.no_match_fallthrough = fall_on_miss;
          build_const_map(i);
          return i; }
    static Instruction SwitchOnConstantA2(std::vector<std::pair<Value, std::size_t>> table,
                                          bool fall_on_miss = false)
        { Instruction i; i.op = Op::SwitchOnConstantA2;
          i.const_table = std::move(table);
          i.no_match_fallthrough = fall_on_miss;
          build_const_map(i);
          return i; }

private:
    // try_emplace gives first-insert-wins semantics, matching the
    // pre-hash linear-scan behaviour: if the WAM compiler emits two
    // entries with the same key (rare), the first is the target.
    static void build_const_map(Instruction& i) {
        i.const_map.reserve(i.const_table.size());
        for (auto& kv : i.const_table) i.const_map.try_emplace(kv.first, kv.second);
    }
public:
    static Instruction SwitchOnStructure(std::vector<std::pair<std::string, std::size_t>> table)
        { Instruction i; i.op = Op::SwitchOnStructure; i.struct_table = std::move(table); return i; }
    static Instruction SwitchOnStructureA2(std::vector<std::pair<std::string, std::size_t>> table)
        { Instruction i; i.op = Op::SwitchOnStructureA2; i.struct_table = std::move(table); return i; }
    static Instruction SwitchOnTerm(std::vector<std::pair<Value, std::size_t>> consts,
                                    std::vector<std::pair<std::string, std::size_t>> structs,
                                    std::size_t list_pc)
        { Instruction i; i.op = Op::SwitchOnTerm;
          i.const_table = std::move(consts);
          i.struct_table = std::move(structs);
          i.target = list_pc;
          return i; }
    static Instruction SwitchOnTermA2(std::vector<std::pair<Value, std::size_t>> consts,
                                      std::vector<std::pair<std::string, std::size_t>> structs,
                                      std::size_t list_pc)
        { Instruction i; i.op = Op::SwitchOnTermA2;
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
    static Instruction OutputCaptureReturn()
        { Instruction i; i.op = Op::OutputCaptureReturn; return i; }
    static Instruction CurrentPredNext()
        { Instruction i; i.op = Op::CurrentPredNext; return i; }
    static Instruction Nop()
        { Instruction i; i.op = Op::Nop; return i; }
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
    // Set of (env_path, db_name) pairs already loaded via
    // cpp_load_lmdb_fact_source. Used to make the loader idempotent
    // -- a second call for the same pair short-circuits without
    // re-streaming. Per WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md.
    std::set<std::pair<std::string, std::string>> loaded_lmdb_sources;
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
    // Output capture for with_output_to/2. Each frame holds a sink
    // cell (atom(V) / string(V) / codes(V)) and a growing buffer
    // that the I/O builtins append to instead of writing stdout
    // while the frame is on top. The goal''s continuation is
    // output_capture_return_pc, which pops the frame and unifies
    // the buffer with the sink. Backtrack drops the frame at the
    // frame''s base_cp_count (parallel to CatcherFrame).
    struct OutputCaptureFrame {
        CellPtr     sink;          // the atom(V) / string(V) / codes(V)
        std::string buffer;
        std::size_t saved_cp = 0;
        std::size_t base_cp_count = 0;
        std::size_t base_agg_count = 0;
        std::size_t base_catcher_count = 0;
        std::size_t trail_mark = 0;
        std::size_t saved_cut_barrier = 0;
        std::unordered_map<std::string, CellPtr> saved_regs;
        std::vector<ModeFrame> saved_mode_stack;
        std::vector<EnvFrame> saved_env_stack;
    };
    std::vector<OutputCaptureFrame> output_capture_frames;
    std::size_t output_capture_return_pc = 0;
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
        // clause/2 reuses this iterator with these set: instead of
        // removing the matched clause, we keep it; and we unify the
        // clause''s body half with body_pattern (treating fact-only
        // entries as having body = true).
        bool is_clause_only = false;
        CellPtr body_pattern;     // A2 of clause/2 (when is_clause_only).
    };
    std::vector<RetractIterator> retract_iters;
    std::size_t retract_next_pc = 0;
    // current_predicate/1 nondet enumeration. Each iterator holds
    // a pre-filtered list of matching "name/arity" keys (drawn from
    // labels + dynamic_db), the spec''s Name + Arity cells, and an
    // after_pc to return to on each success. On backtrack the alt
    // PC runs CurrentPredNext which pops the just-used CP and
    // re-enters current_pred_try_next.
    struct CurrentPredIterator {
        std::vector<std::string> keys;
        std::size_t next_idx = 0;
        CellPtr name_cell;
        CellPtr arity_cell;
        std::size_t after_pc = 0;
    };
    std::vector<CurrentPredIterator> current_pred_iters;
    std::size_t current_pred_next_pc = 0;
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
    // True iff SwitchOnTerm just jumped directly into the middle/end
    // of a clause chain, bypassing the chain''s entry TryMeElse. The
    // subsequent RetryMeElse / TrustMe consults this flag to decide
    // whether to synthesize a fresh CP (RetryMeElse) or skip the pop
    // (TrustMe), then clears it. Reset on query entry.
    bool indexed_entry = false;

    // File-stream registry. open/3 inserts; close/1 erases; the read
    // / write builtins look up by atom handle. Each entry owns its
    // underlying fstream so RAII closes it if the user forgets. The
    // handle atom is "$stream_N" with a per-WamState monotonically
    // increasing counter.
    struct StreamEntry {
        std::unique_ptr<std::fstream> file;
        bool is_read = false;
    };
    std::unordered_map<std::string, StreamEntry> streams;
    std::uint64_t stream_counter = 0;

    // Cell-aware accessors. get_reg/put_reg keep their Value-shaped API
    // so existing lowered code keeps compiling; get_cell exposes the cell
    // for instructions that need sharing semantics.
    Value   get_reg(const std::string& name) const;
    void    put_reg(const std::string& name, Value v);
    CellPtr get_cell(const std::string& name);
    void    set_cell(const std::string& name, CellPtr c);
    // Create one fresh unbound variable cell and alias it into BOTH
    // registers, so they share identity (binding one binds the other).
    // Used by lowered put_variable: copying a Value into two registers
    // would give two independent cells, losing variable identity.
    void    put_variable_reg(const std::string& a, const std::string& b);
    // Assign a register to a fresh cell holding v (repoint, do NOT mutate
    // the existing cell — any register aliasing this one must not see the
    // new value). Mirrors the interpreter PutConstant. Used by lowered
    // put_constant.
    void    assign_reg(const std::string& name, Value v);

    // Deref through Unbound chains until a concrete value (or a terminal
    // unbound cell) is reached. Returns by value (snapshot).
    Value   deref(const Value& v) const;

    // bind_cell mutates *cell, recording the previous content on the trail.
    void    bind_cell(CellPtr cell, Value v);
    void    trail_binding(const std::string& name); // legacy reg-name trail
    // Undo all bindings recorded since `mark` (trail.size() at the mark).
    // Used by lowered if-then-else when the condition fails, before the
    // else branch runs.
    void    unwind_trail_to(std::size_t mark);

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
    // phrase/2 + phrase/3 meta-call: append [List, Rest] to the
    // goal''s args before dispatching. phrase/2 supplies [] for
    // Rest. Mirrors dispatch_call_meta''s extras-append shape.
    bool    dispatch_phrase_call(bool has_rest, std::size_t after_pc);
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
    // clause/2: dynamic-db iteration without removal, unifying both
    // Head and Body. Shares the RetractIterator infrastructure via
    // the is_clause_only flag.
    bool    dispatch_clause(std::size_t after_pc);
    // current_predicate/1: nondet enumeration over labels +
    // dynamic_db keys, filtered by the (possibly partial) Name/Arity
    // spec in A1. Same dispatch + iterator pattern as retract /
    // clause but with its own iterator type and next-PC slot since
    // the iteration target is strings, not stored cells.
    bool    dispatch_current_predicate(std::size_t after_pc);
    bool    current_pred_try_next();
    bool    retract_try_next();
    // body_next — dispatch the top BodyFrame''s next goal, or pop
    // and proceed to outer after_pc when goals are exhausted.
    bool    body_next();
    // Term parser for term_to_atom/2''s reverse mode and read_term/1.
    // Reads canonical-form syntax matching render()''s output:
    //   integer / float / atom / var / [list] / functor(args).
    // Returns the parsed Value via `out` and the rest-of-input
    // position via `pos` (advances past consumed chars). Skips
    // leading whitespace. Returns false on any parse error.
    bool    parse_term(const std::string& s, std::size_t& pos,
                       CellPtr& out);
    // Internal helpers for parse_term.
    void    parse_skip_ws(const std::string& s, std::size_t& pos);
    bool    parse_atom_or_compound(const std::string& s, std::size_t& pos,
                                   CellPtr& out);
    bool    parse_list(const std::string& s, std::size_t& pos,
                       CellPtr& out);
    bool    parse_number(const std::string& s, std::size_t& pos,
                         CellPtr& out);
    // sub_atom/5 — enumerate (Before, Length) tuples matching the
    // bound constraints. Pre-filters the candidate list at entry,
    // then iterates via the SubAtomIterator + sub_atom_next_pc CP
    // pattern. Returns true on first successful match (with a CP
    // pushed for subsequent matches when any remain).
    bool    dispatch_sub_atom(std::size_t after_pc);
    bool    sub_atom_try_next();
    bool    execute_catch();
    bool    execute_throw();
    // Output emission helper. When output_capture_frames is non-empty,
    // appends to the top frame''s buffer. Otherwise writes to stdout.
    // All I/O builtins (write/nl/writeln/print/display/tab/
    // write_canonical/format) route through this so with_output_to/2
    // can intercept their output.
    void    emit_output(const std::string& s);
    void    emit_output_char(char c);

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

// ----------------------------------------------------------------------
// LMDB FactSource (v1) -- gated by WAM_CPP_ENABLE_LMDB at build time.
//
// v1 mirrors the C target: eager load of the entire DB into the
// existing dynamic_db at startup, UTF-8 atom-only encoding, arity
// fixed at 2. Per WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md.
//
// When WAM_CPP_ENABLE_LMDB is not defined, cpp_load_lmdb_fact_source
// is still declared (matching the C target stub pattern) but the
// definition returns false without touching the dynamic_db -- the
// generated program compiles fine, the LMDB load just no-ops.
// ----------------------------------------------------------------------

#ifdef WAM_CPP_ENABLE_LMDB
#include <lmdb.h>

class LmdbFactSource {
public:
    // Open env at env_path. db_name == nullptr uses the unnamed
    // default DB. Throws std::runtime_error on open failure --
    // load-time errors are surfaced to the caller, not silenced.
    LmdbFactSource(const std::string& env_path, const char* db_name);
    ~LmdbFactSource();

    LmdbFactSource(const LmdbFactSource&) = delete;
    LmdbFactSource& operator=(const LmdbFactSource&) = delete;

    // Iterate every (key, value) pair once. Caller-supplied sink
    // sees borrowed string_views valid only for that callback.
    void stream_all(
        const std::function<void(std::string_view,
                                 std::string_view)>& sink);

    // Read the optional __meta__ sub-DB. Per
    // WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md v1, schema info lives in
    // a named __meta__ sub-DB (unprefixed keys when the data DB
    // is the default unnamed one; prefixed "<db_name>:" when the
    // data DB is named). LmdbMeta::present is false when the
    // __meta__ sub-DB is absent entirely -- that case is treated
    // as a warning by the loader, not a hard error, for
    // transitional accommodation of LMDB files built before this
    // PR.
    struct Meta {
        bool present = false;
        int schema_version = 0;
        std::string predicate;            // e.g. "edge/2"
        std::vector<std::string> columns; // e.g. ["child", "parent"]
    };
    Meta read_meta(const char* db_name);

private:
    MDB_env* env_ = nullptr;
    MDB_dbi  dbi_ = 0;
    bool dbi_open_ = false;
};
#endif // WAM_CPP_ENABLE_LMDB

// Policy options driving load-time enforcement. Populated by the
// codegen from relation_policy/2 declarations and per-source
// overrides via get_effective_policy/4 (PR #2325). v1 enforcement
// is value-uniqueness only -- LMDB keys are already unique by
// construction without DUPSORT, so the meaningful uniqueness
// check is on the value column.
//
// Defined outside the WAM_CPP_ENABLE_LMDB gate so generated code
// can build an instance regardless of build flags. The stub
// cpp_load_lmdb_fact_source ignores it.
struct LmdbLoadOptions {
    bool unique_check = false;
    enum class OnDup {
        keep_all   = 0,
        throw_     = 1,
        warn       = 2,
        overwrite  = 3,
        first_wins = 4
    } on_duplicate = OnDup::keep_all;

    // Sort keys applied to the staged rows AFTER stream_all and
    // BEFORE commit to dynamic_db. Lexicographic over the listed
    // keys; empty vector means "no sort" (LMDB natural iteration
    // order wins). column is 1-based; v1 only supports arity 2 so
    // values are 1 or 2. The codegen omits trivially-satisfied
    // order specs (e.g. asc by arg1, which LMDB gives for free)
    // so sort_keys is non-empty only when actual reordering is
    // needed.
    struct SortKey {
        int column;        // 1-based column index
        bool ascending;
    };
    std::vector<SortKey> sort_keys;
};

// Always declared; the implementation no-ops when LMDB is not
// compiled in. Returns true on successful load (or already loaded);
// false on error or when LMDB support is absent.
//
// Idempotent: a second call for the same (env_path, db_name) pair
// is a no-op, matching the design doc resolution.
bool cpp_load_lmdb_fact_source(
    WamState& vm,
    const std::string& functor_arity_key,
    const std::string& env_path,
    const char* db_name,
    const LmdbLoadOptions& opts = {});

} // namespace wam_cpp

using wam_cpp::Value;
using wam_cpp::Instruction;
using wam_cpp::WamState;
using wam_cpp::Program;
using wam_cpp::LmdbLoadOptions;
using wam_cpp::cpp_load_lmdb_fact_source;

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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

namespace wam_cpp {

// Forward declarations for free functions defined later in this TU.
static int standard_order_cmp(const Value& a, const Value& b);

// Process-wide PRNG for the random/* family. Mersenne Twister 64-bit
// with the standard default seed (5489) so unseeded runs are
// reproducible; set_random(seed(N)) replaces it, set_random(seed(random))
// re-seeds from std::random_device.
static std::mt19937_64& cpp_global_rng() {
    static std::mt19937_64 rng(5489);
    return rng;
}

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

void WamState::put_variable_reg(const std::string& a, const std::string& b) {
    CellPtr c = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
    set_cell(a, c);
    set_cell(b, c);
}

void WamState::assign_reg(const std::string& name, Value v) {
    set_cell(name, make_cell(std::move(v)));
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

void WamState::unwind_trail_to(std::size_t mark) {
    while (trail.size() > mark) {
        TrailEntry t = std::move(trail.back());
        trail.pop_back();
        *t.cell = std::move(t.prev);
    }
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
            // SWI-style arithmetic constants. pi/e are doubles;
            // inf/nan come from <cmath> + <limits>.
            if (v.s == "pi") return Value::Float(3.14159265358979323846);
            if (v.s == "e")  return Value::Float(2.71828182845904523536);
            if (v.s == "inf" || v.s == "infinite")
                return Value::Float(std::numeric_limits<double>::infinity());
            if (v.s == "nan")
                return Value::Float(std::nan(""));
            if (v.s == "max_tagged_integer")
                return Value::Integer(std::numeric_limits<std::int64_t>::max());
            if (v.s == "min_tagged_integer")
                return Value::Integer(std::numeric_limits<std::int64_t>::min());
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
            // Unary functions. -/1 is the historical entry; the rest
            // (abs, sign, sqrt, trunc/floor/ceil/round, bitnot \\/1)
            // follow SWI semantics with one wrinkle: trunc/floor/ceil/round
            // always return Integer (matches SWI even when fed a
            // float). +/1 is identity, useful for term-level
            // arithmetic where you parse "(+ X)" as a positive.
            if (v.args.size() == 1) {
                Value a = eval_arith(v.args[0], ok);
                if (!ok) return Value{};
                auto as_d1 = [](const Value& w){
                    return w.tag == Value::Tag::Float ? w.f : (double)w.i;
                };
                if (v.s == "-/1") {
                    if (a.tag == Value::Tag::Integer) return Value::Integer(-a.i);
                    return Value::Float(-a.f);
                }
                if (v.s == "+/1") return a;
                if (v.s == "abs/1") {
                    if (a.tag == Value::Tag::Integer)
                        return Value::Integer(a.i < 0 ? -a.i : a.i);
                    return Value::Float(std::fabs(a.f));
                }
                if (v.s == "sign/1") {
                    if (a.tag == Value::Tag::Integer)
                        return Value::Integer(a.i > 0 ? 1 : (a.i < 0 ? -1 : 0));
                    return Value::Float(a.f > 0 ? 1.0
                                       : (a.f < 0 ? -1.0 : 0.0));
                }
                if (v.s == "sqrt/1") return Value::Float(std::sqrt(as_d1(a)));
                // Trig + transcendentals: all promote Integer to
                // double and always return Float.
                if (v.s == "sin/1") return Value::Float(std::sin(as_d1(a)));
                if (v.s == "cos/1") return Value::Float(std::cos(as_d1(a)));
                if (v.s == "tan/1") return Value::Float(std::tan(as_d1(a)));
                if (v.s == "asin/1") return Value::Float(std::asin(as_d1(a)));
                if (v.s == "acos/1") return Value::Float(std::acos(as_d1(a)));
                if (v.s == "atan/1") return Value::Float(std::atan(as_d1(a)));
                if (v.s == "sinh/1") return Value::Float(std::sinh(as_d1(a)));
                if (v.s == "cosh/1") return Value::Float(std::cosh(as_d1(a)));
                if (v.s == "tanh/1") return Value::Float(std::tanh(as_d1(a)));
                if (v.s == "asinh/1") return Value::Float(std::asinh(as_d1(a)));
                if (v.s == "acosh/1") return Value::Float(std::acosh(as_d1(a)));
                if (v.s == "atanh/1") return Value::Float(std::atanh(as_d1(a)));
                if (v.s == "log/1") return Value::Float(std::log(as_d1(a)));
                if (v.s == "exp/1") return Value::Float(std::exp(as_d1(a)));
                // Integer bit-count functions. lsb/msb of 0 throw a
                // domain error per SWI (no defined bit position).
                if (v.s == "popcount/1") {
                    if (a.tag != Value::Tag::Integer)
                        { ok = false; return Value{}; }
                    std::uint64_t u = (std::uint64_t)a.i;
                    return Value::Integer(__builtin_popcountll(u));
                }
                if (v.s == "lsb/1") {
                    if (a.tag != Value::Tag::Integer || a.i == 0)
                        { ok = false; return Value{}; }
                    std::uint64_t u = (std::uint64_t)a.i;
                    return Value::Integer(__builtin_ctzll(u));
                }
                if (v.s == "msb/1") {
                    if (a.tag != Value::Tag::Integer || a.i == 0)
                        { ok = false; return Value{}; }
                    std::uint64_t u = (std::uint64_t)a.i;
                    return Value::Integer(63 - __builtin_clzll(u));
                }
                if (v.s == "truncate/1")
                    return Value::Integer((std::int64_t)std::trunc(as_d1(a)));
                if (v.s == "floor/1")
                    return Value::Integer((std::int64_t)std::floor(as_d1(a)));
                if (v.s == "ceiling/1")
                    return Value::Integer((std::int64_t)std::ceil(as_d1(a)));
                if (v.s == "round/1")
                    return Value::Integer((std::int64_t)std::round(as_d1(a)));
                if (v.s == "\\\\/1") { // bitwise NOT
                    if (a.tag != Value::Tag::Integer) { ok = false; return Value{}; }
                    return Value::Integer(~a.i);
                }
                ok = false; return Value{};
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
            // Min / max preserve the wider numeric type.
            if (v.s == "min/2") {
                if (either_float) return Value::Float(std::fmin(as_d(a), as_d(b)));
                return Value::Integer(std::min(a.i, b.i));
            }
            if (v.s == "max/2") {
                if (either_float) return Value::Float(std::fmax(as_d(a), as_d(b)));
                return Value::Integer(std::max(a.i, b.i));
            }
            // **/2 always returns float (SWI); ^/2 stays integer when
            // both args are int and the exponent is non-negative.
            if (v.s == "**/2") {
                return Value::Float(std::pow(as_d(a), as_d(b)));
            }
            // atan2(Y, X) -- standard library order.
            if (v.s == "atan2/2") {
                return Value::Float(std::atan2(as_d(a), as_d(b)));
            }
            // log(Base, X) -- log of X to base Base. Standard
            // identity: log(X) / log(Base).
            if (v.s == "log/2") {
                double base = as_d(a);
                double x = as_d(b);
                return Value::Float(std::log(x) / std::log(base));
            }
            // copysign(X, Y) returns |X| with the sign of Y; always Float.
            if (v.s == "copysign/2") {
                return Value::Float(std::copysign(as_d(a), as_d(b)));
            }
            if (v.s == "^/2") {
                if (either_float || b.i < 0) {
                    return Value::Float(std::pow(as_d(a), as_d(b)));
                }
                std::int64_t result = 1, base = a.i, exp = b.i;
                while (exp > 0) {
                    if (exp & 1) result *= base;
                    base *= base;
                    exp >>= 1;
                }
                return Value::Integer(result);
            }
            // Integer-only functions below. Reject floats outright.
            if (v.s == "gcd/2") {
                if (either_float) { ok = false; return Value{}; }
                std::int64_t x = a.i < 0 ? -a.i : a.i;
                std::int64_t y = b.i < 0 ? -b.i : b.i;
                while (y) { std::int64_t t = y; y = x % y; x = t; }
                return Value::Integer(x);
            }
            if (v.s == "rem/2") {
                // C99/C++11 ``%'' follows dividend''s sign; matches
                // ISO rem. (mod follows divisor''s sign -- different.)
                if (either_float) { ok = false; return Value{}; }
                if (b.i == 0) { ok = false; return Value{}; }
                return Value::Integer(a.i % b.i);
            }
            if (v.s == "/\\\\/2") { // bitwise AND
                if (either_float) { ok = false; return Value{}; }
                return Value::Integer(a.i & b.i);
            }
            if (v.s == "\\\\//2") { // bitwise OR
                if (either_float) { ok = false; return Value{}; }
                return Value::Integer(a.i | b.i);
            }
            if (v.s == "xor/2") {
                if (either_float) { ok = false; return Value{}; }
                return Value::Integer(a.i ^ b.i);
            }
            if (v.s == ">>/2") {
                if (either_float) { ok = false; return Value{}; }
                return Value::Integer(a.i >> b.i);
            }
            if (v.s == "<</2") {
                if (either_float) { ok = false; return Value{}; }
                return Value::Integer(a.i << b.i);
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
        || op == "string_codes/2" || op == "string_chars/2"
        || op == "number_codes/2") {
        bool is_codes = (op != "atom_chars/2" && op != "string_chars/2");
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
    if (op == "atom_length/2" || op == "string_length/2") {
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

    // ---- atom_string/2 ----------------------------------------------
    // Bidirectional. In our runtime atoms and strings are
    // interchangeable, so this is effectively a unify-or-coerce
    // helper: (+Atom, -String) renders A1 and unifies; (-Atom,
    // +String) parses A2 as an atom and unifies A1.
    if (op == "atom_string/2") {
        Value a = deref(*get_cell("A1"));
        Value b = deref(*get_cell("A2"));
        if (!a.is_unbound()) {
            std::string s;
            if (a.tag == Value::Tag::Atom) s = a.s;
            else if (a.tag == Value::Tag::Integer) s = std::to_string(a.i);
            else if (a.tag == Value::Tag::Float) s = render(a);
            else return false;
            Value sv = Value::Atom(s);
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, sv); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(sv))) return false;
            pc += 1; return true;
        }
        if (b.is_unbound()) return false;
        std::string s;
        if (b.tag == Value::Tag::Atom) s = b.s;
        else if (b.tag == Value::Tag::Integer) s = std::to_string(b.i);
        else if (b.tag == Value::Tag::Float) s = render(b);
        else return false;
        Value av = Value::Atom(s);
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, av); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(av))) return false;
        pc += 1; return true;
    }

    // ---- string_concat/3 -- alias for atom_concat/3 -----------------
    if (op == "string_concat/3") {
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

    // ---- number_chars/2 ---------------------------------------------
    // Bidirectional. Forward (A1 bound to a number): render A1 and
    // build a list of single-char atoms. Reverse (A2 bound to a
    // chars list): join the chars and parse as int/float.
    if (op == "number_chars/2") {
        Value a1 = deref(*get_cell("A1"));
        if (!a1.is_unbound()) {
            std::string buf;
            if (a1.tag == Value::Tag::Integer) buf = std::to_string(a1.i);
            else if (a1.tag == Value::Tag::Float) buf = render(a1);
            else return false;
            CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
            for (auto it = buf.rbegin(); it != buf.rend(); ++it) {
                CellPtr head = std::make_shared<Cell>(
                    Value::Atom(std::string(1, *it)));
                std::vector<CellPtr> ca;
                ca.push_back(head);
                ca.push_back(list);
                list = std::make_shared<Cell>(
                    Value::Compound("[|]/2", std::move(ca)));
            }
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, *list); pc += 1; return true; }
            if (!unify_cells(tgt, list)) return false;
            pc += 1; return true;
        }
        std::string buf;
        CellPtr lc = get_cell("A2");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            Value hv = deref(*lv.args[0]);
            if (hv.tag != Value::Tag::Atom || hv.s.size() != 1) return false;
            buf.push_back(hv.s[0]);
            lc = lv.args[1];
        }
        if (buf.empty()) return false;
        Value result;
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
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
        pc += 1; return true;
    }

    // ---- atom_number/2 ----------------------------------------------
    // atom_number(?Atom, ?Number) bidirectional atom/number conversion.
    // Forward (A1 bound): parse atom text as int first, then float.
    // Fail (NOT throw) on unparseable input -- thats the difference
    // between atom_number/2 and number_codes/2 (which throws).
    // Reverse (A2 bound): render the number as an atom.
    if (op == "atom_number/2") {
        Value a1 = deref(*get_cell("A1"));
        Value a2 = deref(*get_cell("A2"));
        if (!a1.is_unbound()) {
            if (a1.tag != Value::Tag::Atom) return false;
            const std::string& buf = a1.s;
            if (buf.empty()) return false;
            Value result;
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
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
            pc += 1; return true;
        }
        if (a2.is_unbound()) return false;
        std::string s;
        if (a2.tag == Value::Tag::Integer) s = std::to_string(a2.i);
        else if (a2.tag == Value::Tag::Float) s = render(a2);
        else return false;
        Value av = Value::Atom(s);
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, av); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(av))) return false;
        pc += 1; return true;
    }

    // ---- atomic_list_concat/2, /3 -----------------------------------
    // atomic_list_concat(+List, ?Atom)              concat with no sep
    // atomic_list_concat(?List, +Separator, ?Atom)  with separator
    //   (+, +, ?)  join List with Separator into Atom
    //   (-, +, +)  split Atom by Separator into List
    if (op == "atomic_list_concat/2") {
        // Walk A1 as a list of atomics; render each; concatenate.
        std::string buf;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            Value hv = deref(*lv.args[0]);
            if (hv.tag == Value::Tag::Atom) buf += hv.s;
            else if (hv.tag == Value::Tag::Integer) buf += std::to_string(hv.i);
            else if (hv.tag == Value::Tag::Float) buf += render(hv);
            else return false;
            lc = lv.args[1];
        }
        Value result = Value::Atom(buf);
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
        pc += 1; return true;
    }
    if (op == "atomic_list_concat/3") {
        Value a1 = deref(*get_cell("A1"));
        Value a2 = deref(*get_cell("A2"));
        Value a3 = deref(*get_cell("A3"));
        if (a2.is_unbound()) return false;
        std::string sep;
        if (a2.tag == Value::Tag::Atom) sep = a2.s;
        else if (a2.tag == Value::Tag::Integer) sep = std::to_string(a2.i);
        else return false;
        if (!a1.is_unbound()) {
            // Join mode: render each element + separator.
            std::string buf;
            CellPtr lc = get_cell("A1");
            bool first = true;
            for (;;) {
                Value lv = deref(*lc);
                if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
                if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                    || lv.args.size() != 2) return false;
                if (!first) buf += sep;
                first = false;
                Value hv = deref(*lv.args[0]);
                if (hv.tag == Value::Tag::Atom) buf += hv.s;
                else if (hv.tag == Value::Tag::Integer) buf += std::to_string(hv.i);
                else if (hv.tag == Value::Tag::Float) buf += render(hv);
                else return false;
                lc = lv.args[1];
            }
            Value result = Value::Atom(buf);
            CellPtr tgt = get_cell("A3");
            if (tgt->is_unbound()) { bind_cell(tgt, result); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(result))) return false;
            pc += 1; return true;
        }
        // Split mode: A1 unbound, A3 bound. Walk A3 splitting on `sep`.
        if (a3.is_unbound()) return false;
        std::string src;
        if (a3.tag == Value::Tag::Atom) src = a3.s;
        else if (a3.tag == Value::Tag::Integer) src = std::to_string(a3.i);
        else return false;
        if (sep.empty()) return false; // can''t split on empty sep
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (start <= src.size()) {
            std::size_t hit = src.find(sep, start);
            if (hit == std::string::npos) {
                parts.push_back(src.substr(start));
                break;
            }
            parts.push_back(src.substr(start, hit - start));
            start = hit + sep.size();
        }
        // Build list of atoms.
        CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
            CellPtr head = std::make_shared<Cell>(Value::Atom(*it));
            std::vector<CellPtr> ca;
            ca.push_back(head);
            ca.push_back(list);
            list = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, *list); pc += 1; return true; }
        if (!unify_cells(tgt, list)) return false;
        pc += 1; return true;
    }

    // ---- atom_to_term/3 ---------------------------------------------
    // atom_to_term(+Atom, -Term, -Bindings). Parse Atom as a term;
    // Bindings is a list of ''Name''=Var pairs. Our parser doesn''t
    // track source variable names (each var becomes a fresh unbound
    // cell), so Bindings is unified with [] -- adequate for the
    // common pattern of round-tripping ground terms.
    if (op == "atom_to_term/3") {
        Value a = deref(*get_cell("A1"));
        if (a.is_unbound()) return false;
        std::string src;
        if (a.tag == Value::Tag::Atom) src = a.s;
        else if (a.tag == Value::Tag::Integer) src = std::to_string(a.i);
        else return false;
        std::size_t pos = 0;
        CellPtr parsed;
        if (!parse_term(src, pos, parsed)) return false;
        parse_skip_ws(src, pos);
        if (pos != src.size()) return false;
        CellPtr term_tgt = get_cell("A2");
        if (term_tgt->is_unbound()) { bind_cell(term_tgt, *parsed); }
        else if (!unify_cells(term_tgt, parsed)) return false;
        Value nil = Value::Atom("[]");
        CellPtr bind_tgt = get_cell("A3");
        if (bind_tgt->is_unbound()) bind_cell(bind_tgt, nil);
        else if (!unify_cells(bind_tgt, std::make_shared<Cell>(nil))) return false;
        pc += 1; return true;
    }

    // ---- split_string/4 ---------------------------------------------
    // split_string(+String, +SepChars, +PadChars, -SubStrings).
    // Walks String left-to-right, splitting on any char in SepChars.
    // Each resulting substring has any leading/trailing chars in
    // PadChars stripped. Adjacent separators produce empty
    // substrings. Empty SepChars means "no splits, just pad".
    // ---- term_to_atom/2 ---------------------------------------------
    // term_to_atom(?Term, ?Atom) — bidirectional canonical-form
    // serialisation. (+Term, ?Atom): render Term and unify Atom
    // with the resulting atom. (-Term, +Atom): parse Atom using
    // the canonical-form reader (parse_term) and unify Term with
    // the parsed value. (+Term, +Atom): render Term and require
    // equality with Atom. Variables in a parsed Term are fresh.
    if (op == "term_to_atom/2") {
        Value t = deref(*get_cell("A1"));
        Value a = deref(*get_cell("A2"));
        if (!t.is_unbound()) {
            std::string rendered = render(t);
            Value av = Value::Atom(rendered);
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, av); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(av))) return false;
            pc += 1; return true;
        }
        // Reverse mode: A2 must be an atom (or atom-like).
        std::string src;
        if (a.tag == Value::Tag::Atom) src = a.s;
        else if (a.tag == Value::Tag::Integer) src = std::to_string(a.i);
        else if (a.tag == Value::Tag::Float) src = render(a);
        else return false;
        std::size_t pos = 0;
        CellPtr parsed;
        if (!parse_term(src, pos, parsed)) return false;
        parse_skip_ws(src, pos);
        if (pos != src.size()) return false; // trailing garbage
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, *parsed); pc += 1; return true; }
        if (!unify_cells(tgt, parsed)) return false;
        pc += 1; return true;
    }

    // ---- read/1, read_term/1 ----------------------------------------
    // Read a term from stdin, terminated by `.` (followed by EOF or
    // whitespace). Bind A1 to the parsed term. On EOF before any
    // input, bind A1 to the atom `end_of_file` (per ISO). Uses
    // parse_term (added in #2189) so the syntax matches what
    // term_to_atom/2 produces.
    if (op == "read/1" || op == "read_term/1") {
        std::string buf;
        bool got_period = false;
        while (true) {
            int ch = std::getc(stdin);
            if (ch == EOF) break;
            buf.push_back(static_cast<char>(ch));
            // Terminator: a `.` followed by whitespace or EOF.
            if (ch == ''.'') {
                int peek = std::getc(stdin);
                if (peek == EOF) {
                    buf.pop_back(); // drop the period
                    got_period = true;
                    break;
                }
                if (peek == '' '' || peek == ''\\t'' || peek == ''\\n'') {
                    buf.pop_back();
                    got_period = true;
                    break;
                }
                // Not a real terminator (e.g. floating-point digit). Push back.
                std::ungetc(peek, stdin);
            }
        }
        CellPtr tgt = get_cell("A1");
        if (buf.empty() && !got_period) {
            // EOF before any input → end_of_file.
            Value eof = Value::Atom("end_of_file");
            if (tgt->is_unbound()) { bind_cell(tgt, eof); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(eof))) return false;
            pc += 1; return true;
        }
        std::size_t pos = 0;
        CellPtr parsed;
        if (!parse_term(buf, pos, parsed)) return false;
        parse_skip_ws(buf, pos);
        if (pos != buf.size()) return false; // trailing non-ws garbage
        if (tgt->is_unbound()) { bind_cell(tgt, *parsed); pc += 1; return true; }
        if (!unify_cells(tgt, parsed)) return false;
        pc += 1; return true;
    }

    // ---- get_char/1, get_code/1, peek_char/1 ------------------------
    // Single-char input. On EOF, get_char/peek_char bind A1 to atom
    // `end_of_file`; get_code binds A1 to -1 (per ISO).
    if (op == "get_char/1" || op == "peek_char/1") {
        int ch = std::getc(stdin);
        if (op == "peek_char/1" && ch != EOF) std::ungetc(ch, stdin);
        Value v;
        if (ch == EOF) v = Value::Atom("end_of_file");
        else v = Value::Atom(std::string(1,
            static_cast<char>(static_cast<unsigned char>(ch))));
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, v); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(v))) return false;
        pc += 1; return true;
    }
    if (op == "get_code/1") {
        int ch = std::getc(stdin);
        Value v = Value::Integer(static_cast<std::int64_t>(ch)); // -1 on EOF
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, v); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(v))) return false;
        pc += 1; return true;
    }

    // ---- put_char/1, put_code/1 -------------------------------------
    // Single-char output. Routes through emit_output_char so the byte
    // is captured by an enclosing with_output_to/2.
    if (op == "put_char/1") {
        Value v = deref(*get_cell("A1"));
        if (v.tag != Value::Tag::Atom || v.s.size() != 1) return false;
        emit_output_char(v.s[0]);
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "put_code/1") {
        Value v = deref(*get_cell("A1"));
        if (v.tag != Value::Tag::Integer || v.i < 0 || v.i > 255)
            return false;
        emit_output_char(static_cast<char>(
            static_cast<unsigned char>(v.i)));
        std::fflush(stdout);
        pc += 1; return true;
    }

    if (op == "split_string/4") {
        auto read_str = [&](CellPtr c, std::string& out) -> bool {
            Value v = deref(*c);
            if (v.tag == Value::Tag::Atom) { out = v.s; return true; }
            if (v.tag == Value::Tag::Integer) {
                out = std::to_string(v.i); return true;
            }
            if (v.tag == Value::Tag::Float) {
                out = render(v); return true;
            }
            return false;
        };
        std::string str, seps, pads;
        if (!read_str(get_cell("A1"), str)) return false;
        if (!read_str(get_cell("A2"), seps)) return false;
        if (!read_str(get_cell("A3"), pads)) return false;
        auto in_chars = [](char c, const std::string& set) -> bool {
            return set.find(c) != std::string::npos;
        };
        // Split.
        std::vector<std::string> parts;
        std::string current;
        for (char c : str) {
            if (in_chars(c, seps)) {
                parts.push_back(std::move(current));
                current.clear();
            } else {
                current.push_back(c);
            }
        }
        parts.push_back(std::move(current));
        // Strip pad chars from each part''s ends.
        for (auto& p : parts) {
            std::size_t start = 0;
            while (start < p.size() && in_chars(p[start], pads)) ++start;
            std::size_t end = p.size();
            while (end > start && in_chars(p[end - 1], pads)) --end;
            p = p.substr(start, end - start);
        }
        // Build the result list.
        CellPtr result = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
            CellPtr head = std::make_shared<Cell>(Value::Atom(*it));
            std::vector<CellPtr> cons_args;
            cons_args.push_back(head);
            cons_args.push_back(result);
            result = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(cons_args)));
        }
        CellPtr tgt = get_cell("A4");
        if (tgt->is_unbound()) { bind_cell(tgt, *result); pc += 1; return true; }
        if (!unify_cells(tgt, result)) return false;
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

    // ---- string_code/3 ----------------------------------------------
    // string_code(+Index, +String, -Code) -- get the Code (integer)
    // at 1-based Index in String. Out-of-range fails. Index 1 returns
    // the first char''s code; SWI uses 1-based indexing here.
    if (op == "string_code/3") {
        Value iv = deref(*get_cell("A1"));
        Value sv = deref(*get_cell("A2"));
        if (iv.tag != Value::Tag::Integer) return false;
        if (sv.tag != Value::Tag::Atom) return false;
        if (iv.i < 1 || (std::size_t)iv.i > sv.s.size()) return false;
        Value code = Value::Integer(
            static_cast<std::int64_t>(
                static_cast<unsigned char>(sv.s[iv.i - 1])));
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, code); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(code))) return false;
        pc += 1; return true;
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

    // ---- sort/4 -----------------------------------------------------
    // sort(+Key, +Order, +List, -Sorted)
    //   Key:   0 for whole-term, N>0 for the Nth arg of a compound.
    //   Order: @<  ascending,  dup keys removed.
    //          @=< ascending,  dup keys kept (stable).
    //          @>  descending, dup keys removed.
    //          @>= descending, dup keys kept (stable).
    if (op == "sort/4") {
        Value kv = *get_cell("A1");
        Value ov = *get_cell("A2");
        if (kv.tag != Value::Tag::Integer)
            return throw_iso_error(make_type_error("integer", kv));
        if (ov.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", ov));
        std::int64_t key_pos = kv.i;
        if (key_pos < 0)
            return throw_iso_error(make_domain_error("not_less_than_zero", kv));
        bool ascending, dedup;
        if      (ov.s == "@<")  { ascending = true;  dedup = true;  }
        else if (ov.s == "@=<") { ascending = true;  dedup = false; }
        else if (ov.s == "@>")  { ascending = false; dedup = true;  }
        else if (ov.s == "@>=") { ascending = false; dedup = false; }
        else return throw_iso_error(make_domain_error("order", ov));
        std::vector<CellPtr> items;
        CellPtr lc = get_cell("A3");
        for (;;) {
            Value lv = deref(*lc);
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            items.push_back(lv.args[0]);
            lc = lv.args[1];
        }
        auto extract_key = [key_pos](const CellPtr& c) -> CellPtr {
            if (key_pos == 0) return c;
            if (c->tag == Value::Tag::Compound
                && (std::size_t)key_pos <= c->args.size())
                return c->args[key_pos - 1];
            return c; // fall back to whole-term comparison
        };
        std::stable_sort(items.begin(), items.end(),
            [&](const CellPtr& a, const CellPtr& b) {
                int cmp = standard_order_cmp(*extract_key(a), *extract_key(b));
                return ascending ? (cmp < 0) : (cmp > 0);
            });
        if (dedup) {
            items.erase(std::unique(items.begin(), items.end(),
                [&](const CellPtr& a, const CellPtr& b) {
                    return standard_order_cmp(*extract_key(a),
                                              *extract_key(b)) == 0;
                }), items.end());
        }
        CellPtr result = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            std::vector<CellPtr> cons_args;
            cons_args.push_back(*it);
            cons_args.push_back(result);
            result = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(cons_args)));
        }
        CellPtr tgt = get_cell("A4");
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
            } else if (v.tag == Value::Tag::Integer
                       || v.tag == Value::Tag::Float) {
                // Numbers are their own functor with arity 0. Distinct
                // from the atom case because Name unifies as the number
                // itself, not Value::Atom(render(v)).
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
        // With Arity == 0, Name may be any atomic (atom or number);
        // T is bound directly to Name. With Arity > 0, Name must be
        // an atom (compound terms can''t have numeric functors).
        Value name_v = *get_cell("A2");
        Value ar_v   = *get_cell("A3");
        if (ar_v.tag != Value::Tag::Integer) return false;
        if (ar_v.i == 0) {
            if (name_v.is_unbound()) return false;
            if (name_v.tag == Value::Tag::Compound) return false;
            bind_cell(t, name_v); pc += 1; return true;
        }
        if (name_v.tag != Value::Tag::Atom) return false;
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

    // ---- term_variables/2 ------------------------------------------
    // Walk +Term and collect a list of the unique unbound cells in
    // left-to-right order of first occurrence. Each shared variable
    // appears once. Identity is by CellPtr (shared_ptr address).
    if (op == "term_variables/2") {
        std::vector<CellPtr> vars;
        std::unordered_set<Value*> seen;
        std::function<void(CellPtr)> walk = [&](CellPtr c) {
            if (!c) return;
            if (c->is_unbound()) {
                if (seen.insert(c.get()).second) vars.push_back(c);
                return;
            }
            if (c->tag == Value::Tag::Compound) {
                for (auto& a : c->args) walk(a);
            }
        };
        walk(get_cell("A1"));
        CellPtr list = std::make_shared<Value>(Value::Atom("[]"));
        for (auto it = vars.rbegin(); it != vars.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(*it);
            ca.push_back(list);
            list = std::make_shared<Value>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A2"), list)) return false;
        pc += 1; return true;
    }

    // ---- numbervars/3 ----------------------------------------------
    // numbervars(+Term, +Start, -End). For each free variable in Term
    // (left-to-right first occurrence), bind it to ''$VAR''(N) where N
    // starts at Start and increments. End is one past the last N.
    if (op == "numbervars/3") {
        Value sv = *get_cell("A2");
        if (sv.tag != Value::Tag::Integer) return false;
        std::int64_t n = sv.i;
        std::unordered_set<Value*> seen;
        std::function<void(CellPtr)> walk = [&](CellPtr c) {
            if (!c) return;
            if (c->is_unbound()) {
                if (seen.insert(c.get()).second) {
                    std::vector<CellPtr> ca;
                    ca.push_back(std::make_shared<Value>(Value::Integer(n++)));
                    bind_cell(c, Value::Compound("$VAR/1", std::move(ca)));
                }
                return;
            }
            if (c->tag == Value::Tag::Compound) {
                for (auto& a : c->args) walk(a);
            }
        };
        walk(get_cell("A1"));
        Value end_v = Value::Integer(n);
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, end_v); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Value>(end_v))) return false;
        pc += 1; return true;
    }

    // ---- =@=/2 and \\=@=/2 (variant equivalence) -------------------
    // Two terms are variant if they''re structurally identical
    // modulo a bijective variable renaming. Walk both in parallel
    // maintaining two maps (A->B and B->A) so each unbound cell on
    // one side maps to exactly one on the other.
    if (op == "=@=/2" || op == "\\\\=@=/2") {
        std::unordered_map<Value*, Value*> ab, ba;
        std::function<bool(CellPtr,CellPtr)> variant = [&](CellPtr a, CellPtr b) -> bool {
            if (!a || !b) return false;
            bool au = a->is_unbound(), bu = b->is_unbound();
            if (au && bu) {
                Value* ap = a.get(); Value* bp = b.get();
                auto it1 = ab.find(ap);
                auto it2 = ba.find(bp);
                if (it1 == ab.end() && it2 == ba.end()) {
                    ab[ap] = bp; ba[bp] = ap; return true;
                }
                if (it1 != ab.end() && it2 != ba.end()
                    && it1->second == bp && it2->second == ap) return true;
                return false;
            }
            if (au != bu) return false;
            if (a->tag != b->tag) return false;
            switch (a->tag) {
                case Value::Tag::Atom:    return a->s == b->s;
                case Value::Tag::Integer: return a->i == b->i;
                case Value::Tag::Float:   return a->f == b->f;
                case Value::Tag::Compound:
                    if (a->s != b->s || a->args.size() != b->args.size())
                        return false;
                    for (std::size_t k = 0; k < a->args.size(); ++k)
                        if (!variant(a->args[k], b->args[k])) return false;
                    return true;
                default: return false;
            }
        };
        bool eq = variant(get_cell("A1"), get_cell("A2"));
        bool want = (op == "=@=/2");
        if (eq != want) return false;
        pc += 1; return true;
    }

    // ---- unifiable/3 -----------------------------------------------
    // unifiable(?Term1, ?Term2, -Bindings). Succeeds iff Term1 and
    // Term2 unify, and unifies A3 with the resulting bindings as a
    // list of ''=''(Var, Value) pairs. Does NOT leave the actual
    // bindings in place -- we undo via the trail after recording.
    if (op == "unifiable/3") {
        std::size_t mark = trail.size();
        if (!unify_cells(get_cell("A1"), get_cell("A2"))) {
            // unify_cells may have pushed partial bindings before failing.
            while (trail.size() > mark) {
                *trail.back().cell = trail.back().prev;
                trail.pop_back();
            }
            return false;
        }
        // Collect bindings list while bindings are still in effect.
        std::vector<CellPtr> pairs;
        for (std::size_t k = mark; k < trail.size(); ++k) {
            std::vector<CellPtr> ca;
            ca.push_back(trail[k].cell); // the variable
            // Snapshot the bound value (post-binding) into a fresh cell.
            ca.push_back(std::make_shared<Value>(*trail[k].cell));
            pairs.push_back(std::make_shared<Value>(
                Value::Compound("=/2", std::move(ca))));
        }
        // Undo so the original variables become unbound again.
        while (trail.size() > mark) {
            *trail.back().cell = trail.back().prev;
            trail.pop_back();
        }
        CellPtr list = std::make_shared<Value>(Value::Atom("[]"));
        for (auto it = pairs.rbegin(); it != pairs.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(*it);
            ca.push_back(list);
            list = std::make_shared<Value>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A3"), list)) return false;
        pc += 1; return true;
    }

    // ---- plus/3 ----------------------------------------------------
    // plus(?X, ?Y, ?Z) -- integer addition with any two bound.
    // Modes: (+,+,?) Z = X+Y;  (+,?,+) Y = Z-X;  (?,+,+) X = Z-Y.
    // All-bound checks X+Y == Z. Two or three unbound throws an
    // instantiation_error per SWI/ISO.
    if (op == "plus/3") {
        Value x = *get_cell("A1");
        Value y = *get_cell("A2");
        Value z = *get_cell("A3");
        auto is_int = [](const Value& v){ return v.tag == Value::Tag::Integer; };
        auto type_err_int = [&](const Value& v) {
            return throw_iso_error(make_type_error("integer", v));
        };
        if (!x.is_unbound() && !is_int(x)) return type_err_int(x);
        if (!y.is_unbound() && !is_int(y)) return type_err_int(y);
        if (!z.is_unbound() && !is_int(z)) return type_err_int(z);
        if (!x.is_unbound() && !y.is_unbound()) {
            Value r = Value::Integer(x.i + y.i);
            CellPtr tgt = get_cell("A3");
            if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Value>(r))) return false;
            pc += 1; return true;
        }
        if (!x.is_unbound() && !z.is_unbound()) {
            Value r = Value::Integer(z.i - x.i);
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Value>(r))) return false;
            pc += 1; return true;
        }
        if (!y.is_unbound() && !z.is_unbound()) {
            Value r = Value::Integer(z.i - y.i);
            CellPtr tgt = get_cell("A1");
            if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Value>(r))) return false;
            pc += 1; return true;
        }
        return throw_iso_error(make_instantiation_error());
    }

    // ---- delete/3 --------------------------------------------------
    // delete(+List, +Elem, -Result) -- remove all elements of List
    // that match Elem under == (structural equality, no binding).
    if (op == "delete/3") {
        Value elem = *get_cell("A2");
        std::vector<CellPtr> kept;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = *lc;
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            CellPtr head = lv.args[0];
            if (!(*head == elem)) kept.push_back(head);
            lc = lv.args[1];
        }
        CellPtr list = std::make_shared<Value>(Value::Atom("[]"));
        for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(*it);
            ca.push_back(list);
            list = std::make_shared<Value>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A3"), list)) return false;
        pc += 1; return true;
    }

    // ---- subtract/3 ------------------------------------------------
    // subtract(+List1, +List2, -Result) -- set difference: Result
    // is List1 with all elements that appear in List2 removed.
    // Matching uses == (no binding).
    if (op == "subtract/3") {
        std::vector<Value> excl;
        CellPtr ec = get_cell("A2");
        for (;;) {
            Value ev = *ec;
            if (ev.tag == Value::Tag::Atom && ev.s == "[]") break;
            if (ev.tag != Value::Tag::Compound || ev.s != "[|]/2"
                || ev.args.size() != 2) return false;
            excl.push_back(*ev.args[0]);
            ec = ev.args[1];
        }
        std::vector<CellPtr> kept;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = *lc;
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            CellPtr head = lv.args[0];
            bool found = false;
            for (auto& e : excl) {
                if (*head == e) { found = true; break; }
            }
            if (!found) kept.push_back(head);
            lc = lv.args[1];
        }
        CellPtr list = std::make_shared<Value>(Value::Atom("[]"));
        for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(*it);
            ca.push_back(list);
            list = std::make_shared<Value>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A3"), list)) return false;
        pc += 1; return true;
    }

    // ---- must_be/2 -------------------------------------------------
    // must_be(+Type, @Value). Throws type_error(Type, Value) when
    // Value isn''t of Type, or instantiation_error if Type is an
    // instantiation-requiring type and Value is unbound. The
    // recognized types follow SWI conventions; unknown types throw
    // domain_error(type, Type).
    if (op == "must_be/2") {
        Value type_v = *get_cell("A1");
        if (type_v.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (type_v.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", type_v));
        const std::string& t = type_v.s;
        Value val = *get_cell("A2");
        bool ok_pred = false;
        bool need_inst = (t != "var" && t != "nonvar");
        if (need_inst && val.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (t == "atom")     ok_pred = (val.tag == Value::Tag::Atom);
        else if (t == "integer") ok_pred = (val.tag == Value::Tag::Integer);
        else if (t == "float")   ok_pred = (val.tag == Value::Tag::Float);
        else if (t == "number")  ok_pred = (val.tag == Value::Tag::Integer
                                            || val.tag == Value::Tag::Float);
        else if (t == "compound") ok_pred = (val.tag == Value::Tag::Compound);
        else if (t == "atomic")  ok_pred = (!val.is_unbound()
                                            && val.tag != Value::Tag::Compound);
        else if (t == "callable") ok_pred = (val.tag == Value::Tag::Atom
                                             || val.tag == Value::Tag::Compound);
        else if (t == "boolean") ok_pred = (val.tag == Value::Tag::Atom
                                            && (val.s == "true" || val.s == "false"));
        else if (t == "var")     ok_pred = val.is_unbound();
        else if (t == "nonvar")  ok_pred = !val.is_unbound();
        else if (t == "ground") {
            // ground failure is always due to an unbound subterm,
            // which SWI reports as instantiation_error -- not type_error.
            std::function<bool(const Value&)> g = [&](const Value& w) -> bool {
                if (w.is_unbound()) return false;
                if (w.tag == Value::Tag::Compound)
                    for (auto& c : w.args) if (!g(*c)) return false;
                return true;
            };
            if (!g(val))
                return throw_iso_error(make_instantiation_error());
            ok_pred = true;
        }
        else if (t == "is_list" || t == "list") {
            const Value* cur = &val;
            ok_pred = true;
            while (cur && cur->tag == Value::Tag::Compound
                   && cur->s == "[|]/2" && cur->args.size() == 2) {
                cur = cur->args[1].get();
            }
            if (!(cur && cur->tag == Value::Tag::Atom && cur->s == "[]"))
                ok_pred = false;
        }
        else {
            return throw_iso_error(make_domain_error("type", type_v));
        }
        if (!ok_pred) return throw_iso_error(make_type_error(t, val));
        pc += 1; return true;
    }

    // ---- random/1 --------------------------------------------------
    // random(-X) -- X is a Float in [0, 1).
    if (op == "random/1") {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        Value r = Value::Float(dist(cpp_global_rng()));
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(r))) return false;
        pc += 1; return true;
    }

    // ---- random_between/3 ------------------------------------------
    // random_between(+L, +H, -X) -- X is an Integer in [L, H] inclusive.
    // Throws type_error on non-integer bounds; fails if L > H.
    if (op == "random_between/3") {
        Value lv = *get_cell("A1");
        Value hv = *get_cell("A2");
        if (lv.is_unbound() || hv.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (lv.tag != Value::Tag::Integer)
            return throw_iso_error(make_type_error("integer", lv));
        if (hv.tag != Value::Tag::Integer)
            return throw_iso_error(make_type_error("integer", hv));
        if (lv.i > hv.i) return false;
        std::uniform_int_distribution<std::int64_t> dist(lv.i, hv.i);
        Value r = Value::Integer(dist(cpp_global_rng()));
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(r))) return false;
        pc += 1; return true;
    }

    // ---- random_member/2 -------------------------------------------
    // random_member(-X, +List) -- X is a uniformly random element of List.
    // Empty List fails; same shape as member but picks one item.
    if (op == "random_member/2") {
        std::vector<CellPtr> items;
        CellPtr lc = get_cell("A2");
        for (;;) {
            Value lv = *lc;
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            items.push_back(lv.args[0]);
            lc = lv.args[1];
        }
        if (items.empty()) return false;
        std::uniform_int_distribution<std::size_t> dist(0, items.size() - 1);
        std::size_t idx = dist(cpp_global_rng());
        if (!unify_cells(get_cell("A1"), items[idx])) return false;
        pc += 1; return true;
    }

    // ---- random_permutation/2 --------------------------------------
    // random_permutation(+List, -PermList) -- Fisher-Yates shuffle of List
    // via std::shuffle on a copy of the cell pointers.
    if (op == "random_permutation/2") {
        std::vector<CellPtr> items;
        CellPtr lc = get_cell("A1");
        for (;;) {
            Value lv = *lc;
            if (lv.tag == Value::Tag::Atom && lv.s == "[]") break;
            if (lv.tag != Value::Tag::Compound || lv.s != "[|]/2"
                || lv.args.size() != 2) return false;
            items.push_back(lv.args[0]);
            lc = lv.args[1];
        }
        std::shuffle(items.begin(), items.end(), cpp_global_rng());
        CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(*it);
            ca.push_back(list);
            list = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A2"), list)) return false;
        pc += 1; return true;
    }

    // ---- set_random/1 ----------------------------------------------
    // set_random(seed(Seed))    -- seed with integer Seed.
    // set_random(seed(random))  -- seed from std::random_device.
    if (op == "set_random/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Compound || a.s != "seed/1"
            || a.args.size() != 1)
            return throw_iso_error(make_domain_error("random_option", a));
        Value sv = *a.args[0];
        if (sv.tag == Value::Tag::Integer) {
            cpp_global_rng().seed((std::uint64_t)sv.i);
        } else if (sv.tag == Value::Tag::Atom && sv.s == "random") {
            std::random_device rd;
            cpp_global_rng().seed(rd());
        } else {
            return throw_iso_error(make_domain_error("seed_value", sv));
        }
        pc += 1; return true;
    }

    // ---- get_time/1 ------------------------------------------------
    // get_time(-Stamp) -- Float seconds since the Unix epoch with
    // sub-second precision via std::chrono::system_clock.
    if (op == "get_time/1") {
        auto now = std::chrono::system_clock::now();
        auto dur = now.time_since_epoch();
        double sec = std::chrono::duration_cast<
            std::chrono::duration<double>>(dur).count();
        Value r = Value::Float(sec);
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(r))) return false;
        pc += 1; return true;
    }

    // ---- stamp_date_time/3 -----------------------------------------
    // stamp_date_time(+Stamp, -DateTime, +TZ)
    //   Stamp:    Integer or Float seconds since the Unix epoch.
    //   TZ:       atom -- ''local'' or ''UTC''.
    //   DateTime: date(Y, Mo, D, H, Mi, S, Off, TZ, DST).
    //     Off is the seconds offset from UTC (0 for UTC; for local
    //     we fill it in from tm_gmtoff when the platform provides
    //     it, else 0).
    //     S is Float (whole second + sub-second remainder of Stamp).
    //     DST is the integer flag from tm_isdst (-1, 0, or 1).
    if (op == "stamp_date_time/3") {
        Value sv = *get_cell("A1");
        Value tzv = *get_cell("A3");
        if (sv.is_unbound() || tzv.is_unbound())
            return throw_iso_error(make_instantiation_error());
        double stamp;
        if (sv.tag == Value::Tag::Integer)      stamp = (double)sv.i;
        else if (sv.tag == Value::Tag::Float)   stamp = sv.f;
        else return throw_iso_error(make_type_error("number", sv));
        if (tzv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", tzv));
        std::time_t whole = (std::time_t)stamp;
        double frac = stamp - (double)whole;
        std::tm tm{};
        if (tzv.s == "UTC") {
#if defined(_WIN32)
            gmtime_s(&tm, &whole);
#else
            gmtime_r(&whole, &tm);
#endif
        } else if (tzv.s == "local") {
#if defined(_WIN32)
            localtime_s(&tm, &whole);
#else
            localtime_r(&whole, &tm);
#endif
        } else {
            return throw_iso_error(make_domain_error("timezone", tzv));
        }
        std::int64_t off = 0;
#if !defined(_WIN32)
        off = (std::int64_t)tm.tm_gmtoff;
#endif
        std::vector<CellPtr> dargs;
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_year + 1900)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_mon + 1)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_mday)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_hour)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_min)));
        dargs.push_back(std::make_shared<Cell>(Value::Float((double)tm.tm_sec + frac)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(off)));
        dargs.push_back(std::make_shared<Cell>(Value::Atom(tzv.s)));
        dargs.push_back(std::make_shared<Cell>(Value::Integer(tm.tm_isdst)));
        Value dt = Value::Compound("date/9", std::move(dargs));
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, dt); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(dt))) return false;
        pc += 1; return true;
    }

    // ---- date_time_stamp/2 -----------------------------------------
    // date_time_stamp(+DateTime, -Stamp). Inverse of stamp_date_time/3.
    // Accepts date/9 (full form) or date/6 (Y,Mo,D,H,Mi,S -- assumes
    // local time, ignores DST/off). Returns Float seconds since epoch.
    if (op == "date_time_stamp/2") {
        Value dt = *get_cell("A1");
        if (dt.tag != Value::Tag::Compound)
            return throw_iso_error(make_type_error("compound", dt));
        std::tm tm{};
        double frac = 0.0;
        bool is_utc = false;
        if (dt.s == "date/9" && dt.args.size() == 9) {
            Value y  = *dt.args[0];
            Value mo = *dt.args[1];
            Value d  = *dt.args[2];
            Value h  = *dt.args[3];
            Value mi = *dt.args[4];
            Value s  = *dt.args[5];
            Value tz = *dt.args[7];
            if (y.tag != Value::Tag::Integer || mo.tag != Value::Tag::Integer
                || d.tag != Value::Tag::Integer || h.tag != Value::Tag::Integer
                || mi.tag != Value::Tag::Integer)
                return throw_iso_error(make_type_error("integer", y));
            tm.tm_year = (int)y.i - 1900;
            tm.tm_mon  = (int)mo.i - 1;
            tm.tm_mday = (int)d.i;
            tm.tm_hour = (int)h.i;
            tm.tm_min  = (int)mi.i;
            if (s.tag == Value::Tag::Integer) { tm.tm_sec = (int)s.i; }
            else if (s.tag == Value::Tag::Float) {
                tm.tm_sec = (int)s.f;
                frac = s.f - (double)tm.tm_sec;
            } else return throw_iso_error(make_type_error("number", s));
            tm.tm_isdst = -1;
            if (tz.tag == Value::Tag::Atom && tz.s == "UTC") is_utc = true;
        } else if (dt.s == "date/6" && dt.args.size() == 6) {
            Value y  = *dt.args[0];
            Value mo = *dt.args[1];
            Value d  = *dt.args[2];
            Value h  = *dt.args[3];
            Value mi = *dt.args[4];
            Value s  = *dt.args[5];
            tm.tm_year = (int)y.i - 1900;
            tm.tm_mon  = (int)mo.i - 1;
            tm.tm_mday = (int)d.i;
            tm.tm_hour = (int)h.i;
            tm.tm_min  = (int)mi.i;
            if (s.tag == Value::Tag::Integer) tm.tm_sec = (int)s.i;
            else if (s.tag == Value::Tag::Float) {
                tm.tm_sec = (int)s.f;
                frac = s.f - (double)tm.tm_sec;
            } else return throw_iso_error(make_type_error("number", s));
            tm.tm_isdst = -1;
        } else {
            return throw_iso_error(make_type_error("date", dt));
        }
        std::time_t stamp;
        if (is_utc) {
#if defined(_WIN32)
            stamp = _mkgmtime(&tm);
#else
            stamp = timegm(&tm);
#endif
        } else {
            stamp = std::mktime(&tm);
        }
        if (stamp == (std::time_t)-1)
            return throw_iso_error(make_domain_error("date", dt));
        Value r = Value::Float((double)stamp + frac);
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, r); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(r))) return false;
        pc += 1; return true;
    }

    // ---- format_time/3 ---------------------------------------------
    // format_time(-Out, +Format, +Stamp). strftime(3)-style format.
    // Out unifies with the formatted atom. (Variant taking a stream
    // destination is deferred -- string sinks (atom/string/codes)
    // are the common case.) Stamp is Float seconds since epoch (UTC
    // interpretation); pre-decomposed date/9 terms are also accepted.
    if (op == "format_time/3") {
        Value fmt = *get_cell("A2");
        Value sv = *get_cell("A3");
        if (fmt.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", fmt));
        std::tm tm{};
        if (sv.tag == Value::Tag::Integer || sv.tag == Value::Tag::Float) {
            double d = (sv.tag == Value::Tag::Float) ? sv.f : (double)sv.i;
            std::time_t whole = (std::time_t)d;
#if defined(_WIN32)
            gmtime_s(&tm, &whole);
#else
            gmtime_r(&whole, &tm);
#endif
        } else if (sv.tag == Value::Tag::Compound
                   && sv.s == "date/9" && sv.args.size() == 9) {
            Value y = *sv.args[0]; Value mo = *sv.args[1];
            Value d = *sv.args[2]; Value h = *sv.args[3];
            Value mi = *sv.args[4]; Value s = *sv.args[5];
            tm.tm_year = (int)y.i - 1900;
            tm.tm_mon  = (int)mo.i - 1;
            tm.tm_mday = (int)d.i;
            tm.tm_hour = (int)h.i;
            tm.tm_min  = (int)mi.i;
            tm.tm_sec  = (s.tag == Value::Tag::Float)
                         ? (int)s.f : (int)s.i;
        } else {
            return throw_iso_error(make_type_error("date_or_stamp", sv));
        }
        char buf[256];
        std::size_t n = std::strftime(buf, sizeof(buf), fmt.s.c_str(), &tm);
        if (n == 0 && !fmt.s.empty())
            return throw_iso_error(make_domain_error("format_time", fmt));
        Value out = Value::Atom(std::string(buf, n));
        CellPtr tgt = get_cell("A1");
        if (tgt->is_unbound()) { bind_cell(tgt, out); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(out))) return false;
        pc += 1; return true;
    }

    // ---- open/3 ----------------------------------------------------
    // open(+File, +Mode, -Stream). Mode: read | write | append.
    // Stream unifies with a freshly minted atom handle "$stream_N"
    // that the read/write builtins look up in the streams registry.
    // Throws existence_error(source_sink, File) if the file can''t
    // be opened (mostly for read mode where the file must exist).
    if (op == "open/3") {
        Value fv = *get_cell("A1");
        Value mv = *get_cell("A2");
        if (fv.is_unbound() || mv.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (fv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", fv));
        if (mv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", mv));
        std::ios_base::openmode m;
        bool is_read = false;
        if      (mv.s == "read")   { m = std::ios::in;  is_read = true; }
        else if (mv.s == "write")  { m = std::ios::out | std::ios::trunc; }
        else if (mv.s == "append") { m = std::ios::out | std::ios::app; }
        else return throw_iso_error(make_domain_error("io_mode", mv));
        auto fs = std::make_unique<std::fstream>(fv.s, m);
        if (!fs->is_open())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("source_sink")),
                std::make_shared<Cell>(fv)
            }));
        std::string handle = "$stream_" + std::to_string(stream_counter++);
        StreamEntry entry;
        entry.file = std::move(fs);
        entry.is_read = is_read;
        streams.emplace(handle, std::move(entry));
        Value sv = Value::Atom(handle);
        CellPtr tgt = get_cell("A3");
        if (tgt->is_unbound()) { bind_cell(tgt, sv); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(sv))) return false;
        pc += 1; return true;
    }

    // ---- close/1 ---------------------------------------------------
    // close(+Stream). Closes and erases the handle. Throws
    // existence_error(stream, Stream) if the handle isn''t known.
    if (op == "close/1") {
        Value sv = *get_cell("A1");
        if (sv.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        it->second.file->close();
        streams.erase(it);
        pc += 1; return true;
    }

    // ---- read_line_to_string/2 -------------------------------------
    // read_line_to_string(+Stream, -Line). Reads up to but not
    // including the next newline. On EOF, Line unifies with the atom
    // end_of_file. Trailing CR (Windows CRLF) is stripped.
    if (op == "read_line_to_string/2") {
        Value sv = *get_cell("A1");
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        if (!it->second.is_read) return false;
        std::string line;
        if (!std::getline(*it->second.file, line)) {
            Value eof = Value::Atom("end_of_file");
            CellPtr tgt = get_cell("A2");
            if (tgt->is_unbound()) { bind_cell(tgt, eof); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(eof))) return false;
            pc += 1; return true;
        }
        if (!line.empty() && line.back() == ''\\r'') line.pop_back();
        Value lv = Value::Atom(line);
        CellPtr tgt = get_cell("A2");
        if (tgt->is_unbound()) { bind_cell(tgt, lv); pc += 1; return true; }
        if (!unify_cells(tgt, std::make_shared<Cell>(lv))) return false;
        pc += 1; return true;
    }

    // ---- read_string/5 ---------------------------------------------
    // read_string(+Stream, +Length, ?Length, -String) -- a subset
    // of SWI''s 5-arg form. With +Length, reads up to Length chars
    // (less if EOF reached); the third arg unifies with the actual
    // count read. The fourth arg (PadEnd in SWI; sometimes used as
    // a sentinel) is accepted but ignored in our minimal impl.
    if (op == "read_string/5") {
        Value sv = *get_cell("A1");
        Value lv = *get_cell("A2");
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        if (lv.tag != Value::Tag::Integer)
            return throw_iso_error(make_type_error("integer", lv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        if (!it->second.is_read) return false;
        std::string buf;
        buf.resize((std::size_t)lv.i);
        it->second.file->read(&buf[0], lv.i);
        std::streamsize got = it->second.file->gcount();
        buf.resize((std::size_t)got);
        Value count_v = Value::Integer((std::int64_t)got);
        Value str_v = Value::Atom(buf);
        CellPtr tgt3 = get_cell("A3");
        if (tgt3->is_unbound()) bind_cell(tgt3, count_v);
        else if (!unify_cells(tgt3, std::make_shared<Cell>(count_v))) return false;
        CellPtr tgt5 = get_cell("A5");
        if (tgt5->is_unbound()) bind_cell(tgt5, str_v);
        else if (!unify_cells(tgt5, std::make_shared<Cell>(str_v))) return false;
        pc += 1; return true;
    }

    // ---- at_end_of_stream/1 ----------------------------------------
    // at_end_of_stream(+Stream). Succeeds when the next read on
    // Stream would yield EOF.
    if (op == "at_end_of_stream/1") {
        Value sv = *get_cell("A1");
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        if (!it->second.is_read) return false;
        std::fstream& f = *it->second.file;
        if (f.eof()) { pc += 1; return true; }
        int c = f.peek();
        if (c == EOF) { pc += 1; return true; }
        return false;
    }

    // ---- write_to_stream/2 -----------------------------------------
    // write_to_stream(+Stream, +Term). Renders Term and writes it
    // (no newline). Companion: nl_to_stream/1 writes one newline.
    // Not standard SWI; convenient before with_output_to(stream(_),...)
    // lands.
    if (op == "write_to_stream/2") {
        Value sv = *get_cell("A1");
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        if (it->second.is_read) return false;
        Value v = *get_cell("A2");
        *it->second.file << render(v);
        pc += 1; return true;
    }
    if (op == "nl_to_stream/1") {
        Value sv = *get_cell("A1");
        if (sv.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", sv));
        auto it = streams.find(sv.s);
        if (it == streams.end())
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("stream")),
                std::make_shared<Cell>(sv)
            }));
        if (it->second.is_read) return false;
        *it->second.file << ''\\n'';
        pc += 1; return true;
    }

    // ---- Filesystem helpers ----------------------------------------
    // Thin wrappers over std::filesystem. Path arguments must be
    // bound atoms; non-atom paths throw type_error(atom, _).
    if (op == "exists_file/1") {
        Value p = *get_cell("A1");
        if (p.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (p.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", p));
        std::error_code ec;
        bool ok = std::filesystem::is_regular_file(p.s, ec);
        if (!ok) return false;
        pc += 1; return true;
    }
    if (op == "exists_directory/1") {
        Value p = *get_cell("A1");
        if (p.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (p.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", p));
        std::error_code ec;
        bool ok = std::filesystem::is_directory(p.s, ec);
        if (!ok) return false;
        pc += 1; return true;
    }
    if (op == "directory_files/2") {
        Value p = *get_cell("A1");
        if (p.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (p.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", p));
        std::error_code ec;
        if (!std::filesystem::is_directory(p.s, ec))
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("directory")),
                std::make_shared<Cell>(p)
            }));
        std::vector<std::string> names;
        names.push_back(".");
        names.push_back("..");
        for (auto& entry : std::filesystem::directory_iterator(p.s, ec)) {
            names.push_back(entry.path().filename().string());
        }
        // Standard ordering: SWI returns names in unspecified order;
        // we sort lexically for deterministic test output.
        std::sort(names.begin() + 2, names.end());
        CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
        for (auto it = names.rbegin(); it != names.rend(); ++it) {
            std::vector<CellPtr> ca;
            ca.push_back(std::make_shared<Cell>(Value::Atom(*it)));
            ca.push_back(list);
            list = std::make_shared<Cell>(
                Value::Compound("[|]/2", std::move(ca)));
        }
        if (!unify_cells(get_cell("A2"), list)) return false;
        pc += 1; return true;
    }
    if (op == "make_directory/1") {
        Value p = *get_cell("A1");
        if (p.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (p.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", p));
        std::error_code ec;
        std::filesystem::create_directory(p.s, ec);
        if (ec)
            return throw_iso_error(Value::Compound("permission_error/3", {
                std::make_shared<Cell>(Value::Atom("create")),
                std::make_shared<Cell>(Value::Atom("directory")),
                std::make_shared<Cell>(p)
            }));
        pc += 1; return true;
    }
    if (op == "delete_file/1") {
        Value p = *get_cell("A1");
        if (p.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (p.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", p));
        std::error_code ec;
        bool ok = std::filesystem::remove(p.s, ec);
        if (!ok || ec)
            return throw_iso_error(Value::Compound("existence_error/2", {
                std::make_shared<Cell>(Value::Atom("source_sink")),
                std::make_shared<Cell>(p)
            }));
        pc += 1; return true;
    }

    // ---- current_predicate/1 ---------------------------------------
    // current_predicate(?PredSpec). PredSpec is Name/Arity. Check
    // mode: both Name and Arity bound -- succeeds iff the predicate
    // exists in the WAM (either as a static label or in the dynamic
    // database). Enum mode (partial spec) is deferred -- callers can
    // use findall + a known-name list for now.
    if (op == "current_predicate/1") {
        Value spec = *get_cell("A1");
        if (spec.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (spec.tag != Value::Tag::Compound || spec.s != "//2"
            || spec.args.size() != 2)
            return throw_iso_error(make_type_error("predicate_indicator", spec));
        Value name = *spec.args[0];
        Value arity = *spec.args[1];
        if (name.is_unbound() || arity.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (name.tag != Value::Tag::Atom)
            return throw_iso_error(make_type_error("atom", name));
        if (arity.tag != Value::Tag::Integer)
            return throw_iso_error(make_type_error("integer", arity));
        std::string key = name.s + "/" + std::to_string(arity.i);
        if (labels.find(key) != labels.end()) { pc += 1; return true; }
        if (dynamic_db.find(key) != dynamic_db.end()) { pc += 1; return true; }
        return false;
    }

    // ---- predicate_property/2 --------------------------------------
    // predicate_property(+Head, +Property). Property is checked
    // against the predicate''s status:
    //   defined        -- in labels OR dynamic_db.
    //   dynamic        -- in dynamic_db (asserted at runtime).
    //   static         -- in labels but not dynamic_db.
    //   number_of_clauses(?N) -- N is the number of clauses (for
    //                    dynamic preds; for static preds we report
    //                    1 since we can''t introspect the clause list).
    // Head must be bound to an atom or compound term. Multi-clause
    // ranking of properties (per ISO) is not supported in this
    // minimal impl -- just check-mode against the current property.
    if (op == "predicate_property/2") {
        Value head = *get_cell("A1");
        if (head.is_unbound())
            return throw_iso_error(make_instantiation_error());
        std::string key;
        if (head.tag == Value::Tag::Atom) {
            key = head.s + "/0";
        } else if (head.tag == Value::Tag::Compound) {
            // head.s is "name/arity" already.
            key = head.s;
        } else {
            return throw_iso_error(make_type_error("callable", head));
        }
        bool in_labels = (labels.find(key) != labels.end());
        bool in_dynamic = (dynamic_db.find(key) != dynamic_db.end());
        Value prop = *get_cell("A2");
        if (prop.is_unbound())
            return throw_iso_error(make_instantiation_error());
        if (prop.tag == Value::Tag::Atom) {
            if (prop.s == "defined") {
                if (in_labels || in_dynamic) { pc += 1; return true; }
                return false;
            }
            if (prop.s == "dynamic") {
                if (in_dynamic) { pc += 1; return true; }
                return false;
            }
            if (prop.s == "static") {
                if (in_labels && !in_dynamic) { pc += 1; return true; }
                return false;
            }
            return throw_iso_error(make_domain_error("predicate_property", prop));
        }
        if (prop.tag == Value::Tag::Compound
            && prop.s == "number_of_clauses/1" && prop.args.size() == 1) {
            std::int64_t n;
            if (in_dynamic) {
                n = (std::int64_t)dynamic_db[key].size();
            } else if (in_labels) {
                n = 1; // best-effort: can''t introspect static clause list.
            } else {
                return false;
            }
            Value nv = Value::Integer(n);
            CellPtr tgt = prop.args[0];
            if (tgt->is_unbound()) { bind_cell(tgt, nv); pc += 1; return true; }
            if (!unify_cells(tgt, std::make_shared<Cell>(nv))) return false;
            pc += 1; return true;
        }
        return throw_iso_error(make_domain_error("predicate_property", prop));
    }

    // ---- I/O -------------------------------------------------------
    // All write paths route through emit_output / emit_output_char so
    // with_output_to/2 can intercept them into a capture buffer.
    if (op == "write/1") {
        emit_output(render(*get_cell("A1")));
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "nl/0") {
        emit_output_char(''\\n'');
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "write_atom/1" || op == "writeln/1") {
        emit_output(render(*get_cell("A1")));
        emit_output_char(''\\n'');
        std::fflush(stdout);
        pc += 1; return true;
    }
    // print/1 — alias for write/1 (no portray hook in this runtime).
    if (op == "print/1") {
        emit_output(render(*get_cell("A1")));
        std::fflush(stdout);
        pc += 1; return true;
    }
    // display/1 — write a term without operator notation. Our render
    // path doesn''t use operator syntax anyway, so display == write.
    if (op == "display/1") {
        emit_output(render(*get_cell("A1")));
        std::fflush(stdout);
        pc += 1; return true;
    }
    // tab/1 — write N spaces. N must be a non-negative integer.
    if (op == "tab/1") {
        Value n = deref(*get_cell("A1"));
        if (n.tag != Value::Tag::Integer || n.i < 0) return false;
        for (std::int64_t i = 0; i < n.i; ++i) emit_output_char('' '');
        std::fflush(stdout);
        pc += 1; return true;
    }
    // write_canonical/1 — write a term with quotes around atoms that
    // need them (atoms whose textual form is empty, contains special
    // chars, or would re-parse as a number). For non-atoms we fall
    // back to render — close enough for simple cases.
    if (op == "write_canonical/1") {
        Value v = deref(*get_cell("A1"));
        if (v.tag == Value::Tag::Atom) {
            const std::string& s = v.s;
            // Quote atoms that are empty, contain control / quote
            // chars, or whose form would parse as a number.
            // Char codes: 39 = '', 34 = ", 92 = \\.
            bool needs_quote = s.empty();
            if (!needs_quote) {
                for (unsigned char c : s) {
                    if (c <= 32 || c == 39 || c == 34 || c == '','') {
                        needs_quote = true;
                        break;
                    }
                }
            }
            if (!needs_quote) {
                try {
                    std::size_t pos = 0;
                    std::stoll(s, &pos);
                    if (pos == s.size()) needs_quote = true;
                } catch (...) {}
                if (!needs_quote) {
                    try {
                        std::size_t pos = 0;
                        std::stod(s, &pos);
                        if (pos == s.size()) needs_quote = true;
                    } catch (...) {}
                }
            }
            if (needs_quote) {
                emit_output_char(static_cast<char>(39)); // opening quote
                for (unsigned char c : s) {
                    if (c == 39 || c == 92) emit_output_char(static_cast<char>(92));
                    emit_output_char(static_cast<char>(c));
                }
                emit_output_char(static_cast<char>(39)); // closing quote
            } else {
                emit_output(s);
            }
            std::fflush(stdout);
            pc += 1; return true;
        }
        emit_output(render(v));
        std::fflush(stdout);
        pc += 1; return true;
    }
    // ---- with_output_to/2 -------------------------------------------
    // with_output_to(+Sink, :Goal): capture Goal''s output into Sink.
    // Sink is atom(V) / string(V) / codes(V) / stream(Handle).
    // The I/O builtins write to the top OutputCaptureFrame''s buffer
    // while it''s active; when Goal proceeds normally, control lands
    // at output_capture_return_pc which pops the frame and either
    // unifies the buffer with the sink''s arg (atom/string/codes) or
    // writes the buffer to the stream (stream(Handle)).
    if (op == "with_output_to/2") {
        Value sv = deref(*get_cell("A1"));
        if (sv.tag != Value::Tag::Compound || sv.args.size() != 1
            || (sv.s != "atom/1" && sv.s != "string/1"
                && sv.s != "codes/1" && sv.s != "stream/1")) return false;
        OutputCaptureFrame f;
        f.sink = get_cell("A1");
        // saved_cp: for the direct BuiltinCall instr path, pc + 1
        // is the next instruction. For invoke_goal_as_call goal-term
        // dispatch, the caller pre-sets cp = after_pc so we use that.
        f.saved_cp = (cp != 0) ? cp : (pc + 1);
        f.base_cp_count = choice_points.size();
        f.base_agg_count = aggregate_frames.size();
        f.base_catcher_count = catcher_frames.size();
        f.trail_mark = trail.size();
        f.saved_cut_barrier = cut_barrier;
        f.saved_regs = regs;
        f.saved_mode_stack = mode_stack;
        f.saved_env_stack = env_stack;
        std::size_t my_depth = output_capture_frames.size();
        output_capture_frames.push_back(std::move(f));
        CellPtr goal = get_cell("A2");
        if (!invoke_goal_as_call(goal, output_capture_return_pc)) {
            // Goal failed to dispatch — pop and propagate.
            if (output_capture_frames.size() > my_depth)
                output_capture_frames.pop_back();
            return false;
        }
        return true;
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
            emit_output(buf);
            std::fflush(stdout);
            pc += 1; return true;
        }
        // format/3: A1 selects destination.
        Value dv = deref(*get_cell("A1"));
        if (dv.tag == Value::Tag::Atom) {
            if (dv.s == "user_output") {
                emit_output(buf);
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
            // sub_string/5 is the SWI string-typed alias and shares
            // the same semantics on this runtime (atoms and strings
            // are unified as Atom-tagged values).
            if (instr.a == "sub_atom/5" || instr.a == "sub_string/5")
                return dispatch_sub_atom(pc + 1);
            // retract/1 — nondeterministic clause removal.
            if (instr.a == "retract/1") return dispatch_retract(pc + 1);
            // clause/2 — nondet enumeration of dynamic-db clauses.
            if (instr.a == "clause/2") return dispatch_clause(pc + 1);
            if (instr.a == "current_predicate/1")
                return dispatch_current_predicate(pc + 1);
            // ^/2 — existential quantification. Transparent for our
            // find-style aggregation: invoke A2 (the goal) with the
            // standard non-tail after-pc.
            if (instr.a == "^/2") {
                return invoke_goal_as_call(get_cell("A2"), pc + 1);
            }
            // phrase/2, phrase/3 — DCG entry. Add [List, Rest] to the
            // body goal''s args then dispatch as a regular call.
            if (instr.a == "phrase/2") {
                return dispatch_phrase_call(false, pc + 1);
            }
            if (instr.a == "phrase/3") {
                return dispatch_phrase_call(true, pc + 1);
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
            if (instr.a == "sub_atom/5" || instr.a == "sub_string/5")
                return dispatch_sub_atom(cp);
            if (instr.a == "retract/1") return dispatch_retract(cp);
            if (instr.a == "clause/2") return dispatch_clause(cp);
            if (instr.a == "current_predicate/1")
                return dispatch_current_predicate(cp);
            if (instr.a == "^/2") {
                return invoke_goal_as_call(get_cell("A2"), cp);
            }
            // phrase/2, phrase/3 in tail position.
            if (instr.a == "phrase/2") {
                return dispatch_phrase_call(false, cp);
            }
            if (instr.a == "phrase/3") {
                return dispatch_phrase_call(true, cp);
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
        case Instruction::Op::CurrentPredNext: {
            // Reached when current_predicate/1 backtracks for the
            // next match. Pop the CP and let current_pred_try_next
            // resume from iter.next_idx.
            if (!choice_points.empty()) choice_points.pop_back();
            return current_pred_try_next();
        }
        case Instruction::Op::OutputCaptureReturn: {
            // Reached when a with_output_to/2 goal proceeds normally.
            // Pop the OutputCaptureFrame, unify the captured buffer
            // with the sink (atom/string/codes), and continue at the
            // saved_cp.
            if (output_capture_frames.empty()) return false;
            OutputCaptureFrame f = std::move(output_capture_frames.back());
            output_capture_frames.pop_back();
            Value sink = deref(*f.sink);
            if (sink.tag != Value::Tag::Compound
                || sink.args.size() != 1) return false;
            // stream/1 sink: write the buffer to the stream and skip
            // unification entirely. Stream must already be open in
            // write or append mode.
            if (sink.s == "stream/1") {
                Value sh = *sink.args[0];
                if (sh.tag != Value::Tag::Atom) return false;
                auto sit = streams.find(sh.s);
                if (sit == streams.end()) return false;
                if (sit->second.is_read) return false;
                *sit->second.file << f.buffer;
                if (f.saved_cp == 0) { halt = true; return true; }
                pc = f.saved_cp;
                cp = 0;
                return true;
            }
            CellPtr tgt = sink.args[0];
            Value out_v;
            if (sink.s == "atom/1" || sink.s == "string/1") {
                out_v = Value::Atom(f.buffer);
            } else if (sink.s == "codes/1") {
                // Build a list of integer codes from buf bottom-up.
                CellPtr list = std::make_shared<Cell>(Value::Atom("[]"));
                for (auto it = f.buffer.rbegin();
                     it != f.buffer.rend(); ++it) {
                    CellPtr head = std::make_shared<Cell>(Value::Integer(
                        static_cast<std::int64_t>(
                            static_cast<unsigned char>(*it))));
                    std::vector<CellPtr> ca;
                    ca.push_back(head);
                    ca.push_back(list);
                    list = std::make_shared<Cell>(
                        Value::Compound("[|]/2", std::move(ca)));
                }
                out_v = *list;
            } else {
                return false;
            }
            if (tgt->is_unbound()) bind_cell(tgt, out_v);
            else if (!unify_cells(tgt,
                std::make_shared<Cell>(out_v))) return false;
            if (f.saved_cp == 0) { halt = true; return true; }
            pc = f.saved_cp;
            cp = 0;
            return true;
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
        case Instruction::Op::Nop:
            // Placeholder emitted for WAM-asm pseudo-instructions that
            // currently have no runtime semantics (e.g.
            // switch_on_constant_a2). Counted in PC accounting so labels
            // resolve correctly; behaves as a fall-through.
            pc += 1;
            return true;

        // ---- Choice points -----------------------------------------
        case Instruction::Op::TryMeElse: {
            // A normal entry to a clause chain. Clear any
            // indexed_entry flag in case SwitchOnTerm pointed here
            // (defensive -- our current compiler points indexed
            // jumps at RetryMeElse / TrustMe rather than TryMeElse,
            // but the flag should be local to a single dispatch).
            indexed_entry = false;
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
            // Normal failure-driven retry path: update top CP''s alt.
            // Indexed-entry path (set by SwitchOnTerm bypassing the
            // chain''s entry TryMeElse): synthesize a fresh CP so this
            // predicate level has its own backtrack handle, even if
            // an outer level''s CP is already on the stack.
            if (!choice_points.empty() && !indexed_entry) {
                choice_points.back().alt_pc = instr.target;
            } else {
                indexed_entry = false;
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
            }
            pc += 1; return true;
        }
        case Instruction::Op::TrustMe: {
            // Normally pops the top CP (the one TryMeElse/RetryMeElse
            // installed for this chain). When SwitchOnTerm jumped
            // directly to a TrustMe (last clause), no CP was installed
            // and indexed_entry flags this -- nothing to pop, just
            // clear the flag and continue.
            if (indexed_entry) {
                indexed_entry = false;
            } else if (!choice_points.empty()) {
                choice_points.pop_back();
            }
            pc += 1; return true;
        }

        // ---- Indexed-dispatch chain ops (issue #2400) -------------
        // Emitted by wam_target.pl''s build_term_index_with_chains for
        // SwitchOnTerm / SwitchOnConstant / SwitchOnStructure targets
        // whose dispatch group has >1 matching clauses.  Layout:
        //   L_<P>_<A>_<group>_dispatch:
        //       try   L_<P>_<A>_<I1>_body
        //       retry L_<P>_<A>_<I2>_body
        //       ...
        //       trust L_<P>_<A>_<IN>_body
        // Each instruction''s target is the body label (PC).  CP push
        // captures pc+1 (the next chain entry) so backtrack resumes
        // through the chain.  Unlike TryMeElse/RetryMeElse, these
        // JUMP to the body and don''t need the indexed_entry hack —
        // the chain is self-contained.
        case Instruction::Op::Try: {
            // Push a CP that resumes the chain at pc+1 (= next chain
            // instr) on backtrack, then JUMP to the body label.
            indexed_entry = false; // chain is self-contained
            ChoicePoint cp_;
            cp_.alt_pc = pc + 1;
            cp_.saved_cp = cp;
            cp_.trail_mark = trail.size();
            cp_.cut_barrier = cut_barrier;
            cp_.saved_regs = regs;
            cp_.saved_mode_stack = mode_stack;
            cp_.saved_env_stack = env_stack;
            cp_.saved_body_frames = body_frames;
            choice_points.push_back(std::move(cp_));
            pc = instr.target; return true;
        }
        case Instruction::Op::Retry: {
            // Mutate the top CP''s alt_pc to pc+1 (next chain instr).
            // The CP itself stays on the stack -- C++''s backtrack
            // restores from but never pops, so Try''s CP survives
            // each backtrack and we just update where to resume.
            // (Symmetric to RetryMeElse''s mutate-don''t-push branch.)
            // Defensive fallback: if no CP exists (shouldn''t happen
            // in well-formed chains), synthesize one so the chain
            // stays consistent.
            indexed_entry = false;
            if (!choice_points.empty()) {
                choice_points.back().alt_pc = pc + 1;
            } else {
                ChoicePoint cp_;
                cp_.alt_pc = pc + 1;
                cp_.saved_cp = cp;
                cp_.trail_mark = trail.size();
                cp_.cut_barrier = cut_barrier;
                cp_.saved_regs = regs;
                cp_.saved_mode_stack = mode_stack;
                cp_.saved_env_stack = env_stack;
                cp_.saved_body_frames = body_frames;
                choice_points.push_back(std::move(cp_));
            }
            pc = instr.target; return true;
        }
        case Instruction::Op::Trust: {
            // Last entry in the chain: pop the chain CP and jump to
            // the body label.  Mirrors TrustMe but jumps rather
            // than falling through.  Skipping the pop leaks the
            // chain CP and causes an infinite retry loop when this
            // clause body fails and backtrack restores from it.
            indexed_entry = false;
            if (!choice_points.empty()) {
                choice_points.pop_back();
            }
            pc = instr.target; return true;
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
        // M17 soft cut. get_level snapshots the choicepoint level into a
        // Y register BEFORE the if-then-else / negation try_me_else; cut
        // truncates choicepoints back to that level at the commit site,
        // removing the ITE/negation CP AND every CP the condition (or a
        // negated goals generator) pushed above it. This fixes the legacy
        // cut_ite, which only dropped the single topmost CP and so could
        // not cut a negation over a generator (e.g. the forall negative
        // case left the generators CP alive).
        case Instruction::Op::GetLevel: {
            CellPtr c = get_cell(instr.a);
            bind_cell(c, Value::Integer(
                static_cast<std::int64_t>(choice_points.size())));
            pc += 1; return true;
        }
        case Instruction::Op::Cut: {
            CellPtr c = get_cell(instr.a);
            Value v = deref(*c);
            if (v.tag == Value::Tag::Integer) {
                std::size_t target = static_cast<std::size_t>(v.i);
                if (choice_points.size() > target) {
                    choice_points.resize(target);
                }
                if (cut_barrier > target) cut_barrier = target;
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
            // Nondeterministic / CP-aware builtins need their own dispatch
            // arms (mirroring the Call opcode) because builtin() only
            // handles the deterministic ops. The shared WAM compiler emits
            // these as builtin_call, so without these arms they fall
            // through to builtin() and silently fail.
            if (instr.a == "sub_atom/5" || instr.a == "sub_string/5")
                return dispatch_sub_atom(pc + 1);
            if (instr.a == "retract/1") return dispatch_retract(pc + 1);
            if (instr.a == "clause/2") return dispatch_clause(pc + 1);
            if (instr.a == "current_predicate/1")
                return dispatch_current_predicate(pc + 1);
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
            // Dispatch on A1''s atom/integer value. When the table has
            // multiple entries for the same key (e.g. several clauses
            // sharing A1=grew), only the first is the jump target; the
            // rest are reached via the RetryMeElse chain at that label.
            // Setting indexed_entry tells the receiving RetryMeElse to
            // synthesize a fresh CP at this predicate level (otherwise
            // it would mutate an outer level''s CP). Symmetric to
            // SwitchOnTerm.
            CellPtr ac = get_cell("A1");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag != Value::Tag::Atom && a.tag != Value::Tag::Integer
                && a.tag != Value::Tag::Float) {
                pc += 1; return true; // not a constant we index on
            }
            auto it = instr.const_map.find(a);
            if (it != instr.const_map.end()) {
                if (it->second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                if (it->second == Instruction::SWITCH_NONE)    return false;
                indexed_entry = true;
                pc = it->second; return true;
            }
            // Bound constant with no matching indexed clause. If the
            // predicate has variable-headed clauses (mixed-mode
            // indexing), fall through to the try_me_else chain so
            // those clauses still get a chance. Otherwise fail fast.
            if (instr.no_match_fallthrough) { pc += 1; return true; }
            return false;
        }
        case Instruction::Op::SwitchOnConstantA2: {
            // Dispatch on A2''s atom/integer value. Emitted when A1 is
            // too variable to index on but A2 has all constants (e.g.
            // a tag-style second argument). Identical shape to
            // SwitchOnConstant — see comments there.
            CellPtr ac = get_cell("A2");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag != Value::Tag::Atom && a.tag != Value::Tag::Integer
                && a.tag != Value::Tag::Float) {
                pc += 1; return true;
            }
            auto it = instr.const_map.find(a);
            if (it != instr.const_map.end()) {
                if (it->second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                if (it->second == Instruction::SWITCH_NONE)    return false;
                indexed_entry = true;
                pc = it->second; return true;
            }
            // See SwitchOnConstant: bound A2 with no entry in the
            // table falls through to the try_me_else chain when the
            // predicate has variable-A2 clauses (mixed-mode A2).
            if (instr.no_match_fallthrough) { pc += 1; return true; }
            return false;
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
                    indexed_entry = true;
                    pc = kv.second; return true;
                }
            }
            return false;
        }
        case Instruction::Op::SwitchOnStructureA2: {
            // A2 mirror of SwitchOnStructure.
            CellPtr ac = get_cell("A2");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag != Value::Tag::Compound) { pc += 1; return true; }
            for (auto& kv : instr.struct_table) {
                if (kv.first == a.s) {
                    if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (kv.second == Instruction::SWITCH_NONE)    return false;
                    indexed_entry = true;
                    pc = kv.second; return true;
                }
            }
            return false;
        }
        case Instruction::Op::SwitchOnTerm: {
            // Direct-jump branches (pc = target) bypass the entry
            // TryMeElse of the clause chain, so we flag them via
            // indexed_entry. The receiving RetryMeElse/TrustMe sees
            // the flag and synthesizes a CP (or skips a pop) so the
            // current predicate level gets its own backtrack handle
            // even when an outer level''s CP is on the stack.
            CellPtr ac = get_cell("A1");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag == Value::Tag::Atom || a.tag == Value::Tag::Integer
                || a.tag == Value::Tag::Float) {
                for (auto& kv : instr.const_table) {
                    if (kv.first == a) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        indexed_entry = true;
                        pc = kv.second; return true;
                    }
                }
                return false;
            }
            if (a.tag == Value::Tag::Compound) {
                if (a.s == "[|]/2") {
                    if (instr.target == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (instr.target == Instruction::SWITCH_NONE)    return false;
                    indexed_entry = true;
                    pc = instr.target; return true;
                }
                for (auto& kv : instr.struct_table) {
                    if (kv.first == a.s) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        indexed_entry = true;
                        pc = kv.second; return true;
                    }
                }
                return false;
            }
            pc += 1; return true;
        }
        case Instruction::Op::SwitchOnTermA2: {
            // A2 mirror of SwitchOnTerm.
            CellPtr ac = get_cell("A2");
            const Value& a = *ac;
            if (a.is_unbound()) { pc += 1; return true; }
            if (a.tag == Value::Tag::Atom || a.tag == Value::Tag::Integer
                || a.tag == Value::Tag::Float) {
                for (auto& kv : instr.const_table) {
                    if (kv.first == a) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        indexed_entry = true;
                        pc = kv.second; return true;
                    }
                }
                return false;
            }
            if (a.tag == Value::Tag::Compound) {
                if (a.s == "[|]/2") {
                    if (instr.target == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                    if (instr.target == Instruction::SWITCH_NONE)    return false;
                    indexed_entry = true;
                    pc = instr.target; return true;
                }
                for (auto& kv : instr.struct_table) {
                    if (kv.first == a.s) {
                        if (kv.second == Instruction::SWITCH_DEFAULT) { pc += 1; return true; }
                        if (kv.second == Instruction::SWITCH_NONE)    return false;
                        indexed_entry = true;
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

// clause/2 dispatch. Reuses the RetractIterator infrastructure but
// with is_clause_only=true so retract_try_next unifies both head
// and body and DOESN''T remove the matched clause.
bool WamState::dispatch_clause(std::size_t after_pc) {
    CellPtr head_pat = get_cell("A1");
    CellPtr body_pat = get_cell("A2");
    Value hv = deref(*head_pat);
    if (hv.is_unbound())
        return throw_iso_error(make_instantiation_error());
    std::string key;
    if (hv.tag == Value::Tag::Atom) key = hv.s + "/0";
    else if (hv.tag == Value::Tag::Compound) key = hv.s;
    else return throw_iso_error(make_type_error("callable", hv));
    // clause/2 in our runtime only sees dynamic-db clauses; static
    // predicates compiled to bytecode aren''t introspectable here.
    if (dynamic_db.find(key) == dynamic_db.end()) return false;
    RetractIterator it;
    it.key = std::move(key);
    it.pattern = head_pat;
    it.body_pattern = body_pat;
    it.is_clause_only = true;
    it.next_idx = 0;
    it.after_pc = after_pc;
    retract_iters.push_back(std::move(it));
    return retract_try_next();
}

// dispatch_current_predicate: prebuild the candidate-key list at
// entry (drawing from labels + dynamic_db, deduped + sorted for
// deterministic enumeration), filter by any ground portion of the
// Name/Arity spec, and delegate the per-match unification +
// CP-push to current_pred_try_next.
bool WamState::dispatch_current_predicate(std::size_t after_pc) {
    Value spec = *get_cell("A1");
    if (spec.is_unbound())
        return throw_iso_error(make_instantiation_error());
    if (spec.tag != Value::Tag::Compound || spec.s != "//2"
        || spec.args.size() != 2)
        return throw_iso_error(make_type_error("predicate_indicator", spec));
    CellPtr name_cell = spec.args[0];
    CellPtr arity_cell = spec.args[1];
    Value name_v = *name_cell;
    Value arity_v = *arity_cell;
    if (!name_v.is_unbound() && name_v.tag != Value::Tag::Atom)
        return throw_iso_error(make_type_error("atom", name_v));
    if (!arity_v.is_unbound() && arity_v.tag != Value::Tag::Integer)
        return throw_iso_error(make_type_error("integer", arity_v));
    std::set<std::string> all_keys;
    for (auto& p : labels) all_keys.insert(p.first);
    for (auto& p : dynamic_db) all_keys.insert(p.first);
    std::vector<std::string> matched;
    matched.reserve(all_keys.size());
    for (auto& k : all_keys) {
        auto slash = k.rfind(\'/\');
        if (slash == std::string::npos) continue;
        std::string n = k.substr(0, slash);
        std::int64_t a;
        try { a = std::stoll(k.substr(slash + 1)); }
        catch (...) { continue; }
        if (!name_v.is_unbound() && name_v.s != n) continue;
        if (!arity_v.is_unbound() && arity_v.i != a) continue;
        matched.push_back(k);
    }
    if (matched.empty()) return false;
    CurrentPredIterator it;
    it.keys = std::move(matched);
    it.next_idx = 0;
    it.name_cell = name_cell;
    it.arity_cell = arity_cell;
    it.after_pc = after_pc;
    current_pred_iters.push_back(std::move(it));
    return current_pred_try_next();
}

bool WamState::current_pred_try_next() {
    if (current_pred_iters.empty()) return false;
    CurrentPredIterator& it = current_pred_iters.back();
    while (it.next_idx < it.keys.size()) {
        std::size_t i = it.next_idx;
        std::size_t mark = trail.size();
        const std::string& key = it.keys[i];
        auto slash = key.rfind(\'/\');
        std::string n = key.substr(0, slash);
        std::int64_t a;
        try { a = std::stoll(key.substr(slash + 1)); }
        catch (...) { ++it.next_idx; continue; }
        Value n_v = Value::Atom(n);
        Value a_v = Value::Integer(a);
        bool ok = true;
        if (it.name_cell->is_unbound()) bind_cell(it.name_cell, n_v);
        else if (!(*it.name_cell == n_v)) ok = false;
        if (ok) {
            if (it.arity_cell->is_unbound()) bind_cell(it.arity_cell, a_v);
            else if (!(*it.arity_cell == a_v)) ok = false;
        }
        if (ok) {
            it.next_idx = i + 1;
            bool has_more = (it.next_idx < it.keys.size());
            std::size_t saved_after_pc = it.after_pc;
            if (has_more) {
                ChoicePoint cp_;
                cp_.alt_pc            = current_pred_next_pc;
                cp_.saved_cp          = cp;
                cp_.trail_mark        = mark;
                cp_.cut_barrier       = cut_barrier;
                cp_.saved_regs        = regs;
                cp_.saved_mode_stack  = mode_stack;
                cp_.saved_env_stack   = env_stack;
                cp_.saved_body_frames = body_frames;
                choice_points.push_back(std::move(cp_));
            } else {
                current_pred_iters.pop_back();
            }
            if (saved_after_pc == 0) { halt = true; return true; }
            pc = saved_after_pc;
            cp = 0;
            return true;
        }
        while (trail.size() > mark) {
            TrailEntry te = std::move(trail.back());
            trail.pop_back();
            *te.cell = std::move(te.prev);
        }
        ++it.next_idx;
    }
    current_pred_iters.pop_back();
    return false;
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
        // For clause/2: split the stored term into Head / Body. A
        // bare-fact entry has Body = true; a :-/2 entry uses its
        // args. We then unify pattern with Head and body_pattern
        // with Body. retract/1 just unifies the full term with
        // pattern (no body slot).
        bool matched = false;
        if (it.is_clause_only) {
            CellPtr head_part, body_part;
            if (fresh->tag == Value::Tag::Compound
                && fresh->s == ":-/2" && fresh->args.size() == 2) {
                head_part = fresh->args[0];
                body_part = fresh->args[1];
            } else {
                head_part = fresh;
                body_part = std::make_shared<Cell>(Value::Atom("true"));
            }
            matched = unify_cells(it.pattern, head_part)
                   && unify_cells(it.body_pattern, body_part);
        } else {
            matched = unify_cells(it.pattern, fresh);
        }
        if (matched) {
            // Match. We need to:
            // 1) push a CP (if more candidates exist) whose trail
            //    mark is BEFORE the unification, so backtrack undoes
            //    the pattern''s bindings and the pattern is reusable;
            // 2) for retract: remove the matched clause from vec.
            //    for clause/2: leave it in place.
            // 3) leave the unification bindings in place (per ISO
            //    retract / clause-success semantics) and proceed.
            // The CP''s saved regs/etc are the current values — at
            // entry to retract_try_next, before any binding — which
            // match what the caller of dispatch_retract set up.
            if (it.is_clause_only) {
                // Don''t remove; advance past this index for the
                // next backtrack iteration.
                it.next_idx = i + 1;
            } else {
                vec.erase(vec.begin() + i);
                it.next_idx = i;
            }
            bool has_more = (it.next_idx < vec.size());
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

// ----------------------------------------------------------------------
// Term parser — reads canonical syntax produced by render().
// Used by term_to_atom/2 (reverse mode). Doesn''t handle operator
// syntax; just integer/float/atom/var/list/compound in canonical form.
// ----------------------------------------------------------------------

void WamState::parse_skip_ws(const std::string& s, std::size_t& pos) {
    while (pos < s.size() && (s[pos] == '' '' || s[pos] == ''\\t''
                              || s[pos] == ''\\n'')) ++pos;
}

bool WamState::parse_number(const std::string& s, std::size_t& pos,
                            CellPtr& out) {
    std::size_t start = pos;
    if (pos < s.size() && s[pos] == ''-'') ++pos;
    std::size_t digits_start = pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])))
        ++pos;
    if (pos == digits_start) { pos = start; return false; }
    bool is_float = false;
    if (pos < s.size() && s[pos] == ''.''
        && pos + 1 < s.size()
        && std::isdigit(static_cast<unsigned char>(s[pos + 1])))
    {
        is_float = true;
        ++pos; // consume ''.''
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])))
            ++pos;
    }
    // Optional exponent: [eE][+-]?digits
    if (pos < s.size() && (s[pos] == ''e'' || s[pos] == ''E'')) {
        std::size_t save = pos;
        ++pos;
        if (pos < s.size() && (s[pos] == ''+'' || s[pos] == ''-'')) ++pos;
        std::size_t exp_start = pos;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])))
            ++pos;
        if (pos == exp_start) { pos = save; }
        else { is_float = true; }
    }
    std::string num_str = s.substr(start, pos - start);
    try {
        if (is_float) {
            double d = std::stod(num_str);
            out = std::make_shared<Cell>(Value::Float(d));
        } else {
            std::int64_t n = std::stoll(num_str);
            out = std::make_shared<Cell>(Value::Integer(n));
        }
    } catch (...) { pos = start; return false; }
    return true;
}

bool WamState::parse_list(const std::string& s, std::size_t& pos,
                          CellPtr& out) {
    // Caller already at ''[''. Consume it and walk elements.
    if (pos >= s.size() || s[pos] != ''['') return false;
    ++pos;
    parse_skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '']'') {
        ++pos;
        out = std::make_shared<Cell>(Value::Atom("[]"));
        return true;
    }
    // Collect elements.
    std::vector<CellPtr> elems;
    CellPtr first;
    if (!parse_term(s, pos, first)) return false;
    elems.push_back(first);
    parse_skip_ws(s, pos);
    while (pos < s.size() && s[pos] == '','') {
        ++pos;
        parse_skip_ws(s, pos);
        CellPtr e;
        if (!parse_term(s, pos, e)) return false;
        elems.push_back(e);
        parse_skip_ws(s, pos);
    }
    CellPtr tail;
    if (pos < s.size() && s[pos] == ''|'') {
        ++pos;
        parse_skip_ws(s, pos);
        if (!parse_term(s, pos, tail)) return false;
        parse_skip_ws(s, pos);
    } else {
        tail = std::make_shared<Cell>(Value::Atom("[]"));
    }
    if (pos >= s.size() || s[pos] != '']'') return false;
    ++pos;
    // Build the [|]/2 spine right-to-left.
    CellPtr result = tail;
    for (auto it = elems.rbegin(); it != elems.rend(); ++it) {
        std::vector<CellPtr> args;
        args.push_back(*it);
        args.push_back(result);
        result = std::make_shared<Cell>(
            Value::Compound("[|]/2", std::move(args)));
    }
    out = result;
    return true;
}

bool WamState::parse_atom_or_compound(const std::string& s,
                                      std::size_t& pos,
                                      CellPtr& out) {
    std::size_t start = pos;
    std::string name;
    // Quoted atom or bareword. Char codes: 39 = '', 92 = \\.
    if (pos < s.size() && static_cast<unsigned char>(s[pos]) == 39) {
        ++pos;
        while (pos < s.size()
               && static_cast<unsigned char>(s[pos]) != 39) {
            if (static_cast<unsigned char>(s[pos]) == 92
                && pos + 1 < s.size()) {
                ++pos;
                name.push_back(s[pos]);
            } else {
                name.push_back(s[pos]);
            }
            ++pos;
        }
        if (pos >= s.size()) { pos = start; return false; }
        ++pos; // consume closing quote
    } else if (pos < s.size()
               && std::islower(static_cast<unsigned char>(s[pos]))) {
        // Bareword: lowercase start, then [a-zA-Z0-9_]*.
        while (pos < s.size()
               && (std::isalnum(static_cast<unsigned char>(s[pos]))
                   || s[pos] == ''_''))
        {
            name.push_back(s[pos]);
            ++pos;
        }
    } else {
        return false;
    }
    // Optional compound: name "(" args ")".
    parse_skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ''('') {
        ++pos;
        parse_skip_ws(s, pos);
        std::vector<CellPtr> args;
        CellPtr first;
        if (!parse_term(s, pos, first)) return false;
        args.push_back(first);
        parse_skip_ws(s, pos);
        while (pos < s.size() && s[pos] == '','') {
            ++pos;
            parse_skip_ws(s, pos);
            CellPtr a;
            if (!parse_term(s, pos, a)) return false;
            args.push_back(a);
            parse_skip_ws(s, pos);
        }
        if (pos >= s.size() || s[pos] != '')'') return false;
        ++pos;
        std::string functor = name + "/" + std::to_string(args.size());
        out = std::make_shared<Cell>(
            Value::Compound(functor, std::move(args)));
        return true;
    }
    // Plain atom.
    out = std::make_shared<Cell>(Value::Atom(name));
    return true;
}

bool WamState::parse_term(const std::string& s, std::size_t& pos,
                          CellPtr& out) {
    parse_skip_ws(s, pos);
    if (pos >= s.size()) return false;
    char c = s[pos];
    // Variable: uppercase letter or underscore start (no compound form).
    if (std::isupper(static_cast<unsigned char>(c)) || c == ''_'') {
        std::size_t start = pos;
        while (pos < s.size()
               && (std::isalnum(static_cast<unsigned char>(s[pos]))
                   || s[pos] == ''_''))
            ++pos;
        std::string name = s.substr(start, pos - start);
        out = std::make_shared<Cell>(
            Value::Unbound("_P" + std::to_string(var_counter++)));
        return true;
    }
    // List.
    if (c == ''['') return parse_list(s, pos, out);
    // Number (possibly negative).
    if (c == ''-'' || std::isdigit(static_cast<unsigned char>(c))) {
        std::size_t save = pos;
        if (parse_number(s, pos, out)) return true;
        pos = save;
    }
    // Atom or compound.
    return parse_atom_or_compound(s, pos, out);
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
        if (key == "sub_atom/5" || key == "sub_string/5")
            return dispatch_sub_atom(after_pc);
        if (key == "retract/1") return dispatch_retract(after_pc);
        if (key == "with_output_to/2") {
            // Set cp = after_pc so the builtin captures the correct
            // continuation. The builtin saves cp as saved_cp; without
            // this, it would use pc + 1 (which only makes sense for
            // the direct BuiltinCall instr — not for goal-term
            // dispatch where pc points at a synthetic op).
            cp = after_pc;
            // Saved_cp inside the builtin will then == after_pc.
            // We call builtin() directly here so the goal-term path
            // doesn''t fall through to the generic post-builtin
            // continuation override below.
            if (!builtin(key, 2)) return false;
            return true;
        }
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
    // Strip Module:Goal wrapping (compound ":/2" -- treat as transparent
    // for our single-module runtime; the module qualifier becomes a no-op).
    while (goal.tag == Value::Tag::Compound
           && goal.s == ":/2" && goal.args.size() == 2) {
        goal = deref(*goal.args[1]);
    }
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

bool WamState::dispatch_phrase_call(bool has_rest, std::size_t after_pc) {
    // phrase(Body, List)        : extras = [List, []]
    // phrase(Body, List, Rest)  : extras = [List, Rest]
    CellPtr goal_cell = get_cell("A1");
    CellPtr list_cell = get_cell("A2");
    CellPtr rest_cell = has_rest
        ? get_cell("A3")
        : std::make_shared<Cell>(Value::Atom("[]"));
    Value goal = deref(*goal_cell);
    // Strip Module:Goal wrapping so phrase(user:greeting, L) works.
    while (goal.tag == Value::Tag::Compound
           && goal.s == ":/2" && goal.args.size() == 2) {
        goal = deref(*goal.args[1]);
    }
    std::string base_name;
    std::size_t base_arity = 0;
    std::vector<CellPtr> all_args;
    if (goal.tag == Value::Tag::Atom) {
        base_name = goal.s;
    } else if (goal.tag == Value::Tag::Compound) {
        auto slash = goal.s.rfind(\'/\');
        if (slash == std::string::npos) return false;
        base_name = goal.s.substr(0, slash);
        try { base_arity = std::stoul(goal.s.substr(slash + 1)); }
        catch (...) { return false; }
        if (base_arity != goal.args.size()) return false;
        all_args = goal.args;
    } else {
        return throw_iso_error(make_type_error("callable", goal));
    }
    all_args.push_back(list_cell);
    all_args.push_back(rest_cell);
    std::size_t new_arity = base_arity + 2;
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

// Output emission router. with_output_to/2 pushes an
// OutputCaptureFrame; while one is on top, all I/O builtins write
// here instead of stdout, so the goal''s output gets captured.
void WamState::emit_output(const std::string& s) {
    if (!output_capture_frames.empty()) {
        output_capture_frames.back().buffer += s;
    } else {
        std::fwrite(s.data(), 1, s.size(), stdout);
    }
}

void WamState::emit_output_char(char c) {
    if (!output_capture_frames.empty()) {
        output_capture_frames.back().buffer.push_back(c);
    } else {
        std::fputc(static_cast<unsigned char>(c), stdout);
    }
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
        // OutputCaptureFrame: with_output_to/2 goal failed. Drop the
        // frame and the partially-accumulated buffer; propagate the
        // failure outside.
        if (!output_capture_frames.empty()
            && choice_points.size() == output_capture_frames.back().base_cp_count
            && (aggregate_frames.empty()
                || aggregate_frames.back().base_cp_count
                       <= output_capture_frames.back().base_cp_count)
            && (catcher_frames.empty()
                || catcher_frames.back().base_cp_count
                       <= output_capture_frames.back().base_cp_count))
        {
            OutputCaptureFrame f = std::move(output_capture_frames.back());
            output_capture_frames.pop_back();
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
    retract_iters.clear();
    current_pred_iters.clear();
    pc = it->second;
    cp = 0;
    cut_barrier = 0;
    indexed_entry = false;
    halt = false;
    return run();
}

Program::Setup& Program::setup_hook() {
    static Setup s = nullptr;
    return s;
}

void Program::register_setup(Setup s) { setup_hook() = s; }
void Program::apply_setup(WamState& vm) { if (setup_hook()) setup_hook()(vm); }

// ----------------------------------------------------------------------
// LMDB FactSource implementation (v1).
// Gated by WAM_CPP_ENABLE_LMDB. When the flag is absent, only the
// stub cpp_load_lmdb_fact_source below is compiled -- it returns
// false and leaves dynamic_db untouched.
// ----------------------------------------------------------------------

#ifdef WAM_CPP_ENABLE_LMDB
#include <filesystem>
#include <stdexcept>

LmdbFactSource::LmdbFactSource(const std::string& env_path,
                               const char* db_name) {
    int rc = mdb_env_create(&env_);
    if (rc != MDB_SUCCESS) {
        throw std::runtime_error(
            std::string("mdb_env_create: ") + mdb_strerror(rc));
    }
    // Allow named sub-DBs; v1 only opens one per LmdbFactSource but
    // reserves the headroom so users can keep multiple predicates
    // in one env via repeated load calls with different db_name.
    mdb_env_set_maxdbs(env_, 16);
    // LMDB envs come in two flavours: a directory containing
    // data.mdb + lock.mdb (no MDB_NOSUBDIR), or a single file path
    // where the file is data.mdb and lock.mdb sits alongside
    // (MDB_NOSUBDIR). Pick the right flag by probing the path --
    // retrying mdb_env_open on the same env after failure is not
    // safe per the LMDB docs.
    unsigned int flags = MDB_RDONLY;
    std::error_code ec;
    if (std::filesystem::is_regular_file(env_path, ec)) {
        flags |= MDB_NOSUBDIR;
    }
    rc = mdb_env_open(env_, env_path.c_str(), flags, 0664);
    if (rc != MDB_SUCCESS) {
        mdb_env_close(env_);
        env_ = nullptr;
        throw std::runtime_error(
            std::string("mdb_env_open ") + env_path + ": "
            + mdb_strerror(rc));
    }
    MDB_txn* txn = nullptr;
    rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) {
        mdb_env_close(env_);
        env_ = nullptr;
        throw std::runtime_error(
            std::string("mdb_txn_begin: ") + mdb_strerror(rc));
    }
    MDB_dbi dbi;
    rc = mdb_dbi_open(txn, db_name, 0, &dbi);
    if (rc != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env_);
        env_ = nullptr;
        throw std::runtime_error(
            std::string("mdb_dbi_open ")
            + (db_name ? db_name : "<default>") + ": "
            + mdb_strerror(rc));
    }
    mdb_txn_commit(txn);
    dbi_ = dbi;
    dbi_open_ = true;
}

LmdbFactSource::~LmdbFactSource() {
    if (env_) {
        if (dbi_open_) {
            mdb_dbi_close(env_, dbi_);
        }
        mdb_env_close(env_);
        env_ = nullptr;
    }
}

void LmdbFactSource::stream_all(
    const std::function<void(std::string_view,
                             std::string_view)>& sink) {
    MDB_txn* txn = nullptr;
    int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) {
        throw std::runtime_error(
            std::string("stream_all/txn_begin: ")
            + mdb_strerror(rc));
    }
    MDB_cursor* cur = nullptr;
    rc = mdb_cursor_open(txn, dbi_, &cur);
    if (rc != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        throw std::runtime_error(
            std::string("stream_all/cursor_open: ")
            + mdb_strerror(rc));
    }
    MDB_val k{}, v{};
    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        std::string_view ksv(static_cast<const char*>(k.mv_data),
                             k.mv_size);
        std::string_view vsv(static_cast<const char*>(v.mv_data),
                             v.mv_size);
        sink(ksv, vsv);
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    if (rc != MDB_NOTFOUND) {
        throw std::runtime_error(
            std::string("stream_all/cursor_get: ")
            + mdb_strerror(rc));
    }
}

LmdbFactSource::Meta LmdbFactSource::read_meta(const char* db_name) {
    Meta meta;
    MDB_txn* txn = nullptr;
    int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) {
        throw std::runtime_error(
            std::string("read_meta/txn_begin: ")
            + mdb_strerror(rc));
    }
    MDB_dbi meta_dbi;
    rc = mdb_dbi_open(txn, "__meta__", 0, &meta_dbi);
    if (rc == MDB_NOTFOUND) {
        // __meta__ sub-DB absent -- transitional accommodation.
        mdb_txn_abort(txn);
        return meta;  // present == false
    }
    if (rc != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        throw std::runtime_error(
            std::string("read_meta/dbi_open(__meta__): ")
            + mdb_strerror(rc));
    }
    // Key prefix per design doc: empty when data DB is the default
    // unnamed DB, "<db_name>:" when a named sub-DB is in use.
    std::string prefix = (db_name && *db_name)
        ? (std::string(db_name) + ":") : std::string();
    auto get_key = [&](const std::string& name) -> std::string {
        std::string full = prefix + name;
        MDB_val K{full.size(), const_cast<char*>(full.data())};
        MDB_val V{};
        int gr = mdb_get(txn, meta_dbi, &K, &V);
        if (gr == MDB_NOTFOUND) return std::string();
        if (gr != MDB_SUCCESS) {
            throw std::runtime_error(
                std::string("read_meta/get ") + full + ": "
                + mdb_strerror(gr));
        }
        return std::string(static_cast<const char*>(V.mv_data),
                           V.mv_size);
    };
    std::string sv = get_key("schema_version");
    std::string pr = get_key("predicate");
    std::string cs = get_key("columns");
    mdb_txn_abort(txn);
    if (sv.empty() && pr.empty() && cs.empty()) {
        // __meta__ sub-DB exists but holds no entries for this DB.
        // Treat as absent so the loader emits a warning.
        return meta;
    }
    meta.present = true;
    try { meta.schema_version = std::stoi(sv); }
    catch (...) { meta.schema_version = 0; }
    meta.predicate = pr;
    // Split columns by comma. Trailing/leading whitespace is not
    // tolerated -- the design doc specifies ASCII exact match.
    std::string cur;
    for (char c : cs) {
        if (c == \',\') {
            meta.columns.push_back(std::move(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty() || !cs.empty()) {
        meta.columns.push_back(std::move(cur));
    }
    return meta;
}

bool cpp_load_lmdb_fact_source(WamState& vm,
                               const std::string& functor_arity_key,
                               const std::string& env_path,
                               const char* db_name,
                               const LmdbLoadOptions& opts) {
    std::string dbn = db_name ? db_name : "";
    auto key = std::make_pair(env_path, dbn);
    if (vm.loaded_lmdb_sources.count(key)) {
        return true;  // idempotent
    }
    try {
        LmdbFactSource src(env_path, db_name);
        // Validate the __meta__ sub-DB before loading. Per design
        // doc resolved-question 1: meta is REQUIRED to match the
        // registered predicate; absent meta is a warning, mismatch
        // is a hard error.
        LmdbFactSource::Meta meta = src.read_meta(db_name);
        if (meta.present) {
            if (meta.schema_version != 1) {
                throw std::runtime_error(
                    "schema_version mismatch: expected 1, got "
                    + std::to_string(meta.schema_version));
            }
            if (meta.predicate != functor_arity_key) {
                throw std::runtime_error(
                    "predicate mismatch: file is for "
                    + meta.predicate + ", registered as "
                    + functor_arity_key);
            }
            // v1: arity 2, so exactly 2 columns expected.
            if (meta.columns.size() != 2) {
                throw std::runtime_error(
                    "columns count mismatch: expected 2, got "
                    + std::to_string(meta.columns.size()));
            }
        } else {
            std::fprintf(stderr,
                "[wam-cpp] warning: %s (%s) has no __meta__ "
                "sub-DB; loading without schema validation\\n",
                env_path.c_str(), functor_arity_key.c_str());
        }
        // Recover functor name from the key "name/arity" for the
        // synthesised compound. v1 only supports arity 2; the key
        // is assumed well-formed (caller is the codegen).
        auto slash = functor_arity_key.find(\'/\');
        std::string functor = (slash == std::string::npos)
            ? functor_arity_key
            : functor_arity_key.substr(0, slash);
        // Phase 2 enforcement: when opts.unique_check is set, the
        // value column (arg2) is asserted unique across all rows.
        // LMDB keys (arg1) are unique by construction without
        // DUPSORT so we do not need to check them here; the
        // meaningful check is on the value column. Per
        // RELATION_POLICY_DECLARATIONS.md the on_duplicate policy
        // decides what to do when the contract is violated.
        //
        // Build into a local vector and only commit to dynamic_db
        // after stream_all returns without throwing -- otherwise a
        // partial load would leave dynamic_db in a half-populated
        // state and queries against the predicate would return
        // wrong answers even after the loader reported failure.
        std::vector<CellPtr> staged;
        std::unordered_map<std::string, std::size_t> seen_value_to_row;
        src.stream_all([&](std::string_view ks, std::string_view vs) {
            // LMDB stores named sub-DBs as pointer records in the
            // parent DB. When the data DB IS the default unnamed
            // one, iterating it surfaces "__meta__" as a phantom
            // row whose value is an internal sub-DB pointer.
            // Skip it -- it is never user data.
            if (ks == "__meta__") return;
            std::string vstr(vs);
            if (opts.unique_check) {
                auto it = seen_value_to_row.find(vstr);
                if (it != seen_value_to_row.end()) {
                    switch (opts.on_duplicate) {
                    case LmdbLoadOptions::OnDup::throw_:
                        throw std::runtime_error(
                            "duplicate value \\"" + vstr
                            + "\\" with unique(true) and "
                              "on_duplicate(throw)");
                    case LmdbLoadOptions::OnDup::warn:
                        std::fprintf(stderr,
                            "[wam-cpp] warning: duplicate value "
                            "in %s: %s\\n",
                            functor_arity_key.c_str(),
                            vstr.c_str());
                        break;
                    case LmdbLoadOptions::OnDup::overwrite:
                        // Replace the earlier row with this one.
                        // staged[it->second] holds the prior cell.
                        {
                            auto k_cell = std::make_shared<Cell>(
                                Value::Atom(std::string(ks)));
                            auto v_cell = std::make_shared<Cell>(
                                Value::Atom(vstr));
                            std::vector<CellPtr> args;
                            args.push_back(k_cell);
                            args.push_back(v_cell);
                            staged[it->second] = std::make_shared<Cell>(
                                Value::Compound(
                                    functor + "/2",
                                    std::move(args)));
                        }
                        return;
                    case LmdbLoadOptions::OnDup::first_wins:
                        return;  // skip the duplicate
                    case LmdbLoadOptions::OnDup::keep_all:
                        break;   // fall through and append
                    }
                }
            }
            auto k_cell = std::make_shared<Cell>(
                Value::Atom(std::string(ks)));
            auto v_cell = std::make_shared<Cell>(
                Value::Atom(vstr));
            std::vector<CellPtr> args;
            args.push_back(k_cell);
            args.push_back(v_cell);
            std::size_t pos = staged.size();
            staged.push_back(std::make_shared<Cell>(
                Value::Compound(functor + "/2", std::move(args))));
            if (opts.unique_check) {
                seen_value_to_row[vstr] = pos;
            }
        });
        // Commit the staged rows only after stream_all completes
        // without throwing.
        auto& bucket = vm.dynamic_db[functor_arity_key];
        // Apply order(...) policy. sort_keys is empty when the
        // declared order is trivially satisfied by LMDB natural
        // iteration (default, or [arg(1)] asc). std::sort is stable
        // when applied to already-sorted input -- worst case here
        // when sort_keys=[arg(2)] is O(n log n) but only fires
        // when the user actually asked for a non-trivial sort.
        if (!opts.sort_keys.empty()) {
            std::sort(staged.begin(), staged.end(),
                [&](const CellPtr& a, const CellPtr& b) {
                    for (const auto& sk : opts.sort_keys) {
                        const auto& av = a->args[sk.column - 1]->s;
                        const auto& bv = b->args[sk.column - 1]->s;
                        if (av != bv) {
                            return sk.ascending ? (av < bv)
                                                : (av > bv);
                        }
                    }
                    return false;
                });
        }
        bucket.insert(bucket.end(),
                      std::make_move_iterator(staged.begin()),
                      std::make_move_iterator(staged.end()));
        vm.loaded_lmdb_sources.insert(key);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "cpp_load_lmdb_fact_source(%s, %s): %s\\n",
                     functor_arity_key.c_str(),
                     env_path.c_str(),
                     e.what());
        return false;
    }
}
#else
// Stub when WAM_CPP_ENABLE_LMDB is not defined. Matches the C
// target''s stub (wam_c_target.pl:2392-2398): returns false and
// touches nothing, so generated code that calls this still links.
bool cpp_load_lmdb_fact_source(WamState&,
                               const std::string&,
                               const std::string&,
                               const char*,
                               const LmdbLoadOptions&) {
    return false;
}
#endif // WAM_CPP_ENABLE_LMDB

} // namespace wam_cpp
').

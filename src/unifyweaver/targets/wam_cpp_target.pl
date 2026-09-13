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
:- use_module(wam_cpp_templates,
              [cpp_render_template/3, cpp_preflight_stache/0,
               cpp_preflight_lowered_function/0]).
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
    to_string(K, KS), to_string(V, VS), to_string(R, RS), to_string(W, WS0),
    % The witness spec is emitted as a quoted atom in the WAM text (e.g.
    % 'Y3' or 'Y1;Y2'), so the parser token keeps the surrounding single
    % quotes. Strip them — otherwise witness_regs holds "'Y3'" and the
    % runtime's get_cell("'Y3'") fails, leaving witness_cells empty and
    % silently degrading bagof/setof grouping to a flat findall.
    strip_surrounding_squotes(WS0, WS),
    escape_cpp_string(KS, EK), escape_cpp_string(VS, EV),
    escape_cpp_string(RS, ER), escape_cpp_string(WS, EW),
    format(atom(Code),
           'Instruction::BeginAggregate("~w", "~w", "~w", "~w")',
           [EK, EV, ER, EW]).
strip_surrounding_squotes(S0, S) :-
    string_chars(S0, Cs),
    (   Cs = ['\''|Rest], append(Mid, ['\''], Rest)
    ->  string_chars(S, Mid)
    ;   S = S0
    ).
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
            ( cpp_predicate_wam_text(PI, Options, WamText),
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
            current_pred_next, foreign_next_clause],
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
    ForeignNextClausePC is CatchReturnPC + 14,
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
    % D43 store-backed seek fact sources (cpp_wam_fact_sources option).
    % One register_seek_fact_source(...) call per source, appended to
    % wam_cpp_setup so the sources are live before any query runs.
    seek_sources_from_options(Options, SeekSources),
    findall(SeekLine, (
        member(seek_source(SKey, SKind, SPath), SeekSources),
        emit_seek_register_call(SKey, SKind, SPath, SeekLine)
    ), SeekLines),
    atomic_list_concat(SeekLines, '\n', SeekRegBody),
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
    vm.foreign_next_clause_pc = ~w;
~w
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
    CurrentPredNextPC, ForeignNextClausePC,
    LabelBody, InstrBody, LmdbLoadBody, SeekRegBody]).

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

% ---------------------------------------------------------------------------
% D43 store-backed seek fact sources (cpp_wam_fact_sources codegen option).
%
% Analogous to rust_wam_fact_sources / go_wam_fact_sources: a store-backed P/2
% predicate is compiled to a two-instruction body [call_foreign, proceed] so
% callers reach it by label exactly like any other predicate; the seek source
% itself is registered in wam_cpp_setup and dispatched at runtime via
% CallForeign → dispatch_foreign_call (keyed lazy seek). Distinct from the
% eager, whole-DB cpp_fact_sources (LMDB) option, which stays as-is.
%
%   cpp_wam_fact_sources([ source(P/2, indexed(Prefix)) | source(P/2, lmdb(Dir)) ])
% ---------------------------------------------------------------------------

%% seek_sources_from_options(+Options, -Sources)
%  Resolve the cpp_wam_fact_sources option into seek_source(Key, Kind, Path)
%  terms. Key is "Name/Arity"; Kind is "indexed" | "lmdb"; Path is the absolute
%  store prefix (indexed) or dir (lmdb), baked in at build time so the binary
%  needs no runtime store argument (mirrors the Go/Rust lanes).
seek_sources_from_options(Options, Sources) :-
    (   member(cpp_wam_fact_sources(Specs), Options)
    ->  findall(seek_source(Key, Kind, AbsPath), (
            member(source(PI, Spec), Specs),
            cpp_seek_source_pred(PI, Pred, Arity),
            validate_seek_v1_arity(Pred, Arity),
            format(atom(Key), '~w/~w', [Pred, Arity]),
            cpp_seek_source_spec(Spec, Kind, Path0),
            cpp_store_abs_path(Path0, AbsPath)
        ), Sources)
    ;   Sources = []
    ).

cpp_seek_source_pred(_:Name/Ar, Name, Ar) :- !.
cpp_seek_source_pred(Name/Ar, Name, Ar).

% v1 serves arity 2 only (mirrors the Rust/Go contract); reject anything else
% loudly so a mis-declared source is a codegen-time error, not a silent skip.
validate_seek_v1_arity(_, 2) :- !.
validate_seek_v1_arity(Name, Arity) :-
    throw(error(domain_error(cpp_wam_fact_source_arity_2, Name/Arity), _)).

cpp_seek_source_spec(indexed(Prefix), indexed, Prefix).
cpp_seek_source_spec(lmdb(Dir), lmdb, Dir).

% Resolve a store path/prefix to an absolute path relative to the cwd so the
% baked-in store location survives a different run directory.
cpp_store_abs_path(Path, Abs) :-
    atom_string(Path, PathStr),
    working_directory(Cwd, Cwd),
    (   catch(absolute_file_name(PathStr, Abs0, [relative_to(Cwd)]), _, fail),
        Abs0 \== []
    ->  Abs = Abs0
    ;   Abs = PathStr
    ).

%% cpp_wam_fact_source_spec(+P, +Arity, +Options, -Spec)
%  True when Options declare a store-backed source for P/Arity (arity 2).
cpp_wam_fact_source_spec(P, Arity, Options, Spec) :-
    Arity =:= 2,
    member(cpp_wam_fact_sources(Sources), Options),
    member(source(PI, Spec), Sources),
    cpp_seek_source_pred(PI, Name, Ar),
    Name == P, Ar =:= Arity.

%% cpp_fact_stream_wam_text(+P, +Arity, -WamText)
%  The two-instruction predicate body served for a store fact source: dispatch
%  to CallForeign (which routes to the seek source) then proceed.
cpp_fact_stream_wam_text(P, Arity, WamText) :-
    format(atom(WamText),
        '~w/~w:\n    call_foreign ~w/~w ~w\n    proceed\n',
        [P, Arity, P, Arity, Arity]).

%% cpp_predicate_wam_text(+PI, +Options, -WamText)
%  WAM text for one predicate: the fact-stream body when PI is a declared
%  cpp_wam_fact_sources source, else the shared compiler's output.
cpp_predicate_wam_text(PI, Options, WamText) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    (   cpp_wam_fact_source_spec(Pred, Arity, Options, _Spec)
    ->  cpp_fact_stream_wam_text(Pred, Arity, WamText)
    ;   compile_predicate_to_wam(PI,
            [inline_bagof_setof(true), ite_use_y_level(true)], WamText)
    ).

%% emit_seek_register_call(+Key, +Kind, +Path, -Line)
%  Render one register_seek_fact_source(...) call for wam_cpp_setup.
emit_seek_register_call(Key, Kind, Path, Line) :-
    escape_cpp_string(Key, EKey),
    escape_cpp_string(Kind, EKind),
    escape_cpp_string(Path, EPath),
    format(atom(Line),
        '    vm.register_seek_fact_source("~w", "~w", "~w");',
        [EKey, EKind, EPath]).

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
instr_to_setup_line(foreign_next_clause, _Labels, Line) :- !,
    Line = '    vm.instrs.push_back(Instruction::ForeignNextClause());'.
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
    % Complete every template read and all source generation before creating
    % the project directory. A failed asset must not leave usable-looking
    % partial C++ files behind.
    cpp_preflight_stache,
    cpp_preflight_lowered_function,
    compile_wam_runtime_header_to_cpp(Options, HeaderCode),
    compile_wam_runtime_to_cpp(Options, RuntimeCode),
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    emit_setup_function(Predicates, Options, SetupCpp),
    cpp_render_template(generated_program,
                        [predicates_code=PredicatesCode, setup_code=SetupCpp],
                        ProgramCode),
    (   option(emit_main(true), Options, false)
    ->  emit_main_shim(MainCode),
        MainFiles = ['main.cpp'-MainCode]
    ;   MainFiles = []
    ),
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'cpp', CppDir),
    make_directory_path(CppDir),
    append(['wam_runtime.h'-HeaderCode,
            'wam_runtime.cpp'-RuntimeCode,
            'generated_program.cpp'-ProgramCode], MainFiles, Files),
    forall(member(File-Content, Files),
           (directory_file_path(CppDir, File, Path),
            write_text_file(Path, Content))).

%% emit_main_shim(-MainCpp)
%  A tiny CLI driver: argv[1] is the predicate key, remaining args are
%  parsed as Prolog-style terms (atom, integer, compound foo(a,b),
%  list [a,b,c]). Prints "true" / "false" and exits 0 / 1.
emit_main_shim(MainCpp) :-
    cpp_render_template(main_shim, [], MainCpp).

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
            ( cpp_predicate_wam_text(PI, Options1, WamCode),
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
handle_compile_error(_, _, error(cpp_wam_stache_failure(Id, Path, Cause), Context)) :-
    !,
    throw(error(cpp_wam_stache_failure(Id, Path, Cause), Context)).
handle_compile_error(_, _, error(cpp_wam_stache_empty(Id, Path), Context)) :-
    !,
    throw(error(cpp_wam_stache_empty(Id, Path), Context)).
handle_compile_error(_, _, Err) :-
    lowered_function_template_error(Err),
    !,
    throw(Err).
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

% A template edit between project preflight and a lowered predicate must not
% become a silently omitted predicate under the default warn policy.
lowered_function_template_error(error(cpp_wam_template_load(lowered_function, _, _), _)).
lowered_function_template_error(error(cpp_wam_template_empty(lowered_function, _), _)).
lowered_function_template_error(error(cpp_wam_template_tags(lowered_function, _), _)).
lowered_function_template_error(error(domain_error(cpp_wam_lowered_function_variables, _), _)).

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
% C++ runtime header & source. These are loaded from generation-time assets;
% the emitted project contains complete source files. The header wrapper
% below enables LMDB for projects with fact sources.
% ============================================================================

compile_wam_runtime_header_to_cpp(Options, Code) :-
    compile_wam_runtime_header_body_to_cpp(Options, Body),
    lmdb_sources_from_options(Options, LmdbSources0),
    % Also enable the LMDB gate when a D43 seek fact source selects the lmdb
    % backend (Stage-2 lazy+cached reader), not only the eager cpp_fact_sources.
    seek_sources_from_options(Options, SeekSources),
    (   member(seek_source(_, lmdb, _), SeekSources)
    ->  LmdbSources = [seek_lmdb|LmdbSources0]
    ;   LmdbSources = LmdbSources0
    ),
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

compile_wam_runtime_header_body_to_cpp(_Options, Body) :-
    cpp_render_template(runtime_header, [], Body).

compile_wam_runtime_to_cpp(_Options, Code) :-
    cpp_render_template(runtime_source, [], Code).

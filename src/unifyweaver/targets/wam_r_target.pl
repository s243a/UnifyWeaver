:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_r_target.pl - WAM-to-R Hybrid Transpilation Target (scaffold)
%
% Generates a hybrid WAM R project from a set of Prolog predicates.
% Modelled on the Scala/Haskell/Rust hybrid WAM targets:
%
%   Phase 1: WAM compilation (via wam_target:compile_predicate_to_wam/3)
%   Phase 2: WAM text  -> R Instruction literals (this file)
%   Phase 3: lowered native R per-predicate functions
%            (see wam_r_lowered_emitter.pl -- stub for now)
%
% Status: SCAFFOLD. The instruction-array path covers a useful core
% subset (head/body unification, choice points, builtin & foreign
% calls, switch_on_constant). The lowered emitter is a Phase-1 stub
% that always fails, so every predicate routes to the interpreter.
%
% Design notes (R-specific):
%   - Atoms are interned to integer IDs at codegen time. Well-known
%     atoms: true=0, fail=1, []=2, .=3, ""=4, [|]=5. Mirrors Scala/Haskell.
%   - Register names map to integer indices: A1->1, X3->103, Y2->202.
%   - Values are tagged lists:
%       list(tag="int",     val=N)
%       list(tag="float",   val=N)
%       list(tag="atom",    id=ID)
%       list(tag="unbound", name="V_n")
%       list(tag="struct",  fid=ID, args=list(...))
%     Performance work (R6/S4/external pointer) is deferred.
%   - WamState is an R environment (mutable, no copy on update).
%     Choice points snapshot only the fields that need restoring.

:- module(wam_r_target, [
    compile_wam_predicate_to_r/4,    % +Pred/Arity, +WamCode, +Options, -RCode
    write_wam_r_project/3,           % +Predicates, +Options, +ProjectDir
    r_foreign_predicate/3,           % +Pred, +Arity, +Options
    init_r_atom_intern_table/0,      % reinitialize atom intern table
    classify_r_fact_predicate/4,     % +PredIndicator, +WamLines, +Options, -Info
    r_fact_only/2,                   % +Segments, -Bool
    r_first_arg_groundness/3,        % +Segments, +Arity, -Status
    r_pick_layout/5,                 % +PredIndicator, +NClauses, +FactOnly, +Options, -Layout
    split_wam_into_segments_r/2,     % +Lines, -Segments
    % --- Re-exported helpers used by wam_r_lowered_emitter -----
    tokenize_wam_line/2,             % +Line, -Tokens
    wam_parts_to_r/3,                % +Tokens, +Options, -RLiteral
    parse_functor_arity/3,           % +Str, -Name, -Arity
    reg_to_int/2,                    % +RegName, -Index
    constant_to_r_term/2,            % +ConstStr, -RTermLiteral
    intern_r_atom/2,                 % +AtomStr, -Id
    wam_r_resolve_emit_mode/2        % +Options, -Mode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/template_system', [render_template/3]).
% Lowered emitter: real Phase-2 implementation lives there. We keep the
% module load lazy via catch/3 so the file remains usable even if the
% emitter module is temporarily missing during refactors.
:- use_module(wam_r_lowered_emitter, [
    wam_r_lowerable/3,
    lower_predicate_to_r/4
]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_config/2]).

% ============================================================================
% EMIT MODE
% ============================================================================
% Mirror of wam_haskell_target.pl: Mode is one of
%   - interpreter           : never lower (default, byte-identical to pre-
%                             Phase-2 output)
%   - functions             : try to lower every predicate; failed
%                             lowerability falls through to the array path
%   - mixed([Pred/Arity, ...]) : try to lower only listed preds
%
% Resolution order: emit_mode(M) Option -> user:wam_r_emit_mode(M)
% multifile fact -> default interpreter.

:- multifile user:wam_r_emit_mode/1.
:- multifile user:fact_layout/2.
:- multifile user:wam_r_layout_policy/5.

wam_r_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_emit_mode(M0, Mode)
    ;   catch(user:wam_r_emit_mode(M1), _, fail)
    ->  validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_emit_mode(interpreter, interpreter) :- !.
validate_emit_mode(functions,   functions)   :- !.
validate_emit_mode(mixed(L),    mixed(L))    :- is_list(L), !.
validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_r_emit_mode, Other),
                wam_r_resolve_emit_mode/2)).

%% should_try_lower(+Mode, +Pred, +Arity) is semidet.
should_try_lower(functions,    _, _) :- !.
should_try_lower(mixed(HotPreds), P, A) :-
    member(P/A, HotPreds), !.
should_try_lower(_, _, _) :- fail.

% ============================================================================
% ATOM INTERNING TABLE (compile-time)
% ============================================================================
% Mirrors wam_scala_target.pl / wam_haskell_target.pl. The runtime
% intern table is mutable and seeds itself from this list; ids are
% positional (index 0 -> id 0).

:- dynamic r_atom_intern_id/2.    % r_atom_intern_id(String, IntId)
:- dynamic r_atom_intern_next/1.  % r_atom_intern_next(NextId)

init_r_atom_intern_table :-
    retractall(r_atom_intern_id(_, _)),
    retractall(r_atom_intern_next(_)),
    assertz(r_atom_intern_id("true", 0)),
    assertz(r_atom_intern_id("fail", 1)),
    assertz(r_atom_intern_id("[]",   2)),
    assertz(r_atom_intern_id(".",    3)),
    assertz(r_atom_intern_id("",     4)),
    assertz(r_atom_intern_id("[|]",  5)),
    assertz(r_atom_intern_next(6)).

%% intern_r_atom(+AtomStr, -Id) is det.
intern_r_atom(AtomStr, Id) :-
    atom_string(AtomStr, Str),
    (   r_atom_intern_id(Str, Id0)
    ->  Id = Id0
    ;   retract(r_atom_intern_next(Next)),
        Id = Next,
        Next1 is Next + 1,
        assertz(r_atom_intern_id(Str, Id)),
        assertz(r_atom_intern_next(Next1))
    ).

%% emit_r_intern_table(-IdToStringEntries) is det.
%  Builds an R character-vector body of the intern table seed array.
%  Order matters -- index 0 -> id 0, etc.
emit_r_intern_table(IdToStringEntries) :-
    findall(Id-Str, r_atom_intern_id(Str, Id), Pairs),
    sort(Pairs, Sorted),
    maplist([_Id-Str, E]>>(
        r_string_literal(Str, Lit),
        format(string(E), '    ~w', [Lit])
    ), Sorted, Entries),
    atomic_list_concat(Entries, ',\n', IdToStringEntries).

% ============================================================================
% REGISTER ENCODING
% ============================================================================
% Mirrors wam_haskell_lowered_emitter.pl / wam_scala_target.pl.

%% reg_to_int(+RegName, -Int) is det.
%  A1 -> 1, X1 -> 101, Y1 -> 201.
reg_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Prefix),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Prefix == 'A' -> Int = Num
    ;   Prefix == 'X' -> Int is Num + 100
    ;   Prefix == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

% ============================================================================
% WAM LINE -> R INSTRUCTION LITERAL
% ============================================================================
% Each WAM assembly line maps to a call to one of the constructors in
% the runtime (GetConstant, PutVariable, Call, ...). Constructors return
% lightweight tagged lists; see runtime.R.mustache.

%% wam_line_to_r_literal(+Line, -RLiteral) is semidet.
wam_line_to_r_literal(Line, Lit) :-
    tokenize_wam_line(Line, Parts),
    Parts \= [],
    Parts = [First|_],
    \+ sub_string(First, _, 1, 0, ":"),
    wam_parts_to_r(Parts, [], Lit).

%% tokenize_wam_line(+Line, -Tokens)
%  Same single-quote-aware tokenizer as the Scala target.
tokenize_wam_line(Line, Tokens) :-
    string_chars(Line, Chars),
    tokenize_wam_chars(Chars, [], [], outside, Tokens).

tokenize_wam_chars([], [], Acc, _, Tokens) :- !,
    reverse(Acc, Tokens).
tokenize_wam_chars([], CurR, Acc, outside, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T0, CurC),
    strip_operand_comma(T0, T),
    % Bare comma at end-of-input: skip rather than emit "". Quoted-empty
    % atoms come through the inside-mode branch below and are kept.
    (   T == ""
    ->  reverse(Acc, Tokens)
    ;   reverse([T|Acc], Tokens)
    ).
tokenize_wam_chars([], CurR, Acc, inside, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T, CurC),
    reverse([T|Acc], Tokens).
tokenize_wam_chars([C|Rest], CurR, Acc, outside, Tokens) :-
    (   (C == ' ' ; C == '\t')
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, outside, Tokens)
        ;   reverse(CurR, CurC), string_chars(T0, CurC),
            strip_operand_comma(T0, T),
            (   T == ""
            ->  NewAcc = Acc
            ;   NewAcc = [T|Acc]
            ),
            tokenize_wam_chars(Rest, [], NewAcc, outside, Tokens)
        )
    ;   C == '\''
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, inside, Tokens)
        ;   tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
        )
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
    ).
tokenize_wam_chars([C|Rest], CurR, Acc, inside, Tokens) :-
    (   C == '\\',
        Rest = [Escaped|More]
    ->  tokenize_wam_chars(More, [Escaped|CurR], Acc, inside, Tokens)
    ;   C == '\''
    ->  reverse(CurR, CurC), string_chars(T, CurC),
        tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, inside, Tokens)
    ).

strip_operand_comma(Token0, Token) :-
    sub_string(Token0, _, 1, 0, ","),
    !,
    sub_string(Token0, 0, _, 1, Token).
strip_operand_comma(Token, Token).

% --- Control instructions ---
% Foreign call dispatch (overrides plain call/execute when foreign).
wam_parts_to_r(["call", PredArity], Options, Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    r_foreign_predicate(PredName, Arity, Options), !,
    format(string(Lit), 'CallForeign("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["call", Pred, ArityStr], Options, Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    r_foreign_predicate(PredName, Arity, Options), !,
    format(string(Lit), 'CallForeign("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["execute", PredArity], Options, Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    r_foreign_predicate(PredName, Arity, Options), !,
    format(string(Lit), 'ExecuteForeign("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["execute", Pred, ArityStr], Options, Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    r_foreign_predicate(PredName, Arity, Options), !,
    format(string(Lit), 'ExecuteForeign("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(Parts, _Options, Lit) :-
    wam_parts_to_r(Parts, Lit).

wam_parts_to_r(["call", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    format(string(Lit), 'Call("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    format(string(Lit), 'Call("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["execute", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    format(string(Lit), 'Execute("~w", ~w)', [PredName, Arity]).
wam_parts_to_r(["execute", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    format(string(Lit), 'Execute("~w", ~w)', [PredName, Arity]).

wam_parts_to_r(["proceed"], 'Proceed()').
wam_parts_to_r(["jump", Label], Lit) :-
    format(string(Lit), 'Jump("~w")', [Label]).

% --- Choice instructions ---
wam_parts_to_r(["try_me_else", Label], Lit) :-
    format(string(Lit), 'TryMeElse("~w")', [Label]).
% Soft variant for if-then-else's choice point (introduced by
% mark_ite_try_me_else). Pushes the same CP shape as TryMeElse but
% tagged kind="ite", so CutIte can find and truncate to it
% regardless of how deep Cond's evaluation buried the stack.
wam_parts_to_r(["try_me_else_ite", Label], Lit) :-
    format(string(Lit), 'TryMeElseIte("~w")', [Label]).
wam_parts_to_r(["retry_me_else", Label], Lit) :-
    format(string(Lit), 'RetryMeElse("~w")', [Label]).
wam_parts_to_r(["trust_me"], 'TrustMe()').

% --- Environment ---
wam_parts_to_r(["allocate"],   'Allocate()').
wam_parts_to_r(["deallocate"], 'Deallocate()').

% --- Register: get ---
wam_parts_to_r(["get_constant", C, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    constant_to_r_term(C, TermLit),
    format(string(Lit), 'GetConstant(~w, ~w)', [TermLit, RegIdx]).
wam_parts_to_r(["get_variable", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'GetVariable(~w, ~w)', [VIdx, AIdx]).
wam_parts_to_r(["get_value", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'GetValue(~w, ~w)', [VIdx, AIdx]).

% --- Register: put ---
wam_parts_to_r(["put_constant", C, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    constant_to_r_term(C, TermLit),
    format(string(Lit), 'PutConstant(~w, ~w)', [TermLit, RegIdx]).
wam_parts_to_r(["put_variable", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'PutVariable(~w, ~w)', [VIdx, AIdx]).
wam_parts_to_r(["put_value", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'PutValue(~w, ~w)', [VIdx, AIdx]).

% --- Structure / list ---
wam_parts_to_r(["put_structure", Functor, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    parse_functor_arity(Functor, FName, FArity),
    intern_r_atom(FName, FId),
    format(string(Lit), 'PutStructure(~w, ~w, ~w)', [FId, RegIdx, FArity]).
wam_parts_to_r(["put_list", Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    intern_r_atom("[|]", FId),
    format(string(Lit), 'PutList(~w, ~w)', [RegIdx, FId]).
wam_parts_to_r(["get_structure", Functor, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    parse_functor_arity(Functor, FName, FArity),
    intern_r_atom(FName, FId),
    format(string(Lit), 'GetStructure(~w, ~w, ~w)', [FId, RegIdx, FArity]).
wam_parts_to_r(["get_list", Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    intern_r_atom("[|]", FId),
    format(string(Lit), 'GetList(~w, ~w)', [RegIdx, FId]).
wam_parts_to_r(["set_variable", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'SetVariable(~w)', [Idx]).
wam_parts_to_r(["set_value", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'SetValue(~w)', [Idx]).
wam_parts_to_r(["set_constant", C], Lit) :-
    constant_to_r_term(C, TermLit),
    format(string(Lit), 'SetConstant(~w)', [TermLit]).
wam_parts_to_r(["unify_variable", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'UnifyVariable(~w)', [Idx]).
wam_parts_to_r(["unify_value", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'UnifyValue(~w)', [Idx]).
wam_parts_to_r(["unify_constant", C], Lit) :-
    constant_to_r_term(C, TermLit),
    format(string(Lit), 'UnifyConstant(~w)', [TermLit]).

% --- Builtins ---
wam_parts_to_r(["builtin_call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    r_string_literal(Pred, PredLit),
    format(string(Lit), 'BuiltinCall(~w, ~w)', [PredLit, Arity]).

% --- Foreign call (explicit) ---
wam_parts_to_r(["call_foreign", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    format(string(Lit), 'CallForeign("~w", ~w)', [Pred, Arity]).

% --- arg N, Reg, OutReg ---
% The WAM compiler optimises arg/3 into a dedicated opcode (faster
% than builtin_call when N is a literal and the source term is already
% in a register). Format: "arg <N> <Reg> <OutReg>".
wam_parts_to_r(["arg", NStr, RegStr, OutRegStr], Lit) :-
    number_string(N, NStr),
    reg_to_int(RegStr, RegIdx),
    reg_to_int(OutRegStr, OutIdx),
    format(string(Lit), 'ArgInstr(~w, ~w, ~w)', [N, RegIdx, OutIdx]).

% --- Switch on constant ---
wam_parts_to_r(["switch_on_constant" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, NormalizedCases),
    parse_switch_cases(NormalizedCases, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'SwitchOnConstant(list(~w))', [CasesStr]).

% --- Switch on term (type-mixed dispatch) ---
% Emitted by the WAM compiler when a predicate's clauses have a mix
% of first-arg types (constant + struct + list). The runtime
% implements this as a no-op that advances PC; the standard
% try/retry chain visits each clause and clause-head unification
% filters non-matching ones, so correctness is preserved without
% the optimisation.
wam_parts_to_r(["switch_on_term" | _], 'SwitchOnTerm()').

% --- Switch on structure (functor-shape dispatch) ---
% Each case is "F/N:label". Behaviour parallels switch_on_constant: when
% the first arg is a struct of matching shape, jump to the labelled
% clause; otherwise (or if the label is "default") fall through to the
% try/retry chain. The runtime treats the case list as advisory — we
% always re-run the matching clause body afterwards, so worst case is
% an extra try/retry traversal.
wam_parts_to_r(["switch_on_structure" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, NormalizedCases),
    parse_struct_switch_cases(NormalizedCases, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'SwitchOnStructure(list(~w))', [CasesStr]).

% --- ITE soft cut ---
wam_parts_to_r(["cut_ite"], 'CutIte()').

% --- Aggregation (findall/3 etc.) ---
wam_parts_to_r(["begin_aggregate", Kind, TemplateReg, BagReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    reg_to_int(BagReg, BIdx),
    format(string(Lit), 'BeginAggregate("~w", ~w, ~w)', [Kind, TIdx, BIdx]).
wam_parts_to_r(["end_aggregate", TemplateReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    format(string(Lit), 'EndAggregate(~w)', [TIdx]).

% --- Fallback: emit raw text the runtime can pretty-print on dispatch ---
wam_parts_to_r(Parts, Lit) :-
    atomic_list_concat(Parts, ' ', Text),
    r_string_literal(Text, TextLit),
    format(string(Lit), 'Raw(~w)', [TextLit]).

%% parse_switch_cases(+Tokens, -CaseLiterals)
parse_switch_cases([], []).
parse_switch_cases([Token | Rest], [Lit | More]) :-
    split_at_first_colon(Token, ValStr, LabelStr),
    intern_r_atom(ValStr, AtomId),
    format(string(Lit), 'SwitchCase(Atom(~w), "~w")', [AtomId, LabelStr]),
    parse_switch_cases(Rest, More).

%% parse_struct_switch_cases(+Tokens, -CaseLiterals)
%  Each token is "Functor/Arity:label" — e.g. "f/2:L_pair_1_2".
parse_struct_switch_cases([], []).
parse_struct_switch_cases([Token | Rest], [Lit | More]) :-
    split_at_first_colon(Token, FAStr, LabelStr),
    parse_functor_arity(FAStr, FName, FArity),
    intern_r_atom(FName, FId),
    format(string(Lit), 'StructCase(~w, ~w, "~w")', [FId, FArity, LabelStr]),
    parse_struct_switch_cases(Rest, More).

normalize_switch_case_tokens([], []).
normalize_switch_case_tokens([Value, Label0 | Rest], [Token | More]) :-
    \+ sub_string(Value, _, 1, _, ":"),
    sub_string(Label0, 0, 1, _, ":"),
    !,
    sub_string(Label0, 1, _, 0, Label),
    string_concat(Value, ":", Prefix),
    string_concat(Prefix, Label, Token),
    normalize_switch_case_tokens(Rest, More).
normalize_switch_case_tokens([Token | Rest], [Token | More]) :-
    normalize_switch_case_tokens(Rest, More).

strip_arity_suffix(Pred, Name) :-
    (   sub_string(Pred, B, 1, _, "/")
    ->  sub_string(Pred, 0, B, _, Name)
    ;   Name = Pred
    ).

%% constant_to_r_term(+ConstStr, -RTermLit)
%  Numbers become Integer(N) / FloatTerm(N); everything else interns as Atom.
constant_to_r_term(C, Lit) :-
    (   number_string(N, C),
        integer(N)
    ->  format(string(Lit), 'IntTerm(~w)', [N])
    ;   number_string(F, C),
        float(F)
    ->  format(string(Lit), 'FloatTerm(~w)', [F])
    ;   intern_r_atom(C, AtomId),
        format(string(Lit), 'Atom(~w)', [AtomId])
    ).

%% r_string_literal(+Raw, -Quoted)
%  Wraps Raw in double quotes and escapes per R rules.
r_string_literal(Raw, Quoted) :-
    atom_string(Raw, S),
    string_chars(S, Chars),
    maplist(r_string_escape_char, Chars, EscapedLists),
    append(EscapedLists, EscChars),
    string_chars(EscBody, EscChars),
    format(string(Quoted), '"~w"', [EscBody]).

r_string_escape_char('\\', ['\\', '\\']) :- !.
r_string_escape_char('"',  ['\\', '"'])  :- !.
r_string_escape_char('\n', ['\\', 'n'])  :- !.
r_string_escape_char('\t', ['\\', 't'])  :- !.
r_string_escape_char(C,    [C]).

%% parse_functor_arity(+FunctorStr, -Name, -Arity)
parse_functor_arity(FStr, Name, Arity) :-
    atom_string(FA, FStr),
    (   last_slash_index(FA, B)
    ->  sub_atom(FA, 0, B, _, Name),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ;   Name = FA, Arity = 0
    ).

last_slash_index(Atom, Index) :-
    findall(B, sub_atom(Atom, B, 1, _, '/'), Bs),
    Bs \= [],
    last(Bs, Index).

split_at_first_colon(Token, Before, After) :-
    sub_string(Token, B, 1, _, ":"),
    !,
    sub_string(Token, 0, B, _, Before),
    B1 is B + 1,
    sub_string(Token, B1, _, 0, After).

% ============================================================================
% WAM TEXT -> R INSTRUCTION ARRAY
% ============================================================================

%% wam_code_to_r_data(+WamCode, +Options, -Instructions, -LabelMap, -LabelEntries) is det.
%  Returns:
%    Instructions: list of R Instruction-constructor literals
%    LabelMap:     list of "label" - PC pairs (R uses 1-based indexing)
%    LabelEntries: formatted "<label>" = N pair lines
wam_code_to_r_data(WamCode, Options, Instructions, LabelMap, LabelEntries) :-
    atom_string(WamCode, Str),
    split_string(Str, "\n", "", Lines0),
    mark_ite_try_me_else(Lines0, Lines),
    wam_lines_to_data(Lines, Options, 1, Instructions, LabelMap, LabelEntries).

%% mark_ite_try_me_else(+Lines, -Lines)
%  Rewrites every `try_me_else <label>` whose NEXT branch marker
%  (skipping Allocate / Get* / Put* / Call / etc.) is `cut_ite` so
%  it emits as `try_me_else_ite <label>` instead. The runtime then
%  pushes a CP marked `kind="ite"` for if-then-else's choice point,
%  and CutIte truncates state$cps back to that CP's pre-push depth
%  rather than dropping a stale topmost CP. Regular clause try-
%  chains (`try_me_else` followed by `retry_me_else` or `trust_me`)
%  are left untouched. Nested if-then-else works because each level's
%  own `try_me_else` finds its own `cut_ite` first.
mark_ite_try_me_else([], []).
mark_ite_try_me_else([Line|Rest], [Line2|Rest2]) :-
    (   tokenize_wam_line(Line, ["try_me_else", LabelStr]),
        next_branch_marker(Rest, "cut_ite")
    ->  format(string(Line2),
               "    try_me_else_ite ~w", [LabelStr])
    ;   Line2 = Line
    ),
    mark_ite_try_me_else(Rest, Rest2).

%% next_branch_marker(+Lines, -Marker)
%  Walks forward through Lines, skipping non-branch instructions and
%  blank lines. Succeeds with the first branch marker found; fails
%  if none are found before the lines run out.
next_branch_marker([], _) :- fail.
next_branch_marker([Line|Rest], Marker) :-
    tokenize_wam_line(Line, Parts),
    (   Parts = [First|_],
        memberchk(First, ["try_me_else", "retry_me_else",
                           "trust_me", "cut_ite", "try_me_else_ite"])
    ->  Marker = First
    ;   next_branch_marker(Rest, Marker)
    ).

wam_lines_to_data([], _, _, [], [], []).
wam_lines_to_data([Line|Rest], Options, PC, Instructions, LabelMap, LabelEntries) :-
    tokenize_wam_line(Line, Parts),
    (   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        format(string(LEntry), '    "~w" = ~wL', [LabelName, PC]),
        LabelMap     = [LabelName-PC | LM2],
        LabelEntries = [LEntry      | LE2],
        wam_lines_to_data(Rest, Options, PC, Instructions, LM2, LE2)
    ;   Parts = []
    ->  wam_lines_to_data(Rest, Options, PC, Instructions, LabelMap, LabelEntries)
    ;   wam_parts_to_r(Parts, Options, Lit),
        PC1 is PC + 1,
        Instructions = [Lit | Instrs2],
        wam_lines_to_data(Rest, Options, PC1, Instrs2, LabelMap, LabelEntries)
    ).

% ============================================================================
% FACT SHAPE CLASSIFICATION
% ============================================================================

%% classify_r_fact_predicate(+PredIndicator, +WamLines, +Options, -Info) is det.
%  Classifies each predicate as fact-only or rule-bearing and selects a
%  layout strategy. This mirrors the Haskell F1 classifier; R currently
%  emits all layouts through the compiled WAM path, so this is metadata
%  and testable policy plumbing rather than a runtime behavior change.
classify_r_fact_predicate(PredIndicator, WamLines, Options, Info) :-
    split_wam_into_segments_r(WamLines, Segments),
    length(Segments, NClauses),
    r_fact_only(Segments, FactOnly),
    (PredIndicator = _:_/Arity -> true ; PredIndicator = _/Arity),
    r_first_arg_groundness(Segments, Arity, FirstArg),
    r_pick_layout(PredIndicator, NClauses, FactOnly, Options, Layout),
    Info = fact_shape_info(NClauses, FactOnly, FirstArg, Layout).

%% split_wam_into_segments_r(+Lines, -Segments) is det.
%  Groups WAM text lines into Label-InstrList pairs.
split_wam_into_segments_r([], []).
split_wam_into_segments_r([Line|Rest], Segments) :-
    tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  split_wam_into_segments_r(Rest, Segments)
    ;   Parts = [First|_],
        sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        extract_r_segment_instrs(Rest, Instrs, Remaining),
        Segments = [LabelName-Instrs | RestSegs],
        split_wam_into_segments_r(Remaining, RestSegs)
    ;   split_wam_into_segments_r(Rest, Segments)
    ).

extract_r_segment_instrs([], [], []).
extract_r_segment_instrs([Line|Rest], Instrs, Remaining) :-
    tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  extract_r_segment_instrs(Rest, Instrs, Remaining)
    ;   Parts = [First|_],
        sub_string(First, _, 1, 0, ":")
    ->  Instrs = [],
        Remaining = [Line|Rest]
    ;   (   parse_wam_instr_r(Parts, Instr)
        ->  Instrs = [Instr | RestInstrs],
            extract_r_segment_instrs(Rest, RestInstrs, Remaining)
        ;   extract_r_segment_instrs(Rest, Instrs, Remaining)
        )
    ).

parse_wam_instr_r(["get_constant", C, Ai], get_constant(C, Ai)).
parse_wam_instr_r(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
parse_wam_instr_r(["get_value", Xn, Ai], get_value(Xn, Ai)).
parse_wam_instr_r(["get_structure", F, Ai], get_structure(F, Ai)).
parse_wam_instr_r(["put_constant", C, Ai], put_constant(C, Ai)).
parse_wam_instr_r(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
parse_wam_instr_r(["put_value", Xn, Ai], put_value(Xn, Ai)).
parse_wam_instr_r(["put_structure", F|_], put_structure(F)).
parse_wam_instr_r(["put_list", Ai], put_list(Ai)).
parse_wam_instr_r(["unify_variable", Xn], unify_variable(Xn)).
parse_wam_instr_r(["unify_value", Xn], unify_value(Xn)).
parse_wam_instr_r(["unify_constant", C], unify_constant(C)).
parse_wam_instr_r(["call", P, N], call(P, N)).
parse_wam_instr_r(["execute", P], execute(P)).
parse_wam_instr_r(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
parse_wam_instr_r(["proceed"], proceed).
parse_wam_instr_r(["try_me_else", L], try_me_else(L)).
parse_wam_instr_r(["retry_me_else", L], retry_me_else(L)).
parse_wam_instr_r(["trust_me"], trust_me).
parse_wam_instr_r(["allocate"], allocate).
parse_wam_instr_r(["deallocate"], deallocate).
parse_wam_instr_r(["switch_on_constant"|_], switch_on_constant).

%% r_fact_only(+Segments, -Bool) is det.
%  true iff no clause has a body-level call instruction.
r_fact_only(Segments, true) :-
    forall(member(_-Instrs, Segments),
           forall(member(Instr, Instrs),
                  \+ r_is_body_call(Instr))), !.
r_fact_only(_, false).

r_is_body_call(call(_, _)).
r_is_body_call(execute(_)).
r_is_body_call(builtin_call(_, _)).

%% r_first_arg_groundness(+Segments, +Arity, -Status) is det.
%  Status is one of: none, all_ground, all_variable, mixed.
r_first_arg_groundness(_Segments, 0, none) :- !.
r_first_arg_groundness(Segments, Arity, Status) :-
    Arity > 0,
    maplist(r_clause_arg1_type, Segments, Types),
    r_combine_groundness(Types, Status).

r_clause_arg1_type(_-Instrs, Type) :-
    (   member(get_constant(_, "A1"), Instrs) -> Type = ground
    ;   member(get_structure(_, "A1"), Instrs) -> Type = ground
    ;   member(get_variable(_, "A1"), Instrs) -> Type = variable
    ;   member(get_value(_, "A1"), Instrs) -> Type = variable
    ;   Type = unknown
    ).

r_combine_groundness(Types, all_ground) :-
    forall(member(T, Types), T == ground), !.
r_combine_groundness(Types, all_variable) :-
    forall(member(T, Types), T == variable), !.
r_combine_groundness(_, mixed).

%% r_pick_layout(+PredIndicator, +NClauses, +FactOnly, +Options, -Layout) is det.
%  User overrides win. Built-in policies mirror the Haskell target.
r_pick_layout(PredIndicator, _NClauses, _FactOnly, Options, Layout) :-
    option(fact_layout(PredIndicator, UserLayout), Options), !,
    Layout = UserLayout.
r_pick_layout(PredIndicator, _NClauses, _FactOnly, _Options, Layout) :-
    catch(user:fact_layout(PredIndicator, UserLayout), _, fail), !,
    Layout = UserLayout.
r_pick_layout(PredIndicator, NClauses, FactOnly, Options, Layout) :-
    option(fact_layout_policy(PolicyName), Options, auto),
    r_layout_policy(PolicyName, PredIndicator, NClauses, FactOnly, Options, Layout).

r_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout) :-
    catch(user:wam_r_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Layout0),
          _, fail),
    !,
    (   nonvar(Layout0)
    ->  Layout = Layout0
    ;   r_builtin_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout)
    ).
r_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout) :-
    r_builtin_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout).

r_builtin_layout_policy(auto, _Pred, NClauses, FactOnly, Options, Layout) :-
    option(fact_count_threshold(Threshold), Options, 100),
    (   FactOnly == true,
        NClauses > Threshold
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).
r_builtin_layout_policy(compiled_only, _, _, _, _, compiled).
r_builtin_layout_policy(inline_eager, _, _, FactOnly, _, Layout) :-
    (   FactOnly == true
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).
r_builtin_layout_policy(cost_aware, PredIndicator, NClauses, FactOnly, Options, Layout) :-
    (PredIndicator = _:_/Arity -> true ; PredIndicator = _/Arity),
    option(fact_cost_threshold(Threshold), Options, 200),
    Mult is max(1, Arity),
    CostScore is NClauses * Mult,
    (   FactOnly == true,
        CostScore > Threshold
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).

format_r_fact_shape_comment(PredIndicator, fact_shape_info(N, FO, FA, Layout), Comment) :-
    (PredIndicator = _:P/A -> true ; PredIndicator = P/A),
    format(string(Comment),
           '# ~w/~w: fact_only=~w, clauses=~w, first_arg=~w, layout=~w',
           [P, A, FO, N, FA, Layout]).

% ============================================================================
% PREDICATE COMPILATION
% ============================================================================

%% compile_wam_predicate_to_r(+PredIndicator, +WamCode, +Options, -RCode)
%  Provided for symmetry with the other targets. The hybrid pipeline
%  uses write_wam_r_project/3, not this.
compile_wam_predicate_to_r(_Pred, _WamCode, _Options, "").

%% compile_predicates_for_project(+Predicates, +Options,
%%                                -AllInstrs, -TopLabels,
%%                                -AllLabels, -WrapperCode, -LoweredCode,
%%                                -FactShapeComments, -LoweredDispatchCode)
compile_predicates_for_project(Predicates, Options,
                               AllInstrs, TopLevelLabelEntries,
                               AllLabelEntries, WrapperCode, LoweredCode,
                               FactShapeComments,
                               LoweredDispatchCode) :-
    init_r_atom_intern_table,
    option(intern_atoms(ExtraAtoms), Options, []),
    forall(member(A, ExtraAtoms),
           (atom_string(A, S), intern_r_atom(S, _))),
    option(foreign_predicates(ForeignPredicates), Options, []),
    append_missing_foreign_predicates(Predicates, ForeignPredicates,
                                      CompilePredicates),
    wam_r_resolve_emit_mode(Options, Mode),
    compile_all_predicates(CompilePredicates, Options, Mode, 1,
                           [], [], [], [], [], [], [],
                           AllInstrs, TopLevelLabelEntries,
                           AllLabelEntries, Wrappers, LoweredEntries,
                           FactComments, LoweredDispatchEntries),
    atomic_list_concat(Wrappers, '\n', WrapperCode),
    atomic_list_concat(LoweredEntries, '\n', LoweredCode),
    atomic_list_concat(FactComments, '\n', FactShapeComments),
    atomic_list_concat(LoweredDispatchEntries, '\n', LoweredDispatchCode).

append_missing_foreign_predicates(Predicates, ForeignPredicates,
                                  CompilePredicates) :-
    findall(Foreign,
            (   member(Foreign, ForeignPredicates),
                \+ ( member(Pred, Predicates),
                     same_predicate_indicator(Pred, Foreign)
                   )
            ),
            Missing),
    append(Predicates, Missing, CompilePredicates).

same_predicate_indicator(P0, P1) :-
    predicate_indicator_key(P0, K),
    predicate_indicator_key(P1, K).

predicate_indicator_key(_:Pred/Arity, Pred/Arity) :- !.
predicate_indicator_key(Pred/Arity, Pred/Arity).

compile_all_predicates([], _, _, _, Instrs, TopLabels, AllLabels, Wrappers,
                       Lowered, FactComments, Dispatch,
                       Instrs, TopLabels, AllLabels, Wrappers,
                       Lowered, FactComments, Dispatch).
compile_all_predicates([Pred|Rest], Options, Mode, BasePC,
                       InstrAcc, TopLabelAcc, AllLabelAcc, WrapperAcc, LoweredAcc,
                       FactCommentAcc, LoweredDispAcc,
                       AllInstrs, TopLevelLabelEntries,
                       AllLabelEntries, AllWrappers, AllLowered,
                       AllFactComments, AllLoweredDispatch) :-
    (   Pred = _Module:P/Arity -> true ; Pred = P/Arity ),
    (   r_fact_source_spec(P, Arity, Options, _FactSourceSpec)
    ->  % External fact source -- skip Prolog clause extraction.
        % The body is a single Execute("P", A) which falls through
        % to lowered_dispatch (set up below), and the lowered fn
        % loads the file at program-init time and dispatches via
        % WamRuntime$fact_table_dispatch.
        format(string(EFLit), 'Execute("~w", ~w)', [P, Arity]),
        ExternalSeq = [EFLit],
        append(InstrAcc, ExternalSeq, NewInstrs),
        NewPC is BasePC + 1,
        format(string(MainEntry), '    "~w/~w" = ~wL', [P, Arity, BasePC]),
        NewTopLabels = [MainEntry | TopLabelAcc],
        NewAllLabels = [MainEntry | AllLabelAcc],
        WamCodeForLower = "",
        % External fact sources have no clause body to classify, so
        % no fact-shape comment is emitted. Without this binding the
        % accumulator carries an unbound tail, blowing up the
        % atomic_list_concat in compile_predicates_for_project.
        NewFactCommentAcc = FactCommentAcc
    ;   r_foreign_predicate(P, Arity, Options)
    ->  format(string(FLit), 'CallForeign("~w", ~w)', [P, Arity]),
        ForeignSeq = [FLit, 'Proceed()'],
        append(InstrAcc, ForeignSeq, NewInstrs),
        NewPC is BasePC + 2,
        format(string(MainEntry), '    "~w/~w" = ~wL', [P, Arity, BasePC]),
        NewTopLabels = [MainEntry | TopLabelAcc],
        NewAllLabels = [MainEntry | AllLabelAcc],
        WamCodeForLower = "",
        NewFactCommentAcc = FactCommentAcc
    ;   % Pass Pred (which may be Module:P/Arity or P/Arity) so cross-
        % module compilation can find clauses outside `user`. Stripping
        % to P/Arity defaults compile_predicate_to_wam's module to
        % `user`, which silently produces "no clauses" for predicates
        % defined in any other module.
        compile_predicate_to_wam(Pred, [], WamCode),
        WamCodeForLower = WamCode,
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", WamLines),
        classify_r_fact_predicate(Pred, WamLines, Options, FactInfo),
        format_r_fact_shape_comment(Pred, FactInfo, FactComment),
        NewFactCommentAcc = [FactComment | FactCommentAcc],
        wam_code_to_r_data(WamCode, Options, PredInstrs, _LMap,
                           PredSubLabelEntries0),
        length(PredInstrs, PredLen),
        NewPC is BasePC + PredLen,
        Offset is BasePC - 1,
        maplist(offset_label_entry(Offset), PredSubLabelEntries0,
                PredSubLabelEntries1),
        format(string(MainKey), '~w/~w', [P, Arity]),
        exclude(is_pred_label(MainKey), PredSubLabelEntries1,
                PredSubLabelEntries),
        format(string(MainEntry), '    "~w/~w" = ~wL', [P, Arity, BasePC]),
        append(InstrAcc, PredInstrs, NewInstrs),
        NewTopLabels = [MainEntry | TopLabelAcc],
        append([MainEntry | PredSubLabelEntries], AllLabelAcc, NewAllLabels)
    ),
    % Decide whether this predicate should be lowered. External
    % fact sources (r_fact_sources option) win first -- they have
    % no Prolog clauses, just a file-loader; then the kernel
    % detector (recognised graph pattern -> native R fast path);
    % then the fact-table path; then the regular Phase-3 lowered
    % emitter; otherwise the WAM array.
    (   r_fact_source_spec(P, Arity, Options, ExternalFactSpec)
    ->  emit_external_fact_source(P, Arity, ExternalFactSpec,
                                   ExtData, ExtFunc, ExtFuncName),
        NewLoweredAcc = [ExtData, ExtFunc | LoweredAcc],
        emit_r_lowered_wrapper(P, Arity, ExtFuncName, WrapperCode),
        emit_lowered_dispatch_entry(P, Arity, ExtFuncName, DispEntry),
        NewLoweredDispAcc = [DispEntry | LoweredDispAcc]
    ;   kernel_layout_enabled(Options),
        catch(wam_r_kernel_detect(Pred, Kernel), _, fail),
        catch(emit_kernel(P, Kernel, KData, KFunc, KFuncName), _, fail)
    ->  (   KData == ""
        ->  NewLoweredAcc = [KFunc | LoweredAcc]
        ;   NewLoweredAcc = [KData, KFunc | LoweredAcc]
        ),
        emit_r_lowered_wrapper(P, Arity, KFuncName, WrapperCode),
        emit_lowered_dispatch_entry(P, Arity, KFuncName, DispEntry),
        NewLoweredDispAcc = [DispEntry | LoweredDispAcc]
    ;   WamCodeForLower \= "",
        catch(wam_r_fact_classify(WamCodeForLower,
                                   fact_info(NCls, _FArity, FTuples)),
              _, fail),
        fact_layout_enabled(P, Arity, NCls, Options)
    ->  emit_fact_table(P, Arity, FTuples, FactData, FactFunc, FactFuncName,
                        RangeReg),
        NewLoweredAcc = [FactData, FactFunc | LoweredAcc],
        emit_r_lowered_wrapper(P, Arity, FactFuncName, WrapperCode),
        emit_lowered_dispatch_entry(P, Arity, FactFuncName, DispEntry),
        % Two entries per fact-tabled pred: the lowered-dispatch
        % registration (consumed by dispatch_call's fast path) and
        % the fact_range_indexes registration (consumed by the
        % fact_in_range/5 builtin). Both go into the same
        % lowered-dispatch-assignments block of the program template
        % so they run after the lowered functions are defined.
        NewLoweredDispAcc = [RangeReg, DispEntry | LoweredDispAcc]
    ;   should_try_lower(Mode, P, Arity),
        WamCodeForLower \= "",
        catch(wam_r_lowerable(Pred, WamCodeForLower, _Reason), _, fail),
        catch(lower_predicate_to_r(Pred, WamCodeForLower, Options,
                                   lowered(_PName, FuncName, LoweredR)),
              _, fail)
    ->  NewLoweredAcc = [LoweredR | LoweredAcc],
        emit_r_lowered_wrapper(P, Arity, FuncName, WrapperCode),
        % Phase-3 lowered functions are NOT registered for internal
        % dispatch -- they expect their own pre-set state via the
        % per-pred wrapper. Internal calls keep going through the
        % WAM array, matching the pre-PR behaviour.
        NewLoweredDispAcc = LoweredDispAcc
    ;   NewLoweredAcc = LoweredAcc,
        NewLoweredDispAcc = LoweredDispAcc,
        emit_r_wrapper(P, Arity, BasePC, WrapperCode)
    ),
    compile_all_predicates(Rest, Options, Mode, NewPC,
                           NewInstrs, NewTopLabels, NewAllLabels,
                           [WrapperCode|WrapperAcc], NewLoweredAcc,
                           NewFactCommentAcc,
                           NewLoweredDispAcc,
                           AllInstrs, TopLevelLabelEntries,
                           AllLabelEntries, AllWrappers, AllLowered,
                           AllFactComments, AllLoweredDispatch).

%% emit_lowered_dispatch_entry(+Pred, +Arity, +FuncName, -Entry)
%  Generates an `assign("Pred/Arity", FuncName, envir = ...)` line
%  that's stitched into the program template. The runtime tier
%  dispatch_call consults program$lowered_dispatch first, so kernel
%  / fact-table fast paths fire for both top-level R-API calls and
%  internal Call/Execute instructions.
emit_lowered_dispatch_entry(Pred, Arity, FuncName, Entry) :-
    format(string(Entry),
           'assign("~w/~w", ~w, envir = shared_program$lowered_dispatch)',
           [Pred, Arity, FuncName]).

offset_label_entry(Offset, Entry0, Entry) :-
    atom_string(Entry0, S),
    (   sub_string(S, B, 3, _, " = ")
    ->  B1 is B + 3,
        sub_string(S, 0, B, _, LabelPart),
        sub_string(S, B1, _, 0, PCStr0),
        % strip trailing 'L' (R integer suffix)
        (   sub_string(PCStr0, _, 1, 0, "L")
        ->  sub_string(PCStr0, 0, _, 1, PCStr)
        ;   PCStr = PCStr0
        ),
        number_string(PC0, PCStr),
        PC is PC0 + Offset,
        format(string(Entry), '~w = ~wL', [LabelPart, PC])
    ;   Entry = Entry0
    ).

is_pred_label(PredKey, Entry) :-
    atom_string(Entry, S),
    sub_string(S, _, _, _, PredKey).

%% emit_r_wrapper(+Pred, +Arity, +StartPc, -Code)
%  Emits an R top-level function: pred_name(arg1, arg2, ...).
%  Calls run_predicate(shared_program, start_pc, list(arg1, ...)).
emit_r_wrapper(Pred, Arity, StartPc, Code) :-
    pred_arg_strings(Arity, ArgDeclStr, ArgListStr),
    r_pred_name(Pred, RName),
    format(string(Code),
           '~w <- function(~w) {\n  WamRuntime$run_predicate(shared_program, ~wL, ~w)\n}\n',
           [RName, ArgDeclStr, StartPc, ArgListStr]).

%% emit_r_lowered_wrapper(+Pred, +Arity, +LoweredFuncName, -Code)
%  Wrapper for predicates that have a lowered R function. Builds a
%  fresh state, seeds A1..An with the call-site args, and dispatches
%  directly to the lowered function (bypassing the instruction-array
%  driver). Returns the boolean returned by the lowered fn.
emit_r_lowered_wrapper(Pred, Arity, LoweredFuncName, Code) :-
    pred_arg_strings(Arity, ArgDeclStr, ArgListStr),
    r_pred_name(Pred, RName),
    format(string(Code),
'~w <- function(~w) {
  state <- WamRuntime$new_state()
  WamRuntime$promote_regs(state)
  args <- ~w
  for (i in seq_along(args)) WamRuntime$put_reg(state, i, args[[i]])
  state$cp <- 0L
  isTRUE(~w(shared_program, state))
}
', [RName, ArgDeclStr, ArgListStr, LoweredFuncName]).

pred_arg_strings(0, '', 'list()') :- !.
pred_arg_strings(Arity, ArgDeclStr, ArgListStr) :-
    Arity > 0,
    numlist(1, Arity, ArgNums),
    maplist([N, A]>>(format(string(A), 'a~w', [N])), ArgNums, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgDeclStr),
    format(string(ArgListStr), 'list(~w)', [ArgDeclStr]).

%% r_pred_name(+PrologName, -RName)
%  Generates the R identifier for a per-predicate wrapper. We
%  always prefix with `pred_` so the wrapper can never collide
%  with a base R function (e.g. `c`, `t`, `q`, `cat`, `paste`,
%  `tryCatch`); without the prefix, asserting a Prolog predicate
%  literally named `c/2` would shadow `base::c` and crash the
%  runtime's own use of `c(...)` for vector construction.
r_pred_name(Pred, RName) :-
    atom_string(Pred, PStr),
    string_chars(PStr, Chars),
    maplist(r_safe_ident_char, Chars, SafeChars),
    string_chars(SafeStr, SafeChars),
    string_concat("pred_", SafeStr, RName0),
    atom_string(RName, RName0).

r_safe_ident_char(C, C) :-
    char_type(C, alnum), !.
r_safe_ident_char('.', '.') :- !.
r_safe_ident_char('_', '_') :- !.
r_safe_ident_char(_, '_').

% ============================================================================
% RECURSIVE KERNEL DETECTION + EMISSION
% ============================================================================
%
% Detects predicates that match a registered kernel pattern (so far:
% transitive_closure2 -- ancestor(X,Y) :- edge(X,Y). ancestor(X,Y) :-
% edge(X,Z), ancestor(Z,Y).) and emits a native R BFS that runs the
% reachability set in one pass, streaming hits via iter-CP. The
% detector lives in src/unifyweaver/core/recursive_kernel_detection.pl
% and is shared with the Haskell/Rust targets.
%
% Per-call: kernel_layout(off) in Options or
% user:wam_r_kernel_layout(off) globally disables detection.

:- multifile user:wam_r_kernel_layout/1.

kernel_layout_enabled(Options) :-
    option(kernel_layout(Setting), Options, auto),
    kernel_layout_setting_ok(Setting).

kernel_layout_setting_ok(auto) :-
    (   catch(user:wam_r_kernel_layout(Override), _, fail)
    ->  Override \== off
    ;   true
    ).
kernel_layout_setting_ok(on).

%% wam_r_kernel_detect(+PredIndicator, -Kernel) is semidet.
%  Pulls the user clauses for PredIndicator and asks the shared
%  detector to classify them. Fails silently when the predicate
%  doesn't match any registered kernel pattern.
wam_r_kernel_detect(PI, Kernel) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(Head, Pred, Arity),
    catch(findall(Head-StrippedBody,
                  ( user:clause(Head, RawBody),
                    strip_module_qualifiers(RawBody, StrippedBody) ),
                  Clauses), _, fail),
    Clauses \= [],
    detect_recursive_kernel(Pred, Arity, Clauses, Kernel).

% Recursively peel `Mod:Goal` wrappers and walk through the goal
% structure (conjunction, disjunction, if-then) so the recursive-
% kernel detector sees the underlying predicate calls. This is
% needed because plunit wraps test-asserted clause bodies in its
% own `plunit_<test>:user:Goal` -- bare `clause/2` returns the
% wrapped form, which `detect_recursive_kernel` doesn't recognise.
strip_module_qualifiers(Var, Var) :-
    var(Var), !.
strip_module_qualifiers(_:G, Stripped) :-
    !, strip_module_qualifiers(G, Stripped).
strip_module_qualifiers((A, B), (As, Bs)) :-
    !, strip_module_qualifiers(A, As),
       strip_module_qualifiers(B, Bs).
strip_module_qualifiers((A ; B), (As ; Bs)) :-
    !, strip_module_qualifiers(A, As),
       strip_module_qualifiers(B, Bs).
strip_module_qualifiers((A -> B), (As -> Bs)) :-
    !, strip_module_qualifiers(A, As),
       strip_module_qualifiers(B, Bs).
strip_module_qualifiers(Goal, Goal).

%% emit_kernel(+Pred, +Kernel, -DataDecl, -LoweredFunc, -FuncName)
%  Emits the lowered R function for a detected kernel. Currently
%  handles transitive_closure2 and transitive_distance3; other
%  kinds fall through (the caller falls back to the fact-table or
%  compiled path).
emit_kernel(Pred, recursive_kernel(transitive_closure2, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 2,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_tc2', [RName]),
    DataDecl = "",  % no per-pred data; the kernel just dispatches.
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  WamRuntime$transitive_closure2(program, state, "~w/2", "~w", "~w/2",
                                  source, target, state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred]).
emit_kernel(Pred, recursive_kernel(transitive_distance3, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 2,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_td3', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  dist   <- WamRuntime$get_reg(state, 3L)
  WamRuntime$transitive_distance3(program, state, "~w/3", "~w", "~w/2",
                                   source, target, dist, state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred]).
emit_kernel(Pred, recursive_kernel(weighted_shortest_path3, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 3,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_wsp3', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  weight <- WamRuntime$get_reg(state, 3L)
  WamRuntime$weighted_shortest_path3(program, state, "~w/3", "~w", "~w/3",
                                      source, target, weight, state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred]).
emit_kernel(Pred, recursive_kernel(transitive_parent_distance4, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 2,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_tpd4', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  parent <- WamRuntime$get_reg(state, 3L)
  dist   <- WamRuntime$get_reg(state, 4L)
  WamRuntime$transitive_parent_distance4(program, state, "~w/4", "~w", "~w/2",
                                          source, target, parent, dist,
                                          state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred]).
emit_kernel(Pred, recursive_kernel(transitive_step_parent_distance5, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 2,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_tspd5', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  step   <- WamRuntime$get_reg(state, 3L)
  parent <- WamRuntime$get_reg(state, 4L)
  dist   <- WamRuntime$get_reg(state, 5L)
  WamRuntime$transitive_step_parent_distance5(program, state, "~w/5", "~w", "~w/2",
                                                source, target, step, parent,
                                                dist, state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred]).
emit_kernel(Pred, recursive_kernel(category_ancestor, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 2,
    member(max_depth(MaxDepth), ConfigOps),
    integer(MaxDepth),
    MaxDepth > 0,
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_ca', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source   <- WamRuntime$get_reg(state, 1L)
  ancestor <- WamRuntime$get_reg(state, 2L)
  WamRuntime$category_ancestor(program, state, "~w/4", "~w", "~w/2",
                                ~wL, source, ancestor, state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred, MaxDepth]).
emit_kernel(Pred, recursive_kernel(astar_shortest_path4, _, ConfigOps),
            DataDecl, LoweredFunc, FuncName) :-
    member(edge_pred(EdgePred/EdgeArity), ConfigOps),
    EdgeArity =:= 3,
    % direct_dist_pred(Spec) -- Spec may be a `Name/Arity` pair or
    % a bare atom. Normalize to a name + arity for the codegen.
    member(direct_dist_pred(DistSpec), ConfigOps),
    astar_dist_pred_name(DistSpec, DistPred),
    r_pred_name(Pred, RName),
    format(atom(FuncName), '~w_kernel_astar4', [RName]),
    DataDecl = "",
    format(string(LoweredFunc),
'~w <- function(program, state) {
  source <- WamRuntime$get_reg(state, 1L)
  target <- WamRuntime$get_reg(state, 2L)
  dim    <- WamRuntime$get_reg(state, 3L)
  dist   <- WamRuntime$get_reg(state, 4L)
  WamRuntime$astar_shortest_path4(program, state, "~w/4",
                                    "~w", "~w/3",
                                    "~w", "~w/3",
                                    source, target, dim, dist,
                                    state$pc + 1L)
}',
           [FuncName, Pred, EdgePred, EdgePred, DistPred, DistPred]).

astar_dist_pred_name(Name/_Arity, Name) :- atom(Name), !.
astar_dist_pred_name(Name, Name) :- atom(Name).

% ============================================================================
% EXTERNAL FACT SOURCES
% ============================================================================
%
% Mirrors the Scala target's scala_fact_sources option: users
% declare `r_fact_sources([source(P/A, file('data.csv')), ...])`
% and the codegen wires up a runtime CSV loader that populates the
% standard fact-table data structures (`<pred>_facts` /
% `<pred>_index`) at program-load time. The predicate then
% dispatches via the same fact_table_dispatch path used by
% inline-clause fact tables (PR #1921).
%
% The external-source path WINS over Prolog clause compilation:
% the predicate's body is replaced by a single
% `Execute("P", A)` instruction that falls through to the
% lowered_dispatch tier (which the codegen registers below). No
% Prolog clauses are required for the predicate.

r_fact_source_spec(P, Arity, Options, Spec) :-
    option(r_fact_sources(Sources), Options, []),
    member(source(PI, Spec), Sources),
    fact_source_pi_match(PI, P, Arity).

fact_source_pi_match(_:Name/Ar, P, Arity) :- !,
    Name == P, Ar =:= Arity.
fact_source_pi_match(Name/Ar, P, Arity) :-
    Name == P, Ar =:= Arity.

%% emit_external_fact_source(+P, +Arity, +Spec,
%%                            -DataDecl, -LoweredFunc, -FuncName)
%  Emits the loader + fact-iter dispatch fn. Spec dispatches to a
%  per-shape loader call (CSV, grouped-by-first TSV, ...); the
%  generated loader output feeds the same build_fact_indexes +
%  fact_table_dispatch pipeline used by inline fact tables, so
%  per-arg indexing and dispatch are unchanged across backends.
emit_external_fact_source(Pred, Arity, Spec,
                          DataDecl, LoweredFunc, FuncName) :-
    r_pred_name(Pred, RName),
    format(atom(DataName), '~w_facts', [RName]),
    format(atom(IndexesName), '~w_indexes', [RName]),
    format(atom(FuncName), '~w_fact_iter', [RName]),
    fact_source_loader_call(Spec, Arity, LoaderCall, SpecComment),
    format(string(DataDecl),
'# External fact source for ~w/~w (~w)
~w <- ~w
~w <- WamRuntime$build_fact_indexes(~w, ~wL)',
        [Pred, Arity, SpecComment, DataName, LoaderCall, IndexesName,
         DataName, Arity]),
    fact_args_collect(Arity, ArgsCollect),
    format(string(LoweredFunc),
'~w <- function(program, state) {
  args <- ~w
  WamRuntime$fact_table_dispatch(program, state, "~w/~w", ~w, ~w,
                                  args, state$pc + 1L)
}',
        [FuncName, ArgsCollect, Pred, Arity, DataName, IndexesName]).

%% fact_source_loader_call(+Spec, +Arity, -LoaderCallSrc, -CommentTag)
%  Maps a Spec to the R source that loads the table, plus a short
%  human-readable tag for the comment header. New backends slot in
%  by adding a clause here.
fact_source_loader_call(file(Path), _Arity, LoaderCall, Comment) :-
    atom_string(Path, PathStr),
    r_string_literal(PathStr, PathLit),
    format(string(LoaderCall),
           'WamRuntime$read_facts_csv(~w, intern_table)',
           [PathLit]),
    format(string(Comment), 'csv file: ~w', [PathStr]).
fact_source_loader_call(grouped_by_first(Path), Arity, LoaderCall, Comment) :-
    (   Arity =:= 2
    ->  true
    ;   throw(error(domain_error(arity_2_for_grouped_by_first, Arity), _))
    ),
    atom_string(Path, PathStr),
    r_string_literal(PathStr, PathLit),
    format(string(LoaderCall),
           'WamRuntime$read_facts_grouped_tsv(~w, intern_table)',
           [PathLit]),
    format(string(Comment), 'grouped-by-first tsv file: ~w', [PathStr]).
fact_source_loader_call(lmdb(Path), _Arity, LoaderCall, Comment) :-
    % Step-1 backend: load-everything. Treats LMDB as a serialization
    % format -- the runtime reads all key/value pairs at program-load
    % time and feeds them through the same build_fact_indexes +
    % fact_table_dispatch pipeline as inline / CSV / grouped-TSV
    % tables. Step-2 (probe-on-demand) is a separate follow-up that
    % bypasses the in-memory tuple list; see
    % docs/handoff/wam_r_session_handoff.md item #1.
    atom_string(Path, PathStr),
    r_string_literal(PathStr, PathLit),
    format(string(LoaderCall),
           'WamRuntime$read_facts_lmdb(~w, intern_table)',
           [PathLit]),
    format(string(Comment), 'lmdb env: ~w', [PathStr]).

% ============================================================================
% FACT-TABLE CLASSIFICATION + EMISSION
% ============================================================================
%
% Policy: by default the fact-table path triggers for any predicate
% the classifier accepts (i.e. pure get_constant + proceed clauses)
% with at least one clause. The behaviour can be disabled per-call
% via fact_table_layout(off) in Options or globally via
% user:wam_r_fact_layout/1 — useful for regression bisection.

:- multifile user:wam_r_fact_layout/1.

fact_layout_enabled(_P, _Arity, _NCls, Options) :-
    option(fact_table_layout(Setting), Options, auto),
    fact_layout_setting_ok(Setting).

fact_layout_setting_ok(auto) :-
    (   catch(user:wam_r_fact_layout(Override), _, fail)
    ->  Override \== off
    ;   true
    ).
fact_layout_setting_ok(on).
fact_layout_setting_ok(eager).
%
% A pure fact predicate (every clause is `get_constant` for each arg
% position then `proceed`, with no body call) compiles to a flat R
% list of arg-tuples plus a one-liner lowered function that calls
% WamRuntime$fact_table_iter. This bypasses the WAM stepping engine
% entirely on dispatch -- big speedup for large fact tables.
%
% wam_r_fact_classify(+WamCode, -fact_info(NClauses, Arity, Tuples))
%   Tuples is a list of length NClauses; each element is a list of
%   Arity R-source strings (one per arg slot, e.g. "Atom(7)" /
%   "IntTerm(42)") in arg-1..arg-N order.
%   Fails if any clause body has a call/builtin_call/execute, or if
%   any arg slot is missing a get_constant. (Any other unsupported
%   shape such as get_structure or get_variable also fails.)

wam_r_fact_classify(WamCode, fact_info(NClauses, Arity, Tuples)) :-
    split_string(WamCode, "\n", "", Lines0),
    exclude(=( ""), Lines0, Lines),
    fact_segment_lines(Lines, Segments),
    Segments \= [],
    length(Segments, NClauses),
    Segments = [FirstSeg | _],
    fact_seg_arity(FirstSeg, Arity),
    Arity > 0,
    maplist(fact_seg_arity_eq(Arity), Segments),
    maplist(fact_seg_to_tuple(Arity), Segments, Tuples).

% Group lines into segments by label boundary. A segment is a list of
% non-label lines belonging to one clause (between two label lines or
% between the last label and EOF). The very first label is the
% predicate entry; subsequent labels are clause-alt entries.
fact_segment_lines(Lines, Segments) :-
    fact_segment_walk(Lines, [], [], Segments).

fact_segment_walk([], CurRev, AccRev, Segments) :-
    (   CurRev = []
    ->  reverse(AccRev, Segments)
    ;   reverse(CurRev, Cur),
        reverse([Cur | AccRev], Segments)
    ).
fact_segment_walk([L | Rest], CurRev, AccRev, Segments) :-
    string_chars(L, Chars),
    list_to_set(Chars, _),  % cheap touch
    (   sub_string(L, _, 1, 0, ":")
    ->  % Label line: end current segment if non-empty.
        (   CurRev = []
        ->  fact_segment_walk(Rest, [], AccRev, Segments)
        ;   reverse(CurRev, Cur),
            fact_segment_walk(Rest, [], [Cur | AccRev], Segments)
        )
    ;   tokenize_wam_line(L, Parts),
        (   Parts = []
        ->  fact_segment_walk(Rest, CurRev, AccRev, Segments)
        ;   fact_part_supported(Parts),
            fact_segment_walk(Rest, [Parts | CurRev], AccRev, Segments)
        )
    ).

% Allowed instructions inside a fact clause's body. switch_on_constant
% is variable-length, so we admit any tokenisation that begins with it.
fact_part_supported(["get_constant", _, _]).
fact_part_supported(["proceed"]).
fact_part_supported(["allocate"]).
fact_part_supported(["deallocate"]).
fact_part_supported(["try_me_else", _]).
fact_part_supported(["retry_me_else", _]).
fact_part_supported(["trust_me"]).
fact_part_supported([Head | _]) :- Head == "switch_on_constant".

% Determine the arity of a clause segment from the highest A_i it
% touches via get_constant.
fact_seg_arity(SegParts, Arity) :-
    findall(N, (member(["get_constant", _, AReg], SegParts),
                fact_a_idx(AReg, N)),
            Ns),
    Ns \= [],
    max_list(Ns, Arity).

fact_a_idx(Str, N) :-
    string_chars(Str, [C | Rest]),
    (C == 'A' ; C == 'a'), !,
    string_chars(NStr, Rest),
    number_string(N, NStr),
    integer(N), N >= 1.

fact_seg_arity_eq(Arity, SegParts) :-
    fact_seg_arity(SegParts, Arity).

% Extract the R-source literal for each arg slot in order.
fact_seg_to_tuple(Arity, SegParts, Tuple) :-
    numlist(1, Arity, Idxs),
    maplist(fact_seg_arg_lit(SegParts), Idxs, Tuple).

fact_seg_arg_lit(SegParts, Idx, Lit) :-
    format(atom(Want), 'A~w', [Idx]),
    atom_string(Want, WantStr),
    member(["get_constant", ValStr, WantStr], SegParts), !,
    constant_to_r_term(ValStr, Lit).

%% emit_fact_table(+Pred, +Arity, +Tuples, -DataDecl, -LoweredFunc, -FuncName,
%%                 -RangeRegistration)
%  Renders the per-predicate fact-data declaration and a lowered
%  function that delegates to WamRuntime$fact_table_dispatch.
%  Emits N per-arg hash indexes (one R env per arg position, each
%  mapping "a<id>" / "i<val>" -> integer vector of matching tuple
%  indices), bundled into an `<RName>_indexes <- list(...)` list.
%  Dispatch picks the smallest matching bucket among bound atom/int
%  args and iterates just that bucket; a non-leading-ground query
%  is no longer a full scan.
%
%  Additionally emits per-arg SORTED indexes for any position whose
%  literals are numeric (Int/FloatTerm); these power the range-query
%  builtin `fact_in_range/5`. Each sorted index is a parallel pair
%  of vectors `<RName>_sorted_arg<K> <- list(vals = c(...), idxs =
%  c(...))` sorted ascending by val; the dispatch builtin
%  binary-searches the vals vector to extract the tuple-idx subset
%  for [Lo, Hi]. Positions whose literals are atom-only (no
%  numerics) emit no sorted index and appear as NULL in the
%  per-pred bundle; the builtin treats NULL as "no range support at
%  this position" and returns FALSE.
%
%  RangeRegistration is an R assignment block that registers the
%  per-pred range entry into shared_program$fact_range_indexes;
%  callers stitch it into the lowered-dispatch assignments block.
%
%  Memory: O(N * F) for the hash indexes, plus O(F) per numeric-
%  valued arg for the sorted index (parallel vectors). Per-arg,
%  no composite keys.
emit_fact_table(Pred, Arity, Tuples, DataDecl, LoweredFunc, FuncName,
                RangeRegistration) :-
    r_pred_name(Pred, RName),
    format(atom(DataName), '~w_facts', [RName]),
    format(atom(IndexesName), '~w_indexes', [RName]),
    format(atom(RangeIndexesName), '~w_range_indexes', [RName]),
    format(atom(FuncName), '~w_fact_iter', [RName]),
    length(Tuples, NTuples),
    fact_tuples_to_r_list(Tuples, ListBody),
    fact_indexes_block(Tuples, Arity, RName, IndexesName, IndexBody),
    fact_sorted_indexes_block(Tuples, Arity, RName, RangeIndexesName,
                              SortedBody),
    format(string(DataDecl),
'# Fact table for ~w/~w (~w tuples)
~w <- list(
~w
)
~w
~w', [Pred, Arity, NTuples, DataName, ListBody, IndexBody, SortedBody]),
    fact_args_collect(Arity, ArgsCollect),
    format(string(LoweredFunc),
'~w <- function(program, state) {
  args <- ~w
  WamRuntime$fact_table_dispatch(program, state, "~w/~w", ~w, ~w,
                                  args, state$pc + 1L)
}', [FuncName, ArgsCollect, Pred, Arity, DataName, IndexesName]),
    format(string(RangeRegistration),
'assign("~w/~w", list(facts = ~w, range = ~w), envir = shared_program$fact_range_indexes)',
           [Pred, Arity, DataName, RangeIndexesName]).

%% fact_sorted_indexes_block(+Tuples, +Arity, +RName, +RangeIndexesName, -Body)
%  Emits per-arg sorted-by-value indexes alongside the hash indexes
%  (one `<RName>_sorted_arg<K> <- list(vals = c(...), idxs = c(...))`
%  block per numeric-valued position) plus a bundle list
%  `<RangeIndexesName> <- list(arg<K> = <sorted-or-NULL>, ...)`.
%  Numeric-valued = IntTerm() / FloatTerm() literals at that
%  position; atom positions contribute no sorted entry.
fact_sorted_indexes_block(Tuples, Arity, RName, RangeIndexesName, Body) :-
    must_be(positive_integer, Arity),
    numlist(1, Arity, ArgPositions),
    maplist(fact_sorted_per_arg(Tuples, RName), ArgPositions,
            PerArgBlocks, PerArgBundleEntries),
    exclude(==(''), PerArgBlocks, NonEmptyBlocks),
    atomic_list_concat(NonEmptyBlocks, '\n', BlocksJoined),
    atomic_list_concat(PerArgBundleEntries, ', ', BundleEntries),
    (   BlocksJoined == ''
    ->  format(string(Body), '~w <- list(~w)',
               [RangeIndexesName, BundleEntries])
    ;   format(string(Body), '~w\n~w <- list(~w)',
               [BlocksJoined, RangeIndexesName, BundleEntries])
    ).

% Emit the sorted-index block for one arg position. Block is the R
% assignment for `<RName>_sorted_arg<K>` (empty string when no
% numerics at that position). BundleEntry is the `arg<K> = <name>` or
% `arg<K> = NULL` fragment for the bundle list.
fact_sorted_per_arg(Tuples, RName, ArgPos, Block, BundleEntry) :-
    format(atom(SortedName), '~w_sorted_arg~w', [RName, ArgPos]),
    fact_sorted_pairs_at(Tuples, ArgPos, 1, Pairs0),
    (   Pairs0 == []
    ->  Block = '',
        format(atom(BundleEntry), 'arg~w = NULL', [ArgPos])
    ;   keysort(Pairs0, Pairs1),
        pairs_keys_values(Pairs1, Vals, Idxs),
        atomic_list_concat(Vals, ', ', ValsStr),
        maplist([N, S]>>format(string(S), '~wL', [N]), Idxs, IdxStrs),
        atomic_list_concat(IdxStrs, ', ', IdxsStr),
        format(string(Block),
               '~w <- list(vals = c(~w), idxs = c(~w))',
               [SortedName, ValsStr, IdxsStr]),
        format(atom(BundleEntry), 'arg~w = ~w', [ArgPos, SortedName])
    ).

% Collect (NumericValue, TupleIdx) pairs for an arg position. Skips
% tuples whose literal at ArgPos isn't IntTerm() / FloatTerm() --
% those positions contribute nothing to the sorted index.
fact_sorted_pairs_at([], _ArgPos, _Idx, []).
fact_sorted_pairs_at([Tuple | Rest], ArgPos, Idx, Pairs) :-
    (   nth1(ArgPos, Tuple, Lit),
        fact_sorted_value(Lit, N)
    ->  Pairs = [N-Idx | RestPairs]
    ;   Pairs = RestPairs
    ),
    Idx1 is Idx + 1,
    fact_sorted_pairs_at(Rest, ArgPos, Idx1, RestPairs).

% Extract a numeric value from a fact-table literal. IntTerm(N) and
% FloatTerm(N) are recognised; everything else (Atom, struct, list)
% fails so the position is skipped.
fact_sorted_value(Lit, N) :-
    (   sub_string(Lit, 0, 8, _, "IntTerm(")
    ->  string_concat("IntTerm(", AfterPrefix, Lit),
        string_concat(NumStr, ")", AfterPrefix),
        number_string(N, NumStr)
    ;   sub_string(Lit, 0, 10, _, "FloatTerm(")
    ->  string_concat("FloatTerm(", AfterPrefix, Lit),
        string_concat(NumStr, ")", AfterPrefix),
        number_string(N, NumStr)
    ).

% Build the per-arg index population block. For each arg position
% 1..Arity, emit a `<RName>_index_arg<K> <- new.env(...)` plus one
% `assign("a<id>"/"i<val>", c(...), envir = ...)` line per bucket.
% Finally bundle all per-arg envs into `<IndexesName> <- list(...)`.
%
% Arity must be a positive integer: the fact-table classifier requires
% `get_constant + proceed`, which implies arity >= 1. An arity-0 fact
% table would emit `<IndexesName> <- list()`, which fact_table_dispatch
% would reject loudly via its own arity-vs-indexes check, but it's
% still better to fail at codegen time with a clear domain error.
fact_indexes_block(Tuples, Arity, RName, IndexesName, Body) :-
    must_be(positive_integer, Arity),
    numlist(1, Arity, ArgPositions),
    % maplist/4 with two output lists: each call to fact_index_per_arg/5
    % produces one block of R code (Block) and the matching env name
    % (IndexName), zipped over ArgPositions.
    maplist(fact_index_per_arg(Tuples, RName), ArgPositions, PerArgBlocks,
            PerArgNames),
    atomic_list_concat(PerArgBlocks, '\n', Joined),
    atomic_list_concat(PerArgNames, ', ', NamesList),
    format(string(Body), '~w\n~w <- list(~w)',
           [Joined, IndexesName, NamesList]).

% Emit the new.env + bucket assigns for a single arg position.
fact_index_per_arg(Tuples, RName, ArgPos, Block, IndexName) :-
    format(atom(IndexName), '~w_index_arg~w', [RName, ArgPos]),
    fact_index_pairs_at(Tuples, ArgPos, 1, Pairs),
    fact_index_group(Pairs, [], Buckets),
    maplist(fact_index_emit_assign(IndexName), Buckets, AssignLines),
    atomic_list_concat(AssignLines, '\n', AssignsBody),
    (   AssignsBody == ''
    ->  format(string(Block), '~w <- new.env(parent = emptyenv())',
               [IndexName])
    ;   format(string(Block), '~w <- new.env(parent = emptyenv())\n~w',
               [IndexName, AssignsBody])
    ).

fact_index_pairs_at([], _ArgPos, _Idx, []).
fact_index_pairs_at([Tuple | Rest], ArgPos, Idx, [Key-Idx | RestPairs]) :-
    nth1(ArgPos, Tuple, Lit),
    fact_index_key(Lit, Key), !,
    Idx1 is Idx + 1,
    fact_index_pairs_at(Rest, ArgPos, Idx1, RestPairs).
fact_index_pairs_at([_ | Rest], ArgPos, Idx, RestPairs) :-
    Idx1 is Idx + 1,
    fact_index_pairs_at(Rest, ArgPos, Idx1, RestPairs).

% Source literals are produced by constant_to_r_term: "Atom(<id>)"
% or "IntTerm(<val>)" / "FloatTerm(<val>)". We pull the prefix to
% pick the bucket namespace; floats get no index (still scanable via
% the fallback linear path).
fact_index_key(Lit, Key) :-
    (   sub_string(Lit, 0, 5, _, "Atom(")
    ->  string_concat("Atom(", AfterPrefix, Lit),
        string_concat(NumStr, ")", AfterPrefix),
        format(string(Key), 'a~w', [NumStr])
    ;   sub_string(Lit, 0, 8, _, "IntTerm(")
    ->  string_concat("IntTerm(", AfterPrefix, Lit),
        string_concat(NumStr, ")", AfterPrefix),
        format(string(Key), 'i~w', [NumStr])
    ).

fact_index_group([], Acc, Buckets) :-
    reverse(Acc, Buckets).
fact_index_group([Key-Idx | Rest], Acc, Buckets) :-
    (   select(Key-Idxs, Acc, Acc1)
    ->  append(Idxs, [Idx], NewIdxs),
        fact_index_group(Rest, [Key-NewIdxs | Acc1], Buckets)
    ;   fact_index_group(Rest, [Key-[Idx] | Acc], Buckets)
    ).

fact_index_emit_assign(IndexName, Key-Idxs, Line) :-
    maplist([N, S]>>format(string(S), '~wL', [N]), Idxs, IdxStrs),
    atomic_list_concat(IdxStrs, ', ', IdxList),
    format(string(Line),
           'assign("~w", c(~w), envir = ~w)', [Key, IdxList, IndexName]).

fact_tuples_to_r_list(Tuples, Body) :-
    maplist(fact_tuple_to_r_entry, Tuples, Entries),
    atomic_list_concat(Entries, ',\n', Body).

fact_tuple_to_r_entry(Tuple, Entry) :-
    atomic_list_concat(Tuple, ', ', ArgList),
    format(string(Entry), '  list(~w)', [ArgList]).

fact_args_collect(0, 'list()') :- !.
fact_args_collect(Arity, Code) :-
    Arity > 0,
    numlist(1, Arity, Idxs),
    maplist([N, S]>>format(string(S),
            'WamRuntime$get_reg(state, ~wL)', [N]), Idxs, Parts),
    atomic_list_concat(Parts, ', ', Joined),
    format(string(Code), 'list(~w)', [Joined]).

% ============================================================================
% FOREIGN PREDICATE DETECTION
% ============================================================================

%% r_foreign_predicate(+Pred, +Arity, +Options) is semidet.
r_foreign_predicate(Pred, Arity, Options) :-
    (   string(Pred)
    ->  atom_string(PredAtom, Pred)
    ;   PredAtom = Pred
    ),
    option(foreign_predicates(FPs), Options, []),
    (   member(PredAtom/Arity, FPs)
    ;   member(_:PredAtom/Arity, FPs)
    ), !.

%% r_foreign_handlers_code(+Options, -Code) is det.
%  Renders the body of the foreign-handler list. Reads
%  r_foreign_handlers([handler(P/A, "<R-expr>"), ...]) from Options.
%  Each handler value is an R expression returning a function
%  function(state, args) -> list(ok=TRUE/FALSE, sols=list(...)).
r_foreign_handlers_code(Options, Code) :-
    option(r_foreign_handlers(Handlers), Options, []),
    maplist(r_foreign_handler_entry, Handlers, Entries),
    atomic_list_concat(Entries, ',\n', Code).

r_foreign_handler_entry(handler(Pred/Arity, HandlerCode), Entry) :-
    format(string(Entry), '    "~w/~w" = ~w', [Pred, Arity, HandlerCode]).

%% r_op_decls_code(+Options, -Code) is det.
%  Renders the program-init operator-declaration block. Reads
%  r_op_decls([op(P, T, N), ...]) from Options. Each `op(P, T, N)`
%  emits one `WamRuntime$op_set("<N>", <P>L, "<T>")` call so the
%  parser sees the operator before any read_term_from_atom or CLI
%  arg parsing happens. Names that collide with the built-in ops
%  override the built-in entry.
r_op_decls_code(Options, Code) :-
    option(r_op_decls(Decls), Options, []),
    maplist(r_op_decl_line, Decls, Lines),
    atomic_list_concat(Lines, '\n', Code).

r_op_decl_line(op(Prec, Type, Name), Line) :-
    must_be(integer, Prec),
    must_be(atom, Type),
    Prec >= 0, Prec =< 1200,
    memberchk(Type, [xfx, xfy, yfx, fx, fy, xf, yf]),
    (   atom(Name)
    ->  r_string_literal_atom(Name, NameLit),
        format(string(Line), 'WamRuntime$op_set(~w, ~wL, "~w")',
               [NameLit, Prec, Type])
    ;   is_list(Name),
        forall(member(N, Name), must_be(atom, N))
    ->  maplist([N, NL]>>r_string_literal_atom(N, NL), Name, NameLits),
        atomic_list_concat(NameLits, ', ', NamesJoined),
        format(string(Line),
               'for (nm in c(~w)) WamRuntime$op_set(nm, ~wL, "~w")',
               [NamesJoined, Prec, Type])
    ).

% Quote an atom as a double-quoted R string literal, escaping
% backslashes and double quotes. Used by r_op_decls_code so an op
% name like `\\+` survives as `"\\\\+"` in emitted R source.
r_string_literal_atom(Atom, Lit) :-
    atom_string(Atom, S),
    string_chars(S, Chars),
    r_escape_chars(Chars, Escaped),
    string_chars(EscStr, Escaped),
    format(string(Lit), '"~w"', [EscStr]).

r_escape_chars([], []).
r_escape_chars(['\\' | T], ['\\', '\\' | RestEsc]) :- !,
    r_escape_chars(T, RestEsc).
r_escape_chars(['"'  | T], ['\\', '"'  | RestEsc]) :- !,
    r_escape_chars(T, RestEsc).
r_escape_chars([C    | T], [C | RestEsc]) :-
    r_escape_chars(T, RestEsc).

% ============================================================================
% PROJECT WRITER
% ============================================================================

%% write_wam_r_project(+Predicates, +Options, +ProjectDir) is det.
%  Creates a complete R hybrid-WAM project at ProjectDir:
%    DESCRIPTION                R package metadata
%    R/wam_runtime.R            WAM runtime
%    R/generated_program.R      compiled program (instr array, labels, wrappers)
write_wam_r_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'R', RDir),
    make_directory_path(RDir),
    option(module_name(ModName), Options, 'wam.r.generated'),
    write_description(ProjectDir, ModName),
    compile_predicates_for_project(Predicates, Options,
        AllInstrs, TopLevelLabelEntries, AllLabelEntries,
        WrapperCode, LoweredFunctionsCode, FactShapeComments,
        LoweredDispatchCode),
    emit_r_intern_table(IdToStringStr),
    maplist([I, Line]>>(format(string(Line), '    ~w', [I])), AllInstrs,
            InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    atomic_list_concat(TopLevelLabelEntries, ',\n', DispatchBody),
    atomic_list_concat(AllLabelEntries, ',\n', LabelBody),
    r_foreign_handlers_code(Options, ForeignHandlersBody),
    r_op_decls_code(Options, OpDeclsBody),
    write_runtime_source(RDir),
    write_program_source(RDir, InstrBody, LabelBody, DispatchBody,
                         WrapperCode, IdToStringStr, ForeignHandlersBody,
                         LoweredFunctionsCode, FactShapeComments,
                         LoweredDispatchCode, OpDeclsBody).

write_description(ProjectDir, ModName) :-
    find_template('templates/targets/r_wam/DESCRIPTION.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template,
        ['module_name'=ModName, 'date'=DateStr], Content),
    directory_file_path(ProjectDir, 'DESCRIPTION', Path),
    write_file(Path, Content).

write_runtime_source(RDir) :-
    find_template('templates/targets/r_wam/runtime.R.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template, ['date'=DateStr], Content),
    directory_file_path(RDir, 'wam_runtime.R', Path),
    write_file(Path, Content).

write_program_source(RDir, InstrBody, LabelBody, DispatchBody,
                     WrapperCode, IdToStringStr, ForeignHandlersBody,
                     LoweredFunctionsCode, FactShapeComments,
                     LoweredDispatchCode, OpDeclsBody) :-
    find_template('templates/targets/r_wam/program.R.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template,
        [ 'date'=DateStr,
          'instructions'=InstrBody,
          'labels'=LabelBody,
          'dispatch'=DispatchBody,
          'wrappers'=WrapperCode,
          'intern_id_to_string'=IdToStringStr,
          'lowered_dispatch_assignments'=LoweredDispatchCode,
          'foreign_handlers'=ForeignHandlersBody,
          'lowered_functions'=LoweredFunctionsCode,
          'fact_shape_comments'=FactShapeComments,
          'op_decls'=OpDeclsBody
        ], Content),
    directory_file_path(RDir, 'generated_program.R', Path),
    write_file(Path, Content).

% ============================================================================
% HELPERS
% ============================================================================

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, Content),
        close(Stream)
    ).

%% find_template(+RelPath, -Template) is det.
%  Locates a template file relative to the UnifyWeaver project root.
find_template(RelPath, Template) :-
    (   source_file(wam_r_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, UnifyWeaverDir),
        atomic_list_concat([UnifyWeaverDir, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, []).

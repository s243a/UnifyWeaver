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
    reverse([T|Acc], Tokens).
tokenize_wam_chars([], CurR, Acc, inside, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T, CurC),
    reverse([T|Acc], Tokens).
tokenize_wam_chars([C|Rest], CurR, Acc, outside, Tokens) :-
    (   (C == ' ' ; C == '\t')
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, outside, Tokens)
        ;   reverse(CurR, CurC), string_chars(T0, CurC),
            strip_operand_comma(T0, T),
            tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
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
    split_string(Str, "\n", "", Lines),
    wam_lines_to_data(Lines, Options, 1, Instructions, LabelMap, LabelEntries).

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
% PREDICATE COMPILATION
% ============================================================================

%% compile_wam_predicate_to_r(+PredIndicator, +WamCode, +Options, -RCode)
%  Provided for symmetry with the other targets. The hybrid pipeline
%  uses write_wam_r_project/3, not this.
compile_wam_predicate_to_r(_Pred, _WamCode, _Options, "").

%% compile_predicates_for_project(+Predicates, +Options,
%%                                -AllInstrs, -TopLabels,
%%                                -AllLabels, -WrapperCode)
compile_predicates_for_project(Predicates, Options,
                               AllInstrs, TopLevelLabelEntries,
                               AllLabelEntries, WrapperCode, LoweredCode) :-
    init_r_atom_intern_table,
    option(intern_atoms(ExtraAtoms), Options, []),
    forall(member(A, ExtraAtoms),
           (atom_string(A, S), intern_r_atom(S, _))),
    option(foreign_predicates(ForeignPredicates), Options, []),
    append_missing_foreign_predicates(Predicates, ForeignPredicates,
                                      CompilePredicates),
    wam_r_resolve_emit_mode(Options, Mode),
    compile_all_predicates(CompilePredicates, Options, Mode, 1,
                           [], [], [], [], [],
                           AllInstrs, TopLevelLabelEntries,
                           AllLabelEntries, Wrappers, LoweredEntries),
    atomic_list_concat(Wrappers, '\n', WrapperCode),
    atomic_list_concat(LoweredEntries, '\n', LoweredCode).

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

compile_all_predicates([], _, _, _, Instrs, TopLabels, AllLabels, Wrappers, Lowered,
                       Instrs, TopLabels, AllLabels, Wrappers, Lowered).
compile_all_predicates([Pred|Rest], Options, Mode, BasePC,
                       InstrAcc, TopLabelAcc, AllLabelAcc, WrapperAcc, LoweredAcc,
                       AllInstrs, TopLevelLabelEntries,
                       AllLabelEntries, AllWrappers, AllLowered) :-
    (   Pred = _Module:P/Arity -> true ; Pred = P/Arity ),
    (   r_foreign_predicate(P, Arity, Options)
    ->  format(string(FLit), 'CallForeign("~w", ~w)', [P, Arity]),
        ForeignSeq = [FLit, 'Proceed()'],
        append(InstrAcc, ForeignSeq, NewInstrs),
        NewPC is BasePC + 2,
        format(string(MainEntry), '    "~w/~w" = ~wL', [P, Arity, BasePC]),
        NewTopLabels = [MainEntry | TopLabelAcc],
        NewAllLabels = [MainEntry | AllLabelAcc],
        WamCodeForLower = ""
    ;   compile_predicate_to_wam(P/Arity, [], WamCode),
        WamCodeForLower = WamCode,
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
    % Decide whether this predicate should be lowered.
    (   should_try_lower(Mode, P, Arity),
        WamCodeForLower \= "",
        catch(wam_r_lowerable(Pred, WamCodeForLower, _Reason), _, fail),
        catch(lower_predicate_to_r(Pred, WamCodeForLower, [],
                                   lowered(_PName, FuncName, LoweredR)),
              _, fail)
    ->  NewLoweredAcc = [LoweredR | LoweredAcc],
        emit_r_lowered_wrapper(P, Arity, FuncName, WrapperCode)
    ;   NewLoweredAcc = LoweredAcc,
        emit_r_wrapper(P, Arity, BasePC, WrapperCode)
    ),
    compile_all_predicates(Rest, Options, Mode, NewPC,
                           NewInstrs, NewTopLabels, NewAllLabels,
                           [WrapperCode|WrapperAcc], NewLoweredAcc,
                           AllInstrs, TopLevelLabelEntries,
                           AllLabelEntries, AllWrappers, AllLowered).

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
%  R uses snake_case by convention; identifiers are looser than Scala
%  so we keep the Prolog name verbatim, with safety substitutions.
r_pred_name(Pred, RName) :-
    atom_string(Pred, PStr),
    string_chars(PStr, Chars),
    maplist(r_safe_ident_char, Chars, SafeChars),
    string_chars(SafeStr, SafeChars),
    % R identifiers can't start with a digit
    (   string_chars(SafeStr, [First|_]),
        char_type(First, digit)
    ->  string_concat("p_", SafeStr, RName0)
    ;   RName0 = SafeStr
    ),
    atom_string(RName, RName0).

r_safe_ident_char(C, C) :-
    char_type(C, alnum), !.
r_safe_ident_char('.', '.') :- !.
r_safe_ident_char('_', '_') :- !.
r_safe_ident_char(_, '_').

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
        WrapperCode, LoweredFunctionsCode),
    emit_r_intern_table(IdToStringStr),
    maplist([I, Line]>>(format(string(Line), '    ~w', [I])), AllInstrs,
            InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    atomic_list_concat(TopLevelLabelEntries, ',\n', DispatchBody),
    atomic_list_concat(AllLabelEntries, ',\n', LabelBody),
    r_foreign_handlers_code(Options, ForeignHandlersBody),
    write_runtime_source(RDir),
    write_program_source(RDir, InstrBody, LabelBody, DispatchBody,
                         WrapperCode, IdToStringStr, ForeignHandlersBody,
                         LoweredFunctionsCode).

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
                     LoweredFunctionsCode) :-
    find_template('templates/targets/r_wam/program.R.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template,
        [ 'date'=DateStr,
          'instructions'=InstrBody,
          'labels'=LabelBody,
          'dispatch'=DispatchBody,
          'wrappers'=WrapperCode,
          'intern_id_to_string'=IdToStringStr,
          'foreign_handlers'=ForeignHandlersBody,
          'lowered_functions'=LoweredFunctionsCode
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

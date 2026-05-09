:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_lua_target.pl - WAM-to-Lua Hybrid Transpilation Target
%
% Generates a Lua project with an instruction-array WAM interpreter plus
% optional lowered per-predicate Lua functions. The architecture mirrors
% the R/Scala/Haskell hybrid WAM targets while using Lua tables for both
% terms and instructions.

:- module(wam_lua_target, [
    compile_wam_predicate_to_lua/4,
    write_wam_lua_project/3,
    lua_foreign_predicate/3,
    init_lua_atom_intern_table/0,
    tokenize_wam_line/2,
    wam_parts_to_lua/3,
    parse_functor_arity/3,
    reg_to_int/2,
    constant_to_lua_term/2,
    intern_lua_atom/2,
    lua_string_literal/2,
    wam_lua_resolve_emit_mode/2
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module(wam_lua_lowered_emitter, [
    wam_lua_lowerable/3,
    lower_predicate_to_lua/4
]).

:- multifile user:wam_lua_emit_mode/1.

wam_lua_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_emit_mode(M0, Mode)
    ;   catch(user:wam_lua_emit_mode(M1), _, fail)
    ->  validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_emit_mode(interpreter, interpreter) :- !.
validate_emit_mode(functions, functions) :- !.
validate_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_lua_emit_mode, Other),
                wam_lua_resolve_emit_mode/2)).

should_try_lower(functions, _, _) :- !.
should_try_lower(mixed(HotPreds), P, A) :-
    member(P/A, HotPreds), !.
should_try_lower(_, _, _) :- fail.

% ============================================================================
% Atom interning
% ============================================================================

:- dynamic lua_atom_intern_id/2.
:- dynamic lua_atom_intern_next/1.

init_lua_atom_intern_table :-
    retractall(lua_atom_intern_id(_, _)),
    retractall(lua_atom_intern_next(_)),
    assertz(lua_atom_intern_id("true", 0)),
    assertz(lua_atom_intern_id("fail", 1)),
    assertz(lua_atom_intern_id("[]", 2)),
    assertz(lua_atom_intern_id(".", 3)),
    assertz(lua_atom_intern_id("", 4)),
    assertz(lua_atom_intern_id("[|]", 5)),
    assertz(lua_atom_intern_next(6)).

intern_lua_atom(AtomStr, Id) :-
    (   lua_atom_intern_next(_)
    ->  true
    ;   init_lua_atom_intern_table
    ),
    text_to_string(AtomStr, Str),
    (   lua_atom_intern_id(Str, Id0)
    ->  Id = Id0
    ;   retract(lua_atom_intern_next(Next)),
        Id = Next,
        Next1 is Next + 1,
        assertz(lua_atom_intern_id(Str, Id)),
        assertz(lua_atom_intern_next(Next1))
    ).

emit_lua_intern_table(Code) :-
    findall(Id-Str, lua_atom_intern_id(Str, Id), Pairs),
    sort(Pairs, Sorted),
    maplist([_Id-Str, E]>>(
        lua_string_literal(Str, Lit),
        format(string(E), '  ~w', [Lit])
    ), Sorted, Entries),
    atomic_list_concat(Entries, ',\n', Code).

% ============================================================================
% WAM tokens and literals
% ============================================================================

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

tokenize_wam_line(Line, Tokens) :-
    string_chars(Line, Chars),
    tokenize_wam_chars(Chars, [], [], outside, Tokens).

tokenize_wam_chars([], [], Acc, _, Tokens) :- !,
    reverse(Acc, Tokens).
tokenize_wam_chars([], CurR, Acc, outside, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T0, CurC),
    strip_operand_comma(T0, T),
    (T == "" -> reverse(Acc, Tokens) ; reverse([T|Acc], Tokens)).
tokenize_wam_chars([], CurR, Acc, inside, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T, CurC),
    reverse([T|Acc], Tokens).
tokenize_wam_chars([C|Rest], CurR, Acc, outside, Tokens) :-
    (   (C == ' ' ; C == '\t')
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, outside, Tokens)
        ;   reverse(CurR, CurC), string_chars(T0, CurC),
            strip_operand_comma(T0, T),
            (T == "" -> NewAcc = Acc ; NewAcc = [T|Acc]),
            tokenize_wam_chars(Rest, [], NewAcc, outside, Tokens)
        )
    ;   C == '\''
    ->  (CurR == [] -> tokenize_wam_chars(Rest, [], Acc, inside, Tokens)
        ; tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens))
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
    ).
tokenize_wam_chars([C|Rest], CurR, Acc, inside, Tokens) :-
    (   C == '\\', Rest = [Escaped|More]
    ->  tokenize_wam_chars(More, [Escaped|CurR], Acc, inside, Tokens)
    ;   C == '\''
    ->  reverse(CurR, CurC), string_chars(T, CurC),
        tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, inside, Tokens)
    ).

strip_operand_comma(Token0, Token) :-
    sub_string(Token0, _, 1, 0, ","), !,
    sub_string(Token0, 0, _, 1, Token).
strip_operand_comma(Token, Token).

wam_parts_to_lua(["call", PredArity], Options, Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    lua_foreign_predicate(PredName, Arity, Options), !,
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.CallForeign(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["call", Pred, ArityStr], Options, Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    lua_foreign_predicate(PredName, Arity, Options), !,
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.CallForeign(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["execute", PredArity], Options, Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    lua_foreign_predicate(PredName, Arity, Options), !,
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.ExecuteForeign(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["execute", Pred, ArityStr], Options, Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    lua_foreign_predicate(PredName, Arity, Options), !,
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.ExecuteForeign(~w, ~w)', [P, Arity]).
wam_parts_to_lua(Parts, _Options, Lit) :-
    wam_parts_to_lua(Parts, Lit).

wam_parts_to_lua(["call", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.Call(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.Call(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["execute", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.Execute(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["execute", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    lua_string_literal(PredName, P),
    format(string(Lit), 'I.Execute(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["proceed"], 'I.Proceed()').
wam_parts_to_lua(["fail"], 'I.Fail()').
wam_parts_to_lua(["jump", Label], Lit) :-
    lua_string_literal(Label, L),
    format(string(Lit), 'I.Jump(~w)', [L]).
wam_parts_to_lua(["try_me_else", Label], Lit) :-
    lua_string_literal(Label, L),
    format(string(Lit), 'I.TryMeElse(~w)', [L]).
wam_parts_to_lua(["retry_me_else", Label], Lit) :-
    lua_string_literal(Label, L),
    format(string(Lit), 'I.RetryMeElse(~w)', [L]).
wam_parts_to_lua(["trust_me"], 'I.TrustMe()').
wam_parts_to_lua(["allocate"], 'I.Allocate()').
wam_parts_to_lua(["deallocate"], 'I.Deallocate()').
wam_parts_to_lua(["get_constant", C, Reg], Lit) :-
    reg_to_int(Reg, R), constant_to_lua_term(C, T),
    format(string(Lit), 'I.GetConstant(~w, ~w)', [T, R]).
wam_parts_to_lua(["get_variable", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.GetVariable(~w, ~w)', [XI, AI]).
wam_parts_to_lua(["get_value", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.GetValue(~w, ~w)', [XI, AI]).
wam_parts_to_lua(["put_constant", C, Reg], Lit) :-
    reg_to_int(Reg, R), constant_to_lua_term(C, T),
    format(string(Lit), 'I.PutConstant(~w, ~w)', [T, R]).
wam_parts_to_lua(["put_variable", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.PutVariable(~w, ~w)', [XI, AI]).
wam_parts_to_lua(["put_value", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.PutValue(~w, ~w)', [XI, AI]).
wam_parts_to_lua(["put_structure", F, Reg], Lit) :-
    reg_to_int(Reg, R), parse_functor_arity(F, Name, Arity),
    intern_lua_atom(Name, Id),
    format(string(Lit), 'I.PutStructure(~w, ~w, ~w)', [Id, R, Arity]).
wam_parts_to_lua(["get_structure", F, Reg], Lit) :-
    reg_to_int(Reg, R), parse_functor_arity(F, Name, Arity),
    intern_lua_atom(Name, Id),
    format(string(Lit), 'I.GetStructure(~w, ~w, ~w)', [Id, R, Arity]).
wam_parts_to_lua(["put_list", Reg], Lit) :-
    reg_to_int(Reg, R), intern_lua_atom("[|]", Id),
    format(string(Lit), 'I.PutList(~w, ~w)', [R, Id]).
wam_parts_to_lua(["get_list", Reg], Lit) :-
    reg_to_int(Reg, R), intern_lua_atom("[|]", Id),
    format(string(Lit), 'I.GetList(~w, ~w)', [R, Id]).
wam_parts_to_lua(["set_variable", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.SetVariable(~w)', [XI]).
wam_parts_to_lua(["set_value", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.SetValue(~w)', [XI]).
wam_parts_to_lua(["set_constant", C], Lit) :-
    constant_to_lua_term(C, T), format(string(Lit), 'I.SetConstant(~w)', [T]).
wam_parts_to_lua(["unify_variable", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.UnifyVariable(~w)', [XI]).
wam_parts_to_lua(["unify_value", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.UnifyValue(~w)', [XI]).
wam_parts_to_lua(["unify_constant", C], Lit) :-
    constant_to_lua_term(C, T), format(string(Lit), 'I.UnifyConstant(~w)', [T]).
wam_parts_to_lua(["builtin_call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr), lua_string_literal(Pred, P),
    format(string(Lit), 'I.BuiltinCall(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["call_foreign", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr), lua_string_literal(Pred, P),
    format(string(Lit), 'I.CallForeign(~w, ~w)', [P, Arity]).
wam_parts_to_lua(["call_indexed_atom_fact2", Pred], Lit) :-
    strip_operand_comma(Pred, CleanPred),
    lua_string_literal(CleanPred, P),
    format(string(Lit), 'I.CallIndexedAtomFact2(~w)', [P]).
wam_parts_to_lua(["arg", NStr, RegStr, OutRegStr], Lit) :-
    number_string(N, NStr), reg_to_int(RegStr, R), reg_to_int(OutRegStr, O),
    format(string(Lit), 'I.ArgInstr(~w, ~w, ~w)', [N, R, O]).
wam_parts_to_lua(["switch_on_constant" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, Norm),
    parse_switch_cases(Norm, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'I.SwitchOnConstant({~w})', [CasesStr]).
wam_parts_to_lua(["switch_on_constant_a2" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, Norm),
    parse_switch_cases(Norm, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'I.SwitchOnConstantA2({~w})', [CasesStr]).
wam_parts_to_lua(["switch_on_structure" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, Norm),
    parse_struct_switch_cases(Norm, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'I.SwitchOnStructure({~w})', [CasesStr]).
wam_parts_to_lua(["cut_ite"], 'I.CutIte()').
wam_parts_to_lua(["begin_aggregate", Kind, TemplateReg, BagReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    reg_to_int(BagReg, BIdx),
    lua_string_literal(Kind, K),
    format(string(Lit), 'I.BeginAggregate(~w, ~w, ~w)', [K, TIdx, BIdx]).
wam_parts_to_lua(["end_aggregate", TemplateReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    format(string(Lit), 'I.EndAggregate(~w)', [TIdx]).
wam_parts_to_lua(Parts, Lit) :-
    atomic_list_concat(Parts, ' ', Text),
    lua_string_literal(Text, Q),
    format(string(Lit), 'I.Raw(~w)', [Q]).

parse_switch_cases([], []).
parse_switch_cases([Token|Rest], [Lit|More]) :-
    split_at_first_colon(Token, ValStr, LabelStr),
    intern_lua_atom(ValStr, AtomId),
    lua_string_literal(LabelStr, L),
    format(string(Lit), '{value = V.Atom(~w), label = ~w}', [AtomId, L]),
    parse_switch_cases(Rest, More).

parse_struct_switch_cases([], []).
parse_struct_switch_cases([Token|Rest], [Lit|More]) :-
    split_at_first_colon(Token, FAStr, LabelStr),
    parse_functor_arity(FAStr, FName, FArity),
    intern_lua_atom(FName, FId),
    lua_string_literal(LabelStr, L),
    format(string(Lit), '{fid = ~w, arity = ~w, label = ~w}', [FId, FArity, L]),
    parse_struct_switch_cases(Rest, More).

normalize_switch_case_tokens([], []).
normalize_switch_case_tokens([Value, Label0|Rest], [Token|More]) :-
    \+ sub_string(Value, _, 1, _, ":"),
    sub_string(Label0, 0, 1, _, ":"), !,
    sub_string(Label0, 1, _, 0, Label),
    string_concat(Value, ":", Prefix),
    string_concat(Prefix, Label, Token),
    normalize_switch_case_tokens(Rest, More).
normalize_switch_case_tokens([Token|Rest], [Token|More]) :-
    normalize_switch_case_tokens(Rest, More).

constant_to_lua_term(C, Lit) :-
    (   number_string(N, C), integer(N)
    ->  format(string(Lit), 'V.Int(~w)', [N])
    ;   number_string(F, C), float(F)
    ->  format(string(Lit), 'V.Float(~w)', [F])
    ;   intern_lua_atom(C, Id),
        format(string(Lit), 'V.Atom(~w)', [Id])
    ).

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
    sub_string(Token, B, 1, _, ":"), !,
    sub_string(Token, 0, B, _, Before),
    B1 is B + 1,
    sub_string(Token, B1, _, 0, After).

strip_arity_suffix(Pred, Name) :-
    (sub_string(Pred, B, 1, _, "/") -> sub_string(Pred, 0, B, _, Name) ; Name = Pred).

lua_string_literal(Raw, Quoted) :-
    text_to_string(Raw, S),
    string_chars(S, Chars),
    maplist(lua_string_escape_char, Chars, EscapedLists),
    append(EscapedLists, EscChars),
    string_chars(EscBody, EscChars),
    format(string(Quoted), '"~w"', [EscBody]).

lua_string_escape_char('\\', ['\\', '\\']) :- !.
lua_string_escape_char('"', ['\\', '"']) :- !.
lua_string_escape_char('\n', ['\\', 'n']) :- !.
lua_string_escape_char('\t', ['\\', 't']) :- !.
lua_string_escape_char(C, [C]).

text_to_string(Value, Str) :-
    (   string(Value)
    ->  Str = Value
    ;   atom(Value)
    ->  atom_string(Value, Str)
    ;   term_string(Value, Str)
    ).

% ============================================================================
% Program assembly
% ============================================================================

wam_code_to_lua_data(WamCode, Options, Instructions, LabelEntries) :-
    atom_string(WamCode, Str),
    split_string(Str, "\n", "", Lines),
    wam_lines_to_data(Lines, Options, 1, Instructions, LabelEntries).

wam_lines_to_data([], _, _, [], []).
wam_lines_to_data([Line|Rest], Options, PC, Instructions, LabelEntries) :-
    tokenize_wam_line(Line, Parts),
    (   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        lua_string_literal(LabelName, L),
        format(string(LEntry), '  [~w] = ~w', [L, PC]),
        LabelEntries = [LEntry|LE2],
        wam_lines_to_data(Rest, Options, PC, Instructions, LE2)
    ;   Parts = []
    ->  wam_lines_to_data(Rest, Options, PC, Instructions, LabelEntries)
    ;   wam_parts_to_lua(Parts, Options, Lit),
        PC1 is PC + 1,
        Instructions = [Lit|I2],
        wam_lines_to_data(Rest, Options, PC1, I2, LabelEntries)
    ).

compile_wam_predicate_to_lua(_Pred, _WamCode, _Options, "").

compile_predicates_for_project(Predicates, Options,
                               AllInstrs, TopLabels, AllLabels,
                               WrapperCode, LoweredCode) :-
    init_lua_atom_intern_table,
    option(intern_atoms(ExtraAtoms), Options, []),
    forall(member(A, ExtraAtoms), (atom_string(A, S), intern_lua_atom(S, _))),
    option(foreign_predicates(ForeignPredicates), Options, []),
    append_missing_foreign_predicates(Predicates, ForeignPredicates, CompilePreds),
    wam_lua_resolve_emit_mode(Options, Mode),
    compile_all_predicates(CompilePreds, Options, Mode, 1,
        [], [], [], [], [],
        AllInstrs, TopLabels, AllLabels, Wrappers, LoweredEntries),
    atomic_list_concat(Wrappers, '\n', WrapperCode),
    atomic_list_concat(LoweredEntries, '\n', LoweredCode).

append_missing_foreign_predicates(Predicates, ForeignPredicates, CompilePredicates) :-
    findall(F, (member(F, ForeignPredicates), \+ (member(P, Predicates), same_pi(P, F))), Missing),
    append(Predicates, Missing, CompilePredicates).

same_pi(P0, P1) :- pi_key(P0, K), pi_key(P1, K).
pi_key(_:P/A, P/A) :- !.
pi_key(P/A, P/A).

compile_all_predicates([], _, _, _, Instrs, TopLabels, AllLabels, Wrappers, Lowered,
                       Instrs, TopLabels, AllLabels, Wrappers, Lowered).
compile_all_predicates([Pred|Rest], Options, Mode, BasePC,
                       InstrAcc, TopLabelAcc, AllLabelAcc, WrapperAcc, LoweredAcc,
                       AllInstrs, TopLabels, AllLabels, Wrappers, Lowered) :-
    (Pred = _M:P/Arity -> true ; Pred = P/Arity),
    (   lua_foreign_predicate(P, Arity, Options)
    ->  lua_string_literal(P, PQ),
        format(string(FLit), 'I.CallForeign(~w, ~w)', [PQ, Arity]),
        PredInstrs = [FLit, 'I.Proceed()'],
        WamForLower = ""
    ;   compile_predicate_to_wam(P/Arity, [], WamForLower),
        wam_code_to_lua_data(WamForLower, Options, PredInstrs, PredSubLabelEntries0)
    ),
    length(PredInstrs, PredLen),
    append(InstrAcc, PredInstrs, NewInstrs),
    NewPC is BasePC + PredLen,
    Offset is BasePC - 1,
    (   lua_foreign_predicate(P, Arity, Options)
    ->  PredSubLabelEntries = []
    ;   maplist(offset_label_entry(Offset), PredSubLabelEntries0, PredSubLabelEntries1),
        format(string(MainKey), '~w/~w', [P, Arity]),
        exclude(is_pred_label(MainKey), PredSubLabelEntries1, PredSubLabelEntries)
    ),
    format(string(Key), '~w/~w', [P, Arity]),
    lua_string_literal(Key, KeyQ),
    format(string(MainEntry), '  [~w] = ~w', [KeyQ, BasePC]),
    NewTopLabels = [MainEntry|TopLabelAcc],
    append([MainEntry|PredSubLabelEntries], AllLabelAcc, NewAllLabels),
    (   should_try_lower(Mode, P, Arity),
        WamForLower \= "",
        catch(wam_lua_lowerable(Pred, WamForLower, _), _, fail),
        catch(lower_predicate_to_lua(Pred, WamForLower, [],
                                     lowered(_, FuncName, LoweredLua)), _, fail)
    ->  NewLoweredAcc = [LoweredLua|LoweredAcc],
        emit_lua_lowered_wrapper(P, Arity, FuncName, Wrapper)
    ;   NewLoweredAcc = LoweredAcc,
        emit_lua_wrapper(P, Arity, BasePC, Wrapper)
    ),
    compile_all_predicates(Rest, Options, Mode, NewPC,
        NewInstrs, NewTopLabels, NewAllLabels, [Wrapper|WrapperAcc], NewLoweredAcc,
        AllInstrs, TopLabels, AllLabels, Wrappers, Lowered).

offset_label_entry(Offset, Entry0, Entry) :-
    atom_string(Entry0, S),
    (   sub_string(S, B, 3, _, "] =")
    ->  B1 is B + 4,
        sub_string(S, 0, B1, _, Prefix),
        sub_string(S, B1, _, 0, PCStr),
        number_string(PC0, PCStr),
        PC is PC0 + Offset,
        format(string(Entry), '~w~w', [Prefix, PC])
    ;   Entry = Entry0
    ).

is_pred_label(PredKey, Entry) :-
    atom_string(Entry, S),
    sub_string(S, _, _, _, PredKey).

emit_lua_wrapper(Pred, Arity, StartPc, Code) :-
    pred_arg_strings(Arity, ArgDecl, ArgList),
    lua_pred_name(Pred, Name),
    format(string(Code),
'function M.~w(~w)
  return Runtime.run_predicate(shared_program, ~w, ~w)
end
', [Name, ArgDecl, StartPc, ArgList]).

emit_lua_lowered_wrapper(Pred, Arity, FuncName, Code) :-
    pred_arg_strings(Arity, ArgDecl, ArgList),
    lua_pred_name(Pred, Name),
    format(string(Code),
'function M.~w(~w)
  local state = Runtime.new_state()
  local args = ~w
  for i, arg in ipairs(args) do Runtime.put_reg(state, i, arg) end
  state.cp = 0
  return ~w(shared_program, state) == true
end
', [Name, ArgDecl, ArgList, FuncName]).

pred_arg_strings(0, '', '{}') :- !.
pred_arg_strings(Arity, ArgDecl, ArgList) :-
    numlist(1, Arity, Ns),
    maplist([N, A]>>(format(string(A), 'a~w', [N])), Ns, Args),
    atomic_list_concat(Args, ', ', ArgDecl),
    format(string(ArgList), '{~w}', [ArgDecl]).

lua_pred_name(Pred, Name) :-
    atom_string(Pred, S),
    string_codes(S, Codes),
    maplist(lua_safe_code, Codes, Safe),
    string_codes(SafeS, Safe),
    (   Safe = [C|_], C >= 0'0, C =< 0'9
    ->  string_concat("p_", SafeS, Out)
    ;   Out = SafeS
    ),
    atom_string(Name, Out).

lua_safe_code(C, C) :-
    (C >= 0'a, C =< 0'z ; C >= 0'A, C =< 0'Z ; C >= 0'0, C =< 0'9 ; C =:= 0'_), !.
lua_safe_code(_, 0'_).

lua_foreign_predicate(Pred, Arity, Options) :-
    (string(Pred) -> atom_string(PA, Pred) ; PA = Pred),
    option(foreign_predicates(FPs), Options, []),
    (member(PA/Arity, FPs) ; member(_:PA/Arity, FPs)), !.

lua_foreign_handlers_code(Options, Code) :-
    option(lua_foreign_handlers(Handlers), Options, []),
    maplist(lua_foreign_handler_entry, Handlers, Entries),
    atomic_list_concat(Entries, ',\n', Code).

lua_foreign_handler_entry(handler(Pred/Arity, HandlerCode), Entry) :-
    format(string(Key), '~w/~w', [Pred, Arity]),
    lua_string_literal(Key, Q),
    format(string(Entry), '  [~w] = ~w', [Q, HandlerCode]).

write_wam_lua_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'lua', LuaDir),
    make_directory_path(LuaDir),
    compile_predicates_for_project(Predicates, Options,
        AllInstrs, TopLabels, AllLabels, WrapperCode, LoweredCode),
    emit_lua_intern_table(InternSeed),
    maplist([I, Line]>>(format(string(Line), '  ~w', [I])), AllInstrs, InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    atomic_list_concat(TopLabels, ',\n', DispatchBody),
    atomic_list_concat(AllLabels, ',\n', LabelBody),
    lua_foreign_handlers_code(Options, ForeignHandlers),
    write_runtime_source(LuaDir),
    write_program_source(LuaDir, InstrBody, LabelBody, DispatchBody,
                         WrapperCode, InternSeed, ForeignHandlers, LoweredCode).

write_runtime_source(LuaDir) :-
    find_template('templates/targets/lua_wam/runtime.lua.mustache', Template),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(Template, ['date'=Date], Content),
    directory_file_path(LuaDir, 'wam_runtime.lua', Path),
    write_file(Path, Content).

write_program_source(LuaDir, InstrBody, LabelBody, DispatchBody,
                     WrapperCode, InternSeed, ForeignHandlers, LoweredCode) :-
    find_template('templates/targets/lua_wam/program.lua.mustache', Template),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(Template,
        ['date'=Date,
         'instructions'=InstrBody,
         'labels'=LabelBody,
         'dispatch'=DispatchBody,
         'wrappers'=WrapperCode,
         'intern_id_to_string'=InternSeed,
         'foreign_handlers'=ForeignHandlers,
         'lowered_functions'=LoweredCode], Content),
    directory_file_path(LuaDir, 'generated_program.lua', Path),
    write_file(Path, Content).

write_file(Path, Content) :-
    setup_call_cleanup(open(Path, write, Stream), write(Stream, Content), close(Stream)).

find_template(RelPath, Template) :-
    (   source_file(wam_lua_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, Root),
        atomic_list_concat([Root, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, []).

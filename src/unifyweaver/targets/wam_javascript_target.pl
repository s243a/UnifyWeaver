:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_javascript_target.pl - WAM-to-JavaScript hybrid target
%
% Consumes shared WAM bytecode from wam_target.pl and emits a Node project
% with an instruction-array WAM interpreter plus an optional Tier-2
% lowered-function path. Architecture mirrors wam_lua_target.pl
% (closest dynamically typed model). emit_mode is interpreter | functions
% | mixed(List); default remains interpreter.
% javascript_wam_fact_sources([source(P/2, file(Path))]) streams binary
% facts from a TSV/CSV or JSONL file (Lua-style; no LMDB/CSR).
% javascript_wam_ops([op(Prec, Type, Name), ...]) (alias js_op_decls/1)
% seeds the runtime Pratt op table at program startup (R's r_op_decls/1).

:- module(wam_javascript_target, [
    write_wam_javascript_project/3,
    compile_wam_predicate_to_javascript/4,
    init_js_atom_intern_table/0,
    intern_js_atom/2,
    parse_functor_arity/3,
    reg_to_int/2,
    wam_parts_to_js/3,
    wam_javascript_resolve_emit_mode/2,
    javascript_wam_resolve_emit_mode/2
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [
    compile_predicate_to_wam_text/3,
    compile_predicate_to_wam_items/3
]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module(wam_text_parser, [
    wam_tokenize_line/2,
    wam_recognise_label/2,
    wam_recognise_instruction/2,
    wam_classify_constant_token/2,
    wam_constant_token_is_string/1
]).
:- use_module(wam_javascript_lowered_emitter, [
    wam_javascript_lowerable/3,
    lower_predicate_to_javascript/4
]).

:- multifile user:wam_javascript_emit_mode/1.

%% javascript_wam_resolve_emit_mode(+Options, -Mode)
%  Public emit-mode resolver (Lua/Haskell naming). Alias of
%  wam_javascript_resolve_emit_mode/2.
javascript_wam_resolve_emit_mode(Options, Mode) :-
    wam_javascript_resolve_emit_mode(Options, Mode).

wam_javascript_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_emit_mode(M0, Mode)
    ;   catch(user:wam_javascript_emit_mode(M1), _, fail)
    ->  validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_emit_mode(interpreter, interpreter) :- !.
validate_emit_mode(functions, functions) :- !.
validate_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_javascript_emit_mode, Other),
                wam_javascript_resolve_emit_mode/2)).

should_try_lower(functions, _, _) :- !.
should_try_lower(mixed(HotPreds), P, A) :-
    member(P/A, HotPreds), !.
should_try_lower(_, _, _) :- fail.

% ============================================================================
% Atom interning
% ============================================================================

:- dynamic js_atom_intern_id/2.
:- dynamic js_atom_intern_next/1.

init_js_atom_intern_table :-
    retractall(js_atom_intern_id(_, _)),
    retractall(js_atom_intern_next(_)),
    assertz(js_atom_intern_id("true", 0)),
    assertz(js_atom_intern_id("fail", 1)),
    assertz(js_atom_intern_id("[]", 2)),
    assertz(js_atom_intern_id(".", 3)),
    assertz(js_atom_intern_id("", 4)),
    assertz(js_atom_intern_id("[|]", 5)),
    assertz(js_atom_intern_next(6)).

intern_js_atom(AtomStr, Id) :-
    (   js_atom_intern_next(_)
    ->  true
    ;   init_js_atom_intern_table
    ),
    text_to_string(AtomStr, Str),
    (   js_atom_intern_id(Str, Id0)
    ->  Id = Id0
    ;   retract(js_atom_intern_next(Next)),
        Id = Next,
        Next1 is Next + 1,
        assertz(js_atom_intern_id(Str, Id)),
        assertz(js_atom_intern_next(Next1))
    ).

emit_js_intern_table(Code) :-
    findall(Id-Str, js_atom_intern_id(Str, Id), Pairs),
    sort(Pairs, Sorted),
    maplist([_Id-Str, E]>>(
        js_string_literal(Str, Lit),
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
    wam_tokenize_line(Line, Tokens).

strip_operand_comma(Token0, Token) :-
    sub_string(Token0, _, 1, 0, ","), !,
    sub_string(Token0, 0, _, 1, Token).
strip_operand_comma(Token, Token).

wam_parts_to_js(Parts, _Options, Lit) :-
    wam_parts_to_js(Parts, Lit).

wam_parts_to_js(["call", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    js_string_literal(PredName, P),
    format(string(Lit), 'I.Call(~w, ~w)', [P, Arity]).
wam_parts_to_js(["call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    js_string_literal(PredName, P),
    format(string(Lit), 'I.Call(~w, ~w)', [P, Arity]).
wam_parts_to_js(["execute", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    js_string_literal(PredName, P),
    format(string(Lit), 'I.Execute(~w, ~w)', [P, Arity]).
wam_parts_to_js(["execute", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    js_string_literal(PredName, P),
    format(string(Lit), 'I.Execute(~w, ~w)', [P, Arity]).
wam_parts_to_js(["proceed"], 'I.Proceed()').
wam_parts_to_js(["fail"], 'I.Fail()').
wam_parts_to_js(["jump", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.Jump(~w)', [L]).
wam_parts_to_js(["try_me_else", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.TryMeElse(~w)', [L]).
wam_parts_to_js(["retry_me_else", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.RetryMeElse(~w)', [L]).
wam_parts_to_js(["trust_me"], 'I.TrustMe()').
wam_parts_to_js(["try", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.Try(~w)', [L]).
wam_parts_to_js(["retry", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.Retry(~w)', [L]).
wam_parts_to_js(["trust", Label], Lit) :-
    js_string_literal(Label, L),
    format(string(Lit), 'I.Trust(~w)', [L]).
wam_parts_to_js(["allocate"], 'I.Allocate()').
wam_parts_to_js(["deallocate"], 'I.Deallocate()').
wam_parts_to_js(["get_constant", C, Reg], Lit) :-
    reg_to_int(Reg, R), constant_to_js_term(C, T),
    format(string(Lit), 'I.GetConstant(~w, ~w)', [T, R]).
wam_parts_to_js(["get_variable", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.GetVariable(~w, ~w)', [XI, AI]).
wam_parts_to_js(["get_value", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.GetValue(~w, ~w)', [XI, AI]).
wam_parts_to_js(["put_constant", C, Reg], Lit) :-
    reg_to_int(Reg, R), constant_to_js_term(C, T),
    format(string(Lit), 'I.PutConstant(~w, ~w)', [T, R]).
wam_parts_to_js(["put_variable", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.PutVariable(~w, ~w)', [XI, AI]).
wam_parts_to_js(["put_value", X, A], Lit) :-
    reg_to_int(X, XI), reg_to_int(A, AI),
    format(string(Lit), 'I.PutValue(~w, ~w)', [XI, AI]).
wam_parts_to_js(["put_structure", F, Reg], Lit) :-
    reg_to_int(Reg, R), parse_functor_arity(F, Name, Arity),
    intern_js_atom(Name, Id),
    format(string(Lit), 'I.PutStructure(~w, ~w, ~w)', [Id, R, Arity]).
wam_parts_to_js(["get_structure", F, Reg], Lit) :-
    reg_to_int(Reg, R), parse_functor_arity(F, Name, Arity),
    intern_js_atom(Name, Id),
    format(string(Lit), 'I.GetStructure(~w, ~w, ~w)', [Id, R, Arity]).
wam_parts_to_js(["put_list", Reg], Lit) :-
    reg_to_int(Reg, R), intern_js_atom("[|]", Id),
    format(string(Lit), 'I.PutList(~w, ~w)', [R, Id]).
wam_parts_to_js(["get_list", Reg], Lit) :-
    reg_to_int(Reg, R), intern_js_atom("[|]", Id),
    format(string(Lit), 'I.GetList(~w, ~w)', [R, Id]).
wam_parts_to_js(["get_nil", Reg], Lit) :-
    intern_js_atom("[]", Id), reg_to_int(Reg, R),
    format(string(Lit), 'I.GetConstant(V.Atom(~w), ~w)', [Id, R]).
wam_parts_to_js(["get_integer", NStr, Reg], Lit) :-
    (number_string(N, NStr) -> true ; N = NStr),
    reg_to_int(Reg, R),
    format(string(Lit), 'I.GetConstant(V.Int(~w), ~w)', [N, R]).
wam_parts_to_js(["set_variable", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.SetVariable(~w)', [XI]).
wam_parts_to_js(["set_value", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.SetValue(~w)', [XI]).
wam_parts_to_js(["set_constant", C], Lit) :-
    constant_to_js_term(C, T), format(string(Lit), 'I.SetConstant(~w)', [T]).
wam_parts_to_js(["unify_variable", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.UnifyVariable(~w)', [XI]).
wam_parts_to_js(["unify_value", X], Lit) :-
    reg_to_int(X, XI), format(string(Lit), 'I.UnifyValue(~w)', [XI]).
wam_parts_to_js(["unify_constant", C], Lit) :-
    constant_to_js_term(C, T), format(string(Lit), 'I.UnifyConstant(~w)', [T]).
wam_parts_to_js(["builtin_call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr), js_string_literal(Pred, P),
    format(string(Lit), 'I.BuiltinCall(~w, ~w)', [P, Arity]).
wam_parts_to_js(["call_foreign", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr), js_string_literal(Pred, P),
    format(string(Lit), 'I.CallForeign(~w, ~w)', [P, Arity]).
wam_parts_to_js(["call_indexed_atom_fact2", Pred], Lit) :-
    strip_operand_comma(Pred, CleanPred),
    js_string_literal(CleanPred, P),
    format(string(Lit), 'I.CallIndexedAtomFact2(~w)', [P]).
wam_parts_to_js(["arg", NStr, RegStr, OutRegStr], Lit) :-
    number_string(N, NStr), reg_to_int(RegStr, R), reg_to_int(OutRegStr, O),
    format(string(Lit), 'I.ArgInstr(~w, ~w, ~w)', [N, R, O]).
wam_parts_to_js(["switch_on_constant" | Cases], Lit) :-
    emit_js_switch_constant(Cases, false, 1, Lit).
wam_parts_to_js(["switch_on_constant_fallthrough" | Cases], Lit) :-
    emit_js_switch_constant(Cases, true, 1, Lit).
wam_parts_to_js(["switch_on_constant_a2" | Cases], Lit) :-
    emit_js_switch_constant(Cases, false, 2, Lit).
wam_parts_to_js(["switch_on_constant_a2_fallthrough" | Cases], Lit) :-
    emit_js_switch_constant(Cases, true, 2, Lit).
wam_parts_to_js(["switch_on_structure" | Cases], Lit) :-
    emit_js_switch_structure(Cases, 1, Lit).
wam_parts_to_js(["switch_on_structure_a2" | Cases], Lit) :-
    emit_js_switch_structure(Cases, 2, Lit).
wam_parts_to_js(["switch_on_term" | Tokens], Lit) :-
    emit_js_switch_term(Tokens, 1, Lit).
wam_parts_to_js(["switch_on_term_a2" | Tokens], Lit) :-
    emit_js_switch_term(Tokens, 2, Lit).
wam_parts_to_js(["cut_ite"], 'I.CutIte()').
wam_parts_to_js(["get_level", Yn], Lit) :-
    reg_to_int(Yn, R),
    format(string(Lit), 'I.GetLevel(~w)', [R]).
wam_parts_to_js(["cut", Yn], Lit) :-
    reg_to_int(Yn, R),
    format(string(Lit), 'I.Cut(~w)', [R]).
wam_parts_to_js(["begin_aggregate", Kind, TemplateReg, BagReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    reg_to_int(BagReg, BIdx),
    js_string_literal(Kind, K),
    format(string(Lit), 'I.BeginAggregate(~w, ~w, ~w, [])', [K, TIdx, BIdx]).
wam_parts_to_js(["begin_aggregate", Kind, TemplateReg, BagReg, Witness], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    reg_to_int(BagReg, BIdx),
    parse_witness_regs(Witness, WRegs),
    atomic_list_concat(WRegs, ', ', WStr),
    js_string_literal(Kind, K),
    format(string(Lit), 'I.BeginAggregate(~w, ~w, ~w, [~w])', [K, TIdx, BIdx, WStr]).
wam_parts_to_js(["end_aggregate", TemplateReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    format(string(Lit), 'I.EndAggregate(~w)', [TIdx]).
% Convention 6: unrecognised WAM text becomes a real one-slot NoOp
% (I.Raw), never dropped, so later label PCs stay aligned.
wam_parts_to_js(Parts, Lit) :-
    atomic_list_concat(Parts, ' ', Text),
    js_string_literal(Text, Q),
    format(string(Lit), 'I.Raw(~w)', [Q]).

emit_js_switch_constant(Cases, Fallthrough, Reg, Lit) :-
    normalize_switch_case_tokens(Cases, Norm),
    parse_switch_cases(Norm, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    (   Fallthrough == true -> FT = true ; FT = false ),
    (   Reg =:= 2
    ->  format(string(Lit), 'I.SwitchOnConstantA2([~w], ~w)', [CasesStr, FT])
    ;   format(string(Lit), 'I.SwitchOnConstant([~w], ~w)', [CasesStr, FT])
    ).

emit_js_switch_structure(Cases, Reg, Lit) :-
    normalize_switch_case_tokens(Cases, Norm),
    parse_struct_switch_cases(Norm, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    (   Reg =:= 2
    ->  format(string(Lit), 'I.SwitchOnStructureA2([~w])', [CasesStr])
    ;   format(string(Lit), 'I.SwitchOnStructure([~w])', [CasesStr])
    ).

emit_js_switch_term(Tokens, Reg, Lit) :-
    parse_switch_term_tokens(Tokens, ConstLits, StructLits, ListLabel),
    atomic_list_concat(ConstLits, ', ', ConstStr),
    atomic_list_concat(StructLits, ', ', StructStr),
    js_string_literal(ListLabel, LQ),
    format(string(Lit), 'I.SwitchOnTerm([~w], [~w], ~w, ~w)',
           [ConstStr, StructStr, LQ, Reg]).

parse_switch_term_tokens(Tokens, ConstLits, StructLits, ListLabel) :-
    Tokens = [NCStr|Rest1],
    number_string(NC, NCStr),
    length(ConstToks, NC),
    append(ConstToks, Rest2, Rest1),
    Rest2 = [NSStr|Rest3],
    number_string(NS, NSStr),
    length(StructToks, NS),
    append(StructToks, [ListLabel0], Rest3),
    text_to_string(ListLabel0, ListLabel),
    normalize_switch_case_tokens(ConstToks, ConstNorm),
    parse_switch_cases(ConstNorm, ConstLits),
    normalize_switch_case_tokens(StructToks, StructNorm),
    parse_struct_switch_cases(StructNorm, StructLits).

parse_switch_cases([], []).
parse_switch_cases([Token|Rest], [Lit|More]) :-
    split_at_first_colon(Token, ValStr, LabelStr),
    constant_to_js_term(ValStr, ValLit),
    js_string_literal(LabelStr, L),
    format(string(Lit), '{value: ~w, label: ~w}', [ValLit, L]),
    parse_switch_cases(Rest, More).

parse_struct_switch_cases([], []).
parse_struct_switch_cases([Token|Rest], [Lit|More]) :-
    split_at_first_colon(Token, FAStr, LabelStr),
    parse_functor_arity(FAStr, FName, FArity),
    intern_js_atom(FName, FId),
    js_string_literal(LabelStr, L),
    format(string(Lit), '{fid: ~w, arity: ~w, label: ~w}', [FId, FArity, L]),
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

constant_to_js_term(C, Lit) :-
    (   wam_constant_token_is_string(C)
    ->  wam_classify_constant_token(C, atom(Name)),
        js_string_literal(Name, SLit),
        format(string(Lit), 'V.String(~w)', [SLit])
    ;   wam_classify_constant_token(C, Class),
        (   Class = integer(N)
        ->  format(string(Lit), 'V.Int(~w)', [N])
        ;   Class = float(F)
        ->  format(string(Lit), 'V.Float(~w)', [F])
        ;   Class = atom(Name),
            intern_js_atom(Name, Id),
            format(string(Lit), 'V.Atom(~w)', [Id])
        )
    ).

% Convention 2: arity is the trailing /<digits> segment so names that
% contain `/` (///2, //2, =../2) parse correctly.
parse_functor_arity(FStr, Name, Arity) :-
    atom_string(FA, FStr),
    (   last_slash_index(FA, B),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ->  sub_atom(FA, 0, B, _, Name)
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
    (   last_slash_index_str(Pred, B),
        B1 is B + 1,
        sub_string(Pred, B1, _, 0, AS),
        number_string(_, AS)
    ->  sub_string(Pred, 0, B, _, Name)
    ;   Name = Pred
    ).

last_slash_index_str(Str, Index) :-
    findall(B, sub_string(Str, B, 1, _, "/"), Bs),
    Bs \= [],
    last(Bs, Index).

% 4th begin_aggregate operand is "'Y1;Y2'" (ISO bagof/setof witnesses).
parse_witness_regs(Raw, Regs) :-
    text_to_string(Raw, S0),
    strip_witness_quotes(S0, S),
    (   S == ""
    ->  Regs = []
    ;   split_string(S, ";", " \t", Parts0),
        exclude([P]>>(P == ""), Parts0, Parts),
        maplist(reg_to_int, Parts, Regs)
    ).

strip_witness_quotes(S0, S) :-
    (   string_concat("'", Rest, S0),
        string_concat(Mid, "'", Rest)
    ->  S = Mid
    ;   S = S0
    ).

js_string_literal(Raw, Quoted) :-
    text_to_string(Raw, S),
    string_chars(S, Chars),
    maplist(js_string_escape_char, Chars, EscapedLists),
    append(EscapedLists, EscChars),
    string_chars(EscBody, EscChars),
    format(string(Quoted), '"~w"', [EscBody]).

js_string_escape_char('\\', ['\\', '\\']) :- !.
js_string_escape_char('"', ['\\', '"']) :- !.
js_string_escape_char('\n', ['\\', 'n']) :- !.
js_string_escape_char('\t', ['\\', 't']) :- !.
js_string_escape_char(C, [C]).

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

wam_code_to_js_data(WamCode, Options, Instructions, LabelEntries) :-
    wam_code_to_js_items(WamCode, Items),
    wam_items_to_data(Items, Options, 1, Instructions, LabelEntries).

wam_code_to_js_items(WamCode, Items) :-
    is_list(WamCode), !,
    Items = WamCode.
wam_code_to_js_items(WamCode, Items) :-
    js_wam_text_to_items(WamCode, Items).

js_wam_text_to_items(WamText, Items) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    js_wam_lines_to_items(Lines, Items).

js_wam_lines_to_items([], []).
js_wam_lines_to_items([Line|Rest], Items) :-
    tokenize_wam_line(Line, Tokens),
    (   Tokens == []
    ->  js_wam_lines_to_items(Rest, Items)
    ;   wam_recognise_label(Tokens, Name)
    ->  Items = [label(Name)|More],
        js_wam_lines_to_items(Rest, More)
    ;   wam_recognise_instruction(Tokens, Item)
    ->  Items = [Item|More],
        js_wam_lines_to_items(Rest, More)
    ;   js_recognise_instruction(Tokens, Item)
    ->  Items = [Item|More],
        js_wam_lines_to_items(Rest, More)
    ;   js_wam_lines_to_items(Rest, Items)
    ).

js_recognise_instruction(["arg", N, Reg, OutReg], arg(N, Reg, OutReg)).
js_recognise_instruction(["call_indexed_atom_fact2", Pred], call_indexed_atom_fact2(Pred)).

wam_items_to_data([], _, _, [], []).
wam_items_to_data([label(LabelName)|Rest], Options, PC, Instructions, LabelEntries) :- !,
        js_string_literal(LabelName, L),
        format(string(LEntry), '  ~w: ~w', [L, PC]),
        LabelEntries = [LEntry|LE2],
        wam_items_to_data(Rest, Options, PC, Instructions, LE2).
wam_items_to_data([Item|Rest], Options, PC, [Lit|I2], LabelEntries) :-
    once(wam_item_parts(Item, Parts)),
    once(wam_parts_to_js(Parts, Options, Lit)),
    PC1 is PC + 1,
    wam_items_to_data(Rest, Options, PC1, I2, LabelEntries).

compile_wam_predicate_to_javascript(_Pred, _WamCode, _Options, "").

compile_predicates_for_project(Predicates, Options, AllInstrs, TopLabels, AllLabels, WrapperCode, LoweredCode, FactSourcesCode) :-
    init_js_atom_intern_table,
    option(intern_atoms(ExtraAtoms), Options, []),
    forall(member(A, ExtraAtoms), (atom_string(A, S), intern_js_atom(S, _))),
    wam_javascript_resolve_emit_mode(Options, EmitMode),
    compile_all_predicates(Predicates, Options, EmitMode, 1,
        [], [], [], [], [], [],
        AllInstrs, TopLabels, AllLabels, Wrappers, LoweredEntries, FactSources),
    atomic_list_concat(Wrappers, '\n', WrapperCode),
    atomic_list_concat(LoweredEntries, '\n', LoweredCode),
    atomic_list_concat(FactSources, ',\n', FactSourcesCode).

compile_all_predicates([], _, _, _, Instrs, TopLabels, AllLabels, Wrappers, Lowered, FactSources,
                       Instrs, TopLabels, AllLabels, Wrappers, Lowered, FactSources).
compile_all_predicates([Pred|Rest], Options, EmitMode, BasePC,
                       InstrAcc, TopLabelAcc, AllLabelAcc, WrapperAcc, LoweredAcc, FactSourceAcc,
                       AllInstrs, TopLabels, AllLabels, Wrappers, Lowered, FactSources) :-
    (Pred = _M:P/Arity -> true ; Pred = P/Arity),
    (   javascript_wam_fact_source_spec(P, Arity, Options, SourceSpec)
    ->  format(string(MainKey), '~w/~w', [P, Arity]),
        js_string_literal(MainKey, KeyQ),
        format(string(FLit), 'I.CallFactStream(~w, ~w)', [KeyQ, Arity]),
        PredInstrs = [FLit, 'I.Proceed()'],
        PredSubLabelEntries0 = [],
        javascript_wam_fact_source_entry(MainKey, SourceSpec, FactSourceEntry),
        SkipLower = true
    ;   compile_js_predicate_wam(P/Arity, WamCode),
        wam_code_to_js_data(WamCode, Options, PredInstrs, PredSubLabelEntries0),
        FactSourceEntry = none,
        SkipLower = false
    ),
    length(PredInstrs, PredLen),
    append(InstrAcc, PredInstrs, NewInstrs),
    NewPC is BasePC + PredLen,
    Offset is BasePC - 1,
    maplist(offset_label_entry(Offset), PredSubLabelEntries0, PredSubLabelEntries1),
    format(string(MainKey), '~w/~w', [P, Arity]),
    exclude(is_pred_label(MainKey), PredSubLabelEntries1, PredSubLabelEntries),
    js_string_literal(MainKey, KeyQ),
    format(string(MainEntry), '  ~w: ~w', [KeyQ, BasePC]),
    NewTopLabels = [MainEntry|TopLabelAcc],
    append([MainEntry|PredSubLabelEntries], AllLabelAcc, NewAllLabels),
    (   SkipLower \== true,
        should_try_lower(EmitMode, P, Arity),
        compile_js_predicate_wam_text(P/Arity, WamText),
        WamText \= "",
        catch(wam_javascript_lowerable(Pred, WamText, _), _, fail),
        catch(lower_predicate_to_javascript(Pred, WamText, [],
                                           lowered(_, FuncName, LoweredJs)), _, fail)
    ->  format(string(DispatchLine), 'lowered_dispatch[~w] = ~w;', [KeyQ, FuncName]),
        NewLoweredAcc = [LoweredJs, DispatchLine|LoweredAcc],
        emit_js_lowered_wrapper(P, Arity, FuncName, Wrapper)
    ;   NewLoweredAcc = LoweredAcc,
        emit_js_wrapper(P, Arity, BasePC, Wrapper)
    ),
    (FactSourceEntry == none -> NewFactSourceAcc = FactSourceAcc ; NewFactSourceAcc = [FactSourceEntry|FactSourceAcc]),
    compile_all_predicates(Rest, Options, EmitMode, NewPC,
        NewInstrs, NewTopLabels, NewAllLabels, [Wrapper|WrapperAcc], NewLoweredAcc, NewFactSourceAcc,
        AllInstrs, TopLabels, AllLabels, Wrappers, Lowered, FactSources).

%% javascript_wam_fact_sources([source(P/A, file(Path)), ...])
%  Lightweight file-backed binary facts (Lua's lua_fact_sources/1).
%  Only P/2 is streamed; other arities keep compiled inline WAM.
javascript_wam_fact_source_spec(P, Arity, Options, Spec) :-
    Arity =:= 2,
    (   option(javascript_wam_fact_sources(Sources), Options)
    ->  true
    ;   option(js_fact_sources(Sources), Options)
    ->  true
    ;   Sources = []
    ),
    member(source(PI, Spec), Sources),
    javascript_wam_fact_source_pi_match(PI, P, Arity).

javascript_wam_fact_source_pi_match(_:Name/Ar, P, Arity) :- !,
    Name == P, Ar =:= Arity.
javascript_wam_fact_source_pi_match(Name/Ar, P, Arity) :-
    Name == P, Ar =:= Arity.

javascript_wam_fact_source_entry(Key, file(Path), Entry) :-
    atom_string(Path, PathStr),
    (   absolute_file_name(PathStr, AbsPath, [access(read), file_errors(fail)])
    ->  SourcePath = AbsPath
    ;   SourcePath = PathStr
    ),
    js_string_literal(Key, KeyQ),
    js_string_literal(SourcePath, PathQ),
    format(string(Entry), '  ~w: { path: ~w }', [KeyQ, PathQ]).

compile_js_predicate_wam(PredIndicator, WamCode) :-
    CompileOpts = [ite_use_y_level(true), inline_bagof_setof(true)],
    (   catch(compile_predicate_to_wam_items(PredIndicator, CompileOpts, Items), _, fail),
        is_list(Items), Items \== []
    ->  WamCode = Items
    ;   compile_predicate_to_wam_text(PredIndicator, CompileOpts, WamCode)
    ).

compile_js_predicate_wam_text(PredIndicator, WamText) :-
    CompileOpts = [ite_use_y_level(true), inline_bagof_setof(true)],
    compile_predicate_to_wam_text(PredIndicator, CompileOpts, WamText).

wam_item_parts(get_constant(C, Ai), ["get_constant", C, Ai]).
wam_item_parts(get_variable(Xn, Ai), ["get_variable", Xn, Ai]).
wam_item_parts(get_value(Xn, Ai), ["get_value", Xn, Ai]).
wam_item_parts(get_structure(F, Ai), ["get_structure", F, Ai]).
wam_item_parts(get_list(Ai), ["get_list", Ai]).
wam_item_parts(get_nil(Ai), ["get_nil", Ai]).
wam_item_parts(get_integer(N, Ai), ["get_integer", N, Ai]).
wam_item_parts(unify_variable(Xn), ["unify_variable", Xn]).
wam_item_parts(unify_value(Xn), ["unify_value", Xn]).
wam_item_parts(unify_constant(C), ["unify_constant", C]).
wam_item_parts(put_variable(Xn, Ai), ["put_variable", Xn, Ai]).
wam_item_parts(put_value(Xn, Ai), ["put_value", Xn, Ai]).
wam_item_parts(put_constant(C, Ai), ["put_constant", C, Ai]).
wam_item_parts(put_structure(F, Ai), ["put_structure", F, Ai]).
wam_item_parts(put_list(Ai), ["put_list", Ai]).
wam_item_parts(set_variable(Xn), ["set_variable", Xn]).
wam_item_parts(set_value(Xn), ["set_value", Xn]).
wam_item_parts(set_constant(C), ["set_constant", C]).
wam_item_parts(call(P, N), ["call", P, N]).
wam_item_parts(execute(P), ["execute", P]).
wam_item_parts(proceed, ["proceed"]).
wam_item_parts(fail, ["fail"]).
wam_item_parts(allocate, ["allocate"]).
wam_item_parts(deallocate, ["deallocate"]).
wam_item_parts(builtin_call(Op, Ar), ["builtin_call", Op, Ar]).
wam_item_parts(call_foreign(Pred, Ar), ["call_foreign", Pred, Ar]).
wam_item_parts(arg(N, Reg, OutReg), ["arg", N, Reg, OutReg]).
wam_item_parts(call_indexed_atom_fact2(Pred), ["call_indexed_atom_fact2", Pred]).
wam_item_parts(try_me_else(L), ["try_me_else", L]).
wam_item_parts(retry_me_else(L), ["retry_me_else", L]).
wam_item_parts(trust_me, ["trust_me"]).
wam_item_parts(try(L), ["try", L]).
wam_item_parts(retry(L), ["retry", L]).
wam_item_parts(trust(L), ["trust", L]).
wam_item_parts(jump(L), ["jump", L]).
wam_item_parts(cut_ite, ["cut_ite"]).
wam_item_parts(get_level(Yn), ["get_level", Yn]).
wam_item_parts(cut(Yn), ["cut", Yn]).
wam_item_parts(begin_aggregate(K, V, R), ["begin_aggregate", K, V, R]).
wam_item_parts(begin_aggregate(K, V, R, W), ["begin_aggregate", K, V, R, W]).
wam_item_parts(end_aggregate(R), ["end_aggregate", R]).
wam_item_parts(switch_on_constant(Es), ["switch_on_constant"|Es]) :- !.
wam_item_parts(switch_on_constant_fallthrough(Es), ["switch_on_constant_fallthrough"|Es]) :- !.
wam_item_parts(switch_on_constant_a2(Es), ["switch_on_constant_a2"|Es]) :- !.
wam_item_parts(switch_on_constant_a2_fallthrough(Es), ["switch_on_constant_a2_fallthrough"|Es]) :- !.
wam_item_parts(switch_on_structure(Es), ["switch_on_structure"|Es]) :- !.
wam_item_parts(switch_on_structure_a2(Es), ["switch_on_structure_a2"|Es]) :- !.
wam_item_parts(switch_on_term(Ts), ["switch_on_term"|Ts]) :- is_list(Ts), !.
wam_item_parts(switch_on_term_a2(Ts), ["switch_on_term_a2"|Ts]) :- is_list(Ts), !.
wam_item_parts(Item, Parts) :-
    Item =.. [Name|Args],
    atom_string(Name, NameStr),
    maplist(item_arg_string, Args, ArgStrs),
    Parts = [NameStr|ArgStrs].

item_arg_string(Value, Str) :-
    text_to_string(Value, Str).

offset_label_entry(Offset, Entry0, Entry) :-
    atom_string(Entry0, S),
    (   sub_string(S, B, 2, _, ": ")
    ->  sub_string(S, 0, B, _, Prefix),
        B2 is B + 2,
        sub_string(S, B2, _, 0, PCStr),
        number_string(PC0, PCStr),
        PC is PC0 + Offset,
        format(string(Entry), '~w: ~w', [Prefix, PC])
    ;   Entry = Entry0
    ).

is_pred_label(PredKey, Entry) :-
    atom_string(Entry, S),
    sub_string(S, _, _, _, PredKey).

emit_js_wrapper(Pred, Arity, StartPc, Code) :-
    pred_arg_strings(Arity, ArgDecl, ArgList),
    js_pred_name(Pred, Name),
    format(string(Code),
'function ~w(~w) {
  return Runtime.run_predicate(shared_program, ~w, ~w);
}
M.~w = ~w;
', [Name, ArgDecl, StartPc, ArgList, Name, Name]).

emit_js_lowered_wrapper(Pred, Arity, FuncName, Code) :-
    pred_arg_strings(Arity, ArgDecl, ArgList),
    js_pred_name(Pred, Name),
    format(string(Code),
'function ~w(~w) {
  const state = Runtime.new_state();
  const args = ~w;
  for (let i = 0; i < ~w; i++) {
    Runtime.put_reg(state, i + 1, i < args.length && args[i] !== undefined ? args[i] : Runtime.new_var(state));
  }
  state.cp = 0;
  state.program = shared_program;
  return ~w(shared_program, state) === true;
}
M.~w = ~w;
', [Name, ArgDecl, ArgList, Arity, FuncName, Name, Name]).

pred_arg_strings(0, '', '[]') :- !.
pred_arg_strings(Arity, ArgDecl, ArgList) :-
    numlist(1, Arity, Ns),
    maplist([N, A]>>(format(string(A), 'a~w', [N])), Ns, Args),
    atomic_list_concat(Args, ', ', ArgDecl),
    format(string(ArgList), '[~w]', [ArgDecl]).

js_pred_name(Pred, Name) :-
    atom_string(Pred, S),
    string_codes(S, Codes),
    maplist(js_safe_code, Codes, Safe),
    string_codes(SafeS, Safe),
    (   Safe = [C|_], C >= 0'0, C =< 0'9
    ->  string_concat("p_", SafeS, Out)
    ;   Out = SafeS
    ),
    atom_string(Name, Out).

js_safe_code(C, C) :-
    (C >= 0'a, C =< 0'z ; C >= 0'A, C =< 0'Z ; C >= 0'0, C =< 0'9 ; C =:= 0'_), !.
js_safe_code(_, 0'_).

write_wam_javascript_project(Predicates, Options, ProjectDir) :-
    wam_javascript_resolve_emit_mode(Options, _Mode),
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'js', JsDir),
    make_directory_path(JsDir),
    compile_predicates_for_project(Predicates, Options,
        AllInstrs, TopLabels, AllLabels, WrapperCode, LoweredCode, FactSourcesCode),
    emit_js_intern_table(InternSeed),
    javascript_wam_op_decls_code(Options, OpDeclsCode),
    maplist([I, Line]>>(format(string(Line), '  ~w', [I])), AllInstrs, InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    atomic_list_concat(TopLabels, ',\n', DispatchBody),
    atomic_list_concat(AllLabels, ',\n', LabelBody),
    write_runtime_source(JsDir),
    write_program_source(JsDir, InstrBody, LabelBody, DispatchBody,
                         WrapperCode, InternSeed, "", LoweredCode, FactSourcesCode,
                         OpDeclsCode).

write_runtime_source(JsDir) :-
    find_template('templates/targets/javascript_wam/runtime.js.mustache', Template),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(Template, ['date'=Date], Content),
    directory_file_path(JsDir, 'wam_runtime.js', Path),
    write_file(Path, Content).

write_program_source(JsDir, InstrBody, LabelBody, DispatchBody,
                     WrapperCode, InternSeed, ForeignHandlers, LoweredCode, FactSourcesCode,
                     OpDeclsCode) :-
    find_template('templates/targets/javascript_wam/program.js.mustache', Template),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(Template,
        ['date'=Date,
         'instructions'=InstrBody,
         'labels'=LabelBody,
         'dispatch'=DispatchBody,
         'wrappers'=WrapperCode,
         'intern_id_to_string'=InternSeed,
         'foreign_handlers'=ForeignHandlers,
         'lowered_functions'=LoweredCode,
         'fact_sources'=FactSourcesCode,
         'op_decls'=OpDeclsCode], Content),
    directory_file_path(JsDir, 'generated_program.js', Path),
    write_file(Path, Content).

%% javascript_wam_ops([op(Prec, Type, Name), ...])
%  Alias: js_op_decls/1. Emits Runtime.install_declared_ops([...]) so
%  the Pratt reader sees :- op/3 declarations before CLI / read_term.
javascript_wam_op_decls_code(Options, Code) :-
    (   option(javascript_wam_ops(Decls), Options)
    ->  true
    ;   option(js_op_decls(Decls), Options)
    ->  true
    ;   Decls = []
    ),
    javascript_wam_op_decl_lines(Decls, Lines),
    (   Lines == []
    ->  Code = ""
    ;   atomic_list_concat(Lines, ',\n', Body),
        format(string(Code),
               'Runtime.install_declared_ops([\n~w\n]);', [Body])
    ).

javascript_wam_op_decl_lines([], []).
javascript_wam_op_decl_lines([op(Prec, Type, Name)|Rest], Lines) :-
    integer(Prec), Prec >= 0, Prec =< 1200,
    memberchk(Type, [xfx, xfy, yfx, fx, fy, xf, yf]),
    (   atom(Name)
    ->  javascript_wam_op_decl_one(Prec, Type, Name, Line),
        javascript_wam_op_decl_lines(Rest, RestLines),
        Lines = [Line|RestLines]
    ;   is_list(Name)
    ->  maplist(javascript_wam_op_decl_one(Prec, Type), Name, Sub),
        javascript_wam_op_decl_lines(Rest, RestLines),
        append(Sub, RestLines, Lines)
    ).

javascript_wam_op_decl_one(Prec, Type, Name, Line) :-
    atom(Name),
    js_string_literal(Name, NameQ),
    format(string(Line), '  { name: ~w, prec: ~d, type: "~w" }',
           [NameQ, Prec, Type]).

write_file(Path, Content) :-
    setup_call_cleanup(open(Path, write, Stream, [encoding(utf8)]),
                       write(Stream, Content), close(Stream)).

find_template(RelPath, Template) :-
    (   source_file(wam_javascript_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, Root),
        atomic_list_concat([Root, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, []).

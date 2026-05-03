:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_scala_target.pl - WAM-to-Scala Transpilation Target
%
% Generates a hybrid WAM Scala project from a set of Prolog predicates.
% Follows the same two-phase approach as wam_clojure_target.pl:
%   Phase 1: WAM compilation (via wam_target:compile_predicate_to_wam/3)
%   Phase 2: WAM → Scala instruction literals (this file)
%
% Key design decisions (see WAM_SCALA_HYBRID_SPEC.md, §3.4–3.5):
%   - Register names are converted to integer indices at codegen time
%     (A1→1, X3→103, Y2→202), following the Haskell target's reg_to_int.
%   - Atom strings are interned to integer IDs at codegen time.
%     Well-known atoms: true=0, fail=1, []=2.
%   - WamState is mutable; WamProgram is immutable.
%   - Step function mutates WamState in place (no copying per step).

:- module(wam_scala_target, [
    compile_wam_predicate_to_scala/4,  % +Pred/Arity, +WamCode, +Options, -ScalaCode
    write_wam_scala_project/3,         % +Predicates, +Options, +ProjectDir
    scala_foreign_predicate/3          % +Pred, +Arity, +Options
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/template_system', [render_template/3]).

% ============================================================================
% ATOM INTERNING TABLE (compile-time)
% ============================================================================
% Mirrors wam_haskell_target.pl's intern_atom/2 exactly.
% Well-known atoms are pre-assigned; others are assigned sequentially.

:- dynamic scala_atom_intern_id/2.    % scala_atom_intern_id(String, IntId)
:- dynamic scala_atom_intern_next/1.  % scala_atom_intern_next(NextId)

init_scala_atom_intern_table :-
    retractall(scala_atom_intern_id(_, _)),
    retractall(scala_atom_intern_next(_)),
    assertz(scala_atom_intern_id("true", 0)),
    assertz(scala_atom_intern_id("fail", 1)),
    assertz(scala_atom_intern_id("[]",   2)),
    assertz(scala_atom_intern_id(".",    3)),
    assertz(scala_atom_intern_id("",     4)),
    % "[|]" is the SWI/WAM cons functor; pre-intern so it has a stable id
    % regardless of whether the user predicate body emits put_list/get_list
    % (which carry it implicitly) or put_structure [|]/2 (explicit).
    assertz(scala_atom_intern_id("[|]",  5)),
    assertz(scala_atom_intern_next(6)).

%% intern_scala_atom(+AtomStr, -Id) is det.
intern_scala_atom(AtomStr, Id) :-
    atom_string(AtomStr, Str),
    (   scala_atom_intern_id(Str, Id0)
    ->  Id = Id0
    ;   retract(scala_atom_intern_next(Next)),
        Id = Next,
        Next1 is Next + 1,
        assertz(scala_atom_intern_id(Str, Id)),
        assertz(scala_atom_intern_next(Next1))
    ).

%% emit_scala_intern_table(-IdToStringEntries) is det.
%  Generates the Mustache-substitutable string for the intern table seed
%  array. The runtime InternTable is mutable and seeds itself from this
%  array; ids are positional (index 0 -> id 0, etc.) so the order matters.
emit_scala_intern_table(IdToStringEntries) :-
    findall(Id-Str, scala_atom_intern_id(Str, Id), Pairs),
    sort(Pairs, Sorted),
    maplist([_Id-Str, E]>>(format(string(E), '    "~w"', [Str])), Sorted, IEntries),
    atomic_list_concat(IEntries, ',\n', IdToStringEntries).

% ============================================================================
% REGISTER ENCODING
% ============================================================================
% Mirrors wam_haskell_lowered_emitter.pl's reg_to_int/2 exactly.

%% reg_to_int(+RegName, -Int) is det.
%  Converts a WAM register name string to an integer index:
%   A1→1, A2→2  (argument/temp registers)
%   X1→101       (extended temp, offset 100)
%   Y1→201       (permanent, offset 200)
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
% WAM LINE → SCALA INSTRUCTION LITERAL
% ============================================================================

%% wam_line_to_scala_literal(+Line, -ScalaLiteral) is semidet.
%  Converts one WAM assembly text line to a Scala Instruction constructor call.
%  Returns false for label lines and blank lines.
wam_line_to_scala_literal(Line, Literal) :-
    tokenize_wam_line(Line, Parts),
    Parts \= [],
    Parts = [First|_],
    \+ sub_string(First, _, 1, 0, ":"),
    wam_parts_to_scala(Parts, Literal).

%% tokenize_wam_line(+Line, -Tokens)
%  Splits Line on whitespace and commas. Single-quoted segments are
%  treated as opaque tokens — internal spaces are preserved and the
%  surrounding quotes are stripped. Required for things like the
%  format/2 string `'~a is ~w!'` which the simpler `split_string` path
%  would shred into multiple tokens.
tokenize_wam_line(Line, Tokens) :-
    string_chars(Line, Chars),
    tokenize_wam_chars(Chars, [], [], outside, Tokens).

% tokenize_wam_chars(+Chars, +CurReversed, +TokensReversedAcc, +State, -Tokens)
tokenize_wam_chars([], [], Acc, _, Tokens) :- !,
    reverse(Acc, Tokens).
tokenize_wam_chars([], CurR, Acc, _, Tokens) :- !,
    reverse(CurR, CurC), string_chars(T, CurC),
    reverse([T|Acc], Tokens).
tokenize_wam_chars([C|Rest], CurR, Acc, outside, Tokens) :-
    (   (C == ' ' ; C == '\t' ; C == ',')
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, outside, Tokens)
        ;   reverse(CurR, CurC), string_chars(T, CurC),
            tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
        )
    ;   C == '\''
    ->  (   CurR == []
        ->  tokenize_wam_chars(Rest, [], Acc, inside, Tokens)
        ;   % Stray quote in the middle of an unquoted token — keep it.
            tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
        )
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
    ).
tokenize_wam_chars([C|Rest], CurR, Acc, inside, Tokens) :-
    (   C == '\''
    ->  reverse(CurR, CurC), string_chars(T, CurC),
        tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, inside, Tokens)
    ).

% --- Control instructions ---
% The WAM emitter produces both:
%   "execute wam_fact/1"        — 2 tokens, name/arity in one
%   "call wam_fact/1, 1"        — 3 tokens after comma stripping, name and arity
% Either form should yield Call("wam_fact", 1) / Execute("wam_fact", 1).
wam_parts_to_scala(["call", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    format(string(Lit), 'Call("~w", ~w)', [PredName, Arity]).

wam_parts_to_scala(["call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    format(string(Lit), 'Call("~w", ~w)', [PredName, Arity]).

wam_parts_to_scala(["execute", PredArity], Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    format(string(Lit), 'Execute("~w", ~w)', [PredName, Arity]).

wam_parts_to_scala(["execute", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    format(string(Lit), 'Execute("~w", ~w)', [PredName, Arity]).

wam_parts_to_scala(["proceed"], 'Proceed').

wam_parts_to_scala(["jump", Label], Lit) :-
    format(string(Lit), 'Jump("~w")', [Label]).

% --- Choice instructions ---
wam_parts_to_scala(["try_me_else", Label], Lit) :-
    format(string(Lit), 'TryMeElse("~w")', [Label]).

wam_parts_to_scala(["retry_me_else", Label], Lit) :-
    format(string(Lit), 'RetryMeElse("~w")', [Label]).

wam_parts_to_scala(["trust_me"], 'TrustMe').

% --- Environment ---
wam_parts_to_scala(["allocate"], 'Allocate').
wam_parts_to_scala(["deallocate"], 'Deallocate').

% --- Register: get ---
wam_parts_to_scala(["get_constant", C, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    constant_to_scala_term(C, TermLit),
    format(string(Lit), 'GetConstant(~w, ~w)', [TermLit, RegIdx]).

wam_parts_to_scala(["get_variable", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'GetVariable(~w, ~w)', [VIdx, AIdx]).

wam_parts_to_scala(["get_value", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'GetValue(~w, ~w)', [VIdx, AIdx]).

% --- Register: put ---
wam_parts_to_scala(["put_constant", C, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    constant_to_scala_term(C, TermLit),
    format(string(Lit), 'PutConstant(~w, ~w)', [TermLit, RegIdx]).

wam_parts_to_scala(["put_variable", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'PutVariable(~w, ~w)', [VIdx, AIdx]).

wam_parts_to_scala(["put_value", VarReg, ArgReg], Lit) :-
    reg_to_int(VarReg, VIdx), reg_to_int(ArgReg, AIdx),
    format(string(Lit), 'PutValue(~w, ~w)', [VIdx, AIdx]).

% --- Structure / list ---
wam_parts_to_scala(["put_structure", Functor, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    parse_functor_arity(Functor, FName, FArity),
    intern_scala_atom(FName, FId),
    format(string(Lit), 'PutStructure(~w, ~w, ~w)', [FId, RegIdx, FArity]).

wam_parts_to_scala(["put_list", Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    intern_scala_atom("[|]", FId),
    format(string(Lit), 'PutList(~w, ~w)', [RegIdx, FId]).

wam_parts_to_scala(["get_structure", Functor, Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    parse_functor_arity(Functor, FName, _),
    intern_scala_atom(FName, FId),
    format(string(Lit), 'GetStructure(~w, ~w)', [FId, RegIdx]).

wam_parts_to_scala(["get_list", Reg], Lit) :-
    reg_to_int(Reg, RegIdx),
    intern_scala_atom("[|]", FId),
    format(string(Lit), 'GetList(~w, ~w)', [RegIdx, FId]).

wam_parts_to_scala(["set_variable", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'SetVariable(~w)', [Idx]).

wam_parts_to_scala(["set_value", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'SetValue(~w)', [Idx]).

wam_parts_to_scala(["set_constant", C], Lit) :-
    constant_to_scala_term(C, TermLit),
    format(string(Lit), 'SetConstant(~w)', [TermLit]).

wam_parts_to_scala(["unify_variable", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'UnifyVariable(~w)', [Idx]).

wam_parts_to_scala(["unify_value", Reg], Lit) :-
    reg_to_int(Reg, Idx),
    format(string(Lit), 'UnifyValue(~w)', [Idx]).

wam_parts_to_scala(["unify_constant", C], Lit) :-
    constant_to_scala_term(C, TermLit),
    format(string(Lit), 'UnifyConstant(~w)', [TermLit]).

% --- Builtins ---
wam_parts_to_scala(["builtin_call", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    scala_string_literal(Pred, PredLit),
    format(string(Lit), 'BuiltinCall(~w, ~w)', [PredLit, Arity]).

% --- Foreign call ---
wam_parts_to_scala(["call_foreign", Pred, ArityStr], Lit) :-
    number_string(Arity, ArityStr),
    format(string(Lit), 'CallForeign("~w", ~w)', [Pred, Arity]).

% --- Switch on constant ---
wam_parts_to_scala(["switch_on_constant" | Cases], Lit) :-
    parse_switch_cases(Cases, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'SwitchOnConstant(Array(~w))', [CasesStr]).

% --- Switch on term (type-based dispatch) ---
% First-arg type indexing emitted by the WAM compiler when a predicate
% mixes constant, list, and compound first-arg shapes. Format:
%   switch_on_term <CLen> <const_cases...> <SLen> <struct_cases...> <ListLabel>
% e.g. `switch_on_term 1 []:default 0  L_x_2`  (empty struct cases section)
%      `switch_on_term 2 0:default []:L_2 2 f/1:L_4 g/2:L_5 L_3`
% Each const case is `<value>:<label>`; each struct case is
% `<functor>/<arity>:<label>`. ListLabel routes any cons cell ([|]/2).
wam_parts_to_scala(["switch_on_term" | Rest], Lit) :-
    parse_switch_on_term_tokens(Rest, ConstCases, StructCases, ListLabel),
    format_switch_on_term_lit(ConstCases, StructCases, ListLabel, Lit).

% --- ITE soft cut ---
wam_parts_to_scala(["cut_ite"], 'CutIte').

% --- Aggregation (findall/3 etc.) ---
% begin_aggregate Kind, TemplateReg, BagReg
% end_aggregate   TemplateReg
% These come from the WAM lowering of findall(Template, Goal, Bag).
% Kind is the aggregation mode ("collect" for findall).
wam_parts_to_scala(["begin_aggregate", Kind, TemplateReg, BagReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    reg_to_int(BagReg, BIdx),
    format(string(Lit), 'BeginAggregate("~w", ~w, ~w)', [Kind, TIdx, BIdx]).

wam_parts_to_scala(["end_aggregate", TemplateReg], Lit) :-
    reg_to_int(TemplateReg, TIdx),
    format(string(Lit), 'EndAggregate(~w)', [TIdx]).

% --- Fallback ---
wam_parts_to_scala(Parts, Lit) :-
    atomic_list_concat(Parts, ' ', Text),
    format(string(Lit), 'Raw("~w")', [Text]).

%% parse_switch_cases(+Tokens, -CaseLiterals)
%  Parses switch_on_constant case list into SwitchCase constructor calls.
%  Each token has the form "value:label" (e.g. "a:default", "b:L_x_2").
parse_switch_cases([], []).
parse_switch_cases([Token | Rest], [Lit | More]) :-
    split_string(Token, ":", "", [ValStr, LabelStr | _]),
    intern_scala_atom(ValStr, AtomId),
    format(string(Lit), 'SwitchCase(Atom(~w), "~w")', [AtomId, LabelStr]),
    parse_switch_cases(Rest, More).

%% strip_arity_suffix(+Pred, -Name)
%  If Pred has the form "name/N", returns "name"; otherwise returns Pred unchanged.
strip_arity_suffix(Pred, Name) :-
    (   sub_string(Pred, B, 1, _, "/")
    ->  sub_string(Pred, 0, B, _, Name)
    ;   Name = Pred
    ).

%% constant_to_scala_term(+ConstStr, -ScalaTermLit) is det.
%  Converts a WAM constant token to its Scala-source-literal form.
%  Integer tokens become IntTerm(N); float tokens become FloatTerm(N);
%  everything else is interned as an atom.
constant_to_scala_term(C, Lit) :-
    (   number_string(N, C),
        integer(N)
    ->  format(string(Lit), 'IntTerm(~w)', [N])
    ;   number_string(F, C),
        float(F)
    ->  format(string(Lit), 'FloatTerm(~w)', [F])
    ;   intern_scala_atom(C, AtomId),
        format(string(Lit), 'Atom(~w)', [AtomId])
    ).

%% scala_string_literal(+Raw, -Quoted) is det.
%  Wraps Raw in double quotes and escapes backslashes and double quotes
%  so it is a valid Scala string literal. Used for builtin predicate
%  names like `=\=/2` that contain backslashes.
scala_string_literal(Raw, Quoted) :-
    atom_string(Raw, S),
    string_chars(S, Chars),
    maplist(scala_string_escape_char, Chars, EscapedLists),
    append(EscapedLists, EscChars),
    string_chars(EscBody, EscChars),
    format(string(Quoted), '"~w"', [EscBody]).

scala_string_escape_char('\\', ['\\', '\\']) :- !.
scala_string_escape_char('"',  ['\\', '"'])  :- !.
scala_string_escape_char(C, [C]).

%% parse_functor_arity(+FunctorStr, -Name, -Arity)
%  Splits a WAM functor token of the form "name/arity" into Name and Arity.
%  Uses the *last* "/" as the separator so functor names that themselves
%  contain a slash (e.g. the division operator "/" rendered as "//2") are
%  parsed correctly.
parse_functor_arity(FStr, Name, Arity) :-
    atom_string(FA, FStr),
    (   last_slash_index(FA, B)
    ->  sub_atom(FA, 0, B, _, Name),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ;   Name = FA, Arity = 0
    ).

%% last_slash_index(+Atom, -Index)
%  Index of the last "/" in Atom, or fails if none.
last_slash_index(Atom, Index) :-
    findall(B, sub_atom(Atom, B, 1, _, '/'), Bs),
    Bs \= [],
    last(Bs, Index).

% ----------------------------------------------------------
% switch_on_term parsing helpers
% ----------------------------------------------------------
% Lifted out of the wam_parts_to_scala block so that block stays
% contiguous (no `:- discontiguous` directive needed).

parse_switch_on_term_tokens([CLenStr | Rest0], ConstCases, StructCases, ListLabel) :-
    number_string(CLen, CLenStr),
    length(CTokens, CLen),
    append(CTokens, [SLenStr | Rest1], Rest0),
    number_string(SLen, SLenStr),
    length(STokens, SLen),
    append(STokens, [ListLabel], Rest1),
    maplist(parse_const_case_token, CTokens, ConstCases),
    maplist(parse_struct_case_token, STokens, StructCases).

%% parse_const_case_token("<value>:<label>", -case(ValueLit, Label))
parse_const_case_token(Token, case(ValueLit, Label)) :-
    split_at_first_colon(Token, ValueStr, Label),
    constant_to_scala_term(ValueStr, ValueLit).

%% parse_struct_case_token("<functor>/<arity>:<label>", -case(FId, Arity, Label))
parse_struct_case_token(Token, struct(FId, Arity, Label)) :-
    split_at_first_colon(Token, FAStr, Label),
    parse_functor_arity(FAStr, FName, Arity),
    intern_scala_atom(FName, FId).

%% split_at_first_colon(+Token, -Before, -After)
%  Splits at the first ":" — used by switch_on_term parsers where the
%  value half ([], 0, f/2) never contains a ":".
split_at_first_colon(Token, Before, After) :-
    sub_string(Token, B, 1, _, ":"),
    !,
    sub_string(Token, 0, B, _, Before),
    B1 is B + 1,
    sub_string(Token, B1, _, 0, After).

format_switch_on_term_lit(ConstCases, StructCases, ListLabel, Lit) :-
    maplist(const_case_lit, ConstCases, ConstLits),
    atomic_list_concat(ConstLits, ', ', ConstStr),
    maplist(struct_case_lit, StructCases, StructLits),
    atomic_list_concat(StructLits, ', ', StructStr),
    format(string(Lit),
           'SwitchOnTerm(Array(~w), Array(~w), "~w")',
           [ConstStr, StructStr, ListLabel]).

const_case_lit(case(ValueLit, Label), Lit) :-
    format(string(Lit), 'TermSwitchConst(~w, "~w")', [ValueLit, Label]).
struct_case_lit(struct(FId, Arity, Label), Lit) :-
    format(string(Lit), 'TermSwitchStruct(~w, ~w, "~w")', [FId, Arity, Label]).

% ============================================================================
% WAM TEXT → SCALA INSTRUCTION ARRAY
% ============================================================================

%% wam_code_to_scala_data(+WamCode, -Instructions, -LabelMap, -LabelEntries) is det.
%  Converts WAM assembly text to:
%    Instructions: list of Scala Instruction literals (strings)
%    LabelMap:     list of "label" -> pc pairs (for label resolution)
%    LabelEntries: list of formatted '"label" -> N' strings
wam_code_to_scala_data(WamCode, Instructions, LabelMap, LabelEntries) :-
    atom_string(WamCode, Str),
    split_string(Str, "\n", "", Lines),
    wam_lines_to_data(Lines, 0, Instructions, LabelMap, LabelEntries).

wam_lines_to_data([], _, [], [], []).
wam_lines_to_data([Line|Rest], PC, Instructions, LabelMap, LabelEntries) :-
    tokenize_wam_line(Line, Parts),
    (   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  % Label line: extract name, no instruction emitted
        sub_string(First, 0, _, 1, LabelName),
        format(string(LEntry), '    "~w" -> ~w', [LabelName, PC]),
        LabelMap  = [LabelName-PC | LM2],
        LabelEntries = [LEntry | LE2],
        wam_lines_to_data(Rest, PC, Instructions, LM2, LE2)
    ;   Parts = []
    ->  % Blank line
        wam_lines_to_data(Rest, PC, Instructions, LabelMap, LabelEntries)
    ;   % Instruction line
        wam_parts_to_scala(Parts, Lit),
        PC1 is PC + 1,
        Instructions = [Lit | Instrs2],
        wam_lines_to_data(Rest, PC1, Instrs2, LabelMap, LabelEntries)
    ).

% ============================================================================
% PREDICATE COMPILATION
% ============================================================================

%% compile_wam_predicate_to_scala(+PredIndicator, +WamCode, +Options, -ScalaCode)
compile_wam_predicate_to_scala(_Pred, _WamCode, _Options, "").

%% compile_predicates_for_project(+Predicates, +Options, -AllInstrs, -TopLevelLabelEntries, -AllLabelEntries, -WrapperCode)
%  Compiles all predicates. Returns:
%    TopLevelLabelEntries: only "pred/arity" -> PC entries (for Scala Map literal)
%    AllLabelEntries: all labels including sub-clause labels (for instruction resolution)
compile_predicates_for_project(Predicates, Options, AllInstrs, TopLevelLabelEntries, AllLabelEntries, WrapperCode) :-
    init_scala_atom_intern_table,
    % Pre-intern atoms requested via the intern_atoms option. Useful when
    % user-supplied foreign handlers reference atoms that don't appear in
    % any WAM body (otherwise they collapse to the unknown-atom id -1 and
    % can't be distinguished from each other).
    option(intern_atoms(ExtraAtoms), Options, []),
    forall(member(A, ExtraAtoms),
           (atom_string(A, S), intern_scala_atom(S, _))),
    compile_all_predicates(Predicates, Options, 0, [], [], [], [], AllInstrs, TopLevelLabelEntries, AllLabelEntries, Wrappers),
    atomic_list_concat(Wrappers, '\n', WrapperCode).

compile_all_predicates([], _, _, Instrs, TopLabels, AllLabels, Wrappers,
                       Instrs, TopLabels, AllLabels, Wrappers).
compile_all_predicates([Pred|Rest], Options, BasePC,
                       InstrAcc, TopLabelAcc, AllLabelAcc, WrapperAcc,
                       AllInstrs, TopLevelLabelEntries, AllLabelEntries, AllWrappers) :-
    (   Pred = _Module:P/Arity -> true ; Pred = P/Arity ),
    (   scala_foreign_predicate(P, Arity, Options)
    ->  % Foreign stub: CallForeign followed by Proceed. The trailing
        % Proceed is what returns control to the caller after the handler
        % succeeds; without it, pc falls through into the next predicate
        % and re-executes its body as if continuing the foreign call.
        format(string(FLit), 'CallForeign("~w", ~w)', [P, Arity]),
        ForeignSeq = [FLit, 'Proceed'],
        append(InstrAcc, ForeignSeq, NewInstrs),
        NewPC is BasePC + 2,
        format(string(MainEntry), '    "~w/~w" -> ~w', [P, Arity, BasePC]),
        NewTopLabels = [MainEntry | TopLabelAcc],
        NewAllLabels = [MainEntry | AllLabelAcc]
    ;   % WAM compile
        compile_predicate_to_wam(P/Arity, [], WamCode),
        wam_code_to_scala_data(WamCode, PredInstrs, _LMap, PredSubLabelEntries0),
        length(PredInstrs, PredLen),
        NewPC is BasePC + PredLen,
        % Offset sub-clause labels by BasePC
        maplist(offset_label_entry(BasePC), PredSubLabelEntries0, PredSubLabelEntries1),
        % Filter out labels that duplicate the MainEntry (WAM emits pred/arity: as first label)
        format(string(MainKey), '~w/~w', [P, Arity]),
        exclude(is_pred_label(MainKey), PredSubLabelEntries1, PredSubLabelEntries),
        % Main predicate entry label
        format(string(MainEntry), '    "~w/~w" -> ~w', [P, Arity, BasePC]),
        append(InstrAcc, PredInstrs, NewInstrs),
        NewTopLabels = [MainEntry | TopLabelAcc],
        append([MainEntry | PredSubLabelEntries], AllLabelAcc, NewAllLabels)
    ),
    emit_scala_wrapper(P, Arity, BasePC, WrapperCode),
    compile_all_predicates(Rest, Options, NewPC,
                           NewInstrs, NewTopLabels, NewAllLabels, [WrapperCode|WrapperAcc],
                           AllInstrs, TopLevelLabelEntries, AllLabelEntries, AllWrappers).

offset_label_entry(Offset, Entry0, Entry) :-
    % Entry0 is a string like '    "label" -> N'
    % Find the last occurrence of ' -> ' to split label from PC.
    atom_string(Entry0, S),
    (   sub_string(S, B, 4, _, " -> ")
    ->  B1 is B + 4,
        sub_string(S, 0, B, _, LabelPart),
        sub_string(S, B1, _, 0, PCStr),
        number_string(PC0, PCStr),
        PC is PC0 + Offset,
        format(string(Entry), '~w -> ~w', [LabelPart, PC])
    ;   Entry = Entry0  % no ' -> ' found — pass through unchanged
    ).

%% is_pred_label(+PredKey, +LabelEntry) is semidet.
%  True if LabelEntry contains PredKey (e.g. "wam_fact/1").
%  Used to filter redundant WAM predicate-signature labels from sub-clause lists.
is_pred_label(PredKey, Entry) :-
    atom_string(Entry, S),
    sub_string(S, _, _, _, PredKey).

%% emit_scala_wrapper(+Pred, +Arity, +StartPc, -Code)
%  Generates a def wrapper that calls runPredicate with the right start PC.
emit_scala_wrapper(Pred, Arity, StartPc, Code) :-
    % Build argument list: a1: WamTerm, a2: WamTerm, ...
    numlist(1, Arity, ArgNums),
    maplist([N, Arg]>>(format(string(Arg), 'a~w: WamTerm', [N])), ArgNums, ArgDecls),
    atomic_list_concat(ArgDecls, ', ', ArgDeclStr),
    maplist([N, Arg]>>(format(string(Arg), 'a~w', [N])), ArgNums, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgNameStr),
    scala_pred_name(Pred, ScalaName),
    format(string(Code),
           '  def ~w(~w): Boolean =\n    WamRuntime.runPredicate(sharedProgram, ~w, Array(~w))\n',
           [ScalaName, ArgDeclStr, StartPc, ArgNameStr]).

%% scala_pred_name(+PrologName, -ScalaName)
%  Converts a Prolog predicate atom to a Scala camelCase identifier.
%  wam_fact -> wamFact, category_parent -> categoryParent
scala_pred_name(Pred, ScalaName) :-
    atom_string(Pred, PStr),
    split_string(PStr, "_", "", Parts),
    capitalize_parts(Parts, Capitalized),
    atomic_list_concat(Capitalized, '', CamelCase),
    atom_string(CamelCase, CCStr),
    % Lowercase the first character
    sub_string(CCStr, 0, 1, _, First),
    sub_string(CCStr, 1, _, 0, Rest),
    string_lower(First, Lower),
    string_concat(Lower, Rest, ScalaName).

capitalize_parts([], []).
capitalize_parts([P|Rest], [C|More]) :-
    (   P = ""
    ->  C = ""
    ;   sub_string(P, 0, 1, _, H),
        sub_string(P, 1, _, 0, T),
        string_upper(H, HU),
        string_concat(HU, T, C)
    ),
    capitalize_parts(Rest, More).

string_lower(S, L) :- string_lower_char(S, L).
string_lower_char(S, L) :-
    string_codes(S, [C|_]),
    (   C >= 0'A, C =< 0'Z
    ->  LC is C + 32
    ;   LC = C
    ),
    string_codes(L, [LC]).

% ============================================================================
% FOREIGN PREDICATE DETECTION
% ============================================================================

%% scala_foreign_predicate(+Pred, +Arity, +Options) is semidet.
%  True if Pred/Arity should be treated as a foreign predicate stub.
scala_foreign_predicate(Pred, Arity, Options) :-
    option(foreign_predicates(FPs), Options, []),
    (   member(Pred/Arity, FPs)
    ;   member(_:Pred/Arity, FPs)
    ), !.

%% scala_foreign_handlers_code(+Options, -Code) is det.
%  Renders the body of the `foreignHandlers` Map in the generated
%  program. Reads `scala_foreign_handlers([handler(P/A, "<scala>"), ...])`
%  from Options. Each handler value is a Scala expression of type
%  `ForeignHandler` (typically `new ForeignHandler { def apply(...) = ... }`).
%  When no handlers are configured, returns the empty string.
scala_foreign_handlers_code(Options, Code) :-
    option(scala_foreign_handlers(Handlers), Options, []),
    maplist(scala_foreign_handler_entry, Handlers, Entries),
    atomic_list_concat(Entries, ',\n', Code).

scala_foreign_handler_entry(handler(Pred/Arity, HandlerCode), Entry) :-
    format(string(Entry), '    "~w/~w" -> ~w', [Pred, Arity, HandlerCode]).

% ============================================================================
% FACT BACKEND SEAM (Phase S7)
% ============================================================================
% A FactSource is a declarative way to provide fact-shaped data to a
% predicate at codegen time without writing a full Scala ForeignHandler.
% The user supplies `scala_fact_sources([source(P/A, Tuples), ...])` where
% each Tuple is a list of Arity atoms. The codegen synthesises:
%   - a ForeignHandler that enumerates all tuples as ForeignMulti solutions
%     (the runtime's existing applyBindings + backtracking machinery filters
%     them against the input args);
%   - a foreign_predicates entry so the WAM body is replaced by a
%     CallForeign stub;
%   - intern_atoms entries for every atom used in any tuple.
% This is the "inline" implementation; sidecar (file/LMDB) backends will
% slot into the same option in later phases.

%% expand_fact_sources_in_options(+Options0, -Options) is det.
%  Replaces scala_fact_sources(...) entries with equivalent
%  foreign_predicates + scala_foreign_handlers + intern_atoms entries,
%  preserving any user-supplied entries in those lists by union.
expand_fact_sources_in_options(Options0, Options) :-
    (   option(scala_fact_sources(Sources), Options0, []),
        Sources \= []
    ->  findall(P/A, member(source(P/A, _), Sources), SourcePreds),
        findall(handler(P/A, Code),
                (   member(source(P/A, Spec), Sources),
                    fact_source_spec_to_handler_code(A, Spec, Code)
                ),
                SourceHandlers),
        % Collect atoms from inline tuple sources for pre-interning. File-
        % backed sources can't pre-intern at codegen time (the file is read
        % at runtime), so the user must declare expected atoms via
        % intern_atoms(...) if their handler-side code needs them.
        findall(Atom,
                (   member(source(_/_, Spec), Sources),
                    is_list(Spec),
                    member(Tuple, Spec),
                    member(Atom, Tuple),
                    \+ number(Atom)
                ),
                FactAtomsBag),
        list_to_set(FactAtomsBag, FactAtoms),
        % Union with any user-supplied lists. option/3 picks the first
        % occurrence; we replace it with the unioned form below.
        option(foreign_predicates(FPs0), Options0, []),
        option(scala_foreign_handlers(FHs0), Options0, []),
        option(intern_atoms(IA0), Options0, []),
        list_union(FPs0, SourcePreds, FPsAll),
        list_union(FHs0, SourceHandlers, FHsAll),
        list_union(IA0, FactAtoms, IAAll),
        replace_option(foreign_predicates, FPsAll, Options0, O1),
        replace_option(scala_foreign_handlers, FHsAll, O1, O2),
        replace_option(intern_atoms, IAAll, O2, Options)
    ;   Options = Options0
    ).

%% fact_source_spec_to_handler_code(+Arity, +Spec, -ScalaCode) is det.
%  Dispatches on the shape of Spec:
%    - Spec is a list of tuples → inline handler enumerating them.
%    - Spec is file('path') → handler that reads CSV at runtime.
%    - Spec is grouped_by_first('path') → handler that reads TSV rows
%      shaped as Key<TAB>Value... and probes by the first argument.
fact_source_spec_to_handler_code(Arity, Spec, Code) :-
    is_list(Spec), !,
    fact_source_to_handler_code(Arity, Spec, Code).
fact_source_spec_to_handler_code(Arity, file(Path), Code) :-
    !,
    atom_string(Path, PathStr),
    scala_string_literal(PathStr, PathLit),
    % The CSV is read and parsed ONCE at handler-instance init (the
    % `private val sols`), not on every apply call. Caching this way
    % closes most of the per-iteration gap with inline tuples.
    format(string(Code),
           "new ForeignHandler {\n      private val sols: Seq[Map[Int, WamTerm]] = {\n        val src = scala.io.Source.fromFile(~w)\n        try {\n          src.getLines().toList.map { line =>\n            val parts = line.split(\",\").map(_.trim)\n            parts.zipWithIndex.map { case (p, i) => (i + 1) -> parseFactArg(p) }.toMap\n          }\n        } finally { src.close() }\n      }\n      def apply(args: Array[WamTerm]): ForeignResult =\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n    }",
           [PathLit]),
    % Arity isn't enforced here (each line uses whatever columns are
    % present); a malformed CSV will simply produce wrong-arity bindings
    % that the runtime will fail to unify.
    _ = Arity.
fact_source_spec_to_handler_code(2, grouped_by_first(Path), Code) :-
    !,
    atom_string(Path, PathStr),
    scala_string_literal(PathStr, PathLit),
    format(string(Code),
           "new ForeignHandler {\n      private def termKey(term: WamTerm): Option[String] = term match {\n        case Atom(id) if internTable.isInRange(id) => Some(internTable.stringAt(id))\n        case IntTerm(value) => Some(value.toString)\n        case FloatTerm(value) => Some(value.toString)\n        case _ => None\n      }\n      private val parentsByChild: Map[String, Vector[String]] = {\n        val src = scala.io.Source.fromFile(~w)\n        try {\n          src.getLines().toVector.flatMap { raw =>\n            val line = raw.trim\n            if (line.isEmpty || line.startsWith(\"#\")) None\n            else {\n              val parts = line.split(\"\\\\t\").map(_.trim).filter(_.nonEmpty).toVector\n              if (parts.length >= 2) Some(parts.head -> parts.tail) else None\n            }\n          }.toMap\n        } finally { src.close() }\n      }\n      private val allSols: Vector[Map[Int, WamTerm]] =\n        parentsByChild.toVector.flatMap { case (child, parents) =>\n          parents.map(parent => Map(1 -> parseFactArg(child), 2 -> parseFactArg(parent)))\n        }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val sols = termKey(args(0)) match {\n          case Some(child) => parentsByChild.getOrElse(child, Vector.empty).map(parent => Map(1 -> parseFactArg(child), 2 -> parseFactArg(parent)))\n          case None => allSols\n        }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [PathLit]).

%% fact_source_to_handler_code(+Arity, +Tuples, -ScalaCode) is det.
%  The solutions Seq is hoisted to a `val` on the anonymous class so it
%  is built once at handler construction, not on every apply call. For
%  large tuple lists this is a notable per-iteration win.
fact_source_to_handler_code(Arity, Tuples, Code) :-
    maplist(tuple_to_solution_map_lit(Arity), Tuples, SolLits),
    atomic_list_concat(SolLits, ',\n        ', SolBody),
    format(string(Code),
           "new ForeignHandler {\n      private val sols: Seq[Map[Int, WamTerm]] = Seq(\n        ~w\n      )\n      def apply(args: Array[WamTerm]): ForeignResult =\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n    }",
           [SolBody]).

tuple_to_solution_map_lit(Arity, Tuple, Lit) :-
    is_list(Tuple),
    length(Tuple, Arity),
    map_args_to_pairs(Tuple, 1, Pairs),
    atomic_list_concat(Pairs, ', ', PairsStr),
    format(string(Lit), 'Map(~w)', [PairsStr]).

map_args_to_pairs([], _, []).
map_args_to_pairs([A|Rest], N, [Pair|More]) :-
    fact_arg_to_scala_term(A, TermLit),
    format(string(Pair), '~w -> ~w', [N, TermLit]),
    N1 is N + 1,
    map_args_to_pairs(Rest, N1, More).

%% fact_arg_to_scala_term(+Arg, -ScalaTermLit)
%  Same numeric/atom split as constant_to_scala_term/2 but operating
%  on Prolog terms (not WAM-text strings).
fact_arg_to_scala_term(N, Lit) :-
    integer(N), !,
    format(string(Lit), 'IntTerm(~w)', [N]).
fact_arg_to_scala_term(F, Lit) :-
    float(F), !,
    format(string(Lit), 'FloatTerm(~w)', [F]).
fact_arg_to_scala_term(A, Lit) :-
    atom_string(A, S),
    intern_scala_atom(S, AtomId),
    format(string(Lit), 'Atom(~w)', [AtomId]).

%% list_union(+L1, +L2, -Union) — preserves order of L1, then appends new of L2.
list_union(L1, L2, Union) :-
    findall(X, (member(X, L2), \+ member(X, L1)), New),
    append(L1, New, Union).

%% replace_option(+Key, +Value, +Options0, -Options)
%  Removes any existing Key(...) entry from Options0 and prepends
%  Key(Value).
replace_option(Key, Value, Options0, [NewEntry | Cleaned]) :-
    exclude([X]>>( compound(X), X =.. [Key, _] ), Options0, Cleaned),
    NewEntry =.. [Key, Value].

% ============================================================================
% PROJECT WRITER
% ============================================================================

%% write_wam_scala_project(+Predicates, +Options0, +ProjectDir) is det.
%  Creates a complete Scala WAM project in ProjectDir. The S7 fact-backend
%  seam is processed before anything else: scala_fact_sources(...) entries
%  expand into equivalent foreign_predicates + scala_foreign_handlers +
%  intern_atoms entries, so the rest of the pipeline doesn't need to know
%  about fact sources as a distinct concept.
write_wam_scala_project(Predicates, Options0, ProjectDir) :-
    expand_fact_sources_in_options(Options0, Options),
    make_directory_path(ProjectDir),
    % --- build.sbt ---
    option(module_name(ModName), Options, 'wam-scala-generated'),
    write_build_sbt(ProjectDir, ModName),
    % --- project/build.properties ---
    write_build_properties(ProjectDir),
    % --- Compile all predicates ---
    compile_predicates_for_project(Predicates, Options,
        AllInstrs, TopLevelLabelEntries, AllLabelEntries, WrapperCode),
    % --- Intern table ---
    emit_scala_intern_table(IdToStringStr),
    % --- Format instruction array body ---
    maplist([I, Line]>>(format(string(Line), '    ~w', [I])), AllInstrs, InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    % --- Format dispatch map body (top-level pred/arity only) ---
    atomic_list_concat(TopLevelLabelEntries, ',\n', DispatchBody),
    % --- Format full label map for instruction resolution (top-level + sub-clause) ---
    atomic_list_concat(AllLabelEntries, ',\n', LabelBody),
    % --- Package and runtime package ---
    option(package(Pkg), Options, 'generated.wam_scala.core'),
    option(runtime_package(RPkg), Options, 'generated.wam_scala.runtime'),
    % --- Foreign handler bodies ---
    scala_foreign_handlers_code(Options, ForeignHandlersBody),
    % --- Render runtime template ---
    write_runtime_source(ProjectDir, Pkg, RPkg),
    % --- Render program template ---
    write_program_source(ProjectDir, Pkg, RPkg,
                         InstrBody, LabelBody, DispatchBody,
                         WrapperCode, IdToStringStr,
                         ForeignHandlersBody).

write_build_sbt(ProjectDir, ModName) :-
    find_template('templates/targets/scala_wam/build.sbt.mustache', Template),
    render_template(Template, ['module_name'=ModName], Content),
    directory_file_path(ProjectDir, 'build.sbt', Path),
    write_file(Path, Content).

write_build_properties(ProjectDir) :-
    find_template('templates/targets/scala_wam/build.properties.mustache', Template),
    render_template(Template, [], Content),
    directory_file_path(ProjectDir, 'project', ProjDir),
    make_directory_path(ProjDir),
    directory_file_path(ProjDir, 'build.properties', Path),
    write_file(Path, Content).

write_runtime_source(ProjectDir, Package, _RuntimePkg) :-
    find_template('templates/targets/scala_wam/runtime.scala.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template, ['package'=Package, 'date'=DateStr], Content),
    scala_source_path(ProjectDir, Package, 'WamRuntime', Path),
    make_directory_path_for(Path),
    write_file(Path, Content).

write_program_source(ProjectDir, Package, RuntimePkg,
                     InstrBody, LabelBody, DispatchBody,
                     WrapperCode, IdToStringStr,
                     ForeignHandlersBody) :-
    find_template('templates/targets/scala_wam/program.scala.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template,
        [ 'package'=Package,
          'runtime_package'=RuntimePkg,
          'date'=DateStr,
          'instructions'=InstrBody,
          'labels'=LabelBody,
          'dispatch'=DispatchBody,
          'wrappers'=WrapperCode,
          'intern_id_to_string'=IdToStringStr,
          'foreign_handlers'=ForeignHandlersBody
        ], Content),
    scala_source_path(ProjectDir, Package, 'GeneratedProgram', Path),
    make_directory_path_for(Path),
    write_file(Path, Content).

% ============================================================================
% HELPERS
% ============================================================================

%% scala_source_path(+ProjectDir, +Package, +ClassName, -AbsPath)
%  Converts a Scala package + class name to a src/main/scala/... path.
scala_source_path(ProjectDir, Package, ClassName, Path) :-
    atom_string(Package, PkgStr),
    split_string(PkgStr, ".", "", Parts),
    atomic_list_concat(Parts, '/', PkgPath),
    format(string(RelPath), 'src/main/scala/~w/~w.scala', [PkgPath, ClassName]),
    directory_file_path(ProjectDir, RelPath, Path).

make_directory_path_for(FilePath) :-
    file_directory_name(FilePath, Dir),
    make_directory_path(Dir).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, Content),
        close(Stream)
    ).

%% find_template(+RelPath, -Template) is det.
%  Locates a template file relative to the UnifyWeaver project root.
find_template(RelPath, Template) :-
    (   source_file(wam_scala_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, UnifyWeaverDir),
        atomic_list_concat([UnifyWeaverDir, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, []).

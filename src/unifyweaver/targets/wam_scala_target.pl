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
    scala_foreign_predicate/3,         % +Pred, +Arity, +Options
    % --- Hooks for the lowered emitter (wam_scala_lowered_emitter.pl) ---
    % These expose the codegen-time atom interning / register / constant /
    % functor helpers so the lowered emitter produces atom IDs and register
    % indices that match the shared instruction array exactly.
    scala_lowered_constant_term/2,     % +ConstTokenStr, -ScalaTermLiteral
    scala_lowered_functor_arity/3,     % +FunctorTokenStr, -Name, -Arity
    scala_lowered_reg_index/2,         % +RegNameStr, -IntIndex
    scala_lowered_intern_atom/2,       % +AtomStr, -IntId
    scala_resolve_emit_mode/2,         % +Options, -Mode
    scala_partition_predicates/4       % +Mode, +Predicates, -Interp, -Lowered
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../targets/wam_text_parser', [wam_classify_constant_token/2]).
:- use_module('../core/recursive_kernel_detection',
              [detect_recursive_kernel/4, kernel_config/2]).

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
% LOWERED-EMITTER HOOKS
% ============================================================================
% Thin, exported wrappers around the codegen-time interning / register /
% constant / functor helpers, so wam_scala_lowered_emitter.pl produces atom
% IDs and register indices that match the shared instruction array.  They
% MUST be called within the same intern-table session as the main program
% emission (see compile_predicates_for_project/6) so the IDs line up.

scala_lowered_reg_index(RegStr, Int)      :- reg_to_int(RegStr, Int).
scala_lowered_constant_term(CStr, Term)   :- constant_to_scala_term(CStr, Term).
scala_lowered_functor_arity(FStr, N, A)   :- parse_functor_arity(FStr, N, A).
scala_lowered_intern_atom(AtomStr, Id)    :- intern_scala_atom(AtomStr, Id).

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
    tokenize_wam_chars(Chars, [], [], outside, Tokens0),
    % M150: a quoted operand followed by an operand comma (e.g.
    % put_constant '~a is ~w!', A1) left a residual empty token after
    % comma-stripping, so the parts list never matched the 3-arg
    % put_constant clause and the instruction fell to the failing
    % fallback - any format string containing a space wrong-failed
    % its whole predicate.
    exclude(==(""), Tokens0, Tokens).

% tokenize_wam_chars(+Chars, +CurReversed, +TokensReversedAcc, +State, -Tokens)
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
        ->  % Enter quoted region — keep the opening quote attached
            % to the token so atom-vs-number is recoverable
            % downstream via wam_classify_constant_token/2.
            tokenize_wam_chars(Rest, ['\''], Acc, inside, Tokens)
        ;   % Stray quote in the middle of an unquoted token — keep it.
            tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
        )
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, outside, Tokens)
    ).
tokenize_wam_chars([C|Rest], CurR, Acc, inside, Tokens) :-
    (   C == '\\',
        Rest = [Escaped|More]
    ->  tokenize_wam_chars(More, [Escaped|CurR], Acc, inside, Tokens)
    ;   C == '\''
    ->  % Closing quote — keep it attached so the outer quotes
        % survive to the constant classifier.
        reverse(['\''|CurR], CurC), string_chars(T, CurC),
        tokenize_wam_chars(Rest, [], [T|Acc], outside, Tokens)
    ;   tokenize_wam_chars(Rest, [C|CurR], Acc, inside, Tokens)
    ).

strip_operand_comma(Token0, Token) :-
    sub_string(Token0, _, 1, 0, ","),
    !,
    sub_string(Token0, 0, _, 1, Token).
strip_operand_comma(Token, Token).

% --- Control instructions ---
% The WAM emitter produces both:
%   "execute wam_fact/1"        — 2 tokens, name/arity in one
%   "call wam_fact/1, 1"        — 3 tokens after comma stripping, name and arity
% Either form should yield Call("wam_fact", 1) / Execute("wam_fact", 1).
wam_parts_to_scala(Parts, Options, Lit) :-
    wam_parts_to_foreign_call(Parts, Options, Lit), !.
wam_parts_to_scala(Parts, _Options, Lit) :-
    wam_parts_to_scala(Parts, Lit).

wam_parts_to_foreign_call(["call", PredArity], Options, Lit) :-
    parse_functor_arity(PredArity, PredName, Arity),
    scala_foreign_predicate(PredName, Arity, Options),
    format(string(Lit), 'CallForeign("~w", ~w)', [PredName, Arity]).
wam_parts_to_foreign_call(["call", Pred, ArityStr], Options, Lit) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix(Pred, PredName),
    scala_foreign_predicate(PredName, Arity, Options),
    format(string(Lit), 'CallForeign("~w", ~w)', [PredName, Arity]).

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

%% Indexed-dispatch chain ops (issue #2400).  See wam_target.pl''s
%% build_term_index_with_chains: synthesized try/retry/trust chains
%% target the L_<Pred>_<Arity>_<I>_body labels when a dispatch group
%% has >1 matching clauses.  Unlike try_me_else (CP points to alt,
%% advance pc), these instructions JUMP to the body label and the
%% CP holds the in-chain fall-through PC (= next chain instruction).
wam_parts_to_scala(["try", Label], Lit) :-
    format(string(Lit), 'Try("~w")', [Label]).

wam_parts_to_scala(["retry", Label], Lit) :-
    format(string(Lit), 'Retry("~w")', [Label]).

wam_parts_to_scala(["trust", Label], Lit) :-
    format(string(Lit), 'Trust("~w")', [Label]).

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
    normalize_switch_case_tokens(Cases, NormalizedCases),
    parse_switch_cases(NormalizedCases, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'SwitchOnConstant(Array(~w))', [CasesStr]).

% switch_on_constant_fallthrough is shape-compatible with switch_on_constant:
% both match A1 against the listed constants and jump to the matching label
% on a hit. The runtime's switchTarget already falls through to pc+1 on a
% miss (or to a "default"-labelled case with targetPc = -1), which is exactly
% fall-through semantics — so we reuse the SwitchOnConstant instruction.
% Mirrors wam_fsharp_target.pl. Without this the instruction degraded to a
% Raw(...) stub and broke first-argument-indexed predicates (e.g. the
% mixed fact+rule shape of factorial/Ackermann/Fibonacci base cases).
wam_parts_to_scala(["switch_on_constant_fallthrough" | Cases], Lit) :-
    wam_parts_to_scala(["switch_on_constant" | Cases], Lit).

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
    format_switch_on_term_lit(ConstCases, StructCases, ListLabel, 1, Lit).

% A2-indexed variants. The WAM compiler emits the `_a2` family when it
% indexes on the second argument (e.g. member/2's list arg). Same case
% layout; only the indexed register differs (regs(2) instead of regs(1)).
% Without these the instructions degraded to Raw(...) stubs and broke the
% interpreter for A2-indexed predicates. Mirrors the F#/C/C++ targets.
wam_parts_to_scala(["switch_on_term_a2" | Rest], Lit) :-
    parse_switch_on_term_tokens(Rest, ConstCases, StructCases, ListLabel),
    format_switch_on_term_lit(ConstCases, StructCases, ListLabel, 2, Lit).

wam_parts_to_scala(["switch_on_constant_a2" | Cases], Lit) :-
    normalize_switch_case_tokens(Cases, NormalizedCases),
    parse_switch_cases(NormalizedCases, CaseLits),
    atomic_list_concat(CaseLits, ', ', CasesStr),
    format(string(Lit), 'SwitchOnConstant(Array(~w), 2)', [CasesStr]).

wam_parts_to_scala(["switch_on_constant_a2_fallthrough" | Cases], Lit) :-
    wam_parts_to_scala(["switch_on_constant_a2" | Cases], Lit).

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
    scala_string_literal(Text, TextLit),
    format(string(Lit), 'Raw(~w)', [TextLit]).

%% parse_switch_cases(+Tokens, -CaseLiterals)
%  Parses switch_on_constant case list into SwitchCase constructor calls.
%  Each token has the form "value:label" (e.g. "a:default", "b:L_x_2").
parse_switch_cases([], []).
parse_switch_cases([Token | Rest], [Lit | More]) :-
    split_at_first_colon(Token, ValStr, LabelStr),
    intern_scala_atom(ValStr, AtomId),
    format(string(Lit), 'SwitchCase(Atom(~w), "~w")', [AtomId, LabelStr]),
    parse_switch_cases(Rest, More).

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
%
%  Atom-vs-number disambiguation goes through
%  wam_text_parser:wam_classify_constant_token/2: a bare token `5`
%  is the integer 5, a quoted token `'5'` is the atom whose name is
%  "5". tokenize_wam_chars/5 above preserves outer quotes attached
%  to the token so the discriminator reaches this predicate.
constant_to_scala_term(C, Lit) :-
    wam_classify_constant_token(C, Class),
    (   Class = integer(N)
    ->  format(string(Lit), 'IntTerm(~w)', [N])
    ;   Class = float(F)
    ->  format(string(Lit), 'FloatTerm(~w)', [F])
    ;   Class = atom(Name),
        intern_scala_atom(Name, AtomId),
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

%% format_switch_on_term_lit(+ConstCases, +StructCases, +ListLabel, +Reg, -Lit)
%  Reg selects the indexed argument register (1 for switch_on_term, 2 for
%  switch_on_term_a2). The Scala SwitchOnTerm case class defaults reg to 1,
%  so for Reg==1 we omit it (keeping output identical to the pre-A2 codegen)
%  and for Reg==2 we pass it as a named argument.
format_switch_on_term_lit(ConstCases, StructCases, ListLabel, Reg, Lit) :-
    maplist(const_case_lit, ConstCases, ConstLits),
    atomic_list_concat(ConstLits, ', ', ConstStr),
    maplist(struct_case_lit, StructCases, StructLits),
    atomic_list_concat(StructLits, ', ', StructStr),
    (   Reg =:= 1
    ->  format(string(Lit),
               'SwitchOnTerm(Array(~w), Array(~w), "~w")',
               [ConstStr, StructStr, ListLabel])
    ;   format(string(Lit),
               'SwitchOnTerm(Array(~w), Array(~w), "~w", reg = ~w)',
               [ConstStr, StructStr, ListLabel, Reg])
    ).

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
    wam_code_to_scala_data(WamCode, [], Instructions, LabelMap, LabelEntries).

wam_code_to_scala_data(WamCode, Options, Instructions, LabelMap, LabelEntries) :-
    atom_string(WamCode, Str),
    split_string(Str, "\n", "", Lines),
    wam_lines_to_data(Lines, Options, 0, Instructions, LabelMap, LabelEntries).

wam_lines_to_data([], _, _, [], [], []).
wam_lines_to_data([Line|Rest], Options, PC, Instructions, LabelMap, LabelEntries) :-
    tokenize_wam_line(Line, Parts),
    (   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  % Label line: extract name, no instruction emitted
        sub_string(First, 0, _, 1, LabelName),
        format(string(LEntry), '    "~w" -> ~w', [LabelName, PC]),
        LabelMap  = [LabelName-PC | LM2],
        LabelEntries = [LEntry | LE2],
        wam_lines_to_data(Rest, Options, PC, Instructions, LM2, LE2)
    ;   Parts = []
    ->  % Blank line
        wam_lines_to_data(Rest, Options, PC, Instructions, LabelMap, LabelEntries)
    ;   % Instruction line
        wam_parts_to_scala(Parts, Options, Lit),
        PC1 is PC + 1,
        Instructions = [Lit | Instrs2],
        wam_lines_to_data(Rest, Options, PC1, Instrs2, LabelMap, LabelEntries)
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
    option(foreign_predicates(ForeignPredicates), Options, []),
    append_missing_foreign_predicates(Predicates, ForeignPredicates, CompilePredicates),
    compile_all_predicates(CompilePredicates, Options, 0, [], [], [], [], AllInstrs, TopLevelLabelEntries, AllLabelEntries, Wrappers),
    atomic_list_concat(Wrappers, '\n', WrapperCode).

append_missing_foreign_predicates(Predicates, ForeignPredicates, CompilePredicates) :-
    findall(Foreign,
            (   member(Foreign, ForeignPredicates),
                \+ ( member(Pred, Predicates),
                     same_predicate_indicator(Pred, Foreign)
                   )
            ),
            MissingForeignPredicates),
    append(Predicates, MissingForeignPredicates, CompilePredicates).

same_predicate_indicator(Pred0, Pred1) :-
    predicate_indicator_key(Pred0, Key),
    predicate_indicator_key(Pred1, Key).

predicate_indicator_key(_Module:Pred/Arity, Pred/Arity) :- !.
predicate_indicator_key(Pred/Arity, Pred/Arity).

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
        wam_code_to_scala_data(WamCode, Options, PredInstrs, _LMap, PredSubLabelEntries0),
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
    % numlist/3 fails when Low > High, so guard the 0-arity case (e.g.
    % `p :- 3 > 2.`): without this the wrapper — and the whole project
    % write — silently fails for any 0-arity predicate.
    (   Arity >= 1 -> numlist(1, Arity, ArgNums) ; ArgNums = [] ),
    maplist([N, Arg]>>(format(string(Arg), 'a~w: WamTerm', [N])), ArgNums, ArgDecls),
    atomic_list_concat(ArgDecls, ', ', ArgDeclStr),
    maplist([N, Arg]>>(format(string(Arg), 'a~w', [N])), ArgNums, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgNameStr),
    scala_pred_name(Pred, ScalaName),
    % Route through runEntry so a lowered fast path (when present) is used;
    % runEntry falls back to runPredicate at StartPc otherwise. Behaviour is
    % identical in interpreter mode (loweredEntries is empty).
    % Array[WamTerm](...) is typed explicitly so the 0-arg case emits a
    % well-typed empty array (`Array()` alone infers Array[Nothing] and
    % won't match runEntry's Array[WamTerm] parameter).
    format(string(Code),
           '  def ~w(~w): Boolean =\n    runEntry("~w/~w", ~w, Array[WamTerm](~w))\n',
           [ScalaName, ArgDeclStr, Pred, Arity, StartPc, ArgNameStr]).

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
    (   string(Pred)
    ->  atom_string(PredAtom, Pred)
    ;   PredAtom = Pred
    ),
    option(foreign_predicates(FPs), Options, []),
    (   member(PredAtom/Arity, FPs)
    ;   member(_:PredAtom/Arity, FPs)
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
% HOT-PATH GRAPH KERNELS
% ============================================================================
% Opt-in (kernel_dispatch(true)) native lowering of recursive graph
% predicates, bringing the Scala target to parity with the
% Rust/Haskell/Elixir/Go kernel route. A predicate matching one of the
% shapes recognised by core/recursive_kernel_detection.pl is replaced by a
% synthesized Scala ForeignHandler that performs the traversal natively
% (bypassing the WAM step loop), reusing the existing foreign-predicate
% seam. The handler builds its adjacency map by enumerating the kernel's
% edge relation through WamRuntime.collectBinarySolutions/2, so it works
% whether the edges are WAM-compiled facts or a declarative fact source.
%
% Currently implemented kernel kinds: transitive_closure2, transitive_distance3,
% transitive_parent_distance4, transitive_step_parent_distance5,
% category_ancestor, weighted_shortest_path3, astar_shortest_path4
% (all seven recognised kinds).
% The remaining six (category_ancestor, transitive_distance3,
% weighted_shortest_path3, transitive_parent_distance4,
% astar_shortest_path4, transitive_step_parent_distance5) follow the same
% pattern and are slated as follow-ups.

%% expand_kernels_in_options(+Predicates, +Options0, -Options) is det.
%  When kernel_dispatch(true) is set, detect kernels among Predicates and
%  fold the implementable ones into foreign_predicates + scala_foreign_handlers
%  (union with any user-supplied entries). Otherwise a no-op.
expand_kernels_in_options(Predicates, Options0, Options) :-
    (   option(kernel_dispatch(true), Options0)
    ->  detect_scala_kernels(Predicates, Kernels),
        findall(P/A-Code,
                ( member(_Key-Kernel, Kernels),
                  kernel_predicate_indicator(Kernel, P/A),
                  emit_scala_kernel_handler(Kernel, Code)
                ),
                KernelPairs),
        (   KernelPairs == []
        ->  Options = Options0
        ;   findall(KP, member(KP-_, KernelPairs), KernelPreds),
            findall(handler(KP, C), member(KP-C, KernelPairs), KernelHandlers),
            option(foreign_predicates(FPs0), Options0, []),
            option(scala_foreign_handlers(FHs0), Options0, []),
            list_union(FPs0, KernelPreds, FPsAll),
            list_union(FHs0, KernelHandlers, FHsAll),
            replace_option(foreign_predicates, FPsAll, Options0, O1),
            replace_option(scala_foreign_handlers, FHsAll, O1, Options)
        )
    ;   Options = Options0
    ).

%% detect_scala_kernels(+Predicates, -Kernels)
%  Kernels = list of "pred/arity"-recursive_kernel(...) pairs. Mirrors the
%  Rust/Elixir detect_kernels/2.
detect_scala_kernels([], []).
detect_scala_kernels([PI|Rest], Kernels) :-
    ( PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        catch(detect_recursive_kernel(Pred, Arity, Clauses, Kernel), _, fail)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel | RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_scala_kernels(Rest, RestKernels).

kernel_predicate_indicator(recursive_kernel(_, Pred/Arity, _), Pred/Arity).

%% emit_scala_kernel_handler(+Kernel, -ScalaHandlerCode) is semidet.
%  Synthesizes the Scala ForeignHandler source for a detected kernel.
%  Fails for kernel kinds not yet implemented (so they fall back to the
%  ordinary WAM compilation path and remain correct, just not accelerated).
emit_scala_kernel_handler(recursive_kernel(transitive_closure2, _Pred/2, ConfigOps), Code) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    format(atom(EdgeKey), '~w/2', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    % BFS from arg(0) over the edge relation; every node reachable in >=1
    % step is a solution binding register 2. Adjacency is built once
    % (lazy val) from collectBinarySolutions so repeated queries reuse it.
    format(string(Code),
"new ForeignHandler {\n      private lazy val adj: Map[WamTerm, Vector[WamTerm]] =\n        WamRuntime.collectBinarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        val visited = scala.collection.mutable.LinkedHashSet[WamTerm]()\n        val queue = scala.collection.mutable.Queue[WamTerm](source)\n        while (queue.nonEmpty) {\n          val node = queue.dequeue()\n          for (nb <- adj.getOrElse(node, Vector.empty) if !visited.contains(nb)) {\n            visited += nb\n            queue.enqueue(nb)\n          }\n        }\n        val sols = visited.toVector.map(t => Map(2 -> t))\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EdgeKeyLit]).

% transitive_distance3: tdist(Start, Target, Distance). BFS over the edge
% relation; each reachable node (excluding the source) is a solution
% binding register 2 = target and register 3 = the BFS shortest-path
% distance (IntTerm). Matches the Haskell/Rust/Elixir kernels, which
% return the shortest distance only (the Prolog source enumerates one
% dist+ (docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md): BFS shortest
% positive distance; each target once. Visited not seeded with source so
% Source appears only via self-loop / nonempty cycle. Inline kept (matches
% surrounding emit_scala_kernel_handler style).
emit_scala_kernel_handler(recursive_kernel(transitive_distance3, _Pred/3, ConfigOps), Code) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    format(atom(EdgeKey), '~w/2', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private lazy val adj: Map[WamTerm, Vector[WamTerm]] =\n        WamRuntime.collectBinarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        val dist = scala.collection.mutable.LinkedHashMap[WamTerm, Int]()\n        val seen = scala.collection.mutable.HashSet[WamTerm]()\n        val queue = scala.collection.mutable.Queue[(WamTerm, Int)]((source, 0))\n        while (queue.nonEmpty) {\n          val (node, d) = queue.dequeue()\n          for (nb <- adj.getOrElse(node, Vector.empty) if !seen.contains(nb)) {\n            seen += nb\n            dist(nb) = d + 1\n            queue.enqueue((nb, d + 1))\n          }\n        }\n        val sols = dist.toVector.map { case (t, dd) => Map(2 -> t, 3 -> IntTerm(dd)) }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EdgeKeyLit]).

% transitive_parent_distance4: pd(Start, Target, Parent, Distance). BFS over
% the edge relation; each reachable node (excluding the source) is a
% solution binding register 2 = target, register 3 = the immediate
% predecessor on the BFS shortest path, and register 4 = the distance.
% Matches the Haskell/Rust/Elixir kernels (shortest path only; see the
% transitive_distance3 note on path-length divergence).
emit_scala_kernel_handler(recursive_kernel(transitive_parent_distance4, _Pred/4, ConfigOps), Code) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    format(atom(EdgeKey), '~w/2', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private lazy val adj: Map[WamTerm, Vector[WamTerm]] =\n        WamRuntime.collectBinarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        // node -> (parent-on-shortest-path, distance)\n        val info = scala.collection.mutable.LinkedHashMap[WamTerm, (WamTerm, Int)]()\n        val seen = scala.collection.mutable.HashSet[WamTerm](source)\n        val queue = scala.collection.mutable.Queue[(WamTerm, Int)]((source, 0))\n        while (queue.nonEmpty) {\n          val (node, d) = queue.dequeue()\n          for (nb <- adj.getOrElse(node, Vector.empty) if !seen.contains(nb)) {\n            seen += nb\n            info(nb) = (node, d + 1)\n            queue.enqueue((nb, d + 1))\n          }\n        }\n        val sols = info.toVector.map { case (t, (p, dd)) => Map(2 -> t, 3 -> p, 4 -> IntTerm(dd)) }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EdgeKeyLit]).

% transitive_step_parent_distance5: tspd(Start, Target, Step, Parent, Distance).
% BFS over the edge relation; each reachable node (excluding the source) is
% a solution binding register 2 = target, register 3 = the FIRST hop from
% the source on the shortest path, register 4 = the immediate predecessor
% of the target, and register 5 = the distance. The first hop is the source's
% direct neighbour that begins the path (propagated through the BFS frontier);
% matches the Haskell/Rust/Elixir kernels.
emit_scala_kernel_handler(recursive_kernel(transitive_step_parent_distance5, _Pred/5, ConfigOps), Code) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    format(atom(EdgeKey), '~w/2', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private lazy val adj: Map[WamTerm, Vector[WamTerm]] =\n        WamRuntime.collectBinarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        // node -> (first-hop-from-source, parent-on-shortest-path, distance)\n        val info = scala.collection.mutable.LinkedHashMap[WamTerm, (WamTerm, WamTerm, Int)]()\n        val seen = scala.collection.mutable.HashSet[WamTerm](source)\n        // queue entries: (node, first-hop-step, distance)\n        val queue = scala.collection.mutable.Queue[(WamTerm, WamTerm, Int)]((source, source, 0))\n        while (queue.nonEmpty) {\n          val (node, step, d) = queue.dequeue()\n          for (nb <- adj.getOrElse(node, Vector.empty) if !seen.contains(nb)) {\n            seen += nb\n            val nbStep = if (node == source) nb else step\n            info(nb) = (nbStep, node, d + 1)\n            queue.enqueue((nb, nbStep, d + 1))\n          }\n        }\n        val sols = info.toVector.map { case (t, (st, p, dd)) => Map(2 -> t, 3 -> st, 4 -> p, 5 -> IntTerm(dd)) }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EdgeKeyLit]).

% category_ancestor: ca(Cat, Root, Hops, Visited). Depth-bounded DFS up the
% parent (edge) relation, returning one Hops solution per acyclic path from
% Cat to Root within max_depth, skipping nodes already in Visited (cycle
% break). Inputs: register 1 = cat, register 2 = root, register 4 = visited
% list; output: register 3 = hop count. max_depth is baked in from the
% kernel config. Mirrors the Haskell/Rust/Elixir reference (the base hop
% check is depth-unguarded; only the recursive descent is bounded, so the
% deepest hop found is max_depth + 1). Returns all paths' hop counts, so it
% agrees with the interpreter (no shortest-only collapsing).
emit_scala_kernel_handler(recursive_kernel(category_ancestor, _Pred/4, ConfigOps), Code) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    member(max_depth(MaxDepth), ConfigOps),
    integer(MaxDepth),
    format(atom(EdgeKey), '~w/2', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private val maxDepth: Int = ~w\n      private lazy val parents: Map[WamTerm, Vector[WamTerm]] =\n        WamRuntime.collectBinarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val cat = args(0)\n        val root = args(1)\n        val init = WamRuntime.wamListToVector(sharedProgram, args(3)).toSet\n        val hits = scala.collection.mutable.ArrayBuffer.empty[Int]\n        def go(acc: Int, c: WamTerm, depth: Int, visited: Set[WamTerm]): Unit = {\n          val ps = parents.getOrElse(c, Vector.empty)\n          val hop = acc + 1\n          for (p <- ps if p == root) hits += hop\n          if (depth < maxDepth)\n            for (mid <- ps if !visited.contains(mid))\n              go(hop, mid, depth + 1, visited + mid)\n        }\n        go(0, cat, init.size, init)\n        val sols = hits.toVector.map(h => Map(3 -> (IntTerm(h): WamTerm)))\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [MaxDepth, EdgeKeyLit]).

% weighted_shortest_path3: wsp(Start, Target, Weight). Dijkstra over a ternary
% weighted edge relation edge(From, To, W); each reachable node (excluding the
% source) is a solution binding register 2 = target and register 3 = the
% shortest total weight as a FloatTerm. Weights are read as Double (the
% register contract is output(3, float)), so use float-valued edge weights for
% the interpreter and kernel to agree exactly. Mirrors the Haskell/Rust/Elixir
% Dijkstra kernel (shortest weight only; like the distance kernels it returns
% one solution per node, so it agrees with the interpreter on graphs where
% each target has a single path).
emit_scala_kernel_handler(recursive_kernel(weighted_shortest_path3, _Pred/3, ConfigOps), Code) :-
    member(edge_pred(EdgePred/3), ConfigOps),
    format(atom(EdgeKey), '~w/3', [EdgePred]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private lazy val adj: Map[WamTerm, Vector[(WamTerm, Double)]] =\n        WamRuntime.collectTernarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(t => (t._2, WamRuntime.numericWeight(t._3))) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        val dist = scala.collection.mutable.HashMap[WamTerm, Double](source -> 0.0)\n        val pq = scala.collection.mutable.PriorityQueue.empty[(Double, WamTerm)](\n          Ordering.by[(Double, WamTerm), Double](_._1).reverse)\n        pq.enqueue((0.0, source))\n        val out = scala.collection.mutable.LinkedHashMap[WamTerm, Double]()\n        while (pq.nonEmpty) {\n          val (cost, node) = pq.dequeue()\n          val best = dist.getOrElse(node, Double.PositiveInfinity)\n          if (cost <= best) {\n            if (node != source && !out.contains(node)) out(node) = cost\n            for ((nxt, w) <- adj.getOrElse(node, Vector.empty)) {\n              val nc = cost + w\n              if (nc < dist.getOrElse(nxt, Double.PositiveInfinity)) {\n                dist(nxt) = nc\n                pq.enqueue((nc, nxt))\n              }\n            }\n          }\n        }\n        val sols = out.toVector.map { case (t, c) => Map(2 -> (t: WamTerm), 3 -> (FloatTerm(c): WamTerm)) }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EdgeKeyLit]).

% astar_shortest_path4: astar(Source, Target, Dim, Dist). Goal-directed A*
% over a ternary weighted edge relation, using a heuristic oracle
% (direct_dist_pred(Node, Target, H), falling back to the edges themselves).
% Priority f(n) = g(n)^D + h(n)^D where D is the Minkowski dimensionality
% taken from register 3 at runtime (config value baked in as the fallback).
% Inputs: register 1 = source, register 2 = bound target, register 3 = dim;
% output: register 4 = the shortest-path distance as a FloatTerm (at most one
% solution). Mirrors the Haskell/Rust/Elixir A* kernel. Float weights so
% interpreter and kernel agree exactly (see weighted_shortest_path3 note).
emit_scala_kernel_handler(recursive_kernel(astar_shortest_path4, _Pred/4, ConfigOps), Code) :-
    member(edge_pred(EdgePred/3), ConfigOps),
    member(direct_dist_pred(DistPred/DistArity), ConfigOps),
    member(dimensionality(ConfigDim), ConfigOps),
    integer(ConfigDim),
    format(atom(EdgeKey), '~w/3', [EdgePred]),
    format(atom(DistKey), '~w/~w', [DistPred, DistArity]),
    scala_string_literal(EdgeKey, EdgeKeyLit),
    scala_string_literal(DistKey, DistKeyLit),
    format(string(Code),
"new ForeignHandler {\n      private val configDim: Double = ~w.0\n      private lazy val adj: Map[WamTerm, Vector[(WamTerm, Double)]] =\n        WamRuntime.collectTernarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(t => (t._2, WamRuntime.numericWeight(t._3))) }\n      private lazy val heur: Map[WamTerm, Vector[(WamTerm, Double)]] =\n        WamRuntime.collectTernarySolutions(sharedProgram, ~w)\n          .groupBy(_._1).map { case (k, vs) => k -> vs.map(t => (t._2, WamRuntime.numericWeight(t._3))) }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val source = args(0)\n        val target = args(1)\n        val dim = args(2) match { case IntTerm(n) => n.toDouble; case FloatTerm(d) => d; case _ => configDim }\n        def heuristic(node: WamTerm): Double =\n          heur.getOrElse(node, Vector.empty).collectFirst { case (t, h) if t == target => h }.getOrElse(0.0)\n        val g = scala.collection.mutable.HashMap[WamTerm, Double](source -> 0.0)\n        val pq = scala.collection.mutable.PriorityQueue.empty[(Double, WamTerm)](\n          Ordering.by[(Double, WamTerm), Double](_._1).reverse)\n        pq.enqueue((math.pow(0.0, dim) + math.pow(heuristic(source), dim), source))\n        var result: Option[Double] = None\n        while (pq.nonEmpty && result.isEmpty) {\n          val (_f, node) = pq.dequeue()\n          if (node == target) result = g.get(node)\n          else {\n            val gn = g.getOrElse(node, Double.PositiveInfinity)\n            for ((nxt, w) <- adj.getOrElse(node, Vector.empty)) {\n              val newG = gn + w\n              if (newG < g.getOrElse(nxt, Double.PositiveInfinity)) {\n                g(nxt) = newG\n                pq.enqueue((math.pow(newG, dim) + math.pow(heuristic(nxt), dim), nxt))\n              }\n            }\n          }\n        }\n        result match {\n          case Some(d) => ForeignMulti(Vector(Map(4 -> (FloatTerm(d): WamTerm))))\n          case None    => ForeignFail\n        }\n      }\n    }",
           [ConfigDim, EdgeKeyLit, DistKeyLit]).

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
           "new ForeignHandler {\n      private def termKey(term: WamTerm): Option[String] = term match {\n        case Atom(id) if internTable.isInRange(id) => Some(internTable.stringOf(id))\n        case IntTerm(value) => Some(value.toString)\n        case FloatTerm(value) => Some(value.toString)\n        case _ => None\n      }\n      private val parentsByChild: Map[String, Vector[String]] = {\n        val src = scala.io.Source.fromFile(~w)\n        try {\n          src.getLines().toVector.flatMap { raw =>\n            val line = raw.trim\n            if (line.isEmpty || line.startsWith(\"#\")) None\n            else {\n              val parts = line.split(\"\\\\t\").map(_.trim).filter(_.nonEmpty).toVector\n              if (parts.length >= 2) Some(parts.head -> parts.tail) else None\n            }\n          }.toMap\n        } finally { src.close() }\n      }\n      private val allSols: Vector[Map[Int, WamTerm]] =\n        parentsByChild.toVector.flatMap { case (child, parents) =>\n          parents.map(parent => Map(1 -> parseFactArg(child), 2 -> parseFactArg(parent)))\n        }\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val sols = termKey(args(0)) match {\n          case Some(child) => parentsByChild.getOrElse(child, Vector.empty).map(parent => Map(1 -> parseFactArg(child), 2 -> parseFactArg(parent)))\n          case None => allSols\n        }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [PathLit]).
fact_source_spec_to_handler_code(Arity, lmdb(SpecOpts), Code) :-
    integer(Arity), Arity >= 2,
    !,
    is_list(SpecOpts),
    option(env_path(EnvPath), SpecOpts),
    atom_string(EnvPath, EnvPathStr),
    scala_string_literal(EnvPathStr, EnvPathLit),
    option(dbi(DbName), SpecOpts, ''),
    atom_string(DbName, DbNameStr),
    scala_string_literal(DbNameStr, DbNameLit),
    option(dupsort(Dupsort), SpecOpts, false),
    (Dupsort == true -> DupsortLit = "true" ; DupsortLit = "false"),
    % The ForeignHandler is a thin delegator: it owns one
    % LmdbFactSource (constructed at handler-instance init) and
    % dispatches to lookupByArg1 if arg1 is ground, streamAll
    % otherwise. Same shape as the Elixir adaptor's open/3 +
    % lookup_by_arg1/3 + stream_all/2 callback split. For arity > 2 the
    % LMDB value holds args 2..N tab-joined; LmdbFactSource splits them
    % back out into registers 2..N.
    format(string(Code),
           "new ForeignHandler {\n      private val source: LmdbFactSource = new LmdbFactSource(\n        envPath = ~w,\n        dbName  = ~w,\n        arity   = ~w,\n        dupsort = ~w,\n        internTable = internTable\n      )\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val sols = args(0) match {\n          case Atom(_) | IntTerm(_) | FloatTerm(_) => source.lookupByArg1(args(0))\n          case _                                   => source.streamAll()\n        }\n        if (sols.isEmpty) ForeignFail else ForeignMulti(sols)\n      }\n    }",
           [EnvPathLit, DbNameLit, Arity, DupsortLit]).

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
write_wam_scala_project(Predicates, Options1, ProjectDir) :-
    % Expand auto-detected graph kernels (opt-in: kernel_dispatch(true))
    % into foreign_predicates + scala_foreign_handlers entries, then expand
    % declarative fact sources. Both run before compilation so the rest of
    % the pipeline treats kernels/fact-sources as ordinary foreign preds.
    expand_kernels_in_options(Predicates, Options1, Options0),
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
    % --- Lowered per-predicate fast-path functions (emit_mode functions/mixed).
    %     Runs in the SAME intern-table session as the compile above so atom
    %     IDs match the shared instruction array. Must precede emit_scala_intern_table/1
    %     so any atom it interns is captured in the emitted seed array. --
    scala_generate_lowered(Predicates, Options, LoweredFunctionsBody, LoweredEntriesBody),
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
                         ForeignHandlersBody,
                         LoweredFunctionsBody, LoweredEntriesBody).

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

% WamRuntime is placed in the runtime_package, which GeneratedProgram imports
% (`import <runtime_package>.WamRuntime._`). Previously this used Package
% (the program's package) and ignored RuntimePkg, so whenever the two
% differed — including the DEFAULT options (core vs runtime) — the generated
% program failed to compile ("value runtime is not a member of ..."). All the
% smoke/kernel tests happened to pass the same package for both, hiding it.
write_runtime_source(ProjectDir, _Package, RuntimePkg) :-
    find_template('templates/targets/scala_wam/runtime.scala.mustache', Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    render_template(Template, ['package'=RuntimePkg, 'date'=DateStr], Content),
    scala_source_path(ProjectDir, RuntimePkg, 'WamRuntime', Path),
    make_directory_path_for(Path),
    write_file(Path, Content).

write_program_source(ProjectDir, Package, RuntimePkg,
                     InstrBody, LabelBody, DispatchBody,
                     WrapperCode, IdToStringStr,
                     ForeignHandlersBody,
                     LoweredFunctionsBody, LoweredEntriesBody) :-
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
          'foreign_handlers'=ForeignHandlersBody,
          'lowered_functions'=LoweredFunctionsBody,
          'lowered_entries'=LoweredEntriesBody
        ], Content),
    scala_source_path(ProjectDir, Package, 'GeneratedProgram', Path),
    make_directory_path_for(Path),
    write_file(Path, Content).

% ============================================================================
% EMIT MODE + LOWERED-FUNCTION GENERATION
% ============================================================================
% Mirrors the dual-mode lowering seam in the F#/Rust/Haskell targets.
%   emit_mode(interpreter)  — default; all predicates run in the step loop.
%   emit_mode(functions)    — every lowerable predicate also gets a native
%                             Scala fast-path function tried before the loop.
%   emit_mode(mixed([P/A])) — only the listed predicates are lowered.
% Resolution order: Options, then user:wam_scala_emit_mode/1, then default.

:- multifile user:wam_scala_emit_mode/1.

%% scala_resolve_emit_mode(+Options, -Mode)
scala_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  scala_validate_emit_mode(M0, Mode)
    ;   catch(user:wam_scala_emit_mode(M1), _, fail)
    ->  scala_validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

scala_validate_emit_mode(interpreter, interpreter) :- !.
scala_validate_emit_mode(functions,   functions)   :- !.
scala_validate_emit_mode(mixed(L),    mixed(L))    :- is_list(L), !.
scala_validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_scala_emit_mode, Other),
                scala_resolve_emit_mode/2)).

%% scala_partition_predicates(+Mode, +Predicates, -Interpreted, -Lowered)
scala_partition_predicates(interpreter, Predicates, Predicates, []) :- !.
scala_partition_predicates(functions, Predicates, Interp, Lowered) :- !,
    scala_partition_try_lower(Predicates, Interp, Lowered).
scala_partition_predicates(mixed(Hot), Predicates, Interp, Lowered) :- !,
    scala_partition_mixed(Predicates, Hot, Interp, Lowered).

scala_partition_try_lower([], [], []).
scala_partition_try_lower([P|Rest], Interp, Lowered) :-
    (   scala_predicate_is_lowerable(P)
    ->  Lowered = [P|LR], scala_partition_try_lower(Rest, Interp, LR)
    ;   Interp = [P|IR], scala_partition_try_lower(Rest, IR, Lowered)
    ).

scala_partition_mixed([], _, [], []).
scala_partition_mixed([P|Rest], Hot, Interp, Lowered) :-
    (   scala_indicator_in_list(P, Hot),
        scala_predicate_is_lowerable(P)
    ->  Lowered = [P|LR], scala_partition_mixed(Rest, Hot, Interp, LR)
    ;   Interp = [P|IR], scala_partition_mixed(Rest, Hot, IR, Lowered)
    ).

scala_indicator_in_list(P, L) :- memberchk(P, L), !.
scala_indicator_in_list(_:Pred/Arity, L) :- memberchk(Pred/Arity, L), !.

scala_predicate_is_lowerable(P) :-
    catch(scala_predicate_wamcode(P, WamCode), _, fail),
    catch(wam_scala_lowerable(P, WamCode, _), _, fail).

scala_predicate_wamcode(user:Pred/Arity, WamCode) :- !,
    compile_predicate_to_wam(Pred/Arity, [], WamCode).
scala_predicate_wamcode(Module:Pred/Arity, WamCode) :- !,
    compile_predicate_to_wam(Module:Pred/Arity, [], WamCode).
scala_predicate_wamcode(Pred/Arity, WamCode) :-
    compile_predicate_to_wam(Pred/Arity, [], WamCode).

%% scala_generate_lowered(+Predicates, +Options, -FunctionsBody, -EntriesBody)
%  Produces the Scala source for the lowered functions and the
%  loweredEntries Map body. Empty strings in interpreter mode so the
%  generated program is byte-identical to the pre-lowering output.
scala_generate_lowered(Predicates, Options, FunctionsBody, EntriesBody) :-
    scala_resolve_emit_mode(Options, Mode),
    (   Mode == interpreter
    ->  FunctionsBody = "", EntriesBody = ""
    ;   scala_partition_predicates(Mode, Predicates, _Interp, Lowered0),
        exclude(scala_pred_is_foreign(Options), Lowered0, Lowered),
        scala_lower_each(Lowered, FuncCodes, Entries),
        atomic_list_concat(FuncCodes, '\n', FunctionsBody),
        atomic_list_concat(Entries, ',\n', EntriesBody)
    ).

scala_pred_is_foreign(Options, P) :-
    ( P = _:Pred/Arity -> true ; P = Pred/Arity ),
    scala_foreign_predicate(Pred, Arity, Options).

scala_lower_each([], [], []).
scala_lower_each([P|Rest], [Code|Cs], [Entry|Es]) :-
    scala_predicate_wamcode(P, WamCode),
    lower_predicate_to_scala(P, WamCode, [], lowered(PredKey, FuncName, Code)),
    (   % T4 (multi_clause_n): the lowered function lowers EVERY clause and is
        % complete (first-solution, deterministic-prefix), so it never needs
        % the interpreter fallback — emit a direct entry. This is what makes
        % "the interpreter is never entered for the predicate" hold.
        catch(wam_scala_lowerable(P, WamCode, multi_clause_n), _, fail)
    ->  format(string(Entry),
            '    "~w" -> ((prog: WamProgram, args: Array[WamTerm]) => ~w(WamRuntime.newState(prog.dispatch("~w"), args), prog))',
            [PredKey, FuncName, PredKey])
    ;   % All other shapes: try the lowered fast path; on failure fall back to
        % a fresh interpreter run (a lowered `false` defers to the complete
        % step loop — e.g. T5's unbound-A1 case, or clause-2+ of multi_clause_1).
        format(string(Entry),
            '    "~w" -> ((prog: WamProgram, args: Array[WamTerm]) => { val startPc = prog.dispatch("~w"); if (~w(WamRuntime.newState(startPc, args), prog)) true else WamRuntime.runPredicate(prog, startPc, args) })',
            [PredKey, PredKey, FuncName])
    ),
    scala_lower_each(Rest, Cs, Es).

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
    % UTF-8 so the runtime/generated sources (which contain non-ASCII
    % characters) write correctly regardless of the process locale
    % (POSIX/ASCII in many CI containers); otherwise write/2 raises an
    % encoding error.
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        write(Stream, Content),
        close(Stream)
    ).

%% find_template(+RelPath, -Template) is det.
%  Locates a template file relative to the UnifyWeaver project root,
%  derived from THIS module's source location so it works regardless of
%  the current working directory. The previous version called
%  source_file(wam_scala_target, _) -- a bare module atom, which is a
%  predicate Head (wam_scala_target/0, undefined), so the lookup failed
%  and silently fell back to the cwd-relative path. It also walked up
%  only three directories (to .../src), not four (the repo root). Both
%  meant templates were found ONLY when the cwd was the repo root (e.g.
%  the conformance harness, cwd=tests/, could not build Scala at all).
find_template(RelPath, Template) :-
    (   source_file(find_template(_, _), SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),        % .../src/unifyweaver/targets
        file_directory_name(SrcDir, UWDir),          % .../src/unifyweaver
        file_directory_name(UWDir, SrcRoot),         % .../src
        file_directory_name(SrcRoot, ProjectRoot),   % .../  (repo root)
        atomic_list_concat([ProjectRoot, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, []).

% ============================================================================
% LOWERED EMITTER WIRING
% ============================================================================
% Loaded last so that, when wam_scala_lowered_emitter.pl re-imports the
% scala_lowered_* hook predicates from this module, they are already defined.
% The dependency is mutual (the emitter uses our hooks; we use its
% lower_predicate_to_scala/4 + wam_scala_lowerable/3), so load order matters.
:- use_module('../targets/wam_scala_lowered_emitter', [
       wam_scala_lowerable/3,
       lower_predicate_to_scala/4
   ]).

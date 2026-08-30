:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_javascript_target.pl - WAM-to-JavaScript (Node) Hybrid Transpilation Target
%
% Consumes the shared WAM bytecode front end
% (wam_target:compile_predicate_to_wam/3) and emits a Node.js project whose
% runtime executes the WAM instruction stream (the "interpreter" emit tier).
%
% Architecture mirrors the compact wam_lua_target: two files per project,
%   js/wam_runtime.js       -- the WAM virtual machine (static template)
%   js/generated_program.js -- the instruction vector + label table + CLI
% JavaScript is dynamically typed like Lua, so terms and instructions are
% plain tagged objects.
%
% The JS runtime honours all six WAM backend conventions
% (docs/WAM_BACKEND_CONVENTIONS.md); see runtime.js.mustache for details.
%
% Lowered/FFI emit tiers are intentionally out of scope for this baseline
% (WAMJS-1 research spike): only emit_mode(interpreter) is supported.

:- module(wam_javascript_target, [
    compile_wam_predicate_to_javascript/4,   % +Pred/Arity, +WamCode, +Options, -JSInstrFragment
    compile_wam_runtime_to_javascript/2,     % +Options, -RuntimeJSCode
    write_wam_javascript_project/3,          % +Predicates, +Options, +ProjectDir
    javascript_wam_resolve_emit_mode/2,      % +Options, -Mode
    js_reg_to_int/2,                         % +RegToken, -Int
    js_parse_functor_arity/3,                % +FunctorToken, -Name, -Arity
    js_intern_atom/2,                        % +AtomString, -Id
    init_js_atom_intern_table/0,
    js_string_literal/2                      % +Raw, -QuotedJSString
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [
    compile_predicate_to_wam_text/3
]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).

% ============================================================================
% Emit-mode resolution
% ============================================================================

:- multifile user:wam_javascript_emit_mode/1.

%% javascript_wam_resolve_emit_mode(+Options, -Mode)
%  Only the interpreter tier exists in this baseline. Anything else is a
%  domain error so a caller that asks for a not-yet-implemented tier fails
%  loudly rather than silently degrading.
javascript_wam_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_emit_mode(M0, Mode)
    ;   catch(user:wam_javascript_emit_mode(M1), _, fail)
    ->  validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_emit_mode(interpreter, interpreter) :- !.
validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_javascript_emit_mode, Other),
                javascript_wam_resolve_emit_mode/2)).

% ============================================================================
% Atom interning (shared between emitted constants and runtime list detection)
% ============================================================================

:- dynamic js_atom_id/2.
:- dynamic js_atom_next/1.

init_js_atom_intern_table :-
    retractall(js_atom_id(_, _)),
    retractall(js_atom_next(_)),
    % Seed the cons-cell atoms at fixed low ids. Order here defines the
    % emitted intern seed; the runtime rebuilds the same table from it.
    assertz(js_atom_id("[]", 0)),
    assertz(js_atom_id("[|]", 1)),
    assertz(js_atom_id(".", 2)),
    assertz(js_atom_next(3)).

js_intern_atom(Raw, Id) :-
    (   js_atom_next(_) -> true ; init_js_atom_intern_table ),
    text_to_str(Raw, Str),
    (   js_atom_id(Str, Id0)
    ->  Id = Id0
    ;   retract(js_atom_next(Next)),
        Id = Next,
        Next1 is Next + 1,
        assertz(js_atom_id(Str, Id)),
        assertz(js_atom_next(Next1))
    ).

emit_js_intern_seed(Code) :-
    findall(Id-Str, js_atom_id(Str, Id), Pairs0),
    sort(Pairs0, Pairs),
    maplist([_Id-Str, Line]>>(
        js_string_literal(Str, Lit),
        format(string(Line), '  ~w', [Lit])
    ), Pairs, Lines),
    atomic_list_concat(Lines, ',\n', Code).

% ============================================================================
% Token helpers
% ============================================================================

text_to_str(V, S) :-
    (   string(V) -> S = V
    ;   atom(V)   -> atom_string(V, S)
    ;   number(V) -> number_string(V, S)
    ;   term_string(V, S)
    ).

%% js_reg_to_int(+RegToken, -Int)  A_n -> n, X_n -> n+100, Y_n -> n+200
js_reg_to_int(Reg, Int) :-
    text_to_str(Reg, S),
    string_chars(S, [Prefix|NumChars]),
    string_chars(NumStr, NumChars),
    ( number_string(Num, NumStr) -> true ; Num = 0 ),
    (   Prefix == 'A' -> Int = Num
    ;   Prefix == 'X' -> Int is Num + 100
    ;   Prefix == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% js_parse_functor_arity(+FunctorToken, -Name, -Arity)
%  §2: the arity is the trailing /<digits>; the name is everything before it,
%  so ///2 parses as (//, 2) and //2 as (/, 2).
js_parse_functor_arity(FStr, Name, Arity) :-
    text_to_str(FStr, S),
    atom_string(FA, S),
    (   last_slash_index(FA, B),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, ArAtom),
        atom_number(ArAtom, Ar),
        integer(Ar)
    ->  sub_atom(FA, 0, B, _, NameAtom),
        atom_string(NameAtom, Name),
        Arity = Ar
    ;   Name = S, Arity = 0
    ).

last_slash_index(Atom, Index) :-
    findall(B, sub_atom(Atom, B, 1, _, '/'), Bs),
    Bs \= [],
    last(Bs, Index).

%% js_string_literal(+Raw, -Quoted): a double-quoted JS string literal.
js_string_literal(Raw, Quoted) :-
    text_to_str(Raw, S),
    string_chars(S, Chars),
    maplist(js_escape_char, Chars, EscLists),
    append(EscLists, EscChars),
    string_chars(Body, EscChars),
    format(string(Quoted), '"~w"', [Body]).

js_escape_char('\\', ['\\', '\\']) :- !.
js_escape_char('"', ['\\', '"']) :- !.
js_escape_char('\n', ['\\', 'n']) :- !.
js_escape_char('\t', ['\\', 't']) :- !.
js_escape_char('\r', ['\\', 'r']) :- !.
js_escape_char(C, [C]).

%% const_js(+ConstantToken, -JSLiteral): a tagged value object literal.
const_js(C, Lit) :-
    text_to_str(C, S),
    wam_classify_constant_token(S, Class),
    (   Class = integer(N)
    ->  format(string(Lit), '{tag:"int",val:~w}', [N])
    ;   Class = float(F)
    ->  format(string(Lit), '{tag:"float",val:~w}', [F])
    ;   Class = atom(Name)
    ->  js_intern_atom(Name, Id),
        format(string(Lit), '{tag:"atom",id:~w}', [Id])
    ;   % defensive: treat anything unclassified as an atom
        js_intern_atom(S, Id),
        format(string(Lit), '{tag:"atom",id:~w}', [Id])
    ).

% ============================================================================
% WAM text -> JS instruction literals
% ============================================================================

%% parse_wam_text_to_js(+Text, +PC0, -InstrLits, -LabelPairs, -PC1)
%  Tokenises each WAM text line. Labels map to the PC of the next emitted
%  instruction; every instruction occupies exactly one PC slot (§6).
parse_wam_text_to_js(Text, PC0, InstrLits, LabelPairs, PC1) :-
    text_to_str(Text, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, PC0, InstrLits, LabelPairs, PC1).

parse_lines([], PC, [], [], PC).
parse_lines([Line|Rest], PC0, Instrs, Labels, PC1) :-
    line_tokens(Line, Toks),
    (   Toks == []
    ->  parse_lines(Rest, PC0, Instrs, Labels, PC1)
    ;   is_label_tokens(Toks, LabelName)
    ->  Labels = [LabelName-PC0|LabelsRest],
        parse_lines(Rest, PC0, Instrs, LabelsRest, PC1)
    ;   item_to_js(Toks, Lit)
    ->  Instrs = [Lit|InstrsRest],
        PCn is PC0 + 1,
        parse_lines(Rest, PCn, InstrsRest, Labels, PC1)
    ;   % Unknown line -> real no-op (§6), preserving PC alignment.
        Instrs = ['{op:"NoOp"}'|InstrsRest],
        PCn is PC0 + 1,
        parse_lines(Rest, PCn, InstrsRest, Labels, PC1)
    ).

line_tokens(Line, Toks) :-
    split_string(Line, " \t", " \t", Parts0),
    exclude(==(""), Parts0, Parts1),
    maplist(strip_trailing_comma, Parts1, Toks).

strip_trailing_comma(Tok0, Tok) :-
    (   sub_string(Tok0, _, 1, 0, ",")
    ->  sub_string(Tok0, 0, _, 1, Tok)
    ;   Tok = Tok0
    ).

is_label_tokens([Tok], Name) :-
    sub_string(Tok, _, 1, 0, ":"),
    sub_string(Tok, 0, _, 1, Name0),
    atom_string(Name, Name0).

% --- instruction mapping ---------------------------------------------------

item_to_js(["get_constant", C, Reg], Lit) :- !,
    js_reg_to_int(Reg, R), const_js(C, T),
    format(string(Lit), '{op:"GetConstant",c:~w,ai:~w}', [T, R]).
item_to_js(["get_variable", X, A], Lit) :- !,
    js_reg_to_int(X, XI), js_reg_to_int(A, AI),
    format(string(Lit), '{op:"GetVariable",xn:~w,ai:~w}', [XI, AI]).
item_to_js(["get_value", X, A], Lit) :- !,
    js_reg_to_int(X, XI), js_reg_to_int(A, AI),
    format(string(Lit), '{op:"GetValue",xn:~w,ai:~w}', [XI, AI]).
item_to_js(["get_structure", F, Reg], Lit) :- !,
    js_reg_to_int(Reg, R), js_parse_functor_arity(F, Name, Arity),
    js_intern_atom(Name, Id),
    format(string(Lit), '{op:"GetStructure",fid:~w,ai:~w,arity:~w}', [Id, R, Arity]).
item_to_js(["get_list", Reg], Lit) :- !,
    js_reg_to_int(Reg, R), js_intern_atom("[|]", Id),
    format(string(Lit), '{op:"GetList",ai:~w,fid:~w,arity:2}', [R, Id]).
item_to_js(["put_constant", C, Reg], Lit) :- !,
    js_reg_to_int(Reg, R), const_js(C, T),
    format(string(Lit), '{op:"PutConstant",c:~w,ai:~w}', [T, R]).
item_to_js(["put_variable", X, A], Lit) :- !,
    js_reg_to_int(X, XI), js_reg_to_int(A, AI),
    format(string(Lit), '{op:"PutVariable",xn:~w,ai:~w}', [XI, AI]).
item_to_js(["put_value", X, A], Lit) :- !,
    js_reg_to_int(X, XI), js_reg_to_int(A, AI),
    format(string(Lit), '{op:"PutValue",xn:~w,ai:~w}', [XI, AI]).
item_to_js(["put_structure", F, Reg], Lit) :- !,
    js_reg_to_int(Reg, R), js_parse_functor_arity(F, Name, Arity),
    js_intern_atom(Name, Id),
    format(string(Lit), '{op:"PutStructure",fid:~w,ai:~w,arity:~w}', [Id, R, Arity]).
item_to_js(["put_list", Reg], Lit) :- !,
    js_reg_to_int(Reg, R), js_intern_atom("[|]", Id),
    format(string(Lit), '{op:"PutList",ai:~w,fid:~w,arity:2}', [R, Id]).
item_to_js(["set_variable", X], Lit) :- !,
    js_reg_to_int(X, XI), format(string(Lit), '{op:"SetVariable",xn:~w}', [XI]).
item_to_js(["set_value", X], Lit) :- !,
    js_reg_to_int(X, XI), format(string(Lit), '{op:"SetValue",xn:~w}', [XI]).
item_to_js(["set_constant", C], Lit) :- !,
    const_js(C, T), format(string(Lit), '{op:"SetConstant",c:~w}', [T]).
item_to_js(["unify_variable", X], Lit) :- !,
    js_reg_to_int(X, XI), format(string(Lit), '{op:"UnifyVariable",xn:~w}', [XI]).
item_to_js(["unify_value", X], Lit) :- !,
    js_reg_to_int(X, XI), format(string(Lit), '{op:"UnifyValue",xn:~w}', [XI]).
item_to_js(["unify_constant", C], Lit) :- !,
    const_js(C, T), format(string(Lit), '{op:"UnifyConstant",c:~w}', [T]).
item_to_js(["call", PredKey], Lit) :- !,
    js_parse_functor_arity(PredKey, _Name, Arity),
    js_string_literal(PredKey, P),
    format(string(Lit), '{op:"Call",pred:~w,arity:~w}', [P, Arity]).
item_to_js(["call", PredKey, ArityStr], Lit) :- !,
    number_string(Arity, ArityStr),
    js_string_literal(PredKey, P),
    format(string(Lit), '{op:"Call",pred:~w,arity:~w}', [P, Arity]).
item_to_js(["execute", PredKey], Lit) :- !,
    js_string_literal(PredKey, P),
    format(string(Lit), '{op:"Execute",pred:~w}', [P]).
item_to_js(["execute", PredKey, _ArityStr], Lit) :- !,
    js_string_literal(PredKey, P),
    format(string(Lit), '{op:"Execute",pred:~w}', [P]).
item_to_js(["proceed"], '{op:"Proceed"}') :- !.
item_to_js(["fail"], '{op:"Fail"}') :- !.
item_to_js(["allocate"], '{op:"Allocate"}') :- !.
item_to_js(["deallocate"], '{op:"Deallocate"}') :- !.
item_to_js(["try_me_else", L], Lit) :- !,
    js_string_literal(L, LS), format(string(Lit), '{op:"TryMeElse",label:~w}', [LS]).
item_to_js(["retry_me_else", L], Lit) :- !,
    js_string_literal(L, LS), format(string(Lit), '{op:"RetryMeElse",label:~w}', [LS]).
item_to_js(["trust_me"], '{op:"TrustMe"}') :- !.
item_to_js(["jump", L], Lit) :- !,
    js_string_literal(L, LS), format(string(Lit), '{op:"Jump",label:~w}', [LS]).
item_to_js(["builtin_call", PA, ArityStr], Lit) :- !,
    number_string(Arity, ArityStr),
    js_parse_functor_arity(PA, Name, _Ar),
    js_string_literal(Name, P),
    format(string(Lit), '{op:"BuiltinCall",pred:~w,arity:~w}', [P, Arity]).
item_to_js(["cut_ite"], '{op:"CutIte"}') :- !.
item_to_js(["get_level", Y], Lit) :- !,
    js_reg_to_int(Y, YI), format(string(Lit), '{op:"GetLevel",yn:~w}', [YI]).
item_to_js(["cut", Y], Lit) :- !,
    js_reg_to_int(Y, YI), format(string(Lit), '{op:"Cut",yn:~w}', [YI]).
% §6: first-argument indexing hints are optimisations only. Emitting a no-op
% (one slot) and letting the try_me_else chain run is always correct and keeps
% every later label PC aligned.
item_to_js([Op|_], '{op:"NoOp"}') :-
    sub_string(Op, 0, 9, _, "switch_on"), !.

% ============================================================================
% Per-predicate compile (helper; the project writer does full assembly)
% ============================================================================

%% compile_wam_predicate_to_javascript(+Pred/Arity, +WamCode, +Options, -JSInstrFragment)
%  Returns the comma-separated JS instruction literals for one predicate's
%  WAM code, starting at PC 1 (labels resolved by the project writer). Mainly
%  useful for inspection/testing a single predicate in isolation.
compile_wam_predicate_to_javascript(_Pred, WamCode, _Options, Fragment) :-
    parse_wam_text_to_js(WamCode, 1, InstrLits, _Labels, _PC1),
    maplist([I, Line]>>format(string(Line), '  ~w', [I]), InstrLits, IndentedLines),
    atomic_list_concat(IndentedLines, ',\n', Fragment).

% ============================================================================
% Runtime source
% ============================================================================

%% compile_wam_runtime_to_javascript(+Options, -RuntimeJSCode)
compile_wam_runtime_to_javascript(_Options, RuntimeJS) :-
    find_template('templates/targets/javascript_wam/runtime.js.mustache', Template),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(Template, ['date'=Date], RuntimeJS0),
    text_to_str(RuntimeJS0, RuntimeJS).

% ============================================================================
% Project assembly
% ============================================================================

%% write_wam_javascript_project(+Predicates, +Options, +ProjectDir)
write_wam_javascript_project(Predicates, Options, ProjectDir) :-
    javascript_wam_resolve_emit_mode(Options, interpreter),
    init_js_atom_intern_table,
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'js', JsDir),
    make_directory_path(JsDir),
    compile_predicates_for_project(Predicates, Options, 1,
                                   [], [], AllInstrs, AllLabels),
    emit_js_intern_seed(InternSeed),
    maplist([I, Line]>>format(string(Line), '  ~w', [I]), AllInstrs, InstrLines),
    atomic_list_concat(InstrLines, ',\n', InstrBody),
    maplist(label_entry_line, AllLabels, LabelLines),
    atomic_list_concat(LabelLines, ',\n', LabelBody),
    % runtime.js
    compile_wam_runtime_to_javascript(Options, RuntimeJS),
    directory_file_path(JsDir, 'wam_runtime.js', RuntimePath),
    write_text_file(RuntimePath, RuntimeJS),
    % generated_program.js
    find_template('templates/targets/javascript_wam/program.js.mustache', ProgTemplate),
    get_time(T), format_time(string(Date), "%Y-%m-%d", T),
    render_template(ProgTemplate,
        ['date'=Date,
         'intern_seed'=InternSeed,
         'instructions'=InstrBody,
         'labels'=LabelBody], ProgJS),
    directory_file_path(JsDir, 'generated_program.js', ProgPath),
    write_text_file(ProgPath, ProgJS).

label_entry_line(Name-PC, Line) :-
    js_string_literal(Name, NameLit),
    format(string(Line), '  ~w: ~w', [NameLit, PC]).

compile_predicates_for_project([], _Options, _PC,
                               InstrAcc, LabelAcc, InstrAcc, LabelAcc).
compile_predicates_for_project([Pred|Rest], Options, PC0,
                               InstrAcc, LabelAcc, AllInstrs, AllLabels) :-
    pred_indicator(Pred, PI),
    compile_predicate_wam_text(PI, Text),
    parse_wam_text_to_js(Text, PC0, PredInstrs, PredLabels, PC1),
    append(InstrAcc, PredInstrs, InstrAcc1),
    append(LabelAcc, PredLabels, LabelAcc1),
    compile_predicates_for_project(Rest, Options, PC1,
                                   InstrAcc1, LabelAcc1, AllInstrs, AllLabels).

pred_indicator(_M:N/A, N/A) :- !.
pred_indicator(N/A, N/A).

compile_predicate_wam_text(N/A, Text) :-
    (   compile_predicate_to_wam_text(N/A, [ite_use_y_level(true)], Text)
    ->  true
    ;   compile_predicate_to_wam_text(user:N/A, [ite_use_y_level(true)], Text)
    ->  true
    ;   throw(error(wam_compile_failed(N/A), write_wam_javascript_project/3))
    ).

write_text_file(Path, Content) :-
    setup_call_cleanup(open(Path, write, Stream, [encoding(utf8)]),
                       write(Stream, Content),
                       close(Stream)).

find_template(RelPath, Template) :-
    (   source_file(init_js_atom_intern_table, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),          % .../targets
        file_directory_name(SrcDir, UwDir),            % .../unifyweaver
        file_directory_name(UwDir, SrcRoot),           % .../src
        file_directory_name(SrcRoot, Root),            % repo root
        atomic_list_concat([Root, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ),
    read_file_to_string(AbsPath, Template, [encoding(utf8)]).

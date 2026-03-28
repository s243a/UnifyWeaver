:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025–2026 John William Creighton (@s243a)
%
% input_source.pl — Unified I/O abstraction for generated code.
%
% Input source is a compilation option, not hardcoded. The generated
% code's logic (BFS, recursion, etc.) is independent of the input
% source. Stdin is just fd 0 — all modes are "read from a source."
%
% Modes:
%   input(stdin)       — Read from standard input (default for CLI)
%   input(embedded)    — Embed asserted Prolog facts directly in code
%   input(file(Path))  — Read from a file path at runtime
%   input(vfs(Cell))   — Read from a NotebookVFS cell (for sciREPL)
%   input(function)    — Generate a function API (no I/O)

:- module(input_source, [
    resolve_input_mode/2,       % +Options, -Mode
    base_seed_code/4,           % +Target, +Module, +BasePred, -SeedCode
    input_default/2,            % +Context, -Mode
    lua_literal/2               % +Value, -Literal
]).

%% resolve_input_mode(+Options, -Mode)
%  Extract input mode from options, defaulting based on context.
resolve_input_mode(Options, Mode) :-
    (   member(input(M), Options)
    ->  Mode = M
    ;   member(context(Ctx), Options)
    ->  input_default(Ctx, Mode)
    ;   detect_context(DetectedCtx)
    ->  input_default(DetectedCtx, Mode)
    ;   Mode = stdin
    ).

%% detect_context(-Context)
%  Auto-detect the execution context. In swipl-wasm (sciREPL),
%  the nb_read/3 predicate is available from the prelude.
detect_context(notebook) :-
    predicate_property(user:nb_read(_,_,_), defined), !.
detect_context(cli).

%% input_default(+Context, -Mode)
%  Context-aware defaults (from design doc).
input_default(wasm, embedded).
input_default(notebook, embedded).
input_default(workbook, embedded).
input_default(browser, embedded).
input_default(test, embedded).
input_default(cli, stdin).
input_default(_, stdin).

%% base_seed_code(+Target, +Module, +BasePred, -SeedCode)
%  Extract base facts from the Prolog database and generate
%  target-language seed statements. Generalizes TypR's base_seed_code/3.
base_seed_code(Target, Module, BasePred, SeedCode) :-
    findall(From-To, (
        functor(H, BasePred, 2),
        clause(Module:H, true),
        H =.. [BasePred, From, To],
        nonvar(From), nonvar(To)
    ), AllPairs),
    sort(AllPairs, Pairs),  %% deduplicate
    findall(Statement, (
        member(From-To, Pairs),
        seed_statement(Target, BasePred, From, To, Statement)
    ), Statements),
    (   Statements = []
    ->  SeedCode = ""
    ;   atomic_list_concat(Statements, '\n', SeedCode)
    ).

%% seed_statement(+Target, +BasePred, +From, +To, -Statement)
%  Generate one seed statement in the target language.
seed_statement(lua, _BasePred, From, To, Statement) :-
    lua_literal(From, FL),
    lua_literal(To, TL),
    format(string(Statement), 'add_fact(~w, ~w)', [FL, TL]).

seed_statement(python, _BasePred, From, To, Statement) :-
    python_literal(From, FL),
    python_literal(To, TL),
    format(string(Statement), 'query.add_fact(~w, ~w)', [FL, TL]).

seed_statement(r, BasePred, From, To, Statement) :-
    r_literal(From, FL),
    r_literal(To, TL),
    format(string(Statement), 'add_~w(~w, ~w)', [BasePred, FL, TL]).

seed_statement(bash, _BasePred, From, To, Statement) :-
    bash_literal(From, FL),
    bash_literal(To, TL),
    format(string(Statement), 'add_fact ~w ~w', [FL, TL]).

seed_statement(ruby, _BasePred, From, To, Statement) :-
    ruby_literal(From, FL),
    ruby_literal(To, TL),
    format(string(Statement), 'add_fact(~w, ~w)', [FL, TL]).

seed_statement(perl, _BasePred, From, To, Statement) :-
    perl_literal(From, FL),
    perl_literal(To, TL),
    format(string(Statement), 'add_fact(~w, ~w);', [FL, TL]).

seed_statement(c, _BasePred, From, To, Statement) :-
    c_literal(From, FL),
    c_literal(To, TL),
    format(string(Statement), '    add_fact(~w, ~w);', [FL, TL]).

seed_statement(cpp, _BasePred, From, To, Statement) :-
    cpp_literal(From, FL),
    cpp_literal(To, TL),
    format(string(Statement), '    add_fact(~w, ~w);', [FL, TL]).

seed_statement(rust, _BasePred, From, To, Statement) :-
    rust_literal(From, FL),
    rust_literal(To, TL),
    format(string(Statement), '    add_fact(~w, ~w);', [FL, TL]).

seed_statement(go, _BasePred, From, To, Statement) :-
    go_literal(From, FL),
    go_literal(To, TL),
    format(string(Statement), '\taddFact(~w, ~w)', [FL, TL]).

seed_statement(typescript, _BasePred, From, To, Statement) :-
    js_literal(From, FL),
    js_literal(To, TL),
    format(string(Statement), 'addFact(~w, ~w);', [FL, TL]).

seed_statement(javascript, _BasePred, From, To, Statement) :-
    js_literal(From, FL),
    js_literal(To, TL),
    format(string(Statement), 'addFact(~w, ~w);', [FL, TL]).

seed_statement(kotlin, _BasePred, From, To, Statement) :-
    format(string(Statement), '    addFact("~w", "~w")', [From, To]).

seed_statement(scala, _BasePred, From, To, Statement) :-
    format(string(Statement), '    addFact("~w", "~w")', [From, To]).

seed_statement(clojure, _BasePred, From, To, Statement) :-
    format(string(Statement), '(add-fact "~w" "~w")', [From, To]).

seed_statement(jython, _BasePred, From, To, Statement) :-
    format(string(Statement), 'add_fact("~w", "~w")', [From, To]).

seed_statement(elixir, _BasePred, From, To, Statement) :-
    format(string(Statement), '    add_fact.({"~w", "~w"})', [From, To]).

seed_statement(fsharp, _BasePred, From, To, Statement) :-
    format(string(Statement), '    addFact "~w" "~w"', [From, To]).

seed_statement(haskell, _BasePred, From, To, Statement) :-
    format(string(Statement), '    addFact "~w" "~w"', [From, To]).

seed_statement(powershell, _BasePred, From, To, Statement) :-
    format(string(Statement), 'Add-Fact "~w" "~w"', [From, To]).

seed_statement(jamaica, _BasePred, From, To, Statement) :-
    format(string(Statement),
        '    ldc "~w"\n    ldc "~w"\n    invokestatic addFact(String, String)void',
        [From, To]).

seed_statement(krakatau, _BasePred, From, To, Statement) :-
    format(string(Statement),
        '    ldc "~w"\n    ldc "~w"\n    invokestatic AncestorQuery addFact (Ljava/lang/String;Ljava/lang/String;)V',
        [From, To]).

seed_statement(awk, _BasePred, From, To, Statement) :-
    format(string(Statement), '    add_fact("~w", "~w")', [From, To]).

seed_statement(vbnet, _BasePred, From, To, Statement) :-
    format(string(Statement), '        query.AddFact("~w", "~w")', [From, To]).

bash_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

%% --- Target-specific literal quoting ---

lua_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

python_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

r_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

ruby_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

perl_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

c_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

cpp_literal(V, L) :- c_literal(V, L).
rust_literal(V, L) :- c_literal(V, L).

go_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

js_literal(Value, Literal) :-
    (   number(Value)
    ->  format(string(Literal), '~w', [Value])
    ;   format(string(Literal), '"~w"', [Value])
    ).

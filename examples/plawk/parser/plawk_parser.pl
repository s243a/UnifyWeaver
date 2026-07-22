% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_parser, [
    plawk_parse_string/2,
    plawk_parse_source/3
]).

%% plawk_parse_string(+Source, -Program) is semidet.
%
%  Parse the first Phase-2 surface slice:
%
%      /^PREFIX/ { print $0 }
%      /LITERAL/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%      $N == "ERROR" { errors++ } $N == "WARN" { warnings++ } END { print errors, warnings }
%      { counts[$1]++ } END { print counts["ERROR"], counts["WARN"] }
%      BEGIN { print "kind", "count" } { count++ } END { print "count", count }
%      BEGIN { FS = ":" } $1 == "ERROR" { counts[$2]++ } END { print counts["disk"] }
%      BEGIN { FS = ":"; OFS = "," } $1 == "ERROR" { print $2, $3 }
%      $1 == "ERROR" { printf "%s=%s\n", $2, $3 }
%      { count++ } END { print "count", count }
%      $3 > 100 { big++ } END { print big }
%      $1 == "ERROR" { print $3, int($3) }
%      $1 == "ERROR" { print int($3) + 1 }
%      $1 == "ERROR" { print int($3) - 1 }
%      $1 == "ERROR" { print NR - 1, NF + 1, length($0) - 3, index($2, "sk") + 1 }
%      $1 == "ERROR" { bytes += $3; last = $3 } END { print bytes, last }
%      $1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }
%      { last_pos = index($2, "sk") + 1; total_pos += index($0, "disk") - 1 } END { print last_pos, total_pos }
%      { adjusted += length($0) - 3; width = NF; fields += NF } END { print adjusted, width, fields }
%      { last = NR; prev = NR - 1; total += NR + 1 } END { print last, prev, total }
%      $1 == "DEBUG" { skipped++; next } { total++ } END { print total, skipped }
%      $1 == "ERROR" { hits++; break } { total++ } END { print hits, total }
%      $1 == "ERROR" { last_len = length($0); hits++ } END { print hits, last_len }
%      { if ($1 == "ERROR") { errors++ } else { warnings++ } } END { print errors, warnings }
%
%  The AST is deliberately small and explicit so later syntax can extend it
%  without changing the native codegen contract.
plawk_parse_string(Source, Program) :-
    plawk_parse_source(Source, Program, _PrologClauses).

%% plawk_parse_source(+Source, -Program, -PrologClauses) is semidet.
%
%  Like plawk_parse_string/2, but also lifts embedded Prolog blocks:
%
%      @prolog
%      weight(I, F, R) :- R is I * F.
%      @end
%
%  Block markers sit alone on their line (leading/trailing blanks ok).
%  A heredoc-style tag makes the fence unambiguous when the Prolog
%  text itself contains an @end-shaped line: `@prolog-TAG` closes only
%  at `@end-TAG` with the exact same tag. Blocks may appear anywhere
%  between top-level program parts; their text never routes through
%  the awk grammar -- it is term-read as ordinary Prolog and returned
%  as PrologClauses for the compile driver to hand to
%  write_wam_llvm_project alongside the program.
plawk_parse_source(Source, Program, PrologClauses) :-
    string(Source),
    plawk_split_prolog_blocks(Source, Stripped, BlockTexts),
    maplist(plawk_read_block_clauses, BlockTexts, ClausesNested),
    append(ClausesNested, BlockClauses),
    string_codes(Stripped, Codes),
    phrase(plawk_program(Program0, FunctionClauses, DynEntries), Codes),
    plawk_normalise_c_for(Program0, Program1),
    plawk_normalise_getline_loops(Program1, Program),
    append(BlockClauses, FunctionClauses, PrologClauses),
    plawk_dynentry_reserved_check(DynEntries, PrologClauses).

%% plawk_normalise_c_for(+Program0, -Program)
%
%  Desugar a C-style `for (INIT; COND; UPDATE) BODY` (parsed to
%  c_for(Init, Cond, Update, Body)) into `INIT; while (COND) { BODY; UPDATE }`,
%  reusing the while runtime. Applied before the getline normalisation so a
%  getline loop nested in a for body lands in the resulting while_loop body
%  (which the getline walker recurses).
%
%  `break` desugars correctly (it exits the while == exits the for). `continue`
%  does NOT: the while runtime's continue jumps to the condition test, skipping
%  the appended UPDATE -- for a counter loop that is an infinite loop. So a
%  c_for whose body has an own-level `continue` (directly, or inside an if, but
%  not inside a nested loop that consumes its own continue) is left un-desugared;
%  codegen then rejects it cleanly rather than miscompiling. `continue` in a
%  C-style for is a follow-on.
plawk_normalise_c_for(program(Begin, Rules0, End), program(Begin, Rules, End)) :-
    !,
    plawk_norm_cfor_rules(Rules0, Rules).
plawk_normalise_c_for(Program, Program).

plawk_norm_cfor_rules(Rules0, Rules) :-
    is_list(Rules0),
    !,
    maplist(plawk_norm_cfor_rule, Rules0, Rules).
plawk_norm_cfor_rules(Other, Other).

plawk_norm_cfor_rule(rule(Pattern, Actions0), rule(Pattern, Actions)) :-
    !,
    plawk_norm_cfor_actions(Actions0, Actions).
plawk_norm_cfor_rule(Other, Other).

plawk_norm_cfor_actions([], []).
plawk_norm_cfor_actions([c_for(Init, Cond, Update, Body0) | Rest], Out) :-
    \+ plawk_actions_own_continue(Body0),
    !,
    plawk_norm_cfor_actions(Body0, Body1),
    plawk_norm_cfor_action(Init, InitN),
    plawk_norm_cfor_action(Update, UpdateN),
    append(Body1, [UpdateN], LoopBody),
    plawk_norm_cfor_actions(Rest, RestN),
    Out = [InitN, while_loop(Cond, LoopBody) | RestN].
plawk_norm_cfor_actions([Action0 | Rest], [Action | RestN]) :-
    plawk_norm_cfor_action(Action0, Action),
    plawk_norm_cfor_actions(Rest, RestN).

plawk_norm_cfor_action(if(Pattern, Then0, Else0), if(Pattern, Then, Else)) :-
    !,
    plawk_norm_cfor_actions(Then0, Then),
    plawk_norm_cfor_actions(Else0, Else).
plawk_norm_cfor_action(while_loop(Cond, Body0), while_loop(Cond, Body)) :-
    !,
    plawk_norm_cfor_actions(Body0, Body).
plawk_norm_cfor_action(do_while_loop(Body0, Cond), do_while_loop(Body, Cond)) :-
    !,
    plawk_norm_cfor_actions(Body0, Body).
plawk_norm_cfor_action(foreach_loop(Layout, Body0), foreach_loop(Layout, Body)) :-
    !,
    plawk_norm_cfor_actions(Body0, Body).
plawk_norm_cfor_action(for_in(Var, Array, Body0), for_in(Var, Array, Body)) :-
    !,
    plawk_norm_cfor_actions(Body0, Body).
% A nested c_for that itself carries an own-level continue: keep the node (so
% codegen rejects) but still normalise its body so inner desugarable loops work.
plawk_norm_cfor_action(c_for(Init, Cond, Update, Body0), c_for(Init, Cond, Update, Body)) :-
    !,
    plawk_norm_cfor_actions(Body0, Body).
plawk_norm_cfor_action(Action, Action).

%% plawk_actions_own_continue(+Actions) is semidet.
%
%  True if a `continue` targets THIS loop level -- a continue directly in the
%  action list, or inside an if branch (which does not consume continue), but
%  NOT one inside a nested loop (while/do-while/for-in/c_for), which consumes
%  its own continue.
plawk_actions_own_continue(Actions) :-
    member(Action, Actions),
    plawk_action_own_continue(Action).

plawk_action_own_continue(continue).
plawk_action_own_continue(if(_Pattern, Then, Else)) :-
    ( plawk_actions_own_continue(Then)
    ; plawk_actions_own_continue(Else)
    ).

%% plawk_normalise_getline_loops(+Program0, -Program)
%
%  Desugar the supported redirected, main-input, and command-pipe getline
%  conditions into the existing while machinery. Each becomes a priming
%  status-capturing getline plus a `while (status > 0)` whose body ends in
%  another read of the same form.
%  `$getline_status` is a reserved i64 slot (users cannot write a `$`-prefixed
%  name). v1 rewrites rule bodies (and nested if/loop bodies); getline loops in
%  BEGIN/END/case blocks remain explicit unsupported nodes.
plawk_normalise_getline_loops(program(Begin, Rules0, End),
        program(Begin, Rules, End)) :-
    !,
    plawk_norm_getline_rules(Rules0, Rules).
% Any other top-level program shape (program_passes for `pass ... of ... as`,
% and the like) has no plain rule bodies to rewrite; pass it through unchanged.
% (getline inside a pass block is a follow-on.)
plawk_normalise_getline_loops(Program, Program).

plawk_norm_getline_rules(Rules0, Rules) :-
    is_list(Rules0),
    !,
    maplist(plawk_norm_getline_rule, Rules0, Rules).
plawk_norm_getline_rules(Other, Other).

plawk_norm_getline_rule(rule(Pattern, Actions0), rule(Pattern, Actions)) :-
    !,
    plawk_norm_getline_actions(Actions0, Actions).
plawk_norm_getline_rule(Other, Other).

plawk_norm_getline_actions([], []).
plawk_norm_getline_actions([getline_file_record_loop(File, Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_file_record_capture('$getline_status', File),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([getline_loop(Var, File, Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_capture('$getline_status', Var, File),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([getline_main_record_loop(Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_main_record_capture('$getline_status'),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([getline_main_var_loop(Var, Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_main_var_capture('$getline_status', Var),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([getline_pipe_record_loop(Command, Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_pipe_record_capture('$getline_status', Command),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([getline_pipe_loop(Var, Command, Body0) | Rest], Out) :-
    !,
    plawk_norm_getline_actions(Body0, Body),
    Read = getline_pipe_capture('$getline_status', Var, Command),
    append(Body, [Read], LoopBody),
    plawk_norm_getline_actions(Rest, Rest1),
    Out = [ Read,
            while_loop(cmp(var('$getline_status'), gt, int(0)), LoopBody)
          | Rest1 ].
plawk_norm_getline_actions([Action0 | Rest], [Action | Rest1]) :-
    plawk_norm_getline_action(Action0, Action),
    plawk_norm_getline_actions(Rest, Rest1).

plawk_norm_getline_action(if(Pattern, Then0, Else0), if(Pattern, Then, Else)) :-
    !,
    plawk_norm_getline_actions(Then0, Then),
    plawk_norm_getline_actions(Else0, Else).
plawk_norm_getline_action(while_loop(Cond, Body0), while_loop(Cond, Body)) :-
    !,
    plawk_norm_getline_actions(Body0, Body).
plawk_norm_getline_action(do_while_loop(Body0, Cond), do_while_loop(Body, Cond)) :-
    !,
    plawk_norm_getline_actions(Body0, Body).
plawk_norm_getline_action(foreach_loop(Layout, Body0), foreach_loop(Layout, Body)) :-
    !,
    plawk_norm_getline_actions(Body0, Body).
plawk_norm_getline_action(Action, Action).

%% plawk_dynentry_reserved_check(+DynEntries, +PrologClauses)
%
%  DYNENTRY reserves a name for the compiled object -- declaring one
%  that an @prolog block or `function` definition also defines, or a
%  surface builtin/keyword name, throws (a parse error at the CLI):
%  never silent shadowing in either direction.
plawk_dynentry_reserved_check([], _) :- !.
plawk_dynentry_reserved_check(DynEntries, PrologClauses) :-
    findall(Defined,
        ( member(Clause, PrologClauses),
          plawk_clause_head_name(Clause, Defined)
        ),
        DefinedNames),
    forall(member(Name, DynEntries),
        (   memberchk(Name, DefinedNames)
        ->  throw(error(plawk_dynentry_reserved(Name),
                context(plawk_parse_source/3,
                    'DYNENTRY name is also defined by an @prolog block or function -- reservation forbids shadowing')))
        ;   plawk_surface_reserved_name(Name)
        ->  throw(error(plawk_dynentry_reserved(Name),
                context(plawk_parse_source/3,
                    'DYNENTRY name collides with a surface builtin/keyword')))
        ;   true
        )).

plawk_clause_head_name((Head :- _Body), Name) :-
    !,
    functor(Head, Name, _).
plawk_clause_head_name(Head, Name) :-
    compound(Head),
    functor(Head, Name, _).

plawk_surface_reserved_name(Name) :-
    memberchk(Name, [length, index, substr, int, tolower, toupper,
        float, blob, dyncall, dyncall_at, compile, compile_file,
        print, printf, function, for, in, if, else, next, break]).

plawk_split_prolog_blocks(Source, Stripped, BlockTexts) :-
    split_string(Source, "\n", "", Lines),
    plawk_split_block_lines(Lines, KeptLines, BlockTexts),
    atomic_list_concat(KeptLines, '\n', StrippedAtom),
    atom_string(StrippedAtom, Stripped).

plawk_split_block_lines([], [], []).
plawk_split_block_lines([Line | Lines], KeptLines, [BlockText | BlockTexts]) :-
    plawk_prolog_open_marker(Line, EndMarker),
    !,
    plawk_take_block_lines(Lines, EndMarker, BlockLines, Rest),
    atomic_list_concat(BlockLines, '\n', BlockAtom),
    atom_string(BlockAtom, BlockText),
    plawk_split_block_lines(Rest, KeptLines, BlockTexts).
plawk_split_block_lines([Line | Lines], [Line | KeptLines], BlockTexts) :-
    plawk_split_block_lines(Lines, KeptLines, BlockTexts).

plawk_prolog_open_marker(Line, EndMarker) :-
    split_string(Line, "", " \t", [Trimmed]),
    ( Trimmed == "@prolog"
    ->  EndMarker = "@end"
    ;   string_concat("@prolog-", Tag, Trimmed),
        Tag \== "",
        string_concat("@end-", Tag, EndMarker)
    ).

% an unterminated block fails the parse (no clause for [])
plawk_take_block_lines([Line | Lines], EndMarker, BlockLines, Rest) :-
    split_string(Line, "", " \t", [Trimmed]),
    ( Trimmed == EndMarker
    ->  BlockLines = [],
        Rest = Lines
    ;   BlockLines = [Line | BlockLines1],
        plawk_take_block_lines(Lines, EndMarker, BlockLines1, Rest)
    ).

plawk_read_block_clauses(BlockText, Clauses) :-
    setup_call_cleanup(
        open_string(BlockText, Stream),
        plawk_read_stream_clauses(Stream, Clauses),
        close(Stream)).

plawk_read_stream_clauses(Stream, Clauses) :-
    read_term(Stream, Term, []),
    ( Term == end_of_file
    ->  Clauses = []
    ;   Clauses = [Term | Rest],
        plawk_read_stream_clauses(Stream, Rest)
    ).

plawk_program(Program) -->
    plawk_program(Program, _FunctionClauses).

plawk_program(Program, FunctionClauses) -->
    plawk_program(Program, FunctionClauses, _DynEntries).

% Multi-pass form (PLAWK_MULTIPASS_CACHE.md, phase 2): one or more explicit
% `pass { ... }` blocks between BEGIN and END, each a full record loop with
% its own actions. Produces program_passes(Begin, [pass(Rules), ...], End).
% Tried before the single implicit-main form; the `pass` keyword is
% unambiguous, and a program with no `pass` block falls through (the guard
% requires at least one) to the ordinary program(...) clause.
plawk_program(program_passes(BeginClauses, Passes, EndClauses),
        FunctionClauses, DynEntries) -->
    ws,
    begin_clauses(BeginClauses),
    function_defs(FunctionClauses),
    dynentry_decls(DynEntries),
    pass_clauses(Passes0),
    { Passes0 = [_ | _] },
    end_clauses(EndClauses0),
    eos,
    { maplist(plawk_pass_dynentry_rewrite(DynEntries), Passes0, Passes),
      plawk_dynentry_rewrite_all(EndClauses0, DynEntries, EndClauses)
    }.
plawk_program(program(BeginClauses, Rules, EndClauses), FunctionClauses,
        DynEntries) -->
    ws,
    begin_clauses(BeginClauses),
    function_defs(FunctionClauses),
    dynentry_decls(DynEntries),
    program_rules(Rules0),
    end_clauses(EndClauses0),
    eos,
    { plawk_dynentry_rewrite_all(Rules0, DynEntries, Rules),
      plawk_dynentry_rewrite_all(EndClauses0, DynEntries, EndClauses)
    }.

% A `pass records of TABLE as VAR { print VAR["col"], ... }` block: the safe,
% named row reader (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.3). Iterates
% TABLE's row entries, binding each row to VAR; columns are addressed BY NAME
% only (`VAR["col"]`), resolved through TABLE's declared schema. Distinct from
% `pass over ... as k` (which binds a scalar key). Tried before `over` and the
% plain `pass`; the `records of` keyword pair is unambiguous. Body reuses the
% for-in print body, so numeric column addressing (`VAR[1]`) never appears
% here -- that is the positional `rows of` reader's job.
pass_clauses([pass_records(var(Var), var(Table), Body) | Rest]) -->
    "pass", identifier_boundary, ws,
    "records", identifier_boundary, ws,
    "of", identifier_boundary, ws,
    table_ident(Table), ws,
    "as", identifier_boundary, ws,
    identifier(Var), ws,
    for_in_body(Body),
    ws,
    pass_clauses(Rest).
% A `pass rows of TABLE as VAR { print VAR[N], ... }` block: the POSITIONAL
% row reader (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.5). Like `records of`,
% but columns are addressed BY POSITION (`VAR[1]`, 1-indexed) instead of by
% schema name -- for raw / schema-less stores. Tried before `over` / plain
% `pass`; the `rows of` keyword pair is unambiguous. (The `unsafe` modifier
% and an inline check-or-rename spec are a follow-on; core is positional.)
pass_clauses([pass_rows(var(Var), var(Table), Body) | Rest]) -->
    "pass", identifier_boundary, ws,
    "rows", identifier_boundary, ws,
    "of", identifier_boundary, ws,
    table_ident(Table), ws,
    "as", identifier_boundary, ws,
    identifier(Var), ws,
    for_in_body(Body),
    ws,
    pass_clauses(Rest).
% `pass rows of TABLE { print $1, $2 }` -- the positional reader WITHOUT an
% `as VAR` binding uses awk-native field addressing: a stored row is a
% field-separated record, so `$N` addresses its Nth column directly. Tried
% after the `as` form (which owns `VAR[N]`); the two spellings do not mix --
% no `as` => `$N`, `as VAR` => `VAR[N]`. Represented as pass_rows_anon/2.
pass_clauses([pass_rows_anon(var(Table), Body) | Rest]) -->
    "pass", identifier_boundary, ws,
    "rows", identifier_boundary, ws,
    "of", identifier_boundary, ws,
    table_ident(Table), ws,
    for_in_body(Body),
    ws,
    pass_clauses(Rest).
% A `pass over TABLE as VAR { print ... }` block: a configurable reader
% (PLAWK_MULTIPASS_CACHE.md phase 4). Instead of re-scanning the input, the
% pass iterates TABLE's entries as records, binding each key to VAR -- the
% "process what a previous pass stored" shape. The body reuses the for-in
% print body (key / TABLE[VAR] / literal). Tried before the plain `pass`
% (the `over` keyword is unambiguous). Represented as pass_over/3 so the
% driver can emit a table-iterating pass function rather than an input scan.
% A `pass over query(PRED(V1, ..., Vn)) { print $1, ... }` block: the
% query-driven reader (PLAWK_MULTIPASS_CACHE.md §3.4, phase 6;
% PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md). Each SOLUTION of the goal is a
% record; the goal's argument variables (V1..Vn) are its output fields, mapped
% positionally to `$1..$n`. Parses to pass_query(query(Pred, Vars), Body). Tried
% before `pass over TABLE` (the `(` after the predicate name distinguishes a
% query from a bare table name). The codegen surface is staged (PR 1 diagnoses
% it; the runtime materialisation lands later).
pass_clauses([pass_query(query(Pred, Vars), Body) | Rest]) -->
    "pass", identifier_boundary, ws,
    "over", identifier_boundary, ws,
    "query", ws, "(", ws,
    identifier(Pred), ws, "(", ws, query_var_list(Vars), ws, ")", ws,
    ")", ws,
    for_in_body(Body),
    ws,
    pass_clauses(Rest).
pass_clauses([pass_over(var(Var), var(Table), Body) | Rest]) -->
    "pass", identifier_boundary, ws,
    "over", identifier_boundary, ws,
    table_ident(Table), ws,
    "as", identifier_boundary, ws,
    identifier(Var), ws,
    for_in_body(Body),
    ws,
    pass_clauses(Rest).

% The output-variable list of a query goal: one or more identifiers. Their
% NAMES are placeholders; their POSITION is what maps a solution's bindings to
% `$1..$n`.
query_var_list([V | Vs]) -->
    identifier(V),
    query_var_list_rest(Vs).
query_var_list_rest([V | Vs]) -->
    ws, ",", ws, identifier(V),
    !,
    query_var_list_rest(Vs).
query_var_list_rest([]) -->
    [].
% A `gen over query(SRC(A, ...)) as (a, ...) { BODY } as name` generator with an
% input iterator (PLAWK_GENERATOR_BLOCKS.md, PR 3 + views/PR 5): a producer that
% transforms/projects another relation. Each solution of the source goal binds
% the loop vars positionally; the body emits a projection of them. Parses to
% gen_block(name(Name), over(query(Src, SrcVars), LoopVars), Body) where LoopVars
% is a list (`as v` -> [v], `as (a, b)` -> [a, b]). A single-column source keeps
% the `as v` spelling; a multi-column source binds each column and the body may
% emit a subset (a column-projection view -- "don't materialise the whole row").
% Tried before the pure `gen { ... }` form. The body is emit-based (`emit a`,
% `emit (a, b)`, or `if (a CMP int) emit a` to filter).
pass_clauses([gen_block(name(Name),
        over(query(Src, SrcVars), LoopVars), Body) | Rest]) -->
    "gen", identifier_boundary, ws,
    "over", identifier_boundary, ws,
    "query", ws, "(", ws,
    identifier(Src), ws, "(", ws, query_var_list(SrcVars), ws, ")", ws,
    ")", ws,
    "as", identifier_boundary, ws, gen_over_binding(LoopVars), ws,
    gen_over_body(Body), ws,
    "as", identifier_boundary, ws, identifier(Name), ws,
    pass_clauses(Rest).
% A `gen { BODY } as name` generator block (PLAWK_GENERATOR_BLOCKS.md): a
% producer whose `emit E` statements define a relation `name/1`, callable from
% a Prolog goal (e.g. a query pass's `over query(name(X))`). The producer dual
% of the query reader. Parses to gen_block(name(Name), none, Body) (`none` =
% no input source, distinguishing it from the `gen over ...` input iterator).
% A pure block of constant emits compiles to facts; the `gen` keyword is
% unambiguous (distinct from `pass`); tried after the `gen over` form and
% before the plain `pass` clause.
pass_clauses([gen_block(name(Name), none, Body) | Rest]) -->
    "gen", identifier_boundary, ws,
    action_block(Body), ws,
    "as", identifier_boundary, ws,
    identifier(Name), ws,
    pass_clauses(Rest).

% The body of an input-iterator generator: an `emit E` (transform each source
% solution), optionally gated by an `if (COND)` reader guard (filter). Mirrors
% for_in_body's if-print shape, but emit-based -- the guard uses the same
% guard_expr (so `if (v > 3)` compares the bound var `v` via forin_key_cmp).
gen_over_body([if(Guard, [emit(Emit)], [])]) -->
    "{", ws, "if", ws, "(", ws, guard_expr(Guard), ws, ")", ws,
    emit_action(emit(Emit)), ws, "}",
    !.
gen_over_body([emit(Emit)]) -->
    "{", ws, emit_action(emit(Emit)), ws, "}".

% The loop-variable binding of an input iterator: `as v` binds one column,
% `as (a, b, ...)` binds several (one per source column, positionally). Both
% produce a list of variable names.
gen_over_binding(Vars) -->
    "(", ws, query_var_list(Vars), ws, ")",
    !.
gen_over_binding([V]) -->
    identifier(V).
% A `materialize NAME` declaration (PLAWK_MULTIPASS_CACHE.md §3.9): mark a
% derived relation (a view produced by a `gen ... as NAME` block, or a @prolog
% relation) as materialised-and-cached -- its projected/filtered rows are
% computed once into a shared table and reused by every query pass that reads
% it, instead of re-running the goal per consumer. Parses to materialize(Name);
% the runtime is a follow-on (surface-first, like the query-reader / generator
% arcs), so bin/plawk currently emits a clean not-yet diagnostic.
pass_clauses([materialize(Name) | Rest]) -->
    "materialize", identifier_boundary, ws,
    identifier(Name), ws,
    pass_clauses(Rest).
% A `pass { ACTIONS }` block is one pass carrying a single always-rule with
% those actions (per-pattern rules within a pass are a later extension).
pass_clauses([pass([rule(always, Actions)]) | Rest]) -->
    "pass", identifier_boundary, ws, action_block(Actions), ws,
    pass_clauses(Rest).
pass_clauses([]) -->
    [].

plawk_pass_dynentry_rewrite(DynEntries, pass(Rules0), pass(Rules)) :-
    plawk_dynentry_rewrite_all(Rules0, DynEntries, Rules).
% An `over TABLE` pass has no rule actions to rewrite -- pass it through.
plawk_pass_dynentry_rewrite(_DynEntries, pass_over(V, T, Body), pass_over(V, T, Body)).
% A `records of TABLE` reader likewise has no rule actions -- pass it through.
plawk_pass_dynentry_rewrite(_DynEntries, pass_records(V, T, Body), pass_records(V, T, Body)).
% A `rows of TABLE` reader likewise has no rule actions -- pass it through.
plawk_pass_dynentry_rewrite(_DynEntries, pass_rows(V, T, Body), pass_rows(V, T, Body)).
% The no-`as` (`$N`) positional reader likewise has no rule actions.
plawk_pass_dynentry_rewrite(_DynEntries, pass_rows_anon(T, Body), pass_rows_anon(T, Body)).
% The `over query(...)` reader has no rule actions to rewrite -- pass it through.
plawk_pass_dynentry_rewrite(_DynEntries, pass_query(Q, Body), pass_query(Q, Body)).
% A `gen { ... } as name` generator block -- pass it through (its body is
% lowered by the generator runtime in a later PR).
plawk_pass_dynentry_rewrite(_DynEntries, gen_block(N, S, Body), gen_block(N, S, Body)).
% A `materialize NAME` declaration has no rule actions -- pass it through.
plawk_pass_dynentry_rewrite(_DynEntries, materialize(Name), materialize(Name)).

%% dynentry_decls(-Names)//
%
%  Surface B (declaration-bound library names): for a fixed DYNLOAD
%  shipping a known entry family, bind names once at declaration and
%  call them like ordinary functions:
%
%      DYNENTRY parse, score
%      { total += parse($1) + score($2) }
%
%  A declared name is RESERVED for the compiled object: every bare
%  `name(args)` call site (and `float(name(args))`) rewrites at parse
%  time to the named-entry node (dyncall_named / float_dyncall_named),
%  so resolution stays static -- the entry's PC resolves once at
%  startup and is cached, exactly like an explicit dyncall@name site.
%  Reservation makes the name sets disjoint by construction: declaring
%  a name that an @prolog block or `function` definition also defines,
%  or a surface builtin name, is a PARSE ERROR (never silent
%  shadowing) -- checked in plawk_parse_source once the block clauses
%  are known.
dynentry_decls(Names) -->
    dynentry_decl(First),
    !,
    dynentry_decls(Rest),
    { append(First, Rest, Names) }.
dynentry_decls([]) -->
    [].

dynentry_decl(Names) -->
    "DYNENTRY",
    identifier_boundary,
    ws,
    dynentry_names(Names),
    ws.

dynentry_names([Name | Names]) -->
    identifier(Name),
    dynentry_names_rest(Names).

dynentry_names_rest([Name | Names]) -->
    ws,
    ",",
    ws,
    !,
    identifier(Name),
    dynentry_names_rest(Names).
dynentry_names_rest([]) -->
    [].

%% plawk_dynentry_rewrite_all(+Terms0, +Names, -Terms)
%
%  Substitute bare-call nodes over declared names with their
%  named-entry counterparts, everywhere in the parsed rules/END:
%  prolog_call -> dyncall_named, float_call -> float_dyncall_named.
%  A declared name in a position the named machinery does not compile
%  (e.g. a guard pattern) yields the ordinary
%  outside-the-compilable-surface build error rather than silently
%  calling userspace.
plawk_dynentry_rewrite_all(Terms, [], Terms) :- !.
plawk_dynentry_rewrite_all(Terms0, Names, Terms) :-
    plawk_dynentry_rewrite(Terms0, Names, Terms).

plawk_dynentry_rewrite(Term0, Names, Term) :-
    (   Term0 = prolog_call(Name, Args0),
        memberchk(Name, Names)
    ->  plawk_dynentry_rewrite(Args0, Names, Args),
        Term = dyncall_named(Name, Args)
    ;   Term0 = float_call(Name, Args0),
        memberchk(Name, Names)
    ->  plawk_dynentry_rewrite(Args0, Names, Args),
        Term = float_dyncall_named(Name, Args)
    ;   compound(Term0)
    ->  Term0 =.. [F | Args0],
        maplist(plawk_dynentry_rewrite_arg(Names), Args0, Args),
        Term =.. [F | Args]
    ;   Term = Term0
    ).

plawk_dynentry_rewrite_arg(Names, Arg0, Arg) :-
    plawk_dynentry_rewrite(Arg0, Names, Arg).

%% function_defs(-Clauses)//
%
%  awk-style expression functions, pure sugar over the foreign bridge:
%
%      function scale(a, b) { return a * b + 1 }
%
%  desugars at parse time into the Prolog clause
%
%      scale(A, B, R) :- R is A * B + 1.
%
%  and is called like any bridged predicate: `scale($1, $2)` as an
%  integer expression, `float(scale($1, $2))` to keep fractions. The
%  body is one `return` of an arithmetic expression over the
%  parameters (awk precedence; % maps to Prolog mod); an identifier
%  that is not a parameter fails the parse.
function_defs([Clause | Clauses]) -->
    function_def(Clause),
    !,
    function_defs(Clauses).
function_defs([]) -->
    [].

function_def((Head :- Body)) -->
    "function",
    identifier_boundary,
    ws,
    identifier(Name),
    ws,
    "(",
    ws,
    function_params(Params, ParamTypes),
    ws,
    ")",
    ws,
    "{",
    ws,
    "return",
    identifier_boundary,
    ws,
    function_expr(Params, Pairs, ArithTerm),
    action_block_close,
    { pairs_values(Pairs, Vars),
      % awk auto-coercion (PLAWK_AWK_FEATURE_AUDIT.md gap 2): a function body is
      % `Result is ArithTerm`, so every parameter is used numerically. Text
      % fields arrive as atoms (e.g. '5'), which `is/2` would reject -- so an
      % UNTYPED parameter takes a fresh head var that is coerced to a number
      % (`number(H) -> V = H ; atom_number(H, V)`) before the arithmetic.
      %
      % An OPTIONAL type annotation (`function f(num x)`) is a typed-fast path:
      % the declared param is already numeric, so the head var IS the value var
      % -- no coercion goal at all (Principle 2: static type knowledge elides
      % the runtime coercion). The dynamic default (auto-coerce) stays for
      % unannotated params; the annotation buys the elision, layered on top.
      maplist(plawk_fn_param_binding, ParamTypes, Vars, HeadVars, CoerceGoals0),
      exclude(==(true), CoerceGoals0, CoerceGoals),
      append(HeadVars, [Result], HeadArgs),
      Head =.. [Name | HeadArgs],
      append(CoerceGoals, [Result is ArithTerm], BodyGoals),
      plawk_conjunction(BodyGoals, Body)
    }.

% Per-parameter head-var binding. Untyped: a fresh head var coerced to a number.
% Typed (numeric): the head var IS the value var, so no coercion goal (`true`,
% filtered out) -- the declared arg is already the right type.
plawk_fn_param_binding(untyped, Value, Head,
    (number(Head) -> Value = Head ; atom_number(Head, Value))).
plawk_fn_param_binding(number, Value, Value, true).

% Fold a non-empty goal list into a Prolog conjunction.
plawk_conjunction([Goal], Goal) :- !.
plawk_conjunction([Goal | Goals], (Goal, Rest)) :-
    plawk_conjunction(Goals, Rest).

function_params([Param | Params], [Type | Types]) -->
    function_param(Param, Type),
    function_params_rest(Params, Types).

function_params_rest([Param | Params], [Type | Types]) -->
    ws,
    ",",
    ws,
    !,
    function_param(Param, Type),
    function_params_rest(Params, Types).
function_params_rest([], []) -->
    [].

% A parameter is an identifier, optionally prefixed by a numeric type keyword
% (`num` / `int` / `float`). The typed form is tried first; it backtracks to an
% untyped param when the keyword is actually the parameter's own name (e.g.
% `function f(num)` -- `num` is the name, not a type).
function_param(Name, number) -->
    param_type_keyword,
    identifier_boundary,
    ws,
    identifier(Name).
function_param(Name, untyped) -->
    identifier(Name).

param_type_keyword --> "num".
param_type_keyword --> "int".
param_type_keyword --> "float".

% arithmetic over the parameters, with awk precedence: * / % bind
% tighter than + -, both associate left, parentheses group
function_expr(Params, Pairs, Term) -->
    { pairs_keys(Pairs, Params) },
    function_additive(Pairs, Term).

function_additive(Pairs, Term) -->
    function_multiplicative(Pairs, First),
    function_additive_chain(Pairs, First, Term).

function_additive_chain(Pairs, Acc, Term) -->
    ws,
    function_add_op(Op),
    ws,
    !,
    function_multiplicative(Pairs, Right),
    { Acc1 =.. [Op, Acc, Right] },
    function_additive_chain(Pairs, Acc1, Term).
function_additive_chain(_Pairs, Term, Term) -->
    [].

function_add_op(+) --> "+".
function_add_op(-) --> "-".

function_multiplicative(Pairs, Term) -->
    function_factor(Pairs, First),
    function_multiplicative_chain(Pairs, First, Term).

function_multiplicative_chain(Pairs, Acc, Term) -->
    ws,
    function_mul_op(Op),
    ws,
    !,
    function_factor(Pairs, Right),
    { Acc1 =.. [Op, Acc, Right] },
    function_multiplicative_chain(Pairs, Acc1, Term).
function_multiplicative_chain(_Pairs, Term, Term) -->
    [].

function_mul_op(*) --> "*".
function_mul_op(/) --> "/".
function_mul_op(mod) --> "%".

function_factor(Pairs, Term) -->
    "(",
    ws,
    !,
    function_additive(Pairs, Term),
    ws,
    ")".
function_factor(_Pairs, Value) -->
    float_literal_expr(float_const(Mantissa, Denominator)),
    !,
    { Value is Mantissa / Denominator }.
function_factor(_Pairs, Value) -->
    integer_codes(ValueCodes),
    { ValueCodes \== [] },
    !,
    { number_codes(Value, ValueCodes) }.
function_factor(Pairs, Var) -->
    identifier(Param),
    { memberchk(Param-Var, Pairs) }.

% Tagged-union programs route rules through per-arm case blocks: the
% arm index scopes the field types of every rule inside the block.
program_rules(case_blocks(Blocks)) -->
    case_block(Block),
    !,
    case_blocks_rest(Blocks0),
    { Blocks = [Block | Blocks0] }.
program_rules(Rules) -->
    rules(Rules).

case_blocks_rest([Block | Blocks]) -->
    ws,
    case_block(Block),
    !,
    case_blocks_rest(Blocks).
case_blocks_rest([]) -->
    ws.

case_block(case_arm(Index, Rules)) -->
    "case",
    identifier_boundary,
    ws,
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    },
    ws,
    "{",
    ws,
    rules(Rules),
    ws,
    "}".

rules([Rule | Rules]) -->
    rule(Rule),
    rules_rest(Rules).

rules_rest([Rule | Rules]) -->
    ws,
    rule(Rule),
    !,
    rules_rest(Rules).
rules_rest([]) -->
    ws.

% A range pattern `/start/,/end/ { ... }`: the rule fires for every record from
% the one matching /start/ through the one matching /end/ (inclusive), tracked
% by a per-rule latch. Endpoints are regexes (v1). Tried before the general rule
% so the comma between the two regexes is seen (a plain `/re/ { }` has no comma,
% so this clause fails before its cut and falls through).
rule(rule(range(Start, End), Actions)) -->
    slash_regex_pattern(Start),
    ws,
    ",",
    ws,
    slash_regex_pattern(End),
    !,
    ws,
    action_block(Actions).
rule(rule(Pattern, Actions)) -->
    pattern(Pattern),
    !,
    ws,
    action_block(Actions).
rule(rule(always, Actions)) -->
    action_block(Actions).

action_block(Actions) -->
    "{",
    ws,
    actions(Actions),
    action_block_close.

% a trailing `;` or newline before the closing brace is harmless
action_block_close -->
    action_sep,
    !,
    "}",
    ws.
action_block_close -->
    ws,
    "}",
    ws.

%% body_block(-Actions)//
%
%  A control-flow body: either a braced `{ ... }` action block, or -- awk-style
%  -- a single braceless statement (`if (c) print`, `while (c) x++`). The braced
%  form is tried first; a braceless body is exactly one action wrapped in a
%  singleton list, so downstream lowering (which already takes an action list)
%  is unchanged.
body_block(Actions) -->
    action_block(Actions),
    !.
body_block([Action]) -->
    action(Action).

%% pattern(-Pattern)//
%
%  awk pattern combinators: `!` binds tighter than `&&`, which binds
%  tighter than `||`; both binary forms associate left and parentheses
%  group. The base patterns are the existing prefix, contains,
%  numeric-compare, and field-equality guards.
pattern(Pattern) -->
    or_pattern(Pattern).

or_pattern(Pattern) -->
    and_pattern(First),
    or_pattern_chain(First, Pattern).

or_pattern_chain(Acc, Pattern) -->
    ws,
    "||",
    ws,
    and_pattern(Right),
    !,
    or_pattern_chain(or_pat(Acc, Right), Pattern).
or_pattern_chain(Pattern, Pattern) -->
    [].

and_pattern(Pattern) -->
    not_pattern(First),
    and_pattern_chain(First, Pattern).

and_pattern_chain(Acc, Pattern) -->
    ws,
    "&&",
    ws,
    not_pattern(Right),
    !,
    and_pattern_chain(and_pat(Acc, Right), Pattern).
and_pattern_chain(Pattern, Pattern) -->
    [].

not_pattern(not_pat(Pattern)) -->
    "!",
    ws,
    not_pattern(Pattern),
    !.
not_pattern(Pattern) -->
    "(",
    ws,
    or_pattern(Pattern),
    ws,
    ")",
    !.
not_pattern(Pattern) -->
    base_pattern(Pattern).

base_pattern(Pattern) -->
    slash_regex_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    in_arr_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_match_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_i64_cmp_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    special_i64_cmp_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_eq_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    tag_eq_pattern(Pattern),
    !.
% blob(dyncall...) == "literal" -- an equality guard on a runtime
% grammar's byte output (JIT roadmap item 2 follow-on). Must precede
% prolog_guard_pattern: blob(...) would otherwise parse as a foreign
% guard call named blob.
base_pattern(Pattern) -->
    blob_eq_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    prolog_guard_pattern(Pattern).

%% in_arr_pattern(-Pattern)//
%
%  Associative-array membership: `KEY in ARRAY`. A scalar key is one
%  assoc_key_atom; a multi-part key uses awk's parenthesised `(K1,K2,...)`
%  spelling and the existing SUBSEP AST. Keep this parser deliberately as
%  broad as assoc_key_expr: codegen owns the supported-key policy and can
%  diagnose shapes (such as three-part keys) that are not implemented yet.
in_arr_pattern(in_arr(KeyExpr, ArrayName)) -->
    in_arr_key(KeyExpr),
    ws,
    "in",
    identifier_boundary,
    ws,
    table_ident(ArrayName).

in_arr_key(KeyExpr) -->
    "(",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    ")".
% Membership accepts the empty-string key even though the older general
% subscript atom grammar still excludes an empty literal. Keep this before the
% shared atom clause so `"" in arr` reaches the membership codegen path.
in_arr_key(string(Value)) -->
    quoted_string(ValueCodes),
    { string_codes(Value, ValueCodes) }.
in_arr_key(KeyExpr) -->
    assoc_key_atom(KeyExpr).

% Parenthesised membership in a value position is parsed only so the compiler
% can report that broader awk expression surface as unsupported (exit 3),
% instead of misclassifying valid syntax as a parse error. It intentionally
% remains a distinct node with no codegen lowering.
in_arr_value_expr(in_expr(Pattern)) -->
    "(",
    ws,
    in_arr_pattern(Pattern),
    ws,
    ")".

%% blob_eq_pattern(-Pattern)//
blob_eq_pattern(blob_eq(Blob, Value)) -->
    blob_call_expr(Blob),
    ws,
    "==",
    ws,
    quoted_string(ValueCodes),
    { ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.

%% tag_eq_pattern(-Pattern)//
%
%  TAG == K guards a rule by the record tag of a tagged-union layout.
%  Surface sugar: the codegen groups TAG-guarded rules into the same
%  per-arm case blocks that `case K { ... }` produces, so the tag test
%  must be the leftmost conjunct of the rule's pattern.
tag_eq_pattern(tag_pat(Tag)) -->
    "TAG",
    identifier_boundary,
    ws,
    "==",
    ws,
    integer_codes(TagCodes),
    { TagCodes \== [],
      number_codes(Tag, TagCodes),
      Tag >= 0
    }.

%% prolog_guard_pattern(-Pattern)//
%
%  A named Prolog predicate as a rule guard: `pred(args...)` matches
%  when the compiled predicate succeeds. Arguments are field atoms
%  ($0 is the whole record), string literal atoms, or integers.
prolog_guard_pattern(prolog_guard(Name, Args)) -->
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")".

foreign_args([Arg | Args]) -->
    foreign_arg(Arg),
    foreign_args_rest(Args).

foreign_args_rest([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    foreign_arg(Arg),
    foreign_args_rest(Args).
foreign_args_rest([]) -->
    [].

%% dyncall_at_source(-Source)//
%
%  The first argument of dyncall_at(...): either a path (field / string
%  literal, as before), or compile(field-or-string) -- the eval surface.
%  compile(Src) runs the shipped bootstrap-compiler .wamo on the Prolog
%  source text Src at runtime and yields a HANDLE to the freshly
%  compiled grammar (deduplicated by source text, so repeated compiles
%  of the same source reuse one loaded grammar). "compile" is reserved
%  in this position only; elsewhere it still parses as an ordinary
%  identifier.
% compile_file(field-or-string): the path names a grammar SOURCE file
% read at runtime; its contents compile through the same handle
% registry as compile(...), so editing the file changes behaviour with
% no rebuild (content dedup recompiles exactly when the bytes differ).
% Parsed before compile: they share a prefix.
dyncall_at_source(compile_file_src(Arg)) -->
    "compile_file",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
dyncall_at_source(compile_src(Arg)) -->
    "compile",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
% a bare identifier: a grammar HANDLE stored in a scalar --
%
%     { h = compile("[(sq(...)...)]") ; total += dyncall_at@sq(h, $1) }
%
% names the compiled source once instead of repeating it per call
% site. The handle IS an i64 (a registry index), so it rides the
% scalar machinery; the source marshals as (null path, handle id), the
% registry discriminator @plawk_dyncall_at_get already speaks.
dyncall_at_source(handle_src(var(Name))) -->
    identifier(Name),
    !.
dyncall_at_source(Source) -->
    foreign_arg(Source).

foreign_arg(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    !,
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
foreign_arg(string(Value)) -->
    quoted_string(ValueCodes),
    !,
    { string_codes(Value, ValueCodes) }.
foreign_arg(int(Value)) -->
    signed_integer_value(Value).

%% slash_regex_pattern(-Pattern)//
%
%  A bare /re/ pattern matches the whole record, as in awk. Bodies with
%  no ERE metacharacters keep their existing fast native lowerings: a
%  leading ^ plus a literal rest stays prefix/1 and an all-literal body
%  stays contains/1. Anything else becomes field_match(0, Regex) and is
%  matched with POSIX ERE at runtime.
slash_regex_pattern(Pattern) -->
    "/",
    regex_body_codes(Codes),
    "/",
    { Codes \== [],
      classify_regex_codes(Codes, Pattern)
    }.

classify_regex_codes([0'^ | Rest], prefix(Prefix)) :-
    Rest \== [],
    \+ ere_metachar_in_codes(Rest),
    !,
    string_codes(Prefix, Rest).
classify_regex_codes(Codes, contains(Literal)) :-
    \+ ere_metachar_in_codes(Codes),
    !,
    string_codes(Literal, Codes).
classify_regex_codes(Codes, field_match(0, Regex)) :-
    string_codes(Regex, Codes).

ere_metachar_in_codes(Codes) :-
    member(Code, Codes),
    memberchk(Code, [0'., 0'[, 0'], 0'(, 0'), 0'*, 0'+, 0'?,
                     0'{, 0'}, 0'|, 0'^, 0'$, 0'\\]),
    !.

%% field_match_pattern(-Pattern)//
%
%  awk's match operators: $N ~ /re/ and $N !~ /re/. $0 works via
%  field index 0. !~ reuses the combinator AST as not_pat(field_match).
field_match_pattern(Pattern) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    match_operator(Negated),
    ws,
    "/",
    regex_body_codes(Codes),
    "/",
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0,
      Codes \== [],
      string_codes(Regex, Codes),
      (   Negated == false
      ->  Pattern = field_match(Index, Regex)
      ;   Pattern = not_pat(field_match(Index, Regex))
      )
    }.

match_operator(true) -->
    "!~".
match_operator(false) -->
    "~".

%% regex_body_codes(-Codes)//
%
%  Codes between the pattern slashes. Backslash pairs pass through
%  unchanged so ERE escapes like \. reach regcomp, except \/ which
%  unescapes to a literal slash.
regex_body_codes([Code | Codes]) -->
    regex_body_code(Code),
    regex_body_codes_rest(Codes).

regex_body_codes_rest(Codes) -->
    regex_body_code(Code),
    !,
    { Codes = [Code | Rest] },
    regex_body_codes_rest(Rest).
regex_body_codes_rest([]) -->
    [].

regex_body_code(0'/) -->
    "\\/",
    !.
regex_body_code(Code) -->
    [Code],
    { Code =\= 0'/,
      Code =\= 0'\n,
      Code =\= 0'\r
    }.

field_eq_pattern(field_eq(Index, Value)) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    "==",
    ws,
    quoted_string(ValueCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index > 0,
      ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.

field_i64_cmp_pattern(field_cmp(Index, Op, Value)) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    numeric_cmp_op(Op),
    ws,
    signed_integer_value(Value),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index > 0
    }.

% Expression patterns: a numeric special/builtin compared to an integer, e.g.
% `NF > 3 { … }`, `NF == 0 { … }`, `length > 80 { … }`, `length($0) <= 40 { … }`.
% The special is the record's field count `NF` or its byte length `length` (of
% `$0`). Both operand orders parse: `SPECIAL OP int` and `int OP SPECIAL` (the
% reversed form swaps the operator). `identifier_boundary` keeps `NFX` an
% ordinary identifier, and `length_no_argument` keeps `length(…)` a call rather
% than the bare builtin.
special_i64_cmp_pattern(special_cmp(Special, Op, Value)) -->
    special_cmp_operand(Special),
    ws,
    numeric_cmp_op(Op),
    ws,
    signed_integer_value(Value).
special_i64_cmp_pattern(special_cmp(Special, SwappedOp, Value)) -->
    signed_integer_value(Value),
    ws,
    numeric_cmp_op(Op),
    ws,
    special_cmp_operand(Special),
    { swap_cmp_op(Op, SwappedOp) }.

% The special operand of an expression pattern: the field count or the record
% byte length (of `$0`, bare or parenthesised).
special_cmp_operand('NF') -->
    "NF",
    identifier_boundary.
special_cmp_operand(length) -->
    "length",
    ws,
    "(",
    ws,
    "$0",
    ws,
    ")".
special_cmp_operand(length) -->
    "length",
    length_no_argument.

% Swap a comparison operator for the reversed-operand (`int OP SPECIAL`) form.
swap_cmp_op(eq, eq).
swap_cmp_op(ne, ne).
swap_cmp_op(lt, gt).
swap_cmp_op(gt, lt).
swap_cmp_op(le, ge).
swap_cmp_op(ge, le).

numeric_cmp_op(eq) -->
    "==".
numeric_cmp_op(ne) -->
    "!=".
numeric_cmp_op(le) -->
    "<=".
numeric_cmp_op(ge) -->
    ">=".
numeric_cmp_op(lt) -->
    "<".
numeric_cmp_op(gt) -->
    ">".


integer_codes([Code | Codes]) -->
    [Code],
    { code_type(Code, digit) },
    integer_codes_rest(Codes).

integer_codes_rest([Code | Codes]) -->
    [Code],
    { code_type(Code, digit) },
    !,
    integer_codes_rest(Codes).
integer_codes_rest([]) -->
    [].

signed_integer_value(Value) -->
    "-",
    !,
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Magnitude, Digits),
      Value is -Magnitude
    }.
signed_integer_value(Value) -->
    "+",
    !,
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Value, Digits)
    }.
signed_integer_value(Value) -->
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Value, Digits)
    }.

quoted_string(Codes) -->
    "\"",
    quoted_string_codes(Codes),
    "\"".

quoted_string_codes(Codes) -->
    "\\",
    quoted_string_escape_codes(EscapedCodes),
    !,
    quoted_string_codes(RestCodes),
    { append(EscapedCodes, RestCodes, Codes) }.
quoted_string_codes([Code | Codes]) -->
    [Code],
    { Code =\= 0'", Code =\= 0'\n, Code =\= 0'\r },
    !,
    quoted_string_codes(Codes).
quoted_string_codes([]) -->
    [].

quoted_string_escape_codes([10]) -->
    "n".
quoted_string_escape_codes([9]) -->
    "t".
quoted_string_escape_codes([13]) -->
    "r".
quoted_string_escape_codes([0'"]) -->
    "\"".
quoted_string_escape_codes([0'\\]) -->
    "\\".
quoted_string_escape_codes([0'\\, Code]) -->
    [Code],
    { Code =\= 0'\n, Code =\= 0'\r }.

% Backed BEGIN block (multi-pass persistent cache, phase 1b): `BEGIN
% cache("path") { declare NAME ... }` declares one or more tables backed by
% the store at "path". Each `declare NAME` becomes a cache_table(NAME,
% "path") begin action; the codegen opens the store into NAME at setup and
% commits it at END. Tried before the plain BEGIN clause. See
% PLAWK_MULTIPASS_CACHE.md.
begin_clauses([begin(Actions)]) -->
    "BEGIN", ws, "cache", ws, "(", ws, quoted_string(PathCodes), ws,
    cache_backend(Backend), cache_namespace(NS), ws, ")", ws,
    "{", ws,
    { string_codes(Path, PathCodes) },
    cache_decl_list(Path, Backend, NS, Actions),
    ws, "}", ws,
    !.
begin_clauses([begin(Actions)]) -->
    "BEGIN",
    ws,
    "{",
    ws,
    begin_actions(Actions),
    ws,
    "}",
    ws,
    !.
begin_clauses([]) -->
    [].

% Optional backend selector inside cache(...): `backend "lmdb"` (durable
% LMDB store) or `backend "file"` (the default portable file backend).
cache_backend(Backend) -->
    ws, "backend", required_ws, quoted_string(BCodes),
    { atom_codes(Backend, BCodes) },
    !.
cache_backend(file) -->
    [].

% Optional namespace alias inside cache(...): `cache("db" as ns)` (phase 8.9
% PR 4, PLAWK_MULTIPASS_CACHE.md §3.7). `ns` prefixes the store's tables so
% they are referenced `ns.table` and never collide with global names; the
% store holds them as named sub-DBs (each a `ns.table` internal name whose
% LOCAL part is the sub-DB name). No alias -> `none` (global bare names).
cache_namespace(NS) -->
    ws, "as", required_ws, identifier(NS),
    !.
cache_namespace(none) -->
    [].

% A namespace-qualified table name: bare `Local` stays `Local`; under a
% namespace `NS` it becomes the dotted atom `'NS.Local'`, matching how a
% `ns.table` reference (table_ident) parses.
plawk_qualify(none, Local, Local) :- !.
plawk_qualify(NS, Local, QName) :-
    atomic_list_concat([NS, '.', Local], QName).

cache_decl_list(Path, Backend, NS, Actions) -->
    cache_decl(Path, Backend, NS, First),
    cache_decl_list_rest(Path, Backend, NS, Rest),
    { append(First, Rest, Actions) }.
cache_decl_list_rest(Path, Backend, NS, Actions) -->
    ws, cache_decl_sep, ws, cache_decl(Path, Backend, NS, First),
    !,
    cache_decl_list_rest(Path, Backend, NS, Rest),
    { append(First, Rest, Actions) }.
cache_decl_list_rest(_Path, _Backend, _NS, []) -->
    [].
cache_decl_sep --> ";", !.
cache_decl_sep --> [].

% One `declare` in a backed BEGIN block. A bare `declare NAME` is today's
% i64-valued table (one begin action, cache_table/3). `declare NAME(col type,
% ...)` (PLAWK_MULTIPASS_CACHE.md §3.6, row-oriented records) additionally
% carries a ROW SCHEMA -- named columns with types -- emitted as a separate
% cache_schema(NAME, Columns) action, so the existing cache_table/3 consumers
% are untouched and the schema is available to the record readers. Columns are
% col(Name, Type) with Type in {str, i64}. The column-list clause is tried
% first; a bare declare falls through.
cache_decl(Path, Backend, NS,
        [cache_table(Name, Path, Backend), cache_schema(Name, Columns)]) -->
    "declare", required_ws, identifier(Local), ws,
    cache_col_list(Columns),
    !,
    { plawk_qualify(NS, Local, Name) }.
cache_decl(Path, Backend, NS, [cache_table(Name, Path, Backend)]) -->
    "declare", required_ws, identifier(Local),
    { plawk_qualify(NS, Local, Name) }.
% `use NAME` (PLAWK_MULTIPASS_CACHE.md §3.7, phase 8.8): attach to an EXISTING
% store without re-stating its columns -- the schema is taken from the store's
% persisted header (§8.7). Parses to cache_use(NAME, Path, Backend); the plawk
% build reads the store's schema and expands it into the same
% cache_table/cache_schema a matching `declare NAME(cols)` would produce, so
% the readers need no new machinery. Tried after `declare`; `use` is
% unambiguous.
cache_decl(Path, Backend, NS, [cache_use(Name, Path, Backend)]) -->
    "use", required_ws, identifier(Local),
    { plawk_qualify(NS, Local, Name) }.

cache_col_list(Columns) -->
    "(", ws, cache_cols(Columns), ws, ")".
cache_cols([col(Name, Type) | Rest]) -->
    identifier(Name), required_ws, cache_col_type(Type),
    cache_cols_rest(Rest).
cache_cols_rest([col(Name, Type) | Rest]) -->
    ws, ",", ws, identifier(Name), required_ws, cache_col_type(Type),
    !,
    cache_cols_rest(Rest).
cache_cols_rest([]) --> [].
cache_col_type(str) --> "str".
cache_col_type(i64) --> "i64".

begin_actions([Action | Actions]) -->
    begin_action(Action),
    begin_actions_rest(Actions).

begin_actions_rest([Action | Actions]) -->
    ws,
    ";",
    ws,
    !,
    begin_action(Action),
    begin_actions_rest(Actions).
begin_actions_rest([]) -->
    [].

% getline is deliberately not executable in BEGIN in this phase. Preserve the
% surface as an explicit node so codegen reports a compile error (exit 3)
% instead of a parse error or an ignored BEGIN action. Try this before an
% ordinary BEGIN assignment: `FS = "cmd" | getline` otherwise commits to the
% valid `FS = "cmd"` prefix and leaves the pipe behind.
begin_action(unsupported_getline(begin(Action))) -->
    getline_context_action(Action),
    !.
begin_action(Action) -->
    begin_assignment(Action),
    !.
begin_action(Action) -->
    print_action(Action),
    !.

begin_assignment(set(var(Name), string(Value))) -->
    begin_assignment_name(Name),
    ws,
    "=",
    ws,
    quoted_string(ValueCodes),
    { string_codes(Value, ValueCodes) }.

begin_assignment_name('BINFMT') -->
    "BINFMT".
begin_assignment_name('OUTFMT') -->
    "OUTFMT".
begin_assignment_name('DYNLOAD') -->
    "DYNLOAD".
begin_assignment_name('DYNCACHE') -->
    "DYNCACHE".
begin_assignment_name('FS') -->
    "FS".
begin_assignment_name('OFS') -->
    "OFS".
begin_assignment_name('ORS') -->
    "ORS".
begin_assignment_name('RS') -->
    "RS".
begin_assignment_name('SUBSEP') -->
    "SUBSEP".

% Exact END cardinality idiom: initialise an accumulator to zero, increment it
% once for each occupied associative-array entry, then print it. Lower directly
% to the established for-in accumulation AST; sharing Acc across all three
% parser terms ensures that near-misses are not silently normalised.
%
%   END { n = 0; for (k in seen) n++; print n }
end_clauses([end([for_in(var(LoopVar), var(ArrayName),
                  [add(var(Acc), int(1))]),
                  print([var(Acc)])])]) -->
    "END", ws, "{", ws,
    assignment_action(set(var(Acc), int(0))),
    action_sep,
    "for", ws, "(", ws,
    identifier(LoopVar), ws, "in", identifier_boundary, ws,
    identifier(ArrayName), ws, ")", ws,
    identifier(Acc), ws, "++",
    action_sep,
    print_action(print([var(Acc)])),
    action_block_close,
    !.

% END decode-into-struct (assoc for-in, stage 3): a for-in whose body
% destructures the iterated value `arr[k]` through a grammar into typed
% fields, then prints them. `END { for (k in arr) { (n, m) =
% dyncall@decode(arr[k]) as (i64 i64) ; print k, n, m } }`. The decode
% argument is the for-in value `arr[k]` (array/key must match the loop);
% the destructured variables are for-in-scoped and read back in the print.
% Tried before the accumulate and single-action END clauses -- the
% `for (...) { ( vars ) = dyncall@... }` shape is unambiguous. See
% PLAWK_ASSOC_FORIN.md.
end_clauses([end([for_in(var(LoopVar), var(ArrayName),
                  [dynrec_bind(Vars, dyncall_named(Name, [forin_val(ArrayName)]),
                       Types),
                   print(PrintFields)])])]) -->
    "END", ws, "{", ws,
    "for", ws, "(", ws,
    identifier(LoopVar), ws, "in", identifier_boundary, ws,
    identifier(ArrayName), ws, ")", ws,
    "{", ws,
    "(", ws, dynrec_var_list(Vars), ws, ")", ws, "=", ws,
    "dyncall@", identifier(Name), ws, "(", ws,
    identifier(ArrayName), ws, "[", ws, identifier(LoopVar), ws, "]", ws, ")", ws,
    "as", identifier_boundary, ws, "(", ws, dynrec_type_list(Types), ws, ")", ws,
    action_sep, ws,
    print_action(print(PrintFields)), ws,
    "}", ws,
    "}", ws,
    !.
% END accumulate-then-print (assoc for-in, stage 2): a for-in that folds
% the hash into a scalar, followed by a print that reads the accumulator.
% `END { for (k in arr) acc += arr[k] ; print acc }`. The for-in body is a
% single `acc += OPERAND` where OPERAND is the iterated value `arr[k]`, the
% loop key `k`, or an integer; the trailing print reads `acc` (the
% loop-carried total). Tried before the single-action END clause -- the
% two-statement shape (for-in `;` print) is unambiguous. See
% PLAWK_ASSOC_FORIN.md.
end_clauses([end([for_in(var(LoopVar), var(ArrayName), [AccAction]),
                  print(PrintFields)])]) -->
    "END", ws, "{", ws,
    "for", ws, "(", ws,
    identifier(LoopVar), ws, "in", identifier_boundary, ws,
    identifier(ArrayName), ws, ")", ws,
    forin_accum_action(ArrayName, LoopVar, AccAction), ws,
    action_sep, ws,
    print_action(print(PrintFields)), ws,
    "}", ws,
    !.
end_clauses([end([Action])]) -->
    "END",
    ws,
    "{",
    ws,
    end_action(Action),
    ws,
    "}",
    ws,
    !.
end_clauses([]) -->
    [].

% `acc += OPERAND` inside a for-in accumulate body. The operand is
% for-in-scoped: `arr[k]` (the iterated value; array/key must match the
% loop), `k` (the loop key), or an integer literal.
forin_accum_action(Array, Key, add(var(Acc), Operand)) -->
    identifier(Acc), ws, "+=", ws,
    forin_accum_operand(Array, Key, Operand).

forin_accum_operand(Array, Key, forin_val(Array)) -->
    identifier(Array), ws, "[", ws, identifier(Key), ws, "]",
    !.
forin_accum_operand(_Array, Key, forin_key) -->
    identifier(Key),
    !.
forin_accum_operand(_Array, _Key, int(Value)) -->
    signed_integer_value(Value).

end_action(Action) -->
    for_in_action(Action),
    !.
% As in BEGIN, retain getline syntax for a clean context rejection in codegen.
end_action(unsupported_getline(end(Action))) -->
    getline_context_action(Action),
    !.
% a scalar-guarded print in END: `if (n > 1) print ...` [`else print ...`].
% The condition is a scalar comparison (END has no current record), lowered
% against the final slot values; each branch is a single print.
end_action(Action) -->
    if_action(Action),
    !.
end_action(Action) -->
    print_action(Action).

for_in_action(for_in(var(LoopVar), var(ArrayName), Body)) -->
    "for",
    ws,
    "(",
    ws,
    identifier(LoopVar),
    ws,
    "in",
    identifier_boundary,
    ws,
    identifier(ArrayName),
    ws,
    ")",
    ws,
    for_in_body(Body).

% for-in filter (assoc for-in, stage 1): `{ if (GUARD) print ... }` where
% GUARD compares the loop key `k` or the value `arr[k]` to an integer.
% A for-in-scoped condition -- k / arr[k] are only meaningful inside the
% loop, so the operands live here rather than in the global pattern
% grammar. Gates the per-key print; no cross-iteration state. Tried BEFORE the
% general action_block: `if (k > 0)` would otherwise parse as a scalar_if (`k`
% is the loop key, not a scalar slot), so the for-in-scoped guard must win.
for_in_body([if(Guard, [PrintAction], [])]) -->
    "{", ws, "if", ws, "(", ws, guard_expr(Guard), ws, ")", ws,
    print_action(PrintAction), ws, "}",
    !.
for_in_body(Actions) -->
    action_block(Actions),
    !.
for_in_body([WritebinAction]) -->
    writebin_action(WritebinAction),
    !.
for_in_body([PrintAction]) -->
    print_action(PrintAction).

%% for_c_action(-Action)//
%
%  C-style `for (INIT; COND; UPDATE) BODY` -- the awk three-part for. INIT and
%  UPDATE are simple statements (an assignment `i = 0` or an increment `i++` /
%  `i += n`); COND is a general loop condition (the same grammar as `while`).
%  Parses to c_for(Init, Cond, Update, Body); a normalisation pass
%  (plawk_normalise_c_for) desugars it to `INIT; while (COND) { BODY; UPDATE }`,
%  reusing the while runtime. Tried after for-in (which commits on `in`); the
%  `;` separators distinguish the two. v1 requires all three parts (empty parts
%  like `for (;;)` are a follow-on).
for_c_action(c_for(Init, Cond, Update, Body)) -->
    "for",
    ws,
    "(",
    ws,
    for_c_simple(Init),
    ws,
    ";",
    ws,
    while_condition(Cond),
    ws,
    ";",
    ws,
    for_c_simple(Update),
    ws,
    ")",
    ws,
    body_block(Body).

% A simple statement usable as a C-for init / update: an increment (`i++`), a
% compound assignment (`i += n`), or a plain assignment (`i = 0`).
for_c_simple(Action) -->
    increment_action(Action),
    !.
for_c_simple(Action) -->
    add_assign_action(Action),
    !.
for_c_simple(Action) -->
    assignment_action(Action).

% A guard expression: one or more comparisons combined with `&&` / `||`
% (short-circuit, `&&` binding tighter than `||`, left-associative, parens
% allowed). A single comparison parses to the bare guard term (unchanged), so
% existing single-guard `if`s are untouched; combinations parse to and(L, R) /
% or(L, R). Leaves are the existing forin_guard comparisons (reader guards
% `r["col"] CMP L` etc.) plus associative membership; unary `!` is available
% for the scoped membership filter.
guard_expr(Expr) -->
    guard_or(Expr).
guard_or(Expr) -->
    guard_and(Left),
    guard_or_rest(Left, Expr).
guard_or_rest(Left, Expr) -->
    ws, "||", ws, guard_and(Right),
    !,
    guard_or_rest(or(Left, Right), Expr).
guard_or_rest(Expr, Expr) -->
    [].
guard_and(Expr) -->
    guard_atom(Left),
    guard_and_rest(Left, Expr).
guard_and_rest(Left, Expr) -->
    ws, "&&", ws, guard_atom(Right),
    !,
    guard_and_rest(and(Left, Right), Expr).
guard_and_rest(Expr, Expr) -->
    [].
guard_atom(not_pat(Expr)) -->
    "!", ws, guard_atom(Expr),
    !.
guard_atom(Expr) -->
    "(", ws, guard_expr(Expr), ws, ")",
    !.
guard_atom(Guard) -->
    in_arr_pattern(Guard),
    !.
guard_atom(Guard) -->
    forin_guard(Guard).

% arr[k] CMP "str" -- string comparison of an array element (split / assoc(str)
% values are strings). `==`/`!=` lower to a canonical atom-id compare, the
% ordering ops (`<` `<=` `>` `>=`) to strcmp; carried as str(Text). Tried before
% the integer form so a quoted RHS takes the string branch; a non-string RHS
% backtracks.
forin_guard(forin_val_cmp(Array, LoopVar, Op, str(Text))) -->
    identifier(Array), ws, "[", ws, identifier(LoopVar), ws, "]", ws,
    numeric_cmp_op(Op), ws, quoted_string(SCodes),
    { string_codes(Text, SCodes) },
    !.
% arr[k] CMP <float> -- a float-literal RHS (`a[k] < 3.5`), compared via strnum
% on the element text (numeric vs the double, else lexical). Tried before the
% integer form: a float has a `.`, so `3.5` takes this branch and `3` the next.
forin_guard(forin_val_cmp(Array, LoopVar, Op, FloatConst)) -->
    identifier(Array), ws, "[", ws, identifier(LoopVar), ws, "]", ws,
    numeric_cmp_op(Op), ws, signed_float_lit(FloatConst),
    !.
% arr[k] CMP int -- value comparison (tried before the bare-key form,
% since `arr[` also starts with an identifier).
forin_guard(forin_val_cmp(Array, LoopVar, Op, Value)) -->
    identifier(Array), ws, "[", ws, identifier(LoopVar), ws, "]", ws,
    numeric_cmp_op(Op), ws, signed_integer_value(Value),
    !.
% k CMP int -- loop-key comparison.
forin_guard(forin_key_cmp(LoopVar, Op, Value)) -->
    identifier(LoopVar), ws, numeric_cmp_op(Op), ws,
    signed_integer_value(Value).
% Reader guards (row readers, a WHERE-style filter): a column compared to an
% integer. Named `r["col"] CMP int`, positional `r[N] CMP int`, or awk-field
% `$N CMP int`. Distinct functors so the reader dispatch resolves the column;
% the for-in planner does not generate them. Tried after the for-in forms
% (which fail cleanly on a non-identifier key / `$`, then backtrack here).
forin_guard(rcol_cmp(Var, Col, Op, Value)) -->
    identifier(Var), ws, "[", ws, quoted_string(CCodes), ws, "]", ws,
    numeric_cmp_op(Op), ws, guard_rhs(Value),
    { string_codes(Col, CCodes) },
    !.
forin_guard(rpos_cmp(Var, N, Op, Value)) -->
    identifier(Var), ws, "[", ws, integer_codes(NCodes), ws, "]", ws,
    numeric_cmp_op(Op), ws, guard_rhs(Value),
    { NCodes \== [], number_codes(N, NCodes), N > 0 },
    !.
forin_guard(rfield_cmp(N, Op, Value)) -->
    "$", integer_codes(NCodes), ws,
    numeric_cmp_op(Op), ws, guard_rhs(Value),
    { NCodes \== [], number_codes(N, NCodes), N > 0 }.

% A reader-guard right-hand side: a bare signed integer (an i64 comparison,
% unchanged); a signed decimal float literal (`3.5`, an f64 comparison, carried
% as float_const(Mantissa, Denominator) like other float literals); or a string
% literal (`"alice"`, a byte comparison, carried as str(Text) -- only `==` /
% `!=` are meaningful). A string starts with `"`, a float has a `.`, and a bare
% integer is the fallthrough, so the clauses are unambiguous.
guard_rhs(str(Text)) -->
    quoted_string(Codes),
    !,
    { string_codes(Text, Codes) }.
guard_rhs(float_const(M, D)) -->
    "-", float_literal_expr(float_const(M0, D)),
    !,
    { M is -M0 }.
guard_rhs(FloatConst) -->
    float_literal_expr(FloatConst),
    !.
guard_rhs(Value) -->
    signed_integer_value(Value).

% A signed decimal float literal (`3.5`, `-0.25`) as float_const(M, D); the
% same shape guard_rhs uses, factored out for the for-in element float guard.
signed_float_lit(float_const(M, D)) -->
    "-", float_literal_expr(float_const(M0, D)),
    !,
    { M is -M0 }.
signed_float_lit(FloatConst) -->
    float_literal_expr(FloatConst).

actions([Action | Actions]) -->
    action(Action),
    actions_rest(Action, Actions).

actions_rest(_Prev, [Action | Actions]) -->
    action_sep,
    action(Action),
    !,
    actions_rest(Action, Actions).
% as in awk/C, no separator is needed after a compound statement's
% closing brace (whose trailing ws has already been consumed)
actions_rest(Prev, [Action | Actions]) -->
    { plawk_block_action(Prev) },
    action(Action),
    !,
    actions_rest(Action, Actions).
actions_rest(_Prev, []) -->
    [].

plawk_block_action(if(_Pattern, _Then, _Else)).
plawk_block_action(foreach(_Body)).
% a loop is a braced block too: a statement may follow it without a separator
% (`while (..) { .. } i++`), like an if / foreach.
plawk_block_action(while_loop(_Cond, _Body)).
plawk_block_action(do_while_loop(_Body, _Cond)).
plawk_block_action(getline_loop(_Var, _File, _Body)).
plawk_block_action(getline_file_record_loop(_File, _Body)).
plawk_block_action(getline_main_record_loop(_Body)).
plawk_block_action(getline_main_var_loop(_Var, _Body)).
plawk_block_action(getline_pipe_record_loop(_Command, _Body)).
plawk_block_action(getline_pipe_loop(_Var, _Command, _Body)).
plawk_block_action(unsupported_getline(while(_Getline, _Body))).

%% action_sep//0
%
%  One statement separator, as in awk: any run of blanks, comments,
%  semicolons, and newlines containing at least one `;` or newline.
%  (A trailing separator before `}` is harmless: the following
%  action// fails and actions_rest backtracks to its empty clause.)
action_sep -->
    action_sep_scan(no).

action_sep_scan(_Seen) -->
    ";",
    !,
    action_sep_scan(yes).
action_sep_scan(_Seen) -->
    "\n",
    !,
    action_sep_scan(yes).
action_sep_scan(Seen) -->
    [Code],
    { Code =\= 0'\n, code_type(Code, space) },
    !,
    action_sep_scan(Seen).
action_sep_scan(Seen) -->
    "#",
    !,
    comment_rest,
    action_sep_scan(Seen).
action_sep_scan(yes) -->
    [].

action(Action) -->
    do_while_action(Action),
    !.
action(Action) -->
    while_action(Action),
    !.
action(Action) -->
    if_action(Action),
    !.
action(Action) -->
    printf_action(Action),
    !.
action(Action) -->
    emit_action(Action),
    !.
action(Action) -->
    writebin_action(Action),
    !.
action(Action) -->
    foreach_action(Action),
    !.
action(Action) -->
    print_action(Action),
    !.
action(Action) -->
    next_action(Action),
    !.
action(Action) -->
    break_action(Action),
    !.
action(Action) -->
    continue_action(Action),
    !.
action(Action) -->
    exit_action(Action),
    !.
action(Action) -->
    srand_action(Action),
    !.
action(Action) -->
    dynrec_view_action(Action),
    !.
action(Action) -->
    dynassoc_bind_action(Action),
    !.
action(Action) -->
    dynposarray_bind_action(Action),
    !.
action(Action) -->
    dynrec_bind_action(Action),
    !.
% for (k in arr) { ... } as a RULE-BODY action: per-record iteration
% over an assoc table (e.g. one a grammar just populated via
% `arr = dyncall@t($1) as assoc`), not just the END report. The body
% grammar is shared with the END form; what compiles is gated by the
% assoc-route codegen (a print body over the loop key / lookups).
action(Action) -->
    for_in_action(Action),
    !.
action(Action) -->
    for_c_action(Action),
    !.
action(Action) -->
    add_assign_action(Action),
    !.
action(Action) -->
    getline_action(Action),
    !.
action(Action) -->
    assignment_action(Action),
    !.
action(Action) -->
    delete_action(Action),
    !.
action(Action) -->
    split_action(Action),
    !.
action(Action) -->
    subgsub_action(Action),
    !.
action(Action) -->
    increment_action(Action),
    !.

%% getline_action(-Action)//
%
%  Redirected forms read through a filename-keyed shared handle. Main-input
%  forms read the next record from the driver's stream: `getline` replaces `$0`
%  while `getline var` only updates the scalar target. A string-literal command
%  may precede `| getline`; dynamic command and filename shapes are retained as
%  unsupported_getline nodes so they fail at compile time rather than being
%  approximated. The scalar-target pipe clause precedes its record sibling so
%  the latter cannot accept only the `... | getline` prefix.
getline_action(getline_pipe_read(Var, Command)) -->
    quoted_string(CCodes),
    ws, "|", ws,
    "getline",
    required_inline_ws,
    identifier(Var),
    { string_codes(Command, CCodes) },
    !.
getline_action(getline_pipe_record(Command)) -->
    quoted_string(CCodes),
    ws, "|", ws,
    "getline",
    identifier_boundary,
    { string_codes(Command, CCodes) },
    !.
getline_action(getline_file_record(File)) -->
    "getline",
    identifier_boundary,
    ws, "<", ws,
    quoted_string(FCodes),
    { string_codes(File, FCodes) },
    !.
getline_action(getline_read(Var, File)) -->
    "getline",
    required_ws,
    identifier(Var),
    ws, "<", ws,
    quoted_string(FCodes),
    { string_codes(File, FCodes) },
    !.
% Try unsupported redirected shapes before the main-input forms: otherwise the
% latter would accept only the `getline` / `getline var` prefix of a dynamic
% redirect and the enclosing parse would fail instead of reaching codegen's
% clean unsupported-feature diagnostic.
getline_action(unsupported_getline(Reason)) -->
    unsupported_getline_statement(Reason).
getline_action(getline_main_var(Var)) -->
    "getline",
    required_inline_ws,
    identifier(Var),
    !.
getline_action(getline_main_record) -->
    "getline",
    identifier_boundary.

% A context grammar used by BEGIN/END rejection. Assignment getline forms are
% included so `BEGIN { r = getline < "f" }` is rejected at compile time too.
getline_context_action(Action) -->
    getline_action(Action).
getline_context_action(Action) -->
    getline_assignment_action(Action).

% Unsupported statement forms. Longest shapes come first so the trailing
% tokens cannot be left for the enclosing action grammar. A pipe target is
% consumed before the record form for the same prefix reason as above.
unsupported_getline_statement(pipe_var(Command, Var)) -->
    getline_dynamic_pipe_expr(Command), ws,
    "|", ws, "getline", required_inline_ws, identifier(Var),
    !.
unsupported_getline_statement(pipe(Command)) -->
    getline_dynamic_pipe_expr(Command), ws,
    "|", ws, "getline", identifier_boundary,
    !.
unsupported_getline_statement(file_var(Var, File)) -->
    "getline", required_ws, identifier(Var), ws,
    "<", ws, getline_dynamic_file_expr(File),
    !.
unsupported_getline_statement(file_record_dynamic(File)) -->
    "getline", identifier_boundary, ws,
    "<", ws, getline_dynamic_file_expr(File),
    !.

% Consume a non-literal command expression solely for an explicit v1
% rejection. Parentheses need a dedicated shape for the same reason as a
% dynamic redirected filename. Literal strings are handled by the supported
% clauses above and deliberately excluded here.
getline_dynamic_pipe_expr(paren(Expr)) -->
    "(", ws, scalar_value_expr(Expr), ws, ")",
    !.
getline_dynamic_pipe_expr(Expr) -->
    scalar_value_expr(Expr),
    { Expr \= string(_) }.

% Consume a non-string-literal filename expression solely so v1 can reject it
% as an explicit unsupported node (build exit 3). Parenthesised scalar values
% need their own clause because a bare `(name)` is not a general field_expr.
getline_dynamic_file_expr(paren(Expr)) -->
    "(", ws, scalar_value_expr(Expr), ws, ")",
    !.
getline_dynamic_file_expr(Expr) -->
    scalar_value_expr(Expr),
    { Expr \= string(_) }.

%% subgsub_action(-Action)//
%
%  `sub(/re/, "repl")` / `gsub(/re/, "repl")` -- substitute the first / every ERE
%  match in $0 with the replacement (an unescaped `&` in the replacement expands
%  to the matched text). Parses to regex_sub(Global, Regex, Repl) with Global 1
%  for gsub, 0 for sub. v1: the target is $0 (no third arg); the codegen supports
%  the `Pattern { sub/gsub(...); print }` stream-editor idiom.
% `gsub(/re/, "repl", $N)` -- substitute into field $N in place, rebuilding $0
% (the field-buffer stream editor). Tried before the var-target form; the `$`
% distinguishes it.
subgsub_action(regex_sub_field(Global, Regex, Repl, N)) -->
    subgsub_keyword(Global),
    ws, "(", ws,
    regex_arg(Regex),
    ws, ",", ws,
    quoted_string(RCodes),
    { string_codes(Repl, RCodes) },
    ws, ",", ws,
    "$", integer_codes(NCodes),
    { NCodes \== [], number_codes(N, NCodes), N >= 1 },
    ws, ")".
% `gsub(/re/, "repl", var)` -- substitute into the string scalar `var` (in place),
% not $0. Tried before the no-target form; a missing third arg backtracks.
subgsub_action(regex_sub_var(Global, Regex, Repl, Name)) -->
    subgsub_keyword(Global),
    ws, "(", ws,
    regex_arg(Regex),
    ws, ",", ws,
    quoted_string(RCodes),
    { string_codes(Repl, RCodes) },
    ws, ",", ws,
    identifier(Name),
    ws, ")".
subgsub_action(regex_sub(Global, Regex, Repl)) -->
    subgsub_keyword(Global),
    ws, "(", ws,
    regex_arg(Regex),
    ws, ",", ws,
    quoted_string(RCodes),
    { string_codes(Repl, RCodes) },
    ws, ")".

subgsub_keyword(1) --> "gsub".
subgsub_keyword(0) --> "sub".

%% split_action(-Action)//
%
%  `split($N, arr, "sep")` -- split field $N on the single-char separator into
%  positional array `arr` (keys 1..n, string values). Parses to
%  split_into(field(N), var(Arr), string(Sep)). v1: field source, string-literal
%  separator; the return count and a default (FS) separator are follow-ons.
split_action(split_into(field(N), var(Arr), string(Sep))) -->
    "split",
    ws,
    "(",
    ws,
    "$",
    integer_codes(NCodes),
    { NCodes \== [], number_codes(N, NCodes), N >= 0 },
    ws,
    ",",
    ws,
    identifier(Arr),
    ws,
    ",",
    ws,
    quoted_string(SepCodes),
    { SepCodes \== [], string_codes(Sep, SepCodes) },
    ws,
    ")".

%% delete_action(-Action)//
%
%  `delete arr[k]` -- remove the entry keyed by `k` from an assoc table.
%  Parses to delete_assoc(var(Name), KeyExpr). Same key forms as `arr[k]++`
%  (a field `$N`, an integer, a quoted string, a blob call, or a bare var).
delete_action(delete_assoc(var(Name), KeyExpr)) -->
    "delete",
    identifier_boundary,
    ws,
    table_ident(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]".

%% if_action(-Action)//
%
%  awk conditionals: `else` is optional (an absent else parses as an
%  empty branch), and `else if` chains nest as a single-element else
%  branch containing the next if.
% A `while (VAR CMP int) { BODY }` loop -- iterate the body while a scalar
% variable compares to an integer bound (the awk `while` control structure).
% Parses to while_loop(cmp(var(V), Op, int(N)), Body). This is the SURFACE
% (mirrors the query reader / generator surface-first PRs): the loop body reuses
% the general action block, and the codegen (bin/plawk) rejects it with a clean
% not-yet diagnostic until the loop runtime lands. The condition is a scalar
% comparison for now; a general boolean condition is a follow-on.
% `while (("cmd" | getline var) > 0) BODY` -- drain a literal command's
% stdout into a scalar. Keep this before the record form so the target cannot
% be left behind as trailing input.
while_action(getline_pipe_loop(Var, Command, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, quoted_string(CCodes), ws,
    "|", ws, "getline", required_inline_ws, identifier(Var), ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    { string_codes(Command, CCodes) },
    !.
% `while (("cmd" | getline) > 0) BODY` -- drain it into `$0`.
while_action(getline_pipe_record_loop(Command, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, quoted_string(CCodes), ws,
    "|", ws, "getline", identifier_boundary, ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    { string_codes(Command, CCodes) },
    !.
% `while ((getline < "file") > 0) BODY` -- read successive newline-delimited
% file records into `$0`. It normalises to a priming record read plus the same
% status-slot while machinery used by scalar-target getline.
while_action(getline_file_record_loop(File, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, "getline", identifier_boundary, ws,
    "<", ws, quoted_string(FCodes), ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    { string_codes(File, FCodes) },
    !.
% `while ((getline var < "file") > 0) BODY` -- the scalar-target sibling.
while_action(getline_loop(Var, File, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, "getline", required_ws, identifier(Var), ws,
    "<", ws, quoted_string(FCodes), ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    { string_codes(File, FCodes) }.
% `while ((getline) > 0) BODY` -- drain the remaining main input into `$0`.
while_action(getline_main_record_loop(Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, "getline", identifier_boundary, ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    !.
% `while ((getline var) > 0) BODY` -- drain it into a scalar without changing
% the surrounding current record.
while_action(getline_main_var_loop(Var, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, "getline", required_inline_ws, identifier(Var), ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    !.
% Other getline operands in the same while shape are deliberate v1 rejections:
% dynamic filenames must reach codegen rather than falling out of the parser.
while_action(unsupported_getline(while(Getline, Body))) -->
    "while", identifier_boundary, ws,
    "(", ws,
    "(", ws, getline_action(Getline), ws, ")",
    ws, ">", ws, "0", ws,
    ")", ws,
    body_block(Body),
    !.
while_action(while_loop(Cond, Body)) -->
    "while", identifier_boundary, ws,
    "(", ws,
    while_condition(Cond), ws,
    ")", ws,
    body_block(Body).

% A `do { BODY } while (VAR CMP int)` loop -- the body runs at least once, then
% repeats while the condition holds (the awk do-while control structure). Parses
% to do_while_loop(Body, cmp(var(V), Op, int(N))). Surface only, like `while`;
% both share the same loop runtime (a later PR). Tried via its own action
% clause; the leading `do` keyword distinguishes it.
do_while_action(do_while_loop(Body, Cond)) -->
    "do", identifier_boundary, ws,
    body_block(Body), ws,
    "while", identifier_boundary, ws,
    "(", ws,
    while_condition(Cond), ws,
    ")".

% A loop condition: scalar comparisons (`VAR CMP int` or `VAR CMP VAR`) combined
% with `&&` / `||`. `&&` binds tighter than `||` (awk precedence). A single
% `VAR CMP int` still parses to the bare `cmp(...)` term, so the earlier runtime
% is a strict subset. (PLAWK_CONTROL_FLOW_PLAN.md PR 3.)
while_condition(Cond) -->
    while_cond_and(First),
    while_cond_or_rest(First, Cond).

while_cond_or_rest(Acc, Cond) -->
    ws, "||", ws, !,
    while_cond_and(Next),
    while_cond_or_rest(or(Acc, Next), Cond).
while_cond_or_rest(Cond, Cond) -->
    [].

while_cond_and(Cond) -->
    while_cmp(First),
    while_cond_and_rest(First, Cond).

while_cond_and_rest(Acc, Cond) -->
    ws, "&&", ws, !,
    while_cmp(Next),
    while_cond_and_rest(and(Acc, Next), Cond).
while_cond_and_rest(Cond, Cond) -->
    [].

% A scalar comparison: the left side is a loop variable or an i64 special
% (RSTART/RLENGTH, set by match()); the right side is an integer literal or
% another loop variable. The special clause is tried first (with a boundary so
% `RSTARTED` still parses as an identifier) so `if (RSTART > 0)` reads the
% special, not a phantom slot.
while_cmp(cmp(special(Name), Op, Rhs)) -->
    match_special_name(Name), identifier_boundary,
    ws, numeric_cmp_op(Op), ws, while_cmp_rhs(Rhs).
while_cmp(cmp(var(V), Op, Rhs)) -->
    identifier(V), ws, numeric_cmp_op(Op), ws, while_cmp_rhs(Rhs).

match_special_name('RSTART') --> "RSTART".
match_special_name('RLENGTH') --> "RLENGTH".
% NR (the record number) is a special in a comparison guard too, so `if (NR > 1)`
% / `while (NR < 3)` read the record counter rather than a phantom scalar slot.
match_special_name('NR') --> "NR".
% FNR (the per-file record number) is an alias for NR here: plawk streams a
% single input, so no per-file counter reset ever happens and FNR == NR. It
% parses to special('NR') so every NR context handles it with no codegen.
match_special_name('NR') --> "FNR".
% NF (the field count of the current record) is a guard special too: `if (NF > 3)`.
% The condition lowering threads the field separator to compute it from the
% record; it has no defined value in an END condition (no current record).
match_special_name('NF') --> "NF".
% ARGC (the command-line argument count) is a numeric special in a guard too:
% `if (ARGC > 1)`.
match_special_name('ARGC') --> "ARGC".

while_cmp_rhs(int(N)) -->
    signed_integer_value(N),
    !.
% a string literal RHS -- used by `if (s == "text")` string-equality guards on a
% string scalar (lowered as interned atom-id comparison). A while loop with a
% string condition is a clean codegen error (no string loop bound).
while_cmp_rhs(string(Value)) -->
    quoted_string(Codes),
    { string_codes(Value, Codes) },
    !.
while_cmp_rhs(var(W)) -->
    identifier(W).

%% if_condition(-Cond)//
%
%  An `if` condition is either a SCALAR comparison over variables (`if (i > 2)`,
%  `if (i < n && j > 0)`) -- the same `VAR CMP int/VAR` shape as a loop condition,
%  wrapped as scalar_if(_) so the lowering reads slot values -- or the existing
%  field/pattern guard (`$1 > 2`, `$0 ~ /re/`, combinators). The scalar form is
%  tried first; it fails fast on a field condition (a `$` is not an identifier),
%  falling through to the pattern. A single condition is scalar OR pattern, not a
%  mix. (PLAWK_CONTROL_FLOW_PLAN.md 3b -- unblocks counter-based loop control.)
if_condition(scalar_if(Cond)) -->
    while_condition(Cond).
if_condition(Pattern) -->
    condition_pattern(Pattern).

if_action(if(Pattern, ThenActions, ElseActions)) -->
    "if",
    ws,
    "(",
    ws,
    if_condition(Pattern),
    ws,
    ")",
    ws,
    body_block(ThenActions),
    if_else_part(ElseActions).

% `else` may follow a braced then-body directly, or a braceless one across a
% statement separator (`if (c) print x; else print y`) -- so tolerate an
% optional separator before `else`. If no `else` follows, the clause fails and
% backtracks (un-consuming the separator) to the empty else.
if_else_part(ElseActions) -->
    opt_action_sep,
    "else",
    identifier_boundary,
    if_else_body(ElseActions),
    !.
if_else_part([]) -->
    [].

opt_action_sep -->
    action_sep,
    !.
opt_action_sep -->
    ws.

if_else_body([ElseIfAction]) -->
    required_ws,
    if_action(ElseIfAction),
    !.
if_else_body(ElseActions) -->
    ws,
    body_block(ElseActions).

condition_pattern(Pattern) -->
    or_pattern(Pattern).

increment_action(inc_assoc(var(Name), KeyExpr)) -->
    table_ident(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !,
    "++".
increment_action(inc(var(Name))) -->
    identifier(Name),
    "++".

next_action(next) -->
    "next",
    identifier_boundary.

break_action(break) -->
    "break",
    identifier_boundary.

continue_action(continue) -->
    "continue",
    identifier_boundary.

% `exit` / `exit N` -- stop reading records, run END, and return N (default 0).
% Parses to exit(int(N)). Unlike `break`, it always ends the program (it is not
% consumed by an enclosing loop).
exit_action(exit(int(N))) -->
    "exit",
    identifier_boundary,
    ws,
    signed_integer_value(N),
    !.
exit_action(exit(int(0))) -->
    "exit",
    identifier_boundary.

%% srand_action(-Action)//
%
%  `srand(N)` / `srand()` -- seed the PRNG that rand() draws from (libm
%  srand48). `srand(N)` seeds with the integer N so a run is reproducible;
%  `srand()` seeds from the wall clock (time). Parses to srand(int(N)) or
%  srand_time. A rule-body statement -- BEGIN blocks do not yet host non-print
%  actions, and srand's awk return value (the previous seed) is a follow-on.
srand_action(srand(int(N))) -->
    "srand",
    identifier_boundary,
    ws,
    "(",
    ws,
    signed_integer_value(N),
    ws,
    ")",
    !.
srand_action(srand_time) -->
    "srand",
    identifier_boundary,
    ws,
    "(",
    ws,
    ")",
    !.

%% dynrec_bind_action(-Action)//
%
%  Structured-return destructuring: bind each field of a grammar's compound
%  return to a scalar variable, typed by a return shape.
%
%      (n, half) = dyncall@rec($1) as (i64 f64)
%
%  desugars to dynrec_bind([n, half], dyncall_named(rec, [field(1)]),
%  [i64, f64]); field 0 lands in the i64 scalar n, field 1 in the f64
%  scalar half. The call is a bare dyncall(...) (default entry) or a
%  dyncall@name(...) (named entry); the type list is whitespace-separated
%  i64 / f64 tokens (one per bound variable).
%% dynrec_view_action(-Action)//
%
%  Structured-return record view: the returned compound becomes the current
%  record for a scoped block, so `$1`,`$2` read its fields like a BINFMT
%  line.
%
%      dyncall@rec($1) as (i64 f64) { total += $1 ; sum += $2 }
%
%  desugars (in codegen) to a destructure into hidden temporaries plus the
%  block body with `$N` rewritten to the Nth temporary -- so it rides the
%  same machinery as the explicit destructure, no field-pointer repoint.
%% dynassoc_bind_action(-Action)//
%
%  Associative-array return: a grammar returning a list of integer key-value
%  pairs populates a plawk assoc array.
%
%      arr = dyncall@tally($1) as assoc
%      arr = dyncall($1) as assoc          % DYNLOAD object's default entry
%      arr = dyncall@label($1) as assoc(str)   % string VALUES
%
%  desugars to dynassoc_bind(var(arr), dyncall_named(tally, [field(1)]))
%  (or dynassoc_bind(var(arr), dyncall([field(1)])) for the default-entry
%  form); per record the returned [K-V, ...] pairs are inserted into arr's
%  i64 table, so END `arr[key]` lookups see the accumulated result.
%
%  The `(str)` value kind yields dynassoc_bind_str(var(arr), Call): the
%  grammar returns ATOM values, stored by registry id with replace (not
%  accumulate) semantics, and reads print the text.
dynassoc_bind_action(Action) -->
    identifier(Name),
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "assoc",
    dynassoc_value_kind(Kind),
    { Kind == str
    ->  Action = dynassoc_bind_str(var(Name), Call)
    ;   Action = dynassoc_bind(var(Name), Call)
    }.

dynassoc_value_kind(str) -->
    ws, "(", ws, "str", ws, ")",
    !.
dynassoc_value_kind(i64) -->
    [].

%% dynposarray_bind_action(-Action)//
%
%  Positional-array return: a grammar returning a FLAT list of integers
%  populates a plawk array by POSITION -- element i lands at key i
%  (1-indexed, the awk `split` convention), replace semantics (the array
%  reflects the most recent record's list).
%
%      arr = dyncall@fields($1) as array
%      arr = dyncall($1) as array          % DYNLOAD object's default entry
%
%  desugars to dynposarray_bind(var(arr), dyncall_named(fields, [field(1)]))
%  (or dynposarray_bind(var(arr), dyncall([field(1)])) for the default
%  entry). The last target container of JIT roadmap item 4: same i64
%  table as `as assoc`, but filled from a flat [V1, ..., Vn] list instead
%  of key-value pairs, so END `arr[1]`, `arr[2]`, ... and for-in read the
%  positional values.
dynposarray_bind_action(Action) -->
    identifier(Name),
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "array",
    dynassoc_value_kind(Kind),
    { Kind == str
    ->  Action = dynposarray_bind_str(var(Name), Call)
    ;   Action = dynposarray_bind(var(Name), Call)
    }.

dynrec_view_action(dynrec_view(Call, Types, Body)) -->
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "(",
    ws,
    dynrec_type_list(Types),
    ws,
    ")",
    ws,
    action_block(Body).

dynrec_bind_action(dynrec_bind(Vars, Call, Types)) -->
    "(",
    ws,
    dynrec_var_list(Vars),
    ws,
    ")",
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "(",
    ws,
    dynrec_type_list(Types),
    ws,
    ")".

dynrec_var_list([Name | Rest]) -->
    identifier(Name),
    dynrec_var_list_rest(Rest).
dynrec_var_list_rest([Name | Rest]) -->
    ws, ",", ws, identifier(Name), !, dynrec_var_list_rest(Rest).
dynrec_var_list_rest([]) -->
    [].

dynrec_call_expr(Call) -->
    prolog_call_expr(Call),
    { ( Call = dyncall(_)
      ; Call = dyncall_named(_, _)
      ; Call = dyncall_at(_, _)                % runtime source (JIT reader)
      ; Call = dyncall_at_named(_, _, _)
      ) }.

dynrec_type_list([Type | Rest]) -->
    dynrec_type(Type),
    dynrec_type_list_rest(Rest).
dynrec_type_list_rest([Type | Rest]) -->
    ws, dynrec_type(Type), !, dynrec_type_list_rest(Rest).
dynrec_type_list_rest([]) -->
    [].

dynrec_type(i64) --> "i64".
dynrec_type(f64) --> "f64".
dynrec_type(string) --> "string".

% Associative add-assign: `arr[k] += DELTA` folds DELTA into the table at
% the key. The natural pass-1 half of per-key normalise (`total[$1] += $2`),
% and the general form of `arr[k]++` (which is `+= 1`). DELTA is a field or
% an integer literal (i64 table values); other deltas are a follow-on. Tried
% before the scalar clause -- the `[` distinguishes them.
add_assign_action(add_assoc(var(Name), KeyExpr, Delta)) -->
    table_ident(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !,
    ws,
    "+=",
    ws,
    assoc_add_delta(Delta).
add_assign_action(add(var(Name), Delta)) -->
    identifier(Name),
    ws,
    "+=",
    ws,
    scalar_delta_expr(Delta).

% Delta for an assoc add-assign: a numeric field or an integer literal.
assoc_add_delta(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    },
    !.
assoc_add_delta(int(Value)) -->
    integer_codes(ValueCodes),
    { ValueCodes \== [],
      number_codes(Value, ValueCodes)
    }.

% Row store (PLAWK_MULTIPASS_CACHE.md §3.6): `TABLE[$k] = RHS` stores a row
% value in TABLE, keyed by field k. RHS is either `$0` (capture the whole
% current record -- set_row/2) or `row($a, $b, ...)` (construct a row from
% chosen fields, in that order -- set_row_cons/3), the producer of row-valued
% tables. A later pass reads it back (`over TABLE`, and, with a schema,
% `records of TABLE as r`). Tried before the scalar `set` -- the `[`
% distinguishes them; the `!` commits once `TABLE[key] =` is seen.
assignment_action(Action) -->
    table_ident(Name),
    ws, "[", ws, assoc_key_expr(KeyExpr), ws, "]", ws, "=", ws,
    !,
    assoc_row_rhs(var(Name), KeyExpr, Action).

assoc_row_rhs(VarName, KeyExpr, set_row(VarName, KeyExpr)) -->
    "$0",
    !.
assoc_row_rhs(VarName, KeyExpr, set_row_cons(VarName, KeyExpr, Fields)) -->
    "row", ws, "(", ws, row_field_list(Fields), ws, ")".

% Row constructor fields: one or more `$N` field references.
row_field_list([F | Fs]) -->
    row_field(F),
    row_field_list_rest(Fs).
row_field_list_rest([F | Fs]) -->
    ws, ",", ws, row_field(F),
    !,
    row_field_list_rest(Fs).
row_field_list_rest([]) --> [].
row_field(field(Index)) -->
    "$",
    integer_codes(Codes),
    { Codes \== [], number_codes(Index, Codes), Index >= 0 }.

% Field assignment `$N = expr` -- mutate field N of the current record, which
% rebuilds `$0` (fields re-joined with OFS). Parses to set_field(N, Value). N>=1
% (whole-record `$0 = ...` is a separate operation). RHS v1: a string literal,
% a non-negative integer literal, or another field `$M` (M>=1, read from the
% current record). Tried before the scalar `set` -- the leading `$` is
% unambiguous. The codegen supports the canonical `{ $N = expr; ...; print }`
% shape in single-char-FS text mode.
assignment_action(set_field(N, Value)) -->
    "$",
    integer_codes(NCodes),
    { NCodes \== [], number_codes(N, NCodes), N >= 1 },
    ws,
    "=",
    ws,
    field_assign_rhs(Value).

field_assign_rhs(string(Value)) -->
    quoted_string(Codes),
    { string_codes(Value, Codes) },
    !.
field_assign_rhs(field(M)) -->
    "$",
    integer_codes(MCodes),
    { MCodes \== [], number_codes(M, MCodes), M >= 1 },
    !.
field_assign_rhs(int(Value)) -->
    integer_codes(VCodes),
    { VCodes \== [], number_codes(Value, VCodes), Value >= 0 }.

% `n = gsub(/re/, "repl", var)` -- count capture: substitute into the string
% scalar `var` (in place) and assign the substitution count to `n`. A dual-slot
% write: `var` becomes a string scalar, `n` an i64 count. Tried before the
% generic scalar `set`; the `gsub(`/`sub(` keyword after `=` is unambiguous.
assignment_action(gsub_count(CountName, Global, Regex, Repl, Target)) -->
    identifier(CountName),
    ws, "=", ws,
    subgsub_keyword(Global),
    ws, "(", ws,
    regex_arg(Regex),
    ws, ",", ws,
    quoted_string(RCodes),
    { string_codes(Repl, RCodes) },
    ws, ",", ws,
    identifier(Target),
    ws, ")",
    !.

% Getline assignments are tried before the generic scalar `set`. This is
% essential for unsupported `status = getline`: without the dedicated fallback
% it would be misparsed as a read of an ordinary variable named `getline`.
assignment_action(Action) -->
    getline_assignment_action(Action),
    !.

assignment_action(set(var(Name), Value)) -->
    identifier(Name),
    ws,
    "=",
    ws,
    scalar_value_expr(Value).

% Pipe captures use a literal command and mirror the two statement targets.
% The scalar-target form is tried first because it extends the record prefix.
getline_assignment_action(getline_pipe_capture(Status, Var, Command)) -->
    identifier(Status),
    ws, "=", ws,
    quoted_string(CCodes),
    ws, "|", ws,
    "getline",
    required_inline_ws,
    identifier(Var),
    { string_codes(Command, CCodes) },
    !.
getline_assignment_action(getline_pipe_record_capture(Status, Command)) -->
    identifier(Status),
    ws, "=", ws,
    quoted_string(CCodes),
    ws, "|", ws,
    "getline",
    identifier_boundary,
    { string_codes(Command, CCodes) },
    !.

% `status = getline < "file"`: update `$0` and capture 1/0/-1 in status.
getline_assignment_action(getline_file_record_capture(Status, File)) -->
    identifier(Status),
    ws, "=", ws,
    "getline", identifier_boundary, ws,
    "<", ws,
    quoted_string(FCodes),
    { string_codes(File, FCodes) },
    !.

% `status = getline var < "file"`: scalar-target form (a dual-slot write).
getline_assignment_action(getline_capture(Status, Var, File)) -->
    identifier(Status),
    ws, "=", ws,
    "getline",
    required_ws,
    identifier(Var),
    ws, "<", ws,
    quoted_string(FCodes),
    { string_codes(File, FCodes) },
    !.

getline_assignment_action(unsupported_getline(capture(Status, Reason))) -->
    identifier(Status),
    ws, "=", ws,
    unsupported_getline_rhs(Reason).

% Main-input captures are deliberately tried after the unsupported dynamic
% redirects so `status = getline var < expr` cannot be accepted as a shorter
% main-input prefix.
getline_assignment_action(getline_main_var_capture(Status, Var)) -->
    identifier(Status),
    ws, "=", ws,
    "getline",
    required_inline_ws,
    identifier(Var),
    !.
getline_assignment_action(getline_main_record_capture(Status)) -->
    identifier(Status),
    ws, "=", ws,
    "getline",
    identifier_boundary.

unsupported_getline_rhs(file_var(Var, File)) -->
    "getline", required_ws, identifier(Var), ws,
    "<", ws, getline_dynamic_file_expr(File),
    !.
unsupported_getline_rhs(file_record_dynamic(File)) -->
    "getline", identifier_boundary, ws,
    "<", ws, getline_dynamic_file_expr(File),
    !.
unsupported_getline_rhs(pipe_var(Command, Var)) -->
    getline_dynamic_pipe_expr(Command), ws,
    "|", ws, "getline", required_inline_ws, identifier(Var),
    !.
unsupported_getline_rhs(pipe(Command)) -->
    getline_dynamic_pipe_expr(Command), ws,
    "|", ws, "getline", identifier_boundary,
    !.

% `x = sprintf("fmt", args...)`: format into a string scalar (the printf format
% engine + string-valued scalars). Tried first -- the `sprintf(` keyword is
% unambiguous.
scalar_value_expr(Sprintf) -->
    sprintf_expr(Sprintf),
    !.
scalar_value_expr(InExpr) -->
    in_arr_value_expr(InExpr),
    !.
% A ternary assignment `x = COND ? A : B`: numeric comparison condition, numeric
% branches (the assignment mirror of the print/printf ternary). Tried first --
% its `?`/`:` structure is unambiguous, and a plain concat / arithmetic RHS has
% no cmp-then-`?`, so it backtracks cleanly.
scalar_value_expr(Ternary) -->
    ternary_expr(Ternary),
    !.
% String-valued assignment: a concatenation `x = $1 $2` (juxtaposition, no
% separator -- the assignment mirror of `print $1 $2`) or a bare string literal
% `x = "text"`. Both make `x` a string scalar (an interned atom id in an i64
% slot). Tried before the numeric clause; concat needs two whitespace-separated
% field_expr operands, so `x = $1 + $2` (arithmetic) still falls through.
scalar_value_expr(concat([First, Second | Rest])) -->
    field_expr(First),
    required_ws,
    field_expr(Second),
    concat_rest(Rest),
    !.
scalar_value_expr(string(Value)) -->
    quoted_string(Codes),
    { string_codes(Value, Codes) },
    !.
% `x = ENVIRON["NAME"]`: the value of an environment variable as a string scalar
% (empty if unset). Tried before the bare scalar expr; the `ENVIRON[` prefix is
% unambiguous.
scalar_value_expr(Environ) -->
    environ_expr(Environ),
    !.
% `x = ARGV[N]`: the N-th command-line argument as a string scalar (empty if out
% of range). Tried before the bare scalar expr; the `ARGV[` prefix is unambiguous.
scalar_value_expr(Argv) -->
    argv_expr(Argv),
    !.
scalar_value_expr(FileName) -->
    filename_expr(FileName),
    !.
scalar_value_expr(Value) -->
    scalar_delta_expr(Value).

%% environ_expr(-Environ)//
%  `ENVIRON["NAME"]` -- read-only lookup of environment variable NAME (a string
%  literal key). Parses to environ(Name).
environ_expr(environ(Name)) -->
    "ENVIRON", ws, "[", ws,
    quoted_string(KCodes), ws, "]",
    { string_codes(Name, KCodes) }.

%% argv_expr(-Argv)//
%  `ARGV[N]` -- the N-th command-line argument (a non-negative integer-literal
%  index). Parses to argv_at(N). ARGV[0] is the program, ARGV[1] the first arg.
argv_expr(argv_at(N)) -->
    "ARGV", ws, "[", ws,
    integer_codes(NCodes),
    { NCodes \== [], number_codes(N, NCodes), N >= 0 },
    ws, "]".

%% filename_expr(-Argv)//
%  `FILENAME` -- the current input filename. plawk streams a single input, which
%  the native driver takes as `ARGV[1]` (the path; `-` denotes stdin), so
%  FILENAME reads as `argv_at(1)` and reuses the ARGV[N] string machinery (a
%  string scalar: print, `x = FILENAME`, and string-equality via assign-then-
%  compare). A trailing identifier_boundary keeps `FILENAMEX` an identifier.
filename_expr(argv_at(1)) -->
    "FILENAME", identifier_boundary.

% match/RSTART/RLENGTH as a scalar RHS: `n = match($0, /re/)`, `x = RSTART`.
% Tried before the arithmetic clause so the `match(` keyword and the special
% names win over a bare-identifier parse.
scalar_delta_expr(match_expr(field(Index), Regex)) -->
    "match", ws, "(", ws,
    "$", integer_codes(ICodes),
    { ICodes \== [], number_codes(Index, ICodes), Index >= 0 },
    ws, ",", ws,
    regex_arg(Regex),
    ws, ")",
    !.
scalar_delta_expr(special('RSTART')) -->
    "RSTART".
scalar_delta_expr(special('RLENGTH')) -->
    "RLENGTH".
scalar_delta_expr(Expr) -->
    i64_binary_surface_expr(Expr).
% A standalone math builtin as the RHS (`x = sqrt($1)`). Tried after the binary
% surface expr so `x = sqrt($1) + 1` parses as the full arithmetic expression.
scalar_delta_expr(Expr) -->
    math_call_expr(Expr).
% Bare float leaves before the integer clause: "0.5" must not stop at
% the integer prefix "0".
scalar_delta_expr(Expr) -->
    float_field_expr(Expr).
scalar_delta_expr(Expr) -->
    float_call_expr(Expr).
scalar_delta_expr(Expr) -->
    float_literal_expr(Expr).
scalar_delta_expr(int(Value)) -->
    integer_codes(ValueCodes),
    { ValueCodes \== [],
      number_codes(Value, ValueCodes),
      Value >= 0 }.
scalar_delta_expr(special('NR')) -->
    "NR".
scalar_delta_expr(special('NR')) -->
    "FNR", identifier_boundary.
scalar_delta_expr(special('NF')) -->
    "NF".
scalar_delta_expr(special('ARGC')) -->
    "ARGC".
scalar_delta_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
scalar_delta_expr(int(Field)) -->
    int_field_expr(int(Field)).
scalar_delta_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
scalar_delta_expr(length(field(0))) -->
    "length",
    length_no_argument.
scalar_delta_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { Field = field(_),
      NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.
scalar_delta_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
scalar_delta_expr(var(Name)) -->
    identifier(Name).

print_action(print(Fields)) -->
    "print",
    required_ws,
    print_fields(Fields).

% `emit E` -- the producer counterpart of `print` inside a generator block
% (PLAWK_GENERATOR_BLOCKS.md). Instead of writing a record to stdout, it
% contributes the value of E to the generated relation's solution set. Parses
% to emit(Expr) reusing the print field-expression grammar (a field, number,
% string, or arithmetic expression). Explicit emission keeps the value typed
% (design doc section 2.1). `emit` is only meaningful inside a `gen { ... }`
% block; the codegen (bin/plawk) rejects it elsewhere.
%
% A TUPLE emit (`emit (A, B, ...)` -> a row of the generated relation, arity n)
% parses to emit(tuple([V1, ..., Vn])). Tried first -- the `(` after `emit`
% distinguishes it. Tuple elements are constants (integer or string literals);
% a computed tuple element is a runtime-collection follow-on.
emit_action(emit(tuple([V0, V1 | Vs]))) -->
    "emit", required_ws, "(", ws,
    emit_tuple_value(V0), ws, ",", ws, emit_tuple_value(V1),
    emit_tuple_rest(Vs), ws, ")",
    !.
emit_action(emit(Expr)) -->
    "emit",
    required_ws,
    field_expr(Expr).
% A bare numeric literal (`emit 1`) -- the canonical generator form. field_expr
% (above) covers fields, strings, NR/NF and arithmetic but not a leading bare
% integer, so fall back to an integer literal, carried as int(N).
emit_action(emit(int(N))) -->
    "emit",
    required_ws,
    signed_integer_value(N).

% The remaining elements of a tuple emit, comma-separated.
emit_tuple_rest([V | Vs]) -->
    ws, ",", ws, emit_tuple_value(V),
    !,
    emit_tuple_rest(Vs).
emit_tuple_rest([]) -->
    [].
% A tuple element: an integer or string literal (constant emits -> facts), or a
% bound loop variable (an input-iterator projection -> a derived rule).
emit_tuple_value(int(N)) -->
    signed_integer_value(N),
    !.
emit_tuple_value(string(S)) -->
    quoted_string(Codes),
    !,
    { string_codes(S, Codes) }.
emit_tuple_value(var(V)) -->
    identifier(V).

printf_action(printf(string(Format), Args)) -->
    "printf",
    required_ws,
    quoted_string(FormatCodes),
    printf_args(Args),
    { string_codes(Format, FormatCodes) }.

%% sprintf_expr(-Expr)//
%
%  `sprintf("fmt", args...)` -- the same format + args as `printf`, but the
%  result is a string (built into a buffer and interned) rather than printed.
%  Used as a scalar assignment RHS (`x = sprintf(...)`). The format is a string
%  literal; the args are the printf argument list (a leading comma before each).
sprintf_expr(sprintf(string(Format), Args)) -->
    "sprintf",
    ws,
    "(",
    ws,
    quoted_string(FormatCodes),
    printf_args(Args),
    ws,
    ")",
    { string_codes(Format, FormatCodes) }.

%% foreach_action(-Action)//
%
%  foreach { actions } - run the block once per repetition element of
%  the current record; inside, $1..$M are the element's fields.
foreach_action(foreach(Actions)) -->
    "foreach",
    identifier_boundary,
    ws,
    action_block(Actions).

%% writebin_action(-Action)//
%
%  writebin expr, expr, ... - emit one fixed-layout binary record on
%  stdout, laid out per BEGIN { OUTFMT = "..." }. With a tagged-union
%  OUTFMT (`OUTFMT = "case(arm0 | arm1)"`), each site statically
%  targets one arm: `writebin case K, expr, ...` emits the 8-byte tag
%  K then arm K's slots.
writebin_action(writebin_arm(Index, Fields)) -->
    "writebin",
    required_ws,
    "case",
    identifier_boundary,
    ws,
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    },
    ws,
    ",",
    !,
    ws,
    print_fields(Fields).
writebin_action(writebin(Fields)) -->
    "writebin",
    required_ws,
    print_fields(Fields).

printf_args([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    printf_arg_expr(Arg),
    printf_args_rest(Args).
printf_args([]) -->
    [].

printf_args_rest([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    printf_arg_expr(Arg),
    printf_args_rest(Args).
printf_args_rest([]) -->
    [].

% A printf argument: a ternary (tried first for its `?`/`:` structure) or a
% plain field_expr.
printf_arg_expr(Ternary) -->
    ternary_expr(Ternary),
    !.
printf_arg_expr(Expr) -->
    field_expr(Expr).

print_fields([Field | Fields]) -->
    print_field_expr(Field),
    print_fields_rest(Fields).

print_fields_rest([Field | Fields]) -->
    ws,
    ",",
    ws,
    !,
    print_field_expr(Field),
    print_fields_rest(Fields).
print_fields_rest([]) -->
    [].

% A print field: the general field expression, or a bare numeric literal
% (`print 1`, `print -5`). awk prints a number as its text, and a string
% literal print field is now compilable, so a bare integer literal lowers to
% the string of its digits (`print 1` == `print "1"`) -- correct output with no
% codegen change. This is PRINT-specific: `field_expr` (shared with `emit`,
% arithmetic, ...) is left untouched, so a bare integer keeps its numeric
% meaning there (`emit 1` stays the integer 1, not the atom '1'). field_expr is
% tried first, so arithmetic (`print 1 + 2`) and floats are unaffected; only a
% lone integer falls through to the literal clause.
% String concatenation by juxtaposition (awk): `print $1 $2`, `print "x" $1`,
% `print $1 "-" $2` output the operands adjacent, with NO field separator (unlike
% the comma list). Two or more operands separated by whitespace parse to
% concat([...]). Each operand is a field_expr, so arithmetic binds tighter than
% concatenation (`$1 $2 + $3` == concat($1, $2 + $3)) and a comma still splits
% print args. A single operand keeps its plain form (no concat wrapper).
% Ternary `COND ? A : B` (awk `?:`). COND is a numeric comparison `L <op> R`;
% both branches are numeric field_exprs. Tried before concat / plain field so
% the `?`/`:` structure is seen first. Defined one level above field_expr (its
% condition and branches call field_expr) so there is no left recursion.
print_field_expr(InExpr) -->
    in_arr_value_expr(InExpr),
    !.
print_field_expr(Environ) -->
    environ_expr(Environ),
    !.
print_field_expr(Argv) -->
    argv_expr(Argv),
    !.
print_field_expr(FileName) -->
    filename_expr(FileName),
    !.
print_field_expr(Ternary) -->
    ternary_expr(Ternary),
    !.
print_field_expr(concat([First, Second | Rest])) -->
    field_expr(First),
    required_ws,
    field_expr(Second),
    concat_rest(Rest),
    !.
print_field_expr(Field) -->
    field_expr(Field),
    !.
print_field_expr(string(Text)) -->
    signed_integer_value(N),
    { format(string(Text), '~w', [N]) }.

concat_rest([Operand | Rest]) -->
    required_ws,
    field_expr(Operand),
    !,
    concat_rest(Rest).
concat_rest([]) -->
    [].

% A ternary `L <op> R ? THEN : ELSE`. The condition is a numeric comparison and
% each branch is a numeric field_expr (nested ternaries are a follow-on). Used
% by print fields and printf args.
ternary_expr(ternary(cmp(Left, Op, Right), Then, Else)) -->
    ternary_operand(Left),
    ws,
    numeric_cmp_op(Op),
    ws,
    ternary_operand(Right),
    ws,
    "?",
    ws,
    ternary_operand(Then),
    ws,
    ":",
    ws,
    ternary_operand(Else).

% A ternary condition operand or branch: a field_expr (`$N`, `NR`, `NF`,
% arithmetic, `length(...)`, ...) or a bare integer literal (which field_expr
% does not accept on its own). Both lower to i64.
ternary_operand(Expr) -->
    field_expr(Expr),
    !.
ternary_operand(int(Value)) -->
    signed_integer_value(Value).

% `match(SRC, /re/)` -- search SRC (a field `$N`, `$0` = whole record) for the
% ERE; returns the 1-based match position (0 if none) and sets RSTART/RLENGTH.
% Tried before the arithmetic clause so the leading `match(` keyword wins.
field_expr(match_expr(field(Index), Regex)) -->
    "match", ws, "(", ws,
    "$", integer_codes(ICodes),
    { ICodes \== [], number_codes(Index, ICodes), Index >= 0 },
    ws, ",", ws,
    regex_arg(Regex),
    ws, ")",
    !.
field_expr(special('RSTART')) -->
    "RSTART".
field_expr(special('RLENGTH')) -->
    "RLENGTH".
% RT is the text matched by the current regex record separator. It is a
% read-only string special (unlike the numeric RSTART/RLENGTH pair), so it is
% admitted as a print field but not as an arithmetic scalar.
field_expr(special('RT')) -->
    "RT", identifier_boundary.
field_expr(Expr) -->
    i64_binary_surface_expr(Expr).
field_expr(special('NR')) -->
    "NR".
field_expr(special('NR')) -->
    "FNR", identifier_boundary.
field_expr(special('NF')) -->
    "NF".
field_expr(special('ARGC')) -->
    "ARGC".

%% regex_arg(-Regex)//
%
%  A regex argument to match/sub/gsub: a `/re/` literal or a `"re"` string. Both
%  yield the ERE source string (a single-char string is still a literal-char ERE).
regex_arg(Regex) -->
    "/", regex_body_codes(Codes), "/",
    { Codes \== [], string_codes(Regex, Codes) },
    !.
regex_arg(Regex) -->
    quoted_string(Codes),
    { Codes \== [], string_codes(Regex, Codes) }.
field_expr(int(Field)) -->
    int_field_expr(int(Field)).
field_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(length(field(0))) -->
    "length",
    length_no_argument.
field_expr(substr(Field, Start, Len)) -->
    "substr",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    integer_codes(StartCodes),
    ws,
    ",",
    ws,
    integer_codes(LenCodes),
    ws,
    ")",
    { Field = field(_),
      StartCodes \== [], LenCodes \== [],
      number_codes(Start, StartCodes), Start >= 1,
      number_codes(Len, LenCodes), Len >= 0 }.
% Two-argument `substr(s, m)` returns the tail from position m to the end of the
% string (awk semantics). The `to_end` length flows through the same slice path;
% the emitter maps it to a max byte count that the runtime clamps to whatever
% remains after m.
field_expr(substr(Field, Start, to_end)) -->
    "substr",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    integer_codes(StartCodes),
    ws,
    ")",
    { Field = field(_),
      StartCodes \== [],
      number_codes(Start, StartCodes), Start >= 1 }.
field_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { Field = field(_),
      NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.
field_expr(tolower(Field)) -->
    "tolower",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(toupper(Field)) -->
    "toupper",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(assoc(var(Name), KeyExpr)) -->
    table_ident(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !.
field_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
field_expr(string(Value)) -->
    quoted_string(ValueCodes),
    { string_codes(Value, ValueCodes)
    }.
field_expr(Expr) -->
    float_field_expr(Expr),
    !.
field_expr(Expr) -->
    float_call_expr(Expr),
    !.
field_expr(Expr) -->
    blob_call_expr(Expr),
    !.
field_expr(Expr) -->
    float_literal_expr(Expr),
    !.
field_expr(Expr) -->
    math_call_expr(Expr),
    !.
field_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
field_expr(var(Name)) -->
    identifier(Name).

int_field_expr(int(Field)) -->
    "int",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.

%% math_call_expr(-Expr)//
%
%  Standard awk numeric builtins that return a double: sqrt / sin / cos / exp /
%  log (unary) and atan2 (binary). Parsed to math_call(Fn, Args); each argument
%  is a full arithmetic expression, evaluated in f64 at codegen time (see
%  plawk_f64_expr_ir). Tried before the generic prolog_call so these names are
%  builtins rather than foreign predicate calls. A name only commits when
%  directly followed by `(`, so a variable that merely starts with one of these
%  names (e.g. `sine`) still parses as an identifier.
math_call_expr(math_call(rand, [])) -->
    "rand",
    ws,
    "(",
    ws,
    ")",
    !.
math_call_expr(math_call(atan2, [Y, X])) -->
    "atan2",
    ws,
    "(",
    ws,
    i64_additive_expr(Y),
    ws,
    ",",
    ws,
    i64_additive_expr(X),
    ws,
    ")",
    !.
math_call_expr(math_call(Fn, [Arg])) -->
    math_unary_name(Fn),
    ws,
    "(",
    ws,
    i64_additive_expr(Arg),
    ws,
    ")",
    !.

math_unary_name(sqrt) --> "sqrt".
math_unary_name(sin) --> "sin".
math_unary_name(cos) --> "cos".
math_unary_name(exp) --> "exp".
math_unary_name(log) --> "log".

%% i64_binary_surface_expr(-Expr)//
%
%  General native i64 arithmetic with awk precedence: * / % bind tighter
%  than + -, both levels associate left, and parentheses group. Factors
%  are the native i64 primaries plus integer literals and bare numeric
%  field coercions such as `$3` (zero when the field is not a strict
%  signed decimal). The top-level result must contain at least one
%  operator so bare primaries keep their existing print/slice meaning.
i64_binary_surface_expr(Expr) -->
    i64_additive_expr(Expr),
    { i64_binary_expr_ast(Expr) }.

i64_binary_expr_ast(Expr) :-
    compound(Expr),
    functor(Expr, Functor, 2),
    memberchk(Functor, [add_i64, sub_i64, mul_i64, div_i64, mod_i64, pow_i64]).

i64_additive_expr(Expr) -->
    i64_multiplicative_expr(First),
    i64_additive_chain(First, Expr).

i64_additive_chain(Acc, Expr) -->
    ws,
    i64_additive_operator(Functor),
    ws,
    i64_multiplicative_expr(Right),
    !,
    { Acc1 =.. [Functor, Acc, Right] },
    i64_additive_chain(Acc1, Expr).
i64_additive_chain(Expr, Expr) -->
    [].

i64_multiplicative_expr(Expr) -->
    i64_power_expr(First),
    i64_multiplicative_chain(First, Expr).

i64_multiplicative_chain(Acc, Expr) -->
    ws,
    i64_multiplicative_operator(Functor),
    ws,
    i64_power_expr(Right),
    !,
    { Acc1 =.. [Functor, Acc, Right] },
    i64_multiplicative_chain(Acc1, Expr).
i64_multiplicative_chain(Expr, Expr) -->
    [].

% Exponentiation `L ^ R` / `L ** R`: binds tighter than `* / %` and is
% right-associative (`2 ^ 3 ^ 2` == `2 ^ (3 ^ 2)`), as in awk. It is always
% computed in floating point (the result keeps its fraction), so the AST node
% pow_i64/2 is typed double at codegen -- see plawk_expr_is_double/1.
i64_power_expr(Expr) -->
    i64_factor_expr(Base),
    i64_power_rest(Base, Expr).

i64_power_rest(Base, pow_i64(Base, Exp)) -->
    ws,
    i64_power_operator,
    ws,
    !,
    i64_power_expr(Exp).
i64_power_rest(Expr, Expr) -->
    [].

% `**` is matched before `^`; both are the same operator. The `**` spelling is
% distinct from `*` because it is only reached at the power level, inside a
% single factor, before the multiplicative chain sees an operator.
i64_power_operator -->
    "**".
i64_power_operator -->
    "^".

i64_additive_operator(add_i64) -->
    "+".
i64_additive_operator(sub_i64) -->
    "-".

i64_multiplicative_operator(mul_i64) -->
    "*".
i64_multiplicative_operator(div_i64) -->
    "/".
i64_multiplicative_operator(mod_i64) -->
    "%".

i64_factor_expr(Expr) -->
    "(",
    ws,
    i64_additive_expr(Expr),
    ws,
    ")",
    !.
i64_factor_expr(Expr) -->
    i64_binary_primary_expr(Expr),
    !.
i64_factor_expr(Expr) -->
    float_field_expr(Expr),
    !.
i64_factor_expr(Expr) -->
    float_literal_expr(Expr),
    !.
i64_factor_expr(int(Value)) -->
    integer_codes(ValueCodes),
    !,
    { ValueCodes \== [],
      number_codes(Value, ValueCodes)
    }.
i64_factor_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    !,
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
i64_factor_expr(Expr) -->
    math_call_expr(Expr),
    !.
i64_factor_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
% An assoc lookup `arr[k]` as an arithmetic operand, e.g. `$2 / total[$1]`
% (per-key normalise). Tried before the bare identifier -- the `[` commits.
i64_factor_expr(assoc(var(Name), KeyExpr)) -->
    table_ident(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !.
i64_factor_expr(var(Name)) -->
    identifier(Name).

%% prolog_call_expr(-Expr)//
%
%  A named Prolog predicate as an i64 expression: `pred(args...)` calls
%  the compiled predicate with one extra trailing output argument and
%  yields its integer binding, or 0 when the call fails or binds a
%  non-integer.
% dyncall(args...) is reserved: it routes to a runtime-loaded .wamo
% object's entry (BEGIN { DYNLOAD = "file.wamo" }) rather than a
% compiled predicate, so it parses to its own node and never touches the
% compiled-foreign-call machinery. The cut fires only after a full
% `dyncall(...)`, so an identifier like `dyncalls(...)` still falls
% through to the generic prolog call below.
% dyncall_at(Source, args...) is the dynamic-source form: Source (a field
% or string literal) names the .wamo object at runtime, chosen per call,
% and args... are the entry's inputs. Reserved like dyncall; parsed before
% it so the longer keyword wins.
% The source may also be compile(field-or-string): compile the Prolog
% source text through the shipped bootstrap-compiler object at runtime
% (the eval surface -- JIT roadmap item 5 payoff) and dyncall the
% freshly compiled grammar. compile() dedups by source text, so a
% per-record `dyncall_at(compile($1), $2)` compiles each distinct
% grammar once and reuses it from the cache thereafter.
% dyncall_at@name(Source, args...) selects a NAMED entry of the
% runtime-chosen object -- the composition of the two selection axes:
% the source is runtime data (a path expression or a compile() handle),
% the entry name is a compile-time token. Because the object is not
% fixed, the name resolves per call against the loaded VM's
% materialized entry table (@wam_object_vm_entry_pc) rather than a
% startup-cached PC. Parsed before the bare dyncall_at so the @ form
% wins.
% compile(src) / compile_file(path) as EXPRESSIONS: yield the grammar
% HANDLE (an i64 registry index; 0 on failure) for storing in a scalar.
% Content dedup makes a per-record re-assignment a registry hit, so
% `h = compile("...")` in a rule body costs one compile per distinct
% source. Reserved like dyncall; `compilex(...)` still parses as an
% ordinary identifier call.
prolog_call_expr(compile_file_handle(Arg)) -->
    "compile_file",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
prolog_call_expr(compile_handle(Arg)) -->
    "compile",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
prolog_call_expr(dyncall_at_named(Name, Source, Args)) -->
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    !.
prolog_call_expr(dyncall_at(Source, Args)) -->
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    !.
% dyncall@name(args...) selects a named entry of the runtime object (a
% multi-entry .wamo, one file exposing several predicates via
% wamo_entries([...])). The @name is a compile-time token -- the entry
% name is fixed in the source, so the shim resolves its label index once
% at startup (via @wam_object_entry_index) and caches the PC. Parsed
% before the bare `dyncall(...)` so the `@` form wins; a plain
% `dyncall(...)` still falls through to the clause below.
prolog_call_expr(dyncall_named(Name, Args)) -->
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    !.
prolog_call_expr(dyncall(Args)) -->
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    !.
prolog_call_expr(prolog_call(Name, Args)) -->
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")".

%% float_literal_expr(-Expr)//
%
%  A decimal float literal such as 2.5 or 0.1, kept exact as
%  float_const(Mantissa, Denominator) with Denominator = 10^k so
%  codegen can emit a correctly rounded double without a lossy
%  Prolog-float round trip (LLVM rejects inexact decimal FP text).
float_literal_expr(float_const(Mantissa, Denominator)) -->
    integer_codes(IntCodes),
    ".",
    integer_codes(FracCodes),
    { IntCodes \== [],
      FracCodes \== [],
      append(IntCodes, FracCodes, AllCodes),
      number_codes(Mantissa, AllCodes),
      length(FracCodes, FracLen),
      Denominator is 10 ** FracLen
    }.

%% float_call_expr(-Expr)//
%
%  float(name(args)): a compiled-Prolog call whose output argument is
%  numeric and lands in a double context -- Float results keep their
%  fraction (an i64-context call would truncate the surface to
%  integers). The float(...) wrapper is what selects the
%  double-returning wrapper at codegen time.
%
%  float(dyncall(...)) / float(dyncall_at(...)) are the runtime-object
%  variants: the loaded grammar's numeric output is read as a double.
%  These must precede the generic clause, whose inner prolog_call_expr
%  would otherwise parse `dyncall` as an ordinary identifier call.
% float(dyncall_at@name(Source, args...)): a named entry on a runtime
% source, read as a double -- parsed before the bare at form so the @
% form wins.
float_call_expr(float_dyncall_at_named(Name, Source, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall_at(Source, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall_named(Name, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall(Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_call(Name, Args)) -->
    "float",
    ws,
    "(",
    ws,
    prolog_call_expr(prolog_call(Name, Args)),
    ws,
    ")".

%% blob_call_expr(-Expr)//
%
%  blob(dyncall(...)) / blob(dyncall_at(...)): read the runtime grammar's
%  Atom output as opaque bytes (a slice), for print / writebin positions.
%  Reserved like the dyncall forms; the inner keyword disambiguates.
% blob(dyncall_at@name(Source, args...)): a named entry on a runtime
% source, read as opaque bytes -- parsed before the bare at form.
blob_call_expr(blob_dyncall_at_named(Name, Source, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall_at(Source, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall_named(Name, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall(Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.

%% float_field_expr(-Expr)//
%
%  awk-style numeric coercion to double: float($N) parses the field
%  with strtod semantics (leading number, trailing text ignored, 0.0
%  when non-numeric).
float_field_expr(float_field(Index)) -->
    "float",
    ws,
    "(",
    ws,
    "$",
    integer_codes(IndexCodes),
    ws,
    ")",
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.

i64_binary_primary_expr(special('NR')) -->
    "NR".
i64_binary_primary_expr(special('NR')) -->
    "FNR", identifier_boundary.
i64_binary_primary_expr(special('NF')) -->
    "NF".
i64_binary_primary_expr(special('ARGC')) -->
    "ARGC".
i64_binary_primary_expr(Expr) -->
    int_field_expr(Expr).
i64_binary_primary_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    simple_field_expr(Field),
    ws,
    ")".
i64_binary_primary_expr(length(field(0))) -->
    "length",
    length_no_argument.
i64_binary_primary_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    simple_field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.

simple_field_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.

% A subscript is either one expression, or a comma-separated list
% `[i,j,...]` -- awk's multi-dimensional array notation, which joins the
% subscripts with SUBSEP into one string key. A single subscript collapses to
% the bare expression (unchanged from before); two or more yield
% subsep_key([E1,E2,...]), which the codegen lowers by interning the fields
% joined with the SUBSEP byte.
assoc_key_expr(Key) -->
    assoc_key_atom(First),
    assoc_key_rest(Rest),
    { ( Rest == []
      ->  Key = First
      ;   Key = subsep_key([First | Rest])
      )
    }.

assoc_key_rest([Atom | Rest]) -->
    ws, ",", ws,
    assoc_key_atom(Atom),
    assoc_key_rest(Rest).
assoc_key_rest([]) -->
    [].

assoc_key_atom(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
assoc_key_atom(int(Value)) -->
    signed_integer_value(Value),
    !.
assoc_key_atom(string(Value)) -->
    quoted_string(ValueCodes),
    { ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.
% counts[blob(dyncall...)]++ -- key an assoc table by a runtime
% grammar's byte output (JIT roadmap item 2 follow-on). Must precede
% var(Name): identifier would otherwise eat "blob".
assoc_key_atom(Blob) -->
    blob_call_expr(Blob),
    !.
assoc_key_atom(var(Name)) -->
    identifier(Name).

identifier(Name) -->
    identifier_start(Start),
    identifier_rest(Rest),
    { atom_codes(Name, [Start | Rest]) }.

% A table name, optionally namespace-qualified (`ns.table`, phase 8.9 PR 4,
% PLAWK_MULTIPASS_CACHE.md §3.7). A bare name stays the atom `table`; the
% qualified form is the dotted atom `'ns.table'`, matching how a namespaced
% `declare` qualifies it (plawk_qualify). The dotted form's LOCAL part (after
% the dot) is the sub-DB it routes to. Falls back to a bare identifier, so
% existing programs are unchanged. Used only in TABLE-name positions (readers
% and assoc write/read targets), never for scalars or loop variables.
table_ident(Name) -->
    identifier(NS), ".", identifier(Local),
    !,
    { atomic_list_concat([NS, '.', Local], Name) }.
table_ident(Name) -->
    identifier(Name).

identifier_start(Code) -->
    [Code],
    { code_type(Code, alpha) -> true ; Code =:= 0'_ }.

identifier_rest([Code | Codes]) -->
    [Code],
    { code_type(Code, alnum) -> true ; Code =:= 0'_ },
    !,
    identifier_rest(Codes).
identifier_rest([]) -->
    [].

identifier_boundary([Code | Rest], [Code | Rest]) :-
    \+ identifier_continue_code(Code).
identifier_boundary([], []).

% Bare `length` (no argument) is awk shorthand for `length($0)`. A non-consuming
% lookahead: `length` must be a complete token that is NOT the start of a call,
% so `length(...)` / `length (...)` stay function calls (handled by the
% parenthesised clauses) and `lengthy` / `length_x` stay identifiers.
length_no_argument(Rest, Rest) :-
    \+ ( Rest = [Code | _], identifier_continue_code(Code) ),
    \+ phrase(length_call_open, Rest, _).

length_call_open -->
    ws, "(".

identifier_continue_code(Code) :-
    code_type(Code, alnum),
    !.
identifier_continue_code(0'_).

required_ws -->
    [Code],
    { code_type(Code, space) },
    ws.

% A getline scalar target must occur in the same statement as the keyword.
% Unlike required_ws//0 this deliberately stops before newlines and comments,
% so `getline\nprint $0` is a record getline followed by a print rather than a
% read into a scalar named `print`.
required_inline_ws -->
    [Code],
    { code_type(Code, space), Code =\= 0'\n, Code =\= 0'\r },
    inline_ws.

inline_ws -->
    [Code],
    { code_type(Code, space), Code =\= 0'\n, Code =\= 0'\r },
    !,
    inline_ws.
inline_ws -->
    [].

% Whitespace, including newlines and awk-style # comments (a comment
% runs to end of line; its terminating newline still counts as a
% statement separator in action_sep//0). `#` is not a token anywhere
% in the surface, and strings/regex bodies never route through ws//0,
% so comments cannot be consumed inside literals.
ws -->
    [Code],
    { code_type(Code, space) },
    !,
    ws.
ws -->
    "#",
    !,
    comment_rest,
    ws.
ws -->
    [].

comment_rest -->
    [Code],
    { Code =\= 0'\n },
    !,
    comment_rest.
comment_rest -->
    [].

eos([], []).

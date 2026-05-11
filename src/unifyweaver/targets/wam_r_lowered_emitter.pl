:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_r_lowered_emitter.pl -- WAM-lowered R emission (Phase 3)
%
% Plan mirrors the Haskell hybrid path in
%   docs/design/WAM_HASKELL_LOWERED_{PHILOSOPHY,SPECIFICATION,
%                                     IMPLEMENTATION_PLAN}.md
%
% Phase 2 covered deterministic single-clause predicates and emitted
% one R function per predicate, delegating every instruction to
% WamRuntime$step. Phase 3 (this iteration) adds two refinements:
%
%   1. Expanded lowerability. Multi-clause predicates whose first
%      clause is in the supported set are now lowerable with reason
%      multi_clause_1. The lowered function inlines clause 1 and, if
%      clause 1 fails, drops back into the array path so the standard
%      backtracking machinery can run clause 2+. This is the same
%      design the Haskell lowered emitter uses for its multi-clause
%      path.
%
%   2. Native per-instruction emission for the simple register ops
%      (allocate/deallocate, put_constant/put_variable/put_value,
%      get_variable). These never fail and never allocate heap, so we
%      avoid the WamRuntime$step dispatch entirely. Anything that
%      involves deref/unify/heap-build (get_constant, get_value,
%      get_structure, *_list, unify_*, set_*, builtin_call,
%      call_foreign) still delegates to step.
%
% Output under emit_mode(interpreter) is unchanged; this file is only
% consulted under emit_mode(functions) / emit_mode(mixed([...])).

:- module(wam_r_lowered_emitter, [
    wam_r_lowerable/3,           % +Pred, +WamCode, -Reason
    lower_predicate_to_r/4,      % +Pred, +WamCode, +Options, -Entry
    r_lowered_func_name/2,       % +Functor/Arity, -RFuncName
    gather_pred_mode_records/2   % +PredIndicator, -Records
]).

:- use_module(library(lists)).
:- use_module('../core/binding_state_analysis').
:- use_module('../core/demand_analysis').

% =====================================================================
% Emission plan
% =====================================================================
%
% A predicate's WAM text is summarised into a `plan(Mode, AltLabel,
% ClauseLines)` record:
%
%   - Mode is `deterministic` or `multi_clause_1`.
%   - AltLabel is the label to backtrack to under multi_clause_1; it's
%     the literal label string from the try_me_else operand. For
%     deterministic predicates this is the atom `none`.
%   - ClauseLines is the list of WAM-text lines that make up the
%     emitted body. For deterministic mode this is every non-switch
%     instruction line; for multi_clause_1 it's the lines of clause 1
%     only (the lines between try_me_else and the corresponding
%     trust_me, ending at proceed).
%
% switch_on_constant / switch_on_structure prefixes are dropped: they
% were optimisations the array path uses to short-circuit dispatch.
% Skipping them in the lowered path is correctness-preserving (clause
% 1 either matches or we fall back to the array, which still runs the
% switch).

build_emission_plan(WamCode, plan(Mode, AltInfo, ClauseInfo)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", AllLines),
    skip_to_first_real_instr(AllLines, Filtered),
    classify_clause_shape(Filtered, plan(Mode, AltInfo, ClauseInfo)).

% Drop blank lines, label-only lines, and switch_on_* dispatch prefixes
% until we find the first real WAM instruction.
skip_to_first_real_instr([], []).
skip_to_first_real_instr([Line|Rest], Out) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   skippable_prefix_line(Parts)
    ->  skip_to_first_real_instr(Rest, Out)
    ;   Out = [Line|Rest]
    ).

skippable_prefix_line([]).
skippable_prefix_line([First|_]) :- sub_string(First, _, 1, 0, ":").
skippable_prefix_line(["switch_on_constant"|_]).
skippable_prefix_line(["switch_on_structure"|_]).

% Classifier. Three shapes:
%
%   deterministic  -- single clause, no try_me_else / trust_me boundary.
%     ClauseInfo = ClauseLines (a single line list).
%
%   multi_clause_1 -- multi-clause, but only clause 1 is lowered;
%     clauses 2+ stay in the WAM array and the lowered fn falls back
%     to the run loop on clause-1 failure. ClauseInfo = ClauseLines.
%
%   multi_clause_n(N) -- multi-clause, ALL clauses lowered inline.
%     ClauseInfo = list of clause-line groups [Clause1Lines, ...,
%     ClauseNLines]. No array fallback (the array path is still
%     populated for caller-side backtracking via lowered_dispatch,
%     but the lowered fn handles clause-2+ inline by itself).
%
% Strategy: walk the filtered lines, splitting on retry_me_else /
% trust_me boundaries (which always sit directly after a clause-entry
% label like L_pn_1_2:). build_emission_plan returns multi_clause_n
% when more than one clause group is found; the caller (lower_predicate_to_r)
% chooses between multi_clause_n and multi_clause_1 based on whether ALL
% clauses are lowerable.
classify_clause_shape([FirstLine|Rest], Plan) :-
    wam_r_target:tokenize_wam_line(FirstLine, ["try_me_else", AltStr]), !,
    atom_string(AltAtom, AltStr),
    split_all_clauses(Rest, [], AllClauseGroups),
    (   AllClauseGroups = [Clause1]
    ->  % Only one clause group found after try_me_else. Should not
        % normally happen (try_me_else implies a sibling), but be
        % defensive.
        Plan = plan(multi_clause_1, AltAtom, Clause1)
    ;   AllClauseGroups = [Clause1|_]
    ->  Plan = plan(multi_clause_n(N), AltAtom,
                    multi(AllClauseGroups, Clause1)),
        length(AllClauseGroups, N)
    ).
classify_clause_shape(Lines, plan(deterministic, none, Lines)).

% split_all_clauses(+Lines, +CurAccRev, -ClauseGroups)
%
% Walks the post-try_me_else line stream and splits into per-clause
% line lists. Boundary markers (retry_me_else / trust_me) and their
% preceding entry labels are dropped; everything in between is one
% clause's body. The first clause's `proceed` ends its body but is
% kept (the emitter renders it as `return(TRUE)`).
split_all_clauses([], CurAccRev, [Clause]) :-
    reverse(CurAccRev, Clause).
split_all_clauses([Line|Rest], CurAccRev, Clauses) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] ; Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  % Empty or pure label -- drop, keep accumulating same clause.
        split_all_clauses(Rest, CurAccRev, Clauses)
    ;   Parts = ["retry_me_else"|_]
    ->  reverse(CurAccRev, Clause),
        Clauses = [Clause|MoreClauses],
        split_all_clauses(Rest, [], MoreClauses)
    ;   Parts == ["trust_me"]
    ->  reverse(CurAccRev, Clause),
        Clauses = [Clause|MoreClauses],
        split_all_clauses(Rest, [], MoreClauses)
    ;   split_all_clauses(Rest, [Line|CurAccRev], Clauses)
    ).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_r_lowerable(+Pred, +WamCode, -Reason) is semidet.
%  Reason is one of `deterministic`, `multi_clause_n(N)`, or
%  `multi_clause_1`. Decided against the emission plan's clause lines.
%
%  For multi-clause predicates we prefer multi_clause_n when ALL
%  clauses' lines are supported (every clause runs inline, no array
%  fallback). When only clause 1 is supported, we fall back to
%  multi_clause_1 (clause 1 inline, clauses 2+ via the WAM array).
wam_r_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, Plan), _, fail),
    lowerability_of_plan(Plan, Reason).

lowerability_of_plan(plan(deterministic, _, Lines), deterministic) :-
    forall(member(Line, Lines), line_supported(Line)).
lowerability_of_plan(plan(multi_clause_1, _, Lines), multi_clause_1) :-
    forall(member(Line, Lines), line_supported(Line)).
lowerability_of_plan(plan(multi_clause_n(N), _, multi(AllGroups, Clause1)),
                     Reason) :-
    (   forall(( member(Group, AllGroups), member(Line, Group) ),
               line_supported(Line))
    ->  Reason = multi_clause_n(N)
    ;   % All-clauses lowering failed; degrade to clause-1-only if
        % clause 1 alone is still lowerable.
        forall(member(Line, Clause1), line_supported(Line)),
        Reason = multi_clause_1
    ).

line_supported(Line) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   parts_supported(Parts)
    ).

parts_supported(["allocate"]).
parts_supported(["deallocate"]).
parts_supported(["proceed"]).
parts_supported(["fail"]).
parts_supported(["get_constant", _, _]).
parts_supported(["get_variable", _, _]).
parts_supported(["get_value", _, _]).
parts_supported(["get_structure", _, _]).
parts_supported(["get_list", _]).
parts_supported(["put_constant", _, _]).
parts_supported(["put_variable", _, _]).
parts_supported(["put_value", _, _]).
parts_supported(["put_structure", _, _]).
parts_supported(["put_list", _]).
parts_supported(["unify_variable", _]).
parts_supported(["unify_value", _]).
parts_supported(["unify_constant", _]).
parts_supported(["set_variable", _]).
parts_supported(["set_value", _]).
parts_supported(["set_constant", _]).
parts_supported(["call", _]).
parts_supported(["call", _, _]).
parts_supported(["execute", _]).
parts_supported(["execute", _, _]).
parts_supported(["call_foreign", _, _]).
parts_supported(["builtin_call", _, _]).
parts_supported(["arg", _, _, _]).

% =====================================================================
% Function-name generation
% =====================================================================

%% r_lowered_func_name(+Functor/Arity, -RFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3".
r_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_r_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_r_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(r_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

r_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
r_safe_code(_, 0'_).

% =====================================================================
% Lowered-function emission
% =====================================================================

%% lower_predicate_to_r(+Pred/Arity, +WamCode, +Options, -Entry)
%  Entry = lowered(PredName, FuncName, RCode).
%
%  Options:
%    mode_comments(on)  -- prepend a `# Mode analysis:` comment block
%                          summarising the inferred binding env per
%                          clause. Off by default. Visibility-only;
%                          phase 1 of WAM-R mode-analysis integration.
%    mode_specialise(off) -- disable the analyser-driven instruction
%                          specialisations (currently: inline
%                          get_constant when the target A-register is
%                          provably bound). Phase 2 specialisations
%                          are ON by default since they're always
%                          sound when mode declarations are honest
%                          and degrade to the step-delegating path
%                          when mode info is insufficient.
%
%  See docs/design/WAM_R_MODE_ANALYSIS_PLAN.md for the roadmap.
lower_predicate_to_r(PI, WamCode, Opts,
                     lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    r_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(PlanMode, AltLabel, ClauseInfo)),
    % Decide whether to use multi_clause_n or fall back to
    % multi_clause_1 (mirrors wam_r_lowerable/3's logic so callers
    % that pre-checked lowerability with a particular Reason get the
    % corresponding emitter here).
    wam_r_lowerable(PI, WamCode, EffectiveMode),
    catch(gather_pred_mode_records(PI, Records), _, Records = []),
    set_lowered_mode_context(Records, Opts),
    mode_comment_header_from_records(Opts, Records, ModeHeader),
    (   EffectiveMode == deterministic
    ->  ClauseInfo = ClauseLines,
        emit_deterministic_function(PredName, FuncName, ClauseLines,
                                    ModeHeader, Code)
    ;   EffectiveMode = multi_clause_n(_)
    ->  ClauseInfo = multi(AllClauseGroups, _),
        emit_multi_clause_n_function(PredName, FuncName, AllClauseGroups,
                                     ModeHeader, Code)
    ;   EffectiveMode == multi_clause_1
    ->  % Either the plan was multi_clause_1 (only clause 1 lines),
        % or it was multi_clause_n but not all clauses were lowerable
        % so we degraded. In the latter case ClauseInfo carries the
        % multi-group struct; extract clause 1 from it.
        ( ClauseInfo = multi(_, Clause1) -> ClauseLines = Clause1
        ; ClauseLines = ClauseInfo
        ),
        emit_multi_clause_function(PredName, FuncName, AltLabel,
                                   ClauseLines, ModeHeader, Code)
    ),
    clear_lowered_mode_context,
    % Reference PlanMode so the singleton warning doesn't fire (kept
    % for future debugging hooks; build_emission_plan's full plan
    % shape is observable here without re-parsing).
    ignore(PlanMode = PlanMode).

%% set_lowered_mode_context(+Records, +Opts) is det.
%  Stashes clause 1's `:- mode/1` declaration on a non-backtrackable
%  global so emit_line_parts can consult it without threading extra
%  args through the entire emission chain. Mirrors the shared
%  compiler's wam_clause_binding_records pattern.
%
%  Specialisations are gated by `mode_specialise(off)` in Opts --
%  defaulting to ON. When OFF (or when no mode declaration exists),
%  the stashed value is `none` so the specialised emit_line_parts
%  clauses all fail and the default step-delegating path runs.
%
%  IMPORTANT: the specialisation uses the mode declaration directly,
%  not the analyser's BeforeEnv. The BeforeEnv conflates head-pattern
%  binding (a literal `alice` looks `bound`) with caller-side binding
%  (whether A_k was bound when the caller invoked us). For head-match
%  instructions like get_constant we need the latter, and only the
%  mode declaration gives it to us.
set_lowered_mode_context(Records, Opts) :-
    (   member(mode_specialise(off), Opts)
    ->  ModeDecl = none
    ;   clause1_mode_decl(Records, ModeDecl)
    ),
    b_setval(wam_r_lowered_mode_decl, ModeDecl).

clear_lowered_mode_context :-
    b_setval(wam_r_lowered_mode_decl, none).

%% clause1_mode_decl(+Records, -ModeDecl) is det.
%  Pulls the ModeDecl component of clause 1's record. Returns `none`
%  when there are no records or no declaration.
clause1_mode_decl([mode_record(1, ModeDecl, _, _) | _], ModeDecl) :- !.
clause1_mode_decl(_, none).

%% head_state_for_areg(+AiStr, -State) is semidet.
%  Returns the caller-side binding state for the A-register named by
%  AiStr (e.g. "A1" -> first head arg position). Uses the stashed
%  mode declaration:
%    +  (input)  -> bound
%    -  (output) -> unbound
%    ?  (any)    -> fails (no provable state)
%  Fails if AiStr isn't in the A-register range (1..100), if no mode
%  declaration is stashed, or if the position is out of range in the
%  mode list. Used by the get_constant specialised emit_line_parts
%  clause.
head_state_for_areg(AiStr, State) :-
    wam_r_target:reg_to_int(AiStr, AIdx),
    AIdx >= 1, AIdx =< 100,
    catch(b_getval(wam_r_lowered_mode_decl, ModeDecl), _, fail),
    is_list(ModeDecl),
    nth1(AIdx, ModeDecl, Mode),
    mode_atom_to_state(Mode, State).

mode_atom_to_state(input,  bound).
mode_atom_to_state(output, unbound).

%% mode_comment_header_from_records(+Opts, +Records, -Header) is det.
%  Builds the R comment block for the predicate's mode analysis when
%  Opts contains mode_comments(on). Otherwise Header is the empty
%  string.
mode_comment_header_from_records(Opts, Records, Header) :-
    (   member(mode_comments(on), Opts),
        Records \= []
    ->  format_mode_records(Records, Lines),
        atomic_list_concat(Lines, '', Header)
    ;   Header = ""
    ).

%% gather_pred_mode_records(+PredIndicator, -Records) is det.
%  Pulls the user-asserted clauses for the predicate, runs the
%  binding-state analyser per clause, and returns one
%  `mode_record(Idx, ModeDecl, HeadVars, GoalBindings)` per clause.
%
%    - Idx        : 1-based clause number.
%    - ModeDecl   : list of mode atoms (input/output/any) from
%                   `:- mode/1`, or `none` if undeclared.
%    - HeadVars   : list of `Name-State` pairs, one per head arg
%                   position. Name is `Arg<N>` for the Nth arg
%                   position (1-based); State is the analyser's
%                   binding-state for that arg right before the
%                   first body goal (== entry to the clause body).
%                   Non-variable head args render as `Arg<N>-bound`
%                   (literal terms are always bound).
%    - GoalBindings : the raw goal_binding(Idx, Before, After) list
%                     from the analyser (renderable via
%                     format_mode_records/2).
%
%  IMPORTANT: the analyser is called on plain Prolog variables (NOT
%  numbervars-renamed), because `get_binding_state(_, NonVar, bound)`
%  treats `'$VAR'(N)` as a nonvar and would always return `bound`.
%  Head arg names in HeadVars are position-based for display only.
%
%  Module qualifiers in clause bodies are stripped (plunit wraps
%  asserted bodies in plunit_<test>:user:Goal). Failure is silent --
%  any error in the analyser causes the records to be empty so the
%  caller can degrade gracefully.
gather_pred_mode_records(PI, Records) :-
    ( PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(HeadTpl, Pred, Arity),
    catch(findall(HeadCopy-BodyCopy,
                  ( user:clause(HeadTpl, RawBody),
                    wam_r_target:strip_module_qualifiers(RawBody,
                                                         StrippedBody),
                    copy_term(HeadTpl-StrippedBody,
                              HeadCopy-BodyCopy) ),
                  Clauses), _, Clauses = []),
    (   demand_analysis:read_mode_declaration(Pred, Arity, ModeDecl)
    ->  true
    ;   ModeDecl = none
    ),
    gather_clause_records(Clauses, 1, ModeDecl, Records).

gather_clause_records([], _, _, []).
gather_clause_records([Head-Body | Rest], Idx, ModeDecl,
                      [mode_record(Idx, ModeDecl, HeadVars, GBs) | Tail]) :-
    catch(binding_state_analysis:analyse_clause_bindings(Head, Body, GBs0),
          _, GBs0 = []),
    GBs = GBs0,
    head_var_states(Head, ModeDecl, GBs, HeadVars),
    Idx1 is Idx + 1,
    gather_clause_records(Rest, Idx1, ModeDecl, Tail).

%% head_var_states(+Head, +ModeDecl, +GoalBindings, -HeadVars) is det.
%  Returns one `Arg<N>-State` pair per head arg position, where State
%  is the analyser's binding state for that arg right before the first
%  body goal (the env that an optimizer would consult when deciding
%  whether to specialise a head-match instruction). Non-variable head
%  args (compound, atomic, struct, ...) always render as `bound`
%  because head unification by definition binds them. Variable head
%  args render with the analyser's actual state -- `bound` (mode +),
%  `unbound` (mode -), or `unknown` (no info).
%
%  When the body has no goals (e.g. `true` / a fact clause), the
%  analyser returns an empty GoalBindings list; we fall back to the
%  initial env from `initial_binding_env/3` so head modes still
%  surface in the visibility comment.
head_var_states(Head, ModeDecl, GBs, HeadVars) :-
    Head =.. [_ | Args],
    (   member(goal_binding(1, BeforeEnv, _), GBs)
    ->  Env = BeforeEnv
    ;   binding_state_analysis:initial_binding_env(Head, ModeDecl, Env)
    ),
    head_var_states_(Args, 1, Env, HeadVars).

head_var_states_([], _, _, []).
head_var_states_([Arg | Rest], Pos, Env, [Name-State | Tail]) :-
    format(atom(Name), 'Arg~w', [Pos]),
    head_arg_state(Arg, Env, State),
    Pos1 is Pos + 1,
    head_var_states_(Rest, Pos1, Env, Tail).

head_arg_state(Arg, _Env, bound) :-
    nonvar(Arg), !.
head_arg_state(Arg, Env, State) :-
    var(Arg),
    binding_state_analysis:get_binding_state(Env, Arg, State).

%% format_mode_records(+Records, -Lines) is det.
%  Renders the analyser output as a block of R comment lines.
%  Empty record list yields the empty string (no comment block).
format_mode_records([], [""]) :- !.
format_mode_records(Records, [Header | Body]) :-
    Header = "# Mode analysis (phase 1, visibility-only):\n",
    maplist(format_one_clause, Records, Bodies),
    flatten(Bodies, Body0),
    append(Body0, ["#\n"], Body).

format_one_clause(mode_record(Idx, ModeDecl, HeadVars, _GBs), Lines) :-
    format(string(L1), "#   clause ~w head: ~w   (mode_decl=~w)\n",
           [Idx, HeadVars, ModeDecl]),
    Lines = [L1].

emit_deterministic_function(PredName, FuncName, ClauseLines,
                            ModeHeader, Code) :-
    with_output_to(string(Body), emit_lines(ClauseLines, "  ")),
    % The clause body always ends with `proceed`, which emits
    % `return(TRUE)`. We keep an explicit `invisible(TRUE)` after the
    % body as a defensive fallback in case future codegen produces a
    % proceed-less clause; the wrapper does isTRUE() on the result.
    format(string(Code),
'~w# Lowered: ~w  (deterministic single-clause)
~w <- function(program, state) {
~w  invisible(TRUE)
}', [ModeHeader, PredName, FuncName, Body]).

% Multi-clause: push a CP whose next_pc points at clause 2+, run
% clause 1 inline. If clause 1 succeeds the lowered fn returns TRUE
% (the CP stays in $cps for downstream backtracking). If clause 1
% fails we backtrack via the runtime helper, advance past the
% trust_me, and drop into the run loop so it can drive clause 2+.
% multi_clause_n: emit all clauses inline. Strategy:
%
% Snapshot the entering state once (regs/cp/trail-len/var-counter).
% Try each clause inline in order; on success return TRUE; on failure
% restore the snapshot and try the next clause. The last clause's
% failure returns FALSE.
%
% No CP is pushed onto state$cps. Caller-side backtracking through
% this predicate is provided by the lowered_dispatch tier: dispatch_call
% / "Call" / "Execute" all consult lowered_dispatch first, so when the
% caller's CP is popped + ran, the runtime re-enters this same lowered
% fn rather than the WAM array. That means re-entry behaviour through
% standard backtracking is "retry all clauses from scratch" -- the
% same termination guarantees as the WAM array path, with no extra
% cut-barrier subtleties because we never registered a CP.
%
% The emission is built recursively over the clause list. Each clause
% reuses the same `saved_*` snapshot bindings (set up once at function
% entry), so trail/reg restoration is a constant-size operation
% regardless of clause count.
emit_multi_clause_n_function(PredName, FuncName, AllClauseGroups,
                             ModeHeader, Code) :-
    with_output_to(string(Body), emit_clause_attempts(AllClauseGroups, "  ")),
    format(string(Code),
'~w# Lowered: ~w  (multi-clause; all clauses inline, no array fallback)
~w <- function(program, state) {
  saved_regs_       <- state$regs2
  saved_cp_         <- state$cp
  saved_trail_len_  <- length(state$trail)
  saved_var_count_  <- state$var_counter
~w  return(FALSE)
}', [ModeHeader, PredName, FuncName, Body]).

% Recursive emitter. For each clause:
%   - Wrap the clause body in a closure so the body\'s `return(TRUE)`
%     (emitted by `proceed`) becomes the closure\'s return value.
%   - On TRUE, the outer function returns TRUE.
%   - On FALSE, restore the snapshot and recurse into the next clause.
emit_clause_attempts([], _Ind).
emit_clause_attempts([Clause|RestClauses], Ind) :-
    length(RestClauses, RestLen),                  % 0 if last clause
    ClauseNum is 1 + (1 + RestLen) - (1 + RestLen),
    % ClauseNum is unused below -- present so future error diagnostics
    % can identify which clause failed. Keep the binding to silence
    % a warning if added later.
    ignore(ClauseNum = ClauseNum),
    with_output_to(string(InnerBody), emit_lines(Clause, "    ")),
    format("~wclause_ok_ <- (function() {~n", [Ind]),
    format("~w", [InnerBody]),
    format("~w  invisible(FALSE)~n", [Ind]),
    format("~w})()~n", [Ind]),
    format("~wif (isTRUE(clause_ok_)) return(TRUE)~n", [Ind]),
    (   RestClauses == []
    ->  true                                         % last clause; fall through to FALSE
    ;   % Restore entry state for the next clause attempt.
        format("~wstate$regs2       <- saved_regs_~n", [Ind]),
        format("~wstate$cp          <- saved_cp_~n", [Ind]),
        format("~wWamRuntime$undo_trail_to(state, saved_trail_len_)~n", [Ind]),
        format("~wstate$var_counter <- saved_var_count_~n", [Ind]),
        emit_clause_attempts(RestClauses, Ind)
    ).

emit_multi_clause_function(PredName, FuncName, AltLabel, ClauseLines,
                           ModeHeader, Code) :-
    with_output_to(string(Body), emit_lines(ClauseLines, "    ")),
    % clause1_ok is the value of the closure's last expression, which
    % is the `return(TRUE)` emitted by the proceed line at the end of
    % clause 1. invisible(FALSE) is a defensive fallback for the
    % proceed-less case (caller does isTRUE() on the result).
    format(string(Code),
'~w# Lowered: ~w  (multi-clause; clause 1 inline, fall back to array)
~w <- function(program, state) {
  alt_pc <- program$labels[["~w"]]
  if (is.null(alt_pc)) return(FALSE)
  state$cps <- c(state$cps, list(list(
    next_pc     = as.integer(alt_pc),
    regs        = state$regs2,
    cp          = state$cp,
    trail_len   = length(state$trail),
    var_counter = state$var_counter
  )))
  clause1_ok <- (function() {
~w    invisible(FALSE)
  })()
  if (isTRUE(clause1_ok)) return(TRUE)
  if (!isTRUE(WamRuntime$backtrack(state))) return(FALSE)
  state$pc   <- state$pc + 1L
  state$halt <- FALSE
  isTRUE(WamRuntime$run(program, state))
}', [ModeHeader, PredName, FuncName, AltLabel, Body]).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  true
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

% --- Control instructions ------------------------------------------------

emit_line_parts(["proceed"], I) :- !,
    format("~wreturn(TRUE)~n", [I]).
emit_line_parts(["fail"], I) :- !,
    format("~wreturn(FALSE)~n", [I]).
emit_line_parts(["call", PredArity], I) :- !,
    emit_call(PredArity, I).
emit_line_parts(["call", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_call(PredArity, I).
emit_line_parts(["execute", PredArity], I) :- !,
    emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_execute(PredArity, I).

% --- Native register ops (always succeed; skip the step dispatcher) ------
%
% These are the paydirt of Phase 3's per-instruction inlining: each
% one is a direct mutation on state$regs2 / state$stack / state, no
% heap-build state machine, no deref, no unify. The R generated for
% them is the same code WamRuntime$step would have run, just inline.

% Inline Allocate / Deallocate. Match the runtime\'s "Allocate" /
% "Deallocate" semantics exactly: frame has fields {cp, ys, cps_barrier},
% Deallocate retains the popped frame as `state$shadow_frame` so post-
% Deallocate Y reads still resolve. The earlier emission (locals=...,
% no shadow) only worked because clause-1-only multi_clause_1 lowering
% never actually hit an `allocate` inline -- clause 2 stayed in the
% array and went through the proper runtime handler. Phase 4 puts
% clause 2 inline, so the inline ops have to match the array path.
emit_line_parts(["allocate"], I) :- !,
    format("~w{ ys_ <- new.env(parent = emptyenv(), hash = TRUE); state$stack <- c(state$stack, list(list(cp = state$cp, ys = ys_, cps_barrier = state$pending_call_barrier))); state$shadow_frame <- NULL }~n",
           [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~w{ n_ <- length(state$stack); if (n_ > 0L) { frame_ <- state$stack[[n_]]; state$cp <- frame_$cp; state$shadow_frame <- frame_; state$stack <- state$stack[-n_] } }~n",
           [I]).
emit_line_parts(["put_constant", CStr, RegStr], I) :- !,
    wam_r_target:reg_to_int(RegStr, RegIdx),
    wam_r_target:constant_to_r_term(CStr, CTerm),
    format("~wWamRuntime$put_reg(state, ~w, ~w)~n", [I, RegIdx, CTerm]).
emit_line_parts(["put_variable", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~w{ v_ <- Unbound(paste0(\"V\", state$var_counter)); state$var_counter <- state$var_counter + 1L; WamRuntime$put_reg(state, ~w, v_); WamRuntime$put_reg(state, ~w, v_) }~n",
           [I, XIdx, AIdx]).
emit_line_parts(["put_value", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~wWamRuntime$put_reg(state, ~w, WamRuntime$get_reg(state, ~w))~n",
           [I, AIdx, XIdx]).
emit_line_parts(["get_variable", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~wWamRuntime$put_reg(state, ~w, WamRuntime$get_reg(state, ~w))~n",
           [I, XIdx, AIdx]).

% --- Mode-driven specialisation: inline get_constant when the analyser
% proves the target A-register is bound. The default path delegates to
% WamRuntime$step which handles the unbound branch (binds A_i to the
% constant). When mode info promises A_i is already bound, the unbound
% branch is dead code -- we just need a deref + identical-equality
% check. Saves one function call (~0.5us, the dominant cost) per
% get_constant; significant on head-match-heavy workloads.
%
% Soundness: relies on the analyser's `bound` answer being correct,
% which in turn relies on `:- mode/1` declarations being honest. If a
% caller passes an unbound term where the mode says +, the inline
% form returns FALSE (treating unbound as a tag mismatch) where the
% step path would have bound it. Document; user is responsible.
emit_line_parts(["get_constant", CStr, AiStr], I) :-
    head_state_for_areg(AiStr, bound),
    !,
    wam_r_target:reg_to_int(AiStr, AIdx),
    wam_r_target:constant_to_r_term(CStr, CTerm),
    format("~w{ val_ <- WamRuntime$deref(state, WamRuntime$get_reg(state, ~w)); if (is.null(val_) || !identical(val_, ~w)) return(FALSE) }~n",
           [I, AIdx, CTerm]).

% --- Builtin specialisations: inline the most common BuiltinCall
% targets so they skip the WamRuntime$step -> WamRuntime$call_builtin
% function-call + switch-dispatch hop. The inline path is semantically
% identical to the slow path; it just avoids two function calls and
% two switch lookups per call. Used heavily on arith-heavy workloads
% where is/2 is the dominant builtin.

emit_line_parts(["builtin_call", "is/2", "2"], I) :- !,
    format("~w{~n", [I]),
    format("~w  is_target_ <- WamRuntime$get_reg(state, 1L)~n", [I]),
    format("~w  is_expr_   <- WamRuntime$get_reg(state, 2L)~n", [I]),
    format("~w  is_n_      <- WamRuntime$eval_arith(state, is_expr_, intern_table)~n", [I]),
    format("~w  if (is.null(is_n_)) return(FALSE)~n", [I]),
    format("~w  is_res_    <- WamRuntime$arith_to_term(is_n_)~n", [I]),
    format("~w  if (is.null(is_res_)) return(FALSE)~n", [I]),
    format("~w  is_target_d_ <- WamRuntime$deref(state, is_target_)~n", [I]),
    format("~w  if (!is.null(is_target_d_) && !is.null(is_target_d_$tag) && is_target_d_$tag == \"unbound\") {~n", [I]),
    format("~w    WamRuntime$bind(state, is_target_d_$name, is_res_)~n", [I]),
    format("~w  } else if (!isTRUE(WamRuntime$unify(state, is_target_, is_res_))) return(FALSE)~n", [I]),
    format("~w}~n", [I]).

% --- Default: delegate to step with the same R literal the array uses
emit_line_parts(Parts, I) :-
    wam_r_target:wam_parts_to_r(Parts, [], Lit),
    format("~wif (!isTRUE(WamRuntime$step(program, state, ~w))) return(FALSE)~n",
           [I, Lit]).

% emit_call / emit_execute check program$lowered_dispatch first. When
% the target predicate has a registered lowered fn (kernel, fact-table,
% or phase-4 multi_clause_n), invoke it directly -- bypassing the WAM
% array iteration through `step`. That's where phase-4's recursive
% material win comes from: a clause whose body ends with `execute pn/1`
% can now jump straight back into lowered_pn_1 instead of step-iterating
% pn/1's WAM array.

% emit_call: non-tail call. Invokes the lowered fn (via the
% trampoline helper so any tail-calls inside the callee are drained
% before we return) and continues with the rest of the caller's body.
emit_call(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  saved_cp <- state$cp~n", [I]),
    format("~w  disp_key_ <- \"~w\"~n", [I, PredArity]),
    format("~w  if (!is.null(program$lowered_dispatch) && exists(disp_key_, envir = program$lowered_dispatch, inherits = FALSE)) {~n", [I]),
    format("~w    if (!isTRUE(WamRuntime$invoke_lowered_with_tco(program, state, disp_key_))) return(FALSE)~n", [I]),
    format("~w    state$cp <- saved_cp~n", [I]),
    format("~w  } else {~n", [I]),
    format("~w    tgt <- program$labels[[disp_key_]]~n", [I]),
    format("~w    if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w    state$cp <- 0L~n", [I]),
    format("~w    state$pc <- as.integer(tgt)~n", [I]),
    format("~w    if (!isTRUE(WamRuntime$run(program, state))) return(FALSE)~n", [I]),
    format("~w    state$halt <- FALSE~n", [I]),
    format("~w    state$cp <- saved_cp~n", [I]),
    format("~w  }~n", [I]),
    format("~w}~n", [I]).

% emit_execute: tail call (Prolog TCO). Instead of recursing into the
% target lowered fn directly (which would grow R\'s C stack on every
% self-recursion level), set state$tail_call and return TRUE. The
% closest enclosing trampoline -- the per-pred wrapper, the runtime
% Call/Execute handlers, or an outer emit_call\'s invoke_lowered_with_tco
% loop -- consumes the signal and dispatches the next iteration
% without adding a frame.
emit_execute(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  disp_key_ <- \"~w\"~n", [I, PredArity]),
    format("~w  if (!is.null(program$lowered_dispatch) && exists(disp_key_, envir = program$lowered_dispatch, inherits = FALSE)) {~n", [I]),
    format("~w    state$tail_call <- disp_key_~n", [I]),
    format("~w    return(TRUE)~n", [I]),
    format("~w  }~n", [I]),
    format("~w  tgt <- program$labels[[disp_key_]]~n", [I]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  return(isTRUE(WamRuntime$run(program, state)))~n", [I]),
    format("~w}~n", [I]).

%% strip_arity_local(+TokenWithArity, -Name)
%  Same convention as wam_r_target's strip_arity_suffix: drop trailing
%  "/N" if present so we can rebuild "Name/N" from the explicit arity
%  token that follows in the WAM line.
strip_arity_local(Tok, Name) :-
    (   sub_string(Tok, B, 1, _, "/")
    ->  sub_string(Tok, 0, B, _, Name)
    ;   Name = Tok
    ).

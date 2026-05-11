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

build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", AllLines),
    skip_to_first_real_instr(AllLines, Filtered),
    classify_clause_shape(Filtered, plan(Mode, AltLabel, ClauseLines)).

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

% First-real-instruction discriminates between deterministic and
% multi_clause_1. A try_me_else here means clause 1 starts at the next
% line and ends at the trust_me/proceed boundary.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_1, AltAtom, ClauseLines)) :-
    wam_r_target:tokenize_wam_line(FirstLine, ["try_me_else", AltStr]), !,
    atom_string(AltAtom, AltStr),
    take_clause1_lines(Rest, ClauseLines).
classify_clause_shape(Lines, plan(deterministic, none, Lines)).

% Clause 1 ends either at a `proceed` (success path; we keep the
% proceed line so the emitter renders return(TRUE) at that point) or
% at a `trust_me` (start of clause 2; we stop without including it).
take_clause1_lines([], []).
take_clause1_lines([Line|Rest], Out) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == ["proceed"]
    ->  Out = [Line]
    ;   Parts == ["trust_me"]
    ->  Out = []
    ;   Out = [Line|RestOut],
        take_clause1_lines(Rest, RestOut)
    ).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_r_lowerable(+Pred, +WamCode, -Reason) is semidet.
%  Reason is one of `deterministic` or `multi_clause_1`. Lowerability
%  is decided against the emission-plan's clause lines.
wam_r_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, plan(Mode, _, ClauseLines)), _, fail),
    forall(member(Line, ClauseLines), line_supported(Line)),
    Reason = Mode.

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
%                          no codegen change. Phase 1 of WAM-R mode-
%                          analysis integration (see
%                          docs/design/WAM_R_MODE_ANALYSIS_PLAN.md).
lower_predicate_to_r(PI, WamCode, Opts,
                     lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    r_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)),
    mode_comment_header(PI, Opts, ModeHeader),
    (   Mode == deterministic
    ->  emit_deterministic_function(PredName, FuncName, ClauseLines,
                                    ModeHeader, Code)
    ;   Mode == multi_clause_1
    ->  emit_multi_clause_function(PredName, FuncName, AltLabel,
                                   ClauseLines, ModeHeader, Code)
    ).

%% mode_comment_header(+PI, +Opts, -Header) is det.
%  Builds the R comment block for the predicate's mode analysis when
%  Opts contains mode_comments(on). Otherwise Header is the empty
%  string. The block is intended to be visible in the generated R
%  file so developers can see what the analyser inferred without
%  re-running it.
mode_comment_header(PI, Opts, Header) :-
    (   member(mode_comments(on), Opts),
        catch(gather_pred_mode_records(PI, Records), _, fail),
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
%    - HeadVars   : list of `Var-Name-State` triples for each head
%                   variable, where Name is the numbervars-assigned
%                   atom and State is the binding state immediately
%                   after head unification (i.e. BeforeEnv of goal 1).
%    - GoalBindings : the raw goal_binding(Idx, Before, After) list
%                     from the analyser (renderable via
%                     format_mode_records/2).
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
                              HeadCopy-BodyCopy),
                    numbervars(HeadCopy-BodyCopy, 0, _) ),
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
    head_var_states(Head, GBs, HeadVars),
    Idx1 is Idx + 1,
    gather_clause_records(Rest, Idx1, ModeDecl, Tail).

%% head_var_states(+Head, +GoalBindings, -HeadVars) is det.
%  Returns a list of `Name-State` pairs, one per direct head-arg
%  variable. State is read from goal_binding(1, BeforeEnv, _) --
%  the env right before the first body goal, which is what an
%  optimizer choosing a head-match specialisation would consult.
head_var_states(Head, GBs, HeadVars) :-
    Head =.. [_ | Args],
    (   member(goal_binding(1, BeforeEnv, _), GBs)
    ->  true
    ;   binding_state_analysis:empty_binding_env(BeforeEnv)
    ),
    head_var_states_(Args, BeforeEnv, HeadVars).

head_var_states_([], _, []).
head_var_states_([Arg | Rest], Env, [Name-State | Tail]) :-
    (   var(Arg)
    ->  format(atom(Name), '_var_~w', [Arg])
    ;   Arg = '$VAR'(N)
    ->  numbered_var_atom(N, Name)
    ;   Name = '<nonvar>'
    ),
    (   var(Arg)
    ->  State = unbound
    ;   binding_state_analysis:get_binding_state(Env, Arg, State0)
    ->  State = State0
    ;   State = unknown
    ),
    head_var_states_(Rest, Env, Tail).

%% numbered_var_atom(+N, -Atom) is det.
%  Mirrors the SWI numbervars convention: 0..25 -> A..Z, then A1..Z1,
%  A2..Z2, etc. Used so head-arg variable names in the emitted
%  comments match what numbervars/3 displays elsewhere.
numbered_var_atom(N, Atom) :-
    Letter is N mod 26,
    Suffix is N // 26,
    Code is 0'A + Letter,
    (   Suffix =:= 0
    ->  char_code(Char, Code), atom_concat(Char, '', Atom)
    ;   char_code(Char, Code), atom_concat(Char, Suffix, Atom)
    ).

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

emit_line_parts(["allocate"], I) :- !,
    format("~wstate$stack <- c(state$stack, list(list(cp = state$cp, locals = new.env(parent = emptyenv()))))~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~w{ n_ <- length(state$stack); if (n_ > 0L) { state$cp <- state$stack[[n_]]$cp; state$stack <- state$stack[-n_] } }~n", [I]).
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

% --- Default: delegate to step with the same R literal the array uses
emit_line_parts(Parts, I) :-
    wam_r_target:wam_parts_to_r(Parts, [], Lit),
    format("~wif (!isTRUE(WamRuntime$step(program, state, ~w))) return(FALSE)~n",
           [I, Lit]).

emit_call(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  saved_cp <- state$cp~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$cp <- 0L~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  if (!isTRUE(WamRuntime$run(program, state))) return(FALSE)~n", [I]),
    format("~w  state$halt <- FALSE~n", [I]),
    format("~w  state$cp <- saved_cp~n", [I]),
    format("~w}~n", [I]).

emit_execute(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
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

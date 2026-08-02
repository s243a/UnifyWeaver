:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_where.pl - PROTOTYPE fourth consumer for pattern_stache:
% the where-clause form over pe_emit.
%
% Implements, without extending, the ruled and recorded design:
%
%   DESIGN_desugaring_to_prolog_goals.md §12 — "where C=simplemind" is
%   a BINDING GOAL.  It is ground at elaboration time, so it is
%   discharged: a side condition that constructed the term, never part
%   of it.  The ground term is byte-identical to the repeated-literal
%   spelling (re-measured here against the sealed bundle, in Prolog).
%   Bindings are identity-DETERMINING, so they are not pins and never
%   reach a pin channel.
%
%   DESIGN_registry_v0.4.md §10.1 — the repeated literal IS the
%   deliberate v0.4 ground form; variables stay OUT of the sealed
%   grammar.  The where-form is therefore an INPUT convention of this
%   driver — like pe_emit's goal convention — never a change to the
%   v0.4 surface, the registry, or the sealed bundles.
%
% ============================================================
% WHERE CONVENTION (the one concrete convention, documented)
% ============================================================
% A where-term is:
%
%     where(Expr, Bindings)
%
% where Expr is a goal term under pe_emit's goal convention except
% that Prolog VARIABLES may stand in value positions, and Bindings is
% a list of Var = Value with Var a Prolog variable occurring in Expr.
% Because the variables are real Prolog variables, two occurrences of
% C are ONE variable, so one binding reaches both BY CONSTRUCTION —
% the functional connection §12 describes is enforced by the term
% representation itself, not by a check.
%
% Example (the task's own):
%
%     where(product(hop_decay(C, gamma(0.6)), lca_frac(C)),
%           [C = simplemind])
%
% elaborates to product(hop_decay(simplemind, gamma(0.6)),
% lca_frac(simplemind)) and is handed to the EXISTING pe_emit path
% unchanged.
%
% ============================================================
% ELABORATION SEMANTICS (all fail closed)
% ============================================================
%   - the caller's term is never mutated (copy_term first; internal
%     variable sharing is preserved by copy_term);
%   - a binding that is not `Var = Value` is an error (bad_binding);
%   - two bindings for one variable are an error (duplicate_binding),
%     even if the values agree — a duplicate is a spelling accident;
%   - a binding whose variable does not occur in Expr is an error
%     (dead_binding) — dead bindings hide typos;
%   - a binding variable sitting in a PIN position is an error
%     (binding_reaches_pin_channel): §12 rules the channels disjoint —
%     pins are identity-transparent, bindings identity-determining —
%     so the binding machinery refuses to touch provenance positions;
%     likewise a binding VALUE containing a pin/2 is refused
%     (a pin smuggled through the binding channel);
%   - a binding value must be structurally legal for EVERY position
%     its variable reaches; the first illegal position is named in the
%     error (illegal_binding_value(Value, at(Position))).  Legality is
%     STRUCTURAL (what pe_emit can render in that position), not
%     output-type checking — μ-typing remains vNext's job, and nothing
%     here claims otherwise;
%   - after all bindings apply, the goal must be GROUND; a leftover
%     variable is an error (unbound_after_elaboration).  This
%     prototype targets ground emission; residuation is the
%     elaborator's future, not this driver's.
%
% pe_emit.pl is used UNCHANGED (its non-exported registry mirror is
% consulted by module-qualified calls, which modifies nothing).  The
% vNext Python machinery is neither imported, wrapped, nor ported —
% the oracle is the sealed bundle bytes.

:- module(pe_where, [
    elaborate_where/2,        % +WhereTerm, -GroundGoal
    where_semantic/2,         % +WhereTerm, -CanonicalSemanticText
    where_full/2              % +WhereTerm, -CanonicalFullText
]).

:- use_module(pe_emit, [pe_semantic/2, pe_full/2]).
:- use_module(library(lists)).
:- use_module(library(apply)).

%% where_semantic(+WhereTerm, -Text)
where_semantic(W, Text) :-
    elaborate_where(W, Ground),
    pe_semantic(Ground, Text).

%% where_full(+WhereTerm, -Text)
where_full(W, Text) :-
    elaborate_where(W, Ground),
    pe_full(Ground, Text).

%% elaborate_where(+WhereTerm, -GroundGoal)
%  Validate, substitute, and discharge.  The output is an ordinary
%  ground goal for pe_emit; the where-form has vanished.
elaborate_where(W0, Ground) :-
    (   W0 = where(_, _)
    ->  true
    ;   throw(error(pe_where(not_a_where_term(W0)), _))
    ),
    copy_term(W0, where(Expr, Bindings)),      % caller's term untouched
    check_binding_shapes(Bindings),
    check_no_duplicates(Bindings),
    term_variables(Expr, ExprVars),
    check_no_dead_bindings(Bindings, ExprVars),
    check_positions(Expr, Bindings),
    apply_bindings(Bindings),
    (   ground(Expr)
    ->  Ground = Expr
    ;   throw(error(pe_where(unbound_after_elaboration(Expr)), _))
    ).

%% ============================================
%% BINDING LIST VALIDATION
%% ============================================

check_binding_shapes(Bindings) :-
    (   is_list(Bindings)
    ->  true
    ;   throw(error(pe_where(bad_binding_list(Bindings)), _))
    ),
    forall(member(B, Bindings), check_binding_shape(B)).

check_binding_shape(B) :-
    (   nonvar(B),
        B = (V = _Val),
        var(V)
    ->  true
    ;   throw(error(pe_where(bad_binding(B)), _))
    ).

check_no_duplicates([]).
check_no_duplicates([V = _|Rest]) :-
    (   member(W = _, Rest),
        W == V
    ->  throw(error(pe_where(duplicate_binding_for_one_variable), _))
    ;   check_no_duplicates(Rest)
    ).

check_no_dead_bindings(Bindings, ExprVars) :-
    forall(member(V = Val, Bindings),
           (   var_occurs_in(V, ExprVars)
           ->  true
           ;   throw(error(pe_where(dead_binding(V = Val)), _))
           )).

var_occurs_in(V, Vars) :-
    member(X, Vars),
    X == V,
    !.

%% ============================================
%% POSITION LEGALITY
%% ============================================
%
% For each binding, every position its variable reaches is collected
% with a structural context, and the bound value is checked against
% each context BEFORE any substitution happens.  The first failing
% position is named in the error.

check_positions(Expr, Bindings) :-
    occurrences(Expr, root, Occs),
    forall(member(V = Val, Bindings),
           forall(( member(occ(X, Ctx), Occs), X == V ),
                  check_value_at(Ctx, Val))).

%% occurrences(+Term, +Ctx, -Occs)
%  Occs is a list of occ(Var, Context) for every variable occurrence
%  in Term.  Context names the position structurally:
%    root, arg(Op, I), kwarg(Op, K), list_elem(Op, K),
%    mod_base, mod_name, pin_expr, pin_name.
occurrences(T, Ctx, [occ(T, Ctx)]) :-
    var(T),
    !.
occurrences(T, _, []) :-
    atomic(T),
    !.
occurrences(mod(B, M), _, Occs) :-
    !,
    occurrences(B, mod_base, O1),
    occurrences(M, mod_name, O2),
    append(O1, O2, Occs).
occurrences(pin(E, P), _, Occs) :-
    !,
    occurrences(E, pin_expr, O1),
    occurrences(P, pin_name, O2),
    append(O1, O2, Occs).
occurrences(Goal, _, Occs) :-
    compound(Goal),
    compound_name_arguments(Goal, Name, RawArgs),
    pe_emit:pe_operator(Name),
    !,
    op_arg_occurrences(RawArgs, Name, 1, Occs).
occurrences(Goal, Ctx, Occs) :-
    % unregistered compound: recurse generically so the variable is
    % still found; pe_emit fails closed on the form itself later
    compound(Goal),
    compound_name_arguments(Goal, _, Args),
    maplist([A, O]>>occurrences(A, Ctx, O), Args, Nested),
    append(Nested, Occs).

op_arg_occurrences([], _, _, []).
op_arg_occurrences([A|Rest], Op, I, Occs) :-
    (   nonvar(A),
        compound(A),
        compound_name_arguments(A, K, [V0]),
        pe_emit:pe_kwspec(Op, K, _, _)
    ->  kw_value_occurrences(V0, Op, K, O1),
        I1 = I                                  % kwargs don't consume an index
    ;   occurrences(A, arg(Op, I), O1),
        I1 is I + 1
    ),
    op_arg_occurrences(Rest, Op, I1, O2),
    append(O1, O2, Occs).

kw_value_occurrences(V0, Op, K, Occs) :-
    (   var(V0)
    ->  Occs = [occ(V0, kwarg(Op, K))]
    ;   is_list(V0)
    ->  list_var_occurrences(V0, Op, K, Occs)
    ;   occurrences(V0, kwarg(Op, K), Occs)
    ).

% findall/3 would COPY the variables out of the occ/2 terms, breaking
% the ==-identity the position check depends on; collect by walking.
list_var_occurrences([], _, _, []).
list_var_occurrences([X|Xs], Op, K, Occs) :-
    (   var(X)
    ->  Occs = [occ(X, list_elem(Op, K))|Rest]
    ;   Occs = Rest
    ),
    list_var_occurrences(Xs, Op, K, Rest).

%% check_value_at(+Ctx, +Val)
%  Structural legality of Val at Ctx; throws with the position named.
check_value_at(pin_name, _Val) :-
    !,
    % §12: bindings never reach a pin channel.  Refused regardless of
    % the value — the channels are disjoint by ruling.
    throw(error(pe_where(binding_reaches_pin_channel(pin_name)), _)).
check_value_at(Ctx, Val) :-
    (   legal_at(Ctx, Val)
    ->  true
    ;   throw(error(pe_where(illegal_binding_value(Val, at(Ctx))), _))
    ).

legal_at(mod_name, Val) :-
    atom(Val).
legal_at(mod_base, Val) :-
    atom(Val),
    pe_emit:pe_atom(Val).
legal_at(root, Val)       :- legal_expr(Val).
legal_at(pin_expr, Val)   :- legal_expr(Val).
legal_at(arg(_, _), Val)  :- legal_expr(Val).
legal_at(kwarg(Op, K), Val) :-
    once(pe_emit:pe_kwspec(Op, K, Kind, _)),
    legal_kind(Kind, Val).
legal_at(list_elem(Op, K), Val) :-
    once(pe_emit:pe_kwspec(Op, K, Kind, _)),
    (   Kind == number_list -> number(Val)
    ;   Kind == int_list    -> integer(Val)
    ).

legal_kind(number, Val)      :- number(Val).
legal_kind(int, Val)         :- integer(Val).
legal_kind(string, Val)      :- ( string(Val) ; atom(Val) ), !.
legal_kind(number_list, Val) :- is_list(Val), maplist(number, Val).
legal_kind(int_list, Val)    :- is_list(Val), maplist(integer, Val).
legal_kind(expr, Val)        :- legal_expr(Val).

%% legal_expr(+Val)
%  Structurally renderable expression values.  pin/2 is refused here
%  too: a pin arriving THROUGH the binding channel would smuggle
%  provenance into an identity-determining substitution (§12).
legal_expr(V) :- number(V), !.
legal_expr(V) :- atom(V), !, pe_emit:pe_atom(V).
legal_expr(mod(B, M)) :- !, atom(M), legal_expr(B).
legal_expr(pin(_, _)) :- !, fail.
legal_expr(V) :-
    compound(V),
    compound_name_arguments(V, Name, _),
    pe_emit:pe_operator(Name).

%% ============================================
%% SUBSTITUTION
%% ============================================
%
% Plain unification: the variables are real Prolog variables, so one
% binding reaches every occurrence at once.  All validation has
% already passed, so this cannot fail (a duplicate with conflicting
% values was refused above, not left for unification to fail on).

apply_bindings([]).
apply_bindings([V = Val|Rest]) :-
    V = Val,
    apply_bindings(Rest).

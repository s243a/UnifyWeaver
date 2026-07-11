:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_bootstrap_compiler.pl - the self-hosted Prolog->.wamo bootstrap compiler
%
% A minimal Prolog->.wamo compiler written entirely in the LOADABLE WAM
% subset, so that -- compiled ahead of time with write_wam_object/3 -- it
% runs as a loaded object and compiles grammars from source text at
% runtime (the eval/compile surface of the plawk JIT roadmap, item 5).
% This is the compiler that closed the self-host fixpoint: compiled to
% gen1, it compiles its own source to gen2; gen2 compiles the same
% source to a byte-identical gen3 (F(F) = F); and the self-compiled
% compiler compiles fresh programs byte-identically to cgfull_term/2.
% See docs/design/PLAWK_SELFHOST.md and docs/design/PLAWK_EVAL_BOOTSTRAP.md.
%
% The file keeps the full STAGED history of the campaign -- each stage is
% still compiled to an object and exercised by tests/test_wam_object.pl
% (where this code lived until milestone 6 completed):
%   Stage A  wamoserz/2   -- .wamo serializer (difference-list emitters)
%   Stage B  cgcompile/2  -- one clause shape, source text in
%   Stage C  cgcprog/2    -- multi-clause programs + predicate calls
%            cgconj/2     -- conjunction + register allocation
%            cgarith/2    -- runtime arithmetic + functor table
%   Stage D  cgfull/2     -- the UNIFIED compiler: multi-clause predicates,
%                            try/retry/trust chains, lists, structures with
%                            X-temp deferral both directions, comparison
%                            guards, if-then-else, ~37 whitelisted builtins
% cgfull_term/2 is cgfull/2 minus the reader call (read_term_from_atom/2
% exists only in the loaded runtime), so SWI can compute golden bytes from
% clause TERMS -- the production-side oracle every self-compile is checked
% against.
%
% Everything here must stay inside the loadable subset (no cuts in the
% cgfull chain beyond first-argument commitment, ITE dispatch, =.. instead
% of control-functor pattern literals, '$VAR'(integer) as the numbervars
% marker) -- the compiler is its own test case.

:- module(wam_bootstrap_compiler, [
    wamoserz/2,        % +Src, -Wamo  (Stage A: fixed program, real serializer)
    cgcompile/2,       % +Src, -Wamo  (Stage B: one clause shape)
    cgcprog/2,         % +Src, -Wamo  (Stage C: multi-clause + calls)
    cgconj/2,          % +Src, -Wamo  (Stage C: conjunction + registers)
    cgarith/2,         % +Src, -Wamo  (Stage C: arithmetic + functor table)
    cgfull/2,          % +Src, -Wamo  (Stage D: the unified compiler)
    cgfull_term/2,     % +Clauses, -Wamo (cgfull minus the reader; SWI oracle)
    cgfullm/2,         % +Src, -Wamo  (cgfull + MULTI-ENTRY name table)
    cgfullm_term/2,    % +Clauses, -Wamo (cgfullm minus the reader)
    wza_serialize/8,   % low-level serializer (golden-byte construction)
    wzam_serialize/7   % multi-entry serializer (cgfullm back end)
]).

:- discontiguous walk_term/3.

% Milestone 6 (self-host) Stage A: a .wamo serializer written in the loadable
% subset. wamoserz/2 is a real eval-pipeline compiler entry -- compile(Src,
% Wamo) -- but Stage A ignores Src and serializes a FIXED ground instruction
% list for a 42-returning program (`ea(R):-R=42`), exercising the string-
% assembly back end end to end. It reproduces wamo_serialize/8's byte format
% using only loadable builtins (atom_codes/number_codes/append/length), so it
% runs as a loaded object; the emitted bytes load via @wam_object_eval and run
% to 42. NA/NF are 0 here (no atom/functor tables); the PC and instruction
% lists drive the rest, so this is a genuine serializer, not a string literal.
%
% Written as SMALL accumulator-threaded clauses (a code list is threaded
% through, each clause holding only a few call-spanning variables) with only
% list / enc(...) head dispatch -- deliberately avoiding two loaded-runtime
% limitations this stage surfaced: (1) a clause holding ~16+ call-spanning
% variables emits register indices past the 64-slot register file and corrupts
% memory; (2) multi-way first-argument functor dispatch across >=4 tagged
% variants (with a list-carrying variant present) mis-dispatches in loaded
% objects. Both are recorded in PLAWK_SELFHOST.md as prerequisites for later
% stages; the "small clauses, correctness over register reuse" style the design
% doc prescribes routes around both.
wamoserz(_Src, Wamo) :-
    atom_codes('ea/1', NameCodes),
    wz_serialize(0, NameCodes, 0, 0, 0, [0],
        [enc(0,42,65536,0), enc(20,0,0,0)], Codes),
    atom_codes(Wamo, Codes).

% DIFFERENCE-LIST emitters: A0 is the open list being built, A1 its tail
% after this item -- so each emission appends only its OWN codes (linear in
% total output), never copying the accumulated prefix. The old accumulator
% style (append(A0, Cs, B) -- copy everything emitted so far, per token) is
% QUADRATIC in output size and was the entire compile-time/memory cliff of
% the self-host fixpoint (246 MB arena for a 3.6 KB object). Threading
% predicates (wz_header/wz_body/wz_pcs_rows/...) are direction-agnostic and
% unchanged; only the leaf emitters and the top wrappers (which now close
% the tail with []) know the representation.
% wz_i "<int>\n", wz_a "<atom>\n", wz_n "<len> <bytes>\n", wz_si " <int>".
wz_i(N, A0, A1)  :- number_codes(N, Cs), append(Cs, [10|A1], A0).
wz_a(X, A0, A1)  :- atom_codes(X, Cs), append(Cs, [10|A1], A0).
wz_n(Cs, A0, A1) :- length(Cs, Len), number_codes(Len, LC),
    append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid).
wz_si(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0).

wz_serialize(EntryIdx, NameCodes, LabelIdx, NA, NF, PCs, Instrs, Out) :-
    wz_header(EntryIdx, NameCodes, LabelIdx, NA, NF, Out, Hdr),
    wz_body(PCs, Instrs, Hdr, []).

% header: WAMO / version 2 / entry index / NE=1 / name / label / NA / NF,
% split across two clauses so neither holds too many call-spanning vars.
wz_header(EntryIdx, NameCodes, LabelIdx, NA, NF, A0, Out) :-
    wz_a('WAMO', A0, A1), wz_i(2, A1, A2), wz_i(EntryIdx, A2, A3), wz_i(1, A3, A4),
    wz_header2(NameCodes, LabelIdx, NA, NF, A4, Out).
wz_header2(NameCodes, LabelIdx, NA, NF, A0, Out) :-
    wz_n(NameCodes, A0, A1), wz_i(LabelIdx, A1, A2), wz_i(NA, A2, A3), wz_i(NF, A3, Out).

% body: PC section, instruction section, then NM=0.
wz_body(PCs, Instrs, A0, Out) :-
    wz_pcs_sec(PCs, A0, A1),
    wz_instr_sec(Instrs, A1, A2),
    wz_i(0, A2, Out).

wz_pcs_sec(PCs, A0, A2) :- length(PCs, NL), wz_i(NL, A0, A1), wz_pcs_rows(PCs, A1, A2).
wz_pcs_rows([], A, A).
wz_pcs_rows([P|Ps], A0, A2) :- wz_i(P, A0, A1), wz_pcs_rows(Ps, A1, A2).

wz_instr_sec(Instrs, A0, A2) :-
    length(Instrs, NC), wz_i(NC, A0, A1), wz_instr_rows(Instrs, A1, A2).
wz_instr_rows([], A, A).
wz_instr_rows([enc(T,O1,O2,R)|Is], A0, A2) :-
    wz_row(T, O1, O2, R, A0, A1), wz_instr_rows(Is, A1, A2).
wz_row(T, O1, O2, R, A0, A5) :-
    number_codes(T, Tc), append(Tc, A1, A0),
    wz_si(O1, A1, A2), wz_si(O2, A2, A3), wz_si(R, A3, A4), A4 = [10|A5].

% Milestone 6 (self-host) Stage B: minimal CODEGEN -- source text to a .wamo,
% end to end. cgcompile/2 is the eval-pipeline entry compile(Src,Wamo): it
% parses Src with the runtime reader (read_term_from_atom/2, milestone 3b),
% walks the clause to an instruction list (clause_to_instrs/3), and hands it to
% the Stage A serializer (wz_serialize/8). The loadable subset for now: a
% one-argument clause whose body binds the head variable to an integer, either
% directly (`p(R) :- R = 42`) or by evaluating a ground arithmetic expression
% (`p(R) :- R is 6*7`). Both lower to [get_constant(V,A1), proceed] -- the
% golden shape from Stage A, parameterized by the computed value V.
%
% body_int/2 dispatches on the body's functor (=/2 vs is/2); this is exactly
% the tagged first-argument dispatch that the get_structure functor-check fix
% made correct in loaded objects. No constant-index arg/3 (the known subset
% gap) -- functor/3 gives the predicate name and body_int gives the value.
cgcompile(Src, Wamo) :-
    read_term_from_atom(Src, Clause),
    clause_to_instrs(Clause, Instrs, NameCodes),
    wz_serialize(0, NameCodes, 0, 0, 0, [0], Instrs, Codes),
    atom_codes(Wamo, Codes).

clause_to_instrs((Head :- Body), Instrs, NameCodes) :-
    body_int(Body, V),
    functor(Head, Pred, Arity),
    pred_name_codes(Pred, Arity, NameCodes),
    % get_constant(V, A1): tag 0, op1 = V, op2 = (Integer-tag 1 << 16)|reg A1(0).
    Instrs = [enc(0, V, 65536, 0), enc(20, 0, 0, 0)].

body_int((_ = V), V)     :- integer(V).
body_int((_ is Expr), V) :- V is Expr.

pred_name_codes(Pred, Arity, Codes) :-
    atom_codes(Pred, PC), number_codes(Arity, AC),
    append(PC, [0'/ | AC], Codes).

% Milestone 6 (self-host) Stage C: multi-clause programs with predicate calls.
% cgcprog/2 is the eval entry compile(Src,Wamo): Src parses (one reader call)
% to a LIST of clauses -- e.g. "[(main0(R):-helper(R)), helper(42)]" -- and the
% program compiles to a multi-predicate .wamo. Each clause gets a label (its
% index); a predicate-call body becomes execute(CalleeLabel) resolved through
% a name/arity->label map (execute references the label directly, so no
% meta-call table is needed). The label->PC table (PCs, one entry per clause)
% is exactly what the Stage A serializer already accepts. Facts P(Int) and the
% Stage B body forms (= / is) still lower to [get_constant(V,A1), proceed].
% A single-goal predicate-call body is a tail call: [allocate, deallocate,
% execute] -- matching the host writer (verified byte-identical). Register
% allocation for temporaries across conjoined goals is a Stage C follow-on.
cgcprog(Src, Wamo) :-
    read_term_from_atom(Src, Clauses),
    cgc_clauses(Clauses, EntryName, PCs, Instrs),
    wz_serialize(0, EntryName, 0, 0, 0, PCs, Instrs, Codes),
    atom_codes(Wamo, Codes).

cgc_clauses(Clauses, EntryName, PCs, AllInstrs) :-
    assign_labels(Clauses, 0, PL),
    codegen_all(Clauses, PL, 0, AllInstrs, PCs),
    Clauses = [First|_], cgc_head(First, H0), functor(H0, P0, A0),
    pred_name_codes(P0, A0, EntryName).

cgc_head((H :- _), H) :- !.
cgc_head(H, H).

assign_labels([], _, []).
assign_labels([C|Cs], L, [key(P,A)-L | Rest]) :-
    cgc_head(C, H), functor(H, P, A), L1 is L + 1,
    assign_labels(Cs, L1, Rest).

codegen_all([], _, _, [], []).
codegen_all([C|Cs], PL, PC, All, [PC|PCs]) :-
    codegen_clause(C, PL, Instrs),
    length(Instrs, N), PC1 is PC + N,
    codegen_all(Cs, PL, PC1, Rest, PCs),
    append(Instrs, Rest, All).

codegen_clause((_ :- Body), PL, Instrs) :- !, codegen_body(Body, PL, Instrs).
codegen_clause(Fact, _, [enc(0,V,65536,0), enc(20,0,0,0)]) :-   % fact P(Int)
    Fact =.. [_, V], integer(V).

codegen_body(Body, PL, Instrs) :-
    (   Body = (_ = V), integer(V)
    ->  Instrs = [enc(0,V,65536,0), enc(20,0,0,0)]
    ;   Body = (_ is Expr)
    ->  V is Expr, Instrs = [enc(0,V,65536,0), enc(20,0,0,0)]
    ;   functor(Body, P, A), lookup_label(P, A, PL, Label)
    ->  Instrs = [enc(16,0,0,0), enc(17,0,0,0), enc(19,Label,0,0)]  % allocate,deallocate,execute
    ).

lookup_label(P, A, [key(P,A)-L|_], L) :- !.
lookup_label(P, A, [_|R], L) :- lookup_label(P, A, R, L).

% Milestone 6 (self-host) Stage C (rest): conjunction + register allocation.
% cgconj/2 compiles a clause whose body is a conjunction of unification goals
% (X = Y, X = Int) with a real (simple) register allocator: numbervars binds
% each source variable to '$VAR'(N), mapped to permanent register Y(N+1); the
% initialized-Y set is threaded through so a variable's first occurrence emits
% put_variable / get_variable and later occurrences put_value. Head arguments
% are saved with get_variable; each `=` goal sets up A1/A2 (put_variable /
% put_value / put_constant) then builtin_call =/2 (id 24). This is the WAM
% register-allocation core -- verified byte-identical to the host writer for
% `pconj(R):-Y=42,R=Y`. Runtime arithmetic (is/2 building expression terms via
% put_structure) and non-tail calls are the remaining Stage C pieces.
cgconj(Src, Wamo) :-
    read_term_from_atom(Src, Clause),
    numbervars(Clause, 0, _),
    conj_instrs(Clause, NameCodes, Instrs),
    wz_serialize(0, NameCodes, 0, 0, 0, [0], Instrs, Codes),
    atom_codes(Wamo, Codes).

conj_instrs(Clause, NameCodes, Instrs) :-
    clause_hb(Clause, Head, Goals),
    functor(Head, Pred, Arity), pred_name_codes(Pred, Arity, NameCodes),
    head_args(Head, 1, Arity, [], Init1, HeadIs),
    goals_instrs(Goals, Init1, _, GoalIs),
    append([enc(16,0,0,0) | HeadIs], GoalIs, B0),          % allocate + head
    append(B0, [enc(17,0,0,0), enc(20,0,0,0)], Instrs).     % deallocate, proceed

clause_hb((H :- B), H, Goals) :- !, conj_list(B, Goals).
clause_hb(H, H, []).
conj_list((A,B), [A|Rest]) :- !, conj_list(B, Rest).
conj_list(G, [G]).

is_init(N, [N|_]) :- !.
is_init(N, [_|T]) :- is_init(N, T).

head_args(_, I, Arity, Init, Init, []) :- I > Arity, !.
head_args(Head, I, Arity, Init0, Init, [Instr|Rest]) :-
    arg(I, Head, Arg), AiIdx is I - 1,
    head_arg_instr(Arg, AiIdx, Init0, Init1, Instr),
    I1 is I + 1, head_args(Head, I1, Arity, Init1, Init, Rest).
head_arg_instr('$VAR'(N), AiIdx, Init0, [N|Init0], enc(1, YIdx, AiIdx, 0)) :- YIdx is 48 + N.
head_arg_instr(V, AiIdx, Init, Init, enc(0, V, Op2, 0)) :- integer(V), Op2 is (1 << 16) \/ AiIdx.

operand_instr('$VAR'(N), AiIdx, Init0, Init1, [Instr]) :-
    YIdx is 48 + N,
    (   is_init(N, Init0)
    ->  Instr = enc(10, YIdx, AiIdx, 0), Init1 = Init0     % put_value (subsequent)
    ;   Instr = enc(9, YIdx, AiIdx, 0), Init1 = [N|Init0]  % put_variable (first)
    ).
operand_instr(V, AiIdx, Init, Init, [enc(8, V, Op2, 0)]) :- integer(V), Op2 is (1 << 16) \/ AiIdx.

goal_instrs((L = R), Init0, Init2, Is) :-
    operand_instr(L, 0, Init0, Init1, ILs),
    operand_instr(R, 1, Init1, Init2, IRs),
    append(ILs, IRs, LR), append(LR, [enc(21, 24, 2, 0)], Is).   % builtin_call =/2

goals_instrs([], Init, Init, []).
goals_instrs([G|Gs], Init0, Init, Is) :-
    goal_instrs(G, Init0, Init1, GIs),
    goals_instrs(Gs, Init1, Init, RestIs), append(GIs, RestIs, Is).

% Milestone 6 (self-host) Stage C (arithmetic): runtime is/2. cgarith/2 extends
% the conjunction compiler with `Var is BinExpr` goals -- e.g. `X is 6*7`. A
% binary expression op(A,B) builds a compound term on the heap: put_structure
% op/2 into A2 (op1 = functor-table index, reloc 2), then set_value (a variable
% arg) / set_constant (an integer arg) for each operand, then builtin_call is/2
% (id 0). Functor names used across the clause are collected into the object's
% functor table (NF>0, emitted by the functor-aware serializer wzf_serialize).
% Reuses the cgconj register allocator (operand_instr / head_args / is_init).
% Verified byte-identical to the host writer for `ca(R) :- X is 6*7, R = X`,
% which combines conjunction, a shared temporary, arithmetic, and unification.
cgarith(Src, Wamo) :-
    read_term_from_atom(Src, Clause),
    numbervars(Clause, 0, _),
    a_conj_instrs(Clause, NameCodes, Functors, Instrs),
    wzf_serialize(0, NameCodes, 0, Functors, [0], Instrs, Codes),
    atom_codes(Wamo, Codes).

a_conj_instrs(Clause, NameCodes, Functors, Instrs) :-
    clause_hb(Clause, Head, Goals),
    functor(Head, Pred, Arity), pred_name_codes(Pred, Arity, NameCodes),
    collect_functors(Goals, Functors),
    head_args(Head, 1, Arity, [], Init1, HeadIs),
    a_goals_instrs(Goals, Functors, Init1, _, GoalIs),
    append([enc(16,0,0,0) | HeadIs], GoalIs, B0),
    append(B0, [enc(17,0,0,0), enc(20,0,0,0)], Instrs).

a_goals_instrs([], _, Init, Init, []).
a_goals_instrs([G|Gs], FT, Init0, Init, Is) :-
    a_goal_instrs(G, FT, Init0, Init1, GIs),
    a_goals_instrs(Gs, FT, Init1, Init, RestIs), append(GIs, RestIs, Is).

a_goal_instrs((L is Expr), FT, Init0, Init1, Is) :- !,
    operand_instr(L, 0, Init0, Init1, LhsIs),   % LHS var -> A1
    expr_build(Expr, FT, ExprIs),               % put_structure + set args -> A2
    append(LhsIs, ExprIs, LE), append(LE, [enc(21, 0, 2, 0)], Is).   % builtin is/2
a_goal_instrs((L = R), _FT, Init0, Init2, Is) :-
    operand_instr(L, 0, Init0, Init1, IL), operand_instr(R, 1, Init1, Init2, IR),
    append(IL, IR, LR), append(LR, [enc(21, 24, 2, 0)], Is).         % builtin =/2

expr_build(Expr, FT, [PutS, SA, SB]) :-
    Expr =.. [Op, A, B], functor_index(Op, FT, FIdx),
    Op2 is (2 << 16) \/ 1,                       % arity 2, reg A2(1)
    PutS = enc(11, FIdx, Op2, 2),                % put_structure, reloc functor
    arg_set(A, SA), arg_set(B, SB).
arg_set('$VAR'(N), enc(14, YIdx, 0, 0)) :- YIdx is 48 + N.   % set_value Y
arg_set(V, enc(15, V, 1, 0)) :- integer(V).                   % set_constant integer
functor_index(Op, [Op|_], 0) :- !.
functor_index(Op, [_|T], I) :- functor_index(Op, T, I0), I is I0 + 1.

collect_functors(Goals, FT) :- collect_f(Goals, [], FT).
collect_f([], Acc, Acc).
collect_f([G|Gs], Acc0, FT) :-
    ( G = (_ is E), functor(E, Op, 2) -> add_unique(Op, Acc0, Acc1) ; Acc1 = Acc0 ),
    collect_f(Gs, Acc1, FT).
% memberchk/3-append table dedup. memberchk is a whitelisted NATIVE
% builtin in the compiled object (the hand-rolled memberchk_op it
% replaced ran interpreted -- a WAM call with choice-point machinery
% per element), and the table-collection walk calls this once per atom
% and functor occurrence, so the interpreted scan made table building
% quadratic-with-a-heavy-constant on atom-rich sources: a 20 KB
% synthetic fact-table grammar compiled in 353 ms; with the native
% scan it is ~4x faster and near-linear until far larger tables.
% Table ORDER (first occurrence) is unchanged, so emitted objects are
% byte-identical.
add_unique(Op, Acc, Acc) :- memberchk(Op, Acc), !.
add_unique(Op, Acc, Acc1) :- append(Acc, [Op], Acc1).

% functor-aware serializer: like wz_serialize but emits the functor table
% (NF + NF length-prefixed functor name strings) after the (empty) atom table.
wzf_serialize(EI, NC, LI, Functors, PCs, Instrs, Out) :-
    wz_a('WAMO', Out, A1), wz_i(2, A1, A2), wz_i(EI, A2, A3), wz_i(1, A3, A4),
    wz_n(NC, A4, A5), wz_i(LI, A5, A6), wz_i(0, A6, A7),          % NA = 0
    length(Functors, NF), wz_i(NF, A7, A8), wz_funcs(Functors, A8, A9),
    wz_pcs_sec(PCs, A9, A10), wz_instr_sec(Instrs, A10, A11), wz_i(0, A11, []).
wz_fname(F, A0, A1) :- atom_codes(F, Cs), wz_n(Cs, A0, A1).
wz_funcs([], A, A).
wz_funcs([F|Fs], A0, A2) :- wz_fname(F, A0, A1), wz_funcs(Fs, A1, A2).

% Milestone 6 (self-host) Stage C (non-tail calls) + Stage D (multi-clause):
% the UNIFIED compiler. cgfull/2 merges every piece: labels + PC table
% (from cgcprog), per-clause register allocation + conjunction (cgconj),
% runtime arithmetic + functor table (cgarith), non-tail predicate-call
% goals -- and now MULTIPLE CLAUSES PER PREDICATE. Consecutive clauses with
% the same name/arity group into one predicate (group_clauses); the predicate
% gets the entry label (its group index), and clauses 2..k get alternative
% labels laid out after all entry labels. A multi-clause predicate compiles
% to a try_me_else(Alt1) / retry_me_else(Alt_k+1) / trust_me chain (tags
% 22/23/24) around the per-clause code, so a failing head match backtracks to
% the next clause -- backtracking dispatch AND recursion now compile.
% A call goal p(Args) sets up A1.. with the args (operand_instr) then
% call(CalleeLabel, arity) (tag 18) -- a non-tail call: cp = the next PC, so
% execution resumes after the call when the callee proceeds. A permanent holds
% the result across the call. Each clause is copy_term'd + numbervars'd so its
% variables are clause-local. Facts (no body) emit head + proceed (no env).
% Single-clause predicates emit no chain, so the Stage C output (verified
% byte-identical to the host writer for the mnt program) is unchanged.
% Stage D (lists): cgfull also compiles LIST PATTERNS in heads, list literals
% in call arguments, and atom constants -- so list-walking recursion (the shape
% of every helper in the compiler's own source) compiles from source text.
% - collect_tables walks every clause term (var-safe) gathering the ATOM table
%   ('[]' and any atom constants) and every data functor incl. '[|]' and
%   the arithmetic-operator functors. NB: in SWI, nil is NOT an atom
%   (atom([]) fails), so nil gets dedicated clauses in each argument compiler;
%   in the loaded runtime nil IS the atom "[]", and those clauses match it the
%   same way (get_constant on the relocated atom id).
% - head list pattern [H|T] -> get_list Ai (tag 4) + a unify_* per element:
%   unify_variable (5) first occurrence / unify_value (6) later /
%   unify_constant (7) for integers and atoms.
% - a REPEATED head variable now emits get_value (tag 2), not a second
%   get_variable (which would have overwritten the first binding unchecked).
% - a list literal in a call argument builds top-down like the host: put_list
%   TARGET (12), set_* for the head, set_variable Xtemp (13) for the tail,
%   then put_structure cons/2 into the temp (write mode binds through the
%   fresh cell) -- X temps start at reg 16 and live only within the build.
% - atom constants carry the atom-table INDEX with reloc class 1; the loader
%   relocates them to interned atom ids (matching the reader's atoms).
% Stage D (guards): cgfull also compiles COMPARISON GUARDS and IF-THEN-ELSE.
% A comparison goal (>, <, >=, =<, =:=, =\=, ==, \==) stages A1/A2 and emits
% builtin_call with the comparison's id. ( Cond -> Then ; Else ) compiles to
% the host's ITE shape: try_me_else(ElseLabel) pushes the guard CP; the Cond
% goals run; cut_ite (tag 31, soft cut) pops the guard CP on success; the
% Then goals run and jump(JoinLabel) (tag 32, label operand) skips the else;
% at ElseLabel a trust_me pops the CP (reached by backtracking when Cond
% fails, which also UNDOES Cond's bindings and register writes) and the Else
% goals run to the join. Codegen is now PC- and label-aware: else/join labels
% are allocated mid-clause from the same counter as clause-chain alternatives,
% each recorded as a Label-PC pair; cgfull keysorts the pairs and appends
% pairs_values after the predicate-entry PCs (labels stay positional).
% Init-set rule: Then continues from Cond's set (bindings persist on the then
% path); Else restarts from the pre-ITE set (backtracking undid Cond); after
% the ITE the set is the INTERSECTION of the two branch out-sets (initialized
% on both paths). Variables introduced inside one branch are branch-local.
cgfull(Src, Wamo) :-
    read_term_from_atom(Src, Clauses),
    cgfull_term(Clauses, Wamo).
% Split off the reader call so the middle+back end can run in SWI too
% (read_term_from_atom/2 exists only in the loaded runtime): tests
% compute expected golden bytes by feeding cgfull_term/2 clause TERMS.
% The compilation core is shared with cgfullm_term below; cgfull_term's
% output stays BYTE-IDENTICAL (the single-entry header, first predicate
% named) -- it is the oracle every self-host golden compares against.
cgfull_term(Clauses, Wamo) :-
    cgfull_core(Clauses, Groups, Atoms0, Functors, PCs, AllIs),
    cg_meta_rows(AllIs, Groups, Atoms0, Functors, Atoms, MetaRows),
    Clauses = [First|_], clause_hb(First, H0, _), functor(H0, P0, A0),
    pred_name_codes(P0, A0, EntryName),
    wza_serialize_m(0, EntryName, 0, Atoms, Functors, PCs, AllIs,
        MetaRows, Codes),
    atom_codes(Wamo, Codes).

% The meta-call table (pay-for-what-you-use, mirroring the host writer):
% emitted only when the instruction stream contains a meta-call
% (op1 = -1 on a call), so call-free programs serialize byte-identically
% to before -- every self-host golden is call-free. One row per
% predicate group: <atomIdx> <funIdx> <arity> <labelIdx>. The predicate
% name joins the ATOM table (appended, so existing reloc indices are
% unchanged); funIdx points at the name's functor-table row when the
% object builds such a compound (call(k(V)) collected k into the table
% already), else -1.
cg_meta_rows(AllIs, Groups, Atoms0, Functors, Atoms, Rows) :-
    ( memberchk(enc(18, -1, _, _), AllIs)
    -> cg_meta_names(Groups, 0, Names),
       cg_meta_atoms(Names, Atoms0, Atoms),
       cg_meta_rows_list(Names, Atoms, Functors, Rows)
    ;  Atoms = Atoms0, Rows = []
    ).
cg_meta_names([], _, []).
cg_meta_names([pred(P, A, _)|Gs], I, [mp(P, A, I)|R]) :-
    I1 is I + 1, cg_meta_names(Gs, I1, R).
cg_meta_atoms([], At, At).
cg_meta_atoms([mp(P, _, _)|Ms], At0, At) :-
    add_unique(P, At0, At1), cg_meta_atoms(Ms, At1, At).
cg_meta_rows_list([], _, _, []).
cg_meta_rows_list([mp(P, A, I)|Ms], At, FT, [mr(AI, FI, A, I)|R]) :-
    cg_idx_of(P, At, AI),
    ( cg_idx_of(P, FT, FI0) -> FI = FI0 ; FI = -1 ),
    cg_meta_rows_list(Ms, At, FT, R).
cg_idx_of(X, [Y|Ys], I) :-
    ( X == Y -> I = 0 ; cg_idx_of(X, Ys, I1), I is I1 + 1 ).

% the shared front+middle: grouping, labels, tables, per-group codegen,
% and the closed PC table -- everything up to the header choice.
cgfull_core(Clauses, Groups, Atoms, Functors, PCs, AllIs) :-
    group_clauses(Clauses, Groups),
    group_labels(Groups, 0, PL),
    length(Groups, NP),
    collect_tables(Clauses, s([],[]), s(Atoms, Functors)),
    g_groups(Groups, PL, Atoms, Functors, 0, NP, AllIs, EntryPCs, Pairs),
    keysort(Pairs, SortedPairs), pairs_values(SortedPairs, ExtraPCs),
    append(EntryPCs, ExtraPCs, PCs).

% cgfullm: cgfull with a MULTI-ENTRY name table -- every predicate group
% gets an entry row ("name/arity" -> its group label), so a runtime-
% compiled object exposes its whole predicate family to dyncall_at@name.
% For a single-predicate source the header is byte-identical to cgfull's
% (NE=1, first predicate named, label 0), so content-dedup handles and
% every existing single-grammar compile are unchanged. This is the entry
% the plawk CLI ships as <bin>.evalc.wamo; cgfull stays the self-host
% fixpoint subject (the goldens compare against ITS bytes).
cgfullm(Src, Wamo) :-
    read_term_from_atom(Src, Clauses),
    cgfullm_term(Clauses, Wamo).
cgfullm_term(Clauses, Wamo) :-
    cgfull_core(Clauses, Groups, Atoms0, Functors, PCs, AllIs),
    cg_meta_rows(AllIs, Groups, Atoms0, Functors, Atoms, MetaRows),
    group_entries(Groups, 0, Entries),
    wzam_serialize_m(0, Entries, Atoms, Functors, PCs, AllIs, MetaRows,
        Codes),
    atom_codes(Wamo, Codes).

group_entries([], _, []).
group_entries([pred(P, A, _)|Gs], I, [NC-I|R]) :-
    pred_name_codes(P, A, NC),
    I1 is I + 1,
    group_entries(Gs, I1, R).

% comparison-guard builtin ids
cmp_id(>, 1). cmp_id(<, 2). cmp_id(>=, 3). cmp_id(=<, 4).
cmp_id(=:=, 5). cmp_id(=\=, 6). cmp_id(==, 7). cmp_id(\==, 21).

inter([], _, []).
inter([X|Xs], Ys, Out) :-
    ( memberchk(X, Ys) -> Out = [X|R] ; Out = R ), inter(Xs, Ys, R).

% group consecutive clauses with the same name/arity into pred(P,A,Clauses)
group_clauses([], []).
group_clauses([C|Cs], [pred(P,A,[C|Same])|Gs]) :-
    clause_hb(C, H, _), functor(H, P, A),
    take_same(Cs, P, A, Same, Rest),
    group_clauses(Rest, Gs).
take_same([], _, _, [], []).
take_same([C|Cs], P, A, Same, Rest) :-
    clause_hb(C, H, _), functor(H, P2, A2),
    (   P2 == P, A2 =:= A
    ->  Same = [C|Same1], take_same(Cs, P, A, Same1, Rest)
    ;   Same = [], Rest = [C|Cs]
    ).
group_labels([], _, []).
group_labels([pred(P,A,_)|Gs], I, [key(P,A)-I|R]) :- I1 is I+1, group_labels(Gs, I1, R).

% atom + cons-functor collection: a var-safe walk over every clause term
walk_term(T, S0, S) :- var(T), !, S = S0.
walk_term([], S0, S) :- !, S0 = s(At0, Fn0), add_unique('[]', At0, At1), S = s(At1, Fn0).
walk_term([H|T], S0, S) :- !, S0 = s(At0, Fn0), add_unique('[|]', Fn0, Fn1),
    walk_term(H, s(At0,Fn1), S1), walk_term(T, S1, S).
walk_term(T, S0, S) :- compound(T), !, T =.. [F|Args],
    S0 = s(At0, Fn0),
    ( skip_fn(F) -> Fn1 = Fn0 ; add_unique(F, Fn0, Fn1) ),
    walk_list(Args, s(At0, Fn1), S).
% control / goal functors never appear as data terms in instructions
skip_fn(':-'). skip_fn(','). skip_fn(';'). skip_fn('->'). skip_fn(is).
skip_fn(=). skip_fn(>). skip_fn(<). skip_fn(>=). skip_fn(=<).
skip_fn(=:=). skip_fn(=\=). skip_fn(==). skip_fn(\==). skip_fn(\+).
walk_term(T, S0, S) :- atom(T), !,          % data atom -> atom table
    S0 = s(At0, Fn0), add_unique(T, At0, At1), S = s(At1, Fn0).
walk_term(_, S, S).
walk_list([], S, S).
walk_list([A|As], S0, S) :- walk_term(A, S0, S1), walk_list(As, S1, S).
collect_tables([], S, S).
collect_tables([C|Cs], S0, S) :- walk_term(C, S0, S1), collect_tables(Cs, S1, S).

% per-group codegen; threads the absolute PC AND a label counter (chain
% alternatives and mid-clause ITE labels share it, in codegen order), and
% collects Label-PC pairs (keysorted by cgfull, so any assignment order works).
g_groups([], _, _, _, _, _, [], [], []).
g_groups([pred(_,_,Cls)|Gs], PL, At, FT, PC, L0, AllIs, [PC|EPCs], Prs) :-
    g_pred(Cls, PL, At, FT, PC, L0, Is, PC1, L1, Prs1),
    g_groups(Gs, PL, At, FT, PC1, L1, RestIs, EPCs, Prs2),
    append(Is, RestIs, AllIs), append(Prs1, Prs2, Prs).

g_pred([C], PL, At, FT, PC, L0, Is, PCout, L, Prs) :-   % single clause: no chain
    g_one(C, PL, At, FT, PC, L0, L, Prs, Is), length(Is, N), PCout is PC + N.
g_pred([C1,C2|Rest], PL, At, FT, PC, L0, Is, PCout, L, Prs) :-
    ChainL = L0, L1 is L0 + 1,
    PCc is PC + 1,
    g_one(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is),
    length(C1Is, N1), PC1 is PCc + N1,
    g_alts([C2|Rest], PL, At, FT, PC1, ChainL, L2, AltIs, PCout, L, Prs2),
    append(Prs1, Prs2, Prs),
    append([enc(22,ChainL,0,0)|C1Is], AltIs, Is).       % try_me_else(ChainL)

g_alts([C], PL, At, FT, PC, MyL, L0, Is, PCout, L, [MyL-PC|Prs]) :-  % last alt
    PCc is PC + 1,
    g_one(C, PL, At, FT, PCc, L0, L, Prs, CIs),
    Is = [enc(24,0,0,0)|CIs],                           % trust_me
    length(Is, N), PCout is PC + N.
g_alts([C1,C2|Rest], PL, At, FT, PC, MyL, L0, Is, PCout, L, Prs) :-
    NextL = L0, L1 is L0 + 1,
    PCc is PC + 1,
    g_one(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is),
    length(C1Is, N1), PC1 is PCc + N1,
    g_alts([C2|Rest], PL, At, FT, PC1, NextL, L2, RestIs, PCout, L, Prs2),
    append([MyL-PC|Prs1], Prs2, Prs),
    append([enc(23,NextL,0,0)|C1Is], RestIs, Is).       % retry_me_else(NextL)

g_one(C0, PL, At, FT, StartPC, L0, L, Prs, Is) :-
    copy_term(C0, C), numbervars(C, 0, _),
    f_clause_instrs(C, PL, At, FT, StartPC, L0, L, Prs, Is).

f_clause_instrs(Clause, PL, At, FT, StartPC, L0, L, Prs, Instrs) :-
    clause_hb(Clause, Head, Goals), functor(Head, _, Arity),
    h_args(Head, 1, Arity, At, FT, 16, [], Init1, HeadIs),
    (   Goals == []
    ->  % Facts wrap in allocate/deallocate too: cgfull assigns ALL clause
        % variables to Y registers (the numbervars scheme), and without an
        % environment a fact's get_variable Y-writes CLOBBER THE CALLER'S
        % Y window -- observable whenever a fact is called in non-tail
        % position (the caller's later goals read corrupted permanents).
        % The host AOT compiler routes fact variables through X registers
        % instead; cgfull buys correctness with an environment. Campaign
        % finding: the walkers' deferred-reads nil fact, the first
        % non-tail fact call in the self-host, silently corrupted its
        % caller and sent the compile down a divergent re-satisfaction.
        append([enc(16,0,0,0)|HeadIs], [enc(17,0,0,0), enc(20,0,0,0)], Instrs),
        L = L0, Prs = []
    ;   length(HeadIs, NH), PC0 is StartPC + 1 + NH,     % after allocate + head
        f_goals(Goals, PL, At, FT, PC0, _, L0, L, [], Prs, Init1, _, GoalIs),
        append([enc(16,0,0,0) | HeadIs], GoalIs, B0),
        append(B0, [enc(17,0,0,0), enc(20,0,0,0)], Instrs)
    ).

% head-argument compilation: vars (first/repeated), integers, atoms, nil,
% list patterns [H|T], and GENERAL STRUCTURE PATTERNS f(...). A compound or
% list CHILD inside a pattern is deferred: unify_variable saves it into a
% fresh X temp during the parent's unify sequence, and after the parent
% completes the child is read out of the temp (get_structure / get_list into
% the temp -- the host's canonical nesting shape). X temps thread from reg 16
% through the whole head.
h_args(_, I, Ar, _, _, _, In, In, []) :- I > Ar, !.
h_args(H, I, Ar, At, Fn, Xt0, In0, In, Is) :-
    arg(I, H, A), Ai is I - 1,
    head_arg_instrs(A, Ai, At, Fn, Xt0, Xt1, In0, In1, AIs),
    I1 is I + 1, h_args(H, I1, Ar, At, Fn, Xt1, In1, In, RIs),
    append(AIs, RIs, Is).
head_arg_instrs('$VAR'(N), Ai, _, _, Xt, Xt, In0, In1, [I]) :- integer(N), !,
    Y is 48 + N,
    (   is_init(N, In0)
    ->  I = enc(2, Y, Ai, 0), In1 = In0                 % get_value (repeated var)
    ;   I = enc(1, Y, Ai, 0), In1 = [N|In0]             % get_variable (first)
    ).
head_arg_instrs([], Ai, At, _, Xt, Xt, In, In, [enc(0,Idx,Ai,1)]) :- !,  % get_constant []
    functor_index('[]', At, Idx).
head_arg_instrs([H|T], Ai, At, Fn, Xt0, Xt, In0, In2, [enc(4,Ai,0,0)|Rest]) :- !,  % get_list
    u_seq([H,T], At, Xt0, Xt1, In0, In1, UIs, [], Defs),
    defer_reads(Defs, At, Fn, Xt1, Xt, In1, In2, DIs),
    append(UIs, DIs, Rest).
head_arg_instrs(V, Ai, _, _, Xt, Xt, In, In, [enc(0,V,Op2,0)]) :- integer(V), !, Op2 is (1<<16)\/Ai.
head_arg_instrs(A, Ai, At, _, Xt, Xt, In, In, [enc(0,Idx,Ai,1)]) :- atom(A), !, functor_index(A, At, Idx).
head_arg_instrs(T, Ai, At, Fn, Xt0, Xt, In0, In2, [enc(3,FI,Op2,2)|Rest]) :-
    compound(T),                                        % general structure pattern
    functor(T, F, N), functor_index(F, Fn, FI), Op2 is (N<<16)\/Ai,
    T =.. [_|Args],
    u_seq(Args, At, Xt0, Xt1, In0, In1, UIs, [], Defs),
    defer_reads(Defs, At, Fn, Xt1, Xt, In1, In2, DIs),
    append(UIs, DIs, Rest).
% fail fast: unsupported head-argument kind (see the f_goal catch-all)
head_arg_instrs(T, _, _, _, _, _, _, _, _) :- throw(cg_unsupported_head_arg(T)).
% one unify_* per pattern arg; compound/list children deferred to X temps
u_seq([], _, Xt, Xt, In, In, [], D, D).
u_seq([A|As], At, Xt0, Xt, In0, In, Is, D0, D) :-
    u_arg(A, At, Xt0, Xt1, In0, In1, AIs, D0, D1),
    u_seq(As, At, Xt1, Xt, In1, In, RIs, D1, D), append(AIs, RIs, Is).
u_arg('$VAR'(N), _, Xt, Xt, In0, In1, [I], D, D) :- integer(N), !,
    Y is 48 + N,
    (   is_init(N, In0)
    ->  I = enc(6, Y, 0, 0), In1 = In0                  % unify_value
    ;   I = enc(5, Y, 0, 0), In1 = [N|In0]              % unify_variable
    ).
u_arg([], At, Xt, Xt, In, In, [enc(7,Idx,0,1)], D, D) :- !, functor_index('[]', At, Idx).
u_arg(V, _, Xt, Xt, In, In, [enc(7,V,65536,0)], D, D) :- integer(V), !.
u_arg(A, At, Xt, Xt, In, In, [enc(7,Idx,0,1)], D, D) :- atom(A), !, functor_index(A, At, Idx).
u_arg(C, _, Xt0, Xt1, In, In, [enc(5,Xt0,0,0)], D0, D) :-   % compound/list child
    Xt1 is Xt0 + 1, append(D0, [d(C,Xt0)], D).
defer_reads([], _, _, Xt, Xt, In, In, []).
defer_reads([d(C,Reg)|Ds], At, Fn, Xt0, Xt, In0, In, Is) :-
    (   C = [H|T]
    ->  u_seq([H,T], At, Xt0, Xt1, In0, In1, UIs, [], Defs),
        CIs = [enc(4,Reg,0,0)|UIs]                       % get_list Xtemp
    ;   functor(C, F, N), functor_index(F, Fn, FI), Op2 is (N<<16)\/Reg,
        C =.. [_|Args],
        u_seq(Args, At, Xt0, Xt1, In0, In1, UIs, [], Defs),
        CIs = [enc(3,FI,Op2,2)|UIs]                      % get_structure Xtemp
    ),
    defer_reads(Defs, At, Fn, Xt1, Xt2, In1, In2, DIs),
    defer_reads(Ds, At, Fn, Xt2, Xt, In2, In, RIs),
    append(CIs, DIs, A1s), append(A1s, RIs, Is).

% goal codegen, PC- and label-aware:
% f_goals(Goals, PL, At, FT, PC0,PC, L0,L, Prs0,Prs, In0,In, Is)
f_goals([], _, _, _, PC, PC, L, L, Prs, Prs, In, In, []).
f_goals([G|Gs], PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :-
    f_goal(G, PL, At, FT, PC0, PC1, L0, L1, Prs0, Prs1, In0, In1, GI),
    f_goals(Gs, PL, At, FT, PC1, PC, L1, L, Prs1, Prs, In1, In, RI),
    append(GI, RI, Is).

f_goal(( C -> T ; E ), PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- !,
    ElseL = L0, JoinL is L0 + 1, L1 is L0 + 2,
    conj_list(C, CGs), conj_list(T, TGs), conj_list(E, EGs),
    PC1 is PC0 + 1,                                    % try_me_else(ElseL)
    f_goals(CGs, PL, At, FT, PC1, PC2, L1, L2, Prs0, Prs1, In0, InC, CondIs),
    PC3 is PC2 + 1,                                    % cut_ite
    f_goals(TGs, PL, At, FT, PC3, PC4, L2, L3, Prs1, Prs2, InC, InT, ThenIs),
    ElsePC is PC4 + 1,                                 % after jump(JoinL)
    PC5 is ElsePC + 1,                                 % after trust_me
    f_goals(EGs, PL, At, FT, PC5, JoinPC, L3, L, Prs2, Prs3, In0, InE, ElseIs),
    PC = JoinPC,
    append(Prs3, [ElseL-ElsePC, JoinL-JoinPC], Prs),
    inter(InT, InE, In),
    append([enc(22,ElseL,0,0)|CondIs], [enc(31,0,0,0)|ThenIs], Front),
    append(Front, [enc(32,JoinL,0,0), enc(24,0,0,0)|ElseIs], Is).
% BARE disjunction ( A ; B ) -- the ITE shape WITHOUT the soft cut, so
% the alternatives stay backtrackable: a failure after the first branch
% succeeds re-enters at the second (try_me_else's CP is still live).
% Demand-driven subset growth: common in real grammars fed to
% compile(...). Placed after the ITE clause, so ( C -> T ; E ) keeps
% its committed-choice compilation.
f_goal(( A ; B ), PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- !,
    ElseL = L0, JoinL is L0 + 1, L1 is L0 + 2,
    conj_list(A, AGs), conj_list(B, BGs),
    PC1 is PC0 + 1,                                    % try_me_else(ElseL)
    f_goals(AGs, PL, At, FT, PC1, PC2, L1, L2, Prs0, Prs1, In0, InA, AIs),
    ElsePC is PC2 + 1,                                 % after jump(JoinL)
    PC5 is ElsePC + 1,                                 % after trust_me
    f_goals(BGs, PL, At, FT, PC5, JoinPC, L2, L, Prs1, Prs2, In0, InB, BIs),
    PC = JoinPC,
    append(Prs2, [ElseL-ElsePC, JoinL-JoinPC], Prs),
    inter(InA, InB, In),
    append([enc(22,ElseL,0,0)|AIs],
        [enc(32,JoinL,0,0), enc(24,0,0,0)|BIs], Is).
% Negation as failure: \+ G desugars to ( G -> fail ; true ) and rides
% the ITE machinery unchanged (fail/0 and true/0 are runtime builtins).
f_goal((\+ G), PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- !,
    f_goal(( G -> fail ; true ), PL, At, FT, PC0, PC, L0, L, Prs0, Prs,
        In0, In, Is).
% call/N meta-call (demand-driven subset growth): stage the goal term
% into A1 and the extra args into A2.., then call with the meta
% sentinel (op1 = -1, op2 = total arity). Dispatch resolves the runtime
% goal through the object's meta-call table (emitted by the serializer
% when any meta-call is present -- see cg_meta_rows) with the dynamic
% clause store as fallback, so call/1 also reaches assertz'd facts.
f_goal(Goal, _, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In, Is) :-
    functor(Goal, call, A), A >= 1, !,
    call_args(Goal, 1, A, At, FT, In0, In, SetupIs),
    append(SetupIs, [enc(18, -1, A, 0)], Is),
    length(Is, N), PC is PC0 + N.
% findall(Template, Goal, List) with a VARIABLE template and result
% (the dominant shape; compound templates stay outside the subset and
% throw). Emits the host's aggregate bracket: initialize the template
% and result registers if fresh, begin_aggregate(collect,
% valreg<<16|resreg), the goal inline, end_aggregate(valreg) -- the
% runtime collects one frozen copy per solution and binds the list on
% finalize (begin scans forward for its matching end, so no label is
% needed). Register inits made INSIDE the goal are discarded from the
% out-set: backtracking rewinds them per solution.
f_goal(findall('$VAR'(TN), G, '$VAR'(LN)), PL, At, FT, PC0, PC, L0, L,
        Prs0, Prs, In0, In, Is) :-
    integer(TN), integer(LN), !,
    TY is 48 + TN,
    LY is 48 + LN,
    ( memberchk(TN, In0) -> TInit = [], In1 = In0
    ; TInit = [enc(9, TY, TY, 0)], In1 = [TN | In0] ),
    ( memberchk(LN, In1) -> LInit = [], In2 = In1
    ; LInit = [enc(9, LY, LY, 0)], In2 = [LN | In1] ),
    append(TInit, LInit, Inits),
    length(Inits, NI),
    GoalPC is PC0 + NI + 1,
    conj_list(G, Gs),
    f_goals(Gs, PL, At, FT, GoalPC, PCg, L0, L, Prs0, Prs, In2, _InG, GIs),
    PC is PCg + 1,
    Op2 is (TY << 16) \/ LY,
    append(Inits, [enc(28, 4, Op2, 0) | GIs], Front),
    append(Front, [enc(29, TY, 0, 0)], Is),
    In = In2.
f_goal((L is E), _, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In2, Is) :- !,
    % NESTED arithmetic: the expression is just a term -- stage it with
    % c_operand (build_struct + X-temp deferral handles arbitrary nesting,
    % e.g. (X + Y) * (X - 1) + 100). The operators land in the functor
    % table automatically: walk_term skips is/2 itself as a control
    % functor but walks its argument tree, collecting + / * / ... as data
    % functors. LHS -> A1, expression term -> A2, builtin is/2 (id 0).
    c_operand(L, 0, At, FT, In0, In1, IL),
    c_operand(E, 1, At, FT, In1, In2, IE),
    append(IL, IE, LE), append(LE, [enc(21, 0, 2, 0)], Is),
    length(Is, N), PC is PC0 + N.
f_goal((L = R), _, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In2, Is) :- !,
    % full-term unification: either side may be a variable, constant, list
    % or structure literal (c_operand builds it) -- then builtin =/2
    c_operand(L, 0, At, FT, In0, In1, IL),
    c_operand(R, 1, At, FT, In1, In2, IR),
    append(IL, IR, LR), append(LR, [enc(21,24,2,0)], Is),
    length(Is, N), PC is PC0 + N.
f_goal(Cmp, _, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In, Is) :-
    functor(Cmp, Op, 2), cmp_id(Op, Id), !,            % comparison guard
    arg(1, Cmp, A1), arg(2, Cmp, A2),
    c_operand(A1, 0, At, FT, In0, In1, IL),
    c_operand(A2, 1, At, FT, In1, In, IR),
    append(IL, IR, LR), append(LR, [enc(21,Id,2,0)], Is),
    length(Is, N), PC is PC0 + N.
f_goal(Goal, _, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In, Is) :-
    functor(Goal, P, A), bi_id(P, A, Id), !,           % builtin goal
    call_args(Goal, 1, A, At, FT, In0, In, SetupIs),
    append(SetupIs, [enc(21, Id, A, 0)], Is),          % builtin_call(Id, A)
    length(Is, N), PC is PC0 + N.
f_goal(Goal, PL, At, FT, PC0, PC, Lb, Lb, Prs, Prs, In0, In, Is) :-  % predicate call
    functor(Goal, P, A), lookup_label(P, A, PL, Label),
    call_args(Goal, 1, A, At, FT, In0, In, SetupIs),
    append(SetupIs, [enc(18, Label, A, 0)], Is),       % call(Label, arity)
    length(Is, N), PC is PC0 + N.
% FAIL FAST on anything the subset does not cover. Without this, an
% unsupported goal makes codegen FAIL, and in the loaded compiler that
% failure explodes into catastrophic backtracking through the compile's
% stale choice points (unindexed, cut-free clause chains) -- a silent
% multi-minute hang instead of an error (the gen-3 round lost real time
% to exactly this, via a nested `is` before the lift above). throw/1 is
% loadable (call sentinel), so the loaded compiler aborts immediately.
f_goal(Goal, _, _, _, _, _, _, _, _, _, _, _, _) :-
    functor(Goal, P, A), throw(cg_unsupported_goal(P, A)).

% builtin goals the grammar can emit directly: name/arity -> builtin id (the
% host runtime's builtin_op_to_id table). Staged like a call (c_operand per
% arg into A1..An) then builtin_call(Id, Arity). NB: the grammar emits
% builtin_call for arg/3 even with a constant index -- the host tier-2
% compiler's specialised arg opcode (the known subset gap) is simply never
% generated here, so loaded objects get the loadable builtin form.
bi_id(functor, 3, 26).
bi_id(arg, 3, 27).
bi_id(=.., 2, 28).
bi_id(copy_term, 2, 29).
bi_id(atom_codes, 2, 36).
bi_id(atom_chars, 2, 38).
bi_id(char_code, 2, 37).
bi_id(number_codes, 2, 43).
bi_id(atom_number, 2, 42).
bi_id(atom_length, 2, 35).
bi_id(atom_concat, 3, 40).
bi_id(length, 2, 31).
bi_id(append, 3, 55).
bi_id(reverse, 2, 54).
bi_id(nth0, 3, 51).
bi_id(nth1, 3, 52).
bi_id(last, 2, 53).
bi_id(memberchk, 2, 56).
bi_id(between, 3, 39).
bi_id(msort, 2, 30).
bi_id(sort, 2, 81).
bi_id(keysort, 2, 80).
bi_id(pairs_keys, 2, 70).
bi_id(pairs_values, 2, 71).
bi_id(atom, 1, 13).
bi_id(integer, 1, 14).
bi_id(number, 1, 16).
bi_id(compound, 1, 17).
bi_id(var, 1, 18).
bi_id(nonvar, 1, 19).
bi_id(is_list, 1, 20).
bi_id(ground, 1, 132).
bi_id(succ, 2, 22).
bi_id(\=, 2, 25).
bi_id(sum_list, 2, 59).
bi_id(numbervars, 3, 124).
% demand-driven subset growth: control atoms and the text/string family
% (all existing runtime builtins -- these rows only let source-level
% grammars reach them). Plain `!` is the runtime's clause-level cut
% builtin; true/fail anchor the \+ desugar above. assertz-family ids
% exist in the runtime but stay unlisted until call/N emission lands (a
% grammar cannot read the store back without it).
bi_id(!, 0, 10).
bi_id(true, 0, 8).
bi_id(fail, 0, 9).
% assert family -- readable now that call/N emission lands (call/1
% reaches assertz'd facts through the dynamic-store fallback of the
% meta-call dispatch). Nondet retract/1 (call sentinel -3) stays out.
bi_id(assertz, 1, 175).
bi_id(asserta, 1, 176).
bi_id(retractall, 1, 177).
bi_id(sub_atom, 5, 41).
bi_id(upcase_atom, 2, 45).
bi_id(downcase_atom, 2, 46).
bi_id(string_concat, 3, 47).
bi_id(atom_string, 2, 49).
bi_id(split_string, 4, 85).
bi_id(term_to_atom, 2, 173).
bi_id(char_type, 2, 75).
bi_id(code_type, 2, 172).
bi_id(term_to_atom, 2, 173).
bi_id(read_term_from_atom, 2, 174).

call_args(_, I, Ar, _, _, In, In, []) :- I > Ar, !.
call_args(Goal, I, Ar, At, FT, In0, In, Is) :-
    arg(I, Goal, Arg), Ai is I - 1,
    c_operand(Arg, Ai, At, FT, In0, In1, AI),
    I1 is I + 1, call_args(Goal, I1, Ar, At, FT, In1, In, R), append(AI, R, Is).

% call-argument staging: vars/ints via operand_instr; nil/atoms as atom
% constants; list literals built on the heap
c_operand('$VAR'(N), Ai, _, _, In0, In1, Is) :- integer(N), !, operand_instr('$VAR'(N), Ai, In0, In1, Is).
c_operand(V, Ai, _, _, In, In, Is) :- integer(V), !, operand_instr(V, Ai, In, In, Is).
c_operand([], Ai, At, _, In, In, [enc(8,Idx,Ai,1)]) :- !, functor_index('[]', At, Idx).
c_operand([H|T], Ai, At, FT, In0, In1, Is) :- !,
    build_list([H|T], Ai, first, At, FT, 16, _, In0, In1, Is).
c_operand(A, Ai, At, _, In, In, [enc(8,Idx,Ai,1)]) :- atom(A), !, functor_index(A, At, Idx).
c_operand(T, Ai, At, FT, In0, In1, Is) :- compound(T),   % structure literal
    build_struct(T, Ai, At, FT, 16, _, In0, In1, Is).
% fail fast: unsupported operand kind (e.g. a float literal), or a
% build_struct failure (e.g. a functor missing from the table)
c_operand(T, _, _, _, _, _, _) :- throw(cg_unsupported_operand(T)).

% top-down term builds. Lists: the first cell is put_list TARGET; each nested
% cons is put_structure cons/2 into a fresh X temp created by set_variable
% (write mode binds through the fresh cell). Structures: put_structure F/N
% into the target, one set_* per arg, with compound/list CHILDREN deferred to
% X temps and built after the parent (the mirror of the head-side deferral).
build_list([E|Rest], TR, Mode, At, FT, Xt0, Xt, In0, In2, Is) :-
    (   Mode = first
    ->  Head = [enc(12,TR,0,0)]                          % put_list
    ;   functor_index('[|]', FT, CI), Op2 is (2<<16)\/TR,
        Head = [enc(11,CI,Op2,2)]                        % put_structure cons/2
    ),
    s_arg(E, At, Xt0, Xta, In0, In1, SE, [], DefsE),
    (   Rest == []
    ->  functor_index('[]', At, NI), Tail = [enc(15,NI,0,1)],
        In1b = In1, Xtb = Xta, RestIs = []
    ;   Rest = '$VAR'(_)
    ->  s_arg(Rest, At, Xta, Xtb, In1, In1b, Tail, [], []),  % var tail: [E|Var]
        RestIs = []
    ;   Tail = [enc(13,Xta,0,0)],                        % set_variable Xtemp
        Xtc is Xta + 1,
        build_list(Rest, Xta, nested, At, FT, Xtc, Xtb, In1, In1b, RestIs)
    ),
    defer_builds(DefsE, At, FT, Xtb, Xt, In1b, In2, DIs),
    append(Head, SE, HS), append(HS, Tail, HST),
    append(HST, RestIs, HSTR), append(HSTR, DIs, Is).
build_struct(T, TR, At, FT, Xt0, Xt, In0, In2, [enc(11,FI,Op2,2)|Rest]) :-
    functor(T, F, N), functor_index(F, FT, FI), Op2 is (N<<16)\/TR,
    T =.. [_|Args],
    s_seq(Args, At, Xt0, Xt1, In0, In1, SIs, [], Defs),
    defer_builds(Defs, At, FT, Xt1, Xt, In1, In2, DIs),
    append(SIs, DIs, Rest).
s_seq([], _, Xt, Xt, In, In, [], D, D).
s_seq([A|As], At, Xt0, Xt, In0, In, Is, D0, D) :-
    s_arg(A, At, Xt0, Xt1, In0, In1, AIs, D0, D1),
    s_seq(As, At, Xt1, Xt, In1, In, RIs, D1, D), append(AIs, RIs, Is).
s_arg('$VAR'(N), _, Xt, Xt, In0, In1, [I], D, D) :- integer(N), !,
    Y is 48 + N,
    (   is_init(N, In0)
    ->  I = enc(14, Y, 0, 0), In1 = In0                  % set_value
    ;   I = enc(13, Y, 0, 0), In1 = [N|In0]              % set_variable
    ).
s_arg([], At, Xt, Xt, In, In, [enc(15,Idx,0,1)], D, D) :- !, functor_index('[]', At, Idx).
s_arg(V, _, Xt, Xt, In, In, [enc(15,V,1,0)], D, D) :- integer(V), !.
s_arg(A, At, Xt, Xt, In, In, [enc(15,Idx,0,1)], D, D) :- atom(A), !, functor_index(A, At, Idx).
s_arg(C, _, Xt0, Xt1, In, In, [enc(13,Xt0,0,0)], D0, D) :-  % compound/list child
    Xt1 is Xt0 + 1, append(D0, [d(C,Xt0)], D).
defer_builds([], _, _, Xt, Xt, In, In, []).
defer_builds([d(C,Reg)|Ds], At, FT, Xt0, Xt, In0, In, Is) :-
    (   C = [_|_]
    ->  build_list(C, Reg, nested, At, FT, Xt0, Xt1, In0, In1, CIs)
    ;   build_struct(C, Reg, At, FT, Xt0, Xt1, In0, In1, CIs)
    ),
    defer_builds(Ds, At, FT, Xt1, Xt, In1, In, RIs),
    append(CIs, RIs, Is).

% serializer with an atom table: NA + NA length-prefixed atom name strings
% between the label index and the functor table
wza_serialize(EI, NC, LI, Atoms, Fs, PCs, Is, Out) :-
    wza_serialize_m(EI, NC, LI, Atoms, Fs, PCs, Is, [], Out).

% +MetaRows variant: the trailing section carries the meta-call rows
% (count, then <atomIdx> <funIdx> <arity> <labelIdx> per row). An empty
% list emits the count 0 -- exactly the old trailing byte, so call-free
% programs are byte-identical.
wza_serialize_m(EI, NC, LI, Atoms, Fs, PCs, Is, MetaRows, Out) :-
    wz_a('WAMO', Out, A1), wz_i(2, A1, A2), wz_i(EI, A2, A3), wz_i(1, A3, A4),
    wz_n(NC, A4, A5), wz_i(LI, A5, A6),
    length(Atoms, NA), wz_i(NA, A6, A7), wz_funcs(Atoms, A7, A8),
    length(Fs, NF), wz_i(NF, A8, A9), wz_funcs(Fs, A9, A10),
    wz_pcs_sec(PCs, A10, A11), wz_instr_sec(Is, A11, A12),
    wz_meta_sec(MetaRows, A12, []).

% multi-entry variant: NE from the Entries list (NameCodes-LabelIdx
% pairs), one name/label row per entry. With a single entry the emitted
% bytes equal wza_serialize's -- the header shape is the same, only the
% count and rows generalize.
wzam_serialize(EI, Entries, Atoms, Fs, PCs, Is, Out) :-
    wzam_serialize_m(EI, Entries, Atoms, Fs, PCs, Is, [], Out).
wzam_serialize_m(EI, Entries, Atoms, Fs, PCs, Is, MetaRows, Out) :-
    wz_a('WAMO', Out, A1), wz_i(2, A1, A2), wz_i(EI, A2, A3),
    length(Entries, NE), wz_i(NE, A3, A4), wz_entry_rows(Entries, A4, A5),
    length(Atoms, NA), wz_i(NA, A5, A6), wz_funcs(Atoms, A6, A7),
    length(Fs, NF), wz_i(NF, A7, A8), wz_funcs(Fs, A8, A9),
    wz_pcs_sec(PCs, A9, A10), wz_instr_sec(Is, A10, A11),
    wz_meta_sec(MetaRows, A11, []).
wz_entry_rows([], A, A).
wz_entry_rows([NC-LI|Es], A0, A3) :-
    wz_n(NC, A0, A1), wz_i(LI, A1, A2), wz_entry_rows(Es, A2, A3).

wz_meta_sec(Rows, A0, A2) :-
    length(Rows, NM), wz_i(NM, A0, A1), wz_meta_rows(Rows, A1, A2).
wz_meta_rows([], A, A).
wz_meta_rows([mr(AI, FI, Ar, LI)|Rs], A0, A5) :-
    wz_i(AI, A0, A1), wz_i(FI, A1, A2), wz_i(Ar, A2, A3), wz_i(LI, A3, A4),
    wz_meta_rows(Rs, A4, A5).
